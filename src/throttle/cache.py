"""Two-tier similarity cache for Throttle.

Tier 1 (fast-path): lexical Jaccard similarity
Tier 2 (slow-path): ONNX sentence embeddings + cosine similarity

The public interface remains intentionally small:
- get(prompt) -> cached payload | None
- put(prompt, response_data)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore


@dataclass
class CacheMetrics:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    lexical_hits: int = 0
    embedding_hits: int = 0
    embedding_checks: int = 0
    embedding_ms_total: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        avg_embed_ms = (
            self.embedding_ms_total / self.embedding_checks
            if self.embedding_checks
            else 0.0
        )
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "lexical_hits": self.lexical_hits,
            "embedding_hits": self.embedding_hits,
            "embedding_checks": self.embedding_checks,
            "avg_embedding_ms": round(avg_embed_ms, 3),
            "hit_rate": (
                self.hits / (self.hits + self.misses)
                if (self.hits + self.misses)
                else 0.0
            ),
        }


class _OnnxEmbedder:
    """Lazy ONNX embedder. Loaded only when semantic slow-path is enabled."""

    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if np is None:
            raise ImportError("numpy is required for ONNX embeddings")

        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # export=True converts once if needed; subsequent loads reuse local cache
        self.model = ORTModelForFeatureExtraction.from_pretrained(
            model_id, export=True
        )

    def embed(self, text: str) -> "np.ndarray":
        import torch

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        outputs = self.model(**inputs)
        token_embeddings = outputs[0]
        attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * attention_mask, dim=1)
        counts = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        emb = (summed / counts)[0].detach().cpu().numpy().astype("float32")
        # L2 normalize for stable cosine similarity via dot product
        norm = float(np.linalg.norm(emb))
        if norm > 0:
            emb = emb / norm
        return emb


@dataclass
class _CacheEntry:
    prompt: str
    response_data: Any
    timestamp: float
    embedding: Any | None = None


class SimilarityCache:
    """Thread-safe two-tier similarity cache.

    Compatibility notes:
    - Keeps get/put interface used by benchmark + proxy
    - ONNX is optional: if unavailable/disabled, falls back to Jaccard only
    """

    def __init__(
        self,
        ttl_seconds: float = 3600.0,
        max_size: int = 1000,
        similarity_threshold: float = 0.85,
        *,
        enable_embeddings: bool = False,
        embedding_threshold: float = 0.80,
        embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_max_entries_scanned: int = 256,
    ):
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        if not (0.0 <= similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        if not (0.0 <= embedding_threshold <= 1.0):
            raise ValueError("embedding_threshold must be between 0.0 and 1.0")

        self.ttl_seconds = float(ttl_seconds)
        self.max_size = int(max_size)
        self.similarity_threshold = float(similarity_threshold)
        self.enable_embeddings = bool(enable_embeddings)
        self.embedding_threshold = float(embedding_threshold)
        self.embedding_model_id = embedding_model_id
        self.embedding_max_entries_scanned = int(embedding_max_entries_scanned)

        self.metrics = CacheMetrics()
        self._store: Dict[str, _CacheEntry] = {}
        self._lock = Lock()
        self._embedder: _OnnxEmbedder | None = None

        if self.enable_embeddings:
            # Fail fast on startup if user asked for embeddings but deps are missing
            self._embedder = _OnnxEmbedder(model_id=self.embedding_model_id)

    @staticmethod
    def _jaccard_similarity(prompt_a: str, prompt_b: str) -> float:
        set_a = set(prompt_a.lower().split())
        set_b = set(prompt_b.lower().split())
        if not set_a and not set_b:
            return 1.0
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return intersection / union if union else 0.0

    @staticmethod
    def _cosine_normalized(a: "np.ndarray", b: "np.ndarray") -> float:
        # embeddings are L2-normalized => cosine == dot
        return float(np.dot(a, b))

    def _evict_expired_unsafe(self, now: float) -> None:
        expired = [
            key
            for key, entry in self._store.items()
            if now - entry.timestamp > self.ttl_seconds
        ]
        for key in expired:
            del self._store[key]
            self.metrics.evictions += 1

    def _evict_fifo_if_needed_unsafe(self, incoming_key: str) -> None:
        if len(self._store) < self.max_size:
            return
        if incoming_key in self._store:
            return
        oldest_key = next(iter(self._store))
        del self._store[oldest_key]
        self.metrics.evictions += 1

    def _embed_prompt(self, prompt: str) -> Tuple[Any, float]:
        assert self._embedder is not None
        started = time.perf_counter()
        emb = self._embedder.embed(prompt)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.metrics.embedding_checks += 1
        self.metrics.embedding_ms_total += elapsed_ms
        return emb, elapsed_ms

    def get(self, prompt: str) -> Optional[Any]:
        now = time.time()
        with self._lock:
            self._evict_expired_unsafe(now)

            # 1) Exact match O(1)
            exact = self._store.get(prompt)
            if exact is not None:
                self.metrics.hits += 1
                self.metrics.lexical_hits += 1
                return exact.response_data

            # 2) Lexical fast-path
            for entry in self._store.values():
                if self._jaccard_similarity(prompt, entry.prompt) >= self.similarity_threshold:
                    self.metrics.hits += 1
                    self.metrics.lexical_hits += 1
                    return entry.response_data

            # 3) Semantic slow-path (optional ONNX)
            if self.enable_embeddings and self._embedder is not None and self._store:
                query_emb, _ = self._embed_prompt(prompt)
                best_score = -1.0
                best_payload = None

                # Scan newest-first; dict preserves insertion order
                entries: Sequence[_CacheEntry] = list(self._store.values())[
                    -self.embedding_max_entries_scanned :
                ]
                for entry in reversed(entries):
                    if entry.embedding is None:
                        entry.embedding, _ = self._embed_prompt(entry.prompt)
                    score = self._cosine_normalized(query_emb, entry.embedding)
                    if score > best_score:
                        best_score = score
                        best_payload = entry.response_data

                if best_payload is not None and best_score >= self.embedding_threshold:
                    self.metrics.hits += 1
                    self.metrics.embedding_hits += 1
                    return best_payload

            self.metrics.misses += 1
            return None

    def put(self, prompt: str, response_data: Any) -> None:
        now = time.time()
        with self._lock:
            self._evict_expired_unsafe(now)
            self._evict_fifo_if_needed_unsafe(prompt)

            embedding = None
            # Eager embed on write keeps read-path faster for future semantic checks
            if self.enable_embeddings and self._embedder is not None:
                embedding, _ = self._embed_prompt(prompt)

            self._store[prompt] = _CacheEntry(
                prompt=prompt,
                response_data=response_data,
                timestamp=now,
                embedding=embedding,
            )

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                **self.metrics.as_dict(),
                "size": len(self._store),
                "embeddings_enabled": self.enable_embeddings,
                "similarity_threshold": self.similarity_threshold,
                "embedding_threshold": self.embedding_threshold,
            }
