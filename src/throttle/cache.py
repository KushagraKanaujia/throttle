"""Two-tier similarity cache for Throttle with Promise-based in-flight dedup.

Tier 1 (fast-path): lexical Jaccard similarity
Tier 2 (slow-path): ONNX sentence embeddings + cosine similarity

Public interface:
- get(prompt) -> cached payload | None          (backward compat for benchmark)
- put(prompt, response_data)                     (backward compat for benchmark)
- get_or_create_promise(prompt, loop) -> (status, result_or_future, entry)
- resolve_promise(entry, response_data)          (backend success)
- reject_promise(entry, exception)               (backend failure)
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore


class EntryStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"


@dataclass
class CacheMetrics:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    lexical_hits: int = 0
    embedding_hits: int = 0
    embedding_checks: int = 0
    embedding_ms_total: float = 0.0
    promise_waits: int = 0
    promise_rejections: int = 0

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
            "promise_waits": self.promise_waits,
            "promise_rejections": self.promise_rejections,
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
        attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()
        summed = torch.sum(token_embeddings * attention_mask, dim=1)
        counts = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        emb = (summed / counts)[0].detach().cpu().numpy().astype("float32")
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
    status: EntryStatus = EntryStatus.READY
    future: asyncio.Future[Any] | None = None


class SimilarityCache:
    """Thread-safe two-tier similarity cache with Promise-based dedup.

    Compatibility notes:
    - Keeps get/put interface used by benchmark
    - Adds get_or_create_promise / resolve_promise / reject_promise for proxy
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
        pending_timeout_seconds: float = 30.0,
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
        self.pending_timeout_seconds = float(pending_timeout_seconds)

        self.metrics = CacheMetrics()
        self._store: Dict[str, _CacheEntry] = {}
        self._lock = Lock()
        self._embedder: _OnnxEmbedder | None = None

        if self.enable_embeddings:
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
        return float(np.dot(a, b))

    def _evict_expired_unsafe(self, now: float) -> None:
        expired = [
            key
            for key, entry in self._store.items()
            if (
                (entry.status == EntryStatus.READY and now - entry.timestamp > self.ttl_seconds)
                or (entry.status == EntryStatus.PENDING and now - entry.timestamp > self.pending_timeout_seconds)
            )
        ]
        for key in expired:
            entry = self._store[key]
            if entry.status == EntryStatus.PENDING and entry.future and not entry.future.done():
                entry.future.set_exception(
                    TimeoutError("pending cache entry timed out")
                )
                self.metrics.promise_rejections += 1
            del self._store[key]
            self.metrics.evictions += 1

    def _evict_fifo_if_needed_unsafe(self, incoming_key: str) -> None:
        if len(self._store) < self.max_size:
            return
        if incoming_key in self._store:
            return
        ready_keys = [
            k for k, e in self._store.items() if e.status == EntryStatus.READY
        ]
        if not ready_keys:
            return
        oldest_key = ready_keys[0]
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

    # ------------------------------------------------------------------
    # Backward-compatible interface (used by benchmark harness)
    # ------------------------------------------------------------------

    def get(self, prompt: str) -> Optional[Any]:
        """Synchronous cache lookup. Returns payload or None."""
        now = time.time()
        with self._lock:
            self._evict_expired_unsafe(now)

            exact = self._store.get(prompt)
            if exact is not None and exact.status == EntryStatus.READY:
                self.metrics.hits += 1
                self.metrics.lexical_hits += 1
                return exact.response_data

            for entry in self._store.values():
                if entry.status == EntryStatus.READY:
                    if self._jaccard_similarity(prompt, entry.prompt) >= self.similarity_threshold:
                        self.metrics.hits += 1
                        self.metrics.lexical_hits += 1
                        return entry.response_data

            if self.enable_embeddings and self._embedder is not None and self._store:
                query_emb, _ = self._embed_prompt(prompt)
                best_score = -1.0
                best_payload = None

                entries: Sequence[_CacheEntry] = list(self._store.values())[
                    -self.embedding_max_entries_scanned :
                ]
                for entry in reversed(entries):
                    if entry.status != EntryStatus.READY:
                        continue
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
        """Synchronous cache write. Used by benchmark harness."""
        now = time.time()
        with self._lock:
            self._evict_expired_unsafe(now)
            self._evict_fifo_if_needed_unsafe(prompt)

            embedding = None
            if self.enable_embeddings and self._embedder is not None:
                embedding, _ = self._embed_prompt(prompt)

            self._store[prompt] = _CacheEntry(
                prompt=prompt,
                response_data=response_data,
                timestamp=now,
                embedding=embedding,
                status=EntryStatus.READY,
            )

    # ------------------------------------------------------------------
    # Promise-based interface (used by proxy for in-flight dedup)
    # ------------------------------------------------------------------

    def get_or_create_promise(
        self, prompt: str, loop: asyncio.AbstractEventLoop
    ) -> Tuple[EntryStatus, Any, _CacheEntry]:
        """Look up cache with semantic matching; create PENDING entry on miss.

        Returns:
            (READY, response_data, entry)   — immediate cache hit
            (PENDING, future, entry)        — waiter or new owner
        """
        now = time.time()

        # Phase 1: Fast-path Jaccard under lock
        with self._lock:
            self._evict_expired_unsafe(now)

            exact = self._store.get(prompt)
            if exact is not None:
                if exact.status == EntryStatus.READY:
                    self.metrics.hits += 1
                    self.metrics.lexical_hits += 1
                    return EntryStatus.READY, exact.response_data, exact
                else:
                    self.metrics.hits += 1
                    self.metrics.promise_waits += 1
                    return EntryStatus.PENDING, exact.future, exact

            for entry in self._store.values():
                if self._jaccard_similarity(prompt, entry.prompt) >= self.similarity_threshold:
                    self.metrics.hits += 1
                    self.metrics.lexical_hits += 1
                    if entry.status == EntryStatus.READY:
                        return EntryStatus.READY, entry.response_data, entry
                    else:
                        self.metrics.promise_waits += 1
                        return EntryStatus.PENDING, entry.future, entry

            snapshot = [
                e for e in self._store.values()
                if e.status == EntryStatus.READY
            ][-self.embedding_max_entries_scanned:]

        # Phase 2: Semantic ONNX match OUTSIDE lock
        matched_entry: _CacheEntry | None = None
        query_emb = None
        if self.enable_embeddings and self._embedder is not None and snapshot:
            query_emb, _ = self._embed_prompt(prompt)
            best_score = -1.0
            for entry in reversed(snapshot):
                if entry.embedding is None:
                    with self._lock:
                        if entry.embedding is None and entry in self._store.values():
                            entry.embedding, _ = self._embed_prompt(entry.prompt)
                if entry.embedding is not None:
                    score = self._cosine_normalized(query_emb, entry.embedding)
                    if score > best_score:
                        best_score = score
                        matched_entry = entry

            if matched_entry is not None and best_score >= self.embedding_threshold:
                with self._lock:
                    if matched_entry in self._store.values():
                        self.metrics.hits += 1
                        self.metrics.embedding_hits += 1
                        if matched_entry.status == EntryStatus.READY:
                            return EntryStatus.READY, matched_entry.response_data, matched_entry
                        else:
                            self.metrics.promise_waits += 1
                            return EntryStatus.PENDING, matched_entry.future, matched_entry

        # Phase 3: True miss — create PENDING promise
        with self._lock:
            self._evict_expired_unsafe(time.time())
            self._evict_fifo_if_needed_unsafe(prompt)

            self.metrics.misses += 1
            future: asyncio.Future[Any] = loop.create_future()
            new_entry = _CacheEntry(
                prompt=prompt,
                response_data=None,
                timestamp=time.time(),
                embedding=query_emb,
                status=EntryStatus.PENDING,
                future=future,
            )
            self._store[prompt] = new_entry
            return EntryStatus.PENDING, future, new_entry

    def resolve_promise(self, entry: _CacheEntry, response_data: Any) -> None:
        """Mark a PENDING entry as READY and wake all waiters."""
        with self._lock:
            if entry in self._store.values():
                entry.status = EntryStatus.READY
                entry.response_data = response_data
                entry.timestamp = time.time()
                fut = entry.future
                entry.future = None
                if fut and not fut.done():
                    fut.set_result(response_data)

    def reject_promise(self, entry: _CacheEntry, exc: Exception) -> None:
        """Reject a PENDING entry, evict it, and propagate error to waiters."""
        with self._lock:
            self.metrics.promise_rejections += 1
            key_to_remove = None
            for k, v in self._store.items():
                if v is entry:
                    key_to_remove = k
                    break
            if key_to_remove is not None:
                del self._store[key_to_remove]

            fut = entry.future
            entry.future = None
            if fut and not fut.done():
                fut.set_exception(exc)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                **self.metrics.as_dict(),
                "size": len(self._store),
                "pending_count": sum(
                    1 for e in self._store.values() if e.status == EntryStatus.PENDING
                ),
                "embeddings_enabled": self.enable_embeddings,
                "similarity_threshold": self.similarity_threshold,
                "embedding_threshold": self.embedding_threshold,
            }
