"""Two-tier similarity cache for LLM inference.

Tier 1 (fast): Jaccard lexical similarity
Tier 2 (slow): ONNX sentence embeddings + cosine similarity (optional)

The embedding tier is off by default and requires the 'embeddings' extra.
"""

import logging
import time
from dataclasses import dataclass
from threading import Lock
from typing import Dict, Optional, Tuple, Any

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

from . import embeddings

logger = logging.getLogger(__name__)

@dataclass
class CacheMetrics:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    exact_hits: int = 0
    lexical_hits: int = 0
    embedding_hits: int = 0
    embedding_scans_attempted: int = 0
    embedding_comparisons_performed: int = 0

@dataclass
class _CacheEntry:
    """Internal cache entry with optional embedding."""
    prompt: str
    response_data: Any
    timestamp: float
    embedding: Optional["np.ndarray"] = None

class SimilarityCache:
    """Two-tier similarity cache for LLM inference.

    Tier 1: Jaccard lexical similarity (always enabled)
    Tier 2: ONNX semantic embeddings (optional, requires 'embeddings' extra)

    Thread-safe for concurrent use.
    """
    def __init__(
        self,
        ttl_seconds: float = 3600.0,
        max_size: int = 1000,
        similarity_threshold: float = 0.85,
        # NOTE: Do NOT raise this to 0.95 globally. Measured 2026-08-24:
        # at 0.95, short-query paraphrase recall drops from 66.7% to 28.6%
        # (e.g. "what is RAG" vs "explain RAG" scores 0.9189 — missed at 0.95).
        # False positives are handled by _has_negation_or_version_conflict()
        # which explicitly blocks antonyms, negations, and version conflicts.
        *,
        enable_embeddings: bool = False,
        embedding_threshold: float = 0.95,
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

        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.enable_embeddings = enable_embeddings
        self.embedding_threshold = embedding_threshold
        self.embedding_model_id = embedding_model_id
        self.embedding_max_entries_scanned = embedding_max_entries_scanned
        self.metrics = CacheMetrics()

        # Store maps: prompt -> (_CacheEntry)
        # Response data is scope dict when used via proxy, raw response otherwise
        self._store: Dict[str, _CacheEntry] = {}
        self._lock = Lock()
        self._embedding_fallback_logged = False

        # Stacked embeddings: (N, 384) array for vectorized scan
        # Keys list maintains same order as rows in stacked array
        self._embedding_matrix: Optional["np.ndarray"] = None
        self._embedding_keys: list[str] = []

        if self.enable_embeddings:
            if not embeddings.EMBEDDINGS_AVAILABLE:
                logger.warning(
                    "Embeddings requested but dependencies not installed. "
                    "Falling back to Jaccard-only matching. "
                    "Install with: pip install throttle-pro[embeddings]"
                )
                self._embedding_fallback_logged = True
                self.enable_embeddings = False

    # Contraction expansion for better lexical matching
    _CONTRACTIONS = {
        "what's": "what is", "it's": "it is", "that's": "that is",
        "don't": "do not", "doesn't": "does not", "can't": "cannot",
        "won't": "will not", "isn't": "is not", "aren't": "are not",
        "i'm": "i am", "you're": "you are", "we're": "we are",
    }

    def _jaccard_similarity(self, prompt_a: str, prompt_b: str) -> float:
        def normalize(text: str) -> set[str]:
            lowered = text.lower()
            for contraction, expansion in self._CONTRACTIONS.items():
                lowered = lowered.replace(contraction, expansion)
            return set(lowered.split())

        set_a = normalize(prompt_a)
        set_b = normalize(prompt_b)

        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _cosine_normalized(a: "np.ndarray", b: "np.ndarray") -> float:
        """Cosine similarity for L2-normalized vectors (== dot product)."""
        return float(np.dot(a, b))

    @staticmethod
    def _has_negation_or_version_conflict(prompt_a: str, prompt_b: str) -> bool:
        """Check if two prompts differ on negation, antonyms, or version/number tokens.

        Returns True if the prompts should NOT match (conflict detected).

        This guard prevents catastrophic semantic cache failures where:
        - "How to enable X?" matches "How to disable X?"
        - "Is X safe?" matches "Is X dangerous?"
        - "Install pandas 1.5" matches "Install pandas 2.0"

        Measured 2026-08-24: cosine similarity encodes topic, not polarity.
        At threshold 0.95, "Is it safe to use eval?" vs "Is it dangerous to use eval?"
        scored 0.9874. This is structural, not tunable.
        """
        import re
        # Normalize: lowercase, remove punctuation, split into tokens
        def normalize(text):
            text = text.lower()
            text = re.sub(r'[^\w\s]', ' ', text)  # Replace punctuation with space
            return set(text.split())

        tokens_a = normalize(prompt_a)
        tokens_b = normalize(prompt_b)

        # Negation words
        negations = {"not", "no", "never", "without", "don't", "doesn't", "can't",
                     "won't", "shouldn't", "wouldn't", "couldn't", "isn't", "aren't",
                     "wasn't", "weren't", "haven't", "hasn't", "hadn't"}

        # Check for negation presence difference
        has_neg_a = bool(tokens_a & negations)
        has_neg_b = bool(tokens_b & negations)
        if has_neg_a != has_neg_b:
            return True  # One has negation, other doesn't - conflict

        # Antonym pairs (if one appears in A and its antonym in B, reject)
        antonym_pairs = [
            ("enable", "disable"),
            ("safe", "dangerous"),
            ("safe", "risky"),
            ("secure", "insecure"),
            ("secure", "vulnerable"),
            ("max", "min"),
            ("maximum", "minimum"),
            ("increase", "decrease"),
            ("start", "stop"),
            ("install", "uninstall"),
            ("allow", "block"),
            ("benefits", "limitations"),
            ("benefits", "drawbacks"),
        ]

        for word1, word2 in antonym_pairs:
            if (word1 in tokens_a and word2 in tokens_b) or (word2 in tokens_a and word1 in tokens_b):
                return True  # Antonym conflict

        # Version/number tokens: extract all tokens that contain digits
        # If they differ, it's a version/number conflict
        version_tokens_a = {t for t in tokens_a if re.search(r'\d', t)}
        version_tokens_b = {t for t in tokens_b if re.search(r'\d', t)}

        # If version tokens exist in both and they differ, reject
        if version_tokens_a and version_tokens_b:
            if version_tokens_a != version_tokens_b:
                return True  # Version/number conflict

        return False  # No conflict detected

    def _append_embedding_row(self, key: str, embedding: "np.ndarray"):
        """Append one embedding row to the matrix."""
        if self._embedding_matrix is None:
            self._embedding_matrix = embedding.reshape(1, -1)
            self._embedding_keys = [key]
        else:
            self._embedding_matrix = np.vstack([self._embedding_matrix, embedding])
            self._embedding_keys.append(key)

    def _remove_embedding_rows(self, keys_to_remove: set):
        """Remove rows for evicted keys without full rebuild."""
        if self._embedding_matrix is None or not keys_to_remove:
            return

        # Find indices to keep
        indices_to_keep = [
            i for i, key in enumerate(self._embedding_keys)
            if key not in keys_to_remove
        ]

        if not indices_to_keep:
            self._embedding_matrix = None
            self._embedding_keys = []
        else:
            self._embedding_matrix = self._embedding_matrix[indices_to_keep, :]
            self._embedding_keys = [self._embedding_keys[i] for i in indices_to_keep]

    def _evict_expired_unsafe(self, current_time: float):
        expired_keys = [
            k for k, entry in self._store.items()
            if current_time - entry.timestamp > self.ttl_seconds
        ]
        if expired_keys:
            for k in expired_keys:
                del self._store[k]
                self.metrics.evictions += 1
            # Remove evicted keys from embedding matrix
            if self.enable_embeddings:
                self._remove_embedding_rows(set(expired_keys))

    def _embed_prompt(self, prompt: str) -> Optional["np.ndarray"]:
        """Generate embedding for prompt. Caller must hold lock."""
        return embeddings.get_embedding(prompt)

    def get(self, prompt: str) -> Optional[Any]:
        """Retrieves structured response data if an exact or similarity match is found."""
        result = self.get_with_key(prompt)
        return result[1] if result else None

    def get_with_key(self, prompt: str) -> Optional[tuple[str, Any]]:
        """Retrieves (canonical_key, response_data) if an exact or similarity match is found.

        Returns the matched cache key along with the value, allowing callers to update
        the same entry when adding scope variants.
        """
        now = time.time()
        with self._lock:
            self._evict_expired_unsafe(now)

            # Fast-path: Exact match (O(1))
            if prompt in self._store:
                self.metrics.hits += 1
                self.metrics.exact_hits += 1
                return (prompt, self._store[prompt].response_data)

            # Slow-path: Lexical match (O(N))
            for cached_prompt, entry in self._store.items():
                if self._jaccard_similarity(prompt, cached_prompt) >= self.similarity_threshold:
                    self.metrics.hits += 1
                    self.metrics.lexical_hits += 1
                    return (cached_prompt, entry.response_data)

            # Embedding tier: Semantic match (optional)
            if self.enable_embeddings and self._store:
                query_emb = self._embed_prompt(prompt)
                if query_emb is not None:
                    best_score = -1.0
                    best_key = None
                    best_data = None

                    # Scan last N entries (most recent in insertion order)
                    entries_list = list(self._store.items())
                    scan_start = max(0, len(entries_list) - self.embedding_max_entries_scanned)
                    candidates = entries_list[scan_start:]

                    for cached_prompt, entry in candidates:
                        if entry.embedding is None:
                            entry.embedding = self._embed_prompt(cached_prompt)

                        if entry.embedding is not None:
                            # Apply negation/version guard before accepting semantic match
                            if self._has_negation_or_version_conflict(prompt, cached_prompt):
                                continue  # Skip this candidate - conflict detected

                            score = self._cosine_normalized(query_emb, entry.embedding)
                            if score > best_score:
                                best_score = score
                                best_key = cached_prompt
                                best_data = entry.response_data

                    if best_score >= self.embedding_threshold and best_data is not None:
                        self.metrics.hits += 1
                        self.metrics.embedding_hits += 1
                        return (best_key, best_data)

            self.metrics.misses += 1
            return None

    def get_exact_no_metrics(self, prompt: str) -> Optional[Any]:
        """Get exact match without incrementing metrics. For internal use during cache updates."""
        with self._lock:
            if prompt in self._store:
                return self._store[prompt].response_data
            return None

    def get_with_key_no_metrics(self, prompt: str) -> Optional[tuple[str, Any]]:
        """Retrieves (canonical_key, response_data) without incrementing metrics.

        Returns the matched cache key along with the value, for scope validation
        before committing to a cache hit or miss in metrics.
        """
        now = time.time()
        with self._lock:
            self._evict_expired_unsafe(now)

            # Fast-path: Exact match (O(1))
            if prompt in self._store:
                self.metrics.exact_hits += 1
                return (prompt, self._store[prompt].response_data)

            # Slow-path: Lexical match (O(N))
            for cached_prompt, entry in self._store.items():
                if self._jaccard_similarity(prompt, cached_prompt) >= self.similarity_threshold:
                    self.metrics.lexical_hits += 1
                    return (cached_prompt, entry.response_data)

            # Embedding tier: Semantic match (optional, vectorized)
            if self.enable_embeddings and self._store:
                self.metrics.embedding_scans_attempted += 1
                query_emb = self._embed_prompt(prompt)

                if query_emb is not None and self._embedding_matrix is not None and len(self._embedding_keys) > 0:
                    # Scan last N entries (window)
                    scan_start = max(0, len(self._embedding_keys) - self.embedding_max_entries_scanned)
                    scan_keys = self._embedding_keys[scan_start:]
                    scan_matrix = self._embedding_matrix[scan_start:, :]

                    # Vectorized cosine: (N,384) @ (384,) = (N,)
                    similarities = scan_matrix @ query_emb

                    # Find best match
                    best_idx = int(np.argmax(similarities))
                    best_score = float(similarities[best_idx])

                    self.metrics.embedding_comparisons_performed += len(scan_keys)

                    logger.debug(f"Embedding scan: best_score={best_score:.4f}, threshold={self.embedding_threshold}, hit={best_score >= self.embedding_threshold}")

                    if best_score >= self.embedding_threshold:
                        self.metrics.embedding_hits += 1
                        best_key = scan_keys[best_idx]
                        best_data = self._store[best_key].response_data
                        # DELIBERATE: Best match wins, then scope gate applies in proxy.
                        # If scope dict lacks requesting scope, proxy will mark miss.
                        # Do not fall through to next candidate - fail safe.
                        return (best_key, best_data)

            return None

    def put(self, prompt: str, response_data: Any):
        """Stores structured response data in the cache."""
        now = time.time()
        with self._lock:
            self._evict_expired_unsafe(now)

            # Evict oldest (FIFO) if we hit the size limit
            if len(self._store) >= self.max_size and prompt not in self._store:
                oldest_key = next(iter(self._store))
                del self._store[oldest_key]
                self.metrics.evictions += 1
                # Remove evicted key from embedding matrix
                if self.enable_embeddings:
                    self._remove_embedding_rows({oldest_key})

            # Eager embed on write if embeddings enabled
            embedding = None
            if self.enable_embeddings:
                embedding = self._embed_prompt(prompt)

            self._store[prompt] = _CacheEntry(
                prompt=prompt,
                response_data=response_data,
                timestamp=now,
                embedding=embedding,
            )

            # Append embedding row after adding to store
            if self.enable_embeddings and embedding is not None:
                self._append_embedding_row(prompt, embedding)
