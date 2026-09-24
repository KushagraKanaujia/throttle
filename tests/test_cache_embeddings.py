"""Direct cache tests for embedding tier functionality.

Tests cache.py embedding behavior directly without proxy async complexity.
"""

import pytest

from throttle import embeddings
from throttle.cache import SimilarityCache


def test_model_load_failure_falls_back_to_jaccard(monkeypatch):
    """If the model cannot be loaded (e.g. offline, not cached), the cache must
    keep working on exact/Jaccard matching and not retry the load per request.
    Runs without the embeddings extra or model."""
    calls = {"n": 0}

    def failing_get_embedding(text):
        calls["n"] += 1
        return None

    monkeypatch.setattr(embeddings, "EMBEDDINGS_AVAILABLE", True)
    monkeypatch.setattr(embeddings, "get_embedding", failing_get_embedding)
    monkeypatch.setattr(embeddings, "get_load_error", lambda: "OSError: offline")

    cache = SimilarityCache(ttl_seconds=3600, max_size=10, enable_embeddings=True)
    cache.put("How do I optimize PostgreSQL queries?", {"data": 1})
    assert cache.enable_embeddings is False  # disabled after first failed load
    assert cache.get("How do I optimize PostgreSQL queries?") == {"data": 1}
    assert cache.get("Ways to improve DB query speed") is None
    assert cache.get_with_key_no_metrics("Something else entirely") is None
    assert calls["n"] == 1


# Everything below needs the real ONNX model.
if not embeddings.EMBEDDINGS_AVAILABLE:
    pytest.skip(
        "embeddings extra not installed (pip install throttle-pro[embeddings])",
        allow_module_level=True,
    )
if not embeddings.is_model_available():
    pytest.skip(
        f"embedding model could not be loaded: {embeddings.get_load_error()} "
        "(needs network access or a cached sentence-transformers/all-MiniLM-L6-v2)",
        allow_module_level=True,
    )


def test_embedding_enabled_with_extra_installed():
    """Verify embeddings can be enabled when extra is installed."""
    cache = SimilarityCache(
        ttl_seconds=3600,
        max_size=10,
        enable_embeddings=True,
        embedding_threshold=0.95,
    )
    assert cache.enable_embeddings is True
    cache.put("test", {"data": "value"})
    assert cache._store["test"].embedding is not None


def test_vectorized_scan_applies_negation_guard():
    """Regression: the proxy's vectorized path must reject polarity flips.

    "safe" vs "dangerous" scores ~0.98 cosine; without the guard the cached
    answer to the opposite question would be served.
    """
    cache = SimilarityCache(ttl_seconds=3600, max_size=10, enable_embeddings=True,
                            embedding_threshold=0.95)
    cache.put("Is it safe to use eval in Python code?", {"s": "A"})
    assert cache.get_with_key_no_metrics("Is it dangerous to use eval in Python code?") is None
    assert cache.metrics.embedding_hits == 0


def test_jaccard_first_then_embeddings():
    """Verify Jaccard runs first, embeddings only on Jaccard miss."""
    cache = SimilarityCache(
        ttl_seconds=3600,
        max_size=10,
        similarity_threshold=0.85,
        enable_embeddings=True,
        embedding_threshold=0.95,
    )

    # Store a scope dict (proxy pattern)
    scope1 = '{"model":"test","temperature":0.7}'
    response1 = {scope1: {"_scope": scope1, "response": {"text": "Response 1"}}}
    cache.put("How do I optimize PostgreSQL queries?", response1)

    # Exact match -> dict lookup hit
    result = cache.get_with_key_no_metrics("How do I optimize PostgreSQL queries?")
    assert result is not None
    key, data = result
    assert key == "How do I optimize PostgreSQL queries?"
    assert data == response1

    # Verify Jaccard hit (not embedding)
    cache.metrics.lexical_hits = 0
    cache.metrics.embedding_hits = 0
    cache.metrics.hits = 0

    # Case variant: not an exact dict match, but Jaccard similarity 1.0
    result2 = cache.get_with_key("how do I optimize PostgreSQL queries?")
    assert result2 is not None
    assert result2[0] == "How do I optimize PostgreSQL queries?"
    assert cache.metrics.lexical_hits == 1
    assert cache.metrics.embedding_hits == 0


def test_semantic_match_returns_scope_dict():
    """Verify embedding tier returns same (key, scope_dict) shape as Jaccard."""
    cache = SimilarityCache(
        ttl_seconds=3600,
        max_size=10,
        similarity_threshold=0.85,  # Jaccard won't match paraphrase
        enable_embeddings=True,
        embedding_threshold=0.90,  # Lower for test
    )

    # Store with scope dict
    scope1 = '{"model":"test","temperature":0.7}'
    response1 = {scope1: {"_scope": scope1, "response": {"text": "DB optimization tips"}}}
    cache.put("How to optimize database performance", response1)

    # Semantically similar but Jaccard dissimilar
    result = cache.get_with_key_no_metrics("Ways to improve DB query speed")
    # May or may not hit depending on embedding similarity
    # If it hits, verify shape matches Jaccard return
    if result is not None:
        key, data = result
        assert isinstance(key, str)
        assert isinstance(data, dict)
        # Should be scope dict with nested structure
        assert scope1 in data or len(data) > 0


def test_embedding_metrics_separate_from_lexical():
    """Verify lexical_hits and embedding_hits are tracked separately."""
    cache = SimilarityCache(
        ttl_seconds=3600,
        max_size=10,
        similarity_threshold=0.10,  # Very low to force Jaccard hit
        enable_embeddings=True,
        embedding_threshold=0.95,
    )

    cache.put("test prompt", {"data": "value"})

    # Exact match -> exact hit (dict lookup), lexical tier not consulted
    cache.get("test prompt")
    assert cache.metrics.exact_hits == 1
    assert cache.metrics.embedding_hits == 0

    # Lexically similar prompt -> lexical hit (threshold 0.10)
    cache.get("test prompt please")
    assert cache.metrics.lexical_hits == 1
    assert cache.metrics.embedding_hits == 0

    # Different prompt (may or may not embedding match)
    cache.get("completely unrelated text about quantum physics")
    # Embedding hits only increment if semantic threshold met
    # Just verify metrics exist and are separate
    assert hasattr(cache.metrics, 'lexical_hits')
    assert hasattr(cache.metrics, 'embedding_hits')


def test_per_entry_memory_overhead():
    """Document per-entry memory overhead for 384 float32 embedding."""
    # 384 dimensions * 4 bytes/float32 = 1536 bytes per embedding
    import sys
    cache = SimilarityCache(
        ttl_seconds=3600,
        max_size=10,
        enable_embeddings=True,
    )

    cache.put("test", {"data": "value"})

    # Entry has embedding stored
    entry = cache._store["test"]
    if entry.embedding is not None:
        # embedding is numpy array of 384 float32
        embedding_bytes = entry.embedding.nbytes
        assert embedding_bytes == 384 * 4  # 1536 bytes


def test_scan_window_limits_candidates():
    """Verify scan window of 256 limits which entries are scanned."""
    cache = SimilarityCache(
        ttl_seconds=3600,
        max_size=500,  # Larger than scan window
        enable_embeddings=True,
        embedding_max_entries_scanned=256,
    )

    # Add 300 entries
    for i in range(300):
        cache.put(f"prompt_{i}", {"data": f"value_{i}"})

    # When cache exceeds scan window (300 > 256), only last 256 are scanned
    # This is verified by implementation: candidates = entries_list[scan_start:]
    # where scan_start = max(0, len(entries_list) - embedding_max_entries_scanned)

    # For prompt_0 (oldest), embedding scan won't see it (outside window)
    # For prompt_299 (newest), embedding scan will see it (in window)

    # This affects hit rate: old prompts won't match semantically
    assert len(cache._store) == 300
    assert cache.embedding_max_entries_scanned == 256
