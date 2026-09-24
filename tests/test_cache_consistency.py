"""Test embedding matrix consistency with store operations."""
import time
import pytest

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

from throttle.cache import SimilarityCache

# Skip all tests in this module if numpy is not available
pytestmark = pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy required for embedding tests")


def _assert_matrix_consistent(cache):
    """Assert embedding matrix is consistent with store."""
    if not cache.enable_embeddings:
        assert cache._embedding_matrix is None
        assert cache._embedding_keys == []
        return

    # Collect entries with embeddings from store
    store_keys_with_embeddings = [
        k for k, entry in cache._store.items()
        if entry.embedding is not None
    ]

    # Matrix keys should match store keys with embeddings
    assert set(cache._embedding_keys) == set(store_keys_with_embeddings)
    assert len(cache._embedding_keys) == len(store_keys_with_embeddings)

    if len(store_keys_with_embeddings) == 0:
        assert cache._embedding_matrix is None
        assert cache._embedding_keys == []
    else:
        assert cache._embedding_matrix is not None
        assert cache._embedding_matrix.shape == (len(store_keys_with_embeddings), 384)

        # Verify each row matches corresponding entry
        for i, key in enumerate(cache._embedding_keys):
            store_embedding = cache._store[key].embedding
            matrix_row = cache._embedding_matrix[i]
            assert np.allclose(matrix_row, store_embedding)


def test_matrix_consistency_on_put():
    """Matrix stays consistent after put operations."""
    cache = SimilarityCache(
        ttl_seconds=3600.0,
        max_size=5,
        enable_embeddings=True,
    )

    # Initial state: empty
    _assert_matrix_consistent(cache)

    # Add first entry
    cache.put("prompt1", {"response": "data1"})
    _assert_matrix_consistent(cache)

    # Add more entries
    cache.put("prompt2", {"response": "data2"})
    cache.put("prompt3", {"response": "data3"})
    _assert_matrix_consistent(cache)


def test_matrix_consistency_on_ttl_eviction():
    """Matrix stays consistent after TTL eviction."""
    cache = SimilarityCache(
        ttl_seconds=0.5,  # Short TTL
        max_size=10,
        enable_embeddings=True,
    )

    # Add entries
    cache.put("prompt1", {"response": "data1"})
    cache.put("prompt2", {"response": "data2"})
    cache.put("prompt3", {"response": "data3"})
    _assert_matrix_consistent(cache)

    # Wait for TTL expiration
    time.sleep(0.6)

    # Trigger eviction via get (calls _evict_expired_unsafe)
    cache.get("prompt4")
    _assert_matrix_consistent(cache)

    # Store should be empty after TTL eviction
    assert len(cache._store) == 0
    assert cache._embedding_matrix is None
    assert cache._embedding_keys == []


def test_matrix_consistency_on_fifo_eviction():
    """Matrix stays consistent after FIFO eviction."""
    cache = SimilarityCache(
        ttl_seconds=3600.0,
        max_size=3,  # Small size to trigger FIFO
        enable_embeddings=True,
    )

    # Fill cache to max_size
    cache.put("prompt1", {"response": "data1"})
    cache.put("prompt2", {"response": "data2"})
    cache.put("prompt3", {"response": "data3"})
    _assert_matrix_consistent(cache)

    assert len(cache._store) == 3

    # Add one more to trigger FIFO eviction
    cache.put("prompt4", {"response": "data4"})
    _assert_matrix_consistent(cache)

    # Store should still be at max_size
    assert len(cache._store) == 3
    # prompt1 should have been evicted (FIFO)
    assert "prompt1" not in cache._store
    assert "prompt4" in cache._store


def test_matrix_not_rebuilt_during_get():
    """Assert matrix is never rebuilt during a get operation."""
    cache = SimilarityCache(
        ttl_seconds=3600.0,
        max_size=100,
        enable_embeddings=True,
    )

    # Add entries
    for i in range(10):
        cache.put(f"prompt{i}", {"response": f"data{i}"})

    # Capture matrix id before get
    matrix_id_before = id(cache._embedding_matrix)

    # Perform get that will miss
    result = cache.get_with_key_no_metrics("query that will not match")

    # Matrix object should be the same instance (not rebuilt)
    matrix_id_after = id(cache._embedding_matrix)

    assert matrix_id_before == matrix_id_after, "Matrix was rebuilt during get"
    assert result is None  # Confirm it was a miss
