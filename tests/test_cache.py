import time
import unittest

from throttle.cache import SimilarityCache


class TestSimilarityCache(unittest.TestCase):
    def setUp(self):
        self.cache = SimilarityCache(
            ttl_seconds=1.0,
            max_size=2,
            similarity_threshold=0.80,
            enable_embeddings=False,
        )

    def test_validation(self):
        with self.assertRaises(ValueError):
            SimilarityCache(ttl_seconds=-1)
        with self.assertRaises(ValueError):
            SimilarityCache(max_size=0)
        with self.assertRaises(ValueError):
            SimilarityCache(similarity_threshold=1.5)

    def test_exact_match(self):
        self.cache.put("What is the capital of France?", {"text": "Paris"})
        res = self.cache.get("What is the capital of France?")
        self.assertEqual(res["text"], "Paris")
        self.assertEqual(self.cache.metrics.hits, 1)
        self.assertEqual(self.metrics_lexical_hits(), 1)

    def metrics_lexical_hits(self):
        return self.cache.metrics.lexical_hits

    def test_lexical_match(self):
        self.cache.put(
            "Explain the theory of relativity briefly",
            {"text": "Einstein's theory..."},
        )
        res = self.cache.get("Explain the theory of relativity please briefly")
        self.assertIsNotNone(res)
        self.assertEqual(res["text"], "Einstein's theory...")
        self.assertEqual(self.cache.metrics.lexical_hits, 1)

    def test_miss(self):
        self.cache.put("Tell me a joke", {"text": "Why did the chicken..."})
        res = self.cache.get("Write a python script")
        self.assertIsNone(res)
        self.assertEqual(self.cache.metrics.misses, 1)

    def test_ttl_eviction(self):
        self.cache.put("Temporary prompt", {"text": "Value"})
        time.sleep(1.1)
        res = self.cache.get("Temporary prompt")
        self.assertIsNone(res)
        self.assertGreaterEqual(self.cache.metrics.evictions, 1)

    def test_max_size_eviction(self):
        self.cache.put("P 1", {"text": "A"})
        self.cache.put("P 2", {"text": "B"})
        self.cache.put("P 3", {"text": "C"})
        self.assertIsNone(self.cache.get("P 1"))
        self.assertIsNotNone(self.cache.get("P 2"))
        self.assertIsNotNone(self.cache.get("P 3"))
        self.assertGreaterEqual(self.cache.metrics.evictions, 1)

    def test_stats_shape(self):
        self.cache.put("hello world", {"text": "ok"})
        stats = self.cache.stats()
        self.assertIn("hit_rate", stats)
        self.assertIn("embeddings_enabled", stats)
        self.assertFalse(stats["embeddings_enabled"])


if __name__ == "__main__":
    unittest.main()
