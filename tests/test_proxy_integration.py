"""Integration tests for proxy against a real inference backend.

These tests verify cache behavior, scope isolation, in-flight deduplication,
streaming, and error handling with real HTTP calls against OpenAI-compatible backends.

Backend and model can be configured via environment variables:
- BACKEND_URL: Backend server URL (default: http://localhost:11434 for Ollama)
- BACKEND_URL_2: Secondary backend URL for MODEL_2 (default: same as BACKEND_URL)
- BACKEND_MODEL: Model name (default: llama3.2:1b for Ollama)
- BACKEND_MODEL_2: Secondary model name (default: llama3.2:3b for Ollama)

Tests are skipped if no backend is available rather than failing.
"""

import asyncio
import json
import os
import time
import unittest
from typing import List, Dict, Any

import httpx
import pytest
import uvicorn

from throttle.proxy import ProxyServer

# The proxy only caches and dedupes requests that carry a key.
AUTH = {"Authorization": "Bearer test-key"}


def is_backend_available(backend_url: str) -> bool:
    """Check if backend is reachable.

    Tries both Ollama-specific endpoint and generic health endpoint.
    """
    try:
        # Try Ollama-specific endpoint
        response = httpx.get(f"{backend_url.rstrip('/')}/api/tags", timeout=5.0)
        if response.status_code == 200:
            return True
    except Exception:
        pass

    try:
        # Try generic health endpoint
        response = httpx.get(f"{backend_url.rstrip('/')}/health", timeout=5.0)
        if response.status_code == 200:
            return True
    except Exception:
        pass

    try:
        # Try OpenAI-compatible models endpoint
        response = httpx.get(f"{backend_url.rstrip('/')}/v1/models", timeout=5.0)
        if response.status_code == 200:
            return True
    except Exception:
        pass

    return False


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:11434")
BACKEND_URL_2 = os.getenv("BACKEND_URL_2", os.getenv("BACKEND_URL", "http://localhost:11434"))
BACKEND_MODEL = os.getenv("BACKEND_MODEL", "llama3.2:1b")
BACKEND_MODEL_2 = os.getenv("BACKEND_MODEL_2", "llama3.2:3b")
SKIP_REASON = f"Backend not available at {BACKEND_URL}"


@pytest.mark.skipif(not is_backend_available(BACKEND_URL), reason=SKIP_REASON)
class ProxyIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """Integration tests with real Ollama backend."""

    async def asyncSetUp(self):
        """Start proxy server with cache enabled."""
        # Build model-to-backend mapping
        # If BACKEND_URL_2 differs from BACKEND_URL, route BACKEND_MODEL_2 to it
        model_backends = {}
        if BACKEND_URL_2 != BACKEND_URL:
            model_backends[BACKEND_MODEL_2] = BACKEND_URL_2

        self.proxy = ProxyServer(
            backend_url=BACKEND_URL,
            enable_cache=True,
            cache_ttl_seconds=3600,
            cache_max_size=100,
            model_backends=model_backends if model_backends else None,
        )

        app = self.proxy.app

        @app.on_event("startup")
        async def startup():
            await self.proxy.startup()

        @app.on_event("shutdown")
        async def shutdown():
            await self.proxy.shutdown()

        # Use random port
        config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
        self.server = uvicorn.Server(config)

        # Start server in background
        self.server_task = asyncio.create_task(self.server.serve())
        await asyncio.sleep(1)  # Wait for server to start

        # Get the actual port assigned
        self.proxy_port = self.server.servers[0].sockets[0].getsockname()[1]
        self.proxy_url = f"http://127.0.0.1:{self.proxy_port}/v1"

    async def asyncTearDown(self):
        """Stop proxy server."""
        if hasattr(self, 'server'):
            self.server.should_exit = True
            await self.server_task

    async def _make_request(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Make chat completion request through proxy."""
        if model is None:
            model = BACKEND_MODEL

        async with httpx.AsyncClient(timeout=60.0, headers=AUTH) as client:
            payload = {
                "model": model,
                "messages": messages,
                "stream": stream,
                **kwargs
            }
            response = await client.post(
                f"{self.proxy_url}/chat/completions",
                json=payload,
            )
            response.raise_for_status()

            if stream:
                # Collect all chunks
                chunks = []
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data != "[DONE]":
                            chunks.append(json.loads(data))
                return {"chunks": chunks}
            else:
                return response.json()

    async def _get_health(self) -> Dict[str, Any]:
        """Get proxy health metrics."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"http://127.0.0.1:{self.proxy_port}/health")
            response.raise_for_status()
            return response.json()

    async def test_cache_miss_then_hit_on_exact_repeat(self):
        """Cache miss on first request, hit on exact repeat with identical response."""
        messages = [{"role": "user", "content": "What is 2+2?"}]

        # First request - cache miss
        health_before = await self._get_health()
        misses_before = health_before.get("cache_stats", {}).get("misses", 0)

        response1 = await self._make_request(messages)

        health_after_first = await self._get_health()
        misses_after_first = health_after_first.get("cache_stats", {}).get("misses", 0)
        hits_after_first = health_after_first.get("cache_stats", {}).get("hits", 0)

        self.assertEqual(misses_after_first, misses_before + 1, "First request should be cache miss")

        # Second request - cache hit
        response2 = await self._make_request(messages)

        health_after_second = await self._get_health()
        hits_after_second = health_after_second.get("cache_stats", {}).get("hits", 0)

        self.assertEqual(hits_after_second, hits_after_first + 1, "Second request should be cache hit")

        # Responses should be identical
        self.assertEqual(
            response1.get("choices", [{}])[0].get("message", {}).get("content"),
            response2.get("choices", [{}])[0].get("message", {}).get("content"),
            "Cached response content should match original"
        )

    async def test_exact_repeat_hits_paraphrase_misses_current_behavior(self):
        """CURRENT BEHAVIOR: Exact repeat hits cache, paraphrase misses, unrelated misses.

        This test documents current behavior with Jaccard similarity at 0.85 threshold.
        Paraphrases are NOT cached because lexical similarity is too strict.
        When semantic cache lands, update this test to expect paraphrase hits.
        """
        original = [{"role": "user", "content": "What is the capital of France?"}]
        exact_repeat = [{"role": "user", "content": "What is the capital of France?"}]
        paraphrase = [{"role": "user", "content": "Tell me the capital city of France"}]
        unrelated = [{"role": "user", "content": "What is the square root of 144?"}]

        # Prime cache with original
        await self._make_request(original)
        health_after_prime = await self._get_health()
        backend_after_prime = health_after_prime.get("cache_stats", {}).get("backend_calls", 0)

        # Exact repeat should hit cache
        await self._make_request(exact_repeat)
        health_after_exact = await self._get_health()
        hits_after_exact = health_after_exact.get("cache_stats", {}).get("hits", 0)
        backend_after_exact = health_after_exact.get("cache_stats", {}).get("backend_calls", 0)

        self.assertEqual(hits_after_exact, 1, "Exact repeat should hit cache")
        self.assertEqual(backend_after_exact, backend_after_prime, "Exact repeat should not hit backend")

        # CURRENT BEHAVIOR: Paraphrase misses cache (Jaccard too strict)
        # When semantic cache lands, expect this to be a hit
        await self._make_request(paraphrase)
        health_after_paraphrase = await self._get_health()
        backend_after_paraphrase = health_after_paraphrase.get("cache_stats", {}).get("backend_calls", 0)

        self.assertEqual(
            backend_after_paraphrase,
            backend_after_exact + 1,
            "CURRENT: Paraphrase should miss cache and hit backend"
        )

        # Unrelated query should miss
        await self._make_request(unrelated)
        health_after_unrelated = await self._get_health()
        backend_after_unrelated = health_after_unrelated.get("cache_stats", {}).get("backend_calls", 0)

        self.assertEqual(
            backend_after_unrelated,
            backend_after_paraphrase + 1,
            "Unrelated query should miss cache and hit backend"
        )

    async def test_scope_isolation_different_parameters(self):
        """Different model, temperature, max_tokens each reach backend separately."""
        messages = [{"role": "user", "content": "Count to three"}]

        health_before = await self._get_health()
        backend_calls_before = health_before.get("cache_stats", {}).get("backend_calls", 0)

        # Same messages, different model
        await self._make_request(messages)
        await self._make_request(messages, model=BACKEND_MODEL_2)

        # Same messages and model, different temperature
        await self._make_request(messages, temperature=0.7)
        await self._make_request(messages, temperature=0.9)

        # Same messages and model, different max_tokens
        await self._make_request(messages, max_tokens=10)
        await self._make_request(messages, max_tokens=20)

        health_after = await self._get_health()
        backend_calls_after = health_after.get("cache_stats", {}).get("backend_calls", 0)

        # Each unique scope should hit the backend
        # model variants: 2, temperature variants: 2, max_tokens variants: 2
        # Total unique scopes: 6
        expected_calls = backend_calls_before + 6

        self.assertEqual(
            backend_calls_after,
            expected_calls,
            f"Each unique scope should hit backend: expected {expected_calls} calls, got {backend_calls_after}"
        )

    async def test_scope_variants_coexisting(self):
        """Prime model A, then B, then A again - A should still be cached."""
        messages = [{"role": "user", "content": "Say hello"}]

        # Prime model A cache
        await self._make_request(messages)
        health_after_a1 = await self._get_health()
        hits_after_a1 = health_after_a1.get("cache_stats", {}).get("hits", 0)

        # Use model B - different cache scope
        await self._make_request(messages, model=BACKEND_MODEL_2)

        # Use model A again - should hit its cache
        await self._make_request(messages)
        health_after_a2 = await self._get_health()
        hits_after_a2 = health_after_a2.get("cache_stats", {}).get("hits", 0)

        self.assertEqual(
            hits_after_a2,
            hits_after_a1 + 1,
            "Model A cache should survive after using model B"
        )

    async def test_inflight_deduplication_concurrent_identical_requests(self):
        """5 concurrent identical requests result in exactly one backend call."""
        messages = [{"role": "user", "content": "What is the meaning of life?"}]

        health_before = await self._get_health()
        backend_calls_before = health_before.get("cache_stats", {}).get("backend_calls", 0)

        # Launch 5 concurrent identical requests
        tasks = [
            self._make_request(messages)
            for _ in range(5)
        ]
        responses = await asyncio.gather(*tasks)

        health_after = await self._get_health()
        backend_calls_after = health_after.get("cache_stats", {}).get("backend_calls", 0)

        # Exactly one should reach backend (in-flight deduplication worked)
        self.assertEqual(
            backend_calls_after,
            backend_calls_before + 1,
            "Exactly one request should reach backend"
        )

        # All responses should be identical
        first_content = responses[0].get("choices", [{}])[0].get("message", {}).get("content")
        for response in responses[1:]:
            content = response.get("choices", [{}])[0].get("message", {}).get("content")
            self.assertEqual(
                content,
                first_content,
                "All concurrent responses should be identical"
            )

    async def test_streaming_cached_response(self):
        """Cached response served as fake stream, chunks match non-streamed body."""
        messages = [{"role": "user", "content": "Write a haiku"}]

        # Non-streaming request to prime cache
        non_streamed = await self._make_request(messages, stream=False)
        non_streamed_content = non_streamed.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Streaming request should serve from cache
        health_before_stream = await self._get_health()
        hits_before_stream = health_before_stream.get("cache_stats", {}).get("hits", 0)

        streamed = await self._make_request(messages, stream=True)

        health_after_stream = await self._get_health()
        hits_after_stream = health_after_stream.get("cache_stats", {}).get("hits", 0)

        self.assertEqual(
            hits_after_stream,
            hits_before_stream + 1,
            "Streaming request should hit cache"
        )

        # Reconstruct content from chunks
        chunks = streamed.get("chunks", [])
        self.assertGreater(len(chunks), 0, "Stream should have chunks")

        streamed_content = "".join(
            chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
            for chunk in chunks
        )

        self.assertEqual(
            streamed_content,
            non_streamed_content,
            "Streamed content should match non-streamed cached content"
        )

    async def test_backend_error_propagation(self):
        """Backend error propagates with correct status and is not cached."""
        # Use invalid model to trigger 404
        messages = [{"role": "user", "content": "test"}]

        health_before = await self._get_health()
        backend_calls_before = health_before.get("cache_stats", {}).get("backend_calls", 0)

        # First request should fail
        with self.assertRaises(httpx.HTTPStatusError) as cm:
            await self._make_request(messages, model="nonexistent-model-12345")

        self.assertEqual(cm.exception.response.status_code, 404, "Should propagate 404 status")

        # Second identical request should also fail (not cached)
        with self.assertRaises(httpx.HTTPStatusError) as cm:
            await self._make_request(messages, model="nonexistent-model-12345")

        self.assertEqual(cm.exception.response.status_code, 404, "Error should not be cached")

        health_after = await self._get_health()
        backend_calls_after = health_after.get("cache_stats", {}).get("backend_calls", 0)

        # Both requests should reach backend (errors not cached)
        self.assertEqual(
            backend_calls_after,
            backend_calls_before + 2,
            "Error responses should not be cached"
        )
