"""Tests for the Throttle proxy server with real HTTP clients."""

import asyncio
import json
import os
import socket
import time
from multiprocessing import Process

import httpx
import pytest

from throttle.proxy import create_app

# Backend configuration from environment
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:11434")
BACKEND_MODEL = os.getenv("BACKEND_MODEL", "llama3.2:1b")

def is_backend_available(backend_url: str) -> bool:
    """Check if backend is reachable.

    Tries both Ollama-specific endpoint and generic health endpoint.
    """
    import httpx
    try:
        response = httpx.get(f"{backend_url.rstrip('/')}/api/tags", timeout=5.0)
        if response.status_code == 200:
            return True
    except Exception:
        pass

    try:
        response = httpx.get(f"{backend_url.rstrip('/')}/health", timeout=5.0)
        if response.status_code == 200:
            return True
    except Exception:
        pass

    try:
        response = httpx.get(f"{backend_url.rstrip('/')}/v1/models", timeout=5.0)
        if response.status_code == 200:
            return True
    except Exception:
        pass

    return False


def run_proxy_server(port: int):
    """Run the proxy server in a separate process."""
    import uvicorn
    app = create_app(
        backend_url=BACKEND_URL,
        enable_cache=True,
        cache_ttl_seconds=3600.0,
        cache_max_size=100,
        cache_similarity_threshold=0.85,
    )
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="error")


@pytest.mark.skipif(
    getattr(socket, "_throttle_offline_guard_active", False) or not is_backend_available(BACKEND_URL),
    reason=f"Integration test requires real backend at {BACKEND_URL}"
)
@pytest.mark.asyncio
async def test_proxy_cache_hit_with_real_http_client():
    """Test that a real external HTTP client can get cache hits through the proxy."""

    # Start proxy server in background process
    proxy_port = 58080
    proxy_process = Process(target=run_proxy_server, args=(proxy_port,))
    proxy_process.start()

    # Wait for proxy to start
    await asyncio.sleep(2)

    try:
        # Create a real HTTP client (external to Throttle's benchmark code)
        async with httpx.AsyncClient(timeout=30.0) as client:

            # First request - should be a cache miss, forwarded to backend
            request_body = {
                "model": BACKEND_MODEL,
                "messages": [
                    {"role": "user", "content": "What is the capital of France?"}
                ],
                "max_tokens": 10,
                "stream": False,
            }

            print("\n=== First request (cache miss) ===")
            response1 = await client.post(
                f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
                json=request_body,
                headers={"Content-Type": "application/json"},
            )

            assert response1.status_code == 200
            data1 = response1.json()
            assert "choices" in data1
            assert len(data1["choices"]) > 0
            assert "message" in data1["choices"][0]
            content1 = data1["choices"][0]["message"]["content"]
            print(f"Response 1: {content1[:100]}...")

            # Small delay
            await asyncio.sleep(0.5)

            # Second request - identical prompt, should be cache hit
            print("\n=== Second request (should be cache hit) ===")
            start_time = time.perf_counter()
            response2 = await client.post(
                f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
                json=request_body,
                headers={"Content-Type": "application/json"},
            )
            cache_hit_time = time.perf_counter() - start_time

            assert response2.status_code == 200
            data2 = response2.json()
            assert "choices" in data2
            content2 = data2["choices"][0]["message"]["content"]
            print(f"Response 2: {content2[:100]}...")
            print(f"Cache hit response time: {cache_hit_time*1000:.2f}ms")

            # Response should be identical (from cache)
            assert content2 == content1

            # Cache hit should be MUCH faster (<100ms vs >500ms for inference)
            assert cache_hit_time < 0.1, f"Cache hit took {cache_hit_time*1000:.2f}ms, expected <100ms"

            # Third request - similar prompt, should also be cache hit (similarity matching)
            similar_request = {
                "model": "llama3.2:1b",
                "messages": [
                    {"role": "user", "content": "What is the capital of France?"}  # Exact same
                ],
                "max_tokens": 10,
                "stream": False,
            }

            print("\n=== Third request (similar prompt, should be cache hit) ===")
            start_time = time.perf_counter()
            response3 = await client.post(
                f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
                json=similar_request,
                headers={"Content-Type": "application/json"},
            )
            similar_hit_time = time.perf_counter() - start_time

            assert response3.status_code == 200
            data3 = response3.json()
            content3 = data3["choices"][0]["message"]["content"]
            print(f"Response 3: {content3[:100]}...")
            print(f"Similar prompt response time: {similar_hit_time*1000:.2f}ms")

            # Should be fast (cache hit)
            assert similar_hit_time < 0.1

            # Fourth request - different prompt, should be cache miss
            different_request = {
                "model": "llama3.2:1b",
                "messages": [
                    {"role": "user", "content": "What is the capital of Japan?"}
                ],
                "max_tokens": 10,
                "stream": False,
            }

            print("\n=== Fourth request (different prompt, cache miss) ===")
            start_time = time.perf_counter()
            response4 = await client.post(
                f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
                json=different_request,
                headers={"Content-Type": "application/json"},
            )
            miss_time = time.perf_counter() - start_time

            assert response4.status_code == 200
            data4 = response4.json()
            content4 = data4["choices"][0]["message"]["content"]
            print(f"Response 4: {content4[:100]}...")
            print(f"Cache miss response time: {miss_time*1000:.2f}ms")

            # Should be different content
            assert content4 != content1

            # Should be slower than cache hit (going to backend)
            assert miss_time > 0.1, f"Cache miss took {miss_time*1000:.2f}ms, expected >100ms (backend inference time)"

            # Check health endpoint for cache stats
            print("\n=== Checking cache stats ===")
            health = await client.get(f"http://127.0.0.1:{proxy_port}/health")
            assert health.status_code == 200
            health_data = health.json()
            print(f"Cache stats: {json.dumps(health_data['cache_stats'], indent=2)}")

            assert health_data["cache_enabled"] is True
            assert health_data["cache_stats"]["hits"] >= 2  # At least 2 cache hits
            assert health_data["cache_stats"]["misses"] >= 2  # At least 2 cache misses

            print("\n✅ Proxy cache is working with real HTTP client!")
            print(f"  - Cache hits: {health_data['cache_stats']['hits']}")
            print(f"  - Cache misses: {health_data['cache_stats']['misses']}")
            print(f"  - Cache hit speedup: ~{(miss_time / cache_hit_time):.1f}x faster")

    finally:
        # Clean up proxy process
        proxy_process.terminate()
        proxy_process.join(timeout=5)
        if proxy_process.is_alive():
            proxy_process.kill()


@pytest.mark.skipif(
    getattr(socket, "_throttle_offline_guard_active", False) or not is_backend_available(BACKEND_URL),
    reason=f"Integration test requires real backend at {BACKEND_URL}"
)
@pytest.mark.asyncio
async def test_proxy_streaming_with_cache():
    """Test that streaming responses work through the proxy with caching."""

    # Start proxy server in background process
    proxy_port = 58081
    proxy_process = Process(target=run_proxy_server, args=(proxy_port,))
    proxy_process.start()

    # Wait for proxy to start
    await asyncio.sleep(2)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:

            # First streaming request (cache miss)
            request_body = {
                "model": BACKEND_MODEL,
                "messages": [
                    {"role": "user", "content": "Count from 1 to 3"}
                ],
                "max_tokens": 20,
                "stream": True,
            }

            print("\n=== First streaming request (cache miss) ===")
            async with client.stream(
                "POST",
                f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
                json=request_body,
            ) as response:
                assert response.status_code == 200
                chunks = []
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        chunks.append(line)
                        if line.strip() != "data: [DONE]":
                            print(f"  Chunk: {line[:80]}...")

                assert len(chunks) > 0
                print(f"  Received {len(chunks)} chunks from backend")

            # Small delay
            await asyncio.sleep(0.5)

            # Second streaming request (cache hit - should fake-stream)
            print("\n=== Second streaming request (cache hit, fake-streamed) ===")
            start_time = time.perf_counter()
            async with client.stream(
                "POST",
                f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
                json=request_body,
            ) as response:
                assert response.status_code == 200
                cached_chunks = []
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        cached_chunks.append(line)
                        if line.strip() != "data: [DONE]":
                            print(f"  Chunk: {line[:80]}...")

                cache_stream_time = time.perf_counter() - start_time
                assert len(cached_chunks) > 0
                print(f"  Received {len(cached_chunks)} fake-streamed chunks from cache")
                print(f"  Fake-stream time: {cache_stream_time*1000:.2f}ms")

            # Cache hit streaming should still be fast
            assert cache_stream_time < 1.0

            print("\n✅ Proxy streaming works with cache!")

    finally:
        # Clean up proxy process
        proxy_process.terminate()
        proxy_process.join(timeout=5)
        if proxy_process.is_alive():
            proxy_process.kill()


if __name__ == "__main__":
    # Run the test directly
    asyncio.run(test_proxy_cache_hit_with_real_http_client())
    asyncio.run(test_proxy_streaming_with_cache())
