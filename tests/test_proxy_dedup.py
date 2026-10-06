"""Tests for proxy in-flight request deduplication without requiring live backend."""

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from throttle.proxy import ProxyServer

# Each server gets a fresh free port, so a server left over from an earlier
# test (or another process) can't make the next bind fail.
PORTS: Dict[str, int] = {}


def _bind(name: str) -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        PORTS[name] = sock.getsockname()[1]
    return PORTS[name]


class MockBackend:
    """Mock inference backend for testing deduplication behavior."""

    def __init__(self, delay_seconds: float = 0.0):
        self.delay_seconds = delay_seconds
        self.request_count = 0
        self.requests_received: List[Dict] = []
        self._lock = asyncio.Lock()
        self._should_error = False
        self._should_timeout = False
        self._timeout_duration = 30.0

    def configure_error(self):
        """Configure backend to return errors."""
        self._should_error = True

    def configure_timeout(self, duration: float):
        """Configure backend to timeout."""
        self._should_timeout = True
        self._timeout_duration = duration

    async def handle_request(self, request: Request):
        """Handle a chat completion request."""
        async with self._lock:
            self.request_count += 1
            request_num = self.request_count

        body = await request.json()
        self.requests_received.append({
            "request_num": request_num,
            "body": body,
            "timestamp": time.time(),
        })

        # Artificial delay to make concurrency observable
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)

        # Simulate timeout
        if self._should_timeout:
            await asyncio.sleep(self._timeout_duration)

        # Simulate error
        if self._should_error:
            return JSONResponse(
                status_code=500,
                content={"error": "Mock backend error"}
            )

        # Return unique identifiable response
        return JSONResponse({
            "id": f"mock-response-{request_num}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mock-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"Mock response #{request_num}"
                },
                "finish_reason": "stop"
            }]
        })


@pytest.fixture
async def mock_backend_server():
    """Fixture providing a mock backend server."""
    import uvicorn

    mock = MockBackend(delay_seconds=0.1)  # 100ms delay makes concurrency observable
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        return await mock.handle_request(request)

    config = uvicorn.Config(app, host="127.0.0.1", port=_bind("backend"), log_level="error")
    server = uvicorn.Server(config)

    # Start server in background
    task = asyncio.create_task(server.serve())
    await asyncio.sleep(1)  # Wait for server to start

    yield mock

    # Shutdown
    server.should_exit = True
    await task


@pytest.fixture
async def proxy_to_mock(mock_backend_server):
    """Fixture providing a proxy pointing to mock backend."""
    import uvicorn

    # ProxyServer will be captured by the lifespan closure
    proxy = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        await proxy.startup()
        yield
        # Shutdown
        await proxy.shutdown()

    proxy = ProxyServer(
        backend_url=f"http://127.0.0.1:{PORTS['backend']}",
        enable_cache=True,
        cache_ttl_seconds=3600.0,
        cache_max_size=100,
        cache_similarity_threshold=0.85,
        lifespan=lifespan,
    )

    app = proxy.app

    config = uvicorn.Config(app, host="127.0.0.1", port=_bind("proxy"), log_level="error")
    server = uvicorn.Server(config)

    # Start server in background
    task = asyncio.create_task(server.serve())
    await asyncio.sleep(1)  # Wait for proxy to start

    yield proxy

    # Shutdown server and wait for lifespan cleanup
    server.should_exit = True
    await task


async def test_five_concurrent_identical_requests_one_backend_call(proxy_to_mock, mock_backend_server):
    """5 concurrent identical requests result in exactly 1 backend call and 5 byte-identical responses."""
    mock = mock_backend_server
    proxy = proxy_to_mock

    payload = {
        "model": "mock-model",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "stream": False,
    }

    async def send_request(client_id: int):
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"http://127.0.0.1:{PORTS['proxy']}/v1/chat/completions",
                json=payload,
            )
            return client_id, response.json()

    # Send 5 concurrent identical requests
    tasks = [send_request(i) for i in range(5)]
    results = await asyncio.gather(*tasks)

    # Assert exactly 1 backend call
    assert mock.request_count == 1, f"Expected 1 backend call, got {mock.request_count}"

    # Assert all 5 responses are byte-identical
    responses = [r[1] for r in results]
    first_content = responses[0]["choices"][0]["message"]["content"]
    for i, resp in enumerate(responses):
        content = resp["choices"][0]["message"]["content"]
        assert content == first_content, f"Response {i} differs from response 0"

    # Assert backend_calls counter matches mock's count
    async with httpx.AsyncClient() as health_client:
        health = await health_client.get(f"http://127.0.0.1:{PORTS['proxy']}/health")
        stats = health.json()["cache_stats"]
        assert stats["backend_calls"] == 1, f"Counter shows {stats['backend_calls']}, mock received {mock.request_count}"
        assert stats["misses"] == 5, f"Expected 5 cache misses (all 5 checked cache)"


async def test_ten_concurrent_paraphrases_current_behavior(proxy_to_mock, mock_backend_server):
    """10 concurrent paraphrases: assert CURRENT behavior (each reaches backend).

    This test documents current behavior where paraphrases are NOT deduplicated.
    When Promise Cache lands, this test should be updated to expect deduplication.
    """
    mock = mock_backend_server

    # 10 prompts: 2 groups of paraphrases
    prompts = [
        "How do I sort a Python list?",           # 0
        "How can I sort a list in Python?",       # 1 - paraphrase of 0
        "What's the way to sort a Python list?",  # 2 - paraphrase of 0
        "How to sort lists in Python?",           # 3 - paraphrase of 0
        "What is the capital of France?",         # 4
        "What's France's capital city?",          # 5 - paraphrase of 4
        "Tell me the capital of France",          # 6 - paraphrase of 4
        "How do I sort a Python list?",           # 7 - exact dup of 0
        "What is the capital of France?",         # 8 - exact dup of 4
        "Which city is the capital of France?",   # 9 - paraphrase of 4
    ]

    async def send_request(idx: int, prompt: str):
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"http://127.0.0.1:{PORTS['proxy']}/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                },
            )
            return idx, response.json()

    tasks = [send_request(i, prompts[i]) for i in range(10)]
    results = await asyncio.gather(*tasks)

    # CURRENT BEHAVIOR: Paraphrases are NOT deduplicated, only exact matches
    # Expected: 8 backend calls (10 prompts - 2 exact dups)
    # When Promise Cache lands, expect: 2 backend calls (one per semantic cluster)
    expected_backend_calls = 8
    assert mock.request_count == expected_backend_calls, \
        f"CURRENT behavior: Expected {expected_backend_calls} backend calls, got {mock.request_count}"

    # Group responses by content
    from collections import defaultdict
    response_groups = defaultdict(list)
    for idx, response in results:
        content = response["choices"][0]["message"]["content"]
        response_groups[content].append(idx)

    # CURRENT: Exact duplicates share responses, paraphrases don't
    # [0,7] should share (exact match)
    # [4,8] should share (exact match)
    assert 0 in response_groups[results[0][1]["choices"][0]["message"]["content"]]
    assert 7 in response_groups[results[0][1]["choices"][0]["message"]["content"]]
    assert 4 in response_groups[results[4][1]["choices"][0]["message"]["content"]]
    assert 8 in response_groups[results[4][1]["choices"][0]["message"]["content"]]

    # Assert backend_calls counter matches mock
    async with httpx.AsyncClient() as health_client:
        health = await health_client.get(f"http://127.0.0.1:{PORTS['proxy']}/health")
        stats = health.json()["cache_stats"]
        assert stats["backend_calls"] == expected_backend_calls


async def test_backend_error_propagates_to_all_waiters(mock_backend_server):
    """Backend error during in-flight request returns same HTTP status to all concurrent waiters.

    This test verifies that:
    1. The originator raises the exception after setting it on the future (per PR #16)
    2. All waiters receive the same HTTPStatusError with the correct status code
    3. No requests hang waiting for results

    This is load-bearing for the exception handling in proxy.py:465-467.
    """
    import uvicorn

    mock = mock_backend_server
    mock.configure_error()  # Returns HTTP 500

    # ProxyServer will be captured by the lifespan closure
    proxy = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        await proxy.startup()
        yield
        # Shutdown
        await proxy.shutdown()

    proxy = ProxyServer(
        backend_url=f"http://127.0.0.1:{PORTS['backend']}",
        enable_cache=True,
        lifespan=lifespan,
    )

    # Start proxy HTTP server on a free port
    config = uvicorn.Config(proxy.app, host="127.0.0.1", port=_bind("proxy2"), log_level="error")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())
    await asyncio.sleep(1)  # Wait for server to start

    try:
        payload = {
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Test error propagation"}],
            "stream": False,
        }

        async def send_request(client_id: int):
            """Send request through HTTP client and capture status code."""
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.post(
                        f"http://127.0.0.1:{PORTS['proxy2']}/v1/chat/completions",
                        json=payload,
                    )
                    response.raise_for_status()
                    return client_id, "success", response.status_code
            except httpx.HTTPStatusError as e:
                return client_id, "http_error", e.response.status_code
            except Exception as e:
                return client_id, "error", str(type(e).__name__)

        # Send 5 concurrent requests that will all encounter the error
        tasks = [send_request(i) for i in range(5)]
        results = await asyncio.gather(*tasks)

        # All 5 waiters should receive HTTP 500 (the backend's error status)
        assert len(results) == 5, "All 5 requests should complete (no hangs)"
        for client_id, status, code in results:
            assert status == "http_error", (
                f"Client {client_id} got {status} instead of http_error"
            )
            assert code == 500, (
                f"Client {client_id} should receive HTTP 500 from backend error, "
                f"got {code}"
            )

    finally:
        # Shutdown server
        server.should_exit = True
        await server_task


async def test_backend_timeout_does_not_hang_waiters(mock_backend_server):
    """Backend timeout during in-flight request doesn't leave waiters hanging."""
    import uvicorn

    mock = mock_backend_server
    mock.configure_timeout(duration=0.5)

    # ProxyServer will be captured by the lifespan closure
    proxy = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        await proxy.startup()
        yield
        # Shutdown
        await proxy.shutdown()

    proxy = ProxyServer(
        backend_url=f"http://127.0.0.1:{PORTS['backend']}",
        enable_cache=True,
        backend_timeout_seconds=0.3,  # Short timeout to trigger before mock's 0.5s
        lifespan=lifespan,
    )

    # Start proxy HTTP server on a free port
    config = uvicorn.Config(proxy.app, host="127.0.0.1", port=_bind("proxy"), log_level="error")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())
    await asyncio.sleep(1)  # Wait for server to start

    try:
        payload = {
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Test timeout"}],
            "stream": False,
        }

        async def send_request(client_id: int):
            try:
                # Client timeout longer than proxy's backend timeout
                async with httpx.AsyncClient(timeout=2.0) as client:
                    # Call through proxy's HTTP endpoint
                    response = await client.post(
                        f"http://127.0.0.1:{PORTS['proxy']}/v1/chat/completions",
                        json=payload,
                    )
                    response.raise_for_status()  # Raise exception for 4xx/5xx
                    return client_id, "success", response.status_code
            except httpx.TimeoutException:
                return client_id, "timeout", None
            except httpx.HTTPStatusError as e:
                return client_id, "http_error", e.response.status_code
            except Exception as e:
                return client_id, "error", str(type(e).__name__)

        # Send 5 concurrent identical requests
        tasks = [send_request(i) for i in range(5)]
        results = await asyncio.gather(*tasks)

        # All requests should get timeout errors without hanging
        # Proxy should return 504 Gateway Timeout when backend times out
        for client_id, status, code in results:
            assert status == "http_error", f"Client {client_id} got {status} (code={code}) instead of http_error"
            assert code == 504, f"Client {client_id} got status {code} instead of 504"

    finally:
        # Shutdown server
        server.should_exit = True
        await server_task


async def test_sequential_identical_requests_hit_cache(proxy_to_mock, mock_backend_server):
    """Sequential identical requests hit cache normally, confirming dedup didn't break ordinary path."""
    mock = mock_backend_server

    payload = {
        "model": "mock-model",
        "messages": [{"role": "user", "content": "Sequential test"}],
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        # First request - cache miss
        response1 = await client.post(
            f"http://127.0.0.1:{PORTS['proxy']}/v1/chat/completions",
            json=payload,
        )
        assert response1.status_code == 200
        content1 = response1.json()["choices"][0]["message"]["content"]

        # Small delay
        await asyncio.sleep(0.2)

        # Second request - should be cache hit
        response2 = await client.post(
            f"http://127.0.0.1:{PORTS['proxy']}/v1/chat/completions",
            json=payload,
        )
        assert response2.status_code == 200
        content2 = response2.json()["choices"][0]["message"]["content"]

        # Should be identical (from cache)
        assert content2 == content1

        # Only 1 backend call should have been made
        assert mock.request_count == 1, f"Expected 1 backend call, got {mock.request_count}"

        # Check cache stats
        health = await client.get(f"http://127.0.0.1:{PORTS['proxy']}/health")
        stats = health.json()["cache_stats"]
        assert stats["backend_calls"] == 1
        assert stats["hits"] == 1, "Second request should be cache hit"
        assert stats["misses"] == 1, "First request should be cache miss"


async def test_backend_calls_counter_matches_mock_count(proxy_to_mock, mock_backend_server):
    """backend_calls counter must match mock backend's actual received count."""
    mock = mock_backend_server

    # Send various requests
    async with httpx.AsyncClient(timeout=10.0) as client:
        # 3 different prompts = 3 backend calls
        for i in range(3):
            await client.post(
                f"http://127.0.0.1:{PORTS['proxy']}/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": f"Prompt {i}"}],
                    "stream": False,
                },
            )

        # Check agreement
        health = await client.get(f"http://127.0.0.1:{PORTS['proxy']}/health")
        stats = health.json()["cache_stats"]

        assert stats["backend_calls"] == mock.request_count, \
            f"Counter disagreement: counter={stats['backend_calls']}, mock={mock.request_count}"
        assert stats["backend_calls"] == 3


async def test_concurrent_requests_with_mixed_prompts(proxy_to_mock, mock_backend_server):
    """Mixed concurrent requests: some identical, some unique."""
    mock = mock_backend_server

    # 8 requests: 3 identical, 3 identical, 2 unique
    prompts = [
        "Prompt A",  # 0
        "Prompt A",  # 1 - dup
        "Prompt A",  # 2 - dup
        "Prompt B",  # 3
        "Prompt B",  # 4 - dup
        "Prompt B",  # 5 - dup
        "Prompt C",  # 6 - unique
        "Prompt D",  # 7 - unique
    ]

    async def send_request(idx: int):
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"http://127.0.0.1:{PORTS['proxy']}/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": prompts[idx]}],
                    "stream": False,
                },
            )
            return idx, response.json()

    tasks = [send_request(i) for i in range(8)]
    results = await asyncio.gather(*tasks)

    # Expected: 4 backend calls (A, B, C, D)
    assert mock.request_count == 4, f"Expected 4 backend calls, got {mock.request_count}"

    # Verify responses
    # Requests 0,1,2 should be identical
    content_0 = results[0][1]["choices"][0]["message"]["content"]
    content_1 = results[1][1]["choices"][0]["message"]["content"]
    content_2 = results[2][1]["choices"][0]["message"]["content"]
    assert content_0 == content_1 == content_2

    # Requests 3,4,5 should be identical
    content_3 = results[3][1]["choices"][0]["message"]["content"]
    content_4 = results[4][1]["choices"][0]["message"]["content"]
    content_5 = results[5][1]["choices"][0]["message"]["content"]
    assert content_3 == content_4 == content_5

    # Counter should match
    async with httpx.AsyncClient() as health_client:
        health = await health_client.get(f"http://127.0.0.1:{PORTS['proxy']}/health")
        stats = health.json()["cache_stats"]
        assert stats["backend_calls"] == 4
        assert stats["backend_calls"] == mock.request_count


async def test_cross_scope_hit_inflation_bug_regression(proxy_to_mock, mock_backend_server):
    """Regression test: cross-scope requests should not inflate hit counter.

    Before fix, metrics.hits was incremented when finding a similar prompt,
    before scope validation. This caused model B to report a cache hit even
    though it missed and hit the backend.

    This test verifies hits counter only increments on true hits (semantic + scope match).
    """
    mock = mock_backend_server
    messages = [{"role": "user", "content": "Say hello"}]

    async with httpx.AsyncClient(timeout=10.0) as client:
        # Prime cache with model A
        await client.post(
            f"http://127.0.0.1:{PORTS['proxy']}/v1/chat/completions",
            json={"model": "model-a", "messages": messages, "stream": False},
        )

        # Get health after priming
        health = await client.get(f"http://127.0.0.1:{PORTS['proxy']}/health")
        stats_after_a = health.json()["cache_stats"]
        assert stats_after_a["hits"] == 0, "First request should not be a hit"
        assert stats_after_a["misses"] == 1, "First request should be a miss"
        assert stats_after_a["backend_calls"] == 1

        # Request with model B (same messages, different scope)
        await client.post(
            f"http://127.0.0.1:{PORTS['proxy']}/v1/chat/completions",
            json={"model": "model-b", "messages": messages, "stream": False},
        )

        # Get health after model B
        health = await client.get(f"http://127.0.0.1:{PORTS['proxy']}/health")
        stats_after_b = health.json()["cache_stats"]

        # REGRESSION: Before fix, hits would be 1 (false positive from similarity before scope check)
        # CORRECT: hits should still be 0 (similar prompt but different scope = miss)
        assert stats_after_b["hits"] == 0, \
            "Model B should not increment hits (similar prompt, different scope)"
        assert stats_after_b["misses"] == 2, \
            "Model B should increment misses (scope mismatch)"
        assert stats_after_b["backend_calls"] == 2, \
            "Model B should hit backend (cache miss)"

        # Verify mock actually received 2 distinct requests
        assert mock.request_count == 2, "Both requests should reach backend"
