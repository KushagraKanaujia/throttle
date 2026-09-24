"""Tests for ONNX embedding tier through proxy endpoint.

All tests assert through the proxy endpoint on responses received,
never on internal structures like _store or _embedder.
"""

import asyncio
import pytest
from contextlib import asynccontextmanager

from throttle.proxy import ProxyServer
from throttle import embeddings


def _embedding_model_skip_reason():
    if not embeddings.EMBEDDINGS_AVAILABLE:
        return "embeddings extra not installed (pip install throttle-pro[embeddings])"
    if not embeddings.is_model_available():
        return (
            f"embedding model could not be loaded: {embeddings.get_load_error()} "
            "(needs network access or a cached sentence-transformers/all-MiniLM-L6-v2)"
        )
    return None


# Tests asserting on embedding-tier behavior need the real ONNX model. Without
# it the cache falls back to Jaccard-only and these assertions cannot hold.
# The model is probed lazily (first use of the fixture), not at import time.
@pytest.fixture
def require_embedding_model():
    reason = _embedding_model_skip_reason()
    if reason:
        pytest.skip(reason)


async def _wait_for_server(host: str, port: int, timeout: float = 5.0):
    """Poll TCP connection until server accepts connections or timeout."""
    import socket
    import time

    start = time.time()
    interval = 0.05  # 50ms between attempts

    while time.time() - start < timeout:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.settimeout(0.5)
            sock.connect((host, port))
            return  # Success
        except (ConnectionRefusedError, OSError):
            await asyncio.sleep(interval)
        finally:
            sock.close()

    raise TimeoutError(f"Server at {host}:{port} did not accept connections within {timeout}s")


@asynccontextmanager
async def _fake_backend():
    """Mock backend that returns responses encoding scope parameters."""
    async def handler(request):
        import json
        body = await request.json()
        prompt_text = " ".join(m["content"] for m in body.get("messages", []))

        # Encode scope in response content so tests can validate correct scope
        model = body.get("model", "unknown-model")
        temperature = body.get("temperature", "no-temperature")

        # Response content encodes the scope it was called with
        content = f"Response from model={model} temperature={temperature} to prompt: {prompt_text[:50]}"

        return {
            "id": f"chatcmpl-{hash(content)}",
            "object": "chat.completion",
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
        }

    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    import uvicorn
    import socket

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        response = await handler(request)
        return JSONResponse(response)

    # Bind socket to loopback with auto-assigned port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)

    task = asyncio.create_task(server.serve(sockets=[sock]))
    await _wait_for_server("127.0.0.1", port)

    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await task


@pytest.mark.asyncio
async def test_embedding_hit_that_jaccard_misses(require_embedding_model):
    """Test a: Semantically similar prompts under identical scope produce embedding hit."""

    @asynccontextmanager
    async def lifespan(app):
        async with _fake_backend() as backend_url:
            proxy.backend_url = backend_url
            await proxy.startup()
            yield
        await proxy.shutdown()

    proxy = ProxyServer(
        backend_url="http://placeholder",
        enable_cache=True,
        enable_embeddings=True,
        cache_max_size=10,
        lifespan=lifespan,
    )

    async with lifespan(proxy.app):
        from httpx import AsyncClient
        import socket

        # Start proxy on real port
        import uvicorn

        # Bind socket to loopback with auto-assigned port
        proxy_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        proxy_sock.bind(("127.0.0.1", 0))
        proxy_port = proxy_sock.getsockname()[1]

        proxy_config = uvicorn.Config(proxy.app, host="127.0.0.1", port=proxy_port, log_level="error")
        proxy_server = uvicorn.Server(proxy_config)
        proxy_task = asyncio.create_task(proxy_server.serve(sockets=[proxy_sock]))
        await _wait_for_server("127.0.0.1", proxy_port)

        proxy_url = f"http://127.0.0.1:{proxy_port}"

        try:
            async with AsyncClient() as client:
                # Pair: cosine=0.9762, jaccard=0.1667 (measured 2026-08-23 with direct ONNX)
                resp1 = await client.post(
                    f"{proxy_url}/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "What are the benefits of using Docker containers?"}],
                        "temperature": 0.7,
                    },
                )
                assert resp1.status_code == 200
                response1 = resp1.json()

                # Second request: strong paraphrase (cosine > 0.95, jaccard < 0.85)
                resp2 = await client.post(
                    f"{proxy_url}/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "What advantages do Docker containers provide?"}],
                        "temperature": 0.7,
                    },
                )
                assert resp2.status_code == 200
                response2 = resp2.json()

                # Should be cache hit via embedding tier
                assert proxy.cache.metrics.hits >= 1
                assert proxy.cache.metrics.embedding_hits >= 1
                # Responses should match (same cached content)
                assert response2["choices"][0]["message"]["content"] == response1["choices"][0]["message"]["content"]
        finally:
            proxy_server.should_exit = True
            await proxy_task


@pytest.mark.asyncio
async def test_weak_paraphrase_does_not_hit_at_threshold_095(require_embedding_model):
    """Test that PostgreSQL pair (cosine=0.878) correctly misses at threshold 0.95."""

    @asynccontextmanager
    async def lifespan(app):
        async with _fake_backend() as backend_url:
            proxy.backend_url = backend_url
            await proxy.startup()
            yield
        await proxy.shutdown()

    proxy = ProxyServer(
        backend_url="http://placeholder",
        enable_cache=True,
        enable_embeddings=True,
        cache_max_size=10,
        lifespan=lifespan,
    )

    async with lifespan(proxy.app):
        from httpx import AsyncClient
        import uvicorn
        import socket

        # Bind socket to loopback with auto-assigned port
        proxy_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        proxy_sock.bind(("127.0.0.1", 0))
        proxy_port = proxy_sock.getsockname()[1]

        proxy_config = uvicorn.Config(proxy.app, host="127.0.0.1", port=proxy_port, log_level="error")
        proxy_server = uvicorn.Server(proxy_config)
        proxy_task = asyncio.create_task(proxy_server.serve(sockets=[proxy_sock]))
        await _wait_for_server("127.0.0.1", proxy_port)

        proxy_url = f"http://127.0.0.1:{proxy_port}"

        try:
            async with AsyncClient() as client:
                # Pair: cosine=0.8783, jaccard<0.85 (measured 2026-08-23 with direct ONNX)
                resp1 = await client.post(
                    f"{proxy_url}/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "How do I optimize PostgreSQL queries?"}],
                        "temperature": 0.7,
                    },
                )
                assert resp1.status_code == 200
                content1 = resp1.json()["choices"][0]["message"]["content"]

                # Weak paraphrase - should NOT hit at threshold 0.95
                resp2 = await client.post(
                    f"{proxy_url}/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "How can I improve database query performance in PostgreSQL?"}],
                        "temperature": 0.7,
                    },
                )
                assert resp2.status_code == 200
                content2 = resp2.json()["choices"][0]["message"]["content"]

                # CRITICAL: Responses must be DIFFERENT (not cached)
                assert content2 != content1, \
                    f"Weak paraphrase incorrectly served cached response: {content2!r}"
                # Metrics assertions (supplementary)
                assert proxy.cache.metrics.misses >= 1
                assert proxy.cache.metrics.embedding_hits == 0
                assert proxy.cache.metrics.embedding_scans_attempted >= 1
        finally:
            proxy_server.should_exit = True
            await proxy_task


@pytest.mark.asyncio
async def test_embedding_miss_on_different_model():
    """Test b: Same prompts under different model values return scope-correct responses."""

    @asynccontextmanager
    async def lifespan(app):
        async with _fake_backend() as backend_url:
            proxy.backend_url = backend_url
            await proxy.startup()
            yield
        await proxy.shutdown()

    proxy = ProxyServer(
        backend_url="http://placeholder",
        enable_cache=True,
        enable_embeddings=True,
        cache_max_size=10,
        lifespan=lifespan,
    )

    async with lifespan(proxy.app):
        from httpx import AsyncClient
        import uvicorn
        import socket

        # Bind socket to loopback with auto-assigned port
        proxy_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        proxy_sock.bind(("127.0.0.1", 0))
        proxy_port = proxy_sock.getsockname()[1]

        proxy_config = uvicorn.Config(proxy.app, host="127.0.0.1", port=proxy_port, log_level="error")
        proxy_server = uvicorn.Server(proxy_config)
        proxy_task = asyncio.create_task(proxy_server.serve(sockets=[proxy_sock]))
        await _wait_for_server("127.0.0.1", proxy_port)

        proxy_url = f"http://127.0.0.1:{proxy_port}"

        try:
            async with AsyncClient() as client:
                # Request under scope A (model-a)
                resp1 = await client.post(
                    f"{proxy_url}/v1/chat/completions",
                    json={
                        "model": "model-a",
                        "messages": [{"role": "user", "content": "What is machine learning?"}],
                    },
                )
                assert resp1.status_code == 200
                content1 = resp1.json()["choices"][0]["message"]["content"]
                # Verify response encodes scope A
                assert "model=model-a" in content1

                # Request under scope B (model-b) with identical prompt
                resp2 = await client.post(
                    f"{proxy_url}/v1/chat/completions",
                    json={
                        "model": "model-b",
                        "messages": [{"role": "user", "content": "What is machine learning?"}],
                    },
                )
                assert resp2.status_code == 200
                content2 = resp2.json()["choices"][0]["message"]["content"]

                # CRITICAL: Response must match scope B, not scope A
                assert "model=model-b" in content2, \
                    f"Cross-scope contamination: got {content2!r}, expected model=model-b"
                assert content2 != content1, \
                    "Responses from different scopes must differ"
        finally:
            proxy_server.should_exit = True
            await proxy_task


@pytest.mark.asyncio
async def test_embedding_miss_on_different_temperature():
    """Test c: Same prompts under different temperature values return scope-correct responses."""

    @asynccontextmanager
    async def lifespan(app):
        async with _fake_backend() as backend_url:
            proxy.backend_url = backend_url
            await proxy.startup()
            yield
        await proxy.shutdown()

    proxy = ProxyServer(
        backend_url="http://placeholder",
        enable_cache=True,
        enable_embeddings=True,
        cache_max_size=10,
        lifespan=lifespan,
    )

    async with lifespan(proxy.app):
        from httpx import AsyncClient
        import uvicorn
        import socket

        # Bind socket to loopback with auto-assigned port
        proxy_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        proxy_sock.bind(("127.0.0.1", 0))
        proxy_port = proxy_sock.getsockname()[1]

        proxy_config = uvicorn.Config(proxy.app, host="127.0.0.1", port=proxy_port, log_level="error")
        proxy_server = uvicorn.Server(proxy_config)
        proxy_task = asyncio.create_task(proxy_server.serve(sockets=[proxy_sock]))
        await _wait_for_server("127.0.0.1", proxy_port)

        proxy_url = f"http://127.0.0.1:{proxy_port}"

        try:
            async with AsyncClient() as client:
                # Request under scope A (temperature=0.7)
                resp1 = await client.post(
                    f"{proxy_url}/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "Explain quantum computing"}],
                        "temperature": 0.7,
                    },
                )
                assert resp1.status_code == 200
                content1 = resp1.json()["choices"][0]["message"]["content"]
                # Verify response encodes scope A
                assert "temperature=0.7" in content1

                # Request under scope B (temperature=0.9) with identical prompt
                resp2 = await client.post(
                    f"{proxy_url}/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "Explain quantum computing"}],
                        "temperature": 0.9,
                    },
                )
                assert resp2.status_code == 200
                content2 = resp2.json()["choices"][0]["message"]["content"]

                # CRITICAL: Response must match scope B, not scope A
                assert "temperature=0.9" in content2, \
                    f"Cross-scope contamination: got {content2!r}, expected temperature=0.9"
                assert content2 != content1, \
                    "Responses from different scopes must differ"
        finally:
            proxy_server.should_exit = True
            await proxy_task


@pytest.mark.asyncio
async def test_exact_match_hits_via_dict_lookup():
    """Test d: Exact match hits via O(1) dict lookup, not Jaccard."""

    @asynccontextmanager
    async def lifespan(app):
        async with _fake_backend() as backend_url:
            proxy.backend_url = backend_url
            await proxy.startup()
            yield
        await proxy.shutdown()

    proxy = ProxyServer(
        backend_url="http://placeholder",
        enable_cache=True,
        enable_embeddings=True,
        cache_max_size=10,
        lifespan=lifespan,
    )

    async with lifespan(proxy.app):
        from httpx import AsyncClient
        import uvicorn
        import socket

        # Bind socket to loopback with auto-assigned port
        proxy_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        proxy_sock.bind(("127.0.0.1", 0))
        proxy_port = proxy_sock.getsockname()[1]

        proxy_config = uvicorn.Config(proxy.app, host="127.0.0.1", port=proxy_port, log_level="error")
        proxy_server = uvicorn.Server(proxy_config)
        proxy_task = asyncio.create_task(proxy_server.serve(sockets=[proxy_sock]))
        await _wait_for_server("127.0.0.1", proxy_port)

        proxy_url = f"http://127.0.0.1:{proxy_port}"

        try:
            async with AsyncClient() as client:
                resp1 = await client.post(
                    f"{proxy_url}/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "Test prompt"}],
                    },
                )
                assert resp1.status_code == 200
                content1 = resp1.json()["choices"][0]["message"]["content"]

                # Identical request should hit cache
                resp2 = await client.post(
                    f"{proxy_url}/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "Test prompt"}],
                    },
                )
                assert resp2.status_code == 200
                content2 = resp2.json()["choices"][0]["message"]["content"]

                # CRITICAL: Second response must match first (cached)
                assert content2 == content1, \
                    f"Cache miss on exact match: got {content2!r}, expected {content1!r}"
                # Metrics assertions (supplementary)
                assert proxy.cache.metrics.hits >= 1
                assert proxy.cache.metrics.exact_hits >= 1
        finally:
            proxy_server.should_exit = True
            await proxy_task


@pytest.mark.asyncio
async def test_jaccard_fuzzy_match_hits():
    """Test e: Jaccard fuzzy match on prompts that differ but score >= 0.85."""

    @asynccontextmanager
    async def lifespan(app):
        async with _fake_backend() as backend_url:
            proxy.backend_url = backend_url
            await proxy.startup()
            yield
        await proxy.shutdown()

    proxy = ProxyServer(
        backend_url="http://placeholder",
        enable_cache=True,
        enable_embeddings=False,  # Disable embeddings to test Jaccard only
        cache_max_size=10,
        cache_similarity_threshold=0.85,
        lifespan=lifespan,
    )

    async with lifespan(proxy.app):
        from httpx import AsyncClient
        import uvicorn
        import socket

        # Bind socket to loopback with auto-assigned port
        proxy_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        proxy_sock.bind(("127.0.0.1", 0))
        proxy_port = proxy_sock.getsockname()[1]

        proxy_config = uvicorn.Config(proxy.app, host="127.0.0.1", port=proxy_port, log_level="error")
        proxy_server = uvicorn.Server(proxy_config)
        proxy_task = asyncio.create_task(proxy_server.serve(sockets=[proxy_sock]))
        await _wait_for_server("127.0.0.1", proxy_port)

        proxy_url = f"http://127.0.0.1:{proxy_port}"

        try:
            async with AsyncClient() as client:
                # First request with prompt A
                resp1 = await client.post(
                    f"{proxy_url}/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "How to use Python pip command line"}],
                        "temperature": 0.7,
                    },
                )
                assert resp1.status_code == 200
                content1 = resp1.json()["choices"][0]["message"]["content"]

                # Second request with prompt B - differs but Jaccard >= 0.85
                # Prompt A: {how, do, i, install, python, packages, using, pip} = 8 words
                # Prompt B: {how, do, i, install, python, packages, with, pip} = 8 words
                # Intersection: {how, do, i, install, python, packages, pip} = 7 words
                # Union: {how, do, i, install, python, packages, using, with, pip} = 9 words
                # Jaccard: 7/9 = 0.78 < 0.85... need simpler
                # Actually: "How do I install Python using pip" vs "How do I install Python with pip"
                # A: {how, do, i, install, python, using, pip} = 7 words
                # B: {how, do, i, install, python, with, pip} = 7 words
                # Intersection: {how, do, i, install, python, pip} = 6 words
                # Union: {how, do, i, install, python, using, with, pip} = 8 words
                # Jaccard: 6/8 = 0.75 still not enough
                # Try: "How to install Python packages" vs "How to install Python package"
                # A: {how, to, install, python, packages} = 5 words
                # B: {how, to, install, python, package} = 5 words
                # Intersection: {how, to, install, python} = 4 words (packages != package)
                # Union: {how, to, install, python, packages, package} = 6 words
                # Jaccard: 4/6 = 0.67 still too low
                # Best: same words, different order won't help with Jaccard
                # Try: "Install Python packages using pip command" vs "Install Python packages using pip"
                # A: {install, python, packages, using, pip, command} = 6 words
                # B: {install, python, packages, using, pip} = 5 words
                # Intersection: {install, python, packages, using, pip} = 5 words
                # Union: {install, python, packages, using, pip, command} = 6 words
                # Jaccard: 5/6 = 0.833 still < 0.85
                # Try: "How do I use Python pip" vs "How do I use Python"
                # A: {how, do, i, use, python, pip} = 6 words
                # B: {how, do, i, use, python} = 5 words
                # Intersection: {how, do, i, use, python} = 5 words
                # Union: {how, do, i, use, python, pip} = 6 words
                # Jaccard: 5/6 = 0.833
                # Final try: "How to use Python pip command line" vs "How to use Python pip command"
                # A: {how, to, use, python, pip, command, line} = 7 words
                # B: {how, to, use, python, pip, command} = 6 words
                # Intersection: {how, to, use, python, pip, command} = 6 words
                # Union: {how, to, use, python, pip, command, line} = 7 words
                # Jaccard: 6/7 = 0.857 >= 0.85 ✓
                resp2 = await client.post(
                    f"{proxy_url}/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "How to use Python pip command"}],
                        "temperature": 0.7,
                    },
                )
                assert resp2.status_code == 200
                content2 = resp2.json()["choices"][0]["message"]["content"]

                # CRITICAL: Should serve cached response from first request
                assert content2 == content1, \
                    f"Jaccard match should serve cached response, got {content2!r}, expected {content1!r}"
                # Metrics assertions (supplementary)
                assert proxy.cache.metrics.hits >= 1
                assert proxy.cache.metrics.lexical_hits >= 1
                assert proxy.cache.metrics.exact_hits == 0, "Should not be exact match"
        finally:
            proxy_server.should_exit = True
            await proxy_task


@pytest.mark.asyncio
async def test_embedding_miss_on_cross_scope():
    """Test e: Semantically similar prompts under different scopes return scope-correct responses."""

    @asynccontextmanager
    async def lifespan(app):
        async with _fake_backend() as backend_url:
            proxy.backend_url = backend_url
            await proxy.startup()
            yield
        await proxy.shutdown()

    proxy = ProxyServer(
        backend_url="http://placeholder",
        enable_cache=True,
        enable_embeddings=True,
        cache_max_size=10,
        lifespan=lifespan,
    )

    async with lifespan(proxy.app):
        from httpx import AsyncClient
        import uvicorn
        import socket

        # Bind socket to loopback with auto-assigned port
        proxy_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        proxy_sock.bind(("127.0.0.1", 0))
        proxy_port = proxy_sock.getsockname()[1]

        proxy_config = uvicorn.Config(proxy.app, host="127.0.0.1", port=proxy_port, log_level="error")
        proxy_server = uvicorn.Server(proxy_config)
        proxy_task = asyncio.create_task(proxy_server.serve(sockets=[proxy_sock]))
        await _wait_for_server("127.0.0.1", proxy_port)

        proxy_url = f"http://127.0.0.1:{proxy_port}"

        try:
            async with AsyncClient() as client:
                # Request under scope A (model-a) - use Docker pair that triggers embedding hits
                resp1 = await client.post(
                    f"{proxy_url}/v1/chat/completions",
                    json={
                        "model": "model-a",
                        "messages": [{"role": "user", "content": "What are the benefits of using Docker containers?"}],
                    },
                )
                assert resp1.status_code == 200
                content1 = resp1.json()["choices"][0]["message"]["content"]
                # Verify response encodes scope A
                assert "model=model-a" in content1

                # Request under scope B (model-b) with semantically similar prompt
                resp2 = await client.post(
                    f"{proxy_url}/v1/chat/completions",
                    json={
                        "model": "model-b",
                        "messages": [{"role": "user", "content": "What advantages do Docker containers provide?"}],
                    },
                )
                assert resp2.status_code == 200
                content2 = resp2.json()["choices"][0]["message"]["content"]

                # CRITICAL: Response must match scope B, not scope A
                assert "model=model-b" in content2, \
                    f"Cross-scope contamination: got {content2!r}, expected model=model-b"
                assert content2 != content1, \
                    "Responses from different scopes must differ"
        finally:
            proxy_server.should_exit = True
            await proxy_task

@pytest.mark.asyncio
async def test_embedding_hit_with_multiple_scopes_returns_requesting_scope(require_embedding_model):
    """Test: Embedding hit with cached entries under multiple scopes returns requesting scope's response.

    Scenario: Two scopes (model-a, model-b) both cache responses to near-identical prompts.
    The responses differ only by scope encoding. A third request under model-a with a
    paraphrase should hit via embeddings and return model-a's cached response, not model-b's.
    """

    @asynccontextmanager
    async def lifespan(app):
        async with _fake_backend() as backend_url:
            proxy.backend_url = backend_url
            await proxy.startup()
            yield
        await proxy.shutdown()

    proxy = ProxyServer(
        backend_url="http://placeholder",
        enable_cache=True,
        enable_embeddings=True,
        cache_max_size=10,
        lifespan=lifespan,
    )

    async with lifespan(proxy.app):
        from httpx import AsyncClient
        import uvicorn
        import socket

        # Bind socket to loopback with auto-assigned port
        proxy_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        proxy_sock.bind(("127.0.0.1", 0))
        proxy_port = proxy_sock.getsockname()[1]

        proxy_config = uvicorn.Config(proxy.app, host="127.0.0.1", port=proxy_port, log_level="error")
        proxy_server = uvicorn.Server(proxy_config)
        proxy_task = asyncio.create_task(proxy_server.serve(sockets=[proxy_sock]))
        await _wait_for_server("127.0.0.1", proxy_port)

        proxy_url = f"http://127.0.0.1:{proxy_port}"

        try:
            async with AsyncClient() as client:
                # Populate cache under scope A (model-a)
                resp1 = await client.post(
                    f"{proxy_url}/v1/chat/completions",
                    json={
                        "model": "model-a",
                        "messages": [{"role": "user", "content": "What are the benefits of using Docker containers?"}],
                    },
                )
                assert resp1.status_code == 200
                content_a = resp1.json()["choices"][0]["message"]["content"]
                assert "model=model-a" in content_a

                # Populate cache under scope B (model-b) with IDENTICAL prompt
                resp2 = await client.post(
                    f"{proxy_url}/v1/chat/completions",
                    json={
                        "model": "model-b",
                        "messages": [{"role": "user", "content": "What are the benefits of using Docker containers?"}],
                    },
                )
                assert resp2.status_code == 200
                content_b = resp2.json()["choices"][0]["message"]["content"]
                assert "model=model-b" in content_b

                # Now request under scope A with paraphrase - should hit via embeddings and return scope A response
                resp3 = await client.post(
                    f"{proxy_url}/v1/chat/completions",
                    json={
                        "model": "model-a",
                        "messages": [{"role": "user", "content": "What advantages do Docker containers provide?"}],
                    },
                )
                assert resp3.status_code == 200
                content3 = resp3.json()["choices"][0]["message"]["content"]

                # CRITICAL: Must return scope A's cached response, not scope B's
                assert "model=model-a" in content3, \
                    f"Cross-scope contamination: got {content3!r}, expected model=model-a"
                assert content3 == content_a, \
                    f"Embedding hit should return cached response from requesting scope, got {content3!r}"
                assert proxy.cache.metrics.embedding_hits >= 1, \
                    "Third request should have hit via embedding tier"
        finally:
            proxy_server.should_exit = True
            await proxy_task
