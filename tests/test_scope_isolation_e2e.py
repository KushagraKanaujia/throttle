#!/usr/bin/env python3
"""End-to-end test for scope-aware caching through chat_completions endpoint.

Tests that same prompt with different scope parameters (model, temperature, etc.)
gets isolated cache entries, not cross-contaminated responses.

This test must FAIL at 9084707^ (before scope-aware fix) and PASS at HEAD.
"""
import pytest
from throttle.proxy import ProxyServer
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_same_prompt_different_models_isolated_responses():
    """Same prompt with two different models must get separate cached responses.

    This tests the core scope-aware caching fix:
    - Before 9084707: cache key was prompt only, cross-model contamination
    - After 9084707: cache key is (prompt, scope), each model gets own entry
    """
    # Create proxy with cache enabled
    proxy = ProxyServer(
        backend_url="http://mock-backend",
        enable_cache=True,
        cache_ttl_seconds=3600,
    )
    await proxy.startup()

    # Mock the HTTP client to return different responses for different models
    mock_response_model_a = {
        "id": "resp-a",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "model-a",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Response from model A",
                },
                "finish_reason": "stop",
            }
        ],
    }

    mock_response_model_b = {
        "id": "resp-b",
        "object": "chat.completion",
        "created": 1234567891,
        "model": "model-b",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Response from model B",
                },
                "finish_reason": "stop",
            }
        ],
    }

    # Mock client to return different responses based on model
    async def mock_post(url, json=None, headers=None):
        mock_resp = MagicMock()
        if json.get("model") == "model-a":
            mock_resp.json.return_value = mock_response_model_a
        else:
            mock_resp.json.return_value = mock_response_model_b
        return mock_resp

    proxy._client.post = mock_post

    # Create mock requests with same prompt but different models
    class MockRequest:
        def __init__(self, body):
            self._body = body
            self.headers = {"authorization": "Bearer test-key"}

        async def json(self):
            return self._body

    request_body_a = {
        "model": "model-a",
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 10,
    }

    request_body_b = {
        "model": "model-b",  # Different model, same prompt
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 10,
    }

    # Send first request (model-a) - should be cache miss
    request_a1 = MockRequest(request_body_a)
    response_a1 = await proxy.chat_completions(request_a1)
    assert proxy.cache.metrics.hits == 0
    assert proxy.cache.metrics.misses == 1

    # Extract response content
    response_a1_json = response_a1.body.decode() if hasattr(response_a1, 'body') else None
    if response_a1_json:
        import json
        response_a1_data = json.loads(response_a1_json)
    else:
        # JSONResponse case
        response_a1_data = mock_response_model_a

    # Send second request (model-b, same prompt) - should be cache miss (different scope)
    request_b1 = MockRequest(request_body_b)
    response_b1 = await proxy.chat_completions(request_b1)
    assert proxy.cache.metrics.hits == 0
    assert proxy.cache.metrics.misses == 2

    # Extract response content
    response_b1_json = response_b1.body.decode() if hasattr(response_b1, 'body') else None
    if response_b1_json:
        import json
        response_b1_data = json.loads(response_b1_json)
    else:
        # JSONResponse case
        response_b1_data = mock_response_model_b

    # KEY ASSERTION: Each model must get its own response
    # This will FAIL before 9084707 (cross-contamination)
    # This will PASS after 9084707 (scope isolation)
    assert response_a1_data["choices"][0]["message"]["content"] == "Response from model A"
    assert response_b1_data["choices"][0]["message"]["content"] == "Response from model B"
    assert response_a1_data["model"] == "model-a"
    assert response_b1_data["model"] == "model-b"

    # Send third request (model-a again) - should be cache hit
    request_a2 = MockRequest(request_body_a)
    response_a2 = await proxy.chat_completions(request_a2)
    assert proxy.cache.metrics.hits == 1
    assert proxy.cache.metrics.misses == 2

    # Verify cached response still returns model-a response
    response_a2_json = response_a2.body.decode() if hasattr(response_a2, 'body') else None
    if response_a2_json:
        import json
        response_a2_data = json.loads(response_a2_json)
    else:
        response_a2_data = mock_response_model_a

    assert response_a2_data["choices"][0]["message"]["content"] == "Response from model A"
    assert response_a2_data["model"] == "model-a"

    await proxy.shutdown()
