"""Test TTFT capture in proxy streaming path."""
import asyncio
import json
import time
import unittest
from unittest.mock import AsyncMock, MagicMock
from contextlib import asynccontextmanager

from throttle.proxy import ProxyServer


class TestStreamingTTFT(unittest.IsolatedAsyncioTestCase):
    """_make_backend_request must capture real TTFT from SSE."""

    async def test_ttft_captured_on_first_nonempty_delta(self):
        """First chunk has empty content, second has text after a short delay.
        ttft_ms must be not-None and less than total elapsed time."""
        proxy = ProxyServer(backend_url="http://localhost:9999", enable_cache=False)

        sse_lines = [
            "data: " + json.dumps({
                "id": "x", "model": "m", "created": 1,
                "choices": [{"delta": {"content": ""}, "finish_reason": None}],
            }),
            "data: " + json.dumps({
                "id": "x", "model": "m", "created": 1,
                "choices": [{"delta": {"content": "Hello"}, "finish_reason": None}],
            }),
            "data: " + json.dumps({
                "id": "x", "model": "m", "created": 1,
                "choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}],
            }),
            "data: [DONE]",
        ]

        async def fake_aiter_lines():
            for line in sse_lines:
                if "Hello" in line:
                    await asyncio.sleep(0.05)
                yield line

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = fake_aiter_lines

        @asynccontextmanager
        async def mock_stream(method, url, **kwargs):
            yield mock_response

        mock_client = MagicMock()
        mock_client.stream = mock_stream
        proxy._client = mock_client

        request_body = {
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
        }

        t0 = time.perf_counter()
        response_data, ttft_ms = await proxy._make_backend_request(
            request_body, {}, "hi", is_streaming=True
        )
        total_ms = (time.perf_counter() - t0) * 1000

        self.assertIsNotNone(ttft_ms, "ttft_ms must not be None for streaming")
        self.assertGreater(ttft_ms, 0, "ttft_ms must be positive")
        self.assertLess(ttft_ms, total_ms, "ttft_ms must be less than total time")
        self.assertEqual(
            response_data["choices"][0]["message"]["content"], "Hello world"
        )

    async def test_ttft_none_for_non_streaming(self):
        """Non-streaming requests must return ttft_ms=None."""
        proxy = ProxyServer(backend_url="http://localhost:9999", enable_cache=False)

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "id": "x", "choices": [{"message": {"content": "hi"}}],
        }

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        proxy._client = mock_client

        request_body = {
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
        }

        response_data, ttft_ms = await proxy._make_backend_request(
            request_body, {}, "hi", is_streaming=False
        )

        self.assertIsNone(ttft_ms, "non-streaming must return ttft_ms=None")
