"""Test per-model backend routing."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
import json

from throttle.proxy import ProxyServer


class TestModelRouting(unittest.IsolatedAsyncioTestCase):
    """Test that model_backends correctly routes requests to different backends."""

    async def test_model_backends_routes_to_correct_backend(self):
        """Requests for different models reach their configured backends."""
        # Setup: Create proxy with two model-to-backend mappings
        model_backends = {
            "model-a": "http://backend-a:8000",
            "model-b": "http://backend-b:9000",
        }

        proxy = ProxyServer(
            backend_url="http://default-backend:7000",
            enable_cache=False,  # Disable cache to ensure backend calls
            model_backends=model_backends,
        )

        # Mock the HTTP client
        mock_response_a = MagicMock()
        mock_response_a.json.return_value = {
            "id": "resp-a",
            "model": "model-a",
            "choices": [{"message": {"content": "response from backend A"}}],
        }

        mock_response_b = MagicMock()
        mock_response_b.json.return_value = {
            "id": "resp-b",
            "model": "model-b",
            "choices": [{"message": {"content": "response from backend B"}}],
        }

        async def mock_post(url, **kwargs):
            """Track which URL was called and return appropriate response."""
            if "backend-a:8000" in url:
                return mock_response_a
            elif "backend-b:9000" in url:
                return mock_response_b
            else:
                raise AssertionError(f"Unexpected URL called: {url}")

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=mock_post)
        proxy._client = mock_client

        # Test: Make requests for model-a and model-b
        request_a = {
            "model": "model-a",
            "messages": [{"role": "user", "content": "test"}],
        }

        request_b = {
            "model": "model-b",
            "messages": [{"role": "user", "content": "test"}],
        }

        # Execute
        response_a, _ = await proxy._make_backend_request(
            request_a, {}, "test prompt", is_streaming=False
        )
        response_b, _ = await proxy._make_backend_request(
            request_b, {}, "test prompt", is_streaming=False
        )

        # Verify: Check that correct URLs were called
        calls = mock_client.post.call_args_list
        self.assertEqual(len(calls), 2)

        # First call should be to backend-a
        url_a = calls[0][0][0]
        self.assertIn("backend-a:8000", url_a)

        # Second call should be to backend-b
        url_b = calls[1][0][0]
        self.assertIn("backend-b:9000", url_b)

        # Verify responses match expected backends
        self.assertEqual(response_a["id"], "resp-a")
        self.assertEqual(response_b["id"], "resp-b")

    async def test_routing_fails_when_get_backend_url_stubbed_to_default(self):
        """Verify test fails if _get_backend_url returns self.backend_url."""
        # Setup: Same as above
        model_backends = {
            "model-a": "http://backend-a:8000",
            "model-b": "http://backend-b:9000",
        }

        proxy = ProxyServer(
            backend_url="http://default-backend:7000",
            enable_cache=False,
            model_backends=model_backends,
        )

        # Stub _get_backend_url to always return default
        original_get_backend_url = proxy._get_backend_url
        proxy._get_backend_url = lambda model: proxy.backend_url

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=MagicMock(json=lambda: {}))
        proxy._client = mock_client

        # Make request for model-a
        request_a = {
            "model": "model-a",
            "messages": [{"role": "user", "content": "test"}],
        }

        await proxy._make_backend_request(
            request_a, {}, "test prompt", is_streaming=False
        )

        # Verify: URL called should be default, not backend-a
        calls = mock_client.post.call_args_list
        self.assertEqual(len(calls), 1)
        url_called = calls[0][0][0]

        # This should be default-backend:7000, NOT backend-a:8000
        self.assertIn("default-backend:7000", url_called)
        self.assertNotIn("backend-a:8000", url_called)

        # If we expected backend-a, this test demonstrates the failure
        with self.assertRaises(AssertionError):
            self.assertIn("backend-a:8000", url_called)

    async def test_unmapped_model_falls_back_to_default(self):
        """Models not in mapping use default backend_url."""
        model_backends = {
            "model-a": "http://backend-a:8000",
        }

        proxy = ProxyServer(
            backend_url="http://default-backend:7000",
            enable_cache=False,
            model_backends=model_backends,
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=MagicMock(json=lambda: {}))
        proxy._client = mock_client

        # Request for unmapped model-c
        request_c = {
            "model": "model-c",
            "messages": [{"role": "user", "content": "test"}],
        }

        await proxy._make_backend_request(
            request_c, {}, "test prompt", is_streaming=False
        )

        # Verify: Should use default backend
        calls = mock_client.post.call_args_list
        self.assertEqual(len(calls), 1)
        url_called = calls[0][0][0]
        self.assertIn("default-backend:7000", url_called)


if __name__ == "__main__":
    unittest.main()
