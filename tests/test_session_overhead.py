"""Performance tests for session tracking overhead.

Verifies that session tracking adds <5ms overhead per request.
"""

import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from throttle.proxy import ProxyServer


def create_mock_request(session_header=None, client_ip="127.0.0.1"):
    """Create a mock FastAPI Request object."""
    request = MagicMock()
    request.client = MagicMock()
    request.client.host = client_ip
    request.headers = {"X-Throttle-Session": session_header} if session_header else {}
    request.headers.get = lambda k, default=None: request.headers.get(k, default)
    return request


@pytest.mark.asyncio
async def test_session_tracking_overhead_under_5ms():
    """Verify session tracking adds less than 5ms overhead per request.

    This is a critical performance requirement for production use.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_sessions.db"

        # Create proxy WITHOUT session tracking
        proxy_no_tracking = ProxyServer(
            backend_url="http://localhost:8000",
            enable_cache=False,
            enable_session_tracking=False,
        )

        # Create proxy WITH session tracking
        from throttle.sessions import SessionTracker

        tracker = SessionTracker(db_path=db_path)
        await tracker.start_background_flush()

        proxy_with_tracking = ProxyServer(
            backend_url="http://localhost:8000",
            enable_cache=False,
            enable_session_tracking=True,
        )
        # Replace with our temp tracker
        proxy_with_tracking._session_tracker = tracker

        # Mock request/response
        request = create_mock_request()
        request_body = {
            "messages": [{"role": "user", "content": "Hello world"}],
            "model": "gpt-4",
        }
        response = {
            "choices": [{"message": {"content": "Hi there!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 15, "completion_tokens": 8},
        }

        # Measure overhead 100 times for statistical significance
        overhead_samples = []

        for i in range(100):
            # Baseline: no tracking
            start = time.perf_counter()
            # Simulate just the session tracking part (not full request)
            # We're measuring the overhead of record_turn() call itself
            baseline_time = time.perf_counter() - start

            # With tracking
            start = time.perf_counter()
            await tracker.record_turn(
                request=request,
                request_body=request_body,
                response=response,
                latency_ms=50.0,
                ttft_ms=10.0,
            )
            tracked_time = time.perf_counter() - start

            overhead_ms = (tracked_time - baseline_time) * 1000
            overhead_samples.append(overhead_ms)

        # Compute statistics
        avg_overhead = sum(overhead_samples) / len(overhead_samples)
        max_overhead = max(overhead_samples)
        p95_overhead = sorted(overhead_samples)[int(len(overhead_samples) * 0.95)]

        print(f"\nSession Tracking Overhead:")
        print(f"  Average: {avg_overhead:.3f}ms")
        print(f"  Max: {max_overhead:.3f}ms")
        print(f"  P95: {p95_overhead:.3f}ms")

        # Assert that overhead is under 5ms
        # Using p95 to be conservative (not just average)
        assert (
            p95_overhead < 5.0
        ), f"P95 overhead {p95_overhead:.3f}ms exceeds 5ms limit"

        # Cleanup
        await tracker.shutdown()


@pytest.mark.asyncio
async def test_batch_flush_performance():
    """Verify batch flushing can handle high throughput."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_sessions.db"
        tracker = SessionTracker(db_path=db_path, batch_size=100)

        await tracker.start_background_flush()

        request = create_mock_request()

        # Simulate high throughput: 1000 requests
        start = time.perf_counter()

        for i in range(1000):
            request_body = {
                "messages": [{"role": "user", "content": f"Message {i}"}],
                "model": "gpt-4",
            }
            response = {
                "choices": [{"message": {"content": f"Response {i}"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }

            await tracker.record_turn(
                request=request,
                request_body=request_body,
                response=response,
                latency_ms=50.0,
            )

        # Force flush
        await tracker._flush_batch()

        elapsed = time.perf_counter() - start
        throughput = 1000 / elapsed

        print(f"\nBatch Flush Performance:")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Throughput: {throughput:.0f} requests/sec")

        # Should handle at least 500 requests/sec
        assert throughput > 500, f"Throughput {throughput:.0f} req/s is too low"

        await tracker.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
