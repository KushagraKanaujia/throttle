"""Performance tests for session tracking overhead.

The proxy calls SessionTracker.record_turn() once per completed request (as a
background task). These tests bound the cost of that call and of batch
flushing. Thresholds are wall-clock based; they have large headroom on a
developer laptop (see the printed numbers with -s) but could in principle be
exceeded on a heavily loaded CI machine.
"""

import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from starlette.datastructures import Headers

from throttle.sessions import SessionTracker


def create_mock_request(session_header=None, client_ip="127.0.0.1"):
    """Create a mock FastAPI Request object."""
    request = MagicMock()
    request.client = MagicMock()
    request.client.host = client_ip
    request.headers = Headers({"x-throttle-session": session_header} if session_header else {})
    return request


def _p95(samples):
    ordered = sorted(samples)
    return ordered[int(len(ordered) * 0.95)]


@pytest.mark.asyncio
async def test_session_tracking_overhead_under_5ms():
    """record_turn() must cost < 5ms at p95, including amortized batch flushes.

    Uses a realistic growing agent conversation (each turn resends the full
    history, which is what session grouping and prefix-overlap work on) and a
    batch size small enough that flushes to SQLite happen inside the measured
    window.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = SessionTracker(db_path=Path(tmpdir) / "test_sessions.db", batch_size=20)
        await tracker.start_background_flush()
        request = create_mock_request()

        history = [{"role": "system", "content": "You are a coding agent. " * 50}]
        samples_ms = []
        for i in range(200):
            history = history + [
                {"role": "user", "content": f"step {i}: " + "context " * 40},
            ]
            response = {
                "choices": [{"message": {"content": f"ok {i}"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100 + 60 * i, "completion_tokens": 8},
            }
            start = time.perf_counter()
            await tracker.record_turn(
                request=request,
                request_body={"model": "gpt-4", "messages": history},
                response=response,
                latency_ms=50.0,
                ttft_ms=10.0,
            )
            samples_ms.append((time.perf_counter() - start) * 1000)
            history = history + [{"role": "assistant", "content": f"ok {i}"}]

        await tracker.shutdown()

        avg = sum(samples_ms) / len(samples_ms)
        p95 = _p95(samples_ms)
        print("\nSession Tracking Overhead (record_turn):")
        print(f"  Average: {avg:.3f}ms  P95: {p95:.3f}ms  Max: {max(samples_ms):.3f}ms")

        # The whole growing conversation must have been grouped into one session
        verify = SessionTracker(db_path=Path(tmpdir) / "test_sessions.db")
        sessions = verify.get_recent_sessions(since_seconds=3600)
        verify.close()
        assert len(sessions) == 1
        assert sessions[0]["turn_count"] == 200

        assert p95 < 5.0, f"P95 overhead {p95:.3f}ms exceeds 5ms limit"


@pytest.mark.asyncio
async def test_batch_flush_performance():
    """Batch flushing must sustain well over 500 recorded turns/sec."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_sessions.db"
        tracker = SessionTracker(db_path=db_path, batch_size=100)
        await tracker.start_background_flush()
        request = create_mock_request()

        async def run():
            for i in range(1000):
                await tracker.record_turn(
                    request=request,
                    request_body={
                        "messages": [{"role": "user", "content": f"Message {i}"}],
                        "model": "gpt-4",
                    },
                    response={
                        "choices": [
                            {"message": {"content": f"Response {i}"}, "finish_reason": "stop"}
                        ],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                    },
                    latency_ms=50.0,
                )
            await tracker._flush_batch()

        start = time.perf_counter()
        # A full batch triggers a flush from inside record_turn; this used to
        # deadlock on the tracker's non-reentrant lock, so bound it.
        await asyncio.wait_for(run(), timeout=30)
        elapsed = time.perf_counter() - start
        throughput = 1000 / elapsed

        print("\nBatch Flush Performance:")
        print(f"  Total time: {elapsed:.3f}s  Throughput: {throughput:.0f} turns/sec")

        with tracker._lock:
            written = tracker._conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
        assert written == 1000

        assert throughput > 500, f"Throughput {throughput:.0f} req/s is too low"

        await tracker.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
