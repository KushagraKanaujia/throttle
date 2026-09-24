"""Tests for agent session tracking and profiling."""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

from starlette.datastructures import Headers

import pytest

from throttle.sessions import SessionTracker, Turn
from throttle.session_findings import analyze_session, Finding


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_sessions.db"


@pytest.fixture
def tracker(temp_db):
    """Create a SessionTracker with temporary database, closed on teardown.

    Tests that already awaited ``shutdown()`` are unaffected (``close()`` is a
    no-op once the connection is gone). Without this, every test that uses the
    fixture leaks an open sqlite3 connection, which surfaces under
    ``PYTHONWARNINGS=error`` as a ResourceWarning in whichever later test the
    garbage collector happens to run in.
    """
    t = SessionTracker(db_path=temp_db)
    yield t
    t.close()


def create_mock_request(session_header=None, client_ip="127.0.0.1"):
    """Create a mock FastAPI Request object."""
    request = MagicMock()
    request.client = MagicMock()
    request.client.host = client_ip
    # Real Starlette Headers so lookups are case-insensitive like production.
    request.headers = Headers({"x-throttle-session": session_header} if session_header else {})
    return request


def test_session_tracker_init(tracker, temp_db):
    """Test SessionTracker initialization creates database."""
    assert temp_db.exists()
    assert tracker.db_path == temp_db


@pytest.mark.asyncio
async def test_explicit_session_header(tracker):
    """Test session assignment with explicit X-Throttle-Session header."""
    request_body = {"messages": [{"role": "user", "content": "Hello"}], "model": "gpt-4"}
    response = {
        "choices": [{"message": {"content": "Hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }

    request = create_mock_request(session_header="my-session-123")

    await tracker.record_turn(
        request=request,
        request_body=request_body,
        response=response,
        latency_ms=100.0,
        ttft_ms=20.0,
    )

    # Flush pending turns
    await tracker._flush_batch()

    # Session ID should match header
    sessions = tracker.get_recent_sessions(since_seconds=60)
    assert [s["session_id"] for s in sessions] == ["my-session-123"]


@pytest.mark.asyncio
async def test_multiple_turns_same_session(tracker):
    """Test multiple turns are grouped into same session."""
    request = create_mock_request(session_header="session-abc")

    for i in range(5):
        request_body = {
            "messages": [
                {"role": "user", "content": f"Message {i}"}
            ],
            "model": "gpt-4",
        }
        response = {
            "choices": [{"message": {"content": f"Response {i}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10 + i, "completion_tokens": 5 + i},
        }

        await tracker.record_turn(
            request=request,
            request_body=request_body,
            response=response,
            latency_ms=100.0 + i * 10,
            ttft_ms=20.0,
        )

    await tracker._flush_batch()

    # Get sessions
    sessions = tracker.get_recent_sessions(since_seconds=60)

    assert [s["session_id"] for s in sessions] == ["session-abc"]
    assert sessions[0]["turn_count"] == 5
    turns = tracker.get_session_turns("session-abc")
    assert [t.turn_index for t in turns] == [0, 1, 2, 3, 4]


@pytest.mark.asyncio
async def test_turn_gap_calculation(tracker):
    """Gap = start of this turn minus end of the previous turn (client-side time)."""
    request = create_mock_request(session_header="gap-test")
    response = {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    t0 = 1_700_000_000.0
    # Turn 0: starts at t0, takes 100ms -> ends t0 + 0.1
    await tracker.record_turn(
        request=request,
        request_body={"messages": [{"role": "user", "content": "First"}]},
        response=response,
        latency_ms=100.0,
        started_at=t0,
    )
    # Turn 1: starts 2.6s after t0 -> gap is 2.5s
    await tracker.record_turn(
        request=request,
        request_body={"messages": [{"role": "user", "content": "Second"}]},
        response=response,
        latency_ms=120.0,
        started_at=t0 + 2.6,
    )
    await tracker._flush_batch()

    turns = tracker.get_session_turns("gap-test")
    assert [t.turn_index for t in turns] == [0, 1]
    assert turns[0].gap_since_previous_turn_seconds is None
    assert turns[1].gap_since_previous_turn_seconds == pytest.approx(2.5, abs=1e-6)

    metrics = tracker.compute_session_metrics("gap-test")
    # wall clock = t0 .. t0+2.6+0.12
    assert metrics.wall_clock_seconds == pytest.approx(2.72, abs=1e-6)
    assert metrics.generation_seconds == pytest.approx(0.22, abs=1e-6)
    assert metrics.idle_seconds == pytest.approx(2.5, abs=1e-6)


@pytest.mark.asyncio
async def test_turn_gap_inferred_from_wall_clock(tracker):
    """Without started_at, the start is inferred as (record time - latency)."""
    request = create_mock_request(session_header="gap-wall")
    response = {"choices": [{"message": {"content": "ok"}}], "usage": {}}
    body = {"messages": [{"role": "user", "content": "x"}]}

    await tracker.record_turn(request=request, request_body=body, response=response, latency_ms=100.0)
    await asyncio.sleep(0.5)
    await tracker.record_turn(request=request, request_body=body, response=response, latency_ms=120.0)
    await tracker._flush_batch()

    turns = tracker.get_session_turns("gap-wall")
    assert len(turns) == 2
    # ~0.5s sleep minus the 0.12s attributed to turn 1's own latency
    gap = turns[1].gap_since_previous_turn_seconds
    assert gap is not None
    assert 0.3 < gap < 1.5


@pytest.mark.asyncio
async def test_tool_call_detection(tracker):
    """Test detection of tool calls in responses."""
    request = create_mock_request(session_header="tool-test")

    request_body = {"messages": [{"role": "user", "content": "Use a tool"}], "model": "gpt-4"}
    response_with_tools = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [{"id": "call_123", "function": {"name": "get_weather"}}],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10},
    }

    await tracker.record_turn(
        request=request,
        request_body=request_body,
        response=response_with_tools,
        latency_ms=150.0,
        ttft_ms=30.0,
    )

    await tracker._flush_batch()

    turns = tracker.get_session_turns("tool-test")
    assert len(turns) == 1
    assert turns[0].contains_tool_calls is True
    assert turns[0].finish_reason == "tool_calls"
    # Empty content -> no completion hash
    assert turns[0].completion_hash is None


@pytest.mark.asyncio
async def test_tool_call_with_null_content(tracker):
    """OpenAI returns content: null alongside tool_calls; must not crash."""
    request = create_mock_request(session_header="tool-null")
    response = {
        "choices": [
            {
                "message": {"content": None, "tool_calls": [{"id": "c1"}]},
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10},
    }
    await tracker.record_turn(
        request=request,
        request_body={"messages": [{"role": "user", "content": "go"}]},
        response=response,
        latency_ms=10.0,
    )
    await tracker._flush_batch()
    turns = tracker.get_session_turns("tool-null")
    assert len(turns) == 1 and turns[0].contains_tool_calls is True


@pytest.mark.asyncio
async def test_session_metrics_computation(tracker):
    """Test computation of session metrics."""
    request = create_mock_request(session_header="metrics-test")

    # Create a session with multiple turns
    for i in range(3):
        request_body = {
            "messages": [{"role": "user", "content": f"Turn {i}"}],
            "model": "gpt-4",
        }
        response = {
            "choices": [{"message": {"content": f"Response {i}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }

        await tracker.record_turn(
            request=request,
            request_body=request_body,
            response=response,
            latency_ms=200.0,  # 0.2s per turn
            ttft_ms=50.0,
        )

        if i < 2:  # Add gap between turns
            await tracker._flush_batch()
            await asyncio.sleep(0.3)

    await tracker._flush_batch()

    sessions = tracker.get_recent_sessions(since_seconds=60)
    assert [s["session_id"] for s in sessions] == ["metrics-test"]
    metrics = tracker.compute_session_metrics("metrics-test")
    assert metrics is not None
    assert metrics.turn_count == 3
    assert metrics.total_prompt_tokens == 300
    assert metrics.total_completion_tokens == 150
    # Generation time is the sum of reported latencies: 3 turns * 0.2s
    assert metrics.generation_seconds == pytest.approx(0.6)
    # Two 0.3s sleeps, each minus the next turn's 0.2s latency (inferred start)
    assert metrics.idle_seconds >= 0.19
    assert metrics.generation_seconds + metrics.idle_seconds == pytest.approx(
        metrics.wall_clock_seconds, abs=0.05
    )


def test_findings_high_redundant_prefill():
    """Test finding generation for high redundant prefill."""
    from throttle.sessions import SessionMetrics

    metrics = SessionMetrics(
        session_id="test",
        turn_count=10,
        wall_clock_seconds=100.0,
        generation_seconds=20.0,
        generation_percent=20.0,
        idle_seconds=80.0,
        idle_percent=80.0,
        redundant_prefill_tokens=5000,
        redundant_percent=45.0,  # High redundancy
        total_prompt_tokens=11000,
        total_completion_tokens=2000,
        avg_ttft_ms=100.0,
        median_gap_seconds=8.0,
    )

    findings = analyze_session(metrics, [])

    # Should have a finding about prefix caching
    prefix_findings = [f for f in findings if f.category == "prefix_caching"]
    assert len(prefix_findings) > 0
    assert prefix_findings[0].severity == "high"
    assert "prefix caching" in prefix_findings[0].recommendation.lower()


def test_findings_high_idle_time():
    """Test finding generation for high idle time."""
    from throttle.sessions import SessionMetrics

    metrics = SessionMetrics(
        session_id="test",
        turn_count=5,
        wall_clock_seconds=100.0,
        generation_seconds=15.0,
        generation_percent=15.0,
        idle_seconds=85.0,
        idle_percent=85.0,  # Very high idle
        redundant_prefill_tokens=100,
        redundant_percent=5.0,
        total_prompt_tokens=2000,
        total_completion_tokens=1000,
        avg_ttft_ms=50.0,
        median_gap_seconds=17.0,
    )

    findings = analyze_session(metrics, [])

    # Should have a finding about utilization
    util_findings = [f for f in findings if f.category == "utilization"]
    assert len(util_findings) > 0
    assert util_findings[0].severity == "high"


@pytest.mark.asyncio
async def test_background_flush(tracker):
    """Test background flush task works correctly."""
    await tracker.start_background_flush()

    request = create_mock_request(session_header="flush-test")

    # Add a turn
    request_body = {"messages": [{"role": "user", "content": "Test"}], "model": "gpt-4"}
    response = {
        "choices": [{"message": {"content": "Response"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }

    await tracker.record_turn(
        request=request,
        request_body=request_body,
        response=response,
        latency_ms=100.0,
    )

    # Wait for background flush
    await asyncio.sleep(tracker.flush_interval_seconds + 1)

    # Should be flushed to database
    sessions = tracker.get_recent_sessions(since_seconds=60)
    assert len(sessions) > 0

    await tracker.shutdown()


@pytest.mark.asyncio
async def test_session_tracking_no_prompt_storage(tracker):
    """Test that prompt/completion text is never stored."""
    request = create_mock_request(session_header="privacy-test")

    secret_prompt = "My secret API key is sk-1234567890"
    secret_response = "Here's your password: hunter2"

    request_body = {
        "messages": [{"role": "user", "content": secret_prompt}],
        "model": "gpt-4",
    }
    response = {
        "choices": [{"message": {"content": secret_response}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10},
    }

    await tracker.record_turn(
        request=request,
        request_body=request_body,
        response=response,
        latency_ms=100.0,
    )

    await tracker._flush_batch()

    # Read database directly to verify no text storage
    with tracker._lock:
        row = tracker._conn.execute("SELECT prompt_hash, completion_hash FROM turns").fetchone()
        assert row is not None
        assert row["prompt_hash"] is not None
        assert row["completion_hash"] is not None

        # Materialize every column of every row (sqlite3.Row's repr hides values)
        dumped = []
        for table in ("turns", "sessions"):
            for r in tracker._conn.execute(f"SELECT * FROM {table}").fetchall():
                dumped.append(repr(tuple(r)))
        all_data = "\n".join(dumped)

    assert "sk-1234567890" not in all_data
    assert "hunter2" not in all_data
    # Raw client IP is not stored either (session IDs use a hashed client key)
    assert "127.0.0.1" not in all_data

    # Also check the raw bytes on disk, including the WAL file
    await tracker.shutdown()
    for path in tracker.db_path.parent.glob(tracker.db_path.name + "*"):
        raw = path.read_bytes()
        assert b"sk-1234567890" not in raw
        assert b"hunter2" not in raw


def _resp(prompt_tokens=100, content="ok"):
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 5},
    }


@pytest.mark.asyncio
async def test_growing_conversation_grouped_without_header(tracker):
    """An agent that resends the full history each turn is one session."""
    request = create_mock_request()
    history = [{"role": "system", "content": "sys " * 100}]
    for i in range(4):
        history = history + [{"role": "user", "content": f"q{i}"}]
        await tracker.record_turn(
            request=request,
            request_body={"messages": history},
            response=_resp(prompt_tokens=100 + 20 * i),
            latency_ms=10.0,
        )
        history = history + [{"role": "assistant", "content": f"a{i}"}]
    await tracker._flush_batch()

    sessions = tracker.get_recent_sessions(since_seconds=60)
    assert len(sessions) == 1
    assert sessions[0]["turn_count"] == 4
    turns = tracker.get_session_turns(sessions[0]["session_id"])
    assert turns[0].prefix_overlap_tokens == 0
    for t in turns[1:]:
        # Most of each later prompt repeats the previous one, but the estimate
        # can never exceed the backend-reported prompt tokens.
        assert 0.5 < t.prefix_overlap_percent < 1.0
        assert 0 < t.prefix_overlap_tokens <= t.prompt_tokens
    metrics = tracker.compute_session_metrics(sessions[0]["session_id"])
    assert metrics.redundant_prefill_tokens <= metrics.total_prompt_tokens
    assert 0 < metrics.redundant_percent <= 100


@pytest.mark.asyncio
async def test_unrelated_conversations_and_clients_are_separate(tracker):
    body_a = {"messages": [{"role": "user", "content": "conversation A"}]}
    body_b = {"messages": [{"role": "user", "content": "conversation B"}]}
    await tracker.record_turn(create_mock_request(client_ip="10.0.0.1"), body_a, _resp(), 10.0)
    await tracker.record_turn(create_mock_request(client_ip="10.0.0.1"), body_b, _resp(), 10.0)
    # Same messages as A but from a different client -> different session
    await tracker.record_turn(create_mock_request(client_ip="10.0.0.2"), body_a, _resp(), 10.0)
    await tracker._flush_batch()

    sessions = tracker.get_recent_sessions(since_seconds=60)
    assert len(sessions) == 3
    assert all("10.0.0" not in s["session_id"] for s in sessions)


@pytest.mark.asyncio
async def test_full_batch_flushes_without_deadlock(temp_db):
    """Reaching batch_size inside record_turn must flush, not deadlock."""
    tracker = SessionTracker(db_path=temp_db, batch_size=3)
    request = create_mock_request(session_header="batch")

    async def run():
        for i in range(7):
            await tracker.record_turn(
                request, {"messages": [{"role": "user", "content": str(i)}]}, _resp(), 5.0
            )

    await asyncio.wait_for(run(), timeout=10)
    # 2 full batches written, 1 turn still pending
    assert len(tracker.get_session_turns("batch")) == 6
    await tracker.shutdown()


@pytest.mark.asyncio
async def test_explicit_session_resumes_after_restart(temp_db):
    """Turn numbering continues across a proxy restart for explicit session IDs."""
    request = create_mock_request(session_header="long-running")
    body = {"messages": [{"role": "user", "content": "x"}]}

    first = SessionTracker(db_path=temp_db)
    await first.record_turn(request, body, _resp(), 10.0, started_at=1000.0)
    await first.record_turn(request, body, _resp(), 10.0, started_at=1001.0)
    await first.shutdown()

    second = SessionTracker(db_path=temp_db)
    await second.record_turn(request, body, _resp(), 10.0, started_at=1005.0)
    await second.shutdown()

    reader = SessionTracker(db_path=temp_db)
    turns = reader.get_session_turns("long-running")
    reader.close()
    assert [t.turn_index for t in turns] == [0, 1, 2]
    assert turns[2].gap_since_previous_turn_seconds == pytest.approx(1005.0 - 1001.01)


@pytest.mark.asyncio
async def test_proxy_records_sessions_end_to_end(temp_db):
    """Requests through the real proxy app land in the session store."""
    import httpx

    from throttle.proxy import ProxyServer

    def backend(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_resp(prompt_tokens=42, content="pong"))

    proxy = ProxyServer(
        "http://backend.invalid",
        enable_cache=False,
        enable_session_tracking=True,
        session_db_path=str(temp_db),
    )
    await proxy.startup()
    await proxy._client.aclose()
    proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(backend))
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app), base_url="http://proxy"
        ) as client:
            for i in range(3):
                r = await client.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": f"ping {i}"}]},
                    headers={"X-Throttle-Session": "e2e"},
                )
                assert r.status_code == 200
                assert r.json()["choices"][0]["message"]["content"] == "pong"
    finally:
        await proxy.shutdown()  # waits for background record tasks and flushes

    reader = SessionTracker(db_path=temp_db)
    turns = reader.get_session_turns("e2e")
    reader.close()
    assert [t.turn_index for t in turns] == [0, 1, 2]
    assert all(t.prompt_tokens == 42 for t in turns)


def _sessions_args(**kw):
    import argparse

    ns = argparse.Namespace(session_id=None, since="24h", limit=50, db=None)
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def test_sessions_cli_empty_store_does_not_create_db(tmp_path, capsys):
    from throttle.sessions_cli import handle_sessions_command

    db = tmp_path / "missing" / "sessions.db"
    assert handle_sessions_command(_sessions_args(db=str(db))) == 0
    out = capsys.readouterr().out
    assert "No sessions found in the last 24h 0m." in out
    assert "--enable-session-tracking" in out
    assert not db.exists()


def test_sessions_cli_invalid_since(tmp_path, capsys):
    from throttle.sessions_cli import handle_sessions_command

    assert handle_sessions_command(_sessions_args(db=str(tmp_path / "s.db"), since="abc")) == 2
    assert "Invalid time window" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_sessions_cli_populated_store(temp_db, capsys):
    from throttle.sessions_cli import handle_sessions_command

    tracker = SessionTracker(db_path=temp_db)
    for sid in ("alpha-session-1", "beta-session-2"):
        req = create_mock_request(session_header=sid)
        for i in range(2):
            await tracker.record_turn(
                req, {"messages": [{"role": "user", "content": str(i)}]}, _resp(), 100.0
            )
    await tracker.shutdown()

    assert handle_sessions_command(_sessions_args(db=str(temp_db))) == 0
    out = capsys.readouterr().out
    assert "Recent Sessions:" in out
    # Full IDs are shown so they can be passed back to `throttle sessions <id>`
    assert "alpha-session-1" in out and "beta-session-2" in out

    # Detail view by unique prefix
    assert handle_sessions_command(_sessions_args(db=str(temp_db), session_id="alpha")) == 0
    out = capsys.readouterr().out
    assert "Session: alpha-session-1" in out
    assert "Turn-by-Turn Timeline" in out

    # Ambiguous prefix and unknown ID are errors, not crashes
    assert handle_sessions_command(_sessions_args(db=str(temp_db), session_id="")) in (0, 1)
    capsys.readouterr()
    assert handle_sessions_command(_sessions_args(db=str(temp_db), session_id="zzz")) == 1


def test_format_duration():
    """Test duration formatting helper."""
    from throttle.sessions_cli import format_duration

    assert "5.0s" in format_duration(5.0)
    assert "1m" in format_duration(65.0)
    assert "1h" in format_duration(3700.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
