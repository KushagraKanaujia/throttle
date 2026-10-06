"""`throttle check --workload agent`: multi-turn synthetic agent sessions.

Drives the real CLI (`throttle.cli.main`) against a fake OpenAI-compatible
server (no GPU). The fake answers with a short deterministic reply and counts
usage tokens as words, so every request it saw can be inspected.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import httpx
import pytest

import throttle.check as check_module
import throttle.config as config_module
from throttle import agent_workload
from throttle.cli import main

URL = "http://127.0.0.1:8000"
TURNS = 3
SESSIONS = 2
BLOCKS = 3
AGENT_ARGS = (
    "--workload", "agent", "--turns", str(TURNS), "--sessions-concurrency", str(SESSIONS),
    "--system-prompt-tokens", "60", "--tool-output-tokens", "20",
)


class AgentFakeServer:
    """Records every chat request; replies are a deterministic function of the
    prompt, so equal prompts get equal replies (temperature 0)."""

    def __init__(
        self, delay: float = 0.005, *, reply_salt: str = "", cached_fraction: float | None = None
    ) -> None:
        self.delay = delay
        self.reply_salt = reply_salt
        self.cached_fraction = cached_fraction
        self.requests: list[list[dict[str, str]]] = []
        self.usage: list[tuple[int, int]] = []

    async def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": [
                {"id": "mock-model", "object": "model", "owned_by": "vllm"}]})
        if path == "/metrics":
            return httpx.Response(404, text="not found")
        if path == "/v1/chat/completions":
            body = json.loads(request.content)
            messages = body["messages"]
            self.requests.append(messages)
            await asyncio.sleep(self.delay)
            digest = hashlib.sha256(json.dumps(messages).encode()).hexdigest()[:8]
            reply = f"I will call the next tool now ({digest}).{self.reply_salt}"
            prompt_tokens = sum(len(m["content"].split()) for m in messages)
            completion_tokens = len(reply.split())
            self.usage.append((prompt_tokens, completion_tokens))
            usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                     "total_tokens": prompt_tokens + completion_tokens}
            if self.cached_fraction is not None:
                usage["prompt_tokens_details"] = {
                    "cached_tokens": int(prompt_tokens * self.cached_fraction)
                }
            return httpx.Response(200, json={
                "id": "x", "object": "chat.completion", "model": "mock-model",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": reply}}],
                "usage": usage,
            })
        return httpx.Response(404)

    def measured(self) -> list[list[dict[str, str]]]:
        """Measured requests only (the warm-up asks for 'OK')."""
        return [m for m in self.requests if "Warm-up" not in m[-1]["content"]]


@pytest.fixture
def history(tmp_path, monkeypatch) -> Path:
    directory = tmp_path / "history"
    monkeypatch.setenv(check_module.HISTORY_ENV, str(directory))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", tmp_path / "no-config.yaml")
    return directory


def run_check(monkeypatch, capsys, server, *extra: str) -> tuple[int, str]:
    monkeypatch.setattr(check_module, "_TRANSPORT", httpx.MockTransport(server.handle))
    code = main([
        "check", "--url", URL, "--model", "mock-model", "--gpu-hourly-rate", "2.00",
        "--blocks", str(BLOCKS), "--warmup", "1", "--max-tokens", "16", *extra,
    ])
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def records(history: Path) -> list[dict]:
    return [json.loads(line) for line in (history / "checks.ndjson").read_text().splitlines()]


def by_session(requests: list[list[dict[str, str]]]) -> dict[str, list[list[dict[str, str]]]]:
    sessions: dict[str, list[list[dict[str, str]]]] = {}
    for messages in requests:
        sessions.setdefault(messages[1]["content"], []).append(messages)
    return sessions


# --------------------------------------------------------------------------
# The generator
# --------------------------------------------------------------------------


def _simulate(shape: agent_workload.AgentShape, run_id: str) -> list[list[dict[str, str]]]:
    return [
        agent_workload.turn_messages(shape, run_id, session, turn)
        for session in range(3)
        for turn in range(shape.turns)
    ]


def test_same_seed_gives_the_same_request_sequence():
    shape = agent_workload.AgentShape(turns=4, system_prompt_tokens=200, tool_output_tokens=50)
    assert _simulate(shape, "123456") == _simulate(shape, "123456")
    other_seed = agent_workload.AgentShape(
        turns=4, system_prompt_tokens=200, tool_output_tokens=50, seed=1
    )
    assert _simulate(other_seed, "123456") != _simulate(shape, "123456")


def test_sizes_follow_the_shape():
    small = agent_workload.AgentShape(system_prompt_tokens=100, tool_output_tokens=10)
    large = agent_workload.AgentShape(system_prompt_tokens=1500, tool_output_tokens=300)
    assert len(agent_workload.system_prompt(large)) > 10 * len(agent_workload.system_prompt(small))
    words = agent_workload.tool_message(large, 0, 0).split()
    assert 300 <= len(words) <= 310


# --------------------------------------------------------------------------
# End to end against the fake server
# --------------------------------------------------------------------------


def test_prefix_grows_within_each_session(history, monkeypatch, capsys):
    server = AgentFakeServer()
    code, out = run_check(monkeypatch, capsys, server, *AGENT_ARGS)
    assert code == 0, out
    measured = server.measured()
    assert len(measured) == BLOCKS * SESSIONS * TURNS
    assert len(server.requests) == len(measured) + 1  # one warm-up
    sessions = by_session(measured)
    assert len(sessions) == BLOCKS * SESSIONS  # every session is distinct
    for turns in sessions.values():
        assert len(turns) == TURNS
        for previous, current in zip(turns, turns[1:]):
            # Turn k's prompt starts with turn k-1's prompt, then adds the
            # model's reply and a tool result.
            assert current[: len(previous)] == previous
            assert len(current) == len(previous) + 2
            assert current[len(previous)]["role"] == "assistant"
            assert current[-1]["content"].startswith("Tool result (")
    recorded = records(history)[-1]["workload"]["workload_shape"]
    by_turn = recorded["mean_prompt_tokens_by_turn"]
    assert by_turn == sorted(by_turn) and by_turn[0] < by_turn[-1]


def test_prompts_do_not_depend_on_the_model_replies(history, monkeypatch, capsys):
    """Two runs whose server answers differently send identical prompts (apart
    from the run tag): the history is scripted, so reply drift cannot compound."""

    sent = []
    usage = []
    for salt in ("", " And a much longer, different answer this time around."):
        server = AgentFakeServer(reply_salt=salt)
        code, out = run_check(monkeypatch, capsys, server, *AGENT_ARGS)
        assert code == 0, out
        run_id = records(history)[-1]["workload"]["prompt_nonce"]["run_id"]
        tag = f"[run {run_id}] "
        sent.append([
            [dict(m, content=m["content"].replace(tag, "")) for m in messages]
            for messages in server.requests
        ])
        usage.append([u[0] for u in server.usage])
    assert sent[0] == sent[1]
    assert usage[0] == usage[1]  # identical prompt tokens per request
    shapes = [r["workload"]["workload_shape"] for r in records(history)]
    assert shapes[0]["mean_prompt_tokens_by_turn"] == shapes[1]["mean_prompt_tokens_by_turn"]
    assert shapes[0]["mean_completion_tokens_per_turn"] < shapes[1]["mean_completion_tokens_per_turn"]
    # The scripted assistant turns are what later prompts carry.
    for messages in server.requests:
        for message in messages:
            if message["role"] == "assistant":
                assert message["content"].startswith("Step ")


def test_cached_prompt_tokens_are_recorded_when_reported(history, monkeypatch, capsys):
    code, out = run_check(
        monkeypatch, capsys, AgentFakeServer(cached_fraction=0.5), *AGENT_ARGS
    )
    assert code == 0, out
    first = records(history)[-1]
    shape = first["workload"]["workload_shape"]
    assert shape["mean_cached_prompt_tokens_per_turn"] == pytest.approx(
        shape["mean_prompt_tokens_per_turn"] * 0.5, rel=0.05
    )
    assert "served from the prefix cache [MEASURED" in out
    # Measured only: a check whose server reports no cached tokens is the same workload.
    code, out = run_check(monkeypatch, capsys, AgentFakeServer(), *AGENT_ARGS)
    assert "the workload differs" not in out


def test_run_tag_is_constant_within_a_run_and_differs_across_runs(history, monkeypatch, capsys):
    systems = []
    run_ids = []
    for _ in range(2):
        server = AgentFakeServer()
        code, out = run_check(monkeypatch, capsys, server, *AGENT_ARGS)
        assert code == 0, out
        within = {m[0]["content"] for m in server.requests}  # warm-up included
        assert len(within) == 1
        system = within.pop()
        run_id = records(history)[-1]["workload"]["prompt_nonce"]["run_id"]
        assert system.startswith(f"[run {run_id}] You are an autonomous software agent")
        systems.append(system)
        run_ids.append(run_id)
    # The same system prompt apart from the tag, so the workload is the same.
    assert systems[0] != systems[1]
    assert systems[0].split("] ", 1)[1] == systems[1].split("] ", 1)[1]
    first, second = records(history)
    assert first["workload"]["prompts_sha256"] == second["workload"]["prompts_sha256"]


def test_workload_shape_is_recorded(history, monkeypatch, capsys):
    server = AgentFakeServer()
    code, out = run_check(monkeypatch, capsys, server, *AGENT_ARGS)
    assert code == 0, out
    record = records(history)[-1]
    workload = record["workload"]
    shape = workload["workload_shape"]
    assert {k: shape[k] for k in agent_workload.SHAPE_IDENTITY_KEYS} == {
        "profile": "agent", "turns": TURNS, "sessions_concurrency": SESSIONS,
        "system_prompt_tokens": 60, "tool_output_tokens": 20,
        "assistant_turn_tokens": agent_workload.DEFAULT_ASSISTANT_TURN_TOKENS,
        "history": "scripted",
    }
    # The fake reports no cached tokens, so none are claimed.
    assert shape["mean_cached_prompt_tokens_per_turn"] is None
    measured_usage = server.usage[1:]  # drop the warm-up
    assert shape["mean_prompt_tokens_per_turn"] == pytest.approx(
        sum(u[0] for u in measured_usage) / len(measured_usage)
    )
    assert shape["mean_completion_tokens_per_turn"] == pytest.approx(
        sum(u[1] for u in measured_usage) / len(measured_usage)
    )
    assert len(shape["mean_prompt_tokens_by_turn"]) == TURNS
    assert workload["requests_per_block"] == SESSIONS * TURNS
    assert workload["concurrency"] == SESSIONS
    assert workload["sent_prompts_sha256"]
    assert sum(b["input_tokens"] for b in record["blocks"]) == sum(u[0] for u in measured_usage)
    assert "shape   " in out and "prompt tokens by turn:" in out
    assert "AGENT: 3 blocks x 2 concurrent sessions x 3 sequential turns" in out
    # History and share show the shape too.
    assert main(["check", "--history"]) == 0
    assert "workload  agent: 3 turns x 2 concurrent sessions" in capsys.readouterr().out
    summary = check_module.build_share_summary(record, None)
    assert "| Workload shape | agent: 3 turns x 2 concurrent sessions" in summary


def test_agent_and_default_checks_are_never_compared(history, monkeypatch, capsys):
    code, out = run_check(monkeypatch, capsys, AgentFakeServer())
    assert code == 0, out
    code, out = run_check(
        monkeypatch, capsys, AgentFakeServer(), *AGENT_ARGS, "--fail-if-costlier", "0"
    )
    assert "Verdict: NO WINNER, the workload differs (" in out
    assert "workload_shape" in out
    assert "The workload profile changed (default -> agent: 3 turns" in out
    assert "change  not computed" in out
    assert code == check_module.EXIT_NOT_CALIBRATED, out
    # Nor are two agent checks of different shapes.
    previous, current = records(history)[-1], dict(records(history)[-1])
    current["workload"] = dict(current["workload"])
    current["workload"]["workload_shape"] = dict(
        current["workload"]["workload_shape"], turns=TURNS + 1
    )
    comparison = check_module.compare_checks(previous, current)
    assert comparison["workload_differs"] == ["workload_shape"]
    assert comparison["delta_dollars_per_million"] is None


def test_agent_options_need_the_agent_workload(history, monkeypatch, capsys):
    server = AgentFakeServer()
    code, out = run_check(monkeypatch, capsys, server, "--turns", "4")
    assert code == check_module.EXIT_USAGE
    assert "--turns only applies with --workload agent" in out
    code, out = run_check(monkeypatch, capsys, server, *AGENT_ARGS, "--warm-cache")
    assert code == check_module.EXIT_USAGE
    assert "--warm-cache does not apply to --workload agent" in out
    assert server.requests == []


def test_savings_accepts_an_agent_cheaper_pair(history, monkeypatch, capsys):
    for _ in range(3):  # three repeats calibrate run-to-run noise (2 df)
        code, out = run_check(
            monkeypatch, capsys, AgentFakeServer(delay=0.04), *AGENT_ARGS, "--config", "quant=none"
        )
        assert code == 0, out
    code, out = run_check(
        monkeypatch, capsys, AgentFakeServer(delay=0.005), *AGENT_ARGS, "--config", "quant=fp8"
    )
    assert code == 0, out
    assert "Verdict: CHEAPER" in out
    ids = [r["id"] for r in records(history)]
    code = main([
        "savings", "--baseline", ids[2], "--candidate", ids[3], "--tokens", "1B",
    ])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "Verified savings (conservative): $" in captured.out
    assert "workload    agent: 3 turns x 2 concurrent sessions" in captured.out
    assert "synthetic agent workload" in captured.out


# --------------------------------------------------------------------------
# The default workload is unchanged
# --------------------------------------------------------------------------

# The workload and block keys a default check recorded before agent checks existed.
DEFAULT_WORKLOAD_KEYS = {
    "prompts_sha256", "prompt_cache_mode", "prompt_nonce", "sent_prompts_sha256",
    "prompt_count", "prompts_source", "blocks", "requests_per_block", "concurrency",
    "max_tokens", "warmup_requests", "temperature",
}
DEFAULT_BLOCK_KEYS = {
    "input_tokens", "tag_input_tokens", "output_tokens", "max_tokens_hits",
    "wall_clock_seconds", "dollars_per_million", "gpu_dollars",
}
PRE_AGENT_IDENTITY_FIELDS = (
    "prompts_sha256", "requests_per_block", "concurrency", "max_tokens", "prompt_cache_mode",
)
FIXTURE = Path(__file__).parent / "fixtures" / "console_checks.ndjson"


def test_default_check_records_exactly_what_it_did_before(history, monkeypatch, capsys):
    code, out = run_check(monkeypatch, capsys, AgentFakeServer())
    assert code == 0, out
    record = records(history)[-1]
    assert set(record["workload"]) == DEFAULT_WORKLOAD_KEYS
    assert all(set(block) == DEFAULT_BLOCK_KEYS for block in record["blocks"])
    assert "shape   " not in out and "AGENT" not in out
    assert check_module.workload_value(record, "workload_shape") is None


def test_recorded_history_compares_exactly_as_before(monkeypatch):
    lines = [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]
    old = [r for r in lines if check_module._valid_record(r)]
    assert len(old) >= 4
    now = [check_module.judge_recorded(old, i)[1] for i in range(len(old))]
    monkeypatch.setattr(check_module, "WORKLOAD_IDENTITY_FIELDS", PRE_AGENT_IDENTITY_FIELDS)
    before = [check_module.judge_recorded(old, i)[1] for i in range(len(old))]
    assert now == before
