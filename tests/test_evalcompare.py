"""`throttle compare --baseline URL --candidate URL --eval-set FILE`: live
model comparison against mocked OpenAI-compatible endpoints (httpx.MockTransport,
no sockets). Each fake model answers from a table and sleeps a fixed time per
request, so cost and throughput differences are deterministic."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

import throttle.config as config_module
import throttle.evalcompare as ev
from throttle.cli import main

BASE = "http://127.0.0.1:30000"
CAND = "http://127.0.0.1:30001"

EVALS = [
    {"id": "capital", "prompt": "Capital of France?", "check": "contains:Paris"},
    {"id": "json", "prompt": "Give me {\"a\": 1}", "check": "json"},
    {"id": "num", "prompt": "2+2?", "check": "regex:^\\s*4\\s*$"},
    {"id": "free", "prompt": "Say hello"},
]

GOOD = {
    "Capital of France?": "The capital is Paris.",
    "Give me {\"a\": 1}": "```json\n{\"a\": 1}\n```",
    "2+2?": "4",
    "Say hello": "Hello there!",
}


def _sse(text: str, tokens: int, usage: bool = True) -> bytes:
    lines = []
    for i, piece in enumerate([text[: len(text) // 2], text[len(text) // 2:]]):
        lines.append({"choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]})
    lines.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    if usage:
        lines.append({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": tokens}})
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in lines) + "data: [DONE]\n\n"
    return body.encode()


class FakeModels:
    """Routes by port: each side has an answer table and a per-request delay."""

    def __init__(self, base_answers, cand_answers, base_delay=0.02, cand_delay=0.005,
                 cand_status=200, cand_usage=True):
        self.sides = {
            30000: (base_answers, base_delay, 200, True),
            30001: (cand_answers, cand_delay, cand_status, cand_usage),
        }
        self.requests: list[dict] = []

    async def handle(self, request: httpx.Request) -> httpx.Response:
        answers, delay, status, usage = self.sides[request.url.port]
        if request.method == "GET":
            if request.url.path == "/version":
                return httpx.Response(200, json={"version": f"0.{request.url.port}.0"})
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "base"}, {"id": "cand"}]})
            return httpx.Response(404)
        body = json.loads(request.content)
        self.requests.append(body)
        if status != 200:
            return httpx.Response(status, text="boom")
        await asyncio.sleep(delay)
        prompt = body["messages"][-1]["content"]
        text = answers[prompt]
        return httpx.Response(200, content=_sse(text, tokens=10 * len(text.split()) + 10,
                                                usage=usage),
                              headers={"content-type": "text/event-stream"})

    def install(self, monkeypatch):
        monkeypatch.setattr(ev, "_TRANSPORT", httpx.MockTransport(self.handle))

        def sync_handle(request: httpx.Request) -> httpx.Response:
            return asyncio.run(self.handle(request))

        monkeypatch.setattr(ev, "_SYNC_TRANSPORT", httpx.MockTransport(sync_handle))


@pytest.fixture
def evalset(tmp_path, monkeypatch) -> Path:
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", tmp_path / "no-config.yaml")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    path = tmp_path / "evals.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in EVALS) + "\n")
    return path


def run(evalset: Path, tmp_path: Path, *extra: str) -> int:
    return main([
        "compare", "--baseline", BASE, "--baseline-model", "base",
        "--candidate", CAND, "--candidate-model", "cand",
        "--eval-set", str(evalset), "--concurrency", "1", "--runs", "3",
        "--gpu-price", "2.00", "--out", str(tmp_path / "out"), *extra,
    ])


def test_cheaper_candidate_with_same_answers_says_switch(evalset, tmp_path, monkeypatch, capsys):
    FakeModels(GOOD, GOOD).install(monkeypatch)
    code = run(evalset, tmp_path)
    out = capsys.readouterr().out
    assert code == ev.EXIT_SWITCH
    last = out.strip().splitlines()[-1]
    assert "cheaper per passing answer and passes the same checks: SWITCH" in last
    assert "TTFT p50" in out and "throughput tok/s" in out and "$ / 1k passing ans." in out

    result = json.loads((tmp_path / "out" / "result.json").read_text())
    assert result["verdict"] == "SWITCH"
    assert result["baseline"]["summary"]["pass_rate"]["median"] == 1.0
    assert result["candidate"]["summary"]["pass_rate"]["median"] == 1.0
    assert len(result["baseline"]["runs"]) == 3
    assert result["drift"]["exact_match_rate"] == 1.0
    assert result["drift"]["output_length_change_percent"] == 0.0
    assert result["cost_change_percent"] < -30

    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    import hashlib
    assert manifest["eval_set"]["sha256"] == hashlib.sha256(evalset.read_bytes()).hexdigest()
    assert manifest["sides"]["baseline"]["model"] == "base"
    assert manifest["sides"]["baseline"]["engine_version"] == "0.30000.0"
    assert manifest["sides"]["candidate"]["engine_version"] == "0.30001.0"
    assert "--concurrency" in manifest["command"] and manifest["flags"]["runs"] == 3
    assert manifest["created_at"]

    rows = [json.loads(l) for l in (tmp_path / "out" / "outputs.jsonl").read_text().splitlines()]
    assert len(rows) == 2 * 3 * len(EVALS)
    assert {r["side"] for r in rows} == {"baseline", "candidate"}
    assert all(r["ttft_s"] is not None for r in rows)
    # runs alternate B C B C ...
    assert [r["side"] for r in rows[:: len(EVALS)]] == ["baseline", "candidate"] * 3


def test_candidate_failing_json_says_dont_switch(evalset, tmp_path, monkeypatch, capsys):
    bad = dict(GOOD)
    bad["Give me {\"a\": 1}"] = "Sure! Here it is: a=1"
    FakeModels(GOOD, bad).install(monkeypatch)
    code = run(evalset, tmp_path)
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert code == ev.EXIT_DONT_SWITCH
    assert "fails 1 JSON check the baseline passes" in last
    assert last.endswith("DON'T SWITCH")
    result = json.loads((tmp_path / "out" / "result.json").read_text())
    assert result["candidate"]["summary"]["json_error_rate"]["median"] == 1.0
    assert result["drift"]["regressed_json_items"] == ["json"]


def test_same_speed_is_within_noise(evalset, tmp_path, monkeypatch, capsys):
    FakeModels(GOOD, GOOD, base_delay=0.01, cand_delay=0.01).install(monkeypatch)
    # identical timings: make every run's wall clock identical too
    ticks = iter(range(10_000))
    monkeypatch.setattr(ev.time, "perf_counter", lambda: float(next(ticks)))
    code = run(evalset, tmp_path)
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert code == ev.EXIT_SWITCH
    assert "within noise" in last


def test_slower_candidate_says_dont_switch(evalset, tmp_path, monkeypatch, capsys):
    FakeModels(GOOD, GOOD, base_delay=0.005, cand_delay=0.03).install(monkeypatch)
    code = run(evalset, tmp_path)
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert code == ev.EXIT_DONT_SWITCH
    assert "more expensive per passing answer: DON'T SWITCH" in last


def test_candidate_drift_and_missing_usage(evalset, tmp_path, monkeypatch, capsys):
    drifted = dict(GOOD)
    drifted["Say hello"] = "Hi! How can I help you today, friend?"
    FakeModels(GOOD, drifted, cand_usage=False).install(monkeypatch)
    run(evalset, tmp_path)
    out = capsys.readouterr().out
    assert "drift vs baseline" in out and "75% identical" in out
    assert "estimated" in out
    result = json.loads((tmp_path / "out" / "result.json").read_text())
    assert result["drift"]["exact_match_rate"] == 0.75
    assert result["drift"]["output_length_change_percent"] > 10
    assert result["drift"]["baseline_self_exact_match_rate"] == 1.0
    assert result["candidate"]["summary"]["tokens_estimated"] is True


def test_failing_endpoint_exits_3(evalset, tmp_path, monkeypatch, capsys):
    FakeModels(GOOD, GOOD, cand_status=500).install(monkeypatch)
    code = run(evalset, tmp_path)
    assert code == ev.EXIT_FAILED
    assert "every request to the candidate failed" in capsys.readouterr().err


@pytest.mark.parametrize("line, message", [
    ("not json", "not valid JSON"),
    ('{"foo": 1}', "needs \"prompt\""),
    ('{"prompt": "x", "check": "fuzzy"}', "\"check\" must be"),
    ('{"prompt": "x", "check": "exact"}', "needs \"expected\""),
    ('{"prompt": "x", "check": "regex:("}', "bad regex"),
    ('{"prompt": "x", "check": "json", "expected": "not json"}', "must be JSON"),
])
def test_bad_eval_set_is_a_usage_error(tmp_path, monkeypatch, capsys, line, message):
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", tmp_path / "no-config.yaml")
    path = tmp_path / "bad.jsonl"
    path.write_text(line + "\n")
    assert run(path, tmp_path) == ev.EXIT_USAGE
    assert message in capsys.readouterr().err


def test_missing_flags_and_price(evalset, tmp_path, capsys):
    assert main(["compare", "--baseline", BASE, "--eval-set", str(evalset)]) == ev.EXIT_USAGE
    assert "--baseline-model" in capsys.readouterr().err
    code = main(["compare", "--baseline", BASE, "--baseline-model", "b", "--candidate", CAND,
                 "--candidate-model", "c", "--eval-set", str(evalset)])
    assert code == ev.EXIT_USAGE
    assert "--gpu-price" in capsys.readouterr().err


def test_checks():
    item = ev.EvalItem("1", [], expected="Paris", check=None)
    assert ev.run_check(item, "  paris. ") is True
    assert ev.run_check(ev.EvalItem("2", []), "anything") is None
    assert ev.run_check(ev.EvalItem("3", [], expected="42", check="exact"), "42\n") is True
    assert ev.run_check(ev.EvalItem("4", [], expected='{"a": 1}', check="json"), '{"a":1}') is True
    assert ev.run_check(ev.EvalItem("5", [], expected='{"a": 1}', check="json"), '{"a":2}') is False
    assert ev.tool_call_errors([{"function": {"name": "f", "arguments": '{"x": 1}'}}]) == 0
    assert ev.tool_call_errors([{"function": {"name": "f", "arguments": '{"x": '}},
                                {"function": {"name": "", "arguments": "{}"}}]) == 2


def test_tool_call_stream_is_merged_and_checked():
    calls: list[dict] = []
    ev._merge_tool_delta(calls, [{"index": 0, "id": "c1", "function": {"name": "get", "arguments": '{"ci'}}])
    ev._merge_tool_delta(calls, [{"index": 0, "function": {"arguments": 'ty": "SF"}'}}])
    assert calls[0]["function"] == {"name": "get", "arguments": '{"city": "SF"}'}
    assert ev.tool_call_errors(calls) == 0


def test_saved_report_compare_still_requires_reports(capsys):
    with pytest.raises(SystemExit):
        main(["compare"])
    assert "--eval-set" in capsys.readouterr().err


def test_odd_stream_events_do_not_crash():
    """null choices/deltas and non-numeric usage from a quirky server are skipped."""

    events = [
        {"choices": [None, {"delta": None}, {"delta": {"content": "hi"}}]},
        {"choices": {"oops": 1}},
        {"choices": [], "usage": {"prompt_tokens": "x", "completion_tokens": None}},
    ]
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"

    async def go():
        transport = httpx.MockTransport(lambda r: httpx.Response(200, content=body.encode()))
        async with httpx.AsyncClient(transport=transport) as client:
            return await ev.stream_one(client, BASE + "/v1/chat/completions", {}, "m",
                                       ev.EvalItem("1", [{"role": "user", "content": "q"}]),
                                       max_tokens=8, temperature=0.0)

    answer = asyncio.run(go())
    assert answer.error is None and answer.text == "hi"
    assert answer.prompt_tokens == 0 and answer.output_tokens == 1 and answer.tokens_estimated
