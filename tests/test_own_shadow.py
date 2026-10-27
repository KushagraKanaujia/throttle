import asyncio
import copy
import json

import httpx
import pytest

from throttle.cli import build_parser
from throttle.own import records as r
from throttle.own import shadow as sh

from own_eval_helpers import by_workload_field, candidate_transport, chat_response, exchange, last_user


def _seed(tmp_path, items):
    for ex in items:
        r.append_exchange(ex, tmp_path)
    return items


def _args(tmp_path, *extra):
    return build_parser().parse_args([
        "shadow", "--candidate-url", "http://gpu:8000/v1", "--candidate-model", "qwen",
        "--label", "qwen@vllm", "--dir", str(tmp_path), *extra,
    ])


def test_select_backfill_is_stratified_seeded_and_skips_ineligible():
    items = [exchange(f"c{i}", kind="classification") for i in range(10)]
    items += [exchange(f"x{i}", kind="extraction") for i in range(3)]
    bad = [
        exchange("err", kind="chat", status=500),
        exchange("img", kind="chat", images=True),
        exchange("meta", kind="chat"),
    ]
    bad[2].redaction = r.REDACTION_METADATA
    pool = items + bad
    first = sh.select_backfill(pool, per_type=4, max_requests=100, seed=7, classify=by_workload_field)
    again = sh.select_backfill(pool, per_type=4, max_requests=100, seed=7, classify=by_workload_field)
    assert [e.id for e, _, _ in first] == [e.id for e, _, _ in again]
    kinds = [k for _, _, k in first]
    assert kinds.count("classification") == 4 and kinds.count("extraction") == 3
    assert "chat" not in kinds
    capped = sh.select_backfill(pool, per_type=4, max_requests=3, seed=7, classify=by_workload_field)
    assert sorted(k for _, _, k in capped) == ["classification", "classification", "extraction"]
    skip = {items[0].id}
    assert items[0].id not in {e.id for e, _, _ in sh.select_backfill(pool, skip=skip, classify=by_workload_field)}


def test_classifier_falls_back_to_all_without_workloads_module(monkeypatch):
    import importlib

    real = importlib.import_module

    def fake(name, *a, **k):
        if name == "throttle.own.workloads":
            raise ImportError(name)
        return real(name, *a, **k)

    monkeypatch.setattr(sh.importlib, "import_module", fake)
    picked = sh.select_backfill([exchange("a"), exchange("b")])
    assert {k for _, _, k in picked} == {"all"}


def test_backfill_stores_results_and_skips_on_rerun(tmp_path, capsys, monkeypatch):
    items = _seed(tmp_path, [exchange(f"q{i}", kind="classification", max_tokens=17) for i in range(3)])
    calls = []
    transport = candidate_transport(lambda body: chat_response("ans " + last_user(body)), calls)
    monkeypatch.setenv("CAND_KEY", "sk-cand")
    code = sh.handle(_args(tmp_path, "--api-key-env", "CAND_KEY"), classify=by_workload_field, transport=transport)
    assert code == 0
    assert len(calls) == 3
    url, headers, body = calls[0]
    assert url == "http://gpu:8000/v1/chat/completions"
    assert headers["authorization"] == "Bearer sk-cand"
    assert body["model"] == "qwen" and body["max_tokens"] == 17 and body["stream"] is False
    results = {s.exchange_id: s for s in r.iter_shadow("qwen@vllm", tmp_path)}
    assert set(results) == {e.id for e in items}
    one = results[items[0].id]
    assert one.ok and one.output_text == "ans q0" and one.usage.input_tokens == 11 and one.latency_ms >= 0
    out = capsys.readouterr().out
    assert "classification" in out and "3/3 done" in out and "errors: none" in out
    assert "sk-cand" not in out

    calls.clear()
    assert sh.handle(_args(tmp_path), classify=by_workload_field, transport=transport) == 0
    assert calls == []
    assert "3 already shadowed" in capsys.readouterr().out


def test_backfill_names_context_overflow_and_retries_transport_errors(tmp_path, capsys):
    a, b, c = _seed(tmp_path, [exchange("long", kind="chat"), exchange("down", kind="chat"), exchange("fine", kind="chat")])
    sent = []

    def answer(body):
        text = last_user(body)
        sent.append(text)
        if text == "long":
            return httpx.Response(400, json={"error": {"message": "This model's maximum context length is 2048 tokens."}})
        if text == "down":
            raise httpx.ConnectError("refused")
        return chat_response("yes")

    assert sh.handle(_args(tmp_path), classify=by_workload_field, transport=candidate_transport(answer)) == 0
    out = capsys.readouterr().out
    assert "context_overflow 1" in out and "connect 1" in out
    stored = {s.exchange_id: s for s in r.iter_shadow("qwen@vllm", tmp_path)}
    assert sh.error_kind(stored[a.id].error) == "context_overflow"
    assert "2048-token context window" in stored[a.id].error
    # Retried once right away; the second failure is stored as permanent.
    assert sent.count("down") == 2
    assert stored[b.id].status == sh.FINAL_TRANSPORT_STATUS and not stored[b.id].ok
    assert sh.error_kind(stored[b.id].error) == "connect"
    assert sh.done_ids("qwen@vllm", tmp_path) == {a.id, b.id, c.id}


def test_anthropic_exchange_is_translated_for_the_candidate(tmp_path):
    request = {"model": "claude-x", "system": "Sort it.", "max_tokens": 99,
               "messages": [{"role": "user", "content": [{"type": "text", "text": "b a"}]}]}
    response = {"content": [{"type": "text", "text": "a b"}], "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 2}}
    ex = r.Exchange(provider=r.PROVIDER_ANTHROPIC, endpoint="/v1/messages", model="claude-x",
                    request=request, response=response, status=200, latency_ms=3.0)
    r.append_exchange(ex, tmp_path)
    calls = []
    sh.handle(_args(tmp_path), classify=by_workload_field,
              transport=candidate_transport(lambda body: chat_response("a b"), calls))
    body = calls[0][2]
    assert body["messages"] == [{"role": "system", "content": "Sort it."}, {"role": "user", "content": "b a"}]
    assert body["max_tokens"] == 99


def test_backfill_never_writes_candidate_output_into_capture(tmp_path):
    _seed(tmp_path, [exchange("q", output="API ANSWER", kind="chat")])
    capture_files = sorted(r.capture_dir(tmp_path).glob("*.jsonl"))
    before = {p: p.read_bytes() for p in capture_files}
    sh.handle(_args(tmp_path), classify=by_workload_field,
              transport=candidate_transport(lambda body: chat_response("SHADOW-SENTINEL")))
    assert {p: p.read_bytes() for p in sorted(r.capture_dir(tmp_path).glob("*.jsonl"))} == before
    assert all("SHADOW-SENTINEL" not in json.dumps(e.to_json()) for e in r.iter_exchanges(tmp_path))


def test_missing_api_key_env_is_a_usage_error(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("NOPE_KEY", raising=False)
    assert sh.handle(_args(tmp_path, "--api-key-env", "NOPE_KEY")) == 2
    assert "NOPE_KEY" in capsys.readouterr().err


def test_parse_since():
    from datetime import datetime, timezone

    now = datetime(2026, 10, 27, tzinfo=timezone.utc)
    assert sh.parse_since("7d", now=now) == datetime(2026, 10, 20, tzinfo=timezone.utc)
    assert sh.parse_since(None) is None
    with pytest.raises(ValueError):
        sh.parse_since("soon")


def test_chat_url_forms():
    assert sh.chat_url("http://h:8000") == "http://h:8000/v1/chat/completions"
    assert sh.chat_url("http://h:8000/v1/") == "http://h:8000/v1/chat/completions"
    assert sh.chat_url("http://h/x/chat/completions") == "http://h/x/chat/completions"


# ---------------------------------------------------------------------------
# LiveShadow
# ---------------------------------------------------------------------------


def _live(tmp_path, transport, **kw):
    kw.setdefault("sample_rate", 1.0)
    return sh.LiveShadow(url="http://gpu:8000", model="qwen", label="live", root=tmp_path,
                         seed=1, classify=by_workload_field, transport=transport, **kw)


def test_live_shadow_budget_caps_total(tmp_path):
    calls = []
    live = _live(tmp_path, candidate_transport(lambda b: chat_response("x"), calls), max_total=2)

    async def go():
        for i in range(5):
            await live(exchange(f"q{i}", kind="chat"))
        await live.drain()

    asyncio.run(go())
    assert len(calls) == 2
    assert live.stats["started"] == 2 and live.stats["dropped_budget"] == 3 and live.stats["stored"] == 2
    assert len(list(r.iter_shadow("live", tmp_path))) == 2


def test_live_shadow_drops_when_busy_and_does_not_wait_for_candidate(tmp_path):
    release = asyncio.Event()
    calls = []

    async def handler(request):
        calls.append(1)
        await release.wait()
        return httpx.Response(200, json=chat_response("late"))

    live = _live(tmp_path, httpx.MockTransport(handler), max_in_flight=1)

    async def go():
        await live(exchange("one", kind="chat"))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        # The callback returned while the candidate is still blocked.
        assert live.in_flight == 1
        await live(exchange("two", kind="chat"))
        await live(exchange("three", kind="chat"))
        assert live.stats["dropped_busy"] == 2
        release.set()
        await live.drain()

    asyncio.run(go())
    assert live.stats["started"] == 1 and live.in_flight == 0 and len(calls) == 1


def test_live_shadow_per_type_rates_and_ineligible(tmp_path):
    calls = []
    live = _live(tmp_path, candidate_transport(lambda b: chat_response("x"), calls),
                 sample_rate={"classification": 1.0, "*": 0.0})

    async def go():
        await live(exchange("a", kind="classification"))
        await live(exchange("b", kind="chat"))
        await live(exchange("c", kind="classification", status=500))
        await live.drain()

    asyncio.run(go())
    assert len(calls) == 1
    assert live.stats["not_sampled"] == 1 and live.stats["ineligible"] == 1


def test_live_shadow_swallows_candidate_failures(tmp_path):
    def boom(body):
        raise RuntimeError("bug in fake")

    live = _live(tmp_path, candidate_transport(boom))

    async def go():
        assert await live(exchange("a", kind="chat")) is None
        await live.drain()

    asyncio.run(go())
    assert live.stats["errors"] == 1 and live.in_flight == 0


def test_shadow_output_is_never_served(tmp_path):
    """End to end through a minimal capture-style proxy: the client gets the
    upstream API answer, the exchange log keeps the upstream answer, and the
    candidate's answer exists only in the shadow log."""

    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    upstream = candidate_transport(lambda body: chat_response("API ANSWER"))
    candidate_calls = []
    live = _live(tmp_path, candidate_transport(lambda b: chat_response("SHADOW-SENTINEL"), candidate_calls))
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def proxy(request: Request):
        body = await request.json()
        async with httpx.AsyncClient(transport=upstream) as client:
            upstream_response = await client.post("http://api.example/v1/chat/completions", json=body)
        payload = upstream_response.json()
        ex = r.Exchange(provider=r.PROVIDER_OPENAI, endpoint="/v1/chat/completions", model=body["model"],
                        request=body, response=copy.deepcopy(payload), status=upstream_response.status_code,
                        latency_ms=1.0, usage=r.usage_from_openai(payload), workload_type="chat")
        r.append_exchange(ex, tmp_path)
        snapshot = copy.deepcopy(ex.to_json())
        await live(ex)
        assert ex.to_json() == snapshot  # the callback did not touch the exchange
        return JSONResponse(payload)

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as client:
            replies = [
                await client.post("/v1/chat/completions", json={
                    "model": "gpt-4o-mini", "messages": [{"role": "user", "content": f"hi {i}"}]})
                for i in range(3)
            ]
        await live.drain()
        return replies

    replies = asyncio.run(go())
    assert len(candidate_calls) == 3
    for reply in replies:
        assert reply.json()["choices"][0]["message"]["content"] == "API ANSWER"
        assert "SHADOW-SENTINEL" not in reply.text
    for ex in r.iter_exchanges(tmp_path):
        assert "SHADOW-SENTINEL" not in json.dumps(ex.to_json())
    shadows = list(r.iter_shadow("live", tmp_path))
    assert len(shadows) == 3 and all(s.output_text == "SHADOW-SENTINEL" for s in shadows)


# ---------------------------------------------------------------------------
# Review fixes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("content", [None, ""])
def test_empty_candidate_answer_is_bad_response(content):
    ex = exchange("hi", kind="chat")

    async def go():
        transport = candidate_transport(lambda body: chat_response(content))
        async with httpx.AsyncClient(transport=transport) as client:
            return await sh.shadow_one(client, ex, r.normalize(ex), url="http://c/v1", model="m", label="c")

    res = asyncio.run(go())
    assert res.error == "bad_response: empty answer" and not res.ok


def test_tool_call_only_answer_is_ok():
    ex = exchange("hi", kind="agent")
    calls = [{"id": "1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]

    async def go():
        transport = candidate_transport(lambda body: chat_response(None, tool_calls=calls, finish="tool_calls"))
        async with httpx.AsyncClient(transport=transport) as client:
            return await sh.shadow_one(client, ex, r.normalize(ex), url="http://c/v1", model="m", label="c")

    res = asyncio.run(go())
    assert res.ok and res.output_text == "" and res.tool_calls[0]["name"] == "f"


def test_transport_failure_that_recovers_on_retry_is_stored_ok(tmp_path):
    (ex,) = _seed(tmp_path, [exchange("flaky", kind="chat")])
    tries = []

    def answer(body):
        tries.append(1)
        if len(tries) == 1:
            raise httpx.ReadTimeout("slow")
        return chat_response("yes")

    sh.handle(_args(tmp_path), classify=by_workload_field, transport=candidate_transport(answer))
    (stored,) = list(r.iter_shadow("qwen@vllm", tmp_path))
    assert len(tries) == 2 and stored.ok and stored.output_text == "yes"
