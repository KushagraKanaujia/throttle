import json

import httpx
import pytest

from throttle.cli import build_parser
from throttle.own import records as r
from throttle.own import route as rt
from throttle.own.capture import AnthropicStreamAssembler, OpenAIStreamAssembler, SSEParser, create_capture_app

OPENAI = "https://api.openai.test"
ANTHROPIC = "https://api.anthropic.test"
OWNED_URL = "http://gpu-box:8000/v1"
INCUMBENT_OPENAI = "gpt-6.1-terra"        # $1 in / $5 out per 1M in the bundled table
INCUMBENT_ANTHROPIC = "claude-fable-5"    # $10 in / $50 out

API_OPENAI_ANSWER = {
    "id": "chatcmpl-api", "object": "chat.completion", "model": INCUMBENT_OPENAI,
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "from the API"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 3},
}
API_ANTHROPIC_ANSWER = {
    "id": "msg_api", "type": "message", "role": "assistant", "model": INCUMBENT_ANTHROPIC,
    "content": [{"type": "text", "text": "from the API"}], "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 3},
}


def owned_answer(content="from my model", *, tool_calls=None, finish="stop", usage=(1000, 200)):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    body = {"id": "x", "object": "chat.completion", "model": "qwen-72b",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}]}
    if usage:
        body["usage"] = {"prompt_tokens": usage[0], "completion_tokens": usage[1]}
    return body


class Recorder:
    def __init__(self, respond):
        self.respond = respond
        self.seen: list[httpx.Request] = []
        self.bodies: list[bytes] = []

    async def __call__(self, request):
        self.bodies.append(await request.aread())
        self.seen.append(request)
        result = self.respond(request)
        if isinstance(result, Exception):
            raise result
        return result


def api_respond(request):
    if request.url.path == "/v1/messages":
        return httpx.Response(200, json=API_ANTHROPIC_ANSWER)
    return httpx.Response(200, json=API_OPENAI_ANSWER)


def write_eval(root, label, verdicts, *, ts="2026-10-27T10:00:00.000Z", run_id=None, model="qwen-72b"):
    run_id = run_id or f"eval-{ts[:13]}-{label}"
    doc = {"schema": "throttle.own.eval/1", "run_id": run_id, "ts": ts, "candidate": label,
           "candidate_model": model, "types": {k: {"verdict": v} for k, v in verdicts.items()}}
    (root / "evals").mkdir(parents=True, exist_ok=True)
    (root / "evals" / f"{run_id}.json").write_text(json.dumps(doc))
    return run_id


def approve_type(root, kind, **kw):
    write_eval(root, "qwen", {kind: "NO MATERIAL DIFFERENCE"})
    route, check = rt.approve(root, kind, "qwen", OWNED_URL, "qwen-72b", **kw)
    assert route is not None, check.reason
    return route


def make_app(root, owned_respond, *, api=api_respond, seed=0, router=None):
    upstream = Recorder(api)
    owned = Recorder(owned_respond)
    router = router or rt.Router(root, transport=httpx.MockTransport(owned), seed=seed)
    app = create_capture_app(OPENAI, ANTHROPIC, redaction=r.REDACTION_OFF, root=root, salt="s",
                             client=httpx.AsyncClient(transport=httpx.MockTransport(upstream)), router=router)
    return app, upstream, owned


def front(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://capture")


async def post(app, path, body, headers=None):
    async with front(app) as client:
        resp = await client.post(path, content=json.dumps(body).encode(),
                                 headers={"content-type": "application/json", **(headers or {})})
    await app.state.capture.drain()
    return resp


def chat_request(**extra):
    return {"model": INCUMBENT_OPENAI, "messages": [{"role": "user", "content": "hello there, how are you"}], **extra}


def captured(root):
    return list(r.iter_exchanges(root))


def ledger(root):
    return [line for path in sorted((root / "ledger").glob("*.jsonl")) for line in r.read_jsonl(path)]


# ---------------------------------------------------------------------------
# approve / list / revoke
# ---------------------------------------------------------------------------


def test_approve_refused_without_eval_and_with_failing_verdict(tmp_path):
    route, check = rt.approve(tmp_path, "chat", "qwen", OWNED_URL, "qwen-72b")
    assert route is None and "no eval run" in check.reason
    write_eval(tmp_path, "qwen", {"chat": "WORSE"})
    route, check = rt.approve(tmp_path, "chat", "qwen", OWNED_URL, "qwen-72b")
    assert route is None and "WORSE" in check.reason
    assert not rt.routes_path(tmp_path).exists()


def test_approve_uses_latest_eval_run_and_checks_model(tmp_path):
    write_eval(tmp_path, "qwen", {"chat": "BETTER"}, ts="2026-10-01T00:00:00.000Z")
    write_eval(tmp_path, "qwen", {"chat": "INCONCLUSIVE"}, ts="2026-10-20T00:00:00.000Z")
    write_eval(tmp_path, "other", {"chat": "BETTER"}, ts="2026-10-25T00:00:00.000Z")
    route, check = rt.approve(tmp_path, "chat", "qwen", OWNED_URL, "qwen-72b")
    assert route is None and "INCONCLUSIVE" in check.reason
    run = write_eval(tmp_path, "qwen", {"chat": "BETTER"}, ts="2026-10-26T00:00:00.000Z")
    route, _ = rt.approve(tmp_path, "chat", "qwen", OWNED_URL, "other-model")
    assert route is None
    route, _ = rt.approve(tmp_path, "chat", "qwen", OWNED_URL, "qwen-72b", canary=0.1)
    assert route.approved_eval_run_id == run and not route.forced
    stored = json.loads(rt.routes_path(tmp_path).read_text())
    assert stored["chat"]["mode"] == "canary" and stored["chat"]["canary_rate"] == 0.1
    assert set(stored["chat"]) >= {"candidate_label", "url", "model", "api_key_env", "approved_at",
                                   "approved_eval_run_id", "mode", "canary_rate", "forced"}


def test_force_marks_route_forced_and_it_shows_loudly(tmp_path, capsys):
    parser = build_parser()
    base = ["route", "--dir", str(tmp_path), "approve", "--type", "chat", "--candidate", "qwen",
            "--url", OWNED_URL, "--model", "qwen-72b"]
    from throttle.own.cli import dispatch

    assert dispatch(parser.parse_args(base)) == 1
    assert "refused" in capsys.readouterr().err
    assert dispatch(parser.parse_args(base + ["--force"])) == 0
    assert "FORCED" in capsys.readouterr().out
    assert rt.load_routes(tmp_path)["chat"].forced is True
    assert dispatch(parser.parse_args(["route", "--dir", str(tmp_path), "list"])) == 0
    assert "FORCED" in capsys.readouterr().out
    assert any("FORCED" in line for line in rt.Router(tmp_path).describe())


def test_revoke_cli(tmp_path, capsys):
    approve_type(tmp_path, "chat")
    from throttle.own.cli import dispatch

    args = build_parser().parse_args(["route", "revoke", "--dir", str(tmp_path), "--type", "chat"])
    assert dispatch(args) == 0
    assert rt.load_routes(tmp_path) == {}
    assert dispatch(args) == 1


def test_capture_has_route_flag():
    args = build_parser().parse_args(["capture", "--route"])
    assert args.route is True
    assert build_parser().parse_args(["capture"]).route is False


# ---------------------------------------------------------------------------
# Owned serving
# ---------------------------------------------------------------------------


async def test_openai_request_served_by_owned_model(tmp_path):
    approve_type(tmp_path, "chat")
    app, upstream, owned = make_app(tmp_path, lambda req: httpx.Response(200, json=owned_answer()))
    resp = await post(app, "/v1/chat/completions", chat_request(temperature=0.2),
                      headers={"authorization": "Bearer sk-secret"})
    assert resp.status_code == 200
    assert resp.headers["x-throttle-served-by"] == "owned"
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "from my model"
    assert body["usage"] == {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200}
    assert upstream.seen == []
    sent = json.loads(owned.bodies[0])
    assert sent["model"] == "qwen-72b" and sent["temperature"] == 0.2
    assert str(owned.seen[0].url) == OWNED_URL + "/chat/completions"
    assert "authorization" not in owned.seen[0].headers  # the client's API key never reaches the owned model
    [exchange] = captured(tmp_path)
    assert exchange.served_by == r.SERVED_OWNED and not exchange.fallback
    assert exchange.route_reason == "live" and exchange.workload_type == "chat"
    assert exchange.model == INCUMBENT_OPENAI
    assert (exchange.usage.input_tokens, exchange.usage.output_tokens) == (1000, 200)


async def test_anthropic_tool_call_mapped_to_tool_use(tmp_path, monkeypatch):
    monkeypatch.setenv("OWNED_KEY", "owned-secret")
    approve_type(tmp_path, "agent", api_key_env="OWNED_KEY")
    call = {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "SF"}'}}
    app, upstream, owned = make_app(
        tmp_path, lambda req: httpx.Response(200, json=owned_answer("", tool_calls=[call], finish="tool_calls")))
    request = {
        "model": INCUMBENT_ANTHROPIC, "max_tokens": 500, "system": "You help.",
        "tools": [{"name": "get_weather", "description": "w", "input_schema": {"type": "object"}},
                  {"name": "get_time", "description": "t", "input_schema": {"type": "object"}}],
        "messages": [{"role": "user", "content": "weather in SF?"}],
    }
    resp = await post(app, "/v1/messages", request, headers={"x-api-key": "sk-ant", "anthropic-version": "2023-06-01"})
    body = resp.json()
    assert body["type"] == "message" and body["stop_reason"] == "tool_use"
    assert body["content"] == [{"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "SF"}}]
    assert body["usage"] == {"input_tokens": 1000, "output_tokens": 200}
    assert owned.seen[0].headers["authorization"] == "Bearer owned-secret"
    sent = json.loads(owned.bodies[0])
    assert sent["messages"][0] == {"role": "system", "content": "You help."}
    assert {t["function"]["name"] for t in sent["tools"]} == {"get_weather", "get_time"}
    assert upstream.seen == []
    [exchange] = captured(tmp_path)
    assert exchange.served_by == r.SERVED_OWNED and exchange.workload_type == "agent"


def parse_stream(raw: bytes, assembler):
    parser = SSEParser()
    for name, data in parser.feed(raw.decode()) + parser.flush():
        assembler.event(name, data)
    return assembler


async def test_openai_stream_is_synthesized(tmp_path):
    approve_type(tmp_path, "chat")
    long_text = "word " * 120
    app, upstream, _ = make_app(tmp_path, lambda req: httpx.Response(200, json=owned_answer(long_text)))
    resp = await post(app, "/v1/chat/completions", chat_request(stream=True, stream_options={"include_usage": True}))
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.content.rstrip().endswith(b"data: [DONE]")
    assembler = parse_stream(resp.content, OpenAIStreamAssembler())
    assert assembler.done
    result = assembler.result()
    assert result["choices"][0]["message"]["content"] == long_text
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["usage"]["completion_tokens"] == 200
    assert upstream.seen == []
    [exchange] = captured(tmp_path)
    assert exchange.stream and exchange.served_by == r.SERVED_OWNED
    assert exchange.response["choices"][0]["message"]["content"] == long_text


async def test_anthropic_stream_is_synthesized_with_tool_use(tmp_path):
    approve_type(tmp_path, "agent")
    call = {"id": "call_9", "type": "function", "function": {"name": "lookup", "arguments": '{"q": 1}'}}
    app, _, _ = make_app(tmp_path, lambda req: httpx.Response(
        200, json=owned_answer("Let me look.", tool_calls=[call], finish="tool_calls")))
    request = {"model": INCUMBENT_ANTHROPIC, "max_tokens": 300, "stream": True,
               "tools": [{"name": "lookup", "input_schema": {"type": "object"}}, {"name": "other", "input_schema": {}}],
               "messages": [{"role": "user", "content": "find it"}]}
    resp = await post(app, "/v1/messages", request, headers={"anthropic-version": "2023-06-01"})
    assembler = parse_stream(resp.content, AnthropicStreamAssembler())
    assert assembler.done
    result = assembler.result()
    assert result["content"][0] == {"type": "text", "text": "Let me look."}
    assert result["content"][1]["type"] == "tool_use" and result["content"][1]["input"] == {"q": 1}
    assert result["stop_reason"] == "tool_use"
    assert result["usage"] == {"input_tokens": 1000, "output_tokens": 200}


# ---------------------------------------------------------------------------
# Fallbacks
# ---------------------------------------------------------------------------


SCHEMA_REQUEST = chat_request(response_format={"type": "json_schema", "json_schema": {
    "name": "x", "schema": {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}}})

FALLBACKS = [
    ("connect_error", "chat", chat_request(), lambda req: httpx.ConnectError("refused", request=req)),
    ("timeout", "chat", chat_request(), lambda req: httpx.ReadTimeout("slow", request=req)),
    ("http_error", "chat", chat_request(), lambda req: httpx.Response(503, text="overloaded")),
    ("bad_response", "chat", chat_request(), lambda req: httpx.Response(200, text="not json")),
    ("truncated", "chat", chat_request(), lambda req: httpx.Response(200, json=owned_answer("cut", finish="length"))),
    ("empty_output", "chat", chat_request(), lambda req: httpx.Response(200, json=owned_answer("   "))),
    ("invalid_json", "extraction", chat_request(response_format={"type": "json_object"}),
     lambda req: httpx.Response(200, json=owned_answer("sure! {a: 1"))),
    ("schema_invalid", "extraction", SCHEMA_REQUEST, lambda req: httpx.Response(200, json=owned_answer('{"a": "one"}'))),
    ("unknown_tool", "agent",
     chat_request(tools=[{"type": "function", "function": {"name": "real", "parameters": {}}},
                         {"type": "function", "function": {"name": "also_real", "parameters": {}}}]),
     lambda req: httpx.Response(200, json=owned_answer(None, finish="tool_calls", tool_calls=[
         {"id": "c", "type": "function", "function": {"name": "made_up", "arguments": "{}"}}]))),
]


@pytest.mark.parametrize("reason,kind,request_body,owned_respond", FALLBACKS, ids=[f[0] for f in FALLBACKS])
async def test_fallback_forwards_request_unchanged(tmp_path, reason, kind, request_body, owned_respond):
    approve_type(tmp_path, kind)
    app, upstream, owned = make_app(tmp_path, owned_respond)
    raw = json.dumps(request_body).encode()
    async with front(app) as client:
        resp = await client.post("/v1/chat/completions", content=raw,
                                 headers={"authorization": "Bearer sk-x", "content-type": "application/json"})
    await app.state.capture.drain()
    assert len(owned.seen) == 1
    assert resp.json() == API_OPENAI_ANSWER
    assert "x-throttle-served-by" not in resp.headers
    assert upstream.bodies == [raw]
    assert upstream.seen[0].headers["authorization"] == "Bearer sk-x"
    [exchange] = captured(tmp_path)
    assert exchange.fallback is True and exchange.served_by == r.SERVED_API
    assert exchange.route_reason.split(":")[0] == reason
    assert exchange.workload_type == kind
    assert ledger(tmp_path) == []


async def test_missing_owned_key_falls_back(tmp_path, monkeypatch):
    monkeypatch.delenv("NOPE_KEY", raising=False)
    approve_type(tmp_path, "chat", api_key_env="NOPE_KEY")
    app, upstream, owned = make_app(tmp_path, lambda req: httpx.Response(200, json=owned_answer()))
    resp = await post(app, "/v1/chat/completions", chat_request())
    assert resp.json() == API_OPENAI_ANSWER and owned.seen == []
    assert captured(tmp_path)[0].route_reason.startswith("config_error")


async def test_stream_fallback_passes_upstream_stream_through(tmp_path):
    approve_type(tmp_path, "chat")
    chunks = [b'data: {"id":"c","choices":[{"index":0,"delta":{"content":"api"}}]}\n\n', b"data: [DONE]\n\n"]

    def api(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=b"".join(chunks))

    app, upstream, _ = make_app(tmp_path, lambda req: httpx.Response(500), api=api)
    resp = await post(app, "/v1/chat/completions", chat_request(stream=True))
    assert resp.content == b"".join(chunks)
    assert captured(tmp_path)[0].fallback


async def test_router_exception_falls_back_to_api(tmp_path):
    class Broken:
        async def route(self, *a, **k):
            raise RuntimeError("bug")

        def record(self, *a):
            raise AssertionError("not called")

        async def aclose(self):
            pass

    app, upstream, _ = make_app(tmp_path, None, router=Broken())
    resp = await post(app, "/v1/chat/completions", chat_request())
    assert resp.status_code == 200 and resp.json() == API_OPENAI_ANSWER
    assert len(upstream.seen) == 1
    [exchange] = captured(tmp_path)
    assert exchange.fallback and exchange.route_reason == "router_error: RuntimeError"


async def test_corrupt_routes_file_falls_back(tmp_path):
    rt.routes_path(tmp_path).write_text("{not json")
    app, upstream, _ = make_app(tmp_path, lambda req: httpx.Response(200, json=owned_answer()))
    resp = await post(app, "/v1/chat/completions", chat_request())
    assert resp.json() == API_OPENAI_ANSWER
    assert captured(tmp_path)[0].route_reason.startswith("router_error")


async def test_unrouted_type_and_images_go_to_api(tmp_path):
    approve_type(tmp_path, "extraction")
    app, upstream, owned = make_app(tmp_path, lambda req: httpx.Response(200, json=owned_answer()))
    await post(app, "/v1/chat/completions", chat_request())
    image = {"model": INCUMBENT_OPENAI, "response_format": {"type": "json_object"}, "messages": [
        {"role": "user", "content": [{"type": "text", "text": "x"}, {"type": "image_url", "image_url": {"url": "u"}}]}]}
    await post(app, "/v1/chat/completions", image)
    assert owned.seen == [] and len(upstream.seen) == 2
    exchanges = captured(tmp_path)
    assert [e.fallback for e in exchanges] == [False, False]
    assert [e.workload_type for e in exchanges] == ["chat", "extraction"]


# ---------------------------------------------------------------------------
# Canary, revoke, ledger
# ---------------------------------------------------------------------------


def test_canary_sampling_is_deterministic_with_seed(tmp_path):
    route = rt.Route("chat", "qwen", OWNED_URL, "qwen-72b", mode=rt.MODE_CANARY, canary_rate=0.3)
    first = [rt.Router(tmp_path, seed=7).sampled(route) for _ in range(1)]
    a, b = rt.Router(tmp_path, seed=7), rt.Router(tmp_path, seed=7)
    seq_a = [a.sampled(route) for _ in range(200)]
    seq_b = [b.sampled(route) for _ in range(200)]
    assert seq_a == seq_b and first[0] == seq_a[0]
    assert 30 < sum(seq_a) < 90
    live = rt.Route("chat", "qwen", OWNED_URL, "qwen-72b")
    assert all(a.sampled(live) for _ in range(20))


async def test_canary_routes_only_sampled_requests(tmp_path):
    approve_type(tmp_path, "chat", canary=0.5)
    app, upstream, owned = make_app(tmp_path, lambda req: httpx.Response(200, json=owned_answer()), seed=3)
    for _ in range(20):
        await post(app, "/v1/chat/completions", chat_request())
    expected = [rt.Router(tmp_path, seed=3).sampled(rt.load_routes(tmp_path)["chat"]) for _ in range(1)]
    served = [e.served_by for e in captured(tmp_path)]
    assert len(owned.seen) == served.count("owned") and len(upstream.seen) == served.count("api")
    assert 0 < len(owned.seen) < 20
    assert all(e.route_reason == "canary" for e in captured(tmp_path) if e.served_by == "owned")
    assert (served[0] == "owned") == expected[0]


async def test_revoke_takes_effect_on_next_request(tmp_path):
    approve_type(tmp_path, "chat")
    app, upstream, owned = make_app(tmp_path, lambda req: httpx.Response(200, json=owned_answer()))
    first = await post(app, "/v1/chat/completions", chat_request())
    assert first.headers.get("x-throttle-served-by") == "owned"
    assert rt.revoke(tmp_path, "chat")
    second = await post(app, "/v1/chat/completions", chat_request())
    assert second.json() == API_OPENAI_ANSWER
    assert len(owned.seen) == 1 and len(upstream.seen) == 1


async def test_ledger_written_per_owned_request(tmp_path):
    approve_type(tmp_path, "chat")
    app, _, _ = make_app(tmp_path, lambda req: httpx.Response(200, json=owned_answer(usage=(1000, 200))))
    await post(app, "/v1/chat/completions", chat_request())
    await post(app, "/v1/messages", {"model": INCUMBENT_ANTHROPIC, "max_tokens": 900,
                                     "messages": [{"role": "user", "content": "hello there, how are you"}]},
               headers={"anthropic-version": "2023-06-01"})
    exchanges = captured(tmp_path)
    lines = ledger(tmp_path)
    assert [l["exchange_id"] for l in lines] == [e.id for e in exchanges]
    openai_line, anthropic_line = lines
    assert openai_line["type"] == "chat" and openai_line["incumbent_model"] == INCUMBENT_OPENAI
    assert (openai_line["owned_input_tokens"], openai_line["owned_output_tokens"]) == (1000, 200)
    assert openai_line["api_cost_no_cache_usd"] == pytest.approx(1000 * 1e-6 + 200 * 5e-6)
    assert anthropic_line["api_cost_no_cache_usd"] == pytest.approx(1000 * 10e-6 + 200 * 50e-6)
    assert openai_line["price_label"].startswith("LIST PRICE")
    assert openai_line["forced"] is False and openai_line["excluded"] is None
    path = rt.ledger_path(tmp_path, exchanges[0].ts[:7])
    assert path.exists()


async def test_ledger_excludes_missing_usage_and_unknown_price(tmp_path):
    approve_type(tmp_path, "chat")
    app, _, _ = make_app(tmp_path, lambda req: httpx.Response(200, json=owned_answer(usage=None)))
    await post(app, "/v1/chat/completions", chat_request())
    app, _, _ = make_app(tmp_path, lambda req: httpx.Response(200, json=owned_answer()))
    await post(app, "/v1/chat/completions", {**chat_request(), "model": "some-unpriced-model"})
    no_usage, no_price = ledger(tmp_path)
    assert no_usage["api_cost_no_cache_usd"] is None and "usage" in no_usage["excluded"]
    assert no_price["api_cost_no_cache_usd"] is None and "no list price" in no_price["excluded"]


# ---------------------------------------------------------------------------
# Savings statement
# ---------------------------------------------------------------------------


# gpt-6.1-terra: $1 input, $0.05 cached input, $5 output per 1M tokens.
# One ledger line = 1M owned input + 100k owned output tokens.
UNCACHED = 1.0 + 0.5
ALL_CACHED = 0.05 + 0.5
SHARE_40 = 0.6 * 1.0 + 0.4 * 0.05 + 0.5


def ledger_line(i, *, forced=False, kind="chat", model=INCUMBENT_OPENAI, usage=(1_000_000, 100_000)):
    return {"schema": rt.LEDGER_SCHEMA, "ts": "2026-10-05T00:00:00.000Z", "exchange_id": f"e{i}", "type": kind,
            "incumbent_provider": "openai", "incumbent_model": model, "forced": forced,
            "owned_input_tokens": usage[0] if usage else None, "owned_output_tokens": usage[1] if usage else None}


def api_exchange(root, *, kind="chat", model=INCUMBENT_OPENAI, input_tokens=1000, cached=400, ts="2026-10-04T00:00:00.000Z",
                 served_by=r.SERVED_API, request=None, fallback=False, reason=None):
    r.append_exchange(r.Exchange(provider="openai", endpoint="/v1/chat/completions", model=model, request=request,
                                 response=None, status=200, latency_ms=1, served_by=served_by, fallback=fallback,
                                 route_reason=reason, workload_type=kind, ts=ts, redaction=r.REDACTION_OFF,
                                 usage=r.Usage(input_tokens, 10, cached)), root)


def seed_month(root, *, owned_lines, forced=0, fallbacks=(), history=True, no_usage=True):
    path = rt.ledger_path(root, "2026-10")
    for i in range(owned_lines):
        r.append_jsonl(path, ledger_line(i, forced=i < forced))
    if no_usage:
        r.append_jsonl(path, ledger_line("nousage", usage=None))
    if history:
        api_exchange(root, input_tokens=600, cached=200)
        api_exchange(root, input_tokens=400, cached=200)       # pooled share: 400 / 1000
    for reason in fallbacks:
        api_exchange(root, fallback=True, reason=reason, cached=None)
    api_exchange(root, fallback=True, reason="timeout", ts="2026-09-06T00:00:00.000Z", cached=None)


def test_savings_math_positive_and_forced_not_billable(tmp_path):
    seed_month(tmp_path, owned_lines=10, forced=2, fallbacks=["timeout", "truncated", "http_error: 503"])
    doc = rt.savings_statement(tmp_path, "2026-10", gpu_hourly_rate=0.005, gpus=1)
    assert doc["period_hours"] == 31 * 24
    assert doc["api_cost_avoided_usd"] == pytest.approx(8 * SHARE_40)
    assert doc["forced"] == {"requests": 2, "api_cost_avoided_usd": pytest.approx(2 * SHARE_40), "billable": False}
    assert doc["owned_cost_usd"] == pytest.approx(0.005 * 744)
    assert doc["net_realized_savings_usd"] == pytest.approx(8 * SHARE_40 - 3.72)
    assert doc["billable"] is True
    assert doc["requests_served_owned"] == 11 and doc["requests_priced"] == 10
    assert doc["excluded"] == {"owned model reported no token usage": 1}
    assert doc["cache_basis_counts"] == {rt.CACHE_MEASURED: 10}
    assert doc["fallbacks"] == 3
    assert doc["fallback_rate"] == pytest.approx(3 / 14)
    assert set(doc["fallbacks_by_reason"]) == {"timeout", "truncated", "http_error"}
    assert "ASSUMED" in doc["owned_cost_label"] and "MEASURED" in doc["api_cost_avoided_label"]
    row = doc["by_type"]["chat"]
    assert row["forced_requests"] == 2 and row["billable_avoided_usd"] == pytest.approx(8 * SHARE_40)
    text = rt.render_statement(doc)
    assert "FORCED (not billable; quality not proven)" in text and "no bill" not in text
    assumptions = " ".join(doc["assumptions"])
    assert "input + output tokens" in assumptions and "cache-hit share is measured" in assumptions
    assert "not billable" in assumptions


def test_forced_only_savings_are_never_billed(tmp_path):
    seed_month(tmp_path, owned_lines=5, forced=5, no_usage=False)
    doc = rt.savings_statement(tmp_path, "2026-10", owned_cost_usd=0.01)
    assert doc["forced"]["api_cost_avoided_usd"] == pytest.approx(5 * SHARE_40)
    assert doc["api_cost_avoided_usd"] == 0
    assert doc["net_realized_savings_usd"] == pytest.approx(-0.01)
    assert doc["billable"] is False
    text = rt.render_statement(doc)
    assert "No savings this period, no bill." in text and "FORCED (not billable" in text


def test_no_cache_history_prices_all_input_at_cached_rate(tmp_path):
    seed_month(tmp_path, owned_lines=2, history=False, no_usage=False)
    # Not history for chat/terra: another model, an owned-served call, unreported cache counts, another month.
    api_exchange(tmp_path, model="gpt-6.1-sol", cached=0)
    api_exchange(tmp_path, served_by=r.SERVED_OWNED, cached=0)
    api_exchange(tmp_path, cached=None)
    api_exchange(tmp_path, cached=0, ts="2026-09-01T00:00:00.000Z")
    doc = rt.savings_statement(tmp_path, "2026-10", owned_cost_usd=0)
    assert doc["api_cost_avoided_usd"] == pytest.approx(2 * ALL_CACHED)
    assert doc["cache_basis_counts"] == {rt.CACHE_NONE: 2}
    assert "no cache history, priced conservatively" in " ".join(rt.render_statement(doc).split())


def test_cache_share_zero_prices_input_at_base_rate(tmp_path):
    seed_month(tmp_path, owned_lines=1, history=False, no_usage=False)
    api_exchange(tmp_path, cached=0)
    doc = rt.savings_statement(tmp_path, "2026-10", owned_cost_usd=0)
    assert doc["api_cost_avoided_usd"] == pytest.approx(UNCACHED)


def test_cache_history_is_per_type_and_classifies_untyped_captures(tmp_path):
    path = rt.ledger_path(tmp_path, "2026-10")
    r.append_jsonl(path, ledger_line(1, kind="chat"))
    r.append_jsonl(path, ledger_line(2, kind="extraction"))
    untyped = {"model": INCUMBENT_OPENAI, "messages": [{"role": "user", "content": "hello there, how are you"}]}
    api_exchange(tmp_path, kind=None, request=untyped, input_tokens=1000, cached=0)   # classified as chat
    doc = rt.savings_statement(tmp_path, "2026-10", owned_cost_usd=0)
    assert doc["by_type"]["chat"]["billable_avoided_usd"] == pytest.approx(UNCACHED)
    assert doc["by_type"]["extraction"]["billable_avoided_usd"] == pytest.approx(ALL_CACHED)
    assert doc["by_type"]["extraction"]["cache_basis"] == [rt.CACHE_NONE]


def test_no_published_cached_rate_is_labelled(tmp_path):
    prices_file = tmp_path / "prices.json"
    prices_file.write_text(json.dumps([{"provider": "openai", "model": INCUMBENT_OPENAI,
                                        "input_per_m": 1, "output_per_m": 5}]))
    seed_month(tmp_path, owned_lines=1, history=False, no_usage=False)
    from throttle.own import pricing

    doc = rt.savings_statement(tmp_path, "2026-10", owned_cost_usd=0, prices=pricing.load_prices(prices_file))
    assert doc["api_cost_avoided_usd"] == pytest.approx(UNCACHED)
    assert doc["cache_basis_counts"] == {rt.CACHE_NO_RATE: 1}
    assert "ASSUMED" in doc["api_cost_avoided_label"]


def test_savings_zero_or_negative_says_no_bill(tmp_path, capsys):
    seed_month(tmp_path, owned_lines=2, no_usage=False)
    doc = rt.savings_statement(tmp_path, "2026-10", owned_cost_usd=2 * SHARE_40)
    assert doc["net_realized_savings_usd"] == pytest.approx(0.0)
    assert doc["billable"] is False
    assert doc["owned_cost_label"].startswith("OPERATOR-REPORTED")
    assert "No savings this period, no bill." in rt.render_statement(doc)
    from throttle.own.cli import dispatch

    args = build_parser().parse_args(["route", "savings", "--dir", str(tmp_path), "--month", "2026-10",
                                      "--gpu-hourly-rate", "1", "--gpus", "2", "--json"])
    assert dispatch(args) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["net_realized_savings_usd"] == pytest.approx(2 * SHARE_40 - 1488.0)
    assert out["billable"] is False


def test_savings_requires_owned_cost_inputs(tmp_path, capsys):
    from throttle.own.cli import dispatch

    args = build_parser().parse_args(["route", "savings", "--dir", str(tmp_path), "--gpus", "1"])
    assert dispatch(args) == 2
    assert "--gpu-hourly-rate" in capsys.readouterr().err
    args = build_parser().parse_args(["route", "savings", "--dir", str(tmp_path), "--month", "10-2026",
                                      "--owned-cost-usd", "1"])
    assert dispatch(args) == 2
