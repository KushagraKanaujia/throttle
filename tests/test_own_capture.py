import asyncio
import gzip
import json
import logging
import time
from pathlib import Path

import httpx
import pytest

from throttle.cli import build_parser
from throttle.own import records as r
from throttle.own.capture import (
    OpenAIStreamAssembler,
    SSEParser,
    compute_stats,
    create_capture_app,
    handle,
    startup_panel,
)

OPENAI = "https://api.openai.test"
ANTHROPIC = "https://api.anthropic.test/v1/"

OPENAI_KEY = "sk-proj-SUPERSECRETopenaiKEY123456"
ANTHROPIC_KEY = "sk-ant-api03-SUPERSECRETanthropicKEY"


class Upstream:
    """httpx.MockTransport handler that records what it received."""

    def __init__(self, respond):
        self.respond = respond
        self.seen: list[httpx.Request] = []
        self.bodies: list[bytes] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(await request.aread())
        self.seen.append(request)
        result = self.respond(request)
        if asyncio.iscoroutine(result):
            result = await result
        return result


def make_app(tmp_path, respond, *, redaction=r.REDACTION_MASK, on_exchange=(), salt="s"):
    upstream = Upstream(respond)
    client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    app = create_capture_app(OPENAI, ANTHROPIC, redaction=redaction, root=tmp_path, salt=salt,
                             on_exchange=on_exchange, client=client)
    return app, upstream


def front(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://capture")


def captured(tmp_path):
    return list(r.iter_exchanges(tmp_path))


def all_files_text(root: Path) -> str:
    return "".join(p.read_text(errors="replace") for p in root.rglob("*") if p.is_file())


def sse(events):
    out = []
    for event in events:
        if isinstance(event, tuple):
            name, data = event
            out.append(f"event: {name}\ndata: {json.dumps(data)}\n\n")
        elif event == "[DONE]":
            out.append("data: [DONE]\n\n")
        else:
            out.append(f"data: {json.dumps(event)}\n\n")
    return [s.encode() for s in out]


def stream_response(chunks, status=200, headers=None):
    async def body():
        for chunk in chunks:
            yield chunk

    return httpx.Response(status, headers={"content-type": "text/event-stream", **(headers or {})}, content=body())


OPENAI_RESPONSE = {
    "id": "chatcmpl-1", "object": "chat.completion", "model": "gpt-4o-mini-2024-07-18",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Mail me at kush@example.com"},
                 "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 21, "completion_tokens": 7, "prompt_tokens_details": {"cached_tokens": 0}},
}


# ---------------------------------------------------------------------------
# Non-streaming
# ---------------------------------------------------------------------------


async def test_openai_non_stream_forwards_bytes_and_headers(tmp_path):
    app, upstream = make_app(tmp_path, lambda req: httpx.Response(
        200, json=OPENAI_RESPONSE, headers={"x-request-id": "req_123", "openai-processing-ms": "41"}))
    raw = b'{"model":  "gpt-4o-mini",\n "messages": [{"role": "user", "content": "hi kush@example.com"}]}'
    async with front(app) as client:
        resp = await client.post(
            "/v1/chat/completions?api-version=1", content=raw,
            headers={
                "Authorization": f"Bearer {OPENAI_KEY}",
                "OpenAI-Organization": "org-abc",
                "OpenAI-Project": "proj_abc",
                "X-Custom": "kept",
                "Content-Type": "application/json",
                "Connection": "keep-alive, x-drop-me",
                "X-Drop-Me": "hop",
                "Keep-Alive": "timeout=5",
            },
        )
    assert resp.status_code == 200
    assert resp.json() == OPENAI_RESPONSE
    assert resp.headers["x-request-id"] == "req_123"
    assert resp.headers["openai-processing-ms"] == "41"

    sent = upstream.seen[0]
    assert str(sent.url) == "https://api.openai.test/v1/chat/completions?api-version=1"
    assert upstream.bodies[0] == raw
    assert sent.headers["authorization"] == f"Bearer {OPENAI_KEY}"
    assert sent.headers["openai-organization"] == "org-abc"
    assert sent.headers["openai-project"] == "proj_abc"
    assert sent.headers["x-custom"] == "kept"
    assert sent.headers["host"] == "api.openai.test"
    for dropped in ("connection", "keep-alive", "x-drop-me"):
        assert dropped not in sent.headers

    (ex,) = captured(tmp_path)
    assert ex.provider == r.PROVIDER_OPENAI and ex.endpoint == "/v1/chat/completions"
    assert ex.model == "gpt-4o-mini" and ex.status == 200 and ex.ok and not ex.stream
    assert ex.usage.input_tokens == 21 and ex.usage.output_tokens == 7 and ex.usage.cached_input_tokens == 0
    assert ex.redaction == r.REDACTION_MASK
    assert ex.request["messages"][0]["content"] == "hi <EMAIL>"
    assert ex.response["choices"][0]["message"]["content"] == "Mail me at <EMAIL>"
    assert ex.latency_ms >= 0 and ex.ttft_ms is None


async def test_anthropic_non_stream_with_tool_use(tmp_path):
    response = {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-4-5",
        "content": [
            {"type": "text", "text": "Looking that up."},
            {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"email": "kush@example.com"}},
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 100, "output_tokens": 30, "cache_read_input_tokens": 50,
                  "cache_creation_input_tokens": 10},
    }
    app, upstream = make_app(tmp_path, lambda req: httpx.Response(200, json=response, headers={"request-id": "req_a"}))
    body = {"model": "claude-sonnet-4-5", "max_tokens": 256,
            "tools": [{"name": "lookup", "input_schema": {"type": "object"}}],
            "messages": [{"role": "user", "content": "find kush@example.com"}]}
    async with front(app) as client:
        resp = await client.post("/v1/messages", json=body, headers={
            "x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01",
            "anthropic-beta": "prompt-caching-2024-07-31"})
    assert resp.status_code == 200 and resp.json() == response
    assert resp.headers["request-id"] == "req_a"
    sent = upstream.seen[0]
    assert str(sent.url) == "https://api.anthropic.test/v1/messages"
    assert sent.headers["x-api-key"] == ANTHROPIC_KEY
    assert sent.headers["anthropic-version"] == "2023-06-01"
    assert sent.headers["anthropic-beta"] == "prompt-caching-2024-07-31"

    (ex,) = captured(tmp_path)
    assert ex.provider == r.PROVIDER_ANTHROPIC and ex.model == "claude-sonnet-4-5"
    assert ex.usage.input_tokens == 160 and ex.usage.cached_input_tokens == 50 and ex.usage.output_tokens == 30
    assert ex.response["content"][1]["input"] == {"email": "<EMAIL>"}
    norm = r.normalize(ex)
    assert norm.output_tool_calls[0]["name"] == "lookup" and norm.finish_reason == "tool_use"


async def test_gzip_response_passes_through_raw_and_is_still_captured(tmp_path):
    payload = gzip.compress(json.dumps(OPENAI_RESPONSE).encode())

    async def body():
        # Streamed like a real network response, so the proxy sees raw gzip.
        yield payload[:10]
        yield payload[10:]

    app, _ = make_app(tmp_path, lambda req: httpx.Response(
        200, content=body(), headers={"content-type": "application/json", "content-encoding": "gzip"}))
    async with front(app) as client:
        resp = await client.post("/v1/chat/completions", json={"model": "gpt-4o-mini", "messages": []},
                                 headers={"accept-encoding": "gzip"})
    assert resp.headers["content-encoding"] == "gzip"
    assert resp.json() == OPENAI_RESPONSE  # httpx decoded the untouched gzip bytes
    (ex,) = captured(tmp_path)
    assert ex.usage.complete and ex.response["id"] == "chatcmpl-1"


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


OPENAI_TOOL_STREAM = [
    {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o-2024-08-06",
     "choices": [{"index": 0, "delta": {"role": "assistant", "content": None}, "finish_reason": None}]},
    {"id": "c1", "choices": [{"index": 0, "delta": {"tool_calls": [
        {"index": 0, "id": "call_a", "type": "function", "function": {"name": "get_weather", "arguments": ""}},
    ]}}]},
    {"id": "c1", "choices": [{"index": 0, "delta": {"tool_calls": [
        {"index": 1, "id": "call_b", "type": "function", "function": {"name": "send_mail", "arguments": ""}},
    ]}}]},
    {"id": "c1", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"city": '}}]}}]},
    {"id": "c1", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 1, "function": {"arguments": '{"to": "kush@'}}]}}]},
    {"id": "c1", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"Paris"}'}}]}}]},
    {"id": "c1", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 1, "function": {"arguments": 'example.com"}'}}]}}]},
    {"id": "c1", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    {"id": "c1", "choices": [], "usage": {"prompt_tokens": 40, "completion_tokens": 22}},
    "[DONE]",
]


async def test_openai_stream_with_tool_calls_and_usage(tmp_path):
    # Split the SSE bytes at awkward places to exercise the parser.
    blob = b"".join(sse(OPENAI_TOOL_STREAM))
    chunks = [blob[i:i + 37] for i in range(0, len(blob), 37)]
    app, upstream = make_app(tmp_path, lambda req: stream_response(chunks, headers={"x-request-id": "s1"}))
    raw = json.dumps({"model": "gpt-4o", "stream": True, "stream_options": {"include_usage": True},
                      "messages": [{"role": "user", "content": "weather?"}]}).encode()
    async with front(app) as client:
        resp = await client.post("/v1/chat/completions", content=raw, headers={"authorization": f"Bearer {OPENAI_KEY}"})
    assert resp.status_code == 200
    assert resp.content == blob
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.headers["x-request-id"] == "s1"
    assert upstream.bodies[0] == raw

    (ex,) = captured(tmp_path)
    assert ex.stream and ex.ttft_ms is not None and ex.ttft_ms <= ex.latency_ms
    assert ex.usage.input_tokens == 40 and ex.usage.output_tokens == 22
    message = ex.response["choices"][0]["message"]
    assert message["content"] is None
    assert message["tool_calls"] == [
        {"id": "call_a", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}},
        {"id": "call_b", "type": "function", "function": {"name": "send_mail", "arguments": '{"to": "<EMAIL>"}'}},
    ]
    assert ex.response["choices"][0]["finish_reason"] == "tool_calls"
    assert ex.response["object"] == "chat.completion" and ex.response["model"] == "gpt-4o-2024-08-06"
    norm = r.normalize(ex)
    assert [c["name"] for c in norm.output_tool_calls] == ["get_weather", "send_mail"]


async def test_openai_stream_without_usage_is_not_modified_and_usage_is_missing(tmp_path):
    events = [
        {"id": "c2", "model": "gpt-4o", "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
        {"id": "c2", "choices": [{"index": 0, "delta": {"content": "Hel"}}]},
        {"id": "c2", "choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}]},
        "[DONE]",
    ]
    app, upstream = make_app(tmp_path, lambda req: stream_response(sse(events)))
    raw = b'{"model":"gpt-4o","stream":true,"messages":[{"role":"user","content":"hi"}]}'
    async with front(app) as client:
        resp = await client.post("/v1/chat/completions", content=raw)
    assert resp.content == b"".join(sse(events))
    assert upstream.bodies[0] == raw
    assert b"stream_options" not in upstream.bodies[0]
    (ex,) = captured(tmp_path)
    assert ex.response["choices"][0]["message"]["content"] == "Hello"
    assert not ex.usage.complete and ex.usage.input_tokens is None
    assert ex.ok


ANTHROPIC_STREAM = [
    ("message_start", {"type": "message_start", "message": {
        "id": "msg_s", "type": "message", "role": "assistant", "model": "claude-haiku-4-5", "content": [],
        "stop_reason": None, "usage": {"input_tokens": 55, "output_tokens": 1, "cache_read_input_tokens": 5}}}),
    ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
    ("ping", {"type": "ping"}),
    ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Sending to "}}),
    ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "kush@example.com"}}),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    ("content_block_start", {"type": "content_block_start", "index": 1, "content_block": {
        "type": "tool_use", "id": "toolu_9", "name": "send", "input": {}}}),
    ("content_block_delta", {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"to": '}}),
    ("content_block_delta", {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '"a"}'}}),
    ("content_block_stop", {"type": "content_block_stop", "index": 1}),
    ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                       "usage": {"output_tokens": 33}}),
    ("message_stop", {"type": "message_stop"}),
]


async def test_anthropic_stream_reassembles(tmp_path):
    chunks = sse(ANTHROPIC_STREAM)
    app, _ = make_app(tmp_path, lambda req: stream_response(chunks))
    async with front(app) as client:
        resp = await client.post("/v1/messages", json={"model": "claude-haiku-4-5", "stream": True, "max_tokens": 10,
                                                        "messages": [{"role": "user", "content": "hi"}]},
                                 headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01"})
    assert resp.content == b"".join(chunks)
    (ex,) = captured(tmp_path)
    assert ex.stream and ex.ttft_ms is not None
    assert ex.response["content"] == [
        {"type": "text", "text": "Sending to <EMAIL>"},
        {"type": "tool_use", "id": "toolu_9", "name": "send", "input": {"to": "a"}},
    ]
    assert ex.response["stop_reason"] == "tool_use" and ex.response["model"] == "claude-haiku-4-5"
    assert ex.usage.input_tokens == 60 and ex.usage.output_tokens == 33 and ex.usage.cached_input_tokens == 5


async def test_anthropic_stream_error_event_is_recorded(tmp_path):
    events = ANTHROPIC_STREAM[:4] + [("error", {"type": "error", "error": {"type": "overloaded_error", "message": "x"}})]
    app, _ = make_app(tmp_path, lambda req: stream_response(sse(events)))
    async with front(app) as client:
        resp = await client.post("/v1/messages", json={"model": "claude-haiku-4-5", "stream": True, "messages": []})
    assert resp.status_code == 200
    (ex,) = captured(tmp_path)
    assert ex.error == "stream error: overloaded_error" and not ex.ok


async def test_stream_chunks_reach_client_before_upstream_finishes(tmp_path):
    """The proxy must not buffer: chunk 1 is sent before chunk 2 exists."""
    gate = asyncio.Event()
    first, second = sse([
        {"id": "c", "model": "m", "choices": [{"index": 0, "delta": {"content": "A"}}]},
        {"id": "c", "choices": [{"index": 0, "delta": {"content": "B"}, "finish_reason": "stop"}]},
    ])

    async def body():
        yield first
        await asyncio.wait_for(gate.wait(), timeout=5)
        yield second

    app, _ = make_app(tmp_path, lambda req: httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body()))
    sent_bodies: list[bytes] = []
    got_first = asyncio.Event()
    payload = b'{"model":"m","stream":true,"messages":[]}'
    delivered = False

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": payload, "more_body": False}
        await asyncio.Event().wait()

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            sent_bodies.append(message["body"])
            got_first.set()

    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
        "method": "POST", "scheme": "http", "path": "/v1/chat/completions", "raw_path": b"/v1/chat/completions",
        "query_string": b"", "root_path": "", "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 5000), "server": ("127.0.0.1", 8790),
    }
    task = asyncio.create_task(app(scope, receive, send))
    await asyncio.wait_for(got_first.wait(), timeout=5)
    assert sent_bodies == [first]
    gate.set()
    await asyncio.wait_for(task, timeout=5)
    assert sent_bodies == [first, second]
    (ex,) = captured(tmp_path)
    assert ex.response["choices"][0]["message"]["content"] == "AB"
    assert ex.ttft_ms is not None and ex.ttft_ms < ex.latency_ms


def test_sse_parser_and_openai_assembler_units():
    parser = SSEParser()
    events = parser.feed("event: x\r\ndata: {\"a\":\r\n")
    assert events == []
    events = parser.feed("data: 1}\r\n\r\n: comment\n\ndata: [DONE]\n\n")
    assert events == [("x", '{"a":\n1}'), (None, "[DONE]")]
    assembler = OpenAIStreamAssembler()
    assert assembler.result() is None
    assembler.event(None, json.dumps({"id": "z", "choices": [
        {"index": 1, "delta": {"content": "second"}}, {"index": 0, "delta": {"content": "first"}}]}))
    out = assembler.result()
    assert [c["message"]["content"] for c in out["choices"]] == ["first", "second"]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


async def test_error_passthrough_and_logging(tmp_path):
    error = {"error": {"type": "rate_limit_error", "message": "Slow down kush@example.com"}}
    app, _ = make_app(tmp_path, lambda req: httpx.Response(429, json=error, headers={"retry-after": "7"}))
    async with front(app) as client:
        resp = await client.post("/v1/messages", json={"model": "claude-x", "stream": True, "messages": []})
    assert resp.status_code == 429 and resp.json() == error and resp.headers["retry-after"] == "7"
    (ex,) = captured(tmp_path)
    assert ex.status == 429 and not ex.ok
    assert ex.error == "upstream 429: rate_limit_error"
    assert ex.response["error"]["message"] == "Slow down <EMAIL>"


async def test_openai_error_with_code_only(tmp_path):
    error = {"error": {"message": "Incorrect API key provided: sk-proj-abcdefghijk", "code": "invalid_api_key"}}
    app, _ = make_app(tmp_path, lambda req: httpx.Response(401, json=error))
    async with front(app) as client:
        resp = await client.post("/v1/chat/completions", json={"model": "gpt-4o", "messages": []})
    assert resp.status_code == 401 and resp.json() == error
    (ex,) = captured(tmp_path)
    assert ex.error == "upstream 401: invalid_api_key"
    assert "sk-proj-abcdefghijk" not in all_files_text(tmp_path)


async def test_unreachable_upstream_is_502_and_logged(tmp_path):
    def boom(req):
        raise httpx.ConnectError("refused", request=req)

    app, _ = make_app(tmp_path, boom)
    async with front(app) as client:
        resp = await client.post("/v1/chat/completions", json={"model": "gpt-4o", "messages": []})
    assert resp.status_code == 502 and resp.json()["error"]["type"] == "upstream_unreachable"
    (ex,) = captured(tmp_path)
    assert ex.status == 502 and ex.error == "upstream unreachable: ConnectError"


# ---------------------------------------------------------------------------
# Secrets and redaction modes
# ---------------------------------------------------------------------------


async def test_auth_headers_are_never_persisted_or_logged(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)

    def respond(req):
        if req.url.path.endswith("/messages"):
            return stream_response(sse(ANTHROPIC_STREAM))
        return httpx.Response(200, json=OPENAI_RESPONSE)

    failing = []

    async def bad_callback(exchange):
        failing.append(exchange)
        raise RuntimeError("callback broke")

    app, _ = make_app(tmp_path, respond, redaction=r.REDACTION_OFF, on_exchange=[bad_callback])
    async with front(app) as client:
        await client.post("/v1/chat/completions", json={"model": "gpt-4o", "messages": []},
                          headers={"Authorization": f"Bearer {OPENAI_KEY}", "OpenAI-Organization": "org-SECRETORG"})
        await client.post("/v1/messages", json={"model": "claude", "stream": True, "messages": []},
                          headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01",
                                   "Proxy-Authorization": "Basic SECRETPROXY"})
        await client.get("/v1/models", headers={"Authorization": f"Bearer {OPENAI_KEY}"})
    await app.state.capture.drain()
    assert len(captured(tmp_path)) == 2 and len(failing) == 2
    stored = all_files_text(tmp_path)
    for secret in (OPENAI_KEY, ANTHROPIC_KEY, "SECRETORG", "SECRETPROXY", "Bearer"):
        assert secret not in stored
        assert secret not in caplog.text
    assert '"headers"' not in stored


async def test_hash_mode_is_stable_within_a_capture(tmp_path):
    app, _ = make_app(tmp_path, lambda req: httpx.Response(200, json=OPENAI_RESPONSE), redaction=r.REDACTION_HASH)
    async with front(app) as client:
        for _ in range(2):
            await client.post("/v1/chat/completions", json={
                "model": "gpt-4o", "messages": [{"role": "user", "content": "I am kush@example.com"}]})
    first, second = captured(tmp_path)
    text = first.request["messages"][0]["content"]
    assert text.startswith("I am <EMAIL:") and text == second.request["messages"][0]["content"]
    assert first.redaction == r.REDACTION_HASH
    assert "kush@example.com" not in all_files_text(tmp_path)


async def test_default_salt_is_random_per_capture(tmp_path):
    outs = []
    for i in range(2):
        app, _ = make_app(tmp_path / str(i), lambda req: httpx.Response(200, json=OPENAI_RESPONSE),
                          redaction=r.REDACTION_HASH, salt=None)
        async with front(app) as client:
            await client.post("/v1/chat/completions", json={
                "model": "gpt-4o", "messages": [{"role": "user", "content": "kush@example.com"}]})
        outs.append(captured(tmp_path / str(i))[0].request["messages"][0]["content"])
    assert outs[0] != outs[1]


async def test_metadata_mode_stores_no_bodies(tmp_path):
    app, _ = make_app(tmp_path, lambda req: httpx.Response(200, json=OPENAI_RESPONSE), redaction=r.REDACTION_METADATA)
    async with front(app) as client:
        resp = await client.post("/v1/chat/completions", json={
            "model": "gpt-4o", "messages": [{"role": "user", "content": "very private prompt text"}]})
    assert resp.json() == OPENAI_RESPONSE
    (ex,) = captured(tmp_path)
    assert ex.request is None and ex.response is None and not ex.has_bodies
    assert ex.model == "gpt-4o" and ex.usage.complete and ex.redaction == r.REDACTION_METADATA
    stored = all_files_text(tmp_path)
    assert "very private" not in stored and "Mail me" not in stored


async def test_off_mode_stores_bodies_verbatim(tmp_path):
    app, _ = make_app(tmp_path, lambda req: httpx.Response(200, json=OPENAI_RESPONSE), redaction=r.REDACTION_OFF)
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "kush@example.com"}]}
    async with front(app) as client:
        await client.post("/v1/chat/completions", json=body)
    (ex,) = captured(tmp_path)
    assert ex.request == body and ex.response == OPENAI_RESPONSE


def test_bad_redaction_mode_is_refused(tmp_path):
    with pytest.raises(ValueError):
        create_capture_app(OPENAI, ANTHROPIC, redaction="nope", root=tmp_path)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("streaming", [False, True])
async def test_slow_and_failing_callbacks_do_not_touch_the_response(tmp_path, streaming):
    release = asyncio.Event()
    seen = []

    async def slow(exchange):
        await release.wait()
        seen.append(("slow", exchange.id))

    async def failing(exchange):
        raise RuntimeError("shadow exploded")

    async def recorder(exchange):
        seen.append(("ok", exchange.id))

    def respond(req):
        if streaming:
            return stream_response(sse(OPENAI_TOOL_STREAM))
        return httpx.Response(200, json=OPENAI_RESPONSE)

    app, _ = make_app(tmp_path, respond, on_exchange=[slow, failing, recorder])
    baseline_app, _ = make_app(tmp_path / "base", respond)
    body = {"model": "gpt-4o", "stream": streaming, "messages": [{"role": "user", "content": "hi"}]}
    async with front(baseline_app) as client:
        expected = await client.post("/v1/chat/completions", json=body)

    started = time.perf_counter()
    async with front(app) as client:
        resp = await asyncio.wait_for(client.post("/v1/chat/completions", json=body), timeout=2)
    elapsed = time.perf_counter() - started
    assert elapsed < 1.0
    assert resp.status_code == expected.status_code and resp.content == expected.content
    assert dict(resp.headers) == dict(expected.headers)

    for _ in range(50):
        if app.state.capture.callback_errors and seen:
            break
        await asyncio.sleep(0.01)
    state = app.state.capture
    assert state.callback_errors == 1 and state.written == 1
    (ex,) = captured(tmp_path)
    assert seen == [("ok", ex.id)]
    assert len(state.tasks) == 1  # the slow one is still running
    release.set()
    await state.drain()
    assert seen[-1] == ("slow", ex.id)


async def test_callback_receives_redacted_exchange(tmp_path):
    got = []

    async def cb(exchange):
        got.append(exchange)

    app, _ = make_app(tmp_path, lambda req: httpx.Response(200, json=OPENAI_RESPONSE), on_exchange=[cb])
    async with front(app) as client:
        await client.post("/v1/chat/completions", json={"model": "gpt-4o", "messages": [
            {"role": "user", "content": "kush@example.com"}]})
    await app.state.capture.drain()
    assert got[0].request["messages"][0]["content"] == "<EMAIL>"


# ---------------------------------------------------------------------------
# Pass-through routes
# ---------------------------------------------------------------------------


async def test_passthrough_routes_pick_upstream_by_header_and_are_not_logged(tmp_path):
    def respond(req):
        return httpx.Response(200, json={"host": req.url.host, "path": req.url.path}, headers={"x-up": "1"})

    app, upstream = make_app(tmp_path, respond)
    async with front(app) as client:
        models = await client.get("/v1/models", headers={"Authorization": f"Bearer {OPENAI_KEY}"})
        claude_models = await client.get("/v1/models", headers={"x-api-key": ANTHROPIC_KEY})
        emb = await client.post("/v1/embeddings", json={"model": "text-embedding-3-small", "input": "kush@example.com"})
        count = await client.post("/v1/messages/count_tokens", json={"model": "claude", "messages": []},
                                  headers={"anthropic-version": "2023-06-01"})
        other = await client.delete("/v1/files/file-1")
        health = await client.get("/health")
    assert models.json() == {"host": "api.openai.test", "path": "/v1/models"}
    assert models.headers["x-up"] == "1"
    assert claude_models.json()["host"] == "api.anthropic.test"
    assert emb.json() == {"host": "api.openai.test", "path": "/v1/embeddings"}
    assert count.json() == {"host": "api.anthropic.test", "path": "/v1/messages/count_tokens"}
    assert other.json()["path"] == "/v1/files/file-1" and upstream.seen[-1].method == "DELETE"
    assert upstream.seen[0].headers["authorization"] == f"Bearer {OPENAI_KEY}"
    assert upstream.seen[1].headers["x-api-key"] == ANTHROPIC_KEY
    assert health.json()["status"] == "ok" and health.json()["redaction"] == r.REDACTION_MASK
    assert captured(tmp_path) == []


async def test_passthrough_error_status_is_kept(tmp_path):
    app, _ = make_app(tmp_path, lambda req: httpx.Response(404, json={"error": "nope"}))
    async with front(app) as client:
        resp = await client.get("/v1/models/unknown")
    assert resp.status_code == 404 and resp.json() == {"error": "nope"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_parser_accepts_capture_and_stats(tmp_path):
    parser = build_parser()
    args = parser.parse_args(["capture", "--port", "9999", "--redact", "metadata", "--dir", str(tmp_path),
                              "--openai-upstream", "http://localhost:1/v1"])
    assert args.command == "capture" and args.port == 9999 and args.redact == "metadata"
    assert args.capture_action is None and args.dir == tmp_path
    args = parser.parse_args(["capture", "stats", "--dir", str(tmp_path)])
    assert args.capture_action == "stats" and args.dir == tmp_path
    args = parser.parse_args(["capture", "--dir", str(tmp_path), "stats"])
    assert args.capture_action == "stats" and args.dir == tmp_path
    defaults = parser.parse_args(["capture"])
    assert defaults.host == "127.0.0.1" and defaults.port == 8790 and defaults.redact == "mask"


def test_startup_panel_says_where_and_how(tmp_path):
    from throttle.style import Style

    args = build_parser().parse_args(["capture", "--dir", str(tmp_path), "--redact", "hash"])
    text = startup_panel(args, tmp_path, Style(colour=False, unicode=False, width=160))
    assert str(tmp_path / "capture") in text
    assert "hash" in text
    assert "Nothing is sent anywhere except the upstream you already call" in text
    assert "OPENAI_BASE_URL=http://127.0.0.1:8790/v1" in text
    assert "ANTHROPIC_BASE_URL=http://127.0.0.1:8790" in text


async def test_stats_command(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("COLUMNS", "160")

    def respond(req):
        if req.url.path == "/v1/messages":
            return httpx.Response(200, json={"model": "claude-x", "content": [], "usage": {"input_tokens": 3, "output_tokens": 1}})
        if b"nousage" in req.content:
            return httpx.Response(200, json={"choices": []})
        return httpx.Response(200, json=OPENAI_RESPONSE)

    app, _ = make_app(tmp_path, respond)
    async with front(app) as client:
        await client.post("/v1/chat/completions", json={"model": "gpt-4o", "messages": []})
        await client.post("/v1/chat/completions", json={"model": "gpt-4o", "messages": [], "user": "nousage"})
        await client.post("/v1/messages", json={"model": "claude-x", "messages": []})
    stats = compute_stats(tmp_path)
    assert stats.total == 3 and stats.with_usage == 2
    assert stats.by_provider == {"openai": 2, "anthropic": 1}
    assert stats.by_model[("openai", "gpt-4o")] == 2

    args = build_parser().parse_args(["capture", "stats", "--dir", str(tmp_path)])
    assert handle(args) == 0
    out = capsys.readouterr().out
    assert "gpt-4o" in out and "claude-x" in out
    assert "2 of 3 (67%)" in out and "mask 3" in out
    assert stats.first_ts[:4] in out


def test_stats_on_empty_dir(tmp_path, capsys):
    args = build_parser().parse_args(["capture", "stats", "--dir", str(tmp_path)])
    assert handle(args) == 0
    assert "No captured calls" in capsys.readouterr().out


async def test_client_disconnect_mid_stream_is_still_recorded(tmp_path):
    never = asyncio.Event()
    first = sse([{"id": "c", "model": "m", "choices": [{"index": 0, "delta": {"content": "A"}}]}])[0]

    async def body():
        yield first
        await asyncio.wait_for(never.wait(), timeout=5)
        yield b"unreachable"

    app, _ = make_app(tmp_path, lambda req: httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body()))
    got_first = asyncio.Event()
    stage = 0

    async def receive():
        nonlocal stage
        stage += 1
        if stage == 1:
            return {"type": "http.request", "body": b'{"model":"m","stream":true}', "more_body": False}
        await got_first.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            got_first.set()

    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
        "method": "POST", "scheme": "http", "path": "/v1/chat/completions", "raw_path": b"/v1/chat/completions",
        "query_string": b"", "root_path": "", "headers": [], "client": ("127.0.0.1", 5000), "server": ("127.0.0.1", 8790),
    }
    started = time.perf_counter()
    await asyncio.wait_for(app(scope, receive, send), timeout=4)
    await app.state.capture.drain()
    assert time.perf_counter() - started < 3
    (ex,) = captured(tmp_path)
    assert ex.error == "client disconnected" and ex.response["choices"][0]["message"]["content"] == "A"


# ---------------------------------------------------------------------------
# Review fixes: redaction off the event loop, inline images dropped
# ---------------------------------------------------------------------------


async def test_redaction_runs_off_the_event_loop(tmp_path, monkeypatch):
    import threading

    from throttle.own import capture as cap

    threads = []
    real = cap.redact_body

    def spy(*args, **kwargs):
        threads.append(threading.current_thread())
        return real(*args, **kwargs)

    monkeypatch.setattr(cap, "redact_body", spy)
    app, _ = make_app(tmp_path, lambda req: httpx.Response(200, json=OPENAI_RESPONSE))
    async with front(app) as client:
        await client.post("/v1/chat/completions", json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})
        await app.state.capture.drain()
    assert threads and all(t is not threading.main_thread() for t in threads)
    (ex,) = captured(tmp_path)
    assert ex.response["choices"][0]["message"]["content"] == "Mail me at <EMAIL>"


@pytest.mark.parametrize("mode", [r.REDACTION_MASK, r.REDACTION_HASH])
async def test_inline_image_payloads_are_not_stored(tmp_path, mode):
    payload = "iVBORw0KGgo" + "A" * 5000
    app, upstream = make_app(tmp_path, lambda req: httpx.Response(200, json=OPENAI_RESPONSE), redaction=mode)
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "what is this? kush@example.com"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{payload}"}},
    ]}]}
    async with front(app) as client:
        await client.post("/v1/chat/completions", json=body)
    (ex,) = captured(tmp_path)
    parts = ex.request["messages"][0]["content"]
    assert parts[1] == {"type": "image_url", "image_url": {"url": "<IMAGE>"}}
    assert "<EMAIL" in parts[0]["text"]
    assert payload not in all_files_text(tmp_path)
    assert payload.encode() in upstream.bodies[0]          # the upstream still gets the image
    assert r.normalize(ex).has_images


async def test_off_mode_keeps_inline_images(tmp_path):
    app, _ = make_app(tmp_path, lambda req: httpx.Response(200, json=OPENAI_RESPONSE), redaction=r.REDACTION_OFF)
    url = "data:image/png;base64,iVBORw0KGgoAAAA"
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}]}
    async with front(app) as client:
        await client.post("/v1/chat/completions", json=body)
    (ex,) = captured(tmp_path)
    assert ex.request["messages"][0]["content"][0]["image_url"]["url"] == url
