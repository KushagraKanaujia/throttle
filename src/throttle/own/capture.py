"""``throttle capture``: a pass-through proxy that records API traffic locally.

Clients point their OpenAI or Anthropic base URL at this proxy. Every request
goes to the upstream the client already uses, byte for byte, with the same
headers; the upstream answer comes back unchanged. Streaming is a true
pass-through: each upstream chunk is sent to the client as it arrives, and a
copy is reassembled on the side so the stored record looks like a
non-streaming response.

After a response finishes, one :class:`records.Exchange` is written (bodies
redacted per ``--redact``) and each ``on_exchange`` callback is started as a
fire-and-forget task. Auth headers are never written anywhere. Nothing is
cached. Nothing is sent anywhere except the upstream.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import secrets
import threading
import time
import zlib
from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from . import records
from .redact import redact_body

log = logging.getLogger("throttle.own.capture")

ExchangeCallback = Callable[[records.Exchange], Awaitable[Any]]

DEFAULT_PORT = 8790
DEFAULT_OPENAI_UPSTREAM = "https://api.openai.com"
DEFAULT_ANTHROPIC_UPSTREAM = "https://api.anthropic.com"

HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "proxy-connection", "te", "trailer", "trailers", "transfer-encoding", "upgrade",
})
_DROP_REQUEST = HOP_BY_HOP | {"host", "content-length"}
_DROP_RESPONSE = HOP_BY_HOP | {"content-length"}
_PASSTHROUGH_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]


# ---------------------------------------------------------------------------
# Header handling
# ---------------------------------------------------------------------------


def _connection_listed(raw: Iterable[tuple[bytes, bytes]]) -> set[str]:
    listed: set[str] = set()
    for name, value in raw:
        if name.lower() == b"connection":
            listed.update(t.strip().lower() for t in value.decode("latin-1").split(",") if t.strip())
    return listed


def forward_request_headers(raw: Sequence[tuple[bytes, bytes]]) -> list[tuple[bytes, bytes]]:
    drop = _DROP_REQUEST | _connection_listed(raw)
    return [(k, v) for k, v in raw if k.decode("latin-1").lower() not in drop]


def forward_response_headers(raw: Sequence[tuple[bytes, bytes]]) -> list[tuple[bytes, bytes]]:
    drop = _DROP_RESPONSE | _connection_listed(raw)
    return [(k.lower(), v) for k, v in raw if k.decode("latin-1").lower() not in drop]


def _strip_base(url: str) -> str:
    url = url.rstrip("/")
    return url[:-3] if url.endswith("/v1") else url


# ---------------------------------------------------------------------------
# Decoding the tee copy (the client always gets the raw bytes)
# ---------------------------------------------------------------------------


class _Decoder:
    """Decode a Content-Encoding for the captured copy. ``None`` result means
    the encoding is not supported here; the client is unaffected."""

    def __init__(self, encoding: str | None) -> None:
        self.encoding = (encoding or "identity").strip().lower()
        self.failed = False
        self._obj: Any = None
        if self.encoding in {"identity", ""}:
            return
        if self.encoding in {"gzip", "x-gzip"}:
            self._obj = zlib.decompressobj(16 + zlib.MAX_WBITS)
        elif self.encoding == "deflate":
            self._obj = zlib.decompressobj()
        elif self.encoding == "br":
            try:
                import brotli  # type: ignore[import-not-found]

                self._obj = brotli.Decompressor()
            except ImportError:
                self.failed = True
        elif self.encoding == "zstd":
            try:
                from compression import zstd  # type: ignore[import-not-found]

                self._obj = zstd.ZstdDecompressor()
            except ImportError:
                self.failed = True
        else:
            self.failed = True

    def feed(self, data: bytes) -> bytes:
        if self.failed:
            return b""
        if self._obj is None:
            return data
        try:
            if hasattr(self._obj, "decompress"):
                return self._obj.decompress(data)
            return self._obj.process(data)
        except Exception:  # noqa: BLE001 - a broken tee never touches the client
            self.failed = True
            return b""


# ---------------------------------------------------------------------------
# Server-sent events reassembly
# ---------------------------------------------------------------------------


class SSEParser:
    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, text: str) -> list[tuple[str | None, str]]:
        self._buffer += text.replace("\r\n", "\n").replace("\r", "\n")
        events: list[tuple[str | None, str]] = []
        while "\n\n" in self._buffer:
            block, self._buffer = self._buffer.split("\n\n", 1)
            event = _parse_block(block)
            if event is not None:
                events.append(event)
        return events

    def flush(self) -> list[tuple[str | None, str]]:
        block, self._buffer = self._buffer, ""
        event = _parse_block(block)
        return [event] if event is not None else []


def _parse_block(block: str) -> tuple[str | None, str] | None:
    name: str | None = None
    data: list[str] = []
    for line in block.split("\n"):
        if not line or line.startswith(":"):
            continue
        key, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if key == "event":
            name = value
        elif key == "data":
            data.append(value)
    if not data and name is None:
        return None
    return name, "\n".join(data)


class OpenAIStreamAssembler:
    """Rebuild a ``chat.completion`` from ``chat.completion.chunk`` deltas."""

    def __init__(self) -> None:
        self.meta: dict[str, Any] = {}
        self.choices: dict[int, dict[str, Any]] = {}
        self.usage: dict[str, Any] | None = None
        self.error: str | None = None
        self.saw_content = False
        self.done = False

    def event(self, name: str | None, data: str) -> bool:
        """Feed one event; return True if it carried content."""
        if data.strip() == "[DONE]":
            self.done = True
            return False
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            return False
        if not isinstance(chunk, dict):
            return False
        if "error" in chunk and "choices" not in chunk:
            self.error = _error_label(chunk) or "stream error"
            return False
        for key in ("id", "created", "model", "system_fingerprint", "service_tier"):
            if chunk.get(key) is not None:
                self.meta.setdefault(key, chunk[key])
        if isinstance(chunk.get("usage"), dict):
            self.usage = chunk["usage"]
        content = False
        for raw in chunk.get("choices") or []:
            if not isinstance(raw, dict):
                continue
            index = raw.get("index", 0)
            choice = self.choices.setdefault(index, {"role": None, "content": [], "refusal": [], "tool_calls": {}, "finish_reason": None})
            delta = raw.get("delta") or {}
            if delta.get("role"):
                choice["role"] = delta["role"]
            if isinstance(delta.get("content"), str) and delta["content"]:
                choice["content"].append(delta["content"])
                content = True
            if isinstance(delta.get("refusal"), str) and delta["refusal"]:
                choice["refusal"].append(delta["refusal"])
                content = True
            for call in delta.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                slot = choice["tool_calls"].setdefault(call.get("index", 0), {"id": None, "type": "function", "name": [], "arguments": []})
                if call.get("id"):
                    slot["id"] = call["id"]
                if call.get("type"):
                    slot["type"] = call["type"]
                function = call.get("function") or {}
                if function.get("name"):
                    slot["name"].append(function["name"])
                if function.get("arguments"):
                    slot["arguments"].append(function["arguments"])
                content = True
            if raw.get("finish_reason"):
                choice["finish_reason"] = raw["finish_reason"]
        self.saw_content = self.saw_content or content
        return content

    def result(self) -> dict[str, Any] | None:
        if not self.meta and not self.choices:
            return None
        choices = []
        for index in sorted(self.choices):
            choice = self.choices[index]
            message: dict[str, Any] = {
                "role": choice["role"] or "assistant",
                "content": "".join(choice["content"]) if choice["content"] else None,
            }
            if choice["refusal"]:
                message["refusal"] = "".join(choice["refusal"])
            if choice["tool_calls"]:
                message["tool_calls"] = [
                    {"id": slot["id"], "type": slot["type"],
                     "function": {"name": "".join(slot["name"]), "arguments": "".join(slot["arguments"])}}
                    for _, slot in sorted(choice["tool_calls"].items())
                ]
            choices.append({"index": index, "message": message, "finish_reason": choice["finish_reason"]})
        out: dict[str, Any] = {"object": "chat.completion", **self.meta, "choices": choices}
        if self.usage is not None:
            out["usage"] = self.usage
        return out


class AnthropicStreamAssembler:
    """Rebuild a Messages API response from its stream events."""

    def __init__(self) -> None:
        self.message: dict[str, Any] = {}
        self.blocks: dict[int, dict[str, Any]] = {}
        self.partial_json: dict[int, list[str]] = {}
        self.usage: dict[str, Any] = {}
        self.error: str | None = None
        self.saw_content = False
        self.done = False

    def event(self, name: str | None, data: str) -> bool:
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            return False
        if not isinstance(payload, dict):
            return False
        kind = payload.get("type") or name
        if kind == "message_start":
            message = dict(payload.get("message") or {})
            self.usage.update(message.pop("usage", None) or {})
            message.pop("content", None)
            self.message = message
        elif kind == "content_block_start":
            index = payload.get("index", len(self.blocks))
            self.blocks[index] = dict(payload.get("content_block") or {})
        elif kind == "content_block_delta":
            index = payload.get("index", 0)
            block = self.blocks.setdefault(index, {"type": "text", "text": ""})
            delta = payload.get("delta") or {}
            dtype = delta.get("type")
            if dtype == "text_delta":
                block["text"] = block.get("text", "") + delta.get("text", "")
            elif dtype == "input_json_delta":
                self.partial_json.setdefault(index, []).append(delta.get("partial_json", ""))
            elif dtype == "thinking_delta":
                block["thinking"] = block.get("thinking", "") + delta.get("thinking", "")
            elif dtype == "signature_delta":
                block["signature"] = block.get("signature", "") + delta.get("signature", "")
            elif dtype == "citations_delta" and delta.get("citation") is not None:
                block.setdefault("citations", []).append(delta["citation"])
            self.saw_content = True
            return True
        elif kind == "message_delta":
            delta = payload.get("delta") or {}
            for key in ("stop_reason", "stop_sequence"):
                if key in delta:
                    self.message[key] = delta[key]
            self.usage.update({k: v for k, v in (payload.get("usage") or {}).items() if v is not None})
        elif kind == "message_stop":
            self.done = True
        elif kind == "error":
            self.error = _error_label(payload) or "stream error"
        return False

    def result(self) -> dict[str, Any] | None:
        if not self.message and not self.blocks:
            return None
        content = []
        for index in sorted(self.blocks):
            block = dict(self.blocks[index])
            if index in self.partial_json:
                raw = "".join(self.partial_json[index])
                try:
                    block["input"] = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    block["input"] = raw
            content.append(block)
        out = {**self.message, "content": content}
        if self.usage:
            out["usage"] = dict(self.usage)
        return out


def _error_label(body: Any) -> str | None:
    if not isinstance(body, Mapping):
        return None
    err = body.get("error")
    if isinstance(err, Mapping):
        label = err.get("type") or err.get("code")
        return str(label) if label else "error"
    if isinstance(err, str):
        return "error"
    return None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


@dataclass
class CaptureState:
    redaction: str
    root: Path
    salt: bytes
    callbacks: tuple[ExchangeCallback, ...]
    written: int = 0
    write_errors: int = 0
    callback_errors: int = 0
    tasks: set[asyncio.Task[Any]] = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def _append(self, exchange: records.Exchange) -> None:
        with self.lock:
            records.append_exchange(exchange, self.root)

    async def record(self, exchange: records.Exchange) -> None:
        try:
            await asyncio.to_thread(self._append, exchange)
            self.written += 1
        except Exception:  # noqa: BLE001 - capture must never break serving
            self.write_errors += 1
            log.exception("could not write captured exchange %s", exchange.id)
        for callback in self.callbacks:
            self._spawn(self._run_callback(callback, exchange))

    async def _run_callback(self, callback: ExchangeCallback, exchange: records.Exchange) -> None:
        try:
            await callback(exchange)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            self.callback_errors += 1
            log.exception("on_exchange callback %r failed", getattr(callback, "__name__", callback))

    def _spawn(self, coro: Awaitable[Any]) -> asyncio.Task[Any]:
        task = asyncio.ensure_future(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    async def drain(self) -> None:
        """Wait for pending writes and callbacks (tests, shutdown)."""
        while self.tasks:
            await asyncio.gather(*list(self.tasks), return_exceptions=True)


@dataclass
class _Pending:
    provider: str
    endpoint: str
    request_json: dict[str, Any] | None
    started: float
    stream: bool
    status: int = 0
    response_json: dict[str, Any] | None = None
    ttft_ms: float | None = None
    error: str | None = None
    finished: float | None = None
    recorded: bool = False


def create_capture_app(
    openai_upstream: str,
    anthropic_upstream: str,
    *,
    redaction: str,
    root: Path | str,
    salt: str | bytes | None = None,
    on_exchange: Sequence[ExchangeCallback] = (),
    client: httpx.AsyncClient | None = None,
) -> FastAPI:
    if redaction not in records.REDACTION_MODES:
        raise ValueError(f"unknown redaction mode {redaction!r}")
    salt_bytes = salt.encode("utf-8") if isinstance(salt, str) else (salt or secrets.token_bytes(32))
    state = CaptureState(redaction=redaction, root=Path(root), salt=salt_bytes, callbacks=tuple(on_exchange))
    upstreams = {
        records.PROVIDER_OPENAI: _strip_base(openai_upstream),
        records.PROVIDER_ANTHROPIC: _strip_base(anthropic_upstream),
    }
    owns_client = client is None
    http = client or httpx.AsyncClient(
        timeout=httpx.Timeout(connect=15.0, read=600.0, write=60.0, pool=15.0),
        follow_redirects=False,
        trust_env=False,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await state.drain()
            if owns_client:
                await http.aclose()

    app = FastAPI(title="Throttle capture", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.capture = state

    def build_exchange(p: _Pending) -> records.Exchange:
        finished = p.finished or time.perf_counter()
        request = p.request_json if isinstance(p.request_json, dict) else None
        response = p.response_json if isinstance(p.response_json, dict) else None
        model = ""
        if request and isinstance(request.get("model"), str):
            model = request["model"]
        elif response and isinstance(response.get("model"), str):
            model = response["model"]
        if p.provider == records.PROVIDER_ANTHROPIC:
            usage = records.usage_from_anthropic(response)
        else:
            usage = records.usage_from_openai(response)
        error = p.error
        if error is None and not 200 <= p.status < 300:
            error = f"upstream {p.status}: {_error_label(response) or 'error'}"
        return records.Exchange(
            provider=p.provider,
            endpoint=p.endpoint,
            model=model,
            request=redact_body(request, state.redaction, state.salt) if request is not None else None,
            response=redact_body(response, state.redaction, state.salt) if response is not None else None,
            status=p.status,
            latency_ms=round((finished - p.started) * 1000, 3),
            usage=usage,
            stream=p.stream,
            ttft_ms=None if p.ttft_ms is None else round(p.ttft_ms, 3),
            redaction=state.redaction,
            error=error,
        )

    async def finalize(p: _Pending) -> None:
        if p.recorded:
            return
        p.recorded = True
        try:
            exchange = build_exchange(p)
        except Exception:  # noqa: BLE001
            state.write_errors += 1
            log.exception("could not build captured exchange")
            return
        await state.record(exchange)

    def upstream_url(provider: str, request: Request) -> str:
        url = upstreams[provider] + request.url.path
        if request.url.query:
            url += "?" + request.url.query
        return url

    async def open_upstream(provider: str, request: Request, body: bytes) -> httpx.Response:
        upstream_request = httpx.Request(
            request.method,
            upstream_url(provider, request),
            headers=forward_request_headers(request.headers.raw),
            content=body,
        )
        return await http.send(upstream_request, stream=True)

    def unreachable(exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=502,
            content={"error": {"type": "upstream_unreachable", "message": f"throttle capture could not reach the upstream: {type(exc).__name__}"}},
        )

    async def captured(provider: str, request: Request) -> Response:
        body = await request.body()
        started = time.perf_counter()
        try:
            parsed = json.loads(body) if body else None
        except (json.JSONDecodeError, UnicodeDecodeError):
            parsed = None
        request_json = parsed if isinstance(parsed, dict) else None
        wants_stream = bool(request_json and request_json.get("stream") is True)
        pending = _Pending(provider=provider, endpoint=request.url.path, request_json=request_json, started=started, stream=wants_stream)
        try:
            upstream = await open_upstream(provider, request, body)
        except httpx.HTTPError as exc:
            pending.status = 502
            pending.error = f"upstream unreachable: {type(exc).__name__}"
            pending.finished = time.perf_counter()
            return _with_background(unreachable(exc), finalize, pending)

        pending.status = upstream.status_code
        headers = _response_headers(upstream)
        content_type = upstream.headers.get("content-type", "")
        decoder = _Decoder(None if upstream.is_stream_consumed else upstream.headers.get("content-encoding"))

        if "text/event-stream" not in content_type:
            chunks: list[bytes] = []
            try:
                async for chunk in _raw_stream(upstream):
                    chunks.append(chunk)
            except httpx.HTTPError as exc:
                pending.status = 502
                pending.error = f"upstream read failed: {type(exc).__name__}"
                pending.finished = time.perf_counter()
                return _with_background(unreachable(exc), finalize, pending)
            finally:
                await upstream.aclose()
            raw = b"".join(chunks)
            pending.finished = time.perf_counter()
            decoded = decoder.feed(raw)
            if not decoder.failed:
                try:
                    value = json.loads(decoded) if decoded else None
                    pending.response_json = value if isinstance(value, dict) else None
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
            response = Response(content=raw, status_code=upstream.status_code)
            _replace_headers(response, headers, keep_length=True)
            response.background = BackgroundTask(finalize, pending)
            return response

        assembler: OpenAIStreamAssembler | AnthropicStreamAssembler
        assembler = AnthropicStreamAssembler() if provider == records.PROVIDER_ANTHROPIC else OpenAIStreamAssembler()
        sse = SSEParser()
        text_decoder = _utf8_incremental()
        pending.stream = True

        def tee(chunk: bytes) -> None:
            if decoder.failed:
                return
            try:
                text = text_decoder.decode(decoder.feed(chunk))
                for name, data in sse.feed(text):
                    if assembler.event(name, data) and pending.ttft_ms is None:
                        pending.ttft_ms = (time.perf_counter() - started) * 1000
            except Exception:  # noqa: BLE001 - the tee never breaks the client stream
                log.exception("stream reassembly failed")

        async def relay() -> AsyncIterator[bytes]:
            completed = False
            try:
                async for chunk in _raw_stream(upstream):
                    tee(chunk)
                    yield chunk
                completed = True
            except httpx.HTTPError as exc:
                pending.error = f"upstream stream broke: {type(exc).__name__}"
                raise
            finally:
                pending.finished = time.perf_counter()
                await upstream.aclose()
                try:
                    for name, data in sse.feed(text_decoder.decode(b"", final=True)) + sse.flush():
                        assembler.event(name, data)
                except Exception:  # noqa: BLE001
                    pass
                pending.response_json = assembler.result()
                if pending.error is None:
                    if assembler.error:
                        pending.error = f"stream error: {assembler.error}"
                    elif not completed:
                        pending.error = "client disconnected"
                if not completed:
                    # Background tasks may not run after a disconnect.
                    state._spawn(finalize(pending))

        response = StreamingResponse(relay(), status_code=upstream.status_code)
        _replace_headers(response, headers, keep_length=False)
        response.background = BackgroundTask(finalize, pending)
        return response

    async def passthrough(request: Request) -> Response:
        headers = request.headers
        provider = (
            records.PROVIDER_ANTHROPIC
            if "x-api-key" in headers or "anthropic-version" in headers
            else records.PROVIDER_OPENAI
        )
        body = await request.body()
        try:
            upstream = await open_upstream(provider, request, body)
        except httpx.HTTPError as exc:
            return unreachable(exc)

        async def relay() -> AsyncIterator[bytes]:
            try:
                async for chunk in _raw_stream(upstream):
                    yield chunk
            finally:
                await upstream.aclose()

        response = StreamingResponse(relay(), status_code=upstream.status_code)
        _replace_headers(response, _response_headers(upstream), keep_length=False)
        return response

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "redaction": state.redaction,
            "captured": state.written,
            "write_errors": state.write_errors,
            "callback_errors": state.callback_errors,
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await captured(records.PROVIDER_OPENAI, request)

    @app.post("/v1/messages")
    async def messages(request: Request) -> Response:
        return await captured(records.PROVIDER_ANTHROPIC, request)

    @app.api_route("/v1/{path:path}", methods=_PASSTHROUGH_METHODS)
    async def other(request: Request, path: str) -> Response:  # noqa: ARG001
        return await passthrough(request)

    return app


def _raw_stream(response: httpx.Response) -> AsyncIterator[bytes]:
    """The upstream bytes exactly as received. An in-memory response (a test
    or demo transport) may already be read; its decoded content is used."""
    if response.is_stream_consumed:
        async def once() -> AsyncIterator[bytes]:
            yield response.content

        return once()
    return response.aiter_raw()


def _response_headers(response: httpx.Response) -> list[tuple[bytes, bytes]]:
    headers = forward_response_headers(response.headers.raw)
    if response.is_stream_consumed:
        headers = [(k, v) for k, v in headers if k != b"content-encoding"]
    return headers


def _utf8_incremental() -> Any:
    import codecs

    return codecs.getincrementaldecoder("utf-8")(errors="replace")


def _replace_headers(response: Response, headers: list[tuple[bytes, bytes]], *, keep_length: bool) -> None:
    kept = [(k, v) for k, v in response.raw_headers if keep_length and k == b"content-length"]
    response.raw_headers[:] = kept + headers


def _with_background(response: Response, func: Callable[..., Awaitable[Any]], *args: Any) -> Response:
    response.background = BackgroundTask(func, *args)
    return response


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "capture",
        help="record your OpenAI/Anthropic API traffic locally (pass-through proxy)",
        description=(
            "Run a local pass-through proxy in front of the OpenAI and Anthropic APIs. "
            "Every call goes to the upstream you already use, unchanged; a redacted copy "
            "is written to a local folder. Nothing is cached and nothing is sent anywhere else."
        ),
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--openai-upstream", default=DEFAULT_OPENAI_UPSTREAM)
    parser.add_argument("--anthropic-upstream", default=DEFAULT_ANTHROPIC_UPSTREAM)
    parser.add_argument(
        "--redact",
        choices=records.REDACTION_MODES,
        default=records.REDACTION_MASK,
        help="off: keep bodies; mask: <EMAIL>-style placeholders; hash: placeholders with a salted hash; "
        "metadata: keep no bodies (default: mask)",
    )
    parser.add_argument("--dir", type=Path, default=None, help="data folder (default: ~/.throttle/own or $THROTTLE_OWN_DIR)")
    actions = parser.add_subparsers(dest="capture_action", metavar="{stats}")
    stats = actions.add_parser("stats", help="summarize what has been captured")
    # SUPPRESS keeps `throttle capture --dir X stats` working too.
    stats.add_argument("--dir", type=Path, default=argparse.SUPPRESS, help="data folder (default: ~/.throttle/own or $THROTTLE_OWN_DIR)")


def handle(args: argparse.Namespace) -> int:
    root = Path(args.dir).expanduser() if getattr(args, "dir", None) else records.default_root()
    if getattr(args, "capture_action", None) == "stats":
        print(render_stats(compute_stats(root), root))
        return 0
    return _serve(args, root)


def _serve(args: argparse.Namespace, root: Path) -> int:
    import os

    import uvicorn

    from .. import style

    salt = os.environ.get("THROTTLE_OWN_SALT") or None
    app = create_capture_app(args.openai_upstream, args.anthropic_upstream, redaction=args.redact, root=root, salt=salt)
    print(startup_panel(args, root, style.get()), flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


_REDACTION_WORDS = {
    records.REDACTION_OFF: "off (bodies stored as sent)",
    records.REDACTION_MASK: "mask (emails, phones, SSNs, cards, IPs, secrets become <EMAIL>-style placeholders)",
    records.REDACTION_HASH: "hash (like mask, with a salted hash so repeats can be matched)",
    records.REDACTION_METADATA: "metadata (no bodies stored, only model, usage and timing)",
}


def startup_panel(args: argparse.Namespace, root: Path, st: Any) -> str:
    base = f"http://{args.host}:{args.port}"
    rows: list[Any] = [
        ("Data is written to " + str(records.capture_dir(root)), ()),
        ("Redaction: " + _REDACTION_WORDS[args.redact], ()),
        ("Redaction is pattern-based. It does not catch names or street addresses.", (), ""),
        "",
        (f"OpenAI calls go to {_strip_base(args.openai_upstream)}", ()),
        (f"Anthropic calls go to {_strip_base(args.anthropic_upstream)}", ()),
        ("Nothing is sent anywhere except the upstream you already call. Nothing is cached.", ()),
        "",
        "Point your clients here:",
        f"  OPENAI_BASE_URL={base}/v1",
        f"  ANTHROPIC_BASE_URL={base}",
        "",
        st.dim("Stop with Ctrl+C. See what you have: throttle capture stats"),
    ]
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        rows.insert(3, (f"Listening on {args.host}: anyone who can reach this port can send traffic through it.", ()))
    return st.panel(rows, title=f"Throttle capture on {base}")


@dataclass
class CaptureStats:
    total: int = 0
    ok: int = 0
    errors: int = 0
    streamed: int = 0
    with_usage: int = 0
    first_ts: str | None = None
    last_ts: str | None = None
    by_provider: Counter[str] = field(default_factory=Counter)
    by_model: Counter[tuple[str, str]] = field(default_factory=Counter)
    by_redaction: Counter[str] = field(default_factory=Counter)


def compute_stats(root: Path) -> CaptureStats:
    stats = CaptureStats()
    for exchange in records.iter_exchanges(root):
        stats.total += 1
        stats.ok += exchange.ok
        stats.errors += not exchange.ok
        stats.streamed += exchange.stream
        stats.with_usage += exchange.usage.complete
        stats.by_provider[exchange.provider] += 1
        stats.by_model[(exchange.provider, exchange.model or "(unknown)")] += 1
        stats.by_redaction[exchange.redaction] += 1
        if stats.first_ts is None or exchange.ts < stats.first_ts:
            stats.first_ts = exchange.ts
        if stats.last_ts is None or exchange.ts > stats.last_ts:
            stats.last_ts = exchange.ts
    return stats


def render_stats(stats: CaptureStats, root: Path) -> str:
    from .. import style

    st = style.get()
    if stats.total == 0:
        return st.panel(
            [(f"No captured calls in {records.capture_dir(root)} yet.", ()), "Start one with: throttle capture"],
            title="Throttle capture stats",
        )
    share = stats.with_usage / stats.total
    rows: list[Any] = [
        (f"Folder: {records.capture_dir(root)}", ()),
        (f"Window: {stats.first_ts} to {stats.last_ts}", ()),
        f"Calls: {stats.total} ({stats.ok} ok, {stats.errors} errors, {stats.streamed} streamed)",
        (
            f"With token usage: {stats.with_usage} of {stats.total} ({share:.0%}). "
            "Calls without usage are left out of cost math, never estimated.",
            (),
        ),
        "Redaction: " + ", ".join(f"{mode} {n}" for mode, n in sorted(stats.by_redaction.items())),
        "",
        st.bold("By provider"),
    ]
    rows += [f"  {provider:<10} {n}" for provider, n in stats.by_provider.most_common()]
    rows += ["", st.bold("By model")]
    rows += [f"  {provider:<10} {model:<36} {n}" for (provider, model), n in stats.by_model.most_common()]
    return st.panel(rows, title="Throttle capture stats")
