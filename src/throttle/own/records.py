"""Shared record format for the own-your-intelligence loop.

One captured API call is one :class:`Exchange`. Every later stage (cluster,
shadow, evaluate, report, route, refresh) reads exchanges through this module
so the on-disk format has exactly one owner.

Storage is append-only JSONL, one file per UTC month, under a local directory
(default ``~/.throttle/own``)::

    <root>/capture/2026-10.jsonl     Exchange records
    <root>/shadow/<slug>.jsonl       ShadowResult records, one file per candidate
    <root>/evals/<run_id>.json       evaluation results
    <root>/routes.json               approved routes
    <root>/ledger/2026-10.jsonl      realized-savings ledger

Bodies are stored *after* redaction. ``redaction`` records which mode was
applied so later stages can refuse to build eval sets from metadata-only
captures.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

SCHEMA_VERSION = 1

PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDERS = (PROVIDER_OPENAI, PROVIDER_ANTHROPIC)

# Redaction modes, from most to least content kept.
REDACTION_OFF = "off"            # bodies stored verbatim
REDACTION_MASK = "mask"          # PII replaced with typed placeholders, e.g. <EMAIL>
REDACTION_HASH = "hash"          # PII replaced with a salted hash, stable within a capture
REDACTION_METADATA = "metadata"  # no bodies at all, only sizes, usage, timing
REDACTION_MODES = (REDACTION_OFF, REDACTION_MASK, REDACTION_HASH, REDACTION_METADATA)

# Workload types. ``throttle.own.workloads.classify(exchange, normalized)``
# returns one of these.
WORKLOAD_TYPES = ("classification", "extraction", "summarization", "agent", "code", "chat")

SERVED_API = "api"
SERVED_OWNED = "owned"


def default_root() -> Path:
    override = os.environ.get("THROTTLE_OWN_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".throttle" / "own"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@dataclass
class Usage:
    """Token counts exactly as reported by the upstream. ``None`` means the
    upstream did not report it; it is never estimated."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None

    @property
    def complete(self) -> bool:
        return self.input_tokens is not None and self.output_tokens is not None


@dataclass
class Exchange:
    """One request/response pair seen by the capture proxy."""

    provider: str
    endpoint: str
    model: str
    request: dict[str, Any] | None
    response: dict[str, Any] | None
    status: int
    latency_ms: float
    usage: Usage = field(default_factory=Usage)
    stream: bool = False
    ttft_ms: float | None = None
    redaction: str = REDACTION_MASK
    served_by: str = SERVED_API
    fallback: bool = False
    route_reason: str | None = None
    workload_type: str | None = None
    error: str | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    ts: str = field(default_factory=utc_now_iso)
    v: int = SCHEMA_VERSION

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "Exchange":
        payload = dict(data)
        usage = payload.pop("usage", None) or {}
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        payload = {k: v for k, v in payload.items() if k in known}
        return cls(usage=Usage(**{k: usage.get(k) for k in ("input_tokens", "output_tokens", "cached_input_tokens")}), **payload)

    @property
    def has_bodies(self) -> bool:
        return self.redaction != REDACTION_METADATA and self.request is not None

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300 and self.error is None


@dataclass
class ShadowResult:
    """A candidate model's answer to a captured request. Never served."""

    exchange_id: str
    candidate: str              # operator-facing label, e.g. "qwen2.5-72b@vllm"
    candidate_model: str
    output_text: str | None
    tool_calls: list[dict[str, Any]]
    usage: Usage
    latency_ms: float
    status: int
    finish_reason: str | None = None
    error: str | None = None
    ts: str = field(default_factory=utc_now_iso)
    v: int = SCHEMA_VERSION

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "ShadowResult":
        payload = dict(data)
        usage = payload.pop("usage", None) or {}
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        payload = {k: v for k, v in payload.items() if k in known}
        return cls(usage=Usage(**{k: usage.get(k) for k in ("input_tokens", "output_tokens", "cached_input_tokens")}), **payload)

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300 and self.error is None and self.output_text is not None


# ---------------------------------------------------------------------------
# JSONL storage
# ---------------------------------------------------------------------------


def _month_file(directory: Path, ts: str) -> Path:
    return directory / f"{ts[:7]}.jsonl"


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, separators=(",", ":"), ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield records; a truncated final line (crash mid-write) is skipped."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record


def capture_dir(root: Path | None = None) -> Path:
    return (root or default_root()) / "capture"


def append_exchange(exchange: Exchange, root: Path | None = None) -> Path:
    path = _month_file(capture_dir(root), exchange.ts)
    append_jsonl(path, exchange.to_json())
    return path


def iter_exchanges(root: Path | None = None, *, since: datetime | None = None) -> Iterator[Exchange]:
    directory = capture_dir(root)
    if not directory.is_dir():
        return
    for path in sorted(directory.glob("*.jsonl")):
        for record in read_jsonl(path):
            try:
                exchange = Exchange.from_json(record)
            except (TypeError, ValueError):
                continue
            if since is not None and parse_iso(exchange.ts) < since:
                continue
            yield exchange


_SLUG = re.compile(r"[^a-zA-Z0-9._-]+")


def slug(label: str) -> str:
    cleaned = _SLUG.sub("-", label).strip("-")
    return cleaned[:80] or "candidate"


def shadow_path(candidate: str, root: Path | None = None) -> Path:
    return (root or default_root()) / "shadow" / f"{slug(candidate)}.jsonl"


def append_shadow(result: ShadowResult, root: Path | None = None) -> Path:
    path = shadow_path(result.candidate, root)
    append_jsonl(path, result.to_json())
    return path


def iter_shadow(candidate: str, root: Path | None = None) -> Iterator[ShadowResult]:
    for record in read_jsonl(shadow_path(candidate, root)):
        try:
            yield ShadowResult.from_json(record)
        except (TypeError, ValueError):
            continue


# ---------------------------------------------------------------------------
# Provider-neutral view of a request/response
# ---------------------------------------------------------------------------


@dataclass
class Message:
    role: str           # system | user | assistant | tool
    text: str           # concatenated text parts; non-text parts become placeholders
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Normalized:
    """What every stage needs to know about one exchange, independent of the
    provider wire format."""

    messages: list[Message]
    tools: list[dict[str, Any]]           # OpenAI function-tool shape
    response_format: str | None           # None | "json" | "json_schema" | "text"
    json_schema: dict[str, Any] | None
    max_tokens: int | None
    temperature: float | None
    output_text: str | None
    output_tool_calls: list[dict[str, Any]]  # [{"name": str, "arguments": str}]
    finish_reason: str | None
    has_images: bool = False

    @property
    def system_text(self) -> str:
        return "\n".join(m.text for m in self.messages if m.role == "system")

    @property
    def last_user_text(self) -> str:
        for message in reversed(self.messages):
            if message.role == "user":
                return message.text
        return ""

    @property
    def turns(self) -> int:
        return sum(1 for m in self.messages if m.role == "user")


def _text_from_openai_content(content: Any) -> tuple[str, bool]:
    if content is None:
        return "", False
    if isinstance(content, str):
        return content, False
    parts: list[str] = []
    images = False
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, Mapping):
                continue
            kind = part.get("type")
            if kind == "text":
                parts.append(str(part.get("text", "")))
            elif kind in {"image_url", "input_image", "image"}:
                images = True
                parts.append("<IMAGE>")
            elif kind in {"input_audio", "audio"}:
                parts.append("<AUDIO>")
    return "\n".join(parts), images


def _text_from_anthropic_content(content: Any) -> tuple[str, list[dict[str, Any]], bool]:
    if content is None:
        return "", [], False
    if isinstance(content, str):
        return content, [], False
    parts: list[str] = []
    calls: list[dict[str, Any]] = []
    images = False
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, Mapping):
                continue
            kind = block.get("type")
            if kind == "text":
                parts.append(str(block.get("text", "")))
            elif kind == "tool_use":
                calls.append({
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "arguments": json.dumps(block.get("input", {}), sort_keys=True),
                })
            elif kind == "tool_result":
                inner, _, inner_images = _text_from_anthropic_content(block.get("content"))
                images = images or inner_images
                parts.append(inner)
            elif kind in {"image", "document"}:
                images = True
                parts.append("<IMAGE>" if kind == "image" else "<DOCUMENT>")
            elif kind in {"thinking", "redacted_thinking"}:
                continue
    return "\n".join(p for p in parts if p), calls, images


def _openai_tool_calls(raw: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return calls
    for call in raw:
        if not isinstance(call, Mapping):
            continue
        function = call.get("function") or {}
        calls.append({
            "id": call.get("id"),
            "name": function.get("name"),
            "arguments": function.get("arguments") if isinstance(function.get("arguments"), str) else json.dumps(function.get("arguments", {}), sort_keys=True),
        })
    return calls


def normalize_openai(request: Mapping[str, Any] | None, response: Mapping[str, Any] | None) -> Normalized:
    request = request or {}
    messages: list[Message] = []
    has_images = False
    for raw in request.get("messages") or []:
        if not isinstance(raw, Mapping):
            continue
        text, images = _text_from_openai_content(raw.get("content"))
        has_images = has_images or images
        role = str(raw.get("role", "user"))
        if role == "developer":
            role = "system"
        messages.append(Message(role=role, text=text, tool_calls=_openai_tool_calls(raw.get("tool_calls"))))
    response_format = None
    json_schema = None
    fmt = request.get("response_format")
    if isinstance(fmt, Mapping):
        response_format = fmt.get("type")
        if response_format == "json_object":
            response_format = "json"
        if response_format == "json_schema":
            schema_wrapper = fmt.get("json_schema") or {}
            json_schema = schema_wrapper.get("schema") if isinstance(schema_wrapper, Mapping) else None
    tools = [t for t in (request.get("tools") or []) if isinstance(t, Mapping)]
    output_text = None
    output_calls: list[dict[str, Any]] = []
    finish = None
    if response:
        choices = response.get("choices") or []
        if choices and isinstance(choices[0], Mapping):
            message = choices[0].get("message") or {}
            output_text, _ = _text_from_openai_content(message.get("content"))
            output_calls = _openai_tool_calls(message.get("tool_calls"))
            finish = choices[0].get("finish_reason")
    max_tokens = request.get("max_completion_tokens", request.get("max_tokens"))
    return Normalized(
        messages=messages,
        tools=[dict(t) for t in tools],
        response_format=response_format,
        json_schema=json_schema,
        max_tokens=max_tokens if isinstance(max_tokens, int) else None,
        temperature=request.get("temperature") if isinstance(request.get("temperature"), (int, float)) else None,
        output_text=output_text,
        output_tool_calls=output_calls,
        finish_reason=finish,
        has_images=has_images,
    )


def normalize_anthropic(request: Mapping[str, Any] | None, response: Mapping[str, Any] | None) -> Normalized:
    request = request or {}
    messages: list[Message] = []
    has_images = False
    system = request.get("system")
    if system:
        text, _, images = _text_from_anthropic_content(system)
        has_images = has_images or images
        messages.append(Message(role="system", text=text))
    for raw in request.get("messages") or []:
        if not isinstance(raw, Mapping):
            continue
        text, calls, images = _text_from_anthropic_content(raw.get("content"))
        has_images = has_images or images
        role = str(raw.get("role", "user"))
        content = raw.get("content")
        if role == "user" and isinstance(content, list) and content and all(
            isinstance(b, Mapping) and b.get("type") == "tool_result" for b in content
        ):
            role = "tool"
        messages.append(Message(role=role, text=text, tool_calls=calls))
    tools = []
    for tool in request.get("tools") or []:
        if not isinstance(tool, Mapping) or "name" not in tool:
            continue
        tools.append({
            "type": "function",
            "function": {
                "name": tool.get("name"),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object"}),
            },
        })
    output_text = None
    output_calls: list[dict[str, Any]] = []
    finish = None
    if response:
        output_text, output_calls, _ = _text_from_anthropic_content(response.get("content"))
        finish = response.get("stop_reason")
    max_tokens = request.get("max_tokens")
    return Normalized(
        messages=messages,
        tools=tools,
        response_format=None,
        json_schema=None,
        max_tokens=max_tokens if isinstance(max_tokens, int) else None,
        temperature=request.get("temperature") if isinstance(request.get("temperature"), (int, float)) else None,
        output_text=output_text,
        output_tool_calls=output_calls,
        finish_reason=finish,
        has_images=has_images,
    )


def normalize(exchange: Exchange) -> Normalized:
    if exchange.provider == PROVIDER_ANTHROPIC:
        return normalize_anthropic(exchange.request, exchange.response)
    return normalize_openai(exchange.request, exchange.response)


def usage_from_openai(response: Mapping[str, Any] | None) -> Usage:
    raw = (response or {}).get("usage") or {}
    if not isinstance(raw, Mapping):
        return Usage()
    details = raw.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens") if isinstance(details, Mapping) else None
    return Usage(
        input_tokens=_int_or_none(raw.get("prompt_tokens")),
        output_tokens=_int_or_none(raw.get("completion_tokens")),
        cached_input_tokens=_int_or_none(cached),
    )


def usage_from_anthropic(response: Mapping[str, Any] | None) -> Usage:
    raw = (response or {}).get("usage") or {}
    if not isinstance(raw, Mapping):
        return Usage()
    base = _int_or_none(raw.get("input_tokens"))
    created = _int_or_none(raw.get("cache_creation_input_tokens")) or 0
    read = _int_or_none(raw.get("cache_read_input_tokens"))
    total_input = None if base is None else base + created + (read or 0)
    return Usage(
        input_tokens=total_input,
        output_tokens=_int_or_none(raw.get("output_tokens")),
        cached_input_tokens=read,
    )


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def to_openai_chat_request(norm: Normalized, *, model: str, max_tokens: int | None = None) -> dict[str, Any]:
    """Translate any captured request into an OpenAI chat request for a
    self-hosted candidate (vLLM, SGLang, Ollama and LMDeploy all speak it)."""
    messages: list[dict[str, Any]] = []
    for message in norm.messages:
        if message.role == "tool":
            messages.append({"role": "user", "content": message.text})
            continue
        entry: dict[str, Any] = {"role": message.role, "content": message.text}
        if message.tool_calls and message.role == "assistant":
            entry["tool_calls"] = [
                {"id": c.get("id") or f"call_{i}", "type": "function",
                 "function": {"name": c.get("name"), "arguments": c.get("arguments") or "{}"}}
                for i, c in enumerate(message.tool_calls)
            ]
            # Tool results are flattened to user text above, so keep the
            # transcript valid for servers that require matching tool ids.
            entry["content"] = message.text or json.dumps(
                [{"tool": c.get("name"), "arguments": c.get("arguments")} for c in message.tool_calls]
            )
            entry.pop("tool_calls")
        messages.append(entry)
    body: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
    if norm.tools:
        body["tools"] = norm.tools
    if norm.response_format == "json":
        body["response_format"] = {"type": "json_object"}
    elif norm.response_format == "json_schema" and norm.json_schema is not None:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "output", "schema": norm.json_schema}}
    limit = max_tokens if max_tokens is not None else norm.max_tokens
    if limit is not None:
        body["max_tokens"] = limit
    if norm.temperature is not None:
        body["temperature"] = norm.temperature
    return body


def iter_with_bodies(exchanges: Sequence[Exchange]) -> Iterator[tuple[Exchange, Normalized]]:
    """Successful exchanges that still carry content, normalized."""
    for exchange in exchanges:
        if exchange.ok and exchange.has_bodies:
            yield exchange, normalize(exchange)
