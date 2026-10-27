"""Stage 6, ROUTE: serve approved workload types from the model you own.

``throttle route approve`` writes ``<root>/routes.json``::

    {"<type>": {"candidate_label", "url", "model", "api_key_env",
                "approved_at", "approved_eval_run_id", "eval_verdict",
                "mode": "live" | "canary", "canary_rate", "forced"}}

Approval is refused unless the latest ``throttle eval`` run for the candidate
gave that type BETTER or NO MATERIAL DIFFERENCE. ``--force`` overrides the
refusal; the route is then stored with ``forced: true`` and shown as FORCED in
``route list``, in the capture startup output and in the savings statement.

``throttle capture --route`` hands every chat/messages request to
:class:`Router` before forwarding it:

1. The request is typed with ``workloads.classify`` from the request alone
   (there is no response yet). Request-only typing can differ from what
   ``throttle workloads`` says after the fact: for example "classification"
   needs a short answer, which before the call can only be read from a small
   ``max_tokens``.
2. If the type has a route (and, in canary mode, the request is sampled), the
   request is translated with ``records.to_openai_chat_request`` and sent to
   the owned model.
3. The owned answer is translated back to the client's wire format: an OpenAI
   chat completion, or an Anthropic message with tool calls as ``tool_use``
   blocks. The response's ``model`` is the owned model and the header
   ``x-throttle-served-by: owned`` is set.
4. Anything doubtful falls back to the API: the original request is forwarded
   unchanged. Fallback reasons (``Exchange.route_reason``, a ``": detail"``
   may follow the code): ``router_error``, ``config_error``,
   ``connect_error``, ``timeout``, ``http_error``, ``bad_response``,
   ``truncated``, ``empty_output``, ``invalid_json``, ``schema_invalid``,
   ``unknown_tool``, ``invalid_tool_arguments``.

Limitations (MVP):

* Streaming is not streamed. When the client asks for a stream and the owned
  model serves it, the owned model is called without streaming and the full
  answer is replayed as a short SSE stream in the client's format. Time to
  first token is therefore the owned model's full answer time.
* Every routed request waits for the owned call first. A failed or doubtful
  owned answer adds the owned latency (up to ``timeout``) before the API call.
* Requests with images, ``n`` > 1, or no messages are never routed.
* Low-confidence checks are cheap and explicit only: JSON is checked when the
  request set ``response_format``; a schema the validator does not support is
  not checked; a prompt that merely asks for JSON in words is not checked.
* Anthropic-only request features the translation drops (``thinking``,
  ``cache_control``, ``tool_choice``) are not sent to the owned model.

Realized-savings ledger: ``<root>/ledger/YYYY-MM.jsonl``, append-only, one
line per owned-served request: exchange id, type, incumbent model, owned
tokens, and the serve-time list price with no cache discount (an upper bound,
kept for audit). ``throttle route savings`` builds the bill number from it:

* Avoided cost prices the owned input + output tokens at the incumbent's list
  price (the incumbent's own counts are unknown). Each type's share of input
  that the API served from prompt cache, measured from that month's
  API-served captures of the same type and model, is priced at the cached
  rate. No cache history means all input is priced at the cached rate.
* Only routes approved on a passing eval are billable. Forced routes are
  shown on their own line and never billed.
* Owned cost is GPU $/hr x GPUs x every hour of the month (always-on), or an
  operator-reported invoice. Net <= 0 means no savings, no bill.
"""

from __future__ import annotations

import argparse
import calendar
import json
import logging
import os
import random
import sys
import threading
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import httpx

from . import pricing as pricing_mod
from . import records
from . import workloads
from .evaluate import BETTER, NO_DIFF, SCHEMA_ID as EVAL_SCHEMA_ID, TRUNCATED_REASONS, _tool_names, validate_schema
from .shadow import chat_url
from ..evalcompare import parse_json_answer

log = logging.getLogger("throttle.own.route")

MODE_LIVE = "live"
MODE_CANARY = "canary"
PASSING = (BETTER, NO_DIFF)
DEFAULT_TIMEOUT = 30.0
LEDGER_SCHEMA = "throttle.own.ledger/1"
SAVINGS_SCHEMA = "throttle.own.savings/1"
EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 2
BOLD_RED = "1;31"

FALLBACK_REASONS = (
    "router_error", "config_error", "connect_error", "timeout", "http_error", "bad_response",
    "truncated", "empty_output", "invalid_json", "schema_invalid", "unknown_tool", "invalid_tool_arguments",
)

ASSUMPTION_TOKENS = (
    "The API model's own token counts for these requests are not known. Avoided cost prices the "
    "owned model's input + output tokens at the API model's list price."
)
ASSUMPTION_CACHE = (
    "Prompt caching: for each workload type and API model, the cache-hit share is measured from this "
    "month's API-served captured calls (cached input tokens / input tokens), and that share of the "
    "avoided input tokens is priced at the cached-input rate. With no cache history for a type, all "
    "avoided input is priced at the cached rate (no cache history, priced conservatively)."
)
ASSUMPTION_FORCED = (
    "Routes approved with --force have no passing eval: their avoided cost is shown but not billable."
)
CACHE_MEASURED = "measured cache share"
CACHE_NONE = "no cache history, priced conservatively"
CACHE_NO_RATE = "no published cached rate, input priced at base rate"
IN_PROGRESS = "This month is not over: owned cost counts the whole month, avoided cost only the requests so far."
ASSUMPTION_OWNED = "Owned cost assumes the GPUs ran all hours of the period (always-on)."


# ---------------------------------------------------------------------------
# Route table
# ---------------------------------------------------------------------------


@dataclass
class Route:
    workload_type: str
    candidate_label: str
    url: str
    model: str
    api_key_env: str | None = None
    approved_at: str = ""
    approved_eval_run_id: str | None = None
    eval_verdict: str | None = None
    mode: str = MODE_LIVE
    canary_rate: float | None = None
    forced: bool = False

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("workload_type")
        return data

    @classmethod
    def from_json(cls, workload_type: str, data: Mapping[str, Any]) -> "Route":
        known = set(cls.__dataclass_fields__) - {"workload_type"}  # type: ignore[attr-defined]
        return cls(workload_type=workload_type, **{k: v for k, v in data.items() if k in known})


def routes_path(root: Path) -> Path:
    return root / "routes.json"


def load_routes(root: Path) -> dict[str, Route]:
    path = routes_path(root)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected an object keyed by workload type")
    return {kind: Route.from_json(kind, entry) for kind, entry in data.items() if isinstance(entry, Mapping)}


def save_routes(root: Path, routes: Mapping[str, Route]) -> None:
    path = routes_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps({k: r.to_json() for k, r in sorted(routes.items())}, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def latest_eval(root: Path, label: str) -> dict[str, Any] | None:
    """The newest ``throttle eval`` document for ``label`` (by its ``ts``)."""
    best: dict[str, Any] | None = None
    for path in (root / "evals").glob("*.json"):
        if path.name.startswith("evalset-"):
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(doc, dict) or doc.get("schema") != EVAL_SCHEMA_ID or doc.get("candidate") != label:
            continue
        if best is None or str(doc.get("ts", "")) > str(best.get("ts", "")):
            best = doc
    return best


@dataclass
class ApprovalCheck:
    ok: bool
    run_id: str | None
    verdict: str | None
    reason: str


def check_approval(root: Path, workload_type: str, label: str, model: str) -> ApprovalCheck:
    doc = latest_eval(root, label)
    if doc is None:
        return ApprovalCheck(False, None, None, f"no eval run found for {label}. Run: throttle eval --candidate {label}")
    run_id = doc.get("run_id")
    entry = (doc.get("types") or {}).get(workload_type)
    if not isinstance(entry, Mapping):
        return ApprovalCheck(False, run_id, None, f"latest eval run {run_id} for {label} has no result for type {workload_type}")
    verdict = entry.get("verdict")
    if verdict not in PASSING:
        return ApprovalCheck(False, run_id, verdict,
                             f"latest eval run {run_id} gave {workload_type} the verdict {verdict}; "
                             f"approval needs {BETTER} or {NO_DIFF}")
    evaluated_model = doc.get("candidate_model")
    if evaluated_model and evaluated_model != model:
        return ApprovalCheck(False, run_id, verdict,
                             f"latest eval run {run_id} measured model {evaluated_model}, not {model}")
    return ApprovalCheck(True, run_id, verdict, f"latest eval run {run_id} gave {workload_type} {verdict}")


def approve(root: Path, workload_type: str, label: str, url: str, model: str, *, api_key_env: str | None = None,
            canary: float | None = None, force: bool = False) -> tuple[Route | None, ApprovalCheck]:
    check = check_approval(root, workload_type, label, model)
    if not check.ok and not force:
        return None, check
    route = Route(
        workload_type=workload_type, candidate_label=label, url=url, model=model, api_key_env=api_key_env,
        approved_at=records.utc_now_iso(), approved_eval_run_id=check.run_id, eval_verdict=check.verdict,
        mode=MODE_CANARY if canary is not None else MODE_LIVE, canary_rate=canary, forced=not check.ok,
    )
    routes = load_routes(root)
    routes[workload_type] = route
    save_routes(root, routes)
    return route, check


def revoke(root: Path, workload_type: str) -> bool:
    routes = load_routes(root)
    if workload_type not in routes:
        return False
    del routes[workload_type]
    save_routes(root, routes)
    return True


# ---------------------------------------------------------------------------
# Translating the owned answer to the client's wire format
# ---------------------------------------------------------------------------


@dataclass
class OwnedAnswer:
    text: str
    tool_calls: list[dict[str, Any]]   # [{"id", "name", "arguments": str}]
    finish_reason: str | None
    usage: records.Usage
    model: str


def openai_body(answer: OwnedAnswer) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": answer.text or None}
    if answer.tool_calls:
        message["tool_calls"] = [
            {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
            for c in answer.tool_calls
        ]
    finish = "tool_calls" if answer.tool_calls else (answer.finish_reason or "stop")
    body: dict[str, Any] = {
        "id": f"chatcmpl-throttle-{uuid.uuid4().hex[:24]}", "object": "chat.completion",
        "created": int(time.time()), "model": answer.model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
    }
    if answer.usage.complete:
        body["usage"] = {
            "prompt_tokens": answer.usage.input_tokens, "completion_tokens": answer.usage.output_tokens,
            "total_tokens": (answer.usage.input_tokens or 0) + (answer.usage.output_tokens or 0),
        }
    return body


_ANTHROPIC_STOP = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens"}


def anthropic_body(answer: OwnedAnswer) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if answer.text:
        content.append({"type": "text", "text": answer.text})
    for call in answer.tool_calls:
        content.append({"type": "tool_use", "id": call["id"], "name": call["name"],
                        "input": json.loads(call["arguments"] or "{}")})
    stop = "tool_use" if answer.tool_calls else _ANTHROPIC_STOP.get(answer.finish_reason or "stop", "end_turn")
    usage: dict[str, Any] = {}
    if answer.usage.complete:
        usage = {"input_tokens": answer.usage.input_tokens, "output_tokens": answer.usage.output_tokens}
    return {
        "id": f"msg_throttle_{uuid.uuid4().hex[:24]}", "type": "message", "role": "assistant",
        "model": answer.model, "content": content, "stop_reason": stop, "stop_sequence": None, "usage": usage,
    }


def _chunks(text: str, size: int = 256) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)] or [""]


def _data(payload: Any) -> bytes:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def _event(name: str, payload: Any) -> bytes:
    return f"event: {name}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def openai_sse(body: Mapping[str, Any], *, include_usage: bool) -> list[bytes]:
    """A complete chat.completion replayed as chunks (not a live stream)."""
    head = {"id": body["id"], "object": "chat.completion.chunk", "created": body["created"], "model": body["model"]}
    choice = body["choices"][0]
    message = choice["message"]

    def chunk(delta: dict[str, Any], finish: str | None = None) -> bytes:
        return _data({**head, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})

    out = [chunk({"role": "assistant", "content": ""})]
    if message.get("content"):
        out += [chunk({"content": piece}) for piece in _chunks(message["content"])]
    for i, call in enumerate(message.get("tool_calls") or []):
        out.append(chunk({"tool_calls": [{"index": i, "id": call["id"], "type": "function",
                                          "function": {"name": call["function"]["name"],
                                                       "arguments": call["function"]["arguments"]}}]}))
    out.append(chunk({}, choice["finish_reason"]))
    if include_usage and body.get("usage"):
        out.append(_data({**head, "choices": [], "usage": body["usage"]}))
    out.append(b"data: [DONE]\n\n")
    return out


def anthropic_sse(body: Mapping[str, Any]) -> list[bytes]:
    usage = body.get("usage") or {}
    start = {k: v for k, v in body.items() if k not in {"content", "usage", "stop_reason", "stop_sequence"}}
    start.update({"content": [], "stop_reason": None, "stop_sequence": None,
                  "usage": {"input_tokens": usage.get("input_tokens", 0), "output_tokens": 0}})
    out = [_event("message_start", {"type": "message_start", "message": start})]
    for index, block in enumerate(body["content"]):
        if block["type"] == "text":
            out.append(_event("content_block_start", {"type": "content_block_start", "index": index,
                                                      "content_block": {"type": "text", "text": ""}}))
            out += [_event("content_block_delta", {"type": "content_block_delta", "index": index,
                                                   "delta": {"type": "text_delta", "text": piece}})
                    for piece in _chunks(block["text"])]
        else:
            out.append(_event("content_block_start", {"type": "content_block_start", "index": index,
                                                      "content_block": {"type": "tool_use", "id": block["id"],
                                                                        "name": block["name"], "input": {}}}))
            out.append(_event("content_block_delta", {"type": "content_block_delta", "index": index,
                                                      "delta": {"type": "input_json_delta",
                                                                "partial_json": json.dumps(block["input"])}}))
        out.append(_event("content_block_stop", {"type": "content_block_stop", "index": index}))
    out.append(_event("message_delta", {"type": "message_delta",
                                        "delta": {"stop_reason": body["stop_reason"], "stop_sequence": None},
                                        "usage": {"output_tokens": usage.get("output_tokens", 0)}}))
    out.append(_event("message_stop", {"type": "message_stop"}))
    return out


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


@dataclass
class RouteDecision:
    workload_type: str | None = None
    route: Route | None = None
    served: bool = False
    fallback: bool = False
    reason: str | None = None
    body: dict[str, Any] | None = None          # client wire format, when served
    sse: list[bytes] | None = None              # when served and the client asked for a stream
    owned_usage: records.Usage = field(default_factory=records.Usage)
    owned_latency_ms: float | None = None

    @property
    def served_by(self) -> str:
        return records.SERVED_OWNED if self.served else records.SERVED_API

    @property
    def route_reason(self) -> str | None:
        if self.served and self.route is not None:
            return self.route.mode
        return self.reason if self.fallback else None


def router_failure(exc: BaseException) -> RouteDecision:
    return RouteDecision(fallback=True, reason=f"router_error: {type(exc).__name__}")


class _Fallback(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _fallback(code: str, detail: str | None = None) -> _Fallback:
    return _Fallback(f"{code}: {detail}" if detail else code)


def check_answer(norm: records.Normalized, answer: OwnedAnswer) -> None:
    """Raise :class:`_Fallback` when the owned answer should not be served."""
    if (answer.finish_reason or "") in TRUNCATED_REASONS:
        raise _fallback("truncated")
    if not answer.text.strip() and not answer.tool_calls:
        raise _fallback("empty_output")
    if norm.response_format in ("json", "json_schema"):
        parsed_ok, value = parse_json_answer(answer.text)
        if not parsed_ok:
            raise _fallback("invalid_json")
        if norm.response_format == "json_schema" and norm.json_schema is not None:
            if validate_schema(norm.json_schema, value) is False:
                raise _fallback("schema_invalid")
    names = _tool_names(norm.tools)
    for call in answer.tool_calls:
        if call["name"] not in names:
            raise _fallback("unknown_tool", str(call["name"]))
        try:
            parsed = json.loads(call["arguments"] or "{}")
        except ValueError:
            parsed = None
        if not isinstance(parsed, dict):
            raise _fallback("invalid_tool_arguments", str(call["name"]))


def routable(request: Mapping[str, Any], norm: records.Normalized) -> bool:
    n = request.get("n")
    return bool(norm.messages) and not norm.has_images and (n is None or n == 1)


class Router:
    """Decides, per request, whether the owned model answers it. Reads
    ``routes.json`` again whenever it changes, so approve and revoke take
    effect on the next request without a restart."""

    def __init__(
        self,
        root: Path | str,
        *,
        prices: pricing_mod.PriceTable | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        seed: int | None = None,
        classify: Callable[[records.Exchange, records.Normalized], str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.root = Path(root)
        self.prices = prices or pricing_mod.load_prices()
        self.timeout = timeout
        self.classify = classify or workloads.classify
        self._rng = random.Random(seed)
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._routes: dict[str, Route] = {}
        self._routes_key: tuple[int, int] | None = None
        self._ledger_lock = threading.Lock()
        self.stats: Counter[str] = Counter()

    def routes(self) -> dict[str, Route]:
        path = routes_path(self.root)
        try:
            stat = path.stat()
            key = (stat.st_mtime_ns, stat.st_size)
        except FileNotFoundError:
            self._routes, self._routes_key = {}, None
            return self._routes
        if key != self._routes_key:
            self._routes = load_routes(self.root)
            self._routes_key = key
        return self._routes

    def sampled(self, route: Route) -> bool:
        if route.mode != MODE_CANARY:
            return True
        rate = min(1.0, max(0.0, float(route.canary_rate or 0.0)))
        return self._rng.random() < rate

    async def route(self, provider: str, endpoint: str, request: Mapping[str, Any], stream: bool) -> RouteDecision:
        exchange = records.Exchange(
            provider=provider, endpoint=endpoint, model=str(request.get("model") or ""),
            request=dict(request), response=None, status=200, latency_ms=0.0, redaction=records.REDACTION_OFF,
        )
        norm = records.normalize(exchange)
        kind = self.classify(exchange, norm)
        decision = RouteDecision(workload_type=kind)
        route = self.routes().get(kind)
        if route is None or not routable(request, norm) or not self.sampled(route):
            self.stats["api"] += 1
            return decision
        decision.route = route
        started = time.perf_counter()
        try:
            answer = await self._call_owned(route, norm)
            check_answer(norm, answer)
            decision.owned_latency_ms = (time.perf_counter() - started) * 1000
            if provider == records.PROVIDER_ANTHROPIC:
                decision.body = anthropic_body(answer)
                decision.sse = anthropic_sse(decision.body) if stream else None
            else:
                decision.body = openai_body(answer)
                include_usage = bool((request.get("stream_options") or {}).get("include_usage"))
                decision.sse = openai_sse(decision.body, include_usage=include_usage) if stream else None
        except _Fallback as exc:
            decision.fallback, decision.reason = True, exc.reason
            self.stats["fallback"] += 1
            return decision
        decision.served = True
        decision.owned_usage = answer.usage
        self.stats["owned"] += 1
        return decision

    async def _call_owned(self, route: Route, norm: records.Normalized) -> OwnedAnswer:
        headers = {"content-type": "application/json"}
        if route.api_key_env:
            key = os.environ.get(route.api_key_env)
            if not key:
                raise _fallback("config_error", f"{route.api_key_env} is not set")
            headers["authorization"] = f"Bearer {key}"
        if self._client is None:
            self._client = httpx.AsyncClient(transport=self._transport, trust_env=False)
        body = records.to_openai_chat_request(norm, model=route.model)
        try:
            response = await self._client.post(chat_url(route.url), json=body, headers=headers, timeout=self.timeout)
        except httpx.TimeoutException:
            raise _fallback("timeout") from None
        except httpx.HTTPError as exc:
            raise _fallback("connect_error", type(exc).__name__) from None
        if not 200 <= response.status_code < 300:
            raise _fallback("http_error", str(response.status_code))
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if not isinstance(payload, dict) or not payload.get("choices"):
            raise _fallback("bad_response")
        view = records.normalize_openai(None, payload)
        calls = [
            {"id": c.get("id") or f"call_{uuid.uuid4().hex[:12]}", "name": c.get("name"), "arguments": c.get("arguments") or "{}"}
            for c in view.output_tool_calls
        ]
        model = payload.get("model") if isinstance(payload.get("model"), str) else route.model
        return OwnedAnswer(text=view.output_text or "", tool_calls=calls, finish_reason=view.finish_reason,
                           usage=records.usage_from_openai(payload), model=model)

    def record(self, exchange: records.Exchange, decision: RouteDecision) -> None:
        """Append the ledger line for an owned-served exchange."""
        if not decision.served or decision.route is None:
            return
        entry = ledger_entry(exchange, decision, self.prices)
        with self._ledger_lock:
            records.append_jsonl(ledger_path(self.root, exchange.ts[:7]), entry)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def describe(self) -> list[str]:
        routes = self.routes()
        if not routes:
            return [f"Routing is on, but {routes_path(self.root)} has no approved routes: everything goes to the API."]
        lines = ["Routing is on. These workload types are served by your own model:"]
        lines += ["  " + _route_line(r) for r in routes.values()]
        lines.append("Anything doubtful (error, cut off, empty, bad JSON, unknown tool) goes to the API unchanged.")
        lines.append("Streamed requests served by your model arrive in one burst after the full answer (MVP).")
        return lines


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


def ledger_path(root: Path, month: str) -> Path:
    return root / "ledger" / f"{month}.jsonl"


def ledger_entry(exchange: records.Exchange, decision: RouteDecision, prices: pricing_mod.PriceTable) -> dict[str, Any]:
    route = decision.route
    assert route is not None
    usage = decision.owned_usage
    price = prices.lookup(exchange.provider, exchange.model)
    # Serve-time upper bound (no cache discount), kept for audit. The savings
    # statement recomputes avoided cost with the month's measured cache share.
    priced_usage = records.Usage(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens)
    avoided = pricing_mod.cost_usd(price, priced_usage)
    excluded = None
    if not usage.complete:
        excluded = "owned model reported no token usage"
    elif price is None:
        excluded = f"no list price for {exchange.provider} {exchange.model or '(unknown model)'}"
    elif avoided is None:
        excluded = "list price does not cover this request size"
    return {
        "schema": LEDGER_SCHEMA,
        "ts": exchange.ts,
        "exchange_id": exchange.id,
        "type": decision.workload_type,
        "candidate_label": route.candidate_label,
        "owned_model": route.model,
        "mode": route.mode,
        "forced": route.forced,
        "approved_eval_run_id": route.approved_eval_run_id,
        "incumbent_provider": exchange.provider,
        "incumbent_model": exchange.model,
        "owned_input_tokens": usage.input_tokens,
        "owned_output_tokens": usage.output_tokens,
        "api_cost_no_cache_usd": None if avoided is None else round(avoided, 10),
        "price_label": price.label if price is not None else None,
        "excluded": excluded,
    }


# ---------------------------------------------------------------------------
# Savings statement
# ---------------------------------------------------------------------------


def month_hours(month: str) -> int:
    year, mon = (int(x) for x in month.split("-"))
    return calendar.monthrange(year, mon)[1] * 24


def parse_month(value: str | None) -> str:
    if not value:
        return datetime.now(timezone.utc).strftime("%Y-%m")
    try:
        datetime.strptime(value, "%Y-%m")
    except ValueError:
        raise ValueError(f"--month {value!r}: use YYYY-MM, e.g. 2026-10") from None
    return value


@dataclass
class CacheShare:
    """Share of the API model's input tokens that were prompt-cache hits for
    one (type, provider, model), measured from API-served captured exchanges."""

    cached: int = 0
    input: int = 0
    exchanges: int = 0

    @property
    def share(self) -> float | None:
        return self.cached / self.input if self.input else None


def cache_shares(root: Path, month: str, classify: Callable[[records.Exchange], str] | None = None
                 ) -> dict[tuple[str, str, str], CacheShare]:
    """Measured cache-hit share per (type, provider, model) over ``month``.
    Exchanges whose usage does not report cached tokens are left out (unknown
    is not zero), so they can only make the result more conservative."""
    classify = classify or workloads.classify
    shares: dict[tuple[str, str, str], CacheShare] = {}
    for exchange in records.iter_exchanges(root):
        usage = exchange.usage
        if (exchange.ts[:7] != month or exchange.served_by != records.SERVED_API or not exchange.ok
                or not usage.input_tokens or usage.cached_input_tokens is None):
            continue
        kind = exchange.workload_type or classify(exchange)
        row = shares.setdefault((kind, exchange.provider, exchange.model), CacheShare())
        row.cached += min(usage.cached_input_tokens, usage.input_tokens)
        row.input += usage.input_tokens
        row.exchanges += 1
    return shares


def avoided_cost(entry: Mapping[str, Any], prices: pricing_mod.PriceTable,
                 shares: Mapping[tuple[str, str, str], CacheShare]) -> tuple[float | None, str, str | None]:
    """``(usd, cache_basis, excluded_reason)`` for one ledger line."""
    input_tokens, output_tokens = entry.get("owned_input_tokens"), entry.get("owned_output_tokens")
    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        return None, "", "owned model reported no token usage"
    provider, model = str(entry.get("incumbent_provider") or ""), str(entry.get("incumbent_model") or "")
    price = prices.lookup(provider, model)
    if price is None:
        return None, "", f"no list price for {provider} {model or '(unknown model)'}"
    history = shares.get((str(entry.get("type")), provider, model))
    if history is None or history.share is None:
        cached, basis = input_tokens, CACHE_NONE
    else:
        cached, basis = round(input_tokens * history.share), CACHE_MEASURED
    if price.cached_input_per_m is None:
        basis = CACHE_NO_RATE
    usd = pricing_mod.cost_usd(price, records.Usage(input_tokens, output_tokens, cached))
    if usd is None:
        return None, basis, "list price does not cover this request size"
    return usd, basis, None


def savings_statement(
    root: Path,
    month: str,
    *,
    gpu_hourly_rate: float | None = None,
    gpus: int | None = None,
    owned_cost_usd: float | None = None,
    prices: pricing_mod.PriceTable | None = None,
) -> dict[str, Any]:
    """The realized-savings statement for one month. Only routes approved on a
    passing eval count toward the billable net; forced routes are reported
    beside it and never billed."""
    prices = prices or pricing_mod.load_prices()
    entries = list(records.read_jsonl(ledger_path(root, month)))
    shares = cache_shares(root, month)
    avoided = forced_avoided = 0.0
    priced = forced_requests = 0
    excluded: Counter[str] = Counter()
    bases: Counter[str] = Counter()
    by_type: dict[str, dict[str, Any]] = {}
    labels: set[str] = set()
    tokens = {"input": 0, "output": 0}
    for entry in entries:
        kind = str(entry.get("type"))
        forced = bool(entry.get("forced"))
        row = by_type.setdefault(kind, {"requests": 0, "billable_avoided_usd": 0.0, "forced_requests": 0,
                                        "forced_avoided_usd": 0.0, "excluded": 0, "cache_basis": set()})
        row["requests"] += 1
        tokens["input"] += entry.get("owned_input_tokens") or 0
        tokens["output"] += entry.get("owned_output_tokens") or 0
        if forced:
            forced_requests += 1
            row["forced_requests"] += 1
        usd, basis, reason = avoided_cost(entry, prices, shares)
        if usd is None:
            excluded[reason or "not priced"] += 1
            row["excluded"] += 1
            continue
        priced += 1
        bases[basis] += 1
        row["cache_basis"].add(basis)
        price = prices.lookup(str(entry.get("incumbent_provider")), str(entry.get("incumbent_model")))
        if price is not None:
            labels.add(price.label)
        if forced:
            forced_avoided += usd
            row["forced_avoided_usd"] += usd
        else:
            avoided += usd
            row["billable_avoided_usd"] += usd

    fallbacks: Counter[str] = Counter()
    for exchange in records.iter_exchanges(root):
        if exchange.ts[:7] == month and exchange.fallback:
            fallbacks[(exchange.route_reason or "unknown").split(":", 1)[0]] += 1
    attempted = len(entries) + sum(fallbacks.values())

    hours = month_hours(month)
    if owned_cost_usd is not None:
        owned = float(owned_cost_usd)
        owned_label = "OPERATOR-REPORTED (invoiced amount)"
        owned_basis = None
    elif gpu_hourly_rate is not None and gpus is not None:
        owned = float(gpu_hourly_rate) * int(gpus) * hours
        owned_label = f"ASSUMED (operator GPU rate ${gpu_hourly_rate:g}/hr x {gpus} GPU x {hours} h, always-on)"
        owned_basis = {"gpu_hourly_rate": gpu_hourly_rate, "gpus": gpus, "hours": hours}
    else:
        raise ValueError("give --gpu-hourly-rate and --gpus, or --owned-cost-usd")
    net = avoided - owned
    return {
        "schema": SAVINGS_SCHEMA,
        "month": month,
        "period_hours": hours,
        "requests_served_owned": len(entries),
        "requests_priced": priced,
        "excluded": dict(excluded),
        "owned_tokens": tokens,
        "fallbacks": sum(fallbacks.values()),
        "fallback_rate": (sum(fallbacks.values()) / attempted) if attempted else None,
        "fallbacks_by_reason": {k: {"count": n, "rate": n / attempted} for k, n in fallbacks.most_common()},
        "api_cost_avoided_usd": round(avoided, 6),
        "api_cost_avoided_label": "MEASURED owned tokens x " + (", ".join(sorted(labels)) if labels else "list price")
        + ", cache-adjusted, approved routes only",
        "cache_basis_counts": dict(bases),
        "cache_history": [
            {"type": k[0], "provider": k[1], "model": k[2], "cached_share": v.share, "exchanges": v.exchanges}
            for k, v in sorted(shares.items())
        ],
        "owned_cost_usd": round(owned, 6),
        "owned_cost_label": owned_label,
        "owned_cost_basis": owned_basis,
        "net_realized_savings_usd": round(net, 6),
        "billable": net > 0,
        "forced": {"requests": forced_requests, "api_cost_avoided_usd": round(forced_avoided, 6), "billable": False},
        "by_type": {
            k: {**v, "billable_avoided_usd": round(v["billable_avoided_usd"], 6),
                "forced_avoided_usd": round(v["forced_avoided_usd"], 6), "cache_basis": sorted(v["cache_basis"])}
            for k, v in sorted(by_type.items())
        },
        "assumptions": [
            ASSUMPTION_TOKENS,
            ASSUMPTION_CACHE,
            ASSUMPTION_FORCED,
            ASSUMPTION_OWNED if owned_basis else "Owned cost is the amount the operator reported as invoiced.",
            "Fallbacks were answered by the API and are already in the API bill; they are not counted as savings.",
            "Requests with no owned token usage or no list price are excluded and counted, never estimated.",
        ] + ([IN_PROGRESS] if owned_basis and month == datetime.now(timezone.utc).strftime("%Y-%m") else []),
        "ledger": str(ledger_path(root, month)),
    }


def _usd(value: float) -> str:
    return f"-${-value:,.2f}" if value < 0 else f"${value:,.2f}"


def render_statement(doc: Mapping[str, Any]) -> str:
    from .. import style

    st = style.get()
    rows: list[Any] = [
        f"Requests served by your own model: {doc['requests_served_owned']:,}   MEASURED (ledger)",
    ]
    rate = doc["fallback_rate"]
    rows.append(f"Fallbacks to the API: {doc['fallbacks']:,}" + (f" ({rate:.1%} of routed)" if rate is not None else "")
                + "   MEASURED (capture)")
    for reason, info in doc["fallbacks_by_reason"].items():
        rows.append(f"  {reason:<24} {info['count']:>6}  ({info['rate']:.1%})")
    if doc["excluded"]:
        total = sum(doc["excluded"].values())
        rows.append((f"Excluded from avoided cost (never estimated): {total} - "
                     + "; ".join(f"{k}: {n}" for k, n in doc["excluded"].items()), ()))
    if doc["cache_basis_counts"]:
        rows.append((f"Prompt-cache pricing (requests): "
                     + "; ".join(f"{k}: {n}" for k, n in sorted(doc["cache_basis_counts"].items())), ()))
    rows += [
        "",
        (f"API cost avoided:      {_usd(doc['api_cost_avoided_usd'])}   {doc['api_cost_avoided_label']}", ()),
        (f"Owned cost:            {_usd(doc['owned_cost_usd'])}   {doc['owned_cost_label']}", ()),
        st.bold(f"Net realized savings:  {_usd(doc['net_realized_savings_usd'])}   (billable)"),
    ]
    if not doc["billable"]:
        rows.append(st.warn("No savings this period, no bill."))
    if doc["forced"]["requests"]:
        rows += ["", (f"FORCED (not billable; quality not proven): {doc['forced']['requests']} requests, "
                      f"{_usd(doc['forced']['api_cost_avoided_usd'])} avoided through routes approved with --force.",
                      (BOLD_RED,))]
    rows += ["", st.bold("By type")]
    for kind, row in doc["by_type"].items():
        line = f"  {kind:<15} {row['requests']:>7} requests  {_usd(row['billable_avoided_usd']):>12} billable"
        if row["forced_requests"]:
            line += f"  FORCED {row['forced_requests']} ({_usd(row['forced_avoided_usd'])}, not billable)"
        rows.append(line)
        if row["cache_basis"]:
            rows.append(st.dim(f"    cache: {', '.join(row['cache_basis'])}"))
    rows += [""] + [(f"- {a}", (), "  ") for a in doc["assumptions"]]
    rows.append(st.dim(f"Ledger: {doc['ledger']}"))
    return st.panel(rows, title=f"Realized savings {doc['month']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _route_line(route: Route) -> str:
    mode = f"canary {route.canary_rate:.0%}" if route.mode == MODE_CANARY else "live"
    forced = "  FORCED (no passing eval)" if route.forced else ""
    return f"{route.workload_type:<15} -> {route.candidate_label} ({route.model} at {route.url}), {mode}{forced}"


def _rate(value: str) -> float:
    rate = float(value)
    if not 0.0 < rate <= 1.0:
        raise argparse.ArgumentTypeError("canary rate must be in (0, 1]")
    return rate


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "route",
        help="serve approved workload types from your own model, and show the realized savings",
        description=(
            "Approve a workload type for your own model once `throttle eval` shows it is as good as the API, "
            "then run `throttle capture --route`. Doubtful answers fall back to the API unchanged."
        ),
    )
    dir_help = "data folder (default: ~/.throttle/own or $THROTTLE_OWN_DIR)"
    parser.add_argument("--dir", type=Path, default=None, help=dir_help)
    actions = parser.add_subparsers(dest="route_action", metavar="{approve,list,revoke,savings}")
    actions.required = True

    a = actions.add_parser("approve", help="route one workload type to your model (needs a passing eval)")
    a.add_argument("--type", required=True, choices=records.WORKLOAD_TYPES, dest="workload_type")
    a.add_argument("--candidate", required=True, help="candidate label used in throttle shadow / eval")
    a.add_argument("--url", required=True, help="OpenAI-compatible server, e.g. http://gpu-box:8000/v1")
    a.add_argument("--model", required=True, help="model name the server expects")
    a.add_argument("--api-key-env", default=None, metavar="VAR", help="environment variable with the server's key")
    a.add_argument("--canary", type=_rate, default=None, metavar="RATE", help="serve only this share, e.g. 0.1")
    a.add_argument("--force", action="store_true", help="approve without a passing eval (marked FORCED everywhere)")

    actions.add_parser("list", help="show approved routes")

    r = actions.add_parser("revoke", help="stop routing a workload type (takes effect on the next request)")
    r.add_argument("--type", required=True, dest="workload_type")

    s = actions.add_parser("savings", help="realized-savings statement for one month")
    s.add_argument("--month", default=None, help="YYYY-MM (default: this month, UTC)")
    s.add_argument("--gpu-hourly-rate", type=float, default=None, help="USD per GPU-hour you pay for the owned model")
    s.add_argument("--gpus", type=int, default=None, help="GPUs serving the owned model")
    s.add_argument("--owned-cost-usd", type=float, default=None,
                   help="invoiced owned cost for the month (labelled OPERATOR-REPORTED)")
    s.add_argument("--prices", type=Path, default=None, metavar="FILE",
                   help="JSON price file that replaces or adds entries (labelled ASSUMED)")
    s.add_argument("--json", action="store_true")
    for sub in (a, s):
        sub.add_argument("--dir", type=Path, default=argparse.SUPPRESS, help=dir_help)
    for name in ("list", "revoke"):
        actions.choices[name].add_argument("--dir", type=Path, default=argparse.SUPPRESS, help=dir_help)


def handle(args: argparse.Namespace) -> int:
    root = Path(args.dir).expanduser() if getattr(args, "dir", None) else records.default_root()
    action = args.route_action
    try:
        if action == "approve":
            return _approve(args, root)
        if action == "list":
            return _list(root)
        if action == "revoke":
            if revoke(root, args.workload_type):
                print(f"Revoked: {args.workload_type} goes to the API from the next request.")
                return EXIT_OK
            print(f"throttle route revoke: no route for {args.workload_type}.", file=sys.stderr)
            return EXIT_FAILED
        if action == "savings":
            return _savings(args, root)
    except ValueError as exc:
        print(f"throttle route {action}: {exc}", file=sys.stderr)
        return EXIT_USAGE
    return EXIT_USAGE


def _approve(args: argparse.Namespace, root: Path) -> int:
    route, check = approve(root, args.workload_type, args.candidate, args.url, args.model,
                           api_key_env=args.api_key_env, canary=args.canary, force=args.force)
    if route is None:
        print(f"throttle route approve: refused. {check.reason}.", file=sys.stderr)
        print("  Approve only what the eval shows is as good as the API, or pass --force (marked FORCED).",
              file=sys.stderr)
        return EXIT_FAILED
    if route.forced:
        print(f"FORCED: {check.reason}. Route stored with forced: true and flagged everywhere.")
    else:
        print(f"Approved: {check.reason}.")
    print("  " + _route_line(route))
    print("  Serve it with: throttle capture --route")
    return EXIT_OK


def _list(root: Path) -> int:
    routes = load_routes(root)
    if not routes:
        print(f"No approved routes in {routes_path(root)}. Everything goes to the API.")
        return EXIT_OK
    for route in routes.values():
        print(_route_line(route))
        print(f"    approved {route.approved_at}, eval run {route.approved_eval_run_id or '(none)'}: "
              f"{route.eval_verdict or 'no verdict'}")
    return EXIT_OK


def _savings(args: argparse.Namespace, root: Path) -> int:
    month = parse_month(args.month)
    if args.owned_cost_usd is None and (args.gpu_hourly_rate is None or args.gpus is None):
        raise ValueError("give --gpu-hourly-rate and --gpus (always-on owned cost), or --owned-cost-usd")
    doc = savings_statement(root, month, gpu_hourly_rate=args.gpu_hourly_rate, gpus=args.gpus,
                            owned_cost_usd=args.owned_cost_usd, prices=pricing_mod.load_prices(args.prices))
    if args.json:
        print(json.dumps(doc, indent=2, sort_keys=True))
    else:
        print(render_statement(doc))
    return EXIT_OK
