"""Stage 3, SHADOW: send captured requests to a self-hosted candidate model.

Two ways to run it:

* ``throttle shadow`` (backfill) samples already-captured exchanges,
  stratified by workload type, and replays each one against an
  OpenAI-compatible candidate (vLLM, SGLang, Ollama, LMDeploy).
* :class:`LiveShadow` is an ``on_exchange`` callback for the capture proxy.
  It samples live traffic after the exchange has been served and logged.

Either way the candidate's answer goes only to ``<root>/shadow/<label>.jsonl``
as a :class:`records.ShadowResult`. Shadow output is never served to a client
and never written into the captured exchange.

``ShadowResult.error`` is ``"<kind>: <message>"``. Kinds: ``context_overflow``,
``http_<status>``, ``timeout``, ``connect``, ``bad_response``. An answer with
no text and no tool calls is ``bad_response: empty answer``.

A ``timeout`` or ``connect`` failure has status 0 (``TRANSPORT_STATUS``).
Backfill retries it once right away. If the retry fails in transport too, the
result is stored with status ``FINAL_TRANSPORT_STATUS`` (599): a permanent
candidate failure that backfill does not retry and evaluate counts as a LOSS.
A status-0 result (live shadow, or an older run) is left out of evaluate and
counted as excluded; backfill retries those exchanges on its next run.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Collection, Iterable, Mapping, Sequence

import httpx

from . import records
from ..request_errors import context_overflow_limit, describe_http_failure

ALL_TYPES = "all"
DEFAULT_SEED = 0
TRANSPORT_STATUS = 0
FINAL_TRANSPORT_STATUS = 599   # transport failed again on retry; a candidate failure
OVERFLOW_REMEDY = (
    "serve the candidate with a larger max_model_len, or keep this workload type on the API"
)

Classifier = Callable[[records.Exchange, records.Normalized], str]


# ---------------------------------------------------------------------------
# Shared helpers (evaluate imports these too)
# ---------------------------------------------------------------------------


def resolve_classifier(classify: Classifier | None = None) -> Classifier:
    """The workload classifier: the one passed in, else ``throttle.own.workloads.classify``,
    else every exchange is type ``"all"``."""

    if classify is not None:
        return classify
    try:
        module = importlib.import_module("throttle.own.workloads")
        found = getattr(module, "classify", None)
    except ImportError:
        found = None
    if callable(found):
        return found  # type: ignore[return-value]
    return lambda exchange, norm: ALL_TYPES


def workload_type(exchange: records.Exchange, norm: records.Normalized, classify: Classifier) -> str:
    try:
        value = classify(exchange, norm)
    except Exception:  # a classifier bug must not stop shadowing
        value = None
    return value if isinstance(value, str) and value else ALL_TYPES


def eligible(exchange: records.Exchange) -> records.Normalized | None:
    """The normalized view if this exchange can be replayed: ok, with bodies, no images."""

    if not (exchange.ok and exchange.has_bodies):
        return None
    norm = records.normalize(exchange)
    if norm.has_images or not norm.messages:
        return None
    return norm


def chat_url(url: str) -> str:
    """Accept a server root, a ``/v1`` base or a full chat-completions URL."""

    url = url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return url + "/chat/completions"
    return url + "/v1/chat/completions"


_DURATION = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhdw])\s*$", re.IGNORECASE)
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def parse_since(value: str | None, *, now: datetime | None = None) -> datetime | None:
    if not value:
        return None
    match = _DURATION.match(value)
    if not match:
        raise ValueError(f"--since {value!r}: use a duration like 30m, 24h or 7d")
    seconds = float(match.group(1)) * _UNIT_SECONDS[match.group(2).lower()]
    return (now or datetime.now(timezone.utc)) - timedelta(seconds=seconds)


def error_kind(error: str | None) -> str | None:
    if not error:
        return None
    return error.split(":", 1)[0]


def api_key_from_env(var: str | None) -> str | None:
    if not var:
        return None
    value = os.environ.get(var)
    if not value:
        raise ValueError(f"environment variable {var} is not set")
    return value


# ---------------------------------------------------------------------------
# One candidate call
# ---------------------------------------------------------------------------


async def shadow_one(
    client: httpx.AsyncClient,
    exchange: records.Exchange,
    norm: records.Normalized,
    *,
    url: str,
    model: str,
    label: str,
    api_key: str | None = None,
    timeout: float = 120.0,
) -> records.ShadowResult:
    """Replay one captured request against the candidate. Never raises for
    HTTP or transport failures; they come back as an errored ShadowResult."""

    target = chat_url(url)
    body = records.to_openai_chat_request(norm, model=model, max_tokens=norm.max_tokens)
    headers = {"content-type": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"

    def failed(status: int, kind: str, message: str, latency_ms: float) -> records.ShadowResult:
        return records.ShadowResult(
            exchange_id=exchange.id, candidate=label, candidate_model=model,
            output_text=None, tool_calls=[], usage=records.Usage(),
            latency_ms=latency_ms, status=status, error=f"{kind}: {message}",
        )

    started = time.perf_counter()
    try:
        response = await client.post(target, json=body, headers=headers, timeout=timeout)
    except httpx.TimeoutException:
        return failed(TRANSPORT_STATUS, "timeout", f"no answer from {target} within {timeout:g}s",
                      (time.perf_counter() - started) * 1000.0)
    except httpx.HTTPError as exc:
        return failed(TRANSPORT_STATUS, "connect", f"POST {target} failed: {type(exc).__name__}",
                      (time.perf_counter() - started) * 1000.0)
    latency_ms = (time.perf_counter() - started) * 1000.0
    if not 200 <= response.status_code < 300:
        text = response.text
        overflow, _ = context_overflow_limit(response.status_code, text)
        message = describe_http_failure(
            what=f"Shadow request for exchange {exchange.id[:12]}",
            url=target, status=response.status_code, body=text,
            max_tokens=body.get("max_tokens"), remedy=OVERFLOW_REMEDY,
        )
        kind = "context_overflow" if overflow else f"http_{response.status_code}"
        return failed(response.status_code, kind, message, latency_ms)
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if not isinstance(payload, dict) or not payload.get("choices"):
        return failed(response.status_code, "bad_response", "candidate answer had no choices", latency_ms)
    view = records.normalize_openai(None, payload)
    empty = not view.output_text and not view.output_tool_calls
    return records.ShadowResult(
        exchange_id=exchange.id, candidate=label, candidate_model=model,
        output_text=None if empty else (view.output_text or ""),
        tool_calls=view.output_tool_calls,
        usage=records.usage_from_openai(payload),
        latency_ms=latency_ms, status=response.status_code,
        finish_reason=view.finish_reason,
        error="bad_response: empty answer" if empty else None,
    )


async def shadow_with_retry(
    client: httpx.AsyncClient,
    exchange: records.Exchange,
    norm: records.Normalized,
    **kwargs: Any,
) -> records.ShadowResult:
    """:func:`shadow_one`, retried once on a transport failure. A second
    transport failure comes back with ``FINAL_TRANSPORT_STATUS``."""

    result = await shadow_one(client, exchange, norm, **kwargs)
    if result.status != TRANSPORT_STATUS:
        return result
    result = await shadow_one(client, exchange, norm, **kwargs)
    if result.status == TRANSPORT_STATUS:
        result.status = FINAL_TRANSPORT_STATUS
    return result


# ---------------------------------------------------------------------------
# Backfill
# ---------------------------------------------------------------------------


def done_ids(label: str, root: Path | None) -> set[str]:
    """Exchanges that already have a stored result for ``label`` (transport
    failures excluded, so they are retried)."""

    return {
        result.exchange_id
        for result in records.iter_shadow(label, root)
        if result.status != TRANSPORT_STATUS
    }


def select_backfill(
    exchanges: Iterable[records.Exchange],
    *,
    skip: set[str] = frozenset(),  # type: ignore[assignment]
    per_type: int = 50,
    max_requests: int = 500,
    seed: int = DEFAULT_SEED,
    classify: Classifier | None = None,
    only_ids: Collection[str] | None = None,
) -> list[tuple[records.Exchange, records.Normalized, str]]:
    """Stratified, seeded sample: up to ``per_type`` per workload type, then
    round-robin across types until ``max_requests``. ``only_ids`` restricts
    the pool to those exchange ids (refresh reruns a saved eval set)."""

    classifier = resolve_classifier(classify)
    wanted = set(only_ids) if only_ids is not None else None
    groups: dict[str, list[tuple[records.Exchange, records.Normalized, str]]] = defaultdict(list)
    for exchange in sorted(exchanges, key=lambda e: (e.ts, e.id)):
        if exchange.id in skip or (wanted is not None and exchange.id not in wanted):
            continue
        norm = eligible(exchange)
        if norm is None:
            continue
        kind = workload_type(exchange, norm, classifier)
        groups[kind].append((exchange, norm, kind))
    rng = random.Random(seed)
    picked: dict[str, list[tuple[records.Exchange, records.Normalized, str]]] = {}
    for kind in sorted(groups):
        items = groups[kind]
        rng.shuffle(items)
        picked[kind] = items[: max(0, per_type)]
    out: list[tuple[records.Exchange, records.Normalized, str]] = []
    index = 0
    while len(out) < max_requests and any(index < len(v) for v in picked.values()):
        for kind in sorted(picked):
            if index < len(picked[kind]) and len(out) < max_requests:
                out.append(picked[kind][index])
        index += 1
    return out


async def run_backfill(
    selection: Sequence[tuple[records.Exchange, records.Normalized, str]],
    *,
    url: str,
    model: str,
    label: str,
    root: Path | None = None,
    api_key: str | None = None,
    concurrency: int = 4,
    timeout: float = 120.0,
    transport: httpx.AsyncBaseTransport | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    totals = Counter(kind for _, _, kind in selection)
    done: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    errors_by_type: Counter[str] = Counter()
    gate = asyncio.Semaphore(max(1, concurrency))

    async with httpx.AsyncClient(transport=transport) as client:

        async def one(exchange: records.Exchange, norm: records.Normalized, kind: str) -> None:
            async with gate:
                result = await shadow_with_retry(client, exchange, norm, url=url, model=model,
                                                 label=label, api_key=api_key, timeout=timeout)
            records.append_shadow(result, root)
            done[kind] += 1
            if result.error:
                errors[error_kind(result.error) or "unknown"] += 1
                errors_by_type[kind] += 1
            if progress and done[kind] == totals[kind]:
                progress(f"  {kind:<16} {done[kind]}/{totals[kind]} done, {errors_by_type[kind]} errors")

        await asyncio.gather(*(one(e, n, k) for e, n, k in selection))

    return {
        "done": sum(done.values()),
        "ok": sum(done.values()) - sum(errors.values()),
        "by_type": {k: {"done": done[k], "errors": errors_by_type[k]} for k in sorted(totals)},
        "errors": dict(sorted(errors.items())),
    }


# ---------------------------------------------------------------------------
# Live
# ---------------------------------------------------------------------------


class LiveShadow:
    """``on_exchange`` callback that shadows a sample of live traffic.

    The proxy calls it after the client already has its response and the
    exchange is logged. The call only decides whether to sample and, if so,
    starts a background task; it never awaits the candidate and never touches
    the exchange. When ``max_in_flight`` calls are already running the sample
    is dropped, not queued. ``max_total`` is the budget for this process.

    ``sample_rate`` is one rate for every type, or ``{type: rate}`` with an
    optional ``"*"`` default (missing types default to 0).
    """

    def __init__(
        self,
        *,
        url: str,
        model: str,
        label: str,
        root: Path | None = None,
        sample_rate: float | Mapping[str, float] = 0.1,
        max_in_flight: int = 4,
        max_total: int = 500,
        api_key: str | None = None,
        timeout: float = 120.0,
        seed: int | None = None,
        classify: Classifier | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.url = url
        self.model = model
        self.label = label
        self.root = root
        self.sample_rate = sample_rate
        self.max_in_flight = max(1, max_in_flight)
        self.max_total = max(0, max_total)
        self.api_key = api_key
        self.timeout = timeout
        self.classify = resolve_classifier(classify)
        self._rng = random.Random(seed)
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self.in_flight = 0
        self.stats: Counter[str] = Counter()

    def rate_for(self, kind: str) -> float:
        if isinstance(self.sample_rate, Mapping):
            rate = self.sample_rate.get(kind, self.sample_rate.get("*", 0.0))
        else:
            rate = self.sample_rate
        return min(1.0, max(0.0, float(rate)))

    async def __call__(self, exchange: records.Exchange) -> None:
        self.stats["seen"] += 1
        norm = eligible(exchange)
        if norm is None:
            self.stats["ineligible"] += 1
            return
        kind = workload_type(exchange, norm, self.classify)
        if self._rng.random() >= self.rate_for(kind):
            self.stats["not_sampled"] += 1
            return
        if self.stats["started"] >= self.max_total:
            self.stats["dropped_budget"] += 1
            return
        if self.in_flight >= self.max_in_flight:
            self.stats["dropped_busy"] += 1
            return
        self.in_flight += 1
        self.stats["started"] += 1
        task = asyncio.get_running_loop().create_task(self._run(exchange, norm))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run(self, exchange: records.Exchange, norm: records.Normalized) -> None:
        try:
            if self._client is None:
                self._client = httpx.AsyncClient(transport=self._transport)
            result = await shadow_one(self._client, exchange, norm, url=self.url, model=self.model,
                                      label=self.label, api_key=self.api_key, timeout=self.timeout)
            records.append_shadow(result, self.root)
            self.stats["errors" if result.error else "stored"] += 1
        except Exception:  # never let a shadow failure surface in the proxy
            self.stats["errors"] += 1
        finally:
            self.in_flight -= 1

    async def drain(self) -> None:
        """Wait for running shadow calls, then close the client."""

        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "shadow",
        help="replay captured requests against your own open-weight model (never served)",
        description=(
            "Sample captured API requests by workload type and send each one to a "
            "self-hosted OpenAI-compatible model (vLLM, SGLang, Ollama, LMDeploy). "
            "The answers are stored for `throttle eval` and are never served."
        ),
    )
    parser.add_argument("--candidate-url", required=True, help="candidate server, e.g. http://gpu-box:8000/v1")
    parser.add_argument("--candidate-model", required=True, help="model name the candidate server expects")
    parser.add_argument("--label", required=True, help="name for this candidate, e.g. qwen2.5-72b@vllm")
    parser.add_argument("--api-key-env", default=None, metavar="VAR",
                        help="environment variable holding the candidate's API key, if it needs one")
    parser.add_argument("--per-type", type=int, default=50, help="requests per workload type (default 50)")
    parser.add_argument("--max-requests", type=int, default=500, help="total cap (default 500)")
    parser.add_argument("--concurrency", type=int, default=4, help="requests in flight (default 4)")
    parser.add_argument("--timeout", type=float, default=120.0, help="seconds per request (default 120)")
    parser.add_argument("--since", default=None, help="only exchanges newer than this, e.g. 7d or 24h")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="sampling seed (default 0)")
    parser.add_argument("--dir", type=Path, default=None, help="data directory (default ~/.throttle/own)")


def handle(args: argparse.Namespace, *, classify: Classifier | None = None,
           transport: httpx.AsyncBaseTransport | None = None) -> int:
    root = args.dir or records.default_root()
    try:
        since = parse_since(args.since)
        api_key = api_key_from_env(args.api_key_env)
    except ValueError as exc:
        print(f"throttle shadow: {exc}", file=sys.stderr)
        return 2
    skip = done_ids(args.label, root)
    selection = select_backfill(
        records.iter_exchanges(root, since=since), skip=skip, per_type=args.per_type,
        max_requests=args.max_requests, seed=args.seed, classify=classify,
    )
    if not selection:
        print(f"Nothing to shadow for {args.label}: no new captured requests with bodies "
              f"in {records.capture_dir(root)} ({len(skip)} already shadowed).")
        return 0
    counts = Counter(kind for _, _, kind in selection)
    print(f"Shadowing {len(selection)} captured requests on {args.label} ({args.candidate_model}).")
    print("  " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    print("  Answers are stored for `throttle eval` only. Nothing is served.")
    summary = asyncio.run(run_backfill(
        selection, url=args.candidate_url, model=args.candidate_model, label=args.label,
        root=root, api_key=api_key, concurrency=args.concurrency, timeout=args.timeout,
        transport=transport, progress=print,
    ))
    errors = summary["errors"]
    error_text = ", ".join(f"{k} {v}" for k, v in errors.items()) if errors else "none"
    print(f"Done: {summary['done']} requests, {summary['ok']} answered, errors: {error_text}.")
    print(f"  Stored in {records.shadow_path(args.label, root)}")
    if errors.get("context_overflow"):
        print("  Some prompts do not fit the candidate's context window; "
              "`throttle eval` counts those as losses.")
    if summary["ok"] == 0:
        return 1
    return 0
