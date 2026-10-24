"""``throttle compare --baseline URL --candidate URL --eval-set FILE``: should
you switch models?

Runs your own prompts against two OpenAI-compatible endpoints (vLLM, SGLang,
Ollama, ...) ``--runs`` times each, alternating baseline and candidate runs so
slow drift on the machine hits both sides alike. For each side it reports
throughput at ``--concurrency``, TTFT p50/p95 (from streaming), $/M output
tokens and $ per *passing* answer at your GPU price, the pass rate on your
checks, the JSON-validity and tool-call-format error rates, and how far the
candidate's answers drift from the baseline's.

Every number is shown with its spread over the runs (min..max). A difference
whose run ranges overlap is called "within noise", the same rule ``throttle
check`` uses for its NO WINNER verdict, just on the raw run ranges.

Nothing leaves the machine except the requests to the two endpoints you name.
The run writes ``manifest.json`` (models, engine versions, every flag, the
eval-set sha256, timestamp), ``outputs.jsonl`` (every answer, to inspect) and
``result.json`` into ``--out``.

Eval-set lines (JSONL)::

    {"prompt": "Capital of France?", "expected": "Paris", "check": "contains:Paris"}
    {"messages": [{"role": "user", "content": "..."}], "check": "json"}
    {"prompt": "...", "check": "regex:^\\\\d+$"}
    {"prompt": "...", "expected": "42", "check": "exact"}
    {"prompt": "...", "tools": [...]}             # tool calls are format-checked

``check`` is optional. With ``expected`` and no ``check`` the answer must match
``expected`` after normalizing case, whitespace and punctuation. With neither,
the prompt is not scored, but still counts for speed, cost and drift.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import platform
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

import httpx

from . import __version__
from . import style as style_module
from .benchmark import normalize_chat_completions_url
from .statistics import percentile

RECORD_TYPE = "throttle_eval_compare"
RECORD_VERSION = 1
DEFAULT_RUNS = 3
DEFAULT_CONCURRENCY = 8
DEFAULT_MAX_TOKENS = 512
DEFAULT_TIMEOUT = 300.0
INFO_TIMEOUT = 10.0
MAX_EVAL_ITEMS = 100_000

EXIT_SWITCH = 0
EXIT_DONT_SWITCH = 1
EXIT_USAGE = 2
EXIT_FAILED = 3

VERDICT_SWITCH = "SWITCH"
VERDICT_DONT = "DON'T SWITCH"
VERDICT_NOISE = "WITHIN NOISE"

# Set by tests to an httpx.MockTransport; None means real sockets.
_TRANSPORT: httpx.AsyncBaseTransport | None = None
_SYNC_TRANSPORT: httpx.BaseTransport | None = None

LIVE_EPILOG = """\
live model comparison (your prompts, two endpoints):
  throttle compare \\
      --baseline http://localhost:30000 --baseline-model Qwen/Qwen2.5-7B-Instruct \\
      --candidate http://localhost:30001 --candidate-model meta-llama/Llama-3.1-8B-Instruct \\
      --eval-set evals.jsonl --concurrency 16 --runs 3 --gpu-price 2.49

  Eval-set lines: {"prompt" or "messages", optional "expected",
  optional "check": "json" | "contains:<text>" | "regex:<re>" | "exact"}.
  Exit 0 = SWITCH or within noise, 1 = DON'T SWITCH, 2 = usage error,
  3 = an endpoint failed. Writes manifest.json, outputs.jsonl and result.json
  to --out. Sends nothing anywhere except the two endpoints.
"""


class EvalSetError(ValueError):
    pass


def add_live_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        "live model comparison",
        "Compare two running endpoints on your own prompts (instead of two saved reports).",
    )
    group.add_argument("--baseline", dest="live_baseline", metavar="URL",
                       help="the endpoint you run today (OpenAI-compatible base URL)")
    group.add_argument("--baseline-model", metavar="MODEL", help="model name on --baseline")
    group.add_argument("--candidate", dest="live_candidate", metavar="URL",
                       help="the endpoint you are thinking of switching to")
    group.add_argument("--candidate-model", metavar="MODEL", help="model name on --candidate")
    group.add_argument("--eval-set", type=Path, metavar="FILE.jsonl",
                       help="your prompts, one JSON object per line")
    group.add_argument("--concurrency", type=_positive_int, default=DEFAULT_CONCURRENCY,
                       metavar="N", help=f"requests in flight per side (default {DEFAULT_CONCURRENCY})")
    group.add_argument("--runs", type=_positive_int, default=DEFAULT_RUNS, metavar="K",
                       help=f"times to run the whole eval set per side (default {DEFAULT_RUNS})")
    group.add_argument("--gpu-price", type=_non_negative_float, metavar="$/HR",
                       help="GPU $/hour for both sides (total for the machine serving the model)")
    group.add_argument("--baseline-gpu-price", type=_non_negative_float, metavar="$/HR",
                       help="GPU $/hour for the baseline (overrides --gpu-price)")
    group.add_argument("--candidate-gpu-price", type=_non_negative_float, metavar="$/HR",
                       help="GPU $/hour for the candidate (overrides --gpu-price)")
    group.add_argument("--max-tokens", type=_positive_int, default=DEFAULT_MAX_TOKENS,
                       metavar="N", help=f"max_tokens per request unless the line sets one "
                       f"(default {DEFAULT_MAX_TOKENS})")
    group.add_argument("--temperature", type=_non_negative_float, default=0.0,
                       help="sampling temperature (default 0, so drift means the model changed)")
    group.add_argument("--request-timeout", type=_positive_float, default=DEFAULT_TIMEOUT,
                       metavar="SECONDS", help=f"per-request timeout (default {DEFAULT_TIMEOUT:g})")
    group.add_argument("--api-key-env", default="OPENAI_API_KEY", metavar="NAME",
                       help="environment variable holding the bearer key (default OPENAI_API_KEY)")
    group.add_argument("--allow-insecure-http", action="store_true",
                       help="allow plain http to a non-loopback host (the key is sent in clear)")
    group.add_argument("--out", type=Path, metavar="DIR",
                       help="where to write manifest.json, outputs.jsonl and result.json "
                       "(default ./throttle-compare-<timestamp>)")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value.lstrip("$"))
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a number >= 0")
    return parsed


def is_live(args: argparse.Namespace) -> bool:
    return any(
        getattr(args, name, None) is not None
        for name in ("live_baseline", "live_candidate", "eval_set")
    )


# --------------------------------------------------------------------------
# Eval set and checks
# --------------------------------------------------------------------------


@dataclass
class EvalItem:
    id: str
    messages: list[dict[str, Any]]
    expected: str | None = None
    check: str | None = None
    tools: list[Any] | None = None
    max_tokens: int | None = None


_CHECK_KINDS = ("json", "exact")
_CHECK_PREFIXES = ("contains:", "regex:")


def load_eval_set(path: Path) -> tuple[list[EvalItem], str]:
    """The items and the sha256 of the file's bytes."""

    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise EvalSetError(f"--eval-set {path}: cannot read it ({exc.strerror or exc})") from None
    digest = hashlib.sha256(raw).hexdigest()
    items: list[EvalItem] = []
    seen: set[str] = set()
    for number, line in enumerate(raw.decode("utf-8", errors="strict").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        where = f"--eval-set {path} line {number}"
        try:
            data = json.loads(line)
        except ValueError:
            raise EvalSetError(f"{where}: not valid JSON") from None
        if not isinstance(data, dict):
            raise EvalSetError(f"{where}: must be a JSON object")
        if "messages" in data:
            messages = data["messages"]
            if (not isinstance(messages, list) or not messages
                    or not all(isinstance(m, dict) and "role" in m for m in messages)):
                raise EvalSetError(f"{where}: \"messages\" must be a non-empty list of "
                                   "{\"role\": ..., \"content\": ...} objects")
        elif isinstance(data.get("prompt"), str) and data["prompt"]:
            messages = [{"role": "user", "content": data["prompt"]}]
        else:
            raise EvalSetError(f"{where}: needs \"prompt\" (a string) or \"messages\"")
        expected = data.get("expected")
        if expected is not None and not isinstance(expected, str):
            expected = json.dumps(expected, sort_keys=True)
        check = data.get("check")
        if check is not None:
            if not isinstance(check, str) or not (
                check in _CHECK_KINDS or check.startswith(_CHECK_PREFIXES)
            ):
                raise EvalSetError(f"{where}: \"check\" must be \"json\", \"exact\", "
                                   "\"contains:<text>\" or \"regex:<re>\"")
            if check == "exact" and expected is None:
                raise EvalSetError(f"{where}: \"check\": \"exact\" needs \"expected\"")
            if check.startswith("regex:"):
                try:
                    re.compile(check[len("regex:"):])
                except re.error as exc:
                    raise EvalSetError(f"{where}: bad regex ({exc})") from None
        tools = data.get("tools")
        if tools is not None and not isinstance(tools, list):
            raise EvalSetError(f"{where}: \"tools\" must be a list")
        max_tokens = data.get("max_tokens")
        if max_tokens is not None and (not isinstance(max_tokens, int) or max_tokens < 1):
            raise EvalSetError(f"{where}: \"max_tokens\" must be a positive integer")
        item_id = str(data.get("id", number))
        if item_id in seen:
            raise EvalSetError(f"{where}: duplicate id {item_id!r}")
        seen.add(item_id)
        items.append(EvalItem(item_id, messages, expected, check, tools, max_tokens))
        if len(items) > MAX_EVAL_ITEMS:
            raise EvalSetError(f"--eval-set {path}: more than {MAX_EVAL_ITEMS} prompts")
    if not items:
        raise EvalSetError(f"--eval-set {path}: no prompts in it")
    return items, digest


_FENCE = re.compile(r"^```[A-Za-z0-9_-]*\s*\n(.*?)\n?```\s*$", re.DOTALL)


def parse_json_answer(text: str) -> tuple[bool, Any]:
    """Whether ``text`` is JSON (one surrounding ```json fence allowed)."""

    body = text.strip()
    fenced = _FENCE.match(body)
    if fenced:
        body = fenced.group(1).strip()
    try:
        return True, json.loads(body)
    except ValueError:
        return False, None


_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize(text: str) -> str:
    return " ".join(_PUNCT.sub(" ", text.lower()).split())


def run_check(item: EvalItem, answer: str) -> bool | None:
    """True/False for a scored prompt, None when it has no check and no expected."""

    check = item.check
    if check is None:
        if item.expected is None:
            return None
        return normalize(answer) == normalize(item.expected)
    if check == "json":
        ok, value = parse_json_answer(answer)
        if not ok:
            return False
        if item.expected is not None:
            exp_ok, exp_value = parse_json_answer(item.expected)
            return value == exp_value if exp_ok else True
        return True
    if check == "exact":
        return answer.strip() == (item.expected or "").strip()
    if check.startswith("contains:"):
        return check[len("contains:"):] in answer
    if check.startswith("regex:"):
        return re.search(check[len("regex:"):], answer) is not None
    return None


def tool_call_errors(tool_calls: Sequence[Mapping[str, Any]]) -> int:
    """Malformed tool calls: no function name, or arguments that are not a JSON object."""

    bad = 0
    for call in tool_calls:
        function = call.get("function") or {}
        name = function.get("name")
        args = function.get("arguments", "")
        if not isinstance(name, str) or not name:
            bad += 1
            continue
        try:
            parsed = json.loads(args) if isinstance(args, str) else args
        except ValueError:
            bad += 1
            continue
        if not isinstance(parsed, dict):
            bad += 1
    return bad


# --------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------


@dataclass
class Answer:
    item_id: str
    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    output_tokens: int = 0
    tokens_estimated: bool = False
    ttft_s: float | None = None
    latency_s: float = 0.0
    finish_reason: str | None = None
    error: str | None = None


def _merge_tool_delta(calls: list[dict[str, Any]], deltas: Sequence[Any]) -> None:
    for delta in deltas:
        if not isinstance(delta, dict):
            continue
        index = delta.get("index", len(calls))
        if not isinstance(index, int) or index < 0 or index > 1000:
            index = len(calls)
        while len(calls) <= index:
            calls.append({"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
        call = calls[index]
        if delta.get("id"):
            call["id"] = delta["id"]
        function = delta.get("function") or {}
        if function.get("name"):
            call["function"]["name"] += function["name"]
        if function.get("arguments"):
            arguments = function["arguments"]
            call["function"]["arguments"] += (
                arguments if isinstance(arguments, str) else json.dumps(arguments)
            )


async def stream_one(
    client: httpx.AsyncClient,
    chat_url: str,
    headers: Mapping[str, str],
    model: str,
    item: EvalItem,
    *,
    max_tokens: int,
    temperature: float,
) -> Answer:
    """One streamed chat completion: the text, tool calls, usage and TTFT
    (time until the first content, reasoning or tool-call delta)."""

    body: dict[str, Any] = {
        "model": model,
        "messages": item.messages,
        "max_tokens": item.max_tokens or max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if item.tools:
        body["tools"] = item.tools
    answer = Answer(item.id)
    pieces: list[str] = []
    chunks = 0
    start = time.perf_counter()
    try:
        async with client.stream("POST", chat_url, headers=headers, json=body) as response:
            if response.status_code != 200:
                text = (await response.aread()).decode(errors="replace")[:200]
                answer.error = f"HTTP {response.status_code}: {text}"
                answer.latency_s = time.perf_counter() - start
                return answer
            async for line in response.aiter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except ValueError:
                    continue
                if not isinstance(event, dict):
                    continue
                usage = event.get("usage")
                if isinstance(usage, dict):
                    answer.prompt_tokens = int(usage.get("prompt_tokens") or 0)
                    answer.output_tokens = int(usage.get("completion_tokens") or 0)
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    content = delta.get("content")
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                    tools = delta.get("tool_calls")
                    if (content or reasoning or tools) and answer.ttft_s is None:
                        answer.ttft_s = time.perf_counter() - start
                    if content:
                        pieces.append(content)
                        chunks += 1
                    if reasoning:
                        chunks += 1
                    if tools:
                        _merge_tool_delta(answer.tool_calls, tools)
                        chunks += 1
                    if choice.get("finish_reason"):
                        answer.finish_reason = choice["finish_reason"]
    except httpx.HTTPError as exc:
        answer.error = f"{type(exc).__name__}: {exc}"[:200]
    answer.latency_s = time.perf_counter() - start
    answer.text = "".join(pieces)
    if answer.error is None and answer.output_tokens == 0 and chunks:
        # No usage block from the server: one streamed delta is about one token.
        answer.output_tokens = chunks
        answer.tokens_estimated = True
    return answer


async def run_once(
    chat_url: str,
    headers: Mapping[str, str],
    model: str,
    items: Sequence[EvalItem],
    *,
    concurrency: int,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> tuple[list[Answer], float]:
    """The whole eval set once at ``concurrency``; answers in item order and
    the wall-clock seconds from the first request to the last answer."""

    semaphore = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(transport=_TRANSPORT, timeout=timeout, limits=limits) as client:
        async def one(item: EvalItem) -> Answer:
            async with semaphore:
                return await stream_one(client, chat_url, headers, model, item,
                                        max_tokens=max_tokens, temperature=temperature)

        start = time.perf_counter()
        answers = await asyncio.gather(*(one(item) for item in items))
        wall = time.perf_counter() - start
    return list(answers), wall


def _base_url(chat_url: str) -> str:
    parts = urlsplit(chat_url)
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))


def engine_info(chat_url: str, headers: Mapping[str, str], model: str) -> dict[str, Any]:
    """What the server says about itself: GET /version (vLLM, SGLang) and the
    model's entry in /v1/models. Missing endpoints are simply left out."""

    info: dict[str, Any] = {}
    base = _base_url(chat_url)
    models_url = chat_url[: -len("/chat/completions")] + "/models"
    with httpx.Client(transport=_SYNC_TRANSPORT, timeout=INFO_TIMEOUT) as client:
        for key, url in (("version", base + "/version"), ("models", models_url)):
            try:
                response = client.get(url, headers=headers)
            except httpx.HTTPError:
                continue
            if response.status_code != 200:
                continue
            server = response.headers.get("server")
            if server and "server_header" not in info:
                info["server_header"] = server[:100]
            try:
                data = response.json()
            except ValueError:
                continue
            if key == "version" and isinstance(data, dict):
                version = data.get("version")
                if isinstance(version, str):
                    info["engine_version"] = version[:100]
            elif key == "models" and isinstance(data, dict):
                for entry in data.get("data") or []:
                    if isinstance(entry, dict) and entry.get("id") == model:
                        info["model_entry"] = {
                            k: entry[k] for k in ("id", "owned_by", "root", "max_model_len", "created")
                            if k in entry and isinstance(entry[k], (str, int))
                        }
                        break
                else:
                    ids = [e.get("id") for e in data.get("data") or [] if isinstance(e, dict)]
                    info["served_models"] = [str(i)[:200] for i in ids[:20]]
    return info


# --------------------------------------------------------------------------
# Per-run metrics and the comparison
# --------------------------------------------------------------------------


def run_metrics(
    items: Sequence[EvalItem], answers: Sequence[Answer], wall: float, price: float
) -> dict[str, Any]:
    ok = [a for a in answers if a.error is None]
    out_tokens = sum(a.output_tokens for a in ok)
    dollars = price * wall / 3600.0
    by_id = {item.id: item for item in items}
    scored = passed = 0
    json_checked = json_bad = 0
    tool_answers = tool_bad = 0
    for answer in answers:
        item = by_id[answer.item_id]
        result = None if answer.error else run_check(item, answer.text)
        if item.check is not None or item.expected is not None:
            scored += 1
            passed += 1 if result else 0
        if item.check == "json":
            json_checked += 1
            if answer.error or not parse_json_answer(answer.text)[0]:
                json_bad += 1
        if answer.tool_calls:
            tool_answers += 1
            if tool_call_errors(answer.tool_calls):
                tool_bad += 1
    ttfts = [a.ttft_s for a in ok if a.ttft_s is not None]
    passing_units = passed if scored else len(ok)
    return {
        "wall_s": wall,
        "requests": len(answers),
        "errors": len(answers) - len(ok),
        "output_tokens": out_tokens,
        "tokens_estimated": any(a.tokens_estimated for a in ok),
        "throughput_tok_s": out_tokens / wall if wall > 0 else None,
        "ttft_p50_s": percentile(ttfts, 0.5),
        "ttft_p95_s": percentile(ttfts, 0.95),
        "gpu_dollars": dollars,
        "usd_per_m_output": dollars / out_tokens * 1_000_000 if out_tokens else None,
        "usd_per_passing": dollars / passing_units if passing_units else None,
        "scored": scored,
        "passed": passed,
        "pass_rate": passed / scored if scored else None,
        "json_checked": json_checked,
        "json_error_rate": json_bad / json_checked if json_checked else None,
        "tool_call_answers": tool_answers,
        "tool_call_error_rate": tool_bad / tool_answers if tool_answers else None,
    }


def spread(values: Sequence[float | None]) -> dict[str, Any]:
    clean = [float(v) for v in values if v is not None and math.isfinite(v)]
    if not clean:
        return {"median": None, "min": None, "max": None, "n": 0}
    return {"median": statistics.median(clean), "min": min(clean), "max": max(clean),
            "n": len(clean)}


def overlaps(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    if a["min"] is None or b["min"] is None:
        return True
    return a["min"] <= b["max"] and b["min"] <= a["max"]


SPREAD_METRICS = (
    "throughput_tok_s", "ttft_p50_s", "ttft_p95_s", "usd_per_m_output",
    "usd_per_passing", "pass_rate", "json_error_rate", "tool_call_error_rate",
)


def side_summary(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary = {name: spread([r[name] for r in runs]) for name in SPREAD_METRICS}
    summary["errors"] = sum(r["errors"] for r in runs)
    summary["requests"] = sum(r["requests"] for r in runs)
    summary["tokens_estimated"] = any(r["tokens_estimated"] for r in runs)
    summary["scored"] = runs[0]["scored"] if runs else 0
    return summary


def drift(
    items: Sequence[EvalItem],
    base_runs: Sequence[Sequence[Answer]],
    cand_runs: Sequence[Sequence[Answer]],
) -> dict[str, Any]:
    """Candidate vs baseline, paired by run and prompt. Exact and normalized
    match rates are over prompts with no ``expected``; the baseline's own
    run-to-run match rate is the noise reference."""

    no_expected = {item.id for item in items if item.expected is None}

    def match_rates(left_runs: Sequence[Sequence[Answer]], right_runs: Sequence[Sequence[Answer]],
                    pairs: Sequence[tuple[int, int]]) -> tuple[float | None, float | None]:
        exact = norm = total = 0
        for li, ri in pairs:
            for a, b in zip(left_runs[li], right_runs[ri]):
                if a.item_id not in no_expected or a.error or b.error:
                    continue
                total += 1
                exact += a.text.strip() == b.text.strip()
                norm += normalize(a.text) == normalize(b.text)
        return (exact / total if total else None, norm / total if total else None)

    k = min(len(base_runs), len(cand_runs))
    cand_exact, cand_norm = match_rates(base_runs, cand_runs, [(i, i) for i in range(k)])
    self_pairs = [(i, i + 1) for i in range(len(base_runs) - 1)]
    base_exact, base_norm = match_rates(base_runs, base_runs, self_pairs)

    base_len = cand_len = 0
    for i in range(k):
        for a, b in zip(base_runs[i], cand_runs[i]):
            if a.error or b.error:
                continue
            # Characters, not tokens: the two models may use different
            # tokenizers, and a server without a usage block only has estimates.
            base_len += len(a.text) + sum(len(json.dumps(c)) for c in a.tool_calls)
            cand_len += len(b.text) + sum(len(json.dumps(c)) for c in b.tool_calls)
    length_change = (cand_len - base_len) / base_len * 100 if base_len else None

    # Prompts the baseline passes in most runs and the candidate fails in most.
    by_id = {item.id: item for item in items}

    def majority(runs: Sequence[Sequence[Answer]], item_id: str) -> bool | None:
        votes = []
        for run in runs:
            for answer in run:
                if answer.item_id == item_id:
                    votes.append(bool(not answer.error and run_check(by_id[item_id], answer.text)))
        return sum(votes) * 2 > len(votes) if votes else None

    regressed: list[str] = []
    improved: list[str] = []
    for item in items:
        if item.check is None and item.expected is None:
            continue
        b, c = majority(base_runs, item.id), majority(cand_runs, item.id)
        if b and not c:
            regressed.append(item.id)
        elif c and not b:
            improved.append(item.id)
    return {
        "unscored_prompts": len(no_expected),
        "exact_match_rate": cand_exact,
        "normalized_match_rate": cand_norm,
        "baseline_self_exact_match_rate": base_exact,
        "baseline_self_normalized_match_rate": base_norm,
        "output_length_change_percent": length_change,
        "regressed_items": regressed,
        "regressed_json_items": [i for i in regressed if by_id[i].check == "json"],
        "improved_items": improved,
    }


def decide(base: Mapping[str, Any], cand: Mapping[str, Any], dr: Mapping[str, Any]) -> dict[str, Any]:
    """The verdict and one plain sentence."""

    scored = base.get("scored", 0) > 0
    cost_key = "usd_per_passing" if scored else "usd_per_m_output"
    cost_word = "per passing answer" if scored else "per output token"
    b_cost, c_cost = base[cost_key], cand[cost_key]
    pct = None
    if b_cost["median"] and c_cost["median"] is not None:
        pct = (c_cost["median"] - b_cost["median"]) / b_cost["median"] * 100

    regressed = list(dr["regressed_items"])
    json_regressed = list(dr["regressed_json_items"])
    b_pass, c_pass = base["pass_rate"], cand["pass_rate"]
    b_json, c_json = base["json_error_rate"], cand["json_error_rate"]
    b_tool, c_tool = base["tool_call_error_rate"], cand["tool_call_error_rate"]
    pass_worse = scored and not overlaps(b_pass, c_pass) and c_pass["median"] < b_pass["median"]
    pass_better = scored and not overlaps(b_pass, c_pass) and c_pass["median"] > b_pass["median"]
    json_worse = (c_json["median"] is not None and b_json["median"] is not None
                  and not overlaps(b_json, c_json) and c_json["median"] > b_json["median"])
    tool_worse = (c_tool["median"] is not None and c_tool["min"] > 0 and
                  (b_tool["median"] is None or (not overlaps(b_tool, c_tool)
                                                and c_tool["median"] > b_tool["median"])))

    def fails_text() -> str:
        if json_regressed:
            extra = len(regressed) - len(json_regressed)
            text = f"fails {len(json_regressed)} JSON check{'s' if len(json_regressed) != 1 else ''}"
            if extra:
                text += f" and {extra} other check{'s' if extra != 1 else ''}"
            return text + " the baseline passes"
        if regressed:
            return f"fails {len(regressed)} check{'s' if len(regressed) != 1 else ''} the baseline passes"
        if json_worse:
            return "returns invalid JSON more often"
        if tool_worse:
            return "returns malformed tool calls"
        return "passes fewer checks"

    if cand["errors"] > base["errors"] and cand["errors"] > 0:
        reason = (f"Candidate had {cand['errors']} failed request(s) "
                  f"(baseline {base['errors']}): DON'T SWITCH")
        return {"verdict": VERDICT_DONT, "line": reason, "cost_change_percent": pct}

    if pass_worse or json_worse or tool_worse or (regressed and len(regressed) > len(dr["improved_items"])
                                                   and (c_pass["median"] or 0) < (b_pass["median"] or 0)):
        cost_note = ""
        if pct is not None and pct < 0 and not overlaps(b_cost, c_cost):
            cost_note = f" (it is {abs(pct):.0f}% cheaper {cost_word}, but that does not count)"
        return {"verdict": VERDICT_DONT,
                "line": f"Candidate {fails_text()}{cost_note}: DON'T SWITCH",
                "cost_change_percent": pct}

    checks_text = ("passes more checks" if pass_better else "passes the same checks") if scored \
        else "has no checks to fail (add \"check\" or \"expected\" to your eval set)"
    if pct is None:
        return {"verdict": VERDICT_NOISE,
                "line": "Not enough answers to compare cost: within noise", "cost_change_percent": None}
    if not overlaps(b_cost, c_cost):
        if pct < 0:
            return {"verdict": VERDICT_SWITCH,
                    "line": f"Candidate is {abs(pct):.0f}% cheaper {cost_word} and {checks_text}: SWITCH",
                    "cost_change_percent": pct}
        if pass_better:
            return {"verdict": VERDICT_NOISE,
                    "line": (f"Candidate passes more checks but is {pct:.0f}% more expensive "
                             f"{cost_word}: your call (within noise on the trade-off)"),
                    "cost_change_percent": pct}
        return {"verdict": VERDICT_DONT,
                "line": f"Candidate is {pct:.0f}% more expensive {cost_word}: DON'T SWITCH",
                "cost_change_percent": pct}
    if pass_better:
        return {"verdict": VERDICT_SWITCH,
                "line": f"Candidate passes more checks at the same cost {cost_word} (within noise): SWITCH",
                "cost_change_percent": pct}
    return {"verdict": VERDICT_NOISE,
            "line": (f"Cost {cost_word} differs by {pct:+.0f}% but the run ranges overlap: "
                     "within noise, keep the baseline"),
            "cost_change_percent": pct}


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def _fmt(value: float | None, kind: str) -> str:
    if value is None:
        return "-"
    if kind == "money1k":
        kind, value = "money", value * 1000
    if kind == "money":
        if value >= 100:
            return f"${value:,.0f}"
        if value >= 1:
            return f"${value:.2f}"
        if value <= 0:
            return "$0"
        decimals = min(12, 2 - math.floor(math.log10(value)))
        return f"${value:.{decimals}f}"
    if kind == "pct":
        return f"{value * 100:.0f}%"
    if kind == "ms":
        return f"{value * 1000:.0f}ms"
    return f"{value:,.0f}"


def _range(s: Mapping[str, Any], kind: str) -> str:
    if s["median"] is None:
        return "-"
    mid, low, high = _fmt(s["median"], kind), _fmt(s["min"], kind), _fmt(s["max"], kind)
    if s["n"] < 2 or low == high:
        return mid
    return f"{mid} ({low}..{high})"


ROWS = (
    ("throughput tok/s", "throughput_tok_s", "num", "higher"),
    ("TTFT p50", "ttft_p50_s", "ms", "lower"),
    ("TTFT p95", "ttft_p95_s", "ms", "lower"),
    ("$ / 1M output tok", "usd_per_m_output", "money", "lower"),
    ("$ / 1k passing ans.", "usd_per_passing", "money1k", "lower"),
    ("pass rate", "pass_rate", "pct", "higher"),
    ("JSON errors", "json_error_rate", "pct", "lower"),
    ("tool-call errors", "tool_call_error_rate", "pct", "lower"),
)


def render_table(base: Mapping[str, Any], cand: Mapping[str, Any]) -> list[str]:
    lines = [f"  {'':<20}{'baseline':<30}candidate"]
    for label, key, kind, _ in ROWS:
        b, c = base[key], cand[key]
        if b["median"] is None and c["median"] is None:
            continue
        note = "within noise" if overlaps(b, c) and b["median"] != c["median"] else ""
        lines.append(f"  {label:<20}{_range(b, kind):<29} {_range(c, kind):<29} {note}".rstrip())
    return lines


def handle_live_compare(args: argparse.Namespace, argv: Sequence[str] | None = None) -> int:
    st = style_module.Style()
    missing = [flag for flag, value in (
        ("--baseline", args.live_baseline), ("--baseline-model", args.baseline_model),
        ("--candidate", args.live_candidate), ("--candidate-model", args.candidate_model),
        ("--eval-set", args.eval_set),
    ) if not value]
    if missing:
        print(f"Error: a live comparison needs {', '.join(missing)}", file=sys.stderr)
        return EXIT_USAGE
    if getattr(args, "reports", None):
        print("Error: give either two saved reports or --baseline/--candidate endpoints, not both",
              file=sys.stderr)
        return EXIT_USAGE
    base_price = args.baseline_gpu_price if args.baseline_gpu_price is not None else args.gpu_price
    cand_price = args.candidate_gpu_price if args.candidate_gpu_price is not None else args.gpu_price
    if base_price is None or cand_price is None:
        print("Error: give --gpu-price (or --baseline-gpu-price and --candidate-gpu-price)",
              file=sys.stderr)
        return EXIT_USAGE
    try:
        base_url = normalize_chat_completions_url(
            args.live_baseline, allow_insecure_http=args.allow_insecure_http)
        cand_url = normalize_chat_completions_url(
            args.live_candidate, allow_insecure_http=args.allow_insecure_http)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    try:
        items, digest = load_eval_set(args.eval_set)
    except (EvalSetError, UnicodeDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    started = datetime.now(timezone.utc)
    out_dir = args.out or Path(f"throttle-compare-{started.strftime('%Y%m%dT%H%M%SZ')}")
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"Error: cannot create --out {out_dir}: {exc}", file=sys.stderr)
        return EXIT_USAGE

    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    sides = {
        "baseline": {"url": base_url, "model": args.baseline_model, "price": base_price},
        "candidate": {"url": cand_url, "model": args.candidate_model, "price": cand_price},
    }
    flags = list(argv) if argv is not None else sys.argv[1:]
    manifest = {
        "type": RECORD_TYPE,
        "version": RECORD_VERSION,
        "throttle_version": __version__,
        "created_at": started.isoformat(),
        "command": "throttle " + " ".join(flags),
        "flags": {k: (str(v) if isinstance(v, Path) else v) for k, v in sorted(vars(args).items())
                  if k not in ("func",)},
        "eval_set": {"path": str(args.eval_set), "sha256": digest, "prompts": len(items)},
        "python": platform.python_version(),
        "platform": platform.platform(terse=True),
        "sides": {},
    }
    print(f"{st.brand()} compare {st.dot} {len(items)} prompts {st.dot} {args.runs} runs per side "
          f"{st.dot} concurrency {args.concurrency}")
    for name, side in sides.items():
        info = engine_info(side["url"], headers, side["model"])
        side["info"] = info
        manifest["sides"][name] = {
            "url": side["url"], "model": side["model"], "gpu_price_per_hour": side["price"],
            "engine_version": info.get("engine_version"), "server": info,
        }
        engine = info.get("engine_version") or "engine version unknown"
        print(f"  {name:<9} {side['model']} @ {side['url']} ({engine}, ${side['price']:g}/hr)")
    _write_json(out_dir / "manifest.json", manifest)

    runs: dict[str, list[list[Answer]]] = {"baseline": [], "candidate": []}
    metrics: dict[str, list[dict[str, Any]]] = {"baseline": [], "candidate": []}
    outputs_path = out_dir / "outputs.jsonl"
    by_id = {item.id: item for item in items}
    try:
        with outputs_path.open("w") as outputs:
            for run in range(args.runs):
                for name in ("baseline", "candidate"):  # alternate B C B C ...
                    side = sides[name]
                    answers, wall = asyncio.run(run_once(
                        side["url"], headers, side["model"], items,
                        concurrency=args.concurrency, max_tokens=args.max_tokens,
                        temperature=args.temperature, timeout=args.request_timeout,
                    ))
                    runs[name].append(answers)
                    m = run_metrics(items, answers, wall, side["price"])
                    m["run"] = run + 1
                    metrics[name].append(m)
                    for a in answers:
                        outputs.write(json.dumps({
                            "side": name, "run": run + 1, "model": side["model"],
                            "id": a.item_id, "answer": a.text, "tool_calls": a.tool_calls or None,
                            "pass": None if a.error else run_check(by_id[a.item_id], a.text),
                            "check": by_id[a.item_id].check,
                            "output_tokens": a.output_tokens, "prompt_tokens": a.prompt_tokens,
                            "tokens_estimated": a.tokens_estimated,
                            "ttft_s": a.ttft_s, "latency_s": a.latency_s,
                            "finish_reason": a.finish_reason, "error": a.error,
                        }) + "\n")
                    outputs.flush()
                    status = f"{m['errors']} errors" if m["errors"] else "ok"
                    print(st.dim(f"  run {run + 1}/{args.runs} {name:<9} "
                                 f"{wall:.1f}s, {m['output_tokens']} output tokens, {status}"))
                    if m["errors"] == len(answers):
                        first = next(a.error for a in answers if a.error)
                        print(f"Error: every request to the {name} failed ({first}). "
                              f"Raw outputs: {outputs_path}", file=sys.stderr)
                        return EXIT_FAILED
    except KeyboardInterrupt:
        print(f"\nInterrupted. Partial raw outputs: {outputs_path}", file=sys.stderr)
        return 130

    base_sum = side_summary(metrics["baseline"])
    cand_sum = side_summary(metrics["candidate"])
    dr = drift(items, runs["baseline"], runs["candidate"])
    decision = decide(base_sum, cand_sum, dr)

    print()
    for line in render_table(base_sum, cand_sum):
        print(line)
    print(st.dim("  median (min..max) over the runs; \"within noise\" = the run ranges overlap"))
    drift_bits = []
    if dr["output_length_change_percent"] is not None:
        drift_bits.append(f"answers {dr['output_length_change_percent']:+.0f}% length (characters)")
    if dr["exact_match_rate"] is not None:
        drift_bits.append(
            f"{dr['exact_match_rate'] * 100:.0f}% identical, "
            f"{dr['normalized_match_rate'] * 100:.0f}% same after normalizing "
            f"(baseline vs itself: "
            + (f"{dr['baseline_self_normalized_match_rate'] * 100:.0f}%"
               if dr["baseline_self_normalized_match_rate"] is not None else "-")
            + f") on {dr['unscored_prompts']} prompts with no expected answer"
        )
    if drift_bits:
        print(f"  drift vs baseline: {'; '.join(drift_bits)}")
    if dr["regressed_items"]:
        shown = ", ".join(dr["regressed_items"][:10])
        more = "" if len(dr["regressed_items"]) <= 10 else f" (+{len(dr['regressed_items']) - 10} more)"
        print(f"  fails where the baseline passes: {shown}{more}")
    if base_sum["tokens_estimated"] or cand_sum["tokens_estimated"]:
        print(st.warn("  note: a server sent no usage block; its output tokens are estimated "
                      "from streamed chunks"))
    print(f"  saved: {out_dir}/ (manifest.json, outputs.jsonl, result.json)")

    result = {
        "type": RECORD_TYPE, "version": RECORD_VERSION, "manifest": "manifest.json",
        "baseline": {"summary": base_sum, "runs": metrics["baseline"]},
        "candidate": {"summary": cand_sum, "runs": metrics["candidate"]},
        "drift": dr, "verdict": decision["verdict"], "verdict_line": decision["line"],
        "cost_change_percent": decision["cost_change_percent"],
    }
    _write_json(out_dir / "result.json", result)
    word = decision["verdict"]
    paint = st.good if word == VERDICT_SWITCH else st.bad if word == VERDICT_DONT else st.warn
    print()
    print(st.bold(paint(decision["line"])))
    return EXIT_DONT_SWITCH if word == VERDICT_DONT else EXIT_SWITCH


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")
