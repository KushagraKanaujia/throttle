"""A first look at a brand-new AI job that has no traffic yet.

For a call site whose prompt Throttle can read, it:

1. asks the judge model to write a few realistic inputs for the prompt's
   ``{slots}`` (default 8);
2. sends each filled-in prompt to the job's API model and to your model;
3. has the judge compare the two answers in both orders, as ``throttle eval``
   does, and runs the same JSON and cut-off checks.

This is labelled FIRST LOOK everywhere. A handful of made-up inputs cannot
prove anything about real traffic, so it never gives a verdict and never
moves a job. It answers "is this worth testing?" and "about what does one
call cost?" before the code ships. The real test starts when real traffic is
captured.

Every call it makes goes to an endpoint the team configured: the API (on the
team's own key), their model and their judge. Calls are capped: one to write
the inputs, then per input one API call, one call to your model and two
judge calls.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import httpx

from . import judge as judge_mod
from . import pricing as pricing_mod
from . import records
from . import scan as scan_mod
from . import workloads
from .evaluate import TRUNCATED_REASONS
from .shadow import chat_url
from ..evalcompare import parse_json_answer

LABEL = "FIRST LOOK"
SCHEMA = "throttle.own.first-look/1"
DEFAULT_INPUTS = 8
MAX_INPUTS = 20
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_API_URLS = {records.PROVIDER_OPENAI: "https://api.openai.com/v1",
                    records.PROVIDER_ANTHROPIC: "https://api.anthropic.com"}
NOTE = ("First look, not enough data yet: a few made-up inputs, not your real traffic. "
        "It is not a verdict and moves nothing.")


@dataclass
class FirstLookConfig:
    candidate_url: str
    candidate_model: str
    candidate_label: str
    judge: judge_mod.JudgeConfig
    api_url: str | None = None            # OpenAI-compatible base for OpenAI jobs (default: OpenAI)
    api_key: str | None = field(default=None, repr=False)
    anthropic_url: str | None = None      # for Anthropic jobs (default: Anthropic)
    anthropic_key: str | None = field(default=None, repr=False)
    candidate_key: str | None = field(default=None, repr=False)
    api_model: str | None = None          # when the code does not name the model literally
    inputs: int = DEFAULT_INPUTS
    max_tokens: int = 512
    timeout: float = 180.0
    concurrency: int = 2

    def cache_key(self, site: scan_mod.CallSite) -> str:
        payload = [site.job, site.prompt_fingerprint, self.model_for(site), self.candidate_model, self.judge.model,
                   self.inputs, site.wants_json]
        return hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:20]

    def model_for(self, site: scan_mod.CallSite) -> str | None:
        return site.model or self.api_model


def can_look(site: scan_mod.CallSite) -> str | None:
    """None when a first look can run on this site, else why not."""
    if site.prompt_fingerprint is None:
        return "its prompt is built at run time, so there is nothing to write test inputs from"
    if site.match_template is None and not (site.user and site.user.slots):
        return "its prompt has too little fixed text to test"
    return None


GEN_SYSTEM = "You write realistic test inputs for software. You answer with JSON only."


def generator_prompt(site: scan_mod.CallSite, slots: Sequence[str], n: int) -> str:
    lines = [
        "An app sends the prompt below to an AI model. The parts in {curly braces} are filled in by the app "
        "with real user data each time.",
        "",
    ]
    if site.system is not None:
        lines += ["SYSTEM MESSAGE:", site.system.text, ""]
    if site.user is not None:
        lines += ["USER MESSAGE:", site.user.text, ""]
    else:
        lines += ["USER MESSAGE: {input} (whatever the user sends)", ""]
    keys = ", ".join(f'"{s}"' for s in slots)
    lines += [
        f"Write {n} different, realistic examples of what the app fills in, the way real users would write them. "
        "Vary the length and difficulty, and make one of them a tricky case.",
        f'Answer with JSON only, in this form: {{"examples": [{{{keys}: "..."}}]}}',
        f"Every example must have exactly these keys: {keys}.",
    ]
    return "\n".join(lines)


def slots_of(site: scan_mod.CallSite) -> list[str]:
    out: list[str] = []
    for template in (site.system, site.user):
        for slot in template.slots if template is not None else ():
            if slot not in out:
                out.append(slot)
    if site.user is None and "input" not in out:
        out.append("input")
    return out


def parse_examples(text: str | None, slots: Sequence[str], n: int) -> list[dict[str, str]]:
    if not text:
        return []
    ok, value = parse_json_answer(text)
    if not ok:
        start, end = text.find("{"), text.rfind("}")
        try:
            value = json.loads(text[start:end + 1]) if 0 <= start < end else None
        except ValueError:
            value = None
    items: list[Any] = []
    if isinstance(value, Mapping):
        found = value.get("examples") or value.get("inputs") or next(
            (v for v in value.values() if isinstance(v, list)), [])
        items = found if isinstance(found, list) else []
    elif isinstance(value, list):
        items = value
    out: list[dict[str, str]] = []
    for item in items:
        if isinstance(item, Mapping):
            row = {s: _text(item.get(s, "")) for s in slots}
            if len(slots) == 1 and not row[slots[0]] and len(item) == 1:
                row = {slots[0]: _text(next(iter(item.values())))}
        elif isinstance(item, str) and len(slots) == 1:
            row = {slots[0]: item}
        else:
            continue
        if any(v.strip() for v in row.values()):
            out.append(row)
    if not out and len(slots) == 1 and text:
        lines = [re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip() for line in text.splitlines()]
        out = [{slots[0]: line} for line in lines if line and not line.startswith(("{", "}", "[", "]", "`"))]
    return out[:n]


def _text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value)


def build_messages(site: scan_mod.CallSite, values: Mapping[str, str]) -> tuple[str | None, str]:
    system = site.system.fill(values) if site.system is not None else None
    user = site.user.fill(values) if site.user is not None else values.get("input", "")
    return system, user


@dataclass
class Answer:
    text: str | None
    finish_reason: str | None
    usage: records.Usage
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text)


RETRIES = 2            # transport failures and 429/5xx are retried with backoff
BACKOFF_SECONDS = 2.0


async def _post(client: httpx.AsyncClient, url: str, **kwargs: Any) -> httpx.Response:
    """POST with retries on a transport error, 429 or 5xx (2 s, then 4 s)."""
    for attempt in range(RETRIES + 1):
        try:
            response = await client.post(url, **kwargs)
        except httpx.HTTPError:
            if attempt == RETRIES:
                raise
        else:
            if response.status_code != 429 and response.status_code < 500 or attempt == RETRIES:
                return response
        await asyncio.sleep(BACKOFF_SECONDS * (2 ** attempt))
    raise AssertionError("unreachable")


async def call_openai(client: httpx.AsyncClient, url: str, model: str, system: str | None, user: str, *,
                      api_key: str | None, max_tokens: int, wants_json: bool, timeout: float) -> Answer:
    messages = ([{"role": "system", "content": system}] if system is not None else []) + [{"role": "user", "content": user}]
    body: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
    if wants_json:
        body["response_format"] = {"type": "json_object"}
    headers = {"content-type": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    try:
        response = await _post(client, chat_url(url), json=body, headers=headers, timeout=timeout)
    except httpx.HTTPError as exc:
        return Answer(None, None, records.Usage(), f"could not reach {chat_url(url)}: {type(exc).__name__}")
    if response.status_code != 200:
        return Answer(None, None, records.Usage(), f"HTTP {response.status_code} from {chat_url(url)}")
    try:
        payload = response.json()
    except ValueError:
        return Answer(None, None, records.Usage(), "the answer was not JSON")
    view = records.normalize_openai(None, payload)
    return Answer(view.output_text, view.finish_reason, records.usage_from_openai(payload))


async def call_anthropic(client: httpx.AsyncClient, url: str, model: str, system: str | None, user: str, *,
                         api_key: str | None, max_tokens: int, timeout: float) -> Answer:
    base = url.rstrip("/")
    target = base if base.endswith("/v1/messages") else (base[:-3] if base.endswith("/v1") else base) + "/v1/messages"
    body: dict[str, Any] = {"model": model, "max_tokens": max_tokens, "temperature": 0,
                            "messages": [{"role": "user", "content": user}]}
    if system is not None:
        body["system"] = system
    headers = {"content-type": "application/json", "anthropic-version": ANTHROPIC_VERSION}
    if api_key:
        headers["x-api-key"] = api_key
    try:
        response = await _post(client, target, json=body, headers=headers, timeout=timeout)
    except httpx.HTTPError as exc:
        return Answer(None, None, records.Usage(), f"could not reach {target}: {type(exc).__name__}")
    if response.status_code != 200:
        return Answer(None, None, records.Usage(), f"HTTP {response.status_code} from {target}")
    try:
        payload = response.json()
    except ValueError:
        return Answer(None, None, records.Usage(), "the answer was not JSON")
    view = records.normalize_anthropic(None, payload)
    return Answer(view.output_text, view.finish_reason, records.usage_from_anthropic(payload))


def _checks(site: scan_mod.CallSite, answer: Answer) -> dict[str, bool]:
    out = {"not_cut_off": answer.ok and (answer.finish_reason or "") not in TRUNCATED_REASONS}
    if site.wants_json:
        out["json_valid"] = answer.ok and parse_json_answer(answer.text or "")[0]
    return out


async def run_first_look(site: scan_mod.CallSite, config: FirstLookConfig, *,
                         prices: pricing_mod.PriceTable | None = None,
                         transport: httpx.AsyncBaseTransport | None = None,
                         root: Path | None = None, lessons: bool = False) -> dict[str, Any]:
    """Run a first look on one call site. Never raises for HTTP failures;
    they are counted and reported. ``lessons=True`` (with ``root``) saves
    judged losses as lessons; ``throttle pr-check`` never does, because a
    pull request may never merge and its made-up inputs are not real traffic."""
    prices = prices or pricing_mod.load_prices()
    model = config.model_for(site)
    base: dict[str, Any] = {
        "schema": SCHEMA, "label": LABEL, "note": NOTE, "job": site.job,
        "prompt_fingerprint": site.prompt_fingerprint, "api_model": model, "provider": site.provider,
        "candidate": config.candidate_label, "candidate_model": config.candidate_model,
        "judge": {"provider": config.judge.provider, "model": config.judge.model},
        "ts": records.utc_now_iso(),
    }
    why_not = can_look(site)
    if why_not is None and not model:
        why_not = "the code does not name the model; pass --api-model"
    if why_not:
        return {**base, "skipped": why_not}
    n = max(1, min(MAX_INPUTS, config.inputs))
    slots = slots_of(site)
    generator_calls = 0
    async with httpx.AsyncClient(transport=transport) as client:
        if slots:
            gen_config = judge_mod.JudgeConfig(provider=config.judge.provider, model=config.judge.model,
                                               url=config.judge.url, api_key=config.judge.api_key,
                                               max_tokens=2048, timeout=config.timeout)
            text = await _generate(client, gen_config, generator_prompt(site, slots, n))
            generator_calls += 1
            examples = parse_examples(text, slots, n)
        else:
            examples = [{}]
        if not examples:
            return {**base, "skipped": "the judge model did not write usable test inputs"}
        gate = asyncio.Semaphore(max(1, config.concurrency))
        kind_probe = records.Exchange(provider=records.PROVIDER_OPENAI, endpoint="/v1/chat/completions",
                                      model=model, request=None, response=None, status=200, latency_ms=0.0)

        async def one(values: Mapping[str, str]) -> dict[str, Any]:
            system, user = build_messages(site, values)
            async with gate:
                if site.provider == records.PROVIDER_ANTHROPIC:
                    api = await call_anthropic(client, config.anthropic_url or DEFAULT_API_URLS[records.PROVIDER_ANTHROPIC],
                                               model, system, user, api_key=config.anthropic_key,
                                               max_tokens=config.max_tokens, timeout=config.timeout)
                else:
                    api = await call_openai(client, config.api_url or DEFAULT_API_URLS[records.PROVIDER_OPENAI],
                                            model, system, user, api_key=config.api_key,
                                            max_tokens=config.max_tokens, wants_json=site.wants_json,
                                            timeout=config.timeout)
                cand = await call_openai(client, config.candidate_url, config.candidate_model, system, user,
                                         api_key=config.candidate_key, max_tokens=config.max_tokens,
                                         wants_json=site.wants_json, timeout=config.timeout)
                messages = ([records.Message("system", system)] if system is not None else []) + [records.Message("user", user)]
                request = {"model": model, "messages": [{"role": m.role, "content": m.text} for m in messages],
                           "max_tokens": config.max_tokens}
                if site.wants_json:
                    request["response_format"] = {"type": "json_object"}
                kind_probe.request = request
                kind = workloads.classify(kind_probe)
                outcome = None
                if api.ok and cand.ok:
                    outcome = await judge_mod.judge_both_orders(
                        messages, api.text or "", cand.text or "", client=client, config=config.judge,
                        rubric=judge_mod.rubric_for(kind),
                    )
                elif api.ok and not cand.ok:
                    outcome = judge_mod.LOSS
            return {"values": dict(values), "system": system, "user": user, "api": api, "candidate": cand,
                    "outcome": outcome, "workload_type": kind}

        items = await asyncio.gather(*(one(v) for v in examples))
    return summarize(site, config, base, items, prices, generator_calls, root=root if lessons else None)


async def _generate(client: httpx.AsyncClient, config: judge_mod.JudgeConfig, prompt: str) -> str | None:
    """One call to the judge model to write test inputs (JSON mode asked for
    on OpenAI-compatible servers)."""
    if config.provider == records.PROVIDER_ANTHROPIC:
        answer = await call_anthropic(client, config.url, config.model, GEN_SYSTEM, prompt, api_key=config.api_key,
                                      max_tokens=config.max_tokens, timeout=config.timeout)
    else:
        answer = await call_openai(client, config.url, config.model, GEN_SYSTEM, prompt, api_key=config.api_key,
                                   max_tokens=config.max_tokens, wants_json=True, timeout=config.timeout)
    return answer.text


def _usage_totals(answers: Sequence[Answer]) -> dict[str, Any]:
    complete = [a.usage for a in answers if a.ok and a.usage.complete]
    return {
        "calls": len(answers), "ok": sum(1 for a in answers if a.ok), "with_usage": len(complete),
        "input_tokens": sum(u.input_tokens or 0 for u in complete),
        "output_tokens": sum(u.output_tokens or 0 for u in complete),
    }


def summarize(site: scan_mod.CallSite, config: FirstLookConfig, base: dict[str, Any], items: Sequence[Mapping[str, Any]],
              prices: pricing_mod.PriceTable, generator_calls: int, *, root: Path | None = None) -> dict[str, Any]:
    outcomes = [i["outcome"] for i in items]
    count = {k: sum(1 for o in outcomes if o == k) for k in (judge_mod.WIN, judge_mod.TIE, judge_mod.LOSS,
                                                             judge_mod.ORDER_DISAGREE, judge_mod.ERROR)}
    api_answers = [i["api"] for i in items]
    cand_answers = [i["candidate"] for i in items]
    model = config.model_for(site)
    price = prices.lookup(site.provider if site.provider != scan_mod.PROVIDER_UNKNOWN else records.PROVIDER_OPENAI, model)
    costs = [pricing_mod.cost_usd(price, a.usage) for a in api_answers if a.ok]
    priced = [c for c in costs if c is not None]
    checks: dict[str, dict[str, Any]] = {}
    for side, answers in (("api", api_answers), ("candidate", cand_answers)):
        for item in answers:
            for name, passed in _checks(site, item).items():
                row = checks.setdefault(name, {"api": 0, "candidate": 0, "n": len(items)})
                row[side] += int(bool(passed))
    judged = count[judge_mod.WIN] + count[judge_mod.TIE] + count[judge_mod.LOSS] + count[judge_mod.ORDER_DISAGREE]
    examples = []
    for item in items[:3]:
        examples.append({
            "input": _clip(item["user"], 240), "api_answer": _clip(item["api"].text, 240),
            "candidate_answer": _clip(item["candidate"].text, 240), "outcome": item["outcome"],
        })
    if root is not None:
        from .jobs import add_lesson

        for item in items:
            if item["outcome"] == judge_mod.LOSS and item["api"].ok:
                norm = records.normalize_openai({"messages": [m for m in (
                    [{"role": "system", "content": item["system"]}] if item["system"] is not None else []) + [
                    {"role": "user", "content": item["user"]}]]}, None)
                add_lesson(root, site.job, source="first_look_loss", reason="judged worse on a made-up input",
                           exchange_id=None, norm=norm, preferred=item["api"].text, rejected=item["candidate"].text,
                           candidate=config.candidate_label, workload_type=item["workload_type"])
    kinds = [i["workload_type"] for i in items]
    return {
        **base,
        "inputs": len(items),
        "judged": judged,
        "wins": count[judge_mod.WIN], "ties": count[judge_mod.TIE], "losses": count[judge_mod.LOSS],
        "order_disagree": count[judge_mod.ORDER_DISAGREE], "judge_errors": count[judge_mod.ERROR],
        "api_errors": sum(1 for a in api_answers if not a.ok),
        "candidate_errors": sum(1 for a in cand_answers if not a.ok),
        "api_usage": _usage_totals(api_answers),
        "candidate_usage": _usage_totals(cand_answers),
        "api_cost_per_call_usd": (sum(priced) / len(priced)) if priced else None,
        "api_cost_label": (price.label if price is not None else None),
        "api_cost_note": None if priced else (f"no list price for {model}" if price is None else "the API reported no token counts"),
        "checks": checks,
        "workload_type": max(set(kinds), key=kinds.count) if kinds else None,
        "examples": examples,
        "calls": {"generator": generator_calls, "api": len(items), "candidate": len(items),
                  "judge": 2 * sum(1 for i in items if i["api"].ok and i["candidate"].ok)},
    }


def _clip(text: str | None, limit: int) -> str | None:
    if text is None:
        return None
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def cache_path(root: Path, key: str) -> Path:
    return Path(root) / "jobs" / "firstlook" / f"{key}.json"


def cached_or_run(site: scan_mod.CallSite, config: FirstLookConfig, *, root: Path,
                  prices: pricing_mod.PriceTable | None = None,
                  transport: httpx.AsyncBaseTransport | None = None) -> dict[str, Any]:
    """A saved first look for the same job, prompt and models, else a new one."""
    path = cache_path(root, config.cache_key(site))
    if path.exists():
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(doc, dict) and doc.get("schema") == SCHEMA:
                doc["cached"] = True
                return doc
        except (OSError, json.JSONDecodeError):
            pass
    doc = asyncio.run(run_first_look(site, config, prices=prices, transport=transport, root=root))
    if not doc.get("skipped"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return doc
