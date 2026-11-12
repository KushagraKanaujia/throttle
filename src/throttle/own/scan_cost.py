"""``throttle scan``: the 30-second first run.

It reads the code in a folder, finds every OpenAI and Anthropic call (with
:mod:`throttle.own.scan`, the code behind ``throttle jobs scan``), prices each
one at list price, and names up to two cheaper models to test. It needs no API
key, no network, no GPU and no captured traffic, and it writes nothing.

What it can and cannot say:

* Every dollar figure is an ESTIMATE: list price (from the bundled
  ``prices.json``, via :mod:`throttle.own.pricing`) times an assumed token
  shape per call (default 1,000 tokens in and 300 out, labelled ASSUMED, or
  the user's ``--tokens-in`` / ``--tokens-out``). When the code passes a
  literal output cap (``max_tokens`` and friends) smaller than the assumed
  output, the cap is used, because a call cannot write more than that.
* A cheaper model is only cheaper. Nothing here tests quality, so every
  candidate is UNTESTED. A model with a lower list price can still cost more
  in real use if it writes longer answers or thinks longer; the loop
  (``capture`` -> ``shadow`` -> ``eval`` -> ``report``) measures that.
* ``--calls-per-day`` turns per-call cost into $/month (x 30 days), with the
  same daily call count assumed for every job. Without it the output is per
  1,000 calls.

How candidates are picked is in :func:`pick_candidates`.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import pricing as pricing_mod
from . import records
from . import scan as scan_mod

SCHEMA = "throttle.own.scan/1"
DEFAULT_TOKENS_IN = 1_000
DEFAULT_TOKENS_OUT = 300
MAX_TOKENS_FLAG = 10_000_000
DAYS_PER_MONTH = 30
MAX_CANDIDATES = 2
# A candidate must be at least this much cheaper per call, so the list never
# suggests a switch that saves a rounding error.
MIN_CUT = 0.10
# Models built for one kind of work (code). Never suggested for a general job.
SPECIALIZED = ("codex",)

ESTIMATE = "ESTIMATE"
ASSUMED = "ASSUMED"
UNTESTED = "UNTESTED"

STATUS_PRICED = "priced"
STATUS_RUNTIME = "model_set_at_runtime"
STATUS_UNKNOWN = "model_not_in_price_table"
STATUS_UNPRICED = "not_priced_at_this_size"   # over a long-context threshold the price page does not price

WHY_FAMILY = "same provider, smaller tier of the same family"
WHY_PROVIDER = "same provider, next price step down"
WHY_OTHER = "other provider, similar price to the first pick"
WHY_OTHER_ALONE = "other provider, closest cheaper price"

EXIT_OK = 0
EXIT_USAGE = 2

_DATED = re.compile(r"[-@](\d{4}-\d{2}-\d{2}|\d{8})$")


# ---------------------------------------------------------------------------
# Pricing and candidates
# ---------------------------------------------------------------------------


def per_call_usd(price: pricing_mod.Price, tokens_in: int, tokens_out: int) -> float | None:
    """List-price cost of one call of this shape (no cached input), or
    ``None`` when the price page does not price a prompt this long."""
    usage = records.Usage(input_tokens=tokens_in, output_tokens=tokens_out, cached_input_tokens=0)
    return pricing_mod.cost_usd(price, usage)


def family(price: pricing_mod.Price) -> tuple[str, str]:
    """Which models are tiers of one family, read from the name only.

    * Anthropic names end in the version: ``claude-haiku-4-5`` and
      ``claude-opus-4-5`` are both family ``4-5``.
    * Other names start with it: everything up to and including the first
      part that has a digit, so ``gpt-5-mini`` and ``gpt-5-pro`` are family
      ``gpt-5`` and ``o3-mini`` is ``o3``.
    """
    parts = price.model.lower().split("-")
    if price.provider == records.PROVIDER_ANTHROPIC:
        tail: list[str] = []
        while parts and parts[-1].isdigit():
            tail.insert(0, parts.pop())
        return price.provider, "-".join(tail) or price.model.lower()
    for i, part in enumerate(parts):
        if any(ch.isdigit() for ch in part):
            return price.provider, "-".join(parts[: i + 1])
    return price.provider, price.model.lower()


def _suggestable(price: pricing_mod.Price) -> bool:
    name = price.model.lower()
    return not _DATED.search(name) and not any(word in name for word in SPECIALIZED)


@dataclass(frozen=True)
class Candidate:
    price: pricing_mod.Price
    per_call_usd: float
    why: str


def pick_candidates(current: pricing_mod.Price, table: Iterable[pricing_mod.Price], tokens_in: int,
                    tokens_out: int, *, limit: int = MAX_CANDIDATES) -> list[Candidate]:
    """Up to ``limit`` cheaper models to test instead of ``current``.

    All comparisons are list cost per call at this token shape.

    1. The pool: every other model in the price table that costs at least
       ``MIN_CUT`` (10%) less per call, leaving out dated snapshots (pinned
       old versions) and models built for code only (``SPECIALIZED``).
    2. First pick, same provider: one tier down in the same family (the most
       expensive cheaper model of that family, e.g. ``gpt-4o`` ->
       ``gpt-4o-mini``, ``claude-sonnet-4-5`` -> ``claude-haiku-4-5``). With
       no cheaper tier in the family, the same provider's next price step
       down (the most expensive model in the pool).
    3. Second pick, other provider: the model whose cost per call is closest
       (as a ratio) to the first pick's, so it sits in the same price class.
       With no first pick, the closest to the current model's.

    Ties go to the entry listed first in ``prices.json``. The picks say
    nothing about quality: they are what to test, and are UNTESTED.
    """
    if limit <= 0:
        return []
    now = per_call_usd(current, tokens_in, tokens_out)
    if now is None or now <= 0:
        return []
    pool: list[Candidate] = []
    for price in table:
        if price is current or (price.provider, price.model.lower()) == (current.provider, current.model.lower()):
            continue
        if not _suggestable(price):
            continue
        cost = per_call_usd(price, tokens_in, tokens_out)
        if cost is None or cost > now * (1 - MIN_CUT) * (1 + 1e-9):  # 1e-9: float noise at the edge
            continue
        pool.append(Candidate(price, cost, ""))

    def most_expensive(items: list[Candidate]) -> Candidate | None:
        best: Candidate | None = None
        for item in items:  # table order, so the first listed wins a tie
            if best is None or item.per_call_usd > best.per_call_usd:
                best = item
        return best

    picks: list[Candidate] = []
    same = [c for c in pool if c.price.provider == current.provider]
    in_family = [c for c in same if family(c.price) == family(current)]
    first = most_expensive(in_family)
    if first is not None:
        picks.append(Candidate(first.price, first.per_call_usd, WHY_FAMILY))
    else:
        first = most_expensive(same)
        if first is not None:
            picks.append(Candidate(first.price, first.per_call_usd, WHY_PROVIDER))

    target = first.per_call_usd if first is not None else now
    best_other: Candidate | None = None
    best_gap = math.inf
    for item in pool:
        if item.price.provider == current.provider or item.per_call_usd <= 0:
            continue
        gap = abs(math.log(item.per_call_usd / target))
        if gap < best_gap - 1e-12:
            best_other, best_gap = item, gap
    if best_other is not None:
        why = WHY_OTHER if first is not None else WHY_OTHER_ALONE
        picks.append(Candidate(best_other.price, best_other.per_call_usd, why))
    return picks[:limit]


# ---------------------------------------------------------------------------
# The scan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Shape:
    tokens_in: int = DEFAULT_TOKENS_IN
    tokens_out: int = DEFAULT_TOKENS_OUT
    calls_per_day: int | None = None
    tokens_in_given: bool = False
    tokens_out_given: bool = False

    @property
    def calls_per_month(self) -> int | None:
        return None if self.calls_per_day is None else self.calls_per_day * DAYS_PER_MONTH


@dataclass
class JobCost:
    site: scan_mod.CallSite
    status: str
    tokens_out: int
    capped: bool = False
    price: pricing_mod.Price | None = None
    per_call_usd: float | None = None
    candidates: list[Candidate] = field(default_factory=list)

    @property
    def priced(self) -> bool:
        return self.status == STATUS_PRICED


def _lookup(table: pricing_mod.PriceTable, site: scan_mod.CallSite) -> pricing_mod.Price | None:
    model = (site.model or "").strip()
    provider = site.provider
    for prefix, named in (("openai/", records.PROVIDER_OPENAI), ("anthropic/", records.PROVIDER_ANTHROPIC)):
        if model.lower().startswith(prefix):  # litellm style "anthropic/claude-..."
            model, provider = model[len(prefix):], named
    if provider in records.PROVIDERS:
        return table.lookup(provider, model)
    for name in records.PROVIDERS:
        found = table.lookup(name, model)
        if found is not None:
            return found
    return None


def cost_job(site: scan_mod.CallSite, table: pricing_mod.PriceTable, shape: Shape) -> JobCost:
    tokens_out = shape.tokens_out
    capped = False
    if site.max_tokens is not None and site.max_tokens < tokens_out:
        tokens_out, capped = site.max_tokens, True
    if not site.model:
        return JobCost(site, STATUS_RUNTIME, tokens_out, capped)
    price = _lookup(table, site)
    if price is None:
        return JobCost(site, STATUS_UNKNOWN, tokens_out, capped)
    cost = per_call_usd(price, shape.tokens_in, tokens_out)
    if cost is None:
        return JobCost(site, STATUS_UNPRICED, tokens_out, capped, price)
    candidates = pick_candidates(price, table, shape.tokens_in, tokens_out)
    return JobCost(site, STATUS_PRICED, tokens_out, capped, price, cost, candidates)


def _per_1k(value: float | None) -> float | None:
    return None if value is None else value * 1000


def _monthly(value: float | None, shape: Shape) -> float | None:
    if value is None or shape.calls_per_month is None:
        return None
    return value * shape.calls_per_month


def _cut(now: float | None, then: float | None) -> float | None:
    if now is None or then is None or now <= 0:
        return None
    return 1 - then / now


def summarize(jobs: Sequence[JobCost], shape: Shape) -> dict[str, Any]:
    priced = [j for j in jobs if j.priced]
    now = sum(j.per_call_usd or 0.0 for j in priced)
    first = sum((j.candidates[0].per_call_usd if j.candidates else j.per_call_usd or 0.0) for j in priced)
    cut = _cut(now, first) if priced else None
    return {
        "calls_found": len(jobs),
        "priced": len(priced),
        "with_candidate": sum(1 for j in priced if j.candidates),
        "not_priced": len(jobs) - len(priced),
        "per_1k_calls_usd": _per_1k(now) if priced else None,
        "first_candidates_per_1k_calls_usd": _per_1k(first) if priced else None,
        "monthly_usd": _monthly(now, shape) if priced else None,
        "first_candidates_monthly_usd": _monthly(first, shape) if priced else None,
        "cut_pct": None if cut is None else round(cut * 100, 1),
        "label": ESTIMATE,
    }


def run_scan(root: Path, shape: Shape, table: pricing_mod.PriceTable) -> list[JobCost]:
    return [cost_job(site, table, shape) for site in scan_mod.scan_dir(root)]


def next_steps(jobs: Sequence[JobCost]) -> list[str]:
    """Copyable commands that prove the biggest suggestion on real traffic."""
    with_pick = [j for j in jobs if j.priced and j.candidates]
    if not with_pick:
        return []
    # The costliest job whose first pick is served at a known OpenAI-compatible
    # URL (so the commands run as printed); else the costliest job.
    openai = [j for j in with_pick if j.candidates[0].price.provider == records.PROVIDER_OPENAI]
    top = max(openai or with_pick, key=lambda j: j.per_call_usd or 0.0)
    pick = top.candidates[0].price
    if pick.provider == records.PROVIDER_OPENAI:
        url, key = "https://api.openai.com/v1", " --api-key-env OPENAI_API_KEY"
    else:
        url, key = "URL", " --api-key-env KEY_VAR"
    return [
        "throttle capture",
        "export OPENAI_BASE_URL=http://127.0.0.1:8790/v1",
        "export ANTHROPIC_BASE_URL=http://127.0.0.1:8790",
        f"throttle shadow --candidate-url {url}{key} --candidate-model {pick.model} --label {pick.model}",
        f"throttle eval --candidate {pick.model} --judge-model gpt-4.1 --judge-key-env OPENAI_API_KEY",
        f"throttle report --candidate {pick.model} --candidate-model {pick.model}",
    ]


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 8)


def _price_json(price: pricing_mod.Price | None) -> dict[str, Any] | None:
    if price is None:
        return None
    data = price.to_json()
    keep = ("provider", "model", "input_per_m", "output_per_m", "as_of", "source", "label", "note")
    return {key: data[key] for key in keep if key in data}


def to_json(path: str, jobs: Sequence[JobCost], shape: Shape, table: pricing_mod.PriceTable) -> dict[str, Any]:
    out_jobs = []
    for j in jobs:
        site = j.site
        candidates = []
        for c in j.candidates:
            candidates.append({
                "model": c.price.model, "provider": c.price.provider, "status": UNTESTED, "why": c.why,
                "price": _price_json(c.price),
                "per_call_usd": _round(c.per_call_usd),
                "per_1k_calls_usd": _round(_per_1k(c.per_call_usd)),
                "monthly_usd": _round(_monthly(c.per_call_usd, shape)),
                "cut_pct": round(_cut(j.per_call_usd, c.per_call_usd) * 100, 1),  # type: ignore[operator]
            })
        out_jobs.append({
            "job": site.job, "path": site.path, "line": site.line, "function": site.function,
            "language": site.language, "provider": site.provider, "api": site.api,
            "model": site.model, "model_expr": site.model_expr,
            "model_source": "literal" if site.model else "runtime",
            "status": j.status,
            "tokens_in": shape.tokens_in, "tokens_out": j.tokens_out,
            "max_tokens": site.max_tokens, "tokens_out_capped_by_max_tokens": j.capped,
            "price": _price_json(j.price),
            "per_call_usd": _round(j.per_call_usd),
            "per_1k_calls_usd": _round(_per_1k(j.per_call_usd)),
            "monthly_usd": _round(_monthly(j.per_call_usd, shape)),
            "candidates": candidates,
        })
    summary = summarize(jobs, shape)
    for key in ("per_1k_calls_usd", "first_candidates_per_1k_calls_usd", "monthly_usd",
                "first_candidates_monthly_usd"):
        summary[key] = _round(summary[key])
    return {
        "schema": SCHEMA,
        "path": path,
        "label": ESTIMATE,
        "assumptions": {
            "tokens_in": shape.tokens_in,
            "tokens_in_source": "flag" if shape.tokens_in_given else ASSUMED,
            "tokens_out": shape.tokens_out,
            "tokens_out_source": "flag" if shape.tokens_out_given else ASSUMED,
            "cached_input_tokens": 0,
            "calls_per_day": shape.calls_per_day,
            "calls_per_day_source": None if shape.calls_per_day is None else "flag",
            "days_per_month": DAYS_PER_MONTH,
            "price_labels": sorted({j.price.label for j in jobs if j.price}
                                   | {c.price.label for j in jobs for c in j.candidates}) or table.labels,
            "quality": UNTESTED,
        },
        "summary": summary,
        "jobs": out_jobs,
        "next_steps": next_steps(jobs),
    }


# ---------------------------------------------------------------------------
# Terminal
# ---------------------------------------------------------------------------


def money(value: float | None) -> str:
    if value is None:
        return "-"
    if value >= 100:
        return f"${value:,.0f}"
    if value >= 0.01:
        return f"${value:,.2f}"
    if value == 0:
        return "$0"
    return f"${value:.4f}".rstrip("0") if value >= 0.0001 else "under $0.0001"


def _fit(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    return text[: max(0, width - 2)] + ".."


def _display(path: str) -> str:
    """The folder name for '.', so the title says what was scanned."""
    if path in (".", "./"):
        return Path.cwd().name or path
    return path


def _continued(command: str, *flags: str) -> list[str]:
    """Break a long shell command before each of ``flags`` with a backslash,
    so each printed line stays short and the whole still pastes as one."""
    lines = [command]
    for flag in flags:
        head, sep, tail = lines[-1].partition(" " + flag)
        if sep:
            lines[-1:] = [head + " \\", "    " + flag + tail]
    return lines


def _int(value: int) -> str:
    return f"{value:,}"


LOOKS_FOR = (
    "Python: client.chat.completions.create, client.responses.create, client.messages.create "
    "(Anthropic), openai.ChatCompletion.create, litellm.completion, ChatOpenAI / ChatAnthropic.",
    "JavaScript / TypeScript: the same SDK calls, plus the Vercel AI SDK (generateText, streamText, "
    "generateObject, streamObject).",
    f"It skips these folders: {', '.join(sorted(scan_mod.SKIP_DIRS))}.",
)


def render(path: str, jobs: Sequence[JobCost], shape: Shape, table: pricing_mod.PriceTable) -> str:
    from .. import style as style_mod

    st = style_mod.get()
    inner = st.columns if st.narrow else st.width - 4
    arrow = "↳ " if st.unicode else "> "
    monthly = shape.calls_per_day is not None
    cost_head = "$ / MONTH" if monthly else "$ / 1K CALLS"
    cost_w, cut_w = 13, 7
    rest = max(20, inner - cost_w - cut_w - 2)
    model_w = min(26, max(18, rest // 2))
    job_w = max(8, rest - model_w)

    def row(job: str, model: str, cost: str, cut: str) -> str:
        return f"{_fit(job, job_w):<{job_w}} {_fit(model, model_w):<{model_w}}{cost:>{cost_w}} {cut:>{cut_w - 1}}"

    def amount(per_call: float | None) -> str:
        return money(_monthly(per_call, shape) if monthly else _per_1k(per_call))

    if not jobs:
        rows: list[Any] = [
            (f"No OpenAI or Anthropic calls found in {_display(path)}.", ()),
            "",
            ("What scan looks for:", ()),
            *[(f"- {text}", (), "  ") for text in LOOKS_FOR],
            "",
            ("Calls through your own wrapper or another SDK are not found yet. Tag one with a "
             "'# throttle-job: name' comment on the line above the call to name it.", ()),
        ]
        return "\n".join([
            st.panel(rows, title=f"Throttle scan: {_display(path)}"),
            "",
            "No code to scan yet? See the whole loop on made-up data:  throttle demo",
        ])

    rows = [st.bold(row("JOB", "MODEL", cost_head, "CHANGE")), ""]
    for j in jobs:
        site = j.site
        name = site.tag or site.function
        where = site.location
        if j.status == STATUS_RUNTIME:
            rows.append(row(name, "set at runtime", "-", ""))
            rows.append(st.dim(row("  " + where, "  " + (site.model_expr or ""), "", "")))
        elif j.status == STATUS_UNKNOWN:
            rows.append(row(name, site.model or "", "no price", ""))
            rows.append(st.dim(row("  " + where, "  not in price table", "", "")))
        elif j.status == STATUS_UNPRICED:
            rows.append(row(name, site.model or "", "no price", ""))
            rows.append(st.dim(row("  " + where, "  prompt too long to price", "", "")))
        else:
            mark = "*" if j.capped else ""
            rows.append(row(name, (site.model or "") + mark, amount(j.per_call_usd), ""))
            if not j.candidates:
                rows.append(st.dim(row("  " + where, "  no cheaper model listed", "", "")))
            for i, c in enumerate(j.candidates):
                left = "  " + where if i == 0 else ""
                cut = _cut(j.per_call_usd, c.per_call_usd)
                cut_text = f"-{cut * 100:.0f}%" if cut is not None else ""
                line = row(left, arrow + c.price.model, amount(c.per_call_usd), cut_text)
                head, tail = line[:job_w], line[job_w:]
                rows.append(st.dim(head) + tail[: -cut_w + 1] + st.good(tail[-cut_w + 1:]))
        rows.append("")

    summary = summarize(jobs, shape)
    rows.append(st.dim(f"{arrow}cheaper at list price, {UNTESTED} on your traffic."))
    if any(j.capped for j in jobs):
        rows.append(st.dim("* output capped by a literal max_tokens in the code."))
    rows.append("")
    found = summary["calls_found"]
    head = f"{found} AI call{'s' if found != 1 else ''} found, {summary['priced']} priced"
    if summary["cut_pct"] is not None and summary["with_candidate"]:
        text = (f"{head}; moving the priced ones to the first candidate could cut about "
                f"{summary['cut_pct']:.0f}% of their list-price cost, {ESTIMATE}.")
    elif summary["priced"]:
        text = f"{head}; no cheaper model in the price table for them."
    else:
        text = f"{head}."
    rows.append((text, ("1",)))
    if monthly and summary["monthly_usd"] is not None:
        rows.append((f"At {_int(shape.calls_per_day or 0)} calls a day per job: about "
                     f"{money(summary['monthly_usd'])}/mo now, about "
                     f"{money(summary['first_candidates_monthly_usd'])}/mo on the first candidates "
                     f"({ESTIMATE}).", ()))
    rows.append("")
    if shape.tokens_in_given == shape.tokens_out_given:
        source = "your flags" if shape.tokens_in_given else ASSUMED
        shape_text = f"{_int(shape.tokens_in)} tokens in, {_int(shape.tokens_out)} out, {source}."
    else:
        tin = "your flag" if shape.tokens_in_given else ASSUMED
        tout = "your flag" if shape.tokens_out_given else ASSUMED
        shape_text = (f"{_int(shape.tokens_in)} tokens in ({tin}), "
                      f"{_int(shape.tokens_out)} out ({tout}).")
    rows.append((f"Per call: {shape_text} Change with --tokens-in / --tokens-out.", ()))
    if not monthly:
        rows.append(("Add --calls-per-day N for a monthly figure.", ()))
    else:
        rows.append((f"Per month: --calls-per-day {_int(shape.calls_per_day or 0)} x {DAYS_PER_MONTH} days, "
                     "the same for every job.", ()))
    labels = sorted({j.price.label for j in jobs if j.price} | {c.price.label for j in jobs for c in j.candidates})
    rows.append((f"Prices: {', '.join(labels or table.labels)}, standard tier, no cached input.", ()))
    rows.append(("A lower list price can still cost more if the model writes longer answers or thinks "
                 "longer. Quality is not tested here.", ()))

    unpriced = [j for j in jobs if not j.priced]
    if unpriced:
        rows.append("")
        rows.append((f"Not priced ({len(unpriced)}):", ("1",)))
        for j in unpriced:
            site = j.site
            if j.status == STATUS_RUNTIME:
                why = f"model set at runtime ({site.model_expr})" if site.model_expr else "model set at runtime"
            elif j.status == STATUS_UNKNOWN:
                why = f"{site.model} is not in the price table; add it with --prices FILE"
            else:
                why = f"{site.model}: the price page does not price a prompt this long"
            rows.append((f"- {site.location}  {why}", (), "  "))

    out = [st.panel(rows, title=f"Throttle scan: {_display(path)}")]
    steps = next_steps(jobs)
    if steps:
        out.append("")
        out.append(st.bold("Prove it on your own traffic") + " (see README \"The loop\"):")
        out.append(st.dim("  1. Capture: a local proxy in front of the API, then point your app at it"))
        out.extend(f"     {st.accent(cmd) if i == 0 else cmd}" for i, cmd in enumerate(steps[:3]))
        out.append(st.dim("  2. Shadow: replay captured requests on the candidate (never served)"))
        out.extend(f"     {piece}" for piece in _continued(steps[3], "--api-key-env", "--candidate-model"))
        out.append(st.dim("  3. Eval and report: judge the answers, then price it on measured tokens"))
        out.extend(f"     {piece}" for piece in _continued(steps[4], "--judge-model"))
        out.append(f"     {steps[5]}")
        if "URL" in steps[3].split():
            out.append(st.dim("     URL: an OpenAI-compatible endpoint that serves the candidate."))
    else:
        out.append("")
        out.append("Next: capture real traffic with 'throttle capture' (see README \"The loop\"), "
                   "or try the whole loop on made-up data with 'throttle demo'.")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _token_count(value: str) -> int:
    try:
        parsed = int(value.replace(",", "").replace("_", ""))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a whole number of tokens") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    if parsed > MAX_TOKENS_FLAG:
        raise argparse.ArgumentTypeError(f"must be at most {MAX_TOKENS_FLAG:,}")
    return parsed


def _calls_per_day(value: str) -> int:
    try:
        parsed = int(value.replace(",", "").replace("_", ""))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a whole number of calls") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "scan",
        help="start here: find the AI calls in your code, their list-price cost and cheaper models to test",
        description=(
            "Find every OpenAI and Anthropic call in the code, price each one at list price, and name up "
            "to two cheaper models to test. No API key, no network, no traffic, and it writes nothing. "
            "Every dollar figure is an ESTIMATE from list prices and an assumed token shape per call. "
            "Candidates are UNTESTED: scan says they are cheaper, not that they are as good."
        ),
    )
    parser.add_argument("path", nargs="?", default=".", help="the code to scan (default: current folder)")
    parser.add_argument("--tokens-in", type=_token_count, default=None, metavar="N",
                        help=f"input tokens per call (default {DEFAULT_TOKENS_IN:,}, ASSUMED)")
    parser.add_argument("--tokens-out", type=_token_count, default=None, metavar="N",
                        help=f"output tokens per call (default {DEFAULT_TOKENS_OUT}, ASSUMED; a literal "
                             "max_tokens in the code caps it)")
    parser.add_argument("--calls-per-day", type=_calls_per_day, default=None, metavar="N",
                        help=f"calls a day for each job, to show $/month (x {DAYS_PER_MONTH} days)")
    parser.add_argument("--prices", type=Path, default=None, metavar="FILE",
                        help="JSON price file that adds or replaces prices (labelled ASSUMED)")
    parser.add_argument("--json", action="store_true", help=f"print JSON ({SCHEMA}) instead of the table")


def handle(args: argparse.Namespace) -> int:
    root = Path(args.path).expanduser()
    if not root.is_dir():
        print(f"throttle scan: {args.path} is not a folder", file=sys.stderr)
        return EXIT_USAGE
    try:
        table = pricing_mod.load_prices(args.prices)
    except ValueError as exc:
        print(f"throttle scan: {exc}", file=sys.stderr)
        return EXIT_USAGE
    shape = Shape(
        tokens_in=args.tokens_in or DEFAULT_TOKENS_IN,
        tokens_out=args.tokens_out or DEFAULT_TOKENS_OUT,
        calls_per_day=args.calls_per_day,
        tokens_in_given=args.tokens_in is not None,
        tokens_out_given=args.tokens_out is not None,
    )
    jobs = run_scan(root, shape, table)
    if args.json:
        print(json.dumps(to_json(args.path, jobs, shape, table), indent=2))
    else:
        print(render(args.path, jobs, shape, table))
    return EXIT_OK
