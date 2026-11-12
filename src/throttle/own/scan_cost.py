"""``throttle scan``: the 30-second first run.

It reads the code in a folder, finds the direct OpenAI and Anthropic SDK
calls it can read (with :mod:`throttle.own.scan`, the code behind ``throttle
jobs scan``), prices each one at list price, and names up to two cheaper
models to test. It needs no API key, no network, no GPU and no captured
traffic, and it writes nothing. Calls through a wrapper of the user's own,
another SDK or an agent framework are not found; ``throttle capture`` sees
them on real traffic.

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
  1,000 calls, and the headline cut weights every call site the same.
* A model set at run time is priced only from a literal default written next
  to it (``os.getenv("MODEL", "gpt-4o")``, labelled "default") or from the
  user's ``--model`` (labelled "--model"). Otherwise it is listed, not priced.
* Tests, examples, evals and benchmarks are left out of the totals (``--all``
  includes them), unless every call is in one of them.

How candidates are picked is in :func:`pick_candidates`.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

from . import pricing as pricing_mod
from . import records
from . import scan as scan_mod

SCHEMA = "throttle.own.scan/1"
# The first release with `throttle scan`. Install steps that pin an older
# release must not run it (tests/test_own_scan_cost.py checks the docs).
FIRST_RELEASE = "0.8.0"
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




# How the model of a job was found.
SOURCE_LITERAL = "literal"            # written in the call
SOURCE_DEFAULT = "default_in_code"    # set at run time, with a literal default next to it
SOURCE_FLAG = "flag"                  # set at run time; priced as the user's --model
SOURCE_RUNTIME = "runtime"            # set at run time; not priced

# Where a call lives. Tests, examples and evals are not counted by default.
WHERE_APP = "app"
WHERE_TEST = "test"
WHERE_EXAMPLE = "example"

MAX_ROWS = 15          # priced jobs in the table before "--all"
MAX_NOT_PRICED = 8     # lines in the "Not priced" list before "--all"
EXPR_WIDTH = 40

_TEST_DIRS = frozenset({"test", "tests", "__tests__", "spec", "specs", "__mocks__", "mocks", "fixtures",
                        "e2e", "testing"})
_EXAMPLE_DIRS = frozenset({"example", "examples", "sample", "samples", "eval", "evals", "evaluation",
                           "evaluations", "benchmark", "benchmarks", "bench", "cookbook", "cookbooks"})
_TEST_FILE = re.compile(r"(^test_.*\.py$|_test\.py$|^conftest\.py$|\.(test|spec)\.[cm]?[jt]sx?$"
                        r"|_spec\.[cm]?[jt]sx?$)", re.IGNORECASE)
_EXAMPLE_FILE = re.compile(r"(^|[_.-])(benchmark|bench|eval|evals|example|examples)([_.-][^/]*)?\.[a-z]+$",
                           re.IGNORECASE)
# Model names from other providers (the SDK call can point at any
# OpenAI-compatible server), and older OpenAI / Anthropic models that the
# bundled price table does not carry.
_OTHER_PROVIDER = re.compile(r"^(gemini|gemma|llama|meta-llama|mistral|mixtral|codestral|command|grok|deepseek|"
                             r"qwen|phi|ollama|groq|together|accounts/)", re.IGNORECASE)
_OLDER = re.compile(r"^(gpt-3\.5|gpt-4$|gpt-4-|text-|davinci|babbage|claude-instant|claude-[123]([-.]|$)|"
                    r"claude-(opus|sonnet)-4(-[01])?(-\d{8})?$)", re.IGNORECASE)
_HTTP_VERBS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE"})
_GENERIC_STEMS = frozenset({"index", "route", "main", "__init__", "app", "server", "handler"})


def where(path: str) -> str:
    """``test`` for test and spec files, ``example`` for examples, evals and
    benchmarks, else ``app``."""
    parts = PurePosixPath(path).parts
    dirs = [part.lower() for part in parts[:-1]]
    if any(d in _TEST_DIRS for d in dirs) or (parts and _TEST_FILE.search(parts[-1])):
        return WHERE_TEST
    if any(d in _EXAMPLE_DIRS for d in dirs) or (parts and _EXAMPLE_FILE.search(parts[-1])):
        return WHERE_EXAMPLE
    return WHERE_APP


@dataclass(frozen=True)
class Shape:
    tokens_in: int = DEFAULT_TOKENS_IN
    tokens_out: int = DEFAULT_TOKENS_OUT
    calls_per_day: int | None = None
    tokens_in_given: bool = False
    tokens_out_given: bool = False
    # --model: price every job whose model is set at run time as this model
    runtime_model: str | None = None

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
    model: str | None = None              # the model priced (or looked up)
    model_source: str = SOURCE_RUNTIME
    where: str = WHERE_APP
    counted: bool = True                  # False: in tests or examples, left out of the totals

    @property
    def priced(self) -> bool:
        return self.status == STATUS_PRICED


def _lookup(table: pricing_mod.PriceTable, model: str, provider: str) -> pricing_mod.Price | None:
    """The price for ``model``: the call's provider first, then the other one
    (an OpenAI client can call a Claude model through a compatible endpoint)."""
    model = model.strip()
    for prefix, named in (("openai/", records.PROVIDER_OPENAI), ("anthropic/", records.PROVIDER_ANTHROPIC)):
        if model.lower().startswith(prefix):  # litellm style "anthropic/claude-..."
            model, provider = model[len(prefix):], named
    order = [provider] if provider in records.PROVIDERS else []
    order += [name for name in records.PROVIDERS if name not in order]
    for name in order:
        found = table.lookup(name, model)
        if found is not None:
            return found
    return None


def cost_job(site: scan_mod.CallSite, table: pricing_mod.PriceTable, shape: Shape) -> JobCost:
    tokens_out = shape.tokens_out
    capped = False
    if site.max_tokens is not None and site.max_tokens < tokens_out:
        tokens_out, capped = site.max_tokens, True
    place = where(site.path)
    if site.model:
        model, source = site.model, SOURCE_LITERAL
    elif shape.runtime_model:
        model, source = shape.runtime_model, SOURCE_FLAG
    elif site.model_default:
        model, source = site.model_default, SOURCE_DEFAULT
    else:
        return JobCost(site, STATUS_RUNTIME, tokens_out, capped, where=place)
    price = _lookup(table, model, site.provider)
    if price is None:
        return JobCost(site, STATUS_UNKNOWN, tokens_out, capped, model=model, model_source=source, where=place)
    cost = per_call_usd(price, shape.tokens_in, tokens_out)
    if cost is None:
        return JobCost(site, STATUS_UNPRICED, tokens_out, capped, price, model=model, model_source=source,
                       where=place)
    candidates = pick_candidates(price, table, shape.tokens_in, tokens_out)
    return JobCost(site, STATUS_PRICED, tokens_out, capped, price, cost, candidates, model=model,
                   model_source=source, where=place)


def mark_counted(jobs: Sequence[JobCost], include_all: bool) -> None:
    """Leave tests, examples and evals out of the totals, unless ``include_all``
    or every call is in one of them (a repo of examples is still worth a look)."""
    in_app = any(j.where == WHERE_APP for j in jobs)
    for j in jobs:
        j.counted = include_all or not in_app or j.where == WHERE_APP


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
    """Totals over the counted jobs. Each call site counts once: the code says
    nothing about how often a call runs."""
    counted = [j for j in jobs if j.counted]
    priced = [j for j in counted if j.priced]
    now = sum(j.per_call_usd or 0.0 for j in priced)
    first = sum((j.candidates[0].per_call_usd if j.candidates else j.per_call_usd or 0.0) for j in priced)
    cut = _cut(now, first) if priced else None
    return {
        "calls_found": len(jobs),
        "counted": len(counted),
        "not_counted_tests_examples": len(jobs) - len(counted),
        "priced": len(priced),
        "with_candidate": sum(1 for j in priced if j.candidates),
        "not_priced": len(counted) - len(priced),
        "per_1k_calls_usd": _per_1k(now) if priced else None,
        "first_candidates_per_1k_calls_usd": _per_1k(first) if priced else None,
        "monthly_usd": _monthly(now, shape) if priced else None,
        "first_candidates_monthly_usd": _monthly(first, shape) if priced else None,
        "cut_pct": None if cut is None else round(cut * 100, 1),
        "volume": "the same for every call site",
        "label": ESTIMATE,
    }


def run_scan(root: Path, shape: Shape, table: pricing_mod.PriceTable, *, include_all: bool = False) -> list[JobCost]:
    jobs = [cost_job(site, table, shape) for site in scan_mod.scan_dir(root)]
    mark_counted(jobs, include_all)
    return jobs


JUDGE_OPENAI = ("gpt-4.1", "gpt-4o")
JUDGE_ANTHROPIC = ("claude-sonnet-4-6", "claude-opus-4-8")
CANDIDATE_URL = {
    records.PROVIDER_OPENAI: ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    # Anthropic's OpenAI-compatible endpoint, which `throttle shadow` speaks
    records.PROVIDER_ANTHROPIC: ("https://api.anthropic.com/v1", "ANTHROPIC_API_KEY"),
}


def next_steps(jobs: Sequence[JobCost]) -> list[str]:
    """Copyable commands that prove the biggest suggestion on real traffic.

    The candidate is the first pick of the costliest counted job, served at
    its provider's OpenAI-compatible URL. The judge is an OpenAI model when the
    code calls OpenAI (so the user has that key), else a Claude model."""
    counted = [j for j in jobs if j.counted]
    with_pick = [j for j in counted if j.priced and j.candidates]
    if not with_pick:
        return []
    top = max(with_pick, key=lambda j: j.per_call_usd or 0.0)
    pick = top.candidates[0].price
    url, key = CANDIDATE_URL.get(pick.provider, ("URL", "KEY_VAR"))
    uses_openai = any(j.site.provider == records.PROVIDER_OPENAI
                      or (j.price is not None and j.price.provider == records.PROVIDER_OPENAI) for j in counted)
    if uses_openai:
        judge = next(m for m in JUDGE_OPENAI if m != pick.model)
        judge_flags = f"--judge-model {judge} --judge-key-env OPENAI_API_KEY"
    else:
        judge = next(m for m in JUDGE_ANTHROPIC if m != pick.model)
        judge_flags = f"--judge-provider anthropic --judge-model {judge} --judge-key-env ANTHROPIC_API_KEY"
    return [
        "throttle capture",
        "export OPENAI_BASE_URL=http://127.0.0.1:8790/v1",
        "export ANTHROPIC_BASE_URL=http://127.0.0.1:8790",
        f"throttle shadow --candidate-url {url} --api-key-env {key} --candidate-model {pick.model} "
        f"--label {pick.model}",
        f"throttle eval --candidate {pick.model} {judge_flags}",
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


def _labels(jobs: Sequence[JobCost], table: pricing_mod.PriceTable) -> list[str]:
    counted = [j for j in jobs if j.counted]
    return sorted({j.price.label for j in counted if j.price}
                  | {c.price.label for j in counted for c in j.candidates}) or table.labels


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
            "model_default": site.model_default, "model_spread": site.model_spread,
            "priced_model": j.model, "model_source": j.model_source,
            "where": j.where, "counted": j.counted,
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
            "runtime_model": shape.runtime_model,
            "runtime_model_source": None if shape.runtime_model is None else "flag",
            "price_labels": _labels(jobs, table),
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


def _fit_left(text: str, width: int) -> str:
    """Cut from the left, so a location keeps its file name and line."""
    if len(text) <= width:
        return text
    return ".." + text[len(text) - max(0, width - 2):]


_TS_CAST = re.compile(r"\s+as\s+[\w.$<>\[\]|'\"\s]+$|!$")


def _short(expr: str | None, width: int = EXPR_WIDTH) -> str:
    """Code on one line, whitespace runs collapsed, a TypeScript cast
    (``x as T``) dropped, at most ``width`` long."""
    text = " ".join((expr or "").split())
    return _fit(_TS_CAST.sub("", text) or text, width)


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


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def job_name(site: scan_mod.CallSite) -> str:
    """A short name for the table: the tag, else the function, else the file."""
    if site.tag:
        return site.tag
    path = PurePosixPath(site.path)
    place = path.stem if path.stem not in _GENERIC_STEMS else f"{path.parent.name or '.'}/{path.stem}"
    if site.function == "<module>":
        return place
    if site.function in _HTTP_VERBS:
        return f"{path.parent.name or path.stem} {site.function}"
    return site.function


def _model_label(j: JobCost, width: int) -> str:
    suffix = "*" if j.capped else ""
    suffix += {SOURCE_DEFAULT: " (default)", SOURCE_FLAG: " (--model)"}.get(j.model_source, "")
    return _fit(j.model or "", max(4, width - len(suffix))) + suffix


def not_priced_reason(j: JobCost, table: pricing_mod.PriceTable) -> str:
    site = j.site
    if j.status == STATUS_RUNTIME:
        expr = _short(site.model_expr) or _short(site.model_spread)
        return f"model set at runtime ({expr})" if expr else "no model named in the call"
    model = j.model or ""
    via = {SOURCE_DEFAULT: " (default in code)", SOURCE_FLAG: " (your --model)"}.get(j.model_source, "")
    if j.status == STATUS_UNKNOWN:
        if _OTHER_PROVIDER.search(model):
            return f"{model}{via}: not an OpenAI or Anthropic model; add its price with --prices FILE"
        if _OLDER.search(model):
            dates = sorted({p.as_of for p in table if p.origin == pricing_mod.ORIGIN_BUNDLED})
            checked = f", prices checked {dates[-1]}" if dates else ""
            return f"{model}{via}: older model, not in the price table{checked}; add its price with --prices FILE"
        return f"{model}{via} is not in the price table; add it with --prices FILE"
    return f"{model}{via}: the price page does not price a prompt this long"


def _not_priced_lines(jobs: Sequence[JobCost], table: pricing_mod.PriceTable) -> list[str]:
    """One line per (file, reason): many calls of one kind in one file fold."""
    groups: dict[tuple[str, str], list[int]] = {}
    for j in jobs:
        groups.setdefault((j.site.path, not_priced_reason(j, table)), []).append(j.site.line)
    out = []
    for (path, reason), lines in groups.items():
        if len(lines) == 1:
            out.append(f"- {path}:{lines[0]}  {reason}")
        else:
            out.append(f"- {path}: {len(lines)} calls, lines {', '.join(map(str, lines))}  {reason}")
    return out


LOOKS_FOR = (
    "Python: client.chat.completions.create, client.responses.create, client.messages.create "
    "(Anthropic), openai.ChatCompletion.create, litellm.completion, ChatOpenAI / ChatAnthropic.",
    "JavaScript / TypeScript: the same SDK calls, plus the Vercel AI SDK (generateText, streamText, "
    "generateObject, streamObject).",
    f"It skips these folders: {', '.join(sorted(scan_mod.SKIP_DIRS))}.",
)
NOT_FOUND = ("Not found by scan: calls through your own wrapper, another SDK or an agent framework. "
             "'throttle capture' records those from real traffic.")


def render(path: str, jobs: Sequence[JobCost], shape: Shape, table: pricing_mod.PriceTable, *,
           show_all: bool = False) -> str:
    from .. import style as style_mod

    st = style_mod.get()
    inner = st.columns if st.narrow else st.width - 4
    arrow = "↳ " if st.unicode else "> "
    monthly = shape.calls_per_day is not None
    cost_head = "$ / MONTH" if monthly else "$ / 1K CALLS"
    show_cut = inner >= 50
    cost_w = 13
    cut_w = 7 if show_cut else 0
    rest = max(20, inner - cost_w - (cut_w + 2 if show_cut else 1))
    model_w = min(26, max(18, rest // 2))
    job_w = max(8, rest - model_w)

    def row(job: str, model: str, cost: str, cut: str) -> str:
        line = f"{_fit(job, job_w):<{job_w}} {_fit(model, model_w):<{model_w}}{cost:>{cost_w}}"
        return line + (f" {cut:>{cut_w - 1}}" if show_cut else "")

    def amount(per_call: float | None) -> str:
        return money(_monthly(per_call, shape) if monthly else _per_1k(per_call))

    if not jobs:
        rows: list[Any] = [
            (f"No OpenAI or Anthropic calls found in {_display(path)}.", ()),
            "",
            ("What scan looks for:", ()),
            *[(f"- {text}", (), "  ") for text in LOOKS_FOR],
            "",
            ("Calls through your own wrapper, another SDK or an agent framework are not found. "
             "'throttle capture' records them from real traffic.", ()),
        ]
        return "\n".join([
            st.panel(rows, title=f"Throttle scan: {_display(path)}"),
            "",
            "No code to scan yet? See the whole loop on made-up data:  throttle demo",
        ])

    counted = [j for j in jobs if j.counted]
    hidden = len(jobs) - len(counted)
    priced = sorted((j for j in counted if j.priced), key=lambda j: -(j.per_call_usd or 0.0))
    shown = priced if show_all else priced[:MAX_ROWS]

    rows = []
    if shown:
        rows += [st.bold(row("JOB", "MODEL", cost_head, "CHANGE" if show_cut else "")), ""]
    for j in shown:
        site = j.site
        rows.append(row(job_name(site), _model_label(j, model_w), amount(j.per_call_usd), ""))
        where_text = "  " + _fit_left(site.location, job_w - 2)
        if not show_cut:   # narrow: the location gets a line of its own
            rows.append(st.dim("  " + _fit_left(site.location, inner - 2)))
            where_text = ""
        if not j.candidates:
            rows.append(st.dim(row(where_text, "  no cheaper model listed", "", "").rstrip()))
        for i, c in enumerate(j.candidates):
            cut = _cut(j.per_call_usd, c.per_call_usd)
            cut_text = f"-{cut * 100:.0f}%" if cut is not None else ""
            line = row(where_text if i == 0 else "", arrow + c.price.model, amount(c.per_call_usd), cut_text)
            head, tail = line[:job_w], line[job_w:]
            if show_cut:
                rows.append(st.dim(head) + tail[: -cut_w + 1] + st.good(tail[-cut_w + 1:]))
            else:
                rows.append(st.dim(head) + tail)
        rows.append("")
    if len(shown) < len(priced):
        rows.append(st.dim(f"{_plural(len(priced) - len(shown), 'more priced call')}, each cheaper per call "
                           "than these. --all lists them."))
    if any(j.candidates for j in shown):
        rows.append(st.dim(f"{arrow}cheaper at list price, {UNTESTED} on your traffic."))
    if any(j.capped for j in shown):
        rows.append(st.dim("* output capped by a literal max_tokens in the code."))
    if any(j.model_source == SOURCE_DEFAULT for j in shown):
        rows.append(st.dim("(default): model set at runtime, priced at the default written in the code."))
    if any(j.model_source == SOURCE_FLAG for j in shown):
        rows.append(st.dim(f"(--model): model set at runtime, priced as {shape.runtime_model} (your flag)."))
    if shown:
        rows.append("")

    summary = summarize(jobs, shape)
    head = f"{_plural(summary['counted'], 'AI call')} found, {summary['priced']} priced"
    if summary["cut_pct"] is not None and summary["with_candidate"]:
        weight = "" if monthly else ", if every call site runs equally often"
        text = (f"{head}. Moving the priced ones to the first candidate could cut about "
                f"{summary['cut_pct']:.0f}% of their list-price cost{weight} ({ESTIMATE}).")
    elif summary["priced"]:
        text = f"{head}. No cheaper model in the price table for them."
    else:
        text = f"{head}."
    rows.append((text, ("1",)))
    if monthly and summary["monthly_usd"] is not None:
        rows.append((f"At {_int(shape.calls_per_day or 0)} calls a day per job: about "
                     f"{money(summary['monthly_usd'])}/mo now, about "
                     f"{money(summary['first_candidates_monthly_usd'])}/mo on the first candidates "
                     f"({ESTIMATE}).", ()))
    if hidden:
        rows.append((f"Not counted: {hidden} more in tests, examples or evals. --all includes them.", ()))
    elif not show_all and any(j.where != WHERE_APP for j in jobs):
        rows.append((f"All {summary['counted']} are in tests, examples or evals, so they are counted.", ()))
    runtime = [j for j in counted if j.status == STATUS_RUNTIME]
    if runtime and not shape.runtime_model:
        providers = [j.site.provider for j in runtime]
        example = ("claude-sonnet-4-6" if providers.count(records.PROVIDER_ANTHROPIC)
                   > providers.count(records.PROVIDER_OPENAI) else "gpt-4o")
        rows.append((f"{len(runtime)} of {summary['counted']} set the model at runtime. To price them, rerun "
                     f"with the model you use, for example --model {example}.", ()))
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
    rows.append((f"Prices: {', '.join(_labels(jobs, table))}, standard tier, no cached input.", ()))
    rows.append(("A lower list price can still cost more if the model writes longer answers or thinks "
                 "longer. Quality is not tested here.", ()))
    rows.append((NOT_FOUND, ()))

    unpriced = [j for j in counted if not j.priced]
    if unpriced:
        lines = _not_priced_lines(unpriced, table)
        rows.append("")
        rows.append((f"Not priced ({len(unpriced)}):", ("1",)))
        for line in lines if show_all else lines[:MAX_NOT_PRICED]:
            rows.append((line, (), "  "))
        if not show_all and len(lines) > MAX_NOT_PRICED:
            rows.append((f"... {len(lines) - MAX_NOT_PRICED} more. --all lists them.", ()))

    out = [st.panel(rows, title=f"Throttle scan: {_display(path)}")]
    steps = next_steps(jobs)
    if steps:
        out.append("")
        out.append(st.bold("Prove it on your own traffic") + " (see README \"The loop\"):")
        out.append(st.dim("  1. Capture: a local proxy in front of the API, then point your app at it"))
        out.extend(f"     {st.accent(cmd) if i == 0 else cmd}" for i, cmd in enumerate(steps[:3]))
        out.append(st.dim("  2. Shadow: replay captured requests on the candidate (never served)"))
        out.extend(f"     {piece}" for piece in _continued(steps[3], "--api-key-env", "--candidate-model"))
        if "api.anthropic.com" in steps[3]:
            out.append(st.dim("     That is Anthropic's OpenAI-compatible endpoint, offered for testing."))
        out.append(st.dim("  3. Eval and report: judge the answers, then price it on measured tokens"))
        out.extend(f"     {piece}" for piece in _continued(steps[4], "--judge-provider", "--judge-model"))
        out.extend(f"     {piece}" for piece in _continued(steps[5], "--candidate-model"))
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


def _model_name(value: str) -> str:
    name = value.strip()
    if not name or len(name) > 100 or any(ch.isspace() for ch in name):
        raise argparse.ArgumentTypeError("must be one model name, e.g. gpt-4o")
    return name


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "scan",
        help="start here: find the AI calls in your code, their list-price cost and cheaper models to test",
        description=(
            "Find the direct OpenAI and Anthropic SDK calls in the code, price each one at list price, and "
            "name up to two cheaper models to test. No API key, no network, no traffic, and it writes "
            "nothing. Every dollar figure is an ESTIMATE from list prices and an assumed token shape per "
            "call. Candidates are UNTESTED: scan says they are cheaper, not that they are as good. Tests, "
            "examples and evals are left out of the totals unless you pass --all."
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
    parser.add_argument("--model", type=_model_name, default=None, metavar="NAME",
                        help="price every call whose model is set at runtime as NAME (overrides a default "
                             "written in the code)")
    parser.add_argument("--all", action="store_true",
                        help="include tests, examples and evals, and list every row")
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
        runtime_model=getattr(args, "model", None),
    )
    show_all = bool(getattr(args, "all", False))
    jobs = run_scan(root, shape, table, include_all=show_all)
    if args.json:
        print(json.dumps(to_json(args.path, jobs, shape, table), indent=2))
    else:
        print(render(args.path, jobs, shape, table, show_all=show_all))
    return EXIT_OK
