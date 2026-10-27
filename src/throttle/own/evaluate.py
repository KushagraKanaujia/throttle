"""Stage 4, EVALUATE: is the candidate as good as the API model, per workload type?

``throttle eval --candidate LABEL`` pairs each captured exchange with its
shadow answer for LABEL, groups the pairs by workload type and compares the
two answers with automatic checks and (unless ``--no-judge``) a pairwise LLM
judge run in both orders.

Eval set
--------
Per type: eligible exchanges (ok, bodies kept, no images) that have a shadow
result for LABEL, with identical prompts deduplicated (first capture wins),
ordered by exchange id. ``evalset_sha256`` is the sha256 of
``{"candidate": LABEL, "exchange_ids": [...]}`` (canonical JSON). The ids are
saved to ``<root>/evals/evalset-<type>-<sha12>.json`` so refresh can rerun the
same set. Shadow results that failed in transport with status 0 (timeout or
connect, not retried) are left out of the set but counted per type as
``transport_excluded``. Every other candidate failure (context overflow, HTTP
error, empty answer, a transport failure that failed again on backfill's
retry) stays in and counts as a LOSS and as a failure on every check.

Checks (pass rate, Wilson 95% CI, scored the same way for both sides)
-------------------------------------------------------------------
``json_valid``        the request asked for JSON; the answer parses as JSON.
``schema_valid``      the request gave a json_schema; the answer satisfies it.
                      Schemas using keywords outside type / properties /
                      required / items / enum / additionalProperties=false are
                      counted as unsupported, not failed.
``exact_match``       classification only; normalized answer equals the
                      incumbent's (so the incumbent scores 1.0 by definition).
``tool_calls_valid``  the request offered tools; every tool call names a
                      listed tool and its arguments are a JSON object. Fails
                      when the answer makes no call but the API model made
                      one or ``tool_choice`` forced one.
``truncated``         every item; passes when the answer was NOT cut off by
                      the token limit (finish_reason length / max_tokens).

Verdict rule (per type)
-----------------------
An item is WIN or LOSS when both judge orders agree, TIE when both say TIE,
and ORDER_DISAGREE otherwise; ORDER_DISAGREE scores 0 like a tie. Let ``n`` be
the judged items (wins + ties + losses + order_disagree), ``net`` the paired
bootstrap 95% CI of mean(win - loss) with item scores +1 / 0 / -1, ``win`` and
``loss`` the Wilson 95% CIs of the win and loss rates over n, ``m`` the
margin, and "a check is worse" mean the candidate's pass-rate CI lies entirely
below the incumbent's (``cand.high < inc.low``). In order:

1. INSUFFICIENT DATA if n < min_n; or judge errors are more than 20% of the
   items sent to the judge; or order_disagree / n > 30% (the judge is too
   position-biased to trust).
2. WORSE if net.high < -m, or any check is worse.
3. INCONCLUSIVE if transport_excluded is more than 10% of the type
   (eval set + transport_excluded): the missing answers could be losses.
4. BETTER if win.low - loss.high > m. Using the Wilson bounds, not only the
   bootstrap, keeps an all-wins or all-ties sample (bootstrap CI of width 0)
   from claiming more than its size supports.
5. NO MATERIAL DIFFERENCE if -m <= net.low and net.high <= m and
   loss.high <= m. The loss bound needs about 75 loss-free items at m = 0.05,
   so a small all-TIE sample is INCONCLUSIVE, not "no difference".
6. INCONCLUSIVE otherwise (the CIs are too wide to tell; get more samples).

With ``--no-judge``, n is the eval set size and the verdict is INSUFFICIENT
DATA (n < min_n), WORSE (a check is worse), INCONCLUSIVE (rule 3) or CHECKS
ONLY. It is never BETTER.

Output: ``<root>/evals/<run_id>.json``
--------------------------------------
::

    {
      "schema": "throttle.own.eval/1",
      "run_id": str, "ts": ISO-8601 UTC,
      "candidate": str, "candidate_model": str | null,
      "judge": {"provider": str, "model": str} | null,
      "margin": float, "min_n": int,
      "judge_calls": int, "judge_usd": float | null,   # null when not priced
      "judge_stopped": str | null,                      # why judging stopped early
      "types": {
        "<type>": {
          "n": int,                       # eval set size
          "judged": int,                  # wins + ties + losses + order_disagree
          "evalset_sha256": str, "evalset_path": str,
          "wins": int, "ties": int, "losses": int,   # ties: both orders said TIE
          "order_disagree": int,          # the two orders disagreed; scored as a tie
          "judge_errors": int, "judge_skipped": int,   # skipped = budget ran out
          "judge_sent": int,              # items sent to the judge (error-rate denominator)
          "candidate_errors": int,        # candidate failed; counted as losses
          "transport_excluded": int,      # status-0 shadow results left out, not judged
          "win_rate_ci":  {"low", "high", "point", "n"},   # Wilson, over judged
          "tie_rate_ci":  {"low", "high", "point", "n"},   # ties + order_disagree
          "loss_rate_ci": {"low", "high", "point", "n"},
          "net_ci": {"estimate", "low", "high", "n", "confidence", "method", "resamples"},
          "checks": {"<name>": {"incumbent": wilson, "candidate": wilson,
                                "unsupported": int}},
          "candidate_latency_ms_p50": float | null,
          "candidate_usage": {"input_tokens": int, "output_tokens": int,
                              "excluded": int},  # results without usage, not estimated
          "verdict": "BETTER" | "NO MATERIAL DIFFERENCE" | "WORSE" |
                     "INCONCLUSIVE" | "INSUFFICIENT DATA" | "CHECKS ONLY",
          "reasons": [str]
        }
      }
    }

All CI bounds are null when there is nothing to compute them from.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import httpx

from . import judge as judge_mod
from . import records
from .shadow import TRANSPORT_STATUS, Classifier, api_key_from_env, eligible, resolve_classifier, workload_type
from .. import statistics as stats
from ..evalcompare import normalize as normalize_text
from ..evalcompare import parse_json_answer, tool_call_errors

SCHEMA_ID = "throttle.own.eval/1"
BETTER = "BETTER"
NO_DIFF = "NO MATERIAL DIFFERENCE"
WORSE = "WORSE"
INCONCLUSIVE = "INCONCLUSIVE"
INSUFFICIENT = "INSUFFICIENT DATA"
CHECKS_ONLY = "CHECKS ONLY"
MAX_JUDGE_ERROR_RATE = 0.20
MAX_ORDER_DISAGREE_RATE = 0.30
MAX_TRANSPORT_EXCLUDED_RATE = 0.10
TRUNCATED_REASONS = {"length", "max_tokens"}
CHECK_NAMES = ("json_valid", "schema_valid", "exact_match", "tool_calls_valid", "truncated")


# ---------------------------------------------------------------------------
# JSON schema (small subset)
# ---------------------------------------------------------------------------

_SUPPORTED = {"type", "properties", "required", "items", "enum", "additionalProperties"}
_ANNOTATIONS = {"title", "description", "$schema", "$id", "default", "examples", "$comment"}


def schema_supported(schema: Any) -> bool:
    if not isinstance(schema, Mapping):
        return False
    for key, value in schema.items():
        if key in _ANNOTATIONS:
            continue
        if key not in _SUPPORTED:
            return False
        if key == "additionalProperties" and not isinstance(value, bool):
            return False
        if key == "properties":
            if not isinstance(value, Mapping) or not all(schema_supported(v) for v in value.values()):
                return False
        if key == "items" and not schema_supported(value):
            return False
        if key == "required" and not (isinstance(value, list) and all(isinstance(v, str) for v in value)):
            return False
        if key == "enum" and not isinstance(value, list):
            return False
        if key == "type":
            names = value if isinstance(value, list) else [value]
            if not all(n in _TYPE_CHECKS for n in names):
                return False
    return True


_TYPE_CHECKS: dict[str, Callable[[Any], bool]] = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool)
    or (isinstance(v, float) and v.is_integer()),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def _valid(schema: Mapping[str, Any], value: Any) -> bool:
    if "type" in schema:
        names = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_TYPE_CHECKS[n](value) for n in names):
            return False
    if "enum" in schema and not any(
        value == option and isinstance(value, bool) == isinstance(option, bool) for option in schema["enum"]
    ):
        return False
    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        for name in schema.get("required") or []:
            if name not in value:
                return False
        if schema.get("additionalProperties") is False and any(k not in properties for k in value):
            return False
        for name, sub in properties.items():
            if name in value and not _valid(sub, value[name]):
                return False
    if isinstance(value, list) and "items" in schema:
        if not all(_valid(schema["items"], item) for item in value):
            return False
    return True


def validate_schema(schema: Any, value: Any) -> bool | None:
    """True / False against a small JSON Schema subset; None when the schema
    uses something outside it (counted as unsupported, not failed)."""

    if not schema_supported(schema):
        return None
    return _valid(schema, value)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


@dataclass
class Side:
    text: str | None
    tool_calls: list[dict[str, Any]]
    finish_reason: str | None
    failed: bool = False


def _tool_names(tools: Sequence[Mapping[str, Any]]) -> set[str]:
    names = set()
    for tool in tools:
        function = tool.get("function") if isinstance(tool.get("function"), Mapping) else tool
        name = function.get("name") if isinstance(function, Mapping) else None
        if isinstance(name, str):
            names.add(name)
    return names


def run_checks(norm: records.Normalized, workload: str, side: Side, incumbent_text: str | None) -> dict[str, bool | None]:
    """Applicable checks for one answer: True pass, False fail, None unsupported.
    Checks that do not apply are absent."""

    out: dict[str, bool | None] = {}
    text = side.text or ""
    wants_json = norm.response_format in ("json", "json_schema")
    if wants_json:
        parsed_ok, value = (False, None) if side.failed else parse_json_answer(text)
        out["json_valid"] = parsed_ok
        if norm.response_format == "json_schema" and norm.json_schema is not None:
            if not schema_supported(norm.json_schema):
                out["schema_valid"] = None
            else:
                out["schema_valid"] = parsed_ok and bool(validate_schema(norm.json_schema, value))
    if workload == "classification" and incumbent_text is not None:
        out["exact_match"] = not side.failed and normalize_text(text) == normalize_text(incumbent_text)
    if norm.tools:
        names = _tool_names(norm.tools)
        calls = [{"function": {"name": c.get("name"), "arguments": c.get("arguments") or ""}} for c in side.tool_calls]
        call_expected = bool(norm.output_tool_calls) or records.tool_choice_forces_call(norm.tool_choice)
        out["tool_calls_valid"] = (
            not side.failed and tool_call_errors(calls) == 0
            and all(c.get("name") in names for c in side.tool_calls)
            and not (call_expected and not side.tool_calls)
        )
    out["truncated"] = not side.failed and (side.finish_reason or "") not in TRUNCATED_REASONS
    return out


# ---------------------------------------------------------------------------
# Eval set
# ---------------------------------------------------------------------------


@dataclass
class Pair:
    exchange: records.Exchange
    norm: records.Normalized
    shadow: records.ShadowResult
    workload: str


def prompt_key(norm: records.Normalized) -> str:
    payload = {
        "messages": [[m.role, m.text, m.tool_calls] for m in norm.messages],
        "tools": norm.tools, "format": norm.response_format, "schema": norm.json_schema,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def evalset_sha256(candidate: str, exchange_ids: Sequence[str]) -> str:
    payload = {"candidate": candidate, "exchange_ids": list(exchange_ids)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def pick_shadow(results: Sequence[records.ShadowResult]) -> records.ShadowResult | None:
    """Latest ok result, else the latest failure that is not an unretried
    transport failure (status 0)."""

    usable = [r for r in results if r.status != TRANSPORT_STATUS]
    ok = [r for r in usable if r.ok]
    if ok:
        return ok[-1]
    return usable[-1] if usable else None


def build_eval_sets(
    root: Path | None,
    candidate: str,
    *,
    classify: Classifier | None = None,
    only_ids: Mapping[str, Sequence[str]] | None = None,
    excluded: dict[str, int] | None = None,
) -> dict[str, list[Pair]]:
    """Per-type eval pairs. ``only_ids`` ({type: ids}) restricts to a saved set.
    ``excluded``, when given, is filled with {type: count} of eligible
    exchanges left out because every shadow result was a status-0 transport
    failure."""

    by_exchange: dict[str, list[records.ShadowResult]] = defaultdict(list)
    for result in records.iter_shadow(candidate, root):
        by_exchange[result.exchange_id].append(result)
    if not by_exchange:
        return {}
    classifier = resolve_classifier(classify)
    wanted = {i: t for t, ids in (only_ids or {}).items() for i in ids} if only_ids else None
    groups: dict[str, list[Pair]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    exchanges = sorted(
        (e for e in records.iter_exchanges(root) if e.id in by_exchange),
        key=lambda e: (e.ts, e.id),
    )
    for exchange in exchanges:
        if wanted is not None and exchange.id not in wanted:
            continue
        norm = eligible(exchange)
        if norm is None:
            continue
        shadow = pick_shadow(by_exchange[exchange.id])
        kind = wanted[exchange.id] if wanted is not None else workload_type(exchange, norm, classifier)
        if shadow is None:
            if excluded is not None:
                excluded[kind] = excluded.get(kind, 0) + 1
            continue
        key = prompt_key(norm)
        if key in seen[kind]:
            continue
        seen[kind].add(key)
        groups[kind].append(Pair(exchange, norm, shadow, kind))
    return {k: sorted(v, key=lambda p: p.exchange.id) for k, v in sorted(groups.items())}


def save_evalset(root: Path, workload: str, candidate: str, pairs: Sequence[Pair]) -> tuple[str, Path]:
    ids = [p.exchange.id for p in pairs]
    digest = evalset_sha256(candidate, ids)
    path = root / "evals" / f"evalset-{records.slug(workload)}-{digest[:12]}.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "type": workload, "candidate": candidate, "sha256": digest,
            "exchange_ids": ids, "created": records.utc_now_iso(),
        }, indent=2) + "\n", encoding="utf-8")
    return digest, path


def load_evalset(path: Path) -> tuple[str, str, list[str]]:
    """(type, candidate, exchange_ids) from a saved eval set; checks its sha."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = [str(i) for i in data["exchange_ids"]]
    if evalset_sha256(data["candidate"], ids) != data.get("sha256"):
        raise ValueError(f"{path}: eval set sha256 does not match its ids")
    return str(data["type"]), str(data["candidate"]), ids


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def checks_worse(checks: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> list[str]:
    worse = []
    for name, sides in checks.items():
        inc, cand = sides.get("incumbent") or {}, sides.get("candidate") or {}
        if inc.get("low") is None or cand.get("high") is None:
            continue
        if float(cand["high"]) < float(inc["low"]):
            worse.append(name)
    return worse


def decide(
    *,
    judged: bool,
    n: int,
    min_n: int,
    margin: float,
    net_ci: Mapping[str, Any] | None,
    checks: Mapping[str, Mapping[str, Mapping[str, Any]]],
    judge_errors: int = 0,
    judge_attempts: int = 0,
    order_disagree: int = 0,
    win_rate_ci: Mapping[str, Any] | None = None,
    loss_rate_ci: Mapping[str, Any] | None = None,
    transport_excluded: int = 0,
    type_total: int | None = None,
) -> tuple[str, list[str]]:
    """Apply the verdict rule in the module docstring, in order:

    1. INSUFFICIENT DATA: n < min_n; judge_errors / judge_attempts > 0.20
       (attempts = items sent to the judge); order_disagree / n > 0.30.
    2. WORSE: net.high < -margin, or any check is worse.
    3. INCONCLUSIVE: transport_excluded / type_total > 0.10.
    4. BETTER: win_rate_ci.low - loss_rate_ci.high > margin.
    5. NO MATERIAL DIFFERENCE: -margin <= net.low, net.high <= margin and
       loss_rate_ci.high <= margin.
    6. INCONCLUSIVE otherwise.

    Without ``win_rate_ci`` / ``loss_rate_ci`` neither BETTER nor NO MATERIAL
    DIFFERENCE can be shown. Unjudged runs stop after rule 3 with CHECKS ONLY.
    """

    reasons: list[str] = []
    if n < min_n:
        return INSUFFICIENT, [f"only {n} {'judged ' if judged else ''}items; need at least {min_n}"]
    if judged and judge_attempts and judge_errors / judge_attempts > MAX_JUDGE_ERROR_RATE:
        return INSUFFICIENT, [f"judge failed on {judge_errors} of {judge_attempts} items (over 20%)"]
    if judged and n and order_disagree / n > MAX_ORDER_DISAGREE_RATE:
        return INSUFFICIENT, [
            f"the judge disagreed with itself when answers were swapped on {order_disagree / n:.0%} of items; "
            "use a stronger judge"
        ]
    total = type_total if type_total is not None else n + transport_excluded
    too_many_excluded = bool(total) and transport_excluded / total > MAX_TRANSPORT_EXCLUDED_RATE
    excluded_reason = (
        f"candidate gave no answer (timeout or connect error, not retried) on {transport_excluded} of {total} "
        f"items ({transport_excluded / total:.0%}, over 10%); those could be losses. Rerun `throttle shadow`"
    ) if too_many_excluded else ""
    worse = checks_worse(checks)
    for name in worse:
        cand, inc = checks[name]["candidate"], checks[name]["incumbent"]
        reasons.append(
            f"{name}: candidate pass rate {cand['point']:.0%} is below the API model's {inc['point']:.0%} "
            "(95% CIs do not overlap)"
        )
    if not judged:
        if worse:
            return WORSE, reasons
        if too_many_excluded:
            return INCONCLUSIVE, [excluded_reason]
        return CHECKS_ONLY, ["automatic checks only; no judge, so no quality claim"]
    low = (net_ci or {}).get("low")
    high = (net_ci or {}).get("high")
    if low is None or high is None:
        return INSUFFICIENT, reasons + ["no confidence interval for win minus loss"]
    span = f"win minus loss 95% CI [{low:+.2f}, {high:+.2f}]"
    if high < -margin or worse:
        if high < -margin:
            reasons.insert(0, f"{span} is below -{margin:.2f}")
        return WORSE, reasons
    if too_many_excluded:
        return INCONCLUSIVE, [excluded_reason]
    win_low = (win_rate_ci or {}).get("low")
    loss_high = (loss_rate_ci or {}).get("high")
    if win_low is not None and loss_high is not None and win_low - loss_high > margin:
        return BETTER, [
            f"win rate low bound {win_low:.2f} minus loss rate high bound {loss_high:.2f} is above +{margin:.2f}; {span}"
        ]
    if -margin <= low and high <= margin:
        if loss_high is not None and loss_high <= margin:
            return NO_DIFF, [f"{span} lies within +/-{margin:.2f} and the loss rate is at most {loss_high:.0%}"]
        bound = f"{loss_high:.0%}" if loss_high is not None else "unknown"
        return INCONCLUSIVE, [
            f"{span} lies within +/-{margin:.2f}, but the loss rate could be as high as {bound} "
            f"(95% CI) at this sample size; shadow more requests"
        ]
    return INCONCLUSIVE, [f"{span} is too wide to call at margin {margin:.2f}; shadow more requests"]


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


@dataclass
class TypeOutcome:
    outcomes: list[str] = field(default_factory=list)   # WIN/TIE/LOSS/ORDER_DISAGREE/ERROR, one per item
    skipped: int = 0
    not_sent: int = 0      # outcomes decided without a judge call (a failed answer)


async def _judge_all(
    sets: Mapping[str, Sequence[Pair]],
    *,
    config: judge_mod.JudgeConfig,
    budget: judge_mod.JudgeBudget,
    concurrency: int,
    transport: httpx.AsyncBaseTransport | None,
) -> tuple[dict[str, TypeOutcome], str | None]:
    results = {k: TypeOutcome() for k in sets}
    stopped: list[str] = []
    gate = asyncio.Semaphore(max(1, concurrency))
    async with httpx.AsyncClient(transport=transport) as client:

        async def one(kind: str, pair: Pair) -> None:
            if pair.shadow.ok is False:
                results[kind].outcomes.append(judge_mod.LOSS)
                results[kind].not_sent += 1
                return
            async with gate:
                if stopped:
                    results[kind].skipped += 1
                    return
                incumbent = judge_mod.render_answer(pair.norm.output_text, pair.norm.output_tool_calls)
                candidate = judge_mod.render_answer(pair.shadow.output_text, pair.shadow.tool_calls)
                try:
                    outcome = await judge_mod.judge_both_orders(
                        pair.norm.messages, incumbent, candidate, client=client, config=config,
                        rubric=judge_mod.rubric_for(kind), budget=budget,
                    )
                except judge_mod.BudgetExhausted as exc:
                    if not stopped:
                        stopped.append(str(exc))
                    results[kind].skipped += 1
                    return
            results[kind].outcomes.append(outcome)

        # Items run in eval-set order so a budget cutoff is reproducible
        # at concurrency 1 and close to it otherwise.
        await asyncio.gather(*(one(k, p) for k, pairs in sets.items() for p in pairs))
    return results, (stopped[0] if stopped else None)


def _wilson_pair(values: Sequence[bool | None]) -> tuple[dict[str, Any], int]:
    scored = [v for v in values if v is not None]
    return stats.wilson_interval_95(sum(1 for v in scored if v), len(scored)), len(values) - len(scored)


def summarize_type(
    kind: str,
    pairs: Sequence[Pair],
    *,
    candidate: str,
    root: Path,
    outcome: TypeOutcome | None,
    margin: float,
    min_n: int,
    transport_excluded: int = 0,
) -> dict[str, Any]:
    digest, path = save_evalset(root, kind, candidate, pairs)
    inc_checks: dict[str, list[bool | None]] = defaultdict(list)
    cand_checks: dict[str, list[bool | None]] = defaultdict(list)
    for pair in pairs:
        norm, shadow = pair.norm, pair.shadow
        inc_side = Side(norm.output_text, norm.output_tool_calls, norm.finish_reason)
        cand_side = Side(shadow.output_text, shadow.tool_calls, shadow.finish_reason, failed=not shadow.ok)
        for name, value in run_checks(norm, kind, inc_side, norm.output_text).items():
            inc_checks[name].append(value)
        for name, value in run_checks(norm, kind, cand_side, norm.output_text).items():
            cand_checks[name].append(value)
    checks: dict[str, Any] = {}
    for name in CHECK_NAMES:
        if name not in inc_checks:
            continue
        inc_ci, unsupported = _wilson_pair(inc_checks[name])
        cand_ci, _ = _wilson_pair(cand_checks[name])
        checks[name] = {"incumbent": inc_ci, "candidate": cand_ci, "unsupported": unsupported}

    ok_shadows = [p.shadow for p in pairs if p.shadow.ok]
    complete = [s.usage for s in ok_shadows if s.usage.complete]
    usage = {
        "input_tokens": sum(u.input_tokens or 0 for u in complete),
        "output_tokens": sum(u.output_tokens or 0 for u in complete),
        "excluded": len(ok_shadows) - len(complete),
    }
    latency = stats.percentile([s.latency_ms for s in ok_shadows], 0.5)
    candidate_errors = sum(1 for p in pairs if not p.shadow.ok)

    counts = Counter(outcome.outcomes if outcome else [])
    wins, ties, losses, errors, disagree = (
        counts[judge_mod.WIN], counts[judge_mod.TIE], counts[judge_mod.LOSS],
        counts[judge_mod.ERROR], counts[judge_mod.ORDER_DISAGREE],
    )
    judged_n = wins + ties + losses + disagree
    sent = judged_n + errors - (outcome.not_sent if outcome else 0)
    diffs = [1] * wins + [0] * (ties + disagree) + [-1] * losses
    net_ci = stats.paired_bootstrap_interval_95(diffs, seed=int(digest[:8], 16)) if outcome else None
    win_ci = stats.wilson_interval_95(wins, judged_n)
    loss_ci = stats.wilson_interval_95(losses, judged_n)
    total = len(pairs) + transport_excluded
    if outcome is not None:
        verdict, reasons = decide(
            judged=True, n=judged_n, min_n=min_n, margin=margin, net_ci=net_ci, checks=checks,
            judge_errors=errors, judge_attempts=sent, order_disagree=disagree,
            win_rate_ci=win_ci, loss_rate_ci=loss_ci,
            transport_excluded=transport_excluded, type_total=total,
        )
        if outcome.skipped:
            reasons.append(f"{outcome.skipped} items not judged: judge budget ran out")
        if disagree and verdict != INSUFFICIENT:
            reasons.append(f"judge disagreed with itself on {disagree} items when answers were swapped (scored as ties)")
    else:
        verdict, reasons = decide(judged=False, n=len(pairs), min_n=min_n, margin=margin, net_ci=None, checks=checks,
                                  transport_excluded=transport_excluded, type_total=total)
    if candidate_errors:
        reasons.append(f"candidate failed on {candidate_errors} items (counted as losses)")
    if transport_excluded and not any("not retried" in x for x in reasons):
        reasons.append(f"{transport_excluded} of {total} items left out: candidate timeout or connect error, not retried")
    return {
        "n": len(pairs),
        "judged": judged_n,
        "evalset_sha256": digest,
        "evalset_path": str(path),
        "wins": wins, "ties": ties, "losses": losses,
        "order_disagree": disagree,
        "judge_errors": errors,
        "judge_skipped": outcome.skipped if outcome else 0,
        "judge_sent": sent if outcome else 0,
        "candidate_errors": candidate_errors,
        "transport_excluded": transport_excluded,
        "win_rate_ci": win_ci,
        "tie_rate_ci": stats.wilson_interval_95(ties + disagree, judged_n),
        "loss_rate_ci": loss_ci,
        "net_ci": net_ci,
        "checks": checks,
        "candidate_latency_ms_p50": latency,
        "candidate_usage": usage,
        "verdict": verdict,
        "reasons": reasons,
    }


def run_eval(
    root: Path,
    candidate: str,
    *,
    judge_config: judge_mod.JudgeConfig | None,
    budget: judge_mod.JudgeBudget | None = None,
    margin: float = 0.05,
    min_n: int = 30,
    classify: Classifier | None = None,
    only_ids: Mapping[str, Sequence[str]] | None = None,
    judge_concurrency: int = 4,
    transport: httpx.AsyncBaseTransport | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Evaluate and write ``<root>/evals/<run_id>.json``; returns the document."""

    excluded: dict[str, int] = {}
    sets = build_eval_sets(root, candidate, classify=classify, only_ids=only_ids, excluded=excluded)
    for kind in excluded:
        sets.setdefault(kind, [])
    sets = dict(sorted(sets.items()))
    outcomes: dict[str, TypeOutcome] | None = None
    stopped = None
    if judge_config is not None and sets:
        budget = budget or judge_mod.JudgeBudget(max_calls=600)
        outcomes, stopped = asyncio.run(_judge_all(
            sets, config=judge_config, budget=budget, concurrency=judge_concurrency, transport=transport,
        ))
    models = Counter(p.shadow.candidate_model for pairs in sets.values() for p in pairs)
    ts = records.utc_now_iso()
    run_id = run_id or f"eval-{ts[:19].replace(':', '').replace('-', '')}-{records.slug(candidate)}-{uuid.uuid4().hex[:6]}"
    document = {
        "schema": SCHEMA_ID,
        "run_id": run_id,
        "ts": ts,
        "candidate": candidate,
        "candidate_model": models.most_common(1)[0][0] if models else None,
        "judge": {"provider": judge_config.provider, "model": judge_config.model} if judge_config else None,
        "margin": margin,
        "min_n": min_n,
        "judge_calls": budget.calls if (budget and judge_config) else 0,
        "judge_usd": round(budget.usd, 6) if (budget and judge_config and budget.price) else None,
        "judge_stopped": stopped,
        "types": {
            kind: summarize_type(
                kind, pairs, candidate=candidate, root=root,
                outcome=(outcomes or {}).get(kind) if outcomes is not None else None,
                margin=margin, min_n=min_n, transport_excluded=excluded.get(kind, 0),
            )
            for kind, pairs in sets.items()
        },
    }
    path = root / "evals" / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    document["path"] = str(path)
    return document


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "eval",
        help="judge your own model against the API model, per workload type",
        description=(
            "Compare a candidate's shadow answers with the API model's captured answers: "
            "automatic checks plus a pairwise judge run in both orders, with 95% confidence "
            "intervals and a verdict per workload type."
        ),
    )
    parser.add_argument("--candidate", required=True, help="candidate label used with `throttle shadow`")
    parser.add_argument("--judge-url", default=None, help="judge endpoint (default: the provider's public API)")
    parser.add_argument("--judge-model", default=None, help="judge model name")
    parser.add_argument("--judge-provider", choices=judge_mod.PROVIDERS, default=records.PROVIDER_OPENAI)
    parser.add_argument("--judge-key-env", default=None, metavar="VAR",
                        help="environment variable holding the judge API key (never printed)")
    parser.add_argument("--no-judge", action="store_true", help="automatic checks only (verdict is CHECKS ONLY)")
    parser.add_argument("--max-judge-calls", type=int, default=600, help="hard cap on judge calls (default 600)")
    parser.add_argument("--judge-max-usd", type=float, default=None, help="hard cap on judge spend, when the price is known")
    parser.add_argument("--judge-concurrency", type=int, default=4, help="judge calls in flight (default 4)")
    parser.add_argument("--margin", type=float, default=0.05, help="material difference in win minus loss (default 0.05)")
    parser.add_argument("--min-n", type=int, default=30, help="minimum judged items per type (default 30)")
    parser.add_argument("--dir", type=Path, default=None, help="data directory (default ~/.throttle/own)")
    parser.add_argument("--json", action="store_true", help="print the result JSON")


def _pct(count: int, total: int) -> str:
    return f"{count / total:.0%}" if total else "-"


def render_table(document: Mapping[str, Any]) -> str:
    from .. import style as style_mod

    s = style_mod.get()
    header = f"{'type':<15}{'n':>5}  {'win/tie/loss':<14}{'net 95% CI':<17}verdict"
    rows: list[Any] = [s.dim(header)]
    for kind, t in document["types"].items():
        judged = t["judged"]
        tied = t["ties"] + t.get("order_disagree", 0)
        wtl = f"{_pct(t['wins'], judged)}/{_pct(tied, judged)}/{_pct(t['losses'], judged)}" if document["judge"] else "-"
        net = t.get("net_ci") or {}
        ci = f"[{net['low']:+.2f}, {net['high']:+.2f}]" if net.get("low") is not None else "-"
        rows.append(f"{kind[:14]:<15}{t['n']:>5}  {wtl:<14}{ci:<17}{s.verdict(t['verdict'])}")
    if not document["types"]:
        rows.append(("No shadow answers matched captured requests.", ()))
    judge = document["judge"]
    rows.append("")
    for kind, t in document["types"].items():
        if t["reasons"]:
            rows.append((f"{kind}: {t['reasons'][0]}", (style_mod.DIM,), "  "))
        if document["judge"] and t.get("order_disagree"):
            rows.append((f"{kind}: the judge flipped its answer on {t['order_disagree']} of {t['judged']} items "
                         "when the answers were swapped; those count as ties above", (style_mod.AMBER,), "  "))
    rows.append((f"Judge: {judge['provider']} {judge['model']}, {document['judge_calls']} calls, both orders."
                  if judge else "No judge: automatic checks only.", (style_mod.DIM,)))
    if document.get("judge_stopped"):
        rows.append((f"Judging stopped early: {document['judge_stopped']}.", (style_mod.AMBER,)))
    rows.append((f"Saved {document.get('path', '')}", (style_mod.DIM,)))
    return s.panel(rows, title=f"Eval: {document['candidate']} vs API model (MEASURED)")


def handle(args: argparse.Namespace, *, classify: Classifier | None = None,
           transport: httpx.AsyncBaseTransport | None = None) -> int:
    root = args.dir or records.default_root()
    config = None
    budget = None
    if not args.no_judge:
        if not args.judge_model:
            print("throttle eval: pass --judge-model (and --judge-url / --judge-key-env as needed), "
                  "or --no-judge for automatic checks only", file=sys.stderr)
            return 2
        try:
            api_key = api_key_from_env(args.judge_key_env)
        except ValueError as exc:
            print(f"throttle eval: {exc}", file=sys.stderr)
            return 2
        url = args.judge_url or judge_mod.DEFAULT_URLS[args.judge_provider]
        if args.judge_provider == records.PROVIDER_OPENAI and args.judge_url:
            from .shadow import chat_url
            url = chat_url(args.judge_url)
        config = judge_mod.JudgeConfig(provider=args.judge_provider, model=args.judge_model, url=url, api_key=api_key)
        price = judge_mod.pricing_for(args.judge_provider, args.judge_model) if args.judge_max_usd is not None else None
        if args.judge_max_usd is not None and price is None:
            print(f"Note: no known price for {args.judge_model}; --judge-max-usd is not enforced, "
                  f"only --max-judge-calls {args.max_judge_calls}.", file=sys.stderr)
        budget = judge_mod.JudgeBudget(max_calls=max(0, args.max_judge_calls), max_usd=args.judge_max_usd, price=price)
    if not any(True for _ in records.iter_shadow(args.candidate, root)):
        print(f"throttle eval: no shadow answers for {args.candidate} in {records.shadow_path(args.candidate, root)}. "
              "Run `throttle shadow` first.", file=sys.stderr)
        return 2
    document = run_eval(
        root, args.candidate, judge_config=config, budget=budget, margin=args.margin,
        min_n=args.min_n, classify=classify, judge_concurrency=args.judge_concurrency, transport=transport,
    )
    if args.json:
        print(json.dumps(document, indent=2, sort_keys=True))
    else:
        print(render_table(document))
    return 0
