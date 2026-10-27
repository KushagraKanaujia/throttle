"""Stage 5, REPORT: which workload types can move to your own model, and what
that saves per month.

``throttle report --candidate LABEL --gpu-hourly-rate USD --gpus N`` joins
three things:

* captured API traffic (``throttle workloads``): requests and API spend per
  workload type, projected to a month at the captured rate;
* the latest eval run for the candidate (``throttle eval``): the quality
  verdict, win/tie/loss and the candidate's own token usage per request;
* a ``throttle check`` history record for the candidate model: the MEASURED
  self-hosted $/M tokens (input + output) with its 95% CI. The check's GPU
  rate is operator-supplied, so it is ASSUMED.

Without a check record the report still shows quality, but every owned-cost
and savings cell says "not measured" and carries no number.

Cost math
---------
Per type: owned $/mo = requests/mo x mean candidate tokens per request
(measured in the shadow run) x check $/M, as point, CI low and CI high.
Conservative savings for a type = API $/mo - owned CI high.

Portfolio: the deployment is billed every hour whether busy or not, so owned
total = max(floor, sum of MOVE types' token cost), with
floor = gpu_hourly x gpus x 730 h. The floor applies once across all MOVE
types. Conservative savings = API $/mo of MOVE types - owned total at CI high.
If that is not positive there is no savings claim and every MOVE becomes
KEEP ON API.

Recommendation per type (first match wins)
-----------------------------------------
1. NEEDS MORE DATA: no shadow results, or verdict INSUFFICIENT DATA,
   INCONCLUSIVE or CHECKS ONLY (unless 2 applies).
2. KEEP ON API: owned cost measured and conservative savings <= 0.
3. KEEP ON API: verdict WORSE and a format check (json_valid, schema_valid,
   tool_calls_valid) has a candidate pass-rate CI entirely below the API
   model's.
4. NEEDS FINETUNE: verdict WORSE, format checks overlap.
5. NEEDS MORE DATA: verdict BETTER / NO MATERIAL DIFFERENCE but the owned
   cost is not measured.
6. MOVE: verdict BETTER / NO MATERIAL DIFFERENCE, cost measured,
   conservative savings > 0.

Output: ``<out>/report.json`` and ``<out>/report.html``
------------------------------------------------------
::

    {
      "schema": "throttle.own.report/1",
      "generated_at": ISO-8601 UTC,
      "demo": bool,                      # true only for `throttle demo`
      "candidate": str, "candidate_model": str | null,
      "eval_run_id": str | null, "check_id": str | null,
      "window": {"first_ts", "last_ts", "days", "projectable"},
      "traffic_scale": int,              # 1 except in the demo
      "owned_cost": {                    # null without a check record
        "check_id", "model", "label": "MEASURED" | "DEMO",
        "usd_per_m_tokens": {"mean", "low", "high", "n_blocks"},
        "gpu_hourly_rate_usd", "gpu_count", "rate_label": "ASSUMED" | "DEMO",
        "endpoint", "created_at"
      } | null,
      "not_measured_hint": str,          # what to run to get owned cost
      "deployment": {"gpu_hourly_rate_usd", "gpus", "hours_per_month": 730,
                     "floor_usd_per_month", "label"},
      "workloads": [{
        "type": str,
        "requests_captured": int,
        "requests_per_month": float | null,
        "api_usd_per_month": float | null, "api_note": str,
        "candidate_tokens_per_request": {"input", "output", "total", "n"} | null,
        "owned_usd_per_month": {"point", "low", "high"} | null,
        "owned_note": str,               # "not measured: ..." when null
        "conservative_savings_usd_per_month": float | null,
        "quality": {"verdict", "n", "judged", "wins", "ties", "losses", "order_disagree",
                    "win_rate", "tie_rate", "loss_rate", "order_disagree_rate",
                    "net_ci": [low, high] | null,
                    "format_checks": {name: {"incumbent", "candidate", "overlap"}},
                    "reasons": [str]} | null,
        "recommendation": "MOVE" | "NEEDS FINETUNE" | "KEEP ON API" | "NEEDS MORE DATA",
        "reasons": [str]
      }],
      "totals": {
        "monthly_api_usd": float | null,
        "movable_api_usd": float | null, "movable_share": float | null,
        "owned_usd_per_month": {"point", "low", "high"} | null,   # floor applied once
        "floor_applied": bool | null,
        "conservative_savings_usd": float | null,   # null = no savings claim
        "point_savings_usd": float | null,
        "savings_note": str
      },
      "labels": [{"what", "label", "detail"}],
      "does_not_prove": [str]
    }
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import evaluate as ev
from . import pricing as pricing_mod
from . import records
from . import workloads as wl

SCHEMA_ID = "throttle.own.report/1"
MOVE = "MOVE"
NEEDS_FINETUNE = "NEEDS FINETUNE"
KEEP = "KEEP ON API"
MORE_DATA = "NEEDS MORE DATA"
RECOMMENDATIONS = (MOVE, NEEDS_FINETUNE, KEEP, MORE_DATA)
HOURS_PER_MONTH = 730
FORMAT_CHECKS = ("json_valid", "schema_valid", "tool_calls_valid")
QUALITY_OK = (ev.BETTER, ev.NO_DIFF)
QUALITY_UNSURE = (ev.INSUFFICIENT, ev.INCONCLUSIVE, ev.CHECKS_ONLY)
DEMO_LABEL = "DEMO"
DEMO_BANNER = "DEMO: synthetic traffic, not a customer"
EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 2


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def load_eval(root: Path, candidate: str, ref: str | None = "latest") -> dict[str, Any] | None:
    """The eval document ``ref`` (a run id) or the latest one for ``candidate``."""

    directory = root / "evals"
    if not directory.is_dir():
        return None
    if ref and ref != "latest":
        path = directory / f"{ref}.json"
        if not path.exists():
            raise ValueError(f"no eval run {ref!r} in {directory}")
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("candidate") != candidate:
            raise ValueError(f"eval run {ref!r} is for {document.get('candidate')!r}, not {candidate!r}")
        return document
    best: dict[str, Any] | None = None
    for path in directory.glob("*.json"):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(document, dict) or document.get("schema") != ev.SCHEMA_ID:
            continue
        if document.get("candidate") != candidate:
            continue
        if best is None or (document.get("ts") or "", document.get("run_id") or "") > (best.get("ts") or "", best.get("run_id") or ""):
            best = document
    return best


def _matches(record: Mapping[str, Any], models: Sequence[str]) -> bool:
    fingerprint = record.get("fingerprint") or {}
    names = {str(fingerprint.get(k) or "").strip().lower() for k in ("model", "label")} - {""}
    return any(m.strip().lower() in names for m in models if m)


def load_check(check_dir: Path, models: Sequence[str], ref: str | None = "latest") -> dict[str, Any] | None:
    """A ``throttle check`` history record for one of ``models`` (the
    candidate model name or label). ``ref`` is a check id or ``latest``."""

    from .. import check as check_mod

    history, _ = check_mod.load_history(check_dir)
    if ref and ref != "latest":
        for record in history:
            if record.get("id") == ref:
                if not _matches(record, models):
                    model = (record.get("fingerprint") or {}).get("model")
                    raise ValueError(
                        f"check {ref!r} measured {model!r}, not the candidate ({', '.join(m for m in models if m)})"
                    )
                return record
        raise ValueError(f"no check with id {ref!r} in {check_dir}")
    matching = [r for r in history if _matches(r, models)]
    if not matching:
        return None
    return max(matching, key=lambda r: (r.get("created_at") or "", r.get("id") or ""))


def owned_cost(record: Mapping[str, Any] | None) -> tuple[dict[str, Any] | None, str | None]:
    """(owned-cost block, why not) from a check record. Needs the total $/M
    mean and a 95% CI (two or more blocks)."""

    if record is None:
        return None, None
    total = (record.get("result") or {}).get("total") or {}
    mean, low, high = total.get("mean"), total.get("ci_low"), total.get("ci_high")
    if mean is None:
        return None, f"check {record.get('id')} has no $/M for input + output tokens"
    if low is None or high is None:
        return None, f"check {record.get('id')} has no 95% CI (needs at least 2 blocks)"
    fingerprint = record.get("fingerprint") or {}
    demo = bool(record.get("demo"))
    gpus = fingerprint.get("gpu_count") or (fingerprint.get("user_config") or {}).get("gpus") or 1
    return {
        "check_id": record.get("id"),
        "model": fingerprint.get("model"),
        "label": DEMO_LABEL if demo else "MEASURED",
        "usd_per_m_tokens": {"mean": float(mean), "low": float(low), "high": float(high),
                             "n_blocks": total.get("n_blocks")},
        "gpu_hourly_rate_usd": fingerprint.get("gpu_hourly_rate_usd"),
        "gpu_count": int(gpus) if str(gpus).isdigit() else 1,
        "rate_label": DEMO_LABEL if demo else "ASSUMED",
        "endpoint": record.get("endpoint"),
        "created_at": record.get("created_at"),
    }, None


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------


def candidate_tokens(type_eval: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Mean candidate tokens per request from the shadow answers that reported usage."""

    if not type_eval:
        return None
    usage = type_eval.get("candidate_usage") or {}
    counted = int(type_eval.get("n") or 0) - int(type_eval.get("candidate_errors") or 0) - int(usage.get("excluded") or 0)
    if counted <= 0:
        return None
    inp, out = float(usage.get("input_tokens") or 0), float(usage.get("output_tokens") or 0)
    return {"input": inp / counted, "output": out / counted, "total": (inp + out) / counted, "n": counted}


def format_checks(type_eval: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name in FORMAT_CHECKS:
        sides = (type_eval.get("checks") or {}).get(name)
        if not sides:
            continue
        inc, cand = sides.get("incumbent") or {}, sides.get("candidate") or {}
        overlap = not (inc.get("low") is not None and cand.get("high") is not None
                       and float(cand["high"]) < float(inc["low"]))
        out[name] = {"incumbent": inc.get("point"), "candidate": cand.get("point"), "overlap": overlap}
    return out


def _quality(type_eval: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not type_eval:
        return None
    judged = int(type_eval.get("judged") or 0)
    net = type_eval.get("net_ci") or {}
    rate = lambda k: (int(type_eval.get(k) or 0) / judged) if judged else None  # noqa: E731
    return {
        "verdict": type_eval.get("verdict"),
        "n": type_eval.get("n"),
        "judged": judged,
        "wins": type_eval.get("wins"), "ties": type_eval.get("ties"), "losses": type_eval.get("losses"),
        "order_disagree": int(type_eval.get("order_disagree") or 0),
        "win_rate": rate("wins"), "tie_rate": rate("ties"), "loss_rate": rate("losses"),
        "order_disagree_rate": rate("order_disagree"),
        "net_ci": [net["low"], net["high"]] if net.get("low") is not None and net.get("high") is not None else None,
        "format_checks": format_checks(type_eval),
        "reasons": list(type_eval.get("reasons") or []),
    }


def recommend(
    quality: Mapping[str, Any] | None,
    *,
    owned_measured: bool,
    conservative_savings: float | None,
    api_usd: float | None,
    owned_high: float | None,
    not_measured: str,
) -> tuple[str, list[str]]:
    """Apply the rule in the module docstring to one workload type."""

    if quality is None:
        return MORE_DATA, ["no shadow answers for this type yet; run `throttle shadow` and `throttle eval`"]
    verdict = quality.get("verdict")
    cost_loses = owned_measured and conservative_savings is not None and conservative_savings <= 0
    if cost_loses:
        return KEEP, [
            f"owned cost at the high end of its 95% CI (${owned_high:,.0f}/mo) is not below "
            f"API spend (${api_usd:,.0f}/mo), so there is no saving to claim"
        ]
    if verdict in QUALITY_UNSURE:
        first = (quality.get("reasons") or [""])[0]
        return MORE_DATA, [f"quality verdict is {verdict}" + (f": {first}" if first else "")]
    if verdict == ev.WORSE:
        broken = [name for name, c in quality["format_checks"].items() if not c["overlap"]]
        if broken:
            details = ", ".join(
                f"{name} {quality['format_checks'][name]['candidate']:.0%} vs API {quality['format_checks'][name]['incumbent']:.0%}"
                for name in broken
            )
            return KEEP, [
                "the open model loses on quality and breaks the output format: " + details,
                "fix the format first (constrained decoding or a different model), then re-run eval",
            ]
        checked = list(quality["format_checks"])
        why = (f"format checks ({', '.join(checked)}) pass about as often as the API model"
               if checked else "no format checks apply; the gap is in content only")
        return NEEDS_FINETUNE, [
            f"the open model loses on quality ({_wtl(quality)})",
            why + "; this kind of gap is what fine-tuning on your own traffic usually closes",
        ]
    if verdict in QUALITY_OK:
        if not owned_measured:
            return MORE_DATA, [f"quality is {verdict}, but the owned cost is {not_measured}"]
        if api_usd is None:
            return MORE_DATA, [f"quality is {verdict}, but there is no monthly API spend to compare with"]
        return MOVE, [
            f"quality is {verdict} ({_wtl(quality)})",
            f"conservative saving ${conservative_savings:,.0f}/mo before the deployment floor",
        ]
    return MORE_DATA, [f"unknown quality verdict {verdict!r}"]


def _wtl(quality: Mapping[str, Any]) -> str:
    def pct(x: float | None) -> str:
        return "-" if x is None else f"{x:.0%}"

    text = f"win/tie/loss {pct(quality.get('win_rate'))}/{pct(quality.get('tie_rate'))}/{pct(quality.get('loss_rate'))}"
    net = quality.get("net_ci")
    if net:
        text += f", net 95% CI [{net[0]:+.2f}, {net[1]:+.2f}]"
    return text


def build_report(
    exchanges: Sequence[records.Exchange],
    prices: pricing_mod.PriceTable,
    *,
    candidate: str,
    eval_doc: Mapping[str, Any] | None,
    check_record: Mapping[str, Any] | None,
    gpu_hourly_rate: float,
    gpus: int,
    traffic_scale: int = 1,
    demo: bool = False,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """The report document (see the module docstring). ``traffic_scale`` is
    only for the demo: each synthetic request stands for that many."""

    rows = wl.summarize(exchanges, prices)
    window = wl.capture_window(exchanges)
    owned, owned_problem = owned_cost(check_record)
    candidate_model = (eval_doc or {}).get("candidate_model")
    hint = f"run `throttle check --url ... --model {candidate_model or candidate}`"
    not_measured = "not measured: " + (f"{owned_problem}; " if owned_problem else "") + hint
    floor = gpu_hourly_rate * gpus * HOURS_PER_MONTH
    types_eval = (eval_doc or {}).get("types") or {}
    per_m = owned["usd_per_m_tokens"] if owned else None

    workloads: list[dict[str, Any]] = []
    for row in rows:
        served = row.requests - row.excluded.errors
        per_month = served / window.days * 30 * traffic_scale if window.projectable else None
        api = row.monthly_projection_usd * traffic_scale if row.monthly_projection_usd is not None else None
        type_eval = types_eval.get(row.workload_type)
        tokens = candidate_tokens(type_eval)
        owned_cell = None
        owned_note = not_measured
        if per_m and tokens and per_month is not None:
            million = per_month * tokens["total"] / 1_000_000
            owned_cell = {"point": million * per_m["mean"], "low": million * per_m["low"], "high": million * per_m["high"]}
            owned_note = f"{tokens['total']:,.0f} candidate tokens/request x ${per_m['mean']:.3f}/M"
        elif per_m and per_month is None:
            owned_note = "not projected: capture window under 1 day"
        elif per_m:
            owned_note = "not measured: no candidate token usage from the shadow run for this type"
        savings = api - owned_cell["high"] if (api is not None and owned_cell) else None
        quality = _quality(type_eval)
        rec, reasons = recommend(
            quality, owned_measured=owned_cell is not None, conservative_savings=savings,
            api_usd=api, owned_high=owned_cell["high"] if owned_cell else None, not_measured=not_measured,
        )
        if row.workload_type == wl.UNKNOWN and quality is None:
            reasons = ["captured without bodies (metadata only), so it cannot be shadowed or judged"]
        workloads.append({
            "type": row.workload_type,
            "requests_captured": row.requests,
            "requests_per_month": per_month,
            "api_usd_per_month": api,
            "api_note": row.projection_note,
            "candidate_tokens_per_request": tokens,
            "owned_usd_per_month": owned_cell,
            "owned_note": owned_note,
            "conservative_savings_usd_per_month": savings,
            "quality": quality,
            "recommendation": rec,
            "reasons": reasons,
        })

    totals = _portfolio(workloads, floor=floor, measured=owned is not None, not_measured=not_measured)
    labels = _labels(prices, owned, window, demo=demo, traffic_scale=traffic_scale, eval_doc=eval_doc,
                     gpu_hourly_rate=gpu_hourly_rate, gpus=gpus)
    return {
        "schema": SCHEMA_ID,
        "generated_at": generated_at or records.utc_now_iso(),
        "demo": demo,
        "candidate": candidate,
        "candidate_model": candidate_model,
        "eval_run_id": (eval_doc or {}).get("run_id"),
        "check_id": owned["check_id"] if owned else None,
        "window": {"first_ts": window.first_ts, "last_ts": window.last_ts,
                   "days": round(window.days, 4), "projectable": window.projectable},
        "traffic_scale": traffic_scale,
        "owned_cost": owned,
        "not_measured_hint": not_measured,
        "deployment": {"gpu_hourly_rate_usd": gpu_hourly_rate, "gpus": gpus, "hours_per_month": HOURS_PER_MONTH,
                       "floor_usd_per_month": floor, "label": DEMO_LABEL if demo else "ASSUMED"},
        "workloads": workloads,
        "totals": totals,
        "labels": labels,
        "does_not_prove": _limits(owned, eval_doc, demo=demo),
    }


def _portfolio(workloads: list[dict[str, Any]], *, floor: float, measured: bool, not_measured: str) -> dict[str, Any]:
    apis = [w["api_usd_per_month"] for w in workloads]
    monthly_api = sum(a for a in apis if a is not None) if any(a is not None for a in apis) else None
    moving = [w for w in workloads if w["recommendation"] == MOVE]
    movable = sum(w["api_usd_per_month"] for w in moving)
    owned = None
    conservative = point = None
    floor_applied = None
    if not measured:
        note = not_measured
    elif not moving:
        note = "no workload type is ready to move, so there is no savings claim"
    else:
        sums = {k: sum(w["owned_usd_per_month"][k] for w in moving) for k in ("point", "low", "high")}
        owned = {k: max(floor, v) for k, v in sums.items()}
        floor_applied = sums["high"] < floor
        conservative = movable - owned["high"]
        point = movable - owned["point"]
        if conservative <= 0:
            note = (f"the always-on deployment (${owned['high']:,.0f}/mo at the high end) costs more than the "
                    f"${movable:,.0f}/mo of API spend that could move, so there is no savings claim")
            for w in moving:
                w["recommendation"] = KEEP
                w["reasons"] = w["reasons"] + [
                    f"quality passes, but the deployment floor (${floor:,.0f}/mo) is more than all movable API spend"
                ]
            movable, owned, conservative, point, floor_applied = 0.0, None, None, None, None
        else:
            note = ("API spend of the MOVE types minus owned cost at the high end of its 95% CI; "
                    "the always-on GPU floor is counted once across all MOVE types")
    return {
        "monthly_api_usd": monthly_api,
        "movable_api_usd": movable if monthly_api is not None else None,
        "movable_share": (movable / monthly_api) if monthly_api else None,
        "owned_usd_per_month": owned,
        "floor_applied": floor_applied,
        "conservative_savings_usd": conservative,
        "point_savings_usd": point,
        "savings_note": note,
    }


def _labels(prices: pricing_mod.PriceTable, owned: Mapping[str, Any] | None, window: wl.CaptureWindow, *,
            demo: bool, traffic_scale: int, eval_doc: Mapping[str, Any] | None,
            gpu_hourly_rate: float, gpus: int) -> list[dict[str, str]]:
    def lab(word: str) -> str:
        return f"{DEMO_LABEL} / {word}" if demo else word

    price_labels = sorted({p.label for p in prices if p.origin == pricing_mod.ORIGIN_OVERRIDE}) or [
        f"LIST PRICE ({max(p.as_of for p in prices)})"
    ]
    out = [
        {"what": "API tokens per request", "label": lab("MEASURED"),
         "detail": "usage reported by the API in each captured response; calls without usage are excluded, never estimated"},
        {"what": "API $ per token", "label": lab(", ".join(price_labels)),
         "detail": "published standard-tier prices, dated; batch and priority tiers not covered"},
        {"what": "Requests and API $ per month", "label": lab("MEASURED, projected"),
         "detail": f"captured over {window.days * 24:.1f} hours, scaled to 30 days at the captured rate"},
    ]
    if traffic_scale != 1:
        out.append({"what": "Traffic volume", "label": DEMO_LABEL,
                    "detail": f"each synthetic request stands for {traffic_scale:,} requests of a made-up company"})
    judge = (eval_doc or {}).get("judge") or {}
    out.append({"what": "Quality (win/tie/loss, net CI)", "label": lab("MEASURED"),
                "detail": (f"pairwise judge {judge.get('provider', '')} {judge.get('model', '')}, both answer orders; "
                           "Wilson 95% CIs per rate, paired bootstrap 95% CI on win minus loss")
                if judge else "no judge run"})
    out.append({"what": "Candidate tokens per request", "label": lab("MEASURED"),
                "detail": "usage reported by your model for its shadow answers to the same requests"})
    if owned:
        per_m = owned["usd_per_m_tokens"]
        out.append({"what": "Owned $ per 1M tokens", "label": owned["label"],
                    "detail": f"`throttle check` {owned['check_id']}: ${per_m['mean']:.3f}/M input+output tokens, "
                              f"95% CI ${per_m['low']:.3f} to ${per_m['high']:.3f} over {per_m['n_blocks']} blocks"})
        out.append({"what": "GPU $ per hour in that check", "label": owned["rate_label"],
                    "detail": f"${owned['gpu_hourly_rate_usd']}/h for {owned['gpu_count']} GPU(s), supplied by the operator"})
        per_gpu = float(owned["gpu_hourly_rate_usd"] or 0) / max(1, owned["gpu_count"])
        if abs(per_gpu - gpu_hourly_rate) > 0.005:
            out.append({"what": "GPU price mismatch", "label": "WARNING",
                        "detail": f"the check priced a GPU at ${per_gpu:.2f}/h but this report's floor uses "
                                  f"${gpu_hourly_rate:.2f}/h; owned $/M is not rescaled, so re-run the check at the real price"})
    else:
        out.append({"what": "Owned $ per 1M tokens", "label": "NOT MEASURED",
                    "detail": "no `throttle check` record for the candidate model; no owned-cost or savings number is shown"})
    out.append({"what": "Deployment floor", "label": lab("ASSUMED"),
                "detail": f"${gpu_hourly_rate:g}/GPU-hour x {gpus} GPU(s) x {HOURS_PER_MONTH} h; a dedicated GPU is billed "
                          "every hour, so it is charged once across all MOVE types even when traffic is low"})
    return out


def _limits(owned: Mapping[str, Any] | None, eval_doc: Mapping[str, Any] | None, *, demo: bool) -> list[str]:
    out = []
    if demo:
        out.append("Everything here comes from synthetic traffic, a fake API and a fake open model. It shows how the "
                   "report works; it says nothing about any real company or model.")
    out += [
        "The judge is a model too. Win/tie/loss is its opinion under a fixed rubric, run in both orders to cancel "
        "position bias, not a human review.",
        "Quality is measured on the sampled, de-duplicated requests captured so far. New kinds of requests can behave differently.",
        "Owned cost assumes your traffic's input/output mix costs about what the check's workload cost per token. "
        "Prefill-heavy traffic is usually cheaper per token than decode-heavy traffic.",
        "Monthly figures assume next month looks like the capture window. Seasonality and growth are not modelled.",
        "Engineering time, on-call, fine-tuning runs and egress are not included in owned cost.",
        "NEEDS FINETUNE means the format already works and the gap is in content; it does not prove fine-tuning will close it.",
    ]
    if owned is None:
        out.append("No self-hosted cost was measured, so this report makes no savings claim.")
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def usd(value: float | None) -> str:
    if value is None:
        return "-"
    return f"${value:,.0f}" if abs(value) >= 100 else f"${value:,.2f}"


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.0%}"


def _quality_text(q: Mapping[str, Any] | None) -> str:
    if not q:
        return "no shadow results"
    if not q.get("judged"):
        return f"{q.get('verdict')}"
    net = q.get("net_ci")
    ci = f" net [{net[0]:+.2f}, {net[1]:+.2f}]" if net else ""
    return f"{_pct(q['win_rate'])}/{_pct(q['tie_rate'])}/{_pct(q['loss_rate'])}{ci}"


def summary_lines(report: Mapping[str, Any]) -> list[tuple[str, str]]:
    totals = report["totals"]
    savings = totals["conservative_savings_usd"]
    return [
        ("Monthly API spend", usd(totals["monthly_api_usd"]) if totals["monthly_api_usd"] is not None
         else "not projected (capture under 1 day)"),
        ("Share movable now", (_pct(totals["movable_share"]) + f" ({usd(totals['movable_api_usd'])}/mo)")
         if totals["movable_share"] is not None else "-"),
        ("Conservative monthly savings", usd(savings) if savings is not None else
         ("not measured" if report["owned_cost"] is None else "no savings claim")),
    ]


_CSS = """
:root{--ink:#1d2330;--muted:#5d6676;--line:#dfe3ea;--bg:#fff;--soft:#f6f7f9;--accent:#0e7490;
--move:#166534;--move-bg:#dcfce7;--ft:#92400e;--ft-bg:#fef3c7;--keep:#991b1b;--keep-bg:#fee2e2;--more:#374151;--more-bg:#e5e7eb}
*{box-sizing:border-box}
body{margin:0 auto;max-width:1040px;padding:32px 24px 48px;background:var(--bg);color:var(--ink);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}
h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}
.sub{color:var(--muted);font-size:13px}
.demo{background:#fff7ed;border:2px solid #f97316;color:#9a3412;font-weight:700;padding:10px 14px;border-radius:8px;margin:16px 0;letter-spacing:.02em}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:20px 0}
.card{border:1px solid var(--line);border-radius:10px;padding:14px 16px;background:var(--soft)}
.card span{display:block;color:var(--muted);font-size:13px}.card b{display:block;font-size:24px;margin-top:2px}
.card.hi{border-color:var(--accent)}.card.hi b{color:var(--accent)}
.tag{display:inline-block;font-size:11px;font-weight:700;letter-spacing:.04em;padding:1px 6px;border-radius:4px;background:#ffedd5;color:#9a3412;vertical-align:middle}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th,td{text-align:left;padding:8px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:12px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.03em}
td.n,th.n{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}
.nm{color:var(--muted);font-style:italic;font-size:12.5px}
.rec{display:inline-block;font-weight:700;font-size:12px;padding:2px 8px;border-radius:999px;white-space:nowrap}
.r-move{background:var(--move-bg);color:var(--move)}.r-ft{background:var(--ft-bg);color:var(--ft)}
.r-keep{background:var(--keep-bg);color:var(--keep)}.r-more{background:var(--more-bg);color:var(--more)}
ul.why{margin:0;padding-left:16px}ul.why li{margin:0 0 2px}
.note{color:var(--muted);font-size:13px}
ul.limits li{margin-bottom:6px}
footer{margin-top:40px;color:var(--muted);font-size:12px;border-top:1px solid var(--line);padding-top:10px}
@media (max-width:720px){.cards{grid-template-columns:1fr}table{font-size:12.5px}}
@media print{body{padding:0;max-width:none}.card{break-inside:avoid}tr{break-inside:avoid}}
"""

_REC_CLASS = {MOVE: "r-move", NEEDS_FINETUNE: "r-ft", KEEP: "r-keep", MORE_DATA: "r-more"}


def render_html(report: Mapping[str, Any]) -> str:
    e = lambda v: html.escape(str(v), quote=True)  # noqa: E731
    demo = bool(report.get("demo"))
    tag = ' <span class="tag">DEMO</span>' if demo else ""
    totals = report["totals"]
    parts: list[str] = [
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        f"<title>{'DEMO ' if demo else ''}Throttle report: {e(report['candidate'])}</title>",
        f"<style>{_CSS}</style></head><body>",
        f"<h1>Own your intelligence: {e(report['candidate'])}{tag}</h1>",
        f"<div class=\"sub\">Throttle report {e(report['generated_at'])} &middot; candidate model "
        f"{e(report.get('candidate_model') or '-')} &middot; eval {e(report.get('eval_run_id') or 'none')} &middot; "
        f"check {e(report.get('check_id') or 'none')}</div>",
    ]
    if demo:
        parts.append(f"<div class=\"demo\" role=\"note\">{e(DEMO_BANNER)}. Every number on this page is DEMO.</div>")
    parts.append("<div class=\"cards\">")
    for i, (name, value) in enumerate(summary_lines(report)):
        parts.append(f"<div class=\"card{' hi' if i == 2 else ''}\"><span>{e(name)}{tag}</span><b>{e(value)}</b></div>")
    parts.append("</div>")
    parts.append(f"<p class=\"note\">{e(totals['savings_note'])}.</p>")
    if totals.get("owned_usd_per_month"):
        owned = totals["owned_usd_per_month"]
        floor = report["deployment"]["floor_usd_per_month"]
        if totals.get("floor_applied"):
            text = (f"Owned total for the MOVE types: {usd(owned['high'])}/mo, the always-on deployment floor. "
                    "Their token cost at measured $/M is below it, and the floor applies once across all MOVE types.")
        else:
            text = (f"Owned total for the MOVE types: {usd(owned['point'])}/mo (95% CI {usd(owned['low'])} to "
                    f"{usd(owned['high'])}), above the always-on deployment floor of {usd(floor)}/mo, "
                    "which applies once across all MOVE types.")
        parts.append(f"<p class=\"note\">{e(text)}</p>")

    parts.append(f"<h2>Per workload type{tag}</h2><table><thead><tr>"
                 "<th>Type</th><th class=\"n\">Requests/mo</th><th class=\"n\">API $/mo</th>"
                 "<th class=\"n\">Owned $/mo</th><th>Quality (win/tie/loss, net CI)</th>"
                 "<th>Recommendation</th><th>Why</th></tr></thead><tbody>")
    for w in report["workloads"]:
        owned = w["owned_usd_per_month"]
        owned_cell = (f"{e(usd(owned['point']))}<div class=\"nm\">{e(usd(owned['low']))}&ndash;{e(usd(owned['high']))}</div>"
                      if owned else f"<span class=\"nm\">{e(w['owned_note'])}</span>")
        api_cell = e(usd(w["api_usd_per_month"])) if w["api_usd_per_month"] is not None else f"<span class=\"nm\">{e(w['api_note'])}</span>"
        req = f"{w['requests_per_month']:,.0f}" if w["requests_per_month"] is not None else "-"
        q = w["quality"]
        quality = e(_quality_text(q)) + (f"<div class=\"nm\">{e(q['verdict'])}</div>" if q else "")
        why = "".join(f"<li>{e(r)}</li>" for r in w["reasons"])
        parts.append(
            f"<tr><td><b>{e(w['type'])}</b></td><td class=\"n\">{e(req)}</td><td class=\"n\">{api_cell}</td>"
            f"<td class=\"n\">{owned_cell}</td><td>{quality}</td>"
            f"<td><span class=\"rec {_REC_CLASS[w['recommendation']]}\">{e(w['recommendation'])}</span></td>"
            f"<td><ul class=\"why\">{why}</ul></td></tr>"
        )
    parts.append("</tbody></table>")
    parts.append("<p class=\"note\">Owned $/mo shows the point estimate and the 95% CI from the measured $/M; "
                 "a type's owned cost here is its token cost only, before the deployment floor.</p>")

    parts.append("<h2>How we got these numbers</h2><table><thead><tr><th>Number</th><th>Label</th><th>Source</th></tr></thead><tbody>")
    for item in report["labels"]:
        parts.append(f"<tr><td>{e(item['what'])}</td><td><b>{e(item['label'])}</b></td><td>{e(item['detail'])}</td></tr>")
    parts.append("</tbody></table>")
    parts.append("<p class=\"note\">Recommendation rules: MOVE needs quality BETTER or NO MATERIAL DIFFERENCE, a measured "
                 "owned cost, and a positive saving at the high end of the cost CI. NEEDS FINETUNE: quality WORSE but "
                 "format checks pass about as often as the API model. KEEP ON API: quality WORSE with broken format, "
                 "or no saving. NEEDS MORE DATA: too few judged requests, an undecided CI, or no owned-cost measurement.</p>")
    parts.append("<h2>What this does not prove</h2><ul class=\"limits\">")
    parts += [f"<li>{e(item)}</li>" for item in report["does_not_prove"]]
    parts.append("</ul>")
    parts.append(f"<footer>Generated by <code>throttle {'demo' if demo else 'report'}</code>. "
                 "Throttle helps companies own their intelligence. Works offline; no external assets.</footer>")
    parts.append("</body></html>")
    return "\n".join(parts) + "\n"


def _owned_gap(workload: Mapping[str, Any]) -> str:
    note = workload.get("owned_note") or ""
    return "not projected" if note.startswith("not projected") else "not measured"


def render_panel(report: Mapping[str, Any], paths: Sequence[Path] = ()) -> str:
    from .. import style as style_mod

    st = style_mod.get()
    rows: list[Any] = []
    if report.get("demo"):
        rows.append((DEMO_BANNER, (style_mod.BOLD, style_mod.AMBER)))
    for name, value in summary_lines(report):
        rows.append(f"{name + ':':<30}{st.bold(value)}")
    rows.append("")
    rows.append(st.dim(f"{'type':<15}{'API $/mo':>11}{'owned $/mo':>12}  recommendation"))
    for w in report["workloads"]:
        owned = w["owned_usd_per_month"]
        rows.append(f"{w['type'][:14]:<15}{usd(w['api_usd_per_month']):>11}"
                    f"{(usd(owned['point']) if owned else _owned_gap(w)):>12}  {st.verdict(w['recommendation'])}")
    owned_total = report["totals"].get("owned_usd_per_month")
    if owned_total:
        rows.append("")
        rows.append((f"Owned $/mo per type is token cost only. The MOVE types together cost "
                     f"{usd(owned_total['high'])}/mo at the high end, never less than the always-on "
                     f"{usd(report['deployment']['floor_usd_per_month'])}/mo GPU floor, counted once.", (style_mod.DIM,)))
    if report["owned_cost"] is None:
        rows.append("")
        rows.append((report["not_measured_hint"], (style_mod.AMBER,)))
    if paths:
        rows.append("")
        rows += [(f"Saved {p}", (style_mod.DIM,)) for p in paths]
    title = f"Report: {report['candidate']} vs API" + (" (DEMO)" if report.get("demo") else "")
    return st.panel(rows, title=title)


def write_outputs(report: Mapping[str, Any], out: Path) -> tuple[Path, Path]:
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "report.json"
    html_path = out / "report.html"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    html_path.write_text(render_html(report), encoding="utf-8")
    return json_path, html_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "report",
        help="which workload types can move to your own model, and what that saves per month",
        description=(
            "Join captured API spend, the latest eval of your candidate model and a measured "
            "`throttle check` of it into a recommendation per workload type (MOVE, NEEDS FINETUNE, "
            "KEEP ON API, NEEDS MORE DATA), written as report.json and a self-contained report.html. "
            "Without a check record no owned-cost or savings number is shown."
        ),
    )
    parser.add_argument("--candidate", required=True, help="candidate label used with `throttle shadow` / `throttle eval`")
    parser.add_argument("--gpu-hourly-rate", type=float, required=True, metavar="USD",
                        help="price of one GPU per hour for the deployment floor (ASSUMED)")
    parser.add_argument("--gpus", type=int, required=True, metavar="N", help="GPUs the always-on deployment needs")
    parser.add_argument("--check", default="latest", metavar="ID|latest",
                        help="`throttle check` record to price the candidate with (default: latest for its model)")
    parser.add_argument("--eval", dest="eval_run", default="latest", metavar="RUN_ID|latest",
                        help="eval run to use (default: latest for the candidate)")
    parser.add_argument("--prices", type=Path, default=None, metavar="FILE",
                        help="JSON price file that replaces or adds API prices; its prices are labelled ASSUMED")
    parser.add_argument("--out", type=Path, default=Path("throttle-report"), metavar="DIR",
                        help="where report.json and report.html go (default ./throttle-report/)")
    parser.add_argument("--json", action="store_true", help="also print the report JSON")
    parser.add_argument("--dir", type=Path, default=None, help="data directory (default ~/.throttle/own or $THROTTLE_OWN_DIR)")


def handle(args: argparse.Namespace) -> int:
    from .. import check as check_mod

    root = args.dir.expanduser() if args.dir else records.default_root()
    if args.gpu_hourly_rate <= 0 or args.gpus <= 0:
        print("throttle report: --gpu-hourly-rate and --gpus must be positive", file=sys.stderr)
        return EXIT_USAGE
    try:
        prices = pricing_mod.load_prices(args.prices)
        eval_doc = load_eval(root, args.candidate, args.eval_run)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"throttle report: {exc}", file=sys.stderr)
        return EXIT_USAGE
    exchanges = list(records.iter_exchanges(root))
    if not exchanges:
        print(f"throttle report: no captured calls in {records.capture_dir(root)}. "
              "Run 'throttle capture' in front of your API traffic first.", file=sys.stderr)
        return EXIT_FAILED
    if eval_doc is None:
        print(f"Note: no eval run for {args.candidate}; every type is NEEDS MORE DATA. "
              "Run `throttle shadow` and `throttle eval` first.", file=sys.stderr)
    models = [m for m in ((eval_doc or {}).get("candidate_model"), args.candidate) if m]
    try:
        record = load_check(check_mod.history_dir(None), models, args.check)
    except ValueError as exc:
        print(f"throttle report: {exc}", file=sys.stderr)
        return EXIT_USAGE
    report = build_report(
        exchanges, prices, candidate=args.candidate, eval_doc=eval_doc, check_record=record,
        gpu_hourly_rate=args.gpu_hourly_rate, gpus=args.gpus,
    )
    paths = write_outputs(report, args.out.expanduser())
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_panel(report, paths))
    return EXIT_OK
