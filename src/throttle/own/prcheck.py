"""``throttle pr-check``: what a pull request does to your AI bill.

It scans the code at two git refs (see :mod:`throttle.own.scan`), finds the
AI jobs that were added, changed, moved or removed, and for each one looks
up what Throttle already knows:

* **Cost on the API**: captured traffic that matches the job (by the
  ``X-Throttle-Job`` header or its prompt template), priced at list price.
  A monthly figure needs at least a day of captured traffic; before that the
  cost is shown per 1,000 calls.
* **Proof**: the newest job-level eval (``throttle jobs eval``) of the job's
  current prompt. A brand-new job has no traffic, so when a candidate model
  and a judge are configured it gets a FIRST LOOK (:mod:`throttle.own.firstlook`):
  a few made-up inputs, labelled as such, never a verdict.
* **What happens**: from the job's state in ``<root>/jobs/registry.json``
  and the proof. A moved job whose prompt changed goes back to testing when
  the change merges; a removed job's route is retired. pr-check itself
  changes nothing: it only reads. ``throttle jobs sync`` applies the code
  once it lands.

Output: a Markdown comment (``--markdown``, stdout by default) and a JSON
document (``--json``). ``throttle pr-comment`` posts the Markdown as one
sticky comment on the pull request.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import httpx

from . import evaluate as ev
from . import firstlook as fl
from . import jobs as jobs_mod
from . import judge as judge_mod
from . import pricing as pricing_mod
from . import records
from . import scan as scan_mod
from . import workloads as wl

SCHEMA = "throttle.own.pr-check/1"
MARKER = "<!-- throttle:pr-check -->"
EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 2

ADDED, CHANGED, MOVED_IN_CODE, REMOVED = "added", "changed", "moved", "removed"

# What happens next, as codes in the JSON and plain words in the comment.
ACTIONS = {
    "move": "Ready to move",
    "measure": "Passed, measure cost",
    "keep": "Keep on the API",
    "finetune": "Keep on the API for now",
    "test_more": "Test it more",
    "test_again": "Back to testing",
    "stays": "Stays on your model",
    "pinned": "Pinned to the API",
    "retire": "Route retired",
    "nothing": "Nothing to do",
}


@dataclass
class JobChange:
    kind: str                              # added | changed | moved | removed
    before: scan_mod.CallSite | None
    after: scan_mod.CallSite | None
    what: list[str] = field(default_factory=list)

    @property
    def site(self) -> scan_mod.CallSite:
        return self.after or self.before  # type: ignore[return-value]

    @property
    def job(self) -> str:
        return self.site.job


def diff_sites(base: Sequence[scan_mod.CallSite], head: Sequence[scan_mod.CallSite]) -> tuple[list[JobChange], int]:
    """(changes, number of jobs left as they were)."""
    before = {s.job: s for s in base}
    after = {s.job: s for s in head}
    added = [after[k] for k in sorted(set(after) - set(before))]
    removed = [before[k] for k in sorted(set(before) - set(after))]
    out: list[JobChange] = []
    moves = jobs_mod.pair_moves(removed, added)
    moved_old = {o.job for o, _ in moves}
    moved_new = {n.job for _, n in moves}
    for old, new in moves:
        out.append(JobChange(MOVED_IN_CODE, old, new, [f"moved from {old.job}"]))
    unchanged = 0
    for job in sorted(set(before) & set(after)):
        what = jobs_mod.changes(before[job], after[job])
        if what:
            out.append(JobChange(CHANGED, before[job], after[job], what))
        else:
            unchanged += 1
    out += [JobChange(ADDED, None, s, ["new AI call"]) for s in added if s.job not in moved_new]
    out += [JobChange(REMOVED, s, None, ["AI call removed"]) for s in removed if s.job not in moved_old]
    order = {ADDED: 0, CHANGED: 1, MOVED_IN_CODE: 2, REMOVED: 3}
    return sorted(out, key=lambda c: (order[c.kind], c.job)), unchanged


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


def traffic_by_job(exchanges: Sequence[records.Exchange], sites: Sequence[scan_mod.CallSite]) -> dict[str, list[records.Exchange]]:
    matcher = jobs_mod.JobMatcher(sites)
    out: dict[str, list[records.Exchange]] = {}
    for exchange in exchanges:
        norm = records.normalize(exchange) if exchange.has_bodies else None
        job = matcher.match(exchange, norm)
        if job is not None:
            out.setdefault(job, []).append(exchange)
    return out


def cost_of(exchanges: Sequence[records.Exchange], window: wl.CaptureWindow, prices: pricing_mod.PriceTable) -> dict[str, Any]:
    """API spend of one job's captured calls: total, per 1,000 calls, and per
    month when the capture window is at least a day."""
    spend = 0.0
    priced = 0
    labels: set[str] = set()
    excluded = 0
    for exchange in exchanges:
        if not exchange.ok or exchange.served_by == records.SERVED_OWNED:
            excluded += 1
            continue
        price = prices.lookup(exchange.provider, exchange.model)
        cost = pricing_mod.cost_usd(price, exchange.usage)
        if cost is None:
            excluded += 1
            continue
        spend += cost
        priced += 1
        labels.add(price.label)  # type: ignore[union-attr]
    per_call = spend / priced if priced else None
    monthly = None
    if priced and window.projectable:
        monthly = len(exchanges) / window.days * 30 * per_call  # every call of the job, at the priced mean
    return {
        "requests": len(exchanges), "priced": priced, "excluded": excluded,
        "spend_usd": round(spend, 8) if priced else None,
        "per_call_usd": per_call, "per_1k_usd": per_call * 1000 if per_call is not None else None,
        "monthly_usd": monthly, "window_days": round(window.days, 4),
        "labels": sorted(labels), "basis": "MEASURED tokens x " + (", ".join(sorted(labels)) or "price"),
    }


def quality_of(result: Mapping[str, Any] | None, run: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not result:
        return None
    judged = int(result.get("judged") or 0)
    net = result.get("net_ci") or {}
    return {
        "source": "job eval", "run_id": (run or {}).get("run_id"), "candidate": (run or {}).get("candidate"),
        "candidate_model": (run or {}).get("candidate_model"), "judge": (run or {}).get("judge"),
        "verdict": result.get("verdict"), "n": result.get("n"), "judged": judged,
        "wins": result.get("wins"), "ties": int(result.get("ties") or 0) + int(result.get("order_disagree") or 0),
        "losses": result.get("losses"), "min_n": (run or {}).get("min_n"),
        "net_ci": [net.get("low"), net.get("high")] if net.get("low") is not None else None,
        "format_broken": [name for name, c in _format_checks(result).items() if not c],
        "reasons": list(result.get("reasons") or []),
        "candidate_usage": result.get("candidate_usage"), "candidate_errors": result.get("candidate_errors"),
        "prompt_fingerprint": result.get("prompt_fingerprint"),
    }


def _format_checks(result: Mapping[str, Any]) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for name in ("json_valid", "schema_valid", "tool_calls_valid"):
        sides = (result.get("checks") or {}).get(name)
        if not sides:
            continue
        inc, cand = sides.get("incumbent") or {}, sides.get("candidate") or {}
        out[name] = not (inc.get("low") is not None and cand.get("high") is not None and float(cand["high"]) < float(inc["low"]))
    return out


def owned_per_call(tokens: Mapping[str, Any] | None, owned: Mapping[str, Any] | None) -> float | None:
    """Your model's token cost for one call at the high end of the measured
    $/M (no GPU floor; the full report applies it)."""
    if not tokens or not owned:
        return None
    calls = int(tokens.get("calls") or tokens.get("with_usage") or 0)
    total = float(tokens.get("input_tokens") or 0) + float(tokens.get("output_tokens") or 0)
    if calls <= 0 or total <= 0:
        return None
    return total / calls * float(owned["usd_per_m_tokens"]["high"]) / 1_000_000


def decide(change: JobChange, entry: jobs_mod.JobEntry | None, quality: Mapping[str, Any] | None,
           first_look: Mapping[str, Any] | None, owned_call: float | None, api_call: float | None,
           old_quality: Mapping[str, Any] | None = None) -> tuple[str, str]:
    """(action code, what happens, in plain words)."""
    status = entry.status if entry is not None else None
    if change.kind == REMOVED:
        if status == jobs_mod.MOVED:
            return "retire", "This call is gone. When this merges, its route to your model is removed."
        return "retire", "This call is gone. When this merges, Throttle stops tracking it."
    if status == jobs_mod.PINNED:
        return "pinned", "Your team pinned this job to the API. Throttle will not move it."
    if change.kind == MOVED_IN_CODE:
        if status == jobs_mod.MOVED:
            return "stays", "Same prompt in a new place. It keeps running on your model."
        return "nothing", "Same prompt in a new place. Nothing changes where it runs."
    if change.kind == CHANGED and status == jobs_mod.MOVED:
        return "test_again", ("This job runs on your model today. When this merges it goes back to the API until "
                              "your model passes on the new version. It is never moved silently.")
    if quality is not None:
        verdict = quality.get("verdict")
        if verdict in jobs_mod.PASSING:
            if status == jobs_mod.MOVED:
                return "stays", "It keeps running on your model, with fallback to the API."
            if owned_call is None:
                return "measure", ("Your model passed on this job. Measure what it costs to run with `throttle check`, "
                                   "then approve the move.")
            if api_call is not None and owned_call >= api_call:
                return "keep", "Your model passed, but it costs more per call than the API here."
            return "move", f"Your model passed. Approve the move with `throttle jobs approve {change.job} ...`."
        if verdict == ev.WORSE:
            if quality.get("format_broken"):
                return "keep", f"Your model breaks the output format here ({', '.join(quality['format_broken'])})."
            return "finetune", ("Your model loses on quality here. Its misses are saved as lessons for a "
                                "future fine-tune.")
        need = quality.get("min_n") or 30
        return "test_more", (f"Stays on the API while Throttle collects more calls "
                             f"({quality.get('judged') or 0} judged so far, needs {need}).")
    if change.kind == ADDED:
        return "test_more", "Stays on the API. Throttle starts testing it on real calls once this ships."
    if old_quality is not None and old_quality.get("verdict") in jobs_mod.PASSING:
        return "test_again", "Your model passed on the old prompt. The new one stays on the API until it passes too."
    return "test_more", "Stays on the API. The new version has not been tested on real calls yet."


# ---------------------------------------------------------------------------
# Building the check
# ---------------------------------------------------------------------------


@dataclass
class Options:
    candidate: str | None = None
    first_look: fl.FirstLookConfig | None = None
    owned: Mapping[str, Any] | None = None       # report.owned_cost() block for the candidate
    prices: pricing_mod.PriceTable | None = None
    transport: httpx.AsyncBaseTransport | None = None


def run_check(root: Path, base_sites: Sequence[scan_mod.CallSite], head_sites: Sequence[scan_mod.CallSite], *,
              base_ref: str, head_ref: str, options: Options | None = None) -> dict[str, Any]:
    options = options or Options()
    prices = options.prices or pricing_mod.load_prices()
    changes_, unchanged = diff_sites(base_sites, head_sites)
    exchanges = list(records.iter_exchanges(root))
    window = wl.capture_window(exchanges)
    head_traffic = traffic_by_job(exchanges, head_sites)
    base_traffic = traffic_by_job(exchanges, base_sites)
    try:
        registry = jobs_mod.load_registry(root)
    except (OSError, ValueError):
        registry = {}
    rows = []
    for change in changes_:
        rows.append(_row(root, change, registry, head_traffic, base_traffic, window, prices, options))
    totals = _totals(rows)
    return {
        "schema": SCHEMA, "generated_at": records.utc_now_iso(), "base": base_ref, "head": head_ref,
        "candidate": options.candidate,
        "jobs": rows, "unchanged_jobs": unchanged, "jobs_in_head": len(head_sites),
        "capture": {"calls": len(exchanges), "first_ts": window.first_ts, "last_ts": window.last_ts,
                    "days": round(window.days, 4), "monthly_projection": window.projectable},
        "owned_cost": dict(options.owned) if options.owned else None,
        "totals": totals,
    }


def _row(root: Path, change: JobChange, registry: Mapping[str, jobs_mod.JobEntry],
         head_traffic: Mapping[str, list[records.Exchange]], base_traffic: Mapping[str, list[records.Exchange]],
         window: wl.CaptureWindow, prices: pricing_mod.PriceTable, options: Options) -> dict[str, Any]:
    site = change.site
    entry = registry.get(change.job) or (registry.get(change.before.job) if change.before else None)
    now = head_traffic.get(change.job, []) if change.after else []
    old = base_traffic.get(change.before.job, []) if change.before else []
    traffic_basis = "current"
    cost = cost_of(now, window, prices) if now else None
    if (cost is None or not cost["priced"]) and old:
        cost = cost_of(old, window, prices)
        traffic_basis = "old version" if change.kind == CHANGED else "current"
    quality = None
    old_quality = None
    if change.after is not None:
        found = jobs_mod.latest_job_result(root, change.job, options.candidate, change.after.prompt_fingerprint)
        if found:
            quality = quality_of(found[1], found[0])
    if change.before is not None and change.kind == CHANGED:
        found = jobs_mod.latest_job_result(root, change.before.job, options.candidate, change.before.prompt_fingerprint)
        if found:
            old_quality = quality_of(found[1], found[0])
    first_look = None
    if (change.after is not None and quality is None and not now and options.first_look is not None
            and change.kind in (ADDED, CHANGED)):
        first_look = fl.cached_or_run(change.after, options.first_look, root=root, prices=prices,
                                      transport=options.transport)
    api_call = cost["per_call_usd"] if cost and cost.get("per_call_usd") is not None else None
    if api_call is None and first_look and first_look.get("api_cost_per_call_usd") is not None:
        api_call = first_look["api_cost_per_call_usd"]
    tokens = None
    if quality and quality.get("candidate_usage"):
        usage = quality["candidate_usage"]
        counted = int(quality.get("n") or 0) - int(quality.get("candidate_errors") or 0) - int(usage.get("excluded") or 0)
        tokens = {"calls": counted, "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens")}
    elif first_look and first_look.get("candidate_usage"):
        tokens = first_look["candidate_usage"]
        tokens = {**tokens, "calls": tokens.get("with_usage")}
    owned_call = owned_per_call(tokens, options.owned)
    action, words = decide(change, entry, quality, first_look, owned_call, api_call, old_quality)
    monthly = cost.get("monthly_usd") if cost else None
    saving_monthly = None
    if monthly is not None and owned_call is not None and api_call:
        saving_monthly = max(0.0, monthly * (1 - owned_call / api_call))
    return {
        "job": change.job, "change": change.kind, "what": change.what,
        "location": site.location, "function": site.function, "provider": site.provider, "api": site.api,
        "model": site.model or site.model_expr, "language": site.language,
        "prompt_fingerprint": site.prompt_fingerprint,
        "before": change.before.to_json() if change.before else None,
        "after": change.after.to_json() if change.after else None,
        "status": entry.status if entry else None,
        "cost": cost, "traffic_basis": traffic_basis if cost else None,
        "api_per_call_usd": api_call, "owned_per_call_usd": owned_call,
        "monthly_saving_usd": saving_monthly,
        "quality": quality, "old_quality": old_quality, "first_look": first_look,
        "action": action, "action_title": ACTIONS[action], "action_text": words,
    }


def _totals(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    monthly = [r["cost"]["monthly_usd"] for r in rows
               if r["cost"] and r["cost"].get("monthly_usd") is not None and r["change"] != REMOVED
               and r.get("traffic_basis") == "current"]
    savings = [r["monthly_saving_usd"] for r in rows if r.get("monthly_saving_usd") and r["action"] in ("move", "stays")]
    return {
        "monthly_api_usd": sum(monthly) if monthly else None,
        "monthly_saving_usd": sum(savings) if savings else None,
        "jobs_priced_monthly": len(monthly),
        "changed": len(rows),
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def money(value: float | None) -> str:
    if value is None:
        return "-"
    if value >= 100:
        return f"${value:,.0f}"
    if value >= 1:
        return f"${value:,.2f}"
    if value >= 0.01:
        return f"${value:.2f}"
    if value == 0:
        return "$0"
    return f"${value:.4f}".rstrip("0") if value >= 0.0001 else "under $0.0001"


def _code(text: str) -> str:
    return "`" + text.replace("`", "'") + "`"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _api_cell(row: Mapping[str, Any]) -> str:
    cost = row.get("cost")
    if row["change"] == REMOVED:
        return "-"
    if cost and cost.get("monthly_usd") is not None:
        suffix = " (old version)" if row.get("traffic_basis") == "old version" else ""
        return f"{money(cost['monthly_usd'])} a month{suffix}"
    if cost and cost.get("per_1k_usd") is not None:
        suffix = ", old version" if row.get("traffic_basis") == "old version" else ""
        return f"{money(cost['per_1k_usd'])} per 1k calls ({cost['priced']} seen{suffix})"
    look = row.get("first_look") or {}
    if look.get("api_cost_per_call_usd") is not None:
        return f"{money(look['api_cost_per_call_usd'] * 1000)} per 1k calls (first look)"
    return "not known yet"


def _model_cell(row: Mapping[str, Any]) -> str:
    if row["change"] == REMOVED:
        return "-"
    q = row.get("quality")
    if q:
        return f"{q['verdict'].lower()} ({q['judged']} judged)"
    look = row.get("first_look")
    if look and not look.get("skipped"):
        good = int(look.get("wins") or 0) + int(look.get("ties") or 0)
        return f"first look: as good on {good} of {look['inputs']}"
    oq = row.get("old_quality")
    if oq:
        return f"old version: {oq['verdict'].lower()}; new one not tested"
    return "not tested yet"


def _change_cell(row: Mapping[str, Any]) -> str:
    if row["change"] == ADDED:
        return "new"
    if row["change"] == REMOVED:
        return "removed"
    return ", ".join(row["what"])


def headline(doc: Mapping[str, Any]) -> list[str]:
    totals = doc["totals"]
    capture = doc["capture"]
    lines: list[str] = []
    if totals.get("monthly_api_usd") is not None:
        line = (f"**About {money(totals['monthly_api_usd'])} a month on the API** for the AI jobs in this pull request "
                f"(captured calls over {capture['days']:.1f} days, projected to 30).")
        if totals.get("monthly_saving_usd"):
            line += f" Your own model could save about **{money(totals['monthly_saving_usd'])} a month** of it."
        lines.append(line)
    elif capture["calls"] and not capture["monthly_projection"]:
        lines.append("**No monthly figure yet.** Throttle needs at least a day of captured traffic to project a month. "
                     "Cost per 1,000 calls is below.")
    elif not capture["calls"]:
        lines.append("**No captured traffic yet,** so no monthly cost. Point your app at `throttle capture` to see it.")
    else:
        lines.append("**No cost yet** for these jobs: no captured calls match them.")
    return lines


def render_markdown(doc: Mapping[str, Any]) -> str:
    rows = doc["jobs"]
    n = len(rows)
    out = [MARKER]
    if not rows:
        out += ["### Throttle: no AI calls changed", "",
                f"This pull request does not add, edit or remove an AI call. Throttle tracks {doc['jobs_in_head']} in this code."]
        return "\n".join(out) + "\n"
    out += [f"### Throttle: {n} AI call{'s' if n != 1 else ''} changed in this pull request", ""]
    out += headline(doc)
    out += ["", "| AI job | Change | On the API | Your model | What happens |", "|---|---|---|---|---|"]
    for row in rows:
        out.append("| " + " | ".join([
            _code(row["job"]), _cell(_change_cell(row)), _cell(_api_cell(row)), _cell(_model_cell(row)),
            f"**{row['action_title']}**",
        ]) + " |")
    if doc.get("unchanged_jobs"):
        n_same = doc["unchanged_jobs"]
        out.append(f"{n_same} other AI job{'s' if n_same != 1 else ''} in this code {'are' if n_same != 1 else 'is'} unchanged.")
    out.append("")
    for row in rows:
        out += _details(row)
    out += _footer(doc)
    return "\n".join(out) + "\n"


def _details(row: Mapping[str, Any]) -> list[str]:
    title = {ADDED: "New", CHANGED: "Changed", MOVED_IN_CODE: "Moved", REMOVED: "Removed"}[row["change"]]
    model = row.get("model") or "model set at run time"
    lines = [f"#### {title}: {_code(row['job'])}", "",
             f"{_code(row['location'])}, {row['provider']} {row['api']}, {_code(str(model))}. "
             + ("" if row["change"] in (ADDED, REMOVED) else _cell(", ".join(row["what"]).capitalize() + ". "))
             + row["action_text"], ""]
    if row["change"] == REMOVED:
        return lines
    cost = row.get("cost")
    if cost and cost.get("priced"):
        basis = " on the old version of the prompt" if row.get("traffic_basis") == "old version" else ""
        text = f"- **On the API:** {cost['priced']} captured calls{basis} cost {money(cost['spend_usd'])}, "
        text += f"about {money(cost['per_1k_usd'])} per 1,000 calls"
        if cost.get("monthly_usd") is not None:
            text += f", about {money(cost['monthly_usd'])} a month at the captured rate"
        text += f". {cost['basis']}."
        lines.append(text)
    elif row.get("first_look") and row["first_look"].get("api_cost_per_call_usd") is not None:
        look = row["first_look"]
        lines.append(f"- **On the API:** about {money(look['api_cost_per_call_usd'] * 1000)} per 1,000 calls, "
                     f"from {look['api_usage']['with_usage']} first-look calls to {_code(look['api_model'])}. "
                     f"MEASURED tokens x {look.get('api_cost_label')}. No traffic yet, so no monthly figure.")
    elif row.get("first_look") and row["first_look"].get("api_cost_note"):
        lines.append(f"- **On the API:** not priced: {row['first_look']['api_cost_note']}.")
    else:
        lines.append("- **On the API:** no captured calls for this job yet.")
    if row.get("owned_per_call_usd") is not None:
        lines.append(f"- **On your model:** about {money(row['owned_per_call_usd'] * 1000)} per 1,000 calls in tokens "
                     "(measured $/M, high end of its 95% CI; the GPU is paid for every hour, so `throttle report` "
                     "makes the final call).")
    q = row.get("quality")
    look = row.get("first_look")
    if q:
        lines.append(f"- **Proof:** job eval {_code(str(q['run_id']))} of {q.get('candidate') or 'your model'}: "
                     f"better on {q['wins']}, same on {q['ties']}, worse on {q['losses']} of {q['judged']} real calls "
                     f"(judge asked twice, answers swapped). Verdict: {q['verdict']}.")
        if q.get("reasons"):
            lines.append("  <details><summary>Why</summary>" + "<br>".join(_cell(r) for r in q["reasons"])
                         + (f"<br>win minus loss 95% CI [{q['net_ci'][0]:+.2f}, {q['net_ci'][1]:+.2f}]" if q.get("net_ci") else "")
                         + "</details>")
    elif look and look.get("skipped"):
        lines.append(f"- **Proof:** none yet. No first look: {look['skipped']}.")
    elif look:
        lines += _first_look_lines(look)
    elif row.get("old_quality"):
        oq = row["old_quality"]
        lines.append(f"- **Proof:** the old version was {oq['verdict']} on {oq['judged']} judged calls. "
                     "The new version has not been tested.")
    else:
        lines.append("- **Proof:** none yet. Throttle tests it on captured calls (`throttle shadow` + `throttle jobs eval`).")
    lines.append("")
    return lines


def _first_look_lines(look: Mapping[str, Any]) -> list[str]:
    good = int(look.get("wins") or 0) + int(look.get("ties") or 0)
    lines = [
        f"- **Proof (first look, not enough data yet):** {look['inputs']} made-up inputs, not your traffic. "
        f"{look['candidate']} ({_code(look['candidate_model'])}) was as good or better on {good}, worse on "
        f"{look.get('losses', 0)}" + (f", and the judge flipped on {look['order_disagree']}" if look.get("order_disagree") else "")
        + f" (judge {_code(look['judge']['model'])}, asked twice with the answers swapped). This is not a verdict and moves nothing.",
    ]
    problems = []
    if look.get("api_errors"):
        problems.append(f"the API failed on {look['api_errors']}")
    if look.get("candidate_errors"):
        problems.append(f"your model failed on {look['candidate_errors']}")
    if look.get("judge_errors"):
        problems.append(f"the judge failed on {look['judge_errors']}")
    for name, row in (look.get("checks") or {}).items():
        if name == "json_valid":
            problems.append(f"valid JSON: API {row['api']} of {row['n']}, your model {row['candidate']} of {row['n']}")
        elif row["candidate"] < row["n"]:
            problems.append(f"your model was cut off on {row['n'] - row['candidate']}")
    if problems:
        lines.append("  " + "; ".join(problems).capitalize() + ".")
    examples = look.get("examples") or []
    if examples:
        body = ["<details><summary>Sample inputs and answers</summary>", ""]
        for ex in examples:
            body.append(f"**Input:** {_cell(ex.get('input') or '')}<br>")
            body.append(f"**API:** {_cell(ex.get('api_answer') or '(no answer)')}<br>")
            body.append(f"**Your model:** {_cell(ex.get('candidate_answer') or '(no answer)')}<br>")
            body.append(f"**Judge:** {_outcome_words(ex.get('outcome'))}")
            body.append("")
        body.append("</details>")
        lines += ["", *body]
    return lines


def _outcome_words(outcome: str | None) -> str:
    return {
        judge_mod.WIN: "your model's answer was better (both orders)",
        judge_mod.TIE: "same quality (both orders)",
        judge_mod.LOSS: "the API's answer was better (both orders)",
        judge_mod.ORDER_DISAGREE: "no clear winner (the judge changed its mind when the answers were swapped)",
        judge_mod.ERROR: "the judge failed",
    }.get(outcome or "", "not judged")


def _footer(doc: Mapping[str, Any]) -> list[str]:
    return [
        "<sub>MEASURED: token counts Throttle saw. LIST PRICE: the published price on the date shown. ASSUMED: a price "
        "your team gave Throttle. FIRST LOOK: made-up inputs, not your traffic. Ran in your CI: prompts went only to the "
        f"API, model and judge your team set up. Nothing moves without a passing test and your approval. "
        f"Base {_code(doc['base'][:12])}, head {_code(doc['head'][:12])}.</sub>",
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "pr-check",
        help="what a pull request's AI changes cost, what proof exists, and what Throttle will do",
        description=(
            "Compare the AI calls at two git refs. For each added, changed or removed call: its cost on the API, "
            "whether your own model passed the tests for it, and what happens next. Reads only; changes nothing."
        ),
    )
    parser.add_argument("--repo", default=".", help="git repository (default: current folder)")
    parser.add_argument("--base", required=True, help="the ref the pull request merges into, e.g. origin/main")
    parser.add_argument("--head", default="HEAD", help="the pull request's ref (default HEAD)")
    parser.add_argument("--dir", type=Path, default=None, help="Throttle data folder (default ~/.throttle/own or $THROTTLE_OWN_DIR)")
    parser.add_argument("--candidate", default=None, help="your model's label from `throttle shadow` (default: any)")
    parser.add_argument("--candidate-url", default=None, help="your model's OpenAI-compatible URL, for first looks")
    parser.add_argument("--candidate-model", default=None, help="your model's name on that server")
    parser.add_argument("--candidate-key-env", default=None, metavar="VAR")
    parser.add_argument("--judge-url", default=None)
    parser.add_argument("--judge-model", default=None, help="judge for first looks; also writes their test inputs")
    parser.add_argument("--judge-provider", choices=judge_mod.PROVIDERS, default=records.PROVIDER_OPENAI)
    parser.add_argument("--judge-key-env", default=None, metavar="VAR")
    parser.add_argument("--api-url", default=None,
                        help="OpenAI-compatible API for first looks of OpenAI jobs (default https://api.openai.com/v1)")
    parser.add_argument("--api-key-env", default=None, metavar="VAR", help="environment variable with that API's key")
    parser.add_argument("--anthropic-url", default=None, help="API for first looks of Anthropic jobs (default Anthropic's)")
    parser.add_argument("--anthropic-key-env", default=None, metavar="VAR")
    parser.add_argument("--api-model", default=None, help="model for first looks when the code does not name one")
    parser.add_argument("--first-look-inputs", type=int, default=fl.DEFAULT_INPUTS, metavar="N",
                        help=f"made-up inputs per new job (default {fl.DEFAULT_INPUTS}, max {fl.MAX_INPUTS})")
    parser.add_argument("--no-first-look", action="store_true", help="never call any model")
    parser.add_argument("--prices", type=Path, default=None, metavar="FILE",
                        help="JSON price file that adds or replaces API prices (labelled ASSUMED)")
    parser.add_argument("--markdown", type=Path, default=None, metavar="FILE", help="write the comment here (default: print it)")
    parser.add_argument("--json", dest="json_path", type=Path, default=None, metavar="FILE", help="write the JSON here")


def first_look_config(args: argparse.Namespace) -> fl.FirstLookConfig | None:
    from .shadow import api_key_from_env, chat_url

    if args.no_first_look or not (args.candidate_url and args.candidate_model and args.judge_model):
        return None
    judge_url = args.judge_url or judge_mod.DEFAULT_URLS[args.judge_provider]
    if args.judge_provider == records.PROVIDER_OPENAI and args.judge_url:
        judge_url = chat_url(args.judge_url)
    judge = judge_mod.JudgeConfig(provider=args.judge_provider, model=args.judge_model, url=judge_url,
                                  api_key=api_key_from_env(args.judge_key_env))
    return fl.FirstLookConfig(
        candidate_url=args.candidate_url, candidate_model=args.candidate_model,
        candidate_label=args.candidate or args.candidate_model, judge=judge,
        api_url=args.api_url, api_key=api_key_from_env(args.api_key_env),
        anthropic_url=args.anthropic_url, anthropic_key=api_key_from_env(args.anthropic_key_env),
        candidate_key=api_key_from_env(args.candidate_key_env),
        api_model=args.api_model, inputs=args.first_look_inputs,
    )


def handle(args: argparse.Namespace, *, transport: httpx.AsyncBaseTransport | None = None) -> int:
    root = args.dir.expanduser() if args.dir else records.default_root()
    repo = Path(args.repo).expanduser()
    try:
        base_ref = scan_mod.resolve_ref(repo, args.base)
        head_ref = scan_mod.resolve_ref(repo, args.head)
        base_sites = scan_mod.scan_ref(repo, base_ref)
        head_sites = scan_mod.scan_ref(repo, head_ref)
        prices = pricing_mod.load_prices(args.prices)
        look = first_look_config(args)
    except (scan_mod.GitError, ValueError, OSError) as exc:
        print(f"throttle pr-check: {exc}", file=sys.stderr)
        return EXIT_USAGE
    owned = _owned(args, root)
    doc = run_check(root, base_sites, head_sites, base_ref=base_ref, head_ref=head_ref,
                    options=Options(candidate=args.candidate, first_look=look, owned=owned, prices=prices,
                                    transport=transport))
    markdown = render_markdown(doc)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)
    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown:
        n = len(doc["jobs"])
        print(f"{n} AI job{'s' if n != 1 else ''} changed. Comment written to {args.markdown}.")
    return EXIT_OK


def _owned(args: argparse.Namespace, root: Path) -> dict[str, Any] | None:
    """The measured owned cost for the candidate, from `throttle check` history."""
    from .. import check as check_mod
    from .report import load_check, owned_cost

    names = [n for n in (args.candidate_model, args.candidate) if n]
    if not names:
        return None
    try:
        record = load_check(check_mod.history_dir(None), names)
    except (ValueError, OSError):
        return None
    block, _ = owned_cost(record)
    return block
