"""The Meter: the HTML face of ``throttle report`` and ``throttle demo``.

One self-contained page, no scripts and no network: fonts are system stacks,
the dial is inline SVG and every motion is CSS (switched off under
``prefers-reduced-motion``). It reads only the report document built by
:mod:`throttle.own.report` and changes nothing in it.

What a reader sees, top to bottom:

1. the instrument: "Saved per month" on an odometer (the conservative, low
   end figure) and a dial for the share of the API bill that can move;
2. the receipt: one line per kind of job, where it runs, what it costs on the
   API and on your model, and the proof in plain words;
3. the proof, one click deeper: win/tie/loss with 95% error bars, sample
   sizes and the judge's order disagreements;
4. the honest footer: what was measured and what was assumed.
"""

from __future__ import annotations

import html
import math
from typing import Any, Mapping, Sequence

MOVE = "MOVE"
NEEDS_FINETUNE = "NEEDS FINETUNE"
KEEP = "KEEP ON API"
MORE_DATA = "NEEDS MORE DATA"

JOB_NAMES = {
    "classification": "Sorting and labels",
    "extraction": "Pulling out fields",
    "summarization": "Summaries",
    "code": "Writing code",
    "agent": "Agent steps (tool calls)",
    "chat": "Chat answers",
    "unknown": "Unknown (no bodies kept)",
}

# recommendation -> (css class, receipt status, switch text)
STATUS = {
    MOVE: ("own", "Moves to your model", "YOUR MODEL"),
    KEEP: ("api", "Stays on the API", "API"),
    NEEDS_FINETUNE: ("test", "Needs a fine-tune first", "TESTING"),
    MORE_DATA: ("wait", "Needs more data", "TESTING"),
}

LOGO = ('<svg class="logo" viewBox="0 0 24 24" aria-hidden="true"><path d="M4.5 17a8 8 0 1 1 15 0"/>'
        '<path d="M12 13l4-4"/></svg>')


def e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def money(value: float | None, *, cents: bool | None = None) -> str:
    if value is None:
        return "-"
    if cents is None:
        cents = abs(value) < 100
    return f"${value:,.2f}" if cents else f"${value:,.0f}"


def pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.0%}"


def job_name(kind: str) -> str:
    return JOB_NAMES.get(kind, kind.replace("_", " ").capitalize())


# ---------------------------------------------------------------------------
# Report kind and labels
# ---------------------------------------------------------------------------


def mode(report: Mapping[str, Any]) -> str:
    if report.get("demo"):
        return "demo"
    if (report.get("source") or {}).get("kind") == "public_replay":
        return "replay"
    return "live"


def stamp(report: Mapping[str, Any]) -> tuple[str, str]:
    """(css class, words) for the label printed next to every big number."""

    kind = mode(report)
    if kind == "demo":
        return "demo", "DEMO"
    if kind == "replay":
        return "replay", "PUBLIC DATASET REPLAY"
    return "live", "YOUR TRAFFIC"


def tag(report: Mapping[str, Any]) -> str:
    cls, words = stamp(report)
    return f' <span class="tag t-{cls}">{e(words)}</span>' if cls != "live" else ""


# ---------------------------------------------------------------------------
# Pieces
# ---------------------------------------------------------------------------


def odometer(value: float | None, *, min_digits: int = 6) -> str:
    """Rolling digit wheels. Leading zeros are printed dim, like a real meter."""

    if value is None:
        blanks = "".join('<span class="wheel blank"><span class="strip"><i>-</i></span></span>' for _ in range(min_digits))
        return ('<div class="odometer" role="img" aria-label="no savings claim">'
                f'<span class="odo-cur">$</span>{blanks}<span class="odo-unit">/mo</span></div>')
    amount = max(0, int(round(value)))
    digits = str(amount).rjust(min_digits, "0")
    lead = len(digits) - len(str(amount)) if amount else len(digits) - 1
    wheels: list[str] = ['<span class="odo-cur">$</span>']
    count = len(digits)
    for i, ch in enumerate(digits):
        place = count - i
        if i and place % 3 == 0:
            wheels.append('<span class="odo-sep">,</span>')
        d = int(ch)
        strip = "".join(f"<i>{n}</i>" for n in range(10))
        wheels.append(
            f'<span class="wheel{" dim" if i < lead else ""}" style="--d:{d};--i:{i}">'
            f'<span class="strip">{strip}</span></span>'
        )
    label = money(value, cents=False)
    return (f'<div class="odometer" role="img" aria-label="{e(label)} per month">'
            + "".join(wheels) + '<span class="odo-unit">/mo</span></div>')


def _polar(cx: float, cy: float, r: float, deg: float) -> tuple[float, float]:
    rad = math.radians(deg)
    return cx + r * math.cos(rad), cy + r * math.sin(rad)


def _arc(cx: float, cy: float, r: float, start: float, end: float) -> str:
    x1, y1 = _polar(cx, cy, r, start)
    x2, y2 = _polar(cx, cy, r, end)
    large = 1 if (end - start) > 180 else 0
    return f"M{x1:.2f} {y1:.2f} A{r} {r} 0 {large} 1 {x2:.2f} {y2:.2f}"


def dial(share: float | None, caption: str) -> str:
    """A 240 degree dial; the needle sits on the share of the bill that can move."""

    cx, cy, r = 130.0, 128.0, 96.0
    start, sweep = 150.0, 240.0
    value = 0.0 if share is None else max(0.0, min(1.0, share))
    split = start + sweep * value
    parts = [f'<svg class="dial" viewBox="0 0 260 252" role="img" aria-label="{e(caption)}">',
             f'<circle cx="{cx}" cy="{cy}" r="118" class="bezel"/>',
             f'<circle cx="{cx}" cy="{cy}" r="110" class="face"/>']
    if value > 0.002:
        parts.append(f'<path d="{_arc(cx, cy, r, start, split)}" class="arc-own"/>')
    if value < 0.998:
        parts.append(f'<path d="{_arc(cx, cy, r, split, start + sweep)}" class="arc-api"/>')
    for k in range(0, 41):
        deg = start + sweep * k / 40
        major = k % 10 == 0
        x1, y1 = _polar(cx, cy, r - (16 if major else 10), deg)
        x2, y2 = _polar(cx, cy, r - 5, deg)
        parts.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" class="{"tk-major" if major else "tk"}"/>')
        if major:
            tx, ty = _polar(cx, cy, r - 30, deg)
            parts.append(f'<text x="{tx:.2f}" y="{ty + 4:.2f}" class="tk-num">{k * 25 // 10}</text>')
    angle = sweep * value - sweep / 2
    parts.append(f'<text x="{cx}" y="{cy + 46}" class="dial-read">{e(pct(share) if share is not None else "-")}</text>')
    parts.append(f'<text x="{cx}" y="{cy + 64}" class="dial-cap">CAN MOVE</text>')
    parts.append(f'<g class="needle" style="--a:{angle:.2f}deg">'
                 f'<path d="M{cx - 3.2} {cy + 14} L{cx} {cy - r + 8} L{cx + 3.2} {cy + 14} Z"/></g>')
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="9" class="hub"/><circle cx="{cx}" cy="{cy}" r="3" class="hub-dot"/>')
    parts.append("</svg>")
    return "".join(parts)


def switch(rec: str) -> str:
    cls, _, words = STATUS[rec]
    return (f'<span class="switch s-{cls}" aria-label="runs on: {e(words.lower())}">'
            f'<span class="track"><span class="knob"></span></span><span class="sw-text">{e(words)}</span></span>')


def proof_words(q: Mapping[str, Any] | None) -> str:
    if not q:
        return "Not tested yet: no answers from your model for this job."
    judged = int(q.get("judged") or 0)
    if not judged:
        return f"Not judged yet ({q.get('verdict') or 'no verdict'})."
    good = int(q.get("wins") or 0) + int(q.get("ties") or 0)
    text = f"As good or better in {good} of {judged} tests"
    unsure = int(q.get("order_disagree") or 0)
    if unsure:
        text += f"; judge unsure on {unsure}"
    worse = int(q.get("losses") or 0)
    if worse:
        text += f"; worse on {worse}"
    return text + "."


def _rate_bar(label: str, rate: float | None, ci: Sequence[float] | None, cls: str) -> str:
    if rate is None:
        return ""
    whisker = ""
    if ci and ci[0] is not None and ci[1] is not None:
        lo, hi = max(0.0, float(ci[0])), min(1.0, float(ci[1]))
        whisker = (f'<span class="whisker" style="left:{lo * 100:.1f}%;width:{max(0.4, (hi - lo) * 100):.1f}%"></span>')
        ci_text = f"{lo:.0%} to {hi:.0%}"
    else:
        ci_text = "no interval"
    return (f'<div class="rate"><span class="rate-name">{e(label)}</span>'
            f'<span class="rate-track"><span class="rate-fill {cls}" style="width:{rate * 100:.1f}%"></span>{whisker}</span>'
            f'<span class="rate-num">{rate:.0%} <small>95% CI {e(ci_text)}</small></span></div>')


def _net_axis(net: Sequence[float] | None) -> str:
    if not net:
        return '<p class="fine">No net interval: the judge did not score enough answers.</p>'
    lo, hi = float(net[0]), float(net[1])

    def x(v: float) -> float:
        return (max(-1.0, min(1.0, v)) + 1) * 50

    side = "own" if lo > 0 else ("api" if hi < 0 else "even")
    words = {"own": "your model wins more often", "api": "the API model wins more often",
             "even": "the range crosses zero: no clear winner"}[side]
    return (f'<div class="net"><div class="net-axis"><span class="net-band"></span><span class="net-zero"></span>'
            f'<span class="net-ci n-{side}" style="left:{x(lo):.1f}%;width:{max(0.6, x(hi) - x(lo)):.1f}%"></span></div>'
            f'<div class="net-scale"><span>API better</span><span>even</span><span>your model better</span></div>'
            f'<p class="fine">Wins minus losses, 95% CI [{lo:+.2f}, {hi:+.2f}]: {e(words)}. '
            'The shaded band is the plus or minus 5 point margin for "no material difference".</p></div>')


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def _banner(report: Mapping[str, Any], demo_banner: str) -> str:
    kind = mode(report)
    if kind == "demo":
        return f'<div class="banner b-demo" role="note"><b>{e(demo_banner)}.</b> Every number on this page is DEMO.</div>'
    if kind == "replay":
        src = report.get("source") or {}
        return (f'<div class="banner b-replay" role="note"><b>{e(src.get("headline_label") or "public dataset replay")}.</b> '
                f'{e(src.get("requests"))} real requests from {e(src.get("dataset"))} ({e(src.get("license"))}). '
                f'{e(src.get("api_stand_in"))} plays the API model, priced as {e(src.get("priced_as"))}. '
                'Not a customer.</div>')
    return ""


def _instrument(report: Mapping[str, Any]) -> str:
    totals = report["totals"]
    t = tag(report)
    savings = totals.get("conservative_savings_usd")
    point = totals.get("point_savings_usd")
    share = totals.get("movable_share")
    owned_total = totals.get("owned_usd_per_month")
    measured = report.get("owned_cost") is not None
    if savings is not None:
        lamp, lamp_text = "own", "SAVING"
        note = "Conservative: the low end of the 95% range, after paying for the always-on GPU."
        if point is not None and abs(point - savings) >= 1:
            note += f" Likely figure: {money(point)}/mo."
        elif totals.get("floor_applied"):
            note += " The always-on GPU sets your model's cost here, so the likely figure is the same."
    elif not measured:
        lamp, lamp_text = "api", "NOT MEASURED"
        hint = str(report.get("not_measured_hint") or "")
        hint = hint[len("not measured: "):] if hint.startswith("not measured: ") else hint
        note = "No savings claim: your model's running cost was not measured yet." + (f" To measure it, {hint}." if hint else "")
    else:
        lamp, lamp_text = "api", "NO CLAIM"
        note = "No savings claim. " + str(totals.get("savings_note") or "").rstrip(".") + "."
    jobs = report["workloads"]
    testing = sum(1 for w in jobs if w["recommendation"] in (NEEDS_FINETUNE, MORE_DATA))
    moving = sum(1 for w in jobs if w["recommendation"] == MOVE)
    staying = sum(1 for w in jobs if w["recommendation"] == KEEP)
    api = totals.get("monthly_api_usd")
    movable = totals.get("movable_api_usd")
    floor_note = ""
    if owned_total:
        floor_note = " (GPU floor)" if totals.get("floor_applied") else ""
    readouts = [
        ("api", "API bill per month", money(api) if api is not None else "not projected",
         "what the captured traffic costs at API prices"),
        ("own", "Share that can move", (f"{pct(share)}" + (f" &middot; {e(money(movable))}" if movable is not None else ""))
         if share is not None else "-", f"{moving} job{'s' if moving != 1 else ''} ready to move"),
        ("own2", "Your model costs", money(owned_total["high"]) if owned_total else
         ("not measured" if not measured else "-"),
         "per month, high end" + (", the always-on GPU floor" if floor_note else ", for the jobs that move")),
        ("test", "Still testing", str(testing), f"{staying} stay{'s' if staying == 1 else ''} on the API"),
    ]
    cells = "".join(
        f'<div class="readout r-{cls}"><span class="ro-name"><i class="dot"></i>{e(name)}</span>'
        f'<b class="ro-value">{value if cls == "own" else e(value)}</b><span class="ro-sub">{e(sub)}</span></div>'
        for cls, name, value, sub in readouts
    )
    # Legacy names kept for scripts that grep the page.
    legacy = "".join(f'<span class="legacy">{e(n)}: {e(v)}</span>' for n, v in _summary(report))
    return (
        '<section class="instrument" aria-label="Savings meter">'
        '<span class="screw s1"></span><span class="screw s2"></span><span class="screw s3"></span><span class="screw s4"></span>'
        '<div class="inst-main">'
        f'<div class="inst-label"><span>Saved per month{t}</span><span class="lamp l-{lamp}"><i></i>{e(lamp_text)}</span></div>'
        f'{odometer(savings)}'
        f'<p class="inst-note">{e(note)}</p>'
        '</div>'
        f'<div class="inst-dial">{dial(share, "share of the API bill that can move to your model")}'
        '<div class="dial-legend"><span><i class="dot d-own"></i>can move</span><span><i class="dot d-api"></i>stays on API</span></div></div>'
        f'<div class="readouts">{cells}</div>{legacy}'
        '</section>'
    )


def _summary(report: Mapping[str, Any]) -> list[tuple[str, str]]:
    from .report import summary_lines

    return summary_lines(report)


def _receipt(report: Mapping[str, Any]) -> str:
    totals = report["totals"]
    t = tag(report)
    jobs = sorted(report["workloads"], key=lambda w: -(w.get("api_usd_per_month") or 0))
    top = max([w.get("api_usd_per_month") or 0 for w in jobs] + [1e-9])
    lines: list[str] = []
    for w in jobs:
        rec = w["recommendation"]
        cls, status, _ = STATUS[rec]
        api = w.get("api_usd_per_month")
        owned = w.get("owned_usd_per_month")
        api_text = money(api) if api is not None else w.get("api_note") or "-"
        own_text = money(owned["high"]) if owned else ("not measured" if not str(w.get("owned_note", "")).startswith("not projected") else "not projected")
        api_w = (api or 0) / top * 100
        own_w = min(100.0, (owned["high"] if owned else 0) / top * 100)
        reqs = w.get("requests_per_month")
        reason = (w.get("reasons") or [""])[0]
        lines.append(
            f'<li class="line ln-{cls}">'
            f'<div class="ln-head">{switch(rec)}<span class="ln-name">{e(job_name(w["type"]))}</span>'
            f'<span class="leader"></span><span class="ln-status st-{cls}">{e(status)}</span></div>'
            f'<div class="ln-money"><span class="m-api">API <b class="cost-api">{e(api_text)}</b>/mo</span>'
            f'<span class="m-arrow">&rarr;</span><span class="m-own">{"your model" if rec == MOVE else "on your model it would be"} '
            f'<b class="cost-own{"" if owned else " nm"}">{e(own_text)}</b>{"/mo" if owned else ""}</span>'
            f'<span class="m-req">{e(f"{reqs:,.0f} requests/mo" if reqs is not None else "")}</span></div>'
            f'<div class="ln-bars"><span class="bar-api" style="width:{api_w:.1f}%"></span>'
            f'<span class="bar-own" style="width:{own_w:.1f}%"></span></div>'
            f'<p class="ln-proof">{e(proof_words(w.get("quality")))}</p>'
            + (f'<p class="ln-why">{e(reason)}</p>' if reason else "")
            + '</li>'
        )
    owned_total = totals.get("owned_usd_per_month")
    savings = totals.get("conservative_savings_usd")
    rows = [
        ("API bill now", money(totals.get("monthly_api_usd")) if totals.get("monthly_api_usd") is not None else "not projected"),
        ("Moves to your model", money(totals.get("movable_api_usd")) if totals.get("movable_api_usd") is not None else "-"),
        ("Your model, all-in", money(owned_total["high"]) if owned_total else "-"),
    ]
    total_rows = "".join(f'<div class="tot"><span>{e(a)}</span><span class="leader"></span><b>{e(b)}</b></div>' for a, b in rows)
    saved = money(savings) if savings is not None else ("not measured" if report.get("owned_cost") is None else "no claim")
    saved_cls = "" if savings is not None else ' class="nm"'
    when = str(report.get("generated_at") or "")[:16].replace("T", " ")
    return (
        f'<section class="receipt-wrap"><div class="sec-head"><h2>The receipt{t}</h2>'
        '<p>One line per kind of job. Where it runs, what it costs, and the proof.</p></div>'
        '<div class="receipt"><div class="rc-top">'
        f'<div class="rc-brand">{LOGO}<b>THROTTLE</b><span>METER RECEIPT</span></div>'
        f'<div class="rc-meta"><span>{e(when)} UTC</span><span>model {e(report.get("candidate_model") or report.get("candidate"))}</span>'
        f'<span>eval {e(report.get("eval_run_id") or "none")}</span></div></div>'
        f'<ol class="lines">{"".join(lines)}</ol>'
        f'<div class="totals">{total_rows}<div class="tot grand"><span>SAVED PER MONTH</span><span class="leader"></span><b{saved_cls}>{e(saved)}</b></div>'
        '<p class="fine">Saved = API bill of the jobs that move, minus your model at the high end of its 95% range. '
        'The always-on GPU is counted once. Per-job model cost above is token cost only; the GPU floor is in '
        '"Your model, all-in".</p></div>'
        '<div class="rc-foot"><span class="barcode" aria-hidden="true"></span>'
        '<p>Nothing moves without your OK. If your model stumbles, the request goes to the API.</p></div>'
        '</div></section>'
    )


def _proof(report: Mapping[str, Any]) -> str:
    t = tag(report)
    items: list[str] = []
    for w in report["workloads"]:
        q = w.get("quality")
        rec = w["recommendation"]
        cls, status, _ = STATUS[rec]
        if not q:
            body = '<p class="fine">No answers from your model for this job yet, so nothing was judged.</p>'
            n_text = "not tested"
        else:
            judged = int(q.get("judged") or 0)
            n_text = f"n={q.get('n')}, judged {judged}"
            stack = ""
            if judged:
                parts = [("own", q.get("wins") or 0, "better"), ("tie", q.get("ties") or 0, "same"),
                         ("unsure", q.get("order_disagree") or 0, "judge unsure"), ("api", q.get("losses") or 0, "worse")]
                stack = '<div class="stack">' + "".join(
                    f'<span class="sk-{c}" style="flex:{v}" title="{e(word)}: {v}"></span>' for c, v, word in parts if v
                ) + '</div><div class="stack-key">' + "".join(
                    f'<span><i class="dot d-{c}"></i>{e(word)} {v}</span>' for c, v, word in parts
                ) + '</div>'
            bars = (_rate_bar("Your model better", q.get("win_rate"), q.get("win_rate_ci"), "own")
                    + _rate_bar("Same", q.get("tie_rate"), q.get("tie_rate_ci"), "tie")
                    + _rate_bar("Your model worse", q.get("loss_rate"), q.get("loss_rate_ci"), "api"))
            disagree = int(q.get("order_disagree") or 0)
            judge_note = (f"The judge read every pair twice with the answers swapped. It changed its mind on {disagree} "
                          f"of {judged} ({pct(q.get('order_disagree_rate'))}); those count as a tie, not a win."
                          if judged else "Not judged yet.")
            fmt = q.get("format_checks") or {}
            fmt_rows = "".join(
                f'<tr><td>{e(name.replace("_", " "))}</td><td>{e(pct(c.get("incumbent")))}</td>'
                f'<td>{e(pct(c.get("candidate")))}</td><td>{"overlap" if c.get("overlap") else "<b>below the API</b>"}</td></tr>'
                for name, c in fmt.items()
            )
            fmt_html = (f'<table class="fmt"><thead><tr><th>Format check</th><th>API</th><th>Your model</th><th>95% CIs</th>'
                        f'</tr></thead><tbody>{fmt_rows}</tbody></table>') if fmt_rows else ""
            reasons = "".join(f"<li>{e(r)}</li>" for r in (q.get("reasons") or []))
            body = (f'{stack}<div class="rates">{bars}</div>{_net_axis(q.get("net_ci"))}'
                    f'<p class="fine">{e(judge_note)}</p>{fmt_html}'
                    f'<div class="verdict">Verdict: <b>{e(q.get("verdict"))}</b></div>'
                    + (f'<ul class="why">{reasons}</ul>' if reasons else ""))
        recs = "".join(f"<li>{e(r)}</li>" for r in w.get("reasons") or [])
        items.append(
            f'<details class="pf pf-{cls}"><summary><span class="pf-name">{e(job_name(w["type"]))}</span>'
            f'<span class="pf-n">{e(n_text)}</span><span class="chip c-{cls}">{e(status)}</span>'
            '<span class="chev" aria-hidden="true"></span></summary>'
            f'<div class="pf-body">{body}<div class="pf-rec"><span>Recommendation: <b>{e(rec)}</b></span>'
            f'<ul class="why">{recs}</ul></div></div></details>'
        )
    return (f'<section class="proof"><div class="sec-head"><h2>The proof, one click deeper{t}</h2>'
            '<p>Open a job to see win, tie and loss with 95% error bars, sample sizes and how often the judge '
            'disagreed with itself.</p></div>' + "".join(items) + '</section>')


def _ledger(report: Mapping[str, Any], rules: str) -> str:
    t = tag(report)

    def chip(label: str) -> str:
        word = label.upper()
        cls = ("assumed" if "ASSUMED" in word else "warn" if "WARNING" in word or "NOT MEASURED" in word
               else "demo" if "DEMO" in word else "public" if "PUBLIC" in word else "measured")
        return f'<span class="lbl lb-{cls}">{e(label)}</span>'

    rows = "".join(f'<tr><td>{e(i["what"])}</td><td>{chip(i["label"])}</td><td>{e(i["detail"])}</td></tr>'
                   for i in report["labels"])
    limits = "".join(f"<li>{e(x)}</li>" for x in report["does_not_prove"])
    return (
        f'<section class="ledger"><div class="sec-head"><h2>How we got these numbers{t}</h2>'
        '<p>Every number is labelled. MEASURED came from real responses. ASSUMED is a number you or we chose.</p></div>'
        f'<table class="labels"><thead><tr><th>Number</th><th>Label</th><th>Where it comes from</th></tr></thead><tbody>{rows}</tbody></table>'
        f'<h3>What this does not prove</h3><ul class="limits">{limits}</ul>'
        f'<p class="fine rules">{e(rules)}</p></section>'
    )


def render(report: Mapping[str, Any], *, demo_banner: str, rules: str) -> str:
    demo = bool(report.get("demo"))
    kind = mode(report)
    title_prefix = "DEMO " if demo else ("Replay " if kind == "replay" else "")
    cls, words = stamp(report)
    sub = {"demo": "synthetic traffic, fake models", "replay": "public dataset replay, local models, GPU price assumed",
           "live": "your captured traffic, GPU price assumed"}[kind]
    head = (
        '<header class="top">'
        f'<div class="brand">{LOGO}<span class="word">Throttle</span><span class="kind">Meter report</span></div>'
        f'<div class="meta"><span class="stamp st-{cls}">{e(words)}</span>'
        f'<span>{e(report.get("candidate"))} vs the API</span><span>{e(sub)}</span></div>'
        '</header>'
    )
    footer = (f'<footer><span>{LOGO} Generated by <code>throttle {"demo" if demo else "report"}</code> '
              f'at {e(report.get("generated_at"))}.</span><span>Stop renting your AI. '
              'Self-contained page: no scripts, no external assets.</span></footer>')
    return "\n".join([
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f'<title>{title_prefix}Throttle report: {e(report["candidate"])}</title>',
        f'<style>{CSS}</style></head><body><main class="page">',
        head,
        _banner(report, demo_banner),
        _instrument(report),
        _receipt(report),
        _proof(report),
        _ledger(report, rules),
        footer,
        '</main></body></html>',
    ]) + "\n"


CSS = """
:root{
--paper:#f3ede1;--paper2:#e9e1d1;--card:#fbf8f1;--ink:#1c1d20;--ink2:#3a3b40;--muted:#6f6a60;--rule:#d8cfbd;
--graphite:#1d1e21;--graphite2:#2a2b2f;--graphite3:#3a3c41;--face:#f6f1e6;
--api:#ff5b1f;--api-soft:#ffe1d3;--api-ink:#b83700;
--own:#13a55a;--own-soft:#d5f1df;--own-ink:#0b6e3a;
--test:#7c5cff;--test-soft:#e7e0ff;--test-ink:#4d32c9;
--cobalt:#2443d6;--tie:#b9b2a3;--warn:#c2410c;
--sans:"Helvetica Neue",Helvetica,"Inter","Segoe UI",Arial,system-ui,sans-serif;
--mono:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,"Liberation Mono",monospace;
}
*{box-sizing:border-box}
html{background:var(--paper)}
body{margin:0;color:var(--ink);font:15px/1.5 var(--sans);background:var(--paper);
background-image:radial-gradient(rgba(60,50,30,.05) 1px,transparent 1px);background-size:4px 4px}
.page{max-width:1100px;margin:0 auto;padding:28px 24px 56px}
code{font-family:var(--mono);font-size:.92em}
.logo{width:22px;height:22px;fill:none;stroke:currentColor;stroke-width:2.4;stroke-linecap:round;vertical-align:middle}
.top{display:flex;justify-content:space-between;align-items:flex-end;gap:16px;flex-wrap:wrap;padding-bottom:14px;border-bottom:2px solid var(--ink)}
.brand{display:flex;align-items:center;gap:10px}
.brand .logo{width:30px;height:30px;color:var(--api)}
.brand .word{font-weight:800;font-size:26px;letter-spacing:-.02em}
.brand .kind{font:600 11px/1 var(--mono);letter-spacing:.14em;text-transform:uppercase;color:var(--muted);
border:1px solid var(--rule);border-radius:3px;padding:5px 7px;margin-left:4px}
.meta{display:flex;gap:14px;align-items:center;flex-wrap:wrap;font:12px/1.3 var(--mono);color:var(--muted)}
.stamp{font-weight:700;letter-spacing:.1em;padding:4px 8px;border-radius:3px;color:#fff;background:var(--ink)}
.st-demo{background:var(--api)}.st-replay{background:var(--cobalt)}.st-live{background:var(--own)}
.banner{margin:16px 0 0;padding:11px 14px;border-radius:6px;font-size:13.5px;border:1.5px solid}
.b-demo{background:#fff1e8;border-color:var(--api);color:#7a2600}
.b-replay{background:#eaeefe;border-color:var(--cobalt);color:#182c8c}
.tag{display:inline-block;font:700 10px/1 var(--mono);letter-spacing:.08em;padding:4px 6px;border-radius:3px;vertical-align:middle;margin-left:6px}
.t-demo{background:var(--api);color:#fff}.t-replay{background:var(--cobalt);color:#fff}
.legacy{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}

/* instrument */
.instrument{position:relative;margin:22px 0 0;padding:30px 34px 22px;border-radius:18px;color:var(--face);
background:linear-gradient(180deg,#26272b 0%,var(--graphite) 60%,#18191b 100%);
box-shadow:0 1px 0 rgba(255,255,255,.08) inset,0 0 0 1px #0d0e10,0 18px 40px -18px rgba(30,20,10,.55);
display:grid;grid-template-columns:1fr 290px;gap:10px 34px}
.screw{position:absolute;width:10px;height:10px;border-radius:50%;background:radial-gradient(circle at 35% 35%,#6a6c72,#2b2c30);
box-shadow:0 0 0 1px #0b0b0c}
.screw:after{content:"";position:absolute;left:2px;right:2px;top:4.2px;height:1.6px;background:#18191b;transform:rotate(35deg)}
.s1{left:12px;top:12px}.s2{right:12px;top:12px}.s3{left:12px;bottom:12px}.s4{right:12px;bottom:12px}
.inst-main{display:flex;flex-direction:column;justify-content:center;min-width:0}
.inst-label{display:flex;align-items:center;gap:14px;font:600 12px/1 var(--mono);letter-spacing:.16em;text-transform:uppercase;color:#bdb6a8}
.lamp{display:inline-flex;align-items:center;gap:7px;font-size:10.5px;letter-spacing:.12em;padding:5px 8px;border:1px solid #3d3f44;border-radius:20px}
.lamp i{width:8px;height:8px;border-radius:50%}
.l-own i{background:var(--own);box-shadow:0 0 0 2px rgba(19,165,90,.25),0 0 10px var(--own)}.l-own{color:#7ee2a8}
.l-api i{background:var(--api);box-shadow:0 0 0 2px rgba(255,91,31,.25),0 0 10px var(--api)}.l-api{color:#ffb08f}
.odometer{display:flex;align-items:center;gap:5px;margin:18px 0 14px;font:700 clamp(40px,7.4vw,82px)/1 var(--mono);flex-wrap:nowrap}
.odo-cur{color:#8d887d;font-size:.62em;margin-right:4px}
.odo-sep{color:#8d887d;font-size:.6em;align-self:flex-end;margin:0 -1px .05em}
.odo-unit{font:600 15px/1 var(--mono);color:#8d887d;align-self:flex-end;margin-left:8px;padding-bottom:.4em}
.wheel{position:relative;display:inline-block;width:.72em;height:1.18em;overflow:hidden;border-radius:6px;
background:linear-gradient(180deg,#0b0b0d 0%,#232428 18%,#2c2d31 50%,#232428 82%,#0b0b0d 100%);
box-shadow:inset 0 0 0 1px #000,inset 0 2px 6px rgba(0,0,0,.7),0 1px 0 rgba(255,255,255,.07)}
.wheel:after{content:"";position:absolute;left:0;right:0;top:50%;height:1px;background:rgba(0,0,0,.55);box-shadow:0 1px 0 rgba(255,255,255,.05)}
.wheel .strip{position:absolute;left:0;right:0;top:0;transform:translateY(calc(var(--d) * -1.18em));
animation:roll 1.5s cubic-bezier(.3,1.35,.5,1) both;animation-delay:calc(var(--i) * 90ms + 200ms)}
.wheel i{display:block;height:1.18em;line-height:1.18em;text-align:center;font-style:normal;color:#f8f3e8}
.wheel.dim i{color:#56544f}
.wheel.blank .strip{animation:none;transform:none}.wheel.blank i{color:#56544f}
@keyframes roll{from{transform:translateY(-10.62em)}}
.inst-note{margin:0;color:#bdb6a8;font-size:13.5px;max-width:560px}
.inst-dial{grid-row:span 1;display:flex;flex-direction:column;align-items:center}
.dial{width:100%;max-width:280px;height:auto}
.dial .bezel{fill:#121315;stroke:#3a3c41;stroke-width:1}
.dial .face{fill:var(--face)}
.dial .arc-own{fill:none;stroke:var(--own);stroke-width:12}
.dial .arc-api{fill:none;stroke:var(--api);stroke-width:12}
.dial .tk{stroke:#7d776b;stroke-width:1}
.dial .tk-major{stroke:var(--ink);stroke-width:2.2}
.dial .tk-num{font:600 11px var(--mono);fill:var(--ink2);text-anchor:middle}
.dial .dial-read{font:700 26px var(--mono);fill:var(--ink);text-anchor:middle}
.dial .dial-cap{font:700 9.5px var(--mono);letter-spacing:.18em;fill:var(--muted);text-anchor:middle}
.dial .needle{transform-box:view-box;transform-origin:130px 128px;transform:rotate(var(--a));
animation:swing 1.6s cubic-bezier(.25,1.6,.45,1) .25s both}
.dial .needle path{fill:var(--ink)}
.dial .hub{fill:var(--graphite)}.dial .hub-dot{fill:var(--api)}
@keyframes swing{from{transform:rotate(-120deg)}}
.dial-legend{display:flex;gap:14px;font:11px var(--mono);color:#bdb6a8;margin-top:2px}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;vertical-align:-1px}
.d-own{background:var(--own)}.d-api{background:var(--api)}.d-tie{background:var(--tie)}.d-unsure{background:var(--test)}
.readouts{grid-column:1/-1;display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:14px;
padding-top:16px;border-top:1px solid #34363b}
.readout{background:#141517;border:1px solid #303237;border-radius:10px;padding:10px 12px 11px;min-width:0}
.ro-name{display:block;font:600 10.5px/1.2 var(--mono);letter-spacing:.1em;text-transform:uppercase;color:#a29c90}
.ro-value{display:block;font:700 22px/1.25 var(--mono);color:#fbf6ea;margin-top:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ro-sub{display:block;font-size:11.5px;color:#8d887d;margin-top:2px}
.r-api .dot{background:var(--api)}.r-own .dot{background:var(--own)}.r-own2 .dot{background:transparent;box-shadow:inset 0 0 0 2px var(--own)}
.r-test .dot{background:var(--test)}
.r-api .ro-value{color:#ffa47f}.r-own .ro-value{color:#7ee2a8}.r-test .ro-value{color:#b9a8ff}

/* sections */
section{margin-top:44px}
.sec-head{display:flex;align-items:baseline;justify-content:space-between;gap:20px;flex-wrap:wrap;margin-bottom:14px}
.sec-head h2{margin:0;font-size:24px;letter-spacing:-.015em}
.sec-head p{margin:0;color:var(--muted);font-size:13.5px;max-width:520px}
h3{font-size:15px;margin:26px 0 8px}
.fine{color:var(--muted);font-size:12.5px;margin:8px 0 0}

/* receipt */
.receipt{position:relative;max-width:780px;margin:0 auto;background:var(--card);padding:30px 34px 26px;font-family:var(--mono);
box-shadow:0 1px 0 #fff inset,0 14px 30px -18px rgba(60,40,10,.45),0 2px 4px rgba(60,40,10,.08)}
.receipt:before,.receipt:after{content:"";position:absolute;left:0;right:0;height:12px;
background:radial-gradient(circle at 8px 0,var(--paper) 6px,transparent 6.5px) 0 0/16px 12px repeat-x}
.receipt:before{top:-1px}
.receipt:after{bottom:-1px;transform:rotate(180deg)}
.rc-top{display:flex;justify-content:space-between;gap:14px;flex-wrap:wrap;padding-bottom:12px;border-bottom:1.5px dashed var(--rule)}
.rc-brand{display:flex;align-items:center;gap:8px}
.rc-brand .logo{color:var(--api)}
.rc-brand b{letter-spacing:.2em;font-size:14px}.rc-brand span{font-size:11px;color:var(--muted);letter-spacing:.14em}
.rc-meta{display:flex;flex-direction:column;align-items:flex-end;font-size:11px;color:var(--muted)}
.lines{list-style:none;margin:0;padding:0}
.line{padding:14px 0 12px;border-bottom:1.5px dashed var(--rule)}
.ln-head{display:flex;align-items:center;gap:12px}
.ln-name{font-weight:700;font-size:14.5px;white-space:nowrap}
.leader{flex:1;border-bottom:2px dotted #c7bda9;transform:translateY(-4px);min-width:16px}
.ln-status{font-weight:700;font-size:12.5px;letter-spacing:.02em;white-space:nowrap}
.st-own{color:var(--own-ink)}.st-api{color:var(--api-ink)}.st-test,.st-wait{color:var(--test-ink)}
.ln-money{display:flex;flex-wrap:wrap;align-items:baseline;gap:6px 10px;margin:8px 0 0 64px;font-size:13px;color:var(--ink2)}
.ln-money b{font-size:15px}.cost-api{color:var(--api-ink)}.cost-own{color:var(--own-ink)}
.m-arrow{color:var(--muted)}.cost-own.nm{color:var(--muted);font-size:13px;font-weight:600}
.tot.grand b.nm{color:#bdb6a8;font-size:15px}.m-req{margin-left:auto;color:var(--muted);font-size:11.5px}
.ln-bars{position:relative;margin:7px 0 0 64px;height:6px;background:var(--paper2);border-radius:3px;overflow:hidden}
.bar-api{position:absolute;left:0;top:0;bottom:0;background:var(--api);opacity:.85;border-radius:3px}
.bar-own{position:absolute;left:0;top:0;bottom:0;background:var(--own);border-radius:3px}
.ln-proof{margin:8px 0 0 64px;font-size:12.5px;color:var(--ink)}
.ln-why{margin:3px 0 0 64px;font-size:11.5px;color:var(--muted)}
.switch{display:inline-flex;align-items:center;gap:6px;width:52px;flex-direction:column}
.switch .track{position:relative;width:44px;height:22px;border-radius:11px;background:#d9d1c1;box-shadow:inset 0 1px 3px rgba(0,0,0,.25)}
.switch .knob{position:absolute;top:2px;left:2px;width:18px;height:18px;border-radius:50%;background:#fff;
box-shadow:0 1px 2px rgba(0,0,0,.35);transition:left .2s}
.sw-text{font-size:8.5px;letter-spacing:.1em;font-weight:700;color:var(--muted);white-space:nowrap}
.s-own .track{background:var(--own)}.s-own .knob{left:24px}.s-own .sw-text{color:var(--own-ink)}
.s-api .track{background:var(--api)}.s-api .sw-text{color:var(--api-ink)}
.s-test .track,.s-wait .track{background:var(--test)}.s-test .knob,.s-wait .knob{left:13px}
.s-wait .track{background:repeating-linear-gradient(135deg,var(--test) 0 5px,#9a83ff 5px 10px)}
.s-test .sw-text,.s-wait .sw-text{color:var(--test-ink)}
.totals{padding-top:12px}
.tot{display:flex;align-items:baseline;gap:10px;font-size:13.5px;padding:3px 0}
.tot b{font-size:15px}
.tot.grand{margin-top:8px;padding:10px 12px;background:var(--graphite);color:var(--face);border-radius:4px;letter-spacing:.06em}
.tot.grand .leader{border-color:#55575c}.tot.grand b{color:#7ee2a8;font-size:20px;letter-spacing:0}
.rc-foot{display:flex;align-items:center;gap:16px;margin-top:18px;padding-top:14px;border-top:1.5px dashed var(--rule)}
.rc-foot p{margin:0;font-size:11.5px;color:var(--muted)}
.barcode{flex:none;width:130px;height:34px;background:repeating-linear-gradient(90deg,var(--ink) 0 2px,transparent 2px 4px,var(--ink) 4px 5px,
transparent 5px 8px,var(--ink) 8px 11px,transparent 11px 12px,var(--ink) 12px 13px,transparent 13px 16px)}

/* proof */
.pf{background:var(--card);border:1px solid var(--rule);border-left:6px solid var(--tie);border-radius:10px;margin:0 0 10px;overflow:hidden}
.pf-own{border-left-color:var(--own)}.pf-api{border-left-color:var(--api)}.pf-test,.pf-wait{border-left-color:var(--test)}
.pf summary{list-style:none;cursor:pointer;display:flex;align-items:center;gap:14px;padding:14px 16px}
.pf summary::-webkit-details-marker{display:none}
.pf-name{font-weight:700;font-size:15.5px}
.pf-n{font:12px var(--mono);color:var(--muted);margin-left:auto}
.chip{font:700 11px/1 var(--mono);letter-spacing:.04em;padding:6px 9px;border-radius:20px;white-space:nowrap}
.c-own{background:var(--own-soft);color:var(--own-ink)}.c-api{background:var(--api-soft);color:var(--api-ink)}
.c-test,.c-wait{background:var(--test-soft);color:var(--test-ink)}
.chev{width:9px;height:9px;border-right:2px solid var(--ink2);border-bottom:2px solid var(--ink2);transform:rotate(45deg);
transition:transform .15s;margin:0 4px 4px}
.pf[open] .chev{transform:rotate(-135deg);margin-bottom:-2px}
.pf-body{padding:4px 18px 18px;border-top:1px dashed var(--rule)}
.stack{display:flex;height:16px;border-radius:4px;overflow:hidden;margin-top:14px;box-shadow:inset 0 0 0 1px rgba(0,0,0,.06)}
.sk-own{background:var(--own)}.sk-tie{background:var(--tie)}.sk-unsure{background:var(--test)}.sk-api{background:var(--api)}
.stack-key{display:flex;gap:16px;flex-wrap:wrap;font:11.5px var(--mono);color:var(--ink2);margin-top:7px}
.rates{margin-top:14px}
.rate{display:grid;grid-template-columns:150px 1fr 190px;gap:12px;align-items:center;padding:4px 0}
.rate-name{font-size:13px}
.rate-track{position:relative;height:12px;background:var(--paper2);border-radius:3px}
.rate-fill{position:absolute;left:0;top:0;bottom:0;border-radius:3px}
.rate-fill.own{background:var(--own)}.rate-fill.tie{background:var(--tie)}.rate-fill.api{background:var(--api)}
.whisker{position:absolute;top:-4px;bottom:-4px;border-left:2px solid var(--ink);border-right:2px solid var(--ink)}
.whisker:after{content:"";position:absolute;left:0;right:0;top:50%;border-top:2px solid var(--ink)}
.rate-num{font:700 13px var(--mono)}.rate-num small{font-weight:400;color:var(--muted);font-size:11px;margin-left:4px}
.net{margin-top:16px}
.net-axis{position:relative;height:22px;background:linear-gradient(90deg,var(--api-soft),#f3eee4 50%,var(--own-soft));border-radius:4px}
.net-band{position:absolute;left:47.5%;width:5%;top:0;bottom:0;background:rgba(0,0,0,.07)}
.net-zero{position:absolute;left:50%;top:-3px;bottom:-3px;border-left:1.5px dashed var(--ink2)}
.net-ci{position:absolute;top:7px;height:8px;border-radius:4px;background:var(--ink)}
.n-own{background:var(--own-ink)}.n-api{background:var(--api-ink)}.n-even{background:var(--test-ink)}
.net-scale{display:flex;justify-content:space-between;font:10.5px var(--mono);color:var(--muted);margin-top:4px;text-transform:uppercase;letter-spacing:.08em}
.fmt{width:100%;border-collapse:collapse;font-size:12.5px;margin-top:12px}
.fmt th,.fmt td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--rule)}
.fmt th{font:600 10.5px var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.verdict{margin-top:12px;font:12.5px var(--mono)}
.pf-rec{margin-top:12px;padding:10px 12px;background:var(--paper);border-radius:6px;font-size:13px}
ul.why{margin:6px 0 0;padding-left:18px;font-size:12.5px;color:var(--ink2)}
ul.why li{margin:2px 0}

/* ledger */
.labels{width:100%;border-collapse:collapse;font-size:13px;background:var(--card);border:1px solid var(--rule);border-radius:10px;overflow:hidden}
.labels th,.labels td{text-align:left;padding:9px 12px;border-bottom:1px solid var(--rule);vertical-align:top}
.labels th{font:600 10.5px var(--mono);letter-spacing:.1em;text-transform:uppercase;color:var(--muted);background:var(--paper2)}
.labels td:first-child{font-weight:600;white-space:nowrap}
.lbl{display:inline-block;font:700 10.5px/1.2 var(--mono);letter-spacing:.04em;padding:4px 7px;border-radius:3px;white-space:nowrap}
.lb-measured{background:var(--own-soft);color:var(--own-ink)}.lb-assumed{background:var(--api-soft);color:var(--api-ink)}
.lb-demo{background:var(--api);color:#fff}.lb-warn{background:#fde2d0;color:var(--warn)}
.lb-public{background:#e1e7fd;color:#182c8c}
ul.limits{padding-left:18px;font-size:13.5px;color:var(--ink2)}ul.limits li{margin-bottom:6px}
.rules{max-width:820px}
footer{margin-top:44px;padding-top:14px;border-top:2px solid var(--ink);display:flex;justify-content:space-between;gap:14px;
flex-wrap:wrap;font:11.5px var(--mono);color:var(--muted)}
footer .logo{width:14px;height:14px;color:var(--api)}

@media (max-width:860px){
.instrument{grid-template-columns:1fr;padding:26px 20px 18px}
.readouts{grid-template-columns:1fr 1fr}
.inst-dial{order:2}.readouts{order:3}
.rate{grid-template-columns:1fr;gap:4px}
}
@media (max-width:560px){
.page{padding:18px 14px 40px}
.receipt{padding:26px 16px 20px}
.ln-money,.ln-bars,.ln-proof,.ln-why{margin-left:0}
.ln-name{white-space:normal}
.readouts{grid-template-columns:1fr}
.odometer{gap:3px}
.labels td:first-child{white-space:normal}
.ln-head{flex-wrap:wrap;gap:8px 12px}.ln-head .leader{display:none}.ln-status{white-space:normal;flex-basis:100%}
.m-req{margin-left:0}
.labels,.fmt{display:block;overflow-x:auto}
.pf summary{flex-wrap:wrap;gap:8px}.pf-n{margin-left:0}
.rc-meta{align-items:flex-start}
.odometer{font-size:34px}
}
@media (prefers-reduced-motion:reduce){*,*:before,*:after{animation:none!important;transition:none!important}}
@media print{
body{background:#fff}.page{max-width:none;padding:0}
.instrument,.receipt,.pf{break-inside:avoid;box-shadow:none}
.wheel .strip,.dial .needle{animation:none}
details.pf>.pf-body{display:block}
}
"""
