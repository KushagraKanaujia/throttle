"""The HTML face of ``throttle report`` and ``throttle demo``.

One self-contained page: no scripts, no network, no external assets. Type is
a high-contrast serif for headlines and figures (Fraunces or Newsreader when
installed, else the system's best serif) and a clean sans for reading, with
tabular figures. Midnight navy ground, ivory text, warm gold for money.
Pure typography and data. It reads only the report document built by
:mod:`throttle.own.report` and changes nothing in it.

What a reader sees, top to bottom:

1. saved per month (the conservative, low end figure), the four numbers
   behind it, and where the API bill goes, job by job, as one bar;
2. job by job: where each kind of job runs, what it costs on the API and on
   your model, and the proof in plain words;
3. the proof, one click deeper: win/tie/loss with 95% error bars, sample
   sizes and the judge's order disagreements;
4. how we got these numbers: what was measured and what was assumed.
"""

from __future__ import annotations

import html
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
    "agent": "Agent steps",
    "chat": "Chat answers",
    "unknown": "Unknown (no bodies kept)",
}

# recommendation -> (css class, plain status)
STATUS = {
    MOVE: ("own", "Moves to your model"),
    KEEP: ("api", "Stays on the API"),
    NEEDS_FINETUNE: ("test", "Needs a fine-tune first"),
    MORE_DATA: ("test", "Needs more data"),
}


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
    """(css class, words) for the label printed next to every headline number."""

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
        whisker = f'<span class="whisker" style="left:{lo * 100:.1f}%;width:{max(0.4, (hi - lo) * 100):.1f}%"></span>'
        ci_text = f"{lo:.0%} to {hi:.0%}"
    else:
        ci_text = "no interval"
    return (f'<div class="rate"><span class="rate-name">{e(label)}</span>'
            f'<span class="rate-track"><span class="rate-fill f-{cls}" style="width:{rate * 100:.1f}%"></span>{whisker}</span>'
            f'<span class="rate-num">{rate:.0%}<small>95% CI {e(ci_text)}</small></span></div>')


def _net_axis(net: Sequence[float] | None) -> str:
    if not net:
        return '<p class="fine">No net interval: the judge did not score enough answers.</p>'
    lo, hi = float(net[0]), float(net[1])

    def x(v: float) -> float:
        return (max(-1.0, min(1.0, v)) + 1) * 50

    side = "own" if lo > 0 else ("api" if hi < 0 else "even")
    words = {"own": "your model wins more often", "api": "the API model wins more often",
             "even": "the range crosses zero, so there is no clear winner"}[side]
    return (f'<div class="net"><div class="net-axis"><span class="net-band"></span><span class="net-zero"></span>'
            f'<span class="net-ci n-{side}" style="left:{x(lo):.1f}%;width:{max(0.6, x(hi) - x(lo)):.1f}%"></span></div>'
            '<div class="net-scale"><span>API better</span><span>even</span><span>your model better</span></div>'
            f'<p class="fine">Wins minus losses, 95% CI [{lo:+.2f}, {hi:+.2f}]: {e(words)}. '
            'The shaded band is the plus or minus 5 point margin for "no material difference".</p></div>')


# ---------------------------------------------------------------------------
# Sections
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


def _bill_bar(report: Mapping[str, Any]) -> str:
    """Where the API bill goes: one bar, one segment per job, colored by where it runs."""

    jobs = sorted(report["workloads"], key=lambda w: (
        {MOVE: 0, NEEDS_FINETUNE: 1, MORE_DATA: 1, KEEP: 2}[w["recommendation"]], -(w.get("api_usd_per_month") or 0)))
    total = sum(w.get("api_usd_per_month") or 0 for w in jobs)
    if total <= 0:
        return ""
    segs, keys = [], []
    for w in jobs:
        api = w.get("api_usd_per_month") or 0
        if api <= 0:
            continue
        cls, status = STATUS[w["recommendation"]]
        share = api / total * 100
        segs.append(f'<span class="seg g-{cls}" style="flex:{api:.4f}" title="{e(job_name(w["type"]))}: {e(money(api))}/mo">'
                    + (f'<em>{e(job_name(w["type"]))}</em>' if share >= 12 else "") + '</span>')
        keys.append(f'<li><i class="dot d-{cls}"></i><span>{e(job_name(w["type"]))}</span><b>{e(money(api))}</b></li>')
    owned = (report["totals"].get("owned_usd_per_month") or {}).get("high")
    owned_row = ""
    if owned:
        owned_row = (f'<div class="bar-label">Your model, all-in, for the jobs that move</div>'
                     f'<div class="bill own-bar"><span class="seg g-gold" style="width:{min(100.0, owned / total * 100):.2f}%"></span></div>')
    return (
        '<div class="billing"><div class="bar-label">Where the API bill goes, per month</div>'
        f'<div class="bill">{"".join(segs)}</div>{owned_row}'
        f'<ul class="bill-key">{"".join(keys)}</ul>'
        '<p class="legend"><span><i class="dot d-own"></i>moves to your model</span>'
        '<span><i class="dot d-test"></i>still testing</span><span><i class="dot d-api"></i>stays on the API</span>'
        + ('<span><i class="dot d-gold"></i>your model, all-in</span>' if owned else '') + '</p></div>'
    )


def _hero(report: Mapping[str, Any]) -> str:
    totals = report["totals"]
    t = tag(report)
    savings = totals.get("conservative_savings_usd")
    point = totals.get("point_savings_usd")
    share = totals.get("movable_share")
    owned_total = totals.get("owned_usd_per_month")
    measured = report.get("owned_cost") is not None
    if savings is not None:
        figure = (f'<p class="figure" aria-label="{e(money(savings, cents=False))} per month">'
                  f'<span class="cur">$</span>{e(f"{round(savings):,}")}<span class="per">a month</span></p>')
        note = "Conservative: the low end of the 95% range, after paying for the always-on GPU."
        if point is not None and abs(point - savings) >= 1:
            note += f" Likely figure: {money(point)} a month."
        elif totals.get("floor_applied"):
            note += " The always-on GPU sets your model's cost here, so the likely figure is the same."
    else:
        figure = '<p class="figure none" aria-label="no savings claim">No savings claim yet</p>'
        if not measured:
            hint = str(report.get("not_measured_hint") or "")
            hint = hint[len("not measured: "):] if hint.startswith("not measured: ") else hint
            note = "Your model's running cost was not measured yet." + (f" To measure it, {hint}." if hint else "")
        else:
            note = str(totals.get("savings_note") or "").rstrip(".") + "."
            note = note[:1].upper() + note[1:]
    jobs = report["workloads"]
    testing = sum(1 for w in jobs if w["recommendation"] in (NEEDS_FINETUNE, MORE_DATA))
    moving = sum(1 for w in jobs if w["recommendation"] == MOVE)
    staying = sum(1 for w in jobs if w["recommendation"] == KEEP)
    api = totals.get("monthly_api_usd")
    movable = totals.get("movable_api_usd")
    facts = [
        ("api", "API bill", money(api) if api is not None else "not projected", "a month, at API prices"),
        ("own", "Can move", pct(share) if share is not None else "-",
         (f"{money(movable)} a month, " if movable is not None else "") + f"{moving} job{'s' if moving != 1 else ''}"),
        ("gold", "Your model costs", money(owned_total["high"]) if owned_total else
         ("not measured" if not measured else "-"),
         "a month, high end" + (", the always-on GPU" if totals.get("floor_applied") else "")),
        ("test", "Still testing", str(testing), f"{staying} job{'s' if staying != 1 else ''} stay{'s' if staying == 1 else ''} on the API"),
    ]
    cells = "".join(f'<div class="fact f-{cls}"><dt>{e(name)}</dt><dd><b>{e(value)}</b><span>{e(sub)}</span></dd></div>'
                    for cls, name, value, sub in facts)
    legacy = "".join(f'<span class="legacy">{e(n)}: {e(v)}</span>' for n, v in _summary(report))
    if api is not None and movable:
        lede = (f"Of a {money(api)} monthly API bill, <em>{e(money(movable))}</em> can move to your own model. "
                + (f"Running it costs about {e(money(owned_total['high']))}." if owned_total else ""))
    elif api is not None:
        lede = (f"Nothing is ready to move off the {e(money(api))} monthly API bill yet. "
                f"{testing} job{'s are' if testing != 1 else ' is'} still being tested.")
    else:
        lede = "The capture window is too short to project a monthly bill."
    return (
        '<section class="hero" aria-label="Savings"><div class="hero-top"><div>'
        f'<p class="kicker">Saved per month{t}</p>{figure}<p class="hero-note">{e(note)}</p></div>'
        f'<p class="lede">{lede}</p></div>'
        f'<dl class="facts">{cells}</dl>{_bill_bar(report)}{legacy}'
        '</section>'
    )


def _summary(report: Mapping[str, Any]) -> list[tuple[str, str]]:
    from .report import summary_lines

    return summary_lines(report)


def _jobs(report: Mapping[str, Any]) -> str:
    totals = report["totals"]
    t = tag(report)
    rows: list[str] = []
    for w in sorted(report["workloads"], key=lambda w: -(w.get("api_usd_per_month") or 0)):
        rec = w["recommendation"]
        cls, status = STATUS[rec]
        api = w.get("api_usd_per_month")
        owned = w.get("owned_usd_per_month")
        api_text = money(api) if api is not None else w.get("api_note") or "-"
        gap = "not projected" if str(w.get("owned_note", "")).startswith("not projected") else "not measured"
        own_text = money(owned["high"]) if owned else gap
        reqs = w.get("requests_per_month")
        reason = (w.get("reasons") or [""])[0]
        rows.append(
            f'<tr class="job j-{cls}"><th scope="row"><span class="job-name">{e(job_name(w["type"]))}</span>'
            f'<span class="job-reqs">{e(f"{reqs:,.0f} requests a month" if reqs is not None else "")}</span></th>'
            f'<td class="where"><i class="dot d-{cls}"></i>{e(status)}</td>'
            f'<td class="num"><b class="cost-api">{e(api_text)}</b></td>'
            f'<td class="num"><b class="cost-own{"" if owned else " nm"}">{e(own_text)}</b>'
            + ('' if rec == MOVE or not owned else '<small>if moved</small>') + '</td>'
            f'<td class="proof-cell"><p>{e(proof_words(w.get("quality")))}</p>'
            + (f'<p class="why">{e(reason)}</p>' if reason else "") + '</td></tr>'
        )
    owned_total = totals.get("owned_usd_per_month")
    savings = totals.get("conservative_savings_usd")
    saved = money(savings) if savings is not None else ("not measured" if report.get("owned_cost") is None else "no claim")
    sums = [
        ("API bill", money(totals.get("monthly_api_usd")) if totals.get("monthly_api_usd") is not None else "not projected", ""),
        ("Of that, moves to your model", money(totals.get("movable_api_usd")) if totals.get("movable_api_usd") is not None else "-", ""),
        ("Your model, all-in", money(owned_total["high"]) if owned_total else "-", ""),
        ("Saved per month", saved, "grand"),
    ]
    sum_rows = "".join(f'<div class="sum {cls}"><span>{e(a)}</span><b>{e(b)}</b></div>' for a, b, cls in sums)
    return (
        f'<section class="jobs"><header class="sec"><h2>Job by job{t}</h2>'
        '<p>Where each kind of job runs, what it costs, and the proof. Nothing moves without your OK, '
        'and if your model stumbles the request goes to the API.</p></header>'
        '<div class="table-wrap"><table class="jobtable"><thead><tr><th scope="col">Job</th><th scope="col">Where it runs</th>'
        '<th scope="col" class="num">API, a month</th><th scope="col" class="num">Your model, a month</th>'
        f'<th scope="col">Proof</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
        f'<div class="sums">{sum_rows}</div>'
        '<p class="fine">Saved = API bill of the jobs that move, minus your model at the high end of its 95% range. '
        'The always-on GPU is counted once. A job\'s model cost is its token cost only; the GPU floor is in '
        '"Your model, all-in".</p></section>'
    )


def _proof(report: Mapping[str, Any]) -> str:
    t = tag(report)
    items: list[str] = []
    for w in report["workloads"]:
        q = w.get("quality")
        rec = w["recommendation"]
        cls, status = STATUS[rec]
        if not q:
            body = '<p class="fine">No answers from your model for this job yet, so nothing was judged.</p>'
            n_text = "not tested"
        else:
            judged = int(q.get("judged") or 0)
            n_text = f"n = {q.get('n')} · judged {judged}"
            stack = ""
            if judged:
                parts = [("own", q.get("wins") or 0, "better"), ("tie", q.get("ties") or 0, "same"),
                         ("test", q.get("order_disagree") or 0, "judge unsure"), ("api", q.get("losses") or 0, "worse")]
                stack = '<div class="stack">' + "".join(
                    f'<span class="sk-{c}" style="flex:{v}" title="{e(word)}: {v}"></span>' for c, v, word in parts if v
                ) + '</div><div class="stack-key">' + "".join(
                    f'<span><i class="dot d-{c}"></i>{e(word)} <b>{v}</b></span>' for c, v, word in parts
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
                f'<tr><td>{e(name.replace("_", " "))}</td><td class="num">{e(pct(c.get("incumbent")))}</td>'
                f'<td class="num">{e(pct(c.get("candidate")))}</td><td>{"overlap" if c.get("overlap") else "<b>below the API</b>"}</td></tr>'
                for name, c in fmt.items()
            )
            fmt_html = (f'<table class="fmt"><thead><tr><th>Format check</th><th class="num">API</th>'
                        f'<th class="num">Your model</th><th>95% CIs</th></tr></thead><tbody>{fmt_rows}</tbody></table>'
                        ) if fmt_rows else ""
            reasons = "".join(f"<li>{e(r)}</li>" for r in (q.get("reasons") or []))
            body = (f'{stack}<div class="rates">{bars}</div>{_net_axis(q.get("net_ci"))}'
                    f'<p class="fine">{e(judge_note)}</p>{fmt_html}'
                    f'<p class="verdict">Quality verdict <b>{e(q.get("verdict"))}</b></p>'
                    + (f'<ul class="why">{reasons}</ul>' if reasons else ""))
        recs = "".join(f"<li>{e(r)}</li>" for r in w.get("reasons") or [])
        items.append(
            f'<details class="pf"><summary><span class="pf-name">{e(job_name(w["type"]))}</span>'
            f'<span class="pf-n">{e(n_text)}</span><span class="pf-status"><i class="dot d-{cls}"></i>{e(status)}</span>'
            '<span class="chev" aria-hidden="true"></span></summary>'
            f'<div class="pf-body">{body}<div class="pf-rec">Recommendation <b>{e(rec)}</b>'
            f'<ul class="why">{recs}</ul></div></div></details>'
        )
    return (f'<section class="proof"><header class="sec"><h2>The proof, one click deeper{t}</h2>'
            '<p>Open a job to see win, tie and loss with 95% error bars, sample sizes and how often the judge '
            'disagreed with itself.</p></header>' + "".join(items) + '</section>')


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
        f'<section class="ledger"><header class="sec"><h2>How we got these numbers{t}</h2>'
        '<p>Every number is labelled. MEASURED came from real responses. ASSUMED is a number someone chose.</p></header>'
        '<div class="table-wrap"><table class="labels"><thead><tr><th>Number</th><th>Label</th><th>Where it comes from</th>'
        f'</tr></thead><tbody>{rows}</tbody></table></div>'
        f'<h3>What this does not prove</h3><ol class="limits">{limits}</ol>'
        f'<p class="fine rules">{e(rules)}</p></section>'
    )


def render(report: Mapping[str, Any], *, demo_banner: str, rules: str) -> str:
    demo = bool(report.get("demo"))
    kind = mode(report)
    title_prefix = "DEMO " if demo else ("Replay " if kind == "replay" else "")
    cls, words = stamp(report)
    sub = {"demo": "synthetic traffic, fake models", "replay": "public dataset replay, local models, GPU price assumed",
           "live": "your captured traffic, GPU price assumed"}[kind]
    when = str(report.get("generated_at") or "")[:10]
    head = (
        '<header class="top">'
        '<div class="brand"><span class="word">throttle</span><span class="kind">Savings report</span></div>'
        f'<div class="meta"><span class="stamp st-{cls}">{e(words)}</span>'
        f'<span>{e(report.get("candidate"))} vs the API</span><span>{e(sub)}</span><span>{e(when)}</span></div>'
        '</header>'
    )
    footer = (f'<footer><span><span class="word">throttle</span> Generated by <code>throttle {"demo" if demo else "report"}</code> '
              f'at {e(report.get("generated_at"))}, eval {e(report.get("eval_run_id") or "none")}.</span>'
              '<span>Stop renting your AI.</span></footer>')
    return "\n".join([
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f'<title>{title_prefix}Throttle report: {e(report["candidate"])}</title>',
        f'<style>{CSS}</style></head><body><main class="page">',
        head,
        _banner(report, demo_banner),
        _hero(report),
        _jobs(report),
        _proof(report),
        _ledger(report, rules),
        footer,
        '</main></body></html>',
    ]) + "\n"


CSS = """
:root{
--navy:#0a1430;--navy2:#0e1b3d;--navy3:#14244f;--deep:#060d22;
--ivory:#f3eee3;--ivory2:rgba(243,238,227,.74);--muted:rgba(243,238,227,.52);--faint:rgba(243,238,227,.10);--line:rgba(243,238,227,.14);
--gold:#d8b46a;--gold2:#e9cf95;--bronze:#9c7434;
--sage:#93bb9d;--brick:#c96a4c;--mist:#8e9cc4;--tie:rgba(243,238,227,.30);
--serif:"Fraunces","Newsreader","Iowan Old Style","Palatino Linotype",Palatino,"Book Antiqua",Georgia,serif;
--sans:"Hanken Grotesk","Figtree","Avenir Next","Segoe UI",system-ui,-apple-system,"Helvetica Neue",Arial,sans-serif;
--mono:"JetBrains Mono","IBM Plex Mono",ui-monospace,"SF Mono",Menlo,Consolas,monospace;
}
*{box-sizing:border-box}
html{background:var(--deep)}
body{margin:0;color:var(--ivory);font:15.5px/1.6 var(--sans);font-variant-numeric:tabular-nums lining-nums;
background:radial-gradient(1200px 520px at 78% -6%,#18295a 0%,rgba(24,41,90,0) 62%),linear-gradient(180deg,var(--navy2),var(--navy) 38%,var(--deep));
min-height:100vh;-webkit-font-smoothing:antialiased}
.page{max-width:1120px;margin:0 auto;padding:30px 32px 64px}
code{font-family:var(--mono);font-size:.88em;color:var(--ivory2)}
b{font-weight:600}
.word{font-family:var(--serif);font-weight:600;letter-spacing:-.01em}
.top{display:flex;justify-content:space-between;align-items:baseline;gap:18px;flex-wrap:wrap;padding-bottom:18px;border-bottom:1px solid var(--line)}
.brand{display:flex;align-items:baseline;gap:14px}
.brand .word{font-size:30px;color:var(--ivory)}
.brand .kind{font-size:12px;letter-spacing:.18em;text-transform:uppercase;color:var(--muted)}
.meta{display:flex;gap:16px;align-items:baseline;flex-wrap:wrap;font-size:12.5px;color:var(--muted)}
.stamp{font-size:10.5px;font-weight:700;letter-spacing:.16em;padding:4px 9px;border-radius:2px;border:1px solid}
.st-demo{color:var(--gold2);border-color:var(--bronze)}.st-replay{color:var(--gold2);border-color:var(--bronze)}
.st-live{color:var(--sage);border-color:rgba(147,187,157,.5)}
.banner{margin:18px 0 0;padding:12px 16px;font-size:13.5px;color:var(--ivory2);border-left:2px solid var(--gold);background:rgba(216,180,106,.07)}
.banner b{color:var(--gold2)}
.tag{display:inline-block;font:700 9.5px/1 var(--sans);letter-spacing:.16em;padding:4px 7px;border:1px solid var(--bronze);
color:var(--gold2);border-radius:2px;vertical-align:middle;margin-left:10px;text-transform:uppercase}
.legacy{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}

/* hero */
section.hero{margin-top:0;padding:56px 0 10px}
.hero-top{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(0,1fr);gap:30px 56px;align-items:end}
.lede{margin:0 0 6px;font:400 26px/1.35 var(--serif);color:var(--ivory);letter-spacing:-.01em;padding-left:28px;border-left:1px solid var(--line)}
.lede em{font-style:italic;color:var(--gold2)}
.kicker{margin:0;font-size:12px;letter-spacing:.2em;text-transform:uppercase;color:var(--ivory2)}
.figure{margin:10px 0 0;font:500 clamp(68px,12vw,148px)/.95 var(--serif);letter-spacing:-.035em;color:var(--gold);
font-variation-settings:"opsz" 144;font-feature-settings:"lnum","tnum"}
.figure .cur{font-size:.5em;vertical-align:.62em;margin-right:.04em;color:var(--gold2);letter-spacing:0}
.figure .per{font:400 italic 22px/1 var(--serif);letter-spacing:0;color:var(--ivory2);margin-left:16px}
.figure.none{font-size:clamp(44px,7vw,84px);font-style:italic;color:var(--ivory2);letter-spacing:-.02em}
.hero-note{margin:18px 0 0;max-width:620px;color:var(--ivory2);font-size:15px}
.facts{display:grid;grid-template-columns:repeat(4,1fr);margin:42px 0 0;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}
.fact{padding:18px 20px 18px 0;margin:0}
.fact+.fact{padding-left:22px;border-left:1px solid var(--line)}
.fact dt{font-size:11.5px;letter-spacing:.16em;text-transform:uppercase;color:var(--muted)}
.fact dd{margin:6px 0 0}
.fact dd b{display:block;font:500 34px/1.1 var(--serif);letter-spacing:-.015em;color:var(--ivory);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.fact dd span{display:block;font-size:12.5px;color:var(--muted);margin-top:4px}
.f-api dd b{color:#e9a58c}.f-own dd b{color:var(--sage)}.f-gold dd b{color:var(--gold2)}.f-test dd b{color:#b7c2e2}
.billing{margin-top:34px}
.bar-label{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin:0 0 8px}
.bill{display:flex;height:38px;gap:2px;border-radius:2px;overflow:hidden}
.bill .seg{position:relative;display:flex;align-items:center;min-width:3px;padding:0 10px;overflow:hidden}
.bill .seg em{font:italic 500 14px/1 var(--serif);white-space:nowrap;color:var(--navy)}
.g-own{background:var(--sage)}.g-api{background:var(--brick)}.g-gold{background:linear-gradient(90deg,var(--bronze),var(--gold))}
.g-test{background:repeating-linear-gradient(135deg,var(--mist) 0 6px,#7d8bb5 6px 12px)}
.own-bar{height:12px;margin-bottom:6px}
.billing .own-bar{margin-top:0}
.billing .bar-label+.bill+.bar-label{margin-top:16px}
.bill-key{list-style:none;padding:0;margin:16px 0 0;display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:6px 28px;font-size:13.5px}
.bill-key li{display:flex;align-items:center;gap:8px;border-bottom:1px dotted var(--line);padding:4px 0}
.bill-key li span{flex:1;color:var(--ivory2)}.bill-key li b{font-weight:500}
.legend{display:flex;gap:20px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin:14px 0 0}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:8px;vertical-align:1px;flex:none}
.d-own{background:var(--sage)}.d-api{background:var(--brick)}.d-test{background:var(--mist)}.d-tie{background:var(--tie)}.d-gold{background:var(--gold)}

/* sections */
section{margin-top:76px}
.sec{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,420px);gap:12px 40px;align-items:end;margin-bottom:22px;
padding-bottom:16px;border-bottom:1px solid var(--line)}
.sec h2{margin:0;font:400 44px/1.05 var(--serif);letter-spacing:-.02em;color:var(--ivory);font-variation-settings:"opsz" 96}
.sec p{margin:0;color:var(--muted);font-size:14px}
h3{font:400 26px/1.2 var(--serif);margin:44px 0 12px;letter-spacing:-.01em}
.fine{color:var(--muted);font-size:12.5px;margin:10px 0 0}
.table-wrap{overflow-x:auto}

/* job by job */
.jobtable{width:100%;border-collapse:collapse}
.jobtable thead th{font-size:11px;font-weight:600;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);text-align:left;padding:0 14px 12px 0}
.jobtable td,.jobtable tbody th{padding:20px 14px 20px 0;border-top:1px solid var(--line);vertical-align:top;text-align:left;font-weight:400}
.num{text-align:right!important;white-space:nowrap}
.job-name{display:block;font:400 23px/1.15 var(--serif);letter-spacing:-.01em;color:var(--ivory)}
.job-reqs{display:block;font-size:12px;color:var(--muted);margin-top:5px}
.where{white-space:nowrap;font-size:14px;color:var(--ivory2);padding-top:26px!important}
.j-own .where{color:var(--sage)}.j-api .where{color:#e9a58c}.j-test .where{color:#b7c2e2}
.jobtable .num b{font:500 22px/1.2 var(--serif)}
.jobtable td.num{padding-top:22px}
.cost-api{color:#e9a58c}.cost-own{color:var(--gold2)}.cost-own.nm{font:italic 400 15px var(--serif)!important;color:var(--muted)}
.jobtable .num small{display:block;font-size:11.5px;color:var(--muted);margin-top:3px}
.proof-cell{max-width:340px;font-size:14px}
.proof-cell p{margin:0}.proof-cell .why{margin-top:5px;font-size:12.5px;color:var(--muted)}
.sums{margin:6px 0 0 auto;max-width:520px;border-top:1px solid var(--ivory2)}
.sum{display:flex;justify-content:space-between;gap:20px;padding:10px 0;border-bottom:1px solid var(--line);font-size:14.5px;color:var(--ivory2)}
.sum b{font:500 20px/1.2 var(--serif);color:var(--ivory)}
.sum.grand{border-bottom:none;padding-top:16px}
.sum.grand span{color:var(--ivory);font-size:12.5px;letter-spacing:.18em;text-transform:uppercase;align-self:center}
.sum.grand b{font-size:40px;color:var(--gold);letter-spacing:-.02em}

/* proof */
.pf{border-top:1px solid var(--line)}
.pf:last-of-type{border-bottom:1px solid var(--line)}
.pf summary{list-style:none;cursor:pointer;display:flex;align-items:baseline;gap:20px;padding:18px 0}
.pf summary::-webkit-details-marker{display:none}
.pf summary:hover .pf-name{color:var(--gold2)}
.pf-name{font:400 22px/1.2 var(--serif);flex:1}
.pf-n{font-size:12.5px;color:var(--muted)}
.pf-status{font-size:13.5px;color:var(--ivory2);min-width:190px}
.chev{width:8px;height:8px;border-right:1.5px solid var(--ivory2);border-bottom:1.5px solid var(--ivory2);transform:rotate(45deg) translateY(-3px);transition:transform .2s}
.pf[open] .chev{transform:rotate(-135deg)}
.pf-body{padding:4px 0 26px}
.stack{display:flex;height:10px;gap:2px;margin-top:6px}
.sk-own{background:var(--sage)}.sk-tie{background:var(--tie)}.sk-test{background:var(--mist)}.sk-api{background:var(--brick)}
.stack-key{display:flex;gap:22px;flex-wrap:wrap;font-size:13px;color:var(--ivory2);margin-top:10px}
.stack-key b{color:var(--ivory)}
.rates{margin-top:20px}
.rate{display:grid;grid-template-columns:160px 1fr 200px;gap:16px;align-items:center;padding:5px 0}
.rate-name{font-size:13.5px;color:var(--ivory2)}
.rate-track{position:relative;height:8px;background:var(--faint)}
.rate-fill{position:absolute;left:0;top:0;bottom:0}
.f-own.rate-fill{background:var(--sage)}.f-tie.rate-fill{background:var(--tie)}.f-api.rate-fill{background:var(--brick)}
.whisker{position:absolute;top:-5px;bottom:-5px;border-left:1.5px solid var(--gold2);border-right:1.5px solid var(--gold2)}
.whisker:after{content:"";position:absolute;left:0;right:0;top:50%;border-top:1.5px solid var(--gold2)}
.rate-num{font:500 18px/1 var(--serif)}.rate-num small{font:400 11.5px var(--sans);color:var(--muted);margin-left:8px}
.net{margin-top:22px}
.net-axis{position:relative;height:20px;background:linear-gradient(90deg,rgba(201,106,76,.22),rgba(243,238,227,.05) 50%,rgba(147,187,157,.22))}
.net-band{position:absolute;left:47.5%;width:5%;top:0;bottom:0;background:rgba(243,238,227,.10)}
.net-zero{position:absolute;left:50%;top:-4px;bottom:-4px;border-left:1px dashed var(--ivory2)}
.net-ci{position:absolute;top:7px;height:6px;background:var(--gold)}
.net-scale{display:flex;justify-content:space-between;font-size:10.5px;color:var(--muted);margin-top:6px;text-transform:uppercase;letter-spacing:.14em}
.fmt{width:100%;max-width:640px;border-collapse:collapse;font-size:13px;margin-top:16px}
.fmt th,.fmt td{text-align:left;padding:7px 12px 7px 0;border-bottom:1px solid var(--line)}
.fmt th{font-size:10.5px;font-weight:600;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}
.verdict{margin:16px 0 0;font-size:13px;color:var(--muted)}.verdict b{color:var(--ivory);margin-left:6px;letter-spacing:.04em}
.pf-rec{margin-top:16px;padding:14px 18px;background:rgba(243,238,227,.04);border-left:2px solid var(--gold);font-size:13.5px;color:var(--ivory2)}
.pf-rec b{color:var(--gold2);margin-left:6px;letter-spacing:.04em}
ul.why{margin:6px 0 0;padding-left:18px;font-size:13px;color:var(--ivory2)}
ul.why li{margin:3px 0}

/* ledger */
.labels{width:100%;border-collapse:collapse;font-size:13.5px}
.labels th,.labels td{text-align:left;padding:12px 16px 12px 0;border-bottom:1px solid var(--line);vertical-align:top}
.labels th{font-size:10.5px;font-weight:600;letter-spacing:.16em;text-transform:uppercase;color:var(--muted)}
.labels td:first-child{color:var(--ivory);white-space:nowrap}
.labels td:last-child{color:var(--ivory2)}
.lbl{display:inline-block;font-size:10.5px;font-weight:700;letter-spacing:.08em;padding:3px 7px;border:1px solid;border-radius:2px;white-space:nowrap}
.lb-measured{color:var(--sage);border-color:rgba(147,187,157,.45)}.lb-assumed{color:var(--gold2);border-color:var(--bronze)}
.lb-demo{color:var(--gold2);border-color:var(--bronze);background:rgba(216,180,106,.08)}.lb-warn{color:#e9a58c;border-color:rgba(201,106,76,.6)}
.lb-public{color:#b7c2e2;border-color:rgba(142,156,196,.55)}
ol.limits{padding-left:22px;font-size:14px;color:var(--ivory2);max-width:860px}ol.limits li{margin-bottom:8px;padding-left:4px}
ol.limits li::marker{font-family:var(--serif);color:var(--gold)}
.rules{max-width:860px}
footer{margin-top:80px;padding-top:18px;border-top:1px solid var(--line);display:flex;justify-content:space-between;gap:14px;
flex-wrap:wrap;font-size:12px;color:var(--muted)}
footer .word{font-size:16px;color:var(--ivory);margin-right:8px}

@media (max-width:900px){
.hero-top{grid-template-columns:1fr}.lede{padding-left:0;border-left:none;font-size:22px}
.facts{grid-template-columns:1fr 1fr}
.fact:nth-child(3){padding-left:0;border-left:none}
.fact:nth-child(n+3){border-top:1px solid var(--line)}
.sec{grid-template-columns:1fr}
.rate{grid-template-columns:1fr;gap:4px}
.jobtable thead{display:none}
.jobtable,.jobtable tbody,.jobtable tr,.jobtable td,.jobtable tbody th{display:block;width:100%}
.jobtable tr{border-top:1px solid var(--line);padding:18px 0}
.jobtable td,.jobtable tbody th{border:none;padding:4px 0!important;text-align:left!important}
.jobtable td.num{display:inline-block;width:auto;margin-right:22px}
.jobtable td.num:before{display:block;font:600 10.5px var(--sans);letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin-bottom:2px}
.jobtable td.num:nth-of-type(2):before{content:"API, a month"}
.jobtable td.num:nth-of-type(3):before{content:"Your model, a month"}
.proof-cell{max-width:none}
.pf-status{min-width:0}
}
@media (max-width:560px){
.page{padding:20px 16px 48px}
.hero{padding-top:36px}
.facts{grid-template-columns:1fr}
.fact,.fact+.fact{padding:14px 0;border-left:none}
.fact+.fact{border-top:1px solid var(--line)}
.figure .per{display:block;margin:10px 0 0}
.sec h2{font-size:34px}
.pf summary{flex-wrap:wrap;gap:6px 14px}.pf-name{flex-basis:100%}
.labels td:first-child{white-space:normal}
.sums{max-width:none}
.bill .seg em{display:none}
}
@media (prefers-reduced-motion:reduce){*,*:before,*:after{animation:none!important;transition:none!important}}
@media print{
html,body{background:#0a1430;-webkit-print-color-adjust:exact;print-color-adjust:exact}
.page{max-width:none}
.pf,.jobtable tr{break-inside:avoid}
details.pf>.pf-body{display:block}
}
"""
