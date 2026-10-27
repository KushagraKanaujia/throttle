"""Stage 7, REFRESH: should a newly released open model replace the one you own?

``throttle refresh --current LABEL --new-label LABEL --new-url URL --new-model M``

1. For each workload type with a saved eval set for the CURRENT label
   (``<root>/evals/evalset-<type>-<sha12>.json``, newest per type), shadow the
   same exchange ids on the new model. Ids that already have a result for the
   new label are not sent again.
2. Evaluate the new model against the API model's captured answers on those
   ids (the same comparison as ``throttle eval``; that run is saved too).
3. Judge the new model directly against the current owned model on the same
   items: pairwise, both orders, order disagreement is a TIE, and the same
   automatic checks and verdict rule as ``throttle eval`` (with the current
   owned model in the incumbent's place). A failed answer loses to an
   answered one without a judge call; two failed answers are a TIE.

Verdict per type
----------------
* STAY: the new model is WORSE than the API model, or WORSE than the
  current owned model.
* UPGRADE: the new model is NO MATERIAL DIFFERENCE or BETTER vs the API
  model, AND NO MATERIAL DIFFERENCE or BETTER vs the current owned model.
  "Not worse" has to be shown; an INCONCLUSIVE comparison is not enough.
* NEEDS MORE DATA: anything else.

Refresh never measures cost. Every UPGRADE carries the note
"measure cost with throttle check before switching".

Output: ``<root>/evals/refresh-<run_id>.json``
----------------------------------------------
::

    {
      "schema": "throttle.own.refresh/1",
      "run_id": str, "ts": ISO-8601 UTC,
      "current": str, "new": str, "new_model": str,
      "judge": {"provider": str, "model": str},
      "margin": float, "min_n": int,
      "judge_calls": int, "judge_stopped": str | null,
      "shadow": {"done": int, "ok": int, "already_had": int, "errors": {kind: int}},
      "eval_run_id": str, "eval_path": str,      # the new-vs-API `throttle eval` run
      "types": {
        "<type>": {
          "evalset_sha256": str,                 # the saved set for CURRENT that was rerun
          "evalset_path": str,
          "exchange_ids": int,                   # ids in that saved set
          "compared": int,                       # ids with answers from both owned models
          "missing_ids": [str],                  # saved ids with no usable new answer
          "vs_api": {"verdict": str, "n": int, "wins": int, "ties": int, "losses": int,
                     "net_ci": {...} | null, "new_evalset_sha256": str | null},
          "vs_current": {"verdict": str, "n": int, "wins": int, "ties": int, "losses": int,
                         "judge_errors": int, "judge_skipped": int,
                         "net_ci": {...} | null,
                         "checks": {"<name>": {"current": wilson, "new": wilson, "unsupported": int}},
                         "reasons": [str]},
          "verdict": "UPGRADE" | "STAY" | "NEEDS MORE DATA",
          "better_than_current": bool,
          "cost": {"measured": false},
          "notes": [str]
        }
      }
    }

``throttle refresh --watch-hf ORG/MODEL`` only asks the Hugging Face API for a
newer revision of that model, or newer models from the same org with the same
name stem, and prints what it found. It is the only network call refresh makes
on its own, and only when the flag is passed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import httpx

from . import evaluate as ev
from . import judge as judge_mod
from . import records
from .shadow import api_key_from_env, chat_url, done_ids, run_backfill, select_backfill
from .. import statistics as stats

SCHEMA_ID = "throttle.own.refresh/1"
UPGRADE = "UPGRADE"
STAY = "STAY"
NEEDS_MORE_DATA = "NEEDS MORE DATA"
COST_NOTE = "measure cost with throttle check before switching"
SHOWN_NOT_WORSE = (ev.NO_DIFF, ev.BETTER)
HF_API = "https://huggingface.co/api/models"


class NoEvalSet(Exception):
    """No saved eval set for the current label."""


# ---------------------------------------------------------------------------
# Saved eval sets
# ---------------------------------------------------------------------------


def find_evalsets(root: Path, current: str, types: Sequence[str] | None = None) -> dict[str, dict[str, Any]]:
    """Newest saved eval set per type for ``current``: {type: {sha256, path, ids}}."""

    found: dict[str, tuple[str, dict[str, Any]]] = {}
    for path in sorted((root / "evals").glob("evalset-*.json")):
        try:
            kind, candidate, ids = ev.load_evalset(path)
            data = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, KeyError, OSError):
            continue
        if candidate != current or (types and kind not in types):
            continue
        created = str(data.get("created") or "")
        if kind not in found or created >= found[kind][0]:
            found[kind] = (created, {"sha256": data["sha256"], "path": str(path), "ids": ids})
    if not found:
        wanted = f" for types {', '.join(types)}" if types else ""
        raise NoEvalSet(
            f"no saved eval set for {current}{wanted} in {root / 'evals'}. "
            f"Run `throttle eval --candidate {current}` first."
        )
    return {kind: found[kind][1] for kind in sorted(found)}


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def decide(vs_api: str, vs_current: str) -> tuple[str, bool, list[str]]:
    """(verdict, better_than_current, notes) from the two per-type verdicts."""

    if vs_api == ev.WORSE or vs_current == ev.WORSE:
        notes = []
        if vs_api == ev.WORSE:
            notes.append("new model is worse than the API model")
        if vs_current == ev.WORSE:
            notes.append("new model is worse than the current owned model")
        return STAY, False, notes
    if vs_api in SHOWN_NOT_WORSE and vs_current in SHOWN_NOT_WORSE:
        better = vs_current == ev.BETTER
        notes = ["new model is better than the current owned model"] if better else []
        return UPGRADE, better, notes + [COST_NOTE]
    return NEEDS_MORE_DATA, False, [f"vs API: {vs_api}; vs current: {vs_current}"]


# ---------------------------------------------------------------------------
# Current vs new
# ---------------------------------------------------------------------------


def _side(pair: ev.Pair) -> ev.Side:
    s = pair.shadow
    return ev.Side(s.output_text, s.tool_calls, s.finish_reason, failed=not s.ok)


async def _judge_current_vs_new(
    matched: Mapping[str, Sequence[tuple[ev.Pair, ev.Pair]]],
    *,
    config: judge_mod.JudgeConfig,
    budget: judge_mod.JudgeBudget,
    concurrency: int,
    transport: httpx.AsyncBaseTransport | None,
) -> tuple[dict[str, ev.TypeOutcome], str | None]:
    results = {k: ev.TypeOutcome() for k in matched}
    stopped: list[str] = []
    gate = asyncio.Semaphore(max(1, concurrency))
    async with httpx.AsyncClient(transport=transport) as client:

        async def one(kind: str, cur: ev.Pair, new: ev.Pair) -> None:
            if not (cur.shadow.ok and new.shadow.ok):
                if cur.shadow.ok:
                    results[kind].outcomes.append(judge_mod.LOSS)
                elif new.shadow.ok:
                    results[kind].outcomes.append(judge_mod.WIN)
                else:
                    results[kind].outcomes.append(judge_mod.TIE)
                return
            async with gate:
                if stopped:
                    results[kind].skipped += 1
                    return
                try:
                    outcome = await judge_mod.judge_both_orders(
                        cur.norm.messages,
                        judge_mod.render_answer(cur.shadow.output_text, cur.shadow.tool_calls),
                        judge_mod.render_answer(new.shadow.output_text, new.shadow.tool_calls),
                        client=client, config=config, rubric=judge_mod.rubric_for(kind), budget=budget,
                    )
                except judge_mod.BudgetExhausted as exc:
                    if not stopped:
                        stopped.append(str(exc))
                    results[kind].skipped += 1
                    return
            results[kind].outcomes.append(outcome)

        await asyncio.gather(*(one(k, c, n) for k, items in matched.items() for c, n in items))
    return results, (stopped[0] if stopped else None)


def summarize_vs_current(
    kind: str,
    items: Sequence[tuple[ev.Pair, ev.Pair]],
    outcome: ev.TypeOutcome,
    *,
    margin: float,
    min_n: int,
    seed: int,
) -> dict[str, Any]:
    cur_checks: dict[str, list[bool | None]] = defaultdict(list)
    new_checks: dict[str, list[bool | None]] = defaultdict(list)
    for cur, new in items:
        api_text = cur.norm.output_text
        for name, value in ev.run_checks(cur.norm, kind, _side(cur), api_text).items():
            cur_checks[name].append(value)
        for name, value in ev.run_checks(new.norm, kind, _side(new), api_text).items():
            new_checks[name].append(value)
    checks: dict[str, Any] = {}
    for name in ev.CHECK_NAMES:
        if name not in cur_checks:
            continue
        cur_ci, unsupported = ev._wilson_pair(cur_checks[name])
        new_ci, _ = ev._wilson_pair(new_checks[name])
        checks[name] = {"current": cur_ci, "new": new_ci, "unsupported": unsupported}
    counts = Counter(outcome.outcomes)
    wins, ties, losses, errors = (counts[judge_mod.WIN], counts[judge_mod.TIE],
                                  counts[judge_mod.LOSS], counts[judge_mod.ERROR])
    judged = wins + ties + losses
    net_ci = stats.paired_bootstrap_interval_95([1] * wins + [0] * ties + [-1] * losses, seed=seed)
    as_eval = {k: {"incumbent": v["current"], "candidate": v["new"]} for k, v in checks.items()}
    verdict, reasons = ev.decide(
        judged=True, n=judged, min_n=min_n, margin=margin, net_ci=net_ci, checks=as_eval,
        judge_errors=errors, judge_attempts=judged + errors,
    )
    reasons = [r.replace("API model's", "current model's") for r in reasons]
    if outcome.skipped:
        reasons.append(f"{outcome.skipped} items not judged: judge budget ran out")
    return {
        "verdict": verdict, "n": judged, "wins": wins, "ties": ties, "losses": losses,
        "judge_errors": errors, "judge_skipped": outcome.skipped,
        "net_ci": net_ci, "checks": checks, "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def run_refresh(
    root: Path,
    *,
    current: str,
    new_label: str,
    new_url: str,
    new_model: str,
    judge_config: judge_mod.JudgeConfig,
    api_key: str | None = None,
    budget: judge_mod.JudgeBudget | None = None,
    types: Sequence[str] | None = None,
    margin: float = 0.05,
    min_n: int = 30,
    concurrency: int = 4,
    judge_concurrency: int = 4,
    timeout: float = 120.0,
    candidate_transport: httpx.AsyncBaseTransport | None = None,
    judge_transport: httpx.AsyncBaseTransport | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run the refresh and write ``<root>/evals/refresh-<run_id>.json``."""

    sets = find_evalsets(root, current, types)
    only_ids = {kind: s["ids"] for kind, s in sets.items()}
    all_ids = {i for ids in only_ids.values() for i in ids}

    skip = done_ids(new_label, root)
    selection = select_backfill(
        records.iter_exchanges(root), skip=skip, only_ids=all_ids,
        per_type=len(all_ids), max_requests=len(all_ids),
        classify=lambda exchange, norm: next(k for k, ids in only_ids.items() if exchange.id in ids),
    )
    shadow_summary = {"done": 0, "ok": 0, "errors": {}}
    if selection:
        shadow_summary = asyncio.run(run_backfill(
            selection, url=new_url, model=new_model, label=new_label, root=root, api_key=api_key,
            concurrency=concurrency, timeout=timeout, transport=candidate_transport,
        ))

    budget = budget or judge_mod.JudgeBudget(max_calls=1200)
    api_doc = ev.run_eval(
        root, new_label, judge_config=judge_config, budget=budget, margin=margin, min_n=min_n,
        only_ids=only_ids, judge_concurrency=judge_concurrency, transport=judge_transport,
    )

    cur_sets = ev.build_eval_sets(root, current, only_ids=only_ids)
    new_sets = ev.build_eval_sets(root, new_label, only_ids=only_ids)
    matched: dict[str, list[tuple[ev.Pair, ev.Pair]]] = {}
    missing: dict[str, list[str]] = {}
    for kind, saved in sets.items():
        cur_by_id = {p.exchange.id: p for p in cur_sets.get(kind, [])}
        new_by_id = {p.exchange.id: p for p in new_sets.get(kind, [])}
        matched[kind] = [(cur_by_id[i], new_by_id[i]) for i in saved["ids"] if i in cur_by_id and i in new_by_id]
        missing[kind] = [i for i in saved["ids"] if i not in new_by_id]

    outcomes, stopped = asyncio.run(_judge_current_vs_new(
        matched, config=judge_config, budget=budget, concurrency=judge_concurrency, transport=judge_transport,
    ))
    stopped = stopped or api_doc.get("judge_stopped")

    type_docs: dict[str, Any] = {}
    for kind, saved in sets.items():
        api_t = api_doc["types"].get(kind)
        vs_api = {
            "verdict": api_t["verdict"] if api_t else ev.INSUFFICIENT,
            "n": api_t["judged"] if api_t else 0,
            "wins": api_t["wins"] if api_t else 0,
            "ties": api_t["ties"] if api_t else 0,
            "losses": api_t["losses"] if api_t else 0,
            "net_ci": api_t["net_ci"] if api_t else None,
            "new_evalset_sha256": api_t["evalset_sha256"] if api_t else None,
        }
        vs_current = summarize_vs_current(
            kind, matched[kind], outcomes[kind], margin=margin, min_n=min_n, seed=int(saved["sha256"][:8], 16),
        )
        verdict, better, notes = decide(vs_api["verdict"], vs_current["verdict"])
        if missing[kind]:
            notes.append(f"{len(missing[kind])} of {len(saved['ids'])} saved items have no usable new answer")
        type_docs[kind] = {
            "evalset_sha256": saved["sha256"],
            "evalset_path": saved["path"],
            "exchange_ids": len(saved["ids"]),
            "compared": len(matched[kind]),
            "missing_ids": missing[kind],
            "vs_api": vs_api,
            "vs_current": vs_current,
            "verdict": verdict,
            "better_than_current": better,
            "cost": {"measured": False},
            "notes": notes,
        }

    ts = records.utc_now_iso()
    run_id = run_id or f"{ts[:19].replace(':', '').replace('-', '')}-{records.slug(new_label)}-{uuid.uuid4().hex[:6]}"
    document = {
        "schema": SCHEMA_ID,
        "run_id": run_id,
        "ts": ts,
        "current": current,
        "new": new_label,
        "new_model": new_model,
        "judge": {"provider": judge_config.provider, "model": judge_config.model},
        "margin": margin,
        "min_n": min_n,
        "judge_calls": budget.calls,
        "judge_stopped": stopped,
        "shadow": {
            "done": shadow_summary["done"], "ok": shadow_summary["ok"],
            "already_had": len(all_ids & skip), "errors": shadow_summary["errors"],
        },
        "eval_run_id": api_doc["run_id"],
        "eval_path": api_doc["path"],
        "types": type_docs,
    }
    path = root / "evals" / f"refresh-{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    document["path"] = str(path)
    return document


# ---------------------------------------------------------------------------
# --watch-hf
# ---------------------------------------------------------------------------


def name_stem(model: str) -> str:
    """``Qwen2.5-72B-Instruct`` -> ``Qwen2.5``; ``Llama-3.1-8B`` -> ``Llama-3.1``."""

    parts = model.split("-")
    stem: list[str] = []
    for part in parts:
        if re.fullmatch(r"\d+(\.\d+)?[BbMm]", part) or (stem and part.lower() in {"instruct", "chat", "base", "it"}):
            break
        stem.append(part)
    return "-".join(stem) or model


def watch_hf(repo: str, *, transport: httpx.BaseTransport | None = None) -> dict[str, Any]:
    """The model's current revision and same-org models with the same name stem
    that were modified after it. One or two GETs to the public Hugging Face API."""

    org, _, name = repo.partition("/")
    if not org or not name:
        raise ValueError(f"--watch-hf {repo!r}: use ORG/MODEL")
    with httpx.Client(transport=transport, timeout=30.0) as client:
        info = client.get(f"{HF_API}/{repo}")
        info.raise_for_status()
        model = info.json()
        listing = client.get(HF_API, params={
            "author": org, "search": name_stem(name), "sort": "lastModified", "direction": "-1", "limit": "20",
        })
        listing.raise_for_status()
        siblings = listing.json()
    modified = str(model.get("lastModified") or "")
    newer = [
        {"id": m.get("id") or m.get("modelId"), "lastModified": m.get("lastModified")}
        for m in siblings if isinstance(m, Mapping)
        and (m.get("id") or m.get("modelId")) != repo and str(m.get("lastModified") or "") > modified
    ]
    return {"model": repo, "sha": model.get("sha"), "lastModified": modified or None, "newer": newer}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "refresh",
        help="test a new open model against the one you own, on the same saved eval sets",
        description=(
            "Rerun the saved eval sets of your current owned model on a new open model, "
            "compare it with the API model and with your current model, and say per "
            "workload type whether to UPGRADE, STAY or get more data."
        ),
    )
    parser.add_argument("--current", default=None, help="label of the model you run now (as used with `throttle eval`)")
    parser.add_argument("--new-label", default=None, help="name for the new model, e.g. qwen3-72b@vllm")
    parser.add_argument("--new-url", default=None, help="new model's server, e.g. http://gpu-box:8000/v1")
    parser.add_argument("--new-model", default=None, help="model name the new server expects")
    parser.add_argument("--api-key-env", default=None, metavar="VAR",
                        help="environment variable holding the new server's API key, if it needs one")
    parser.add_argument("--judge-url", default=None, help="judge endpoint (default: the provider's public API)")
    parser.add_argument("--judge-model", default=None, help="judge model name")
    parser.add_argument("--judge-provider", choices=judge_mod.PROVIDERS, default=records.PROVIDER_OPENAI)
    parser.add_argument("--judge-key-env", default=None, metavar="VAR",
                        help="environment variable holding the judge API key (never printed)")
    parser.add_argument("--max-judge-calls", type=int, default=1200,
                        help="hard cap on judge calls across both comparisons (default 1200)")
    parser.add_argument("--judge-max-usd", type=float, default=None, help="hard cap on judge spend, when the price is known")
    parser.add_argument("--judge-concurrency", type=int, default=4, help="judge calls in flight (default 4)")
    parser.add_argument("--concurrency", type=int, default=4, help="new-model requests in flight (default 4)")
    parser.add_argument("--timeout", type=float, default=120.0, help="seconds per new-model request (default 120)")
    parser.add_argument("--types", default=None, help="only these workload types, comma separated")
    parser.add_argument("--margin", type=float, default=0.05, help="material difference in win minus loss (default 0.05)")
    parser.add_argument("--min-n", type=int, default=30, help="minimum judged items per type (default 30)")
    parser.add_argument("--watch-hf", default=None, metavar="ORG/MODEL",
                        help="only check Hugging Face for a newer revision or sibling model, then exit")
    parser.add_argument("--dir", type=Path, default=None, help="data directory (default ~/.throttle/own)")
    parser.add_argument("--json", action="store_true", help="print the result JSON")


def _wtl(block: Mapping[str, Any]) -> str:
    n = block["n"]
    if not n:
        return "-"
    return f"{block['wins'] / n:.0%}/{block['ties'] / n:.0%}/{block['losses'] / n:.0%}"


def render_table(document: Mapping[str, Any]) -> str:
    from .. import style as style_mod

    s = style_mod.get()
    header = f"{'type':<15}{'n':>4}  {'vs API':<24}{'vs current':<24}verdict"
    rows: list[Any] = [s.dim(header)]
    for kind, t in document["types"].items():
        api, cur = t["vs_api"], t["vs_current"]
        rows.append(f"{kind[:14]:<15}{t['compared']:>4}  {api['verdict'][:23]:<24}{cur['verdict'][:23]:<24}"
                    f"{s.verdict(t['verdict'])}")
    rows.append("")
    for kind, t in document["types"].items():
        rows.append((f"{kind}: same test as before, eval set sha256 {t['evalset_sha256'][:12]} "
                     f"({t['exchange_ids']} items); win/tie/loss vs current {_wtl(t['vs_current'])}",
                     (style_mod.DIM,), "  "))
        for note in t["notes"]:
            rows.append((f"{kind}: {note}", (style_mod.DIM,), "  "))
    judge = document["judge"]
    rows.append((f"Judge: {judge['provider']} {judge['model']}, {document['judge_calls']} calls, both orders.",
                 (style_mod.DIM,)))
    if document.get("judge_stopped"):
        rows.append((f"Judging stopped early: {document['judge_stopped']}.", (style_mod.AMBER,)))
    rows.append((f"Saved {document.get('path', '')}", (style_mod.DIM,)))
    return s.panel(rows, title=f"Refresh: {document['new']} vs {document['current']} (MEASURED)")


def _handle_watch(args: argparse.Namespace, transport: httpx.BaseTransport | None) -> int:
    try:
        found = watch_hf(args.watch_hf, transport=transport)
    except (ValueError, httpx.HTTPError) as exc:
        print(f"throttle refresh: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(found, indent=2, sort_keys=True))
        return 0
    print(f"{found['model']}: revision {str(found['sha'] or '?')[:12]}, last modified {found['lastModified'] or '?'}")
    if found["newer"]:
        print("Newer models from the same org:")
        for m in found["newer"]:
            print(f"  {m['id']}  ({m['lastModified']})")
        print("To test one: serve it, then run `throttle refresh --current ... --new-label ...`.")
    else:
        print("No newer sibling models found.")
    return 0


def handle(args: argparse.Namespace, *, transport: httpx.AsyncBaseTransport | None = None,
           judge_transport: httpx.AsyncBaseTransport | None = None,
           hf_transport: httpx.BaseTransport | None = None) -> int:
    if args.watch_hf:
        return _handle_watch(args, hf_transport)
    root = args.dir or records.default_root()
    needed = [f for f in ("current", "new_label", "new_url", "new_model") if not getattr(args, f)]
    if needed:
        print("throttle refresh: pass " + ", ".join("--" + f.replace("_", "-") for f in needed)
              + " (or --watch-hf ORG/MODEL)", file=sys.stderr)
        return 2
    if not args.judge_model:
        print("throttle refresh: pass --judge-model (and --judge-url / --judge-key-env as needed)", file=sys.stderr)
        return 2
    try:
        judge_key = api_key_from_env(args.judge_key_env)
        api_key = api_key_from_env(args.api_key_env)
    except ValueError as exc:
        print(f"throttle refresh: {exc}", file=sys.stderr)
        return 2
    types = [t.strip() for t in args.types.split(",") if t.strip()] if args.types else None
    try:
        find_evalsets(root, args.current, types)
    except NoEvalSet as exc:
        print(f"throttle refresh: {exc}", file=sys.stderr)
        return 2
    url = args.judge_url or judge_mod.DEFAULT_URLS[args.judge_provider]
    if args.judge_provider == records.PROVIDER_OPENAI and args.judge_url:
        url = chat_url(args.judge_url)
    config = judge_mod.JudgeConfig(provider=args.judge_provider, model=args.judge_model, url=url, api_key=judge_key)
    price = judge_mod.pricing_for(args.judge_provider, args.judge_model) if args.judge_max_usd is not None else None
    if args.judge_max_usd is not None and price is None:
        print(f"Note: no known price for {args.judge_model}; --judge-max-usd is not enforced, "
              f"only --max-judge-calls {args.max_judge_calls}.", file=sys.stderr)
    budget = judge_mod.JudgeBudget(max_calls=max(0, args.max_judge_calls), max_usd=args.judge_max_usd, price=price)
    document = run_refresh(
        root, current=args.current, new_label=args.new_label, new_url=args.new_url, new_model=args.new_model,
        judge_config=config, api_key=api_key, budget=budget, types=types, margin=args.margin, min_n=args.min_n,
        concurrency=args.concurrency, judge_concurrency=args.judge_concurrency, timeout=args.timeout,
        candidate_transport=transport, judge_transport=judge_transport,
    )
    if args.json:
        print(json.dumps(document, indent=2, sort_keys=True))
    else:
        print(render_table(document))
    return 0
