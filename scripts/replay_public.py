#!/usr/bin/env python3
"""Replay real public chat requests through the own-intelligence loop, on
local models, and write a Meter report.

Nothing here is a customer. The requests are real people's first messages
from a public dataset (WildChat-1M, ODC-BY); the "API model" and the
candidate are small local models served by Ollama; the GPU price is assumed.
It proves the machinery on real prompts, not frontier-model quality parity.

Steps (each one resumable; re-running skips work that is already on disk):

  prepare  raw WildChat rows (JSON pages from the Hugging Face datasets
           server) -> prompts.jsonl: English, non-toxic, first user turn,
           20..2500 characters, de-duplicated, PII-masked with Throttle's
           own redaction, seeded sample. Country, hashed IP and headers are
           dropped and never written.
  capture  each prompt goes through the real ``throttle capture`` app to the
           stand-in API model on Ollama. Exchanges are stamped on a replay
           clock spread evenly over 30 days so monthly figures are
           "per replayed request x traffic scale" (both stated in the report).
  shadow   the real backfill replays every request against each candidate.
  eval     the real evaluator with a local pairwise judge, both orders.
  check    ``throttle check`` measures the candidate's tokens per second on
           this machine; the GPU $/hour is ASSUMED.
  report   the real report, with the replay's source and labels attached.

Usage (from the repo root, with Ollama running):

  python -I scripts/replay_public.py prepare --raw DIR --out WORK/prompts.jsonl --n 500
  python -I scripts/replay_public.py run --prompts WORK/prompts.jsonl --work WORK --out OUT

Read the raw download with ``python -I`` from a different directory than the
data: it is untrusted input and is only ever parsed as JSON.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

DATASET = "allenai/WildChat-1M"
DATASET_URL = "https://huggingface.co/datasets/allenai/WildChat-1M"
DATASET_LICENSE = "ODC-BY 1.0 (Open Data Commons Attribution License)"
OLLAMA = "http://localhost:11434"
API_STAND_IN = "llama3.2:3b"
PRICE_AS = "gpt-4.1-mini"
CANDIDATES = ["qwen2.5:0.5b"]
JUDGE = "llama3.2:3b-instruct-q8_0"
MAX_TOKENS = 350
MIN_CHARS, MAX_CHARS = 20, 2500
WINDOW_DAYS = 30
DEFAULT_SCALE = 6000
GPU_HOURLY = 0.80
SEED = 20261028


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------


def prepare(raw: Path, out: Path, n: int, seed: int) -> int:
    from throttle.own.records import REDACTION_MASK
    from throttle.own.redact import redact_text

    rows: list[dict[str, Any]] = []
    for path in sorted(glob.glob(str(raw / "rows_*.json"))):
        try:
            page = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for item in page.get("rows") or []:
            row = item.get("row") if isinstance(item, dict) else None
            if isinstance(row, dict):
                rows.append(row)
    seen: set[str] = set()
    pool: list[dict[str, str]] = []
    for row in rows:
        if row.get("language") != "English" or row.get("toxic") is not False:
            continue
        turns = row.get("conversation") or []
        first = next((t for t in turns if isinstance(t, dict) and t.get("role") == "user"), None)
        text = first.get("content") if first else None
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not MIN_CHARS <= len(text) <= MAX_CHARS:
            continue
        key = re.sub(r"\s+", " ", text.lower())[:150]
        if key in seen:
            continue
        seen.add(key)
        pool.append({"id": hashlib.sha256(str(row.get("conversation_hash")).encode()).hexdigest()[:16],
                     "text": redact_text(text, REDACTION_MASK)})
    pool.sort(key=lambda p: p["id"])
    random.Random(seed).shuffle(pool)
    picked = pool[:n]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for p in picked:
            handle.write(json.dumps(p, ensure_ascii=False) + "\n")
    meta = {"dataset": DATASET, "license": DATASET_LICENSE, "rows_read": len(rows), "eligible_unique": len(pool),
            "picked": len(picked), "seed": seed,
            "filters": f"language=English, toxic=false, first user turn, {MIN_CHARS}..{MAX_CHARS} chars, "
                       "de-duplicated on the first 150 characters, PII masked by throttle.own.redact"}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    log(f"prepare: {len(rows)} rows read, {len(pool)} eligible, wrote {len(picked)} to {out}")
    return len(picked)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def _load_prompts(path: Path) -> list[dict[str, str]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _clock(index: int, total: int) -> str:
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    offset = WINDOW_DAYS * 86400 * index / max(1, total - 1)
    return (start + timedelta(seconds=offset)).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _register_texts(prompts: list[dict[str, str]]) -> dict[str, str]:
    from throttle.own.redact import redact_body

    out = {}
    for p in prompts:
        messages = [{"role": "user", "content": p["text"]}]
        out[json.dumps(redact_body(messages, "mask", b"throttle-public-replay"))] = p["id"]
    return out


def shadow(root: Path, label: str, model: str, n: int, concurrency: int) -> None:
    from throttle.own import records
    from throttle.own import shadow as shadow_mod

    skip = shadow_mod.done_ids(label, root)
    selection = shadow_mod.select_backfill(records.iter_exchanges(root), skip=skip, per_type=n,
                                           max_requests=n, seed=SEED)
    log(f"shadow {label}: {len(skip)} done, {len(selection)} to go")
    if selection:
        summary = asyncio.run(shadow_mod.run_backfill(
            selection, url=f"{OLLAMA}/v1", model=model, label=label, root=root,
            concurrency=concurrency, timeout=600, progress=log))
        log(f"shadow {label}: {summary['ok']}/{summary['done']} ok, errors {summary['errors']}")


def evaluate(root: Path, label: str, n: int, concurrency: int) -> dict[str, Any]:
    from throttle.own import evaluate as ev
    from throttle.own import judge as judge_mod
    from throttle.own import records

    run_id = f"replay-eval-{records.slug(label)}"
    path = root / "evals" / f"{run_id}.json"
    if path.exists():
        log(f"eval {label}: reusing {path}")
        return json.loads(path.read_text(encoding="utf-8"))
    config = judge_mod.JudgeConfig(provider=records.PROVIDER_OPENAI, model=JUDGE,
                                   url=f"{OLLAMA}/v1/chat/completions", max_tokens=400, timeout=900)
    log(f"eval {label}: judging with {JUDGE}, both orders")
    return ev.run_eval(root, label, judge_config=config, budget=judge_mod.JudgeBudget(max_calls=4 * n + 20),
                       judge_concurrency=concurrency, run_id=run_id)


def check(root: Path, model: str, prompts: list[dict[str, str]], gpu_hourly: float) -> None:
    check_dir = root / "checks"
    if check_dir.exists() and any(model in p.read_text(encoding="utf-8") for p in check_dir.glob("*.jsonl")):
        log(f"check {model}: reusing earlier record")
        return
    prompt_file = root / "check-prompts.jsonl"
    prompt_file.write_text("".join(json.dumps({"prompt": p["text"]}) + "\n" for p in prompts[:40]), encoding="utf-8")
    cmd = [sys.executable, "-m", "throttle.cli", "check", "--url", f"{OLLAMA}/v1", "--model", model,
           "--gpu-hourly-rate", str(gpu_hourly), "--gpus", "1", "--metric", "total", "--no-metrics",
           "--blocks", "5", "--requests-per-block", "8", "--concurrency", "4", "--max-tokens", str(MAX_TOKENS),
           "--prompts", str(prompt_file), "--history-dir", str(check_dir), "--no-calibrate",
           "--max-elapsed-seconds", "1500", "--max-estimated-spend", "5", "--request-timeout", "600",
           "--label", f"{model}@ollama-m3pro", "--config", "hardware=Apple M3 Pro (Ollama)"]
    log("check: " + " ".join(cmd))
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    subprocess.run(cmd, check=True, env=env)


def source_block(meta: dict[str, Any], scale: int, gpu_hourly: float, candidates: list[str]) -> dict[str, Any]:
    return {
        "kind": "public_replay",
        "title": "Public dataset replay",
        "headline_label": "public dataset replay, local models, GPU price assumed",
        "dataset": DATASET, "dataset_url": DATASET_URL, "license": DATASET_LICENSE,
        "rows_read": meta.get("rows_read"), "requests": meta.get("picked"), "filters": meta.get("filters"),
        "api_stand_in": API_STAND_IN, "priced_as": PRICE_AS, "candidates": candidates, "judge": JUDGE,
        "hardware": "Apple M3 Pro laptop, Ollama", "max_tokens": MAX_TOKENS,
        "traffic_scale": scale, "replay_clock_days": WINDOW_DAYS, "gpu_hourly_rate_usd": gpu_hourly,
        "caveats": [
            f"The 'API model' is {API_STAND_IN} running locally, priced at {PRICE_AS}'s list price. "
            "Small local models stand in for frontier API models, so this proves the loop on real prompts, "
            "not GPT-level quality parity.",
            f"Throughput was measured on a laptop (Apple M3 Pro, Ollama); the ${gpu_hourly:.2f}/hour GPU price is "
            "ASSUMED. A data-center GPU running vLLM usually serves many more tokens per hour.",
            f"Volume is ASSUMED: each replayed request stands for {scale:,} requests a month.",
            f"Answers were capped at {MAX_TOKENS} tokens on both sides; temperature 0.",
            "The judge is a 3B local model; its verdicts are noisier than a frontier judge's.",
        ],
    }


def build(root: Path, out: Path, label: str, scale: int, gpu_hourly: float, source: dict[str, Any],
          prices_path: Path) -> dict[str, Any]:
    from throttle.own import pricing as pricing_mod
    from throttle.own import records
    from throttle.own import report as report_mod

    eval_doc = report_mod.load_eval(root, label)
    models = [m for m in ((eval_doc or {}).get("candidate_model"), label) if m]
    record = report_mod.load_check(root / "checks", models)
    exchanges = list(records.iter_exchanges(root))
    source = {**source, "requests": len(exchanges)}
    report = report_mod.build_report(
        exchanges, pricing_mod.load_prices(prices_path), candidate=label,
        eval_doc=eval_doc, check_record=record, gpu_hourly_rate=gpu_hourly, gpus=1,
        traffic_scale=scale, source=source,
    )
    report_mod.write_outputs(report, out)
    log(f"report: {out / 'report.html'}")
    return report


def run(args: argparse.Namespace) -> int:
    work, out = args.work.expanduser(), args.out.expanduser()
    root = work / "own"
    root.mkdir(parents=True, exist_ok=True)
    os.environ["THROTTLE_OWN_DIR"] = str(root)
    prompts = _load_prompts(args.prompts.expanduser())
    meta_path = args.prompts.expanduser().with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    prices_path = work / "prices-replay.json"
    prices_path.write_text(json.dumps({"prices": [{
        "provider": "openai", "model": API_STAND_IN, "input_per_m": 0.40, "cached_input_per_m": 0.10,
        "output_per_m": 1.60, "as_of": "2026-10-27",
        "source": f"{PRICE_AS} standard list price (bundled table, 2026-10-27) applied to the local {API_STAND_IN} stand-in",
    }]}, indent=1) + "\n", encoding="utf-8")
    n = min(len(prompts), args.max_requests) if args.max_requests else len(prompts)
    if "capture" in args.steps:
        _capture_with_ids(root, prompts, args.concurrency)
    for model in args.candidates:
        label = f"{model}@ollama"
        if "shadow" in args.steps:
            shadow(root, label, model, n, args.concurrency)
        if "eval" in args.steps:
            evaluate(root, label, n, args.concurrency)
        if "check" in args.steps:
            check(root, model, prompts, args.gpu_hourly)
        if "report" in args.steps:
            target = out if model == args.candidates[0] else out / records_slug(label)
            build(root, target, label, args.scale, args.gpu_hourly,
                  source_block(meta, args.scale, args.gpu_hourly, args.candidates), prices_path)
    return 0


def records_slug(label: str) -> str:
    from throttle.own import records

    return records.slug(label)


def _capture_with_ids(root: Path, prompts: list[dict[str, str]], concurrency: int) -> None:
    import httpx

    from throttle.own import records
    from throttle.own.capture import create_capture_app

    by_text = _register_texts(prompts)
    order = {p["id"]: i for i, p in enumerate(prompts)}
    have = {e.id for e in records.iter_exchanges(root)}
    todo = [p for p in prompts if hashlib.sha256(f"replay:{p['id']}".encode()).hexdigest()[:32] not in have]
    log(f"capture: {len(prompts) - len(todo)} already captured, {len(todo)} to go via {API_STAND_IN}")
    if not todo:
        return

    async def go() -> int:
        upstream = httpx.AsyncClient(timeout=httpx.Timeout(connect=15, read=900, write=60, pool=900),
                                     trust_env=False)
        app = create_capture_app(OLLAMA, "http://127.0.0.1:9", redaction=records.REDACTION_MASK, root=root,
                                 salt=b"throttle-public-replay", client=upstream)
        state = app.state.capture
        real_append = state._append

        def stamped(exchange: records.Exchange) -> None:
            pid = by_text.get(json.dumps((exchange.request or {}).get("messages")))
            if pid is not None:
                exchange.ts = _clock(order[pid], len(prompts))
                exchange.id = hashlib.sha256(f"replay:{pid}".encode()).hexdigest()[:32]
            real_append(exchange)

        state._append = stamped
        gate = asyncio.Semaphore(concurrency)
        done = 0
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://capture.replay",
                                     timeout=900) as client:

            async def one(p: dict[str, str]) -> None:
                nonlocal done
                async with gate:
                    await client.post("/v1/chat/completions", headers={"authorization": "Bearer local"},
                                      json={"model": API_STAND_IN,
                                            "messages": [{"role": "user", "content": p["text"]}],
                                            "max_tokens": MAX_TOKENS, "temperature": 0})
                done += 1
                if done % 25 == 0:
                    log(f"capture: {done}/{len(todo)}")

            await asyncio.gather(*(one(p) for p in todo))
        await state.drain()
        await upstream.aclose()
        return state.written

    written = asyncio.run(go())
    log(f"capture: wrote {written} exchanges")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="raw WildChat pages -> redacted prompts.jsonl")
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seed", type=int, default=SEED)
    r = sub.add_parser("run", help="capture, shadow, eval, check and report on local Ollama models")
    r.add_argument("--prompts", type=Path, required=True)
    r.add_argument("--work", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--candidates", nargs="+", default=CANDIDATES)
    r.add_argument("--scale", type=int, default=DEFAULT_SCALE, help="requests a month each replayed request stands for")
    r.add_argument("--gpu-hourly", type=float, default=GPU_HOURLY, help="ASSUMED GPU $/hour")
    r.add_argument("--concurrency", type=int, default=4)
    r.add_argument("--max-requests", type=int, default=0,
                   help="shadow and judge at most this many captured requests (stratified, seeded); 0 = all")
    r.add_argument("--steps", nargs="+", default=["capture", "shadow", "eval", "check", "report"])
    args = parser.parse_args(argv)
    if args.cmd == "prepare":
        return 0 if prepare(args.raw.expanduser(), args.out.expanduser(), args.n, args.seed) else 1
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
