"""
throttle/forecast.py
====================
Offline cache viability forecasting.

Takes a file of the user's own prompts (JSONL or plain text, one per line)
and projects what a Throttle semantic cache would do to that traffic —
without touching any proxy, backend, or production system.

CLI entry point: throttle forecast <file> [options]

Answers four questions:
1. Projected hit rate by tier (Jaccard / embedding / total)
2. Domain classification (bounded / open / insufficient signal)
   based on entity fingerprint variance across near-duplicate clusters
3. FP risk with entity guard on vs off — measured on the user's data
4. Break-even, computed from supplied backend latency or 300ms default,
   using the measured lookup cost curve (15us@10, 134us@100, 1313us@1000)

EMBEDDINGS REQUIREMENT
----------------------
The embedding tier carries the majority of hits on bounded-domain traffic
(44.2% of stream on Bitext vs 7.1% Jaccard). A Jaccard-only forecast on
the same traffic would project ~7% instead of ~51% and print "don't bother"
to exactly the user who should be enabling caching.

Therefore: if embedding dependencies are unavailable, this tool runs the
Jaccard tier, reports those numbers, and refuses to issue a verdict.
Answering wrong is worse than refusing to answer.

To enable the full forecast:
    pip install numpy optimum transformers onnxruntime

MINIMUM SAMPLE GATE
-------------------
Below 100 prompts: exit 3, "insufficient data"
100-500 prompts: report with low-sample warning
500+ prompts: full report

VALIDATION
----------
Run against known datasets to verify the tool reproduces published numbers:
    python -m throttle.forecast --validate-bitext
    python -m throttle.forecast --validate-qqp

Expected:
    Bitext: ~51.3% hit rate, domain=bounded
    QQP   : ~3.8%  hit rate, domain=open
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from throttle.keys import extract_scope_key, extract_prompt

# ---------------------------------------------------------------------------
# Embedding availability check
# ---------------------------------------------------------------------------

from throttle.embeddings import (
    EMBEDDINGS_AVAILABLE,
    get_embedding,
    get_embeddings,
    get_failure_count,
)
try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JACCARD_THRESHOLD = 0.85
EMBEDDING_THRESHOLD = 0.95
MIN_PROMPTS_HARD = 100       # below this: refuse entirely
MIN_PROMPTS_WARN = 500       # below this: warn about low sample
OPEN_DOMAIN_THRESHOLD = 0.15 # >15% of hits trigger entity guard → open
BOUNDED_DOMAIN_THRESHOLD = 0.05  # <5% → bounded

# Break-even lookup cost model (measured: 15us@10, 134us@100, 1313us@1000)
_SLOPE_US = (1313 - 15) / (1000 - 10)
_INTERCEPT_US = 15 - _SLOPE_US * 10


# ---------------------------------------------------------------------------
# Entity fingerprint (from entity guard work)
# ---------------------------------------------------------------------------


def entity_fingerprint(text: str) -> frozenset:
    tokens = text.split()
    entities = set()
    for i, tok in enumerate(tokens):
        clean = tok.strip(".,?!\"'()[]")
        if i > 0 and clean and clean[0].isupper() and not clean.isupper():
            entities.add(clean.lower())
        if re.match(r'^\d[\d\-\.]*$', clean):
            entities.add(clean)
    return frozenset(entities)


def _all_tokens_lower(text: str) -> set:
    return set(
        tok.strip(".,?!\"'()[]").lower()
        for tok in text.split()
        if tok.strip(".,?!\"'()[]")
    )


def entity_guard_fires(query: str, candidate: str) -> bool:
    qf = entity_fingerprint(query)
    cf = entity_fingerprint(candidate)
    diff = qf.symmetric_difference(cf)
    if not diff:
        return False
    q_tok = _all_tokens_lower(query)
    c_tok = _all_tokens_lower(candidate)
    real_diff = set()
    for tok in diff:
        if tok in qf and tok not in c_tok:
            real_diff.add(tok)
        elif tok in cf and tok not in q_tok:
            real_diff.add(tok)
    return bool(real_diff)


# ---------------------------------------------------------------------------
# Jaccard
# ---------------------------------------------------------------------------


def jaccard(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ---------------------------------------------------------------------------
# Embedding model (lazy-loaded singleton)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Lookup cost model
# ---------------------------------------------------------------------------


def lookup_cost_ms(cache_entries: int) -> float:
    return max(_INTERCEPT_US + _SLOPE_US * cache_entries, 0.0) / 1000.0


def is_extrapolated(cache_entries: int) -> bool:
    return cache_entries > 1000


# ---------------------------------------------------------------------------
# Prompt extraction
# ---------------------------------------------------------------------------


_SINGLE_SCOPE = "__single_scope__"


def load_prompts(path: Path) -> list[dict]:
    """
    Load prompts from JSONL or plain text.

    Returns list of {"text": str, "scope_key": str}.

    OpenAI chat-completions JSONL (has "messages" field):
        scope_key via extract_scope_key() — same function the proxy
        uses. One implementation, no divergence possible.

    All other formats (plain text, prompt/query/content/instruction):
        scope_key = _SINGLE_SCOPE (no scope info available).
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    requests = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            try:
                obj = json.loads(line)
                if "messages" in obj:
                    text = extract_prompt(obj)
                    scope_key = extract_scope_key(obj)
                    if text.strip():
                        requests.append({"text": text, "scope_key": scope_key})
                elif "prompt" in obj:
                    requests.append({"text": str(obj["prompt"]), "scope_key": _SINGLE_SCOPE})
                elif "query" in obj:
                    requests.append({"text": str(obj["query"]), "scope_key": _SINGLE_SCOPE})
                elif "content" in obj:
                    requests.append({"text": str(obj["content"]), "scope_key": _SINGLE_SCOPE})
                elif "instruction" in obj:
                    requests.append({"text": str(obj["instruction"]), "scope_key": _SINGLE_SCOPE})
            except json.JSONDecodeError:
                if line.strip():
                    requests.append({"text": line, "scope_key": _SINGLE_SCOPE})
        else:
            requests.append({"text": line, "scope_key": _SINGLE_SCOPE})
    return [r for r in requests if r["text"].strip()]


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


def run_simulation(
    requests: list[dict],
    embedding_available: bool,
) -> dict:
    """
    Simulate cache behavior on the prompt stream.

    Each request is {"text": str, "scope_key": str}.
    Matches are only made within the same scope_key — same logic as
    the proxy. Single-scope input uses _SINGLE_SCOPE for all entries.

    Returns raw simulation results for report generation.
    """
    # Per-scope cache: scope_key -> list[{text, embedding}]
    cache: dict[str, list[dict]] = {}
    hits = []
    misses = []
    distinct_scopes: set[str] = set()

    for idx, req in enumerate(requests):
        prompt = req["text"].strip()
        scope_key = req["scope_key"]
        if not prompt:
            continue

        distinct_scopes.add(scope_key)
        scope_cache = cache.setdefault(scope_key, [])

        t0 = time.perf_counter()

        # Tier 1: Jaccard scan (within scope only)
        jaccard_match = None
        jaccard_score = 0.0
        for entry in scope_cache:
            score = jaccard(prompt, entry["text"])
            if score >= JACCARD_THRESHOLD:
                jaccard_match = entry
                jaccard_score = score
                break

        if jaccard_match is not None:
            elapsed = (time.perf_counter() - t0) * 1000
            guard = entity_guard_fires(prompt, jaccard_match["text"])
            hits.append({
                "stream_index": idx,
                "query": prompt,
                "matched_text": jaccard_match["text"],
                "scope_key": scope_key,
                "tier": "jaccard",
                "similarity": round(jaccard_score, 4),
                "lookup_ms": round(elapsed, 3),
                "entity_guard_fires": guard,
            })
            continue

        # Tier 2: Embedding scan (within scope only)
        if embedding_available:
            emb = get_embedding(prompt)
            if emb is not None and np.any(emb != 0):
                emb_match = None
                emb_score = 0.0
                for entry in scope_cache:
                    entry_emb = entry.get("embedding")
                    if entry_emb is None or not np.any(entry_emb != 0):
                        continue
                    score = float(np.dot(emb, entry_emb))
                    if score >= EMBEDDING_THRESHOLD:
                        emb_match = entry
                        emb_score = score
                        break

                if emb_match is not None:
                    elapsed = (time.perf_counter() - t0) * 1000
                    guard = entity_guard_fires(prompt, emb_match["text"])
                    hits.append({
                        "stream_index": idx,
                        "query": prompt,
                        "matched_text": emb_match["text"],
                        "scope_key": scope_key,
                        "tier": "embedding",
                        "similarity": round(emb_score, 4),
                        "lookup_ms": round(elapsed, 3),
                        "entity_guard_fires": guard,
                    })
                    continue

                scope_cache.append({"text": prompt, "embedding": emb})
                misses.append({"stream_index": idx, "query": prompt, "scope_key": scope_key})
                continue

        # Miss — Jaccard-only mode
        scope_cache.append({"text": prompt, "embedding": None})
        misses.append({"stream_index": idx, "query": prompt, "scope_key": scope_key})

    # Sort hits ascending by similarity — riskiest matches first
    hits.sort(key=lambda h: h["similarity"])

    total_cache_entries = sum(len(v) for v in cache.values())

    return {
        "total": len(requests),
        "hits": hits,
        "misses": misses,
        "cache_size": total_cache_entries,
        "distinct_scopes": len(distinct_scopes),
        "is_multi_scope": len(distinct_scopes) > 1,
    }


# ---------------------------------------------------------------------------
# Domain classification
# ---------------------------------------------------------------------------


def classify_domain(hits: list[dict]) -> tuple[str, float, str]:
    """
    Classify traffic domain based on entity guard trigger rate.

    Returns (classification, guard_rate, explanation)
    """
    if not hits:
        return ("insufficient_signal",
                0.0,
                "No cache hits to classify.")

    guard_fired = sum(1 for h in hits if h["entity_guard_fires"])
    rate = guard_fired / len(hits)

    if rate > OPEN_DOMAIN_THRESHOLD:
        return (
            "open",
            rate,
            f"Entity guard fired on {guard_fired} of {len(hits)} hits "
            f"({rate:.1%}). High entity variation across near-duplicate "
            f"templates indicates open-domain traffic. Serving cached "
            f"responses risks returning wrong answers where the key "
            f"semantic difference is a single swapped entity."
        )
    elif rate < BOUNDED_DOMAIN_THRESHOLD:
        return (
            "bounded",
            rate,
            f"Entity guard fired on {guard_fired} of {len(hits)} hits "
            f"({rate:.1%}). Low entity variation indicates bounded-domain "
            f"traffic. Near-duplicate queries are likely genuine paraphrases "
            f"with the same correct answer."
        )
    else:
        return (
            "mixed",
            rate,
            f"Entity guard fired on {guard_fired} of {len(hits)} hits "
            f"({rate:.1%}). Mixed signal — between bounded and open domain. "
            f"Inspect flagged pairs in hit_pairs.jsonl before enabling."
        )


# ---------------------------------------------------------------------------
# Break-even table
# ---------------------------------------------------------------------------


def build_breakeven_table(
    cache_entries: int,
    backend_latency_ms: float,
) -> list[dict]:
    sizes = [10, 100, 1_000, 10_000, 50_000]
    rows = []
    for size in sizes:
        cost = lookup_cost_ms(size)
        be = cost / backend_latency_ms if backend_latency_ms > 0 else float("inf")
        rows.append({
            "cache_entries": size,
            "lookup_cost_ms": round(cost, 4),
            "break_even_pct": round(be * 100, 2),
            "extrapolated": is_extrapolated(size),
            "is_user_size": abs(size - cache_entries) == min(
                abs(s - cache_entries) for s in sizes
            ),
        })
    return rows


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def compute_verdict(
    hit_rate: float,
    domain: str,
    above_break_even: bool,
    break_even_rate: float,
    embedding_available: bool,
) -> tuple[str, str]:
    if not embedding_available:
        return (
            "no_verdict",
            "Embedding tier unavailable. Cannot issue a verdict — the "
            "embedding tier carries the majority of hits on bounded-domain "
            "traffic and projecting without it would mislead rather than "
            "inform. Install numpy, optimum, transformers, and onnxruntime "
            "to enable the full forecast."
        )

    if domain == "open":
        return (
            "do_not_enable",
            "Your traffic shows open-domain entity variation. A meaningful "
            "fraction of cache hits would return wrong answers where the "
            "correct answer differs by a single swapped entity. Do not "
            "enable caching without an entity-aware guard."
        )

    if not above_break_even:
        return (
            "dont_bother",
            f"Your projected hit rate ({hit_rate:.1%}) is below break-even "
            f"({break_even_rate:.1%}). Caching would add more lookup "
            f"overhead than it saves in backend calls. Not worth enabling "
            f"at this traffic shape."
        )

    if domain == "mixed":
        return (
            "dont_bother",
            f"Your projected hit rate ({hit_rate:.1%}) is above break-even "
            f"({break_even_rate:.1%}), but domain classification is mixed. "
            f"Inspect hit_pairs.jsonl — the flagged pairs are at the top, "
            f"sorted by similarity score ascending. If manual review looks "
            f"clean, enable with monitoring."
        )

    return (
        "enable",
        f"Your traffic is bounded-domain with {hit_rate:.1%} projected hit "
        f"rate, well above the {break_even_rate:.1%} break-even threshold. "
        f"Entity substitution risk is low. Enable caching."
    )


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def render_report(
    prompts: list[str],
    sim: dict,
    domain: str,
    domain_rate: float,
    domain_explanation: str,
    backend_latency_ms: float,
    breakeven_table: list[dict],
    verdict: str,
    verdict_reason: str,
    embedding_available: bool,
    low_sample_warning: bool,
    output_dir: Optional[Path],
) -> None:
    total = sim["total"]
    hits = sim["hits"]
    misses = sim["misses"]
    cache_size = sim["cache_size"]

    n_hits = len(hits)
    n_misses = len(misses)
    hit_rate = n_hits / total if total else 0.0

    jaccard_hits = [h for h in hits if h["tier"] == "jaccard"]
    emb_hits = [h for h in hits if h["tier"] == "embedding"]

    guard_fired_hits = [h for h in hits if h["entity_guard_fires"]]
    guard_fired_idxs = [h["stream_index"] for h in guard_fired_hits]

    # Break-even at current cache size
    current_cost = lookup_cost_ms(cache_size)
    be_rate = current_cost / backend_latency_ms if backend_latency_ms > 0 else float("inf")
    above_be = hit_rate > be_rate

    VERDICT_LABELS = {
        "enable": "✅  ENABLE CACHING",
        "dont_bother": "⚠   DON'T BOTHER",
        "do_not_enable": "🚫  DO NOT ENABLE",
        "no_verdict": "❓  NO VERDICT — INSTALL EMBEDDINGS",
    }

    distinct_scopes = sim.get("distinct_scopes", 1)
    is_multi_scope = sim.get("is_multi_scope", False)

    lines = [
        "=" * 70,
        "THROTTLE FORECAST — Cache Viability Report",
        "=" * 70,
        "",
    ]

    if is_multi_scope:
        lines += [
            f"ℹ  MULTI-SCOPE INPUT DETECTED",
            f"   Your log contains {distinct_scopes} distinct scopes (model/temperature",
            f"   combinations). Hit rates are projected within scope — a prompt",
            f"   from scope A will never match a cached response from scope B.",
            f"   This means your actual hit rate may be lower than single-scope",
            f"   traffic with the same paraphrase density.",
            "",
        ]

    if low_sample_warning:
        lines += [
            "⚠  LOW SAMPLE WARNING",
            f"   {total} prompts is below the 500-prompt threshold for a",
            "   stable projection. Results are directional, not decision-grade.",
            "",
        ]

    lines += [
        "INPUT",
        f"  Prompts supplied      : {total:,}",
        f"  Embedding encode failures : {encode_failures:,}" + (" ⚠" if encode_failures else ""),
        f"  Effective sample size : {effective_sample:,}",
        f"  Unique in cache       : {cache_size:,}",
        f"  Embedding tier        : {'available' if embedding_available else 'NOT AVAILABLE — Jaccard-only run'}",
        "",
        "PROJECTED CACHE PERFORMANCE",
        f"  Total hit rate        : {hit_rate:.1%}  ({n_hits:,} of {total:,} prompts)",
        f"    Jaccard tier        : {len(jaccard_hits):,}  ({len(jaccard_hits)/total:.1%} of stream)",
        f"    Embedding tier      : {len(emb_hits):,}  ({len(emb_hits)/total:.1%} of stream)",
        "",
    ]

    if not embedding_available:
        lines += [
            "  ⚠  Embedding tier not projected. Install dependencies to",
            "     see the full hit rate:",
            "     pip install numpy optimum transformers onnxruntime",
            "",
        ]

    lines += [
        "DOMAIN CLASSIFICATION",
        f"  Entity guard trigger  : {len(guard_fired_hits)} of {n_hits} hits ({domain_rate:.1%})",
        f"  Classification        : {domain.upper()}",
        f"  {domain_explanation}",
        "",
        "FALSE POSITIVE RISK",
        f"  Without entity guard  : {n_hits:,} hits unscreened",
        f"  With entity guard     : {len(guard_fired_hits)} hits flagged as potential entity",
        f"                          substitution ({domain_rate:.1%} of hits)",
    ]

    if guard_fired_hits:
        lines += [
            f"  → Inspect these {len(guard_fired_hits)} pairs in hit_pairs.jsonl.",
            f"    They appear at the TOP of the file (sorted by similarity",
            f"    ascending — riskiest matches first).",
            f"    Stream indexes: {sorted(guard_fired_idxs)[:10]}"
            + (" ..." if len(guard_fired_idxs) > 10 else ""),
        ]
    else:
        lines += [
            "  → No entity substitution patterns detected in hits.",
        ]

    lines += [
        "",
        f"BREAK-EVEN ANALYSIS (backend latency: {backend_latency_ms:.0f}ms)",
        f"  Lookup cost at {cache_size:,} cache entries : {current_cost:.3f}ms",
        f"  Break-even hit rate              : {be_rate:.2%}",
        f"  Your projected hit rate          : {hit_rate:.1%}",
        f"  Above break-even                 : {'YES' if above_be else 'NO'}",
        "",
        f"  {'Cache size':>14} | {'Lookup cost':>12} | {'Break-even':>12} | {'Note':>15}",
        f"  {'-'*14}-+-{'-'*12}-+-{'-'*12}-+-{'-'*15}",
    ]

    for row in breakeven_table:
        extrap = "extrapolated" if row["extrapolated"] else "measured"
        marker = " ← you" if row["is_user_size"] else ""
        lines.append(
            f"  {row['cache_entries']:>14,} | "
            f"{row['lookup_cost_ms']:>10.3f}ms | "
            f"{row['break_even_pct']:>10.2f}%  | "
            f"{extrap}{marker}"
        )

    lines += [
        "",
        "=" * 70,
        f"VERDICT:  {VERDICT_LABELS.get(verdict, verdict)}",
        "=" * 70,
        f"  {verdict_reason}",
        "",
    ]

    if output_dir:
        lines += [
            "FILES",
            f"  hit_pairs.jsonl — all {n_hits} hits, sorted by similarity",
            f"                    ascending (riskiest first)",
            f"                    entity-guard-flagged pairs at top",
            f"  forecast.json   — machine-readable summary",
            "",
        ]

    report_text = "\n".join(lines)
    print(report_text)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # hit_pairs.jsonl — already sorted ascending by similarity from simulation
        hits_path = output_dir / "hit_pairs.jsonl"
        with hits_path.open("w") as f:
            for h in hits:
                f.write(json.dumps(h) + "\n")

        # forecast.json
        summary = {
            "total_prompts": total,
            "cache_size": cache_size,
            "embedding_available": embedding_available,
            "hit_rate": round(hit_rate, 4),
            "jaccard_hits": len(jaccard_hits),
            "embedding_hits": len(emb_hits),
            "domain": domain,
            "entity_guard_trigger_rate": round(domain_rate, 4),
            "entity_guard_flagged_hits": len(guard_fired_hits),
            "backend_latency_ms": backend_latency_ms,
            "lookup_cost_ms_at_cache_size": round(current_cost, 4),
            "break_even_rate": round(be_rate, 4),
            "above_break_even": above_be,
            "verdict": verdict,
            "verdict_reason": verdict_reason,
            "jaccard_threshold": JACCARD_THRESHOLD,
            "embedding_threshold": EMBEDDING_THRESHOLD,
        }
        summary_path = output_dir / "forecast.json"
        summary_path.write_text(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Validation runs
# ---------------------------------------------------------------------------


def validate(dataset: str) -> None:
    """
    Validate forecast against known datasets.
    Expected: Bitext ~51.3% bounded, QQP ~3.8% open.
    """
    from datasets import load_dataset
    import random

    print(f"Running validation against {dataset} ...")

    if dataset == "bitext":
        ds = load_dataset(
            "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
            split="train",
        )
        prompts = [
            row["instruction"].strip()
            for row in ds
            if row.get("instruction", "").strip()
        ]
        expected_hit_rate = 0.513
        expected_domain = "bounded"

    elif dataset == "qqp":
        ds = load_dataset("glue", "qqp", split="train+validation")
        freq: Counter = Counter()
        for row in ds:
            if row["question1"]:
                freq[row["question1"].strip()] += 1
            if row["question2"]:
                freq[row["question2"].strip()] += 1
        stream = []
        for q, c in freq.items():
            stream.extend([q] * c)
        random.seed(42)
        random.shuffle(stream)
        prompts = stream[:10_000]
        expected_hit_rate = 0.038
        expected_domain = "open"
    else:
        print(f"Unknown dataset: {dataset}. Use 'bitext' or 'qqp'.")
        sys.exit(2)

    embedding_ok = EMBEDDINGS_AVAILABLE
    sim = run_simulation(
        [{"text": p, "scope_key": "__single_scope__"} for p in prompts],
        embedding_ok,
    )
    hits = sim["hits"]
    total = sim["total"]
    actual_hit_rate = len(hits) / total if total else 0.0
    domain, domain_rate, _ = classify_domain(hits)

    print(f"  Dataset  : {dataset}")
    print(f"  Prompts  : {total:,}")
    print(f"  Hit rate : {actual_hit_rate:.1%}  (expected ~{expected_hit_rate:.1%})")
    print(f"  Domain   : {domain}  (expected {expected_domain})")

    hr_ok = abs(actual_hit_rate - expected_hit_rate) < 0.10
    dom_ok = domain == expected_domain

    if hr_ok and dom_ok:
        print(f"  ✅ PASS — within 10pp of expected hit rate, domain matches")
    else:
        issues = []
        if not hr_ok:
            issues.append(
                f"hit rate {actual_hit_rate:.1%} more than 10pp from {expected_hit_rate:.1%}"
            )
        if not dom_ok:
            issues.append(f"domain={domain}, expected {expected_domain}")
        print(f"  ❌ FAIL — {'; '.join(issues)}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "file",
        nargs="?",
        type=Path,
        help="Prompt file (JSONL or plain text, one prompt per line).",
    )
    p.add_argument(
        "--backend-latency-ms",
        type=float,
        default=300.0,
        metavar="MS",
        help="Your backend round-trip latency in ms (default: 300).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/forecast"),
        metavar="DIR",
        help="Output directory for hit_pairs.jsonl and forecast.json "
             "(default: results/forecast/).",
    )
    p.add_argument(
        "--no-output",
        action="store_true",
        help="Print report only, do not write files.",
    )
    p.add_argument(
        "--validate-bitext",
        action="store_true",
        help="Validate against Bitext dataset. Expects ~51.3%% hit rate, "
             "domain=bounded.",
    )
    p.add_argument(
        "--validate-qqp",
        action="store_true",
        help="Validate against QQP dataset. Expects ~3.8%% hit rate, "
             "domain=open.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.validate_bitext:
        validate("bitext")
        return

    if args.validate_qqp:
        validate("qqp")
        return

    if args.file is None:
        print(
            "Usage: throttle forecast <file> [options]\n"
            "       throttle forecast --validate-bitext\n"
            "       throttle forecast --validate-qqp\n"
            "\nRun 'throttle forecast --help' for full options.",
            file=sys.stderr,
        )
        sys.exit(2)

    if not args.file.exists():
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(2)

    # Load prompts
    requests = load_prompts(args.file)

    if len(requests) < MIN_PROMPTS_HARD:
        print(
            f"Insufficient data: {len(requests)} prompts found, "
            f"{MIN_PROMPTS_HARD} required.\n"
            "Provide a larger sample for a meaningful projection.\n"
            "Exit 3: inconclusive.",
            file=sys.stderr,
        )
        sys.exit(3)

    low_sample_warning = len(requests) < MIN_PROMPTS_WARN

    # Check embedding availability
    embedding_available = EMBEDDINGS_AVAILABLE
    if not embedding_available:
        print(
            "Note: embedding dependencies not found. Running Jaccard-only "
            "mode. No verdict will be issued.\n"
            "To enable full forecast:\n"
            "    pip install numpy optimum transformers onnxruntime",
            file=sys.stderr,
        )

    # Run simulation
    sim = run_simulation(requests, embedding_available)

    # Classify domain
    domain, domain_rate, domain_explanation = classify_domain(sim["hits"])

    # Break-even
    total = sim["total"]
    hits = sim["hits"]
    cache_size = sim["cache_size"]
    hit_rate = len(hits) / total if total else 0.0
    current_cost = lookup_cost_ms(cache_size)
    be_rate = (
        current_cost / args.backend_latency_ms
        if args.backend_latency_ms > 0
        else float("inf")
    )
    above_be = hit_rate > be_rate

    breakeven_table = build_breakeven_table(cache_size, args.backend_latency_ms)

    # Verdict
    verdict, verdict_reason = compute_verdict(
        hit_rate=hit_rate,
        domain=domain,
        above_break_even=above_be,
        break_even_rate=be_rate,
        embedding_available=embedding_available,
    )

    # Render
    render_report(
        prompts=requests,
        sim=sim,
        domain=domain,
        domain_rate=domain_rate,
        domain_explanation=domain_explanation,
        backend_latency_ms=args.backend_latency_ms,
        breakeven_table=breakeven_table,
        verdict=verdict,
        verdict_reason=verdict_reason,
        embedding_available=embedding_available,
        low_sample_warning=low_sample_warning,
        output_dir=None if args.no_output else args.output_dir,
    )


if __name__ == "__main__":
    main()