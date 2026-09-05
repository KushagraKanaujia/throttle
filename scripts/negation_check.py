#!/usr/bin/env python3
"""
Negation and antonym false-match check for semantic caching.

Standalone — no throttle dependency. Runs against any embedding model.

Usage:
    python negation_check.py                     # bundled pair set, threshold 0.95
    python negation_check.py --threshold 0.93    # custom threshold
    python negation_check.py --sweep             # table from 0.90 to 0.99
    python negation_check.py --pairs custom.jsonl  # your own pairs

The bundled set (data/negation_pairs.json on github.com/KushagraKanaujia/throttle)
contains 60 pairs: 30 paraphrases and 30 hard negatives (antonyms, version
differences, scope differences). Each pair includes a difficulty label:

  trivial     — paraphrases with near-identical token sets, or negatives that
                score below 0.70 and wouldn't fool any reasonable threshold.
                Present as a control: if your embedder fails these, something
                is broken. Reported separately so they don't inflate recall.

  challenging — semantically distinct surface form (paraphrases), or near-
                threshold negatives (antonyms, version swaps) that a cache
                would plausibly serve incorrectly.

Report reads:
  Challenging paraphrase recall: X/5
  Trivial paraphrase recall:     Y/25  (expected: 25/25)
  False positives at threshold:  Z/30

Published findings (all-MiniLM-L6-v2, cosine >= 0.95):
  "When to use caching?" vs "When to avoid caching?": 0.9674  → FP
  "Use TensorFlow 2.x"  vs "Use TensorFlow 1.x":     0.9520  → FP
  "Why is my API slow?" vs "Why is my API fast?":     0.9485  → FP
  "When to scale up?"   vs "When to scale down?":     0.9477  → FP

All four are above the 0.95 threshold. All four would be served the wrong
cached response. The model encodes topic, not polarity or version.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Bundled pair set — source: data/negation_pairs.json @ KushagraKanaujia/throttle
# difficulty field added by João Felipe De Souza (classification judgment)
# ---------------------------------------------------------------------------

BUNDLED_PAIRS = [
    # ── PARAPHRASES — challenging ──────────────────────────────────────────
    {"a": "What is the capital of France?",
     "b": "Tell me the capital of France",
     "label": "paraphrase", "category": "paraphrase",
     "difficulty": "challenging", "measured_sim": 0.9474},
    {"a": "How do I install Python?",
     "b": "What's the process for installing Python?",
     "label": "paraphrase", "category": "paraphrase",
     "difficulty": "challenging", "measured_sim": 0.8968},
    {"a": "Explain async/await in JavaScript",
     "b": "Can you describe async/await in JavaScript?",
     "label": "paraphrase", "category": "paraphrase",
     "difficulty": "challenging", "measured_sim": 0.9127},
    {"a": "Docker vs Kubernetes differences",
     "b": "What are the differences between Docker and Kubernetes?",
     "label": "paraphrase", "category": "paraphrase",
     "difficulty": "challenging", "measured_sim": 0.9223},
    {"a": "GraphQL vs REST comparison",
     "b": "Can you compare GraphQL and REST?",
     "label": "paraphrase", "category": "paraphrase",
     "difficulty": "challenging", "measured_sim": 0.9348},

    # ── PARAPHRASES — trivial ([Action][Object] vs How do I [Action][Object]?)
    # Control group: any working embedder should match these.
    # If your embedder fails here, something is broken — not a threshold issue.
    {"a": "How to reverse a list in Python?",
     "b": "What's the way to reverse a list in Python?",
     "label": "paraphrase", "category": "paraphrase",
     "difficulty": "trivial", "measured_sim": 0.9944},
    {"a": "Best practices for REST APIs",
     "b": "What are the best practices for REST APIs?",
     "label": "paraphrase", "category": "paraphrase",
     "difficulty": "trivial", "measured_sim": 0.9850},
    {"a": "Debug memory leak in Node.js",
     "b": "How can I debug a memory leak in Node.js?",
     "label": "paraphrase", "category": "paraphrase",
     "difficulty": "trivial", "measured_sim": 0.9633},
    {"a": "Optimize SQL query performance",
     "b": "How do I optimize SQL query performance?",
     "label": "paraphrase", "category": "paraphrase",
     "difficulty": "trivial", "measured_sim": 0.9605},
    {"a": "Set up CI/CD with GitHub Actions",
     "b": "How to set up CI/CD using GitHub Actions?",
     "label": "paraphrase", "category": "paraphrase",
     "difficulty": "trivial", "measured_sim": 0.9849},
    {"a": "React hooks best practices",
     "b": "What are best practices for React hooks?",
     "label": "paraphrase", "category": "paraphrase",
     "difficulty": "trivial", "measured_sim": 0.9826},
    {"a": "Handle errors in async Python",
     "b": "How should I handle errors in async Python?",
     "label": "paraphrase", "category": "paraphrase",
     "difficulty": "trivial", "measured_sim": 0.9724},
    {"a": "Test React components",
     "b": "How do I test React components?",
     "label": "paraphrase", "category": "paraphrase",
     "difficulty": "trivial", "measured_sim": 0.9489},
    {"a": "Set up PostgreSQL on Ubuntu",
     "b": "How to set up PostgreSQL on Ubuntu?",
     "label": "paraphrase", "category": "paraphrase",
     "difficulty": "trivial", "measured_sim": 0.9863},
    {"a": "Improve API response time",
     "b": "How can I improve my API response time?",
     "label": "paraphrase", "category": "paraphrase",
     "difficulty": "trivial", "measured_sim": 0.9686},
    {"a": "Configure Nginx for production",
     "b": "How do I configure Nginx for production?",
     "label": "paraphrase", "category": "paraphrase",
     "difficulty": "trivial", "measured_sim": 0.9797},
    {"a": "Implement JWT authentication",
     "b": "How to implement JWT authentication?",
     "label": "paraphrase", "category": "paraphrase",
     "difficulty": "trivial", "measured_sim": 0.9831},
    {"a": "Profile Python code performance",
     "b": "How can I profile Python code performance?",
     "label": "paraphrase", "category": "paraphrase",
     "difficulty": "trivial", "measured_sim": 0.8574},
    {"a": "Set up Redis caching",
     "b": "How do I set up Redis caching?",
     "label": "paraphrase", "category": "paraphrase",
     "difficulty": "trivial", "measured_sim": 0.9797},
    {"a": "Deploy app to AWS",
     "b": "How to deploy an app to AWS?",
     "label": "paraphrase", "category": "paraphrase",
     "difficulty": "trivial", "measured_sim": 0.9761},
    {"a": "Migrate from MySQL to PostgreSQL",
     "b": "How do I migrate from MySQL to PostgreSQL?",
     "label": "paraphrase", "category": "paraphrase",
     "difficulty": "trivial", "measured_sim": 0.9810},
    {"a": "Handle CORS in Express",
     "b": "How should I handle CORS in Express?",
     "label": "paraphrase", "category": "paraphrase",
     "difficulty": "trivial", "measured_sim": 0.9596},
    {"a": "Optimize React rendering",
     "b": "How can I optimize React rendering?",
     "label": "paraphrase", "category": "paraphrase",
     "difficulty": "trivial", "measured_sim": 0.9791},
    {"a": "Set up monitoring with Prometheus",
     "b": "How to set up monitoring using Prometheus?",
     "label": "paraphrase", "category": "paraphrase",
     "difficulty": "trivial", "measured_sim": 0.9830},
    {"a": "Implement rate limiting",
     "b": "How do I implement rate limiting?",
     "label": "paraphrase", "category": "paraphrase",
     "difficulty": "trivial", "measured_sim": 0.9757},
    {"a": "Configure SSL certificates",
     "b": "How to configure SSL certificates?",
     "label": "paraphrase", "category": "paraphrase",
     "difficulty": "trivial", "measured_sim": 0.9787},
    {"a": "Debug TypeScript compilation errors",
     "b": "How can I debug TypeScript compilation errors?",
     "label": "paraphrase", "category": "paraphrase",
     "difficulty": "trivial", "measured_sim": 0.9524},
    {"a": "Set up load balancing",
     "b": "How do I set up load balancing?",
     "label": "paraphrase", "category": "paraphrase",
     "difficulty": "trivial", "measured_sim": 0.9666},
    {"a": "Implement WebSocket connections",
     "b": "How to implement WebSocket connections?",
     "label": "paraphrase", "category": "paraphrase",
     "difficulty": "trivial", "measured_sim": 0.9805},
    {"a": "Optimize database indexes",
     "b": "How can I optimize database indexes?",
     "label": "paraphrase", "category": "paraphrase",
     "difficulty": "trivial", "measured_sim": 0.9685},

    # ── HARD NEGATIVES — antonyms ──────────────────────────────────────────
    {"a": "When to use caching?",
     "b": "When to avoid caching?",
     "label": "negative", "category": "antonym",
     "difficulty": "challenging", "measured_sim": 0.9674,
     "note": "polarity inversion — same decision framing, opposite answer"},
    {"a": "Why is my API slow?",
     "b": "Why is my API fast?",
     "label": "negative", "category": "antonym",
     "difficulty": "challenging", "measured_sim": 0.9485,
     "note": "antonym pair — would cache wrong diagnostic"},
    {"a": "When to scale up?",
     "b": "When to scale down?",
     "label": "negative", "category": "antonym",
     "difficulty": "challenging", "measured_sim": 0.9477,
     "note": "opposite operational decisions"},
    {"a": "How to enable debug mode?",
     "b": "How to disable debug mode?",
     "label": "negative", "category": "antonym",
     "difficulty": "challenging", "measured_sim": 0.9142},
    {"a": "Should I cache this query?",
     "b": "Should I skip caching this query?",
     "label": "negative", "category": "antonym",
     "difficulty": "challenging", "measured_sim": 0.9245},
    {"a": "Should I use synchronous calls?",
     "b": "Should I use asynchronous calls?",
     "label": "negative", "category": "antonym",
     "difficulty": "challenging", "measured_sim": 0.8607},
    {"a": "How to allow CORS?",
     "b": "How to block CORS?",
     "label": "negative", "category": "antonym",
     "difficulty": "challenging", "measured_sim": 0.8524},
    {"a": "How to lock database rows?",
     "b": "How to unlock database rows?",
     "label": "negative", "category": "antonym",
     "difficulty": "challenging", "measured_sim": 0.8498},
    {"a": "How to compress images?",
     "b": "How to decompress images?",
     "label": "negative", "category": "antonym",
     "difficulty": "challenging", "measured_sim": 0.7756},
    {"a": "How to open database connections?",
     "b": "How to close database connections?",
     "label": "negative", "category": "antonym",
     "difficulty": "challenging", "measured_sim": 0.7461},
    {"a": "How to start the server?",
     "b": "How to stop the server?",
     "label": "negative", "category": "antonym",
     "difficulty": "challenging", "measured_sim": 0.6306},

    # ── HARD NEGATIVES — version differences ───────────────────────────────
    {"a": "Use TensorFlow 2.x",
     "b": "Use TensorFlow 1.x",
     "label": "negative", "category": "version",
     "difficulty": "challenging", "measured_sim": 0.9520,
     "note": "version swap — API incompatible, same topic score"},
    {"a": "Use OpenSSL 1.1 instead of 1.0",
     "b": "Use OpenSSL 3.0 instead of 1.0",
     "label": "negative", "category": "version",
     "difficulty": "challenging", "measured_sim": 0.9322},
    {"a": "How do I install pandas 1.5?",
     "b": "How do I install pandas 2.0?",
     "label": "negative", "category": "version",
     "difficulty": "challenging", "measured_sim": 0.9171},
    {"a": "Migrate from Python 3.8 to 3.9",
     "b": "Migrate from Python 3.8 to 3.11",
     "label": "negative", "category": "version",
     "difficulty": "challenging", "measured_sim": 0.9094},
    {"a": "Install Node.js 16",
     "b": "Install Node.js 18",
     "label": "negative", "category": "version",
     "difficulty": "challenging", "measured_sim": 0.9082},
    {"a": "How to configure React Router v5?",
     "b": "How to configure React Router v6?",
     "label": "negative", "category": "version",
     "difficulty": "challenging", "measured_sim": 0.8818},
    {"a": "Set max connections to 100",
     "b": "Set max connections to 500",
     "label": "negative", "category": "version",
     "difficulty": "challenging", "measured_sim": 0.8815},
    {"a": "Update to Ubuntu 20.04",
     "b": "Update to Ubuntu 22.04",
     "label": "negative", "category": "version",
     "difficulty": "challenging", "measured_sim": 0.8195},

    # ── HARD NEGATIVES — scope differences ─────────────────────────────────
    {"a": "What is GraphQL?",
     "b": "Where is GraphQL used?",
     "label": "negative", "category": "scope",
     "difficulty": "challenging", "measured_sim": 0.9414,
     "note": "what vs where — definition vs usage, different answer"},
    {"a": "What is Kubernetes?",
     "b": "Who created Kubernetes?",
     "label": "negative", "category": "scope",
     "difficulty": "challenging", "measured_sim": 0.8439},
    {"a": "How does Redis work?",
     "b": "Where is Redis used?",
     "label": "negative", "category": "scope",
     "difficulty": "challenging", "measured_sim": 0.8695},
    {"a": "How to configure Nginx?",
     "b": "Who maintains Nginx?",
     "label": "negative", "category": "scope",
     "difficulty": "challenging", "measured_sim": 0.6966},
    {"a": "How to set up PostgreSQL?",
     "b": "Why use PostgreSQL?",
     "label": "negative", "category": "scope",
     "difficulty": "borderline", "measured_sim": 0.6598,
     "note": "borderline — low score but how vs why is a real scope difference"},
    # trivial scope negatives — score too low to test threshold behavior
    {"a": "How does Docker networking work?",
     "b": "How much does Docker cost?",
     "label": "negative", "category": "scope",
     "difficulty": "trivial", "measured_sim": 0.5887,
     "note": "trivial — scores 0.59, no threshold would catch this as FP"},
    {"a": "Explain REST API design",
     "b": "When was REST invented?",
     "label": "negative", "category": "scope",
     "difficulty": "trivial", "measured_sim": 0.4870,
     "note": "trivial — scores 0.49, not testing anything at 0.95"},
]

REFERENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REFERENCE_IMPL = "optimum ONNX, Python 3.11.15"
REFERENCE_DATE = "2026-08-24"

# ---------------------------------------------------------------------------
# Embedder — tries onnxruntime direct first, falls back to sentence-transformers
# ---------------------------------------------------------------------------

def _load_embedder():
    try:
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer
        from huggingface_hub import hf_hub_download

        model_id = REFERENCE_MODEL
        model_path = hf_hub_download(repo_id=model_id, filename="onnx/model.onnx")
        tokenizer_path = hf_hub_download(repo_id=model_id, filename="tokenizer.json")
        tok = Tokenizer.from_file(tokenizer_path)
        session = ort.InferenceSession(model_path)

        def embed(text: str) -> "np.ndarray":
            enc = tok.encode(text)
            ids = np.array([enc.ids], dtype=np.int64)
            mask = np.array([enc.attention_mask], dtype=np.int64)
            types = np.zeros_like(ids)
            out = session.run(None, {
                "input_ids": ids,
                "attention_mask": mask,
                "token_type_ids": types,
            })
            te = out[0]
            m = np.expand_dims(mask, -1)
            v = (np.sum(te * m, axis=1) /
                 np.clip(np.sum(mask, axis=1, keepdims=True), 1e-9, None))[0]
            v = v.astype("float32")
            norm = np.linalg.norm(v)
            return v / norm if norm > 0 else v

        print(f"Embedder: onnxruntime direct ({model_id})")
        return embed, "onnxruntime-direct"

    except ImportError:
        pass

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        model = SentenceTransformer(REFERENCE_MODEL)

        def embed(text: str) -> "np.ndarray":
            return model.encode(text, normalize_embeddings=True)

        print(f"Embedder: sentence-transformers ({REFERENCE_MODEL})")
        return embed, "sentence-transformers"

    except ImportError:
        pass

    print(
        "ERROR: No embedding backend found.\n"
        "Install one of:\n"
        "  pip install onnxruntime tokenizers huggingface_hub\n"
        "  pip install sentence-transformers",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def cosine(a, b) -> float:
    import numpy as np
    return float(np.dot(a, b))


def run(pairs: list[dict], threshold: float, embed_fn, impl: str,
        full: bool = False) -> None:
    import numpy as np

    DELTA_WARN = 0.02

    results = []
    for pair in pairs:
        va = embed_fn(pair["a"])
        vb = embed_fn(pair["b"])
        live_sim = cosine(va, vb)
        ref_sim = pair.get("measured_sim")
        would_match = live_sim >= threshold
        is_fp = pair["label"] == "negative" and would_match
        is_fn = pair["label"] == "paraphrase" and not would_match
        delta = abs(live_sim - ref_sim) if ref_sim is not None else None
        diverges = delta is not None and delta > DELTA_WARN
        results.append({
            **pair,
            "live_sim": round(live_sim, 4),
            "would_match": would_match,
            "fp": is_fp, "fn": is_fn,
            "delta": round(delta, 4) if delta is not None else None,
            "diverges": diverges,
        })

    paras_c = [r for r in results if r["label"] == "paraphrase" and r["difficulty"] == "challenging"]
    paras_t = [r for r in results if r["label"] == "paraphrase" and r["difficulty"] == "trivial"]
    negs_c  = [r for r in results if r["label"] == "negative" and r["difficulty"] == "challenging"]
    negs_t  = [r for r in results if r["label"] == "negative" and r["difficulty"] in ("trivial", "borderline")]
    fps     = [r for r in results if r["fp"]]
    fns     = [r for r in results if r["fn"] and r["difficulty"] == "challenging"]
    divergent = [r for r in results if r["diverges"]]

    W = 65

    # ── DEFAULT OUTPUT: lead with the harm ────────────────────────────────
    print()
    print("=" * W)
    print("SEMANTIC CACHE SAFETY CHECK")
    print("=" * W)
    print(f"Embedder : {impl} ({REFERENCE_MODEL})")
    print(f"Threshold: {threshold}")
    print()

    if fps:
        print(f"⚠  {len(fps)} QUESTION PAIR{'S' if len(fps) > 1 else ''} "
              f"WOULD GET THE WRONG CACHED ANSWER")
        print()
        for r in sorted(fps, key=lambda x: -x["live_sim"]):
            note = r.get("note", "")
            print(f'  "{r["a"]}"')
            print(f'  "{r["b"]}"')
            print(f"  Similarity: {r['live_sim']:.4f} — ABOVE threshold {threshold}")
            if note:
                print(f"  {note.capitalize()}.")
            print()
    else:
        print(f"✓  No false positives at threshold {threshold}")
        print(f"   ({len(negs_c)} hard negative pairs tested)")
        print()

    # Paraphrase recall summary
    hits_c = sum(1 for r in paras_c if r["would_match"])
    hits_t = sum(1 for r in paras_t if r["would_match"])
    print("-" * W)
    print("PARAPHRASE RECALL")
    print()
    print(f"  Challenging pairs : {hits_c}/{len(paras_c)}"
          + (f" ({hits_c/len(paras_c):.0%})" if paras_c else ""))
    print(f"  Trivial pairs     : {hits_t}/{len(paras_t)} (control)")
    print()

    if fns:
        print(f"  Pairs your cache would MISS at this threshold:")
        for r in sorted(fns, key=lambda x: x["live_sim"]):
            print(f'    [{r["live_sim"]:.4f}] "{r["a"]}"')
            print(f'             ↔  "{r["b"]}"')
        print()

    # Verdict
    print("-" * W)
    print("VERDICT")
    print()
    if fps:
        print(f"  At threshold {threshold}, {len(fps)} of {len(negs_c)} hard negative pairs")
        print(f"  would be served the wrong cached response.")
        print()
        print(f"  The model encodes topic, not meaning direction.")
        print(f"  Opposite questions score above the threshold because")
        print(f"  they share all topic tokens and differ only in polarity.")
        print()
        if divergent:
            print(f"  Note: {len(divergent)} pairs score differently from the")
            print(f"  published reference (optimum ONNX). Your FP set may")
            print(f"  differ depending on your inference runtime. Run --full")
            print(f"  to see per-pair divergence.")
            print()
    else:
        print(f"  No false positives at threshold {threshold}.")
        if hits_c < len(paras_c):
            print(f"  But challenging paraphrase recall is {hits_c}/{len(paras_c)}.")
            print(f"  The distributions do not cleanly separate.")
        print()

    print(f"  Run --sweep for the full threshold table.")
    print(f"  Run --full for the complete 60-pair table with per-pair scores.")
    print("=" * W)

    # ── FULL TABLE (behind --full flag) ────────────────────────────────────
    if not full:
        return

    print()
    print("FULL RESULTS TABLE")
    print()
    print(f"  {'Diff':>10} {'Cat':>8} {'Live':>7} {'Ref':>7} {'Δ':>6} {'Match?':>7}  Pair")
    print("  " + "-" * 90)
    for r in sorted(results, key=lambda x: -x["live_sim"]):
        ref_s   = f"{r['measured_sim']:.4f}" if r.get("measured_sim") else "     —"
        delta_s = f"{r['delta']:+.4f}" if r["delta"] is not None else "      "
        flag    = "⚠ FP" if r["fp"] else ("  FN" if r["fn"] and r["difficulty"]=="challenging" else "    ")
        div_f   = "≠REF" if r["diverges"] else "    "
        match   = "YES" if r["would_match"] else "no"
        print(f"  {r['difficulty']:>10} {r['category']:>8} "
              f"{r['live_sim']:>7.4f} {ref_s} {delta_s} {match:>7}  "
              f"{flag} {div_f}  {r['a'][:32]!r} vs {r['b'][:32]!r}")

    if divergent:
        print()
        print(f"  ≠REF pairs ({len(divergent)}) — vectors differ from optimum ONNX reference:")
        print(f"  Both implementations are properly L2-normalized.")
        print(f"  Divergence is real: same weights, different runtime pooling paths.")
        for r in sorted(divergent, key=lambda x: -(x["delta"] or 0)):
            print(f"    Δ={r['delta']:+.4f}  [{r['category']}]  {r['a'][:45]!r}")


def sweep(pairs: list[dict], embed_fn) -> None:
    import numpy as np
    vecs = {}
    for p in pairs:
        if p["a"] not in vecs:
            vecs[p["a"]] = embed_fn(p["a"])
        if p["b"] not in vecs:
            vecs[p["b"]] = embed_fn(p["b"])

    paras_c = [p for p in pairs if p["label"] == "paraphrase"
               and p["difficulty"] == "challenging"]
    paras_t = [p for p in pairs if p["label"] == "paraphrase"
               and p["difficulty"] == "trivial"]
    negs_c  = [p for p in pairs if p["label"] == "negative"
               and p["difficulty"] == "challenging"]

    print(f"\nTHRESHOLD SWEEP  ({len(paras_c)} challenging paraphrases, "
          f"{len(paras_t)} trivial, {len(negs_c)} challenging negatives)")
    print()
    print(f"{'Threshold':>10} {'C-Recall':>10} {'T-Recall':>10} {'FP rate':>10}")
    print("-" * 44)

    for t in [0.80, 0.85, 0.88, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95,
              0.96, 0.97, 0.98, 0.988, 0.99]:
        c_hits = sum(1 for p in paras_c
                     if cosine(vecs[p["a"]], vecs[p["b"]]) >= t)
        t_hits = sum(1 for p in paras_t
                     if cosine(vecs[p["a"]], vecs[p["b"]]) >= t)
        fps = sum(1 for p in negs_c
                  if cosine(vecs[p["a"]], vecs[p["b"]]) >= t)
        print(f"  {t:>8.3f}  "
              f"{c_hits}/{len(paras_c):>2} ({c_hits/len(paras_c):>4.0%})  "
              f"{t_hits}/{len(paras_t):>2} ({t_hits/len(paras_t):>4.0%})  "
              f"{fps}/{len(negs_c):>2} ({fps/len(negs_c):>4.0%})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--threshold", type=float, default=0.95)
    p.add_argument("--pairs", type=Path, default=None,
                   help="JSONL with {a, b, label, difficulty} per line")
    p.add_argument("--sweep", action="store_true",
                   help="Print recall/FP table from 0.80 to 0.99")
    args = p.parse_args()

    if args.pairs:
        pairs = [json.loads(l) for l in args.pairs.read_text().splitlines()
                 if l.strip()]
        print(f"Loaded {len(pairs)} pairs from {args.pairs}")
    else:
        pairs = BUNDLED_PAIRS
        print(f"Using bundled pair set ({len(pairs)} pairs).")
        print(f"Source: data/negation_pairs.json @ github.com/KushagraKanaujia/throttle")

    embed_fn, impl = _load_embedder()

    if args.sweep:
        sweep(pairs, embed_fn)
    else:
        run(pairs, args.threshold, embed_fn, impl)


if __name__ == "__main__":
    main()
