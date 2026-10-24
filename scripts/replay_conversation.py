#!/usr/bin/env python3
"""
replay_conversation.py — replay a fixed multi-turn conversation against a
vLLM-compatible OpenAI endpoint and print per-turn cost signals.

Usage:
    python scripts/replay_conversation.py \
        --base-url http://localhost:8000 \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --turns 10 \
        --sleep 2.0 \
        --out replay

Outputs:
    replay.jsonl          one JSON line per turn
    stdout                per-turn summary + final utilisation table
"""

import argparse
import json
import time
import sys
from pathlib import Path

try:
    import httpx
except ImportError:
    sys.exit("httpx is required: pip install httpx")

# ---------------------------------------------------------------------------
# Fixed OS-concepts conversation (growing prefix, realistic cache pressure)
# ---------------------------------------------------------------------------
TURNS = [
    "What is a process in an operating system?",
    "How does the OS scheduler decide which process runs next?",
    "What is the difference between preemptive and cooperative multitasking?",
    "Explain virtual memory and why it is useful.",
    "What is a page fault and how does the OS handle it?",
    "How do semaphores differ from mutexes?",
    "What is a deadlock and what are the four Coffman conditions?",
    "How does the banker's algorithm avoid deadlock?",
    "What is the difference between a monolithic kernel and a microkernel?",
    "Summarise everything we have discussed about operating systems.",
]


def parse_args():
    p = argparse.ArgumentParser(description="Replay a fixed conversation and measure cache signals.")
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--model", required=True)
    p.add_argument("--turns", type=int, default=10)
    p.add_argument("--sleep", type=float, default=2.0, help="Seconds between turns")
    p.add_argument("--out", default="replay", help="Output file prefix (no extension)")
    return p.parse_args()


def main():
    args = parse_args()
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    turns = TURNS[: args.turns]
    out_path = Path(args.out + ".jsonl")

    messages = []
    records = []

    print(f"Replaying {len(turns)} turns against {args.base_url}")
    print(f"Model : {args.model}")
    print(f"Output: {out_path}")
    print()
    print(f"  {'Turn':>4}  {'elapsed_s':>9}  {'prompt_tok':>10}  {'cached_tok':>10}  {'completion_tok':>14}")
    print("  " + "-" * 55)

    gpu_busy_s = 0.0
    gpu_idle_s = 0.0
    total_cached = 0
    total_prompt = 0

    with httpx.Client(timeout=120.0) as client:
        for i, user_text in enumerate(turns):
            messages.append({"role": "user", "content": user_text})

            payload = {
                "model": args.model,
                "messages": messages,
                "temperature": 0,
                "stream": False,
            }

            t0 = time.perf_counter()
            resp = client.post(url, json=payload)
            elapsed = time.perf_counter() - t0

            resp.raise_for_status()
            data = resp.json()

            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            # vLLM reports cached tokens under prompt_tokens_details
            prompt_details = usage.get("prompt_tokens_details") or {}
            cached_tokens = (
                prompt_details.get("cached_tokens", 0)
                or usage.get("cached_tokens", 0)
            )
            if not prompt_details and i == 0:
                print("  Note: server returned prompt_tokens_details=null;")
                print("        add --enable-prompt-tokens-details to the vLLM")
                print("        launch command for cached-token counts.")
                print()

            assistant_text = data["choices"][0]["message"]["content"]
            messages.append({"role": "assistant", "content": assistant_text})

            print(f"  {i+1:>4}  {elapsed:>9.2f}  {prompt_tokens:>10}  {cached_tokens:>10}  {completion_tokens:>14}")

            record = {
                "turn": i + 1,
                "user": user_text,
                "assistant": assistant_text,
                "elapsed_seconds": round(elapsed, 3),
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached_tokens,
                "completion_tokens": completion_tokens,
            }
            records.append(record)

            gpu_busy_s += elapsed
            total_cached += cached_tokens
            total_prompt += prompt_tokens

            if i < len(turns) - 1:
                time.sleep(args.sleep)
                gpu_idle_s += args.sleep

    # Write JSONL
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Summary
    total_wall = gpu_busy_s + gpu_idle_s
    utilisation = (gpu_busy_s / total_wall * 100) if total_wall > 0 else 0.0
    cache_hit_rate = (total_cached / total_prompt * 100) if total_prompt > 0 else 0.0

    print()
    print("  Summary")
    print(f"    Request time    : {gpu_busy_s:.2f}s")
    print(f"    Gap time        : {gpu_idle_s:.2f}s")
    print(f"    Time in requests: {utilisation:.1f}%")
    print(f"    Cache hit rate  : {cache_hit_rate:.1f}%  ({total_cached}/{total_prompt} prompt tokens cached)")
    print(f"    Output written  : {out_path}")


if __name__ == "__main__":
    main()
