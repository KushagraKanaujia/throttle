"""Command-line interface for Throttle v2."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import queue
import re
import sys
import textwrap
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import __version__
from .benchmark import (
    ARTIFACT_TYPE,
    SCHEMA_VERSION,
    RunBudget,
    RunProgress,
    build_plan,
    load_prompts,
    run_native,
    validate_config,
)
from .compare import (
    ComparisonInputError,
    compare_reports,
    load_report,
)
from .golden import (
    GOLDEN_SESSION_ARTIFACT_TYPE,
    GoldenTreatmentError,
    build_golden_plan,
    golden_positions,
    golden_position_config,
    parse_golden_treatment_flags,
    validate_golden_sequence,
)
from .models import CostModel, EndpointConfig, LoadCondition, RunConfig, SafetyLimits
from .check import (
    CHECK_DESCRIPTION,
    CHECK_EPILOG,
    add_check_arguments,
    handle_check,
)
from .provenance import ACCELERATOR_BACKENDS
from .result_store import (
    Provenance,
    ResultStoreError,
    append_record,
    build_record,
    find_match,
    format_match_message,
    load_records,
)

DEFAULT_OUTPUT = Path("throttle-report.json")
DEFAULT_COMPARE_OUTPUT = Path("throttle-comparison.json")
DEFAULT_GOLDEN_OUTPUT_DIR = Path("throttle-golden")
ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_INCONCLUSIVE = 3
EXIT_CANCELLED = 130

GETTING_STARTED = """\
Get your first $/M-token number:

  1. Measure a real endpoint and calibrate (local Ollama shown; any OpenAI-compatible
     URL works, with or without /v1). The hourly rate is what your GPU(s) cost you.
     The first check repeats itself 3 times back to back to measure run-to-run noise
     (a few minutes on a laptop; --no-calibrate runs once):
       throttle check --url http://localhost:11434 --model llama3.2:3b --gpu-hourly-rate 1.50 --label before

  2. Change one setting (Ollama on macOS: quit the Ollama app, then run
     OLLAMA_NUM_PARALLEL=4 ollama serve in another terminal).

  3. Run again for the verdict (CHEAPER / MORE EXPENSIVE only beyond the noise and
     the 95% CIs; otherwise NO WINNER). --config only records what you changed:
       throttle check --url http://localhost:11434 --model llama3.2:3b --gpu-hourly-rate 1.50 --label after --config OLLAMA_NUM_PARALLEL=4

Cut the bill (free, local):
  throttle autotune --dry-run             try Autotune on a simulated server
  throttle autotune --launch ... --space ...   find the cheapest config that passes OUTPUT CHANGED
  throttle gate --baseline A --candidate B     CI cost gate (exit 1 on MORE EXPENSIVE / OUTPUT CHANGED)

Switch models or engines? (free, local):
  throttle compare --baseline URL --candidate URL --eval-set FILE ...  your prompts -> SWITCH?
  throttle coldstart --cmd "vllm serve ..." --url URL --model M        time a cold start

Other commands: plan / smoke / benchmark / golden (capped, evidence-labelled runs),
compare (saved reports), savings, watch (read vLLM /metrics),
proxy + sessions (optional cache and agent profiling), ui.
No server yet? Run 'throttle check --help'.
Run 'throttle COMMAND --help' for details.
"""

EXPLORATORY_SWEEP_WARNING = (
    "NOTE: this {kind} sweep is exploratory only and cannot reach "
    "decision_eligible: true because its load order is not counterbalanced. "
    "Use `throttle golden --help` for a decision-grade run."
)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _non_negative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative and finite")
    return parsed


def _positive_float(value: str) -> float:
    parsed = _non_negative_float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return parsed


class _ExplicitApiKeyEnv(argparse.Action):
    """Store --api-key-env and remember that the user typed it."""

    def __call__(self, parser, namespace, values, option_string=None):  # type: ignore[override]
        setattr(namespace, self.dest, values)
        setattr(namespace, "api_key_env_explicit", True)


def _add_endpoint_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model", required=True, help="model identifier sent to the API"
    )
    parser.add_argument(
        "--url",
        "--endpoint-url",
        dest="url",
        required=True,
        help=(
            "server URL: http://host:port, http://host:port/v1, or the exact "
            "chat-completions route (--endpoint-url is accepted too)"
        ),
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        metavar="NAME",
        action=_ExplicitApiKeyEnv,
        help=(
            "environment variable containing the bearer key (default: "
            "OPENAI_API_KEY; for a localhost endpoint with no key set, no key is sent)"
        ),
    )
    parser.add_argument(
        "--allow-insecure-http",
        action="store_true",
        help="allow plaintext HTTP away from loopback (unsafe; recorded in manifest)",
    )


class _HourlyPriceAction(argparse.Action):
    """Store the hourly price and remember which spelling the user typed.

    --total-hourly-price clearly means all GPUs; --gpu-hourly-rate reads like
    a per-GPU rate, so combining it with --gpus N > 1 is refused as ambiguous.
    """

    def __call__(self, parser, namespace, values, option_string=None):  # type: ignore[override]
        setattr(namespace, self.dest, values)
        setattr(namespace, "hourly_price_flag", option_string)


def _add_cost_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cost-model",
        choices=(
            "unknown",
            "dedicated-hourly",
            "serverless-active-seconds",
            "user-supplied",
        ),
        default=None,
        help=(
            "how you pay for the endpoint (default: dedicated-hourly when an "
            "hourly price is given, otherwise unknown)"
        ),
    )
    parser.add_argument("--gpus", type=_positive_int, default=1)
    dedicated = parser.add_mutually_exclusive_group()
    dedicated.add_argument(
        "--total-hourly-price",
        "--gpu-hourly-rate",
        "--gpu-rate-per-hour",
        dest="total_hourly_price",
        type=_positive_float,
        action=_HourlyPriceAction,
        metavar="USD",
        help=(
            "what ALL GPUs behind the endpoint cost per hour together, in dollars "
            "(--gpu-hourly-rate is accepted too; implies --cost-model dedicated-hourly). "
            "For a per-GPU price use --per-gpu-hourly-price with --gpus"
        ),
    )
    dedicated.add_argument("--per-gpu-hourly-price", type=_positive_float)
    parser.add_argument("--active-second-price", type=_positive_float)
    parser.add_argument("--max-active-workers", type=_positive_int)
    parser.add_argument("--billed-active-seconds", type=_non_negative_float)
    parser.add_argument("--user-supplied-total", type=_positive_float)
    parser.add_argument(
        "--allow-unknown-cost",
        action="store_true",
        help="explicitly acknowledge that the spend ceiling cannot be enforced",
    )


def _add_workload_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompts", type=Path, help="measured JSONL workload")
    parser.add_argument(
        "--warmup-prompts",
        type=Path,
        help="separate warm-up JSONL workload (default: bundled separate set)",
    )
    traffic = parser.add_mutually_exclusive_group()
    traffic.add_argument("--concurrency", nargs="+", type=_positive_int, metavar="N")
    traffic.add_argument(
        "--request-rate", nargs="+", type=_positive_float, metavar="RPS"
    )
    parser.add_argument(
        "--open-loop-max-in-flight",
        type=_positive_int,
        default=64,
        help="in-flight ceiling for constant-rate traffic",
    )
    parser.add_argument("--max-tokens", type=_positive_int, default=128)
    parser.add_argument("--blocks", type=_positive_int)
    block_bound = parser.add_mutually_exclusive_group()
    block_bound.add_argument(
        "--requests-per-block",
        "--requests",
        dest="requests_per_block",
        type=_positive_int,
    )
    block_bound.add_argument("--block-seconds", type=_positive_float)
    parser.add_argument("--warmup-requests", type=int)
    parser.add_argument("--timeout-seconds", type=_positive_float, default=120.0)
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--p95-slo-ms", type=_positive_float)
    parser.add_argument("--ttft-slo-ms", type=_positive_float)
    parser.add_argument("--seed", type=int, default=42)


def _add_safety_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-requests", type=_positive_int, default=10_000)
    parser.add_argument("--max-tokens-per-request", type=_positive_int, default=1_024)
    parser.add_argument(
        "--max-total-requested-tokens", type=_positive_int, default=2_000_000
    )
    parser.add_argument(
        "--max-elapsed-seconds",
        type=_positive_float,
        help=(
            "whole-run ceiling (default: 120 for smoke, 900 for benchmark, "
            "5400 for the six-position golden session)"
        ),
    )
    parser.add_argument("--max-errors", type=_positive_int, default=1)
    parser.add_argument("--max-concurrency", type=_positive_int, default=64)
    parser.add_argument("--max-response-bytes", type=_positive_int, default=2_000_000)
    parser.add_argument("--max-estimated-spend", type=_positive_float, default=3.0)


def _add_manifest_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cache-policy",
        choices=("unknown", "disabled", "cold", "warm", "representative"),
        default="unknown",
        help=(
            "declare the server's prefix-cache state for the report (default: "
            "unknown; golden requires an explicit value). This flag does not "
            "change the traffic: smoke, benchmark and golden resend the same "
            "measured prompts in every block and position (that fixed set is "
            "what the recorded workload sha256 covers), so a server with "
            "prefix caching on (vLLM V1 by default, SGLang, Ollama) serves "
            "repeats warm. Use 'disabled' only if caching is off on the "
            "server, 'warm' if it is on. 'throttle check' sends unique "
            "prompts by default (see its --warm-cache)"
        ),
    )
    parser.add_argument("--model-revision", default="unknown")
    parser.add_argument("--image-digest", default="unknown")
    parser.add_argument("--gpu", "--accelerator", dest="gpu", default="unknown")
    parser.add_argument(
        "--gpu-fingerprint",
        "--accelerator-fingerprint",
        dest="gpu_fingerprint",
        default="unknown",
    )
    parser.add_argument("--cuda-version", default="unknown")
    parser.add_argument("--driver-version", default="unknown")
    parser.add_argument(
        "--accelerator-backend",
        choices=ACCELERATOR_BACKENDS,
        default="cuda",
    )
    parser.add_argument("--accelerator-runtime-version", default="unknown")
    parser.add_argument("--host-os-version", default="unknown")
    parser.add_argument("--software-environment-digest", default="unknown")
    parser.add_argument("--server-name", default="unknown")
    parser.add_argument("--server-version", default="unknown")
    parser.add_argument(
        "--engine-flag",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="effective, non-secret engine flag; repeat as needed",
    )
    parser.add_argument(
        "--engine-flags-provenance",
        choices=("operator_attested", "runtime_verified"),
        default="operator_attested",
    )
    parser.add_argument(
        "--variant",
        choices=("baseline", "candidate", "unspecified"),
        default="unspecified",
    )
    parser.add_argument("--sequence-position", default="unspecified")
    parser.add_argument(
        "--evidence-source",
        choices=("unverified_endpoint", "live_inference", "synthetic_validation"),
        default="unverified_endpoint",
    )


def _add_cache_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--enable-cache",
        action="store_true",
        help="Enable similarity-based semantic caching to bypass API requests"
    )
    parser.add_argument(
        "--cache-ttl-seconds",
        type=_positive_float,
        default=3600.0,
        help="Time-to-live for cached responses in seconds"
    )
    parser.add_argument(
        "--cache-max-size",
        type=_positive_int,
        default=1000,
        help="Maximum number of items to hold in the cache (FIFO eviction)"
    )
    parser.add_argument(
        "--cache-similarity-threshold",
        type=_positive_float,
        default=0.85,
        help="Jaccard similarity threshold [0.0 - 1.0] for a cache hit"
    )

def _add_run_options(parser: argparse.ArgumentParser) -> None:
    _add_endpoint_options(parser)
    _add_cost_options(parser)
    _add_workload_options(parser)
    _add_safety_options(parser)
    _add_manifest_options(parser)
    _add_cache_options(parser)


def _server_root_url(value: str) -> str:
    """Strip a trailing /v1 or /v1/chat/completions (proxy appends its own)."""

    base = value.strip().rstrip("/")
    for suffix in ("/chat/completions", "/v1"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="throttle",
        description=(
            "Measure what LLM inference costs in dollars per million tokens, and "
            "which serving-config change lowers it."
        ),
        epilog=GETTING_STARTED,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, metavar="COMMAND"
    )

    plan = subparsers.add_parser(
        "plan",
        help="show traffic, cost, destination, and privacy without sending traffic",
    )
    plan.add_argument("--run-mode", choices=("smoke", "benchmark"), default="smoke")
    _add_run_options(plan)

    smoke = subparsers.add_parser(
        "smoke", help="run a short, explicitly non-decision-grade check"
    )
    _add_run_options(smoke)
    smoke.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    benchmark = subparsers.add_parser(
        "benchmark",
        help="run sustained exploratory evidence blocks (sweeps are not counterbalanced)",
    )
    _add_run_options(benchmark)
    benchmark.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    golden = subparsers.add_parser(
        "golden",
        help="orchestrate the six-position counterbalanced decision protocol",
        description=(
            "Run B1/C1/B2/C2/B3/C3 against one endpoint/accelerator. Throttle pauses "
            "for the operator to apply and verify each server configuration; it "
            "never reconfigures the server itself."
        ),
        epilog=(
            "Start with the same arguments plus --dry-run to inspect all six "
            "positions and session ceilings without reading the API key or sending "
            "traffic. See the Golden live protocol section in the project README "
            "for the complete pinned example."
        ),
    )
    _add_run_options(golden)
    golden.add_argument(
        "--baseline-config",
        "--baseline-engine-flag",
        dest="baseline_config",
        required=True,
        metavar="NAME=VALUE",
        help=(
            "verified baseline treatment as canonical "
            "max_num_seqs=INTEGER (1..2147483647)"
        ),
    )
    golden.add_argument(
        "--candidate-config",
        "--candidate-engine-flag",
        dest="candidate_config",
        required=True,
        metavar="NAME=VALUE",
        help=(
            "verified candidate treatment as a distinct canonical "
            "max_num_seqs=INTEGER (1..2147483647)"
        ),
    )
    golden.add_argument(
        "--dry-run",
        "--plan",
        dest="dry_run",
        action="store_true",
        help="show all six positions and session ceilings without reading a key or sending traffic",
    )
    golden.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_GOLDEN_OUTPUT_DIR,
        help="new directory for B1.json through C3.json and golden.json",
    )
    golden.add_argument(
        "--operator",
        default=None,
        help=(
            "who is running this, for the result store's provenance record "
            "(defaults to $USER@hostname if not set)"
        ),
    )
    golden.add_argument(
        "--hardware-ownership",
        choices=("owned", "rented"),
        default=None,
        help=(
            "required to persist this run to the result store; a decision-eligible "
            "result whose ownership can't be determined is not stored"
        ),
    )
    golden.add_argument(
        "--hardware-provider",
        default="unknown",
        help="e.g. runpod, lambda; only meaningful when --hardware-ownership rented",
    )
    golden.add_argument(
        "--hardware-rate-usd-per-hour",
        type=float,
        default=None,
    )
    golden.add_argument(
        "--environment-note",
        default="unknown",
        help="free text, e.g. 'RunPod pod, on-demand, deleted after run'",
    )
    golden.add_argument(
        "--no-result-store",
        action="store_true",
        help="don't check for or persist to the result store for this run",
    )

    from .evalcompare import LIVE_EPILOG, add_live_arguments
    compare = subparsers.add_parser(
        "compare",
        help="compare two saved reports offline, or two live endpoints on your own "
        "prompts (--baseline/--candidate/--eval-set): should you switch models?",
        epilog=LIVE_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    compare.add_argument(
        "reports",
        nargs="*",
        type=Path,
        help="two saved reports, or six ordered B1 C1 B2 C2 B3 C3 reports for the golden protocol",
    )
    compare.add_argument("--output", type=Path, default=DEFAULT_COMPARE_OUTPUT)
    add_live_arguments(compare)

    golden_report = subparsers.add_parser(
        "golden-report",
        help="generate self-contained HTML report from golden protocol artifacts",
        description=(
            "Transform golden protocol artifacts (golden.json + position reports) into "
            "a self-contained HTML deliverable answering: should you change this setting "
            "(yes/no), what it's worth (throughput + cost), what was tested, why believe it, "
            "and what this doesn't prove. Single file, no external assets, opens offline."
        ),
    )
    golden_report.add_argument(
        "--golden-dir",
        type=Path,
        required=True,
        help="directory containing golden.json and position reports (B1.json, C1.json, etc.)",
    )
    golden_report.add_argument(
        "--output",
        type=Path,
        required=True,
        help="output HTML file path",
    )
    golden_report.add_argument(
        "--gpu-hourly-rate",
        type=float,
        help="GPU hourly rate in dollars for cost calculation (e.g., 1.39 for A100 80GB)",
    )
    golden_report.add_argument(
        "--operator-name",
        default="Throttle",
        help="operator name for report footer (default: 'Throttle')",
    )
    golden_report.add_argument(
        "--operator-email",
        default="kush@throttle-pro.com",
        help="operator email for report footer (default: 'kush@throttle-pro.com')",
    )

    # throttle check: repeat-use cost drift check (see throttle/check.py)
    check = subparsers.add_parser(
        "check",
        help="after a serving change, measure $/M tokens and compare with the last check",
        description=CHECK_DESCRIPTION,
        epilog=CHECK_EPILOG,
    )
    add_check_arguments(check)

    # throttle savings: verified-savings statement from two recorded checks
    from .savings import SAVINGS_DESCRIPTION, SAVINGS_EPILOG, add_savings_arguments
    savings = subparsers.add_parser(
        "savings",
        help="conservative, auditable verified-savings statement from two recorded checks",
        description=SAVINGS_DESCRIPTION,
        epilog=SAVINGS_EPILOG,
    )
    add_savings_arguments(savings)

    from .autotune import AUTOTUNE_EPILOG, add_autotune_arguments
    autotune = subparsers.add_parser(
        "autotune",
        help="find the cheapest serving config that is verifiably cheaper and passes "
        "OUTPUT CHANGED (free, runs locally)",
        description=textwrap.fill(
            "Searches serving configs: restarts your server with each one, measures "
            "$/M tokens like `throttle check` against a calibrated baseline, rejects "
            "OUTPUT CHANGED and anything inside the noise, and prints the best config, "
            "the $/M before and after, the projected monthly saving and a ready-to-paste "
            "launch command. Free and fully local. --dry-run simulates the "
            "server (no GPU needed).", width=79,
        ),
        epilog=AUTOTUNE_EPILOG,
    )
    add_autotune_arguments(autotune)

    from .coldstart import COLDSTART_DESCRIPTION, COLDSTART_EPILOG, add_coldstart_arguments
    coldstart = subparsers.add_parser(
        "coldstart",
        help="launch your server, time until it answers and until its first token; "
        "compare engines with repeated --cmd/--label",
        description=COLDSTART_DESCRIPTION,
        epilog=COLDSTART_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_coldstart_arguments(coldstart)

    from .gate import GATE_DESCRIPTION, add_gate_arguments
    gate = subparsers.add_parser(
        "gate",
        help="CI cost gate: exit 1 if a candidate check is MORE EXPENSIVE or OUTPUT CHANGED",
        description=GATE_DESCRIPTION,
    )
    add_gate_arguments(gate)

    ui = subparsers.add_parser(
        "ui",
        help="open the Throttle Console: your check history, verdicts and docs in a local web UI",
        description=(
            "Serves the Throttle Console on 127.0.0.1: every `throttle check` you have run, "
            "with its $/M tokens, 95% CI and verdict, side-by-side comparisons, endpoint "
            "trends, CI setup and the docs. Reads the local check history only; sends no "
            "traffic to any endpoint and loads nothing from the internet."
        ),
    )
    ui.add_argument("--port", type=int, default=8787, metavar="PORT",
                    help="port to serve on (default: 8787)")
    ui.add_argument("--host", default="127.0.0.1", metavar="HOST",
                    help="address to bind (default: 127.0.0.1; anything else exposes your history)")
    ui.add_argument("--history-dir", metavar="PATH",
                    help="check history directory (default: $THROTTLE_CHECK_HISTORY_DIR or ~/.throttle/checks)")
    ui.add_argument("--no-open", action="store_true",
                    help="don't open a browser window")

    watch = subparsers.add_parser(
        "watch",
        help="read vLLM /metrics and report cost per million tokens (no requests sent)",
        description=(
            "Passively reads vLLM /metrics (Prometheus text format) and translates "
            "throughput into dollars per million tokens. Nothing enters the request path. "
            "Requires --gpu-rate-per-hour. Refuses to print a cost figure when generation "
            "throughput is unavailable."
        ),
    )
    watch.add_argument(
        "--metrics-url",
        default="http://localhost:8000/metrics",
        metavar="URL",
        help="vLLM /metrics endpoint (default: http://localhost:8000/metrics)",
    )
    watch.add_argument(
        "--gpu-rate-per-hour",
        "--gpu-hourly-rate",
        "--total-hourly-price",
        dest="gpu_rate_per_hour",
        type=float,
        required=True,
        metavar="DOLLARS",
        help="GPU cost in $/hr. Required — no default is honest.",
    )
    watch.add_argument(
        "--interval",
        type=float,
        default=15.0,
        metavar="SECONDS",
        help="scrape interval in seconds (default: 15)",
    )
    watch.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        metavar="N",
        help="vLLM max_num_seqs for batch fill computation (optional)",
    )
    watch.add_argument(
        "--json",
        action="store_true",
        help="emit raw JSON snapshots instead of formatted text",
    )

    proxy = subparsers.add_parser(
        "proxy",
        help="run an OpenAI-compatible caching proxy server",
        description=(
            "Start a lightweight HTTP proxy that sits in front of a real inference "
            "backend and caches responses using semantic similarity matching. "
            "Verified compatible with Ollama. Expected compatible with vLLM, SGLang, "
            "LMDeploy, and other OpenAI-compatible servers (GPU verification pending, "
            "see validation/gpu_backend_verification.sh)."
        ),
    )
    proxy.add_argument(
        "--backend-url",
        "--url",
        "--endpoint-url",
        dest="backend_url",
        required=True,
        help=(
            "backend inference server URL, with or without /v1 "
            "(e.g., http://localhost:8000; --url is accepted too)"
        ),
    )
    proxy.add_argument(
        "--host",
        default="127.0.0.1",
        help="proxy server host (default: 127.0.0.1)",
    )
    proxy.add_argument(
        "--port",
        type=int,
        default=8080,
        help="proxy server port (default: 8080)",
    )
    proxy.add_argument(
        "--enable-cache",
        action="store_true",
        help="enable semantic similarity caching",
    )
    proxy.add_argument(
        "--cache-ttl-seconds",
        type=float,
        default=3600.0,
        help="cache entry TTL in seconds (default: 3600)",
    )
    proxy.add_argument(
        "--cache-max-size",
        type=int,
        default=1000,
        help="maximum cache entries (default: 1000)",
    )
    proxy.add_argument(
        "--cache-similarity-threshold",
        type=float,
        default=0.85,
        help="Jaccard similarity threshold (0.0-1.0, default: 0.85)",
    )
    embeddings_group = proxy.add_mutually_exclusive_group()
    embeddings_group.add_argument(
        "--enable-embeddings",
        action="store_true",
        help="enable ONNX semantic embedding tier (requires embeddings extra); default: OFF",
    )
    embeddings_group.add_argument(
        "--no-embeddings",
        action="store_true",
        help="force disable embeddings even if extra is installed",
    )
    proxy.add_argument(
        "--embedding-threshold",
        type=float,
        default=0.95,
        help="semantic embedding similarity threshold (0.0-1.0, default: 0.95)",
    )
    proxy.add_argument(
        "--embedding-max-entries-scanned",
        type=int,
        default=256,
        help="maximum cache entries scanned for embedding match (default: 256)",
    )
    proxy.add_argument(
        "--backend-timeout-seconds",
        type=float,
        default=120.0,
        help="backend request timeout in seconds (default: 120.0). NOTE: this value is not evidence-based; 30 seconds risks killing cold model loads and long generations.",
    )
    proxy.add_argument(
        "--enable-session-tracking",
        action="store_true",
        help="enable agent session profiling (records to ~/.throttle/sessions.db)",
    )

    # Sessions command (agent profiling)
    from .sessions_cli import add_sessions_subcommand
    add_sessions_subcommand(subparsers)

    from .own.cli import register as register_own_commands
    register_own_commands(subparsers)

    parser._throttle_subparsers = dict(subparsers.choices)  # type: ignore[attr-defined]
    return parser


def _parser_error(parser: argparse.ArgumentParser, message: str) -> None:
    """Report a usage error under the subcommand the user actually ran."""

    command = getattr(parser, "_throttle_active_command", None)
    subparser = getattr(parser, "_throttle_subparsers", {}).get(command)
    if subparser is None:
        parser.error(message)
    print(f"{subparser.prog}: error: {message}", file=sys.stderr)
    print(f"Run '{subparser.prog} --help' for its options.", file=sys.stderr)
    raise SystemExit(EXIT_USAGE)


def _inferred_cost_model(args: argparse.Namespace) -> str:
    """Pick the billing model when --cost-model was not given.

    Only an explicitly supplied hourly price selects dedicated-hourly; a price
    is never assumed. Anything else stays ``unknown`` so the spend gate holds.
    """

    if args.cost_model is not None:
        return args.cost_model
    if args.total_hourly_price is not None or args.per_gpu_hourly_price is not None:
        return "dedicated-hourly"
    return "unknown"


def _cost_model(parser: argparse.ArgumentParser, args: argparse.Namespace) -> CostModel:
    explicit_kind = args.cost_model is not None
    kind = _inferred_cost_model(args).replace("-", "_")
    dedicated_values = (args.total_hourly_price, args.per_gpu_hourly_price)
    serverless_values = (
        args.active_second_price,
        args.max_active_workers,
        args.billed_active_seconds,
    )
    if kind == "unknown":
        for value, flag, model in (
            (args.total_hourly_price, "--total-hourly-price", "dedicated-hourly"),
            (args.per_gpu_hourly_price, "--per-gpu-hourly-price", "dedicated-hourly"),
            (args.active_second_price, "--active-second-price", "serverless-active-seconds"),
            (args.max_active_workers, "--max-active-workers", "serverless-active-seconds"),
            (args.billed_active_seconds, "--billed-active-seconds", "serverless-active-seconds"),
            (args.user_supplied_total, "--user-supplied-total", "user-supplied"),
        ):
            if value is not None:
                _parser_error(
                    parser,
                    f"price fields require their matching --cost-model: "
                    f"{flag} requires --cost-model {model}",
                )
        return CostModel()
    if kind == "dedicated_hourly" and not explicit_kind:
        print(
            "Cost model: dedicated-hourly (inferred from the hourly price you gave)",
            file=sys.stderr,
        )
    if kind == "dedicated_hourly":
        if sum(value is not None for value in dedicated_values) != 1:
            _parser_error(parser, "dedicated-hourly requires exactly one hourly price")
        if any(
            value is not None
            for value in (*serverless_values, args.user_supplied_total)
        ):
            _parser_error(
                parser, "dedicated-hourly cannot include another billing model"
            )
        alias = getattr(args, "hourly_price_flag", None)
        if (
            args.total_hourly_price is not None
            and alias in ("--gpu-hourly-rate", "--gpu-rate-per-hour")
            and args.gpus > 1
        ):
            _parser_error(
                parser,
                f"{alias} with --gpus {args.gpus} is ambiguous: is ${args.total_hourly_price:g} "
                "per GPU or for all of them? Use --per-gpu-hourly-price "
                f"{args.total_hourly_price:g} --gpus {args.gpus} (per GPU) or "
                f"--total-hourly-price {args.total_hourly_price:g} --gpus {args.gpus} (all GPUs)",
            )
        total = args.total_hourly_price
        if total is None:
            total = float(args.per_gpu_hourly_price) * args.gpus
            basis = f"${float(args.per_gpu_hourly_price):g}/hr per GPU x {args.gpus}"
        else:
            basis = "total for all GPUs"
        print(
            f"Hourly price used: ${total:.2f}/hr for {args.gpus} GPU(s) ({basis}) [ASSUMED]",
            file=sys.stderr,
        )
        return CostModel(kind=kind, total_hourly_rate=total, gpu_count=args.gpus)
    if kind == "serverless_active_seconds":
        if args.active_second_price is None or args.max_active_workers is None:
            _parser_error(
                parser,
                "serverless-active-seconds requires active-second price and max workers",
            )
        if any(
            value is not None for value in (*dedicated_values, args.user_supplied_total)
        ):
            _parser_error(
                parser, "serverless-active-seconds cannot include another billing model"
            )
        return CostModel(
            kind=kind,
            active_second_rate=args.active_second_price,
            max_active_workers=args.max_active_workers,
            billed_active_seconds=args.billed_active_seconds,
        )
    if kind == "user_supplied":
        if args.user_supplied_total is None:
            _parser_error(parser, "user-supplied requires --user-supplied-total")
        if any(value is not None for value in (*dedicated_values, *serverless_values)):
            _parser_error(parser, "user-supplied cannot include another billing model")
        return CostModel(kind=kind, user_supplied_total=args.user_supplied_total)
    raise AssertionError("argparse accepted an unknown cost kind")


def _engine_flags(
    parser: argparse.ArgumentParser, values: Sequence[str]
) -> tuple[tuple[str, str], ...]:
    parsed: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            _parser_error(parser, "--engine-flag must use NAME=VALUE")
        name, flag_value = value.split("=", 1)
        if not name or not flag_value:
            _parser_error(parser, "--engine-flag needs non-empty NAME and VALUE")
        parsed.append((name, flag_value))
    return tuple(parsed)


def _endpoint_is_loopback(url: str) -> bool:
    from urllib.parse import urlsplit

    from .benchmark import _is_loopback_host

    try:
        host = urlsplit(url.strip()).hostname
    except ValueError:
        return False
    return bool(host) and _is_loopback_host(str(host))


def _key_status(args: argparse.Namespace, url: str) -> tuple[str, str]:
    """Return (status, message) for the bearer key a traffic run would use.

    status is "set", "none_local" (no key will be sent to a local endpoint),
    or "missing" (a traffic run would refuse).
    """

    env_name = args.api_key_env
    if os.environ.get(env_name):
        return "set", f"API key: read from {env_name}"
    explicit = bool(getattr(args, "api_key_env_explicit", False))
    native = getattr(args, "backend", "native") == "native"
    if not explicit and native and _endpoint_is_loopback(url):
        return (
            "none_local",
            f"API key: none sent ({env_name} is not set and the endpoint is local; "
            "pass --api-key-env NAME to send one)",
        )
    hint = (
        "set it or pass --api-key-env NAME"
        if explicit or not native
        else "set it, pass --api-key-env NAME, or use a localhost endpoint that needs no key"
    )
    return "missing", f"{env_name} is not set or is empty ({hint})"


def _resolve_key(
    parser: argparse.ArgumentParser, args: argparse.Namespace, url: str
) -> str:
    env_name = args.api_key_env
    if not ENV_NAME_PATTERN.fullmatch(env_name):
        _parser_error(parser, "--api-key-env must be a valid environment variable name")
    status, message = _key_status(args, url)
    if status == "missing":
        _parser_error(parser, f"API key environment variable is missing or empty: {message}")
    if status == "none_local":
        print(message, file=sys.stderr)
        return ""
    return os.environ[env_name]


def _run_mode(args: argparse.Namespace) -> str:
    if args.command == "plan":
        return args.run_mode
    if args.command == "golden":
        return "benchmark"
    return args.command


def _warn_if_exploratory_sweep(args: argparse.Namespace) -> None:
    """Warn before config validation, key resolution, or benchmark traffic."""

    if _run_mode(args) != "benchmark":
        return
    requested_levels = args.request_rate or args.concurrency or [1, 4, 8]
    if len(requested_levels) > 1:
        kind = "multi-request-rate" if args.request_rate else "multi-concurrency"
        print(
            EXPLORATORY_SWEEP_WARNING.format(kind=kind),
            file=sys.stderr,
            flush=True,
        )


def _build_config(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    resolve_key: bool = True,
) -> tuple[
    RunConfig,
    tuple[tuple[dict[str, str], ...], ...],
    tuple[tuple[dict[str, str], ...], ...],
]:
    mode = _run_mode(args)
    if args.warmup_requests is not None and args.warmup_requests < 0:
        _parser_error(parser, "--warmup-requests must be non-negative")
    blocks = args.blocks if args.blocks is not None else (1 if mode == "smoke" else 3)
    requests_per_block = args.requests_per_block
    if requests_per_block is None and args.block_seconds is None:
        requests_per_block = 8 if mode == "smoke" else 67
    warmups = args.warmup_requests
    if warmups is None:
        warmups = 1 if mode == "smoke" else 3
    if args.request_rate:
        conditions = tuple(
            LoadCondition("open_loop", float(rate), args.open_loop_max_in_flight)
            for rate in args.request_rate
        )
    else:
        concurrency = args.concurrency or ([8] if args.command == "golden" else [1, 4, 8])
        conditions = tuple(
            LoadCondition("closed_loop", float(level), level) for level in concurrency
        )
    limits = SafetyLimits(
        max_requests=args.max_requests,
        max_tokens_per_request=args.max_tokens_per_request,
        max_total_requested_tokens=args.max_total_requested_tokens,
        max_elapsed_seconds=(
            args.max_elapsed_seconds
            if args.max_elapsed_seconds is not None
            else 5_400.0
            if args.command == "golden"
            else 120.0
            if mode == "smoke"
            else 900.0
        ),
        max_errors=args.max_errors,
        max_concurrency=args.max_concurrency,
        max_response_bytes=args.max_response_bytes,
        max_estimated_spend=args.max_estimated_spend,
    )
    api_key = _resolve_key(parser, args, args.url) if resolve_key else ""
    config = RunConfig(
        mode=mode,
        backend="native",
        model=args.model,
        endpoint=EndpointConfig(url=args.url, api_key=api_key),
        cost=_cost_model(parser, args),
        limits=limits,
        max_tokens=args.max_tokens,
        conditions=conditions,
        blocks=blocks,
        requests_per_block=requests_per_block,
        block_duration_seconds=args.block_seconds,
        warmup_requests_per_condition=warmups,
        request_timeout_seconds=args.timeout_seconds,
        p95_slo_ms=args.p95_slo_ms,
        ttft_slo_ms=args.ttft_slo_ms,
        seed=args.seed,
        stream=args.stream,
        cache_policy=getattr(args, "cache_policy", "unknown"),
        model_revision=getattr(args, "model_revision", "unknown"),
        image_digest=getattr(args, "image_digest", "unknown"),
        gpu=getattr(args, "gpu", "unknown"),
        gpu_fingerprint=getattr(args, "gpu_fingerprint", "unknown"),
        cuda_version=getattr(args, "cuda_version", "unknown"),
        driver_version=getattr(args, "driver_version", "unknown"),
        accelerator_backend=getattr(args, "accelerator_backend", "cuda"),
        accelerator_runtime_version=getattr(args, "accelerator_runtime_version", "unknown"),
        host_os_version=getattr(args, "host_os_version", "unknown"),
        software_environment_digest=getattr(args, "software_environment_digest", "unknown"),
        server_name=getattr(args, "server_name", "unknown"),
        server_version=getattr(args, "server_version", "unknown"),
        engine_flags=_engine_flags(parser, getattr(args, "engine_flag", [])),
        engine_flags_provenance=getattr(args, "engine_flags_provenance", "operator_attested"),
        variant=getattr(args, "variant", "unspecified"),
        sequence_position=getattr(args, "sequence_position", "unspecified"),
        allow_unknown_cost=args.allow_unknown_cost,
        allow_insecure_http=args.allow_insecure_http,
        evidence_source=getattr(args, "evidence_source", "unverified_endpoint"),
        enable_cache=getattr(args, "enable_cache", False),
        cache_ttl_seconds=getattr(args, "cache_ttl_seconds", 3600.0),
        cache_max_size=getattr(args, "cache_max_size", 1000),
        cache_similarity_threshold=getattr(args, "cache_similarity_threshold", 0.85),
    )

    try:
        prompts = load_prompts(args.prompts)
        warmup_prompts = load_prompts(args.warmup_prompts, warmup=True)
        validate_config(config, for_traffic=resolve_key)
    except (OSError, ValueError, RuntimeError) as exc:
        _parser_error(parser, str(exc))
        
    return config, prompts, warmup_prompts
        


def _atomic_write(report: Mapping[str, Any], output: Path) -> None:
    committed = _atomic_write_guarded(report, output, lambda: True)
    if not committed:  # pragma: no cover - the unconditional guard is fixed true
        raise RuntimeError("atomic_write_commit_guard_failed")


def _atomic_write_guarded(
    report: Mapping[str, Any],
    output: Path,
    commit_guard: Callable[[], bool],
) -> bool:
    """Stage and fsync JSON, then publish it only if the final guard still passes."""

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = None
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        if not commit_guard():
            return False
        os.replace(temporary, output)
        return True
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _print_plan(
    plan: Mapping[str, Any],
    *,
    runtime_declared: bool = True,
) -> None:
    print(f"Throttle plan — {str(plan['mode']).upper()} (zero traffic sent)")
    count = plan["request_count"]
    if count["exact"] is not None:
        print(f"Requests: exactly {count['exact']} including warm-ups")
    else:
        print(f"Requests: duration-bounded, up to hard cap {count['upper_bound']}")
    print(
        f"Tokens: max {plan['max_tokens_per_request']} output tokens/request; "
        f"reserved total ceiling {plan['requested_output_token_ceiling']}"
    )
    print(f"Duration hard limit: {plan['duration_limit_seconds']:.2f}s")
    estimate = plan["estimated_cost_upper_bound"]
    print(
        "Estimated cost upper bound: unavailable for this billing model"
        if estimate is None
        else f"Estimated cost upper bound: ${estimate:.6f} USD"
    )
    print(
        f"Spend ceiling: ${plan['limits']['max_estimated_spend']:.2f} USD "
        "(--max-estimated-spend; a limit, not an estimate: runs whose upper bound "
        "exceeds it are refused)"
    )
    print(f"Destination: {plan['destination']['normalized_url']}")
    runtime = plan["runtime"]
    backend = runtime["accelerator_backend"] if runtime_declared else "unknown (not declared)"
    print(
        "Runtime: "
        f"{backend} / {runtime['accelerator']} / "
        f"{runtime['accelerator_runtime_version']}"
    )
    runtime_reasons = plan["runtime_provenance_reasons"]
    print(
        "Runtime evidence: complete"
        if not runtime_reasons
        else "Runtime evidence: " + ", ".join(runtime_reasons)
    )
    if plan["traffic_preflight"]["requires_unknown_cost_acknowledgement"]:
        print(
            "Traffic preflight: blocked until a price or --allow-unknown-cost is supplied.\n"
            "  For a $/M figure, give your hourly price: --gpu-hourly-rate 1.50 "
            "(= --cost-model dedicated-hourly --total-hourly-price 1.50)\n"
            "  Or pass --allow-unknown-cost to run with no dollar figure."
        )
    elif plan["traffic_preflight"]["estimated_cost_exceeds_limit"]:
        print(
            "Traffic preflight: blocked because the estimated ceiling exceeds the spend limit"
        )
    elif plan["traffic_preflight"]["planned_requests_exceed_limit"]:
        print(
            "Traffic preflight: blocked because the planned calls exceed --max-requests"
        )
    elif plan["traffic_preflight"]["planned_tokens_exceed_limit"]:
        print(
            "Traffic preflight: blocked because requested tokens exceed the total token limit"
        )
    else:
        print("Traffic preflight: cost acknowledgement/ceiling check passed")
    print(
        "Privacy: prompts leave this process and the endpoint or intermediaries may log "
        "them. JSON reports omit endpoint URLs, keys, prompts, and responses, but retain "
        "stable workload fingerprints; hashes reveal equality and can confirm a guessed "
        "workload. Ambient proxy variables are ignored."
    )
    if plan["privacy"]["plaintext_non_loopback_transport"]:
        print(
            "PLAINTEXT WARNING: --allow-insecure-http sends bearer credentials and "
            "prompt content to this non-loopback destination without TLS."
        )
    print(plan["warning"])


def _metric(value: Any, decimals: int = 2) -> str:
    return "-" if value is None else f"{float(value):.{decimals}f}"


def _print_run(report: Mapping[str, Any], output: Path) -> None:
    mode = str(report["mode"])
    if mode == "smoke":
        print("Throttle SMOKE — SHORT SAMPLE, NON-DECISION-GRADE")
    else:
        print("Throttle BENCHMARK — repeated measured blocks")

    # Check if cache is enabled to adjust table format
    cache_enabled = report.get("run_totals", {}).get("cache_enabled", False)

    if cache_enabled:
        print(
            " condition              valid  grade   requests  GPU tok/s  cache hits   p95 e2e   p95 TTFT  SLO goodput"
        )
    else:
        print(
            " condition              valid  grade   requests  block-mean tok/s   p95 e2e   p95 TTFT  SLO goodput"
        )

    for item in report["conditions"]:
        condition = item["condition"]
        metrics = item.get("metrics") or {}
        slo = metrics.get("slo_goodput")

        if cache_enabled:
            # Show GPU throughput and cache hit count
            cache_hits = metrics.get("cache_hit_count", 0)
            print(
                f" {condition['id']:<22} "
                f"{'yes' if item.get('valid') else 'NO':>5}  "
                f"{'yes' if item.get('decision_grade') else 'no':>5}  "
                f"{item.get('request_counts', {}).get('valid', 0):>4}/"
                f"{item.get('request_counts', {}).get('attempted', 0):<4}  "
                f"{_metric(metrics.get('block_mean_output_tokens_per_second')):>9}  "
                f"{cache_hits:>10}  "
                f"{_metric((metrics.get('e2e_latency_ms') or {}).get('p95')):>9}  "
                f"{_metric((metrics.get('ttft_ms') or {}).get('p95')):>9}  "
                f"{_metric((slo or {}).get('requests_per_second')):>11}"
            )
        else:
            print(
                f" {condition['id']:<22} "
                f"{'yes' if item.get('valid') else 'NO':>5}  "
                f"{'yes' if item.get('decision_grade') else 'no':>5}  "
                f"{item.get('request_counts', {}).get('valid', 0):>4}/"
                f"{item.get('request_counts', {}).get('attempted', 0):<4}  "
                f"{_metric(metrics.get('block_mean_output_tokens_per_second')):>16}  "
                f"{_metric((metrics.get('e2e_latency_ms') or {}).get('p95')):>9}  "
                f"{_metric((metrics.get('ttft_ms') or {}).get('p95')):>9}  "
                f"{_metric((slo or {}).get('requests_per_second')):>11}"
            )
        if item.get("decision_ineligible_reasons"):
            print("   evidence note: " + ", ".join(item["decision_ineligible_reasons"]))
    best = report.get("best_tested", {})
    if best.get("available"):
        print(
            f"{str(best['field']).replace('_', ' ').title()}: {best['value']} "
            f"({best['state']}; boundary reached={str(best['boundary_reached']).lower()})."
        )
        print(f"Claim boundary: {best['claim']}.")
    else:
        print(
            f"Best tested condition unavailable: {best.get('reason', 'not evaluated')}."
        )
    if report.get("stop_reason"):
        print(f"Stopped cleanly: {report['stop_reason']}.")
    cost = report.get("cost_summary", {})
    print(
        f"Cost model: {cost.get('kind')}; total="
        f"{_metric(cost.get('total_cost'), 6)}; $/1M output="
        f"{_metric(cost.get('cost_per_million_output_tokens'), 6)}."
    )
    print(report["disclaimer"])
    print(f"Sanitized JSON report: {output}")


def _print_comparison(report: Mapping[str, Any], output: Path) -> None:
    print("Throttle saved-run comparison (no traffic sent)")
    print(f"Compatibility: {'yes' if report['compatibility']['compatible'] else 'NO'}")
    if report["compatibility"]["reasons"]:
        print("Reasons: " + ", ".join(report["compatibility"]["reasons"]))
    print(
        "Attribution: "
        f"{report.get('attribution', {}).get('state', 'unavailable')} "
        f"({report.get('attribution', {}).get('reason', 'not evaluated')})"
    )
    for condition in report["conditions"]:
        interval = condition.get("throughput_delta_percent_ci") or {}
        print(
            f" {condition['condition_id']}: {condition['state']}; throughput delta "
            f"{_metric(interval.get('estimate'))}% "
            f"(95% CI {_metric(interval.get('low'))}% to {_metric(interval.get('high'))}%)"
        )
    print(f"Outcome: {report.get('overall_outcome') or 'inconclusive'}")
    if report.get("descriptive_statistical_outcome"):
        print(
            "Descriptive statistical direction (decision-ineligible): "
            f"{report['descriptive_statistical_outcome']}"
        )
    if report.get("decision_ineligible_reasons"):
        print("Decision gate: " + ", ".join(report["decision_ineligible_reasons"]))
    print(report["disclaimer"])
    print(f"Sanitized JSON comparison: {output}")


def _print_golden(
    report: Mapping[str, Any], output: Path, *, live_session: bool = False
) -> None:
    if live_session:
        print("Throttle golden live result (six sequential measurements completed)")
    else:
        print(
            "Throttle golden protocol validation "
            "(six saved reports analyzed; no traffic sent)"
        )
    print(f"Protocol eligible: {'yes' if report['golden_protocol_eligible'] else 'NO'}")
    if report["eligibility_reasons"]:
        print("Reasons: " + ", ".join(report["eligibility_reasons"]))
    treatment = report.get("treatment")
    if (
        isinstance(treatment, Mapping)
        and treatment.get("field") == "max_num_seqs"
        and type(treatment.get("baseline_value")) is int
        and type(treatment.get("candidate_value")) is int
        and type(treatment.get("closed_loop_concurrency")) is int
    ):
        print(
            "Treatment: baseline max_num_seqs="
            f"{treatment['baseline_value']}; candidate max_num_seqs="
            f"{treatment['candidate_value']}; closed-loop concurrency "
            f"{treatment['closed_loop_concurrency']}"
        )
    for condition in report["conditions"]:
        interval = condition["throughput_delta_percent_ci"]
        print(
            f" {condition['condition_id']}: {condition['state']}; throughput delta "
            f"{_metric(interval.get('estimate'))}% "
            f"(95% CI {_metric(interval.get('low'))}% to {_metric(interval.get('high'))}%)"
        )
    print(f"Outcome: {report.get('overall_outcome') or 'inconclusive'}")
    summary = report.get("decision_summary")
    if report.get("decision_eligible") is True and isinstance(summary, Mapping):
        print(str(summary["text"]))
    print(report["disclaimer"])
    print(f"Sanitized golden artifact: {output}")


def _golden_config_flags(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> tuple[tuple[str, str], tuple[str, str], int, int]:
    baseline = _engine_flags(parser, [args.baseline_config])[0]
    candidate = _engine_flags(parser, [args.candidate_config])[0]
    try:
        baseline_value, candidate_value = parse_golden_treatment_flags(
            baseline, candidate
        )
    except GoldenTreatmentError as exc:
        _parser_error(parser, exc.code)
        raise AssertionError("argparse.error must terminate")  # pragma: no cover
    return baseline, candidate, baseline_value, candidate_value


def _print_golden_plan(plan: Mapping[str, Any], output_dir: Path) -> None:
    print("Throttle golden plan — ZERO TRAFFIC SENT")
    print("Sequence: " + " → ".join(item["position"] for item in plan["positions"]))
    treatment = plan["treatment"]
    print(
        "Treatment: baseline max_num_seqs="
        f"{treatment['baseline_value']}; candidate max_num_seqs="
        f"{treatment['candidate_value']}"
    )
    concurrency = treatment["closed_loop_concurrency"]
    print(
        "Load: closed-loop concurrency "
        f"{concurrency} at every position"
        if concurrency is not None
        else "Load: invalid; Golden requires one closed-loop concurrency"
    )
    print(
        "Demand evidence boundary: reaching this client concurrency proves "
        "sufficient offered demand, not direct server-scheduler saturation."
    )
    measurement = plan["measurement"]
    block_shape = (
        f"{measurement['blocks_per_position']} blocks × "
        f"{measurement['requests_per_block']} measured requests"
        if measurement["requests_per_block"] is not None
        else f"{measurement['blocks_per_position']} duration-bounded blocks"
    )
    print(
        f"Per position: {block_shape}; "
        f"{measurement['warmup_requests_per_position']} separate warm-ups; "
        f"max {measurement['max_tokens_per_request']} output tokens/request"
    )
    if plan["session_requests"] is None:
        print("Requests: duration-bounded; exact session count unavailable")
        print("Requested output tokens: governed by the hard session token ceiling")
    else:
        print(
            f"Requests: {plan['per_position_requests']} per position; "
            f"{plan['session_requests']} across all six positions"
        )
        print(
            f"Requested output tokens: {plan['per_position_requested_output_tokens']} "
            f"per position; {plan['session_requested_output_tokens']} session ceiling"
        )
    print(
        f"Whole-session elapsed ceiling: "
        f"{float(plan['session_duration_limit_seconds']):.2f}s"
    )
    limits = plan["limits"]
    print(
        "Safety: "
        f"max requests {limits['max_requests']}; errors {limits['max_errors']}; "
        f"in-flight {limits['max_concurrency']}; response bytes "
        f"{limits['max_response_bytes']}; request timeout "
        f"{float(measurement['request_timeout_seconds']):.2f}s"
    )
    estimate = plan["session_estimated_cost_upper_bound"]
    print(
        "Whole-session estimated cost ceiling: unavailable for this billing model"
        if estimate is None
        else f"Whole-session estimated cost ceiling: ${float(estimate):.6f} USD"
    )
    if plan["spend_limit_enforceable"]:
        print(
            f"Whole-session max estimated spend: "
            f"${float(plan['session_max_estimated_spend']):.6f} USD"
        )
    else:
        print(
            "Whole-session spend guard: NOT ENFORCEABLE for this billing model; "
            "use a provider-side cap or auto-stop."
        )
    print(f"Destination: {plan['destination']['normalized_url']}")
    print(f"Artifacts: {output_dir}/B1.json … C3.json and {output_dir}/golden.json")
    print(
        "Operator boundary: Throttle never reconfigures or restarts the server. "
        "It pauses before every position for the operator to apply and verify the "
        "required config."
    )
    print(
        "Session guard: transition and inference time share the displayed client "
        "deadline. Throttle cannot stop provider resources, so keep an independent "
        "provider-side budget/auto-stop active."
    )
    print(
        "Privacy: saved artifacts omit endpoint URLs, credentials, prompts, responses, "
        "and the raw accelerator fingerprint."
    )
    reasons = plan["preflight_reasons"]
    print(
        "Decision-grade preflight: READY"
        if not reasons
        else "Decision-grade preflight: BLOCKED — " + ", ".join(reasons)
    )


def _golden_session_artifact(
    *,
    status: str,
    reason: str,
    completed_positions: Sequence[str],
    saved_positions: Sequence[str],
    elapsed_seconds: float,
    estimated_cost: float | None,
    treatment: Mapping[str, int | str],
    run_totals: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    session_totals = dict(run_totals or {})
    session_totals["elapsed_seconds"] = elapsed_seconds
    session_totals["estimated_cost"] = estimated_cost
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": GOLDEN_SESSION_ARTIFACT_TYPE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "golden_protocol_eligible": False,
        "decision_eligible": False,
        "decision_state": "inconclusive",
        "stop_reason": reason,
        "completed_positions": list(completed_positions),
        "saved_positions": list(saved_positions),
        "treatment": {
            "field": "max_num_seqs",
            "baseline_value": int(treatment["baseline_value"]),
            "candidate_value": int(treatment["candidate_value"]),
            "closed_loop_concurrency": int(
                treatment["closed_loop_concurrency"]
            ),
        },
        "session_totals": session_totals,
        "decision_summary": None,
        "disclaimer": (
            "This partial golden session is not decision-grade and cannot support a "
            "configuration recommendation."
        ),
    }


def _timed_operator_input(prompt: str, timeout_seconds: float) -> str:
    """Read one confirmation while keeping the outer golden deadline enforceable."""

    if timeout_seconds <= 0:
        raise TimeoutError("golden_session_limit")
    replies: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

    def read() -> None:
        try:
            replies.put((True, input(prompt)))
        except BaseException as exc:
            replies.put((False, exc))

    reader = threading.Thread(target=read, name="throttle-golden-input", daemon=True)
    reader.start()
    try:
        succeeded, value = replies.get(timeout=timeout_seconds)
    except queue.Empty as exc:
        raise TimeoutError("golden_session_limit") from exc
    if not succeeded:
        if isinstance(value, BaseException):
            raise value
        raise RuntimeError("operator_input_failed")
    return str(value)


def _golden_runtime_remaining(config: RunConfig, started: float) -> float:
    elapsed = max(0.0, time.perf_counter() - started)
    remaining = config.limits.max_elapsed_seconds - elapsed
    spend_rate = config.cost.elapsed_estimate(1.0)
    if spend_rate is not None and spend_rate > 0:
        spent = config.cost.elapsed_estimate(elapsed) or 0.0
        remaining = min(
            remaining,
            (config.limits.max_estimated_spend - spent) / spend_rate,
        )
    return remaining


def _position_is_usable_for_golden(report: Mapping[str, Any]) -> bool:
    conditions = report.get("conditions", [])
    return bool(
        report.get("status") == "complete"
        and len(conditions) == 1
        and conditions[0].get("valid") is True
        and conditions[0].get("decision_grade") is True
        and report.get("run_totals", {}).get("errors") == 0
    )


async def _run_golden_position(
    config: RunConfig,
    prompts: object,
    warmup_prompts: object,
    progress: RunProgress,
    timeout_seconds: float,
    session_budget: RunBudget,
) -> dict[str, Any]:
    return await asyncio.wait_for(
        run_native(
            config,
            prompts,
            warmup_prompts,
            progress=progress,
            shared_budget=session_budget,
        ),
        timeout=timeout_seconds,
    )


def _failure_report(mode: str, code: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "status": "failed",
        "decision_eligible": False,
        "stop_reason": code,
        "conditions": [],
        "best_tested": {
            "available": False,
            "state": "not_evaluated",
            "optimum_found": False,
        },
        "operational_error": {"code": code},
        "disclaimer": "No decision can be drawn from this sanitized failure artifact.",
    }


def _handle_run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    config, prompts, warmup_prompts = _build_config(parser, args, resolve_key=True)
    progress = RunProgress()
    try:
        report = asyncio.run(
            run_native(config, prompts, warmup_prompts, progress=progress)
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        report = progress.snapshot() or _failure_report(
            config.mode, "cancelled_by_user"
        )
        report["status"] = "cancelled"
        report["decision_eligible"] = False
        report["stop_reason"] = "cancelled_by_user"
        try:
            _atomic_write(report, args.output)
        except (OSError, TypeError, ValueError, OverflowError):
            print(
                "Cancelled; sanitized partial report could not be written.",
                file=sys.stderr,
            )
            return EXIT_CANCELLED
        print(
            f"Cancelled; sanitized partial report written to {args.output}.",
            file=sys.stderr,
        )
        return EXIT_CANCELLED
    except Exception:
        report = progress.snapshot() or _failure_report(config.mode, "execution_failed")
        report["status"] = "failed"
        report["decision_eligible"] = False
        report["stop_reason"] = "execution_failed"
        try:
            _atomic_write(report, args.output)
        except (OSError, TypeError, ValueError, OverflowError):
            print(
                "Execution failed; sanitized report could not be written.",
                file=sys.stderr,
            )
            return EXIT_FAILED
        print(
            f"Execution failed; sanitized report written to {args.output}.",
            file=sys.stderr,
        )
        return EXIT_FAILED
    try:
        _atomic_write(report, args.output)
    except (OSError, TypeError, ValueError, OverflowError):
        print(
            "Measurement finished but the report could not be written.", file=sys.stderr
        )
        return EXIT_FAILED
    _print_run(report, args.output)
    if report.get("status") != "complete" or any(
        not condition.get("valid") for condition in report.get("conditions", [])
    ):
        return EXIT_FAILED
    if config.mode == "benchmark" and not report.get("decision_eligible"):
        return EXIT_INCONCLUSIVE
    return EXIT_OK


def _result_store_provenance(args: argparse.Namespace) -> Provenance | None:
    """Build Provenance from CLI args, or None if it can't be determined.
    Never guesses hardware_ownership; a run without it explicitly set just
    isn't checked against or persisted to the result store.
    """
    ownership = getattr(args, "hardware_ownership", None)
    if ownership is None:
        return None
    operator = getattr(args, "operator", None)
    if not operator:
        import getpass
        import socket

        try:
            operator = f"{getpass.getuser()}@{socket.gethostname()}"
        except OSError:
            return None
    try:
        return Provenance(
            operator=operator,
            hardware_ownership=ownership,
            environment_note=getattr(args, "environment_note", "unknown"),
            hardware_provider=getattr(args, "hardware_provider", "unknown"),
            hardware_rate_usd_per_hour=getattr(
                args, "hardware_rate_usd_per_hour", None
            ),
        )
    except ResultStoreError:
        return None


def _check_result_store_before_run(
    base: RunConfig, baseline_value: int, candidate_value: int
) -> None:
    """Look for a prior matching or near-matching result before any GPU
    time is spent. Never blocks the run either way, this is information for
    the operator to act on, not an automatic decision.
    """
    identity = {
        "model_id": base.model,
        "model_revision": base.model_revision,
        "gpu": base.gpu,
        "gpu_count": base.cost.gpu_count,
        "engine_name": base.server_name,
        "engine_version": base.server_version,
    }
    if "unknown" in identity.values() or None in identity.values():
        return
    comparison = {
        "changed_flag": "max_num_seqs",
        "baseline_value": str(baseline_value),
        "candidate_value": str(candidate_value),
        "cache_policy": base.cache_policy,
    }
    try:
        records = load_records()
    except OSError:
        return
    match = find_match(identity, comparison, records)
    if match is not None:
        print(format_match_message(match), file=sys.stderr)


def _persist_golden_result(
    args: argparse.Namespace,
    result: Mapping[str, Any],
    completed_reports: Sequence[Mapping[str, Any]],
    display_aggregate_path: Path,
    base: RunConfig,
    baseline_value: int,
    candidate_value: int,
    cost_estimate: float,
) -> None:
    """Persist a completed, decision-eligible golden result. Never raises:
    a result-store failure must not turn a successful golden run into a
    failed one, it's reported and skipped instead.
    """
    if getattr(args, "no_result_store", False):
        return
    if not result.get("decision_eligible"):
        return
    provenance = _result_store_provenance(args)
    if provenance is None:
        print(
            "Result not persisted to the result store: --hardware-ownership "
            "wasn't set, so provenance can't be determined.",
            file=sys.stderr,
        )
        return
    if not completed_reports:
        return
    manifest = completed_reports[0].get("manifest", {})
    engine = manifest.get("engine", {})
    runtime = manifest.get("runtime", {})
    model = manifest.get("model", {})
    workload = manifest.get("workload", {})
    outcome_field = None
    for condition in result.get("conditions", []):
        ci = condition.get("throughput_delta_percent_ci")
        if ci:
            outcome_field = ci
            break
    try:
        record = build_record(
            decision_eligible=True,
            decision_state=result.get("decision_state", "unknown"),
            overall_outcome=result.get("overall_outcome", "unknown"),
            throughput_delta_percent_estimate=(outcome_field or {}).get("estimate"),
            throughput_delta_percent_low=(outcome_field or {}).get("low"),
            throughput_delta_percent_high=(outcome_field or {}).get("high"),
            model_id=model.get("id", base.model),
            model_revision=model.get("immutable_revision", base.model_revision),
            gpu=runtime.get("gpu", base.gpu),
            gpu_count=base.cost.gpu_count or 1,
            gpu_fingerprint_sha256=runtime.get("gpu_fingerprint_sha256", "unknown"),
            engine_name=engine.get("server_name", base.server_name),
            engine_version=engine.get("server_version", base.server_version),
            throttle_client_backend=engine.get("backend", "unknown"),
            throttle_client_backend_version=engine.get("backend_version", "unknown"),
            cuda_version=runtime.get("cuda_version", base.cuda_version),
            driver_version=runtime.get("driver_version", base.driver_version),
            image_digest=runtime.get("image_digest", base.image_digest),
            changed_flag="max_num_seqs",
            baseline_value=str(baseline_value),
            candidate_value=str(candidate_value),
            measured_sha256=workload.get("measured_sha256", "unknown"),
            warmup_sha256=workload.get("warmup_sha256", "unknown"),
            measured_prompt_count=workload.get("measured_prompt_count"),
            warmup_prompt_count=workload.get("warmup_prompt_count"),
            seed=workload.get("seed"),
            cache_policy=workload.get("cache_policy", base.cache_policy),
            source_run_fingerprints=result.get("run_fingerprints", []),
            artifact_paths=[str(display_aggregate_path)],
            cost_usd_estimate=cost_estimate,
            result_id=hashlib.sha256(
                json.dumps(result, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest(),
            provenance=provenance,
        )
        path = append_record(record)
    except (ResultStoreError, OSError, TypeError, ValueError) as exc:
        print(f"Result not persisted to the result store: {exc}", file=sys.stderr)
        return
    print(f"Persisted to result store: {path}", file=sys.stderr)


def _handle_golden(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    (
        baseline_flag,
        candidate_flag,
        baseline_value,
        candidate_value,
    ) = _golden_config_flags(parser, args)

    # Golden protocol requires exactly one closed-loop concurrency value
    if args.request_rate is not None:
        _parser_error(
            parser,
            "Golden protocol requires closed-loop concurrency, not --request-rate. "
            "Remove --request-rate or use a different command."
        )
    if args.concurrency is not None and len(args.concurrency) != 1:
        _parser_error(
            parser,
            f"Golden protocol requires a single concurrency value, but received {len(args.concurrency)} "
            f"values: {args.concurrency}. This may come from your ~/.throttle/config.yaml. "
            f"Either set a single concurrency value in your config, or pass --concurrency N explicitly."
        )

    if args.concurrency is None:
        # Preserve the historical 1-versus-8 default while making every other
        # pair exercise at least its larger configured treatment value.
        args.concurrency = [max(baseline_value, candidate_value)]
    base, prompts, warmup_prompts = _build_config(parser, args, resolve_key=False)
    plan = build_golden_plan(
        base,
        prompts,
        warmup_prompts,
        baseline_flag=baseline_flag,
        candidate_flag=candidate_flag,
    )
    _print_golden_plan(plan, args.output_dir)
    if args.dry_run:
        return EXIT_OK
    if plan["preflight_reasons"]:
        print(
            "Golden run blocked before key resolution or traffic; fix every preflight reason.",
            file=sys.stderr,
        )
        return EXIT_USAGE
    if not getattr(args, "no_result_store", False):
        _check_result_store_before_run(base, baseline_value, candidate_value)
    display_output_dir = args.output_dir.expanduser()
    output_dir = display_output_dir.resolve()
    if output_dir.exists():
        print(
            "Golden output directory already exists; choose a new --output-dir so "
            "prior evidence is never overwritten.",
            file=sys.stderr,
        )
        return EXIT_USAGE

    # Resolve only the credential after the operator approves the plan. Reuse the
    # exact preflighted config and immutable prompt tuples so a file change cannot
    # swap the measured workload between plan and traffic.
    api_key = _resolve_key(parser, args, base.endpoint.url)
    base = replace(
        base,
        endpoint=EndpointConfig(url=base.endpoint.url, api_key=api_key),
    )
    validate_config(base, for_traffic=True)
    completed_reports: list[dict[str, Any]] = []
    completed_positions: list[str] = []
    saved_positions: list[str] = []
    session_started = time.perf_counter()
    session_budget = RunBudget(base, started=session_started)
    treatment = plan["treatment"]
    positions = golden_positions(baseline_value, candidate_value)
    aggregate_path = output_dir / "golden.json"
    display_aggregate_path = display_output_dir / "golden.json"

    def persist_session_failure(status: str, reason: str) -> int:
        elapsed = max(0.0, time.perf_counter() - session_started)
        artifact = _golden_session_artifact(
            status=status,
            reason=reason,
            completed_positions=completed_positions,
            saved_positions=saved_positions,
            elapsed_seconds=elapsed,
            estimated_cost=base.cost.elapsed_estimate(elapsed),
            treatment=treatment,
            run_totals=session_budget.public_dict(),
        )
        print(f"Golden session stopped: {reason}.", file=sys.stderr)
        try:
            _atomic_write(artifact, aggregate_path)
        except (OSError, TypeError, ValueError, OverflowError):
            print(
                "Sanitized partial session artifact could not be written; no "
                "decision-grade result was produced.",
                file=sys.stderr,
            )
        else:
            print(
                f"Sanitized partial session artifact: {display_aggregate_path}",
                file=sys.stderr,
            )
        return EXIT_CANCELLED if status == "cancelled" else EXIT_FAILED

    try:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.mkdir(output_dir, mode=0o700)
        initial = _golden_session_artifact(
            status="partial",
            reason="awaiting_first_position",
            completed_positions=(),
            saved_positions=(),
            elapsed_seconds=0.0,
            estimated_cost=base.cost.elapsed_estimate(0.0),
            treatment=treatment,
            run_totals=session_budget.public_dict(),
        )
        _atomic_write(initial, aggregate_path)
    except FileExistsError:
        print(
            "Golden output directory was created concurrently; choose a new "
            "--output-dir.",
            file=sys.stderr,
        )
        return EXIT_USAGE
    except (OSError, TypeError, ValueError, OverflowError):
        print(
            "Golden output directory could not be reserved and verified before "
            "traffic; no requests were sent.",
            file=sys.stderr,
        )
        return EXIT_FAILED

    try:
        for position, variant, max_num_seqs in positions:
            expected = f"{position} verified"
            print()
            print(
                f"Position {position} — operator action required: apply {variant} config "
                f"max_num_seqs={max_num_seqs}, restart if needed, and verify the "
                "effective runtime value."
            )
            remaining = _golden_runtime_remaining(base, session_started)
            if remaining <= 0:
                return persist_session_failure("stopped", "golden_session_limit")
            try:
                confirmation = _timed_operator_input(
                    f'Type "{expected}" to run this position: ', remaining
                )
            except TimeoutError:
                return persist_session_failure("stopped", "golden_session_limit")
            if confirmation.strip() != expected:
                return persist_session_failure(
                    "stopped", f"operator_confirmation_failed_{position}"
                )
            remaining = _golden_runtime_remaining(base, session_started)
            if remaining <= 0:
                return persist_session_failure("stopped", "golden_session_limit")
            position_config = golden_position_config(
                base,
                position=position,
                variant=variant,
                max_num_seqs=max_num_seqs,
            )
            progress = RunProgress()
            try:
                report = asyncio.run(
                    _run_golden_position(
                        position_config,
                        prompts,
                        warmup_prompts,
                        progress,
                        remaining,
                        session_budget,
                    )
                )
            except TimeoutError:
                report = progress.snapshot() or _failure_report(
                    "benchmark", "golden_session_limit"
                )
                report["status"] = "stopped"
                report["decision_eligible"] = False
                report["stop_reason"] = "golden_session_limit"
            except (KeyboardInterrupt, asyncio.CancelledError):
                report = progress.snapshot()
                if report is not None:
                    report["status"] = "cancelled"
                    report["decision_eligible"] = False
                    report["stop_reason"] = "cancelled_by_user"
                    try:
                        _atomic_write(report, output_dir / f"{position}.json")
                    except (OSError, TypeError, ValueError, OverflowError):
                        pass
                    else:
                        saved_positions.append(position)
                return persist_session_failure("cancelled", "cancelled_by_user")
            except Exception:
                report = progress.snapshot() or _failure_report(
                    "benchmark", "execution_failed"
                )
                report["status"] = "failed"
                report["decision_eligible"] = False
                report["stop_reason"] = "execution_failed"

            position_path = output_dir / f"{position}.json"
            display_position_path = display_output_dir / f"{position}.json"
            try:
                _atomic_write(report, position_path)
            except (OSError, TypeError, ValueError, OverflowError):
                return persist_session_failure(
                    "stopped", f"position_{position}_report_write_failed"
                )
            saved_positions.append(position)
            _print_run(report, display_position_path)
            if not _position_is_usable_for_golden(report):
                return persist_session_failure(
                    "stopped", f"position_{position}_not_decision_grade"
                )
            completed_reports.append(report)
            completed_positions.append(position)
            if _golden_runtime_remaining(base, session_started) <= 0:
                return persist_session_failure("stopped", "golden_session_limit")
    except (KeyboardInterrupt, EOFError):
        return persist_session_failure("cancelled", "cancelled_by_user")
    except Exception:
        return persist_session_failure("stopped", "golden_orchestration_failed")

    try:
        result = validate_golden_sequence(completed_reports)
        if _golden_runtime_remaining(base, session_started) <= 0:
            return persist_session_failure("stopped", "golden_session_limit")
        elapsed = max(0.0, time.perf_counter() - session_started)
        result["session_totals"] = {
            **session_budget.public_dict(),
            "elapsed_seconds": elapsed,
            "estimated_cost": base.cost.elapsed_estimate(elapsed),
            "completed_positions": completed_positions,
        }
        committed = _atomic_write_guarded(
            result,
            aggregate_path,
            lambda: _golden_runtime_remaining(base, session_started) > 0,
        )
        if not committed:
            return persist_session_failure("stopped", "golden_session_limit")
    except (KeyboardInterrupt, asyncio.CancelledError):
        return persist_session_failure("cancelled", "cancelled_by_user")
    except Exception:
        return persist_session_failure("stopped", "golden_validation_or_write_failed")
    _print_golden(result, display_aggregate_path, live_session=True)
    if not result["golden_protocol_eligible"]:
        return EXIT_USAGE
    _persist_golden_result(
        args,
        result,
        completed_reports,
        display_aggregate_path,
        base,
        baseline_value,
        candidate_value,
        result.get("session_totals", {}).get("estimated_cost", 0.0),
    )
    return EXIT_OK if result["decision_eligible"] else EXIT_INCONCLUSIVE


def _handle_golden_report(args: argparse.Namespace) -> int:
    """Generate self-contained HTML report from golden protocol artifacts."""
    from throttle.golden_report import generate_html_report
    import sys

    golden_dir = args.golden_dir.expanduser().resolve()

    # Check golden_dir exists
    if not golden_dir.exists():
        print(f"Error: Directory not found: {golden_dir}", file=sys.stderr)
        return EXIT_FAILED

    if not golden_dir.is_dir():
        print(f"Error: Not a directory: {golden_dir}", file=sys.stderr)
        return EXIT_FAILED

    # Find golden.json
    golden_json_path = golden_dir / "golden.json"
    if not golden_json_path.exists():
        print(f"Error: golden.json not found in {golden_dir}", file=sys.stderr)
        return EXIT_FAILED

    # Find position reports (B1, B2, B3, C1, C2, C3)
    position_files = []
    for pos in ['B1', 'B2', 'B3', 'C1', 'C2', 'C3']:
        pos_file = golden_dir / f"{pos}.json"
        if not pos_file.exists():
            print(f"Error: {pos}.json not found in {golden_dir}", file=sys.stderr)
            return EXIT_FAILED
        position_files.append(pos_file)

    # Generate report
    try:
        generate_html_report(
            golden_json_path=golden_json_path,
            position_reports=position_files,
            output_path=args.output.expanduser().resolve(),
            gpu_hourly_rate=args.gpu_hourly_rate,
            operator_name=args.operator_name,
            operator_email=args.operator_email,
        )
        print(f"Golden protocol report generated: {args.output}")
        return EXIT_OK
    except Exception as e:
        print(f"Error generating report: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return EXIT_FAILED


def _handle_proxy(args: argparse.Namespace) -> int:
    """Start the caching proxy server."""
    try:
        import uvicorn
        from .proxy import create_app
        from . import embeddings
    except ImportError as e:
        print(
            f"Error: FastAPI and uvicorn are required for proxy mode: {e}",
            file=sys.stderr,
        )
        print(
            "Install with: pip install 'fastapi>=0.115.0' 'uvicorn[standard]>=0.30.0'",
            file=sys.stderr,
        )
        return EXIT_FAILED

    # Resolve embeddings: only enabled if explicitly requested with --enable-embeddings
    enable_embeddings_resolved = False
    if args.no_embeddings:
        enable_embeddings_resolved = False
    elif args.enable_embeddings:
        enable_embeddings_resolved = True

    # The proxy appends /v1/chat/completions itself; accept a /v1 base too.
    args.backend_url = _server_root_url(args.backend_url)
    print(f"Starting Throttle proxy server on {args.host}:{args.port}")
    print(f"Backend: {args.backend_url}")
    print(f"Backend timeout: {args.backend_timeout_seconds}s")
    print(f"Cache enabled: {args.enable_cache}")
    print(f"Session tracking: {args.enable_session_tracking}")
    if args.enable_session_tracking:
        print(f"  Database: ~/.throttle/sessions.db")
        print(f"  View sessions: throttle sessions")
    if args.enable_cache:
        print(f"  TTL: {args.cache_ttl_seconds}s")
        print(f"  Max size: {args.cache_max_size}")
        print(f"  Similarity threshold: {args.cache_similarity_threshold}")

        # Startup diagnostics for embeddings
        if enable_embeddings_resolved and embeddings.EMBEDDINGS_AVAILABLE:
            if embeddings.is_model_available():
                # State a: embeddings active
                print(f"  Embeddings: ACTIVE (model: sentence-transformers/all-MiniLM-L6-v2)")
            else:
                # State a2: packages present but the model could not load
                print(
                    "  Embeddings: REQUESTED BUT MODEL UNAVAILABLE "
                    f"({embeddings.get_load_error()}) - Jaccard only"
                )
        elif enable_embeddings_resolved and not embeddings.EMBEDDINGS_AVAILABLE:
            # State b: requested but extra missing
            print(f"  Embeddings: REQUESTED BUT UNAVAILABLE")
            print(f"    Install with: pip install 'throttle-pro[embeddings]'")
        elif not enable_embeddings_resolved:
            # State c: disabled by explicit request or not applicable
            if args.no_embeddings:
                print(f"  Embeddings: DISABLED (explicit --no-embeddings)")
            else:
                print(f"  Embeddings: OFF (Jaccard lexical matching only)")
    print()
    print(f"Health endpoint: http://{args.host}:{args.port}/health")
    print(f"Chat completions: http://{args.host}:{args.port}/v1/chat/completions")
    print()
    print("Press Ctrl+C to stop")
    print()

    app = create_app(
        backend_url=args.backend_url,
        enable_cache=args.enable_cache,
        cache_ttl_seconds=args.cache_ttl_seconds,
        cache_max_size=args.cache_max_size,
        cache_similarity_threshold=args.cache_similarity_threshold,
        enable_embeddings=enable_embeddings_resolved,
        embedding_threshold=args.embedding_threshold,
        embedding_max_entries_scanned=args.embedding_max_entries_scanned,
        backend_timeout_seconds=args.backend_timeout_seconds,
        enable_session_tracking=args.enable_session_tracking,
    )

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        return EXIT_OK
    except KeyboardInterrupt:
        print("\nProxy server stopped")
        return EXIT_OK
    except Exception as e:
        print(f"Proxy server error: {e}", file=sys.stderr)
        return EXIT_FAILED


def _handle_watch(args) -> int:
    """Stream vLLM cost metrics to stdout. Nothing enters the request path."""
    from throttle.advisor import stream_metrics

    print(f"Watching {args.metrics_url} every {args.interval:.0f}s")
    print(f"GPU rate: ${args.gpu_rate_per_hour:.2f}/hr")
    print("Press Ctrl+C to stop.")
    print()

    try:
        first_snapshot = True
        for snap in stream_metrics(
            args.metrics_url,
            args.gpu_rate_per_hour,
            interval_seconds=args.interval,
            max_num_seqs=args.max_num_seqs,
        ):
            # On first snapshot, check for connection failure and exit immediately
            if first_snapshot:
                first_snapshot = False
                # Check for connection error (refusal with figure="all")
                conn_errors = [r for r in snap.refusals if r.get("figure") == "all"]
                if conn_errors:
                    print(f"ERROR: Cannot reach vLLM metrics endpoint at {args.metrics_url}")
                    print(f"       {conn_errors[0]['reason']}")
                    print()
                    print("This usually means:")
                    print("  - No vLLM server is running")
                    print("  - The server is at a different URL")
                    print()
                    print("To fix:")
                    print("  1. Start a vLLM server first:")
                    print("     vllm serve <model>")
                    print("     (watch needs a vLLM Prometheus /metrics endpoint;")
                    print("      Ollama does not expose one)")
                    print("  2. Then run 'throttle watch' again")
                    print("  3. Or specify a different endpoint:")
                    print(f"     throttle watch --metrics-url http://host:port/metrics --gpu-rate-per-hour {args.gpu_rate_per_hour}")
                    return EXIT_FAILED

            if args.json:
                import json as _json
                print(_json.dumps(snap.to_dict()))
                import sys as _sys
                _sys.stdout.flush()
            else:
                _render_watch_snap(snap)
    except KeyboardInterrupt:
        print("\nStopped.")
    return EXIT_OK


def _render_watch_snap(snap) -> None:
    """Format a CostSnapshot for human display."""
    import time as _time
    print(f"--- {_time.strftime('%H:%M:%S')} ---")
    if snap.generation_throughput_toks_per_sec is not None:
        print(f"  Gen throughput : {snap.generation_throughput_toks_per_sec:.1f} tok/s")
    if snap.num_requests_running is not None:
        fill = f" ({snap.batch_fill:.0%} fill)" if snap.batch_fill is not None else ""
        print(f"  Requests       : {snap.num_requests_running:.0f} running{fill}")
    if snap.cost_per_million_tokens is not None:
        print(f"  Cost           : ${snap.cost_per_hour:.2f}/hr  "
              f"${snap.cost_per_million_tokens:.4f}/Mtok")
    for r in snap.refusals:
        print(f"  ⚠ {r['figure']}: {r['reason'][:80]}")
    print()


WELCOME_TAGLINE = "Throttle cuts what your LLM inference costs on your own GPUs."
WELCOME_STEPS = (
    ("Install Ollama: ollama pull llama3.2:3b", "or use your vLLM / SGLang OpenAI-compatible URL.", ()),
    ("Run once to calibrate (a few minutes on a laptop):", None, (
        "throttle check --url http://localhost:11434 \\",
        "    --model llama3.2:3b --gpu-hourly-rate 1.50",
    )),
    ("Change one setting, then run it again for a verdict.", None, ()),
    ("For agent traffic (multi-turn sessions, tool results):", None, (
        "throttle check --workload agent \\",
        "    --url http://localhost:11434 \\",
        "    --model llama3.2:3b --gpu-hourly-rate 1.50",
    )),
)


def welcome_screen() -> str:
    """`throttle` with no arguments: a short branded first-run screen."""

    from . import style as style_module

    st = style_module.Style()
    rows: list = [(WELCOME_TAGLINE, ("1",)), ""]
    for number, (first, second, commands) in enumerate(WELCOME_STEPS, 1):
        rows.append(f"{st.accent(f'{number}.')} {first}")
        if second:
            rows.append(f"   {second}")
        rows.extend(f"   {st.paint(command, '36')}" for command in commands)
    return "\n".join([
        st.panel(rows, title=f"Throttle {__version__}"),
        st.dim("No server yet? throttle check --help. All commands: throttle --help"),
    ])


def main(argv: Sequence[str] | None = None) -> int:
    from .config import load_config, apply_config_defaults

    parser = build_parser()

    # Load config file and apply as defaults (CLI flags will override)
    config = load_config()
    apply_config_defaults(parser, config)

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not raw_argv:
        print(welcome_screen())
        return EXIT_OK

    args = parser.parse_args(raw_argv)
    parser._throttle_active_command = args.command  # type: ignore[attr-defined]
    if args.command in {"plan", "smoke", "benchmark"}:
        _warn_if_exploratory_sweep(args)
    if args.command == "plan":
        config, prompts, warmup_prompts = _build_config(parser, args, resolve_key=False)
        plan = build_plan(config, prompts, warmup_prompts)
        _print_plan(
            plan,
            runtime_declared=any(
                token == "--accelerator-backend" or token.startswith("--accelerator-backend=")
                for token in raw_argv
            ),
        )
        key_state, key_message = _key_status(args, args.url)
        if key_state == "missing":
            print(f"API key: {key_message}; smoke/benchmark will refuse to send traffic")
        else:
            print(key_message)
        return EXIT_OK
    if args.command in {"smoke", "benchmark"}:
        return _handle_run(parser, args)
    if args.command == "golden":
        return _handle_golden(parser, args)
    if args.command == "golden-report":
        return _handle_golden_report(args)
    if args.command == "compare":
        from .evalcompare import handle_live_compare, is_live
        if is_live(args):
            return handle_live_compare(args, raw_argv)
        if args.out is not None:
            # Before --out existed, argparse expanded "--out" to "--output";
            # keep that working for saved-report comparisons.
            args.output = args.out
        if not args.reports:
            parser.error(
                "compare needs two saved reports, or --baseline URL --candidate URL "
                "--eval-set FILE for a live comparison"
            )
        # Original benchmark report comparison
        try:
            if len(args.reports) not in {2, 6}:
                parser.error(
                    "compare requires exactly two reports, or six ordered golden-protocol reports"
                )
            reports = [load_report(path) for path in args.reports]
            report = (
                compare_reports(reports[0], reports[1])
                if len(reports) == 2
                else validate_golden_sequence(reports)
            )
            _atomic_write(report, args.output)
        except (ComparisonInputError, OSError, TypeError, ValueError, ArithmeticError):
            print("Saved reports could not be compared safely.", file=sys.stderr)
            return EXIT_FAILED
        if len(args.reports) == 6:
            _print_golden(report, args.output)
            if not report["golden_protocol_eligible"]:
                return EXIT_USAGE
            return (
                EXIT_OK
                if report["decision_state"] == "supported"
                else EXIT_INCONCLUSIVE
            )
        _print_comparison(report, args.output)
        if not report["compatibility"]["compatible"]:
            return EXIT_USAGE
        return EXIT_OK if report.get("decision_eligible") else EXIT_INCONCLUSIVE
    if args.command == "proxy":
        return _handle_proxy(args)
    if args.command == "watch":
        return _handle_watch(args)
    if args.command == "check":
        return handle_check(args)
    if args.command == "savings":
        from .savings import handle_savings
        return handle_savings(args)
    if args.command == "autotune":
        from .autotune import handle_autotune
        return handle_autotune(args)
    if args.command == "coldstart":
        from .coldstart import handle_coldstart
        return handle_coldstart(args)
    if args.command == "gate":
        from .gate import handle_gate
        return handle_gate(args)
    if args.command == "ui":
        from .check import history_dir as check_history_dir
        from .console import run_ui
        return run_ui(check_history_dir(args), host=args.host, port=args.port, open_browser=not args.no_open)
    if args.command == "sessions":
        from .sessions_cli import handle_sessions_command
        return handle_sessions_command(args)
    from .own.cli import commands as own_commands, dispatch as own_dispatch
    if args.command in own_commands():
        return own_dispatch(args)
    parser.error("a subcommand is required")
    return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
