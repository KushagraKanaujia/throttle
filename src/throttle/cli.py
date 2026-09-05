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
import stat
import sys
import tempfile
import threading
import time
import unicodedata
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
    normalize_chat_completions_url,
    run_native,
    validate_config,
)
from .compare import (
    ComparisonInputError,
    compare_reports,
    load_report,
)
from .diagnose import (
    handle_diagnose,
    print_diagnose,
)
from .experimental_tuning import (
    ExperimentalTuningError,
    prepare_metrics_collector,
    run_experimental_tuning,
    validated_experimental_envelope,
    validated_experimental_report,
    validate_experimental_config,
    validate_experimental_run_report,
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
DEFAULT_EXPERIMENTAL_OUTPUT = Path("throttle-experimental-tuning.json")
DEFAULT_DIAGNOSE_OUTPUT = Path("throttle-diagnose.json")
ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_INCONCLUSIVE = 3
EXIT_CANCELLED = 130

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


def _add_endpoint_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model", required=True, help="model identifier sent to the API"
    )
    parser.add_argument(
        "--url", required=True, help="base URL or exact chat-completions route"
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        metavar="NAME",
        help="environment variable containing the bearer key",
    )
    parser.add_argument(
        "--allow-insecure-http",
        action="store_true",
        help="allow plaintext HTTP away from loopback (unsafe; recorded in manifest)",
    )


def _add_cost_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cost-model",
        choices=(
            "unknown",
            "dedicated-hourly",
            "serverless-active-seconds",
            "user-supplied",
        ),
        default="unknown",
    )
    parser.add_argument("--gpus", type=_positive_int, default=1)
    dedicated = parser.add_mutually_exclusive_group()
    dedicated.add_argument("--total-hourly-price", type=_positive_float)
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
            "5400 for the six-position golden session, 60 for diagnose)"
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
    parser.add_argument("--backend", choices=("native", "guidellm"), default="native")
    parser.add_argument(
        "--guidellm-prompt-tokens",
        type=_positive_int,
        help="required with --backend guidellm; synthetic_text prompt length",
    )
    parser.add_argument(
        "--guidellm-executable",
        default="guidellm",
        help="GuideLLM 0.7.3 executable (version is verified before traffic)",
    )
    parser.add_argument(
        "--allow-guidellm-validation-gaps",
        action="store_true",
        help="acknowledge cross-check-only completion/response-size limitations",
    )
    _add_cost_options(parser)
    _add_workload_options(parser)
    _add_safety_options(parser)
    _add_manifest_options(parser)
    _add_cache_options(parser)


def _get_api_key(args: argparse.Namespace) -> str | None:
    """Get API key from args or environment, return None if not set."""
    if hasattr(args, 'api_key') and args.api_key:
        return args.api_key
    return os.environ.get('OPENAI_API_KEY')


def _build_headers(api_key: str | None) -> dict[str, str]:
    """Build HTTP headers with optional Authorization bearer token."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="throttle",
        description="Safety-first measurements for an existing OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

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

    diagnose = subparsers.add_parser(
        "diagnose",
        help="pre-flight bottleneck regime classification (not decision-grade)",
        description=(
            "Run a short probe at multiple concurrency levels and classify "
            "the dominant bottleneck regime. Use this before throttle golden "
            "to identify which config dimensions are worth testing."
        ),
    )
    _add_endpoint_options(diagnose)
    diagnose.add_argument("--backend", choices=("native",), default="native")
    _add_cost_options(diagnose)
    _add_workload_options(diagnose)
    _add_safety_options(diagnose)
    diagnose.add_argument(
        "--output", type=Path, default=DEFAULT_DIAGNOSE_OUTPUT
    )

    experimental = subparsers.add_parser(
        "experimental-tuning",
        help="run an opt-in, suggestion-only server-metrics analysis",
        description=(
            "Same inference deployment; no unrelated inference traffic. Both "
            "facts are operator-attested, not independently proven. Run one "
            "native closed-loop smoke workload while polling one explicit vLLM "
            "metrics exporter. Output is suggestion-only: it never changes "
            "configuration, decision eligibility, or Golden eligibility."
        ),
        epilog=(
            "Defaults: one smoke block with 201 measured requests, 3 separate "
            "warm-ups, and a 900-second traffic-run ceiling, plus bounded "
            "exporter-scrape and processing overhead. Supply exactly one "
            "--concurrency and runtime-effective max_num_seqs and "
            "max_num_batched_tokens engine flags."
        ),
    )
    _add_run_options(experimental)
    experimental.add_argument(
        "--metrics-url",
        required=True,
        help=(
            "explicit vLLM Prometheus endpoint; no credentials, redirects, "
            "ambient proxies, or non-loopback plaintext are allowed"
        ),
    )
    experimental.add_argument(
        "--attest-same-deployment-exclusive-metrics",
        action="store_true",
        help=(
            "attest that this exporter belongs to the same inference "
            "deployment and that no unrelated inference traffic reaches it "
            "during the sampled window; neither fact is independently proven"
        ),
    )
    experimental.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "ordinary schema-2.0 non-decision-grade smoke report; parent "
            "directory must already exist and the create-only file must not exist"
        ),
    )
    experimental.add_argument(
        "--experimental-output",
        type=Path,
        default=DEFAULT_EXPERIMENTAL_OUTPUT,
        help=(
            "separate bound safety-validation envelope; parent directory "
            "must already exist and the create-only file must not exist"
        ),
    )

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

    compare = subparsers.add_parser(
        "compare", help="compare two saved benchmark reports without network traffic"
    )
    compare.add_argument(
        "reports",
        nargs="+",
        type=Path,
        help="two saved reports, or six ordered B1 C1 B2 C2 B3 C3 reports for the golden protocol",
    )
    compare.add_argument("--output", type=Path, default=DEFAULT_COMPARE_OUTPUT)

    report = subparsers.add_parser(
        "report", help="generate HTML report with chart comparing measure outputs"
    )
    report.add_argument(
        "reports",
        nargs=2,
        type=Path,
        help="two measure output JSON files to compare",
    )
    report.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output HTML file path",
    )

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
        default="kushthrottle@gmail.com",
        help="operator email for report footer (default: 'kushthrottle@gmail.com')",
    )

    demo = subparsers.add_parser(
        "demo",
        help="run a fast simulator demo comparing baseline vs tuned configuration",
        description=(
            "Generate a sample workload, simulate vLLM-like continuous batching inference "
            "for two configurations (baseline and tuned), and print a side-by-side cost comparison. "
            "Runs entirely locally with no GPU or network required. Completes in under 5 minutes."
        ),
    )
    demo.add_argument(
        "--light-load",
        action="store_true",
        help="use a light workload that does not saturate either configuration (demonstrates NO SIGNIFICANT DIFFERENCE path)",
    )

    cost = subparsers.add_parser(
        "cost",
        help="measure cost per million tokens against a live endpoint",
        description=(
            "Send a small workload to an OpenAI-compatible endpoint, measure actual "
            "throughput and timing, and calculate dollars per million tokens. "
            "Requires a running inference server."
        ),
    )
    cost.add_argument(
        "--endpoint-url",
        required=True,
        help="inference server URL (e.g., http://localhost:8000/v1)",
    )
    cost.add_argument(
        "--model",
        default="default",
        help="model name to request (default: 'default')",
    )
    cost.add_argument(
        "--gpu-hourly-rate",
        type=float,
        required=True,
        help="GPU hourly rate in dollars (e.g., 1.50 for A100 spot pricing)",
    )
    cost.add_argument(
        "--num-requests",
        type=int,
        default=20,
        help="number of test requests to send (default: 20)",
    )
    cost.add_argument(
        "--api-key",
        help="API key for authentication (also reads OPENAI_API_KEY env var)",
    )


    validate_sim = subparsers.add_parser(
        "validate-sim",
        help=argparse.SUPPRESS,
        description=(
            "Compare simulator predictions to actual measurements from a live endpoint. "
            "This helps validate simulator assumptions and identify which parameters need "
            "adjustment for your specific hardware setup."
        ),
    )
    validate_sim.add_argument(
        "--endpoint-url",
        required=True,
        help="inference server URL to validate against",
    )
    validate_sim.add_argument(
        "--model",
        default="default",
        help="model name (default: 'default')",
    )
    validate_sim.add_argument(
        "--gpu-hourly-rate",
        type=float,
        required=True,
        help="GPU hourly rate in dollars",
    )
    validate_sim.add_argument(
        "--api-key",
        help="API key for authentication (also reads OPENAI_API_KEY env var)",
    )

    measure = subparsers.add_parser(
        "measure",
        help="measure cost per million tokens with statistical rigor",
        description=(
            "Run repeated measurements against a live endpoint to measure cost per "
            "million tokens with confidence intervals. Results are saved to JSON for "
            "later comparison. The operator is responsible for server configuration - "
            "Throttle does not restart or reconfigure the server."
        ),
    )
    measure.add_argument(
        "--endpoint-url",
        required=True,
        help="inference server URL to measure",
    )
    measure.add_argument(
        "--model",
        default="default",
        help="model name (default: 'default')",
    )
    measure.add_argument(
        "--gpu-hourly-rate",
        type=float,
        required=True,
        help="GPU hourly rate in dollars",
    )
    measure.add_argument(
        "--label",
        required=True,
        help="label for this measurement (used as output filename)",
    )
    measure.add_argument(
        "--repeat",
        type=int,
        default=10,
        help="number of times to repeat the workload (default: 10, runtime ~100 seconds at default arrival rate)",
    )
    measure.add_argument(
        "--arrival-rate",
        type=float,
        default=10.0,
        help="arrival rate in requests per second (default: 10.0)",
    )
    measure.add_argument(
        "--num-requests",
        type=int,
        default=100,
        help="number of requests per trial (default: 100)",
    )
    measure.add_argument(
        "--note",
        help="optional note about server configuration",
    )
    measure.add_argument(
        "--api-key",
        help="API key for authentication (also reads OPENAI_API_KEY env var)",
    )

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
        required=True,
        help="backend inference server URL (e.g., http://localhost:8000)",
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

    return parser


def _parser_error(parser: argparse.ArgumentParser, message: str) -> None:
    parser.error(message)


def _cost_model(parser: argparse.ArgumentParser, args: argparse.Namespace) -> CostModel:
    kind = args.cost_model.replace("-", "_")
    dedicated_values = (args.total_hourly_price, args.per_gpu_hourly_price)
    serverless_values = (
        args.active_second_price,
        args.max_active_workers,
        args.billed_active_seconds,
    )
    if kind == "unknown":
        if any(
            value is not None
            for value in (
                *dedicated_values,
                *serverless_values,
                args.user_supplied_total,
            )
        ):
            _parser_error(parser, "price fields require their matching --cost-model")
        return CostModel()
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
        total = args.total_hourly_price
        if total is None:
            total = float(args.per_gpu_hourly_price) * args.gpus
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


def _resolve_key(parser: argparse.ArgumentParser, env_name: str) -> str:
    if not ENV_NAME_PATTERN.fullmatch(env_name):
        _parser_error(parser, "--api-key-env must be a valid environment variable name")
    key = os.environ.get(env_name)
    if not key:
        _parser_error(parser, "API key environment variable is missing or empty")
    return key


def _run_mode(args: argparse.Namespace) -> str:
    if args.command == "plan":
        return args.run_mode
    if args.command == "golden":
        return "benchmark"
    if args.command in {"experimental-tuning", "diagnose"}:
        return "smoke"
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
    experimental = args.command == "experimental-tuning"
    diagnose = args.command == "diagnose"
    if args.backend == "guidellm" and args.guidellm_prompt_tokens is None:
        _parser_error(parser, "--backend guidellm requires --guidellm-prompt-tokens")
    if args.warmup_requests is not None and args.warmup_requests < 0:
        _parser_error(parser, "--warmup-requests must be non-negative")
    blocks = args.blocks if args.blocks is not None else (1 if mode == "smoke" else 3)
    requests_per_block = args.requests_per_block
    if requests_per_block is None and args.block_seconds is None:
        requests_per_block = 201 if experimental else 20 if diagnose else 8 if mode == "smoke" else 67
    warmups = args.warmup_requests
    if warmups is None:
        warmups = 3 if (experimental or diagnose) else 1 if mode == "smoke" else 3
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
            else 900.0
            if experimental
            else 60.0
            if diagnose
            else 120.0
            if mode == "smoke"
            else 900.0
        ),
        max_errors=args.max_errors,
        max_concurrency=args.max_concurrency,
        max_response_bytes=args.max_response_bytes,
        max_estimated_spend=args.max_estimated_spend,
    )
    api_key = _resolve_key(parser, args.api_key_env) if resolve_key else ""
    config = RunConfig(
        mode=mode,
        backend=args.backend,
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
        guidellm_gaps_acknowledged=getattr(args, "allow_guidellm_validation_gaps", False),
        enable_cache=getattr(args, "enable_cache", False),
        cache_ttl_seconds=getattr(args, "cache_ttl_seconds", 3600.0),
        cache_max_size=getattr(args, "cache_max_size", 1000),
        cache_similarity_threshold=getattr(args, "cache_similarity_threshold", 0.85),
    )

    try:
        prompts = load_prompts(args.prompts)
        warmup_prompts = load_prompts(args.warmup_prompts, warmup=True)
        validate_config(config, for_traffic=resolve_key)
        if config.backend == "guidellm":
            from .guidellm_backend import preflight_guidellm_config
            preflight_guidellm_config(config, for_traffic=resolve_key)
    except (OSError, ValueError, RuntimeError) as exc:
        _parser_error(parser, str(exc))
        
    return config, prompts, warmup_prompts
        


def _copy_prompt_workload(
    prompts: Sequence[Sequence[Mapping[str, str]]],
) -> tuple[tuple[dict[str, str], ...], ...]:
    """Detach a workload so a called stage cannot redefine CLI evidence."""

    return tuple(
        tuple(
            {"role": message["role"], "content": message["content"]}
            for message in messages
        )
        for messages in prompts
    )


def _atomic_write(report: Mapping[str, Any], output: Path) -> None:
    committed = _atomic_write_guarded(report, output, lambda: True)
    if not committed:  # pragma: no cover - the unconditional guard is fixed true
        raise RuntimeError("atomic_write_commit_guard_failed")


def _atomic_write_new(report: Mapping[str, Any], output: Path) -> None:
    """Atomically create a mode-0600 artifact without replacing evidence."""

    output = output.expanduser().resolve()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".throttle-experimental-",
        suffix=".tmp",
        dir=output.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.link(temporary, output)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _preflight_new_artifact_path(path: Path) -> Path:
    """Validate one absent target and probe its existing parent safely."""

    requested = path.expanduser()
    if os.path.lexists(requested):
        raise FileExistsError("experimental_output_already_exists")
    expanded = requested.resolve(strict=False)
    if os.path.lexists(expanded):
        raise FileExistsError("experimental_output_already_exists")
    parent_metadata = os.lstat(expanded.parent)
    if not stat.S_ISDIR(parent_metadata.st_mode):
        raise NotADirectoryError("experimental_output_parent_not_directory")
    name_max = os.pathconf(expanded.parent, "PC_NAME_MAX")
    if (
        not expanded.name
        or (name_max >= 0 and len(os.fsencode(expanded.name)) > name_max)
    ):
        raise OSError("experimental_output_name_too_long")

    descriptor: int | None = None
    temporary: Path | None = None
    created_identity: tuple[int, int] | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".throttle-experimental-preflight-",
            suffix=".tmp",
            dir=expanded.parent,
        )
        temporary = Path(temporary_name)
        os.fchmod(descriptor, 0o600)
        metadata = os.fstat(descriptor)
        created_identity = (metadata.st_dev, metadata.st_ino)
        os.close(descriptor)
        descriptor = None
        current = os.lstat(temporary)
        if (
            not stat.S_ISREG(current.st_mode)
            or (current.st_dev, current.st_ino) != created_identity
        ):
            raise OSError("experimental_output_probe_replaced")
        os.unlink(temporary)
        created_identity = None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if created_identity is not None and temporary is not None:
            try:
                current = os.lstat(temporary)
            except FileNotFoundError:
                pass
            else:
                if (
                    stat.S_ISREG(current.st_mode)
                    and (current.st_dev, current.st_ino) == created_identity
                ):
                    os.unlink(temporary)
    if os.path.lexists(expanded):
        raise FileExistsError("experimental_output_already_exists")
    return expanded


def _experimental_output_paths_may_alias(left: Path, right: Path) -> bool:
    """Conservatively catch common case-insensitive APFS leaf aliases."""

    try:
        same_parent = os.path.samefile(left.parent, right.parent)
    except OSError:
        return True
    return same_parent and unicodedata.normalize("NFKC", left.name).casefold() == (
        unicodedata.normalize("NFKC", right.name).casefold()
    )


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
    plan: Mapping[str, Any], *, guidellm_prompt_tokens: int | None = None
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
    print(f"Max estimated spend: ${plan['limits']['max_estimated_spend']:.6f} USD")
    print(f"Destination: {plan['destination']['normalized_url']}")
    runtime = plan["runtime"]
    print(
        "Runtime: "
        f"{runtime['accelerator_backend']} / {runtime['accelerator']} / "
        f"{runtime['accelerator_runtime_version']}"
    )
    runtime_reasons = plan["runtime_provenance_reasons"]
    print(
        "Runtime evidence: complete"
        if not runtime_reasons
        else "Runtime evidence: " + ", ".join(runtime_reasons)
    )
    if not plan["traffic_preflight"]["backend_supported_on_this_platform"]:
        print(
            "Traffic preflight: blocked because GuideLLM subprocess-tree safety "
            "requires a POSIX platform"
        )
    elif plan["traffic_preflight"]["requires_unknown_cost_acknowledgement"]:
        print("Traffic preflight: blocked until --allow-unknown-cost is supplied")
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
    if plan["backend"] == "guidellm":
        print(
            "GuideLLM workload: synthetic_text with "
            f"{guidellm_prompt_tokens} prompt tokens; supplied prompt JSONL is not used "
            "and no parity claim is made. This backend is cross-check-only because "
            "strict completion validation and response-byte enforcement are unavailable. "
            "Tokenizer files must already be cached locally; tokenizer network downloads "
            "are disabled."
        )
        if not plan.get("guidellm_gaps_acknowledged", False):
            print(
                "GuideLLM traffic preflight: blocked until --allow-guidellm-validation-gaps is supplied"
            )


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


def _print_experimental_tuning(
    projection: Mapping[str, object],
    output: Path,
) -> None:
    analysis = projection["analysis"]
    assert isinstance(analysis, dict)
    print("Throttle EXPERIMENTAL TUNING — SUGGESTION ONLY")
    print("Safety boundary: passed")
    print(
        "Evidence scope: same-deployment matching and traffic isolation are "
        "operator-attested, not independently proven; this does not prove "
        "scheduler saturation or savings."
    )
    analysis_status = projection["analysis_status"]
    print(f"Analysis status: {analysis_status}")
    suggestion = analysis.get("suggestion")
    if isinstance(suggestion, dict):
        print(
            "Candidate test only: max_num_seqs "
            f"{suggestion['current_value']} -> "
            f"{suggestion['candidate_test_value']}."
        )
        print(f"Hypothesis: {suggestion['hypothesis']}")
        print(f"Risk: {suggestion['risk']}")
    else:
        reasons = (
            analysis.get("quality_reasons")
            if analysis_status == "insufficient_evidence"
            else analysis.get("no_suggestion_reasons")
        )
        if isinstance(reasons, list) and reasons:
            print("Suggestion unavailable: " + ", ".join(reasons))
    print(
        "Hard locks: decision eligible=false; auto-apply=false; "
        "Golden performed=false; Golden eligible=false; changes applied=false."
    )
    print(
        "Authorization boundary: this artifact cannot authorize its own CLI, "
        "standard-report, Golden, or configuration path."
    )
    print(f"Sanitized experimental artifact: {output}")


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


def _run_guidellm_backend(
    config: RunConfig,
    *,
    prompt_tokens: int,
    executable: str,
    progress: RunProgress,
    credential_env_name: str,
) -> dict[str, Any]:
    try:
        from .guidellm_backend import run_guidellm_matrix
    except ImportError as exc:
        raise RuntimeError("guidellm_backend_unavailable") from exc
    child_environment = dict(os.environ)
    child_environment.pop(credential_env_name, None)
    return run_guidellm_matrix(
        config,
        prompt_tokens=prompt_tokens,
        executable=executable,
        progress=progress,
        environ=child_environment,
    )


def _write_experimental_failure(
    args: argparse.Namespace,
    *,
    progress: RunProgress,
    config: RunConfig,
    prompts: Sequence[Sequence[Mapping[str, str]]],
    warmup_prompts: Sequence[Sequence[Mapping[str, str]]],
    cancelled: bool,
) -> int:
    code = "cancelled_by_user" if cancelled else "execution_failed"
    report = progress.snapshot()
    if type(report) is dict and report.get("status") == "complete":
        try:
            validate_experimental_run_report(
                report,
                config,
                prompts=prompts,
                warmup_prompts=warmup_prompts,
            )
        except Exception:
            report = None
    else:
        report = None
    if report is None:
        report = _failure_report("smoke", code)
        if cancelled:
            report["status"] = "cancelled"
    elif report.get("status") != "complete":
        report["status"] = "cancelled" if cancelled else "failed"
        report["decision_eligible"] = False
        report["stop_reason"] = code
    try:
        _atomic_write_new(report, args.output)
    except (OSError, TypeError, ValueError, OverflowError):
        message = (
            "Cancelled; sanitized smoke artifact could not be written."
            if cancelled
            else "Experimental tuning failed; sanitized smoke artifact could not "
            "be written."
        )
        print(message, file=sys.stderr)
        return EXIT_CANCELLED if cancelled else EXIT_FAILED
    message = (
        f"Cancelled; sanitized smoke artifact written to {args.output}."
        if cancelled
        else (
            "Experimental tuning failed safely; sanitized smoke artifact written "
            f"to {args.output}. No new experimental artifact was written."
        )
    )
    print(message, file=sys.stderr)
    return EXIT_CANCELLED if cancelled else EXIT_FAILED


def _handle_experimental_tuning(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> int:
    if args.backend != "native":
        _parser_error(parser, "experimental_requires_native_backend")
    if (
        args.request_rate is not None
        or args.concurrency is None
        or len(args.concurrency) != 1
    ):
        _parser_error(
            parser,
            "experimental_requires_single_closed_loop_condition",
        )
    if args.blocks is not None and args.blocks != 1:
        _parser_error(parser, "experimental_requires_one_block")
    try:
        ordinary_output = _preflight_new_artifact_path(args.output)
        experimental_output = _preflight_new_artifact_path(
            args.experimental_output
        )
    except FileExistsError:
        _parser_error(parser, "experimental_output_already_exists")
    except (OSError, RuntimeError):
        _parser_error(parser, "experimental_output_path_invalid")
    if _experimental_output_paths_may_alias(
        ordinary_output, experimental_output
    ):
        _parser_error(parser, "experimental_outputs_must_be_separate")
    args.output = ordinary_output
    args.experimental_output = experimental_output

    try:
        collector = prepare_metrics_collector(args.metrics_url)
    except ExperimentalTuningError as exc:
        _parser_error(parser, exc.code)
        raise AssertionError("argparse.error must terminate")

    config, prompts, warmup_prompts = _build_config(
        parser,
        args,
        resolve_key=False,
    )
    try:
        validate_experimental_config(config)
    except ExperimentalTuningError as exc:
        _parser_error(parser, exc.code)
        raise AssertionError("argparse.error must terminate")

    evidence_prompts = _copy_prompt_workload(prompts)
    evidence_warmup_prompts = _copy_prompt_workload(warmup_prompts)
    runner_prompts = _copy_prompt_workload(prompts)
    runner_warmup_prompts = _copy_prompt_workload(warmup_prompts)

    pre_key_config = replace(
        config,
        endpoint=EndpointConfig(
            url=config.endpoint.url,
            api_key="experimental-preflight-only",
        ),
    )
    try:
        validate_config(pre_key_config, for_traffic=True)
    except (ValueError, RuntimeError) as exc:
        _parser_error(parser, str(exc))

    api_key = _resolve_key(parser, args.api_key_env)
    config = replace(
        config,
        endpoint=EndpointConfig(url=config.endpoint.url, api_key=api_key),
    )
    try:
        validate_config(config, for_traffic=True)
    except (ValueError, RuntimeError) as exc:
        _parser_error(parser, str(exc))

    progress = RunProgress()
    traffic_scope = (
        "operator_attested_exclusive"
        if args.attest_same_deployment_exclusive_metrics
        else "unconfirmed"
    )
    try:
        outcome = asyncio.run(
            run_experimental_tuning(
                config,
                runner_prompts,
                runner_warmup_prompts,
                collector=collector,
                traffic_scope=traffic_scope,
                progress=progress,
                run_traffic=run_native,
            )
        )
        ordinary_report = validated_experimental_report(
            outcome,
            config,
            prompts=evidence_prompts,
            warmup_prompts=evidence_warmup_prompts,
        )
        validated_experimental_envelope(
            outcome,
            config,
            ordinary_report,
            prompts=evidence_prompts,
            warmup_prompts=evidence_warmup_prompts,
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        return _write_experimental_failure(
            args,
            progress=progress,
            config=config,
            prompts=evidence_prompts,
            warmup_prompts=evidence_warmup_prompts,
            cancelled=True,
        )
    except ExperimentalTuningError as exc:
        print(f"Experimental safety stop: {exc.code}.", file=sys.stderr)
        return _write_experimental_failure(
            args,
            progress=progress,
            config=config,
            prompts=evidence_prompts,
            warmup_prompts=evidence_warmup_prompts,
            cancelled=False,
        )
    except Exception:
        print(
            "Experimental safety stop: experimental_internal_failure.",
            file=sys.stderr,
        )
        return _write_experimental_failure(
            args,
            progress=progress,
            config=config,
            prompts=evidence_prompts,
            warmup_prompts=evidence_warmup_prompts,
            cancelled=False,
        )

    try:
        _atomic_write_new(ordinary_report, args.output)
    except (OSError, TypeError, ValueError, OverflowError):
        print(
            "Experimental measurement finished but the smoke report could not "
            "be written; no experimental artifact was written.",
            file=sys.stderr,
        )
        return EXIT_FAILED
    try:
        envelope = validated_experimental_envelope(
            outcome,
            config,
            ordinary_report,
            prompts=evidence_prompts,
            warmup_prompts=evidence_warmup_prompts,
        )
        _atomic_write_new(envelope, args.experimental_output)
    except (
        ExperimentalTuningError,
        OSError,
        TypeError,
        ValueError,
        OverflowError,
    ):
        print(
            "The smoke report was written, but the audited experimental artifact "
            "could not be written.",
            file=sys.stderr,
        )
        return EXIT_FAILED

    try:
        envelope = validated_experimental_envelope(
            outcome,
            config,
            ordinary_report,
            prompts=evidence_prompts,
            warmup_prompts=evidence_warmup_prompts,
        )
        projection = envelope["safety_projection"]
        if type(projection) is not dict:
            raise TypeError("invalid experimental projection")
        _print_run(ordinary_report, args.output)
        _print_experimental_tuning(
            projection,
            args.experimental_output,
        )
    except (
        ExperimentalTuningError,
        KeyError,
        TypeError,
        ValueError,
        ArithmeticError,
    ):
        print(
            "Audited artifacts were written, but terminal rendering failed safely.",
            file=sys.stderr,
        )
        return EXIT_FAILED
    if ordinary_report.get("status") != "complete" or any(
        not condition.get("valid")
        for condition in ordinary_report.get("conditions", [])
    ):
        return EXIT_FAILED
    return (
        EXIT_OK
        if projection.get("analysis_status") == "suggestion_available"
        else EXIT_INCONCLUSIVE
    )


def _handle_diagnose(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    return handle_diagnose(parser, args, _build_config, _atomic_write)


def _handle_run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    config, prompts, warmup_prompts = _build_config(parser, args, resolve_key=True)
    progress = RunProgress()
    try:
        if config.backend == "native":
            report = asyncio.run(
                run_native(config, prompts, warmup_prompts, progress=progress)
            )
        else:
            report = _run_guidellm_backend(
                config,
                prompt_tokens=args.guidellm_prompt_tokens,
                executable=args.guidellm_executable,
                progress=progress,
                credential_env_name=args.api_key_env,
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
    api_key = _resolve_key(parser, args.api_key_env)
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


def _handle_demo(args: argparse.Namespace) -> int:
    """Run simulator demo comparing baseline vs tuned configuration."""
    from throttle.simulator import VLLMSimulator, SimulatorConfig
    from throttle.workload import WorkloadGenerator
    from throttle.cost_model import calculate_cost
    import time

    print("Throttle Configuration Comparison Demo")
    print("=" * 60)
    print()

    # Choose workload parameters based on --light-load flag
    if args.light_load:
        # Light load: does not saturate either config (demonstrates NO SIGNIFICANT DIFFERENCE)
        num_requests = 100
        arrival_rate = 2.0
        mean_prompt = 500
        mean_output = 150
        load_desc = "light (2 req/sec, ~150 output tokens)"
    else:
        # Default: saturating load that differentiates configs
        num_requests = 300
        arrival_rate = 30.0
        mean_prompt = 300
        mean_output = 1000
        load_desc = "high concurrency (30 req/sec, ~1000 output tokens)"

    # Generate workload once for both configs
    print(f"[SIMULATED] Generating sample workload ({load_desc})...")
    workload_gen = WorkloadGenerator(seed=42)
    workload = workload_gen.generate_chat_workload(
        num_requests=num_requests,
        arrival_rate_requests_per_sec=arrival_rate,
        mean_prompt_tokens=mean_prompt,
        mean_output_tokens=mean_output,
    )
    print(f"[SIMULATED] Generated {len(workload)} requests")
    print()

    # Shared config parameters
    gpu_rate = 1.50
    prefill_throughput = 5000.0
    decode_throughput = 100.0

    # Baseline configuration: max_num_seqs=128
    baseline_config = SimulatorConfig(
        prefill_throughput_tokens_per_sec=prefill_throughput,
        decode_throughput_tokens_per_sec=decode_throughput,
        max_num_seqs=128,
        saturation_knee_sequences=102,  # 80% of 128
        kv_cache_capacity_tokens=500_000,
        gpu_hourly_rate_dollars=gpu_rate,
    )

    # Tuned configuration: max_num_seqs=256
    tuned_config = SimulatorConfig(
        prefill_throughput_tokens_per_sec=prefill_throughput,
        decode_throughput_tokens_per_sec=decode_throughput,
        max_num_seqs=256,
        saturation_knee_sequences=200,  # 80% of 256
        kv_cache_capacity_tokens=500_000,
        gpu_hourly_rate_dollars=gpu_rate,
    )

    print("Configuration being compared:")
    print(f"  Parameter: max_num_seqs")
    print(f"  Baseline: 128 concurrent sequences")
    print(f"  Tuned:    256 concurrent sequences")
    print()

    # Run baseline first (counterbalanced order)
    print("[SIMULATED] Running baseline configuration...")
    start_time = time.time()
    baseline_sim = VLLMSimulator(baseline_config)
    for arrival_time, prompt_tokens, output_tokens in workload:
        baseline_sim.add_request(arrival_time, prompt_tokens, output_tokens)
    baseline_completed, baseline_wall_clock = baseline_sim.run()
    baseline_elapsed = time.time() - start_time
    print(f"[SIMULATED] Baseline simulation complete in {baseline_elapsed:.2f} seconds")

    # Run tuned second
    print("[SIMULATED] Running tuned configuration...")
    start_time = time.time()
    tuned_sim = VLLMSimulator(tuned_config)
    for arrival_time, prompt_tokens, output_tokens in workload:
        tuned_sim.add_request(arrival_time, prompt_tokens, output_tokens)
    tuned_completed, tuned_wall_clock = tuned_sim.run()
    tuned_elapsed = time.time() - start_time
    print(f"[SIMULATED] Tuned simulation complete in {tuned_elapsed:.2f} seconds")
    print()

    # Calculate costs for both
    total_input = sum(r.prompt_tokens for r in baseline_completed)
    total_output = sum(r.tokens_generated for r in baseline_completed)

    baseline_cost = calculate_cost(
        input_tokens=total_input,
        output_tokens=total_output,
        wall_clock_seconds=baseline_wall_clock,
        gpu_hourly_rate_dollars=gpu_rate,
    )

    tuned_cost = calculate_cost(
        input_tokens=total_input,
        output_tokens=total_output,
        wall_clock_seconds=tuned_wall_clock,
        gpu_hourly_rate_dollars=gpu_rate,
    )

    # Calculate deltas
    wall_clock_delta = tuned_wall_clock - baseline_wall_clock
    gpu_hours_delta = tuned_cost.gpu_hours - baseline_cost.gpu_hours
    input_cost_delta = tuned_cost.dollars_per_million_input_tokens - baseline_cost.dollars_per_million_input_tokens
    output_cost_delta = tuned_cost.dollars_per_million_output_tokens - baseline_cost.dollars_per_million_output_tokens
    total_cost_delta = tuned_cost.total_dollars - baseline_cost.total_dollars

    # Confidence intervals (±20% conservative estimate for parameter uncertainty)
    ci_margin = 0.20
    wall_clock_ci = abs(wall_clock_delta * ci_margin)
    input_cost_ci = abs(input_cost_delta * ci_margin)
    output_cost_ci = abs(output_cost_delta * ci_margin)
    total_cost_ci = abs(total_cost_delta * ci_margin)

    # Print side-by-side comparison
    print("Cost Comparison Results")
    print("=" * 60)
    print()
    print(f"Workload:")
    print(f"  Total requests: {len(baseline_completed)}")
    print(f"  Total input tokens: {total_input:,}")
    print(f"  Total output tokens: {total_output:,}")
    print()
    print(f"{'Metric':<40} {'Baseline':>12} {'Tuned':>12} {'Delta':>12}")
    print("-" * 80)
    print(f"{'Wall clock time (seconds)':<40} {baseline_wall_clock:>12.2f} {tuned_wall_clock:>12.2f} {wall_clock_delta:>12.2f}")
    print(f"{'GPU hours':<40} {baseline_cost.gpu_hours:>12.6f} {tuned_cost.gpu_hours:>12.6f} {gpu_hours_delta:>12.6f}")
    print(f"{'Input cost ($/M tokens)':<40} {baseline_cost.dollars_per_million_input_tokens:>12.2f} {tuned_cost.dollars_per_million_input_tokens:>12.2f} {input_cost_delta:>12.2f}")
    print(f"{'Output cost ($/M tokens)':<40} {baseline_cost.dollars_per_million_output_tokens:>12.2f} {tuned_cost.dollars_per_million_output_tokens:>12.2f} {output_cost_delta:>12.2f}")
    print(f"{'Total cost ($)':<40} {baseline_cost.total_dollars:>12.4f} {tuned_cost.total_dollars:>12.4f} {total_cost_delta:>12.4f}")
    print()

    # Confidence intervals and significance check
    print("95% Confidence Intervals on Delta (±20% parameter uncertainty):")
    print(f"  Wall clock delta: {wall_clock_delta:.2f} ± {wall_clock_ci:.2f} seconds")
    print(f"  Input cost delta: ${input_cost_delta:.2f} ± ${input_cost_ci:.2f} per million")
    print(f"  Output cost delta: ${output_cost_delta:.2f} ± ${output_cost_ci:.2f} per million")
    print(f"  Total cost delta: ${total_cost_delta:.4f} ± ${total_cost_ci:.4f}")
    print()

    # Check if confidence intervals overlap zero
    wall_clock_overlaps_zero = (wall_clock_delta - wall_clock_ci) <= 0 <= (wall_clock_delta + wall_clock_ci)
    total_cost_overlaps_zero = (total_cost_delta - total_cost_ci) <= 0 <= (total_cost_delta + total_cost_ci)

    if total_cost_overlaps_zero:
        print("RESULT: NO SIGNIFICANT DIFFERENCE")
        print()
        print("The confidence interval on total cost delta includes zero. This means")
        print("the assumed parameters (prefill throughput, decode throughput, etc.)")
        print("have enough uncertainty that we cannot conclude which config is cheaper.")
        print()
        print("Run 'throttle cost' against a real endpoint to get measured results.")
    else:
        if total_cost_delta < 0:
            print("RESULT: Tuned configuration is cheaper")
            pct_savings = abs(total_cost_delta / baseline_cost.total_dollars) * 100
            print(f"  Estimated savings: ${abs(total_cost_delta):.4f} ({pct_savings:.1f}%)")
        else:
            print("RESULT: Baseline configuration is cheaper")
            pct_increase = (total_cost_delta / baseline_cost.total_dollars) * 100
            print(f"  Estimated increase: ${total_cost_delta:.4f} ({pct_increase:.1f}%)")
        print()
        print(f"This result is for a workload with {arrival_rate:.0f} req/sec arrival rate")
        print(f"and ~{mean_output} output tokens per request. The saving would be smaller")
        print("under lighter traffic where neither configuration saturates.")
        print()
        print("All estimates depend on ASSUMED throughput parameters.")
        print("Run 'throttle cost' against a real endpoint for measured costs.")
    print()

    # Sensitivity analysis
    print("Sensitivity Analysis: Decode Throughput Impact")
    print("=" * 60)
    print()
    print("The following shows how results change if decode throughput is")
    print("halved or doubled from the assumed 100 tok/sec:")
    print()

    for multiplier, label in [(0.5, "50% (halved)"), (2.0, "200% (doubled)")]:
        sens_decode = decode_throughput * multiplier

        sens_baseline_config = SimulatorConfig(
            prefill_throughput_tokens_per_sec=prefill_throughput,
            decode_throughput_tokens_per_sec=sens_decode,
            max_num_seqs=128,
            saturation_knee_sequences=102,
            kv_cache_capacity_tokens=500_000,
            gpu_hourly_rate_dollars=gpu_rate,
        )

        sens_tuned_config = SimulatorConfig(
            prefill_throughput_tokens_per_sec=prefill_throughput,
            decode_throughput_tokens_per_sec=sens_decode,
            max_num_seqs=256,
            saturation_knee_sequences=200,
            kv_cache_capacity_tokens=500_000,
            gpu_hourly_rate_dollars=gpu_rate,
        )

        # Run sensitivity simulations
        sens_baseline_sim = VLLMSimulator(sens_baseline_config)
        for arrival_time, prompt_tokens, output_tokens in workload:
            sens_baseline_sim.add_request(arrival_time, prompt_tokens, output_tokens)
        _, sens_baseline_wall = sens_baseline_sim.run()

        sens_tuned_sim = VLLMSimulator(sens_tuned_config)
        for arrival_time, prompt_tokens, output_tokens in workload:
            sens_tuned_sim.add_request(arrival_time, prompt_tokens, output_tokens)
        _, sens_tuned_wall = sens_tuned_sim.run()

        sens_baseline_cost = calculate_cost(
            input_tokens=total_input,
            output_tokens=total_output,
            wall_clock_seconds=sens_baseline_wall,
            gpu_hourly_rate_dollars=gpu_rate,
        )

        sens_tuned_cost = calculate_cost(
            input_tokens=total_input,
            output_tokens=total_output,
            wall_clock_seconds=sens_tuned_wall,
            gpu_hourly_rate_dollars=gpu_rate,
        )

        sens_delta = sens_tuned_cost.total_dollars - sens_baseline_cost.total_dollars

        print(f"Decode throughput: {sens_decode:.0f} tok/sec ({label})")
        print(f"  Baseline total cost: ${sens_baseline_cost.total_dollars:.4f}")
        print(f"  Tuned total cost:    ${sens_tuned_cost.total_dollars:.4f}")
        print(f"  Delta:               ${sens_delta:.4f}")
        if sens_delta < 0:
            print(f"  Result: Tuned is cheaper by ${abs(sens_delta):.4f}")
        elif sens_delta > 0:
            print(f"  Result: Baseline is cheaper by ${sens_delta:.4f}")
        else:
            print(f"  Result: Same cost")
        print()

    print("All values above are [SIMULATED] - they depend entirely on assumed")
    print("throughput parameters, not real hardware measurements.")
    print()

    return EXIT_OK


def _handle_cost(args: argparse.Namespace) -> int:
    """Measure cost per million tokens against a live endpoint."""
    import time
    import statistics
    from throttle.cost_model import calculate_cost
    from throttle.workload import WorkloadGenerator

    try:
        import httpx
    except ImportError:
        print("Error: httpx is required for cost measurement")
        print("Install with: pip install httpx")
        return EXIT_FAILED

    api_key = _get_api_key(args)
    headers = _build_headers(api_key)

    print(f"Measuring cost against {args.endpoint_url}")
    print(f"GPU hourly rate: ${args.gpu_hourly_rate:.2f}/hour")
    print(f"Test requests: {args.num_requests}")
    print()

    # Generate test workload
    workload_gen = WorkloadGenerator(seed=42)
    workload = workload_gen.generate_chat_workload(
        num_requests=args.num_requests,
        arrival_rate_requests_per_sec=1.0,
        mean_prompt_tokens=100,
        mean_output_tokens=50,
    )

    total_input_tokens = 0
    total_output_tokens = 0
    request_times = []

    print("Sending requests...")
    overall_start = time.time()

    try:
        with httpx.Client(timeout=30.0) as client:
            for i, (_, prompt_tokens, max_tokens) in enumerate(workload):
                # Create a simple prompt
                prompt = "Test " * prompt_tokens

                req_start = time.time()
                response = client.post(
                    f"{args.endpoint_url}/chat/completions",
                    headers=headers,
                    json={
                        "model": args.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                    },
                )
                req_end = time.time()

                if response.status_code != 200:
                    print(f"Error: Request {i+1} failed with status {response.status_code}")
                    return EXIT_FAILED

                data = response.json()
                usage = data.get("usage", {})
                total_input_tokens += usage.get("prompt_tokens", 0)
                total_output_tokens += usage.get("completion_tokens", 0)
                request_times.append(req_end - req_start)

                if (i + 1) % 5 == 0:
                    print(f"  Completed {i+1}/{args.num_requests} requests...")

    except httpx.RequestError as e:
        print(f"Error: Failed to connect to endpoint: {e}")
        print("Make sure the inference server is running and the URL is correct.")
        return EXIT_FAILED
    except Exception as e:
        print(f"Error: {e}")
        return EXIT_FAILED

    overall_end = time.time()
    wall_clock = overall_end - overall_start

    print()
    print("Measurement Results")
    print("=" * 60)
    print()

    # Calculate cost
    cost_result = calculate_cost(
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        wall_clock_seconds=wall_clock,
        gpu_hourly_rate_dollars=args.gpu_hourly_rate,
    )

    # Calculate confidence intervals (95% CI using t-distribution approximation)
    if len(request_times) >= 2:
        mean_time = statistics.mean(request_times)
        stdev_time = statistics.stdev(request_times)
        margin = 1.96 * (stdev_time / (len(request_times) ** 0.5))
        ci_lower = mean_time - margin
        ci_upper = mean_time + margin
    else:
        ci_lower = ci_upper = request_times[0] if request_times else 0

    print(f"Workload:")
    print(f"  Requests: {args.num_requests}")
    print(f"  Total input tokens: {total_input_tokens:,}")
    print(f"  Total output tokens: {total_output_tokens:,}")
    print()

    print(f"Measured Performance:")
    print(f"  Wall clock time: {wall_clock:.2f} seconds")
    print(f"  Average request time: {statistics.mean(request_times):.3f}s (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])")
    print()

    print(f"Measured Cost:")
    print(f"  GPU hours: {cost_result.gpu_hours:.6f}")
    print(f"  Total cost: ${cost_result.total_dollars:.4f}")
    print(f"  Input cost: ${cost_result.dollars_per_million_input_tokens:.2f} per million tokens")
    print(f"  Output cost: ${cost_result.dollars_per_million_output_tokens:.2f} per million tokens")
    print()

    return EXIT_OK




def _handle_compare_measure(report_paths: list[str]) -> int:
    """Compare measure outputs with statistical testing."""
    import json
    import random
    import statistics

    # Load all measure outputs
    measures = []
    for path in report_paths:
        with open(path) as f:
            measures.append(json.load(f))

    print("Throttle Compare - Measure Outputs")
    print("=" * 80)
    print()

    # Extract data for each measure
    results = []
    for m in measures:
        label = m["label"]
        note = m.get("note", "")
        median_input = m["median_dollars_per_million_input"]
        median_output = m["median_dollars_per_million_output"]
        ci_input = m["ci_95_input"]
        ci_output = m["ci_95_output"]

        # Get per-run costs for bootstrap
        input_costs = [r["dollars_per_million_input"] for r in m["runs"]]
        output_costs = [r["dollars_per_million_output"] for r in m["runs"]]

        results.append({
            "label": label,
            "note": note,
            "median_input": median_input,
            "median_output": median_output,
            "ci_input": ci_input,
            "ci_output": ci_output,
            "input_costs": input_costs,
            "output_costs": output_costs,
        })

    # Check for overlaps between all pairs
    overlaps = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            r1, r2 = results[i], results[j]

            # Check if input intervals overlap
            input_overlap = not (r1["ci_input"][1] < r2["ci_input"][0] or
                                r2["ci_input"][1] < r1["ci_input"][0])

            # Check if output intervals overlap
            output_overlap = not (r1["ci_output"][1] < r2["ci_output"][0] or
                                  r2["ci_output"][1] < r1["ci_output"][0])

            if input_overlap or output_overlap:
                # Compute overlap amounts
                if input_overlap:
                    input_overlap_start = max(r1["ci_input"][0], r2["ci_input"][0])
                    input_overlap_end = min(r1["ci_input"][1], r2["ci_input"][1])
                    input_overlap_amount = input_overlap_end - input_overlap_start
                else:
                    input_overlap_amount = 0

                if output_overlap:
                    output_overlap_start = max(r1["ci_output"][0], r2["ci_output"][0])
                    output_overlap_end = min(r1["ci_output"][1], r2["ci_output"][1])
                    output_overlap_amount = output_overlap_end - output_overlap_start
                else:
                    output_overlap_amount = 0

                overlaps.append({
                    "label1": r1["label"],
                    "label2": r2["label"],
                    "input_overlap": input_overlap,
                    "output_overlap": output_overlap,
                    "input_overlap_amount": input_overlap_amount,
                    "output_overlap_amount": output_overlap_amount,
                })

    # If overlaps exist, cannot rank
    if overlaps:
        print("NO SIGNIFICANT DIFFERENCE")
        print()
        for overlap in overlaps:
            print(f"{overlap['label1']} vs {overlap['label2']}:")
            if overlap['input_overlap']:
                print(f"  Input intervals overlap by ${overlap['input_overlap_amount']:.2f}/M")
            if overlap['output_overlap']:
                print(f"  Output intervals overlap by ${overlap['output_overlap_amount']:.2f}/M")
        print()
        print("Cannot rank configurations with overlapping confidence intervals.")
        print()

        # Still print table but without ranking
        print("Configuration Details:")
        print()
        print(f"{'Label':<20} {'Note':<30} {'$/M Input':<25} {'$/M Output':<25}")
        print("-" * 100)
        for r in results:
            note = r['note'] or ""
            note_display = note[:28] + ".." if len(note) > 30 else note
            input_display = f"${r['median_input']:.2f} [{r['ci_input'][0]:.2f}, {r['ci_input'][1]:.2f}]"
            output_display = f"${r['median_output']:.2f} [{r['ci_output'][0]:.2f}, {r['ci_output'][1]:.2f}]"
            print(f"{r['label']:<20} {note_display:<30} {input_display:<25} {output_display:<25}")

        return EXIT_OK

    # No overlaps - can rank
    # Sort by total cost (input + output)
    results_sorted = sorted(results, key=lambda r: r['median_input'] + r['median_output'])

    print("Ranked by Total Cost ($/M tokens)")
    print()

    # Print header
    print(f"{'Rank':<6} {'Label':<20} {'Note':<30} {'$/M Input':<25} {'$/M Output':<25}")
    print("-" * 106)

    # Print each configuration
    for rank, r in enumerate(results_sorted, 1):
        note = r['note'] or ""
        note_display = note[:28] + ".." if len(note) > 30 else note
        input_display = f"${r['median_input']:.2f} [{r['ci_input'][0]:.2f}, {r['ci_input'][1]:.2f}]"
        output_display = f"${r['median_output']:.2f} [{r['ci_output'][0]:.2f}, {r['ci_output'][1]:.2f}]"
        print(f"{rank:<6} {r['label']:<20} {note_display:<30} {input_display:<25} {output_display:<25}")

    # Compute pairwise deltas with bootstrap
    if len(results_sorted) >= 2:
        print()
        print("Pairwise Differences (bootstrap difference method, 10000 resamples):")
        print()

        for i in range(len(results_sorted) - 1):
            r1 = results_sorted[i]
            r2 = results_sorted[i + 1]

            # Bootstrap the difference
            random.seed(42)
            n_bootstrap = 10000
            input_diffs = []
            output_diffs = []

            for _ in range(n_bootstrap):
                # Resample from each configuration's runs
                sample1_input = statistics.median(random.choices(r1["input_costs"], k=len(r1["input_costs"])))
                sample2_input = statistics.median(random.choices(r2["input_costs"], k=len(r2["input_costs"])))
                input_diffs.append(sample2_input - sample1_input)

                sample1_output = statistics.median(random.choices(r1["output_costs"], k=len(r1["output_costs"])))
                sample2_output = statistics.median(random.choices(r2["output_costs"], k=len(r2["output_costs"])))
                output_diffs.append(sample2_output - sample1_output)

            input_diffs.sort()
            output_diffs.sort()
            ci_lower_idx = int(0.025 * n_bootstrap)
            ci_upper_idx = int(0.975 * n_bootstrap)

            input_diff_median = statistics.median(input_diffs)
            input_diff_ci = [input_diffs[ci_lower_idx], input_diffs[ci_upper_idx]]

            output_diff_median = statistics.median(output_diffs)
            output_diff_ci = [output_diffs[ci_lower_idx], output_diffs[ci_upper_idx]]

            print(f"{r2['label']} - {r1['label']}:")
            print(f"  Input:  ${input_diff_median:+.2f}/M  [{input_diff_ci[0]:+.2f}, {input_diff_ci[1]:+.2f}]")
            print(f"  Output: ${output_diff_median:+.2f}/M  [{output_diff_ci[0]:+.2f}, {output_diff_ci[1]:+.2f}]")
            print()

    return EXIT_OK


def _handle_report(args: argparse.Namespace) -> int:
    """Generate HTML report comparing two measure outputs."""
    import json
    import statistics

    # Load both reports
    try:
        with open(args.reports[0]) as f:
            report1 = json.load(f)
        with open(args.reports[1]) as f:
            report2 = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading reports: {e}")
        return EXIT_FAILED

    # Validate format
    if "ci_95_input" not in report1 or "ci_95_input" not in report2:
        print("Error: Reports must be measure output JSON files")
        return EXIT_FAILED

    # Extract data
    label1 = report1["label"]
    label2 = report2["label"]

    r1_input_median = report1["median_dollars_per_million_input"]
    r1_input_ci = report1["ci_95_input"]
    r1_output_median = report1["median_dollars_per_million_output"]
    r1_output_ci = report1["ci_95_output"]

    r2_input_median = report2["median_dollars_per_million_input"]
    r2_input_ci = report2["ci_95_input"]
    r2_output_median = report2["median_dollars_per_million_output"]
    r2_output_ci = report2["ci_95_output"]

    # Determine if significant (using same logic as compare)
    # Check if confidence intervals overlap
    input_overlap = not (r1_input_ci[1] < r2_input_ci[0] or r2_input_ci[1] < r1_input_ci[0])
    output_overlap = not (r1_output_ci[1] < r2_output_ci[0] or r2_output_ci[1] < r1_output_ci[0])

    is_significant = not (input_overlap or output_overlap)
    banner_text = "SIGNIFICANT DIFFERENCE" if is_significant else "NO SIGNIFICANT DIFFERENCE"
    banner_color = "#dc3545" if is_significant else "#6c757d"

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Throttle Cost Comparison</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
    <style>
        body {{ font-family: system-ui, -apple-system, sans-serif; margin: 40px; background: #f8f9fa; }}
        .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; margin-bottom: 10px; }}
        .banner {{ background: {banner_color}; color: white; padding: 15px; text-align: center; font-size: 20px; font-weight: bold; border-radius: 4px; margin: 20px 0; }}
        .chart-container {{ position: relative; height: 400px; margin: 30px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        .note {{ color: #6c757d; font-size: 14px; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Throttle Cost Comparison</h1>
        <p>Comparing <strong>{label1}</strong> vs <strong>{label2}</strong></p>

        <div class="banner">{banner_text}</div>

        <div class="chart-container">
            <canvas id="costChart"></canvas>
        </div>

        <table>
            <thead>
                <tr>
                    <th>Configuration</th>
                    <th>Input ($/M tokens)</th>
                    <th>Output ($/M tokens)</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>{label1}</strong></td>
                    <td>${r1_input_median:.2f} <span style="color: #6c757d;">[${r1_input_ci[0]:.2f}, ${r1_input_ci[1]:.2f}]</span></td>
                    <td>${r1_output_median:.2f} <span style="color: #6c757d;">[${r1_output_ci[0]:.2f}, ${r1_output_ci[1]:.2f}]</span></td>
                </tr>
                <tr>
                    <td><strong>{label2}</strong></td>
                    <td>${r2_input_median:.2f} <span style="color: #6c757d;">[${r2_input_ci[0]:.2f}, ${r2_input_ci[1]:.2f}]</span></td>
                    <td>${r2_output_median:.2f} <span style="color: #6c757d;">[${r2_output_ci[0]:.2f}, ${r2_output_ci[1]:.2f}]</span></td>
                </tr>
            </tbody>
        </table>

        <p class="note">
            95% confidence intervals shown in brackets.
            {"Intervals do not overlap - difference is statistically significant." if is_significant else "Intervals overlap - difference is not statistically significant."}
        </p>
    </div>

    <script>
        const ctx = document.getElementById('costChart').getContext('2d');
        new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: ['{label1}', '{label2}'],
                datasets: [
                    {{
                        label: 'Input ($/M tokens)',
                        data: [{r1_input_median:.2f}, {r2_input_median:.2f}],
                        backgroundColor: 'rgba(54, 162, 235, 0.7)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 1,
                        errorBars: {{
                            '{label1}': {{minus: {r1_input_median - r1_input_ci[0]:.2f}, plus: {r1_input_ci[1] - r1_input_median:.2f}}},
                            '{label2}': {{minus: {r2_input_median - r2_input_ci[0]:.2f}, plus: {r2_input_ci[1] - r2_input_median:.2f}}}
                        }}
                    }},
                    {{
                        label: 'Output ($/M tokens)',
                        data: [{r1_output_median:.2f}, {r2_output_median:.2f}],
                        backgroundColor: 'rgba(255, 99, 132, 0.7)',
                        borderColor: 'rgba(255, 99, 132, 1)',
                        borderWidth: 1,
                        errorBars: {{
                            '{label1}': {{minus: {r1_output_median - r1_output_ci[0]:.2f}, plus: {r1_output_ci[1] - r1_output_median:.2f}}},
                            '{label2}': {{minus: {r2_output_median - r2_output_ci[0]:.2f}, plus: {r2_output_ci[1] - r2_output_median:.2f}}}
                        }}
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Cost per Million Tokens (95% CI)',
                        font: {{ size: 16 }}
                    }},
                    legend: {{
                        position: 'top'
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: '$ / Million Tokens'
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""

    # Write HTML file
    try:
        with open(args.out, 'w') as f:
            f.write(html)
        print(f"Report written to: {args.out}")
        return EXIT_OK
    except OSError as e:
        print(f"Error writing report: {e}")
        return EXIT_FAILED


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


def _handle_measure(args: argparse.Namespace) -> int:
    """Measure cost per million tokens with repeated trials."""
    from throttle.workload import WorkloadGenerator
    from throttle.cost_model import calculate_cost
    import time
    import json
    from datetime import datetime
    import statistics

    try:
        import httpx
    except ImportError:
        print("Error: httpx is required for measure")
        print("Install with: pip install httpx")
        return EXIT_FAILED

    api_key = _get_api_key(args)
    headers = _build_headers(api_key)

    print(f"Throttle Measure - {args.label}")
    print("=" * 60)
    print()

    # Test endpoint connectivity first
    print(f"Testing connection to {args.endpoint_url}...")
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                f"{args.endpoint_url}/v1/chat/completions",
                headers=headers,
                json={
                    "model": args.model,
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 1,
                },
            )
            if response.status_code != 200:
                print(f"Error: Endpoint returned status {response.status_code}")
                return EXIT_FAILED
    except httpx.RequestError as e:
        print(f"Error: Cannot connect to endpoint: {e}")
        return EXIT_FAILED

    print("Connection successful.")
    print()

    # Workload parameters from args
    arrival_rate = args.arrival_rate
    num_requests = args.num_requests
    mean_prompt_tokens = 200
    mean_output_tokens = 150

    print(f"Running {args.repeat} trials...")
    print(f"  Workload: {num_requests} requests at {arrival_rate:.1f} req/sec")
    print(f"  Mean prompt tokens: {mean_prompt_tokens}")
    print(f"  Mean output tokens: {mean_output_tokens}")
    print(f"  Fixed seed: trials measure server variance, not workload variance")
    print()

    workload_gen = WorkloadGenerator(seed=42)
    all_runs = []

    for run_idx in range(args.repeat):
        print(f"Trial {run_idx + 1}/{args.repeat}...", end=" ", flush=True)

        # Generate workload (same seed for all runs to measure server variance only)
        workload = workload_gen.generate_chat_workload(
            num_requests=num_requests,
            arrival_rate_requests_per_sec=arrival_rate,
            mean_prompt_tokens=mean_prompt_tokens,
            mean_output_tokens=mean_output_tokens,
        )

        # Reuse validate-sim's concurrent dispatch
        async def run_concurrent_workload():
            import asyncio
            import threading

            measured_requests = []
            real_total_input = 0
            real_total_output = 0
            peak_concurrent = 0
            current_in_flight = 0
            lock = threading.Lock()

            async def send_request(arrival_time, prompt_tokens, max_tokens, request_idx):
                nonlocal real_total_input, real_total_output, peak_concurrent, current_in_flight

                # Wait until scheduled arrival time
                if arrival_time > 0:
                    await asyncio.sleep(arrival_time)

                # Track in-flight requests
                with lock:
                    current_in_flight += 1
                    if current_in_flight > peak_concurrent:
                        peak_concurrent = current_in_flight

                prompt = "Test " * prompt_tokens
                req_start = time.perf_counter_ns()

                async with httpx.AsyncClient(timeout=120.0) as client:
                    try:
                        response = await client.post(
                            f"{args.endpoint_url}/v1/chat/completions",
                            headers=headers,
                            json={
                                "model": args.model,
                                "messages": [{"role": "user", "content": prompt}],
                                "max_tokens": max_tokens,
                            },
                        )
                        req_end = time.perf_counter_ns()

                        if response.status_code != 200:
                            raise Exception(f"Request {request_idx+1} failed with status {response.status_code}")

                        data = response.json()
                        usage = data.get("usage", {})
                        actual_input = usage.get("prompt_tokens", 0)
                        actual_output = usage.get("completion_tokens", 0)

                        with lock:
                            real_total_input += actual_input
                            real_total_output += actual_output
                            current_in_flight -= 1

                        req_duration_ns = req_end - req_start
                        result = {
                            "prompt_tokens": actual_input,
                            "completion_tokens": actual_output,
                            "duration_seconds": req_duration_ns / 1e9,
                        }

                        return result
                    except Exception as e:
                        with lock:
                            current_in_flight -= 1
                        raise

            # Launch all tasks concurrently
            tasks = [
                send_request(arrival_time, prompt_tokens, max_tokens, i)
                for i, (arrival_time, prompt_tokens, max_tokens) in enumerate(workload)
            ]

            overall_start = time.perf_counter_ns()
            measured_requests = await asyncio.gather(*tasks)
            overall_end = time.perf_counter_ns()
            real_wall_clock = (overall_end - overall_start) / 1e9

            return measured_requests, real_total_input, real_total_output, real_wall_clock, peak_concurrent

        try:
            import asyncio
            measured_requests, real_total_input, real_total_output, real_wall_clock, peak_concurrent = asyncio.run(run_concurrent_workload())
        except Exception as e:
            print(f"FAILED: {e}")
            return EXIT_FAILED

        real_cost = calculate_cost(
            input_tokens=real_total_input,
            output_tokens=real_total_output,
            wall_clock_seconds=real_wall_clock,
            gpu_hourly_rate_dollars=args.gpu_hourly_rate,
        )

        all_runs.append({
            "run_index": run_idx,
            "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            "wall_clock_seconds": real_wall_clock,
            "total_input_tokens": real_total_input,
            "total_output_tokens": real_total_output,
            "dollars_per_million_input": real_cost.dollars_per_million_input_tokens,
            "dollars_per_million_output": real_cost.dollars_per_million_output_tokens,
            "total_dollars": real_cost.total_dollars,
            "peak_concurrent_requests": peak_concurrent,
            "per_request_timings": measured_requests,
        })

        print(f"${real_cost.dollars_per_million_input_tokens:.2f}/M in, ${real_cost.dollars_per_million_output_tokens:.2f}/M out")

    # Compute median and bootstrap 95% CI across runs
    input_costs = [r["dollars_per_million_input"] for r in all_runs]
    output_costs = [r["dollars_per_million_output"] for r in all_runs]
    median_input = statistics.median(input_costs)
    median_output = statistics.median(output_costs)

    # Bootstrap percentile confidence intervals
    import random
    random.seed(42)
    n_bootstrap = 10000
    bootstrap_input = []
    bootstrap_output = []
    for _ in range(n_bootstrap):
        resample = random.choices(range(len(all_runs)), k=len(all_runs))
        bootstrap_input.append(statistics.median([input_costs[i] for i in resample]))
        bootstrap_output.append(statistics.median([output_costs[i] for i in resample]))

    bootstrap_input.sort()
    bootstrap_output.sort()
    ci_lower_idx = int(0.025 * n_bootstrap)
    ci_upper_idx = int(0.975 * n_bootstrap)

    input_ci_lower = bootstrap_input[ci_lower_idx]
    input_ci_upper = bootstrap_input[ci_upper_idx]
    output_ci_lower = bootstrap_output[ci_lower_idx]
    output_ci_upper = bootstrap_output[ci_upper_idx]

    print()
    print(f"Results (95% CI via bootstrap percentile method, {n_bootstrap} resamples):")
    print(f"  Input:  ${median_input:.2f}/M  [${input_ci_lower:.2f}, ${input_ci_upper:.2f}]")
    print(f"  Output: ${median_output:.2f}/M  [${output_ci_lower:.2f}, ${output_ci_upper:.2f}]")
    print()
    print(f"Interval reflects server variance at a fixed workload, not variation in traffic mix.")
    print()

    # Write results to JSON
    output = {
        "label": args.label,
        "endpoint_url": args.endpoint_url,
        "model": args.model,
        "gpu_hourly_rate": args.gpu_hourly_rate,
        "note": args.note if args.note else None,
        "workload": {
            "num_requests": num_requests,
            "arrival_rate_req_per_sec": arrival_rate,
            "mean_prompt_tokens": mean_prompt_tokens,
            "mean_output_tokens": mean_output_tokens,
            "fixed_seed_explanation": "trials measure server variance, not workload variance",
        },
        "num_trials": args.repeat,
        "median_dollars_per_million_input": median_input,
        "median_dollars_per_million_output": median_output,
        "ci_95_input": [input_ci_lower, input_ci_upper],
        "ci_95_output": [output_ci_lower, output_ci_upper],
        "ci_method": "bootstrap percentile",
        "ci_resamples": n_bootstrap,
        "runs": all_runs,
    }

    output_file = f"{args.label}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results written to: {output_file}")
    print()

    return EXIT_OK


def _handle_validate_sim(args: argparse.Namespace) -> int:
    """Validate simulator accuracy against real measurements at three load levels."""
    import os

    # Require explicit opt-in via environment variable
    if os.environ.get("THROTTLE_ENABLE_EXPERIMENTAL") != "1":
        print("Error: validate-sim is an experimental command.", file=sys.stderr)
        print("", file=sys.stderr)
        print("This command has known issues:", file=sys.stderr)
        print("  - Serial execution bug (requests sent one-by-one, not concurrently)", file=sys.stderr)
        print("  - Results are not decision-eligible", file=sys.stderr)
        print("", file=sys.stderr)
        print("To enable experimental commands, set:", file=sys.stderr)
        print("  export THROTTLE_ENABLE_EXPERIMENTAL=1", file=sys.stderr)
        return EXIT_USAGE

    from throttle.simulator import VLLMSimulator, SimulatorConfig
    from throttle.workload import WorkloadGenerator
    from throttle.cost_model import calculate_cost
    import time
    import json
    from datetime import datetime

    try:
        import httpx
    except ImportError:
        print("Error: httpx is required for validation")
        print("Install with: pip install httpx")
        return EXIT_FAILED

    api_key = _get_api_key(args)
    headers = _build_headers(api_key)

    chat_completions_url = normalize_chat_completions_url(
        args.endpoint_url,
        allow_insecure_http=True,
    )

    print("Throttle Simulator Validation")
    print("=" * 60)
    print()

    # Test endpoint connectivity first
    print(f"Testing connection to {args.endpoint_url}...")
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                chat_completions_url,
                headers=headers,
                json={
                    "model": args.model,
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 1,
                },
            )
            if response.status_code != 200:
                print(f"Error: Endpoint returned status {response.status_code}")
                print("Cannot reach endpoint. Check the URL and ensure the server is running.")
                return EXIT_FAILED
    except httpx.RequestError as e:
        print(f"Error: Cannot connect to endpoint: {e}")
        print("Check that the endpoint URL is correct and the server is running.")
        return EXIT_FAILED

    print("Connection successful.")
    print()

    # Simulator config with all default parameters
    sim_config = SimulatorConfig(
        prefill_throughput_tokens_per_sec=5000.0,
        decode_throughput_tokens_per_sec=100.0,
        max_num_seqs=256,
        saturation_knee_sequences=200,
        kv_cache_capacity_tokens=500_000,
        preemption_overhead_per_token_sec=0.0002,
        saturation_penalty_at_max=2.0,
        gpu_hourly_rate_dollars=args.gpu_hourly_rate,
    )

    # Test at three arrival rates to see if error grows with load
    test_scenarios = [
        {"name": "Light load", "arrival_rate": 1.0, "num_requests": 20},
        {"name": "Medium load", "arrival_rate": 5.0, "num_requests": 50},
        {"name": "Heavy load", "arrival_rate": 10.0, "num_requests": 100},
    ]

    validation_results = {
        "endpoint_url": args.endpoint_url,
        "model": args.model,
        "gpu_hourly_rate": args.gpu_hourly_rate,
        "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        "simulator_config": {
            "prefill_throughput_tokens_per_sec": sim_config.prefill_throughput_tokens_per_sec,
            "decode_throughput_tokens_per_sec": sim_config.decode_throughput_tokens_per_sec,
            "max_num_seqs": sim_config.max_num_seqs,
            "saturation_knee_sequences": sim_config.saturation_knee_sequences,
            "kv_cache_capacity_tokens": sim_config.kv_cache_capacity_tokens,
            "preemption_overhead_per_token_sec": sim_config.preemption_overhead_per_token_sec,
            "saturation_penalty_at_max": sim_config.saturation_penalty_at_max,
        },
        "scenarios": []
    }

    workload_gen = WorkloadGenerator(seed=42)

    for scenario in test_scenarios:
        print(f"Running {scenario['name']} ({scenario['arrival_rate']:.0f} req/sec, {scenario['num_requests']} requests)...")
        print()

        # Generate workload
        workload = workload_gen.generate_chat_workload(
            num_requests=scenario['num_requests'],
            arrival_rate_requests_per_sec=scenario['arrival_rate'],
            mean_prompt_tokens=200,
            mean_output_tokens=150,
        )

        # Run simulator
        sim = VLLMSimulator(sim_config)
        for arrival_time, prompt_tokens, output_tokens in workload:
            sim.add_request(arrival_time, prompt_tokens, output_tokens)
        sim_completed, sim_wall_clock = sim.run()

        sim_total_input = sum(r.prompt_tokens for r in sim_completed)
        sim_total_output = sum(r.tokens_generated for r in sim_completed)
        sim_cost = calculate_cost(
            input_tokens=sim_total_input,
            output_tokens=sim_total_output,
            wall_clock_seconds=sim_wall_clock,
            gpu_hourly_rate_dollars=args.gpu_hourly_rate,
        )

        # Run real measurements with concurrent requests respecting arrival times
        async def run_concurrent_workload():
            import asyncio
            import threading

            measured_requests = []
            real_total_input = 0
            real_total_output = 0
            first_token_times = []
            peak_concurrent = 0
            current_in_flight = 0
            lock = threading.Lock()

            async def send_request(arrival_time, prompt_tokens, max_tokens, request_idx):
                nonlocal real_total_input, real_total_output, peak_concurrent, current_in_flight

                # Wait until scheduled arrival time
                if arrival_time > 0:
                    await asyncio.sleep(arrival_time)

                # Track in-flight requests
                with lock:
                    current_in_flight += 1
                    if current_in_flight > peak_concurrent:
                        peak_concurrent = current_in_flight

                prompt = "Test " * prompt_tokens
                req_start = time.perf_counter_ns()

                async with httpx.AsyncClient(timeout=120.0) as client:
                    try:
                        response = await client.post(
                            chat_completions_url,
                            headers=headers,
                            json={
                                "model": args.model,
                                "messages": [{"role": "user", "content": prompt}],
                                "max_tokens": max_tokens,
                            },
                        )
                        req_end = time.perf_counter_ns()

                        if response.status_code != 200:
                            raise Exception(f"Request {request_idx+1} failed with status {response.status_code}")

                        data = response.json()
                        usage = data.get("usage", {})
                        actual_input = usage.get("prompt_tokens", 0)
                        actual_output = usage.get("completion_tokens", 0)

                        with lock:
                            real_total_input += actual_input
                            real_total_output += actual_output
                            current_in_flight -= 1

                        req_duration_ns = req_end - req_start
                        result = {
                            "prompt_tokens": actual_input,
                            "completion_tokens": actual_output,
                            "duration_seconds": req_duration_ns / 1e9,
                        }

                        # Estimate TTFT
                        if actual_output > 0:
                            est_ttft_ns = req_duration_ns // (actual_output + 1)
                            with lock:
                                first_token_times.append(est_ttft_ns / 1e9)

                        return result
                    except Exception as e:
                        with lock:
                            current_in_flight -= 1
                        raise

            # Launch all tasks concurrently
            tasks = [
                send_request(arrival_time, prompt_tokens, max_tokens, i)
                for i, (arrival_time, prompt_tokens, max_tokens) in enumerate(workload)
            ]

            overall_start = time.perf_counter_ns()
            measured_requests = await asyncio.gather(*tasks)
            overall_end = time.perf_counter_ns()
            real_wall_clock = (overall_end - overall_start) / 1e9

            return measured_requests, real_total_input, real_total_output, first_token_times, real_wall_clock, peak_concurrent

        try:
            measured_requests, real_total_input, real_total_output, first_token_times, real_wall_clock, peak_concurrent = asyncio.run(run_concurrent_workload())
        except Exception as e:
            print(f"Error: Request failed: {e}")
            return EXIT_FAILED

        # Assert peak concurrency > 1 for load validation
        if peak_concurrent == 1:
            print(f"ERROR: Peak concurrent requests was 1 - harness is still serial!")
            print(f"This invalidates the load level test. Arrival rate had no effect.")
            return EXIT_FAILED

        real_cost = calculate_cost(
            input_tokens=real_total_input,
            output_tokens=real_total_output,
            wall_clock_seconds=real_wall_clock,
            gpu_hourly_rate_dollars=args.gpu_hourly_rate,
        )

        # Calculate metrics
        sim_input_throughput = sim_total_input / sim_wall_clock if sim_wall_clock > 0 else 0
        sim_output_throughput = sim_total_output / sim_wall_clock if sim_wall_clock > 0 else 0
        real_input_throughput = real_total_input / real_wall_clock if real_wall_clock > 0 else 0
        real_output_throughput = real_total_output / real_wall_clock if real_wall_clock > 0 else 0

        avg_ttft = sum(first_token_times) / len(first_token_times) if first_token_times else 0

        # Calculate errors
        wall_clock_error_pct = ((sim_wall_clock - real_wall_clock) / real_wall_clock) * 100 if real_wall_clock > 0 else 0
        input_throughput_error_pct = ((sim_input_throughput - real_input_throughput) / real_input_throughput) * 100 if real_input_throughput > 0 else 0
        output_throughput_error_pct = ((sim_output_throughput - real_output_throughput) / real_output_throughput) * 100 if real_output_throughput > 0 else 0
        cost_per_m_error_pct = ((sim_cost.dollars_per_million_input_tokens - real_cost.dollars_per_million_input_tokens) / real_cost.dollars_per_million_input_tokens) * 100 if real_cost.dollars_per_million_input_tokens > 0 else 0

        # Print comparison
        print(f"  Metric                      Simulated      Measured    Error")
        print(f"  -------------------------  -----------  -----------  --------")
        print(f"  Wall clock (s)             {sim_wall_clock:11.2f}  {real_wall_clock:11.2f}  {wall_clock_error_pct:+7.1f}%")
        print(f"  Input tok/sec              {sim_input_throughput:11.1f}  {real_input_throughput:11.1f}  {input_throughput_error_pct:+7.1f}%")
        print(f"  Output tok/sec             {sim_output_throughput:11.1f}  {real_output_throughput:11.1f}  {output_throughput_error_pct:+7.1f}%")
        print(f"  TTFT (s)                   {'N/A':>11}  {avg_ttft:11.3f}  {'N/A':>8}")
        print(f"  $/M input tokens           {sim_cost.dollars_per_million_input_tokens:11.2f}  {real_cost.dollars_per_million_input_tokens:11.2f}  {cost_per_m_error_pct:+7.1f}%")
        print(f"  Peak concurrent requests   {'N/A':>11}  {peak_concurrent:11d}  {'N/A':>8}")
        print()

        # Store results
        validation_results["scenarios"].append({
            "name": scenario["name"],
            "arrival_rate_req_per_sec": scenario["arrival_rate"],
            "num_requests": scenario["num_requests"],
            "simulated": {
                "wall_clock_seconds": sim_wall_clock,
                "input_throughput_tok_per_sec": sim_input_throughput,
                "output_throughput_tok_per_sec": sim_output_throughput,
                "dollars_per_million_input": sim_cost.dollars_per_million_input_tokens,
                "dollars_per_million_output": sim_cost.dollars_per_million_output_tokens,
                "total_dollars": sim_cost.total_dollars,
            },
            "measured": {
                "wall_clock_seconds": real_wall_clock,
                "input_throughput_tok_per_sec": real_input_throughput,
                "output_throughput_tok_per_sec": real_output_throughput,
                "avg_ttft_seconds": avg_ttft,
                "dollars_per_million_input": real_cost.dollars_per_million_input_tokens,
                "dollars_per_million_output": real_cost.dollars_per_million_output_tokens,
                "total_dollars": real_cost.total_dollars,
                "peak_concurrent_requests": peak_concurrent,
                "per_request_timings": measured_requests,
            },
            "errors_pct": {
                "wall_clock": wall_clock_error_pct,
                "input_throughput": input_throughput_error_pct,
                "output_throughput": output_throughput_error_pct,
                "cost_per_million_input": cost_per_m_error_pct,
            }
        })

    # Write JSON output
    output_file = f"validation_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(validation_results, f, indent=2)

    print(f"Full results written to: {output_file}")
    print()

    return EXIT_OK


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

    print(f"Starting Throttle proxy server on {args.host}:{args.port}")
    print(f"Backend: {args.backend_url}")
    print(f"Backend timeout: {args.backend_timeout_seconds}s")
    print(f"Cache enabled: {args.enable_cache}")
    if args.enable_cache:
        print(f"  TTL: {args.cache_ttl_seconds}s")
        print(f"  Max size: {args.cache_max_size}")
        print(f"  Similarity threshold: {args.cache_similarity_threshold}")

        # Startup diagnostics for embeddings
        if enable_embeddings_resolved and embeddings.EMBEDDINGS_AVAILABLE:
            # State a: embeddings active
            print(f"  Embeddings: ACTIVE (model: sentence-transformers/all-MiniLM-L6-v2)")
        elif enable_embeddings_resolved and not embeddings.EMBEDDINGS_AVAILABLE:
            # State b: requested but extra missing
            print(f"  Embeddings: REQUESTED BUT UNAVAILABLE")
            print(f"    Install with: pip install throttle-pro[embeddings]")
        elif not enable_embeddings_resolved:
            # State c: disabled by explicit request or not applicable
            if args.no_embeddings:
                print(f"  Embeddings: DISABLED (explicit --no-embeddings)")
            else:
                print(f"  Embeddings: OFF (Jaccard lexical matching only)")
    print()
    print("Health endpoint: http://{args.host}:{args.port}/health")
    print("Chat completions: http://{args.host}:{args.port}/v1/chat/completions")
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
                    print("  2. Or use Ollama:")
                    print("     ollama serve")
                    print("  3. Then run 'throttle watch' again")
                    print("  4. Or specify a different endpoint:")
                    print(f"     throttle watch --url http://host:port/metrics --gpu-rate-per-hour {args.gpu_rate_per_hour}")
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
    # One-line summary: understandable by a founder or eng lead, not just an engineer
    if snap.cost_per_million_tokens is not None and snap.generation_throughput_toks_per_sec is not None:
        print(f"  You're spending ${snap.cost_per_hour:.2f}/hr to generate "
              f"{snap.generation_throughput_toks_per_sec:.0f} tok/s "
              f"(${snap.cost_per_million_tokens:.2f} per million tokens)")
    elif snap.refusals:
        print(f"  Cost unavailable — {snap.refusals[0]['reason'][:60]}")
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
    # Tier 2 suggestion — only appears after 5 minutes of observation
    suggestion = getattr(snap, "suggestion", None)
    window_ready = getattr(snap, "window_ready", False)
    window_minutes = getattr(snap, "window_elapsed_minutes", 0.0)
    if suggestion:
        print(f"  💡 {suggestion['action']}")
        print(f"     {suggestion['reason']}")
        if suggestion.get("suggested_next_step"):
            print(f"     Next: {suggestion['suggested_next_step']}")
    elif not window_ready and window_minutes > 0:
        print(f"  ⏳ Collecting baseline... ({window_minutes:.0f}m of 5m needed for suggestions)")
    print()


def main(argv: Sequence[str] | None = None) -> int:
    from .config import load_config, apply_config_defaults

    parser = build_parser()

    # Load config file and apply as defaults (CLI flags will override)
    config = load_config()
    apply_config_defaults(parser, config)

    args = parser.parse_args(argv)
    if args.command in {"plan", "smoke", "benchmark"}:
        _warn_if_exploratory_sweep(args)
    if args.command == "plan":
        config, prompts, warmup_prompts = _build_config(parser, args, resolve_key=False)
        plan = build_plan(config, prompts, warmup_prompts)
        _print_plan(plan, guidellm_prompt_tokens=args.guidellm_prompt_tokens)
        return EXIT_OK
    if args.command in {"smoke", "benchmark"}:
        return _handle_run(parser, args)
    if args.command == "diagnose":
        return _handle_diagnose(parser, args)
    if args.command == "experimental-tuning":
        return _handle_experimental_tuning(parser, args)
    if args.command == "golden":
        return _handle_golden(parser, args)
    if args.command == "golden-report":
        return _handle_golden_report(args)
    if args.command == "report":
        return _handle_report(args)
    if args.command == "compare":
        # Detect if these are measure outputs (have ci_95_input field) or benchmark reports
        try:
            with open(args.reports[0]) as f:
                first_report = json.load(f)

            if "ci_95_input" in first_report:
                # Measure outputs - use new comparison logic
                return _handle_compare_measure(args.reports)
        except (FileNotFoundError, json.JSONDecodeError, KeyError, IndexError):
            # Fall through to original benchmark report comparison
            pass

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
    if args.command == "demo":
        return _handle_demo(args)
    if args.command == "cost":
        return _handle_cost(args)
    if args.command == "measure":
        return _handle_measure(args)
    if args.command == "validate-sim":
        return _handle_validate_sim(args)
    if args.command == "proxy":
        return _handle_proxy(args)
    if args.command == "watch":
        return _handle_watch(args)
    parser.error("a subcommand is required")
    return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
