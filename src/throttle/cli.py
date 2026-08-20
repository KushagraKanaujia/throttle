"""Command-line interface for Throttle v2."""

from __future__ import annotations

import argparse
import asyncio
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
    run_native,
    validate_config,
)
from .compare import (
    ComparisonInputError,
    compare_reports,
    load_report,
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
from .request_profile import validate_request_fields

DEFAULT_OUTPUT = Path("throttle-report.json")
DEFAULT_COMPARE_OUTPUT = Path("throttle-comparison.json")
DEFAULT_GOLDEN_OUTPUT_DIR = Path("throttle-golden")
DEFAULT_EXPERIMENTAL_OUTPUT = Path("throttle-experimental-tuning.json")
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


def _probability(value: str) -> float:
    parsed = _positive_float(value)
    if parsed > 1:
        raise argparse.ArgumentTypeError("must be at most one")
    return parsed


def _reject_json_constant(_: str) -> None:
    raise ValueError("non-finite JSON number")


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
    parser.add_argument("--temperature", type=_non_negative_float, default=0)
    parser.add_argument("--top-p", type=_probability)
    parser.add_argument("--request-seed", type=int)
    parser.add_argument(
        "--request-field",
        action="append",
        default=[],
        metavar="NAME=JSON_SCALAR",
        help="safe non-standard request field; repeat as needed",
    )


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


def _request_fields(
    parser: argparse.ArgumentParser, values: Sequence[str]
) -> tuple[tuple[str, str | int | float | bool | None], ...]:
    parsed: list[tuple[str, str | int | float | bool | None]] = []
    for item in values:
        if "=" not in item:
            _parser_error(parser, "--request-field must use NAME=JSON_SCALAR")
        name, encoded = item.split("=", 1)
        if not name or not encoded:
            _parser_error(parser, "--request-field needs a non-empty name and value")
        try:
            value = json.loads(
                encoded,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, ValueError):
            _parser_error(parser, "--request-field value must be valid JSON")
        parsed.append((name, value))  # type: ignore[arg-type]
    output = tuple(parsed)
    try:
        validate_request_fields(output)
    except ValueError as exc:
        _parser_error(parser, str(exc))
    return tuple(sorted(output, key=lambda pair: pair[0]))


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
    if args.command == "experimental-tuning":
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
    if args.backend == "guidellm" and args.guidellm_prompt_tokens is None:
        _parser_error(parser, "--backend guidellm requires --guidellm-prompt-tokens")
    if args.warmup_requests is not None and args.warmup_requests < 0:
        _parser_error(parser, "--warmup-requests must be non-negative")
    blocks = args.blocks if args.blocks is not None else (1 if mode == "smoke" else 3)
    requests_per_block = args.requests_per_block
    if requests_per_block is None and args.block_seconds is None:
        requests_per_block = 201 if experimental else 8 if mode == "smoke" else 67
    warmups = args.warmup_requests
    if warmups is None:
        warmups = 3 if experimental else 1 if mode == "smoke" else 3
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
        temperature=args.temperature,
        top_p=args.top_p,
        request_seed=args.request_seed,
        request_fields=_request_fields(parser, args.request_field),
        stream=args.stream,
        cache_policy=args.cache_policy,
        model_revision=args.model_revision,
        image_digest=args.image_digest,
        gpu=args.gpu,
        gpu_fingerprint=args.gpu_fingerprint,
        cuda_version=args.cuda_version,
        driver_version=args.driver_version,
        accelerator_backend=args.accelerator_backend,
        accelerator_runtime_version=args.accelerator_runtime_version,
        host_os_version=args.host_os_version,
        software_environment_digest=args.software_environment_digest,
        server_version=args.server_version,
        engine_flags=_engine_flags(parser, args.engine_flag),
        engine_flags_provenance=args.engine_flags_provenance,
        variant=args.variant,
        sequence_position=args.sequence_position,
        allow_unknown_cost=args.allow_unknown_cost,
        allow_insecure_http=args.allow_insecure_http,
        evidence_source=args.evidence_source,
        guidellm_gaps_acknowledged=args.allow_guidellm_validation_gaps,
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
    request_profile = plan.get("request_profile")
    if request_profile is not None:
        print(
            "Request profile: "
            + json.dumps(
                request_profile,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
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
    print(
        " condition              valid  grade   requests  block-mean tok/s   p95 e2e   p95 TTFT  SLO goodput"
    )
    for item in report["conditions"]:
        condition = item["condition"]
        metrics = item.get("metrics") or {}
        slo = metrics.get("slo_goodput")
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
        except BaseException as exc:  # forwarded to the main thread below
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
        ),  # type: ignore[arg-type]
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
        raise AssertionError("argparse.error must terminate")  # pragma: no cover

    config, prompts, warmup_prompts = _build_config(
        parser,
        args,
        resolve_key=False,
    )
    try:
        validate_experimental_config(config)
    except ExperimentalTuningError as exc:
        _parser_error(parser, exc.code)
        raise AssertionError("argparse.error must terminate")  # pragma: no cover

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


def _handle_golden(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    (
        baseline_flag,
        candidate_flag,
        baseline_value,
        candidate_value,
    ) = _golden_config_flags(parser, args)
    if args.concurrency is None and args.request_rate is None:
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
    return EXIT_OK if result["decision_eligible"] else EXIT_INCONCLUSIVE


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
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
    if args.command == "experimental-tuning":
        return _handle_experimental_tuning(parser, args)
    if args.command == "golden":
        return _handle_golden(parser, args)
    if args.command == "compare":
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
    parser.error("a subcommand is required")
    return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
