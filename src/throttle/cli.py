"""Command-line interface for Throttle v2."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import __version__
from .benchmark import (
    ARTIFACT_TYPE,
    SCHEMA_VERSION,
    RunProgress,
    build_plan,
    load_prompts,
    run_native,
    validate_config,
)
from .compare import ComparisonInputError, compare_reports, load_report
from .golden import validate_golden_sequence
from .models import CostModel, EndpointConfig, LoadCondition, RunConfig, SafetyLimits

DEFAULT_OUTPUT = Path("throttle-report.json")
DEFAULT_COMPARE_OUTPUT = Path("throttle-comparison.json")
ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_INCONCLUSIVE = 3
EXIT_CANCELLED = 130


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
    parser.add_argument("--max-elapsed-seconds", type=_positive_float, default=900.0)
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
    parser.add_argument("--gpu", default="unknown")
    parser.add_argument("--gpu-fingerprint", default="unknown")
    parser.add_argument("--cuda-version", default="unknown")
    parser.add_argument("--driver-version", default="unknown")
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
        "benchmark", help="run sustained repeated measurement blocks"
    )
    _add_run_options(benchmark)
    benchmark.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

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


def _resolve_key(parser: argparse.ArgumentParser, env_name: str) -> str:
    if not ENV_NAME_PATTERN.fullmatch(env_name):
        _parser_error(parser, "--api-key-env must be a valid environment variable name")
    key = os.environ.get(env_name)
    if not key:
        _parser_error(parser, "API key environment variable is missing or empty")
    return key


def _run_mode(args: argparse.Namespace) -> str:
    return args.run_mode if args.command == "plan" else args.command


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
    if args.backend == "guidellm" and args.guidellm_prompt_tokens is None:
        _parser_error(parser, "--backend guidellm requires --guidellm-prompt-tokens")
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
        concurrency = args.concurrency or [1, 4, 8]
        conditions = tuple(
            LoadCondition("closed_loop", float(level), level) for level in concurrency
        )
    limits = SafetyLimits(
        max_requests=args.max_requests,
        max_tokens_per_request=args.max_tokens_per_request,
        max_total_requested_tokens=args.max_total_requested_tokens,
        max_elapsed_seconds=args.max_elapsed_seconds,
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
        cache_policy=args.cache_policy,
        model_revision=args.model_revision,
        image_digest=args.image_digest,
        gpu=args.gpu,
        gpu_fingerprint=args.gpu_fingerprint,
        cuda_version=args.cuda_version,
        driver_version=args.driver_version,
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


def _atomic_write(report: Mapping[str, Any], output: Path) -> None:
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
        os.replace(temporary, output)
        os.chmod(output, 0o600)
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


def _print_golden(report: Mapping[str, Any], output: Path) -> None:
    print(
        "Throttle golden protocol validation (six sequential saved runs; no traffic sent)"
    )
    print(f"Protocol eligible: {'yes' if report['golden_protocol_eligible'] else 'NO'}")
    if report["eligibility_reasons"]:
        print("Reasons: " + ", ".join(report["eligibility_reasons"]))
    for condition in report["conditions"]:
        interval = condition["throughput_delta_percent_ci"]
        print(
            f" {condition['condition_id']}: {condition['state']}; throughput delta "
            f"{_metric(interval.get('estimate'))}% "
            f"(95% CI {_metric(interval.get('low'))}% to {_metric(interval.get('high'))}%)"
        )
    print(f"Outcome: {report.get('overall_outcome') or 'inconclusive'}")
    print(report["disclaimer"])
    print(f"Sanitized golden artifact: {output}")


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "plan":
        config, prompts, warmup_prompts = _build_config(parser, args, resolve_key=False)
        plan = build_plan(config, prompts, warmup_prompts)
        _print_plan(plan, guidellm_prompt_tokens=args.guidellm_prompt_tokens)
        return EXIT_OK
    if args.command in {"smoke", "benchmark"}:
        return _handle_run(parser, args)
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
