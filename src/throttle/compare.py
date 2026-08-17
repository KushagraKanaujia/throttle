"""Offline, fail-closed comparison of saved Throttle v2 benchmark reports."""

from __future__ import annotations

import json
import math
import os
import re
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from . import __version__
from .benchmark import ALLOWED_FINISH_REASONS, ARTIFACT_TYPE, SCHEMA_VERSION
from .models import (
    OPEN_LOOP_RATE_RELATIVE_TOLERANCE,
    OPEN_LOOP_SCHEDULER_LAG_INTERVAL_TOLERANCE,
    LoadCondition,
)
from .statistics import (
    paired_relative_delta_interval_95,
    relative_delta_percent,
    t_interval_95,
)

COMPARISON_ARTIFACT_TYPE = "throttle_comparison"
MAX_REPORT_BYTES = 20_000_000
OUTPUT_TOKEN_TOLERANCE = 0.05
MAX_SAFE_NUMERIC_MAGNITUDE = 9_007_199_254_740_991


class ComparisonInputError(ValueError):
    """A sanitized input error whose message never includes raw report values."""


def _reject_json_constant(_: str) -> None:
    raise ValueError("non-finite JSON number")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate JSON object key")
        output[key] = value
    return output


def load_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path).expanduser()
    try:
        flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(report_path, flags)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ComparisonInputError("saved report must be a regular file")
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                encoded = handle.read(MAX_REPORT_BYTES + 1)
        finally:
            os.close(descriptor)
        if len(encoded) > MAX_REPORT_BYTES:
            raise ComparisonInputError("saved report exceeds the comparison size limit")
        rendered = encoded.decode("utf-8", errors="strict")
        parsed = json.loads(
            rendered,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except ComparisonInputError:
        raise
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ValueError,
        RecursionError,
    ) as exc:
        raise ComparisonInputError(
            "saved report is unreadable or not valid JSON"
        ) from exc
    if not isinstance(parsed, dict):
        raise ComparisonInputError("saved report must contain a JSON object")
    return parsed


def _path(report: Mapping[str, Any], *parts: str) -> Any:
    current: Any = report
    for part in parts:
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _finite_number(value: object, *, positive: bool = False) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if isinstance(value, int):
        finite_and_bounded = abs(value) <= MAX_SAFE_NUMERIC_MAGNITUDE
    else:
        finite_and_bounded = (
            math.isfinite(value) and abs(value) <= MAX_SAFE_NUMERIC_MAGNITUDE
        )
    return finite_and_bounded and (value > 0 if positive else True)


def _normalized_engine_flags(value: object) -> dict[str, str] | None:
    """Validate persisted flags before a name can enter a derived artifact."""

    if not isinstance(value, Mapping):
        return None
    forbidden_name_markers = (
        "api-key",
        "apikey",
        "auth-token",
        "access-token",
        "refresh-token",
        "bearer-token",
        "id-token",
        "private-key",
        "key-file",
        "keyfile",
        "ssl-key",
        "tls-key",
        "secret",
        "password",
        "credential",
        "authorization",
        "bearer",
        "header",
        "url",
        "endpoint",
    )
    normalized: dict[str, str] = {}
    for raw_name, raw_value in value.items():
        if (
            not isinstance(raw_name, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", raw_name)
            or any(
                marker in raw_name.replace("_", "-").lower()
                for marker in forbidden_name_markers
            )
            or not isinstance(raw_value, str)
            or not raw_value
            or raw_value != raw_value.strip()
            or len(raw_value) > 256
        ):
            return None
        lowered_value = raw_value.lower()
        if any(
            marker in lowered_value
            for marker in (
                "://",
                "bearer ",
                "authorization:",
                "api_key",
                "api-key",
            )
        ):
            return None
        name = raw_name.replace("_", "-").lower()
        if name in normalized:
            return None
        normalized[name] = raw_value
    return normalized


def _positive_int(value: object) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 < value <= MAX_SAFE_NUMERIC_MAGNITUDE
    )


def _valid_success_count_maps(counts: Mapping[str, Any], attempted: int) -> bool:
    """Prove that persisted count maps describe only strict HTTP-200 completions."""

    status_counts = counts.get("status_counts")
    error_counts = counts.get("error_counts")
    finish_counts = counts.get("finish_reason_counts")
    if not all(
        isinstance(value, Mapping)
        for value in (status_counts, error_counts, finish_counts)
    ):
        return False
    expected_statuses = {"200": attempted} if attempted else {}
    if dict(status_counts) != expected_statuses or dict(error_counts) != {}:
        return False
    if any(
        name not in ALLOWED_FINISH_REASONS
        or not isinstance(count, int)
        or isinstance(count, bool)
        or count <= 0
        for name, count in finish_counts.items()
    ):
        return False
    return sum(finish_counts.values()) == attempted


def _valid_safety_contract(safety: object) -> bool:
    if not isinstance(safety, Mapping):
        return False
    limits = safety.get("limits")
    overrides = safety.get("overrides")
    if not isinstance(limits, Mapping) or not isinstance(overrides, Mapping):
        return False
    for name in (
        "max_requests",
        "max_tokens_per_request",
        "max_total_requested_tokens",
        "max_errors",
        "max_concurrency",
        "max_response_bytes",
    ):
        if not _positive_int(limits.get(name)):
            return False
    for name in ("max_elapsed_seconds", "max_estimated_spend"):
        if not _finite_number(limits.get(name), positive=True):
            return False
    for name in (
        "insecure_http",
        "unknown_cost_acknowledged",
        "guidellm_validation_gaps_acknowledged",
    ):
        if not isinstance(overrides.get(name), bool):
            return False
    return (
        safety.get("ambient_proxy_environment_used") is False
        and safety.get("redirects_followed") is False
    )


def _valid_cost_manifest(cost: object, limits: Mapping[str, Any]) -> bool:
    if not isinstance(cost, Mapping) or cost.get("currency") != "USD":
        return False
    kind = cost.get("kind")
    max_elapsed = limits.get("max_elapsed_seconds")
    if not _finite_number(max_elapsed, positive=True):
        return False
    pre_run = cost.get("pre_run_upper_bound")
    if not isinstance(cost.get("accounting_basis"), str):
        return False
    cross_model_fields = {
        "total_hourly_rate": cost.get("total_hourly_rate"),
        "gpu_count": cost.get("gpu_count"),
        "active_second_rate": cost.get("active_second_rate"),
        "max_active_workers": cost.get("max_active_workers"),
        "billed_active_seconds": cost.get("billed_active_seconds"),
        "user_supplied_total": cost.get("user_supplied_total"),
    }
    if kind == "unknown":
        return pre_run is None and all(
            value is None for value in cross_model_fields.values()
        )
    if kind == "dedicated_hourly":
        rate = cost.get("total_hourly_rate")
        if not _finite_number(rate, positive=True) or not _positive_int(
            cost.get("gpu_count")
        ):
            return False
        if any(
            cross_model_fields[name] is not None
            for name in (
                "active_second_rate",
                "max_active_workers",
                "billed_active_seconds",
                "user_supplied_total",
            )
        ):
            return False
        return _finite_number(pre_run, positive=True) and _close(
            float(pre_run), float(rate) * float(max_elapsed) / 3600.0
        )
    if kind == "serverless_active_seconds":
        rate = cost.get("active_second_rate")
        workers = cost.get("max_active_workers")
        billed = cost.get("billed_active_seconds")
        if not _finite_number(rate, positive=True) or not _positive_int(workers):
            return False
        if billed is not None and (not _finite_number(billed) or float(billed) < 0):
            return False
        if any(
            cross_model_fields[name] is not None
            for name in ("total_hourly_rate", "gpu_count", "user_supplied_total")
        ):
            return False
        expected = max(
            float(rate) * int(workers) * float(max_elapsed),
            float(rate) * float(billed or 0.0),
        )
        return _finite_number(pre_run, positive=True) and _close(
            float(pre_run), expected
        )
    if kind == "user_supplied":
        total = cost.get("user_supplied_total")
        if not _finite_number(total, positive=True):
            return False
        if any(
            cross_model_fields[name] is not None
            for name in (
                "total_hourly_rate",
                "gpu_count",
                "active_second_rate",
                "max_active_workers",
                "billed_active_seconds",
            )
        ):
            return False
        return _finite_number(pre_run, positive=True) and _close(
            float(pre_run), float(total)
        )
    return False


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-9)


def _repeated_block_interval_matches(reported: Any, values: list[float]) -> bool:
    if not isinstance(reported, Mapping):
        return False
    expected = t_interval_95(values)
    if (
        reported.get("method") != "student_t_blocks"
        or reported.get("confidence") != 0.95
        or reported.get("n") != len(values)
    ):
        return False
    for endpoint in ("low", "high"):
        observed = reported.get(endpoint)
        target = expected.get(endpoint)
        if target is None:
            if observed is not None:
                return False
        elif not _finite_number(observed) or not _close(float(observed), float(target)):
            return False
    return True


def _preflight_reason(report: Mapping[str, Any]) -> str | None:
    schema = report.get("schema_version")
    if schema == "1.0":
        return "legacy_schema_not_decision_grade"
    if schema != SCHEMA_VERSION:
        return "unsupported_schema"
    if report.get("artifact_type") != ARTIFACT_TYPE:
        return "unsupported_artifact_type"
    if report.get("mode") != "benchmark":
        return "smoke_or_nonbenchmark_report"
    if report.get("status") != "complete":
        return "partial_or_failed_report"
    if report.get("stop_reason") is not None:
        return "complete_report_has_stop_reason"
    manifest = report.get("manifest")
    if not isinstance(manifest, Mapping) or manifest.get("manifest_version") != "1.0":
        return "missing_or_invalid_manifest"
    required_text = (
        _path(report, "manifest", "tool", "name"),
        _path(report, "manifest", "tool", "version"),
        _path(report, "manifest", "engine", "backend"),
        _path(report, "manifest", "engine", "backend_version"),
        _path(report, "manifest", "engine", "http_client_version"),
        _path(report, "manifest", "engine", "server_version"),
        _path(report, "manifest", "model", "id"),
        _path(report, "manifest", "model", "immutable_revision"),
        _path(report, "manifest", "runtime", "image_digest"),
        _path(report, "manifest", "runtime", "gpu"),
        _path(report, "manifest", "runtime", "cuda_version"),
        _path(report, "manifest", "runtime", "driver_version"),
        _path(report, "manifest", "workload", "cache_policy"),
    )
    if any(
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or value.lower() == "unknown"
        for value in required_text
    ):
        return "unverified_manifest_metadata"
    for digest_path in (
        ("manifest", "runtime", "gpu_fingerprint_sha256"),
        ("manifest", "workload", "measured_sha256"),
        ("manifest", "workload", "warmup_sha256"),
    ):
        digest = _path(report, *digest_path)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            return "invalid_manifest_digest"
    if (
        _path(report, "manifest", "workload", "warmup_is_separate") is not True
        or _path(report, "manifest", "workload", "warmup_prompts_disjoint") is not True
    ):
        return "warmup_workload_not_separate"
    if _path(report, "manifest", "runtime", "gpu_fingerprint_supplied") is not True:
        return "gpu_fingerprint_not_supplied"
    seed = _path(report, "manifest", "workload", "seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        return "invalid_manifest_seed"
    request = _path(report, "manifest", "request")
    traffic = _path(report, "manifest", "traffic")
    safety_contract = _path(report, "manifest", "safety")
    flags = _path(report, "manifest", "engine", "effective_flags")
    provenance = _path(report, "manifest", "engine", "effective_flags_provenance")
    if not isinstance(request, Mapping) or not isinstance(traffic, Mapping):
        return "incomplete_manifest_contract"
    if not _valid_safety_contract(safety_contract):
        return "invalid_safety_manifest"
    safety = _path(report, "manifest", "safety", "limits")
    if not isinstance(safety, Mapping) or not _valid_cost_manifest(
        _path(report, "manifest", "cost"), safety
    ):
        return "invalid_cost_manifest"
    cost_manifest = _path(report, "manifest", "cost")
    safety_overrides = _path(report, "manifest", "safety", "overrides")
    if not isinstance(cost_manifest, Mapping) or not isinstance(
        safety_overrides, Mapping
    ):
        return "incomplete_manifest_contract"
    if cost_manifest.get("pre_run_upper_bound") is not None and float(
        cost_manifest["pre_run_upper_bound"]
    ) > float(safety["max_estimated_spend"]):
        return "pre_run_cost_exceeded_spend_limit"
    if (
        cost_manifest.get("kind") == "unknown"
        and safety_overrides.get("unknown_cost_acknowledged") is not True
    ):
        return "unknown_cost_not_acknowledged"
    if _normalized_engine_flags(flags) is None or provenance not in {
        "operator_attested",
        "runtime_verified",
    }:
        return "invalid_engine_flag_manifest"
    required_request_keys = {
        "type",
        "temperature",
        "max_tokens",
        "stop",
        "stream",
        "timeout_seconds",
    }
    required_traffic_keys = {
        "conditions",
        "blocks",
        "requests_per_block",
        "block_duration_seconds",
        "warmup_requests_per_condition",
        "p95_slo_ms",
        "ttft_slo_ms",
        "open_loop_rate_relative_tolerance",
        "open_loop_scheduler_lag_interval_tolerance",
    }
    if not required_request_keys.issubset(
        request
    ) or not required_traffic_keys.issubset(traffic):
        return "incomplete_manifest_contract"
    if (
        traffic.get("open_loop_rate_relative_tolerance")
        != OPEN_LOOP_RATE_RELATIVE_TOLERANCE
        or traffic.get("open_loop_scheduler_lag_interval_tolerance")
        != OPEN_LOOP_SCHEDULER_LAG_INTERVAL_TOLERANCE
    ):
        return "unsupported_open_loop_tolerance_contract"
    if (
        request.get("type") != "chat_completions"
        or request.get("temperature") != 0
        or request.get("stop") is not None
        or not isinstance(request.get("stream"), bool)
        or not _positive_int(request.get("max_tokens"))
        or not _finite_number(request.get("timeout_seconds"), positive=True)
        or int(request["max_tokens"]) > int(safety["max_tokens_per_request"])
    ):
        return "unsupported_request_contract"
    for slo_name in ("p95_slo_ms", "ttft_slo_ms"):
        slo = traffic.get(slo_name)
        if slo is not None and not _finite_number(slo, positive=True):
            return "invalid_slo_manifest"
    blocks_expected = traffic.get("blocks")
    if (
        not isinstance(blocks_expected, int)
        or isinstance(blocks_expected, bool)
        or blocks_expected < 3
    ):
        return "fewer_than_three_blocks"
    requests_per_block = traffic.get("requests_per_block")
    block_duration = traffic.get("block_duration_seconds")
    if (requests_per_block is None) == (block_duration is None):
        return "ambiguous_block_bound"
    if requests_per_block is not None and not _positive_int(requests_per_block):
        return "invalid_request_bound"
    if block_duration is not None and not _finite_number(block_duration, positive=True):
        return "invalid_duration_bound"
    expected_warmups = traffic.get("warmup_requests_per_condition")
    if (
        not isinstance(expected_warmups, int)
        or isinstance(expected_warmups, bool)
        or expected_warmups < 0
    ):
        return "invalid_warmup_count"
    conditions = report.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        return "missing_conditions"
    declared_conditions = traffic.get("conditions")
    if not isinstance(declared_conditions, list) or len(declared_conditions) != len(
        conditions
    ):
        return "condition_manifest_mismatch"
    total_attempted = 0
    total_completion_tokens = 0
    total_measured_wall = 0.0
    condition_peaks: list[int] = []
    for condition in conditions:
        if (
            not isinstance(condition, Mapping)
            or condition.get("valid") is not True
            or condition.get("decision_grade") is not True
        ):
            return "one_or_more_conditions_not_decision_grade"
        counts = condition.get("request_counts")
        blocks = condition.get("blocks")
        warmup = condition.get("warmup")
        wall = condition.get("measured_wall_seconds")
        metrics = condition.get("metrics")
        if not all(isinstance(value, Mapping) for value in (counts, warmup, metrics)):
            return "condition_evidence_missing"
        if not isinstance(blocks, list) or len(blocks) != blocks_expected:
            return "condition_block_count_mismatch"
        descriptor = condition.get("condition")
        if not isinstance(descriptor, Mapping):
            return "condition_descriptor_invalid"
        kind = descriptor.get("kind")
        descriptor_value = descriptor.get("value")
        declared_max = descriptor.get("max_in_flight")
        if (
            kind not in {"closed_loop", "open_loop"}
            or not _finite_number(descriptor_value, positive=True)
            or not _positive_int(declared_max)
        ):
            return "condition_descriptor_invalid"
        if kind == "closed_loop" and (
            not float(descriptor_value).is_integer()
            or int(float(descriptor_value)) != int(declared_max)
        ):
            return "condition_descriptor_invalid"
        expected_id = LoadCondition(
            kind, float(descriptor_value), int(declared_max)
        ).condition_id
        if descriptor.get("id") != expected_id:
            return "condition_id_does_not_match_descriptor"
        observed_peak = condition.get("observed_peak_in_flight")
        if (
            not isinstance(observed_peak, int)
            or isinstance(observed_peak, bool)
            or observed_peak <= 0
        ):
            return "condition_peak_in_flight_missing"
        if observed_peak > declared_max:
            return "condition_peak_in_flight_invalid"
        if kind == "closed_loop" and observed_peak != declared_max:
            return "closed_loop_target_concurrency_not_achieved"
        attempted = counts.get("attempted")
        valid = counts.get("valid")
        invalid = counts.get("invalid")
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in (attempted, valid, invalid)
        ):
            return "condition_request_counts_invalid"
        if attempted != valid + invalid or invalid != 0:
            return "condition_contains_invalid_responses"
        if not _valid_success_count_maps(counts, attempted):
            return "condition_request_count_maps_invalid"
        if not _finite_number(wall, positive=True):
            return "condition_duration_invalid"
        if valid < 200 and wall < 60.0:
            return "condition_measurement_floor_not_met"
        if warmup.get("invalid") != 0:
            return "warmup_contains_invalid_responses"
        if (
            warmup.get("attempted") != expected_warmups
            or warmup.get("valid") != expected_warmups
            or warmup.get("invalid") != 0
        ):
            return "warmup_counts_do_not_reconcile"
        if not _valid_success_count_maps(warmup, expected_warmups):
            return "warmup_request_count_maps_invalid"
        completion_tokens = metrics.get("completion_tokens")
        throughput = metrics.get("output_tokens_per_second")
        if (
            not isinstance(completion_tokens, int)
            or isinstance(completion_tokens, bool)
            or completion_tokens <= 0
        ):
            return "condition_token_evidence_invalid"
        if completion_tokens < valid:
            return "condition_token_evidence_invalid"
        if completion_tokens > valid * int(request["max_tokens"]):
            return "condition_tokens_exceed_request_contract"
        if not _finite_number(throughput, positive=True):
            return "condition_throughput_invalid"
        summed_attempted = 0
        summed_valid = 0
        summed_invalid = 0
        summed_tokens = 0
        summed_wall = 0.0
        block_peaks: list[int] = []
        summed_finish_reasons: dict[str, int] = {}
        summed_launch_gaps = 0
        summed_launch_window = 0.0
        block_output_rates: list[float] = []
        block_request_rates: list[float] = []
        block_p95_values: dict[str, list[float]] = {
            "e2e_latency_ms": [],
            "ttft_ms": [],
        }
        for expected_block_index, block in enumerate(blocks, start=1):
            if not isinstance(block, Mapping) or block.get("valid") is not True:
                return "invalid_or_partial_block"
            if block.get("block_index") != expected_block_index:
                return "block_indexes_not_sequential"
            if block.get("invalid_reasons") not in ([], ()):
                return "invalid_or_partial_block"
            block_counts = block.get("request_counts")
            block_metrics = block.get("metrics")
            if (
                not isinstance(block_counts, Mapping)
                or block_counts.get("invalid") != 0
                or not isinstance(block_metrics, Mapping)
            ):
                return "invalid_or_partial_block"
            value = block_metrics.get("output_tokens_per_second")
            if not _finite_number(value, positive=True):
                return "invalid_block_metrics"
            block_attempted = block_counts.get("attempted")
            block_valid = block_counts.get("valid")
            block_invalid = block_counts.get("invalid")
            block_tokens = block_metrics.get("completion_tokens")
            block_wall = block.get("wall_duration_seconds")
            block_peak = block.get("observed_peak_in_flight")
            if not all(
                isinstance(item, int) and not isinstance(item, bool) and item >= 0
                for item in (block_attempted, block_valid, block_invalid)
            ):
                return "invalid_block_counts"
            if block_attempted != block_valid + block_invalid or block_invalid != 0:
                return "invalid_block_counts"
            if not _valid_success_count_maps(block_counts, block_attempted):
                return "invalid_block_count_maps"
            if requests_per_block is not None and block_attempted != requests_per_block:
                return "block_request_count_does_not_match_manifest"
            if (
                not isinstance(block_tokens, int)
                or isinstance(block_tokens, bool)
                or block_tokens <= 0
            ):
                return "invalid_block_token_evidence"
            if block_tokens < block_valid:
                return "invalid_block_token_evidence"
            if block_tokens > block_valid * int(request["max_tokens"]):
                return "block_tokens_exceed_request_contract"
            if not _finite_number(block_wall, positive=True):
                return "invalid_block_duration"
            if block_duration is not None and float(block_wall) < float(block_duration):
                return "block_duration_does_not_meet_manifest"
            if block.get("offered_requests") != block_attempted:
                return "block_offered_requests_do_not_reconcile"
            if (
                not isinstance(block_peak, int)
                or isinstance(block_peak, bool)
                or block_peak <= 0
                or block_peak > declared_max
            ):
                return "invalid_block_peak_in_flight"
            if kind == "closed_loop" and block_peak != declared_max:
                return "closed_loop_target_concurrency_not_achieved_in_every_block"
            if kind == "open_loop":
                target_rate = block.get("target_offered_request_rate")
                launch_window = block.get("launch_window_seconds")
                achieved_rate = block.get("achieved_offered_request_rate")
                relative_error = block.get("offered_rate_relative_error")
                lag_ratio = block.get("scheduler_lag_interval_ratio_p95")
                scheduler_lag = block.get("scheduler_lag_ms")
                if (
                    not _finite_number(target_rate, positive=True)
                    or not _close(float(target_rate), float(descriptor_value))
                    or block_attempted < 2
                    or not _finite_number(launch_window, positive=True)
                    or float(launch_window) > float(block_wall)
                    or not _finite_number(achieved_rate, positive=True)
                    or not _finite_number(relative_error)
                    or float(relative_error) < 0
                    or not _finite_number(lag_ratio)
                    or float(lag_ratio) < 0
                    or not isinstance(scheduler_lag, Mapping)
                ):
                    return "open_loop_rate_evidence_invalid"
                expected_achieved = (block_attempted - 1) / float(launch_window)
                expected_error = abs(
                    expected_achieved - float(descriptor_value)
                ) / float(descriptor_value)
                scheduler_values = [
                    scheduler_lag.get(name)
                    for name in ("mean", "p50", "p90", "p95", "p99")
                ]
                if (
                    scheduler_lag.get("count") != block_attempted
                    or any(
                        not _finite_number(item) or float(item) < 0
                        for item in scheduler_values
                    )
                    or not all(
                        float(left) <= float(right)
                        for left, right in zip(
                            scheduler_values[1:], scheduler_values[2:]
                        )
                    )
                    or not _close(float(achieved_rate), expected_achieved)
                    or not _close(float(relative_error), expected_error)
                    or not _close(
                        float(lag_ratio),
                        float(scheduler_lag["p95"])
                        / (1000.0 / float(descriptor_value)),
                    )
                ):
                    return "open_loop_rate_evidence_does_not_reconcile"
                if (
                    expected_error > OPEN_LOOP_RATE_RELATIVE_TOLERANCE
                    or float(lag_ratio) > OPEN_LOOP_SCHEDULER_LAG_INTERVAL_TOLERANCE
                    or block.get("open_loop_target_achieved") is not True
                ):
                    return "open_loop_target_rate_not_achieved"
                summed_launch_gaps += block_attempted - 1
                summed_launch_window += float(launch_window)
            if not _close(float(value), block_tokens / float(block_wall)):
                return "block_throughput_does_not_reconcile"
            block_request_rate = block_metrics.get("requests_per_second")
            if not _finite_number(block_request_rate, positive=True) or not _close(
                float(block_request_rate), block_valid / float(block_wall)
            ):
                return "block_request_throughput_does_not_reconcile"
            block_output_rates.append(float(value))
            block_request_rates.append(float(block_request_rate))
            if (
                block_metrics.get("valid_response_count") != block_valid
                or block_metrics.get("error_rate") != 0
            ):
                return "block_metric_counts_do_not_reconcile"
            for metric_name in block_p95_values:
                distribution = block_metrics.get(metric_name)
                if not isinstance(distribution, Mapping):
                    return "invalid_block_latency_evidence"
                if metric_name == "ttft_ms" and request.get("stream") is False:
                    if (
                        distribution.get("count") != 0
                        or distribution.get("p95") is not None
                    ):
                        return "invalid_nonstream_ttft_evidence"
                    continue
                if not _finite_number(distribution.get("p95"), positive=True):
                    return "invalid_block_latency_evidence"
                block_p95_values[metric_name].append(float(distribution["p95"]))
            for reason, count in block_counts["finish_reason_counts"].items():
                summed_finish_reasons[reason] = (
                    summed_finish_reasons.get(reason, 0) + count
                )
            summed_attempted += block_attempted
            summed_valid += block_valid
            summed_invalid += block_invalid
            summed_tokens += block_tokens
            summed_wall += float(block_wall)
            block_peaks.append(block_peak)
        if (summed_attempted, summed_valid, summed_invalid) != (
            attempted,
            valid,
            invalid,
        ):
            return "condition_counts_do_not_reconcile"
        if dict(counts["finish_reason_counts"]) != summed_finish_reasons:
            return "condition_finish_reason_counts_do_not_reconcile"
        if summed_tokens != completion_tokens:
            return "condition_tokens_do_not_reconcile"
        if not _close(summed_wall, float(wall)):
            return "condition_wall_time_does_not_reconcile"
        if not _close(float(throughput), completion_tokens / float(wall)):
            return "condition_throughput_does_not_reconcile"
        request_rate = metrics.get("requests_per_second")
        if not _finite_number(request_rate, positive=True) or not _close(
            float(request_rate), valid / float(wall)
        ):
            return "condition_request_throughput_does_not_reconcile"
        if (
            metrics.get("valid_response_count") != valid
            or metrics.get("error_rate") != 0
        ):
            return "condition_metric_counts_do_not_reconcile"
        block_mean_output = metrics.get("block_mean_output_tokens_per_second")
        block_mean_requests = metrics.get("block_mean_requests_per_second")
        if (
            not _finite_number(block_mean_output, positive=True)
            or not _finite_number(block_mean_requests, positive=True)
            or not _close(
                float(block_mean_output),
                math.fsum(block_output_rates) / len(block_output_rates),
            )
            or not _close(
                float(block_mean_requests),
                math.fsum(block_request_rates) / len(block_request_rates),
            )
            or not _repeated_block_interval_matches(
                metrics.get("block_mean_output_tokens_per_second_ci"),
                block_output_rates,
            )
            or not _repeated_block_interval_matches(
                metrics.get("block_mean_requests_per_second_ci"),
                block_request_rates,
            )
        ):
            return "repeated_block_throughput_estimate_does_not_reconcile"
        if max(block_peaks) != observed_peak:
            return "condition_peak_in_flight_does_not_reconcile"
        if kind == "open_loop":
            aggregate_achieved = (
                summed_launch_gaps / summed_launch_window
                if summed_launch_gaps > 0 and summed_launch_window > 0
                else None
            )
            aggregate_error = (
                abs(aggregate_achieved - float(descriptor_value))
                / float(descriptor_value)
                if aggregate_achieved is not None
                else None
            )
            reported_achieved = condition.get("achieved_offered_request_rate")
            reported_error = condition.get("offered_rate_relative_error")
            if (
                not _finite_number(aggregate_achieved, positive=True)
                or not _finite_number(aggregate_error)
                or not _finite_number(reported_achieved, positive=True)
                or not _finite_number(reported_error)
                or condition.get("target_offered_request_rate") != descriptor_value
                or not _close(
                    float(reported_achieved),
                    float(aggregate_achieved),
                )
                or not _close(
                    float(reported_error),
                    float(aggregate_error),
                )
                or condition.get("open_loop_target_achieved") is not True
            ):
                return "condition_open_loop_rate_evidence_does_not_reconcile"
        for slo_name, metric_name in (
            ("p95_slo_ms", "e2e_latency_ms"),
            ("ttft_slo_ms", "ttft_ms"),
        ):
            aggregate_distribution = metrics.get(metric_name)
            if not isinstance(
                aggregate_distribution, Mapping
            ) or not _repeated_block_interval_matches(
                aggregate_distribution.get("p95_repeated_block_ci"),
                block_p95_values[metric_name],
            ):
                return "repeated_block_slo_interval_does_not_reconcile"
            if traffic.get(slo_name) is not None:
                high = aggregate_distribution["p95_repeated_block_ci"]["high"]
                if not _finite_number(high, positive=True):
                    return "invalid_slo_confidence_interval"
        total_attempted += attempted + expected_warmups
        total_completion_tokens += completion_tokens
        total_measured_wall += float(wall)
        condition_peaks.append(observed_peak)
    actual_descriptors = [condition.get("condition") for condition in conditions]
    if actual_descriptors != declared_conditions:
        return "condition_manifest_mismatch"

    run_totals = report.get("run_totals")
    if not isinstance(run_totals, Mapping):
        return "run_totals_missing"
    integer_totals = (
        "requests_started",
        "requests_completed",
        "requests_cancelled",
        "requests_in_flight",
        "peak_in_flight",
        "errors",
        "reserved_output_tokens",
    )
    if any(
        not isinstance(run_totals.get(name), int)
        or isinstance(run_totals.get(name), bool)
        or int(run_totals[name]) < 0
        for name in integer_totals
    ):
        return "run_totals_invalid"
    if (
        run_totals["requests_started"] != total_attempted
        or run_totals["requests_completed"] != total_attempted
        or run_totals["requests_cancelled"] != 0
        or run_totals["requests_in_flight"] != 0
        or run_totals["errors"] != 0
        or run_totals["peak_in_flight"] < max(condition_peaks)
        or run_totals["reserved_output_tokens"]
        != total_attempted * int(request["max_tokens"])
    ):
        return "run_totals_do_not_reconcile"
    elapsed = run_totals.get("elapsed_seconds")
    if (
        not _finite_number(elapsed, positive=True)
        or float(elapsed) < total_measured_wall
        or float(elapsed) > float(safety["max_elapsed_seconds"])
        or total_attempted > int(safety["max_requests"])
        or int(run_totals["reserved_output_tokens"])
        > int(safety["max_total_requested_tokens"])
        or int(run_totals["peak_in_flight"]) > int(safety["max_concurrency"])
    ):
        return "run_totals_violate_safety_or_wall_time"
    try:
        started = datetime.fromisoformat(
            str(report.get("started_at")).replace("Z", "+00:00")
        )
        completed = datetime.fromisoformat(
            str(report.get("completed_at")).replace("Z", "+00:00")
        )
    except ValueError:
        return "run_timestamps_invalid"
    if (
        started.tzinfo is None
        or completed.tzinfo is None
        or completed < started
        or (completed - started).total_seconds() + 5.0 < float(elapsed)
    ):
        return "run_timestamps_do_not_reconcile"

    cost = _path(report, "manifest", "cost")
    cost_summary = report.get("cost_summary")
    if not isinstance(cost, Mapping) or not isinstance(cost_summary, Mapping):
        return "cost_evidence_missing"
    if (
        cost_summary.get("kind") != cost.get("kind")
        or cost_summary.get("completion_tokens") != total_completion_tokens
    ):
        return "cost_summary_does_not_reconcile"
    kind = cost.get("kind")
    expected_total: float | None
    if kind == "dedicated_hourly":
        expected_total = float(cost["total_hourly_rate"]) * float(elapsed) / 3600.0
    elif (
        kind == "serverless_active_seconds"
        and cost.get("billed_active_seconds") is not None
    ):
        expected_total = float(cost["active_second_rate"]) * float(
            cost["billed_active_seconds"]
        )
    elif kind == "user_supplied":
        expected_total = float(cost["user_supplied_total"])
    else:
        expected_total = None
    observed_total = cost_summary.get("total_cost")
    observed_per_million = cost_summary.get("cost_per_million_output_tokens")
    if expected_total is None:
        if observed_total is not None or observed_per_million is not None:
            return "cost_summary_does_not_reconcile"
    else:
        expected_per_million = expected_total / total_completion_tokens * 1_000_000.0
        if (
            not _finite_number(observed_total)
            or not _finite_number(observed_per_million)
            or not _close(float(observed_total), expected_total)
            or not _close(float(observed_per_million), expected_per_million)
        ):
            return "cost_summary_does_not_reconcile"
    if kind == "dedicated_hourly" and any(
        not _dedicated_cost_metric_valid(report, condition) for condition in conditions
    ):
        return "condition_cost_evidence_does_not_reconcile"
    return None


def _safe_preflight_reason(report: Mapping[str, Any]) -> str | None:
    try:
        return _preflight_reason(report)
    except (ArithmeticError, TypeError, ValueError):
        return "malformed_numeric_or_structural_evidence"


_CONTROLLED_MANIFEST_PATHS: tuple[tuple[str, ...], ...] = (
    ("manifest_version",),
    ("tool", "name"),
    ("tool", "version"),
    ("model", "id"),
    ("model", "immutable_revision"),
    ("runtime", "image_digest"),
    ("runtime", "gpu"),
    ("runtime", "gpu_fingerprint_sha256"),
    ("runtime", "gpu_fingerprint_supplied"),
    ("runtime", "cuda_version"),
    ("runtime", "driver_version"),
    ("workload", "seed"),
    ("workload", "measured_sha256"),
    ("workload", "warmup_sha256"),
    ("workload", "warmup_is_separate"),
    ("workload", "warmup_prompts_disjoint"),
    ("workload", "cache_policy"),
    ("request", "type"),
    ("request", "temperature"),
    ("request", "max_tokens"),
    ("request", "stop"),
    ("request", "stream"),
    ("request", "timeout_seconds"),
    ("traffic", "conditions"),
    ("traffic", "blocks"),
    ("traffic", "requests_per_block"),
    ("traffic", "block_duration_seconds"),
    ("traffic", "warmup_requests_per_condition"),
    ("traffic", "p95_slo_ms"),
    ("traffic", "ttft_slo_ms"),
    ("traffic", "open_loop_rate_relative_tolerance"),
    ("traffic", "open_loop_scheduler_lag_interval_tolerance"),
    ("metric_definitions",),
    ("engine", "backend"),
    ("engine", "backend_version"),
    ("engine", "http_client_version"),
    ("engine", "server_version"),
    ("safety",),
)


def _has_path(report: Mapping[str, Any], *parts: str) -> bool:
    current: Any = report
    for part in parts:
        if not isinstance(current, Mapping) or part not in current:
            return False
        current = current[part]
    return True


def _compatibility_reasons(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> list[str]:
    reasons: list[str] = []
    for parts in _CONTROLLED_MANIFEST_PATHS:
        if not _has_path(baseline, "manifest", *parts) or not _has_path(
            candidate, "manifest", *parts
        ):
            reasons.append("manifest_missing_" + "_".join(parts))
        elif _path(baseline, "manifest", *parts) != _path(
            candidate, "manifest", *parts
        ):
            reasons.append("manifest_mismatch_" + "_".join(parts))
    return reasons


def _condition_map(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    mapped: dict[str, Mapping[str, Any]] = {}
    for condition in report["conditions"]:
        identifier = _path(condition, "condition", "id")
        if (
            not isinstance(identifier, str)
            or not re.fullmatch(
                r"(?:closed_loop|open_loop):[0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?",
                identifier,
            )
            or identifier in mapped
        ):
            raise ComparisonInputError(
                "saved report has malformed or duplicate conditions"
            )
        mapped[identifier] = condition
    return mapped


def _block_metric(condition: Mapping[str, Any], metric: str) -> list[float] | None:
    values: list[float] = []
    blocks = condition.get("blocks")
    if not isinstance(blocks, list) or len(blocks) < 3:
        return None
    for block in blocks:
        value = _path(block, "metrics", metric)
        if not _finite_number(value, positive=True):
            return None
        values.append(float(value))
    return values


def _dedicated_cost_metric_valid(
    report: Mapping[str, Any], condition: Mapping[str, Any]
) -> bool:
    rate = _path(report, "manifest", "cost", "total_hourly_rate")
    wall = condition.get("measured_wall_seconds")
    tokens = _path(condition, "metrics", "completion_tokens")
    observed = _path(condition, "metrics", "cost_per_million_output_tokens")
    if not all(
        _finite_number(value, positive=True) for value in (rate, wall, tokens, observed)
    ):
        return False
    expected = float(rate) * float(wall) / 3600.0 / float(tokens) * 1_000_000.0
    return _close(float(observed), expected)


def _engine_flag_difference(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    left = _normalized_engine_flags(
        _path(baseline, "manifest", "engine", "effective_flags")
    )
    right = _normalized_engine_flags(
        _path(candidate, "manifest", "engine", "effective_flags")
    )
    if left is None or right is None:
        return {"changed_flag_names": [], "count": 0, "values_omitted_for_safety": True}
    names = sorted(
        name for name in set(left) | set(right) if left.get(name) != right.get(name)
    )
    return {
        "changed_flag_names": names,
        "count": len(names),
        "values_omitted_for_safety": True,
    }


def _attribution_guard(
    changed_flags: Mapping[str, Any],
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    names = [
        name.replace("_", "-").lower() for name in changed_flags["changed_flag_names"]
    ]
    if not names:
        return {"state": "unattributable", "reason": "no_treatment_difference_declared"}
    if set(names) == {"enable-chunked-prefill"}:
        return {
            "state": "unattributable",
            "reason": "chunked_prefill_alone_receives_no_optimization_credit_in_vllm_v1",
        }
    if set(names) == {"max-num-seqs"}:
        normalized_left = (
            _normalized_engine_flags(
                _path(baseline, "manifest", "engine", "effective_flags")
            )
            or {}
        )
        normalized_right = (
            _normalized_engine_flags(
                _path(candidate, "manifest", "engine", "effective_flags")
            )
            or {}
        )
        try:
            limits = (
                int(normalized_left["max-num-seqs"]),
                int(normalized_right["max-num-seqs"]),
            )
        except (KeyError, TypeError, ValueError):
            return {
                "state": "unattributable",
                "reason": "max_num_seqs_values_not_verifiable",
            }
        if min(limits) <= 0 or limits[0] == limits[1]:
            return {
                "state": "unattributable",
                "reason": "max_num_seqs_values_not_meaningfully_different",
            }
        required_peak = max(limits)

        def exercised(report: Mapping[str, Any]) -> bool:
            conditions = report.get("conditions")
            return isinstance(conditions, list) and any(
                isinstance(item, Mapping)
                and isinstance(item.get("observed_peak_in_flight"), int)
                and item.get("observed_peak_in_flight") >= required_peak
                for item in conditions
            )

        if not exercised(baseline) or not exercised(candidate):
            return {
                "state": "unattributable",
                "reason": "max_num_seqs_change_not_exercised_by_load",
            }
    if len(names) > 1:
        return {"state": "confounded", "reason": "multiple_engine_flags_changed"}
    return {
        "state": "controlled_difference_declared",
        "reason": "one_engine_flag_changed; causality is still not claimed",
    }


def compare_reports(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare two already-saved runs without opening a network connection."""

    generated = datetime.now(timezone.utc).isoformat()
    output: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": COMPARISON_ARTIFACT_TYPE,
        "generated_at": generated,
        "tool_version": __version__,
        "status": "incompatible",
        "decision_state": "inconclusive",
        "decision_eligible": False,
        "compatibility": {"compatible": False, "reasons": []},
        "treatment": {
            "changed_flag_names": [],
            "count": 0,
            "values_omitted_for_safety": True,
        },
        "attribution": {"state": "unavailable", "reason": "inputs_not_validated"},
        "conditions": [],
        "overall_outcome": None,
        "descriptive_statistical_outcome": None,
        "decision_ineligible_reasons": [],
        "disclaimer": "Deltas are measurements of matched saved workloads, not causal proof or projected savings.",
    }
    baseline_reason = _safe_preflight_reason(baseline)
    candidate_reason = _safe_preflight_reason(candidate)
    if baseline_reason:
        output["compatibility"]["reasons"].append("baseline_" + baseline_reason)
    if candidate_reason:
        output["compatibility"]["reasons"].append("candidate_" + candidate_reason)
    if output["compatibility"]["reasons"]:
        return output
    reasons = _compatibility_reasons(baseline, candidate)
    if reasons:
        output["compatibility"]["reasons"] = reasons
        return output
    try:
        baseline_conditions = _condition_map(baseline)
        candidate_conditions = _condition_map(candidate)
    except ComparisonInputError:
        output["compatibility"]["reasons"] = ["malformed_condition_map"]
        return output
    if set(baseline_conditions) != set(candidate_conditions):
        output["compatibility"]["reasons"] = ["condition_set_mismatch"]
        return output

    treatment = _engine_flag_difference(baseline, candidate)
    output["treatment"] = treatment
    output["attribution"] = _attribution_guard(treatment, baseline, candidate)
    output["compatibility"] = {"compatible": True, "reasons": []}
    outcomes: set[str] = set()
    cost_kinds_match = _path(baseline, "manifest", "cost", "kind") == _path(
        candidate, "manifest", "cost", "kind"
    )
    all_supported = True
    for condition_id in baseline_conditions:
        left = baseline_conditions[condition_id]
        right = candidate_conditions[condition_id]
        left_tokens = _path(left, "metrics", "completion_tokens")
        right_tokens = _path(right, "metrics", "completion_tokens")
        condition_result: dict[str, Any] = {
            "condition_id": condition_id,
            "available": False,
            "state": "inconclusive",
            "reason": None,
        }
        if not all(
            isinstance(value, int) and value > 0
            for value in (left_tokens, right_tokens)
        ):
            condition_result["reason"] = "invalid_completion_token_totals"
            output["conditions"].append(condition_result)
            all_supported = False
            continue
        token_difference = abs(left_tokens - right_tokens) / max(
            left_tokens, right_tokens
        )
        condition_result["completion_token_relative_difference"] = token_difference
        condition_result["completion_token_tolerance"] = OUTPUT_TOKEN_TOLERANCE
        if token_difference > OUTPUT_TOKEN_TOLERANCE:
            condition_result["reason"] = "completion_tokens_outside_5_percent_tolerance"
            output["conditions"].append(condition_result)
            all_supported = False
            continue
        left_block_tokens = _block_metric(left, "completion_tokens")
        right_block_tokens = _block_metric(right, "completion_tokens")
        if (
            left_block_tokens is None
            or right_block_tokens is None
            or len(left_block_tokens) != len(right_block_tokens)
        ):
            condition_result["reason"] = "missing_matched_block_token_evidence"
            output["conditions"].append(condition_result)
            all_supported = False
            continue
        block_token_differences = [
            abs(left_value - right_value) / max(left_value, right_value)
            for left_value, right_value in zip(left_block_tokens, right_block_tokens)
        ]
        condition_result["maximum_block_completion_token_relative_difference"] = max(
            block_token_differences, default=0.0
        )
        if any(
            difference > OUTPUT_TOKEN_TOLERANCE
            for difference in block_token_differences
        ):
            condition_result["reason"] = (
                "one_or_more_block_pairs_outside_5_percent_completion_token_tolerance"
            )
            output["conditions"].append(condition_result)
            all_supported = False
            continue
        p95_slo = _path(baseline, "manifest", "traffic", "p95_slo_ms")
        ttft_slo = _path(baseline, "manifest", "traffic", "ttft_slo_ms")
        slo_failed = False
        for report_condition in (left, right):
            if p95_slo is not None:
                high = _path(
                    report_condition,
                    "metrics",
                    "e2e_latency_ms",
                    "p95_repeated_block_ci",
                    "high",
                )
                if not isinstance(high, (int, float)) or high > p95_slo:
                    slo_failed = True
            if ttft_slo is not None:
                high = _path(
                    report_condition,
                    "metrics",
                    "ttft_ms",
                    "p95_repeated_block_ci",
                    "high",
                )
                if not isinstance(high, (int, float)) or high > ttft_slo:
                    slo_failed = True
        if slo_failed:
            condition_result["reason"] = "one_or_both_runs_fail_declared_slo"
            output["conditions"].append(condition_result)
            all_supported = False
            continue
        left_blocks = _block_metric(left, "output_tokens_per_second")
        right_blocks = _block_metric(right, "output_tokens_per_second")
        if (
            left_blocks is None
            or right_blocks is None
            or len(left_blocks) != len(right_blocks)
        ):
            condition_result["reason"] = "missing_matched_block_metrics"
            output["conditions"].append(condition_result)
            all_supported = False
            continue
        interval = paired_relative_delta_interval_95(left_blocks, right_blocks)
        low = interval.get("low")
        high = interval.get("high")
        state = "inconclusive"
        outcome = None
        if isinstance(low, (int, float)) and isinstance(high, (int, float)):
            if low > 0:
                state = "supported"
                outcome = "candidate_higher_throughput"
            elif high < 0:
                state = "supported"
                outcome = "baseline_higher_throughput"
        if outcome:
            outcomes.add(outcome)
        else:
            all_supported = False
        left_cost = _path(left, "metrics", "cost_per_million_output_tokens")
        right_cost = _path(right, "metrics", "cost_per_million_output_tokens")
        cost_kind = _path(baseline, "manifest", "cost", "kind")
        if not cost_kinds_match:
            cost_delta = {"available": False, "reason": "incompatible_cost_models"}
        elif cost_kind != "dedicated_hourly":
            cost_delta = {
                "available": False,
                "reason": "condition_cost_not_exactly_available",
            }
        elif not _dedicated_cost_metric_valid(
            baseline, left
        ) or not _dedicated_cost_metric_valid(candidate, right):
            cost_delta = {
                "available": False,
                "reason": "cost_evidence_does_not_reconcile",
            }
        elif all(
            isinstance(value, (int, float)) and value > 0
            for value in (left_cost, right_cost)
        ):
            cost_delta = {
                "available": True,
                "delta_percent": relative_delta_percent(
                    float(right_cost), float(left_cost)
                ),
                "claim": "descriptive cost delta; not a savings projection",
            }
        else:
            cost_delta = {
                "available": False,
                "reason": "condition_cost_not_exactly_available",
            }
        condition_result.update(
            available=True,
            state=state,
            reason=None if outcome else "paired_block_ci_includes_zero",
            outcome=outcome,
            throughput_delta_percent_ci=interval,
            cost_delta=cost_delta,
        )
        output["conditions"].append(condition_result)
    output["status"] = "complete"
    if all_supported and len(outcomes) == 1:
        output["decision_state"] = "supported"
        output["overall_outcome"] = next(iter(outcomes))
    elif len(outcomes) > 1:
        output["decision_state"] = "inconclusive"
        output["overall_outcome"] = None
        output["compatibility"]["reasons"].append("condition_outcomes_conflict")
    baseline_live = (
        _path(baseline, "manifest", "provenance", "evidence_source") == "live_inference"
    )
    candidate_live = (
        _path(candidate, "manifest", "provenance", "evidence_source")
        == "live_inference"
    )
    statistical_supported = output["decision_state"] == "supported"
    eligibility_reasons: list[str] = []
    if not baseline_live or not candidate_live:
        eligibility_reasons.append("both_inputs_must_be_live_inference")
    if output["attribution"]["state"] != "controlled_difference_declared":
        eligibility_reasons.append(str(output["attribution"]["reason"]))
    if (
        _path(baseline, "manifest", "engine", "backend") != "native"
        or _path(candidate, "manifest", "engine", "backend") != "native"
    ):
        eligibility_reasons.append("strict_native_completion_validation_required")
    baseline_streaming = _path(baseline, "manifest", "request", "stream") is True
    candidate_streaming = _path(candidate, "manifest", "request", "stream") is True
    if not baseline_streaming or not candidate_streaming:
        eligibility_reasons.append("streaming_required_for_decision_grade")
    if (
        _path(baseline, "manifest", "engine", "effective_flags_provenance")
        != "runtime_verified"
        or _path(candidate, "manifest", "engine", "effective_flags_provenance")
        != "runtime_verified"
    ):
        eligibility_reasons.append("runtime_verified_engine_flags_required")
    image = _path(baseline, "manifest", "runtime", "image_digest")
    revision = _path(baseline, "manifest", "model", "immutable_revision")
    if not isinstance(image, str) or not re.fullmatch(
        r"(?:[^\s]+@)?sha256:[0-9a-f]{64}", image
    ):
        eligibility_reasons.append("immutable_image_digest_required")
    if not isinstance(revision, str) or not re.fullmatch(
        r"(?:[0-9a-f]{40}|[0-9a-f]{64})", revision
    ):
        eligibility_reasons.append("immutable_model_revision_required")
    output["decision_eligible"] = (
        statistical_supported
        and baseline_live
        and candidate_live
        and output["attribution"]["state"] == "controlled_difference_declared"
        and _path(baseline, "manifest", "engine", "backend") == "native"
        and _path(candidate, "manifest", "engine", "backend") == "native"
        and baseline_streaming
        and candidate_streaming
        and _path(baseline, "manifest", "engine", "effective_flags_provenance")
        == "runtime_verified"
        and _path(candidate, "manifest", "engine", "effective_flags_provenance")
        == "runtime_verified"
        and isinstance(_path(baseline, "manifest", "runtime", "image_digest"), str)
        and bool(
            re.fullmatch(
                r"(?:[^\s]+@)?sha256:[0-9a-f]{64}",
                _path(baseline, "manifest", "runtime", "image_digest"),
            )
        )
        and isinstance(_path(baseline, "manifest", "model", "immutable_revision"), str)
        and bool(
            re.fullmatch(
                r"(?:[0-9a-f]{40}|[0-9a-f]{64})",
                _path(baseline, "manifest", "model", "immutable_revision"),
            )
        )
    )
    output["decision_ineligible_reasons"] = sorted(set(eligibility_reasons))
    if statistical_supported and not output["decision_eligible"]:
        output["descriptive_statistical_outcome"] = output["overall_outcome"]
        output["overall_outcome"] = None
        output["decision_state"] = "inconclusive"
    if not baseline_live or not candidate_live:
        output["evidence_caveat"] = (
            "synthetic_or_unverified_inputs_cannot_support_a_live_decision"
        )
    return output
