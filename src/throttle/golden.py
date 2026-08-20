"""Eligibility and aggregation for the order-balanced golden live protocol."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from . import __version__
from .benchmark import SCHEMA_VERSION, build_plan
from .compare import _condition_map, _path, _safe_preflight_reason
from .models import RunConfig
from .provenance import (
    CURRENT_MANIFEST_VERSION,
    LEGACY_RUNTIME_CONTROLLED_PATHS,
    PLATFORM_RUNTIME_CONTROLLED_PATHS,
    build_runtime_manifest,
    is_safe_public_metadata,
    runtime_provenance_reasons,
)
from .statistics import relative_delta_percent, t_interval_95

GOLDEN_ARTIFACT_TYPE = "throttle_golden_live_comparison"
GOLDEN_SESSION_ARTIFACT_TYPE = "throttle_golden_session"
RUN_FINGERPRINT_BASIS = "validated_consumed_evidence_projection_v1"
GOLDEN_SEQUENCE = (
    ("B1", "baseline"),
    ("C1", "candidate"),
    ("B2", "baseline"),
    ("C2", "candidate"),
    ("B3", "baseline"),
    ("C3", "candidate"),
)
EXPECTED_VARIANTS = tuple(variant for _, variant in GOLDEN_SEQUENCE)
IMMUTABLE_REVISION = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
CANONICAL_POSITIVE_INTEGER = re.compile(r"^[1-9][0-9]*$")
MAX_GOLDEN_MAX_NUM_SEQS = 2_147_483_647


class GoldenTreatmentError(ValueError):
    """A fixed-code treatment error that never reflects caller input."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _normalized_flag_name(name: str) -> str:
    return name.replace("_", "-").lower()


def _canonical_positive_integer(value: object) -> int | None:
    if (
        type(value) is not str
        or len(value) > 10
        or not CANONICAL_POSITIVE_INTEGER.fullmatch(value)
    ):
        return None
    parsed = int(value)
    if parsed > MAX_GOLDEN_MAX_NUM_SEQS:
        return None
    return parsed


def parse_golden_treatment_flags(
    baseline_flag: tuple[str, str], candidate_flag: tuple[str, str]
) -> tuple[int, int]:
    """Return one sanitized ``max_num_seqs`` pair or a fixed-code error."""

    if (
        type(baseline_flag) is not tuple
        or type(candidate_flag) is not tuple
        or len(baseline_flag) != 2
        or len(candidate_flag) != 2
    ):
        raise GoldenTreatmentError("golden_treatment_flags_malformed")
    baseline_name, baseline_raw = baseline_flag
    candidate_name, candidate_raw = candidate_flag
    if type(baseline_name) is not str or type(candidate_name) is not str:
        raise GoldenTreatmentError(
            "golden_treatment_must_only_change_max_num_seqs"
        )
    if type(baseline_raw) is not str or type(candidate_raw) is not str:
        raise GoldenTreatmentError(
            "golden_max_num_seqs_values_must_be_canonical_positive_integers"
        )
    if (
        _normalized_flag_name(baseline_name) != "max-num-seqs"
        or _normalized_flag_name(candidate_name) != "max-num-seqs"
    ):
        raise GoldenTreatmentError(
            "golden_treatment_must_only_change_max_num_seqs"
        )
    baseline = _canonical_positive_integer(baseline_raw)
    candidate = _canonical_positive_integer(candidate_raw)
    if baseline is None or candidate is None:
        raise GoldenTreatmentError(
            "golden_max_num_seqs_values_must_be_canonical_positive_integers"
        )
    if baseline == candidate:
        raise GoldenTreatmentError(
            "golden_max_num_seqs_values_must_be_distinct"
        )
    return baseline, candidate


def golden_positions(
    baseline_max_num_seqs: int, candidate_max_num_seqs: int
) -> tuple[tuple[str, str, int], ...]:
    """Materialize the fixed six-position order for one validated treatment pair."""

    if (
        type(baseline_max_num_seqs) is not int
        or type(candidate_max_num_seqs) is not int
        or baseline_max_num_seqs <= 0
        or candidate_max_num_seqs <= 0
        or baseline_max_num_seqs > MAX_GOLDEN_MAX_NUM_SEQS
        or candidate_max_num_seqs > MAX_GOLDEN_MAX_NUM_SEQS
    ):
        raise GoldenTreatmentError(
            "golden_max_num_seqs_values_must_be_canonical_positive_integers"
        )
    if baseline_max_num_seqs == candidate_max_num_seqs:
        raise GoldenTreatmentError(
            "golden_max_num_seqs_values_must_be_distinct"
        )
    values = {
        "baseline": baseline_max_num_seqs,
        "candidate": candidate_max_num_seqs,
    }
    return tuple(
        (position, variant, values[variant])
        for position, variant in GOLDEN_SEQUENCE
    )


# Kept as the historical 1-versus-8 sequence for import compatibility. Live
# orchestration and planning materialize positions from the requested pair.
GOLDEN_POSITIONS = golden_positions(1, 8)


def golden_position_config(
    base: RunConfig, *, position: str, variant: str, max_num_seqs: int
) -> RunConfig:
    """Build one immutable position config without changing the endpoint."""

    # Preserve the preflighted client load exactly. A reached client
    # concurrency is demand evidence, not proof of server-scheduler saturation.
    return replace(
        base,
        mode="benchmark",
        variant=variant,
        sequence_position=position,
        engine_flags=base.engine_flags + (("max_num_seqs", str(max_num_seqs)),),
    )


def golden_preflight_reasons(
    base: RunConfig,
    prompts: Sequence[Sequence[Mapping[str, str]]],
    warmup_prompts: Sequence[Sequence[Mapping[str, str]]],
    *,
    baseline_flag: tuple[str, str],
    candidate_flag: tuple[str, str],
) -> list[str]:
    """Return every condition that would make a live golden session ineligible."""

    reasons: list[str] = []
    try:
        baseline_value, candidate_value = parse_golden_treatment_flags(
            baseline_flag, candidate_flag
        )
    except GoldenTreatmentError as exc:
        reasons.append(exc.code)
        baseline_value, candidate_value = 1, 8
    if base.backend != "native":
        reasons.append("golden_requires_native_backend")
    if base.stream is not True:
        reasons.append("golden_requires_streaming")
    if (
        len(base.conditions) != 1
        or base.conditions[0].kind != "closed_loop"
        or not float(base.conditions[0].value).is_integer()
        or base.conditions[0].max_in_flight != int(base.conditions[0].value)
    ):
        reasons.append("golden_requires_one_closed_loop_concurrency")
    elif int(base.conditions[0].value) < max(baseline_value, candidate_value):
        reasons.append("golden_concurrency_must_reach_both_max_num_seqs_values")
    if base.blocks < 3:
        reasons.append("golden_requires_at_least_three_blocks")
    if base.requests_per_block is None:
        reasons.append("golden_orchestrator_requires_count_bounded_blocks")
    elif base.blocks * base.requests_per_block < 200:
        reasons.append("golden_requires_at_least_200_measured_requests_per_position")
    if (
        base.requests_per_block is not None
        and len(base.conditions) == 1
        and base.conditions[0].kind == "closed_loop"
        and float(base.conditions[0].value).is_integer()
        and base.requests_per_block < int(base.conditions[0].value)
    ):
        reasons.append("golden_requests_per_block_must_reach_declared_concurrency")
    if base.warmup_requests_per_condition <= 0:
        reasons.append("golden_requires_separate_warmup_requests")
    if base.limits.max_errors != 1:
        reasons.append("golden_requires_zero_error_tolerance")
    if base.evidence_source != "live_inference":
        reasons.append("golden_requires_live_inference_evidence")
    if base.cache_policy == "unknown":
        reasons.append("golden_requires_explicit_cache_policy")
    if base.engine_flags_provenance != "runtime_verified":
        reasons.append("golden_requires_runtime_verified_engine_flags")
    if base.variant != "unspecified" or base.sequence_position != "unspecified":
        reasons.append("golden_assigns_variant_and_sequence_position_automatically")
    if not IMMUTABLE_REVISION.fullmatch(base.model_revision):
        reasons.append("golden_requires_immutable_model_revision")
    runtime = build_runtime_manifest(base)
    reasons.extend(
        f"golden_runtime_{reason}"
        for reason in runtime_provenance_reasons(runtime, CURRENT_MANIFEST_VERSION)
    )
    if base.server_version == "unknown":
        reasons.append("golden_requires_server_version")

    if any(
        _normalized_flag_name(name) == "max-num-seqs"
        for name, _ in base.engine_flags
    ):
        reasons.append("max_num_seqs_must_use_baseline_and_candidate_config_flags")

    if base.cost.kind == "user_supplied":
        reasons.append("golden_user_supplied_run_total_is_ambiguous_across_six_positions")
    if (
        base.cost.kind == "serverless_active_seconds"
        and base.cost.billed_active_seconds is not None
    ):
        reasons.append(
            "golden_serverless_billed_active_seconds_must_be_added_after_the_session"
        )

    estimated_cost = base.cost.estimated_upper_bound(
        base.limits.max_elapsed_seconds
    )
    if estimated_cost is None and not base.allow_unknown_cost:
        reasons.append("golden_requires_unknown_cost_acknowledgement")
    elif (
        estimated_cost is not None
        and estimated_cost > base.limits.max_estimated_spend
    ):
        reasons.append("golden_session_estimated_cost_exceeds_spend_limit")

    position_plan = build_plan(
        golden_position_config(
            base,
            position="B1",
            variant="baseline",
            max_num_seqs=baseline_value,
        ),
        tuple(tuple(dict(message) for message in prompt) for prompt in prompts),
        tuple(tuple(dict(message) for message in prompt) for prompt in warmup_prompts),
    )
    if not position_plan["workload"]["separate_warmup_workload"]:
        reasons.append("golden_requires_separate_warmup_workload")
    if not position_plan["workload"]["warmup_prompts_disjoint"]:
        reasons.append("golden_requires_disjoint_warmup_prompts")
    position_requests = position_plan["request_count"]["exact"]
    if position_requests is None:
        reasons.append("golden_session_request_count_must_be_exact")
    else:
        session_requests = int(position_requests) * len(GOLDEN_SEQUENCE)
        if session_requests > base.limits.max_requests:
            reasons.append("golden_session_requests_exceed_max_requests")
        session_tokens = session_requests * base.max_tokens
        if session_tokens > base.limits.max_total_requested_tokens:
            reasons.append("golden_session_tokens_exceed_total_token_limit")
    return sorted(set(reasons))


def build_golden_plan(
    base: RunConfig,
    prompts: Sequence[Sequence[Mapping[str, str]]],
    warmup_prompts: Sequence[Sequence[Mapping[str, str]]],
    *,
    baseline_flag: tuple[str, str],
    candidate_flag: tuple[str, str],
) -> dict[str, Any]:
    """Build a zero-traffic plan for all six positions."""

    baseline_value, candidate_value = parse_golden_treatment_flags(
        baseline_flag, candidate_flag
    )
    positions = golden_positions(baseline_value, candidate_value)
    position = golden_position_config(
        base,
        position="B1",
        variant="baseline",
        max_num_seqs=baseline_value,
    )
    position_plan = build_plan(
        position,
        tuple(tuple(dict(message) for message in prompt) for prompt in prompts),
        tuple(tuple(dict(message) for message in prompt) for prompt in warmup_prompts),
    )
    per_position_requests = position_plan["request_count"]["exact"]
    session_requests = (
        int(per_position_requests) * len(positions)
        if per_position_requests is not None
        else None
    )
    return {
        "traffic_sent": False,
        "treatment": {
            "field": "max_num_seqs",
            "baseline_value": baseline_value,
            "candidate_value": candidate_value,
            "closed_loop_concurrency": (
                int(base.conditions[0].value)
                if len(base.conditions) == 1
                and base.conditions[0].kind == "closed_loop"
                and float(base.conditions[0].value).is_integer()
                else None
            ),
        },
        "positions": [
            {
                "position": position_name,
                "variant": variant,
                "max_num_seqs": max_num_seqs,
            }
            for position_name, variant, max_num_seqs in positions
        ],
        "per_position_requests": per_position_requests,
        "session_requests": session_requests,
        "per_position_requested_output_tokens": (
            int(per_position_requests) * base.max_tokens
            if per_position_requests is not None
            else None
        ),
        "session_requested_output_tokens": (
            int(session_requests) * base.max_tokens
            if session_requests is not None
            else None
        ),
        "session_duration_limit_seconds": base.limits.max_elapsed_seconds,
        "session_estimated_cost_upper_bound": base.cost.estimated_upper_bound(
            base.limits.max_elapsed_seconds
        ),
        "session_max_estimated_spend": base.limits.max_estimated_spend,
        "spend_limit_enforceable": base.cost.elapsed_estimate(0.0) is not None,
        "measurement": {
            "blocks_per_position": base.blocks,
            "requests_per_block": base.requests_per_block,
            "warmup_requests_per_position": base.warmup_requests_per_condition,
            "max_tokens_per_request": base.max_tokens,
            "request_timeout_seconds": base.request_timeout_seconds,
        },
        "limits": base.limits.public_dict(),
        "destination": position_plan["destination"],
        "privacy": position_plan["privacy"],
        "preflight_reasons": golden_preflight_reasons(
            base,
            prompts,
            warmup_prompts,
            baseline_flag=baseline_flag,
            candidate_flag=candidate_flag,
        ),
    }


_FINGERPRINT_ROOT_PATHS: tuple[tuple[str, ...], ...] = (
    ("schema_version",),
    ("artifact_type",),
    ("started_at",),
    ("completed_at",),
    ("mode",),
    ("status",),
    ("stop_reason",),
    ("manifest", "manifest_version"),
    ("manifest", "tool", "name"),
    ("manifest", "tool", "version"),
    ("manifest", "model", "id"),
    ("manifest", "model", "immutable_revision"),
    ("manifest", "workload", "seed"),
    ("manifest", "workload", "measured_sha256"),
    ("manifest", "workload", "warmup_sha256"),
    ("manifest", "workload", "warmup_is_separate"),
    ("manifest", "workload", "warmup_prompts_disjoint"),
    ("manifest", "workload", "cache_policy"),
    ("manifest", "request", "type"),
    ("manifest", "request", "temperature"),
    ("manifest", "request", "max_tokens"),
    ("manifest", "request", "stop"),
    ("manifest", "request", "stream"),
    ("manifest", "request", "timeout_seconds"),
    # A sealed request profile is covered by its own canonical hash, so the
    # fingerprint carries the version and the seal rather than every field.
    # Legacy fixed-request reports project both as null.
    ("manifest", "request", "profile_version"),
    ("manifest", "request", "profile_sha256"),
    ("manifest", "traffic", "blocks"),
    ("manifest", "traffic", "requests_per_block"),
    ("manifest", "traffic", "block_duration_seconds"),
    ("manifest", "traffic", "warmup_requests_per_condition"),
    ("manifest", "traffic", "p95_slo_ms"),
    ("manifest", "traffic", "ttft_slo_ms"),
    ("manifest", "traffic", "open_loop_rate_relative_tolerance"),
    (
        "manifest",
        "traffic",
        "open_loop_scheduler_lag_interval_tolerance",
    ),
    ("manifest", "engine", "backend"),
    ("manifest", "engine", "backend_version"),
    ("manifest", "engine", "http_client_version"),
    ("manifest", "engine", "server_version"),
    ("manifest", "engine", "effective_flags_provenance"),
    ("manifest", "provenance", "evidence_source"),
    ("manifest", "provenance", "variant"),
    ("manifest", "provenance", "sequence_position"),
    ("manifest", "safety", "ambient_proxy_environment_used"),
    ("manifest", "safety", "redirects_followed"),
)

_FINGERPRINT_TRAFFIC_CONDITION_FIELDS = (
    "id",
    "kind",
    "value",
    "max_in_flight",
)
_FINGERPRINT_SAFETY_LIMIT_FIELDS = (
    "max_requests",
    "max_tokens_per_request",
    "max_total_requested_tokens",
    "max_errors",
    "max_elapsed_seconds",
    "max_estimated_spend",
    "max_concurrency",
    "max_response_bytes",
)
_FINGERPRINT_SAFETY_OVERRIDE_FIELDS = (
    "insecure_http",
    "unknown_cost_acknowledged",
    "guidellm_validation_gaps_acknowledged",
)
_FINGERPRINT_COST_FIELDS = (
    "kind",
    "currency",
    "accounting_basis",
    "pre_run_upper_bound",
    "total_hourly_rate",
    "gpu_count",
    "active_second_rate",
    "max_active_workers",
    "billed_active_seconds",
    "user_supplied_total",
)
_FINGERPRINT_METRIC_DEFINITION_FIELDS = (
    "e2e_latency",
    "ttft",
    "tpot",
    "itl",
    "inter_chunk_latency",
    "throughput",
    "decision_throughput",
    "slo_goodput",
    "slo_decision_ci",
)
_FINGERPRINT_COUNT_FIELDS = ("attempted", "valid", "invalid")
_FINGERPRINT_COUNT_MAP_FIELDS = (
    "status_counts",
    "error_counts",
    "finish_reason_counts",
)
_FINGERPRINT_CONDITION_FIELDS = (
    "valid",
    "decision_grade",
    "measured_wall_seconds",
    "observed_peak_in_flight",
)
_FINGERPRINT_OPEN_LOOP_CONDITION_FIELDS = (
    "target_offered_request_rate",
    "achieved_offered_request_rate",
    "offered_rate_relative_error",
    "open_loop_target_achieved",
)
_FINGERPRINT_BLOCK_FIELDS = (
    "block_index",
    "valid",
    "wall_duration_seconds",
    "observed_peak_in_flight",
    "offered_requests",
)
_FINGERPRINT_OPEN_LOOP_BLOCK_FIELDS = (
    "target_offered_request_rate",
    "launch_window_seconds",
    "achieved_offered_request_rate",
    "offered_rate_relative_error",
    "scheduler_lag_interval_ratio_p95",
    "open_loop_target_achieved",
)
_FINGERPRINT_AGGREGATE_METRIC_FIELDS = (
    "valid_response_count",
    "completion_tokens",
    "requests_per_second",
    "output_tokens_per_second",
    "block_mean_output_tokens_per_second",
    "block_mean_requests_per_second",
    "error_rate",
)
_FINGERPRINT_BLOCK_METRIC_FIELDS = (
    "valid_response_count",
    "completion_tokens",
    "requests_per_second",
    "output_tokens_per_second",
    "error_rate",
)
_FINGERPRINT_INTERVAL_FIELDS = (
    "low",
    "high",
    "confidence",
    "method",
    "n",
)
_FINGERPRINT_SCHEDULER_FIELDS = (
    "count",
    "mean",
    "p50",
    "p90",
    "p95",
    "p99",
)
_FINGERPRINT_RUN_TOTAL_FIELDS = (
    "requests_started",
    "requests_completed",
    "requests_cancelled",
    "requests_in_flight",
    "peak_in_flight",
    "errors",
    "reserved_output_tokens",
    "elapsed_seconds",
)
_FINGERPRINT_COST_SUMMARY_FIELDS = (
    "kind",
    "total_cost",
    "completion_tokens",
    "cost_per_million_output_tokens",
)
_JSON_SCALAR_TYPES = {type(None), bool, int, float, str}


def _fingerprint_scalar(value: object) -> object:
    if type(value) not in _JSON_SCALAR_TYPES:
        raise ValueError("fingerprint evidence must be scalar")
    return value


def _fingerprint_fields(
    value: object, fields: Sequence[str]
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("fingerprint evidence must be a mapping")
    return {
        field: _fingerprint_scalar(value.get(field))
        for field in fields
    }


def _fingerprint_count_map(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("fingerprint count evidence must be a mapping")
    projected = _fingerprint_fields(value, _FINGERPRINT_COUNT_FIELDS)
    for field in _FINGERPRINT_COUNT_MAP_FIELDS:
        count_map = value.get(field)
        if not isinstance(count_map, Mapping):
            raise ValueError("fingerprint count evidence must be a mapping")
        projected[field] = {
            str(name): _fingerprint_scalar(count)
            for name, count in sorted(count_map.items())
        }
    return projected


def _fingerprint_interval(value: object) -> dict[str, object]:
    return _fingerprint_fields(value, _FINGERPRINT_INTERVAL_FIELDS)


def _fingerprint_aggregate_metrics(value: object) -> dict[str, object]:
    projected = _fingerprint_fields(value, _FINGERPRINT_AGGREGATE_METRIC_FIELDS)
    assert isinstance(value, Mapping)
    for field in (
        "block_mean_output_tokens_per_second_ci",
        "block_mean_requests_per_second_ci",
    ):
        interval = value.get(field)
        projected[field] = (
            _fingerprint_interval(interval) if isinstance(interval, Mapping) else None
        )
    for field in ("e2e_latency_ms", "ttft_ms"):
        distribution = value.get(field)
        if not isinstance(distribution, Mapping):
            raise ValueError("fingerprint latency evidence must be a mapping")
        projected[field] = {
            "p95_repeated_block_ci": _fingerprint_interval(
                distribution.get("p95_repeated_block_ci")
            )
        }
    return projected


def _fingerprint_block_metrics(value: object) -> dict[str, object]:
    projected = _fingerprint_fields(value, _FINGERPRINT_BLOCK_METRIC_FIELDS)
    assert isinstance(value, Mapping)
    for field in ("e2e_latency_ms", "ttft_ms"):
        distribution = value.get(field)
        if not isinstance(distribution, Mapping):
            raise ValueError("fingerprint latency evidence must be a mapping")
        projected[field] = {
            "p95": _fingerprint_scalar(distribution.get("p95"))
        }
    return projected


def _fingerprint_condition(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("fingerprint condition evidence must be a mapping")
    projected = _fingerprint_fields(value, _FINGERPRINT_CONDITION_FIELDS)
    condition_projection = _fingerprint_fields(
        value.get("condition"), _FINGERPRINT_TRAFFIC_CONDITION_FIELDS
    )
    projected["condition"] = condition_projection
    open_loop = condition_projection["kind"] == "open_loop"
    if open_loop:
        projected.update(
            _fingerprint_fields(value, _FINGERPRINT_OPEN_LOOP_CONDITION_FIELDS)
        )
    projected["warmup"] = _fingerprint_count_map(value.get("warmup"))
    projected["request_counts"] = _fingerprint_count_map(
        value.get("request_counts")
    )
    projected["metrics"] = _fingerprint_aggregate_metrics(value.get("metrics"))
    blocks = value.get("blocks")
    if not isinstance(blocks, list):
        raise ValueError("fingerprint block evidence must be a list")
    projected_blocks: list[dict[str, object]] = []
    for block in blocks:
        block_projection = _fingerprint_fields(block, _FINGERPRINT_BLOCK_FIELDS)
        assert isinstance(block, Mapping)
        if open_loop:
            block_projection.update(
                _fingerprint_fields(block, _FINGERPRINT_OPEN_LOOP_BLOCK_FIELDS)
            )
        block_projection["request_counts"] = _fingerprint_count_map(
            block.get("request_counts")
        )
        if open_loop:
            block_projection["scheduler_lag_ms"] = _fingerprint_fields(
                block.get("scheduler_lag_ms"), _FINGERPRINT_SCHEDULER_FIELDS
            )
        block_projection["metrics"] = _fingerprint_block_metrics(
            block.get("metrics")
        )
        projected_blocks.append(block_projection)
    projected["blocks"] = projected_blocks
    return projected


def _fingerprint_projection(report: Mapping[str, Any]) -> dict[str, object]:
    """Return only evidence consumed by report and Golden validation.

    Ignored extension fields are intentionally absent so the public digest
    cannot be used as an oracle for attacker-controlled payloads.
    """

    projection: dict[str, object] = {
        ".".join(path): _fingerprint_scalar(_path(report, *path))
        for path in _FINGERPRINT_ROOT_PATHS
    }
    projection["fingerprint_basis"] = RUN_FINGERPRINT_BASIS
    manifest = _path(report, "manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("fingerprint manifest evidence must be a mapping")
    runtime = manifest.get("runtime")
    runtime_paths = (
        PLATFORM_RUNTIME_CONTROLLED_PATHS
        if manifest.get("manifest_version") == CURRENT_MANIFEST_VERSION
        else LEGACY_RUNTIME_CONTROLLED_PATHS
    )
    projection["manifest.runtime"] = _fingerprint_fields(
        runtime, tuple(path[-1] for path in runtime_paths)
    )
    projection["manifest.metric_definitions"] = _fingerprint_fields(
        manifest.get("metric_definitions"),
        _FINGERPRINT_METRIC_DEFINITION_FIELDS,
    )
    engine_flags = _flag_map(report)
    if engine_flags is None:
        raise ValueError("fingerprint engine flags are invalid")
    projection["manifest.engine.effective_flags"] = dict(
        sorted(engine_flags.items())
    )
    traffic = manifest.get("traffic")
    if not isinstance(traffic, Mapping):
        raise ValueError("fingerprint traffic evidence must be a mapping")
    conditions = traffic.get("conditions")
    if not isinstance(conditions, list):
        raise ValueError("fingerprint traffic conditions must be a list")
    projection["manifest.traffic.conditions"] = [
        _fingerprint_fields(item, _FINGERPRINT_TRAFFIC_CONDITION_FIELDS)
        for item in conditions
    ]
    safety = manifest.get("safety")
    if not isinstance(safety, Mapping):
        raise ValueError("fingerprint safety evidence must be a mapping")
    projection["manifest.safety.limits"] = _fingerprint_fields(
        safety.get("limits"), _FINGERPRINT_SAFETY_LIMIT_FIELDS
    )
    projection["manifest.safety.overrides"] = _fingerprint_fields(
        safety.get("overrides"), _FINGERPRINT_SAFETY_OVERRIDE_FIELDS
    )
    projection["manifest.cost"] = _fingerprint_fields(
        manifest.get("cost"), _FINGERPRINT_COST_FIELDS
    )
    run_conditions = report.get("conditions")
    if not isinstance(run_conditions, list):
        raise ValueError("fingerprint conditions must be a list")
    projection["conditions"] = [
        _fingerprint_condition(condition) for condition in run_conditions
    ]
    projection["run_totals"] = _fingerprint_fields(
        report.get("run_totals"), _FINGERPRINT_RUN_TOTAL_FIELDS
    )
    projection["cost_summary"] = _fingerprint_fields(
        report.get("cost_summary"), _FINGERPRINT_COST_SUMMARY_FIELDS
    )
    return projection


def _safe_report_fingerprint(report: Mapping[str, Any]) -> str | None:
    if _safe_preflight_reason(report) is not None:
        return None
    try:
        encoded = json.dumps(
            _fingerprint_projection(report),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError):
        return None
    return hashlib.sha256(encoded).hexdigest()


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _flag_map(report: Mapping[str, Any]) -> dict[str, str] | None:
    raw = _path(report, "manifest", "engine", "effective_flags")
    if not isinstance(raw, Mapping):
        return None
    output: dict[str, str] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not is_safe_public_metadata(
            name, max_length=128
        ):
            return None
        normalized = name.replace("_", "-").lower()
        if not re.fullmatch(
            r"[a-z0-9-]{1,128}", normalized
        ) or not is_safe_public_metadata(value):
            return None
        assert isinstance(value, str)
        if normalized in output:
            return None
        output[normalized] = value
    return output


def _has_normalized_flag_alias_collision(report: Mapping[str, Any]) -> bool:
    # This diagnostic may run alongside a structural preflight failure. Only
    # traverse exact JSON-object dictionaries so a forged Mapping cannot
    # execute custom iteration before the fixed-code early return.
    if type(report) is not dict:
        return False
    manifest = report.get("manifest")
    if type(manifest) is not dict:
        return False
    engine = manifest.get("engine")
    if type(engine) is not dict:
        return False
    raw = engine.get("effective_flags")
    if type(raw) is not dict:
        return False
    seen: set[str] = set()
    for name in raw:
        if type(name) is not str:
            continue
        normalized = _normalized_flag_name(name)
        if normalized in seen:
            return True
        seen.add(normalized)
    return False


def _present(report: Mapping[str, Any], *path: str) -> bool:
    current: Any = report
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return False
        current = current[part]
    return True


def _all_equal_present(reports: Sequence[Mapping[str, Any]], *path: str) -> bool:
    if not reports or any(not _present(report, *path) for report in reports):
        return False
    first = _path(reports[0], *path)
    return all(_path(report, *path) == first for report in reports[1:])


def _report_closed_loop_concurrency(report: Mapping[str, Any]) -> int | None:
    """Return one canonical declared client load without trusting its ID alone."""

    try:
        mapped = _condition_map(report)
    except Exception:
        return None
    if len(mapped) != 1:
        return None
    identifier, condition = next(iter(mapped.items()))
    descriptor = condition.get("condition")
    if not isinstance(descriptor, Mapping):
        return None
    value = descriptor.get("value")
    max_in_flight = descriptor.get("max_in_flight")
    if (
        type(value) is not int
        or type(max_in_flight) is not int
        or value <= 0
        or value > MAX_GOLDEN_MAX_NUM_SEQS
        or max_in_flight != value
        or descriptor.get("kind") != "closed_loop"
        or descriptor.get("id") != f"closed_loop:{value}"
        or identifier != f"closed_loop:{value}"
    ):
        return None
    return value


def _common_closed_loop_concurrency(
    reports: Sequence[Mapping[str, Any]],
) -> int | None:
    values = tuple(_report_closed_loop_concurrency(report) for report in reports)
    if not values or values[0] is None or any(value != values[0] for value in values):
        return None
    return values[0]


def _every_measured_block_attempts_declared_concurrency(
    reports: Sequence[Mapping[str, Any]], concurrency: int
) -> bool:
    """Prove every measured block had enough requests to reach client load N."""

    for report in reports:
        try:
            conditions = _condition_map(report)
        except Exception:
            return False
        if len(conditions) != 1:
            return False
        condition = next(iter(conditions.values()))
        blocks = condition.get("blocks")
        if not isinstance(blocks, list) or not blocks:
            return False
        for block in blocks:
            if not isinstance(block, Mapping):
                return False
            attempted = _path(block, "request_counts", "attempted")
            if type(attempted) is not int or attempted < concurrency:
                return False
    return True


def _infer_golden_treatment(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, int | str] | None:
    """Infer a sanitized treatment only from the six saved evidence artifacts."""

    if len(reports) != 6:
        return None
    if tuple(
        _path(report, "manifest", "provenance", "variant")
        for report in reports
    ) != EXPECTED_VARIANTS or tuple(
        _path(report, "manifest", "provenance", "sequence_position")
        for report in reports
    ) != tuple(position for position, _ in GOLDEN_SEQUENCE):
        return None
    if any(
        _path(report, "manifest", "engine", "effective_flags_provenance")
        != "runtime_verified"
        for report in reports
    ):
        return None
    flags = tuple(_flag_map(report) for report in reports)
    if any(flag is None for flag in flags):
        return None
    assert all(flag is not None for flag in flags)
    typed_flags = tuple(flag for flag in flags if flag is not None)
    baseline_flags = tuple(typed_flags[index] for index in (0, 2, 4))
    candidate_flags = tuple(typed_flags[index] for index in (1, 3, 5))
    if (
        any(flag != baseline_flags[0] for flag in baseline_flags[1:])
        or any(flag != candidate_flags[0] for flag in candidate_flags[1:])
    ):
        return None
    changed = {
        name
        for name in set(baseline_flags[0]) | set(candidate_flags[0])
        if baseline_flags[0].get(name) != candidate_flags[0].get(name)
    }
    if changed != {"max-num-seqs"}:
        return None
    baseline = _canonical_positive_integer(
        baseline_flags[0].get("max-num-seqs")
    )
    candidate = _canonical_positive_integer(
        candidate_flags[0].get("max-num-seqs")
    )
    concurrency = _common_closed_loop_concurrency(reports)
    if (
        baseline is None
        or candidate is None
        or baseline == candidate
        or concurrency is None
    ):
        return None
    return {
        "field": "max_num_seqs",
        "baseline_value": baseline,
        "candidate_value": candidate,
        "closed_loop_concurrency": concurrency,
    }


def _protocol_checks(reports: Sequence[Mapping[str, Any]]) -> list[str]:
    reasons: list[str] = []
    if len(reports) != 6:
        return ["golden_sequence_requires_exactly_six_runs"]
    for index, report in enumerate(reports):
        reason = _safe_preflight_reason(report)
        if reason:
            reasons.append(f"run_{index + 1}_{reason}")
    if reasons:
        return sorted(set(reasons))
    if any(_has_normalized_flag_alias_collision(report) for report in reports):
        reasons.append("engine_flags_malformed")
    if reasons:
        return sorted(set(reasons))
    variants = tuple(
        _path(report, "manifest", "provenance", "variant") for report in reports
    )
    if variants != EXPECTED_VARIANTS:
        reasons.append(
            "sequence_must_be_baseline_candidate_baseline_then_candidate_baseline_candidate"
        )
    positions = tuple(
        _path(report, "manifest", "provenance", "sequence_position")
        for report in reports
    )
    if positions != tuple(position for position, _ in GOLDEN_SEQUENCE):
        reasons.append("sequence_position_labels_must_be_B1_C1_B2_C2_B3_C3")
    starts = [_timestamp(report.get("started_at")) for report in reports]
    ends = [_timestamp(report.get("completed_at")) for report in reports]
    if any(value is None for value in (*starts, *ends)):
        reasons.append("missing_or_invalid_run_timestamps")
    else:
        if any(starts[index] > ends[index] for index in range(6)):  # type: ignore[operator]
            reasons.append("run_starts_after_its_completion")
        if any(starts[index] < ends[index - 1] for index in range(1, 6)):  # type: ignore[operator]
            reasons.append("runs_overlap_or_are_out_of_order")

    manifest_version = _path(reports[0], "manifest", "manifest_version")
    runtime_paths = (
        PLATFORM_RUNTIME_CONTROLLED_PATHS
        if manifest_version == CURRENT_MANIFEST_VERSION
        else LEGACY_RUNTIME_CONTROLLED_PATHS
    )
    required_common = (
        ("manifest", "manifest_version"),
        ("manifest", "tool"),
        ("manifest", "model", "id"),
        ("manifest", "model", "immutable_revision"),
        *(tuple(("manifest", *path) for path in runtime_paths)),
        ("manifest", "engine", "backend"),
        ("manifest", "engine", "backend_version"),
        ("manifest", "engine", "http_client_version"),
        ("manifest", "engine", "server_version"),
        ("manifest", "workload", "seed"),
        ("manifest", "workload", "measured_sha256"),
        ("manifest", "workload", "warmup_sha256"),
        ("manifest", "workload", "cache_policy"),
        ("manifest", "request"),
        ("manifest", "traffic", "conditions"),
        ("manifest", "traffic", "blocks"),
        ("manifest", "traffic", "requests_per_block"),
        ("manifest", "traffic", "block_duration_seconds"),
        ("manifest", "traffic", "warmup_requests_per_condition"),
        ("manifest", "traffic", "p95_slo_ms"),
        ("manifest", "traffic", "ttft_slo_ms"),
        ("manifest", "metric_definitions"),
        ("manifest", "safety"),
        ("manifest", "cost"),
    )
    for path in required_common:
        if not _all_equal_present(reports, *path):
            reasons.append("uncontrolled_or_missing_" + "_".join(path[1:]))
    revision = _path(reports[0], "manifest", "model", "immutable_revision")
    if not isinstance(revision, str) or not IMMUTABLE_REVISION.fullmatch(revision):
        reasons.append("model_revision_is_not_immutable")
    for path, code in (
        (("manifest", "engine", "server_version"), "server_version_missing"),
    ):
        value = _path(reports[0], *path)
        if not isinstance(value, str) or value == "unknown":
            reasons.append(code)
    if _path(reports[0], "manifest", "engine", "backend") != "native":
        reasons.append("golden_protocol_requires_strict_native_completion_validation")
    if any(
        _path(report, "manifest", "engine", "effective_flags_provenance")
        != "runtime_verified"
        for report in reports
    ):
        reasons.append("effective_engine_flags_not_runtime_verified")
    if any(
        _path(report, "manifest", "provenance", "evidence_source") != "live_inference"
        for report in reports
    ):
        reasons.append("all_six_runs_must_be_live_inference")
    if _path(reports[0], "manifest", "workload", "cache_policy") == "unknown":
        reasons.append("cache_policy_must_be_explicit")
    if _path(reports[0], "manifest", "request", "stream") is not True:
        reasons.append("golden_protocol_requires_streaming")
    warmup_count = _path(
        reports[0], "manifest", "traffic", "warmup_requests_per_condition"
    )
    if (
        not isinstance(warmup_count, int)
        or isinstance(warmup_count, bool)
        or warmup_count <= 0
    ):
        reasons.append("golden_protocol_requires_separate_warmup_requests")
    if any(
        _path(report, "manifest", "workload", "warmup_is_separate") is not True
        for report in reports
    ):
        reasons.append("warmup_workload_not_proven_separate")
    if any(
        _path(report, "manifest", "workload", "warmup_prompts_disjoint") is not True
        for report in reports
    ):
        reasons.append("warmup_prompts_not_disjoint_from_measured_workload")
    try:
        condition_sets = [set(_condition_map(report)) for report in reports]
    except Exception:
        reasons.append("condition_maps_malformed")
    else:
        if not condition_sets or any(
            items != condition_sets[0] for items in condition_sets[1:]
        ):
            reasons.append("condition_sets_do_not_match")
        elif _common_closed_loop_concurrency(reports) is None:
            reasons.append(
                "golden_treatment_requires_one_common_closed_loop_concurrency"
            )

    flags = [_flag_map(report) for report in reports]
    if any(flag is None for flag in flags):
        reasons.append("engine_flags_malformed")
        return sorted(set(reasons))
    baseline_flags = [flags[index] for index in (0, 2, 4)]
    candidate_flags = [flags[index] for index in (1, 3, 5)]
    if not all(flag == baseline_flags[0] for flag in baseline_flags[1:]):
        reasons.append("baseline_effective_flags_changed_between_repeats")
    if not all(flag == candidate_flags[0] for flag in candidate_flags[1:]):
        reasons.append("candidate_effective_flags_changed_between_repeats")
    changed = {
        name
        for name in set(baseline_flags[0]) | set(candidate_flags[0])
        if baseline_flags[0].get(name) != candidate_flags[0].get(name)
    }
    if changed != {"max-num-seqs"}:
        reasons.append("golden_treatment_must_only_change_max_num_seqs")
    baseline_limit = _canonical_positive_integer(
        baseline_flags[0].get("max-num-seqs")
    )
    candidate_limit = _canonical_positive_integer(
        candidate_flags[0].get("max-num-seqs")
    )
    if baseline_limit is None or candidate_limit is None:
        reasons.append(
            "golden_treatment_requires_canonical_positive_max_num_seqs_values"
        )
    elif baseline_limit == candidate_limit:
        reasons.append("golden_treatment_requires_distinct_max_num_seqs_values")

    concurrency = _common_closed_loop_concurrency(reports)
    if concurrency is None:
        reasons.append(
            "golden_treatment_requires_one_common_closed_loop_concurrency"
        )
    elif baseline_limit is not None and candidate_limit is not None:
        requests_per_block = _path(
            reports[0], "manifest", "traffic", "requests_per_block"
        )
        if type(requests_per_block) is int and requests_per_block < concurrency:
            reasons.append(
                "golden_requests_per_block_must_reach_declared_concurrency"
            )
        if concurrency < max(baseline_limit, candidate_limit):
            reasons.append(
                "golden_concurrency_must_reach_both_max_num_seqs_values"
            )
        if not _every_measured_block_attempts_declared_concurrency(
            reports, concurrency
        ):
            reasons.append(
                "golden_measured_blocks_must_attempt_declared_concurrency"
            )
        # Reaching N is evidence of sufficient offered client demand. It does
        # not prove that the server scheduler itself saturated at either
        # max_num_seqs value.
        actually_exercised = all(
            any(
                isinstance(condition, Mapping)
                and type(condition.get("observed_peak_in_flight")) is int
                and condition.get("observed_peak_in_flight") >= concurrency
                for condition in report.get("conditions", [])
            )
            for report in reports
        )
        if not actually_exercised:
            reasons.append("traffic_does_not_reach_declared_closed_loop_concurrency")
    # Chunked prefill may be present, but must be common and receives no credit.
    chunked = [flag.get("enable-chunked-prefill") for flag in flags if flag is not None]
    if any(value != chunked[0] for value in chunked[1:]):
        reasons.append("chunked_prefill_must_not_be_the_treatment")
    return sorted(set(reasons))


def _supported_decision_summary(
    outcome: str,
    condition: Mapping[str, Any],
    reports: Sequence[Mapping[str, Any]],
    treatment: Mapping[str, int | str],
) -> dict[str, Any]:
    interval = condition["throughput_delta_percent_ci"]
    estimate = float(interval["estimate"])
    low = float(interval["low"])
    high = float(interval["high"])
    p95_slo = _path(reports[0], "manifest", "traffic", "p95_slo_ms")
    ttft_slo = _path(reports[0], "manifest", "traffic", "ttft_slo_ms")
    declared_slos = [
        name
        for name, value in (("E2E", p95_slo), ("TTFT", ttft_slo))
        if value is not None
    ]
    if len(declared_slos) == 2:
        slo_statement = "all declared E2E and TTFT SLO gates passed"
    elif declared_slos:
        slo_statement = f"the declared {declared_slos[0]} SLO gate passed"
    else:
        slo_statement = "no latency SLO was declared"

    if outcome == "candidate_higher_throughput":
        winner = "candidate"
        winner_value = int(treatment["candidate_value"])
        comparison = (
            f"candidate throughput was {estimate:.1f}% higher than baseline"
        )
    else:
        winner = "baseline"
        winner_value = int(treatment["baseline_value"])
        comparison = (
            f"candidate throughput was {abs(estimate):.1f}% lower than baseline"
        )
    text = (
        "Golden recommendation — tested workload only: "
        f"{winner} max_num_seqs={winner_value} won; {comparison} "
        f"(order-balanced 95% CI {low:.1f}% to {high:.1f}%, excludes zero); "
        f"{slo_statement}."
    )
    return {
        "label": "golden_recommendation_tested_workload_only",
        "winner": winner,
        "winner_config": {"max_num_seqs": winner_value},
        "candidate_throughput_delta_percent": estimate,
        "throughput_delta_percent_ci": {
            "low": low,
            "high": high,
            "confidence": 0.95,
            "method": interval.get("method"),
        },
        "ci_excludes_zero": True,
        "declared_slo_gates_passed": declared_slos,
        "text": text,
    }


def validate_golden_sequence(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate and compare six B-C-B / C-B-C sequential live reports."""

    reasons = _protocol_checks(reports)
    treatment = None if reasons else _infer_golden_treatment(reports)
    output: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": GOLDEN_ARTIFACT_TYPE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tool_version": __version__,
        "status": "ineligible" if reasons else "complete",
        "golden_protocol_eligible": not reasons,
        "decision_eligible": False,
        "decision_state": "inconclusive",
        "eligibility_reasons": reasons,
        "run_fingerprint_basis": RUN_FINGERPRINT_BASIS,
        "run_fingerprints": [_safe_report_fingerprint(report) for report in reports],
        "sequence": list(EXPECTED_VARIANTS),
        "treatment": treatment,
        "conditions": [],
        "overall_outcome": None,
        "decision_summary": None,
        "optimization_credit": {
            "changed_flag": "max_num_seqs" if not reasons else None,
            "chunked_prefill_credited": False,
        },
        "verification_scope": (
            "Throttle validates internal report consistency, ordering, pins, and declared "
            "runtime-verified provenance. It cannot independently prove operator-supplied "
            "hardware or runtime attestations; retain external audit evidence."
        ),
        "client_demand_scope": (
            "Reaching the declared client concurrency proves sufficient offered demand "
            "for this test; it does not prove direct server-scheduler saturation."
        ),
        "disclaimer": "This order-balanced live result applies only to the pinned manifest and tested workload; it is not a savings projection or universal causal claim.",
    }
    if reasons:
        return output
    if treatment is None:  # defensive: protocol eligibility must imply inference
        output["status"] = "ineligible"
        output["golden_protocol_eligible"] = False
        output["eligibility_reasons"] = ["golden_treatment_could_not_be_inferred"]
        output["optimization_credit"]["changed_flag"] = None
        return output
    condition_maps = [_condition_map(report) for report in reports]
    identifiers = list(condition_maps[0])
    outcomes: set[str] = set()
    all_supported = True
    for identifier in identifiers:
        rates = [
            float(
                _path(
                    condition_maps[index][identifier],
                    "metrics",
                    "block_mean_output_tokens_per_second",
                )
            )
            for index in range(6)
        ]
        # Preserve the intended order balance instead of manufacturing three
        # B-before-C pairs. Phase one contrasts C1 with its B1/B2 brackets;
        # phase two contrasts the C2/C3 brackets with B3.
        phase_one_baseline = (rates[0] + rates[2]) / 2.0
        phase_two_candidate = (rates[3] + rates[5]) / 2.0
        contrasts = [
            relative_delta_percent(rates[1], phase_one_baseline),
            relative_delta_percent(phase_two_candidate, rates[4]),
        ]
        if any(value is None for value in contrasts):
            interval = {
                "estimate": None,
                "low": None,
                "high": None,
                "confidence": 0.95,
                "method": "order_balanced_phase_contrasts",
                "n": 0,
            }
        else:
            interval = t_interval_95([float(value) for value in contrasts])
            interval["estimate"] = sum(float(value) for value in contrasts) / 2.0
            interval["method"] = "order_balanced_phase_contrasts"
        position_tokens = [
            int(
                _path(condition_maps[index][identifier], "metrics", "completion_tokens")
            )
            for index in range(6)
        ]
        baseline_tokens = sum(position_tokens[index] for index in (0, 2, 4))
        candidate_tokens = sum(position_tokens[index] for index in (1, 3, 5))
        token_difference = abs(candidate_tokens - baseline_tokens) / max(
            candidate_tokens, baseline_tokens
        )
        position_token_spread = (
            (max(position_tokens) - min(position_tokens)) / max(position_tokens)
            if position_tokens and max(position_tokens) > 0
            else 1.0
        )
        block_token_spreads: list[float] = []
        blocks_per_position = [
            condition_maps[index][identifier].get("blocks", []) for index in range(6)
        ]
        for block_index in range(len(blocks_per_position[0])):
            block_tokens = [
                int(
                    _path(
                        blocks_per_position[position][block_index],
                        "metrics",
                        "completion_tokens",
                    )
                )
                for position in range(6)
            ]
            block_token_spreads.append(
                (max(block_tokens) - min(block_tokens)) / max(block_tokens)
                if max(block_tokens) > 0
                else 1.0
            )
        maximum_block_token_spread = max(block_token_spreads, default=1.0)
        p95_slo = _path(reports[0], "manifest", "traffic", "p95_slo_ms")
        ttft_slo = _path(reports[0], "manifest", "traffic", "ttft_slo_ms")
        slo_failed = False
        for run_condition in (mapping[identifier] for mapping in condition_maps):
            if p95_slo is not None:
                high = _path(
                    run_condition,
                    "metrics",
                    "e2e_latency_ms",
                    "p95_repeated_block_ci",
                    "high",
                )
                if not isinstance(high, (int, float)) or high > p95_slo:
                    slo_failed = True
            if ttft_slo is not None:
                high = _path(
                    run_condition,
                    "metrics",
                    "ttft_ms",
                    "p95_repeated_block_ci",
                    "high",
                )
                if not isinstance(high, (int, float)) or high > ttft_slo:
                    slo_failed = True
        low, high = interval.get("low"), interval.get("high")
        state = "inconclusive"
        outcome = None
        tokens_comparable = (
            token_difference <= 0.05
            and position_token_spread <= 0.05
            and maximum_block_token_spread <= 0.05
        )
        if (
            tokens_comparable
            and not slo_failed
            and isinstance(low, (int, float))
            and isinstance(high, (int, float))
        ):
            if low > 0:
                state, outcome = "supported", "candidate_higher_throughput"
            elif high < 0:
                state, outcome = "supported", "baseline_higher_throughput"
        if outcome:
            outcomes.add(outcome)
        else:
            all_supported = False
        output["conditions"].append(
            {
                "condition_id": identifier,
                "state": state,
                "outcome": outcome,
                "throughput_delta_percent_ci": interval,
                "completion_token_relative_difference": token_difference,
                "completion_token_relative_spread_across_positions": position_token_spread,
                "maximum_block_completion_token_relative_spread_across_positions": maximum_block_token_spread,
                "completion_token_tolerance": 0.05,
                "reason": (
                    "completion_tokens_outside_5_percent_tolerance"
                    if not tokens_comparable
                    else "one_or_more_runs_fail_declared_slo"
                    if slo_failed
                    else None
                    if outcome
                    else "order_balanced_ci_includes_zero"
                ),
                "independent_unit": "two order-balanced B-C-B / C-B-C phase contrasts; each position contains >=3 measured blocks",
            }
        )
    if all_supported and len(outcomes) == 1:
        output["decision_state"] = "supported"
        output["decision_eligible"] = True
        output["overall_outcome"] = next(iter(outcomes))
        output["decision_summary"] = _supported_decision_summary(
            output["overall_outcome"], output["conditions"][0], reports, treatment
        )
    return output
