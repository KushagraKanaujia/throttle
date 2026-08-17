"""Eligibility and aggregation for the order-balanced golden live protocol."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from . import __version__
from .benchmark import SCHEMA_VERSION
from .compare import _condition_map, _path, _safe_preflight_reason
from .statistics import relative_delta_percent, t_interval_95

GOLDEN_ARTIFACT_TYPE = "throttle_golden_live_comparison"
EXPECTED_VARIANTS = (
    "baseline",
    "candidate",
    "baseline",
    "candidate",
    "baseline",
    "candidate",
)
IMMUTABLE_IMAGE = re.compile(r"^(?:[^\s]+@)?sha256:[0-9a-f]{64}$")
IMMUTABLE_REVISION = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


def _safe_report_fingerprint(report: Mapping[str, Any]) -> str | None:
    try:
        encoded = json.dumps(
            report, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError):
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
        normalized = str(name).replace("_", "-").lower()
        if not re.fullmatch(r"[a-z0-9-]{1,128}", normalized) or not isinstance(
            value, str
        ):
            return None
        output[normalized] = value
    return output


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


def _protocol_checks(reports: Sequence[Mapping[str, Any]]) -> list[str]:
    reasons: list[str] = []
    if len(reports) != 6:
        return ["golden_sequence_requires_exactly_six_runs"]
    for index, report in enumerate(reports):
        reason = _safe_preflight_reason(report)
        if reason:
            reasons.append(f"run_{index + 1}_{reason}")
    if reasons:
        return reasons
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
    if positions != ("B1", "C1", "B2", "C2", "B3", "C3"):
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

    required_common = (
        ("manifest", "tool"),
        ("manifest", "model", "id"),
        ("manifest", "model", "immutable_revision"),
        ("manifest", "runtime", "image_digest"),
        ("manifest", "runtime", "gpu"),
        ("manifest", "runtime", "gpu_fingerprint_sha256"),
        ("manifest", "runtime", "gpu_fingerprint_supplied"),
        ("manifest", "runtime", "cuda_version"),
        ("manifest", "runtime", "driver_version"),
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
    image = _path(reports[0], "manifest", "runtime", "image_digest")
    revision = _path(reports[0], "manifest", "model", "immutable_revision")
    if not isinstance(image, str) or not IMMUTABLE_IMAGE.fullmatch(image):
        reasons.append("image_is_not_pinned_by_digest")
    if not isinstance(revision, str) or not IMMUTABLE_REVISION.fullmatch(revision):
        reasons.append("model_revision_is_not_immutable")
    for path, code in (
        (("manifest", "runtime", "gpu"), "gpu_identity_missing"),
        (("manifest", "runtime", "gpu_fingerprint_sha256"), "gpu_fingerprint_missing"),
        (("manifest", "runtime", "cuda_version"), "cuda_version_missing"),
        (("manifest", "runtime", "driver_version"), "driver_version_missing"),
        (("manifest", "engine", "server_version"), "server_version_missing"),
    ):
        value = _path(reports[0], *path)
        if not isinstance(value, str) or value == "unknown":
            reasons.append(code)
    if _path(reports[0], "manifest", "runtime", "gpu_fingerprint_supplied") is not True:
        reasons.append("gpu_fingerprint_not_supplied")
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
        elif condition_sets[0] != {"closed_loop:8"}:
            # The controlled 1-versus-8 max_num_seqs treatment is not expected
            # to affect lower loads. Requiring one exercised level prevents an
            # honest no-effect control level from vetoing (or being pooled
            # into) the treatment decision.
            reasons.append("golden_treatment_requires_only_closed_loop_concurrency_8")

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
    try:
        baseline_limit = int(baseline_flags[0]["max-num-seqs"])
        candidate_limit = int(candidate_flags[0]["max-num-seqs"])
    except (KeyError, TypeError, ValueError):
        reasons.append("max_num_seqs_values_missing")
    else:
        if (baseline_limit, candidate_limit) != (1, 8):
            reasons.append("golden_treatment_requires_max_num_seqs_1_vs_8")
        actually_exercised = all(
            any(
                isinstance(condition, Mapping)
                and isinstance(condition.get("observed_peak_in_flight"), int)
                and condition.get("observed_peak_in_flight") >= 8
                for condition in report.get("conditions", [])
            )
            for report in reports
        )
        if not actually_exercised:
            reasons.append("traffic_does_not_exercise_candidate_max_num_seqs")
    # Chunked prefill may be present, but must be common and receives no credit.
    chunked = [flag.get("enable-chunked-prefill") for flag in flags if flag is not None]
    if any(value != chunked[0] for value in chunked[1:]):
        reasons.append("chunked_prefill_must_not_be_the_treatment")
    return sorted(set(reasons))


def validate_golden_sequence(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate and compare six B-C-B / C-B-C sequential live reports."""

    reasons = _protocol_checks(reports)
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
        "run_fingerprints": [_safe_report_fingerprint(report) for report in reports],
        "sequence": list(EXPECTED_VARIANTS),
        "conditions": [],
        "overall_outcome": None,
        "optimization_credit": {
            "changed_flag": "max_num_seqs" if not reasons else None,
            "chunked_prefill_credited": False,
        },
        "verification_scope": (
            "Throttle validates internal report consistency, ordering, pins, and declared "
            "runtime-verified provenance. It cannot independently prove operator-supplied "
            "hardware or runtime attestations; retain external audit evidence."
        ),
        "disclaimer": "This order-balanced live result applies only to the pinned manifest and tested workload; it is not a savings projection or universal causal claim.",
    }
    if reasons:
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
    return output
