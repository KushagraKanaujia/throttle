from __future__ import annotations

import asyncio
import copy
import hashlib
import ipaddress
import json
import os
import socket
import tempfile
import time
import unittest
from collections.abc import Iterator, Mapping
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import httpx

from throttle import __version__
from throttle.benchmark import (
    ARTIFACT_TYPE,
    BlockOutcome,
    RequestResult,
    RunBudget,
    SCHEMA_VERSION,
    _block_report,
    _best_tested,
    canonical_workload_hash,
    run_native,
)
from throttle.compare import (
    MAX_REPORT_BYTES,
    ComparisonInputError,
    compare_reports,
    load_report,
)
from throttle.golden import (
    GoldenTreatmentError,
    golden_positions,
    parse_golden_treatment_flags,
    validate_golden_sequence,
)
from throttle.models import (
    CostModel,
    EndpointConfig,
    LoadCondition,
    RunConfig,
    SafetyLimits,
)
from throttle.request_profile import build_request_manifest
from throttle.statistics import (
    _t_critical_975,
    paired_relative_delta_interval_95,
    relative_delta_percent,
    t_interval_95,
)


_REAL_GETADDRINFO = socket.getaddrinfo
_REAL_CONNECT = socket.socket.connect
_NETWORK_PATCHES: tuple[object, ...] = ()


def _loopback_host(value: object) -> bool:
    if isinstance(value, bytes):
        value = value.decode("ascii", errors="ignore")
    if not isinstance(value, str):
        return False
    host = value.strip("[]").rstrip(".").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _offline_getaddrinfo(host: object, *args: object, **kwargs: object) -> object:
    if not _loopback_host(host):
        raise AssertionError(f"offline test attempted non-loopback DNS: {host!r}")
    return _REAL_GETADDRINFO(host, *args, **kwargs)


def _offline_connect(sock: socket.socket, address: object) -> object:
    if isinstance(address, str):
        # AF_UNIX is local by definition.
        return _REAL_CONNECT(sock, address)
    host = address[0] if isinstance(address, tuple) and address else None
    if not _loopback_host(host):
        raise AssertionError(f"offline test attempted non-loopback connect: {host!r}")
    return _REAL_CONNECT(sock, address)


def setUpModule() -> None:
    global _NETWORK_PATCHES
    dns = patch.object(socket, "getaddrinfo", side_effect=_offline_getaddrinfo)
    connect = patch.object(socket.socket, "connect", new=_offline_connect)
    dns.start()
    connect.start()
    _NETWORK_PATCHES = (dns, connect)


def tearDownModule() -> None:
    for network_patch in reversed(_NETWORK_PATCHES):
        network_patch.stop()  # type: ignore[attr-defined]


PROMPTS = (({"role": "user", "content": "private measured prompt"},),)
WARMUPS = (({"role": "user", "content": "private separate warmup"},),)


def _completion(tokens: int = 4) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "private generated response",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": tokens},
        },
    )


def _limits() -> SafetyLimits:
    return SafetyLimits(
        max_requests=1_000,
        max_tokens_per_request=32,
        max_total_requested_tokens=32_000,
        max_elapsed_seconds=5.0,
        max_errors=1,
        max_concurrency=16,
        max_response_bytes=100_000,
        max_estimated_spend=3.0,
    )


def _run_config(
    *,
    mode: str = "smoke",
    conditions: tuple[LoadCondition, ...] = (LoadCondition("closed_loop", 1.0, 1),),
    blocks: int | None = None,
    requests_per_block: int = 1,
) -> RunConfig:
    return RunConfig(
        mode=mode,  # type: ignore[arg-type]
        backend="native",
        model="model-a",
        endpoint=EndpointConfig("https://offline-only.example/v1", "private-key"),
        cost=CostModel(
            kind="dedicated_hourly",
            total_hourly_rate=0.25,
            gpu_count=1,
        ),
        limits=_limits(),
        max_tokens=8,
        conditions=conditions,
        blocks=(1 if mode == "smoke" else 3) if blocks is None else blocks,
        requests_per_block=requests_per_block,
        warmup_requests_per_condition=0,
        request_timeout_seconds=1.0,
        stream=False,
        cache_policy="disabled",
        model_revision="a" * 40,
        image_digest="example/image@sha256:" + "b" * 64,
        gpu="test-gpu",
        gpu_fingerprint="same-test-gpu",
        cuda_version="test-cuda",
        driver_version="test-driver",
        server_version="test-server",
        engine_flags_provenance="runtime_verified",
        evidence_source="synthetic_validation",
    )


def _condition_result(value: int, throughput: float) -> dict[str, object]:
    return {
        "condition": {
            "id": f"closed_loop:{value}",
            "kind": "closed_loop",
            "value": value,
            "max_in_flight": value,
        },
        "valid": True,
        "decision_grade": True,
        "metrics": {
            "valid_response_count": 201,
            "completion_tokens": 2010,
            "output_tokens_per_second": throughput,
            "block_mean_output_tokens_per_second": throughput,
            "block_mean_output_tokens_per_second_ci": {
                "low": throughput - 0.1,
                "high": throughput + 0.1,
                "confidence": 0.95,
                "method": "student_t_blocks",
                "n": 3,
            },
            "e2e_latency_ms": {"p95": 10.0},
            "ttft_ms": {"p95": 2.0},
        },
    }


def _saved_report(
    block_rates: tuple[float, float, float],
    *,
    completion_tokens: int = 300,
    cost_kind: str = "dedicated_hourly",
    flag_value: str = "1",
    evidence_source: str = "live_inference",
    variant: str = "baseline",
    sequence_position: str = "B1",
    started_at: str = "2026-08-17T00:00:00+00:00",
    completed_at: str = "2026-08-17T00:00:35+00:00",
    closed_loop_concurrency: int = 8,
    observed_peak_in_flight: int | None = None,
) -> dict[str, object]:
    if observed_peak_in_flight is None:
        observed_peak_in_flight = closed_loop_concurrency
    condition_descriptor = {
        "id": f"closed_loop:{closed_loop_concurrency}",
        "kind": "closed_loop",
        "value": closed_loop_concurrency,
        "max_in_flight": closed_loop_concurrency,
    }
    if len(block_rates) != 3 or any(rate <= 0 for rate in block_rates):
        raise ValueError("saved fixture requires three positive block rates")
    quotient, remainder = divmod(completion_tokens, len(block_rates))
    block_tokens = tuple(
        quotient + (1 if index < remainder else 0) for index in range(len(block_rates))
    )
    if any(tokens <= 0 for tokens in block_tokens):
        raise ValueError("saved fixture requires positive tokens in every block")
    block_walls = tuple(
        tokens / rate for tokens, rate in zip(block_tokens, block_rates)
    )
    aggregate_wall = sum(block_walls)
    aggregate_throughput = completion_tokens / aggregate_wall
    block_mean_throughput = sum(block_rates) / len(block_rates)
    block_request_rates = tuple(67 / wall for wall in block_walls)
    hourly_rate = 0.25
    dedicated_cost_per_million = (
        hourly_rate * aggregate_wall / 3_600.0 / completion_tokens * 1_000_000.0
    )

    def counts(attempted: int) -> dict[str, object]:
        return {
            "attempted": attempted,
            "valid": attempted,
            "invalid": 0,
            "status_counts": {"200": attempted},
            "error_counts": {},
            "finish_reason_counts": {"stop": attempted},
        }

    def distribution(mean: float, count: int = 201) -> dict[str, object]:
        return {
            "count": count,
            "mean": mean,
            "p50": mean,
            "p90": mean,
            "p95": mean,
            "p99": mean,
            "p95_ci": {
                "low": mean,
                "high": mean,
                "confidence": 0.95,
                "method": "percentile_bootstrap_requests",
                "n": count,
                "resamples": 1_000,
            },
        }

    condition_metrics = {
        "valid_response_count": 201,
        "completion_tokens": completion_tokens,
        "prompt_tokens": 603,
        "requests_per_second": 201 / aggregate_wall,
        "output_tokens_per_second": aggregate_throughput,
        "block_mean_output_tokens_per_second": block_mean_throughput,
        "block_mean_output_tokens_per_second_ci": t_interval_95(block_rates),
        "block_mean_requests_per_second": sum(block_request_rates)
        / len(block_request_rates),
        "block_mean_requests_per_second_ci": t_interval_95(block_request_rates),
        "error_rate": 0.0,
        "e2e_latency_ms": distribution(10.0),
        "ttft_ms": distribution(2.0),
        "tpot_ms": distribution(1.0),
        "itl_ms": {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "source": "unavailable",
            "unavailable_reason": "native SSE chunks do not prove token boundaries",
        },
        "inter_chunk_latency_ms": {
            **distribution(1.0),
            "source": "client_observed_nonempty_sse_chunks",
            "not_equivalent_to_itl": True,
        },
        "slo_goodput": None,
        "independent_ci_unit": "repeated_block",
        "cost_per_million_output_tokens": (
            dedicated_cost_per_million if cost_kind == "dedicated_hourly" else None
        ),
        "cost_metric_basis": (
            "dedicated hourly rate times measured condition wall time"
            if cost_kind == "dedicated_hourly"
            else "not exactly available for this fixture billing model"
        ),
    }
    condition_metrics["e2e_latency_ms"]["p95_repeated_block_ci"] = t_interval_95(
        [10.0] * len(block_rates)
    )
    condition_metrics["ttft_ms"]["p95_repeated_block_ci"] = t_interval_95(
        [2.0] * len(block_rates)
    )
    blocks = []
    for index, (value, tokens, wall) in enumerate(
        zip(block_rates, block_tokens, block_walls), start=1
    ):
        block_metrics = {
            **condition_metrics,
            "valid_response_count": 67,
            "completion_tokens": tokens,
            "prompt_tokens": 201,
            "requests_per_second": 67 / wall,
            "output_tokens_per_second": value,
            "e2e_latency_ms": distribution(10.0, 67),
            "ttft_ms": distribution(2.0, 67),
            "tpot_ms": distribution(1.0, 67),
            "inter_chunk_latency_ms": {
                **distribution(1.0, 67),
                "source": "client_observed_nonempty_sse_chunks",
                "not_equivalent_to_itl": True,
            },
            "cost_per_million_output_tokens": (
                hourly_rate * wall / 3_600.0 / tokens * 1_000_000.0
                if cost_kind == "dedicated_hourly"
                else None
            ),
        }
        blocks.append(
            {
                "block_index": index,
                "valid": True,
                "invalid_reasons": [],
                "wall_duration_seconds": wall,
                "observed_peak_in_flight": observed_peak_in_flight,
                "request_counts": counts(67),
                "offered_requests": 67,
                "scheduler_lag_ms": distribution(0.0, 67),
                "metrics": block_metrics,
                "diagnostic_metrics": copy.deepcopy(block_metrics),
            }
        )

    limits = _limits().public_dict()
    # Saved decision-grade fixtures model 128-token requests over sustained blocks.
    limits["max_tokens_per_request"] = 128
    limits["max_elapsed_seconds"] = 120.0
    cost_manifest: dict[str, object] = {
        "kind": cost_kind,
        "currency": "USD",
        "accounting_basis": "fixture",
    }
    if cost_kind == "dedicated_hourly":
        cost_manifest.update(
            pre_run_upper_bound=hourly_rate
            * float(limits["max_elapsed_seconds"])
            / 3_600.0,
            total_hourly_rate=hourly_rate,
            gpu_count=1,
        )
    elif cost_kind == "serverless_active_seconds":
        cost_manifest.update(
            pre_run_upper_bound=0.001 * 8 * float(limits["max_elapsed_seconds"]),
            active_second_rate=0.001,
            max_active_workers=8,
            billed_active_seconds=None,
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "generated_at": started_at,
        "started_at": started_at,
        "completed_at": completed_at,
        "mode": "benchmark",
        "status": "complete",
        "decision_eligible": False,
        "manifest": {
            "manifest_version": "1.0",
            "tool": {"name": "throttle-bench", "version": __version__},
            "model": {"id": "model-a", "immutable_revision": "a" * 40},
            "runtime": {
                "image_digest": "example/image@sha256:" + "b" * 64,
                "gpu": "test-gpu",
                "gpu_fingerprint_sha256": "e" * 64,
                "gpu_fingerprint_supplied": True,
                "cuda_version": "test-cuda",
                "driver_version": "test-driver",
            },
            "workload": {
                "seed": 42,
                "measured_sha256": "c" * 64,
                "warmup_sha256": "d" * 64,
                "measured_prompt_count": 8,
                "warmup_prompt_count": 3,
                "warmup_is_separate": True,
                "warmup_prompts_disjoint": True,
                "cache_policy": "disabled",
            },
            "request": {
                "type": "chat_completions",
                "temperature": 0,
                "max_tokens": 128,
                "stop": None,
                "stream": True,
                "timeout_seconds": 120.0,
            },
            "traffic": {
                "conditions": [condition_descriptor],
                "blocks": 3,
                "requests_per_block": 67,
                "block_duration_seconds": None,
                "warmup_requests_per_condition": 3,
                "p95_slo_ms": None,
                "ttft_slo_ms": None,
                "open_loop_rate_relative_tolerance": 0.05,
                "open_loop_scheduler_lag_interval_tolerance": 1.0,
            },
            "metric_definitions": {
                "e2e_latency": "validated terminal completion",
                "ttft": "first nonempty output-bearing delta",
                "tpot": "first-to-last output over token gaps",
                "itl": "unavailable when chunks do not prove token boundaries",
                "inter_chunk_latency": "observed output-bearing chunk gaps",
                "throughput": "validated output tokens per measured wall second",
                "slo_goodput": "valid SLO-passing requests per measured wall second",
            },
            "engine": {
                "backend": "native",
                "backend_version": "native-protocol-1",
                "http_client_version": httpx.__version__,
                "server_version": "test-server",
                "effective_flags": {
                    "max_num_seqs": flag_value,
                    "enable_chunked_prefill": "true",
                },
                "effective_flags_provenance": "runtime_verified",
            },
            "cost": cost_manifest,
            "provenance": {
                "evidence_source": evidence_source,
                "variant": variant,
                "sequence_position": sequence_position,
            },
            "safety": {
                "limits": limits,
                "overrides": {
                    "insecure_http": False,
                    "unknown_cost_acknowledged": False,
                    "guidellm_validation_gaps_acknowledged": False,
                },
                "ambient_proxy_environment_used": False,
                "redirects_followed": False,
            },
            "optimization_credit_exclusions": [
                {
                    "feature": "enable_chunked_prefill",
                    "reason": "present in both variants and receives no optimization credit",
                }
            ],
        },
        "conditions": [
            {
                "condition": condition_descriptor,
                "valid": True,
                "decision_grade": True,
                "decision_ineligible_reasons": [],
                "qualification_floor": {
                    "minimum_valid_requests": 200,
                    "or_minimum_measured_seconds": 60.0,
                    "minimum_blocks": 3,
                },
                "warmup": counts(3),
                "blocks": blocks,
                "request_counts": counts(201),
                "measured_wall_seconds": aggregate_wall,
                "observed_peak_in_flight": observed_peak_in_flight,
                "target_offered_request_rate": None,
                "achieved_offered_request_rate": None,
                "offered_rate_relative_error": None,
                "open_loop_target_achieved": None,
                "metrics": condition_metrics,
                "diagnostic_metrics": copy.deepcopy(condition_metrics),
            }
        ],
        "best_tested": {
            "field": "best_tested_concurrency",
            "available": True,
            "value": closed_loop_concurrency,
            "condition_id": f"closed_loop:{closed_loop_concurrency}",
            "block_mean_output_tokens_per_second": block_mean_throughput,
            "block_mean_output_tokens_per_second_ci": condition_metrics[
                "block_mean_output_tokens_per_second_ci"
            ],
            "pooled_output_tokens_per_second": aggregate_throughput,
            "state": "inconclusive",
            "reasons": ["search_boundary_reached"],
            "boundary_reached": True,
            "optimum_found": False,
            "claim": "best condition among only the tested values for this workload",
        },
        "run_totals": {
            "requests_started": 204,
            "requests_completed": 204,
            "requests_cancelled": 0,
            "requests_in_flight": 0,
            "peak_in_flight": observed_peak_in_flight,
            "errors": 0,
            "reserved_output_tokens": 26_112,
            "elapsed_seconds": aggregate_wall,
        },
        "cost_summary": {
            "kind": cost_kind,
            "total_cost": (
                hourly_rate * aggregate_wall / 3_600.0
                if cost_kind == "dedicated_hourly"
                else None
            ),
            "basis": "fixture",
            "completion_tokens": completion_tokens,
            "cost_per_million_output_tokens": (
                dedicated_cost_per_million if cost_kind == "dedicated_hourly" else None
            ),
        },
        "stop_reason": None,
        "disclaimer": "Fixture measurements only; not a savings projection.",
    }


def _set_block_completion_tokens(
    report: dict[str, object], block_tokens: tuple[int, int, int]
) -> None:
    """Change block output lengths while preserving all recomputable evidence."""

    condition = report["conditions"][0]  # type: ignore[index]
    blocks = condition["blocks"]  # type: ignore[index]
    hourly_rate = report["manifest"]["cost"]["total_hourly_rate"]  # type: ignore[index]
    walls: list[float] = []
    for block, tokens in zip(blocks, block_tokens):
        metrics = block["metrics"]
        rate = float(metrics["output_tokens_per_second"])
        wall = tokens / rate
        walls.append(wall)
        block["wall_duration_seconds"] = wall
        metrics["completion_tokens"] = tokens
        metrics["requests_per_second"] = block["request_counts"]["valid"] / wall
        metrics["cost_per_million_output_tokens"] = (
            hourly_rate * wall / 3_600.0 / tokens * 1_000_000.0
        )
        block["diagnostic_metrics"] = copy.deepcopy(metrics)

    total_tokens = sum(block_tokens)
    total_wall = sum(walls)
    condition_metrics = condition["metrics"]
    condition_metrics["completion_tokens"] = total_tokens
    condition_metrics["output_tokens_per_second"] = total_tokens / total_wall
    condition_metrics["requests_per_second"] = (
        condition["request_counts"]["valid"] / total_wall
    )
    block_rates = [
        float(block["metrics"]["output_tokens_per_second"]) for block in blocks
    ]
    block_request_rates = [
        float(block["metrics"]["requests_per_second"]) for block in blocks
    ]
    condition_metrics["block_mean_output_tokens_per_second"] = sum(block_rates) / len(
        block_rates
    )
    condition_metrics["block_mean_output_tokens_per_second_ci"] = t_interval_95(
        block_rates
    )
    condition_metrics["block_mean_requests_per_second"] = sum(
        block_request_rates
    ) / len(block_request_rates)
    condition_metrics["block_mean_requests_per_second_ci"] = t_interval_95(
        block_request_rates
    )
    condition_metrics["cost_per_million_output_tokens"] = (
        hourly_rate * total_wall / 3_600.0 / total_tokens * 1_000_000.0
    )
    condition["measured_wall_seconds"] = total_wall
    condition["diagnostic_metrics"] = copy.deepcopy(condition_metrics)
    report["best_tested"]["block_mean_output_tokens_per_second"] = (  # type: ignore[index]
        condition_metrics["block_mean_output_tokens_per_second"]
    )
    report["best_tested"]["block_mean_output_tokens_per_second_ci"] = (  # type: ignore[index]
        condition_metrics["block_mean_output_tokens_per_second_ci"]
    )
    report["best_tested"]["pooled_output_tokens_per_second"] = (  # type: ignore[index]
        total_tokens / total_wall
    )
    report["run_totals"]["elapsed_seconds"] = total_wall  # type: ignore[index]
    total_cost = hourly_rate * total_wall / 3_600.0
    report["cost_summary"].update(  # type: ignore[union-attr]
        completion_tokens=total_tokens,
        total_cost=total_cost,
        cost_per_million_output_tokens=total_cost / total_tokens * 1_000_000.0,
    )


def _set_nonstream_ttft_unavailable(report: dict[str, object]) -> None:
    """Make saved timing evidence match a genuine native non-streaming run."""

    diagnostic_interval = {
        "low": None,
        "high": None,
        "confidence": 0.95,
        "method": "bounded_percentile_bootstrap_requests",
        "n": 0,
        "analysis_sample_n": 0,
        "resamples": 300,
    }
    unavailable = {
        "count": 0,
        "mean": None,
        "p50": None,
        "p90": None,
        "p95": None,
        "p99": None,
        "p95_ci": diagnostic_interval,
    }
    condition = report["conditions"][0]  # type: ignore[index]
    for block in condition["blocks"]:
        block["metrics"]["ttft_ms"] = copy.deepcopy(unavailable)
        block["diagnostic_metrics"]["ttft_ms"] = copy.deepcopy(unavailable)
    aggregate = {
        **copy.deepcopy(unavailable),
        "p95_repeated_block_ci": t_interval_95([]),
    }
    condition["metrics"]["ttft_ms"] = copy.deepcopy(aggregate)
    condition["diagnostic_metrics"]["ttft_ms"] = copy.deepcopy(aggregate)


NON_CUDA_RUNTIME_CASES = (
    (
        "rocm",
        "AMD Instinct MI300X",
        "ROCm 6.3.1",
        "Ubuntu 24.04 kernel 6.8.0",
    ),
    (
        "cpu",
        "AMD EPYC 9654",
        "oneDNN 3.7.1",
        "Ubuntu 24.04 kernel 6.8.0",
    ),
)


def _set_platform_runtime(
    report: dict[str, object],
    *,
    backend: str,
    accelerator: str,
    runtime_version: str,
    host_os_version: str,
) -> None:
    manifest = report["manifest"]  # type: ignore[index]
    manifest["manifest_version"] = "1.1"  # type: ignore[index]
    runtime = manifest["runtime"]  # type: ignore[index]
    runtime.update(  # type: ignore[union-attr]
        image_digest="unknown",
        gpu=accelerator,
        gpu_fingerprint_sha256="e" * 64,
        gpu_fingerprint_supplied=True,
        cuda_version="unknown",
        driver_version="unknown",
        accelerator_backend=backend,
        accelerator=accelerator,
        accelerator_fingerprint_sha256="e" * 64,
        accelerator_fingerprint_supplied=True,
        accelerator_runtime_version=runtime_version,
        host_os_version=host_os_version,
        software_environment_digest=f"{backend}-environment@sha256:" + "f" * 64,
    )


def _set_metal_runtime(report: dict[str, object]) -> None:
    _set_platform_runtime(
        report,
        backend="metal",
        accelerator="Apple Silicon integrated GPU",
        runtime_version="MLX 0.32.0",
        host_os_version="macOS 15.0 build 24A335",
    )
def _set_request_profile(
    report: dict[str, object],
    fields: tuple[tuple[str, str | int | float | bool | None], ...],
) -> None:
    config = replace(
        _run_config(),
        max_tokens=128,
        request_timeout_seconds=120.0,
        stream=True,
        request_fields=fields,
    )
    report["manifest"]["request"] = build_request_manifest(config)  # type: ignore[index]


def _golden_sequence(
    completion_tokens_by_position: tuple[int, int, int, int, int, int] = (
        300,
        300,
        300,
        300,
        300,
        300,
    ),
    *,
    baseline_block_rates: tuple[float, float, float] = (10.0, 10.0, 10.0),
    candidate_block_rates: tuple[float, float, float] = (12.0, 12.0, 12.0),
    baseline_max_num_seqs: int | str = 1,
    candidate_max_num_seqs: int | str = 8,
    closed_loop_concurrency: int = 8,
    observed_peak_in_flight: int | None = None,
) -> list[dict[str, object]]:
    variants = (
        "baseline",
        "candidate",
        "baseline",
        "candidate",
        "baseline",
        "candidate",
    )
    positions = ("B1", "C1", "B2", "C2", "B3", "C3")
    origin = datetime(2026, 8, 17, tzinfo=timezone.utc)
    reports: list[dict[str, object]] = []
    for index, (variant, position, completion_tokens) in enumerate(
        zip(variants, positions, completion_tokens_by_position)
    ):
        started = origin + timedelta(seconds=index * 40)
        completed = started + timedelta(seconds=35)
        baseline = variant == "baseline"
        reports.append(
            _saved_report(
                baseline_block_rates if baseline else candidate_block_rates,
                completion_tokens=completion_tokens,
                flag_value=str(
                    baseline_max_num_seqs if baseline else candidate_max_num_seqs
                ),
                evidence_source="live_inference",
                variant=variant,
                sequence_position=position,
                started_at=started.isoformat(),
                completed_at=completed.isoformat(),
                closed_loop_concurrency=closed_loop_concurrency,
                observed_peak_in_flight=observed_peak_in_flight,
            )
        )
    return reports


class RunConfigCompatibilityTests(unittest.TestCase):
    def test_legacy_positional_server_version_slot_is_preserved(self) -> None:
        config = RunConfig(
            "smoke",
            "native",
            "model-a",
            EndpointConfig("https://offline-only.example/v1", "private-key"),
            CostModel(),
            SafetyLimits(),
            128,
            (LoadCondition("closed_loop", 1.0, 1),),
            1,
            8,
            None,
            1,
            120.0,
            None,
            None,
            42,
            True,
            "unknown",
            "unknown",
            "unknown",
            "unknown",
            "unknown",
            "unknown",
            "unknown",
            "legacy-server-version",
        )

        self.assertEqual(config.server_version, "legacy-server-version")
        self.assertEqual(config.accelerator_backend, "cuda")


class CostModelTests(unittest.TestCase):
    def test_cost_models_are_tagged_and_never_mix_math(self) -> None:
        dedicated = CostModel(
            kind="dedicated_hourly",
            total_hourly_rate=2.0,
            gpu_count=2,
        )
        dedicated.validate()
        self.assertEqual(dedicated.estimated_upper_bound(1800.0), 1.0)
        self.assertEqual(
            dedicated.final_cost(1800.0), (1.0, "observed_client_wall_time")
        )

        serverless = CostModel(
            kind="serverless_active_seconds",
            active_second_rate=0.01,
            max_active_workers=2,
            billed_active_seconds=4.0,
        )
        serverless.validate()
        self.assertEqual(serverless.estimated_upper_bound(10.0), 0.2)
        self.assertEqual(
            serverless.final_cost(999.0),
            (0.04, "user_supplied_billed_active_seconds"),
        )

        supplied = CostModel(kind="user_supplied", user_supplied_total=0.75)
        supplied.validate()
        self.assertEqual(supplied.final_cost(1.0), (0.75, "user_supplied_run_total"))

        unknown = CostModel()
        unknown.validate()
        self.assertIsNone(unknown.estimated_upper_bound(100.0))
        self.assertEqual(unknown.final_cost(100.0), (None, "billing_unknown"))

    def test_cost_model_rejects_cross_model_fields(self) -> None:
        invalid = (
            CostModel(kind="unknown", total_hourly_rate=1.0),
            CostModel(
                kind="dedicated_hourly",
                total_hourly_rate=1.0,
                gpu_count=1,
                active_second_rate=0.1,
            ),
            CostModel(
                kind="serverless_active_seconds",
                active_second_rate=0.1,
                max_active_workers=1,
                user_supplied_total=1.0,
            ),
        )
        for cost in invalid:
            with self.subTest(kind=cost.kind), self.assertRaises(ValueError):
                cost.validate()


class StatisticsAndBoundaryTests(unittest.TestCase):
    def test_block_confidence_intervals_are_deterministic(self) -> None:
        interval = t_interval_95([10.0, 10.0, 10.0])
        self.assertEqual(interval["low"], 10.0)
        self.assertEqual(interval["high"], 10.0)
        self.assertEqual(interval["n"], 3)
        self.assertEqual(interval["method"], "student_t_blocks")

        delta = paired_relative_delta_interval_95(
            [10.0, 10.0, 10.0],
            [12.0, 12.0, 12.0],
        )
        self.assertAlmostEqual(float(delta["estimate"]), 20.0)
        self.assertAlmostEqual(float(delta["low"]), 20.0)
        self.assertAlmostEqual(float(delta["high"]), 20.0)
        self.assertEqual(delta["n"], 3)

    def test_student_t_df_31_does_not_fall_back_to_normal_critical(self) -> None:
        critical = _t_critical_975(31)
        self.assertGreater(critical, 1.96)
        self.assertLess(critical, _t_critical_975(30))

    def test_highest_winner_is_boundary_not_optimum(self) -> None:
        config = _run_config(
            mode="benchmark",
            conditions=(
                LoadCondition("closed_loop", 1.0, 1),
                LoadCondition("closed_loop", 4.0, 4),
                LoadCondition("closed_loop", 8.0, 8),
            ),
            requests_per_block=67,
        )
        boundary = _best_tested(
            [
                _condition_result(1, 10.0),
                _condition_result(4, 20.0),
                _condition_result(8, 30.0),
            ],
            config,
        )
        self.assertTrue(boundary["boundary_reached"])
        self.assertFalse(boundary["optimum_found"])
        self.assertEqual(boundary["state"], "inconclusive")
        self.assertIn("search_boundary_reached", boundary["reasons"])

        interior = _best_tested(
            [
                _condition_result(1, 10.0),
                _condition_result(4, 30.0),
                _condition_result(8, 20.0),
            ],
            config,
        )
        self.assertFalse(interior["boundary_reached"])
        self.assertFalse(interior["optimum_found"])
        self.assertEqual(interior["state"], "inconclusive")
        self.assertIn("multi_condition_order_not_counterbalanced", interior["reasons"])
        self.assertEqual(interior["field"], "best_tested_concurrency")

    def test_best_tested_ranks_the_same_block_mean_estimand_as_its_ci(self) -> None:
        config = _run_config(
            mode="benchmark",
            conditions=(
                LoadCondition("closed_loop", 1.0, 1),
                LoadCondition("closed_loop", 4.0, 4),
                LoadCondition("closed_loop", 8.0, 8),
            ),
            blocks=3,
            requests_per_block=67,
        )
        conditions = [
            _condition_result(1, 10.0),
            _condition_result(4, 30.0),
            _condition_result(8, 20.0),
        ]
        # Pooled token/wall rates can differ from the arithmetic mean of block
        # rates when count-bounded blocks have unequal durations. They remain
        # descriptive and must not drive the selection paired with a block CI.
        conditions[0]["metrics"]["output_tokens_per_second"] = 100.0
        conditions[1]["metrics"]["output_tokens_per_second"] = 20.0
        conditions[2]["metrics"]["output_tokens_per_second"] = 15.0

        selected = _best_tested(conditions, config)
        self.assertEqual(selected["value"], 4)
        self.assertEqual(selected["block_mean_output_tokens_per_second"], 30.0)
        self.assertEqual(selected["pooled_output_tokens_per_second"], 20.0)
        self.assertEqual(selected["state"], "inconclusive")
        self.assertIn("multi_condition_order_not_counterbalanced", selected["reasons"])

    def test_best_tested_requires_comparable_output_work_per_response(self) -> None:
        count_config = _run_config(
            mode="benchmark",
            conditions=(
                LoadCondition("closed_loop", 1.0, 1),
                LoadCondition("closed_loop", 4.0, 4),
                LoadCondition("closed_loop", 8.0, 8),
            ),
            blocks=3,
            requests_per_block=67,
        )
        conditions = [
            _condition_result(1, 10.0),
            _condition_result(4, 30.0),
            _condition_result(8, 20.0),
        ]
        conditions[2]["metrics"]["completion_tokens"] = 1608
        guarded = _best_tested(conditions, count_config)
        self.assertEqual(guarded["value"], 4)
        self.assertEqual(guarded["state"], "inconclusive")
        self.assertIn(
            "completion_tokens_per_response_not_comparable_across_conditions",
            guarded["reasons"],
        )
        self.assertGreater(
            guarded["completion_tokens_per_response_relative_spread"],
            guarded["completion_tokens_per_response_tolerance"],
        )

        duration_config = replace(
            count_config,
            requests_per_block=None,
            block_duration_seconds=20.0,
        )
        duration_conditions = copy.deepcopy(conditions)
        for condition, valid_count in zip(duration_conditions, (100, 200, 300)):
            condition["metrics"]["valid_response_count"] = valid_count
            condition["metrics"]["completion_tokens"] = valid_count * 10
        comparable = _best_tested(duration_conditions, duration_config)
        self.assertEqual(comparable["value"], 4)
        self.assertEqual(comparable["state"], "inconclusive")
        self.assertNotIn(
            "completion_tokens_per_response_not_comparable_across_conditions",
            comparable["reasons"],
        )
        self.assertEqual(
            comparable["completion_tokens_per_response_relative_spread"], 0.0
        )

    def test_best_tested_checks_output_work_before_slo_filtering(self) -> None:
        config = replace(
            _run_config(
                mode="benchmark",
                conditions=(
                    LoadCondition("closed_loop", 1.0, 1),
                    LoadCondition("closed_loop", 4.0, 4),
                    LoadCondition("closed_loop", 8.0, 8),
                ),
                requests_per_block=67,
            ),
            p95_slo_ms=5.0,
        )
        conditions = [
            _condition_result(1, 10.0),
            _condition_result(4, 30.0),
            _condition_result(8, 20.0),
        ]
        for condition, p95_high in zip(conditions, (10.0, 2.0, 10.0)):
            condition["metrics"]["e2e_latency_ms"][  # type: ignore[index]
                "p95_repeated_block_ci"
            ] = {
                "low": p95_high,
                "high": p95_high,
                "confidence": 0.95,
                "method": "student_t_blocks",
                "n": 3,
            }
        # The sole SLO qualifier emits one token/response while the other valid
        # tested conditions emit ten.  SLO filtering must not hide that mismatch.
        conditions[1]["metrics"]["completion_tokens"] = 201  # type: ignore[index]

        selected = _best_tested(conditions, config)

        self.assertEqual(selected["value"], 4)
        self.assertEqual(selected["state"], "inconclusive")
        self.assertIn(
            "completion_tokens_per_response_not_comparable_across_conditions",
            selected["reasons"],
        )
        self.assertGreater(
            selected["completion_tokens_per_response_relative_spread"],
            selected["completion_tokens_per_response_tolerance"],
        )


class LoadAndComparisonTests(unittest.IsolatedAsyncioTestCase):
    async def test_shared_budget_enforces_cumulative_limits_across_native_runs(
        self,
    ) -> None:
        async def execute(
            config: RunConfig,
            shared_budget: RunBudget,
            handler: object,
        ) -> dict[str, object]:
            return await run_native(
                config,
                PROMPTS,
                WARMUPS,
                transport=httpx.MockTransport(handler),  # type: ignore[arg-type]
                shared_budget=shared_budget,
            )

        for name, limits, expected_reason in (
            (
                "requests",
                replace(_limits(), max_requests=2),
                "max_requests",
            ),
            (
                "tokens",
                replace(
                    _limits(),
                    max_requests=10,
                    max_total_requested_tokens=16,
                ),
                "max_total_requested_tokens",
            ),
        ):
            with self.subTest(limit=name):
                config = replace(_run_config(), limits=limits)
                shared_budget = RunBudget(config)
                calls = 0

                def success(_: httpx.Request) -> httpx.Response:
                    nonlocal calls
                    calls += 1
                    return _completion()

                first = await execute(config, shared_budget, success)
                second = await execute(config, shared_budget, success)
                blocked = await execute(config, shared_budget, success)

                self.assertEqual(first["status"], "complete")
                self.assertEqual(second["status"], "complete")
                self.assertNotEqual(blocked["status"], "complete")
                self.assertEqual(blocked["stop_reason"], expected_reason)
                self.assertEqual(calls, 2)
                totals = shared_budget.public_dict()
                self.assertEqual(totals["requests_started"], 2)
                self.assertEqual(totals["requests_completed"], 2)
                self.assertEqual(totals["reserved_output_tokens"], 16)
                self.assertEqual(totals["errors"], 0)

        error_config = replace(
            _run_config(),
            limits=replace(
                _limits(),
                max_requests=10,
                max_total_requested_tokens=80,
                max_errors=1,
            ),
        )
        error_budget = RunBudget(error_config)
        error_calls = 0

        def fail(_: httpx.Request) -> httpx.Response:
            nonlocal error_calls
            error_calls += 1
            return httpx.Response(500, json={"error": "deliberate test failure"})

        failed = await execute(error_config, error_budget, fail)
        blocked_after_error = await execute(error_config, error_budget, fail)

        self.assertNotEqual(failed["status"], "complete")
        self.assertEqual(failed["stop_reason"], "max_errors")
        self.assertNotEqual(blocked_after_error["status"], "complete")
        self.assertEqual(blocked_after_error["stop_reason"], "max_errors")
        self.assertEqual(error_calls, 1)
        error_totals = error_budget.public_dict()
        self.assertEqual(error_totals["requests_started"], 1)
        self.assertEqual(error_totals["requests_completed"], 1)
        self.assertEqual(error_totals["errors"], 1)

    async def test_cuda_runtime_keeps_the_existing_image_and_driver_contract(
        self,
    ) -> None:
        config = _run_config()
        report = await run_native(
            config,
            PROMPTS,
            WARMUPS,
            transport=httpx.MockTransport(lambda _: _completion()),
        )
        runtime = report["manifest"]["runtime"]
        self.assertEqual(runtime["accelerator_backend"], "cuda")
        self.assertEqual(
            runtime["accelerator_runtime_version"], config.cuda_version
        )
        self.assertEqual(
            runtime["software_environment_digest"], config.image_digest
        )
        self.assertNotIn(
            "complete_runtime_provenance_required",
            report["decision_ineligible_reasons"],
        )
        self.assertNotIn(
            "immutable_image_digest_required",
            report["decision_ineligible_reasons"],
        )

    async def test_metal_runtime_uses_platform_pins_instead_of_cuda_fields(
        self,
    ) -> None:
        config = replace(
            _run_config(),
            accelerator_backend="metal",
            accelerator_runtime_version="MLX 0.32.0",
            host_os_version="macOS 15.0 build 24A335",
            software_environment_digest="python-environment@sha256:" + "f" * 64,
            image_digest="unknown",
            gpu="Apple Silicon integrated GPU",
            gpu_fingerprint="private-apple-device-id",
            cuda_version="unknown",
            driver_version="unknown",
        )
        report = await run_native(
            config,
            PROMPTS,
            WARMUPS,
            transport=httpx.MockTransport(lambda _: _completion()),
        )

        runtime = report["manifest"]["runtime"]
        self.assertEqual(report["manifest"]["manifest_version"], "1.1")
        self.assertEqual(runtime["accelerator_backend"], "metal")
        self.assertEqual(runtime["accelerator"], "Apple Silicon integrated GPU")
        self.assertEqual(runtime["image_digest"], "unknown")
        self.assertEqual(runtime["cuda_version"], "unknown")
        self.assertNotIn(
            "complete_runtime_provenance_required",
            report["decision_ineligible_reasons"],
        )
        self.assertNotIn(
            "immutable_image_digest_required",
            report["decision_ineligible_reasons"],
        )

    async def test_metal_runtime_requires_an_immutable_environment_digest(
        self,
    ) -> None:
        config = replace(
            _run_config(),
            accelerator_backend="metal",
            accelerator_runtime_version="MLX 0.32.0",
            host_os_version="macOS 15.0 build 24A335",
            image_digest="unknown",
            gpu="Apple Silicon integrated GPU",
            cuda_version="unknown",
            driver_version="unknown",
        )
        report = await run_native(
            config,
            PROMPTS,
            WARMUPS,
            transport=httpx.MockTransport(lambda _: _completion()),
        )
        self.assertIn(
            "immutable_software_environment_digest_required",
            report["decision_ineligible_reasons"],
        )

    async def test_rocm_and_cpu_native_reports_use_platform_runtime_evidence(
        self,
    ) -> None:
        for backend, accelerator, runtime_version, host_os_version in (
            NON_CUDA_RUNTIME_CASES
        ):
            with self.subTest(backend=backend):
                config = replace(
                    _run_config(),
                    accelerator_backend=backend,  # type: ignore[arg-type]
                    accelerator_runtime_version=runtime_version,
                    host_os_version=host_os_version,
                    software_environment_digest=(
                        f"{backend}-environment@sha256:" + "f" * 64
                    ),
                    image_digest="unknown",
                    gpu=accelerator,
                    gpu_fingerprint=f"private-{backend}-device-id",
                    cuda_version="unknown",
                    driver_version="unknown",
                )
                report = await run_native(
                    config,
                    PROMPTS,
                    WARMUPS,
                    transport=httpx.MockTransport(lambda _: _completion()),
                )

                runtime = report["manifest"]["runtime"]
                self.assertEqual(runtime["accelerator_backend"], backend)
                self.assertEqual(runtime["accelerator"], accelerator)
                self.assertEqual(
                    runtime["accelerator_runtime_version"], runtime_version
                )
                self.assertEqual(runtime["host_os_version"], host_os_version)
                self.assertNotIn(
                    "complete_runtime_provenance_required",
                    report["decision_ineligible_reasons"],
                )
                self.assertNotIn(
                    "immutable_software_environment_digest_required",
                    report["decision_ineligible_reasons"],
                )

    async def test_hostile_generated_runtime_metadata_is_rejected_without_echo(
        self,
    ) -> None:
        def fail_if_called(_: httpx.Request) -> httpx.Response:
            raise AssertionError("invalid metadata reached traffic")

        base = replace(
            _run_config(),
            accelerator_backend="metal",
            accelerator_runtime_version="MLX 0.32.0",
            host_os_version="macOS 15.0 build 24A335",
            software_environment_digest="metal-environment@sha256:" + "f" * 64,
            image_digest="unknown",
            gpu="Apple Silicon integrated GPU",
            gpu_fingerprint="private-apple-device-id",
            cuda_version="unknown",
            driver_version="unknown",
        )
        hostile_cases = (
            ("gpu", "https://private-accelerator.example/credential"),
            ("accelerator_runtime_version", "Bearer private-runtime-token"),
            ("accelerator_runtime_version", "token=private-runtime-token"),
            ("accelerator_runtime_version", "access_token=private-runtime-token"),
            ("host_os_version", "authorization: private-host-token"),
            ("host_os_version", "ghp_1234567890abcdef"),
            ("host_os_version", "github_pat_1234567890abcdef"),
            ("host_os_version", "AKIA" + "A" * 16),
            ("host_os_version", "xoxb-1234567890abcdef"),
            (
                "accelerator_runtime_version",
                "runtime ghp_abcdefghijk",
            ),
            (
                "accelerator_runtime_version",
                "runtime eyJabc.eyJdef.signature",
            ),
            (
                "host_os_version",
                "device alice:hunter2@private.example",
            ),
            (
                "software_environment_digest",
                "env-ghp_abcdefghijk@sha256:" + "a" * 64,
            ),
            (
                "software_environment_digest",
                "env-eyJabc.eyJdef.signature@sha256:" + "a" * 64,
            ),
            (
                "software_environment_digest",
                "https://private-environment.example/secret@sha256:" + "a" * 64,
            ),
            (
                "software_environment_digest",
                "user:password@private-registry@sha256:" + "a" * 64,
            ),
            (
                "software_environment_digest",
                "/Users/example/env@sha256:" + "a" * 64,
            ),
            (
                "software_environment_digest",
                "private-environment\x1bcontrol@sha256:" + "a" * 64,
            ),
            (
                "software_environment_digest",
                "user:password@sha256:" + "a" * 64,
            ),
            (
                "software_environment_digest",
                "file:/Users/example/env@sha256:" + "a" * 64,
            ),
        )
        for field, hostile in hostile_cases:
            with self.subTest(field=field):
                config = replace(base, **{field: hostile})
                with self.assertRaises(ValueError) as raised:
                    await run_native(
                        config,
                        PROMPTS,
                        WARMUPS,
                        transport=httpx.MockTransport(fail_if_called),
                    )
                self.assertNotIn(hostile, str(raised.exception))

    async def test_saved_metal_runs_can_support_a_decision_grade_comparison(
        self,
    ) -> None:
        baseline = _saved_report((10.0, 10.0, 10.0), flag_value="1")
        candidate = _saved_report((12.0, 12.0, 12.0), flag_value="8")
        for report in (baseline, candidate):
            _set_metal_runtime(report)

        comparison = compare_reports(baseline, candidate)
        self.assertTrue(comparison["compatibility"]["compatible"])
        self.assertTrue(comparison["decision_eligible"])
        self.assertEqual(
            comparison["overall_outcome"], "candidate_higher_throughput"
        )

        candidate["manifest"]["runtime"][  # type: ignore[index]
            "accelerator_runtime_version"
        ] = "MLX 0.32.1"
        changed_runtime = compare_reports(baseline, candidate)
        self.assertFalse(changed_runtime["compatibility"]["compatible"])
        self.assertIn(
            "manifest_mismatch_runtime_accelerator_runtime_version",
            changed_runtime["compatibility"]["reasons"],
        )

    def test_rocm_and_cpu_saved_runs_support_comparison(self) -> None:
        for backend, accelerator, runtime_version, host_os_version in (
            NON_CUDA_RUNTIME_CASES
        ):
            with self.subTest(backend=backend):
                baseline = _saved_report((10.0, 10.0, 10.0), flag_value="1")
                candidate = _saved_report((12.0, 12.0, 12.0), flag_value="8")
                for report in (baseline, candidate):
                    _set_platform_runtime(
                        report,
                        backend=backend,
                        accelerator=accelerator,
                        runtime_version=runtime_version,
                        host_os_version=host_os_version,
                    )

                comparison = compare_reports(baseline, candidate)
                self.assertEqual(comparison["status"], "complete")
                self.assertTrue(comparison["compatibility"]["compatible"])
                self.assertTrue(comparison["decision_eligible"])
                self.assertEqual(
                    comparison["overall_outcome"],
                    "candidate_higher_throughput",
                )

    def test_saved_runtime_alias_mismatch_and_malformed_hashes_fail_closed(
        self,
    ) -> None:
        cases = (
            (
                "label_mismatch",
                "gpu",
                "contradictory accelerator label",
                "runtime_aliases_do_not_reconcile",
            ),
            (
                "fingerprint_mismatch",
                "gpu_fingerprint_sha256",
                "a" * 64,
                "runtime_aliases_do_not_reconcile",
            ),
            (
                "supplied_flag_wrong_type",
                "gpu_fingerprint_supplied",
                "true",
                "runtime_aliases_do_not_reconcile",
            ),
            (
                "legacy_fingerprint_malformed",
                "gpu_fingerprint_sha256",
                "not-a-digest",
                "invalid_manifest_digest",
            ),
            (
                "generic_fingerprint_malformed",
                "accelerator_fingerprint_sha256",
                "not-a-digest",
                "invalid_manifest_digest",
            ),
        )
        for name, field, value, reason in cases:
            with self.subTest(case=name):
                baseline = _saved_report((10.0, 10.0, 10.0), flag_value="1")
                candidate = _saved_report((12.0, 12.0, 12.0), flag_value="8")
                for report in (baseline, candidate):
                    _set_metal_runtime(report)
                baseline["manifest"]["runtime"][field] = value  # type: ignore[index]

                comparison = compare_reports(baseline, candidate)
                self.assertEqual(comparison["status"], "incompatible")
                self.assertFalse(comparison["compatibility"]["compatible"])
                self.assertFalse(comparison["decision_eligible"])
                self.assertIn(
                    f"baseline_{reason}", comparison["compatibility"]["reasons"]
                )

    def test_saved_hostile_runtime_metadata_fails_closed_without_echo(self) -> None:
        hostile_cases = (
            (
                ("gpu", "accelerator"),
                "https://private-accelerator.example/credential",
            ),
            (("accelerator_runtime_version",), "Bearer private-runtime-token"),
            (("accelerator_runtime_version",), "token=private-runtime-token"),
            (("accelerator_runtime_version",), "access_token=private-runtime-token"),
            (("host_os_version",), "authorization: private-host-token"),
            (("host_os_version",), "ghp_1234567890abcdef"),
            (("host_os_version",), "github_pat_1234567890abcdef"),
            (("host_os_version",), "AKIA" + "A" * 16),
            (("host_os_version",), "xoxb-1234567890abcdef"),
            (("accelerator_runtime_version",), "runtime ghp_abcdefghijk"),
            (
                ("accelerator_runtime_version",),
                "runtime eyJabc.eyJdef.signature",
            ),
            (("host_os_version",), "device alice:hunter2@private.example"),
            (
                ("software_environment_digest",),
                "env-ghp_abcdefghijk@sha256:" + "a" * 64,
            ),
            (
                ("software_environment_digest",),
                "env-eyJabc.eyJdef.signature@sha256:" + "a" * 64,
            ),
            (
                ("software_environment_digest",),
                "https://private-environment.example/secret@sha256:" + "a" * 64,
            ),
            (
                ("software_environment_digest",),
                "user:password@private-registry@sha256:" + "a" * 64,
            ),
            (
                ("software_environment_digest",),
                "/Users/example/env@sha256:" + "a" * 64,
            ),
            (
                ("software_environment_digest",),
                "private-environment\x1bcontrol@sha256:" + "a" * 64,
            ),
            (
                ("software_environment_digest",),
                "user:password@sha256:" + "a" * 64,
            ),
            (
                ("software_environment_digest",),
                "file:/Users/example/env@sha256:" + "a" * 64,
            ),
        )
        for fields, hostile in hostile_cases:
            with self.subTest(fields=fields):
                baseline = _saved_report((10.0, 10.0, 10.0), flag_value="1")
                candidate = _saved_report((12.0, 12.0, 12.0), flag_value="8")
                for report in (baseline, candidate):
                    _set_metal_runtime(report)
                runtime = baseline["manifest"]["runtime"]  # type: ignore[index]
                for field in fields:
                    runtime[field] = hostile  # type: ignore[index]

                comparison = compare_reports(baseline, candidate)
                self.assertEqual(comparison["status"], "incompatible")
                self.assertFalse(comparison["compatibility"]["compatible"])
                self.assertIn(
                    "baseline_unsafe_runtime_metadata",
                    comparison["compatibility"]["reasons"],
                )
                self.assertNotIn(hostile, json.dumps(comparison, sort_keys=True))

    def test_saved_unknown_fingerprint_hash_cannot_be_forged_as_supplied(
        self,
    ) -> None:
        baseline = _saved_report((10.0, 10.0, 10.0), flag_value="1")
        candidate = _saved_report((12.0, 12.0, 12.0), flag_value="8")
        for report in (baseline, candidate):
            _set_metal_runtime(report)
        forged = hashlib.sha256(b"unknown").hexdigest()
        runtime = baseline["manifest"]["runtime"]  # type: ignore[index]
        runtime["gpu_fingerprint_sha256"] = forged  # type: ignore[index]
        runtime["accelerator_fingerprint_sha256"] = forged  # type: ignore[index]
        runtime["gpu_fingerprint_supplied"] = True  # type: ignore[index]
        runtime["accelerator_fingerprint_supplied"] = True  # type: ignore[index]

        comparison = compare_reports(baseline, candidate)
        self.assertEqual(comparison["status"], "incompatible")
        self.assertFalse(comparison["compatibility"]["compatible"])
        self.assertFalse(comparison["decision_eligible"])
        self.assertIn(
            "baseline_accelerator_fingerprint_not_supplied",
            comparison["compatibility"]["reasons"],
        )
        self.assertNotIn(forged, json.dumps(comparison, sort_keys=True))

    def test_legacy_mutable_image_remains_descriptively_comparable(self) -> None:
        baseline = _saved_report((10.0, 10.0, 10.0), flag_value="1")
        candidate = _saved_report((12.0, 12.0, 12.0), flag_value="8")
        for report in (baseline, candidate):
            report["manifest"]["runtime"]["image_digest"] = (  # type: ignore[index]
                "example/image:latest"
            )

        comparison = compare_reports(baseline, candidate)
        self.assertEqual(comparison["status"], "complete")
        self.assertTrue(comparison["compatibility"]["compatible"])
        self.assertFalse(comparison["decision_eligible"])
        self.assertEqual(comparison["decision_state"], "inconclusive")
        self.assertIsNone(comparison["overall_outcome"])
        self.assertEqual(
            comparison["descriptive_statistical_outcome"],
            "candidate_higher_throughput",
        )
        self.assertIn(
            "immutable_image_digest_required",
            comparison["decision_ineligible_reasons"],
        )

    def test_legacy_oci_and_non_sha256_references_keep_compatibility(self) -> None:
        cases = (
            ("ubuntu:24.04@sha256:" + "b" * 64, True),
            ("example/image@sha512:" + "b" * 128, False),
        )
        for image_reference, decision_eligible in cases:
            with self.subTest(image_reference=image_reference[:32]):
                baseline = _saved_report(
                    (10.0, 10.0, 10.0), flag_value="1"
                )
                candidate = _saved_report(
                    (12.0, 12.0, 12.0), flag_value="8"
                )
                for report in (baseline, candidate):
                    report["manifest"]["runtime"][  # type: ignore[index]
                        "image_digest"
                    ] = image_reference

                comparison = compare_reports(baseline, candidate)
                self.assertEqual(comparison["status"], "complete")
                self.assertTrue(comparison["compatibility"]["compatible"])
                self.assertEqual(
                    comparison["decision_eligible"], decision_eligible
                )
                if not decision_eligible:
                    self.assertIn(
                        "immutable_image_digest_required",
                        comparison["decision_ineligible_reasons"],
                    )

    def test_mixed_runtime_manifest_versions_are_explicitly_incompatible(
        self,
    ) -> None:
        baseline = _saved_report((10.0, 10.0, 10.0), flag_value="1")
        candidate = _saved_report((12.0, 12.0, 12.0), flag_value="8")
        candidate_manifest = candidate["manifest"]  # type: ignore[index]
        candidate_manifest["manifest_version"] = "1.1"  # type: ignore[index]
        runtime = candidate_manifest["runtime"]  # type: ignore[index]
        runtime.update(  # type: ignore[union-attr]
            accelerator_backend="cuda",
            accelerator=runtime["gpu"],  # type: ignore[index]
            accelerator_fingerprint_sha256=runtime[  # type: ignore[index]
                "gpu_fingerprint_sha256"
            ],
            accelerator_fingerprint_supplied=runtime[  # type: ignore[index]
                "gpu_fingerprint_supplied"
            ],
            accelerator_runtime_version=runtime["cuda_version"],  # type: ignore[index]
            host_os_version="unknown",
            software_environment_digest=runtime["image_digest"],  # type: ignore[index]
        )

        comparison = compare_reports(baseline, candidate)
        self.assertEqual(comparison["status"], "incompatible")
        self.assertFalse(comparison["compatibility"]["compatible"])
        self.assertIn(
            "manifest_mismatch_manifest_version",
            comparison["compatibility"]["reasons"],
        )

    async def test_saved_comparison_requires_identical_sealed_request_profiles(
        self,
    ) -> None:
        baseline = _saved_report((10.0, 10.0, 10.0), flag_value="1")
        candidate = _saved_report((12.0, 12.0, 12.0), flag_value="8")
        fields = (("enable_thinking", False), ("top_k", 1))
        _set_request_profile(baseline, fields)
        _set_request_profile(candidate, tuple(reversed(fields)))

        comparison = compare_reports(baseline, candidate)
        self.assertTrue(comparison["compatibility"]["compatible"])
        self.assertTrue(comparison["decision_eligible"])

        legacy_candidate = _saved_report(
            (12.0, 12.0, 12.0), flag_value="8"
        )
        sealed_vs_legacy = compare_reports(baseline, legacy_candidate)
        self.assertFalse(sealed_vs_legacy["compatibility"]["compatible"])
        self.assertIn(
            "manifest_missing_request_profile_version",
            sealed_vs_legacy["compatibility"]["reasons"],
        )

        _set_request_profile(candidate, (("enable_thinking", True), ("top_k", 1)))
        mismatch = compare_reports(baseline, candidate)
        self.assertFalse(mismatch["compatibility"]["compatible"])
        self.assertIn(
            "manifest_mismatch_request_extensions",
            mismatch["compatibility"]["reasons"],
        )

        candidate["manifest"]["request"]["extensions"][  # type: ignore[index]
            "enable_thinking"
        ] = False
        tampered = compare_reports(baseline, candidate)
        self.assertEqual(tampered["status"], "incompatible")
        self.assertIn(
            "candidate_invalid_request_profile_hash",
            tampered["compatibility"]["reasons"],
        )

    async def test_open_loop_uses_rate_shape_and_respects_in_flight_ceiling(
        self,
    ) -> None:
        async def handler(_: httpx.Request) -> httpx.Response:
            await asyncio.sleep(0.002)
            return _completion()

        config = _run_config(
            conditions=(LoadCondition("open_loop", 100.0, 8),),
            # Keep enough in-flight capacity to test transport scheduling
            # without making validity depend on a sub-100 ms host-clock sample.
            # Target-rate reconciliation is tested deterministically below.
            requests_per_block=8,
        )
        report = await run_native(
            config,
            PROMPTS,
            WARMUPS,
            transport=httpx.MockTransport(handler),
        )
        condition = report["conditions"][0]
        self.assertEqual(condition["condition"]["kind"], "open_loop")
        self.assertEqual(condition["request_counts"]["valid"], 8)
        self.assertEqual(condition["target_offered_request_rate"], 100.0)
        self.assertIsNotNone(condition["achieved_offered_request_rate"])
        self.assertIsNotNone(condition["offered_rate_relative_error"])
        self.assertEqual(
            condition["metrics"]["e2e_latency_ms"]["p95_repeated_block_ci"]["method"],
            "student_t_blocks",
        )
        self.assertLessEqual(report["run_totals"]["peak_in_flight"], 8)
        self.assertEqual(report["best_tested"]["field"], "best_tested_request_rate")

    def test_open_loop_exact_offered_rate_is_reconciled_deterministically(
        self,
    ) -> None:
        config = _run_config(
            conditions=(LoadCondition("open_loop", 100.0, 8),),
            requests_per_block=8,
        )
        results = [
            RequestResult(
                status_code=200,
                e2e_seconds=0.01,
                completion_tokens=1,
                prompt_tokens=1,
                finish_reason="stop",
            )
            for _ in range(8)
        ]
        report = _block_report(
            BlockOutcome(
                results=results,
                wall_seconds=0.08,
                complete=True,
                scheduler_lag_seconds=[0.0] * 8,
                offered_requests=8,
                peak_in_flight=8,
                target_offered_request_rate=100.0,
                launch_window_seconds=0.07,
            ),
            config,
            index=0,
            seed=1,
        )

        self.assertAlmostEqual(report["achieved_offered_request_rate"], 100.0)
        self.assertAlmostEqual(report["offered_rate_relative_error"], 0.0)
        self.assertEqual(report["scheduler_lag_interval_ratio_p95"], 0.0)
        self.assertTrue(report["open_loop_target_achieved"])

    async def test_open_loop_prestart_backpressure_reconciles_cancellation(
        self,
    ) -> None:
        async def handler(_: httpx.Request) -> httpx.Response:
            return _completion()

        config = _run_config(
            conditions=(LoadCondition("open_loop", 100.0, 2),),
            requests_per_block=8,
        )
        shared_budget = RunBudget(config)
        real_create_task = asyncio.create_task
        real_perf_counter = time.perf_counter
        clock_offset = 0.0
        task_count = 0

        def perf_counter() -> float:
            return real_perf_counter() + clock_offset

        def create_task(coro: Any) -> asyncio.Task[Any]:
            nonlocal clock_offset, task_count
            task = real_create_task(coro)
            task_count += 1
            if task_count == 1:
                # Model host preemption before the first reserved request gets
                # an event-loop turn. Three arrivals are then due at once.
                clock_offset = 0.035
            return task

        with (
            patch(
                "throttle.benchmark.time.perf_counter",
                side_effect=perf_counter,
            ),
            patch(
                "throttle.benchmark.asyncio.create_task",
                side_effect=create_task,
            ),
        ):
            report = await run_native(
                config,
                PROMPTS,
                WARMUPS,
                transport=httpx.MockTransport(handler),
                shared_budget=shared_budget,
            )

        condition = report["conditions"][0]
        self.assertEqual(report["status"], "stopped")
        self.assertEqual(report["stop_reason"], "open_loop_backpressure")
        self.assertFalse(report["decision_eligible"])
        self.assertEqual(condition["request_counts"]["valid"], 0)
        self.assertEqual(condition["blocks"][0]["offered_requests"], 2)
        self.assertIn(
            "open_loop_backpressure",
            condition["blocks"][0]["invalid_reasons"],
        )
        self.assertEqual(report["run_totals"]["requests_started"], 2)
        self.assertEqual(report["run_totals"]["requests_completed"], 0)
        self.assertEqual(report["run_totals"]["requests_cancelled"], 2)
        self.assertEqual(report["run_totals"]["requests_in_flight"], 0)
        self.assertEqual(report["run_totals"]["peak_in_flight"], 2)
        self.assertEqual(shared_budget.requests_cancelled, 2)
        self.assertEqual(shared_budget.in_flight, 0)

    async def test_open_loop_missed_offered_rate_is_not_decision_grade(self) -> None:
        config = _run_config(
            mode="benchmark",
            conditions=(LoadCondition("open_loop", 100.0, 8),),
            requests_per_block=67,
        )
        results = [
            RequestResult(
                status_code=200,
                e2e_seconds=0.01,
                completion_tokens=1,
                prompt_tokens=1,
                finish_reason="stop",
            )
            for _ in range(67)
        ]
        outcomes = [
            BlockOutcome(
                results=copy.deepcopy(results),
                wall_seconds=1.5,
                complete=True,
                scheduler_lag_seconds=[0.0] * 67,
                offered_requests=67,
                peak_in_flight=8,
                target_offered_request_rate=100.0,
                launch_window_seconds=66 / 50.0,
            )
            for _ in range(3)
        ]

        async def next_outcome(*_: object, **__: object) -> BlockOutcome:
            return outcomes.pop(0)

        with patch(
            "throttle.benchmark._run_open_block",
            side_effect=next_outcome,
        ):
            report = await run_native(
                config,
                PROMPTS,
                WARMUPS,
                transport=httpx.MockTransport(lambda _: _completion()),
            )

        condition = report["conditions"][0]
        self.assertTrue(condition["valid"])
        self.assertFalse(condition["decision_grade"])
        self.assertFalse(condition["open_loop_target_achieved"])
        self.assertIn(
            "open_loop_target_rate_not_achieved",
            condition["decision_ineligible_reasons"],
        )

    async def test_workload_hash_is_stable_without_persisting_prompt(self) -> None:
        same = tuple(tuple(dict(message) for message in prompt) for prompt in PROMPTS)
        changed = (({"role": "user", "content": "different private prompt"},),)
        digest = canonical_workload_hash(PROMPTS)
        self.assertEqual(digest, canonical_workload_hash(same))
        self.assertNotEqual(digest, canonical_workload_hash(changed))
        self.assertNotIn("private measured prompt", digest)

    async def test_saved_comparison_uses_paired_ci_and_cost_compatibility(self) -> None:
        baseline = _saved_report((10.0, 10.0, 10.0), flag_value="1")
        candidate = _saved_report((12.0, 12.0, 12.0), flag_value="8")
        comparison = compare_reports(baseline, candidate)
        self.assertEqual(comparison["status"], "complete")
        self.assertEqual(comparison["decision_state"], "supported")
        self.assertEqual(comparison["overall_outcome"], "candidate_higher_throughput")
        self.assertTrue(comparison["decision_eligible"])
        self.assertEqual(
            comparison["attribution"]["state"],
            "controlled_difference_declared",
        )
        baseline_condition = baseline["conditions"][0]  # type: ignore[index]
        expected_cost_per_million = (
            baseline["manifest"]["cost"]["total_hourly_rate"]  # type: ignore[index]
            * baseline_condition["measured_wall_seconds"]  # type: ignore[index]
            / 3_600.0
            / baseline_condition["metrics"]["completion_tokens"]  # type: ignore[index]
            * 1_000_000.0
        )
        self.assertAlmostEqual(
            baseline_condition["metrics"]["cost_per_million_output_tokens"],  # type: ignore[index]
            expected_cost_per_million,
        )
        self.assertTrue(comparison["conditions"][0]["cost_delta"]["available"])

        mixed_cost = _saved_report(
            (12.0, 12.0, 12.0),
            cost_kind="serverless_active_seconds",
            flag_value="8",
        )
        cost_guarded = compare_reports(baseline, mixed_cost)
        self.assertFalse(cost_guarded["conditions"][0]["cost_delta"]["available"])
        self.assertEqual(
            cost_guarded["conditions"][0]["cost_delta"]["reason"],
            "incompatible_cost_models",
        )

    async def test_saved_open_loop_evidence_round_trips_through_compare(self) -> None:
        baseline = _saved_report((10.0, 10.0, 10.0), flag_value="1")
        candidate = _saved_report((12.0, 12.0, 12.0), flag_value="8")
        descriptor = LoadCondition("open_loop", 10.0, 8).public_dict()
        for report in (baseline, candidate):
            report["manifest"]["traffic"]["conditions"] = [descriptor]  # type: ignore[index]
            condition = report["conditions"][0]  # type: ignore[index]
            condition["condition"] = descriptor
            condition["target_offered_request_rate"] = 10.0
            condition["achieved_offered_request_rate"] = 10.0
            condition["offered_rate_relative_error"] = 0.0
            condition["open_loop_target_achieved"] = True
            for block in condition["blocks"]:
                block["target_offered_request_rate"] = 10.0
                block["launch_window_seconds"] = 6.6
                block["achieved_offered_request_rate"] = 10.0
                block["offered_rate_relative_error"] = 0.0
                block["scheduler_lag_interval_ratio_p95"] = 0.0
                block["open_loop_target_achieved"] = True

        comparison = compare_reports(baseline, candidate)
        self.assertTrue(comparison["compatibility"]["compatible"])
        self.assertEqual(comparison["status"], "complete")
        self.assertTrue(comparison["decision_eligible"])

        baseline["conditions"][0]["blocks"][0][  # type: ignore[index]
            "achieved_offered_request_rate"
        ] = 9.0
        tampered = compare_reports(baseline, candidate)
        self.assertFalse(tampered["compatibility"]["compatible"])
        self.assertIn(
            "baseline_open_loop_rate_evidence_does_not_reconcile",
            tampered["compatibility"]["reasons"],
        )

    async def test_nonstream_saved_compare_is_descriptive_but_not_decision_grade(
        self,
    ) -> None:
        baseline = _saved_report((10.0, 10.0, 10.0), flag_value="1")
        candidate = _saved_report((12.0, 12.0, 12.0), flag_value="8")
        for report in (baseline, candidate):
            report["manifest"]["request"]["stream"] = False  # type: ignore[index]
            _set_nonstream_ttft_unavailable(report)

        comparison = compare_reports(baseline, candidate)
        self.assertTrue(comparison["compatibility"]["compatible"])
        self.assertEqual(comparison["status"], "complete")
        self.assertEqual(comparison["conditions"][0]["state"], "supported")
        self.assertEqual(
            comparison["conditions"][0]["outcome"],
            "candidate_higher_throughput",
        )
        self.assertFalse(comparison["decision_eligible"])
        self.assertEqual(comparison["decision_state"], "inconclusive")
        self.assertIsNone(comparison["overall_outcome"])
        self.assertEqual(
            comparison["descriptive_statistical_outcome"],
            "candidate_higher_throughput",
        )
        self.assertIn(
            "streaming_required_for_decision_grade",
            comparison["decision_ineligible_reasons"],
        )

    async def test_saved_comparison_suppresses_output_length_mismatch(self) -> None:
        baseline = _saved_report((10.0, 10.0, 10.0), completion_tokens=300)
        candidate = _saved_report(
            (12.0, 12.0, 12.0),
            completion_tokens=320,
            flag_value="8",
        )
        comparison = compare_reports(baseline, candidate)
        condition = comparison["conditions"][0]
        self.assertFalse(condition["available"])
        self.assertEqual(
            condition["reason"],
            "completion_tokens_outside_5_percent_tolerance",
        )
        self.assertEqual(comparison["decision_state"], "inconclusive")
        self.assertFalse(comparison["decision_eligible"])

    async def test_opposite_block_token_mismatches_cannot_cancel_in_aggregate(
        self,
    ) -> None:
        baseline = _saved_report((10.0, 10.0, 10.0), flag_value="1")
        candidate = _saved_report((12.0, 12.0, 12.0), flag_value="8")
        _set_block_completion_tokens(baseline, (80, 100, 120))
        _set_block_completion_tokens(candidate, (120, 100, 80))

        comparison = compare_reports(baseline, candidate)
        self.assertTrue(comparison["compatibility"]["compatible"])
        self.assertEqual(comparison["status"], "complete")
        self.assertFalse(comparison["decision_eligible"])
        self.assertEqual(comparison["decision_state"], "inconclusive")
        condition = comparison["conditions"][0]
        self.assertEqual(condition["completion_token_relative_difference"], 0.0)
        self.assertGreater(
            condition["maximum_block_completion_token_relative_difference"],
            condition["completion_token_tolerance"],
        )
        self.assertEqual(
            condition["reason"],
            "one_or_more_block_pairs_outside_5_percent_completion_token_tolerance",
        )

    async def test_saved_compare_rejects_false_or_missing_disjoint_warmup_evidence(
        self,
    ) -> None:
        for evidence in (False, "missing"):
            with self.subTest(evidence=evidence):
                baseline = _saved_report((10.0, 10.0, 10.0), flag_value="1")
                candidate = _saved_report((12.0, 12.0, 12.0), flag_value="8")
                workload = baseline["manifest"]["workload"]  # type: ignore[index]
                if evidence == "missing":
                    workload.pop("warmup_prompts_disjoint")  # type: ignore[union-attr]
                else:
                    workload["warmup_prompts_disjoint"] = False  # type: ignore[index]

                comparison = compare_reports(baseline, candidate)
                self.assertEqual(comparison["status"], "incompatible")
                self.assertFalse(comparison["compatibility"]["compatible"])
                self.assertIn(
                    "baseline_warmup_workload_not_separate",
                    comparison["compatibility"]["reasons"],
                )

    async def test_every_closed_loop_block_must_reach_declared_concurrency(
        self,
    ) -> None:
        baseline = _saved_report((10.0, 10.0, 10.0), flag_value="1")
        candidate = _saved_report((12.0, 12.0, 12.0), flag_value="8")
        # Other blocks still reach eight, so the aggregate maximum alone looks valid.
        baseline["conditions"][0]["blocks"][1]["observed_peak_in_flight"] = 7  # type: ignore[index]

        comparison = compare_reports(baseline, candidate)
        self.assertEqual(comparison["status"], "incompatible")
        self.assertFalse(comparison["compatibility"]["compatible"])
        self.assertIn(
            "baseline_closed_loop_target_concurrency_not_achieved_in_every_block",
            comparison["compatibility"]["reasons"],
        )

    async def test_native_aggregation_checks_concurrency_in_each_block(self) -> None:
        config = _run_config(
            mode="benchmark",
            conditions=(LoadCondition("closed_loop", 8.0, 8),),
            requests_per_block=67,
        )
        results = [
            RequestResult(
                status_code=200,
                e2e_seconds=0.01,
                completion_tokens=1,
                prompt_tokens=1,
                finish_reason="stop",
            )
            for _ in range(67)
        ]
        outcomes = [
            BlockOutcome(
                results=copy.deepcopy(results),
                wall_seconds=1.0,
                complete=True,
                offered_requests=67,
                peak_in_flight=peak,
            )
            for peak in (8, 7, 8)
        ]

        async def next_outcome(*_: object, **__: object) -> BlockOutcome:
            return outcomes.pop(0)

        with patch(
            "throttle.benchmark._run_closed_block",
            side_effect=next_outcome,
        ):
            report = await run_native(
                config,
                PROMPTS,
                WARMUPS,
                transport=httpx.MockTransport(lambda _: _completion()),
            )

        condition = report["conditions"][0]
        self.assertTrue(condition["valid"])
        self.assertEqual(condition["observed_peak_in_flight"], 8)
        self.assertEqual(
            [block["observed_peak_in_flight"] for block in condition["blocks"]],
            [8, 7, 8],
        )
        self.assertFalse(condition["decision_grade"])
        self.assertIn(
            "closed_loop_target_concurrency_not_observed",
            condition["decision_ineligible_reasons"],
        )
        self.assertFalse(report["decision_eligible"])

    async def test_small_scientific_notation_open_loop_id_is_canonical_json(
        self,
    ) -> None:
        condition = LoadCondition("open_loop", 0.00001, 8)
        self.assertEqual(condition.condition_id, "open_loop:1e-05")
        descriptor = condition.public_dict()
        self.assertEqual(
            json.loads(json.dumps(descriptor, allow_nan=False))["id"],
            condition.condition_id,
        )

    async def test_hostile_engine_flag_name_or_value_never_enters_comparison(
        self,
    ) -> None:
        hostile_cases = (
            ("private_endpoint_secret", "opaque", "private_endpoint_secret"),
            (
                "served_model_name",
                "https://private-flag.example/credential",
                "private-flag.example",
            ),
        )
        for name, value, forbidden in hostile_cases:
            with self.subTest(name=name):
                baseline = _saved_report((10.0, 10.0, 10.0), flag_value="1")
                candidate = _saved_report((12.0, 12.0, 12.0), flag_value="8")
                baseline["manifest"]["engine"]["effective_flags"][name] = value  # type: ignore[index]

                comparison = compare_reports(baseline, candidate)
                serialized = json.dumps(comparison, allow_nan=False, sort_keys=True)
                self.assertEqual(comparison["status"], "incompatible")
                self.assertIn(
                    "baseline_invalid_engine_flag_manifest",
                    comparison["compatibility"]["reasons"],
                )
                self.assertNotIn(name, serialized)
                self.assertNotIn(value, serialized)
                self.assertNotIn(forbidden, serialized)

    async def test_skeletal_hand_edited_reports_cannot_claim_decision_grade(
        self,
    ) -> None:
        skeletal = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "mode": "benchmark",
            "status": "complete",
            "decision_eligible": True,
            "manifest": {"model": {"id": "model-a"}},
            "conditions": [
                {
                    "valid": True,
                    "decision_grade": True,
                    "metrics": {
                        "completion_tokens": 999_999,
                        "output_tokens_per_second": 999_999.0,
                    },
                }
            ],
        }
        comparison = compare_reports(skeletal, copy.deepcopy(skeletal))
        self.assertEqual(comparison["status"], "incompatible")
        self.assertFalse(comparison["compatibility"]["compatible"])
        self.assertFalse(comparison["decision_eligible"])
        self.assertIn(
            "baseline_missing_or_invalid_manifest",
            comparison["compatibility"]["reasons"],
        )

    async def test_large_max_num_seqs_change_is_unattributable_at_load_eight(
        self,
    ) -> None:
        baseline = _saved_report((10.0, 10.0, 10.0), flag_value="256")
        candidate = _saved_report((12.0, 12.0, 12.0), flag_value="2048")
        comparison = compare_reports(baseline, candidate)
        self.assertTrue(comparison["compatibility"]["compatible"])
        self.assertEqual(comparison["decision_state"], "inconclusive")
        self.assertIsNone(comparison["overall_outcome"])
        self.assertEqual(
            comparison["descriptive_statistical_outcome"],
            "candidate_higher_throughput",
        )
        self.assertEqual(comparison["attribution"]["state"], "unattributable")
        self.assertEqual(
            comparison["attribution"]["reason"],
            "max_num_seqs_change_not_exercised_by_load",
        )
        self.assertFalse(comparison["decision_eligible"])
        self.assertIn(
            "max_num_seqs_change_not_exercised_by_load",
            comparison["decision_ineligible_reasons"],
        )

    async def test_nonsecret_token_engine_flag_survives_saved_preflight(self) -> None:
        baseline = _saved_report((10.0, 10.0, 10.0), flag_value="1")
        candidate = _saved_report((12.0, 12.0, 12.0), flag_value="8")
        for report in (baseline, candidate):
            report["manifest"]["engine"]["effective_flags"][  # type: ignore[index]
                "max-num-batched-tokens"
            ] = "4096"
        comparison = compare_reports(baseline, candidate)
        self.assertTrue(comparison["compatibility"]["compatible"])
        self.assertNotIn(
            "invalid_engine_flag_manifest", comparison["compatibility"]["reasons"]
        )

    async def test_internal_count_and_throughput_tampering_is_rejected(self) -> None:
        for tamper, expected_reason in (
            ("counts", "baseline_condition_request_count_maps_invalid"),
            ("throughput", "baseline_condition_throughput_does_not_reconcile"),
        ):
            with self.subTest(tamper=tamper):
                baseline = _saved_report((10.0, 10.0, 10.0), flag_value="1")
                candidate = _saved_report((12.0, 12.0, 12.0), flag_value="8")
                condition = baseline["conditions"][0]  # type: ignore[index]
                if tamper == "counts":
                    condition["request_counts"] = {  # type: ignore[index]
                        **condition["request_counts"],  # type: ignore[index]
                        "attempted": 200,
                        "valid": 200,
                    }
                else:
                    condition["metrics"]["output_tokens_per_second"] = 999.0  # type: ignore[index]

                comparison = compare_reports(baseline, candidate)
                self.assertEqual(comparison["status"], "incompatible")
                self.assertFalse(comparison["compatibility"]["compatible"])
                self.assertIn(
                    expected_reason,
                    comparison["compatibility"]["reasons"],
                )

    async def test_completion_token_totals_cannot_fall_below_valid_responses(
        self,
    ) -> None:
        cases = (
            ("condition", "baseline_condition_token_evidence_invalid"),
            ("block", "baseline_invalid_block_token_evidence"),
        )
        for level, expected_reason in cases:
            with self.subTest(level=level):
                baseline = _saved_report((10.0, 10.0, 10.0), flag_value="1")
                candidate = _saved_report((12.0, 12.0, 12.0), flag_value="8")
                condition = baseline["conditions"][0]  # type: ignore[index]
                if level == "condition":
                    condition["metrics"]["completion_tokens"] = 200  # type: ignore[index]
                else:
                    condition["blocks"][0]["metrics"][  # type: ignore[index]
                        "completion_tokens"
                    ] = 66

                comparison = compare_reports(baseline, candidate)
                self.assertEqual(comparison["status"], "incompatible")
                self.assertFalse(comparison["compatibility"]["compatible"])
                self.assertIn(
                    expected_reason,
                    comparison["compatibility"]["reasons"],
                )

    async def test_saved_strict_completion_count_maps_are_reconciled(self) -> None:
        cases = (
            ("condition_status", "baseline_condition_request_count_maps_invalid"),
            ("condition_error", "baseline_condition_request_count_maps_invalid"),
            ("condition_finish", "baseline_condition_request_count_maps_invalid"),
            ("warmup_status", "baseline_warmup_request_count_maps_invalid"),
            ("block_status", "baseline_invalid_block_count_maps"),
            ("block_error", "baseline_invalid_block_count_maps"),
            ("block_finish", "baseline_invalid_block_count_maps"),
            (
                "aggregate_finish",
                "baseline_condition_finish_reason_counts_do_not_reconcile",
            ),
            ("block_valid_metric", "baseline_block_metric_counts_do_not_reconcile"),
            ("block_error_metric", "baseline_block_metric_counts_do_not_reconcile"),
        )
        for tamper, expected_reason in cases:
            with self.subTest(tamper=tamper):
                baseline = _saved_report((10.0, 10.0, 10.0), flag_value="1")
                candidate = _saved_report((12.0, 12.0, 12.0), flag_value="8")
                condition = baseline["conditions"][0]  # type: ignore[index]
                counts = condition["request_counts"]
                warmup = condition["warmup"]
                block = condition["blocks"][0]
                block_counts = block["request_counts"]
                block_metrics = block["metrics"]
                if tamper == "condition_status":
                    counts["status_counts"] = {"404": 201}
                elif tamper == "condition_error":
                    counts["error_counts"] = {"non_200_response": 1}
                elif tamper == "condition_finish":
                    counts["finish_reason_counts"] = {"stop": 200}
                elif tamper == "warmup_status":
                    warmup["status_counts"] = {"404": 3}
                elif tamper == "block_status":
                    block_counts["status_counts"] = {"404": 67}
                elif tamper == "block_error":
                    block_counts["error_counts"] = {"transport_error": 1}
                elif tamper == "block_finish":
                    block_counts["finish_reason_counts"] = {"stop": 66}
                elif tamper == "aggregate_finish":
                    counts["finish_reason_counts"] = {"stop": 200, "length": 1}
                elif tamper == "block_valid_metric":
                    block_metrics["valid_response_count"] = 66
                elif tamper == "block_error_metric":
                    block_metrics["error_rate"] = 0.01

                comparison = compare_reports(baseline, candidate)
                self.assertEqual(comparison["status"], "incompatible")
                self.assertFalse(comparison["compatibility"]["compatible"])
                self.assertIn(
                    expected_reason,
                    comparison["compatibility"]["reasons"],
                )

    async def test_saved_preflight_rejects_cross_field_evidence_tampering(self) -> None:
        cases = (
            ("missing_safety_override", "baseline_invalid_safety_manifest"),
            ("ambient_proxy", "baseline_invalid_safety_manifest"),
            ("cost_ceiling", "baseline_invalid_cost_manifest"),
            ("condition_id", "baseline_condition_id_does_not_match_descriptor"),
            ("block_index", "baseline_block_indexes_not_sequential"),
            (
                "request_bound",
                "baseline_block_request_count_does_not_match_manifest",
            ),
            ("duration_bound", "baseline_block_duration_does_not_meet_manifest"),
            (
                "offered_requests",
                "baseline_block_offered_requests_do_not_reconcile",
            ),
            ("run_totals", "baseline_run_totals_do_not_reconcile"),
            (
                "safety_limit",
                "baseline_run_totals_violate_safety_or_wall_time",
            ),
            ("complete_stop_reason", "baseline_complete_report_has_stop_reason"),
            ("timestamps", "baseline_run_timestamps_do_not_reconcile"),
            ("cost_summary", "baseline_cost_summary_does_not_reconcile"),
            (
                "condition_cost",
                "baseline_condition_cost_evidence_does_not_reconcile",
            ),
        )
        for tamper, expected_reason in cases:
            with self.subTest(tamper=tamper):
                baseline = _saved_report((10.0, 10.0, 10.0), flag_value="1")
                candidate = _saved_report((12.0, 12.0, 12.0), flag_value="8")
                manifest = baseline["manifest"]  # type: ignore[assignment]
                condition = baseline["conditions"][0]  # type: ignore[index]
                if tamper == "missing_safety_override":
                    manifest["safety"]["overrides"].pop(  # type: ignore[index]
                        "guidellm_validation_gaps_acknowledged"
                    )
                elif tamper == "ambient_proxy":
                    manifest["safety"]["ambient_proxy_environment_used"] = True  # type: ignore[index]
                elif tamper == "cost_ceiling":
                    manifest["cost"]["pre_run_upper_bound"] = 99.0  # type: ignore[index]
                elif tamper == "condition_id":
                    condition["condition"]["id"] = "closed_loop:08"  # type: ignore[index]
                    manifest["traffic"]["conditions"][0]["id"] = "closed_loop:08"  # type: ignore[index]
                elif tamper == "block_index":
                    condition["blocks"][1]["block_index"] = 1  # type: ignore[index]
                elif tamper == "request_bound":
                    manifest["traffic"]["requests_per_block"] = 68  # type: ignore[index]
                elif tamper == "duration_bound":
                    manifest["traffic"]["requests_per_block"] = None  # type: ignore[index]
                    manifest["traffic"]["block_duration_seconds"] = 100.0  # type: ignore[index]
                elif tamper == "offered_requests":
                    condition["blocks"][0]["offered_requests"] = 66  # type: ignore[index]
                elif tamper == "run_totals":
                    baseline["run_totals"]["requests_completed"] = 203  # type: ignore[index]
                elif tamper == "safety_limit":
                    manifest["safety"]["limits"]["max_requests"] = 203  # type: ignore[index]
                elif tamper == "complete_stop_reason":
                    baseline["stop_reason"] = "max_errors"
                elif tamper == "timestamps":
                    baseline["completed_at"] = baseline["started_at"]
                elif tamper == "cost_summary":
                    baseline["cost_summary"]["total_cost"] += 1.0  # type: ignore[index,operator]
                elif tamper == "condition_cost":
                    condition["metrics"]["cost_per_million_output_tokens"] += 1.0  # type: ignore[index,operator]

                comparison = compare_reports(baseline, candidate)
                self.assertEqual(comparison["status"], "incompatible")
                self.assertFalse(comparison["compatibility"]["compatible"])
                self.assertIn(
                    expected_reason,
                    comparison["compatibility"]["reasons"],
                )

    async def test_saved_cost_authorization_is_recomputed_fail_closed(self) -> None:
        candidate = _saved_report((12.0, 12.0, 12.0), flag_value="8")

        over_budget = _saved_report((10.0, 10.0, 10.0), flag_value="1")
        over_budget["manifest"]["safety"]["limits"][  # type: ignore[index]
            "max_estimated_spend"
        ] = 0.001
        rejected_known = compare_reports(over_budget, candidate)
        self.assertEqual(rejected_known["status"], "incompatible")
        self.assertIn(
            "baseline_pre_run_cost_exceeded_spend_limit",
            rejected_known["compatibility"]["reasons"],
        )

        def use_unknown_cost(report: dict[str, object], *, acknowledged: bool) -> None:
            report["manifest"]["cost"] = {  # type: ignore[index]
                "kind": "unknown",
                "currency": "USD",
                "pre_run_upper_bound": None,
                "accounting_basis": "billing unknown",
            }
            report["manifest"]["safety"]["overrides"][  # type: ignore[index]
                "unknown_cost_acknowledged"
            ] = acknowledged
            report["cost_summary"] = {
                "kind": "unknown",
                "total_cost": None,
                "basis": "billing_unknown",
                "completion_tokens": report["conditions"][0]["metrics"][  # type: ignore[index]
                    "completion_tokens"
                ],
                "cost_per_million_output_tokens": None,
            }

        unknown = _saved_report((10.0, 10.0, 10.0), flag_value="1")
        use_unknown_cost(unknown, acknowledged=False)
        rejected_unknown = compare_reports(unknown, candidate)
        self.assertEqual(rejected_unknown["status"], "incompatible")
        self.assertIn(
            "baseline_unknown_cost_not_acknowledged",
            rejected_unknown["compatibility"]["reasons"],
        )

        acknowledged_baseline = _saved_report((10.0, 10.0, 10.0), flag_value="1")
        acknowledged_candidate = _saved_report((12.0, 12.0, 12.0), flag_value="8")
        use_unknown_cost(acknowledged_baseline, acknowledged=True)
        use_unknown_cost(acknowledged_candidate, acknowledged=True)
        accepted_unknown = compare_reports(
            acknowledged_baseline,
            acknowledged_candidate,
        )
        self.assertTrue(accepted_unknown["compatibility"]["compatible"])
        self.assertEqual(accepted_unknown["status"], "complete")
        self.assertFalse(accepted_unknown["conditions"][0]["cost_delta"]["available"])

    async def test_strict_json_loader_rejects_nan_and_duplicate_keys(self) -> None:
        malicious = "https://private-json-value.example/credential"
        cases = {
            "nan": '{"schema_version": "2.0", "value": NaN, "secret": "'
            + malicious
            + '"}',
            "duplicate_key": '{"schema_version": "2.0", "schema_version": "2.0", '
            '"secret": "' + malicious + '"}',
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            for name, payload in cases.items():
                with self.subTest(name=name):
                    path = Path(temp_dir) / f"{name}.json"
                    path.write_text(payload, encoding="utf-8")
                    with self.assertRaises(ComparisonInputError) as raised:
                        load_report(path)
                    self.assertEqual(
                        str(raised.exception),
                        "saved report is unreadable or not valid JSON",
                    )
                    self.assertNotIn(malicious, str(raised.exception))

    def test_saved_report_loader_rejects_special_and_oversized_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fifo = root / "report.fifo"
            os.mkfifo(fifo)
            with self.assertRaises(ComparisonInputError) as fifo_error:
                load_report(fifo)
            self.assertEqual(
                str(fifo_error.exception), "saved report must be a regular file"
            )

            oversized = root / "oversized.json"
            with oversized.open("wb") as handle:
                handle.truncate(MAX_REPORT_BYTES + 1)
            with self.assertRaises(ComparisonInputError) as size_error:
                load_report(oversized)
            self.assertEqual(
                str(size_error.exception),
                "saved report exceeds the comparison size limit",
            )

    def test_extreme_saved_numbers_fail_closed_without_nonfinite_output(self) -> None:
        baseline = _saved_report((10.0, 10.0, 10.0), flag_value="1")
        candidate = _saved_report((12.0, 12.0, 12.0), flag_value="8")
        huge = 10**400
        for report in (baseline, candidate):
            descriptor = report["conditions"][0]["condition"]  # type: ignore[index]
            descriptor.update(  # type: ignore[union-attr]
                id=f"closed_loop:{huge}",
                value=huge,
                max_in_flight=huge,
            )
            report["manifest"]["traffic"]["conditions"][0] = copy.deepcopy(  # type: ignore[index]
                descriptor
            )

        comparison = compare_reports(baseline, candidate)
        self.assertEqual(comparison["status"], "incompatible")
        self.assertFalse(comparison["compatibility"]["compatible"])
        json.dumps(comparison, allow_nan=False)
        self.assertIsNone(relative_delta_percent(1e308, 5e-324))


class GoldenProtocolTests(unittest.TestCase):
    def test_treatment_parser_and_position_builder_fail_closed_on_forged_types(
        self,
    ) -> None:
        parser_cases = (
            (None, ("max_num_seqs", "10"), "golden_treatment_flags_malformed"),
            ((), ("max_num_seqs", "10"), "golden_treatment_flags_malformed"),
            (
                ("max_num_seqs", "8", "extra"),
                ("max_num_seqs", "10"),
                "golden_treatment_flags_malformed",
            ),
            (
                (True, "8"),
                ("max_num_seqs", "10"),
                "golden_treatment_must_only_change_max_num_seqs",
            ),
            (
                ("max_num_seqs", True),
                ("max_num_seqs", "10"),
                "golden_max_num_seqs_values_must_be_canonical_positive_integers",
            ),
            (
                ("private-flag-secret", "8"),
                ("max_num_seqs", "10"),
                "golden_treatment_must_only_change_max_num_seqs",
            ),
            (
                ("max_num_seqs", "9" * 5_000),
                ("max_num_seqs", "10"),
                "golden_max_num_seqs_values_must_be_canonical_positive_integers",
            ),
        )
        for baseline, candidate, expected_code in parser_cases:
            with self.subTest(expected_code=expected_code):
                with self.assertRaises(GoldenTreatmentError) as raised:
                    parse_golden_treatment_flags(baseline, candidate)  # type: ignore[arg-type]
                self.assertEqual(raised.exception.code, expected_code)
                self.assertEqual(str(raised.exception), expected_code)
                self.assertNotIn("private-flag-secret", str(raised.exception))
                self.assertNotIn("9" * 5_000, str(raised.exception))

        for pair in ((True, 8), (1.0, 8), ("1", 8), (0, 8), (-1, 8), (1, 1)):
            with self.subTest(pair=pair):
                with self.assertRaises(GoldenTreatmentError) as raised:
                    golden_positions(*pair)  # type: ignore[arg-type]
                self.assertIn(
                    raised.exception.code,
                    {
                        "golden_max_num_seqs_values_must_be_canonical_positive_integers",
                        "golden_max_num_seqs_values_must_be_distinct",
                    },
                )

    def test_metal_runs_can_pass_the_golden_runtime_gate(self) -> None:
        reports = _golden_sequence()
        for report in reports:
            _set_metal_runtime(report)
        result = validate_golden_sequence(reports)
        self.assertTrue(result["golden_protocol_eligible"])
        self.assertEqual(result["eligibility_reasons"], [])

    def test_rocm_and_cpu_runs_can_pass_the_golden_runtime_gate(self) -> None:
        for backend, accelerator, runtime_version, host_os_version in (
            NON_CUDA_RUNTIME_CASES
        ):
            with self.subTest(backend=backend):
                reports = _golden_sequence()
                for report in reports:
                    _set_platform_runtime(
                        report,
                        backend=backend,
                        accelerator=accelerator,
                        runtime_version=runtime_version,
                        host_os_version=host_os_version,
                    )
                result = validate_golden_sequence(reports)
                self.assertTrue(result["golden_protocol_eligible"])
                self.assertEqual(result["eligibility_reasons"], [])

    def test_golden_rejects_platform_runtime_drift_between_positions(self) -> None:
        reports = _golden_sequence()
        for report in reports:
            _set_metal_runtime(report)
        _set_platform_runtime(
            reports[3],
            backend="cpu",
            accelerator="AMD EPYC 9654",
            runtime_version="oneDNN 3.7.1",
            host_os_version="Ubuntu 24.04 kernel 6.8.0",
        )

        result = validate_golden_sequence(reports)

        self.assertEqual(result["status"], "ineligible")
        self.assertFalse(result["golden_protocol_eligible"])
        self.assertFalse(result["decision_eligible"])
        self.assertIn(
            "uncontrolled_or_missing_runtime_accelerator_backend",
            result["eligibility_reasons"],
        )
        self.assertIn(
            "uncontrolled_or_missing_runtime_accelerator",
            result["eligibility_reasons"],
        )
        self.assertIn(
            "uncontrolled_or_missing_runtime_software_environment_digest",
            result["eligibility_reasons"],
        )

    def test_golden_runtime_alias_mismatch_and_malformed_legacy_hash_fail_closed(
        self,
    ) -> None:
        cases = (
            (
                "alias_mismatch",
                "gpu",
                "contradictory accelerator label",
                "run_1_runtime_aliases_do_not_reconcile",
            ),
            (
                "malformed_legacy_fingerprint",
                "gpu_fingerprint_sha256",
                "not-a-digest",
                "run_1_invalid_manifest_digest",
            ),
        )
        for name, field, value, reason in cases:
            with self.subTest(case=name):
                reports = _golden_sequence()
                for report in reports:
                    _set_metal_runtime(report)
                reports[0]["manifest"]["runtime"][field] = value  # type: ignore[index]

                result = validate_golden_sequence(reports)
                self.assertEqual(result["status"], "ineligible")
                self.assertFalse(result["golden_protocol_eligible"])
                self.assertFalse(result["decision_eligible"])
                self.assertIn(reason, result["eligibility_reasons"])

    def test_six_sequential_live_runs_pass_the_golden_gate(self) -> None:
        result = validate_golden_sequence(_golden_sequence())
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["golden_protocol_eligible"])
        self.assertEqual(result["eligibility_reasons"], [])
        self.assertEqual(result["decision_state"], "supported")
        self.assertEqual(result["overall_outcome"], "candidate_higher_throughput")
        self.assertEqual(result["optimization_credit"]["changed_flag"], "max_num_seqs")
        self.assertFalse(result["optimization_credit"]["chunked_prefill_credited"])
        self.assertIn(
            "does not prove direct server-scheduler saturation",
            result["client_demand_scope"],
        )
        self.assertEqual(
            result["treatment"],
            {
                "field": "max_num_seqs",
                "baseline_value": 1,
                "candidate_value": 8,
                "closed_loop_concurrency": 8,
            },
        )
        self.assertEqual(len(result["run_fingerprints"]), 6)
        summary = result["decision_summary"]
        self.assertEqual(summary["winner"], "candidate")
        self.assertEqual(summary["winner_config"], {"max_num_seqs": 8})
        self.assertTrue(summary["ci_excludes_zero"])
        self.assertIn("tested workload only", summary["text"])
        self.assertIn("candidate throughput was 20.0% higher", summary["text"])
        self.assertIn("95% CI", summary["text"])
        self.assertIn("excludes zero", summary["text"])
        self.assertIn("no latency SLO was declared", summary["text"])

    def test_arbitrary_max_num_seqs_pairs_are_inferred_from_saved_evidence(
        self,
    ) -> None:
        cases = (
            (8, 10, 10),
            (8, 10, 16),
            (8, 6, 8),
            (10, 8, 12),
        )
        for baseline, candidate, load in cases:
            with self.subTest(
                baseline=baseline, candidate=candidate, load=load
            ):
                result = validate_golden_sequence(
                    _golden_sequence(
                        baseline_max_num_seqs=baseline,
                        candidate_max_num_seqs=candidate,
                        closed_loop_concurrency=load,
                    )
                )

                self.assertTrue(
                    result["golden_protocol_eligible"],
                    result["eligibility_reasons"],
                )
                self.assertEqual(result["status"], "complete")
                self.assertEqual(
                    result["treatment"],
                    {
                        "field": "max_num_seqs",
                        "baseline_value": baseline,
                        "candidate_value": candidate,
                        "closed_loop_concurrency": load,
                    },
                )
                self.assertEqual(
                    result["conditions"][0]["condition_id"],
                    f"closed_loop:{load}",
                )
                self.assertEqual(
                    result["decision_summary"]["winner_config"],
                    {"max_num_seqs": candidate},
                )
                self.assertIn(
                    f"candidate max_num_seqs={candidate} won",
                    result["decision_summary"]["text"],
                )

    def test_arbitrary_pair_treatment_survives_a_complete_inconclusive_result(
        self,
    ) -> None:
        result = validate_golden_sequence(
            _golden_sequence(
                baseline_max_num_seqs=8,
                candidate_max_num_seqs=10,
                closed_loop_concurrency=16,
                baseline_block_rates=(10.0, 10.0, 10.0),
                candidate_block_rates=(10.0, 10.0, 10.0),
            )
        )

        self.assertTrue(result["golden_protocol_eligible"])
        self.assertFalse(result["decision_eligible"])
        self.assertEqual(result["decision_state"], "inconclusive")
        self.assertIsNone(result["decision_summary"])
        self.assertEqual(
            result["treatment"],
            {
                "field": "max_num_seqs",
                "baseline_value": 8,
                "candidate_value": 10,
                "closed_loop_concurrency": 16,
            },
        )

    def test_arbitrary_pair_baseline_winner_uses_inferred_value_and_wording(
        self,
    ) -> None:
        result = validate_golden_sequence(
            _golden_sequence(
                baseline_max_num_seqs=10,
                candidate_max_num_seqs=8,
                closed_loop_concurrency=12,
                baseline_block_rates=(12.0, 12.0, 12.0),
                candidate_block_rates=(9.0, 9.0, 9.0),
            )
        )

        self.assertTrue(result["decision_eligible"])
        summary = result["decision_summary"]
        self.assertEqual(summary["winner"], "baseline")
        self.assertEqual(summary["winner_config"], {"max_num_seqs": 10})
        self.assertIn("baseline max_num_seqs=10 won", summary["text"])
        self.assertNotIn("max_num_seqs=1 won", summary["text"])

    def test_golden_load_and_observed_peak_must_cover_the_declared_pair(
        self,
    ) -> None:
        cases = (
            (
                "load_below_pair",
                _golden_sequence(
                    baseline_max_num_seqs=8,
                    candidate_max_num_seqs=10,
                    closed_loop_concurrency=9,
                ),
            ),
            (
                "one_position_did_not_reach_declared_load",
                _golden_sequence(
                    baseline_max_num_seqs=8,
                    candidate_max_num_seqs=10,
                    closed_loop_concurrency=16,
                ),
            ),
        )
        cases[1][1][4]["conditions"][0]["observed_peak_in_flight"] = 15  # type: ignore[index]
        for name, reports in cases:
            with self.subTest(case=name):
                result = validate_golden_sequence(reports)
                self.assertEqual(result["status"], "ineligible")
                self.assertFalse(result["golden_protocol_eligible"])
                self.assertFalse(result["decision_eligible"])
                self.assertTrue(
                    any(
                        "concurrency" in reason or "traffic" in reason
                        for reason in result["eligibility_reasons"]
                    ),
                    result["eligibility_reasons"],
                )

    def test_golden_rejects_noncanonical_or_equal_max_num_seqs_evidence(
        self,
    ) -> None:
        malformed = (
            "0",
            "-1",
            "+8",
            " 8",
            "8 ",
            "08",
            "8.0",
            "1e1",
            "2147483648",
            "9" * 5_000,
            "private-treatment-secret",
        )
        for value in malformed:
            with self.subTest(value=value[:32]):
                result = validate_golden_sequence(
                    _golden_sequence(
                        baseline_max_num_seqs=value,
                        candidate_max_num_seqs=10,
                        closed_loop_concurrency=16,
                    )
                )
                serialized = json.dumps(result, allow_nan=False)
                self.assertEqual(result["status"], "ineligible")
                self.assertFalse(result["golden_protocol_eligible"])
                self.assertFalse(result["decision_eligible"])
                if value == "private-treatment-secret" or len(value) > 128:
                    self.assertNotIn(value, serialized)

        equal = validate_golden_sequence(
            _golden_sequence(
                baseline_max_num_seqs=8,
                candidate_max_num_seqs=8,
                closed_loop_concurrency=8,
            )
        )
        self.assertEqual(equal["status"], "ineligible")
        self.assertFalse(equal["golden_protocol_eligible"])
        self.assertFalse(equal["decision_eligible"])

    def test_duration_bounded_arbitrary_pair_remains_offline_validatable(
        self,
    ) -> None:
        reports = _golden_sequence(
            baseline_max_num_seqs=8,
            candidate_max_num_seqs=10,
            closed_loop_concurrency=16,
            baseline_block_rates=(3.0, 3.0, 3.0),
            candidate_block_rates=(3.6, 3.6, 3.6),
        )
        origin = datetime(2026, 8, 17, tzinfo=timezone.utc)
        for index, report in enumerate(reports):
            traffic = report["manifest"]["traffic"]  # type: ignore[index]
            traffic["requests_per_block"] = None
            traffic["block_duration_seconds"] = 20.0
            started = origin + timedelta(seconds=index * 120)
            report["started_at"] = started.isoformat()
            report["completed_at"] = (started + timedelta(seconds=110)).isoformat()

        result = validate_golden_sequence(reports)

        self.assertTrue(
            result["golden_protocol_eligible"], result["eligibility_reasons"]
        )
        self.assertEqual(
            result["treatment"],
            {
                "field": "max_num_seqs",
                "baseline_value": 8,
                "candidate_value": 10,
                "closed_loop_concurrency": 16,
            },
        )
        self.assertTrue(
            all(
                report["conditions"][0]["measured_wall_seconds"] >= 60.0  # type: ignore[index]
                and report["conditions"][0]["observed_peak_in_flight"] == 16  # type: ignore[index]
                for report in reports
            )
        )

    def test_duration_bounded_blocks_must_attempt_declared_concurrency(
        self,
    ) -> None:
        reports = _golden_sequence(
            baseline_max_num_seqs=8,
            candidate_max_num_seqs=10,
            closed_loop_concurrency=16,
            baseline_block_rates=(3.0, 3.0, 3.0),
            candidate_block_rates=(3.6, 3.6, 3.6),
        )
        origin = datetime(2026, 8, 17, tzinfo=timezone.utc)
        for index, report in enumerate(reports):
            traffic = report["manifest"]["traffic"]  # type: ignore[index]
            traffic["requests_per_block"] = None
            traffic["block_duration_seconds"] = 20.0
            started = origin + timedelta(seconds=index * 120)
            report["started_at"] = started.isoformat()
            report["completed_at"] = (started + timedelta(seconds=110)).isoformat()

            condition = report["conditions"][0]  # type: ignore[index]
            request_rates: list[float] = []
            for block in condition["blocks"]:
                counts = {
                    "attempted": 1,
                    "valid": 1,
                    "invalid": 0,
                    "status_counts": {"200": 1},
                    "error_counts": {},
                    "finish_reason_counts": {"stop": 1},
                }
                wall = float(block["wall_duration_seconds"])
                request_rate = 1.0 / wall
                request_rates.append(request_rate)
                block["request_counts"] = counts
                block["offered_requests"] = 1
                for key in ("metrics", "diagnostic_metrics"):
                    block[key]["valid_response_count"] = 1
                    block[key]["requests_per_second"] = request_rate

            condition_counts = {
                "attempted": 3,
                "valid": 3,
                "invalid": 0,
                "status_counts": {"200": 3},
                "error_counts": {},
                "finish_reason_counts": {"stop": 3},
            }
            measured_wall = float(condition["measured_wall_seconds"])
            for key in ("metrics", "diagnostic_metrics"):
                condition[key]["valid_response_count"] = 3
                condition[key]["requests_per_second"] = 3.0 / measured_wall
                condition[key]["block_mean_requests_per_second"] = (
                    sum(request_rates) / len(request_rates)
                )
                condition[key]["block_mean_requests_per_second_ci"] = (
                    t_interval_95(request_rates)
                )
            condition["request_counts"] = condition_counts
            report["run_totals"].update(  # type: ignore[index]
                requests_started=6,
                requests_completed=6,
                reserved_output_tokens=768,
            )

        result = validate_golden_sequence(reports)

        self.assertEqual(result["status"], "ineligible")
        self.assertFalse(result["golden_protocol_eligible"])
        self.assertFalse(result["decision_eligible"])
        self.assertIsNone(result["decision_summary"])
        self.assertIsNone(result["treatment"])
        self.assertIn(
            "golden_measured_blocks_must_attempt_declared_concurrency",
            result["eligibility_reasons"],
        )

    def test_golden_structural_preflight_precedes_custom_mapping_traversal(
        self,
    ) -> None:
        class ExplosiveMapping(Mapping[str, str]):
            def __getitem__(self, key: str) -> str:
                del key
                raise AssertionError("custom mapping was executed")

            def __iter__(self) -> Iterator[str]:
                raise AssertionError("custom mapping was executed")

            def __len__(self) -> int:
                raise AssertionError("custom mapping was executed")

        reports = _golden_sequence()
        reports[0]["manifest"]["engine"][  # type: ignore[index]
            "effective_flags"
        ] = ExplosiveMapping()

        result = validate_golden_sequence(reports)

        self.assertEqual(result["status"], "ineligible")
        self.assertFalse(result["golden_protocol_eligible"])
        self.assertFalse(result["decision_eligible"])
        self.assertIsNone(result["decision_summary"])
        self.assertIsNone(result["treatment"])
        self.assertIn(
            "run_1_report_structure_non_json_type",
            result["eligibility_reasons"],
        )

    def test_golden_structural_preflight_precedes_forged_key_lookup(self) -> None:
        class ExplosiveKey(str):
            def __hash__(self) -> int:
                return hash(str(self))

            def __eq__(self, other: object) -> bool:
                del other
                raise AssertionError("forged key equality was executed")

        reports = [{ExplosiveKey("manifest"): {}} for _ in range(6)]

        result = validate_golden_sequence(reports)

        self.assertEqual(result["status"], "ineligible")
        self.assertFalse(result["golden_protocol_eligible"])
        self.assertFalse(result["decision_eligible"])
        self.assertIsNone(result["decision_summary"])
        self.assertIsNone(result["treatment"])
        self.assertEqual(
            result["eligibility_reasons"],
            [
                f"run_{index}_report_structure_non_json_type"
                for index in range(1, 7)
            ],
        )

    def test_golden_rejects_wrong_changed_flag_second_flag_and_repeat_drift(
        self,
    ) -> None:
        wrong_flag = _golden_sequence(
            baseline_max_num_seqs=8,
            candidate_max_num_seqs=10,
            closed_loop_concurrency=16,
        )
        for report in wrong_flag:
            flags = report["manifest"]["engine"]["effective_flags"]  # type: ignore[index]
            flags["max_num_batched_tokens"] = flags.pop("max_num_seqs")

        second_flag = _golden_sequence(
            baseline_max_num_seqs=8,
            candidate_max_num_seqs=10,
            closed_loop_concurrency=16,
        )
        for report in second_flag[1::2]:
            report["manifest"]["engine"]["effective_flags"][  # type: ignore[index]
                "tensor_parallel_size"
            ] = "2"

        repeat_drift = _golden_sequence(
            baseline_max_num_seqs=8,
            candidate_max_num_seqs=10,
            closed_loop_concurrency=16,
        )
        repeat_drift[2]["manifest"]["engine"]["effective_flags"][  # type: ignore[index]
            "max_num_seqs"
        ] = "9"

        normalized_duplicate = _golden_sequence(
            baseline_max_num_seqs=8,
            candidate_max_num_seqs=10,
            closed_loop_concurrency=16,
        )
        normalized_duplicate[0]["manifest"]["engine"]["effective_flags"][  # type: ignore[index]
            "max-num-seqs"
        ] = "11"

        for name, reports, reason in (
            (
                "wrong_flag",
                wrong_flag,
                "golden_treatment_must_only_change_max_num_seqs",
            ),
            (
                "second_flag",
                second_flag,
                "golden_treatment_must_only_change_max_num_seqs",
            ),
            (
                "repeat_drift",
                repeat_drift,
                "baseline_effective_flags_changed_between_repeats",
            ),
            (
                "normalized_duplicate",
                normalized_duplicate,
                "run_1_invalid_engine_flag_manifest",
            ),
        ):
            with self.subTest(case=name):
                result = validate_golden_sequence(reports)
                self.assertEqual(result["status"], "ineligible")
                self.assertFalse(result["golden_protocol_eligible"])
                self.assertFalse(result["decision_eligible"])
                self.assertIn(reason, result["eligibility_reasons"])

    def test_supported_golden_summary_orients_a_baseline_win_correctly(self) -> None:
        result = validate_golden_sequence(
            _golden_sequence(
                baseline_block_rates=(12.0, 12.0, 12.0),
                candidate_block_rates=(9.0, 9.0, 9.0),
            )
        )

        self.assertTrue(result["decision_eligible"])
        self.assertEqual(result["overall_outcome"], "baseline_higher_throughput")
        summary = result["decision_summary"]
        self.assertEqual(summary["winner"], "baseline")
        self.assertEqual(summary["winner_config"], {"max_num_seqs": 1})
        self.assertLess(summary["candidate_throughput_delta_percent"], 0)
        self.assertIn("candidate throughput was 25.0% lower", summary["text"])
        self.assertNotIn("25.0% higher", summary["text"])

    def test_golden_gate_recomputes_source_order_and_nonoverlap(self) -> None:
        reports = _golden_sequence()
        reports[1]["manifest"]["provenance"]["evidence_source"] = "synthetic_validation"  # type: ignore[index]
        reports[2]["manifest"]["provenance"]["variant"] = "candidate"  # type: ignore[index]
        reports[3]["started_at"] = reports[2]["started_at"]
        result = validate_golden_sequence(reports)
        self.assertEqual(result["status"], "ineligible")
        self.assertFalse(result["golden_protocol_eligible"])
        self.assertEqual(result["decision_state"], "inconclusive")
        self.assertIn(
            "all_six_runs_must_be_live_inference", result["eligibility_reasons"]
        )
        self.assertIn(
            "sequence_must_be_baseline_candidate_baseline_then_candidate_baseline_candidate",
            result["eligibility_reasons"],
        )
        self.assertIn("runs_overlap_or_are_out_of_order", result["eligibility_reasons"])

    def test_supported_golden_summary_names_only_declared_slo_gates(self) -> None:
        reports = _golden_sequence()
        for report in reports:
            report["manifest"]["traffic"]["p95_slo_ms"] = 100.0  # type: ignore[index]
            report["manifest"]["traffic"]["ttft_slo_ms"] = 50.0  # type: ignore[index]

        result = validate_golden_sequence(reports)

        self.assertTrue(result["decision_eligible"])
        summary = result["decision_summary"]
        self.assertEqual(summary["declared_slo_gates_passed"], ["E2E", "TTFT"])
        self.assertIn(
            "all declared E2E and TTFT SLO gates passed", summary["text"]
        )
        self.assertNotIn("latency parity", summary["text"])

    def test_golden_rejects_false_or_missing_disjoint_warmup_evidence(self) -> None:
        for evidence in (False, "missing"):
            with self.subTest(evidence=evidence):
                reports = _golden_sequence()
                workload = reports[3]["manifest"]["workload"]  # type: ignore[index]
                if evidence == "missing":
                    workload.pop("warmup_prompts_disjoint")  # type: ignore[union-attr]
                else:
                    workload["warmup_prompts_disjoint"] = False  # type: ignore[index]

                result = validate_golden_sequence(reports)
                self.assertEqual(result["status"], "ineligible")
                self.assertFalse(result["golden_protocol_eligible"])
                self.assertIn(
                    "run_4_warmup_workload_not_separate",
                    result["eligibility_reasons"],
                )

    def test_golden_position_token_mismatch_cannot_cancel_in_aggregate(self) -> None:
        # Baseline and candidate each total 900, but individual positions differ by 18%.
        reports = _golden_sequence((270, 330, 300, 270, 330, 300))
        result = validate_golden_sequence(reports)

        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["golden_protocol_eligible"])
        self.assertFalse(result["decision_eligible"])
        self.assertEqual(result["decision_state"], "inconclusive")
        self.assertIsNone(result["overall_outcome"])
        self.assertIsNone(result["decision_summary"])
        condition = result["conditions"][0]
        self.assertEqual(condition["completion_token_relative_difference"], 0.0)
        self.assertGreater(
            condition["completion_token_relative_spread_across_positions"],
            condition["completion_token_tolerance"],
        )
        self.assertEqual(
            condition["reason"],
            "completion_tokens_outside_5_percent_tolerance",
        )

    def test_golden_block_token_spread_cannot_cancel_at_position_totals(self) -> None:
        reports = _golden_sequence()
        distributions = (
            (80, 100, 120),
            (120, 100, 80),
            (100, 120, 80),
            (80, 120, 100),
            (120, 80, 100),
            (100, 80, 120),
        )
        for report, block_tokens in zip(reports, distributions):
            _set_block_completion_tokens(report, block_tokens)

        result = validate_golden_sequence(reports)
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["golden_protocol_eligible"])
        self.assertFalse(result["decision_eligible"])
        self.assertEqual(result["decision_state"], "inconclusive")
        self.assertIsNone(result["overall_outcome"])
        condition = result["conditions"][0]
        self.assertEqual(condition["completion_token_relative_difference"], 0.0)
        self.assertEqual(
            condition["completion_token_relative_spread_across_positions"],
            0.0,
        )
        self.assertGreater(
            condition[
                "maximum_block_completion_token_relative_spread_across_positions"
            ],
            condition["completion_token_tolerance"],
        )
        self.assertEqual(
            condition["reason"],
            "completion_tokens_outside_5_percent_tolerance",
        )

    def test_golden_declared_slo_failure_is_inconclusive(self) -> None:
        reports = _golden_sequence()
        for report in reports:
            report["manifest"]["traffic"]["p95_slo_ms"] = 5.0  # type: ignore[index]

        result = validate_golden_sequence(reports)
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["golden_protocol_eligible"])
        self.assertFalse(result["decision_eligible"])
        self.assertEqual(result["decision_state"], "inconclusive")
        self.assertIsNone(result["overall_outcome"])
        self.assertEqual(
            result["conditions"][0]["reason"],
            "one_or_more_runs_fail_declared_slo",
        )

    def test_golden_requires_the_whole_safety_manifest_to_match(self) -> None:
        reports = _golden_sequence()
        reports[3]["manifest"]["safety"]["overrides"]["insecure_http"] = True  # type: ignore[index]

        result = validate_golden_sequence(reports)
        self.assertEqual(result["status"], "ineligible")
        self.assertFalse(result["golden_protocol_eligible"])
        self.assertIn(
            "uncontrolled_or_missing_safety",
            result["eligibility_reasons"],
        )


class OfflineNetworkGuardTests(unittest.TestCase):
    def test_candidate_secret_regression_is_blocked_before_dns(self) -> None:
        with self.assertRaisesRegex(AssertionError, "non-loopback DNS"):
            socket.getaddrinfo("candidate-secret.example", 443)

    def test_reports_remain_json_serializable_without_nan(self) -> None:
        comparison = compare_reports(
            _saved_report((10.0, 10.0, 10.0), flag_value="1"),
            _saved_report((12.0, 12.0, 12.0), flag_value="8"),
        )
        serialized = json.dumps(comparison, allow_nan=False)
        self.assertNotIn("private measured prompt", serialized)
        self.assertNotIn("private generated response", serialized)


if __name__ == "__main__":
    unittest.main()
