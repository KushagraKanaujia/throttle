"""Fail-closed, out-of-process integration for the pinned GuideLLM backend.

GuideLLM reports intentionally cross a narrow import boundary here.  The raw
report is kept in a private temporary directory and only explicitly allowlisted
numeric aggregates are copied into the value returned to Throttle.  In
particular, endpoint details, credentials, prompts, responses, labels, and
GuideLLM error strings never enter an imported result.

GuideLLM 0.7.3 cannot prove the completion ``finish_reason`` or the provenance
of token usage, and it does not enforce Throttle's response-size limit.  Those
are hard decision gates, so every result imported by this module is explicitly
decision-ineligible.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import math
import os
import re
import signal
import stat
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping
from urllib.parse import urlsplit, urlunsplit

from .models import (
    OPEN_LOOP_RATE_RELATIVE_TOLERANCE,
    OPEN_LOOP_SCHEDULER_LAG_INTERVAL_TOLERANCE,
)

PINNED_GUIDELLM_VERSION = "0.7.3"
GUIDELLM_VERSION = PINNED_GUIDELLM_VERSION
GUIDELLM_REPORT_VERSION = 2
GUIDELLM_API_KEY_ENV = "GUIDELLM__SPEC__BACKEND__API_KEY"

DECISION_INELIGIBLE_REASONS = (
    "guidellm_cannot_verify_finish_reason_or_usage_provenance",
    "max_response_bytes_unenforced",
)

_VERSION_PATTERN = re.compile(r"guidellm version:\s*([0-9]+\.[0-9]+\.[0-9]+)\s*")
_MAX_VERSION_OUTPUT_BYTES = 4_096
_MAX_RAW_REPORT_BYTES = 64 * 1024 * 1024
_MAX_SAFE_NUMERIC_MAGNITUDE = 9_007_199_254_740_991
_PROCESS_EXIT_GRACE_SECONDS = 2.0

_SUMMARY_FIELDS = (
    "mean",
    "median",
    "mode",
    "variance",
    "std_dev",
    "min",
    "max",
    "count",
    "total_sum",
)
_PERCENTILE_FIELDS = ("p50", "p90", "p95", "p99")
_METRIC_FIELDS = (
    "requests_per_second",
    "request_concurrency",
    "request_latency",
    "time_to_first_token_ms",
    "time_per_output_token_ms",
    "inter_token_latency_ms",
    "output_tokens_per_second",
    "prompt_token_count",
    "output_token_count",
)
_REQUEST_COUNT_METRICS = (
    "requests_per_second",
    "request_concurrency",
    "request_latency",
    "time_to_first_token_ms",
    "prompt_token_count",
    "output_token_count",
)
_REQUEST_TOTAL_FIELDS = ("successful", "errored", "incomplete", "total")


class GuideLLMBackendError(RuntimeError):
    """A deliberately sanitized GuideLLM integration failure.

    ``code`` is a fixed, non-secret identifier.  Child stdout, stderr, raw
    report values, filesystem paths, and endpoint details are never included in
    the exception message.
    """

    def __init__(self, code: str):
        self.code = code
        super().__init__(f"GuideLLM backend failed safely ({code})")


@dataclass(frozen=True)
class GuideLLMTraffic:
    """One GuideLLM traffic registry entry."""

    kind: Literal["concurrent", "constant"]
    value: float
    max_concurrency: int

    @classmethod
    def concurrent(cls, streams: int) -> GuideLLMTraffic:
        return cls(kind="concurrent", value=float(streams), max_concurrency=streams)

    @classmethod
    def constant(
        cls, requests_per_second: float, max_concurrency: int
    ) -> GuideLLMTraffic:
        return cls(
            kind="constant",
            value=float(requests_per_second),
            max_concurrency=max_concurrency,
        )

    def validate(self) -> None:
        _require_positive_int(self.max_concurrency)
        _require_positive_number(self.value)
        if self.kind == "concurrent":
            if not self.value.is_integer():
                raise GuideLLMBackendError("invalid_run_spec")
            if int(self.value) != self.max_concurrency:
                raise GuideLLMBackendError("invalid_run_spec")
        elif self.kind != "constant":
            raise GuideLLMBackendError("invalid_run_spec")

    def registry_args(self) -> dict[str, int | float | str]:
        """Return the exact 0.7.3 profile registry arguments."""

        self.validate()
        if self.kind == "concurrent":
            return {"kind": "concurrent", "streams": int(self.value)}
        return {
            "kind": "constant",
            "rate": self.value,
            "max_concurrency": self.max_concurrency,
        }

    def public_dict(self) -> dict[str, int | float | str]:
        if self.kind == "concurrent":
            return {
                "kind": "concurrent",
                "streams": int(self.value),
                "max_concurrency": self.max_concurrency,
            }
        return {
            "kind": "constant",
            "requests_per_second": self.value,
            "max_concurrency": self.max_concurrency,
        }


@dataclass(frozen=True)
class GuideLLMLimits:
    """GuideLLM-enforceable request, duration, error, and wall-clock limits."""

    max_requests: int
    max_duration_seconds: float
    max_errors: int
    process_timeout_seconds: float | None = None

    def validate(self) -> None:
        _require_positive_int(self.max_requests)
        _require_positive_number(self.max_duration_seconds)
        _require_positive_int(self.max_errors)
        if self.process_timeout_seconds is not None:
            _require_positive_number(self.process_timeout_seconds)

    def registry_args(self) -> list[dict[str, int | float | str]]:
        """Return exact 0.7.3 constraint registry arguments."""

        self.validate()
        return [
            {"kind": "max_requests", "count": self.max_requests},
            {"kind": "max_duration", "seconds": self.max_duration_seconds},
            {
                "kind": "max_errors",
                "count": self.max_errors,
                "stopping_scope": "all",
            },
        ]

    @property
    def wall_timeout_seconds(self) -> float:
        # Account for tokenizer/backend initialization while still imposing a
        # finite outer wall-clock ceiling on the whole child process.
        if self.process_timeout_seconds is not None:
            return self.process_timeout_seconds
        return self.max_duration_seconds + 60.0


@dataclass(frozen=True)
class GuideLLMRunSpec:
    """Private inputs needed to create a single GuideLLM 0.7.3 scenario.

    The API key is intentionally absent.  It is accepted separately by
    :func:`run_guidellm` so it cannot appear in a dataclass repr, scenario file,
    or subprocess argument.
    """

    endpoint_url: str
    model: str
    tokenizer_model: str
    traffic: GuideLLMTraffic
    limits: GuideLLMLimits
    prompt_tokens: int
    output_tokens: int
    seed: int = 42
    tokenizer_revision: str | None = None
    request_timeout_seconds: float = 120.0
    connect_timeout_seconds: float = 10.0
    allow_insecure_http: bool = False

    def validate(self) -> None:
        if not isinstance(self.allow_insecure_http, bool):
            raise GuideLLMBackendError("invalid_run_spec")
        _normalized_endpoint(
            self.endpoint_url, allow_insecure_http=self.allow_insecure_http
        )
        if not _nonempty_text(self.model) or not _nonempty_text(self.tokenizer_model):
            raise GuideLLMBackendError("invalid_run_spec")
        if self.tokenizer_revision is not None and not _nonempty_text(
            self.tokenizer_revision
        ):
            raise GuideLLMBackendError("invalid_run_spec")
        self.traffic.validate()
        self.limits.validate()
        _require_positive_int(self.prompt_tokens)
        _require_positive_int(self.output_tokens)
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise GuideLLMBackendError("invalid_run_spec")
        _require_positive_number(self.request_timeout_seconds)
        _require_positive_number(self.connect_timeout_seconds)


def build_guidellm_scenario(
    spec: GuideLLMRunSpec, output_path: str | os.PathLike[str]
) -> dict[str, Any]:
    """Build one exact GuideLLM 0.7.3 scenario without including a secret."""

    spec.validate()
    output = Path(output_path)
    if not output.is_absolute():
        raise GuideLLMBackendError("invalid_run_spec")

    tokenizer: dict[str, Any] = {
        "kind": "huggingface_auto",
        "model": spec.tokenizer_model,
        "load_kwargs": {
            "trust_remote_code": False,
            # Plan names one inference destination. Do not let tokenizer
            # setup add an undisclosed Hugging Face network destination.
            "local_files_only": True,
        },
    }
    if spec.tokenizer_revision is not None:
        tokenizer["load_kwargs"]["revision"] = spec.tokenizer_revision

    return {
        "metadata": {"labels": {}},
        "spec": {
            "backend": {
                "kind": "openai_http",
                "target": _normalized_endpoint(
                    spec.endpoint_url,
                    allow_insecure_http=spec.allow_insecure_http,
                ),
                "model": spec.model,
                "request_format": "/v1/chat/completions",
                "timeout": spec.request_timeout_seconds,
                "timeout_connect": spec.connect_timeout_seconds,
                "http2": True,
                "follow_redirects": False,
                "verify": True,
                # GuideLLM's backend validation performs an unconstrained
                # authenticated GET /health before each child run. Disable it
                # so Throttle's request ceiling and plan remain exact; the
                # constrained inference request fails closed if the backend is
                # not ready.
                "validate_backend": False,
                "stream": True,
                "max_tokens": spec.output_tokens,
                "extras": {
                    "body": {
                        "temperature": 0,
                        "max_tokens": spec.output_tokens,
                    }
                },
            },
            "profile": spec.traffic.registry_args(),
            "constraints": spec.limits.registry_args(),
            "tokenizer": tokenizer,
            "data": [
                {
                    "kind": "synthetic_text",
                    "prompt_tokens": spec.prompt_tokens,
                    "output_tokens": spec.output_tokens,
                }
            ],
            "data_loader": {
                "kind": "pytorch",
                "samples": -1,
                "shuffle": False,
                "num_workers": 0,
            },
            "seed": {"kind": "static", "value": spec.seed},
            "metrics": {
                "kind": "generative",
                "sample_size": 0,
                "prefer_response_metrics": True,
            },
            "outputs": [{"kind": "json", "path": str(output)}],
        },
        "benchmarks": [None],
    }


def verify_guidellm_version(
    executable: str | os.PathLike[str] = "guidellm",
    *,
    timeout_seconds: float = 10.0,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Verify that ``executable`` reports the immutable supported version."""

    _require_positive_number(timeout_seconds)
    clean_env = _sanitized_child_env(environ)
    try:
        with tempfile.TemporaryDirectory(prefix="throttle-guidellm-version-") as name:
            workdir = Path(name)
            os.chmod(workdir, 0o700)
            output = _run_version_process(
                os.fspath(executable), clean_env, workdir, timeout_seconds
            )
    except GuideLLMBackendError:
        raise
    except (OSError, ValueError):
        raise GuideLLMBackendError("version_check_failed") from None

    if len(output) > _MAX_VERSION_OUTPUT_BYTES:
        raise GuideLLMBackendError("version_check_failed")
    try:
        rendered = output.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise GuideLLMBackendError("version_check_failed") from None
    match = _VERSION_PATTERN.fullmatch(rendered)
    if match is None or match.group(1) != GUIDELLM_VERSION:
        raise GuideLLMBackendError("version_mismatch")
    return GUIDELLM_VERSION


def run_guidellm(
    spec: GuideLLMRunSpec,
    *,
    api_key: str,
    executable: str | os.PathLike[str] = "guidellm",
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run GuideLLM once and return a sanitized, decision-ineligible import."""

    if os.name != "posix":
        raise GuideLLMBackendError("guidellm_requires_posix_process_groups")
    spec.validate()
    if not _nonempty_text(api_key):
        raise GuideLLMBackendError("invalid_api_key")

    # Verify without putting the credential into the version process.
    verify_guidellm_version(executable, environ=environ)
    return _run_guidellm_verified(
        spec,
        api_key=api_key,
        executable=executable,
        environ=environ,
    )


def preflight_guidellm_config(config: Any, *, for_traffic: bool = False) -> None:
    """Validate a GuideLLM matrix config without I/O or credential exposure.

    ``for_traffic=False`` is intended for ``throttle plan``: common safety and
    adapter structure are validated, but an API key and the explicit GuideLLM
    validation-gap acknowledgement are not required.  The traffic form enables
    those gates.  Neither form resolves a hostname, checks the executable, or
    starts a subprocess.
    """

    from .benchmark import validate_config

    try:
        validate_config(config, for_traffic=for_traffic)
    except (AttributeError, TypeError, ValueError):
        raise GuideLLMBackendError("invalid_run_config") from None
    if config.backend != "guidellm":
        raise GuideLLMBackendError("invalid_run_config")
    if for_traffic and os.name != "posix":
        raise GuideLLMBackendError("guidellm_requires_posix_process_groups")
    if config.stream is not True:
        raise GuideLLMBackendError("guidellm_requires_streaming")
    _normalized_endpoint(
        config.endpoint.url,
        allow_insecure_http=config.allow_insecure_http,
    )


@dataclass
class _GuideLLMMatrixBudget:
    config: Any
    started: float
    requests_started: int = 0
    requests_completed: int = 0
    requests_cancelled: int = 0
    errors: int = 0
    reserved_output_tokens: int = 0
    declared_peak_in_flight_cap: int = 0
    stop_reason: str | None = None
    accounting_incomplete: bool = False
    incomplete_children: list[dict[str, Any]] = field(default_factory=list)

    def elapsed(self) -> float:
        return max(0.0, time.monotonic() - self.started)

    def set_stop(self, reason: str) -> None:
        if self.stop_reason is None:
            self.stop_reason = reason

    def remaining_wall_seconds(self) -> float:
        remaining = self.config.limits.max_elapsed_seconds - self.elapsed()
        spent = self.config.cost.elapsed_estimate(self.elapsed())
        if spent is not None:
            spend_remaining = self.config.limits.max_estimated_spend - spent
            if spend_remaining <= 0:
                self.set_stop("max_estimated_spend")
                return 0.0
            rate = _cost_rate_per_wall_second(self.config)
            if rate is not None and rate > 0:
                remaining = min(remaining, spend_remaining / rate)
        if remaining <= 0:
            if self.stop_reason is None:
                self.set_stop("max_elapsed_time")
            return 0.0
        return remaining

    def child_limits(
        self,
        *,
        requested_requests: int | None,
        requested_duration: float | None,
    ) -> GuideLLMLimits | None:
        """Reserve a safe child envelope without launching any traffic."""

        if self.stop_reason is not None:
            return None
        wall_remaining = self.remaining_wall_seconds()
        # Leave time for GuideLLM to serialize the private JSON report. Traffic
        # is still bounded by the outer wall timeout if serialization overruns.
        report_grace = min(5.0, wall_remaining * 0.1)
        traffic_seconds = wall_remaining - report_grace
        if traffic_seconds <= 0:
            self.set_stop("max_elapsed_time")
            return None

        requests_remaining = self.config.limits.max_requests - self.requests_started
        token_request_capacity = (
            self.config.limits.max_total_requested_tokens - self.reserved_output_tokens
        ) // self.config.max_tokens
        errors_remaining = self.config.limits.max_errors - self.errors
        if requests_remaining <= 0:
            self.set_stop("max_requests")
            return None
        if token_request_capacity <= 0:
            self.set_stop("max_total_requested_tokens")
            return None
        if errors_remaining <= 0:
            self.set_stop("max_errors")
            return None

        capacity = min(requests_remaining, token_request_capacity)
        if requested_requests is not None:
            if requested_requests > requests_remaining:
                self.set_stop("max_requests")
                return None
            if requested_requests > token_request_capacity:
                self.set_stop("max_total_requested_tokens")
                return None
            child_requests = requested_requests
        else:
            child_requests = capacity

        if requested_duration is not None:
            # A duration-only block that cannot reach its requested measurement
            # window must not be launched as though it were a complete block.
            if requested_requests is None and requested_duration > traffic_seconds:
                self.set_stop("max_elapsed_time")
                return None
            child_duration = min(requested_duration, traffic_seconds)
        else:
            child_duration = traffic_seconds
        if child_duration <= 0 or child_requests <= 0:
            self.set_stop("max_elapsed_time")
            return None
        return GuideLLMLimits(
            max_requests=child_requests,
            max_duration_seconds=child_duration,
            max_errors=errors_remaining,
            process_timeout_seconds=wall_remaining,
        )

    def child_envelope(
        self,
        *,
        phase: Literal["warmup", "measurement"],
        condition_index: int,
        block_index: int | None,
        seed: int,
        limits: GuideLLMLimits,
        max_in_flight: int,
    ) -> dict[str, Any]:
        """Describe the conservative accounting range for a launched child.

        GuideLLM only exposes aggregate accounting after it has successfully
        serialized a valid report.  Until that report crosses the import
        boundary, any amount from zero through the child's declared safety
        envelope may have reached the endpoint.
        """

        self.declared_peak_in_flight_cap = max(
            self.declared_peak_in_flight_cap, max_in_flight
        )
        request_upper = limits.max_requests
        return {
            "phase": phase,
            "condition_index": condition_index,
            "block_index": block_index,
            "seed": seed,
            "accounting_incomplete": True,
            "completion_state": "unknown_after_child_failure",
            "request_count_bounds": {"lower": 0, "upper": request_upper},
            "completed_request_count_bounds": {
                "lower": 0,
                "upper": request_upper,
            },
            "cancelled_request_count_bounds": {
                "lower": 0,
                "upper": request_upper,
            },
            "requested_output_token_bounds": {
                "lower": 0,
                "upper": request_upper * self.config.max_tokens,
            },
            "error_count_bounds": {"lower": 0, "upper": request_upper},
            "declared_peak_in_flight_cap": max_in_flight,
            "observed_peak_in_flight": None,
        }

    def mark_child_incomplete(self, envelope: Mapping[str, Any]) -> None:
        self.accounting_incomplete = True
        self.incomplete_children.append(dict(envelope))

    def record(self, imported: Mapping[str, Any]) -> None:
        totals = imported["request_totals"]
        attempted = int(totals["total"])
        failures = int(totals["errored"]) + int(totals["incomplete"])
        self.requests_started += attempted
        self.requests_completed += attempted - int(totals["incomplete"])
        self.requests_cancelled += int(totals["incomplete"])
        self.errors += failures
        self.reserved_output_tokens += attempted * self.config.max_tokens
        if self.requests_started > self.config.limits.max_requests:
            self.set_stop("max_requests")
        if self.reserved_output_tokens > self.config.limits.max_total_requested_tokens:
            self.set_stop("max_total_requested_tokens")
        if self.errors >= self.config.limits.max_errors:
            self.set_stop("max_errors")
        self.remaining_wall_seconds()

    def public_dict(self, *, elapsed_seconds: float | None = None) -> dict[str, Any]:
        uncertain_requests = sum(
            int(child["request_count_bounds"]["upper"])
            for child in self.incomplete_children
        )
        request_upper = min(
            self.config.limits.max_requests,
            self.requests_started + uncertain_requests,
        )
        token_upper = min(
            self.config.limits.max_total_requested_tokens,
            self.reserved_output_tokens
            + sum(
                int(child["requested_output_token_bounds"]["upper"])
                for child in self.incomplete_children
            ),
        )
        error_upper = min(
            request_upper,
            self.errors
            + sum(
                int(child["error_count_bounds"]["upper"])
                for child in self.incomplete_children
            ),
        )
        completed_upper = min(
            request_upper,
            self.requests_completed + uncertain_requests,
        )
        cancelled_upper = min(
            request_upper,
            self.requests_cancelled + uncertain_requests,
        )
        exact_or_unknown = lambda value: None if self.accounting_incomplete else value
        return {
            "requests_started": exact_or_unknown(self.requests_started),
            "requests_completed": exact_or_unknown(self.requests_completed),
            "requests_cancelled": exact_or_unknown(self.requests_cancelled),
            "requests_in_flight": 0,
            # GuideLLM 0.7.3 reports aggregate concurrency statistics, not an
            # auditable peak.  Never relabel the configured ceiling as an
            # observation, even for a successfully parsed child.
            "peak_in_flight": None,
            "observed_peak_in_flight": None,
            "declared_peak_in_flight_cap": self.declared_peak_in_flight_cap,
            "errors": exact_or_unknown(self.errors),
            "reserved_output_tokens": exact_or_unknown(self.reserved_output_tokens),
            "elapsed_seconds": (
                self.elapsed() if elapsed_seconds is None else elapsed_seconds
            ),
            "accounting_incomplete": self.accounting_incomplete,
            "accounted_requests_started": self.requests_started,
            "accounted_requests_completed": self.requests_completed,
            "accounted_requests_cancelled": self.requests_cancelled,
            "accounted_errors": self.errors,
            "accounted_reserved_output_tokens": self.reserved_output_tokens,
            "request_count_bounds": {
                "lower": self.requests_started,
                "upper": request_upper,
            },
            "completed_request_count_bounds": {
                "lower": self.requests_completed,
                "upper": completed_upper,
            },
            "cancelled_request_count_bounds": {
                "lower": self.requests_cancelled,
                "upper": cancelled_upper,
            },
            "requested_output_token_bounds": {
                "lower": self.reserved_output_tokens,
                "upper": token_upper,
            },
            "error_count_bounds": {
                "lower": self.errors,
                "upper": error_upper,
            },
            "incomplete_children": list(self.incomplete_children),
        }


def run_guidellm_matrix(
    config: Any,
    *,
    prompt_tokens: int,
    executable: str | os.PathLike[str] = "guidellm",
    progress: Any = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run every configured condition/block as isolated GuideLLM invocations.

    This is a schema-2 ``throttle_run`` adapter for GuideLLM's ``synthetic_text``
    workload. It does not consume or claim parity with Throttle's prompt JSONL.
    Warm-up invocations use separate seeds and their metrics are discarded.
    The executable version is verified exactly once before any credential is
    exposed to a child process.
    """

    from .benchmark import ARTIFACT_TYPE, SCHEMA_VERSION

    preflight_guidellm_config(config, for_traffic=True)
    _require_positive_int(prompt_tokens)

    report = _initial_matrix_report(
        config,
        prompt_tokens=prompt_tokens,
        schema_version=SCHEMA_VERSION,
        artifact_type=ARTIFACT_TYPE,
    )
    budget = _GuideLLMMatrixBudget(config=config, started=time.monotonic())
    _progress_set(progress, report)
    try:
        version_timeout = min(10.0, budget.remaining_wall_seconds())
        if version_timeout <= 0:
            raise GuideLLMBackendError("max_elapsed_time")
        verify_guidellm_version(
            executable,
            timeout_seconds=version_timeout,
            environ=environ,
        )
        for condition_index, condition in enumerate(config.conditions):
            if budget.stop_reason is not None:
                break
            traffic = (
                GuideLLMTraffic.concurrent(int(condition.value))
                if condition.kind == "closed_loop"
                else GuideLLMTraffic.constant(
                    float(condition.value), condition.max_in_flight
                )
            )
            entry: dict[str, Any] = {
                "condition": condition.public_dict(),
                "valid": False,
                "decision_grade": False,
                "strict_completion_validation": False,
                "decision_ineligible_reasons": list(DECISION_INELIGIBLE_REASONS),
                "declared_peak_in_flight_cap": condition.max_in_flight,
                "observed_peak_in_flight": None,
                "warmup": _empty_request_counts(),
                "blocks": [],
            }
            report["conditions"].append(entry)
            _progress_set(progress, report)

            if config.warmup_requests_per_condition:
                warmup_seed = config.seed + condition_index * 10_000 - 1
                limits = budget.child_limits(
                    requested_requests=config.warmup_requests_per_condition,
                    requested_duration=None,
                )
                if limits is None:
                    break
                warmup = _run_matrix_child_accounted(
                    config,
                    budget=budget,
                    phase="warmup",
                    condition_index=condition_index + 1,
                    block_index=None,
                    prompt_tokens=prompt_tokens,
                    traffic=traffic,
                    limits=limits,
                    seed=warmup_seed,
                    executable=executable,
                    environ=environ,
                )
                entry["warmup"] = _canonical_request_counts(warmup)
                entry["warmup"]["metrics_discarded"] = True
                entry["warmup"]["seed"] = warmup_seed
                _progress_set(progress, report)

            blocks: list[dict[str, Any]] = entry["blocks"]
            for block_index in range(config.blocks):
                if budget.stop_reason is not None:
                    break
                limits = budget.child_limits(
                    requested_requests=config.requests_per_block,
                    requested_duration=config.block_duration_seconds,
                )
                if limits is None:
                    break
                block_seed = config.seed + condition_index * 10_000 + block_index
                imported = _run_matrix_child_accounted(
                    config,
                    budget=budget,
                    phase="measurement",
                    condition_index=condition_index + 1,
                    block_index=block_index + 1,
                    prompt_tokens=prompt_tokens,
                    traffic=traffic,
                    limits=limits,
                    seed=block_seed,
                    executable=executable,
                    environ=environ,
                )
                blocks.append(
                    _canonical_block(
                        imported,
                        index=block_index + 1,
                        seed=block_seed,
                        requested_requests=config.requests_per_block,
                        requested_duration=config.block_duration_seconds,
                        declared_peak_in_flight_cap=condition.max_in_flight,
                    )
                )
                _progress_set(progress, report)

            aggregate = _canonical_condition(config, condition, entry)
            entry.clear()
            entry.update(aggregate)
            _progress_set(progress, report)

        if budget.stop_reason is not None:
            status = "stopped"
        elif not report["conditions"] or any(
            not item.get("valid") for item in report["conditions"]
        ):
            status = "failed"
        else:
            status = "complete"
        _finalize_matrix_report(
            report,
            config,
            budget,
            status=status,
            stop_reason=budget.stop_reason,
        )
        _progress_set(progress, report)
        return report
    except GuideLLMBackendError as error:
        budget.set_stop(error.code)
        report["operational_error"] = {"code": error.code}
        _finalize_matrix_report(
            report,
            config,
            budget,
            status="failed",
            stop_reason=error.code,
        )
        _progress_set(progress, report)
        return report
    except (KeyboardInterrupt, asyncio.CancelledError):
        budget.set_stop("cancelled_by_user")
        _finalize_matrix_report(
            report,
            config,
            budget,
            status="cancelled",
            stop_reason="cancelled_by_user",
        )
        _progress_set(progress, report)
        raise


def _run_matrix_child_accounted(
    config: Any,
    *,
    budget: _GuideLLMMatrixBudget,
    phase: Literal["warmup", "measurement"],
    condition_index: int,
    block_index: int | None,
    prompt_tokens: int,
    traffic: GuideLLMTraffic,
    limits: GuideLLMLimits,
    seed: int,
    executable: str | os.PathLike[str],
    environ: Mapping[str, str] | None,
) -> dict[str, Any]:
    """Run one child without turning a missing report into zero traffic."""

    envelope = budget.child_envelope(
        phase=phase,
        condition_index=condition_index,
        block_index=block_index,
        seed=seed,
        limits=limits,
        max_in_flight=traffic.max_concurrency,
    )
    try:
        imported = _run_matrix_child(
            config,
            prompt_tokens=prompt_tokens,
            traffic=traffic,
            limits=limits,
            seed=seed,
            executable=executable,
            environ=environ,
        )
        if imported["request_totals"]["total"] > limits.max_requests:
            raise GuideLLMBackendError("child_exceeded_request_limit")
        budget.record(imported)
        return imported
    except BaseException:
        # A process may have sent any traffic through its declared envelope
        # before timing out, being cancelled, exiting nonzero, or emitting a
        # report that fails closed at the schema boundary.
        budget.mark_child_incomplete(envelope)
        raise


def _run_matrix_child(
    config: Any,
    *,
    prompt_tokens: int,
    traffic: GuideLLMTraffic,
    limits: GuideLLMLimits,
    seed: int,
    executable: str | os.PathLike[str],
    environ: Mapping[str, str] | None,
) -> dict[str, Any]:
    revision = config.model_revision if config.model_revision != "unknown" else None
    spec = GuideLLMRunSpec(
        endpoint_url=config.endpoint.url,
        model=config.model,
        tokenizer_model=config.model,
        tokenizer_revision=revision,
        traffic=traffic,
        limits=limits,
        prompt_tokens=prompt_tokens,
        output_tokens=config.max_tokens,
        seed=seed,
        request_timeout_seconds=config.request_timeout_seconds,
        connect_timeout_seconds=min(10.0, config.request_timeout_seconds),
        allow_insecure_http=config.allow_insecure_http,
    )
    return _run_guidellm_verified(
        spec,
        api_key=config.endpoint.api_key,
        executable=executable,
        environ=environ,
    )


def _initial_matrix_report(
    config: Any,
    *,
    prompt_tokens: int,
    schema_version: str,
    artifact_type: str,
) -> dict[str, Any]:
    now = _utc_now()
    return {
        "schema_version": schema_version,
        "artifact_type": artifact_type,
        "generated_at": now,
        "started_at": now,
        "completed_at": None,
        "mode": config.mode,
        "status": "running",
        "accounting_incomplete": False,
        "decision_eligible": False,
        "strict_completion_validation": False,
        "golden_gate_eligible": False,
        "decision_ineligible_reasons": list(DECISION_INELIGIBLE_REASONS),
        "manifest": _matrix_manifest(config, prompt_tokens),
        "conditions": [],
        "best_tested": {
            "available": False,
            "state": "not_evaluated",
            "optimum_found": False,
        },
        "run_totals": {},
        "cost_summary": {
            "kind": config.cost.kind,
            "total_cost": None,
            "cost_per_million_output_tokens": None,
        },
        "stop_reason": None,
        "disclaimer": (
            "GuideLLM measurements use generated synthetic_text token shapes, not "
            "the supplied prompt JSONL. Completion strictness and response-size "
            "enforcement are unavailable, so this artifact is never decision-grade "
            "or a production recommendation."
        ),
    }


def _matrix_manifest(config: Any, prompt_tokens: int) -> dict[str, Any]:
    from . import __version__

    engine_flags = {name: value for name, value in config.engine_flags}
    measured_seeds = [
        config.seed + condition_index * 10_000 + block_index
        for condition_index, _ in enumerate(config.conditions)
        for block_index in range(config.blocks)
    ]
    warmup_seeds = (
        [
            config.seed + condition_index * 10_000 - 1
            for condition_index, _ in enumerate(config.conditions)
        ]
        if config.warmup_requests_per_condition
        else []
    )
    shape = {
        "generator": "guidellm-0.7.3-synthetic_text",
        "prompt_tokens": prompt_tokens,
        "output_tokens": config.max_tokens,
        "temperature": 0,
        "stream": True,
    }
    shape_hash = _canonical_hash(shape)
    return {
        "manifest_version": "1.0",
        "tool": {"name": "throttle-bench", "version": __version__},
        "engine": {
            "backend": "guidellm",
            "backend_version": GUIDELLM_VERSION,
            "http_client_version": None,
            "server_version": config.server_version,
            "effective_flags": engine_flags,
            "effective_flags_provenance": config.engine_flags_provenance,
        },
        "model": {
            "id": config.model,
            "immutable_revision": config.model_revision,
        },
        "runtime": {
            "image_digest": config.image_digest,
            "gpu": config.gpu,
            "gpu_fingerprint_sha256": hashlib.sha256(
                config.gpu_fingerprint.encode("utf-8")
            ).hexdigest(),
            "gpu_fingerprint_supplied": config.gpu_fingerprint != "unknown",
            "cuda_version": config.cuda_version,
            "driver_version": config.driver_version,
        },
        "workload": {
            "format_version": "guidellm-synthetic-text-v1",
            "source": "guidellm_synthetic_text",
            "seed": config.seed,
            "seed_derivation": (
                "measured=base+condition_index*10000+block_index; "
                "warmup=base+condition_index*10000-1"
            ),
            "synthetic_shape_sha256": shape_hash,
            "measured_sha256": _canonical_hash(
                {
                    "role": "measured",
                    "shape_sha256": shape_hash,
                    "seeds": measured_seeds,
                }
            ),
            "warmup_sha256": _canonical_hash(
                {"role": "warmup", "shape_sha256": shape_hash, "seeds": warmup_seeds}
            ),
            "measured_seed_count": len(measured_seeds),
            "warmup_seed_count": len(warmup_seeds),
            "measured_prompt_count": None,
            "warmup_prompt_count": None,
            "warmup_is_separate": bool(warmup_seeds),
            # Static-seed separation does not prove that GuideLLM's generated
            # warm-up text is disjoint from its measured synthetic text.
            "warmup_prompts_disjoint": False,
            "synthetic_prompt_tokens": prompt_tokens,
            "synthetic_output_tokens": config.max_tokens,
            "supplied_prompt_jsonl_used": False,
            "parity_with_supplied_prompt_jsonl": False,
            "parity_claim": "none",
            "cache_policy": config.cache_policy,
        },
        "request": {
            "type": "guidellm_synthetic_text_chat_completions",
            "temperature": 0,
            "max_tokens": config.max_tokens,
            "stop": None,
            "stream": True,
            "timeout_seconds": config.request_timeout_seconds,
        },
        "metric_definitions": {
            "e2e_latency": "GuideLLM client-observed request latency",
            "ttft": (
                "GuideLLM aggregate; 0.7.3 may zero-fill unavailable per-request "
                "stream timing, so availability is not proven"
            ),
            "tpot": (
                "GuideLLM 0.7.3 time_per_output_token; includes the first output "
                "token and is not Throttle's decode-only TPOT definition"
            ),
            "itl": (
                "GuideLLM aggregate; 0.7.3 may zero-fill unavailable inter-token "
                "timing, so availability is not proven"
            ),
            "throughput": (
                "GuideLLM successful output-token aggregate divided by measured "
                "block duration; token-usage provenance is not strict"
            ),
            "decision_throughput": (
                "arithmetic mean of repeated-block throughput with a Student-t "
                "block interval; still decision-ineligible because GuideLLM cannot "
                "prove strict response validity"
            ),
            "slo_goodput": (
                "unavailable because allowlisted aggregates cannot classify each "
                "request against the configured SLO"
            ),
        },
        "traffic": {
            "conditions": [item.public_dict() for item in config.conditions],
            "blocks": config.blocks,
            "requests_per_block": config.requests_per_block,
            "block_duration_seconds": config.block_duration_seconds,
            "warmup_requests_per_condition": config.warmup_requests_per_condition,
            "p95_slo_ms": config.p95_slo_ms,
            "ttft_slo_ms": config.ttft_slo_ms,
            "open_loop_rate_relative_tolerance": OPEN_LOOP_RATE_RELATIVE_TOLERANCE,
            "open_loop_scheduler_lag_interval_tolerance": OPEN_LOOP_SCHEDULER_LAG_INTERVAL_TOLERANCE,
        },
        "provenance": {
            "evidence_source": config.evidence_source,
            "variant": config.variant,
            "sequence_position": config.sequence_position,
        },
        "cost": config.cost.public_dict(config.limits.max_elapsed_seconds),
        "safety": {
            "limits": config.limits.public_dict(),
            "strict_completion_validation": False,
            "max_response_bytes_enforced": False,
            "overrides": {
                "insecure_http": config.allow_insecure_http,
                "unknown_cost_acknowledged": config.allow_unknown_cost,
                "guidellm_validation_gaps_acknowledged": config.guidellm_gaps_acknowledged,
            },
            "ambient_proxy_environment_used": False,
            "redirects_followed": False,
        },
        "optimization_credit_exclusions": (
            [
                {
                    "feature": "enable_chunked_prefill",
                    "reason": (
                        "vLLM V1 enables chunked prefill by default when possible; "
                        "its presence alone receives no optimization credit"
                    ),
                }
            ]
            if any(
                name.replace("_", "-").lower() == "enable-chunked-prefill"
                for name in engine_flags
            )
            else []
        ),
    }


def _canonical_block(
    imported: Mapping[str, Any],
    *,
    index: int,
    seed: int,
    requested_requests: int | None,
    requested_duration: float | None,
    declared_peak_in_flight_cap: int,
) -> dict[str, Any]:
    totals = imported["request_totals"]
    attempted = int(totals["total"])
    failures = int(totals["errored"]) + int(totals["incomplete"])
    request_floor = requested_requests is not None and attempted >= requested_requests
    duration_floor = (
        requested_duration is not None
        and float(imported["duration_seconds"]) >= requested_duration
    )
    completed = request_floor or duration_floor
    valid = failures == 0 and attempted > 0 and completed
    reasons: list[str] = []
    if failures:
        reasons.append("guidellm_reported_error_or_incomplete_request")
    if attempted == 0:
        reasons.append("no_requests_measured")
    if not completed:
        reasons.append("configured_block_floor_not_reached")
    metrics = _canonical_metrics(imported)
    return {
        "block_index": index,
        "seed": seed,
        "valid": valid,
        "strict_completion_validation": False,
        "decision_eligible": False,
        "decision_ineligible_reasons": list(DECISION_INELIGIBLE_REASONS),
        "invalid_reasons": reasons,
        "wall_duration_seconds": imported["duration_seconds"],
        "request_counts": _canonical_request_counts(imported),
        "offered_requests": attempted,
        "target_offered_request_rate": (
            float(imported["traffic"]["requests_per_second"])
            if imported.get("traffic", {}).get("kind") == "constant"
            else None
        ),
        "launch_window_seconds": None,
        "achieved_offered_request_rate": None,
        "offered_rate_relative_error": None,
        "scheduler_lag_interval_ratio_p95": None,
        "open_loop_target_achieved": (
            False if imported.get("traffic", {}).get("kind") == "constant" else True
        ),
        "declared_peak_in_flight_cap": declared_peak_in_flight_cap,
        "observed_peak_in_flight": None,
        "scheduler_lag_ms": {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "source": "unavailable_from_allowlisted_guidellm_aggregates",
        },
        "metrics": metrics if valid else None,
        "diagnostic_metrics": metrics,
    }


def _canonical_condition(
    config: Any, condition: Any, entry: Mapping[str, Any]
) -> dict[str, Any]:
    blocks = list(entry["blocks"])
    warmup = dict(entry["warmup"])
    valid = (
        len(blocks) == config.blocks
        and bool(blocks)
        and all(block["valid"] for block in blocks)
        and warmup.get("invalid", 0) == 0
    )
    attempted = sum(int(block["request_counts"]["attempted"]) for block in blocks)
    successful = sum(int(block["request_counts"]["valid"]) for block in blocks)
    failures = attempted - successful
    wall = _bounded_derived(
        math.fsum(float(block["wall_duration_seconds"]) for block in blocks)
    )
    completion_tokens = sum(
        int(block["diagnostic_metrics"]["completion_tokens"]) for block in blocks
    )
    prompt_total = sum(
        int(block["diagnostic_metrics"]["prompt_tokens"]) for block in blocks
    )
    if (
        completion_tokens > _MAX_SAFE_NUMERIC_MAGNITUDE
        or prompt_total > _MAX_SAFE_NUMERIC_MAGNITUDE
    ):
        raise GuideLLMBackendError("invalid_report_number")
    metrics: dict[str, Any] | None = None
    if valid:
        block_rates = [
            float(block["metrics"]["output_tokens_per_second"]) for block in blocks
        ]
        request_rates = [
            float(block["metrics"]["requests_per_second"]) for block in blocks
        ]
        metrics = {
            "valid_response_count": successful,
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_total,
            "requests_per_second": (
                _bounded_ratio(successful, wall) if wall > 0 else None
            ),
            "output_tokens_per_second": (
                _bounded_ratio(completion_tokens, wall) if wall > 0 else None
            ),
            "block_mean_output_tokens_per_second": _bounded_ratio(
                math.fsum(block_rates), len(block_rates)
            ),
            "block_mean_output_tokens_per_second_ci": _t_interval(block_rates),
            "block_mean_requests_per_second": _bounded_ratio(
                math.fsum(request_rates), len(request_rates)
            ),
            "block_mean_requests_per_second_ci": _t_interval(request_rates),
            "error_rate": (_bounded_ratio(failures, attempted) if attempted else None),
            "e2e_latency_ms": _merge_block_distributions(blocks, "e2e_latency_ms"),
            "ttft_ms": _merge_block_distributions(blocks, "ttft_ms"),
            "tpot_ms": _merge_block_distributions(blocks, "tpot_ms"),
            "itl_ms": _merge_block_distributions(blocks, "itl_ms"),
            "inter_chunk_latency_ms": {
                "count": 0,
                "mean": None,
                "p50": None,
                "p95": None,
                "source": "unavailable_from_guidellm_aggregate_contract",
            },
            "slo_goodput": None,
            "slo_goodput_unavailable_reason": (
                "allowlisted_guidellm_aggregates_do_not_support_per_request_slo_classification"
            ),
            "independent_ci_unit": "repeated_block",
            "cost_per_million_output_tokens": None,
            "cost_metric_basis": (
                "guidellm process overhead cannot be safely allocated to a condition"
            ),
        }
    reasons = list(DECISION_INELIGIBLE_REASONS)
    if config.mode == "smoke":
        reasons.append("smoke_mode_is_not_decision_grade")
    if not valid:
        reasons.append("invalid_or_incomplete_block")
    return {
        "condition": condition.public_dict(),
        "valid": valid,
        "decision_grade": False,
        "strict_completion_validation": False,
        "decision_ineligible_reasons": reasons,
        "qualification_floor": {
            "minimum_valid_requests": 200,
            "or_minimum_measured_seconds": 60.0,
            "minimum_blocks": 3,
            "gate_status": "unavailable_due_to_backend_validation_gaps",
        },
        "warmup": warmup,
        "blocks": blocks,
        "request_counts": {
            "attempted": attempted,
            "valid": successful,
            "invalid": failures,
            "status_counts": {},
            "error_counts": (
                {"guidellm_reported_error_or_incomplete": failures} if failures else {}
            ),
            "finish_reason_counts": {},
            "validation_basis": "guidellm_success_classification_only",
        },
        "measured_wall_seconds": wall,
        "target_offered_request_rate": (
            condition.value if condition.kind == "open_loop" else None
        ),
        "achieved_offered_request_rate": None,
        "offered_rate_relative_error": None,
        "open_loop_target_achieved": (False if condition.kind == "open_loop" else None),
        "declared_peak_in_flight_cap": condition.max_in_flight,
        "observed_peak_in_flight": None,
        "metrics": metrics,
        "diagnostic_metrics": metrics or _empty_canonical_metrics(),
    }


def _canonical_request_counts(imported: Mapping[str, Any]) -> dict[str, Any]:
    totals = imported["request_totals"]
    attempted = int(totals["total"])
    successful = int(totals["successful"])
    invalid = int(totals["errored"]) + int(totals["incomplete"])
    return {
        "attempted": attempted,
        "valid": successful,
        "invalid": invalid,
        "status_counts": {},
        "error_counts": (
            {"guidellm_reported_error_or_incomplete": invalid} if invalid else {}
        ),
        "finish_reason_counts": {},
        "validation_basis": "guidellm_success_classification_only",
        "strict_completion_validation": False,
    }


def _empty_request_counts() -> dict[str, Any]:
    return {
        "attempted": 0,
        "valid": 0,
        "invalid": 0,
        "status_counts": {},
        "error_counts": {},
        "finish_reason_counts": {},
        "validation_basis": "guidellm_success_classification_only",
        "strict_completion_validation": False,
    }


def _canonical_metrics(imported: Mapping[str, Any]) -> dict[str, Any]:
    summaries = imported["metrics"]
    duration = float(imported["duration_seconds"])
    successful = int(imported["request_totals"]["successful"])
    attempted = int(imported["request_totals"]["total"])
    output_tokens = _integral_total(summaries["output_token_count"]["total_sum"])
    prompt_tokens = _integral_total(summaries["prompt_token_count"]["total_sum"])
    return {
        "valid_response_count": successful,
        "completion_tokens": output_tokens,
        "prompt_tokens": prompt_tokens,
        "requests_per_second": (
            _bounded_ratio(successful, duration) if duration > 0 else None
        ),
        "output_tokens_per_second": (
            _bounded_ratio(output_tokens, duration) if duration > 0 else None
        ),
        "error_rate": (
            _bounded_ratio(attempted - successful, attempted) if attempted else None
        ),
        "e2e_latency_ms": _canonical_distribution(
            summaries["request_latency"], scale=1_000.0
        ),
        "ttft_ms": {
            **_canonical_distribution(summaries["time_to_first_token_ms"], scale=1.0),
            "availability_proven": False,
            "availability_caveat": (
                "guidellm_0_7_3_may_zero_fill_missing_stream_timing"
            ),
        },
        "tpot_ms": {
            **_canonical_distribution(summaries["time_per_output_token_ms"], scale=1.0),
            "definition": ("guidellm_0_7_3_tpot_includes_first_output_token"),
        },
        "itl_ms": {
            **_canonical_distribution(summaries["inter_token_latency_ms"], scale=1.0),
            "availability_proven": False,
            "availability_caveat": (
                "guidellm_0_7_3_may_zero_fill_missing_stream_timing"
            ),
        },
        "inter_chunk_latency_ms": {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "source": "unavailable_from_guidellm_aggregate_contract",
        },
        "slo_goodput": None,
        "slo_goodput_unavailable_reason": (
            "allowlisted_guidellm_aggregates_do_not_support_per_request_slo_classification"
        ),
    }


def _canonical_distribution(
    summary: Mapping[str, Any], *, scale: float
) -> dict[str, Any]:
    percentiles = summary["percentiles"]
    scaled = {
        name: _bounded_derived(float(value) * scale)
        for name, value in {
            "mean": summary["mean"],
            "p50": percentiles["p50"],
            "p90": percentiles["p90"],
            "p95": percentiles["p95"],
            "p99": percentiles["p99"],
        }.items()
    }
    return {
        "count": int(summary["count"]),
        **scaled,
        "p95_ci": {
            "low": None,
            "high": None,
            "confidence": 0.95,
            "method": "unavailable_from_aggregate_only_report",
            "n": int(summary["count"]),
        },
        "source": "guidellm_0_7_3_allowlisted_aggregate",
    }


def _merge_block_distributions(
    blocks: list[Mapping[str, Any]], name: str
) -> dict[str, Any]:
    values = [block["diagnostic_metrics"][name] for block in blocks]
    count = sum(int(item["count"]) for item in values)
    if count > _MAX_SAFE_NUMERIC_MAGNITUDE:
        raise GuideLLMBackendError("invalid_report_number")
    weighted = [
        _bounded_derived(float(item["mean"]) * int(item["count"])) for item in values
    ]
    mean = _bounded_derived(math.fsum(weighted) / count) if count else None
    merged = {
        "count": count,
        "mean": mean,
        "p50": None,
        "p90": None,
        "p95": None,
        "p99": None,
        "p95_ci": {
            "low": None,
            "high": None,
            "confidence": 0.95,
            "method": "unavailable_without_request_samples",
            "n": count,
        },
        "source": "merged_guidellm_block_aggregates",
        "percentiles_unavailable_reason": (
            "percentiles_cannot_be_merged_from_block_summaries"
        ),
    }
    if name in {"ttft_ms", "itl_ms"}:
        merged.update(
            availability_proven=False,
            availability_caveat=("guidellm_0_7_3_may_zero_fill_missing_stream_timing"),
        )
    return merged


def _empty_canonical_metrics() -> dict[str, Any]:
    return {
        "valid_response_count": 0,
        "completion_tokens": 0,
        "prompt_tokens": 0,
        "requests_per_second": None,
        "output_tokens_per_second": None,
        "error_rate": None,
        "slo_goodput": None,
    }


def _finalize_matrix_report(
    report: dict[str, Any],
    config: Any,
    budget: _GuideLLMMatrixBudget,
    *,
    status: str,
    stop_reason: str | None,
) -> None:
    final_elapsed = budget.elapsed()
    if budget.stop_reason is None:
        if final_elapsed >= config.limits.max_elapsed_seconds:
            budget.set_stop("max_elapsed_time")
        else:
            spend = config.cost.elapsed_estimate(final_elapsed)
            if spend is not None and spend >= config.limits.max_estimated_spend:
                budget.set_stop("max_estimated_spend")
    if status == "complete" and budget.stop_reason is not None:
        status = "stopped"
    if stop_reason is None:
        stop_reason = budget.stop_reason
    report["status"] = status
    report["stop_reason"] = stop_reason
    report["completed_at"] = _utc_now()
    report["accounting_incomplete"] = budget.accounting_incomplete
    report["run_totals"] = budget.public_dict(elapsed_seconds=final_elapsed)
    report["decision_eligible"] = False
    report["strict_completion_validation"] = False
    report["golden_gate_eligible"] = False
    full_matrix_completed = (
        status == "complete"
        and len(report["conditions"]) == len(config.conditions)
        and all(
            isinstance(item.get("blocks"), list)
            and len(item["blocks"]) == config.blocks
            for item in report["conditions"]
        )
    )
    if budget.accounting_incomplete:
        report["best_tested"] = {
            "field": (
                "best_tested_concurrency"
                if config.conditions[0].kind == "closed_loop"
                else "best_tested_request_rate"
            ),
            "available": False,
            "state": "inconclusive",
            "reason": "accounting_incomplete_after_child_failure",
            "optimum_found": False,
        }
    elif not full_matrix_completed:
        report["best_tested"] = {
            "field": (
                "best_tested_concurrency"
                if config.conditions[0].kind == "closed_loop"
                else "best_tested_request_rate"
            ),
            "available": False,
            "state": "inconclusive",
            "reason": "partial_or_failed_run",
            "optimum_found": False,
        }
    else:
        report["best_tested"] = _matrix_best_tested(report["conditions"], config)
    completion_tokens = sum(
        int(item.get("diagnostic_metrics", {}).get("completion_tokens", 0))
        for item in report["conditions"]
    )
    total_cost, basis = config.cost.final_cost(final_elapsed)
    report["cost_summary"] = {
        "kind": config.cost.kind,
        "total_cost": total_cost,
        "basis": basis,
        "completion_tokens": completion_tokens,
        "cost_per_million_output_tokens": (
            total_cost / completion_tokens * 1_000_000.0
            if total_cost is not None and completion_tokens > 0
            else None
        ),
    }


def _matrix_best_tested(
    conditions: list[Mapping[str, Any]], config: Any
) -> dict[str, Any]:
    field = (
        "best_tested_concurrency"
        if config.conditions[0].kind == "closed_loop"
        else "best_tested_request_rate"
    )
    valid = [item for item in conditions if item.get("valid") and item.get("metrics")]
    if not valid:
        return {
            "field": field,
            "available": False,
            "state": "invalid",
            "reason": "no_operationally_complete_conditions",
            "optimum_found": False,
        }
    if config.p95_slo_ms is not None or config.ttft_slo_ms is not None:
        return {
            "field": field,
            "available": False,
            "state": "inconclusive",
            "reason": "guidellm_aggregate_contract_cannot_calculate_slo_goodput",
            "optimum_found": False,
        }
    selected = max(
        valid,
        key=lambda item: float(item["metrics"]["block_mean_output_tokens_per_second"]),
    )
    value = float(selected["condition"]["value"])
    boundary = value == max(float(item["condition"]["value"]) for item in conditions)
    return {
        "field": field,
        "available": True,
        "value": int(value) if value.is_integer() else value,
        "condition_id": selected["condition"]["id"],
        "block_mean_output_tokens_per_second": selected["metrics"][
            "block_mean_output_tokens_per_second"
        ],
        "block_mean_output_tokens_per_second_ci": selected["metrics"][
            "block_mean_output_tokens_per_second_ci"
        ],
        "pooled_output_tokens_per_second": selected["metrics"][
            "output_tokens_per_second"
        ],
        "state": ("not_applicable_smoke" if config.mode == "smoke" else "inconclusive"),
        "reasons": list(DECISION_INELIGIBLE_REASONS),
        "boundary_reached": boundary,
        "optimum_found": False,
        "claim": (
            "descriptive GuideLLM synthetic_text observation only; never a "
            "production recommendation or golden-gate result"
        ),
    }


def _cost_rate_per_wall_second(config: Any) -> float | None:
    if config.cost.kind == "dedicated_hourly":
        return float(config.cost.total_hourly_rate) / 3600.0
    if config.cost.kind == "serverless_active_seconds":
        return float(config.cost.active_second_rate) * int(
            config.cost.max_active_workers
        )
    return None


def _integral_total(value: Any) -> int:
    parsed = _nonnegative_number(value)
    if not float(parsed).is_integer():
        raise GuideLLMBackendError("invalid_report_number")
    return int(parsed)


def _t_interval(values: list[float]) -> dict[str, Any]:
    from .statistics import t_interval_95

    return t_interval_95(values)


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _progress_set(progress: Any, report: Mapping[str, Any]) -> None:
    if progress is not None:
        progress.set(report)


def _run_guidellm_verified(
    spec: GuideLLMRunSpec,
    *,
    api_key: str,
    executable: str | os.PathLike[str],
    environ: Mapping[str, str] | None,
) -> dict[str, Any]:
    """Run one child after the caller has verified the executable version."""

    spec.validate()
    if not _nonempty_text(api_key):
        raise GuideLLMBackendError("invalid_api_key")
    child_env = _sanitized_child_env(environ, api_key=api_key)

    try:
        with tempfile.TemporaryDirectory(prefix="throttle-guidellm-run-") as name:
            workdir = Path(name)
            os.chmod(workdir, 0o700)
            scenario_path = workdir / "scenario.json"
            raw_report_path = workdir / "report.json"
            scenario = build_guidellm_scenario(spec, raw_report_path)
            _write_private_json(scenario_path, scenario)
            _create_private_file(raw_report_path)

            argv = [
                os.fspath(executable),
                "run",
                "--config",
                os.fspath(scenario_path),
                "--disable-console",
            ]
            _run_benchmark_process(
                argv,
                child_env,
                workdir,
                timeout_seconds=spec.limits.wall_timeout_seconds,
            )
            imported = parse_guidellm_report(raw_report_path)
            _validate_imported_limits(imported, spec)
    except GuideLLMBackendError:
        raise
    except (OSError, ValueError):
        raise GuideLLMBackendError("runner_failed") from None

    imported["traffic"] = spec.traffic.public_dict()
    return imported


def parse_guidellm_report(
    report_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Import only the allowlisted numeric contract from a 0.7.3 JSON report."""

    document = _load_private_json(Path(report_path))
    root = _mapping(document)
    metadata = _mapping(root.get("metadata"))
    if _strict_int(metadata.get("version")) != GUIDELLM_REPORT_VERSION:
        raise GuideLLMBackendError("unsupported_report_schema")
    if metadata.get("guidellm_version") != GUIDELLM_VERSION:
        raise GuideLLMBackendError("unsupported_report_version")

    benchmarks = root.get("benchmarks")
    if not isinstance(benchmarks, list) or len(benchmarks) != 1:
        raise GuideLLMBackendError("invalid_report_shape")
    benchmark = _mapping(benchmarks[0])
    if benchmark.get("type_") != "generative_benchmark":
        raise GuideLLMBackendError("invalid_report_shape")

    duration = _nonnegative_number(benchmark.get("duration"))
    metrics = _mapping(benchmark.get("metrics"))
    request_totals = _parse_request_totals(metrics.get("request_totals"))
    if request_totals["total"] > 0 and duration <= 0:
        raise GuideLLMBackendError("invalid_report_duration")

    scheduler = _mapping(benchmark.get("scheduler_metrics"))
    scheduler_totals = _parse_request_totals(scheduler.get("requests_made"))
    if scheduler_totals != request_totals:
        raise GuideLLMBackendError("inconsistent_report_totals")

    summaries = {
        name: _parse_summary(_mapping(metrics.get(name)).get("successful"))
        for name in _METRIC_FIELDS
    }
    # GuideLLM's latency/request/token-count distributions contain one value
    # per request. TPOT and token-rate distributions contain one value per
    # output token, while ITL contains one value per inter-token interval.
    # Conflating those cardinalities would reject valid 0.7.3 reports.
    for name in _REQUEST_COUNT_METRICS:
        if summaries[name]["count"] != request_totals["successful"]:
            raise GuideLLMBackendError("inconsistent_report_totals")
    _integral_total(summaries["prompt_token_count"]["total_sum"])
    _integral_total(summaries["output_token_count"]["total_sum"])

    total = request_totals["total"]
    failures = request_totals["errored"] + request_totals["incomplete"]
    success_rate = _bounded_ratio(request_totals["successful"], total) if total else 0.0
    error_rate = _bounded_ratio(failures, total) if total else 0.0

    return {
        "backend": "guidellm",
        "guidellm_version": GUIDELLM_VERSION,
        "guidellm_report_version": GUIDELLM_REPORT_VERSION,
        "strict_completion_validation": False,
        "decision_eligible": False,
        "golden_gate_eligible": False,
        "decision_ineligible_reasons": list(DECISION_INELIGIBLE_REASONS),
        "duration_seconds": duration,
        "request_totals": request_totals,
        "success_rate": success_rate,
        "error_rate": error_rate,
        "metrics": summaries,
    }


def _validate_imported_limits(
    imported: Mapping[str, Any], spec: GuideLLMRunSpec
) -> None:
    successful = int(imported["request_totals"]["successful"])
    attempted = int(imported["request_totals"]["total"])
    if attempted > spec.limits.max_requests:
        raise GuideLLMBackendError("child_exceeded_request_limit")
    output = imported["metrics"]["output_token_count"]
    output_total = _integral_total(output["total_sum"])
    if (
        float(output["max"]) > spec.output_tokens
        or output_total > successful * spec.output_tokens
    ):
        raise GuideLLMBackendError("child_exceeded_token_limit")


def _run_version_process(
    executable: str,
    env: Mapping[str, str],
    cwd: Path,
    timeout_seconds: float,
) -> bytes:
    try:
        process = subprocess.Popen(
            [executable, "--version"],
            shell=False,
            cwd=cwd,
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=(os.name == "posix"),
        )
    except OSError:
        raise GuideLLMBackendError("version_check_failed") from None
    if process.stdout is None:  # pragma: no cover - defensive subprocess contract
        _terminate_process_group(process)
        raise GuideLLMBackendError("version_check_failed")
    output = bytearray()
    overflow = threading.Event()

    def bounded_reader() -> None:
        while True:
            chunk = process.stdout.read(1024)
            if not chunk:
                return
            if len(output) + len(chunk) > _MAX_VERSION_OUTPUT_BYTES:
                overflow.set()
                return
            output.extend(chunk)

    reader = threading.Thread(target=bounded_reader, daemon=True)
    reader.start()
    deadline = time.monotonic() + timeout_seconds
    try:
        while process.poll() is None:
            if overflow.is_set():
                _terminate_process_group(process)
                raise GuideLLMBackendError("version_check_failed")
            if time.monotonic() >= deadline:
                _terminate_process_group(process)
                raise GuideLLMBackendError("version_check_timeout")
            time.sleep(0.01)
    except BaseException:
        _cleanup_process_group(process)
        reader.join(timeout=_PROCESS_EXIT_GRACE_SECONDS)
        process.stdout.close()
        raise
    # The parent may exit while a forked worker keeps the stdout pipe and the
    # process group alive.  Clean the group before joining the reader so no
    # descendant can outlive Throttle's wall-clock and cancellation controls.
    _cleanup_process_group(process)
    reader.join(timeout=_PROCESS_EXIT_GRACE_SECONDS)
    reader_stuck = reader.is_alive()
    process.stdout.close()
    if reader_stuck or overflow.is_set():
        raise GuideLLMBackendError("version_check_failed")
    if process.returncode != 0:
        raise GuideLLMBackendError("version_check_failed")
    return bytes(output)


def _run_benchmark_process(
    argv: list[str],
    env: Mapping[str, str],
    cwd: Path,
    *,
    timeout_seconds: float,
) -> None:
    try:
        process = subprocess.Popen(
            argv,
            shell=False,
            cwd=cwd,
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=(os.name == "posix"),
        )
    except OSError:
        raise GuideLLMBackendError("process_start_failed") from None
    try:
        process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        _terminate_process_group(process)
        raise GuideLLMBackendError("process_timeout") from None
    except BaseException:
        _terminate_process_group(process)
        raise
    # GuideLLM (or a substituted executable on PATH) may fork a worker and
    # then let its parent exit.  Always tear down the isolated POSIX process
    # group before accepting either a zero or non-zero parent status.
    _cleanup_process_group(process)
    if process.returncode != 0:
        raise GuideLLMBackendError("process_failed")


def _cleanup_process_group(process: subprocess.Popen[bytes]) -> None:
    """Ensure no subprocess descendant survives a completed parent."""

    if os.name == "posix" or process.poll() is None:
        _terminate_process_group(process)


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    """Terminate the child and all workers it may have started."""

    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return
    else:  # pragma: no cover - exercised on Windows only
        process.terminate()
    try:
        process.wait(timeout=_PROCESS_EXIT_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass
    else:
        # A worker can ignore SIGTERM even after its parent exits. Probe and
        # kill the whole group below instead of assuming parent exit means all
        # descendants are gone.
        if os.name != "posix":  # pragma: no cover - exercised on Windows only
            return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            return
    else:  # pragma: no cover - exercised on Windows only
        process.kill()
    try:
        process.wait(timeout=_PROCESS_EXIT_GRACE_SECONDS)
    except subprocess.TimeoutExpired:  # pragma: no cover - OS-level failure
        pass


def _sanitized_child_env(
    environ: Mapping[str, str] | None, *, api_key: str | None = None
) -> dict[str, str]:
    source = os.environ if environ is None else environ
    clean: dict[str, str] = {}
    hugging_face_secret_names = {
        "hf_token",
        "hugging_face_hub_token",
        "huggingface_hub_token",
        "huggingfacehub_api_token",
    }
    ambient_tls_override_names = {
        "ssl_cert_file",
        "ssl_cert_dir",
        "sslkeylogfile",
        "requests_ca_bundle",
        "curl_ca_bundle",
    }
    for key, value in source.items():
        folded = key.casefold()
        compact = re.sub(r"[^a-z0-9]", "", folded)
        credential_named = (
            any(
                marker in folded
                for marker in (
                    "secret",
                    "password",
                    "passwd",
                    "credential",
                    "authorization",
                    "cookie",
                    "session_token",
                )
            )
            or "token" in folded
            or "apikey" in compact
            or "accesskey" in compact
            or folded.endswith("_key")
            or "_key_" in folded
            or folded.startswith("key_")
            or "keylog" in compact
        )
        if (
            folded.startswith("guidellm")
            or folded.endswith("_proxy")
            or folded in hugging_face_secret_names
            or folded in ambient_tls_override_names
            or credential_named
        ):
            continue
        clean[str(key)] = str(value)
    clean["GUIDELLM__LOGGING__DISABLED"] = "true"
    clean["HF_HUB_OFFLINE"] = "1"
    clean["TRANSFORMERS_OFFLINE"] = "1"
    clean["HF_DATASETS_OFFLINE"] = "1"
    if api_key is not None:
        clean[GUIDELLM_API_KEY_ENV] = api_key
    return clean


def _normalized_endpoint(value: str, *, allow_insecure_http: bool = False) -> str:
    if (
        not _nonempty_text(value)
        or value != value.strip()
        or any(ord(character) < 32 for character in value)
        or not isinstance(allow_insecure_http, bool)
    ):
        raise GuideLLMBackendError("invalid_endpoint")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise GuideLLMBackendError("invalid_endpoint") from None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.hostname.rstrip(".").casefold() in {"0.0.0.0", "::", "*"}
    ):
        raise GuideLLMBackendError("invalid_endpoint")
    if (
        parsed.scheme == "http"
        and not _is_loopback_host(parsed.hostname)
        and not allow_insecure_http
    ):
        raise GuideLLMBackendError("insecure_endpoint")

    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = host if port is None else f"{host}:{port}"
    path = parsed.path
    # GuideLLM 0.7.3 strips only a trailing /v1 from its target and then
    # appends v1/chat/completions. Restrict inputs to the three conventional
    # endpoint forms whose native route is provably identical. Custom prefixes
    # are rejected rather than silently redirected to a different path.
    if path not in {
        "",
        "/",
        "/v1",
        "/v1/",
        "/v1/chat/completions",
        "/v1/chat/completions/",
    }:
        raise GuideLLMBackendError("unsupported_guidellm_route")
    return urlunsplit((parsed.scheme, netloc, "", "", ""))


def _is_loopback_host(host: str) -> bool:
    if host.rstrip(".").casefold() == "localhost":
        return True
    try:
        # IPv6 zone identifiers are valid in URLs but not accepted by
        # ipaddress; the address portion is sufficient for loopback checking.
        return ipaddress.ip_address(host.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def _write_private_json(path: Path, value: Mapping[str, Any]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")


def _create_private_file(path: Path) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)


def _load_private_json(path: Path) -> Any:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise GuideLLMBackendError("missing_report") from None
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode):
            raise GuideLLMBackendError("invalid_report_file")
        if details.st_size <= 0 or details.st_size > _MAX_RAW_REPORT_BYTES:
            raise GuideLLMBackendError("invalid_report_size")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read(_MAX_RAW_REPORT_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(payload) > _MAX_RAW_REPORT_BYTES:
        raise GuideLLMBackendError("invalid_report_size")
    try:
        return json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        GuideLLMBackendError,
        ValueError,
        RecursionError,
    ):
        raise GuideLLMBackendError("invalid_report_json") from None


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GuideLLMBackendError("duplicate_report_key")
        result[key] = value
    return result


def _reject_json_constant(_: str) -> None:
    raise GuideLLMBackendError("nonfinite_report_number")


def _parse_request_totals(value: Any) -> dict[str, int]:
    source = _mapping(value)
    parsed = {
        name: _nonnegative_int(source.get(name)) for name in _REQUEST_TOTAL_FIELDS
    }
    if (
        parsed["successful"] + parsed["errored"] + parsed["incomplete"]
        != parsed["total"]
    ):
        raise GuideLLMBackendError("inconsistent_report_totals")
    return parsed


def _parse_summary(value: Any) -> dict[str, Any]:
    source = _mapping(value)
    parsed: dict[str, Any] = {}
    for name in _SUMMARY_FIELDS:
        if name == "count":
            parsed[name] = _nonnegative_int(source.get(name))
        else:
            parsed[name] = _nonnegative_number(source.get(name))
    percentiles = _mapping(source.get("percentiles"))
    parsed["percentiles"] = {
        name: _nonnegative_number(percentiles.get(name)) for name in _PERCENTILE_FIELDS
    }
    return parsed


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise GuideLLMBackendError("invalid_report_shape")
    return value


def _strict_int(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise GuideLLMBackendError("invalid_report_number")
    return value


def _nonnegative_int(value: Any) -> int:
    parsed = _strict_int(value)
    if parsed < 0 or parsed > _MAX_SAFE_NUMERIC_MAGNITUDE:
        raise GuideLLMBackendError("invalid_report_number")
    return parsed


def _nonnegative_number(value: Any) -> int | float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise GuideLLMBackendError("invalid_report_number")
    if not math.isfinite(value) or value < 0 or value > _MAX_SAFE_NUMERIC_MAGNITUDE:
        raise GuideLLMBackendError("invalid_report_number")
    return value


def _bounded_derived(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        raise GuideLLMBackendError("invalid_report_number") from None
    if not math.isfinite(parsed) or abs(parsed) > _MAX_SAFE_NUMERIC_MAGNITUDE:
        raise GuideLLMBackendError("invalid_report_number")
    return parsed


def _bounded_ratio(numerator: Any, denominator: Any) -> float:
    """Divide imported aggregates without allowing overflow or non-finite output."""

    try:
        result = float(numerator) / float(denominator)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        raise GuideLLMBackendError("invalid_report_number") from None
    return _bounded_derived(result)


def _require_positive_int(value: Any) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise GuideLLMBackendError("invalid_run_spec")


def _require_positive_number(value: Any) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise GuideLLMBackendError("invalid_run_spec")


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and "\x00" not in value
