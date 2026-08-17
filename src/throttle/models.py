"""Validated configuration contracts for Throttle v2.

Connection details live in :class:`EndpointConfig` and are never serialized.
The public report is assembled from the remaining, non-secret fields.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

CostKind = Literal[
    "unknown",
    "dedicated_hourly",
    "serverless_active_seconds",
    "user_supplied",
]
TrafficKind = Literal["closed_loop", "open_loop"]
RunMode = Literal["smoke", "benchmark"]
BackendKind = Literal["native", "guidellm"]
OPEN_LOOP_RATE_RELATIVE_TOLERANCE = 0.05
OPEN_LOOP_SCHEDULER_LAG_INTERVAL_TOLERANCE = 1.0


def _positive_finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value > 0


@dataclass(frozen=True)
class CostModel:
    """One explicit billing model; values from different models never combine."""

    kind: CostKind = "unknown"
    currency: str = "USD"
    total_hourly_rate: float | None = None
    active_second_rate: float | None = None
    max_active_workers: int | None = None
    billed_active_seconds: float | None = None
    user_supplied_total: float | None = None
    gpu_count: int | None = None

    def validate(self) -> None:
        if self.currency != "USD":
            raise ValueError("only USD cost inputs are currently supported")
        supplied = {
            "total_hourly_rate": self.total_hourly_rate,
            "active_second_rate": self.active_second_rate,
            "max_active_workers": self.max_active_workers,
            "billed_active_seconds": self.billed_active_seconds,
            "user_supplied_total": self.user_supplied_total,
            "gpu_count": self.gpu_count,
        }
        if self.kind == "unknown":
            if any(value is not None for value in supplied.values()):
                raise ValueError("unknown billing cannot include price or GPU fields")
            return
        if self.kind == "dedicated_hourly":
            if not _positive_finite(self.total_hourly_rate):
                raise ValueError(
                    "dedicated_hourly requires a positive total hourly rate"
                )
            if (
                not isinstance(self.gpu_count, int)
                or isinstance(self.gpu_count, bool)
                or self.gpu_count <= 0
            ):
                raise ValueError("dedicated_hourly requires a positive GPU count")
            forbidden = (
                self.active_second_rate,
                self.max_active_workers,
                self.billed_active_seconds,
                self.user_supplied_total,
            )
            if any(value is not None for value in forbidden):
                raise ValueError(
                    "dedicated_hourly cannot include another billing model"
                )
            return
        if self.kind == "serverless_active_seconds":
            if not _positive_finite(self.active_second_rate):
                raise ValueError(
                    "serverless billing requires a positive active-second rate"
                )
            if (
                not isinstance(self.max_active_workers, int)
                or isinstance(self.max_active_workers, bool)
                or self.max_active_workers <= 0
            ):
                raise ValueError(
                    "serverless billing requires positive max_active_workers"
                )
            if self.billed_active_seconds is not None and (
                not math.isfinite(self.billed_active_seconds)
                or self.billed_active_seconds < 0
            ):
                raise ValueError(
                    "billed_active_seconds must be non-negative and finite"
                )
            if any(
                value is not None
                for value in (
                    self.total_hourly_rate,
                    self.user_supplied_total,
                    self.gpu_count,
                )
            ):
                raise ValueError(
                    "serverless billing cannot include another billing model"
                )
            return
        if self.kind == "user_supplied":
            if not _positive_finite(self.user_supplied_total):
                raise ValueError("user_supplied billing requires a positive run total")
            if any(
                value is not None
                for value in (
                    self.total_hourly_rate,
                    self.active_second_rate,
                    self.max_active_workers,
                    self.billed_active_seconds,
                    self.gpu_count,
                )
            ):
                raise ValueError(
                    "user_supplied billing cannot include another billing model"
                )
            return
        raise ValueError(f"unsupported cost model: {self.kind}")

    def estimated_upper_bound(self, max_elapsed_seconds: float) -> float | None:
        """Return a conservative pre-run ceiling, or ``None`` if unknowable."""

        if self.kind == "dedicated_hourly":
            return float(self.total_hourly_rate) * max_elapsed_seconds / 3600.0
        if self.kind == "serverless_active_seconds":
            wall_ceiling = (
                float(self.active_second_rate)
                * int(self.max_active_workers)
                * max_elapsed_seconds
            )
            billed = (
                float(self.active_second_rate) * self.billed_active_seconds
                if self.billed_active_seconds is not None
                else 0.0
            )
            return max(wall_ceiling, billed)
        if self.kind == "user_supplied":
            return float(self.user_supplied_total)
        return None

    def elapsed_estimate(self, elapsed_seconds: float) -> float | None:
        """Conservative spend consumed so far for enforceable rate models."""

        if self.kind == "dedicated_hourly":
            return float(self.total_hourly_rate) * elapsed_seconds / 3600.0
        if self.kind == "serverless_active_seconds":
            return (
                float(self.active_second_rate)
                * int(self.max_active_workers)
                * elapsed_seconds
            )
        return None

    def final_cost(self, elapsed_seconds: float) -> tuple[float | None, str]:
        if self.kind == "dedicated_hourly":
            return self.elapsed_estimate(elapsed_seconds), "observed_client_wall_time"
        if self.kind == "serverless_active_seconds":
            if self.billed_active_seconds is None:
                return None, "billing_active_seconds_not_supplied"
            return (
                float(self.active_second_rate) * self.billed_active_seconds,
                "user_supplied_billed_active_seconds",
            )
        if self.kind == "user_supplied":
            return self.user_supplied_total, "user_supplied_run_total"
        return None, "billing_unknown"

    def public_dict(self, max_elapsed_seconds: float) -> dict[str, Any]:
        estimate = self.estimated_upper_bound(max_elapsed_seconds)
        data: dict[str, Any] = {
            "kind": self.kind,
            "currency": self.currency,
            "pre_run_upper_bound": estimate,
        }
        if self.kind == "dedicated_hourly":
            data.update(
                total_hourly_rate=self.total_hourly_rate,
                gpu_count=self.gpu_count,
                accounting_basis="client wall time",
            )
        elif self.kind == "serverless_active_seconds":
            data.update(
                active_second_rate=self.active_second_rate,
                max_active_workers=self.max_active_workers,
                billed_active_seconds=self.billed_active_seconds,
                accounting_basis="provider active seconds; client wall time is only an upper bound",
            )
        elif self.kind == "user_supplied":
            data.update(
                user_supplied_total=self.user_supplied_total,
                accounting_basis="user-attributed whole-run bill",
            )
        else:
            data["accounting_basis"] = "unknown; no cost metric calculated"
        return data


@dataclass(frozen=True)
class SafetyLimits:
    """Non-optional ceilings checked before and throughout traffic generation."""

    max_requests: int = 10_000
    max_tokens_per_request: int = 1_024
    max_total_requested_tokens: int = 2_000_000
    max_elapsed_seconds: float = 900.0
    # The count at which scheduling stops. A value of 1 gives zero tolerance:
    # the first error invalidates the condition and stops further launches.
    max_errors: int = 1
    max_concurrency: int = 64
    max_response_bytes: int = 2_000_000
    max_estimated_spend: float = 3.0

    def validate(self) -> None:
        integer_positive = {
            "max_requests": self.max_requests,
            "max_tokens_per_request": self.max_tokens_per_request,
            "max_total_requested_tokens": self.max_total_requested_tokens,
            "max_concurrency": self.max_concurrency,
            "max_response_bytes": self.max_response_bytes,
        }
        for name, value in integer_positive.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not isinstance(self.max_errors, int)
            or isinstance(self.max_errors, bool)
            or self.max_errors <= 0
        ):
            raise ValueError("max_errors must be a positive integer")
        if not _positive_finite(self.max_elapsed_seconds):
            raise ValueError("max_elapsed_seconds must be positive and finite")
        if not _positive_finite(self.max_estimated_spend):
            raise ValueError("max_estimated_spend must be positive and finite")

    def public_dict(self) -> dict[str, Any]:
        return {
            "max_requests": self.max_requests,
            "max_tokens_per_request": self.max_tokens_per_request,
            "max_total_requested_tokens": self.max_total_requested_tokens,
            "max_elapsed_seconds": self.max_elapsed_seconds,
            "max_errors": self.max_errors,
            "max_concurrency": self.max_concurrency,
            "max_response_bytes": self.max_response_bytes,
            "max_estimated_spend": self.max_estimated_spend,
        }


@dataclass(frozen=True)
class EndpointConfig:
    """Private destination and credential; neither field enters a report."""

    url: str
    api_key: str


@dataclass(frozen=True)
class LoadCondition:
    kind: TrafficKind
    value: float
    max_in_flight: int

    @property
    def condition_id(self) -> str:
        value = int(self.value) if float(self.value).is_integer() else self.value
        return f"{self.kind}:{value}"

    def public_dict(self) -> dict[str, Any]:
        value: int | float = (
            int(self.value) if float(self.value).is_integer() else self.value
        )
        return {
            "id": self.condition_id,
            "kind": self.kind,
            "value": value,
            "max_in_flight": self.max_in_flight,
        }


@dataclass(frozen=True)
class RunConfig:
    mode: RunMode
    backend: BackendKind
    model: str
    endpoint: EndpointConfig
    cost: CostModel = field(default_factory=CostModel)
    limits: SafetyLimits = field(default_factory=SafetyLimits)
    max_tokens: int = 128
    conditions: tuple[LoadCondition, ...] = (
        LoadCondition("closed_loop", 1.0, 1),
        LoadCondition("closed_loop", 4.0, 4),
        LoadCondition("closed_loop", 8.0, 8),
    )
    blocks: int = 1
    requests_per_block: int | None = 8
    block_duration_seconds: float | None = None
    warmup_requests_per_condition: int = 1
    request_timeout_seconds: float = 120.0
    p95_slo_ms: float | None = None
    ttft_slo_ms: float | None = None
    seed: int = 42
    stream: bool = True
    cache_policy: str = "unknown"
    model_revision: str = "unknown"
    image_digest: str = "unknown"
    gpu: str = "unknown"
    gpu_fingerprint: str = "unknown"
    cuda_version: str = "unknown"
    driver_version: str = "unknown"
    server_version: str = "unknown"
    engine_flags: tuple[tuple[str, str], ...] = ()
    engine_flags_provenance: str = "operator_attested"
    variant: str = "unspecified"
    sequence_position: str = "unspecified"
    allow_unknown_cost: bool = False
    allow_insecure_http: bool = False
    evidence_source: str = "unverified_endpoint"
    guidellm_gaps_acknowledged: bool = False

    def planned_request_count(self) -> int | None:
        if self.requests_per_block is None:
            return None
        return len(self.conditions) * (
            self.warmup_requests_per_condition + self.blocks * self.requests_per_block
        )

    def planned_requested_output_tokens(self) -> int | None:
        count = self.planned_request_count()
        return None if count is None else count * self.max_tokens
