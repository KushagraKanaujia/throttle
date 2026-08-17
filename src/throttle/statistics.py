"""Small, dependency-free statistical helpers used by Throttle reports."""

from __future__ import annotations

import math
import random
import statistics
from typing import Iterable, Sequence

# Two-sided 95% Student-t critical values for the small block counts Throttle
# accepts. Larger samples use the asymptotic normal value.
_T_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}

# Request-level percentile intervals are diagnostics, not the independent-unit
# confidence intervals used for decisions. Bound their work so a large run
# cannot spend unbounded time or memory in post-processing.
MAX_BOOTSTRAP_SAMPLE_SIZE = 1_024
DEFAULT_BOOTSTRAP_RESAMPLES = 300


def _t_critical_975(degrees_of_freedom: int) -> float:
    exact = _T_975.get(degrees_of_freedom)
    if exact is not None:
        return exact
    # Cornish-Fisher expansion around the normal 97.5th percentile. This stays
    # conservative relative to a hard 1.96 switch for configurable block counts
    # just above the exact table while converging smoothly for large df.
    z = 1.959963984540054
    df = float(degrees_of_freedom)
    return (
        z
        + (z**3 + z) / (4.0 * df)
        + (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * df**2)
        + (3.0 * z**7 + 19.0 * z**5 + 17.0 * z**3 - 15.0 * z) / (384.0 * df**3)
    )


def finite(values: Iterable[float | int | None]) -> list[float]:
    output: list[float] = []
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        try:
            converted = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(converted):
            output.append(converted)
    return output


def percentile(values: Sequence[float], fraction: float) -> float | None:
    clean = sorted(finite(values))
    if not clean:
        return None
    if not 0 <= fraction <= 1:
        raise ValueError("percentile fraction must be between zero and one")
    position = (len(clean) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return clean[lower]
    weight = position - lower
    return clean[lower] * (1.0 - weight) + clean[upper] * weight


def t_interval_95(values: Sequence[float]) -> dict[str, float | int | str | None]:
    """95% CI for a block mean, treating repeated blocks as independent units."""

    clean = finite(values)
    if not clean:
        return {
            "low": None,
            "high": None,
            "confidence": 0.95,
            "method": "student_t_blocks",
            "n": 0,
        }
    mean = statistics.fmean(clean)
    if len(clean) < 2:
        return {
            "low": None,
            "high": None,
            "confidence": 0.95,
            "method": "student_t_blocks",
            "n": len(clean),
        }
    standard_error = statistics.stdev(clean) / math.sqrt(len(clean))
    critical = _t_critical_975(len(clean) - 1)
    return {
        "low": mean - critical * standard_error,
        "high": mean + critical * standard_error,
        "confidence": 0.95,
        "method": "student_t_blocks",
        "n": len(clean),
    }


def bootstrap_percentile_interval_95(
    values: Sequence[float],
    fraction: float,
    *,
    seed: int,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict[str, float | int | str | None]:
    """Bounded deterministic bootstrap for request-level diagnostic CIs."""

    clean = finite(values)
    original_count = len(clean)
    if len(clean) > MAX_BOOTSTRAP_SAMPLE_SIZE:
        sampler = random.Random(seed ^ 0x5A17)
        clean = [
            clean[index]
            for index in sorted(
                sampler.sample(range(len(clean)), MAX_BOOTSTRAP_SAMPLE_SIZE)
            )
        ]
    if len(clean) < 2:
        return {
            "low": None,
            "high": None,
            "confidence": 0.95,
            "method": "bounded_percentile_bootstrap_requests",
            "n": original_count,
            "analysis_sample_n": len(clean),
            "resamples": resamples,
        }
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(resamples):
        sample = [clean[rng.randrange(len(clean))] for _ in clean]
        estimate = percentile(sample, fraction)
        if estimate is not None:
            estimates.append(estimate)
    return {
        "low": percentile(estimates, 0.025),
        "high": percentile(estimates, 0.975),
        "confidence": 0.95,
        "method": "bounded_percentile_bootstrap_requests",
        "n": original_count,
        "analysis_sample_n": len(clean),
        "resamples": resamples,
    }


def summarize_distribution_ms(
    seconds: Sequence[float], *, seed: int
) -> dict[str, object]:
    values_ms = [value * 1000.0 for value in finite(seconds)]
    return {
        "count": len(values_ms),
        "mean": statistics.fmean(values_ms) if values_ms else None,
        "p50": percentile(values_ms, 0.50),
        "p90": percentile(values_ms, 0.90),
        "p95": percentile(values_ms, 0.95),
        "p99": percentile(values_ms, 0.99),
        "p95_ci": bootstrap_percentile_interval_95(values_ms, 0.95, seed=seed),
    }


def relative_delta_percent(candidate: float, baseline: float) -> float | None:
    try:
        candidate_value = float(candidate)
        baseline_value = float(baseline)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        not all(math.isfinite(value) for value in (candidate_value, baseline_value))
        or baseline_value == 0
    ):
        return None
    result = (candidate_value / baseline_value - 1.0) * 100.0
    return result if math.isfinite(result) else None


def paired_relative_delta_interval_95(
    baseline_blocks: Sequence[float], candidate_blocks: Sequence[float]
) -> dict[str, object]:
    if len(baseline_blocks) != len(candidate_blocks):
        return {
            "estimate": None,
            "low": None,
            "high": None,
            "confidence": 0.95,
            "method": "paired_student_t_blocks",
            "n": 0,
        }
    deltas = [
        delta
        for base, candidate in zip(baseline_blocks, candidate_blocks)
        if (delta := relative_delta_percent(candidate, base)) is not None
    ]
    interval = t_interval_95(deltas)
    interval["estimate"] = statistics.fmean(deltas) if deltas else None
    interval["method"] = "paired_student_t_blocks"
    return interval


def intervals_overlap(left: dict[str, object], right: dict[str, object]) -> bool:
    values = (left.get("low"), left.get("high"), right.get("low"), right.get("high"))
    if not all(
        isinstance(value, (int, float)) and math.isfinite(value) for value in values
    ):
        return True
    return float(left["low"]) <= float(right["high"]) and float(right["low"]) <= float(
        left["high"]
    )
