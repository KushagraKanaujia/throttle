"""
throttle/advisor.py
===================
vLLM metric-to-dollar translation layer. Tier 1 only.

Reads vLLM /metrics (Prometheus text format), computes cost figures,
and streams one self-contained JSON snapshot per scrape interval.

Design constraints (from spec):
- Never estimates cost for unobserved configs
- Refuses to print $/Mtok when gen_throughput is unavailable
- Stops at Tier 1 — no recommendations, no window logic yet
- Small and readable: metrics arithmetic only

Consumed by a live view (not built here). Caller iterates stream_metrics()
and renders each snapshot however it wants.
"""

from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Iterator, Optional


# ---------------------------------------------------------------------------
# Prometheus scraper — no dependencies
# ---------------------------------------------------------------------------

class _Window:
    """Accumulates CostSnapshots for Tier 2 trend computation.

    Minimum 5 minutes (20 snapshots at 15s) before reporting idle
    fraction or batch inefficiency. Minimum 2 minutes for warnings.
    """

    MIN_SNAPSHOTS_WARN = 8    # ~2 minutes at 15s interval
    MIN_SNAPSHOTS_FULL = 20   # ~5 minutes at 15s interval

    def __init__(self, maxlen: int = 240):  # 1 hour at 15s
        self._snaps: deque = deque(maxlen=maxlen)

    def push(self, snap: "CostSnapshot") -> None:
        self._snaps.append(snap)

    @property
    def n(self) -> int:
        return len(self._snaps)

    @property
    def ready_full(self) -> bool:
        return self.n >= self.MIN_SNAPSHOTS_FULL

    @property
    def elapsed_minutes(self) -> float:
        return self.n * 15 / 60

    def idle_fraction(self) -> Optional[float]:
        if not self.ready_full:
            return None
        snaps = [s for s in self._snaps if s.num_requests_running is not None]
        if not snaps:
            return None
        idle = sum(1 for s in snaps if s.num_requests_running == 0)
        return round(idle / len(snaps), 3)

    def avg_batch_fill(self) -> Optional[float]:
        if not self.ready_full:
            return None
        fills = [s.batch_fill for s in self._snaps if s.batch_fill is not None]
        if not fills:
            return None
        return round(sum(fills) / len(fills), 3)

    def idle_cost_per_hour(self, gpu_hourly_rate: float) -> Optional[float]:
        frac = self.idle_fraction()
        if frac is None:
            return None
        return round(gpu_hourly_rate * frac, 4)

    def suggest(self, gpu_hourly_rate: float, max_num_seqs: Optional[int]) -> Optional[dict]:
        """Return the single highest-leverage suggestion, or None.

        Only fires after MIN_SNAPSHOTS_FULL. Never guesses.
        """
        if not self.ready_full:
            return None

        idle = self.idle_fraction()
        fill = self.avg_batch_fill()

        # Priority 1: sustained idle
        if idle is not None and idle > 0.20:
            idle_cost = self.idle_cost_per_hour(gpu_hourly_rate)
            return {
                "action": "reduce max_num_seqs or scale down",
                "basis": "observed",
                "observation_minutes": round(self.elapsed_minutes, 1),
                "reason": (
                    f"GPU idle {idle:.0%} of the last "
                    f"{self.elapsed_minutes:.0f} minutes. "
                    f"Estimated idle cost: ${idle_cost:.2f}/hr."
                ),
                "latency_impact": "none — idle means no active requests",
                "estimated_saving_per_hour": idle_cost,
                "confidence": "high",
            }

        # Priority 2: sustained underfilled batch
        if fill is not None and fill < 0.50 and max_num_seqs is not None:
            return {
                "action": f"reduce max_num_seqs from {max_num_seqs}",
                "basis": "observed",
                "observation_minutes": round(self.elapsed_minutes, 1),
                "reason": (
                    f"Batch fill averaged {fill:.0%} over "
                    f"{self.elapsed_minutes:.0f} minutes. "
                    f"Config reserves capacity current traffic does not use."
                ),
                "latency_impact": "unknown — may increase TTFT under burst. Monitor after applying.",
                "estimated_saving_per_hour": None,
                "confidence": "medium — assumes linear batch-throughput scaling",
                "suggested_next_step": (
                    f"throttle watch --try max_num_seqs=<lower_value> "
                    f"--try-duration-minutes 10 "
                    f"--gpu-rate-per-hour {gpu_hourly_rate}"
                ),
            }

        return None


def _scrape(url: str, timeout: float = 5.0) -> dict[str, float]:
    """
    Scrape Prometheus text format from url.
    Returns {metric_name: value} for gauge and counter lines.
    Ignores histograms and summaries (not needed for Tier 1).
    Raises on connection failure — caller decides how to handle.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.URLError as e:
        raise ConnectionError(f"Cannot reach {url}: {e}") from e

    metrics: dict[str, float] = {}
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            # metric_name{labels} value [timestamp]
            # or metric_name value [timestamp]
            parts = line.rsplit(" ", 2)
            value = float(parts[-2] if len(parts) == 3 else parts[-1])
            name_part = parts[0]
            # strip labels
            name = name_part.split("{")[0]
            # keep last value if duplicated (e.g. per-model labels)
            metrics[name] = value
        except (ValueError, IndexError):
            continue
    return metrics


# ---------------------------------------------------------------------------
# Tier 1 cost computation
# ---------------------------------------------------------------------------

@dataclass
class CostSnapshot:
    # Metadata
    timestamp: float
    scrape_url: str
    gpu_hourly_rate: float
    observation_seconds: float  # how long this tool has been running

    # Raw metrics (None = not exposed by endpoint)
    generation_throughput_toks_per_sec: Optional[float]
    prompt_throughput_toks_per_sec: Optional[float]
    num_requests_running: Optional[float]
    num_requests_waiting: Optional[float]
    gpu_cache_usage_perc: Optional[float]
    num_preemptions_total: Optional[float]

    # Derived cost (None = refused — see basis field)
    cost_per_hour: Optional[float]
    cost_per_million_tokens: Optional[float]

    # Batch fill (None if max_num_seqs unknown)
    batch_fill: Optional[float]
    max_num_seqs: Optional[int]

    # Tier 2 — window-derived (None until 5 minutes of observation)
    idle_fraction: Optional[float] = None
    suggestion: Optional[dict] = None
    window_ready: bool = False
    window_elapsed_minutes: float = 0.0

    # Data quality
    metrics_present: list[str] = field(default_factory=list)
    metrics_unavailable: list[str] = field(default_factory=list)
    refusals: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


_TIER1_METRICS = [
    "vllm:avg_generation_throughput_toks_per_s",
    "vllm:avg_prompt_throughput_toks_per_s",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:gpu_cache_usage_perc",
    "vllm:num_preemptions_total",
]


def _build_snapshot(
    raw: dict[str, float],
    gpu_hourly_rate: float,
    scrape_url: str,
    observation_seconds: float,
    max_num_seqs: Optional[int],
) -> CostSnapshot:
    """
    Translate raw Prometheus metrics into a CostSnapshot.
    Refuses cost computation when gen_throughput is unavailable.
    """
    present = []
    unavailable = []
    refusals = []

    def get(name: str) -> Optional[float]:
        if name in raw:
            present.append(name)
            return raw[name]
        else:
            unavailable.append(name)
            return None

    gen_tput = get("vllm:avg_generation_throughput_toks_per_s")
    prompt_tput = get("vllm:avg_prompt_throughput_toks_per_s")
    running = get("vllm:num_requests_running")
    waiting = get("vllm:num_requests_waiting")
    kv_usage = get("vllm:gpu_cache_usage_perc")
    preemptions = get("vllm:num_preemptions_total")

    # Cost computation
    gpu_rate_per_sec = gpu_hourly_rate / 3600.0

    if gen_tput is None or gen_tput <= 0:
        cost_per_hour = None
        cost_per_mtok = None
        if gen_tput is None:
            refusals.append({
                "figure": "cost_per_million_tokens",
                "reason": (
                    "vllm:avg_generation_throughput_toks_per_s not exposed. "
                    "Cannot compute cost without throughput denominator."
                ),
            })
        else:
            refusals.append({
                "figure": "cost_per_million_tokens",
                "reason": (
                    "Generation throughput is zero. "
                    "Is the backend serving requests?"
                ),
            })
    else:
        cost_per_hour = gpu_hourly_rate
        cost_per_mtok = round((gpu_rate_per_sec / gen_tput) * 1_000_000, 4)

    # Batch fill
    batch_fill = None
    if running is not None and max_num_seqs is not None and max_num_seqs > 0:
        batch_fill = round(running / max_num_seqs, 3)

    return CostSnapshot(
        timestamp=time.time(),
        scrape_url=scrape_url,
        gpu_hourly_rate=gpu_hourly_rate,
        observation_seconds=round(observation_seconds, 1),
        generation_throughput_toks_per_sec=gen_tput,
        prompt_throughput_toks_per_sec=prompt_tput,
        num_requests_running=running,
        num_requests_waiting=waiting,
        gpu_cache_usage_perc=kv_usage,
        num_preemptions_total=preemptions,
        cost_per_hour=cost_per_hour,
        cost_per_million_tokens=cost_per_mtok,
        idle_fraction=None,     # Tier 2 — window logic not yet built
        batch_fill=batch_fill,
        max_num_seqs=max_num_seqs,
        metrics_present=present,
        metrics_unavailable=unavailable,
        refusals=refusals,
    )


# ---------------------------------------------------------------------------
# Public streaming interface
# ---------------------------------------------------------------------------

def stream_metrics(
    metrics_url: str,
    gpu_rate_per_hour: float,
    interval_seconds: float = 15.0,
    max_num_seqs: Optional[int] = None,
) -> Iterator[CostSnapshot]:
    """
    Yields one CostSnapshot per scrape interval. Never returns.
    Each snapshot is self-contained — caller renders or stores it.

    Args:
        metrics_url:      vLLM /metrics endpoint, e.g. http://localhost:8000/metrics
        gpu_rate_per_hour: GPU cost in $/hr. Required — no default.
        interval_seconds: scrape interval (default 15s)
        max_num_seqs:     vLLM max_num_seqs for batch fill computation.
                          If None, batch_fill is omitted from snapshots.
    """
    start = time.time()
    window = _Window()

    while True:
        t0 = time.perf_counter()
        observation_seconds = time.time() - start

        try:
            raw = _scrape(metrics_url)
        except ConnectionError as e:
            # Yield an error snapshot rather than crashing the stream
            yield CostSnapshot(
                timestamp=time.time(),
                scrape_url=metrics_url,
                gpu_hourly_rate=gpu_rate_per_hour,
                observation_seconds=round(observation_seconds, 1),
                generation_throughput_toks_per_sec=None,
                prompt_throughput_toks_per_sec=None,
                num_requests_running=None,
                num_requests_waiting=None,
                gpu_cache_usage_perc=None,
                num_preemptions_total=None,
                cost_per_hour=None,
                cost_per_million_tokens=None,
                idle_fraction=None,
                batch_fill=None,
                max_num_seqs=max_num_seqs,
                metrics_present=[],
                metrics_unavailable=_TIER1_METRICS,
                refusals=[{"figure": "all", "reason": str(e)}],
            )
        else:
            snap = _build_snapshot(
                raw=raw,
                gpu_hourly_rate=gpu_rate_per_hour,
                scrape_url=metrics_url,
                observation_seconds=observation_seconds,
                max_num_seqs=max_num_seqs,
            )
            window.push(snap)
            snap.idle_fraction = window.idle_fraction()
            snap.suggestion = window.suggest(gpu_rate_per_hour, max_num_seqs)
            snap.window_ready = window.ready_full
            snap.window_elapsed_minutes = round(window.elapsed_minutes, 1)
            yield snap

        # Sleep for remainder of interval
        elapsed = time.perf_counter() - t0
        sleep = max(0.0, interval_seconds - elapsed)
        time.sleep(sleep)


# ---------------------------------------------------------------------------
# One-shot snapshot (for --try and testing)
# ---------------------------------------------------------------------------

def snapshot_once(
    metrics_url: str,
    gpu_rate_per_hour: float,
    max_num_seqs: Optional[int] = None,
) -> CostSnapshot:
    """Single scrape — used by --try and tests. Does not loop."""
    raw = _scrape(metrics_url)
    return _build_snapshot(
        raw=raw,
        gpu_hourly_rate=gpu_rate_per_hour,
        scrape_url=metrics_url,
        observation_seconds=0.0,
        max_num_seqs=max_num_seqs,
    )
