"""Findings analyzer for agent session profiling.

Generates actionable recommendations from session metrics.
Each finding maps observed evidence to specific config changes.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import List, Literal

from .sessions import SessionMetrics, Turn


@dataclass
class Finding:
    """A single actionable finding with severity and recommendation."""

    severity: Literal["high", "medium", "low"]
    category: str
    observation: str
    recommendation: str
    config_change: str | None


def analyze_session(metrics: SessionMetrics, turns: List[Turn]) -> List[Finding]:
    """Generate findings for a single session.

    Args:
        metrics: Computed session-level metrics
        turns: Individual turn data for detailed analysis

    Returns:
        List of findings ordered by severity
    """
    findings: List[Finding] = []

    # Rule 1: High redundant prefill → enable prefix caching
    if metrics.redundant_percent > 30:
        findings.append(
            Finding(
                severity="high",
                category="prefix_caching",
                observation=f"{metrics.redundant_percent:.1f}% of prompt tokens are redundant prefill ({metrics.redundant_prefill_tokens:,} tokens across {metrics.turn_count} turns)",
                recommendation="Make sure prefix caching is enabled on the backend so the KV cache for shared conversation history is reused across turns instead of being prefilled again. (Redundant token counts are an estimate from message-level prefix overlap.)",
                config_change="vLLM: --enable-prefix-caching (on by default in recent V1 releases) | SGLang: RadixAttention prefix cache is on unless --disable-radix-cache is set",
            )
        )

    # Rule 2: High idle time + low concurrency → GPU underutilized
    if metrics.idle_percent > 70:
        findings.append(
            Finding(
                severity="high",
                category="utilization",
                observation=f"{metrics.idle_percent:.1f}% of wall clock spent idle ({metrics.idle_seconds:.1f}s of {metrics.wall_clock_seconds:.1f}s total)",
                recommendation="GPU is idle while waiting for client tool execution or user interaction. Consider: (1) batching with other traffic, (2) scaling down to save costs, or (3) serving multiple users.",
                config_change="Enable request batching or reduce instance size if traffic is consistently low-concurrency",
            )
        )

    # Rule 3: TTFT grows with turn index → KV cache pressure
    ttft_by_turn = _analyze_ttft_growth(turns)
    if ttft_by_turn and ttft_by_turn["has_growth"]:
        findings.append(
            Finding(
                severity="high",
                category="kv_cache_pressure",
                observation=f"TTFT increases from {ttft_by_turn['early_median']:.0f}ms (early turns) to {ttft_by_turn['late_median']:.0f}ms (late turns) - {ttft_by_turn['growth_percent']:.0f}% growth",
                recommendation="Time to first token grows as the conversation grows. Likely causes: prefill of an ever-longer context without prefix-cache reuse, or KV cache pressure causing evictions/preemption. Check prefix caching first, then KV cache capacity.",
                config_change="vLLM: --enable-prefix-caching; if still growing, raise --gpu-memory-utilization (default 0.9) or lower --max-model-len",
            )
        )

    # Rule 4: Frequent tool calls with long gaps → slow tool execution
    tool_call_analysis = _analyze_tool_call_gaps(turns)
    if tool_call_analysis and tool_call_analysis["avg_gap"] > 10.0:
        findings.append(
            Finding(
                severity="medium",
                category="agent_orchestration",
                observation=f"Average {tool_call_analysis['avg_gap']:.1f}s gap after tool calls ({tool_call_analysis['tool_call_count']} occurrences)",
                recommendation="Tool execution is slow. Profile agent framework for bottlenecks: network latency, serial tool execution, inefficient tool implementations.",
                config_change="Client-side: parallelize independent tools, cache tool results, use async execution",
            )
        )

    # Rule 5: Very short turns with high frequency → inefficient agent loop
    if metrics.turn_count > 50 and metrics.wall_clock_seconds / metrics.turn_count < 5.0:
        avg_turn_seconds = metrics.wall_clock_seconds / metrics.turn_count
        findings.append(
            Finding(
                severity="low",
                category="agent_design",
                observation=f"Very high turn frequency: {metrics.turn_count} turns in {metrics.wall_clock_seconds:.1f}s (avg {avg_turn_seconds:.1f}s/turn)",
                recommendation="Agent may be making too many small requests. Consider: batching multiple reasoning steps, using longer context windows, or redesigning agent loop.",
                config_change="Client-side: reduce agent loop iterations, batch reasoning steps",
            )
        )

    # Rule 6: High generation time but low completion tokens → inefficient sampling
    if metrics.generation_percent > 50:
        avg_tokens_per_second = (
            metrics.total_completion_tokens / metrics.generation_seconds
            if metrics.generation_seconds > 0
            else 0
        )
        if avg_tokens_per_second < 50:  # Suspiciously low throughput
            findings.append(
                Finding(
                    severity="medium",
                    category="throughput",
                    observation=f"Low throughput: {avg_tokens_per_second:.1f} tokens/sec during generation ({metrics.total_completion_tokens} tokens in {metrics.generation_seconds:.1f}s)",
                    recommendation="Check backend configuration. Possible issues: no batching, suboptimal max_num_seqs, memory-bound operations.",
                    config_change="vLLM: check --max-num-seqs / --max-num-batched-tokens if requests are queuing",
                )
            )

    # Sort by severity
    severity_order = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda f: severity_order[f.severity])

    return findings


def _analyze_ttft_growth(turns: List[Turn]) -> dict | None:
    """Detect if TTFT grows significantly with turn index.

    Returns dict with growth analysis or None if insufficient data.
    """
    ttft_values = [(t.turn_index, t.ttft_ms) for t in turns if t.ttft_ms is not None]

    if len(ttft_values) < 10:
        return None

    # Split into early and late turns
    mid_point = len(ttft_values) // 2
    early = [ttft for _, ttft in ttft_values[:mid_point]]
    late = [ttft for _, ttft in ttft_values[mid_point:]]

    early_median = statistics.median(early)
    late_median = statistics.median(late)

    growth_percent = ((late_median - early_median) / early_median * 100) if early_median > 0 else 0

    # Consider it growth if >30% increase
    has_growth = growth_percent > 30

    return {
        "has_growth": has_growth,
        "early_median": early_median,
        "late_median": late_median,
        "growth_percent": growth_percent,
    }


def _analyze_tool_call_gaps(turns: List[Turn]) -> dict | None:
    """Analyze gaps after tool calls.

    Returns dict with analysis or None if no tool calls.
    """
    tool_call_turns = [i for i, t in enumerate(turns) if t.contains_tool_calls]

    if not tool_call_turns:
        return None

    # Get gaps after tool call turns
    gaps = []
    for tool_idx in tool_call_turns:
        next_idx = tool_idx + 1
        if next_idx < len(turns):
            gap = turns[next_idx].gap_since_previous_turn_seconds
            if gap is not None:
                gaps.append(gap)

    if not gaps:
        return None

    avg_gap = statistics.mean(gaps)

    return {
        "tool_call_count": len(tool_call_turns),
        "avg_gap": avg_gap,
        "median_gap": statistics.median(gaps),
    }


def format_findings(findings: List[Finding]) -> str:
    """Format findings for terminal output."""
    if not findings:
        return "No issues detected."

    lines = ["FINDINGS:\n"]

    for finding in findings:
        severity_label = {
            "high": "[HIGH]",
            "medium": "[MED]",
            "low": "[LOW]",
        }[finding.severity]

        lines.append(f"{severity_label} {finding.category.replace('_', ' ').title()}")
        lines.append(f"  Observation: {finding.observation}")
        lines.append(f"  Recommendation: {finding.recommendation}")

        if finding.config_change:
            lines.append(f"  Config: {finding.config_change}")

        lines.append("")  # Blank line between findings

    return "\n".join(lines)
