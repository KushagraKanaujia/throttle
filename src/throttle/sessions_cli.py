"""CLI handlers for `throttle sessions` command.

Displays agent session profiling data collected by the proxy.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, List

from .sessions import SessionTracker, SessionMetrics
from .session_findings import analyze_session, format_findings


def format_duration(seconds: float) -> str:
    """Format duration as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def format_table_row(columns: List[str], widths: List[int]) -> str:
    """Format a table row with proper column widths."""
    parts = []
    for col, width in zip(columns, widths):
        parts.append(col.ljust(width))
    return "│ " + " │ ".join(parts) + " │"


def print_session_table(sessions_data: List[dict], tracker: SessionTracker) -> None:
    """Print table of recent sessions."""
    if not sessions_data:
        print("No sessions found.")
        return

    # Compute metrics for each session
    rows = []
    for session in sessions_data:
        session_id = session["session_id"]
        metrics = tracker.compute_session_metrics(session_id)

        if metrics:
            rows.append({
                "session_id": session_id[:16] + "...",  # Truncate for display
                "turns": str(metrics.turn_count),
                "duration": format_duration(metrics.wall_clock_seconds),
                "idle_pct": f"{metrics.idle_percent:.1f}%",
                "redundant": f"{metrics.redundant_prefill_tokens:,} ({metrics.redundant_percent:.1f}%)",
            })

    if not rows:
        print("No complete sessions with metrics.")
        return

    # Column widths
    widths = [20, 7, 12, 10, 20]
    headers = ["Session ID", "Turns", "Duration", "Idle %", "Redundant Pfx"]

    # Print table
    print("\nRecent Sessions:")
    print("┌─" + "─┬─".join("─" * w for w in widths) + "─┐")
    print(format_table_row(headers, widths))
    print("├─" + "─┼─".join("─" * w for w in widths) + "─┤")

    for row in rows:
        cols = [
            row["session_id"],
            row["turns"],
            row["duration"],
            row["idle_pct"],
            row["redundant"],
        ]
        print(format_table_row(cols, widths))

    print("└─" + "─┴─".join("─" * w for w in widths) + "─┘")
    print()


def print_session_detail(session_id: str, tracker: SessionTracker) -> None:
    """Print detailed view of a single session."""
    metrics = tracker.compute_session_metrics(session_id)

    if not metrics:
        print(f"Session {session_id} not found or has no data.")
        return

    turns = tracker.get_session_turns(session_id)

    print(f"\nSession: {session_id}")
    print(f"Duration: {format_duration(metrics.wall_clock_seconds)}")
    print(f"Turns: {metrics.turn_count}")
    print()

    print("Wall Clock Breakdown:")
    gen_bar = "█" * int(metrics.generation_percent / 5)
    idle_bar = "█" * int(metrics.idle_percent / 5)
    print(f"  Generation:  {metrics.generation_seconds:.1f}s ({metrics.generation_percent:.1f}%)  {gen_bar}")
    print(f"  Client Wait: {metrics.idle_seconds:.1f}s ({metrics.idle_percent:.1f}%)  {idle_bar}")
    print()

    # Turn-by-turn timeline (first 20 turns)
    print("Turn-by-Turn Timeline (first 20 turns):")
    widths = [7, 12, 12, 12, 15, 11]
    headers = ["Turn", "Gap (s)", "TTFT (ms)", "Total (ms)", "Tokens", "Tool Call"]

    print("┌─" + "─┬─".join("─" * w for w in widths) + "─┐")
    print(format_table_row(headers, widths))
    print("├─" + "─┼─".join("─" * w for w in widths) + "─┤")

    for turn in turns[:20]:
        gap_str = "-" if turn.gap_since_previous_turn_seconds is None else f"{turn.gap_since_previous_turn_seconds:.1f}"
        ttft_str = "-" if turn.ttft_ms is None else f"{turn.ttft_ms:.0f}"
        tokens_str = f"{turn.prompt_tokens} → {turn.completion_tokens}"
        tool_str = "✓" if turn.contains_tool_calls else ""

        cols = [
            str(turn.turn_index),
            gap_str,
            ttft_str,
            f"{turn.total_latency_ms:.0f}",
            tokens_str,
            tool_str,
        ]
        print(format_table_row(cols, widths))

    if len(turns) > 20:
        print(f"  ... ({len(turns) - 20} more turns)")

    print("└─" + "─┴─".join("─" * w for w in widths) + "─┘")
    print()

    # Prefix overlap
    if metrics.redundant_prefill_tokens > 0:
        avg_overlap_pct = (
            sum(
                t.prefix_overlap_percent
                for t in turns
                if t.prefix_overlap_percent is not None
            )
            / len([t for t in turns if t.prefix_overlap_percent is not None])
            * 100
        ) if any(t.prefix_overlap_percent for t in turns) else 0

        print("Prefix Overlap:")
        print(f"  Average: {avg_overlap_pct:.1f}%")
        print(f"  Total redundant: {metrics.redundant_prefill_tokens:,} tokens ({metrics.redundant_percent:.1f}% of all prompt tokens)")
        print()

    # Findings
    findings = analyze_session(metrics, turns)
    if findings:
        print(format_findings(findings))
    else:
        print("No issues detected.")
    print()


def handle_sessions_command(args: Any) -> int:
    """Handle `throttle sessions` CLI command."""
    tracker = SessionTracker()

    # Determine time window
    if args.since:
        # Parse time window (e.g., "7d", "24h", "2w")
        since_str = args.since.lower()
        if since_str.endswith("d"):
            days = int(since_str[:-1])
            since_seconds = days * 86400
        elif since_str.endswith("h"):
            hours = int(since_str[:-1])
            since_seconds = hours * 3600
        elif since_str.endswith("w"):
            weeks = int(since_str[:-1])
            since_seconds = weeks * 7 * 86400
        else:
            print(f"Invalid time window: {args.since}. Use format like '7d', '24h', '2w'", file=sys.stderr)
            return 2
    else:
        # Default: last 24 hours
        since_seconds = 86400

    # Session detail view
    if args.session_id:
        print_session_detail(args.session_id, tracker)
        return 0

    # List view
    sessions = tracker.get_recent_sessions(since_seconds=since_seconds, limit=args.limit)

    if not sessions:
        print(f"No sessions found in the last {format_duration(since_seconds)}.")
        print("\nTo collect session data:")
        print("  1. Start proxy with session tracking:")
        print("     throttle proxy --backend-url <url> --enable-session-tracking")
        print("  2. Send agent traffic through the proxy")
        print("  3. Run `throttle sessions` again to see results")
        return 0

    print(f"\nShowing sessions from last {format_duration(since_seconds)}:")
    print_session_table(sessions, tracker)

    # Aggregate stats if enough data
    if len(sessions) >= 5:
        print("Aggregate Statistics:")
        all_metrics = [
            tracker.compute_session_metrics(s["session_id"])
            for s in sessions
        ]
        all_metrics = [m for m in all_metrics if m is not None]

        if all_metrics:
            import statistics

            median_turns = statistics.median([m.turn_count for m in all_metrics])
            median_idle = statistics.median([m.idle_percent for m in all_metrics])
            median_redundant = statistics.median([m.redundant_percent for m in all_metrics])

            print(f"  Total sessions: {len(all_metrics)}")
            print(f"  Median turns per session: {median_turns:.0f}")
            print(f"  Median idle time: {median_idle:.1f}%")
            print(f"  Median redundant prefill: {median_redundant:.1f}%")
            print()

            # Common findings
            print("Common Issues:")
            finding_counts = {}
            for m in all_metrics:
                turns = tracker.get_session_turns(m.session_id)
                findings = analyze_session(m, turns)
                for f in findings:
                    key = (f.severity, f.category)
                    finding_counts[key] = finding_counts.get(key, 0) + 1

            if finding_counts:
                for (severity, category), count in sorted(
                    finding_counts.items(),
                    key=lambda x: (-x[1], {"high": 0, "medium": 1, "low": 2}[x[0][0]]),
                )[:5]:
                    pct = count / len(all_metrics) * 100
                    severity_label = {"high": "[HIGH]", "medium": "[MED]", "low": "[LOW]"}[severity]
                    print(f"  {severity_label} {category.replace('_', ' ').title()}: {count}/{len(all_metrics)} sessions ({pct:.0f}%)")
            else:
                print("  No common issues detected.")

            print()

    return 0


def add_sessions_subcommand(subparsers: Any) -> None:
    """Add `sessions` subcommand to argparse."""
    sessions = subparsers.add_parser(
        "sessions",
        help="View agent session profiling data",
        description=(
            "Analyze multi-turn agent sessions captured by the proxy. "
            "Shows where wall clock time is spent and provides actionable "
            "findings to improve performance."
        ),
    )

    sessions.add_argument(
        "session_id",
        nargs="?",
        help="Show detailed view of a specific session",
    )

    sessions.add_argument(
        "--since",
        default="24h",
        help="Time window to query (e.g., '7d', '24h', '2w'). Default: 24h",
    )

    sessions.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of sessions to show (default: 50)",
    )
