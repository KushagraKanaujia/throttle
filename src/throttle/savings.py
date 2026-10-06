"""``throttle savings``: a conservative, auditable verified-savings statement.

Takes two recorded checks (a baseline and a candidate), re-judges the pair
with the same rules ``throttle check`` uses, and refuses to produce a figure
unless the verdict is a calibrated CHEAPER on the same model and workload.

Rules:

* Savings use the conservative bound, never the point estimate: the smallest
  plausible $/M reduction (baseline CI low minus candidate CI high) times the
  production token count / 1e6. The point estimate is shown, labelled.
* Both sides are priced at the baseline's per-GPU rate, exactly as the
  verdict judges them, so a change in the ASSUMED price never counts as a
  saving. A change in GPU count does count (it is a real cost).
* Token counts are REPORTED BY OPERATOR (``--tokens``) or MEASURED as a delta
  of vLLM's ``vllm:generation_tokens_total`` counter since a snapshot taken
  with ``throttle savings snapshot``. A counter reset is refused.
* No fees, no pricing, no network calls except the user-given metrics URL.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .advisor import _scrape
from .check import (
    EXIT_FAILED,
    EXIT_OK,
    EXIT_USAGE,
    HISTORY_FILENAME,
    METRIC_LABELS,
    VERDICT_CHEAPER,
    WORKLOAD_IDENTITY_FIELDS,
    _metrics_url_problem,
    _money,
    _token_count,
    compare_checks,
    gpu_count,
    history_dir,
    load_history,
    per_gpu_rate,
    workload_value,
)

RECORD_TYPE = "savings_statement"
RECORD_VERSION = 1
SNAPSHOT_RECORD_TYPE = "savings_counter_snapshot"
SNAPSHOT_RECORD_VERSION = 1
GENERATION_COUNTER = "vllm:generation_tokens_total"
SOURCE_OPERATOR = "REPORTED BY OPERATOR"
SOURCE_MEASURED = "MEASURED"

SAVINGS_DESCRIPTION = (
    "Produce a conservative, auditable savings statement for a baseline -> candidate\n"
    "config change. Re-judges the two recorded checks and refuses (exit 1, one-line\n"
    "reason) unless the verdict is a calibrated CHEAPER on the same model and workload.\n\n"
    "Verified savings = smallest plausible $/M reduction (baseline CI low - candidate\n"
    "CI high) x production tokens / 1e6. The point estimate is shown too, labelled.\n"
    "GPU $/hr is ASSUMED (what you passed to 'throttle check'). Sends no traffic,\n"
    "except one read of --metrics-url when given."
)
SAVINGS_EPILOG = (
    "token source (pick one):\n"
    "  --tokens N                      REPORTED BY OPERATOR, e.g. 1.2B\n"
    "  --metrics-url URL --window-start FILE\n"
    "                                  MEASURED: vLLM generation-token counter now,\n"
    "                                  minus the value recorded at the period start by\n"
    "                                  'throttle savings snapshot'\n\n"
    "examples:\n"
    "  throttle savings snapshot --metrics-url http://gpu1:8000/metrics --out start.json\n"
    "  throttle savings --baseline ID1 --candidate ID2 \\\n"
    "      --metrics-url http://gpu1:8000/metrics --window-start start.json\n"
    "  throttle savings --baseline ID1 --candidate ID2 --tokens 1.2B --period-label 2026-10\n\n"
    "exit codes: 0 statement produced, 1 refused (reason printed), 2 usage error"
)


class SavingsRefused(Exception):
    """The pair or token source cannot back a savings figure; one-line reason."""


def add_savings_arguments(parser: argparse.ArgumentParser) -> None:
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.add_argument("--baseline", metavar="CHECK_ID",
                        help="recorded check id of the config you moved away from")
    parser.add_argument("--candidate", metavar="CHECK_ID",
                        help="recorded check id of the config you run now")
    parser.add_argument("--tokens", type=_token_count, metavar="N",
                        help="production tokens served in the period (REPORTED BY OPERATOR)")
    parser.add_argument("--metrics-url", metavar="URL",
                        help="vLLM /metrics URL to read the generation-token counter from (MEASURED)")
    parser.add_argument("--window-start", metavar="FILE",
                        help="counter snapshot written by 'throttle savings snapshot' at the period start")
    parser.add_argument("--period-label", metavar="TEXT",
                        help="name for the period, e.g. 2026-10 (default: the snapshot window, if any)")
    parser.add_argument("--history-dir", metavar="PATH",
                        help="check history directory (default: $THROTTLE_CHECK_HISTORY_DIR or ~/.throttle/checks)")
    parser.add_argument("--json", action="store_true",
                        help="print the statement as one JSON record (record_type savings_statement)")
    sub = parser.add_subparsers(dest="savings_command", metavar="{snapshot}")
    snapshot = sub.add_parser(
        "snapshot",
        help="record the vLLM generation-token counter at the start of a period",
        description=(
            f"Reads {GENERATION_COUNTER} from --metrics-url once and writes it with a "
            "UTC timestamp to --out. Pass that file to 'throttle savings --window-start'."
        ),
    )
    snapshot.add_argument("--metrics-url", required=True, metavar="URL",
                          help="vLLM /metrics URL, e.g. http://localhost:8000/metrics")
    snapshot.add_argument("--out", required=True, metavar="FILE",
                          help="where to write the snapshot (JSON)")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


def read_generation_counter(url: str) -> float:
    """Current value of vLLM's generation-token counter at ``url``."""

    problem = _metrics_url_problem(url)
    if problem:
        raise SavingsRefused(f"--metrics-url {problem}")
    try:
        metrics = _scrape(url)
    except (ConnectionError, OSError, ValueError) as exc:
        raise SavingsRefused(f"could not read {url}: {exc}") from None
    value = metrics.get(GENERATION_COUNTER)
    if value is None:
        raise SavingsRefused(f"{url} does not expose {GENERATION_COUNTER} (is it a vLLM /metrics endpoint?)")
    return value


def take_snapshot(url: str) -> dict[str, Any]:
    return {
        "record_type": SNAPSHOT_RECORD_TYPE,
        "record_version": SNAPSHOT_RECORD_VERSION,
        "metrics_url": url,
        "counter": GENERATION_COUNTER,
        "value": read_generation_counter(url),
        "taken_at": _iso(_now()),
    }


def measured_tokens(url: str, start_file: str) -> dict[str, Any]:
    """Counter delta since the snapshot in ``start_file``; refuses a reset."""

    try:
        start = json.loads(Path(start_file).expanduser().read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SavingsRefused(f"could not read --window-start {start_file}: {exc}") from None
    if not isinstance(start, dict) or start.get("record_type") != SNAPSHOT_RECORD_TYPE:
        raise SavingsRefused(f"{start_file} is not a 'throttle savings snapshot' file")
    start_value = start.get("value")
    if not isinstance(start_value, (int, float)) or isinstance(start_value, bool) or start_value < 0:
        raise SavingsRefused(f"{start_file} has no usable counter value")
    if start.get("metrics_url") != url:
        raise SavingsRefused(
            f"the snapshot was taken from {start.get('metrics_url')}, not {url}; "
            "a counter delta across two endpoints is meaningless"
        )
    current = read_generation_counter(url)
    if current < start_value:
        raise SavingsRefused(
            f"{GENERATION_COUNTER} went down ({start_value:,.0f} at the snapshot, "
            f"{current:,.0f} now): the server restarted and the counter reset, so the "
            "tokens served this period are unknown. Take a new snapshot, or use --tokens"
        )
    return {
        "count": int(round(current - start_value)),
        "source": SOURCE_MEASURED,
        "detail": f"{GENERATION_COUNTER} delta at {url}",
        "window_start": start.get("taken_at"),
        "window_end": _iso(_now()),
        "counter_start": start_value,
        "counter_end": current,
    }


def _find(records: list[dict[str, Any]], check_id: str, role: str, path: Path) -> int:
    for index, record in enumerate(records):
        if record.get("id") == check_id:
            return index
    raise SavingsRefused(f"no {role} check with id {check_id!r} in {path}")


def _side(record: Mapping[str, Any], metric: str, factor: float) -> dict[str, Any]:
    summary = record["result"][metric]
    fingerprint = record["fingerprint"]

    def scaled(key: str) -> float | None:
        value = summary.get(key)
        return None if value is None else float(value) * factor

    return {
        "check_id": record.get("id"),
        "created_at": record.get("created_at"),
        "endpoint": record.get("endpoint"),
        "model": fingerprint.get("model"),
        "label": fingerprint.get("label"),
        "config": dict(fingerprint.get("user_config") or {}),
        "gpu_count": gpu_count(record),
        "gpu_hourly_rate_usd": float(fingerprint["gpu_hourly_rate_usd"]),
        "gpu_hourly_rate_source": "ASSUMED (user-supplied --gpu-hourly-rate)",
        "dollars_per_million_recorded": {
            "mean": summary.get("mean"), "ci_low": summary.get("ci_low"),
            "ci_high": summary.get("ci_high"),
        },
        "dollars_per_million": {
            "mean": scaled("mean"), "ci_low": scaled("ci_low"), "ci_high": scaled("ci_high"),
        },
    }


def build_statement(
    records: list[dict[str, Any]],
    baseline_id: str,
    candidate_id: str,
    tokens: Mapping[str, Any],
    history_path: Path,
    period_label: str | None = None,
) -> dict[str, Any]:
    """The savings statement record, or SavingsRefused with a one-line reason."""

    if baseline_id == candidate_id:
        raise SavingsRefused("--baseline and --candidate are the same check")
    b_index = _find(records, baseline_id, "baseline", history_path)
    c_index = _find(records, candidate_id, "candidate", history_path)
    baseline, candidate = records[b_index], records[c_index]
    b_model, c_model = baseline["fingerprint"].get("model"), candidate["fingerprint"].get("model")
    if b_model != c_model:
        raise SavingsRefused(f"different models ({b_model} vs {c_model}): not the same workload")
    metric = candidate.get("metric", "output")
    if baseline.get("metric", "output") != metric:
        raise SavingsRefused(
            f"the checks use different $/M metrics ({baseline.get('metric', 'output')} vs {metric})"
        )
    workload_diffs = [
        f for f in WORKLOAD_IDENTITY_FIELDS
        if workload_value(baseline, f) != workload_value(candidate, f)
    ]
    if workload_diffs:
        raise SavingsRefused(
            "the workload differs (" + ", ".join(workload_diffs) + "): not comparable"
        )
    if tokens["source"] == SOURCE_MEASURED and metric != "output":
        raise SavingsRefused(
            f"the checks' primary metric is $/M {METRIC_LABELS.get(metric, metric)}, but the "
            "vLLM counter counts output tokens only; use --tokens with the matching count"
        )

    # Re-judge with the history that existed when the later of the two was recorded.
    history = records[: max(b_index, c_index) + 1]
    comparison = compare_checks(baseline, candidate, history)
    verdict = comparison["verdict"]
    if verdict != VERDICT_CHEAPER:
        raise SavingsRefused(f"re-judged verdict is {verdict}, not CHEAPER: {comparison['reason']}")

    factor = per_gpu_rate(baseline) / per_gpu_rate(candidate)
    base = _side(baseline, metric, 1.0)
    cand = _side(candidate, metric, factor)
    b_dpm, c_dpm = base["dollars_per_million"], cand["dollars_per_million"]
    if b_dpm["ci_low"] is None or c_dpm["ci_high"] is None:
        raise SavingsRefused("a check has no 95% CI, so there is no conservative bound")
    conservative_dpm = b_dpm["ci_low"] - c_dpm["ci_high"]
    point_dpm = b_dpm["mean"] - c_dpm["mean"]
    if conservative_dpm <= 0:
        raise SavingsRefused("the conservative $/M reduction is not positive")
    count = int(tokens["count"])

    if period_label:
        period = period_label
    elif tokens.get("window_start"):
        period = f"{tokens['window_start']} to {tokens['window_end']}"
    else:
        period = None

    noise = comparison.get("noise") or {}
    return {
        "record_type": RECORD_TYPE,
        "record_version": RECORD_VERSION,
        "created_at": _iso(_now()),
        "period": period,
        "history_file": str(history_path),
        "metric": metric,
        "metric_label": f"$/M {METRIC_LABELS.get(metric, metric)}",
        "verdict": verdict,
        "verdict_reason": comparison["reason"],
        "noise_floor_percent": noise.get("floor_percent"),
        "measured_delta_percent": comparison.get("measured_delta_percent"),
        "priced_at": "baseline per-GPU rate (candidate $/M scaled by "
                     f"{factor:.6g}); GPU count changes are kept as real cost",
        "rate_factor": factor,
        "baseline": base,
        "candidate": cand,
        "tokens": dict(tokens),
        "savings": {
            "conservative_usd": conservative_dpm * count / 1e6,
            "conservative_dollars_per_million": conservative_dpm,
            "point_estimate_usd": point_dpm * count / 1e6,
            "point_estimate_dollars_per_million": point_dpm,
            "rule": "conservative = (baseline CI low - candidate CI high) x tokens / 1e6; "
                    "point estimate = (baseline mean - candidate mean) x tokens / 1e6",
        },
        "assumptions": [
            f"GPU $/hr is ASSUMED, as the user supplied it: baseline "
            f"{_money(base['gpu_hourly_rate_usd'])}/hr for {base['gpu_count']} GPU(s), candidate "
            f"{_money(cand['gpu_hourly_rate_usd'])}/hr for {cand['gpu_count']} GPU(s)",
            "$/M is MEASURED by 'throttle check' on its fixed workload (95% CI across blocks); "
            "production traffic is assumed to cost per token what that workload did",
            f"token count is {tokens['source']}",
        ],
    }


def _config_text(side: Mapping[str, Any]) -> str:
    config = " ".join(f"{k}={v}" for k, v in sorted(side["config"].items())) or "-"
    label = side.get("label") or "-"
    return f"label {label}, config {config}"


def _dpm_text(dpm: Mapping[str, Any]) -> str:
    return (f"{_money(dpm['mean'])}/M (95% CI {_money(dpm['ci_low'])} to "
            f"{_money(dpm['ci_high'])})")


def format_statement(statement: Mapping[str, Any]) -> str:
    base, cand = statement["baseline"], statement["candidate"]
    tokens, savings = statement["tokens"], statement["savings"]
    lines = [
        "Throttle verified-savings statement",
        f"  period      {statement['period'] or '(not stated; pass --period-label)'}",
        f"  model       {base['model']}",
        f"  metric      {statement['metric_label']} (the primary metric of these checks)",
        f"  baseline    {base['check_id']}  {base['gpu_count']} GPU(s)  {_config_text(base)}",
        f"              {_dpm_text(base['dollars_per_million'])}  [MEASURED]",
        f"  candidate   {cand['check_id']}  {cand['gpu_count']} GPU(s)  {_config_text(cand)}",
        f"              {_dpm_text(cand['dollars_per_million'])}  [MEASURED]",
    ]
    if statement["rate_factor"] != 1.0:
        lines.append(
            f"              (recorded {_dpm_text(cand['dollars_per_million_recorded'])}; "
            "shown at the baseline's per-GPU rate)"
        )
    lines += [
        f"  verdict     {statement['verdict']} (re-judged): {statement['verdict_reason']}",
        f"  tokens      {tokens['count']:,}  [{tokens['source']}]"
        + (f"  {tokens['detail']}" if tokens.get("detail") else ""),
        "",
        f"  Verified savings (conservative): {_money(savings['conservative_usd'])}",
        f"    = ({_money(base['dollars_per_million']['ci_low'])} baseline CI low - "
        f"{_money(cand['dollars_per_million']['ci_high'])} candidate CI high) "
        f"x {tokens['count']:,} / 1e6",
        f"  Point estimate (not the verified figure): {_money(savings['point_estimate_usd'])}",
        "",
        "  Assumptions",
        *(f"    - {item}" for item in statement["assumptions"]),
    ]
    return "\n".join(lines)


def _refuse(message: str) -> int:
    print(f"Savings refused: {message}", file=sys.stderr)
    return EXIT_FAILED


def handle_savings(args: argparse.Namespace) -> int:
    if getattr(args, "savings_command", None) == "snapshot":
        try:
            snapshot = take_snapshot(args.metrics_url)
        except SavingsRefused as exc:
            return _refuse(str(exc))
        out = Path(args.out).expanduser()
        out.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Recorded {GENERATION_COUNTER} = {snapshot['value']:,.0f} at "
              f"{snapshot['taken_at']} in {out}")
        return EXIT_OK

    if not args.baseline or not args.candidate:
        print("Error: --baseline and --candidate are required", file=sys.stderr)
        return EXIT_USAGE
    measured = bool(args.metrics_url and args.window_start)
    if (args.tokens is None) == (not measured) or (
        args.tokens is not None and (args.metrics_url or args.window_start)
    ):
        print("Error: give either --tokens N, or both --metrics-url and --window-start",
              file=sys.stderr)
        return EXIT_USAGE

    directory = history_dir(args)
    records, _skipped = load_history(directory)
    path = directory / HISTORY_FILENAME
    try:
        if args.tokens is not None:
            tokens: dict[str, Any] = {"count": args.tokens, "source": SOURCE_OPERATOR}
        else:
            tokens = measured_tokens(args.metrics_url, args.window_start)
        statement = build_statement(
            records, args.baseline, args.candidate, tokens, path, args.period_label
        )
    except SavingsRefused as exc:
        return _refuse(str(exc))
    if args.json:
        print(json.dumps(statement, indent=2, sort_keys=True))
    else:
        print(format_statement(statement))
    return EXIT_OK
