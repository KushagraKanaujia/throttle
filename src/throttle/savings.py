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
  with ``throttle savings snapshot``, per labelled series (e.g. one per
  data-parallel engine), keeping only the checks' model_name, then summed.
  A series that went down or disappeared is a reset and is refused.
* No fees, no pricing, no network calls except the user-given metrics URL.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from . import style as style_module
from .check import (
    EXIT_FAILED,
    EXIT_OK,
    EXIT_USAGE,
    HISTORY_FILENAME,
    MAX_METRICS_BYTES,
    METRIC_LABELS,
    METRICS_TIMEOUT_SECONDS,
    VERDICT_CHEAPER,
    WORKLOAD_IDENTITY_FIELDS,
    _LABEL,
    _metrics_url_problem,
    _money,
    _shape_text,
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
            f"Reads every series of {GENERATION_COUNTER} from --metrics-url once and "
            "writes them with a UTC timestamp to --out. Pass that file to 'throttle savings --window-start'."
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


def fetch_metrics_text(url: str) -> str:
    """One bounded GET of the user-given /metrics URL."""

    problem = _metrics_url_problem(url)
    if problem:
        raise SavingsRefused(f"--metrics-url {problem}")
    try:
        with urllib.request.urlopen(url, timeout=METRICS_TIMEOUT_SECONDS) as response:
            body = response.read(MAX_METRICS_BYTES + 1)
    except (OSError, ValueError) as exc:
        raise SavingsRefused(f"could not read {url}: {exc}") from None
    if len(body) > MAX_METRICS_BYTES:
        raise SavingsRefused(f"{url} returned more than {MAX_METRICS_BYTES:,} bytes")
    return body.decode("utf-8", errors="replace")


def _series_key(labels: Mapping[str, str]) -> str:
    return json.dumps(sorted(labels.items()))


def parse_counter_series(text: str) -> dict[str, dict[str, Any]]:
    """Every labelled series of the generation-token counter, keyed by label set."""

    series: dict[str, dict[str, Any]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith(GENERATION_COUNTER):
            continue
        rest = line[len(GENERATION_COUNTER):]
        labels: dict[str, str] = {}
        if rest.startswith("{"):
            end = rest.rfind("}")
            if end < 0:
                continue
            labels = {k: v for k, v in _LABEL.findall(rest[1:end])}
            rest = rest[end + 1:]
        elif not rest[:1].isspace():
            continue  # another metric whose name starts with this one
        try:
            value = float(rest.split()[0])
        except (IndexError, ValueError):
            continue
        if not math.isfinite(value) or value < 0:
            continue
        key = _series_key(labels)
        if key in series:
            raise SavingsRefused(f"{GENERATION_COUNTER} lists the label set {labels} twice")
        series[key] = {"labels": labels, "value": value}
    return series


def read_counter_series(url: str) -> dict[str, dict[str, Any]]:
    series = parse_counter_series(fetch_metrics_text(url))
    if not series:
        raise SavingsRefused(f"{url} does not expose {GENERATION_COUNTER} (is it a vLLM /metrics endpoint?)")
    return series


def take_snapshot(url: str) -> dict[str, Any]:
    series = read_counter_series(url)
    return {
        "record_type": SNAPSHOT_RECORD_TYPE,
        "record_version": SNAPSHOT_RECORD_VERSION,
        "metrics_url": url,
        "counter": GENERATION_COUNTER,
        "series": [series[key] for key in sorted(series)],
        "taken_at": _iso(_now()),
    }


def _for_model(
    series: Mapping[str, Mapping[str, Any]], has_model_label: bool, model: str
) -> dict[str, Mapping[str, Any]]:
    if not has_model_label:
        return dict(series)
    return {k: v for k, v in series.items() if v["labels"].get("model_name") == model}


def measured_tokens(url: str, start_file: str, model: str) -> dict[str, Any]:
    """Per-series counter delta since the snapshot in ``start_file``, summed.

    Only series whose model_name label is ``model`` count (every series when
    the server reports no model_name label). A series that went down, or that
    disappeared, is a reset and is refused; a new series counts from 0.
    """

    try:
        start = json.loads(Path(start_file).expanduser().read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SavingsRefused(f"could not read --window-start {start_file}: {exc}") from None
    if not isinstance(start, dict) or start.get("record_type") != SNAPSHOT_RECORD_TYPE:
        raise SavingsRefused(f"{start_file} is not a 'throttle savings snapshot' file")
    entries = start.get("series")
    if not isinstance(entries, list) or not all(
        isinstance(e, dict) and isinstance(e.get("labels"), dict)
        and isinstance(e.get("value"), (int, float)) and not isinstance(e.get("value"), bool)
        and e["value"] >= 0
        for e in entries
    ):
        raise SavingsRefused(f"{start_file} has no usable per-series counter values")
    if start.get("metrics_url") != url:
        raise SavingsRefused(
            f"the snapshot was taken from {start.get('metrics_url')}, not {url}; "
            "a counter delta across two endpoints is meaningless"
        )
    before_all = {_series_key(e["labels"]): e for e in entries}
    now_all = read_counter_series(url)
    has_model_label = any(
        "model_name" in e["labels"] for e in [*before_all.values(), *now_all.values()]
    )
    before = _for_model(before_all, has_model_label, model)
    now = _for_model(now_all, has_model_label, model)
    if not now:
        raise SavingsRefused(
            f"no {GENERATION_COUNTER} series at {url} has model_name={model!r}, the checks' model"
        )
    missing = [before[k]["labels"] for k in before if k not in now]
    if missing:
        raise SavingsRefused(
            f"{GENERATION_COUNTER} series {missing} existed at the snapshot but are gone now: "
            "the server restarted or changed shape, so the tokens served this period are "
            "unknown. Take a new snapshot, or use --tokens"
        )
    per_series = []
    for key in sorted(now):
        start_value = float(before[key]["value"]) if key in before else 0.0
        end_value = float(now[key]["value"])
        if end_value < start_value:
            raise SavingsRefused(
                f"{GENERATION_COUNTER}{now[key]['labels']} went down ({start_value:,.0f} at "
                f"the snapshot, {end_value:,.0f} now): the counter reset, so the tokens "
                "served this period are unknown. Take a new snapshot, or use --tokens"
            )
        per_series.append({
            "labels": dict(now[key]["labels"]),
            "start": start_value,
            "end": end_value,
            "delta": end_value - start_value,
            "new_since_snapshot": key not in before,
        })
    new = [s["labels"] for s in per_series if s["new_since_snapshot"]]
    return {
        "count": int(round(sum(s["delta"] for s in per_series))),
        "source": SOURCE_MEASURED,
        "detail": f"{GENERATION_COUNTER} delta at {url}, summed over {len(per_series)} series"
        + (f" with model_name={model}" if has_model_label else ""),
        "window_start": start.get("taken_at"),
        "window_end": _iso(_now()),
        "series": per_series,
        "new_series_note": (
            f"{len(new)} series new since the snapshot, counted from 0: {new}" if new else None
        ),
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
    shape = workload_value(baseline, "workload_shape")
    if shape is None:
        workload_assumption = (
            "$/M is MEASURED by 'throttle check' on its fixed workload (95% CI across blocks); "
            "production traffic is assumed to cost per token what that workload did"
        )
    else:
        workload_assumption = (
            f"$/M is MEASURED by 'throttle check' on its synthetic {shape['profile']} "
            f"workload ({_shape_text(shape)}; 95% CI across blocks); production traffic "
            "is assumed to cost per token what that workload did"
        )
    statement = {
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
            workload_assumption,
            f"token count is {tokens['source']}",
        ],
    }
    if shape is not None:
        # Only for agent checks, so a default statement's JSON is unchanged.
        statement["workload_shape"] = shape
    return statement


def _config_text(side: Mapping[str, Any]) -> str:
    config = " ".join(f"{k}={v}" for k, v in sorted(side["config"].items())) or "-"
    label = side.get("label") or "-"
    return f"label {label}, config {config}"


def _dpm_text(dpm: Mapping[str, Any]) -> str:
    return (f"{_money(dpm['mean'])}/M (95% CI {_money(dpm['ci_low'])} to "
            f"{_money(dpm['ci_high'])})")


def format_statement(statement: Mapping[str, Any], st: "style_module.Style | None" = None) -> str:
    """Human statement: the conservative $ figure as the headline, in a panel.

    Every field of the statement is printed; the JSON form (--json) is separate
    and never styled.
    """

    st = style_module.Style() if st is None else st
    base, cand = statement["baseline"], statement["candidate"]
    tokens, savings = statement["tokens"], statement["savings"]
    indent = " " * 12
    rows: list[Any] = [
        "Verified savings (conservative): "
        + st.paint(_money(savings["conservative_usd"]), "1", "32"),
        (f"= ({_money(base['dollars_per_million']['ci_low'])} baseline CI low - "
         f"{_money(cand['dollars_per_million']['ci_high'])} candidate CI high) "
         f"x {tokens['count']:,} / 1e6", ("2",), "  "),
        (f"Point estimate (not the verified figure): {_money(savings['point_estimate_usd'])}", ("2",)),
        "",
        f"verdict     {st.verdict(statement['verdict'])} (re-judged)",
        (f"{indent}{statement['verdict_reason']}", (), indent),
        (f"period      {statement['period'] or '(not stated; pass --period-label)'}", ()),
        (f"model       {base['model']}", ()),
        (f"metric      {statement['metric_label']} (the primary metric of these checks)", (), indent),
        *(
            [(f"workload    {_shape_text(statement['workload_shape'])}", (), indent)]
            if statement.get("workload_shape") else []
        ),
        (f"baseline    {base['check_id']}  {base['gpu_count']} GPU(s)  {_config_text(base)}", (), indent),
        (f"{indent}{_dpm_text(base['dollars_per_million'])}  [MEASURED]", (), indent),
        (f"candidate   {cand['check_id']}  {cand['gpu_count']} GPU(s)  {_config_text(cand)}", (), indent),
        (f"{indent}{_dpm_text(cand['dollars_per_million'])}  [MEASURED]", (), indent),
    ]
    if statement["rate_factor"] != 1.0:
        rows.append((
            f"{indent}(recorded {_dpm_text(cand['dollars_per_million_recorded'])}; "
            "shown at the baseline's per-GPU rate)", (), indent
        ))
    rows.append((f"tokens      {tokens['count']:,}  [{tokens['source']}]", ()))
    if tokens.get("detail"):
        rows.append((f"{indent}{tokens['detail']}", (), indent))
    if tokens.get("new_series_note"):
        rows.append((f"{indent}note: {tokens['new_series_note']}", (), indent))
    lines = [
        st.panel(rows, title="Verified savings"),
        st.bold("Assumptions"),
        *(f"  - {item}" for item in statement["assumptions"]),
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
        print(f"Recorded {len(snapshot['series'])} series of {GENERATION_COUNTER} at "
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
            baseline = records[_find(records, args.baseline, "baseline", path)]
            tokens = measured_tokens(
                args.metrics_url, args.window_start, str(baseline["fingerprint"].get("model"))
            )
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
