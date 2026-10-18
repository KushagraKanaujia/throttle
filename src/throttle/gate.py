"""``throttle gate``: a CI cost gate between two recorded checks.

Exit 1 when the candidate is a calibrated MORE EXPENSIVE than the baseline,
or OUTPUT CHANGED (its answers changed length or hit the token cap). Exit 0
otherwise: CHEAPER, NO WINNER (within noise) and NOT CALIBRATED cannot prove
a regression, so they pass (NOT CALIBRATED says so loudly). Exit 2 on a usage
error.

The verdict is the same code ``throttle check`` uses (``compare_checks``),
judged locally from the two check JSON files (and the local check history,
for the run-to-run noise bound). When logged in, the gate is also sent to
Throttle Cloud (POST /ci/gate, $0.10 of credits) so it shows in your team's
history; ``--no-cloud`` skips that. A baseline given as an id that is not in
the local history is judged by Throttle Cloud.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from . import check as check_module
from . import cloud
from . import style as style_module

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_USAGE = 2

GATE_DESCRIPTION = (
    "CI cost gate: compare a candidate check with a baseline check and exit 1 if "
    "the candidate is MORE EXPENSIVE (beyond the run-to-run noise bound) or OUTPUT "
    "CHANGED; exit 0 otherwise. Inputs are files written by `throttle check --json` "
    "or check ids from the local history (--history-dir). When logged in to Throttle "
    "Cloud the result is also recorded there (POST /ci/gate, $0.10 of credits); "
    "--no-cloud keeps it local."
)


def add_gate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--baseline", required=True, metavar="CHECK_JSON|CHECK_ID",
        help="the baseline: a `throttle check --json` file, a local check id, or a "
        "Throttle Cloud check id (judged by the cloud; needs login)",
    )
    parser.add_argument(
        "--candidate", required=True, metavar="CHECK_JSON|CHECK_ID",
        help="the candidate: a `throttle check --json` file or a local check id",
    )
    parser.add_argument(
        "--history-dir", type=Path, metavar="PATH",
        help="check history used to look up ids and calibrate the noise bound "
        f"(default: ${check_module.HISTORY_ENV} or ~/.throttle/checks)",
    )
    parser.add_argument(
        "--max-increase", type=float, default=0.0, metavar="PCT",
        help="only fail a MORE EXPENSIVE verdict at least this many percent more "
        "expensive (default: 0, any calibrated increase fails)",
    )
    parser.add_argument(
        "--no-cloud", action="store_true",
        help="judge locally only; never contact Throttle Cloud even when logged in",
    )
    parser.add_argument(
        "--json", dest="json_output", type=Path, metavar="PATH",
        help="write the gate result as JSON",
    )


def _strip(record: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in record.items() if k not in ("comparison", "calibration")}


def _load(
    value: str, records: list[dict[str, Any]], role: str
) -> tuple[dict[str, Any] | None, str | None]:
    """(record, None) from a file or a local id; (None, id) for an unknown id;
    raises ValueError for an unreadable file."""

    path = Path(value)
    if path.is_file():
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise ValueError(f"--{role} {value}: not readable JSON ({exc})") from None
        if not isinstance(data, dict) or not check_module._valid_record(_strip(data)):
            raise ValueError(
                f"--{role} {value}: not a check record (write one with `throttle check --json PATH`)"
            )
        return _strip(data), None
    matches = [r for r in records if r.get("id") == value]
    if matches:
        return dict(matches[-1]), None
    return None, value


def _judge(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    # The history the verdict may use: everything recorded before the
    # candidate, plus the baseline itself when it came from a file.
    history = [r for r in records if r.get("id") != candidate.get("id")]
    if not any(r.get("id") == baseline.get("id") for r in history):
        history.append(dict(baseline))
    history.sort(key=lambda r: str(r.get("created_at") or ""))
    return check_module.compare_checks(baseline, candidate, history)


def _fails(comparison: Mapping[str, Any], max_increase: float) -> bool:
    verdict = comparison.get("verdict")
    if verdict == check_module.VERDICT_OUTPUT_CHANGED:
        return True
    if verdict == check_module.VERDICT_COSTLIER:
        pct = comparison.get("measured_delta_percent")
        return pct is None or pct >= max_increase
    return False


def handle_gate(args: argparse.Namespace) -> int:
    st = style_module.Style()
    directory = check_module.history_dir(args)
    records, _ = check_module.load_history(directory)
    try:
        baseline, baseline_cloud_id = _load(args.baseline, records, "baseline")
        candidate, missing = _load(args.candidate, records, "candidate")
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    if candidate is None:
        print(
            f"Error: --candidate {missing}: no such file, and no check with that id in "
            f"{directory / check_module.HISTORY_FILENAME}",
            file=sys.stderr,
        )
        return EXIT_USAGE

    client: cloud.CloudClient | None = None
    if not args.no_cloud:
        try:
            client = cloud.client_or_none()
        except cloud.CloudError as exc:
            print(f"warning: {exc}", file=sys.stderr)
    if baseline is None and client is None:
        print(
            f"Error: --baseline {baseline_cloud_id}: no such file, and no check with that id "
            f"in {directory / check_module.HISTORY_FILENAME}. A Throttle Cloud check id "
            "needs `throttle login` (or THROTTLE_API_KEY).",
            file=sys.stderr,
        )
        return EXIT_USAGE

    comparison = _judge(baseline, candidate, records) if baseline is not None else None
    result: dict[str, Any] = {
        "baseline_id": baseline.get("id") if baseline else baseline_cloud_id,
        "candidate_id": candidate.get("id"),
        "max_increase_percent": args.max_increase,
        "judged_by": "local" if comparison is not None else "cloud",
        "verdict": comparison.get("verdict") if comparison else None,
        "reason": comparison.get("reason") if comparison else None,
        "measured_delta_percent": comparison.get("measured_delta_percent") if comparison else None,
        "cloud": None,
    }

    if client is not None:
        body: dict[str, Any] = {"candidate": cloud.check_payload(candidate)}
        if baseline is not None:
            body["baseline"] = cloud.check_payload(baseline)
        else:
            body["baseline_check_id"] = baseline_cloud_id
        key = "gate:" + hashlib.sha256(
            f"{result['baseline_id']}|{candidate.get('id')}|{args.max_increase}".encode()
        ).hexdigest()[:32]
        try:
            response = client.ci_gate(body, key)
            result["cloud"] = {
                "pass": response.get("pass"),
                "verdict": response.get("verdict"),
                "charged_usd": response.get("charged_usd"),
                "credits_usd": response.get("credits_usd"),
            }
        except cloud.CloudError as exc:
            result["cloud"] = {"error": exc.kind, "message": str(exc)}
            if comparison is None:
                print(f"Error: {exc}", file=sys.stderr)
                print("The gate could not judge this change (the baseline is only in "
                      "Throttle Cloud).", file=sys.stderr)
                return EXIT_USAGE
            print(f"warning: {exc} The gate is judged locally.", file=sys.stderr)

    metric = candidate.get("metric", "output")
    label = check_module.METRIC_LABELS.get(metric, metric)
    print(f"{st.brand()} gate {st.dot} baseline {result['baseline_id']} -> candidate {candidate.get('id')}")
    if comparison is not None:
        before = (comparison.get("previous") or {})
        now = candidate["result"][metric]
        print(f"  before  {check_module._money(before.get('mean'))}/M {label}  ({check_module._ci_text(before)})")
        print(f"  now     {check_module._money(now.get('mean'))}/M {label}  ({check_module._ci_text(now)})")
        pct = comparison.get("measured_delta_percent")
        if pct is not None:
            print(f"  change  {pct:+.1f}%")
        print(f"  Verdict: {st.verdict(comparison['verdict'])}, {comparison['reason']}.")
        failed = _fails(comparison, args.max_increase)
    else:
        cloud_result = result["cloud"] or {}
        print(f"  Verdict (Throttle Cloud): {st.verdict(str(cloud_result.get('verdict')))}")
        result["verdict"] = cloud_result.get("verdict")
        failed = cloud_result.get("pass") is False
    cloud_result = result["cloud"]
    if cloud_result and "error" not in cloud_result:
        extra = cloud.credits_text(cloud_result)
        print(st.dim("  recorded in Throttle Cloud" + (f" ({extra})" if extra else "")))

    verdict = result["verdict"]
    if failed:
        why = (
            "the answers changed (length or token cap), so the $/M figure is not trustworthy"
            if verdict == check_module.VERDICT_OUTPUT_CHANGED
            else "the candidate costs more per token beyond the run-to-run noise"
        )
        print(f"FAIL: {verdict}: {why}. Exit code {EXIT_FAIL}.", file=sys.stderr)
    elif verdict == check_module.VERDICT_NOT_CALIBRATED:
        print(
            "PASS (unverified): NOT CALIBRATED, so this gate cannot prove a regression. "
            "Keep the check history between runs (e.g. the Actions cache) so the baseline "
            "has 3 repeat checks.",
            file=sys.stderr,
        )
    else:
        print(f"PASS: {verdict}.")
    result["pass"] = not failed
    if args.json_output:
        try:
            args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        except OSError as exc:
            print(f"Error: could not write {args.json_output}: {exc}", file=sys.stderr)
            return EXIT_USAGE
    return EXIT_FAIL if failed else EXIT_PASS
