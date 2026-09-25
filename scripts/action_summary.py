#!/usr/bin/env python3
"""Steps of the 'Throttle cost check' GitHub Action (action.yml).

Standard library only, so the first step can run on the caller's Python
before Throttle is installed. Every value arrives through environment
variables (never interpolated into a shell script), and nothing here prints
an API key: the key is read by Throttle itself from the variable named by
the api-key-env input.

Subcommands, in the order action.yml runs them:

  resolve        validate inputs, mask the endpoint, pick the pip spec and
                 the history cache key and directories
  merge-history  union the history restored from the cache into the working
                 history directory (dedupe, oldest first)
  run            run 'throttle check ... --json', record its exit code
  summarize      set outputs and write the Markdown job summary (no URLs or
                 hosts)
  gate           turn Throttle's exit code into the step's exit code

Local simulation: set GITHUB_OUTPUT, GITHUB_STEP_SUMMARY, RUNNER_TEMP and the
INPUT_* variables below, then call the subcommands in order.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

VERDICTS = ("CHEAPER", "MORE EXPENSIVE", "NO WINNER", "NOT CALIBRATED")
VERDICT_ERROR = "ERROR"
HISTORY_FILENAME = "checks.ndjson"
KEY_PREFIX = "throttle-check-v1"
# Options the action sets itself; passing them through extra-args would make
# the action read the wrong file or list history instead of measuring.
RESERVED_ARGS = ("--json", "--history-dir", "--history", "--url", "--endpoint-url")
_URL_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\s|)`'\"<>]+")


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def set_output(name: str, value: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        print(f"(GITHUB_OUTPUT unset) {name}={value}")
        return
    if "\n" in value:
        raise ValueError(f"output {name} must be one line")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"{name}={value}\n")


def command(kind: str, message: str) -> None:
    """A workflow command (::warning:: etc.); newlines escaped per the spec."""

    message = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::{kind}::{message}", flush=True)


def fail(message: str, code: int = 2) -> int:
    command("error", f"Throttle cost check: {message}")
    return code


# --------------------------------------------------------------------------
# resolve
# --------------------------------------------------------------------------


def _pip_spec(requested: str, action_ref: str) -> tuple[str | None, str]:
    """(pip requirement, how it was chosen); None means invalid input."""

    if requested and requested != "auto":
        if requested == "latest":
            return "throttle-pro", "latest from PyPI (throttle-version: latest)"
        if requested.startswith(("/", "./", "../")) or requested == ".":
            return requested, "local path (throttle-version input)"
        match = re.fullmatch(r"v?(\d+\.\d+\.\d+(?:[.-]?(?:a|b|rc|post|dev)\d+)?)", requested)
        if match:
            return f"throttle-pro=={match.group(1)}", "throttle-version input"
        return None, ""
    match = re.fullmatch(r"v?(\d+\.\d+\.\d+(?:[.-]?(?:a|b|rc|post|dev)\d+)?)", action_ref)
    if match:
        return f"throttle-pro=={match.group(1)}", f"same as the action ref {action_ref}"
    shown = action_ref or "(none: local action)"
    return "throttle-pro", f"latest from PyPI (action ref {shown} is not a release tag)"


def cache_prefix(history_key: str) -> str:
    """restore-keys prefix for one history key.

    actions/cache matches restore-keys by prefix, so 'throttle-check-v1-prod-'
    would also restore a cache saved for the key 'prod-eu'. The key's length
    goes in front of it: two different keys then never give prefixes where
    one starts with the other (same length means same key; different lengths
    differ within the length field or right after it).
    """

    return f"{KEY_PREFIX}-{len(history_key)}-{history_key}-"


def cmd_resolve() -> int:
    if sys.version_info < (3, 11):
        return fail(
            f"Python {sys.version_info.major}.{sys.version_info.minor} found; Throttle "
            "needs 3.11+. Add actions/setup-python (python-version: '3.12') before this action."
        )
    url = env("INPUT_URL")
    rate = env("INPUT_GPU_HOURLY_RATE")
    if not url:
        return fail("the 'url' input is required")
    parsed = urlsplit(url)
    if not parsed.scheme or not parsed.hostname:
        return fail("the 'url' input is not a URL (expected e.g. https://vllm.internal:8000)")
    # Mask before anything else can print the endpoint (Throttle's own log
    # names it). Masks cover logs only; the summary is scrubbed separately.
    for secret in {url, url.rstrip("/"), parsed.netloc, parsed.hostname}:
        if secret and len(secret) >= 3:
            print(f"::add-mask::{secret}", flush=True)
    if not rate:
        return fail("the 'gpu-hourly-rate' input is required (what the GPUs cost you per hour)")
    try:
        if float(rate) <= 0 or not math.isfinite(float(rate)):
            raise ValueError
    except ValueError:
        return fail("'gpu-hourly-rate' must be a positive number of dollars per hour")
    gate = env("INPUT_FAIL_IF_COSTLIER")
    if gate:
        try:
            if float(gate) < 0 or not math.isfinite(float(gate)):
                raise ValueError
        except ValueError:
            return fail("'fail-if-costlier' must be a non-negative percent, e.g. 5")
    policy = env("INPUT_NOT_CALIBRATED", "warn").lower() or "warn"
    if policy not in ("warn", "fail"):
        return fail("'not-calibrated' must be warn or fail")
    if policy == "fail" and not gate:
        command(
            "warning",
            "not-calibrated: fail has no effect without fail-if-costlier "
            "(there is no gate to be uncalibrated)",
        )
    try:
        extra = shlex.split(env("INPUT_EXTRA_ARGS"))
    except ValueError as exc:
        return fail(f"'extra-args' could not be parsed: {exc}")
    for arg in extra:
        if arg.split("=", 1)[0] in RESERVED_ARGS:
            return fail(f"'extra-args' may not contain {arg.split('=', 1)[0]}; the action sets it")

    spec, why = _pip_spec(env("INPUT_THROTTLE_VERSION"), env("ACTION_REF"))
    if spec is None:
        return fail("'throttle-version' must be auto, latest, a version like 0.4.1, or a local path")

    model = env("INPUT_MODEL")
    history_key = env("INPUT_HISTORY_KEY")
    if not history_key:
        # A hash, so the endpoint never appears in the cache list.
        basis = f"{url.rstrip('/').lower()}|{model}"
        history_key = hashlib.sha256(basis.encode()).hexdigest()[:16]
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,200}", history_key):
        return fail("'history-key' may only contain letters, digits, '.', '_' and '-'")
    prefix = cache_prefix(history_key)
    unique = f"{env('GITHUB_RUN_ID', 'local')}-{env('GITHUB_RUN_ATTEMPT', '1')}-{time.time_ns()}"
    temp = Path(env("RUNNER_TEMP") or "/tmp")
    slug = hashlib.sha256(history_key.encode()).hexdigest()[:12]
    work = temp / "throttle-check" / slug
    work.mkdir(parents=True, exist_ok=True)

    print(f"Throttle package: {spec} ({why})")
    print(f"History cache key prefix: {prefix}")
    set_output("pip-spec", spec)
    set_output("cache-prefix", prefix)
    set_output("cache-key", prefix + unique)
    set_output("history-dir", str(work))
    set_output("venv", str(temp / "throttle-check-venv"))
    set_output("json-path", str(work / f"check-{unique}.json"))
    return 0


# --------------------------------------------------------------------------
# merge-history
# --------------------------------------------------------------------------


def _created(line: str) -> str:
    try:
        value = json.loads(line).get("created_at")
    except (ValueError, AttributeError):
        return ""
    return value if isinstance(value, str) else ""


def cmd_merge_history(restored: str, work: str) -> int:
    """Union restored + local history into work/, oldest first.

    Throttle takes the LAST record of an endpoint as the baseline, so the
    merged file is ordered by created_at. Unreadable lines are kept (Throttle
    skips and reports them) rather than silently dropped.
    """

    lines: list[str] = []
    seen: set[str] = set()
    for directory in (Path(restored), Path(work)):
        path = directory / HISTORY_FILENAME
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and line not in seen:
                seen.add(line)
                lines.append(line)
    lines.sort(key=_created)  # stable: undated lines stay first, in order
    target = Path(work) / HISTORY_FILENAME
    Path(work).mkdir(parents=True, exist_ok=True)
    target.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    print(f"History: {len(lines)} earlier check(s) available for this endpoint key")
    return 0


def cmd_stage_history(work: str, cache_dir: str) -> int:
    """Copy the working history into the fixed cache path before saving."""

    source = Path(work) / HISTORY_FILENAME
    target = Path(cache_dir)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    if source.is_file():
        shutil.copy2(source, target / HISTORY_FILENAME)
    return 0


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


def build_argv(throttle: str, history: str, json_path: str) -> list[str]:
    argv = [throttle, "check", "--url", env("INPUT_URL"),
            "--gpu-hourly-rate", env("INPUT_GPU_HOURLY_RATE")]
    for flag, name in (
        ("--model", "INPUT_MODEL"),
        ("--blocks", "INPUT_BLOCKS"),
        ("--requests-per-block", "INPUT_REQUESTS_PER_BLOCK"),
        ("--concurrency", "INPUT_CONCURRENCY"),
        ("--max-tokens", "INPUT_MAX_TOKENS"),
        ("--api-key-env", "INPUT_API_KEY_ENV"),
        ("--fail-if-costlier", "INPUT_FAIL_IF_COSTLIER"),
    ):
        if env(name):
            argv += [flag, env(name)]
    label = env("INPUT_LABEL") or env("GITHUB_SHA")[:7]
    if label:
        argv += ["--label", label]
    for line in os.environ.get("INPUT_CONFIG", "").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            argv += ["--config", line]
    argv += shlex.split(env("INPUT_EXTRA_ARGS"))
    argv += ["--history-dir", history, "--json", json_path]
    return argv


def cmd_run(venv: str, history: str, json_path: str) -> int:
    bindir = "Scripts" if os.name == "nt" else "bin"
    throttle = str(Path(venv) / bindir / "throttle")
    key_env = env("INPUT_API_KEY_ENV")
    if key_env and not os.environ.get(key_env):
        command(
            "warning",
            f"api-key-env names {key_env}, but it is empty in this step's env; "
            "requests go out without an Authorization header",
        )
    Path(json_path).unlink(missing_ok=True)
    argv = build_argv(throttle, history, json_path)
    print("Running: throttle check (arguments not echoed; the endpoint is masked)", flush=True)
    completed = subprocess.run(argv, check=False)
    code = completed.returncode
    print(f"throttle check exited {code}", flush=True)
    set_output("exit-code", str(code))
    return 0


# --------------------------------------------------------------------------
# summarize
# --------------------------------------------------------------------------


class Scrubber:
    """Removes URLs and the endpoint's host from text bound for the summary."""

    def __init__(self, url: str) -> None:
        parsed = urlsplit(url) if url else None
        needles = {url, url.rstrip("/")}
        if parsed:
            needles |= {parsed.netloc, parsed.hostname or ""}
        self.needles = sorted((n for n in needles if n and len(n) >= 3), key=len, reverse=True)

    def __call__(self, value: Any) -> str:
        text = "" if value is None else str(value)
        text = _URL_RE.sub("[url hidden]", text)
        for needle in self.needles:
            text = re.sub(re.escape(needle), "[endpoint]", text, flags=re.IGNORECASE)
        return text


def _cell(text: str) -> str:
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").replace("`", "'")


def _money(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        return "n/a"
    if abs(value) >= 1000:
        return f"${value:,.0f}"
    if abs(value) >= 1:
        return f"${value:,.2f}"
    return f"${value:.4f}"


def _ci(summary: Mapping[str, Any] | None) -> str:
    if not summary or summary.get("ci_low") is None:
        return "n/a"
    return f"{_money(summary.get('ci_low'))} to {_money(summary.get('ci_high'))}"


def _pct(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        return "n/a"
    return f"{value:+.1f}%"


METRIC_LABELS = {"output": "output tokens", "input": "input tokens", "total": "tokens"}


def render_summary(
    record: Mapping[str, Any] | None, code: int, scrub: Scrubber, gate: str
) -> tuple[str, dict[str, str]]:
    outputs = {"verdict": VERDICT_ERROR, "cost-per-million": "", "change-percent": "", "check-id": ""}
    lines = []
    if record is None:
        lines += [
            "### Throttle cost check: ERROR",
            "",
            f"`throttle check` exited {code} and wrote no result "
            "(1 = measurement failed, 2 = usage error). Nothing was recorded; see the step log.",
        ]
        return "\n".join(lines) + "\n", outputs

    metric = record.get("metric") or "output"
    unit = f"$/M {METRIC_LABELS.get(metric, metric)}"
    now = (record.get("result") or {}).get(metric) or {}
    fingerprint = record.get("fingerprint") or {}
    comparison = record.get("comparison")
    first = comparison is None
    verdict = "NOT CALIBRATED" if first else str(comparison.get("verdict"))
    if verdict not in VERDICTS:
        verdict = VERDICT_ERROR
    outputs["verdict"] = verdict
    outputs["check-id"] = scrub(record.get("id"))
    mean = now.get("mean")
    if isinstance(mean, (int, float)) and math.isfinite(mean):
        outputs["cost-per-million"] = f"{mean:.6g}"
    change = None if first else comparison.get("measured_delta_percent")
    if isinstance(change, (int, float)) and math.isfinite(change):
        outputs["change-percent"] = f"{change:.2f}"

    lines.append(f"### Throttle cost check: {verdict}")
    lines.append("")
    label = scrub(fingerprint.get("label")) or "-"
    rate = fingerprint.get("gpu_hourly_rate_usd")
    lines.append(
        f"Model `{_cell(scrub(fingerprint.get('model')))}`, label `{_cell(label)}`, "
        f"GPU rate {_money(rate)}/hr (ASSUMED: from the gpu-hourly-rate input). "
        "Tokens and time are MEASURED."
    )
    lines.append("")
    lines.append(f"| | {unit} | 95% CI | label |")
    lines.append("|---|---|---|---|")
    if not first:
        before = comparison.get("previous") or {}
        lines.append(
            f"| before | {_money(before.get('mean'))} | {_ci(before)} | "
            f"{_cell(scrub(comparison.get('previous_label') or '-'))} |"
        )
    lines.append(f"| now | {_money(mean)} | {_ci(now)} | {_cell(label)} |")
    lines.append("")

    if first:
        lines.append(
            "**Verdict: NOT CALIBRATED.** This is the first check for this endpoint "
            "in the restored history, so there is nothing to compare with yet. The next "
            "run is compared against it."
        )
    else:
        if comparison.get("delta_percent") is not None:
            change_line = f"Change: {_pct(comparison.get('delta_percent'))}"
            if comparison.get("rate_changed"):
                change_line += (
                    f" ({_pct(comparison.get('measured_delta_percent'))} measured, at the "
                    "baseline's GPU rate; the rest is the ASSUMED rate change you typed)"
                )
            lines.append(change_line + ".")
            lines.append("")
        reason = scrub(comparison.get("reason"))
        lines.append(f"**Verdict: {verdict}**" + (f": {reason}." if reason else "."))
    lines.append("")

    noise = None if first else comparison.get("noise")
    if first:
        noise_text = "not measured (needs 3 earlier checks of one unchanged config within 24 h)"
    elif noise is None:
        noise_text = "not evaluated (" + (scrub(comparison.get("reason")) or "no comparable baseline") + ")"
    elif noise.get("calibrated"):
        noise_text = (
            f"calibrated: {noise.get('floor_percent', 0):.1f}% "
            f"(t x sqrt(2) x {noise.get('sd_percent', 0):.1f}% run-to-run SD, "
            f"{noise.get('degrees_of_freedom')} df, from {noise.get('n_checks')} earlier "
            f"checks within {noise.get('window_hours', 24):g} h)"
        )
    else:
        noise_text = (
            f"not calibrated yet: {noise.get('degrees_of_freedom', 0)} of 2 degrees of "
            "freedom (run the same config 3 times within 24 h)"
        )
    lines.append(f"Noise floor: {noise_text}.")
    lines.append("")

    if not first:
        changes = list(comparison.get("changes") or [])
        for field in comparison.get("workload_differs") or []:
            changes.append({
                "key": f"workload.{field}",
                "old": (comparison.get("previous_workload") or {}).get(field),
                "new": (comparison.get("current_workload") or {}).get(field),
            })
        if changes:
            lines.append("<details open><summary>What changed</summary>")
            lines.append("")
            lines.append("| setting | before | now |")
            lines.append("|---|---|---|")
            for item in changes:
                key = str(item.get("key"))
                if key == "endpoint":
                    before, after = "(hidden)", "(hidden)"
                else:
                    before = scrub(item.get("old")) if item.get("old") is not None else "(not set)"
                    after = scrub(item.get("new")) if item.get("new") is not None else "(not set)"
                lines.append(f"| {_cell(scrub(key))} | {_cell(before)} | {_cell(after)} |")
            lines.append("")
            lines.append("</details>")
        else:
            lines.append(
                "What changed: nothing Throttle can see (add settings with the `config` input)."
            )
        for note in comparison.get("notes") or []:
            lines.append(f"- note: {_cell(scrub(note))}")
        lines.append("")

    gate_text = f"fail-if-costlier {gate}%" if gate else "no gate (fail-if-costlier not set)"
    lines.append(
        f"Gate: {gate_text}. `throttle check` exit code {code}. "
        f"Check id `{_cell(outputs['check-id'])}`."
    )
    return "\n".join(lines) + "\n", outputs


def cmd_summarize(json_path: str) -> int:
    code_text = env("THROTTLE_EXIT_CODE")
    code = int(code_text) if code_text.lstrip("-").isdigit() else 1
    record = None
    path = Path(json_path)
    if path.is_file():
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            record = None
        if record is not None:
            # The comparison's baseline label is not in the comparison block;
            # look it up in the history so the table can show it.
            comparison = record.get("comparison")
            if isinstance(comparison, dict):
                comparison["previous_label"] = _label_of(
                    Path(env("HISTORY_DIR")) / HISTORY_FILENAME, comparison.get("previous_id")
                )
    scrub = Scrubber(env("INPUT_URL"))
    markdown, outputs = render_summary(record, code, scrub, env("INPUT_FAIL_IF_COSTLIER"))
    for name, value in outputs.items():
        set_output(name, value)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(markdown + "\n")
    else:
        print(markdown)
    copy = os.environ.get("THROTTLE_ACTION_SUMMARY_COPY")
    if copy:
        # For the action's self-test: the runner keeps step summaries in
        # internal files, so the test asserts on this copy instead.
        with open(copy, "a", encoding="utf-8") as handle:
            handle.write(markdown + "\n")
    print(f"verdict={outputs['verdict']} cost-per-million={outputs['cost-per-million']} "
          f"change-percent={outputs['change-percent']}")
    return 0


def _label_of(history: Path, check_id: Any) -> str | None:
    if not check_id or not history.is_file():
        return None
    for line in history.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and record.get("id") == check_id:
            return (record.get("fingerprint") or {}).get("label")
    return None


# --------------------------------------------------------------------------
# gate
# --------------------------------------------------------------------------


def cmd_gate() -> int:
    code_text = env("THROTTLE_EXIT_CODE")
    code = int(code_text) if code_text.lstrip("-").isdigit() else 1
    verdict = env("VERDICT")
    policy = env("INPUT_NOT_CALIBRATED", "warn").lower() or "warn"
    gate = env("INPUT_FAIL_IF_COSTLIER")
    if code == 4:
        command(
            "error",
            f"Throttle cost check: calibrated MORE EXPENSIVE by at least {gate}% "
            "(exit 4). See the job summary.",
        )
        return 4
    if code == 5 and verdict == "NO WINNER":
        # Throttle's exit 5 with NO WINNER: the baseline ran a different
        # workload (e.g. another prompt cache mode, as after upgrading from
        # 0.4.0), so the gate judged nothing. Same policy as NOT CALIBRATED.
        message = (
            "Throttle cost check: the baseline ran a different workload (see 'What "
            "changed' in the job summary), so fail-if-costlier could not judge this "
            "change. Keep the workload inputs identical; after an upgrade from 0.4.0, "
            "run 3 checks of the unchanged config to rebuild the baseline."
        )
        if policy == "fail":
            command("error", message + " (not-calibrated: fail)")
            return 5
        command("warning", message + " (not-calibrated: warn, so the job passes)")
        return 0
    if code == 5 or (code == 0 and gate and verdict == "NOT CALIBRATED"):
        # 5 is Throttle's own 'gate could not judge'. A first check (no
        # baseline) exits 0 in Throttle but is just as unjudged, so it
        # follows the same policy.
        message = (
            "Throttle cost check: NOT CALIBRATED, so fail-if-costlier could not judge "
            "this change. Run-to-run noise needs 3 checks of one unchanged config "
            "within 24 h; see docs/github-action.md."
        )
        if policy == "fail":
            command("error", message + " (not-calibrated: fail)")
            return 5
        command("warning", message + " (not-calibrated: warn, so the job passes)")
        return 0
    if code != 0:
        command("error", f"Throttle cost check: throttle check exited {code}; see the log.")
        return code
    if verdict == "NOT CALIBRATED":
        command("notice", "Throttle cost check: NOT CALIBRATED (no gate set).")
    return 0


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    name, rest = argv[0], argv[1:]
    handlers = {
        "resolve": (cmd_resolve, 0),
        "merge-history": (cmd_merge_history, 2),
        "stage-history": (cmd_stage_history, 2),
        "run": (cmd_run, 3),
        "summarize": (cmd_summarize, 1),
        "gate": (cmd_gate, 0),
    }
    if name not in handlers or len(rest) != handlers[name][1]:
        print(f"usage: action_summary.py {name} ... (see the module docstring)", file=sys.stderr)
        return 2
    return handlers[name][0](*rest)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
