"""The first-run path: a self-calibrating first check, plain display on noisy
machines, and an agent workload that is its own baseline.

Every test drives the real CLI entry point against the fake servers the other
check tests use, and asserts on what a user sees and on the history file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx
import pytest

import throttle.check as check_module
from throttle.cli import main
from throttle.style import strip_ansi

from test_agent_workload import AGENT_ARGS, AgentFakeServer
from test_check import (  # noqa: F401  (history is a fixture)
    URL,
    FakeClock,
    FakeServer,
    constant,
    history,
    rewrite_history,
    run_check,
    run_clocked,
)


@pytest.fixture(autouse=True)
def not_in_ci(monkeypatch):
    # Calibration is off in CI (CI / GITHUB_ACTIONS); these tests exercise it.
    for name in check_module.CI_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

CALIBRATED_LINE = re.compile(
    r"Calibrated \(noise bound (\d+\.\d)%\)\. Now change one setting and run the same "
    r"command again for a verdict\."
)


def records(history: Path) -> list[dict]:
    return [json.loads(line) for line in (history / "checks.ndjson").read_text().splitlines()]


def flat(out: str) -> str:
    return " ".join(
        " ".join(line.strip().strip("│|").split()) for line in strip_ansi(out).splitlines()
    )


# --------------------------------------------------------------------------
# 1. Self-calibrating first check
# --------------------------------------------------------------------------


def test_fresh_history_calibrates_in_one_command_and_judges_in_two(
    history, monkeypatch, capsys
):
    noisy = [0.010, 0.014, 0.011, 0.013, 0.012]
    code, out = run_check(
        monkeypatch, capsys, FakeServer(lambda b: noisy[b % 5]), "--label", "before",
        calibrate=True,
    )
    assert code == 0, out
    assert "calibrating: run 1 of 3" in out
    assert "calibrating: run 2 of 3, about " in out and " s left (run 1 took " in out
    assert "calibrating: run 3 of 3" in out
    assert "calibrate  no recent repeats of this config: Throttle runs this check up to 3 times" in out
    # Each run is recorded exactly as a separate check: own id and run id,
    # compared with the check before it.
    saved = records(history)
    assert len(saved) == 3
    assert len({r["id"] for r in saved}) == 3
    assert len({r["workload"]["prompt_nonce"]["run_id"] for r in saved}) == 3
    assert "baseline_id" not in saved[0]
    assert saved[1]["baseline_id"] == saved[0]["id"]
    assert saved[2]["baseline_id"] == saved[1]["id"]
    # Each record says it came from one back-to-back calibration.
    assert {r["calibration_group"] for r in saved} == {saved[0]["id"]}
    # Calibration runs print one plain line, not degrees of freedom.
    assert "Baseline 2 of 3 recorded; unchanged server, yet " in out
    assert "this is the noise Throttle is measuring." in out
    assert "degree(s) of freedom" not in out and "t x sqrt(2)" not in out
    panel = flat(out)
    assert "CALIBRATED 3 checks of this config; run-to-run noise bound" in panel
    assert "next change one setting, then run the same command again for a verdict" in panel
    # The command ends with the calibrated line, and the bound is exactly
    # the one the next check is judged with (statistics unchanged).
    last_line = strip_ansi(out).strip().splitlines()[-1]
    match = CALIBRATED_LINE.fullmatch(last_line)
    assert match, last_line
    expected = check_module.run_to_run_noise(saved, saved[-1], {**saved[-1], "id": None}, "output")
    assert expected["calibrated"] and expected["degrees_of_freedom"] == 2
    assert match.group(1) == f"{expected['floor_percent']:.1f}"

    # Second command, after a change: a calibrated verdict, one check only.
    code, out = run_check(
        monkeypatch, capsys, FakeServer(constant(0.08)), "--label", "after",
        "--config", "max_num_seqs=64", calibrate=True,
    )
    assert code == 0, out
    assert "calibrating" not in out
    assert len(records(history)) == 4
    assert "Verdict: MORE EXPENSIVE" in out
    assert f"run-to-run noise bound {match.group(1)}%" in out
    judged = check_module.judge_recorded(records(history), 3)[1]
    assert judged["noise"]["floor_percent"] == expected["floor_percent"]


def test_no_calibrate_keeps_the_single_run(history, monkeypatch, capsys):
    code, out = run_check(monkeypatch, capsys, FakeServer(constant(0.01)))
    assert code == 0, out
    assert len(records(history)) == 1
    assert "calibrating" not in out
    assert "FIRST CHECK" in flat(out)


def test_calibration_only_runs_the_repeats_still_missing(history, monkeypatch, capsys):
    run_check(monkeypatch, capsys, FakeServer(constant(0.01)), "--label", "x")
    code, out = run_check(
        monkeypatch, capsys, FakeServer(constant(0.01)), "--label", "x", calibrate=True
    )
    assert code == 0, out
    assert "calibrating: run 1 of 2" in out and "calibrating: run 2 of 2" in out
    assert "run 3 of" not in out
    assert len(records(history)) == 3
    assert CALIBRATED_LINE.search(strip_ansi(out))
    # Already calibrated: a further unchanged check runs once and is judged.
    code, out = run_check(
        monkeypatch, capsys, FakeServer(constant(0.01)), "--label", "x", calibrate=True
    )
    assert "calibrating" not in out
    assert len(records(history)) == 4
    assert "Verdict: NO WINNER" in out


def test_no_calibration_with_no_save_or_a_ci_gate(history, monkeypatch, capsys):
    code, out = run_check(
        monkeypatch, capsys, FakeServer(constant(0.01)), "--no-save", calibrate=True
    )
    assert code == 0, out
    assert "calibrating" not in out
    assert not (history / "checks.ndjson").exists()
    code, out = run_check(
        monkeypatch, capsys, FakeServer(constant(0.01)), "--fail-if-costlier", "5",
        calibrate=True,
    )
    assert code == 0, out
    assert "calibrating" not in out
    assert len(records(history)) == 1


def test_calibration_shares_one_time_budget_and_says_how_to_continue(
    history, monkeypatch, capsys
):
    # Every _monotonic() reading is 200 s after the previous one, so run 1 (two
    # readings apart: 400 s) leaves nothing of the 300 s budget for run 2.
    ticks = iter(range(0, 10**6, 200))
    monkeypatch.setattr(check_module, "_monotonic", lambda: float(next(ticks)))
    code, out = run_check(
        monkeypatch, capsys, FakeServer(constant(0.01)), "--label", "slow", calibrate=True
    )
    assert code == 0, out
    assert "calibrating: run 1 of 3" in out and "run 2 of 3" not in out
    assert len(records(history)) == 1
    assert "Stopped calibrating after 1 run(s): the next run would take about 400 s, but only 0 s of the" in out
    assert "--max-elapsed-seconds 300 limit is left. Run the same command again to " \
        "record the other 2" in out
    assert "NOT CALIBRATED YET baseline 1 of 3 recorded" in flat(out)
    assert not CALIBRATED_LINE.search(strip_ansi(out))

    # Running the same command again picks up where it stopped.
    monkeypatch.setattr(check_module, "_monotonic", __import__("time").monotonic)
    code, out = run_check(
        monkeypatch, capsys, FakeServer(constant(0.01)), "--label", "slow", calibrate=True
    )
    assert code == 0, out
    assert "calibrating: run 1 of 2" in out
    assert len(records(history)) == 3
    assert CALIBRATED_LINE.search(strip_ansi(out))


def test_calibration_json_keeps_the_raw_detail(history, monkeypatch, capsys, tmp_path):
    json_path = tmp_path / "calibrated.json"
    code, out = run_check(
        monkeypatch, capsys, FakeServer(lambda b: [0.01, 0.02][b % 2]),
        "--json", str(json_path), calibrate=True,
    )
    assert code == 0, out
    written = json.loads(json_path.read_text())
    saved = records(history)
    assert written["id"] == saved[-1]["id"]
    calibration = written["calibration"]
    assert calibration["ids"] == [r["id"] for r in saved]
    assert calibration["noise"]["degrees_of_freedom"] == 2
    assert "t(0.975, df) x sqrt(2)" in calibration["noise"]["rule"]
    # The last repeat's own comparison is kept too, with its df reason.
    assert "1 degree(s) of freedom" in written["comparison"]["reason"]
    # Run 1 had nothing to compare with.
    assert calibration["first_comparison"] is None


def test_calibration_json_keeps_run_one_against_the_real_baseline(
    history, monkeypatch, capsys, tmp_path
):
    # One unchanged check from two days ago: stale, so run 1 is NOT CALIBRATED
    # against it and the config is re-measured now.
    run_check(monkeypatch, capsys, FakeServer(constant(0.01)), "--label", "a")
    old_id = records(history)[0]["id"]
    rewrite_history(history, lambda _i, r: r.update(created_at="2020-01-01T00:00:00Z"))
    json_path = tmp_path / "c.json"
    code, out = run_check(
        monkeypatch, capsys, FakeServer(constant(0.01)), "--label", "a",
        "--json", str(json_path), calibrate=True,
    )
    assert code == 0, out
    assert len(records(history)) == 4
    first = json.loads(json_path.read_text())["calibration"]["first_comparison"]
    assert first["previous_id"] == old_id
    assert first["verdict"] == "NOT CALIBRATED" and "old" in first["reason"]


# --------------------------------------------------------------------------
# Calibration never hides a real comparison, and respects every limit
# --------------------------------------------------------------------------


def test_calibration_fits_under_request_and_token_limits(history, monkeypatch, capsys):
    # A default check here is 25 requests and up to 1,348 output tokens.
    code, out = run_check(
        monkeypatch, capsys, FakeServer(constant(0.01)), "--max-requests", "30",
        calibrate=True,
    )
    assert code == 0, out
    assert "calibrating" not in out and "calibrate  " not in out
    assert len(records(history)) == 1
    code, out = run_check(
        monkeypatch, capsys, FakeServer(constant(0.01)), "--label", "t",
        "--max-total-requested-tokens", "2000", calibrate=True,
    )
    assert "calibrating" not in out
    assert len(records(history)) == 2


def test_calibration_runs_only_as_many_repeats_as_the_limits_allow(
    history, monkeypatch, capsys
):
    # Room for exactly 2 runs: it runs 2, says so, and stops.
    code, out = run_check(
        monkeypatch, capsys, FakeServer(constant(0.01)), "--max-requests", "60",
        calibrate=True,
    )
    assert code == 0, out
    assert "up to 2 times back to back" in out
    assert "at most 50 requests and 2,696 output tokens in all (--max-requests 60" in out
    assert "calibrating: run 2 of 2" in out and "run 3 of" not in out
    assert len(records(history)) == 2
    assert "Stopped calibrating after 2 run(s): another run would pass --max-requests 60" in out


def test_no_calibration_after_a_config_change(history, monkeypatch, capsys):
    for _ in range(2):
        run_check(monkeypatch, capsys, FakeServer(constant(0.01)), "--label", "a")
    code, out = run_check(
        monkeypatch, capsys, FakeServer(constant(0.02)), "--label", "b",
        "--config", "k=v", calibrate=True,
    )
    assert code == 0, out
    assert len(records(history)) == 3  # one check, not a re-baseline of b
    assert "calibrating" not in out and "calibrate  " not in out
    assert "Calibrated (" not in out and "CALIBRATED  3 checks" not in out
    assert "Verdict: NOT CALIBRATED" in out


def test_no_calibration_with_against(history, monkeypatch, capsys):
    run_check(monkeypatch, capsys, FakeServer(constant(0.01)), "--label", "a")
    first = records(history)[0]["id"]
    code, out = run_check(
        monkeypatch, capsys, FakeServer(constant(0.01)), "--label", "a",
        "--against", first, calibrate=True,
    )
    assert code == 0, out
    assert "calibrating" not in out
    assert len(records(history)) == 2


def test_stale_baseline_then_config_change_is_not_rebaselined(history, monkeypatch, capsys):
    for _ in range(3):
        run_check(monkeypatch, capsys, FakeServer(constant(0.01)), "--label", "a")
    rewrite_history(history, lambda _i, r: r.update(created_at="2020-01-01T00:00:00Z"))
    code, out = run_check(
        monkeypatch, capsys, FakeServer(constant(0.02)), "--label", "b", calibrate=True,
    )
    assert code == 0, out
    assert len(records(history)) == 4
    assert "calibrating" not in out and "Calibrated (" not in out
    assert "Verdict: NOT CALIBRATED" in out and "old; run-to-run noise is only" in out


def test_a_repeat_that_reaches_a_verdict_shows_it(history, monkeypatch, capsys, tmp_path):
    real = check_module.compare_checks
    calls = []

    def compare(previous, current, history_records=()):
        result = real(previous, current, history_records)
        calls.append(result)
        if len(calls) == 1:  # run 2 against run 1: pretend it was judged
            result = {**result, "verdict": "MORE EXPENSIVE", "reason": "stand-in verdict"}
        return result

    monkeypatch.setattr(check_module, "compare_checks", compare)
    json_path = tmp_path / "v.json"
    code, out = run_check(
        monkeypatch, capsys, FakeServer(constant(0.01)), "--json", str(json_path),
        calibrate=True,
    )
    assert code == 0, out
    assert len(records(history)) == 2
    assert "Verdict: MORE EXPENSIVE, stand-in verdict." in out
    assert "Calibrated (" not in out and "CALIBRATED  " not in flat(out)
    written = json.loads(json_path.read_text())
    assert written["comparison"]["verdict"] == "MORE EXPENSIVE"
    assert written["calibration"]["stopped_early"] == "a repeat reached a verdict"


@pytest.mark.parametrize("name, value", [("CI", "true"), ("GITHUB_ACTIONS", "true"), ("CI", "1")])
def test_ci_runs_one_check(history, monkeypatch, capsys, name, value):
    monkeypatch.setenv(name, value)
    code, out = run_check(monkeypatch, capsys, FakeServer(constant(0.01)), calibrate=True)
    assert code == 0, out
    assert "calibrating" not in out
    assert len(records(history)) == 1


def test_ci_false_still_calibrates(history, monkeypatch, capsys):
    monkeypatch.setenv("CI", "false")
    code, out = run_check(monkeypatch, capsys, FakeServer(constant(0.01)), calibrate=True)
    assert code == 0, out
    assert len(records(history)) == 3


def test_a_timed_out_repeat_stops_calibration_and_keeps_outputs(
    history, monkeypatch, capsys, tmp_path
):
    # Run 1 (warm-up + 5 blocks of 4) is fast; from run 2 on every request
    # takes 5 s, so run 2 hits the time left of a 3 s budget.
    server = FakeServer(lambda block: 0.0 if block < 5 else 5.0)
    json_path = tmp_path / "t.json"
    code, out = run_check(
        monkeypatch, capsys, server, "--max-elapsed-seconds", "3", "--json", str(json_path),
        calibrate=True,
    )
    assert code == 0, out
    assert len(records(history)) == 1
    assert "Calibration run 2 failed: stopped at the time left of the " \
        "--max-elapsed-seconds 3s budget (" in out
    assert "Stopped calibrating after 1 run(s): run 2 failed" in out
    assert "Run the same command again to record the other 2" in out
    written = json.loads(json_path.read_text())
    assert written["id"] == records(history)[0]["id"]
    assert written["calibration"]["stopped_early"].startswith("run 2 failed")


# --------------------------------------------------------------------------
# 3. Display: $0 CI floor and the noisy-machine hint
# --------------------------------------------------------------------------


def test_printed_ci_floor_is_zero_but_json_keeps_the_raw_bound(
    history, monkeypatch, capsys, tmp_path
):
    # Blocks of 1 s, 1 s and 100 s: the t interval's low end is far below $0.
    monkeypatch.setattr(check_module, "time", FakeClock([1.0, 1.0, 100.0]))
    monkeypatch.setattr(check_module, "_TRANSPORT", httpx.MockTransport(
        FakeServer(constant(0.0), fixed_usage=(100, 50)).handle
    ))
    json_path = tmp_path / "check.json"
    code = main([
        "check", "--url", URL, "--model", "mock-model", "--gpu-hourly-rate", "2.00",
        "--blocks", "3", "--requests-per-block", "4", "--concurrency", "2",
        "--warmup", "0", "--no-calibrate", "--monthly-tokens", "1B", "--json", str(json_path),
    ])
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert code == 0, out
    record = json.loads(json_path.read_text())
    assert record["result"]["output"]["ci_low"] < 0  # raw value kept
    assert "95% CI $0.00 to $" in out
    assert "95% CI $-" not in out and "CI -$" not in out
    # Block times varied 100x: one plain hint.
    assert "hint    this machine is too noisy for a verdict (blocks took 1.0 s to 100.0 s)" in out
    assert "Close other apps and run again, or point Throttle at a dedicated GPU server" in out


def test_noise_bound_above_half_prints_the_noisy_hint(history, monkeypatch, capsys):
    for wall in (10.0, 20.0, 6.0):
        run_clocked(monkeypatch, capsys, wall, "--label", "same")
    code, out = run_clocked(monkeypatch, capsys, 12.0, "--label", "same")
    assert code == 0, out
    match = re.search(r"run-to-run noise bound (\d+\.\d)%", out)
    assert match and float(match.group(1)) > 50
    assert "hint    this machine is too noisy for a verdict (run-to-run noise bound" in out


def test_quiet_machine_gets_no_noisy_hint(history, monkeypatch, capsys):
    for wall in (10.0, 10.02, 10.01, 10.03):
        code, out = run_clocked(monkeypatch, capsys, wall, "--label", "same")
    assert code == 0, out
    assert "run-to-run noise bound" in out
    assert "too noisy" not in out


# --------------------------------------------------------------------------
# 4. Agent workload: own baseline, and a time-cap estimate after block 1
# --------------------------------------------------------------------------


def run_agent(monkeypatch, capsys, *extra: str) -> tuple[int, str]:
    monkeypatch.setattr(check_module, "_TRANSPORT", httpx.MockTransport(AgentFakeServer().handle))
    code = main([
        "check", "--url", URL, "--model", "mock-model", "--gpu-hourly-rate", "2.00",
        "--blocks", "3", "--warmup", "1", "--max-tokens", "16", *AGENT_ARGS, *extra,
    ])
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def test_first_agent_check_calibrates_its_own_baseline(history, monkeypatch, capsys):
    run_check(monkeypatch, capsys, FakeServer(constant(0.01)))  # a default check exists
    code, out = run_agent(monkeypatch, capsys)
    assert code == 0, out
    assert "This is the first agent check for this endpoint" in out
    assert "workload differs" not in out and "prompts_sha256" not in out
    assert "calibrate  no recent repeats of this agent config" in out
    assert "calibrating: run 3 of 3" in out
    saved = records(history)
    assert len(saved) == 4
    assert all(check_module.workload_profile(r) == "agent" for r in saved[1:])
    assert "baseline_id" not in saved[1]
    assert "CALIBRATED 3 checks of this agent config" in flat(out)
    assert CALIBRATED_LINE.search(strip_ansi(out))


def test_agent_block_one_warns_when_the_run_will_pass_the_time_cap(
    history, monkeypatch, capsys
):
    # Each _monotonic() reading is 100 s after the last: by the end of block 1
    # the run looks 100 s old, so 3 blocks cannot fit in a 50 s limit.
    ticks = iter(range(0, 10**6, 100))
    monkeypatch.setattr(check_module, "_monotonic", lambda: float(next(ticks)))
    code, out = run_agent(monkeypatch, capsys, "--no-calibrate", "--max-elapsed-seconds", "50")
    assert code == 0, out
    assert "at this pace the check needs about 100 s, over its 50 s limit" in out
    assert "Rerun with --max-elapsed-seconds 180 (or fewer --turns)" in out


def test_agent_on_pace_prints_no_time_warning(history, monkeypatch, capsys):
    code, out = run_agent(monkeypatch, capsys, "--no-calibrate")
    assert code == 0, out
    assert "at this pace" not in out
