"""scripts/action_summary.py, the steps of the 'Throttle cost check' Action.

Runs the subcommands the way action.yml does (INPUT_* environment, GITHUB_OUTPUT
file) and asserts on the step outputs, workflow commands and exit codes.
"""

from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "action_summary.py"


@pytest.fixture(scope="module")
def action():
    spec = importlib.util.spec_from_file_location("action_summary", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve(action, monkeypatch, tmp_path, history_key: str) -> dict[str, str]:
    output = tmp_path / f"output-{len(list(tmp_path.iterdir()))}.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path / "runner"))
    monkeypatch.setenv("INPUT_URL", "https://vllm.internal:8000")
    monkeypatch.setenv("INPUT_GPU_HOURLY_RATE", "2.5")
    monkeypatch.setenv("INPUT_MODEL", "m")
    monkeypatch.setenv("INPUT_HISTORY_KEY", history_key)
    assert action.main(["resolve"]) == 0
    return dict(line.split("=", 1) for line in output.read_text().splitlines())


def test_history_keys_never_restore_each_others_cache(action, monkeypatch, tmp_path, capsys):
    # actions/cache restores the newest cache whose key STARTS WITH the
    # restore-keys prefix, so one key's prefix must not match another's keys.
    keys = ["prod", "prod-eu", "prod-2", "prod-", "prod.eu", "p", "pr", "1", "10",
            "a" * 9, "a" * 10, ""]  # "" = the default hash of url + model
    resolved = {key: resolve(action, monkeypatch, tmp_path, key) for key in keys}
    capsys.readouterr()
    for key, out in resolved.items():
        assert out["cache-key"].startswith(out["cache-prefix"]), key
    for first, second in itertools.permutations(keys, 2):
        prefix = resolved[first]["cache-prefix"]
        other_key = resolved[second]["cache-key"]
        assert not other_key.startswith(prefix), (first, second, prefix, other_key)
    assert resolved["prod"]["cache-prefix"] == "throttle-check-v1-4-prod-"


def gate(action, monkeypatch, capsys, code: int, verdict: str, policy: str) -> tuple[int, str]:
    monkeypatch.setenv("THROTTLE_EXIT_CODE", str(code))
    monkeypatch.setenv("VERDICT", verdict)
    monkeypatch.setenv("INPUT_NOT_CALIBRATED", policy)
    monkeypatch.setenv("INPUT_FAIL_IF_COSTLIER", "5")
    result = action.main(["gate"])
    return result, capsys.readouterr().out


def test_gate_reports_an_unjudged_workload_mismatch(action, monkeypatch, capsys):
    # throttle check exits 5 with NO WINNER when the baseline ran another
    # workload (e.g. cold vs a 0.4.0 warm baseline): the gate judged nothing.
    code, out = gate(action, monkeypatch, capsys, 5, "NO WINNER", "warn")
    assert code == 0
    assert out.startswith("::warning::Throttle cost check: the baseline ran a different workload")
    assert "not-calibrated: warn, so the job passes" in out
    code, out = gate(action, monkeypatch, capsys, 5, "NO WINNER", "fail")
    assert code == 5
    assert out.startswith("::error::Throttle cost check: the baseline ran a different workload")
    # NOT CALIBRATED keeps its own message.
    code, out = gate(action, monkeypatch, capsys, 5, "NOT CALIBRATED", "warn")
    assert code == 0 and "NOT CALIBRATED, so fail-if-costlier could not judge" in out


def test_summary_lists_workload_differences(action):
    record = {
        "id": "c2",
        "metric": "output",
        "result": {"output": {"mean": 1.0, "ci_low": 0.9, "ci_high": 1.1}},
        "fingerprint": {"model": "m", "label": "x", "gpu_hourly_rate_usd": 2.0},
        "comparison": {
            "verdict": "NO WINNER",
            "reason": "the workload differs (prompt_cache_mode)",
            "previous": {"mean": 1.0},
            "changes": [],
            "workload_differs": ["prompt_cache_mode"],
            "previous_workload": {"prompt_cache_mode": "warm"},
            "current_workload": {"prompt_cache_mode": "cold"},
        },
    }
    markdown, outputs = action.render_summary(record, 5, action.Scrubber(""), "5")
    assert outputs["verdict"] == "NO WINNER"
    assert "| workload.prompt_cache_mode | warm | cold |" in markdown
    assert "What changed: nothing Throttle can see" not in markdown
