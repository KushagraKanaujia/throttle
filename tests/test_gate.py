"""`throttle gate`: exit codes and local judgment from check JSON files and ids.
The gate is fully local: it never sends anything anywhere.

Checks are produced by the real `throttle check` against autotune's
SimulatedServer, whose speed depends on the --config values.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

import throttle.check as check_module
import throttle.config as config_module
from throttle.autotune import SimulatedServer
from throttle.cli import main

URL = "http://127.0.0.1:8000"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv(check_module.HISTORY_ENV, str(tmp_path / "history"))
    for name in ("OPENAI_API_KEY", "CI", "GITHUB_ACTIONS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", tmp_path / "no-config.yaml")


def check(capsys, tmp_path: Path, name: str, config: dict[str, str], gpus: int = 1) -> Path:
    sim = SimulatedServer("m", 2)
    sim.start(config, name)
    path = tmp_path / f"{name}.json"
    args = ["check", "--no-calibrate", "--url", URL, "--model", "m",
            "--gpu-hourly-rate", str(2 * gpus), "--gpus", str(gpus),
            "--blocks", "3", "--requests-per-block", "2", "--no-metrics", "--json", str(path)]
    for key, value in config.items():
        args += ["--config", f"{key}={value}"]
    with sim.measuring():
        assert main(args) == 0
    capsys.readouterr()
    return path


def baseline(capsys, tmp_path: Path, runs: int = 3) -> Path:
    for run in range(runs):
        path = check(capsys, tmp_path, f"base{run}", {"seqs": "64", "tp": "1"})
    return path


def gate(capsys, *argv: str) -> tuple[int, str, str]:
    code = main(["gate", *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_cheaper_passes(env, capsys, tmp_path):
    base = baseline(capsys, tmp_path)
    cand = check(capsys, tmp_path, "cand", {"seqs": "256", "tp": "1"})
    code, out, err = gate(capsys, "--baseline", str(base), "--candidate", str(cand),
                          "--json", str(tmp_path / "g.json"))
    assert code == 0, out + err
    assert "Verdict: CHEAPER" in out and "PASS: CHEAPER." in out
    result = json.loads((tmp_path / "g.json").read_text())
    assert result["pass"] is True and result["judged_by"] == "local"
    assert "cloud" not in result


def test_more_expensive_fails(env, capsys, tmp_path):
    base = baseline(capsys, tmp_path)
    cand = check(capsys, tmp_path, "cand", {"seqs": "64", "tp": "2"}, gpus=2)
    code, out, err = gate(capsys, "--baseline", str(base), "--candidate", str(cand))
    assert code == 1
    assert "Verdict: MORE EXPENSIVE" in out
    assert "FAIL: MORE EXPENSIVE" in err


def test_max_increase_threshold(env, capsys, tmp_path):
    base = baseline(capsys, tmp_path)
    cand = check(capsys, tmp_path, "cand", {"seqs": "64", "tp": "2"}, gpus=2)
    code, out, err = gate(capsys, "--baseline", str(base), "--candidate", str(cand),
                          "--max-increase", "90")
    assert code == 0  # +28% is below the 90% threshold


def test_output_changed_fails_even_though_cheaper_per_token(env, capsys, tmp_path):
    base = baseline(capsys, tmp_path)
    cand = check(capsys, tmp_path, "cand", {"seqs": "64", "tp": "1", "quant": "fp8"})
    code, out, err = gate(capsys, "--baseline", str(base), "--candidate", str(cand))
    assert code == 1
    assert "OUTPUT CHANGED" in out
    assert "FAIL: OUTPUT CHANGED" in err


def test_not_calibrated_passes_with_a_warning(env, capsys, tmp_path, monkeypatch):
    base = baseline(capsys, tmp_path, runs=1)
    cand = check(capsys, tmp_path, "cand", {"seqs": "64", "tp": "2"}, gpus=2)
    code, out, err = gate(capsys, "--baseline", str(base), "--candidate", str(cand))
    assert code == 0
    assert "NOT CALIBRATED" in out
    assert "PASS (unverified)" in err


def test_ids_from_local_history_work_too(env, capsys, tmp_path):
    base = json.loads(baseline(capsys, tmp_path).read_text())["id"]
    cand = json.loads(check(capsys, tmp_path, "cand", {"seqs": "256", "tp": "1"}).read_text())["id"]
    code, out, err = gate(capsys, "--baseline", base, "--candidate", cand)
    assert code == 0, err
    assert "CHEAPER" in out


def test_usage_errors_exit_2(env, capsys, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert gate(capsys, "--baseline", str(bad), "--candidate", str(bad))[0] == 2
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"hello": "world"}))
    code, _, err = gate(capsys, "--baseline", str(other), "--candidate", str(other))
    assert code == 2 and "not a check record" in err
    base = baseline(capsys, tmp_path, runs=1)
    code, _, err = gate(capsys, "--baseline", str(base), "--candidate", "nope")
    assert code == 2 and "--candidate nope" in err
    code, _, err = gate(capsys, "--baseline", "chk_unknown", "--candidate", str(base))
    assert code == 2 and "--baseline chk_unknown: no such file" in err


def test_gate_sends_nothing_and_still_accepts_no_cloud(env, capsys, tmp_path, monkeypatch):
    base = baseline(capsys, tmp_path)
    cand = check(capsys, tmp_path, "cand", {"seqs": "256", "tp": "1"})

    def refuse(*args, **kwargs):
        raise AssertionError("the gate tried to make an HTTP request")

    monkeypatch.setattr(httpx.Client, "send", refuse)
    monkeypatch.setattr(httpx.AsyncClient, "send", refuse)
    code, out, err = gate(capsys, "--baseline", str(base), "--candidate", str(cand))
    assert code == 0, err
    assert "PASS: CHEAPER." in out
    # --no-cloud is kept as a silent no-op so existing CI workflows keep working.
    assert gate(capsys, "--baseline", str(base), "--candidate", str(cand), "--no-cloud")[0] == 0
