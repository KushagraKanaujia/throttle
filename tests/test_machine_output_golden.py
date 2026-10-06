"""Machine-readable output is byte-identical to a recorded golden copy.

The human output of `throttle check` and `throttle savings` is restyled from
time to time; their machine-readable output (`check --json PATH`, the
`--share` block and `savings --json`) must never move with it. These goldens
were recorded from the code before the CLI front-door restyle, with the clock,
the run id and the timestamps pinned so every byte is reproducible.

Regenerate (only for a deliberate, changelogged format change) with:
    THROTTLE_REGEN_GOLDEN=1 PYTHONPATH=src python3 -m pytest tests/test_machine_output_golden.py
"""

from __future__ import annotations

import itertools
import os
import time as real_time
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

import throttle
import throttle.check as check_module
import throttle.config as config_module
import throttle.savings as savings_module
from throttle.cli import main

from test_check import FakeServer, REQUESTS_PER_BLOCK, URL, WARMUP

GOLDEN = Path(__file__).parent / "fixtures" / "golden"
REGEN = os.environ.get("THROTTLE_REGEN_GOLDEN") == "1"
T0 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


def _pin(monkeypatch, tmp_path: Path) -> Path:
    directory = tmp_path / "history"
    monkeypatch.setenv(check_module.HISTORY_ENV, str(directory))
    monkeypatch.setenv("THROTTLE_NO_NUDGE", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", tmp_path / "no-config.yaml")

    # Wall-clock per block comes from a fixed sequence, so $/M and its CI are fixed.
    steps = itertools.cycle([1.00, 1.10, 0.95, 1.05, 0.90, 1.02, 0.98])
    clock = {"t": 0.0, "scale": 1.0}

    def perf_counter() -> float:
        clock["t"] += next(steps) * clock["scale"]
        return clock["t"]

    fake_time = types.SimpleNamespace(**{k: getattr(real_time, k) for k in dir(real_time) if not k.startswith("_")})
    fake_time.perf_counter = perf_counter
    monkeypatch.setattr(check_module, "time", fake_time)

    ticks = itertools.count()

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):  # one minute later on every call
            return T0 + timedelta(minutes=next(ticks))

    monkeypatch.setattr(check_module, "datetime", FixedDatetime)
    monkeypatch.setattr(savings_module, "_now", lambda: T0 + timedelta(days=1))
    monkeypatch.setattr(check_module, "new_run_id", lambda: "424242")
    return directory, clock


def _check(monkeypatch, capsys, delay: float, *extra: str) -> str:
    server = FakeServer(lambda _block: delay)
    monkeypatch.setattr(check_module, "_TRANSPORT", httpx.MockTransport(server.handle))
    code = main([
        "check", "--url", URL, "--model", "mock-model", "--gpu-hourly-rate", "2.00",
        "--blocks", "5", "--requests-per-block", str(REQUESTS_PER_BLOCK),
        "--concurrency", "2", "--warmup", str(WARMUP), *extra,
    ])
    out = capsys.readouterr().out
    assert code == 0, out
    return out


def _normalise(text: str, tmp_path: Path) -> str:
    return text.replace(str(tmp_path), "<TMP>").replace(throttle.__version__, "<VERSION>")


def _compare(name: str, actual: str) -> None:
    path = GOLDEN / name
    if REGEN:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(actual.encode("utf-8"))
    assert path.read_bytes() == actual.encode("utf-8"), f"{name} changed"


def test_check_json_share_and_savings_json_are_byte_identical(monkeypatch, capsys, tmp_path):
    directory, clock = _pin(monkeypatch, tmp_path)
    for _ in range(3):  # calibrate run-to-run noise
        _check(monkeypatch, capsys, 0.0, "--label", "before", "--config", "quant=none")
    clock["scale"] = 0.5  # the candidate config runs twice as fast: CHEAPER
    json_path = tmp_path / "check.json"
    out = _check(
        monkeypatch, capsys, 0.0, "--label", "after", "--config", "quant=fp8",
        "--monthly-tokens", "1B", "--json", str(json_path), "--share",
    )
    _compare("check.json", _normalise(json_path.read_text(encoding="utf-8"), tmp_path))
    share = out[out.index(check_module.SHARE_BEGIN):]
    _compare("check_share.txt", _normalise(share, tmp_path))

    records, _ = check_module.load_history(directory)
    assert main([
        "savings", "--baseline", records[0]["id"], "--candidate", records[-1]["id"],
        "--tokens", "1B", "--period-label", "2026-10", "--json",
    ]) == 0
    captured = capsys.readouterr()
    _compare("savings.json", _normalise(captured.out + captured.err, tmp_path))


@pytest.mark.parametrize("env", [{"FORCE_COLOR": "1"}, {"NO_COLOR": "1"}])
def test_machine_output_ignores_colour_settings(monkeypatch, capsys, tmp_path, env):
    directory, clock = _pin(monkeypatch, tmp_path)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    for _ in range(3):
        _check(monkeypatch, capsys, 0.0, "--label", "before", "--config", "quant=none")
    clock["scale"] = 0.5  # the candidate config runs twice as fast: CHEAPER
    json_path = tmp_path / "check.json"
    out = _check(
        monkeypatch, capsys, 0.0, "--label", "after", "--config", "quant=fp8",
        "--monthly-tokens", "1B", "--json", str(json_path), "--share",
    )
    assert _normalise(json_path.read_text(encoding="utf-8"), tmp_path).encode() == (GOLDEN / "check.json").read_bytes()
    share = out[out.index(check_module.SHARE_BEGIN):]
    assert _normalise(share, tmp_path).encode() == (GOLDEN / "check_share.txt").read_bytes()
    assert "\x1b[" not in share
    records, _ = check_module.load_history(directory)
    main([
        "savings", "--baseline", records[0]["id"], "--candidate", records[-1]["id"],
        "--tokens", "1B", "--period-label", "2026-10", "--json",
    ])
    captured = capsys.readouterr()
    assert _normalise(captured.out + captured.err, tmp_path).encode() == (GOLDEN / "savings.json").read_bytes()
