"""Caller-visible behaviour of `throttle check` (repeat-use cost drift check).

Every test drives the real CLI entry point (`throttle.cli.main`) against a
mock OpenAI-compatible backend built on tests/mock_backend.py, and asserts on
what a user or CI job sees: printed output, exit code, and the history file.
"""

from __future__ import annotations

import asyncio
import json
import math
import statistics
from pathlib import Path
from typing import Callable

import httpx
import pytest

import throttle.check as check_module
import throttle.config as config_module
from mock_backend import mock_chat_completion
from throttle.cli import main

URL = "http://127.0.0.1:8000"
REQUESTS_PER_BLOCK = 4
WARMUP = 1


class FakeServer:
    """OpenAI-compatible server whose speed we control per measured block.

    Request latency is ``delay_for_block(block_index)`` seconds; warm-up
    requests are not delayed. Responses come from mock_backend so the usage
    block is realistic (word-count token numbers).
    """

    def __init__(
        self,
        delay_for_block: Callable[[int], float],
        *,
        metrics_body: str | None = None,
        include_usage: bool = True,
        max_model_len: int = 4096,
        extra_models: tuple[str, ...] = (),
        fixed_usage: tuple[int, int] | None = None,
    ) -> None:
        self.extra_models = extra_models
        self.fixed_usage = fixed_usage
        self.delay_for_block = delay_for_block
        self.metrics_body = metrics_body
        self.include_usage = include_usage
        self.max_model_len = max_model_len
        self.chat_requests = 0

    async def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {
                            "id": "mock-model",
                            "object": "model",
                            "owned_by": "vllm",
                            "max_model_len": self.max_model_len,
                        },
                        *({"id": extra, "object": "model"} for extra in self.extra_models),
                    ],
                },
            )
        if path == "/metrics":
            if self.metrics_body is None:
                return httpx.Response(404, text="not found")
            return httpx.Response(
                200, text=self.metrics_body, headers={"content-type": "text/plain"}
            )
        if path == "/v1/chat/completions":
            index = self.chat_requests
            self.chat_requests += 1
            body = json.loads(request.content)
            prompt = " ".join(m["content"] for m in body["messages"])
            if index >= WARMUP:
                block = (index - WARMUP) // REQUESTS_PER_BLOCK
                await asyncio.sleep(self.delay_for_block(block))
            payload = mock_chat_completion(prompt, simulate_latency=False)
            if self.fixed_usage is not None:
                prompt_tokens, completion_tokens = self.fixed_usage
                payload["usage"] = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
            if not self.include_usage:
                payload.pop("usage")
            return httpx.Response(200, json=payload)
        return httpx.Response(404)


@pytest.fixture
def history(tmp_path, monkeypatch) -> Path:
    directory = tmp_path / "history"
    monkeypatch.setenv(check_module.HISTORY_ENV, str(directory))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # Never let a developer's ~/.throttle/config.yaml leak into these tests.
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", tmp_path / "no-config.yaml")
    return directory


def run_check(
    monkeypatch,
    capsys,
    server: FakeServer,
    *extra: str,
    rate: str = "2.00",
    url: str = URL,
    model: str | None = "mock-model",
) -> tuple[int, str]:
    monkeypatch.setattr(check_module, "_TRANSPORT", httpx.MockTransport(server.handle))
    code = main(
        [
            "check",
            "--url",
            url,
            *(["--model", model] if model else []),
            "--gpu-hourly-rate",
            rate,
            "--blocks",
            "5",
            "--requests-per-block",
            str(REQUESTS_PER_BLOCK),
            "--concurrency",
            "2",
            "--warmup",
            str(WARMUP),
            *extra,
        ]
    )
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def constant(seconds: float) -> Callable[[int], float]:
    return lambda _block: seconds


def test_first_check_labels_inputs_and_is_saved(history, monkeypatch, capsys):
    code, out = run_check(
        monkeypatch, capsys, FakeServer(constant(0.01)),
        "--label", "baseline", "--config", "max_num_seqs=256", "--monthly-tokens", "500M",
    )
    assert code == 0, out
    assert "[ASSUMED: you supplied --gpu-hourly-rate" in out
    assert "per million output tokens   [MEASURED]" in out
    assert "95% CI $" in out
    assert "monthly at 500M output tokens: $" in out
    assert "[PROJECTED: MEASURED $/M x ASSUMED volume]" in out
    assert "This is the first check for this endpoint" in out
    assert "config.max_num_seqs = 256" in out
    lines = (history / "checks.ndjson").read_text().splitlines()
    assert len(lines) == 1
    assert "Saved as check " + json.loads(lines[0])["id"] in out


def test_costlier_config_shows_change_dollar_delta_and_fails_ci(history, monkeypatch, capsys):
    for _ in range(3):  # three repeats calibrate run-to-run noise (2 df)
        code, out = run_check(
            monkeypatch, capsys, FakeServer(constant(0.01)),
            "--label", "fast", "--config", "max_num_seqs=256",
        )
        assert code == 0, out
    code, out = run_check(
        monkeypatch, capsys, FakeServer(constant(0.08)),
        "--label", "slow", "--config", "max_num_seqs=64",
        "--monthly-tokens", "1B", "--fail-if-costlier", "10",
    )
    assert code == 4, out
    assert "config.max_num_seqs: 256 -> 64" in out
    assert "label: fast -> slow" in out
    assert "Verdict: MORE EXPENSIVE" in out
    assert "change  +$" in out
    assert "per month at 1B output tokens" in out
    assert "monthly +$" in out
    assert "FAIL: significantly more expensive" in out


def test_cheaper_config_passes_ci_gate(history, monkeypatch, capsys):
    for _ in range(3):  # three repeats calibrate run-to-run noise (2 df)
        run_check(monkeypatch, capsys, FakeServer(constant(0.08)), "--config", "quant=none")
    code, out = run_check(
        monkeypatch, capsys, FakeServer(constant(0.01)),
        "--config", "quant=fp8", "--fail-if-costlier", "0",
    )
    assert code == 0, out
    assert "config.quant: none -> fp8" in out
    assert "Verdict: CHEAPER" in out
    assert "change  -$" in out
    assert "FAIL" not in out


def test_overlapping_intervals_are_no_winner_and_never_fail_ci(history, monkeypatch, capsys):
    # Same noisy speed profile both times: wide, overlapping CIs.
    noisy = [0.01, 0.06, 0.02, 0.05, 0.03]
    for _ in range(3):  # three repeats calibrate run-to-run noise (2 df)
        run_check(monkeypatch, capsys, FakeServer(lambda b: noisy[b]), "--config", "batch=8")
    code, out = run_check(
        monkeypatch, capsys, FakeServer(lambda b: noisy[b]),
        "--config", "batch=16", "--monthly-tokens", "500M", "--fail-if-costlier", "0",
    )
    assert code == 0, out
    assert "Verdict: NO WINNER" in out
    assert "the 95% confidence intervals overlap, so the difference is within measurement noise" in out
    assert "monthly not projected" in out
    assert "Verdict: CHEAPER" not in out and "Verdict: MORE EXPENSIVE" not in out


def test_different_workload_is_never_a_winner(history, monkeypatch, capsys):
    run_check(monkeypatch, capsys, FakeServer(constant(0.01)))
    code, out = run_check(
        monkeypatch, capsys, FakeServer(constant(0.08)),
        "--max-tokens", "32", "--fail-if-costlier", "0",
    )
    assert code == 0, out
    assert "Verdict: NO WINNER, the workload differs (max_tokens)" in out


def test_server_reported_config_is_fingerprinted_and_diffed(history, monkeypatch, capsys):
    def metrics(block_size: int) -> str:
        return (
            "# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig\n"
            "# TYPE vllm:cache_config_info gauge\n"
            f'vllm:cache_config_info{{block_size="{block_size}",'
            'gpu_memory_utilization="0.9"} 1.0\n'
            'vllm:num_requests_running{model_name="mock-model"} 0.0\n'
        )

    code, out = run_check(
        monkeypatch, capsys, FakeServer(constant(0.01), metrics_body=metrics(16))
    )
    assert code == 0, out
    assert "/metrics    read" in out
    assert "vllm.cache_config.block_size = 16" in out
    assert "server.model.max_model_len = 4096" in out
    code, out = run_check(
        monkeypatch, capsys,
        FakeServer(constant(0.01), metrics_body=metrics(32), max_model_len=8192),
    )
    assert code == 0, out
    assert "vllm.cache_config.block_size: 16 -> 32" in out
    assert "server.model.max_model_len: 4096 -> 8192" in out
    assert "gpu_memory_utilization:" not in out  # unchanged keys are not listed


def test_gpu_rate_change_is_listed_and_separated_from_throughput(history, monkeypatch, capsys):
    run_check(monkeypatch, capsys, FakeServer(constant(0.02)), rate="2.00")
    code, out = run_check(monkeypatch, capsys, FakeServer(constant(0.02)), rate="4.00")
    assert code == 0, out
    assert "gpu_hourly_rate_usd (ASSUMED): 2.0000 -> 4.0000" in out
    assert (
        "the GPU rate changed ($2.00/hr -> $4.00/hr) [ASSUMED]: that alone moves $/M "
        "by +100.0%, arithmetic on rates you typed, not a measurement." in out
    )
    assert "At the old rate this check measures $" in out
    assert "[MEASURED throughput]; the verdict judges only that part." in out


def test_history_lists_checks_and_filters_by_endpoint(history, monkeypatch, capsys):
    run_check(monkeypatch, capsys, FakeServer(constant(0.01)), "--label", "first-cfg")
    run_check(monkeypatch, capsys, FakeServer(constant(0.01)), "--label", "second-cfg")
    code = main(["check", "--history"])
    out = capsys.readouterr().out
    assert code == 0
    assert "showing 2 of 2" in out
    assert out.index("first-cfg") < out.index("second-cfg")
    assert "[MEASURED]" in out and "[ASSUMED]" in out
    code = main(["check", "--history", "--url", "http://127.0.0.1:9999"])
    out = capsys.readouterr().out
    assert code == 0
    assert "No checks recorded yet" in out


def test_against_compares_with_a_chosen_check(history, monkeypatch, capsys):
    run_check(monkeypatch, capsys, FakeServer(constant(0.08)), "--label", "old-slow")
    first_id = json.loads((history / "checks.ndjson").read_text().splitlines()[0])["id"]
    run_check(monkeypatch, capsys, FakeServer(constant(0.01)), "--label", "mid-fast")
    code, out = run_check(
        monkeypatch, capsys, FakeServer(constant(0.01)),
        "--label", "new-fast", "--against", first_id, "--no-save",
    )
    assert code == 0, out
    assert f"Compared with check {first_id}" in out
    assert "label: old-slow -> new-fast" in out
    assert "Not saved (--no-save)." in out
    assert len((history / "checks.ndjson").read_text().splitlines()) == 2


def test_missing_usage_fails_without_guessing_or_recording(history, monkeypatch, capsys):
    code, out = run_check(
        monkeypatch, capsys, FakeServer(constant(0.0), include_usage=False)
    )
    assert code == 1
    assert "does not guess token counts" in out
    assert "Nothing was recorded." in out
    assert not (history / "checks.ndjson").exists()


def test_missing_required_inputs_is_a_usage_error(history, capsys):
    code = main(["check", "--url", URL, "--model", "mock-model"])
    err = capsys.readouterr().err
    assert code == 2
    assert "--gpu-hourly-rate" in err


def test_secret_looking_config_keys_are_refused(history, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["check", "--config", "api_key=sk-123"])
    assert excinfo.value.code == 2
    assert "looks like a secret" in capsys.readouterr().err


# --------------------------------------------------------------------------
# The dollar math itself, against hand-computed figures
# --------------------------------------------------------------------------


class FakeClock:
    """Stands in for time.perf_counter inside throttle.check.

    _run_block reads the clock once before and once after each block, so
    each pair of readings below is one block's exact wall-clock time.
    """

    def __init__(self, walls: list[float]) -> None:
        self.readings: list[float] = []
        for index, wall in enumerate(walls):
            start = 1000.0 * index
            self.readings += [start, start + wall]

    def perf_counter(self) -> float:
        return self.readings.pop(0)


def test_dollars_per_million_match_hand_computed_values(history, monkeypatch, capsys, tmp_path):
    # 3 blocks x 4 requests, every request reports 100 prompt / 50 completion
    # tokens, so each block is 400 input / 200 output tokens. Block wall
    # times are exactly 10 s, 12 s and 14 s. At $2.00/hr:
    #   output $/M = 2.00 * wall / 3600 / 200 * 1e6 -> 27.78, 33.33, 38.89
    #   input  $/M = 2.00 * wall / 3600 / 400 * 1e6 -> 13.89, 16.67, 19.44
    #   total  $/M = 2.00 * wall / 3600 / 600 * 1e6 ->  9.26, 11.11, 12.96
    walls = [10.0, 12.0, 14.0]
    rate = 2.00
    expected = {
        "output": [rate * w / 3600 / 200 * 1e6 for w in walls],
        "input": [rate * w / 3600 / 400 * 1e6 for w in walls],
        "total": [rate * w / 3600 / 600 * 1e6 for w in walls],
    }
    assert round(sum(expected["output"]) / 3, 2) == 33.33  # the hand figure above
    clock = FakeClock(walls)
    monkeypatch.setattr(check_module, "time", clock)
    monkeypatch.setattr(check_module, "_TRANSPORT", httpx.MockTransport(
        FakeServer(constant(0.0), fixed_usage=(100, 50)).handle
    ))
    json_path = tmp_path / "check.json"
    code = main([
        "check", "--url", URL, "--model", "mock-model", "--gpu-hourly-rate", "2.00",
        "--blocks", "3", "--requests-per-block", "4", "--concurrency", "2",
        "--warmup", "0", "--monthly-tokens", "500M", "--json", str(json_path),
    ])
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert code == 0, out
    assert clock.readings == []  # exactly one start/stop pair per block

    record = json.loads(json_path.read_text())
    t_crit_2df = 4.303  # Student t 97.5th percentile, 2 df (standard 3-decimal table value)
    for metric, values in expected.items():
        mean = sum(values) / 3
        half_width = t_crit_2df * statistics.stdev(values) / math.sqrt(3)
        summary = record["result"][metric]
        assert summary["mean"] == pytest.approx(mean, rel=1e-9), metric
        assert summary["ci_low"] == pytest.approx(mean - half_width, rel=1e-9), metric
        assert summary["ci_high"] == pytest.approx(mean + half_width, rel=1e-9), metric
    for block, wall in zip(record["blocks"], walls):
        assert block["input_tokens"] == 400 and block["output_tokens"] == 200
        assert block["gpu_dollars"] == pytest.approx(rate * wall / 3600, rel=1e-9)

    assert "$33.33 per million output tokens   [MEASURED]" in out
    assert "$16.67/M input tokens; $11.11/M tokens (input + output)" in out
    output_mean = sum(expected["output"]) / 3
    assert f"monthly at 500M output tokens: ${output_mean * 500:,.0f}" in out  # $16,667
    assert "monthly at 500M output tokens: $16,667" in out


# --------------------------------------------------------------------------
# Robustness: history, fingerprints, URLs, limits
# --------------------------------------------------------------------------


def test_malformed_history_lines_are_skipped_not_crashed_on(history, monkeypatch, capsys):
    history.mkdir(parents=True)
    endpoint = "http://127.0.0.1:8000/v1/chat/completions"
    bad = [
        {"record_type": "throttle_check", "result": {"output": {"mean": 1}}, "endpoint": endpoint},
        {"record_type": "throttle_check", "result": {"output": "x"}, "fingerprint": {},
         "endpoint": endpoint},
        {"record_type": "throttle_check", "result": {"output": {"mean": "cheap"}},
         "fingerprint": {"gpu_hourly_rate_usd": 2.0}, "endpoint": endpoint},
    ]
    with (history / "checks.ndjson").open("wb") as handle:
        for record in bad:
            handle.write(json.dumps(record).encode() + b"\n")
        handle.write(b"\xff\xfe not utf-8\n")

    code = main(["check", "--history"])
    out = capsys.readouterr().out
    assert code == 0
    assert "skipped 4 unreadable line(s)" in out

    code, out = run_check(monkeypatch, capsys, FakeServer(constant(0.0)))
    assert code == 0, out
    assert "This is the first check for this endpoint" in out
    assert "Traceback" not in out


def test_metrics_read_failure_is_not_reported_as_a_config_change(history, monkeypatch, capsys):
    body = 'vllm:cache_config_info{block_size="16",gpu_memory_utilization="0.9"} 1.0\n'
    run_check(monkeypatch, capsys, FakeServer(constant(0.0), metrics_body=body))
    code, out = run_check(monkeypatch, capsys, FakeServer(constant(0.0), metrics_body=None))
    assert code == 0, out
    assert "(not set)" not in out
    assert "block_size" not in out.split("Compared with")[1]
    assert "/metrics was not read this time (not exposed (HTTP 404))" in out


def test_unrelated_model_on_the_server_is_not_a_config_change(history, monkeypatch, capsys):
    run_check(monkeypatch, capsys, FakeServer(constant(0.0)))
    code, out = run_check(
        monkeypatch, capsys, FakeServer(constant(0.0), extra_models=("other-model",))
    )
    assert code == 0, out
    assert "What changed in the config: nothing Throttle can see" in out
    assert "other-model" not in out


def test_comparing_against_another_endpoint_says_so(history, monkeypatch, capsys):
    run_check(monkeypatch, capsys, FakeServer(constant(0.0)), url="http://localhost:8000")
    first_id = json.loads((history / "checks.ndjson").read_text().splitlines()[0])["id"]
    code, out = run_check(
        monkeypatch, capsys, FakeServer(constant(0.0)), "--against", first_id,
    )
    assert code == 0, out
    assert "WARNING: that check measured a different endpoint" in out
    assert (
        "endpoint: http://localhost:8000/v1/chat/completions -> "
        "http://127.0.0.1:8000/v1/chat/completions"
    ) in out
    assert "nothing Throttle can see" not in out


def test_workload_mismatch_prints_no_dollar_change(history, monkeypatch, capsys):
    run_check(monkeypatch, capsys, FakeServer(constant(0.01)))
    code, out = run_check(monkeypatch, capsys, FakeServer(constant(0.01)), "--max-tokens", "32")
    assert code == 0, out
    assert "change  not computed" in out
    assert "change  +$" not in out and "change  -$" not in out


def test_malformed_metrics_url_is_a_usage_error_before_traffic(history, monkeypatch, capsys):
    server = FakeServer(constant(0.0))
    code, out = run_check(monkeypatch, capsys, server, "--metrics-url", "http://[::1")
    assert code == 2
    assert "--metrics-url is not a valid URL" in out
    assert "Traceback" not in out
    assert server.chat_requests == 0


@pytest.mark.parametrize(
    "extra, message",
    [
        (("--blocks", "100000", "--requests-per-block", "10000"), "above --max-requests"),
        (("--max-tokens", "1000000"), "above --max-tokens-per-request"),
        (("--concurrency", "5000"), "above --max-concurrency"),
        (("--requests-per-block", "500", "--max-tokens", "1024"),
         "above --max-total-requested-tokens"),
    ],
)
def test_oversized_checks_are_refused_before_traffic(history, monkeypatch, capsys, extra, message):
    server = FakeServer(constant(0.0))
    code, out = run_check(monkeypatch, capsys, server, *extra)
    assert code == 2, out
    assert message in out
    assert server.chat_requests == 0


def test_spend_ceiling_is_enforced_and_overridable(history, monkeypatch, capsys):
    server = FakeServer(constant(0.0))
    # $98/hr x 300 s default time ceiling = $8.17 > $3.00 default spend ceiling.
    code, out = run_check(monkeypatch, capsys, server, rate="98")
    assert code == 2, out
    assert "worst-case GPU time for this check is $8.17" in out
    assert server.chat_requests == 0
    code, out = run_check(
        monkeypatch, capsys, server, "--max-estimated-spend", "10", rate="98"
    )
    assert code == 0, out
    assert "worst-case GPU time $8.17" in out


def test_model_is_detected_when_the_server_lists_one(history, monkeypatch, capsys):
    code, out = run_check(monkeypatch, capsys, FakeServer(constant(0.0)), model=None)
    assert code == 0, out
    assert "Model: mock-model (the only model this server lists" in out


def test_missing_model_lists_what_the_server_offers(history, monkeypatch, capsys):
    server = FakeServer(constant(0.0), extra_models=("other-model",))
    code, out = run_check(monkeypatch, capsys, server, model=None)
    assert code == 2
    assert "--model is required; this server lists: mock-model, other-model" in out
    assert server.chat_requests == 0


# --------------------------------------------------------------------------
# Run-to-run noise: a within-run CI cannot see drift between runs
# --------------------------------------------------------------------------


def test_nothing_changed_but_machine_got_faster_is_not_a_verdict(history, monkeypatch, capsys):
    # The confirmed bug: identical checks, but the machine was busier during
    # the earlier ones. Each run's CI is tight and they do not overlap, which
    # used to print "Verdict: CHEAPER (-87%)" and a monthly saving.
    for delay in (0.08, 0.04, 0.08):
        code, out = run_check(
            monkeypatch, capsys, FakeServer(constant(delay)), "--label", "prod",
        )
        assert code == 0, out
    code, out = run_check(
        monkeypatch, capsys, FakeServer(constant(0.01)), "--label", "prod",
        "--monthly-tokens", "1B", "--fail-if-costlier", "0",
    )
    assert code == 0, out
    assert "Verdict: CHEAPER" not in out and "Verdict: MORE EXPENSIVE" not in out
    assert "Verdict: NO WINNER" in out
    assert "is not larger than the run-to-run noise bound" in out
    assert "per month at 1B" not in out
    # The runs of one config disagree beyond their CIs: say so.
    assert "your machine's run-to-run noise is larger than within-run noise" in out


def test_relabelled_rerun_with_drift_is_not_calibrated(history, monkeypatch, capsys):
    # Same drift, but the second check is labelled differently, so no config
    # has a repeat yet: the change is shown, the verdict is withheld.
    run_check(monkeypatch, capsys, FakeServer(constant(0.08)), "--label", "before")
    code, out = run_check(
        monkeypatch, capsys, FakeServer(constant(0.01)), "--label", "rerun",
        "--monthly-tokens", "1B",
    )
    assert code == 0, out
    assert "Verdict: CHEAPER" not in out and "Verdict: MORE EXPENSIVE" not in out
    assert (
        "Verdict: NOT CALIBRATED \u2014 run-to-run noise unknown. Two checks at "
        "different times can differ from load alone. Run 'throttle check' at least "
        "3 times without changing anything (within 24 h) to measure your noise, or "
        "use 'throttle golden' for a counterbalanced decision."
    ) in out
    assert "change  -$" in out  # the measured change is still shown
    assert "before  $" in out and "now     $" in out and out.count("95% CI $") >= 3
    assert "per month at 1B" not in out  # no projected saving headline
    assert "monthly not projected: run-to-run noise is not calibrated" in out


def test_calibrated_small_noise_and_large_change_gives_a_verdict(history, monkeypatch, capsys, tmp_path):
    for _ in range(3):
        code, out = run_check(
            monkeypatch, capsys, FakeServer(constant(0.08)), "--config", "quant=none",
        )
        assert code == 0, out
    json_path = tmp_path / "cmp.json"
    code, out = run_check(
        monkeypatch, capsys, FakeServer(constant(0.01)), "--config", "quant=fp8",
        "--json", str(json_path),
    )
    assert code == 0, out
    assert "Verdict: CHEAPER, the -" in out
    assert "change is larger than the run-to-run noise bound (" in out
    assert "and the 95% confidence intervals do not overlap." in out
    assert "MEASURED from 3 earlier checks of unchanged configs within 24 h" in out
    noise = json.loads(json_path.read_text())["comparison"]["noise"]
    assert noise["calibrated"] is True and noise["n_checks"] == 3
    assert noise["degrees_of_freedom"] == 2


def test_calibrated_large_noise_floor_is_no_winner(history, monkeypatch, capsys, tmp_path):
    # Three runs of config A differ up to ~4x (a loaded machine); config B is
    # ~4x slower than A's latest run with tight, non-overlapping CIs. The
    # change is inside the run-to-run bound, so there is no winner.
    for delay in (0.01, 0.04, 0.01):
        run_check(monkeypatch, capsys, FakeServer(constant(delay)), "--config", "batch=8")
    json_path = tmp_path / "cmp.json"
    code, out = run_check(
        monkeypatch, capsys, FakeServer(constant(0.04)), "--config", "batch=16",
        "--fail-if-costlier", "0", "--json", str(json_path),
    )
    assert code == 0, out
    assert "Verdict: MORE EXPENSIVE" not in out
    assert "Verdict: NO WINNER, the change (+" in out
    assert "is not larger than the run-to-run noise bound (" in out
    assert "overlap" not in out.split("Verdict:")[1]  # the CI condition passed
    # The documented rule: t(0.975, df) x sqrt(2) x pooled relative SD of
    # the group's $/M means (one group of 3 -> 2 df, t = 4.303).
    lines = (history / "checks.ndjson").read_text().splitlines()
    means = [json.loads(line)["result"]["output"]["mean"] for line in lines[:3]]
    centre = statistics.fmean(means)
    sd = math.sqrt(sum(((m - centre) / centre) ** 2 for m in means) / 2) * 100
    expected = 4.303 * math.sqrt(2) * sd
    noise = json.loads(json_path.read_text())["comparison"]["noise"]
    assert noise["floor_percent"] == pytest.approx(expected)
    assert f"noise bound {expected:.1f}%" in out
    assert f"t x sqrt(2) x {sd:.1f}% SD, 2 df" in out


def test_not_calibrated_has_its_own_exit_code_only_under_the_ci_gate(history, monkeypatch, capsys):
    run_check(monkeypatch, capsys, FakeServer(constant(0.01)), "--label", "a")
    # Measured much costlier, but uncalibrated: never exit 4.
    code, out = run_check(
        monkeypatch, capsys, FakeServer(constant(0.08)), "--label", "b",
        "--fail-if-costlier", "0", "--no-save",
    )
    assert code == 5, out
    assert "FAIL" not in out
    assert "WARNING: NOT CALIBRATED, so --fail-if-costlier could not judge this change. Exit code 5" in out
    # Without the gate it is informational: exit 0.
    code, out = run_check(
        monkeypatch, capsys, FakeServer(constant(0.08)), "--label", "b", "--no-save",
    )
    assert code == 0, out
    assert "Verdict: NOT CALIBRATED" in out


def test_calibrated_costlier_below_threshold_passes_gate(history, monkeypatch, capsys):
    for _ in range(3):
        run_check(monkeypatch, capsys, FakeServer(constant(0.01)), "--label", "a")
    code, out = run_check(
        monkeypatch, capsys, FakeServer(constant(0.08)), "--label", "b",
        "--fail-if-costlier", "100000",
    )
    assert code == 0, out
    assert "Verdict: MORE EXPENSIVE" in out
    assert "FAIL" not in out


# --------------------------------------------------------------------------
# Calibration honesty: repeats, duplicates, stale evidence, assumed rates
# --------------------------------------------------------------------------


def run_clocked(
    monkeypatch, capsys, wall: float, *extra: str, rate: str = "2.00"
) -> tuple[int, str]:
    """One check whose 3 block wall times are exactly wall, wall*1.001, wall*1.002.

    Every request reports 100 prompt / 50 completion tokens, so the $/M of
    each check is fully determined by ``wall`` and ``rate``: deterministic
    means with tight, known CIs.
    """

    monkeypatch.setattr(check_module, "time", FakeClock([wall, wall * 1.001, wall * 1.002]))
    monkeypatch.setattr(check_module, "_TRANSPORT", httpx.MockTransport(
        FakeServer(constant(0.0), fixed_usage=(100, 50)).handle
    ))
    code = main([
        "check", "--url", URL, "--model", "mock-model", "--gpu-hourly-rate", rate,
        "--blocks", "3", "--requests-per-block", "4", "--concurrency", "2",
        "--warmup", "0", *extra,
    ])
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def rewrite_history(history: Path, edit: Callable[[int, dict], None]) -> None:
    path = history / "checks.ndjson"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for index, record in enumerate(records):
        edit(index, record)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_three_unchanged_checks_are_not_calibrated_until_two_earlier_repeats(
    history, monkeypatch, capsys
):
    # Runs 2 and 3 have only 1 and 2 earlier checks of this config: a range
    # of one pair is not a noise estimate, so no verdict either way.
    outputs = []
    for wall in (10.0, 10.3, 9.8):
        code, out = run_clocked(monkeypatch, capsys, wall, "--label", "same")
        assert code == 0, out
        outputs.append(out)
    assert "This is the first check for this endpoint" in outputs[0]
    for out in outputs[1:]:
        assert "Verdict: NOT CALIBRATED" in out
        assert "CHEAPER" not in out and "MORE EXPENSIVE" not in out
    assert "1 degree(s) of freedom from repeat checks within 24 h, 2 needed" in outputs[2]
    code, out = run_clocked(monkeypatch, capsys, 10.1, "--label", "same")
    assert code == 0, out
    assert "Verdict: NO WINNER" in out
    assert "MEASURED from 3 earlier checks" in out


def test_one_lucky_pair_does_not_certify_a_winner(history, monkeypatch, capsys):
    # Two quiet repeats 0.6% apart used to become a 0.6% "floor", so a new
    # config 5% slower with tight CIs was MORE EXPENSIVE and failed CI (exit 4).
    for wall in (10.0, 10.06):
        run_clocked(monkeypatch, capsys, wall, "--label", "before")
    code, out = run_clocked(
        monkeypatch, capsys, 10.563, "--label", "after", "--fail-if-costlier", "1",
    )
    assert code == 5, out
    assert "Verdict: NOT CALIBRATED" in out
    assert "FAIL" not in out and "MORE EXPENSIVE" not in out
    # A third repeat shows more drift (range 3%). The old max-min rule would
    # call +5% a winner (5% > 3%); the t prediction bound from 3 checks
    # (t 4.303 x sqrt(2) x 1.6% SD = 9.5%) does not.
    run_clocked(monkeypatch, capsys, 10.3, "--label", "before")
    code, out = run_clocked(
        monkeypatch, capsys, 10.815, "--label", "after", "--fail-if-costlier", "1",
    )
    assert code == 0, out
    assert "Verdict: NO WINNER, the change (+5.0%) is not larger than the run-to-run noise bound (9.5%" in out
    assert "FAIL" not in out and "MORE EXPENSIVE" not in out


def test_a_history_record_without_id_is_skipped_not_counted_twice(history, monkeypatch, capsys):
    for wall in (10.0, 10.01):
        run_clocked(monkeypatch, capsys, wall, "--label", "a")
    rewrite_history(history, lambda index, record: record.pop("id") if index == 1 else None)
    code, out = run_clocked(
        monkeypatch, capsys, 5.0, "--label", "b", "--fail-if-costlier", "0",
    )
    assert code == 5, out
    assert "note: skipped 1 unreadable line(s)" in out
    assert "Verdict: NOT CALIBRATED" in out
    assert "CHEAPER" not in out


def test_stale_checks_neither_calibrate_nor_serve_as_a_judged_baseline(
    history, monkeypatch, capsys
):
    two_days_ago = "2020-01-01T00:00:00Z"
    for wall in (10.0, 10.01, 10.02):
        run_clocked(monkeypatch, capsys, wall, "--label", "a")
    rewrite_history(history, lambda _i, record: record.update(created_at=two_days_ago))
    run_clocked(monkeypatch, capsys, 10.0, "--label", "a")
    # Fresh baseline, but its only repeats are years old: not calibrated.
    code, out = run_clocked(
        monkeypatch, capsys, 5.0, "--label", "b", "--fail-if-costlier", "0", "--no-save",
    )
    assert code == 5, out
    assert "ignored 3 check(s) older than 24 h or undated" in out
    assert "Verdict: NOT CALIBRATED" in out and "CHEAPER" not in out

    # Fresh calibration of the new config, judged against a years-old baseline.
    old_id = json.loads((history / "checks.ndjson").read_text().splitlines()[0])["id"]
    for wall in (5.0, 5.005, 5.01):
        run_clocked(monkeypatch, capsys, wall, "--label", "b")
    code, out = run_clocked(
        monkeypatch, capsys, 5.0, "--label", "b", "--against", old_id,
        "--fail-if-costlier", "0",
    )
    assert code == 5, out
    assert "the baseline check is " in out and "days old; run-to-run noise is only" in out
    assert "Verdict: NOT CALIBRATED" in out and "CHEAPER" not in out


@pytest.mark.parametrize("new_rate", ["2.20", "1.80"])
def test_changing_only_the_assumed_rate_is_never_a_measured_winner(
    history, monkeypatch, capsys, new_rate
):
    for wall in (10.02, 9.99, 10.0):
        run_clocked(monkeypatch, capsys, wall, rate="2.00")
    # Identical measured throughput; only the rate the user typed changed.
    code, out = run_clocked(
        monkeypatch, capsys, 10.0, "--fail-if-costlier", "5", rate=new_rate,
    )
    assert code == 0, out
    assert f"gpu_hourly_rate_usd (ASSUMED): 2.0000 -> {float(new_rate):.4f}" in out
    assert "Verdict: NO WINNER" in out
    assert "CHEAPER" not in out and "MORE EXPENSIVE" not in out
    assert "the measured change at the baseline's GPU rate (+0.0%)" in out


def test_real_throughput_gain_with_a_rate_change_is_judged_on_the_measured_part(
    history, monkeypatch, capsys
):
    for wall in (10.02, 9.99, 10.0):
        run_clocked(monkeypatch, capsys, wall, rate="2.00")
    # 2.5x faster at twice the assumed rate: $/M -20% at the typed rates,
    # -60% at the baseline's rate (the measured part).
    code, out = run_clocked(monkeypatch, capsys, 4.0, rate="4.00")
    assert code == 0, out
    assert "(-20.0%)" in out  # the change at the rates as typed
    assert "At the old rate this check measures $11.12/M (-60.0% vs before)" in out
    assert "Verdict: CHEAPER, the -60.0% measured change at the baseline's GPU rate" in out


def test_a_throttle_upgrade_is_listed_as_a_change(history, monkeypatch, capsys):
    run_clocked(monkeypatch, capsys, 10.0)
    rewrite_history(history, lambda _i, record: record.update(throttle_version="0.0.1"))
    code, out = run_clocked(monkeypatch, capsys, 10.0)
    assert code == 0, out
    assert "throttle_version (the measuring tool): 0.0.1 -> " in out
    assert "nothing Throttle can see" not in out
