"""Caller-visible behaviour of `throttle savings` (verified-savings statement).

Checks are written straight into a history file as fixtures, and the vLLM
/metrics counter is faked, so no server is needed. Every test drives the real
CLI entry point (`throttle.cli.main`).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import throttle.check as check_module
import throttle.savings as savings_module
from throttle.cli import main

T0 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


def flat(out: str) -> str:
    """The statement with panel borders removed and wrapped lines rejoined."""

    return " ".join(
        " ".join(line.strip().strip("│|").split()) for line in out.splitlines()
    )
METRICS_URL = "http://127.0.0.1:8000/metrics"


def make_check(
    check_id: str,
    minutes: int,
    mean: float,
    *,
    half_width: float = 0.01,
    model: str = "qwen-32b",
    quant: str = "none",
    output_per_request: int = 100,
    gpus: int | None = None,
    rate: float = 2.0,
) -> dict:
    """A recorded check: 3 blocks x 4 requests, $/M output tokens = mean +/- half_width."""

    return {
        "record_type": check_module.RECORD_TYPE,
        "record_version": check_module.RECORD_VERSION,
        "id": check_id,
        "created_at": (T0 + timedelta(minutes=minutes)).isoformat().replace("+00:00", "Z"),
        "throttle_version": "test",
        "endpoint": "http://127.0.0.1:8000/v1/chat/completions",
        "metric": "output",
        "fingerprint": {
            "model": model,
            "label": quant,
            "gpu_hourly_rate_usd": rate,
            **({"gpu_count": gpus} if gpus else {}),
            "user_config": {"quant": quant},
        },
        "workload": {
            "prompts_sha256": "abc123",
            "requests_per_block": 4,
            "concurrency": 2,
            "max_tokens": 256,
            "prompt_cache_mode": "cold",
        },
        "blocks": [
            {"output_tokens": output_per_request * 4, "max_tokens_hits": 0} for _ in range(3)
        ],
        "result": {
            "output": {"mean": mean, "ci_low": mean - half_width, "ci_high": mean + half_width},
        },
    }


def write_history(directory: Path, *candidates: dict) -> None:
    # Three repeats of the baseline config calibrate run-to-run noise (2 df,
    # ~1% SD); "base-3" is the baseline the candidates are judged against.
    for record in (
        make_check("base-1", 0, 1.01),
        make_check("base-2", 10, 0.99),
        make_check("base-3", 20, 1.00),
        *candidates,
    ):
        check_module.append_history(directory, record)


def run_savings(capsys, history: Path, *extra: str) -> tuple[int, str, str]:
    code = main(["savings", "--history-dir", str(history), *extra])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def pair(*extra: str) -> list[str]:
    return ["--baseline", "base-3", "--candidate", "cand", *extra]


@pytest.fixture
def history(tmp_path) -> Path:
    return tmp_path / "history"


def test_cheaper_pair_gives_conservative_dollars(history, capsys):
    write_history(history, make_check("cand", 30, 0.60, quant="fp8"))
    code, out, err = run_savings(capsys, history, *pair("--tokens", "1B", "--period-label", "2026-10"))
    assert code == 0, err
    # CI term (1.00 - 0.01) - (0.60 + 0.01) = $0.38/M; noise term
    # (1 - 6.085%) x 1.00 - 0.60 = $0.3391/M; the smaller x 1,000M = $339.15.
    assert "Verified savings (conservative): $339.15" in out
    assert "$0.3800/M" in flat(out) and "$0.3391/M" in flat(out)
    # The point estimate (0.40/M) is shown, labelled, and is not the headline.
    assert "Point estimate (not the verified figure): $400.00" in out
    assert "period      2026-10" in out
    assert "1,000,000,000  [REPORTED BY OPERATOR]" in out
    assert "$/M output tokens" in out
    assert "base-3" in out and "cand" in out and "1 GPU(s)" in out
    assert "GPU $/hr is ASSUMED" in out


def test_no_winner_is_refused(history, capsys):
    write_history(history, make_check("cand", 30, 0.995, quant="fp8"))
    code, out, err = run_savings(capsys, history, *pair("--tokens", "1B"))
    assert code == 1
    assert out == ""
    assert err.startswith("Savings refused: re-judged verdict is NO WINNER")
    assert len(err.strip().splitlines()) == 1


def test_output_changed_is_refused(history, capsys):
    # Much cheaper per token, but answers are 30% longer: not a saving.
    write_history(history, make_check("cand", 30, 0.60, quant="fp8", output_per_request=130))
    code, out, err = run_savings(capsys, history, *pair("--tokens", "1B"))
    assert code == 1
    assert "re-judged verdict is OUTPUT CHANGED" in err
    assert "$" not in out


def test_not_calibrated_is_refused(history, capsys):
    check_module.append_history(history, make_check("base-3", 0, 1.00))
    check_module.append_history(history, make_check("cand", 10, 0.60, quant="fp8"))
    code, _out, err = run_savings(capsys, history, *pair("--tokens", "1B"))
    assert code == 1
    assert "re-judged verdict is NOT CALIBRATED" in err


def test_model_mismatch_is_refused(history, capsys):
    write_history(history, make_check("cand", 30, 0.60, quant="fp8", model="llama-8b"))
    code, _out, err = run_savings(capsys, history, *pair("--tokens", "1B"))
    assert code == 1
    assert "different models (qwen-32b vs llama-8b)" in err


def test_workload_mismatch_is_refused(history, capsys):
    candidate = make_check("cand", 30, 0.60, quant="fp8")
    candidate["workload"]["prompts_sha256"] = "other"
    write_history(history, candidate)
    code, _out, err = run_savings(capsys, history, *pair("--tokens", "1B"))
    assert code == 1
    assert "the workload differs (prompts_sha256)" in err


def test_unknown_check_id_is_refused(history, capsys):
    write_history(history)
    code, _out, err = run_savings(capsys, history, *pair("--tokens", "1B"))
    assert code == 1
    assert "no candidate check with id 'cand'" in err


def test_token_source_must_be_exactly_one(history, capsys):
    write_history(history, make_check("cand", 30, 0.60, quant="fp8"))
    assert run_savings(capsys, history, *pair())[0] == 2
    assert run_savings(capsys, history, *pair("--metrics-url", METRICS_URL))[0] == 2
    assert run_savings(
        capsys, history, *pair("--tokens", "1B", "--metrics-url", METRICS_URL, "--window-start", "x")
    )[0] == 2


def fake_counter(monkeypatch, value: float | str) -> list[str]:
    """Serve a fake /metrics body: a number is one unlabelled series, a str is the body."""

    body = value if isinstance(value, str) else (
        "# TYPE vllm:generation_tokens_total counter\n"
        f"vllm:generation_tokens_total {value}\n"
        "vllm:num_requests_running 3.0\n"
    )
    calls: list[str] = []

    def fetch(url: str) -> str:
        calls.append(url)
        return body

    monkeypatch.setattr(savings_module, "fetch_metrics_text", fetch)
    return calls


def engines(*values: tuple[str, str, float]) -> str:
    """A vLLM V1 body with one series per (model_name, engine, value)."""

    return "".join(
        f'vllm:generation_tokens_total{{engine="{engine}",model_name="{model}"}} {value}\n'
        for model, engine, value in values
    ) + 'vllm:generation_tokens_total_created{engine="0",model_name="qwen-32b"} 1.7e9\n'


def snapshot_then_report(monkeypatch, capsys, history, tmp_path, start_body, end_body):
    write_history(history, make_check("cand", 30, 0.60, quant="fp8"))
    start = tmp_path / "start.json"
    fake_counter(monkeypatch, start_body)
    assert main(["savings", "snapshot", "--metrics-url", METRICS_URL, "--out", str(start)]) == 0
    capsys.readouterr()
    fake_counter(monkeypatch, end_body)
    return run_savings(
        capsys, history, *pair("--metrics-url", METRICS_URL, "--window-start", str(start))
    )


def test_snapshot_delta_is_measured_tokens(history, tmp_path, monkeypatch, capsys):
    write_history(history, make_check("cand", 30, 0.60, quant="fp8"))
    start = tmp_path / "start.json"
    fake_counter(monkeypatch, 5_000_000)
    # The snapshot is taken after the candidate check (T0 + 30 min).
    monkeypatch.setattr(savings_module, "_now", lambda: T0 + timedelta(hours=1))
    assert main(["savings", "snapshot", "--metrics-url", METRICS_URL, "--out", str(start)]) == 0
    capsys.readouterr()
    saved = json.loads(start.read_text())
    assert saved["series"] == [{"labels": {}, "value": 5_000_000}]
    assert saved["taken_at"] == "2026-10-01T13:00:00Z"

    calls = fake_counter(monkeypatch, 505_000_000)
    monkeypatch.setattr(savings_module, "_now", lambda: T0 + timedelta(days=30))
    code, out, err = run_savings(
        capsys, history, *pair("--metrics-url", METRICS_URL, "--window-start", str(start))
    )
    assert code == 0, err
    assert calls == [METRICS_URL]
    assert "500,000,000  [MEASURED]" in out
    assert "period      2026-10-01T13:00:00Z to 2026-10-31T12:00:00Z" in out
    # $0.3391/M (the noise term) x 500M = $169.57.
    assert "Verified savings (conservative): $169.57" in out


def test_counter_reset_is_refused(history, tmp_path, monkeypatch, capsys):
    write_history(history, make_check("cand", 30, 0.60, quant="fp8"))
    start = tmp_path / "start.json"
    fake_counter(monkeypatch, 900_000_000)
    assert main(["savings", "snapshot", "--metrics-url", METRICS_URL, "--out", str(start)]) == 0
    capsys.readouterr()
    fake_counter(monkeypatch, 1_000)  # the server restarted
    code, out, err = run_savings(
        capsys, history, *pair("--metrics-url", METRICS_URL, "--window-start", str(start))
    )
    assert code == 1
    assert out == ""
    assert "the counter reset" in err


def test_snapshot_from_another_endpoint_is_refused(history, tmp_path, monkeypatch, capsys):
    write_history(history, make_check("cand", 30, 0.60, quant="fp8"))
    start = tmp_path / "start.json"
    fake_counter(monkeypatch, 1_000)
    main(["savings", "snapshot", "--metrics-url", METRICS_URL, "--out", str(start)])
    capsys.readouterr()
    code, _out, err = run_savings(
        capsys, history,
        *pair("--metrics-url", "http://127.0.0.1:9000/metrics", "--window-start", str(start)),
    )
    assert code == 1
    assert "snapshot was taken from" in err


def test_json_statement_shape(history, capsys):
    write_history(history, make_check("cand", 30, 0.60, quant="fp8"))
    code, out, err = run_savings(capsys, history, *pair("--tokens", "2M", "--json"))
    assert code == 0, err
    statement = json.loads(out)
    assert statement["record_type"] == "savings_statement"
    assert statement["record_version"] == 2
    assert set(statement) >= {
        "created_at", "period", "metric", "metric_label", "verdict", "verdict_reason",
        "baseline", "candidate", "tokens", "savings", "assumptions", "rate_factor",
    }
    assert statement["verdict"] == "CHEAPER"
    assert statement["metric"] == "output"
    for side, check_id in (("baseline", "base-3"), ("candidate", "cand")):
        assert statement[side]["check_id"] == check_id
        assert statement[side]["gpu_count"] == 1
        assert set(statement[side]["dollars_per_million"]) == {"mean", "ci_low", "ci_high"}
        assert statement[side]["gpu_hourly_rate_source"].startswith("ASSUMED")
    assert statement["tokens"] == {"count": 2_000_000, "source": "REPORTED BY OPERATOR"}
    assert statement["savings"]["ci_term_dollars_per_million"] == pytest.approx(0.38)
    assert statement["savings"]["noise_bound_percent"] == pytest.approx(6.0854, abs=1e-3)
    assert statement["savings"]["noise_term_dollars_per_million"] == pytest.approx(0.33915, abs=1e-4)
    assert statement["savings"]["conservative_usd"] == pytest.approx(0.33915 * 2, abs=1e-3)
    assert statement["savings"]["point_estimate_usd"] == pytest.approx(0.80)
    assert statement["savings"]["conservative_usd"] <= statement["savings"]["point_estimate_usd"]


# ---------------------------------------------------------------------------
# Per-series counters: vLLM V1 with data parallelism exposes one series per
# engine; other models on the same /metrics must not be billed.
# ---------------------------------------------------------------------------


def test_data_parallel_engines_are_summed(history, tmp_path, monkeypatch, capsys):
    code, out, err = snapshot_then_report(
        monkeypatch, capsys, history, tmp_path,
        engines(("qwen-32b", "0", 1_000), ("qwen-32b", "1", 2_000)),
        engines(("qwen-32b", "0", 300_001_000), ("qwen-32b", "1", 200_002_000)),
    )
    assert code == 0, err
    assert "500,000,000  [MEASURED]" in out
    assert "summed over 2 series with model_name=qwen-32b" in flat(out)
    assert "Verified savings (conservative): $169.57" in out


def test_series_order_swapped_still_totals_per_series(history, tmp_path, monkeypatch, capsys):
    # Engine 1 listed first at the report: last-series-wins would compare
    # engine 1 now against engine 0 then.
    code, out, err = snapshot_then_report(
        monkeypatch, capsys, history, tmp_path,
        engines(("qwen-32b", "0", 10_000_000), ("qwen-32b", "1", 900_000_000)),
        engines(("qwen-32b", "1", 1_000_000_000), ("qwen-32b", "0", 20_000_000)),
    )
    assert code == 0, err
    assert "110,000,000  [MEASURED]" in out


def test_reset_in_one_series_is_refused(history, tmp_path, monkeypatch, capsys):
    code, out, err = snapshot_then_report(
        monkeypatch, capsys, history, tmp_path,
        engines(("qwen-32b", "0", 5_000), ("qwen-32b", "1", 900_000)),
        engines(("qwen-32b", "0", 9_000_000), ("qwen-32b", "1", 10)),
    )
    assert code == 1 and out == ""
    assert "the counter reset" in err and "'engine': '1'" in err


def test_series_missing_at_report_is_refused(history, tmp_path, monkeypatch, capsys):
    code, _out, err = snapshot_then_report(
        monkeypatch, capsys, history, tmp_path,
        engines(("qwen-32b", "0", 5_000), ("qwen-32b", "1", 6_000)),
        engines(("qwen-32b", "0", 9_000_000)),
    )
    assert code == 1
    assert "gone now" in err


def test_new_series_counts_from_zero_and_is_noted(history, tmp_path, monkeypatch, capsys):
    code, out, err = snapshot_then_report(
        monkeypatch, capsys, history, tmp_path,
        engines(("qwen-32b", "0", 1_000_000)),
        engines(("qwen-32b", "0", 101_000_000), ("qwen-32b", "1", 50_000_000)),
    )
    assert code == 0, err
    assert "150,000,000  [MEASURED]" in out
    assert "note: 1 series new since the snapshot, counted from 0" in flat(out)


def test_other_models_series_are_not_counted(history, tmp_path, monkeypatch, capsys):
    code, out, err = snapshot_then_report(
        monkeypatch, capsys, history, tmp_path,
        engines(("qwen-32b", "0", 0), ("llama-8b", "0", 0)),
        engines(("qwen-32b", "0", 40_000_000), ("llama-8b", "0", 999_000_000)),
    )
    assert code == 0, err
    assert "40,000,000  [MEASURED]" in out
    assert "summed over 1 series" in flat(out)


def test_no_series_for_the_model_is_refused(history, tmp_path, monkeypatch, capsys):
    code, out, err = snapshot_then_report(
        monkeypatch, capsys, history, tmp_path,
        engines(("llama-8b", "0", 0)),
        engines(("llama-8b", "0", 999_000_000)),
    )
    assert code == 1 and out == ""
    assert "has model_name='qwen-32b'" in err
    assert len(err.strip().splitlines()) == 1


# ---------------------------------------------------------------------------
# Pre-release review fixes: a statement that may be invoiced never overstates.
# ---------------------------------------------------------------------------


def test_per_gpu_price_rise_is_real_cost_not_a_saving(history, capsys):
    # Baseline 1 GPU at $2/hr, $1.00/M. Candidate 1 GPU at $4/hr (a pricier
    # GPU type), $1.40/M: 40% more per token. Scaling the candidate down by
    # 2/4 used to call it CHEAPER and bill $285.
    write_history(history, make_check("cand", 30, 1.40, rate=4.0, quant="h100"))
    code, out, err = run_savings(capsys, history, *pair("--tokens", "1B"))
    assert code == 1 and out == ""
    assert "re-judged verdict is MORE EXPENSIVE" in err
    assert "higher per-GPU price counted as real cost" in err


def test_total_rate_for_more_gpus_without_gpus_flag_is_not_a_saving(history, capsys):
    # TP=2 on 2 GPUs, the total $4/hr passed but --gpus forgotten.
    write_history(history, make_check("cand", 30, 1.40, rate=4.0, quant="tp2"))
    code, _out, err = run_savings(capsys, history, *pair("--tokens", "1B"))
    assert code == 1
    assert "not CHEAPER" in err


def test_assumed_price_cut_still_never_counts(history, capsys):
    # Same speed, rate typed lower ($1/hr): $/M halves but nothing was measured faster.
    write_history(history, make_check("cand", 30, 0.50, rate=1.0, quant="cheap"))
    code, _out, err = run_savings(capsys, history, *pair("--tokens", "1B"))
    assert code == 1
    assert "re-judged verdict is NO WINNER" in err


def test_window_starting_before_candidate_is_refused(history, tmp_path, monkeypatch, capsys):
    write_history(history, make_check("cand", 30, 0.60, quant="fp8"))
    start = tmp_path / "start.json"
    fake_counter(monkeypatch, 0)
    monkeypatch.setattr(savings_module, "_now", lambda: T0 + timedelta(minutes=29))
    assert main(["savings", "snapshot", "--metrics-url", METRICS_URL, "--out", str(start)]) == 0
    capsys.readouterr()
    fake_counter(monkeypatch, 5_000_000_000)
    monkeypatch.setattr(savings_module, "_now", lambda: T0 + timedelta(days=30))
    code, out, err = run_savings(
        capsys, history, *pair("--metrics-url", METRICS_URL, "--window-start", str(start))
    )
    assert code == 1 and out == ""
    assert err.strip() == (
        "Savings refused: the window starts before the candidate config was measured; "
        "take a new snapshot after deploying it."
    )


def test_conservative_figure_includes_run_to_run_noise(history, capsys):
    # Tight candidate CI: the CI term alone is (1.00 - 0.01) - (0.90 + 0.001) = $0.089/M.
    # The 6.085% noise bound leaves (1 - 0.06085) x 1.00 - 0.90 = $0.0391/M.
    write_history(history, make_check("cand", 30, 0.90, half_width=0.001, quant="fp8"))
    code, out, err = run_savings(capsys, history, *pair("--tokens", "1B", "--json"))
    assert code == 0, err
    savings = json.loads(out)["savings"]
    assert savings["ci_term_dollars_per_million"] == pytest.approx(0.089, abs=1e-6)
    assert savings["noise_term_dollars_per_million"] == pytest.approx(0.03915, abs=1e-4)
    assert savings["conservative_dollars_per_million"] == savings["noise_term_dollars_per_million"]
    assert savings["conservative_usd"] == pytest.approx(39.15, abs=0.01)


def test_noise_term_not_positive_is_refused(monkeypatch, history, capsys):
    # Guard: a CHEAPER verdict whose noise-adjusted reduction is not
    # positive must refuse rather than bill. Hand the statement a CHEAPER
    # comparison whose noise bound (45%) exceeds the 40% reduction.
    write_history(history, make_check("cand", 30, 0.60, quant="fp8"))
    real = savings_module.compare_checks

    def wide(*args, **kwargs):
        comparison = real(*args, **kwargs)
        return {**comparison, "noise": {**comparison["noise"], "floor_percent": 45.0}}

    monkeypatch.setattr(savings_module, "compare_checks", wide)
    code, out, err = run_savings(capsys, history, *pair("--tokens", "1B"))
    assert code == 1 and out == ""
    assert "the conservative $/M reduction is not positive" in err
    assert "noise term" in err
