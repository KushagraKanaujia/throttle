"""`throttle autotune`: search space parsing, the search and its guards,
cloud metering (402/403/401, idempotency, offline), the dry run, and one real
launch of a stub server subprocess on 127.0.0.1.

Cloud calls go to tests/cloud_stub.py through httpx.MockTransport.
"""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import pytest

import throttle.autotune as autotune
import throttle.check as check_module
import throttle.cloud as cloud
import throttle.config as config_module
from cloud_stub import KEY, CloudStub
from throttle.cli import build_parser, main

FAKE_SERVER = Path(__file__).parent / "fixtures" / "fake_openai_server.py"
SMALL = ["--blocks", "3", "--requests-per-block", "2", "--no-metrics"]


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv(check_module.HISTORY_ENV, str(tmp_path / "history"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for name in (cloud.API_KEY_ENV, cloud.CLOUD_URL_ENV, "OPENAI_API_KEY", "CI", "GITHUB_ACTIONS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", tmp_path / "no-config.yaml")
    monkeypatch.setattr(cloud, "RETRY_DELAY_SECONDS", 0.0)
    stub = CloudStub()
    monkeypatch.setattr(cloud, "_TRANSPORT", stub.transport())
    return stub


def parse(*argv: str):
    return build_parser().parse_args(["autotune", *argv])


def run(capsys, *argv: str) -> tuple[int, str, str]:
    code = main(["autotune", *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def run_with(capsys, args, **kwargs) -> tuple[int, str, str]:
    code = autotune.handle_autotune(args, **kwargs)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


DEMO = [
    "--launch", "serve {model} --tp {tp} --seqs {seqs} {quant}",
    "--space", "seqs=64,128,256", "quant=,--quantization fp8", "tp=1,2",
    "--url", "http://127.0.0.1:8000", "--model", "m", "--gpu-hourly-rate", "2",
]


# -- parsing and validation ---------------------------------------------------


def test_parse_space_items_and_files(tmp_path):
    assert autotune.parse_space(["tp=1,2", "seqs=64,128"]) == {"tp": ["1", "2"], "seqs": ["64", "128"]}
    js = tmp_path / "space.json"
    js.write_text(json.dumps({"tp": [1, 2], "mem": [0.85, 0.9], "quant": ["", "--quantization fp8"]}))
    assert autotune.parse_space([str(js)]) == {
        "tp": ["1", "2"], "mem": ["0.85", "0.9"], "quant": ["", "--quantization fp8"],
    }
    yaml = pytest.importorskip("yaml")
    ym = tmp_path / "space.yaml"
    ym.write_text(yaml.safe_dump({"seqs": [64, 128, 256]}))
    assert autotune.parse_space([str(ym)]) == {"seqs": ["64", "128", "256"]}
    for bad in (["tp"], ["tp=1,1"], ["bad key=1"], ["tp=1", "tp=2"]):
        with pytest.raises(autotune.AutotuneError):
            autotune.parse_space(bad)


def test_baseline_defaults_to_first_values_and_must_be_in_space():
    space = {"tp": ["1", "2"], "seqs": ["64", "128"]}
    assert autotune.parse_baseline(None, space) == {"tp": "1", "seqs": "64"}
    assert autotune.parse_baseline(["seqs=128"], space) == {"tp": "1", "seqs": "128"}
    with pytest.raises(autotune.AutotuneError):
        autotune.parse_baseline(["seqs=512"], space)
    with pytest.raises(autotune.AutotuneError):
        autotune.parse_baseline(["mem=0.9"], space)


def test_render_inserts_values_verbatim_and_drops_empty_flags():
    template = "vllm serve {model} --max-num-seqs {seqs} {quant}"
    assert autotune.render(template, {"seqs": "64", "quant": ""}, "m") == "vllm serve m --max-num-seqs 64"
    assert autotune.render(template, {"seqs": "64", "quant": "--quantization fp8"}, "m") == (
        "vllm serve m --max-num-seqs 64 --quantization fp8"
    )


@pytest.mark.parametrize(
    "argv,message",
    [
        (["--launch", "serve {tp} {mem}", "--space", "tp=1,2"], "--space does not define"),
        (["--launch", "serve {tp}", "--space", "tp=1,2", "seqs=1,2"], "never appears in --launch"),
        (["--launch", "serve {tp}", "--space", "tp=1,x"], "must be whole"),
        (["--launch", "serve {tp}", "--space", "tp=1,2", "--gpus", "2"], "conflicts"),
        (["--launch", "serve {model} {tp}", "--space", "tp=1,2", "--model", ""], "pass --model"),
        (["--launch", "serve {tp}", "--space", "tp=1,2", "--baseline-runs", "2"], "at least 3"),
        (["--launch", "serve {tp}", "--space", "tp=1,2", "--against", "x"], "--against"),
        (["--launch", "serve {tp}", "--space", "tp=1,2", "--fail-if-costlier", "5"], "--fail-if-costlier"),
    ],
)
def test_usage_errors_before_anything_runs_or_is_charged(env, capsys, argv, message):
    cloud.write_credentials(KEY)
    model = [] if "--model" in argv else ["--model", "m"]
    code, out, err = run(capsys, *argv, *model, "--url", "http://127.0.0.1:8000",
                         "--gpu-hourly-rate", "2")
    assert code == 2
    assert message in err
    assert env.requests == []


def test_priciest_trial_is_checked_against_safety_limits_first(env, capsys):
    cloud.write_credentials(KEY)
    code, out, err = run(capsys, "--launch", "serve {tp}", "--space", "tp=1,8",
                         "--url", "http://127.0.0.1:8000", "--model", "m",
                         "--gpu-hourly-rate", "4", "--max-estimated-spend", "1")
    assert code == 2
    assert "the 8-GPU trial would be refused" in err
    assert env.requests == []


def test_autotune_requires_login_but_dry_run_does_not(env, capsys, tmp_path):
    code, out, err = run(capsys, *DEMO)
    assert code == 2
    assert "throttle login" in err and "--dry-run" in err and "pricing" in err
    code, out, err = run(capsys, "--dry-run", "--history-dir", str(tmp_path / "at"), *SMALL)
    assert code == 0, err
    assert env.requests == []  # a dry run never contacts Throttle Cloud
    assert "DRY RUN" in out and "SIMULATED" in out


def test_autotune_help_says_rate_is_per_gpu_and_hides_check_only_flags(capsys):
    with pytest.raises(SystemExit):
        main(["autotune", "--help"])
    text = capsys.readouterr().out
    assert "ONE GPU" in text and "--launch" in text and "--space" in text
    for hidden in ("--share-id", "--against", "--fail-if-costlier", "--push"):
        assert hidden not in text


# -- the search, on the simulator ----------------------------------------------


def dry(capsys, tmp_path, *extra: str) -> tuple[int, str, dict]:
    out_json = tmp_path / "result.json"
    code, out, err = run(capsys, "--dry-run", "--history-dir", str(tmp_path / "at"),
                         "--json", str(out_json), *SMALL, *extra)
    assert code == 0, err + out
    return code, out, json.loads(out_json.read_text())


def test_dry_run_demo_finds_a_verified_cheaper_config(env, capsys, tmp_path):
    _, out, result = dry(capsys, tmp_path, "--monthly-tokens", "2B")
    assert result["simulated"] is True and result["found_cheaper"] is True
    best = result["best"]["usd_per_m"]
    base = result["baseline"]["usd_per_m"]
    assert best < base
    assert result["projected_monthly_savings_usd"] == pytest.approx((base - best) * 2000)
    assert result["conservative_monthly_savings_usd"] is not None
    assert 0 < result["conservative_monthly_savings_usd"] <= result["projected_monthly_savings_usd"]
    trials = result["trials"]
    assert trials[0]["role"] == "baseline" and len(trials[0]["check_ids"]) == 3
    assert len(trials) - 1 <= autotune.DEFAULT_MAX_TRIALS
    winner = next(t for t in trials if t["config"] == result["best_config"])
    assert winner["verdict"] == check_module.VERDICT_CHEAPER and winner["eligible"]
    assert result["launch_command"] == autotune.render(autotune.DEMO_LAUNCH, result["best_config"], "demo-model")
    assert "Launch command (ready to paste):" in out
    assert result["launch_command"] in out
    assert "saves $" in out and "[PROJECTED: SIMULATED $/M x ASSUMED volume]" in out


def test_output_changed_and_more_gpus_are_never_picked(env, capsys, tmp_path):
    _, out, result = dry(capsys, tmp_path, "--max-trials", "20")
    assert "fp8" not in result["best_config"]["quant"]
    assert result["best_config"]["tp"] == "1"
    verdicts = {(t["config"]["quant"], t["config"]["tp"]): t["verdict"] for t in result["trials"][1:]}
    assert any(v == check_module.VERDICT_OUTPUT_CHANGED for (q, _), v in verdicts.items() if q)
    fp8 = [t for t in result["trials"] if t["config"]["quant"]]
    assert fp8 and all(t["verdict"] == check_module.VERDICT_OUTPUT_CHANGED for t in fp8)
    assert all(not t["eligible"] for t in fp8)
    # the simulated fp8 is far cheaper per token, and still rejected
    assert min(t["usd_per_m"] for t in fp8) < result["best"]["usd_per_m"]
    two_gpu = [t for t in result["trials"] if t["config"]["tp"] == "2"]
    assert two_gpu and all(t["gpus"] == 2 for t in two_gpu)
    assert "rejected as OUTPUT CHANGED" in out


def test_gpu_count_is_cost(env, capsys, tmp_path):
    # Only tp changes: 1.56x faster on 2 GPUs is 28% more per token.
    _, out, result = dry(capsys, tmp_path, "--launch", "serve {tp}", "--space", "tp=1,2")
    two = result["trials"][1]
    assert two["config"] == {"tp": "2"} and two["gpus"] == 2
    assert two["verdict"] == check_module.VERDICT_COSTLIER
    assert result["found_cheaper"] is False
    assert "keep your current config" in out
    assert result["launch_command"] == "serve 1"
    # --gpus-param none: the same speedup at a 1-GPU price would look cheaper
    _, out, result = dry(capsys, tmp_path, "--launch", "serve {tp}", "--space", "tp=1,2",
                         "--gpus-param", "none")
    assert result["trials"][1]["gpus"] == 1
    assert result["trials"][1]["verdict"] == check_module.VERDICT_CHEAPER


def test_noise_bound_rejects_a_change_inside_the_noise(env, capsys, tmp_path):
    # mem 0.85 -> 0.86 is under 1% in the simulator: within noise, never picked.
    _, out, result = dry(capsys, tmp_path, "--launch", "serve {mem}", "--space", "mem=0.85,0.86")
    assert result["trials"][1]["verdict"] in (check_module.VERDICT_NO_WINNER,)
    assert result["found_cheaper"] is False
    assert result["noise_bound_percent"] is not None


def test_budget_and_early_stopping(env, capsys, tmp_path):
    _, _, result = dry(capsys, tmp_path, "--max-trials", "2")
    assert len(result["trials"]) == 3
    assert result["stopped"] == "reached --max-trials 2"
    # seqs only, peaked at 256: walks up 128 -> 256, one tie step to 512, stops.
    _, _, result = dry(capsys, tmp_path, "--launch", "serve {seqs}", "--space",
                       "seqs=64,128,256,512,1024,2048")
    tried = [t["config"]["seqs"] for t in result["trials"]]
    assert tried[0] == "64" and "2048" not in tried
    assert result["best_config"] == {"seqs": "256"} or result["best_config"] == {"seqs": "512"}


def test_baseline_in_the_middle_walks_both_ways(env, capsys, tmp_path):
    _, _, result = dry(capsys, tmp_path, "--launch", "serve {seqs}", "--space",
                       "seqs=64,128,256,512", "--baseline", "seqs=64")
    assert result["baseline_config"] == {"seqs": "64"}
    _, _, result = dry(capsys, tmp_path, "--launch", "serve {seqs}", "--space",
                       "seqs=64,128,256,512,1024,2048", "--baseline", "seqs=2048")
    tried = [t["config"]["seqs"] for t in result["trials"]]
    assert tried[:2] == ["2048", "1024"]  # nothing above 2048: walks down
    assert result["found_cheaper"] is True


def test_trial_checks_are_saved_and_never_pushed(env, capsys, tmp_path):
    cloud.write_credentials(KEY)
    _, _, result = dry(capsys, tmp_path, "--push")
    run_dir = tmp_path / "at" / result["run_id"]
    records, skipped = check_module.load_history(run_dir)
    assert skipped == 0
    assert len(records) == sum(len(t["check_ids"]) for t in result["trials"])
    assert (run_dir / "autotune.json").exists()
    assert all(r["fingerprint"]["label"].startswith("autotune-") for r in records)
    assert env.requests == []  # --push on autotune never uploads trial checks


# -- cloud metering --------------------------------------------------------------


def metered(capsys, env, tmp_path, *extra: str):
    args = parse("--dry-run", "--history-dir", str(tmp_path / "at"), "--json",
                 str(tmp_path / "r.json"), *SMALL, *extra)
    sim = autotune.SimulatedServer("demo-model", args.concurrency)
    code, out, err = run_with(capsys, args, backend=sim, client=cloud.CloudClient(KEY))
    result = json.loads((tmp_path / "r.json").read_text()) if (tmp_path / "r.json").exists() else None
    return code, out, err, result, sim


def test_every_trial_is_metered_before_it_runs_with_an_idempotency_key(env, capsys, tmp_path):
    code, out, err, result, sim = metered(capsys, env, tmp_path, "--max-trials", "4")
    assert code == 0, err
    trial_posts = [r for r in env.requests if r.url.path == "/api/v1/autotune/trials"]
    assert len(trial_posts) == len(result["trials"]) == sim.trials
    run_id = result["run_id"]
    assert [r.headers["idempotency-key"] for r in trial_posts] == [
        f"autotune:{run_id}:trial:{i}" for i in range(len(trial_posts))
    ]
    bodies = [json.loads(r.content) for r in trial_posts]
    assert [b["trial_index"] for b in bodies] == list(range(len(bodies)))
    assert bodies[0]["config"] == result["baseline_config"]
    assert env.paths()[0] == "/api/v1/me"
    assert env.paths()[-1] == f"/api/v1/autotune/runs/{run_id}/result"
    final = env.bodies[-1]
    assert final["best_config"] == result["best_config"]
    assert final["baseline_usd_per_m"] == result["baseline"]["usd_per_m"]
    assert final["best_usd_per_m"] == result["best"]["usd_per_m"]
    assert len(final["trials"]) == len(result["trials"])
    assert "launch" not in json.dumps(final) and "127.0.0.1" not in json.dumps(final)
    assert result["cloud_url"] == "https://cloud.throttle-pro.com/autotune/run1"
    assert "$1.00" in out and "credits left" in out
    assert env.credits() == f"{10 - len(trial_posts):.2f}"


def test_402_mid_run_stops_cleanly_with_the_best_so_far(env, capsys, tmp_path):
    env.credits_micro = 3_000_000  # baseline + 2 candidates
    code, out, err, result, sim = metered(capsys, env, tmp_path)
    assert code == 0, err
    assert sim.trials == 3  # the 4th trial was refused before its server started
    assert "Out of Throttle Cloud credits ($0.00 left)" in out
    assert "https://www.throttle-pro.com/pricing" in out
    assert result["stopped"].startswith("Out of Throttle Cloud credits")
    assert result["trials"][-1]["status"] == "not run"
    assert result["found_cheaper"] is True  # seqs 128 and 256 were measured first
    assert env.paths()[-1].endswith("/result")


def test_402_before_the_baseline_runs_nothing(env, capsys, tmp_path):
    env.credits_micro = 0
    code, out, err, result, sim = metered(capsys, env, tmp_path)
    assert code == 1
    assert sim.trials == 0
    assert "Out of Throttle Cloud credits" in out


@pytest.mark.parametrize("status,text", [(403, "plan does not include Autotune"),
                                          (401, "rejected the API key")])
def test_403_and_401_stop_before_any_trial(env, capsys, tmp_path, status, text):
    env.forced["/api/v1/autotune/trials"] = status
    code, out, err, result, sim = metered(capsys, env, tmp_path)
    assert code == 1
    assert sim.trials == 0
    assert text in out


def test_offline_cloud_stops_autotune_before_the_first_trial(env, capsys, tmp_path):
    env.fail_first = 100
    code, out, err, result, sim = metered(capsys, env, tmp_path)
    assert code == 1
    assert sim.trials == 0
    assert "Could not reach Throttle Cloud" in err


def test_result_upload_failure_is_a_warning(env, capsys, tmp_path):
    env.forced["/api/v1/autotune/runs/"] = 500
    code, out, err, result, sim = metered(capsys, env, tmp_path, "--max-trials", "2")
    assert code == 0
    assert "was not saved to Throttle Cloud" in err
    assert result["cloud_url"] is None


def test_logged_in_cli_autotune_uses_saved_key(env, capsys, tmp_path, monkeypatch):
    cloud.write_credentials(KEY)
    sim = autotune.SimulatedServer("m", 2)
    original = autotune.handle_autotune
    monkeypatch.setattr(autotune, "handle_autotune",
                        lambda args: original(args, backend=sim))
    code, out, err = run(capsys, *DEMO, *SMALL, "--history-dir", str(tmp_path / "at"),
                         "--max-trials", "2")
    assert code == 0, err
    assert env.paths().count("/api/v1/autotune/trials") == 3
    assert env.requests[0].headers["authorization"] == f"Bearer {KEY}"


# -- a real server process ---------------------------------------------------------


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def fast_polls(monkeypatch):
    monkeypatch.setattr(autotune, "HEALTH_POLL_SECONDS", 0.05)
    monkeypatch.setattr(autotune, "STOP_TIMEOUT_SECONDS", 10.0)


def launcher_args(tmp_path, port: int, space: list[str], *extra: str):
    template = f"{sys.executable} {FAKE_SERVER} --port {port} --speed {{speed}}"
    return parse(
        "--launch", template, "--space", *space, "--url", f"http://127.0.0.1:{port}",
        "--model", "fake", "--gpu-hourly-rate", "2", "--history-dir", str(tmp_path / "at"),
        "--json", str(tmp_path / "r.json"), "--blocks", "3", "--requests-per-block", "2",
        "--warmup", "0", "--no-metrics", "--startup-timeout", "20", *extra,
    )


def test_real_launcher_starts_measures_and_stops_each_server(env, capsys, tmp_path, fast_polls):
    port = free_port()
    args = launcher_args(tmp_path, port, ["speed=1,3"])
    code, out, err = run_with(capsys, args, client=cloud.CloudClient(KEY))
    assert code == 0, err + out
    result = json.loads((tmp_path / "r.json").read_text())
    assert result["simulated"] is False
    assert [t["config"]["speed"] for t in result["trials"]] == ["1", "3"]
    assert result["best_config"] == {"speed": "3"}
    assert result["trials"][1]["verdict"] == check_module.VERDICT_CHEAPER
    assert "[MEASURED]" in out
    run_dir = tmp_path / "at" / result["run_id"]
    assert (run_dir / "trial-00.server.log").read_text().startswith("$ ")
    # every server was stopped: nothing answers on the port any more
    with socket.socket() as sock:
        assert sock.connect_ex(("127.0.0.1", port)) != 0


def test_real_launcher_refuses_when_a_server_is_already_running(env, capsys, tmp_path, fast_polls):
    import subprocess

    port = free_port()
    proc = subprocess.Popen([sys.executable, str(FAKE_SERVER), "--port", str(port)])
    try:
        launcher = autotune.ServerLauncher(
            "true", model=None, health_url=f"http://127.0.0.1:{port}/v1/models",
            startup_timeout=5, stop_command=None, log_dir=tmp_path,
        )
        import time
        deadline = time.monotonic() + 10
        while not launcher._healthy() and time.monotonic() < deadline:
            time.sleep(0.05)
        args = launcher_args(tmp_path, port, ["speed=1,3"])
        code, out, err = run_with(capsys, args, client=cloud.CloudClient(KEY))
        assert code == 2
        assert "already answering" in err
        assert "/api/v1/autotune/trials" not in env.paths()  # nothing charged
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_real_launcher_reports_a_crashing_server(env, capsys, tmp_path, fast_polls):
    port = free_port()
    template = f"{sys.executable} {FAKE_SERVER} --port {port} {{crash}}"
    args = parse(
        "--launch", template, "--space", "crash=,--crash", "--url", f"http://127.0.0.1:{port}",
        "--model", "fake", "--gpu-hourly-rate", "2", "--history-dir", str(tmp_path / "at"),
        "--json", str(tmp_path / "r.json"), *SMALL, "--warmup", "0", "--startup-timeout", "20",
    )
    code, out, err = run_with(capsys, args, client=cloud.CloudClient(KEY))
    assert code == 0, err + out
    result = json.loads((tmp_path / "r.json").read_text())
    crashed = result["trials"][1]
    assert crashed["status"] == "failed"
    assert "exited with code" in crashed["error"] and "CUDA out of memory" in crashed["error"]
    assert result["found_cheaper"] is False


def test_a_server_that_will_not_stop_ends_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(autotune, "HEALTH_POLL_SECONDS", 0.01)
    monkeypatch.setattr(autotune, "STOP_TIMEOUT_SECONDS", 0.05)
    launcher = autotune.ServerLauncher(
        "true", model=None, health_url="http://127.0.0.1:9/v1/models", startup_timeout=1,
        stop_command=None, log_dir=tmp_path,
    )
    monkeypatch.setattr(launcher, "_healthy", lambda: True)
    with pytest.raises(autotune.ServerStuck, match="before trial"):
        launcher.start({}, "trial-01")
    with pytest.raises(autotune.ServerStuck, match="still answers"):
        launcher.stop()
