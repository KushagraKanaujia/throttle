"""Throttle Cloud from the CLI: login/logout/whoami, credentials, errors,
idempotency, offline behaviour and `throttle check --push`.

Every cloud call goes to tests/cloud_stub.py through httpx.MockTransport, and
every check runs against autotune's SimulatedServer: no sockets, no network.
"""

from __future__ import annotations

import io
import json
import os
import stat
import sys
from pathlib import Path

import pytest

import throttle.check as check_module
import throttle.cloud as cloud
import throttle.config as config_module
from cloud_stub import KEY, CloudStub
from throttle.autotune import SimulatedServer
from throttle.cli import main

URL = "http://127.0.0.1:8000"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv(check_module.HISTORY_ENV, str(tmp_path / "history"))
    monkeypatch.delenv(cloud.API_KEY_ENV, raising=False)
    monkeypatch.delenv(cloud.CLOUD_URL_ENV, raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", tmp_path / "no-config.yaml")
    monkeypatch.setattr(cloud, "RETRY_DELAY_SECONDS", 0.0)
    stub = CloudStub()
    monkeypatch.setattr(cloud, "_TRANSPORT", stub.transport())
    return stub


def creds_file(tmp_path: Path) -> Path:
    return tmp_path / "xdg" / "throttle" / "credentials"


def run(capsys, *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# -- login / logout / whoami ------------------------------------------------


def test_login_verifies_key_and_saves_it_0600(env, tmp_path, capsys):
    code, out, err = run(capsys, "login", "--key", KEY)
    assert code == 0, err
    assert "Logged in to Throttle Cloud as dev@example.com (Example), plan free, $10.00 credits" in out
    path = creds_file(tmp_path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert f"api_key = {KEY}" in path.read_text()
    assert KEY not in out  # only a masked form is printed
    assert env.paths() == ["/api/v1/me"]
    assert env.requests[0].headers["authorization"] == f"Bearer {KEY}"
    assert env.requests[0].headers["user-agent"].startswith("throttle-cli/")


def test_login_with_a_rejected_key_saves_nothing(env, tmp_path, capsys):
    bad = "thr_live_" + "b" * 32
    code, out, err = run(capsys, "login", "--key", bad)
    assert code == 1
    assert "rejected the API key (401)" in err
    assert not creds_file(tmp_path).exists()


def test_login_rejects_a_key_that_is_not_a_throttle_key_without_a_request(env, tmp_path, capsys):
    code, out, err = run(capsys, "login", "--key", "sk-not-ours")
    assert code == 2
    assert "does not look like a Throttle API key" in err
    assert env.requests == []
    assert not creds_file(tmp_path).exists()


def test_login_reads_the_key_from_stdin_when_not_a_tty(env, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO(KEY + "\n"))
    code, out, err = run(capsys, "login")
    assert code == 0, err
    assert creds_file(tmp_path).exists()


def test_login_offline_says_so_and_saves_nothing(env, tmp_path, capsys):
    env.fail_first = 5
    code, out, err = run(capsys, "login", "--key", KEY)
    assert code == 1
    assert "Could not reach Throttle Cloud" in err and "nothing was uploaded or charged" in err.lower()
    assert len(env.requests) == 2  # one retry, then give up
    assert not creds_file(tmp_path).exists()


def test_overwriting_credentials_keeps_0600_even_from_a_loose_file(env, tmp_path, capsys):
    path = creds_file(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("api_key = old\n")
    os.chmod(path, 0o644)
    assert run(capsys, "login", "--key", KEY)[0] == 0
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert cloud.read_credentials_file(path) == KEY


def test_loose_credentials_file_warns(env, tmp_path, capsys):
    path = cloud.write_credentials(KEY)
    os.chmod(path, 0o644)
    code, out, err = run(capsys, "whoami")
    assert code == 0
    assert "readable by other users" in err and "chmod 600" in err


def test_whoami_shows_plan_and_credits(env, tmp_path, capsys):
    cloud.write_credentials(KEY)
    code, out, err = run(capsys, "whoami")
    assert code == 0, err
    assert "dev@example.com (Example), plan free, $10.00 credits" in out
    assert str(creds_file(tmp_path)) in out


def test_whoami_not_logged_in(env, capsys):
    code, out, err = run(capsys, "whoami")
    assert code == 1
    assert "Not logged in" in err and "https://www.throttle-pro.com/pricing" in err
    assert env.requests == []


def test_env_key_overrides_the_saved_key(env, tmp_path, capsys, monkeypatch):
    cloud.write_credentials("thr_live_" + "z" * 32)
    monkeypatch.setenv(cloud.API_KEY_ENV, KEY)
    code, out, err = run(capsys, "whoami")
    assert code == 0, err
    assert "from THROTTLE_API_KEY" in out
    assert env.requests[-1].headers["authorization"] == f"Bearer {KEY}"


def test_401_with_env_key_points_at_the_env_var(env, capsys, monkeypatch):
    monkeypatch.setenv(cloud.API_KEY_ENV, "thr_live_" + "q" * 32)
    code, out, err = run(capsys, "whoami")
    assert code == 1
    assert "THROTTLE_API_KEY" in err and "401" in err


def test_logout_removes_the_file(env, tmp_path, capsys, monkeypatch):
    cloud.write_credentials(KEY)
    code, out, _ = run(capsys, "logout")
    assert code == 0 and "Logged out" in out
    assert not creds_file(tmp_path).exists()
    code, out, _ = run(capsys, "logout")
    assert "Not logged in" in out
    monkeypatch.setenv(cloud.API_KEY_ENV, KEY)
    _, out, _ = run(capsys, "logout")
    assert "THROTTLE_API_KEY is still set" in out


def test_plain_http_cloud_url_is_refused_except_localhost(env, capsys, monkeypatch):
    cloud.write_credentials(KEY)
    monkeypatch.setenv(cloud.CLOUD_URL_ENV, "http://cloud.example.com")
    code, out, err = run(capsys, "whoami")
    assert code == 1
    assert "must use https" in err
    assert env.requests == []
    monkeypatch.setenv(cloud.CLOUD_URL_ENV, "http://localhost:3000")
    assert run(capsys, "whoami")[0] == 0
    assert str(env.requests[-1].url) == "http://localhost:3000/api/v1/me"


@pytest.mark.parametrize(
    "status,text",
    [
        (401, "rejected the API key (401)"),
        (402, "Out of Throttle Cloud credits ($10.00 left)"),
        (403, "plan does not include"),
        (422, "refused the upload (422)"),
        (500, "returned HTTP 500"),
    ],
)
def test_error_messages(env, status, text):
    env.forced["/api/v1/checks"] = status
    client = cloud.CloudClient(KEY)
    with pytest.raises(cloud.CloudError) as excinfo:
        client.push_check({"id": "x"}, idempotency_key="check:x")
    assert text in str(excinfo.value)
    if status == 402:
        assert "https://www.throttle-pro.com/pricing" in str(excinfo.value)
        assert "local feature keeps working" in str(excinfo.value)
    if status == 500:
        assert len(env.requests) == 2  # retried once with the same key
        assert env.requests[0].headers["idempotency-key"] == env.requests[1].headers["idempotency-key"]


def test_same_idempotency_key_never_charges_twice(env):
    client = cloud.CloudClient(KEY)
    first = client.push_check({"id": "a"}, idempotency_key="check:a")
    second = client.push_check({"id": "a"}, idempotency_key="check:a")
    assert first == second
    assert env.charges == ["/api/v1/checks"]
    assert env.credits() == "9.75"


def test_a_dropped_connection_is_retried_with_the_same_key(env):
    env.fail_first = 1
    body = cloud.CloudClient(KEY).push_check({"id": "a"}, idempotency_key="check:a")
    assert body["url"]
    assert [r.headers["idempotency-key"] for r in env.requests] == ["check:a", "check:a"]
    assert env.charges == ["/api/v1/checks"]


# -- throttle check --push ---------------------------------------------------


def sim_check(capsys, sim: SimulatedServer, *extra: str, config=("seqs=64",)) -> tuple[int, str, str]:
    sim.start(dict(pair.split("=", 1) for pair in config), "t")
    args = ["check", "--no-calibrate", "--url", URL, "--model", "m", "--gpu-hourly-rate", "2",
            "--blocks", "3", "--requests-per-block", "2", "--no-metrics"]
    for pair in config:
        args += ["--config", pair]
    with sim.measuring():
        return run(capsys, *args, *extra)


def test_check_without_push_never_contacts_the_cloud_even_when_logged_in(env, capsys):
    cloud.write_credentials(KEY)
    code, out, err = sim_check(capsys, SimulatedServer("m", 2))
    assert code == 0, err
    assert env.requests == []
    assert "Pushed" not in out


def test_check_push_uploads_prints_url_and_credits(env, tmp_path, capsys):
    cloud.write_credentials(KEY)
    json_path = tmp_path / "c.json"
    code, out, err = sim_check(capsys, SimulatedServer("m", 2), "--push", "--project", "prod",
                               "--json", str(json_path))
    assert code == 0, err
    assert "Pushed to Throttle Cloud: https://cloud.throttle-pro.com/c/chk_1 (charged $0.25, $9.75 credits left)" in out
    assert env.paths() == ["/api/v1/checks"]
    record = json.loads(json_path.read_text())
    assert env.requests[0].headers["idempotency-key"] == f"check:{record['id']}"
    payload = env.bodies[0]
    assert payload["id"] == record["id"] and payload["project"] == "prod"
    assert payload["usd_per_m"] == record["result"]["output"]["mean"]


def test_push_payload_has_no_url_host_prompt_or_secret(env, tmp_path, capsys, monkeypatch):
    cloud.write_credentials(KEY)
    prompts = tmp_path / "secret-prompts.jsonl"
    prompts.write_text(json.dumps({"messages": [{"role": "user", "content": "TOPSECRETPROMPT about payroll"}]}) + "\n")
    code, out, err = sim_check(
        capsys, SimulatedServer("m", 2), "--push", "--prompts", str(prompts),
        "--label", "run at gpu-box.internal.example",
        config=("seqs=64", "adapter=hf_abcdefghijklmnopqrstuvwxyz123456", "model_path=/home/kush/models/llama"),
    )
    assert code == 0, err
    text = json.dumps(env.bodies[0])
    for leaked in ("127.0.0.1", "http://", "TOPSECRETPROMPT", "payroll", "secret-prompts",
                   "hf_abcdefghijklmnopqrstuvwxyz123456", "/home/kush", "gpu-box.internal.example",
                   "simulated answer", KEY):
        assert leaked not in text, leaked
    payload = env.bodies[0]
    assert payload["workload"]["prompts_source"] == "custom"
    assert len(payload["endpoint_sha256"]) == 16
    assert payload["fingerprint"]["config"]["config.seqs"] == "64"
    assert payload["model"] == "m" and payload["engine"].startswith("vLLM")


def test_push_out_of_credits_keeps_the_check_and_its_exit_code(env, capsys):
    cloud.write_credentials(KEY)
    env.credits_micro = 0
    code, out, err = sim_check(capsys, SimulatedServer("m", 2), "--push")
    assert code == 0
    assert "Not pushed: Out of Throttle Cloud credits ($0.00 left)" in err
    assert "Saved as check" in out


def test_push_offline_is_a_warning(env, capsys):
    cloud.write_credentials(KEY)
    env.fail_first = 10
    code, out, err = sim_check(capsys, SimulatedServer("m", 2), "--push")
    assert code == 0
    assert "Not pushed: Could not reach Throttle Cloud" in err


def test_push_not_logged_in_is_a_warning(env, capsys):
    code, out, err = sim_check(capsys, SimulatedServer("m", 2), "--push")
    assert code == 0
    assert "Not pushed: not logged in" in err
    assert env.requests == []


def test_push_true_in_config_file_and_no_push_override(env, tmp_path, capsys, monkeypatch):
    pytest.importorskip("yaml")
    cloud.write_credentials(KEY)
    config = tmp_path / "config.yaml"
    config.write_text("push: true\n")
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", config)
    code, out, err = sim_check(capsys, SimulatedServer("m", 2))
    assert code == 0, err
    assert env.paths() == ["/api/v1/checks"]
    code, out, err = sim_check(capsys, SimulatedServer("m", 2), "--no-push")
    assert code == 0
    assert env.paths() == ["/api/v1/checks"]  # nothing new


def test_payload_from_a_record_is_stable_for_the_same_check(env, tmp_path, capsys):
    json_path = tmp_path / "c.json"
    assert sim_check(capsys, SimulatedServer("m", 2), "--json", str(json_path))[0] == 0
    record = json.loads(json_path.read_text())
    assert cloud.check_payload(record) == cloud.check_payload(record)
    assert "endpoint" not in cloud.check_payload(record)
