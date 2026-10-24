"""`throttle coldstart`: launches tests/fake_openai_server.py (a real stdlib
HTTP server on a free loopback port, no GPU), times ready / first token, and
always kills the whole process group."""

from __future__ import annotations

import json
import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

import throttle.coldstart as coldstart
import throttle.config as config_module
from throttle.cli import main

FAKE = Path(__file__).with_name("fake_openai_server.py")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def server_cmd(port: int, *extra: str) -> str:
    return " ".join(shlex.quote(p) for p in (sys.executable, str(FAKE), str(port), *extra))


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def pids(path: Path) -> list[int]:
    return [int(line) for line in path.read_text().split()] if path.exists() else []


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", tmp_path / "no-config.yaml")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(coldstart, "POLL_SECONDS", 0.05)


def test_times_ready_and_first_token_and_kills_server(tmp_path, capsys):
    port = free_port()
    pidfile = tmp_path / "pids"
    out = tmp_path / "cold.json"
    code = main([
        "coldstart", "--cmd", server_cmd(port, "--delay", "0.3", "--pidfile", str(pidfile)),
        "--url", f"http://127.0.0.1:{port}", "--model", "m", "--runs", "2",
        "--log-dir", str(tmp_path / "logs"), "--json", str(out),
    ])
    text = capsys.readouterr().out
    assert code == coldstart.EXIT_OK, text
    assert "server run 1: ready" in text and "server run 2: ready" in text
    assert "2/2" in text
    record = json.loads(out.read_text())
    assert len(record["runs"]) == 2
    for run in record["runs"]:
        assert run["error"] is None
        assert run["ready_s"] >= 0.3
        assert run["first_token_s"] >= run["ready_s"]
        assert run["ready_via"] == "/health"
    summary = record["summary"][0]
    assert summary["ok"] == 2 and summary["ready_s"]["n"] == 2
    assert len(pids(pidfile)) == 2
    assert not any(alive(pid) for pid in pids(pidfile))


def test_compares_labelled_commands(tmp_path, capsys):
    port = free_port()
    code = main([
        "coldstart", "--url", f"http://127.0.0.1:{port}", "--model", "m", "--runs", "1",
        "--cmd", server_cmd(port, "--delay", "0.05"), "--label", "fast",
        "--cmd", server_cmd(port, "--delay", "0.8"), "--label", "slow",
        "--log-dir", str(tmp_path / "logs"),
    ])
    text = capsys.readouterr().out
    assert code == coldstart.EXIT_OK, text
    table = text[text.index("command"):]
    assert "fast" in table and "slow" in table and "first token" in table
    assert "fast reaches its first token" in text


def test_kills_the_whole_group_even_if_sigterm_is_ignored(tmp_path, capsys):
    port = free_port()
    pidfile = tmp_path / "pids"
    # The server is a grandchild of the shell, and ignores SIGTERM.
    cmd = server_cmd(port, "--pidfile", str(pidfile), "--ignore-term") + " & wait"
    code = main([
        "coldstart", "--cmd", cmd, "--url", f"http://127.0.0.1:{port}", "--model", "m",
        "--runs", "1", "--stop-timeout", "1", "--log-dir", str(tmp_path / "logs"),
    ])
    capsys.readouterr()
    assert code == coldstart.EXIT_OK
    (pid,) = pids(pidfile)
    deadline = time.monotonic() + 5
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not alive(pid)


def test_ctrl_c_stops_the_server(tmp_path, capsys, monkeypatch):
    port = free_port()
    pidfile = tmp_path / "pids"

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(coldstart, "first_token", interrupt)
    code = main([
        "coldstart", "--cmd", server_cmd(port, "--pidfile", str(pidfile)),
        "--url", f"http://127.0.0.1:{port}", "--model", "m", "--log-dir", str(tmp_path / "logs"),
    ])
    assert code == coldstart.EXIT_INTERRUPTED
    assert "Interrupted" in capsys.readouterr().err
    assert not any(alive(pid) for pid in pids(pidfile))


def test_server_that_exits_is_reported(tmp_path, capsys):
    port = free_port()
    code = main([
        "coldstart", "--cmd", "echo loading; exit 3", "--url", f"http://127.0.0.1:{port}",
        "--model", "m", "--runs", "1", "--log-dir", str(tmp_path / "logs"),
    ])
    text = capsys.readouterr().out
    assert code == coldstart.EXIT_FAILED
    assert "exited with code 3" in text and "loading" in text
    assert "0/1" in text


def test_refuses_when_something_already_answers(tmp_path, capsys):
    port = free_port()
    proc = subprocess.Popen([sys.executable, str(FAKE), str(port)])
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.05)
        code = main(["coldstart", "--cmd", "true", "--url", f"http://127.0.0.1:{port}",
                     "--model", "m", "--log-dir", str(tmp_path / "logs")])
        assert code == coldstart.EXIT_USAGE
        assert "already answering" in capsys.readouterr().err
    finally:
        proc.kill()
        proc.wait()


def test_label_count_must_match(tmp_path, capsys):
    code = main(["coldstart", "--cmd", "a", "--cmd", "b", "--label", "x",
                 "--url", "http://127.0.0.1:9", "--model", "m"])
    assert code == coldstart.EXIT_USAGE
    assert "one --label per --cmd" in capsys.readouterr().err


def test_second_interrupt_while_stopping_does_not_leak(tmp_path):
    """A Ctrl-C during the SIGTERM wait must not lose the server: the next
    stop() goes straight to SIGKILL."""

    port = free_port()
    pidfile = tmp_path / "pids"
    server = coldstart.Server(server_cmd(port, "--pidfile", str(pidfile), "--ignore-term"),
                              tmp_path / "log", stop_timeout=30)
    server.start()
    deadline = time.monotonic() + 5
    while not pids(pidfile) and time.monotonic() < deadline:
        time.sleep(0.05)
    process = server.process
    real_wait = process.wait
    calls = {"n": 0}

    def wait(timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyboardInterrupt
        return real_wait(timeout=timeout)

    process.wait = wait
    with pytest.raises(KeyboardInterrupt):
        server.stop()
    assert server.process is process  # still tracked
    started = time.monotonic()
    server.stop()
    assert time.monotonic() - started < 10 and server.process is None
    (pid,) = pids(pidfile)
    deadline = time.monotonic() + 5
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not alive(pid)


def test_rejected_chat_request_fails_fast():
    import httpx

    transport = httpx.MockTransport(lambda r: httpx.Response(404, text="model not found"))
    with httpx.Client(transport=transport) as client:
        with pytest.raises(coldstart.ColdstartError, match="HTTP 404"):
            coldstart.first_token(client, "http://127.0.0.1:9/v1/chat/completions", {}, "nope")
