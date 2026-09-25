"""Regression tests for the two bugs waqasm86 hit running validate-sim on vLLM (#22).

1. The Heavy workload stopped at request 86 with a bare "failed with status
   400". vLLM had rejected it because prompt + max_tokens exceeded
   --max-model-len 2048. The error must name the limit and say how to fit it.
2. At shutdown, asyncio printed a traceback ("ValueError: second argument
   (exceptions) must be a non-empty sequence") from anyio's connect_tcp. The
   failing request left its siblings running; asyncio.run cancelled them
   mid-connect during shutdown and reported what they raised. Now the other
   requests are never cancelled: unsent ones are skipped, sent ones finish.

Every test drives the real CLI (``throttle.cli.main``) over real HTTP against a
stdlib server that answers like vLLM, and asserts on what the user sees.
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator

import httpx
import pytest

import throttle.config as config_module
import throttle.check as check_module
from throttle.cli import main
from throttle.request_errors import context_overflow_limit, describe_http_failure
from throttle.workload import WorkloadGenerator

# Qwen2.5 tokenizes "Test Test ..." at about 0.855 tokens per word: validate-sim's
# Heavy request 86 (2281 words) came to 1951 prompt tokens in waqasm86's log.
TOKENS_PER_WORD = 0.855
# What anyio's connect_tcp raised when cancelled mid-connect at shutdown (#22).
ANYIO_SHUTDOWN_ERROR = "second argument (exceptions) must be a non-empty sequence"


class FakeVLLM:
    """OpenAI-compatible server that enforces max_model_len the way vLLM does."""

    def __init__(self, max_model_len: int, latency: float) -> None:
        self.max_model_len = max_model_len
        self.latency = latency
        self.rejected: list[int] = []
        self.chat_posts = 0  # POSTs with max_tokens > 1 (connectivity probes excluded)
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: object) -> None:
                pass

            def _send(self, status: int, payload: object) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if self.path == "/v1/models":
                    self._send(200, {"object": "list", "data": [
                        {"id": "m", "object": "model", "max_model_len": fake.max_model_len},
                    ]})
                else:
                    self._send(404, {"detail": "Not Found"})

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                words = sum(len(m["content"].split()) for m in body["messages"])
                prompt_tokens = int(words * TOKENS_PER_WORD) + 1
                max_tokens = body.get("max_tokens", 16)
                total = prompt_tokens + max_tokens
                if max_tokens > 1:
                    fake.chat_posts += 1
                if total > fake.max_model_len:
                    fake.rejected.append(words)
                    # vLLM's VLLMValidationError, as served to the client.
                    self._send(400, {"error": {
                        "message": (
                            f"This model's maximum context length is {fake.max_model_len} "
                            f"tokens. However, you requested {max_tokens} output tokens and "
                            f"your prompt contains at least {prompt_tokens} input tokens, for "
                            f"a total of at least {total} tokens. Please reduce the length of "
                            "the input prompt or the number of requested output tokens. "
                            f"(parameter=input_tokens, value={prompt_tokens})"
                        ),
                        "type": "BadRequestError",
                        "param": "input_tokens",
                        "code": 400,
                    }})
                    return
                if max_tokens > 1:  # connectivity probes answer at once
                    time.sleep(fake.latency)
                self._send(200, {
                    "id": "chatcmpl-fake",
                    "object": "chat.completion",
                    "model": "m",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "length",
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": max_tokens,
                        "total_tokens": total,
                    },
                })

        class Server(ThreadingHTTPServer):
            daemon_threads = True
            request_queue_size = 256  # validate-sim opens up to 100 connections at once

            def handle_error(self, request, client_address) -> None:
                pass  # a client that cancels mid-request resets its connection

        self.server = Server(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "FakeVLLM":
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)  # validate-sim / measure write result files here
    monkeypatch.setenv("THROTTLE_ENABLE_EXPERIMENTAL", "1")
    monkeypatch.setenv(check_module.HISTORY_ENV, str(tmp_path / "history"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", tmp_path / "no-config.yaml")


@pytest.fixture
def fast_arrivals(monkeypatch) -> None:
    """Same seeded requests (sizes, order), arrival times compressed 50x."""

    original = WorkloadGenerator.generate_chat_workload

    def compressed(self, *args, **kwargs):
        return [(t / 50, p, m) for t, p, m in original(self, *args, **kwargs)]

    monkeypatch.setattr(WorkloadGenerator, "generate_chat_workload", compressed)


@pytest.fixture
def anyio_cancel_race(monkeypatch) -> list[int]:
    """Make a request cancelled mid-flight raise what anyio raised in #22.

    anyio's connect_tcp can lose a cancellation that lands during its
    happy-eyeballs task group and then raise ValueError instead (or leak the
    socket). The race is timing-dependent, so this reproduces its outcome
    deterministically. Returns a list of the requests cancelled mid-flight.
    """

    fired: list[int] = []
    original = httpx.AsyncHTTPTransport.handle_async_request

    async def racy(self, request):
        try:
            return await original(self, request)
        except asyncio.CancelledError:
            fired.append(1)
            raise ValueError(ANYIO_SHUTDOWN_ERROR) from None

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", racy)
    return fired


def assert_no_shutdown_noise(output: str, caplog) -> None:
    gc.collect()  # a task whose exception nobody retrieved is reported when collected
    shutdown_errors = [
        r.getMessage() for r in caplog.records
        if r.name == "asyncio" and r.levelno >= logging.ERROR
    ]
    assert shutdown_errors == [], f"asyncio reported leftover tasks: {shutdown_errors}"
    assert "Traceback" not in output
    assert ANYIO_SHUTDOWN_ERROR not in output
    assert "was never awaited" not in output


# --- Bug 1: HTTP 400 from context overflow -------------------------------------


def test_validate_sim_heavy_request_86_names_the_context_limit(capsys, fast_arrivals):
    """waqasm86's run: --max-model-len 2048, Heavy load stops at request 86."""

    with FakeVLLM(max_model_len=2048, latency=0.05) as server:
        code = main([
            "validate-sim", "--endpoint-url", server.url, "--model", "m",
            "--gpu-hourly-rate", "1.00",
        ])
    out = capsys.readouterr().out

    assert code == 1
    assert server.rejected == [2281]  # the same single request vLLM rejected
    assert "Running Light load" in out and "Running Medium load" in out
    assert "failed with status 400" not in out
    assert (
        "Error: Heavy load request 86 was rejected with HTTP 400: its prompt plus "
        "max_tokens does not fit the model's 2048-token context window (max_model_len). "
        "It sent a 2281-word prompt and max_tokens 98."
    ) in out
    assert "Server said: This model's maximum context length is 2048 tokens." in out
    assert "Start the server with a larger --max-model-len (4096 should fit it)" in out
    assert "throttle check --prompts FILE --max-tokens N" in out


def test_validate_sim_completes_when_the_workload_fits(capsys, fast_arrivals):
    with FakeVLLM(max_model_len=4096, latency=0.05) as server:
        code = main([
            "validate-sim", "--endpoint-url", server.url, "--model", "m",
            "--gpu-hourly-rate", "1.00",
        ])
    out = capsys.readouterr().out

    assert code == 0, out
    assert server.rejected == []
    assert "Running Heavy load" in out and "Full results written to" in out


def test_measure_names_the_context_limit(capsys):
    with FakeVLLM(max_model_len=800, latency=0.05) as server:
        code = main([
            "measure", "--endpoint-url", server.url, "--model", "m",
            "--gpu-hourly-rate", "1.00", "--label", "t4", "--repeat", "1",
            "--num-requests", "20", "--arrival-rate", "200",
        ])
    out = capsys.readouterr().out

    assert code == 1
    assert "Trial 1 request 6 was rejected with HTTP 400" in out
    assert "800-token context window (max_model_len)" in out
    assert "measure sends a fixed synthetic workload" in out
    assert "--max-model-len" in out
    assert not list(Path.cwd().glob("*.json"))  # nothing recorded


def test_cost_names_the_context_limit(capsys):
    with FakeVLLM(max_model_len=256, latency=0.0) as server:
        code = main([
            "cost", "--endpoint-url", server.url, "--model", "m",
            "--gpu-hourly-rate", "1.00", "--num-requests", "5",
        ])
    out = capsys.readouterr().out

    assert code == 1
    assert "Error: Request 4 was rejected with HTTP 400" in out
    assert "256-token context window (max_model_len)" in out
    assert "cost sends a fixed synthetic workload" in out


def test_check_suggests_lowering_max_tokens(capsys):
    with FakeVLLM(max_model_len=2048, latency=0.0) as server:
        code = main([
            "check", "--url", server.url, "--model", "m", "--gpu-hourly-rate", "1",
            "--max-tokens", "2040", "--max-tokens-per-request", "4096",
            "--no-metrics", "--no-save", "--blocks", "3", "--requests-per-block", "2",
            "--concurrency", "1", "--warmup", "0",
        ])
    err = capsys.readouterr().err

    assert code == 1
    assert "2048-token context window (max_model_len). It sent max_tokens 2040." in err
    assert "Fix: lower --max-tokens (now 2040) or pass --prompts with shorter prompts" in err


@pytest.mark.parametrize(
    ("body", "limit"),
    [
        # vLLM (and OpenAI's wording)
        ({"object": "error", "message": "This model's maximum context length is 2048 "
          "tokens. However, you requested 2049 tokens.", "code": 400}, 2048),
        # SGLang
        ({"error": {"message": "The input (5000 tokens) is longer than the model's "
          "context length (4096 tokens)."}}, 4096),
        # TGI
        ({"error": "`inputs` tokens + `max_new_tokens` must be <= 8192. Given: 8000 "
          "`inputs` tokens and 500 `max_new_tokens`"}, 8192),
        # OpenAI error code without a number
        ({"error": {"message": "Too long.", "code": "context_length_exceeded"}}, None),
    ],
)
def test_context_overflow_is_recognised_across_servers(body, limit):
    text = json.dumps(body)
    assert context_overflow_limit(400, text) == (True, limit)


def test_other_http_errors_keep_the_server_message():
    body = json.dumps({"error": {"message": "The model `x` does not exist."}})
    assert context_overflow_limit(404, body) == (False, None)
    message = describe_http_failure(
        what="Request 3", url="http://h/v1/chat/completions", status=404, body=body,
        remedy="unused",
    )
    assert message == (
        "Request 3 failed: POST http://h/v1/chat/completions returned HTTP 404: "
        "The model `x` does not exist."
    )


# --- Bug 2: traceback at shutdown ----------------------------------------------


def test_validate_sim_failure_cancels_in_flight_requests_cleanly(
    capsys, caplog, fast_arrivals, anyio_cancel_race
):
    """Requests still in flight when #86 fails must not surface at shutdown."""

    caplog.set_level(logging.ERROR, logger="asyncio")
    with FakeVLLM(max_model_len=2048, latency=1.0) as server:
        code = main([
            "validate-sim", "--endpoint-url", server.url, "--model", "m",
            "--gpu-hourly-rate", "1.00",
        ])
    captured = capsys.readouterr()

    assert code == 1
    assert anyio_cancel_race == [], "a request was cancelled mid-flight"
    assert "Heavy load request 86 was rejected with HTTP 400" in captured.out
    assert_no_shutdown_noise(captured.out + captured.err, caplog)


def test_measure_failure_cancels_in_flight_requests_cleanly(
    capsys, caplog, anyio_cancel_race
):
    caplog.set_level(logging.ERROR, logger="asyncio")
    with FakeVLLM(max_model_len=800, latency=1.0) as server:
        code = main([
            "measure", "--endpoint-url", server.url, "--model", "m",
            "--gpu-hourly-rate", "1.00", "--label", "t4", "--repeat", "1",
            "--num-requests", "20", "--arrival-rate", "200",
        ])
    captured = capsys.readouterr()

    assert code == 1
    assert anyio_cancel_race == [], "a request was cancelled mid-flight"
    assert "request 6 was rejected with HTTP 400" in captured.out
    assert_no_shutdown_noise(captured.out + captured.err, caplog)


def test_check_failure_cancels_in_flight_requests_cleanly(
    capsys, caplog, tmp_path, anyio_cancel_race
):
    """One over-long prompt fails while a short one is still being served."""

    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        json.dumps({"prompt": "Test " * 3000}) + "\n"
        + json.dumps({"prompt": "short question"}) + "\n"
    )
    caplog.set_level(logging.ERROR, logger="asyncio")
    with FakeVLLM(max_model_len=2048, latency=1.0) as server:
        code = main([
            "check", "--url", server.url, "--model", "m", "--gpu-hourly-rate", "1",
            "--prompts", str(prompts), "--max-tokens", "16", "--no-metrics", "--no-save",
            "--blocks", "3", "--requests-per-block", "2", "--concurrency", "2",
            "--warmup", "0", "--warm-cache",
        ])
    captured = capsys.readouterr()

    assert code == 1
    assert anyio_cancel_race == [], "the short request was cancelled mid-flight"
    assert "context window" in captured.err
    assert_no_shutdown_noise(captured.out + captured.err, caplog)


def test_measure_sends_nothing_after_a_request_fails(capsys):
    """Requests 7-20 are due after request 6 fails, so none of them is sent."""

    with FakeVLLM(max_model_len=800, latency=0.05) as server:
        code = main([
            "measure", "--endpoint-url", server.url, "--model", "m",
            "--gpu-hourly-rate", "1.00", "--label", "t4", "--repeat", "1",
            "--num-requests", "20", "--arrival-rate", "20",
        ])

    assert code == 1
    assert "request 6 was rejected" in capsys.readouterr().out
    assert server.chat_posts == 6


def test_check_sends_nothing_after_a_request_fails(capsys, tmp_path):
    """Concurrency 1: the block's second request is queued when the first fails."""

    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        json.dumps({"prompt": "Test " * 3000}) + "\n"
        + json.dumps({"prompt": "Test " * 2500}) + "\n"
    )
    with FakeVLLM(max_model_len=2048, latency=0.0) as server:
        code = main([
            "check", "--url", server.url, "--model", "m", "--gpu-hourly-rate", "1",
            "--prompts", str(prompts), "--max-tokens", "16", "--no-metrics", "--no-save",
            "--blocks", "3", "--requests-per-block", "2", "--concurrency", "1",
            "--warmup", "0", "--warm-cache",
        ])

    assert code == 1
    assert "context window" in capsys.readouterr().err
    assert server.chat_posts == 1
