"""First-run friction fixes, tested the way a new user meets them.

Each test drives the real CLI entry point (``throttle.cli.main``) and asserts
on what the user sees: printed output, exit codes, and which URL / headers
actually reached a (mock) server. No real network traffic is sent.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

import throttle.config as config_module
from throttle import __version__
from throttle import cli as cli_module
from throttle.benchmark import run_native, validate_config
from throttle.cli import build_parser, main
from throttle.models import EndpointConfig

from test_benchmark import PROMPTS, WARMUPS, _completion, _run_config


# --------------------------------------------------------------------------
# A tiny OpenAI-compatible server behind httpx.MockTransport
# --------------------------------------------------------------------------


class FakeOpenAI:
    def __init__(
        self, models: list[str] | None = None, *, models_route: bool = True, delay: float = 0.0
    ):
        self.models = ["llama3.2:3b"] if models is None else models
        self.delay = delay
        self.models_route = models_route
        self.chat_paths: list[str] = []
        self.chat_models: list[str] = []
        self.chat_headers: list[dict[str, str]] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/models" and self.models_route:
            return httpx.Response(
                200, json={"object": "list", "data": [{"id": m} for m in self.models]}
            )
        if path == "/v1/chat/completions":
            body = json.loads(request.content)
            self.chat_paths.append(path)
            self.chat_models.append(body["model"])
            self.chat_headers.append(dict(request.headers))
            if body["model"] not in self.models:
                return httpx.Response(404, json={"error": "model not found"})
            if self.delay:
                import time

                time.sleep(self.delay)
            return httpx.Response(
                200,
                json={
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                },
            )
        return httpx.Response(404, text="404 page not found")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", tmp_path / "no-config.yaml")
    monkeypatch.chdir(tmp_path)


def install(monkeypatch, server: FakeOpenAI) -> None:
    real_client = httpx.Client
    real_async = httpx.AsyncClient

    def client(*args: Any, **kwargs: Any) -> httpx.Client:
        kwargs["transport"] = httpx.MockTransport(server.handle)
        return real_client(*args, **kwargs)

    def async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        async def handler(request: httpx.Request) -> httpx.Response:
            return server.handle(request)

        kwargs["transport"] = httpx.MockTransport(handler)
        return real_async(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", client)
    monkeypatch.setattr(httpx, "AsyncClient", async_client)


def run(capsys, *argv: str) -> tuple[int, str]:
    try:
        code = main(list(argv))
    except SystemExit as exc:
        code = int(exc.code or 0)
    captured = capsys.readouterr()
    return code, captured.out + captured.err


# --------------------------------------------------------------------------
# Rank 1: one URL form works everywhere
# --------------------------------------------------------------------------


def test_proxy_backend_url_with_v1_is_normalized(monkeypatch, capsys):
    import uvicorn

    import throttle.proxy as proxy_module

    seen: dict[str, Any] = {}

    def fake_create_app(**kwargs: Any) -> object:
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(proxy_module, "create_app", fake_create_app)
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
    code, out = run(capsys, "proxy", "--url", "http://localhost:11434/v1/")
    assert code == 0, out
    assert seen["backend_url"] == "http://localhost:11434"


# --------------------------------------------------------------------------
# Rank 3: --model default
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Rank 4/5: the first $/M number is one comparable figure, with its workload
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Rank 2: no key needed for a local endpoint; clear error otherwise
# --------------------------------------------------------------------------

SMOKE_LOCAL = (
    "smoke", "--model", "llama3.2:3b", "--url", "http://localhost:11434/v1",
    "--gpu-hourly-rate", "1.50", "--concurrency", "1", "--requests", "1",
    "--max-tokens", "8",
)


def test_local_smoke_runs_without_a_key(monkeypatch, capsys):
    seen: dict[str, Any] = {}

    async def fake_run_native(config, prompts, warmups, **kwargs):
        seen["api_key"] = config.endpoint.api_key
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_module, "run_native", fake_run_native)
    code, out = run(capsys, *SMOKE_LOCAL, "--output", "r.json")
    assert seen == {"api_key": ""}, out
    assert "API key: none sent (OPENAI_API_KEY is not set and the endpoint is local" in out


def test_remote_smoke_without_key_names_the_variable(capsys):
    code, out = run(
        capsys, "smoke", "--model", "m", "--url", "https://api.example.invalid/v1",
        "--gpu-hourly-rate", "1.50", "--output", "r.json",
    )
    assert code == 2
    assert "OPENAI_API_KEY is not set or is empty" in out
    assert "--api-key-env NAME" in out
    # Scoped to the subcommand the user ran, not the whole-program usage.
    assert out.startswith("throttle smoke: error:")
    assert "{plan,smoke" not in out


def test_explicit_api_key_env_that_is_unset_still_fails_locally(capsys):
    code, out = run(capsys, *SMOKE_LOCAL, "--api-key-env", "MY_LOCAL_KEY", "--output", "r.json")
    assert code == 2
    assert "MY_LOCAL_KEY is not set or is empty" in out


def test_plan_reports_key_status_for_local_and_remote(capsys):
    code, out = run(capsys, "plan", "--model", "m", "--url", "http://127.0.0.1:8000", "--gpu-hourly-rate", "1")
    assert code == 0, out
    assert "API key: none sent" in out
    code, out = run(capsys, "plan", "--model", "m", "--url", "https://api.example.invalid", "--gpu-hourly-rate", "1")
    assert code == 0, out
    assert "OPENAI_API_KEY is not set" in out
    assert "smoke/benchmark will refuse" in out


def test_keyless_run_sends_no_authorization_header_and_remote_needs_a_key():
    local = replace_endpoint(_run_config(), "http://127.0.0.1:8000/v1", "")
    validate_config(local, for_traffic=True)
    headers: list[httpx.Headers] = []

    def handler(request: httpx.Request) -> httpx.Response:
        headers.append(request.headers)
        return _completion()

    report = asyncio.run(
        run_native(local, PROMPTS, WARMUPS, transport=httpx.MockTransport(handler))
    )
    assert headers and all("authorization" not in h for h in headers)
    assert report["status"] != "failed"

    remote = replace_endpoint(_run_config(), "https://offline-only.example/v1", "")
    with pytest.raises(ValueError, match="API key must not be empty"):
        validate_config(remote, for_traffic=True)


def replace_endpoint(config, url: str, key: str):
    from dataclasses import replace

    return replace(config, endpoint=EndpointConfig(url, key))


# --------------------------------------------------------------------------
# Rank 7/8: one set of flag names; the price gate says how to get a price
# --------------------------------------------------------------------------


def test_plan_accepts_gpu_hourly_rate_and_infers_dedicated_hourly(capsys):
    code, out = run(
        capsys, "plan", "--model", "m", "--endpoint-url", "http://localhost:11434/v1",
        "--gpu-hourly-rate", "1.50",
    )
    assert code == 0, out
    assert "Cost model: dedicated-hourly (inferred" in out
    assert "cost acknowledgement/ceiling check passed" in out
    assert "Spend ceiling: $3.00 USD" in out


def test_price_without_matching_cost_model_names_the_model(capsys):
    code, out = run(
        capsys, "plan", "--model", "m", "--url", "http://localhost:1",
        "--cost-model", "unknown", "--total-hourly-price", "1.50",
    )
    assert code == 2
    assert "--total-hourly-price requires --cost-model dedicated-hourly" in out


def test_unpriced_plan_offers_the_priced_option_first(capsys):
    code, out = run(capsys, "plan", "--model", "m", "--url", "http://localhost:1")
    assert code == 0
    blocked = out[out.index("Traffic preflight: blocked"):]
    assert blocked.index("--gpu-hourly-rate") < blocked.index("--allow-unknown-cost to run")


def test_unpriced_smoke_is_still_refused_and_says_how_to_price(capsys):
    code, out = run(
        capsys, "smoke", "--model", "m", "--url", "http://localhost:1", "--output", "r.json"
    )
    assert code == 2
    assert "cannot enforce max spend" in out
    assert "--gpu-hourly-rate 1.50" in out
    assert not Path("r.json").exists()


@pytest.mark.parametrize(
    ("argv", "dest", "expected"),
    [
        (["check", "--endpoint-url", "http://h:1", "--gpu-rate-per-hour", "2"], "url", "http://h:1"),
        (["check", "--url", "http://h:1", "--total-hourly-price", "2"], "gpu_hourly_rate", 2.0),
        (["watch", "--gpu-hourly-rate", "2"], "gpu_rate_per_hour", 2.0),
        (["proxy", "--endpoint-url", "http://h:1"], "backend_url", "http://h:1"),
    ],
)
def test_flag_aliases_parse_to_the_same_destination(argv, dest, expected):
    args = build_parser().parse_args(argv)
    assert getattr(args, dest) == expected


def test_old_flag_names_still_work():
    parser = build_parser()
    assert parser.parse_args(["watch", "--gpu-rate-per-hour", "1"]).gpu_rate_per_hour == 1.0
    assert parser.parse_args(["proxy", "--backend-url", "u"]).backend_url == "u"
    assert parser.parse_args(["check", "--url", "u"]).url == "u"


# --------------------------------------------------------------------------
# Rank 17: `throttle` with no arguments
# --------------------------------------------------------------------------


def test_no_args_prints_three_command_start_and_exits_zero(capsys, monkeypatch):
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    code, out = run(capsys)
    assert code == 0
    lines = out.rstrip("\n").splitlines()
    assert len(lines) <= 20
    assert "Throttle " + __version__ in out
    assert "Throttle cuts what your LLM inference costs on your own GPUs." in out
    assert "1. Install Ollama" in out and "vLLM / SGLang" in out
    # One path, no promise of a verdict from the first run: run once to
    # calibrate, change one setting, run again.
    assert "2. Run once to calibrate" in out and "throttle check --url http://localhost:11434" in out
    assert "3. Change one setting, then run it again for a verdict." in out
    assert "and a verdict:" not in out
    assert "4. For agent traffic" in out and "throttle check --workload agent" in out
    assert "No server yet? throttle check --help" in out
    for retired in ("Throttle Cloud", "credits", "throttle upgrade", "throttle demo",
                    "3 months", "2 weeks", "20%", "$500", "floor"):
        assert retired not in out, retired
    assert "\x1b[" not in out
    # Every command it tells the user to run must parse.
    parser = build_parser()
    parser.parse_args(["check", "--url", "http://localhost:11434", "--model", "llama3.2:3b", "--gpu-hourly-rate", "1.50"])
    parser.parse_args(["check", "--workload", "agent", "--url", "http://localhost:11434", "--model", "llama3.2:3b", "--gpu-hourly-rate", "1.50"])


def test_no_args_screen_with_force_color_and_ascii(capsys, monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    code, out = run(capsys)
    assert code == 0
    assert "\x1b[" in out
    from throttle.style import strip_ansi
    plain = strip_ansi(out)
    assert "Throttle cuts what your LLM inference costs on your own GPUs." in plain
    assert "throttle check --workload agent" in plain


def test_help_lists_no_removed_command(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert "==SUPPRESS==" not in out
    assert "Get your first $/M-token number" in out
    for removed in ("validate-sim", "experimental-tuning", "diagnose", "upgrade", "login",
                    "logout", "whoami", "throttle cost", "throttle measure", "Throttle Cloud",
                    "credits", "guidellm"):
        assert removed not in out, removed
    choices = build_parser()._throttle_subparsers  # type: ignore[attr-defined]
    for removed in ("validate-sim", "experimental-tuning", "diagnose", "upgrade", "login",
                    "logout", "whoami", "cost", "measure"):
        assert removed not in choices, removed


def test_gpu_hourly_rate_with_several_gpus_is_refused_as_ambiguous(capsys):
    code, out = run(
        capsys, "plan", "--model", "m", "--url", "http://localhost:11434",
        "--gpus", "4", "--gpu-hourly-rate", "2.0",
    )
    assert code == 2
    assert "--gpu-hourly-rate with --gpus 4 is ambiguous" in out
    assert "--per-gpu-hourly-price 2 --gpus 4" in out


def test_plan_prints_the_hourly_price_it_used(capsys):
    code, out = run(
        capsys, "plan", "--model", "m", "--url", "http://localhost:11434",
        "--gpus", "4", "--per-gpu-hourly-price", "2.0",
    )
    assert code == 0, out
    assert "Hourly price used: $8.00/hr for 4 GPU(s) ($2/hr per GPU x 4) [ASSUMED]" in out
    assert "Estimated cost upper bound: $0.266667 USD" in out
    code, out = run(
        capsys, "plan", "--model", "m", "--url", "http://localhost:11434",
        "--gpus", "4", "--total-hourly-price", "8.0",
    )
    assert code == 0, out
    assert "Hourly price used: $8.00/hr for 4 GPU(s) (total for all GPUs) [ASSUMED]" in out


def test_plan_does_not_claim_cuda_unless_declared(capsys):
    code, out = run(capsys, "plan", "--model", "m", "--url", "http://localhost:11434",
                    "--gpu-hourly-rate", "1.5")
    assert code == 0, out
    assert "Runtime: unknown (not declared)" in out
    code, out = run(capsys, "plan", "--model", "m", "--url", "http://localhost:11434",
                    "--gpu-hourly-rate", "1.5", "--accelerator-backend", "metal")
    assert code == 0, out
    assert "Runtime: metal /" in out
