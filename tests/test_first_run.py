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


COST = ("cost", "--gpu-hourly-rate", "1.50", "--num-requests", "2")


# --------------------------------------------------------------------------
# Rank 1: one URL form works everywhere
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url", ["http://localhost:11434", "http://localhost:11434/v1", "http://localhost:11434/v1/"]
)
def test_cost_accepts_url_with_or_without_v1(monkeypatch, capsys, url):
    server = FakeOpenAI()
    install(monkeypatch, server)
    code, out = run(capsys, *COST, "--endpoint-url", url, "--model", "llama3.2:3b")
    assert code == 0, out
    assert server.chat_paths == ["/v1/chat/completions"] * 2


@pytest.mark.parametrize("url", ["http://localhost:11434", "http://localhost:11434/v1"])
def test_measure_accepts_url_with_or_without_v1(monkeypatch, capsys, url):
    server = FakeOpenAI()
    install(monkeypatch, server)
    code, out = run(
        capsys, "measure", "--url", url, "--model", "llama3.2:3b",
        "--gpu-hourly-rate", "1.50", "--label", "base", "--repeat", "2",
        "--num-requests", "2", "--arrival-rate", "50",
    )
    assert code == 0, out
    assert "Connection successful." in out
    assert set(server.chat_paths) == {"/v1/chat/completions"}
    assert "throttle compare base.json <other-label>.json" in out
    assert Path("base.json").exists()


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


def test_wrong_url_404_names_the_url_and_both_forms(monkeypatch, capsys):
    install(monkeypatch, FakeOpenAI())
    code, out = run(capsys, *COST, "--url", "http://localhost:11434/wrong", "--model", "llama3.2:3b")
    assert code == 1
    assert "POST http://localhost:11434/wrong/chat/completions returned HTTP 404" in out
    assert "both are accepted" in out


# --------------------------------------------------------------------------
# Rank 3: --model default
# --------------------------------------------------------------------------


def test_cost_without_model_uses_the_only_served_model(monkeypatch, capsys):
    server = FakeOpenAI(["llama3.2:3b"])
    install(monkeypatch, server)
    code, out = run(capsys, *COST, "--url", "http://localhost:11434")
    assert code == 0, out
    assert "Model: llama3.2:3b (the only model this server lists" in out
    assert server.chat_models == ["llama3.2:3b", "llama3.2:3b"]


def test_cost_without_model_and_several_served_lists_them(monkeypatch, capsys):
    server = FakeOpenAI(["a-model", "b-model"])
    install(monkeypatch, server)
    code, out = run(capsys, *COST, "--url", "http://localhost:11434")
    assert code == 2
    assert "--model is required; this server lists: a-model, b-model" in out
    assert server.chat_paths == []


def test_wrong_model_404_lists_available_models(monkeypatch, capsys):
    install(monkeypatch, FakeOpenAI(["llama3.2:3b", "llama3.2:1b"]))
    code, out = run(capsys, *COST, "--url", "http://localhost:11434", "--model", "nope")
    assert code == 1
    assert "Model 'nope' is not served here" in out
    assert "llama3.2:3b, llama3.2:1b" in out
    assert "Re-run with --model llama3.2:3b" in out


# --------------------------------------------------------------------------
# Rank 4/5: the first $/M number is one comparable figure, with its workload
# --------------------------------------------------------------------------


def test_cost_prints_blended_figure_caveat_workload_and_next_step(monkeypatch, capsys):
    install(monkeypatch, FakeOpenAI(delay=0.05))
    code, out = run(
        capsys, "cost", "--gpu-hourly-rate", "3600", "--num-requests", "2",
        "--url", "http://localhost:11434/v1", "--model", "llama3.2:3b",
    )
    assert code == 0, out

    def dollars(label: str) -> float:
        line = next(line for line in out.splitlines() if label in line)
        return float(line.split("$")[1].split()[0])

    blended_line = next(line for line in out.splitlines() if "Blended cost: $" in line)
    # 2 requests x (100 + 50) tokens, taken from the endpoint's usage field.
    assert "300 tokens" in blended_line
    blended, per_in, per_out = dollars("Blended cost: $"), dollars("Input cost: $"), dollars("Output cost: $")
    # Same measured spend C over I, O and I+O tokens: 1/blended = 1/in + 1/out.
    assert blended > 0
    assert abs(1 / blended - (1 / per_in + 1 / per_out)) * blended < 0.02
    assert "not additive" in out
    assert "ASSUMED $3600.00/hr" in out
    assert "one at a time (concurrency 1)" in out
    assert "throttle check --url http://localhost:11434/v1 --model llama3.2:3b" in out


def test_cost_refuses_when_endpoint_reports_no_usage(monkeypatch, capsys):
    server = FakeOpenAI()
    original = server.handle

    def no_usage(request: httpx.Request) -> httpx.Response:
        response = original(request)
        if request.url.path == "/v1/chat/completions" and response.status_code == 200:
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
        return response

    server.handle = no_usage  # type: ignore[method-assign]
    install(monkeypatch, server)
    code, out = run(capsys, *COST, "--url", "http://localhost:11434", "--model", "llama3.2:3b")
    assert code == 1
    assert "did not report token usage" in out
    assert "Blended cost" not in out


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
        (["cost", "--url", "http://h:1", "--gpu-rate-per-hour", "2"], "endpoint_url", "http://h:1"),
        (["measure", "--url", "http://h:1", "--gpu-hourly-rate", "2", "--label", "x"], "endpoint_url", "http://h:1"),
        (["proxy", "--endpoint-url", "http://h:1"], "backend_url", "http://h:1"),
    ],
)
def test_flag_aliases_parse_to_the_same_destination(argv, dest, expected):
    args = build_parser().parse_args(argv)
    assert getattr(args, dest) == expected


def test_old_flag_names_still_work():
    parser = build_parser()
    assert parser.parse_args(["cost", "--endpoint-url", "u", "--gpu-hourly-rate", "1"]).endpoint_url == "u"
    assert parser.parse_args(["watch", "--gpu-rate-per-hour", "1"]).gpu_rate_per_hour == 1.0
    assert parser.parse_args(["proxy", "--backend-url", "u"]).backend_url == "u"
    assert parser.parse_args(["check", "--url", "u"]).url == "u"


# --------------------------------------------------------------------------
# Rank 17: `throttle` with no arguments
# --------------------------------------------------------------------------


def test_no_args_prints_three_command_start_and_exits_zero(capsys):
    code, out = run(capsys)
    assert code == 0
    assert "Get your first $/M-token number (3 commands)" in out
    assert "throttle demo" in out
    assert "throttle cost --url http://localhost:11434" in out
    assert "throttle check --url" in out
    # Every command it tells the user to run must parse.
    parser = build_parser()
    parser.parse_args(["cost", "--url", "http://localhost:11434", "--model", "llama3.2:3b", "--gpu-hourly-rate", "1.50"])
    parser.parse_args(["check", "--url", "http://localhost:11434", "--model", "llama3.2:3b", "--gpu-hourly-rate", "1.50", "--label", "baseline"])


def test_help_does_not_leak_suppressed_command_but_it_still_parses(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert "==SUPPRESS==" not in out
    assert "validate-sim" not in out
    assert "Get your first $/M-token number" in out
    args = build_parser().parse_args(["validate-sim", "--endpoint-url", "u", "--gpu-hourly-rate", "1"])
    assert args.command == "validate-sim"


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
