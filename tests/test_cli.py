from __future__ import annotations

import asyncio
import contextlib
import io
import ipaddress
import json
import os
import socket
import stat
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import httpx

from throttle.benchmark import load_prompts as load_prompts_core
from throttle.benchmark import run_native as run_native_core
from throttle.cli import (
    EXIT_CANCELLED,
    EXIT_FAILED,
    EXIT_INCONCLUSIVE,
    EXIT_OK,
    EXIT_USAGE,
    _atomic_write,
    _atomic_write_guarded,
    _build_config,
    _golden_runtime_remaining,
    _print_golden,
    build_parser,
    main,
)


_REAL_GETADDRINFO = socket.getaddrinfo
_REAL_CONNECT = socket.socket.connect
_NETWORK_PATCHES: tuple[object, ...] = ()


def _loopback_host(value: object) -> bool:
    if isinstance(value, bytes):
        value = value.decode("ascii", errors="ignore")
    if not isinstance(value, str):
        return False
    host = value.strip("[]").rstrip(".").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _offline_getaddrinfo(host: object, *args: object, **kwargs: object) -> object:
    if not _loopback_host(host):
        raise AssertionError(f"offline test attempted non-loopback DNS: {host!r}")
    return _REAL_GETADDRINFO(host, *args, **kwargs)


def _offline_connect(sock: socket.socket, address: object) -> object:
    if isinstance(address, str):
        return _REAL_CONNECT(sock, address)
    host = address[0] if isinstance(address, tuple) and address else None
    if not _loopback_host(host):
        raise AssertionError(f"offline test attempted non-loopback connect: {host!r}")
    return _REAL_CONNECT(sock, address)


def setUpModule() -> None:
    global _NETWORK_PATCHES
    dns = patch.object(socket, "getaddrinfo", side_effect=_offline_getaddrinfo)
    connect = patch.object(socket.socket, "connect", new=_offline_connect)
    dns.start()
    connect.start()
    _NETWORK_PATCHES = (dns, connect)


def tearDownModule() -> None:
    for network_patch in reversed(_NETWORK_PATCHES):
        network_patch.stop()  # type: ignore[attr-defined]


SECRET_KEY = "private-cli-api-key"
PRIVATE_ENDPOINT = "https://private-cli-endpoint.example/v1"
PRIVATE_RESPONSE = "private CLI generated response"


def _valid_completion() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": PRIVATE_RESPONSE,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4},
        },
    )


def _valid_streaming_completion() -> httpx.Response:
    events = (
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "private CLI generated"},
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": " response"},
                    "finish_reason": None,
                }
            ]
        },
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {
            "choices": [],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4},
        },
    )
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    body += "data: [DONE]\n\n"
    return httpx.Response(200, content=body.encode("utf-8"))


async def _offline_native(
    config: object,
    prompts: object,
    warmups: object,
    *,
    progress: object,
    **run_kwargs: object,
) -> dict[str, object]:
    response = (
        _valid_streaming_completion
        if getattr(config, "stream", False)
        else _valid_completion
    )

    async def delayed_response(_: httpx.Request) -> httpx.Response:
        # Yield long enough for closed-loop workers to overlap so the Golden
        # concurrency-8 qualification observes the configured load.
        await asyncio.sleep(0.001)
        return response()

    return await run_native_core(  # type: ignore[arg-type]
        config,
        prompts,
        warmups,
        transport=httpx.MockTransport(delayed_response),
        progress=progress,
        **run_kwargs,
    )


def _run_args(command: str, output: Path, *extra: str) -> list[str]:
    return [
        command,
        "--model",
        "model-a",
        "--url",
        PRIVATE_ENDPOINT,
        "--api-key-env",
        "THROTTLE_CLI_TEST_KEY",
        "--cost-model",
        "dedicated-hourly",
        "--total-hourly-price",
        "0.25",
        "--no-stream",
        "--max-tokens",
        "8",
        "--max-elapsed-seconds",
        "5",
        *extra,
        "--output",
        str(output),
    ]


def _smoke_artifact() -> dict[str, object]:
    return {
        "schema_version": "2.0",
        "artifact_type": "throttle_run",
        "mode": "smoke",
        "status": "complete",
        "decision_eligible": False,
        "conditions": [],
        "best_tested": {
            "available": False,
            "state": "not_evaluated",
            "optimum_found": False,
        },
        "disclaimer": "Smoke artifact; no production recommendation.",
    }


def _golden_args(
    output_dir: Path,
    *,
    key_env: str = "THROTTLE_GOLDEN_KEY",
    runtime_args: tuple[str, ...] | None = None,
    baseline_max_num_seqs: str = "1",
    candidate_max_num_seqs: str = "8",
    concurrency: int | None = None,
) -> list[str]:
    if runtime_args is None:
        runtime_args = (
            "--image-digest",
            "sha256:" + "a" * 64,
            "--gpu",
            "NVIDIA RTX A6000",
            "--gpu-fingerprint",
            "private-stable-device",
            "--cuda-version",
            "12.8",
            "--driver-version",
            "570.86",
        )
    load_args = () if concurrency is None else ("--concurrency", str(concurrency))
    return [
        "golden",
        "--model",
        "model-a",
        "--url",
        PRIVATE_ENDPOINT,
        "--api-key-env",
        key_env,
        "--baseline-config",
        f"max_num_seqs={baseline_max_num_seqs}",
        "--candidate-config",
        f"max_num_seqs={candidate_max_num_seqs}",
        *load_args,
        "--cost-model",
        "dedicated-hourly",
        "--total-hourly-price",
        "0.25",
        "--cache-policy",
        "disabled",
        "--model-revision",
        "0123456789abcdef0123456789abcdef01234567",
        *runtime_args,
        "--server-name",
        "vllm",
        "--server-version",
        "0.27.1",
        "--engine-flag",
        "enable_chunked_prefill=true",
        "--engine-flags-provenance",
        "runtime_verified",
        "--evidence-source",
        "live_inference",
        "--output-dir",
        str(output_dir),
    ]


def _platform_golden_runtime_args(
    *,
    backend: str,
    accelerator: str,
    runtime_version: str,
    host_os_version: str,
) -> tuple[str, ...]:
    return (
        "--accelerator-backend",
        backend,
        "--accelerator",
        accelerator,
        "--accelerator-fingerprint",
        f"private-{backend}-device",
        "--accelerator-runtime-version",
        runtime_version,
        "--host-os-version",
        host_os_version,
        "--software-environment-digest",
        f"{backend}-environment@sha256:" + "f" * 64,
        "--image-digest",
        "unknown",
        "--cuda-version",
        "unknown",
        "--driver-version",
        "unknown",
    )


def _golden_position_artifact(*, valid: bool = True) -> dict[str, object]:
    return {
        "schema_version": "2.0",
        "artifact_type": "throttle_run",
        "mode": "benchmark",
        "status": "complete" if valid else "failed",
        "decision_eligible": False,
        "stop_reason": None if valid else "max_errors",
        "conditions": [
            {
                "condition": {
                    "id": "closed_loop:8",
                    "kind": "closed_loop",
                    "value": 8,
                    "max_in_flight": 8,
                },
                "valid": valid,
                "decision_grade": valid,
                "decision_ineligible_reasons": []
                if valid
                else ["invalid_or_incomplete_block"],
                "request_counts": {
                    "attempted": 201 if valid else 1,
                    "valid": 201 if valid else 0,
                },
                "metrics": {} if valid else None,
            }
        ],
        "best_tested": {"available": False, "reason": "search_boundary"},
        "run_totals": {"errors": 0 if valid else 1},
        "cost_summary": {
            "kind": "dedicated_hourly",
            "total_cost": 0.001,
            "cost_per_million_output_tokens": 1.0,
        },
        "disclaimer": "Position evidence only.",
    }


def _supported_golden_artifact() -> dict[str, object]:
    return {
        "schema_version": "2.0",
        "artifact_type": "throttle_golden_live_comparison",
        "golden_protocol_eligible": True,
        "decision_eligible": True,
        "decision_state": "supported",
        "eligibility_reasons": [],
        "treatment": {
            "field": "max_num_seqs",
            "baseline_value": 1,
            "candidate_value": 8,
            "closed_loop_concurrency": 8,
        },
        "conditions": [
            {
                "condition_id": "closed_loop:8",
                "state": "supported",
                "throughput_delta_percent_ci": {
                    "estimate": 23.0,
                    "low": 17.0,
                    "high": 29.0,
                },
            }
        ],
        "overall_outcome": "candidate_higher_throughput",
        "decision_summary": {
            "winner": "candidate",
            "text": (
                "Golden recommendation — tested workload only: candidate "
                "max_num_seqs=8 won by 23.0%."
            ),
        },
        "disclaimer": "Pinned workload only.",
    }


class ParserAndPlanTests(unittest.TestCase):
    def test_parser_subcommands_match_expected(self) -> None:
        """Verify all CLI subcommands are present and accounted for."""
        parser = build_parser()
        subparser_action = next(
            action
            for action in parser._actions
            if action.__class__.__name__ == "_SubParsersAction"
        )
        self.assertEqual(
            set(subparser_action.choices),
            {
                "plan",
                "smoke",
                "benchmark",
                "golden",
                "golden-report",
                "compare",
                "proxy",
                "watch",
                "check",
                "sessions",
                "ui",
                "savings",
                "autotune",
                "gate",
                "coldstart",
                "capture",
                "shadow",
                "eval",
                "workloads",
                "refresh",
            },
        )

    def test_platform_neutral_accelerator_options_build_a_metal_config(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "plan",
                "--model",
                "model-a",
                "--url",
                PRIVATE_ENDPOINT,
                "--accelerator-backend",
                "metal",
                "--accelerator",
                "Apple Silicon integrated GPU",
                "--accelerator-fingerprint",
                "private-apple-device-id",
                "--accelerator-runtime-version",
                "MLX 0.32.0",
                "--host-os-version",
                "macOS 15.0 build 24A335",
                "--software-environment-digest",
                "python-environment@sha256:" + "f" * 64,
            ]
        )
        config, _, _ = _build_config(parser, args, resolve_key=False)
        self.assertEqual(config.accelerator_backend, "metal")
        self.assertEqual(config.gpu, "Apple Silicon integrated GPU")
        self.assertEqual(config.gpu_fingerprint, "private-apple-device-id")
        self.assertEqual(config.cuda_version, "unknown")
        self.assertEqual(config.image_digest, "unknown")

    def test_plan_with_unknown_server_never_claims_runtime_evidence_complete(
        self,
    ) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = main(
                [
                    "plan",
                    "--model",
                    "model-a",
                    "--url",
                    PRIVATE_ENDPOINT,
                    "--accelerator-backend",
                    "metal",
                    "--accelerator",
                    "Apple Silicon integrated GPU",
                    "--accelerator-fingerprint",
                    "private-apple-device-id",
                    "--accelerator-runtime-version",
                    "MLX 0.32.0",
                    "--host-os-version",
                    "macOS 15.0 build 24A335",
                    "--software-environment-digest",
                    "metal-environment@sha256:" + "f" * 64,
                ]
            )

        output = stdout.getvalue()
        self.assertEqual(exit_code, EXIT_OK)
        self.assertNotIn("Runtime evidence: complete", output.splitlines())
        self.assertIn("complete_runtime_provenance_required", output)

    def test_plan_needs_no_key_sends_no_traffic_and_shows_27_calls(self) -> None:
        argv = [
            "plan",
            "--model",
            "model-a",
            "--url",
            PRIVATE_ENDPOINT,
            "--api-key-env",
            "MISSING_THROTTLE_KEY",
        ]
        stdout = io.StringIO()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "throttle.cli.run_native",
                side_effect=AssertionError("plan attempted native traffic"),
            ) as native,
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(argv)

        output = stdout.getvalue()
        self.assertEqual(exit_code, EXIT_OK)
        self.assertIn("zero traffic sent", output)
        self.assertIn("Requests: exactly 27 including warm-ups", output)
        self.assertIn("max 128 output tokens/request", output)
        self.assertIn(PRIVATE_ENDPOINT + "/chat/completions", output)
        self.assertIn("blocked until a price or --allow-unknown-cost", output)
        native.assert_not_called()

    def test_plan_per_gpu_cost_is_multiplied_exactly_once(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "plan",
                "--model",
                "model-a",
                "--url",
                PRIVATE_ENDPOINT,
                "--cost-model",
                "dedicated-hourly",
                "--per-gpu-hourly-price",
                "0.50",
                "--gpus",
                "2",
            ]
        )
        config, _, _ = _build_config(parser, args, resolve_key=False)
        self.assertEqual(config.cost.total_hourly_rate, 1.0)
        self.assertEqual(config.cost.gpu_count, 2)

    def test_elapsed_defaults_are_mode_specific_and_overridable(self) -> None:
        parser = build_parser()
        cases = (
            ("plan", ["--run-mode", "smoke"], 120.0),
            ("plan", ["--run-mode", "benchmark"], 900.0),
            ("smoke", [], 120.0),
            ("benchmark", [], 900.0),
            (
                "golden",
                [
                    "--baseline-config",
                    "max_num_seqs=1",
                    "--candidate-config",
                    "max_num_seqs=8",
                ],
                5_400.0,
            ),
        )
        for command, extra, expected in cases:
            with self.subTest(command=command, extra=extra):
                args = parser.parse_args(
                    [
                        command,
                        "--model",
                        "model-a",
                        "--url",
                        PRIVATE_ENDPOINT,
                        *extra,
                    ]
                )
                config, _, _ = _build_config(parser, args, resolve_key=False)
                self.assertEqual(config.limits.max_elapsed_seconds, expected)

        overridden = parser.parse_args(
            [
                "smoke",
                "--model",
                "model-a",
                "--url",
                PRIVATE_ENDPOINT,
                "--max-elapsed-seconds",
                "37",
            ]
        )
        config, _, _ = _build_config(parser, overridden, resolve_key=False)
        self.assertEqual(config.limits.max_elapsed_seconds, 37.0)

    def test_plan_prominently_discloses_plaintext_override_and_hashes(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = main(
                [
                    "plan",
                    "--model",
                    "model-a",
                    "--url",
                    "http://benchmark.example.test/v1",
                    "--allow-insecure-http",
                ]
            )
        output = stdout.getvalue()
        self.assertEqual(exit_code, EXIT_OK)
        self.assertIn("PLAINTEXT WARNING", output)
        self.assertIn("bearer credentials", output)
        self.assertIn("without TLS", output)
        self.assertIn("stable workload fingerprints", output)
        self.assertIn("confirm a guessed workload", output)

    def test_multi_condition_benchmark_warns_before_any_run(self) -> None:
        stderr = io.StringIO()
        with (
            patch("throttle.cli._handle_run", return_value=EXIT_INCONCLUSIVE) as run,
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = main(
                [
                    "benchmark",
                    "--model",
                    "model-a",
                    "--url",
                    PRIVATE_ENDPOINT,
                    "--concurrency",
                    "1",
                    "2",
                    "4",
                    "8",
                ]
            )

        self.assertEqual(exit_code, EXIT_INCONCLUSIVE)
        self.assertIn("exploratory only", stderr.getvalue())
        self.assertIn("cannot reach decision_eligible: true", stderr.getvalue())
        self.assertIn("throttle golden --help", stderr.getvalue())
        run.assert_called_once()

    def test_benchmark_plan_warns_before_key_resolution_or_traffic(self) -> None:
        stderr = io.StringIO()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "throttle.cli._resolve_key",
                side_effect=AssertionError("plan tried to resolve a key"),
            ) as resolve_key,
            patch(
                "throttle.cli.run_native",
                side_effect=AssertionError("plan attempted traffic"),
            ) as runner,
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = main(
                [
                    "plan",
                    "--run-mode",
                    "benchmark",
                    "--model",
                    "model-a",
                    "--url",
                    PRIVATE_ENDPOINT,
                    "--api-key-env",
                    "MISSING_BENCHMARK_KEY",
                    "--concurrency",
                    "1",
                    "2",
                ]
            )

        self.assertEqual(exit_code, EXIT_OK)
        self.assertIn("exploratory only", stderr.getvalue())
        self.assertIn("cannot reach decision_eligible: true", stderr.getvalue())
        resolve_key.assert_not_called()
        runner.assert_not_called()

    def test_default_and_multi_request_rate_benchmarks_emit_sweep_warning(self) -> None:
        cases = (
            ("default concurrency sweep", []),
            ("request-rate sweep", ["--request-rate", "1", "2"]),
        )
        for name, extra in cases:
            with self.subTest(name=name):
                stderr = io.StringIO()
                with (
                    patch(
                        "throttle.cli._handle_run", return_value=EXIT_INCONCLUSIVE
                    ) as run,
                    contextlib.redirect_stderr(stderr),
                ):
                    exit_code = main(
                        [
                            "benchmark",
                            "--model",
                            "model-a",
                            "--url",
                            PRIVATE_ENDPOINT,
                            *extra,
                        ]
                    )

                self.assertEqual(exit_code, EXIT_INCONCLUSIVE)
                self.assertIn("exploratory only", stderr.getvalue())
                run.assert_called_once()

    def test_single_condition_benchmark_does_not_emit_sweep_warning(self) -> None:
        stderr = io.StringIO()
        with (
            patch("throttle.cli._handle_run", return_value=EXIT_INCONCLUSIVE),
            contextlib.redirect_stderr(stderr),
        ):
            main(
                [
                    "benchmark",
                    "--model",
                    "model-a",
                    "--url",
                    PRIVATE_ENDPOINT,
                    "--concurrency",
                    "8",
                ]
            )

        self.assertNotIn("exploratory only", stderr.getvalue())

    def test_golden_dry_run_needs_no_key_or_traffic(self) -> None:
        stdout = io.StringIO()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "throttle.cli.run_native",
                side_effect=AssertionError("golden dry-run attempted traffic"),
            ) as runner,
            patch(
                "builtins.input",
                side_effect=AssertionError("golden dry-run prompted operator"),
            ) as prompt,
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(
                [
                    "golden",
                    "--dry-run",
                    "--model",
                    "model-a",
                    "--url",
                    PRIVATE_ENDPOINT,
                    "--api-key-env",
                    "MISSING_GOLDEN_KEY",
                    "--baseline-config",
                    "max_num_seqs=1",
                    "--candidate-config",
                    "max_num_seqs=8",
                ]
            )

        output = stdout.getvalue()
        self.assertEqual(exit_code, EXIT_OK)
        self.assertIn("ZERO TRAFFIC SENT", output)
        self.assertIn("B1 → C1 → B2 → C2 → B3 → C3", output)
        self.assertIn("204 per position", output)
        self.assertIn("1224 across all six positions", output)
        self.assertIn("156672 session ceiling", output)
        self.assertIn("Decision-grade preflight: BLOCKED", output)
        runner.assert_not_called()
        prompt.assert_not_called()

    def test_ready_golden_dry_run_does_not_create_output_or_resolve_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "must-not-exist"
            argv = [
                *_golden_args(output_dir, key_env="MISSING_GOLDEN_KEY"),
                "--dry-run",
            ]
            stdout = io.StringIO()
            with (
                patch.dict(os.environ, {}, clear=True),
                patch(
                    "throttle.cli._resolve_key",
                    side_effect=AssertionError("dry-run tried to resolve a key"),
                ) as resolve_key,
                patch(
                    "throttle.cli.run_native",
                    side_effect=AssertionError("dry-run attempted traffic"),
                ) as runner,
                patch(
                    "throttle.cli._timed_operator_input",
                    side_effect=AssertionError("dry-run prompted the operator"),
                ) as operator_input,
                contextlib.redirect_stdout(stdout),
            ):
                exit_code = main(argv)

            self.assertEqual(exit_code, EXIT_OK)
            self.assertIn("Decision-grade preflight: READY", stdout.getvalue())
            self.assertIn(
                "Load: closed-loop concurrency 8", stdout.getvalue()
            )
            self.assertFalse(output_dir.exists())
            resolve_key.assert_not_called()
            runner.assert_not_called()
            operator_input.assert_not_called()

    def test_arbitrary_pair_golden_plan_uses_requested_load_without_traffic(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "golden-8-vs-10-plan"
            stdout = io.StringIO()
            argv = [
                *_golden_args(
                    output_dir,
                    key_env="MISSING_GOLDEN_KEY",
                    baseline_max_num_seqs="8",
                    candidate_max_num_seqs="10",
                    concurrency=16,
                ),
                "--dry-run",
            ]
            with (
                patch.dict(os.environ, {}, clear=True),
                patch(
                    "throttle.cli._resolve_key",
                    side_effect=AssertionError("plan resolved a credential"),
                ) as resolve_key,
                patch(
                    "throttle.cli.run_native",
                    side_effect=AssertionError("plan sent benchmark traffic"),
                ) as runner,
                patch(
                    "throttle.cli._timed_operator_input",
                    side_effect=AssertionError("plan prompted the operator"),
                ) as operator_input,
                contextlib.redirect_stdout(stdout),
            ):
                exit_code = main(argv)

            output = stdout.getvalue()
            self.assertEqual(exit_code, EXIT_OK)
            self.assertIn(
                "Treatment: baseline max_num_seqs=8; candidate max_num_seqs=10",
                output,
            )
            self.assertIn("Load: closed-loop concurrency 16", output)
            self.assertIn(
                "not direct server-scheduler saturation", output
            )
            self.assertIn("Decision-grade preflight: READY", output)
            self.assertIn("B1 → C1 → B2 → C2 → B3 → C3", output)
            self.assertFalse(output_dir.exists())
            resolve_key.assert_not_called()
            runner.assert_not_called()
            operator_input.assert_not_called()

    def test_invalid_golden_treatments_fail_before_key_output_or_traffic(self) -> None:
        cases = (
            ("zero", "max_num_seqs=0", "max_num_seqs=10"),
            ("negative", "max_num_seqs=-1", "max_num_seqs=10"),
            ("explicit_plus", "max_num_seqs=+8", "max_num_seqs=10"),
            ("leading_space", "max_num_seqs= 8", "max_num_seqs=10"),
            ("trailing_space", "max_num_seqs=8 ", "max_num_seqs=10"),
            ("leading_zero", "max_num_seqs=08", "max_num_seqs=10"),
            ("decimal", "max_num_seqs=8.0", "max_num_seqs=10"),
            ("exponent", "max_num_seqs=1e1", "max_num_seqs=10"),
            ("over_bound", "max_num_seqs=2147483648", "max_num_seqs=10"),
            ("oversized", "max_num_seqs=" + "9" * 5_000, "max_num_seqs=10"),
            (
                "wrong_flag",
                "max_num_batched_tokens=8",
                "max_num_batched_tokens=10",
            ),
            ("equal", "max_num_seqs=8", "max_num_seqs=8"),
            (
                "private_payload",
                "max_num_seqs=private-treatment-secret",
                "max_num_seqs=10",
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            for name, baseline, candidate in cases:
                with self.subTest(case=name):
                    output_dir = Path(temp_dir) / name
                    argv = _golden_args(
                        output_dir,
                        key_env="MISSING_GOLDEN_KEY",
                        concurrency=16,
                    )
                    argv[argv.index("--baseline-config") + 1] = baseline
                    argv[argv.index("--candidate-config") + 1] = candidate
                    stderr = io.StringIO()
                    with (
                        patch.dict(os.environ, {}, clear=True),
                        patch(
                            "throttle.cli._resolve_key",
                            side_effect=AssertionError(
                                "invalid treatment resolved a credential"
                            ),
                        ) as resolve_key,
                        patch(
                            "throttle.cli.run_native",
                            side_effect=AssertionError(
                                "invalid treatment sent benchmark traffic"
                            ),
                        ) as runner,
                        patch(
                            "throttle.cli._timed_operator_input",
                            side_effect=AssertionError(
                                "invalid treatment prompted the operator"
                            ),
                        ) as operator_input,
                        contextlib.redirect_stderr(stderr),
                    ):
                        with self.assertRaises(SystemExit) as raised:
                            main(argv)

                    self.assertEqual(raised.exception.code, EXIT_USAGE)
                    self.assertFalse(output_dir.exists())
                    resolve_key.assert_not_called()
                    runner.assert_not_called()
                    operator_input.assert_not_called()
                    self.assertNotIn("Traceback", stderr.getvalue())
                    if name == "equal":
                        self.assertIn(
                            "golden_max_num_seqs_values_must_be_distinct",
                            stderr.getvalue(),
                        )
                    elif name == "wrong_flag":
                        self.assertIn(
                            "golden_treatment_must_only_change_max_num_seqs",
                            stderr.getvalue(),
                        )
                    else:
                        self.assertIn(
                            "golden_max_num_seqs_values_must_be_canonical_positive_integers",
                            stderr.getvalue(),
                        )
                    if name == "private_payload":
                        self.assertNotIn("private-treatment-secret", stderr.getvalue())
                    if name == "oversized":
                        self.assertNotIn("9" * 5_000, stderr.getvalue())

    def test_golden_load_below_larger_pair_value_fails_before_key_or_traffic(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "golden-underloaded"
            stdout = io.StringIO()
            stderr = io.StringIO()
            argv = _golden_args(
                output_dir,
                key_env="MISSING_GOLDEN_KEY",
                baseline_max_num_seqs="8",
                candidate_max_num_seqs="10",
                concurrency=9,
            )
            with (
                patch.dict(os.environ, {}, clear=True),
                patch(
                    "throttle.cli._resolve_key",
                    side_effect=AssertionError("underloaded plan resolved a key"),
                ) as resolve_key,
                patch(
                    "throttle.cli.run_native",
                    side_effect=AssertionError("underloaded plan sent traffic"),
                ) as runner,
                patch(
                    "throttle.cli._timed_operator_input",
                    side_effect=AssertionError("underloaded plan prompted operator"),
                ) as operator_input,
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                exit_code = main(argv)

            self.assertEqual(exit_code, EXIT_USAGE)
            self.assertIn(
                "golden_concurrency_must_reach_both_max_num_seqs_values",
                stdout.getvalue(),
            )
            self.assertIn("blocked before key resolution or traffic", stderr.getvalue())
            self.assertFalse(output_dir.exists())
            resolve_key.assert_not_called()
            runner.assert_not_called()
            operator_input.assert_not_called()

    def test_golden_count_bounded_block_must_be_able_to_reach_declared_load(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "golden-underfilled-blocks"
            argv = [
                *_golden_args(
                    output_dir,
                    key_env="MISSING_GOLDEN_KEY",
                    baseline_max_num_seqs="8",
                    candidate_max_num_seqs="10",
                    concurrency=16,
                ),
                "--blocks",
                "20",
                "--requests-per-block",
                "10",
            ]
            stdout = io.StringIO()
            with (
                patch.dict(os.environ, {}, clear=True),
                patch(
                    "throttle.cli._resolve_key",
                    side_effect=AssertionError("underfilled blocks resolved a key"),
                ) as resolve_key,
                patch(
                    "throttle.cli.run_native",
                    side_effect=AssertionError("underfilled blocks sent traffic"),
                ) as runner,
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                exit_code = main(argv)

            self.assertEqual(exit_code, EXIT_USAGE)
            self.assertIn(
                "golden_requests_per_block_must_reach_declared_concurrency",
                stdout.getvalue(),
            )
            self.assertFalse(output_dir.exists())
            resolve_key.assert_not_called()
            runner.assert_not_called()

    def test_golden_reuses_the_exact_preflighted_prompt_sets_for_traffic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "golden-single-prompt-load"
            with (
                patch.dict(
                    os.environ, {"THROTTLE_GOLDEN_KEY": SECRET_KEY}, clear=True
                ),
                patch(
                    "throttle.cli.load_prompts", wraps=load_prompts_core
                ) as loader,
                patch(
                    "throttle.cli.run_native",
                    side_effect=AssertionError(
                        "operator mismatch should stop before traffic"
                    ),
                ) as runner,
                patch(
                    "throttle.cli._timed_operator_input", return_value="not verified"
                ),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                exit_code = main(_golden_args(output_dir))

            self.assertEqual(exit_code, EXIT_FAILED)
            self.assertEqual(loader.call_count, 2)
            self.assertEqual(loader.call_args_list[0].args, (None,))
            self.assertEqual(loader.call_args_list[0].kwargs, {})
            self.assertEqual(loader.call_args_list[1].args, (None,))
            self.assertEqual(loader.call_args_list[1].kwargs, {"warmup": True})
            runner.assert_not_called()

    def test_platform_neutral_golden_dry_run_matrix_is_ready_without_cuda(
        self,
    ) -> None:
        cases = (
            (
                "metal",
                "Apple Silicon integrated GPU",
                "MLX 0.32.0",
                "macOS 15.0 build 24A335",
            ),
            (
                "rocm",
                "AMD Instinct MI300X",
                "ROCm 6.3.1",
                "Ubuntu 24.04 kernel 6.8.0",
            ),
            (
                "cpu",
                "AMD EPYC 9654",
                "oneDNN 3.7.1",
                "Ubuntu 24.04 kernel 6.8.0",
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            for backend, accelerator, runtime_version, host_os_version in cases:
                with self.subTest(backend=backend):
                    output_dir = Path(temp_dir) / f"golden-{backend}"
                    stdout = io.StringIO()
                    argv = [
                        *_golden_args(
                            output_dir,
                            key_env="MISSING_GOLDEN_KEY",
                            runtime_args=_platform_golden_runtime_args(
                                backend=backend,
                                accelerator=accelerator,
                                runtime_version=runtime_version,
                                host_os_version=host_os_version,
                            ),
                        ),
                        "--dry-run",
                    ]
                    with (
                        patch.dict(os.environ, {}, clear=True),
                        patch(
                            "throttle.cli._resolve_key",
                            side_effect=AssertionError(
                                f"{backend} dry-run tried to resolve a key"
                            ),
                        ) as resolve_key,
                        patch(
                            "throttle.cli.run_native",
                            side_effect=AssertionError(
                                f"{backend} dry-run attempted traffic"
                            ),
                        ) as runner,
                        patch(
                            "throttle.cli._timed_operator_input",
                            side_effect=AssertionError(
                                f"{backend} dry-run prompted the operator"
                            ),
                        ) as operator_input,
                        contextlib.redirect_stdout(stdout),
                    ):
                        exit_code = main(argv)

                    output = stdout.getvalue()
                    self.assertEqual(exit_code, EXIT_OK)
                    self.assertIn("Decision-grade preflight: READY", output)
                    self.assertNotIn("golden_runtime_", output)
                    self.assertFalse(output_dir.exists())
                    resolve_key.assert_not_called()
                    runner.assert_not_called()
                    operator_input.assert_not_called()

    def test_golden_orchestrates_six_verified_positions(self) -> None:
        captured_configs: list[object] = []

        async def golden_runner(
            config: object,
            prompts: object,
            warmups: object,
            *,
            progress: object,
            **run_kwargs: object,
        ) -> dict[str, object]:
            del prompts, warmups, run_kwargs
            captured_configs.append(config)
            report = _golden_position_artifact()
            progress.set(report)  # type: ignore[attr-defined]
            return report

        golden_result = _supported_golden_artifact()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "golden-evidence"
            argv = _golden_args(output_dir)
            stdout = io.StringIO()
            with (
                patch.dict(
                    os.environ, {"THROTTLE_GOLDEN_KEY": SECRET_KEY}, clear=True
                ),
                patch("throttle.cli.run_native", side_effect=golden_runner),
                patch(
                    "throttle.cli.validate_golden_sequence",
                    return_value=golden_result,
                ) as validator,
                patch(
                    "throttle.cli._timed_operator_input",
                    side_effect=[
                        "B1 verified",
                        "C1 verified",
                        "B2 verified",
                        "C2 verified",
                        "B3 verified",
                        "C3 verified",
                    ],
                ),
                contextlib.redirect_stdout(stdout),
            ):
                exit_code = main(argv)

            self.assertEqual(exit_code, EXIT_OK)
            self.assertEqual(len(captured_configs), 6)
            self.assertEqual(stat.S_IMODE(output_dir.stat().st_mode), 0o700)
            self.assertEqual(
                [
                    config.sequence_position  # type: ignore[attr-defined]
                    for config in captured_configs
                ],
                ["B1", "C1", "B2", "C2", "B3", "C3"],
            )
            self.assertEqual(
                [
                    config.variant  # type: ignore[attr-defined]
                    for config in captured_configs
                ],
                [
                    "baseline",
                    "candidate",
                    "baseline",
                    "candidate",
                    "baseline",
                    "candidate",
                ],
            )
            self.assertEqual(
                [
                    config.engine_flags[-1]  # type: ignore[attr-defined]
                    for config in captured_configs
                ],
                [
                    ("max_num_seqs", "1"),
                    ("max_num_seqs", "8"),
                    ("max_num_seqs", "1"),
                    ("max_num_seqs", "8"),
                    ("max_num_seqs", "1"),
                    ("max_num_seqs", "8"),
                ],
            )
            self.assertEqual(
                [
                    config.conditions[0].max_in_flight  # type: ignore[attr-defined]
                    for config in captured_configs
                ],
                [8, 8, 8, 8, 8, 8],
            )
            validator.assert_called_once()
            for name in ("B1", "C1", "B2", "C2", "B3", "C3", "golden"):
                path = output_dir / f"{name}.json"
                self.assertTrue(path.is_file())
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            aggregate = json.loads(
                (output_dir / "golden.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                aggregate["decision_summary"], golden_result["decision_summary"]
            )
            self.assertIn(
                golden_result["decision_summary"]["text"],  # type: ignore[index]
                stdout.getvalue(),
            )
            self.assertIn("Throttle golden live result", stdout.getvalue())
            self.assertNotIn("no traffic sent", stdout.getvalue())

    def test_arbitrary_pair_orchestration_builds_exact_dynamic_six_configs(
        self,
    ) -> None:
        captured_configs: list[object] = []

        async def golden_runner(
            config: object,
            prompts: object,
            warmups: object,
            *,
            progress: object,
            **run_kwargs: object,
        ) -> dict[str, object]:
            del prompts, warmups, run_kwargs
            captured_configs.append(config)
            report = _golden_position_artifact()
            progress.set(report)  # type: ignore[attr-defined]
            return report

        golden_result = _supported_golden_artifact()
        golden_result["treatment"] = {
            "field": "max_num_seqs",
            "baseline_value": 8,
            "candidate_value": 10,
            "closed_loop_concurrency": 10,
        }
        golden_result["decision_summary"] = {
            **golden_result["decision_summary"],  # type: ignore[arg-type]
            "winner_config": {"max_num_seqs": 10},
            "text": (
                "Golden recommendation — tested workload only: candidate "
                "max_num_seqs=10 won by 23.0%."
            ),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "golden-8-vs-10"
            argv = _golden_args(
                output_dir,
                baseline_max_num_seqs="8",
                candidate_max_num_seqs="10",
            )
            with (
                patch.dict(
                    os.environ, {"THROTTLE_GOLDEN_KEY": SECRET_KEY}, clear=True
                ),
                patch("throttle.cli.run_native", side_effect=golden_runner),
                patch(
                    "throttle.cli.validate_golden_sequence",
                    return_value=golden_result,
                ),
                patch(
                    "throttle.cli._timed_operator_input",
                    side_effect=[
                        "B1 verified",
                        "C1 verified",
                        "B2 verified",
                        "C2 verified",
                        "B3 verified",
                        "C3 verified",
                    ],
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                exit_code = main(argv)

            self.assertEqual(exit_code, EXIT_OK)
            self.assertEqual(
                [
                    (
                        config.sequence_position,  # type: ignore[attr-defined]
                        config.variant,  # type: ignore[attr-defined]
                        config.engine_flags[-1],  # type: ignore[attr-defined]
                        config.conditions[0].max_in_flight,  # type: ignore[attr-defined]
                    )
                    for config in captured_configs
                ],
                [
                    ("B1", "baseline", ("max_num_seqs", "8"), 10),
                    ("C1", "candidate", ("max_num_seqs", "10"), 10),
                    ("B2", "baseline", ("max_num_seqs", "8"), 10),
                    ("C2", "candidate", ("max_num_seqs", "10"), 10),
                    ("B3", "baseline", ("max_num_seqs", "8"), 10),
                    ("C3", "candidate", ("max_num_seqs", "10"), 10),
                ],
            )
            aggregate = json.loads(
                (output_dir / "golden.json").read_text(encoding="utf-8")
            )
            self.assertEqual(aggregate["treatment"], golden_result["treatment"])

    def test_golden_cli_six_run_orchestration_uses_real_validator(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "golden-real-validator"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                patch.dict(
                    os.environ, {"THROTTLE_GOLDEN_KEY": SECRET_KEY}, clear=True
                ),
                patch("throttle.cli.run_native", side_effect=_offline_native) as runner,
                patch(
                    "throttle.cli._timed_operator_input",
                    side_effect=[
                        "B1 verified",
                        "C1 verified",
                        "B2 verified",
                        "C2 verified",
                        "B3 verified",
                        "C3 verified",
                    ],
                ),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                exit_code = main(_golden_args(output_dir))

            aggregate = json.loads(
                (output_dir / "golden.json").read_text(encoding="utf-8")
            )
            self.assertEqual(runner.call_count, 6)
            self.assertEqual(
                aggregate["artifact_type"], "throttle_golden_live_comparison"
            )
            self.assertEqual(aggregate["status"], "complete")
            self.assertTrue(
                aggregate["golden_protocol_eligible"],
                aggregate["eligibility_reasons"],
            )
            self.assertEqual(aggregate["eligibility_reasons"], [])
            self.assertEqual(len(aggregate["run_fingerprints"]), 6)
            session_totals = aggregate["session_totals"]
            self.assertEqual(
                session_totals["completed_positions"],
                ["B1", "C1", "B2", "C2", "B3", "C3"],
            )
            self.assertEqual(session_totals["requests_started"], 1224)
            self.assertEqual(session_totals["requests_completed"], 1224)
            self.assertEqual(session_totals["requests_cancelled"], 0)
            self.assertEqual(session_totals["requests_in_flight"], 0)
            self.assertEqual(session_totals["errors"], 0)
            self.assertEqual(session_totals["reserved_output_tokens"], 156672)
            self.assertEqual(
                exit_code,
                EXIT_OK if aggregate["decision_eligible"] else EXIT_INCONCLUSIVE,
                stderr.getvalue(),
            )
            self.assertIn("Throttle golden live result", stdout.getvalue())

    def test_golden_operator_mismatch_writes_sanitized_partial_without_traffic(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "golden-mismatch"
            with (
                patch.dict(
                    os.environ, {"THROTTLE_GOLDEN_KEY": SECRET_KEY}, clear=True
                ),
                patch(
                    "throttle.cli.run_native",
                    side_effect=AssertionError("confirmation mismatch sent traffic"),
                ) as runner,
                patch(
                    "throttle.cli._timed_operator_input", return_value="not verified"
                ),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                exit_code = main(_golden_args(output_dir))

            artifact_path = output_dir / "golden.json"
            artifact_text = artifact_path.read_text(encoding="utf-8")
            artifact = json.loads(artifact_text)
            self.assertEqual(exit_code, EXIT_FAILED)
            self.assertEqual(artifact["artifact_type"], "throttle_golden_session")
            self.assertEqual(artifact["status"], "stopped")
            self.assertEqual(
                artifact["stop_reason"], "operator_confirmation_failed_B1"
            )
            self.assertEqual(artifact["completed_positions"], [])
            self.assertEqual(artifact["saved_positions"], [])
            self.assertFalse(artifact["decision_eligible"])
            self.assertIsNone(artifact["decision_summary"])
            self.assertEqual(
                artifact["treatment"],
                {
                    "field": "max_num_seqs",
                    "baseline_value": 1,
                    "candidate_value": 8,
                    "closed_loop_concurrency": 8,
                },
            )
            self.assertEqual(stat.S_IMODE(artifact_path.stat().st_mode), 0o600)
            self.assertNotIn(PRIVATE_ENDPOINT, artifact_text)
            self.assertNotIn(SECRET_KEY, artifact_text)
            runner.assert_not_called()

    def test_golden_first_invalid_position_stops_before_later_traffic(self) -> None:
        async def invalid_runner(
            config: object,
            prompts: object,
            warmups: object,
            *,
            progress: object,
            **run_kwargs: object,
        ) -> dict[str, object]:
            del config, prompts, warmups, run_kwargs
            report = _golden_position_artifact(valid=False)
            progress.set(report)  # type: ignore[attr-defined]
            return report

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "golden-invalid"
            with (
                patch.dict(
                    os.environ, {"THROTTLE_GOLDEN_KEY": SECRET_KEY}, clear=True
                ),
                patch("throttle.cli.run_native", side_effect=invalid_runner) as runner,
                patch(
                    "throttle.cli.validate_golden_sequence",
                    side_effect=AssertionError("invalid position reached validation"),
                ) as validator,
                patch(
                    "throttle.cli._timed_operator_input", return_value="B1 verified"
                ),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                exit_code = main(_golden_args(output_dir))

            partial = json.loads(
                (output_dir / "golden.json").read_text(encoding="utf-8")
            )
            self.assertEqual(exit_code, EXIT_FAILED)
            self.assertEqual(runner.call_count, 1)
            validator.assert_not_called()
            self.assertEqual(partial["stop_reason"], "position_B1_not_decision_grade")
            self.assertEqual(partial["completed_positions"], [])
            self.assertEqual(partial["saved_positions"], ["B1"])
            self.assertFalse(partial["decision_eligible"])
            self.assertIsNone(partial["decision_summary"])
            self.assertTrue((output_dir / "B1.json").is_file())
            self.assertFalse((output_dir / "C1.json").exists())

    def test_golden_position_write_failure_stops_before_later_traffic(self) -> None:
        async def valid_runner(
            config: object,
            prompts: object,
            warmups: object,
            *,
            progress: object,
            **run_kwargs: object,
        ) -> dict[str, object]:
            del config, prompts, warmups, run_kwargs
            report = _golden_position_artifact()
            progress.set(report)  # type: ignore[attr-defined]
            return report

        def fail_only_b1(report: object, output: Path) -> None:
            if output.name == "B1.json":
                raise OSError("private filesystem detail must not escape")
            _atomic_write(report, output)  # type: ignore[arg-type]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "golden-write-failure"
            stderr = io.StringIO()
            with (
                patch.dict(
                    os.environ, {"THROTTLE_GOLDEN_KEY": SECRET_KEY}, clear=True
                ),
                patch("throttle.cli.run_native", side_effect=valid_runner) as runner,
                patch("throttle.cli._atomic_write", side_effect=fail_only_b1),
                patch(
                    "throttle.cli._timed_operator_input", return_value="B1 verified"
                ),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(stderr),
            ):
                exit_code = main(_golden_args(output_dir))

            partial_text = (output_dir / "golden.json").read_text(encoding="utf-8")
            partial = json.loads(partial_text)
            self.assertEqual(exit_code, EXIT_FAILED)
            self.assertEqual(runner.call_count, 1)
            self.assertEqual(
                partial["stop_reason"], "position_B1_report_write_failed"
            )
            self.assertEqual(partial["saved_positions"], [])
            self.assertFalse(partial["decision_eligible"])
            self.assertFalse((output_dir / "B1.json").exists())
            self.assertFalse((output_dir / "C1.json").exists())
            self.assertNotIn("private filesystem detail", partial_text)
            self.assertNotIn("private filesystem detail", stderr.getvalue())

    def test_existing_golden_output_dir_blocks_before_key_or_traffic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "existing-evidence"
            output_dir.mkdir()
            with (
                patch.dict(os.environ, {}, clear=True),
                patch(
                    "throttle.cli._resolve_key",
                    side_effect=AssertionError("existing directory resolved a key"),
                ) as resolve_key,
                patch(
                    "throttle.cli.run_native",
                    side_effect=AssertionError("existing directory sent traffic"),
                ) as runner,
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                exit_code = main(
                    _golden_args(output_dir, key_env="MISSING_GOLDEN_KEY")
                )

            self.assertEqual(exit_code, EXIT_USAGE)
            self.assertEqual(list(output_dir.iterdir()), [])
            resolve_key.assert_not_called()
            runner.assert_not_called()

    def test_golden_operator_wait_timeout_stops_session_without_traffic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "golden-timeout"
            with (
                patch.dict(
                    os.environ, {"THROTTLE_GOLDEN_KEY": SECRET_KEY}, clear=True
                ),
                patch(
                    "throttle.cli._golden_runtime_remaining", return_value=0.01
                ),
                patch(
                    "throttle.cli._timed_operator_input",
                    side_effect=TimeoutError("golden_session_limit"),
                ) as operator_input,
                patch(
                    "throttle.cli.run_native",
                    side_effect=AssertionError("timed-out confirmation sent traffic"),
                ) as runner,
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                exit_code = main(_golden_args(output_dir))

            partial = json.loads(
                (output_dir / "golden.json").read_text(encoding="utf-8")
            )
            self.assertEqual(exit_code, EXIT_FAILED)
            self.assertEqual(partial["status"], "stopped")
            self.assertEqual(partial["stop_reason"], "golden_session_limit")
            self.assertEqual(partial["saved_positions"], [])
            self.assertIsNone(partial["decision_summary"])
            operator_input.assert_called_once()
            self.assertEqual(operator_input.call_args.args[1], 0.01)
            runner.assert_not_called()

    def test_golden_remaining_time_honors_rate_model_spend_caps(self) -> None:
        parser = build_parser()
        cases = (
            (
                "dedicated-hourly",
                [
                    "--cost-model",
                    "dedicated-hourly",
                    "--total-hourly-price",
                    "360",
                ],
                4.0,
                6.0,
            ),
            (
                "serverless-active-seconds",
                [
                    "--cost-model",
                    "serverless-active-seconds",
                    "--active-second-price",
                    "0.05",
                    "--max-active-workers",
                    "4",
                ],
                4.0,
                1.0,
            ),
        )
        for name, cost_args, elapsed, expected_remaining in cases:
            with self.subTest(cost_model=name):
                args = parser.parse_args(
                    [
                        "plan",
                        "--run-mode",
                        "benchmark",
                        "--model",
                        "model-a",
                        "--url",
                        PRIVATE_ENDPOINT,
                        *cost_args,
                        "--max-elapsed-seconds",
                        "100",
                        "--max-estimated-spend",
                        "20",
                    ]
                )
                config, _, _ = _build_config(parser, args, resolve_key=False)
                spend_limited = replace(
                    config,
                    limits=replace(config.limits, max_estimated_spend=1.0),
                )
                with patch(
                    "throttle.cli.time.perf_counter", return_value=100.0 + elapsed
                ):
                    remaining = _golden_runtime_remaining(spend_limited, 100.0)

                self.assertAlmostEqual(remaining, expected_remaining)
                self.assertLess(
                    remaining,
                    spend_limited.limits.max_elapsed_seconds - elapsed,
                )

    def test_golden_rate_model_session_expiry_writes_only_ineligible_artifact(
        self,
    ) -> None:
        cases = (
            (
                "dedicated-hourly",
                [
                    "--cost-model",
                    "dedicated-hourly",
                    "--total-hourly-price",
                    "360",
                ],
                "1",
            ),
            (
                "serverless-active-seconds",
                [
                    "--cost-model",
                    "serverless-active-seconds",
                    "--active-second-price",
                    "0.05",
                    "--max-active-workers",
                    "4",
                ],
                "2",
            ),
        )
        for name, cost_args, spend_cap in cases:
            with self.subTest(cost_model=name), tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir) / f"golden-{name}-spend-expiry"
                argv = _golden_args(output_dir)
                cost_index = argv.index("--cost-model")
                del argv[cost_index : cost_index + 4]
                argv[cost_index:cost_index] = cost_args
                argv.extend(
                    [
                        "--max-elapsed-seconds",
                        "10",
                        "--max-estimated-spend",
                        spend_cap,
                    ]
                )
                clock = [100.0]

                def transition_crosses_session_cap(
                    prompt: str, timeout_seconds: float
                ) -> str:
                    self.assertIn("B1 verified", prompt)
                    self.assertAlmostEqual(timeout_seconds, 10.0)
                    clock[0] = 110.0
                    return "B1 verified"

                stdout = io.StringIO()
                with (
                    patch.dict(
                        os.environ, {"THROTTLE_GOLDEN_KEY": SECRET_KEY}, clear=True
                    ),
                    patch(
                        "throttle.cli.time.perf_counter",
                        side_effect=lambda: clock[0],
                    ),
                    patch(
                        "throttle.cli._timed_operator_input",
                        side_effect=transition_crosses_session_cap,
                    ) as operator_input,
                    patch(
                        "throttle.cli.run_native",
                        side_effect=AssertionError(
                            "expired Golden session attempted inference traffic"
                        ),
                    ) as runner,
                    contextlib.redirect_stdout(stdout),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    exit_code = main(argv)

                artifact = json.loads(
                    (output_dir / "golden.json").read_text(encoding="utf-8")
                )
                self.assertEqual(exit_code, EXIT_FAILED)
                self.assertEqual(
                    artifact["artifact_type"], "throttle_golden_session"
                )
                self.assertEqual(artifact["status"], "stopped")
                self.assertEqual(artifact["stop_reason"], "golden_session_limit")
                self.assertEqual(artifact["completed_positions"], [])
                self.assertEqual(artifact["saved_positions"], [])
                self.assertAlmostEqual(
                    artifact["session_totals"]["estimated_cost"],
                    float(spend_cap),
                )
                self.assertFalse(artifact["decision_eligible"])
                self.assertIsNone(artifact["decision_summary"])
                self.assertNotIn("Golden recommendation", stdout.getvalue())
                self.assertNotIn("Throttle golden live result", stdout.getvalue())
                operator_input.assert_called_once()
                runner.assert_not_called()

    def test_golden_cancellation_writes_position_and_session_partials(self) -> None:
        async def cancelled_runner(
            config: object,
            prompts: object,
            warmups: object,
            *,
            progress: object,
            **run_kwargs: object,
        ) -> dict[str, object]:
            del config, prompts, warmups, run_kwargs
            progress.set(_golden_position_artifact())  # type: ignore[attr-defined]
            raise asyncio.CancelledError

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "golden-cancelled"
            with (
                patch.dict(
                    os.environ, {"THROTTLE_GOLDEN_KEY": SECRET_KEY}, clear=True
                ),
                patch(
                    "throttle.cli.run_native", side_effect=cancelled_runner
                ) as runner,
                patch(
                    "throttle.cli._timed_operator_input", return_value="B1 verified"
                ),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                exit_code = main(_golden_args(output_dir))

            position_text = (output_dir / "B1.json").read_text(encoding="utf-8")
            position = json.loads(position_text)
            partial_text = (output_dir / "golden.json").read_text(encoding="utf-8")
            partial = json.loads(partial_text)
            self.assertEqual(exit_code, EXIT_CANCELLED)
            self.assertEqual(runner.call_count, 1)
            self.assertEqual(position["status"], "cancelled")
            self.assertEqual(position["stop_reason"], "cancelled_by_user")
            self.assertEqual(partial["status"], "cancelled")
            self.assertEqual(partial["stop_reason"], "cancelled_by_user")
            self.assertEqual(partial["completed_positions"], [])
            self.assertEqual(partial["saved_positions"], ["B1"])
            self.assertIsNone(partial["decision_summary"])
            self.assertNotIn(PRIVATE_ENDPOINT, position_text + partial_text)
            self.assertNotIn(SECRET_KEY, position_text + partial_text)
            self.assertFalse((output_dir / "C1.json").exists())

    def test_golden_rechecks_session_deadline_after_final_validation(self) -> None:
        async def valid_runner(
            config: object,
            prompts: object,
            warmups: object,
            *,
            progress: object,
            **run_kwargs: object,
        ) -> dict[str, object]:
            del config, prompts, warmups, run_kwargs
            report = _golden_position_artifact()
            progress.set(report)  # type: ignore[attr-defined]
            return report

        validation_finished = False

        def validate(reports: object) -> dict[str, object]:
            nonlocal validation_finished
            del reports
            validation_finished = True
            return _supported_golden_artifact()

        def remaining(config: object, started: float) -> float:
            del config, started
            return 0.0 if validation_finished else 100.0

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "golden-final-deadline"
            stdout = io.StringIO()
            with (
                patch.dict(
                    os.environ, {"THROTTLE_GOLDEN_KEY": SECRET_KEY}, clear=True
                ),
                patch("throttle.cli.run_native", side_effect=valid_runner) as runner,
                patch("throttle.cli.validate_golden_sequence", side_effect=validate),
                patch(
                    "throttle.cli._golden_runtime_remaining", side_effect=remaining
                ),
                patch(
                    "throttle.cli._timed_operator_input",
                    side_effect=[
                        "B1 verified",
                        "C1 verified",
                        "B2 verified",
                        "C2 verified",
                        "B3 verified",
                        "C3 verified",
                    ],
                ),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                exit_code = main(_golden_args(output_dir))

            artifact = json.loads(
                (output_dir / "golden.json").read_text(encoding="utf-8")
            )
            self.assertEqual(exit_code, EXIT_FAILED)
            self.assertEqual(runner.call_count, 6)
            self.assertEqual(artifact["artifact_type"], "throttle_golden_session")
            self.assertEqual(artifact["status"], "stopped")
            self.assertEqual(artifact["stop_reason"], "golden_session_limit")
            self.assertEqual(
                artifact["completed_positions"],
                ["B1", "C1", "B2", "C2", "B3", "C3"],
            )
            self.assertFalse(artifact["decision_eligible"])
            self.assertIsNone(artifact["decision_summary"])
            self.assertNotIn("Golden recommendation", stdout.getvalue())

    def test_golden_deadline_at_final_commit_never_publishes_result(
        self,
    ) -> None:
        async def valid_runner(
            config: object,
            prompts: object,
            warmups: object,
            *,
            progress: object,
            **run_kwargs: object,
        ) -> dict[str, object]:
            del config, prompts, warmups, run_kwargs
            report = _golden_position_artifact()
            progress.set(report)  # type: ignore[attr-defined]
            return report

        attempted_artifact_types: list[object] = []
        committed_artifact_types: list[object] = []

        def reject_final_commit(
            report: object, output: Path, commit_guard: object
        ) -> bool:
            artifact_type = (
                report.get("artifact_type") if isinstance(report, dict) else None
            )
            attempted_artifact_types.append(artifact_type)
            final_result = (
                output.name == "golden.json"
                and artifact_type == "throttle_golden_live_comparison"
            )
            committed = _atomic_write_guarded(
                report,  # type: ignore[arg-type]
                output,
                (lambda: False)
                if final_result
                else commit_guard,  # type: ignore[arg-type]
            )
            if committed:
                committed_artifact_types.append(artifact_type)
            return committed

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "golden-post-write-deadline"
            stdout = io.StringIO()
            with (
                patch.dict(
                    os.environ, {"THROTTLE_GOLDEN_KEY": SECRET_KEY}, clear=True
                ),
                patch("throttle.cli.run_native", side_effect=valid_runner) as runner,
                patch(
                    "throttle.cli.validate_golden_sequence",
                    return_value=_supported_golden_artifact(),
                ) as validator,
                patch(
                    "throttle.cli._atomic_write_guarded",
                    side_effect=reject_final_commit,
                ),
                patch(
                    "throttle.cli._timed_operator_input",
                    side_effect=[
                        "B1 verified",
                        "C1 verified",
                        "B2 verified",
                        "C2 verified",
                        "B3 verified",
                        "C3 verified",
                    ],
                ),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                exit_code = main(_golden_args(output_dir))

            artifact = json.loads(
                (output_dir / "golden.json").read_text(encoding="utf-8")
            )
            self.assertEqual(exit_code, EXIT_FAILED)
            self.assertEqual(runner.call_count, 6)
            validator.assert_called_once()
            self.assertIn(
                "throttle_golden_live_comparison", attempted_artifact_types
            )
            self.assertNotIn(
                "throttle_golden_live_comparison", committed_artifact_types
            )
            self.assertEqual(
                committed_artifact_types[-1], "throttle_golden_session"
            )
            self.assertEqual(artifact["artifact_type"], "throttle_golden_session")
            self.assertEqual(artifact["status"], "stopped")
            self.assertEqual(artifact["stop_reason"], "golden_session_limit")
            self.assertEqual(
                artifact["completed_positions"],
                ["B1", "C1", "B2", "C2", "B3", "C3"],
            )
            self.assertFalse(artifact["decision_eligible"])
            self.assertIsNone(artifact["decision_summary"])
            self.assertNotIn("Golden recommendation", stdout.getvalue())
            self.assertNotIn("Throttle golden live result", stdout.getvalue())

    def test_rejected_final_commit_and_partial_write_failure_keep_safe_artifact(
        self,
    ) -> None:
        async def valid_runner(
            config: object,
            prompts: object,
            warmups: object,
            *,
            progress: object,
            **run_kwargs: object,
        ) -> dict[str, object]:
            del config, prompts, warmups, run_kwargs
            report = _golden_position_artifact()
            progress.set(report)  # type: ignore[attr-defined]
            return report

        final_commit_rejected = False
        committed_artifact_types: list[object] = []

        def reject_final_then_fail_partial(
            report: object, output: Path, commit_guard: object
        ) -> bool:
            nonlocal final_commit_rejected
            artifact_type = (
                report.get("artifact_type") if isinstance(report, dict) else None
            )
            if artifact_type == "throttle_golden_live_comparison":
                final_commit_rejected = True
                committed = _atomic_write_guarded(
                    report, output, lambda: False  # type: ignore[arg-type]
                )
                self.assertFalse(committed)
                return False
            if (
                final_commit_rejected
                and output.name == "golden.json"
                and artifact_type == "throttle_golden_session"
            ):
                raise OSError("injected sanitized-partial write failure")
            committed = _atomic_write_guarded(
                report, output, commit_guard  # type: ignore[arg-type]
            )
            if committed:
                committed_artifact_types.append(artifact_type)
            return committed

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "golden-partial-write-failure"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                patch.dict(
                    os.environ, {"THROTTLE_GOLDEN_KEY": SECRET_KEY}, clear=True
                ),
                patch("throttle.cli.run_native", side_effect=valid_runner) as runner,
                patch(
                    "throttle.cli.validate_golden_sequence",
                    return_value=_supported_golden_artifact(),
                ) as validator,
                patch(
                    "throttle.cli._atomic_write_guarded",
                    side_effect=reject_final_then_fail_partial,
                ),
                patch(
                    "throttle.cli._timed_operator_input",
                    side_effect=[
                        "B1 verified",
                        "C1 verified",
                        "B2 verified",
                        "C2 verified",
                        "B3 verified",
                        "C3 verified",
                    ],
                ),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                exit_code = main(_golden_args(output_dir))

            artifact = json.loads(
                (output_dir / "golden.json").read_text(encoding="utf-8")
            )
            self.assertEqual(exit_code, EXIT_FAILED)
            self.assertEqual(runner.call_count, 6)
            validator.assert_called_once()
            self.assertTrue(final_commit_rejected)
            self.assertNotIn(
                "throttle_golden_live_comparison", committed_artifact_types
            )
            self.assertEqual(artifact["artifact_type"], "throttle_golden_session")
            self.assertEqual(artifact["status"], "partial")
            self.assertEqual(artifact["stop_reason"], "awaiting_first_position")
            self.assertFalse(artifact["decision_eligible"])
            self.assertIsNone(artifact["decision_summary"])
            self.assertNotIn("Golden recommendation", stdout.getvalue())
            self.assertIn(
                "partial session artifact could not be written", stderr.getvalue()
            )
            self.assertEqual(list(output_dir.glob(".golden.json.*.tmp")), [])

    def test_offline_golden_wording_and_recommendation_gate_are_truthful(self) -> None:
        supported = _supported_golden_artifact()
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            _print_golden(supported, Path("golden.json"))

        self.assertIn("six saved reports analyzed; no traffic sent", stdout.getvalue())
        self.assertIn(
            supported["decision_summary"]["text"],  # type: ignore[index]
            stdout.getvalue(),
        )

        ineligible = {**supported, "decision_eligible": False}
        blocked_stdout = io.StringIO()
        with contextlib.redirect_stdout(blocked_stdout):
            _print_golden(ineligible, Path("golden.json"))
        self.assertNotIn("Golden recommendation", blocked_stdout.getvalue())


class CliRunAndPersistenceTests(unittest.TestCase):
    def test_smoke_exit_zero_writes_atomic_sanitized_mode_600_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "smoke.json"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                patch.dict(
                    os.environ,
                    {"THROTTLE_CLI_TEST_KEY": SECRET_KEY},
                    clear=False,
                ),
                patch("throttle.cli.run_native", side_effect=_offline_native),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                exit_code = main(_run_args("smoke", output))

            self.assertEqual(exit_code, EXIT_OK, stderr.getvalue())
            self.assertTrue(output.exists())
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            self.assertEqual(list(output.parent.glob(f".{output.name}.*.tmp")), [])
            report_text = output.read_text(encoding="utf-8")
            report = json.loads(report_text)
            self.assertEqual(report["mode"], "smoke")
            self.assertEqual(report["status"], "complete")
            self.assertFalse(report["decision_eligible"])
            self.assertIn("SHORT SAMPLE, NON-DECISION-GRADE", stdout.getvalue())
            self.assertNotIn("recommendation", report.keys())
            for forbidden in (
                PRIVATE_ENDPOINT,
                "private-cli-endpoint.example",
                SECRET_KEY,
                PRIVATE_RESPONSE,
                "messages",
                "content",
                "Authorization",
                "Bearer",
            ):
                self.assertNotIn(forbidden, report_text)

    def test_underpowered_benchmark_exits_three_with_inconclusive_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "benchmark.json"
            stdout = io.StringIO()
            with (
                patch.dict(
                    os.environ,
                    {"THROTTLE_CLI_TEST_KEY": SECRET_KEY},
                    clear=False,
                ),
                patch("throttle.cli.run_native", side_effect=_offline_native),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                exit_code = main(
                    _run_args(
                        "benchmark",
                        output,
                        "--concurrency",
                        "1",
                        "--blocks",
                        "3",
                        "--requests",
                        "1",
                        "--warmup-requests",
                        "0",
                    )
                )

            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, EXIT_INCONCLUSIVE)
            self.assertEqual(report["status"], "complete")
            self.assertFalse(report["decision_eligible"])
            self.assertFalse(report["conditions"][0]["decision_grade"])
            self.assertIn(
                "measurement_floor_not_met",
                report["conditions"][0]["decision_ineligible_reasons"],
            )
            self.assertNotIn("Recommendation:", stdout.getvalue())

    def test_incompatible_saved_compare_exits_two_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            output = root / "comparison.json"
            _atomic_write(_smoke_artifact(), baseline)
            _atomic_write(_smoke_artifact(), candidate)

            stdout = io.StringIO()
            with (
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                exit_code = main(
                    [
                        "compare",
                        str(baseline),
                        str(candidate),
                        "--output",
                        str(output),
                    ]
                )

            comparison = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, EXIT_USAGE)
            self.assertEqual(comparison["status"], "incompatible")
            self.assertFalse(comparison["decision_eligible"])
            self.assertIn("no traffic sent", stdout.getvalue())
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)

    def test_saved_compare_still_accepts_out_abbreviation(self) -> None:
        # "--out" used to abbreviate "--output"; the live compare added its own
        # --out, which must not silently swallow it for saved reports.
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            output = root / "comparison.json"
            _atomic_write(_smoke_artifact(), baseline)
            _atomic_write(_smoke_artifact(), candidate)
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                main(["compare", str(baseline), str(candidate), "--out", str(output)])
            self.assertTrue(output.exists())

    def test_unattributable_statistical_direction_exits_three(self) -> None:
        comparison_fixture = {
            "schema_version": "2.0",
            "artifact_type": "throttle_comparison",
            "status": "complete",
            "decision_state": "inconclusive",
            "decision_eligible": False,
            "compatibility": {"compatible": True, "reasons": []},
            "attribution": {
                "state": "unattributable",
                "reason": "max_num_seqs_change_not_exercised_by_load",
            },
            "conditions": [
                {
                    "condition_id": "closed_loop:8",
                    "state": "supported",
                    "throughput_delta_percent_ci": {
                        "estimate": 20.0,
                        "low": 20.0,
                        "high": 20.0,
                    },
                }
            ],
            "overall_outcome": None,
            "descriptive_statistical_outcome": "candidate_higher_throughput",
            "decision_ineligible_reasons": [
                "max_num_seqs_change_not_exercised_by_load"
            ],
            "disclaimer": "Descriptive fixture; no causal claim.",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "comparison.json"
            stdout = io.StringIO()
            with (
                patch("throttle.cli.load_report", side_effect=[{}, {}]),
                patch(
                    "throttle.cli.compare_reports",
                    return_value=comparison_fixture,
                ),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                exit_code = main(
                    [
                        "compare",
                        str(root / "baseline.json"),
                        str(root / "candidate.json"),
                        "--output",
                        str(output),
                    ]
                )

            persisted = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, EXIT_INCONCLUSIVE)
            self.assertEqual(persisted["decision_state"], "inconclusive")
            self.assertIsNone(persisted["overall_outcome"])
            self.assertEqual(
                persisted["descriptive_statistical_outcome"],
                "candidate_higher_throughput",
            )
            self.assertIn(
                "Descriptive statistical direction (decision-ineligible)",
                stdout.getvalue(),
            )

    def test_cancelled_run_exits_130_and_atomically_writes_partial(self) -> None:
        async def cancelled(
            config: object,
            prompts: object,
            warmups: object,
            *,
            progress: object,
        ) -> object:
            del config, prompts, warmups
            progress.set(  # type: ignore[attr-defined]
                {
                    "schema_version": "2.0",
                    "artifact_type": "throttle_run",
                    "mode": "smoke",
                    "status": "running",
                    "decision_eligible": False,
                    "conditions": [],
                    "best_tested": {
                        "available": False,
                        "state": "not_evaluated",
                        "optimum_found": False,
                    },
                    "disclaimer": "Sanitized partial artifact.",
                }
            )
            raise asyncio.CancelledError

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "cancelled.json"
            with (
                patch.dict(
                    os.environ,
                    {"THROTTLE_CLI_TEST_KEY": SECRET_KEY},
                    clear=False,
                ),
                patch("throttle.cli.run_native", side_effect=cancelled),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                exit_code = main(_run_args("smoke", output))

            report_text = output.read_text(encoding="utf-8")
            report = json.loads(report_text)
            self.assertEqual(exit_code, EXIT_CANCELLED)
            self.assertEqual(report["status"], "cancelled")
            self.assertEqual(report["stop_reason"], "cancelled_by_user")
            self.assertFalse(report["decision_eligible"])
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            self.assertEqual(list(output.parent.glob(f".{output.name}.*.tmp")), [])
            self.assertNotIn(PRIVATE_ENDPOINT, report_text)
            self.assertNotIn(SECRET_KEY, report_text)

    def test_atomic_write_replaces_existing_file_without_tmp_residue(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "report.json"
            output.write_text("old private content", encoding="utf-8")
            _atomic_write({"status": "complete", "decision_eligible": False}, output)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8"))["status"],
                "complete",
            )
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            self.assertEqual(list(output.parent.glob(f".{output.name}.*.tmp")), [])
            self.assertNotIn("old private content", output.read_text(encoding="utf-8"))


class OfflineGuardRegressionTests(unittest.TestCase):
    def test_candidate_secret_host_is_never_resolved(self) -> None:
        with self.assertRaisesRegex(AssertionError, "non-loopback DNS"):
            socket.getaddrinfo("candidate-secret.example", 443)


if __name__ == "__main__":
    unittest.main()
