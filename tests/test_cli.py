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
from pathlib import Path
from unittest.mock import patch

import httpx

from throttle.benchmark import run_native as run_native_core
from throttle.cli import (
    EXIT_CANCELLED,
    EXIT_INCONCLUSIVE,
    EXIT_OK,
    EXIT_USAGE,
    _atomic_write,
    _build_config,
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


async def _offline_native(
    config: object, prompts: object, warmups: object, *, progress: object
) -> dict[str, object]:
    return await run_native_core(  # type: ignore[arg-type]
        config,
        prompts,
        warmups,
        transport=httpx.MockTransport(lambda _: _valid_completion()),
        progress=progress,
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


class ParserAndPlanTests(unittest.TestCase):
    def test_parser_has_four_explicit_subcommands(self) -> None:
        parser = build_parser()
        subparser_action = next(
            action
            for action in parser._actions
            if action.__class__.__name__ == "_SubParsersAction"
        )
        self.assertEqual(
            set(subparser_action.choices),
            {"plan", "smoke", "benchmark", "compare"},
        )

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
            patch(
                "throttle.cli._run_guidellm_backend",
                side_effect=AssertionError("plan attempted GuideLLM traffic"),
            ) as guidellm,
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(argv)

        output = stdout.getvalue()
        self.assertEqual(exit_code, EXIT_OK)
        self.assertIn("zero traffic sent", output)
        self.assertIn("Requests: exactly 27 including warm-ups", output)
        self.assertIn("max 128 output tokens/request", output)
        self.assertIn(PRIVATE_ENDPOINT + "/chat/completions", output)
        self.assertIn("blocked until --allow-unknown-cost", output)
        native.assert_not_called()
        guidellm.assert_not_called()

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


class CliRunAndPersistenceTests(unittest.TestCase):
    def test_guidellm_cli_removes_selected_credential_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "guidellm-smoke.json"
            argv = [
                "smoke",
                "--model",
                "model-a",
                "--url",
                "https://example.test/v1",
                "--api-key-env",
                "CUSTOM_UNUSUAL_NAME",
                "--backend",
                "guidellm",
                "--guidellm-prompt-tokens",
                "32",
                "--allow-guidellm-validation-gaps",
                "--cost-model",
                "dedicated-hourly",
                "--total-hourly-price",
                "0.25",
                "--output",
                str(output),
            ]
            with (
                patch.dict(
                    os.environ,
                    {
                        "CUSTOM_UNUSUAL_NAME": SECRET_KEY,
                        "OPENAI_API_KEY": "second-ambient-secret",
                    },
                    clear=False,
                ),
                patch(
                    "throttle.guidellm_backend.run_guidellm_matrix",
                    return_value=_smoke_artifact(),
                ) as runner,
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                exit_code = main(argv)

            self.assertEqual(exit_code, EXIT_OK)
            child_source = runner.call_args.kwargs["environ"]
            self.assertNotIn("CUSTOM_UNUSUAL_NAME", child_source)

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
