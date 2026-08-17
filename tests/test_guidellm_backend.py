from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import throttle.guidellm_backend as guidellm_module
from throttle.guidellm_backend import (
    DECISION_INELIGIBLE_REASONS,
    GUIDELLM_API_KEY_ENV,
    GuideLLMBackendError,
    GuideLLMLimits,
    GuideLLMRunSpec,
    GuideLLMTraffic,
    build_guidellm_scenario,
    parse_guidellm_report,
    preflight_guidellm_config,
    run_guidellm,
    run_guidellm_matrix,
    verify_guidellm_version,
)
from throttle.benchmark import RunProgress
from throttle.models import (
    CostModel,
    EndpointConfig,
    LoadCondition,
    RunConfig,
    SafetyLimits,
)


METRIC_NAMES = (
    "requests_per_second",
    "request_concurrency",
    "request_latency",
    "time_to_first_token_ms",
    "time_per_output_token_ms",
    "inter_token_latency_ms",
    "output_tokens_per_second",
    "prompt_token_count",
    "output_token_count",
)


def summary(count: int = 200, mean: float = 10.0) -> dict[str, object]:
    return {
        "mean": mean,
        "median": mean,
        "mode": mean,
        "variance": 1.0,
        "std_dev": 1.0,
        "min": 1.0,
        "max": mean + 1.0,
        "count": count,
        "total_sum": mean * count,
        "percentiles": {
            "p001": 1.0,
            "p01": 1.0,
            "p05": 1.0,
            "p10": 1.0,
            "p25": 2.0,
            "p50": mean,
            "p75": mean,
            "p90": mean,
            "p95": mean + 0.5,
            "p99": mean + 1.0,
            "p999": mean + 1.0,
        },
        "pdf": None,
    }


def report_fixture() -> dict[str, object]:
    totals = {"successful": 200, "errored": 0, "incomplete": 0, "total": 200}
    metrics: dict[str, object] = {"request_totals": totals}
    for name in METRIC_NAMES:
        metrics[name] = {
            "successful": summary(),
            "errored": summary(0, 0.0),
            "incomplete": summary(0, 0.0),
        }
    return {
        "metadata": {
            "version": 2,
            "guidellm_version": "0.7.3",
            "platform": "private-platform-value",
        },
        # These values model the unsafe parts of a real GuideLLM report.  The
        # parser must never copy them into its returned object.
        "config": {
            "spec": {
                "backend": {
                    "target": "https://candidate-secret.example/v1",
                    "api_key": "credential-that-must-not-leak",
                },
                "data": [{"prompt": "private prompt text"}],
            }
        },
        "benchmarks": [
            {
                "type_": "generative_benchmark",
                "scheduler_metrics": {"requests_made": totals},
                "metrics": metrics,
                "requests": {
                    "successful": [
                        {
                            "output": "private response text",
                            "request_args": {"messages": ["private prompt text"]},
                        }
                    ]
                },
                "duration": 60.0,
            }
        ],
    }


FAKE_GUIDELLM = r"""#!{python}
import json
import os
import pathlib
import stat
import subprocess
import sys
import time

def append_log(value):
    target = os.environ.get("FAKE_CAPTURE_LOG")
    if target:
        with open(target, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(value) + "\n")

if sys.argv[1:] == ["--version"]:
    append_log({{
        "kind": "version",
        "api_key_present": "GUIDELLM__SPEC__BACKEND__API_KEY" in os.environ,
        "ambient_credential_present": any(
            name in os.environ
            for name in ("OPENAI_API_KEY", "VLLM_API_KEY", "GITHUB_TOKEN")
        ),
    }})
    version_marker = os.environ.get("FAKE_VERSION_CHILD_MARKER")
    if version_marker:
        subprocess.Popen([
            sys.executable,
            "-c",
            "import pathlib,sys,time;time.sleep(1);pathlib.Path(sys.argv[1]).write_text('survived')",
            version_marker,
        ])
    if os.environ.get("FAKE_VERSION_BYTES"):
        sys.stdout.write("x" * int(os.environ["FAKE_VERSION_BYTES"]))
    else:
        print("guidellm version: " + os.environ.get("FAKE_VERSION", "0.7.3"))
    raise SystemExit(0)

if (
    len(sys.argv) != 5
    or sys.argv[1] != "run"
    or sys.argv[2] != "--config"
    or sys.argv[4] != "--disable-console"
):
    raise SystemExit(91)

scenario_path = pathlib.Path(sys.argv[3])
scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
report_path = pathlib.Path(scenario["spec"]["outputs"][0]["path"])
capture_path = pathlib.Path(os.environ["FAKE_CAPTURE"])
interesting_env = {{
    key: value
    for key, value in os.environ.items()
    if key.casefold().startswith("guidellm")
    or key.casefold().endswith("_proxy")
    or key in {{"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "OPENAI_API_KEY", "VLLM_API_KEY", "GITHUB_TOKEN", "SSL_CERT_FILE", "SSL_CERT_DIR", "SSLKEYLOGFILE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"}}
}}
capture = {{
    "argv": sys.argv[1:],
    "cwd": os.getcwd(),
    "cwd_mode": stat.S_IMODE(os.stat(os.getcwd()).st_mode),
    "scenario_mode": stat.S_IMODE(scenario_path.stat().st_mode),
    "report_mode": stat.S_IMODE(report_path.stat().st_mode),
    "scenario": scenario,
    "interesting_env": interesting_env,
}}
capture_path.write_text(json.dumps(capture), encoding="utf-8")
append_log({{"kind": "run", **capture}})

marker = os.environ.get("FAKE_CHILD_MARKER")
if marker:
    subprocess.Popen([
        sys.executable,
        "-c",
        "import pathlib,sys,time;time.sleep(1);pathlib.Path(sys.argv[1]).write_text('survived')",
        marker,
    ])
if os.environ.get("FAKE_PARENT_EXIT_AFTER_FORK") == "true":
    raise SystemExit(0)
sleep_seconds = float(os.environ.get("FAKE_SLEEP", "0"))
if sleep_seconds:
    time.sleep(sleep_seconds)

source = pathlib.Path(os.environ["FAKE_REPORT"])
if os.environ.get("FAKE_COPY_RAW_REPORT") == "true":
    report_path.write_bytes(source.read_bytes())
    raise SystemExit(int(os.environ.get("FAKE_EXIT", "0")))
report = json.loads(source.read_text(encoding="utf-8"))
if os.environ.get("FAKE_DYNAMIC_REPORT") == "true":
    constraints = {{item["kind"]: item for item in scenario["spec"]["constraints"]}}
    attempted = int(constraints["max_requests"]["count"])
    errors = min(attempted, int(os.environ.get("FAKE_ERRORS", "0")))
    successful = attempted - errors
    totals = {{"successful": successful, "errored": errors, "incomplete": 0, "total": attempted}}
    benchmark = report["benchmarks"][0]
    benchmark["duration"] = float(os.environ.get("FAKE_DURATION", "1.0"))
    benchmark["scheduler_metrics"]["requests_made"] = totals
    benchmark["metrics"]["request_totals"] = totals
    prompt_tokens = int(scenario["spec"]["data"][0]["prompt_tokens"])
    output_tokens = int(scenario["spec"]["data"][0]["output_tokens"])
    token_metrics = {{"time_per_output_token_ms", "output_tokens_per_second"}}
    request_metrics = {{
        "requests_per_second", "request_concurrency", "request_latency",
        "time_to_first_token_ms", "prompt_token_count", "output_token_count",
    }}
    for metric_name, statuses in benchmark["metrics"].items():
        if metric_name == "request_totals":
            continue
        item = statuses["successful"]
        if metric_name in token_metrics:
            item["count"] = successful * output_tokens
        elif metric_name == "inter_token_latency_ms":
            item["count"] = successful * max(0, output_tokens - 1)
        elif metric_name in request_metrics:
            item["count"] = successful
        if metric_name == "prompt_token_count":
            item["mean"] = float(prompt_tokens)
            item["total_sum"] = float(successful * prompt_tokens)
            item["min"] = item["max"] = float(prompt_tokens)
            item["median"] = item["mode"] = float(prompt_tokens)
            for percentile in item["percentiles"]:
                item["percentiles"][percentile] = float(prompt_tokens)
        elif metric_name == "output_token_count":
            item["mean"] = float(output_tokens)
            item["total_sum"] = float(successful * output_tokens)
            item["min"] = item["max"] = float(output_tokens)
            item["median"] = item["mode"] = float(output_tokens)
            for percentile in item["percentiles"]:
                item["percentiles"][percentile] = float(output_tokens)
        else:
            item["total_sum"] = float(item["mean"]) * item["count"]
report_path.write_text(json.dumps(report), encoding="utf-8")
raise SystemExit(int(os.environ.get("FAKE_EXIT", "0")))
"""


class GuideLLMBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="throttle-guidellm-test-")
        self.root = Path(self.temp.name)
        self.executable = self.root / "fake guidellm; no shell"
        self.executable.write_text(
            FAKE_GUIDELLM.format(python=sys.executable), encoding="utf-8"
        )
        self.executable.chmod(0o700)
        self.raw_report = self.root / "fixture.json"
        self.raw_report.write_text(json.dumps(report_fixture()), encoding="utf-8")
        self.capture = self.root / "capture.json"
        self.capture_log = self.root / "capture.jsonl"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def spec(
        self,
        *,
        traffic: GuideLLMTraffic | None = None,
        process_timeout_seconds: float | None = 5.0,
    ) -> GuideLLMRunSpec:
        return GuideLLMRunSpec(
            endpoint_url="http://127.0.0.1:18123/v1",
            model="private-model-name",
            tokenizer_model="private-tokenizer-name",
            tokenizer_revision="0123456789abcdef",
            traffic=traffic or GuideLLMTraffic.concurrent(4),
            limits=GuideLLMLimits(
                max_requests=200,
                max_duration_seconds=60.0,
                max_errors=1,
                process_timeout_seconds=process_timeout_seconds,
            ),
            prompt_tokens=256,
            output_tokens=128,
            seed=731,
        )

    def environment(self) -> dict[str, str]:
        return {
            "PATH": os.environ.get("PATH", ""),
            "FAKE_CAPTURE": str(self.capture),
            "FAKE_REPORT": str(self.raw_report),
            "FAKE_CAPTURE_LOG": str(self.capture_log),
            "HTTP_PROXY": "http://inherited-proxy-secret.example",
            "https_proxy": "http://lowercase-proxy-secret.example",
            "ALL_PROXY": "socks5://inherited-proxy-secret.example",
            "NO_PROXY": "127.0.0.1",
            "GUIDELLM_ROGUE_SETTING": "must-be-removed",
            GUIDELLM_API_KEY_ENV: "inherited-key-must-be-removed",
            "HF_TOKEN": "inherited-hugging-face-secret",
            "HUGGING_FACE_HUB_TOKEN": "second-inherited-hugging-face-secret",
            "OPENAI_API_KEY": "inherited-openai-secret",
            "VLLM_API_KEY": "inherited-vllm-secret",
            "GITHUB_TOKEN": "inherited-github-secret",
            "SSL_CERT_FILE": "/private/ambient-ca.pem",
            "SSL_CERT_DIR": "/private/ambient-ca-directory",
            "SSLKEYLOGFILE": "/private/tls-secrets.log",
        }

    def matrix_config(self) -> RunConfig:
        return RunConfig(
            mode="benchmark",
            backend="guidellm",
            model="private-model-name",
            endpoint=EndpointConfig("http://127.0.0.1:18123/v1", "matrix-secret-key"),
            cost=CostModel(
                kind="dedicated_hourly", total_hourly_rate=0.25, gpu_count=1
            ),
            limits=SafetyLimits(
                max_requests=100,
                max_tokens_per_request=16,
                max_total_requested_tokens=1_600,
                max_elapsed_seconds=30.0,
                max_errors=1,
                max_concurrency=16,
                max_response_bytes=100_000,
                max_estimated_spend=3.0,
            ),
            max_tokens=8,
            conditions=(
                LoadCondition("closed_loop", 1.0, 1),
                LoadCondition("closed_loop", 4.0, 4),
            ),
            blocks=3,
            requests_per_block=2,
            warmup_requests_per_condition=1,
            request_timeout_seconds=2.0,
            seed=42,
            stream=True,
            cache_policy="disabled",
            model_revision="a" * 40,
            image_digest="image@sha256:" + "b" * 64,
            gpu="test-gpu",
            gpu_fingerprint="one-test-gpu",
            cuda_version="test-cuda",
            driver_version="test-driver",
            server_version="test-server",
            engine_flags_provenance="runtime_verified",
            evidence_source="synthetic_validation",
            guidellm_gaps_acknowledged=True,
        )

    def assert_first_warmup_accounting_is_uncertain(
        self, report: dict[str, object]
    ) -> None:
        self.assertTrue(report["accounting_incomplete"])
        self.assertFalse(report["decision_eligible"])
        self.assertFalse(report["best_tested"]["available"])
        self.assertEqual(report["best_tested"]["state"], "inconclusive")
        self.assertEqual(
            report["best_tested"]["reason"],
            "accounting_incomplete_after_child_failure",
        )
        totals = report["run_totals"]
        self.assertTrue(totals["accounting_incomplete"])
        self.assertIsNone(totals["requests_started"])
        self.assertIsNone(totals["requests_completed"])
        self.assertIsNone(totals["requests_cancelled"])
        self.assertIsNone(totals["errors"])
        self.assertIsNone(totals["reserved_output_tokens"])
        self.assertEqual(totals["accounted_requests_started"], 0)
        self.assertEqual(totals["accounted_requests_completed"], 0)
        self.assertEqual(totals["accounted_requests_cancelled"], 0)
        self.assertEqual(totals["accounted_errors"], 0)
        self.assertEqual(totals["accounted_reserved_output_tokens"], 0)
        self.assertEqual(totals["request_count_bounds"], {"lower": 0, "upper": 1})
        self.assertEqual(
            totals["completed_request_count_bounds"],
            {"lower": 0, "upper": 1},
        )
        self.assertEqual(
            totals["cancelled_request_count_bounds"],
            {"lower": 0, "upper": 1},
        )
        self.assertEqual(
            totals["requested_output_token_bounds"],
            {"lower": 0, "upper": 8},
        )
        self.assertEqual(totals["error_count_bounds"], {"lower": 0, "upper": 1})
        self.assertIsNone(totals["peak_in_flight"])
        self.assertIsNone(totals["observed_peak_in_flight"])
        self.assertEqual(totals["declared_peak_in_flight_cap"], 1)
        self.assertEqual(len(totals["incomplete_children"]), 1)
        child = totals["incomplete_children"][0]
        self.assertEqual(child["phase"], "warmup")
        self.assertEqual(child["condition_index"], 1)
        self.assertIsNone(child["block_index"])
        self.assertEqual(child["seed"], 41)
        self.assertTrue(child["accounting_incomplete"])
        self.assertEqual(child["completion_state"], "unknown_after_child_failure")
        self.assertEqual(child["request_count_bounds"], {"lower": 0, "upper": 1})
        self.assertEqual(
            child["requested_output_token_bounds"],
            {"lower": 0, "upper": 8},
        )
        self.assertEqual(child["error_count_bounds"], {"lower": 0, "upper": 1})
        self.assertEqual(child["declared_peak_in_flight_cap"], 1)
        self.assertIsNone(child["observed_peak_in_flight"])

    def test_scenario_uses_exact_safe_registry_arguments(self) -> None:
        output = self.root / "out.json"
        concurrent = build_guidellm_scenario(self.spec(), output)
        backend = concurrent["spec"]["backend"]

        self.assertEqual(backend["kind"], "openai_http")
        self.assertTrue(backend["stream"])
        self.assertTrue(backend["verify"])
        self.assertFalse(backend["follow_redirects"])
        self.assertFalse(backend["validate_backend"])
        self.assertTrue(
            concurrent["spec"]["tokenizer"]["load_kwargs"]["local_files_only"]
        )
        self.assertEqual(
            concurrent["spec"]["profile"], {"kind": "concurrent", "streams": 4}
        )
        self.assertEqual(
            concurrent["spec"]["constraints"],
            [
                {"kind": "max_requests", "count": 200},
                {"kind": "max_duration", "seconds": 60.0},
                {"kind": "max_errors", "count": 1, "stopping_scope": "all"},
            ],
        )
        self.assertEqual(concurrent["spec"]["seed"], {"kind": "static", "value": 731})
        self.assertEqual(
            concurrent["spec"]["outputs"], [{"kind": "json", "path": str(output)}]
        )
        self.assertEqual(concurrent["spec"]["metrics"]["sample_size"], 0)
        self.assertNotIn("api_key", json.dumps(concurrent))

        constant = build_guidellm_scenario(
            self.spec(traffic=GuideLLMTraffic.constant(12.5, 32)), output
        )
        self.assertEqual(
            constant["spec"]["profile"],
            {"kind": "constant", "rate": 12.5, "max_concurrency": 32},
        )

    def test_non_loopback_http_is_rejected(self) -> None:
        unsafe = self.spec()
        unsafe = GuideLLMRunSpec(
            **{**unsafe.__dict__, "endpoint_url": "http://remote.example/v1"}
        )
        with self.assertRaises(GuideLLMBackendError) as raised:
            build_guidellm_scenario(unsafe, self.root / "out.json")
        self.assertEqual(raised.exception.code, "insecure_endpoint")
        overridden = replace(unsafe, allow_insecure_http=True)
        scenario = build_guidellm_scenario(overridden, self.root / "out.json")
        self.assertEqual(
            scenario["spec"]["backend"]["target"],
            "http://remote.example",
        )

    def test_only_route_equivalent_endpoint_forms_are_accepted(self) -> None:
        accepted = (
            "",
            "/",
            "/v1",
            "/v1/",
            "/v1/chat/completions",
            "/v1/chat/completions/",
        )
        for path in accepted:
            with self.subTest(accepted=path):
                spec = replace(
                    self.spec(),
                    endpoint_url=f"http://127.0.0.1:18123{path}",
                )
                scenario = build_guidellm_scenario(
                    spec, self.root / "route-output.json"
                )
                self.assertEqual(
                    scenario["spec"]["backend"]["target"],
                    "http://127.0.0.1:18123",
                )

        rejected = (
            "/chat/completions",
            "/api",
            "/api/v1",
            "/api/v1/chat/completions",
            "/v1//",
            "/v1/chat/completions//",
        )
        for path in rejected:
            with self.subTest(rejected=path):
                spec = replace(
                    self.spec(),
                    endpoint_url=f"http://127.0.0.1:18123{path}",
                )
                with self.assertRaises(GuideLLMBackendError) as raised:
                    build_guidellm_scenario(spec, self.root / "route-output.json")
                self.assertEqual(raised.exception.code, "unsupported_guidellm_route")

    def test_matrix_rejects_custom_route_before_version_or_traffic(self) -> None:
        config = replace(
            self.matrix_config(),
            endpoint=EndpointConfig(
                "http://127.0.0.1:18123/custom", "matrix-secret-key"
            ),
        )
        with self.assertRaises(GuideLLMBackendError) as raised:
            run_guidellm_matrix(
                config,
                prompt_tokens=32,
                executable=self.executable,
                environ=self.environment(),
            )
        self.assertEqual(raised.exception.code, "unsupported_guidellm_route")
        self.assertFalse(self.capture_log.exists())
        self.assertFalse(self.capture.exists())

    def test_version_verification_is_exact_and_sanitized(self) -> None:
        self.assertEqual(
            verify_guidellm_version(self.executable, environ=self.environment()),
            "0.7.3",
        )
        environment = self.environment()
        environment["FAKE_VERSION"] = "9.9.9-secret-build"
        with self.assertRaises(GuideLLMBackendError) as raised:
            verify_guidellm_version(self.executable, environ=environment)
        self.assertEqual(raised.exception.code, "version_mismatch")
        self.assertNotIn("9.9.9", str(raised.exception))

        oversized_environment = self.environment()
        oversized_environment["FAKE_VERSION_BYTES"] = "1000000"
        with self.assertRaises(GuideLLMBackendError) as oversized:
            verify_guidellm_version(
                self.executable,
                environ=oversized_environment,
            )
        self.assertEqual(oversized.exception.code, "version_check_failed")

    @unittest.skipUnless(os.name == "posix", "process-group semantics are POSIX")
    def test_completed_version_parent_cannot_leave_worker_alive(self) -> None:
        marker = self.root / "version-child-survived"
        environment = self.environment()
        environment["FAKE_VERSION_CHILD_MARKER"] = str(marker)
        self.assertEqual(
            verify_guidellm_version(self.executable, environ=environment),
            "0.7.3",
        )
        time.sleep(1.1)
        self.assertFalse(marker.exists())

    def test_runner_scrubs_env_and_imports_only_sanitized_numbers(self) -> None:
        secret = "run-secret-that-must-only-be-in-one-env-var"
        result = run_guidellm(
            self.spec(),
            api_key=secret,
            executable=self.executable,
            environ=self.environment(),
        )
        capture = json.loads(self.capture.read_text(encoding="utf-8"))

        self.assertEqual(
            capture["argv"],
            ["run", "--config", capture["argv"][2], "--disable-console"],
        )
        self.assertNotIn(secret, json.dumps(capture["argv"]))
        self.assertNotIn(secret, json.dumps(capture["scenario"]))
        self.assertEqual(capture["cwd_mode"], 0o700)
        self.assertEqual(capture["scenario_mode"], 0o600)
        self.assertEqual(capture["report_mode"], 0o600)
        self.assertFalse(Path(capture["cwd"]).exists())

        child_env = capture["interesting_env"]
        self.assertEqual(child_env[GUIDELLM_API_KEY_ENV], secret)
        self.assertEqual(child_env["GUIDELLM__LOGGING__DISABLED"], "true")
        self.assertEqual(child_env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(child_env["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(child_env["HF_DATASETS_OFFLINE"], "1")
        self.assertNotIn("HF_TOKEN", child_env)
        self.assertNotIn("HUGGING_FACE_HUB_TOKEN", child_env)
        self.assertNotIn("OPENAI_API_KEY", child_env)
        self.assertNotIn("VLLM_API_KEY", child_env)
        self.assertNotIn("GITHUB_TOKEN", child_env)
        self.assertNotIn("SSL_CERT_FILE", child_env)
        self.assertNotIn("SSL_CERT_DIR", child_env)
        self.assertNotIn("SSLKEYLOGFILE", child_env)
        self.assertEqual(
            set(child_env),
            {
                GUIDELLM_API_KEY_ENV,
                "GUIDELLM__LOGGING__DISABLED",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "HF_DATASETS_OFFLINE",
            },
        )

        self.assertFalse(result["strict_completion_validation"])
        self.assertFalse(result["decision_eligible"])
        self.assertFalse(result["golden_gate_eligible"])
        self.assertEqual(
            result["decision_ineligible_reasons"],
            list(DECISION_INELIGIBLE_REASONS),
        )
        self.assertEqual(result["request_totals"]["successful"], 200)
        self.assertEqual(result["metrics"]["request_latency"]["count"], 200)
        serialized = json.dumps(result)
        for private in (
            secret,
            "candidate-secret.example",
            "credential-that-must-not-leak",
            "private prompt text",
            "private response text",
            "private-model-name",
            "127.0.0.1",
        ):
            self.assertNotIn(private, serialized)

    def test_matrix_runs_warmups_and_blocks_with_one_version_check(self) -> None:
        environment = self.environment()
        environment["FAKE_DYNAMIC_REPORT"] = "true"
        progress = RunProgress()
        report = run_guidellm_matrix(
            self.matrix_config(),
            prompt_tokens=32,
            executable=self.executable,
            progress=progress,
            environ=environment,
        )

        self.assertEqual(report["schema_version"], "2.0")
        self.assertEqual(report["artifact_type"], "throttle_run")
        self.assertEqual(report["status"], "complete")
        self.assertFalse(report["strict_completion_validation"])
        self.assertFalse(report["decision_eligible"])
        self.assertFalse(report["golden_gate_eligible"])
        self.assertEqual(
            report["decision_ineligible_reasons"],
            list(DECISION_INELIGIBLE_REASONS),
        )
        self.assertEqual(len(report["conditions"]), 2)
        for condition in report["conditions"]:
            self.assertTrue(condition["valid"])
            self.assertFalse(condition["decision_grade"])
            self.assertEqual(condition["warmup"]["attempted"], 1)
            self.assertTrue(condition["warmup"]["metrics_discarded"])
            self.assertEqual(len(condition["blocks"]), 3)
            self.assertTrue(all(block["valid"] for block in condition["blocks"]))
            self.assertEqual(condition["request_counts"]["attempted"], 6)
            self.assertEqual(condition["diagnostic_metrics"]["completion_tokens"], 48)
            self.assertIsNone(condition["diagnostic_metrics"]["e2e_latency_ms"]["p95"])

        workload = report["manifest"]["workload"]
        self.assertEqual(workload["source"], "guidellm_synthetic_text")
        self.assertFalse(workload["supplied_prompt_jsonl_used"])
        self.assertFalse(workload["parity_with_supplied_prompt_jsonl"])
        self.assertEqual(workload["parity_claim"], "none")
        self.assertEqual(workload["measured_seed_count"], 6)
        self.assertEqual(workload["warmup_seed_count"], 2)
        self.assertFalse(workload["warmup_prompts_disjoint"])
        self.assertTrue(report["manifest"]["runtime"]["gpu_fingerprint_supplied"])
        self.assertEqual(report["run_totals"]["requests_started"], 14)
        self.assertEqual(report["run_totals"]["reserved_output_tokens"], 112)
        self.assertFalse(report["accounting_incomplete"])
        self.assertFalse(report["run_totals"]["accounting_incomplete"])
        self.assertEqual(
            report["run_totals"]["request_count_bounds"],
            {"lower": 14, "upper": 14},
        )
        self.assertEqual(
            report["run_totals"]["requested_output_token_bounds"],
            {"lower": 112, "upper": 112},
        )
        self.assertEqual(
            report["run_totals"]["error_count_bounds"],
            {"lower": 0, "upper": 0},
        )
        self.assertEqual(report["run_totals"]["declared_peak_in_flight_cap"], 4)
        self.assertIsNone(report["run_totals"]["peak_in_flight"])
        self.assertIsNone(report["run_totals"]["observed_peak_in_flight"])
        for condition in report["conditions"]:
            self.assertEqual(
                condition["declared_peak_in_flight_cap"],
                condition["condition"]["max_in_flight"],
            )
            self.assertIsNone(condition["observed_peak_in_flight"])
            for block in condition["blocks"]:
                self.assertEqual(
                    block["declared_peak_in_flight_cap"],
                    condition["condition"]["max_in_flight"],
                )
                self.assertIsNone(block["observed_peak_in_flight"])
        self.assertEqual(progress.snapshot(), report)

        invocations = [
            json.loads(line)
            for line in self.capture_log.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(sum(item["kind"] == "version" for item in invocations), 1)
        runs = [item for item in invocations if item["kind"] == "run"]
        self.assertEqual(len(runs), 8)
        self.assertEqual(
            [item["scenario"]["spec"]["seed"]["value"] for item in runs],
            [41, 42, 43, 44, 10041, 10042, 10043, 10044],
        )
        self.assertFalse(invocations[0]["api_key_present"])
        self.assertFalse(invocations[0]["ambient_credential_present"])
        for item in runs:
            self.assertEqual(
                item["interesting_env"][GUIDELLM_API_KEY_ENV],
                "matrix-secret-key",
            )

        serialized = json.dumps(report)
        for private in (
            "matrix-secret-key",
            "127.0.0.1",
            "candidate-secret.example",
            "private prompt text",
            "private response text",
        ):
            self.assertNotIn(private, serialized)

    def test_matrix_requires_gap_acknowledgement_before_version_or_traffic(
        self,
    ) -> None:
        config = replace(self.matrix_config(), guidellm_gaps_acknowledged=False)
        with self.assertRaises(GuideLLMBackendError) as raised:
            run_guidellm_matrix(
                config,
                prompt_tokens=32,
                executable=self.executable,
                environ=self.environment(),
            )
        self.assertEqual(raised.exception.code, "invalid_run_config")
        self.assertFalse(self.capture_log.exists())
        self.assertFalse(self.capture.exists())

    def test_plan_preflight_is_side_effect_free_and_needs_no_key_or_gap_ack(
        self,
    ) -> None:
        config = replace(
            self.matrix_config(),
            endpoint=EndpointConfig("http://127.0.0.1:18123/v1", ""),
            guidellm_gaps_acknowledged=False,
        )
        self.assertIsNone(preflight_guidellm_config(config))
        self.assertFalse(self.capture_log.exists())
        self.assertFalse(self.capture.exists())

        with self.assertRaises(GuideLLMBackendError) as traffic_error:
            preflight_guidellm_config(config, for_traffic=True)
        self.assertEqual(traffic_error.exception.code, "invalid_run_config")

        with self.assertRaises(GuideLLMBackendError) as stream_error:
            preflight_guidellm_config(replace(config, stream=False))
        self.assertEqual(stream_error.exception.code, "guidellm_requires_streaming")
        self.assertFalse(self.capture_log.exists())
        self.assertFalse(self.capture.exists())

    def test_traffic_fails_closed_without_posix_process_group_controls(self) -> None:
        with mock.patch.object(guidellm_module.os, "name", "nt"):
            self.assertIsNone(preflight_guidellm_config(self.matrix_config()))
            with self.assertRaises(GuideLLMBackendError) as matrix_error:
                preflight_guidellm_config(self.matrix_config(), for_traffic=True)
            self.assertEqual(
                matrix_error.exception.code,
                "guidellm_requires_posix_process_groups",
            )
            with self.assertRaises(GuideLLMBackendError) as direct_error:
                run_guidellm(
                    self.spec(),
                    api_key="private-key",
                    executable=self.executable,
                    environ=self.environment(),
                )
            self.assertEqual(
                direct_error.exception.code,
                "guidellm_requires_posix_process_groups",
            )
        self.assertFalse(self.capture_log.exists())
        self.assertFalse(self.capture.exists())

    def test_matrix_stops_after_global_error_bound_with_partial_progress(self) -> None:
        environment = self.environment()
        environment["FAKE_DYNAMIC_REPORT"] = "true"
        environment["FAKE_ERRORS"] = "1"
        progress = RunProgress()
        report = run_guidellm_matrix(
            self.matrix_config(),
            prompt_tokens=32,
            executable=self.executable,
            progress=progress,
            environ=environment,
        )

        self.assertEqual(report["status"], "stopped")
        self.assertEqual(report["stop_reason"], "max_errors")
        self.assertFalse(report["decision_eligible"])
        self.assertEqual(report["run_totals"]["requests_started"], 1)
        self.assertEqual(report["run_totals"]["errors"], 1)
        self.assertFalse(report["accounting_incomplete"])
        self.assertFalse(report["best_tested"]["available"])
        self.assertEqual(report["best_tested"]["state"], "inconclusive")
        self.assertEqual(report["best_tested"]["reason"], "partial_or_failed_run")
        self.assertEqual(progress.snapshot(), report)
        invocations = [
            json.loads(line)
            for line in self.capture_log.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(sum(item["kind"] == "run" for item in invocations), 1)

    def test_finalize_suppresses_prior_best_and_rechecks_final_limits(self) -> None:
        config = self.matrix_config()

        def condition() -> dict[str, object]:
            return {
                "valid": True,
                "blocks": [{}, {}, {}],
                "diagnostic_metrics": {"completion_tokens": 8},
                "metrics": {"output_tokens_per_second": 1.0},
                "condition": config.conditions[0].public_dict(),
            }

        partial = {"conditions": [condition()]}
        partial_budget = guidellm_module._GuideLLMMatrixBudget(
            config=config,
            started=time.monotonic(),
        )
        guidellm_module._finalize_matrix_report(
            partial,
            config,
            partial_budget,
            status="stopped",
            stop_reason="max_errors",
        )
        self.assertFalse(partial["best_tested"]["available"])
        self.assertEqual(partial["best_tested"]["reason"], "partial_or_failed_run")

        complete_conditions = [condition(), condition()]
        complete_conditions[1]["condition"] = config.conditions[1].public_dict()
        over_time = {"conditions": complete_conditions}
        over_time_budget = guidellm_module._GuideLLMMatrixBudget(
            config=config,
            started=time.monotonic() - config.limits.max_elapsed_seconds - 0.1,
        )
        guidellm_module._finalize_matrix_report(
            over_time,
            config,
            over_time_budget,
            status="complete",
            stop_reason=None,
        )
        self.assertEqual(over_time["status"], "stopped")
        self.assertEqual(over_time["stop_reason"], "max_elapsed_time")
        self.assertFalse(over_time["best_tested"]["available"])
        elapsed = over_time["run_totals"]["elapsed_seconds"]
        expected_cost = config.cost.total_hourly_rate * elapsed / 3_600.0
        self.assertEqual(over_time["cost_summary"]["total_cost"], expected_cost)

    def test_matrix_cancellation_updates_sanitized_partial_progress(self) -> None:
        environment = self.environment()
        progress = RunProgress()
        with mock.patch(
            "throttle.guidellm_backend._run_matrix_child",
            side_effect=KeyboardInterrupt,
        ):
            with self.assertRaises(KeyboardInterrupt):
                run_guidellm_matrix(
                    self.matrix_config(),
                    prompt_tokens=32,
                    executable=self.executable,
                    progress=progress,
                    environ=environment,
                )
        partial = progress.snapshot()
        self.assertIsNotNone(partial)
        self.assertEqual(partial["status"], "cancelled")
        self.assertEqual(partial["stop_reason"], "cancelled_by_user")
        self.assert_first_warmup_accounting_is_uncertain(partial)
        serialized = json.dumps(partial)
        self.assertNotIn("matrix-secret-key", serialized)
        self.assertNotIn("127.0.0.1", serialized)

    def test_matrix_nonzero_child_exit_has_conservative_accounting(self) -> None:
        environment = self.environment()
        environment["FAKE_EXIT"] = "7"
        report = run_guidellm_matrix(
            self.matrix_config(),
            prompt_tokens=32,
            executable=self.executable,
            environ=environment,
        )
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["stop_reason"], "process_failed")
        self.assertEqual(report["operational_error"], {"code": "process_failed"})
        self.assert_first_warmup_accounting_is_uncertain(report)

    def test_matrix_schema_failure_has_conservative_accounting(self) -> None:
        malformed = report_fixture()
        malformed["metadata"]["version"] = 3
        self.raw_report.write_text(json.dumps(malformed), encoding="utf-8")
        report = run_guidellm_matrix(
            self.matrix_config(),
            prompt_tokens=32,
            executable=self.executable,
            environ=self.environment(),
        )
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["stop_reason"], "unsupported_report_schema")
        self.assertEqual(
            report["operational_error"],
            {"code": "unsupported_report_schema"},
        )
        self.assert_first_warmup_accounting_is_uncertain(report)

    def test_matrix_timeout_has_conservative_accounting(self) -> None:
        environment = self.environment()
        environment["FAKE_SLEEP"] = "5"
        config = replace(
            self.matrix_config(),
            limits=replace(
                self.matrix_config().limits,
                max_elapsed_seconds=0.6,
            ),
        )
        report = run_guidellm_matrix(
            config,
            prompt_tokens=32,
            executable=self.executable,
            environ=environment,
        )
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["stop_reason"], "process_timeout")
        self.assertEqual(report["operational_error"], {"code": "process_timeout"})
        self.assert_first_warmup_accounting_is_uncertain(report)

    def test_parser_fails_closed_on_contract_drift_and_non_numeric_data(self) -> None:
        mutations = []

        wrong_schema = report_fixture()
        wrong_schema["metadata"]["version"] = 3
        mutations.append(wrong_schema)

        wrong_version = report_fixture()
        wrong_version["metadata"]["guidellm_version"] = "0.7.4"
        mutations.append(wrong_version)

        non_numeric = report_fixture()
        non_numeric["benchmarks"][0]["metrics"]["request_latency"]["successful"][
            "mean"
        ] = "https://secret.example"
        mutations.append(non_numeric)

        extreme_numeric = report_fixture()
        extreme_numeric["benchmarks"][0]["metrics"]["request_latency"]["successful"][
            "mean"
        ] = 1e308
        mutations.append(extreme_numeric)

        inconsistent = report_fixture()
        inconsistent["benchmarks"][0]["scheduler_metrics"]["requests_made"] = {
            "successful": 199,
            "errored": 0,
            "incomplete": 0,
            "total": 199,
        }
        mutations.append(inconsistent)

        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                path = self.root / f"bad-{index}.json"
                path.write_text(json.dumps(mutation), encoding="utf-8")
                with self.assertRaises(GuideLLMBackendError) as raised:
                    parse_guidellm_report(path)
                self.assertNotIn("secret.example", str(raised.exception))

    def test_zero_and_subnormal_durations_fail_closed(self) -> None:
        for index, duration in enumerate((0.0, 5e-324)):
            with self.subTest(duration=duration):
                environment = self.environment()
                environment["FAKE_DYNAMIC_REPORT"] = "true"
                environment["FAKE_DURATION"] = repr(duration)
                report = run_guidellm_matrix(
                    self.matrix_config(),
                    prompt_tokens=32,
                    executable=self.executable,
                    environ=environment,
                )
                expected = (
                    "invalid_report_duration" if index == 0 else "invalid_report_number"
                )
                self.assertEqual(report["status"], "failed")
                self.assertEqual(report["stop_reason"], expected)
                self.assertEqual(report["accounting_incomplete"], index == 0)
                json.dumps(report, allow_nan=False)

    def test_invalid_json_after_traffic_has_conservative_accounting(self) -> None:
        self.raw_report.write_bytes(b'{"metadata":{"version":' + b"9" * 5_000 + b"}}")
        environment = self.environment()
        environment["FAKE_COPY_RAW_REPORT"] = "true"
        report = run_guidellm_matrix(
            self.matrix_config(),
            prompt_tokens=32,
            executable=self.executable,
            environ=environment,
        )
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["stop_reason"], "invalid_report_json")
        self.assertTrue(report["accounting_incomplete"])
        self.assertIsNone(report["run_totals"]["requests_started"])

        self.raw_report.write_text(json.dumps(report_fixture()), encoding="utf-8")
        with mock.patch(
            "throttle.guidellm_backend.json.loads",
            side_effect=RecursionError,
        ):
            recursive = run_guidellm_matrix(
                self.matrix_config(),
                prompt_tokens=32,
                executable=self.executable,
                environ=self.environment(),
            )
        self.assertEqual(recursive["status"], "failed")
        self.assertEqual(recursive["stop_reason"], "invalid_report_json")
        self.assertTrue(recursive["accounting_incomplete"])

    def test_extreme_child_numbers_become_serializable_sanitized_failure(self) -> None:
        extreme = report_fixture()
        extreme["benchmarks"][0]["metrics"]["request_latency"]["successful"][
            "percentiles"
        ]["p95"] = 1e308
        self.raw_report.write_text(json.dumps(extreme), encoding="utf-8")
        report = run_guidellm_matrix(
            self.matrix_config(),
            prompt_tokens=32,
            executable=self.executable,
            environ=self.environment(),
        )
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["stop_reason"], "invalid_report_number")
        json.dumps(report, allow_nan=False)

    def test_runner_rejects_reported_output_above_requested_token_cap(self) -> None:
        oversized = report_fixture()
        oversized["benchmarks"][0]["metrics"]["output_token_count"]["successful"][
            "max"
        ] = 129.0
        self.raw_report.write_text(json.dumps(oversized), encoding="utf-8")
        with self.assertRaises(GuideLLMBackendError) as raised:
            run_guidellm(
                self.spec(),
                api_key="private-key",
                executable=self.executable,
                environ=self.environment(),
            )
        self.assertEqual(raised.exception.code, "child_exceeded_token_limit")

    @unittest.skipUnless(os.name == "posix", "process-group semantics are POSIX")
    def test_timeout_kills_worker_group_and_removes_private_directory(self) -> None:
        marker = self.root / "child-survived"
        environment = self.environment()
        environment["FAKE_SLEEP"] = "5"
        environment["FAKE_CHILD_MARKER"] = str(marker)
        with self.assertRaises(GuideLLMBackendError) as raised:
            run_guidellm(
                self.spec(process_timeout_seconds=0.25),
                api_key="timeout-secret",
                executable=self.executable,
                environ=environment,
            )
        self.assertEqual(raised.exception.code, "process_timeout")
        capture = json.loads(self.capture.read_text(encoding="utf-8"))
        self.assertFalse(Path(capture["cwd"]).exists())
        time.sleep(1.1)
        self.assertFalse(marker.exists())

    @unittest.skipUnless(os.name == "posix", "process-group semantics are POSIX")
    def test_completed_benchmark_parent_cannot_leave_worker_alive(self) -> None:
        marker = self.root / "completed-parent-child-survived"
        environment = self.environment()
        environment["FAKE_CHILD_MARKER"] = str(marker)
        environment["FAKE_PARENT_EXIT_AFTER_FORK"] = "true"
        with self.assertRaises(GuideLLMBackendError) as raised:
            run_guidellm(
                self.spec(),
                api_key="private-key",
                executable=self.executable,
                environ=environment,
            )
        self.assertEqual(raised.exception.code, "invalid_report_size")
        time.sleep(1.1)
        self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
