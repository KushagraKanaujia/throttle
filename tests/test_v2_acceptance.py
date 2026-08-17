from __future__ import annotations

import asyncio
import gzip
import json
import os
import socket
import unittest
from dataclasses import replace
from unittest.mock import patch

import httpx

from throttle.benchmark import (
    RunProgress,
    build_plan,
    normalize_chat_completions_url,
    run_native,
    validate_config,
)
from throttle.compare import compare_reports
from throttle.models import (
    CostModel,
    EndpointConfig,
    LoadCondition,
    RunConfig,
    SafetyLimits,
)


MEASURED_PROMPTS = (({"role": "user", "content": "private measured prompt"},),)
WARMUP_PROMPTS = (({"role": "user", "content": "private warmup prompt"},),)
DEFAULT_CONDITIONS = (
    LoadCondition("closed_loop", 1.0, 1),
    LoadCondition("closed_loop", 4.0, 4),
    LoadCondition("closed_loop", 8.0, 8),
)


def _limits(**overrides: object) -> SafetyLimits:
    values: dict[str, object] = {
        "max_requests": 1_000,
        "max_tokens_per_request": 32,
        "max_total_requested_tokens": 32_000,
        "max_elapsed_seconds": 5.0,
        "max_errors": 1,
        "max_concurrency": 16,
        "max_response_bytes": 100_000,
        "max_estimated_spend": 3.0,
    }
    values.update(overrides)
    return SafetyLimits(**values)  # type: ignore[arg-type]


def _config(
    *,
    mode: str = "smoke",
    endpoint_url: str = "https://private-endpoint.example/v1",
    api_key: str = "private-api-key",
    conditions: tuple[LoadCondition, ...] = (LoadCondition("closed_loop", 1.0, 1),),
    blocks: int | None = None,
    requests_per_block: int | None = 1,
    warmups: int = 0,
    stream: bool = False,
    limits: SafetyLimits | None = None,
    allow_insecure_http: bool = False,
) -> RunConfig:
    return RunConfig(
        mode=mode,  # type: ignore[arg-type]
        backend="native",
        model="model-a",
        endpoint=EndpointConfig(endpoint_url, api_key),
        cost=CostModel(
            kind="dedicated_hourly",
            total_hourly_rate=0.25,
            gpu_count=1,
        ),
        limits=limits or _limits(),
        max_tokens=8,
        conditions=conditions,
        blocks=(1 if mode == "smoke" else 3) if blocks is None else blocks,
        requests_per_block=requests_per_block,
        block_duration_seconds=None,
        warmup_requests_per_condition=warmups,
        request_timeout_seconds=1.0,
        stream=stream,
        cache_policy="disabled",
        model_revision="a" * 40,
        image_digest="example/image@sha256:" + "b" * 64,
        gpu="test-gpu",
        gpu_fingerprint="test-gpu-fingerprint",
        cuda_version="test-cuda",
        driver_version="test-driver",
        server_version="test-server",
        engine_flags_provenance="runtime_verified",
        allow_insecure_http=allow_insecure_http,
        evidence_source="synthetic_validation",
    )


def _completion(
    *,
    response_text: str = "private generated response",
    finish_reason: object = "stop",
    completion_tokens: object = 4,
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": completion_tokens,
            },
        },
    )


def _serialized(report: object) -> str:
    return json.dumps(report, sort_keys=True, allow_nan=False)


def _all_keys(value: object) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            keys.append(str(key))
            keys.extend(_all_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.extend(_all_keys(child))
    return keys


class PlanAndModeAcceptanceTests(unittest.IsolatedAsyncioTestCase):
    def test_plan_is_zero_traffic_and_does_not_require_a_key(self) -> None:
        config = _config(
            api_key="",
            conditions=DEFAULT_CONDITIONS,
            requests_per_block=8,
            warmups=1,
        )
        with (
            patch.object(
                socket,
                "getaddrinfo",
                side_effect=AssertionError("plan attempted DNS"),
            ) as dns,
            patch(
                "throttle.benchmark.httpx.AsyncClient",
                side_effect=AssertionError("plan constructed an HTTP client"),
            ) as client,
            patch.object(
                os.environ,
                "get",
                side_effect=AssertionError("plan resolved an environment key"),
            ) as environment,
        ):
            plan = build_plan(config, MEASURED_PROMPTS, WARMUP_PROMPTS)

        self.assertFalse(plan["traffic_sent"])
        self.assertEqual(plan["request_count"]["exact"], 27)
        self.assertEqual(plan["requested_output_token_ceiling"], 27 * 8)
        self.assertEqual(
            plan["destination"]["normalized_url"],
            "https://private-endpoint.example/v1/chat/completions",
        )
        self.assertTrue(
            plan["privacy"]["report_omits_urls_credentials_prompts_and_responses"]
        )
        dns.assert_not_called()
        client.assert_not_called()
        environment.assert_not_called()

    async def test_default_smoke_is_exactly_27_calls_and_never_decision_eligible(
        self,
    ) -> None:
        calls = 0

        def handler(_: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return _completion()

        config = _config(
            conditions=DEFAULT_CONDITIONS,
            requests_per_block=8,
            warmups=1,
        )
        report = await run_native(
            config,
            MEASURED_PROMPTS,
            WARMUP_PROMPTS,
            transport=httpx.MockTransport(handler),
        )

        self.assertEqual(calls, 27)
        self.assertEqual(report["mode"], "smoke")
        self.assertEqual(report["status"], "complete")
        self.assertFalse(report["decision_eligible"])
        self.assertEqual(report["best_tested"]["field"], "best_tested_concurrency")
        self.assertEqual(report["best_tested"]["state"], "not_applicable_smoke")
        self.assertFalse(report["best_tested"]["optimum_found"])
        self.assertNotIn("recommendation", _all_keys(report))

    def test_mode_contract_rejects_smoke_repeats_and_short_benchmark(self) -> None:
        with self.assertRaisesRegex(ValueError, "smoke mode always uses one"):
            validate_config(_config(mode="smoke", blocks=3))
        with self.assertRaisesRegex(ValueError, "at least three repeated blocks"):
            validate_config(_config(mode="benchmark", blocks=2))

    async def test_underpowered_benchmark_is_explicitly_inconclusive(self) -> None:
        config = _config(mode="benchmark", requests_per_block=1, blocks=3)
        report = await run_native(
            config,
            MEASURED_PROMPTS,
            WARMUP_PROMPTS,
            transport=httpx.MockTransport(lambda _: _completion()),
        )

        condition = report["conditions"][0]
        self.assertEqual(report["status"], "complete")
        self.assertFalse(condition["decision_grade"])
        self.assertIn(
            "measurement_floor_not_met", condition["decision_ineligible_reasons"]
        )
        self.assertFalse(report["decision_eligible"])
        self.assertEqual(report["best_tested"]["state"], "inconclusive")
        self.assertFalse(report["best_tested"]["optimum_found"])

    async def test_saved_smoke_compare_fails_closed_without_network(self) -> None:
        smoke = await run_native(
            _config(),
            MEASURED_PROMPTS,
            WARMUP_PROMPTS,
            transport=httpx.MockTransport(lambda _: _completion()),
        )
        with patch.object(
            socket,
            "getaddrinfo",
            side_effect=AssertionError("compare attempted DNS"),
        ) as dns:
            comparison = compare_reports(smoke, smoke)

        self.assertEqual(comparison["status"], "incompatible")
        self.assertFalse(comparison["decision_eligible"])
        self.assertIn(
            "baseline_smoke_or_nonbenchmark_report",
            comparison["compatibility"]["reasons"],
        )
        dns.assert_not_called()

    async def test_reordered_measured_prompts_are_not_disjoint_warmups(self) -> None:
        first = ({"role": "user", "content": "private first prompt"},)
        second = ({"role": "user", "content": "private second prompt"},)
        measured = (first, second)
        reordered_warmups = (second, first)
        report = await run_native(
            _config(),
            measured,
            reordered_warmups,
            transport=httpx.MockTransport(lambda _: _completion()),
        )

        workload = report["manifest"]["workload"]
        self.assertTrue(workload["warmup_is_separate"])
        self.assertFalse(workload["warmup_prompts_disjoint"])
        serialized = _serialized(report)
        self.assertNotIn("private first prompt", serialized)
        self.assertNotIn("private second prompt", serialized)


class TransportAndValidationAcceptanceTests(unittest.IsolatedAsyncioTestCase):
    def test_direct_config_rejects_non_boolean_safety_switches_before_traffic(
        self,
    ) -> None:
        invalid_values = (
            ("stream", 1),
            ("allow_insecure_http", "false"),
            ("allow_unknown_cost", None),
            ("guidellm_gaps_acknowledged", 0),
        )
        for field, value in invalid_values:
            with (
                self.subTest(field=field),
                patch(
                    "throttle.benchmark.httpx.AsyncClient",
                    side_effect=AssertionError("invalid config attempted traffic"),
                ) as client,
            ):
                config = replace(_config(), **{field: value})
                with self.assertRaisesRegex(
                    ValueError,
                    rf"^{field} must be a boolean$",
                ):
                    validate_config(config)
                client.assert_not_called()

    def test_http_is_limited_to_exact_loopback_without_override(self) -> None:
        for value in (
            "http://localhost:8000",
            "http://localhost.:8000/v1",
            "http://127.0.0.2:8000",
            "http://[::1]:8000/v1",
        ):
            with self.subTest(value=value):
                self.assertIn(
                    "/chat/completions",
                    normalize_chat_completions_url(value),
                )

        for value in (
            "http://example.test/v1",
            "http://localhost.evil/v1",
            "http://0.0.0.0:8000/v1",
            "http://[::]:8000/v1",
            "http://user:password@localhost:8000/v1",
            "https://example.test/v1?secret=value",
            "https://example.test/v1#fragment",
            " https://example.test/v1",
            "https://example.test/v1\x1b[31m",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_chat_completions_url(value)

        self.assertEqual(
            normalize_chat_completions_url(
                "http://example.test/v1", allow_insecure_http=True
            ),
            "http://example.test/v1/chat/completions",
        )

    async def test_native_client_disables_environment_proxies_and_redirects(
        self,
    ) -> None:
        hostile_proxy_environment = {
            "HTTP_PROXY": "http://proxy-secret.example:9999",
            "HTTPS_PROXY": "http://proxy-secret.example:9999",
            "ALL_PROXY": "http://proxy-secret.example:9999",
            "http_proxy": "http://proxy-secret.example:9999",
            "https_proxy": "http://proxy-secret.example:9999",
            "all_proxy": "http://proxy-secret.example:9999",
            "NO_PROXY": "",
            "no_proxy": "",
        }
        with (
            patch.dict(os.environ, hostile_proxy_environment, clear=False),
            patch(
                "throttle.benchmark.httpx.AsyncClient",
                wraps=httpx.AsyncClient,
            ) as client_class,
        ):
            report = await run_native(
                _config(),
                MEASURED_PROMPTS,
                WARMUP_PROMPTS,
                transport=httpx.MockTransport(lambda _: _completion()),
            )

        self.assertEqual(report["status"], "complete")
        self.assertFalse(client_class.call_args.kwargs["trust_env"])
        self.assertFalse(client_class.call_args.kwargs["follow_redirects"])
        self.assertFalse(report["manifest"]["safety"]["ambient_proxy_environment_used"])

    async def test_native_rejects_content_coding_before_decoded_body_iteration(
        self,
    ) -> None:
        for stream in (False, True):
            with self.subTest(stream=stream):
                observed_accept_encoding: list[str | None] = []
                plain = (
                    b'data: {"choices":[]}\n\ndata: [DONE]\n\n'
                    if stream
                    else json.dumps(
                        {
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": "private compressed response",
                                    },
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 1,
                                "completion_tokens": 1,
                            },
                        }
                    ).encode("utf-8")
                )

                def compressed(request: httpx.Request) -> httpx.Response:
                    observed_accept_encoding.append(
                        request.headers.get("accept-encoding")
                    )
                    return httpx.Response(
                        200,
                        content=gzip.compress(plain),
                        headers={"Content-Encoding": "gzip"},
                    )

                report = await run_native(
                    _config(stream=stream),
                    MEASURED_PROMPTS,
                    WARMUP_PROMPTS,
                    transport=httpx.MockTransport(compressed),
                )
                self.assertEqual(observed_accept_encoding, ["identity"])
                self.assertEqual(
                    report["conditions"][0]["request_counts"]["error_counts"],
                    {"unsupported_content_encoding": 1},
                )

    async def test_nonstream_completion_shape_is_validated_fail_closed(self) -> None:
        malicious_finish_reason = "https://private-finish.example/credential"
        invalid_payloads = {
            "missing_or_empty_choices": {"choices": [], "usage": {}},
            "choice_not_object": {"choices": ["bad"], "usage": {}},
            "missing_finish_reason": {
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "text"},
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
            "missing_assistant_message": {
                "choices": [{"index": 0, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
            "empty_completion_output": {
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": ""},
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
            "invalid_completion_tokens": {
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "text"},
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": True},
            },
            "error_response_shape": {
                "error": {"message": "private raw server error"},
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "text"},
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
            "unsupported_finish_reason": {
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": malicious_finish_reason,
                        },
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        }
        for expected_code, payload in invalid_payloads.items():
            with self.subTest(expected_code=expected_code):
                report = await run_native(
                    _config(),
                    MEASURED_PROMPTS,
                    WARMUP_PROMPTS,
                    transport=httpx.MockTransport(
                        lambda _, payload=payload: httpx.Response(200, json=payload)
                    ),
                )
                errors = report["conditions"][0]["request_counts"]["error_counts"]
                self.assertEqual(errors, {expected_code: 1})
                self.assertEqual(report["status"], "stopped")
                self.assertFalse(report["decision_eligible"])
                self.assertNotIn(malicious_finish_reason, _serialized(report))

    async def test_native_response_json_rejects_nonfinite_and_duplicate_keys(
        self,
    ) -> None:
        hostile = "https://private-json-response.example/credential"
        valid_choice = (
            '{"index":0,"message":{"role":"assistant","content":"text"},'
            '"finish_reason":"stop"}'
        )
        nonstream_cases = (
            (
                "nan",
                '{"choices":['
                + valid_choice
                + '],"usage":{"prompt_tokens":1,"completion_tokens":NaN},'
                '"hostile":"' + hostile + '"}',
            ),
            (
                "infinity",
                '{"choices":['
                + valid_choice
                + '],"usage":{"prompt_tokens":1,"completion_tokens":Infinity},'
                '"hostile":"' + hostile + '"}',
            ),
            (
                "duplicate_key",
                '{"choices":['
                + valid_choice
                + '],"choices":['
                + valid_choice
                + '],"usage":{"prompt_tokens":1,"completion_tokens":1},'
                '"hostile":"' + hostile + '"}',
            ),
        )
        for name, body in nonstream_cases:
            with self.subTest(path="nonstream", name=name):
                report = await run_native(
                    _config(),
                    MEASURED_PROMPTS,
                    WARMUP_PROMPTS,
                    transport=httpx.MockTransport(
                        lambda _, body=body: httpx.Response(
                            200,
                            content=body.encode("utf-8"),
                        )
                    ),
                )
                condition = report["conditions"][0]
                self.assertEqual(
                    condition["request_counts"]["error_counts"],
                    {"invalid_json": 1},
                )
                self.assertFalse(condition["valid"])
                serialized = _serialized(report)
                self.assertNotIn(hostile, serialized)
                self.assertNotIn("private-json-response.example", serialized)

        sse_cases = (
            (
                "nan",
                '{"choices":[],"usage":NaN,"hostile":"' + hostile + '"}',
            ),
            (
                "infinity",
                '{"choices":[],"usage":Infinity,"hostile":"' + hostile + '"}',
            ),
            (
                "duplicate_key",
                '{"choices":[],"choices":[],"hostile":"' + hostile + '"}',
            ),
        )
        for name, event in sse_cases:
            with self.subTest(path="sse", name=name):
                body = f"data: {event}\n\ndata: [DONE]\n\n"
                report = await run_native(
                    _config(stream=True),
                    MEASURED_PROMPTS,
                    WARMUP_PROMPTS,
                    transport=httpx.MockTransport(
                        lambda _, body=body: httpx.Response(
                            200,
                            content=body.encode("utf-8"),
                        )
                    ),
                )
                condition = report["conditions"][0]
                self.assertEqual(
                    condition["request_counts"]["error_counts"],
                    {"malformed_stream_json": 1},
                )
                self.assertFalse(condition["valid"])
                serialized = _serialized(report)
                self.assertNotIn(hostile, serialized)
                self.assertNotIn("private-json-response.example", serialized)

    async def test_stream_requires_terminal_shape_and_never_persists_output(
        self,
    ) -> None:
        private_response = "private streamed generated output"
        events = [
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
                        "delta": {"content": private_response},
                        "finish_reason": None,
                    }
                ]
            },
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {
                "choices": [],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4},
            },
        ]
        body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
        body += "data: [DONE]\n\n"
        report = await run_native(
            _config(stream=True),
            MEASURED_PROMPTS,
            WARMUP_PROMPTS,
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, content=body.encode("utf-8"))
            ),
        )

        condition = report["conditions"][0]
        self.assertTrue(condition["valid"])
        self.assertEqual(condition["request_counts"]["valid"], 1)
        self.assertEqual(
            condition["metrics"]["itl_ms"]["unavailable_reason"],
            "native SSE chunks do not prove token boundaries",
        )
        self.assertTrue(
            condition["metrics"]["inter_chunk_latency_ms"]["not_equivalent_to_itl"]
        )
        self.assertEqual(condition["metrics"]["tpot_ms"]["count"], 0)
        self.assertIsNone(condition["metrics"]["tpot_ms"]["mean"])
        self.assertIsNone(condition["metrics"]["tpot_ms"]["p50"])
        self.assertIsNone(condition["metrics"]["tpot_ms"]["p95"])
        self.assertNotIn(private_response, _serialized(report))

        missing_done = body.replace("data: [DONE]\n\n", "")
        invalid = await run_native(
            _config(stream=True),
            MEASURED_PROMPTS,
            WARMUP_PROMPTS,
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, content=missing_done.encode("utf-8"))
            ),
        )
        self.assertEqual(
            invalid["conditions"][0]["request_counts"]["error_counts"],
            {"stream_missing_done": 1},
        )

    async def test_stream_state_machine_rejects_out_of_order_and_typed_events(
        self,
    ) -> None:
        hostile = "https://private-stream-state.example/credential"

        def choice(
            delta: dict[str, object], finish_reason: object = None
        ) -> dict[str, object]:
            return {
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": finish_reason,
                    }
                ]
            }

        role = choice({"role": "assistant"})
        output = choice({"content": hostile})
        finish = choice({}, "stop")
        usage = {
            "choices": [],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4},
        }
        usage_with_choice = {
            "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4},
        }
        cases = (
            (
                "usage_before_finish",
                "stream_usage_before_finish",
                [role, output, usage],
            ),
            (
                "duplicate_usage",
                "duplicate_stream_usage",
                [role, output, finish, usage, usage],
            ),
            (
                "choice_after_finish",
                "stream_choice_after_finish",
                [role, output, finish, choice({})],
            ),
            (
                "output_after_finish",
                "stream_choice_after_finish",
                [role, output, finish, choice({"content": hostile})],
            ),
            (
                "usage_with_choice_data",
                "stream_usage_with_choice_data",
                [role, output, finish, usage_with_choice],
            ),
            (
                "empty_event_without_usage",
                "empty_stream_event_without_usage",
                [role, output, {"choices": []}],
            ),
            (
                "repeated_finish",
                "stream_choice_after_finish",
                [role, output, finish, choice({}, "stop")],
            ),
            (
                "conflicting_finish",
                "stream_choice_after_finish",
                [role, output, finish, choice({}, "length")],
            ),
            (
                "tool_call_finish_for_text_only_workload",
                "unsupported_finish_reason",
                [role, output, choice({}, "tool_calls")],
            ),
            (
                "non_string_content",
                "invalid_stream_content_type",
                [role, choice({"content": {"hostile": hostile}})],
            ),
            (
                "non_string_reasoning",
                "invalid_stream_reasoning_type",
                [role, choice({"reasoning_content": [hostile]})],
            ),
        )

        for name, expected_code, events in cases:
            with self.subTest(name=name):
                body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
                body += "data: [DONE]\n\n"
                report = await run_native(
                    _config(stream=True),
                    MEASURED_PROMPTS,
                    WARMUP_PROMPTS,
                    transport=httpx.MockTransport(
                        lambda _, body=body: httpx.Response(
                            200,
                            content=body.encode("utf-8"),
                        )
                    ),
                )

                condition = report["conditions"][0]
                self.assertEqual(
                    condition["request_counts"]["error_counts"],
                    {expected_code: 1},
                )
                self.assertFalse(condition["valid"])
                self.assertFalse(condition["decision_grade"])
                self.assertEqual(report["status"], "stopped")
                self.assertFalse(report["decision_eligible"])
                serialized = _serialized(report)
                self.assertNotIn(hostile, serialized)
                self.assertNotIn("private-stream-state.example", serialized)

    async def test_response_size_is_enforced_before_report_persistence(self) -> None:
        response_text = "private oversized output " * 20
        config = _config(limits=_limits(max_response_bytes=64))
        report = await run_native(
            config,
            MEASURED_PROMPTS,
            WARMUP_PROMPTS,
            transport=httpx.MockTransport(
                lambda _: _completion(response_text=response_text)
            ),
        )

        self.assertEqual(
            report["conditions"][0]["request_counts"]["error_counts"],
            {"response_too_large": 1},
        )
        self.assertNotIn(response_text, _serialized(report))


class LimitsCancellationAndSanitizationTests(unittest.IsolatedAsyncioTestCase):
    async def test_planned_request_and_token_caps_fail_before_traffic(self) -> None:
        calls = 0

        def handler(_: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return _completion()

        request_limited = _config(
            requests_per_block=2,
            limits=_limits(max_requests=1),
        )
        with self.assertRaisesRegex(ValueError, "exceed max_requests"):
            await run_native(
                request_limited,
                MEASURED_PROMPTS,
                WARMUP_PROMPTS,
                transport=httpx.MockTransport(handler),
            )

        token_limited = _config(
            requests_per_block=2,
            limits=_limits(max_total_requested_tokens=8),
        )
        with self.assertRaisesRegex(ValueError, "exceed the total token limit"):
            await run_native(
                token_limited,
                MEASURED_PROMPTS,
                WARMUP_PROMPTS,
                transport=httpx.MockTransport(handler),
            )
        self.assertEqual(calls, 0)

    async def test_first_error_stops_cleanly_with_sanitized_partial_evidence(
        self,
    ) -> None:
        calls = 0
        secret_exception = (
            "https://private-endpoint.example/v1 private-api-key "
            "private measured prompt"
        )

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            raise httpx.ConnectError(secret_exception, request=request)

        report = await run_native(
            _config(requests_per_block=5),
            MEASURED_PROMPTS,
            WARMUP_PROMPTS,
            transport=httpx.MockTransport(handler),
        )

        self.assertEqual(calls, 1)
        self.assertEqual(report["status"], "stopped")
        self.assertEqual(report["stop_reason"], "max_errors")
        self.assertEqual(report["run_totals"]["requests_started"], 1)
        self.assertEqual(report["run_totals"]["errors"], 1)
        self.assertFalse(report["decision_eligible"])
        self.assertNotIn(secret_exception, _serialized(report))

    async def test_elapsed_limit_stops_and_persists_a_partial_report(self) -> None:
        started = asyncio.Event()

        async def handler(_: httpx.Request) -> httpx.Response:
            started.set()
            await asyncio.sleep(10)
            return _completion()

        config = _config(
            requests_per_block=2,
            limits=_limits(max_elapsed_seconds=0.03),
        )
        report = await asyncio.wait_for(
            run_native(
                config,
                MEASURED_PROMPTS,
                WARMUP_PROMPTS,
                transport=httpx.MockTransport(handler),
            ),
            timeout=1.0,
        )

        self.assertTrue(started.is_set())
        self.assertEqual(report["status"], "stopped")
        self.assertEqual(report["stop_reason"], "max_elapsed_time")
        self.assertFalse(report["decision_eligible"])
        self.assertGreaterEqual(report["run_totals"]["errors"], 1)
        self.assertFalse(report["best_tested"]["available"])
        self.assertEqual(report["best_tested"]["state"], "inconclusive")
        self.assertEqual(
            report["best_tested"]["reason"],
            "partial_or_failed_run",
        )
        self.assertFalse(report["best_tested"]["optimum_found"])

    async def test_cancellation_updates_sanitized_progress_snapshot(self) -> None:
        request_started = asyncio.Event()
        never_finishes = asyncio.Event()
        progress = RunProgress()

        async def handler(_: httpx.Request) -> httpx.Response:
            request_started.set()
            await never_finishes.wait()
            return _completion()

        task = asyncio.create_task(
            run_native(
                _config(requests_per_block=2),
                MEASURED_PROMPTS,
                WARMUP_PROMPTS,
                transport=httpx.MockTransport(handler),
                progress=progress,
            )
        )
        await asyncio.wait_for(request_started.wait(), timeout=1.0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        report = progress.snapshot()
        self.assertIsNotNone(report)
        assert report is not None
        self.assertEqual(report["status"], "cancelled")
        self.assertEqual(report["stop_reason"], "cancelled_by_user")
        self.assertFalse(report["decision_eligible"])
        self.assertEqual(report["run_totals"]["requests_started"], 1)
        self.assertEqual(report["run_totals"]["requests_cancelled"], 1)
        self.assertEqual(report["run_totals"]["requests_in_flight"], 0)
        self.assertFalse(report["best_tested"]["available"])
        self.assertEqual(report["best_tested"]["state"], "inconclusive")
        self.assertEqual(
            report["best_tested"]["reason"],
            "partial_or_failed_run",
        )
        self.assertFalse(report["best_tested"]["optimum_found"])
        for forbidden in (
            "private-endpoint.example",
            "private-api-key",
            "private measured prompt",
            "private warmup prompt",
            "private generated response",
        ):
            self.assertNotIn(forbidden, _serialized(report))

    async def test_every_report_surface_omits_url_key_prompt_and_response(self) -> None:
        private_response = "private generated response"
        report = await run_native(
            _config(
                allow_insecure_http=True, endpoint_url="http://private-host.example/v1"
            ),
            MEASURED_PROMPTS,
            WARMUP_PROMPTS,
            transport=httpx.MockTransport(
                lambda _: _completion(response_text=private_response)
            ),
        )
        serialized = _serialized(report)
        for forbidden in (
            "http://private-host.example/v1",
            "private-host.example",
            "private-api-key",
            "private measured prompt",
            "private warmup prompt",
            private_response,
            "Authorization",
            "Bearer",
        ):
            self.assertNotIn(forbidden, serialized)
        keys = _all_keys(report)
        self.assertNotIn("messages", keys)
        self.assertNotIn("content", keys)
        self.assertTrue(report["manifest"]["safety"]["overrides"]["insecure_http"])

    def test_engine_flag_metadata_rejects_secret_or_url_bearing_values(self) -> None:
        unsafe_configs = (
            replace(_config(), engine_flags=(("api_key", "private-api-key"),)),
            replace(
                _config(),
                engine_flags=(("served-model-name", "https://private-host.example"),),
            ),
            replace(
                _config(),
                engine_flags=(("custom-header", "Bearer private-api-key"),),
            ),
        )
        for config in unsafe_configs:
            with self.subTest(flags=config.engine_flags), self.assertRaises(ValueError):
                validate_config(config)

        validate_config(
            replace(
                _config(),
                engine_flags=(("max-num-batched-tokens", "4096"),),
            )
        )
        with self.assertRaises(ValueError):
            validate_config(replace(_config(), engine_flags=(("max-num-seqs-é", "8"),)))


if __name__ == "__main__":
    unittest.main()
