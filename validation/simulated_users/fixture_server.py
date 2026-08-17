#!/usr/bin/env python3
"""Deterministic loopback fixture for simulated-user smoke runs.

This server is deliberately *not* an inference engine.  It models prompt-size
dependent latency and synthetic contention, emits strictly shaped OpenAI-style
responses, and retains only aggregate counters.  The ``stress_large`` route
injects exactly one HTTP 503 when a large-prompt wave reaches 12 in-flight
requests so Throttle's fail-closed behavior can be inspected.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


EXPECTED_AUTHORIZATION = "Bearer fixture-secret"
EXPECTED_MODEL = "fixture-model"
PROFILE_NAMES = {
    "short_chat",
    "support_ticket",
    "code_assistant",
    "retrieval_qa",
    "document_summary",
    "mixed_workload",
    "stress_large",
}
STRESS_MIN_PROMPT_CHARACTERS = 4_096
STRESS_TRIGGER_IN_FLIGHT = 12
COHORT_HOLD_SECONDS = 0.040


def _profile_from_path(path: str) -> str | None:
    suffix = "/v1/chat/completions"
    if not path.startswith("/") or not path.endswith(suffix):
        return None
    profile = path[1 : -len(suffix)]
    return profile if profile in PROFILE_NAMES else None


def _prompt_characters(messages: list[dict[str, Any]]) -> int:
    return sum(
        len(message.get("content", ""))
        for message in messages
        if isinstance(message, dict) and isinstance(message.get("content"), str)
    )


def _synthetic_delays(prompt_characters: int, active: int, max_tokens: int) -> tuple[float, float]:
    """Return deterministic prefill/decode delays for a synthetic workload.

    The quadratic contention terms intentionally create different knees by
    prompt size: short prompts tolerate larger batches; long prompts saturate
    earlier.  These numbers model fixture behavior only.
    """

    if prompt_characters < 400:
        knee, coefficient = 8, 0.020
    elif prompt_characters < 1_800:
        knee, coefficient = 6, 0.030
    elif prompt_characters < 5_000:
        knee, coefficient = 4, 0.050
    else:
        knee, coefficient = 2, 0.080
    contention = 1.0 + coefficient * max(0, active - knee) ** 2
    prefill = (0.004 + min(prompt_characters, 20_000) * 0.000004) * contention
    decode = (0.004 + max_tokens * 0.00005) * math.sqrt(contention)
    return prefill, decode


class FixtureState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active: Counter[str] = Counter()
        self.peak_active: Counter[str] = Counter()
        self.started: Counter[str] = Counter()
        self.outcomes: dict[str, Counter[str]] = defaultdict(Counter)
        self.prompt_character_min: dict[str, int] = {}
        self.prompt_character_max: Counter[str] = Counter()
        self.max_tokens: dict[str, set[int]] = defaultdict(set)
        self.stress_triggered = False

    def begin(self, profile: str, prompt_characters: int, max_tokens: int) -> tuple[int, bool]:
        with self.lock:
            self.active[profile] += 1
            active = self.active[profile]
            self.peak_active[profile] = max(self.peak_active[profile], active)
            self.started[profile] += 1
            self.prompt_character_min[profile] = min(
                self.prompt_character_min.get(profile, prompt_characters),
                prompt_characters,
            )
            self.prompt_character_max[profile] = max(
                self.prompt_character_max[profile], prompt_characters
            )
            self.max_tokens[profile].add(max_tokens)
            inject_failure = (
                profile == "stress_large"
                and prompt_characters >= STRESS_MIN_PROMPT_CHARACTERS
                and active >= STRESS_TRIGGER_IN_FLIGHT
                and not self.stress_triggered
            )
            if inject_failure:
                self.stress_triggered = True
            return active, inject_failure

    def finish(self, profile: str, outcome: str) -> None:
        with self.lock:
            self.active[profile] = max(0, self.active[profile] - 1)
            self.outcomes[profile][outcome] += 1

    def summary(self) -> dict[str, Any]:
        with self.lock:
            return {
                "fixture_kind": "synthetic_loopback_not_inference",
                "stress_policy": {
                    "profile": "stress_large",
                    "minimum_prompt_characters": STRESS_MIN_PROMPT_CHARACTERS,
                    "trigger_in_flight": STRESS_TRIGGER_IN_FLIGHT,
                    "triggered": self.stress_triggered,
                },
                "profiles": {
                    profile: {
                        "requests_started": self.started[profile],
                        "peak_server_in_flight": self.peak_active[profile],
                        "active_now": self.active[profile],
                        "outcomes": dict(sorted(self.outcomes[profile].items())),
                        "prompt_characters": {
                            "minimum": self.prompt_character_min.get(profile),
                            "maximum": self.prompt_character_max.get(profile),
                        },
                        "max_tokens": sorted(self.max_tokens[profile]),
                    }
                    for profile in sorted(PROFILE_NAMES)
                    if self.started[profile]
                },
            }


STATE = FixtureState()


class FixtureServer(ThreadingHTTPServer):
    daemon_threads = True


class FixtureHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "ThrottleSimulatedUserFixture/1.0"

    def _json_response(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(encoded)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        self.close_connection = True

    def _write_sse(self, payload: dict[str, Any] | str) -> None:
        data = payload if isinstance(payload, str) else json.dumps(payload, separators=(",", ":"))
        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _stream_response(
        self,
        *,
        prefill_delay: float,
        decode_delay: float,
        usage: dict[str, int],
    ) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self._write_sse(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None,
                    }
                ]
            }
        )
        time.sleep(prefill_delay)
        self._write_sse(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "synthetic "},
                        "finish_reason": None,
                    }
                ]
            }
        )
        time.sleep(decode_delay / 2)
        self._write_sse(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "fixture "},
                        "finish_reason": None,
                    }
                ]
            }
        )
        time.sleep(decode_delay / 2)
        self._write_sse(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "output"},
                        "finish_reason": None,
                    }
                ]
            }
        )
        self._write_sse(
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        )
        self._write_sse({"choices": [], "usage": usage})
        self._write_sse("[DONE]")
        self.close_connection = True

    def do_GET(self) -> None:
        if self.path == "/__stats":
            self._json_response(200, STATE.summary())
        else:
            self._json_response(404, {"error": "not_found"})

    def do_POST(self) -> None:
        profile = _profile_from_path(self.path)
        if profile is None:
            self._json_response(404, {"error": "not_found"})
            return
        if self.headers.get("Authorization") != EXPECTED_AUTHORIZATION:
            self._json_response(401, {"error": "unauthorized"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
        except (ValueError, json.JSONDecodeError):
            self._json_response(400, {"error": "invalid_json"})
            return
        messages = body.get("messages") if isinstance(body, dict) else None
        if (
            not isinstance(body, dict)
            or body.get("model") != EXPECTED_MODEL
            or body.get("temperature") != 0
            or not isinstance(body.get("max_tokens"), int)
            or isinstance(body.get("max_tokens"), bool)
            or body["max_tokens"] <= 0
            or not isinstance(messages, list)
            or not messages
            or any(
                not isinstance(message, dict)
                or not isinstance(message.get("role"), str)
                or not isinstance(message.get("content"), str)
                for message in messages
            )
            or "stop" in body
        ):
            self._json_response(422, {"error": "invalid_request_shape"})
            return

        prompt_characters = _prompt_characters(messages)
        active, inject_failure = STATE.begin(
            profile, prompt_characters, body["max_tokens"]
        )
        outcome = "server_error"
        try:
            if inject_failure:
                outcome = "simulated_memory_pressure_503"
                self._json_response(
                    503, {"error": "simulated_memory_pressure", "retryable": False}
                )
                return
            # Keep the first wave resident long enough for the loopback server
            # to observe the requested 1/2/4/8/16 cohort deterministically.
            # This is fixture scheduling, not simulated model execution time.
            time.sleep(COHORT_HOLD_SECONDS)
            prefill_delay, decode_delay = _synthetic_delays(
                prompt_characters, active, body["max_tokens"]
            )
            completion_tokens = body["max_tokens"]
            prompt_tokens = max(1, math.ceil(prompt_characters / 4))
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
            if body.get("stream") is True:
                self._stream_response(
                    prefill_delay=prefill_delay,
                    decode_delay=decode_delay,
                    usage=usage,
                )
            else:
                time.sleep(prefill_delay + decode_delay)
                self._json_response(
                    200,
                    {
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": "synthetic fixture output",
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": usage,
                    },
                )
            outcome = "http_200_valid_shape"
        except (BrokenPipeError, ConnectionResetError):
            outcome = "client_cancelled_connection"
        finally:
            STATE.finish(profile, outcome)

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    server = FixtureServer(("127.0.0.1", args.port), FixtureHandler)
    print(server.server_address[1], flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
