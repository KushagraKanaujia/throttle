#!/usr/bin/env python3
"""Loopback-only OpenAI-compatible fixture used for CLI validation.

This is a deterministic software fixture, not an inference engine. It validates
the safety-critical request shape without retaining prompt or response text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


EXPECTED_AUTHORIZATION = "Bearer fixture-secret"
EXPECTED_MODEL = "fixture-model"
VALID_ROUTES = {
    "/valid/v1/chat/completions": (0.012, 1.0),
    "/candidate/v1/chat/completions": (0.006, 1.0),
    "/mismatch/v1/chat/completions": (0.006, 0.5),
    "/slow/v1/chat/completions": (0.060, 1.0),
}


class FixtureState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.prompt_hashes: dict[str, list[str]] = defaultdict(list)
        self.max_tokens: dict[str, set[int]] = defaultdict(set)
        self.temperatures: dict[str, set[float]] = defaultdict(set)
        self.stop_seen: dict[str, bool] = defaultdict(bool)

    def record(self, route: str, body: dict[str, Any]) -> None:
        messages = json.dumps(
            body["messages"], sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest = hashlib.sha256(messages).hexdigest()
        with self.lock:
            self.prompt_hashes[route].append(digest)
            self.max_tokens[route].add(body["max_tokens"])
            self.temperatures[route].add(body["temperature"])
            self.stop_seen[route] = self.stop_seen[route] or "stop" in body

    def sanitized_summary(self) -> dict[str, Any]:
        with self.lock:
            routes = sorted(
                set(self.prompt_hashes)
                | set(self.max_tokens)
                | set(self.temperatures)
                | set(self.stop_seen)
            )
            return {
                route: {
                    "request_count": len(self.prompt_hashes[route]),
                    "prompt_hashes": list(self.prompt_hashes[route]),
                    "max_tokens": sorted(self.max_tokens[route]),
                    "temperatures": sorted(self.temperatures[route]),
                    "stop_seen": self.stop_seen[route],
                }
                for route in routes
            }


STATE = FixtureState()


class FixtureHandler(BaseHTTPRequestHandler):
    server_version = "ThrottleFixture/1.0"

    def _json_response(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _stream_response(self, events: list[dict[str, Any]]) -> None:
        encoded = (
            "".join(
                f"data: {json.dumps(event, separators=(',', ':'))}\n\n"
                for event in events
            ).encode("utf-8")
            + b"data: [DONE]\n\n"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if self.path != "/__stats":
            self._json_response(404, {"error": "not_found"})
            return
        self._json_response(200, STATE.sanitized_summary())

    def do_POST(self) -> None:
        if self.path not in VALID_ROUTES:
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
        if (
            not isinstance(body, dict)
            or body.get("model") != EXPECTED_MODEL
            or body.get("temperature") != 0
            or not isinstance(body.get("max_tokens"), int)
            or isinstance(body.get("max_tokens"), bool)
            or body["max_tokens"] <= 0
            or not isinstance(body.get("messages"), list)
            or not body["messages"]
            or "stop" in body
        ):
            self._json_response(422, {"error": "invalid_request_shape"})
            return

        STATE.record(self.path, body)
        delay_seconds, token_ratio = VALID_ROUTES[self.path]
        time.sleep(delay_seconds)
        completion_tokens = max(1, round(body["max_tokens"] * token_ratio))
        usage = {
            "prompt_tokens": 8,
            "completion_tokens": completion_tokens,
            "total_tokens": completion_tokens + 8,
        }
        if body.get("stream") is True:
            self._stream_response(
                [
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
                                "delta": {"content": "fixture"},
                                "finish_reason": None,
                            }
                        ]
                    },
                    {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                    {"choices": [], "usage": usage},
                ]
            )
        else:
            self._json_response(
                200,
                {
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "fixture"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": usage,
                },
            )

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), FixtureHandler)
    print(server.server_address[1], flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
