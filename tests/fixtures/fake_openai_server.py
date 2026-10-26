"""A tiny OpenAI-compatible server for the autotune launcher test.

Standard library only. Answers /v1/models and /v1/chat/completions on
127.0.0.1:PORT; each completion sleeps BASE / SPEED seconds (BASE is
--base-seconds), so a higher --speed is a cheaper config. Exits on SIGTERM
like a real server.
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--speed", type=float, default=1.0)
parser.add_argument("--startup-delay", type=float, default=0.0)
parser.add_argument("--crash", action="store_true")
parser.add_argument("--base-seconds", type=float, default=0.04)
args = parser.parse_args()
if args.crash:
    raise SystemExit("simulated crash: CUDA out of memory")
time.sleep(args.startup_delay)
BASE = args.base_seconds


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _json(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            return self._json(200, {"object": "list", "data": [{"id": "fake", "object": "model", "owned_by": "vllm"}]})
        self._json(404, {})

    def do_POST(self):
        length = int(self.headers.get("content-length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        if not self.path.startswith("/v1/chat/completions"):
            return self._json(404, {})
        words = sum(len(str(m.get("content", "")).split()) for m in body.get("messages", []))
        max_tokens = int(body.get("max_tokens") or 16)
        completion = 1 if max_tokens == 1 else max(1, max_tokens // 2)
        if max_tokens > 1:
            time.sleep(BASE / args.speed)
        self._json(200, {
            "id": "x", "object": "chat.completion", "model": "fake",
            "choices": [{"index": 0, "finish_reason": "stop" if max_tokens > 1 else "length",
                         "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": words + 5, "completion_tokens": completion,
                      "total_tokens": words + 5 + completion},
        })


ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
