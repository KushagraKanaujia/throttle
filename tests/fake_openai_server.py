"""A tiny OpenAI-compatible server for `throttle coldstart` tests (stdlib only).

    python fake_openai_server.py PORT [--delay S] [--pidfile PATH] [--ignore-term]

Sleeps --delay seconds (the "model load"), then serves /health, /v1/models,
/version and a streaming /v1/chat/completions on 127.0.0.1:PORT.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def _json(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif self.path == "/v1/models":
            self._json({"data": [{"id": "m", "owned_by": "fake"}]})
        elif self.path == "/version":
            self._json({"version": "0.0.fake"})
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for piece in ("Hi", "!"):
            event = {"choices": [{"index": 0, "delta": {"content": piece}}]}
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("port", type=int)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--pidfile")
    parser.add_argument("--ignore-term", action="store_true")
    args = parser.parse_args()
    if args.ignore_term:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if args.pidfile:
        with open(args.pidfile, "a") as f:
            f.write(f"{os.getpid()}\n")
    time.sleep(args.delay)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    server.serve_forever()


if __name__ == "__main__":
    main()
