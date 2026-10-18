"""An in-memory Throttle Cloud v1 API (~/throttle-hq/cloud/API_CONTRACT.md)
for tests, served through httpx.MockTransport: no sockets, no network.

It keeps a credit balance in micro-dollars, charges per the contract
(check $0.25, autotune trial $1.00, CI gate $0.10), honours Idempotency-Key
(the same key never charges twice and returns the first response), and can be
told to answer any path with a fixed status.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx

KEY = "thr_test_" + "a" * 32
PRICES = {"/api/v1/checks": 250_000, "/api/v1/autotune/trials": 1_000_000, "/api/v1/ci/gate": 100_000}


@dataclass
class CloudStub:
    key: str = KEY
    credits_micro: int = 10_000_000
    plan: str = "free"
    forced: dict[str, int] = field(default_factory=dict)  # path prefix -> status
    requests: list[httpx.Request] = field(default_factory=list)
    bodies: list[Any] = field(default_factory=list)
    seen_keys: dict[str, httpx.Response] = field(default_factory=dict)
    charges: list[str] = field(default_factory=list)
    fail_first: int = 0  # answer this many requests with a connection error

    def credits(self) -> str:
        return f"{self.credits_micro / 1e6:.2f}"

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = json.loads(request.content) if request.content else None
        self.bodies.append(body)
        if self.fail_first > 0:
            self.fail_first -= 1
            raise httpx.ConnectError("connection refused", request=request)
        path = request.url.path
        for prefix, status in self.forced.items():
            if path.startswith(prefix):
                payload: dict[str, Any] = {"error": f"forced {status}"}
                if status == 402:
                    payload = {"error": "out_of_credits", "credits_usd": self.credits(),
                               "upgrade_url": "https://www.throttle-pro.com/pricing"}
                return httpx.Response(status, json=payload)
        if request.headers.get("authorization") != f"Bearer {self.key}":
            return httpx.Response(401, json={"error": "invalid_api_key"})
        if path == "/api/v1/me":
            return httpx.Response(200, json={
                "email": "dev@example.com", "plan": self.plan, "credits_usd": self.credits(),
                "org": {"id": "org_1", "name": "Example"},
            })
        idem = request.headers.get("idempotency-key")
        if idem and idem in self.seen_keys:
            return self.seen_keys[idem]
        price = PRICES.get(path, 0)
        if path.startswith("/api/v1/autotune/runs/"):
            response = httpx.Response(201, json={"url": "https://cloud.throttle-pro.com/autotune/run1"})
        elif price and self.credits_micro < price:
            return httpx.Response(402, json={
                "error": "out_of_credits", "credits_usd": self.credits(),
                "upgrade_url": "https://www.throttle-pro.com/pricing",
            })
        else:
            self.credits_micro -= price
            if price:
                self.charges.append(path)
            charged = f"{price / 1e6:.2f}"
            if path == "/api/v1/checks":
                response = httpx.Response(201, json={
                    "id": "chk_1", "url": "https://cloud.throttle-pro.com/c/chk_1",
                    "charged_usd": charged, "credits_usd": self.credits(),
                })
            elif path == "/api/v1/autotune/trials":
                response = httpx.Response(200, json={
                    "allowed": True, "charged_usd": charged, "credits_usd": self.credits(),
                })
            elif path == "/api/v1/ci/gate":
                response = httpx.Response(200, json={
                    "pass": True, "verdict": "CHEAPER", "charged_usd": charged,
                    "credits_usd": self.credits(),
                })
            else:
                return httpx.Response(404, json={"error": "not_found"})
        if idem:
            self.seen_keys[idem] = response
        return response

    def paths(self) -> list[str]:
        return [r.url.path for r in self.requests]

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)
