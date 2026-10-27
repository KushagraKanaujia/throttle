"""Shared fakes for the shadow / judge / eval tests. All in-process."""

from __future__ import annotations

import json
from typing import Any, Callable

import httpx

from throttle.own import records as r


def exchange(
    text: str,
    output: str = "ok",
    *,
    kind: str | None = None,
    response_format: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
    status: int = 200,
    images: bool = False,
    ts: str | None = None,
    max_tokens: int = 64,
) -> r.Exchange:
    content: Any = text
    if images:
        content = [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": "data:x"}}]
    request: dict[str, Any] = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "system", "content": "Be exact."}, {"role": "user", "content": content}],
        "max_tokens": max_tokens,
    }
    if response_format:
        request["response_format"] = response_format
    if tools:
        request["tools"] = tools
    message: dict[str, Any] = {"role": "assistant", "content": output}
    if tool_calls:
        message["tool_calls"] = tool_calls
    response = {
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3},
    }
    data: dict[str, Any] = dict(
        provider=r.PROVIDER_OPENAI, endpoint="/v1/chat/completions", model="gpt-4o-mini",
        request=request, response=response, status=status, latency_ms=50.0,
        usage=r.usage_from_openai(response), workload_type=kind,
    )
    if ts:
        data["ts"] = ts
    return r.Exchange(**data)


def by_workload_field(ex: r.Exchange, norm: r.Normalized) -> str:
    return ex.workload_type or "chat"


def chat_response(text: str | None, *, finish: str = "stop", tool_calls=None, usage=True) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = tool_calls
    body: dict[str, Any] = {"choices": [{"message": message, "finish_reason": finish}]}
    if usage:
        body["usage"] = {"prompt_tokens": 11, "completion_tokens": 4}
    return body


def candidate_transport(answer: Callable[[dict[str, Any]], Any], calls: list | None = None) -> httpx.MockTransport:
    """A fake OpenAI-compatible server. ``answer(body)`` returns a dict body,
    or an ``httpx.Response`` to send as is."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if calls is not None:
            calls.append((str(request.url), dict(request.headers), body))
        result = answer(body)
        if isinstance(result, httpx.Response):
            return result
        return httpx.Response(200, json=result)

    return httpx.MockTransport(handler)


def last_user(body: dict[str, Any]) -> str:
    return [m for m in body["messages"] if m["role"] == "user"][-1]["content"]
