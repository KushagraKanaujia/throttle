"""Turn a failed load-test request into a clear error, and fail without leftovers.

Two problems reported against a real vLLM server (issue #22):

* A request whose prompt plus ``max_tokens`` exceeds the model's context window
  gets HTTP 400 from vLLM, and Throttle stopped with only "failed with status
  400". :func:`describe_http_failure` names the limit and the fix instead.
* When one concurrent request failed, ``asyncio.gather`` raised at once and left
  the other requests running. ``asyncio.run`` then cancelled them during
  shutdown, mid-connect, and printed their errors as a traceback.
  :func:`gather_or_drain` stops sending and waits for them instead.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Awaitable, Iterable, TypeVar

T = TypeVar("T")

# Messages servers send when prompt + max_tokens does not fit the context window.
# Each pattern captures the limit in tokens.
_LIMIT_PATTERNS = (
    # vLLM and OpenAI: "This model's maximum context length is 2048 tokens."
    re.compile(r"maximum context length is (\d+)", re.IGNORECASE),
    # SGLang: "... is longer than the model's context length (2048 tokens)."
    re.compile(r"context length \((\d+) tokens?\)", re.IGNORECASE),
    # TGI: "`inputs` tokens + `max_new_tokens` must be <= 2048."
    re.compile(r"max_new_tokens`? must be <= (\d+)", re.IGNORECASE),
    # llama.cpp: "... exceeds the available context size (2048 tokens)"
    re.compile(r"context size \((\d+) tokens?\)", re.IGNORECASE),
)
_OVERFLOW_MARKERS = (
    "context_length_exceeded",
    "maximum context length",
    "context length",
    "context window",
    "context size",
    "max_model_len",
)
SNIPPET_CHARS = 500


class RequestFailed(Exception):
    """A load-test request failed; the message is ready to show the user."""


def server_error_text(body: str) -> str:
    """The server's own error message from a JSON error body, else the raw text."""

    try:
        data: Any = json.loads(body)
    except ValueError:
        data = None
    if isinstance(data, dict):
        error = data.get("error", data)
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"]
        if isinstance(error, str):
            return error
        if isinstance(data.get("message"), str):
            return data["message"]
    return body


def context_overflow_limit(status: int, body: str) -> tuple[bool, int | None]:
    """Whether ``body`` reports a context-window overflow, and the limit if stated."""

    if status not in (400, 413, 422):
        return False, None
    text = server_error_text(body)
    for pattern in _LIMIT_PATTERNS:
        match = pattern.search(text)
        if match:
            return True, int(match.group(1))
    # The raw body too: OpenAI puts "context_length_exceeded" in error.code.
    lowered = f"{text} {body}".lower()
    return any(marker in lowered for marker in _OVERFLOW_MARKERS), None


def describe_http_failure(
    *,
    what: str,
    url: str,
    status: int,
    body: str,
    max_tokens: int | None = None,
    prompt_words: int | None = None,
    remedy: str,
) -> str:
    """One user-facing message for a non-200 load-test response.

    ``what`` names the request ("Heavy load request 86"), ``remedy`` says how to
    make the workload fit; it is used only when the failure is a context
    overflow.
    """

    snippet = " ".join(server_error_text(body).split())[:SNIPPET_CHARS] or "(empty body)"
    overflow, limit = context_overflow_limit(status, body)
    if not overflow:
        return f"{what} failed: POST {url} returned HTTP {status}: {snippet}"
    window = f"{limit}-token context window (max_model_len)" if limit else "context window"
    sent = []
    if prompt_words is not None:
        sent.append(f"a {prompt_words}-word prompt")
    if max_tokens is not None:
        sent.append(f"max_tokens {max_tokens}")
    sent_text = f" It sent {' and '.join(sent)}." if sent else ""
    return (
        f"{what} was rejected with HTTP {status}: its prompt plus max_tokens does not "
        f"fit the model's {window}.{sent_text}\n"
        f"  Server said: {snippet}\n"
        f"  Fix: {remedy}"
    )


async def wait_or_stop(stop: asyncio.Event, delay: float) -> bool:
    """Sleep ``delay`` seconds; return True early if ``stop`` is set."""

    if stop.is_set():
        return True
    if delay <= 0:
        return False
    try:
        await asyncio.wait_for(stop.wait(), delay)
    except TimeoutError:
        return False
    return True


async def gather_or_drain(aws: Iterable[Awaitable[T]], stop: asyncio.Event) -> list[T]:
    """``asyncio.gather`` that leaves nothing running when one awaitable fails.

    On the first failure it sets ``stop`` and waits for the others before
    re-raising, retrieving whatever they raise. Awaitables must check ``stop``
    before sending (see :func:`wait_or_stop`), so requests not yet sent are
    skipped and requests already on the wire finish normally.

    Nothing is cancelled mid-request: cancelling a request while anyio's
    connect_tcp is connecting can raise ValueError or leak the socket, which is
    what ``asyncio.run`` reported at shutdown in #22 when plain gather left the
    other requests running.
    """

    tasks = [asyncio.ensure_future(aw) for aw in aws]
    try:
        return list(await asyncio.gather(*tasks))
    except Exception:
        stop.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
