"""Native smoke and sustained benchmark engine for Throttle v2.

No prompt, response, credential, endpoint URL, or exception text is included in
the returned report. Reports contain numeric timing/token evidence only.
"""

from __future__ import annotations

import asyncio
import codecs
import copy
import hashlib
import ipaddress
import json
import math
import os
import random
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any, AsyncIterator, Mapping, Sequence
from urllib.parse import SplitResult, urlsplit, urlunsplit

import httpx

from . import __version__
from .models import (
    OPEN_LOOP_RATE_RELATIVE_TOLERANCE,
    OPEN_LOOP_SCHEDULER_LAG_INTERVAL_TOLERANCE,
    LoadCondition,
    RunConfig,
)
from .statistics import (
    intervals_overlap,
    summarize_distribution_ms,
    t_interval_95,
)

SCHEMA_VERSION = "2.0"
ARTIFACT_TYPE = "throttle_run"
MIN_DECISION_BLOCKS = 3
MIN_DECISION_REQUESTS = 200
MIN_DECISION_SECONDS = 60.0
BEST_TESTED_OUTPUT_TOKEN_TOLERANCE = 0.05
# Throttle's fixed workload does not send tools or functions. Accepting their
# finish reasons without validating matching call payloads would turn a
# structurally inconsistent response into a valid text completion.
ALLOWED_FINISH_REASONS = frozenset({"stop", "length", "content_filter"})

Prompt = tuple[dict[str, str], ...]
Prompts = tuple[Prompt, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _reject_json_constant(_: str) -> None:
    raise ValueError("non-finite JSON number")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate JSON object key")
        output[key] = value
    return output


def _strict_json_loads(value: str | bytes) -> Any:
    """Parse standards-compliant, unambiguous JSON from an untrusted boundary."""

    return json.loads(
        value,
        parse_constant=_reject_json_constant,
        object_pairs_hook=_unique_json_object,
    )


def _is_loopback_host(hostname: str) -> bool:
    lowered = hostname.rstrip(".").lower()
    if lowered == "localhost":
        return True
    try:
        return ipaddress.ip_address(lowered).is_loopback
    except ValueError:
        return False


def _parse_endpoint(value: str) -> SplitResult:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("endpoint URL must not be empty")
    if value != value.strip() or any(
        ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise ValueError("endpoint URL contains whitespace or control characters")
    parsed = urlsplit(value)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("endpoint URL must be an absolute http(s) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("endpoint URL must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("endpoint URL must not contain a query or fragment")
    if parsed.hostname.rstrip(".").lower() in {"0.0.0.0", "::", "*"}:
        raise ValueError("endpoint URL must identify one destination, not a wildcard")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("endpoint URL has an invalid port") from exc
    return parsed


def normalize_chat_completions_url(
    value: str, *, allow_insecure_http: bool = False
) -> str:
    """Normalize a base URL and enforce HTTPS away from exact loopback hosts."""

    parsed = _parse_endpoint(value)
    scheme = parsed.scheme.lower()
    if (
        scheme == "http"
        and not _is_loopback_host(str(parsed.hostname))
        and not allow_insecure_http
    ):
        raise ValueError(
            "non-loopback endpoints require HTTPS; use --allow-insecure-http only "
            "after reviewing the bearer-token risk"
        )
    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions"):
        normalized_path = path
    elif not path:
        normalized_path = "/v1/chat/completions"
    else:
        normalized_path = f"{path}/chat/completions"
    return urlunsplit((scheme, parsed.netloc, normalized_path, "", ""))


def destination_summary(
    value: str, *, allow_insecure_http: bool = False
) -> dict[str, Any]:
    """Terminal-only destination details for ``throttle plan``."""

    normalized = normalize_chat_completions_url(
        value, allow_insecure_http=allow_insecure_http
    )
    parsed = urlsplit(normalized)
    return {
        "normalized_url": normalized,
        "scheme": parsed.scheme,
        "host": parsed.hostname,
        "port": parsed.port,
        "path": parsed.path,
        "loopback": _is_loopback_host(str(parsed.hostname)),
    }


def _normalize_messages(value: object, line_number: int) -> Prompt:
    if not isinstance(value, list) or not value:
        raise ValueError(
            f"prompt line {line_number}: messages must be a non-empty list"
        )
    messages: list[dict[str, str]] = []
    for index, message in enumerate(value, start=1):
        if not isinstance(message, dict):
            raise ValueError(
                f"prompt line {line_number}: message {index} must be an object"
            )
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role.strip():
            raise ValueError(
                f"prompt line {line_number}: message {index} needs a non-empty role"
            )
        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                f"prompt line {line_number}: message {index} needs non-empty text content"
            )
        messages.append({"role": role, "content": content})
    return tuple(messages)


def _parse_prompt_lines(lines: Sequence[str]) -> Prompts:
    prompts: list[Prompt] = []
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        try:
            item = _strict_json_loads(raw_line)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"prompt line {line_number}: invalid JSON") from exc
        if not isinstance(item, dict):
            raise ValueError(f"prompt line {line_number}: value must be an object")
        has_messages = "messages" in item
        has_prompt = "prompt" in item
        if has_messages == has_prompt:
            raise ValueError(
                f"prompt line {line_number}: provide exactly one of messages or prompt"
            )
        if has_prompt:
            prompt = item["prompt"]
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    f"prompt line {line_number}: prompt must be non-empty text"
                )
            prompts.append(({"role": "user", "content": prompt},))
        else:
            prompts.append(_normalize_messages(item["messages"], line_number))
    if not prompts:
        raise ValueError("prompt file contains no prompts")
    return tuple(prompts)


def load_prompts(path: str | Path | None = None, *, warmup: bool = False) -> Prompts:
    """Load measured or separate warm-up JSONL prompts."""

    if path is None:
        filename = "warmup_prompts.jsonl" if warmup else "prompts.jsonl"
        prompt_file = resources.files("throttle").joinpath(filename)
        with prompt_file.open("r", encoding="utf-8") as handle:
            return _parse_prompt_lines(handle.readlines())
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        return _parse_prompt_lines(handle.readlines())


def canonical_workload_hash(prompts: Prompts) -> str:
    canonical = json.dumps(prompts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _prompt_identity_hashes(prompts: Prompts) -> set[str]:
    return {
        hashlib.sha256(
            json.dumps(prompt, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        for prompt in prompts
    }


def _validate_engine_flags(flags: tuple[tuple[str, str], ...]) -> None:
    forbidden_name_markers = (
        "api-key",
        "apikey",
        "auth-token",
        "access-token",
        "refresh-token",
        "bearer-token",
        "id-token",
        "private-key",
        "key-file",
        "keyfile",
        "ssl-key",
        "tls-key",
        "secret",
        "password",
        "credential",
        "authorization",
        "bearer",
        "header",
        "url",
        "endpoint",
    )
    seen: set[str] = set()
    for name, value in flags:
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", name):
            raise ValueError(
                "engine flag names may contain only letters, numbers, '_' and '-'"
            )
        normalized_name = name.replace("_", "-").lower()
        if any(marker in normalized_name for marker in forbidden_name_markers):
            raise ValueError(f"engine flag {name!r} is unsafe to persist")
        if normalized_name in seen:
            raise ValueError(f"duplicate engine flag: {name}")
        seen.add(normalized_name)
        if (
            not isinstance(value, str)
            or not value
            or value != value.strip()
            or len(value) > 256
        ):
            raise ValueError(f"engine flag {name!r} has an invalid value")
        value_lower = value.lower()
        if any(
            marker in value_lower
            for marker in (
                "://",
                "bearer ",
                "authorization:",
                "api_key",
                "api-key",
            )
        ):
            raise ValueError(f"engine flag {name!r} may contain a secret or URL")


def _validate_public_metadata(name: str, value: str, *, max_length: int = 256) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > max_length
    ):
        raise ValueError(
            f"{name} must be non-empty text no longer than {max_length} characters"
        )
    if value.lower() == "unknown" and value != "unknown":
        raise ValueError(f"{name} must use the canonical 'unknown' sentinel")
    lowered = value.lower()
    if any(
        marker in lowered
        for marker in ("://", "bearer ", "authorization:", "api_key=", "api-key=")
    ):
        raise ValueError(f"{name} may contain a URL or credential")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} must not contain control characters")


def validate_config(config: RunConfig, *, for_traffic: bool = True) -> None:
    if config.mode not in {"smoke", "benchmark"}:
        raise ValueError("mode must be smoke or benchmark")
    if config.backend not in {"native", "guidellm"}:
        raise ValueError("backend must be native or guidellm")
    for name, value in (
        ("stream", config.stream),
        ("allow_unknown_cost", config.allow_unknown_cost),
        ("allow_insecure_http", config.allow_insecure_http),
        ("guidellm_gaps_acknowledged", config.guidellm_gaps_acknowledged),
    ):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean")
    if (
        for_traffic
        and config.backend == "guidellm"
        and not config.guidellm_gaps_acknowledged
    ):
        raise ValueError(
            "GuideLLM cannot enforce strict completion shape or max response bytes; "
            "pass --allow-guidellm-validation-gaps for cross-check-only traffic"
        )
    if not isinstance(config.model, str) or not config.model.strip():
        raise ValueError("model must not be empty")
    normalize_chat_completions_url(
        config.endpoint.url, allow_insecure_http=config.allow_insecure_http
    )
    if for_traffic and (
        not isinstance(config.endpoint.api_key, str) or not config.endpoint.api_key
    ):
        raise ValueError("API key must not be empty")
    config.cost.validate()
    config.limits.validate()
    if not _is_int(config.max_tokens) or config.max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    if config.max_tokens > config.limits.max_tokens_per_request:
        raise ValueError("max_tokens exceeds max_tokens_per_request safety limit")
    if not config.conditions:
        raise ValueError("at least one load condition is required")
    condition_ids: set[str] = set()
    kinds: set[str] = set()
    for condition in config.conditions:
        if condition.kind not in {"closed_loop", "open_loop"}:
            raise ValueError("unsupported traffic kind")
        if not math.isfinite(condition.value) or condition.value <= 0:
            raise ValueError("load condition values must be positive and finite")
        if not _is_int(condition.max_in_flight) or condition.max_in_flight <= 0:
            raise ValueError("condition max_in_flight must be a positive integer")
        if condition.max_in_flight > config.limits.max_concurrency:
            raise ValueError("load condition exceeds max_concurrency safety limit")
        if condition.kind == "closed_loop" and int(condition.value) != condition.value:
            raise ValueError("closed-loop concurrency must be an integer")
        if (
            condition.kind == "closed_loop"
            and int(condition.value) > config.limits.max_concurrency
        ):
            raise ValueError("concurrency exceeds max_concurrency safety limit")
        if condition.kind == "closed_loop" and condition.max_in_flight != int(
            condition.value
        ):
            raise ValueError("closed-loop max_in_flight must equal its concurrency")
        if condition.condition_id in condition_ids:
            raise ValueError("load condition values must be unique")
        condition_ids.add(condition.condition_id)
        kinds.add(condition.kind)
    if len(kinds) != 1:
        raise ValueError("one run cannot mix closed-loop and open-loop conditions")
    if not _is_int(config.blocks) or config.blocks <= 0:
        raise ValueError("blocks must be a positive integer")
    if config.mode == "smoke" and config.blocks != 1:
        raise ValueError("smoke mode always uses one measured block")
    if config.mode == "benchmark" and config.blocks < MIN_DECISION_BLOCKS:
        raise ValueError("benchmark mode requires at least three repeated blocks")
    if config.requests_per_block is None and config.block_duration_seconds is None:
        raise ValueError("provide a request or duration bound per block")
    if (
        config.requests_per_block is not None
        and config.block_duration_seconds is not None
    ):
        raise ValueError("choose either a request bound or a duration bound per block")
    if config.requests_per_block is not None and (
        not _is_int(config.requests_per_block) or config.requests_per_block <= 0
    ):
        raise ValueError("requests_per_block must be a positive integer")
    if config.block_duration_seconds is not None and (
        not math.isfinite(config.block_duration_seconds)
        or config.block_duration_seconds <= 0
    ):
        raise ValueError("block_duration_seconds must be positive and finite")
    if (
        not _is_int(config.warmup_requests_per_condition)
        or config.warmup_requests_per_condition < 0
    ):
        raise ValueError("warmup_requests_per_condition must be non-negative")
    if (
        not math.isfinite(config.request_timeout_seconds)
        or config.request_timeout_seconds <= 0
    ):
        raise ValueError("request_timeout_seconds must be positive and finite")
    for name, value in (
        ("p95_slo_ms", config.p95_slo_ms),
        ("ttft_slo_ms", config.ttft_slo_ms),
    ):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be positive and finite")
    if not isinstance(config.seed, int) or isinstance(config.seed, bool):
        raise ValueError("seed must be an integer")
    if config.cache_policy not in {
        "unknown",
        "disabled",
        "cold",
        "warm",
        "representative",
    }:
        raise ValueError("unsupported cache policy")
    if config.evidence_source not in {
        "unverified_endpoint",
        "live_inference",
        "synthetic_validation",
    }:
        raise ValueError("unsupported evidence source")
    if config.engine_flags_provenance not in {"operator_attested", "runtime_verified"}:
        raise ValueError("unsupported engine flag provenance")
    for name, value in (
        ("model", config.model),
        ("model_revision", config.model_revision),
        ("image_digest", config.image_digest),
        ("gpu", config.gpu),
        ("gpu_fingerprint", config.gpu_fingerprint),
        ("cuda_version", config.cuda_version),
        ("driver_version", config.driver_version),
        ("server_version", config.server_version),
        ("variant", config.variant),
        ("sequence_position", config.sequence_position),
    ):
        _validate_public_metadata(name, value)
    _validate_engine_flags(config.engine_flags)

    planned = config.planned_request_count()
    if for_traffic and planned is not None and planned > config.limits.max_requests:
        raise ValueError("planned inference requests exceed max_requests")
    planned_tokens = config.planned_requested_output_tokens()
    if (
        for_traffic
        and planned_tokens is not None
        and planned_tokens > config.limits.max_total_requested_tokens
    ):
        raise ValueError("planned requested output tokens exceed the total token limit")
    estimate = config.cost.estimated_upper_bound(config.limits.max_elapsed_seconds)
    if estimate is None and for_traffic and not config.allow_unknown_cost:
        raise ValueError(
            "the selected billing model cannot enforce max spend; pass "
            "--allow-unknown-cost to acknowledge this explicitly"
        )
    if (
        for_traffic
        and estimate is not None
        and estimate > config.limits.max_estimated_spend
    ):
        raise ValueError("known pre-run cost ceiling exceeds max_estimated_spend")


def build_plan(
    config: RunConfig, prompts: Prompts, warmup_prompts: Prompts
) -> dict[str, Any]:
    """Build a zero-traffic plan. This function never resolves a key or opens I/O."""

    validate_config(config, for_traffic=False)
    destination = destination_summary(
        config.endpoint.url, allow_insecure_http=config.allow_insecure_http
    )
    planned = config.planned_request_count()
    planned_tokens = config.planned_requested_output_tokens()
    return {
        "mode": config.mode,
        "backend": config.backend,
        "guidellm_gaps_acknowledged": config.guidellm_gaps_acknowledged,
        "traffic_sent": False,
        "request_count": {
            "exact": planned,
            "upper_bound": planned
            if planned is not None
            else config.limits.max_requests,
            "includes_warmups": True,
        },
        "max_tokens_per_request": config.max_tokens,
        "requested_output_token_ceiling": (
            planned_tokens
            if planned_tokens is not None
            else config.limits.max_total_requested_tokens
        ),
        "duration_limit_seconds": config.limits.max_elapsed_seconds,
        "estimated_cost_upper_bound": config.cost.estimated_upper_bound(
            config.limits.max_elapsed_seconds
        ),
        "cost_model": config.cost.public_dict(config.limits.max_elapsed_seconds),
        "destination": destination,
        "workload": {
            "measured_prompt_count": len(prompts),
            "warmup_prompt_count": len(warmup_prompts),
            "separate_warmup_workload": canonical_workload_hash(prompts)
            != canonical_workload_hash(warmup_prompts),
            "warmup_prompts_disjoint": _prompt_identity_hashes(prompts).isdisjoint(
                _prompt_identity_hashes(warmup_prompts)
            ),
        },
        "privacy": {
            "prompts_leave_this_machine": not destination["loopback"],
            "endpoint_or_intermediaries_may_log_content": True,
            "report_omits_urls_credentials_prompts_and_responses": True,
            "report_persists_stable_workload_fingerprints": True,
            "ambient_proxy_environment_used": False,
            "plaintext_non_loopback_transport": (
                destination["scheme"] == "http" and not destination["loopback"]
            ),
        },
        "limits": config.limits.public_dict(),
        "traffic_preflight": {
            "backend_supported_on_this_platform": (
                config.backend != "guidellm" or os.name == "posix"
            ),
            "backend_block_reason": (
                "guidellm_requires_posix_process_groups"
                if config.backend == "guidellm" and os.name != "posix"
                else None
            ),
            "spend_limit_enforceable": config.cost.estimated_upper_bound(
                config.limits.max_elapsed_seconds
            )
            is not None,
            "requires_unknown_cost_acknowledgement": (
                config.cost.estimated_upper_bound(config.limits.max_elapsed_seconds)
                is None
                and not config.allow_unknown_cost
            ),
            "estimated_cost_exceeds_limit": (
                config.cost.estimated_upper_bound(config.limits.max_elapsed_seconds)
                is not None
                and float(
                    config.cost.estimated_upper_bound(config.limits.max_elapsed_seconds)
                )
                > config.limits.max_estimated_spend
            ),
            "planned_requests_exceed_limit": (
                planned is not None and planned > config.limits.max_requests
            ),
            "planned_tokens_exceed_limit": (
                planned_tokens is not None
                and planned_tokens > config.limits.max_total_requested_tokens
            ),
        },
        "warning": (
            "Smoke is a connectivity/load-shape check and is never a production "
            "recommendation."
            if config.mode == "smoke"
            else "Benchmark results remain specific to the declared workload and manifest."
        ),
    }


@dataclass
class RequestResult:
    status_code: int | None
    e2e_seconds: float
    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    finish_reason: str | None = None
    ttft_seconds: float | None = None
    tpot_seconds: float | None = None
    inter_chunk_seconds: tuple[float, ...] = ()
    response_bytes: int = 0
    error_code: str | None = None

    @property
    def valid(self) -> bool:
        return (
            self.status_code == 200
            and self.error_code is None
            and self.completion_tokens is not None
            and self.completion_tokens > 0
            and self.prompt_tokens is not None
            and self.finish_reason is not None
        )


class _ResponseTooLarge(Exception):
    pass


async def _read_limited(response: httpx.Response, limit: int) -> bytes:
    body = bytearray()
    async for chunk in response.aiter_bytes():
        if len(body) + len(chunk) > limit:
            raise _ResponseTooLarge
        body.extend(chunk)
    return bytes(body)


async def _sse_events(
    response: httpx.Response, limit: int
) -> AsyncIterator[tuple[str, int]]:
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    buffer = ""
    data_lines: list[str] = []
    byte_count = 0
    async for chunk in response.aiter_bytes():
        byte_count += len(chunk)
        if byte_count > limit:
            raise _ResponseTooLarge
        buffer += decoder.decode(chunk)
        emitted: list[str] = []
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.rstrip("\r")
            if not line:
                if data_lines:
                    emitted.append("\n".join(data_lines))
                    data_lines.clear()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
            elif line.startswith(":"):
                continue
        for event in emitted:
            yield event, byte_count
    buffer += decoder.decode(b"", final=True)
    if buffer:
        line = buffer.rstrip("\r")
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
        elif line and not line.startswith(":"):
            raise ValueError("malformed_sse")
    if data_lines:
        yield "\n".join(data_lines), byte_count


def _usage(payload: Mapping[str, Any]) -> tuple[int, int] | str:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return "missing_usage"
    completion_tokens = usage.get("completion_tokens")
    if not _is_int(completion_tokens) or completion_tokens <= 0:
        return "invalid_completion_tokens"
    prompt_tokens = usage.get("prompt_tokens")
    if not _is_int(prompt_tokens) or prompt_tokens < 0:
        return "invalid_prompt_tokens"
    return prompt_tokens, completion_tokens


def _validate_nonstream_payload(payload: object) -> tuple[int, int, str] | str:
    if not isinstance(payload, dict):
        return "response_not_object"
    if "error" in payload:
        return "error_response_shape"
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return "missing_or_empty_choices"
    choice = choices[0]
    if not isinstance(choice, dict):
        return "choice_not_object"
    if choice.get("index") != 0:
        return "invalid_choice_index"
    finish_reason = choice.get("finish_reason")
    if not isinstance(finish_reason, str) or not finish_reason.strip():
        return "missing_finish_reason"
    if finish_reason not in ALLOWED_FINISH_REASONS:
        return "unsupported_finish_reason"
    message = choice.get("message")
    if not isinstance(message, dict):
        return "missing_assistant_message"
    if message.get("role") != "assistant":
        return "invalid_assistant_role"
    content = message.get("content")
    if not (isinstance(content, str) and bool(content.strip())):
        return "empty_completion_output"
    usage = _usage(payload)
    if isinstance(usage, str):
        return usage
    return usage[0], usage[1], finish_reason


async def _native_request(
    client: httpx.AsyncClient,
    endpoint_url: str,
    config: RunConfig,
    messages: Prompt,
) -> RequestResult:
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": list(messages),
        "temperature": 0,
        "max_tokens": config.max_tokens,
        "stream": config.stream,
    }
    if config.stream:
        payload["stream_options"] = {"include_usage": True}
    try:
        async with client.stream("POST", endpoint_url, json=payload) as response:
            status = response.status_code
            if status != 200:
                return RequestResult(
                    status, time.perf_counter() - started, error_code="non_200_response"
                )
            content_encoding = response.headers.get("content-encoding")
            if (
                content_encoding is not None
                and content_encoding.strip().lower() != "identity"
            ):
                # httpx exposes decoded bytes from aiter_bytes(). Refuse
                # content coding so a small compressed body cannot inflate in
                # memory before the decoded-size ceiling is checked.
                return RequestResult(
                    200,
                    time.perf_counter() - started,
                    error_code="unsupported_content_encoding",
                )
            if not config.stream:
                try:
                    body = await _read_limited(
                        response, config.limits.max_response_bytes
                    )
                except _ResponseTooLarge:
                    return RequestResult(
                        200,
                        time.perf_counter() - started,
                        error_code="response_too_large",
                    )
                try:
                    parsed = _strict_json_loads(body)
                except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                    return RequestResult(
                        200,
                        time.perf_counter() - started,
                        response_bytes=len(body),
                        error_code="invalid_json",
                    )
                checked = _validate_nonstream_payload(parsed)
                if isinstance(checked, str):
                    return RequestResult(
                        200,
                        time.perf_counter() - started,
                        response_bytes=len(body),
                        error_code=checked,
                    )
                prompt_tokens, completion_tokens, finish_reason = checked
                if completion_tokens > config.max_tokens:
                    return RequestResult(
                        200,
                        time.perf_counter() - started,
                        response_bytes=len(body),
                        error_code="reported_completion_tokens_exceed_request_cap",
                    )
                return RequestResult(
                    200,
                    time.perf_counter() - started,
                    completion_tokens=completion_tokens,
                    prompt_tokens=prompt_tokens,
                    finish_reason=finish_reason,
                    response_bytes=len(body),
                )

            first_output_at: float | None = None
            output_event_times: list[float] = []
            finish_reason: str | None = None
            prompt_tokens: int | None = None
            completion_tokens: int | None = None
            saw_done = False
            saw_assistant_role = False
            saw_finish = False
            saw_usage = False
            response_bytes = 0
            try:
                async for event, response_bytes in _sse_events(
                    response, config.limits.max_response_bytes
                ):
                    now = time.perf_counter()
                    if event == "[DONE]":
                        if saw_done:
                            return RequestResult(
                                200,
                                now - started,
                                response_bytes=response_bytes,
                                error_code="duplicate_stream_done",
                            )
                        saw_done = True
                        continue
                    if saw_done:
                        return RequestResult(
                            200,
                            now - started,
                            response_bytes=response_bytes,
                            error_code="stream_data_after_done",
                        )
                    try:
                        chunk = _strict_json_loads(event)
                    except (json.JSONDecodeError, ValueError):
                        return RequestResult(
                            200,
                            now - started,
                            response_bytes=response_bytes,
                            error_code="malformed_stream_json",
                        )
                    if not isinstance(chunk, dict):
                        return RequestResult(
                            200,
                            now - started,
                            response_bytes=response_bytes,
                            error_code="stream_event_not_object",
                        )
                    if "error" in chunk:
                        return RequestResult(
                            200,
                            now - started,
                            response_bytes=response_bytes,
                            error_code="stream_error_event",
                        )
                    has_usage = "usage" in chunk and chunk["usage"] is not None
                    if has_usage:
                        if saw_usage:
                            return RequestResult(
                                200,
                                now - started,
                                response_bytes=response_bytes,
                                error_code="duplicate_stream_usage",
                            )
                        if not saw_finish:
                            return RequestResult(
                                200,
                                now - started,
                                response_bytes=response_bytes,
                                error_code="stream_usage_before_finish",
                            )
                        checked_usage = _usage(chunk)
                        if isinstance(checked_usage, str):
                            return RequestResult(
                                200,
                                now - started,
                                response_bytes=response_bytes,
                                error_code=checked_usage,
                            )
                        prompt_tokens, completion_tokens = checked_usage
                        saw_usage = True
                    choices = chunk.get("choices")
                    if choices == []:
                        if not has_usage:
                            return RequestResult(
                                200,
                                now - started,
                                response_bytes=response_bytes,
                                error_code="empty_stream_event_without_usage",
                            )
                        continue
                    if has_usage:
                        return RequestResult(
                            200,
                            now - started,
                            response_bytes=response_bytes,
                            error_code="stream_usage_with_choice_data",
                        )
                    if saw_usage:
                        return RequestResult(
                            200,
                            now - started,
                            response_bytes=response_bytes,
                            error_code="stream_choice_after_usage",
                        )
                    if saw_finish:
                        return RequestResult(
                            200,
                            now - started,
                            response_bytes=response_bytes,
                            error_code="stream_choice_after_finish",
                        )
                    if (
                        not isinstance(choices, list)
                        or len(choices) != 1
                        or not isinstance(choices[0], dict)
                    ):
                        return RequestResult(
                            200,
                            now - started,
                            response_bytes=response_bytes,
                            error_code="invalid_stream_choices",
                        )
                    choice = choices[0]
                    if choice.get("index") != 0:
                        return RequestResult(
                            200,
                            now - started,
                            response_bytes=response_bytes,
                            error_code="invalid_choice_index",
                        )
                    current_finish = choice.get("finish_reason")
                    if current_finish is not None:
                        if (
                            not isinstance(current_finish, str)
                            or not current_finish.strip()
                        ):
                            return RequestResult(
                                200,
                                now - started,
                                response_bytes=response_bytes,
                                error_code="invalid_finish_reason",
                            )
                        if current_finish not in ALLOWED_FINISH_REASONS:
                            return RequestResult(
                                200,
                                now - started,
                                response_bytes=response_bytes,
                                error_code="unsupported_finish_reason",
                            )
                        finish_reason = current_finish
                    delta = choice.get("delta")
                    if not isinstance(delta, dict):
                        return RequestResult(
                            200,
                            now - started,
                            response_bytes=response_bytes,
                            error_code="invalid_stream_delta",
                        )
                    if "role" in delta:
                        if delta.get("role") != "assistant":
                            return RequestResult(
                                200,
                                now - started,
                                response_bytes=response_bytes,
                                error_code="invalid_assistant_role",
                            )
                        saw_assistant_role = True
                    content = delta.get("content")
                    reasoning = delta.get("reasoning_content")
                    if (
                        "content" in delta
                        and content is not None
                        and not isinstance(content, str)
                    ):
                        return RequestResult(
                            200,
                            now - started,
                            response_bytes=response_bytes,
                            error_code="invalid_stream_content_type",
                        )
                    if (
                        "reasoning_content" in delta
                        and reasoning is not None
                        and not isinstance(reasoning, str)
                    ):
                        return RequestResult(
                            200,
                            now - started,
                            response_bytes=response_bytes,
                            error_code="invalid_stream_reasoning_type",
                        )
                    output_observed = (
                        isinstance(content, str) and bool(content.strip())
                    ) or (isinstance(reasoning, str) and bool(reasoning.strip()))
                    if output_observed:
                        if not saw_assistant_role:
                            return RequestResult(
                                200,
                                now - started,
                                response_bytes=response_bytes,
                                error_code="output_before_assistant_role",
                            )
                        if first_output_at is None:
                            first_output_at = now
                        output_event_times.append(now)
                    if current_finish is not None:
                        saw_finish = True
            except _ResponseTooLarge:
                return RequestResult(
                    200,
                    time.perf_counter() - started,
                    response_bytes=response_bytes,
                    error_code="response_too_large",
                )
            except (UnicodeDecodeError, ValueError):
                return RequestResult(
                    200,
                    time.perf_counter() - started,
                    response_bytes=response_bytes,
                    error_code="malformed_sse",
                )
            ended = time.perf_counter()
            if not saw_done:
                return RequestResult(
                    200,
                    ended - started,
                    response_bytes=response_bytes,
                    error_code="stream_missing_done",
                )
            if first_output_at is None:
                return RequestResult(
                    200,
                    ended - started,
                    response_bytes=response_bytes,
                    error_code="empty_completion_output",
                )
            if finish_reason is None:
                return RequestResult(
                    200,
                    ended - started,
                    response_bytes=response_bytes,
                    error_code="missing_finish_reason",
                )
            if not saw_usage or prompt_tokens is None or completion_tokens is None:
                return RequestResult(
                    200,
                    ended - started,
                    response_bytes=response_bytes,
                    error_code="missing_usage",
                )
            if completion_tokens > config.max_tokens:
                return RequestResult(
                    200,
                    ended - started,
                    response_bytes=response_bytes,
                    error_code="reported_completion_tokens_exceed_request_cap",
                )
            e2e = ended - started
            ttft = first_output_at - started
            if not saw_assistant_role:
                return RequestResult(
                    200,
                    e2e,
                    response_bytes=response_bytes,
                    error_code="missing_assistant_role",
                )
            decode_span = output_event_times[-1] - first_output_at
            # A single SSE event can contain many tokens. With no second output
            # timestamp there is no observed decode span, so TPOT is unavailable
            # rather than a fabricated zero.
            tpot = (
                decode_span / (completion_tokens - 1)
                if completion_tokens > 1 and len(output_event_times) >= 2
                else None
            )
            inter_chunks = tuple(
                later - earlier
                for earlier, later in zip(output_event_times, output_event_times[1:])
            )
            return RequestResult(
                200,
                e2e,
                completion_tokens=completion_tokens,
                prompt_tokens=prompt_tokens,
                finish_reason=finish_reason,
                ttft_seconds=ttft,
                tpot_seconds=tpot,
                inter_chunk_seconds=inter_chunks,
                response_bytes=response_bytes,
            )
    except httpx.TimeoutException:
        return RequestResult(
            None, time.perf_counter() - started, error_code="request_timeout"
        )
    except httpx.RequestError:
        return RequestResult(
            None, time.perf_counter() - started, error_code="transport_error"
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        return RequestResult(
            None, time.perf_counter() - started, error_code="client_error"
        )


@dataclass
class RunBudget:
    config: RunConfig
    started: float = field(default_factory=time.perf_counter)
    requests_started: int = 0
    requests_completed: int = 0
    requests_cancelled: int = 0
    errors: int = 0
    reserved_output_tokens: int = 0
    in_flight: int = 0
    peak_in_flight: int = 0
    stop_reason: str | None = None

    def elapsed(self) -> float:
        return max(0.0, time.perf_counter() - self.started)

    def _runtime_limit(self) -> str | None:
        if self.elapsed() >= self.config.limits.max_elapsed_seconds:
            return "max_elapsed_time"
        spend = self.config.cost.elapsed_estimate(self.elapsed())
        if spend is not None and spend >= self.config.limits.max_estimated_spend:
            return "max_estimated_spend"
        return None

    def set_stop(self, reason: str) -> None:
        if self.stop_reason is None:
            self.stop_reason = reason

    def check_runtime(self) -> bool:
        runtime = self._runtime_limit()
        if runtime:
            self.set_stop(runtime)
            return False
        return self.stop_reason is None

    def reserve(self) -> bool:
        if self.stop_reason is not None:
            return False
        runtime = self._runtime_limit()
        if runtime:
            self.set_stop(runtime)
            return False
        if self.requests_started >= self.config.limits.max_requests:
            self.set_stop("max_requests")
            return False
        if (
            self.reserved_output_tokens + self.config.max_tokens
            > self.config.limits.max_total_requested_tokens
        ):
            self.set_stop("max_total_requested_tokens")
            return False
        self.requests_started += 1
        self.reserved_output_tokens += self.config.max_tokens
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        return True

    def record(self, result: RequestResult) -> None:
        self.in_flight = max(0, self.in_flight - 1)
        self.requests_completed += 1
        if not result.valid:
            self.errors += 1
            if self.errors >= self.config.limits.max_errors:
                self.set_stop("max_errors")
        runtime = self._runtime_limit()
        if runtime:
            self.set_stop(runtime)

    def record_cancelled(self) -> None:
        self.in_flight = max(0, self.in_flight - 1)
        self.requests_cancelled += 1

    def public_dict(self) -> dict[str, Any]:
        return {
            "requests_started": self.requests_started,
            "requests_completed": self.requests_completed,
            "requests_cancelled": self.requests_cancelled,
            "requests_in_flight": self.in_flight,
            "peak_in_flight": self.peak_in_flight,
            "errors": self.errors,
            "reserved_output_tokens": self.reserved_output_tokens,
            "elapsed_seconds": self.elapsed(),
        }


@dataclass
class BlockOutcome:
    results: list[RequestResult]
    wall_seconds: float
    complete: bool
    scheduler_lag_seconds: list[float] = field(default_factory=list)
    offered_requests: int = 0
    invalid_reason: str | None = None
    peak_in_flight: int = 0
    target_offered_request_rate: float | None = None
    launch_window_seconds: float | None = None


class RunProgress:
    """Carries the latest sanitized report so CLI cancellation can persist it."""

    def __init__(self) -> None:
        self.report: dict[str, Any] | None = None

    def set(self, report: Mapping[str, Any]) -> None:
        self.report = copy.deepcopy(dict(report))

    def snapshot(self) -> dict[str, Any] | None:
        return copy.deepcopy(self.report)


async def _execute_reserved(
    client: httpx.AsyncClient,
    endpoint_url: str,
    config: RunConfig,
    budget: RunBudget,
    messages: Prompt,
) -> RequestResult:
    remaining = config.limits.max_elapsed_seconds - budget.elapsed()
    timeout = min(config.request_timeout_seconds, max(0.001, remaining))
    try:
        async with asyncio.timeout(timeout):
            result = await _native_request(client, endpoint_url, config, messages)
    except TimeoutError:
        if budget.elapsed() >= config.limits.max_elapsed_seconds:
            budget.set_stop("max_elapsed_time")
            result = RequestResult(None, timeout, error_code="max_elapsed_time")
        else:
            result = RequestResult(None, timeout, error_code="request_timeout")
    except asyncio.CancelledError:
        budget.record_cancelled()
        raise
    budget.record(result)
    return result


def _prompt_order(prompts: Prompts, seed: int) -> list[int]:
    indexes = list(range(len(prompts)))
    random.Random(seed).shuffle(indexes)
    return indexes


async def _run_closed_block(
    client: httpx.AsyncClient,
    endpoint_url: str,
    config: RunConfig,
    budget: RunBudget,
    prompts: Prompts,
    *,
    concurrency: int,
    seed: int,
    request_target: int | None,
    duration_target: float | None,
) -> BlockOutcome:
    started = time.perf_counter()
    block_deadline = started + duration_target if duration_target is not None else None
    results: list[RequestResult] = []
    next_index = 0
    order = _prompt_order(prompts, seed)
    active = 0
    peak_in_flight = 0

    async def worker() -> None:
        nonlocal next_index, active, peak_in_flight
        while budget.stop_reason is None:
            if request_target is not None and next_index >= request_target:
                return
            if block_deadline is not None and time.perf_counter() >= block_deadline:
                return
            index = next_index
            next_index += 1
            if not budget.reserve():
                return
            active += 1
            peak_in_flight = max(peak_in_flight, active)
            prompt = prompts[order[index % len(order)]]
            try:
                result = await _execute_reserved(
                    client, endpoint_url, config, budget, prompt
                )
                results.append(result)
            finally:
                active = max(0, active - 1)

    tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
    try:
        pending = set(tasks)
        while pending:
            _done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            if budget.stop_reason and pending:
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                pending.clear()
    except asyncio.CancelledError:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    wall = time.perf_counter() - started
    target_complete = (
        request_target is not None and len(results) == request_target
    ) or (duration_target is not None and wall >= duration_target and bool(results))
    return BlockOutcome(
        results,
        wall,
        complete=target_complete and budget.stop_reason is None,
        offered_requests=next_index,
        invalid_reason=budget.stop_reason,
        peak_in_flight=peak_in_flight,
    )


async def _run_open_block(
    client: httpx.AsyncClient,
    endpoint_url: str,
    config: RunConfig,
    budget: RunBudget,
    prompts: Prompts,
    *,
    request_rate: float,
    max_in_flight: int,
    seed: int,
    request_target: int | None,
    duration_target: float | None,
) -> BlockOutcome:
    started = time.perf_counter()
    block_deadline = started + duration_target if duration_target is not None else None
    order = _prompt_order(prompts, seed)
    results: list[RequestResult] = []
    lags: list[float] = []
    tasks: set[asyncio.Task[RequestResult]] = set()
    launched = 0
    peak_in_flight = 0
    first_launch_at: float | None = None
    last_launch_at: float | None = None

    def harvest() -> None:
        done = {task for task in tasks if task.done()}
        for task in done:
            tasks.remove(task)
            if not task.cancelled() and task.exception() is None:
                results.append(task.result())

    try:
        # Assigned here so Python treats it as the enclosing local updated on
        # every launch below.
        while budget.stop_reason is None:
            harvest()
            if not budget.check_runtime():
                break
            if request_target is not None and launched >= request_target:
                break
            now = time.perf_counter()
            if block_deadline is not None and now >= block_deadline:
                break
            scheduled = started + launched / request_rate
            if now < scheduled:
                await asyncio.sleep(
                    min(
                        scheduled - now,
                        0.05,
                        max(
                            0.001, config.limits.max_elapsed_seconds - budget.elapsed()
                        ),
                    )
                )
                continue
            if len(tasks) >= max_in_flight:
                budget.set_stop("open_loop_backpressure")
                break
            if not budget.reserve():
                break
            launched_at = time.perf_counter()
            lags.append(max(0.0, launched_at - scheduled))
            if first_launch_at is None:
                first_launch_at = launched_at
            last_launch_at = launched_at
            prompt = prompts[order[launched % len(order)]]
            tasks.add(
                asyncio.create_task(
                    _execute_reserved(client, endpoint_url, config, budget, prompt)
                )
            )
            peak_in_flight = max(peak_in_flight, len(tasks))
            launched += 1
        # A completed task may be the one that tripped the error limit. Harvest
        # it before cancelling only the requests that are genuinely still live.
        harvest()
        if budget.stop_reason and tasks:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            tasks.clear()
        elif tasks:
            completed = await asyncio.gather(*tasks)
            results.extend(completed)
            tasks.clear()
    except asyncio.CancelledError:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    wall = time.perf_counter() - started
    target_complete = (
        request_target is not None
        and launched == request_target
        and len(results) == launched
    ) or (duration_target is not None and wall >= duration_target and bool(results))
    return BlockOutcome(
        results,
        wall,
        complete=target_complete and budget.stop_reason is None,
        scheduler_lag_seconds=lags,
        offered_requests=launched,
        invalid_reason=budget.stop_reason,
        peak_in_flight=peak_in_flight,
        target_offered_request_rate=request_rate,
        launch_window_seconds=(
            last_launch_at - first_launch_at
            if first_launch_at is not None
            and last_launch_at is not None
            and launched >= 2
            else None
        ),
    )


def _request_counts(results: Sequence[RequestResult]) -> dict[str, Any]:
    statuses: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    finishes: Counter[str] = Counter()
    for result in results:
        statuses[
            str(result.status_code) if result.status_code is not None else "transport"
        ] += 1
        if result.error_code:
            errors[result.error_code] += 1
        if result.finish_reason:
            finishes[result.finish_reason] += 1
    return {
        "attempted": len(results),
        "valid": sum(result.valid for result in results),
        "invalid": sum(not result.valid for result in results),
        "status_counts": dict(sorted(statuses.items())),
        "error_counts": dict(sorted(errors.items())),
        "finish_reason_counts": dict(sorted(finishes.items())),
    }


def _diagnostic_metrics(
    results: Sequence[RequestResult],
    wall_seconds: float,
    config: RunConfig,
    *,
    seed: int,
) -> dict[str, Any]:
    valid = [result for result in results if result.valid]
    completion_tokens = sum(result.completion_tokens or 0 for result in valid)
    prompt_tokens = sum(result.prompt_tokens or 0 for result in valid)
    e2e = [result.e2e_seconds for result in valid]
    ttft = [result.ttft_seconds for result in valid if result.ttft_seconds is not None]
    tpot = [result.tpot_seconds for result in valid if result.tpot_seconds is not None]
    inter_chunk_count = sum(len(result.inter_chunk_seconds) for result in valid)
    inter_chunk: list[float] = []
    inter_chunk_limit = 4_096
    inter_chunk_rng = random.Random(seed + 3)
    seen = 0
    for result in valid:
        for gap in result.inter_chunk_seconds:
            seen += 1
            if len(inter_chunk) < inter_chunk_limit:
                inter_chunk.append(gap)
            else:
                replacement = inter_chunk_rng.randrange(seen)
                if replacement < inter_chunk_limit:
                    inter_chunk[replacement] = gap
    slo_configured = config.p95_slo_ms is not None or config.ttft_slo_ms is not None
    slo_pass = 0
    if slo_configured:
        for result in valid:
            e2e_ok = (
                config.p95_slo_ms is None
                or result.e2e_seconds * 1000 <= config.p95_slo_ms
            )
            ttft_ok = config.ttft_slo_ms is None or (
                result.ttft_seconds is not None
                and result.ttft_seconds * 1000 <= config.ttft_slo_ms
            )
            slo_pass += bool(e2e_ok and ttft_ok)
    return {
        "valid_response_count": len(valid),
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "requests_per_second": len(valid) / wall_seconds if wall_seconds > 0 else None,
        "output_tokens_per_second": completion_tokens / wall_seconds
        if wall_seconds > 0
        else None,
        "error_rate": (len(results) - len(valid)) / len(results) if results else None,
        "e2e_latency_ms": summarize_distribution_ms(e2e, seed=seed),
        "ttft_ms": summarize_distribution_ms(ttft, seed=seed + 1),
        "tpot_ms": summarize_distribution_ms(tpot, seed=seed + 2),
        "itl_ms": {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "source": "unavailable",
            "unavailable_reason": "native SSE chunks do not prove token boundaries",
        },
        "inter_chunk_latency_ms": {
            **summarize_distribution_ms(inter_chunk, seed=seed + 3),
            "count": inter_chunk_count,
            "analysis_sample_count": len(inter_chunk),
            "analysis_sampling": "all gaps up to 4096, then deterministic reservoir sampling",
            "source": "client_observed_nonempty_sse_chunks",
            "not_equivalent_to_itl": True,
        },
        "slo_goodput": (
            {
                "passing_requests": slo_pass,
                "denominator_valid_requests": len(valid),
                "requests_per_second": slo_pass / wall_seconds
                if wall_seconds > 0
                else None,
                "e2e_threshold_ms": config.p95_slo_ms,
                "ttft_threshold_ms": config.ttft_slo_ms,
            }
            if slo_configured
            else None
        ),
    }


def _block_report(
    outcome: BlockOutcome, config: RunConfig, *, index: int, seed: int
) -> dict[str, Any]:
    counts = _request_counts(outcome.results)
    valid = outcome.complete and counts["invalid"] == 0 and bool(outcome.results)
    reasons: list[str] = []
    if not outcome.complete:
        reasons.append(outcome.invalid_reason or "incomplete_block")
    if counts["invalid"]:
        reasons.append("one_or_more_invalid_responses")
    diagnostic = _diagnostic_metrics(
        outcome.results, outcome.wall_seconds, config, seed=seed
    )
    achieved_offered_rate: float | None = None
    offered_rate_relative_error: float | None = None
    if (
        outcome.target_offered_request_rate is not None
        and outcome.launch_window_seconds is not None
        and outcome.launch_window_seconds > 0
        and outcome.offered_requests >= 2
    ):
        achieved_offered_rate = (
            outcome.offered_requests - 1
        ) / outcome.launch_window_seconds
        offered_rate_relative_error = (
            abs(achieved_offered_rate - outcome.target_offered_request_rate)
            / outcome.target_offered_request_rate
        )
    scheduler_lag = summarize_distribution_ms(
        outcome.scheduler_lag_seconds, seed=seed + 4
    )
    lag_interval_ratio = (
        float(scheduler_lag["p95"]) / (1000.0 / outcome.target_offered_request_rate)
        if outcome.target_offered_request_rate is not None
        and scheduler_lag.get("p95") is not None
        else None
    )
    open_loop_target_achieved = outcome.target_offered_request_rate is None or (
        offered_rate_relative_error is not None
        and offered_rate_relative_error <= OPEN_LOOP_RATE_RELATIVE_TOLERANCE
        and lag_interval_ratio is not None
        and lag_interval_ratio <= OPEN_LOOP_SCHEDULER_LAG_INTERVAL_TOLERANCE
    )
    return {
        "block_index": index,
        "valid": valid,
        "invalid_reasons": reasons,
        "wall_duration_seconds": outcome.wall_seconds,
        "request_counts": counts,
        "offered_requests": outcome.offered_requests,
        "target_offered_request_rate": outcome.target_offered_request_rate,
        "launch_window_seconds": outcome.launch_window_seconds,
        "achieved_offered_request_rate": achieved_offered_rate,
        "offered_rate_relative_error": offered_rate_relative_error,
        "scheduler_lag_interval_ratio_p95": lag_interval_ratio,
        "open_loop_target_achieved": open_loop_target_achieved,
        "observed_peak_in_flight": outcome.peak_in_flight,
        "scheduler_lag_ms": scheduler_lag,
        "metrics": diagnostic if valid else None,
        "diagnostic_metrics": diagnostic,
    }


def _aggregate_condition(
    condition: LoadCondition,
    outcomes: Sequence[BlockOutcome],
    block_reports: Sequence[Mapping[str, Any]],
    warmup_summary: Mapping[str, Any],
    config: RunConfig,
    *,
    seed: int,
) -> dict[str, Any]:
    all_results = [result for outcome in outcomes for result in outcome.results]
    wall = sum(outcome.wall_seconds for outcome in outcomes)
    counts = _request_counts(all_results)
    warmup_valid = warmup_summary["invalid"] == 0
    valid = (
        len(block_reports) == config.blocks
        and all(bool(block["valid"]) for block in block_reports)
        and warmup_valid
        and counts["invalid"] == 0
    )
    diagnostic = _diagnostic_metrics(all_results, wall, config, seed=seed)
    observed_peak = max((outcome.peak_in_flight for outcome in outcomes), default=0)
    if condition.kind == "closed_loop":
        load_target_achieved = bool(outcomes) and all(
            outcome.peak_in_flight == condition.max_in_flight for outcome in outcomes
        )
    else:
        load_target_achieved = bool(block_reports) and all(
            block.get("open_loop_target_achieved") is True for block in block_reports
        )
    decision_grade = (
        config.mode == "benchmark"
        and valid
        and load_target_achieved
        and len(block_reports) >= MIN_DECISION_BLOCKS
        and (counts["valid"] >= MIN_DECISION_REQUESTS or wall >= MIN_DECISION_SECONDS)
    )
    reasons: list[str] = []
    if not warmup_valid:
        reasons.append("warmup_failed")
    if any(not bool(block["valid"]) for block in block_reports):
        reasons.append("invalid_or_incomplete_block")
    if config.mode == "smoke":
        reasons.append("smoke_mode_is_not_decision_grade")
    elif len(block_reports) < MIN_DECISION_BLOCKS:
        reasons.append("fewer_than_three_blocks")
    elif counts["valid"] < MIN_DECISION_REQUESTS and wall < MIN_DECISION_SECONDS:
        reasons.append("measurement_floor_not_met")
    if config.mode == "benchmark" and not load_target_achieved:
        reasons.append(
            "closed_loop_target_concurrency_not_observed"
            if condition.kind == "closed_loop"
            else "open_loop_target_rate_not_achieved"
        )

    metrics: dict[str, Any] | None = None
    if valid:
        block_throughputs = [
            float(block["metrics"]["output_tokens_per_second"])
            for block in block_reports
        ]
        block_request_rates = [
            float(block["metrics"]["requests_per_second"]) for block in block_reports
        ]
        metrics = dict(diagnostic)
        metrics["block_mean_output_tokens_per_second"] = math.fsum(
            block_throughputs
        ) / len(block_throughputs)
        metrics["block_mean_output_tokens_per_second_ci"] = t_interval_95(
            block_throughputs
        )
        metrics["block_mean_requests_per_second"] = math.fsum(
            block_request_rates
        ) / len(block_request_rates)
        metrics["block_mean_requests_per_second_ci"] = t_interval_95(
            block_request_rates
        )
        metrics["independent_ci_unit"] = "repeated_block"
        for metric_name in ("e2e_latency_ms", "ttft_ms"):
            block_p95_values = [
                float(block["metrics"][metric_name]["p95"])
                for block in block_reports
                if block["metrics"][metric_name]["p95"] is not None
            ]
            metrics[metric_name]["p95_repeated_block_ci"] = t_interval_95(
                block_p95_values
            )
        if config.cost.kind == "dedicated_hourly" and diagnostic["completion_tokens"]:
            total_cost = float(config.cost.total_hourly_rate) * wall / 3600.0
            metrics["cost_per_million_output_tokens"] = (
                total_cost / int(diagnostic["completion_tokens"]) * 1_000_000.0
            )
            metrics["cost_metric_basis"] = (
                "dedicated hourly rate times measured condition wall time"
            )
        else:
            metrics["cost_per_million_output_tokens"] = None
            metrics["cost_metric_basis"] = (
                "whole-run or active-second cost cannot be allocated to this condition"
            )
    launch_gaps = sum(max(0, outcome.offered_requests - 1) for outcome in outcomes)
    launch_window = sum(outcome.launch_window_seconds or 0.0 for outcome in outcomes)
    achieved_offered_rate = (
        launch_gaps / launch_window
        if condition.kind == "open_loop" and launch_gaps > 0 and launch_window > 0
        else None
    )
    return {
        "condition": condition.public_dict(),
        "valid": valid,
        "decision_grade": decision_grade,
        "decision_ineligible_reasons": sorted(set(reasons)),
        "qualification_floor": {
            "minimum_valid_requests": MIN_DECISION_REQUESTS,
            "or_minimum_measured_seconds": MIN_DECISION_SECONDS,
            "minimum_blocks": MIN_DECISION_BLOCKS,
        },
        "warmup": dict(warmup_summary),
        "blocks": list(block_reports),
        "request_counts": counts,
        "measured_wall_seconds": wall,
        "target_offered_request_rate": condition.value
        if condition.kind == "open_loop"
        else None,
        "achieved_offered_request_rate": achieved_offered_rate,
        "offered_rate_relative_error": (
            abs(achieved_offered_rate - float(condition.value)) / float(condition.value)
            if achieved_offered_rate is not None
            else None
        ),
        "open_loop_target_achieved": (
            load_target_achieved if condition.kind == "open_loop" else None
        ),
        "observed_peak_in_flight": observed_peak,
        "metrics": metrics,
        "diagnostic_metrics": diagnostic,
    }


def _best_tested(
    conditions: Sequence[Mapping[str, Any]], config: RunConfig
) -> dict[str, Any]:
    label = (
        "best_tested_concurrency"
        if config.conditions[0].kind == "closed_loop"
        else "best_tested_request_rate"
    )
    valid = [
        condition
        for condition in conditions
        if condition.get("valid") and condition.get("metrics")
    ]
    if not valid:
        return {
            "field": label,
            "available": False,
            "state": "invalid",
            "reason": "no_valid_conditions",
            "optimum_found": False,
        }
    eligible = valid
    if config.p95_slo_ms is not None:
        eligible = [
            condition
            for condition in valid
            if condition["metrics"]["e2e_latency_ms"]["p95"] is not None
            and condition["metrics"]["e2e_latency_ms"]["p95_repeated_block_ci"]["high"]
            is not None
            and condition["metrics"]["e2e_latency_ms"]["p95_repeated_block_ci"]["high"]
            <= config.p95_slo_ms
        ]
    if config.ttft_slo_ms is not None:
        eligible = [
            condition
            for condition in eligible
            if condition["metrics"]["ttft_ms"]["p95"] is not None
            and condition["metrics"]["ttft_ms"]["p95_repeated_block_ci"]["high"]
            is not None
            and condition["metrics"]["ttft_ms"]["p95_repeated_block_ci"]["high"]
            <= config.ttft_slo_ms
        ]
    if not eligible:
        return {
            "field": label,
            "available": False,
            "state": "inconclusive",
            "reason": "no_valid_condition_meets_slo",
            "optimum_found": False,
        }
    selected = max(
        eligible,
        key=lambda item: float(item["metrics"]["block_mean_output_tokens_per_second"]),
    )
    selected_value = float(selected["condition"]["value"])
    maximum_tested = max(float(item["condition"]["value"]) for item in conditions)
    boundary = selected_value == maximum_tested
    state = "not_applicable_smoke" if config.mode == "smoke" else "supported"
    reasons: list[str] = []
    output_tokens_per_response: list[float] = []
    # Output work must be comparable across every valid tested condition, not
    # only the subset that survives SLO filtering.  Otherwise a short-output
    # condition could be the sole SLO qualifier and make this guard vacuous.
    for condition in valid:
        token_total = condition["metrics"].get("completion_tokens")
        valid_count = condition["metrics"].get("valid_response_count")
        if (
            not isinstance(token_total, int)
            or isinstance(token_total, bool)
            or token_total <= 0
            or not isinstance(valid_count, int)
            or isinstance(valid_count, bool)
            or valid_count <= 0
        ):
            output_tokens_per_response = []
            break
        output_tokens_per_response.append(token_total / valid_count)
    output_token_relative_spread: float | None = None
    output_work_comparable = bool(output_tokens_per_response)
    if output_work_comparable:
        maximum_output = max(output_tokens_per_response)
        output_token_relative_spread = (
            maximum_output - min(output_tokens_per_response)
        ) / maximum_output
        output_work_comparable = (
            output_token_relative_spread <= BEST_TESTED_OUTPUT_TOKEN_TOLERANCE
        )
    if config.mode == "smoke":
        reasons.append("smoke_sample_is_not_decision_grade")
    elif any(not condition.get("decision_grade") for condition in valid):
        state = "inconclusive"
        reasons.append("one_or_more_tested_conditions_are_not_decision_grade")
    elif not selected.get("decision_grade"):
        state = "inconclusive"
        reasons.append("selected_condition_is_not_decision_grade")
    if not output_work_comparable:
        if config.mode == "benchmark":
            state = "inconclusive"
        reasons.append(
            "completion_tokens_per_response_not_comparable_across_conditions"
        )
    # Native multi-load execution is currently condition-major.  Repeated
    # blocks quantify within-condition variation, but they do not remove an
    # order/drift confound between conditions.  Keep the selected value as a
    # useful descriptive observation while refusing a supported claim until a
    # counterbalanced block-major scheduler is implemented.
    if config.mode == "benchmark" and len(valid) > 1:
        state = "inconclusive"
        reasons.append("multi_condition_order_not_counterbalanced")
    if config.mode == "benchmark" and boundary:
        state = "inconclusive"
        reasons.append("search_boundary_reached")
    if len(eligible) > 1:
        overlapping = [
            item
            for item in eligible
            if item is not selected
            and intervals_overlap(
                selected["metrics"]["block_mean_output_tokens_per_second_ci"],
                item["metrics"]["block_mean_output_tokens_per_second_ci"],
            )
        ]
        if overlapping:
            if config.mode == "benchmark":
                state = "inconclusive"
            reasons.append("throughput_confidence_intervals_overlap")
    return {
        "field": label,
        "available": True,
        "value": int(selected_value) if selected_value.is_integer() else selected_value,
        "condition_id": selected["condition"]["id"],
        "block_mean_output_tokens_per_second": selected["metrics"][
            "block_mean_output_tokens_per_second"
        ],
        "block_mean_output_tokens_per_second_ci": selected["metrics"][
            "block_mean_output_tokens_per_second_ci"
        ],
        "pooled_output_tokens_per_second": selected["metrics"][
            "output_tokens_per_second"
        ],
        "completion_tokens_per_response_relative_spread": output_token_relative_spread,
        "completion_tokens_per_response_tolerance": BEST_TESTED_OUTPUT_TOKEN_TOLERANCE,
        "state": state,
        "reasons": reasons,
        "boundary_reached": boundary,
        "optimum_found": False,
        "claim": (
            "descriptive smoke observation only; never a production recommendation"
            if config.mode == "smoke"
            else "best condition among only the tested values for this workload"
        ),
    }


def _manifest(
    config: RunConfig, prompts: Prompts, warmup_prompts: Prompts
) -> dict[str, Any]:
    engine_flags = {name: value for name, value in config.engine_flags}
    chunked_prefill_present = any(
        name.replace("_", "-").lower() == "enable-chunked-prefill"
        for name in engine_flags
    )
    return {
        "manifest_version": "1.0",
        "tool": {"name": "throttle-bench", "version": __version__},
        "engine": {
            "backend": config.backend,
            "backend_version": "native-protocol-1",
            "http_client_version": httpx.__version__
            if config.backend == "native"
            else None,
            "server_version": config.server_version,
            "effective_flags": engine_flags,
            "effective_flags_provenance": config.engine_flags_provenance,
        },
        "model": {"id": config.model, "immutable_revision": config.model_revision},
        "runtime": {
            "image_digest": config.image_digest,
            "gpu": config.gpu,
            "gpu_fingerprint_sha256": hashlib.sha256(
                config.gpu_fingerprint.encode("utf-8")
            ).hexdigest(),
            "gpu_fingerprint_supplied": config.gpu_fingerprint != "unknown",
            "cuda_version": config.cuda_version,
            "driver_version": config.driver_version,
        },
        "workload": {
            "seed": config.seed,
            "measured_sha256": canonical_workload_hash(prompts),
            "warmup_sha256": canonical_workload_hash(warmup_prompts),
            "measured_prompt_count": len(prompts),
            "warmup_prompt_count": len(warmup_prompts),
            "warmup_is_separate": canonical_workload_hash(prompts)
            != canonical_workload_hash(warmup_prompts),
            "warmup_prompts_disjoint": _prompt_identity_hashes(prompts).isdisjoint(
                _prompt_identity_hashes(warmup_prompts)
            ),
            "cache_policy": config.cache_policy,
        },
        "request": {
            "type": "chat_completions",
            "temperature": 0,
            "max_tokens": config.max_tokens,
            "stop": None,
            "stream": config.stream,
            "timeout_seconds": config.request_timeout_seconds,
        },
        "traffic": {
            "conditions": [condition.public_dict() for condition in config.conditions],
            "blocks": config.blocks,
            "requests_per_block": config.requests_per_block,
            "block_duration_seconds": config.block_duration_seconds,
            "warmup_requests_per_condition": config.warmup_requests_per_condition,
            "p95_slo_ms": config.p95_slo_ms,
            "ttft_slo_ms": config.ttft_slo_ms,
            "open_loop_rate_relative_tolerance": OPEN_LOOP_RATE_RELATIVE_TOLERANCE,
            "open_loop_scheduler_lag_interval_tolerance": OPEN_LOOP_SCHEDULER_LAG_INTERVAL_TOLERANCE,
        },
        "metric_definitions": {
            "e2e_latency": "client request start through validated terminal stream completion",
            "ttft": "client request start to first non-whitespace output-bearing SSE delta; role-only chunks are ignored",
            "tpot": "first-to-last output-bearing SSE delta divided by completion-token gaps; unavailable unless at least two output-bearing SSE events are observed",
            "itl": "unavailable in native mode because SSE chunks do not prove token boundaries",
            "inter_chunk_latency": "client-observed gap between nonempty SSE output chunks; explicitly not ITL",
            "throughput": "validated completion tokens divided by measured block wall time",
            "decision_throughput": "arithmetic mean of repeated-block output-token throughput, paired with its Student-t block interval; pooled tokens/wall remains separately reported",
            "slo_goodput": "validated requests meeting every configured per-request SLO divided by measured block wall time",
            "slo_decision_ci": "Student-t 95% interval over repeated-block p95 values; request-bootstrap percentile intervals are diagnostic only",
        },
        "provenance": {
            "evidence_source": config.evidence_source,
            "variant": config.variant,
            "sequence_position": config.sequence_position,
        },
        "cost": config.cost.public_dict(config.limits.max_elapsed_seconds),
        "safety": {
            "limits": config.limits.public_dict(),
            "overrides": {
                "insecure_http": config.allow_insecure_http,
                "unknown_cost_acknowledged": config.allow_unknown_cost,
                "guidellm_validation_gaps_acknowledged": config.guidellm_gaps_acknowledged,
            },
            "ambient_proxy_environment_used": False,
            "redirects_followed": False,
        },
        "optimization_credit_exclusions": (
            [
                {
                    "feature": "enable_chunked_prefill",
                    "reason": "vLLM V1 enables chunked prefill by default when possible; its presence alone receives no optimization credit",
                }
            ]
            if chunked_prefill_present
            else []
        ),
    }


def _initial_report(
    config: RunConfig, prompts: Prompts, warmup_prompts: Prompts
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "generated_at": _utc_now(),
        "started_at": _utc_now(),
        "completed_at": None,
        "mode": config.mode,
        "status": "running",
        "decision_eligible": False,
        "manifest": _manifest(config, prompts, warmup_prompts),
        "conditions": [],
        "best_tested": {
            "available": False,
            "state": "not_evaluated",
            "optimum_found": False,
        },
        "run_totals": {},
        "cost_summary": {
            "kind": config.cost.kind,
            "total_cost": None,
            "cost_per_million_output_tokens": None,
        },
        "stop_reason": None,
        "disclaimer": (
            "Measurements describe only this exact workload and declared manifest. "
            "They are not a universal optimization, production recommendation, or savings claim."
        ),
    }


def _finalize_report(
    report: dict[str, Any],
    config: RunConfig,
    budget: RunBudget,
    *,
    status: str,
    stop_reason: str | None,
) -> None:
    budget.check_runtime()
    if status == "complete" and budget.stop_reason:
        status = "stopped"
        stop_reason = budget.stop_reason
    report["status"] = status
    report["stop_reason"] = stop_reason
    incomplete_best = {
        "field": (
            "best_tested_concurrency"
            if config.conditions[0].kind == "closed_loop"
            else "best_tested_request_rate"
        ),
        "available": False,
        "state": "inconclusive" if report["status"] != "complete" else "not_evaluated",
        "reason": "partial_or_failed_run"
        if report["status"] != "complete"
        else "no_completed_conditions",
        "optimum_found": False,
    }
    report["best_tested"] = (
        _best_tested(report["conditions"], config)
        if report["status"] == "complete" and report["conditions"]
        else incomplete_best
    )
    total_tokens = sum(
        int(condition.get("diagnostic_metrics", {}).get("completion_tokens", 0))
        for condition in report["conditions"]
    )
    # Diagnostic CIs are deliberately bounded, but they still consume time.
    # Reconcile the hard deadline once more after all report computation so a
    # run can never be serialized as complete after crossing its cap.
    budget.check_runtime()
    final_elapsed = budget.elapsed()
    if final_elapsed >= config.limits.max_elapsed_seconds:
        budget.set_stop("max_elapsed_time")
    final_spend_estimate = config.cost.elapsed_estimate(final_elapsed)
    if (
        final_spend_estimate is not None
        and final_spend_estimate >= config.limits.max_estimated_spend
    ):
        budget.set_stop("max_estimated_spend")
    if report["status"] == "complete" and budget.stop_reason:
        report["status"] = "stopped"
        report["stop_reason"] = budget.stop_reason
        report["best_tested"] = {
            **incomplete_best,
            "state": "inconclusive",
            "reason": "partial_or_failed_run",
        }
    decision_reasons: list[str] = []
    if config.mode != "benchmark":
        decision_reasons.append("smoke_mode_is_not_decision_grade")
    if report["status"] != "complete":
        decision_reasons.append("run_is_not_complete")
    if not report["conditions"] or not all(
        condition.get("decision_grade") for condition in report["conditions"]
    ):
        decision_reasons.append("one_or_more_conditions_are_not_decision_grade")
    if report["best_tested"].get("state") != "supported":
        decision_reasons.append("best_tested_result_is_not_statistically_supported")
    if config.backend != "native":
        decision_reasons.append("strict_native_completion_validation_required")
    if not config.stream:
        decision_reasons.append("streaming_required_for_decision_grade")
    if config.evidence_source != "live_inference":
        decision_reasons.append("live_inference_evidence_required")
    if config.engine_flags_provenance != "runtime_verified":
        decision_reasons.append("runtime_verified_engine_flags_required")
    if config.cache_policy == "unknown":
        decision_reasons.append("explicit_cache_policy_required")
    if not re.fullmatch(r"(?:[^\s]+@)?sha256:[0-9a-f]{64}", config.image_digest):
        decision_reasons.append("immutable_image_digest_required")
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", config.model_revision):
        decision_reasons.append("immutable_model_revision_required")
    if any(
        value == "unknown"
        for value in (
            config.gpu,
            config.gpu_fingerprint,
            config.cuda_version,
            config.driver_version,
            config.server_version,
        )
    ):
        decision_reasons.append("complete_runtime_provenance_required")
    if not report["manifest"]["workload"].get("warmup_prompts_disjoint"):
        decision_reasons.append("warmup_prompts_must_be_disjoint")
    report["decision_ineligible_reasons"] = sorted(set(decision_reasons))
    report["decision_eligible"] = not decision_reasons
    if decision_reasons and report["best_tested"].get("state") == "supported":
        report["best_tested"]["state"] = "inconclusive"
        report["best_tested"].setdefault("reasons", []).append(
            "run_manifest_or_evidence_is_not_decision_eligible"
        )
    budget.check_runtime()
    if report["status"] == "complete" and budget.stop_reason:
        report["status"] = "stopped"
        report["stop_reason"] = budget.stop_reason
        report["decision_eligible"] = False
        report["decision_ineligible_reasons"] = sorted(
            set(report["decision_ineligible_reasons"] + ["run_is_not_complete"])
        )
        report["best_tested"] = {
            **incomplete_best,
            "state": "inconclusive",
            "reason": "partial_or_failed_run",
        }
    total_cost, basis = config.cost.final_cost(final_elapsed)
    report["cost_summary"] = {
        "kind": config.cost.kind,
        "total_cost": total_cost,
        "basis": basis,
        "completion_tokens": total_tokens,
        "cost_per_million_output_tokens": (
            total_cost / total_tokens * 1_000_000.0
            if total_cost is not None and total_tokens > 0
            else None
        ),
    }
    run_totals = budget.public_dict()
    run_totals["elapsed_seconds"] = final_elapsed
    report["run_totals"] = run_totals
    report["completed_at"] = _utc_now()


async def run_native(
    config: RunConfig,
    prompts: Prompts,
    warmup_prompts: Prompts,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    progress: RunProgress | None = None,
) -> dict[str, Any]:
    """Run native traffic with continuous hard-limit enforcement."""

    validate_config(config)
    if config.backend != "native":
        raise ValueError("run_native requires the native backend")
    checked_prompts = tuple(
        _normalize_messages(list(messages), index)
        for index, messages in enumerate(prompts, start=1)
    )
    checked_warmups = tuple(
        _normalize_messages(list(messages), index)
        for index, messages in enumerate(warmup_prompts, start=1)
    )
    if not checked_prompts or not checked_warmups:
        raise ValueError("measured and warm-up prompt sets must both be non-empty")
    report = _initial_report(config, checked_prompts, checked_warmups)
    progress = progress or RunProgress()
    progress.set(report)
    budget = RunBudget(config)
    endpoint_url = normalize_chat_completions_url(
        config.endpoint.url, allow_insecure_http=config.allow_insecure_http
    )
    limits = httpx.Limits(
        max_connections=config.limits.max_concurrency,
        max_keepalive_connections=config.limits.max_concurrency,
    )
    client_timeout = httpx.Timeout(config.request_timeout_seconds)
    try:
        async with httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {config.endpoint.api_key}",
                "Accept-Encoding": "identity",
            },
            timeout=client_timeout,
            limits=limits,
            transport=transport,
            trust_env=False,
            follow_redirects=False,
        ) as client:
            for condition_index, condition in enumerate(config.conditions):
                if budget.stop_reason:
                    break
                entry: dict[str, Any] = {
                    "condition": condition.public_dict(),
                    "valid": False,
                    "decision_grade": False,
                    "warmup": {
                        "attempted": 0,
                        "valid": 0,
                        "invalid": 0,
                        "status_counts": {},
                        "error_counts": {},
                        "finish_reason_counts": {},
                    },
                    "blocks": [],
                }
                report["conditions"].append(entry)
                progress.set(report)
                if config.warmup_requests_per_condition:
                    warmup = await _run_closed_block(
                        client,
                        endpoint_url,
                        config,
                        budget,
                        checked_warmups,
                        concurrency=min(
                            condition.max_in_flight,
                            config.warmup_requests_per_condition,
                        ),
                        seed=config.seed + condition_index * 10_000 - 1,
                        request_target=config.warmup_requests_per_condition,
                        duration_target=None,
                    )
                    entry["warmup"] = _request_counts(warmup.results)
                    progress.set(report)
                outcomes: list[BlockOutcome] = []
                for block_index in range(config.blocks):
                    if budget.stop_reason:
                        break
                    block_seed = config.seed + condition_index * 10_000 + block_index
                    if condition.kind == "closed_loop":
                        outcome = await _run_closed_block(
                            client,
                            endpoint_url,
                            config,
                            budget,
                            checked_prompts,
                            concurrency=int(condition.value),
                            seed=block_seed,
                            request_target=config.requests_per_block,
                            duration_target=config.block_duration_seconds,
                        )
                    else:
                        outcome = await _run_open_block(
                            client,
                            endpoint_url,
                            config,
                            budget,
                            checked_prompts,
                            request_rate=condition.value,
                            max_in_flight=condition.max_in_flight,
                            seed=block_seed,
                            request_target=config.requests_per_block,
                            duration_target=config.block_duration_seconds,
                        )
                    outcomes.append(outcome)
                    entry["blocks"].append(
                        _block_report(
                            outcome, config, index=block_index + 1, seed=block_seed
                        )
                    )
                    budget.check_runtime()
                    progress.set(report)
                aggregate = _aggregate_condition(
                    condition,
                    outcomes,
                    entry["blocks"],
                    entry["warmup"],
                    config,
                    seed=config.seed + condition_index * 10_000,
                )
                entry.clear()
                entry.update(aggregate)
                budget.check_runtime()
                progress.set(report)
        budget.check_runtime()
        if budget.stop_reason:
            status = "stopped"
        elif not report["conditions"] or any(
            not condition.get("valid") for condition in report["conditions"]
        ):
            status = "failed"
        else:
            status = "complete"
        _finalize_report(
            report, config, budget, status=status, stop_reason=budget.stop_reason
        )
        progress.set(report)
        return report
    except asyncio.CancelledError:
        budget.set_stop("cancelled_by_user")
        _finalize_report(
            report, config, budget, status="cancelled", stop_reason="cancelled_by_user"
        )
        progress.set(report)
        raise
