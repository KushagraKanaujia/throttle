"""Safe, reproducible request controls for native chat-completion traffic."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping

from .models import RunConfig

REQUEST_PROFILE_VERSION = "1.0"
MAX_SAFE_INTEGER = 9_007_199_254_740_991
MAX_REQUEST_FIELDS = 32

PROTECTED_REQUEST_FIELDS = frozenset(
    {
        "model",
        "messages",
        "temperature",
        "top_p",
        "seed",
        "max_tokens",
        "stream",
        "stream_options",
        "stop",
        "n",
        "tools",
        "tool_choice",
        "functions",
        "function_call",
        "response_format",
        "logit_bias",
        "user",
    }
)

FORBIDDEN_NAME_MARKERS = (
    "api_key",
    "apikey",
    "auth",
    "bearer",
    "credential",
    "endpoint",
    "header",
    "password",
    "secret",
    "token",
    "url",
)

REQUEST_PROFILE_CONTROLLED_PATHS: tuple[tuple[str, ...], ...] = (
    ("request", "profile_version"),
    ("request", "top_p"),
    ("request", "request_seed"),
    ("request", "extensions"),
    ("request", "profile_sha256"),
)

_PROFILE_KEYS = frozenset(
    {
        "profile_version",
        "type",
        "temperature",
        "top_p",
        "request_seed",
        "max_tokens",
        "stop",
        "stream",
        "timeout_seconds",
        "extensions",
        "profile_sha256",
    }
)


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _valid_extension_scalar(value: object) -> bool:
    if value is None or isinstance(value, bool):
        return True
    if isinstance(value, int):
        return not isinstance(value, bool) and abs(value) <= MAX_SAFE_INTEGER
    if isinstance(value, float):
        return math.isfinite(value) and abs(value) <= MAX_SAFE_INTEGER
    if isinstance(value, str):
        return bool(re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", value))
    return False


def canonical_json_number(value: int | float) -> int | float:
    parsed = float(value)
    return int(parsed) if parsed.is_integer() else parsed


def _canonical_extension_scalar(
    value: str | int | float | bool | None,
) -> str | int | float | bool | None:
    if isinstance(value, float):
        return canonical_json_number(value)
    return value


def validate_request_fields(
    fields: tuple[tuple[str, str | int | float | bool | None], ...]
) -> None:
    if not isinstance(fields, tuple):
        raise ValueError("request_fields must be a tuple")
    if len(fields) > MAX_REQUEST_FIELDS:
        raise ValueError(f"at most {MAX_REQUEST_FIELDS} request fields are allowed")
    seen: set[str] = set()
    for item in fields:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError("each request field must be a name/value pair")
        name, value = item
        if not isinstance(name, str) or not re.fullmatch(
            r"[a-z][a-z0-9_]{0,63}", name
        ):
            raise ValueError(
                "request field names must use lower-case letters, numbers, and '_'"
            )
        if (
            name in PROTECTED_REQUEST_FIELDS
            or name == "key"
            or name.endswith("_key")
            or any(marker in name for marker in FORBIDDEN_NAME_MARKERS)
        ):
            raise ValueError(f"request field {name!r} is protected or unsafe")
        if name in seen:
            raise ValueError(f"duplicate request field: {name}")
        seen.add(name)
        if not _valid_extension_scalar(value):
            raise ValueError(
                f"request field {name!r} must contain a safe JSON scalar"
            )


def request_fields_dict(
    fields: tuple[tuple[str, str | int | float | bool | None], ...]
) -> dict[str, str | int | float | bool | None]:
    validate_request_fields(fields)
    return {
        name: _canonical_extension_scalar(value) for name, value in sorted(fields)
    }


def _profile_sha256(profile: Mapping[str, Any]) -> str:
    unsealed = {key: value for key, value in profile.items() if key != "profile_sha256"}
    encoded = json.dumps(
        unsealed,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_request_manifest(config: RunConfig) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "profile_version": REQUEST_PROFILE_VERSION,
        "type": "chat_completions",
        "temperature": canonical_json_number(config.temperature),
        "top_p": (
            canonical_json_number(config.top_p) if config.top_p is not None else None
        ),
        "request_seed": config.request_seed,
        "max_tokens": config.max_tokens,
        "stop": None,
        "stream": config.stream,
        "timeout_seconds": config.request_timeout_seconds,
        "extensions": request_fields_dict(config.request_fields),
    }
    profile["profile_sha256"] = _profile_sha256(profile)
    return profile


def request_manifest_reason(request: object) -> str | None:
    """Validate legacy fixed requests and sealed request-profile manifests."""

    if not isinstance(request, Mapping):
        return "incomplete_manifest_contract"
    profile_version = request.get("profile_version")
    if profile_version is None:
        required = {
            "type",
            "temperature",
            "max_tokens",
            "stop",
            "stream",
            "timeout_seconds",
        }
        if set(request) != required:
            return "incomplete_manifest_contract"
        if (
            request.get("type") != "chat_completions"
            or request.get("temperature") != 0
            or request.get("stop") is not None
            or not isinstance(request.get("stream"), bool)
            or not isinstance(request.get("max_tokens"), int)
            or isinstance(request.get("max_tokens"), bool)
            or int(request["max_tokens"]) <= 0
            or not _finite_number(request.get("timeout_seconds"))
            or float(request["timeout_seconds"]) <= 0
        ):
            return "unsupported_request_contract"
        return None
    if profile_version != REQUEST_PROFILE_VERSION:
        return "unsupported_request_profile_version"
    if set(request) != _PROFILE_KEYS:
        return "incomplete_manifest_contract"
    if (
        request.get("type") != "chat_completions"
        or not _finite_number(request.get("temperature"))
        or float(request["temperature"]) < 0
        or request.get("stop") is not None
        or not isinstance(request.get("stream"), bool)
        or not isinstance(request.get("max_tokens"), int)
        or isinstance(request.get("max_tokens"), bool)
        or int(request["max_tokens"]) <= 0
        or not _finite_number(request.get("timeout_seconds"))
        or float(request["timeout_seconds"]) <= 0
    ):
        return "unsupported_request_contract"
    top_p = request.get("top_p")
    if top_p is not None and (
        not _finite_number(top_p) or not 0 < float(top_p) <= 1
    ):
        return "unsupported_request_contract"
    request_seed = request.get("request_seed")
    if request_seed is not None and (
        not isinstance(request_seed, int)
        or isinstance(request_seed, bool)
        or abs(request_seed) > MAX_SAFE_INTEGER
    ):
        return "unsupported_request_contract"
    extensions = request.get("extensions")
    if not isinstance(extensions, Mapping) or any(
        not isinstance(name, str) for name in extensions
    ):
        return "unsafe_request_extension_manifest"
    if len(extensions) > MAX_REQUEST_FIELDS:
        return "unsafe_request_extension_manifest"
    try:
        validate_request_fields(tuple(extensions.items()))  # type: ignore[arg-type]
    except ValueError:
        return "unsafe_request_extension_manifest"
    profile_sha256 = request.get("profile_sha256")
    if not isinstance(profile_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", profile_sha256
    ):
        return "invalid_request_profile_hash"
    try:
        expected = _profile_sha256(request)
    except (TypeError, ValueError):
        return "invalid_request_profile_hash"
    if profile_sha256 != expected:
        return "invalid_request_profile_hash"
    return None
