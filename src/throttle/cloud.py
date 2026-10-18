"""Throttle Cloud: API keys, the HTTP client, and what the CLI uploads.

The CLI stays free and local. Throttle Cloud is the paid side: check history,
Autotune and the CI cost gate, metered from prepaid credits (start free with
$10 of credits, see https://www.throttle-pro.com/pricing). This module is the
only place that talks to it.

What leaves the machine, and when:

* Nothing, unless you are logged in AND you run a cloud command
  (``throttle whoami``, ``throttle check --push`` or ``push: true`` in
  ~/.throttle/config.yaml, ``throttle autotune``, ``throttle gate``).
* Never prompts, responses, endpoint URLs, hostnames, IPs, file paths or keys.
  :func:`check_payload` builds the upload from numbers, the model name, the
  config fingerprint (secret-looking values masked, hosts and paths removed),
  engine and GPU metadata. docs/cloud.md lists every field.

Every POST carries an ``Idempotency-Key`` derived from what it describes (the
check id, the autotune run and trial index, the gate's two checks), so a retry
or a re-run never charges twice. Out of credits (HTTP 402) only stops cloud
calls: every local feature keeps working.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import ipaddress
import os
import re
import stat
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

import httpx

from . import __version__

DEFAULT_CLOUD_URL = "https://cloud.throttle-pro.com"
CLOUD_URL_ENV = "THROTTLE_CLOUD_URL"
API_KEY_ENV = "THROTTLE_API_KEY"
API_PREFIX = "/api/v1"
PRICING_URL = "https://www.throttle-pro.com/pricing"
DASHBOARD_URL = "https://cloud.throttle-pro.com"
CREDENTIALS_NAME = "credentials"
KEY_PATTERN = re.compile(r"thr_(live|test)_[A-Za-z0-9_\-]{8,}")
TIMEOUT_SECONDS = 20.0
RETRY_DELAY_SECONDS = 1.0

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2

# Tests swap in an httpx.MockTransport; production uses the default transport.
_TRANSPORT: httpx.BaseTransport | None = None


# --------------------------------------------------------------------------
# Credentials
# --------------------------------------------------------------------------


def config_dir() -> Path:
    """~/.config/throttle (or $XDG_CONFIG_HOME/throttle)."""

    base = os.environ.get("XDG_CONFIG_HOME", "").strip()
    root = Path(base) if base else Path.home() / ".config"
    return root / "throttle"


def credentials_path() -> Path:
    return config_dir() / CREDENTIALS_NAME


def valid_key(key: str) -> bool:
    return bool(KEY_PATTERN.fullmatch(key.strip()))


def mask_key(key: str) -> str:
    key = key.strip()
    return f"{key[:9]}...{key[-4:]}" if len(key) > 16 else "thr_..."


def read_credentials_file(path: Path | None = None) -> str | None:
    """The saved API key, or None. Warns when the file is readable by others."""

    path = path or credentials_path()
    try:
        text = path.read_text()
        mode = stat.S_IMODE(path.stat().st_mode)
    except OSError:
        return None
    if os.name == "posix" and mode & 0o077:
        print(
            f"warning: {path} is readable by other users (mode {mode:o}); "
            f"run: chmod 600 {path}",
            file=sys.stderr,
        )
    for line in text.splitlines():
        name, sep, value = line.partition("=")
        if sep and name.strip() == "api_key" and value.strip():
            return value.strip()
    return None


def write_credentials(key: str, path: Path | None = None) -> Path:
    """Save the key with mode 0600 (directory 0700). Never world-readable,
    not even for a moment: the file is created with 0600 before any byte is
    written."""

    path = path or credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.unlink()
    except FileNotFoundError:
        pass
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(
            "# Throttle Cloud credentials (throttle login). Keep this file private.\n"
            f"api_key = {key.strip()}\n"
        )
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    return path


@dataclass(frozen=True)
class Credentials:
    key: str
    source: str  # "THROTTLE_API_KEY" or the credentials file path


def load_credentials() -> Credentials | None:
    """THROTTLE_API_KEY wins over the saved file."""

    env = os.environ.get(API_KEY_ENV, "").strip()
    if env:
        return Credentials(env, API_KEY_ENV)
    path = credentials_path()
    key = read_credentials_file(path)
    return Credentials(key, str(path)) if key else None


# --------------------------------------------------------------------------
# HTTP client
# --------------------------------------------------------------------------


class CloudError(Exception):
    """A cloud call failed. ``kind`` is auth, credits, plan, payload,
    offline, config or server."""

    def __init__(
        self, kind: str, message: str, *, status: int | None = None,
        body: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.status = status
        self.body = dict(body or {})


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def cloud_url() -> str:
    """Base URL ($THROTTLE_CLOUD_URL or https://cloud.throttle-pro.com).

    The API key travels in every request, so plain http is refused except to
    this machine (local development)."""

    url = (os.environ.get(CLOUD_URL_ENV, "").strip() or DEFAULT_CLOUD_URL).rstrip("/")
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise CloudError("config", f"{CLOUD_URL_ENV}={url!r} is not an http(s) URL")
    if parts.scheme == "http" and not _is_loopback(parts.hostname):
        raise CloudError(
            "config",
            f"{CLOUD_URL_ENV} must use https (plain http is only allowed to localhost), "
            "because your API key is sent with every request",
        )
    return url


def _decimal(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def _message_for(status: int, body: Mapping[str, Any], feature: str) -> CloudError:
    if status == 401:
        hint = (
            f"check {API_KEY_ENV}, or unset it and run `throttle login`"
            if os.environ.get(API_KEY_ENV, "").strip()
            else "run `throttle login` with a key from the dashboard"
        )
        return CloudError(
            "auth", f"Throttle Cloud rejected the API key (401): {hint}.",
            status=status, body=body,
        )
    if status == 402:
        credits = body.get("credits_usd", "0.00")
        upgrade = body.get("upgrade_url") or PRICING_URL
        return CloudError(
            "credits",
            f"Out of Throttle Cloud credits (${_decimal(credits)} left), so {feature} "
            f"stopped and nothing more was charged. Every local feature keeps working. "
            f"Add credits or pick a plan: {upgrade}",
            status=status, body=body,
        )
    if status == 403:
        return CloudError(
            "plan",
            f"Your Throttle Cloud plan does not include {feature} (403). "
            f"See plans: {PRICING_URL}",
            status=status, body=body,
        )
    if status == 422:
        detail = body.get("error") or body.get("detail") or "invalid payload"
        return CloudError(
            "payload", f"Throttle Cloud refused the upload (422): {detail}.",
            status=status, body=body,
        )
    detail = body.get("error") or ""
    return CloudError(
        "server",
        f"Throttle Cloud returned HTTP {status}{': ' + str(detail) if detail else ''}. "
        "Nothing is lost locally; try again later.",
        status=status, body=body,
    )


class CloudClient:
    """Small synchronous client for the Throttle Cloud v1 API."""

    def __init__(self, key: str, base_url: str | None = None) -> None:
        self.key = key.strip()
        self.base_url = (base_url or cloud_url()).rstrip("/")

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        idempotency_key: str | None = None,
        feature: str = "this cloud feature",
        params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.key}",
            "User-Agent": f"throttle-cli/{__version__}",
            "Accept": "application/json",
        }
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        url = f"{self.base_url}{API_PREFIX}{path}"
        last_error: Exception | None = None
        # One retry for a dropped connection or a 5xx: safe because every
        # POST is idempotent (same key, never charged twice).
        for attempt in range(2):
            try:
                with httpx.Client(transport=_TRANSPORT, timeout=TIMEOUT_SECONDS) as client:
                    response = client.request(
                        method, url, headers=headers, json=json_body, params=params
                    )
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt == 0 and (idempotency_key or method == "GET"):
                    time.sleep(RETRY_DELAY_SECONDS)
                    continue
                break
            try:
                body = response.json() if response.content else {}
            except ValueError:
                body = {}
            if not isinstance(body, dict):
                body = {"data": body}
            if response.status_code >= 500 and attempt == 0 and (
                idempotency_key or method == "GET"
            ):
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            if 200 <= response.status_code < 300:
                return body
            raise _message_for(response.status_code, body, feature)
        raise CloudError(
            "offline",
            f"Could not reach Throttle Cloud at {self.base_url} ({type(last_error).__name__}: "
            f"{last_error}). Nothing was uploaded or charged; local results are kept.",
        )

    def me(self) -> dict[str, Any]:
        return self._request("GET", "/me", feature="your account")

    def push_check(self, payload: Mapping[str, Any], idempotency_key: str) -> dict[str, Any]:
        return self._request(
            "POST", "/checks", json_body=dict(payload),
            idempotency_key=idempotency_key, feature="check upload",
        )

    def autotune_trial(
        self, run_id: str, trial_index: int, config: Mapping[str, Any]
    ) -> dict[str, Any]:
        return self._request(
            "POST", "/autotune/trials",
            json_body={"run_id": run_id, "trial_index": trial_index, "config": dict(config)},
            idempotency_key=f"autotune:{run_id}:trial:{trial_index}",
            feature="Autotune",
        )

    def autotune_result(self, run_id: str, body: Mapping[str, Any]) -> dict[str, Any]:
        return self._request(
            "POST", f"/autotune/runs/{run_id}/result", json_body=dict(body),
            idempotency_key=f"autotune:{run_id}:result", feature="Autotune",
        )

    def ci_gate(self, body: Mapping[str, Any], idempotency_key: str) -> dict[str, Any]:
        return self._request(
            "POST", "/ci/gate", json_body=dict(body),
            idempotency_key=idempotency_key, feature="the CI cost gate",
        )


def client_or_none() -> CloudClient | None:
    """A client when logged in (env or saved key), else None."""

    creds = load_credentials()
    if creds is None:
        return None
    return CloudClient(creds.key)


def credits_text(body: Mapping[str, Any]) -> str:
    parts = []
    if body.get("charged_usd") is not None:
        parts.append(f"charged ${_decimal(body['charged_usd'])}")
    if body.get("credits_usd") is not None:
        parts.append(f"${_decimal(body['credits_usd'])} credits left")
    return ", ".join(parts)


# --------------------------------------------------------------------------
# What a check upload contains (and does not)
# --------------------------------------------------------------------------

_BLOCK_FIELDS = (
    "input_tokens", "tag_input_tokens", "output_tokens", "max_tokens_hits",
    "wall_clock_seconds", "dollars_per_million", "gpu_dollars",
)
_WORKLOAD_FIELDS = (
    "blocks", "requests_per_block", "concurrency", "max_tokens", "warmup_requests",
    "temperature", "prompt_count", "prompt_cache_mode", "prompts_sha256", "profile",
)
_SUMMARY_FIELDS = ("mean", "ci_low", "ci_high", "n_blocks", "ci_method")


def _numbers_only(value: Any) -> Any:
    """Keep numbers, booleans, None and short enum-like strings; recurse."""

    if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _numbers_only(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_numbers_only(v) for v in value]
    text = str(value)
    return text if re.fullmatch(r"[A-Za-z0-9_\-.]{1,40}", text) else None


def _summary(summary: Any) -> dict[str, Any] | None:
    if not isinstance(summary, Mapping):
        return None
    return {k: summary.get(k) for k in _SUMMARY_FIELDS if k in summary}


def check_payload(
    record: Mapping[str, Any],
    *,
    label: str | None = None,
    project: str | None = None,
) -> dict[str, Any]:
    """The body of POST /checks for one ``throttle check`` record.

    Built from an allow-list, never by copying the record: numbers, the model
    name, the config fingerprint as ``throttle check --share`` shows it
    (secret-looking values masked; URLs, hosts, IPs and paths removed), engine
    and GPU. The endpoint URL is replaced by a one-way hash so checks of one
    server group together without revealing it. Prompts and responses are
    never in a record in the first place; only their sha256 is.
    """

    from . import check as check_module

    fp = record.get("fingerprint") or {}
    metric = record.get("metric", "output")
    result = record.get("result") or {}
    headline = result.get(metric) or {}
    workload = record.get("workload") or {}
    comparison = record.get("comparison") or None
    endpoint = str(record.get("endpoint") or "")
    flat = check_module.flatten_fingerprint(fp) if fp.get("gpu_hourly_rate_usd") is not None else {}
    config = {
        check_module.scrub_key(key): check_module.share_value(key, value)
        for key, value in flat.items()
        if key != "request.model"
    }
    source = workload.get("prompts_source")
    payload: dict[str, Any] = {
        "record_type": record.get("record_type"),
        "record_version": record.get("record_version"),
        "id": record.get("id"),
        "created_at": record.get("created_at"),
        "throttle_version": record.get("throttle_version"),
        "metric": metric,
        "monthly_tokens": record.get("monthly_tokens"),
        "endpoint_sha256": hashlib.sha256(endpoint.encode()).hexdigest()[:16] if endpoint else None,
        "model": check_module.share_value("model", fp.get("model")),
        "engine": check_module._engine_of(fp),
        "gpu": check_module._gpu_of(fp),
        "usd_per_m": headline.get("mean"),
        "ci_low": headline.get("ci_low"),
        "ci_high": headline.get("ci_high"),
        "verdict": (comparison or {}).get("verdict"),
        "fingerprint": {
            "model": check_module.share_value("model", fp.get("model")),
            "label": check_module.share_value("label", fp.get("label")) if fp.get("label") else None,
            "gpu_hourly_rate_usd": fp.get("gpu_hourly_rate_usd"),
            "gpu_hourly_rate_source": "ASSUMED (user-supplied --gpu-hourly-rate)",
            "gpu_count": check_module.gpu_count(record) if fp else None,
            "config": config,
        },
        "workload": {
            **{k: workload.get(k) for k in _WORKLOAD_FIELDS if k in workload},
            "prompts_source": "built-in" if source in (None, "built-in") else "custom",
            **(
                {"workload_shape": _numbers_only(workload["workload_shape"])}
                if workload.get("workload_shape") else {}
            ),
        },
        "blocks": [
            {k: block.get(k) for k in _BLOCK_FIELDS if k in block}
            for block in record.get("blocks") or []
            if isinstance(block, Mapping)
        ],
        "result": {k: _summary(v) for k, v in result.items()},
    }
    if comparison:
        noise = comparison.get("noise") or {}
        shift = comparison.get("output_shift") or {}
        payload["comparison"] = {
            "previous_id": comparison.get("previous_id"),
            "verdict": comparison.get("verdict"),
            "reason": check_module.scrub_text(comparison.get("reason") or ""),
            "previous": _summary(comparison.get("previous")),
            "delta_dollars_per_million": comparison.get("delta_dollars_per_million"),
            "delta_percent": comparison.get("delta_percent"),
            "measured_delta_percent": comparison.get("measured_delta_percent"),
            "rate_changed": comparison.get("rate_changed"),
            "noise": {
                k: noise.get(k)
                for k in ("calibrated", "floor_percent", "sd_percent", "degrees_of_freedom")
                if k in noise
            } or None,
            "output_shift": _numbers_only({
                k: shift.get(k)
                for k in ("before", "after", "length_percent", "max_tokens_hit_points", "changed")
                if k in shift
            }) or None,
        }
    calibration = record.get("calibration")
    if isinstance(calibration, Mapping):
        payload["calibration"] = {
            "ids": list(calibration.get("ids") or []),
            "bound_percent": calibration.get("bound_percent"),
        }
    if label:
        payload["label"] = check_module.share_value("label", label)
    elif fp.get("label"):
        payload["label"] = check_module.share_value("label", fp.get("label"))
    if project:
        payload["project"] = check_module.share_value("project", project)
    return payload


def push_record(record: Mapping[str, Any], *, project: str | None = None) -> None:
    """``throttle check --push``: upload one check, print the URL and credits.

    Never raises and never changes the check's exit code: the measurement is
    already saved locally, the upload is extra.
    """

    creds = load_credentials()
    if creds is None:
        print(
            "Not pushed: not logged in to Throttle Cloud. Run `throttle login` "
            f"(start free with $10 of credits: {PRICING_URL}).",
            file=sys.stderr,
        )
        return
    try:
        client = CloudClient(creds.key)
        body = client.push_check(check_payload(record, project=project),
                                 idempotency_key=f"check:{record.get('id')}")
    except CloudError as exc:
        print(f"Not pushed: {exc}", file=sys.stderr)
        return
    extra = credits_text(body)
    print(f"Pushed to Throttle Cloud: {body.get('url') or body.get('id')}"
          + (f" ({extra})" if extra else ""))


# --------------------------------------------------------------------------
# throttle login / logout / whoami
# --------------------------------------------------------------------------


def add_login_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--key",
        metavar="API_KEY",
        help="API key (thr_live_...). Without it you are prompted, and the key "
        "is not echoed or kept in your shell history. For CI, set "
        f"{API_KEY_ENV} instead of logging in",
    )


def _account_line(me: Mapping[str, Any]) -> str:
    org = (me.get("org") or {}).get("name") if isinstance(me.get("org"), Mapping) else None
    return (
        f"{me.get('email', '?')}"
        + (f" ({org})" if org else "")
        + f", plan {me.get('plan', '?')}, ${_decimal(me.get('credits_usd', '0'))} credits"
    )


def handle_login(args: argparse.Namespace) -> int:
    key = (args.key or "").strip()
    if not key:
        if not sys.stdin.isatty():
            key = sys.stdin.readline().strip()
        else:
            print(f"Paste your Throttle Cloud API key (create one at {DASHBOARD_URL}).")
            try:
                key = getpass.getpass("API key: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return EXIT_USAGE
    if not valid_key(key):
        print(
            "Error: that does not look like a Throttle API key (thr_live_... or "
            f"thr_test_...). Create one at {DASHBOARD_URL}.",
            file=sys.stderr,
        )
        return EXIT_USAGE
    try:
        me = CloudClient(key).me()
    except CloudError as exc:
        print(f"Not logged in: {exc}", file=sys.stderr)
        return EXIT_FAILED
    path = write_credentials(key)
    print(f"Logged in to Throttle Cloud as {_account_line(me)}.")
    print(f"Key {mask_key(key)} saved to {path} (mode 600).")
    if os.environ.get(API_KEY_ENV, "").strip():
        print(f"note: {API_KEY_ENV} is set and takes precedence over the saved key.")
    return EXIT_OK


def handle_logout(args: argparse.Namespace) -> int:
    path = credentials_path()
    try:
        path.unlink()
        print(f"Logged out: removed {path}.")
    except FileNotFoundError:
        print("Not logged in (no saved key).")
    if os.environ.get(API_KEY_ENV, "").strip():
        print(f"note: {API_KEY_ENV} is still set in this shell; unset it to log out fully.")
    return EXIT_OK


def handle_whoami(args: argparse.Namespace) -> int:
    creds = load_credentials()
    if creds is None:
        print(
            "Not logged in. Run `throttle login` (start free with $10 of credits: "
            f"{PRICING_URL}).",
            file=sys.stderr,
        )
        return EXIT_FAILED
    try:
        me = CloudClient(creds.key).me()
    except CloudError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_FAILED
    print(_account_line(me))
    print(f"key {mask_key(creds.key)} from {creds.source}")
    return EXIT_OK
