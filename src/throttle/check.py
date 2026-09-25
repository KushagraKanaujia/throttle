"""``throttle check``: catch cost drift when serving config changes.

Every time a team changes how a model is served (engine flags, batch size,
model, quantization, GPU type, engine version) the dollars per million tokens
can quietly get worse. ``throttle check`` measures $/M tokens against the
live endpoint with a 95% confidence interval, fingerprints the serving
config, stores the result in a local history, and compares it with the
previous check for the same endpoint.

Honesty rules this module follows:

* The GPU hourly rate is user-supplied and always labelled ASSUMED.
  Token counts and wall-clock time come from the endpoint and are MEASURED.
  Monthly figures are PROJECTED (MEASURED $/M x ASSUMED volume).
* Token counts come only from the endpoint's ``usage`` block. If the
  endpoint does not report usage, the check fails instead of guessing.
* Two checks whose 95% confidence intervals overlap are reported as
  NO WINNER, never as cheaper or more expensive.
* Checks run with different workloads are not compared as winners either.
* By default every measured prompt is unique: the first user message starts
  with a short per-request tag (``[run 482915 req 0012] ``) derived from a
  random 6-digit run id stored in the record. The tag's own prompt tokens
  are measured (each base prompt is sent once untagged, unmeasured) and not
  counted as input tokens. A server with prefix caching (vLLM
  V1, SGLang RadixAttention, Ollama KV reuse) therefore cannot serve a
  repeated prompt from cache, which would make $/M look cheaper than real
  traffic. ``--warm-cache`` resends identical prompts on purpose. The mode is
  part of the workload identity, so cold and warm checks are never judged
  against each other.
* A within-run CI cannot see run-to-run drift (machine load, thermal state,
  other tenants), so two checks taken at different times are not a
  controlled comparison. A directional verdict (CHEAPER / MORE EXPENSIVE)
  also needs a run-to-run noise bound measured from recent history, and a
  change larger than it. Without one the verdict is NOT CALIBRATED.
* The verdict judges only what was MEASURED. $/M scales linearly with the
  ASSUMED GPU rate, so when the rate differs the new check is put on the
  baseline's rate before judging; the rate-driven part is reported
  separately and never called significant.

Run-to-run noise bound (the exact rule):

* A "same-config group" is every check with the same endpoint, the same
  flattened config fingerprint (model, label, GPU rate, server-reported
  settings, --config pairs), the same workload identity fields (prompts,
  requests per block, concurrency, max tokens, prompt cache mode) and the same Throttle version.
* Only the two groups on either side of the comparison count: the
  baseline's config and this check's config (one group if they are equal).
  The check being judged is never part of its own calibration sample, each
  stored check counts once, and only checks created within
  CALIBRATION_WINDOW_HOURS of the check being judged count.
* The run-to-run SD is the pooled relative SD of the groups' $/M point
  estimates, with df = sum(n_group - 1). With df < MIN_CALIBRATION_DF (2,
  e.g. fewer than 3 checks of one unchanged config) the comparison is NOT
  CALIBRATED. A baseline older than the window is NOT CALIBRATED too.
* The bound is t(0.975, df) x sqrt(2) x SD: a 95% bound on the difference
  between two single checks of one unchanged config.
* CHEAPER / MORE EXPENSIVE needs ``|measured relative change| > bound`` AND
  non-overlapping 95% CIs. Otherwise NO WINNER, naming which condition
  failed.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import secrets
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote, urlencode, urlsplit, urlunsplit

import httpx

from . import __version__
from .benchmark import (
    canonical_workload_hash,
    load_prompts,
    normalize_chat_completions_url,
)
from .cost_model import calculate_cost
from .models import SafetyLimits
from .statistics import (
    _t_critical_975,
    intervals_overlap,
    relative_delta_percent,
    t_interval_95,
)

RECORD_TYPE = "throttle_check"
RECORD_VERSION = 1
HISTORY_ENV = "THROTTLE_CHECK_HISTORY_DIR"
DEFAULT_HISTORY_DIR = Path.home() / ".throttle" / "checks"
HISTORY_FILENAME = "checks.ndjson"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_COSTLIER = 4
# Only with --fail-if-costlier: the comparison could not be judged, because
# run-to-run noise has not been measured yet (NOT CALIBRATED) or because the
# baseline ran a different workload (NO WINNER, not comparable). CI can treat
# it as a warning.
EXIT_NOT_CALIBRATED = 5

# Run-to-run calibration. Degrees of freedom of the pooled run-to-run SD:
# 2 means e.g. 3 checks of one unchanged config (not counting the check being
# judged). Checks older than the window (relative to the one being judged)
# neither calibrate nor serve as a judged baseline: drift over a gap longer
# than the calibration ever spanned is unmeasured.
MIN_CALIBRATION_DF = 2
CALIBRATION_WINDOW_HOURS = 24.0

VERDICT_CHEAPER = "CHEAPER"
VERDICT_COSTLIER = "MORE EXPENSIVE"
VERDICT_NO_WINNER = "NO WINNER"
VERDICT_NOT_CALIBRATED = "NOT CALIBRATED"

NOT_CALIBRATED_MESSAGE = (
    "Verdict: NOT CALIBRATED \u2014 run-to-run noise unknown. Two checks at "
    "different times can differ from load alone. Run 'throttle check' at least "
    "3 times without changing anything (within 24 h) to measure your noise, or "
    "use 'throttle golden' for a counterbalanced decision."
)

METRIC_LABELS = {
    "output": "output tokens",
    "input": "input tokens",
    "total": "tokens (input + output)",
}

# Workload fields that must match for two checks to be compared as a winner.
WORKLOAD_IDENTITY_FIELDS = (
    "prompts_sha256",
    "requests_per_block",
    "concurrency",
    "max_tokens",
    "prompt_cache_mode",
)

# Prompt cache modes. COLD (default): every measured request carries a unique
# tag at the start of its first user message, so a prefix cache cannot serve
# a repeat. WARM (--warm-cache): identical prompts are resent on purpose.
CACHE_MODE_COLD = "cold"
CACHE_MODE_WARM = "warm"
# Checks recorded before 0.4.1 have no prompt_cache_mode field. They resent
# identical prompts every block and every run, i.e. they were warm-cache.
LEGACY_CACHE_MODE = CACHE_MODE_WARM
NONCE_FORMAT = "[run {run_id} req {index:04d}] "
WARMUP_NONCE_FORMAT = "[run {run_id} warmup {index:04d}] "
# The run id is RUN_ID_DIGITS decimal digits, not hex: common tokenizers
# split digits into fixed-size groups (Llama 3 / tiktoken: up to 3 per token;
# SentencePiece models: 1 per token), so every tag costs the same number of
# prompt tokens in every run. Hex ids (7f3a2c vs e8e8e8) tokenized to a
# different count per run and added run-to-run noise to input $/M.
RUN_ID_DIGITS = 6
# Cold mode only: each distinct base prompt is sent once untagged, unmeasured,
# with this max_tokens, so the tag's prompt tokens can be subtracted from the
# measured input tokens (input and total $/M are per real prompt token).
TAG_PROBE_MAX_TOKENS = 1

METRICS_TIMEOUT_SECONDS = 3.0
# A check is short by design (default workload is ~20 requests), so its time
# ceiling is tighter than benchmark's 900 s. It bounds GPU spend at the
# user's rate: e.g. $2.49/hr x 300 s = $0.21.
CHECK_MAX_ELAPSED_SECONDS = 300.0
MODELS_TIMEOUT_SECONDS = 10.0
MAX_METRICS_BYTES = 4 * 1024 * 1024
MAX_INFO_LABELS = 64
_INFO_LINE = re.compile(r"^(vllm:[A-Za-z0-9_:]*_info)\{(.*)\}\s+\S+")
_LABEL = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"')
_CONFIG_KEY = re.compile(r"^[A-Za-z0-9_.\-]+$")
# A config key is refused when one of its words (split on '.', '_', '-' and
# camelCase) is one of these. Matching whole words, not substrings, keeps real
# serving settings such as max_num_batched_tokens or tokenizer_mode usable
# while still refusing api_key, HF_TOKEN, apiKey or db_password.
_SECRET_KEY_WORDS = frozenset({
    "key", "apikey", "token", "secret", "password", "passwd", "pwd",
    "credential", "credentials", "auth", "bearer",
})


def _looks_like_secret(key: str) -> bool:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", key)
    words = re.split(r"[\s._-]+", spaced.lower())
    return any(w in _SECRET_KEY_WORDS for w in words)

# Test seam: tests install an httpx.MockTransport here. Production leaves it
# as None so real network transport is used.
_TRANSPORT: httpx.AsyncBaseTransport | None = None


class CheckError(Exception):
    """A check that cannot produce an honest number."""


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------


def _config_pair(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"expected KEY=VALUE (e.g. max_num_seqs=256), got {value!r}"
        )
    key, _, raw = value.partition("=")
    key = key.strip()
    if not key or not _CONFIG_KEY.match(key):
        raise argparse.ArgumentTypeError(
            f"config key {key!r} must use letters, digits, '.', '_' or '-'"
        )
    if _looks_like_secret(key):
        raise argparse.ArgumentTypeError(
            f"config key {key!r} looks like a secret; checks are stored in "
            "plain text, so it is refused"
        )
    return key, raw.strip()


def _token_count(value: str) -> int:
    text = value.strip().upper().replace(",", "").replace("_", "")
    multiplier = 1
    for suffix, factor in (("K", 10**3), ("M", 10**6), ("B", 10**9), ("T", 10**12)):
        if text.endswith(suffix):
            multiplier = factor
            text = text[:-1]
            break
    try:
        number = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a token count like 500M, 2B or 750000, got {value!r}"
        ) from None
    total = number * multiplier
    if not math.isfinite(total) or total <= 0:
        raise argparse.ArgumentTypeError("monthly tokens must be positive")
    return int(round(total))


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {value!r}") from None
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _non_negative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {value!r}") from None
    if number < 0:
        raise argparse.ArgumentTypeError("must be zero or more")
    return number


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from None
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return number


def _non_negative_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from None
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be zero or a positive number")
    return number


CHECK_DESCRIPTION = (
    "Measure what this endpoint costs in dollars per million tokens, remember\n"
    "the serving config it was measured on, and compare with the last check of\n"
    "the same endpoint.\n"
    "\n"
    "Run it after every serving change (engine flags, batch size, model,\n"
    "quantization, GPU type, engine version) so a change that quietly makes\n"
    "inference more expensive is caught before the bill shows it.\n"
    "\n"
    "The GPU $/hr is yours (ASSUMED); tokens and time are MEASURED. Overlapping\n"
    "95% confidence intervals are reported as NO WINNER, never as a winner.\n"
    "\n"
    "A 95% CI is computed within one run, so it cannot see run-to-run drift\n"
    "(machine load, thermal state, other tenants). CHEAPER or MORE EXPENSIVE\n"
    "also needs a run-to-run noise bound (t x sqrt(2) x SD) estimated from\n"
    "repeat checks of an unchanged config in the last 24 h, and a measured\n"
    "change larger than it. Until there are 3 such earlier checks (2 df), or\n"
    "when the baseline is more than 24 h old, the verdict is NOT CALIBRATED:\n"
    "run the same check a few more times without changing anything. A change\n"
    "in the ASSUMED GPU rate alone is never a verdict."
)

CHECK_EPILOG = (
    "Examples:\n"
    "  throttle check --url http://localhost:8000 --model my-model \\\n"
    "      --gpu-hourly-rate 2.49 --config max_num_seqs=256 --label before\n"
    "  throttle check --url http://localhost:8000 --model my-model \\\n"
    "      --gpu-hourly-rate 2.49 --config max_num_seqs=512 --label after \\\n"
    "      --monthly-tokens 500M --fail-if-costlier 5\n"
    "  throttle check --history\n"
    "  throttle check --history --share     # share the latest check (no traffic)\n"
    "  throttle check --share-id CHECK_ID   # share a chosen check\n"
    "\n"
    "Prompt cache: by default every measured request's first user message\n"
    "starts with a unique tag ('[run 482915 req 0012] ', run id recorded) so a\n"
    "prefix cache cannot serve repeats; --warm-cache resends identical prompts.\n"
    "Cold and warm checks are never compared (NO WINNER).\n"
    "\n"
    "Exit codes:\n"
    "  0  check done (cheaper, no winner, first check, or not calibrated\n"
    "     without --fail-if-costlier)\n"
    "  1  measurement failed; nothing recorded\n"
    "  2  usage error\n"
    "  4  --fail-if-costlier tripped: calibrated MORE EXPENSIVE by >= PCT\n"
    "     (measured change, at the baseline's GPU rate)\n"
    "  5  --fail-if-costlier given but the verdict is NOT CALIBRATED (run-to-run\n"
    "     noise not measured yet); treat it as a warning or a failure"
)


def add_check_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach ``throttle check`` options to a subparser."""

    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    endpoint = parser.add_argument_group("endpoint")
    endpoint.add_argument(
        "--url",
        "--endpoint-url",
        dest="url",
        help="OpenAI-compatible base URL, e.g. http://localhost:8000 or .../v1 "
        "(--endpoint-url is accepted too)",
    )
    endpoint.add_argument(
        "--model",
        help="model name to send requests to (default: the only model the "
        "server lists, if it lists exactly one)",
    )
    endpoint.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        metavar="NAME",
        help="environment variable holding the bearer key (default: OPENAI_API_KEY)",
    )
    endpoint.add_argument(
        "--allow-insecure-http",
        action="store_true",
        help="allow plain HTTP to a non-loopback host (your key travels unencrypted)",
    )
    endpoint.add_argument(
        "--metrics-url",
        help="vLLM Prometheus /metrics URL to read config labels from "
        "(default: <host>/metrics; skipped quietly if not reachable)",
    )
    endpoint.add_argument(
        "--no-metrics",
        action="store_true",
        help="do not read /metrics for the config fingerprint",
    )

    cost = parser.add_argument_group("cost")
    cost.add_argument(
        "--gpu-hourly-rate",
        "--gpu-rate-per-hour",
        "--total-hourly-price",
        dest="gpu_hourly_rate",
        type=_positive_float,
        metavar="USD",
        help="what the GPU(s) behind this endpoint cost you per hour, in dollars "
        "(recorded as ASSUMED; include every GPU the model uses)",
    )
    cost.add_argument(
        "--metric",
        choices=tuple(METRIC_LABELS),
        default="output",
        help="which tokens the headline $/M figure is per (default: output)",
    )
    cost.add_argument(
        "--monthly-tokens",
        type=_token_count,
        metavar="N",
        help="your monthly volume of the headline tokens (e.g. 500M, 2B) to "
        "turn $/M into a monthly dollar figure",
    )

    fingerprint = parser.add_argument_group("config fingerprint")
    fingerprint.add_argument(
        "--label",
        help="short name for this config, e.g. 'fp8-batch256'",
    )
    fingerprint.add_argument(
        "--config",
        dest="config_pairs",
        action="append",
        type=_config_pair,
        default=None,
        metavar="KEY=VALUE",
        help="a serving setting the endpoint cannot report itself "
        "(repeatable), e.g. --config gpu=H100 --config quantization=fp8",
    )

    workload = parser.add_argument_group("workload (keep identical between checks)")
    workload.add_argument(
        "--blocks",
        type=_positive_int,
        default=5,
        help="independent measurement blocks; the 95%% CI is across blocks "
        "(default: 5, minimum 3)",
    )
    workload.add_argument(
        "--requests-per-block",
        type=_positive_int,
        default=4,
        help="requests sent in each block (default: 4)",
    )
    workload.add_argument(
        "--concurrency",
        type=_positive_int,
        default=2,
        help="requests in flight at once; cost depends heavily on this, so "
        "match your production load (default: 2)",
    )
    workload.add_argument(
        "--max-tokens",
        type=_positive_int,
        default=64,
        help="max output tokens per request (default: 64)",
    )
    workload.add_argument(
        "--warmup",
        type=_non_negative_int,
        default=1,
        help="unmeasured warm-up requests before measuring (default: 1)",
    )
    workload.add_argument(
        "--prompts",
        type=Path,
        help="JSONL prompt file (default: Throttle's built-in prompts)",
    )
    workload.add_argument(
        "--warm-cache",
        action="store_true",
        help="resend identical prompts every block, to measure cache-friendly "
        "traffic. By default each measured request's first user message starts "
        "with a unique tag such as '[run 482915 req 0012] ' so a server with "
        "prefix caching (vLLM, SGLang, Ollama) cannot serve repeats from cache. "
        "Cold and warm checks are never compared with each other",
    )
    workload.add_argument(
        "--request-timeout",
        type=_positive_float,
        default=120.0,
        metavar="SECONDS",
        help="per-request timeout (default: 120)",
    )

    _defaults = SafetyLimits()
    safety = parser.add_argument_group(
        "safety limits (checked before any traffic; same request, token, concurrency and spend defaults as smoke/benchmark)"
    )
    safety.add_argument(
        "--max-requests",
        type=_positive_int,
        default=_defaults.max_requests,
        help=f"refuse a check that would send more requests than this, warm-ups "
        f"included (default: {_defaults.max_requests})",
    )
    safety.add_argument(
        "--max-tokens-per-request",
        type=_positive_int,
        default=_defaults.max_tokens_per_request,
        help=f"refuse --max-tokens above this (default: {_defaults.max_tokens_per_request})",
    )
    safety.add_argument(
        "--max-total-requested-tokens",
        type=_positive_int,
        default=_defaults.max_total_requested_tokens,
        help="refuse a check whose requests x --max-tokens exceeds this; on a "
        f"per-token API this is your bill ceiling (default: {_defaults.max_total_requested_tokens:,})",
    )
    safety.add_argument(
        "--max-concurrency",
        type=_positive_int,
        default=_defaults.max_concurrency,
        help=f"refuse --concurrency above this (default: {_defaults.max_concurrency})",
    )
    safety.add_argument(
        "--max-elapsed-seconds",
        type=_positive_float,
        default=CHECK_MAX_ELAPSED_SECONDS,
        help="stop the check (nothing recorded) once it has run this long "
        f"(default: {CHECK_MAX_ELAPSED_SECONDS:g})",
    )
    safety.add_argument(
        "--max-estimated-spend",
        type=_positive_float,
        default=_defaults.max_estimated_spend,
        metavar="USD",
        help="refuse a check whose worst-case GPU time (--max-elapsed-seconds x "
        f"--gpu-hourly-rate) could cost more than this (default: ${_defaults.max_estimated_spend:.2f})",
    )

    history = parser.add_argument_group("history and CI")
    history.add_argument(
        "--history",
        action="store_true",
        help="list past checks (all endpoints, or only --url) and exit; sends no traffic",
    )
    history.add_argument(
        "--limit",
        type=_positive_int,
        default=20,
        help="rows shown by --history (default: 20)",
    )
    history.add_argument(
        "--history-dir",
        type=Path,
        help=f"where checks are stored (default: ${HISTORY_ENV} or ~/.throttle/checks)",
    )
    history.add_argument(
        "--against",
        metavar="CHECK_ID",
        help="compare with this check id instead of the latest one for the endpoint",
    )
    history.add_argument(
        "--no-save",
        action="store_true",
        help="compare but do not add this check to history",
    )
    history.add_argument(
        "--fail-if-costlier",
        type=_non_negative_float,
        metavar="PCT",
        help="exit with code 4 if this check is a calibrated MORE EXPENSIVE "
        "than the previous one by at least PCT percent; exit 5 if the "
        "change could not be judged (NOT CALIBRATED, or the baseline ran a "
        "different workload, e.g. another prompt cache mode)",
    )
    history.add_argument(
        "--json",
        dest="json_output",
        type=Path,
        metavar="PATH",
        help="also write this check and its comparison as JSON",
    )

    share = parser.add_argument_group("share your results (nothing is uploaded)")
    share.add_argument(
        "--share",
        action="store_true",
        help="print a sanitized markdown summary (no URLs, hosts, IPs or keys; "
        "secret-looking config values masked) and a pre-filled GitHub issue "
        "link you can review and submit yourself. With a measuring check it "
        "shares that check; with --history it shares the latest recorded check "
        "(of --url, if given) without sending traffic",
    )
    share.add_argument(
        "--share-id",
        metavar="CHECK_ID",
        help="share this recorded check (see --history) instead; sends no traffic",
    )


# --------------------------------------------------------------------------
# History store
# --------------------------------------------------------------------------


def history_dir(args: argparse.Namespace | None = None) -> Path:
    explicit = getattr(args, "history_dir", None) if args is not None else None
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get(HISTORY_ENV, "").strip()
    if env:
        return Path(env).expanduser()
    return DEFAULT_HISTORY_DIR


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _valid_summary(summary: Any) -> bool:
    if not isinstance(summary, dict):
        return False
    for key in ("mean", "ci_low", "ci_high"):
        value = summary.get(key)
        if value is not None and not _is_number(value):
            return False
    return True


def _valid_record(record: Any) -> bool:
    """True when a history line has every field later code reads.

    Anything else (hand-edited, truncated, or from another tool) is skipped
    and counted, never allowed to crash a check after its traffic was sent.
    """

    if not isinstance(record, dict) or record.get("record_type") != RECORD_TYPE:
        return False
    if not isinstance(record.get("endpoint"), str):
        return False
    result = record.get("result")
    if not isinstance(result, dict) or not result:
        return False
    if not all(_valid_summary(value) for value in result.values()):
        return False
    metric = record.get("metric", "output")
    if not isinstance(metric, str) or metric not in result:
        return False
    fingerprint = record.get("fingerprint")
    if not isinstance(fingerprint, dict):
        return False
    rate = fingerprint.get("gpu_hourly_rate_usd")
    if not _is_number(rate) or rate <= 0:
        return False
    for key in ("user_config", "server_models", "server_metrics"):
        if fingerprint.get(key) is not None and not isinstance(fingerprint.get(key), dict):
            return False
    for key in ("model", "label"):
        if fingerprint.get(key) is not None and not isinstance(fingerprint.get(key), str):
            return False
    metrics = fingerprint.get("server_metrics") or {}
    info = metrics.get("info")
    if info is not None and not (
        isinstance(info, dict)
        and all(isinstance(labels, dict) for labels in info.values())
    ):
        return False
    models = fingerprint.get("server_models") or {}
    if models.get("model_detail") is not None and not isinstance(models.get("model_detail"), dict):
        return False
    if not isinstance(record.get("workload", {}), dict):
        return False
    if not isinstance(record.get("id"), str) or not record["id"]:
        return False
    if record.get("created_at") is not None and not isinstance(record.get("created_at"), str):
        return False
    return True


def load_history(directory: Path) -> tuple[list[dict[str, Any]], int]:
    """Return (records oldest first, number of unreadable lines skipped)."""

    path = directory / HISTORY_FILENAME
    if not path.exists():
        return [], 0
    records: list[dict[str, Any]] = []
    skipped = 0
    # errors="replace": a stray non-UTF-8 byte spoils one line, not the file.
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, RecursionError):
                skipped += 1
                continue
            if _valid_record(record):
                records.append(record)
            else:
                skipped += 1
    return records, skipped


def append_history(directory: Path, record: Mapping[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / HISTORY_FILENAME
    line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)
    return path


# --------------------------------------------------------------------------
# Config fingerprint
# --------------------------------------------------------------------------


def _models_url(chat_url: str) -> str:
    return chat_url[: -len("/chat/completions")] + "/models"


def _default_metrics_url(chat_url: str) -> str:
    parts = urlsplit(chat_url)
    return urlunsplit((parts.scheme, parts.netloc, "/metrics", "", ""))


def parse_info_labels(text: str) -> dict[str, dict[str, str]]:
    """Pull config labels out of vLLM ``*_info`` gauges, e.g.
    ``vllm:cache_config_info{block_size="16",gpu_memory_utilization="0.9"} 1``.
    Everything else in /metrics is ignored."""

    found: dict[str, dict[str, str]] = {}
    for raw in text.splitlines():
        match = _INFO_LINE.match(raw.strip())
        if not match:
            continue
        name = match.group(1)
        labels = dict(_LABEL.findall(match.group(2))[:MAX_INFO_LABELS])
        found.setdefault(name, {}).update(labels)
    return found


async def _fetch_models(
    client: httpx.AsyncClient, url: str, headers: Mapping[str, str], model: str
) -> dict[str, Any]:
    try:
        response = await client.get(url, headers=headers, timeout=MODELS_TIMEOUT_SECONDS)
    except httpx.HTTPError as exc:
        return {"status": f"not reachable ({type(exc).__name__})", "url": url}
    if response.status_code != 200:
        return {"status": f"not exposed (HTTP {response.status_code})", "url": url}
    try:
        data = response.json().get("data", [])
    except (ValueError, AttributeError):
        return {"status": "unreadable response", "url": url}
    if not isinstance(data, list):
        return {"status": "unreadable response", "url": url}
    ids = sorted(str(item.get("id")) for item in data if isinstance(item, dict) and item.get("id"))
    detail: dict[str, str] = {}
    for item in data:
        if isinstance(item, dict) and item.get("id") == model:
            # 'created' is skipped: several servers stamp it with the
            # response time, which would make every check look "changed".
            for key in ("owned_by", "root", "parent", "max_model_len"):
                if item.get(key) is not None:
                    detail[key] = str(item[key])
    return {
        "status": "read",
        "url": url,
        "model_ids": ids,
        "model_detail": detail,
        "requested_model_listed": model in ids,
    }


async def _fetch_metrics(client: httpx.AsyncClient, url: str) -> dict[str, Any]:
    try:
        async with client.stream(
            "GET", url, timeout=METRICS_TIMEOUT_SECONDS, follow_redirects=False
        ) as response:
            if response.status_code != 200:
                return {"status": f"not exposed (HTTP {response.status_code})", "url": url}
            body = bytearray()
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if len(body) > MAX_METRICS_BYTES:
                    return {"status": "response too large; ignored", "url": url}
    except (httpx.HTTPError, httpx.InvalidURL, ValueError) as exc:
        return {"status": f"not reachable ({type(exc).__name__})", "url": url}
    info = parse_info_labels(body.decode("utf-8", errors="replace"))
    if not info:
        return {"status": "reachable, but no vLLM *_info config labels", "url": url}
    return {"status": "read", "url": url, "info": info}


def flatten_fingerprint(fingerprint: Mapping[str, Any]) -> dict[str, str]:
    """One flat KEY -> VALUE view used for diffs and display."""

    flat: dict[str, str] = {}
    flat["request.model"] = str(fingerprint.get("model"))
    flat["gpu_hourly_rate_usd (ASSUMED)"] = f"{float(fingerprint['gpu_hourly_rate_usd']):.4f}"
    if fingerprint.get("label"):
        flat["label"] = str(fingerprint["label"])
    models = fingerprint.get("server_models") or {}
    if models.get("status") == "read":
        # Only whether the model under test is listed: other models being
        # pulled onto the same server is not a change to this config.
        listed = models.get("requested_model_listed")
        if listed is None:
            listed = fingerprint.get("model") in (models.get("model_ids") or [])
        flat["server.requested_model_listed"] = "yes" if listed else "no"
        for key, value in sorted((models.get("model_detail") or {}).items()):
            flat[f"server.model.{key}"] = value
    metrics = fingerprint.get("server_metrics") or {}
    for info_name, labels in sorted((metrics.get("info") or {}).items()):
        short = info_name.split(":", 1)[-1].removesuffix("_info")
        for key, value in sorted(labels.items()):
            flat[f"vllm.{short}.{key}"] = value
    for key, value in sorted((fingerprint.get("user_config") or {}).items()):
        flat[f"config.{key}"] = value
    return flat


def _comparable_flat(
    previous_fp: Mapping[str, Any], current_fp: Mapping[str, Any]
) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """Flattened fingerprints restricted to what both checks could read.

    A /metrics or /v1/models read that failed on one side would otherwise
    show every server setting as "(not set)", which looks like a config
    change but only means Throttle could not see it this time.
    """

    old_flat = flatten_fingerprint(previous_fp)
    new_flat = flatten_fingerprint(current_fp)
    notes: list[str] = []
    for source, prefixes, what in (
        ("server_metrics", ("vllm.",), "/metrics"),
        ("server_models", ("server.",), "/v1/models"),
    ):
        old_status = (previous_fp.get(source) or {}).get("status")
        new_status = (current_fp.get(source) or {}).get("status")
        if old_status == "read" and new_status == "read":
            continue
        if old_status != "read" and new_status != "read":
            continue
        side = "this time" if new_status != "read" else "last time"
        status = new_status if new_status != "read" else old_status
        notes.append(
            f"{what} was not read {side} ({status or 'unknown'}); "
            "server settings from it were not compared"
        )
        for flat in (old_flat, new_flat):
            for key in [k for k in flat if k.startswith(prefixes)]:
                del flat[key]
    return old_flat, new_flat, notes


def diff_fingerprints(
    old: Mapping[str, str], new: Mapping[str, str]
) -> list[tuple[str, str | None, str | None]]:
    changes = []
    for key in sorted(set(old) | set(new)):
        if old.get(key) != new.get(key):
            changes.append((key, old.get(key), new.get(key)))
    return changes


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


def _block_costs(
    input_tokens: int, output_tokens: int, wall: float, rate: float
) -> dict[str, float]:
    cost = calculate_cost(input_tokens, output_tokens, wall, rate)
    total_tokens = input_tokens + output_tokens
    return {
        "input": cost.dollars_per_million_input_tokens,
        "output": cost.dollars_per_million_output_tokens,
        "total": cost.total_dollars / total_tokens * 1_000_000,
        "gpu_dollars": cost.total_dollars,
    }


async def _send(
    client: httpx.AsyncClient,
    chat_url: str,
    headers: Mapping[str, str],
    model: str,
    messages: Sequence[Mapping[str, Any]],
    max_tokens: int,
) -> tuple[int, int]:
    try:
        response = await client.post(
            chat_url,
            headers=headers,
            json={
                "model": model,
                "messages": list(messages),
                "max_tokens": max_tokens,
                "temperature": 0,
                "stream": False,
            },
        )
    except httpx.TimeoutException:
        raise CheckError("a request timed out (raise --request-timeout or lower --max-tokens)") from None
    except httpx.HTTPError as exc:
        raise CheckError(f"could not reach the endpoint: {type(exc).__name__}: {exc}") from None
    if response.status_code != 200:
        snippet = response.text[:200].replace("\n", " ")
        raise CheckError(f"endpoint returned HTTP {response.status_code}: {snippet}")
    try:
        usage = response.json().get("usage")
    except (ValueError, AttributeError):
        raise CheckError("endpoint returned a response that is not JSON") from None
    if not isinstance(usage, dict):
        raise CheckError(
            "endpoint did not report token usage; Throttle does not guess token counts"
        )
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    for name, value in (("prompt_tokens", prompt_tokens), ("completion_tokens", completion_tokens)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise CheckError(
                f"endpoint usage.{name} is missing or invalid ({value!r}); "
                "Throttle does not guess token counts"
            )
    return prompt_tokens, completion_tokens


async def _run_block(
    client: httpx.AsyncClient,
    chat_url: str,
    headers: Mapping[str, str],
    model: str,
    prompts: Sequence[Sequence[Mapping[str, Any]]],
    requests: int,
    concurrency: int,
    max_tokens: int,
) -> tuple[int, int, float]:
    semaphore = asyncio.Semaphore(concurrency)

    async def one(index: int) -> tuple[int, int]:
        async with semaphore:
            return await _send(
                client, chat_url, headers, model, prompts[index % len(prompts)], max_tokens
            )

    started = time.perf_counter()
    results = await asyncio.gather(*(one(index) for index in range(requests)))
    wall = time.perf_counter() - started
    return sum(r[0] for r in results), sum(r[1] for r in results), wall


def new_run_id() -> str:
    """Random per-check id that seeds the per-request prompt tags."""

    return f"{secrets.randbelow(10 ** RUN_ID_DIGITS):0{RUN_ID_DIGITS}d}"


def tag_probe_count(
    prompts: Sequence[Any], requests_per_block: int, cache_mode: str
) -> int:
    """Unmeasured untagged requests a cold check sends to net out the tag."""

    if cache_mode != CACHE_MODE_COLD:
        return 0
    return min(requests_per_block, len(prompts))


def tag_prompt(messages: Sequence[Mapping[str, Any]], tag: str) -> tuple[dict[str, Any], ...]:
    """Put ``tag`` at the start of the first user message.

    Any message before it (e.g. a system prompt) stays shared, as it would in
    real traffic; everything from the first user message on is unique. A
    prompt without a user message gets the tag on its first message.
    """

    tagged = [dict(message) for message in messages]
    target = next(
        (i for i, m in enumerate(tagged) if m.get("role") == "user"), 0
    )
    tagged[target]["content"] = tag + str(tagged[target].get("content", ""))
    return tuple(tagged)


def measured_prompts(
    prompts: Sequence[Sequence[Mapping[str, Any]]],
    blocks: int,
    requests_per_block: int,
    mode: str,
    run_id: str | None,
) -> list[list[Sequence[Mapping[str, Any]]]]:
    """The exact messages each measured request sends, per block.

    Request j of every block uses prompt j (mod the prompt count), as
    0.4.0 did, so a warm check sends the same traffic as before and a cold
    check differs from it only by the tag. In cold mode request number n
    (counted across blocks, from 0) gets the tag ``NONCE_FORMAT``.
    """

    plan: list[list[Sequence[Mapping[str, Any]]]] = []
    for block in range(blocks):
        row: list[Sequence[Mapping[str, Any]]] = []
        for j in range(requests_per_block):
            base = prompts[j % len(prompts)]
            if mode == CACHE_MODE_COLD:
                index = block * requests_per_block + j
                row.append(tag_prompt(base, NONCE_FORMAT.format(run_id=run_id, index=index)))
            else:
                row.append(base)
        plan.append(row)
    return plan


def workload_value(record: Mapping[str, Any], field: str) -> Any:
    """A workload identity field, reading pre-0.4.1 records as warm-cache."""

    workload = record.get("workload") or {}
    value = workload.get(field)
    if field == "prompt_cache_mode" and value is None:
        return LEGACY_CACHE_MODE
    return value


def summarize(values: Sequence[float]) -> dict[str, Any]:
    interval = t_interval_95(values)
    return {
        "mean": statistics.fmean(values) if values else None,
        "ci_low": interval["low"],
        "ci_high": interval["high"],
        "n_blocks": interval["n"],
        "ci_method": "student_t_blocks_95",
    }


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------


def _config_identity(record: Mapping[str, Any]) -> str:
    """Endpoint + config fingerprint + workload + Throttle version.

    Equal means "nothing changed". The Throttle version is part of it so a
    measurement-code upgrade never silently joins an older calibration group.
    """

    return json.dumps(
        [
            record.get("endpoint"),
            flatten_fingerprint(record["fingerprint"]),
            {field: workload_value(record, field) for field in WORKLOAD_IDENTITY_FIELDS},
            record.get("throttle_version"),
        ],
        sort_keys=True,
    )


def _usable_mean(record: Mapping[str, Any], metric: str) -> float | None:
    summary = (record.get("result") or {}).get(metric)
    if not isinstance(summary, dict):
        return None
    mean = summary.get("mean")
    return float(mean) if _is_number(mean) and mean > 0 else None


def _created(record: Mapping[str, Any]) -> datetime | None:
    value = record.get("created_at")
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _age_text(seconds: float) -> str:
    hours = seconds / 3600
    if hours >= 48:
        return f"{hours / 24:.0f} days"
    if hours >= 1:
        return f"{hours:.1f} h"
    return f"{max(seconds, 0) / 60:.0f} min"


def rate_of(record: Mapping[str, Any]) -> float:
    return float(record["fingerprint"]["gpu_hourly_rate_usd"])


def run_to_run_noise(
    history: Sequence[Mapping[str, Any]],
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    metric: str,
) -> dict[str, Any]:
    """Measure the run-to-run noise bound from repeat checks of one config.

    See the module docstring for the rule. ``history`` may or may not
    already contain ``previous`` and ``current``. The current check is never
    part of its own calibration sample, and every other record counts once
    (by id, or by object identity for a record without one).
    """

    current_id = current.get("id")
    now = _created(current) or datetime.now(timezone.utc)
    window = timedelta(hours=CALIBRATION_WINDOW_HOURS)

    pool: list[Mapping[str, Any]] = []
    seen: set[Any] = set()
    stale = 0
    for record in [*history, previous]:
        if record is current or (current_id is not None and record.get("id") == current_id):
            continue
        record_id = record.get("id")
        key = ("id", record_id) if isinstance(record_id, str) else ("obj", id(record))
        if key in seen:
            continue
        seen.add(key)
        created = _created(record)
        if created is None or abs(now - created) > window:
            stale += 1
            continue
        pool.append(record)

    groups: list[dict[str, Any]] = []
    baseline_key = _config_identity(previous)
    current_key = _config_identity(current)
    if baseline_key == current_key:
        sides = [(baseline_key, "baseline and this check (same config)", previous)]
    else:
        sides = [
            (baseline_key, "baseline config", previous),
            (current_key, "this check's config", current),
        ]
    hints: list[str] = []
    squares = 0.0
    df = 0
    for key, side_name, representative in sides:
        members = [
            r for r in pool if _config_identity(r) == key and _usable_mean(r, metric) is not None
        ]
        means = [float(_usable_mean(r, metric)) for r in members]  # type: ignore[arg-type]
        group: dict[str, Any] = {
            "side": side_name,
            "label": representative["fingerprint"].get("label"),
            "n_checks": len(members),
            "check_ids": [r.get("id") for r in members],
            "spread_percent": None,
            "sd_percent": None,
        }
        if len(members) >= 2:
            low, high = min(means), max(means)
            group["spread_percent"] = (high - low) / low * 100.0
            centre = statistics.fmean(means)
            group_squares = sum(((m - centre) / centre) ** 2 for m in means)
            group["sd_percent"] = math.sqrt(group_squares / (len(means) - 1)) * 100.0
            squares += group_squares
            df += len(means) - 1
            for index, first in enumerate(members):
                for second in members[index + 1:]:
                    a = first["result"][metric]
                    b = second["result"][metric]
                    if not intervals_overlap(
                        {"low": a.get("ci_low"), "high": a.get("ci_high")},
                        {"low": b.get("ci_low"), "high": b.get("ci_high")},
                    ):
                        hints.append(
                            f"checks {first.get('id')} and {second.get('id')} ran the "
                            "same config and workload, yet their 95% CIs do not "
                            "overlap: your machine's run-to-run noise is larger than "
                            "within-run noise, so a within-run CI alone understates "
                            "the uncertainty"
                        )
        groups.append(group)

    calibrated = df >= MIN_CALIBRATION_DF
    sd_percent = math.sqrt(squares / df) * 100.0 if df else None
    t_crit = _t_critical_975(df) if df else None
    floor = t_crit * math.sqrt(2.0) * sd_percent if calibrated else None  # type: ignore[operator]
    return {
        "calibrated": calibrated,
        "floor_percent": floor,
        "sd_percent": sd_percent,
        "degrees_of_freedom": df,
        "t_critical": t_crit,
        "n_checks": sum(g["n_checks"] for g in groups if g["spread_percent"] is not None),
        "stale_or_undated_checks_ignored": stale,
        "window_hours": CALIBRATION_WINDOW_HOURS,
        "rule": "95% bound on the difference between two single checks of one config: "
        "t(0.975, df) x sqrt(2) x pooled run-to-run SD of $/M means, pooled over the "
        "baseline's and this check's same-config groups (endpoint + fingerprint + "
        f"workload + Throttle version), from checks within {CALIBRATION_WINDOW_HOURS:g} h "
        "of this one, excluding this check; needs df >= "
        f"{MIN_CALIBRATION_DF} (e.g. 3 checks of one unchanged config)",
        "groups": groups,
        "hints": hints,
    }


def compare_checks(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    metric = current["metric"]
    old = previous["result"].get(metric)
    new = current["result"][metric]
    old_flat, new_flat, notes = _comparable_flat(
        previous["fingerprint"], current["fingerprint"]
    )
    changes = [
        {"key": key, "old": before, "new": after}
        for key, before, after in diff_fingerprints(old_flat, new_flat)
    ]
    if previous.get("endpoint") != current.get("endpoint"):
        changes.insert(
            0, {"key": "endpoint", "old": previous.get("endpoint"), "new": current.get("endpoint")}
        )
    if previous.get("throttle_version") != current.get("throttle_version"):
        # A measurement-code change, not a serving change: say so.
        changes.append(
            {
                "key": "throttle_version (the measuring tool)",
                "old": previous.get("throttle_version"),
                "new": current.get("throttle_version"),
            }
        )
    comparison: dict[str, Any] = {
        "previous_id": previous.get("id"),
        "previous_endpoint": previous.get("endpoint"),
        "metric": metric,
        "changes": changes,
        "notes": notes,
        "previous": old,
        "current": new,
        "delta_dollars_per_million": None,
        "delta_percent": None,
        "measured_delta_percent": None,
        "rate_changed": False,
        "baseline_age_seconds": None,
        "verdict": VERDICT_NO_WINNER,
        "reason": "",
        "noise": None,
    }
    then, now = _created(previous), _created(current)
    age = (now - then).total_seconds() if then and now else None
    comparison["baseline_age_seconds"] = age
    workload_diffs = [
        field
        for field in WORKLOAD_IDENTITY_FIELDS
        if workload_value(previous, field) != workload_value(current, field)
    ]
    if not isinstance(old, dict) or old.get("mean") is None:
        comparison["reason"] = "the previous check has no usable figure"
        return comparison
    if workload_diffs:
        # No delta at all: a $/M difference between different workloads is
        # not a saving or a regression, and printing one invites reading it
        # as one.
        comparison["workload_differs"] = workload_diffs
        comparison["previous_workload"] = {f: workload_value(previous, f) for f in workload_diffs}
        comparison["current_workload"] = {f: workload_value(current, f) for f in workload_diffs}
        comparison["reason"] = (
            "the workload differs (" + ", ".join(workload_diffs) + "), so the "
            "numbers are not comparable; rerun with the same workload options"
        )
        if "prompt_cache_mode" in workload_diffs:
            before = workload_value(previous, "prompt_cache_mode")
            after = workload_value(current, "prompt_cache_mode")
            legacy = (
                " (checks before Throttle 0.4.1 always repeated identical prompts)"
                if (previous.get("workload") or {}).get("prompt_cache_mode") is None
                else ""
            )
            comparison["reason"] += (
                f". The prompt cache mode changed ({before} -> {after}{legacy}): a "
                "cold check sends a unique prompt per request, a warm check repeats "
                "identical prompts that a prefix cache can serve, so their $/M "
                "measure different traffic. Add or drop --warm-cache to match"
            )
        return comparison
    comparison["delta_dollars_per_million"] = new["mean"] - old["mean"]
    comparison["delta_percent"] = relative_delta_percent(new["mean"], old["mean"])
    # The verdict judges only what was MEASURED (tokens per wall-clock
    # second). $/M scales linearly with the ASSUMED GPU rate, so put the new
    # check on the baseline's rate before judging it.
    factor = rate_of(previous) / rate_of(current)
    comparison["rate_changed"] = factor != 1.0

    def scaled(summary: Mapping[str, Any], key: str) -> float | None:
        value = summary.get(key)
        return None if value is None else float(value) * factor

    pct = relative_delta_percent(new["mean"] * factor, old["mean"])
    comparison["measured_delta_percent"] = pct
    left = {"low": old.get("ci_low"), "high": old.get("ci_high")}
    right = {"low": scaled(new, "ci_low"), "high": scaled(new, "ci_high")}
    overlap = intervals_overlap(left, right)
    noise = run_to_run_noise(history, previous, current, metric)
    comparison["noise"] = noise
    if age is None or abs(age) > CALIBRATION_WINDOW_HOURS * 3600:
        comparison["verdict"] = VERDICT_NOT_CALIBRATED
        age_text = _age_text(age) if age is not None else "of unknown age"
        comparison["reason"] = (
            f"the baseline check is {age_text} old; run-to-run noise is only "
            f"trusted within {CALIBRATION_WINDOW_HOURS:g} h, so drift over that gap "
            "is unmeasured. Re-check the baseline config now"
        )
        return comparison
    if not noise["calibrated"]:
        comparison["verdict"] = VERDICT_NOT_CALIBRATED
        comparison["reason"] = (
            f"run-to-run noise unknown: {noise['degrees_of_freedom']} degree(s) of "
            f"freedom from repeat checks within {CALIBRATION_WINDOW_HOURS:g} h, "
            f"{MIN_CALIBRATION_DF} needed (e.g. 3 checks of the baseline config, not "
            "counting this one)"
        )
        return comparison
    floor = noise["floor_percent"]
    source = (
        f"{floor:.1f}% = t x sqrt(2) x {noise['sd_percent']:.1f}% run-to-run SD, "
        f"{noise['degrees_of_freedom']} df"
    )
    what = "measured change at the baseline's GPU rate" if comparison["rate_changed"] else "change"
    failed: list[str] = []
    if pct is None or abs(pct) <= floor:
        change_text = f"{pct:+.1f}%" if pct is not None else "n/a"
        failed.append(
            f"the {what} ({change_text}) is not larger than the run-to-run "
            f"noise bound ({source})"
        )
    if overlap:
        failed.append(
            "the 95% confidence intervals overlap, so the difference is within "
            "measurement noise"
        )
    if failed:
        comparison["reason"] = "; and ".join(failed)
        return comparison
    comparison["verdict"] = VERDICT_CHEAPER if pct < 0 else VERDICT_COSTLIER  # type: ignore[operator]
    comparison["reason"] = (
        f"the {pct:+.1f}% {what} is larger than the run-to-run noise bound "
        f"({source}) and the 95% confidence intervals do not overlap"
    )
    return comparison


# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------


def _money(value: float | None) -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 1000:
        return f"${value:,.0f}"
    if abs(value) >= 1:
        return f"${value:,.2f}"
    return f"${value:.4f}"


def _signed_money(value: float) -> str:
    sign = "+" if value > 0 else "-" if value < 0 else ""
    return f"{sign}{_money(abs(value))}"


def _ci_text(summary: Mapping[str, Any]) -> str:
    if summary.get("ci_low") is None:
        return "no CI (fewer than 2 blocks)"
    return f"95% CI {_money(summary['ci_low'])} to {_money(summary['ci_high'])}"


def _tokens_text(count: int) -> str:
    for factor, suffix in ((10**12, "T"), (10**9, "B"), (10**6, "M"), (10**3, "K")):
        if count >= factor:
            value = count / factor
            return f"{value:g}{suffix}" if value == int(value) else f"{value:.2f}{suffix}"
    return str(count)


def comparison_changes(comparison: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Config changes plus workload differences, as the terminal and the
    shared summary both list them (workload fields as ``workload.<field>``)."""

    changes = [dict(change) for change in comparison.get("changes") or []]
    for field in comparison.get("workload_differs") or []:
        changes.append({
            "key": f"workload.{field}",
            "old": (comparison.get("previous_workload") or {}).get(field),
            "new": (comparison.get("current_workload") or {}).get(field),
        })
    return changes


def _print_history(directory: Path, endpoint_filter: str | None, limit: int) -> int:
    records, skipped = load_history(directory)
    if endpoint_filter:
        records = [r for r in records if r.get("endpoint") == endpoint_filter]
    print(f"Throttle check history ({directory / HISTORY_FILENAME})")
    if skipped:
        print(f"  note: skipped {skipped} unreadable line(s)")
    if not records:
        print("  No checks recorded yet. Run `throttle check --url ... --model ... "
              "--gpu-hourly-rate ...` to create the first one.")
        return EXIT_OK
    shown = records[-limit:]
    print(f"  showing {len(shown)} of {len(records)} (newest last)")
    print()
    for record in shown:
        metric = record.get("metric", "output")
        summary = record["result"].get(metric) or {}
        config = record["fingerprint"].get("user_config") or {}
        config_text = " ".join(f"{k}={v}" for k, v in sorted(config.items())) or "-"
        print(f"  {record.get('id')}  {record.get('created_at', '')[:19]}Z")
        print(f"    endpoint  {record.get('endpoint')}  model {record['fingerprint'].get('model')}")
        print(f"    label     {record['fingerprint'].get('label') or '-'}   config {config_text}")
        print(f"    cache     {workload_value(record, 'prompt_cache_mode')}")
        print(
            f"    cost      {_money(summary.get('mean'))}/M {METRIC_LABELS.get(metric, metric)} "
            f"[MEASURED] ({_ci_text(summary)}) at "
            f"{_money(record['fingerprint'].get('gpu_hourly_rate_usd'))}/hr [ASSUMED]"
        )
    return EXIT_OK


def _metrics_url_problem(url: str) -> str | None:
    try:
        parts = urlsplit(url)
        parts.port  # raises ValueError for a malformed port
    except ValueError as exc:
        return f"is not a valid URL ({exc})"
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return "must be an http(s) URL with a host, e.g. http://localhost:8000/metrics"
    return None


def _limit_problem(
    args: argparse.Namespace, probes: int = 0
) -> tuple[str | None, float]:
    """Refuse a check that breaks a safety limit, before any traffic."""

    total_requests = args.blocks * args.requests_per_block + args.warmup + probes
    requested_tokens = (
        (args.blocks * args.requests_per_block + args.warmup) * args.max_tokens
        + probes * TAG_PROBE_MAX_TOKENS
    )
    spend_ceiling = args.gpu_hourly_rate * args.max_elapsed_seconds / 3600
    if total_requests > args.max_requests:
        return (
            f"this check would send {total_requests} requests "
            "(--blocks x --requests-per-block + --warmup"
            + (f" + {probes} untagged tag-cost probes" if probes else "")
            + f"), above --max-requests "
            f"{args.max_requests}. Lower the workload or raise --max-requests.",
            spend_ceiling,
        )
    if args.max_tokens > args.max_tokens_per_request:
        return (
            f"--max-tokens {args.max_tokens} is above --max-tokens-per-request "
            f"{args.max_tokens_per_request}. Lower it or raise --max-tokens-per-request.",
            spend_ceiling,
        )
    if requested_tokens > args.max_total_requested_tokens:
        return (
            f"this check could request {requested_tokens:,} output tokens, above "
            f"--max-total-requested-tokens {args.max_total_requested_tokens:,}. "
            "Lower the workload or raise the limit.",
            spend_ceiling,
        )
    if args.concurrency > args.max_concurrency:
        return (
            f"--concurrency {args.concurrency} is above --max-concurrency "
            f"{args.max_concurrency}. Lower it or raise --max-concurrency.",
            spend_ceiling,
        )
    if spend_ceiling > args.max_estimated_spend:
        return (
            f"worst-case GPU time for this check is {_money(spend_ceiling)} "
            f"({args.max_elapsed_seconds:g}s at {_money(args.gpu_hourly_rate)}/hr), above "
            f"--max-estimated-spend {_money(args.max_estimated_spend)}. Lower "
            "--max-elapsed-seconds or raise --max-estimated-spend.",
            spend_ceiling,
        )
    return None, spend_ceiling


async def _detect_model(chat_url: str, headers: Mapping[str, str]) -> tuple[str | None, str]:
    """Pick the model when the server lists exactly one; otherwise explain."""

    url = _models_url(chat_url)
    async with httpx.AsyncClient(transport=_TRANSPORT, timeout=MODELS_TIMEOUT_SECONDS) as client:
        models = await _fetch_models(client, url, headers, "")
    if models.get("status") != "read":
        return None, f"--model is required (could not list models at {url}: {models['status']})"
    ids = models.get("model_ids") or []
    if len(ids) == 1:
        return ids[0], f"Model: {ids[0]} (the only model this server lists; pass --model to choose)"
    listed = ", ".join(ids) if ids else "(none)"
    return None, f"--model is required; this server lists: {listed}"


# --------------------------------------------------------------------------
# Sharing results (--share)
# --------------------------------------------------------------------------

SHARE_ISSUE_URL = "https://github.com/KushagraKanaujia/throttle/issues/new"
SHARE_TEMPLATE = "share-results.yml"
# GitHub rejects very long URLs (and browsers truncate them); keep the
# pre-filled link comfortably below ~8 KB.
MAX_SHARE_URL_CHARS = 7000
SHARE_BEGIN = "----- BEGIN THROTTLE RESULTS (markdown) -----"
SHARE_END = "----- END THROTTLE RESULTS -----"
SHARE_TRUNCATED_NOTE = (
    "_(Summary truncated to fit in a link. The full summary was printed in "
    "your terminal by `throttle check --share`; paste the rest here.)_"
)
MASKED = "[masked: looks like a secret]"
HOST_REMOVED = "[host removed]"

_URL_TEXT = re.compile(r"\b[A-Za-z][A-Za-z0-9+.\-]*://[^\s|)\]>`'\"]+")
# 'user:password@host' (or 'user@host') without a scheme: the userinfo and
# the host both go, whatever the host looks like (bare name, IP, [v6]).
_USERINFO_HOST_TEXT = re.compile(
    r"(?<![\w.+\-])[^\s@/|`'\"()<>\[\],;]+@\[?[A-Za-z0-9_.:%\-]+\]?"
)
# 'token=abc...' / 'password: abc' inside free text: keep the key, mask the value.
_SECRET_PAIR_TEXT = re.compile(
    r"(?<![\w.\-])([A-Za-z][A-Za-z0-9_.\-]*)(\s*[=:]\s*)(\"[^\"]*\"|'[^']*'|[^\s,;&|)`]+)"
)
_IPV4_TEXT = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?::\d{1,5})?(?![\w.])")
# IPv6, including an IPv4-mapped tail (::ffff:10.1.2.3), which must be
# consumed whole: stopping at the first ':' would leave '.1.2.3' behind.
_IPV6_TEXT = re.compile(
    r"(?<![\w:])\[?(?=[0-9A-Fa-f:]*:[0-9A-Fa-f:]*:)"
    r"(?:[0-9A-Fa-f]{0,4}:){2,7}(?:(?:\d{1,3}\.){3}\d{1,3}|[0-9A-Fa-f]{0,4})"
    r"\]?(?::\d{1,5})?(?![\w:]|\.\d)"
)
_HOST_TEXT = re.compile(
    r"(?<![\w.\-])(?:[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?\.)+"
    r"(?:com|net|org|io|ai|dev|app|cloud|co|us|uk|de|eu|in|cn|jp|edu|gov|xyz|"
    r"sh|run|tech|site|internal|local|lan|localdomain|svc|corp|home|intra|"
    r"private|test|example|invalid)(?::\d{1,5})?(?![\w\-])",
    re.IGNORECASE,
)
# Any other dotted name whose last label starts with a letter
# (node7.cluster.k8s, gpu01.rack4): treated as a host. Versions (0.4.1) and
# model names (llama3.2:3b, Llama-3.1-8B, Qwen2.5-7B) have a last label that
# starts with a digit and are left alone, as are file names.
_DOTTED_NAME_TEXT = re.compile(
    r"(?<![\w.\-/@])(?:[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?\.)+"
    r"([A-Za-z][A-Za-z0-9\-]*[A-Za-z0-9])(?::\d{1,5})?(?![\w\-]|\.\w)"
)
_FILE_EXTENSIONS = frozenset({
    "json", "jsonl", "ndjson", "yaml", "yml", "toml", "ini", "cfg", "conf", "txt",
    "md", "py", "csv", "log", "safetensors", "gguf", "ggml", "bin", "pt", "pth",
    "onnx", "ckpt", "tar", "gz", "zip", "whl",
})
_LOCALHOST_TEXT = re.compile(r"(?<![\w\-])localhost(?::\d{1,5})?(?![\w\-])", re.IGNORECASE)
# Bare host:port (gpu01:80, gpu01:443, vllm:8000). Model tags such as
# llama3.2:3b or qwen2:7b have letters after the colon and are left alone.
_HOST_PORT_TEXT = re.compile(r"(?<![\w.\-/])[A-Za-z][\w\-]*:\d{2,5}(?![\w.])")
_ABS_PATH_TEXT = re.compile(r"(?<![\w.])(?:~|/)[^\s|`]*/([^\s/|`]+)")
_WORD_TEXT = re.compile(r"[^\s|`'\"()\[\]<>,;]+")
# Payment-style tokens: sk_live_..., tok_test_..., rk_live_...
_LIVE_TOKEN = re.compile(r"[A-Za-z]{2,}_(?:live|test)_[A-Za-z0-9]{8,}")
_SECRET_VALUE_PREFIXES = (
    "sk-", "sk_", "pk_", "rk_", "hf_", "ghp_", "gho_", "ghu_", "ghs_", "ghr_",
    "github_pat_", "xoxb-", "xoxp-", "xoxa-", "xoxs-", "glpat-", "AKIA", "ASIA",
    "AIza", "eyJ", "Bearer ", "bearer ", "Basic ",
)


def value_looks_secret(value: str) -> bool:
    """True for a value that looks like a key or token.

    Known key prefixes (sk-, hf_, ghp_, AKIA, JWTs, ...) and long unbroken
    strings that mix letters and many digits (random tokens, hex digests).
    Ordinary settings (fp8, 0.9, Meta-Llama-3-8B-Instruct) are not masked.
    """

    text = value.strip()
    if len(text) >= 10 and text.startswith(_SECRET_VALUE_PREFIXES):
        return True
    if _LIVE_TOKEN.fullmatch(text):
        return True
    if len(text) < 20 or re.search(r"\s", text):
        return False
    if not re.fullmatch(r"[A-Za-z0-9_\-+=.~/]+", text):
        return False
    core = text.rstrip("=")
    digits = sum(c.isdigit() for c in core)
    letters = sum(c.isalpha() for c in core)
    return letters > 0 and digits >= max(4, len(core) // 8) and "/" not in core.strip("/")


def _mask_secret_pair(match: re.Match[str]) -> str:
    key, separator, _value = match.groups()
    if _looks_like_secret(key):
        return f"{key}{separator}{MASKED}"
    return match.group(0)


def _drop_dotted_host(match: re.Match[str]) -> str:
    if match.group(1).lower() in _FILE_EXTENSIONS:
        return match.group(0)
    return HOST_REMOVED


def scrub_text(text: Any) -> str:
    """Remove URLs, hostnames, IPs, credentials and local paths from free text."""

    value = str(text)
    value = _URL_TEXT.sub("[url removed]", value)
    value = _USERINFO_HOST_TEXT.sub(HOST_REMOVED, value)
    value = _SECRET_PAIR_TEXT.sub(_mask_secret_pair, value)
    value = _IPV6_TEXT.sub(HOST_REMOVED, value)
    value = _IPV4_TEXT.sub(HOST_REMOVED, value)
    value = _LOCALHOST_TEXT.sub(HOST_REMOVED, value)
    value = _HOST_TEXT.sub(HOST_REMOVED, value)
    value = _DOTTED_NAME_TEXT.sub(_drop_dotted_host, value)
    value = _HOST_PORT_TEXT.sub(HOST_REMOVED, value)
    value = _ABS_PATH_TEXT.sub(lambda m: ".../" + m.group(1), value)
    return value


def scrub_key(key: Any) -> str:
    """A setting name for a shared summary.

    The namespace ('config', 'vllm', 'workload', ...) is ours and stays; the
    rest is user- or server-supplied and is scrubbed like any value, so a
    --config key cannot carry a hostname out.
    """

    text = str(key)
    head, dot, rest = text.partition(".")
    if not dot:
        return scrub_text(text)
    return head + dot + scrub_text(rest)


def share_value(key: str, value: Any) -> str:
    """A config value or label as it may appear in a shared summary."""

    if value is None:
        return "(not set)"
    last = re.split(r"[.]", key)[-1]
    if _looks_like_secret(last) or value_looks_secret(str(value)):
        return MASKED
    # Free text (a label such as 'retry with tok_live_...') can hold a
    # token among ordinary words: mask each word that looks like one.
    return _WORD_TEXT.sub(
        lambda m: MASKED if value_looks_secret(m.group(0)) else m.group(0),
        scrub_text(value),
    )


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _engine_of(fingerprint: Mapping[str, Any]) -> str:
    user = fingerprint.get("user_config") or {}
    for key in ("engine", "server", "backend", "inference_engine"):
        if user.get(key):
            return share_value(key, user[key]) + " (from --config)"
    if ((fingerprint.get("server_metrics") or {}).get("info")):
        return "vLLM (from /metrics)"
    owned = str(
        ((fingerprint.get("server_models") or {}).get("model_detail") or {}).get("owned_by") or ""
    ).lower()
    if owned in ("vllm", "sglang"):
        return {"vllm": "vLLM", "sglang": "SGLang"}[owned] + " (from /v1/models)"
    if owned in ("library", "ollama"):
        return "Ollama (likely; /v1/models owned_by=library)"
    return "unknown (add --config engine=NAME)"


def _gpu_of(fingerprint: Mapping[str, Any]) -> str:
    user = fingerprint.get("user_config") or {}
    for key in ("gpu", "gpu_type", "accelerator", "gpus"):
        if user.get(key):
            return share_value(key, user[key])
    return "not given (add --config gpu=NAME)"


def build_share_summary(
    record: Mapping[str, Any], comparison: Mapping[str, Any] | None
) -> str:
    """Sanitized markdown for one check. Never includes the endpoint URL,
    hostnames, IPs, file paths or keys; secret-looking values are masked."""

    fp = record["fingerprint"]
    metric = record.get("metric", "output")
    metric_label = METRIC_LABELS.get(metric, metric)
    workload = record.get("workload") or {}
    headline = (record.get("result") or {}).get(metric) or {}
    mode = workload_value(record, "prompt_cache_mode")
    mode_text = (
        "cold (unique prompt per request)" if mode == CACHE_MODE_COLD
        else "warm (identical prompts repeated)"
    )
    source = workload.get("prompts_source")
    source_text = "built-in" if source in (None, "built-in") else "custom prompt file"
    digest = str(workload.get("prompts_sha256") or "")[:12]
    model = share_value("model", fp.get("model"))
    lines = [
        f"## Throttle check: {model}",
        "",
        "| | |",
        "|---|---|",
        f"| Throttle version | {_cell(scrub_text(record.get('throttle_version') or 'unknown'))} |",
        f"| Engine | {_cell(_engine_of(fp))} |",
        f"| GPU | {_cell(_gpu_of(fp))} |",
        f"| Model | {_cell(model)} |",
        f"| GPU hourly rate | {_money(rate_of(record))}/hr (ASSUMED, user-supplied) |",
        "| Workload | "
        + _cell(
            f"{workload.get('blocks', '?')} blocks x {workload.get('requests_per_block', '?')} "
            f"requests, concurrency {workload.get('concurrency', '?')}, max "
            f"{workload.get('max_tokens', '?')} output tokens, temperature 0, "
            f"{workload.get('prompt_count', '?')} {source_text} prompts"
            + (f" (sha256 {digest})" if digest else "")
        )
        + " |",
        f"| Prompt cache | {mode_text} |",
        f"| Label | {_cell(share_value('label', fp.get('label')) if fp.get('label') else '-')} |",
        f"| Check | {_cell(scrub_text(record.get('id')))} ({str(record.get('created_at', ''))[:10]}) |",
        "",
        f"**Result:** {_money(headline.get('mean'))} per million {metric_label} "
        f"({_ci_text(headline)}, {headline.get('n_blocks', '?')} blocks) [MEASURED]",
        "",
    ]
    if comparison is None:
        lines += ["**Compared with:** nothing (first check of this endpoint).", ""]
    else:
        old = comparison.get("previous") or {}
        lines.append(f"**Compared with** check {scrub_text(comparison.get('previous_id'))}:")
        lines.append("")
        lines.append(f"- before: {_money(old.get('mean'))}/M {metric_label} ({_ci_text(old)})")
        lines.append(f"- after: {_money(headline.get('mean'))}/M {metric_label} ({_ci_text(headline)})")
        if comparison.get("delta_percent") is not None:
            typed = " at the GPU rates as typed" if comparison.get("rate_changed") else ""
            lines.append(f"- change: {comparison['delta_percent']:+.1f}%{typed}")
            if comparison.get("rate_changed") and comparison.get("measured_delta_percent") is not None:
                lines.append(
                    f"- measured change at the baseline's GPU rate: "
                    f"{comparison['measured_delta_percent']:+.1f}%"
                )
        noise = comparison.get("noise")
        if noise is None:
            noise_text = "not evaluated (no comparable measurement; see the verdict)"
        elif noise.get("calibrated"):
            noise_text = (
                f"calibrated: bound {noise['floor_percent']:.1f}% "
                f"(t x sqrt(2) x {noise['sd_percent']:.1f}% run-to-run SD, "
                f"{noise['degrees_of_freedom']} df)"
            )
        else:
            noise_text = (
                f"NOT CALIBRATED ({noise.get('degrees_of_freedom', 0)} df from repeat "
                f"checks, {MIN_CALIBRATION_DF} needed)"
            )
        lines.append(f"- noise floor: {noise_text}")
        lines.append(
            f"- **Verdict: {comparison.get('verdict')}**, "
            f"{scrub_text(comparison.get('reason') or '')}"
        )
        lines.append("")
        changes = comparison_changes(comparison)
        if changes:
            lines.append("**What changed in config:**")
            lines.append("")
            for change in changes:
                key = str(change.get("key"))
                if key == "endpoint":
                    lines.append("- endpoint: changed (URLs are not shared)")
                    continue
                lines.append(
                    f"- `{scrub_key(key)}`: {share_value(key, change.get('old'))} -> "
                    f"{share_value(key, change.get('new'))}"
                )
        else:
            lines.append("**What changed in config:** nothing Throttle could see.")
        lines.append("")
    lines.append(
        "_Shared with `throttle check --share`. No URLs, hostnames, IPs or keys "
        "are included; config values that look like secrets are masked._"
    )
    return "\n".join(lines)


def _share_title(record: Mapping[str, Any], comparison: Mapping[str, Any] | None) -> str:
    verdict = comparison.get("verdict") if comparison else "first check"
    return f"[Results] {share_value('model', record['fingerprint'].get('model'))}: {verdict}"


def build_share_url(
    record: Mapping[str, Any],
    comparison: Mapping[str, Any] | None,
    summary: str,
    max_chars: int = MAX_SHARE_URL_CHARS,
) -> tuple[str, bool]:
    """Pre-filled issue-form URL; returns (url, whether the summary was cut)."""

    fp = record["fingerprint"]
    fields = {
        "template": SHARE_TEMPLATE,
        "title": _share_title(record, comparison),
        "engine": _engine_of(fp),
        "gpu": _gpu_of(fp),
        "model": share_value("model", fp.get("model")),
        "verdict": comparison.get("verdict") if comparison else "first check (no comparison)",
    }

    def url_for(body: str) -> str:
        return SHARE_ISSUE_URL + "?" + urlencode({**fields, "summary": body}, quote_via=quote)

    url = url_for(summary)
    if len(url) <= max_chars:
        return url, False
    kept = summary.splitlines()
    while kept:
        kept.pop()
        body = "\n".join(kept).rstrip() + "\n\n" + SHARE_TRUNCATED_NOTE
        url = url_for(body)
        if len(url) <= max_chars:
            return url, True
    return url_for(SHARE_TRUNCATED_NOTE), True


def _print_share(record: Mapping[str, Any], comparison: Mapping[str, Any] | None) -> None:
    summary = build_share_summary(record, comparison)
    url, truncated = build_share_url(record, comparison, summary)
    print("Share these results (nothing is uploaded; review before posting)")
    print(SHARE_BEGIN)
    print(summary)
    print(SHARE_END)
    print()
    print("Open a pre-filled GitHub issue (you review it and click Submit yourself):")
    print(f"  {url}")
    if truncated:
        print(
            f"  note: the summary was cut to keep the link under {MAX_SHARE_URL_CHARS:,} "
            "characters; paste the full summary above into the issue."
        )


def _share_recorded(directory: Path, endpoint_filter: str | None, check_id: str | None) -> int:
    """--share-id ID, or --history --share: share a recorded check, no traffic."""

    records, _skipped = load_history(directory)
    if check_id:
        indexes = [i for i, r in enumerate(records) if r.get("id") == check_id]
        if not indexes:
            print(
                f"Error: no check with id {check_id!r} in {directory / HISTORY_FILENAME} "
                "(see `throttle check --history`)",
                file=sys.stderr,
            )
            return EXIT_USAGE
    else:
        indexes = [
            i for i, r in enumerate(records)
            if not endpoint_filter or r.get("endpoint") == endpoint_filter
        ]
        if not indexes:
            print(
                f"Error: no recorded check to share in {directory / HISTORY_FILENAME}",
                file=sys.stderr,
            )
            return EXIT_USAGE
    index = indexes[-1]
    record = records[index]
    earlier = records[:index]
    # Re-judge it as it was judged when it ran: against the baseline it was
    # compared with (recorded since 0.4.1), else the previous check of its
    # endpoint, using only the history that existed then.
    baseline = None
    wanted = record.get("baseline_id")
    if isinstance(wanted, str):
        baseline = next((r for r in reversed(earlier) if r.get("id") == wanted), None)
    if baseline is None:
        same = [r for r in earlier if r.get("endpoint") == record.get("endpoint")]
        baseline = same[-1] if same else None
    comparison = compare_checks(baseline, record, earlier) if baseline else None
    _print_share(record, comparison)
    return EXIT_OK


# --------------------------------------------------------------------------
# Command
# --------------------------------------------------------------------------


def handle_check(args: argparse.Namespace) -> int:
    directory = history_dir(args)

    chat_url: str | None = None
    if args.url:
        try:
            chat_url = normalize_chat_completions_url(
                args.url, allow_insecure_http=args.allow_insecure_http
            )
        except ValueError as exc:
            print(f"Error: --url {exc}", file=sys.stderr)
            return EXIT_USAGE

    if args.share_id or (args.history and args.share):
        return _share_recorded(directory, chat_url, args.share_id)
    if args.history:
        return _print_history(directory, chat_url, args.limit)

    missing = [
        flag
        for flag, value in (
            ("--url", args.url),
            ("--gpu-hourly-rate", args.gpu_hourly_rate),
        )
        if not value
    ]
    if missing:
        print(
            "Error: throttle check needs " + ", ".join(missing)
            + " (or use --history to list past checks, --share-id ID to share one)",
            file=sys.stderr,
        )
        return EXIT_USAGE
    if args.blocks < 3:
        print(
            "Error: --blocks must be at least 3; the confidence interval is "
            "computed across blocks and 2 blocks gives an interval too wide to use",
            file=sys.stderr,
        )
        return EXIT_USAGE
    assert chat_url is not None

    if args.metrics_url and not args.no_metrics:
        problem = _metrics_url_problem(args.metrics_url)
        if problem:
            print(f"Error: --metrics-url {problem}", file=sys.stderr)
            return EXIT_USAGE

    try:
        prompts = load_prompts(args.prompts)
        warmups = load_prompts(warmup=True)
    except (OSError, ValueError) as exc:
        print(f"Error: could not load prompts: {exc}", file=sys.stderr)
        return EXIT_USAGE
    if not prompts:
        print("Error: the prompt file is empty", file=sys.stderr)
        return EXIT_USAGE
    cache_mode = CACHE_MODE_WARM if args.warm_cache else CACHE_MODE_COLD
    probes = tag_probe_count(prompts, args.requests_per_block, cache_mode)

    limit_problem, spend_ceiling = _limit_problem(args, probes)
    if limit_problem:
        print(f"Error: {limit_problem}", file=sys.stderr)
        return EXIT_USAGE

    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    if not args.model:
        chosen, message = asyncio.run(_detect_model(chat_url, headers))
        if chosen is None:
            print(f"Error: {message}", file=sys.stderr)
            return EXIT_USAGE
        print(message)
        args.model = chosen


    previous_records, skipped_lines = load_history(directory)
    if skipped_lines:
        print(
            f"note: skipped {skipped_lines} unreadable line(s) in "
            f"{directory / HISTORY_FILENAME}; they are not used as a baseline or "
            "for calibration"
        )
    baseline: dict[str, Any] | None = None
    if args.against:
        matches = [r for r in previous_records if r.get("id") == args.against]
        if not matches:
            print(
                f"Error: no check with id {args.against!r} in {directory / HISTORY_FILENAME} "
                "(see `throttle check --history`)",
                file=sys.stderr,
            )
            return EXIT_USAGE
        baseline = matches[-1]
    else:
        same = [r for r in previous_records if r.get("endpoint") == chat_url]
        baseline = same[-1] if same else None

    user_config = dict(args.config_pairs or [])
    metric = args.metric
    metric_label = METRIC_LABELS[metric]
    run_id = new_run_id() if cache_mode == CACHE_MODE_COLD else None
    plan = measured_prompts(
        prompts, args.blocks, args.requests_per_block, cache_mode, run_id
    )
    workload = {
        # Hash of the prompt file itself (the same canonical hash smoke,
        # benchmark and golden record), not of the tagged prompts: the tags
        # change every run, the workload does not.
        "prompts_sha256": canonical_workload_hash(prompts),
        "prompt_cache_mode": cache_mode,
        "prompt_nonce": (
            {
                "run_id": run_id,
                "format": NONCE_FORMAT,
                "warmup_format": WARMUP_NONCE_FORMAT,
                "placement": "start of the first user message",
                "index": "measured request number across blocks, from 0",
                # Filled in after the probes: prompt tokens of each base
                # prompt sent untagged, and the tag tokens subtracted.
                "untagged_prompt_tokens": None,
                "tag_input_tokens": None,
                "tag_input_tokens_per_request": None,
                "input_tokens": (
                    "net of the tag: each distinct base prompt is sent once "
                    "untagged (unmeasured, max_tokens "
                    f"{TAG_PROBE_MAX_TOKENS}) and blocks count those prompt tokens"
                ),
            }
            if run_id
            else None
        ),
        # Hash of every measured request's messages in send order, so the
        # exact traffic can be rebuilt from the prompt file and run_id.
        "sent_prompts_sha256": canonical_workload_hash(
            tuple(tuple(m) for row in plan for m in row)
        ),
        "prompt_count": len(prompts),
        "prompts_source": str(args.prompts) if args.prompts else "built-in",
        "blocks": args.blocks,
        "requests_per_block": args.requests_per_block,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "warmup_requests": args.warmup,
        "temperature": 0,
    }

    print(f"Throttle check: what does {args.model} cost per million {metric_label}?")
    print(f"  endpoint   {chat_url}")
    print(
        f"  GPU rate   {_money(args.gpu_hourly_rate)}/hr   "
        "[ASSUMED: you supplied --gpu-hourly-rate; Throttle cannot see your bill]"
    )
    print(
        f"  workload   {args.blocks} blocks x {args.requests_per_block} requests, "
        f"concurrency {args.concurrency}, max {args.max_tokens} output tokens, "
        f"temperature 0, {len(prompts)} prompts ({workload['prompts_source']})"
    )
    if cache_mode == CACHE_MODE_COLD:
        print(
            f"  cache      COLD (default): each request's first user message starts "
            f"with a unique tag, e.g. '{NONCE_FORMAT.format(run_id=run_id, index=0)}', "
            "so a prefix cache cannot serve repeats (run id recorded)"
        )
        print(
            f"             the tag's own prompt tokens are not counted: {probes} "
            "unmeasured request(s) send each base prompt once untagged, and input "
            "and total $/M use those token counts"
        )
    else:
        print(
            "  cache      WARM (--warm-cache): identical prompts are resent, so a "
            "server with prefix caching may serve them from cache; compared only "
            "with other --warm-cache checks"
        )
    total_requests = args.blocks * args.requests_per_block + args.warmup + probes
    requested_tokens = (
        (args.blocks * args.requests_per_block + args.warmup) * args.max_tokens
        + probes * TAG_PROBE_MAX_TOKENS
    )
    print(
        f"  limits     {total_requests} requests, at most "
        f"{requested_tokens:,} output tokens requested, stops after "
        f"{args.max_elapsed_seconds:g}s; worst-case GPU time "
        f"{_money(spend_ceiling)} [ASSUMED rate] (ceiling {_money(args.max_estimated_spend)})"
    )
    print()

    async def run() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
        async with httpx.AsyncClient(
            transport=_TRANSPORT,
            timeout=httpx.Timeout(args.request_timeout),
            limits=httpx.Limits(max_connections=max(args.concurrency, 1) + 2),
        ) as client:
            models = await _fetch_models(client, _models_url(chat_url), headers, args.model)
            if args.no_metrics:
                metrics: dict[str, Any] = {"status": "skipped (--no-metrics)"}
            else:
                metrics = await _fetch_metrics(
                    client, args.metrics_url or _default_metrics_url(chat_url)
                )
            if args.warmup:
                print(f"Warming up ({args.warmup} unmeasured request(s))...", flush=True)
                for index in range(args.warmup):
                    warmup = warmups[index % len(warmups)]
                    if run_id:
                        warmup = tag_prompt(
                            warmup, WARMUP_NONCE_FORMAT.format(run_id=run_id, index=index)
                        )
                    await _send(
                        client, chat_url, headers, args.model, warmup, args.max_tokens,
                    )
            untagged: list[int] = []
            if probes:
                # Untagged, so they share no more prefix with the measured
                # (tagged) prompts than the tagged prompts share with each
                # other: the system message and chat template only.
                for index in range(probes):
                    prompt_tokens, _ = await _send(
                        client, chat_url, headers, args.model, prompts[index],
                        TAG_PROBE_MAX_TOKENS,
                    )
                    untagged.append(prompt_tokens)
                workload["prompt_nonce"]["untagged_prompt_tokens"] = untagged
            real_input_per_block = (
                sum(untagged[j % len(untagged)] for j in range(args.requests_per_block))
                if untagged else None
            )
            tag_total = 0
            blocks: list[dict[str, Any]] = []
            for block_index in range(args.blocks):
                sent_input, output_tokens, wall = await _run_block(
                    client, chat_url, headers, args.model, plan[block_index],
                    args.requests_per_block, args.concurrency, args.max_tokens,
                )
                input_tokens = sent_input
                tag_tokens = 0
                if real_input_per_block is not None:
                    tag_tokens = sent_input - real_input_per_block
                    if tag_tokens < 0:
                        raise CheckError(
                            f"the endpoint reported {sent_input} prompt tokens for a "
                            f"block of tagged prompts but {real_input_per_block} for the "
                            "same prompts untagged, so the tag's token cost cannot be "
                            "netted out (rerun with --warm-cache to send no tags)"
                        )
                    input_tokens = real_input_per_block
                    tag_total += tag_tokens
                if input_tokens == 0 or output_tokens == 0:
                    raise CheckError(
                        f"the endpoint reported {input_tokens} input / {output_tokens} "
                        "output tokens for a whole block, so $/M tokens is undefined"
                    )
                costs = _block_costs(input_tokens, output_tokens, wall, args.gpu_hourly_rate)
                blocks.append(
                    {
                        # Real prompt tokens; in cold mode the tag's tokens
                        # are excluded and recorded separately.
                        "input_tokens": input_tokens,
                        "tag_input_tokens": tag_tokens,
                        "output_tokens": output_tokens,
                        "wall_clock_seconds": wall,
                        "dollars_per_million": {k: costs[k] for k in METRIC_LABELS},
                        "gpu_dollars": costs["gpu_dollars"],
                    }
                )
                tag_text = f" (+{tag_tokens} tag, not counted)" if untagged else ""
                print(
                    f"  block {block_index + 1}/{args.blocks}: {input_tokens} in{tag_text} / "
                    f"{output_tokens} out tokens in {wall:.2f}s -> "
                    f"{_money(costs[metric])}/M {metric_label}",
                    flush=True,
                )
            if untagged:
                workload["prompt_nonce"]["tag_input_tokens"] = tag_total
                workload["prompt_nonce"]["tag_input_tokens_per_request"] = tag_total / (
                    args.blocks * args.requests_per_block
                )
            return models, metrics, blocks

    async def bounded() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
        try:
            return await asyncio.wait_for(run(), timeout=args.max_elapsed_seconds)
        except asyncio.TimeoutError:
            raise CheckError(
                f"stopped at the --max-elapsed-seconds ceiling ({args.max_elapsed_seconds:g}s) "
                "before every block finished"
            ) from None

    try:
        models, metrics, blocks = asyncio.run(bounded())
    except CheckError as exc:
        print()
        print(f"Check failed: {exc}", file=sys.stderr)
        print("Nothing was recorded.", file=sys.stderr)
        return EXIT_FAILED

    result = {
        key: summarize([block["dollars_per_million"][key] for block in blocks])
        for key in METRIC_LABELS
    }
    created = datetime.now(timezone.utc)
    fingerprint = {
        "model": args.model,
        "label": args.label,
        "gpu_hourly_rate_usd": args.gpu_hourly_rate,
        "gpu_hourly_rate_source": "ASSUMED (user-supplied --gpu-hourly-rate)",
        "user_config": user_config,
        "server_models": models,
        "server_metrics": metrics,
    }
    identity = hashlib.sha256(
        json.dumps([chat_url, created.isoformat(), fingerprint], sort_keys=True, default=str).encode()
    ).hexdigest()[:8]
    record: dict[str, Any] = {
        "record_type": RECORD_TYPE,
        "record_version": RECORD_VERSION,
        "id": f"{created.strftime('%Y%m%dT%H%M%SZ')}-{identity}",
        "created_at": created.isoformat().replace("+00:00", "Z"),
        "throttle_version": __version__,
        "endpoint": chat_url,
        "metric": metric,
        "fingerprint": fingerprint,
        "workload": workload,
        "blocks": blocks,
        "result": result,
        "monthly_tokens": args.monthly_tokens,
    }

    headline = result[metric]
    print()
    print("Config fingerprint")
    print(f"  /v1/models  {models['status']}")
    print(f"  /metrics    {metrics['status']}")
    for key, value in flatten_fingerprint(fingerprint).items():
        print(f"    {key} = {value}")
    print()
    print("Result")
    print(
        f"  {_money(headline['mean'])} per million {metric_label}   [MEASURED]"
    )
    print(f"  {_ci_text(headline)}, across {headline['n_blocks']} blocks (Student t)")
    others = [key for key in METRIC_LABELS if key != metric]
    print(
        "  same GPU spend per other tokens (not additive): " + "; ".join(
            f"{_money(result[key]['mean'])}/M {METRIC_LABELS[key]}" for key in others
        )
    )
    print(
        f"  = {_money(args.gpu_hourly_rate)}/hr [ASSUMED] x measured wall-clock / "
        f"measured tokens, at concurrency {args.concurrency}. Cost at a different "
        "production concurrency will differ."
    )
    if args.monthly_tokens:
        volume = args.monthly_tokens / 1_000_000
        low = headline["ci_low"]
        high = headline["ci_high"]
        range_text = (
            f" (95% CI {_money(low * volume)} to {_money(high * volume)})"
            if low is not None
            else ""
        )
        print(
            f"  monthly at {_tokens_text(args.monthly_tokens)} {metric_label}: "
            f"{_money(headline['mean'] * volume)}{range_text}   "
            "[PROJECTED: MEASURED $/M x ASSUMED volume]"
        )

    comparison: dict[str, Any] | None = None
    print()
    if baseline is None:
        print("Compared with: nothing yet. This is the first check for this endpoint;")
        print("the next check will be compared against it.")
    else:
        comparison = compare_checks(baseline, record, previous_records)
        record["baseline_id"] = baseline.get("id")
        base_fp = baseline["fingerprint"]
        age = comparison.get("baseline_age_seconds")
        age_text = f", {_age_text(age)} ago" if age is not None else ", age unknown"
        print(
            f"Compared with check {baseline.get('id')} "
            f"({str(baseline.get('created_at', ''))[:19]}Z{age_text}, "
            f"label {base_fp.get('label') or '-'})"
        )
        if baseline.get("endpoint") != chat_url:
            print(
                f"  WARNING: that check measured a different endpoint "
                f"({baseline.get('endpoint')}); this is not a before/after of one server"
            )
        changes = comparison_changes(comparison)
        if changes:
            print("  What changed in the config:")
            for change in changes:
                before = change["old"] if change["old"] is not None else "(not set)"
                after = change["new"] if change["new"] is not None else "(not set)"
                print(f"    {change['key']}: {before} -> {after}")
        else:
            print(
                "  What changed in the config: nothing Throttle can see "
                "(add --config KEY=VALUE for settings the endpoint does not report)"
            )
        for note in comparison["notes"]:
            print(f"  note    {note}")
        old_summary = comparison["previous"] or {}
        print(f"  before  {_money(old_summary.get('mean'))}/M {metric_label}  ({_ci_text(old_summary)})")
        print(f"  now     {_money(headline['mean'])}/M {metric_label}  ({_ci_text(headline)})")
        delta = comparison["delta_dollars_per_million"]
        pct = comparison["delta_percent"]
        if delta is not None:
            pct_text = f" ({pct:+.1f}%)" if pct is not None else ""
            print(f"  change  {_signed_money(delta)} per million {metric_label}{pct_text}")
        elif comparison.get("workload_differs"):
            print(
                "  change  not computed: the two checks ran different workloads, so the "
                "gap between them is not a saving or a regression"
            )
        old_rate = float(base_fp["gpu_hourly_rate_usd"])
        measured_pct = comparison.get("measured_delta_percent")
        if comparison.get("rate_changed") and delta is not None and measured_pct is not None:
            adjusted = headline["mean"] * old_rate / args.gpu_hourly_rate
            rate_pct = (args.gpu_hourly_rate / old_rate - 1) * 100
            print(
                f"  note    the GPU rate changed ({_money(old_rate)}/hr -> "
                f"{_money(args.gpu_hourly_rate)}/hr) [ASSUMED]: that alone moves $/M by "
                f"{rate_pct:+.1f}%, arithmetic on rates you typed, not a measurement."
            )
            print(
                f"          At the old rate this check measures {_money(adjusted)}/M "
                f"({measured_pct:+.1f}% vs before) [MEASURED throughput]; the verdict "
                "judges only that part."
            )
        noise = comparison.get("noise")
        if noise is not None:
            window = f"{noise['window_hours']:g} h"
            if noise["calibrated"]:
                print(
                    f"  noise   run-to-run noise bound {noise['floor_percent']:.1f}% "
                    f"[MEASURED from {noise['n_checks']} earlier checks of unchanged "
                    f"configs within {window}: t x sqrt(2) x {noise['sd_percent']:.1f}% "
                    f"SD, {noise['degrees_of_freedom']} df]"
                )
            else:
                print(
                    f"  noise   run-to-run noise bound: not measured (needs 3 earlier "
                    f"checks of one config and workload within {window}, not counting "
                    "this one; or 2 each of the before and after configs)"
                )
            for group in noise["groups"]:
                spread = group["spread_percent"]
                spread_text = f"spread {spread:.1f}%" if spread is not None else "no repeat yet"
                print(
                    f"          {group['side']}: label {group['label'] or '-'}, "
                    f"{group['n_checks']} earlier check(s), {spread_text}"
                )
            if noise["stale_or_undated_checks_ignored"]:
                print(
                    f"          ignored {noise['stale_or_undated_checks_ignored']} check(s) "
                    f"older than {window} or undated"
                )
            for hint in noise["hints"]:
                print(f"  hint    {hint}")
        if args.monthly_tokens and delta is not None:
            if comparison["verdict"] == VERDICT_NOT_CALIBRATED:
                print(
                    "  monthly not projected: run-to-run noise is not calibrated, so "
                    "this change is unverified"
                )
            elif comparison["verdict"] == VERDICT_NO_WINNER:
                print(
                    "  monthly not projected: with NO WINNER the difference is not "
                    "distinguishable from noise"
                )
            else:
                monthly = delta * args.monthly_tokens / 1_000_000
                rate_note = (
                    ", includes the ASSUMED GPU rate change"
                    if comparison.get("rate_changed") else ""
                )
                print(
                    f"  monthly {_signed_money(monthly)} per month at "
                    f"{_tokens_text(args.monthly_tokens)} {metric_label}   "
                    f"[PROJECTED: MEASURED $/M x ASSUMED volume{rate_note}]"
                )
        if comparison["verdict"] == VERDICT_NOT_CALIBRATED:
            print(f"  why     {comparison['reason']}.")
            print(f"  {NOT_CALIBRATED_MESSAGE}")
        else:
            print(f"  Verdict: {comparison['verdict']}, {comparison['reason']}.")

    record["comparison"] = comparison
    print()
    if args.no_save:
        print("Not saved (--no-save).")
    else:
        try:
            path = append_history(directory, {k: v for k, v in record.items() if k != "comparison"})
        except OSError as exc:
            print(f"Error: could not save the check to {directory}: {exc}", file=sys.stderr)
            return EXIT_FAILED
        print(f"Saved as check {record['id']} in {path}")
    if args.json_output:
        try:
            args.json_output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        except OSError as exc:
            print(f"Error: could not write {args.json_output}: {exc}", file=sys.stderr)
            return EXIT_FAILED
        print(f"JSON written to {args.json_output}")
    if args.share:
        print()
        _print_share(record, comparison)

    if (
        args.fail_if_costlier is not None
        and comparison is not None
        and comparison["verdict"] == VERDICT_COSTLIER
        and comparison["measured_delta_percent"] is not None
        and comparison["measured_delta_percent"] >= args.fail_if_costlier
    ):
        print(
            f"FAIL: significantly more expensive by {comparison['measured_delta_percent']:.1f}% "
            f"(threshold {args.fail_if_costlier:g}%). Exit code {EXIT_COSTLIER}.",
            file=sys.stderr,
        )
        return EXIT_COSTLIER
    if (
        args.fail_if_costlier is not None
        and comparison is not None
        and comparison.get("workload_differs")
    ):
        # NO WINNER here means "not comparable", not "no difference": the gate
        # judged nothing, so it must not report success. The usual cause is an
        # upgrade from 0.4.0 (old checks count as warm-cache, new ones are cold).
        print(
            "WARNING: the baseline ran a different workload ("
            + ", ".join(comparison["workload_differs"])
            + "), so --fail-if-costlier could not judge this change. Rerun with the "
            "baseline's workload options (or pass --against a check that used them). "
            f"Exit code {EXIT_NOT_CALIBRATED} (treat as a warning or a failure).",
            file=sys.stderr,
        )
        return EXIT_NOT_CALIBRATED
    if (
        args.fail_if_costlier is not None
        and comparison is not None
        and comparison["verdict"] == VERDICT_NOT_CALIBRATED
    ):
        print(
            "WARNING: NOT CALIBRATED, so --fail-if-costlier could not judge this "
            f"change. Exit code {EXIT_NOT_CALIBRATED} (treat as a warning or a failure).",
            file=sys.stderr,
        )
        return EXIT_NOT_CALIBRATED
    return EXIT_OK
