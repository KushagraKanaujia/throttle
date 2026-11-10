"""Platform-aware runtime provenance for persisted benchmark manifests."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any, Mapping

from .models import RunConfig

CURRENT_MANIFEST_VERSION = "1.1"
SUPPORTED_MANIFEST_VERSIONS = frozenset({"1.0", CURRENT_MANIFEST_VERSION})
ACCELERATOR_BACKENDS = ("cuda", "metal", "rocm", "cpu")
_SHA256 = r"sha256:[0-9a-f]{64}"
_ARTIFACT_LABEL = r"[A-Za-z0-9][A-Za-z0-9._/+:-]{0,183}"
_ARTIFACT_DIGEST = r"[A-Za-z][A-Za-z0-9._+-]{0,31}:[0-9A-Fa-f]{16,256}"
IMMUTABLE_ARTIFACT = re.compile(
    rf"^(?:{_ARTIFACT_LABEL}@)?{_SHA256}$"
)
SAFE_ARTIFACT_REFERENCE = re.compile(
    rf"^(?:{_ARTIFACT_DIGEST}|{_ARTIFACT_LABEL}(?:@{_ARTIFACT_DIGEST})?)$"
)

# An env-var style label: OPENAI_KEY, STRIPE_API_KEY, aws-secret, GH_TOKEN.
# Bounded repeats keep the scan linear on long snake_case text.
_ENV_LABEL = (
    r"[a-z0-9]{1,40}(?:[_-][a-z0-9]{1,40}){0,6}[_-]"
    r"(?:api[_-]?)?(?:key|token|secret|password|pass)"
)
_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?:^|[^a-z0-9])(?:access[\s_-]*token|api[\s_-]*key|"
    r"auth[\s_-]*token|authorization|bearer|client[\s_-]*secret|"
    r"credentials?|password|passwd|private[\s_-]*key|"
    r"secret(?:[\s_-]*(?:access[\s_-]*)?key)?|token|"
    + _ENV_LABEL
    + r")\s*[:=]",
    re.IGNORECASE,
)
# A label followed by a space and then the value: "api-key sk123...",
# "OPENAI_KEY abc123...". The value check (letters and digits) is the
# caller's job, so "api key is" is left alone.
_CREDENTIAL_SPACED = re.compile(
    r"(?:^|[^a-z0-9])(?:api[\s_-]*key|access[\s_-]*token|auth[\s_-]*token|"
    r"client[\s_-]*secret|secret[\s_-]*(?:access[\s_-]*)?key|"
    # underscores only: "completion-token gaps" is English, OPENAI_KEY is not
    r"[a-z0-9]{1,40}(?:_[a-z0-9]{1,40}){0,6}_(?:api_?)?(?:key|token|secret)"
    r")[ \t]+[\"']?([^\s\"',;}\])]{8,})",
    re.IGNORECASE,
)


def spaced_credentials(text: str) -> list[re.Match[str]]:
    """``api-key <value>`` style matches whose value has letters and digits
    (group 1 is the value), so "api key is" or "api key for" are left alone."""
    return [
        m for m in _CREDENTIAL_SPACED.finditer(text)
        if re.search(r"\d", m.group(1)) and re.search(r"[A-Za-z]", m.group(1))
        and not m.group(1).startswith("<")
    ]
# PEM / OpenSSH / PGP private keys, header to footer (or to the end of the
# text when the footer is missing). Works on real newlines and on "\\n".
_PRIVATE_KEY_BLOCK = re.compile(
    r"-----BEGIN[A-Z0-9 ]{0,40}PRIVATE KEY(?: BLOCK)?-----.*?"
    r"(?:-----END[A-Z0-9 ]{0,40}PRIVATE KEY(?: BLOCK)?-----|\Z)",
    re.DOTALL,
)
# Incoming-webhook URLs carry their secret in the path.
_WEBHOOK_URL = re.compile(
    r"https?://(?:hooks\.slack\.com/(?:services|workflows|triggers)/[A-Za-z0-9/_-]+|"
    r"(?:[a-z]+\.)?discord(?:app)?\.com/api/webhooks/\d+/[A-Za-z0-9_-]+)",
    re.IGNORECASE,
)
# Discord bot tokens: base64 user id . timestamp . HMAC.
_DISCORD_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_.-])[MNO][A-Za-z0-9_-]{23,27}\.[A-Za-z0-9_-]{6,7}\."
    r"[A-Za-z0-9_-]{27,40}(?![A-Za-z0-9_.-])"
)
# user:password@ in a URL. The whole userinfo is the secret.
_URL_USERINFO = re.compile(r"(?<=://)[^\s/?#@:\"'<>]{0,256}:[^\s/?#@\"'<>]{1,256}@")
# An AWS secret access key (40 base64 characters) within a short distance
# after an access key id.
_AWS_SECRET_NEAR_ID = re.compile(
    r"(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])[^\n]{0,80}?"
    r"(?<![A-Za-z0-9/+=])([A-Za-z0-9/+]{40})(?![A-Za-z0-9/+=])"
)
# A long hex or base64 run right before or after the word key, token or
# secret ("key 9f86d0...", "the token is QUJD...", "...a08 secret").
_LABELLED_RUN = re.compile(
    r"\b(?:keys?|tokens?|secrets?)\b[^\n]{0,24}?"
    r"(?<![A-Za-z0-9+/=_-])([A-Za-z0-9+/_-]{32,}={0,2})(?![A-Za-z0-9+/=_-])"
    r"|(?<![A-Za-z0-9+/=_-])([A-Za-z0-9+/_-]{32,}={0,2})(?![A-Za-z0-9+/=_-])"
    r"[^\n]{0,12}?\b(?:key|token|secret)\b",
    re.IGNORECASE,
)
_CREDENTIAL_TOKEN = re.compile(
    r"(?:^|[^A-Za-z0-9])(?:(?:hf|npm|pk|pypi|rk|rpa|rps|sk)"
    r"[-_][A-Za-z0-9_-]{8,}|"
    r"gh[oprsu]_[A-Za-z0-9_]{8,}|github_pat_[A-Za-z0-9_]{8,}|"
    r"glpat-[A-Za-z0-9_-]{8,}|xox[a-z](?:\.xox[a-z])?-[A-Za-z0-9-]{8,}|"
    r"xapp-[A-Za-z0-9-]{8,}|whsec_[A-Za-z0-9+/=]{16,}|"
    r"(?:AKIA|ASIA)[A-Z0-9]{16}|AIza[A-Za-z0-9_-]{20,}|"
    r"SG\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}|"
    r"ya29\.[A-Za-z0-9_-]{8,})(?=$|@|[^A-Za-z0-9_])",
    re.IGNORECASE,
)
_JWT_TOKEN = re.compile(
    r"(?:^|[^A-Za-z0-9_])eyJ[A-Za-z0-9_-]+\."
    r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"
)
_UNSAFE_URI_SCHEME = re.compile(
    r"^(?:data|file|ftp|ftps|gs|https?|mailto|s3|ssh):",
    re.IGNORECASE,
)
_UNSAFE_URI_AT_OFFSET = re.compile(
    r"(?:https?|ftp|ftps|ssh)://|"
    r"file:(?:/{1,2}|\\)|"
    r"data:[^\s,;]{1,128}[,;]|"
    r"mailto:[^\s@]+@|"
    r"(?:^|[\s=\"'(<\[{,;:])(?:data|file|ftp|ftps|gs|https?|mailto|s3|ssh):",
    re.IGNORECASE,
)
_ABSOLUTE_PATH_AT_OFFSET = re.compile(
    r"(?:^|[\s=\"'(<\[{,;:])(?:"
    r"/(?!/)(?=[A-Za-z0-9._~-])|~[/\\]|"
    r"[A-Za-z]:[/\\]|[/\\]{2}[^/\\\s])"
)
_TRAVERSAL_AT_OFFSET = re.compile(
    r"(?:^|[\s=\"'(<\[{,;:/\\])\.\.?(?=$|[/\\])"
)
_ARTIFACT_DIGEST_SUFFIX = re.compile(rf"{_ARTIFACT_DIGEST}(?=$|\s)")
_CREDENTIAL_VALUE_LABEL = re.compile(
    r"(?:access[\s_-]*token|api[\s_-]*key|auth[\s_-]*token|"
    r"authorization|bearer|client[\s_-]*secret|credentials?|password|"
    r"passwd|private[\s_-]*key|secret(?:[\s_-]*key)?|token)",
    re.IGNORECASE,
)
_UNKNOWN_FINGERPRINT_SHA256 = hashlib.sha256(b"unknown").hexdigest()

LEGACY_RUNTIME_CONTROLLED_PATHS: tuple[tuple[str, ...], ...] = (
    ("runtime", "image_digest"),
    ("runtime", "gpu"),
    ("runtime", "gpu_fingerprint_sha256"),
    ("runtime", "gpu_fingerprint_supplied"),
    ("runtime", "cuda_version"),
    ("runtime", "driver_version"),
)

PLATFORM_RUNTIME_CONTROLLED_PATHS: tuple[tuple[str, ...], ...] = (
    *LEGACY_RUNTIME_CONTROLLED_PATHS,
    ("runtime", "accelerator_backend"),
    ("runtime", "accelerator"),
    ("runtime", "accelerator_fingerprint_sha256"),
    ("runtime", "accelerator_fingerprint_supplied"),
    ("runtime", "accelerator_runtime_version"),
    ("runtime", "host_os_version"),
    ("runtime", "software_environment_digest"),
)


def _contains_userinfo(value: str) -> bool:
    """Detect ``user:password@`` shapes in linear time.

    An OCI artifact's ``name:tag@algorithm:digest`` shape is the one explicit
    exception. The digest grammar is length-bounded, so checking a candidate
    suffix cannot turn a string with many ``@`` characters quadratic.
    """

    artifact_reference = SAFE_ARTIFACT_REFERENCE.fullmatch(value) is not None
    segment_start = 0
    candidate_colon: int | None = None
    for index, character in enumerate(value):
        if character == ":":
            candidate_colon = index if index > segment_start else None
            segment_start = index + 1
            continue
        if character == "@":
            if candidate_colon is not None and index > segment_start:
                candidate_value = value[segment_start:index]
                if _CREDENTIAL_VALUE_LABEL.fullmatch(candidate_value) is not None:
                    return True
                if (
                    not artifact_reference
                    or _ARTIFACT_DIGEST_SUFFIX.match(value, index + 1) is None
                ):
                    return True
            candidate_colon = None
            segment_start = index + 1
            continue
        if character.isspace() or character in "/\\":
            candidate_colon = None
            segment_start = index + 1
    return False


def _metadata_form_is_unsafe(value: str) -> bool:
    lowered = value.lower()
    if (
        "://" in lowered
        or "bearer " in lowered
        or "authorization:" in lowered
        or _CREDENTIAL_ASSIGNMENT.search(value)
        or _CREDENTIAL_TOKEN.search(value)
        or spaced_credentials(value)
        or _PRIVATE_KEY_BLOCK.search(value)
        or _DISCORD_TOKEN.search(value)
        or _contains_userinfo(value)
        or _JWT_TOKEN.search(value)
        or _UNSAFE_URI_AT_OFFSET.search(value)
        or _ABSOLUTE_PATH_AT_OFFSET.search(value)
        or _TRAVERSAL_AT_OFFSET.search(value)
    ):
        return True
    normalized_path = value.replace("\\", "/")
    return (
        value.startswith(("/", "~/", "\\"))
        or re.match(r"^[A-Za-z]:[\\/]", value) is not None
        or any(segment in {".", ".."} for segment in normalized_path.split("/"))
    )


def validated_public_metadata_size(
    value: object, *, max_length: int = 256
) -> int | None:
    """Return bounded string-work bytes when public metadata is safe.

    This is deliberately shape-based rather than an attempt to prove that an
    attestation is truthful.  The same boundary is used for generated and
    loaded reports so hand-edited evidence cannot bypass the CLI sanitizer.
    Charging the larger raw or NFKC form bounds aggregate normalization work.
    """

    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > max_length
    ):
        return None
    if any(
        ord(character) == 127
        or unicodedata.category(character) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
        for character in value
    ):
        return None
    normalized = unicodedata.normalize("NFKC", value)
    # Common compatibility forms (fullwidth text, ligatures, circled digits)
    # expand only a few code points. Reject pathological decompositions before
    # regex scanning so a small persisted string cannot consume outsized work.
    if (
        not normalized
        or normalized != normalized.strip()
        or len(normalized) > max_length
        or len(normalized) > len(value) * 4
    ):
        return None
    raw_bytes = len(value.encode("utf-8"))
    normalized_bytes = len(normalized.encode("utf-8"))
    if raw_bytes > max_length or normalized_bytes > max_length:
        return None
    if normalized.lower() == "unknown" and value != "unknown":
        return None
    for candidate in dict.fromkeys((value, normalized)):
        if _metadata_form_is_unsafe(candidate):
            return None
    return max(raw_bytes, normalized_bytes)


def is_safe_public_metadata(
    value: object, *, max_length: int = 256
) -> bool:
    """Return whether operator metadata is safe to persist and display."""

    return validated_public_metadata_size(value, max_length=max_length) is not None


def _public_text(value: object) -> bool:
    return (
        is_safe_public_metadata(value)
        and isinstance(value, str)
        and value.lower() != "unknown"
    )


def is_safe_artifact_reference(value: object) -> bool:
    """Validate a non-secret artifact label, optionally carrying a digest."""

    if not is_safe_public_metadata(value) or not isinstance(value, str):
        return False
    if value == "unknown":
        return True
    if _UNSAFE_URI_SCHEME.match(value):
        return False
    if value.startswith(("/", "~/", "\\")) or re.match(
        r"^[A-Za-z]:[\\/]", value
    ):
        return False
    if not SAFE_ARTIFACT_REFERENCE.fullmatch(value):
        return False
    label = value.split("@sha256:", 1)[0]
    if label.startswith("sha256:"):
        return True
    if any(segment in {".", ".."} for segment in label.split("/")):
        return False
    return True


def is_immutable_artifact_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and is_safe_artifact_reference(value)
        and bool(IMMUTABLE_ARTIFACT.fullmatch(value))
    )


def build_runtime_manifest(config: RunConfig) -> dict[str, Any]:
    """Build manifest 1.1 runtime metadata while retaining legacy CUDA keys."""

    fingerprint_sha256 = hashlib.sha256(
        config.gpu_fingerprint.encode("utf-8")
    ).hexdigest()
    accelerator_runtime_version = config.accelerator_runtime_version
    software_environment_digest = config.software_environment_digest
    if config.accelerator_backend == "cuda":
        if accelerator_runtime_version == "unknown":
            accelerator_runtime_version = config.cuda_version
        if software_environment_digest == "unknown":
            software_environment_digest = config.image_digest
    return {
        # Manifest 1.0 compatibility fields. They stay explicit so a new reader can
        # still distinguish a real CUDA pin from a platform where CUDA is absent.
        "image_digest": config.image_digest,
        "gpu": config.gpu,
        "gpu_fingerprint_sha256": fingerprint_sha256,
        "gpu_fingerprint_supplied": config.gpu_fingerprint != "unknown",
        "cuda_version": config.cuda_version,
        "driver_version": config.driver_version,
        # Manifest 1.1 platform-neutral fields.
        "accelerator_backend": config.accelerator_backend,
        "accelerator": config.gpu,
        "accelerator_fingerprint_sha256": fingerprint_sha256,
        "accelerator_fingerprint_supplied": config.gpu_fingerprint != "unknown",
        "accelerator_runtime_version": accelerator_runtime_version,
        "host_os_version": config.host_os_version,
        "software_environment_digest": software_environment_digest,
    }


def runtime_provenance_reasons(
    runtime: object, manifest_version: object
) -> list[str]:
    """Return fixed, sanitized decision-gate reasons for runtime provenance."""

    structural_reason = runtime_preflight_reason(runtime, manifest_version)
    if structural_reason is not None:
        return [structural_reason]
    assert isinstance(runtime, Mapping)
    if manifest_version == "1.0":
        reasons: list[str] = []
        if not is_immutable_artifact_digest(runtime.get("image_digest")):
            reasons.append("immutable_image_digest_required")
        return reasons

    backend = runtime.get("accelerator_backend")
    reasons = []
    if not is_immutable_artifact_digest(
        runtime.get("software_environment_digest")
    ):
        reasons.append("immutable_software_environment_digest_required")
    if backend == "cuda":
        if not is_immutable_artifact_digest(runtime.get("image_digest")):
            reasons.append("immutable_image_digest_required")
    return list(dict.fromkeys(reasons))


def runtime_preflight_reason(runtime: object, manifest_version: object) -> str | None:
    """Return a structural failure without applying decision-grade pin gates."""

    if not isinstance(runtime, Mapping):
        return "unverified_manifest_metadata"
    if manifest_version not in SUPPORTED_MANIFEST_VERSIONS:
        return "unsupported_runtime_manifest_version"

    legacy_text_fields = (
        "image_digest",
        "gpu",
        "cuda_version",
        "driver_version",
    )
    if any(
        field not in runtime
        or not is_safe_public_metadata(runtime.get(field))
        for field in legacy_text_fields
    ):
        return "unsafe_runtime_metadata"
    image_digest = runtime.get("image_digest")
    if image_digest != "unknown" and not is_safe_artifact_reference(image_digest):
        return "unsafe_runtime_metadata"

    legacy_fingerprint = runtime.get("gpu_fingerprint_sha256")
    if not isinstance(legacy_fingerprint, str) or not re.fullmatch(
        r"[0-9a-f]{64}", legacy_fingerprint
    ):
        return "invalid_manifest_digest"
    if manifest_version == "1.0":
        if not isinstance(runtime.get("gpu_fingerprint_supplied"), bool):
            return "invalid_runtime_fingerprint_marker"
        if any(not _public_text(runtime.get(field)) for field in legacy_text_fields):
            return "unverified_manifest_metadata"
        if (
            runtime.get("gpu_fingerprint_supplied") is not True
            or legacy_fingerprint == _UNKNOWN_FINGERPRINT_SHA256
        ):
            return "gpu_fingerprint_not_supplied"
        return None

    platform_text_fields = (
        "accelerator_backend",
        "accelerator",
        "accelerator_runtime_version",
        "host_os_version",
        "software_environment_digest",
    )
    if any(
        field not in runtime
        or not is_safe_public_metadata(runtime.get(field))
        for field in platform_text_fields
    ):
        return "unsafe_runtime_metadata"
    backend = runtime.get("accelerator_backend")
    if backend not in ACCELERATOR_BACKENDS:
        return "unverified_manifest_metadata"
    required = ("accelerator", "accelerator_runtime_version")
    if backend != "cuda":
        required = (*required, "host_os_version")
    if any(not _public_text(runtime.get(field)) for field in required):
        return "unverified_manifest_metadata"

    environment_digest = runtime.get("software_environment_digest")
    if environment_digest != "unknown" and not is_safe_artifact_reference(
        environment_digest
    ):
        return "unsafe_runtime_metadata"
    accelerator_fingerprint = runtime.get("accelerator_fingerprint_sha256")
    if not isinstance(accelerator_fingerprint, str) or not re.fullmatch(
        r"[0-9a-f]{64}", accelerator_fingerprint
    ):
        return "invalid_manifest_digest"
    if not isinstance(runtime.get("gpu_fingerprint_supplied"), bool) or not isinstance(
        runtime.get("accelerator_fingerprint_supplied"), bool
    ):
        return "runtime_aliases_do_not_reconcile"
    if (
        runtime.get("accelerator") != runtime.get("gpu")
        or accelerator_fingerprint != legacy_fingerprint
        or runtime.get("accelerator_fingerprint_supplied")
        is not runtime.get("gpu_fingerprint_supplied")
    ):
        return "runtime_aliases_do_not_reconcile"
    if (
        runtime.get("accelerator_fingerprint_supplied") is not True
        or accelerator_fingerprint == _UNKNOWN_FINGERPRINT_SHA256
    ):
        return "accelerator_fingerprint_not_supplied"
    if backend == "cuda" and any(
        not _public_text(runtime.get(field))
        for field in ("image_digest", "cuda_version", "driver_version")
    ):
        return "unverified_manifest_metadata"
    return None
