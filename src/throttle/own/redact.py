"""Pattern-based redaction for captured API bodies.

What this catches: email addresses, phone numbers written with separators,
US social security numbers (dashed), payment card numbers (Luhn-checked, with
a known card prefix and length), IPv4/IPv6 addresses, and secrets (API keys,
tokens, JWTs, ``password: ...`` style assignments, and string values under
keys such as ``api_key`` or ``password``).

What this does NOT catch: names, street addresses, dates of birth, free-text
medical or financial details, or anything else that has no fixed shape. It is
pattern matching, not understanding. Operators who cannot keep that content
on disk should capture with ``metadata`` mode, which stores no bodies at all.

Modes (``records.REDACTION_MODES``):

- ``off``       bodies kept as-is
- ``mask``      each match becomes a typed placeholder, e.g. ``<EMAIL>``
- ``hash``      like mask, plus a salted hash: ``<EMAIL:3f2a9c>``; the same
                value maps to the same placeholder for the same salt
- ``metadata``  bodies dropped entirely

Only string values are rewritten. Dict keys, numbers, booleans and the
``model`` field are never touched, so the JSON structure survives.

In ``mask`` and ``hash`` mode, inline base64 payloads (``data:`` URIs such as
OpenAI ``image_url`` data URLs, and Anthropic ``source`` blocks of type
``base64``) are replaced with ``<IMAGE>`` (or ``<AUDIO>`` / ``<DATA>``) before
any pattern runs: they are not text, and scanning megabytes of base64 is slow.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import re
from typing import Any, Callable

from ..provenance import _CREDENTIAL_ASSIGNMENT, _CREDENTIAL_TOKEN, _CREDENTIAL_VALUE_LABEL, _JWT_TOKEN
from .records import REDACTION_HASH, REDACTION_MASK, REDACTION_METADATA, REDACTION_MODES, REDACTION_OFF

EMAIL = "EMAIL"
PHONE = "PHONE"
SSN = "SSN"
CARD = "CARD"
IP = "IP"
SECRET = "SECRET"

# Higher wins when two matches overlap.
_PRIORITY = {SECRET: 6, EMAIL: 5, CARD: 4, SSN: 3, PHONE: 2, IP: 1}
_NEVER_REDACTED_KEYS = frozenset({"model"})

_EMAIL = re.compile(
    r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9][A-Za-z0-9._%+-]{0,63}@"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,24}(?![A-Za-z0-9-])"
)
# Separators are required: a bare 10-digit run is far more often an order or
# account number than a phone number. NANP area and exchange codes start 2-9.
# A "/" next to a phone number is a list separator ("415-555-0134/415-555-0199",
# "0134/work"); only a "/" right after a letter or punctuation (a URL or file
# path segment) blocks a match.
_PHONE_NANP = re.compile(
    r"(?<![\w+\-.$#])(?<![^\d\s]/)(?:\+?1[\s.-]?)?(?:\([2-9]\d{2}\)\s?|[2-9]\d{2}[\s.-])[2-9]\d{2}[\s.-]\d{4}(?![\w\-]|\.\d)"
)
_PHONE_INTL = re.compile(r"(?<![\w+])(?<![^\d\s]/)\+(?:[1-9]\d{0,2})(?:[\s.-]?\(?\d{1,4}\)?){2,5}(?![\w\-]|\.\d)")
_SSN = re.compile(r"(?<![\w\-./])(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?![\w\-/]|\.\d)")
_CARD = re.compile(
    r"(?<![\w\-./$#])(?:"
    r"\d{13,19}|"
    r"\d{4}([ -])\d{4}\1\d{4}\1\d{1,7}|"
    r"\d{4}([ -])\d{6}\2\d{4,5}"
    r")(?![\w\-/]|\.\d)"
)
_IPV4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w]|\.\d)")
_IPV6_CANDIDATE = re.compile(r"(?<![\w:.])[0-9A-Fa-f]{0,4}(?::[0-9A-Fa-f]{0,4}){2,7}(?![\w:])")
_SECRET_VALUE = re.compile(r"""\s*["']?(?:bearer\s+)?([^\s"',;}\])]+)""", re.IGNORECASE)
_PASSWORD_LABEL = re.compile(r"pass(?:word|wd)|secret|private", re.IGNORECASE)
_JSON_SECRET_PAIR = re.compile(
    r'"(' + _CREDENTIAL_VALUE_LABEL.pattern + r')"\s*:\s*"((?:[^"\\]|\\.)+)"', re.IGNORECASE
)
# A bare "token" key is a logprobs entry far more often than a credential.
_AMBIGUOUS_KEYS = frozenset({"token"})

Span = tuple[int, int, str, str]  # start, end, kind, normalized value

_DATA_URI = re.compile(r"^\s*data:([\w.+-]+/[\w.+-]+)?[^,]*;base64,", re.IGNORECASE)


def _payload_placeholder(media_type: Any) -> str:
    kind = media_type.lower() if isinstance(media_type, str) else ""
    if not kind or kind.startswith("image/"):
        return "<IMAGE>"
    return "<AUDIO>" if kind.startswith("audio/") else "<DATA>"


def strip_inline_payloads(value: Any) -> Any:
    """Replace base64 payloads with placeholders, leaving the structure.

    Catches ``data:<type>;base64,...`` strings anywhere, Anthropic
    ``{"type": "base64", "media_type": ..., "data": ...}`` sources and OpenAI
    ``input_audio`` ``data``. Returns a new structure; the input is unchanged.
    """

    if isinstance(value, dict):
        if value.get("type") == "base64" and isinstance(value.get("data"), str):
            out = dict(value)
            out["data"] = _payload_placeholder(value.get("media_type"))
            return out
        out = {}
        for k, v in value.items():
            if k == "input_audio" and isinstance(v, dict) and isinstance(v.get("data"), str):
                out[k] = {**v, "data": "<AUDIO>"}
            else:
                out[k] = strip_inline_payloads(v)
        return out
    if isinstance(value, list):
        return [strip_inline_payloads(v) for v in value]
    if isinstance(value, str):
        match = _DATA_URI.match(value[:256])
        if match:
            return _payload_placeholder(match.group(1) or "image/")
    return value


def luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = ord(ch) - 48
        if i % 2:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _card_brand_ok(digits: str) -> bool:
    n = len(digits)
    if digits[0] == "4":
        return n in (13, 16, 19)
    if digits[:2] in {"34", "37"}:
        return n == 15
    if 51 <= int(digits[:2]) <= 55 or 2221 <= int(digits[:4]) <= 2720:
        return n == 16
    if digits[:4] == "6011" or digits[:2] == "65" or 644 <= int(digits[:3]) <= 649:
        return 16 <= n <= 19
    if digits[:2] == "62":
        return 16 <= n <= 19
    if digits[:2] in {"36", "38"} or 300 <= int(digits[:3]) <= 305:
        return 14 <= n <= 19
    if 3528 <= int(digits[:4]) <= 3589:
        return 16 <= n <= 19
    return False


def _digits(value: str) -> str:
    return re.sub(r"\D", "", value)


def _find_cards(text: str) -> list[Span]:
    out: list[Span] = []
    for m in _CARD.finditer(text):
        digits = _digits(m.group(0))
        if 13 <= len(digits) <= 19 and _card_brand_ok(digits) and luhn_ok(digits):
            out.append((m.start(), m.end(), CARD, digits))
    return out


def _find_phones(text: str) -> list[Span]:
    out: list[Span] = []
    for pattern in (_PHONE_NANP, _PHONE_INTL):
        for m in pattern.finditer(text):
            raw = m.group(0).rstrip()
            digits = _digits(raw)
            if not 10 <= len(digits) <= 15:
                continue
            out.append((m.start(), m.start() + len(raw), PHONE, digits[-10:] if pattern is _PHONE_NANP else digits))
    return out


def _find_ips(text: str) -> list[Span]:
    out: list[Span] = []
    for m in _IPV4.finditer(text):
        parts = m.group(0).split(".")
        if all(int(p) <= 255 and (p == "0" or not p.startswith("0")) for p in parts):
            out.append((m.start(), m.end(), IP, m.group(0)))
    for m in _IPV6_CANDIDATE.finditer(text):
        value = m.group(0)
        groups = [g for g in value.split(":") if g]
        compressed = "::" in value
        if (compressed and len(groups) < 3) or (not compressed and len(groups) != 8):
            continue
        try:
            ipaddress.IPv6Address(value)
        except ValueError:
            continue
        out.append((m.start(), m.end(), IP, value.lower()))
    return out


def _token_start(match: re.Match[str]) -> int:
    start = match.start()
    while start < match.end() and not match.string[start].isalnum():
        start += 1
    return start


def _find_secrets(text: str) -> list[Span]:
    out: list[Span] = []
    for pattern in (_CREDENTIAL_TOKEN, _JWT_TOKEN):
        for m in pattern.finditer(text):
            start = _token_start(m)
            out.append((start, m.end(), SECRET, text[start:m.end()]))
    for m in _CREDENTIAL_ASSIGNMENT.finditer(text):
        value = _SECRET_VALUE.match(text, m.end())
        if not value:
            continue
        secret = value.group(1)
        password_like = bool(_PASSWORD_LABEL.search(m.group(0)))
        if secret.startswith("<") or len(secret) < (4 if password_like else 8):
            continue
        if not password_like and not (re.search(r"\d", secret) and re.search(r"[A-Za-z]", secret)):
            continue
        out.append((value.start(1), value.end(1), SECRET, secret))
    for m in _JSON_SECRET_PAIR.finditer(text):
        if m.group(1).lower() not in _AMBIGUOUS_KEYS:
            out.append((m.start(2), m.end(2), SECRET, m.group(2)))
    return out


def find_pii(text: str) -> list[Span]:
    """Non-overlapping matches, left to right."""
    candidates: list[Span] = []
    candidates += _find_secrets(text)
    candidates += [(m.start(), m.end(), EMAIL, m.group(0).lower()) for m in _EMAIL.finditer(text)]
    candidates += _find_cards(text)
    candidates += [(m.start(), m.end(), SSN, _digits(m.group(0))) for m in _SSN.finditer(text)]
    candidates += _find_phones(text)
    candidates += _find_ips(text)
    candidates.sort(key=lambda s: (s[0], -_PRIORITY[s[2]], -(s[1] - s[0])))
    chosen: list[Span] = []
    for span in candidates:
        if chosen and span[0] < chosen[-1][1]:
            prev = chosen[-1]
            if _PRIORITY[span[2]] > _PRIORITY[prev[2]]:
                chosen[-1] = span
            continue
        chosen.append(span)
    return chosen


def _salt_bytes(salt: str | bytes | None) -> bytes:
    if salt is None:
        raise ValueError("hash redaction needs a salt")
    return salt.encode("utf-8") if isinstance(salt, str) else bytes(salt)


def _placeholder(mode: str, salt: bytes | None) -> Callable[[str, str], str]:
    if mode == REDACTION_HASH:
        assert salt is not None

        def hashed(kind: str, value: str) -> str:
            digest = hmac.new(salt, f"{kind}:{value}".encode("utf-8"), hashlib.sha256).hexdigest()
            return f"<{kind}:{digest[:6]}>"

        return hashed
    return lambda kind, _value: f"<{kind}>"


def _check_mode(mode: str) -> None:
    if mode not in REDACTION_MODES:
        raise ValueError(f"unknown redaction mode {mode!r}; expected one of {', '.join(REDACTION_MODES)}")


def _redact(text: str, label: Callable[[str, str], str]) -> str:
    spans = find_pii(text)
    if not spans:
        return text
    pieces: list[str] = []
    cursor = 0
    for start, end, kind, value in spans:
        pieces.append(text[cursor:start])
        pieces.append(label(kind, value))
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def redact_text(text: str, mode: str, salt: str | bytes | None = None) -> str:
    """Redact one string. ``metadata`` returns an empty string."""
    _check_mode(mode)
    if mode == REDACTION_OFF:
        return text
    if mode == REDACTION_METADATA:
        return ""
    salt_bytes = _salt_bytes(salt) if mode == REDACTION_HASH else None
    return _redact(text, _placeholder(mode, salt_bytes))


def redact_body(obj: Any, mode: str, salt: str | bytes | None = None) -> Any:
    """Redact every string value in a JSON-like structure.

    Keys and non-string values are kept. ``off`` returns ``obj`` itself;
    ``metadata`` returns ``None`` (no body is kept at all).
    """
    _check_mode(mode)
    if mode == REDACTION_OFF:
        return obj
    if mode == REDACTION_METADATA:
        return None
    salt_bytes = _salt_bytes(salt) if mode == REDACTION_HASH else None
    label = _placeholder(mode, salt_bytes)
    obj = strip_inline_payloads(obj)

    def walk(value: Any, key: str | None) -> Any:
        if isinstance(value, dict):
            return {k: walk(v, k if isinstance(k, str) else None) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v, None) for v in value]
        if isinstance(value, str):
            if key in _NEVER_REDACTED_KEYS:
                return value
            if key and value and key.lower() not in _AMBIGUOUS_KEYS and _CREDENTIAL_VALUE_LABEL.fullmatch(key):
                return label(SECRET, value)
            return _redact(value, label)
        return value

    return walk(obj, None)


__all__ = [
    "CARD", "EMAIL", "IP", "PHONE", "SECRET", "SSN",
    "REDACTION_HASH", "REDACTION_MASK", "REDACTION_METADATA", "REDACTION_OFF",
    "find_pii", "luhn_ok", "redact_body", "redact_text", "strip_inline_payloads",
]
