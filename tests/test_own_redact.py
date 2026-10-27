import json

import pytest

from throttle.own import records as r
from throttle.own.redact import find_pii, luhn_ok, redact_body, redact_text

SALT = "test-salt"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("write to kush@example.com today", "write to <EMAIL> today"),
        ("Jane.Doe+billing@mail.co.uk.", "<EMAIL>."),
        ("call (805) 555-0134", "call <PHONE>"),
        ("call 805-555-0134 or 805.555.0134", "call <PHONE> or <PHONE>"),
        ("call +1 805 555 0134", "call <PHONE>"),
        ("london +44 20 7946 0958", "london <PHONE>"),
        ("ssn 123-45-6789", "ssn <SSN>"),
        ("visa 4111 1111 1111 1111", "visa <CARD>"),
        ("visa 4111-1111-1111-1111", "visa <CARD>"),
        ("visa 4111111111111111", "visa <CARD>"),
        ("amex 3782 822463 10005", "amex <CARD>"),
        ("mc 5555555555554444", "mc <CARD>"),
        ("from 192.168.1.20 ok", "from <IP> ok"),
        ("v6 2001:db8::8a2e:370:7334", "v6 <IP>"),
        ("key sk-proj-abcdefghijklmnop end", "key <SECRET> end"),
        ("gh token ghp_abcdefghijklmnopqrstuvwx", "gh token <SECRET>"),
        ("aws AKIAIOSFODNN7EXAMPLE", "aws <SECRET>"),
        ("jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abc123", "jwt <SECRET>"),
        ("password: hunter2", "password: <SECRET>"),
        ("Authorization: Bearer abcdef123456", "Authorization: Bearer <SECRET>"),
        ('{"password": "hunter22"}', '{"password": "<SECRET>"}'),
    ],
)
def test_mask_catches_each_type(text, expected):
    assert redact_text(text, r.REDACTION_MASK) == expected


# Things that look a bit like PII but are not. All must survive mask mode.
NEAR_MISSES = [
    "Order #1234567890123456 shipped",
    "order 4111111111111112 is not a card (fails Luhn)",
    "order number 1234567890123 and ORD-2026-000123",
    "invoice INV-555-123-4567",
    "account id 8055550134",
    "total $1,234.56 and €4,111.11 and 4111.11 USD",
    "on 2026-10-27, 10/27/2026, 27.10.2026 and 2026/10/27",
    "at 12:30:45 or 23:59",
    "released v0.7.0, 1.2.3, version 10.0.19041.1 and 2.0.0-rc.1",
    "python 3.11.4 and node 20.11.1",
    "UPS 1Z999AA10123456784",
    "uuid 550e8400-e29b-41d4-a716-446655440000",
    "sha 3f2a9c4b5d6e7f80 and sha256:9b74c9897bac770ffc029102a200c5de",
    "C++ std::vector and a::b::c and ns::fn",
    "isbn 978-3-16-148410-0",
    "max_tokens: 512 and token = tokenizer.encode(text)",
    "tokens: 1500, token: 42",
    "zip 93106-1234 and ext. 555-0134",
    "the ratio 1:2:3 and score 98.6",
    "999-45-6789 and 000-12-3456 are not valid SSNs",
    "user@localhost has no tld",
    "range 300.1.1.1 is not an ip",
    "phone 123-456-7890 has an invalid area code",
]


@pytest.mark.parametrize("text", NEAR_MISSES)
def test_near_misses_survive(text):
    assert redact_text(text, r.REDACTION_MASK) == text, find_pii(text)


def test_hash_is_stable_and_salted():
    a = redact_text("mail kush@example.com and KUSH@example.com", r.REDACTION_HASH, SALT)
    first, second = [p for p in a.split() if p.startswith("<EMAIL:")]
    assert first == second and len(first) == len("<EMAIL:abcdef>")
    other_salt = redact_text("kush@example.com", r.REDACTION_HASH, "other")
    assert other_salt != first
    different = redact_text("someone@example.com", r.REDACTION_HASH, SALT)
    assert different != first
    # Same phone written two ways hashes the same.
    p1 = redact_text("805-555-0134", r.REDACTION_HASH, SALT)
    p2 = redact_text("(805) 555-0134", r.REDACTION_HASH, SALT)
    assert p1 == p2 and p1.startswith("<PHONE:")


def test_hash_needs_salt_and_mode_is_checked():
    with pytest.raises(ValueError):
        redact_text("x", r.REDACTION_HASH, None)
    with pytest.raises(ValueError):
        redact_text("x", "scramble")


def test_off_and_metadata():
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "kush@example.com"}]}
    assert redact_body(body, r.REDACTION_OFF) is body
    assert redact_text("kush@example.com", r.REDACTION_OFF) == "kush@example.com"
    assert redact_body(body, r.REDACTION_METADATA) is None
    assert redact_text("kush@example.com", r.REDACTION_METADATA) == ""


def test_body_redacts_values_never_keys_or_model():
    body = {
        "model": "ft:gpt-4o:kush@example.com:abc",
        "kush@example.com": "key stays",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "I am kush@example.com, card 4111 1111 1111 1111"}]},
        ],
        "tools": [{"type": "function", "function": {"name": "login", "parameters": {
            "type": "object", "properties": {"password": {"type": "string"}}}}}],
        "metadata": {"api_key": "plainvalue", "token": "Hello", "temperature": 0.2, "n": 4111111111111111},
        "max_tokens": 64,
        "stream": True,
    }
    out = redact_body(body, r.REDACTION_MASK)
    assert out["model"] == body["model"]
    assert "kush@example.com" in out and out["kush@example.com"] == "key stays"
    assert out["messages"][0]["content"][0]["text"] == "I am <EMAIL>, card <CARD>"
    assert out["tools"] == body["tools"]
    assert out["metadata"] == {"api_key": "<SECRET>", "token": "Hello", "temperature": 0.2, "n": 4111111111111111}
    assert out["max_tokens"] == 64 and out["stream"] is True
    # Input not mutated.
    assert body["messages"][0]["content"][0]["text"].startswith("I am kush@")


def test_tool_call_argument_strings_are_redacted():
    args = json.dumps({"email": "kush@example.com", "password": "hunter22", "order": "1234567890"})
    out = redact_text(args, r.REDACTION_MASK)
    assert json.loads(out) == {"email": "<EMAIL>", "password": "<SECRET>", "order": "1234567890"}


def test_luhn():
    assert luhn_ok("4111111111111111")
    assert not luhn_ok("4111111111111112")


def test_overlaps_prefer_the_stronger_type():
    # The secret regex and the phone regex could both fire inside a key.
    assert redact_text("sk-live-805-555-0134-abcdef", r.REDACTION_MASK) == "<SECRET>"


# ---------------------------------------------------------------------------
# Review fixes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text, expected", [
    ("Phone/Fax: 415-555-0134/415-555-0199", "Phone/Fax: <PHONE>/<PHONE>"),
    ("home 415-555-0134/work 415-555-0199", "home <PHONE>/work <PHONE>"),
    ("tel 415-555-0134 / 415-555-0199", "tel <PHONE> / <PHONE>"),
    ("intl +44 20 7946 0958/+44 20 7946 0959", "intl <PHONE>/<PHONE>"),
])
def test_phone_numbers_next_to_a_slash(text, expected):
    assert redact_text(text, r.REDACTION_MASK) == expected


SLASH_NEAR_MISSES = [
    "see https://example.com/users/415-555-0134",
    "path a/b/415-555-0134 and ./415-555-0134",
    "file docs/415-555-0134-notes.txt",
    "on 10/27/2026 and 2026/10/27 and 1/2/3",
    "ratio 415/555/0134",
]


@pytest.mark.parametrize("text", SLASH_NEAR_MISSES)
def test_slash_near_misses_survive(text):
    assert redact_text(text, r.REDACTION_MASK) == text, find_pii(text)


def test_inline_payloads_replaced_before_redaction():
    from throttle.own.redact import strip_inline_payloads

    body = {"messages": [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ" + "B" * 100}},
            {"type": "input_audio", "input_audio": {"data": "UklGR" + "C" * 100, "format": "wav"}},
        ]},
        {"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBOR" + "D" * 100}},
            {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBER"}},
            {"type": "image", "source": {"type": "url", "url": "https://example.com/cat.png"}},
        ]},
    ]}
    for mode in (r.REDACTION_MASK, r.REDACTION_HASH):
        out = redact_body(body, mode, SALT)
        first, second = out["messages"][0]["content"], out["messages"][1]["content"]
        assert first[0]["image_url"]["url"] == "<IMAGE>"
        assert first[1]["input_audio"] == {"data": "<AUDIO>", "format": "wav"}
        assert second[0]["source"] == {"type": "base64", "media_type": "image/png", "data": "<IMAGE>"}
        assert second[1]["source"]["data"] == "<DATA>"
        assert second[2]["source"]["url"] == "https://example.com/cat.png"
    assert redact_body(body, r.REDACTION_OFF) is body
    assert strip_inline_payloads(body)["messages"][0]["content"][0]["image_url"]["url"] == "<IMAGE>"
    assert body["messages"][0]["content"][0]["image_url"]["url"].startswith("data:")   # input untouched
