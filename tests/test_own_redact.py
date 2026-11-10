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


# ---------------------------------------------------------------------------
# Security audit probes: each must be masked in mask mode.
# ---------------------------------------------------------------------------

AWS_ID = "AKIAIOSFODNN7EXAMPLE"
AWS_SECRET = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
HEX64 = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"

AUDIT_PROBES = [
    # PEM private key blocks, multiline, with real or escaped newlines, or cut off
    ("key:\n-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA1234abcd\nabcdEFGH5678==\n-----END RSA PRIVATE KEY-----\nthanks",
     "key:\n<SECRET>\nthanks"),
    ("-----BEGIN PRIVATE KEY-----\\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASC\\n-----END PRIVATE KEY-----",
     "<SECRET>"),
    ("ssh -----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAABG5vbmU=\n-----END OPENSSH PRIVATE KEY----- done",
     "ssh <SECRET> done"),
    ("-----BEGIN EC PRIVATE KEY-----\nMHcCAQEEIBkg4LVWM9nuwNSk3yByxZpYRTBnVJk5", "<SECRET>"),
    ("-----BEGIN PGP PRIVATE KEY BLOCK-----\nlQOYBF0abc123\n-----END PGP PRIVATE KEY BLOCK-----", "<SECRET>"),
    # AWS secret access keys next to an access key id or a secret label
    (f"creds {AWS_ID} {AWS_SECRET}", "creds <SECRET> <SECRET>"),
    (f"id={AWS_ID}, secret={AWS_SECRET}", "id=<SECRET>, secret=<SECRET>"),
    (f"aws_secret_access_key = {AWS_SECRET}", "aws_secret_access_key = <SECRET>"),
    (f"AWS_SECRET_ACCESS_KEY={AWS_SECRET}", "AWS_SECRET_ACCESS_KEY=<SECRET>"),
    (f"aws secret access key {AWS_SECRET}", "aws secret access key <SECRET>"),
    # Slack, Discord and Stripe webhooks and tokens
    ("post to https://hooks.slack.com/" "services/T0AB12CD3/B0AB12CD3/abcdEFGHijklMNOPqrstUVWX now",
     "post to <SECRET> now"),
    ("https://hooks.slack.com/workflows/T0AB12CD3/A0AB12CD3/123456789/abcDEF", "<SECRET>"),
    ("slack xo" "xb-123456789012-1234567890123-abcdefghijklmnopqrstuvwx", "slack <SECRET>"),
    ("slack xoxe." "xoxp-1-Mi0yLTEyMzQ1Njc4OTAxMi0xMjM0", "slack <SECRET>"),
    ("app token xapp-1-A0123456789-1234567890123-abcdef0123456789", "app token <SECRET>"),
    ("hook https://discord.com/api/webhooks/123456789012345678/abcDEF_ghi-JKLmno0123456789",
     "hook <SECRET>"),
    ("hook https://canary.discordapp.com/api/webhooks/123456789012345678/abcDEF_ghi", "hook <SECRET>"),
    ("bot MTIzNDU2Nzg5MDEyMzQ1Njc4" ".GabcDE.abcdefghijklmnopqrstuvwxyz0123456789 ok", "bot <SECRET> ok"),
    ("stripe sk_" "live_51H8abcdefghijklmnopqrstu", "stripe <SECRET>"),
    ("stripe rk_" "live_51H8abcdefghijklmnopqrstu", "stripe <SECRET>"),
    ("stripe sk_test_51H8abcdefghijklmnopqrstu", "stripe <SECRET>"),
    ("webhook whsec_abcdefghijklmnop1234567890ABCD", "webhook <SECRET>"),
    # URL userinfo: the whole user:pass@ goes
    ("db postgres://admin:hunter22@db.internal:5432/app", "db postgres://<SECRET>db.internal:5432/app"),
    ("clone https://x-access-token:abc123@github.com/acme/repo.git", "clone https://<SECRET>github.com/acme/repo.git"),
    ("redis://:s3cretpass@cache.local:6379/0", "redis://<SECRET>cache.local:6379/0"),
    ("amqp://guest:guest@rabbit/", "amqp://<SECRET>rabbit/"),
    # 32+ hex or base64 runs next to key, token or secret
    (f"the key {HEX64} works", "the key <SECRET> works"),
    ("token QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo0MjQy end", "token <SECRET> end"),
    (f"{HEX64} is my secret", "<SECRET> is my secret"),
    ("Secret is 0123456789abcdef0123456789abcdef", "Secret is <SECRET>"),
    # bare labels
    ("OPENAI_KEY: abc123def456ghi789", "OPENAI_KEY: <SECRET>"),
    ("export ANTHROPIC_API_KEY=abc123def456ghi789", "export ANTHROPIC_API_KEY=<SECRET>"),
    ("api-key q1w2e3r4t5y6u7i8", "api-key <SECRET>"),
    ("api key q1w2e3r4t5y6u7i8", "api key <SECRET>"),
    ("OPENAI_KEY abc123def456ghi789", "OPENAI_KEY <SECRET>"),
    ("DB_PASS=hunter2", "DB_PASS=<SECRET>"),
    ("GH_TOKEN: 0a1b2c3d4e5f6a7b", "GH_TOKEN: <SECRET>"),
    ("Authorization: Bearer abc123def456ghi", "Authorization: Bearer <SECRET>"),
]


@pytest.mark.parametrize("text, expected", AUDIT_PROBES)
def test_security_audit_probes_are_masked(text, expected):
    assert redact_text(text, r.REDACTION_MASK) == expected, find_pii(text)


@pytest.mark.parametrize("text, _expected", AUDIT_PROBES)
def test_security_audit_probes_are_hashed_too(text, _expected):
    out = redact_text(text, r.REDACTION_HASH, SALT)
    assert "<SECRET:" in out


AUDIT_NEAR_MISSES = [
    "the api key is required and the key idea is simple",
    "first-to-last output-bearing SSE delta divided by completion-token gaps",
    "api key for production",
    "see https://example.com/docs and https://hooks.example.com/services/x",
    "the token count was 1234 and the secret sauce is love",
    "cache_key: user_profile and primary_key id",
    "git sha 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b in main",
    "https://github.com/acme/repo and mailto-like a@b",
]


@pytest.mark.parametrize("text", AUDIT_NEAR_MISSES)
def test_security_audit_near_misses_survive(text):
    assert redact_text(text, r.REDACTION_MASK) == text, find_pii(text)


def test_json_key_names_are_matched_anywhere_in_the_key():
    body = {
        "OPENAI_API_KEY": "abc", "STRIPE_SECRET": "short", "GH_TOKEN": "x", "client_secret": "y",
        "githubToken": "z", "x-api-key": "k", "accessToken": "t", "slack_webhook_url": "https://hooks.slack.com/x",
        "openai_key": "abcdef1234567890", "stripeKey": "abcdef1234567890",
        # left alone
        "primary_key": "user_id", "max_tokens_label": "short", "tokenizer": "cl100k_base", "token": "Hello",
        "keyword": "summary", "monkey": "banana",
    }
    out = redact_body(body, r.REDACTION_MASK)
    for key in ("OPENAI_API_KEY", "STRIPE_SECRET", "GH_TOKEN", "client_secret", "githubToken", "x-api-key",
                "accessToken", "slack_webhook_url", "openai_key", "stripeKey"):
        assert out[key] == "<SECRET>", key
    for key in ("primary_key", "max_tokens_label", "tokenizer", "token", "keyword", "monkey"):
        assert out[key] == body[key], key


def test_redaction_stays_fast_on_long_snake_case_text():
    import time

    text = "a_" * 100_000 + "b-c " * 20_000 + "key " * 10_000
    started = time.perf_counter()
    redact_text(text, r.REDACTION_MASK)
    assert time.perf_counter() - started < 5


def test_docstring_lists_what_is_caught():
    from throttle.own import redact

    doc = redact.__doc__ or ""
    for words in ("private key blocks", "AWS secret access key", "Slack and Discord incoming-webhook URLs",
                  "user:password@", "32+ character hex or base64 run", "OPENAI_KEY", "api-key <value>",
                  "matched\n  anywhere in the key", "whsec_", "Discord bot tokens"):
        assert words in doc, words


def test_provenance_metadata_rejects_the_new_secret_shapes():
    from throttle.provenance import is_safe_public_metadata

    for bad in ("-----BEGIN PRIVATE KEY-----", "api-key q1w2e3r4t5y6u7i8",
                "MTIzNDU2Nzg5MDEyMzQ1Njc4" ".GabcDE.abcdefghijklmnopqrstuvwxyz0123456789", "OPENAI_KEY: abc123def456",
                "whsec_abcdefghijklmnop1234567890ABCD"):
        assert not is_safe_public_metadata(bad), bad
    assert is_safe_public_metadata("divided by completion-token gaps")
