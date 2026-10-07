"""Tests for `throttle upgrade` (the Throttle Pilot), its offline QR code and the once-a-day nudge.

The reference matrices in fixtures/upgrade_qr_matrices.json were produced by
the vendored encoder and decoded back to the exact URLs with an independent
decoder (zxing-cpp) when they were recorded (the pilot URL on 2026-10-06); these tests pin them so the
encoder, its settings and the renderers can't drift silently.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from throttle import upgrade
from throttle.cli import main

REFERENCE = json.loads((Path(__file__).parent / "fixtures" / "upgrade_qr_matrices.json").read_text())


def as_text(matrix: list[list[bool]]) -> list[str]:
    return ["".join("#" if dark else "." for dark in row) for row in matrix]


@pytest.mark.parametrize("url", [upgrade.PILOT_URL])
def test_qr_matrix_matches_decoded_reference(url: str) -> None:
    assert url in REFERENCE
    assert as_text(upgrade.qr_matrix(url)) == REFERENCE[url]


@pytest.mark.parametrize("url", [upgrade.PILOT_URL])
def test_terminal_rendering_round_trips_to_the_matrix(url: str) -> None:
    quiet = 2
    lines = upgrade.qr_terminal(url, quiet=quiet).split("\n")
    light: list[list[bool]] = []
    for line in lines:
        light.append([c in "█▀" for c in line])
        light.append([c in "█▄" for c in line])
    size = len(REFERENCE[url])
    total = size + 2 * quiet
    assert all(len(line) == total for line in lines)
    # Quiet zone is light on every side.
    assert all(light[y][x] for y in range(quiet) for x in range(total))
    assert all(light[y][x] for y in range(total) for x in list(range(quiet)) + list(range(total - quiet, total)))
    decoded = [
        "".join("." if light[y + quiet][x + quiet] else "#" for x in range(size))
        for y in range(size)
    ]
    assert decoded == REFERENCE[url]


def test_svg_draws_exactly_the_dark_modules() -> None:
    svg = upgrade.qr_svg(upgrade.PILOT_URL)
    dark = sum(row.count("#") for row in REFERENCE[upgrade.PILOT_URL])
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert svg.count("h1v1h-1z") == dark
    assert 'fill="#fff"' in svg


PILOT_TERMS = (
    "Throttle Pilot",
    "We find savings on your agent workload and prove them.",
    "a small group of teams running agents on their own GPUs",
    "Free for 3 months",
    "find the config changes that lower your cost per token, and prove each one.",
    "We'd love to help you with your results.",
    "pricing is based on the savings we verify together",
    "conservative end of the 95% interval",
    "https://throttle-pro.com",
    "kushthrottle@gmail.com",
    '"Throttle pilot"',
)
RETIRED = ("$19", "$50/", "$50 ", "Cost Audit", "early access", "Early access", "No savings guarantee",
           "no savings guarantee", "formspree", "Throttle Pro", "one-time",
           "2 weeks", "2-week", "20%", "$500", "floor", "design partners")


def _flat(out: str) -> str:
    return " ".join(" ".join(line.strip().strip("│|").split()) for line in out.splitlines())


def test_upgrade_prints_pilot_terms_link_and_qr(capsys, monkeypatch) -> None:
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    assert main(["upgrade"]) == 0
    out = capsys.readouterr().out
    for term in PILOT_TERMS:
        assert term in _flat(out), term
    for retired in RETIRED:
        assert retired not in out, retired
    assert "Open: https://throttle-pro.com" in out
    assert "▀" in out or "▄" in out
    assert "\x1b[" not in out  # not a TTY, no FORCE_COLOR: no colour


def test_audit_is_a_hidden_alias_for_the_same_pilot_page(capsys, monkeypatch) -> None:
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    assert main(["upgrade", "--no-qr"]) == 0
    plain = capsys.readouterr().out
    assert main(["upgrade", "--audit", "--no-qr"]) == 0
    assert capsys.readouterr().out == plain
    assert "▀" not in plain and "▄" not in plain
    for term in PILOT_TERMS:
        assert term in _flat(plain), term
    assert main(["upgrade", "--url-only"]) == 0
    assert capsys.readouterr().out == "https://throttle-pro.com\n"
    assert main(["upgrade", "--audit", "--url-only"]) == 0
    assert capsys.readouterr().out == "https://throttle-pro.com\n"
    with pytest.raises(SystemExit):
        main(["upgrade", "--help"])
    help_text = capsys.readouterr().out
    assert "--audit" not in help_text and "Throttle Pilot" in help_text


def test_upgrade_with_force_color_is_styled_and_still_complete(capsys, monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert main(["upgrade", "--no-qr"]) == 0
    out = capsys.readouterr().out
    assert "\x1b[" in out
    from throttle.style import strip_ansi
    for term in PILOT_TERMS:
        assert term in _flat(strip_ansi(out)), term


def test_nudge_text_points_at_the_pilot() -> None:
    assert "throttle upgrade" in upgrade.NUDGE_TEXT
    assert "Pilot" in upgrade.NUDGE_TEXT
    for retired in RETIRED:
        assert retired not in upgrade.NUDGE_TEXT


@pytest.fixture
def nudge_env(monkeypatch):
    monkeypatch.delenv(upgrade.NUDGE_ENV, raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(upgrade, "_stdout_is_tty", lambda: True)


def test_nudge_needs_a_tty(tmp_path: Path, monkeypatch) -> None:
    import io
    import sys

    monkeypatch.delenv(upgrade.NUDGE_ENV, raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(sys, "stdout", io.StringIO())  # a pipe or a log file
    assert upgrade.maybe_nudge(tmp_path) is None
    assert not (tmp_path / upgrade.NUDGE_FILE).exists()


def test_no_qr_on_a_stream_that_cannot_encode_it(monkeypatch) -> None:
    import io
    import sys

    from throttle.style import Style

    monkeypatch.delenv("FORCE_COLOR", raising=False)
    stream = io.TextIOWrapper(io.BytesIO(), encoding="ascii")
    page = upgrade.pilot_page(Style(stream), qr=True)
    page.encode("ascii")  # every character can be written to the stream
    assert "▀" not in page and "▄" not in page and "█" not in page
    assert "Scan with your phone" not in page
    assert page.rstrip().endswith(f"Open: {upgrade.PILOT_URL}")
    monkeypatch.setattr(sys, "stdout", stream)
    assert main(["upgrade"]) == 0
    stream.flush()
    assert f"Open: {upgrade.PILOT_URL}" in stream.buffer.getvalue().decode("ascii")


def test_nudge_at_most_once_per_24_hours(tmp_path: Path, nudge_env) -> None:
    now = 1_800_000_000.0
    assert upgrade.maybe_nudge(tmp_path, now=now) == upgrade.NUDGE_TEXT
    assert upgrade.maybe_nudge(tmp_path, now=now + 3600) is None
    assert upgrade.maybe_nudge(tmp_path, now=now + 24 * 3600 - 1) is None
    assert upgrade.maybe_nudge(tmp_path, now=now + 24 * 3600) == upgrade.NUDGE_TEXT


def test_nudge_disabled_by_env_and_in_ci(tmp_path: Path, nudge_env, monkeypatch) -> None:
    monkeypatch.setenv(upgrade.NUDGE_ENV, "1")
    assert upgrade.maybe_nudge(tmp_path) is None
    monkeypatch.delenv(upgrade.NUDGE_ENV)
    monkeypatch.setenv("CI", "true")
    assert upgrade.maybe_nudge(tmp_path) is None
    assert not (tmp_path / upgrade.NUDGE_FILE).exists()


def test_nudge_never_repeats_when_it_cannot_remember(tmp_path: Path, nudge_env) -> None:
    missing = tmp_path / "does-not-exist"
    assert upgrade.maybe_nudge(missing) is None
