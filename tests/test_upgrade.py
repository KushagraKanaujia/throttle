"""Tests for `throttle upgrade`, its offline QR codes and the once-a-day nudge.

The reference matrices in fixtures/upgrade_qr_matrices.json were produced by
the vendored encoder and decoded back to the exact URLs with an independent
decoder (zxing-cpp) when they were recorded; these tests pin them so the
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


@pytest.mark.parametrize("url", [upgrade.PRO_URL, upgrade.AUDIT_URL])
def test_qr_matrix_matches_decoded_reference(url: str) -> None:
    assert url in REFERENCE
    assert as_text(upgrade.qr_matrix(url)) == REFERENCE[url]


@pytest.mark.parametrize("url", [upgrade.PRO_URL, upgrade.AUDIT_URL])
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
    svg = upgrade.qr_svg(upgrade.PRO_URL)
    dark = sum(row.count("#") for row in REFERENCE[upgrade.PRO_URL])
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert svg.count("h1v1h-1z") == dark
    assert 'fill="#fff"' in svg


def test_upgrade_prints_pro_summary_link_and_qr(capsys) -> None:
    assert main(["upgrade"]) == 0
    out = capsys.readouterr().out
    assert "$19/month launch price" in out and "$50/month list price" in out
    assert "Open: https://throttle-pro.com/upgrade" in out
    assert "▀" in out or "▄" in out
    assert "throttle upgrade --audit" in out


def test_upgrade_audit_url_only_and_no_qr(capsys) -> None:
    assert main(["upgrade", "--audit", "--no-qr"]) == 0
    out = capsys.readouterr().out
    assert "$500 one-time" in out and "No savings guarantee" in out
    assert "https://throttle-pro.com/audit/book" in out
    assert "▀" not in out and "▄" not in out
    assert main(["upgrade", "--url-only"]) == 0
    assert capsys.readouterr().out == "https://throttle-pro.com/upgrade\n"
    assert main(["upgrade", "--audit", "--url-only"]) == 0
    assert capsys.readouterr().out == "https://throttle-pro.com/audit/book\n"


@pytest.fixture
def nudge_env(monkeypatch):
    monkeypatch.delenv(upgrade.NUDGE_ENV, raising=False)
    monkeypatch.delenv("CI", raising=False)


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
