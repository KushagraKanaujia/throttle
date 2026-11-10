"""The CLI front door: style module, check verdict panel, savings panel, retired offers.

Human output only. Machine-readable output is pinned separately, byte for
byte, in test_machine_output_golden.py.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import pytest

from throttle import style
from throttle.style import Style, strip_ansi, visible_len

from test_check import FakeServer, constant, history, run_check  # noqa: F401  (fixture)
from test_savings import make_check, pair, run_savings, write_history

REPO = Path(__file__).resolve().parents[1]


class FakeStream(io.StringIO):
    def __init__(self, tty: bool, encoding: str = "utf-8") -> None:
        super().__init__()
        self._tty = tty
        self._encoding = encoding

    def isatty(self) -> bool:
        return self._tty

    @property
    def encoding(self) -> str:  # type: ignore[override]
        return self._encoding


@pytest.fixture
def clean_env(monkeypatch):
    for key in ("NO_COLOR", "FORCE_COLOR", "TERM"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    return monkeypatch


def flat(out: str) -> str:
    return " ".join(" ".join(line.strip().strip("│|").split()) for line in strip_ansi(out).splitlines())


# --------------------------------------------------------------------------
# style module
# --------------------------------------------------------------------------


def test_colour_follows_tty(clean_env) -> None:
    assert style.colour_enabled(FakeStream(tty=True)) is True
    assert style.colour_enabled(FakeStream(tty=False)) is False
    assert Style(FakeStream(tty=True)).paint("x", style.GREEN) == "\x1b[32mx\x1b[0m"
    assert Style(FakeStream(tty=False)).paint("x", style.GREEN) == "x"


def test_no_color_wins_over_tty_and_force_color(clean_env) -> None:
    clean_env.setenv("NO_COLOR", "1")
    assert style.colour_enabled(FakeStream(tty=True)) is False
    clean_env.setenv("FORCE_COLOR", "1")
    assert style.colour_enabled(FakeStream(tty=True)) is False
    clean_env.setenv("NO_COLOR", "")  # empty means unset, per no-color.org
    assert style.colour_enabled(FakeStream(tty=False)) is True


def test_force_color_turns_colour_on_without_a_tty(clean_env) -> None:
    clean_env.setenv("FORCE_COLOR", "1")
    assert style.colour_enabled(FakeStream(tty=False)) is True
    clean_env.setenv("FORCE_COLOR", "0")
    assert style.colour_enabled(FakeStream(tty=False)) is False


def test_dumb_terminal_gets_no_colour(clean_env) -> None:
    clean_env.setenv("TERM", "dumb")
    assert style.colour_enabled(FakeStream(tty=True)) is False


def test_verdict_palette() -> None:
    st = Style(colour=True, unicode=True, width=60)
    assert "\x1b[1;32m" in st.verdict("CHEAPER")
    assert "\x1b[1;31m" in st.verdict("MORE EXPENSIVE")
    for word in ("NO WINNER", "NOT CALIBRATED", "OUTPUT CHANGED"):
        assert "\x1b[1;33m" in st.verdict(word)
    assert strip_ansi(st.verdict("CHEAPER")) == "CHEAPER"
    assert Style(colour=False, width=60).verdict("CHEAPER") == "CHEAPER"


def test_ascii_fallback_when_encoding_is_not_utf8(clean_env) -> None:
    st = Style(FakeStream(tty=False, encoding="ascii"))
    assert st.unicode is False
    box = st.panel([("hello", ())], title="T")
    assert box.isascii()
    assert box.splitlines()[0].startswith("+-") and box.splitlines()[1].startswith("| ")
    st = Style(FakeStream(tty=False, encoding="UTF-8"))
    assert st.unicode is True
    assert st.panel([("hello", ())]).startswith("┌")


def test_width_is_capped_and_never_below_60(clean_env) -> None:
    clean_env.setenv("COLUMNS", "40")
    assert style.terminal_width() == 60
    clean_env.setenv("COLUMNS", "72")
    assert style.terminal_width() == 72
    clean_env.setenv("COLUMNS", "300")
    assert style.terminal_width() == style.MAX_WIDTH
    assert Style(width=10).width == 60


@pytest.mark.parametrize("colour", [False, True])
def test_panel_rows_are_one_width_and_wrap(colour: bool) -> None:
    st = Style(colour=colour, unicode=True, width=60)
    long = "word " * 40
    box = st.panel([(long, ("2",)), st.bold("short"), ""], title="Verdict")
    widths = {visible_len(line) for line in box.splitlines()}
    assert widths == {60}
    assert flat(box).count("word") == 40


def test_styled_row_wider_than_the_panel_wraps_instead_of_overflowing() -> None:
    st = Style(colour=True, unicode=True, width=60)
    wide = st.accent("3.") + " " + "a styled row that is far too long for a sixty column panel"
    box = st.panel([wide, st.bold("short")], title="Verdict")
    assert {visible_len(line) for line in box.splitlines()} == {60}
    assert "far too long for a sixty column panel" in flat(box)


def test_welcome_screen_fits_60_columns_with_colour(monkeypatch) -> None:
    from throttle.cli import WELCOME_STEPS, welcome_screen

    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("COLUMNS", "60")
    screen = welcome_screen()
    assert "\x1b[" in screen
    panel = [line for line in screen.splitlines() if strip_ansi(line)[:1] in "┌│└+|"]
    assert panel and all(visible_len(line) == 60 for line in panel), [
        visible_len(line) for line in panel
    ]
    first, second, _ = WELCOME_STEPS[0]
    assert len(f"1. {first}") <= 56 and len(f"   {second}") <= 56  # wraps cleanly: no rewrap
    assert f"1. {first}" in strip_ansi(screen)


def test_verdict_panel_fits_60_columns_with_colour(history, monkeypatch, capsys):  # noqa: F811
    monkeypatch.setenv("THROTTLE_NO_NUDGE", "1")
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("COLUMNS", "60")
    code, out = calibrated_cheaper(monkeypatch, capsys)
    assert code == 0, out
    lines = strip_ansi(out).splitlines()
    start = next(i for i, line in enumerate(lines) if "Verdict " in line and line[:1] in "┌+")
    end = next(i for i in range(start + 1, len(lines)) if lines[i][:1] in "└+")
    assert all(len(line) == 60 for line in lines[start:end + 1]), [
        len(line) for line in lines[start:end + 1]
    ]


def test_narrow_terminal_gets_panels_without_a_frame(monkeypatch) -> None:
    from throttle.cli import welcome_screen

    monkeypatch.setenv("COLUMNS", "40")
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    st = Style(colour=False, unicode=True)
    assert st.narrow is True
    box = st.panel([("word " * 30, ()), "short"], title="Verdict")
    assert all(len(line) <= 40 for line in box.splitlines())
    assert not any(ch in box for ch in "┌┐└┘│")
    assert box.splitlines()[0] == "Verdict"
    screen = welcome_screen()
    assert not any(ch in screen for ch in "┌┐└┘")
    assert "1. See the whole loop" in screen
    # 60 columns and up keep the frame.
    monkeypatch.setenv("COLUMNS", "60")
    assert Style(colour=False, unicode=True).panel(["x"]).startswith("┌")


# --------------------------------------------------------------------------
# throttle check: verdict panel
# --------------------------------------------------------------------------


def panel_of(out: str) -> str:
    plain = strip_ansi(out)
    lines = plain.splitlines()
    start = next(i for i, line in enumerate(lines) if "Verdict " in line and line[:1] in "┌+")
    end = next(i for i in range(start + 1, len(lines)) if lines[i][:1] in "└+")
    return "\n".join(lines[start:end + 1])


def calibrated_cheaper(monkeypatch, capsys, *extra: str) -> tuple[int, str]:
    for _ in range(3):
        run_check(monkeypatch, capsys, FakeServer(constant(0.08)), "--config", "quant=none")
    return run_check(
        monkeypatch, capsys, FakeServer(constant(0.01)), "--config", "quant=fp8",
        "--monthly-tokens", "1B", *extra,
    )


@pytest.mark.parametrize("force", [False, True])
def test_check_verdict_panel_has_every_field(history, monkeypatch, capsys, force):  # noqa: F811
    monkeypatch.setenv("THROTTLE_NO_NUDGE", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)
    if force:
        monkeypatch.setenv("FORCE_COLOR", "1")
    else:
        monkeypatch.delenv("FORCE_COLOR", raising=False)
    code, out = calibrated_cheaper(monkeypatch, capsys)
    assert code == 0, out
    assert ("\x1b[" in out) is force
    if force:
        assert "\x1b[1;32mCHEAPER\x1b[0m" in out
    records = (history / "checks.ndjson").read_text().splitlines()
    import json

    first_id = json.loads(records[2])["id"]
    last_id = json.loads(records[-1])["id"]
    panel = panel_of(out)
    text = flat(panel)
    assert re.search(r"\$\d[\d.,]*/M output tokens 95% CI \$[\d.,]+ to \$[\d.,]+ \[MEASURED\]", text), text
    assert "CHEAPER" in text
    assert "CHEAPER the -" in text and "is larger than the run-to-run noise bound" in text  # the reason
    assert re.search(r"vs base -\$[\d.,]+/M \(-\d+\.\d%\), baseline \$[\d.,]+/M", text), text
    assert first_id in text
    assert re.search(r"monthly -\$[\d.,]+ per month at 1B output tokens \[PROJECTED\]", text), text
    assert f"check {last_id}" in text
    assert f"next throttle savings --baseline {first_id} --candidate {last_id} --tokens N" in text
    # Restyled, not removed: every existing line is still printed above the panel.
    plain = strip_ansi(out)
    for kept in (
        "[ASSUMED: you supplied --gpu-hourly-rate",
        "per million output tokens   [MEASURED]",
        "(Student t)",
        "run-to-run noise bound",
        "[PROJECTED: MEASURED $/M x ASSUMED volume]",
        "Verdict: CHEAPER, ",
        f"Saved as check {last_id}",
        "Config fingerprint",
        "  block 1/5  $",
    ):
        assert kept in plain, kept


def test_check_panel_first_check_and_not_calibrated(history, monkeypatch, capsys):  # noqa: F811
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    code, out = run_check(monkeypatch, capsys, FakeServer(constant(0.01)), "--monthly-tokens", "500M")
    assert code == 0, out
    text = flat(panel_of(out))
    assert "FIRST CHECK" in text and "this check is the baseline" in text
    assert "monthly $" in text and "at 500M output tokens [PROJECTED]" in text
    assert "next run 2 more unchanged checks to calibrate noise" in text

    code, out = run_check(monkeypatch, capsys, FakeServer(constant(0.01)))
    text = flat(panel_of(out))
    assert "NOT CALIBRATED" in text and "CHEAPER" not in out
    assert "next run 1 more unchanged check(s) to calibrate noise" in text
    code, out = run_check(monkeypatch, capsys, FakeServer(constant(0.01)))
    text = flat(panel_of(out))
    assert "noise is calibrated from the next check: change one setting" in text


def test_check_panel_no_save_and_no_winner(history, monkeypatch, capsys):  # noqa: F811
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    noisy = [0.01, 0.06, 0.02, 0.05, 0.03]
    for _ in range(3):
        run_check(monkeypatch, capsys, FakeServer(lambda b: noisy[b]), "--config", "batch=8")
    code, out = run_check(
        monkeypatch, capsys, FakeServer(lambda b: noisy[b]), "--config", "batch=16", "--no-save",
    )
    text = flat(panel_of(out))
    assert "NO WINNER" in text
    assert "check not saved (--no-save)" in text
    assert "next the change is within noise" in text


def test_ascii_terminal_gets_an_ascii_panel(history, monkeypatch, capsys):  # noqa: F811
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setattr(style, "unicode_enabled", lambda stream=None: False)
    code, out = run_check(monkeypatch, capsys, FakeServer(constant(0.01)))
    panel = panel_of(out)
    assert panel.isascii() and panel.startswith("+-")


# --------------------------------------------------------------------------
# throttle savings: panel
# --------------------------------------------------------------------------


@pytest.mark.parametrize("force", [False, True])
def test_savings_panel_headlines_the_conservative_figure(tmp_path, capsys, monkeypatch, force):
    monkeypatch.delenv("NO_COLOR", raising=False)
    if force:
        monkeypatch.setenv("FORCE_COLOR", "1")
    else:
        monkeypatch.delenv("FORCE_COLOR", raising=False)
    history_dir = tmp_path / "history"
    write_history(history_dir, make_check("cand", 30, 0.60, quant="fp8"))
    code, out, err = run_savings(capsys, history_dir, *pair("--tokens", "1B", "--period-label", "2026-10"))
    assert code == 0, err
    assert ("\x1b[" in out) is force
    plain = strip_ansi(out)
    lines = plain.splitlines()
    assert lines[0].startswith(("┌─ Verified savings", "+- Verified savings"))
    # The conservative figure is the first line inside the panel; the point estimate follows, dim.
    assert "Verified savings (conservative): $339.15" in lines[1]
    assert "Point estimate (not the verified figure): $400.00" in plain
    if force:
        assert "\x1b[1;32m$339.15\x1b[0m" in out
        assert "\x1b[2mPoint estimate (not the verified figure): $400.00" in out
        assert "\x1b[1;32mCHEAPER\x1b[0m" in out
    text = flat(out)
    for field in (
        "period 2026-10", "model qwen-32b", "$/M output tokens", "base-3", "cand", "1 GPU(s)",
        "[MEASURED]", "1,000,000,000 [REPORTED BY OPERATOR]", "CHEAPER (re-judged)",
        "baseline CI low", "candidate CI high", "run-to-run noise bound", "baseline mean",
        "candidate mean", "GPU $/hr is ASSUMED", "Assumptions",
    ):
        assert field in text, field
    assert {visible_len(line) for line in lines if line[:1] in "┌│└+|"} == {lines[0].__len__()}


# --------------------------------------------------------------------------
# retired offers
# --------------------------------------------------------------------------

RETIRED_PATTERNS = {
    "$19": re.compile(r"\$19(?![.,]\d)"),  # the $19/month Pro price; $19.72 in a sample output is a number, not an offer
    "Cost Audit": re.compile(r"Cost Audit", re.IGNORECASE),
    # The retired Throttle Pro "(early access)" features, and Autotune's
    # "free during early access" Cloud framing (0.7.0).
    "early access": re.compile(r"early access", re.IGNORECASE),
    "no savings guarantee": re.compile(r"no savings guarantee", re.IGNORECASE),
    "formspree": re.compile(r"formspree", re.IGNORECASE),
    # The 2-week pilot at 20% with a $500 floor (retired 2026-10-06).
    "20% of verified": re.compile(r"20% of verified", re.IGNORECASE),
    "$500/month": re.compile(r"\$500\s*/\s*month", re.IGNORECASE),
    "2 weeks": re.compile(r"\b2 weeks\b", re.IGNORECASE),
    # Throttle Cloud credit plans (retired 2026-10-27). Pricing is a flat
    # monthly fee, and the CLI never logs in or meters anything.
    "Throttle Cloud": re.compile(r"Throttle Cloud", re.IGNORECASE),
    "$10 of credits": re.compile(r"\$10 (?:of )?(?:free )?credits?", re.IGNORECASE),
    "credits left": re.compile(r"credits left", re.IGNORECASE),
    "$30/$50/$100 plans": re.compile(r"\$(?:30|50|100)\s*/\s*mo(?:nth)?\b", re.IGNORECASE),
    "throttle upgrade": re.compile(r"throttle upgrade\b", re.IGNORECASE),
    "throttle login": re.compile(r"throttle (?:login|logout|whoami)\b", re.IGNORECASE),
    # Share-of-savings billing (retired for the flat monthly fee, 2026-11).
    "we bill on": re.compile(r"we bill on", re.IGNORECASE),
    "share of the savings": re.compile(r"share of the savings", re.IGNORECASE),
    "no bill": re.compile(r"\bno bill\b", re.IGNORECASE),
    "billable": re.compile(r"\bbillable\b", re.IGNORECASE),
}

SKIP_SUFFIXES = (".pyc", ".png", ".ico", ".woff2")


def _is_build_junk(path: Path) -> bool:
    """A stale build or cache folder (src/*.egg-info, __pycache__) is not source."""

    return any(part == "__pycache__" or part.endswith(".egg-info") for part in path.parts)


def tracked_src_files() -> list[Path]:
    """The files git tracks under src/, or every file there when git is not available."""

    import subprocess

    try:
        listed = subprocess.run(
            ["git", "ls-files", "-z", "src"], cwd=REPO, capture_output=True, check=True, timeout=30,
        ).stdout.decode("utf-8").split("\0")
        files = [REPO / name for name in listed if name]
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError):
        files = []
    if not files:
        files = sorted((REPO / "src").rglob("*"))
    return [path for path in files if not _is_build_junk(path.relative_to(REPO))]


def test_no_retired_offer_strings_in_src_or_readme() -> None:
    files = [REPO / "README.md", *tracked_src_files()]
    hits = []
    for path in files:
        if not path.is_file() or "_vendor" in path.parts or path.suffix in SKIP_SUFFIXES:
            continue
        if path.name.lower() == "changelog.md":
            continue  # release history may name the offers it retired
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for name, pattern in RETIRED_PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                hits.append(f"{path.relative_to(REPO)}:{line}: {name}")
    assert hits == []


def test_retired_scan_skips_stale_build_folders(tmp_path) -> None:
    assert _is_build_junk(Path("src/throttle_pro.egg-info/PKG-INFO"))
    assert _is_build_junk(Path("src/throttle/__pycache__/cli.cpython-312.pyc"))
    assert not _is_build_junk(Path("src/throttle/cli.py"))
    scanned = {path.relative_to(REPO).as_posix() for path in tracked_src_files()}
    assert "src/throttle/cli.py" in scanned
    assert not any(".egg-info" in name or "__pycache__" in name for name in scanned)


@pytest.mark.parametrize("text", [
    "the number we bill on", "we take a share of the savings", "No savings this period, no bill",
    "net savings (billable)",
])
def test_retired_billing_language_is_caught(text: str) -> None:
    assert any(pattern.search(text) for pattern in RETIRED_PATTERNS.values()), text


# --------------------------------------------------------------------------
# README trust sections
# --------------------------------------------------------------------------

PINNED_SHA = "64cb03e30b87b1de0e4bcc6bed55d0a54641eb95"  # v0.7.0


def readme_section(title: str) -> str:
    text = (REPO / "README.md").read_text(encoding="utf-8")
    start = text.index(f"## {title}\n")
    end = text.find("\n## ", start + 1)
    return text[start:end if end != -1 else None]


def test_readme_install_and_action_are_pinned() -> None:
    text = (REPO / "README.md").read_text(encoding="utf-8")
    assert 'pip install "throttle-pro==0.7.0"' in text
    assert "PyPI is behind" not in text
    assert f"uses: KushagraKanaujia/throttle/.github/actions/throttle-pr@{PINNED_SHA} # v0.7.0" in text
    assert "Pin the action by its full commit SHA" in text
    assert "throttle-pr@main" not in text
    assert not re.search(r"pip install git\+https://github\.com/KushagraKanaujia/throttle\s*$", text, re.M)
    for doc in ("getting-started/installation.md", "getting-started/quickstart.md"):
        body = (REPO / "src/throttle/console/docs" / doc).read_text(encoding="utf-8")
        assert 'throttle-pro==0.7.0' in body, doc
        assert "behind until" not in body, doc
        assert "own/mvp" not in body and "pipx upgrade" not in body, doc


def test_readme_has_the_honest_setup_line() -> None:
    flat_text = " ".join((REPO / "README.md").read_text(encoding="utf-8").split())
    assert ("Setup: one base-URL change to a proxy you run, a model server or cheaper API model to test "
            "against, and a judge key you choose. The GitHub Action comments on PRs.") in flat_text


def test_readme_security_model_and_data_flow() -> None:
    section = " ".join(readme_section("Security model and data flow").split())
    for needed in ("A proxy you run", "127.0.0.1", "Your CI", "The CLI", "Judge calls", "provider and key you choose",
                   "Candidate API models", "Throttle the company receives nothing", "No telemetry",
                   "environment variables you name", "never printed or written to disk", "`mask` (default)",
                   "`hash`", "`metadata`", "`off`", "trusted publishing", "pinned", "sees your API keys",
                   "[SECURITY.md](SECURITY.md)"):
        assert needed in section, needed
    security = " ".join((REPO / "SECURITY.md").read_text(encoding="utf-8").split())
    for needed in ("Security model and data flow", "Throttle the company receives nothing",
                   "Judge calls go to the provider and key you choose", "Candidate API models",
                   "trusted publishing", "0.7.x"):
        assert needed in security, needed


def test_readme_how_throttle_is_different_uses_only_real_numbers() -> None:
    section = readme_section("How Throttle is different")
    flat_text = " ".join(section.split())
    for needed in ("tracked in pull requests", "never moved silently", "inside the noise", '"looked 47% cheaper"',
                   '"!!!!"', "$1.67 to $1.13 per 1M tokens (32% cheaper)", "not against an API",
                   "Fallback per request", "Nothing hosted, nothing to shut down", "Flat fee",
                   "No cut of each request",
                   "default path tests existing models and does not train on provider outputs"):
        assert needed in flat_text, needed
    # Only the real receipts: on-the-fly FP8 looked 47% cheaper (32B said "!!!!"),
    # and pre-quantized FP8 Qwen2.5-72B on an MI300X went $1.67 -> $1.13 (32% cheaper).
    allowed = {"47%", "32B", "Qwen2.5-72B", "MI300X", "$1.67", "$1.13", "1M", "32%", "FP8", "BF16"}
    numbers = set(re.findall(r"[\w$.-]*\d[\w%.-]*", section))
    assert numbers <= allowed, numbers - allowed


def test_readme_and_docs_have_no_em_dashes() -> None:
    docs = REPO / "src/throttle/console/docs"
    paths = [REPO / "README.md", REPO / "SECURITY.md", docs / "index.md",
             *(docs / "getting-started").glob("*.md"), *(docs / "loop").glob("*.md"), *(docs / "more").glob("*.md")]
    offenders = [str(p.relative_to(REPO)) for p in paths if "—" in p.read_text(encoding="utf-8")]
    assert offenders == []


def test_issue_template_links_the_current_pitch() -> None:
    text = (REPO / ".github/ISSUE_TEMPLATE/config.yml").read_text(encoding="utf-8")
    assert "url: https://throttle-pro.com" in text and "Stop renting your AI" in text
    assert "Pilot" not in text and "own GPUs" not in text
