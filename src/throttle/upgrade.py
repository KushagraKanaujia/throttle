"""`throttle upgrade`: Throttle Cloud and the Enterprise pilot, with a scannable QR code.

Everything here is offline. The QR code is generated locally (vendored
qrcodegen, MIT), nothing is fetched, and nothing is tracked. The only state is
the timestamp file that limits the post-verdict nudge to once per 24 hours.

`--audit` is a hidden alias kept so older docs and scripts still work; it
prints the same page.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from ._vendor.qrcodegen import QrCode, QrSegment
from .style import Style

PRICING_URL = "https://www.throttle-pro.com/pricing"
PILOT_EMAIL = "kush@throttle-pro.com"
PILOT_SUBJECT = "Throttle pilot"
# Older names, kept for imports; every offer now points at the pricing page.
PILOT_URL = PRICING_URL
PRO_URL = PRICING_URL
AUDIT_URL = PRICING_URL

NUDGE_ENV = "THROTTLE_NO_NUDGE"
NUDGE_FILE = ".upgrade-nudge"
NUDGE_INTERVAL_SECONDS = 24 * 60 * 60
NUDGE_TEXT = (
    "Want Throttle to cut this bill, not just measure it? "
    "Throttle Cloud Autotune, start free with $10 of credits: `throttle upgrade`"
)

CLOUD_TAGLINE = "Throttle cuts what your LLM inference costs on your own GPUs."
CLOUD_INTRO = "The free CLI shows where the money goes. Throttle Cloud cuts it."
CLOUD_TERMS = (
    "Start free with $10 of credits, no card.",
    "Autotune restarts your server with candidate configs (batching, memory, GPU count, "
    "quantization) and keeps the cheapest one that beats the run-to-run noise and passes "
    "the OUTPUT CHANGED check. About $1 of credits per trial.",
    "CI cost gate (`throttle gate`) and a shared check history for your team.",
    "Enterprise: a hands-on pilot on your stack, priced on the savings we verify together.",
)
CLOUD_STEPS = (
    "throttle login   (paste the API key from your dashboard)",
    "throttle autotune --launch \"vllm serve ...\" --space ...   (try it free: --dry-run)",
    "throttle gate in CI keeps the savings from regressing",
)
# Older names, kept for imports.
PILOT_TAGLINE = CLOUD_TAGLINE
PILOT_INTRO = CLOUD_INTRO
PILOT_TERMS = CLOUD_TERMS
PILOT_STEPS = CLOUD_STEPS
PILOT_ONE_LINE = (
    "Throttle Cloud: Autotune finds the cheapest serving config that passes OUTPUT CHANGED. "
    "Start free with $10 of credits. throttle upgrade"
)

# Served to the console Upgrade page by /api/upgrade.
PLANS = {
    "cloud": {
        "title": "Throttle Cloud",
        "tag": "cuts what your LLM inference costs on your own GPUs",
        "price": "Start free: $10 of credits",
        "note": (
            "The free CLI shows where the money goes. Throttle Cloud cuts it: Autotune "
            "searches serving configs on your GPUs and keeps the cheapest one that beats "
            "the run-to-run noise and passes the OUTPUT CHANGED check. Plans and usage "
            "rates are on the pricing page. Enterprise: a hands-on pilot on your stack, "
            "priced on the savings we verify together."
        ),
        "items": [CLOUD_TAGLINE, *CLOUD_STEPS],
        "url": PRICING_URL,
        "email": PILOT_EMAIL,
        "subject": PILOT_SUBJECT,
        "command": "throttle login",
    },
}


def qr_matrix(text: str) -> list[list[bool]]:
    """Module matrix (True = dark) for ``text`` at error-correction level M."""
    qr = QrCode.encode_segments(QrSegment.make_segments(text), QrCode.Ecc.MEDIUM, boostecl=False)
    size = qr.get_size()
    return [[qr.get_module(x, y) for x in range(size)] for y in range(size)]


def qr_terminal(text: str, quiet: int = 2) -> str:
    """Render a QR code with Unicode half blocks, two module rows per text line.

    Dark modules are drawn as spaces and light modules (plus the quiet zone) as
    full blocks in the terminal's foreground color. On a dark terminal that
    gives dark modules on a light field, which is what phone scanners expect;
    on a light terminal the colors invert, which modern scanners also read.
    """
    matrix = qr_matrix(text)
    size = len(matrix)
    total = size + 2 * quiet

    def light(x: int, y: int) -> bool:
        mx, my = x - quiet, y - quiet
        if 0 <= mx < size and 0 <= my < size:
            return not matrix[my][mx]
        return True

    rows = []
    for y in range(0, total, 2):
        line = []
        for x in range(total):
            top = light(x, y)
            bottom = light(x, y + 1) if y + 1 < total else True
            line.append("█" if top and bottom else "▀" if top else "▄" if bottom else " ")
        rows.append("".join(line))
    return "\n".join(rows)


def qr_svg(text: str, quiet: int = 4, scale: int = 8) -> str:
    """Standalone SVG QR code (dark on white) for the console."""
    matrix = qr_matrix(text)
    size = len(matrix) + 2 * quiet
    path = "".join(
        f"M{x + quiet},{y + quiet}h1v1h-1z"
        for y, row in enumerate(matrix)
        for x, dark in enumerate(row)
        if dark
    )
    px = size * scale
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" '
        f'width="{px}" height="{px}" shape-rendering="crispEdges">'
        f'<rect width="{size}" height="{size}" fill="#fff"/>'
        f'<path d="{path}" fill="#000"/></svg>'
    )


def pilot_page(style: Style, *, qr: bool) -> str:
    """The `throttle upgrade` screen: Throttle Cloud in a panel, then the QR code."""

    rows: list = [(CLOUD_TAGLINE, ("1",)), (CLOUD_INTRO, ()), ""]
    rows += [(f"{style.dot} {term}", (), "  ") for term in CLOUD_TERMS]
    rows += ["", ("Get started", ("1",))]
    rows += [(f"{i}. {step}", (), "   ") for i, step in enumerate(CLOUD_STEPS, 1)]
    rows += [
        "",
        (f"Pricing     {PRICING_URL}", ("1", "36")),
        (f"Enterprise  {PILOT_EMAIL}  (subject \"{PILOT_SUBJECT}\")", ()),
        "",
        ("The CLI stays free, local and open source (MIT).", ("2",)),
    ]
    out = [style.panel(rows, title="Throttle Cloud")]
    # The QR code is drawn with block characters: only on a stream that can
    # encode them. Otherwise the URL line below is the whole link.
    if qr and style.unicode:
        out += ["", style.dim("Scan with your phone:"), qr_terminal(PRICING_URL)]
    out += ["", f"Open: {PRICING_URL}"]
    return "\n".join(out)


def handle_upgrade(args: argparse.Namespace) -> int:
    # --audit is a hidden alias: same page, same link.
    if args.url_only:
        print(PRICING_URL)
        return 0
    print(pilot_page(Style(), qr=not args.no_qr))
    return 0


def _stdout_is_tty() -> bool:
    try:
        return bool(sys.stdout.isatty())
    except (AttributeError, ValueError):
        return False


def maybe_nudge(directory: Path, now: float | None = None) -> str | None:
    """Return the nudge line at most once per 24 hours, else None.

    Disabled by THROTTLE_NO_NUDGE=1, in CI, and when stdout is not a TTY
    (so it never lands in a pipe or a log). The only state is a timestamp
    file in the check history directory; nothing leaves the machine.
    """
    if os.environ.get(NUDGE_ENV, "").strip() not in ("", "0"):
        return None
    if os.environ.get("CI", "").strip() not in ("", "0", "false"):
        return None
    if not _stdout_is_tty():
        return None  # never into a pipe or a log
    now = time.time() if now is None else now
    stamp = directory / NUDGE_FILE
    try:
        last = float(stamp.read_text().strip())
    except (OSError, ValueError):
        last = None
    if last is not None and 0 <= now - last < NUDGE_INTERVAL_SECONDS:
        return None
    try:
        stamp.write_text(f"{now:.0f}\n")
    except OSError:
        return None  # never nudge repeatedly if we can't remember that we did
    return NUDGE_TEXT
