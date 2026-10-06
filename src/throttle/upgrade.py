"""`throttle upgrade`: the Throttle Pilot terms, with a scannable QR code.

Everything here is offline. The QR code is generated locally (vendored
qrcodegen, MIT), nothing is fetched, and nothing is tracked. The only state is
the timestamp file that limits the post-verdict nudge to once per 24 hours.

`--audit` is a hidden alias kept so older docs and scripts still work; it
prints the same pilot page.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from ._vendor.qrcodegen import QrCode, QrSegment
from .style import Style

PILOT_URL = "https://throttle-pro.com"
PILOT_EMAIL = "kushthrottle@gmail.com"
PILOT_SUBJECT = "Throttle pilot"
# Older names, kept for imports; every offer now points at the pilot page.
PRO_URL = PILOT_URL
AUDIT_URL = PILOT_URL

NUDGE_ENV = "THROTTLE_NO_NUDGE"
NUDGE_FILE = ".upgrade-nudge"
NUDGE_INTERVAL_SECONDS = 24 * 60 * 60
NUDGE_TEXT = (
    "Want us to find and prove savings on your agent workload? "
    "Throttle Pilot, free for 3 months: `throttle upgrade`"
)

PILOT_TAGLINE = "We find savings on your agent workload and prove them."
PILOT_INTRO = (
    "We're working with a small group of teams running agents on their own GPUs."
)
PILOT_TERMS = (
    "Free for 3 months: we set Throttle up on your stack, find the config changes that lower your cost per token, and prove each one.",
    "We'd love to help you with your results.",
    "After the pilot, pricing is based on the savings we verify together "
    "(the conservative end of the 95% interval).",
)
PILOT_STEPS = (
    "We measure your agent workload's $/M tokens (baseline, 95% CI) on your own GPUs",
    "We test config changes and ship the winner only if it beats the noise bound with unchanged outputs",
    "Each verified saving is shown line by line, priced at the conservative end",
)
PILOT_ONE_LINE = (
    "Throttle Pilot: we find and prove savings on your agent workload. Free for 3 months, "
    "for a small group of teams. throttle upgrade"
)

# Served to the console Upgrade page by /api/upgrade.
PLANS = {
    "pilot": {
        "title": "Throttle Pilot",
        "tag": "for agent workloads on your own GPUs",
        "price": "Free for 3 months",
        "note": (
            "We're working with a small group of teams running agents on their own "
            "GPUs. We set Throttle up on your stack, find the config changes that "
            "lower your cost per token, and prove each one. We'd love to help you "
            "with your results. After the pilot, pricing is based on the savings we "
            "verify together (the conservative end of the 95% interval)."
        ),
        "items": [PILOT_TAGLINE, *PILOT_STEPS],
        "url": PILOT_URL,
        "email": PILOT_EMAIL,
        "subject": PILOT_SUBJECT,
        "command": "throttle upgrade",
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
    """The `throttle upgrade` screen: pilot terms in a panel, then the QR code."""

    rows: list = [(PILOT_TAGLINE, ("1",)), (PILOT_INTRO, ()), ""]
    rows += [(f"{style.dot} {term}", (), "  ") for term in PILOT_TERMS]
    rows += ["", ("How it works", ("1",))]
    rows += [(f"{i}. {step}", (), "   ") for i, step in enumerate(PILOT_STEPS, 1)]
    rows += [
        "",
        (f"Contact  {PILOT_URL}", ("1", "36")),
        (f"         {PILOT_EMAIL}  (subject \"{PILOT_SUBJECT}\")", ()),
        "",
        ("The CLI stays free, local and open source (MIT).", ("2",)),
    ]
    out = [style.panel(rows, title="Throttle Pilot")]
    if qr:
        out += ["", style.dim("Scan with your phone:"), qr_terminal(PILOT_URL)]
    out += ["", f"Open: {PILOT_URL}"]
    return "\n".join(out)


def handle_upgrade(args: argparse.Namespace) -> int:
    # --audit is a hidden alias: same pilot page, same link.
    if args.url_only:
        print(PILOT_URL)
        return 0
    print(pilot_page(Style(), qr=not args.no_qr))
    return 0


def maybe_nudge(directory: Path, now: float | None = None) -> str | None:
    """Return the nudge line at most once per 24 hours, else None.

    Disabled by THROTTLE_NO_NUDGE=1 and in CI. The only state is a timestamp
    file in the check history directory; nothing leaves the machine.
    """
    if os.environ.get(NUDGE_ENV, "").strip() not in ("", "0"):
        return None
    if os.environ.get("CI", "").strip() not in ("", "0", "false"):
        return None
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
