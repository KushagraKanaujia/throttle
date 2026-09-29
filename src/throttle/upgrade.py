"""`throttle upgrade`: where to get Pro or the Cost Audit, with a scannable QR code.

Everything here is offline. The QR code is generated locally (vendored
qrcodegen, MIT), nothing is fetched, and nothing is tracked. The only state is
the timestamp file that limits the post-verdict nudge to once per 24 hours.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from ._vendor.qrcodegen import QrCode, QrSegment

PRO_URL = "https://throttle-pro.com/upgrade"
AUDIT_URL = "https://throttle-pro.com/audit/book"

NUDGE_ENV = "THROTTLE_NO_NUDGE"
NUDGE_FILE = ".upgrade-nudge"
NUDGE_INTERVAL_SECONDS = 24 * 60 * 60
NUDGE_TEXT = "Want this checked on every deploy? `throttle upgrade`"

# Wording mirrors the Pro and Cost Audit cards on throttle-pro.com.
PRO_SUMMARY = """\
Throttle Pro, for what repeats (early access)
  $19/month launch price ($50/month list price once Pro fully launches).
  Sign up during launch and you keep $19/month for as long as you stay subscribed.

  - Scheduled config-drift checks              (early access)
  - Cost history across deploys                (early access)
  - Alerts when a change makes you pay more    (early access)
  - Team sharing                               (early access)

  These features are in progress. The CLI stays free and local (MIT)."""

AUDIT_SUMMARY = """\
Throttle Cost Audit: $500 one-time, done with you in 2 weeks
  - Your true $ per million tokens, measured on your own stack with 95% confidence intervals
  - Calibrated verdicts on up to 3 config changes you choose: flags, quantization, engine or GPU
  - A CI cost gate wired into your pipeline, so a costlier deploy fails before production
  - A written report in dollars, with the raw results attached

  No savings guarantee. What you get is a measurement you can trust, in dollars."""


PLANS = {
    "pro": {
        "title": "Throttle Pro",
        "tag": "for what repeats · early access",
        "price": "$19/month launch price",
        "note": "$50/month list price once Pro fully launches. Sign up during launch and you keep $19/month for as long as you stay subscribed.",
        "items": [
            "Scheduled config-drift checks (early access)",
            "Cost history across deploys (early access)",
            "Alerts when a change makes you pay more (early access)",
            "Team sharing (early access)",
        ],
        "url": PRO_URL,
        "command": "throttle upgrade",
    },
    "audit": {
        "title": "Cost Audit",
        "tag": "done with you · 2 weeks",
        "price": "$500 one-time",
        "note": "No savings guarantee. What you get is a measurement you can trust, in dollars.",
        "items": [
            "Your true $ per million tokens, measured on your own stack with 95% confidence intervals",
            "Calibrated verdicts on up to 3 config changes you choose: flags, quantization, engine or GPU",
            "A CI cost gate wired into your pipeline, so a costlier deploy fails before production",
            "A written report in dollars, with the raw results attached",
        ],
        "url": AUDIT_URL,
        "command": "throttle upgrade --audit",
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


def handle_upgrade(args: argparse.Namespace) -> int:
    url = AUDIT_URL if args.audit else PRO_URL
    if args.url_only:
        print(url)
        return 0
    print(AUDIT_SUMMARY if args.audit else PRO_SUMMARY)
    print()
    if not args.no_qr:
        print("Scan with your phone:")
        print(qr_terminal(url))
        print()
    print(f"Open: {url}")
    if not args.audit:
        print("Want it done with you instead? throttle upgrade --audit")
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
