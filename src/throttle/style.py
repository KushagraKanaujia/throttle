"""Terminal styling for Throttle's human output: plain ANSI, no dependencies.

Colour and bold are used only when they can help and never when they can hurt:

* on when stdout is a TTY, ``NO_COLOR`` is unset and ``TERM`` is not ``dumb``;
* ``FORCE_COLOR`` (any value but ``0``) turns colour on without a TTY, for
  tests and for piping into ``less -R``;
* ``NO_COLOR`` (any non-empty value, https://no-color.org) always wins.

Panels use box-drawing characters when the stream's encoding is UTF-8 and plain
ASCII (``+-|``) otherwise. Width follows the terminal, capped at
``MAX_WIDTH`` and never below ``MIN_WIDTH`` columns.

Machine-readable output (``--json``, ``--share``) never goes through here.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import textwrap
from typing import IO, Iterable, Sequence

MIN_WIDTH = 60
MAX_WIDTH = 80

RESET = "\x1b[0m"
BOLD = "1"
DIM = "2"
ACCENT = "36"   # brand accent: cyan
GREEN = "32"    # CHEAPER
RED = "31"      # MORE EXPENSIVE
AMBER = "33"    # NO WINNER, NOT CALIBRATED, OUTPUT CHANGED

VERDICT_COLOURS = {
    "CHEAPER": GREEN,
    "MORE EXPENSIVE": RED,
    "NO WINNER": AMBER,
    "NOT CALIBRATED": AMBER,
    "OUTPUT CHANGED": AMBER,
}

_ANSI = re.compile(r"\x1b\[[0-9;]*m")

UNICODE_BOX = {"tl": "┌", "tr": "┐", "bl": "└", "br": "┘", "h": "─", "v": "│", "dot": "·"}
ASCII_BOX = {"tl": "+", "tr": "+", "bl": "+", "br": "+", "h": "-", "v": "|", "dot": "-"}


def colour_enabled(stream: IO[str] | None = None) -> bool:
    """Whether to emit ANSI colour on ``stream`` (default: stdout)."""

    if os.environ.get("NO_COLOR", ""):
        return False
    force = os.environ.get("FORCE_COLOR")
    if force is not None and force.strip() not in ("0", "false"):
        return True
    if os.environ.get("TERM", "") == "dumb":
        return False
    stream = sys.stdout if stream is None else stream
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False


def unicode_enabled(stream: IO[str] | None = None) -> bool:
    """Whether ``stream`` can print box-drawing characters (UTF-8 encoding)."""

    stream = sys.stdout if stream is None else stream
    encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "").replace("_", "")
    return encoding in ("utf8", "utf8sig")


def terminal_width() -> int:
    columns = shutil.get_terminal_size(fallback=(MAX_WIDTH, 24)).columns
    return max(MIN_WIDTH, min(columns, MAX_WIDTH))


def visible_len(text: str) -> int:
    return len(_ANSI.sub("", text))


def strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


class Style:
    """Styling decisions for one output stream, fixed at construction."""

    def __init__(
        self,
        stream: IO[str] | None = None,
        *,
        colour: bool | None = None,
        unicode: bool | None = None,
        width: int | None = None,
    ) -> None:
        self.colour = colour_enabled(stream) if colour is None else colour
        self.unicode = unicode_enabled(stream) if unicode is None else unicode
        self.width = max(MIN_WIDTH, width if width is not None else terminal_width())
        self.box = UNICODE_BOX if self.unicode else ASCII_BOX

    # ----- inline -----
    def paint(self, text: str, *codes: str) -> str:
        if not self.colour or not codes or not text:
            return text
        return f"\x1b[{';'.join(codes)}m{text}{RESET}"

    def bold(self, text: str) -> str:
        return self.paint(text, BOLD)

    def dim(self, text: str) -> str:
        return self.paint(text, DIM)

    def accent(self, text: str) -> str:
        return self.paint(text, BOLD, ACCENT)

    def good(self, text: str) -> str:
        return self.paint(text, GREEN)

    def bad(self, text: str) -> str:
        return self.paint(text, RED)

    def warn(self, text: str) -> str:
        return self.paint(text, AMBER)

    def verdict(self, word: str, *, bold: bool = True) -> str:
        colour = VERDICT_COLOURS.get(word)
        if colour is None:
            return self.bold(word) if bold else word
        return self.paint(word, *((BOLD, colour) if bold else (colour,)))

    @property
    def dot(self) -> str:
        return self.box["dot"]

    def brand(self) -> str:
        return self.accent("Throttle")

    # ----- blocks -----
    def rule(self, title: str = "") -> str:
        h = self.box["h"]
        if not title:
            return self.dim(h * self.width)
        head = f"{h}{h} {title} "
        return self.dim(head + h * max(0, self.width - visible_len(head)))

    def wrap(self, text: str, width: int, indent: str = "") -> list[str]:
        """Wrap plain ``text`` (no ANSI) to ``width``; continuation lines get ``indent``."""

        if not text:
            return [""]
        return textwrap.wrap(
            text, width=width, subsequent_indent=indent,
            break_long_words=True, break_on_hyphens=False,
        ) or [""]

    def panel(
        self,
        rows: Iterable[str | tuple[str, Sequence[str]] | tuple[str, Sequence[str], str]],
        *,
        title: str | None = None,
    ) -> str:
        """A boxed panel.

        Each row is either a pre-styled string (printed as is, padded by its
        visible length; keep it shorter than the panel) or ``(text, codes)`` /
        ``(text, codes, indent)``: plain text wrapped to the panel width, each
        wrapped line painted with ``codes``.
        """

        b = self.box
        inner = self.width - 4  # "│ " + text + " │"
        lines: list[str] = []
        for row in rows:
            if isinstance(row, tuple):
                text, codes, *rest = row
                indent = rest[0] if rest else ""
                for piece in self.wrap(text, inner, indent):
                    lines.append(self.paint(piece, *codes))
            else:
                for piece in row.split("\n"):
                    if visible_len(piece) > inner and "\x1b" not in piece:
                        lines.extend(self.wrap(piece, inner))
                    else:
                        lines.append(piece)
        if title:
            label = f" {title} "
            if len(label) > self.width - 4:
                label = label[: self.width - 6] + ".. "
            top = self.dim(b["tl"] + b["h"]) + self.accent(label) + self.dim(
                b["h"] * max(0, self.width - 3 - len(label)) + b["tr"]
            )
        else:
            top = self.dim(b["tl"] + b["h"] * (self.width - 2) + b["tr"])
        out = [top]
        edge = self.dim(b["v"])
        for line in lines:
            pad = max(0, inner - visible_len(line))
            out.append(f"{edge} {line}{' ' * pad} {edge}")
        out.append(self.dim(b["bl"] + b["h"] * (self.width - 2) + b["br"]))
        return "\n".join(out)


def get() -> Style:
    """A Style for stdout, built on first use (so tests can set env first)."""

    return Style()
