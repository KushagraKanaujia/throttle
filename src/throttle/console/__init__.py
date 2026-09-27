"""Throttle Console: a local, private web UI over `throttle check` history.

`throttle ui` serves it on 127.0.0.1. It reads the check history file and the
docs bundled with the package. It sends no traffic to any endpoint and makes
no network requests of its own; the page loads nothing from outside.
"""

from .server import create_app, run_ui

__all__ = ["create_app", "run_ui"]
