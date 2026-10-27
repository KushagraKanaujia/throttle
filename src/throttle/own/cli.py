"""Command registry for the own-your-intelligence loop.

Each stage lives in its own module and exposes two functions::

    def add_parser(subparsers) -> None   # registers its subcommand(s)
    def handle(args) -> int              # runs it; dispatched on args.command

Keeping registration here means stages can land as separate PRs without all
editing ``throttle/cli.py``.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
from types import ModuleType

# command name -> module path. Order is the order shown in --help.
STAGES: dict[str, str] = {
    "capture": "throttle.own.capture",
    "workloads": "throttle.own.workloads",
    "shadow": "throttle.own.shadow",
    "eval": "throttle.own.evaluate",
    "report": "throttle.own.report",
    "route": "throttle.own.route",
    "refresh": "throttle.own.refresh",
    "demo": "throttle.own.demo",
}


def _available() -> dict[str, ModuleType]:
    modules: dict[str, ModuleType] = {}
    for command, path in STAGES.items():
        if importlib.util.find_spec(path) is None:
            continue
        modules[command] = importlib.import_module(path)
    return modules


def register(subparsers: argparse._SubParsersAction) -> None:
    seen: set[int] = set()
    for command, module in _available().items():
        # A legacy command with the same name wins until its stage PR
        # removes it; argparse refuses duplicate subcommand names.
        if command in subparsers.choices or id(module) in seen:
            continue
        seen.add(id(module))
        module.add_parser(subparsers)


def commands() -> set[str]:
    return set(_available())


def dispatch(args: argparse.Namespace) -> int:
    module = _available()[args.command]
    return int(module.handle(args))
