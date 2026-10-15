"""Configuration file loading for Throttle.

Loads config from ~/.throttle/config.yaml if it exists.
CLI flags always override config file values.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# YAML is not a required dependency - fail gracefully if not installed
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


DEFAULT_CONFIG_PATH = Path.home() / ".throttle" / "config.yaml"


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses ~/.throttle/config.yaml

    Returns:
        Dictionary of config values, empty if file doesn't exist or YAML not available
    """
    if not YAML_AVAILABLE:
        return {}

    path = config_path or DEFAULT_CONFIG_PATH

    if not path.exists():
        return {}

    try:
        with open(path) as f:
            config = yaml.safe_load(f)

        if config is None:
            return {}

        if not isinstance(config, dict):
            print(
                f"Warning: Config file {path} does not contain a valid YAML dictionary. Ignoring.",
                file=sys.stderr,
            )
            return {}

        return config

    except yaml.YAMLError as e:
        print(
            f"Warning: Failed to parse config file {path}: {e}. Ignoring.",
            file=sys.stderr,
        )
        return {}
    except OSError as e:
        print(
            f"Warning: Failed to read config file {path}: {e}. Ignoring.",
            file=sys.stderr,
        )
        return {}


def _find_action(parser: Any, dest: str) -> Any:
    """Find the argparse action for a given dest name, if this parser has one."""
    for action in parser._actions:
        if getattr(action, "dest", None) == dest:
            return action
    return None


def _coerce_config_value(parser: Any, action: Any, key: str, value: Any) -> Any:
    """Re-run an action's type/choices/shape validation against a config value.

    argparse only re-applies an action's ``type`` (and ``choices``) check to
    STRING defaults; a value the YAML loader already parsed into a native
    Python type (int, float, bool, list) skips that check entirely when set
    via ``set_defaults``. That lets a malformed config value (wrong shape,
    out of range, wrong type) flow straight into the program with no error
    at the boundary, surfacing later as a confusing crash or silent
    misbehavior far from the actual cause. Re-validating here, the same way
    a CLI flag would be validated, keeps that failure at the boundary and
    the error message actionable.
    """
    option_strings = getattr(action, "option_strings", None) or []
    arg_display = option_strings[0] if option_strings else f"positional argument '{key}'"

    nargs = getattr(action, "nargs", None)
    # "+"/"*" accept any number of values (>= 1 or >= 0). A positive integer
    # nargs (e.g. nargs=2 for `report`'s two-file positional) requires
    # exactly that many. nargs=0 (store_true/store_false/store_const) is
    # not a list action, those pass their value through unchanged below.
    is_variable_list = nargs in ("+", "*")
    exact_length = nargs if isinstance(nargs, int) and nargs > 0 else None
    is_list_action = is_variable_list or exact_length is not None

    if is_list_action:
        if not isinstance(value, (list, tuple)):
            expected = "multiple values" if is_variable_list else f"exactly {exact_length} values"
            parser.error(
                f"'{key}' in ~/.throttle/config.yaml must be a list for {arg_display} "
                f"(it accepts {expected}); got {value!r}. Use a list instead, "
                f"e.g. '{key}: [{value!r}]'."
            )
        items = list(value)
        if exact_length is not None and len(items) != exact_length:
            parser.error(
                f"'{key}' in ~/.throttle/config.yaml must have exactly {exact_length} "
                f"values for {arg_display}; got {len(items)}: {items!r}."
            )
    else:
        if isinstance(value, (list, tuple)):
            parser.error(
                f"'{key}' in ~/.throttle/config.yaml must be a single value for "
                f"{arg_display}, not a list; got {value!r}."
            )
        items = [value]

    coerced_items = []
    type_fn = getattr(action, "type", None)
    for item in items:
        coerced = item
        if callable(type_fn):
            try:
                coerced = type_fn(str(item))
            except (ValueError, TypeError, argparse.ArgumentTypeError) as exc:
                parser.error(
                    f"'{key}' in ~/.throttle/config.yaml is invalid for {arg_display}: {exc}"
                )
        choices = getattr(action, "choices", None)
        if choices and coerced not in choices:
            parser.error(
                f"'{key}' in ~/.throttle/config.yaml is invalid for {arg_display}: "
                f"invalid choice {coerced!r} (choose from {', '.join(map(repr, choices))})"
            )
        coerced_items.append(coerced)

    return coerced_items if is_list_action else coerced_items[0]


def _apply_validated_defaults(target: Any, defaults: dict[str, Any]) -> None:
    """Set defaults on one parser (main or subparser), validating each value
    against its matching action first. Keys with no matching action on this
    parser are passed through unchanged, matching the previous behavior.
    """
    validated: dict[str, Any] = {}
    for dest, value in defaults.items():
        action = _find_action(target, dest)
        if action is None:
            validated[dest] = value
            continue
        validated[dest] = _coerce_config_value(target, action, dest, value)

    target.set_defaults(**validated)

    # Set required=False for args present in config (João's fix)
    for action in target._actions:
        if hasattr(action, "required") and action.required and hasattr(action, "dest"):
            if action.dest in validated:
                action.required = False


def apply_config_defaults(parser: Any, config: dict[str, Any]) -> None:
    """Apply config values as argument parser defaults.

    CLI arguments will override these defaults naturally through argparse.
    Works with both main parsers and parsers with subparsers. Each value is
    validated against its argument's own type/choices/shape requirements
    before being applied, so a malformed config value errors clearly here
    instead of silently reaching the rest of the program.

    Args:
        parser: argparse.ArgumentParser instance
        config: Dictionary of config values
    """
    if not config:
        return

    # Convert config keys to CLI argument names (replace - with _)
    defaults = {}
    for key, value in config.items():
        # Skip None values
        if value is None:
            continue

        # Convert kebab-case to snake_case for argparse
        arg_name = key.replace("-", "_")
        defaults[arg_name] = value

    # Apply defaults to the main parser
    _apply_validated_defaults(parser, defaults)

    # Also apply to all subparsers if they exist
    if getattr(parser, "_subparsers", None) is not None:
        for action in parser._subparsers._actions:
            if hasattr(action, "choices") and action.choices:
                for subparser in action.choices.values():
                    _apply_validated_defaults(subparser, defaults)
