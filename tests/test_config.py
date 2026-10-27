from __future__ import annotations

import argparse
import unittest

from throttle.config import apply_config_defaults


def _make_parser() -> argparse.ArgumentParser:
    """A tiny parser standing in for one of throttle's real subcommands:
    one list-typed option (like --concurrency) and one bounded scalar
    option (like --max-tokens), both with the same style of type
    validator the real CLI uses.
    """

    def _positive_int(value: str) -> int:
        parsed = int(value)
        if parsed <= 0:
            raise argparse.ArgumentTypeError("must be greater than zero")
        return parsed

    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", nargs="+", type=_positive_int)
    parser.add_argument("--max-tokens", type=_positive_int)
    parser.add_argument("--backend", choices=("native", "other"), default="native")
    return parser


class ConfigValidationTests(unittest.TestCase):
    def test_valid_list_value_applies_as_default(self) -> None:
        parser = _make_parser()
        apply_config_defaults(parser, {"concurrency": [2, 4, 8]})
        args = parser.parse_args([])
        self.assertEqual(args.concurrency, [2, 4, 8])

    def test_valid_scalar_value_applies_as_default(self) -> None:
        parser = _make_parser()
        apply_config_defaults(parser, {"max_tokens": 256})
        args = parser.parse_args([])
        self.assertEqual(args.max_tokens, 256)

    def test_scalar_config_value_for_list_argument_errors_clearly(self) -> None:
        # The natural mistake: writing `concurrency: 4` instead of
        # `concurrency: [4]` for an argument defined with nargs="+".
        # Previously this reached argparse un-coerced and crashed later
        # wherever the caller assumed args.concurrency was a list.
        parser = _make_parser()
        with self.assertRaises(SystemExit):
            apply_config_defaults(parser, {"concurrency": 4})

    def test_list_config_value_for_scalar_argument_errors_clearly(self) -> None:
        parser = _make_parser()
        with self.assertRaises(SystemExit):
            apply_config_defaults(parser, {"max_tokens": [1, 2]})

    def test_out_of_range_native_value_is_still_validated(self) -> None:
        # YAML parses `max_tokens: -5` directly into a Python int, so it
        # never passes through the argument's type() as a string the way
        # a CLI-supplied value or a string config value would. Without
        # re-validation this bypasses _positive_int entirely.
        parser = _make_parser()
        with self.assertRaises(SystemExit):
            apply_config_defaults(parser, {"max_tokens": -5})

    def test_invalid_choice_from_config_errors_clearly(self) -> None:
        parser = _make_parser()
        with self.assertRaises(SystemExit):
            apply_config_defaults(parser, {"backend": "not-a-real-backend"})

    def test_unknown_config_key_passes_through_unchanged(self) -> None:
        # A key with no matching argument on this parser (e.g. it belongs
        # to a different subcommand) should not be rejected here.
        parser = _make_parser()
        apply_config_defaults(parser, {"some_other_tools_setting": "anything"})
        args = parser.parse_args([])
        self.assertEqual(args.some_other_tools_setting, "anything")

    def test_cli_flag_still_overrides_config_value(self) -> None:
        parser = _make_parser()
        apply_config_defaults(parser, {"concurrency": [2, 4]})
        args = parser.parse_args(["--concurrency", "16"])
        self.assertEqual(args.concurrency, [16])


if __name__ == "__main__":
    unittest.main()
