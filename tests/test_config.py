from __future__ import annotations

import argparse
import contextlib
import io
import unittest
from pathlib import Path

from throttle.config import apply_config_defaults


def _make_parser() -> argparse.ArgumentParser:
    """A tiny parser standing in for one of throttle's real subcommands:
    a list-typed option (like --concurrency), a bounded scalar option
    (like --max-tokens), a store_true flag (like --stream), a Path-typed
    option (like --output-dir), and a fixed-length positional (like
    `report`'s two-file `reports` argument), all with the same style of
    validators the real CLI uses.
    """

    def _positive_int(value: str) -> int:
        parsed = int(value)
        if parsed <= 0:
            raise argparse.ArgumentTypeError("must be greater than zero")
        return parsed

    parser = argparse.ArgumentParser(prog="throttle")
    parser.add_argument("--concurrency", nargs="+", type=_positive_int)
    parser.add_argument("--max-tokens", type=_positive_int)
    parser.add_argument("--backend", choices=("native", "guidellm"), default="native")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    return parser


def _make_report_parser() -> argparse.ArgumentParser:
    """Stands in for the real `report` subcommand, whose `reports`
    positional uses nargs=2 rather than "+"/"*".
    """
    parser = argparse.ArgumentParser(prog="throttle report")
    parser.add_argument("reports", nargs=2, type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def _capture_parser_error(parser: argparse.ArgumentParser, config: dict) -> str:
    """Run apply_config_defaults expecting parser.error() to fire, and
    return the printed error message (argparse's error() writes to
    stderr and calls sys.exit(2)).
    """
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        with unittest.TestCase().assertRaises(SystemExit):
            apply_config_defaults(parser, config)
    return stderr.getvalue()


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
        parser = _make_parser()
        with self.assertRaises(SystemExit):
            apply_config_defaults(parser, {"concurrency": 4})

    def test_list_config_value_for_scalar_argument_errors_clearly(self) -> None:
        parser = _make_parser()
        with self.assertRaises(SystemExit):
            apply_config_defaults(parser, {"max_tokens": [1, 2]})

    def test_out_of_range_native_value_is_still_validated(self) -> None:
        parser = _make_parser()
        with self.assertRaises(SystemExit):
            apply_config_defaults(parser, {"max_tokens": -5})

    def test_invalid_choice_from_config_errors_clearly(self) -> None:
        parser = _make_parser()
        with self.assertRaises(SystemExit):
            apply_config_defaults(parser, {"backend": "not-a-real-backend"})

    def test_unknown_config_key_passes_through_unchanged(self) -> None:
        parser = _make_parser()
        apply_config_defaults(parser, {"some_other_tools_setting": "anything"})
        args = parser.parse_args([])
        self.assertEqual(args.some_other_tools_setting, "anything")

    def test_cli_flag_still_overrides_config_value(self) -> None:
        parser = _make_parser()
        apply_config_defaults(parser, {"concurrency": [2, 4]})
        args = parser.parse_args(["--concurrency", "16"])
        self.assertEqual(args.concurrency, [16])

    def test_store_true_bool_from_config_passes_through_unchanged(self) -> None:
        parser = _make_parser()
        apply_config_defaults(parser, {"stream": True})
        args = parser.parse_args([])
        self.assertIs(args.stream, True)

    def test_path_type_value_from_config_is_coerced_to_path(self) -> None:
        parser = _make_parser()
        apply_config_defaults(parser, {"output_dir": "/tmp/results"})
        args = parser.parse_args([])
        self.assertEqual(args.output_dir, Path("/tmp/results"))
        self.assertIsInstance(args.output_dir, Path)

    def test_fixed_nargs_wrong_length_errors_clearly(self) -> None:
        parser = _make_report_parser()
        with self.assertRaises(SystemExit):
            apply_config_defaults(parser, {"reports": ["only_one.json"]})

    def test_fixed_nargs_correct_length_applies(self) -> None:
        parser = _make_report_parser()
        apply_config_defaults(parser, {"reports": ["a.json", "b.json"]})
        args = parser.parse_args(["--out", "/tmp/x.html"])
        self.assertEqual(args.reports, [Path("a.json"), Path("b.json")])

    def test_fixed_nargs_scalar_value_errors_clearly(self) -> None:
        parser = _make_report_parser()
        with self.assertRaises(SystemExit):
            apply_config_defaults(parser, {"reports": "a.json"})

    def test_error_message_format_scalar_for_list_argument(self) -> None:
        parser = _make_parser()
        message = _capture_parser_error(parser, {"concurrency": 4})
        self.assertIn(
            "'concurrency' in ~/.throttle/config.yaml must be a list for "
            "--concurrency (it accepts multiple values); got 4. Use a list "
            "instead, e.g. 'concurrency: [4]'.",
            message,
        )

    def test_error_message_format_out_of_range_value(self) -> None:
        parser = _make_parser()
        message = _capture_parser_error(parser, {"max_tokens": -5})
        self.assertIn(
            "'max_tokens' in ~/.throttle/config.yaml is invalid for "
            "--max-tokens: must be greater than zero",
            message,
        )

    def test_error_message_format_invalid_choice(self) -> None:
        parser = _make_parser()
        message = _capture_parser_error(parser, {"backend": "not-a-real-backend"})
        self.assertIn(
            "'backend' in ~/.throttle/config.yaml is invalid for --backend: "
            "invalid choice 'not-a-real-backend' (choose from 'native', 'guidellm')",
            message,
        )

    def test_error_message_format_fixed_length_mismatch(self) -> None:
        parser = _make_report_parser()
        message = _capture_parser_error(parser, {"reports": ["only_one.json"]})
        self.assertIn(
            "'reports' in ~/.throttle/config.yaml must have exactly 2 values "
            "for positional argument 'reports'; got 1: ['only_one.json'].",
            message,
        )


if __name__ == "__main__":
    unittest.main()
