from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from trainlint import __version__
from trainlint.engine import lint_paths, select_rules
from trainlint.report import render_json, render_text
from trainlint.rules import ALL_RULES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trainlint",
        description="Find ML training and evaluation bugs that do not crash.",
    )
    parser.add_argument("--version", action="version", version=f"trainlint {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check", help="check Python files or directories")
    check.add_argument("paths", nargs="*", type=Path, default=[Path()], help="default: .")
    check.add_argument("--select", metavar="RULES", help="comma-separated rule ids to run")
    check.add_argument("--ignore", metavar="RULES", help="comma-separated rule ids to skip")
    check.add_argument("--format", choices=("text", "json"), default="text")
    check.add_argument("--exit-zero", action="store_true", help="exit 0 even if problems are found")
    check.set_defaults(run=_check)

    rules = commands.add_parser("rules", help="list the available rules")
    rules.set_defaults(run=_rules)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
    if stdout_reconfigure:
        stdout_reconfigure(errors="replace")
    try:
        return args.run(args)
    except (FileNotFoundError, ValueError) as error:
        print(f"trainlint: error: {error}", file=sys.stderr)
        return 2


def _split(value: str | None) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()] if value else []


def _check(args: argparse.Namespace) -> int:
    rules = select_rules(_split(args.select), _split(args.ignore))
    report = lint_paths(args.paths, rules)
    for path, reason in report.skipped:
        print(f"trainlint: skipped {path} ({reason})", file=sys.stderr)
    if args.format == "json":
        print(render_json(report))
    else:
        print(render_text(report, color=_use_color()))
    return 1 if report.findings and not args.exit_zero else 0


def _rules(_: argparse.Namespace) -> int:
    for rule in ALL_RULES:
        print(f"{rule.id}  {rule.title}")
    return 0


def _use_color() -> bool:
    if "NO_COLOR" in os.environ or not sys.stdout.isatty():
        return False
    return os.name != "nt" or _enable_windows_ansi()


def _enable_windows_ansi() -> bool:
    import ctypes

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.GetStdHandle(-11)
    mode = ctypes.c_ulong()
    if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
        return False
    return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
