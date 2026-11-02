"""Parse files, run rules, apply inline suppressions."""

from __future__ import annotations

import ast
import os
import re
import tokenize
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from trainlint.rules import ALL_RULES, Rule

SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "__pycache__",
        ".tox",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "build",
        "dist",
        "site-packages",
    }
)

_IGNORE_COMMENT = re.compile(r"#\s*trainlint:\s*ignore(?:\[([^\]]*)\])?", re.IGNORECASE)
_LINE_BREAK = re.compile(r"\r\n|\r|\n")


@dataclass(frozen=True, order=True)
class Finding:
    path: str
    line: int
    col: int
    end_col: int
    rule: str
    message: str
    hint: str
    text: str


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    files_checked: int = 0


def select_rules(select: Iterable[str] = (), ignore: Iterable[str] = ()) -> list[Rule]:
    """Rules to run. An empty `select` means every rule."""
    known = {rule.id for rule in ALL_RULES}
    wanted = {code.upper() for code in select}
    dropped = {code.upper() for code in ignore}
    unknown = sorted((wanted | dropped) - known)
    if unknown:
        raise ValueError(f"unknown rule {', '.join(unknown)} (see `trainlint rules`)")
    return [r for r in ALL_RULES if (not wanted or r.id in wanted) and r.id not in dropped]


def lint_source(source: str, path: str, rules: Sequence[Rule] = ALL_RULES) -> list[Finding]:
    """Check one file's source. Raises SyntaxError if it does not parse."""
    tree = ast.parse(source, filename=path)
    lines = _LINE_BREAK.split(source)
    findings = []
    for rule in rules:
        for hit in rule.check(tree):
            line = hit.node.lineno
            text = lines[line - 1]
            if _suppressed(text, rule.id):
                continue
            col, end_col = _columns(hit.node, text)
            findings.append(Finding(path, line, col, end_col, rule.id, hit.message, hit.hint, text))
    return sorted(findings)


def lint_paths(paths: Iterable[Path], rules: Sequence[Rule] = ALL_RULES) -> Report:
    report = Report()
    for file in python_files(paths):
        name = file.as_posix()
        try:
            with tokenize.open(file) as handle:
                source = handle.read()
            report.findings.extend(lint_source(source, name, rules))
        except (SyntaxError, UnicodeDecodeError, ValueError) as error:
            report.skipped.append((name, f"{type(error).__name__}: {error}"))
            continue
        report.files_checked += 1
    report.findings.sort()
    return report


def python_files(paths: Iterable[Path]) -> Iterator[Path]:
    """Files named on the command line, plus *.py found under directories."""
    for path in paths:
        if path.is_dir():
            for root, dirs, files in os.walk(path):
                dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
                for name in sorted(files):
                    if name.endswith(".py"):
                        yield Path(root, name)
        elif path.is_file():
            yield path
        else:
            raise FileNotFoundError(f"no such file or directory: {path}")


def _suppressed(line_text: str, rule_id: str) -> bool:
    match = _IGNORE_COMMENT.search(line_text)
    if match is None:
        return False
    codes = match.group(1)
    return codes is None or rule_id in {code.strip().upper() for code in codes.split(",")}


def _columns(node: ast.AST, text: str) -> tuple[int, int]:
    """1-based start and end columns in characters. ast offsets are UTF-8 bytes."""
    start = node.col_offset
    end = node.end_col_offset
    if node.end_lineno != node.lineno:
        end = len(text.encode("utf-8"))
    raw = text.encode("utf-8")
    start_col = len(raw[:start].decode("utf-8", errors="ignore")) + 1
    end_col = len(raw[:end].decode("utf-8", errors="ignore")) + 1
    return start_col, max(end_col, start_col + 1)
