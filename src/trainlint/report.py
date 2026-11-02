"""Render a Report as terminal text or JSON."""

from __future__ import annotations

import json

from trainlint import __version__
from trainlint.engine import Finding, Report

_CODES = {"bold": "1", "red": "31", "cyan": "36", "dim": "2"}


def _paint(text: str, style: str, enabled: bool) -> str:
    return f"\x1b[{_CODES[style]}m{text}\x1b[0m" if enabled else text


def _caret_line(finding: Finding) -> str:
    # Reuse tabs from the source line so the carets stay under the right characters.
    indent = "".join(c if c == "\t" else " " for c in finding.text[: finding.col - 1])
    return indent + "^" * (finding.end_col - finding.col)


def render_text(report: Report, color: bool = False) -> str:
    out: list[str] = []
    for f in report.findings:
        gutter = " " * len(str(f.line))
        location = _paint(f"{f.path}:{f.line}:{f.col}", "bold", color)
        out.append(f"{location}: {_paint(f.rule, 'red', color)} {f.message}")
        out.append(f"{gutter} |")
        out.append(f"{f.line} | {f.text}")
        out.append(f"{gutter} | {_paint(_caret_line(f), 'red', color)}")
        out.append(f"{gutter} = {_paint('help:', 'cyan', color)} {f.hint}")
        out.append("")
    out.append(_summary(report))
    return "\n".join(out)


def _summary(report: Report) -> str:
    checked = _plural(report.files_checked, "file")
    if not report.findings:
        return f"No problems found in {checked}."
    flagged = len({f.path for f in report.findings})
    return (
        f"Found {_plural(len(report.findings), 'problem')} in "
        f"{_plural(flagged, 'file')} (checked {checked})."
    )


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def render_json(report: Report) -> str:
    findings = [
        {
            "path": f.path,
            "line": f.line,
            "column": f.col,
            "rule": f.rule,
            "message": f.message,
            "hint": f.hint,
        }
        for f in report.findings
    ]
    skipped = [{"path": path, "reason": reason} for path, reason in report.skipped]
    payload = {
        "version": __version__,
        "files_checked": report.files_checked,
        "findings": findings,
        "skipped": skipped,
    }
    return json.dumps(payload, indent=2)
