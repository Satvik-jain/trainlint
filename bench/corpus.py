"""Load the regression corpus and score trainlint against it.

Every file in buggy/ starts with `# expect: TLxxx` and has a twin in fixed/ with the same
name and the one bug removed. The twin must produce no findings.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from trainlint.engine import lint_paths

ROOT = Path(__file__).parent
_EXPECT = re.compile(r"^#\s*expect:\s*(.+)$")


@dataclass(frozen=True)
class Pair:
    name: str
    expected: frozenset[str]
    buggy: Path
    fixed: Path


@dataclass
class RuleScore:
    cases: int = 0
    caught: int = 0
    false_positives: int = 0


def load_pairs() -> list[Pair]:
    pairs = []
    for buggy in sorted((ROOT / "buggy").glob("*.py")):
        header = _EXPECT.match(buggy.read_text(encoding="utf-8").splitlines()[0])
        if header is None:
            raise ValueError(f"{buggy.name}: first line must be '# expect: TLxxx'")
        expected = frozenset(header.group(1).split())
        pairs.append(Pair(buggy.stem, expected, buggy, ROOT / "fixed" / buggy.name))
    return pairs


def rules_found(path: Path) -> frozenset[str]:
    return frozenset(f.rule for f in lint_paths([path]).findings)


def score(pairs: list[Pair]) -> tuple[dict[str, RuleScore], list[str]]:
    """Per-rule detection counts and a list of every disagreement with the expectations."""
    scores: dict[str, RuleScore] = defaultdict(RuleScore)
    problems = []
    for pair in pairs:
        found = rules_found(pair.buggy)
        for rule in pair.expected:
            scores[rule].cases += 1
            scores[rule].caught += rule in found
        if found != pair.expected:
            problems.append(f"{pair.name}: expected {sorted(pair.expected)}, got {sorted(found)}")
        for rule in rules_found(pair.fixed):
            scores[rule].false_positives += 1
            problems.append(f"{pair.name}: fixed twin still flagged {rule}")
    return dict(sorted(scores.items())), problems
