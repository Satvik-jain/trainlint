"""Print per-rule detection results: `python -m bench`."""

from __future__ import annotations

import sys

from bench.corpus import load_pairs, score


def main() -> int:
    scores, problems = score(load_pairs())
    print(f"{'rule':<7}{'buggy files':>12}{'caught':>8}{'false alarms on fixed twins':>30}")
    for rule, s in scores.items():
        print(f"{rule:<7}{s.cases:>12}{s.caught:>8}{s.false_positives:>30}")
    total_cases = sum(s.cases for s in scores.values())
    total_caught = sum(s.caught for s in scores.values())
    total_false = sum(s.false_positives for s in scores.values())
    print(f"{'total':<7}{total_cases:>12}{total_caught:>8}{total_false:>30}")
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
