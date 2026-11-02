from __future__ import annotations

from trainlint.rules import leakage
from trainlint.rules.base import Hit, Rule

ALL_RULES: tuple[Rule, ...] = tuple(sorted(leakage.RULES, key=lambda rule: rule.id))

__all__ = ["ALL_RULES", "Hit", "Rule"]
