from __future__ import annotations

from trainlint.rules import checkpoints, evaluation, leakage, mixed_precision, training
from trainlint.rules.base import Hit, Rule

ALL_RULES: tuple[Rule, ...] = tuple(
    sorted(
        [
            *leakage.RULES,
            *training.RULES,
            *mixed_precision.RULES,
            *checkpoints.RULES,
            *evaluation.RULES,
        ],
        key=lambda rule: rule.id,
    )
)

__all__ = ["ALL_RULES", "Hit", "Rule"]
