from __future__ import annotations

import ast
from collections.abc import Callable, Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class Hit:
    """A spot a rule objects to, before the engine attaches file and line text."""

    node: ast.expr | ast.stmt
    message: str
    hint: str


@dataclass(frozen=True)
class Rule:
    id: str
    title: str
    check: Callable[[ast.Module], Iterator[Hit]]
