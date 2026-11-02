"""Helpers for matching calls by name. Imports are not resolved, so `F.softmax`,
`torch.softmax` and a bare `softmax` all have the tail `softmax`."""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable, Iterator

_NESTED_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)

# Variable names are the only signal available for which split a value belongs to.
_TRAIN_NAME = re.compile(r"(?:^|_)(?:x|y)?(?:train|trn|tr)(?:$|_|\d)", re.IGNORECASE)
_EVAL_NAME = re.compile(
    r"(?:^|_)(?:x|y)?(?:test|val|valid|validation|eval|dev|holdout)(?:$|_|\d)", re.IGNORECASE
)


def dotted(node: ast.AST) -> str | None:
    """Return 'a.b.c' for a chain of names and attributes, otherwise None."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def call_tail(call: ast.Call) -> str | None:
    """Last component of the callee: 'softmax' for both F.softmax(x) and softmax(x)."""
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def receiver(call: ast.Call) -> str | None:
    """Dotted name a method is called on: 'self.opt' for self.opt.step()."""
    if isinstance(call.func, ast.Attribute):
        return dotted(call.func.value)
    return None


def root_name(node: ast.AST) -> str | None:
    """Base variable of X, X[idx], X.values and X[cols].values."""
    while isinstance(node, ast.Attribute | ast.Subscript):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def target_names(target: ast.AST) -> list[str]:
    """Names bound by an assignment target, including tuple unpacking."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Tuple | ast.List):
        return [name for elt in target.elts for name in target_names(elt)]
    return []


def kwarg(call: ast.Call, name: str) -> ast.expr | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def is_const(node: ast.AST | None, value: object) -> bool:
    return isinstance(node, ast.Constant) and node.value is value


def is_train_name(name: str) -> bool:
    return bool(_TRAIN_NAME.search(name))


def is_eval_name(name: str) -> bool:
    return bool(_EVAL_NAME.search(name))


def by_position(nodes: Iterable[ast.AST]) -> list[ast.AST]:
    return sorted(nodes, key=lambda n: (n.lineno, n.col_offset))


def scopes(tree: ast.Module) -> Iterator[ast.Module | ast.FunctionDef | ast.AsyncFunctionDef]:
    """The module and every function in it."""
    yield tree
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            yield node


def walk_scope(scope: ast.AST) -> Iterator[ast.AST]:
    """Walk a scope without entering nested functions or classes."""
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, _NESTED_SCOPES):
            stack.extend(ast.iter_child_nodes(node))


def calls_in(nodes: Iterable[ast.AST]) -> list[ast.Call]:
    """Calls among `nodes` in source order."""
    return by_position(n for n in nodes if isinstance(n, ast.Call))


def all_calls(tree: ast.AST) -> list[ast.Call]:
    return calls_in(ast.walk(tree))
