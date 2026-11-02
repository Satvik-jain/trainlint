"""TL007: checkpoints that cannot restore a training run."""

from __future__ import annotations

import ast
from collections.abc import Iterator

from trainlint.astutil import all_calls, call_tail, dotted
from trainlint.rules.base import Hit, Rule

_OPTIMIZER_CLASSES = frozenset({"sgd", "adam", "adamw", "rmsprop", "adagrad", "adadelta", "lbfgs"})


def _makes_optimizer(call: ast.Call) -> bool:
    func = ast.unparse(call.func).lower()
    return "optim" in func or (call_tail(call) or "").lower() in _OPTIMIZER_CLASSES


def _words(node: ast.AST) -> str:
    """The identifier or string a node carries, if any."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.arg):
        return node.arg
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return ""


def _mentions_resume(tree: ast.Module) -> bool:
    words = (_words(node).lower() for node in ast.walk(tree))
    return any("resume" in w or "start_epoch" in w for w in words)


def _saves_optimizer_state(calls: list[ast.Call]) -> bool:
    return any(
        call_tail(c) == "state_dict"
        and isinstance(c.func, ast.Attribute)
        and "opt" in ast.unparse(c.func.value).lower()
        for c in calls
    )


def _resume_without_optimizer_state(tree: ast.Module) -> Iterator[Hit]:
    calls = all_calls(tree)
    saves = [c for c in calls if dotted(c.func) == "torch.save"]
    if not saves or not _mentions_resume(tree):
        return
    if not any(_makes_optimizer(c) for c in calls) or _saves_optimizer_state(calls):
        return
    yield Hit(
        saves[0],
        "this script resumes from checkpoints but never saves optimizer state, so momentum "
        "and adaptive learning rates restart from zero after a resume",
        "Save model, optimizer, scheduler, GradScaler and epoch together, for example "
        "{'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}.",
    )


RULES = (
    Rule(
        "TL007",
        "Resumable script saves checkpoints without optimizer state",
        _resume_without_optimizer_state,
    ),
)
