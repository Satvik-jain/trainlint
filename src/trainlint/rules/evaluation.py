"""TL011 and TL012: evaluation code that measures the wrong thing."""

from __future__ import annotations

import ast
from collections.abc import Iterator

from trainlint.astutil import (
    by_position,
    call_tail,
    call_targets,
    calls_in,
    is_const,
    is_eval_name,
    is_train_name,
    kwarg,
    receiver,
    root_name,
    scopes,
    walk_scope,
)
from trainlint.rules.base import Hit, Rule

# Metrics where swapping y_true and y_pred changes the answer.
_ORDER_SENSITIVE = frozenset(
    {
        "f1_score",
        "fbeta_score",
        "precision_score",
        "recall_score",
        "precision_recall_fscore_support",
        "balanced_accuracy_score",
        "confusion_matrix",
        "classification_report",
        "r2_score",
        "explained_variance_score",
        "log_loss",
    }
)
_NEEDS_SCORES = frozenset({"roc_auc_score", "average_precision_score"})
_HARD_LABEL_CALLS = frozenset({"predict", "argmax"})
# Boosting libraries return probabilities from Booster.predict for binary objectives.
_BOOSTER_HINTS = ("booster", "bst", "gbm", "lgb", "xgb", "catboost")


def _swapped_arguments(call: ast.Call) -> bool:
    if len(call.args) < 2 or call_tail(call) not in _ORDER_SENSITIVE | _NEEDS_SCORES:
        return False
    first, second = root_name(call.args[0]), root_name(call.args[1])
    return bool(first and second and "pred" in first.lower() and "pred" not in second.lower())


def _latest_value(nodes: list[ast.AST], name: str, before: ast.AST) -> ast.expr | None:
    assignments = [
        n
        for n in nodes
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
        and (n.lineno, n.col_offset) < (before.lineno, before.col_offset)
    ]
    return by_position(assignments)[-1].value if assignments else None


def _is_hard_labels(value: ast.expr | None) -> bool:
    if not isinstance(value, ast.Call) or call_tail(value) not in _HARD_LABEL_CALLS:
        return False
    owner = (receiver(value) or "").lower()
    return not any(hint in owner for hint in _BOOSTER_HINTS)


def _metric_misuse(tree: ast.Module) -> Iterator[Hit]:
    for scope in scopes(tree):
        nodes = list(walk_scope(scope))
        for call in calls_in(nodes):
            tail = call_tail(call)
            if _swapped_arguments(call):
                yield Hit(
                    call,
                    f"`{tail}` takes (y_true, y_pred), but the first argument looks like "
                    "predictions",
                    "Pass the ground-truth labels first and the predictions second.",
                )
            elif tail in _NEEDS_SCORES and len(call.args) >= 2:
                scores = call.args[1]
                if isinstance(scores, ast.Name):
                    scores = _latest_value(nodes, scores.id, call) or scores
                if _is_hard_labels(scores):
                    yield Hit(
                        call,
                        f"`{tail}` is given class labels from predict(), which collapses the "
                        "curve to a single point",
                        "Pass scores instead: predict_proba(X)[:, 1] or decision_function(X).",
                    )


def _eval_loader_drops_samples(tree: ast.Module) -> Iterator[Hit]:
    for scope in scopes(tree):
        nodes = list(walk_scope(scope))
        bound = call_targets(nodes)
        for call in calls_in(nodes):
            if call_tail(call) != "DataLoader" or not is_const(kwarg(call, "drop_last"), True):
                continue
            dataset = root_name(call.args[0]) if call.args else None
            names = [*bound.get(id(call), ()), *([dataset] if dataset else [])]
            if any(is_eval_name(n) and not is_train_name(n) for n in names):
                yield Hit(
                    call,
                    "this evaluation DataLoader uses drop_last=True, so the last partial "
                    "batch is never scored",
                    "Use drop_last=False for validation and test loaders.",
                )


RULES = (
    Rule("TL011", "Metric called with swapped arguments or hard labels", _metric_misuse),
    Rule("TL012", "Evaluation DataLoader drops the last batch", _eval_loader_drops_samples),
)
