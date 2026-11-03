"""TL001, TL002 and TL013: held-out rows influencing the model."""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator

from trainlint.astutil import (
    by_position,
    call_tail,
    call_targets,
    calls_in,
    is_eval_name,
    is_train_name,
    receiver,
    root_name,
    scopes,
    target_names,
    walk_scope,
)
from trainlint.rules.base import Hit, Rule

# Methods that always fit on their argument and return something derived from it.
_FIT_AND_RETURN = frozenset({"fit_transform", "fit_resample"})
_FIT_METHODS = _FIT_AND_RETURN | {"fit"}
_SPLITTING_CALLS = frozenset(
    {"train_test_split", "cross_val_score", "cross_validate", "cross_val_predict", "split"}
)
# Encoding labels on the full dataset is routine and leaks nothing the model can use.
_LABEL_NAME = re.compile(r"^(?:y|labels?|targets?)(?:_|\d|$)", re.IGNORECASE)
# Statistics that rescale a column by its own range or spread.
_ROW_STATS = frozenset({"mean", "std", "min", "max"})


def _describe(call: ast.Call) -> str:
    return f"{ast.unparse(call.func)}(...)"


def _fit_before_split(tree: ast.Module) -> Iterator[Hit]:
    for scope in scopes(tree):
        nodes = list(walk_scope(scope))
        calls = calls_in(nodes)
        bound = call_targets(nodes)
        fits = list(_full_data_fits(calls, bound))
        for index, (fit, data, produced) in enumerate(fits):
            tainted = {data, *produced}
            for _, later_data, later_produced in fits[index + 1 :]:
                if later_data in tainted:
                    tainted |= later_produced
            splitter = next(
                (
                    c
                    for c in calls
                    if call_tail(c) in _SPLITTING_CALLS
                    and by_position([fit, c])[0] is fit
                    and c is not fit
                    and any(root_name(arg) in tainted for arg in c.args)
                ),
                None,
            )
            if splitter is not None:
                yield Hit(
                    fit,
                    f"`{_describe(fit)}` runs on all rows before `{call_tail(splitter)}()` "
                    "splits them, so validation and test rows shape the transform",
                    "Split first and fit on the training part only, or put the step in a "
                    "Pipeline so cross-validation refits it on each fold.",
                )


def _full_data_fits(
    calls: list[ast.Call], bound: dict[int, set[str]]
) -> Iterator[tuple[ast.Call, str, set[str]]]:
    """Fit calls on data that is not obviously a training split, with names they taint."""
    for call in calls:
        tail = call_tail(call)
        if tail not in _FIT_METHODS or not isinstance(call.func, ast.Attribute) or not call.args:
            continue
        data = root_name(call.args[0])
        if data is None or is_train_name(data) or is_eval_name(data) or _LABEL_NAME.match(data):
            continue
        produced = set(bound.get(id(call), ()))
        if tail == "fit":
            # A bare fit() only prepares data when the same object transforms it afterwards.
            transforms = [
                c
                for c in calls
                if call_tail(c) == "transform"
                and receiver(c) == receiver(call)
                and by_position([call, c])[0] is call
                and c is not call
            ]
            if not transforms:
                continue
            for transform in transforms:
                produced |= bound.get(id(transform), set())
        yield call, data, produced


def _fit_on_held_out(tree: ast.Module) -> Iterator[Hit]:
    for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
        if call_tail(call) not in _FIT_METHODS or not isinstance(call.func, ast.Attribute):
            continue
        data = root_name(call.args[0]) if call.args else None
        if data and is_eval_name(data) and not is_train_name(data):
            yield Hit(
                call,
                f"`{_describe(call)}` is fitted on `{data}`, which is held-out data",
                "Fit on the training split only. Use the held-out split for transform() "
                "or predict().",
            )


def _self_scaled(value: ast.expr) -> set[str]:
    """Names that `value` rescales by their own mean, std, min or max, as in x - x.mean()."""
    if not isinstance(value, ast.BinOp):
        return set()
    stats = [
        call
        for call in ast.walk(value)
        if isinstance(call, ast.Call)
        and call_tail(call) in _ROW_STATS
        and isinstance(call.func, ast.Attribute)
    ]
    inside_stats = {id(node) for call in stats for node in ast.walk(call)}
    operands = {
        node.id
        for node in ast.walk(value)
        if isinstance(node, ast.Name) and id(node) not in inside_stats
    }
    return {root for call in stats if (root := root_name(call.func.value)) in operands}


def _scaled_before_split(tree: ast.Module) -> Iterator[Hit]:
    for scope in scopes(tree):
        nodes = list(walk_scope(scope))
        calls = calls_in(nodes)
        for assign in by_position(n for n in nodes if isinstance(n, ast.Assign)):
            if not _self_scaled(assign.value):
                continue
            produced = {name for target in assign.targets for name in target_names(target)}
            splitter = next(
                (
                    c
                    for c in calls
                    if call_tail(c) in _SPLITTING_CALLS
                    and by_position([assign, c])[0] is assign
                    and any(root_name(arg) in produced for arg in c.args)
                ),
                None,
            )
            if splitter is not None:
                yield Hit(
                    assign,
                    f"`{ast.unparse(assign.targets[0])}` is rescaled with statistics of all "
                    f"rows before `{call_tail(splitter)}()` splits them, so validation and "
                    "test rows shape the scale",
                    "Split first and compute the mean, std, min or max on the training part "
                    "only, then apply it to each part.",
                )


RULES = (
    Rule("TL001", "Preprocessing is fitted on all rows before the split", _fit_before_split),
    Rule("TL002", "Transform or model is fitted on a validation or test split", _fit_on_held_out),
    Rule(
        "TL013",
        "Data is rescaled with its own statistics before the split",
        _scaled_before_split,
    ),
)
