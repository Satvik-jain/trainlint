"""TL003, TL004, TL005, TL008, TL009, TL010 and TL014: mistakes inside the training loop."""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable, Iterator

from trainlint.astutil import (
    all_calls,
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

_LOOPS = (ast.For, ast.While, ast.AsyncFor)
_STREAM_NAME = re.compile(r"iter|stream", re.IGNORECASE)
# DataLoader(dataset, batch_size, shuffle, sampler, ...): the third positional argument is shuffle.
_ORDER_KEYWORDS = frozenset({"shuffle", "sampler", "batch_sampler"})
_ORDER_POSITION = 2


def _is_method_call(call: ast.Call, name: str) -> bool:
    return isinstance(call.func, ast.Attribute) and call.func.attr == name


def _imports_torch(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(
            a.name.split(".")[0] == "torch" for a in node.names
        ):
            return True
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "torch":
            return True
    return False


def _eval_without_train(tree: ast.Module) -> Iterator[Hit]:
    reported: set[int] = set()
    for loop in (n for n in ast.walk(tree) if isinstance(n, _LOOPS)):
        calls = all_calls(loop)
        trains = any(_is_method_call(c, "train") for c in calls)
        if trains or not any(_is_method_call(c, "backward") for c in calls):
            continue
        for call in calls:
            if _is_method_call(call, "eval") and not call.args and id(call) not in reported:
                reported.add(id(call))
                yield Hit(
                    call,
                    f"`{ast.unparse(call.func)}()` runs inside a training loop that never calls "
                    "train() again, so every epoch after the first trains in eval mode",
                    "Call model.train() at the start of each epoch, after validation.",
                )


def _no_grad_without_eval(tree: ast.Module) -> Iterator[Hit]:
    calls = all_calls(tree)
    if not any(_is_method_call(c, "backward") for c in calls):
        return
    if any(_is_method_call(c, "eval") for c in calls):
        return
    if any(_is_method_call(c, "train") and c.args and is_const(c.args[0], False) for c in calls):
        return
    guards = [
        n
        for n in ast.walk(tree)
        if (isinstance(n, ast.Attribute) and n.attr in {"no_grad", "inference_mode"})
        or (isinstance(n, ast.Name) and n.id in {"no_grad", "inference_mode"})
    ]
    if guards:
        yield Hit(
            by_position(guards)[0],
            "evaluation runs under no_grad but the model is never put in eval mode, so "
            "dropout stays active and batch norm keeps updating its running statistics",
            "Call model.eval() before the evaluation loop. no_grad only disables gradients.",
        )


def _is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    test = node.test
    return (
        isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and any(is_const(c, "__main__") for c in test.comparators)
    )


def _unseeded_training(tree: ast.Module) -> Iterator[Hit]:
    if not _imports_torch(tree):
        return
    calls = all_calls(tree)
    backward = [c for c in calls if _is_method_call(c, "backward")]
    if not backward:
        return
    mentions_seed = any("seed" in (call_tail(c) or "").lower() for c in calls) or any(
        "seed" in (k.arg or "").lower() for c in calls for k in c.keywords
    )
    if mentions_seed:
        return
    top_level = {id(n) for n in walk_scope(tree)}
    runs_as_script = any(_is_main_guard(n) for n in ast.walk(tree)) or any(
        id(c) in top_level for c in backward
    )
    if runs_as_script:
        yield Hit(
            backward[0],
            "this training script never sets a random seed, so two runs cannot be compared",
            "Seed at startup, for example torch.manual_seed(seed), and log the seed you used.",
        )


_ACTIVATIONS = {
    "softmax": ("CrossEntropyLoss", "cross_entropy"),
    "sigmoid": ("BCEWithLogitsLoss", "binary_cross_entropy_with_logits"),
}


def _final_activations(tree: ast.Module) -> Iterator[tuple[ast.expr, str]]:
    """Softmax or sigmoid applied last in forward(), or last in an nn.Sequential."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "forward" and node.body:
            last = node.body[-1]
            if isinstance(last, ast.Return) and isinstance(last.value, ast.Call):
                kind = (call_tail(last.value) or "").lower()
                if kind in _ACTIVATIONS:
                    yield last.value, kind
        elif isinstance(node, ast.Call) and call_tail(node) == "Sequential" and node.args:
            final = node.args[-1]
            kind = (call_tail(final) or "").lower() if isinstance(final, ast.Call) else ""
            if kind in _ACTIVATIONS:
                yield final, kind


def _activation_before_logit_loss(tree: ast.Module) -> Iterator[Hit]:
    used = {call_tail(c) for c in all_calls(tree)}
    for node, kind in _final_activations(tree):
        loss = next((name for name in _ACTIVATIONS[kind] if name in used), None)
        if loss is not None:
            yield Hit(
                node,
                f"the model ends in {kind} but this file trains with {loss}, which expects "
                "raw logits and applies the activation itself",
                f"Return logits from the model and apply {kind} only when you need "
                "probabilities, for example at inference.",
            )


def _statement_blocks(tree: ast.Module) -> Iterator[list[ast.stmt]]:
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if isinstance(block, list) and block and isinstance(block[0], ast.stmt):
                yield block


def _call_statement(stmt: ast.stmt) -> ast.Call | None:
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        return stmt.value
    return None


def _zero_grad_between_backward_and_step(tree: ast.Module) -> Iterator[Hit]:
    for block in _statement_blocks(tree):
        after_backward = False
        pending: ast.Call | None = None
        for stmt in block:
            call = _call_statement(stmt)
            if call is None:
                continue
            tail = call_tail(call)
            if tail == "backward":
                after_backward, pending = True, None
            elif tail == "zero_grad" and after_backward:
                pending = call
            elif tail == "step" and pending is not None and receiver(call) == receiver(pending):
                yield Hit(
                    pending,
                    "zero_grad() runs between backward() and step(), so the optimizer steps "
                    "with the gradients already cleared",
                    "Call zero_grad() before backward() or right after step().",
                )
                pending = None


def _clears_gradients(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and call_tail(node) == "zero_grad":
            return True
        if (
            isinstance(node, ast.Call)
            and _is_method_call(node, "zero_")
            and (receiver(node) or "").endswith(".grad")
        ):
            return True
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Attribute) and t.attr == "grad" for t in node.targets
        ):
            return True
    return False


def _gradients_never_cleared(tree: ast.Module) -> Iterator[Hit]:
    if not _imports_torch(tree) or _clears_gradients(tree):
        return
    calls = all_calls(tree)
    # Distributed autograd keeps gradients per context, so there is nothing to clear.
    backward = [
        c
        for c in calls
        if _is_method_call(c, "backward")
        and not (receiver(c) or "").endswith(("dist_autograd", "distributed.autograd"))
    ]
    if backward and any(_is_method_call(c, "step") for c in calls):
        yield Hit(
            backward[0],
            "gradients are never cleared, so every step adds to the gradients of all earlier steps",
            "Call optimizer.zero_grad() once per iteration.",
        )


def _backpropagated_names(calls: Iterable[ast.Call]) -> set[str]:
    """Variables passed to backward(), as in loss.backward() and scaler.scale(loss).backward()."""
    names: set[str] = set()
    for call in calls:
        if not _is_method_call(call, "backward"):
            continue
        target = call.func.value
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Call):
            names.update(a.id for a in target.args if isinstance(a, ast.Name))
    return names


def _plain_uses(node: ast.AST, names: set[str]) -> Iterator[ast.Name]:
    """Uses of `names` that keep the tensor, not loss.item(), loss.detach() or float(loss)."""
    if isinstance(node, ast.Name):
        if node.id in names:
            yield node
    elif isinstance(node, ast.Attribute) or (
        isinstance(node, ast.Call) and call_tail(node) in {"float", "int"}
    ):
        return
    else:
        for child in ast.iter_child_nodes(node):
            yield from _plain_uses(child, names)


def _loss_kept_with_graph(tree: ast.Module) -> Iterator[Hit]:
    for scope in scopes(tree):
        nodes = list(walk_scope(scope))
        losses = _backpropagated_names(calls_in(nodes))
        if not losses:
            continue
        for node in nodes:
            if isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Add | ast.Sub):
                values: list[ast.expr] = [node.value]
            elif isinstance(node, ast.Call) and _is_method_call(node, "append"):
                values = list(node.args)
            else:
                continue
            kept = next((use for v in values for use in _plain_uses(v, losses)), None)
            if kept is not None:
                yield Hit(
                    node,
                    f"`{kept.id}` is stored as a tensor, which keeps the autograd graph of "
                    "every iteration alive and grows memory",
                    f"Store {kept.id}.item() or {kept.id}.detach() instead.",
                )


def _gradient_clearing(tree: ast.Module) -> Iterator[Hit]:
    yield from _gradients_never_cleared(tree)
    yield from _zero_grad_between_backward_and_step(tree)


def _chooses_batch_order(call: ast.Call) -> bool:
    # A **kwargs splat may carry shuffle or sampler, so it counts as choosing one.
    return len(call.args) > _ORDER_POSITION or any(
        k.arg is None or k.arg in _ORDER_KEYWORDS for k in call.keywords
    )


def _streams_data(call: ast.Call) -> bool:
    # Iterable-style datasets cannot be shuffled by the loader, so a dataset argument named
    # like an iterator or stream is left alone.
    first = call.args[0] if call.args else kwarg(call, "dataset")
    name = root_name(first) if first is not None else None
    return name is not None and _STREAM_NAME.search(name) is not None


def _training_loader_never_shuffles(tree: ast.Module) -> Iterator[Hit]:
    for scope in scopes(tree):
        nodes = list(walk_scope(scope))
        bound = call_targets(nodes)
        for call in calls_in(nodes):
            if call_tail(call) != "DataLoader" or _chooses_batch_order(call):
                continue
            if _streams_data(call):
                continue
            names = bound.get(id(call), set())
            if any(is_train_name(n) and not is_eval_name(n) for n in names):
                yield Hit(
                    call,
                    "this training DataLoader has no shuffle, sampler or batch_sampler, so every "
                    "epoch feeds batches in dataset order",
                    "Pass shuffle=True for training loaders.",
                )


RULES = (
    Rule(
        "TL003",
        "eval() inside the training loop with no train() to switch back",
        _eval_without_train,
    ),
    Rule(
        "TL004",
        "Evaluation under no_grad but the model is never put in eval mode",
        _no_grad_without_eval,
    ),
    Rule("TL005", "Training script never sets a random seed", _unseeded_training),
    Rule(
        "TL008",
        "Softmax or sigmoid applied before a loss that expects logits",
        _activation_before_logit_loss,
    ),
    Rule(
        "TL009", "Gradients never cleared, or cleared between backward and step", _gradient_clearing
    ),
    Rule(
        "TL010", "Loss stored as a tensor, keeping the autograd graph alive", _loss_kept_with_graph
    ),
    Rule(
        "TL014",
        "Training DataLoader that never shuffles",
        _training_loader_never_shuffles,
    ),
)
