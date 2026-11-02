"""TL006: automatic mixed precision set up in a way that loses gradients or steps."""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator

from trainlint.astutil import all_calls, call_tail, dotted, is_const, kwarg, receiver
from trainlint.rules.base import Hit, Rule


def _is_method_call(call: ast.Call, name: str) -> bool:
    return isinstance(call.func, ast.Attribute) and call.func.attr == name


def _runs_in_fp16(autocast: ast.Call) -> bool:
    """True when autocast will use float16: CUDA with no dtype, or an explicit float16."""
    if is_const(kwarg(autocast, "enabled"), False):
        return False
    device = autocast.args[0] if autocast.args else kwarg(autocast, "device_type")
    if isinstance(device, ast.Constant) and device.value != "cuda":
        return False
    dtype = kwarg(autocast, "dtype")
    if dtype is None:
        return True
    return re.search(r"\b(?:float16|half)\b", ast.unparse(dtype)) is not None


def _scaler_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and call_tail(node.value) == "GradScaler"
        ):
            names.update(d for t in node.targets if (d := dotted(t)))
    return names


def _amp_misuse(tree: ast.Module) -> Iterator[Hit]:
    calls = all_calls(tree)
    if not any(_is_method_call(c, "backward") for c in calls):
        return
    creations = [
        c
        for c in calls
        if call_tail(c) == "GradScaler" and not is_const(kwarg(c, "enabled"), False)
    ]
    scale_calls = [c for c in calls if _is_method_call(c, "scale")]

    if not creations:
        for call in calls:
            if call_tail(call) == "autocast" and _runs_in_fp16(call):
                yield Hit(
                    call,
                    "float16 autocast without a GradScaler: small gradients underflow to "
                    "zero and the model can silently stop learning",
                    "Create torch.amp.GradScaler() and use scaler.scale(loss).backward() with "
                    "scaler.step(optimizer), or autocast with dtype=torch.bfloat16.",
                )
        return

    if not scale_calls:
        yield Hit(
            creations[0],
            "a GradScaler is created but scaler.scale(loss) is never called, so it does nothing",
            "Backpropagate scaler.scale(loss) and update with scaler.step(optimizer) and "
            "scaler.update().",
        )
        return

    scalers = _scaler_names(tree)
    stepped = any(_is_method_call(c, "step") and receiver(c) in scalers for c in calls)
    if scalers and not stepped:
        yield Hit(
            scale_calls[0],
            "the loss is scaled but scaler.step(optimizer) is never called, so the optimizer "
            "steps on scaled gradients and skips the overflow check",
            "Replace optimizer.step() with scaler.step(optimizer), then call scaler.update().",
        )


RULES = (Rule("TL006", "Mixed precision without a working GradScaler", _amp_misuse),)
