---
name: ml-training-loop
description: Guides writing correct PyTorch training and evaluation loops. Use when writing or reviewing a training loop, loss computation, optimizer step, mixed precision setup, random seeding, or evaluation under no_grad. Covers train and eval mode, zero_grad order, logits versus probabilities, loss accumulation, GradScaler, gradient clipping, shape and dtype checks.
---

# PyTorch training loop

Most training bugs do not raise errors. The loop runs, the loss moves, and the model is worse than
it should be. Write the loop in this order and check the points below.

## Skeleton

```python
torch.manual_seed(seed)

for epoch in range(start_epoch, epochs):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        running += loss.item() * x.size(0)
    scheduler.step()

    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            ...
```

## Points that cause silent bugs

- **Modes.** `model.eval()` switches dropout and batch norm to inference behavior. `torch.no_grad()`
  only stops gradient tracking. Evaluation needs both, and `model.train()` must run again at the
  start of every epoch, after validation.
- **Gradient order.** `zero_grad()`, forward, `backward()`, `step()`. Gradients add up until they
  are cleared, and clearing them between `backward()` and `step()` makes the step ignore the
  batch you just computed.
- **Loss inputs.** `CrossEntropyLoss` takes raw logits of shape (N, C) and int64 class indices.
  `BCEWithLogitsLoss` takes raw logits and float targets. Do not end the model with `softmax` or
  `sigmoid` for these losses. Apply them at inference only.
- **Shapes.** `mse_loss(pred, target)` with `pred` of shape (N, 1) and `target` of shape (N,)
  broadcasts to (N, N) and trains toward the mean. Make the shapes equal and assert them once on
  the first batch.
- **Dtypes.** Inputs are float32, class labels are int64. `torch.from_numpy` on float64 arrays
  gives float64 tensors.
- **Reshaping.** `view` and `reshape` reinterpret memory in order. To swap dimensions use
  `permute` or `transpose`, not `view`.
- **Tracking the loss.** Store `loss.item()` or `loss.detach()`. Adding the loss tensor to a
  running total keeps every iteration's graph in memory.
- **Averages over batches.** The last batch is usually smaller. Weight each batch mean by its
  size, or accumulate sums and divide by the dataset size at the end.

## Seeds and reproducibility

Seed at startup: `torch.manual_seed(seed)` covers CPU and all CUDA devices. Seed `random` and
`numpy` too if the code uses them. For repeatable batch order pass a seeded `torch.Generator` to
the `DataLoader`. Bit-for-bit repeatable GPU runs need
`torch.use_deterministic_algorithms(True)` and, on CUDA, the environment variable
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, and they cost speed. Log the seed with the results.

## Mixed precision

float16 needs a `GradScaler` because small gradients underflow. bfloat16 does not.

```python
scaler = torch.amp.GradScaler("cuda")  # torch.cuda.amp.GradScaler() on older versions

with torch.autocast("cuda", dtype=torch.float16):
    loss = criterion(model(x), y)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)  # before clipping, so the norm is in real units
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)  # not optimizer.step()
scaler.update()
```

## Gradient accumulation

Divide the loss by the number of accumulation steps, call `backward()` every micro-batch, and call
`step()` and `zero_grad()` once every N micro-batches.

## Sanity checks before a long run

- The loss at initialization for C balanced classes should be close to `ln(C)`.
- Train on one batch for 100 steps. The loss should go close to zero. If it does not, the bug is
  in the model, the loss, or the labels, not in the data volume.
- Print one batch's shapes and dtypes for inputs, outputs and targets.

## Check the code

If `trainlint` is installed, run `trainlint check <file>` after editing. Relevant rules: TL003
(eval mode left on), TL004 (no_grad without eval), TL005 (no seed), TL006 (mixed precision),
TL008 (activation before a logits loss), TL009 (gradient clearing), TL010 (loss kept as a tensor).
Fix each finding, or add `# trainlint: ignore[TL00x]` with a comment saying why it does not apply.
If it is not installed, apply the points above by hand.
