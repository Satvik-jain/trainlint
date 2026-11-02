---
name: ml-checkpoint-resume
description: Covers saving and loading training checkpoints so a run resumes exactly where it stopped. Use when writing checkpoint, save, load, resume or early stopping code in PyTorch. Covers what a checkpoint must contain (model, optimizer, scheduler, GradScaler, epoch, RNG state), atomic writes, best versus last checkpoints, map_location, and verifying that a resume reproduces the loss curve.
---

# Checkpoints and resuming

A checkpoint that holds only model weights can load, but it cannot resume. Optimizer moments, the
learning-rate schedule and the epoch counter restart from zero, the loss jumps, and the run is no
longer the one you started.

## What to save

```python
def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_metric):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),  # only if you use a GradScaler
        "epoch": epoch,
        "best_metric": best_metric,
        "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()},
    }
    tmp = f"{path}.tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)  # a crash mid-write keeps the old file
```

Also store the config or hyperparameters, so a checkpoint can be understood without the code
that wrote it.

## Two files, two purposes

Write `last.pt` every epoch and use it to resume. Write `best.pt` only when the validation metric
improves and use it for evaluation and deployment. After early stopping, load `best.pt` before
running the test set, or the test score belongs to the last epoch, not the best one.

## Loading

```python
checkpoint = torch.load(path, map_location=device)
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
scheduler.load_state_dict(checkpoint["scheduler"])
start_epoch = checkpoint["epoch"] + 1
```

- Build the model, move it to the device, then create the optimizer, then load all the states.
- `map_location` lets a GPU checkpoint load on a CPU machine.
- Recent PyTorch versions load with `weights_only=True` by default, which refuses arbitrary
  pickled objects. Only turn it off for files you wrote yourself.
- A model wrapped in `DataParallel` or `DistributedDataParallel` saves keys prefixed with
  `module.`. Save `model.module.state_dict()` so the file loads into a plain model.
- Early stopping state (best metric, patience counter) belongs in the checkpoint too, or a
  resume resets the patience.

## Verify the resume

Train for 4 epochs in one process and record the losses. Train 2 epochs, stop, resume from the
checkpoint in a new process, and train 2 more. On CPU with fixed seeds the losses should match.
If they diverge, something is missing from the checkpoint. Batch order is a common culprit: seed the
`DataLoader` generator with `seed + epoch` so a resumed epoch sees the same batches.

## Check the code

If `trainlint` is installed, run `trainlint check <file>` after editing. TL007 flags scripts that
resume from checkpoints but never save optimizer state. Fix the finding, or add
`# trainlint: ignore[TL007]` with a comment saying why it does not apply. If it is not installed,
apply the points above by hand.
