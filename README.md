# trainlint

Agent skills and an offline checker for ML bugs that do not crash.

Cross-validated accuracy on 5,000 random features and random labels, where no model can beat 50%:

```console
$ uv run examples/noise_leak.py
cross-validated accuracy on random labels: 85%

$ uvx trainlint check examples/noise_leak.py
examples/noise_leak.py:19:5: TL001 `SelectKBest(f_classif, k=20).fit_transform(...)` runs on all rows before `cross_val_score()` splits them, so validation and test rows shape the transform
   |
19 | X = SelectKBest(f_classif, k=20).fit_transform(X, y)
   |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   = help: Split first and fit on the training part only, or put the step in a Pipeline so cross-validation refits it on each fold.

Found 1 problem in 1 file (checked 1 file).

$ uv run examples/noise_leak_fixed.py
cross-validated accuracy on random labels: 48%
```

## Install

```console
uvx trainlint check .
```

or `pipx install trainlint` / `pip install trainlint`. It needs Python 3.10 or newer, has no
dependencies, and never touches the network or a GPU.

## Quickstart

```console
trainlint check train.py            # one file
trainlint check src/ --ignore TL005 # a folder, skipping a rule
trainlint check . --format json     # for scripts and CI
trainlint rules                     # list every rule
```

Exit code 0 means no findings, 1 means findings, 2 means a usage error. Silence a finding on one
line with a comment:

```python
scaler.fit(X_all)  # trainlint: ignore[TL002] fitting on the pooled set is intended here
```

To run it in GitHub Actions:

```yaml
- run: uvx trainlint check .
```

## What it checks

| Rule | Flags |
| --- | --- |
| TL001 | Scaler, encoder, selector or resampler fitted on all rows before a split or cross-validation |
| TL002 | Transform or model fitted on a validation or test split |
| TL003 | `model.eval()` inside a training loop with no `model.train()` to switch back |
| TL004 | Evaluation under `no_grad` in a script that never calls `model.eval()` |
| TL005 | Training script that never sets a random seed |
| TL006 | float16 autocast without a `GradScaler`, or a scaled loss stepped without `scaler.step` |
| TL007 | Script that resumes from checkpoints but saves no optimizer state |
| TL008 | `softmax` or `sigmoid` at the end of a model trained with a logits loss |
| TL009 | Gradients never cleared, or cleared between `backward()` and `step()` |
| TL010 | Loss added to a running total or list as a tensor, keeping every graph alive |
| TL011 | `y_true` and `y_pred` swapped, or class labels passed to `roc_auc_score` |
| TL012 | Validation or test `DataLoader` with `drop_last=True` |

Each finding prints the line, a pointer to the code, and a one-line fix.

## Agent skills

Coding agents write most of these bugs without noticing. The package ships four skills in the
[Agent Skills](https://agentskills.io/specification) format that tell the agent what to do
instead and to run the checker afterwards:

| Skill | Covers |
| --- | --- |
| `ml-data-leakage` | Splits, pipelines, grouped and time-ordered data, resampling, the test set |
| `ml-training-loop` | Modes, gradient order, loss inputs, shapes, seeds, mixed precision |
| `ml-metrics-eval` | Argument order, scores versus labels, averaging, imbalance, variance |
| `ml-checkpoint-resume` | What a checkpoint must hold, best versus last, verifying a resume |

```console
trainlint skills install                  # into ./.claude/skills
trainlint skills install --agent agents   # into ./.agents/skills
trainlint skills install --global         # into your user folder instead
trainlint skills install --dest some/dir  # anywhere else
```

The skills work without the checker installed. They only use it when it is there.

## How far to trust it

The checks parse each file with Python's `ast` module and match calls by name. There is no type
inference and no cross-file analysis, so:

- Splits are recognized by variable names such as `X_train`, `val_loader` and `y_test`.
- A bug whose cause sits in another file is not seen.
- A heuristic can be wrong. When it is, add an ignore comment and open an issue.

`python -m bench` runs the checker over a corpus of 18 buggy scripts, one or two per rule, each
with a fixed twin:

```
rule    buggy files  caught   false alarms on fixed twins
TL001             3       3                             0
TL002             1       1                             0
TL003             1       1                             0
TL004             1       1                             0
TL005             1       1                             0
TL006             2       2                             0
TL007             1       1                             0
TL008             2       2                             0
TL009             2       2                             0
TL010             1       1                             0
TL011             2       2                             0
TL012             1       1                             0
total            18      18                             0
```

I wrote that corpus together with the rules, so it guards against regressions and says nothing
about accuracy on code I have not seen.

For a check on someone else's code, I ran it over all 93 Python files of
[pytorch/examples](https://github.com/pytorch/examples) at commit `acc295d` and read every
finding:

```console
git clone https://github.com/pytorch/examples && cd examples
git checkout acc295dc7b90714f1bf47f06004fc19a7fe235c4
trainlint check .
```

It reports 21 findings. By my reading, 19 are real: 15 scripts never set a seed (one declares a
`--seed` option and never uses it), 3 tensor-parallel scripts never clear gradients, and one script
resumes from a snapshot without saving optimizer state. Two more are correct as written but
harmless, because those models have no dropout or batch norm to leave in training mode. The first
run also flagged three distributed-autograd scripts for missing `zero_grad()`, which was a false
positive that is now fixed. This is one repository of tutorial code and I did the reviewing, so
treat it as a spot check. It tells you nothing about what the rules miss.

## Development

```console
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run python -m bench
```

## License

MIT
