# trainlint

Agent skills and an offline checker for ML bugs that do not crash.

[![ci](https://github.com/Satvik-jain/trainlint/actions/workflows/ci.yml/badge.svg)](https://github.com/Satvik-jain/trainlint/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Satvik-jain/trainlint/blob/main/LICENSE)

![Cross-validated accuracy on random labels: 85% with a leak, trainlint pointing at it, 48% after the fix](https://raw.githubusercontent.com/Satvik-jain/trainlint/main/assets/demo.gif)

The 5,000 features and the labels are random, so no model can beat 50%. Picking features before
cross-validating reports 85%. The GIF is rendered from the real output of the commands shown;
`assets/make_assets.py` rebuilds it.

## Install

```console
uvx trainlint check .
```

or `pipx install trainlint` / `pip install trainlint`. It needs Python 3.10 or newer, has no
dependencies, and never touches the network or a GPU.

## Quickstart

```console
uvx trainlint check train.py     # find the bugs in a script, a folder, or a repo
uvx trainlint skills install     # teach your coding agent to avoid them
```

`skills install` copies four skills into `./.claude/skills`. Use `--agent agents` for
`./.agents/skills`, `--global` for your user folder, or `--dest DIR` for any other place.

More ways to run the checker:

```console
trainlint check src/ --ignore TL005   # skip a rule
trainlint check . --format json       # for scripts and CI
trainlint rules                       # list every rule
```

Exit code 0 means no findings, 1 means findings, 2 means a usage error. To silence a finding on
one line:

```python
scaler.fit(X_all)  # trainlint: ignore[TL002] fitting on the pooled set is intended here
```

In GitHub Actions:

```yaml
- run: uvx trainlint check .
```

## Why it exists

ML code fails quietly. A scaler fitted before the split, a model left in eval mode, or a softmax
in front of `CrossEntropyLoss` still trains and still prints a number. Coding agents and people
both write these mistakes, and general linters do not know what the code is for. trainlint is a
small offline checker for the common cases, plus skills that tell an agent how to avoid them in
the first place.

## What it does

- Fourteen rules, TL001 to TL014, listed below and explained with examples in
  [docs/rules.md](https://github.com/Satvik-jain/trainlint/blob/main/docs/rules.md).
- Each finding shows the line, a pointer into the code, and a one-line fix.
- Pure standard library. It parses the code and never runs it.
- Text or JSON output, rule selection, and per-line ignore comments.
- Four [Agent Skills](https://agentskills.io/specification) that cover the same ground and run
  the checker when it is installed.

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
| TL013 | Data rescaled with its own mean, std, min or max before a split |
| TL014 | Training `DataLoader` with no `shuffle`, `sampler` or `batch_sampler` |

| Skill | Covers |
| --- | --- |
| `ml-data-leakage` | Splits, pipelines, grouped and time-ordered data, resampling, the test set |
| `ml-training-loop` | Modes, gradient order, loss inputs, shapes, seeds, mixed precision |
| `ml-metrics-eval` | Argument order, scores versus labels, averaging, imbalance, variance |
| `ml-checkpoint-resume` | What a checkpoint must hold, best versus last, verifying a resume |

## How far to trust it

The checks parse each file with Python's `ast` module and match calls by name. There is no type
inference and no cross-file analysis, so:

- Splits are recognized by variable names such as `X_train`, `val_loader` and `y_test`.
- A bug whose cause sits in another file is not seen.
- A heuristic can be wrong. When it is, add an ignore comment and open an issue.

`python -m bench` runs the checker over a corpus of 20 buggy scripts, one or two per rule, each
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
TL013             1       1                             0
TL014             1       1                             0
total            20      20                             0
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

It reports 22 findings. By my reading, 20 are real: 15 scripts never set a seed (one declares a
`--seed` option and never uses it), 3 tensor-parallel scripts never clear gradients, one script
resumes from a snapshot without saving optimizer state, and one trains on a loader that never
shuffles. Two more are correct as written but harmless, because those models have no dropout or
batch norm to leave in training mode. Earlier runs also flagged three distributed-autograd scripts
for missing `zero_grad()` and one torchtext iterator for not shuffling. Both were false positives
and are now fixed. This is one repository of tutorial code and I did the reviewing, so
treat it as a spot check. It tells you nothing about what the rules miss.

## Compared with other tools

| Tool | What it covers | Where trainlint differs |
| --- | --- | --- |
| ruff, pylint, flake8 | General Python style and bugs | None of their rules know about splits, eval mode or metrics. Use them alongside trainlint. |
| [TorchFix](https://github.com/meta-pytorch/torchfix) | Removed and deprecated PyTorch functions, `requires_grad` typos, unsafe `torch.load` | Its README lists no rules for training logic. trainlint does not check API deprecations. |
| [mllint](https://github.com/bvobart/mllint) | Repository-level quality of ML projects: version control, dependencies, code quality | It looks at project hygiene. trainlint reads training and evaluation code. |
| [Deepchecks](https://github.com/deepchecks/deepchecks) | Checks that run on real datasets and models, including drift | Deepchecks needs your data and model loaded. trainlint reads source only and runs nothing. |
| General skill collections | Engineering process for coding agents | They are not specific to ML. trainlint's skills target training and evaluation code and come with a checker. |

## Roadmap

- Jupyter notebook support
- SARIF output for GitHub code scanning
- Settings in `pyproject.toml`
- More rules: `retain_graph` in a loop
- A benchmark that measures how often an agent writes these bugs with and without the skills
  (needs an API key, so it will be optional)

Issues labeled [good first issue](https://github.com/Satvik-jain/trainlint/labels/good%20first%20issue)
are small and have a clear definition of done.

## Contributing

False positives and missed bugs are the most useful reports. See
[CONTRIBUTING.md](https://github.com/Satvik-jain/trainlint/blob/main/CONTRIBUTING.md) for setup and for how to add a rule.

```console
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run python -m bench
```

## License

MIT

If trainlint saved you an afternoon, a star on the repository helps other people find it.
