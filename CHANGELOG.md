# Changelog

## 0.2.0 - 2026-11-03

- TL013 flags data rescaled with its own mean, std, min or max before a split (thanks to
  @jay23606). Per-row statistics such as `axis=1` are not flagged.
- TL014 flags a training `DataLoader` with no `shuffle`, `sampler` or `batch_sampler`, so batches
  arrive in dataset order every epoch (thanks to @jay23606). Evaluation loaders and loaders over
  datasets named like an iterator or stream are not flagged.
- Regression corpus is now 20 buggy scripts with fixed twins.

## 0.1.1 - 2026-11-03

- README links and the demo image are absolute, so they work on the PyPI page.

## 0.1.0 - 2026-11-03

First release.

- `trainlint check` with 12 rules (TL001 to TL012) for data leakage, train and eval mode,
  seeding, mixed precision, checkpoints, loss inputs, gradient clearing, tensor accumulation,
  metric misuse and evaluation loaders. Text and JSON output, `--select`, `--ignore`,
  `--exit-zero`, `--color`, and `# trainlint: ignore[RULE]` comments.
- Four Agent Skills (`ml-data-leakage`, `ml-training-loop`, `ml-metrics-eval`,
  `ml-checkpoint-resume`) and `trainlint skills install`.
- Regression corpus of 18 buggy scripts with fixed twins (`python -m bench`).
- Per-rule reference in `docs/rules.md`.
