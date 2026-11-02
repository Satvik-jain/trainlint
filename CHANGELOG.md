# Changelog

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
