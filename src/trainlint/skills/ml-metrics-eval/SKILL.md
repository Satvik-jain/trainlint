---
name: ml-metrics-eval
description: Helps compute and report evaluation metrics correctly for classification and regression. Use when writing evaluation code, calling scikit-learn metrics, computing AUC, F1, precision or recall, aggregating results over batches, picking a threshold, or comparing models. Covers argument order, probabilities versus labels, averaging, class imbalance, baselines and run-to-run variance.
---

# Metrics and evaluation

A wrong metric call returns a plausible number. These are the mistakes worth checking every time.

## Calls

- **Argument order.** scikit-learn metrics take `(y_true, y_pred)`. Accuracy is symmetric, so a
  swap goes unnoticed there, but precision, recall, F1, R² and the confusion matrix change. Use
  keywords: `f1_score(y_true=y_test, y_pred=y_pred)`.
- **Scores, not labels, for ranking metrics.** `roc_auc_score` and `average_precision_score` need
  probabilities or decision scores: `clf.predict_proba(X)[:, 1]` or `clf.decision_function(X)`.
  Passing `predict(X)` collapses the curve to one point and reports a misleading number. For
  multiclass AUC pass the full probability matrix with `multi_class="ovr"`.
- **Averaging.** For more than two classes say which one you use: `macro` weights classes
  equally, `micro` weights samples equally, `weighted` weights by class size. For binary
  metrics, check `pos_label`.
- **Scaled targets.** If the target was standardized, invert the transform before reporting an
  error, so it is in the target's units.

## Aggregating over batches

- Mean of batch means differs from the dataset mean when the last batch is smaller. Accumulate
  sums and counts, then divide once.
- Do not compute F1, precision or AUC per batch and average them. Collect all predictions, then
  compute the metric once.
- Evaluation loaders use `drop_last=False`. Dropped samples are silently unscored.

## Choosing and judging

- **Class imbalance.** With a 95% majority class, 95% accuracy is a model that predicts one
  class. Report a baseline (`DummyClassifier(strategy="most_frequent")`) next to every result,
  and use precision, recall or average precision for the rare class.
- **Thresholds.** Pick the decision threshold on validation data. Compare models with a
  threshold-free metric first.
- **Regression.** R² can be negative. Report an error in the target's units, and compare it to
  predicting the training mean.
- **Variance.** Report mean and standard deviation over at least three seeds or over the
  cross-validation folds. A difference smaller than the spread is not an improvement.
- **Small test sets.** Add a bootstrap confidence interval. 100 test rows cannot separate 90%
  from 93%.
- **One test pass.** Tune on validation data, then evaluate on the test set once.

## Check the code

If `trainlint` is installed, run `trainlint check <file>` after editing. TL011 flags swapped
`y_true` and `y_pred` and class labels passed to `roc_auc_score`, and TL012 flags evaluation loaders
that drop the last batch. Fix each finding, or add `# trainlint: ignore[TL011]` with a comment
saying why it does not apply. If it is not installed, apply the points above by hand.
