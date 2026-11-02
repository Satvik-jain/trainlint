---
name: ml-data-leakage
description: Prevents train/test leakage when writing or reviewing machine learning code. Use when splitting data, building preprocessing or feature selection steps, setting up cross-validation, oversampling, or when a validation score looks too good. Covers scalers, encoders, imputers, SelectKBest, SMOTE, target encoding, time series and grouped data.
---

# Data leakage

Leakage means information from validation or test rows reached the model during training. The
score goes up and real-world performance does not. It never raises an error, so it has to be
prevented by construction.

To see the size of the effect: selecting the best 20 of 5,000 random features on the full dataset
and then cross-validating reports about 88% accuracy on labels that are coin flips. The same
steps inside a pipeline report 50%.

## Rules

1. Split first. Anything learned from data is fitted on the training split only: scaler means,
   imputer values, vocabularies, PCA components, selected features, class weights, resampling.
   Apply it to the other splits with `transform()`.
2. For cross-validation, put preprocessing and feature selection inside a `Pipeline` and pass the
   pipeline to `cross_val_score` or `GridSearchCV`. The pipeline refits every step on each fold.
3. Oversample (SMOTE, duplication) and augment the training split only, after splitting. A
   resampled copy of a row in the test set is a copy of a training row.
4. Match the split to how the model will be used:
   - Rows that share a person, patient, device, session, or document go through
     `GroupKFold` or `GroupShuffleSplit` with that id.
   - Time-ordered data is split by time (`TimeSeriesSplit`), never shuffled, and features use only
     the past.
   - Exact and near duplicates are dropped before splitting or split by a group key.
5. Exclude columns that are computed from the label or recorded after the outcome. A feature that
   dominates the importances deserves suspicion before it deserves credit.
6. Use the test set once. Choose models, thresholds and epoch counts on validation data and
   report on test data a single time at the end.
7. Target encoding and any feature built from the label are computed inside each training fold.
8. Fit tokenizers, vocabularies and normalization statistics on the training split. Pretrained
   embeddings are fine.

## Patterns

```python
# Wrong: the scaler has seen the test rows.
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Right.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scaler = StandardScaler().fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
```

```python
# Right for cross-validation: every step is refit per fold.
pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression())
scores = cross_val_score(pipe, X, y, cv=5)
```

```python
# Oversample after the split, on the training part only.
X_train, y_train = SMOTE(random_state=0).fit_resample(X_train, y_train)
```

## Check a suspicious score

Run the same pipeline with the labels shuffled (`rng.permutation(y)`). A leak-free pipeline scores
at chance. If it scores clearly above chance, something is leaking.

## Check the code

If `trainlint` is installed, run `trainlint check <file>` after editing. TL001 flags
preprocessing fitted before a split and TL002 flags fits on validation or test data. Fix each
finding, or add `# trainlint: ignore[TL001]` with a comment saying why it does not apply. If it is
not installed, apply the rules above by hand.
