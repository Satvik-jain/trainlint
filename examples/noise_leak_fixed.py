# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn"]
# ///
"""The same experiment with feature selection inside the cross-validation loop."""

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)
X, y = rng.normal(size=(100, 5000)), rng.integers(0, 2, size=100)

pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression())
scores = cross_val_score(pipe, X, y, cv=5)
print(f"cross-validated accuracy on random labels: {scores.mean():.0%}")
