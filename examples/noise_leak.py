# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn"]
# ///
"""Cross-validated accuracy on labels that are pure noise, with a leak in the feature selection.

The 5,000 features and the labels are random, so no model can beat 50%. Picking the 20
"best" features on the full dataset and then cross-validating reports far more.
"""

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

rng = np.random.default_rng(0)
X, y = rng.normal(size=(100, 5000)), rng.integers(0, 2, size=100)

X = SelectKBest(f_classif, k=20).fit_transform(X, y)
scores = cross_val_score(LogisticRegression(), X, y, cv=5)
print(f"cross-validated accuracy on random labels: {scores.mean():.0%}")
