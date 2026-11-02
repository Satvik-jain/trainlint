import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)
X, y = rng.normal(size=(100, 5000)), rng.integers(0, 2, size=100)
pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression())
print(cross_val_score(pipe, X, y, cv=5).mean())
