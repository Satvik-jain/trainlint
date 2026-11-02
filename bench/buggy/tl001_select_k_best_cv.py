# expect: TL001
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)
X, y = rng.normal(size=(100, 5000)), rng.integers(0, 2, size=100)
X = SelectKBest(f_classif, k=20).fit_transform(X, y)
print(cross_val_score(LogisticRegression(), X, y, cv=5).mean())
