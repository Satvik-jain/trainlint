# expect: TL001
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=2000, weights=[0.95], random_state=0)
X_res, y_res = SMOTE(random_state=0).fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, random_state=0)
model = RandomForestClassifier(random_state=0).fit(X_train, y_train)
print(model.score(X_test, y_test))
