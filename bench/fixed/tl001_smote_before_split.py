from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=2000, weights=[0.95], random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
X_train, y_train = SMOTE(random_state=0).fit_resample(X_train, y_train)
model = RandomForestClassifier(random_state=0).fit(X_train, y_train)
print(model.score(X_test, y_test))
