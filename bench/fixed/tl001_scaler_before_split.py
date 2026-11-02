from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scaler = StandardScaler().fit(X_train)
model = LogisticRegression().fit(scaler.transform(X_train), y_train)
print(model.score(scaler.transform(X_test), y_test))
