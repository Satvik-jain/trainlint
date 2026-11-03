# expect: TL013
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("features.csv")
y = df.pop("label")
df = (df - df.mean()) / df.std()
X_train, X_test, y_train, y_test = train_test_split(df, y, random_state=0)
model = LogisticRegression().fit(X_train, y_train)
print(model.score(X_test, y_test))
