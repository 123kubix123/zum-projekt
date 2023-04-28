from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import fetch_covtype
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from DecompositionClassifier import DecompositionClassifier

data = fetch_covtype()

#model = DecompositionClassifier(RandomForestClassifier, max_depth=2, n_estimators=3, max_features=1)
model = DecompositionClassifier(RandomForestClassifier)

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

model.fit(X_train, y_train)
model_oryg = RandomForestClassifier()
model_oryg.fit(X_train, y_train)
#model.print_scores()

#print(data.target[0])
test = X_test.tolist()

our_predictions = model.predict(test)
oryg_predictions = model_oryg.predict(test)

print(f'our acc: {accuracy_score(y_test, our_predictions)}')

print(f'oryg acc: {accuracy_score(y_test, oryg_predictions)}')