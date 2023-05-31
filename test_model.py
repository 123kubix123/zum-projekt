from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import fetch_covtype, load_iris
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
import sklearn

from DecompositionClassifier import DecompositionClassifier

data = load_iris()

#model = DecompositionClassifier(RandomForestClassifier, max_depth=2, n_estimators=3, max_features=1)
model_our = DecompositionClassifier(RandomForestClassifier, code_size=5)
model_oryg = RandomForestClassifier()

#X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
X = data.data.tolist()
y = data.target.tolist()

""" model.fit(X_train, y_train)
model_oryg = RandomForestClassifier()
model_oryg.fit(X_train, y_train)
model.print_scores()


#print(data.target[0])
test = X_test.tolist()

our_predictions = model.predict(test)
oryg_predictions = model_oryg.predict(test)

print(f'our acc: {accuracy_score(y_test, our_predictions)}')

print(f'oryg acc: {accuracy_score(y_test, oryg_predictions)}') """

#print(model_oryg.get_params())
#print(model_our.get_params())

scores_our = cross_val_score(model_our, X, y, cv=10)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores_our.mean(), scores_our.std()))


scores_oryg = cross_val_score(model_oryg, X, y, cv=10)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores_oryg.mean(), scores_oryg.std()))
