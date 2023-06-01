from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import fetch_covtype, load_iris, make_classification
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
import sklearn
import time

from DecompositionClassifier import DecompositionClassifier

data = make_classification(50000, 10, n_informative=6, n_classes=5, random_state=42)

model_our = DecompositionClassifier(RandomForestClassifier, code_size=5)
model_oryg = RandomForestClassifier()

""" 
X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.2)

start_time = time.time()
model_our.fit(X_train, y_train)
print("--- %s seconds ---" % (time.time() - start_time))

model_oryg = RandomForestClassifier()
model_oryg.fit(X_train, y_train)

test = X_test.tolist()

our_predictions = model_our.predict(test)
oryg_predictions = model_oryg.predict(test)

print(f'our acc: {accuracy_score(y_test, our_predictions)}')
print(f'oryg acc: {accuracy_score(y_test, oryg_predictions)}')  """


X = data[0]
y = data[1]


start_time = time.time()

scores_our = cross_val_score(model_our, X, y, cv=10)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores_our.mean(), scores_our.std()))

print("--- %s seconds ---" % (time.time() - start_time))

scores_oryg = cross_val_score(model_oryg, X, y, cv=10)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores_oryg.mean(), scores_oryg.std()))
