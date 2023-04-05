from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
import numpy as np

from DecompositionClassifier import DecompositionClassifier

data = load_wine()

model = DecompositionClassifier(RandomForestClassifier, max_depth=2, n_estimators=3, max_features=1)

model.fit(data.data, data.target)

#model.print_scores()

#print(data.target[0])
test = data.data.tolist()

predictions = model.predict(test)
for i, pred in enumerate(predictions):
    print(str(pred) + ' ?= ' + str(data.target[i]))