from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_covtype, load_iris, make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
import time
import pandas as pd
from DecompositionClassifier import DecompositionClassifier
import pickle


def synth_data(size=3000):
    data = make_classification(size, 10, n_informative=6, n_classes=5, random_state=42)
    y = data[1]
    X = data[0]
    return X, y


def load_letter(n=None):
    data = pd.read_csv('datasets/letter/letter.csv')
    if n:
        data = data.sample(n)
    y = data['Class'].values.tolist()
    X = data.drop('Class', axis=1).values.tolist()
    print('Successfully loaded letters')
    return X, y


def run_single(model_our, model_base, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    start_time = time.time()
    model_our.fit(X_train, y_train)
    print("--- %s seconds ---" % (time.time() - start_time))

    model_base = RandomForestClassifier()
    model_base.fit(X_train, y_train)

    start_time = time.time()
    our_predictions = model_our.predict(X_test)
    print(f'our acc: {accuracy_score(y_test, our_predictions)}')
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    base_predictions = model_base.predict(X_test)
    print(f'base acc: {accuracy_score(y_test, base_predictions)}')
    print("--- %s seconds ---" % (time.time() - start_time))


def run_k_fold(model_our, model_base, X, y):
    start_time = time.time()

    scores_our = cross_val_score(model_our, X, y, cv=10)
    print("our: %0.2f accuracy with a standard deviation of %0.2f" % (scores_our.mean(), scores_our.std()))
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    scores_base = cross_val_score(model_base, X, y, cv=10)
    print("base: %0.2f accuracy with a standard deviation of %0.2f" % (scores_base.mean(), scores_base.std()))
    print("--- %s seconds ---" % (time.time() - start_time))


model_our_path = './tmp_model_our.picle'

model_our = DecompositionClassifier(RandomForestClassifier, code_size=5)
model_base = RandomForestClassifier()

X, y = load_letter(n=5000)
run_k_fold(model_our, model_base, X, y)

# zapis
with open(model_base_path, 'wb') as handle:
    pickle.dump(model_our, handle)

# wczytywanie
with open(model_base_path, 'rb') as handle:
    model = pickle.load(handle)
