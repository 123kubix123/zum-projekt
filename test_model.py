from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_covtype, load_iris, make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
import time
import pandas as pd
from DecompositionClassifier import DecompositionClassifier
import pickle


def synth_data(size=3000):
    data = make_classification(size, 10, n_informative=6, n_classes=4, random_state=42)
    y = data[1]
    X = data[0]
    return X, y


def load_letter(n=None):
    data = pd.read_csv('datasets/letter/letter.csv')
    if n:
        data = data.sample(n)
    y = data['Class'].values.tolist()
    X = data.drop('Class', axis=1).values.tolist()
    print('Successfully loaded letters dataset.')
    return X, y


def load_fars(n=None):
    data = pd.read_csv('datasets/fars/fars.csv')
    if n:
        data = data.sample(n)
    y = data['INJURY_SEVERITY'].values.tolist()
    X = data.drop('INJURY_SEVERITY', axis=1).values.tolist()
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(X)
    X = enc.transform(X).toarray()
    print('Successfully loaded fars dataset.')
    return X, y

def load_thyroid(n=None):
    data = pd.read_csv('datasets/thyroid/thyroid.csv')
    if n:
        data = data.sample(n)
    y = data['Class'].values.tolist()
    X = data.drop('Class', axis=1).values.tolist()
    print('Successfully loaded thyroid dataset.')
    return X, y


def run_single(model_our, model_base, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    r_state = 123
    start_time = time.time()
    model_our.fit(X_train, y_train, random_state=r_state)
    print("--- %s seconds ---" % (time.time() - start_time))

    model_base = RandomForestClassifier(random_state=r_state)
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

    scores_our = cross_val_score(model_our, X, y, cv=5)
    print("our: %0.2f accuracy with a standard deviation of %0.2f" % (scores_our.mean(), scores_our.std()))
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    scores_base = cross_val_score(model_base, X, y, cv=5)
    print("base: %0.2f accuracy with a standard deviation of %0.2f" % (scores_base.mean(), scores_base.std()))
    print("--- %s seconds ---" % (time.time() - start_time))
    return scores_our, scores_base

def test_model(Classifier, code_size, X, y):
    decomp_model = DecompositionClassifier(Classifier, code_size)
    pure_model = Classifier()
    scores_our, scores_base = run_k_fold(decomp_model, pure_model, X, y)
    return scores_our, scores_base


#model_our_path = './tmp_model_our.picle'

model_our = DecompositionClassifier(RandomForestClassifier, code_size=5)
model_base = RandomForestClassifier()

#X, y = load_letter()
X, y = load_thyroid()
#X, y = load_fars()
#run_k_fold(model_our, model_base, X, y)

# zapis
#with open(model_our_path, 'wb') as handle:
#    pickle.dump(model_our, handle)

# wczytywanie
#with open(model_our_path, 'rb') as handle:
#    model = pickle.load(handle)



from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


#X, y = load_letter(n=5000)

classifiers = [
    KNeighborsClassifier,
    SVC,
    GaussianProcessClassifier,
    DecisionTreeClassifier,
    RandomForestClassifier,
    MLPClassifier,
    AdaBoostClassifier,
    GaussianNB,
    QuadraticDiscriminantAnalysis,
]

#test_model(GaussianProcessClassifier, 5, X, y)

