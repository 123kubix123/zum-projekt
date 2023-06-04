import pickle
from threading import Thread
import random
import numpy as np
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist


class DecompositionClassifier:

    def __init__(self, classifier, code_size=6, model_constructor_args={}):
        self.__params = locals()
        self.__params.pop('self')
        self.__classifier = classifier
        self.__model_constructor_args = model_constructor_args
        self.__code_size = code_size

    @staticmethod
    def find_unique_values(values):
        return np.unique(values).tolist()
        #unique_values = []
        #for value in values:
        #    if value not in unique_values:
        #        unique_values.append(value)
        #return unique_values

    def __class_included_in_classifier(self, classifier_no, class_value):
        if self.__ecoc_matrix_[self.__classes_.index(class_value)][classifier_no] == 1:
            return True
        else:
            return False

    def __fit_single_model(self, X, y, classifier_no, random_state, model_fit_args):
        examples = []
        target = []
        model = self.__classifier(**self.__model_constructor_args)

        # assign new class labels
        for i, value in enumerate(y):
            examples.append(X[i])
            if self.__class_included_in_classifier(classifier_no, value):
                target.append(1)
            else:
                target.append(0)

        model.fit(examples, target, **model_fit_args)
        self.__models_[classifier_no] = model

    def fit(self, X, y, random_state=42, **model_fit_args):
        # get all classes into a list
        self.__classes_ = self.find_unique_values(y)
        self.__ecoc_matrix_ = np.zeros((len(self.__classes_), self.__code_size))

        for i in range(0, len(self.__classes_)):
            while True:
                code_for_class = []
                for j in range(0, self.__code_size):
                    code_for_class.append(random.randint(0, 1))
                if code_for_class not in self.__ecoc_matrix_.tolist():
                    self.__ecoc_matrix_[i] = code_for_class
                    break

        self.__models_ = [None] * self.__code_size

        threads = []

        for classifier_no in range(0, self.__code_size):
            t = Thread(target=self.__fit_single_model,
                       args=(X, y, classifier_no, random_state, model_fit_args))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

    @staticmethod
    def __hamming_distance(a, b):
        if len(a) != len(b):
            return False
        distance = 0
        for i in range(0, len(a)):
            if a[i] != b[i]:
                distance += 1
        return distance

    def predict(self, X):

        preds = []
        for model in self.__models_:
            preds.append(model.predict(X))
        prediction_matrix = np.array(preds).T

        distance_matrix = cdist(prediction_matrix, self.__ecoc_matrix_)

        """
        # co tutaj jeśli mamy taki sam min dystans?
        min_distance = np.min(distance_matrix, axis=1)

        diff = (distance_matrix == min_distance.repeat(distance_matrix.shape[1]).reshape(distance_matrix.shape))
        if np.any(np.sum(diff, axis=1) != np.ones(diff.shape[0])):
            # "Ties are broken arbitrarily in favor of the class that comes first in the class ordering."
            # warnings.warn("More than one matching prediction, returning first")
            pass
        """

        predictions_ids = np.argmin(distance_matrix, axis=1)
        predictions = [self.__classes_[id_] for id_ in predictions_ids]

        """
        predictions = []
        for sample in X:
            classification_vector = []
            # go through each model
            for model in self.__models_:
                prediction = model.predict([sample])
                classification_vector.append(prediction[0])

            hamming_distances_for_class_vectors = []
            for i in range(0, len(self.__ecoc_matrix_)):
                hamming_distances_for_class_vectors \
                    .append(self.__hamming_distance(classification_vector, self.__ecoc_matrix_[i]))

            predicted_class_id = hamming_distances_for_class_vectors.index(min(hamming_distances_for_class_vectors))
            prediction = self.__classes_[predicted_class_id]
            predictions.append(prediction)
        """

        return predictions

    def score(self, X, y, sample_weight=None):
        predictions = self.predict(X)
        return accuracy_score(y, predictions, sample_weight=sample_weight)

    def get_params(self, deep=True):
        return self.__params

    def dump(self, filename):
        with open(filename, 'wb') as handle:
            pickle.dump(self, handle)
