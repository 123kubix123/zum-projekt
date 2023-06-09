import itertools
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
        self.name = f'DecompositionClassifier{classifier.__name__}_{code_size}Fold'

    def get_name(self):
        return self.name
    

    def print_test(self):
        print("test")

    @staticmethod
    def find_unique_values(values):
        return np.unique(values).tolist()

    @staticmethod
    def exhaustive_codes(k, n):
        """
        When 3 <= k(lasy) <= 7, we construct a code of length 2^{k-1}-1 as follows. Row 1 is all ones. Row 2
        consists of 2{k-2} zeroes followed by 2{k-2}-1 ones. Row 3 consists of 2{k-3}zeroes, followed by 2{k-3}
        ones, followed by 2{k-3} zeroes, followed by 2{k-3}-1 ones.
        In row i, there are alternating runs of 2{k-i} zeroes and ones.
        :param k: class size
        :param n: code_size
        :return:
        """
        codes = []
        for i in range(k):
            ones = [1 for i in range(2 ** (k - i))]
            zeros = [0 for i in range(2 ** (k - i))]

            if i%2 == 0:
                tmp = (ones + zeros) * (i+1)
            else:
                tmp = (zeros + ones) * (i)
            codes.append(tmp[:n].copy())
        return codes

    @staticmethod
    def rand_hill_climb_codes(k,n):
        """
        k >=11
        :param k: class number
        :param n: code_size
        :return:
        """
        classes_codes_dec = random.sample(range(2 ** n), int(np.ceil(k/2)))
        codes = []
        for code in classes_codes_dec:
            # convert random number to bits and to list
            c_list = list(f'{code:0{n}b}')
            codes.append([int(c) for c in c_list])
            codes.append([1-int(c) for c in c_list])
        return codes[:k]

    def __class_included_in_classifier(self, classifier_no, class_value):
        if self.__ecoc_matrix_[self.__classes_.index(class_value)][classifier_no] == 1:
            return True
        else:
            return False

    def __fit_single_model(self, X, y, classifier_no, random_state, model_fit_args):
        model = self.__classifier(**self.__model_constructor_args)
        target = [1 if self.__class_included_in_classifier(classifier_no, value) else 0 for value in y]
        model.fit(X, target, **model_fit_args)
        self.__models_[classifier_no] = model

    def fit(self, X, y, random_state=42, **model_fit_args):
        # get all classes into a list
        self.__classes_ = self.find_unique_values(y)
        code_size = self.__code_size
        class_len = len(self.__classes_)
        ecoc_matrix = []

        if 2**code_size < class_len:
            raise ValueError("Code size too small")

        if class_len <= 7:
            ecoc_matrix = self.exhaustive_codes(class_len, code_size)
        else:
            ecoc_matrix = self.rand_hill_climb_codes(class_len, code_size)

        self.__ecoc_matrix_ = ecoc_matrix
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

        return predictions

    def score(self, X, y, sample_weight=None):
        predictions = self.predict(X)
        return accuracy_score(y, predictions, sample_weight=sample_weight)

    def get_params(self, deep=True):
        return self.__params

    def dump(self, filename):
        with open(filename, 'wb') as handle:
            pickle.dump(self, handle)
