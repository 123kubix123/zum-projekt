from sklearn.model_selection import train_test_split
from threading import Thread
import random
import numpy as np


class DecompositionClassifier:
  __models = []

  def __init__(self, classifier, code_size = 6, shuffle_data=False, **model_constructor_args):
    self.shuffle_data=shuffle_data
    self.classifier = classifier
    self.model_constructor_args = model_constructor_args
    self.code_size = code_size

  @staticmethod
  def find_unique_values(list):
    unique_values = []
    for value in list:
        if value not in unique_values:
            unique_values.append(value)
    return unique_values

  def __fit_single_model(self, X, y, classifier_no, test_size, random_state, model_fit_args):
 #     print(X)
  #    print(y)
      examples = []
      target = []
      model =  self.classifier(**self.model_constructor_args)

      # assign new class labels
      for i, value in enumerate(y):
        examples.append(X[i])
        if value == c: # TODO: i'm here, i have classifier_no and ecoc matrix, now need to check for class id in given row
          target.append(1)
        else:
          target.append(0)
      
      # train test split
      X_train, X_test, y_train, y_test = train_test_split(examples, target, test_size=test_size, random_state=random_state)

      # fit model
      model['classifier'].fit(X_train, y_train, **model_fit_args)
      model['score'] = model['classifier'].score(X_test, y_test)

      self.__models.append(model)

  def fit(self, X, y, test_size = 0.2, random_state=42, **model_fit_args):
    # get all classes into a list
    self.classes = self.find_unique_values(y)

    self.ecoc_matrix = np.zeros((len(self.classes), self.code_size))
    threads = []

    for i in range(0, len(self.classes)):
      while True:
        code_for_class = []
        for j in range(0,self.code_size):
            code_for_class.append(random.randint(0, 1))
        if code_for_class not in self.ecoc_matrix.tolist():
            self.ecoc_matrix[i] = code_for_class
            break

    self.__models = [None] * self.code_size

    for classifier_no in range(0, self.code_size):
      self.__fit_single_model(X, y, classifier_no, test_size, random_state, model_fit_args)   

    #   t = Thread(target=self.__fit_single_model, args=(X, y, c, test_size, random_state, model_fit_args))
    #   threads.append(t)
    #   t.start()
    # for t in threads:
    #   t.join()

  def print_scores(self):
    for model in self.__models:
      print(str(model['class']) + ' : ' +str(model['score']))

  def predict(self, X):
    predictions = []
    for sample in X:
      class_assignments_for_sample = []
      # go through each model
      for model in self.__models:
        pred = model['classifier'].predict([sample])
        #gather only true predictions
        if pred == 1:
          class_assignments_for_sample.append({'class': model['class'], 'score': model['score']})
      # if there is no model that classified the sample assign class of the model with worse score
      if len(class_assignments_for_sample) == 0:
        predictions.append(min(self.__models, key=lambda val: val['score'])['class'])
      # if one or more models predicted some class, assign class of best scoring model
      else:
        predictions.append(max(class_assignments_for_sample, key=lambda val:val['score'])['class'])
    return predictions

        
      