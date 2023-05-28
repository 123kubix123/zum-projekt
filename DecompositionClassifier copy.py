from sklearn.model_selection import train_test_split
from threading import Thread


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

  def __fit_single_model(self, X, y, c, test_size, random_state, model_fit_args):
 #     print(X)
  #    print(y)
      examples = []
      target = []
      model = {'class': c, 'classifier': self.classifier(**self.model_constructor_args)}

      # assign new class labels
      for i, value in enumerate(y):
        examples.append(X[i])
        if value == c:
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
    classes = self.find_unique_values(y)
    
    threads = []
    # for each class fit an instance of a given model as a binary classifier (class = 1 or not = 0)
    for c in classes:
      t = Thread(target=self.__fit_single_model, args=(X, y, c, test_size, random_state, model_fit_args))
      threads.append(t)
      t.start()
    for t in threads:
      t.join()

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

        
      