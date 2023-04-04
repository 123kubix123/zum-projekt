from sklearn.model_selection import train_test_split

class DecompositionClassifier:
  __models = []

  def __init__(self, classifier, shuffle_data=False, **model_constructor_args):
    self.shuffle_data=shuffle_data
    self.classifier = classifier
    self.model_constructor_args = model_constructor_args

  @staticmethod
  def find_unique_values(list):
    unique_values = []
    for value in list:
        if value not in unique_values:
            unique_values.append(value)
    return unique_values

  def fit(self, X, y, test_size = 0.2, random_state=42, **model_fit_args):
    # get all classes into a list
    classes = self.find_unique_values(y)
    
    # for each class fit an instance of a given model as a binary classifier (class = 1 or not = 0)
    for c in classes:
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

  def print_scores(self):
    for model in self.__models:
      print(str(model['class']) + ' : ' +str(model['score']))
      