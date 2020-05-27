from sklearn import tree
from models import PreparedSet
import numpy as np

class DecisionTree:
    criterion = "gini"
    splitter = "best"
    training_prepared_set = None
    test_prepared_set = None
    classifier = None

    def __init__(self, criterion, splitter, training_set, test_set ):
        self.criterion = criterion
        self.splitter = splitter
        self.training_prepared_set = training_set
        self.test_prepared_set = test_set

        self.classifier = tree.DecisionTreeClassifier(criterion=self.criterion, splitter=self.splitter)

    def train(self):
        print("training")
        #print(self.training_prepared_set)
        X_1 = self.training_prepared_set.get_features()
      #  print(X_1)
        self.classifier.fit(self.training_prepared_set.get_features(), self.training_prepared_set.get_blog_types())

    def test(self):
        print("testing")
        test_results = self.classifier.predict(self.test_prepared_set.get_features())
        print("test result length", len(test_results))
        print("test set classes length", len(self.test_prepared_set.get_blog_types()))

        good_predictions = 0

        for i in range(len(test_results)):
            if test_results[i] == self.test_prepared_set.get_blog_type(i):
                good_predictions += 1

        return good_predictions / len(test_results)
