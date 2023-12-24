from operator import itemgetter
import numpy as np
from .id3 import id3, search_tree


class RandomForestClassifier():
    def __init__(self, k=None, max_depth=None, num_trees=101):
        self.k = k
        self.max_depth = max_depth
        self.num_trees = num_trees
        self.trees = []

    def __generate_random_subset_of_features__(self, X, percent_features=0.8):
        n_features = X.shape[1]
        num_features_to_select = int(percent_features * n_features)
        columns = list(X.columns)
        indexes = np.random.choice(range(n_features), size=num_features_to_select, replace=False)
        return itemgetter(*indexes)(columns)

    def fit(self, X_train, y_train):
        X_subset = X_train[list(self.__generate_random_subset_of_features__(X_train, 0.8))]
        y_train = list(y_train)
        for _ in range(self.num_trees):
            self.trees.append(id3(
                X_subset,
                y_train
            ))

    def predict(self, X_test):
        result = []
        for _, row in X_test.iterrows():
            votes = []
            for tree in self.trees:
                votes.append(search_tree(tree, row, 0, self.k, self.max_depth))
            result.append(max(set(votes), key=votes.count))
        return result
