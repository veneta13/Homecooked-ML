from .id3 import id3, search_tree


class DecisionTreeClassifier():
    def __init__(self, k=None, max_depth=None):
        self.k = k
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X_train, y_train):
        self.tree = id3(X_train, list(y_train))

    def predict(self, X_test):
        result = []
        for _, row in X_test.iterrows():
            result.append(search_tree(self.tree, row, 0, self.k, self.max_depth))
        return result
