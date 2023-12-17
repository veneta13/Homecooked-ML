from .kd_tree import build_tree, knn_search


class KNNClassifier():
    def __init__(self, k=5):
        self.k = k
        self.tree = None

    def fit(self, X_train, y_train):
        points = []
        for _, row in X_train.iterrows():
            points.append(tuple(row))
        self.tree = build_tree(points, 0, list(y_train))

    def predict(self, X_test):
        result = []
        for _, row in X_test.iterrows():
            nns = knn_search(self.tree, tuple(row), self.k)
            result.append(max(set(nns), key=nns.count))
        return result
