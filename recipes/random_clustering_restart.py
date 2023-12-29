import numpy as np


class RandomClusteringRestarter:
    def __init__(self, model, evaluator, max_restarts=10, max_iterations=1000, k=4, metric='silhouette'):
        self.model = model
        self.evaluator = evaluator
        self.max_restarts = max_restarts
        self.max_iterations = max_iterations
        self.k = k
        self.metric = metric

    def run(self, X):
        best_model = None
        best_score = -np.inf

        evaluator = self.evaluator()

        for _ in range(self.max_restarts):
            current_model = self.model(k=self.k, max_iterations=self.max_iterations)
            current_model.fit(X)

            evaluator.fit(X, current_model.labels)
            score = evaluator.evaluate(metric=self.metric)[1]

            if best_score < score:
                best_model = current_model
                best_score = score

        return best_model, best_score