from math import log
from operator import itemgetter


class NaiveBayesClassifier:
    def __init__(self):
        self.log_likelihood = {}
        self.log_class_prior_probability = {}

    def fit(self, X, y):
        self.log_likelihood = {}
        self.log_class_prior_probability = {}

        for y_value in y.unique():
            self.log_class_prior_probability[y_value] = log(len(y[y == y_value]) + 1 / len(y) + len(y.unique()))

            self.log_likelihood[y_value] = {}
            for attribute in X.columns:
                self.log_likelihood[y_value][attribute] = {}
                for x in X[attribute].unique():
                    x_columns = X.index[X[attribute] == x].tolist()
                    y_filtered = list(itemgetter(*x_columns)(y))
                    self.log_likelihood[y_value][attribute][x] = log(
                        y_filtered.count(y_value) + 1 / len(y_filtered) + len(set(y_filtered)))

    def predict(self, X):
        result = []

        for index, row in X.iterrows():
            log_posterior = {}
            for y, log_prior_probability in self.log_class_prior_probability.items():
                log_posterior[y] = log_prior_probability

                for attribute in X.columns:
                    log_posterior[y] += self.log_likelihood[y][attribute][row[attribute]]
            result.append(max(log_posterior.items(), key=itemgetter(1))[0])

        return result
