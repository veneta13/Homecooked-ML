from math import log
from operator import itemgetter


class NaiveBayesClassifier:
    def __init__(self):
        self.__log_likelihood__ = {}
        self.__log_class_prior_probability__ = {}

    def fit(self, X, y):
        self.__log_likelihood__ = {}
        self.__log_class_prior_probability__ = {}

        for y_value in y.unique():
            self.__log_class_prior_probability__[y_value] = log(len(y[y == y_value]) + 1 / len(y) + len(y.unique()))

            self.__log_likelihood__[y_value] = {}
            for attribute in X.columns:
                self.__log_likelihood__[y_value][attribute] = {}
                for x in X[attribute].unique():
                    x_columns = X.index[X[attribute] == x].tolist()
                    y_filtered = list(itemgetter(*x_columns)(y))
                    self.__log_likelihood__[y_value][attribute][x] = log(
                        y_filtered.count(y_value) + 1 / len(y_filtered) + len(set(y_filtered)))

    def predict(self, X):
        result = []

        for index, row in X.iterrows():
            log_posterior = {}
            for y, log_prior_probability in self.__log_class_prior_probability__.items():
                log_posterior[y] = log_prior_probability

                for attribute in X.columns:
                    log_posterior[y] += self.__log_likelihood__[y][attribute][row[attribute]]
            result.append(max(log_posterior.items(), key=itemgetter(1))[0])

        return result
