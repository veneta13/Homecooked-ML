import numpy as np


class KMeans:
    def __init__(self, k, max_iterations=10_000):
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = None
        self.labels = None

    def __label__(self, point):
        distances = {}
        for idx, centroid in enumerate(self.centroids):
            distances[idx] = np.sqrt(np.sum((point - centroid) ** 2))
        label = min(distances, key=distances.get)
        return label

    def __initialize_centroids__(self, X):
        xs = np.random.uniform(low=np.min(X['x']), high=np.max(X['x']), size=self.k)
        ys = np.random.uniform(low=np.min(X['y']), high=np.max(X['y']), size=self.k)
        self.centroids = np.array(zip(xs, ys))

    def __calculate_distances__(self, point):
        distances = np.sqrt(np.sum((point - self.centroids) ** 2, axis=1))
        return distances

    def __assign_labels__(self, X):
        self.labels = [np.argmin(self.__calculate_distances__(np.array(row))) for _, row in X.iterrows()]

    def __update_centroids__(self, X):
        new_centroids = np.array([np.mean(X.iloc[np.array(self.labels) == i], axis=0) for i in range(self.k)])
        return new_centroids

    def fit(self, X):
        self.__initialize_centroids__(X)
        for _ in range(self.max_iterations):
            self.__assign_labels__(X)
            new_centroids = self.__update_centroids__(X)
            if np.all(new_centroids == self.centroids):
                break
            self.centroids = new_centroids

    def predict(self, X):
        labels = [self.__label__(np.array(list(row))) for _, row in X.iterrows()]
        return labels
