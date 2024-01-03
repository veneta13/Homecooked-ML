import numpy as np


class KMeansPlusPlus:
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
        self.centroids = np.array(X.sample(n=1))

        for i in range(self.k - 1):
            points = []
            min_distances = []

            for _, point in X.iterrows():
                if not any([all(same) for same in [np.array(point) == centroid for centroid in self.centroids]]):
                    points.append(np.array(point))
                    min_distances.append(np.min(self.__calculate_distances__(np.array(point))))

            probs = np.array(min_distances) / np.sum(np.array(min_distances))
            new_centroid_idx = np.random.choice(len(points), 1, p=probs)[0]
            new_centroid = points[new_centroid_idx].reshape([1, 2])
            self.centroids = np.append(self.centroids, new_centroid, axis=0)

    def __calculate_distances__(self, point):
        distances = np.sqrt(np.sum((point - self.centroids) ** 2, axis=1))
        return distances

    def __assign_labels__(self, X):
        self.labels = [self.__label__(np.array(row)) for _, row in X.iterrows()]

    def __update_centroids__(self, X):
        return np.array([np.mean(X.iloc[np.array(self.labels) == i], axis=0) for i in range(self.k)])

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
