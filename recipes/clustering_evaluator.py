import numpy as np


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def silhouette(X, labels):
    n_samples = len(X)
    silhouette_scores = np.zeros(n_samples)

    for current_idx in range(n_samples):
        cluster_label = labels[current_idx]
        intra_distances = []
        min_inter_distance = np.inf

        for other_idx in range(n_samples):
            if cluster_label == labels[other_idx] and current_idx != other_idx:
                intra_distances.append(euclidean_distance(X.iloc[current_idx], X.iloc[other_idx]))

        if len(intra_distances) == 0:
            silhouette_scores[current_idx] = 0.0
            continue
        avg_intra_distances = np.mean(intra_distances)

        for current_label in set(labels):
            if current_label != cluster_label:
                current_inter_distance = np.mean(
                    [
                        euclidean_distance(X.iloc[current_idx], X.iloc[idx])
                        for idx in range(n_samples)
                        if labels[idx] == current_label
                    ]
                )
                min_inter_distance = min(min_inter_distance, current_inter_distance)

        if min_inter_distance == np.inf:
            silhouette_scores[current_idx] = 0.0
            continue
        silhouette_scores[current_idx] = (min_inter_distance - avg_intra_distances) / max(avg_intra_distances,
                                                                                          min_inter_distance)

    return silhouette_scores, np.mean(silhouette_scores)


def intra_distance(X, labels):
    n_samples = len(X)
    intra_distances = []

    for current_idx in range(n_samples):
        cluster_label = labels[current_idx]

        for other_idx in range(n_samples):
            if cluster_label == labels[other_idx] and current_idx != other_idx:
                intra_distances.append(euclidean_distance(X.iloc[current_idx], X.iloc[other_idx]))

    return intra_distances, np.mean(intra_distances)


def inter_distance(X, labels):
    n_samples = len(X)
    inter_distances = []

    for current_idx in range(n_samples):
        cluster_label = labels[current_idx]

        for current_label in set(labels):
            if current_label != cluster_label:
                inter_distances.append(np.mean(
                    [
                        euclidean_distance(X.iloc[current_idx], X.iloc[idx])
                        for idx in range(n_samples)
                        if labels[idx] == current_label
                    ]
                ))

    return inter_distances, np.mean(inter_distances)


class ClusteringEvaluator:
    def __init__(self):
        self.X = None
        self.labels = None

    def fit(self, X, labels):
        self.X = X
        self.labels = labels

    def evaluate(self, metric=None):
        assert metric in ['silhouette', 'intra', 'inter']

        if metric == 'silhouette':
            return silhouette(self.X, self.labels)
        if metric == 'intra':
            return intra_distance(self.X, self.labels)
        if metric == 'inter':
            return inter_distance(self.X, self.labels)
