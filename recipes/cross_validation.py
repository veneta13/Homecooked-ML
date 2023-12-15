import numpy as np
from numpy import mean


def cross_validate(X, y, model, k=10):
    def cross_validation_split(X, k):
        num_samples = len(X)
        fold_size = num_samples // k

        for i in range(k):
            start = i * fold_size
            end = start + fold_size

            validation_index = list(range(start, end))
            train_index = [idx for idx in range(num_samples) if idx not in validation_index]

            yield train_index, validation_index

    accuracies = []

    for train_idx, val_idx in cross_validation_split(X, k):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        accuracy = np.mean(y_pred == y_val)
        accuracies.append(accuracy)

    return accuracies, mean(accuracies)
