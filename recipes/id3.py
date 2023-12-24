from math import log2
from operator import itemgetter


class Node:
    def __init__(self, value=None, attribute=None, branches=None, target_class=None, num_examples=1):
        self.value = value
        self.attribute = attribute
        self.branches = branches if branches else {}
        self.target_class = target_class
        self.num_examples = num_examples


def get_entropy_target(y):
    entropy = 0

    for y_value in list(set(y)):
        class_probability = y.count(y_value) / len(y)
        entropy -= (class_probability * log2(class_probability))

    return entropy


def get_entropy_attribute(X, y, attribute):
    entropy_count = {}

    for x in X[attribute].unique():
        x_columns = X.index[X[attribute] == x].tolist()
        if len(x_columns) > 0:
            y_filtered = list(itemgetter(*x_columns)(y)) if len(x_columns) > 1 else y[x_columns[0]]
        entropy = 0

        for y_value in list(set(y)):
            class_probability = y_filtered.count(y_value) / len(y_filtered)
            if class_probability != 0:
                entropy -= (class_probability * log2(class_probability))

        entropy_count[x] = [entropy, len(y_filtered)]

    return entropy_count


def get_information_gain(X, y, attribute):
    n = len(y)
    target_entropy = get_entropy_target(y)
    entropy_count = get_entropy_attribute(X, y, attribute)
    entropies = [x[0] for x in entropy_count.values()]
    counts = [x[1] for x in entropy_count.values()]
    entropy_proportion = sum(- (count / n) * entropy for count, entropy in zip(counts, entropies))
    return target_entropy - entropy_proportion


def get_most_informative_attribute(X, y):
    information_gains = []

    for attribute in X.columns:
        information_gains.append([get_information_gain(X, y, attribute), attribute])

    return sorted(information_gains, key=itemgetter(0), reverse=True)[0][1]


def id3(X, y):
    X = X.reset_index()
    X.drop(columns=['index', ], inplace=True)

    if len(set(y)) == 1:
        return Node(target_class=y[0], num_examples=1)

    majority_class = max(set(y), key=y.count)

    if not any(X.columns):
        return Node(target_class=majority_class, num_examples=len(y))

    best_attribute = get_most_informative_attribute(X, y)
    tree = Node(attribute=best_attribute, target_class=majority_class, num_examples=len(y))

    for value in X[best_attribute].unique():
        X_subset = X[X[best_attribute] == value].drop(columns=[best_attribute])
        y_subset = [y[i] for i in X_subset.index]
        subtree = id3(X_subset, y_subset)
        tree.branches[value] = subtree
    return tree


def search_tree(tree, X, depth, example_threshold, depth_threshold):
    if example_threshold is not None and tree.num_examples <= example_threshold:
        return tree.target_class
    if depth_threshold is not None and depth > depth_threshold:
        return tree.target_class
    if tree.num_examples == 1:
        return tree.target_class
    if tree.attribute is None or X[tree.attribute] not in tree.branches.keys():
        return tree.target_class

    result = search_tree(
        tree.branches[X[tree.attribute]],
        X,
        depth + 1,
        example_threshold,
        depth_threshold
    )

    return result
