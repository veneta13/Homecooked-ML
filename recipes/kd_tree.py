import heapq
import random
import numpy as np


class Node:
    def __init__(self, point, left=None, right=None, depth=0, label=None):
        self.point = point
        self.left = left
        self.right = right
        self.depth = depth
        self.label = label


def build_tree(points, depth=0, labels=None):
    if not points:
        return None
    k = len(points[0])
    axis = depth % k

    sorted_points = sorted(points, key=lambda x: x[axis])
    sorted_labels = [labels[points.index(point)] for point in sorted_points]
    median = len(points) // 2

    root = Node(sorted_points[median], depth=depth, label=sorted_labels[median])
    root.left = build_tree(sorted_points[:median], depth + 1, sorted_labels[:median])
    root.right = build_tree(sorted_points[median + 1:], depth + 1, sorted_labels[median + 1:])
    return root


def insert(root, point, depth=0, label=None):
    if root is None:
        return Node(point, depth=depth, label=label)
    k = len(point)
    axis = depth % k

    if point[axis] < root.point[axis]:
        root.left = insert(root.left, point, depth + 1, label=label)
    else:
        root.right = insert(root.right, point, depth + 1, label=label)
    return root


def knn_search(root, query, neighbours=1):
    candidates = []

    def search(node):
        if node is None:
            return
        k = len(query)
        axis = node.depth % k
        euclidean_distance = np.sqrt(sum((x - y) ** 2 for x, y in zip(query, node.point)))
        heapq.heappush(candidates, (-euclidean_distance, random.random(), node))

        if len(candidates) > neighbours:
            heapq.heappop(candidates)
        best_euclidean_distance = -candidates[0][0]

        if query[axis] < node.point[axis]:
            search(node.left)
            if (query[axis] - node.point[axis]) ** 2 < best_euclidean_distance:
                search(node.right)
        else:
            search(node.right)
            if (query[axis] - node.point[axis]) ** 2 < best_euclidean_distance:
                search(node.left)

    search(root)
    return [node.label for _, _, node in candidates]
