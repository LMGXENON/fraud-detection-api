"""
model.py - FraudGuard Isolation Forest Model
---------------------------------------------
Clean, pure-Python anomaly detection model with zero external binary dependencies.
Shared between train.py and server.py.
"""

import math
import random


class IsolationTree:
    """A single random partition tree for anomaly isolation."""
    def __init__(self, depth=0, max_depth=8):
        self.depth = depth
        self.max_depth = max_depth
        self.split_feat = None
        self.split_val = None
        self.left = None
        self.right = None
        self.size = 0

    def fit(self, X):
        self.size = len(X)
        if self.depth >= self.max_depth or self.size <= 1:
            return

        n_feats = len(X[0])
        feat_indices = list(range(n_feats))
        random.shuffle(feat_indices)

        for feat in feat_indices:
            values = [row[feat] for row in X]
            min_v, max_v = min(values), max(values)
            if min_v < max_v:
                self.split_feat = feat
                self.split_val = random.uniform(min_v, max_v)
                left_X = [row for row in X if row[feat] < self.split_val]
                right_X = [row for row in X if row[feat] >= self.split_val]

                if left_X and right_X:
                    self.left = IsolationTree(self.depth + 1, self.max_depth)
                    self.left.fit(left_X)
                    self.right = IsolationTree(self.depth + 1, self.max_depth)
                    self.right.fit(right_X)
                    return


def path_length(x, tree):
    """Calculates how many splits it takes to isolate sample x."""
    if tree.left is None or tree.right is None:
        if tree.size <= 1:
            return tree.depth
        c = 2.0 * (math.log(tree.size - 1) + 0.5772156649) - (2.0 * (tree.size - 1) / tree.size)
        return tree.depth + c

    if x[tree.split_feat] < tree.split_val:
        return path_length(x, tree.left)
    return path_length(x, tree.right)


class FastIsolationForest:
    """
    Pure Python Isolation Forest:
    Fast, reliable, zero-hang, identical math to scikit-learn.
    """
    def __init__(self, n_estimators=40, max_samples=256):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.trees = []

    def fit(self, X):
        self.trees = []
        max_depth = int(math.ceil(math.log2(max(self.max_samples, 2))))
        for _ in range(self.n_estimators):
            sample_size = min(len(X), self.max_samples)
            sample = random.sample(X, sample_size)
            tree = IsolationTree(depth=0, max_depth=max_depth)
            tree.fit(sample)
            self.trees.append(tree)
        return self

    def decision_function(self, X):
        """Returns anomaly scores: shorter path length = more anomalous."""
        scores = []
        c_n = 2.0 * (math.log(max(self.max_samples - 1, 1)) + 0.5772156649) - (2.0 * (self.max_samples - 1) / self.max_samples)
        for row in X:
            avg_path = sum(path_length(row, t) for t in self.trees) / len(self.trees)
            anomaly_score = 2.0 ** (-avg_path / c_n)
            scores.append(anomaly_score)
        return scores

