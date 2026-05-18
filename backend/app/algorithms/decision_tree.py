import math
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None

    # Calculate entropy
    def entropy(self, y):
        total = len(y)
        counts = Counter(y)

        entropy_value = 0
        for count in counts.values():
            probability = count / total
            entropy_value -= probability * math.log2(probability)

        return entropy_value

    # Split dataset
    def split_dataset(self, X, y, feature_index, threshold):
        left_X, right_X = [], []
        left_y, right_y = [], []

        for i in range(len(X)):
            if X[i][feature_index] <= threshold:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])

        return left_X, right_X, left_y, right_y

    # Information Gain
    def information_gain(self, parent_y, left_y, right_y):
        parent_entropy = self.entropy(parent_y)

        n = len(parent_y)
        n_left = len(left_y)
        n_right = len(right_y)

        if n_left == 0 or n_right == 0:
            return 0

        child_entropy = (
            (n_left / n) * self.entropy(left_y)
            + (n_right / n) * self.entropy(right_y)
        )

        gain = parent_entropy - child_entropy
        return gain

    # Find best split
    def best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None

        n_features = len(X[0])

        for feature_index in range(n_features):

            thresholds = set(row[feature_index] for row in X)

            for threshold in thresholds:

                left_X, right_X, left_y, right_y = self.split_dataset(
                    X, y, feature_index, threshold
                )

                gain = self.information_gain(y, left_y, right_y)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    # Build tree recursively
    def build_tree(self, X, y, depth=0):

        # If all labels are same
        if len(set(y)) == 1:
            return y[0]

        # Stop if max depth reached
        if depth >= self.max_depth:
            return Counter(y).most_common(1)[0][0]

        feature, threshold = self.best_split(X, y)

        if feature is None:
            return Counter(y).most_common(1)[0][0]

        left_X, right_X, left_y, right_y = self.split_dataset(
            X, y, feature, threshold
        )

        left_subtree = self.build_tree(left_X, left_y, depth + 1)
        right_subtree = self.build_tree(right_X, right_y, depth + 1)

        return {
            "feature": feature,
            "threshold": threshold,
            "left": left_subtree,
            "right": right_subtree
        }

    # Train model
    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    # Predict single sample
    def predict_single(self, x, tree):

        if not isinstance(tree, dict):
            return tree

        feature = tree["feature"]
        threshold = tree["threshold"]

        if x[feature] <= threshold:
            return self.predict_single(x, tree["left"])
        else:
            return self.predict_single(x, tree["right"])

    # Predict multiple samples
    def predict(self, X):
        return [self.predict_single(x, self.tree) for x in X]
