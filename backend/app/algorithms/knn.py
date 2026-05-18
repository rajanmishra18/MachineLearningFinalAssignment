import numpy as np
from collections import Counter


class KNN:

    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    # Store training data
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # Euclidean Distance
    def euclidean_distance(self, x1, x2):
        distance = np.sqrt(np.sum((x1 - x2) ** 2))
        return distance

    # Predict one sample
    def predict_single(self, x):

        distances = []

        # Calculate distance from all training points
        for i in range(len(self.X_train)):

            distance = self.euclidean_distance(x, self.X_train[i])

            distances.append((distance, self.y_train[i]))

        # Sort by smallest distance
        distances.sort(key=lambda item: item[0])

        # Take first K neighbors
        k_nearest = distances[:self.k]

        # Extract labels
        k_labels = [label for _, label in k_nearest]

        # Majority vote
        most_common = Counter(k_labels).most_common(1)

        return most_common[0][0]

    # Predict multiple samples
    def predict(self, X):

        predictions = []

        for x in X:
            prediction = self.predict_single(x)
            predictions.append(prediction)

        return np.array(predictions)

    # Accuracy Score
    def accuracy(self, y_true, y_pred):

        correct = np.sum(y_true == y_pred)

        return correct / len(y_true)