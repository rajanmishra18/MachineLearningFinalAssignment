import numpy as np


class SVM:

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):

        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters

        self.weights = None
        self.bias = None

    # -----------------------------
    # TRAIN MODEL
    # -----------------------------

    def fit(self, X, y):

        n_samples, n_features = X.shape

        # Convert labels:
        # 0 -> -1
        # 1 -> +1
        y_ = np.where(y <= 0, -1, 1)

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iters):

            for idx, x_i in enumerate(X):

                condition = y_[idx] * (
                    np.dot(x_i, self.weights) - self.bias
                ) >= 1

                if condition:

                    # Only regularization term
                    dw = 2 * self.lambda_param * self.weights
                    db = 0

                else:

                    # Hinge loss gradient
                    dw = (
                        2 * self.lambda_param * self.weights
                        - np.dot(x_i, y_[idx])
                    )

                    db = y_[idx]

                # Update weights
                self.weights = (
                    self.weights
                    - self.learning_rate * dw
                )

                self.bias = (
                    self.bias
                    - self.learning_rate * db
                )

    # -----------------------------
    # PREDICT
    # -----------------------------

    def predict(self, X):

        linear_output = np.dot(X, self.weights) - self.bias

        predictions = np.sign(linear_output)

        # Convert back:
        # -1 -> 0
        # +1 -> 1
        predictions = np.where(predictions <= 0, 0, 1)

        return predictions

    # -----------------------------
    # ACCURACY
    # -----------------------------

    def accuracy(self, y_true, y_pred):

        correct = np.sum(y_true == y_pred)

        return correct / len(y_true)