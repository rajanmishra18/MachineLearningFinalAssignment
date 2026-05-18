import pandas as pd
import numpy as np

from core.preprocessing_svm import (
    load_data,
    clean_data,
    encode_data,
    normalize,
    train_test_split,
    convert_to_numeric
)

from algorithms.svm import SVM


# -----------------------------------
# LOAD DATA
# -----------------------------------

df = load_data("data/churn.csv")


# -----------------------------------
# CLEAN DATA
# -----------------------------------

df = clean_data(df)


# -----------------------------------
# ENCODE DATA
# -----------------------------------

df = encode_data(df)


# -----------------------------------
# CONVERT TO NUMERIC
# -----------------------------------

df = convert_to_numeric(df)


# -----------------------------------
# NORMALIZE
# -----------------------------------

df = normalize(df)


# -----------------------------------
# SPLIT FEATURES AND LABELS
# -----------------------------------

X = df.drop("Churn Label", axis=1).values
y = df["Churn Label"].values


# -----------------------------------
# TRAIN TEST SPLIT
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y)

X_train = np.array(X_train)
X_test = np.array(X_test)

y_train = np.array(y_train)
y_test = np.array(y_test)


# -----------------------------------
# TRAIN SVM MODEL
# -----------------------------------

model = SVM(
    learning_rate=0.001,
    lambda_param=0.01,
    n_iters=1000
)

model.fit(X_train, y_train)


# -----------------------------------
# PREDICTIONS
# -----------------------------------

predictions = model.predict(X_test)


# -----------------------------------
# ACCURACY
# -----------------------------------

accuracy = model.accuracy(y_test, predictions)

print("\nSVM Accuracy:")
print(round(accuracy * 100, 2), "%")


# -----------------------------------
# SAMPLE PREDICTIONS
# -----------------------------------

print("\nSample Predictions:\n")

for i in range(10):

    print(
        "Actual:",
        y_test[i],
        "| Predicted:",
        predictions[i]
    )