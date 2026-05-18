import pandas as pd
import numpy as np

from core.preprocessing import (
    load_data,
    clean_data,
    encode_data,
    normalize,
    train_test_split,
    convert_to_numeric
)

from algorithms.knn import KNN


# -----------------------------------
# LOAD DATA
# -----------------------------------

df = load_data("data/churn.csv")


# -----------------------------------
# CLEAN DATA
# -----------------------------------

df = clean_data(df)


# -----------------------------------
# ENCODE CATEGORICAL DATA
# -----------------------------------

df = encode_data(df)


# -----------------------------------
# CONVERT TO NUMERIC
# -----------------------------------

df = convert_to_numeric(df)


# -----------------------------------
# NORMALIZE DATA
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


# Convert lists to numpy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)

y_train = np.array(y_train)
y_test = np.array(y_test)


# -----------------------------------
# TRAIN KNN MODEL
# -----------------------------------

model = KNN(k=5)

model.fit(X_train, y_train)


# -----------------------------------
# PREDICTIONS
# -----------------------------------

predictions = model.predict(X_test)


# -----------------------------------
# ACCURACY
# -----------------------------------

accuracy = model.accuracy(y_test, predictions)

print("\nKNN Accuracy:")
print(round(accuracy * 100, 2), "%")

# -----------------------------------
# CONFUSION MATRIX
# -----------------------------------

tp = 0
tn = 0
fp = 0
fn = 0

for actual, predicted in zip(y_test, predictions):

    if actual == 1 and predicted == 1:
        tp += 1

    elif actual == 0 and predicted == 0:
        tn += 1

    elif actual == 0 and predicted == 1:
        fp += 1

    elif actual == 1 and predicted == 0:
        fn += 1


print("\nConfusion Matrix:\n")

print("TP:", tp)
print("TN:", tn)
print("FP:", fp)
print("FN:", fn)

# -----------------------------------
# PRECISION
# -----------------------------------

precision = tp / (tp + fp)

# -----------------------------------
# RECALL
# -----------------------------------

recall = tp / (tp + fn)

# -----------------------------------
# F1 SCORE
# -----------------------------------

f1_score = 2 * ((precision * recall) / (precision + recall))


print("\nPrecision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1_score, 4))


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