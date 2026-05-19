from fastapi import APIRouter
from app.core.preprocessing import load_data, clean_data, encode_data, convert_to_numeric, normalize, train_test_split
from app.algorithms.linear_regression import LinearRegression
from app.algorithms.logistic_regression import LogisticRegression
from app.algorithms.decision_tree import DecisionTree
from app.algorithms.knn import KNN
from app.algorithms.svm import SVM
from app.core.metrics import accuracy, confusion_matrix, precision, recall
import numpy as np

router = APIRouter()

@router.get("/models")
def get_models():
    return {
    "models": [
        "linear_regression",
        "logistic_regression",
        "decision_tree",
        "knn",
        "svm"
    ]
    }

def to_numpy(X):
    return np.array(X)

@router.post("/predict")
def predict(model_name: str):
    
    # Load and preprocess data
    df = load_data("app/data/churn.csv")
    df = clean_data(df)
    df = encode_data(df)
    df = convert_to_numeric(df)
    df = normalize(df)

    # Split
    X = df.drop("Churn Label", axis=1).values.tolist()
    y = df["Churn Label"].values.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Model selection
    if model_name == "linear_regression":
        model = LinearRegression(lr=0.01, epochs=200)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Calculate simple error (MSE)
        mse = sum((yt - yp) ** 2 for yt, yp in zip(y_test, preds)) / len(y_test)

        return {
        "model": model_name,
        "type": "regression",
        "metrics": {
            "mse": mse
        },
        "predictions": preds[:20],
        "actual": y_test[:20]
    }

    elif model_name == "decision_tree":
        model = DecisionTree(max_depth=5)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        return {
            "model": model_name,
            "type": "classification",
            "metrics": {
                "accuracy": accuracy(y_test, preds),
                "precision": precision(y_test, preds),
                "recall": recall(y_test, preds),
                "confusion_matrix": confusion_matrix(y_test, preds)
            },
            "predictions": preds[:20],
            "actual": y_test[:20]
        }
    elif model_name == "logistic_regression":
        model = LogisticRegression(lr=0.01, epochs=200)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        return {
            "model": model_name,
            "type": "classification",
            "metrics": {
                "accuracy": accuracy(y_test, preds),
                "confusion_matrix": confusion_matrix(y_test, preds),
                "precision": precision(y_test, preds),
                "recall": recall(y_test, preds),
                "sample_predictions": preds[:10]
            }
        }
    elif model_name == "knn":

        X_train_np = np.array(X_train)
        X_test_np = np.array(X_test)
        y_train_np = np.array(y_train)
        y_test_np = np.array(y_test)

        model = KNN(k=5)
        model.fit(X_train_np, y_train_np)

        preds = model.predict(X_test_np)

        preds_list = preds.tolist()

        return {
            "model": model_name,
            "type": "classification",
            "metrics": {
                "accuracy": accuracy(y_test, preds_list),
                "precision": precision(y_test, preds_list),
                "recall": recall(y_test, preds_list),
                "confusion_matrix": confusion_matrix(y_test, preds_list)
            },
            "predictions": preds_list[:20],
            "actual": y_test[:20]
        }
    elif model_name == "svm":
        X_train_np = np.array(X_train)
        X_test_np = np.array(X_test)
        y_train_np = np.array(y_train)
        y_test_np = np.array(y_test)

        model = SVM(
            learning_rate=0.001,
            lambda_param=0.01,
            n_iters=1000
        )

        model.fit(X_train_np, y_train_np)
        preds = model.predict(X_test_np)

        preds_list = preds.tolist()

        return {
            "model": model_name,
            "type": "classification",
            "metrics": {
                "accuracy": accuracy(y_test, preds_list),
                "precision": precision(y_test, preds_list),
                "recall": recall(y_test, preds_list),
                "confusion_matrix": confusion_matrix(y_test, preds_list)
            },
            "predictions": preds_list[:20],
            "actual": y_test[:20]
        }
    else:
        return {"error": "Invalid model"}