import pandas as pd
import random

def load_data(path):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    # Keep only important columns
    df.columns = df.columns.str.strip()
    selected_columns = [
    "Age",
    "Tenure in Months",
    "Monthly Charge",
    "Total Charges",
    "Contract",
    "Internet Service",
    "Payment Method",
    "Churn Label"
    ]

    df = df[selected_columns].copy()

    # Fix Total Charges
    df["Total Charges"] = df["Total Charges"].replace(" ", None)
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors='coerce')

    df = df.dropna()

    return df

def encode_data(df):
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype=='str':
            unique_vals = df[col].unique()
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
            df[col] = df[col].map(mapping)

    return df

def normalize(df):

    # Do NOT normalize target column
    feature_columns = [
        col for col in df.columns
        if col != "Churn Label"
    ]

    for col in feature_columns:

        if df[col].dtype != 'object':

            min_val = df[col].min()
            max_val = df[col].max()

            if max_val - min_val != 0:

                df[col] = (
                    (df[col] - min_val)
                    / (max_val - min_val)
                )

    return df

def train_test_split(X, y, test_size=0.2):
    data = list(zip(X, y))
    random.shuffle(data)

    split_index = int(len(data) * (1 - test_size))
    train_data = data[:split_index]
    test_data = data[split_index:]

    X_train, y_train = zip(*train_data)
    X_test, y_test = zip(*test_data)

    return list(X_train), list(X_test), list(y_train), list(y_test)

def convert_to_numeric(df):
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna()
    return df