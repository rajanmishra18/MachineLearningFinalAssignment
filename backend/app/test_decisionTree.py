from core.preprocessing import load_data, clean_data, encode_data, normalize, train_test_split
from algorithms.decision_tree import DecisionTree

df = load_data("data/churn.csv")
df = clean_data(df)
df = encode_data(df)
df = normalize(df)
print(df.head())
# Features
X = df.drop("Churn Label", axis=1).values.tolist()

# Target
y = df["Churn Label"].values.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = DecisionTree(max_depth=5)
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("Sample Predictions:", preds[:5])

print("\nDecision Tree:")
print(model.tree)
