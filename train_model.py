
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

DATA_PATH = "creditcard.csv"
MODEL_PATH = "fraud_model.pkl"

# 1. Load dataset
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df["Class"].value_counts())

# 2. Basic data cleaning
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna().copy()

# 3. Features and target
X = df.drop(columns=["Class"])
y = df["Class"].astype(int)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("scale", StandardScaler(), ["Time", "Amount"])
    ],
    remainder="passthrough"
)

# 6. Two candidate models
models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
}

results = []
best_score = -1
best_model = None
best_name = None

# 7. Train and compare
for name, classifier in models.items():
    print(f"\nTraining: {name}")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    scores = pipeline.predict_proba(X_test)[:, 1]

    ap = average_precision_score(y_test, scores)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    results.append({
        "Model": name,
        "Average Precision": ap,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print("Average Precision:", round(ap, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    if ap > best_score:
        best_score = ap
        best_model = pipeline
        best_name = name

# 8. Save comparison results
results_df = pd.DataFrame(results)
results_df.to_csv("model_comparison.csv", index=False)

# 9. Save best model and feature information
joblib.dump({
    "model": best_model,
    "features": list(X.columns),
    "model_name": best_name,
    "average_precision": best_score
}, MODEL_PATH)

print("\nModel comparison:")
print(results_df.to_string(index=False))
print("\nBest model:", best_name)
print("Model saved to:", MODEL_PATH)
