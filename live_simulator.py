import json
import random
import sqlite3
import time
import uuid
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ---------------- CONFIGURATION ----------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fraud_model.pkl"
DATA_PATH = BASE_DIR / "creditcard.csv"
DB_PATH = BASE_DIR / "cardshield_live.db"

INTERVAL_SECONDS = 3

# ---------------- LOAD MODEL ----------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = list(bundle["features"])

print("CardShield Live Simulator")
print(f"Model: {bundle.get('model_name', type(model).__name__)}")
print(f"Features required: {len(features)}")

# ---------------- DATABASE ----------------

def initialize_database():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS live_transactions (
                transaction_id TEXT PRIMARY KEY,
                transaction_time TEXT NOT NULL,
                amount REAL NOT NULL,
                fraud_probability REAL NOT NULL,
                prediction TEXT NOT NULL,
                risk_band TEXT NOT NULL,
                source TEXT NOT NULL,
                features_json TEXT NOT NULL
            )
        """)
        conn.commit()


# ---------------- TRANSACTION SOURCE ----------------

def load_demo_data():
    """Use the local Kaggle dataset if available."""
    if not DATA_PATH.exists():
        return None

    data = pd.read_csv(DATA_PATH)

    # Class is a label, never a model input.
    usable_features = [f for f in features if f in data.columns]

    if len(usable_features) != len(features):
        print("Dataset columns do not match model features.")
        return None

    return data


demo_data = load_demo_data()

if demo_data is not None:
    print("Source: local Kaggle dataset (simulated feed)")
else:
    print("Source: synthetic demo transactions")


def generate_transaction():
    """Generate one simulated transaction with model-compatible features."""
    if demo_data is not None:
        row = demo_data.sample(n=1).iloc[0]
        transaction = {
            feature: float(row[feature])
            for feature in features
        }
        source = "Kaggle demo replay"
    else:
        transaction = {}

        for feature in features:
            if feature == "Time":
                transaction[feature] = float(
                    random.randint(0, 172800)
                )
            elif feature == "Amount":
                transaction[feature] = round(
                    random.uniform(10, 25000), 2
                )
            else:
                transaction[feature] = float(
                    np.random.normal(0, 1)
                )

        source = "Synthetic demo"

    return transaction, source


# ---------------- ML PREDICTION ----------------

def score_transaction(transaction):
    frame = pd.DataFrame([transaction], columns=features)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(frame)[0]
        classes = list(model.classes_)

        if 1 in classes:
            fraud_probability = float(probabilities[classes.index(1)])
        else:
            fraud_probability = 0.0
    else:
        prediction = int(model.predict(frame)[0])
        fraud_probability = 1.0 if prediction == 1 else 0.0

    # Demo review threshold; tune and validate before real use.
    if fraud_probability >= 0.70:
        prediction = "High Risk"
        risk_band = "High"
    elif fraud_probability >= 0.30:
        prediction = "Review"
        risk_band = "Medium"
    else:
        prediction = "Normal"
        risk_band = "Low"

    return fraud_probability, prediction, risk_band


# ---------------- SAVE TRANSACTION ----------------

def save_transaction(transaction, source):
    probability, prediction, risk_band = score_transaction(transaction)

    amount = float(transaction.get("Amount", 0.0))
    transaction_id = "CS-" + uuid.uuid4().hex[:10].upper()
    transaction_time = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO live_transactions (
                transaction_id,
                transaction_time,
                amount,
                fraud_probability,
                prediction,
                risk_band,
                source,
                features_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            transaction_id,
            transaction_time,
            amount,
            probability,
            prediction,
            risk_band,
            source,
            json.dumps(transaction)
        ))
        conn.commit()

    print(
        f"{transaction_time} | {transaction_id} | "
        f"Amount: ₹{amount:.2f} | "
        f"Risk: {prediction} | "
        f"Fraud score: {probability:.2%}"
    )


# ---------------- RUN SIMULATOR ----------------

if __name__ == "__main__":
    initialize_database()
    print(f"Database: {DB_PATH}")
    print("Generating a demo transaction every 3 seconds.")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            transaction, source = generate_transaction()
            save_transaction(transaction, source)
            time.sleep(INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nSimulator stopped.")