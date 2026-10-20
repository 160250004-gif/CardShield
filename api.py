
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ---------------- LOAD TRAINED MODEL ----------------

MODEL_PATH = Path(__file__).parent / "fraud_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "fraud_model.pkl not found. Run train_model.py first."
    )

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
features = bundle["features"]


# ---------------- CREATE API ----------------

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description=(
        "Machine Learning API for transaction fraud-risk scoring. "
        "Predictions are alerts, not proof of fraud."
    ),
    version="1.0.0"
)


# ---------------- REQUEST FORMAT ----------------

class TransactionRequest(BaseModel):
    values: dict[str, float]

    threshold: float = Field(
        default=0.50,
        ge=0.05,
        le=0.95,
        description="Score threshold for flagging a transaction"
    )


# ---------------- HOME ENDPOINT ----------------

@app.get("/")
def home():
    return {
        "message": "Credit Card Fraud Detection API is running",
        "docs": "/docs",
        "model_features_endpoint": "/features"
    }


# ---------------- HEALTH CHECK ----------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "feature_count": len(features)
    }


# ---------------- REQUIRED FEATURES ----------------

@app.get("/features")
def get_features():
    return {
        "required_features": features,
        "feature_count": len(features),
        "instructions": (
            "Send every required feature as a numeric value "
            "to the /predict endpoint."
        )
    }


# ---------------- PREDICTION ENDPOINT ----------------

@app.post("/predict")
def predict_transaction(request: TransactionRequest):

    values = request.values

    # Check required columns
    missing = [
        feature
        for feature in features
        if feature not in values
    ]

    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Missing required features",
                "missing_features": missing
            }
        )

    # Arrange features in the same order used during training
    try:
        row = [float(values[feature]) for feature in features]
    except (TypeError, ValueError, OverflowError):
        raise HTTPException(
            status_code=400,
            detail="All feature values must be valid numbers."
        )

    # Reject NaN and infinite values
    if not np.isfinite(row).all():
        raise HTTPException(
            status_code=400,
            detail="Feature values must be finite numbers."
        )

    # Convert input to a DataFrame with correct column names
    transaction = pd.DataFrame(
        [row],
        columns=features
    )

    # Predict fraud score
    try:
        score = float(
            model.predict_proba(transaction)[0, 1]
        )
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Model prediction failed: {str(error)}"
        )

    # Apply the selected threshold
    flagged = score >= request.threshold

    return {
        "fraud_score": round(score, 4),
        "risk_percent": round(score * 100, 2),
        "threshold": request.threshold,
        "prediction": (
            "Review for Fraud" if flagged else "Normal"
        ),
        "flagged_for_review": bool(flagged),
        "disclaimer": (
            "A model prediction is not proof of fraud. "
            "Review the transaction before making decisions."
        )
    }

class BatchTransactionRequest(BaseModel):
    transactions: list[dict[str, float]] = Field(
        min_length=1,
        max_length=500
    )
    threshold: float = Field(default=0.50, ge=0.05, le=0.95)


@app.post("/predict-batch")
def predict_batch(request: BatchTransactionRequest):
    rows = []

    for index, values in enumerate(request.transactions):
        missing = [feature for feature in features if feature not in values]

        if missing:
            raise HTTPException(
                status_code=400,
                detail={
                    "transaction_index": index,
                    "missing_features": missing
                }
            )

        try:
            row = [float(values[feature]) for feature in features]
        except (TypeError, ValueError, OverflowError):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid numeric value in transaction {index}"
            )

        if not np.isfinite(row).all():
            raise HTTPException(
                status_code=400,
                detail=f"Non-finite value in transaction {index}"
            )

        rows.append(row)

    transaction_df = pd.DataFrame(rows, columns=features)

    try:
        scores = model.predict_proba(transaction_df)[:, 1]
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Batch prediction failed. Check the trained model."
        )

    results = []

    for index, score in enumerate(scores):
        score = float(score)
        flagged = score >= request.threshold

        results.append({
            "transaction_index": index,
            "fraud_score": round(score, 4),
            "risk_percent": round(score * 100, 2),
            "prediction": "Review for Fraud" if flagged else "Normal",
            "flagged_for_review": bool(flagged)
        })

    return {
        "total_transactions": len(results),
        "flagged_for_review": sum(
            result["flagged_for_review"] for result in results
        ),
        "threshold": request.threshold,
        "results": results,
        "disclaimer": "Predictions are risk alerts, not proof of fraud."
    }
