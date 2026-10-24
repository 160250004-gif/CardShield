
from pathlib import Path
from datetime import datetime, timezone
import sqlite3

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field


# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fraud_model.pkl"
DB_PATH = BASE_DIR / "prediction_history.db"

app = FastAPI(
    title="CardShield Fraud Detection API",
    description=(
        "Credit Card Fraud Detection API using Machine Learning. "
        "Predictions are risk alerts, not proof of fraud."
    ),
    version="2.0.0",
)


# ==================================================
# 2. LOAD TRAINED MODEL
# ==================================================

if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Model file not found: {MODEL_PATH}. "
        "Place fraud_model.pkl in the same folder as api.py."
    )

try:
    bundle = joblib.load(MODEL_PATH)

    if not isinstance(bundle, dict):
        raise ValueError(
            "Expected a model bundle containing 'model' and 'features'."
        )

    model = bundle["model"]
    features = list(bundle["features"])

    if not features:
        raise ValueError("The model feature list is empty.")

except Exception as error:
    raise RuntimeError(
        f"Could not load the trained model: {error}"
    ) from error


# ==================================================
# 3. DATABASE
# ==================================================

def get_connection():
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                fraud_score REAL NOT NULL,
                risk_percent REAL NOT NULL,
                threshold REAL NOT NULL,
                prediction TEXT NOT NULL,
                flagged_for_review INTEGER NOT NULL
            )
            """
        )


init_database()


def save_prediction(score, threshold, prediction, flagged):
    created_at = datetime.now(timezone.utc).isoformat()

    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO prediction_history (
                created_at,
                fraud_score,
                risk_percent,
                threshold,
                prediction,
                flagged_for_review
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                float(score),
                float(score) * 100,
                float(threshold),
                prediction,
                int(flagged),
            ),
        )

        record_id = cursor.lastrowid

    return {
        "id": record_id,
        "created_at": created_at,
        "fraud_score": round(float(score), 4),
        "risk_percent": round(float(score) * 100, 2),
        "threshold": float(threshold),
        "prediction": prediction,
        "flagged_for_review": bool(flagged),
    }


# ==================================================
# 4. REQUEST VALIDATION
# ==================================================

class TransactionRequest(BaseModel):
    values: dict[str, float]

    threshold: float = Field(
        default=0.50,
        ge=0.05,
        le=0.95,
        description="Threshold used to flag transactions for review",
    )


class BatchTransactionRequest(BaseModel):
    transactions: list[dict[str, float]] = Field(
        min_length=1,
        max_length=500,
    )

    threshold: float = Field(
        default=0.50,
        ge=0.05,
        le=0.95,
    )


# ==================================================
# 5. COMMON VALIDATION AND PREDICTION
# ==================================================

def prepare_transaction(values, index=None):
    missing = [
        feature for feature in features
        if feature not in values
    ]

    if missing:
        detail = {
            "message": "Missing required model features",
            "missing_features": missing,
        }

        if index is not None:
            detail["transaction_index"] = index

        raise HTTPException(status_code=422, detail=detail)

    try:
        row = [float(values[feature]) for feature in features]
    except (TypeError, ValueError, OverflowError):
        raise HTTPException(
            status_code=422,
            detail=f"All feature values must be numeric. Transaction index: {index}",
        )

    if not np.isfinite(row).all():
        raise HTTPException(
            status_code=422,
            detail=f"Feature values must be finite. Transaction index: {index}",
        )

    return row


def get_fraud_scores(dataframe):
    if not hasattr(model, "predict_proba"):
        raise HTTPException(
            status_code=500,
            detail="The loaded model does not support predict_proba.",
        )

    try:
        probabilities = model.predict_proba(dataframe)

        if probabilities.ndim != 2 or probabilities.shape[1] < 2:
            raise ValueError(
                "The model did not return a fraud probability."
            )

        # Assumes the second probability column represents fraud class.
        scores = probabilities[:, 1]

        if not np.isfinite(scores).all():
            raise ValueError("The model returned invalid probability values.")

        return np.clip(scores, 0.0, 1.0)

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Model prediction failed: {error}",
        ) from error


def make_result(score, threshold):
    flagged = bool(float(score) >= threshold)

    return {
        "fraud_score": round(float(score), 4),
        "risk_percent": round(float(score) * 100, 2),
        "threshold": threshold,
        "prediction": "Review for Fraud" if flagged else "Normal",
        "flagged_for_review": flagged,
    }


# ==================================================
# 6. HOME
# ==================================================

@app.get("/")
def home():
    return {
        "application": "CardShield",
        "message": "Credit Card Fraud Detection API is running",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "features": "/features",
        "prediction": "/predict",
        "batch_prediction": "/predict-batch",
        "history": "/history",
    }


# ==================================================
# 7. HEALTH CHECK
# ==================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "feature_count": len(features),
        "database": DB_PATH.name,
    }


# ==================================================
# 8. REQUIRED FEATURES
# ==================================================

@app.get("/features")
def get_features():
    return {
        "required_features": features,
        "feature_count": len(features),
        "instructions": (
            "Send every required feature with its exact name "
            "and a numeric value to the prediction endpoint."
        ),
    }


# ==================================================
# 9. SINGLE TRANSACTION PREDICTION
# ==================================================

@app.post("/predict")
def predict_transaction(request: TransactionRequest):
    row = prepare_transaction(request.values)

    transaction_df = pd.DataFrame(
        [row],
        columns=features,
    )

    score = float(get_fraud_scores(transaction_df)[0])
    result = make_result(score, request.threshold)

    try:
        saved = save_prediction(
            score,
            request.threshold,
            result["prediction"],
            result["flagged_for_review"],
        )
    except sqlite3.Error as error:
        raise HTTPException(
            status_code=500,
            detail="Could not save prediction history.",
        ) from error

    return {
        **saved,
        "disclaimer": (
            "A prediction is a risk alert, not proof of fraud. "
            "Review the transaction before making decisions."
        ),
    }


# ==================================================
# 10. BATCH PREDICTION
# ==================================================

@app.post("/predict-batch")
def predict_batch(request: BatchTransactionRequest):
    rows = []

    for index, values in enumerate(request.transactions):
        rows.append(
            prepare_transaction(values, index=index)
        )

    transaction_df = pd.DataFrame(
        rows,
        columns=features,
    )

    scores = get_fraud_scores(transaction_df)

    results = []

    try:
        with get_connection() as conn:
            for index, score in enumerate(scores):
                result = make_result(
                    float(score),
                    request.threshold,
                )

                created_at = datetime.now(timezone.utc).isoformat()

                cursor = conn.execute(
                    """
                    INSERT INTO prediction_history (
                        created_at,
                        fraud_score,
                        risk_percent,
                        threshold,
                        prediction,
                        flagged_for_review
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        created_at,
                        float(score),
                        float(score) * 100,
                        request.threshold,
                        result["prediction"],
                        int(result["flagged_for_review"]),
                    ),
                )

                results.append({
                    "transaction_index": index,
                    "id": cursor.lastrowid,
                    "created_at": created_at,
                    **result,
                })

    except sqlite3.Error as error:
        raise HTTPException(
            status_code=500,
            detail="Could not save batch prediction history.",
        ) from error

    flagged_count = sum(
        item["flagged_for_review"] for item in results
    )

    return {
        "total_transactions": len(results),
        "flagged_for_review": flagged_count,
        "normal_transactions": len(results) - flagged_count,
        "threshold": request.threshold,
        "results": results,
        "disclaimer": (
            "Predictions are risk alerts, not proof of fraud."
        ),
    }


# ==================================================
# 11. PREDICTION HISTORY
# ==================================================

@app.get("/history")
def get_history(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    flagged_only: bool = False,
):
    try:
        with get_connection() as conn:
            if flagged_only:
                total = conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM prediction_history
                    WHERE flagged_for_review = 1
                    """
                ).fetchone()[0]

                rows = conn.execute(
                    """
                    SELECT *
                    FROM prediction_history
                    WHERE flagged_for_review = 1
                    ORDER BY id DESC
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset),
                ).fetchall()
            else:
                total = conn.execute(
                    "SELECT COUNT(*) FROM prediction_history"
                ).fetchone()[0]

                rows = conn.execute(
                    """
                    SELECT *
                    FROM prediction_history
                    ORDER BY id DESC
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset),
                ).fetchall()

    except sqlite3.Error as error:
        raise HTTPException(
            status_code=500,
            detail="Could not read prediction history.",
        ) from error

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "flagged_only": flagged_only,
        "items": [dict(row) for row in rows],
    }


# ==================================================
# 12. GET ONE HISTORY RECORD
# ==================================================

@app.get("/history/{record_id}")
def get_history_record(record_id: int):
    try:
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM prediction_history
                WHERE id = ?
                """,
                (record_id,),
            ).fetchone()

    except sqlite3.Error as error:
        raise HTTPException(
            status_code=500,
            detail="Could not read prediction history.",
        ) from error

    if row is None:
        raise HTTPException(
            status_code=404,
            detail="Prediction record not found.",
        )

    return dict(row)


# ==================================================
# 13. API INFORMATION
# ==================================================

@app.get("/about")
def about():
    return {
        "name": "CardShield Fraud Detection API",
        "version": app.version,
        "model_file": MODEL_PATH.name,
        "feature_count": len(features),
        "batch_limit": 500,
        "history_database": DB_PATH.name,
        "disclaimer": (
            "This API provides model-based risk alerts. "
            "It does not independently confirm fraud."
        ),
    }
