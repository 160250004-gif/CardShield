
import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fraud_model.pkl"
DB_PATH = BASE_DIR / "fraud_monitoring.db"
DEMO_PATH = BASE_DIR / "demo_transactions.csv"
DATA_PATH = BASE_DIR / "creditcard.csv"
THRESHOLD = 0.50


def init_db():
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS live_transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                upload_id TEXT NOT NULL,
                file_name TEXT NOT NULL,
                row_number INTEGER NOT NULL,
                amount REAL,
                fraud_score REAL NOT NULL,
                prediction TEXT NOT NULL,
                flagged_for_review INTEGER NOT NULL,
                actual_class INTEGER,
                transaction_data TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH.name}"
        )

    bundle = joblib.load(MODEL_PATH)

    if not isinstance(bundle, dict):
        raise ValueError("fraud_model.pkl must contain a model bundle.")

    model = bundle.get("model")
    features = bundle.get("features")

    if model is None or not features:
        raise ValueError("Model or model features are missing.")

    if not hasattr(model, "predict_proba"):
        raise ValueError("The model must support predict_proba().")

    return model, list(features)


@st.cache_data
def load_sample_data(features):
    # Prefer the small demo file to avoid loading the large dataset.
    for path in (DEMO_PATH, DATA_PATH):
        if not path.exists():
            continue

        try:
            header = pd.read_csv(path, nrows=0).columns.tolist()
            if not all(col in header for col in features):
                continue

            sample = pd.read_csv(path, usecols=features)

            if not sample.empty:
                for col in features:
                    sample[col] = pd.to_numeric(
                        sample[col], errors="coerce"
                    )

                sample = sample.replace(
                    [np.inf, -np.inf], np.nan
                ).dropna()

                if not sample.empty:
                    return sample, path.name

        except Exception:
            continue

    raise ValueError(
        "No usable sample dataset was found. "
        "demo_transactions.csv or creditcard.csv must contain "
        "all model feature columns."
    )


def save_simulated_transaction(
    transaction, score, prediction, simulation_id, row_number
):
    amount = float(transaction.get("Amount", 0.0))
    flagged = int(prediction == "Potential Fraud")

    record = {
        key: float(value)
        for key, value in transaction.items()
    }

    record["simulation_id"] = simulation_id
    record["fraud_score"] = float(score)
    record["prediction"] = prediction
    record["source"] = "Automatic Simulator"

    now = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        conn.execute("""
            INSERT INTO live_transactions (
                upload_id, file_name, row_number, amount,
                fraud_score, prediction, flagged_for_review,
                actual_class, transaction_data, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            simulation_id,
            "Automatic Simulator",
            row_number,
            amount,
            float(score),
            prediction,
            flagged,
            None,
            json.dumps(record),
            now,
        ))


def render_live_monitoring():
    init_db()

    st.caption(
        "Automatic sample transaction simulator. "
        "This is a demonstration, not a live bank connection."
    )

    try:
        model, features = load_model()
        sample_data, source_name = load_sample_data(features)
    except Exception as exc:
        st.error(f"Simulator could not start: {exc}")
        return

    st.subheader("Automatic Transaction Simulator")
    st.write(
        f"Model: **{type(model).__name__}**  |  "
        f"Threshold: **{THRESHOLD:.0%}**"
    )
    st.caption(f"Sample source: {source_name}")
    
    
    st.subheader("📂 Upload Transaction Data")

    phonepe_file = st.file_uploader(
        "Upload a transaction CSV or PDF file",
        type=["csv"],
        key="phonepe_upload",
    )

    if phonepe_file is not None:
        try:
            phonepe_df = pd.read_csv(phonepe_file)

            required_columns = [
                "Date", "Transaction Details", "Type", "Amount"
            ]

            missing = [
                col for col in required_columns
                if col not in phonepe_df.columns
            ]

            if missing:
                st.error(
                    "Missing columns: " + ", ".join(missing)
                )
            else:
                phonepe_df["Type"] = (
                    phonepe_df["Type"]
                    .astype(str).str.strip().str.upper()
                )
                phonepe_df["Amount"] = pd.to_numeric(
                    phonepe_df["Amount"], errors="coerce"
                ).fillna(0)

                debit = phonepe_df["Type"].eq("DEBIT")
                credit = phonepe_df["Type"].eq("CREDIT")

                details = (
                    phonepe_df["Transaction Details"]
                    .fillna("").astype(str).str.strip().str.lower()
                )

                repeated = details[debit].map(
                    details[debit].value_counts()
                ).fillna(0)

                phonepe_df["Risk Level"] = "Low"
                phonepe_df["Risk Reason"] = (
                    "No configured rule triggered"
                )

                medium = (
                    debit
                    & (phonepe_df["Amount"] >= 2000)
                    & (phonepe_df["Amount"] < 5000)
                )
                high = debit & (phonepe_df["Amount"] >= 5000)
                repeated_debit = debit & (repeated >= 3)

                phonepe_df.loc[medium, "Risk Level"] = "Medium"
                phonepe_df.loc[medium, "Risk Reason"] = (
                    "Debit between ₹2,000 and ₹4,999"
                )

                phonepe_df.loc[high, "Risk Level"] = "High"
                phonepe_df.loc[high, "Risk Reason"] = (
                    "Debit of ₹5,000 or more"
                )

                repeated_low = (
                    repeated_debit
                    & phonepe_df["Risk Level"].eq("Low")
                )
                phonepe_df.loc[repeated_low, "Risk Level"] = "Medium"
                phonepe_df.loc[repeated_low, "Risk Reason"] = (
                    "Same payment details appear 3 or more times"
                )

                st.caption(
                    "Rule-based risk indicators only, not confirmed fraud."
                )

                risk = phonepe_df["Risk Level"].value_counts()
                a, b, c = st.columns(3)
                a.metric("Low Risk", int(risk.get("Low", 0)))
                b.metric("Medium Risk", int(risk.get("Medium", 0)))
                c.metric("High Risk", int(risk.get("High", 0)))

                a, b, c = st.columns(3)
                a.metric("Transactions", len(phonepe_df))
                b.metric(
                    "Total Debits",
                    f"₹{phonepe_df.loc[debit, 'Amount'].sum():,.2f}",
                )
                c.metric(
                    "Total Credits",
                    f"₹{phonepe_df.loc[credit, 'Amount'].sum():,.2f}",
                )

                st.dataframe(
                    phonepe_df,
                    use_container_width=True,
                    hide_index=True,
                )

                st.download_button(
                    "Download Analysis Results",
                    data=phonepe_df.to_csv(
                        index=False
                    ).encode("utf-8-sig"),
                    file_name="phonepe_analysis_results.csv",
                    mime="text/csv",
                    key="download_phonepe_results",
                )

        except Exception as exc:
            st.error("Could not analyse the uploaded PhonePe CSV.")
            st.code(str(exc))


    if "sim_running" not in st.session_state:
        st.session_state.sim_running = False

    if "sim_id" not in st.session_state:
        st.session_state.sim_id = str(uuid.uuid4())

    if "sim_count" not in st.session_state:
        st.session_state.sim_count = 0

    if "sim_events" not in st.session_state:
        st.session_state.sim_events = []

    col1, col2, col3 = st.columns(3)

    if col1.button(
        "▶ Start Simulator",
        type="primary",
        use_container_width=True,
    ):
        st.session_state.sim_running = True
        st.rerun()

    if col2.button("⏸ Stop Simulator", use_container_width=True):
        st.session_state.sim_running = False
        st.rerun()

    if col3.button("🔄 Reset Session", use_container_width=True):
        st.session_state.sim_running = False
        st.session_state.sim_id = str(uuid.uuid4())
        st.session_state.sim_count = 0
        st.session_state.sim_events = []
        st.rerun()

    if st.session_state.sim_running:
        st.success("Simulator running — generating one transaction every 3 seconds.")
    else:
        st.info("Simulator stopped. Click Start Simulator to begin.")

    @st.fragment(run_every=3)
    def live_feed():
        if st.session_state.sim_running:
            # Pick a real feature row from the local sample data.
            # No real customer or card details are generated.
            index = int(np.random.randint(0, len(sample_data)))
            row = sample_data.iloc[index].copy()

            X = pd.DataFrame(
                [[row[col] for col in features]],
                columns=features,
            )

            probabilities = model.predict_proba(X)[0]
            classes = list(model.classes_)

            if 1 not in classes:
                st.error("Model does not contain fraud class 1.")
                return

            score = float(probabilities[classes.index(1)])
            prediction = (
                "Potential Fraud"
                if score >= THRESHOLD
                else "Normal"
            )

            st.session_state.sim_count += 1
            count = st.session_state.sim_count

            transaction = {
                col: float(row[col])
                for col in features
            }

            save_simulated_transaction(
                transaction=transaction,
                score=score,
                prediction=prediction,
                simulation_id=st.session_state.sim_id,
                row_number=count,
            )

            event = {
                "Transaction #": count,
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Amount": transaction.get("Amount", 0.0),
                "Fraud Probability": score,
                "Result": prediction,
            }

            st.session_state.sim_events.insert(0, event)
            st.session_state.sim_events = (
                st.session_state.sim_events[:100]
            )

            if prediction == "Potential Fraud":
                st.error(
                    f"🚨 Risk alert: Transaction #{count} "
                    f"has a model score of {score:.1%}. "
                    "Please review it."
                )

        events = st.session_state.sim_events

        total = st.session_state.sim_count
        flagged = sum(
            e["Result"] == "Potential Fraud" for e in events
        )

        a, b, c = st.columns(3)
        a.metric("Transactions Generated", total)
        b.metric("Alerts in Recent 100", flagged)
        c.metric(
            "Normal in Recent 100",
            len(events) - flagged,
        )

        st.subheader("Latest Simulated Transactions")

        if events:
            st.dataframe(
                pd.DataFrame(events),
                use_container_width=True,
                hide_index=True,
            )
            st.bar_chart(
                pd.DataFrame(events).set_index("Transaction #")[
                    "Fraud Probability"
                ]
            )
        else:
            st.info(
                "No transactions yet. Start the simulator to see results."
            )

        st.caption(
            "The simulator samples rows from the local dataset and "
            "runs model predictions on them. These are demonstrations, "
            "not newly observed bank transactions. A high score is not "
            "proof of fraud."
        )

    live_feed()


if __name__ == "__main__":
    render_live_monitoring()
