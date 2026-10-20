import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    accuracy_score, average_precision_score
)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    
    page_title="CardShield | Fraud Monitoring",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path("fraud_model.pkl")
DATA_PATH = Path("creditcard.csv")
COMPARISON_PATH = Path("model_comparison.csv")


# ---------------- HELPERS ----------------
@st.cache_resource
def load_model_bundle(path_string: str, modified_time: float):
    # modified_time invalidates cache if the model file is replaced.
    return joblib.load(path_string)


@st.cache_data
def load_dataset(path_string: str, modified_time: float):
    return pd.read_csv(path_string)


def money(value):
    return f"€{value:,.2f}"


def add_risk_labels(scores, threshold):
    # These are operational review bands, not calibrated real-world probabilities.
    review_cutoff = min(0.75, max(threshold, threshold + 0.25))
    return np.select(
        [scores >= review_cutoff, scores >= threshold],
        ["High review priority", "Needs review"],
        default="Low model score",
    )


def validate_feature_frame(data, features):
    missing = [col for col in features if col not in data.columns]
    if missing:
        return None, f"Missing required model feature columns: {', '.join(missing)}"

    frame = data[features].copy()
    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)

    if frame.isna().any().any():
        bad_cols = frame.columns[frame.isna().any()].tolist()
        return None, (
            "Some feature values are blank or non-numeric. Check columns: "
            + ", ".join(bad_cols)
        )
    if frame.empty:
        return None, "The uploaded CSV contains no transaction rows."
    return frame, None


def risk_color(value):
    if value == "High review priority":
        return "background-color: #ffe0e0; color: #8b0000; font-weight: 600"
    if value == "Needs review":
        return "background-color: #fff0c2; color: #704900; font-weight: 600"
    return "background-color: #e4f5e9; color: #14532d"


# ---------------- HEADER ----------------
st.title("💳 CardShield — Credit Card Fraud Monitoring")
st.caption(
    "Machine-learning decision support for historical data and uploaded transactions"
)

if not MODEL_PATH.exists():
    st.error("Model file `fraud_model.pkl` was not found.")
    st.info("Open the project folder in the terminal and run: `python train_model.py`")
    st.stop()

try:
    bundle = load_model_bundle(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    model = bundle["model"]
    features = bundle["features"]
    model_name = bundle.get("model_name", type(model).__name__)
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.title("CardShield")
page = st.sidebar.radio(
    "Workspace",
    [
        "Executive Overview",
        "Fraud Analytics",
        "Screen Transactions",
        "Model Evaluation",
        "Data Quality",
    ],
)
st.sidebar.divider()
st.sidebar.write("**Active model**")
st.sidebar.info(str(model_name))
threshold = st.sidebar.slider(
    "Alert threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05,
    help=(
        "Transactions with a model score at or above this value are flagged for review. "
        "A lower threshold usually catches more possible fraud but can create more false alerts."
    ),
)
st.sidebar.caption(
    "This is a demonstration tool. It is not connected to a bank, card network, or live transaction feed."
)

# Load the dataset when available.
df = None
dataset_error = None
if DATA_PATH.exists():
    try:
        df = load_dataset(str(DATA_PATH), DATA_PATH.stat().st_mtime)
    except Exception as exc:
        dataset_error = str(exc)

# ---------------- EXECUTIVE OVERVIEW ----------------
if page == "Executive Overview":
    st.subheader("Executive Overview")
    if df is None:
        st.warning("Historical dataset `creditcard.csv` is unavailable or could not be read.")
        if dataset_error:
            st.caption(dataset_error)
        st.write("You can still use **Screen Transactions** to upload a compatible CSV.")
        st.stop()

    if "Class" not in df.columns:
        st.error("The historical dataset must contain a `Class` label (0 = normal, 1 = fraud).")
        st.stop()

    total = len(df)
    fraud_count = int((df["Class"] == 1).sum())
    normal_count = int((df["Class"] == 0).sum())
    fraud_rate = fraud_count / total * 100 if total else 0
    amount_total = pd.to_numeric(df.get("Amount", pd.Series(dtype=float)), errors="coerce").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Historical transactions", f"{total:,}")
    c2.metric("Fraud-labelled rows", f"{fraud_count:,}")
    c3.metric("Normal-labelled rows", f"{normal_count:,}")
    c4.metric("Labelled fraud rate", f"{fraud_rate:.3f}%")

    st.caption(f"Total recorded transaction amount: {amount_total:,.2f} (dataset currency not verified).")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Label distribution")
        counts = df["Class"].map({0: "Normal", 1: "Fraud"}).fillna("Other").value_counts()
        fig, ax = plt.subplots()
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_ylabel("Rows")
        ax.set_xlabel("Dataset label")
        ax.set_title("Historical class distribution")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("#### Dataset snapshot")
        st.dataframe(df.head(8), use_container_width=True, hide_index=True)

    st.info(
        "These figures describe the CSV supplied with the project. They are not live bank statistics, "
        "and a dataset label is not a prediction made by this dashboard."
    )

# ---------------- FRAUD ANALYTICS ----------------
elif page == "Fraud Analytics":
    st.subheader("Fraud Analytics")
    if df is None:
        st.error("Put `creditcard.csv` in the project folder to use historical analytics.")
        if dataset_error:
            st.caption(dataset_error)
        st.stop()
    required = {"Class", "Amount"}
    if not required.issubset(df.columns):
        st.error("Historical analytics require `Class` and `Amount` columns.")
        st.stop()

    analytics = df.copy()
    analytics["Amount"] = pd.to_numeric(analytics["Amount"], errors="coerce")
    analytics = analytics.dropna(subset=["Amount", "Class"])

    fraud = analytics[analytics["Class"] == 1]
    normal = analytics[analytics["Class"] == 0]
    a, b, c = st.columns(3)
    a.metric("Fraud-labelled amount", f"{fraud['Amount'].sum():,.2f}")
    b.metric("Average fraud amount", f"{fraud['Amount'].mean():,.2f}" if len(fraud) else "N/A")
    c.metric("Median normal amount", f"{normal['Amount'].median():,.2f}" if len(normal) else "N/A")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Normal vs fraud-labelled rows")
        counts = analytics["Class"].value_counts().reindex([0, 1], fill_value=0)
        fig, ax = plt.subplots()
        ax.bar(["Normal (0)", "Fraud (1)"], counts.values)
        ax.set_ylabel("Transactions")
        ax.set_title("Class distribution")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("#### Amount distribution")
        fig, ax = plt.subplots()
        sns.histplot(
            data=analytics,
            x="Amount",
            hue="Class",
            bins=40,
            element="step",
            stat="count",
            common_norm=False,
            ax=ax,
        )
        ax.set_title("Transaction amount by dataset label")
        st.pyplot(fig)
        plt.close(fig)

    if "Time" in analytics.columns:
        st.markdown("#### Fraud-labelled transactions over time")
        analytics["Time"] = pd.to_numeric(analytics["Time"], errors="coerce")
        analytics["Time bucket"] = (analytics["Time"] // 3600).astype("Int64")
        hourly = analytics[analytics["Class"] == 1].groupby("Time bucket").size()
        if not hourly.empty:
            st.line_chart(hourly.rename("Fraud-labelled rows"))
        else:
            st.caption("No fraud-labelled rows were available for the time chart.")

    summary = analytics.groupby("Class")["Amount"].agg(
        Transaction_Count="count",
        Mean_Amount="mean",
        Median_Amount="median",
        Maximum_Amount="max",
        Total_Amount="sum",
    )
    summary.index = [
    {0: "Normal (0)", 1: "Fraud (1)"}.get(i, str(i))
    for i in summary.index
]
    st.markdown("#### Summary by known label")
    st.dataframe(summary.round(2), use_container_width=True)

# ---------------- SCREEN TRANSACTIONS ----------------
elif page == "Screen Transactions":
    st.subheader("Screen Transactions")
    st.write(
        "Upload transaction rows as a CSV. The file must contain every feature used during model training "
        "(typically `Time`, `V1`–`V28`, and `Amount`). Do not include personal card numbers or CVV data."
    )
    uploaded = st.file_uploader("Choose transaction CSV", type=["csv"], key="screen_upload")

    if uploaded is not None:
        try:
            incoming = pd.read_csv(uploaded)
            st.write(f"**Rows received:** {len(incoming):,} · **Columns:** {len(incoming.columns):,}")
            X_new, validation_error = validate_feature_frame(
                incoming.drop(columns=["Class"], errors="ignore"), features
            )
            if validation_error:
                st.error(validation_error)
            else:
                if not hasattr(model, "predict_proba"):
                    st.error("The saved model does not support probability scores (`predict_proba`).")
                    st.stop()

                scores = model.predict_proba(X_new)[:, 1]
                predictions = (scores >= threshold).astype(int)
                results = incoming.copy()
                results["Fraud_Score"] = np.round(scores, 5)
                results["Prediction"] = np.where(
                    predictions == 1, "Flag for review", "No alert"
                )
                results["Risk_Band"] = add_risk_labels(scores, threshold)
                results.insert(0, "Review_ID", [f"TXN-{i:06d}" for i in range(1, len(results) + 1)])

                high = int((results["Risk_Band"] == "High review priority").sum())
                flagged = int(predictions.sum())
                normal_pred = int((predictions == 0).sum())
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Rows screened", f"{len(results):,}")
                m2.metric("Flagged for review", f"{flagged:,}")
                m3.metric("High review priority", f"{high:,}")
                m4.metric("No alert", f"{normal_pred:,}")

                st.warning(
                    "A model alert is a triage signal, not proof of fraud. Review alerts with approved "
                    "procedures before taking action against a cardholder."
                )

                show_flagged_only = st.checkbox("Show flagged transactions only", value=False)
                min_score = st.slider(
                    "Minimum score to display",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.05,
                )
                visible = results[results["Fraud_Score"] >= min_score]
                if show_flagged_only:
                    visible = visible[visible["Prediction"] == "Flag for review"]

                sort_options = ["Fraud_Score", "Risk_Band"]
                sort_by = st.selectbox("Sort results by", sort_options, index=0)
                visible = visible.sort_values(sort_by, ascending=False if sort_by == "Fraud_Score" else True)

                display_cols = ["Review_ID", "Fraud_Score", "Prediction", "Risk_Band"] + [
                    col for col in incoming.columns if col not in ["Class"]
                ]
                display_cols = list(dict.fromkeys(col for col in display_cols if col in visible.columns))
                st.dataframe(
                    visible[display_cols].style.map(risk_color, subset=["Risk_Band"]),
                    use_container_width=True,
                    hide_index=True,
                )

                csv_bytes = results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download all screening results",
                    data=csv_bytes,
                    file_name="fraud_screening_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

                if "Class" in incoming.columns:
                    actual = pd.to_numeric(incoming["Class"], errors="coerce")
                    if actual.notna().all() and actual.isin([0, 1]).all():
                        actual = actual.astype(int)
                        st.markdown("#### Evaluation against supplied labels")
                        p, r, f, acc = st.columns(4)
                        p.metric("Precision", f"{precision_score(actual, predictions, zero_division=0):.3f}")
                        r.metric("Recall", f"{recall_score(actual, predictions, zero_division=0):.3f}")
                        f.metric("F1 score", f"{f1_score(actual, predictions, zero_division=0):.3f}")
                        acc.metric("Accuracy", f"{accuracy_score(actual, predictions):.3f}")
                        fig, ax = plt.subplots()
                        sns.heatmap(
                            confusion_matrix(actual, predictions, labels=[0, 1]),
                            annot=True,
                            fmt="d",
                            cmap="Blues",
                            xticklabels=["Normal", "Fraud"],
                            yticklabels=["Normal", "Fraud"],
                            ax=ax,
                        )
                        ax.set_xlabel("Predicted label")
                        ax.set_ylabel("Actual label")
                        ax.set_title("Confusion matrix")
                        st.pyplot(fig)
                        plt.close(fig)
        except Exception as exc:
            st.error(f"Could not read or screen this CSV: {exc}")

# ---------------- MODEL EVALUATION ----------------
elif page == "Model Evaluation":
    st.subheader("Model Evaluation")
    st.write("**Selected model:**", model_name)

    ap = bundle.get("average_precision")
    if ap is not None:
        st.metric("Saved test-set Average Precision", f"{float(ap):.4f}")
    else:
        st.info("Average Precision was not saved in the model bundle.")

    st.caption(
        "Average Precision is useful for imbalanced fraud datasets, but it should be reviewed alongside "
        "precision, recall, F1 score, false positives, false negatives, and operational cost."
    )

    if COMPARISON_PATH.exists():
        try:
            comparison = pd.read_csv(COMPARISON_PATH)
            st.markdown("#### Model comparison")
            st.dataframe(comparison, use_container_width=True, hide_index=True)
            if {"Model", "Average Precision"}.issubset(comparison.columns):
                plot_df = comparison[["Model", "Average Precision"]].copy()
                plot_df["Average Precision"] = pd.to_numeric(
                    plot_df["Average Precision"], errors="coerce"
                )
                plot_df = plot_df.dropna()
                if not plot_df.empty:
                    st.bar_chart(plot_df.set_index("Model"))
        except Exception as exc:
            st.error(f"Could not read model comparison file: {exc}")
    else:
        st.info("Optional file `model_comparison.csv` not found. Run `python train_model.py` if your training script creates it.")

    st.markdown("#### What to validate before real deployment")
    st.write(
        "- Evaluate on a time-based, untouched test set.\n"
        "- Check recall and precision at the chosen alert threshold.\n"
        "- Review false positives and false negatives with a domain expert.\n"
        "- Monitor data drift and model performance after deployment.\n"
        "- Keep an audit trail and human review for consequential decisions."
    )

# ---------------- DATA QUALITY ----------------
elif page == "Data Quality":
    st.subheader("Data Quality Checks")
    if df is None:
        st.error("Historical dataset `creditcard.csv` is unavailable.")
        if dataset_error:
            st.caption(dataset_error)
        st.stop()

    rows, cols = df.shape
    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{rows:,}")
    c2.metric("Columns", f"{cols:,}")
    c3.metric("Duplicate rows", f"{duplicate_rows:,}")

    st.write(f"**Missing cells:** {missing_cells:,}")
    missing_summary = df.isna().sum().rename("Missing values").to_frame()
    st.markdown("#### Missing values by column")
    st.dataframe(missing_summary[missing_summary["Missing values"] > 0], use_container_width=True)

    st.markdown("#### Column types and unique values")
    profile = pd.DataFrame({
        "Data type": df.dtypes.astype(str),
        "Non-null values": df.notna().sum(),
        "Unique values": df.nunique(dropna=True),
    })
    st.dataframe(profile, use_container_width=True)

    if "Class" in df.columns:
        invalid_labels = int((~df["Class"].isin([0, 1])).sum())
        st.write(f"**Class labels outside 0/1:** {invalid_labels:,}")
    st.caption(
        "Duplicate rows should be investigated before deleting them. In transaction data, duplicates may "
        "be meaningful repeated events or accidental duplicate records."
    )

# ---------------- FOOTER / STYLE ----------------
st.divider()
st.caption(
    "Portfolio / educational prototype • Uses a local dataset and saved model • No live banking integration"
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
      [data-testid="stMetric"] {
        background: var(--secondary-background-color);
        padding: 16px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.25);
      }
      h1 { font-weight: 750; letter-spacing: -0.5px; }
      div[data-testid="stSidebar"] { border-right: 1px solid rgba(128, 128, 128, 0.2); }
    </style>
    """,
    unsafe_allow_html=True,
)
