
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

# -------------------- CONFIGURATION --------------------
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "cardshield_live.db"

st.set_page_config(
    page_title="CardShield Live Monitoring",
    page_icon="🛡️",
    layout="wide",
)

# -------------------- HEADER --------------------
st.title("🛡️ CardShield — Live Transaction Monitoring")
st.caption(
    "Demo mode: transactions replayed from the local dataset. "
    "This is not a real bank feed."
)

# -------------------- DATABASE CHECK --------------------
if not DB_PATH.exists():
    st.warning(
        "Database not found. Please run live_simulator.py first."
    )
    st.stop()


# -------------------- LOAD DATA --------------------
@st.cache_data(ttl=3)
def load_transactions():
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            """
            SELECT
                transaction_id,
                transaction_time,
                amount,
                fraud_probability,
                prediction,
                risk_band,
                source
            FROM live_transactions
            ORDER BY transaction_time DESC
            """,
            conn,
        )


df = load_transactions()

if df.empty:
    st.info("No transactions found. Start live_simulator.py.")
    st.stop()

# -------------------- DATA CLEANING --------------------
df["transaction_time"] = pd.to_datetime(
    df["transaction_time"], errors="coerce"
)

df["amount"] = pd.to_numeric(
    df["amount"], errors="coerce"
).fillna(0)

df["fraud_probability"] = pd.to_numeric(
    df["fraud_probability"], errors="coerce"
).fillna(0)

df["risk_band"] = (
    df["risk_band"]
    .fillna("Unknown")
    .astype(str)
    .str.strip()
)

# -------------------- RISK CATEGORY MAPPING --------------------
# Low = Normal, Medium = Review, High = High Risk
risk_map = {
    "Low": "Normal",
    "Normal": "Normal",
    "Medium": "Review",
    "Review": "Review",
    "High": "High Risk",
    "High Risk": "High Risk",
}

df["display_risk"] = df["risk_band"].map(risk_map).fillna(
    df["risk_band"]
)

# -------------------- SUMMARY METRICS --------------------
total = len(df)

normal = int((df["display_risk"] == "Normal").sum())
review = int((df["display_risk"] == "Review").sum())
high_risk = int((df["display_risk"] == "High Risk").sum())
other = total - normal - review - high_risk

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Transactions", f"{total:,}")
c2.metric("Normal", f"{normal:,}")
c3.metric("Review", f"{review:,}")
c4.metric("High Risk", f"{high_risk:,}")

if other > 0:
    st.info(f"{other} transactions have an unrecognized risk category.")

st.divider()

# -------------------- CHARTS --------------------
left, right = st.columns(2)

with left:
    st.subheader("Risk Category Distribution")

    risk_counts = pd.Series(
        {
            "Normal": normal,
            "Review": review,
            "High Risk": high_risk,
        },
        name="Transactions",
    )

    st.bar_chart(risk_counts)

with right:
    st.subheader("Recent Transaction Amounts")

    chart_df = (
        df.dropna(subset=["transaction_time"])
        .sort_values("transaction_time")
        .tail(30)
    )

    if not chart_df.empty:
        amount_chart = chart_df.set_index("transaction_time")[["amount"]]
        st.line_chart(amount_chart)
    else:
        st.info("No valid transaction timestamps available.")

# -------------------- RECENT TRANSACTIONS --------------------
st.subheader("Recent Transactions")

display_df = df[
    [
        "transaction_id",
        "transaction_time",
        "amount",
        "fraud_probability",
        "prediction",
        "display_risk",
        "source",
    ]
].copy()

display_df = display_df.rename(
    columns={"display_risk": "risk_band"}
)

display_df["fraud_probability"] = (
    display_df["fraud_probability"].clip(0, 1) * 100
).map(lambda x: f"{x:.2f}%")

display_df["amount"] = display_df["amount"].map(
    lambda x: f"₹{x:,.2f}"
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
)

# -------------------- FOOTER --------------------
st.caption(
    "Risk categories are model-based demo classifications, "
    "not confirmed fraud. The data comes from a local demo dataset."
)

# -------------------- REFRESH --------------------
if st.button("🔄 Refresh Dashboard"):
    st.cache_data.clear()
    st.rerun()
