import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import hashlib
import hmac
import secrets
import re
import plotly.express as px

import pdfplumber
from live_monitoring import render_live_monitoring

from pathlib import Path
from sklearn.metrics import(
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

# Use the dataset filename that actually exists in your project folder
DATA_PATH = Path("creditcard.csv")

COMPARISON_PATH = Path("model_comparison.csv")

# SQLite database used by Live Monitoring
LIVE_DB_PATH = Path(__file__).resolve().parent / "cardshield_live.db"

# ---------------- LOGIN / REGISTER ----------------
AUTH_DB = Path("users.db")
PBKDF2_ITERATIONS = 310_000


def init_auth_db():
    conn = sqlite3.connect(str(AUTH_DB), timeout=10)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    finally:
        conn.close()


def hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_bytes(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS
    )
    return salt.hex(), password_hash.hex()


def verify_password(password, salt_hex, saved_hash):
    try:
        salt = bytes.fromhex(salt_hex)
        _, calculated_hash = hash_password(password, salt)
        return hmac.compare_digest(calculated_hash, saved_hash)
    except (ValueError, TypeError):
        return False


def register_user(name, email, password):
    salt, password_hash = hash_password(password)
    conn = sqlite3.connect(str(AUTH_DB), timeout=10)

    try:
        conn.execute(
            """INSERT INTO users
               (name, email, salt, password_hash)
               VALUES (?, ?, ?, ?)""",
            (name, email, salt, password_hash)
        )
        conn.commit()
        return True, "Registration successful! Please log in."
    except sqlite3.IntegrityError:
        return False, "This email is already registered."
    finally:
        conn.close()


def authenticate_user(email, password):
    conn = sqlite3.connect(str(AUTH_DB), timeout=10)

    try:
        user = conn.execute(
            "SELECT name, email, salt, password_hash "
            "FROM users WHERE email = ?",
            (email,)
        ).fetchone()
    finally:
        conn.close()

    if user and verify_password(password, user[2], user[3]):
        return {"name": user[0], "email": user[1]}

    return None


def validate_password_strength(password):
    errors = []
    if len(password) < 8:
        errors.append("Password must have at least 8 characters.")
    if not re.search(r"[A-Z]", password):
        errors.append("Add at least one uppercase letter (A-Z).")
    if not re.search(r"[a-z]", password):
        errors.append("Add at least one lowercase letter (a-z).")
    if not re.search(r"\d", password):
        errors.append("Add at least one number (0-9).")
    if not re.search(r"[^A-Za-z0-9]", password):
        errors.append("Add at least one special character, such as @ or #.")
    return errors



# ---------------- AUTHENTICATION GATE ----------------
init_auth_db()

if "auth_user" not in st.session_state:
    st.session_state.auth_user = None

if st.session_state.auth_user is None:

    # Professional login page styling
    st.markdown("""
        <style>
        .stApp {
            background:
                radial-gradient(ellipse at 5% 5%, #172554 0%, transparent 42%),
                radial-gradient(ellipse at 95% 95%, #172033 0%, transparent 42%),
                #090d16;
        }

        .block-container {
            max-width: 1200px;
            padding-top: 3rem;
            padding-bottom: 3rem;
        }

        .auth-subtitle {
            color: #a5b4fc;
            font-size: 16px;
            line-height: 1.8;
        }

        .auth-feature {
            background: rgba(30, 41, 59, 0.55);
            border: 1px solid #303c53;
            padding: 14px 16px;
            border-radius: 12px;
            margin: 10px 0;
            color: #e2e8f0;
        }

        div[data-testid="stForm"] {
            background: rgba(20, 27, 43, 0.88);
            border: 1px solid #303c53;
            border-radius: 18px;
            padding: 24px;
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.25);
        }

        div[data-testid="stTextInput"] input {
            background: #111827;
            border: 1px solid #374151;
            border-radius: 10px;
            min-height: 44px;
        }

        div[data-testid="stTextInput"] input:focus {
            border-color: #60a5fa;
            box-shadow: 0 0 0 2px rgba(96, 165, 250, 0.15);
        }

        div[data-testid="stFormSubmitButton"] button {
            border-radius: 10px;
            min-height: 46px;
            font-weight: 600;
            transition: all 0.2s ease;
        }

        div[data-testid="stFormSubmitButton"] button:hover {
            filter: brightness(1.12);
            transform: translateY(-1px);
        }

        div[role="radiogroup"] {
            background: rgba(30, 41, 59, 0.5);
            border: 1px solid #303c53;
            border-radius: 10px;
            padding: 10px;
        }

        hr {
            border-color: #303c53;
            margin: 24px 0;
        }

        div[data-testid="stAlert"] {
            border-radius: 10px;
        }

        @media (max-width: 768px) {
            .block-container {
                padding: 1rem;
            }
        }
        </style>
    """, unsafe_allow_html=True)

    left, right = st.columns([1, 1.1], gap="large")

    # LEFT: Product branding and features
    with left:
        st.markdown("# 💳 CardShield")
        st.markdown("### Intelligent Fraud Protection")

        st.markdown(
            '<p class="auth-subtitle">'
            'A smarter workspace for transaction risk analysis, '
            'machine-learning insights and fraud monitoring.'
            '</p>',
            unsafe_allow_html=True,
        )

        st.divider()

        st.markdown("#### ✨ Your security workspace")

        st.markdown("""
        <div class="auth-feature">
            📊 <b>Fraud Intelligence</b><br>
            Explore historical transaction patterns and trends.
        </div>

        <div class="auth-feature">
            🤖 <b>Machine Learning</b><br>
            Screen uploaded transactions with your trained model.
        </div>

        <div class="auth-feature">
            📈 <b>Model Performance</b><br>
            Review precision, recall and model comparisons.
        </div>

        <div class="auth-feature">
            🔐 <b>Personal Workspace</b><br>
            Register and access your own account.
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        st.caption(
            "🛡️ Educational portfolio project. "
            "Not connected to a bank or live payment network."
        )

    # RIGHT: Login and registration
    with right:
        st.markdown("## Welcome")
        st.write("Sign in to continue or create your account.")

        login_mode = st.radio(
            "Choose account option",
            ["Login", "Register"],
            horizontal=True,
            key="auth_mode",
        )

        if login_mode == "Register":
            st.markdown("### Create your account")

            show_register_password = st.checkbox(
                "Show password",
                key="show_register_password",
            )

            with st.form("register_form"):
                name = st.text_input(
                    "Full name",
                    max_chars=100,
                    placeholder="Enter your name",
                )

                email = st.text_input(
                    "Email address",
                    max_chars=254,
                    placeholder="you@example.com",
                )

                password = st.text_input(
                    "Create password",
                    type="default" if show_register_password else "password",
                    help=(
                        "Use at least 8 characters, including uppercase, "
                        "lowercase, a number and a special character."
                    ),
                )

                confirm_password = st.text_input(
                    "Confirm password",
                    type="default" if show_register_password else "password",
                )

                register_clicked = st.form_submit_button(
                    "✨ Create Account",
                    use_container_width=True,
                    type="primary",
                )

            st.caption(
                "Use a strong password with uppercase, lowercase, "
                "a number and a special character."
            )

            if register_clicked:
                name = name.strip()
                email = email.strip().lower()

                if not name or not email or not password or not confirm_password:
                    st.error("Please fill in all fields.")

                elif not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
                    st.error("Please enter a valid email address.")

                elif password != confirm_password:
                    st.error("Passwords do not match.")

                else:
                    password_errors = validate_password_strength(password)

                    if password_errors:
                        st.error("Please improve your password:")
                        for error in password_errors:
                            st.write(f"- {error}")

                    else:
                        success, message = register_user(
                            name, email, password
                        )

                        if success:
                            st.success(message)
                            st.info(
                                "Select Login above to sign in."
                            )
                        else:
                            st.error(message)

        else:
            st.markdown("### Login to your account")

            show_login_password = st.checkbox(
                "Show password",
                key="show_login_password",
            )

            with st.form("login_form"):
                email = st.text_input(
                    "Email address",
                    placeholder="you@example.com",
                )

                password = st.text_input(
                    "Password",
                    type="default" if show_login_password else "password",
                )

                login_clicked = st.form_submit_button(
                    "🔐 Sign In",
                    use_container_width=True,
                    type="primary",
                )

            if login_clicked:
                email = email.strip().lower()

                user = authenticate_user(email, password)

                if user:
                    st.session_state.auth_user = user
                    st.rerun()
                else:
                    st.error(
                        "Incorrect email or password. "
                        "Check your details or register first."
                    )

    # Stop dashboard rendering until the user logs in
    st.stop()


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
    st.error("Model file fraud_model.pkl was not found.")
    st.info("Open the project folder in the terminal and run: python train_model.py")
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

if st.session_state.auth_user:
    st.sidebar.success(f"Welcome, {st.session_state.auth_user['name']}!")
    st.sidebar.caption(st.session_state.auth_user["email"])

    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state.auth_user = None
        st.rerun()

page = st.sidebar.radio(
    "Workspace",
    [
        "Executive Overview",
        "Fraud Analytics",
        "Live Monitoring",
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

# Load the historical dataset when available.
df = None
dataset_error = None

if DATA_PATH.exists():
    try:
        df = load_dataset(str(DATA_PATH), DATA_PATH.stat().st_mtime)
    except Exception as exc:
        dataset_error = str(exc)
else:
    dataset_error = (
        f"Dataset not found at: {DATA_PATH}. "
        "Upload a CSV from the Screen Transactions page."
    )

# ---------------- EXECUTIVE OVERVIEW ----------------


# ---------------- EXECUTIVE OVERVIEW ----------------
if page == "Executive Overview":
    st.markdown("""
        <style>
        .hero-card {
            padding: 30px;
            border-radius: 18px;
            background: linear-gradient(120deg, #101f46, #12334b);
            border: 1px solid #315477;
            margin-bottom: 24px;
        }
        .hero-card h1 {
            color: #f8fafc;
            font-size: 2.2rem;
            margin-bottom: 8px;
        }
        .hero-card p {
            color: #cbd5e1;
            font-size: 1.05rem;
            margin-bottom: 0;
        }
        .section-card {
            background: #171d2b;
            border: 1px solid #30394a;
            border-radius: 14px;
            padding: 20px;
            min-height: 155px;
        }
        .section-card h3 {
            color: #f8fafc;
            margin-top: 0;
        }
        .section-card p {
            color: #cbd5e1;
            font-size: 0.95rem;
        }
        .workflow {
            background: #101827;
            border: 1px solid #30394a;
            border-radius: 14px;
            padding: 20px;
            margin-top: 12px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="hero-card">
            <h1>🛡️ CardShield Fraud Intelligence</h1>
            <p>
                Credit Card Fraud Detection & Transaction Risk Analysis
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.subheader("Welcome to CardShield")
    st.write(
        "Screen supported transaction files, estimate fraud risk, "
        "and review clear prediction results. Upload your own data "
        "from the Transaction Screening or Live Monitoring page."
    )

    st.markdown("### Core Capabilities")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
            <div class="section-card">
                <h3>🔍 Fraud Detection</h3>
                <p>
                    Use the trained machine-learning model to estimate
                    whether supported credit-card transactions may be
                    fraudulent.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
            <div class="section-card">
                <h3>📊 Transaction Risk Analysis</h3>
                <p>
                    Review prediction labels, fraud probability,
                    transaction summaries, and downloadable results
                    after uploading compatible data.
                </p>
            </div>
        """, unsafe_allow_html=True)

    c3, c4 = st.columns(2)

    with c3:
        st.markdown("""
            <div class="section-card">
                <h3>📁 Flexible File Upload</h3>
                <p>
                    Upload a supported CSV or PDF. The file must contain
                    the columns or format required by its analysis method.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
            <div class="section-card">
                <h3>⚙️ Automated Monitoring</h3>
                <p>
                    Use Live Monitoring to observe the demonstration
                    simulator and its transaction risk predictions.
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("### How It Works")

    st.markdown("""
        <div class="workflow">
            <b>1. Upload</b> — Select a supported transaction file.<br><br>
            <b>2. Validate</b> — Check whether the file has the required format.<br><br>
            <b>3. Analyze</b> — Apply the relevant model or risk rules.<br><br>
            <b>4. Review</b> — Inspect predictions and download results.
        </div>
    """, unsafe_allow_html=True)

    st.info(
        "Privacy & scope: CardShield is an educational prototype. "
        "It does not connect to live banking systems. Predictions are "
        "estimates, not proof of fraud. No historical dataset statistics "
        "are displayed on this Overview page."
    )

  # ---------------- SCREEN TRANSACTIONS ----------------
elif page == "Screen Transactions":
    st.title("🔍 CardShield | Transaction Screening")
    st.caption(
        "Upload PhonePe PDF statements or CSV files for transaction analysis."
    )

    uploaded_files = st.file_uploader(
        "Upload PDF and CSV files",
        type=["pdf", "csv"],
        accept_multiple_files=True,
        key="combined_transaction_upload"
    )

    if uploaded_files:
        all_transactions = []

        for uploaded_file in uploaded_files:
            try:
                if uploaded_file.name.lower().endswith(".csv"):
                    df = pd.read_csv(uploaded_file)

                    # Normalize common CSV column names
                    rename_map = {}
                    for col in df.columns:
                        name = str(col).strip().lower()
                        if name in ["date", "transaction date"]:
                            rename_map[col] = "Date"
                        elif name in ["description", "merchant", "details", "narration"]:
                            rename_map[col] = "Description"
                        elif name in ["type", "transaction type", "dr/cr"]:
                            rename_map[col] = "Type"
                        elif name in ["amount", "transaction amount"]:
                            rename_map[col] = "Amount"

                    df = df.rename(columns=rename_map)

                    if "Amount" in df.columns:
                        df["Amount"] = pd.to_numeric(
                            df["Amount"].astype(str).str.replace(
                                r"[₹,\s]", "", regex=True
                            ),
                            errors="coerce"
                        )

                    df["Source File"] = uploaded_file.name
                    all_transactions.append(df)

                elif uploaded_file.name.lower().endswith(".pdf"):
                    import datetime

                    records = []
                    date_pattern = re.compile(
                        r"^(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
                        r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
                        r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
                        r"Dec(?:ember)?)\s+(\d{1,2}),\s*(\d{4})\s+(.*)$",
                        re.IGNORECASE
                    )
                    amount_pattern = re.compile(
                        r"\b(DEBIT|CREDIT)\s*₹\s*([\d,]+(?:\.\d{1,2})?)",
                        re.IGNORECASE
                    )

                    with pdfplumber.open(uploaded_file) as pdf:
                        for page_no, page_obj in enumerate(pdf.pages, 1):
                            text = page_obj.extract_text() or ""
                            blocks = []
                            current = []

                            for line in text.splitlines():
                                line = line.strip()
                                if date_pattern.match(line):
                                    if current:
                                        blocks.append(current)
                                    current = [line]
                                elif current:
                                    current.append(line)

                            if current:
                                blocks.append(current)

                            for block in blocks:
                                match = date_pattern.match(block[0])
                                if not match:
                                    continue

                                month, day, year, remainder = match.groups()
                                try:
                                    date_value = datetime.datetime.strptime(
                                        f"{month[:3].title()} {day}, {year}",
                                        "%b %d, %Y"
                                    ).date()
                                except ValueError:
                                    continue

                                amount_match = amount_pattern.search(remainder)
                                if not amount_match:
                                    continue

                                txn_type = amount_match.group(1).upper()
                                amount = float(
                                    amount_match.group(2).replace(",", "")
                                )
                                description = remainder[:amount_match.start()].strip()

                                txn_time = ""
                                transaction_id = ""
                                utr = ""

                                for detail in block[1:]:
                                    time_match = re.match(
                                        r"^\s*(\d{4})\s*(am|pm)\b",
                                        detail,
                                        re.IGNORECASE
                                    )
                                    if time_match:
                                        digits, ampm = time_match.groups()
                                        hour = int(digits[:2])
                                        minute = int(digits[2:])
                                        if hour <= 12 and minute < 60:
                                            hour = hour % 12
                                            if ampm.lower() == "pm":
                                                hour += 12
                                            txn_time = f"{hour:02d}:{minute:02d}"

                                    tid = re.search(
                                        r"Transaction ID\s+([A-Za-z0-9]+)",
                                        detail, re.IGNORECASE
                                    )
                                    if tid:
                                        transaction_id = tid.group(1)

                                    utr_match = re.search(
                                        r"UTR No\.?\s*([A-Za-z0-9]+)",
                                        detail, re.IGNORECASE
                                    )
                                    if utr_match:
                                        utr = utr_match.group(1)

                                records.append({
                                    "Date": pd.Timestamp(date_value),
                                    "Time": txn_time,
                                    "Description": description,
                                    "Type": txn_type,
                                    "Amount": amount,
                                    "Transaction ID": transaction_id,
                                    "UTR": utr,
                                    "Source File": uploaded_file.name,
                                    "Source Page": page_no
                                })

                    if records:
                        all_transactions.append(pd.DataFrame(records))
                        st.success(
                            f"Extracted {len(records)} transactions from "
                            f"{uploaded_file.name}"
                        )
                    else:
                        st.warning(
                            f"No transactions were recognized in "
                            f"{uploaded_file.name}. The parser expects a "
                            "text-based PhonePe PDF in a supported format."
                        )

            except Exception as exc:
                st.error(f"Could not process {uploaded_file.name}: {exc}")

        if all_transactions:
            combined_df = pd.concat(
                all_transactions, ignore_index=True, sort=False
            )

            if "Amount" not in combined_df.columns:
                st.error("No Amount column found. Check the CSV column names.")
                st.stop()

            combined_df["Amount"] = pd.to_numeric(
                combined_df["Amount"], errors="coerce"
            )
            combined_df = combined_df.dropna(subset=["Amount"]).copy()

            if "Type" not in combined_df.columns:
                combined_df["Type"] = "Unknown"

            combined_df["Type"] = (
                combined_df["Type"].astype(str).str.upper().str.strip()
            )

            if "Date" in combined_df.columns:
                combined_df["Date"] = pd.to_datetime(
                    combined_df["Date"], errors="coerce"
                )

            st.divider()
            st.subheader("📊 Transaction Intelligence Dashboard")

            # Filters
            f1, f2 = st.columns(2)
            with f1:
                types = sorted(combined_df["Type"].dropna().unique().tolist())
                selected_types = st.multiselect(
                    "Transaction type",
                    types,
                    default=types
                )
            with f2:
                max_amount = float(combined_df["Amount"].max())
                min_amount = float(combined_df["Amount"].min())
                amount_range = st.slider(
                    "Amount range",
                    min_value=min_amount,
                    max_value=max_amount if max_amount > min_amount else min_amount + 1.0,
                    value=(min_amount, max_amount)
                )

            filtered = combined_df[
                combined_df["Type"].isin(selected_types)
                & combined_df["Amount"].between(
                    amount_range[0], amount_range[1]
                )
            ].copy()

            debit_mask = filtered["Type"].str.contains(
                "DEBIT|DR|WITHDRAW", regex=True, na=False
            )
            credit_mask = filtered["Type"].str.contains(
                "CREDIT|CR|DEPOSIT", regex=True, na=False
            )

            total_debit = filtered.loc[debit_mask, "Amount"].sum()
            total_credit = filtered.loc[credit_mask, "Amount"].sum()
            avg_amount = filtered["Amount"].mean() if not filtered.empty else 0

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Transactions", f"{len(filtered):,}")
            k2.metric("Total Debit", f"₹{total_debit:,.2f}")
            k3.metric("Total Credit", f"₹{total_credit:,.2f}")
            k4.metric("Average Amount", f"₹{avg_amount:,.2f}")

            if filtered.empty:
                st.warning("No transactions match the selected filters.")
            else:
                left, right = st.columns(2)

                with left:
                    st.markdown("#### Debit vs Credit")
                    flow_df = pd.DataFrame({
                        "Category": ["Debit", "Credit"],
                        "Amount": [total_debit, total_credit]
                    })
                    fig = px.pie(
                        flow_df,
                        names="Category",
                        values="Amount",
                        hole=0.58,
                        title="Money Flow Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with right:
                    st.markdown("#### Transaction Amounts")
                    fig = px.histogram(
                        filtered,
                        x="Amount",
                        color="Type",
                        nbins=25,
                        title="Amount Distribution",
                        marginal="box"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                if "Date" in filtered.columns and filtered["Date"].notna().any():
                    daily = (
                        filtered.dropna(subset=["Date"])
                        .assign(Day=lambda x: x["Date"].dt.date)
                        .groupby(["Day", "Type"], as_index=False)
                        .agg(
                            Transactions=("Amount", "size"),
                            Total_Amount=("Amount", "sum")
                        )
                    )

                    st.markdown("#### Daily Transaction Trend")
                    fig = px.line(
                        daily,
                        x="Day",
                        y="Total_Amount",
                        color="Type",
                        markers=True,
                        title="Daily Transaction Value"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    fig = px.bar(
                        daily,
                        x="Day",
                        y="Transactions",
                        color="Type",
                        barmode="group",
                        title="Daily Transaction Count"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                if "Description" in filtered.columns:
                    top = (
                        filtered.assign(
                            Description=filtered["Description"]
                            .fillna("Unknown").astype(str)
                        )
                        .groupby("Description", as_index=False)
                        .agg(
                            Total_Amount=("Amount", "sum"),
                            Transactions=("Amount", "size")
                        )
                        .sort_values("Total_Amount", ascending=False)
                        .head(10)
                    )

                    st.markdown("#### Top 10 Transaction Descriptions")
                    fig = px.bar(
                        top.sort_values("Total_Amount"),
                        x="Total_Amount",
                        y="Description",
                        orientation="h",
                        hover_data=["Transactions"],
                        title="Highest Transaction Values by Description"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### Combined Transaction Table")
                st.dataframe(
                    filtered,
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "⬇️ Download Filtered Transactions CSV",
                    data=filtered.to_csv(index=False).encode("utf-8"),
                    file_name="cardshield_filtered_transactions.csv",
                    mime="text/csv"
                )

        else:
            st.warning("No transactions were extracted from the selected files.")



# ---------------- MODEL EVALUATION ----------------
elif page == "Model Evaluation":

    st.markdown("""
        <style>
        .model-heading {
            font-size: 30px;
            font-weight: 750;
            color: #38bdf8;
            margin-bottom: 4px;
        }
        .model-subtitle {
            color: #94a3b8;
            margin-bottom: 22px;
        }
        </style>
        <div class="model-heading">📊 Model Performance Center</div>
        <div class="model-subtitle">
            Compare machine learning models and review fraud detection metrics.
        </div>
    """, unsafe_allow_html=True)

    st.caption(f"Active model: {model_name}")

    # Load model comparison data
    comparison = None

    if COMPARISON_PATH.exists():
        try:
            comparison = pd.read_csv(COMPARISON_PATH)
        except Exception as exc:
            st.error(f"Could not load comparison data: {exc}")

    required_cols = [
        "Model", "Average Precision", "Precision", "Recall", "F1 Score"
    ]

    if comparison is not None and all(
        col in comparison.columns for col in required_cols
    ):
        comparison = comparison.copy()

        for col in required_cols[1:]:
            comparison[col] = pd.to_numeric(
                comparison[col], errors="coerce"
            )

        comparison = comparison.dropna(subset=required_cols[1:])

        active_rows = comparison[
            comparison["Model"].astype(str).str.casefold()
            == str(model_name).casefold()
        ]

        if active_rows.empty:
            active_rows = comparison.tail(1)

        if not comparison.empty and not active_rows.empty:
            active = active_rows.iloc[0]

            # KPI cards
            st.subheader("Key Performance Indicators")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Average Precision",
                f"{active['Average Precision']:.4f}"
            )
            c2.metric(
                "Precision",
                f"{active['Precision']:.4f}"
            )
            c3.metric(
                "Recall",
                f"{active['Recall']:.4f}"
            )
            c4.metric(
                "F1 Score",
                f"{active['F1 Score']:.4f}"
            )

            st.caption(
                "These metrics come from the saved model comparison results. "
                "They describe the evaluation recorded during training."
            )

            st.divider()

            # Model comparison table and chart
            left, right = st.columns([1.25, 1])

            with left:
                st.subheader("Model Comparison")

                styled = (
                    comparison.style
                    .format({
                        "Average Precision": "{:.4f}",
                        "Precision": "{:.4f}",
                        "Recall": "{:.4f}",
                        "F1 Score": "{:.4f}",
                    })
                )

                st.dataframe(
                    styled,
                    use_container_width=True,
                    hide_index=True
                )

            with right:
                st.subheader("Average Precision")

                chart_df = comparison[
                    ["Model", "Average Precision"]
                ].set_index("Model")

                st.bar_chart(
                    chart_df,
                    y="Average Precision",
                    color="#38bdf8",
                    height=300
                )

            st.divider()

            # Compare all metrics
            st.subheader("Performance Across Metrics")

            metric_chart = comparison.set_index("Model")[
                ["Precision", "Recall", "F1 Score"]
            ]

            st.bar_chart(
                metric_chart,
                height=350
            )

            # Active model insight
            st.subheader("Model Insights")

            st.markdown(
                f"""
                **Selected model:** {model_name}

                - **Precision:** {active['Precision']:.4f}
                - **Recall:** {active['Recall']:.4f}
                - **F1 Score:** {active['F1 Score']:.4f}
                - **Average Precision:** {active['Average Precision']:.4f}
                """
            )

            st.info(
                "Fraud detection uses imbalanced data. A high precision score "
                "does not mean every fraud case is caught. Review recall, "
                "false negatives, false positives and the chosen alert threshold."
            )

            # Export comparison
            st.download_button(
                "⬇️ Download Model Comparison CSV",
                data=comparison.to_csv(index=False).encode("utf-8"),
                file_name="cardshield_model_comparison.csv",
                mime="text/csv",
                use_container_width=True
            )

        else:
            st.warning("No valid model comparison rows are available.")

    else:
        st.warning(
            "Model comparison data is missing or does not contain the expected columns. "
            "Check model_comparison.csv or run your training script."
        )

        ap = bundle.get("average_precision")

        if ap is not None:
            st.metric(
                "Saved Test-Set Average Precision",
                f"{float(ap):.4f}"
            )

    st.divider()

    st.subheader("Deployment Checklist")

    st.markdown("""
    - Evaluate on an untouched test dataset.
    - Check precision and recall at different thresholds.
    - Review false positives and false negatives.
    - Monitor data quality and model drift.
    - Require human review before consequential action.
    """)

    # Prepare a safe analytics copy
    overview = df.copy()
    overview["Class"] = pd.to_numeric(
        overview["Class"], errors="coerce"
    )
    overview["Amount"] = pd.to_numeric(
        overview["Amount"], errors="coerce"
    )
    overview = overview.replace([np.inf, -np.inf], np.nan)
    overview = overview.dropna(subset=["Class", "Amount"])
    overview = overview[overview["Class"].isin([0, 1])].copy()

    if overview.empty:
        st.warning("No valid transactions available.")
        st.stop()

    # Interactive filters
    st.subheader("🔎 Analytics Filters")

    filter_col1, filter_col2 = st.columns([2, 1])

    amount_min = float(overview["Amount"].min())
    amount_max = float(overview["Amount"].max())

    with filter_col1:
        if amount_min < amount_max:
            selected_amount = st.slider(
                "Transaction amount",
                min_value=amount_min,
                max_value=amount_max,
                value=(amount_min, amount_max),
                key="overview_amount_filter",
            )
            filtered = overview[
                overview["Amount"].between(
                    selected_amount[0], selected_amount[1]
                )
            ].copy()
        else:
            filtered = overview.copy()

    with filter_col2:
        label_filter = st.selectbox(
            "Transaction label",
            ["All", "Normal", "Fraud"],
            key="overview_label_filter",
        )

    if label_filter == "Normal":
        filtered = filtered[filtered["Class"] == 0]
    elif label_filter == "Fraud":
        filtered = filtered[filtered["Class"] == 1]

    if filtered.empty:
        st.warning("No transactions match the selected filters.")
        st.stop()

    # KPI calculations
    total = len(filtered)
    fraud_count = int((filtered["Class"] == 1).sum())
    normal_count = int((filtered["Class"] == 0).sum())
    fraud_rate = fraud_count / total * 100
    total_amount = float(filtered["Amount"].sum())
    avg_amount = float(filtered["Amount"].mean())

    st.subheader("📊 Key Risk Indicators")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Transactions", f"{total:,}")
    k2.metric("Fraud-labelled", f"{fraud_count:,}")
    k3.metric("Normal-labelled", f"{normal_count:,}")
    k4.metric("Labelled fraud rate", f"{fraud_rate:.3f}%")

    k5, k6 = st.columns(2)
    k5.metric("Total transaction amount", f"{total_amount:,.2f}")
    k6.metric("Average transaction amount", f"{avg_amount:,.2f}")

    st.caption(
        "Amounts are shown in dataset units; currency is unverified. "
        "Fraud metrics describe existing dataset labels, not live predictions."
    )

    st.divider()

    # Consistent dark chart theme
    plt.style.use("dark_background")

    left, right = st.columns([1, 1])

    # Donut chart
    with left:
        st.subheader("🍩 Transaction Distribution")

        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")

        values = [normal_count, fraud_count]
        colors = ["#22d3ee", "#fb7185"]

        if sum(values) > 0:
            wedges, _, _ = ax.pie(
                values,
                colors=colors,
                startangle=90,
                autopct=lambda p: f"{p:.2f}%" if p >= 0.1 else "",
                pctdistance=0.78,
                wedgeprops={
                    "width": 0.40,
                    "edgecolor": "#0e1117",
                },
                textprops={"color": "white"},
            )

            ax.text(
                0, 0,
                f"{total:,}\nRows",
                ha="center",
                va="center",
                color="white",
                fontsize=15,
                fontweight="bold",
            )

            ax.legend(
                wedges,
                [
                    f"Normal: {normal_count:,}",
                    f"Fraud: {fraud_count:,}",
                ],
                loc="lower center",
                bbox_to_anchor=(0.5, -0.12),
                ncol=2,
                frameon=False,
            )

        ax.set_title("Labelled Transaction Mix")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Transaction amount chart
    with right:
        st.subheader("💳 Transaction Amount Analysis")

        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")

        for label, color, name in [
            (0, "#22d3ee", "Normal"),
            (1, "#fb7185", "Fraud"),
        ]:
            values = filtered.loc[
                filtered["Class"] == label, "Amount"
            ]

            if not values.empty:
                sns.histplot(
                    values,
                    bins=30,
                    element="step",
                    stat="count",
                    color=color,
                    alpha=0.45,
                    label=name,
                    ax=ax,
                )

        ax.set_xlabel("Transaction amount")
        ax.set_ylabel("Number of rows")
        ax.set_title("Amount Distribution")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.divider()

    # Time trend
    if "Time" in filtered.columns:
        st.subheader("📈 Transaction Activity Over Time")

        time_data = filtered[["Time", "Class"]].copy()
        time_data["Time"] = pd.to_numeric(
            time_data["Time"], errors="coerce"
        )
        time_data = time_data.dropna(subset=["Time"])

        if not time_data.empty:
            time_data["Time interval"] = (
                time_data["Time"] // 3600
            ).astype(int)

            trend = time_data.groupby("Time interval").agg(
                Transactions=("Class", "size"),
                Fraud=("Class", "sum"),
            )

            st.line_chart(
                trend,
                use_container_width=True,
            )

            st.caption(
                "Time intervals represent elapsed hours derived from "
                "the dataset's Time column, not actual calendar hours."
            )

    # Preview and export
    st.divider()
    st.subheader("📋 Filtered Transaction Preview")

    preview_cols = [
        c for c in ["Time", "Amount", "Class"]
        if c in filtered.columns
    ]
    st.dataframe(
        filtered[preview_cols].head(100),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇️ Export Filtered Dataset",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="cardshield_overview_report.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.caption(
        "CardShield | Machine-learning fraud monitoring portfolio project"
    )




# ---------------- FRAUD ANALYTICS ----------------

# ---------------- FRAUD ANALYTICS ----------------
elif page == "Fraud Analytics":
    st.title("📊 CardShield | Advanced Fraud Analytics")
    st.caption(
        "Historical transaction intelligence • Fraud patterns • Risk monitoring"
    )

    if df is None:
        st.error("creditcard.csv could not be loaded.")
        if dataset_error:
            st.caption(dataset_error)
        st.stop()

    if not {"Class", "Amount"}.issubset(df.columns):
        st.error("Dataset must contain Class and Amount columns.")
        st.stop()

    analytics = df.copy()
    analytics["Class"] = pd.to_numeric(
        analytics["Class"], errors="coerce"
    )
    analytics["Amount"] = pd.to_numeric(
        analytics["Amount"], errors="coerce"
    )
    analytics = analytics.replace([np.inf, -np.inf], np.nan)
    analytics = analytics.dropna(subset=["Class", "Amount"])
    analytics = analytics[analytics["Class"].isin([0, 1])].copy()

    if analytics.empty:
        st.warning("No valid transaction rows available.")
        st.stop()

    # Dashboard filters
    st.subheader("🔎 Transaction Filters")
    col1, col2 = st.columns(2)

    with col1:
        min_amount = float(analytics["Amount"].min())
        max_amount = float(analytics["Amount"].max())

        if min_amount < max_amount:
            amount_range = st.slider(
                "Transaction amount range",
                min_value=min_amount,
                max_value=max_amount,
                value=(min_amount, max_amount),
                key="improved_amount_filter",
            )
            analytics = analytics[
                analytics["Amount"].between(
                    amount_range[0], amount_range[1]
                )
            ]

    with col2:
        time_col = (
            "Time"
            if "Time" in analytics.columns
            else None
        )

        if time_col and not analytics.empty:
            analytics["Time"] = pd.to_numeric(
                analytics["Time"], errors="coerce"
            )
            analytics = analytics.dropna(subset=["Time"])

            t_min = float(analytics["Time"].min())
            t_max = float(analytics["Time"].max())

            if t_min < t_max:
                time_range = st.slider(
                    "Elapsed time range (seconds)",
                    min_value=t_min,
                    max_value=t_max,
                    value=(t_min, t_max),
                    key="improved_time_filter",
                )
                analytics = analytics[
                    analytics["Time"].between(
                        time_range[0], time_range[1]
                    )
                ]

    if analytics.empty:
        st.warning("No transactions match these filters.")
        st.stop()

    # KPI calculations from the selected dataset
    total = len(analytics)
    fraud = analytics[analytics["Class"] == 1]
    genuine = analytics[analytics["Class"] == 0]

    fraud_count = len(fraud)
    genuine_count = len(genuine)
    fraud_rate = fraud_count / total * 100 if total else 0
    fraud_amount = float(fraud["Amount"].sum())
    average_amount = float(analytics["Amount"].mean())

    st.subheader("📌 Key Performance Indicators")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Transactions", f"{total:,}")
    c2.metric("Fraud-labelled rows", f"{fraud_count:,}")
    c3.metric("Labelled fraud rate", f"{fraud_rate:.3f}%")
    c4.metric("Fraud amount total", f"{fraud_amount:,.2f}")

    c1, c2 = st.columns(2)
    c1.metric("Genuine-labelled rows", f"{genuine_count:,}")
    c2.metric("Average transaction amount", f"{average_amount:,.2f}")

    st.info(
        "Charts below use the dataset's existing Class labels: "
        "0 = genuine-labelled and 1 = fraud-labelled. "
        "These are historical labels, not new model predictions. "
        "Currency is not verified."
    )

    st.divider()

    # Chart 1: Transaction counts
    st.subheader("📊 Genuine vs Fraud Transactions")
    left, right = st.columns(2)

    with left:
        count_data = pd.DataFrame({
            "Transaction type": ["Genuine", "Fraud"],
            "Transactions": [genuine_count, fraud_count],
        })

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(
            count_data["Transaction type"],
            count_data["Transactions"],
        )
        ax.set_title("Transaction Count by Label")
        ax.set_ylabel("Number of transactions")
        ax.bar_label(bars, fmt="%,.0f")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Chart 2: Total amount
    with right:
        amount_data = pd.DataFrame({
            "Transaction type": ["Genuine", "Fraud"],
            "Total amount": [
                float(genuine["Amount"].sum()),
                fraud_amount,
            ],
        })

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(
            amount_data["Transaction type"],
            amount_data["Total amount"],
        )
        ax.set_title("Total Amount by Label")
        ax.set_ylabel("Amount (currency unverified)")
        ax.bar_label(bars, fmt="%.2f")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.divider()

    # Chart 3: Amount distribution
    st.subheader("💳 Transaction Amount Distribution")
    fig, ax = plt.subplots(figsize=(10, 4))

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
    ax.set_title("Amount Distribution: Class 0 vs Class 1")
    ax.set_xlabel("Transaction amount")
    ax.set_ylabel("Transaction count")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Chart 4: Fraud trend over elapsed time
    if "Time" in analytics.columns:
        st.subheader("⏱️ Fraud Trend Over Time")

        trend = analytics.copy()
        trend["Time Bucket (hour)"] = (
            trend["Time"] // 3600
        ).astype(int)

       

    # Chart 5: Fraud amount distribution
    st.subheader("📈 Fraud Amount Statistics")

    if not fraud.empty:
        stats = fraud["Amount"].describe().to_frame(
            name="Fraud-labelled transactions"
        )
        st.dataframe(stats.round(2), use_container_width=True)
    else:
        st.info("No fraud-labelled transactions match the current filters.")

    # Download filtered dataset
    st.divider()
    st.subheader("⬇️ Export Analytics")

    csv_data = analytics.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download filtered analytics CSV",
        data=csv_data,
        file_name="cardshield_filtered_analytics.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.caption(
        "CardShield educational portfolio prototype • "
        "Not connected to live bank systems."
    )

    # Fraud vs genuine charts
   
    # ---------- NEW: FRAUD VS NORMAL DONUT + TIME TREND ----------
    st.subheader("📊 Fraud Detection Analytics")

    chart_left, chart_right = st.columns([1, 1.4])

    # 1. Donut chart
    with chart_left:
        st.markdown("#### 🍩 Fraud vs Normal Transactions")

        labels = ["Normal", "Fraud"]
        values = [genuine_count, fraud_count]

        fig, ax = plt.subplots(figsize=(6, 4))
        if sum(values) > 0:
            wedges, _, _ = ax.pie(
                values,
                startangle=90,
                colors=["#12C2E9", "#FF4B6E"],
                autopct=lambda p: f"{p:.2f}%" if p >= 0.1 else "",
                pctdistance=0.78,
                wedgeprops={
                    "width": 0.42,
                    "edgecolor": "#0e1117"
                },
                textprops={"color": "white"}
            )
            total_transactions = sum(values)

            ax.text(
                0, 0,
                f"{total_transactions:,}\nTotal",
                ha="center",
                va="center",
                color="white",
                fontsize=15,
                fontweight="bold"
            )
            ax.legend(
                wedges,
                [
                    f"Normal: {genuine_count:,}",
                    f"Fraud: {fraud_count:,}"
                ],
                loc="lower center",
                bbox_to_anchor=(0.5, -0.12),
                ncol=2,
                frameon=False
            )
            ax.set_title("Transaction Distribution", color="white")
            fig.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#0e1117")
            st.pyplot(fig, use_container_width=True)

        else:
            st.info("No transactions available for this chart.")

        plt.close(fig)

    # 2. Fraud detection over elapsed time
    with chart_right:
        st.markdown("#### 📈 Fraud Detection Over Time")

        if "Time" in analytics.columns:
            time_data = analytics[["Time", "Class"]].copy()
            time_data["Time"] = pd.to_numeric(
                time_data["Time"], errors="coerce"
            )
            time_data = time_data.dropna()

            if time_data["Time"].nunique() > 1:
                number_of_bins = min(
                    60, time_data["Time"].nunique()
                )

                time_data["Time_Bin"] = pd.cut(
                    time_data["Time"],
                    bins=number_of_bins,
                    duplicates="drop"
                )

                time_data["Elapsed Seconds"] = time_data[
                    "Time_Bin"
                ].map(lambda interval: interval.mid)

                trend = time_data.groupby(
                    "Time_Bin", observed=False
                ).agg(
                    Fraud=("Class", lambda x: (x == 1).sum()),
                    Normal=("Class", lambda x: (x == 0).sum())
                ).reset_index()

                trend["Elapsed Seconds"] = trend[
                    "Time_Bin"
                ].map(lambda interval: interval.mid)

                fig, ax = plt.subplots(figsize=(9, 4))

                ax.plot(
                    trend["Elapsed Seconds"],
                    trend["Normal"],
                    color="#12C2E9",
                    linewidth=2,
                    label="Normal"
                )

                ax.plot(
                    trend["Elapsed Seconds"],
                    trend["Fraud"],
                    color="#FF4B6E",
                    linewidth=2,
                    label="Fraud"
                )

                ax.set_xlabel("Elapsed Time (seconds)")
                ax.set_ylabel("Transactions per interval")
                ax.set_title("Fraud and Normal Transaction Trend")
                ax.legend()
                ax.grid(alpha=0.2)

                fig.patch.set_facecolor("#0e1117")
                ax.set_facecolor("#0e1117")
                ax.tick_params(colors="white")
                ax.xaxis.label.set_color("white")
                ax.yaxis.label.set_color("white")
                ax.title.set_color("white")
                ax.legend(
                    facecolor="#20232b",
                    labelcolor="white"
                )

                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            else:
                st.info(
                    "Not enough different Time values to draw a trend."
                )
        elif "_date" in analytics.columns:
            daily = analytics.copy()
            daily["Date"] = daily["_date"].dt.floor("D")

            trend = daily.groupby("Date").agg(
                Fraud=("Class", lambda x: (x == 1).sum()),
                Normal=("Class", lambda x: (x == 0).sum())
            )

            st.line_chart(trend)

        else:
            st.info(
                "A Time or real transaction date column is required."
            )

    st.caption(
        "Graphs use the filtered dataset and its existing Class labels. "
        "Time in the original dataset means elapsed seconds, not calendar dates."
    )

    # ---------- TRANSACTION AMOUNT DISTRIBUTION ----------
    st.subheader("💳 Transaction Amount Distribution")

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(
        data=analytics,
        x="Amount",
        hue="Class",
        bins=40,
        element="step",
        stat="count",
        common_norm=False,
        ax=ax
    )
    ax.set_title("Amount Distribution: Normal vs Fraud")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


    # Summary table
    st.subheader("📋 Fraud Summary")
    summary = analytics.groupby("Class").agg(
        Transactions=("Amount", "size"),
        Total_Amount=("Amount", "sum"),
        Average_Amount=("Amount", "mean"),
        Maximum_Amount=("Amount", "max"),
    )
    summary.index = summary.index.map({
        0: "Genuine",
        1: "Fraud",
    })
    st.dataframe(summary.round(2), use_container_width=True)

    # Download filtered rows
    st.subheader("⬇️ Export Filtered Data")
    st.download_button(
        "Download Filtered Transactions CSV",
        data=analytics.drop(
            columns=["_date"], errors="ignore"
        ).to_csv(index=False).encode("utf-8"),
        file_name="filtered_fraud_analytics.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ---------------- LIVE MONITORING ----------------
elif page == "Live Monitoring":

    st.title("📡 CardShield | Live Fraud Monitoring")
    st.caption(
        "Upload transaction data, detect potential fraud, "
        "and review the results."
    )

    try:
        from live_monitoring import render_live_monitoring

        render_live_monitoring()

    except Exception as exc:
        st.error("Live Monitoring could not be loaded.")
        st.code(str(exc))
        st.info(
            "Check that live_monitoring.py exists in the "
            "same project folder as app.py."
        )



# ---------------- SCREEN TRANSACTIONS ----------------
elif page == "Screen Transactions":
    st.title("🔍 CardShield | Transaction Screening")
    st.caption(
        "Upload PhonePe PDF statements or CSV files for transaction analysis."
    )

    uploaded_files = st.file_uploader(
        "Upload PDF and CSV files",
        type=["pdf", "csv"],
        accept_multiple_files=True,
        key="combined_transaction_upload"
    )

    if uploaded_files:
        all_transactions = []

        for uploaded_file in uploaded_files:
            try:
                if uploaded_file.name.lower().endswith(".csv"):
                    df = pd.read_csv(uploaded_file)

                    # Normalize common CSV column names
                    rename_map = {}
                    for col in df.columns:
                        name = str(col).strip().lower()
                        if name in ["date", "transaction date"]:
                            rename_map[col] = "Date"
                        elif name in ["description", "merchant", "details", "narration"]:
                            rename_map[col] = "Description"
                        elif name in ["type", "transaction type", "dr/cr"]:
                            rename_map[col] = "Type"
                        elif name in ["amount", "transaction amount"]:
                            rename_map[col] = "Amount"

                    df = df.rename(columns=rename_map)

                    if "Amount" in df.columns:
                        df["Amount"] = pd.to_numeric(
                            df["Amount"].astype(str).str.replace(
                                r"[₹,\s]", "", regex=True
                            ),
                            errors="coerce"
                        )

                    df["Source File"] = uploaded_file.name
                    all_transactions.append(df)

                elif uploaded_file.name.lower().endswith(".pdf"):
                    import datetime

                    records = []
                    date_pattern = re.compile(
                        r"^(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
                        r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
                        r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
                        r"Dec(?:ember)?)\s+(\d{1,2}),\s*(\d{4})\s+(.*)$",
                        re.IGNORECASE
                    )
                    amount_pattern = re.compile(
                        r"\b(DEBIT|CREDIT)\s*₹\s*([\d,]+(?:\.\d{1,2})?)",
                        re.IGNORECASE
                    )

                    with pdfplumber.open(uploaded_file) as pdf:
                        for page_no, page_obj in enumerate(pdf.pages, 1):
                            text = page_obj.extract_text() or ""
                            blocks = []
                            current = []

                            for line in text.splitlines():
                                line = line.strip()
                                if date_pattern.match(line):
                                    if current:
                                        blocks.append(current)
                                    current = [line]
                                elif current:
                                    current.append(line)

                            if current:
                                blocks.append(current)

                            for block in blocks:
                                match = date_pattern.match(block[0])
                                if not match:
                                    continue

                                month, day, year, remainder = match.groups()
                                try:
                                    date_value = datetime.datetime.strptime(
                                        f"{month[:3].title()} {day}, {year}",
                                        "%b %d, %Y"
                                    ).date()
                                except ValueError:
                                    continue

                                amount_match = amount_pattern.search(remainder)
                                if not amount_match:
                                    continue

                                txn_type = amount_match.group(1).upper()
                                amount = float(
                                    amount_match.group(2).replace(",", "")
                                )
                                description = remainder[:amount_match.start()].strip()

                                txn_time = ""
                                transaction_id = ""
                                utr = ""

                                for detail in block[1:]:
                                    time_match = re.match(
                                        r"^\s*(\d{4})\s*(am|pm)\b",
                                        detail,
                                        re.IGNORECASE
                                    )
                                    if time_match:
                                        digits, ampm = time_match.groups()
                                        hour = int(digits[:2])
                                        minute = int(digits[2:])
                                        if hour <= 12 and minute < 60:
                                            hour = hour % 12
                                            if ampm.lower() == "pm":
                                                hour += 12
                                            txn_time = f"{hour:02d}:{minute:02d}"

                                    tid = re.search(
                                        r"Transaction ID\s+([A-Za-z0-9]+)",
                                        detail, re.IGNORECASE
                                    )
                                    if tid:
                                        transaction_id = tid.group(1)

                                    utr_match = re.search(
                                        r"UTR No\.?\s*([A-Za-z0-9]+)",
                                        detail, re.IGNORECASE
                                    )
                                    if utr_match:
                                        utr = utr_match.group(1)

                                records.append({
                                    "Date": pd.Timestamp(date_value),
                                    "Time": txn_time,
                                    "Description": description,
                                    "Type": txn_type,
                                    "Amount": amount,
                                    "Transaction ID": transaction_id,
                                    "UTR": utr,
                                    "Source File": uploaded_file.name,
                                    "Source Page": page_no
                                })

                    if records:
                        all_transactions.append(pd.DataFrame(records))
                        st.success(
                            f"Extracted {len(records)} transactions from "
                            f"{uploaded_file.name}"
                        )
                    else:
                        st.warning(
                            f"No transactions were recognized in "
                            f"{uploaded_file.name}. The parser expects a "
                            "text-based PhonePe PDF in a supported format."
                        )

            except Exception as exc:
                st.error(f"Could not process {uploaded_file.name}: {exc}")

        if all_transactions:
            combined_df = pd.concat(
                all_transactions, ignore_index=True, sort=False
            )

            if "Amount" not in combined_df.columns:
                st.error("No Amount column found. Check the CSV column names.")
                st.stop()

            combined_df["Amount"] = pd.to_numeric(
                combined_df["Amount"], errors="coerce"
            )
            combined_df = combined_df.dropna(subset=["Amount"]).copy()

            if "Type" not in combined_df.columns:
                combined_df["Type"] = "Unknown"

            combined_df["Type"] = (
                combined_df["Type"].astype(str).str.upper().str.strip()
            )

            if "Date" in combined_df.columns:
                combined_df["Date"] = pd.to_datetime(
                    combined_df["Date"], errors="coerce"
                )
                
            # Save uploaded transactions without overwriting existing records.
            save_db_path = Path(__file__).resolve().parent / "cardshield_live.db"
            saved_count = 0

            with sqlite3.connect(str(save_db_path), timeout=10) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS uploaded_transactions (
                        transaction_id TEXT PRIMARY KEY,
                        transaction_time TEXT,
                        amount REAL,
                        transaction_type TEXT,
                        description TEXT,
                        source_file TEXT,
                        scoring_status TEXT NOT NULL DEFAULT 'Not scored',
                        details_json TEXT,
                        uploaded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                for _, row in combined_df.iterrows():
                    source_file = str(row.get("Source File", "Uploaded CSV"))
                    txn_date = row.get("Date", "")
                    txn_time = str(row.get("Time", "") or "")
                    description = str(row.get("Description", "") or "")
                    txn_type = str(row.get("Type", "Unknown") or "Unknown")
                    amount = float(row["Amount"])

                    if pd.isna(txn_date):
                        date_text = ""
                    else:
                        date_text = str(txn_date)

                    raw_id = "|".join([
                        source_file,
                        date_text,
                        txn_time,
                        description,
                        txn_type,
                        str(amount),
                        str(row.get("Transaction ID", "")),
                        str(row.get("UTR", "")),
                    ])
                    transaction_id = hashlib.sha256(
                        raw_id.encode("utf-8")
                    ).hexdigest()

                    details_json = row.to_json(
                        date_format="iso",
                        default_handler=str
                    )

                    cursor = conn.execute("""
                        INSERT OR IGNORE INTO uploaded_transactions (
                            transaction_id,
                            transaction_time,
                            amount,
                            transaction_type,
                            description,
                            source_file,
                            scoring_status,
                            details_json
                        )
                        VALUES (?, ?, ?, ?, ?, ?, 'Not scored', ?)
                    """, (
                        transaction_id,
                        f"{date_text} {txn_time}".strip(),
                        amount,
                        txn_type,
                        description,
                        source_file,
                        details_json,
                    ))

                    saved_count += cursor.rowcount

                conn.commit()

            st.success(
                f"Saved {saved_count} new transaction(s) to the database. "
                "These records are not yet scored for fraud."
            )


            st.divider()
            st.subheader("📊 Transaction Intelligence Dashboard")

            # Filters
            f1, f2 = st.columns(2)
            with f1:
                types = sorted(combined_df["Type"].dropna().unique().tolist())
                selected_types = st.multiselect(
                    "Transaction type",
                    types,
                    default=types
                )
            with f2:
                max_amount = float(combined_df["Amount"].max())
                min_amount = float(combined_df["Amount"].min())
                amount_range = st.slider(
                    "Amount range",
                    min_value=min_amount,
                    max_value=max_amount if max_amount > min_amount else min_amount + 1.0,
                    value=(min_amount, max_amount)
                )

            filtered = combined_df[
                combined_df["Type"].isin(selected_types)
                & combined_df["Amount"].between(
                    amount_range[0], amount_range[1]
                )
            ].copy()

            debit_mask = filtered["Type"].str.contains(
                "DEBIT|DR|WITHDRAW", regex=True, na=False
            )
            credit_mask = filtered["Type"].str.contains(
                "CREDIT|CR|DEPOSIT", regex=True, na=False
            )

            total_debit = filtered.loc[debit_mask, "Amount"].sum()
            total_credit = filtered.loc[credit_mask, "Amount"].sum()
            avg_amount = filtered["Amount"].mean() if not filtered.empty else 0

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Transactions", f"{len(filtered):,}")
            k2.metric("Total Debit", f"₹{total_debit:,.2f}")
            k3.metric("Total Credit", f"₹{total_credit:,.2f}")
            k4.metric("Average Amount", f"₹{avg_amount:,.2f}")

            if filtered.empty:
                st.warning("No transactions match the selected filters.")
            else:
                left, right = st.columns(2)

                with left:
                    st.markdown("#### Debit vs Credit")
                    flow_df = pd.DataFrame({
                        "Category": ["Debit", "Credit"],
                        "Amount": [total_debit, total_credit]
                    })
                    fig = px.pie(
                        flow_df,
                        names="Category",
                        values="Amount",
                        hole=0.58,
                        title="Money Flow Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with right:
                    st.markdown("#### Transaction Amounts")
                    fig = px.histogram(
                        filtered,
                        x="Amount",
                        color="Type",
                        nbins=25,
                        title="Amount Distribution",
                        marginal="box"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                if "Date" in filtered.columns and filtered["Date"].notna().any():
                    daily = (
                        filtered.dropna(subset=["Date"])
                        .assign(Day=lambda x: x["Date"].dt.date)
                        .groupby(["Day", "Type"], as_index=False)
                        .agg(
                            Transactions=("Amount", "size"),
                            Total_Amount=("Amount", "sum")
                        )
                    )

                    st.markdown("#### Daily Transaction Trend")
                    fig = px.line(
                        daily,
                        x="Day",
                        y="Total_Amount",
                        color="Type",
                        markers=True,
                        title="Daily Transaction Value"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    fig = px.bar(
                        daily,
                        x="Day",
                        y="Transactions",
                        color="Type",
                        barmode="group",
                        title="Daily Transaction Count"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                if "Description" in filtered.columns:
                    top = (
                        filtered.assign(
                            Description=filtered["Description"]
                            .fillna("Unknown").astype(str)
                        )
                        .groupby("Description", as_index=False)
                        .agg(
                            Total_Amount=("Amount", "sum"),
                            Transactions=("Amount", "size")
                        )
                        .sort_values("Total_Amount", ascending=False)
                        .head(10)
                    )

                    st.markdown("#### Top 10 Transaction Descriptions")
                    fig = px.bar(
                        top.sort_values("Total_Amount"),
                        x="Total_Amount",
                        y="Description",
                        orientation="h",
                        hover_data=["Transactions"],
                        title="Highest Transaction Values by Description"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### Combined Transaction Table")
                st.dataframe(
                    filtered,
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "⬇️ Download Filtered Transactions CSV",
                    data=filtered.to_csv(index=False).encode("utf-8"),
                    file_name="cardshield_filtered_transactions.csv",
                    mime="text/csv"
                )

        else:
            st.warning("No transactions were extracted from the selected files.")



# ---------------- MODEL EVALUATION ----------------
elif page == "Model Evaluation":

    st.markdown("""
        <style>
        .model-heading {
            font-size: 30px;
            font-weight: 750;
            color: #38bdf8;
            margin-bottom: 4px;
        }
        .model-subtitle {
            color: #94a3b8;
            margin-bottom: 22px;
        }
        </style>
        <div class="model-heading">📊 Model Performance Center</div>
        <div class="model-subtitle">
            Compare machine learning models and review fraud detection metrics.
        </div>
    """, unsafe_allow_html=True)

    st.caption(f"Active model: {model_name}")

    # Load model comparison data
    comparison = None

    if COMPARISON_PATH.exists():
        try:
            comparison = pd.read_csv(COMPARISON_PATH)
        except Exception as exc:
            st.error(f"Could not load comparison data: {exc}")

    required_cols = [
        "Model", "Average Precision", "Precision", "Recall", "F1 Score"
    ]

    if comparison is not None and all(
        col in comparison.columns for col in required_cols
    ):
        comparison = comparison.copy()

        for col in required_cols[1:]:
            comparison[col] = pd.to_numeric(
                comparison[col], errors="coerce"
            )

        comparison = comparison.dropna(subset=required_cols[1:])

        active_rows = comparison[
            comparison["Model"].astype(str).str.casefold()
            == str(model_name).casefold()
        ]

        if active_rows.empty:
            active_rows = comparison.tail(1)

        if not comparison.empty and not active_rows.empty:
            active = active_rows.iloc[0]

            # KPI cards
            st.subheader("Key Performance Indicators")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Average Precision",
                f"{active['Average Precision']:.4f}"
            )
            c2.metric(
                "Precision",
                f"{active['Precision']:.4f}"
            )
            c3.metric(
                "Recall",
                f"{active['Recall']:.4f}"
            )
            c4.metric(
                "F1 Score",
                f"{active['F1 Score']:.4f}"
            )

            st.caption(
                "These metrics come from the saved model comparison results. "
                "They describe the evaluation recorded during training."
            )

            st.divider()

            # Model comparison table and chart
            left, right = st.columns([1.25, 1])

            with left:
                st.subheader("Model Comparison")

                styled = (
                    comparison.style
                    .format({
                        "Average Precision": "{:.4f}",
                        "Precision": "{:.4f}",
                        "Recall": "{:.4f}",
                        "F1 Score": "{:.4f}",
                    })
                )

                st.dataframe(
                    styled,
                    use_container_width=True,
                    hide_index=True
                )

            with right:
                st.subheader("Average Precision")

                chart_df = comparison[
                    ["Model", "Average Precision"]
                ].set_index("Model")

                st.bar_chart(
                    chart_df,
                    y="Average Precision",
                    color="#38bdf8",
                    height=300
                )

            st.divider()

            # Compare all metrics
            st.subheader("Performance Across Metrics")

            metric_chart = comparison.set_index("Model")[
                ["Precision", "Recall", "F1 Score"]
            ]

            st.bar_chart(
                metric_chart,
                height=350
            )

            # Active model insight
            st.subheader("Model Insights")

            st.markdown(
                f"""
                **Selected model:** {model_name}

                - **Precision:** {active['Precision']:.4f}
                - **Recall:** {active['Recall']:.4f}
                - **F1 Score:** {active['F1 Score']:.4f}
                - **Average Precision:** {active['Average Precision']:.4f}
                """
            )

            st.info(
                "Fraud detection uses imbalanced data. A high precision score "
                "does not mean every fraud case is caught. Review recall, "
                "false negatives, false positives and the chosen alert threshold."
            )

            # Export comparison
            st.download_button(
                "⬇️ Download Model Comparison CSV",
                data=comparison.to_csv(index=False).encode("utf-8"),
                file_name="cardshield_model_comparison.csv",
                mime="text/csv",
                use_container_width=True
            )

        else:
            st.warning("No valid model comparison rows are available.")

    else:
        st.warning(
            "Model comparison data is missing or does not contain the expected columns. "
            "Check model_comparison.csv or run your training script."
        )

        ap = bundle.get("average_precision")

        if ap is not None:
            st.metric(
                "Saved Test-Set Average Precision",
                f"{float(ap):.4f}"
            )

    st.divider()

    st.subheader("Deployment Checklist")

    st.markdown("""
    - Evaluate on an untouched test dataset.
    - Check precision and recall at different thresholds.
    - Review false positives and false negatives.
    - Monitor data quality and model drift.
    - Require human review before consequential action.
    """)


# ---------------- DATA QUALITY ----------------


# ---------------- DATA QUALITY CENTER ----------------
elif page == "Data Quality":

    st.markdown("""
        <style>
        .dq-title {
            font-size: 30px;
            font-weight: 800;
            color: #38bdf8;
            margin-bottom: 0;
        }
        .dq-subtitle {
            color: #94a3b8;
            margin-bottom: 20px;
        }
        </style>
        <div class="dq-title">🛡️ Data Quality Center</div>
        <div class="dq-subtitle">
            Dataset health, validation and interactive profiling
        </div>
    """, unsafe_allow_html=True)

    if df is None:
        st.error("Historical dataset could not be loaded.")
        if dataset_error:
            st.caption(dataset_error)
        st.stop()

    # Keep the original dataframe unchanged.
    quality_df = df.copy()

    rows, cols = quality_df.shape
    missing_cells = int(quality_df.isna().sum().sum())
    duplicate_rows = int(quality_df.duplicated().sum())

    total_cells = max(rows * cols, 1)
    quality_score = max(
        0.0,
        100 * (1 - (missing_cells + duplicate_rows * cols) / total_cells)
    )

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("📁 Total Rows", f"{rows:,}")
    c2.metric("📊 Total Columns", f"{cols:,}")
    c3.metric("⚠️ Missing Cells", f"{missing_cells:,}")
    c4.metric("🔁 Duplicate Rows", f"{duplicate_rows:,}")

    st.progress(
        min(quality_score / 100, 1.0),
        text=f"Data Quality Indicator: {quality_score:.1f}%"
    )

    if missing_cells == 0 and duplicate_rows == 0:
        st.success("Dataset check passed: no missing cells or exact duplicate rows.")
    else:
        st.warning(
            "Potential data-quality issues found. Review the details below "
            "before cleaning or training."
        )

    st.divider()

    # Interactive sections
    profile_tab, missing_tab, duplicate_tab, preview_tab = st.tabs([
        "📊 Dataset Profile",
        "⚠️ Missing Values",
        "🔁 Duplicates",
        "🔎 Data Preview"
    ])

    with profile_tab:
        left, right = st.columns(2)

        with left:
            st.subheader("Column Data Types")

            type_counts = (
                quality_df.dtypes.astype(str)
                .value_counts()
                .rename_axis("Data Type")
                .to_frame("Columns")
            )

            st.bar_chart(type_counts, height=300)

        with right:
            st.subheader("Class Balance")

            if "Class" in quality_df.columns:
                class_counts = (
                    pd.to_numeric(
                        quality_df["Class"], errors="coerce"
                    )
                    .map({0: "Normal (0)", 1: "Fraud (1)"})
                    .fillna("Other / Invalid")
                    .value_counts()
                    .rename_axis("Transaction Label")
                    .to_frame("Rows")
                )

                st.bar_chart(class_counts, height=300)

                st.caption(
                    "Class balance uses existing dataset labels, "
                    "not new model predictions."
                )
            else:
                st.info("No Class column found in this dataset.")

        st.subheader("Column Statistics")

        profile = pd.DataFrame({
            "Data Type": quality_df.dtypes.astype(str),
            "Non-null Values": quality_df.notna().sum(),
            "Missing Values": quality_df.isna().sum(),
            "Unique Values": quality_df.nunique(dropna=True),
        })

        type_filter = st.selectbox(
            "Filter columns by data type",
            ["All"] + sorted(profile["Data Type"].unique().tolist()),
            key="dq_type_filter"
        )

        if type_filter != "All":
            profile = profile[profile["Data Type"] == type_filter]

        st.dataframe(
            profile,
            use_container_width=True
        )

    with missing_tab:
        st.subheader("Missing Values by Column")

        missing_summary = (
            quality_df.isna().sum()
            .rename("Missing Values")
            .to_frame()
        )

        missing_summary = missing_summary[
            missing_summary["Missing Values"] > 0
        ].sort_values("Missing Values", ascending=False)

        if missing_summary.empty:
            st.success("All columns are complete. No missing values found.")
        else:
            st.bar_chart(missing_summary, height=350)
            st.dataframe(missing_summary, use_container_width=True)

        st.caption(
            "Blank values and non-numeric values in numeric columns "
            "are different checks; validate feature values before prediction."
        )

    with duplicate_tab:
        st.subheader("Duplicate Row Analysis")

        st.metric("Exact duplicate rows", f"{duplicate_rows:,}")

        if duplicate_rows:
            st.warning(
                "Review duplicates before removing them. Repeated transactions "
                "may be legitimate."
            )

            if st.checkbox(
                "Preview duplicate rows",
                key="dq_show_duplicates"
            ):
                duplicates = quality_df[
                    quality_df.duplicated(keep=False)
                ]
                st.dataframe(
                    duplicates.head(500),
                    use_container_width=True,
                    hide_index=True
                )

            if st.button(
                "Prepare cleaned CSV without exact duplicates",
                key="dq_prepare_clean"
            ):
                cleaned = quality_df.drop_duplicates().copy()
                st.session_state["dq_clean_csv"] = (
                    cleaned.to_csv(index=False).encode("utf-8")
                )
                st.session_state["dq_clean_rows"] = len(cleaned)

            if "dq_clean_csv" in st.session_state:
                st.caption(
                    f"Prepared rows: {st.session_state['dq_clean_rows']:,}"
                )
                st.download_button(
                    "Download cleaned CSV",
                    data=st.session_state["dq_clean_csv"],
                    file_name="cardshield_cleaned_dataset.csv",
                    mime="text/csv",
                    key="dq_download_clean"
                )
        else:
            st.success("No exact duplicate rows found.")

    with preview_tab:
        st.subheader("Dataset Explorer")

        col1, col2 = st.columns([1, 2])

        with col1:
            preview_rows = st.selectbox(
                "Rows to display",
                [10, 25, 50, 100],
                index=0,
                key="dq_preview_rows"
            )

        with col2:
            search_column = st.selectbox(
                "Search in column",
                ["All columns"] + quality_df.columns.tolist(),
                key="dq_search_column"
            )

        search_text = st.text_input(
            "🔍 Search dataset values",
            placeholder="Enter a value or keyword...",
            key="dq_search_text"
        )

        view_df = quality_df

        if search_text.strip():
            if search_column == "All columns":
                mask = view_df.astype(str).apply(
                    lambda col: col.str.contains(
                        search_text.strip(),
                        case=False,
                        regex=False,
                        na=False
                    )
                ).any(axis=1)
            else:
                mask = view_df[search_column].astype(str).str.contains(
                    search_text.strip(),
                    case=False,
                    regex=False,
                    na=False
                )

            view_df = view_df[mask]

        st.caption(
            f"Showing {min(preview_rows, len(view_df)):,} of "
            f"{len(view_df):,} matching rows"
        )

        st.dataframe(
            view_df.head(preview_rows),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download current dataset",
            data=view_df.to_csv(index=False).encode("utf-8"),
            file_name="cardshield_dataset_export.csv",
            mime="text/csv",
            key="dq_download_data"
        )

    st.divider()
    st.caption(
        "CardShield Data Quality Center • Local dataset only • "
        "Cleaning downloads do not overwrite the original CSV."
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
