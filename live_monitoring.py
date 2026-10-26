
import json
import re
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "fraud_monitoring.db"


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



def read_uploaded_statement(uploaded_file):
    """Read CSV files and PhonePe statement PDFs."""
    import re

    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".csv":
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)

    elif suffix == ".pdf":
        try:
            import pdfplumber
        except ImportError as exc:
            raise ValueError(
                "PDF support missing. Run: "
                "python -m pip install pdfplumber"
            ) from exc

        uploaded_file.seek(0)
        with pdfplumber.open(uploaded_file) as pdf:
            text = "\n".join(
                page.extract_text() or "" for page in pdf.pages
            )

        starts = list(re.finditer(
            r"(?m)^\s*((?:Oct|Sept|Sep)\s+\d{1,2},\s+\d{4})\s+",
            text,
            re.IGNORECASE,
        ))

        records = []

        for i, match in enumerate(starts):
            end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
            block = text[match.start():end]

            type_match = re.search(r"\b(DEBIT|CREDIT)\b", block, re.I)
            amount_match = re.search(
                r"\b(?:DEBIT|CREDIT)\b\s*₹\s*([\d,]+(?:\.\d{1,2})?)",
                block,
                re.I,
            )

            if not type_match or not amount_match:
                continue

            first_line = block.splitlines()[0]
            description = re.sub(
                r"^\s*(?:Oct|Sept|Sep)\s+\d{1,2},\s+\d{4}\s*",
                "",
                first_line,
                flags=re.I,
            )
            description = re.split(
                r"\b(?:DEBIT|CREDIT)\b",
                description,
                maxsplit=1,
                flags=re.I,
            )[0].strip()

            if not description:
                description = "Transaction"

            txn_match = re.search(
                r"Transaction ID\s+([A-Za-z0-9]+)", block, re.I
            )
            utr_match = re.search(r"UTR No\.\s*(\d+)", block, re.I)
            time_match = re.search(
                r"(\d{1,2}\s+\d{2}\s+(?:am|pm))", block, re.I
            )

            records.append({
                "Date": match.group(1),
                "Time": time_match.group(1) if time_match else "",
                "Transaction Details": description,
                "Type": type_match.group(1).upper(),
                "Amount": float(amount_match.group(1).replace(",", "")),
                "Transaction ID": txn_match.group(1) if txn_match else "",
                "UTR No": utr_match.group(1) if utr_match else "",
            })

        df = pd.DataFrame(records)

        if df.empty:
            raise ValueError(
                "No PhonePe transactions could be extracted from this PDF."
            )

    else:
        raise ValueError("Please upload a CSV or PDF file.")

    if df.empty:
        raise ValueError("The uploaded file contains no transaction rows.")

    df.columns = [
        str(col).strip().replace("\n", " ") for col in df.columns
    ]

    return df.dropna(how="all").reset_index(drop=True)



def find_column(df, keywords):

    """Find a column using its name."""
    for col in df.columns:
        name = re.sub(r"[^a-z0-9]", "", str(col).lower())
        if any(word in name for word in keywords):
            return col
    return None


def amount_to_number(value):
    """Convert common currency-formatted amounts to numbers."""
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    negative = text.startswith("(") and text.endswith(")")
    text = text.replace(",", "")
    text = re.sub(r"(?i)INR|Rs\.?|₹|\$", "", text)
    text = re.sub(r"[^0-9.\-]", "", text)

    try:
        amount = float(text)
        if negative:
            amount = -abs(amount)
        return amount if pd.notna(amount) else None
    except (TypeError, ValueError):
        return None



def screen_transactions(df):
    """
    Transparent rule-based screening.
    This is not a trained fraud classifier and cannot prove fraud.
    """
    amount_col = find_column(
        df, ["amount", "transactionvalue", "txnamount"]
    )
    description_col = find_column(
        df, ["description", "remarks", "narration", "details", "merchant"]
    )
    status_col = find_column(df, ["status", "result"])
    reference_col = find_column(
        df, ["transactionid", "referenceid", "utr", "transactionreference"]
    )
    date_col = find_column(
        df, ["date", "time", "timestamp", "transactiondate"]
    )

    if amount_col is None:
        raise ValueError(
            "I could not identify an Amount column. "
            "Please upload a CSV with a clearly named Amount column."
        )

    result = df.copy()
    result["Screening Amount"] = result[amount_col].apply(amount_to_number)

    if result["Screening Amount"].notna().sum() == 0:
        raise ValueError(
            "The Amount column was found, but no usable numeric amounts "
            "could be read."
        )

    descriptions = (
        result[description_col].fillna("").astype(str)
        if description_col else pd.Series("", index=result.index)
    )

    statuses = (
        result[status_col].fillna("").astype(str).str.lower()
        if status_col else pd.Series("", index=result.index)
    )

    scores = []
    reasons = []

    # Duplicate reference numbers are a review signal, not proof of fraud.
    duplicate_refs = pd.Series(False, index=result.index)
    if reference_col:
        refs = result[reference_col].fillna("").astype(str).str.strip()
        valid_refs = refs.ne("")
        duplicate_refs = refs.duplicated(keep=False) & valid_refs

    for i in result.index:
        score = 0
        flags = []
        amount = result.at[i, "Screening Amount"]
        description = descriptions.at[i].lower()
        status = statuses.at[i]

        if pd.isna(amount):
            flags.append("Amount could not be parsed")
        else:
            if abs(amount) >= 50000:
                score += 35
                flags.append("High-value transaction")
            elif abs(amount) >= 20000:
                score += 15
                flags.append("Large transaction amount")

            if amount < 0:
                flags.append("Negative amount; check statement format")

        if any(word in description for word in
               ["failed", "reversed", "dispute", "chargeback"]):
            score += 25
            flags.append("Description mentions failure or dispute")

        if any(word in status for word in ["failed", "reversed", "declined"]):
            score += 25
            flags.append("Status indicates failure or reversal")

        if duplicate_refs.at[i]:
            score += 25
            flags.append("Duplicate transaction reference")

        score = min(score, 100)
        scores.append(score)
        reasons.append("; ".join(flags) if flags else "No configured rule triggered")

    result["Risk Score"] = scores
    result["Screening Result"] = [
        "Review Required" if score >= 40 else "No Rule Triggered"
        for score in scores
    ]
    result["Review Reason"] = reasons

    return result, {
        "amount_column": amount_col,
        "description_column": description_col,
        "status_column": status_col,
        "reference_column": reference_col,
        "date_column": date_col,
    }


def save_results(df, upload_id, file_name):
    now = datetime.now().astimezone().isoformat(timespec="seconds")

    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        for i, row in df.iterrows():
            amount = row.get("Screening Amount")
            amount = None if pd.isna(amount) else float(amount)
            score = float(row["Risk Score"])
            prediction = str(row["Screening Result"])

            record = {}
            for key, value in row.items():
                if pd.isna(value):
                    record[str(key)] = None
                elif hasattr(value, "item"):
                    record[str(key)] = value.item()
                else:
                    record[str(key)] = str(value)

            conn.execute("""
                INSERT INTO live_transactions (
                    upload_id, file_name, row_number, amount,
                    fraud_score, prediction, flagged_for_review,
                    actual_class, transaction_data, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                upload_id,
                file_name,
                int(i) + 1,
                amount,
                score / 100.0,
                prediction,
                int(prediction == "Review Required"),
                None,
                json.dumps(record, ensure_ascii=False),
                now,
            ))


def render_live_monitoring():
    init_db()

    st.title("📂 Uploaded Statement Screening")
    st.caption(
        "Upload your own bank or payment statement. "
        "Screening only starts after the file is read successfully."
    )

    uploaded_file = st.file_uploader(
        "Upload your statement",
        type=["csv", "pdf"],
        key="live_statement_upload",
        help="CSV is recommended if your PDF has no extractable table.",
    )

    if uploaded_file is None:
        st.info("Upload a CSV or PDF file to continue.")
        st.session_state.pop("screening_result", None)
        st.session_state.pop("screening_file_id", None)
        return

    file_id = f"{uploaded_file.name}:{uploaded_file.size}"

    # Do not reuse results from a different uploaded file.
    if st.session_state.get("screening_file_id") != file_id:
        st.session_state.pop("screening_result", None)
        st.session_state["screening_file_id"] = file_id

    try:
        df = read_uploaded_statement(uploaded_file)
    except Exception as exc:
        st.error(f"File could not be read: {exc}")
        st.warning("Screening is disabled until a readable file is uploaded.")
        st.session_state.pop("screening_result", None)
        return

    st.success(
        f"File uploaded and read successfully: {uploaded_file.name}"
    )
    st.write(f"Transactions/rows found: **{len(df):,}**")
    st.subheader("File Preview")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    st.caption(
        "The rules below provide a basic review signal. "
        "They are not proof of fraud and may flag legitimate transactions."
    )

    if st.button(
        "🔎 Screen Uploaded Transactions",
        type="primary",
        use_container_width=True,
    ):
        try:
            screened_df, detected_columns = screen_transactions(df)
            upload_id = str(uuid.uuid4())
            save_results(screened_df, upload_id, uploaded_file.name)

            st.session_state["screening_result"] = screened_df
            st.session_state["screening_file_id"] = file_id
            st.session_state["screening_upload_id"] = upload_id
            st.session_state["screening_columns"] = detected_columns
            st.session_state["screening_saved"] = True
        except Exception as exc:
            st.error(f"Screening could not be completed: {exc}")
            st.session_state.pop("screening_result", None)
            return

    result = st.session_state.get("screening_result")

    if result is not None:
        st.success("Screening completed for the uploaded transactions.")

        total = len(result)
        flagged = int((result["Screening Result"] == "Review Required").sum())
        other = total - flagged

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows Screened", total)
        c2.metric("Review Required", flagged)
        c3.metric("No Rule Triggered", other)

       
        # ---------- CARDShield INTERACTIVE RISK SIMULATOR ----------
        import streamlit.components.v1 as components

        st.divider()
        st.subheader("🚦 CardShield Live Fraud Simulator")
        st.caption(
            "Green = Low Risk | Yellow = Medium Risk | Red = High Risk. "
            "These are rule-based screening signals, not confirmed fraud."
        )

        if "Risk Category" not in result.columns:
            result["Risk Category"] = result["Risk Score"].apply(
                lambda score: (
                    "High Risk" if score >= 40
                    else "Medium Risk" if score >= 20
                    else "Low Risk"
                )
            )
            st.session_state["screening_result"] = result

        total_transactions = len(result)

        if total_transactions == 0:
            st.info("No transactions available.")
        else:
            if "cardshield_selected_tx" not in st.session_state:
                st.session_state.cardshield_selected_tx = 0

            if "cardshield_last_alert_tx" not in st.session_state:
                st.session_state.cardshield_last_alert_tx = None

            # Keep selected transaction within range
            idx = min(
                st.session_state.cardshield_selected_tx,
                total_transactions - 1,
            )

            selector_key = "cardshield_transaction_picker"

            selected_idx = st.selectbox(
                "Select a transaction",
                options=list(range(total_transactions)),
                index=idx,
                format_func=lambda i: f"Transaction {i + 1}",
                key=selector_key,
            )

            st.session_state.cardshield_selected_tx = selected_idx
            current_tx = result.iloc[selected_idx]

            risk = str(current_tx.get("Risk Category", "Low Risk"))

            risk_colors = {
                "Low Risk": "#22c55e",
                "Medium Risk": "#f59e0b",
                "High Risk": "#ef4444",
            }
            risk_labels = {
                "Low Risk": "NORMAL",
                "Medium Risk": "REVIEW",
                "High Risk": "HIGH-RISK ALERT",
            }

            color = risk_colors.get(risk, "#64748b")
            label = risk_labels.get(risk, "CHECK TRANSACTION")

            # Play alert only once per newly selected high-risk row
            if (
                risk == "High Risk"
                and st.session_state.cardshield_last_alert_tx
                != selected_idx
            ):
                st.session_state.cardshield_last_alert_tx = selected_idx
                components.html(
                    """
                    <audio autoplay>
                      <source
                        src="https://actions.google.com/sounds/v1/alarms/beep_short.ogg"
                        type="audio/ogg">
                    </audio>
                    """,
                    height=1,
                )

            st.markdown(
                f"""
                <div style="
                    display:flex;
                    align-items:center;
                    gap:22px;
                    padding:24px;
                    border:1px solid {color};
                    border-radius:18px;
                    background:rgba(100,116,139,0.08);
                ">
                    <div style="
                        width:82px;
                        height:82px;
                        flex-shrink:0;
                        border-radius:50%;
                        background:{color};
                        box-shadow:0 0 22px {color};
                        border:5px solid rgba(255,255,255,0.75);
                    "></div>
                    <div>
                        <h2 style="margin:0;color:{color};">
                            {label}
                        </h2>
                        <p style="margin:8px 0 0;">
                            {risk}
                        </p>
                        <p style="margin:5px 0 0;">
                            Transaction {selected_idx + 1} of
                            {total_transactions}
                        </p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.progress(
                (selected_idx + 1) / total_transactions,
                text=f"Progress: {selected_idx + 1}/{total_transactions}",
            )

            left, right = st.columns(2)
            with left:
                st.metric("Risk Score", f"{current_tx['Risk Score']:.0f}/100")
            with right:
                st.metric(
                    "Amount",
                    str(current_tx.get("Amount", "N/A")),
                )

            st.markdown("#### Current Transaction Details")

            detail_fields = [
                "Date", "Time", "Transaction Details", "Type",
                "Amount", "Transaction ID", "UTR No", "Review Reason",
            ]
            available_fields = [
                field for field in detail_fields
                if field in result.columns
            ]

            if available_fields:
                st.dataframe(
                    result.iloc[[selected_idx]][available_fields],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.dataframe(
                    result.iloc[[selected_idx]],
                    use_container_width=True,
                    hide_index=True,
                )

            def go_previous():
                new_index = max(
                    0,
                    st.session_state["cardshield_transaction_picker"] - 1,
                )
                st.session_state["cardshield_transaction_picker"] = new_index
                st.session_state.cardshield_selected_tx = new_index

            def go_next():
                new_index = min(
                    total_transactions - 1,
                    st.session_state["cardshield_transaction_picker"] + 1,
                )
                st.session_state["cardshield_transaction_picker"] = new_index
                st.session_state.cardshield_selected_tx = new_index

            def reset_simulator():
                st.session_state["cardshield_transaction_picker"] = 0
                st.session_state.cardshield_selected_tx = 0

            prev_col, next_col, reset_col = st.columns(3)

            with prev_col:
                st.button(
                    "Previous",
                    disabled=(selected_idx == 0),
                    key="cardshield_previous",
                    use_container_width=True,
                    on_click=go_previous,
                )

            with next_col:
                st.button(
                    "Next Transaction",
                    disabled=(selected_idx >= total_transactions - 1),
                    key="cardshield_next",
                    type="primary",
                    use_container_width=True,
                    on_click=go_next,
                )

            with reset_col:
                st.button(
                    "Reset",
                    key="cardshield_reset",
                    use_container_width=True,
                    on_click=reset_simulator,
                )

            st.caption(
                "Alert audio requires internet access and may be blocked "
                "by browser autoplay settings. High risk means review is "
                "recommended; it does not prove fraud."
            )

          
          
           