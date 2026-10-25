
import re
import pdfplumber
import pandas as pd

PDF_FILE = "PhonePe_Statement_Sept2026_Oct2026.pdf"
OUTPUT_FILE = "phonepe_transactions.csv"

months = (
    "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
)

pattern = re.compile(
    rf"^({months})\s+(\d{{1,2}}),\s+(\d{{4}})\s+"
    r"(.+?)\s+(DEBIT|CREDIT)\s+₹\s*([\d,]+)\s*$"
)

transactions = []

with pdfplumber.open(PDF_FILE) as pdf:
    for page in pdf.pages:
        text = page.extract_text() or ""

        for line in text.splitlines():
            match = pattern.match(line.strip())

            if match:
                month, day, year, details, txn_type, amount = (
                    match.groups()
                )

                transactions.append({
                    "Date": f"{day} {month} {year}",
                    "Transaction Details": details.strip(),
                    "Type": txn_type,
                    "Amount": int(amount.replace(",", ""))
                })

if not transactions:
    print("No transactions matched. PDF format needs review.")
else:
    df = pd.DataFrame(transactions)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("Transactions extracted:", len(df))
    print(df.head(10).to_string(index=False))
    print(f"\nSaved to: {OUTPUT_FILE}")
