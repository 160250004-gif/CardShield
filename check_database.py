
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

for db_name in ["cardshield_live.db", "fraud_monitoring.db"]:
    db_path = BASE_DIR / db_name

    print(f"\n{'=' * 45}")
    print("Database:", db_name)

    if not db_path.exists():
        print("Database file does not exist.")
        continue

    with sqlite3.connect(db_path) as conn:
        tables = conn.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
        """).fetchall()

        print("Tables:", [t[0] for t in tables])

        for (table_name,) in tables:
            columns = conn.execute(
                f'PRAGMA table_info("{table_name}")'
            ).fetchall()

            print(f"\nTable: {table_name}")
            print("Columns:", [column[1] for column in columns])
