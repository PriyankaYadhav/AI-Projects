"""
Load stage: loads the cleaned CSV into the SQL database.

Usage:
    python load.py ../data/processed/job_postings_clean.csv
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd

DB_PATH = Path("../data/pipeline.db")
SCHEMA_PATH = Path("../sql/schema.sql")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    with open(SCHEMA_PATH, "r") as f:
        conn.executescript(f.read())
    return conn


def main():
    if len(sys.argv) != 2:
        print("Usage: python load.py <path_to_clean_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path)

    conn = get_connection()
    df.to_sql("job_postings", conn, if_exists="replace", index=False)
    conn.commit()
    print(f"Loaded {len(df)} rows into {DB_PATH}")
    conn.close()


if __name__ == "__main__":
    main()