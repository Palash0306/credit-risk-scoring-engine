
# This script reads the raw CSVs from local disk and loads them
# into DuckDB as tables. This replaces the Spark Delta Lake step
# for the dbt transformation layer — dbt will read directly from
# DuckDB tables instead of Delta files on Google Drive.
# Run this once before running dbt.

import duckdb
import os

# ── Config ────────────────────────────────────────────────────
# Path to the DuckDB database file — created automatically if missing
DB_PATH  = "data/credit_risk.duckdb"

# Paths to raw CSV files downloaded from Kaggle
LC_PATH  = os.path.expanduser("~/credit-risk-data/lending-club/accepted_2007_to_2018Q4.csv")
HC_PATH  = os.path.expanduser("~/credit-risk-data/home-credit/application_train.csv")

# ── Connect ───────────────────────────────────────────────────
# DuckDB creates the .duckdb file on first connect if it doesn't exist
con = duckdb.connect(DB_PATH)
print(f"Connected to DuckDB at {DB_PATH} ✅")

# ── Load Lending Club ─────────────────────────────────────────
# read_csv_auto infers column types automatically
# We load the full file here — DuckDB handles it efficiently
# even on 8GB RAM because it streams rather than loading all to memory
print("Loading Lending Club CSV into DuckDB...")
con.execute(f"""
    CREATE OR REPLACE TABLE raw_loans AS
    SELECT * FROM read_csv_auto('{LC_PATH}', 
        header=true,
        sample_size=10000,
        ignore_errors=true
    )
""")

lc_count = con.execute("SELECT COUNT(*) FROM raw_loans").fetchone()[0]
print(f"raw_loans loaded — {lc_count:,} rows ✅")

# ── Load Home Credit ──────────────────────────────────────────
# This is the out-of-sample stress test dataset
# Only application_train.csv is needed for now
print("Loading Home Credit CSV into DuckDB...")
con.execute(f"""
    CREATE OR REPLACE TABLE home_credit_raw AS
    SELECT * FROM read_csv_auto('{HC_PATH}',
        header=true,
        sample_size=10000,
        ignore_errors=true
    )
""")

hc_count = con.execute("SELECT COUNT(*) FROM home_credit_raw").fetchone()[0]
print(f"home_credit_raw loaded — {hc_count:,} rows ✅")

# ── Verify ────────────────────────────────────────────────────
# List all tables in the database to confirm both loaded
tables = con.execute("SHOW TABLES").fetchall()
print("\nTables in DuckDB:")
for t in tables:
    print(f"  → {t[0]}")

con.close()
print("\nDone — DuckDB ready for dbt ✅")