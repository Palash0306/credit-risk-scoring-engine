
# Reads fct_loans from DuckDB and uploads to S3 as parquet.
# Parquet instead of CSV because:
# 1. Much smaller file size — columnar compression
# 2. Preserves data types — no casting needed in Colab
# 3. Faster to read in pandas and PySpark
# Colab reads this file from S3 for ML training in Phase 3.

import duckdb
import boto3
import os
from dotenv import load_dotenv

# ── Load credentials from .env ────────────────────────────────
# .env sits in repo root — never committed to GitHub
load_dotenv()

AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
BUCKET         = os.getenv('S3_BUCKET')
DB_PATH        = "data/credit_risk.duckdb"

# ── Connect to DuckDB ─────────────────────────────────────────
con = duckdb.connect(DB_PATH)
print(f"Connected to DuckDB ✅")

# ── Check row count before export ─────────────────────────────
row_count = con.execute("SELECT COUNT(*) FROM fct_loans").fetchone()[0]
train_count = con.execute(
    "SELECT COUNT(*) FROM fct_loans WHERE is_train = true"
).fetchone()[0]
test_count = con.execute(
    "SELECT COUNT(*) FROM fct_loans WHERE is_train = false"
).fetchone()[0]

print(f"fct_loans total : {row_count:,} rows")
print(f"Training set    : {train_count:,} rows (before 2016)")
print(f"Test set        : {test_count:,} rows (2016 onwards)")

# ── Export to parquet ─────────────────────────────────────────
# DuckDB writes parquet natively — no pandas conversion needed
print("\nExporting fct_loans to parquet...")
con.execute("""
    COPY (SELECT * FROM fct_loans)
    TO 'data/fct_loans.parquet'
    (FORMAT PARQUET)
""")
print("Exported to data/fct_loans.parquet ✅")
con.close()

# ── Upload to S3 ──────────────────────────────────────────────
# Colab will read from s3://bucket/processed/fct_loans.parquet
print("Uploading to S3...")
s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

s3.upload_file(
    'data/fct_loans.parquet',       # local source file
    BUCKET,                          # S3 bucket name
    'processed/fct_loans.parquet'    # S3 destination key
)

print(f"Uploaded to s3://{BUCKET}/processed/fct_loans.parquet ✅")
print("\nPhase 2 complete — fct_loans ready for ML training in Colab ✅")