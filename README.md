# Credit Risk Scoring Engine

A production-grade credit risk model built on free infrastructure — ingesting 2.2 million real loan records from an AWS S3 data lake, transforming them through a dbt pipeline, training ML models with full credit-specific validation, and converting predictions into an interpretable credit scorecard (300–850).

![Python](https://img.shields.io/badge/Python-3.11-blue)
![dbt](https://img.shields.io/badge/dbt-Core-orange)
![AWS](https://img.shields.io/badge/AWS-S3-orange)
![MLflow](https://img.shields.io/badge/MLflow-DagsHub-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)
![LightGBM](https://img.shields.io/badge/LightGBM-4.3-green)
![DuckDB](https://img.shields.io/badge/DuckDB-local-yellow)
![Colab](https://img.shields.io/badge/Compute-Google%20Colab-orange)

---

## What This Platform Does

The Credit Risk Scoring Engine is a full-stack ML system that:

- Stores raw loan data in an AWS S3 data lake and loads it into DuckDB for local transformation
- Transforms 2.2 million loan records through a structured dbt pipeline into ML-ready features
- Trains and compares XGBoost, LightGBM, and Logistic Regression models tracked via MLflow on DagsHub
- Validates models using credit-industry standard metrics — Gini coefficient, KS statistic, Population Stability Index
- Converts model output into an interpretable credit scorecard (300–850 range)
- Stress tests the model against the Home Credit dataset — a completely different borrower population

---

## Why This Project

Credit risk modelling is the core of what banks build for IFRS 9 Expected Credit Loss calculation. This project translates that regulatory context into a working system — using the same data, the same model types, and the same validation framework that a bank's model risk team would apply in production.

The architecture mirrors how real financial institutions structure their ML platforms — S3 data lake, SQL transformation pipeline with lineage and tests, experiment tracking, scorecard conversion, and out-of-sample stress testing.

---

## Architecture

```
Lending Club loan data — 2.2M records (Kaggle)
↓
AWS S3 — raw data lake
    s3://credit-risk-engine-prod-2026/raw/lending_club.csv
    s3://credit-risk-engine-prod-2026/raw/home_credit.csv
↓
scripts/load_raw.py
    DuckDB — raw_loans (2,260,668 rows) + home_credit_raw (307,511 rows)
↓
dbt Core (local, Cursor IDE)
    stg_loans          → drop leakage columns, fix types, filter outcomes
    int_loans_cleaned  → null handling, feature engineering
    fct_loans          → 50 features + binary target + train/test split flag
                         1,344,936 rows | train: 826,602 | test: 518,334
↓
scripts/export_to_s3.py
    s3://credit-risk-engine-prod-2026/processed/fct_loans.parquet (89MB)
↓
Google Colab — ML notebooks
    02_eda.ipynb         → distributions, class balance, default rate by grade
    03_features.ipynb    → Weight of Evidence, Information Value scoring
    04_training.ipynb    → XGBoost + LightGBM + Logistic Regression
    05_validation.ipynb  → Gini, KS, PSI, calibration, ROC curve
    06_scorecard.ipynb   → PD → 300–850 credit score conversion
    07_stress_test.ipynb → Home Credit out-of-sample validation + degradation analysis
↓
DagsHub MLflow
    all experiments tracked → metrics, params, artifacts per run
    best model + artifacts  → s3://credit-risk-engine-prod-2026/models/
```

---

## Tech Stack

| Layer | Technology | Free? |
|---|---|---|
| Raw data lake | AWS S3 | Yes — 5 GB forever |
| Local database | DuckDB | Yes — forever |
| Transformation framework | dbt Core + dbt-duckdb | Yes — forever |
| ML compute | Google Colab | Yes — forever |
| Experiment tracking | DagsHub MLflow | Yes — forever |
| ML artifact store | AWS S3 | Yes — within 5 GB |
| Gradient boosting | XGBoost 2.0, LightGBM 4.3 | Yes |
| Baseline model | scikit-learn LogisticRegression | Yes |
| Preprocessing | scikit-learn Pipeline + ColumnTransformer | Yes |
| Credit scorecard | log-odds scaling (300–850) | Yes |
| IDE | Cursor | Yes |
| Version control | GitHub | Yes |

---

## Data Sources

| Dataset | Records | Use | Source |
|---|---|---|---|
| Lending Club loan data | 2,260,668 | Model training and validation | Kaggle |
| Home Credit Default Risk | 307,511 | Out-of-sample stress test | Kaggle |

### Target variable

`loan_status` is converted to a binary target — Charged Off maps to 1 (default), Fully Paid maps to 0 (no default). Records with unknown outcomes (Current, Late, In Grace Period) are excluded — their outcomes are not yet known.

### Feature groups (50 columns after cleaning)

| Group | Examples |
|---|---|
| Loan characteristics | loan_amnt, term, int_rate, installment, grade_encoded, sub_grade_encoded, purpose |
| Borrower financials | log_annual_inc, dti, emp_length_years, home_ownership, verification_status |
| Credit history | fico_score, credit_age_years, delinq_2yrs, inq_last_6mths, revol_util, pub_rec |
| Credit bureau detail | tot_cur_bal, bc_util, mort_acc, pct_tl_nvr_dlq, pub_rec_bankruptcies |
| Null-filled history | mths_since_last_delinq, mths_since_last_major_derog (nulls → 999) |

### Columns excluded

- **Data leakage (~35 columns)** — post-origination fields: total_pymnt, recoveries, last_pymnt_amnt, hardship fields, settlement fields. These only exist after a loan has defaulted and give the model information it could never have at scoring time.
- **Administrative / identifiers (~25 columns)** — id, member_id, url, desc, policy_code, pymnt_plan.
- **Sparse joint application fields (~15 columns)** — annual_inc_joint, dti_joint, sec_app_* fields populated for less than 2% of loans.

---

## Project Phases

| Phase | What | Status |
|---|---|---|
| Phase 1 | Upload raw data to S3, load into DuckDB | ✅ Complete |
| Phase 2 | dbt transformation pipeline — raw → staging → features → S3 | ✅ Complete |
| Phase 3 | Train XGBoost + LightGBM + Logistic Regression, MLflow tracking | ✅ Complete |
| Phase 4 | Credit scorecard, Gini/KS/PSI validation, Home Credit stress test | ✅ Complete |

---

## Getting Started

### Prerequisites

- AWS account — free at aws.amazon.com
- Kaggle account — free at kaggle.com
- Google account — for Colab notebooks
- DagsHub account — free at dagshub.com
- Python 3.11
- GitHub account

### Step 1 — Clone the repo

```bash
git clone https://github.com/Palash0306/credit-risk-scoring-engine.git
cd credit-risk-scoring-engine
```

### Step 2 — Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install dbt-core dbt-duckdb duckdb boto3 pandas pyarrow python-dotenv
```

### Step 3 — Configure credentials

Create a `.env` file in the repo root — never commit this file:

```
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=ap-south-1
S3_BUCKET=credit-risk-engine-prod-2026
DAGSHUB_USERNAME=Palash0306
DAGSHUB_TOKEN=your_dagshub_token
```

### Step 4 — Download data from Kaggle

```bash
mkdir -p ~/.kaggle
echo '{"username":"your_username","key":"your_token"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

mkdir -p ~/credit-risk-data/lending-club
mkdir -p ~/credit-risk-data/home-credit

# Lending Club — 1.7GB
kaggle datasets download -d wordsforthewise/lending-club \
    --unzip -p ~/credit-risk-data/lending-club

# Home Credit — accept rules at kaggle.com/competitions/home-credit-default-risk first
kaggle competitions download -c home-credit-default-risk \
    -p ~/credit-risk-data/home-credit
unzip ~/credit-risk-data/home-credit/home-credit-default-risk.zip \
    -d ~/credit-risk-data/home-credit/
```

### Step 5 — Upload to S3

```bash
aws configure

aws s3 mb s3://credit-risk-engine-prod-2026 --region ap-south-1

aws s3 cp ~/credit-risk-data/lending-club/accepted_2007_to_2018Q4.csv \
    s3://credit-risk-engine-prod-2026/raw/lending_club.csv

aws s3 cp ~/credit-risk-data/home-credit/application_train.csv \
    s3://credit-risk-engine-prod-2026/raw/home_credit.csv

aws s3 ls s3://credit-risk-engine-prod-2026/raw/
```

### Step 6 — Load into DuckDB and run dbt

```bash
# Load raw CSVs into DuckDB
python scripts/load_raw.py

# Run dbt pipeline
cd dbt_project
dbt debug
dbt run
dbt test
cd ..

# Export fct_loans to S3
python scripts/export_to_s3.py
```

### Step 7 — Run ML notebooks in Google Colab

Open each notebook at colab.research.google.com. Add Colab Secrets via the key icon on the left sidebar:

| Secret | Value |
|---|---|
| `AWS_ACCESS_KEY_ID` | your AWS access key |
| `AWS_SECRET_ACCESS_KEY` | your AWS secret key |
| `S3_BUCKET` | `credit-risk-engine-prod-2026` |
| `DAGSHUB_USERNAME` | `Palash0306` |
| `DAGSHUB_TOKEN` | your DagsHub token |

Run notebooks in order:

```
notebooks/02_eda.ipynb             → exploratory data analysis
notebooks/03_features.ipynb        → WoE and IV feature selection
notebooks/04_training.ipynb        → model training and MLflow tracking
notebooks/05_validation.ipynb      → Gini, KS, PSI, calibration
notebooks/06_scorecard.ipynb       → credit scorecard generation
notebooks/07_stress_test.ipynb     → Home Credit stress test + degradation analysis
```

Save each notebook to GitHub via: `File → Save a copy in GitHub`

---

## dbt Pipeline

```
raw_loans (DuckDB — 2,260,668 rows, 151 cols)
        ↓
stg_loans
    drop 75 leakage, administrative, and sparse columns
    cast int_rate and revol_util from string to float
    cast term from string to integer
    filter to Fully Paid and Charged Off rows only
        ↓
int_loans_cleaned
    fill mths_since_last_delinq nulls with 999
    fill mths_since_last_major_derog nulls with 999
    convert emp_length string to numeric years
    convert earliest_cr_line to credit age in years
    log-transform annual_inc
    average fico_range_low and fico_range_high to fico_score
        ↓
fct_loans (1,344,936 rows)
    binary target: loan_status → default_flag (0/1)
    ordinal encode grade (A=1 to G=7)
    ordinal encode sub_grade (A1=1 to G5=35)
    chronological train/test split flag (is_train)
    train: 826,602 rows (before 2016)
    test:  518,334 rows (2016 onwards)
```

### dbt tests on every build

| Test | Applied to |
|---|---|
| not_null | All 50 feature columns in fct_loans |
| unique | loan_id in stg_loans |
| accepted_values | default_flag in (0, 1) |
| accepted_values | grade in (A, B, C, D, E, F, G) |
| accepted_values | term in (36, 60) |
| not_null | default_flag — no nulls allowed in target |

---

## ML Pipeline

### Models trained

| Model | Why |
|---|---|
| Logistic Regression | Required baseline — interpretable, regulators accept for IFRS 9 |
| XGBoost | Industry standard for credit scoring — high performance on tabular data |
| LightGBM | Faster than XGBoost on 2.2M rows — comparable accuracy |

### Why XGBoost and LightGBM

Gradient boosting dominates credit scoring because loan data is tabular, features are mixed numeric and categorical, and the relationship between features and default is non-linear. Both models handle missing values natively, scale to millions of rows, and produce well-calibrated probability estimates. Logistic Regression is kept as the interpretable baseline — regulators under IFRS 9 and Basel frameworks require understanding what drives the model before approving it for production.

### Validation metrics

| Metric | What it measures | Target |
|---|---|---|
| Gini coefficient | Rank-ordering ability — primary credit model metric | > 0.40 |
| KS statistic | Maximum separation between default and non-default score distributions | > 0.30 |
| AUC-ROC | Overall discrimination | > 0.70 |
| Brier score | Calibration — does PD 0.15 mean 15% actually default? | < 0.15 |
| PSI | Population Stability Index — score distribution shift between train and test | < 0.10 |

### Train / test split

Time-based — not random. Loans issued before 2016 used for training (826,602 rows). Loans issued 2016 onwards used for testing (518,334 rows). Prevents data leakage from future information and mirrors production model validation practice.

### Credit scorecard conversion

PD probabilities are converted to a 300–850 credit score using log-odds scaling:

```
odds       = PD / (1 - PD)
log_odds   = ln(odds)
credit_score = 600 - (20 / ln(2)) × log_odds
```

Parameters: PDO (points to double the odds) = 20, offset = 600 at 1:1 odds. Score is clipped to 300–850 (standard FICO range). Higher score = lower risk.

### Risk tiers

| Risk Tier | Credit Score Range |
|---|---|
| Low | 720 and above |
| Medium | 660 – 719 |
| High | 580 – 659 |
| Very High | Below 580 |

---

## Stress Test

The model is tested against the Home Credit dataset — a different borrower population (different country, loan types, and demographics) — to measure how well it generalises beyond the Lending Club population it was trained on.

The stress test notebook automatically:
- Maps Home Credit features to Lending Club equivalents
- Scores 307,511 Home Credit applications
- Computes Gini and KS on the out-of-sample population
- Measures PSI per feature to identify which features shift most between populations
- Runs a KS test on score distributions between the two datasets
- Generates an automated degradation explanation based on the findings
- Logs everything to DagsHub MLflow

---

## Key Technical Decisions

**Why S3 as the raw data lake?** S3 decouples storage from compute. Raw data lives in S3 — permanent, accessible to any tool. Colab reads from S3. If compute platforms change, the data stays.

**Why DuckDB instead of Spark for dbt?** DuckDB runs locally with zero configuration, handles 2.2M rows efficiently on 8GB RAM by streaming rather than loading all data into memory, and integrates natively with dbt. Spark requires a cluster — unnecessary overhead for a transformation layer that runs once.

**Why dbt instead of pandas notebooks for transformation?** dbt enforces structure. Each transformation is a versioned SQL file with tests and lineage documentation. The pipeline is reproducible and auditable — which matters in a regulatory context.

**Why Google Colab for ML training?** Colab provides 12GB cloud RAM, pre-installed ML libraries, and direct GitHub integration (File → Save a copy in GitHub). It replaces a Databricks cluster at zero cost with no time limit.

**Why DagsHub for MLflow?** DagsHub provides a hosted MLflow tracking server connected to GitHub — free forever. Experiment runs, metrics, and model artifacts are tracked remotely without running an MLflow server locally.

**Why time-based train/test split?** Random splitting leaks future information into training. A loan from 2018 should never appear in training if the test set contains loans from 2016. Time-based splitting mirrors production — train on historical data, score future applications.

**Why Logistic Regression alongside XGBoost?** Regulators under IFRS 9 and Basel frameworks prefer interpretable models. A bank would not deploy a black-box model without understanding its drivers. Logistic Regression is the interpretable baseline. XGBoost is the performance model. Comparing both is exactly what a bank's model validation team would do.

---

## What I Built and Learned

**Data engineering:** dbt transformation pipeline with lineage, tests, and auto-generated documentation. DuckDB for efficient local processing of 2.2M rows. Weight of Evidence and Information Value for feature selection. Chronological train/test splitting to prevent temporal leakage.

**Credit risk domain:** IFRS 9 aligned model design. Data leakage identification across 150 columns. Credit scorecard conversion using log-odds scaling. Industry-standard validation metrics (Gini, KS, PSI). Out-of-sample stress testing with automated degradation analysis.

**ML engineering:** MLflow experiment tracking with DagsHub remote server. Model comparison across three algorithms. Calibration analysis ensuring PD scores are statistically meaningful. Class imbalance handling via scale_pos_weight and class_weight parameters.

**Cloud architecture:** AWS S3 as the central data layer connecting local compute (dbt), cloud compute (Colab), and artifact storage. DagsHub as the experiment tracking layer. Everything connected through environment variables — no credentials committed to GitHub.

---

## Tracked Loan Segments

| Segment | Approx loans | Approx default rate |
|---|---|---|
| Grade A | 420k | 5% |
| Grade B | 580k | 10% |
| Grade C | 490k | 16% |
| Grade D | 310k | 22% |
| Grade E | 180k | 28% |
| Grade F | 80k | 33% |
| Grade G | 40k | 37% |

---

## Author

Palash Aggarwal — [GitHub](https://github.com/Palash0306) · [DagsHub](https://dagshub.com/Palash0306)
```