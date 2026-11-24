# Investigating Mortgage Lending Outcomes Using 2024 HMDA Data

**Big Data Analysis — Final Project · Group 6 · Faculty of Data Science and Artificial Intelligence, National Economics University (NEU), Hanoi**

This repository contains the code, aggregated results and figures for our study of mortgage lending outcomes in the 2024 Home Mortgage Disclosure Act (HMDA) dataset. Descriptive and diagnostic analyses examine application outcomes, borrower financial conditions, loan characteristics, geographic differences and lender heterogeneity. Equivalent workflows are implemented in **Pandas** and **PySpark** to evaluate analytical correctness and execution performance, and an optional predictive extension compares **Logistic Regression** and **XGBoost** for distinguishing originated from denied applications.

📄 **Technical report:** [`Group_6_final_report.pdf`](Group_6_final_report.pdf)

> Findings are statistical associations, not causal effects or evidence of discrimination. The predictive models classify reported outcomes; they do not predict default risk.

---

## Contents

1. [Research questions](#research-questions)
2. [Data source and acquisition](#data-source-and-acquisition)
3. [Repository structure](#repository-structure)
4. [Installation and execution](#installation-and-execution)
5. [Data engineering and preparation](#data-engineering-and-preparation)
6. [Main findings](#main-findings)
7. [Pandas and PySpark implementations](#pandas-and-pyspark-implementations)
8. [Cross-engine correctness](#cross-engine-correctness)
9. [Performance and scalability](#performance-and-scalability)
10. [Predictive analysis (optional extension)](#predictive-analysis-optional-extension)
11. [Figures in the report](#figures-in-the-report)
12. [Limitations](#limitations)
13. [Group information](#group-information)

---

## Research questions

**Main question:** What factors are associated with mortgage application outcomes in 2024?

| | Supporting question | Notebooks |
|---|---|---|
| **RQ1 – Lending landscape** | What are the key patterns in mortgage lending? | `01_rq1_lending_landscape` |
| **RQ2 – Financial conditions** | How do borrowers' financial profiles relate to lending outcomes? | `02_rq2_financial_profile(s)` |
| **RQ3 – Loan characteristics** | How do lending outcomes vary across loan products, regions and institutions? | `03_rq3_geography_lenders` |
| **RQ4 – Prediction** *(optional)* | How accurately can mortgage outcomes be predicted on unseen lenders? | `05`–`07` model notebooks |

RQ1–RQ3 are supported by a denial diagnosis of the reasons lenders report for denied applications (`04_denial_diagnostics`, report Appendix B). Pandas–PySpark correctness and runtime comparisons support the analysis; they are validation and benchmarking activities rather than a separate research question.

---

## Data source and acquisition

| | |
|---|---|
| **Dataset** | 2024 HMDA One-Year National Loan-Level Dataset (Loan/Application Register, LAR) |
| **Publisher** | Consumer Financial Protection Bureau (CFPB) and Federal Financial Institutions Examination Council (FFIEC) |
| **Reporting year** | 2024 (data freeze 2 June 2026) |
| **Download** | [Official 2024 HMDA data-publication page](https://ffiec.cfpb.gov/data-publication/2024) → *One Year National Loan-Level Dataset* (CSV, ≈636 MB zipped) |
| **Documentation** | [Public HMDA LAR data fields](https://ffiec.cfpb.gov/documentation/publications/loan-level-datasets/lar-data-fields/) and the [snapshot national loan-level dataset page](https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset/2024), which describes the static release and its privacy modifications |
| **Size** | 12,259,199 records × 99 variables; CSV ≈4.4 GB, converted to a 522 MB Parquet file |
| **Unit of observation** | One reported mortgage application or covered loan — records include originations, denials, withdrawals, incomplete applications, purchased loans and preapproval outcomes, not only completed loans |

**Acquisition.** `src/pandas/00_data_preparation.ipynb` downloads the official ZIP file, extracts the CSV and converts it once to Parquet; all later notebooks read that Parquet copy. Alternatively, download the file manually and place it outside the repository (or in an ignored `raw/` directory).

Raw LAR files and derived row-level Parquet datasets are **deliberately excluded from version control**. A data dictionary for the prepared datasets — column meanings, units, HMDA code lists, DTI/CLTV bands and cleaning rules — is provided in `results/pandas/reference/HMDA_2024_Data_Dictionary.xlsx`.

---

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── Group_6_final_report.pdf         # final technical report
├── src/
│   ├── pandas/                      # 8 Pandas notebooks, rules.py, evaluation_utils.py
│   ├── spark/                       # 5 PySpark notebooks, rules.py
│   └── comparison/                  # 5 correctness, performance and model-comparison notebooks
├── results/                         # 220 aggregated CSV/XLSX outputs (no row-level data)
│   ├── pandas/                      # 134 Pandas outputs
│   │   ├── reference/               # HMDA_2024_Data_Dictionary.xlsx
│   │   ├── prep/audit/              # data audit, cleaning log, feature and sample summaries
│   │   ├── rq1/  rq2/  rq3/         # descriptive and diagnostic tables
│   │   ├── diagnostics/             # denial-reason analysis
│   │   └── rq4/                     # setup/, logistic/, gradient_boosting/
│   └── spark_benchmark/             # 86 PySpark, cross-engine validation and benchmark outputs
│       ├── prep/audit/              # Spark audit, comparison tables with Pandas, schema
│       ├── rq2/                     # correctness/, tables/
│       ├── rq3/                     # spark/, benchmark/
│       ├── diagnostics/             # correctness/, denial_reasons/
│       ├── general/                 # 25/50/100% performance experiment
│       └── rq4/                     # baseline_benchmark/ (LR), xgboost_benchmark/
└── figures/
    ├── used_in_report/              # the 10 figures used in the report
    └── supplementary/               # 44 additional, non-duplicate figures
```

### Notebooks

| Folder | Notebook | Purpose |
|---|---|---|
| `src/pandas/` | `00_data_preparation` | Download, CSV → Parquet, audit, cleaning, features, S0/S1/S2 |
| | `01_rq1_lending_landscape` | Outcomes, baseline denial rate, distributions, loan type × purpose mix |
| | `02_rq2_financial_profiles` | Financial profiles, DTI/CLTV bands, joint heatmap, Spearman correlations |
| | `03_rq3_geography_lenders` | Loan characteristics, state-level and lender-level differences |
| | `04_denial_diagnostics` | Reported denial reasons and financial profiles by reason |
| | `05_modeling_shared_setup` | Predictors, lender-level split, CV folds, preprocessing |
| | `06_model_logistic_regression` | Logistic Regression (baseline and tuned) |
| | `07_model_gradient_boosting` | XGBoost |
| `src/spark/` | `00_data_preparation` | Independent cleaning from the same raw Parquet |
| | `01_rq1_lending_landscape`, `02_rq2_financial_profile`, `03_rq3_geography_lenders`, `diagnostics_spark_extracted` | PySpark versions of the descriptive and diagnostic analyses |
| `src/comparison/` | `rq1_pandas_spark_comparison` | RQ1 cross-engine checks |
| | `02_rq2_correctness_performance` | RQ2 checks and the 25/50/100% performance experiment |
| | `03_rq3_correctness_performance` | RQ3 checks and workload timing |
| | `04_rq4_logistic_baseline_benchmark` | scikit-learn vs Spark MLlib |
| | `05_rq4_gradient_boosting_benchmark` | XGBoost vs SparkXGBClassifier |

`rules.py` is identical in `src/pandas/` and `src/spark/`. Legacy working folders (`notebooks/`, `figure/`, `hmda/`, `HMDA_Thuy/` and data archives) are kept locally and ignored by Git; `src/`, `results/` and `figures/` are the publishable, organised copies.

---

## Installation and execution

### Technology stack

Python, Pandas, PySpark, PyArrow, NumPy, scikit-learn, XGBoost, Matplotlib, Seaborn, SciPy, statsmodels and GeoPandas, with the supporting utilities listed in `requirements.txt`.

### Environment used

The notebooks were developed and executed on **Google Colab** (CPU runtime, 2 cores, ≈12.7 GB RAM) with **PySpark 4.0.5 in local mode**. Data were stored on Google Drive under `/content/drive/MyDrive/hmda/`.

### Local setup

```bash
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

PySpark also requires a compatible Java runtime with `JAVA_HOME` configured.

### Execution order

1. Download the 2024 HMDA LAR data (see [Data source and acquisition](#data-source-and-acquisition)).
2. Set each notebook's input/output base-directory variable (`BASE`) to your data location. On Colab, mount Google Drive and keep `/content/drive/MyDrive/hmda/`.
3. Run the notebooks in this order:

| Step | Notebook(s) | Main output |
|---|---|---|
| 1 | `src/pandas/00_data_preparation.ipynb` | Raw Parquet, `rules.py`, Pandas S0/S1/S2, audit tables |
| 2 | `src/spark/00_data_preparation.ipynb` | PySpark S0/S1/S2, audit and comparison tables |
| 3 | RQ1–RQ3 and denial-diagnosis notebooks (Pandas, then PySpark) | Descriptive and diagnostic tables and figures |
| 4 | `src/pandas/05`–`07` notebooks | Model setup, Logistic Regression, XGBoost |
| 5 | `src/comparison/*` | Cross-engine correctness checks and performance results |

The committed notebooks retain the outputs of their original runs. Execution was not repeated during repository organisation, so sampled data, seeds, thresholds, model parameters and saved numerical outputs remain unchanged.

---

## Data engineering and preparation

Both engines read the **same raw Parquet file**, select the same **31 source variables** and apply the **same rules** (`rules.py`) independently; the PySpark workflow does not reuse Pandas-cleaned data.

**Data quality.** Missingness and special reporting codes were substantial for financial variables before cleaning: income 14.52%, DTI 35.46%, CLTV 35.22% and property value 22.38%.

**Cleaning rules**
- Blank values, `NA`, `Exempt` and field-specific codes (`1111` in categorical code fields, age code `8888`) are treated as missing where applicable. A value that is a special code in one field is not automatically treated as missing in another.
- Numerical variables are converted to numeric types; identifiers and categorical codes are preserved.
- Income ≤ 0 (165,254 records) and CLTV > 250% (3,680 records) are set to missing and flagged.
- Other extreme values and potential duplicates (71,299 records) are retained to avoid unjustified exclusions; public HMDA data has no application identifier.
- **All 12,259,199 records are preserved.** No blanket zero imputation or complete-case deletion is applied; each analysis applies its own missing-value policy.

**Derived features**

| Feature | Definition |
|---|---|
| `denied_flag` | 1 if `action_taken` = 3; 0 for originated loans in S1 |
| `dti_band` | < 20%, 20–< 30%, 30–< 36%, 36–43%, 44–49%, 50–60%, > 60% |
| `cltv_band` | ≤ 60%, > 60–80%, > 80–90%, > 90–95%, > 95–100%, > 100–250% (inclusive upper boundaries in both engines) |
| `log_income`, `log_loan_amount` | Natural logarithms |
| `loan_to_income` | Loan amount / (reported income × 1,000) |

**Storage.** Each engine saves S0, S1 and S2 as Snappy-compressed Parquet files partitioned by `state_code`; record counts are verified after writing.

| Sample | Selection | Records | Purpose |
|---|---|---:|---|
| S0 | All reported records | 12,259,199 | Lending landscape |
| S1 | `action_taken` ∈ {1, 3} | 8,300,547 | Financial comparisons and prediction |
| S2 | `action_taken` = 3 | 2,103,462 | Denial diagnosis |

Throughout S1 analyses, the denial rate is denied / (originated + denied), giving a **baseline of 25.34%**.

---

## Main findings

**RQ1 – Lending landscape**
- Originations made up 50.55% of S0 records and denials 17.16%. Withdrawn and incomplete applications (2.12 million) nearly equalled denials (2.10 million).
- Loan amount and income were highly right-skewed (median loan USD 225,000; median income USD 105,000, with mean income about three times the median).
- Conventional loans accounted for 75.79% of applications and home purchases for 53.67%. Denial rates varied more by loan purpose (15.01% for home purchase to 44.41% for other purposes) than by loan type.

**RQ2 – Financial conditions**
- Originated applications had higher median income (USD 114,000 vs 82,000) and loan amounts (USD 235,000 vs 125,000) than denied applications.
- Denial rates rose sharply above a DTI of 50%: 40.04% at 50–60% and 92.13% above 60%. Both DTI and CLTV showed non-monotonic relationships with denial.
- CLTV was missing for 17.88% of denied applications compared with 7.54% of originations.

**RQ3 – Loan characteristics, geography and lenders**
- Manufactured housing (59.99% vs 23.34% for site-built) and subordinate liens (37.53% vs 20.71% for first liens) had higher denial rates.
- Northern Midwestern states generally had denial rates below the baseline, while several Southern states were above it.
- Among lenders with over 100,000 applications, denial rates ranged from 3.52% to 76.43%; portfolio composition explains only part of this variation.

**Denial diagnosis** — DTI (36.79% of denied applications) and credit history (32.31%) were the most frequently reported denial reasons, followed by collateral (16.50%).

---

## Pandas and PySpark implementations

- **Pandas** runs on a single Colab machine, using PyArrow for column-selective loading and categorical data types to reduce memory (the initial DataFrame occupied ≈0.63 GB), allowing full-data processing without sampling or chunking.
- **PySpark** runs in local mode on the same environment, using DataFrame expressions for cleaning, window functions for duplicate detection, and `groupBy`, `agg` and `pivot` for analysis. Only aggregated outputs are converted to Pandas.
- Both implementations keep explicit missing categories in DTI/CLTV band comparisons and export tables and figures as CSV and PNG files.

---

## Cross-engine correctness

Outputs were compared using the same analytical populations and definitions and aligned by categorical keys or identifiers rather than row order. Counts and group membership required exact agreement; floating-point statistics used metric-specific tolerances.

| Area | Scope | Result | Evidence |
|---|---|---|---|
| Data preparation | S0/S1/S2 counts, duplicates, validity flags, 12 cleaning-log entries, 15 band counts, missingness, 30 feature summaries, schema | All matched; largest difference 0.0035 (approximate percentile) | `results/spark_benchmark/prep/audit/*compare*.csv` |
| RQ1 | 8 comparison groups | All passed (counts exact; percentiles within 1%) | `results/pandas/rq1/RQ1_compare_pandas_spark.csv` |
| RQ2 | Missingness, financial summaries, DTI/CLTV bands, 56 joint cells, 10 Spearman pairs | All passed; maximum difference 1.69 × 10⁻¹¹ | `results/spark_benchmark/rq2/correctness/` |
| RQ3 | 15 exported tables and 72 workload checks | All passed, no mismatches | `results/spark_benchmark/rq3/benchmark/pandas_spark_correctness_summary.csv` |
| Denial diagnosis | Coverage, 9 reasons, 44 financial-profile groups, 88 DTI cells | Exact match | `results/spark_benchmark/diagnostics/correctness/` |
| Logistic Regression | scikit-learn vs Spark MLlib (baseline configuration) | Closely comparable (test ROC-AUC 0.828668 vs 0.828759) | `results/spark_benchmark/rq4/baseline_benchmark/test_metric_comparison.csv` |
| XGBoost | Python vs SparkXGBClassifier | Identical out-of-fold and test probabilities | `results/spark_benchmark/rq4/xgboost_benchmark/23_test_metrics_comparison.csv` |

The checks are implemented in `src/comparison/` and in the PySpark preparation and diagnosis notebooks.

---

## Performance and scalability

**Workload:** group S1 applications by state and loan type, computing application counts, denial counts, denial rates and mean loan amounts. Both engines used identical four-column Parquet inputs drawn as nested random samples without replacement (seed 42). Each configuration was run five times with alternating execution order. Timing covered Parquet reading, missing-value normalisation, aggregation and result materialisation; Spark was pre-initialised and no explicit caching was used.

| S1 share | Records | Pandas (s) | PySpark (s) | Speedup (T<sub>Pandas</sub>/T<sub>Spark</sub>) | Faster |
|---:|---:|---:|---:|---:|---|
| 25% | 2,075,136 | 0.549 | 1.354 | 0.405 | Pandas |
| 50% | 4,150,273 | 1.039 | 2.421 | 0.429 | Pandas |
| 100% | 8,300,547 | 2.144 | 2.725 | 0.787 | Pandas |

Median of five runs. Pandas was faster at every tested volume, but Spark's relative performance improved as data volume increased; no crossover occurred within the tested range. Run-level timings and the summary table are in `results/spark_benchmark/general/` (`run_20261118T142526405847Z/benchmark_summary.csv`), with the equivalence check of the workload outputs in `workload_correctness.csv`. Supplementary model-execution benchmarks are reported in Appendix E of the report and in `results/spark_benchmark/rq4/`.

---

## Predictive analysis (optional extension)

- **Target:** `denied_flag` on S1 (25.34% denied, 74.66% originated), 8,300,547 applications from 4,909 lenders.
- **Predictors (12):** log income, loan-to-income ratio, DTI band, log loan amount, property value, CLTV, loan type, loan purpose, lien status, occupancy type, construction method and state code. Outcome-dependent variables, lender identifiers, census-tract characteristics and technical flags are excluded; demographic attributes are not used as predictors.
- **Evaluation:** 80:20 lender-level train/test split (no lender overlap), five lender-grouped CV folds, preprocessing and class weights fitted within training folds, model selection by mean CV ROC-AUC, and an F1-maximising threshold chosen on out-of-fold predictions before the single test evaluation.

| Held-out test (unseen lenders) | Optimised LR | XGBoost |
|---|---:|---:|
| ROC-AUC | 0.8289 | 0.8604 |
| Average Precision | 0.6827 | 0.7377 |
| F1-score | 0.6061 | 0.6619 |
| Brier score | 0.1606 | 0.1419 |

DTI band contributed 45.43% of XGBoost's total split gain. Because some missing information may depend on the application outcome, the models are interpreted as retrospective denial classifiers rather than pre-decision underwriting tools. Fitted model binaries, row-level predictions and cross-validation checkpoints are excluded from the repository.

---

## Figures in the report

The 10 figures used in the report are in `figures/used_in_report/`. Every other non-duplicate figure is in `figures/supplementary/`.

| Report figure | File |
|---|---|
| Fig. 1 – Distribution of application outcomes | `application_outcomes.png` |
| Fig. 2 – Denial rates by loan type and purpose | `denial_by_loan_type_purpose.png` |
| Fig. 3 – Financial profiles of originated and denied applications | `01_financial_boxplots.png` |
| Fig. 4 – Denial rates across DTI and CLTV bands | `02_denial_by_dti_cltv.png` |
| Fig. 5 – Joint DTI–CLTV denial rates | `03_dti_cltv_heatmap.png` |
| Fig. 6 – Denial rates by loan characteristics | `denial_by_loan_characteristics_combined.png` |
| Fig. 7 – Denial rates across U.S. states | `denial_rate_us_map.png` |
| Fig. 8 – Loan portfolio composition across lender groups | `lender_portfolio_composition.png` |
| Fig. 9 – Model setup and experimental design | `model_process.png` |
| Fig. 10 – XGBoost feature importance | `02_xgb_grouped_feature_importance.png` |

---

## Limitations

- HMDA public data is modified to protect applicant and borrower privacy (for example, loan amounts are disclosed as midpoints of USD 10,000 intervals), and some fields are exempt or unavailable for certain institutions.
- Results are observational associations. Credit scores and other underwriting variables are not available in the public data, so findings do not establish causal effects or underwriting practices.
- Demographic variables were not used for outcome comparisons or modelling; differences in denial rates must not be interpreted as evidence of discrimination.
- Performance results come from a single Colab machine with Spark in local mode and uncontrolled filesystem caching; multi-node scalability was not tested.
- Results depend on the exact data release, cleaning rules, sampling, compute resources, Spark/Java environment and model configuration.
- The repository does not distribute raw records, intermediate Parquet data, prediction-level files, model binaries or checkpoints.

---

## Group information

**Big Data Group 6 — National Economics University**

| Member | Student ID |
|---|---|
| Hoàng Việt Hưng | 11230749 |
| Ninh Thị Minh Thủy | 11230779 |
| Chu Việt Anh | 11230730 |

**Instructor:** Huan Vu, Faculty of Data Science and Artificial Intelligence, National Economics University.
