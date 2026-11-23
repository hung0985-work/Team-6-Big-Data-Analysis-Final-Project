# Large-Scale Analysis of Mortgage Lending Patterns Using 2024 HMDA Data

**Big Data Group 6 — National Economics University**

## Project overview

This project studies U.S. mortgage lending patterns in the 2024 Home Mortgage Disclosure Act (HMDA) public data. It contains reproducible Pandas and PySpark workflows for data preparation, descriptive and diagnostic analysis, cross-engine validation, scalability benchmarking, and predictive modeling of application denial.

## Research questions

1. What are the national distributions of mortgage application outcomes, loan types, purposes, and financial characteristics?
2. How do income, loan amount, debt-to-income ratio (DTI), and combined loan-to-value ratio (CLTV) differ by outcome?
3. How do denial patterns vary geographically and across lenders and loan characteristics?
4. Do Pandas and PySpark produce equivalent analytical results, and how do their runtimes compare?
5. How accurately do Logistic Regression and XGBoost predict denials?

## Dataset information

The analysis uses the 2024 HMDA national Loan/Application Register (LAR) data. Obtain the raw source from the [official 2024 HMDA data-publication page](https://ffiec.cfpb.gov/data-publication/2024); the [snapshot national loan-level dataset page](https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset/2024) describes the static national release and its privacy modifications. The local `HMDA_2024_Data_Dictionary.xlsx` is retained as a reference and copied to `results/`.

Raw LAR files and derived row-level Parquet datasets are deliberately excluded from version control. Put downloaded data outside the repository (or in an ignored `raw/` directory), then update the data-root variables in the setup cells of the notebooks before execution. The copied notebooks retain their original Google Colab paths so that their historical computations and outputs remain unchanged.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── pandas/       # eight Pandas notebooks and shared helpers
│   ├── spark/        # five PySpark notebooks and rules helper
│   └── comparison/   # five correctness/performance/model comparison notebooks
├── results/           # 221 deduplicated aggregated CSV/XLSX outputs
├── figures/
│   ├── used_in_report/  # 10 figures from the authoritative legacy figure/ folder
│   └── supplementary/   # 44 additional, non-duplicate figures
└── report/            # final_report.pdf was not supplied in the workspace
```

The legacy `notebooks/`, `figure/`, `hmda/`, `HMDA_Thuy/`, and data/archive folders are intentionally retained locally and ignored by Git. `src/`, `results/`, and `figures/` are the publishable, organized copies.

## Technology stack

Python, Pandas, PySpark, PyArrow, NumPy, scikit-learn, XGBoost, Matplotlib, Seaborn, SciPy, statsmodels, GeoPandas, and supporting utilities listed in `requirements.txt`.

## Data cleaning and feature engineering

`00_data_preparation.ipynb` in both engine folders performs the staged cleaning workflow, auditing special/missing codes, action outcomes, implausible values, schema, sample sizes, and engineered numeric features. RQ4 setup artifacts document train/test partitioning, lender-aware folds, preprocessing, feature selection, transformations, and validation checks.

## Descriptive and diagnostic analysis

The RQ1 notebooks cover application outcomes, loan type/purpose mixes, and distributions. RQ2 covers financial profiles, DTI/CLTV bands, missingness, and correlations. RQ3 evaluates state, lender, and portfolio patterns. The denial-diagnostics notebooks investigate denial-reason frequencies and their financial profiles.

## Pandas and PySpark implementations

Parallel implementations are available in `src/pandas/` and `src/spark/`. The content was classified from the actual notebook imports and workflow code; original notebook filenames have been preserved.

## Cross-engine correctness

`src/comparison/` contains the RQ1, RQ2, RQ3, Logistic Regression, and gradient-boosting comparison notebooks. Result tables include matched-key, value-difference, schema, and runtime checks. For example, `results/pandas_spark_correctness_summary.csv` records `PASS` for the collected RQ3 comparisons, and `results/23_test_metrics_comparison.csv` records equal final Python/Spark XGBoost metrics.

## Performance and scalability

Benchmark outputs are retained directly under `results/`. Where different tables shared a filename, the name is prefixed with its former workflow path (for example, `rq3_benchmark__benchmark_summary.csv`) so no output is overwritten. The collection retains timing trials, summaries, schedules, workload results, and model benchmarks without row-level prediction files.

## Predictive modeling

The Pandas workflow includes shared model setup, Logistic Regression, and XGBoost notebooks. Model-output tables include cross-validation results, tuning results, thresholds, test metrics, coefficient/feature importance tables, and LR–XGBoost comparisons. Fitted model binaries, row-level predictions, and cross-validation checkpoints are excluded from the publishable repository.

## Installation and execution

```bash
python -m venv .venv
.venv\\Scripts\\activate        # Windows PowerShell
pip install -r requirements.txt
```

Install a compatible Java runtime and configure `JAVA_HOME` before running PySpark notebooks. Then:

1. Download the official 2024 HMDA LAR source outside version control.
2. Set each notebook's input/output base-directory variables to your local data location and the project root.
3. Run the Pandas and PySpark preparation notebooks, followed by RQ1–RQ3 and diagnostic workflows.
4. Run RQ4 setup and model notebooks, then the comparison notebooks.

Execution was not rerun during repository organization: this avoids changing sampled data, seeds, thresholds, model parameters, or saved numerical outputs.

## Figures and tables

All 10 files that were in the authoritative legacy `figure/` directory are in `figures/used_in_report/`:

- `01_financial_boxplots.png`
- `02_denial_by_dti_cltv.png`
- `02_xgb_grouped_feature_importance.png`
- `03_dti_cltv_heatmap.png`
- `application_outcomes.png`
- `denial_by_loan_characteristics_combined.png`
- `denial_by_loan_type_purpose.png`
- `denial_rate_us_map.png`
- `lender_portfolio_composition.png`
- `model_process.png`

Every other discovered, non-duplicate figure is in `figures/supplementary/`, retaining its originating `hmda/` or `HMDA_Thuy/` path. `results/` contains 221 deduplicated aggregated CSV/XLSX artifacts as direct files. It includes outcomes, financial profiles, DTI/CLTV, geography/lenders, correctness, benchmarks, Logistic Regression, and XGBoost outputs.

## Key findings

- The saved national outcome table reports 6,197,085 originated applications (50.55%) and 2,103,462 denials (17.16%).
- In the saved financial summary, denied applications have lower median income ($82k versus $114k) and lower median loan amount ($125k versus $235k) than originated applications.
- The collected RQ3 benchmark reports exact Pandas–PySpark agreement for its checked tables; on that recorded environment, Pandas is faster for the four RQ3 steps.
- On the recorded test set, XGBoost outperforms Logistic Regression in ROC-AUC (0.8604 versus 0.8289) and average precision (0.7377 versus 0.6827).

## Limitations

HMDA public data is modified to protect applicant and borrower privacy. Results depend on the exact data release, cleaning rules, sampling, available compute resources, Spark/Java environment, and model configuration. The repository does not distribute raw records, intermediate Parquet data, prediction-level files, model binaries, or checkpoints.

## Group information

Big Data Group 6, National Economics University. Individual member information was not supplied in this workspace.
