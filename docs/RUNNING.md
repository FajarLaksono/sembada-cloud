# Running the Predictive Analysis Pipeline

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note:** TensorFlow is optional (commented out in `requirements.txt`). To run deep learning sections (§8.4–8.6):
> ```bash
> pip install tensorflow>=2.15.0
> ```

### 2. Verify Data Files
The notebook requires parquet data in `data/transformed/parquet/`. Expected files:
```
data/transformed/parquet/
├── vmtable.parquet
├── subscriptions.parquet
├── deployments.parquet
├── azure_pricing.parquet
└── cpu_readings/
    └── *.parquet
```

### 3. Run Unit Tests
```bash
# From project root
pytest app/tests/ -v
```
Expected: **9 tests** (5 feature tests + 4 model tests).

### 4. Execute Notebook
```bash
# End-to-end execution (no output file saved)
jupyter nbconvert --to notebook \
  --execute notebooks/03_predictive_analysis.ipynb \
  --output /dev/null \
  --ExecutePreprocessor.timeout=600
```

Or open interactively:
```bash
jupyter notebook notebooks/03_predictive_analysis.ipynb
```

---

## What the Notebook Does (§1–11)

| Section | Topic | Runs Without TF? |
|---------|-------|-----------------|
| §1 | Summary & success criteria | Yes |
| §2 | Imports & data loading (DuckDB) | Yes |
| §3 | Feature engineering, correlation, train-test split | Yes |
| §4 | Regression: Linear → Ridge → RF → XGBoost → CatBoost | Yes |
| §5 | Classification: idle detection + waste tier | Yes |
| §6 | Clustering: K-Means, optimal K, PCA | Yes |
| §7 | Anomaly detection: Isolation Forest, cost impact | Yes |
| §8 | Timeseries: ARIMA, LSTM, BiGRU, CNN-LSTM | No (skips gracefully) |
| §9 | SHAP explainability | No (skips gracefully) |
| §10 | Model comparison, benchmarks, business impact | Yes |
| §11 | Conclusions, limitations, future work | Yes |

All sections gracefully handle missing packages (TensorFlow, SHAP) with clear skip messages.

---

## CI/CD

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs:
1. `pytest app/tests/ -v --tb=short -x`
2. `jupyter nbconvert --execute notebooks/03_predictive_analysis.ipynb`

Triggered on push to `main`/`develop` and PRs to `main`.

---

## Project Structure

```
sembada-cloud/
├── notebooks/
│   └── 03_predictive_analysis.ipynb   ← Predictive analysis (this notebook)
├── app/src/
│   ├── features.py                    ← Feature engineering
│   ├── models.py                      ← Model wrappers (XGBoost, CatBoost, etc.)
│   └── visualize.py                   ← Plotting functions
├── app/tests/
│   ├── conftest.py                    ← Pytest fixtures
│   ├── test_features.py               ← 5 feature tests
│   └── test_model.py                  ← 4 model tests
├── models/
│   ├── .gitkeep
│   └── run_log.csv                    ← Experiment audit trail
├── .github/workflows/ci.yml           ← GitHub Actions
├── requirements.txt
└── docs/
    ├── AZURE_PREDICTIVE_ANALYSIS_PLAN.md
    ├── IMPLEMENTATION_PROGRESS.md
    └── RUNNING.md                     ← This file
```
