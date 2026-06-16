# Sembada Cloud

Cloud resource and cost prediction using machine learning and deep learning, following CRISP-ML(Q) and CAMS DevOps. Part of a master's thesis.

**Dataset:** [Azure Public Dataset V2](https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetV2.md) (2019 VM traces) — 2.7M VMs, 30-day trace. Chosen over Google Borg (1TB+, container scheduling, not VMs) and Alibaba ClusterData (12-day traces, too shallow for waste analysis).

**Stack:** Python 3.13, scikit-learn, XGBoost, TensorFlow/Keras, DuckDB, SHAP.

## Prediction Tasks & Models

| Task | Target | Models |
|------|--------|--------|
| CPU/waste/cost regression | `avg_cpu`, `waste_fraction`, `vm_cost` | Ridge, Random Forest, XGBoost |
| Idle VM detection | `is_idle` (binary) | Logistic Regression, RF, XGBoost |
| Waste tier classification | `waste_tier` (Low/Med/High) | Random Forest, XGBoost |
| Workload segmentation | — (clustering) | K-Means (K=4) |
| Cost spike detection | — (anomaly) | Isolation Forest |
| CPU timeseries forecasting | `avg_cpu` (1-step, 24-lag) | ARIMA, LSTM, BiGRU, CNN-LSTM |

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download & prepare data
Run notebooks in order:
```bash
# Download Azure dataset (vmtable, subscriptions, deployments, CPU readings)
jupyter nbconvert --to notebook --execute notebooks/00_download_azure_v2_dataset.ipynb

# Convert CSV to Parquet
jupyter nbconvert --to notebook --execute notebooks/01_convert_to_parquet.ipynb
```

Expected data layout:
```
data/transformed/parquet/
├── vmtable.parquet
├── subscriptions.parquet
├── deployments.parquet
├── azure_pricing.parquet
└── cpu_readings/
    └── *.parquet
```

### 3. Run tests
```bash
pytest app/tests/ -v --cov=app.src --cov-report=term
```

### 4. Execute analysis notebooks
```bash
# 03a — Feature engineering
papermill notebooks/03a_feature_engineering.ipynb output.ipynb --log-output --progress-bar --execution-timeout 600

# 03b — Tabular models (Ridge, RF, XGBoost, K-Means, Isolation Forest, SHAP)
papermill notebooks/03b_tabular_models.ipynb output.ipynb --log-output --progress-bar --execution-timeout 600

# 03c — Timeseries forecasting (ARIMA, LSTM, BiGRU, CNN-LSTM)
papermill notebooks/03c_timeseries_forecasting.ipynb output.ipynb --log-output --progress-bar --execution-timeout 600
```

Or open interactively:
```bash
jupyter notebook notebooks/03a_feature_engineering.ipynb
```

### 5. QA compliance report
```bash
python -m app.src.qa_report
```

## Quality Gates

Assertion cells embedded in notebooks. Execution exits with non-zero code on failure:

| Gate | Notebook | Fails When |
|------|----------|------------|
| Data Quality | `03a §2.2` | Data empty, no unique VMs, missing columns |
| Feature Validation | `03a §3.2` | Missing target/feature columns, invalid ranges |
| Model Acceptance | `03b §4.8` | R² < 0.7 |
| Classification Gate | `03b §5.3` | F1 < 0.80 |
| Timeseries Gate | `03c §8.8` | MAE ≥ 5.0 |

## CI/CD

`.github/workflows/ci.yml` runs `pytest` on every push/PR (synthetic fixtures, no real data needed due to limited resources of GithubCI). Notebook execution is local only (~100GB dataset).

## Project Structure

```
sembada-cloud/
├── notebooks/
│   ├── 00_download_azure_v2_dataset.ipynb
│   ├── 01_convert_to_parquet.ipynb
│   ├── 02_azure_descriptive_analysis.ipynb
│   ├── 03a_feature_engineering.ipynb
│   ├── 03b_tabular_models.ipynb
│   └── 03c_timeseries_forecasting.ipynb
├── app/src/
│   ├── features.py          — Feature engineering
│   ├── models.py            — Model wrappers (fit/predict/evaluate/save)
│   ├── visualize.py         — Publication-quality plots
│   └── qa_report.py         — QA compliance report
├── app/tests/
│   ├── conftest.py          — Shared fixtures
│   ├── test_features.py     — ~14 tests
│   └── test_model.py        — ~12 tests
├── models/                  — Saved model artifacts + run_log.csv
├── .github/workflows/ci.yml
├── requirements.txt
└── docs/
```

## Linting

```bash
black --check app/src/ app/tests/ && flake8 app/src/
black app/src/ app/tests/        # auto-format
```

See `docs/` for detailed methodology (Chapter 3), thesis sections, and technical specs.
