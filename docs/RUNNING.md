# Running the Predictive Analysis Pipeline

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install nbconvert jupyter nbformat   # for notebook execution
```

For deep learning sections (§8.4–8.6 in `03c`), also install TensorFlow:
```bash
pip install tensorflow>=2.15.0
```

### 2. Verify Data Files
Requires parquet data in `data/transformed/parquet/`:
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
py -m pytest app/tests/ -v
```
Expected: **~16 tests pass** (feature tests + model tests).

With coverage:
```bash
py -m pytest app/tests/ --cov=app.src --cov-report=term
```

### 4. Execute Notebooks (in order)
```bash
# 03a — Feature Engineering (quality gates: data quality, feature validation)
py -m papermill notebooks/03a_feature_engineering.ipynb NUL --log-output --progress-bar --execution-timeout 600

# 03b — Tabular Models (quality gates: model acceptance, classification, Q summary)
py -m papermill notebooks/03b_tabular_models.ipynb NUL --log-output --progress-bar --execution-timeout 600

# 03c — Timeseries Forecasting (quality gate: timeseries acceptance)
py -m papermill notebooks/03c_timeseries_forecasting.ipynb NUL --log-output --progress-bar --execution-timeout 600
```

Or open interactively:
```bash
py -m jupyter notebook notebooks/03a_feature_engineering.ipynb
```

### 5. Generate QA Compliance Report
```bash
# After 03b has generated models/run_log.csv
python -m app.src.qa_report
```

### 6. Dependency Lockfile
```bash
# After a known-good install, freeze exact versions
pip freeze --exclude-editable > requirements.lock
```

---

## Notebook Architecture (3-Notebook Split)

| Notebook | Sections | Scope | Quality Gates |
|----------|----------|-------|---------------|
| `03a_feature_engineering.ipynb` | §1–§3 | Load 5 tables, engineer features, save `features_df.parquet` | Risk Register, Data Quality, Feature Validation |
| `03b_tabular_models.ipynb` | §4–§7, §9–§11 | Train Ridge/RF/XGBoost, K-Means, Isolation Forest, SHAP | Model Acceptance, Classification Gate, Q Summary Report |
| `03c_timeseries_forecasting.ipynb` | §8 only | Load CPU readings (DuckDB), train ARIMA/LSTM/BiGRU/CNN-LSTM | Timeseries Acceptance Gate |

**Artifact handoff:** `03a` saves `features_df.parquet`. `03b` and `03c` load it — no shared memory.

---

## Quality Gates

CRISP-ML(Q) assertion cells embedded in the notebooks. When a gate fails, `nbconvert --execute` exits with non-zero code and a clear error message:

| Notebook | Gate | Asserts | Fails When |
|----------|------|---------|------------|
| 03a §1.5 | Risk Register | 0 (markdown) | Never (informational) |
| 03a §2.2 | Data Quality | 3 | Data empty, no unique VMs, missing required columns |
| 03a §3.2 | Feature Validation | 8+ | Missing target/feature columns, invalid ranges |
| 03b §4.8 | Model Acceptance | ~6/model | MAPE > 15% or R² < 0.7 |
| 03b §5.3 | Classification Gate | ~3/model | F1 < 0.80 |
| 03b §11 | Q Summary Report | 0 (informational) | Never — prints pass/fail status |
| 03c §8.8 | Timeseries Gate | ~3/model | MAE >= 5.0 |

---

## Linting & Formatting

```bash
# Check code style
black --check app/src/ app/tests/
flake8 app/src/

# Auto-format
black app/src/ app/tests/
```

Both tools use configuration from `pyproject.toml` (line-length=120, Python 3.14 target, black-compatible flake8 ignores).

---

## CI/CD Pipeline

`.github/workflows/ci.yml` runs on push to `main`/`develop` and PRs to `main`:

1. **Setup** — Python 3.14 with pip cache
2. **Install** — `requirements.lock` if present, else `requirements.txt`
3. **Unit tests** — `pytest app/tests/ -v --tb=short -x --cov=app.src --cov-report=term`
4. **Quality gates** — `papermill` execution on all 3 notebooks (600s timeout, `--log-output` for real-time progress)

Run full CI locally:
```powershell
pytest app/tests/ -v --tb=short -x --cov=app.src --cov-report=term; if ($?) { papermill notebooks/03a_feature_engineering.ipynb NUL --log-output --progress-bar --execution-timeout 600 }
```

---

## Project Structure

```
sembada-cloud/
├── notebooks/
│   ├── 03a_feature_engineering.ipynb       ← §1–§3: Feature engineering
│   ├── 03b_tabular_models.ipynb            ← §4–§7, §9–§11: Tabular models
│   └── 03c_timeseries_forecasting.ipynb    ← §8: Timeseries forecasting
├── app/src/
│   ├── __init__.py
│   ├── features.py                         ← Feature engineering functions
│   ├── models.py                           ← Model wrappers (XGBoost, RF, etc.)
│   ├── visualize.py                        ← Publication-quality plotting
│   └── qa_report.py                        ← QA compliance report utility
├── app/tests/
│   ├── __init__.py
│   ├── conftest.py                         ← Pytest fixtures
│   ├── test_features.py                    ← Feature tests (~14 tests)
│   └── test_model.py                       ← Model tests (~12 tests)
├── models/
│   ├── .gitkeep
│   ├── regression/                         ← Best regressors
│   ├── classification/                     ← Best classifiers
│   ├── clustering/                         ← K-Means + scaler
│   ├── timeseries/                         ← LSTM/GRU models
│   └── run_log.csv                         ← Training run audit trail
├── .github/workflows/ci.yml                ← GitHub Actions CI/CD
├── pyproject.toml                          ← Lint/test configuration
├── requirements.txt                        ← Loose version bounds
├── requirements.lock                       ← Exact pinned versions
├── .gitignore
├── AGENTS.md
└── docs/
    ├── AZURE_PREDICTIVE_ANALYSIS_PLAN.md
    ├── AUTOMATED_QA_TECH_SPEC.md
    ├── IMPLEMENTATION_PROGRESS.md
    └── RUNNING.md                          ← This file
```

---

## Notebook Contents

| Section | Topic | Runs Without TF? |
|---------|-------|-----------------|
| §1 | Summary & success criteria | Yes |
| §2 | Imports & data loading (DuckDB) | Yes |
| §3 | Feature engineering, correlation, train-test split | Yes |
| §4 | Regression: Ridge → RF → XGBoost | Yes |
| §5 | Classification: idle detection + waste tier | Yes |
| §6 | Clustering: K-Means, optimal K, PCA | Yes |
| §7 | Anomaly detection: Isolation Forest, cost impact | Yes |
| §8 | Timeseries: ARIMA, LSTM, BiGRU, CNN-LSTM | No |
| §9 | SHAP explainability | No |
| §10 | Model comparison, benchmarks, business impact | Yes |
| §11 | Conclusions, limitations, future work | Yes |

All sections gracefully handle missing packages with clear skip messages.
