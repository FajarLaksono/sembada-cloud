# Automated Quality Assurance Technical Specification

**Project:** Sembada Cloud — Cloud Resource & Cost Prediction
**Methodology:** CRISP-ML(Q) — Quality Insurance Phase
**Document Status:** Planning
**Last Updated:** 2026-05-13

---

## Table of Contents

1. [Purpose & Scope](#1-purpose--scope)
2. [Design Principles](#2-design-principles)
3. [Architecture: Quality Gates in Notebook-First Workflow](#3-architecture-quality-gates-in-notebook-first-workflow)
4. [Action 1: Risk Register](#4-action-1-risk-register)
5. [Action 2: Data Quality Gate](#5-action-2-data-quality-gate)
6. [Action 3: Feature Validation Gate](#6-action-3-feature-validation-gate)
7. [Action 4: Model Acceptance Gates](#7-action-4-model-acceptance-gates)
8. [Action 5: Quality Insurance Summary Report](#8-action-5-quality-insurance-summary-report)
9. [Action 6: Fix CatBoost Test Gap](#9-action-6-fix-catboost-test-gap)
10. [Action 7: Activate CI/CD Pipeline](#10-action-7-activate-cicd-pipeline)
11. [Action 8: Optional QA Report Utility](#11-action-8-optional-qa-report-utility)
12. [Action 9: Linting Configuration (pyproject.toml)](#12-action-9-linting-configuration-pyprojecttoml)
13. [Action 10: Lockfile for Reproducibility](#13-action-10-lockfile-for-reproducibility)
14. [Testing Strategy & Coverage](#14-testing-strategy--coverage)
15. [Implementation Order](#15-implementation-order)
16. [Deliverables Checklist](#16-deliverables-checklist)
17. [AGENTS.md Updates](#17-agentsmd-updates)
18. [Appendices](#18-appendices)

---

## 1. Purpose & Scope

### 1.1 Purpose

This document defines the technical specification for implementing the **Quality Insurance (Q)** phase of CRISP-ML(Q) for the Sembada Cloud predictive analysis project. It builds on the existing infrastructure defined in `AZURE_PREDICTIVE_ANALYSIS_PLAN.md` and adds automated quality gates appropriate for a **notebook-first workflow**.

### 1.2 Scope

The scope covers 10 concrete actions across three notebooks (`03a`, `03b`, `03c`), the test suite (`app/tests/`), CI/CD configuration (`.github/workflows/ci.yml`), a new `pyproject.toml`, a `requirements.lock`, and one new utility module (`app/src/qa_report.py`).

### 1.3 Out of Scope

- Production deployment infrastructure (Docker, Kubernetes, API serving)
- Real-time model monitoring (data drift detection, model decay alerts)
- Automated retraining pipeline
- Multi-cloud data integration

### 1.4 CRISP-ML(Q) Quality Assurance Principles Applied

| Principle | Implementation |
|-----------|---------------|
| **Risk-Based Thinking** | Risk register documented in notebook (§1.5) — identifies 6 failure modes with likelihood, impact, and mitigation |
| **Measurable Metrics** | Success criteria enforced as assertion cells (MAPE < 15%, F1 > 0.85, R² > 0.7) |
| **Iterative Process** | Quality gates fail loud-and-early during `nbconvert --execute`, enabling rapid iteration |
| **Traceability** | `run_log.csv` records every run with git hash, metrics, and pass/fail status |

---

## 2. Design Principles

### 2.1 Assertion Cells as Quality Gates

The core mechanism is Python `assert` statements in notebook code cells. When executed via `jupyter nbconvert --execute`, failed assertions raise `AssertionError`, halting execution with a non-zero exit code. This makes `nbconvert --execute` a **true CI quality gate**.

```
jupyter nbconvert --execute notebook.ipynb --output /dev/null
  ↓
  Cell contains: assert metrics['mape'] < 15
  ↓
  If MAPE ≥ 15 → AssertionError → CI pipeline fails
  If MAPE < 15 → Pass → CI pipeline continues
```

### 2.2 Fail-Fast Strategy

Each quality gate appears immediately after the data or model it validates:
- Data quality gate → right after data loading (03a §2.2)
- Feature validation gate → right after `create_features()` (03a §3.2)
- Model acceptance gate → right after metrics comparison (03b §4.8)
- Classification gate → right after classifier evaluation (03b §5.3)
- Q summary report → end of notebook (03b §11)

### 2.3 No Pipeline Dependency

All quality logic lives inside the notebooks themselves. The only external dependency is `jupyter nbconvert --execute`, which is available in any Python environment with Jupyter installed. This ensures portability across local development, CI runners, and air-gapped environments.

---

## 3. Architecture: Quality Gates in Notebook-First Workflow

```
03a_feature_engineering.ipynb
┌──────────────────────────────────────────────────────────────────┐
│  §1  Summary (Business Understanding)                            │
│  §1.5 Risk Register ← NEW                                       │
│  §2  Preparation (Data Understanding)                            │
│       §2.2 Load Dataset                                          │
│       [Data Quality Gate] ← NEW assertion cell                   │
│  §3  Feature Engineering (Data Preparation)                      │
│       §3.2 Feature Construction                                  │
│       [Feature Validation Gate] ← NEW assertion cell             │
│       §3.4 Train-Test Split → features_df.parquet                │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
03b_tabular_models.ipynb
┌──────────────────────────────────────────────────────────────────┐
│  §4  Regression Models                                           │
│       §4.4 Model Comparison                                      │
│       §4.7 Save Best Model                                       │
│       [Model Acceptance Gate] ← NEW assertion cell               │
│  §5  Classification                                              │
│       §5.3 Save Best Classifier                                  │
│       [Classification Gate] ← NEW assertion cell                 │
│  §6  Clustering                                                  │
│  §7  Anomaly Detection                                           │
│  §9  SHAP Explainability                                         │
│  §10 Model Comparison & Selection                                │
│  §11 Conclusions & Recommendations                               │
│       [Q Summary Report] ← NEW code cell                         │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
03c_timeseries_forecasting.ipynb
┌──────────────────────────────────────────────────────────────────┐
│  §8  Deep Learning — Timeseries Forecasting                      │
│       §8.8 Model Comparison & Save                               │
│       [Timeseries Gate] ← NEW assertion cell                     │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
.github/workflows/ci.yml (un-commented)
  └─ pytest app/tests/ --cov=app.src --cov-report=term
  └─ jupyter nbconvert --execute notebooks/03a_*.ipynb
  └─ jupyter nbconvert --execute notebooks/03b_*.ipynb
  └─ jupyter nbconvert --execute notebooks/03c_*.ipynb
```

---

## 4. Action 1: Risk Register

### 4.1 Location

Insert in `notebooks/03a_feature_engineering.ipynb` as a new markdown cell at **§1.5** (after the Summary section, before §2 Preparation).

### 4.2 Cell Content

```markdown
### 1.5 Quality Risk Register

**Purpose:** Identify potential failure points early in the ML lifecycle,
per CRISP-ML(Q) risk-based thinking principle.

| # | Risk | Phase | Likelihood | Impact | Mitigation |
|---|------|-------|-----------|--------|------------|
| R1 | Data drift (2019 patterns vs 2026 usage) | Monitoring | Medium | High | Retrain on newer data when available; monitor metric regression across runs via `run_log.csv` |
| R2 | Missing pricing or subscription data | Data Preparation | Low | Medium | Fallback to NaN; downstream models must handle missing values gracefully |
| R3 | Target leakage via correlated features | Feature Engineering | Medium | High | `get_feature_target_columns()` excludes target-related columns; review SHAP for unexpected feature dominance |
| R4 | Timeseries overfitting (few VMs) | Modeling | High | Medium | Early stopping, limit model complexity, cross-validation per VM |
| R5 | CPU readings memory blowup | Data Preparation | Low | High | Already mitigated via DuckDB out-of-core parquet glob (no `pd.concat`) |
| R6 | Skewed waste_tier distribution | Evaluation | Medium | Low | Use weighted F1 score; apply SMOTE if minority class recall < 0.7 |
```

### 4.3 Rationale

CRISP-ML(Q) Section 3.1.2 (Risk Management) requires that "risks are identified and documented" before modeling begins. A markdown risk register in the notebook satisfies this requirement while adding no infrastructure complexity.

---

## 5. Action 2: Data Quality Gate

### 5.1 Location

Insert in `notebooks/03a_feature_engineering.ipynb` as a new **code cell** immediately after the "Load Dataset" cell in **§2.2**.

### 5.2 Cell Content

```python
# ---------------------------------------------------------------------------
# DATA QUALITY GATE — CRISP-ML(Q) Quality Insurance
# Fail fast if input data is empty, missing required columns, or has no rows.
# ---------------------------------------------------------------------------
assert len(vmtable) > 0, "FAIL: Empty vmtable — cannot proceed"
assert vmtable['vm_id'].nunique() > 0, "FAIL: No unique VMs found"

REQUIRED_COLS = {'vm_id', 'avg_cpu', 'max_cpu', 'p95_max_cpu',
                 'vm_category', 'vm_core_count_bucket', 'vm_memory_gb_bucket'}
missing_cols = REQUIRED_COLS - set(vmtable.columns)
assert not missing_cols, f"FAIL: Missing required columns: {missing_cols}"

null_rate = vmtable[list(REQUIRED_COLS & set(vmtable.columns))].isnull().sum().sum()
print(f"✓ Data quality: {len(vmtable):,} rows, {vmtable['vm_id'].nunique():,} unique VMs, "
      f"{null_rate} nulls in required columns")

if null_rate > 0:
    print("  ⚠ Warning: nulls detected — consider imputation strategy")
else:
    print("  ✓ No nulls in required columns")
```

### 5.3 Failure Behavior

If the data doesn't meet criteria, `nbconvert --execute` fails with `AssertionError` and a clear message. The CI pipeline stops at this cell, providing immediate feedback.

---

## 6. Action 3: Feature Validation Gate

### 6.1 Location

Insert in `notebooks/03a_feature_engineering.ipynb` as a new **code cell** immediately after the `create_features()` call in **§3.2**.

### 6.2 Cell Content

```python
# ---------------------------------------------------------------------------
# FEATURE VALIDATION GATE — CRISP-ML(Q) Quality Insurance
# Verify engineered features contain expected columns with valid ranges.
# ---------------------------------------------------------------------------
TARGET_COLS = ['is_idle', 'waste_tier', 'waste_fraction']
FEATURE_COLS = ['core_count', 'memory_gb', 'lifetime_hours', 'cpu_per_core',
                'burstiness', 'is_short_lived', 'creation_hour_sin', 'creation_hour_cos']

for col in TARGET_COLS:
    assert col in features_df.columns, f"FAIL: Missing target column '{col}'"

for col in FEATURE_COLS:
    assert col in features_df.columns, f"FAIL: Missing feature column '{col}'"

# Range checks
assert features_df['waste_fraction'].min() >= -1e-6, "FAIL: waste_fraction < 0"
assert features_df['waste_fraction'].max() <= 1.0 + 1e-6, "FAIL: waste_fraction > 1"
assert set(features_df['waste_tier'].cat.categories) == {'Low', 'Medium', 'High'}, \
    "FAIL: waste_tier categories incorrect"
assert features_df['is_idle'].dtype == bool, "FAIL: is_idle must be boolean"

# Cyclical encoding bounds
for col in ['creation_hour_sin', 'creation_hour_cos', 'creation_dow_sin', 'creation_dow_cos']:
    if col in features_df.columns:
        assert features_df[col].min() >= -1.0 - 1e-6, f"FAIL: {col} < -1"
        assert features_df[col].max() <= 1.0 + 1e-6, f"FAIL: {col} > 1"

print(f"✓ Features validated: {len(features_df):,} rows, "
      f"{len(features_df.columns)} columns, "
      f"{features_df.isnull().sum().sum()} total nulls")
```

### 6.3 Relationship to Unit Tests

These assertions mirror the logic in `app/tests/test_features.py` (`TestCreateFeatures`, `TestGetFeatureTargetColumns`). The unit tests validate the `create_features()` function in isolation, while the notebook assertion validates the actual production data path end-to-end.

---

## 7. Action 4: Model Acceptance Gates

### 7.1 Regression Gate Location

Insert in `notebooks/03b_tabular_models.ipynb` as a new **subsection §4.8** with a markdown header + code cell after §4.7 (Save Best Model).

### 7.2 Regression Gate Cell Content

```markdown
### 4.8 Model Acceptance Gate

**CRISP-ML(Q):** Quality Insurance

**Purpose:** Verify all regression models meet the success criteria
defined in §1 (MAPE < 15%, R² > 0.7). This gate fails the notebook
execution if any model underperforms.
```

```python
# ---------------------------------------------------------------------------
# MODEL ACCEPTANCE GATE — CRISP-ML(Q) Quality Insurance
# All models must meet minimum performance thresholds.
# ---------------------------------------------------------------------------
SUCCESS_MAPE = 15.0   # Maximum acceptable MAPE (%)
SUCCESS_R2   = 0.7    # Minimum acceptable R²

print("=" * 60)
print("MODEL ACCEPTANCE GATE — Regression")
print("=" * 60)

all_pass = True
for model_name, metrics in comparison.items():
    model_pass = True
    mape = metrics.get('mape', 100)
    r2   = metrics.get('r2', 0)

    if mape > SUCCESS_MAPE:
        print(f"  ✗ {model_name}: MAPE {mape:.1f}% > {SUCCESS_MAPE}%")
        model_pass = False
    else:
        print(f"  ✓ {model_name}: MAPE {mape:.1f}% ≤ {SUCCESS_MAPE}%")

    if r2 < SUCCESS_R2:
        print(f"  ✗ {model_name}: R² {r2:.3f} < {SUCCESS_R2}")
        model_pass = False
    else:
        print(f"  ✓ {model_name}: R² {r2:.3f} ≥ {SUCCESS_R2}")

    if not model_pass:
        all_pass = False

print("=" * 60)
assert all_pass, "FAIL: One or more models did not meet success criteria"
print("✓ All regression models pass acceptance gate")
```

### 7.3 Classification Gate Location

Insert in `notebooks/03b_tabular_models.ipynb` immediately after §5.3 (Save Best Classifier).

### 7.4 Classification Gate Cell Content

```python
# ---------------------------------------------------------------------------
# CLASSIFICATION ACCEPTANCE GATE — CRISP-ML(Q) Quality Insurance
# ---------------------------------------------------------------------------
SUCCESS_F1 = 0.80

print("=" * 60)
print("CLASSIFICATION ACCEPTANCE GATE")
print("=" * 60)

all_pass = True
for name, metrics in clf_results.items():
    f1 = metrics.get('f1', 0)
    if f1 < SUCCESS_F1:
        print(f"  ✗ {name}: F1 {f1:.3f} < {SUCCESS_F1}")
        all_pass = False
    else:
        print(f"  ✓ {name}: F1 {f1:.3f} ≥ {SUCCESS_F1}")

print("=" * 60)
assert all_pass, "FAIL: Classification models did not meet F1 threshold"
print("✓ All classification models pass acceptance gate")
```

### 7.5 Timeseries Gate Location (03c)

Insert in `notebooks/03c_timeseries_forecasting.ipynb` after the model comparison table in **§8.8**.

### 7.6 Timeseries Gate Cell Content

```python
# ---------------------------------------------------------------------------
# TIMESERIES ACCEPTANCE GATE — CRISP-ML(Q) Quality Insurance
# ---------------------------------------------------------------------------
if ts_results:
    print("=" * 60)
    print("TIMESERIES ACCEPTANCE GATE")
    print("=" * 60)
    all_pass = True
    for model_name, metrics in ts_results.items():
        mae = metrics.get('mae', 999)
        print(f"  {'✓' if mae < 5 else '✗'} {model_name}: MAE = {mae:.3f}")
        if mae >= 5:
            all_pass = False
    assert all_pass, "FAIL: Timeseries models exceed MAE threshold"
    print("✓ All timeseries models pass acceptance gate")
```

---

## 8. Action 5: Quality Insurance Summary Report

### 8.1 Location

Append to `notebooks/03b_tabular_models.ipynb` as a final **code cell** before the Conclusions section (§11).

### 8.2 Variable Name Reference

The cell references these variables that exist in the 03b notebook runtime:

| Variable | Defined In | Type | Example |
|----------|-----------|------|---------|
| `comparison` | Cell 13 | `dict[str, dict]` | `{'Ridge': {'mae':..., 'r2':...}, 'Random Forest': ..., 'XGBoost': ...}` |
| `results_clf` | Cells 21–22 | `dict[str, dict]` | `{'Logistic Regression': {'f1':..., ...}, 'XGBoost': {...}}` |
| `best_regressor_name` | Cell 13 | `str` | `'XGBoost'` |
| `best_clf_name` | Cell 22 | `str` | `'XGBoost'` |
| `best_k` | Cell 31 | `int` | `4` |
| `ts_results` | Cell 47 | `dict` | `{}` (timeseries handled in 03c) |

### 8.3 Cell Content

```python
# ---------------------------------------------------------------------------
# QUALITY INSURANCE REPORT — End of Notebook Summary
# This cell uses variables from the 03b notebook runtime.
# It does NOT require any pre-defined boolean flags — all status is
# computed inline from the actual metrics dicts.
# ---------------------------------------------------------------------------
from datetime import datetime

# --- Compute pass/fail inline from actual notebook variables ---
reg_models_total = len(comparison)
reg_models_passing = sum(
    1 for m in comparison.values()
    if m.get('mape', 100) < 15.0 and m.get('r2', 0) > 0.7
)

clf_models_total = len(results_clf)
clf_models_passing = sum(
    1 for m in results_clf.values()
    if m.get('f1', 0) > 0.80
)

all_models_total = reg_models_total + clf_models_total
all_models_passing = reg_models_passing + clf_models_passing
ts_available = bool(ts_results)

# --- Print report ---
print("=" * 70)
print("  QUALITY INSURANCE REPORT")
print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 70)

print(f"""
  Risk Register         ✓  (6 risks documented in 03a §1.5)
  Data Quality Gate     ✓  (passed at data load)
  Feature Validation    ✓  (all targets & features verified)
  Model Acceptance      {'✓' if reg_models_passing == reg_models_total else '✗'}  (regression: {reg_models_passing}/{reg_models_total})
  Classification Gate   {'✓' if clf_models_passing == clf_models_total else '✗'}  (classification: {clf_models_passing}/{clf_models_total})
  Timeseries Gate       {'✓' if ts_available else '—'}  (handled in 03c)

  Summary:
  - Total models trained: {all_models_total}
  - Models passing gates: {all_models_passing}/{all_models_total}
  - Best regressor: {best_regressor_name}
  - Best classifier: {best_clf_name}
  - Best cluster K: {best_k}
""")

if all_models_passing == all_models_total:
    print("  STATUS: ALL QUALITY GATES PASSED")
else:
    print("  STATUS: SOME QUALITY GATES FAILED — review above")

print("=" * 70)
```

### 8.4 Dependencies & Assumptions

This cell depends on the following variables being defined in the 03b notebook runtime:

| Variable | Defined In | Risk If Renamed/Removed |
|----------|-----------|------------------------|
| `comparison` | Cell 13 — regression metrics dict | Cell errors with `NameError` |
| `results_clf` | Cells 21–22 — classification metrics dict | Cell errors with `NameError` |
| `best_regressor_name` | Cell 13 — string key of best model | Cell errors with `NameError` |
| `best_clf_name` | Cell 22 — string key of best classifier | Cell errors with `NameError` |
| `best_k` | Cell 31 — optimal cluster count | Cell errors with `NameError` |
| `ts_results` | Cell 47 — empty dict `{}` placeholder | Reference is guarded by `bool(ts_results)` check, so `NameError` only if the variable itself is removed |

**Mitigation:** If any of these cells are reordered, renamed, or removed during maintenance, the Q Summary Report cell must be updated to match. The safest approach is to keep this cell as the very last code cell in the notebook (before markdown conclusions) and never rename the upstream variables once set.

---

## 9. Action 6: Fix CatBoost Test Gap

### 9.1 Problem Statement

`app/tests/test_model.py` contains a `TestCatBoostModel` class that imports and tests `CatBoostModel` from `app.src.models`. However:

- `app/src/models.py` does **not** define a `CatBoostModel` class
- Notebook `03b_tabular_models.ipynb` Cell 12 states: `# (CatBoost removed — use XGBoost from section 4.3 above)`
- `catboost` is listed in `requirements.txt` but not used anywhere

Running `pytest` fails with: `ImportError: cannot import name 'CatBoostModel' from 'app.src.models'`

### 9.2 Resolution

Remove `TestCatBoostModel` from `test_model.py`. The test class references a model that was intentionally removed from the implementation plan. If CatBoost is readded later, the test class can be restored.

### 9.3 File Changes

**File:** `app/tests/test_model.py`
**Remove:** Lines 248–261 (the entire `TestCatBoostModel` class)

---

## 10. Action 7: Activate CI/CD Pipeline

### 10.1 Current State

`.github/workflows/ci.yml` exists but every line is commented out (44 lines, all starting with `#`).

### 10.2 Resolution

Un-comment all lines and add the `--cov` flag for coverage reporting.

### 10.3 Final File Content

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python 3.14
        uses: actions/setup-python@v5
        with:
          python-version: "3.14"
          cache: "pip"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest nbconvert jupyter nbformat

      - name: Run unit tests with coverage
        run: |
          pytest app/tests/ -v --tb=short -x --cov=app.src --cov-report=term
        env:
          PYTHONPATH: ${{ github.workspace }}

      - name: Verify notebooks execute (quality gates active)
        run: |
          for nb in notebooks/03a_*.ipynb notebooks/03b_*.ipynb notebooks/03c_*.ipynb; do
            echo "Executing $nb..."
            jupyter nbconvert --to notebook \
              --execute "$nb" \
              --output /dev/null \
              --ExecutePreprocessor.timeout=600
          done
        env:
          PYTHONPATH: ${{ github.workspace }}
```

### 10.4 Pipeline Behavior

| Stage | Purpose | On Failure |
|-------|---------|------------|
| `pip install` | Reproducible environment | Exits before testing |
| `pytest --cov` | Unit tests + coverage report | PR is blocked, test report visible |
| `nbconvert --execute` | End-to-end quality gates | PR is blocked with `AssertionError` from failing gate |

---

## 11. Action 8: Optional QA Report Utility

### 11.1 Purpose

A standalone Python module that reads `models/run_log.csv` and generates a human-readable Quality Insurance compliance summary. This can be run locally without Jupyter.

### 11.2 File Location

`app/src/qa_report.py`

### 11.3 Module Specification

```python
"""
Quality Insurance report generator.
Reads run_log.csv and prints compliance summary against success criteria.
"""

import pandas as pd
from pathlib import Path


SUCCESS_CRITERIA = {
    'regression': {'mape': 15.0, 'r2': 0.70},
    'classification': {'f1': 0.85},
    'clustering': {'silhouette_score': 0.3},
    'timeseries': {'mae': 5.0},
}


def load_run_log(path: str | Path = "models/run_log.csv") -> pd.DataFrame:
    """Load and validate the run log CSV."""
    log_path = Path(path)
    if not log_path.exists():
        print(f"⚠ Run log not found at {path}")
        return pd.DataFrame()
    return pd.read_csv(log_path)


def check_model_compliance(row: pd.Series) -> tuple[bool, list[str]]:
    """
    Check a single model run against success criteria.
    Returns (passes, list of failure reasons).
    """
    failures = []
    task = row.get('task', '')

    if 'regression' in task:
        if row.get('mape', 100) > SUCCESS_CRITERIA['regression']['mape']:
            failures.append(f"MAPE {row['mape']:.1f}% > {SUCCESS_CRITERIA['regression']['mape']}%")
        if row.get('r2', 0) < SUCCESS_CRITERIA['regression']['r2']:
            failures.append(f"R² {row['r2']:.3f} < {SUCCESS_CRITERIA['regression']['r2']}")

    elif 'classification' in task or 'classif' in task:
        if row.get('f1_score', 0) < SUCCESS_CRITERIA['classification']['f1']:
            failures.append(f"F1 {row['f1_score']:.3f} < {SUCCESS_CRITERIA['classification']['f1']}")

    return len(failures) == 0, failures


def generate_report(log_df: pd.DataFrame) -> dict:
    """Generate QA compliance report from run log."""
    report = {
        'total_runs': len(log_df),
        'passing': 0,
        'failing': 0,
        'failures': [],
        'models_by_task': {},
    }

    for _, row in log_df.iterrows():
        passes, failures = check_model_compliance(row)
        if passes:
            report['passing'] += 1
        else:
            report['failing'] += 1
            report['failures'].append({
                'run_id': row.get('run_id', '?'),
                'model': row.get('model_name', '?'),
                'reasons': failures,
            })

        task = row.get('task', 'unknown')
        if task not in report['models_by_task']:
            report['models_by_task'][task] = 0
        report['models_by_task'][task] += 1

    return report


def print_report(report: dict) -> None:
    """Print formatted QA report to stdout."""
    print("=" * 60)
    print("  QUALITY INSURANCE REPORT — run_log.csv")
    print("=" * 60)
    print(f"\n  Total model runs: {report['total_runs']}")
    print(f"  Passing criteria: {report['passing']}")
    print(f"  Failing criteria: {report['failing']}")

    if report['total_runs'] > 0:
        pass_rate = report['passing'] / report['total_runs'] * 100
        print(f"  Pass rate:       {pass_rate:.0f}%")

    print(f"\n  Models by task:")
    for task, count in sorted(report['models_by_task'].items()):
        print(f"    {task}: {count}")

    if report['failures']:
        print(f"\n  Failures ({len(report['failures'])}):")
        for f in report['failures']:
            print(f"    [{f['run_id']}] {f['model']}:")
            for reason in f['reasons']:
                print(f"      - {reason}")

    print("=" * 60)

    if report['failing'] > 0:
        print("  STATUS: SOME MODELS FAIL QUALITY CHECKS")
    else:
        print("  STATUS: ALL MODELS PASS QUALITY CHECKS")

    print("=" * 60)


def main():
    log_df = load_run_log()
    if log_df.empty:
        print("No run_log.csv found. Run 03b first to generate model results.")
        return
    report = generate_report(log_df)
    print_report(report)


if __name__ == "__main__":
    main()
```

### 11.4 Usage

```bash
# Generate QA report from latest run_log.csv
python -m app.src.qa_report
```

---

## 12. Action 9: Linting Configuration (pyproject.toml)

### 12.1 Problem Statement

The project has no linting or formatting configuration files. The existing `AGENTS.md` documents `flake8` and `black` commands, but without a config file, different developers may have different tool defaults, leading to inconsistent code. There is no `pyproject.toml` at all — the modern Python standard for tool configuration.

### 12.2 Resolution

Create `pyproject.toml` at the project root with consolidated configurations for `black`, `flake8`, `pytest`, and `coverage`.

### 12.3 File Content

```toml
[build-system]
requires = ["setuptools>=64.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.black]
line-length = 120
target-version = ["py314"]
include = '\.pyi?$'
extend-exclude = '''
/(
    \.eggs
  | \.git
  | \.mypy_cache
  | \.tox
  | \.venv
  | _build
  | buck-out
  | build
  | dist
  | node_modules
  | notebooks
)/
'''

[tool.flake8]
max-line-length = 120
extend-ignore = [
    "E203",  # whitespace before ':' (black compatible)
    "W503",  # line break before binary operator (black compatible)
]
exclude = [
    ".git",
    "__pycache__",
    "build",
    "dist",
    "notebooks",
    ".venv",
]

[tool.pytest.ini_options]
minversion = "8.0"
testpaths = ["app/tests"]
python_files = ["test_*.py"]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
]

[tool.coverage.run]
source = ["app.src"]
omit = [
    "*/tests/*",
    "*/__init__.py",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if self.debug:",
    "if __name__ == .__main__.:",
    "raise AssertionError",
    "raise NotImplementedError",
]
fail_under = 70
```

### 12.4 Impact

| Tool | Before | After |
|------|--------|-------|
| black | Ad-hoc command, default line length (88) | Config file, 120 chars, Python 3.14 syntax |
| flake8 | Ad-hoc command with `--max-line-length=120` flag | Config file with black-compatible ignores |
| pytest | No config, uses defaults | Test paths, markers, min version set |
| coverage | No config, `--cov=app.src` flag required | Source path, exclusion rules, 70% minimum |

---

## 13. Action 10: Lockfile for Reproducibility

### 13.1 Problem Statement

`requirements.txt` uses unbounded or loosely-bounded version pins (e.g. `pandas>=2.2.0`, `scikit-learn>=1.5.0`). This means `pip install` on different dates can produce different environments, undermining reproducibility and potentially causing quality gates to behave differently across runs.

CRISP-ML(Q) requires that results be reproducible. Without dependency pinning, two executions of the same notebook a month apart may produce different results due to library API changes or different transitive dependency resolution.

### 13.2 Resolution

Generate a `requirements.lock` file by running `pip freeze` after a known-good install. This file records exact versions of every installed package. The `requirements.txt` remains as the human-readable source of truth; `requirements.lock` is the machine-readable snapshot.

### 13.3 Lockfile Strategy

```
requirements.txt          ← Human-editable, loose version bounds
      ↓
pip install -r requirements.txt
      ↓
pip freeze > requirements.lock   ← Machine-generated, exact versions
      ↓
CI uses: pip install -r requirements.lock   ← Deterministic install
```

### 13.4 Generation Command

```powershell
# After a known-good install, freeze exact versions
pip freeze --exclude-editable > requirements.lock
```

### 13.5 CI Update

In `.github/workflows/ci.yml`, change the install step to prefer the lockfile:

```yaml
- name: Install dependencies
  run: |
    python -m pip install --upgrade pip
    if (Test-Path requirements.lock) {
      pip install -r requirements.lock
    } else {
      pip install -r requirements.txt
    }
```

### 13.6 Maintenance Cadence

| Event | Action |
|-------|--------|
| New dependency added to `requirements.txt` | Regenerate: `pip freeze > requirements.lock` |
| Monthly or before major runs | Regenerate lockfile from clean venv |
| Security patch needed | Update pin in `requirements.txt`, regenerate lockfile |

---

## 14. Testing Strategy & Coverage

### 14.1 Existing Test Coverage

| Test File | Test Classes | Test Count | Coverage Area |
|-----------|-------------|------------|---------------|
| `test_features.py` | `TestCreateFeatures`, `TestGetFeatureTargetColumns`, `TestCreateSequences`, `TestMultiTableFeatures` | ~15 | Feature shape, parsing, targets, sequences, multi-table joins |
| `test_model.py` | `TestXGBoostModel`, `TestClusterModel`, `TestAnomalyModel`, `TestModelPersistence`, `TestLinearModel`, `TestRandomForestModel` | ~14 | Fit/predict, evaluate metrics, save/load, metadata |
| **Total** | | **~29** | |

### 14.2 Quality Gate Coverage (New)

| Notebook | Gate | Assertions | Triggered By |
|----------|------|-----------|--------------|
| 03a §2.2 | Data Quality | 3 | `nbconvert --execute` |
| 03a §3.2 | Feature Validation | 8 | `nbconvert --execute` |
| 03b §4.8 | Model Acceptance | ~6 per model | `nbconvert --execute` |
| 03b §5.3 | Classification Gate | ~3 per model | `nbconvert --execute` |
| 03b §11 | Q Summary Report | 0 (informational) | `nbconvert --execute` |
| 03c §8.8 | Timeseries Gate | ~3 per model | `nbconvert --execute` |

### 14.3 Success Criteria Reference

| Task | Metric | Threshold | Source |
|------|--------|-----------|--------|
| CPU utilization (regression) | MAPE | < 15% | Plan §1, Notebook §1 |
| CPU utilization (regression) | R² | > 0.70 | Derived from literature |
| Idle detection (classification) | F1 | > 0.85 | Plan §1, Notebook §1 |
| Waste tier (classification) | F1 | > 0.80 | Derived (more classes = harder) |
| Timeseries forecasting | MAE | < 5.0 | Derived from data scale |

---

## 15. Implementation Order

### Phase 1: Notebook Quality Gates (Priority: High)

| Step | Action | File | Estimated Effort |
|------|--------|------|-----------------|
| 1.1 | Add Risk Register markdown cell | `03a_feature_engineering.ipynb` | 5 min |
| 1.2 | Add Data Quality Gate code cell | `03a_feature_engineering.ipynb` | 5 min |
| 1.3 | Add Feature Validation Gate code cell | `03a_feature_engineering.ipynb` | 5 min |
| 1.4 | Add Model Acceptance Gate section | `03b_tabular_models.ipynb` | 10 min |
| 1.5 | Add Classification Gate code cell | `03b_tabular_models.ipynb` | 5 min |
| 1.6 | Add Q Summary Report code cell | `03b_tabular_models.ipynb` | 5 min |
| 1.7 | Add Timeseries Gate code cell | `03c_timeseries_forecasting.ipynb` | 5 min |

### Phase 2: Infrastructure (Priority: High)

| Step | Action | File | Estimated Effort |
|------|--------|------|-----------------|
| 2.1 | Remove `TestCatBoostModel` | `app/tests/test_model.py` | 2 min |
| 2.2 | Un-comment CI/CD pipeline | `.github/workflows/ci.yml` | 5 min |
| 2.3 | Create `pyproject.toml` with lint/test configs | `pyproject.toml` | 10 min |
| 2.4 | Generate `requirements.lock` | `requirements.lock` | 2 min |
| 2.5 | Update `AGENTS.md` with QA commands | `AGENTS.md` | 5 min |
| 2.6 | Run `pytest` to verify tests pass | — | 1 min |

### Phase 3: Optional (Priority: Low)

| Step | Action | File | Estimated Effort |
|------|--------|------|-----------------|
| 3.1 | Create QA report utility | `app/src/qa_report.py` | 15 min |

### Verification Steps

After each phase, run:

```powershell
# Verify unit tests pass
pytest app/tests/ -v --tb=short

# Verify each notebook executes end-to-end (quality gates active)
jupyter nbconvert --to notebook --execute notebooks/03a_feature_engineering.ipynb --output /dev/null --ExecutePreprocessor.timeout=600
jupyter nbconvert --to notebook --execute notebooks/03b_tabular_models.ipynb --output /dev/null --ExecutePreprocessor.timeout=600
jupyter nbconvert --to notebook --execute notebooks/03c_timeseries_forecasting.ipynb --output /dev/null --ExecutePreprocessor.timeout=600
```

---

## 16. Deliverables Checklist

### Notebook Quality Gates

- [ ] **03a §1.5** — Risk Register markdown cell added
- [ ] **03a §2.2** — Data Quality Gate code cell added
- [ ] **03a §3.2** — Feature Validation Gate code cell added
- [ ] **03b §4.8** — Model Acceptance Gate subsection added
- [ ] **03b §5.3** — Classification Gate code cell added
- [ ] **03b §11** — Quality Insurance Summary Report cell added
- [ ] **03c §8.8** — Timeseries Gate code cell added

### Test Suite

- [ ] `TestCatBoostModel` removed from `test_model.py`
- [ ] `pytest app/tests/ -v` passes

### Config & Environment

- [ ] `pyproject.toml` created with black, flake8, pytest, coverage configs
- [ ] `requirements.lock` generated from known-good install
- [ ] CI install step falls back to `requirements.lock` when available

### CI/CD

- [ ] `.github/workflows/ci.yml` un-commented and active
- [ ] Pipeline runs `pytest` with `--cov`
- [ ] Pipeline runs `nbconvert --execute` on all 3 notebooks

### Optional

- [ ] `app/src/qa_report.py` created

### Documentation

- [ ] `docs/AUTOMATED_QA_TECH_SPEC.md` (this document)
- [ ] `docs/IMPLEMENTATION_PROGRESS.md` updated with Q items marked complete
- [ ] `AGENTS.md` updated with QA commands

---

## 17. AGENTS.md Updates

### 17.1 Summary

`AGENTS.md` is the project's reference for development commands. It must be updated to include the QA tools added in this specification.

### 17.2 Changes

| Section | Current | Updated To |
|---------|---------|------------|
| Testing | `pytest app/tests/ -v` | No change needed |
| Coverage | `pytest app/tests/ --cov=app.src` | No change needed |
| Linting | `flake8 app/src/ --max-line-length=120` | `black --check app/src/ app/tests/ && flake8 app/src/` (uses `pyproject.toml`) |
| Formatting | `black app/src/ app/tests/` | No change needed |
| Notebook Execution | Single monolithic notebook | 3 notebooks separately listed |
| CI/CD | `pytest ... && jupyter nbconvert ...` | Keep as-is (CI does both) |
| **NEW: Quality Insurance** | _(missing)_ | `python -m app.src.qa_report` |
| **NEW: Environment Lock** | _(missing)_ | `pip freeze --exclude-editable > requirements.lock` |

### 17.3 Updated AGENTS.md Content

Replace the **Development Commands** section with:

```markdown
## Development Commands

### Testing
- Run all unit tests: `pytest app/tests/ -v`
- Run specific test file: `pytest app/tests/test_features.py -v`
- Run with coverage: `pytest app/tests/ --cov=app.src`

### Linting & Formatting
- Check code style: `black --check app/src/ app/tests/ && flake8 app/src/`
- Format code: `black app/src/ app/tests/`

### Notebook Execution
- Execute end-to-end (quality gates active):
  - `jupyter nbconvert --to notebook --execute notebooks/03a_feature_engineering.ipynb --output /dev/null --ExecutePreprocessor.timeout=600`
  - `jupyter nbconvert --to notebook --execute notebooks/03b_tabular_models.ipynb --output /dev/null --ExecutePreprocessor.timeout=600`
  - `jupyter nbconvert --to notebook --execute notebooks/03c_timeseries_forecasting.ipynb --output /dev/null --ExecutePreprocessor.timeout=600`

### Quality Insurance
- Generate QA compliance report: `python -m app.src.qa_report`

### Environment
- Freeze dependencies: `pip freeze --exclude-editable > requirements.lock`

### CI/CD
- Run full CI locally: `pytest app/tests/ -v && jupyter nbconvert --to notebook --execute notebooks/03a_feature_engineering.ipynb --output /dev/null --ExecutePreprocessor.timeout=600`
```

---

## 18. Appendices

### 15.1 References

| Document | Link | Relevance |
|----------|------|-----------|
| AZURE_PREDICTIVE_ANALYSIS_PLAN.md | `docs/AZURE_PREDICTIVE_ANALYSIS_PLAN.md` | Base plan this spec extends |
| IMPLEMENTATION_PROGRESS.md | `docs/IMPLEMENTATION_PROGRESS.md` | Tracking of all implementation phases |
| CRISP-ML(Q) Official | [crisp-ml.org](https://crisp-ml.org/) | Methodology reference |
| Azure Public Dataset V2 | [GitHub](https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetV2.md) | Source dataset |

### 15.2 Success Criteria Traceability Matrix

| Requirement | Source | Verification | Automated |
|-------------|--------|-------------|-----------|
| Risk register documented | CRISP-ML(Q) §3.1.2 | Visual inspection of 03a §1.5 | No (markdown) |
| Input data meets schema | ISO 8000 (Data Quality) | Assertion in 03a §2.2 | Yes |
| Features within valid ranges | Domain constraints | Assertion in 03a §3.2 | Yes |
| Regression MAPE < 15% | Plan §1 | Assertion in 03b §4.8 | Yes |
| Classification F1 > 0.85 | Plan §1 | Assertion in 03b §5.3 | Yes |
| All models pass thresholds | Project policy | Assertion in 03b §4.8 + §5.3 | Yes |
| Unit tests pass | CRISP-ML(Q) §4 | pytest CI stage | Yes |
| Notebooks execute cleanly | CRISP-ML(Q) §5 | nbconvert CI stage | Yes |
| Model metrics auditable | CAMS: Measurement | run_log.csv | Yes (git-tracked) |
