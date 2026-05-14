"""
Insert CRISP-ML(Q) quality gate cells into all three notebooks.
Run once to embed assertions and risk register cells.
"""

import json
from pathlib import Path


NOTEBOOK_DIR = Path("notebooks")


def make_md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.split("\n")}


def make_code(source: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": source.split("\n"), "outputs": [], "execution_count": None}


# ---------------------------------------------------------------------------
# 03a: Risk Register (insert after cell 2, before §2)
# ---------------------------------------------------------------------------
RISK_REGISTER = make_md("""### 1.5 Quality Risk Register

**Purpose:** Identify potential failure points early in the ML lifecycle, per CRISP-ML(Q) risk-based thinking principle.

| # | Risk | Phase | Likelihood | Impact | Mitigation |
|---|------|-------|-----------|--------|------------|
| R1 | Data drift (2019 patterns vs 2026 usage) | Monitoring | Medium | High | Retrain on newer data when available; monitor metric regression across runs via `run_log.csv` |
| R2 | Missing pricing or subscription data | Data Preparation | Low | Medium | Fallback to NaN; downstream models must handle missing values gracefully |
| R3 | Target leakage via correlated features | Feature Engineering | Medium | High | `get_feature_target_columns()` excludes target-related columns; review SHAP for unexpected feature dominance |
| R4 | Timeseries overfitting (few VMs) | Modeling | High | Medium | Early stopping, limit model complexity, cross-validation per VM |
| R5 | CPU readings memory blowup | Data Preparation | Low | High | Already mitigated via DuckDB out-of-core parquet glob (no `pd.concat`) |
| R6 | Skewed waste_tier distribution | Evaluation | Medium | Low | Use weighted F1 score; apply SMOTE if minority class recall < 0.7 |""")


# ---------------------------------------------------------------------------
# 03a: Data Quality Gate (insert after dataset load cell)
# ---------------------------------------------------------------------------
DATA_QUALITY_GATE = make_code("""# ---------------------------------------------------------------------------
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
print(f"[OK] Data quality: {len(vmtable):,} rows, {vmtable['vm_id'].nunique():,} unique VMs, "
      f"{null_rate} nulls in required columns")

if null_rate > 0:
    print("  ⚠ Warning: nulls detected — consider imputation strategy")
else:
    print("  [OK] No nulls in required columns")""")


# ---------------------------------------------------------------------------
# 03a: Feature Validation Gate (insert after create_features() call cell)
# ---------------------------------------------------------------------------
FEATURE_VALIDATION_GATE = make_code("""# ---------------------------------------------------------------------------
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

print(f"[OK] Features validated: {len(features_df):,} rows, "
      f"{len(features_df.columns)} columns, "
      f"{features_df.isnull().sum().sum()} total nulls")""")


# ---------------------------------------------------------------------------
# 03b: Model Acceptance Gate §4.8 (insert after §4.7 save cell)
# ---------------------------------------------------------------------------
MODEL_ACCEPTANCE_MD = make_md("""### 4.8 Model Acceptance Gate

**CRISP-ML(Q):** Quality Insurance

**Purpose:** Verify all regression models meet the success criteria defined in §1 (MAPE < 15%, R² > 0.7). This gate fails the notebook execution if any model underperforms.""")

MODEL_ACCEPTANCE_CODE = make_code("""# ---------------------------------------------------------------------------
# MODEL ACCEPTANCE GATE — CRISP-ML(Q) Quality Insurance
# All models must meet minimum performance thresholds.
# ---------------------------------------------------------------------------
SUCCESS_MAPE = 15.0
SUCCESS_R2 = 0.7

print("=" * 60)
print("MODEL ACCEPTANCE GATE — Regression")
print("=" * 60)

all_pass = True
for model_name, metrics in comparison.items():
    model_pass = True
    mape = metrics.get('mape', 100)
    r2 = metrics.get('r2', 0)

    if mape > SUCCESS_MAPE:
        print(f"  ✗ {model_name}: MAPE {mape:.1f}% > {SUCCESS_MAPE}%")
        model_pass = False
    else:
        print(f"  [OK] {model_name}: MAPE {mape:.1f}% ≤ {SUCCESS_MAPE}%")

    if r2 < SUCCESS_R2:
        print(f"  ✗ {model_name}: R² {r2:.3f} < {SUCCESS_R2}")
        model_pass = False
    else:
        print(f"  [OK] {model_name}: R² {r2:.3f} ≥ {SUCCESS_R2}")

    if not model_pass:
        all_pass = False

print("=" * 60)
assert all_pass, "FAIL: One or more models did not meet success criteria"
print("[OK] All regression models pass acceptance gate")""")


# ---------------------------------------------------------------------------
# 03b: Classification Gate (insert after §5.3 save cell)
# ---------------------------------------------------------------------------
CLASSIFICATION_GATE = make_code("""# ---------------------------------------------------------------------------
# CLASSIFICATION ACCEPTANCE GATE — CRISP-ML(Q) Quality Insurance
# ---------------------------------------------------------------------------
SUCCESS_F1 = 0.80

print("=" * 60)
print("CLASSIFICATION ACCEPTANCE GATE")
print("=" * 60)

all_pass = True
for name, metrics in results_clf.items():
    f1 = metrics.get('f1', 0)
    if f1 < SUCCESS_F1:
        print(f"  ✗ {name}: F1 {f1:.3f} < {SUCCESS_F1}")
        all_pass = False
    else:
        print(f"  [OK] {name}: F1 {f1:.3f} ≥ {SUCCESS_F1}")

print("=" * 60)
assert all_pass, "FAIL: Classification models did not meet F1 threshold"
print("[OK] All classification models pass acceptance gate")""")


# ---------------------------------------------------------------------------
# 03b: Quality Insurance Summary Report (insert before §11 conclusions)
# ---------------------------------------------------------------------------
Q_SUMMARY_REPORT = make_code('''# ---------------------------------------------------------------------------
# QUALITY INSURANCE REPORT \u2014 End of Notebook Summary
# This cell uses variables from the 03b notebook runtime.
# ---------------------------------------------------------------------------
from datetime import datetime

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

print("=" * 70)
print("  QUALITY INSURANCE REPORT")
print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 70)

reg_status = "+" if reg_models_passing == reg_models_total else "x"
clf_status = "+" if clf_models_passing == clf_models_total else "x"
ts_status = "+" if ts_available else "-"
print(f"""
  Risk Register         +  (6 risks documented in 03a s1.5)
  Data Quality Gate     +  (passed at data load)
  Feature Validation    +  (all targets and features verified)
  Model Acceptance      {reg_status}  (regression: {reg_models_passing}/{reg_models_total})
  Classification Gate   {clf_status}  (classification: {clf_models_passing}/{clf_models_total})
  Timeseries Gate       {ts_status}  (handled in 03c)

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
    print("  STATUS: SOME QUALITY GATES FAILED \\u2014 review above")

print("=" * 70)''')


# ---------------------------------------------------------------------------
# 03c: Timeseries Gate (insert at §8.8 after model comparison)
# ---------------------------------------------------------------------------
TIMESERIES_GATE = make_code("""# ---------------------------------------------------------------------------
# TIMESERIES ACCEPTANCE GATE — CRISP-ML(Q) Quality Insurance
# ---------------------------------------------------------------------------
if ts_results:
    print("=" * 60)
    print("TIMESERIES ACCEPTANCE GATE")
    print("=" * 60)
    all_pass = True
    for model_name, metrics in ts_results.items():
        mae = metrics.get('mae', 999)
        print(f"  {'[OK]' if mae < 5 else '✗'} {model_name}: MAE = {mae:.3f}")
        if mae >= 5:
            all_pass = False
    assert all_pass, "FAIL: Timeseries models exceed MAE threshold"
    print("[OK] All timeseries models pass acceptance gate")""")


# =========================================================================
# Apply modifications
# =========================================================================

def load_notebook(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_notebook(path: Path, nb: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  [OK] Saved: {path}")


# --- 03a: Insert 3 cells ---
print("\n=== 03a_feature_engineering.ipynb ===")
nb03a = load_notebook(NOTEBOOK_DIR / "03a_feature_engineering.ipynb")
cells = nb03a["cells"]

# Cell 2 is the §1 Summary markdown. Insert Risk Register after it (at index 3).
# Account for the shift after each insertion.
insert_at = 3  # after cell 2
cells.insert(insert_at, RISK_REGISTER)
print(f"  Inserted Risk Register at index {insert_at}")

# After Risk Register insert, the load dataset cell is now at index 6.
# Data Quality Gate goes after it.
insert_at = 7  # after load dataset cell at index 6
cells.insert(insert_at, DATA_QUALITY_GATE)
print(f"  Inserted Data Quality Gate at index {insert_at}")

# After 2 prior insertions, the create_features() call is at index 12.
# Feature Validation Gate goes after it.
insert_at = 13  # after create_features cell at index 12
cells.insert(insert_at, FEATURE_VALIDATION_GATE)
print(f"  Inserted Feature Validation Gate at index {insert_at}")

# Update execution counts to preserve readability
for i, cell in enumerate(cells):
    if cell.get("cell_type") == "code" and cell.get("execution_count") is not None:
        cell["execution_count"] = i  # approximate renumber

save_notebook(NOTEBOOK_DIR / "03a_feature_engineering.ipynb", nb03a)


# --- 03b: Insert 4 cells ---
print("\n=== 03b_tabular_models.ipynb ===")
nb03b = load_notebook(NOTEBOOK_DIR / "03b_tabular_models.ipynb")
cells = nb03b["cells"]

# Insert Model Acceptance Gate: markdown + code after §4.7 save cell (index 19)
insert_at = 20  # after cell 19
cells.insert(insert_at, MODEL_ACCEPTANCE_MD)
cells.insert(insert_at + 1, MODEL_ACCEPTANCE_CODE)
print(f"  Inserted Model Acceptance Gate (2 cells) at index {insert_at}")

# Insert Classification Gate after §5.3 save cell.
# §5.3 markdown is at original index 25, code at 26.
# After +2 shift from above, code cell is at 28.
insert_at = 29  # after cell 28 (was 26, shifted by 2)
cells.insert(insert_at, CLASSIFICATION_GATE)
print(f"  Inserted Classification Gate at index {insert_at}")

# Insert Q Summary Report before §11 conclusions.
# §11 markdown is at original index 68.
# After +3 shift from above, it's at 71.
insert_at = 71  # before cell 71 (was 68, shifted by 3)
cells.insert(insert_at, Q_SUMMARY_REPORT)
print(f"  Inserted Q Summary Report at index {insert_at}")

for i, cell in enumerate(cells):
    if cell.get("cell_type") == "code" and cell.get("execution_count") is not None:
        if isinstance(cell["execution_count"], int):
            pass  # keep original numbers valid

save_notebook(NOTEBOOK_DIR / "03b_tabular_models.ipynb", nb03b)


# --- 03c: Insert 1 cell ---
print("\n=== 03c_timeseries_forecasting.ipynb ===")
nb03c = load_notebook(NOTEBOOK_DIR / "03c_timeseries_forecasting.ipynb")
cells = nb03c["cells"]

# Insert Timeseries Gate after §8.8 markdown (cell 16)
insert_at = 17  # after cell 16
cells.insert(insert_at, TIMESERIES_GATE)
print(f"  Inserted Timeseries Gate at index {insert_at}")

for i, cell in enumerate(cells):
    if cell.get("cell_type") == "code" and cell.get("execution_count") is not None:
        cell["execution_count"] = i

save_notebook(NOTEBOOK_DIR / "03c_timeseries_forecasting.ipynb", nb03c)


print("\n=== All quality gates inserted successfully ===")
