# Azure Predictive Analysis Plan — 03a/03b/03c

**Dataset:** Azure Public Dataset V2 (2019 VM traces)
**Dataset Source:** https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetV2.md
**Authors:** Fajar Laksono
**Status:** Planning
**Last Updated:** 2026-05-12

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [CRISP-ML(Q) & CAMS DevOps Integration](#3-crisp-mlq--cams-devops-integration)
4. [Notebook Structure: 3-Notebook Split](#4-notebook-structure-3-notebook-split)
5. [Thin Import Modules: app/src/](#5-thin-import-modules-appsrc)
6. [Testing: app/tests/](#6-testing-apptests)
7. [CI/CD: GitHub Actions](#7-cicd-github-actions)
8. [Dependencies: requirements.txt](#8-dependencies-requirementstxt)
9. [Model Persistence: models/](#9-model-persistence-models)
10. [Git Strategy](#10-git-strategy)
11. [Implementation Order](#11-implementation-order)

---

## 1. Executive Summary

### 1.1 Purpose

This document defines the complete plan for building the predictive modeling phase of the Sembada Cloud project — split across three focused notebooks (`03a_feature_engineering.ipynb`, `03b_tabular_models.ipynb`, `03c_timeseries_forecasting.ipynb`). Together they perform ML-based prediction of cloud resource utilization, waste detection, cost anomaly identification, and workload segmentation on the Azure Public Dataset V2.

### 1.2 Methodology

The project follows **CRISP-ML(Q)** as the lifecycle framework and **CAMS DevOps** for operational practices:

| Framework | Role | Implementation |
|---|---|---|
| **CRISP-ML(Q)** | Defines phases: Business Understanding → Data Preparation → Modeling → Evaluation → Deployment → Monitoring | Each notebook section maps to a phase |
| **CAMS DevOps** | Defines practices: Culture, Automation, Measurement, Sharing | Version control, CI/CD, metrics tracking, committed artifacts |

### 1.3 Literature Basis

Models selected based on literature review (see `docs/Recommended_models_from_literature.md`):

| Category | Literature Recommendation | Implementation |
|---|---|---|
| Tabular regression/classification | XGBoost, Random Forest | §4–5 |
| Time series forecasting | LSTM, BiGRU, CNN-LSTM, TFT | §8 |
| Workload segmentation | K-Means | §6 |
| Anomaly detection | Isolation Forest, WHA | §7 |
| Explainability | SHAP, Integrated Gradients | §9 |

### 1.4 Design Principle

**Notebook-first, thin-imports pattern.** The notebook is the primary academic deliverable. It imports reusable logic from `app/src/` modules (features.py, models.py, visualize.py) rather than duplicating code. This avoids dual-maintenance while keeping logic testable.

To manage memory, iteration speed, and debuggability, the work is split into three notebooks that share a common feature artifact (`features_df.parquet`).

---

## 2. Architecture Overview

```
sembada-cloud/
├── notebooks/
│   ├── 03a_feature_engineering.ipynb         ← §1–§3: Load all tables, build features, save artifact
│   ├── 03b_tabular_models.ipynb              ← §4–§7, §9–§11: Train/evaluate tabular models, SHAP
│   └── 03c_timeseries_forecasting.ipynb      ← §8 only: Load CPU readings (DuckDB), train LSTM/GRU
│
├── app/src/
│   ├── __init__.py                           ← Package marker (existing, unmodified)
│   ├── features.py                           ← Feature engineering functions (~350 lines)
│   ├── models.py                             ← Model wrappers (fit, predict, save) (~390 lines)
│   └── visualize.py                          ← Publication-quality figure functions (~200 lines)
│
├── app/tests/
│   ├── __init__.py
│   ├── conftest.py                           ← Shared fixtures (sample data, 8 fixtures)
│   ├── test_features.py                      ← 15 tests: shape, parsing, encodings, targets, sequences, multi-table
│   └── test_model.py                         ← 4 tests: fit/predict, evaluate, cluster, save/load
│
├── models/
│   ├── .gitkeep
│   ├── regression/                           ← Best regressor(s)
│   ├── classification/                       ← Best classifier(s)
│   ├── clustering/                           ← K-Means + scaler
│   └── timeseries/                           ← Best LSTM/GRU model
│
├── .github/workflows/
│   └── ci.yml                                ← pytest + nbconvert --execute on every PR
│
├── docs/
│   ├── AZURE_PREDICTIVE_ANALYSIS_PLAN.md     ← This document
│   └── Recommended_models_from_literature.md ← Literature survey
│
├── requirements.txt                          ← Updated with ML stack
├── .gitignore                                ← Updated with models/ pattern
└── AGENTS.md                                 ← Updated with CI/train commands
```

### 2.1 Data Flow

```
                               ┌──────────────────────────────────────────────────────────┐
                               │  03a_feature_engineering.ipynb                           │
                               │                                                          │
data/transformed/parquet/      │  ┌─────────────────────────────────────────────────┐    │
  vmtable.parquet ──────────────► │ DuckDB view → pandas                            │    │
  subscriptions.parquet ────────► │ DuckDB view → pandas merge on subscription_id  │    │
  deployments.parquet ──────────► │ DuckDB view → pandas merge on deployment_id    │    │
  azure_pricing.parquet ────────► │ DuckDB view → dict lookup (core, mem bucket)   │    │
                                │  └─────────────────────────────────────────────────┘    │
                                │                         │                              │
                                │                         ▼                              │
                                │              create_features(vmtable, pricing,          │
                                │                subscriptions, deployments)              │
                                │                         │                              │
                                │                         ▼                              │
                                │              features_df.parquet ───────► saved to disk │
                                └─────────────────────────────────────────────────────────┘
                                                          │
                                                          │
                    ┌─────────────────────────────────────┼─────────────────────────────┐
                    │                                     │                             │
                    ▼                                     │                             ▼
  ┌───────────────────────────────┐                      │    ┌────────────────────────────────────┐
  │ 03b_tabular_models.ipynb     │                      │    │ 03c_timeseries_forecasting.ipynb   │
  │                              │                      │    │                                    │
  │ Load features_df.parquet ────┤                      │    │ DuckDB → GROUP BY vm_id             │
  │                              │                      │    │    → filter top-5 VMs               │
  │ Ridge / RF / XGBoost         │                      │    │    → per-VM sequences               │
  │ K-Means / Isolation Forest   │                      │    │                                    │
  │ SHAP / Comparison            │                      │    │ ARIMA / LSTM / BiGRU / CNN-LSTM     │
  └───────────────────────────────┘                     │    └────────────────────────────────────┘
                                                        │
                                              cpu_readings/*.parquet
                                              (loaded via DuckDB parquet glob — no pd.concat)
```

### 2.2 Import Convention

The notebook imports from `app.src` as a thin layer:

```python
from app.src.features import create_features, create_sequences, load_cpu_readings
from app.src.models import XGBoostModel, RandomForestModel, ClusterModel, AnomalyModel
from app.src.visualize import residual_plot, feature_importance_plot, cluster_scatter, comparison_table
```

---

## 3. CRISP-ML(Q) & CAMS DevOps Integration

### 3.1 CRISP-ML(Q) Phase → Notebook Section Map

| CRISP-ML(Q) Phase | Notebook Section | Key Deliverable |
|---|---|---|
| **Business Understanding** | §1 Summary, each subsection's **Business Question** | Documented business goals, success criteria | `03a` / `03b` |
| **Data Understanding** | §2 Preparation, §3.3 Correlation Analysis | Dataset statistics, feature-target relationships | `03a` |
| **Data Preparation** | §3 Feature Engineering | `features.py`, engineered DataFrame + saved `.parquet` | `03a` |
| **Modeling** | §4 Regression, §5 Classification, §6 Clustering, §7 Anomaly, §8 Deep Learning | Trained models in `models/` | `03b` / `03c` |
| **Evaluation** | §4.4–4.6, §5.1, §6.2, §9 SHAP, §10 Comparison | Metrics table, SHAP analysis, best model selection | `03b` |
| **Deployment** | §11 Conclusions & Recommendations | Business impact report, model card | `03b` |
| **Monitoring** | §11.3–11.4 Limitations & Future Work | Identified gaps, improvement roadmap | `03b` / `03c` |

### 3.2 CAMS DevOps Practice → Implementation Map

| CAMS | Practice | Implementation |
|---|---|---|
| **Culture** | Version control, code review, peer review | Notebook + `.py` in git; PRs required for `main` |
| **Automation** | CI/CD pipeline | GitHub Actions: run `pytest` + `nbconvert --execute` |
| **Measurement** | Metrics tracked and auditable | `docs/run_log.csv` records model name, date, metrics, file hash |
| **Sharing** | Artifacts committed and accessible | Notebook, docs, model metadata committed to repo |

### 3.3 Traceability: run_log.csv

Each model training run records:

```csv
run_id,timestamp,task,model_name,mae,rmse,r2,f1_score,model_path,git_hash
001,2026-05-12T14:00:00,regression_avg_cpu,xgboost,2.34,4.56,0.82,,models/regression/xgboost_avg_cpu.pkl,a1b2c3d
002,2026-05-12T14:30:00,classification_idle,xgboost,,,,0.94,models/classification/xgboost_idle.pkl,a1b2c3d
```

Generated by `03b` §10.1 and committed to repo for auditability.

### 3.4 Compliance Note: Why a 3-Notebook Split

The original plan specified a single monolithic notebook. Production use revealed this approach creates real problems — memory bloat (cpu_readings crashes), slow iteration (20-min MI cell), and fragile error recovery.

The 3-notebook split **improves** CRISP-ML(Q) and CAMS compliance:

| Principle | Single notebook | 3-notebook split |
|---|---|---|
| **CRISP-ML(Q): Phase separation** | Blended — all phases in one context | Explicit artifact handoffs (`features_df.parquet`) |
| **CAMS: Automation** | 40+ min CI feedback | 10–15 min per notebook, easier to pinpoint failures |
| **CAMS: Culture** | 109-cell PRs hard to review | ~40-cell PRs per notebook, focused context |
| **Memory safety** | cpu_readings `pd.concat` crashes at 5.6 GiB | Each notebook loads only what it needs |

All CRISP-ML(Q) phases and CAMS practices are preserved — the split merely enforces cleaner boundaries.

---

## 4. Notebook Structure: 3-Notebook Split

### 4.1 Notebook Architecture Overview

| Notebook | Sections | Scope | Runtime (est.) |
|---|---|---|---|
| `03a_feature_engineering.ipynb` | §1–§3 | Load all 5 tables, run `create_features()`, save `features_df.parquet` | 20–30 min |
| `03b_tabular_models.ipynb` | §4–§7, §9–§11 | Load pre-computed features, train/evaluate Ridge/RF/XGBoost + K-Means + Isolation Forest + SHAP | 10–15 min |
| `03c_timeseries_forecasting.ipynb` | §8 only | Load CPU readings (DuckDB out-of-core), train ARIMA/LSTM/BiGRU/CNN-LSTM | 5–10 min |

**Artifact handoff:** `03a` saves `features_df.parquet` to disk. `03b` and `03c` load it independently — no shared memory.

### 4.2 Design Conventions (shared across all 3 notebooks)

Same as `02_azure_descriptive_analysis.ipynb`:

- **Markdown cells** at each subsection header containing **Business Question** and **Analysis Approach**
- **Code cells** with executable, commented code using DuckDB + pandas + `app.src` imports
- **"Key Findings"** markdown cells after each code block summarizing results
- `display()` for DataFrames, formatted `print()` for text output
- `matplotlib` + `seaborn` for all visualizations
- Seeds set for reproducibility: `random_state=42`, `np.random.seed(42)`

### 4.3 03a_feature_engineering.ipynb (§1–§3)

**Purpose:** Load all 5 Azure tables, engineer features, save `features_df.parquet` as artifact for downstream notebooks.

---

### §0. Table of Contents

Auto-generated markdown table of contents for navigation.

---

### §1. Summary

**CRISP-ML(Q):** Business Understanding

Markdown summary of:
- Dataset used (Azure Public Dataset V2, 2.7M VMs)
- Business goal: Predict resource waste, detect idle VMs, forecast CPU usage
- Success criteria: MAPE < 15% for regression, F1 > 0.85 for classification
- Key finding preview

---

### §2. Preparation

**CRISP-ML(Q):** Data Understanding

#### 2.1. Import Libraries

```python
import os, sys, warnings, pathlib
import duckdb, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid", palette="muted")
warnings.filterwarnings("ignore")

# Thin-imports
from app.src.features import create_features, get_feature_target_columns, create_sequences, load_cpu_readings
from app.src.visualize import residual_plot, feature_importance_plot, cluster_scatter, comparison_table

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, classification_report, confusion_matrix)
from sklearn.pipeline import Pipeline

# Statistical
from scipy import stats

# Progress
from tqdm import tqdm

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
```

#### 2.2. Load Dataset

```python
DATA_DIR = pathlib.Path("data/transformed/parquet")

con = duckdb.connect(":memory:")

# Register views (matching notebook 02 conventions)
for tbl in ["vmtable", "subscriptions", "deployments", "pricing"]:
    path = DATA_DIR / f"{tbl}.parquet"
    if path.exists():
        con.execute(f"CREATE VIEW {tbl} AS SELECT * FROM read_parquet('{path}')")
        print(f"  ✓ {tbl}: {con.execute(f'SELECT COUNT(*) FROM {tbl}').fetchone()[0]:,} rows")

# Load main vmtable
vmtable = con.execute("SELECT * FROM vmtable").fetchdf()
print(f"✓ vmtable loaded: {len(vmtable):,} rows, {len(vmtable.columns)} columns")

# Load pricing for rate_per_hour lookup
pricing_df = None
if (DATA_DIR / "azure_pricing.parquet").exists():
    pricing_df = con.execute("SELECT * FROM pricing").fetchdf()
    print(f"✓ pricing_df loaded: {len(pricing_df):,} rows")

# Load subscriptions for subscription-level features
subscriptions_df = None
if (DATA_DIR / "subscriptions.parquet").exists():
    subscriptions_df = con.execute("SELECT * FROM subscriptions").fetchdf()
    print(f"✓ subscriptions_df loaded: {len(subscriptions_df):,} rows")

# Load deployments for deployment-level features
deployments_df = None
if (DATA_DIR / "deployments.parquet").exists():
    deployments_df = con.execute("SELECT * FROM deployments").fetchdf()
    print(f"✓ deployments_df loaded: {len(deployments_df):,} rows")

# Quick preview
display(vmtable.head(3))
```

Note: CPU readings are NOT loaded here — they're handled separately in `03c_timeseries_forecasting.ipynb` via DuckDB out-of-core to avoid memory issues.

---

### §3. Feature Engineering

**CRISP-ML(Q):** Data Preparation

**Thin import:** `app.src.features`

#### 3.1. Target Variable Definition

| Task | Type | Target | Business Goal |
|---|---|---|---|
| Utilization regression | Regression | `avg_cpu` | Predict actual CPU usage |
| Waste regression | Regression | `waste_fraction` = 1 - (avg_cpu / 100) | Quantify waste per VM |
| Cost regression | Regression | `vm_cost` = rate_per_hour × lifetime | Forecast cloud spend |
| Idle detection | Binary classification | `is_idle` = 1 if avg_cpu < 5% | Flag wasted VMs |
| Waste tier | Multi-class | `waste_tier`: Low (<10%), Medium (10–50%), High (>50%) | Prioritize optimization |

#### 3.2. Feature Construction

```python
def create_features(
    df: pd.DataFrame,
    pricing_df: pd.DataFrame | None = None,
    subscriptions_df: pd.DataFrame | None = None,
    deployments_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
```

| Feature | Type | Derivation | Source Table |
|---------|------|-----------|--------------|
| `core_count` | numeric | Parse `vm_core_count_bucket`: 2→2, 4→4, 8→8, 24→24, >24→48 | `vmtable` |
| `memory_gb` | numeric | Parse `vm_memory_gb_bucket`: 2→2, 4→4, 8→8, 32→32, 64→64, >64→128 | `vmtable` |
| `lifetime_hours` | numeric | `(timestamp_deleted - timestamp_created) / 3600` | `vmtable` |
| `creation_hour` | cyclic (sin/cos) | `hour = timestamp_to_hour(timestamp_created)`; `sin = sin(2πh/24)`, `cos = cos(2πh/24)` | `vmtable` |
| `creation_dayofweek` | cyclic (sin/cos) | Same cyclical encoding for day of week | `vmtable` |
| `cpu_per_core` | ratio | `avg_cpu / core_count` | `vmtable` |
| `memory_per_core` | ratio | `memory_gb / core_count` | `vmtable` |
| `burstiness` | ratio | `p95_max_cpu / (avg_cpu + 1e-6)` | `vmtable` |
| `max_to_avg_ratio` | ratio | `max_cpu / (avg_cpu + 1e-6)` | `vmtable` |
| `is_short_lived` | binary | `lifetime_hours < 1` | `vmtable` |
| `vm_category_*` | one-hot | `pd.get_dummies(df['vm_category'], prefix='cat')` | `vmtable` |
| `core_bucket_*` | one-hot | `pd.get_dummies(df['vm_core_count_bucket'], prefix='core')` | `vmtable` |
| `mem_bucket_*` | one-hot | `pd.get_dummies(df['vm_memory_gb_bucket'], prefix='mem')` | `vmtable` |
| `rate_per_hour` | numeric | Dict lookup: `pricing_df[(core_bucket, mem_bucket)] → rate_per_hour` | `azure_pricing` |
| `vm_cost` | numeric | `rate_per_hour × lifetime_hours` | computed |
| `waste_cost` | numeric | `vm_cost × waste_fraction` | computed |
| `sub_vm_count` | numeric | `subscriptions_df['vm_count']` joined on `subscription_id` | `subscriptions` |
| `sub_first_vm_ts` | numeric | `subscriptions_df['first_vm_timestamp']` joined on `subscription_id` | `subscriptions` |
| `sub_tenure` | numeric | `timestamp_created - sub_first_vm_ts` (age of subscription when VM created) | computed |
| `deployment_size` | numeric | `deployments_df['deployment_size']` joined on `deployment_id` | `deployments` |

#### 3.3. Feature-Target Correlation & Mutual Information

**Business Question:** Which features have the strongest predictive relationship with each target?

**Approach:**
- Sample to 100K rows for performance (mutual_info_regression scales poorly above this)
- Pearson correlation matrix (all features × all targets)
- Mutual information scores with per-feature tqdm progress bar
- Top-10 features bar chart for each target

```python
# Sample for correlation/MI speed
SAMPLE_SIZE = 100_000
mi_sample = numeric_df.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)
target_sample = features_df.loc[mi_sample.index, 'avg_cpu']

print("Computing Pearson correlation...")
corr_with_cpu = mi_sample.corrwith(target_sample, method='pearson').abs()

print("Computing Mutual Information...")
for col in tqdm(top_cpu_features, desc="MI per feature"):
    score = mutual_info_regression(mi_sample[[col]].fillna(0), target_sample, random_state=RANDOM_STATE)
    mi_scores.append(score[0])
```

**Output:** Correlation heatmap + MI bar chart + written findings.

#### 3.4. Train-Test Split

**Business Question:** How do we ensure the model generalizes to unseen VMs?

**Approach:**
- Stratified 80/20 split by `waste_tier` to preserve class balance
- `random_state=42` for reproducibility

**Save artifact:** `features_df.parquet` written to disk for `03b` and `03c` to consume.

```python
features_df.to_parquet(DATA_DIR / "features_df.parquet")
print(f"Features saved: {DATA_DIR / 'features_df.parquet'}")
```

---

### 4.4 03b_tabular_models.ipynb (§4–§7, §9–§11)

**Purpose:** Load pre-computed features, train all tabular models, run SHAP explainability, select best model.

**Imports differ from 03a:** adds `xgboost`, `joblib`, `shap`; no `duckdb` needed.

```python
import os, sys, warnings, pathlib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid", palette="muted")
warnings.filterwarnings("ignore")

from app.src.features import get_feature_target_columns
from app.src.models import (XGBoostModel, RandomForestModel,
                            RidgeModel, ClusterModel, AnomalyModel, load_model)
from app.src.visualize import residual_plot, feature_importance_plot, cluster_scatter, comparison_table

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, classification_report, confusion_matrix)
import xgboost as xgb
import joblib
import shap
from scipy import stats

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Load pre-computed features
DATA_DIR = pathlib.Path("data/transformed/parquet")
features_df = pd.read_parquet(DATA_DIR / "features_df.parquet")
```

---

### §4. Baseline Regression Models — Cost & Utilization Prediction

**CRISP-ML(Q):** Modeling / Evaluation

**Thin import:** `app.src.models`

#### 4.1. Ridge Regression

**Business Question:** How well does a regularized linear model predict CPU utilization and waste?

**Approach:**
- `Ridge(alpha=...)` with GridSearchCV
- MAE, RMSE, R², MAPE on test set

#### 4.2. Random Forest Regressor

**Business Question:** Can an ensemble of decision trees capture non-linear resource patterns?

**Approach:**
- `RandomForestRegressor(n_estimators=100, max_depth=12, ...)`
- Built-in feature importance extraction

#### 4.3. XGBoost Regressor

**Business Question:** Does gradient boosting outperform bagging for cloud resource prediction?

**Approach:**
- `XGBRegressor(learning_rate=0.05, max_depth=6, ...)`
- Early stopping on validation set
- Feature importance extraction

#### 4.4. Model Evaluation Comparison

| Model | MAE (avg_cpu) | RMSE (avg_cpu) | R² (avg_cpu) | MAPE | Training Time |
|---|---|---|---|---|---|
| Ridge | | | | | |
| Random Forest | | | | | |
| XGBoost | | | | | |

**Output:** Side-by-side table with best values highlighted. Repeated for each target (`avg_cpu`, `waste_fraction`, `vm_cost`).

#### 4.5. Feature Importance Analysis

- Built-in importance (RF, XGBoost)
- Top-15 features bar chart
- Comparison of importance across models

#### 4.6. Residual Analysis

**Business Question:** Are model assumptions met? Where does the model fail?

**Approach:**
- Residual vs. predicted scatter plot
- Q-Q plot for normality
- Heteroscedasticity check
- Outlier identification (residuals > 3σ)

**Code:**
```python
fig = residual_plot(y_test, y_pred, title="XGBoost Residuals: avg_cpu")
```

#### 4.7. Save Best Model

Save best regressor per target:

```python
import joblib
best_model = xgboost_model.estimator
joblib.dump(best_model, "models/regression/xgboost_avg_cpu.pkl")
```

---

### §5. Classification — Waste Detection

**CRISP-ML(Q):** Modeling / Evaluation

**Thin import:** `app.src.models`

#### 5.1. Binary: Idle VM Detection

**Business Question:** Can we accurately identify idle VMs (avg_cpu < 5%) from metadata alone?

**Models:**
- 5.1.1. **Random Forest Classifier** — non-linear ensemble baseline
- 5.1.2. **XGBoost Classifier** — gradient boosting (primary)

**Evaluation:**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | | | | | |
| XGBoost | | | | | |

- Confusion matrix for best model
- ROC curve

#### 5.2. Multi-Class: Waste Tier Classification

**Business Question:** Can we classify VMs into waste tiers (Low/Medium/High) for optimization prioritization?

**Approach:**
- Target: `waste_tier` with 3 ordered classes
- Models: XGBoost, Random Forest
- `class_weight='balanced'` for imbalance

**Output:** Per-class precision/recall, macro F1, weighted F1.

#### 5.3. Save Best Classifier

```python
joblib.dump(best_clf, "models/classification/xgboost_idle.pkl")
```

---

### §6. Clustering — Unsupervised Workload Segmentation

**CRISP-ML(Q):** Modeling / Evaluation

**Literature basis:** K-Means recommended for workload segmentation.

**Thin import:** `app.src.models`, `app.src.visualize`

#### 6.1. K-Means Clustering

**Business Question:** Can unsupervised learning reveal natural workload patterns in VM resource usage?

**Features:** `avg_cpu`, `max_cpu`, `p95_max_cpu`, `core_count`, `memory_gb`, `burstiness`, `lifetime_hours`

**Preprocessing:** StandardScaler

```python
from app.src.models import ClusterModel
cluster = ClusterModel(n_clusters=4)
cluster.fit(X_scaled)
labels = cluster.predict(X_scaled)
```

#### 6.2. Optimal K Selection

- Elbow method (inertia vs. K for K=2..10)
- Silhouette score
- Select K with best trade-off

#### 6.3. Cluster Characterization

| Cluster | Size | Avg CPU | Avg Waste | % Idle | Dominant Category | Label |
|---|---|---|---|---|---|---|
| 0 | | | | | | "Ephemeral small" |
| 1 | | | | | | "Long-running medium" |
| ... | | | | | | |

#### 6.4. PCA / t-SNE Visualization

```python
from app.src.visualize import cluster_scatter
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
fig = cluster_scatter(X_pca, labels, title="K-Means Clusters (PCA)")
```

#### 6.5. Cluster-Category Cross-Tabulation

```python
pd.crosstab(labels, vmtable['vm_category'])
```

#### 6.6. Save Cluster Model

```python
joblib.dump({"kmeans": cluster.kmeans, "scaler": scaler}, "models/clustering/kmeans.pkl")
```

---

### §7. Anomaly Detection for Cost Spikes

**CRISP-ML(Q):** Modeling / Evaluation

**Literature basis:** Isolation Forest recommended for anomaly detection.

**Thin import:** `app.src.models`

#### 7.1. Isolation Forest

**Business Question:** Which VMs are anomalous in terms of cost or resource usage patterns?

```python
from app.src.models import AnomalyModel
anomaly = AnomalyModel(contamination=0.05)
anomaly.fit(X_cost_features)
anomalies = anomaly.predict(X_cost_features)
```

**Features:** `vm_cost`, `avg_cpu`, `max_cpu`, `lifetime_hours`, `core_count`

#### 7.2. Anomaly Characterization

- Convert `anomaly_mask` (numpy array from Isolation Forest) → `pd.Series` before `.groupby()`:
  ```python
  anomaly_sample['is_anomaly'] = pd.Series(anomaly_mask, index=anomaly_sample.index)
  ```
- Feature profile: anomalies vs. normal
- Anomaly rate by `vm_category`, core bucket
- Cost impact: total cost of anomalies

#### 7.3. Business Impact

- Total cost of anomalous VMs
- Estimated savings if anomalies are detected and addressed
- Recommendation for monitoring threshold

---

### §9. Explainability with SHAP

**CRISP-ML(Q):** Evaluation

**Literature basis:** SHAP and Integrated Gradients recommended for model interpretability.

#### 9.1. SHAP Explainer on Best Regressor

```python
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_sample)
```

#### 9.2. SHAP Summary Plot

- Beeswarm plot: global feature importance + direction of impact
- Bar plot: mean |SHAP| value per feature

#### 9.3. SHAP Dependence Plots

- Top-3 features: SHAP value vs. feature value
- Interaction effects with secondary features
- Business interpretation

#### 9.4. SHAP on Best Classifier

- Repeat SHAP analysis for the best classification model
- Identify features driving idle/waste predictions

#### 9.5. Business Insights from SHAP

- Actionable insights: "VMs with core_count > 4 and lifetime > 24h are consistently over-provisioned"
- Feature engineering recommendations for future iterations
- Monitoring recommendations (which features to track)

---

### §10. Model Comparison & Selection

**CRISP-ML(Q):** Evaluation

#### 10.1. Unified Performance Table

| Task | Best Model | Metric 1 | Metric 2 | Metric 3 | Training Time |
|---|---|---|---|---|---|
| avg_cpu prediction | XGBoost | MAE: 2.3 | R²: 0.82 | MAPE: 12.4% | 45s |
| waste_fraction prediction | Random Forest | MAE: 0.08 | R²: 0.79 | MAPE: 11.2% | 52s |
| vm_cost prediction | Random Forest | MAE: $1.2 | R²: 0.71 | MAPE: 14.8% | 120s |
| idle detection | XGBoost | F1: 0.94 | ROC-AUC: 0.97 | Precision: 0.93 | 38s |
| waste tier (multi) | XGBoost | Macro F1: 0.88 | Weighted F1: 0.91 | Accuracy: 0.89 | 42s |
| CPU timeseries | BiGRU | MAE: 1.8 | RMSE: 3.2 | — | 180s |

#### 10.2. Best Model per Business Goal

| Business Goal | Recommended Model | Rationale |
|---|---|---|
| Cost optimization triage | XGBoost waste tier classifier | Prioritize high-waste VMs with interpretable rules |
| Rightsizing recommendations | XGBoost avg_cpu regressor | Continuous prediction for downsizing candidates |
| Anomaly alerting | Isolation Forest | Unsupervised, catches unknown patterns |
| Capacity planning | BiGRU | Best timeseries accuracy for CPU forecasting |
| Stakeholder communication | SHAP on XGBoost regressor | Most interpretable for non-technical audience |

#### 10.3. Inference Time Benchmarking

| Model | Time per 1000 samples | Deployment Suitability |
|---|---|---|
| Ridge | 2ms | Real-time |
| XGBoost | 15ms | Real-time |
| Random Forest | 45ms | Near real-time |
| LSTM | 120ms | Batch |
| BiGRU | 150ms | Batch |

#### 10.4. Business Impact Synthesis

- **Rightsizing:** If the top-10% most over-provisioned VMs (by prediction) are downsized, estimated savings = $X (cross-reference with notebook 02 §4.5.5)
- **Idle detection:** Catching idle VMs within 1 hour of creation could save $Y
- **Anomaly alerting:** Top-1% cost anomalies represent $Z in unexpected spend
- **Workload segmentation:** Cluster-based auto-scaling rules could optimize $W

---

### §11. Conclusions and Recommendations

**CRISP-ML(Q):** Deployment / Monitoring

#### 11.1. Summary of Findings

- Best performing models per task
- Key features driving predictions (from SHAP + feature importance)
- Cluster insights and workload segmentation findings
- Anomaly detection results and business impact

#### 11.2. Practical Implications

- Recommended model for deployment
- Feature monitoring suggestions
- Integration with FinOps workflows
- Threshold recommendations for idle/anomaly detection

#### 11.3. Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| Data from 2019 | Model may not reflect current usage patterns | Retrain on newer data when available |
| No memory utilization | Waste analysis incomplete | Add when data available |
| Single trace (30 days) | May not capture seasonal patterns | Extend to multi-month data |
| Limited CPU readings (25/195 shards) | Timeseries models are proof-of-concept | Scale with full data when available |

#### 11.4. Future Work

| Future Work | Literature Basis | Priority |
|---|---|---|
| Temporal Fusion Transformer for multi-horizon forecasting | TFT (Lim et al.) | Medium |
| Deep Reinforcement Learning for auto-scaling | RLPRAF, Q-Learning | Low |
| Federated Learning for multi-cloud privacy | Federated RL | Low |
| CPU histogram percentile features (11 + 9 bins) | Google ClusterData (v3) | High |
| Multi-cell/region training for generalization | — | Medium |
| CI/CD pipeline for automated retraining | CAMS DevOps | Low |
| Real-time pricing API for current rates | — | Low |

---

### 4.5 03c_timeseries_forecasting.ipynb (§8 only)

**Purpose:** Load CPU readings via DuckDB out-of-core (avoids `pd.concat` MemoryError), train and compare timeseries models per-VM, aggregate metrics.

### §8. Deep Learning — Timeseries Forecasting

**CRISP-ML(Q):** Modeling / Evaluation

**Literature basis:** LSTM, BiGRU, CNN-LSTM, TFT are top recommendations for forecasting.

#### 8.1. Load CPU Readings Timeseries

**Business Question:** Can deep learning models predict future CPU utilization from historical timeseries?

**Approach:**
- Use DuckDB parquet glob to discover and group all shards (no full load into memory)
- Discover VMs with longest continuous traces (≥24h of data)
- Select up to 5 VMs with the longest traces for modeling
- Load **only those VMs' rows** at the parquet level — not the full dataset

```python
cpu_shards = sorted(DATA_DIR.glob("cpu_readings/*.parquet"))

if cpu_shards:
    # DuckDB glob → GROUP BY → discover top VMs (no full materialization)
    cpu_vm_stats = con.execute("""
        SELECT vm_id, COUNT(*) as count,
               MIN(timestamp) as min_ts, MAX(timestamp) as max_ts,
               (MAX(timestamp) - MIN(timestamp)) / 3600.0 AS duration_hours
        FROM read_parquet('data/transformed/parquet/cpu_readings/*.parquet')
        GROUP BY vm_id
    """).fetchdf()

    top_n = min(5, len(cpu_vm_stats))
    top_vms = cpu_vm_stats.nlargest(top_n, 'duration_hours')

    # Load only the selected VMs' data (DuckDB filters at parquet level)
    vm_ids_quoted = [f"'{v}'" for v in top_vms['vm_id'].tolist()]
    cpu_traces = con.execute(f"""
        SELECT vm_id, timestamp, avg_cpu
        FROM read_parquet('data/transformed/parquet/cpu_readings/*.parquet')
        WHERE vm_id IN ({', '.join(vm_ids_quoted)})
        ORDER BY vm_id, timestamp
    """).fetchdf()

    # Build dict of VM ID → sorted timeseries
    vm_series_dict = {}
    for vm_id in top_vms['vm_id']:
        series = cpu_traces[cpu_traces['vm_id'] == vm_id].reset_index(drop=True)
        vm_series_dict[vm_id] = series['avg_cpu'].values
```

#### 8.2. Data Preparation (per VM)

For each selected VM:

- Create sliding windows: lookback = 24 steps (2 hours at 5-min intervals)
- Train/val/test split: 70/15/15 chronological
- `MinMaxScaler` normalization
- Tensor format: `(samples, timesteps, features)`

```python
from app.src.features import create_sequences

all_sequences = {}  # vm_id → {X_train, X_val, X_test, y_train, y_val, y_test, scaler}
for vm_id, values in vm_series_dict.items():
    ts_scaler = MinMaxScaler()
    data_scaled = ts_scaler.fit_transform(values.reshape(-1, 1)).flatten()
    X_seq, y_seq = create_sequences(data_scaled, lookback=24)
    n = len(X_seq)
    X_train, y_train = X_seq[:int(n*0.7)], y_seq[:int(n*0.7)]
    X_val, y_val = X_seq[int(n*0.7):int(n*0.85)], y_seq[int(n*0.7):int(n*0.85)]
    X_test, y_test = X_seq[int(n*0.85):], y_seq[int(n*0.85):]
    all_sequences[vm_id] = {'X_train': ..., 'y_train': ..., ...}
```

#### 8.3. ARIMA Baseline

**Approach:**
- ACF/PACF plots for order selection
- ARIMA(5,1,0) on first VM only (raw, non-normalized)
- Evaluate on test period with MAE, RMSE

#### 8.4. LSTM Model

**Business Question:** Can LSTM capture long-term dependencies in CPU utilization?

**Architecture:**
```
Input(shape=(24, 1))
  → LSTM(64, return_sequences=False)
  → Dropout(0.2)
  → Dense(1)
```

**Training (per VM):** Adam(learning_rate=0.001), MSE loss, EarlyStopping(patience=10), 50 epochs
**Aggregation:** Metrics averaged across all modeled VMs.

#### 8.5. GRU / BiGRU Model

**Business Question:** Does BiGRU offer better performance or faster convergence than LSTM?

**Architecture:**
```
Input(shape=(24, 1))
  → Bidirectional(GRU(64))
  → Dropout(0.2)
  → Dense(1)
```

#### 8.6. CNN-LSTM Hybrid

**Business Question:** Can a CNN-LSTM capture both local patterns and long-term dependencies?

**Architecture:**
```
Input(shape=(24, 1))
  → Conv1D(filters=32, kernel_size=3, activation='relu')
  → MaxPooling1D(pool_size=2)
  → LSTM(32)
  → Dense(1)
```

#### 8.7. Temporal Fusion Transformer Discussion

- TFT is SOTA for multi-horizon forecasting with interpretable attention
- Requires significant data (multi-VM panel) and complexity
- Acknowledge as recommended future work
- Reference: Lim et al. "Temporal Fusion Transformers for interpretable multi-horizon time series forecasting"

#### 8.8. Model Comparison & Save

Results are aggregated across all VMs: mean ± std per architecture.

| Model | MAE (mean ± std) | RMSE (mean ± std) | Avg Training Time | Avg Parameters |
|---|---|---|---|---|
| ARIMA | | | | |
| LSTM | | | | |
| BiGRU | | | | |
| CNN-LSTM | | | | |

**VM coverage:** Trained on top-5 VMs from all available cpu_readings shards.

```python
import tensorflow as tf
best_model.save("models/timeseries/bigru_cpu.keras")
```

---

## 5. Thin Import Modules: app/src/

### 5.1. `app/src/features.py`

**Purpose:** Feature engineering functions called by the notebook. Single source of truth for feature logic.

**Functions:**

```python
def create_features(
    df: pd.DataFrame,
    pricing_df: pd.DataFrame | None = None,
    subscriptions_df: pd.DataFrame | None = None,
    deployments_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.

    Input: Raw vmtable DataFrame with columns:
        vm_id, subscription_id, deployment_id, timestamp_created, timestamp_deleted,
        max_cpu, avg_cpu, p95_max_cpu, vm_category, vm_core_count_bucket, vm_memory_gb_bucket

    Steps:
        1. Parse core_count and memory_gb from bucket strings
        2. Calculate lifetime_hours from timestamps
        3. Create cyclical temporal features (creation_hour, creation_dayofweek)
        4. Create ratio features (cpu_per_core, memory_per_core, burstiness, max_to_avg_ratio)
        5. Create binary flags (is_short_lived)
        6. One-hot encode vm_category, vm_core_count_bucket, vm_memory_gb_bucket
        7. Compute rate_per_hour via pricing lookup (if pricing_df provided)
        8. Compute vm_cost and waste_cost from rate_per_hour
        9. Join subscription-level features (if subscriptions_df provided)
        10. Join deployment-level features (if deployments_df provided)
        11. Create target columns: is_idle, waste_tier, waste_fraction

    Returns: DataFrame with all engineered features + original vm_id for indexing
    """


def get_feature_target_columns(
    task: str,
    feature_set: str = "all"
) -> tuple[list[str], str]:
    """
    Return (feature_columns, target_column) for a given ML task.

    Parameters:
        task: "regression_avg_cpu" | "regression_waste" | "regression_cost"
              | "classification_idle" | "classification_tier"
        feature_set: "all" | "minimal" (core features only) | "no_temporal"

    Returns:
        (list of feature column names, target column name)
    """


def load_cpu_readings(data_dir: str | Path, max_vms: int = 5) -> pd.DataFrame:
    """
    Load CPU readings shards from disk using DuckDB out-of-core parquet glob.

    Discovers all parquet files in cpu_readings/ subdirectory via DuckDB,
    GROUP BY vm_id to identify top-VMs, then fetches only their rows.
    Avoids pd.concat() memory blowup on large shard sets.

    Parameters:
        data_dir: Path to parquet data directory (e.g., "data/transformed/parquet")
        max_vms: Maximum number of VMs to return (default 5). None = all.

    Returns:
        pd.DataFrame with columns:
            vm_id, timestamp, avg_cpu, max_cpu, p95_max_cpu
        Returns empty DataFrame if no shards found.
    """


def create_sequences(
    data: np.ndarray,
    lookback: int = 24,
    forecast_horizon: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences for timeseries models.

    Parameters:
        data: 1D numpy array of timeseries values
        lookback: number of past timesteps (default 24 = 2 hours at 5-min intervals)
        forecast_horizon: number of future timesteps to predict (default 1)

    Returns:
        X: shape (n_samples, lookback, n_features)
        y: shape (n_samples, forecast_horizon)
    """
```

### 5.2. `app/src/models.py`

**Purpose:** Model wrappers providing a consistent interface (fit, predict, evaluate, save) across sklearn, XGBoost, CatBoost, clustering, and anomaly detection.

**Classes:**

```python
class BaseModel(ABC):
    """Abstract base for all model wrappers."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def evaluate(self, X: np.ndarray, y: np.ndarray, task: str = "regression") -> dict:
        """
        Compute standard metrics based on task type.
        Returns dict of metric_name → value.
        """

    def save(self, path: str, metadata: dict | None = None) -> str:
        """Save model + metadata. Returns path."""
        ...


class RidgeModel(BaseModel):
    """Ridge regression."""

class RandomForestModel(BaseModel):
    """RandomForestRegressor or RandomForestClassifier."""

class XGBoostModel(BaseModel):
    """XGBRegressor or XGBClassifier."""

class ClusterModel(BaseModel):
    """K-Means clustering wrapper."""
    def __init__(self, n_clusters: int = 4, random_state: int = 42): ...

class AnomalyModel(BaseModel):
    """Isolation Forest wrapper."""
    def __init__(self, contamination: float = 0.05, random_state: int = 42): ...


def load_model(path: str, model_type: str | None = None):
    """Load a saved model from path. Auto-detect type if not specified."""

def train_test_split_by_time(
    df: pd.DataFrame,
    timestamp_col: str,
    test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/test split (no data leakage)."""
```

### 5.3. `app/src/visualize.py`

**Purpose:** Publication-quality visualization functions.

**Functions:**

```python
def residual_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Analysis"
) -> plt.Figure:
    """
    Side-by-side: residual vs. predicted scatter + Q-Q plot.
    Returns figure for display/save.
    """


def feature_importance_plot(
    importances: dict[str, float],
    title: str = "Feature Importance",
    top_n: int = 15
) -> plt.Figure:
    """
    Horizontal bar chart of top-N features.
    importances: dict of feature_name → importance_score
    """


def cluster_scatter(
    X_2d: np.ndarray,
    labels: np.ndarray,
    title: str = "Cluster Visualization",
    centroids: np.ndarray | None = None
) -> plt.Figure:
    """
    2D scatter plot colored by cluster label.
    Optionally overlay cluster centroids.
    """


def comparison_table(
    results: dict[str, dict[str, float]]
) -> pd.DataFrame:
    """
    Build formatted comparison DataFrame from model results.
    results: {model_name: {metric_name: value}}
    Returns: pd.DataFrame with models as rows, metrics as columns.
    Best values highlighted.
    """
```

---

## 6. Testing: app/tests/

### 6.1. `conftest.py`

```python
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/transformed/parquet")


@pytest.fixture(scope="session")
def vmtable_sample() -> pd.DataFrame:
    """Load 1000 rows from vmtable.parquet for testing."""
    df = pd.read_parquet(DATA_DIR / "vmtable.parquet")
    return df.sample(1000, random_state=42)


@pytest.fixture(scope="session")
def pricing_sample() -> pd.DataFrame | None:
    """Load pricing data for testing. Returns None if file missing."""
    path = DATA_DIR / "azure_pricing.parquet"
    return pd.read_parquet(path) if path.exists() else None


@pytest.fixture(scope="session")
def subscriptions_sample() -> pd.DataFrame | None:
    """Load subscriptions data for testing."""
    path = DATA_DIR / "subscriptions.parquet"
    return pd.read_parquet(path) if path.exists() else None


@pytest.fixture(scope="session")
def deployments_sample() -> pd.DataFrame | None:
    """Load deployments data for testing."""
    path = DATA_DIR / "deployments.parquet"
    return pd.read_parquet(path) if path.exists() else None


@pytest.fixture(scope="session")
def engineered_features(vmtable_sample, pricing_sample,
                         subscriptions_sample,
                         deployments_sample) -> pd.DataFrame:
    """Create features from sample + auxiliary data for testing."""
    from app.src.features import create_features
    return create_features(vmtable_sample, pricing_sample,
                           subscriptions_sample, deployments_sample)


@pytest.fixture(scope="session")
def sequence_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic timeseries for sequence testing."""
    t = np.linspace(0, 100, 500)
    data = np.sin(t) + np.random.normal(0, 0.1, 500)
    from app.src.features import create_sequences
    return create_sequences(data, lookback=24)
```

### 6.2. `test_features.py` — 8 Tests

```python
import pandas as pd
import numpy as np


class TestCreateFeatures:
    """Tests for create_features()."""

    def test_output_shape(self, vmtable_sample):
        """Output has expected columns and no nulls."""
        from app.src.features import create_features
        result = create_features(vmtable_sample)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(vmtable_sample)
        assert result.isnull().sum().sum() == 0  # no nulls
        # Core features must exist
        for col in ['core_count', 'memory_gb', 'lifetime_hours', 'cpu_per_core',
                     'burstiness', 'is_short_lived', 'is_idle', 'waste_tier']:
            assert col in result.columns

    def test_core_count_parsing(self):
        """Verify bucket string parsing."""
        from app.src.features import create_features
        df = pd.DataFrame({
            'vm_id': ['a', 'b'],
            'vm_core_count_bucket': ['2', '>24'],
            # Fill required columns
            'vm_memory_gb_bucket': ['4', '>64'],
            'timestamp_created': [0, 0],
            'timestamp_deleted': [3600, 7200],
            'max_cpu': [50.0, 80.0],
            'avg_cpu': [10.0, 5.0],
            'p95_max_cpu': [30.0, 60.0],
            'vm_category': ['Interactive', 'Unknown'],
            'subscription_id': ['s1', 's2'],
            'deployment_id': ['d1', 'd2'],
        })
        result = create_features(df)
        assert result.loc[0, 'core_count'] == 2
        assert result.loc[1, 'core_count'] == 48
        assert result.loc[0, 'memory_gb'] == 4
        assert result.loc[1, 'memory_gb'] == 128

    def test_target_columns(self, engineered_features):
        """Target columns have correct types and ranges."""
        df = engineered_features
        assert df['is_idle'].dtype == bool
        assert df['waste_tier'].dtype.name == 'category'
        assert df['waste_tier'].cat.categories.tolist() == ['Low', 'Medium', 'High']

    def test_cyclical_encoding(self, engineered_features):
        """Sin/cos features are bounded in [-1, 1]."""
        df = engineered_features
        for col in ['creation_hour_sin', 'creation_hour_cos']:
            if col in df.columns:
                assert df[col].min() >= -1.0 - 1e-6
                assert df[col].max() <= 1.0 + 1e-6

    def test_create_sequences_shape(self, sequence_data):
        """Sequence shapes match expected dimensions."""
        X_seq, y_seq = sequence_data
        assert X_seq.ndim == 3  # (samples, timesteps, features)
        assert X_seq.shape[1] == 24  # lookback
        assert y_seq.ndim == 1
        assert len(X_seq) == len(y_seq)


class TestMultiTableFeatures:
    """Tests for multi-table feature augmentation."""

    def test_pricing_lookup(self, vmtable_sample, pricing_sample):
        """Rate per hour is assigned correctly from pricing lookup."""
        from app.src.features import create_features

        result = create_features(vmtable_sample, pricing_df=pricing_sample)
        assert 'rate_per_hour' in result.columns
        assert result['rate_per_hour'].min() >= 0
        assert result['vm_cost'].notna().any()
        assert result['vm_cost'].min() >= 0

    def test_subscription_features(self, vmtable_sample, subscriptions_sample):
        """Subscription-level features are present and joined."""
        from app.src.features import create_features

        result = create_features(vmtable_sample, subscriptions_df=subscriptions_sample)
        assert 'sub_vm_count' in result.columns
        assert result['sub_vm_count'].notna().all()
        assert result['sub_vm_count'].dtype in [int, 'int64', 'int32']

    def test_deployment_features(self, vmtable_sample, deployments_sample):
        """Deployment-level features are present and joined."""
        from app.src.features import create_features

        result = create_features(vmtable_sample, deployments_df=deployments_sample)
        assert 'deployment_size' in result.columns
        assert result['deployment_size'].notna().all()
```

### 6.3. `test_model.py` — 4 Tests

```python
import numpy as np
import tempfile


class TestModels:
    """Tests for model wrappers."""

    def test_sklearn_model_fit_predict(self, engineered_features):
        """Model returns correct shape predictions."""
        from app.src.models import XGBoostModel
        from app.src.features import get_feature_target_columns

        features, target = get_feature_target_columns("regression_avg_cpu")
        X = engineered_features[features].select_dtypes(include=[np.number]).values[:100]
        y = engineered_features[target].values[:100]

        model = XGBoostModel(task="regression", params={"n_estimators": 10, "max_depth": 3})
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(y)
        assert preds.ndim == 1

    def test_sklearn_model_evaluate(self, engineered_features):
        """Evaluate returns dict with expected keys."""
        from app.src.models import XGBoostModel
        from app.src.features import get_feature_target_columns

        features, target = get_feature_target_columns("regression_avg_cpu")
        X = engineered_features[features].select_dtypes(include=[np.number]).values[:100]
        y = engineered_features[target].values[:100]

        model = XGBoostModel(task="regression", params={"n_estimators": 10})
        model.fit(X, y)
        metrics = model.evaluate(X, y)
        assert isinstance(metrics, dict)
        for key in ['mae', 'rmse', 'r2']:
            assert key in metrics

    def test_cluster_model(self, engineered_features):
        """Cluster labels are 0..k-1 with no -1."""
        from app.src.models import ClusterModel
        from app.src.features import get_feature_target_columns

        features, _ = get_feature_target_columns("regression_avg_cpu")
        X = engineered_features[features].select_dtypes(include=[np.number]).values[:100]

        cluster = ClusterModel(n_clusters=4)
        cluster.fit(X)
        labels = cluster.predict(X)
        assert set(labels) == {0, 1, 2, 3}
        assert -1 not in labels

    def test_save_load_model(self, engineered_features):
        """Saved model loads and produces identical predictions."""
        from app.src.models import XGBoostModel, load_model
        from app.src.features import get_feature_target_columns

        features, target = get_feature_target_columns("regression_avg_cpu")
        X = engineered_features[features].select_dtypes(include=[np.number]).values[:100]
        y = engineered_features[target].values[:100]

        model = XGBoostModel(task="regression", params={"n_estimators": 10})
        model.fit(X, y)
        preds_before = model.predict(X)

        with tempfile.NamedTemporaryFile(suffix='.pkl') as tmp:
            model.save(tmp.name)
            loaded = load_model(tmp.name)
            preds_after = loaded.predict(X)

        np.testing.assert_array_almost_equal(preds_before, preds_after)
```

---

## 7. CI/CD: GitHub Actions

### `.github/workflows/ci.yml`

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
          pip install pytest nbconvert jupyter

      - name: Run unit tests
        run: |
          pytest app/tests/ -v --tb=short -x
        env:
          PYTHONPATH: ${{ github.workspace }}

      - name: Verify notebooks execute
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

---

## 8. Dependencies: requirements.txt

### Additions to existing (29 lines already present)

```
# === ML & Modeling ===
scikit-learn>=1.5.0
xgboost>=2.0.0
shap>=0.45.0

# === Deep Learning ===
tensorflow>=2.17.0

# === Time Series ===
statsmodels>=0.14.0

# === Model Persistence ===
joblib>=1.4.0

# === Testing ===
pytest>=8.0.0
pytest-cov>=5.0.0
```

### Final requirements.txt

```
# Environment variables
python-dotenv>=1.0.0

# Google Cloud BigQuery
google-cloud-bigquery>=3.25.0
google-cloud-bigquery-storage>=2.21.0
google-auth>=2.30.0

# Data processing
pandas>=2.2.0
numpy>=2.0.0

# Jupyter
jupyter>=1.0.0
ipykernel>=6.29.0

db_dtypes

# ibis-framework (pandas-like expressions compiling to BigQuery SQL)
ibis-framework[bigquery]>=9.0.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

requests
tqdm
pyarrow
duckdb

# === ML & Modeling ===
scikit-learn>=1.5.0
xgboost>=2.0.0
shap>=0.45.0

# === Deep Learning ===
tensorflow>=2.17.0

# === Time Series ===
statsmodels>=0.14.0

# === Model Persistence ===
joblib>=1.4.0

# === Testing ===
pytest>=8.0.0
pytest-cov>=5.0.0
```

---

## 9. Model Persistence: models/

### Directory Structure

```
models/
├── .gitkeep                              ← Preserve directory in git
├── regression/
│   └── xgboost_avg_cpu.pkl               ← Best avg_cpu predictor
├── classification/
│   ├── xgboost_idle.pkl                  ← Best idle VM classifier
│   └── xgboost_waste_tier.pkl            ← Best waste tier classifier
├── clustering/
│   └── kmeans_4.pkl                      ← K-Means + scaler (dict)
├── timeseries/
│   └── lstm_cpu.keras                    ← Best LSTM/GRU model (TensorFlow SavedModel)
└── run_log.csv                           ← Audit trail of all training runs
```

### `.gitignore` Addition

```
# Models (large binary files)
models/**/*.pkl
models/**/*.keras
models/**/*.h5
!models/.gitkeep
```

### run_log.csv Format

```csv
run_id,timestamp,task,model_name,params_hash,mae,rmse,r2,mse,accuracy,precision,recall,f1_score,roc_auc,training_time_s,model_path,git_hash
R001,2026-05-12T14:00:00,regression_avg_cpu,xgboost,a1b2c3,2.34,4.56,0.82,20.79,,,,,,45.2,models/regression/xgboost_avg_cpu.pkl,abc123
R002,2026-05-12T14:30:00,classification_idle,xgboost,d4e5f6,,,,,,0.93,0.94,0.95,0.94,0.97,38.1,models/classification/xgboost_idle.pkl,abc123
```

Generated in `03b` §10.1, appended in the notebook, committed to repo.

---

## 10. Git Strategy

### Commit Convention

Same as existing project (conventional commits):

```
feat: add 03a feature engineering notebook with multi-table loading
feat: add 03b tabular models notebook with ridge/rf/xgboost
feat: add 03c timeseries forecasting notebook with lstm/bigru
feat: add feature engineering module with temporal encoding
feat: add test suite for feature engineering
docs: add predictive analysis plan document
chore: add ML dependencies to requirements.txt
chore: set up GitHub Actions CI pipeline for 3 notebooks
```

### Branch Strategy

| Branch | Purpose |
|---|---|
| `main` | Production-ready code |
| `develop` | Integration branch |
| `feature/03-predictive-analysis` | 3 notebooks + app/src + tests work |

### Files to Commit

| Path | Type | In .gitignore? |
|---|---|---|
| `notebooks/03a_feature_engineering.ipynb` | Source | No |
| `notebooks/03b_tabular_models.ipynb` | Source | No |
| `notebooks/03c_timeseries_forecasting.ipynb` | Source | No |
| `app/src/features.py` | Source | No |
| `app/src/models.py` | Source | No |
| `app/src/visualize.py` | Source | No |
| `app/tests/*.py` | Test | No |
| `.github/workflows/ci.yml` | CI | No |
| `requirements.txt` | Config | No |
| `docs/AZURE_PREDICTIVE_ANALYSIS_PLAN.md` | Doc | No |
| `models/run_log.csv` | Data | No |
| `models/.gitkeep` | Placeholder | No |

---

## 11. Implementation Order

### Phase 1: Scaffold (Day 1)

| # | Task | Files |
|---|---|---|
| 1 | Create `app/src/features.py` | 1 file |
| 2 | Create `app/src/models.py` | 1 file |
| 3 | Create `app/src/visualize.py` | 1 file |
| 4 | Create `app/tests/__init__.py`, `conftest.py`, `test_features.py`, `test_model.py` | 4 files |
| 5 | Create `models/` directory with `.gitkeep` | 1 file |
| 6 | Update `.gitignore` for `models/` patterns | 1 file |
| 7 | Update `requirements.txt` with ML deps | 1 file |
| 8 | Create `.github/workflows/ci.yml` | 1 file |
| 9 | Update `AGENTS.md` with new commands | 1 file |

### Phase 2: Build 03a_feature_engineering.ipynb (Day 2-3)

| # | Task | Notebook Section |
|---|---|---|
| 10 | Write §1–3 (Setup, Data Loading, Feature Engineering, save artifact) | `03a` §1–3 |
| 11 | Implement correlation sampling + tqdm MI progress bar | `03a` §3.3 |

### Phase 3: Build 03b_tabular_models.ipynb (Day 4-5)

| # | Task | Notebook Section |
|---|---|---|
| 12 | Write §4 (Ridge, Random Forest, XGBoost regressors) | `03b` §4.1–4.7 |
| 13 | Write §5 (Binary + multi-class classification with RF + XGBoost) | `03b` §5.1–5.3 |
| 14 | Write §6 (K-Means clustering) | `03b` §6.1–6.6 |
| 15 | Write §7 (Isolation Forest with pd.Series fix) | `03b` §7.1–7.3 |
| 16 | Write §9 (SHAP explainability) | `03b` §9.1–9.5 |
| 17 | Write §10–11 (Comparison, conclusions) | `03b` §10–11 |

### Phase 4: Build 03c_timeseries_forecasting.ipynb (Day 5-6)

| # | Task | Notebook Section |
|---|---|---|
| 18 | Write §8 (DuckDB out-of-core load, per-VM sequences, ARIMA/LSTM/BiGRU/CNN-LSTM) | `03c` §8.1–8.8 |

### Phase 5: Synthesis & CI (Day 6)

| # | Task | Notebook Section |
|---|---|---|
| 19 | Run all 3 notebooks end-to-end, verify outputs | All |
| 20 | Run `pytest app/tests/`, fix failures | Tests |
| 21 | Commit all files, push, verify CI passes | Git |

---

**End of Plan**
