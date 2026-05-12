# Azure Predictive Analysis Plan — 03_predictive_analysis.ipynb

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
4. [Notebook Structure: 03_predictive_analysis.ipynb](#4-notebook-structure-03_predictive_analysisipynb)
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

This document defines the complete plan for building `03_predictive_analysis.ipynb` — the predictive modeling phase of the Sembada Cloud project. The notebook performs ML-based prediction of cloud resource utilization, waste detection, cost anomaly identification, and workload segmentation on the Azure Public Dataset V2.

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
| Tabular regression/classification | XGBoost, CatBoost, Random Forest | §4–5 |
| Time series forecasting | LSTM, BiGRU, CNN-LSTM, TFT | §8 |
| Workload segmentation | K-Means | §6 |
| Anomaly detection | Isolation Forest, WHA | §7 |
| Explainability | SHAP, Integrated Gradients | §9 |

### 1.4 Design Principle

**Notebook-first, thin-imports pattern.** The notebook is the primary academic deliverable. It imports reusable logic from `app/src/` modules (features.py, models.py, visualize.py) rather than duplicating code. This avoids dual-maintenance while keeping logic testable.

---

## 2. Architecture Overview

```
sembada-cloud/
├── notebooks/
│   └── 03_predictive_analysis.ipynb          ← Primary deliverable. CRISP-ML(Q) narrative + experiments
│
├── app/src/
│   ├── __init__.py                           ← Package marker (existing, unmodified)
│   ├── features.py                           ← Feature engineering functions (~80 lines)
│   ├── models.py                             ← Model wrappers (fit, predict, save) (~120 lines)
│   └── visualize.py                          ← Publication-quality figure functions (~60 lines)
│
├── app/tests/
│   ├── __init__.py
│   ├── conftest.py                           ← Shared fixtures (sample data)
│   ├── test_features.py                      ← 5 tests: shape, parsing, encodings, targets, sequences
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
data/transformed/parquet/
  ├── vmtable.parquet          ──► DuckDB ──► pandas ──► feature engineering ──► train/test split ──► models
  ├── subscriptions.parquet         │
  ├── deployments.parquet          │
  ├── azure_pricing.parquet        │
  └── cpu_readings/*.parquet       │
                                    └──► Timeseries: sliding windows ──► LSTM/GRU ──► models
```

### 2.2 Import Convention

The notebook imports from `app.src` as a thin layer:

```python
from app.src.features import create_features, create_sequences
from app.src.models import XGBoostModel, CatBoostModel, ClusterModel, AnomalyModel
from app.src.visualize import residual_plot, comparison_table
```

---

## 3. CRISP-ML(Q) & CAMS DevOps Integration

### 3.1 CRISP-ML(Q) Phase → Notebook Section Map

| CRISP-ML(Q) Phase | Notebook Section | Key Deliverable |
|---|---|---|
| **Business Understanding** | §1 Summary, each subsection's **Business Question** | Documented business goals, success criteria |
| **Data Understanding** | §2 Preparation, §3.3 Correlation Analysis | Dataset statistics, feature-target relationships |
| **Data Preparation** | §3 Feature Engineering | `features.py`, engineered DataFrame, train/test split |
| **Modeling** | §4 Regression, §5 Classification, §6 Clustering, §7 Anomaly, §8 Deep Learning | Trained models in `models/` |
| **Evaluation** | §4.5–4.7, §5.1.4, §6.2, §9 SHAP, §10 Comparison | Metrics table, SHAP analysis, best model selection |
| **Deployment** | §11 Conclusions & Recommendations | Business impact report, model card |
| **Monitoring** | §11.3–11.4 Limitations & Future Work | Identified gaps, improvement roadmap |

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

Generated by notebook §10.1 and committed to repo for auditability.

---

## 4. Notebook Structure: 03_predictive_analysis.ipynb

### 4.1 Design Conventions

Same as `02_azure_descriptive_analysis.ipynb`:

- **Markdown cells** at each subsection header containing **Business Question** and **Analysis Approach**
- **Code cells** with executable, commented code using DuckDB + pandas + `app.src` imports
- **"Key Findings"** markdown cells after each code block summarizing results
- `display()` for DataFrames, formatted `print()` for text output
- `matplotlib` + `seaborn` for all visualizations
- Seeds set for reproducibility: `random_state=42`, `np.random.seed(42)`

### 4.2 Section-by-Section Specification

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
# Standard stack (reused from notebook 02)
import os, sys, warnings, pathlib
import duckdb, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid", palette="muted")
warnings.filterwarnings("ignore")

# Thin-imports
from app.src.features import create_features, get_feature_target_columns, create_sequences
from app.src.models import (XGBoostModel, CatBoostModel, RandomForestModel,
                            LinearModel, ClassifyModel, ClusterModel, AnomalyModel)
from app.src.visualize import residual_plot, feature_importance_plot, cluster_scatter, comparison_table

# Scikit-learn
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, classification_report, confusion_matrix)
from sklearn.pipeline import Pipeline

# Gradient boosting
import xgboost as xgb
import catboost as cb

# Deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Explainability
import shap

# Model persistence
import joblib

# Statistical
from statsmodels.tsa.arima.model import ARIMA

# Imbalanced data
from imblearn.over_sampling import SMOTE
```

#### 2.2. Load Dataset

```python
DATA_DIR = pathlib.Path("data/transformed/parquet")
PRICING_PATH = DATA_DIR / "azure_pricing.parquet"

con = duckdb.connect(":memory:")

# Register views (matching notebook 02 conventions)
for tbl in ["vmtable", "subscriptions", "deployments", "pricing"]:
    path = DATA_DIR / f"{tbl}.parquet"
    if path.exists():
        con.execute(f"CREATE VIEW {tbl} AS SELECT * FROM read_parquet('{path}')")
        print(f"  ✓ {tbl}: {con.execute(f'SELECT COUNT(*) FROM {tbl}').fetchone()[0]:,} rows")

# Load pricing
if PRICING_PATH.exists():
    pricing_df = con.execute("SELECT * FROM pricing").fetchdf()
else:
    pricing_df = None

# Load main table
vmtable = con.execute("SELECT * FROM vmtable").fetchdf()
print(f"✓ vmtable loaded: {len(vmtable):,} rows, {len(vmtable.columns)} columns")
```

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
def create_features(df: pd.DataFrame) -> pd.DataFrame:
```

| Feature | Type | Derivation |
|---|---|---|
| `core_count` | numeric | Parse `vm_core_count_bucket`: 2→2, 4→4, 8→8, 24→24, >24→48 |
| `memory_gb` | numeric | Parse `vm_memory_gb_bucket`: 2→2, 4→4, 8→8, 32→32, 64→64, >64→128 |
| `lifetime_hours` | numeric | `(timestamp_deleted - timestamp_created) / 3600` |
| `creation_hour` | cyclic (sin/cos) | `hour = timestamp_to_hour(timestamp_created)`; `sin = sin(2πh/24)`, `cos = cos(2πh/24)` |
| `creation_dayofweek` | cyclic (sin/cos) | Same cyclical encoding for day of week |
| `cpu_per_core` | ratio | `avg_cpu / core_count` |
| `memory_per_core` | ratio | `memory_gb / core_count` |
| `burstiness` | ratio | `p95_max_cpu / (avg_cpu + 1e-6)` |
| `max_to_avg_ratio` | ratio | `max_cpu / (avg_cpu + 1e-6)` |
| `is_short_lived` | binary | `lifetime_hours < 1` |
| `vm_category_*` | one-hot | `pd.get_dummies(df['vm_category'], prefix='cat')` |
| `core_bucket_*` | one-hot | `pd.get_dummies(df['vm_core_count_bucket'], prefix='core')` |
| `mem_bucket_*` | one-hot | `pd.get_dummies(df['vm_memory_gb_bucket'], prefix='mem')` |

#### 3.3. Feature-Target Correlation & Mutual Information

**Business Question:** Which features have the strongest predictive relationship with each target?

**Approach:**
- Pearson correlation matrix (all features × all targets)
- Mutual information scores (captures non-linear relationships)
- Top-10 features bar chart for each target

**Output:** Correlation heatmap + MI bar chart + written findings.

#### 3.4. Train-Test Split

**Business Question:** How do we ensure the model generalizes to unseen VMs?

**Approach:**
- Stratified 80/20 split by `waste_tier` to preserve class balance
- Optional: time-based split using `timestamp_created` for temporal validation
- `random_state=42` for reproducibility

---

### §4. Baseline Regression Models — Cost & Utilization Prediction

**CRISP-ML(Q):** Modeling / Evaluation

**Thin import:** `app.src.models`

#### 4.1. Linear Regression & Ridge

**Business Question:** How well does a simple linear model predict CPU utilization and waste?

**Approach:**
- `LinearRegression()` as naive baseline
- `Ridge(alpha=...)` with GridSearchCV
- MAE, RMSE, R², MAPE on test set

#### 4.2. Random Forest Regressor

**Business Question:** Can an ensemble of decision trees capture non-linear resource patterns?

**Approach:**
- `RandomForestRegressor(n_estimators=300, max_depth=15, ...)`
- Hyperparameter tuning via `RandomizedSearchCV`
- Feature importance extraction

#### 4.3. XGBoost Regressor

**Business Question:** Does gradient boosting outperform bagging for cloud resource prediction?

**Approach:**
- `XGBRegressor(learning_rate=0.05, max_depth=6, subsample=0.8, ...)`
- Early stopping on validation set
- Learning curve visualization

#### 4.4. CatBoost Regressor

**Business Question:** Can CatBoost's native categorical feature handling improve accuracy?

**Approach:**
- `CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, ...)`
- Pass categorical feature indices directly
- Compare training speed vs. XGBoost

#### 4.5. Model Evaluation Comparison

| Model | MAE (avg_cpu) | RMSE (avg_cpu) | R² (avg_cpu) | MAPE | Training Time |
|---|---|---|---|---|---|
| Linear Regression | | | | | |
| Ridge | | | | | |
| Random Forest | | | | | |
| XGBoost | | | | | |
| CatBoost | | | | | |

**Output:** Side-by-side table with best values highlighted. Repeated for each target (`avg_cpu`, `waste_fraction`, `vm_cost`).

#### 4.6. Feature Importance Analysis

- Built-in importance (RF, XGBoost, CatBoost)
- Permutation importance (model-agnostic)
- Top-15 features bar chart
- Comparison of importance across models

#### 4.7. Residual Analysis

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

#### 4.8. Save Best Model

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
- 5.1.1. **Logistic Regression** — interpretable baseline
- 5.1.2. **Random Forest Classifier** — non-linear ensemble
- 5.1.3. **XGBoost Classifier** — gradient boosting

**Evaluation:**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | | | | | |
| Random Forest | | | | | |
| XGBoost | | | | | |

- Confusion matrix for best model
- Precision-Recall curve (handles class imbalance)
- ROC curve

#### 5.2. Multi-Class: Waste Tier Classification

**Business Question:** Can we classify VMs into waste tiers (Low/Medium/High) for optimization prioritization?

**Approach:**
- Target: `waste_tier` with 3 ordered classes
- Models: XGBoost, Random Forest, LogisticRegression (OvR)
- Class imbalance: apply SMOTE and/or `class_weight='balanced'`

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

```
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

- Feature profile: anomalies vs. normal
- Anomaly rate by `vm_category`, core bucket
- Cost impact: total cost of anomalies
- Temporal pattern analysis (hourly anomaly rate)

#### 7.3. Business Impact

- Total cost of anomalous VMs
- Estimated savings if anomalies are detected and addressed
- Recommendation for monitoring threshold

---

### §8. Deep Learning — Timeseries Forecasting

**CRISP-ML(Q):** Modeling / Evaluation

**Literature basis:** LSTM, BiGRU, CNN-LSTM, TFT are top recommendations for forecasting.

#### 8.1. Load CPU Readings Timeseries

**Business Question:** Can deep learning models predict future CPU utilization from historical timeseries?

**Approach:**
- Select a single VM from `cpu_readings/*.parquet` with ≥24h of continuous data
- Load and sort chronologically by `timestamp`

```python
cpu_shards = sorted(DATA_DIR.glob("cpu_readings/*.parquet"))
sample = pd.read_parquet(cpu_shards[0])
vm_id = sample['vm_id'].value_counts().index[0]
vm_series = sample[sample['vm_id'] == vm_id].sort_values('timestamp')
```

#### 8.2. Data Preparation

- Create sliding windows: lookback = 24 steps (2 hours at 5-min intervals)
- Train/val/test split: 70/15/15 chronological
- `MinMaxScaler` normalization
- Tensor format: `(samples, timesteps, features)`

```python
from app.src.features import create_sequences
X_seq, y_seq = create_sequences(vm_series['avg_cpu'].values, lookback=24)
```

#### 8.3. ARIMA Baseline

**Approach:**
- ACF/PACF plots for order selection
- Grid search for (p, d, q) parameters
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

**Training:** Adam(learning_rate=0.001), MSE loss, EarlyStopping(patience=10), 100 epochs

```python
model = Sequential([
    LSTM(64, input_shape=(24, 1)),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
```

#### 8.5. GRU / BiGRU Model

**Business Question:** Does BiGRU offer better performance or faster convergence than LSTM?

**Architecture:**
```
Input(shape=(24, 1))
  → Bidirectional(GRU(64))
  → Dropout(0.2)
  → Dense(1)
```

#### 8.6. CNN-LSTM Hybrid (Primer)

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

| Model | MAE | RMSE | Training Time | Parameters |
|---|---|---|---|---|
| ARIMA | | | | |
| LSTM | | | | |
| GRU | | | | |
| BiGRU | | | | |
| CNN-LSTM | | | | |

```python
best_model.save("models/timeseries/lstm_cpu.keras")
```

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

```python
shap.summary_plot(shap_values, X_test_sample, feature_names=feature_cols)
```

#### 9.3. SHAP Dependence Plots

- Top-3 features: SHAP value vs. feature value
- Interaction effects with secondary features
- Business interpretation

```python
shap.dependence_plot("core_count", shap_values, X_test_sample)
```

#### 9.4. SHAP on Best Classifier

- Repeat SHAP analysis for the best classification model
- Identify features driving idle/waste predictions
- Compare SHAP patterns with regression model

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
| waste_fraction prediction | CatBoost | MAE: 0.08 | R²: 0.79 | MAPE: 11.2% | 52s |
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
| Logistic Regression | 2ms | Real-time |
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
| CPU readings: 25/195 shards | Timeseries models trained on limited data | Proof-of-concept only; scale with full data |
| No memory utilization | Waste analysis incomplete | Add when data available |
| Pricing approximations | Cost predictions have inherent error | Document margin of error |
| Single trace (30 days) | May not capture seasonal patterns | Extend to multi-month data |

#### 11.4. Future Work

| Future Work | Literature Basis | Priority |
|---|---|---|
| Temporal Fusion Transformer for multi-horizon forecasting | TFT (Lim et al.) | Medium |
| Deep Reinforcement Learning for auto-scaling | RLPRAF, Q-Learning | Low |
| Federated Learning for multi-cloud privacy | Federated RL | Low |
| WHA preprocessing for anomaly detection | Weighted Hybrid Algorithm | Medium |
| CPU histogram percentile features (11 + 9 bins) | Google ClusterData (v3) | High |
| Multi-cell/region training for generalization | — | Medium |
| CI/CD pipeline for automated retraining | CAMS DevOps | Low |

---

## 5. Thin Import Modules: app/src/

### 5.1. `app/src/features.py`

**Purpose:** Feature engineering functions called by the notebook. Single source of truth for feature logic.

**Functions:**

```python
def create_features(df: pd.DataFrame, pricing_df: pd.DataFrame | None = None) -> pd.DataFrame:
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
        7. Create target columns: is_idle, waste_tier
        8. Calculate vm_cost if pricing_df provided (else skip)

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


class LinearModel(BaseModel):
    """LinearRegression or Ridge."""

class RandomForestModel(BaseModel):
    """RandomForestRegressor or RandomForestClassifier."""

class XGBoostModel(BaseModel):
    """XGBRegressor or XGBClassifier."""

class CatBoostModel(BaseModel):
    """CatBoostRegressor or CatBoostClassifier."""

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
def engineered_features(vmtable_sample) -> pd.DataFrame:
    """Create features from sample data for testing."""
    from app.src.features import create_features
    return create_features(vmtable_sample)


@pytest.fixture(scope="session")
def sequence_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic timeseries for sequence testing."""
    t = np.linspace(0, 100, 500)
    data = np.sin(t) + np.random.normal(0, 0.1, 500)
    from app.src.features import create_sequences
    return create_sequences(data, lookback=24)
```

### 6.2. `test_features.py` — 5 Tests

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

      - name: Verify notebook executes
        run: |
          jupyter nbconvert --to notebook \
            --execute notebooks/03_predictive_analysis.ipynb \
            --output /dev/null \
            --ExecutePreprocessor.timeout=600
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
catboost>=1.2.0
shap>=0.45.0

# === Deep Learning ===
tensorflow>=2.17.0

# === Time Series ===
statsmodels>=0.14.0

# === Imbalanced Data ===
imbalanced-learn>=0.12.0

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
catboost>=1.2.0
shap>=0.45.0

# === Deep Learning ===
tensorflow>=2.17.0

# === Time Series ===
statsmodels>=0.14.0

# === Imbalanced Data ===
imbalanced-learn>=0.12.0

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
│   ├── xgboost_avg_cpu.pkl               ← Best avg_cpu predictor
│   ├── catboost_waste_fraction.pkl       ← Best waste predictor
│   └── random_forest_vm_cost.pkl          ← Best cost predictor (if > linear)
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

Generated in notebook §10.1, appended in the notebook, committed to repo.

---

## 10. Git Strategy

### Commit Convention

Same as existing project (conventional commits):

```
feat: add predictive analysis notebook with regression models
feat: add feature engineering module with temporal encoding
feat: add test suite for feature engineering
docs: add predictive analysis plan document
chore: add ML dependencies to requirements.txt
chore: set up GitHub Actions CI pipeline
```

### Branch Strategy

| Branch | Purpose |
|---|---|
| `main` | Production-ready code |
| `develop` | Integration branch |
| `feature/03-predictive-analysis` | Notebook + app/src + tests work |

### Files to Commit

| Path | Type | In .gitignore? |
|---|---|---|
| `notebooks/03_predictive_analysis.ipynb` | Source | No |
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

### Phase 2: Core Implementation (Day 2-3)

| # | Task | Notebook Section |
|---|---|---|
| 10 | Write §1–3 (Setup, Data Loading, Feature Engineering) | §1–3 |
| 11 | Write §4 (Regression models: Linear → XGBoost → CatBoost) | §4.1–4.8 |
| 12 | Write §5 (Classification: idle + waste tier) | §5.1–5.3 |
| 13 | Write §6 (K-Means clustering) | §6.1–6.6 |

### Phase 3: Advanced Implementation (Day 4-5)

| # | Task | Notebook Section |
|---|---|---|
| 14 | Write §7 (Isolation Forest anomaly detection) | §7.1–7.3 |
| 15 | Write §8 (Timeseries: LSTM, GRU, BiGRU, CNN-LSTM) | §8.1–8.8 |
| 16 | Write §9 (SHAP explainability) | §9.1–9.5 |

### Phase 4: Synthesis & CI (Day 5-6)

| # | Task | Notebook Section |
|---|---|---|
| 17 | Write §10 (Model comparison, inference benchmark, business impact) | §10.1–10.4 |
| 18 | Write §11 (Conclusions, limitations, future work) | §11.1–11.4 |
| 19 | Run full notebook end-to-end, verify outputs | All |
| 20 | Run `pytest app/tests/`, fix failures | Tests |
| 21 | Commit all files, push, verify CI passes | Git |

---

**End of Plan**
