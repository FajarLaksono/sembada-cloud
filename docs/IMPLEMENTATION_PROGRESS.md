# Implementation Progress — 03_predictive_analysis.ipynb

**Project:** Sembada Cloud - Predictive Analysis Phase  
**Status:** Planning  
**Started:** 2026-05-12  
**Target Completion:** 2026-05-18 (6 days)

---

## Table of Contents

1. [Overview](#overview)
2. [Phase 1: Scaffolding (Day 1)](#phase-1-scaffolding-day-1)
3. [Phase 2: Core Implementation (Days 2-3)](#phase-2-core-implementation-days-2-3)
4. [Phase 3: Advanced Implementation (Days 4-5)](#phase-3-advanced-implementation-days-4-5)
5. [Phase 4: Synthesis & Validation (Days 5-6)](#phase-4-synthesis--validation-days-5-6)
6. [Deliverables Checklist](#deliverables-checklist)
7. [Progress Log](#progress-log)

---

## Overview

### Goal
Implement a complete predictive analysis notebook (`03_predictive_analysis.ipynb`) following **CRISP-ML(Q)** and **CAMS DevOps** methodologies for:
- Resource utilization & cost prediction
- Idle VM detection
- Cloud waste identification
- Workload segmentation
- Anomaly detection
- Time series forecasting

### Methodology
- **Framework:** CRISP-ML(Q) (Business → Data → Modeling → Evaluation → Deployment → Monitoring)
- **DevOps:** CAMS (Culture, Automation, Measurement, Sharing)
- **Code Pattern:** Notebook-first + thin imports from `app/src/` modules

### Architecture
```
notebooks/03_predictive_analysis.ipynb     ← Main deliverable (11 sections)
app/src/
  ├── features.py                         ← Feature engineering (~80 lines)
  ├── models.py                           ← Model wrappers (~120 lines)
  └── visualize.py                        ← Visualization (~60 lines)
app/tests/
  ├── conftest.py                         ← Shared fixtures
  ├── test_features.py                    ← 5 tests for features
  └── test_model.py                       ← 4 tests for models
models/
  ├── regression/                         ← Best regressors
  ├── classification/                     ← Best classifiers
  ├── clustering/                         ← K-Means + scaler
  ├── timeseries/                         ← LSTM/GRU models
  ├── .gitkeep                            ← Directory placeholder
  └── run_log.csv                         ← Experiment audit trail
.github/workflows/ci.yml                  ← GitHub Actions CI/CD
requirements.txt                          ← Updated ML dependencies
.gitignore                                ← Model patterns
```

---

## Phase 1: Scaffolding (Day 1)

### Objective
Create all infrastructure files, modules, tests, and CI/CD pipeline.

### Files to Create

#### Core Modules (app/src/)
- [x] **features.py** — Feature engineering functions
  - `create_features(df, pricing_df) → DataFrame`
  - `get_feature_target_columns(task, feature_set) → (list, str)`
  - `create_sequences(data, lookback, forecast_horizon) → (X, y)`
  
- [x] **models.py** — Model wrapper classes
  - `BaseModel` (abstract)
  - `LinearModel`, `RandomForestModel`, `XGBoostModel`, `CatBoostModel`
  - `ClusterModel` (K-Means)
  - `AnomalyModel` (Isolation Forest)
  - `load_model(path, model_type)`
  - `train_test_split_by_time(df, timestamp_col, test_size)`

- [x] **visualize.py** — Visualization functions
  - `residual_plot(y_true, y_pred, title) → Figure`
  - `feature_importance_plot(importances, title, top_n) → Figure`
  - `cluster_scatter(X_2d, labels, title, centroids) → Figure`
  - `comparison_table(results) → DataFrame`

#### Test Suite (app/tests/)
- [x] **__init__.py** — Package marker (empty)

- [x] **conftest.py** — Pytest fixtures
  - `vmtable_sample` fixture (1000 rows)
  - `engineered_features` fixture
  - `sequence_data` fixture

- [x] **test_features.py** — 5 feature tests
  - `test_output_shape` — Verify columns and nulls
  - `test_core_count_parsing` — Bucket string parsing
  - `test_target_columns` — Type and range validation
  - `test_cyclical_encoding` — Sin/cos bounds [-1, 1]
  - `test_create_sequences_shape` — Sequence dimensions

- [x] **test_model.py** — 4 model tests
  - `test_sklearn_model_fit_predict` — Shape validation
  - `test_sklearn_model_evaluate` — Metrics dict keys
  - `test_cluster_model` — Cluster labels 0..k-1
  - `test_save_load_model` — Pickle persistence

#### Infrastructure
- [x] **.gitkeep** in `models/` directory (preserve in git)

- [x] **.github/workflows/ci.yml** — GitHub Actions pipeline
  - Python 3.13 setup
  - Install dependencies
  - Run `pytest app/tests/ -v`
  - Execute notebook with `papermill` (real-time output via `--log-output`)

- [x] **requirements.txt** — Update with ML stack
  ```
  scikit-learn>=1.5.0
  xgboost>=2.0.0
  catboost>=1.2.0
  shap>=0.45.0
  tensorflow>=2.17.0
  statsmodels>=0.14.0
  imbalanced-learn>=0.12.0
  joblib>=1.4.0
  pytest>=8.0.0
  pytest-cov>=5.0.0
  ```

- [x] **.gitignore** — Add model patterns
  ```
  models/**/*.pkl
  models/**/*.keras
  models/**/*.h5
  !models/.gitkeep
  ```

#### Documentation Updates
- [x] **AGENTS.md** — Add new commands (training, testing)

### Status
- [x] All files created
- [ ] Code passes linting (PEP 8)
- [x] No import errors

---

## Phase 2: Core Implementation (Days 2-3)

### Objective
Build the main notebook sections: setup, feature engineering, and baseline models.

### Notebook Structure: §1 – §6

#### §1. Summary
- [x] Business understanding section
- [x] Dataset overview (Azure Public Dataset V2, 2.7M VMs)
- [x] Success criteria (MAPE < 15%, F1 > 0.85)
- [x] Key finding preview

**Dependencies:** None  
**Target Lines:** ~50

---

#### §2. Preparation
- [x] 2.1 Import all libraries
  - Pandas, NumPy, Matplotlib, Seaborn
  - Scikit-learn, XGBoost, CatBoost
  - TensorFlow/Keras, SHAP, joblib
  - Thin imports from `app.src`
  
- [x] 2.2 Load dataset from parquet
  - DuckDB for efficient querying
  - Create views for vmtable, subscriptions, deployments, pricing
  - Print row counts

**Dependencies:** `app/src/` modules  
**Target Lines:** ~80

---

#### §3. Feature Engineering
- [ ] 3.1 Define target variables
  - `avg_cpu` (regression)
  - `waste_fraction` (regression)
  - `vm_cost` (regression)
  - `is_idle` (binary classification)
  - `waste_tier` (multi-class)

- [ ] 3.2 Create engineered features
  - Call `create_features(vmtable, pricing_df)`
  - Parse core_count, memory_gb from buckets
  - Calculate lifetime_hours
  - Cyclical encoding (hour, day-of-week)
  - Ratio features (cpu_per_core, burstiness, max_to_avg_ratio)
  - One-hot encoding (vm_category, core_bucket, mem_bucket)

- [ ] 3.3 Feature-target correlation analysis
  - Pearson correlation heatmap
  - Mutual information scores
  - Top-10 features bar chart per target

- [ ] 3.4 Train-test split
  - 80/20 stratified by waste_tier
  - Optional time-based split
  - `random_state=42`

**Dependencies:** `app/src/features`  
**Target Lines:** ~150

---

#### §4. Regression Models (avg_cpu, waste, cost)
- [ ] 4.1 Linear Regression baseline
  - `LinearRegression()`
  - Metrics: MAE, RMSE, R², MAPE

- [ ] 4.2 Ridge regression
  - GridSearchCV for α
  - Metrics table

- [ ] 4.3 Random Forest Regressor
  - `n_estimators=300, max_depth=15`
  - RandomizedSearchCV
  - Feature importance

- [ ] 4.4 XGBoost Regressor
  - `learning_rate=0.05, max_depth=6`
  - Early stopping
  - Learning curve

- [ ] 4.5 CatBoost Regressor
  - `iterations=500, depth=6`
  - Categorical feature handling
  - Speed comparison

- [ ] 4.6 Model comparison table
  - Side-by-side metrics
  - Best values highlighted
  - Repeated for each target

- [ ] 4.7 Feature importance analysis
  - Built-in importance (RF, XGB, CB)
  - Permutation importance
  - Top-15 features bar chart

- [ ] 4.8 Residual analysis
  - Residual vs. predicted scatter
  - Q-Q plot
  - Heteroscedasticity check
  - Outlier identification

- [ ] 4.9 Save best regression model
  - `joblib.dump()` to `models/regression/`

**Dependencies:** `app/src/models`, `app/src/visualize`  
**Target Lines:** ~400

---

#### §5. Classification (Idle & Waste Tier)
- [ ] 5.1 Binary classification: idle detection
  - Logistic Regression
  - Random Forest Classifier
  - XGBoost Classifier
  - Confusion matrix, PR curve, ROC curve
  - Metrics table (accuracy, precision, recall, F1, ROC-AUC)

- [ ] 5.2 Multi-class classification: waste tier
  - Target: Low/Medium/High
  - SMOTE for imbalance
  - `class_weight='balanced'`
  - Per-class and macro F1

- [ ] 5.3 Save best classifiers
  - `joblib.dump()` to `models/classification/`

**Dependencies:** `app/src/models`, `app/src/visualize`  
**Target Lines:** ~250

---

#### §6. Clustering (K-Means Workload Segmentation)
- [ ] 6.1 K-Means clustering
  - Features: avg_cpu, max_cpu, p95_max_cpu, core_count, memory_gb, burstiness, lifetime_hours
  - StandardScaler preprocessing
  - `n_clusters=4`

- [ ] 6.2 Optimal K selection
  - Elbow method (inertia vs K)
  - Silhouette score
  - Select best K

- [ ] 6.3 Cluster characterization
  - Summary table (size, avg metrics, dominant category)
  - Business labels (e.g., "Ephemeral small")

- [ ] 6.4 PCA / t-SNE visualization
  - 2D scatter plot colored by cluster
  - Optional centroids overlay

- [ ] 6.5 Cluster-category cross-tabulation
  - `pd.crosstab()` analysis

- [ ] 6.6 Save cluster model
  - `joblib.dump({"kmeans": ..., "scaler": ...})`

**Dependencies:** `app/src/models`, `app/src/visualize`  
**Target Lines:** ~250

---

### Phase 2 Milestones
- [ ] §1-6 complete and runnable
- [ ] All model outputs saved to `models/`
- [ ] No errors in notebook execution

**Target Completion:** End of Day 3

---

## Phase 3: Advanced Implementation (Days 4-5)

### Objective
Implement anomaly detection, time series forecasting, and explainability.

---

#### §7. Anomaly Detection (Isolation Forest)
- [ ] 7.1 Isolation Forest model
  - Features: vm_cost, avg_cpu, max_cpu, lifetime_hours, core_count
  - `contamination=0.05`
  - Fit on training data

- [ ] 7.2 Anomaly characterization
  - Feature profile: anomalies vs. normal
  - Anomaly rate by category
  - Total cost of anomalies
  - Estimated savings

- [ ] 7.3 Business impact analysis
  - Monitoring threshold recommendations

**Dependencies:** `app/src/models`  
**Target Lines:** ~150

---

#### §8. Deep Learning — Time Series Forecasting
- [ ] 8.1 Load CPU readings timeseries
  - Select single VM with ≥24h data
  - Sort chronologically

- [ ] 8.2 Data preparation
  - Sliding windows: lookback=24 (2 hours @ 5-min intervals)
  - Train/val/test: 70/15/15 chronological split
  - MinMaxScaler normalization
  - Tensor format: (samples, timesteps, features)

- [ ] 8.3 ARIMA baseline
  - ACF/PACF analysis
  - Grid search for (p, d, q)
  - MAE, RMSE on test set

- [ ] 8.4 LSTM model
  - Architecture: LSTM(64) → Dropout(0.2) → Dense(1)
  - Adam optimizer, MSE loss
  - EarlyStopping(patience=10), 100 epochs
  - Plot training history

- [ ] 8.5 GRU / BiGRU model
  - Architecture: Bidirectional(GRU(64)) → Dropout → Dense
  - Compare convergence and performance

- [ ] 8.6 CNN-LSTM hybrid
  - Architecture: Conv1D → MaxPooling → LSTM → Dense
  - Local + long-term dependency capture

- [ ] 8.7 Temporal Fusion Transformer discussion
  - Acknowledge as SOTA for multi-horizon forecasting
  - Reference academic paper (Lim et al.)
  - Mark as future work

- [ ] 8.8 Model comparison & save
  - Table: ARIMA, LSTM, GRU, BiGRU, CNN-LSTM
  - Metrics: MAE, RMSE, training time, parameters
  - Save best to `models/timeseries/`

**Dependencies:** `app/src/features`, TensorFlow/Keras  
**Target Lines:** ~450

---

#### §9. Explainability (SHAP)
- [ ] 9.1 SHAP Explainer on best regressor
  - TreeExplainer for XGBoost
  - Sample 1000 test instances for efficiency

- [ ] 9.2 SHAP summary plot
  - Beeswarm plot (global feature importance + direction)
  - Bar plot (mean |SHAP| per feature)

- [ ] 9.3 SHAP dependence plots
  - Top-3 features: SHAP value vs. feature value
  - Interaction effects

- [ ] 9.4 SHAP on best classifier
  - Repeat analysis for idle/waste tier classifier
  - Compare patterns with regression

- [ ] 9.5 Business insights from SHAP
  - Actionable findings
  - Feature engineering recommendations
  - Monitoring suggestions

**Dependencies:** `shap`, best models from §4, §5  
**Target Lines:** ~250

---

### Phase 3 Milestones
- [ ] §7-9 complete and runnable
- [ ] All deep learning models trained and saved
- [ ] SHAP plots generated

**Target Completion:** End of Day 5

---

## Phase 4: Synthesis & Validation (Days 5-6)

### Objective
Finalize notebook, run full validation, and prepare for production.

---

#### §10. Model Comparison & Selection
- [ ] 10.1 Unified performance table
  - Task × model × metrics
  - 6 rows (avg_cpu, waste_fraction, vm_cost, idle, waste_tier, CPU timeseries)
  - Highlight best values

- [ ] 10.2 Best model per business goal
  - Cost optimization triage → waste tier classifier
  - Rightsizing → avg_cpu regressor
  - Anomaly alerting → Isolation Forest
  - Capacity planning → BiGRU
  - Stakeholder communication → SHAP on XGBoost

- [ ] 10.3 Inference time benchmarking
  - Time per 1000 samples
  - Real-time / batch suitability assessment

- [ ] 10.4 Business impact synthesis
  - Rightsizing savings estimate
  - Idle detection potential savings
  - Anomaly alerting impact
  - Auto-scaling savings

**Dependencies:** All prior models  
**Target Lines:** ~200

---

#### §11. Conclusions and Recommendations
- [ ] 11.1 Summary of findings
  - Best models per task
  - Key predictive features (from SHAP)
  - Cluster insights
  - Anomaly findings

- [ ] 11.2 Practical implications
  - Recommended model for deployment
  - Feature monitoring suggestions
  - FinOps workflow integration
  - Threshold recommendations

- [ ] 11.3 Limitations
  - Data from 2019 (outdated)
  - Limited timeseries coverage (25/195 shards)
  - No memory utilization data
  - Pricing approximations
  - Single 30-day trace

- [ ] 11.4 Future work
  - Temporal Fusion Transformer
  - Deep Reinforcement Learning
  - Federated Learning
  - WHA preprocessing
  - Extended timeseries data
  - Multi-region generalization

**Target Lines:** ~150

---

### Full Notebook Execution
- [ ] Run entire notebook end-to-end: `papermill` (real-time output)
- [ ] Verify no errors
- [ ] Check outputs match expectations
- [ ] Save final notebook

---

### Unit Testing
- [ ] Run `pytest app/tests/ -v`
- [ ] All tests pass
- [ ] Coverage > 80%

---

### Git & CI/CD
- [ ] Commit all files: `git add -A && git commit -m "feat: implement predictive analysis notebook"`
- [ ] Push to branch: `git push origin feature/03-predictive-analysis`
- [ ] Verify GitHub Actions CI passes (`ci.yml` auto on push; `notebooks.yml` manual via `workflow_dispatch`)
- [ ] Create PR if needed

---

### Phase 4 Milestones
- [ ] §10-11 complete
- [ ] Full notebook runs without errors
- [ ] All tests passing
- [ ] Code committed and CI green

**Target Completion:** End of Day 6 (May 18, 2026)

---

## Deliverables Checklist

### Source Code
- [ ] `app/src/features.py` (complete, tested)
- [ ] `app/src/models.py` (complete, tested)
- [ ] `app/src/visualize.py` (complete, tested)
- [ ] `notebooks/03_predictive_analysis.ipynb` (11 sections, runnable)

### Testing
- [ ] `app/tests/__init__.py`
- [ ] `app/tests/conftest.py`
- [ ] `app/tests/test_features.py` (5 tests, all passing)
- [ ] `app/tests/test_model.py` (4 tests, all passing)

### Infrastructure
- [ ] `models/` directory with `.gitkeep`
- [ ] `models/run_log.csv` (experiment audit trail)
- [ ] `.github/workflows/ci.yml` (GitHub Actions)
- [ ] `requirements.txt` (updated)
- [ ] `.gitignore` (model patterns)

### Documentation
- [ ] `docs/AZURE_PREDICTIVE_ANALYSIS_PLAN.md` (reference document)
- [ ] `docs/IMPLEMENTATION_PROGRESS.md` (this file)
- [ ] `AGENTS.md` (updated with new commands)

### Models Saved
- [ ] `models/regression/xgboost_avg_cpu.pkl`
- [ ] `models/regression/catboost_waste_fraction.pkl`
- [ ] `models/classification/xgboost_idle.pkl`
- [ ] `models/classification/xgboost_waste_tier.pkl`
- [ ] `models/clustering/kmeans.pkl`
- [ ] `models/timeseries/lstm_cpu.keras` or `gru_cpu.keras`

---

## Progress Log

### 2026-05-12 — Project Initiation
- [x] Read and understand AZURE_PREDICTIVE_ANALYSIS_PLAN.md
- [x] Create IMPLEMENTATION_PROGRESS.md
- [x] Phase 1: Scaffolding — all files created (features.py, models.py, visualize.py, tests, CI, .gitignore)
- [x] Phase 1 bugs fixed: target leakage, empty categorical_features, type corruption, BaseModel.load(), flaky test
- [x] Phase 2: Core Implementation — notebook §1-6 built (Summary, Preparation, Feature Engineering, Regression, Classification, Clustering)
- [x] Phase 3: Advanced Implementation — notebook §7-9 built (Anomaly Detection, Deep Learning Timeseries, SHAP Explainability)
- [x] Phase 4: Synthesis & Validation — notebook §10-11 built (Model Comparison, Conclusions, run_log.csv)

---

## Notes & Assumptions

### Design Principles
1. **Notebook-first:** The notebook is the primary deliverable, academic-quality narrative
2. **Thin imports:** Reusable logic in `app/src/`, imported and tested separately
3. **Reproducibility:** All seeds set to `random_state=42` or `np.random.seed(42)`
4. **CRISP-ML(Q) compliance:** Each notebook section maps to a methodology phase
5. **CAMS DevOps compliance:** Version control, CI/CD, metrics tracking, shared artifacts

### Data Assumptions
- Dataset: Azure Public Dataset V2 (2.7M VMs, 2019 traces)
- Data location: `data/transformed/parquet/` (vmtable, subscriptions, deployments, pricing)
- CPU readings: 25/195 shards available (limited timeseries coverage)
- Pricing: Approximated from deployment data

### Model Assumptions
- Literature-based selections: XGBoost/CatBoost for tabular, LSTM/GRU for timeseries, K-Means for clustering
- Class imbalance: Handled with SMOTE and `class_weight='balanced'`
- Hyperparameters: Tuned via GridSearchCV / RandomizedSearchCV
- Inference: CPU timeseries models batch-only; tree models real-time

---

## References

- **CRISP-ML(Q):** https://crisp-ml.org/
- **CAMS DevOps:** (internal reference)
- **Plan Document:** `docs/AZURE_PREDICTIVE_ANALYSIS_PLAN.md`
- **Literature:** `docs/Recommended_models_from_literature.md`

---

**Last Updated:** 2026-05-12  
**Status:** Planning  
**Next Action:** Begin Phase 1 scaffolding
