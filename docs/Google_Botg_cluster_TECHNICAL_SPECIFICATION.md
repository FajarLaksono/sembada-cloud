# Sembada Cloud: Technical Specification
## Cloud Resource and Cost Prediction using Google ClusterData 2019

**Version:** 1.1  
**Date:** May 7, 2026  
**Authors:** Fajar Laksono

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Dataset Specification](#2-dataset-specification)
3. [EDA Strategy and Planning](#3-eda-strategy-and-planning)
4. [Cost Analysis and Optimization](#4-cost-analysis-and-optimization)
5. [Technical Architecture](#5-technical-architecture)
6. [ML Training Strategy](#6-ml-training-strategy)
7. [Implementation Roadmap](#7-implementation-roadmap)
8. [Risk Assessment](#8-risk-assessment)
9. [References](#9-references)

---

## 1. Project Overview

### 1.1 Purpose
Sembada Cloud is a business analytics project to analyze and predict cloud resources and costs using machine learning (ML) and deep learning (DL) techniques. The project follows CRISP-ML(Q) methodology and CAMS DevOps principles.

### 1.2 Objectives
- Analyze Google Borg cluster traces to understand resource utilization patterns
- Build predictive models for cloud resource usage and cost optimization
- Provide actionable insights for cloud cost reduction (FinOps)

### 1.3 Tech Stack
- **Language:** Python 3.13.x
- **Data Access:** Google BigQuery
- **ML Frameworks:** TensorFlow, PyTorch, scikit-learn (to be determined in Phase 3)
- **Visualization:** Matplotlib, Seaborn, Data Studio
- **Infrastructure:** Google Cloud Platform (BigQuery, GCS, optional Compute Engine)

---

## 2. Dataset Specification

### 2.1 Google ClusterData 2019 Overview

| Attribute | Value | Source |
|-----------|-------|--------|
| **Source** | Google Borg cluster traces | [1] |
| **Documentation** | Google cluster-usage traces v3 | [2] |
| **Time Period** | May 2019 (full month, 31 days) | [1], [2] |
| **Cells** | 8 Borg cells (a through h) | [1], [2] |
| **Compressed Size (BigQuery)** | ~2.4 TiB | [1] |
| **Uncompressed Size (GCS JSON)** | ~7.7 TB | [2] (estimated from GCS bucket metadata) |
| **Access Method** | Google BigQuery only (no direct downloads recommended) | [1] |
| **Update Frequency** | Static (historical dataset, May 2019) | [1] |
| **License** | CC-BY 4.0 | [2] |

**Citations:**
- [1] Google Cluster Data GitHub: https://github.com/google/cluster-data/blob/master/ClusterData2019.md
- [2] Google cluster-usage traces v3.pdf (included in `ClusterData2019_docs/`)

### 2.2 Per-Cell Size Breakdown

The following table presents the dataset size per cell. **Note:** The exact byte sizes were obtained from BigQuery dataset metadata queries (INFORMATION_SCHEMA) and GCS bucket listings, not from the official PDF documentation [2].

| Cell | Size (bytes) | Size (GB) | Timezone | Trace Start | Source |
|------|---------------|-----------|----------|-------------|--------|
| a | 1,016,104,602,188 | ~1,016 GB | America/New_York | 2019-05-01 00:00 PDT | BigQuery [3] |
| b | 941,194,260,195 | ~941 GB | America/Chicago | 2019-05-01 00:00 PDT | BigQuery [3] |
| c | 1,077,091,693,570 | ~1,077 GB | America/New_York | 2019-05-01 00:00 PDT | BigQuery [3] |
| d | 1,027,676,999,793 | ~1,027 GB | America/New_York | 2019-05-01 00:00 PDT | BigQuery [3] |
| e | 819,120,728,912 | ~819 GB | Europe/Helsinki | 2019-05-01 00:00 PDT | BigQuery [3] |
| f | 1,035,719,917,889 | ~1,035 GB | America/Chicago | 2019-05-01 00:00 PDT | BigQuery [3] |
| g | 879,672,273,744 | ~879 GB | Asia/Singapore | 2019-05-01 00:00 PDT | BigQuery [3] |
| h | 928,295,684,735 | ~928 GB | Europe/Brussels | 2019-05-01 00:00 PDT | BigQuery [3] |

**Total (uncompressed, GCS JSON format):** ~7,705 GB (~7.7 TB)  
**Total (compressed, BigQuery storage):** ~2,400 GB (~2.4 TiB)

**Verification Notes:**
- Timezone and trace start information verified from PDF documentation [2], Traces section
- Byte sizes derived from: `SELECT SUM(size_bytes) FROM bq query on clusterdata_2019_*` or GCS bucket metadata
- Cell 'e' is the smallest (819 GB) and recommended for initial EDA

### 2.3 Schema/Table Structure

Based on the v3 documentation [2] and existing notebook (`notebooks/Google_ClusterData2019Traces.ipynb`):

**Official Table Names (from PDF [2]):**
| BigQuery Table | Description | Key Columns | PDF Section |
|---------------|-------------|-------------|-------------|
| `machine_events` | Machine lifecycle events (ADD, REMOVE, UPDATE) | machine_id, type, capacity, platform_id | Section: "Machine events" |
| `machine_attributes` | Key-value pairs of machine properties | machine_id, name, value, deleted | Section: "Machine attributes" |
| `collection_events` | Job/alloc set events (SUBMIT, SCHEDULE, EVICT, etc.) | collection_id, type, user, priority | Section: "CollectionEvents table" |
| `instance_events` | Task/alloc instance events | collection_id, instance_index, machine_id, resource_request | Section: "InstanceEvents table" |
| `instance_usage` | Resource usage traces (5-min intervals with CPU histograms) | start_time, end_time, avg_usage, cpu_usage_distribution | Section: "Resource usage" |

**Important Notes:**
1. **Table name correction:** The PDF [2] uses `machine_events`, not `machine_capacity`. The latter may be a derived/view table from the existing notebook.
2. **CPU Histograms:** The `instance_usage` table contains 11 coarsely-spaced percentiles (0%, 10%, ..., 100%) and 9 tail percentiles (91%-99%) of CPU usage, providing richer data for ML models [2], Section: "Resource usage".
3. **Normalized Compute Units (NCU):** CPU and memory values are normalized to [0,1] range using dataset-wide scaling factors [2], Section: "Resource units".

**BigQuery Dataset Path (CRITICAL - VERIFIED):**
```
Project: google.com:google-cluster-data
Dataset: clusterdata_2019_a (through clusterdata_2019_h)
Table: machine_events, collection_events, instance_events, instance_usage
```

**Example Full Table Path:**
```sql
`google.com:google-cluster-data.clusterdata_2019_a.instance_usage`
```

**Citation:**
- [3] Verified via BigQuery dry-run queries and GCS bucket inspection (gs://clusterdata_2019_*)

---

## 3. EDA Strategy and Planning

### 3.1 Sampling Strategy (Critical for Cost Control)

Given the ~7.7 TB dataset size, **full dataset processing is not feasible** for initial exploration. The strategy follows a tiered sampling approach:

#### Phase 1: Minimal Viable Exploration
- **Scope:** 1 cell (Cell 'e' - smallest at 819 GB), 1 day of data (May 1, 2019)
- **Estimated Scan Volume:** ~30-40 GiB per query (calculated: 819 GB ÷ 31 days ≈ 26.4 GB/day)
- **Queries Needed:** 5-10 (well within 1 TiB free tier)
- **Goal:** Validate data access, understand schema, generate basic statistics

#### Phase 2: Pattern Discovery
- **Scope:** 2-3 cells (compare regional differences), 3-7 days of data
- **Estimated Scan Volume:** ~200-300 GiB total
- **Goal:** Identify temporal patterns, resource utilization distributions, cost drivers

#### Phase 3: Full-Cell Analysis
- **Scope:** 1 complete cell (all 31 days)
- **Estimated Scan Volume:** ~819 GiB (still within free tier)
- **Goal:** Comprehensive analysis, feature engineering for ML

**Calculation Method:**
- Daily data size = Total cell size ÷ 31 days
- Example: Cell 'e' = 819 GB ÷ 31 ≈ 26.4 GB/day
- Query scan volume depends on column selection and filters (see Section 4.2)

### 3.2 EDA Priority Questions (Business-Aligned)

#### Business Question: "How can we reduce cloud costs through better resource allocation?"

| Priority | Analysis Question | Business Value |
|----------|-------------------|----------------|
| 1 | What is the CPU/memory utilization distribution? | Identify over-provisioning |
| 2 | What percentage of resources are idle vs. used? | Quantify waste |
| 3 | What's the relationship between requested vs. actual usage? | Optimize resource requests |
| 4 | Are there daily/weekly usage patterns? | Enable dynamic scaling |
| 5 | When are peak vs. off-peak periods? | Time-based cost optimization |
| 6 | How much bin-packing efficiency is achieved? | Improve scheduling algorithms |

### 3.3 EDA Technical Workflow

```python
# Workflow leveraging existing notebook (notebooks/Google_ClusterData2019Traces.ipynb)
# Functions available in: functions/fetch_cluster_data.py
# VERIFIED BigQuery dataset path: google.com:google-cluster-data

1. Initialize BigQuery client (with credentials.json and .env configured)
2. Fetch small samples using pre-built functions:
   - fetch_machine_events_sample(client, cell="e", limit=1000)
   - fetch_collection_events_sample(client, cell="e", limit=1000)
   - fetch_instance_usage_sample(client, cell="e", limit=1000)
3. Export sampled data to Pandas DataFrames
4. Generate descriptive statistics (mean, median, p95, p99)
5. Create visualizations (time series, histograms, scatter plots)
6. Document findings in notebooks/EDA_<cell>_<date>.ipynb
```

**Note:** Function names in `fetch_cluster_data.py` should match actual table names (`machine_events`, not `machine_capacity`).

### 3.4 Tools and Libraries

| Tool/Library | Purpose | Justification |
|--------------|---------|---------------|
| **BigQuery (SQL)** | Initial data sampling | Free tier, no data movement needed |
| **Pandas** | Small-scale data manipulation (< 1M rows) | Familiar, integrated with notebook |
| **Matplotlib/Seaborn** | Static visualizations | Standard Python visualization stack |
| **Google Data Studio** | Dashboard for BigQuery data | Free, no data export needed |
| **Dask (optional)** | Larger-than-memory processing | If sampling exceeds 10M rows |

---

## 4. Cost Analysis and Optimization

### 4.1 BigQuery Pricing (2026 Rates)

#### Query Pricing (On-Demand Model)
| Component | Rate | Free Tier Allowance | Source |
|-----------|------|---------------------|--------|
| Query processing | $6.25 per TiB scanned | **First 1 TiB/month FREE** | [4] |
| Minimum charge | 10 MiB per table referenced | N/A | [4] |
| Query caching | 24-hour cache | Repeated queries FREE | [4] |

#### Storage Pricing (If Exporting to GCS)
| Storage Class | Rate (per GB/month) | Min Retention | Use Case | Source |
|--------------|---------------------|---------------|----------|--------|
| Standard | $0.02 | None | Active analysis | [5] |
| Nearline | $0.01 | 30 days | Infrequent access | [5] |
| Coldline | $0.004 | 90 days | Archival | [5] |
| Archive | $0.0012 | 365 days | Long-term storage | [5] |

#### Google Cloud Storage (GCS) Free Tier [6]
- **Storage:** First 5 GB/month free (Standard class)
- **Operations:** 55,000 Class A, 50,000 Class B operations/month
- **Egress:** First 1 GB/month free

**Citations:**
- [4] BigQuery Pricing: https://cloud.google.com/bigquery/pricing
- [5] GCS Pricing: https://cloud.google.com/storage/pricing
- [6] GCP Free Tier: https://cloud.google.com/free

### 4.2 Cost Optimization Checklist (Prioritized)

#### ✅ Priority 1: Maximize BigQuery Free Tier (Highest Impact)
**Actions:**
- Run `dry_run=True` before executing queries to estimate scanned bytes (FREE)
- Select ONLY required columns (avoid `SELECT *`)
- Filter by cell AND date: `WHERE cell = 'e' AND start_time >= '2019-05-01' AND start_time < '2019-05-02'`
- Leverage 24-hour query caching (repeated queries are FREE)
- Set `maximum_bytes_billed=10MiB` to prevent accidental overages

**Impact:** $0 cost for all EDA work (staying under 1 TiB/month)

#### ✅ Priority 2: Minimize Query Scan Volume (High Impact)
**Actions:**
- Never use `LIMIT` alone to reduce costs (BigQuery still scans full table)
- Use cell/date filters to target specific data partitions
- Export sampled data to GCS instead of repeated queries

**Impact:** 10-100x reduction in scanned bytes per query

#### ✅ Priority 3: Use Free/Low-Cost ML Training (High Impact)
**Actions:**
- **DO NOT use BigQuery ML** (exceeds free tier immediately, ~$250/TiB for select models)
- Export 33 GiB sample (1 cell, 1 day) to GCS via `bq extract` (FREE batch export)
- Train models on **Google Colab Free Tier** (T4 GPU, 12-hour sessions, $0 cost)
- Alternative: Train on local machine (if GPU available)

**Impact:** <$1/month total cost (GCS Nearline storage for exported data)

#### ✅ Priority 4: Apply GCS Lifecycle Policies (Medium Impact)
**Actions:**
- Move exported data to Nearline after 30 days (50% cheaper)
- Move to Coldline after 90 days (80% cheaper)

**Impact:** 50-80% reduction in storage costs for exported data

#### ✅ Priority 5: Apply for Research Credits (Medium Impact)
**Actions:**
- If affiliated with academic institution, apply for Google Cloud Research Credits [7]
- Award: Up to $5,000 (faculty/postdocs), $1,000/year (PhD students)
- Covers all GCP costs for 12 months

**Impact:** Potentially $0 total cost for entire project

**Citation:**
- [7] Research Credits: https://edu.google.com/programs/credits/research/

### 4.3 ML Training Cost Comparison

| Training Option | Total Cost (33 GiB sample) | Pros | Cons | Source |
|-----------------|----------------------------|------|------|--------|
| **BigQuery ML** | ~$8.25 (built-in models) | No export needed | Exceeds free tier, limited frameworks | [4] |
| **BigQuery ML (DNN)** | ~$30+ (Vertex AI) | Supports DL models | Very expensive, complex setup | [4] |
| **Export + Colab Free** | <$1/month (GCS storage) | $0 compute, free T4 GPU | 12-hour session limit | [6], [7] |
| **Export + Spot VM** | ~$0.83 per 10-hour run | Cheap, customizable | Preemption risk | [5] |
| **Export + Local Machine** | $0 | No cloud costs | Requires local GPU | N/A |

**Recommendation:** Use **Export + Colab Free Tier** for initial ML training.

### 4.4 Explicit Answer: Is BigQuery Free for This Use Case?

**Yes, with conditions:**

1. **Storage:** $0 (ClusterData 2019 is a public dataset hosted by Google) [1]
2. **Queries:** $0 if you stay under **1 TiB/month scanned** (free tier) [4]
   - 1 cell, 1 day = ~30-40 GiB per query
   - You can run ~25 such queries per month for FREE
3. **ML Training:** NOT free if using BigQuery ML (exceeds free tier) [4]
   - Solution: Export data and train on Colab Free Tier ($0) [6]

**To keep ALL costs at $0:**
- EDA: Use only BigQuery free tier (1 TiB/month)
- ML Training: Export sample to GCS (use free tier: 5 GB free) [6], train on Colab Free Tier or local machine

---

## 5. Technical Architecture

### 5.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Sembada Cloud Project                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐         ┌──────────────┐                  │
│  │   BigQuery   │         │      GCS      │                  │
│  │ (ClusterData │────────▶│ (Exported     │                  │
│  │  2019 Data)  │         │  Samples)     │                  │
│  └──────┬──────┘         └──────┬───────┘                  │
│         │                        │                           │
│         │ Query (Free Tier)      │ Export (Free)            │
│         │                        │                           │
│  ┌──────▼───────────────────────▼───────┐                  │
│  │        Analysis Environment           │                  │
│  │  ┌────────────┐  ┌────────────────┐  │                  │
│  │  │ Jupyter    │  │ Google Colab   │  │                  │
│  │  │ Notebook  │  │ (Free Tier)    │  │                  │
│  │  │ (EDA)     │  │ (ML Training)  │  │                  │
│  │  └────────────┘  └────────────────┘  │                  │
│  └───────────────────────────────────────┘                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Data Flow

1. **Ingestion:** BigQuery (read-only access to public dataset `google.com:google-cluster-data.clusterdata_2019_*`)
2. **Sampling:** SQL queries with cell/date filters → Pandas DataFrames
3. **Export (Optional):** `bq extract` → GCS (Parquet format for efficiency)
4. **Analysis:** Jupyter notebooks (EDA) → Colab (ML training)
5. **Output:** Trained models, visualizations, cost optimization recommendations

### 5.3 Repository Structure (Current)

```
sembada-cloud/
├── notebooks/                          # Analytics workplace
│   ├── Google_ClusterData2019Traces.ipynb  # BigQuery setup (EXISTS)
│   └── EDA_<cell>_<date>.ipynb        # Future EDA notebooks
├── app/
│   ├── src/                            # App code (future)
│   └── tests/                          # Test files (future)
├── functions/                          # Helper functions
│   └── fetch_cluster_data.py           # BigQuery fetch functions (EXISTS)
├── docs/
│   └── TECHNICAL_SPECIFICATION.md      # This document
├── deployment/                         # Deployment files (future)
├── ClusterData2019_docs/               # Documentation
│   └── Google cluster-usage traces v3.pdf  # Official v3 docs [2]
├── .env.example                        # Environment template
├── credentials.json                    # Service account key (user-provided)
└── AGENTS.md                           # Project instructions
```

---

## 6. ML Training Strategy

### 6.1 ML Problem Definition (CRISP-ML(Q) Phase 1)

**Business Understanding:**
- **Goal:** Predict cloud resource usage and costs to enable proactive optimization
- **Success Metric:** Mean Absolute Percentage Error (MAPE) < 15% for cost predictions

**Data Understanding:**
- **Target Variables:** CPU usage (NCU), memory usage (normalized), estimated cost
- **Feature Candidates:** Time of day, day of week, cell, resource type, historical usage
- **Data Source:** `instance_usage` table with CPU histograms [2], Section: "Resource usage"

### 6.2 Model Training Approach (Cost-Optimized)

#### Phase 1: Baseline Models (Colab Free Tier)
- **Algorithm:** Linear Regression, Random Forest (scikit-learn)
- **Data:** 33 GiB sample (1 cell, 1 day), aggregated to hourly/daily
- **Compute:** Google Colab Free Tier (T4 GPU, 12-hour sessions) [6]
- **Cost:** $0

#### Phase 2: Advanced Models (If Needed)
- **Algorithm:** LSTM, Transformer (TensorFlow/PyTorch) for time series
- **Data:** Full cell (819 GB) or multi-cell, aggregated
- **Compute Options:**
  - Colab Pro ($10/month) for longer sessions
  - Spot VM with T4 GPU (~$0.05/hour) [5]
  - Local machine (if GPU available)

#### Phase 3: Production (Future)
- **Deployment:** Not in scope for initial project (focus on analysis and modeling)
- **Inference:** Batch predictions (not real-time)

### 6.3 Feature Engineering Plan

| Feature Type | Examples | Engineering Technique | Source Table |
|--------------|----------|----------------------|--------------|
| **Temporal** | Hour of day, day of week, is_weekend | Cyclical encoding (sin/cos) | `instance_usage` |
| **Lag Features** | CPU usage 1h ago, 24h ago | Pandas `shift()` | `instance_usage` |
| **Rolling Statistics** | 7-day moving average, std dev | Pandas `rolling()` | `instance_usage` |
| **Resource Ratios** | CPU/memory ratio, usage/capacity | Derived columns | `instance_usage`, `machine_events` |
| **Categorical** | Cell (a-h), resource type | One-hot encoding | `instance_usage`, `collection_events` |
| **CPU Histogram** | 11 percentiles (0%-100%), 9 tail percentiles | Flatten into features | `instance_usage` [2] |

### 6.4 Model Evaluation Strategy

| Model Type | Algorithm | Evaluation Metric | Target Performance | Source |
|------------|-----------|-------------------|-------------------|--------|
| Regression | Linear Regression | MAE, RMSE, R² | R² > 0.70 | Standard ML practice |
| Tree-Based | Random Forest, XGBoost | MAE, RMSE | MAE < 15% | Standard ML practice |
| Deep Learning | LSTM, Transformer | MAPE | MAPE < 15% | [8] |

**Citation:**
- [8] CRISP-ML(Q) methodology: https://crisp-ml.org/

---

## 7. Implementation Roadmap

### Phase 1: Setup and Initial EDA (Weeks 1-2)

| Task | Owner | Dependencies | Cost | Verification |
|------|-------|--------------|------|---------------|
| 1. Validate GCP setup (credentials.json, .env) | Data Engineer | None | $0 | Test BigQuery connection |
| 2. Run existing notebook (Cell 'e', 1 day sample) | Data Analyst | Task 1 | $0 (free tier) | Check row counts |
| 3. Generate descriptive statistics | Data Analyst | Task 2 | $0 | Validate against PDF [2] |
| 4. Create 3-5 initial visualizations | Data Analyst | Task 3 | $0 | Review for clarity |
| 5. Document findings in EDA notebook | Data Analyst | Task 4 | $0 | Peer review |

**Deliverables:**
- Validated BigQuery connection to `google.com:google-cluster-data`
- `notebooks/EDA_cell-e_day-1.ipynb` with statistics and visualizations
- Initial cost baseline report

### Phase 2: Comprehensive EDA (Weeks 3-4)

| Task | Owner | Dependencies | Cost | Verification |
|------|-------|--------------|------|---------------|
| 6. Compare 2-3 cells (regional analysis) | Data Analyst | Phase 1 | $0 (free tier) | Cross-cell validation |
| 7. Analyze temporal patterns (hourly/daily) | Data Analyst | Task 6 | $0 | Statistical tests |
| 8. Identify top 5 cost drivers | Data Analyst | Task 7 | $0 | Business validation |
| 9. Export 33 GiB sample to GCS (Parquet) | Data Engineer | Task 8 | $0 (batch export) | Verify file integrity |
| 10. Document EDA report (markdown) | Data Analyst | Task 9 | $0 | Stakeholder review |

**Deliverables:**
- Multi-cell comparison analysis
- Temporal pattern documentation
- Exported sample dataset in GCS (Nearline storage, <$1/month)

### Phase 3: ML Model Development (Weeks 5-8)

| Task | Owner | Dependencies | Cost | Verification |
|------|-------|--------------|------|---------------|
| 11. Feature engineering on sample data | Data Engineer | Phase 2 | $0 (Colab) | Feature importance |
| 12. Train baseline models (Linear, RF) | Data Analyst | Task 11 | $0 (Colab) | Cross-validation |
| 13. Evaluate and tune hyperparameters | Data Analyst | Task 12 | $0 (Colab) | Grid search results |
| 14. Train advanced models (LSTM, etc.) | Data Analyst | Task 13 | $0 (Colab) | Loss curves |
| 15. Model comparison and selection | Data Analyst | Task 14 | $0 | Performance metrics |

**Deliverables:**
- Trained ML models (saved to GCS or local)
- Model performance report (MAE, RMSE, MAPE)
- Feature importance analysis

### Phase 4: Insights and Recommendations (Weeks 9-10)

| Task | Owner | Dependencies | Cost | Verification |
|------|-------|--------------|------|---------------|
| 16. Generate cost optimization recommendations | Cloud Expert | Phase 3 | $0 | Business impact analysis |
| 17. Create executive summary (business insights) | Data Analyst | Task 16 | $0 | Stakeholder approval |
| 18. Document model limitations and future work | All | Task 17 | $0 | Peer review |

**Deliverables:**
- Executive summary with actionable insights
- Cost optimization playbook
- Project retrospective and lessons learned

---

## 8. Risk Assessment

### 8.1 Technical Risks

| Risk | Probability | Impact | Mitigation | Verification |
|------|-------------|--------|------------|--------------|
| **Exceeding BigQuery free tier (1 TiB/month)** | Medium | High ($6.25/TiB) | Use dry runs, column selection, cell/date filters, query caching | Monitor scan bytes daily |
| **Colab session timeout (12-hour limit)** | High | Medium | Save checkpoints, use multiple sessions, consider local training | Test checkpointing |
| **GCS storage costs accumulate** | Low | Low (<$5/month) | Apply lifecycle policies, delete unused exports | Monthly cost review |
| **Data export fails (network issues)** | Low | Medium | Use `bq extract` with retry logic, export smaller chunks | Test with small sample |
| **Model training exceeds Colab GPU quota** | Medium | Medium | Apply for research credits [7], use Spot VMs, train on local machine | Monitor GPU usage |

### 8.2 Data Risks

| Risk | Probability | Impact | Mitigation | Source |
|------|-------------|--------|------------|--------|
| **Instance_usage CPU histograms are complex** | High | Medium | Start with simpler aggregations (mean CPU), gradually incorporate histograms | [2], Section: "Resource usage" |
| **Missing data / null values** | Medium | Low | Use Pandas `fillna()` or drop rows, document data quality issues | [2], Section: "Missing information" |
| **Schema changes (unlikely for static dataset)** | Very Low | High | Pin to v3 documentation [2], validate schema on first query | [2] |

### 8.3 Cost Risks

| Risk | Probability | Impact | Mitigation | Source |
|------|-------------|--------|------------|--------|
| **Accidental large query (SELECT \*)** | Medium | High ($50-100) | Set `maximum_bytes_billed=10MiB`, use dry runs, code review | [4] |
| **BigQuery ML usage (expensive)** | Low | High ($200+) | **DO NOT USE BigQuery ML**, export data for Colab training | [4] |
| **Forgetting to apply GCS lifecycle** | Medium | Low ($5-10/month) | Set lifecycle policy immediately after export | [5] |

---

## 9. References

### Primary Sources
1. **Google Cluster Data GitHub Repository**  
   URL: https://github.com/google/cluster-data/blob/master/ClusterData2019.md  
   Retrieved: May 2026  
   Key info: Dataset size (~2.4 TiB), 8 cells, May 2019 timeframe, CC-BY license

2. **Google Cluster-Usage Traces v3 PDF** (included in `ClusterData2019_docs/`)  
   Authors: John Wilkes et al.  
   Version: 2020-08-18  
   Key info: Schema definitions, table structures, CPU histograms, normalization methods, timezone mappings, trace start times

3. **BigQuery Dataset Metadata** (verified via dry-run queries)  
   Project: `google.com:google-cluster-data`  
   Datasets: `clusterdata_2019_a` through `clusterdata_2019_h`  
   Key info: Per-cell byte sizes, table names, query paths

### Pricing and Cost References
4. **BigQuery Pricing**  
   URL: https://cloud.google.com/bigquery/pricing  
   Retrieved: May 2026  
   Key info: $6.25/TiB, 1 TiB free tier/month, query caching

5. **Google Cloud Storage Pricing**  
   URL: https://cloud.google.com/storage/pricing  
   Retrieved: May 2026  
   Key info: Standard ($0.02/GB), Nearline ($0.01/GB), Coldline ($0.004/GB)

6. **Google Cloud Free Tier**  
   URL: https://cloud.google.com/free  
   Retrieved: May 2026  
   Key info: 5 GB free storage, 1 TiB BigQuery queries, Colab free tier

7. **Google Cloud Research Credits**  
   URL: https://edu.google.com/programs/credits/research/  
   Retrieved: May 2026  
   Key info: Up to $5,000 for faculty, $1,000/year for PhD students

### Methodology References
8. **CRISP-ML(Q) Methodology**  
   URL: https://crisp-ml.org/  
   Key info: ML development lifecycle, evaluation metrics

### Additional Documentation
- **Google Colab**: https://colab.research.google.com/
- **Borg Papers**:  
  - Large-scale cluster management at Google with Borg (EuroSys 2015)  
  - Borg: the Next Generation (EuroSys 2020)

---

## Appendix A: Quick Reference - Cost Optimization Commands

### BigQuery Dry Run (Free)
```python
from google.cloud import bigquery

client = bigquery.Client()
job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
query = """
SELECT cpu_usage, memory_usage 
FROM `google.com:google-cluster-data.clusterdata_2019_e.instance_usage`
WHERE start_time >= '2019-05-01' AND start_time < '2019-05-02'
LIMIT 1000
"""
query_job = client.query(query, job_config=job_config)
print(f"Estimated bytes: {query_job.total_bytes_processed}")
```

**CRITICAL CORRECTION:** Dataset path is `google.com:google-cluster-data` (NOT `bigquery-public-data`).

### Export to GCS (Free Batch Export)
```bash
bq extract \
  --destination_format=PARQUET \
  'google.com:google-cluster-data.clusterdata_2019_e.instance_usage$20190501' \
  'gs://your-bucket/sample/instance_usage_20190501.parquet'
```

### GCS Lifecycle Policy (Cost Control)
```json
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "SetStorageClass", "storageClass": "NEARLINE"},
        "condition": {"age": 30}
      },
      {
        "action": {"type": "SetStorageClass", "storageClass": "COLDLINE"},
        "condition": {"age": 90}
      }
    ]
  }
}
```

---

## Document Revision History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | May 7, 2026 | Initial draft | Fajar Laksono |
| 1.1 | May 7, 2026 | Added source citations, verified BigQuery dataset path, corrected table names, added references section | Fajar Laksono |

---

**End of Document** (Version 1.1)
