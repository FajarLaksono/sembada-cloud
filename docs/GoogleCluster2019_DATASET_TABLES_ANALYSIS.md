# ClusterData 2019: Table Analysis for Cloud Resource Prediction

**Document Type:** Research Reference  
**Dataset:** Google ClusterData 2019 (Version 3)  
**Primary Source:** Google cluster-usage traces v3.pdf  
**Date:** May 7, 2026  
**Author:** Fajar Laksono  

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Dataset Overview](#2-dataset-overview)
3. [Table Descriptions and Schema](#3-table-descriptions-and-schema)
4. [Usefulness for Cloud Resource Prediction](#4-usefulness-for-cloud-resource-prediction)
5. [Recommended Table Usage Strategies](#5-recommended-table-usage-strategies)
6. [Key Insights from Documentation](#6-key-insights-from-documentation)
7. [SQL Examples for ML Preparation](#7-sql-examples-for-ml-preparation)
8. [Summary and Recommendations](#8-summary-and-recommendations)
9. [References](#9-references)

---

## 1. Introduction

This document provides a comprehensive analysis of the Google ClusterData 2019 dataset tables, specifically evaluating their utility for machine learning (ML) and deep learning (DL) models designed to predict cloud resource usage and optimize costs.

The analysis is based on the **Google cluster-usage traces v3** documentation (2020-08-18 update), which describes the schema, semantics, and data format of the Borg cluster traces collected during May 2019.

### 1.1 Research Context

- **Project:** Sembada Cloud - Cloud Resource and Cost Prediction
- **Methodology:** CRISP-ML(Q)
- **Objective:** Predict CPU and memory usage to enable proactive resource optimization
- **Dataset Size:** ~2.4 TiB compressed (~7.7 TB uncompressed)
- **Coverage:** 8 Borg cells (a through h), May 2019 (full month)

---

## 2. Dataset Overview

### 2.1 Dataset Characteristics

| Attribute | Value | Source |
|-----------|-------|--------|
| **Time Period** | May 2019 (31 days) | [1] |
| **Cells** | 8 Borg cells (a-h) | [1], [2] |
| **Compressed Size (BigQuery)** | ~2.4 TiB | [1] |
| **Uncompressed Size (GCS JSON)** | ~7.7 TB | [2] |
| **Access Method** | Google BigQuery (`google.com:google-cluster-data`) | [2] |
| **Normalization** | CPU and memory rescaled to [0,1] (NCU - Normalized Compute Units) | [2] |
| **License** | CC-BY 4.0 | [2] |

### 2.2 BigQuery Dataset Structure

```
Project: google.com:google-cluster-data
Datasets: clusterdata_2019_a through clusterdata_2019_h
Tables per dataset: 5 (machine_events, machine_attributes, collection_events, instance_events, instance_usage)
```

**Citation:**
- [1] Google Cluster Data GitHub: https://github.com/google/cluster-data/blob/master/ClusterData2019.md
- [2] Google cluster-usage traces v3.pdf (included in `ClusterData2019_docs/`)

---

## 3. Table Descriptions and Schema

The ClusterData 2019 dataset contains **5 tables** per cell. Each table captures different aspects of cluster operations, from machine lifecycle to detailed resource usage measurements.

### 3.1 MachineEvents Table

**PDF Section:** "Machine events" (Page 5)  
**Usefulness for Prediction:** ⭐⭐⭐ **Medium**

#### Schema

| Field | Type | Description |
|-------|------|-------------|
| `time` | INT64 | Timestamp (microseconds since 600s before trace start) |
| `machine_id` | INT64 | Unique 64-bit machine identifier |
| `type` | STRING | Event type: ADD, REMOVE, UPDATE |
| `switch_id` | STRING | Network switch ID (top-of-rack) |
| `capacity` | RECORD | Resources structure (CPU in NCU, memory normalized) |
| `platform_id` | STRING | Opaque string (microarchitecture/chipset version) |
| `missing_data_reason` | STRING | NONE or SNAPSHOT_BUT_NO_TRANSITION |

#### Event Types

- **ADD:** Machine became available (all machines have ADD event, many at time=0)
- **REMOVE:** Machine removed (failure or maintenance)
- **UPDATE:** Machine capacity changed

#### Relevance to Prediction

- **Capacity bounds:** `capacity.cpu` and `capacity.memory` define maximum available resources
- **Utilization ratios:** Can calculate `usage/capacity` for efficiency metrics
- **Hardware features:** `platform_id` may explain performance variations (though obfuscated)
- **Availability patterns:** REMOVE events indicate capacity fluctuations

---

### 3.2 MachineAttributes Table

**PDF Section:** "Machine attributes" (Page 7)  
**Usefulness for Prediction:** ⭐⭐ **Low-Medium**

#### Schema

| Field | Type | Description |
|-------|------|-------------|
| `time` | INT64 | Timestamp |
| `machine_id` | INT64 | Machine identifier |
| `name` | STRING | Obfuscated attribute name (kernel version, clock speed, etc.) |
| `value` | STRING/INTEGER | Hashed string or integer |
| `deleted` | BOOLEAN | Attribute being deleted? |

#### Attribute Examples (from PDF)

- Kernel version
- Clock speed
- Presence of external IP address
- Memory technology (DDR2 vs. DDR3)

#### Relevance to Prediction

- **Additional features:** May improve model accuracy if attributes are predictive
- **Limitation:** Values are obfuscated (hashed), reducing interpretability
- **Use case:** Feature engineering for hardware-specific patterns

---

### 3.3 CollectionEvents Table

**PDF Section:** "CollectionEvents table" (Page 11)  
**Usefulness for Prediction:** ⭐⭐⭐⭐ **High**

#### Schema

| Field | Type | Description |
|-------|------|-------------|
| `time` | INT64 | Timestamp |
| `type` | INT64 | Event type (see below) |
| `collection_id` | INT64 | Unique collection ID (jobs + alloc sets) |
| `scheduling_class` | INT64 | Latency-sensitivity (0-3, 3=most sensitive) |
| `missing_type` | STRING | Reason for synthesized records |
| `collection_type` | INT64 | 0=job, 1=alloc set |
| `priority` | INT64 | Small integer (0-500+, higher=higher priority) |
| `alloc_collection_id` | INT64 | Parent alloc set ID (jobs only) |
| `user` | STRING | Obfuscated user name (base64-encoded) |
| `collection_name` | STRING | Hash of complete collection name |
| `collection_logical_name` | STRING | Hash of purpose-based name |
| `parent_collection_id` | INT64 | Parent collection (master/worker patterns) |
| `start_after_collection_ids` | ARRAY | Dependencies (pipeline jobs) |
| `max_per_machine` | INT64 | Placement constraint |
| `max_per_switch` | INT64 | Placement constraint |
| `vertical_scaling` | STRING | AUTO_OFF/CONSTRAINED/FULLY_AUTOMATED |
| `scheduler` | STRING | DEFAULT or BATCH |

#### Event Types (State Transitions)

| Event | Description |
|-------|-------------|
| SUBMIT | Collection submitted to cluster manager |
| QUEUE | Deferred until scheduler acts |
| ENABLE | Eligible for scheduling |
| SCHEDULE | Scheduled on a machine |
| EVICT | Descheduled (higher priority, overcommit, machine failure) |
| FAIL | Descheduled (program failure: segfault, OOM) |
| FINISH | Completed normally |
| KILL | Cancelled by user/driver/parent |
| LOST | Presumed ended, no termination record |
| UPDATE_PENDING | Updated while waiting (resources, constraints) |
| UPDATE_RUNNING | Updated while running |

#### Priority Tiers (from PDF)

| Range | Tier | Description |
|-------|------|-------------|
| ≤99 | Free tier | Lowest priority, little internal charging |
| 100-115 | Best-effort Batch | Managed by batch scheduler, no SLOs |
| 116-119 | Mid-tier | SLOs between free and production |
| 120-359 | Production | Highest priority, eviction protection |
| ≥360 | Monitoring | Monitor health of lower-priority jobs |

#### Relevance to Prediction

- **Resource requests:** Via join with `instance_events.resource_request`
- **Priority & scheduling_class:** Predict preemption likelihood and runtime
- **Collection type:** Jobs vs. alloc sets have different behaviors
- **Vertical scaling:** Indicates auto-scaling behavior
- **User patterns:** Hashed usernames may reveal usage patterns
- **Master/worker patterns:** `parent_collection_id` for MapReduce-style jobs

---

### 3.4 InstanceEvents Table

**PDF Section:** "InstanceEvents table" (Page 13)  
**Usefulness for Prediction:** ⭐⭐⭐⭐ **Very High**

#### Schema

| Field | Type | Description |
|-------|------|-------------|
| `time` | INT64 | Timestamp |
| `type` | INT64 | Event type (same as CollectionEvents) |
| `collection_id` | INT64 | Parent collection |
| `scheduling_class` | INT64 | Latency-sensitivity |
| `missing_type` | STRING | Missing data indicator |
| `collection_type` | INT64 | 0=job, 1=alloc set |
| `priority` | INT64 | Priority level |
| `alloc_collection_id` | INT64 | Parent alloc (tasks only) |
| `instance_index` | INT64 | Position within collection |
| `machine_id` | INT64 | Scheduled machine (-1=dedicated) |
| `alloc_instance_index` | INT64 | Alloc instance index |
| `resource_request` | RECORD | CPU/memory requested (Resources struct) |
| `constraint` | ARRAY | Placement constraints (machine attributes) |

#### Resource Request Structure

```sql
resource_request.cpu      -- NCU (Normalized Compute Units)
resource_request.memory   -- Normalized bytes [0,1]
```

#### Relevance to Prediction

- **CRITICAL:** `resource_request` shows intended usage vs. actual (in `instance_usage`)
- **Machine mapping:** `machine_id` enables capacity utilization calculations
- **Scheduling events:** Predict runtime duration and failures
- **Constraints:** Affect placement and performance
- **Key join table:** Links collections to usage data

---

### 3.5 InstanceUsage Table ⭐ **MOST IMPORTANT**

**PDF Section:** "Resource usage" (Pages 14-15)  
**Usefulness for Prediction:** ⭐⭐⭐⭐⭐ **CRITICAL**

#### Schema

| Field | Type | Description |
|-------|------|-------------|
| `start_time` | INT64 | Measurement window start |
| `end_time` | INT64 | Measurement window end (typically 300s = 5 min) |
| `collection_id` | INT64 | Parent collection |
| `instance_index` | INT64 | Instance within collection |
| `machine_id` | INT64 | Machine running the instance |
| `alloc_collection_id` | INT64 | Parent alloc (or 0) |
| `alloc_instance_index` | INT64 | Alloc instance index |
| `collection_type` | INT64 | job or alloc set |
| `average_usage` | RECORD | Average CPU/memory during window |
| `maximum_usage` | RECORD | Max observed CPU/memory |
| `random_sample_usage` | RECORD | 1s random sample (CPU only) |
| `assigned_memory` | FLOAT64 | Average memory limit from OS |
| `page_cache_memory` | FLOAT64 | File page cache memory |
| `cycles_per_instruction` | FLOAT64 | Mean CPI (processor performance) |
| `memory_accesses_per_instruction` | FLOAT64 | Mean MAI (memory bandwidth) |
| `sample_rate` | FLOAT64 | Samples per second (nominal 1 Hz) |
| `cpu_usage_distribution` | ARRAY | **11 percentiles: 0%,10%,...,100%** |
| `tail_cpu_usage_distribution` | ARRAY | **9 tail percentiles: 91%,...,99%** |

#### Usage Structure

```sql
average_usage.cpu     -- Mean CPU usage (NCU-seconds/second)
average_usage.memory  -- Mean memory usage (normalized bytes)
maximum_usage.cpu    -- Max CPU usage
maximum_usage.memory  -- Max memory usage
```

#### CPU Histogram Distributions (Key Feature!)

**Coarse Distribution (11 points):**
```sql
cpu_usage_distribution[OFFSET(0)]   -- 0%ile (minimum)
cpu_usage_distribution[OFFSET(1)]   -- 10%ile
cpu_usage_distribution[OFFSET(2)]   -- 20%ile
...
cpu_usage_distribution[OFFSET(10)]  -- 100%ile (maximum)
```

**Tail Distribution (9 points):**
```sql
tail_cpu_usage_distribution[OFFSET(0)]  -- 91%ile
tail_cpu_usage_distribution[OFFSET(1)]  -- 92%ile
...
tail_cpu_usage_distribution[OFFSET(8)]  -- 99%ile
```

#### Relevance to Prediction

- **PRIMARY table for ML:** Actual resource usage with 5-minute granularity
- **Target variables:** `average_usage.cpu`, `average_usage.memory`
- **Rich distributions:** 20 percentile points capture full CPU behavior
- **Peak prediction:** `maximum_usage` for capacity planning
- **Time series:** `start_time` enables temporal forecasting (LSTM, Transformer)
- **Performance metrics:** CPI and MAI (if available) indicate efficiency
- **Memory details:** `page_cache_memory` for storage optimization

#### PDF Insight (Page 15)

> "The cpu_usage_distribution and tail_cpu_usage_distribution vectors provide detailed information about the distribution of CPU consumption during the 5 minute measurement window, in NCUs. For example, the NCU usage for the 90%ile represents the CPU consumption that 90% of the CPU usage samples during the window would be equal to or smaller than (i.e., we are providing values from the cumulative distribution function, or CDF)."

**This means:** You get a **full CDF every 5 minutes**—extremely valuable for predicting not just averages, but tail behavior (p95, p99) for cost optimization!

---

## 4. Usefulness for Cloud Resource Prediction

### 4.1 Table Ranking by Prediction Utility

| Rank | Table | Score | Primary Use Case |
|------|-------|-------|------------------|
| **1** | **instance_usage** | ⭐⭐⭐⭐⭐ | **PRIMARY:** Actual usage data, 5-min intervals, CPU histograms |
| **2** | **instance_events** | ⭐⭐⭐⭐ | Resource requests, scheduling events, machine mapping |
| **3** | **collection_events** | ⭐⭐⭐⭐ | Job context, priority, scheduling class, vertical scaling |
| **4** | **machine_events** | ⭐⭐⭐ | Machine capacity, utilization ratios |
| **5** | **machine_attributes** | ⭐⭐ | Additional features (kernel, hardware) |

### 4.2 Prediction Tasks and Required Tables

| Prediction Task | Primary Table | Supporting Tables | Key Fields |
|-----------------|----------------|-------------------|------------|
| **CPU Usage (average)** | `instance_usage` | `instance_events` | `average_usage.cpu`, `resource_request.cpu` |
| **CPU Usage (distribution)** | `instance_usage` | - | `cpu_usage_distribution` (11 points) |
| **CPU Usage (tail)** | `instance_usage` | - | `tail_cpu_usage_distribution` (9 points) |
| **Memory Usage** | `instance_usage` | `instance_events` | `average_usage.memory`, `resource_request.memory` |
| **Peak Resource Needs** | `instance_usage` | `machine_events` | `maximum_usage`, `capacity` |
| **Resource Request Optimization** | `instance_events` | `instance_usage` | `resource_request` vs. `average_usage` |
| **Job Runtime Prediction** | `collection_events` + `instance_events` | `instance_usage` | Event sequences, `start_time`, `end_time` |
| **Preemption Probability** | `collection_events` | `instance_events` | `priority`, `scheduling_class`, event types |
| **Cost Optimization** | All tables | - | Usage/capacity ratios, waste identification |

---

## 5. Recommended Table Usage Strategies

### 5.1 For Time-Series Forecasting (LSTM, Transformer)

**Primary Tables:** `instance_usage` + `instance_events`

```python
# Feature Engineering Approach
features = [
    # Time features
    'start_time',
    'hour_of_day',
    'day_of_week',
    'is_weekend',
    
    # Usage features (targets)
    'average_usage.cpu',      # Target 1
    'average_usage.memory',   # Target 2
    
    # CPU distribution features (rich signal!)
    'cpu_p0', 'cpu_p10', 'cpu_p20', ..., 'cpu_p100',   # 11 features
    'cpu_p91', 'cpu_p92', ..., 'cpu_p99',               # 9 features
    
    # Request vs. actual
    'resource_request.cpu',
    'resource_request.memory',
    'request_cpu_ratio',   # request/actual
    
    # Machine context
    'machine_id',
    'capacity.cpu',
    'capacity.memory',
    'utilization_cpu',    # usage/capacity
    
    # Job context
    'priority',
    'scheduling_class',
    'collection_type'
]
```

### 5.2 For Anomaly Detection (Peak Prediction)

**Focus:** `instance_usage.maximum_usage` + `tail_cpu_usage_distribution`

- Predict p95, p99 CPU usage for capacity planning
- Identify outlier instances consuming disproportionate resources
- Use `maximum_usage / capacity` for over-subscription detection

### 5.3 For Cost Optimization (Waste Identification)

**Key Calculation:** `resource_request` vs. `average_usage`

```sql
-- Find over-provisioned instances
SELECT 
    collection_id,
    instance_index,
    resource_request.cpu AS requested_cpu,
    average_usage.cpu AS actual_cpu,
    (resource_request.cpu - average_usage.cpu) / resource_request.cpu AS waste_ratio
FROM `google.com:google-cluster-data.clusterdata_2019_e.instance_usage` iu
JOIN `google.com:google-cluster-data.clusterdata_2019_e.instance_events` ie
    ON iu.collection_id = ie.collection_id AND iu.instance_index = ie.instance_index
WHERE average_usage.cpu > 0
ORDER BY waste_ratio DESC
LIMIT 100
```

---

## 6. Key Insights from Documentation

### 6.1 Normalized Compute Units (NCU)

**PDF Section:** "Resource units" (Page 4)

- **CPU:** Measured in Google Compute Units (GCUs), normalized to [0,1]
  - 1 GCU ≈ 1 CPU-core worth of compute
  - Rescaling constant = largest GCU capacity across all traces
  
- **Memory:** Measured in bytes, normalized to [0,1]
  - Rescaling constant = maximum machine memory observed

**Implication:** All values are relative, not absolute. A value of 0.5 CPU means "50% of the largest machine's capacity."

### 6.2 CPU Usage Measurement

**PDF Section:** "Resource usage" (Page 14)

- **Sampling:** ~1 Hz (1 sample/second) during 5-minute window
- **Average:** Σ(U_cpu) / T_window
- **Maximum:** max(U_cpu / T_sample)
- **Distribution:** 20 percentile points (11 coarse + 9 tail)

**Key Quote (Page 14):**
> "CPU usage (also known as CPU rate) is measured in units of NCU seconds per second: if a task is using two GCUs all the time, it will be reflected as a usage of 2.0 GCU-s/s before normalization."

### 6.3 Memory Usage Measurement

- **Average:** Σ(U_mem × T_sample) / T_window (area under curve / duration)
- **Maximum:** max(U_mem)
- **Assigned memory:** Kernel limit (may be higher than usage)
- **Page cache:** File cache memory (can be reclaimed)

### 6.4 Obfuscation Techniques

**PDF Section:** "Obfuscation techniques" (Page 2)

- **Hashed:** Usernames, collection names, machine attributes
- **Ordered:** Categorical values mapped to sequential integers
- **Rescaled:** CPU/memory to [0,1] range
- **Special:** A few values treated uniquely

**Implication:** Cannot interpret absolute values (e.g., "user X"), but can identify patterns and relationships.

### 6.5 Missing Data

**PDF Section:** "Missing information" (Page 16)

- Data derived from monitoring RPCs (may be missing under load)
- Synthesized records for consistency (see `missing_type` field)
- Dedicated machines excluded from traces (focus on shared workload)

---

## 7. SQL Examples for ML Preparation

### 7.1 Basic Feature Extraction (Single Cell, Single Day)

```sql
-- Extract features for ML training (Cell 'e', May 1, 2019)
SELECT 
    -- Time features
    TIMESTAMP_MICROS(start_time) AS timestamp,
    EXTRACT(HOUR FROM TIMESTAMP_MICROS(start_time)) AS hour_of_day,
    EXTRACT(DAYOFWEEK FROM TIMESTAMP_MICROS(start_time)) AS day_of_week,
    
    -- Target variables (what we want to predict)
    iu.average_usage.cpu AS target_cpu_avg,
    iu.average_usage.memory AS target_memory_avg,
    iu.maximum_usage.cpu AS target_cpu_max,
    
    -- CPU distribution features (rich signal for ML)
    iu.cpu_usage_distribution[OFFSET(0)] AS cpu_p0,    -- Min
    iu.cpu_usage_distribution[OFFSET(5)] AS cpu_p50,   -- Median
    iu.cpu_usage_distribution[OFFSET(10)] AS cpu_p100,  -- Max
    iu.tail_cpu_usage_distribution[OFFSET(8)] AS cpu_p99,
    
    -- Request vs. actual (for waste identification)
    ie.resource_request.cpu AS requested_cpu,
    ie.resource_request.memory AS requested_memory,
    
    -- Machine capacity (for utilization ratios)
    me.capacity.cpu AS machine_cpu_capacity,
    me.capacity.memory AS machine_memory_capacity,
    
    -- Calculated features
    iu.average_usage.cpu / me.capacity.cpu AS cpu_utilization_ratio,
    ie.resource_request.cpu / iu.average_usage.cpu AS overprovision_ratio,
    
    -- Job context
    ce.priority,
    ce.scheduling_class,
    ce.collection_type,
    ie.machine_id

FROM `google.com:google-cluster-data.clusterdata_2019_e.instance_usage` iu
INNER JOIN `google.com:google-cluster-data.clusterdata_2019_e.instance_events` ie
    ON iu.collection_id = ie.collection_id 
    AND iu.instance_index = ie.instance_index
    AND iu.start_time = ie.time  -- Align events with usage windows
INNER JOIN `google.com:google-cluster-data.clusterdata_2019_e.machine_events` me
    ON ie.machine_id = me.machine_id
LEFT JOIN `google.com:google-cluster-data.clusterdata_2019_e.collection_events` ce
    ON ie.collection_id = ce.collection_id
WHERE 
    iu.start_time >= UNIX_MICROS(TIMESTAMP('2019-05-01'))
    AND iu.start_time < UNIX_MICROS(TIMESTAMP('2019-05-02'))
    AND iu.average_usage.cpu IS NOT NULL
LIMIT 10000
```

### 7.2 Export for ML Training (Parquet Format)

```bash
# Export to GCS (free batch export)
bq extract \
  --destination_format=PARQUET \
  --project_id=your-project \
  'google.com:google-cluster-data.clusterdata_2019_e.instance_usage$20190501' \
  'gs://your-bucket/features/instance_usage_20190501.parquet'
```

### 7.3 Multi-Cell Comparison (Regional Analysis)

```sql
-- Compare CPU usage patterns across cells
WITH cell_stats AS (
  SELECT 
    'e' AS cell,
    AVG(iu.average_usage.cpu) AS avg_cpu,
    APPROX_QUANTILES(iu.average_usage.cpu, 100)[OFFSET(50)] AS p50_cpu,
    APPROX_QUANTILES(iu.average_usage.cpu, 100)[OFFSET(90)] AS p90_cpu
  FROM `google.com:google-cluster-data.clusterdata_2019_e.instance_usage` iu
  WHERE start_time >= UNIX_MICROS(TIMESTAMP('2019-05-01'))
    AND start_time < UNIX_MICROS(TIMESTAMP('2019-05-02'))
  
  UNION ALL
  
  SELECT 
    'f' AS cell,
    AVG(iu.average_usage.cpu) AS avg_cpu,
    APPROX_QUANTILES(iu.average_usage.cpu, 100)[OFFSET(50)] AS p50_cpu,
    APPROX_QUANTILES(iu.average_usage.cpu, 100)[OFFSET(90)] AS p90_cpu
  FROM `google.com:google-cluster-data.clusterdata_2019_f.instance_usage` iu
  WHERE start_time >= UNIX_MICROS(TIMESTAMP('2019-05-01'))
    AND start_time < UNIX_MICROS(TIMESTAMP('2019-05-02'))
)
SELECT * FROM cell_stats
```

---

## 8. Summary and Recommendations

### 8.1 Key Takeaways

1. **InstanceUsage is GOLD:** The `instance_usage` table is the most valuable for ML prediction, providing 5-minute interval data with CPU histograms (20 percentile points).

2. **CPU Histograms are Unique:** No other dataset provides this level of CPU distribution detail (0%, 10%, ..., 100%, plus 91%-99%). This enables predicting not just averages, but tail behavior critical for cost optimization.

3. **Join for Context:** Combine `instance_usage` with `instance_events` (for requests) and `machine_events` (for capacity) to create comprehensive feature sets.

4. **Normalization Matters:** All values are in [0,1] range (NCU). Models will predict relative usage, not absolute CPU cores or GB of RAM.

5. **Time Series Ready:** With 5-minute intervals over 31 days, you have ~8,928 data points per instance—sufficient for LSTM/Transformer models.

### 8.2 Recommended ML Workflow

```
1. Start with Cell 'e' (smallest, 819 GB)
2. Extract 1 day of data (~26 GB) using SQL above
3. Feature engineering:
   - Time features (hour, day, weekend)
   - CPU distribution features (20 points → 20 features!)
   - Request vs. actual ratios
   - Utilization ratios
4. Train baseline models (Linear Regression, Random Forest)
5. Evaluate on MAE, RMSE, MAPE (<15% target)
6. Scale to full cell (31 days) or multiple cells
7. Deploy best model for batch predictions
```

### 8.3 Best Tables for Common Tasks

| If you want to predict... | Use this table... | Because... |
|---------------------------|-------------------|------------|
| **Average CPU usage** | `instance_usage` | Has `average_usage.cpu` target variable |
| **Peak CPU usage** | `instance_usage` | Has `maximum_usage.cpu` + p99 from tail distribution |
| **Memory requirements** | `instance_usage` | Has `average_usage.memory` + `assigned_memory` |
| **Over-provisioning** | `instance_events` + `instance_usage` | Compare `resource_request` vs. `average_usage` |
| **Job runtime** | `collection_events` | Event sequences (SUBMIT→SCHEDULE→FINISH) |
| **Preemption risk** | `collection_events` | `priority` and `scheduling_class` fields |
| **Optimal machine placement** | `instance_events` + `machine_events` | `machine_id` + `capacity` |

---

## 9. References

### Primary Sources

1. **Google Cluster Data GitHub Repository**  
   URL: https://github.com/google/cluster-data/blob/master/ClusterData2019.md  
   Retrieved: May 2026  
   Key info: Dataset size (~2.4 TiB), 8 cells, May 2019 timeframe, CC-BY license

2. **Google Cluster-Usage Traces v3 PDF** (included in `ClusterData2019_docs/`)  
   Authors: John Wilkes, Charles Reiss, Nan Deng, Md Ehtesam Haque, Muhammad Tirmazi  
   Version: 2020-08-18  
   Key info: Complete schema, table structures, CPU histograms, normalization methods, obfuscation techniques

3. **BigQuery Dataset Metadata** (verified via dry-run queries)  
   Project: `google.com:google-cluster-data`  
   Datasets: `clusterdata_2019_a` through `clusterdata_2019_h`  
   Key info: Table names, query paths, per-cell sizes

### Supporting Papers

4. **Large-scale cluster management at Google with Borg**  
   Authors: Abhishek Verma, Luis Pedrosa, Madhukar R. Korupolu, David Oppenheimer, Eric Tune, John Wilkes  
   Proc. European Conference on Computer Systems (EuroSys), Bordeaux, France, 2015  
   URL: https://ai.google/research/pubs/pub43438

5. **Borg: the Next Generation**  
   Authors: Muhammad Tirmazi, Adam Barker, Nan Deng, Md Ehtesam Haque, Zhijing Gene Qin, Steven Hand, Mor Harchol-Balter, John Wilkes  
   Proc. European Conference on Computer Systems (EuroSys), Heraklion, Crete, Greece, 2020

### Methodology References

6. **CRISP-ML(Q) Methodology**  
   URL: https://crisp-ml.org/  
   Key info: ML development lifecycle, evaluation metrics, business understanding

---

## Document Revision History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | May 7, 2026 | Initial document: Complete table analysis for cloud resource prediction | Fajar Laksono |

---

**End of Document** (Version 1.0)  
**File:** `docs/DATASET_TABLES_ANALYSIS.md`  
**Total Pages (estimated):** ~8 pages (when printed)  
**Word Count (estimated):** ~2,500 words
