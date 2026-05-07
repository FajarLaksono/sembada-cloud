# Cloud Waste Identification Using ClusterData 2019

**Document Type:** Research Reference - FinOps Analysis  
**Dataset:** Google ClusterData 2019 (Version 3)  
**Primary Source:** Google cluster-usage traces v3.pdf  
**Date:** May 7, 2026  
**Author:** Fajar Laksono  

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Types of Cloud Waste](#2-types-of-cloud-waste)
3. [Data Sources for Waste Detection](#3-data-sources-for-waste-detection)
4. [Waste Identification Methods](#4-waste-identification-methods)
5. [SQL Queries for Waste Detection](#5-sql-queries-for-waste-detection)
6. [Waste Quantification Metrics](#6-waste-quantification-metrics)
7. [Visualization Strategies](#7-visualization-strategies)
8. [ML Approaches for Waste Prediction](#8-ml-approaches-for-waste-prediction)
9. [Summary and Action Plan](#9-summary-and-action-plan)
10. [References](#10-references)

---

## 1. Introduction

Cloud waste refers to paid-for but underutilized or unused resources. In the context of Google Borg cluster traces, waste manifests as:

- **Over-provisioned resources** (requested ≫ actual usage)
- **Idle resources** (allocated but not used)
- **Inefficient scheduling** (poor bin-packing)
- **Preemption waste** (evicted tasks due to over-commitment)
- **Memory bloat** (requested memory ≫ actual need)

This document analyzes how to identify and quantify these waste types using the ClusterData 2019 dataset.

### 1.1 Why Waste Identification Matters

| Impact Area | Description |
|-------------|-------------|
| **Cost** | Over-provisioning directly increases cloud bills |
| **Efficiency** | Underutilized resources could serve other workloads |
| **Sustainability** | Wasted compute = unnecessary energy consumption |
| **Performance** | Poor scheduling affects all workloads |

---

## 2. Types of Cloud Waste

### 2.1 Over-Provisioning (Most Common)

**Definition:** Requesting more resources than actually needed.

**Example:**
- Requested: 1.0 NCU (100% of largest machine's CPU)
- Actual usage: 0.2 NCU (20% of largest machine's CPU)
- **Waste:** 0.8 NCU (80% over-provisioned)

**Data Sources:**
- `instance_events.resource_request.cpu` (what they asked for)
- `instance_usage.average_usage.cpu` (what they used)

### 2.2 Idle Resources

**Definition:** Resources allocated but minimally used.

**Thresholds (suggested):**
- CPU usage < 5% for extended periods
- Memory usage < 10% for extended periods

**Data Sources:**
- `instance_usage.average_usage.*` (actual usage)
- `instance_usage.cpu_usage_distribution` (check p0, p10 for idle confirmation)

### 2.3 Bin-Packing Inefficiency

**Definition:** Machines running below capacity due to poor scheduling.

**Example:**
- Machine capacity: 1.0 NCU
- Total usage across all instances: 0.3 NCU
- **Utilization:** 30% (70% wasted capacity)

**Data Sources:**
- `machine_events.capacity.*` (machine capacity)
- `instance_usage` joined by `machine_id` (aggregate usage per machine)

### 2.4 Preemption Waste

**Definition:** Resources wasted when tasks are evicted and restarted.

**Causes (from PDF Section "Collection and instance life cycles"):**
- Higher priority tasks preempt lower priority ones
- Scheduler over-commits resources
- Machine failures cause evictions

**Data Sources:**
- `instance_events.type` = EVICT, FAIL, KILL
- `collection_events.priority` (lower priority = higher eviction risk)

### 2.5 Memory Bloat

**Definition:** Requesting more memory than needed, reducing available memory for other tasks.

**Data Sources:**
- `instance_events.resource_request.memory` (requested)
- `instance_usage.average_usage.memory` (actual)
- `instance_usage.assigned_memory` (kernel limit)

### 2.6 Vertical Scaling Missed Opportunities

**Definition:** Not using auto-scaling when available.

**Data Source:**
- `collection_events.vertical_scaling` (AUTO_OFF vs. AUTO_ON)

---

## 3. Data Sources for Waste Detection

### 3.1 Primary Tables for Waste Analysis

| Table | Key Fields for Waste | Waste Type Detected |
|-------|----------------------|---------------------|
| **instance_usage** | `average_usage`, `maximum_usage`, `cpu_usage_distribution` | Over-provisioning, idle resources, memory bloat |
| **instance_events** | `resource_request`, `machine_id`, `type` (EVICT) | Over-provisioning, preemption waste |
| **machine_events** | `capacity`, `machine_id` | Bin-packing inefficiency |
| **collection_events** | `priority`, `vertical_scaling`, `scheduling_class` | Preemption risk, missed auto-scaling |

### 3.2 Key Field Combinations for Waste Calculation

#### Over-Provisioning Ratio
```sql
-- CPU over-provisioning
resource_request.cpu / average_usage.cpu AS cpu_waste_ratio

-- Memory over-provisioning  
resource_request.memory / average_usage.memory AS memory_waste_ratio
```

#### Utilization Rate
```sql
-- CPU utilization (higher = better)
average_usage.cpu / resource_request.cpu AS cpu_utilization_rate

-- Machine utilization
SUM(instance_usage.average_usage.cpu) / machine_events.capacity.cpu AS machine_utilization
```

#### Idle Detection
```sql
-- Check CPU distribution (if p50 < 0.05, likely idle)
cpu_usage_distribution[OFFSET(5)] AS cpu_p50  -- 50th percentile

-- Memory idle check
average_usage.memory < 0.1 AS is_memory_idle
```

---

## 4. Waste Identification Methods

### 4.1 Method 1: Request vs. Usage Analysis (Over-Provisioning)

**Approach:** Compare `resource_request` with `average_usage`

**SQL Logic:**
```sql
SELECT 
    collection_id,
    instance_index,
    ie.resource_request.cpu AS requested_cpu,
    iu.average_usage.cpu AS actual_cpu,
    ie.resource_request.memory AS requested_mem,
    iu.average_usage.memory AS actual_mem,
    
    -- Waste calculations
    (ie.resource_request.cpu - iu.average_usage.cpu) / ie.resource_request.cpu AS cpu_waste_ratio,
    (ie.resource_request.memory - iu.average_usage.memory) / ie.resource_request.memory AS mem_waste_ratio,
    
    -- Categorization
    CASE 
        WHEN (ie.resource_request.cpu - iu.average_usage.cpu) / ie.resource_request.cpu > 0.5 THEN 'High Waste'
        WHEN (ie.resource_request.cpu - iu.average_usage.cpu) / ie.resource_request.cpu > 0.3 THEN 'Medium Waste'
        ELSE 'Low Waste'
    END AS waste_category

FROM `google.com:google-cluster-data.clusterdata_2019_e.instance_usage` iu
JOIN `google.com:google-cluster-data.clusterdata_2019_e.instance_events` ie
    ON iu.collection_id = ie.collection_id 
    AND iu.instance_index = ie.instance_index
WHERE iu.start_time >= UNIX_MICROS(TIMESTAMP('2019-05-01'))
    AND iu.start_time < UNIX_MICROS(TIMESTAMP('2019-05-02'))
    AND ie.resource_request.cpu > 0
LIMIT 10000
```

### 4.2 Method 2: CPU Distribution Analysis (Idle Detection)

**Approach:** Use `cpu_usage_distribution` to detect consistently low usage

**Key Insight from PDF (Page 14):**
> "The cpu_usage_distribution... provides detailed information about the distribution of CPU consumption during the 5 minute measurement window"

**Detection Logic:**
```python
# If p10 (10th percentile) < 0.05, instance is likely idle
idle_threshold_p10 = 0.05
idle_threshold_p50 = 0.10

# If p90 (90th percentile) < 0.20, consistently underutilized
underutilized_threshold_p90 = 0.20
```

**SQL Implementation:**
```sql
SELECT 
    collection_id,
    instance_index,
    cpu_usage_distribution[OFFSET(1)] AS cpu_p10,  -- 10th percentile
    cpu_usage_distribution[OFFSET(5)] AS cpu_p50,  -- 50th percentile
    cpu_usage_distribution[OFFSET(9)] AS cpu_p90,  -- 90th percentile
    
    -- Idle detection
    CASE 
        WHEN cpu_usage_distribution[OFFSET(1)] < 0.05 THEN 'Idle'
        WHEN cpu_usage_distribution[OFFSET(5)] < 0.10 THEN 'Underutilized'
        WHEN cpu_usage_distribution[OFFSET(9)] < 0.20 THEN 'Low Usage'
        ELSE 'Normal'
    END AS usage_category

FROM `google.com:google-cluster-data.clusterdata_2019_e.instance_usage`
WHERE start_time >= UNIX_MICROS(TIMESTAMP('2019-05-01'))
    AND start_time < UNIX_MICROS(TIMESTAMP('2019-05-02'))
```

### 4.3 Method 3: Machine-Level Bin-Packing Analysis

**Approach:** Aggregate all instance usage per machine, compare to machine capacity

**SQL Logic:**
```sql
WITH machine_utilization AS (
    SELECT 
        iu.machine_id,
        me.capacity.cpu AS machine_cpu_capacity,
        me.capacity.memory AS machine_mem_capacity,
        
        -- Aggregate all instances on this machine
        SUM(iu.average_usage.cpu) AS total_cpu_usage,
        SUM(iu.average_usage.memory) AS total_mem_usage,
        COUNT(*) AS instance_count,
        
        -- Calculate utilization
        SUM(iu.average_usage.cpu) / me.capacity.cpu AS cpu_utilization,
        SUM(iu.average_usage.memory) / me.capacity.memory AS mem_utilization
    
    FROM `google.com:google-cluster-data.clusterdata_2019_e.instance_usage` iu
    JOIN `google.com:google-cluster-data.clusterdata_2019_e.machine_events` me
        ON iu.machine_id = me.machine_id
    WHERE iu.start_time >= UNIX_MICROS(TIMESTAMP('2019-05-01'))
        AND iu.start_time < UNIX_MICROS(TIMESTAMP('2019-05-02'))
    GROUP BY iu.machine_id, me.capacity.cpu, me.capacity.memory
)
SELECT 
    machine_id,
    machine_cpu_capacity,
    total_cpu_usage,
    instance_count,
    cpu_utilization,
    
    CASE 
        WHEN cpu_utilization < 0.3 THEN 'Severely Underutilized'
        WHEN cpu_utilization < 0.5 THEN 'Underutilized'
        WHEN cpu_utilization < 0.7 THEN 'Moderately Utilized'
        ELSE 'Well Utilized'
    END AS utilization_category

FROM machine_utilization
ORDER BY cpu_utilization ASC
LIMIT 1000
```

### 4.4 Method 4: Preemption Waste Analysis

**Approach:** Count evictions and calculate wasted resources due to restarts

**From PDF (Page 9):**
> "EVICT: a thing was descheduled because of a higher priority thing, because the scheduler overcommitted..."

**SQL Logic:**
```sql
-- Count evictions by priority
SELECT 
    ce.priority,
    CASE 
        WHEN ce.priority <= 99 THEN 'Free Tier'
        WHEN ce.priority BETWEEN 100 AND 115 THEN 'Best-effort Batch'
        WHEN ce.priority BETWEEN 116 AND 119 THEN 'Mid-tier'
        WHEN ce.priority BETWEEN 120 AND 359 THEN 'Production'
        ELSE 'Monitoring'
    END AS priority_tier,
    
    COUNTIF(ie.type = 4) AS eviction_count,  -- type 4 = EVICT
    COUNTIF(ie.type = 5) AS fail_count,      -- type 5 = FAIL
    COUNTIF(ie.type = 7) AS kill_count,      -- type 7 = KILL
    
    -- Total events for this priority
    COUNT(*) AS total_events,
    
    -- Preemption rate
    (COUNTIF(ie.type = 4) + COUNTIF(ie.type = 5)) / COUNT(*) AS preemption_rate

FROM `google.com:google-cluster-data.clusterdata_2019_e.instance_events` ie
JOIN `google.com:google-cluster-data.clusterdata_2019_e.collection_events` ce
    ON ie.collection_id = ce.collection_id
WHERE ie.time >= UNIX_MICROS(TIMESTAMP('2019-05-01'))
    AND ie.time < UNIX_MICROS(TIMESTAMP('2019-05-02'))
GROUP BY ce.priority
ORDER BY ce.priority
```

### 4.5 Method 5: Vertical Scaling Opportunity

**Approach:** Identify collections that could benefit from auto-scaling but have it disabled

**From PDF (Page 11):**
> "vertical_scaling - if enabled, the system determines how much CPU and RAM to request..."

**SQL Logic:**
```sql
SELECT 
    vertical_scaling,
    COUNT(*) AS collection_count,
    AVG(priority) AS avg_priority,
    
    -- Sample of waste (if we can join to usage)
    AVG(cpu_waste_ratio) AS avg_cpu_waste
    
FROM `google.com:google-cluster-data.clusterdata_2019_e.collection_events` ce
LEFT JOIN (
    -- Subquery to calculate waste ratio per collection
    SELECT 
        collection_id,
        AVG((ie.resource_request.cpu - iu.average_usage.cpu) / ie.resource_request.cpu) AS cpu_waste_ratio
    FROM `google.com:google-cluster-data.clusterdata_2019_e.instance_usage` iu
    JOIN `google.com:google-cluster-data.clusterdata_2019_e.instance_events` ie
        ON iu.collection_id = ie.collection_id AND iu.instance_index = ie.instance_index
    WHERE iu.start_time >= UNIX_MICROS(TIMESTAMP('2019-05-01'))
        AND iu.start_time < UNIX_MICROS(TIMESTAMP('2019-05-02'))
    GROUP BY collection_id
) waste ON ce.collection_id = waste.collection_id

WHERE vertical_scaling = 'VERTICAL_SCALING_OFF'  -- Auto-scaling disabled
GROUP BY vertical_scaling
```

---

## 5. SQL Queries for Waste Detection

### 5.1 Top 10 Most Wasteful Instances (Over-Provisioned)

```sql
SELECT 
    iu.collection_id,
    iu.instance_index,
    ie.resource_request.cpu AS requested_cpu,
    iu.average_usage.cpu AS actual_cpu,
    ROUND((ie.resource_request.cpu - iu.average_usage.cpu) / ie.resource_request.cpu * 100, 2) AS waste_percentage,
    ce.user AS hashed_user,
    ce.priority,
    
    -- Memory waste too
    ie.resource_request.memory AS requested_mem,
    iu.average_usage.memory AS actual_mem

FROM `google.com:google-cluster-data.clusterdata_2019_e.instance_usage` iu
JOIN `google.com:google-cluster-data.clusterdata_2019_e.instance_events` ie
    ON iu.collection_id = ie.collection_id AND iu.instance_index = ie.instance_index
JOIN `google.com:google-cluster-data.clusterdata_2019_e.collection_events` ce
    ON iu.collection_id = ce.collection_id

WHERE iu.start_time >= UNIX_MICROS(TIMESTAMP('2019-05-01'))
    AND iu.start_time < UNIX_MICROS(TIMESTAMP('2019-05-02'))
    AND ie.resource_request.cpu > 0
    AND (ie.resource_request.cpu - iu.average_usage.cpu) / ie.resource_request.cpu > 0.5  -- >50% waste

ORDER BY waste_percentage DESC
LIMIT 10
```

### 5.2 Idle Resource Detection (CPU < 5%)

```sql
SELECT 
    machine_id,
    COUNT(*) AS idle_instance_count,
    AVG(cpu_usage_distribution[OFFSET(1)]) AS avg_cpu_p10,  -- Average 10th percentile
    AVG(average_usage.cpu) AS avg_cpu_usage,
    SUM(ie.resource_request.cpu) AS total_requested_cpu

FROM `google.com:google-cluster-data.clusterdata_2019_e.instance_usage` iu
JOIN `google.com:google-cluster-data.clusterdata_2019_e.instance_events` ie
    ON iu.collection_id = ie.collection_id AND iu.instance_index = ie.instance_index

WHERE start_time >= UNIX_MICROS(TIMESTAMP('2019-05-01'))
    AND start_time < UNIX_MICROS(TIMESTAMP('2019-05-02'))
    AND cpu_usage_distribution[OFFSET(1)] < 0.05  -- p10 < 5%

GROUP BY machine_id
ORDER BY idle_instance_count DESC
LIMIT 100
```

### 5.3 Daily Waste Summary Report

```sql
SELECT 
    DATE(TIMESTAMP_MICROS(start_time)) AS date,
    COUNT(*) AS total_measurements,
    
    -- Over-provisioned instances (>50% waste)
    COUNTIF((ie.resource_request.cpu - average_usage.cpu) / ie.resource_request.cpu > 0.5) AS high_waste_count,
    
    -- Idle instances
    COUNTIF(cpu_usage_distribution[OFFSET(1)] < 0.05) AS idle_count,
    
    -- Average waste ratio
    AVG((ie.resource_request.cpu - average_usage.cpu) / ie.resource_request.cpu) AS avg_waste_ratio,
    
    -- Total wasted CPU (NCU)
    SUM(ie.resource_request.cpu - average_usage.cpu) AS total_wasted_cpu

FROM `google.com:google-cluster-data.clusterdata_2019_e.instance_usage` iu
JOIN `google.com:google-cluster-data.clusterdata_2019_e.instance_events` ie
    ON iu.collection_id = ie.collection_id AND iu.instance_index = ie.instance_index

WHERE start_time >= UNIX_MICROS(TIMESTAMP('2019-05-01'))
    AND start_time < UNIX_MICROS(TIMESTAMP('2019-05-08'))  -- First week

GROUP BY date
ORDER BY date
```

---

## 6. Waste Quantification Metrics

### 6.1 Key Performance Indicators (KPIs) for Waste

| Metric | Formula | Target | Interpretation |
|--------|----------|--------|-----------------|
| **CPU Waste Ratio** | (requested - actual) / requested | < 0.3 (30%) | Lower is better |
| **Memory Waste Ratio** | (requested - actual) / requested | < 0.3 (30%) | Lower is better |
| **Machine Utilization** | total_usage / capacity | > 0.7 (70%) | Higher is better |
| **Preemption Rate** | (evictions + fails) / total events | < 0.05 (5%) | Lower is better |
| **Idle Instance %** | idle_instances / total_instances | < 0.10 (10%) | Lower is better |

### 6.2 Waste Cost Estimation (Conceptual)

**Note:** Since this is normalized data (NCU), we can't calculate real dollars, but we can estimate relative waste:

```python
# Conceptual cost model
wasted_cpu_ncu = sum(requested_cpu - actual_cpu)
wasted_memory_ncu = sum(requested_memory - actual_memory)

# If 1 NCU = $X/hour (hypothetical)
estimated_waste_cost = wasted_cpu_ncu * hourly_rate * hours_running
```

### 6.3 Aggregation Levels for Waste Analysis

| Level | Grouping | Use Case |
|-------|-----------|----------|
| **Instance-level** | collection_id + instance_index | Identify specific wasteful instances |
| **Collection-level** | collection_id | Identify wasteful jobs/users |
| **Machine-level** | machine_id | Identify bin-packing issues |
| **Cell-level** | (all data) | Overall efficiency reporting |
| **Time-based** | DATE(start_time) | Track waste trends over time |

---

## 7. Visualization Strategies

### 7.1 Recommended Visualizations for Waste

| Chart Type | Data | Insight |
|------------|------|---------|
| **Histogram** | cpu_waste_ratio | Distribution of over-provisioning |
| **Scatter Plot** | requested_cpu vs. actual_cpu | Identify outliers |
| **Heatmap** | hour_of_day vs. avg_utilization | Find idle time periods |
| **Bar Chart** | waste by priority tier | Which tier wastes most? |
| **Time Series** | daily_waste_trend | Is waste improving? |
| **Pie Chart** | waste_category breakdown | Proportion of waste types |

### 7.2 Key Charts to Generate

#### Chart 1: Request vs. Actual Scatter Plot
```python
import matplotlib.pyplot as plt

plt.scatter(df['requested_cpu'], df['actual_cpu'], alpha=0.5)
plt.plot([0, 1], [0, 1], 'r--', label='Ideal: requested=actual')
plt.xlabel('Requested CPU (NCU)')
plt.ylabel('Actual CPU Usage (NCU)')
plt.title('CPU Over-Provisioning Detection')
plt.legend()
plt.show()
```

#### Chart 2: Waste by Priority Tier
```python
# Bar chart showing which priority tiers waste most resources
waste_by_tier = df.groupby('priority_tier')['waste_percentage'].mean()
waste_by_tier.plot(kind='bar')
plt.title('Average CPU Waste by Priority Tier')
plt.ylabel('Waste Percentage')
plt.show()
```

#### Chart 3: Machine Utilization Distribution
```python
# Histogram of machine utilization
plt.hist(df['machine_utilization'], bins=20, edgecolor='black')
plt.axvline(x=0.7, color='r', linestyle='--', label='Target (70%)')
plt.xlabel('Machine Utilization')
plt.ylabel('Count')
plt.title('Bin-Packing Efficiency Distribution')
plt.legend()
plt.show()
```

---

## 8. ML Approaches for Waste Prediction

### 8.1 Predictive Models for Waste

| Model Type | Target Variable | Features | Use Case |
|------------|----------------|-----------|----------|
| **Regression** | `cpu_waste_ratio` | time, priority, scheduling_class, user | Predict over-provisioning |
| **Classification** | `is_idle` (binary) | cpu_distribution, memory_usage | Detect idle resources |
| **Time Series** | `daily_waste_trend` | historical_waste, day_of_week | Forecast waste |
| **Clustering** | (unsupervised) | all usage features | Group similar waste patterns |

### 8.2 Feature Engineering for Waste Prediction

```python
features = [
    # Time features
    'hour_of_day',
    'day_of_week',
    'is_weekend',
    
    # Request features
    'resource_request.cpu',
    'resource_request.memory',
    'priority',
    'scheduling_class',
    
    # Historical usage (lag features)
    'cpu_usage_lag_1h',
    'cpu_usage_lag_24h',
    'mem_usage_lag_1h',
    
    # Distribution features (from cpu_usage_distribution)
    'cpu_p10', 'cpu_p50', 'cpu_p90', 'cpu_p99',
    'cpu_spread',  # p90 - p10 (variability)
    
    # Ratios
    'request_to_capacity_ratio',
    'historical_avg_utilization',
    
    # Context
    'machine_utilization',
    'instance_count_on_machine'
]

# Target: Is this instance wasteful?
target = 'is_wasteful'  # Defined as waste_ratio > 0.5
```

### 8.3 Anomaly Detection for Waste

**Approach:** Use isolation forests or autoencoders to detect unusual waste patterns

```python
from sklearn.ensemble import IsolationForest

# Detect anomalous waste patterns
model = IsolationForest(contamination=0.1)  # Expect 10% anomalies
anomalies = model.fit_predict(X)  # X = feature matrix

# Anomalies = -1, Normal = 1
df['is_anomaly'] = anomalies == -1
```

---

## 9. Summary and Action Plan

### 9.1 Waste Types Detectable from ClusterData 2019

| Waste Type | Detectable? | Key Tables | Difficulty |
|------------|--------------|------------|------------|
| **Over-Provisioning** | ✅ Yes | instance_usage + instance_events | Easy |
| **Idle Resources** | ✅ Yes | instance_usage (cpu_distribution) | Easy |
| **Bin-Packing Inefficiency** | ✅ Yes | instance_usage + machine_events | Medium |
| **Preemption Waste** | ✅ Yes | instance_events + collection_events | Medium |
| **Memory Bloat** | ✅ Yes | instance_usage + instance_events | Easy |
| **Missed Auto-Scaling** | ✅ Yes | collection_events | Easy |

### 9.2 Recommended Analysis Workflow

```
Step 1: Start with Cell 'e' (smallest, 819 GB)
   ↓
Step 2: Run Request vs. Usage Analysis (Query 5.1)
   → Identify top 10 most wasteful instances
   ↓
Step 3: Detect Idle Resources (Query 5.2)
   → Find consistently idle instances
   ↓
Step 4: Machine-Level Bin-Packing (Method 3)
   → Identify underutilized machines
   ↓
Step 5: Generate Daily Waste Summary (Query 5.3)
   → Track waste trends over time
   ↓
Step 6: Visualize (Section 7)
   → Create charts for stakeholder presentation
   ↓
Step 7: Build ML Model (Section 8)
   → Predict future waste
   ↓
Step 8: Generate Recommendations
   → Actionable insights for cost optimization
```

### 9.3 Quick Wins (High Impact, Low Effort)

1. **Identify top 100 over-provisioned instances** → Right-size their requests
2. **Find idle instances running 24/7** → Schedule downtime or delete
3. **Detect underutilized machines** → Consolidate workloads
4. **Check for disabled vertical scaling** → Enable auto-scaling where appropriate

### 9.4 Expected Outcomes

| Analysis | Expected Finding | Potential Savings |
|----------|------------------|-------------------|
| Over-provisioning | 30-50% of instances waste >50% CPU | 15-25% cost reduction |
| Idle resources | 10-20% instances are idle | 10-20% cost reduction |
| Bin-packing | Average machine utilization ~40-60% | 20-30% consolidation possible |
| Preemption waste | Low-priority jobs restarted frequently | Improve scheduling efficiency |

---

## 10. References

### Primary Sources

1. **Google Cluster-Usage Traces v3 PDF** (included in `ClusterData2019_docs/`)  
   Authors: John Wilkes et al.  
   Version: 2020-08-18  
   Key sections: "Resource usage" (Page 14), "CollectionEvents table" (Page 11), "InstanceEvents table" (Page 13)

2. **Google Cluster Data GitHub**  
   URL: https://github.com/google/cluster-data/blob/master/ClusterData2019.md  
   Retrieved: May 2026  
   Key info: Dataset overview, access methods

3. **BigQuery Dataset**  
   Project: `google.com:google-cluster-data`  
   Dataset: `clusterdata_2019_e` (for Cell 'e')  
   Tables: instance_usage, instance_events, machine_events, collection_events

### Supporting Resources

4. **FinOps Foundation**  
   URL: https://www.finops.org/  
   Key info: Cloud cost optimization best practices

5. **Google Cloud Pricing**  
   URL: https://cloud.google.com/bigquery/pricing  
   Key info: Cost models for BigQuery queries

6. **Borg Papers**  
   - Large-scale cluster management at Google with Borg (EuroSys 2015)  
   - Borg: the Next Generation (EuroSys 2020)

---

## Document Revision History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | May 7, 2026 | Initial document: Complete waste identification methods using ClusterData 2019 | Fajar Laksono |

---

**End of Document** (Version 1.0)  
**File:** `docs/CLOUD_WASTE_IDENTIFICATION.md`  
**Total Lines:** ~750 lines  
**Word Count:** ~3,500 words  
**Estimated Pages:** ~10 pages (when printed)
