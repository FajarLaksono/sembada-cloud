# EDA Plan — Google ClusterData 2019

**Status:** Final  
**Next Notebook:** `1_eda_exploration.ipynb`  
**Prerequisite:** `0_data_profiling.ipynb` (completed)

## References

**Docs:**
- `CLOUD_WASTE_IDENTIFICATION.md`
- `GoogleCluster2019_DATASET_TABLES_ANALYSIS.md`
- `TECHNICAL_SPECIFICATION.md`

**Official Google:**
- `clusterdata_analysis_colab.ipynb`
- `Google cluster-usage traces v3.pdf`

---

## Objective

Perform cost-efficient Exploratory Data Analysis on the Google ClusterData 2019 traces (~7.5 TiB across 8 cells) to prepare for cloud resource prediction ML and DL modeling.

## Tooling

- **ibis-framework** (`ibis[bigquery]`) — pandas-like expressions that compile to BigQuery SQL
- Aggregations run server-side, only small results returned locally
- Supports local Parquet with the same API — switch backend without changing code
- `to_sql()` for debugging, `dry_run=True` for cost estimation

## Architecture: Stage Once, Query Free

```
ibis → BigQuery (1 scan, ~$0.52)
         ↓
    Parquet (2 weeks, ~4-5 GB)
         ↓
    ibis (local, $0) → EDA notebook + ML training
```

- **Extract once** from BigQuery → save as Parquet
- All subsequent EDA and ML training runs **locally** — zero recurring cost

## Engineering Rules

1. NEVER `SELECT *` without WHERE filter
2. ALWAYS add `WHERE start_time BETWEEN ...` for time-bounded queries
3. ALWAYS run `dry_run=True` before executing new queries
4. ALWAYS select only the columns needed (columnar = pay per column)
5. START with cell `e` and a time-bounded window

## Cost Budget

### Staging Extraction (one-time)

| Table | Window | Columns | Est. Scan | Est. Cost |
|-------|--------|---------|-----------|-----------|
| `instance_usage` | May 6-19 | 12 columns (usage + keys) | ~52 GB | ~$0.33 |
| `instance_events` | May 6-19 | 8 columns (requests + keys) | ~15 GB | ~$0.09 |
| `collection_events` | May 6-19 | 6 columns (context + keys) | ~8 GB | ~$0.05 |
| `machine_events` | Full | 5 columns (capacity + keys) | ~5 GB | ~$0.03 |
| `machine_attributes` | Full | 4 columns (sparse features) | ~3 GB | ~$0.02 |
| **Total staging** | | | **~83 GB** | **~$0.52** |

### Full-Cell Aggregations (runs on BigQuery, returns tiny result)

| Query | Est. Scan | Est. Cost |
|-------|-----------|-----------|
| Null analysis grouped by `missing_type` (1 day) | ~10 GB | ~$0.06 |
| APPROX_QUANTILES across full cell `e` | ~200 GB | ~$1.25 |
| Distinct counts per table | ~10 GB | ~$0.06 |
| **Total aggregations** | **~220 GB** | **~$1.37** |

**Grand total: ~$1.89** — well within 1 TiB/month free tier.

## Data Quality Strategy

### Missing Values — Group by `missing_type`, don't just count nulls

This dataset has **documented missingness**. The `missing_type` column in `instance_events` and `collection_events` explains *why* data is missing:

| `missing_type` | Meaning | Action |
|----------------|---------|--------|
| Not set / empty | Real measurement | Keep |
| `SYNTHESIZED` | Google imputed this row to fill gaps | Filter out |
| Other values | Monitoring gap / load shedding | Flag, don't impute |

**Cost-efficient approach:** run one aggregation query (not row-level scan):
```sql
SELECT
  missing_type,
  COUNT(*) AS rows,
  COUNTIF(average_usage.cpus IS NULL) AS null_avg_cpu,
  COUNTIF(average_usage.memory IS NULL) AS null_avg_mem,
  COUNTIF(cycles_per_instruction IS NULL) AS null_cpi
FROM `...instance_usage`
WHERE start_time BETWEEN 1556611200 AND 1556697600
GROUP BY missing_type
```
→ Scans ~10 GB, returns 1-5 rows. ~$0.06.

### Duplicates — Low Priority

Production trace data from Google. Near-zero natural duplicates.
A quick `COUNT(*) vs COUNT(DISTINCT key)` check on 1 day suffices.

### Data Cleaning Needed — Structural, Not Scrubbing

| Task | Method | Where |
|------|--------|-------|
| Convert INT64 timestamps → datetime | Cast in ibis query | During staging |
| Flatten RECORD fields (`average_usage.cpus` → `avg_cpu`) | Dot notation in ibis | During staging |
| Expand REPEATED arrays (20 percentile columns) | Index each element | During staging |
| Filter synthesized records (`missing_type`) | WHERE filter | During staging |
| Handle nulls | Flag as features, don't impute blindly | In ML pipeline |
| Time-window joins | `start_time BETWEEN event.time AND event.time + delta` | During staging |

**Not needed:** spell-check, duplicates hunt, normalization (already in NCU [0,1]), data type fixes.

## Staging Specification

- **Cell:** `e` (smallest, ~819 GB compressed)
- **Window:** 2 weeks — **May 6-19, 2019** (Monday through Sunday + 2nd week)
- **Format:** Parquet (columnar, compressed, type-preserving, ibis-native)
- **Location:** `data/staging/cell_e_2weeks.parquet` (gitignored)
- **Granularity:** Keep 5-min intervals; create hourly aggregation as a separate file

### Why 2 Weeks (Not 1, Not Full Month)

| Criterion | 1 Week | 2 Weeks | Full Month |
|-----------|--------|---------|------------|
| Train/val/test split | ~4d/1d/2d — too tight | ~10d/2d/2d — reasonable | ~25d/3d/3d — ideal |
| Weekend coverage | 1 weekend | 2 weekends | 4 weekends |
| Weekly cycle detection | Partial | 2 full cycles | 4+ cycles |
| Staging cost | ~$0.26 | ~$0.52 | ~$2.00 |
| Local size | ~2 GB | ~4-5 GB | ~10-20 GB |
| Reusable for ML | Baseline only | Proper models | Production-ready |

2 weeks is the **minimum viable unit** for proper ML splits and pattern capture.

### Columns to Extract (Prediction-Relevant Only)

**`instance_usage`** (primary table — usage measurements, 5-min intervals):

| Column | Renamed to | Purpose |
|--------|-----------|---------|
| `start_time` | `timestamp` | Temporal index (cast to TIMESTAMP) |
| `end_time` | `end_time` | Window end |
| `collection_id` | `collection_id` | Join key |
| `instance_index` | `instance_index` | Join key |
| `machine_id` | `machine_id` | Join key |
| `average_usage.cpus` | `avg_cpu` | **Target**: mean CPU |
| `average_usage.memory` | `avg_mem` | **Target**: mean memory |
| `maximum_usage.cpus` | `max_cpu` | **Target**: peak CPU |
| `maximum_usage.memory` | `max_mem` | **Target**: peak memory |
| `cpu_usage_distribution` | `cpu_p0..cpu_p100` | Feature: 11-point CPU CDF |
| `tail_cpu_usage_distribution` | `cpu_p91..cpu_p99` | Feature: 9-point tail |
| `cycles_per_instruction` | `cpi` | Feature: processor efficiency |
| `memory_accesses_per_instruction` | `mai` | Feature: memory bandwidth |
| `sample_rate` | `sample_rate` | Feature: measurement quality |

**`instance_events`** (resource requests, scheduling):

| Column | Renamed to | Purpose |
|--------|-----------|---------|
| `collection_id` | `collection_id` | Join key |
| `instance_index` | `instance_index` | Join key |
| `resource_request.cpus` | `requested_cpu` | Feature: requested CPU |
| `resource_request.memory` | `requested_memory` | Feature: requested memory |
| `priority` | `priority` | Feature: priority level |
| `scheduling_class` | `scheduling_class` | Feature: latency sensitivity |
| `collection_type` | `collection_type` | Feature: job vs alloc set |
| `missing_type` | `missing_type` | Quality flag |

**`collection_events`** (job context):

| Column | Renamed to | Purpose |
|--------|-----------|---------|
| `collection_id` | `collection_id` | Join key |
| `priority` | `priority` | Feature |
| `scheduling_class` | `scheduling_class` | Feature |
| `user` | `user` | Feature: usage patterns |
| `collection_type` | `collection_type` | Feature: job vs alloc set |
| `vertical_scaling` | `vertical_scaling` | Feature: auto-scaling mode |

**`machine_events`** (machine capacity):

| Column | Renamed to | Purpose |
|--------|-----------|---------|
| `machine_id` | `machine_id` | Join key |
| `capacity.cpus` | `machine_cpu_capacity` | Feature: max available CPU |
| `capacity.memory` | `machine_memory_capacity` | Feature: max available memory |
| `platform_id` | `platform_id` | Feature: hardware generation |

**`machine_attributes`** (sparse hardware metadata):

| Column | Renamed to | Purpose |
|--------|-----------|---------|
| `machine_id` | `machine_id` | Join key |
| `name` | `attr_name` | Feature: attribute type |
| `value` | `attr_value` | Feature: hashed value |

## Notebook Sections / Layout

### Title: 01 — EDA

- Description and objective of the notebook
- From previous notebook we understand the dataset is big — do analysis efficiently

#### 1. Preparation

##### 1.1. Import Libraries
- import any libraries

##### 1.2. Configuration
###### 1.2.1. Variables
- Cost tracker (reuse from profiling notebook)
- Pin to cell `e` by default
- Define 2-week window: `(1557062400, 1558224000)` = May 6-19, 2019

###### 1.2.2. Connection Setup
- BigQuery client
- ibis BigQuery connection

###### 1.2.3. Connection Test
- Simple, $0 cost

###### 1.2.4. Dry Run Test
- Verify dry run works

#### 2. Metadata Summary

- Summarize from `0_data_profiling.ipynb` (reuse, don't re-query)
- Column data types, RECORD/REPEATED fields
- Note: `INFORMATION_SCHEMA` returns 403 on public datasets — use `client.get_table()` (metadata API, $0) instead

#### 3. Data Quality Analysis on BigQuery

- **Null analysis** grouped by `missing_type` (aggregation, $0.06)
- **Distinct counts** per table ($0.03)
- **Duplicate check** on composite key ($0.03)
- All return small result sets — no row-level pulls

#### 4. Data Staging — Extract 2 Weeks to Parquet

- Build ibis expressions for each table with column selection + renaming
- Apply `WHERE start_time BETWEEN ...` (or `time BETWEEN ...`)
- Filter `missing_type != 'SYNTHESIZED'`
- Execute extraction, write to Parquet
- Verify: read Parquet back, check row count, schema, file size

##### 4.1. Flattening RECORD Fields

ibis dot notation automatically flattens structs:
```python
expr = table.select([
    table.average_usage.cpus.name('avg_cpu'),
    table.average_usage.memory.name('avg_mem'),
    table.capacity.cpus.name('machine_cpu_capacity'),
])
```

##### 4.2. Expanding REPEATED Arrays

ibis array indexing:
```python
expr = table.select([
    table.cpu_usage_distribution[0].name('cpu_p0'),
    table.cpu_usage_distribution[1].name('cpu_p10'),
    # ... through cpu_p100
    table.tail_cpu_usage_distribution[0].name('cpu_p91'),
    # ... through cpu_p99
])
```

##### 4.3. Hourly Aggregation (Second Version)

```python
hourly = (
    staging_table
    .group_by([
        table.timestamp.hour().name('hour'),
        table.collection_id,
        table.machine_id,
    ])
    .aggregate([
        table.avg_cpu.mean().name('avg_cpu'),
        table.avg_mem.mean().name('avg_mem'),
        table.max_cpu.max().name('max_cpu'),
        table.max_mem.max().name('max_mem'),
    ])
)
```

Save as `data/processed/cell_e_2weeks_hourly.parquet`.

#### 5. EDA on Sample (Local, $0)

##### 5.1. Standard Analysis
- Missing value heatmap per column (after filtering synthesized)
- Distribution plots: CPU, memory, CPI, MAI
- CPU CDF reconstruction from percentile arrays (unique to this dataset)
- Correlation matrix of numeric features
- Time series plots (hourly CPU/memory patterns)

##### 5.2. Key Visualizations
| Plot | What It Reveals |
|------|----------------|
| Heatmap: avg CPU × hour × day_of_week | Diurnal pattern + weekend dip |
| CPU CDF (from distribution arrays) | Full usage distribution — not just averages |
| Scatter: requested vs actual CPU | Over-provisioning magnitude |
| Missing value bar chart × missing_type | Are missing values systematic or random? |
| Time-series: CPU by scheduling_class | How latency sensitivity affects usage |
| Histogram: cycles_per_instruction | Processor efficiency distribution |

#### 6. Full-Cell Aggregations (BigQuery, ~$1.37)

Run these on BigQuery — they return tiny result sets:
- `APPROX_QUANTILES(avg_cpu, 100)` for p50/p90/p99 across full cell `e`
- Same for memory
- Row counts, distinct machines, distinct collections
- Compare with 2-week sample — does the sample represent the full cell?

#### 7. Key Predictive Insights

| Insight | Calculation | What It Tells Us |
|---------|-------------|------------------|
| **Over-provisioning** | `requested_cpu / avg_cpu` | How much CPU is wasted |
| **Utilization** | `avg_cpu / machine_cpu_capacity` | How full are machines |
| **Preemption rate** | `COUNTIF(evict) / COUNT(*)` by priority | Which jobs get killed |
| **Diurnal amplitude** | max(hrly avg) - min(hrly avg) | Peak-to-off-peak ratio |
| **Machine heterogeneity** | CPU/memory by `platform_id` | Hardware diversity |
| **Tail behavior** | `p99 / p50` ratio | Spikiness of workloads |

#### 8. Conclusion
- Summary and Conclusion of the notebook

## Relevant Tables & Join Strategy

```
instance_usage  (usage measurements, 5-min intervals)
  │
  ├── instance_events   (resource requests, scheduling)  ON collection_id + instance_index
  │                                                        AND start_time ≈ time
  ├── machine_events    (machine capacity)               ON machine_id
  │                                                        (as-of join by time)
  ├── collection_events (job context, priority)          ON collection_id
  │
  └── machine_attributes (hardware metadata)             ON machine_id
                                                        (key-value pivot)
```

## Feature Matrix (Target ML Schema)

| Feature Group | Columns | Source Table |
|---------------|---------|--------------|
| Temporal | `hour_of_day`, `day_of_week`, `is_weekend` | Derived from timestamp |
| Usage (targets) | `avg_cpu`, `avg_mem`, `max_cpu`, `max_mem` | instance_usage |
| CPU Distribution | `cpu_p0`..`cpu_p100`, `cpu_p91`..`cpu_p99` | instance_usage |
| Request | `requested_cpu`, `requested_memory` | instance_events |
| Machine | `machine_cpu_capacity`, `machine_memory_capacity` | machine_events |
| Job Context | `priority`, `scheduling_class`, `collection_type` | collection_events + instance_events |
| Derived | `cpu_utilization_ratio`, `overprovision_ratio` | Calculated |
| Performance | `cpi`, `mai` | instance_usage |

---

## Decisions & Rationale

| Question | Decision | Why |
|----------|----------|-----|
| Which cell? | `e` only (for now) | Smallest at ~819 GB |
| Training window? | 2 weeks (May 6-19) | Minimum for proper train/val/test splits |
| Export format? | **Parquet** | Columnar, 5-10x smaller than CSV, preserves types, ibis-native |
| Granularity? | **Both**: 5-min + hourly | Keep raw for ML flexibility, hourly for fast iteration |
| Missing value approach? | **Aggregate grouped by missing_type** | Cheap ($0.06), returns interpretable summary |
| INFORMATION_SCHEMA? | **Skip** — use `client.get_table()` | Public datasets return 403 |
| Synthesized records? | **Filter out** during staging | Not real measurements |

## Recommended Visualizations

1. **Heatmap**: `avg_cpu` by hour × day_of_week → diurnal pattern + weekend dip
2. **CPU CDF line**: Reconstructed from `cpu_p0..cpu_p100` arrays → full distribution (unique to this dataset)
3. **Scatter**: `requested_cpu` vs `avg_cpu` → over-provisioning magnitude and skew
4. **Missing value bar chart**: grouped by `missing_type` → systematic vs random gaps
5. **Time-series faceted**: `avg_cpu` rolling mean by `scheduling_class` → how latency sensitivity affects usage
6. **Histogram**: `cpi` (cycles_per_instruction) → processor efficiency spread
7. **Hexbin**: `avg_cpu` vs `avg_mem` → resource correlation density
8. **Box plot**: `avg_cpu` by `day_of_week` → weekday vs weekend distribution
9. **Line**: cumulative `COUNT(DISTINCT collection_id)` over time → job arrival rate
10. **Tail ratio**: `p99_cpu / p50_cpu` by `collection_type` → workload spikiness comparison
