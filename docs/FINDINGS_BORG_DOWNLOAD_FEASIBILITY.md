# Findings: Google Borg ClusterData 2019 — Local Download Feasibility

## Context

Attempted to download Google Borg ClusterData 2019 traces (cell `e`, ~819 GB compressed)
from the public GCS bucket `gs://clusterdata_2019_e` for local processing.
Goal: avoid BigQuery free tier quota exhaustion (1 TiB/month).

## Key Findings

### 1. Data Organization

The 5 tables (`instance_usage`, `instance_events`, `machine_events`,
`collection_events`, `machine_attributes`) are stored as **thousands of shard files**
per table. Each shard is hash-partitioned (not time-range-partitioned), so every
shard covers the full month (May 1 – June 1).

### 2. Cross-Table Join Problem

| Join | Overlap (1 shard each) | Verdict |
|------|----------------------|---------|
| instance_usage ↔ instance_events | ~11% | Independently hash-sharded |
| instance_usage ↔ machine_events | ~100% | Works (few unique machines) |
| instance_usage ↔ collection_events | ~50% | Partial |

Because the tables are **independently hash-sharded**, downloading a subset of
shards (e.g., 3 out of thousands) gives unreliable cross-table joins.
To get a properly joined dataset locally, you would need to download
**all shards** — approximately **1 TB+** for cell `e` alone.

### 3. Download Feasibility

- Full cell `e` download: ~819 GB–1 TB
- Even on a fast fiber connection (100 Mbps): **~20+ hours**
- Disk space required: **2+ TB** (download + decompress + process)
- RAM for processing joins at that scale: **32+ GB minimum**
- Single-cell approach is impractical for a single researcher with consumer hardware

## Viable Path Forward

The project **is still achievable** for a master's thesis — the key insight is
not to download raw GCS shards, but to use **BigQuery for a time-filtered,
properly joined extraction**:

### Approach: BigQuery Extraction → Local Parquet

```
ibis → BigQuery (single scan, ~$0.33)
         ↓
    Parquet (2 weeks, ~4-5 GB)
         ↓
    ibis (local, $0) → EDA + ML training
```

### Why This Works

| Step | Cost | Feasibility |
|------|------|-------------|
| BigQuery staging query | ~$0.33 (or free tier) | Trivial — runs in ~30 seconds |
| Download 5 GB Parquet | Free | Minutes on any connection |
| Local EDA + ML on 5 GB | $0 | Runs on any laptop (8 GB RAM is enough) |

### Timeline (3 months = feasible)

| Month | Task | Data size |
|-------|------|-----------|
| 1 | BigQuery extraction → Parquet | ~$0.33, 30 min |
| 1–2 | EDA + feature engineering | 5 GB local |
| 2–3 | ML/DL model training | 5 GB local |
| 3 | Evaluation + thesis writing | — |

### Hardware Needed

- **Any standard laptop** (8–16 GB RAM, 50 GB free disk)
- No GPU required for initial models (CPU training on 5 GB is fine)
- Cloud GPU (Colab free tier) for deep learning if desired

## Recommendation

**Do NOT download raw GCS shards.** The BigQuery staging approach is:

- **Cheaper**: ~$0.33 vs months of downloading
- **Faster**: 30 seconds vs 20+ hours
- **More accurate**: Properly joined data from a single query
- **Lighter**: 5 GB vs 1 TB of raw files
- **Repeatable**: Change the WHERE clause, re-run for 2 cents

The Google Borg ClusterData 2019 is **not** practical to fully download and
process locally with consumer hardware. But it **is** practical to analyze
via BigQuery extraction — which was the plan from the start.
