# Azure Public Datasets for Cloud Resource & Waste Prediction

## Overview

Investigated Microsoft Azure public datasets as an alternative to Google Borg
ClusterData 2019 for the Sembada Cloud thesis project.

Repository: https://github.com/Azure/AzurePublicDataset

## Candidate Datasets

### 1. AzurePublicDatasetV2 (2019 VM Traces) ⭐ RECOMMENDED

**Paper**: "Resource Central: Understanding and Predicting Workloads for Improved
Resource Management in Large Cloud Platforms" (SOSP'17)

#### File Breakdown

| File | Size | Download | Description |
|------|------|----------|-------------|
| `vmtable.csv.gz` | **417 MB** | ✅ Direct, seconds | 2.6M VM summaries with CPU/memory stats |
| `subscriptions.csv.gz` | **0.34 MB** | ✅ Instant | 6,687 subscriptions |
| `deployment.csv.gz` | ~1 MB | ✅ Instant | 33K deployment sizes |
| `vm_cpu_readings-*-of-195.csv.gz` | **~159 GB** | ❌ 195 files × 817 MB each | 5-min CPU timeseries |

#### Schema (vmtable.csv.gz)

| Field | Type | Description |
|-------|------|-------------|
| vmid | STRING_HASH | Encrypted VM ID |
| subscriptionid | STRING_HASH | Encrypted subscription |
| deploymentid | STRING_HASH | Encrypted deployment |
| vmcreated | INTEGER | Timestamp VM created |
| vmdeleted | INTEGER | Timestamp VM deleted |
| maxcpu | DOUBLE | Max CPU utilization over lifetime |
| avgcpu | DOUBLE | Avg CPU utilization over lifetime |
| p95maxcpu | DOUBLE | P95 of max CPU utilization |
| vmcategory | STRING | Interactive / Delay-insensitive / Unknown |
| vmcorecountbucket | STRING | Core count range (e.g., 2, 4, 8) |
| vmmemorybucket | STRING | Memory range in GB (e.g., 4, 8, 32) |

#### Schema (vm_cpu_readings)

| Field | Type | Description |
|-------|------|-------------|
| timestamp | INTEGER | Seconds (every 5 minutes) |
| vmid | STRING_HASH | Encrypted VM ID |
| mincpu | DOUBLE | Min CPU in window |
| maxcpu | DOUBLE | Max CPU in window |
| avgcpu | DOUBLE | Avg CPU in window |

#### Key Finding: CPU reading files are VM-hash-sharded

Each of the 195 CPU reading files covers a **subset of VMs** for the full 30
days (not time-ordered). This means:
- Downloading 1 file = ~10M readings for ~1,159 VMs over 30 days
- To get ALL VMs, you'd need all 195 files (~159 GB)
- Same problem as Google Borg hash sharding

**BUT** — the vmtable.csv.gz (417 MB, single file) already has aggregated CPU
stats (avgcpu, maxcpu, p95maxcpu) for all 2.6M VMs. This is likely sufficient
for waste/over-provisioning analysis without the raw timeseries.

---

### 2. AzureTracesForPacking2020 ⭐ RECOMMENDED

**Paper**: "Protean: VM Allocation Service at Scale" (OSDI'20)

**Format**: SQLite database (single file, ~10 MB)

**Download**: https://azurepublicdatasettraces.blob.core.windows.net/azurepublicdatasetv2/azurevmallocation_dataset2020/AzurePackingTraceV1.zip

#### Schema (VM Requests)

| Field | Description |
|-------|-------------|
| vmId | Anonymized VM ID |
| tenantId | Anonymized tenant |
| vmTypeId | Requested VM type |
| priority | High (0) / Low (1) |
| starttime | Fractional days from collection start |
| endtime | Fractional days (NULL if still alive) |

#### Schema (VM Types)

| Field | Description |
|-------|-------------|
| vmTypeId | VM type name |
| machineId | Hardware generation |
| core | CPU allocation (fractional) |
| memory | Memory allocation (fractional) |
| hdd | HDD allocation (fractional) |
| ssd | SSD allocation (fractional) |
| nic | Network allocation (fractional) |

#### Relevance

- VM resource requirements (CPU/memory/SSD/HDD/NIC) → waste analysis
- Priority-based allocation → over-provisioning patterns
- 14-day collection window

---

### 3. AzurePublicDatasetV1 (2017 VM Traces)

Same structure as V2 but older (2017 data):
- 2M VMs, 1.25B CPU readings
- 78.5 GB compressed (128 CPU files)
- Useful for longitudinal comparison (2017 vs 2019)

---

### 4. Other Datasets (Lower Priority)

| Dataset | Description | Relevance |
|---------|-------------|-----------|
| AzureGreenSKUFramework2023 | Carbon-aware server design | Environmental angle for waste |
| AzureVMNoiseDataset2024 | Performance variability (483 days) | Capacity waste from noise |
| AzureLLMInferenceDataset2023/2024 | LLM inference traces | Not relevant |

---

## Comparison: Azure V2 vs Google Borg

| Criterion | Google Borg | Azure V2 |
|-----------|-------------|----------|
| **Download size (minimal)** | 3 shards × 35 MB = ~106 MB | **vmtable only: 417 MB** |
| **Download size (full)** | 1 TB+ | ~159 GB |
| **Cost** | BigQuery quota / billing | **$0 (direct download)** |
| **Cross-table joins** | ✅ Partially works (small tables) | **✅ Works** |
| **CPU data** | 5-min intervals | **Aggregated (avg/max/p95) + raw optional** |
| **Memory data** | Yes (average_usage, maximum_usage) | Yes (bucket ranges: 4, 8, 32, 64 GB) |
| **Subject** | Borg container scheduling | **Azure VMs (direct match to thesis)** |
| **Active community** | GitHub + mailing list | GitHub + mailing list |
| **LICENSE** | CC-BY 4.0 | CC-BY 4.0 |

## Recommendation

### Phase 1: Download immediately (~430 MB, free)

1. `vmtable.csv.gz` (417 MB) — the core dataset
2. `subscriptions.csv.gz` (0.34 MB)
3. `deployment.csv.gz` (~1 MB)
4. `AzurePackingTraceV1.zip` (~10 MB)

### Phase 2: Experiment with CPU readings (optional)

Download 1-2 CPU reading files (~1.6 GB) to check:
- Time coverage per file
- Whether join keys link to vmtable
- If partial download is sufficient for timeseries analysis

### Phase 3: Waste prediction features

From vmtable alone:
- **Over-provisioning ratio**: `maxcpu / avgcpu`
- **Utilization**: `avgcpu / vmcorecountbucket`
- **Memory utilization**: Based on vmcategory patterns
- **Category analysis**: Interactive vs Delay-insensitive VM behavior
- **Lifetime patterns**: short-lived vs long-lived VM waste

From Packing trace:
- **Allocation efficiency**: requested vs actual resources
- **Priority-based waste**: low-priority VM eviction patterns
- **Resource fragmentation**: per machine type

## Data Access

Azure Blob Storage base URL:
https://azurepublicdatasettraces.blob.core.windows.net/azurepublicdatasetv2/

Download links (raw):
- vmtable: `trace_data/vmtable/vmtable.csv.gz`
- subscriptions: `trace_data/subscriptions/subscriptions.csv.gz`
- deployments: `trace_data/deployment/deployment.csv.gz`
- CPU: `trace_data/vm_cpu_readings/vm_cpu_readings-file-{1..195}-of-195.csv.gz`
- Packing trace: `azurevmallocation_dataset2020/AzurePackingTraceV1.zip`
- Schema: `schema.csv`
- Azure 2019 comparison data: `azure2019_data/{category,cores,cpu,deployment,lifetime,memory}.txt`
- Analysis notebook: `Azure 2019 Public Dataset V2 - Trace Analysis.ipynb`

---

## Cost Analysis Findings (30-Day Trace)

### 5. Total Trace Cost

**Pricing model:** Azure Retail Prices API (East US, pay-as-you-go, Linux), fallback to estimated $0.03/vCPU/hr + $0.004/GB/hr.

| Metric | Value |
|--------|-------|
| **VMs with calculable cost** | 2,551,087 / 2,695,548 |
| **Total cost (30 days)** | **$6,857,797.66** |
| **Total VM hours** | 162,453,670 |
| **Avg cost per VM** | $2.69 |
| **Median cost per VM** | $0.02 |
| **Avg effective hourly rate** | ~$0.042/hr |

### 6. Cost Breakdown by Core Bucket

| Core Bucket | VM Count | Total Cost | Avg Cost/VM |
|-------------|----------|------------|-------------|
| 2 | 1,502,458 | $2,468,380.87 | $1.64 |
| 4 | 776,118 | $1,262,174.46 | $1.63 |
| 8 | 183,887 | $616,372.54 | $3.35 |
| 24 | 78,196 | $1,852,774.56 | $23.69 |
| >24 | 10,428 | $658,095.23 | $63.11 |

### 7. Waste Cost Quantification

| Waste Type | Cost | % of Total |
|------------|------|------------|
| **Total trace cost** | $6,857,797.66 | 100% |
| **Waste cost (unused CPU)** | $6,197,155.10 | 90.4% |
| **Idle VM cost (<5% CPU)** | $3,200,202.27 | 46.7% |

### 8. Context: What Kind of Workloads Cost This Much?

This is **not a single application**. The $6.9M represents the **aggregate compute cost of Microsoft's entire internal Azure fleet** across all engineering teams over 30 days. To understand the scale:

#### 8.1. Who Uses These VMs?

This dataset comes from Microsoft's internal Azure deployment, documented in the **Resource Central** paper (SOSP'17). The VMs are used by thousands of Microsoft engineering teams for:

| Category | % VMs | Typical Workloads |
|----------|-------|-------------------|
| **Unknown** | ~91% | Dev/test environments, CI/CD build agents, temporary scratch VMs, short-lived experiments, engineering sandboxes — could not be classified by the Resource Central system |
| **Interactive** | small | Latency-sensitive user-facing services: web frontends, API gateways, internal dashboards, authentication services |
| **Delay-insensitive** | small | Batch processing, MapReduce jobs, data pipelines, background workers, report generation |

#### 8.2. Why Most VMs Cost So Little (Median $0.02)

86% of VMs live **less than 24 hours** (median lifetime: 0.6 hours). These are typically:

- **CI/CD build agents** — spun up for a build, destroyed after
- **Ephemeral test VMs** — automated test runs that last minutes
- **One-off experiment containers** — data scientists spinning up short-lived compute
- **Scratch/test instances** — engineers testing configurations temporarily

The long tail of expensive VMs (24-core, >24-core) that run for days are likely production services or long-running batch jobs, accounting for the bulk of total cost despite being few in number.

#### 8.3. Is $6.9M Realistic?

| Perspective | Calculation |
|-------------|-------------|
| **Per VM per month** | $2.69 average → extremely low; most are short-lived |
| **Effective hourly rate** | $0.042/hr → consistent with small (2-4 core, low memory) VMs |
| **Yearly extrapolation** | ~$83M/year for ~2.6M internal VMs → reasonable for a company Microsoft's size |
| **Azure revenue context** | Azure generates $60B+/year revenue; $83M internal spend is ~0.14% |

**Conclusion:** The $6.9M figure is accurate for the dataset scope (30-day trace of Microsoft's internal fleet at retail pay-as-you-go rates). Actual costs would be lower with reserved instances and enterprise discounts, but the number serves as a valid baseline for waste analysis.

### 9. Key Takeaway for Thesis

The value of this analysis is not the absolute dollar amount, but the **90.4% waste ratio**: for every dollar Microsoft spends on these internal VMs, ~90 cents pays for unused CPU capacity. This demonstrates the massive optimization opportunity that predictive models (planned for `03_predictive_analysis.ipynb`) could address through rightsizing, smarter scheduling, and idle detection.
