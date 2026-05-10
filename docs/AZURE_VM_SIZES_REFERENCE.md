# Azure VM Sizes — Reference for Dataset Buckets

> **Purpose:** Maps the bucket values in the Azure Public Dataset (`vmcorecountbucket`, `vmmemorybucket`) to real Azure VM sizes, so cloud practitioners can understand what instance types the fleet is composed of.

## Core Count → Azure VM Series

| Cores | VM Series (Gen) | Example SKUs |
|-------|-----------------|--------------|
| 2 | **General Purpose (D)** | `Standard_D2s_v5`, `D2as_v5`, `D2ds_v5` |
| | **Burstable (B)** | `Standard_B2s`, `B2ms` |
| | **Compute Optimized (F)** | `Standard_F2s_v2` |
| 4 | **General Purpose (D)** | `Standard_D4s_v5`, `D4as_v5` |
| | **Burstable (B)** | `Standard_B4ms` |
| | **Compute Optimized (F)** | `Standard_F4s_v2` |
| | **Memory Optimized (E)** | `Standard_E4s_v5` |
| 8 | **General Purpose (D)** | `Standard_D8s_v5`, `D8as_v5` |
| | **Compute Optimized (F)** | `Standard_F8s_v2` |
| | **Memory Optimized (E)** | `Standard_E8s_v5` |
| 24 | **General Purpose (D)** | `Standard_D24s_v5`, `D24as_v5` |
| | **Memory Optimized (E)** | `Standard_E24s_v5` |
| >24 | 32–128+ cores | `Standard_D32s_v5` up to `Standard_D128s_v5` |
| | | `Standard_E32s_v5` up to `Standard_E112is_v5` |
| | | `Standard_M` series (memory-intensive, up to 416 cores) |

## Memory Bucket → Typical VM Sizes

| Memory (GB) | Typical VM Series | Example SKU | Typical Core Count |
|-------------|-------------------|-------------|-------------------|
| 2 | Burstable (B) | `Standard_B1ms` (2 GB) | 1 |
| 4 | Burstable (B) | `Standard_B2s` (4 GB) | 2 |
| 8 | General Purpose | `Standard_D2s_v5` (8 GB) | 2 |
| 32 | General Purpose | `Standard_D8s_v5` (32 GB) | 8 |
| 64 | General Purpose / Memory Optimized | `Standard_D16s_v5` / `E16s_v5` | 16 |
| >64 | Memory Optimized | `Standard_E32s_v5` (256 GB) | 32+ |

## Azure VM Series Overview

| Series | AWS Analogy | Use Case | vCPU:RAM Ratio |
|--------|-------------|----------|----------------|
| **D** (General Purpose) | `m` (e.g., `m5.large`) | Balanced compute/memory | 1:4 |
| **E** (Memory Optimized) | `r` (e.g., `r5.large`) | Memory-intensive workloads | 1:8 |
| **F** (Compute Optimized) | `c` (e.g., `c5.large`) | Compute-intensive / batch | 1:2 |
| **B** (Burstable) | `t` (e.g., `t3.medium`) | Low-usage, sporadic workloads | Varies |

## How to Read an Azure SKU

```
Standard_D4s_v5
         ││ │ │
         ││ │ └── Version (v5 = 5th gen)
         ││ └──── s = premium storage capable
         │└────── vCPU count (4 cores)
         └─────── Series (D = general purpose)
```

## Relevance to This Project

The dataset's `vmcorecountbucket` and `vmmemorybucket` capture the **provisioned** (not actual) resources. The waste analysis in section 4.4 compares these provisioned values against `avgcpu` to detect over-provisioning:

- A VM provisioned as `D4s_v5` (4 cores) but averaging 5% CPU is likely over-provisioned — it could be downsized to `D2s_v5` (2 cores) with minimal impact.
- A VM provisioned with 32 GB but using <2 GB average could be moved to a smaller memory tier.
