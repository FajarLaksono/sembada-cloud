---
name: feature-engineering
description: Feature engineering rules and conventions for Azure VM trace data
license: MIT
compatibility: opencode
metadata:
  audience: engineers
  workflow: feature-creation
---

## What I Do

I encode project-specific knowledge about feature engineering in this
codebase. When an agent works on `features.py`, `03a_feature_engineering.ipynb`,
or related files, I provide the rules and conventions so it doesn't have to
re-discover them every time.

## Bucket Parsing Rules

Core count bucket (`vm_core_count_bucket`):

- `>24` maps to 48
- All other values: `int(bucket_str)`
- Missing/NaN defaults to 4

Memory bucket (`vm_memory_gb_bucket`):

- `>64` maps to 128
- All other values: `int(bucket_str)`
- Missing/NaN defaults to 8

## Target Variables

| Column | Type | Definition |
|--------|------|------------|
| `is_idle` | binary | `avg_cpu < 5.0` |
| `waste_fraction` | continuous [0, 1] | `1.0 - (avg_cpu / 100.0)`, clipped [0, 1] |
| `waste_tier` | ordered categorical | Low: < 0.1, Medium: < 0.5, High: >= 0.5 |
| `waste_cost` | continuous | `vm_cost * waste_fraction` |

## Temporal Feature Encoding

- Creation hour: sin/cos transform with period 24
- Creation day-of-week: sin/cos transform with period 7
- Raw `creation_hour` and `creation_dow` columns are dropped after encoding

## Leakage Rules

- Never include `avg_cpu`, `waste_fraction`, `waste_tier`, `waste_cost`,
  `vm_cost`, `is_idle` as features for their respective tasks
- `avg_cpu` may appear as a feature for `regression_cost` (it is not the target)
- `max_cpu` and `p95_max_cpu` may appear as features for all tasks except
  `regression_avg_cpu` (where avg_cpu is the target)

## Pricing Lookup

Rate per hour is determined by matching (`vm_core_count_bucket`,
`vm_memory_gb_bucket`) against `pricing_df` columns (`core_bucket`,
`mem_bucket`). `vm_cost = rate_per_hour * lifetime_hours`.

## Random State

All random operations use `random_state=42` for reproducibility. Missing values
are forward-filled then dropped.
