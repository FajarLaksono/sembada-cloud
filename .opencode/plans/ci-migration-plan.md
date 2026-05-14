# Final Plan: CI, Data Strategy, and Notebook Execution

## Changes Applied

### 1. Synthetic Fixtures (conftest.py)

All 3 auxiliary fixtures now generate synthetic data when parquet files are missing:
- `pricing_sample` — cartesian product of core/mem buckets with rate_per_hour
- `subscriptions_sample` — 10 subscription IDs matching vmtable_sample
- `deployments_sample` — 5 deployment IDs matching vmtable_sample

Tests pass without any real data files.

### 2. CI Architecture

Single workflow `.github/workflows/ci.yml` — pytest only, auto on push/PR:
- All fixtures are synthetic — no real data needed
- Fast (~1 min)
- Clean status checks (no ghost jobs)

### 3. `notebooks.yml` — DELETED

No CI notebook execution. Notebooks run locally only. Quality gates are embedded in the notebooks and fire identically regardless of environment.

### 4. Limitations Documented

Added to `docs/AZURE_PREDICTIVE_ANALYSIS_PLAN.md` §7.1:
- 100GB CPU readings cannot fit CI
- Notebooks local-only
- Compliance preserved (gates in notebook, `run_log.csv`, documented commands)
- Full automation tracked as future work

## Files Updated

| File | Change |
|------|--------|
| `.github/workflows/ci.yml` | **No change** (already correct — test only) |
| `.github/workflows/notebooks.yml` | **Deleted** |
| `app/tests/conftest.py` | Synthetic fallbacks for pricing, subscriptions, deployments |
| `AGENTS.md` | CI/CD section simplified |
| `docs/RUNNING.md` | CI/CD section simplified to single workflow |
| `docs/AZURE_PREDICTIVE_ANALYSIS_PLAN.md` | Removed notebooks.yml refs; added §7.1 Limitations |
| `docs/AUTOMATED_QA_TECH_SPEC.md` | Removed notebooks.yml YAML; simplified Out of Scope §1.3, Pipeline Behavior §10.4, checklist, AGENTS.md section |
| `docs/IMPLEMENTATION_PROGRESS.md` | CI checklist item updated |
| `.opencode/plans/ci-migration-plan.md` | This file |
