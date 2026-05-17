# Temporal Fusion Transformer (TFT) — Findings

## Overview
TFT (Temporal Fusion Transformer) by Lim et al. is SOTA for multi-horizon forecasting with interpretable attention mechanisms.

## Requirements
- Multi-VM **panel data** (not single-series)
- Significant computational resources
- Complex hyperparameter tuning

## Decision
Not implemented in `03c_timeseries_forecasting.ipynb` due to:
- Data limitations (25/195 CPU shards available)
- Complexity relative to project scope

Acknowledged as recommended **future work**.

## Reference
Lim, B. et al. "Temporal Fusion Transformers for interpretable multi-horizon time series forecasting." *International Journal of Forecasting*, 2021.
