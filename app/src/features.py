"""
Feature engineering module for cloud resource prediction.

This module provides functions to:
- Engineer features from raw VM trace data
- Create target variables for different ML tasks
- Generate timeseries sequences for deep learning models
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple


def create_features(
    df: pd.DataFrame,
    pricing_df: pd.DataFrame | None = None,
    subscriptions_df: pd.DataFrame | None = None,
    deployments_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Complete feature engineering pipeline for VM data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw vmtable DataFrame with columns:
        - vm_id, subscription_id, deployment_id
        - timestamp_created, timestamp_deleted
        - max_cpu, avg_cpu, p95_max_cpu
        - vm_category, vm_core_count_bucket, vm_memory_gb_bucket

    pricing_df : pd.DataFrame, optional
        Pricing data for cost calculation. If None, vm_cost is not computed.
        Must contain columns: core_bucket, mem_bucket, rate_per_hour.

    subscriptions_df : pd.DataFrame, optional
        Subscription metadata for subscription-level features.
        Must contain columns: subscription_id, vm_count, first_vm_timestamp.

    deployments_df : pd.DataFrame, optional
        Deployment metadata for deployment-level features.
        Must contain columns: deployment_id, deployment_size.

    Returns
    -------
    pd.DataFrame
        DataFrame with engineered features:
        - Core features: core_count, memory_gb, lifetime_hours
        - Temporal features: creation_hour_sin, creation_hour_cos, creation_dow_sin, creation_dow_cos
        - Ratio features: cpu_per_core, memory_per_core, burstiness, max_to_avg_ratio
        - Binary flags: is_short_lived
        - One-hot encoded: vm_category_*, core_bucket_*, mem_bucket_*
        - Pricing features: rate_per_hour, vm_cost, waste_cost (if pricing_df provided)
        - Subscription features: sub_vm_count, sub_first_vm_ts, sub_tenure (if subscriptions_df provided)
        - Deployment features: deployment_size (if deployments_df provided)
        - Target variables: is_idle, waste_tier
        - Original identifiers: vm_id, subscription_id, deployment_id

    Notes
    -----
    - All random operations use random_state=42 for reproducibility
    - Missing values are forward-filled then dropped
    - Bucket strings are parsed with fallback to default values
    """
    result = df.copy()

    # ============ Parse bucket strings to numeric values ============
    def parse_core_bucket(bucket_str: str) -> int:
        """Parse vm_core_count_bucket to numeric core count."""
        if pd.isna(bucket_str):
            return 4  # default
        bucket_str = str(bucket_str).strip()
        if bucket_str == '>24':
            return 48
        try:
            return int(bucket_str)
        except ValueError:
            return 4

    def parse_memory_bucket(bucket_str: str) -> int:
        """Parse vm_memory_gb_bucket to numeric memory in GB."""
        if pd.isna(bucket_str):
            return 8  # default
        bucket_str = str(bucket_str).strip()
        if bucket_str == '>64':
            return 128
        try:
            return int(bucket_str)
        except ValueError:
            return 8

    result['core_count'] = result['vm_core_count_bucket'].apply(parse_core_bucket)
    result['memory_gb'] = result['vm_memory_gb_bucket'].apply(parse_memory_bucket)

    # ============ Calculate lifetime ============
    result['lifetime_hours'] = (result['timestamp_deleted'] - result['timestamp_created']) / 3600.0

    # ============ Temporal features (cyclical encoding) ============
    # Creation hour (0-23)
    result['creation_hour'] = (result['timestamp_created'] % 86400) // 3600
    result['creation_hour_sin'] = np.sin(2 * np.pi * result['creation_hour'] / 24.0)
    result['creation_hour_cos'] = np.cos(2 * np.pi * result['creation_hour'] / 24.0)

    # Creation day of week (0=Monday, 6=Sunday)
    result['creation_dow'] = (result['timestamp_created'] // 86400) % 7
    result['creation_dow_sin'] = np.sin(2 * np.pi * result['creation_dow'] / 7.0)
    result['creation_dow_cos'] = np.cos(2 * np.pi * result['creation_dow'] / 7.0)

    # ============ Ratio features ============
    result['cpu_per_core'] = result['avg_cpu'] / (result['core_count'] + 1e-6)
    result['memory_per_core'] = result['memory_gb'] / (result['core_count'] + 1e-6)
    result['burstiness'] = result['p95_max_cpu'] / (result['avg_cpu'] + 1e-6)
    result['max_to_avg_ratio'] = result['max_cpu'] / (result['avg_cpu'] + 1e-6)

    # ============ Binary flags ============
    result['is_short_lived'] = (result['lifetime_hours'] < 1).astype(bool)

    # ============ One-hot encoding ============
    if 'vm_category' in result.columns:
        cat_dummies = pd.get_dummies(result['vm_category'], prefix='cat', dtype=bool)
        result = pd.concat([result, cat_dummies], axis=1)

    if 'vm_core_count_bucket' in result.columns:
        core_dummies = pd.get_dummies(result['vm_core_count_bucket'], prefix='core', dtype=bool)
        result = pd.concat([result, core_dummies], axis=1)

    if 'vm_memory_gb_bucket' in result.columns:
        mem_dummies = pd.get_dummies(result['vm_memory_gb_bucket'], prefix='mem', dtype=bool)
        result = pd.concat([result, mem_dummies], axis=1)

    # ============ Pricing features (rate_per_hour lookup) ============
    if pricing_df is not None and 'core_bucket' in pricing_df.columns:
        price_lookup = {}
        for _, row in pricing_df.iterrows():
            price_lookup[(row['core_bucket'], row['mem_bucket'])] = row['rate_per_hour']

        def get_rate(core_b: str, mem_b: str) -> float:
            key = (core_b, mem_b)
            if key in price_lookup:
                return price_lookup[key]
            return 0.0

        result['rate_per_hour'] = result.apply(
            lambda r: get_rate(r['vm_core_count_bucket'], r['vm_memory_gb_bucket']), axis=1
        )
        result['vm_cost'] = result['rate_per_hour'] * result['lifetime_hours']
    else:
        result['rate_per_hour'] = np.nan
        result['vm_cost'] = np.nan

    # ============ Subscription-level features ============
    if subscriptions_df is not None and 'subscription_id' in subscriptions_df.columns:
        sub_features = subscriptions_df[['subscription_id', 'vm_count', 'first_vm_timestamp']].copy()
        sub_features = sub_features.rename(columns={'vm_count': 'sub_vm_count',
                                                    'first_vm_timestamp': 'sub_first_vm_ts'})
        result = result.merge(sub_features, on='subscription_id', how='left')
        result['sub_tenure'] = result['timestamp_created'] - result['sub_first_vm_ts']
    else:
        result['sub_vm_count'] = np.nan
        result['sub_first_vm_ts'] = np.nan
        result['sub_tenure'] = np.nan

    # ============ Deployment-level features ============
    if deployments_df is not None and 'deployment_id' in deployments_df.columns:
        dep_features = deployments_df[['deployment_id', 'deployment_size']].copy()
        result = result.merge(dep_features, on='deployment_id', how='left')
    else:
        result['deployment_size'] = np.nan

    # ============ Target variables ============
    # is_idle: binary, True if avg_cpu < 5%
    result['is_idle'] = (result['avg_cpu'] < 5.0).astype(bool)

    # waste_fraction: continuous [0, 1]
    result['waste_fraction'] = 1.0 - (result['avg_cpu'] / 100.0)
    result['waste_fraction'] = result['waste_fraction'].clip(0, 1)

    # waste_tier: ordinal categorical
    def assign_waste_tier(waste_frac: float) -> str:
        if waste_frac < 0.1:
            return 'Low'
        elif waste_frac < 0.5:
            return 'Medium'
        else:
            return 'High'

    result['waste_tier'] = result['waste_fraction'].apply(assign_waste_tier)
    result['waste_tier'] = pd.Categorical(result['waste_tier'], categories=['Low', 'Medium', 'High'], ordered=True)

    # waste_cost: computed after vm_cost is set above
    if 'vm_cost' in result.columns and 'waste_fraction' in result.columns:
        result['waste_cost'] = result['vm_cost'] * result['waste_fraction']
    else:
        result['waste_cost'] = np.nan

    # ============ Clean and return ============
    result = result.drop(columns=['creation_hour', 'creation_dow'], errors='ignore')

    return result


def get_feature_target_columns(task: str, feature_set: str = "all") -> Tuple[list, str]:
    """
    Return (feature_columns, target_column) for a given ML task.

    Parameters
    ----------
    task : str
        One of:
        - "regression_avg_cpu" — Predict average CPU utilization
        - "regression_waste" — Predict waste fraction
        - "regression_cost" — Predict VM cost
        - "classification_idle" — Binary: is VM idle?
        - "classification_tier" — Multi-class: waste tier (Low/Medium/High)

    feature_set : str, default="all"
        One of:
        - "all" — Include all engineered features
        - "minimal" — Core numeric features only
        - "no_temporal" — Exclude temporal features

    Returns
    -------
    features : list of str
        Column names to use as features
    target : str
        Column name to use as target

    Raises
    ------
    ValueError
        If task or feature_set is not recognized
    """
    # Base features (no target leakage — target-related columns excluded per task below)
    core_features = [
        'core_count', 'memory_gb', 'lifetime_hours',
        'cpu_per_core', 'memory_per_core', 'burstiness', 'max_to_avg_ratio',
        'is_short_lived',
        'rate_per_hour', 'sub_vm_count', 'sub_tenure', 'deployment_size',
    ]

    temporal_features = ['creation_hour_sin', 'creation_hour_cos', 'creation_dow_sin', 'creation_dow_cos']

    # Task-specific extra features (columns not leaked by the target)
    task_extra_features = {
        "regression_avg_cpu": ['max_cpu', 'p95_max_cpu'],
        "regression_waste": ['max_cpu', 'p95_max_cpu'],
        "regression_cost": ['avg_cpu', 'max_cpu', 'p95_max_cpu'],
        "classification_idle": ['max_cpu', 'p95_max_cpu'],
        "classification_tier": ['max_cpu', 'p95_max_cpu'],
    }

    if feature_set == "all":
        features = core_features + temporal_features
    elif feature_set == "minimal":
        features = core_features
    elif feature_set == "no_temporal":
        features = core_features
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    # Add task-specific extra features (safe from target leakage)
    extra = task_extra_features.get(task, [])
    features = features + [c for c in extra if c not in features]

    # Target column
    task_mapping = {
        "regression_avg_cpu": "avg_cpu",
        "regression_waste": "waste_fraction",
        "regression_cost": "vm_cost",
        "classification_idle": "is_idle",
        "classification_tier": "waste_tier",
    }

    if task not in task_mapping:
        raise ValueError(f"Unknown task: {task}")

    target = task_mapping[task]

    return features, target


def load_cpu_readings(data_dir: str | Path, max_vms: int | None = 5) -> pd.DataFrame:
    """
    Load CPU readings shards via DuckDB out-of-core parquet glob.

    Discovers all parquet files in cpu_readings/ subdirectory via DuckDB,
    GROUP BY vm_id to identify top-VMs by trace duration, then fetches
    only their rows. Avoids pd.concat() memory blowup on large shard sets.

    Parameters
    ----------
    data_dir : str or Path
        Path to parquet data directory (e.g., "data/transformed/parquet")
    max_vms : int or None, default=5
        Maximum number of VMs to return. None = return all VMs.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: vm_id, timestamp, avg_cpu, max_cpu, p95_max_cpu
        Returns empty DataFrame if no shards found.
    """
    data_dir = Path(data_dir)
    shards = sorted(data_dir.glob("cpu_readings/*.parquet"))
    if not shards:
        return pd.DataFrame(columns=['vm_id', 'timestamp', 'avg_cpu', 'max_cpu', 'p95_max_cpu'])
    # Use DuckDB for out-of-core parquet reading
    import duckdb
    con = duckdb.connect(":memory:")
    con.execute(f"CREATE VIEW all_readings AS SELECT * FROM read_parquet('{data_dir / 'cpu_readings/*.parquet'}')")
    if max_vms is not None and max_vms > 0:
        # Get VM stats without loading all data
        vm_stats = con.execute("""
            SELECT vm_id, COUNT(*) as count,
                   MIN(timestamp) as min_ts, MAX(timestamp) as max_ts
            FROM all_readings
            GROUP BY vm_id
        """).fetchdf()
        vm_stats['duration_hours'] = (vm_stats['max_ts'] - vm_stats['min_ts']) / 3600
        top_vms = vm_stats.nlargest(max_vms, 'duration_hours')
        # Load only the top VMs' data
        vm_ids = [f"'{v}'" for v in top_vms['vm_id'].tolist()]
        return con.execute(f"""
            SELECT * FROM all_readings
            WHERE vm_id IN ({', '.join(vm_ids)})
            ORDER BY vm_id, timestamp
        """).fetchdf()
    return con.execute("SELECT * FROM all_readings ORDER BY vm_id, timestamp").fetchdf()


def create_sequences(
    data: np.ndarray,
    lookback: int = 24,
    forecast_horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences for timeseries models.

    Parameters
    ----------
    data : np.ndarray
        1D array of timeseries values (e.g., CPU readings over time)

    lookback : int, default=24
        Number of past timesteps to include in each sequence.
        Default 24 corresponds to 2 hours at 5-minute intervals.

    forecast_horizon : int, default=1
        Number of future timesteps to predict.

    Returns
    -------
    X : np.ndarray
        Shape (n_samples, lookback, 1) — input sequences
    y : np.ndarray
        Shape (n_samples,) or (n_samples, forecast_horizon) — target values

    Notes
    -----
    If data length < lookback + forecast_horizon, returns empty arrays.
    """
    if len(data) < lookback + forecast_horizon:
        return np.array([]), np.array([])

    X, y = [], []

    for i in range(len(data) - lookback - forecast_horizon + 1):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback : i + lookback + forecast_horizon].mean())

    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y)

    return X, y
