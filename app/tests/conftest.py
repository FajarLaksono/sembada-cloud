"""Pytest configuration and shared fixtures for feature and model tests."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


DATA_DIR = Path("data/transformed/parquet")


@pytest.fixture(scope="session")
def vmtable_sample() -> pd.DataFrame:
    """
    Load a sample of vmtable.parquet for testing.

    If file doesn't exist, creates synthetic data for testing purposes.
    """
    vmtable_path = DATA_DIR / "vmtable.parquet"

    if vmtable_path.exists():
        df = pd.read_parquet(vmtable_path)
        return df.sample(min(1000, len(df)), random_state=42)
    else:
        # Create synthetic test data
        n_samples = 1000
        return pd.DataFrame({
            'vm_id': [f'vm_{i}' for i in range(n_samples)],
            'subscription_id': [f'sub_{i % 10}' for i in range(n_samples)],
            'deployment_id': [f'dep_{i % 5}' for i in range(n_samples)],
            'timestamp_created': np.random.randint(0, 86400 * 30, n_samples),
            'timestamp_deleted': np.random.randint(86400, 86400 * 30, n_samples),
            'max_cpu': np.random.uniform(0, 100, n_samples),
            'avg_cpu': np.random.uniform(0, 100, n_samples),
            'p95_max_cpu': np.random.uniform(0, 100, n_samples),
            'vm_category': np.random.choice(['Interactive', 'Batch', 'Unknown'], n_samples),
            'vm_core_count_bucket': np.random.choice(['2', '4', '8', '24', '>24'], n_samples),
            'vm_memory_gb_bucket': np.random.choice(['4', '8', '32', '64', '>64'], n_samples),
        })


@pytest.fixture(scope="session")
def pricing_sample() -> pd.DataFrame:
    """Load pricing data for testing. Generates synthetic data if file missing."""
    path = DATA_DIR / "azure_pricing.parquet"
    if path.exists():
        return pd.read_parquet(path)
    rng = np.random.default_rng(42)
    core_buckets = ['2', '4', '8', '24', '>24']
    mem_buckets = ['4', '8', '32', '64', '>64']
    rows = []
    for c in core_buckets:
        for m in mem_buckets:
            rows.append({
                'core_bucket': c,
                'mem_bucket': m,
                'rate_per_hour': round(float(rng.uniform(0.02, 0.50)), 4),
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def subscriptions_sample() -> pd.DataFrame:
    """Load subscriptions data for testing. Generates synthetic data if file missing."""
    path = DATA_DIR / "subscriptions.parquet"
    if path.exists():
        return pd.read_parquet(path)
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        'subscription_id': [f'sub_{i}' for i in range(10)],
        'vm_count': rng.integers(50, 200, 10),
        'first_vm_timestamp': rng.integers(0, 86400 * 30, 10),
    })


@pytest.fixture(scope="session")
def deployments_sample() -> pd.DataFrame:
    """Load deployments data for testing. Generates synthetic data if file missing."""
    path = DATA_DIR / "deployments.parquet"
    if path.exists():
        return pd.read_parquet(path)
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        'deployment_id': [f'dep_{i}' for i in range(5)],
        'deployment_size': rng.choice(['small', 'medium', 'large'], 5),
    })


@pytest.fixture(scope="session")
def engineered_features(vmtable_sample, pricing_sample,
                         subscriptions_sample,
                         deployments_sample) -> pd.DataFrame:
    """
    Create engineered features from sample data.

    Uses app.src.features.create_features on vmtable_sample
    with all auxiliary data sources.
    """
    from app.src.features import create_features

    return create_features(vmtable_sample, pricing_sample,
                           subscriptions_sample, deployments_sample)


@pytest.fixture(scope="session")
def sequence_data() -> tuple:
    """
    Generate synthetic timeseries data for sequence testing.

    Returns (X_seq, y_seq) from create_sequences with synthetic sine wave data.
    """
    # Create synthetic timeseries: sine wave + noise
    t = np.linspace(0, 100, 500)
    data = np.sin(t) + np.random.normal(0, 0.1, 500)

    from app.src.features import create_sequences

    return create_sequences(data, lookback=24)


@pytest.fixture(scope="function")
def toy_regression_data():
    """
    Simple regression test data (feature matrix X, target y).

    100 samples, 5 features, synthetic target.
    """
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.normal(0, 0.1, 100)
    return X, y


@pytest.fixture(scope="function")
def toy_classification_data():
    """
    Simple classification test data (feature matrix X, binary target y).

    100 samples, 5 features, binary classification.
    """
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y
