"""Unit tests for feature engineering module (app.src.features)."""

import pandas as pd
import numpy as np
import pytest


class TestCreateFeatures:
    """Tests for create_features() function."""

    def test_output_shape(self, vmtable_sample):
        """Verify output has expected columns and no nulls."""
        from app.src.features import create_features

        result = create_features(vmtable_sample)

        # Check output is DataFrame
        assert isinstance(result, pd.DataFrame)

        # Check no rows dropped
        assert len(result) == len(vmtable_sample)

        # Check no null values in core features
        core_cols = ['core_count', 'memory_gb', 'lifetime_hours', 'cpu_per_core',
                     'burstiness', 'is_short_lived', 'is_idle', 'waste_tier']
        for col in core_cols:
            assert col in result.columns
            assert result[col].isnull().sum() == 0, f"Column {col} has nulls"

    def test_core_count_parsing(self):
        """Verify bucket string parsing for core count."""
        from app.src.features import create_features

        df = pd.DataFrame({
            'vm_id': ['a', 'b', 'c'],
            'vm_core_count_bucket': ['2', '4', '>24'],
            'vm_memory_gb_bucket': ['4', '8', '64'],
            'timestamp_created': [0, 0, 0],
            'timestamp_deleted': [3600, 3600, 3600],
            'max_cpu': [50.0, 80.0, 90.0],
            'avg_cpu': [10.0, 5.0, 50.0],
            'p95_max_cpu': [30.0, 60.0, 80.0],
            'vm_category': ['Interactive', 'Batch', 'Unknown'],
            'subscription_id': ['s1', 's2', 's3'],
            'deployment_id': ['d1', 'd2', 'd3'],
        })

        result = create_features(df)

        # Test parsing
        assert result.loc[0, 'core_count'] == 2
        assert result.loc[1, 'core_count'] == 4
        assert result.loc[2, 'core_count'] == 48  # >24 maps to 48

    def test_memory_parsing(self):
        """Verify bucket string parsing for memory."""
        from app.src.features import create_features

        df = pd.DataFrame({
            'vm_id': ['a', 'b', 'c'],
            'vm_core_count_bucket': ['2', '4', '8'],
            'vm_memory_gb_bucket': ['4', '8', '>64'],
            'timestamp_created': [0, 0, 0],
            'timestamp_deleted': [3600, 3600, 3600],
            'max_cpu': [50.0, 80.0, 90.0],
            'avg_cpu': [10.0, 5.0, 50.0],
            'p95_max_cpu': [30.0, 60.0, 80.0],
            'vm_category': ['Interactive', 'Batch', 'Unknown'],
            'subscription_id': ['s1', 's2', 's3'],
            'deployment_id': ['d1', 'd2', 'd3'],
        })

        result = create_features(df)

        # Test parsing
        assert result.loc[0, 'memory_gb'] == 4
        assert result.loc[1, 'memory_gb'] == 8
        assert result.loc[2, 'memory_gb'] == 128  # >64 maps to 128

    def test_target_columns(self, engineered_features):
        """Verify target columns have correct types and ranges."""
        df = engineered_features

        # is_idle should be boolean
        assert df['is_idle'].dtype == bool

        # waste_tier should be categorical with 3 levels
        assert df['waste_tier'].dtype.name == 'category'
        assert set(df['waste_tier'].cat.categories) == {'Low', 'Medium', 'High'}

        # waste_fraction should be in [0, 1]
        assert df['waste_fraction'].min() >= -1e-6
        assert df['waste_fraction'].max() <= 1.0 + 1e-6

    def test_cyclical_encoding(self, engineered_features):
        """Verify sin/cos features are bounded in [-1, 1]."""
        df = engineered_features

        for col in ['creation_hour_sin', 'creation_hour_cos', 'creation_dow_sin', 'creation_dow_cos']:
            if col in df.columns:
                assert df[col].min() >= -1.0 - 1e-6, f"{col} has values < -1"
                assert df[col].max() <= 1.0 + 1e-6, f"{col} has values > 1"

    def test_create_sequences_shape(self, sequence_data):
        """Verify sequence shapes match expected dimensions."""
        X_seq, y_seq = sequence_data

        # X should be (n_samples, lookback, 1)
        assert X_seq.ndim == 3, f"X_seq has {X_seq.ndim} dimensions, expected 3"
        assert X_seq.shape[1] == 24, f"X_seq lookback is {X_seq.shape[1]}, expected 24"
        assert X_seq.shape[2] == 1, f"X_seq features is {X_seq.shape[2]}, expected 1"

        # y should be 1D
        assert y_seq.ndim == 1

        # Same number of samples
        assert len(X_seq) == len(y_seq)


class TestGetFeatureTargetColumns:
    """Tests for get_feature_target_columns() function."""

    def test_valid_tasks(self):
        """Verify all valid tasks return feature and target columns."""
        from app.src.features import get_feature_target_columns

        tasks = [
            "regression_avg_cpu",
            "regression_waste",
            "regression_cost",
            "classification_idle",
            "classification_tier"
        ]

        for task in tasks:
            features, target = get_feature_target_columns(task)
            assert isinstance(features, list)
            assert len(features) > 0
            assert isinstance(target, str)
            assert target in [
                "avg_cpu", "waste_fraction", "vm_cost", "is_idle", "waste_tier"
            ]

    def test_invalid_task(self):
        """Invalid task raises ValueError."""
        from app.src.features import get_feature_target_columns

        with pytest.raises(ValueError):
            get_feature_target_columns("invalid_task")

    def test_feature_set_options(self):
        """Different feature_set options return different columns."""
        from app.src.features import get_feature_target_columns

        features_all, _ = get_feature_target_columns("regression_avg_cpu", feature_set="all")
        features_minimal, _ = get_feature_target_columns("regression_avg_cpu", feature_set="minimal")
        features_no_temporal, _ = get_feature_target_columns("regression_avg_cpu", feature_set="no_temporal")

        # All >= minimal
        assert len(features_all) >= len(features_minimal)

        # all includes temporal, no_temporal doesn't
        assert len(features_all) >= len(features_no_temporal)


class TestCreateSequences:
    """Tests for create_sequences() function."""

    def test_sequences_correct_shape(self):
        """Sequences have correct shape."""
        from app.src.features import create_sequences

        data = np.arange(100)
        X, y = create_sequences(data, lookback=24, forecast_horizon=1)

        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == 24  # lookback
        assert X.shape[2] == 1   # features
        assert y.ndim == 1

    def test_sequences_short_data(self):
        """Short data returns empty arrays."""
        from app.src.features import create_sequences

        data = np.arange(5)
        X, y = create_sequences(data, lookback=24, forecast_horizon=1)

        assert len(X) == 0
        assert len(y) == 0

    def test_sequences_values(self):
        """Sequences contain expected values."""
        from app.src.features import create_sequences

        data = np.arange(50)
        X, y = create_sequences(data, lookback=10, forecast_horizon=1)

        # First sequence should be [0, 1, ..., 9], target = 10
        np.testing.assert_array_equal(X[0, :, 0], np.arange(10))
        assert y[0] == 10


class TestMultiTableFeatures:
    """Tests for multi-table feature augmentation."""

    def test_pricing_lookup(self, vmtable_sample, pricing_sample):
        """Rate per hour is assigned correctly from pricing lookup."""
        from app.src.features import create_features

        result = create_features(vmtable_sample, pricing_df=pricing_sample)
        assert 'rate_per_hour' in result.columns
        assert result['rate_per_hour'].min() >= 0
        assert result['vm_cost'].notna().any()
        assert result['vm_cost'].min() >= 0

    def test_subscription_features(self, vmtable_sample, subscriptions_sample):
        """Subscription-level features are present and joined."""
        from app.src.features import create_features

        result = create_features(vmtable_sample, subscriptions_df=subscriptions_sample)
        assert 'sub_vm_count' in result.columns
        assert result['sub_vm_count'].notna().all()

    def test_deployment_features(self, vmtable_sample, deployments_sample):
        """Deployment-level features are present and joined."""
        from app.src.features import create_features

        result = create_features(vmtable_sample, deployments_df=deployments_sample)
        assert 'deployment_size' in result.columns
        assert result['deployment_size'].notna().all()
