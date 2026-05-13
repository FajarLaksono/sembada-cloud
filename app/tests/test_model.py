"""Unit tests for model wrapper classes (app.src.models)."""

import numpy as np
import pytest
import tempfile
from pathlib import Path


class TestXGBoostModel:
    """Tests for XGBoostModel wrapper."""

    def test_regression_fit_predict(self, toy_regression_data):
        """XGBoostModel fits and predicts for regression."""
        from app.src.models import XGBoostModel

        X, y = toy_regression_data

        model = XGBoostModel(task="regression", params={"n_estimators": 10, "max_depth": 3})
        model.fit(X, y)

        assert model.is_fitted

        preds = model.predict(X)
        assert preds.shape[0] == len(y)
        assert preds.ndim == 1

    def test_classification_fit_predict(self, toy_classification_data):
        """XGBoostModel fits and predicts for classification."""
        from app.src.models import XGBoostModel

        X, y = toy_classification_data

        model = XGBoostModel(task="classification", params={"n_estimators": 10, "max_depth": 3})
        model.fit(X, y)

        preds = model.predict(X)
        assert preds.shape[0] == len(y)
        assert np.isin(preds, [0, 1]).all()

    def test_regression_evaluate(self, toy_regression_data):
        """Model.evaluate returns regression metrics."""
        from app.src.models import XGBoostModel

        X, y = toy_regression_data

        model = XGBoostModel(task="regression", params={"n_estimators": 10})
        model.fit(X, y)

        metrics = model.evaluate(X, y)

        assert isinstance(metrics, dict)
        for metric in ['mae', 'rmse', 'r2', 'mse', 'mape']:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float, np.number))

    def test_classification_evaluate(self, toy_classification_data):
        """Model.evaluate returns classification metrics."""
        from app.src.models import XGBoostModel

        X, y = toy_classification_data

        model = XGBoostModel(task="classification", params={"n_estimators": 10})
        model.fit(X, y)

        metrics = model.evaluate(X, y)

        assert isinstance(metrics, dict)
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            assert metric in metrics


class TestClusterModel:
    """Tests for ClusterModel (K-Means) wrapper."""

    def test_kmeans_fit_predict(self, toy_regression_data):
        """ClusterModel fits K-Means and produces cluster labels."""
        from app.src.models import ClusterModel

        X, _ = toy_regression_data

        model = ClusterModel(n_clusters=4)
        model.fit(X)

        assert model.is_fitted

        labels = model.predict(X)

        # Labels should be in range 0..k-1 (may not populate all clusters with small data)
        assert set(labels) <= {0, 1, 2, 3}
        assert len(set(labels)) >= 1
        assert labels.shape[0] == len(X)

    def test_kmeans_no_negative_labels(self, toy_regression_data):
        """K-Means cluster labels should not be -1 (unlike some anomaly detectors)."""
        from app.src.models import ClusterModel

        X, _ = toy_regression_data

        model = ClusterModel(n_clusters=4)
        model.fit(X)

        labels = model.predict(X)

        assert -1 not in labels


class TestAnomalyModel:
    """Tests for AnomalyModel (Isolation Forest) wrapper."""

    def test_anomaly_fit_predict(self, toy_regression_data):
        """AnomalyModel fits and detects anomalies."""
        from app.src.models import AnomalyModel

        X, _ = toy_regression_data

        model = AnomalyModel(contamination=0.1)
        model.fit(X)

        preds = model.predict(X)

        # Should be -1 (anomaly) or 1 (normal)
        assert set(np.unique(preds)) <= {-1, 1}

        # Should have some anomalies
        assert np.sum(preds == -1) > 0


class TestModelPersistence:
    """Tests for model saving and loading."""

    def test_save_load_xgboost(self, toy_regression_data):
        """XGBoost model saves and loads correctly."""
        from app.src.models import XGBoostModel, load_model

        X, y = toy_regression_data

        model = XGBoostModel(task="regression", params={"n_estimators": 10})
        model.fit(X, y)

        preds_before = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"

            # Save
            saved_path = model.save(str(path))
            assert Path(saved_path).exists()

            # Load
            loaded = load_model(str(path))
            preds_after = loaded.predict(X)

            # Predictions should match
            np.testing.assert_array_almost_equal(preds_before, preds_after)

    def test_save_load_kmeans(self, toy_regression_data):
        """K-Means model saves and loads correctly."""
        from app.src.models import ClusterModel, load_model

        X, _ = toy_regression_data

        model = ClusterModel(n_clusters=4)
        model.fit(X)

        labels_before = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "kmeans.pkl"

            # Save
            saved_path = model.save(str(path))
            assert Path(saved_path).exists()

            # Load
            loaded = load_model(str(path))
            labels_after = loaded.predict(X)

            # Labels should match
            np.testing.assert_array_equal(labels_before, labels_after)

    def test_save_includes_metadata(self, toy_regression_data):
        """Model save includes metadata."""
        from app.src.models import XGBoostModel

        X, y = toy_regression_data

        model = XGBoostModel(task="regression")
        model.fit(X, y)

        metadata = {"version": "1.0", "author": "test"}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            model.save(str(path), metadata=metadata)

            # Load and check metadata is preserved
            import joblib

            data = joblib.load(str(path))
            assert data['metadata'] == metadata


class TestLinearModel:
    """Tests for LinearModel wrapper."""

    def test_linear_fit_predict(self, toy_regression_data):
        """LinearModel fits and predicts."""
        from app.src.models import LinearModel

        X, y = toy_regression_data

        model = LinearModel()
        model.fit(X, y)

        preds = model.predict(X)
        assert preds.shape[0] == len(y)


class TestRandomForestModel:
    """Tests for RandomForestModel wrapper."""

    def test_rf_regression_fit_predict(self, toy_regression_data):
        """RandomForestModel regression works."""
        from app.src.models import RandomForestModel

        X, y = toy_regression_data

        model = RandomForestModel(task="regression")
        model.fit(X, y)

        preds = model.predict(X)
        assert preds.shape[0] == len(y)

    def test_rf_classification_fit_predict(self, toy_classification_data):
        """RandomForestModel classification works."""
        from app.src.models import RandomForestModel

        X, y = toy_classification_data

        model = RandomForestModel(task="classification")
        model.fit(X, y)

        preds = model.predict(X)
        assert preds.shape[0] == len(y)
        assert np.isin(preds, [0, 1]).all()


class TestCatBoostModel:
    """Tests for CatBoostModel wrapper."""

    def test_catboost_fit_predict(self, toy_regression_data):
        """CatBoostModel fits and predicts."""
        from app.src.models import CatBoostModel

        X, y = toy_regression_data

        model = CatBoostModel(task="regression", params={"iterations": 10})
        model.fit(X, y)

        preds = model.predict(X)
        assert preds.shape[0] == len(y)
