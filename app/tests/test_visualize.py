"""Tests for visualization functions."""

import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from app.src.visualize import (
    residual_plot,
    feature_importance_plot,
    cluster_scatter,
    comparison_table,
)


class TestResidualPlot:
    def test_returns_figure(self):
        y_true = np.random.randn(50)
        y_pred = y_true + np.random.normal(0, 0.1, 50)
        fig = residual_plot(y_true, y_pred)
        assert isinstance(fig, Figure)

    def test_with_title(self):
        y_true = np.random.randn(50)
        y_pred = y_true + np.random.normal(0, 0.1, 50)
        fig = residual_plot(y_true, y_pred, title="Custom Title")
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestFeatureImportancePlot:
    def test_returns_figure(self):
        importances = {"feat_a": 0.5, "feat_b": 0.3, "feat_c": 0.2}
        fig = feature_importance_plot(importances, top_n=3)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_with_extra_features_truncates(self):
        importances = {f"feat_{i}": float(i) for i in range(20, 0, -1)}
        fig = feature_importance_plot(importances, top_n=5)
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestClusterScatter:
    def test_returns_figure(self):
        X_2d = np.random.randn(100, 2)
        labels = np.random.randint(0, 4, 100)
        fig = cluster_scatter(X_2d, labels)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_with_centroids(self):
        X_2d = np.random.randn(100, 2)
        labels = np.random.randint(0, 4, 100)
        centroids = np.random.randn(4, 2)
        fig = cluster_scatter(X_2d, labels, centroids=centroids)
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestComparisonTable:
    def test_returns_dataframe(self):
        results = {
            "ModelA": {"mae": 1.0, "rmse": 2.0, "r2": 0.9},
            "ModelB": {"mae": 1.5, "rmse": 2.5, "r2": 0.85},
        }
        df = comparison_table(results)
        assert isinstance(df, pd.DataFrame)
        assert list(df.index) == ["ModelA", "ModelB"]

    def test_highlights_best_per_metric(self):
        results = {
            "ModelA": {"mae": 1.0, "r2": 0.85},
            "ModelB": {"mae": 2.0, "r2": 0.95},
        }
        df = comparison_table(results)
        assert "highlight" in df.attrs
        best_mask = df.attrs["highlight"]
        assert best_mask.loc["ModelA", "mae"] == True  # lower MAE is better
        assert best_mask.loc["ModelB", "r2"] == True  # higher R2 is better
