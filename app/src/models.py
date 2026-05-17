"""
Model wrapper classes for consistent fit/predict/evaluate/save interface.

Provides unified API for sklearn, XGBoost, CatBoost, clustering, and anomaly detection models.
All models follow BaseModel abstract interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Sklearn imports
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# Gradient boosting
import xgboost as xgb


class BaseModel(ABC):
    """Abstract base class for all model wrappers."""

    def __init__(self, task: str = "regression", params: Dict[str, Any] | None = None, random_state: int = 42):
        """
        Initialize base model.

        Parameters
        ----------
        task : str
            "regression" or "classification"
        params : dict, optional
            Hyperparameters for the underlying estimator
        random_state : int
            Seed for reproducibility
        """
        self.task = task
        self.params = params or {}
        self.random_state = random_state
        self.estimator = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit model to training data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass

    def evaluate(self, X: np.ndarray, y: np.ndarray, task: str | None = None) -> Dict[str, float]:
        """
        Compute standard metrics based on task type.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        task : str, optional
            If None, uses self.task

        Returns
        -------
        dict
            Metrics keyed by metric name
        """
        task = task or self.task
        y_pred = self.predict(X)

        metrics = {}

        if task == "regression":
            metrics['mae'] = mean_absolute_error(y, y_pred)
            metrics['rmse'] = np.sqrt(mean_squared_error(y, y_pred))
            metrics['r2'] = r2_score(y, y_pred)
            metrics['mse'] = mean_squared_error(y, y_pred)
            metrics['mape'] = np.mean(np.abs((y - y_pred) / (y + 1e-6))) * 100
            denom = np.sum(np.abs(y)) + 1e-6
            metrics['wmape'] = np.sum(np.abs(y - y_pred)) / denom * 100

        elif task == "classification":
            metrics['accuracy'] = accuracy_score(y, y_pred)
            metrics['precision'] = precision_score(y, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y, y_pred, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(y, y_pred, average='weighted', zero_division=0)
            if hasattr(self.estimator, 'predict_proba'):
                try:
                    metrics['roc_auc'] = roc_auc_score(y, self.estimator.predict_proba(X)[:, 1], multi_class='ovr')
                except:
                    metrics['roc_auc'] = np.nan

        return metrics

    def save(self, path: str, metadata: Dict[str, Any] | None = None) -> str:
        """
        Save model to disk.

        Parameters
        ----------
        path : str
            File path for saving
        metadata : dict, optional
            Additional metadata to save

        Returns
        -------
        str
            Path where model was saved
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        save_dict = {
            'estimator': self.estimator,
            'task': self.task,
            'params': self.params,
            'random_state': self.random_state,
            'metadata': metadata or {}
        }
        joblib.dump(save_dict, path)
        return path

class RidgeModel(BaseModel):
    """Ridge regression wrapper."""

    def __init__(self, task: str = "regression", params: Dict[str, Any] | None = None, random_state: int = 42):
        super().__init__(task, params, random_state)
        params = params or {}
        alpha = params.get('alpha', 1.0)
        self.estimator = Ridge(alpha=alpha, random_state=random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit ridge model."""
        self.estimator.fit(X, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.estimator.predict(X)


class RandomForestModel(BaseModel):
    """Random Forest wrapper for regression or classification."""

    def __init__(self, task: str = "regression", params: Dict[str, Any] | None = None, random_state: int = 42):
        super().__init__(task, params, random_state)
        params = params or {}
        n_estimators = params.get('n_estimators', 300)
        max_depth = params.get('max_depth', 15)

        if task == "regression":
            self.estimator = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1
            )
        else:
            self.estimator = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1,
                class_weight='balanced'
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit random forest."""
        self.estimator.fit(X, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.estimator.predict(X)


class XGBoostModel(BaseModel):
    """XGBoost wrapper for regression or classification."""

    def __init__(self, task: str = "regression", params: Dict[str, Any] | None = None, random_state: int = 42):
        super().__init__(task, params, random_state)
        params = params or {}
        default_params = {
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'n_estimators': 300,
            'random_state': random_state,
            'n_jobs': -1
        }
        default_params.update(params)

        if task == "regression":
            self.estimator = xgb.XGBRegressor(**default_params)
        else:
            default_params['objective'] = 'binary:logistic'
            default_params['scale_pos_weight'] = 1
            self.estimator = xgb.XGBClassifier(**default_params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit XGBoost model."""
        self.estimator.fit(X, y, verbose=0)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.estimator.predict(X)


class ClusterModel(BaseModel):
    """K-Means clustering wrapper."""

    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        super().__init__(task="clustering", random_state=random_state)
        self.n_clusters = n_clusters
        self.estimator = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.scaler = StandardScaler()
        self.is_scaled = False

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> None:
        """
        Fit K-Means model.

        Note: y is ignored for clustering; included for API consistency.
        """
        X_scaled = self.scaler.fit_transform(X)
        self.estimator.fit(X_scaled)
        self.is_fitted = True
        self.is_scaled = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get cluster labels."""
        if self.is_scaled:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        return self.estimator.predict(X_scaled)

    def save(self, path: str, metadata: Dict[str, Any] | None = None) -> str:
        """Save cluster model with scaler."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        save_dict = {
            'estimator': self.estimator,
            'scaler': self.scaler,
            'n_clusters': self.n_clusters,
            'task': self.task,
            'params': self.params,
            'random_state': self.random_state,
            'metadata': metadata or {}
        }
        joblib.dump(save_dict, path)
        return path


class GenericModel(BaseModel):
    """Concrete fallback for loading unknown saved models."""

    def __init__(self, task: str = "regression", params: Dict[str, Any] | None = None, random_state: int = 42):
        super().__init__(task, params, random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit using stored estimator."""
        self.estimator.fit(X, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using stored estimator."""
        return self.estimator.predict(X)


class AnomalyModel(BaseModel):
    """Isolation Forest for anomaly detection."""

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        super().__init__(task="anomaly", random_state=random_state)
        self.contamination = contamination
        self.estimator = IsolationForest(contamination=contamination, random_state=random_state)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> None:
        """
        Fit Isolation Forest.

        Note: y is ignored for anomaly detection; included for API consistency.
        """
        self.estimator.fit(X)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies.

        Returns
        -------
        np.ndarray
            -1 for anomalies, 1 for normal points
        """
        return self.estimator.predict(X)


def load_model(path: str, model_type: str | None = None) -> BaseModel:
    """
    Load a saved model from disk.

    Parameters
    ----------
    path : str
        Path to saved model file
    model_type : str, optional
        Model type hint for loading. If None, infers from file content.

    Returns
    -------
    BaseModel
        Loaded model instance
    """
    data = joblib.load(path)

    # If it's a dict with 'estimator' key, reconstruct the model wrapper
    if isinstance(data, dict) and 'estimator' in data:
        estimator = data['estimator']
        task = data.get('task', 'regression')

        # Reconstruct appropriate wrapper class
        if isinstance(estimator, KMeans):
            model = ClusterModel(n_clusters=estimator.n_clusters)
            model.estimator = estimator
            model.scaler = data.get('scaler')
            model.is_scaled = True
        elif isinstance(estimator, IsolationForest):
            model = AnomalyModel()
            model.estimator = estimator
        else:
            # Generic wrapper
            model = GenericModel()
        model.estimator = estimator
        model.task = task

        model.is_fitted = True
        return model

    return data


def train_test_split_by_time(
    df: pd.DataFrame,
    timestamp_col: str,
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological train/test split (no data leakage).

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    timestamp_col : str
        Column name with timestamps
    test_size : float
        Fraction for test set

    Returns
    -------
    train_df, test_df : (pd.DataFrame, pd.DataFrame)
    """
    df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_size))
    return df_sorted[:split_idx], df_sorted[split_idx:]
