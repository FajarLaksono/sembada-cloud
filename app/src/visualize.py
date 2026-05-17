"""
Publication-quality visualization functions for model analysis and reporting.

Provides unified plotting functions for:
- Residual analysis
- Feature importance
- Cluster visualization
- Model comparison tables
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional
from scipy import stats


def residual_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Analysis",
    figsize: tuple = (14, 5)
) -> plt.Figure:
    """
    Create side-by-side residual analysis plots.

    Parameters
    ----------
    y_true : np.ndarray
        Actual target values
    y_pred : np.ndarray
        Predicted values
    title : str
        Figure title
    figsize : tuple
        Figure size (width, height)

    Returns
    -------
    plt.Figure
        Matplotlib figure with residual plots
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5, s=20)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Values', fontsize=11)
    axes[0].set_ylabel('Residuals', fontsize=11)
    axes[0].set_title('Residuals vs. Predicted', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)

    # Plot 2: Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


def feature_importance_plot(
    importances: Dict[str, float],
    title: str = "Feature Importance",
    top_n: int = 15,
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Create horizontal bar chart of top-N features by importance.

    Parameters
    ----------
    importances : dict
        Dictionary of feature_name → importance_score
    title : str
        Plot title
    top_n : int
        Number of top features to display
    figsize : tuple
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    # Sort and take top N
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, scores = zip(*sorted_imp)

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(features))

    ax.barh(y_pos, scores, color='steelblue', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=10)
    ax.set_xlabel('Importance Score', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    return fig


def cluster_scatter(
    X_2d: np.ndarray,
    labels: np.ndarray,
    title: str = "Cluster Visualization",
    centroids: Optional[np.ndarray] = None,
    figsize: tuple = (10, 8)
) -> plt.Figure:
    """
    Create 2D scatter plot colored by cluster label.

    Parameters
    ----------
    X_2d : np.ndarray
        2D array of points (e.g., from PCA or t-SNE)
    labels : np.ndarray
        Cluster labels for each point
    title : str
        Plot title
    centroids : np.ndarray, optional
        2D centroids to overlay
    figsize : tuple
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot points colored by cluster
    unique_labels = np.unique(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        mask = labels == label
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[color], label=f'Cluster {label}',
                   alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

    # Overlay centroids if provided
    if centroids is not None:
        ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=300,
                   edgecolors='black', linewidth=2, label='Centroids', zorder=5)

    ax.set_xlabel('Component 1', fontsize=11)
    ax.set_ylabel('Component 2', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    return fig


def comparison_table(
    results: Dict[str, Dict[str, float]],
    highlight_best: bool = True
) -> pd.DataFrame:
    """
    Build formatted comparison DataFrame from model results.

    Parameters
    ----------
    results : dict
        Dictionary of {model_name: {metric_name: value}}
    highlight_best : bool
        Whether to identify best values per metric

    Returns
    -------
    pd.DataFrame
        Formatted comparison table with models as rows, metrics as columns
    """
    df = pd.DataFrame(results).T

    if highlight_best:
        best_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
        for col in df.columns:
            try:
                if col.lower() in ['mae', 'rmse', 'mse', 'wmape']:
                    best_idx = df[col].idxmin()
                else:
                    best_idx = df[col].idxmax()
                best_mask.loc[best_idx, col] = True
            except (TypeError, ValueError):
                pass
        df.attrs['highlight'] = best_mask

    return df
