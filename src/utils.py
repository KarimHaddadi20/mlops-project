"""
Utility functions for data loading, splitting, and plotting.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    classification_report,
)
from sklearn.model_selection import train_test_split


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the config YAML file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_data(csv_path: str, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load dataset from CSV and separate features from target.

    Args:
        csv_path: Path to the CSV file.
        target: Name of the target column.

    Returns:
        Tuple of (features DataFrame, target Series).
    """
    df = pd.read_csv(csv_path)

    # Clean TotalCharges column (has some whitespace values)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Convert target to binary if needed
    if df[target].dtype == "object":
        df[target] = (df[target] == "Yes").astype(int)

    # Drop customerID if present (not a feature)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    X = df.drop(columns=[target])
    y = df[target]

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets with stratification.

    Args:
        X: Features DataFrame.
        y: Target Series.
        test_size: Proportion of test set.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def plot_roc_curve(model, X_test, y_test, save_path: str | None = None) -> plt.Figure:
    """
    Plot ROC curve.

    Args:
        model: Trained model with predict_proba method.
        X_test: Test features.
        y_test: Test labels.
        save_path: Path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title("ROC Curve")
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_pr_curve(model, X_test, y_test, save_path: str | None = None) -> plt.Figure:
    """
    Plot Precision-Recall curve.

    Args:
        model: Trained model with predict_proba method.
        X_test: Test features.
        y_test: Test labels.
        save_path: Path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title("Precision-Recall Curve")
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_confusion_matrix(
    model, X_test, y_test, save_path: str | None = None
) -> plt.Figure:
    """
    Plot confusion matrix.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.
        save_path: Path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, ax=ax, cmap="Blues", values_format="d"
    )
    ax.set_title("Confusion Matrix")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def get_classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    """
    Calculate classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities (optional).

    Returns:
        Dictionary of metrics.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }

    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

    return metrics


def save_predictions(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    save_path: str,
) -> None:
    """
    Save predictions to CSV for error analysis.

    Args:
        X_test: Test features.
        y_test: True labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities.
        save_path: Path to save CSV.
    """
    results = X_test.copy()
    results["y_true"] = y_test.values
    results["y_pred"] = y_pred
    results["y_proba"] = y_proba
    results["correct"] = results["y_true"] == results["y_pred"]

    results.to_csv(save_path, index=False)


def ensure_dir(path: str) -> None:
    """Ensure directory exists."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
