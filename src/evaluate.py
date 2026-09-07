"""
Evaluation script for generating metrics, plots, and artifacts.
"""

import os
import sys
import tempfile
from pathlib import Path

import click
import mlflow
import mlflow.sklearn

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    get_classification_metrics,
    load_config,
    load_data,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    save_predictions,
    split_data,
)


def get_latest_run(experiment_name: str) -> str | None:
    """
    Get the latest run ID from an experiment.

    Args:
        experiment_name: Name of the MLflow experiment.

    Returns:
        Run ID of the latest run, or None if no runs found.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )

    if len(runs) == 0:
        return None

    return runs.iloc[0]["run_id"]


def evaluate(
    config_path: str,
    run_id: str | None = None,
    experiment_name: str | None = None,
) -> dict:
    """
    Evaluate model and log artifacts to MLflow.

    Args:
        config_path: Path to configuration file.
        run_id: MLflow run ID to evaluate. If None, uses latest run.
        experiment_name: MLflow experiment name.

    Returns:
        Dictionary of evaluation metrics.
    """
    # Load configuration
    config = load_config(config_path)

    # Setup MLflow
    exp_name = experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME", "churn-classifier")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(exp_name)

    # Get run ID
    if run_id is None:
        run_id = get_latest_run(exp_name)
        if run_id is None:
            raise ValueError(f"No runs found in experiment '{exp_name}'. Run training first.")
        print(f"Using latest run: {run_id}")

    # Load model from MLflow
    print(f"Loading model from run {run_id}...")
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    # Load and split data (using same split as training)
    print(f"Loading data from {config['data']['csv_path']}...")
    X, y = load_data(config["data"]["csv_path"], config["data"]["target"])

    _, X_test, _, y_test = split_data(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )

    print(f"Test set size: {len(X_test)}")

    # Generate predictions
    print("Generating predictions...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = get_classification_metrics(y_test, y_pred, y_proba)

    print("\n=== Evaluation Metrics ===")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    # Log metrics and artifacts to the existing run
    with mlflow.start_run(run_id=run_id):
        # Log evaluation metrics
        for name, value in metrics.items():
            mlflow.log_metric(f"eval_{name}", value)

        # Create temporary directory for artifacts
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate and log plots
            print("\nGenerating plots...")

            # ROC Curve
            roc_path = os.path.join(tmpdir, "roc_curve.png")
            plot_roc_curve(model, X_test, y_test, roc_path)
            mlflow.log_artifact(roc_path, "plots")
            print(f"  ✓ ROC curve saved")

            # Precision-Recall Curve
            pr_path = os.path.join(tmpdir, "pr_curve.png")
            plot_pr_curve(model, X_test, y_test, pr_path)
            mlflow.log_artifact(pr_path, "plots")
            print(f"  ✓ Precision-Recall curve saved")

            # Confusion Matrix
            cm_path = os.path.join(tmpdir, "confusion_matrix.png")
            plot_confusion_matrix(model, X_test, y_test, cm_path)
            mlflow.log_artifact(cm_path, "plots")
            print(f"  ✓ Confusion matrix saved")

            # Save predictions for error analysis
            predictions_path = os.path.join(tmpdir, "predictions.csv")
            save_predictions(X_test, y_test, y_pred, y_proba, predictions_path)
            mlflow.log_artifact(predictions_path, "predictions")
            print(f"  ✓ Predictions CSV saved")

    print(f"\n✓ Evaluation complete! Artifacts logged to run {run_id}")

    return metrics


@click.command()
@click.option(
    "--config",
    "-c",
    default="configs/config.yaml",
    help="Path to configuration file.",
)
@click.option(
    "--run-id",
    "-r",
    default=None,
    help="MLflow run ID to evaluate. If not specified, uses latest run.",
)
@click.option(
    "--experiment",
    "-e",
    default=None,
    help="MLflow experiment name (overrides env variable).",
)
def main(config: str, run_id: str | None, experiment: str | None):
    """Evaluate trained model and generate artifacts."""
    evaluate(config, run_id, experiment)


if __name__ == "__main__":
    main()
