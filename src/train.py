"""
Training script with MLflow tracking and hyperparameter tuning.
"""

import os
import sys
from pathlib import Path

import click
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import build_pipeline, get_param_grid
from src.utils import load_config, load_data, split_data


def setup_mlflow(experiment_name: str | None = None) -> str:
    """
    Setup MLflow tracking.

    Args:
        experiment_name: Name of the experiment.

    Returns:
        Experiment ID.
    """
    # Get experiment name from env or parameter
    exp_name = experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME", "churn-classifier")

    # Set tracking URI if specified
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    # Create or get experiment
    experiment = mlflow.get_experiment_by_name(exp_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(exp_name)
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(exp_name)

    return experiment_id


def train(config_path: str, experiment_name: str | None = None) -> str:
    """
    Train model with hyperparameter tuning and MLflow tracking.

    Args:
        config_path: Path to configuration file.
        experiment_name: Optional MLflow experiment name.

    Returns:
        Run ID of the training run.
    """
    # Load configuration
    config = load_config(config_path)

    # Setup MLflow
    setup_mlflow(experiment_name)

    # Enable autologging
    mlflow.sklearn.autolog(log_models=True, log_input_examples=True)

    # Load and split data
    print(f"Loading data from {config['data']['csv_path']}...")
    X, y = load_data(config["data"]["csv_path"], config["data"]["target"])

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Target distribution (train): {y_train.value_counts(normalize=True).to_dict()}")

    # Build pipeline
    model_type = config["model"]["type"]
    numeric_features = config["features"]["numeric"]
    categorical_features = config["features"]["categorical"]

    pipeline = build_pipeline(
        numeric=numeric_features,
        categorical=categorical_features,
        model_type=model_type,
    )

    # Setup cross-validation
    cv_config = config["cv"]
    cv = StratifiedKFold(
        n_splits=cv_config["n_splits"],
        shuffle=True,
        random_state=config["data"]["random_state"],
    )

    # Get parameter grid
    param_grid = get_param_grid(model_type, config["model"]["params"])

    print(f"\nStarting GridSearchCV with {cv_config['n_splits']}-fold CV...")
    print(f"Parameter grid: {param_grid}")
    print(f"Scoring: {cv_config['scoring']}")

    # Start MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"\nMLflow Run ID: {run_id}")

        # Log configuration
        mlflow.log_params(
            {
                "model_type": model_type,
                "test_size": config["data"]["test_size"],
                "cv_splits": cv_config["n_splits"],
                "scoring": cv_config["scoring"],
                "n_features_numeric": len(numeric_features),
                "n_features_categorical": len(categorical_features),
            }
        )

        # GridSearchCV
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring=cv_config["scoring"],
            n_jobs=-1,
            verbose=1,
            return_train_score=True,
        )

        # Fit
        grid_search.fit(X_train, y_train)

        # Log best parameters and score
        print(f"\nBest CV Score ({cv_config['scoring']}): {grid_search.best_score_:.4f}")
        print(f"Best Parameters: {grid_search.best_params_}")

        mlflow.log_params(
            {f"best_{k}": v for k, v in grid_search.best_params_.items()}
        )
        mlflow.log_metric("best_cv_score", grid_search.best_score_)

        # Evaluate on test set
        test_score = grid_search.score(X_test, y_test)
        print(f"Test Score ({cv_config['scoring']}): {test_score:.4f}")
        mlflow.log_metric("test_score", test_score)

        # Log the best model
        mlflow.sklearn.log_model(
            grid_search.best_estimator_,
            "model",
            registered_model_name="ChurnClassifier",
        )

        # Save test data indices for evaluation
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_test_samples", len(X_test))

        # Save model to artifacts folder
        artifacts_dir = Path(__file__).parent.parent / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        model_path = artifacts_dir / "best_model.joblib"
        joblib.dump(grid_search.best_estimator_, model_path)
        print(f"  Model saved to: {model_path}")

        print(f"\n✓ Training complete! Model logged to MLflow.")
        print(f"  Run ID: {run_id}")
        print(f"  Model registered as: ChurnClassifier")

    return run_id


@click.command()
@click.option(
    "--config",
    "-c",
    default="configs/config.yaml",
    help="Path to configuration file.",
)
@click.option(
    "--experiment",
    "-e",
    default=None,
    help="MLflow experiment name (overrides env variable).",
)
def main(config: str, experiment: str | None):
    """Train churn classifier with MLflow tracking."""
    train(config, experiment)


if __name__ == "__main__":
    main()
