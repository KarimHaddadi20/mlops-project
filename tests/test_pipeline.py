"""
Tests for the ML pipeline module.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.pipeline import (
    build_pipeline,
    build_preprocessor,
    get_model,
    get_param_grid,
)


class TestBuildPreprocessor:
    """Tests for build_preprocessor function."""

    def test_creates_column_transformer(self):
        """Test that preprocessor is created correctly."""
        preprocessor = build_preprocessor(["a", "b"], ["c", "d"])
        assert hasattr(preprocessor, "transformers")
        assert len(preprocessor.transformers) == 2

    def test_handles_empty_numeric(self):
        """Test with no numeric features."""
        preprocessor = build_preprocessor([], ["c", "d"])
        assert preprocessor is not None

    def test_handles_empty_categorical(self):
        """Test with no categorical features."""
        preprocessor = build_preprocessor(["a", "b"], [])
        assert preprocessor is not None


class TestGetModel:
    """Tests for get_model function."""

    def test_returns_logistic_regression(self):
        """Test logreg model creation."""
        model = get_model("logreg")
        assert model.__class__.__name__ == "LogisticRegression"

    def test_returns_random_forest(self):
        """Test random forest model creation."""
        model = get_model("random_forest")
        assert model.__class__.__name__ == "RandomForestClassifier"

    def test_raises_on_unknown_type(self):
        """Test error on unsupported model type."""
        with pytest.raises(ValueError, match="Unsupported model_type"):
            get_model("unknown_model")

    def test_passes_kwargs_to_model(self):
        """Test that kwargs are passed to model."""
        model = get_model("logreg", C=0.5)
        assert model.C == 0.5


class TestBuildPipeline:
    """Tests for build_pipeline function."""

    def test_creates_pipeline(self):
        """Test that pipeline is created with correct steps."""
        pipe = build_pipeline(["a"], ["b"], "logreg")
        assert isinstance(pipe, Pipeline)
        assert "pre" in dict(pipe.named_steps)
        assert "model" in dict(pipe.named_steps)

    def test_pipeline_with_random_forest(self):
        """Test pipeline creation with random forest."""
        pipe = build_pipeline(["a"], ["b"], "random_forest")
        assert pipe.named_steps["model"].__class__.__name__ == "RandomForestClassifier"

    def test_pipeline_can_fit_transform(self):
        """Test that pipeline can fit and transform data."""
        # Create sample data
        X = pd.DataFrame({
            "num1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "num2": [10.0, 20.0, 30.0, 40.0, 50.0],
            "cat1": ["a", "b", "a", "b", "a"],
        })
        y = pd.Series([0, 1, 0, 1, 0])

        pipe = build_pipeline(["num1", "num2"], ["cat1"], "logreg")
        pipe.fit(X, y)

        # Should be able to predict
        predictions = pipe.predict(X)
        assert len(predictions) == len(y)

    def test_pipeline_handles_missing_values(self):
        """Test that pipeline handles missing values."""
        X = pd.DataFrame({
            "num1": [1.0, np.nan, 3.0, 4.0, 5.0],
            "cat1": ["a", "b", None, "b", "a"],
        })
        y = pd.Series([0, 1, 0, 1, 0])

        pipe = build_pipeline(["num1"], ["cat1"], "logreg")
        pipe.fit(X, y)

        # Should not raise an error
        predictions = pipe.predict(X)
        assert len(predictions) == len(y)


class TestGetParamGrid:
    """Tests for get_param_grid function."""

    def test_adds_model_prefix(self):
        """Test that model__ prefix is added."""
        params = {"C": [0.1, 1.0], "penalty": ["l2"]}
        grid = get_param_grid("logreg", params)

        assert "model__C" in grid
        assert "model__penalty" in grid
        assert grid["model__C"] == [0.1, 1.0]

    def test_handles_empty_params(self):
        """Test with empty parameters."""
        grid = get_param_grid("logreg", {})
        assert grid == {}


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline_workflow(self):
        """Test complete pipeline workflow."""
        # Create realistic sample data
        np.random.seed(42)
        n_samples = 100

        X = pd.DataFrame({
            "tenure": np.random.randint(1, 72, n_samples),
            "MonthlyCharges": np.random.uniform(20, 100, n_samples),
            "TotalCharges": np.random.uniform(100, 5000, n_samples),
            "gender": np.random.choice(["Male", "Female"], n_samples),
            "Contract": np.random.choice(
                ["Month-to-month", "One year", "Two year"], n_samples
            ),
        })
        y = pd.Series(np.random.randint(0, 2, n_samples))

        # Build and train pipeline
        pipe = build_pipeline(
            numeric=["tenure", "MonthlyCharges", "TotalCharges"],
            categorical=["gender", "Contract"],
            model_type="logreg",
        )

        pipe.fit(X, y)

        # Test predictions
        predictions = pipe.predict(X)
        probabilities = pipe.predict_proba(X)

        assert predictions.shape == (n_samples,)
        assert probabilities.shape == (n_samples, 2)
        assert all(p in [0, 1] for p in predictions)
