"""
Pipeline construction module.

Builds scikit-learn Pipeline with ColumnTransformer for preprocessing
and a classification model.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """
    Build a ColumnTransformer for preprocessing numeric and categorical features.

    Args:
        numeric: List of numeric column names.
        categorical: List of categorical column names.

    Returns:
        ColumnTransformer with numeric and categorical pipelines.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric),
            ("cat", categorical_pipeline, categorical),
        ]
    )

    return preprocessor


def get_model(model_type: str = "logreg", **kwargs):
    """
    Get a classification model by type.

    Args:
        model_type: Type of model ("logreg" or "random_forest").
        **kwargs: Additional parameters to pass to the model.

    Returns:
        Initialized classifier.

    Raises:
        ValueError: If model_type is not supported.
    """
    if model_type == "logreg":
        return LogisticRegression(max_iter=500, **kwargs)
    elif model_type == "random_forest":
        return RandomForestClassifier(**kwargs)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Use 'logreg' or 'random_forest'.")


def build_pipeline(
    numeric: list[str],
    categorical: list[str],
    model_type: str = "logreg",
    **model_kwargs,
) -> Pipeline:
    """
    Build complete ML pipeline with preprocessing and model.

    Args:
        numeric: List of numeric column names.
        categorical: List of categorical column names.
        model_type: Type of model ("logreg" or "random_forest").
        **model_kwargs: Additional parameters for the model.

    Returns:
        Complete scikit-learn Pipeline.
    """
    preprocessor = build_preprocessor(numeric, categorical)
    model = get_model(model_type, **model_kwargs)

    pipeline = Pipeline(
        steps=[
            ("pre", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def get_param_grid(model_type: str, params: dict) -> dict:
    """
    Convert config params to GridSearchCV param grid format.

    Args:
        model_type: Type of model.
        params: Parameters dictionary from config.

    Returns:
        Parameter grid with proper prefixes for Pipeline.
    """
    param_grid = {}

    for param_name, values in params.items():
        # Add "model__" prefix for Pipeline
        param_grid[f"model__{param_name}"] = values

    return param_grid
