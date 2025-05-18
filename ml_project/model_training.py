"""Module for model training with cross-validation and hyperparameter tuning."""

import numpy as np
import polars as pl
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

from ml_project.utils import create_model_instance, get_hyperparameter_grid, read_config

# Constants for logging messages
_LOG_TUNE_HYPERPARAMS = "Tuning hyperparameters for model {model_name}"
_LOG_BEST_PARAMS = "Best parameters found: {params}"
_LOG_METRICS = "Model performance metrics on test set: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}"


def tune_hyperparameters(
    x_train: pl.DataFrame,
    y_train: pl.Series,
    model_name: str,
    config: dict[str, any] | None = None,
    cv: int = 5,
    scoring: str = "neg_mean_squared_error",
    n_jobs: int = -1,
) -> tuple[object, dict[str, any]]:
    """Tune hyperparameters for a model using GridSearchCV.

    Args:
        x_train: Feature matrix for training (Polars DataFrame)
        y_train: Target vector for training (Polars Series)
        model_name: Name of the model as specified in the configuration file
        config: Configuration dictionary. If None, reads from default config.toml
        cv: Number of cross-validation folds (default: 5)
        scoring: Scoring metric for cross-validation (default: "neg_mean_squared_error")
        n_jobs: Number of jobs to run in parallel (default: -1, using all processors)

    Returns:
        A tuple containing (best_model, grid_search_results) where best_model is the model
        with the best hyperparameters and grid_search_results is the full GridSearchCV object

    Raises:
        ValueError: If model_name is not recognized
        KeyError: If hyperparameters for the model are not found in the configuration
    """
    logger.info(_LOG_TUNE_HYPERPARAMS.format(model_name=model_name))

    if config is None:
        config = read_config()

    # Get hyperparameter grid from configuration
    param_grid = get_hyperparameter_grid(config, model_name)

    # Convert Polars DataFrame to NumPy arrays for scikit-learn
    x_train_np = x_train.to_numpy()
    y_train_np = y_train.to_numpy()

    # Create base model instance
    base_model = create_model_instance(model_name)

    # Create grid search object
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        return_train_score=True,
        verbose=1,
    )

    # Fit grid search
    grid_search.fit(x_train_np, y_train_np)

    # Log best parameters
    logger.info(_LOG_BEST_PARAMS.format(params=grid_search.best_params_))

    return grid_search.best_estimator_, grid_search


def compute_metrics(
    model: object, x_test: pl.DataFrame, y_test: pl.Series, log_results: bool = True
) -> dict[str, float]:
    """Compute evaluation metrics for a trained model on test data.

    Args:
        model: Trained model object
        x_test: Feature matrix for testing (Polars DataFrame)
        y_test: Target vector for testing (Polars Series)
        log_results: Whether to log the results (default: True)

    Returns:
        Dictionary containing the evaluation metrics: RMSE, MAE, and R²
    """
    # Convert Polars DataFrame to NumPy arrays for scikit-learn
    x_test_np = x_test.to_numpy()
    y_test_np = y_test.to_numpy()

    # Make predictions
    y_pred = model.predict(x_test_np)

    # Compute metrics
    rmse = np.sqrt(mean_squared_error(y_test_np, y_pred))
    mae = mean_absolute_error(y_test_np, y_pred)
    r2 = r2_score(y_test_np, y_pred)

    # Create metrics dictionary
    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }

    # Log metrics if requested
    if log_results:
        logger.info(_LOG_METRICS.format(rmse=rmse, mae=mae, r2=r2))

    return metrics
