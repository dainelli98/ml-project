"""Module for model inference operations."""

import numpy as np
import polars as pl
from loguru import logger
from sklearn.base import BaseEstimator

# Constants for logging messages
_LOG_INFERENCE = "Running inference with model {model_name} on {n_samples} samples"
_LOG_OUTPUT_SAVED = "Saved inference results to {output_file}"


def run_inference(
    model: BaseEstimator, data: pl.DataFrame, model_name: str = "Unknown", log_results: bool = True
) -> np.ndarray:
    """Run inference on new data using a trained model.

    Args:
        model: Trained model object (loaded from pickle)
        data: Feature matrix for inference (Polars DataFrame)
        model_name: Name of the model for logging (default: "Unknown")
        log_results: Whether to log the inference process (default: True)

    Returns:
        NumPy array containing the predicted values
    """
    # Convert Polars DataFrame to NumPy array for scikit-learn
    data_np = data.to_numpy()

    # Log inference process
    n_samples = data.shape[0]
    if log_results:
        logger.info(_LOG_INFERENCE.format(model_name=model_name, n_samples=n_samples))

    # Make predictions
    return model.predict(data_np)


def save_predictions(predictions: np.ndarray, output_file: str, log_results: bool = True) -> None:
    """Save model predictions to a CSV file.

    Args:
        predictions: NumPy array containing the predicted values
        output_file: Path to the output CSV file
        log_results: Whether to log the saving process (default: True)
    """
    # Convert predictions to Polars DataFrame
    pred_df = pl.DataFrame({"prediction": predictions})

    # Save to CSV
    pred_df.write_csv(output_file, line_terminator=",\n")

    # Log saving process
    if log_results:
        logger.info(_LOG_OUTPUT_SAVED.format(output_file=output_file))
