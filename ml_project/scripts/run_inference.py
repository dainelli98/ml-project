"""Script to run inference with a trained model on new data.

This script uses the inference module to load a trained model,
run inference on new data, and save the predictions to a CSV file.
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import polars as pl
import typer
from evidently import Report
from evidently.presets import DataDriftPreset
from loguru import logger

from ml_project.inference import run_inference, save_predictions
from ml_project.utils import MLFLOW_TRACKING_URI

# Define options as module-level variables
MODEL_NAME_OPTION = typer.Option(
    ...,  # Required
    "--model-name",
    "-n",
    help="Name of the model in MLflow Model Registry",
)
DATA_FILE_OPTION = typer.Option(
    ...,  # Required
    "--data-file",
    "-d",
    help="Path to the data CSV file for inference",
)
OUTPUT_FILE_OPTION = typer.Option(
    "predictions.csv",
    "--output-file",
    "-o",
    help="Path to save the predictions (CSV file)",
)
TRAINING_DATA_FILE_OPTION = typer.Option(
    None,
    "--training-data-file",
    "-t",
    help="Optional path to the training data CSV file for data drift detection",
)


def inference(
    model_name: str = MODEL_NAME_OPTION,
    data_file: Path = DATA_FILE_OPTION,
    output_file: Path = OUTPUT_FILE_OPTION,
    training_data_file: Path | None = TRAINING_DATA_FILE_OPTION,
) -> None:
    """Run inference with a trained model on new data.

    This command loads a trained model from a pickle file, reads the data
    from a CSV file, runs inference, and saves the predictions to a CSV file.
    If a training data file is provided, it also performs data drift analysis.
    """
    # Log parameters
    logger.debug(f"Model name: {model_name}")
    logger.debug(f"Data file: {data_file}")
    logger.debug(f"Output file: {output_file}")
    logger.debug(f"Training data file: {training_data_file}")

    try:
        # Convert relative paths to absolute paths
        project_root = Path(os.getcwd())

        paths = {
            "data_file": data_file,
            "output_file": output_file,
        }

        if training_data_file is not None:
            paths["training_data_file"] = training_data_file

        # Make all paths absolute
        for key, path in paths.items():
            if not path.is_absolute():
                paths[key] = project_root / path

        # Unpack for clarity
        data_file = paths["data_file"]
        output_file = paths["output_file"]
        if training_data_file is not None:
            training_data_file = paths["training_data_file"]

        # Ensure input files exist
        if not data_file.exists():
            logger.error(f"Data file {data_file} doesn't exist")
            raise typer.Exit(code=1)

        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Load model from MLflow Model Registry
        logger.info(f"Loading model '{model_name}' from MLflow Model Registry...")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{model_name}/latest"
        model = mlflow.sklearn.load_model(model_uri)

        # Set MLflow experiment name: prediction_{model_name}_{timestamp}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"prediction_{model_name}_{timestamp}"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"prediction_{model_name}"):
            # Load data
            logger.info(f"Loading data from {data_file}...")
            data = pl.read_csv(data_file)

            # Log input data as MLflow artifact using a temporary file
            _log_input_data(data)

            # If training data file is provided, detect data drift
            if training_data_file is not None and training_data_file.exists():
                logger.info(f"Loading training data from {training_data_file} for data drift detection...")
                reference_data = pl.read_csv(training_data_file)

                # Verify that columns match between reference and current data
                if set(reference_data.columns) != set(data.columns):
                    logger.warning(
                        "Training data columns do not match inference data columns. "
                        "Data drift detection will be limited to common columns."
                    )
                    # Get common columns
                    common_cols = list(set(reference_data.columns) & set(data.columns))
                    reference_data = reference_data.select(common_cols)
                    data_for_drift = data.select(common_cols)
                else:
                    data_for_drift = data

                # Perform data drift detection
                _detect_data_drift(reference_data, data_for_drift)

                logger.info("Data drift analysis completed and logged to MLflow")

            # Run inference
            logger.info(f"Running inference with model on {len(data)} samples...")
            predictions = run_inference(
                model=model,
                data=data,
                model_name=model_name,
                log_results=True,
            )

            # Save predictions
            logger.info(f"Saving predictions to {output_file}...")
            save_predictions(
                predictions=predictions,
                output_file=str(output_file),
                log_results=True,
            )

            # Log predictions as MLflow artifact
            logger.info(f"Logging predictions to MLflow as artifact: {output_file}")
            mlflow.log_artifact(str(output_file))

            success_message = (
                "✅ Inference completed successfully! Input data and predictions saved and logged to MLflow"
            )
            if training_data_file is not None and training_data_file.exists():
                success_message += "\n✅ Data drift analysis completed and logged to MLflow"

            typer.secho(
                success_message,
                fg=typer.colors.GREEN,
            )
    except Exception as e:
        logger.error(f"Error: {e}")
        raise typer.Exit(code=1) from e


def _log_input_data(data: pl.DataFrame) -> None:
    """Log input data as an MLflow artifact.

    Args:
        data: Polars DataFrame containing the input data.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp_input:
        data.write_csv(tmp_input.name)
        logger.info(f"Logging input data to MLflow as artifact: {tmp_input.name}")
        mlflow.log_artifact(tmp_input.name, artifact_path="input_data")
    try:
        os.remove(tmp_input.name)
    except Exception as cleanup_err:
        logger.warning(f"Could not remove temporary input data file: {cleanup_err}")


def _detect_data_drift(
    reference_data: pl.DataFrame,
    current_data: pl.DataFrame,
) -> None:
    """Detect data drift between training (reference) and inference (current) data.

    Args:
        reference_data: Polars DataFrame containing the training data
        current_data: Polars DataFrame containing the inference data

    This function uses Evidently to calculate data drift metrics and generate
    visualizations that are logged to MLflow.
    """
    logger.info("Performing data drift analysis...")

    # Convert Polars DataFrames to pandas for Evidently compatibility
    reference_data_pd = reference_data.to_pandas()
    current_data_pd = current_data.to_pandas()

    # Create and run Evidently report
    _create_and_log_evidently_report(reference_data_pd, current_data_pd)


def _create_and_log_evidently_report(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> None:
    """Create and log Evidently report with data drift metrics.

    Args:
        reference_data: Pandas DataFrame containing the reference (training) data
        current_data: Pandas DataFrame containing the current (inference) data
    """
    # Create Evidently report with data drift preset
    data_drift_report = Report(metrics=[DataDriftPreset()])
    snapshot = data_drift_report.run(reference_data=reference_data, current_data=current_data)

    # Save report to HTML and log as artifact
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as tmp_report:
        snapshot.save_html(tmp_report.name)
        logger.info(f"Logging data drift report to MLflow as artifact: {tmp_report.name}")
        mlflow.log_artifact(tmp_report.name, artifact_path="data_drift")

    # Clean up the temporary file
    try:
        os.remove(tmp_report.name)
    except Exception as cleanup_err:
        logger.warning(f"Could not remove temporary report file: {cleanup_err}")


def main() -> None:
    """Entrypoint for the run-inference script."""
    typer.run(inference)


if __name__ == "__main__":
    main()
