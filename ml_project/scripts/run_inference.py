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
import polars as pl
import typer
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
MODEL_NAME_OPTION = typer.Option(
    "Unknown",
    "--model-name",
    "-n",
    help="Name of the model for logging purposes",
)


def inference(
    model_name: str = MODEL_NAME_OPTION,
    data_file: Path = DATA_FILE_OPTION,
    output_file: Path = OUTPUT_FILE_OPTION,
) -> None:
    """Run inference with a trained model on new data.

    This command loads a trained model from a pickle file, reads the data
    from a CSV file, runs inference, and saves the predictions to a CSV file.
    """
    # Log parameters
    logger.debug(f"Model name: {model_name}")
    logger.debug(f"Data file: {data_file}")
    logger.debug(f"Output file: {output_file}")

    try:
        # Convert relative paths to absolute paths
        project_root = Path(os.getcwd())

        paths = {
            "data_file": data_file,
            "output_file": output_file,
        }

        # Make all paths absolute
        for key, path in paths.items():
            if not path.is_absolute():
                paths[key] = project_root / path

        # Unpack for clarity
        data_file = paths["data_file"]
        output_file = paths["output_file"]

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

            typer.secho(
                "✅ Inference completed successfully! Input data and predictions saved and logged to MLflow",
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


def main() -> None:
    """Entrypoint for the run-inference script."""
    typer.run(inference)


if __name__ == "__main__":
    main()
