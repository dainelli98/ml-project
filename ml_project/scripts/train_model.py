"""Script to train a model with hyperparameter tuning and save results."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import mlflow.models
import mlflow.sklearn
import polars as pl
import typer
from loguru import logger

from ml_project.model_training import compute_metrics, tune_hyperparameters
from ml_project.utils import MLFLOW_TRACKING_URI, read_config

# Define options as module-level variables
X_TRAIN_FILE_OPTION = typer.Option(
    Path("data/x_train.csv"),
    "--x-train",
    "-x",
    help="Path to the training features CSV file",
)
Y_TRAIN_FILE_OPTION = typer.Option(
    Path("data/y_train.csv"),
    "--y-train",
    "-y",
    help="Path to the training target CSV file",
)
X_TEST_FILE_OPTION = typer.Option(
    Path("data/x_test.csv"),
    "--x-test",
    "-xt",
    help="Path to the test features CSV file",
)
Y_TEST_FILE_OPTION = typer.Option(
    Path("data/y_test.csv"),
    "--y-test",
    "-yt",
    help="Path to the test target CSV file",
)
OUTPUT_DIR_OPTION = typer.Option(
    Path("models/"),
    "--output-dir",
    "-o",
    help="Directory where to save the trained model",
)
RESULTS_DIR_OPTION = typer.Option(
    Path("data/"),
    "--results-dir",
    "-r",
    help="Directory where to save the grid search results and metrics as CSV files",
)
MODEL_NAME_OPTION = typer.Option(
    "GradientBoostingRegressor",
    "--model-name",
    "-m",
    help="Name of the model to train (as specified in config)",
)
CV_OPTION = typer.Option(
    5,
    "--cv",
    "-c",
    help="Number of cross-validation folds",
)
SCORING_OPTION = typer.Option(
    "neg_mean_squared_error",
    "--scoring",
    "-s",
    help="Scoring metric for cross-validation",
)
N_JOBS_OPTION = typer.Option(
    -1,
    "--n-jobs",
    "-n",
    help="Number of jobs to run in parallel (-1 to use all processors)",
)
OVERWRITE_OPTION = typer.Option(
    False,
    "--overwrite",
    "-w",
    help="Overwrite existing output files if they exist",
)


def train(
    x_train_file: Path = X_TRAIN_FILE_OPTION,
    y_train_file: Path = Y_TRAIN_FILE_OPTION,
    x_test_file: Path = X_TEST_FILE_OPTION,
    y_test_file: Path = Y_TEST_FILE_OPTION,
    output_dir: Path = OUTPUT_DIR_OPTION,
    results_dir: Path = RESULTS_DIR_OPTION,
    model_name: str = MODEL_NAME_OPTION,
    cv: int = CV_OPTION,
    scoring: str = SCORING_OPTION,
    n_jobs: int = N_JOBS_OPTION,
    overwrite: bool = OVERWRITE_OPTION,
) -> None:
    """Train a model with hyperparameter tuning and save results.

    This command reads the training and test data, performs hyperparameter tuning
    using GridSearchCV, evaluates the best model on the test set, and saves the
    model, grid search results, and evaluation metrics.
    """
    # Log parameters
    logger.debug(f"X train file: {x_train_file}")
    logger.debug(f"Y train file: {y_train_file}")
    logger.debug(f"X test file: {x_test_file}")
    logger.debug(f"Y test file: {y_test_file}")
    logger.debug(f"Output directory: {output_dir}")
    logger.debug(f"Results directory: {results_dir}")
    logger.debug(f"Model name: {model_name}")
    logger.debug(f"CV folds: {cv}")
    logger.debug(f"Scoring: {scoring}")
    logger.debug(f"N jobs: {n_jobs}")
    logger.debug(f"Overwrite: {overwrite}")

    try:
        # Setup paths and validate files
        _setup_paths_and_validate(
            x_train_file, y_train_file, x_test_file, y_test_file, output_dir, results_dir, model_name, overwrite
        )

        # Load configuration and data
        config = read_config()

        logger.info(f"Reading training data from {x_train_file} and {y_train_file}...")
        x_train = pl.read_csv(x_train_file)
        y_train = pl.read_csv(y_train_file)

        # Extract target column name from config
        target_col = config["feature_selection"]["target"]
        y_train = y_train[target_col]

        logger.info(f"Reading test data from {x_test_file} and {y_test_file}...")
        x_test = pl.read_csv(x_test_file)
        y_test = pl.read_csv(y_test_file)
        y_test = y_test[target_col]

        # Set MLflow tracking URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{model_name}_{timestamp}"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"{model_name}_run"):
            # Log parameters
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("cv", cv)
            mlflow.log_param("scoring", scoring)
            mlflow.log_param("n_jobs", n_jobs)

            # Tune hyperparameters
            logger.info(f"Tuning hyperparameters for model {model_name}...")
            best_model, grid_search = tune_hyperparameters(
                x_train=x_train,
                y_train=y_train,
                model_name=model_name,
                config=config,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
            )

            # Log best hyperparameters
            mlflow.log_params(grid_search.best_params_)

            # Compute metrics on test set
            logger.info("Evaluating model on test set...")
            metrics = compute_metrics(
                model=best_model,
                x_test=x_test,
                y_test=y_test,
                log_results=True,
            )

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log grid search results as artifact
            grid_search_file = results_dir / f"{model_name}_grid_search_results.csv"
            # Save grid search results to CSV (reuse _save_results logic)
            _save_grid_search_results(grid_search, grid_search_file)
            mlflow.log_artifact(str(grid_search_file))

            # Register the model in MLflow Model Registry with input_example and signature
            logger.info("Logging and registering model to MLflow with input example and signature...")

            input_example = x_train.head(5).to_pandas()
            signature = mlflow.models.infer_signature(x_train.to_pandas(), best_model.predict(x_train.to_numpy()))
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="model",
                registered_model_name=model_name,
                input_example=input_example,
                signature=signature,
            )

            typer.secho(
                f"✅ Model successfully trained and registered to MLflow as '{model_name}'",
                fg=typer.colors.GREEN,
            )
            typer.secho(
                "✅ Grid search results logged as MLflow artifact",
                fg=typer.colors.GREEN,
            )
    except Exception as e:
        logger.error(f"Error: {e}")
        raise typer.Exit(code=1) from e


def _setup_paths_and_validate(
    x_train_file: Path,
    y_train_file: Path,
    x_test_file: Path,
    y_test_file: Path,
    output_dir: Path,
    results_dir: Path,
    model_name: str,
    overwrite: bool,
) -> None:
    """Set up paths and validate input/output files."""
    # Convert relative paths to absolute paths
    project_root = Path(os.getcwd())

    paths = {
        "x_train": x_train_file,
        "y_train": y_train_file,
        "x_test": x_test_file,
        "y_test": y_test_file,
        "output_dir": output_dir,
        "results_dir": results_dir,
    }

    # Make all paths absolute
    for key, path in paths.items():
        if not path.is_absolute():
            paths[key] = project_root / path

    # Unpack for clarity
    x_train_file = paths["x_train"]
    y_train_file = paths["y_train"]
    x_test_file = paths["x_test"]
    y_test_file = paths["y_test"]
    output_dir = paths["output_dir"]
    results_dir = paths["results_dir"]

    # Ensure output directories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Check if input files exist
    input_files = [
        (x_train_file, "X train"),
        (y_train_file, "Y train"),
        (x_test_file, "X test"),
        (y_test_file, "Y test"),
    ]

    for file_path, file_desc in input_files:
        if not file_path.exists():
            logger.error(f"{file_desc} file {file_path} doesn't exist")
            raise typer.Exit(code=1)

    # Check if output files would be overwritten
    if not overwrite:
        output_files = [
            (output_dir / f"{model_name}_best_model.pkl", "Model"),
            (results_dir / f"{model_name}_grid_search_results.csv", "Grid search results"),
            (results_dir / f"{model_name}_metrics.csv", "Metrics"),
        ]

        for file_path, file_desc in output_files:
            if file_path.exists():
                logger.error(f"{file_desc} file {file_path} already exists. Use --overwrite to overwrite.")
                raise typer.Exit(code=1)


def _save_grid_search_results(grid_search: Any, grid_search_file: Any) -> None:
    cv_results = grid_search.cv_results_
    results_data = {}
    for key in ["mean_test_score", "std_test_score", "rank_test_score"]:
        if key in cv_results:
            results_data[key] = cv_results[key]
    params_list = cv_results.get("params", [])
    for param_name in grid_search.best_params_.keys() if hasattr(grid_search, "best_params_") else []:
        results_data[f"param_{param_name}"] = [str(params.get(param_name, "")) for params in params_list]
    cv_results_df = pl.DataFrame(results_data)
    best_params_str = json.dumps(grid_search.best_params_)
    cv_results_df = cv_results_df.with_columns([pl.lit(best_params_str).alias("best_params")])
    if hasattr(grid_search, "best_score_"):
        cv_results_df = cv_results_df.with_columns([pl.lit(grid_search.best_score_).alias("best_score")])
    cv_results_df.write_csv(grid_search_file)


def main() -> None:
    """Entrypoint for the train-model script."""
    typer.run(train)


if __name__ == "__main__":
    main()
