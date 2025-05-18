"""Script to train a model with hyperparameter tuning and save results."""

import json
import os
import pickle
from pathlib import Path

import polars as pl
import typer
from loguru import logger

from ml_project.model_training import compute_metrics, tune_hyperparameters
from ml_project.utils import read_config

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


def train(  # noqa: PLR0913
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

        # Compute metrics on test set
        logger.info("Evaluating model on test set...")
        metrics = compute_metrics(
            model=best_model,
            x_test=x_test,
            y_test=y_test,
            log_results=True,
        )

        # Save model, grid search results, and metrics
        _save_results(best_model, grid_search, metrics, model_name, output_dir, results_dir, overwrite)

        typer.secho(
            f"✅ Model successfully trained and saved to {output_dir}",
            fg=typer.colors.GREEN,
        )
        typer.secho(
            f"✅ Grid search results and metrics saved to {results_dir}",
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


def _save_results(
    best_model: object,
    grid_search: object,
    metrics: dict,
    model_name: str,
    output_dir: Path,
    results_dir: Path,
    overwrite: bool,
) -> None:
    """Save model as pickle and grid search results and metrics as CSV files."""
    # Define output files
    model_file = output_dir / f"{model_name}_best_model.pkl"
    grid_search_file = results_dir / f"{model_name}_grid_search_results.csv"
    metrics_file = results_dir / f"{model_name}_metrics.csv"

    # Check if output files exist and if we should overwrite
    if not overwrite:
        for file_path, file_desc in [
            (model_file, "Model"),
            (grid_search_file, "Grid search results"),
            (metrics_file, "Metrics"),
        ]:
            if file_path.exists():
                logger.error(f"{file_desc} file {file_path} already exists. Use --overwrite to overwrite.")
                raise typer.Exit(code=1)

    # Save model
    logger.info(f"Saving best model to {model_file}...")
    with open(model_file, "wb") as f:
        pickle.dump(best_model, f)

    # Save grid search results as CSV
    logger.info(f"Saving grid search results to {grid_search_file}...")

    # Extract and save CV results to CSV
    cv_results = grid_search.cv_results_

    # Convert CV results to Polars DataFrame and save relevant info
    results_data = {}

    # Add general scores
    for key in ["mean_test_score", "std_test_score", "rank_test_score"]:
        if key in cv_results:
            results_data[key] = cv_results[key]

    # Add parameter values
    params_list = cv_results.get("params", [])
    for param_name in grid_search.best_params_.keys() if hasattr(grid_search, "best_params_") else []:
        results_data[f"param_{param_name}"] = [str(params.get(param_name, "")) for params in params_list]

    # Create DataFrame and save
    cv_results_df = pl.DataFrame(results_data)

    # Add best parameters as JSON string in a new column
    best_params_str = json.dumps(grid_search.best_params_)
    cv_results_df = cv_results_df.with_columns([pl.lit(best_params_str).alias("best_params")])

    # Add best score
    if hasattr(grid_search, "best_score_"):
        cv_results_df = cv_results_df.with_columns([pl.lit(grid_search.best_score_).alias("best_score")])

    # Save to CSV
    cv_results_df.write_csv(grid_search_file)

    # Save metrics as CSV
    logger.info(f"Saving metrics to {metrics_file}...")
    metrics_df = pl.DataFrame([metrics])
    metrics_df.write_csv(metrics_file)


def main() -> None:
    """Entrypoint for the train-model script."""
    typer.run(train)


if __name__ == "__main__":
    main()
