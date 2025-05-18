"""Script to prepare data by selecting features, target and splitting into train/test sets."""

import os
from pathlib import Path

import polars as pl
import typer
from loguru import logger

from ml_project.data_preparation import get_features_and_target, split_data
from ml_project.data_storage import save_data_to_csv
from ml_project.utils import read_config

# Define options as module-level variables
INPUT_FILE_OPTION = typer.Option(
    Path("data/processed_data.csv"),
    "--input",
    "-i",
    help="Path to the processed data CSV file",
)
OUTPUT_DIR_OPTION = typer.Option(
    Path("data/"),
    "--output-dir",
    "-o",
    help="Directory where to save the train and test CSV files",
)
TEST_SIZE_OPTION = typer.Option(
    0.2,
    "--test-size",
    "-t",
    help="Proportion of data to use for testing (between 0 and 1)",
)
STRATIFY_OPTION = typer.Option(
    False,
    "--stratify",
    "-s",
    help="Whether to stratify the split based on target values (only for classification)",
)
RANDOM_STATE_OPTION = typer.Option(
    42,
    "--random-state",
    "-r",
    help="Random seed for reproducibility",
)
OVERWRITE_OPTION = typer.Option(
    False,
    "--overwrite",
    "-w",
    help="Overwrite existing output files if they exist",
)


def prepare(
    input_file: Path = INPUT_FILE_OPTION,
    output_dir: Path = OUTPUT_DIR_OPTION,
    test_size: float = TEST_SIZE_OPTION,
    stratify: bool = STRATIFY_OPTION,
    random_state: int = RANDOM_STATE_OPTION,
    overwrite: bool = OVERWRITE_OPTION,
) -> None:
    """Prepare data by selecting features, target and splitting into train/test sets.

    This command reads the processed data, selects features and target based on config,
    splits the data into training and test sets, and saves them to separate CSV files.
    By default, it reads from 'data/processed_data.csv' and saves to 'data/x_train.csv',
    'data/y_train.csv', 'data/x_test.csv', and 'data/y_test.csv'.
    """
    # Log parameters
    logger.debug(f"Input file: {input_file}")
    logger.debug(f"Output directory: {output_dir}")
    logger.debug(f"Test size: {test_size}")
    logger.debug(f"Stratify: {stratify}")
    logger.debug(f"Random state: {random_state}")
    logger.debug(f"Overwrite: {overwrite}")

    try:
        _setup_paths_and_validate(input_file, output_dir, overwrite)

        # Load configuration and read data
        config = read_config()
        logger.info(f"Reading data from {input_file}...")
        data = pl.scan_csv(input_file)

        # Extract features, target and split data
        x, y = get_features_and_target(data, config, lazy_output=True)
        x_train, x_test, y_train, y_test = split_data(
            x, y, test_size=test_size, val_size=None, stratify=stratify, random_state=random_state
        )

        # Create and save datasets
        _create_and_save_datasets(x_train, x_test, y_train, y_test, config, output_dir, overwrite)

        typer.secho(
            f"✅ Data successfully prepared and saved to {output_dir}/x_train.csv, {output_dir}/y_train.csv, "
            f"{output_dir}/x_test.csv, and {output_dir}/y_test.csv",
            fg=typer.colors.GREEN,
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        raise typer.Exit(code=1) from e


def _setup_paths_and_validate(input_file: Path, output_dir: Path, overwrite: bool) -> None:
    """Set up paths and validate input/output files."""
    # Make paths absolute if they're relative
    if not input_file.is_absolute():
        project_root = Path(os.getcwd())
        input_file = project_root / input_file

    if not output_dir.is_absolute():
        project_root = Path(os.getcwd())
        output_dir = project_root / output_dir

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if input file exists
    if not input_file.exists():
        logger.error(f"Input file {input_file} doesn't exist")
        raise typer.Exit(code=1)

    # Check if output files exist and if we should overwrite
    if not overwrite:
        x_train_file = output_dir / "x_train.csv"
        y_train_file = output_dir / "y_train.csv"
        x_test_file = output_dir / "x_test.csv"
        y_test_file = output_dir / "y_test.csv"

        for file_path in [x_train_file, y_train_file, x_test_file, y_test_file]:
            if file_path.exists():
                logger.error(f"Output file {file_path} already exists. Use --overwrite to overwrite.")
                raise typer.Exit(code=1)


def _create_and_save_datasets(
    x_train: pl.LazyFrame | pl.DataFrame,
    x_test: pl.LazyFrame | pl.DataFrame,
    y_train: pl.Series,
    y_test: pl.Series,
    config: dict,
    output_dir: Path,
    overwrite: bool,
) -> None:
    """Create and save datasets as separate X and y files."""
    target_col = config["feature_selection"]["target"]

    # Define output files
    x_train_file = output_dir / "x_train.csv"
    y_train_file = output_dir / "y_train.csv"
    x_test_file = output_dir / "x_test.csv"
    y_test_file = output_dir / "y_test.csv"

    # Check if output files exist and if we should overwrite
    if not overwrite:
        for file_path in [x_train_file, y_train_file, x_test_file, y_test_file]:
            if file_path.exists():
                logger.error(f"Output file {file_path} already exists. Use --overwrite to overwrite.")
                raise typer.Exit(code=1)

    # Convert to DataFrame if they're not already
    if isinstance(x_train, pl.LazyFrame):
        x_train = x_train.collect()
    if isinstance(x_test, pl.LazyFrame):
        x_test = x_test.collect()

    # Create y DataFrames
    y_train_df = pl.DataFrame({target_col: y_train})
    y_test_df = pl.DataFrame({target_col: y_test})

    # Save X and y train data
    logger.info(f"Training features set created with {len(x_train)} rows")
    logger.info(f"Saving training features to {x_train_file}...")
    save_data_to_csv(x_train.lazy(), x_train_file, overwrite=overwrite)

    logger.info(f"Training target set created with {len(y_train_df)} rows")
    logger.info(f"Saving training target to {y_train_file}...")
    save_data_to_csv(y_train_df.lazy(), y_train_file, overwrite=overwrite)

    # Save X and y test data
    logger.info(f"Test features set created with {len(x_test)} rows")
    logger.info(f"Saving test features to {x_test_file}...")
    save_data_to_csv(x_test.lazy(), x_test_file, overwrite=overwrite)

    logger.info(f"Test target set created with {len(y_test_df)} rows")
    logger.info(f"Saving test target to {y_test_file}...")
    save_data_to_csv(y_test_df.lazy(), y_test_file, overwrite=overwrite)


def main() -> None:
    """Entrypoint for the prepare-data script."""
    typer.run(prepare)


if __name__ == "__main__":
    main()
