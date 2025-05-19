"""Script to preprocess raw data and save it."""

import os
from pathlib import Path

import polars as pl
import typer
from loguru import logger

from ml_project.data_storage import save_data_to_csv
from ml_project.preprocessing import preprocess_data

# Define options as module-level variables
INPUT_FILE_OPTION = typer.Option(
    Path("data/raw_data.csv"),
    "--input",
    "-i",
    help="Path to the raw data CSV file",
)
OUTPUT_FILE_OPTION = typer.Option(
    Path("data/processed_data.csv"),
    "--output",
    "-o",
    help="Path where to save the processed CSV file",
)
OVERWRITE_OPTION = typer.Option(
    False,
    "--overwrite",
    "-w",
    help="Overwrite existing output file if it exists",
)


def preprocess(
    input_file: Path = INPUT_FILE_OPTION,
    output_file: Path = OUTPUT_FILE_OPTION,
    overwrite: bool = OVERWRITE_OPTION,
) -> None:
    """Preprocess raw data and save it to a CSV file.

    This command reads the raw data, applies preprocessing steps,
    and saves the processed data to the specified output path.
    By default, it reads from 'data/raw_data.csv' and saves to 'data/processed_data.csv'.
    """
    # Log parameters
    logger.debug(f"Input file: {input_file}")
    logger.debug(f"Output file: {output_file}")
    logger.debug(f"Overwrite: {overwrite}")

    try:
        # Make paths absolute if they're relative
        if not input_file.is_absolute():
            # Assumes the script is being run from the project root
            project_root = Path(os.getcwd())
            input_file = project_root / input_file

        if not output_file.is_absolute():
            project_root = Path(os.getcwd())
            output_file = project_root / output_file

        # Check if input file exists
        if not input_file.exists():
            logger.error(f"Input file {input_file} doesn't exist")
            raise typer.Exit(code=1)

        # Read the data
        logger.info(f"Reading data from {input_file}...")
        logger.debug("Using LazyFrame for reading")
        data = pl.scan_csv(input_file)

        # Preprocess the data
        logger.info("Preprocessing data...")
        processed_data = preprocess_data(data, lazy_output=True)

        # Save the processed data
        logger.info(f"Saving processed data to {output_file}...")
        save_data_to_csv(processed_data, output_file, overwrite=overwrite)

        typer.secho(f"✅ Processed data successfully saved to {output_file}", fg=typer.colors.GREEN)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise typer.Exit(code=1) from e


def main() -> None:
    """Entrypoint for the preprocess script."""
    typer.run(preprocess)


if __name__ == "__main__":
    main()
