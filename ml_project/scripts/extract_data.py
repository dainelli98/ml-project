"""Script to extract data from sources and save it locally."""

import os
from pathlib import Path

import typer
from loguru import logger

from ml_project.data_extraction import fetch_carseats_data
from ml_project.data_storage import save_data_to_csv

# Define options as module-level variables
OUTPUT_FILE_OPTION = typer.Option(
    Path("data/raw_data.csv"),
    "--output",
    "-o",
    help="Path where to save the CSV file",
)
LAZY_OPTION = typer.Option(
    False,
    "--lazy",
    "-l",
    help="Use lazy loading (more memory efficient but slower)",
)
OVERWRITE_OPTION = typer.Option(
    False,
    "--overwrite",
    "-w",
    help="Overwrite existing file if it exists",
)


def extract_data(
    output_file: Path = OUTPUT_FILE_OPTION,
    lazy: bool = LAZY_OPTION,
    overwrite: bool = OVERWRITE_OPTION,
) -> None:
    """Extract raw carseats data and save it to a CSV file.

    This command fetches car seats data from the web and saves it to the specified path.
    By default, it saves to 'data/raw_data.csv' in the project root directory.
    """
    # Log parameters
    logger.debug(f"Output file: {output_file}")
    logger.debug(f"Lazy loading: {lazy}")
    logger.debug(f"Overwrite: {overwrite}")

    try:
        # Fetch data
        logger.info("Fetching carseats data...")
        data = fetch_carseats_data(lazy=lazy)

        # Make path absolute if it's relative
        if not output_file.is_absolute():
            # Assumes the script is being run from the project root
            project_root = Path(os.getcwd())
            output_file = project_root / output_file

        # Save data
        logger.info(f"Saving data to {output_file}...")
        save_data_to_csv(data, output_file, overwrite=overwrite)

        typer.secho(f"✅ Data successfully saved to {output_file}", fg=typer.colors.GREEN)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise typer.Exit(code=1) from e


def main() -> None:
    """Entrypoint for the preprocess script."""
    typer.run(extract_data)


if __name__ == "__main__":
    main()
