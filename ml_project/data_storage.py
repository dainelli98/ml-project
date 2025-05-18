"""Data storage functionality."""

from pathlib import Path

import polars as pl
from loguru import logger


def save_data_to_csv(data: pl.DataFrame | pl.LazyFrame, output_path: Path | str, overwrite: bool = False) -> Path:
    """Save data to a CSV file.

    Args:
        data: Polars DataFrame or LazyFrame to save
        output_path: Path where to save the CSV file
        overwrite: If True, overwrite existing file. Defaults to False.

    Returns:
        Path to the saved file

    Raises:
        FileExistsError: If the output file already exists and overwrite is False
    """
    output_path = Path(output_path)

    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        msg = f"Output file {output_path} already exists. Use overwrite=True to overwrite."
        logger.error(msg)
        raise FileExistsError(msg)

    # Convert LazyFrame to DataFrame if needed
    if isinstance(data, pl.LazyFrame):
        logger.debug("Converting LazyFrame to DataFrame for saving")
        data = data.collect()

    # Save to CSV
    data.write_csv(output_path)
    logger.info(f"Data saved to {output_path}")

    return output_path
