"""Basic data extraction functionality."""

import os

import polars as pl
from loguru import logger

CARSEATS_RAW_DATA_URL = os.getenv(
    "CARSEATS_RAW_DATA_URL",
    "https://raw.githubusercontent.com/intro-stat-learning/ISLP/refs/heads/main/ISLP/data/Carseats.csv",
)


def fetch_carseats_data(lazy: bool = True) -> pl.DataFrame | pl.LazyFrame:
    """Fetch raw carseats data from the web.

    Args:
        lazy: If True, returns a lazy frame. Defaults to True.

    Returns:
        Car seats data.
    """
    logger.debug(f"Fetching carseats data from {CARSEATS_RAW_DATA_URL}")

    fetch_function = pl.scan_csv if lazy else pl.read_csv

    data = fetch_function(CARSEATS_RAW_DATA_URL)
    logger.info("Carseats data fetched successfully")
    logger.debug(f"Data shape: {data.shape}")
    logger.debug(f"Data columns: {data.columns}")

    if lazy:
        logger.info("Returning carseats data as a lazy frame")

    return data
