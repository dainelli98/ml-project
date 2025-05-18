"""Data preprocessing module."""

import polars as pl
from loguru import logger

# Constants for logging messages
_LOG_CONVERT_TO_LAZY = "Converting DataFrame to LazyFrame"
_LOG_COLLECT_TO_DF = "Collecting LazyFrame to DataFrame"

# Column mappings
_BINARY_COLUMNS = ["urban", "us"]  # Updated to use the correct renamed column name
_ORDINAL_CAT_COLUMNS = ["shelve_loc"]


def preprocess_data(df: pl.DataFrame | pl.LazyFrame, lazy_output: bool = False) -> pl.DataFrame | pl.LazyFrame:
    """Preprocess Carseats data.

    Args:
        df: DataFrame or LazyFrame to preprocess.
        lazy_output: If True, returns a LazyFrame. Defaults to ``False``.

    Returns:
        Preprocessed DataFrame or LazyFrame.
    """
    logger.info("Preprocessing data")
    if isinstance(df, pl.DataFrame):
        logger.debug(_LOG_CONVERT_TO_LAZY)
        df = df.lazy()

    data = df.pipe(rename_columns).pipe(set_types).pipe(encode_binary_columns).pipe(encode_ordinal_cat_columns)

    logger.info("Data preprocessing completed")
    if not lazy_output:
        logger.debug(_LOG_COLLECT_TO_DF)
        data = data.collect()

    return data


def rename_columns(df: pl.DataFrame | pl.LazyFrame, lazy_output: bool = True) -> pl.DataFrame | pl.LazyFrame:
    """Rename columns in the DataFrame or LazyFrame.

    Columns names are all set to snake_case.

    Args:
        df: DataFrame or LazyFrame to rename columns for.
        lazy_output: If True, returns a LazyFrame. Defaults to ``True``.

    Returns:
        DataFrame or LazyFrame with renamed columns.
    """
    logger.debug("Renaming columns to snake_case")

    if isinstance(df, pl.DataFrame):
        column_names = df.columns
        logger.debug(_LOG_CONVERT_TO_LAZY)
        df = df.lazy()
    else:
        # Use collect_schema().names() to avoid performance warning
        column_names = df.collect_schema().names()

    # Create a mapping of old column names to snake_case
    column_mapping = {}
    for col in column_names:
        # Convert column name to snake_case
        snake_case = "".join(["_" + c.lower() if c.isupper() else c.lower() for c in col]).lstrip("_")
        if len(snake_case) <= 3:
            snake_case = snake_case.replace("_", "")
        column_mapping[col] = snake_case

    # Apply the column renaming
    data = df.rename(column_mapping)

    logger.info("Columns renamed successfully")

    if not lazy_output:
        logger.debug(_LOG_COLLECT_TO_DF)
        data = data.collect()

    return data


def set_types(df: pl.DataFrame | pl.LazyFrame, lazy_output: bool = True) -> pl.DataFrame | pl.LazyFrame:
    """Set data types for the DataFrame or LazyFrame.

    Args:
        df: DataFrame or LazyFrame to set types for.
        lazy_output: If True, returns a LazyFrame. Defaults to ``True``.

    Returns:
        DataFrame or LazyFrame with set types.
    """
    logger.debug("Setting data types")

    if isinstance(df, pl.DataFrame):
        logger.debug(_LOG_CONVERT_TO_LAZY)
        df = df.lazy()

    data = df.with_columns(
        [
            pl.col("shelve_loc").cast(pl.Categorical),
            # Note: Not casting binary columns to Boolean here
            # They will be processed in encode_binary_columns
        ]
    )

    logger.info("Data types set successfully")

    if not lazy_output:
        logger.debug(_LOG_COLLECT_TO_DF)
        data = data.collect()

    return data


def encode_binary_columns(df: pl.DataFrame | pl.LazyFrame, lazy_output: bool = True) -> pl.DataFrame | pl.LazyFrame:
    """Encode binary columns in the DataFrame or LazyFrame.

    The binary columns defined in ``_BINARY_COLUMNS`` are converted to 0/1 integers.

    Args:
        df: DataFrame or LazyFrame to encode binary columns for.
        lazy_output: If True, returns a LazyFrame. Defaults to ``True``.

    Returns:
        DataFrame or LazyFrame with encoded binary columns.
    """
    logger.debug("Encoding binary columns")

    if isinstance(df, pl.DataFrame):
        logger.debug(_LOG_CONVERT_TO_LAZY)
        df = df.lazy()

    # Convert "Yes" values to 1, all others to 0
    data = df.with_columns([(pl.col(col) == "Yes").cast(pl.Int8) for col in _BINARY_COLUMNS])

    logger.info("Binary columns encoded successfully")

    if not lazy_output:
        logger.debug(_LOG_COLLECT_TO_DF)
        data = data.collect()

    return data


def encode_ordinal_cat_columns(
    df: pl.DataFrame | pl.LazyFrame, lazy_output: bool = True
) -> pl.DataFrame | pl.LazyFrame:
    """Encode ordinal categorical columns in the DataFrame or LazyFrame.

    The ordinal categorical columns defined in ``_ORDINAL_CAT_COLUMNS`` are encoded
    to numerical values based on their ordinal nature.

    Args:
        df: DataFrame or LazyFrame to encode ordinal categorical columns for.
        lazy_output: If True, returns a LazyFrame. Defaults to ``True``.

    Returns:
        DataFrame or LazyFrame with encoded ordinal categorical columns.
    """
    logger.debug("Encoding ordinal categorical columns")

    if isinstance(df, pl.DataFrame):
        logger.debug(_LOG_CONVERT_TO_LAZY)
        df = df.lazy()

    # Encode ShelveLoc column using a when-then-otherwise structure instead of map_dict
    # as map_dict is not available for LazyFrame expressions
    data = df.with_columns(
        pl.when(pl.col("shelve_loc") == "Bad")
        .then(0)
        .when(pl.col("shelve_loc") == "Medium")
        .then(1)
        .when(pl.col("shelve_loc") == "Good")
        .then(2)
        .otherwise(None)
        .cast(pl.Int8)
        .alias("shelve_loc")
    )

    logger.info("Ordinal categorical columns encoded successfully")

    if not lazy_output:
        logger.debug(_LOG_COLLECT_TO_DF)
        data = data.collect()

    return data
