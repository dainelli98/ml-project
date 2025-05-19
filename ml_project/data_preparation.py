"""Module for data preparation functions including feature selection and data splitting."""

import polars as pl
from loguru import logger
from sklearn.model_selection import train_test_split

from ml_project.utils import read_config

# Constants for logging messages
_LOG_CONVERT_TO_LAZY = "Converting DataFrame to LazyFrame"
_LOG_COLLECT_TO_DF = "Collecting LazyFrame to DataFrame"


def get_features_and_target(
    data: pl.DataFrame | pl.LazyFrame, config: dict[str, any] | None = None, lazy_output: bool = False
) -> tuple[pl.DataFrame | pl.LazyFrame, pl.Series]:
    """Get features and target from data based on configuration.

    Args:
        data: The input DataFrame or LazyFrame containing all data
        config: Configuration dictionary. If None, reads from default config.toml
        lazy_output: If True, returns a LazyFrame. Defaults to ``False``.

    Returns:
        A tuple containing (x, y) where x is the feature matrix and y is the target series

    Raises:
        KeyError: If feature_selection or features are not defined in config
        ValueError: If target is not in the data columns
    """
    logger.info("Extracting features and target based on configuration")

    if config is None:
        config = read_config()

    if "feature_selection" not in config or "features" not in config["feature_selection"]:
        error_msg = "Features not defined in config file under 'feature_selection.features'"
        logger.error(error_msg)
        raise KeyError(error_msg)

    if "target" not in config["feature_selection"]:
        error_msg = "Target not defined in config file under 'feature_selection.target'"
        logger.error(error_msg)
        raise KeyError(error_msg)

    features = config["feature_selection"]["features"]
    target = config["feature_selection"]["target"]

    # Ensure we're working with LazyFrame for consistent operations
    if isinstance(data, pl.DataFrame):
        logger.debug(_LOG_CONVERT_TO_LAZY)
        data = data.lazy()

    # Validate features and target exist in the data
    schema = data.collect_schema()
    column_names = schema.names()

    missing_features = [f for f in features if f not in column_names]
    if missing_features:
        error_msg = f"Features {missing_features} not found in data columns"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if target not in column_names:
        error_msg = f"Target '{target}' not found in data columns"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Extract features and target
    x = data.select(features)
    y = data.select(target)

    logger.info(f"Extracted {len(features)} features and target column '{target}'")

    if not lazy_output:
        logger.debug(_LOG_COLLECT_TO_DF)
        x = x.collect()
        y = y.select(target).collect()[target]
    else:
        y = y.select(target)

    return x, y


def split_data(
    x: pl.DataFrame | pl.LazyFrame,
    y: pl.Series | pl.LazyFrame,
    test_size: float = 0.2,
    val_size: float | None = None,
    stratify: bool = False,
    random_state: int = 42,
) -> tuple[pl.DataFrame, ...]:
    """Split data into training and test (and optionally validation) sets.

    Args:
        x: Feature matrix (Polars DataFrame or LazyFrame)
        y: Target vector (Polars Series or LazyFrame)
        test_size: Proportion of data to use for testing (default: 0.2)
        val_size: Proportion of data to use for validation (default: None)
                  If provided, will return train/val/test split
        stratify: Whether to stratify the split based on target values (default: False)
                  Only applicable for classification tasks
        random_state: Random seed for reproducibility (default: 42)

    Returns:
        If val_size is None: (x_train, x_test, y_train, y_test) as Polars DataFrames/Series
        If val_size is not None: (x_train, x_val, x_test, y_train, y_val, y_test) as Polars DataFrames/Series
    """
    logger.info(f"Splitting data with test_size={test_size}, val_size={val_size}, stratify={stratify}")

    # Ensure we have DataFrame objects (not LazyFrame)
    if isinstance(x, pl.LazyFrame):
        logger.debug(_LOG_COLLECT_TO_DF)
        x = x.collect()

    # Handle y which could be a LazyFrame or Series
    if isinstance(y, pl.LazyFrame):
        logger.debug(_LOG_COLLECT_TO_DF)
        y = y.collect()

    stratify_param = y if stratify else None

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )

    if val_size is None:
        return x_train, x_test, y_train, y_test

    logger.debug("Performing two-step split for train/val/test")
    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=val_size,
        random_state=random_state,
        stratify=None if stratify_param is not None else y_train,
    )

    logger.info(f"Split completed: train size={len(x_train)}, val size={len(x_val)}, test size={len(x_test)}")
    return x_train, x_val, x_test, y_train, y_val, y_test
