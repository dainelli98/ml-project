"""Utility functions for the ML project."""

from pathlib import Path
from typing import Any

import tomli
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

# Import model classes
from sklearn.linear_model import LinearRegression


def read_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Read the configuration file and return it as a dictionary.

    Args:
        config_path: Path to the configuration file. If None, use the default path.

    Returns:
        Configuration as a dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """
    if config_path is None:
        # Use the default path relative to the project root
        config_path = Path(__file__).parents[1] / "config" / "config.toml"

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Read and parse the TOML file
    with open(config_path, "rb") as f:
        return tomli.load(f)


def get_model_class(model_name: str) -> type:
    """Convert a model name string to its corresponding scikit-learn class.

    Args:
        model_name: Name of the model as specified in the configuration file.

    Returns:
        The scikit-learn model class.

    Raises:
        ValueError: If the model name is not recognized.
    """
    model_classes = {
        "LinearRegression": LinearRegression,
        "RandomForestRegressor": RandomForestRegressor,
        "GradientBoostingRegressor": GradientBoostingRegressor,
        "LGBMRegressor": LGBMRegressor,
    }

    if model_name not in model_classes:
        raise ValueError(f"Unknown model name: {model_name}. Available models are: {list(model_classes.keys())}")

    if model_classes[model_name] is None:
        raise ImportError(
            f"The model {model_name} requires additional dependencies "
            f"that are not installed. Please install the required packages."
        )

    return model_classes[model_name]


def create_model_instance(model_name: str, hyperparameters: dict[str, Any] | None = None) -> Any:
    """Create an instance of a model with the specified hyperparameters.

    Args:
        model_name: Name of the model as specified in the configuration file.
        hyperparameters: Hyperparameters for the model.

    Returns:
        An instance of the model.
    """
    model_class = get_model_class(model_name)

    if hyperparameters is None:
        return model_class()

    return model_class(**hyperparameters)


def get_hyperparameter_grid(config: dict[str, Any], model_name: str) -> dict[str, list[Any]]:
    """Extract the hyperparameter grid for a specific model from the configuration.

    Args:
        config: The configuration dictionary.
        model_name: Name of the model as specified in the configuration file.

    Returns:
        The hyperparameter grid for the specified model.

    Raises:
        KeyError: If the hyperparameters for the model are not found in the configuration.
    """
    try:
        return config["modelling"]["hyperparameters"][model_name]
    except KeyError as e:
        raise KeyError(f"Hyperparameters for model {model_name} not found in the configuration.") from e
