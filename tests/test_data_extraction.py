"""Tests for data extraction functionality."""

import io
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from ml_project.data_extraction import CARSEATS_RAW_DATA_URL, fetch_carseats_data


@pytest.fixture
def mock_csv_data() -> str:
    """Create a small mock CSV data similar to the Carseats dataset.

    Returns:
        str: Mock CSV data as a string.
    """
    return """Sales,CompPrice,Income,Advertising,Population,Price,ShelveLoc,Age,Education,Urban,US
10,138,73,11,276,120,Bad,42,17,Yes,Yes
9,111,48,16,260,83,Good,65,10,Yes,Yes
7,113,35,10,269,80,Medium,59,12,Yes,Yes"""


@patch("ml_project.data_extraction.pl.scan_csv")
def test_fetch_carseats_data_lazy(mock_scan_csv: MagicMock, mock_csv_data: str) -> None:
    """Test fetching car seats data in lazy mode.

    Args:
        mock_scan_csv: Mock for polars scan_csv function.
        mock_csv_data: Mock CSV data fixture.
    """
    # Setup the mock
    mock_lazy_frame = MagicMock(spec=pl.LazyFrame)
    mock_scan_csv.return_value = mock_lazy_frame

    # Call the function
    result = fetch_carseats_data(lazy=True)

    # Assertions
    mock_scan_csv.assert_called_once_with(CARSEATS_RAW_DATA_URL)
    assert result is mock_lazy_frame


@patch("ml_project.data_extraction.pl.read_csv")
def test_fetch_carseats_data_eager(mock_read_csv: MagicMock, mock_csv_data: str) -> None:
    """Test fetching car seats data in eager mode.

    Args:
        mock_read_csv: Mock for polars read_csv function.
        mock_csv_data: Mock CSV data fixture.
    """
    # Setup the mock
    mock_df = MagicMock(spec=pl.DataFrame)
    mock_read_csv.return_value = mock_df

    # Call the function
    result = fetch_carseats_data(lazy=False)

    # Assertions
    mock_read_csv.assert_called_once_with(CARSEATS_RAW_DATA_URL)
    assert result is mock_df


@patch("ml_project.data_extraction.pl.scan_csv")
@patch("ml_project.data_extraction.pl.read_csv")
def test_fetch_carseats_data_with_env_url(
    mock_read_csv: MagicMock,
    mock_scan_csv: MagicMock,
    mock_csv_data: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test fetching car seats data with a custom URL from environment variable.

    Args:
        mock_read_csv: Mock for polars read_csv function.
        mock_scan_csv: Mock for polars scan_csv function.
        mock_csv_data: Mock CSV data fixture.
        monkeypatch: Pytest fixture for patching environment.
    """
    # Setup the environment variable and patch the CARSEATS_RAW_DATA_URL variable
    test_url = "https://test-url.com/carseats.csv"
    with patch("ml_project.data_extraction.CARSEATS_RAW_DATA_URL", test_url):
        # Setup the mocks
        mock_lazy_frame = MagicMock(spec=pl.LazyFrame)
        mock_scan_csv.return_value = mock_lazy_frame

        # Call the function
        result = fetch_carseats_data(lazy=True)

        # Assertions
        mock_scan_csv.assert_called_once_with(test_url)
        assert result is mock_lazy_frame


def test_fetch_carseats_data_content(mock_csv_data: str) -> None:
    """Test that the function returns the expected data content.

    Args:
        mock_csv_data: Mock CSV data fixture.
    """
    # Create a DataFrame directly from our mock CSV
    expected_df = pl.read_csv(io.StringIO(mock_csv_data))

    # Patch both the URL and the read_csv function
    with (
        patch("ml_project.data_extraction.CARSEATS_RAW_DATA_URL", "mock_url"),
        patch("ml_project.data_extraction.pl.read_csv") as mock_read_csv,
    ):
        # Configure the mock to return our expected DataFrame
        mock_read_csv.return_value = expected_df

        # Call the function with lazy=False to get a DataFrame
        result = fetch_carseats_data(lazy=False)

        # Assertions
        mock_read_csv.assert_called_once_with("mock_url")
        assert result is expected_df
        assert result.shape == (3, 11)  # 3 rows, 11 columns as per our mock data
        assert "Sales" in result.columns
        assert "ShelveLoc" in result.columns
