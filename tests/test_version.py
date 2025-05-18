"""Basic test to check that the project is properly setup."""

from ml_project import __version__


def test_version():
    """Test version."""
    assert __version__ == "0.1.0", "Version is not the expected one."
