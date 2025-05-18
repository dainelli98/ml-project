"""Entrypoint module for ml_project."""

import typer

from ml_project.scripts.extract_data import extract_data

app = typer.Typer(help="Machine Learning Project CLI")
app.add_typer(extract_data, name="extract-data", help="Extract and save raw data")


def main() -> None:
    """Entrypoint for ml_project."""
    app()
