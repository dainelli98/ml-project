# ml-project

![Python version](https://img.shields.io/badge/python-3.12-blue.svg)
![uv version](https://img.shields.io/badge/uv-%3E%3D0.5.0-purple.svg
)

--------

## Important

Please always create a separate environment for your project. There is a handy utility provided that could do
this for you, [read below](#make-init).

Please keep your pyproject.toml and uv.lock up to date with environments dependencies.
You can find more information about the [documentation and set up here](https://docs.astral.sh/uv/).

If your project version supports a different python version, please change in the above badges along with
the pyproject.toml and Dockerfile.

Please remember to edit the pyproject.toml to suit your project. We are hoping to automate it as GitHub templates
allow for it in the future.

Please do not put confidential data useful for this project anywhere other than data folder.
The .gitignore is designed to ignore the folder contents making the project git safe.

## Passwords

Some applications connect to any third-party service (for example a SQL server) requiring
a user name and a password.

This kind of information shall be never be hardcoded in the code or saved in any configuration
file that may be uploaded to the repository.

A simple way to handle critical data is saving them as environment variables.

Simply create a `.env` file at the root of the repository. Then and save user names and passwords
like:

```bash
YOUR_USERNAME=your_username
YOUR_PASSWORD=your_password
```

You can then read `.env` files for Python code with the `dotenv` package.

`.env` files are excluded from the Git repository in the `.gitignore` file.

--------

## Testing

You can run the tests for this project by running the following command in your Python environment:

```shell
pytest tests
```

To verify correct setup for your environment you can run the tests.

--------

## Actions

There are some actions that run on this project by default:

1. Build Documentation
    - Build documentation for the project, ready for use with github pages
2. Version & Release
    - Use Semantic Versioning for your project. This relies on conventional commits, there is a check for this in the pre-commit hooks.
3. Pull Request QA Checks
    - Run a series of checks on the pull request to ensure the code is up to standard.

--------

## Project Organization

```shell
    ├── .github/workflows  <- Github actions.
    ├── Makefile           <- Makefile with utility commands.
    ├── Dockerfile         <- Create a Docker image for this project.
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- The data folder for this project. This is always ignored by git by design.
    │
    ├── docs               <- A default mkdocs project.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries.
    │
    ├── notebooks          <- Jupyter notebooks. Please ensure the output is stripped out of any
    confidential information.
    │
    ├── scripts            <- Helper scripts for running the project.
    │
    ├── ml_project         <- Source code for use in this project.
    │
```

--------

## Docker Usage

This project can be run in a Docker container for reproducibility and ease of use. The Docker setup includes mounting of data and model directories to ensure persistence of data and trained models.

### Building the Docker Image

```shell
make docker-build
```

### Running the Full Training Pipeline

```shell
make docker-pipeline
```

### Running Individual Steps

You can run each step of the pipeline individually:

```shell
# Extract data
make docker-extract

# Prepare data
make docker-prepare

# Preprocess data
make docker-preprocess

# Train models
make docker-train

# Run inference
make docker-inference args="--model-file models/GradientBoostingRegressor_best_model.pkl --input-file data/x_test.csv"
```

### Custom Commands

For running custom commands:

```shell
make docker-run cmd="extract -- --output data/custom_raw_data.csv"
```

### Interactive Shell

To get an interactive shell inside the container:

```shell
make docker-shell
```

--------

## Set up Documentation Web Page

You can get your project documentation as a github website out of the box using this template.
To enable it you must:

1. Go to **Settings**.
1. Under the **Code and automation** header in the sidebar click **Pages**.
1. Under **Build and deployment** select **Deploy from a branch**
1. Choose **Branch** select **gh-pages**
1. Scroll to the bottom of the page and go to the url in the **Enforce HTTPS** section.

You should be able to view your documentation site! You can change the colours and some of the formatting in the `mkdocs.yaml` file.

--------

## Utilities Available

### make init

Would create a uv environment for you along with pre-commit

### make clean

A utility tool to clear pycache and pyc or pyo files
