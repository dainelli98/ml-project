.PHONY: clean

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

PYTHON_INTERPRETER=uv run python
ifeq (,$(shell which uv))
HAS_UV=False
else
HAS_UV=True
endif


#################################################################################
# COMMANDS                                                                      #
#################################################################################
init:
	if [ "$(HAS_UV)" = "True" ]; then \
		git config filter.strip-notebook-output.clean 'jupyter nbconvert --clear-output --to=notebook --stdin --stdout '; \
		uv sync --frozen --all-extras; \
		uvx pre-commit install; \
		uvx pre-commit install --hook-type commit-msg; \
	else \
		git config filter.strip-notebook-output.clean 'jupyter nbconvert --clear-output --to=notebook --stdin --stdout '; \
		@echo ">>>No uv detected, not installing pre-commit and packages"; \
	fi

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Start MLflow server on localhost:5000
mlflow-server:
	$(PYTHON_INTERPRETER) -m mlflow ui --host 0.0.0.0 --port 5000

#################################################################################
# DOCKER COMMANDS                                                              #
#################################################################################
## Build the Docker image
docker-build:
	docker build -t ml-project:latest .

## Run a specific command with mounted data and models directories
docker-run:
	docker run --rm -v $(PROJECT_DIR)/data:/app/data -v $(PROJECT_DIR)/models:/app/models ml-project:latest $(cmd)

## Run the full training pipeline with mounted data and models directories
docker-pipeline:
	docker run --rm -v $(PROJECT_DIR)/data:/app/data -v $(PROJECT_DIR)/models:/app/models ml-project:latest pipeline

## Run data extraction with mounted data directory
docker-extract:
	docker run --rm -v $(PROJECT_DIR)/data:/app/data ml-project:latest extract

## Run data preparation with mounted data directory
docker-prepare:
	docker run --rm -v $(PROJECT_DIR)/data:/app/data ml-project:latest prepare

## Run data preprocessing with mounted data directory
docker-preprocess:
	docker run --rm -v $(PROJECT_DIR)/data:/app/data ml-project:latest preprocess

## Run model training with mounted data and models directories
docker-train:
	docker run --rm -v $(PROJECT_DIR)/data:/app/data -v $(PROJECT_DIR)/models:/app/models ml-project:latest train

## Run inference with mounted data and models directories
docker-inference:
	docker run --rm -v $(PROJECT_DIR)/data:/app/data -v $(PROJECT_DIR)/models:/app/models ml-project:latest inference -- $(args)

## Start an interactive shell inside the container with mounted directories
docker-shell:
	docker run --rm -it -v $(PROJECT_DIR)/data:/app/data -v $(PROJECT_DIR)/models:/app/models ml-project:latest /bin/bash
