.PHONY: clean

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

PYTHON_INTERPRETER=python
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
