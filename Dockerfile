FROM ubuntu:25.10 AS builder

ENV PYTHONDONTWRITEBYTECODE=1

ENV PYTHONUNBUFFERED=1

ENV APP_FOLDER=/app/

# Install uv:
COPY --from=ghcr.io/astral-sh/uv:0.5.14 /uv /uvx /bin/

# Install basic dependencies and keep ubuntu up to date:
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        wget \
    && apt-get upgrade -y \
    && apt-get autoremove -y \
    && apt-get autoclean -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR ${APP_FOLDER}

# Install dependencies:
# We first install the uv dependencies without the app, to maximize layer reuse.
COPY ./pyproject.toml ${APP_FOLDER}
COPY ./uv.lock ${APP_FOLDER}
RUN uv sync --frozen --no-install-project --no-cache

# Install project:
COPY . ${APP_FOLDER}
RUN uv sync --frozen --no-cache

FROM builder AS runtime

WORKDIR ${APP_FOLDER}

# Make the run script executable
RUN chmod +x scripts/run.sh

# Default command to show help
ENTRYPOINT ["scripts/run.sh"]
CMD ["--help"]
