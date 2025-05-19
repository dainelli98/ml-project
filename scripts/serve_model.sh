#!/bin/zsh
# Script to serve an MLflow model from the registry using the CLI
# Usage: ./serve_model.sh <model_name> [<stage>] [<host>] [<port>]
# Defaults: host=127.0.0.1, port=7072

MODEL_NAME="$1"
STAGE="$2"
HOST="${3:-127.0.0.1}"
PORT="${4:-7072}"

if [ -z "$MODEL_NAME" ]; then
  echo "Usage: $0 <model_name> [<stage>] [<host>] [<port>]"
  exit 1
fi


# If no stage is provided, default to 'latest'
if [ -n "$STAGE" ]; then
  MODEL_URI="models:/$MODEL_NAME/$STAGE"
else
  MODEL_URI="models:/$MODEL_NAME/latest"
fi

echo "Serving model from URI: $MODEL_URI on $HOST:$PORT"
uv run python -m mlflow models serve -m "$MODEL_URI" -h "$HOST" -p "$PORT"
