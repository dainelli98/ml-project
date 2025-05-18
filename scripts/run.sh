#!/bin/bash
# This script enables running any ML pipeline operation from Docker

set -e

# Default configurations
SCRIPT=""
ARGS=""

# Help function
function show_help {
    echo "Usage: ./run.sh [OPTIONS] COMMAND"
    echo ""
    echo "Commands:"
    echo "  extract        Extract raw data from sources"
    echo "  prepare        Prepare and process the data"
    echo "  preprocess     Preprocess the data for training"
    echo "  train          Train models"
    echo "  inference      Run inference with trained models"
    echo "  pipeline       Run the full training pipeline"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message and exit"
    echo "  -- ARGS        Any arguments to pass to the underlying script"
    echo ""
    echo "Example:"
    echo "  ./run.sh extract"
    echo "  ./run.sh train -- --model-type GradientBoostingRegressor"
    echo "  ./run.sh inference -- --model-file models/GradientBoostingRegressor_best_model.pkl --input-file data/x_test.csv"
}

# Process arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            show_help
            exit 0
            ;;
        extract)
            SCRIPT="ml_project/scripts/extract_data.py"
            shift
            ;;
        prepare)
            SCRIPT="ml_project/scripts/prepare_data.py"
            shift
            ;;
        preprocess)
            SCRIPT="ml_project/scripts/preprocess_data.py"
            shift
            ;;
        train)
            SCRIPT="ml_project/scripts/train_model.py"
            shift
            ;;
        inference)
            SCRIPT="ml_project/scripts/run_inference.py"
            shift
            ;;
        pipeline)
            # Run full pipeline sequentially
            echo "Running full training pipeline..."
            .venv/bin/python ml_project/scripts/extract_data.py
            .venv/bin/python ml_project/scripts/preprocess_data.py
            .venv/bin/python ml_project/scripts/prepare_data.py
            .venv/bin/python ml_project/scripts/train_model.py
            echo "Training pipeline completed successfully!"
            exit 0
            ;;
        --)
            shift
            ARGS="$@"
            break
            ;;
        *)
            echo "Unknown option: $key"
            show_help
            exit 1
            ;;
    esac
done

# Check if script was specified
if [ -z "$SCRIPT" ]; then
    echo "Error: No command specified"
    show_help
    exit 1
fi

# Run the specified script with any provided arguments
echo "Running: python $SCRIPT $ARGS"
.venv/bin/python $SCRIPT $ARGS
