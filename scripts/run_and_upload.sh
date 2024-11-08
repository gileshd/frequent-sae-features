#!/bin/bash

# Default values
SMOKE_TEST=false
OUTPUT_DIR="prodlda_checkpoints"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke-test)
            SMOKE_TEST=true
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Construct model filename
MODEL_NAME="prodlda_model_${TIMESTAMP}.pt"
MODEL_PATH="${OUTPUT_DIR}/${MODEL_NAME}"

# Construct command based on smoke test flag
CMD="python launch_prodlda.py --output ${MODEL_PATH}"
if [ "$SMOKE_TEST" = true ]; then
    CMD="${CMD} --smoke-test"
fi

# Run the training script
echo "Starting training..."
echo "Command: $CMD"
$CMD

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "Training completed successfully. Uploading to GCS..."
    # Upload to Google Cloud Storage
    gcloud storage cp "${MODEL_PATH}" "gs://jug-villian-208/test/${MODEL_NAME}"
    
    if [ $? -eq 0 ]; then
        echo "Upload successful!"
        echo "File available at: gs://jug-villian-208/test/${MODEL_NAME}"
    else
        echo "Upload failed!"
        exit 1
    fi
else
    echo "Training failed!"
    exit 1
fi

