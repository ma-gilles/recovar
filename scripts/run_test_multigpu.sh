#!/bin/bash

# Multi-GPU testing script for RECOVAR pipeline
# Usage:
#   ./run_test_multigpu.sh 1    # Test with 1 GPU (baseline, no --multi-gpu flag)
#   ./run_test_multigpu.sh 2    # Test with 2 GPUs
#   ./run_test_multigpu.sh 4    # Test with 4 GPUs
#   ./run_test_multigpu.sh 8    # Test with 8 GPUs

set -e  # Exit on error

# Configuration
BASE_DIR="/workspace"
N_GPUS=${1:-1}
IMAGE_SIZE=128
N_IMAGES=100000

# Dataset directory
DATASET_DIR="${BASE_DIR}/data-${IMAGE_SIZE}-${N_IMAGES}"

# Validate input
if [[ ! "$N_GPUS" =~ ^[1248]$ ]]; then
    echo "Error: Invalid number of GPUs. Must be 1, 2, 4, or 8"
    exit 1
fi

# Check if dataset exists
if [ ! -d "$DATASET_DIR/test_dataset" ]; then
    echo "Error: Dataset not found at $DATASET_DIR/test_dataset"
    echo "Please create the dataset first."
    exit 1
fi

echo "=========================================="
echo "Multi-GPU Pipeline Test"
echo "Dataset directory: $DATASET_DIR/test_dataset"
echo "Number of GPUs: $N_GPUS"
echo "=========================================="

cd "$DATASET_DIR/test_dataset/"

# Build the command based on number of GPUs
OUTPUT_DIR="pipeline_output_${N_GPUS}gpu"

if [ "$N_GPUS" -eq 1 ]; then
    # Baseline: no multi-GPU flags
    echo "Running baseline test (1 GPU, no --multi-gpu flag)"
    PIPELINE_CMD="recovar pipeline particles.${IMAGE_SIZE}.mrcs --ctf ctf.pkl --poses poses.pkl --mask=from_halfmaps -o $OUTPUT_DIR"
else
    # Multi-GPU test
    echo "Running multi-GPU test ($N_GPUS GPUs)"
    PIPELINE_CMD="recovar pipeline particles.${IMAGE_SIZE}.mrcs --ctf ctf.pkl --poses poses.pkl --mask=from_halfmaps --multi-gpu --n-gpus $N_GPUS -o $OUTPUT_DIR"
fi

# Run pipeline
echo "Running: $PIPELINE_CMD"
eval "$PIPELINE_CMD"

echo "=========================================="
echo "Pipeline completed successfully!"
echo "Output at: $DATASET_DIR/test_dataset/$OUTPUT_DIR"
echo "=========================================="





