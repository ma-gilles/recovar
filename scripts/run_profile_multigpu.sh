#!/bin/bash

# Multi-GPU profiling script for RECOVAR pipeline (128-100k dataset)
# Usage:
#   ./run_profile_multigpu.sh 1    # Profile with 1 GPU (baseline)
#   ./run_profile_multigpu.sh 2    # Profile with 2 GPUs
#   ./run_profile_multigpu.sh 4    # Profile with 4 GPUs

set -e  # Exit on error

# Configuration
BASE_DIR="/workspace"
N_GPUS=${1:-1}
IMAGE_SIZE=128
N_IMAGES=100000
N_IMAGES_SUBSET=100000  # Use full dataset for profiling

# Dataset directory
DATASET_DIR="${BASE_DIR}/data-${IMAGE_SIZE}-${N_IMAGES}"

# Validate input
if [[ ! "$N_GPUS" =~ ^[124]$ ]]; then
    echo "Error: Invalid number of GPUs. Must be 1, 2, or 4"
    exit 1
fi

# Check if dataset exists
if [ ! -d "$DATASET_DIR/test_dataset" ]; then
    echo "Error: Dataset not found at $DATASET_DIR/test_dataset"
    echo "Please create the dataset first."
    exit 1
fi

echo "=========================================="
echo "Multi-GPU Profiling (128-100k dataset)"
echo "Dataset directory: $DATASET_DIR/test_dataset"
echo "Number of GPUs: $N_GPUS"
echo "Images to process: $N_IMAGES_SUBSET"
echo "=========================================="

cd "$DATASET_DIR/test_dataset/"

# Setup profiling output
PROFILE_OUTPUT="recovar_${N_GPUS}gpu_profile.nsys-rep"

# Nsys flags - minimal set to avoid UI crashes
NSYS_FLAGS="--trace=cuda,nvtx --stats=true --capture-range=cudaProfilerApi --force-overwrite=true"

# Build the pipeline command based on number of GPUs
if [ "$N_GPUS" -eq 1 ]; then
    # Baseline: no multi-GPU flags
    echo "Running baseline profiling (1 GPU, no --multi-gpu flag)"
    PIPELINE_CMD="recovar pipeline particles.${IMAGE_SIZE}.mrcs \
        --ctf ctf.pkl \
        --poses poses.pkl \
        --mask=from_halfmaps \
        -o pipeline_output_${N_GPUS}gpu_profile \
        --n-images $N_IMAGES_SUBSET \
        --lazy"
else
    # Multi-GPU profiling
    echo "Running multi-GPU profiling ($N_GPUS GPUs)"
    PIPELINE_CMD="recovar pipeline particles.${IMAGE_SIZE}.mrcs \
        --ctf ctf.pkl \
        --poses poses.pkl \
        --mask=from_halfmaps \
        --multi-gpu \
        --n-gpus $N_GPUS \
        -o pipeline_output_${N_GPUS}gpu_profile \
        --n-images $N_IMAGES_SUBSET \
        --lazy"
fi

# Run with profiling
echo "Running: nsys profile $NSYS_FLAGS -o $PROFILE_OUTPUT $PIPELINE_CMD"
nsys profile $NSYS_FLAGS -o "$PROFILE_OUTPUT" $PIPELINE_CMD

echo "=========================================="
echo "Profiling completed successfully!"
echo "Profile saved to: $DATASET_DIR/test_dataset/$PROFILE_OUTPUT"
echo "Output at: $DATASET_DIR/test_dataset/pipeline_output_${N_GPUS}gpu_profile"
echo "=========================================="

