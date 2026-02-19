#!/bin/bash

# Multi-GPU profiling script for RECOVAR pipeline (128-100k dataset)
# Usage:
#   ./run_profile_multigpu.sh 1    # Profile with 1 GPU (baseline)
#   ./run_profile_multigpu.sh 2    # Profile with 2 GPUs
#   ./run_profile_multigpu.sh 4    # Profile with 4 GPUs

set -e  # Exit on error

# Configuration
BASE_DIR="${BASE_DIR:-${DATA_BASE:-/workspace}}"
N_GPUS=${1:-1}
IMAGE_SIZE=${IMAGE_SIZE:-128}
N_IMAGES=${N_IMAGES:-100000}
N_IMAGES_SUBSET="${N_IMAGES_SUBSET:-}"

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
if [ -n "$N_IMAGES_SUBSET" ]; then
    echo "Images to process: $N_IMAGES_SUBSET (subset of $N_IMAGES)"
else
    echo "Images to process: $N_IMAGES (full dataset)"
fi
echo "=========================================="

cd "$DATASET_DIR/test_dataset/"

# Setup profiling output
JOB_TAG="${JOB_TAG:-${SLURM_JOB_ID:-local_$$}}"
PROFILE_OUTPUT="recovar_${N_GPUS}gpu_profile_${JOB_TAG}.nsys-rep"

# Nsys flags - minimal set to avoid UI crashes
# If you want to limit capture to cudaProfilerStart/Stop, set:
#   export NSYS_CAPTURE_RANGE=cudaProfilerApi
NSYS_CAPTURE_RANGE="${NSYS_CAPTURE_RANGE:-}"
NSYS_FLAGS="--trace=cuda,nvtx --stats=true --force-overwrite=true"
if [ -n "$NSYS_CAPTURE_RANGE" ]; then
    NSYS_FLAGS="$NSYS_FLAGS --capture-range=$NSYS_CAPTURE_RANGE"
fi

# Build the pipeline command based on number of GPUs
if [ "$N_GPUS" -eq 1 ]; then
    # Baseline: no multi-GPU flags
    echo "Running baseline profiling (1 GPU, no --multi-gpu flag)"
    PIPELINE_CMD="recovar pipeline particles.${IMAGE_SIZE}.mrcs \
        --ctf ctf.pkl \
        --poses poses.pkl \
        --mask=from_halfmaps \
        -o pipeline_output_${N_GPUS}gpu_profile \
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
        --lazy"
fi

if [ -n "$N_IMAGES_SUBSET" ]; then
    PIPELINE_CMD="$PIPELINE_CMD --n-images $N_IMAGES_SUBSET"
fi

# Run with profiling
echo "Running: nsys profile $NSYS_FLAGS -o $PROFILE_OUTPUT $PIPELINE_CMD"
set +e
nsys profile $NSYS_FLAGS -o "$PROFILE_OUTPUT" $PIPELINE_CMD
NSYS_STATUS=$?
set -e
if [ $NSYS_STATUS -ne 0 ]; then
    if [ -f "$PROFILE_OUTPUT" ]; then
        echo "Warning: nsys exited with code $NSYS_STATUS but report exists. Continuing."
    else
        exit $NSYS_STATUS
    fi
fi

echo "=========================================="
echo "Profiling completed successfully!"
echo "Profile saved to: $DATASET_DIR/test_dataset/$PROFILE_OUTPUT"
echo "Output at: $DATASET_DIR/test_dataset/pipeline_output_${N_GPUS}gpu_profile"
echo "=========================================="

