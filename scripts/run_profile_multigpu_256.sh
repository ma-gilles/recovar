#!/bin/bash

# Multi-GPU profiling script for RECOVAR pipeline (256-300k dataset)
# Usage:
#   ./run_profile_multigpu_256.sh 1    # Profile with 1 GPU (baseline)
#   ./run_profile_multigpu_256.sh 2    # Profile with 2 GPUs
#   ./run_profile_multigpu_256.sh 4    # Profile with 4 GPUs

set -e  # Exit on error

# Configuration
BASE_DIR="${BASE_DIR:-${DATA_BASE:-/workspace}}"
N_GPUS=${1:-1}
IMAGE_SIZE=256
N_IMAGES=300000
LAZY_MODE="${LAZY_MODE:-1}"  # Control lazy loading

# Dataset directory
DATASET_DIR="${BASE_DIR}/data-${IMAGE_SIZE}-${N_IMAGES}"

# Use RECOVAR_BIN if set (e.g. by batch script from primed env on scratch); else find recovar
if [ -n "${RECOVAR_BIN:-}" ] && [ -x "$RECOVAR_BIN" ]; then
  :
elif command -v pixi >/dev/null 2>&1; then
  PIXI_PREFIX=$(pixi info -p 2>/dev/null) || true
  if [ -n "$PIXI_PREFIX" ] && [ -x "$PIXI_PREFIX/bin/recovar" ]; then
    export PATH="$PIXI_PREFIX/bin:$PATH"
  fi
  RECOVAR_BIN="$(which recovar 2>/dev/null)" || true
fi
if [ -z "${RECOVAR_BIN:-}" ] && [ -n "${RATTLER_CACHE_DIR:-}" ] && [ -d "$RATTLER_CACHE_DIR/envs" ]; then
  RECOVAR_BIN=$(find "$RATTLER_CACHE_DIR/envs" -name recovar -type f -executable 2>/dev/null | head -1)
fi
if [ -z "${RECOVAR_BIN:-}" ] || [ ! -x "$RECOVAR_BIN" ]; then
  echo "Error: recovar not found. Set RECOVAR_BIN or run: pixi run install-recovar" >&2
  exit 1
fi

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
echo "Multi-GPU Profiling (256-300k dataset)"
echo "Dataset directory: $DATASET_DIR/test_dataset"
echo "Number of GPUs: $N_GPUS"
echo "Images to process: $N_IMAGES (full dataset)"
echo "=========================================="

cd "$DATASET_DIR/test_dataset/"

# Setup profiling output
JOB_TAG="${JOB_TAG:-${SLURM_JOB_ID:-local_$$}}"
PROFILE_OUTPUT="recovar_${N_GPUS}gpu_profile_${JOB_TAG}.nsys-rep"

# Nsys flags - full capture (no capture-range restriction for GUI compatibility)
# Reduce CPU sampling to minimize file size while keeping CUDA/NVTX traces
# Use explicit nsys path if NSYS_BIN is set to ensure correct version
NSYS_CAPTURE_RANGE="${NSYS_CAPTURE_RANGE:-}"
NSYS_SAMPLE="${NSYS_SAMPLE:-none}"  # Disable CPU sampling to reduce file size
NSYS_FLAGS="--trace=cuda,nvtx --stats=true --force-overwrite=true --sample=$NSYS_SAMPLE"
if [ -n "$NSYS_CAPTURE_RANGE" ]; then
    NSYS_FLAGS="$NSYS_FLAGS --capture-range=$NSYS_CAPTURE_RANGE"
fi

# Use explicit nsys binary if provided (for version control)
NSYS_CMD="nsys"
if [ -n "${NSYS_BIN:-}" ] && [ -x "$NSYS_BIN" ]; then
    NSYS_CMD="$NSYS_BIN"
    echo "Using explicit nsys binary: $NSYS_CMD"
    $NSYS_CMD --version || true
fi

# Build the pipeline command based on number of GPUs
if [ "$N_GPUS" -eq 1 ]; then
    # Baseline: no multi-GPU flags
    echo "Running baseline profiling (1 GPU, no --multi-gpu flag)"
    PIPELINE_CMD="$RECOVAR_BIN pipeline particles.${IMAGE_SIZE}.mrcs \
        --ctf ctf.pkl \
        --poses poses.pkl \
        --mask=from_halfmaps \
        -o pipeline_output_${N_GPUS}gpu_profile"
    if [ "$LAZY_MODE" = "1" ]; then
        PIPELINE_CMD="$PIPELINE_CMD --lazy"
    fi
else
    # Multi-GPU profiling
    echo "Running multi-GPU profiling ($N_GPUS GPUs)"
    PIPELINE_CMD="$RECOVAR_BIN pipeline particles.${IMAGE_SIZE}.mrcs \
        --ctf ctf.pkl \
        --poses poses.pkl \
        --mask=from_halfmaps \
        --multi-gpu \
        --n-gpus $N_GPUS \
        -o pipeline_output_${N_GPUS}gpu_profile"
    if [ "$LAZY_MODE" = "1" ]; then
        PIPELINE_CMD="$PIPELINE_CMD --lazy"
    fi
fi

# Run with profiling
echo "Running: $NSYS_CMD profile $NSYS_FLAGS -o $PROFILE_OUTPUT $PIPELINE_CMD"
$NSYS_CMD profile $NSYS_FLAGS -o "$PROFILE_OUTPUT" $PIPELINE_CMD

echo "=========================================="
echo "Profiling completed successfully!"
echo "Profile saved to: $DATASET_DIR/test_dataset/$PROFILE_OUTPUT"
echo "Output at: $DATASET_DIR/test_dataset/pipeline_output_${N_GPUS}gpu_profile"
echo "=========================================="

