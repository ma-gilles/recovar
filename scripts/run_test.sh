#!/bin/bash

# Parameterized script for creating test datasets and running recovar pipeline
# Usage:
#   ./run_test.sh create 128 100000              # Create small dataset
#   ./run_test.sh create 256 300000              # Create large dataset
#   ./run_test.sh pipeline 128 100000            # Run pipeline on small dataset
#   ./run_test.sh pipeline 256 300000            # Run pipeline on large dataset
#   ./run_test.sh pipeline 256 300000 lazy       # Run with lazy loading
#   ./run_test.sh all 128 100000                 # Create dataset and run pipeline
#   ./run_test.sh all 256 300000 lazy            # Create and run with lazy loading

set -e  # Exit on error

# Default values
BASE_DIR="${BASE_DIR:-${DATA_BASE:-/workspace}}"
ACTION=${1:-help}
IMAGE_SIZE=${2:-128}
N_IMAGES=${3:-100000}
LAZY_MODE=${4:-}  # Optional: "lazy" to enable lazy loading
PIPELINE_OUTPUT_DIR="${PIPELINE_OUTPUT_DIR:-pipeline_output}"

# Construct dataset directory name
DATASET_DIR="${BASE_DIR}/data-${IMAGE_SIZE}-${N_IMAGES}"

# Resolve recovar executable (works with detached pixi env on scratch/tigress)
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
    echo "Error: recovar executable not found. Set RECOVAR_BIN or run: pixi run install-recovar" >&2
    exit 1
fi

# Function to print usage
usage() {
    echo "Usage: $0 <action> [image_size] [n_images] [lazy]"
    echo ""
    echo "Actions:"
    echo "  create              Create test dataset"
    echo "  pipeline            Run recovar pipeline on existing dataset"
    echo "  all                 Create dataset and run pipeline"
    echo "  help                Show this help message"
    echo ""
    echo "Parameters:"
    echo "  image_size          Image size (default: 128)"
    echo "  n_images            Number of images (default: 100000)"
    echo "  lazy                Optional: Pass 'lazy' to enable lazy loading"
    echo ""
    echo "Examples:"
    echo "  $0 create 128 100000          # Create small dataset (~30 sec)"
    echo "  $0 create 256 300000          # Create large dataset (~3 min)"
    echo "  $0 pipeline 128 100000        # Run pipeline on small dataset (~30 min)"
    echo "  $0 pipeline 256 300000        # Run pipeline on large dataset (~6 hours)"
    echo "  $0 pipeline 256 300000 lazy   # Run with lazy loading (saves memory)"
    echo "  $0 all 128 100000             # Create and run small dataset"
    echo "  $0 all 256 300000 lazy        # Create and run large dataset with lazy loading"
    echo ""
    echo "Lazy Loading:"
    echo "  Use lazy loading for large datasets that don't fit in memory."
    echo "  It loads images on-demand rather than loading all at once."
    echo "  Trade-off: Saves memory but may be slightly slower due to disk I/O."
    exit 1
}

# Function to create dataset
create_dataset() {
    echo "=========================================="
    echo "Creating test dataset..."
    echo "Dataset directory: $DATASET_DIR"
    echo "Image size: $IMAGE_SIZE"
    echo "Number of images: $N_IMAGES"
    echo "=========================================="
    
    "$RECOVAR_BIN" make_test_dataset "$DATASET_DIR" \
        --image-size="$IMAGE_SIZE" \
        --n-images="$N_IMAGES"
    
    echo "Dataset created successfully at: $DATASET_DIR"
}

# Function to run pipeline
run_pipeline() {
    echo "=========================================="
    echo "Running recovar pipeline..."
    echo "Dataset directory: $DATASET_DIR"
    echo "Image size: $IMAGE_SIZE"
    echo "Output directory name: $PIPELINE_OUTPUT_DIR"
    echo "Lazy loading: $([ "$LAZY_MODE" == "lazy" ] && echo "ENABLED" || echo "DISABLED")"
    echo "=========================================="
    
    # Check if dataset exists
    if [ ! -d "$DATASET_DIR/test_dataset" ]; then
        echo "Error: Dataset not found at $DATASET_DIR/test_dataset"
        echo "Please create the dataset first using: $0 create $IMAGE_SIZE $N_IMAGES"
        exit 1
    fi
    
    cd "$DATASET_DIR/test_dataset/"
    
    # Build the command with optional lazy flag
    PIPELINE_CMD="$RECOVAR_BIN pipeline particles.${IMAGE_SIZE}.mrcs --ctf ctf.pkl --poses poses.pkl --mask=from_halfmaps -o ${PIPELINE_OUTPUT_DIR}"
    
    if [ "$LAZY_MODE" == "lazy" ]; then
        PIPELINE_CMD="$PIPELINE_CMD --lazy"
        echo "Note: Using lazy loading - images will be loaded on-demand to save memory"
    fi
    
    # Run pipeline
    echo "Running: $PIPELINE_CMD"
    eval "$PIPELINE_CMD"
    
    echo "Pipeline completed successfully!"
    echo "Output at: $DATASET_DIR/test_dataset/$PIPELINE_OUTPUT_DIR"
}

# Main script logic
case "$ACTION" in
    create)
        if [ -z "$IMAGE_SIZE" ] || [ -z "$N_IMAGES" ]; then
            echo "Error: Missing parameters for create action"
            usage
        fi
        create_dataset
        ;;
    
    pipeline)
        if [ -z "$IMAGE_SIZE" ] || [ -z "$N_IMAGES" ]; then
            echo "Error: Missing parameters for pipeline action"
            usage
        fi
        run_pipeline
        ;;
    
    all)
        if [ -z "$IMAGE_SIZE" ] || [ -z "$N_IMAGES" ]; then
            echo "Error: Missing parameters for all action"
            usage
        fi
        create_dataset
        echo ""
        run_pipeline
        ;;
    
    help|--help|-h)
        usage
        ;;
    
    *)
        echo "Error: Unknown action '$ACTION'"
        echo ""
        usage
        ;;
esac
