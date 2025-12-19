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
BASE_DIR="/workspace"
ACTION=${1:-help}
IMAGE_SIZE=${2:-128}
N_IMAGES=${3:-100000}
LAZY_MODE=${4:-}  # Optional: "lazy" to enable lazy loading

# Construct dataset directory name
DATASET_DIR="${BASE_DIR}/data-${IMAGE_SIZE}-${N_IMAGES}"

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
    
    recovar make_test_dataset "$DATASET_DIR" \
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
    PIPELINE_CMD="recovar pipeline particles.${IMAGE_SIZE}.mrcs --ctf ctf.pkl --poses poses.pkl --mask=from_halfmaps -o pipeline_output"
    
    if [ "$LAZY_MODE" == "lazy" ]; then
        PIPELINE_CMD="$PIPELINE_CMD --lazy"
        echo "Note: Using lazy loading - images will be loaded on-demand to save memory"
    fi
    
    # Run pipeline
    echo "Running: $PIPELINE_CMD"
    eval "$PIPELINE_CMD"
    
    echo "Pipeline completed successfully!"
    echo "Output at: $DATASET_DIR/test_dataset/pipeline_output"
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