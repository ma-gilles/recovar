#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=800G
#SBATCH --constraint=gpu80
#SBATCH --partition=cryoem
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=mg6942@princeton.edu
#SBATCH --output=/home/mg6942/slurmo/%x.%j.out
#SBATCH --error=/home/mg6942/slurmo/%x.%j.out
#SBATCH --account=amits

# run_comparison_test.sh
# Run this AFTER prepare_test_environments.sh has been run
# This script can be run directly on head node or submitted via sbatch to compute node

set -e

# Configuration: Set TEST_MODE
TEST_MODE="${1:-quick}"  # Default to quick if no argument provided

if [ "$TEST_MODE" = "test" ]; then
    N_IMAGES=1000
    GRID_SIZE=64
    TOMO_TILTS=-1
    echo "=========================================="
    echo "TEST MODE (Script verification only)"
    echo "=========================================="un 
elif [ "$TEST_MODE" = "quick" ]; then
    N_IMAGES=10000
    GRID_SIZE=128
    TOMO_TILTS=-1
    echo "=========================================="
    echo "QUICK TEST MODE (Cryo-EM)"
    echo "=========================================="
elif [ "$TEST_MODE" = "full" ]; then
    N_IMAGES=100000
    GRID_SIZE=256
    TOMO_TILTS=-1
    echo "=========================================="
    echo "FULL TEST MODE (Cryo-EM)"
    echo "=========================================="
elif [ "$TEST_MODE" = "tomo_quick" ]; then
    N_IMAGES=5000
    GRID_SIZE=128
    TOMO_TILTS=10
    echo "=========================================="
    echo "QUICK TOMOGRAPHY TEST MODE"
    echo "=========================================="
elif [ "$TEST_MODE" = "tomo_full" ]; then
    N_IMAGES=50000
    GRID_SIZE=256
    TOMO_TILTS=20
    echo "=========================================="
    echo "FULL TOMOGRAPHY TEST MODE"
    echo "=========================================="
else
    echo "Error: TEST_MODE must be 'test', 'quick', 'full', 'tomo_quick', or 'tomo_full'"
    echo "Usage: ./run_comparison_test.sh [test|quick|full|tomo_quick|tomo_full]"
    echo ""
    echo "Modes:"
    echo "  test       - 100 images, 64³ grid, cryo-EM (~1 min) - script verification only"
    echo "  quick      - 5K images, 128³ grid, cryo-EM (~30 min)"
    echo "  full       - 100K images, 256³ grid, cryo-EM (~6 hours)"
    echo "  tomo_quick - 5K images, 128³ grid, 10 tilts (~1 hour)"
    echo "  tomo_full  - 50K images, 256³ grid, 20 tilts (~12 hours)"
    exit 1
fi

echo "Testing RECOVAR Reimplementation"
echo "Node: $(hostname)"
if [ "$TOMO_TILTS" -gt 0 ]; then
    echo "Configuration: $N_IMAGES images, ${GRID_SIZE}³ grid, $TOMO_TILTS tilts"
else
    echo "Configuration: $N_IMAGES images, ${GRID_SIZE}³ grid (cryo-EM)"
fi
echo "Started at: $(date)"
echo "=========================================="

# Setup directories
BASE_DIR="/home/mg6942/recovar"
RESULTS_DIR="/home/mg6942/mytigress/recovar_test_results_${TEST_MODE}_$$"
OLD_DIR="$RESULTS_DIR/old_version"
NEW_DIR="$RESULTS_DIR/new_version"
VOLUME_DIR="/scratch/gpfs/mg6942/cooperative/models/renamed/"

# Environment names (created by prepare_test_environments.sh)
OLD_ENV="recovar_old_test"
NEW_ENV="recovar_new_test"

mkdir -p "$RESULTS_DIR"
mkdir -p "$OLD_DIR"
mkdir -p "$NEW_DIR"
mkdir -p "/home/mg6942/recovar/test_results"

echo "Results directory: $RESULTS_DIR"
echo "Volume directory: $VOLUME_DIR"
echo ""

# Initialize conda for this shell session
eval "$(conda shell.bash hook)"

# Check that environments exist
echo "Checking for test environments..."
if ! conda env list | grep -q "^${OLD_ENV} "; then
    echo ""
    echo "=========================================="
    echo "ERROR: Environment '$OLD_ENV' not found!"
    echo "=========================================="
    echo ""
    echo "Please run prepare_test_environments.sh first:"
    echo "  ./prepare_test_environments.sh"
    echo ""
    exit 1
fi

if ! conda env list | grep -q "^${NEW_ENV} "; then
    echo ""
    echo "=========================================="
    echo "ERROR: Environment '$NEW_ENV' not found!"
    echo "=========================================="
    echo ""
    echo "Please run prepare_test_environments.sh first:"
    echo "  ./prepare_test_environments.sh"
    echo ""
    exit 1
fi

echo "✓ Found environment: $OLD_ENV"
echo "✓ Found environment: $NEW_ENV"
echo ""

echo "=========================================="
echo "STEP 1: Run OLD version tests"
echo "=========================================="

echo "Activating OLD version environment: $OLD_ENV"
conda activate "$OLD_ENV"

# Verify environment is active
if [[ "$CONDA_DEFAULT_ENV" != "$OLD_ENV" ]]; then
    echo "Error: Failed to activate conda environment $OLD_ENV"
    exit 1
fi

echo "Running tests with OLD version..."
echo "  Environment: $OLD_ENV"
echo "  N_IMAGES=$N_IMAGES, GRID_SIZE=$GRID_SIZE, TOMO_TILTS=$TOMO_TILTS"

# Verify which recovar is being used
echo "  Verifying OLD version installation:"
python -c "import recovar; import os; print(f'    recovar path: {os.path.dirname(recovar.__file__)}'); import subprocess; result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=os.path.dirname(recovar.__file__) + '/..', capture_output=True, text=True); print(f'    git commit: {result.stdout.strip()}')"

START_OLD=$(date +%s)

# Change to a neutral directory to avoid picking up local recovar/ directory
cd "$RESULTS_DIR"
recovar run_test_all_metrics \
    --volume-input "$VOLUME_DIR" \
    --output-dir "$OLD_DIR" \
    --n-images "$N_IMAGES" \
    --grid-size "$GRID_SIZE" \
    --tomo-tilts "$TOMO_TILTS" \
    --noise-level 1.0 \
    --contrast-std 0.1 \
    2>&1 | tee "$OLD_DIR/test_output.log"

END_OLD=$(date +%s)
OLD_TIME=$((END_OLD - START_OLD))

echo ""
echo "OLD version completed in $OLD_TIME seconds"

# Deactivate old environment
conda deactivate

echo ""
echo "=========================================="
echo "STEP 2: Run NEW version tests"
echo "=========================================="

echo "Activating NEW version environment: $NEW_ENV"
conda activate "$NEW_ENV"

# Verify environment is active
if [[ "$CONDA_DEFAULT_ENV" != "$NEW_ENV" ]]; then
    echo "Error: Failed to activate conda environment $NEW_ENV"
    exit 1
fi

echo "Running tests with NEW version..."
echo "  Environment: $NEW_ENV"
echo "  N_IMAGES=$N_IMAGES, GRID_SIZE=$GRID_SIZE, TOMO_TILTS=$TOMO_TILTS"

# Verify which recovar is being used
echo "  Verifying NEW version installation:"
python -c "import recovar; import os; print(f'    recovar path: {os.path.dirname(recovar.__file__)}'); import subprocess; result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=os.path.dirname(recovar.__file__) + '/..', capture_output=True, text=True); print(f'    git commit: {result.stdout.strip()}')"

START_NEW=$(date +%s)

# Change to a neutral directory to avoid picking up local recovar/ directory
cd "$RESULTS_DIR"
recovar run_test_all_metrics \
    --volume-input "$VOLUME_DIR" \
    --output-dir "$NEW_DIR" \
    --n-images "$N_IMAGES" \
    --grid-size "$GRID_SIZE" \
    --tomo-tilts "$TOMO_TILTS" \
    --noise-level 1.0 \
    --contrast-std 0.1 \
    2>&1 | tee "$NEW_DIR/test_output.log"

END_NEW=$(date +%s)
NEW_TIME=$((END_NEW - START_NEW))

echo ""
echo "NEW version completed in $NEW_TIME seconds"

echo ""
echo "=========================================="
echo "STEP 3: Compare results"
echo "=========================================="

# Run comparison script
python "$BASE_DIR/compare_test_results.py" "$RESULTS_DIR" "$OLD_TIME" "$NEW_TIME"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "TEST COMPLETE"
echo "Finished at: $(date)"
echo "=========================================="
echo ""
echo "Results saved to: $RESULTS_DIR"
echo "  - OLD version: $OLD_DIR"
echo "  - NEW version: $NEW_DIR"
echo "  - Comparison: $RESULTS_DIR/comparison_results.json"
echo ""

exit $EXIT_CODE

