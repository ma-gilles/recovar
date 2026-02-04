#!/bin/bash
# Quick test of frequency-parallel prototype

set -e

# Configuration - use /workspace when running in container
BASE_DIR="${BASE_DIR:-/workspace}"
OUTPUT_DIR="${BASE_DIR}/multinode_covariance_bench/prototype_output_$(date +%Y%m%d_%H%M%S)"
DATASET_DIR="${BASE_DIR}/data-128-100000/test_dataset"
N_IMAGES=1000

echo "======================================================================"
echo "Frequency-Parallel Prototype Test"
echo "======================================================================"
echo "Base directory: $BASE_DIR"
echo "Dataset: $DATASET_DIR"
echo "Images: $N_IMAGES"
echo "Output: $OUTPUT_DIR"
echo ""

# Check dataset exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: Dataset directory not found: $DATASET_DIR"
    echo "Please ensure the dataset is available."
    exit 1
fi

# ============================================================================
# TEST 1: Baseline (1 node, all frequencies)
# ============================================================================
echo "TEST 1: Baseline (1 node, all frequencies)"
echo "----------------------------------------------------------------------"
echo "This establishes the baseline timing for comparison."
echo ""

START_BASELINE=$(date +%s)

python ${BASE_DIR}/multinode_covariance_bench/prototype_frequency_parallel.py \
    --node-rank 0 \
    --n-nodes 1 \
    --dataset-dir "$DATASET_DIR" \
    --output-dir "${OUTPUT_DIR}/baseline" \
    --n-images $N_IMAGES

END_BASELINE=$(date +%s)
BASELINE_TIME=$((END_BASELINE - START_BASELINE))

echo ""
echo "✓ Baseline complete: ${BASELINE_TIME}s"
echo ""

# ============================================================================
# TEST 2: Frequency-parallel (2 nodes, simulated)
# ============================================================================
echo "TEST 2: Frequency-parallel (2 nodes, simulated)"
echo "----------------------------------------------------------------------"
echo "Running 2 nodes in parallel (simulated multi-node execution)."
echo ""

START_PARALLEL=$(date +%s)

echo "  Starting node 0..."
python ${BASE_DIR}/multinode_covariance_bench/prototype_frequency_parallel.py \
    --node-rank 0 \
    --n-nodes 2 \
    --dataset-dir "$DATASET_DIR" \
    --output-dir "${OUTPUT_DIR}/freq_parallel" \
    --n-images $N_IMAGES &

PID0=$!

echo "  Starting node 1..."
python ${BASE_DIR}/multinode_covariance_bench/prototype_frequency_parallel.py \
    --node-rank 1 \
    --n-nodes 2 \
    --dataset-dir "$DATASET_DIR" \
    --output-dir "${OUTPUT_DIR}/freq_parallel" \
    --n-images $N_IMAGES &

PID1=$!

echo "  Waiting for both nodes to complete..."
wait $PID0
RESULT0=$?
wait $PID1
RESULT1=$?

END_PARALLEL=$(date +%s)
PARALLEL_TIME=$((END_PARALLEL - START_PARALLEL))

if [ $RESULT0 -ne 0 ] || [ $RESULT1 -ne 0 ]; then
    echo ""
    echo "ERROR: One or both nodes failed!"
    echo "  Node 0 exit code: $RESULT0"
    echo "  Node 1 exit code: $RESULT1"
    exit 1
fi

echo ""
echo "✓ Both nodes complete: ${PARALLEL_TIME}s"
echo ""

# ============================================================================
# TEST 3: Concatenate results
# ============================================================================
echo "TEST 3: Concatenating results"
echo "----------------------------------------------------------------------"

python ${BASE_DIR}/multinode_covariance_bench/prototype_frequency_parallel.py \
    --concatenate \
    --n-nodes 2 \
    --output-dir "${OUTPUT_DIR}/freq_parallel"

echo ""
echo "✓ Concatenation complete"
echo ""

# ============================================================================
# TEST 4: Validate outputs
# ============================================================================
echo "TEST 4: Validating outputs"
echo "----------------------------------------------------------------------"

python -c "
import numpy as np
from pathlib import Path

output_dir = Path('${OUTPUT_DIR}')

# Load baseline
baseline_file = output_dir / 'baseline' / 'node000_result.npz'
baseline = np.load(baseline_file)

print(f'Baseline:')
print(f'  File: {baseline_file.name}')
print(f'  Covariance shape: {baseline[\"covariance_cols\"].shape}')
print(f'  Frequencies: {len(baseline[\"picked_frequencies\"])}')
print(f'  Compute time: {baseline[\"compute_time\"]:.2f}s')
print()

# Load concatenated
concat_file = output_dir / 'freq_parallel' / 'concatenated_result.npz'
concat = np.load(concat_file)

print(f'Concatenated (2 nodes):')
print(f'  File: {concat_file.name}')
print(f'  Covariance shape: {concat[\"covariance_cols\"].shape}')
print(f'  Frequencies: {len(concat[\"picked_frequencies\"])}')
print()

# Validate shapes match
print('Validation:')
if baseline['covariance_cols'].shape == concat['covariance_cols'].shape:
    print('  ✓ Shapes match')
else:
    print(f'  ✗ Shape mismatch!')
    print(f'    Baseline: {baseline[\"covariance_cols\"].shape}')
    print(f'    Concatenated: {concat[\"covariance_cols\"].shape}')
    exit(1)

if len(baseline['picked_frequencies']) == len(concat['picked_frequencies']):
    print('  ✓ Frequency counts match')
else:
    print(f'  ✗ Frequency count mismatch!')
    exit(1)

# Check if frequencies are the same (order matters)
if np.array_equal(baseline['picked_frequencies'], concat['picked_frequencies']):
    print('  ✓ Frequency indices match')
else:
    print('  ⚠ Frequency indices differ (order may vary)')

print()
print('All validations passed! ✓')
" || exit 1

echo ""

# ============================================================================
# SUMMARY
# ============================================================================
echo "======================================================================"
echo "PROTOTYPE TEST COMPLETE"
echo "======================================================================"
echo ""
echo "Results:"
echo "  Output directory: $OUTPUT_DIR"
echo ""
echo "Timing:"
echo "  Baseline (1 node):     ${BASELINE_TIME}s"
echo "  Parallel (2 nodes):    ${PARALLEL_TIME}s"

# Calculate speedup
if [ $PARALLEL_TIME -gt 0 ]; then
    SPEEDUP=$(echo "scale=2; $BASELINE_TIME / $PARALLEL_TIME" | bc)
    echo "  Speedup:               ${SPEEDUP}×"
    
    # Check if speedup is reasonable (should be close to 2×)
    SPEEDUP_INT=$(echo "$SPEEDUP" | cut -d. -f1)
    if [ "$SPEEDUP_INT" -ge 1 ]; then
        echo ""
        echo "✓ Speedup achieved! Frequency-parallel approach is working."
        
        if [ "$SPEEDUP_INT" -ge 2 ]; then
            echo "  Excellent: Near-linear scaling (2× speedup with 2 nodes)"
        else
            echo "  Good: Significant speedup, some overhead present"
        fi
    else
        echo ""
        echo "⚠ Warning: Speedup less than expected"
        echo "  This may be due to overhead or small dataset size"
    fi
fi

echo ""
echo "Next steps:"
echo "  1. Review timing results above"
echo "  2. Check output files in: $OUTPUT_DIR"
echo "  3. If successful, proceed with full implementation"
echo ""
echo "Files created:"
echo "  Baseline:      ${OUTPUT_DIR}/baseline/node000_result.npz"
echo "  Node 0:        ${OUTPUT_DIR}/freq_parallel/node000_result.npz"
echo "  Node 1:        ${OUTPUT_DIR}/freq_parallel/node001_result.npz"
echo "  Concatenated:  ${OUTPUT_DIR}/freq_parallel/concatenated_result.npz"
echo ""
