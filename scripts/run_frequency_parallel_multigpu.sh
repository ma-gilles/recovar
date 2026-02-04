#!/bin/bash
# Frequency-parallel multi-node computation with multi-GPU within nodes
# This script is executed inside Docker containers on each node via crun/SLURM
# 
# Called by: crun_recovar_workload.sh freq-parallel-2node

set -e

# Get SLURM context (set by crun/srun)
NODE_RANK=${SLURM_PROCID:-0}
N_NODES=${SLURM_NTASKS:-1}
JOB_ID=${SLURM_JOB_ID:-unknown}

echo "=========================================="
echo "Frequency-Parallel Node Computation"
echo "Node: ${NODE_RANK}/${N_NODES}"
echo "Job ID: ${JOB_ID}"
echo "=========================================="

# Dataset configuration
DATASET_DIR=${DATASET_DIR:-"/workspace/data-128-100000/test_dataset"}
IMAGE_SIZE=128
N_IMAGES=${N_IMAGES:-100000}
N_GPUS=${N_GPUS:-2}

cd "$DATASET_DIR"

# Output directory (shared across nodes)
OUTPUT_DIR=${OUTPUT_DIR:-"pipeline_output_${N_NODES}node_${N_GPUS}gpu_$(date +%Y%m%d_%H%M%S)"}
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"

# Build the pipeline command with container-compatible paths
PIPELINE_CMD="pixi run python -m recovar.commands.pipeline \
    /workspace/${DATASET_DIR}/particles.${IMAGE_SIZE}.mrcs \
    --ctf /workspace/${DATASET_DIR}/ctf.pkl \
    --poses /workspace/${DATASET_DIR}/poses.pkl \
    --mask=from_halfmaps \
    -o /workspace/${OUTPUT_DIR} \
    --n-images $N_IMAGES \
    --frequency-parallel \
    --multi-gpu \
    --n-gpus ${N_GPUS} \
    --lazy"

echo "Running: $PIPELINE_CMD"
echo ""

$PIPELINE_CMD

echo "=========================================="
echo "Node ${NODE_RANK} computation complete!"
echo "=========================================="
