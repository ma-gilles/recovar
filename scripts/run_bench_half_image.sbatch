#!/bin/bash
#SBATCH --job-name=bench-half
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --account=amits

set -euo pipefail

PYTHON=/home/mg6942/.conda/envs/recovar_dev_1/bin/python3

echo "Node: $(hostname)"
echo "Started at: $(date)"

export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd /scratch/gpfs/GILLES/mg6942/heterogeneity_dev-1

$PYTHON scripts/bench_half_image_jax.py

echo "Completed at: $(date)"
