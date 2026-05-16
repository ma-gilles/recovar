#!/usr/bin/env bash
#SBATCH --job-name=vdam-f10-gpu-bench
#SBATCH --account=amits
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200GB
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-f10-%j.out
#SBATCH --exclusive

# F10: GPU speed benchmark. Target ≤ 60 s per VDAM iter on H100
# for 500 particles, box 64, HEALPix order 1 (576 rots), 29 trans.

set -euo pipefail

cd /scratch/gpfs/GILLES/mg6942/recovar_dev/recovar

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="/scratch/gpfs/GILLES/mg6942/tmp/slurm_${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR"

pixi run python scripts/run_vdam_f10_benchmark.py --n-iter 5
