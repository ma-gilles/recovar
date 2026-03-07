#!/bin/bash
#SBATCH --job-name=test-full
#SBATCH --partition=cryoem
#SBATCH --account=amits
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/%j-%x.out
#SBATCH --exclusive

export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=/scratch/gpfs/GILLES/mg6942/tmp
export LONG_METRICS_OUTPUT_BASE=/scratch/gpfs/GILLES/mg6942/recovar_test_outputs
mkdir -p "$TMPDIR"
mkdir -p "$LONG_METRICS_OUTPUT_BASE"

cd /scratch/gpfs/GILLES/mg6942/heterogeneity_dev

pixi run test-full
