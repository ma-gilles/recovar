#!/bin/bash
#SBATCH --job-name=recovar-10076
#SBATCH --account=amits
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/benchmark/examples/10076_%j.out

# RECOVAR Example: EMPIAR-10076 — Bacterial 50S Ribosome (131,899 particles)
# Dataset: Wong et al. (2014), EMPIAR-10076
# Input: Pre-downsampled 256px particles with poses and CTF from cryoDRGN

set -euo pipefail

export PYTHONNOUSERSITE=1
PYTHON=/home/mg6942/.conda/envs/recovar_dev_2/bin/python
REPO=/scratch/gpfs/GILLES/mg6942/heterogeneity_dev-2
DATA=/scratch/gpfs/AMITS/mg6942/cryodrgn_empiar/empiar10076/inputs
OUT=/scratch/gpfs/GILLES/mg6942/benchmark/examples/10076

cd "$DATA"

echo "============================================================"
echo "  RECOVAR Example: EMPIAR-10076 — 50S Ribosome"
echo "============================================================"
echo "Commit: $(cd $REPO && git log --oneline -1)"
echo "Start:  $(date)"
echo "Host:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
$PYTHON -c "import jax; print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"
echo "============================================================"
echo

# Step 1: Pipeline — mean, covariance, eigenvolumes, embedding
echo ">>> Step 1: Running pipeline..."
rm -rf "$OUT"
$PYTHON -m recovar.command_line pipeline particles.256.mrcs \
    --poses poses.pkl --ctf ctf.pkl \
    --mask mask_from_filtered_variance_mean_0802.mrc \
    --correct-contrast \
    -o "$OUT"
echo ">>> Pipeline finished: $(date)"
echo

# Step 2: Analyze with zdim=10
echo ">>> Step 2: Analyzing with zdim=10..."
$PYTHON -m recovar.command_line analyze "$OUT" \
    --zdim=10 --n-clusters=20 --n-trajectories=2
echo ">>> Analyze zdim=10 finished: $(date)"
echo

# Step 3: Analyze with zdim=20
echo ">>> Step 3: Analyzing with zdim=20..."
$PYTHON -m recovar.command_line analyze "$OUT" \
    --zdim=20 --n-clusters=20 --n-trajectories=2
echo ">>> Analyze zdim=20 finished: $(date)"
echo

echo "============================================================"
echo "  EMPIAR-10076 complete: $(date)"
echo "  Output: $OUT"
echo "============================================================"
