#!/bin/bash
#SBATCH --job-name=recovar-10180
#SBATCH --account=amits
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/benchmark/examples/10180_%j.out

# RECOVAR Example: EMPIAR-10180 — Pre-catalytic Spliceosome (327,490 particles)
# Dataset: Plaschka et al. (2017), EMPIAR-10180
# Input: Pre-downsampled 256px particles; poses/CTF from recovar/assets/

set -euo pipefail

export PYTHONNOUSERSITE=1
PYTHON=/home/mg6942/.conda/envs/recovar_dev_2/bin/python
REPO=/scratch/gpfs/GILLES/mg6942/heterogeneity_dev-2
DATA=/scratch/gpfs/AMITS/mg6942/cryodrgn_empiar/empiar10180/inputs
OUT=/scratch/gpfs/GILLES/mg6942/benchmark/examples/10180

# Poses/CTF from recovar assets (extracted from original metadata)
POSES=$REPO/recovar/assets/poses_10180.pkl
CTF=$REPO/recovar/assets/ctf_10180.pkl

cd "$DATA"

echo "============================================================"
echo "  RECOVAR Example: EMPIAR-10180 — Pre-catalytic Spliceosome"
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
    --poses "$POSES" --ctf "$CTF" \
    --mask recovar_masks/mask_10180.mrc \
    --correct-contrast \
    --ind filtered.ind.pkl \
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
echo "  EMPIAR-10180 complete: $(date)"
echo "  Output: $OUT"
echo "============================================================"
