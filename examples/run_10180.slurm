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
#
# Workflow:
#   1. Pipeline with filtered particle index
#   2. Analyze → k-means clustering + UMAP
#   3. Estimate conformational density
#   4. Compute trajectory through density landscape

set -euo pipefail

export PYTHONNOUSERSITE=1
PYTHON=/home/mg6942/.conda/envs/recovar_dev_2/bin/python
REPO=/scratch/gpfs/GILLES/mg6942/heterogeneity_dev-2
DATA=/scratch/gpfs/AMITS/mg6942/cryodrgn_empiar/empiar10180/inputs
OUT=/scratch/gpfs/GILLES/mg6942/benchmark/examples/10180

# Poses/CTF from recovar assets
POSES=$REPO/recovar/assets/poses_10180.pkl
CTF=$REPO/recovar/assets/ctf_10180.pkl

# Focus mask from paper run
FOCUS_MASK=$DATA/paper_run/output/volumes/focus_mask.mrc

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

# ──────────────────────────────────────────────────────────────────
# Step 1: Pipeline with filtered particle index
# ──────────────────────────────────────────────────────────────────
echo ">>> Step 1: Running pipeline..."
rm -rf "$OUT"
$PYTHON -m recovar.command_line pipeline particles.256.mrcs \
    --poses "$POSES" --ctf "$CTF" \
    --mask recovar_masks/mask_10180.mrc \
    --focus-mask "$FOCUS_MASK" \
    --correct-contrast \
    --ind filtered.ind.pkl \
    -o "$OUT"
echo ">>> Pipeline finished: $(date)"
echo

# ──────────────────────────────────────────────────────────────────
# Step 2: Analyze with zdim=4
# ──────────────────────────────────────────────────────────────────
echo ">>> Step 2: Analyzing with zdim=4..."
$PYTHON -m recovar.command_line analyze "$OUT" \
    --zdim=4 --n-clusters=20 --n-trajectories=2
echo ">>> Analyze zdim=4 finished: $(date)"
echo

# ──────────────────────────────────────────────────────────────────
# Step 3: Estimate conformational density
# ──────────────────────────────────────────────────────────────────
echo ">>> Step 3: Estimating conformational density (pca_dim=4)..."
$PYTHON -m recovar.command_line estimate_conformational_density "$OUT" \
    --pca_dim=4 --z_dim_used=4
echo ">>> Density estimation finished: $(date)"
echo

# ──────────────────────────────────────────────────────────────────
# Step 4: Compute trajectory through density landscape
# ──────────────────────────────────────────────────────────────────
echo ">>> Step 4: Computing trajectory..."
# Use the deconvolved density at the "knee" regularization level
# and endpoints from the two most distant k-means centers (0 and last)
$PYTHON -m recovar.command_line compute_trajectory "$OUT" \
    -o "$OUT/trajectory_density" \
    --zdim=4 \
    --density "$OUT/density/deconv_density_knee.pkl" \
    --endpts "$OUT/analysis_4/kmeans/centers.txt" \
    --ind 0,19
echo ">>> Trajectory computation finished: $(date)"
echo

echo "============================================================"
echo "  EMPIAR-10180 complete: $(date)"
echo "  Output: $OUT"
echo "============================================================"
