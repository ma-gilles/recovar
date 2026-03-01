#!/bin/bash
#SBATCH --job-name=recovar-10076
#SBATCH --account=amits
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/benchmark/examples/10076_%j.out

# RECOVAR Example: EMPIAR-10076 — Bacterial 50S Ribosome (131,899 particles)
#
# Workflow:
#   1. Pipeline (no contrast correction)
#   2. Analyze → k-means clustering + UMAP
#   3. Extract subset: exclude outlier cluster, keep main assembly states
#   4. Re-run pipeline on the cleaned subset
#   5. Re-analyze the subset

set -euo pipefail

export PYTHONNOUSERSITE=1
PYTHON=/home/mg6942/.conda/envs/recovar_dev_2/bin/python
REPO=/scratch/gpfs/GILLES/mg6942/heterogeneity_dev-2
DATA=/scratch/gpfs/AMITS/mg6942/cryodrgn_empiar/empiar10076/inputs
OUT=/scratch/gpfs/GILLES/mg6942/benchmark/examples/10076
OUT_SUBSET=/scratch/gpfs/GILLES/mg6942/benchmark/examples/10076_subset

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

# ──────────────────────────────────────────────────────────────────
# Step 1: Pipeline (no --correct-contrast)
# ──────────────────────────────────────────────────────────────────
echo ">>> Step 1: Running pipeline (no contrast correction)..."
rm -rf "$OUT"
$PYTHON -m recovar.command_line pipeline particles.256.mrcs \
    --poses poses.pkl --ctf ctf.pkl \
    --mask mask_from_filtered_variance_mean_0802.mrc \
    -o "$OUT"
echo ">>> Pipeline finished: $(date)"
echo

# ──────────────────────────────────────────────────────────────────
# Step 2: Analyze with zdim=20
# ──────────────────────────────────────────────────────────────────
echo ">>> Step 2: Analyzing with zdim=20..."
$PYTHON -m recovar.command_line analyze "$OUT" \
    --zdim=20 --n-clusters=20 --n-trajectories=2
echo ">>> Analyze zdim=20 finished: $(date)"
echo

# ──────────────────────────────────────────────────────────────────
# Step 3: Extract subset — exclude outlier cluster 0
# ──────────────────────────────────────────────────────────────────
echo ">>> Step 3: Extracting subset (excluding outlier cluster 0)..."
mkdir -p "$OUT_SUBSET"
$PYTHON -m recovar.command_line extract_image_subset_from_kmeans \
    "$OUT/analysis_20/kmeans_result.pkl" \
    "$OUT_SUBSET/subset_ind.pkl" \
    0 --inverse
echo ">>> Subset extraction finished: $(date)"
echo

# ──────────────────────────────────────────────────────────────────
# Step 4: Re-run pipeline on the cleaned subset
# ──────────────────────────────────────────────────────────────────
echo ">>> Step 4: Re-running pipeline on subset..."
$PYTHON -m recovar.command_line pipeline particles.256.mrcs \
    --poses poses.pkl --ctf ctf.pkl \
    --mask mask_from_filtered_variance_mean_0802.mrc \
    --ind "$OUT_SUBSET/subset_ind.pkl" \
    -o "$OUT_SUBSET"
echo ">>> Subset pipeline finished: $(date)"
echo

# ──────────────────────────────────────────────────────────────────
# Step 5: Analyze the subset
# ──────────────────────────────────────────────────────────────────
echo ">>> Step 5: Analyzing subset with zdim=20..."
$PYTHON -m recovar.command_line analyze "$OUT_SUBSET" \
    --zdim=20 --n-clusters=20 --n-trajectories=2
echo ">>> Subset analyze finished: $(date)"
echo

echo "============================================================"
echo "  EMPIAR-10076 complete: $(date)"
echo "  Full output:   $OUT"
echo "  Subset output: $OUT_SUBSET"
echo "============================================================"
