#!/bin/bash
#SBATCH --job-name=recovar-10180
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=24:00:00

# RECOVAR Example: EMPIAR-10180 — Pre-catalytic Spliceosome (327,490 particles)
#
# Workflow:
#   1. Pipeline with filtered particle index
#   2. Analyze → k-means clustering + UMAP
#   3. Estimate conformational density
#   4. Compute trajectory through density landscape
#
# Adapt these variables for your system:
#   PYTHON  — path to python with recovar installed
#   DATA    — directory containing particles, poses, CTF, and mask files
#   OUT     — where to write pipeline output

set -euo pipefail

# ── Configure these for your system ──────────────────────────────
PYTHON=/path/to/your/python          # Python with recovar installed
DATA=/path/to/particle/data          # Where your input files are
OUT=/path/to/output/10180            # Where to write results
# ─────────────────────────────────────────────────────────────────

cd "$DATA"

echo "============================================================"
echo "  RECOVAR Example: EMPIAR-10180 — Pre-catalytic Spliceosome"
echo "============================================================"
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
    --poses poses.pkl --ctf ctf.pkl \
    --mask mask.mrc \
    --focus-mask focus_mask.mrc \
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
