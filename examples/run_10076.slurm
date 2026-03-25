#!/bin/bash
#SBATCH --job-name=recovar-10076
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=24:00:00

# RECOVAR Example: EMPIAR-10076 — Bacterial 50S Ribosome (131,899 particles)
#
# Workflow:
#   1. Pipeline (no contrast correction)
#   2. Analyze → k-means clustering + UMAP
#   3. Extract subset: exclude outlier cluster, keep main assembly states
#   4. Re-run pipeline on the cleaned subset
#   5. Re-analyze the subset
#
# Adapt these variables for your system:
#   PYTHON  — path to python with recovar installed
#   DATA    — directory containing particles, poses, CTF, and mask files
#   OUT     — where to write pipeline output

set -euo pipefail

# ── Configure these for your system ──────────────────────────────
PYTHON=/path/to/your/python          # Python with recovar installed
DATA=/path/to/particle/data          # Where your input files are
OUT=/path/to/output/10076            # Where to write results
OUT_SUBSET=${OUT}_subset
# ─────────────────────────────────────────────────────────────────

cd "$DATA"

echo "============================================================"
echo "  RECOVAR Example: EMPIAR-10076 — 50S Ribosome"
echo "============================================================"
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
    --mask mask.mrc \
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
    "$OUT/analysis_20/data/kmeans_result.pkl" \
    "$OUT_SUBSET/subset_ind" \
    0 --inverse
echo ">>> Subset extraction finished: $(date)"
echo

# ──────────────────────────────────────────────────────────────────
# Step 4: Re-run pipeline on the cleaned subset
# ──────────────────────────────────────────────────────────────────
echo ">>> Step 4: Re-running pipeline on subset..."
$PYTHON -m recovar.command_line pipeline particles.256.mrcs \
    --poses poses.pkl --ctf ctf.pkl \
    --mask mask.mrc \
    --ind "$OUT_SUBSET/subset_ind/indices.pkl" \
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
