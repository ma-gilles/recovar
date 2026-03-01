#!/bin/bash
#SBATCH --job-name=recovar-10028
#SBATCH --account=amits
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/benchmark/examples/10028_%j.out

# RECOVAR Example: EMPIAR-10028 — Plasmodium 80S Ribosome (105,247 particles)
# Dataset: Wong et al. (2014), EMPIAR-10028
# Input: Pre-downsampled 256px particles with CTF pkl; poses extracted from
#        cryoSPARC .cs file as a preprocessing step.

set -euo pipefail

export PYTHONNOUSERSITE=1
PYTHON=/home/mg6942/.conda/envs/recovar_dev_2/bin/python
REPO=/scratch/gpfs/GILLES/mg6942/heterogeneity_dev-2
DATA=/scratch/gpfs/AMITS/mg6942/cryodrgn_empiar/empiar10028/inputs
OUT=/scratch/gpfs/GILLES/mg6942/benchmark/examples/10028

cd "$DATA"

echo "============================================================"
echo "  RECOVAR Example: EMPIAR-10028 — 80S Ribosome"
echo "============================================================"
echo "Commit: $(cd $REPO && git log --oneline -1)"
echo "Start:  $(date)"
echo "Host:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
$PYTHON -c "import jax; print('JAX backend:', jax.default_backend()); print('JAX devices:', jax.devices())"
echo "============================================================"
echo

# Step 0: Extract poses from cryoSPARC .cs file
# The .cs file has alignments3D/pose but the image paths don't resolve on this
# system. We extract poses to a .pkl and use the pre-downsampled .mrcs directly.
POSES_PKL="$OUT/poses_from_cs.pkl"
mkdir -p "$OUT"

echo ">>> Step 0: Extracting poses from cryoSPARC .cs file..."
$PYTHON -c "
import numpy as np
import pickle, sys
sys.path.insert(0, '$REPO')
from recovar.data_io.metadata_parsing import parse_poses_from_cs

cs_path = '$DATA/cryosparc_P11_J4_003_particles.cs'
D = 256  # box size of pre-downsampled particles
rots, trans = parse_poses_from_cs(cs_path, D)
print(f'Extracted {len(rots)} poses: rot {rots.shape}, trans {trans.shape}')

# Save in cryoDRGN format: (rotations, translations)
with open('$POSES_PKL', 'wb') as f:
    pickle.dump((rots, trans), f)
print(f'Saved to $POSES_PKL')
"
echo ">>> Pose extraction finished: $(date)"
echo

# Step 1: Pipeline — mean, covariance, eigenvolumes, embedding
echo ">>> Step 1: Running pipeline..."
$PYTHON -m recovar.command_line pipeline particles.256.mrcs \
    --poses "$POSES_PKL" --ctf ctf.pkl \
    --mask sphere \
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
echo "  EMPIAR-10028 complete: $(date)"
echo "  Output: $OUT"
echo "============================================================"
