#!/bin/bash
#SBATCH --job-name=recovar-parity
#SBATCH --account=amits
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=300G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/%j-%x.out

# Run recovar's RELION-mode refinement on a single benchmark dataset.
# Used by the 2026-04 RELION-parity stress matrix to launch one recovar
# run per dataset in parallel via slurm.
#
# Required env vars:
#   DATA_DIR    — input dataset directory (with particles.star, reference_init.mrc, etc.)
#   OUTPUT_DIR  — output directory for the run
# Optional:
#   MAX_ITER    — number of EM iterations (default 8)
#   HEALPIX_ORDER — coarse healpix order (default 2)
#   ADAPTIVE_OVERSAMPLING — 0 or 1 (default 0)

set -eo pipefail

export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="/scratch/gpfs/GILLES/mg6942/tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" /scratch/gpfs/GILLES/mg6942/slurmo

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
unset LD_LIBRARY_PATH || true

REPO_DIR="${REPO_DIR:-/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar}"
DATA_DIR="${DATA_DIR:?DATA_DIR required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR required}"
MAX_ITER="${MAX_ITER:-8}"
HEALPIX_ORDER="${HEALPIX_ORDER:-2}"
ADAPTIVE_OVERSAMPLING="${ADAPTIVE_OVERSAMPLING:-0}"
IMAGE_BATCH_SIZE="${IMAGE_BATCH_SIZE:-200}"

mkdir -p "$OUTPUT_DIR"

echo "=== Config ==="
echo "DATA_DIR=$DATA_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "MAX_ITER=$MAX_ITER"
echo "HEALPIX_ORDER=$HEALPIX_ORDER"
echo "ADAPTIVE_OVERSAMPLING=$ADAPTIVE_OVERSAMPLING"
test -f "$DATA_DIR/particles.star"

cd "$REPO_DIR"
pixi run python scripts/run_full_refinement.py \
  --data_dir "$DATA_DIR" \
  --output "$OUTPUT_DIR" \
  --mode relion --max_iter "$MAX_ITER" --healpix_order "$HEALPIX_ORDER" \
  --offset_range 3 --offset_step 1 --offset_sigma_angstrom 10.0 \
  --adaptive_oversampling "$ADAPTIVE_OVERSAMPLING" --adaptive_fraction 0.999 \
  --max_significants -1 --adaptive_skip_threshold 0.5 \
  --image_batch_size "$IMAGE_BATCH_SIZE"

echo "=== Done ==="
ls -la "$OUTPUT_DIR" | head -10
