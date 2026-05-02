#!/bin/bash
#SBATCH --job-name=autorefine-kN
#SBATCH --account=amits
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/autorefine-kN-%j.out
#SBATCH --exclusive

# k=N multi-class auto-refine quality test on 256³ × 100k Ribosembly with
# the 16-class GT volume bank. Drives the dense k-class engine through a
# RELION-style HEALPix-order ramp (init order → max order, advancing on
# convergence) and scores final per-class volumes against GT via
# Hungarian-matched FSC.

set -euo pipefail

DATA_DIR="${DATA_DIR:-/scratch/gpfs/GILLES/mg6942/em_relion_proj/ribosembly_allk_g256_n100000_snr1_cubic}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/gpfs/GILLES/mg6942/_agent_scratch/autorefine_kN_2026_05_02}"
N_CLASSES="${N_CLASSES:-4}"
N_IMAGES="${N_IMAGES:-100000}"
GRID_SIZE="${GRID_SIZE:-}"           # empty = native (256)
MAX_ITERS="${MAX_ITERS:-25}"
HEALPIX_ORDER_INIT="${HEALPIX_ORDER_INIT:-1}"
HEALPIX_ORDER_MAX="${HEALPIX_ORDER_MAX:-3}"
PMAX_ADVANCE_THRESHOLD="${PMAX_ADVANCE_THRESHOLD:-0.85}"
MIN_ITERS_PER_ORDER="${MIN_ITERS_PER_ORDER:-2}"
IMAGE_BATCH_SIZE="${IMAGE_BATCH_SIZE:-16}"
ROTATION_BLOCK_SIZE="${ROTATION_BLOCK_SIZE:-128}"

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_DIR"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="/scratch/gpfs/GILLES/mg6942/tmp/autorefine_kN_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$OUTPUT_ROOT" /scratch/gpfs/GILLES/mg6942/slurmo

pixi run python -c "
import pathlib, recovar, jax
repo = pathlib.Path.cwd().resolve()
assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo)+'/'), 'WRONG recovar'
assert '.pixi/envs/default/' in str(pathlib.Path(jax.__file__).resolve()), 'WRONG jax'
print('ENV_OK', jax.devices())
"

OUTPUT_DIR="$OUTPUT_ROOT/run_${SLURM_JOB_ID}_K${N_CLASSES}"
mkdir -p "$OUTPUT_DIR"

GS_FLAG=""
if [ -n "$GRID_SIZE" ]; then
  GS_FLAG="--grid-size $GRID_SIZE"
fi

pixi run python scripts/run_kclass_autorefine_quality.py \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --n-classes "$N_CLASSES" \
  --n-images "$N_IMAGES" \
  --max-iters "$MAX_ITERS" \
  --healpix-order-init "$HEALPIX_ORDER_INIT" \
  --healpix-order-max "$HEALPIX_ORDER_MAX" \
  --pmax-advance-threshold "$PMAX_ADVANCE_THRESHOLD" \
  --min-iters-per-order "$MIN_ITERS_PER_ORDER" \
  --image-batch-size "$IMAGE_BATCH_SIZE" \
  --rotation-block-size "$ROTATION_BLOCK_SIZE" \
  $GS_FLAG

echo "=== Comparison vs RELION Class3D auto-refine ==="
RELION_REF_DIR="${RELION_REF_DIR:-$DATA_DIR/relion_class3d_k${N_CLASSES}_autorefine}"
if [ -d "$RELION_REF_DIR" ] && [ -n "$(ls -A "$RELION_REF_DIR" 2>/dev/null)" ]; then
  pixi run python scripts/compare_recovar_relion_autorefine.py \
    --mode kN \
    --recovar-dir "$OUTPUT_DIR" \
    --relion-dir "$RELION_REF_DIR" \
    --gt-dir "$DATA_DIR" \
    --output "$OUTPUT_DIR/comparison.json"
else
  echo "Skipping comparison: RELION class3d ref missing at $RELION_REF_DIR"
  echo "Run: sbatch scripts/run_relion_kN_class3d_autorefine.sh"
fi

echo "=== DONE ==="
echo "Results: $OUTPUT_DIR/summary.json"
