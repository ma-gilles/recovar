#!/bin/bash
#SBATCH --job-name=ppca-kc-256x100k
#SBATCH --account=amits
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/ppca-kc-256x100k-%j.out
#SBATCH --exclusive

# PPCA refinement vs k-class refinement quality + perf benchmark.
# Big dataset: 256³ × 100k images × 16-class CryoBench Ribosembly.
#
# Reuses the dataset prepared by run_cryobench_ribosembly_parity_slurm.sh
# (which the k-class ↔ RELION parity job depends on).
#
# Usage:
#   sbatch scripts/run_ppca_kclass_perf_benchmark.sh

set -euo pipefail

DATA_DIR="${DATA_DIR:-/scratch/gpfs/GILLES/mg6942/em_relion_proj/ribosembly_allk_g256_n100000_snr1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/gpfs/GILLES/mg6942/_agent_scratch/ppca_kclass_perf_2026_05_01}"
N_PCS="${N_PCS:-10}"
N_CLASSES="${N_CLASSES:-10}"
EM_ITERS="${EM_ITERS:-15}"
HEALPIX_ORDER="${HEALPIX_ORDER:-2}"
IMAGE_BATCH_SIZE="${IMAGE_BATCH_SIZE:-16}"
ROTATION_BLOCK_SIZE="${ROTATION_BLOCK_SIZE:-64}"

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_DIR"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="/scratch/gpfs/GILLES/mg6942/tmp/ppca_kc_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$OUTPUT_ROOT" /scratch/gpfs/GILLES/mg6942/slurmo

# Provenance gate.
pixi run python -c "
import pathlib, recovar, jax
repo = pathlib.Path.cwd().resolve()
assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo)+'/'), 'WRONG recovar'
assert '.pixi/envs/default/' in str(pathlib.Path(jax.__file__).resolve()), 'WRONG jax'
print('ENV_OK', jax.devices())
"

if [ ! -f "$DATA_DIR/particles.star" ]; then
  echo "ERROR: dataset not found at $DATA_DIR/particles.star"
  echo "Run scripts/run_cryobench_ribosembly_parity_slurm.sh first to prepare it."
  exit 1
fi

OUTPUT_DIR="$OUTPUT_ROOT/run_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

pixi run python scripts/run_ppca_kclass_perf_benchmark.py \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --n-images 100000 \
  --grid-size 256 \
  --n-pcs "$N_PCS" \
  --n-classes-kclass "$N_CLASSES" \
  --em-iters "$EM_ITERS" \
  --healpix-order "$HEALPIX_ORDER" \
  --image-batch-size "$IMAGE_BATCH_SIZE" \
  --rotation-block-size "$ROTATION_BLOCK_SIZE"

echo "=== DONE ==="
echo "Results: $OUTPUT_DIR/benchmark_summary.json"
