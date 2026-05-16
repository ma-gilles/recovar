#!/bin/bash
#SBATCH --job-name=ppca-refit
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/%x-%j.out

set -euo pipefail

export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
export TMPDIR="/scratch/gpfs/GILLES/mg6942/tmp/${SLURM_JOB_ID}"
export PIXI_HOME="/scratch/gpfs/GILLES/mg6942/pixi_home/${SLURM_JOB_ID}"
export RATTLER_CACHE_DIR="/scratch/gpfs/GILLES/mg6942/rattler_cache/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$PIXI_HOME" "$RATTLER_CACHE_DIR"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV

REPO_ROOT="${REPO_ROOT:-/scratch/gpfs/GILLES/mg6942/recovar_wt_ppca_refit}"
cd "$REPO_ROOT"

PYTHON="$REPO_ROOT/.pixi/envs/default/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo "Missing pixi python at $PYTHON" >&2
  exit 1
fi

# Provenance gate
"$PYTHON" -c "
import pathlib, recovar, jax
repo = pathlib.Path('$REPO_ROOT').resolve()
assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo)+'/'), 'WRONG recovar'
assert '.pixi/envs/default/' in str(pathlib.Path(jax.__file__).resolve()), 'WRONG jax'
print('ENV_OK')
"

# Required env vars
PPCA_RESULT_DIR="${PPCA_RESULT_DIR:?PPCA_RESULT_DIR not set}"
METHOD="${METHOD:?METHOD not set}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR not set}"
ZDIM="${ZDIM:-10}"
BATCH_SIZE="${BATCH_SIZE:-128}"
N_ITERS="${N_ITERS:-20}"

echo "=== PPCA Refit ==="
echo "  PPCA_RESULT_DIR=$PPCA_RESULT_DIR"
echo "  METHOD=$METHOD"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  ZDIM=$ZDIM"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo

CMD=(
  "$PYTHON" -m recovar.ppca.refit_postprocess
  --ppca-result-dir "$PPCA_RESULT_DIR"
  --method "$METHOD"
  --output-dir "$OUTPUT_DIR"
  --zdim "$ZDIM"
  --batch-size "$BATCH_SIZE"
  --n-iters "$N_ITERS"
)

echo "Running: ${CMD[*]}"
"${CMD[@]}"
echo "Done: method=$METHOD output=$OUTPUT_DIR"
