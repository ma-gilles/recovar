#!/bin/bash
#SBATCH --job-name=ppca-default-bench
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/%x-%A_%a.out

set -euo pipefail

export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
PPCA_BENCH_SCRATCH_ROOT="${PPCA_BENCH_SCRATCH_ROOT:-/projects/CRYOEM/singerlab/mg6942/ppca_default_benchmark_scratch}"
export TMPDIR="${PPCA_BENCH_SCRATCH_ROOT}/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
export PIXI_HOME="${PPCA_BENCH_SCRATCH_ROOT}/pixi_home/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
export RATTLER_CACHE_DIR="${PPCA_BENCH_SCRATCH_ROOT}/rattler_cache/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR" "$PIXI_HOME" "$RATTLER_CACHE_DIR"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV

REPO_ROOT="${REPO_ROOT:?REPO_ROOT not set}"
MANIFEST="${MANIFEST:?MANIFEST not set}"
GPU_GB="${GPU_GB:-70}"

cd "$REPO_ROOT"
PYTHON="$REPO_ROOT/.pixi/envs/default/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo "Missing pixi python at $PYTHON" >&2
  exit 1
fi

git rev-parse HEAD
git symbolic-ref --short HEAD || true
git status --porcelain

"$PYTHON" -c "import pathlib,recovar,jax; repo=pathlib.Path('$REPO_ROOT').resolve(); assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo)+'/'); assert '.pixi/envs/default/' in str(pathlib.Path(jax.__file__).resolve()); print(jax.devices())"

"$PYTHON" scripts/ppca_default_benchmark.py run-case \
  --manifest "$MANIFEST" \
  --case-index "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}" \
  --gpu-gb "$GPU_GB"
