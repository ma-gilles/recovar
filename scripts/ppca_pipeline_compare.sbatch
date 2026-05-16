#!/bin/bash
#SBATCH --job-name=ppca-pipe-compare
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=12:00:00
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

REPO_ROOT="${REPO_ROOT:-/scratch/gpfs/GILLES/mg6942/recovar_wt_agent_precond_study_20260330}"
cd "$REPO_ROOT"

PYTHON="$REPO_ROOT/.pixi/envs/default/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo "Missing pixi python at $PYTHON" >&2
  exit 1
fi

"$PYTHON" -c "import pathlib,recovar,jax; repo=pathlib.Path('$REPO_ROOT').resolve(); assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo)+'/'); assert '.pixi/envs/default/' in str(pathlib.Path(jax.__file__).resolve())"

COMMON_ARGS=(
  "$PYTHON" -m recovar.ppca.compare_covariance_vs_ppca_pipeline
  --base-dir "${BASE_DIR:?BASE_DIR not set}"
  --results-root "${RESULTS_ROOT:?RESULTS_ROOT not set}"
  --dataset "${DATASET:?DATASET not set}"
  --grid-size "${GRID_SIZE:-128}"
  --n-images "${N_IMAGES:-100000}"
  --noise-level "${NOISE_LEVEL:-1.0}"
  --contrast-std "${CONTRAST_STD:-0.0}"
  --zdim "${ZDIM:-10}"
  --ppca-em-iters "${PPCA_EM_ITERS:-20}"
  --seed "${SEED:-42}"
  --gpu-gb "${GPU_GB:-40}"
  --covariance-gpu-gb "${COVARIANCE_GPU_GB:-${GPU_GB:-40}}"
  --ppca-gpu-gb "${PPCA_GPU_GB:-${GPU_GB:-40}}"
  --lazy
)

if [ "${COVARIANCE_LOW_MEMORY_OPTION:-0}" = "1" ]; then
  COMMON_ARGS+=(--covariance-low-memory-option)
fi

if [ "${COVARIANCE_VERY_LOW_MEMORY_OPTION:-0}" = "1" ]; then
  COMMON_ARGS+=(--covariance-very-low-memory-option)
fi

if [ "${PPCA_LOW_MEMORY_OPTION:-0}" = "1" ]; then
  COMMON_ARGS+=(--ppca-low-memory-option)
fi

if [ "${PPCA_VERY_LOW_MEMORY_OPTION:-0}" = "1" ]; then
  COMMON_ARGS+=(--ppca-very-low-memory-option)
fi

if [ "${FORCE:-0}" = "1" ]; then
  COMMON_ARGS+=(--force)
fi

PREP_CMD=("${COMMON_ARGS[@]}" --stage prepare --print-run-root)
echo "Preparing compare run: ${PREP_CMD[*]}"
RUN_ROOT="$("${PREP_CMD[@]}" | tail -n 1)"

if [ ! -d "$RUN_ROOT" ]; then
  echo "Prepare stage did not produce a run root: $RUN_ROOT" >&2
  exit 1
fi

RUNNER="$RUN_ROOT/run_methods.sh"
if [ ! -x "$RUNNER" ]; then
  echo "Missing runner script at $RUNNER" >&2
  exit 1
fi

echo "Running methods directly via shell: $RUNNER"
bash "$RUNNER"

SCORE_CMD=("${COMMON_ARGS[@]}" --stage score)
echo "Scoring completed outputs: ${SCORE_CMD[*]}"
"${SCORE_CMD[@]}"
