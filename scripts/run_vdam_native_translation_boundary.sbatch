#!/usr/bin/env bash
#SBATCH --job-name=vdam-translate
#SBATCH --account=amits
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-translate-%j.out

set -euo pipefail

: "${NATIVE_DIRECTORY:?set NATIVE_DIRECTORY=/scratch/...}"
: "${LIVE_SCORE:?set LIVE_SCORE=/scratch/...npz}"
: "${OUTPUT_JSON:?set OUTPUT_JSON=/scratch/...json}"

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
RUN_ID="${SLURM_JOB_ID:-manual}"
cd "${REPO_ROOT}"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV JAX_PLATFORM_NAME
export PYTHONNOUSERSITE=1
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/${RUN_ID}/tmp"
export PIXI_HOME="/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/${RUN_ID}/pixi_home"
export RATTLER_CACHE_DIR="/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/${RUN_ID}/rattler_cache"
export RECOVAR_CUDA_LIB="${RECOVAR_CUDA_LIB:-${REPO_ROOT}/recovar/cuda/libcuda_backproject.so}"
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" /scratch/gpfs/GILLES/mg6942/slurmo

PIXI_PY="${REPO_ROOT}/.pixi/envs/default/bin/python"
if [[ "${BUILD_CUSTOM_CUDA:-0}" == "1" ]]; then
  MAKE_ARGS=(LIB="${RECOVAR_CUDA_LIB}")
  if [[ "${BUILD_H100_ONLY:-0}" == "1" ]]; then
    MAKE_ARGS+=(CUDA_ARCH="-gencode arch=compute_90,code=sm_90")
  fi
  if [[ "${FORCE_CUSTOM_CUDA_REBUILD:-0}" == "1" ]]; then
    MAKE_ARGS=(-B "${MAKE_ARGS[@]}")
  fi
  env PYTHON="${PIXI_PY}" NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}" \
    make -C recovar/cuda "${MAKE_ARGS[@]}" all
fi
"${PIXI_PY}" - <<'PY'
import pathlib

import jax
import recovar
from recovar import cuda_backproject

repo = pathlib.Path.cwd().resolve()
assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo) + "/")
assert ".pixi/envs/default/" in str(pathlib.Path(jax.__file__).resolve())
assert jax.default_backend() == "gpu"
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
print("VDAM native translation provenance/GPU/CUDA-FFI gate passed", jax.devices())
PY

ANALYZER_ARGS=(
  --native-directory "${NATIVE_DIRECTORY}"
  --live-score "${LIVE_SCORE}"
  --full-image-size "${FULL_IMAGE_SIZE:-128}"
  --output-json "${OUTPUT_JSON}"
)
if [[ -n "${PARTICLE_STACK:-}" ]]; then
  ANALYZER_ARGS+=(
    --particle-stack "${PARTICLE_STACK}"
    --particle-index "${PARTICLE_INDEX:?set PARTICLE_INDEX with PARTICLE_STACK}"
    --normalization-factor "${NORMALIZATION_FACTOR:-1.0}"
    --integer-pre-shift "${INTEGER_PRE_SHIFT_X:-0}" "${INTEGER_PRE_SHIFT_Y:-0}"
    --mask-radius "${MASK_RADIUS:?set MASK_RADIUS with PARTICLE_STACK}"
    --mask-cosine-width "${MASK_COSINE_WIDTH:-5.0}"
  )
fi
if [[ -n "${NATIVE_PREPROCESS_DIRECTORY:-}" ]]; then
  ANALYZER_ARGS+=(--native-preprocess-directory "${NATIVE_PREPROCESS_DIRECTORY}")
fi
if [[ -n "${NATIVE_FINE_SCORE:-}" ]]; then
  ANALYZER_ARGS+=(--native-fine-score "${NATIVE_FINE_SCORE}")
fi
if [[ -n "${NATIVE_FINE_OPERAND:-}" ]]; then
  ANALYZER_ARGS+=(--native-fine-operand "${NATIVE_FINE_OPERAND}")
fi
"${PIXI_PY}" scripts/analyze_vdam_native_translation_boundary.py "${ANALYZER_ARGS[@]}"
