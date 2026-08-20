#!/usr/bin/env bash
#SBATCH --job-name=vdam-relion-parity
#SBATCH --account=amits
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-relion-parity-%j.out

set -euo pipefail

: "${CASE_ID:?submit with --export=ALL,CASE_ID=vdam-XX,OUTPUT_ROOT=/scratch/...}"
: "${OUTPUT_ROOT:?submit with --export=ALL,CASE_ID=vdam-XX,OUTPUT_ROOT=/scratch/...}"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA RECOVAR_CUDA_DISABLE
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
RUNTIME_ROOT="/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/${SLURM_JOB_ID:-manual}"
export TMPDIR="${RUNTIME_ROOT}/tmp"
export PIXI_HOME="${RUNTIME_ROOT}/pixi_home"
export RATTLER_CACHE_DIR="${RUNTIME_ROOT}/rattler_cache"
export JAX_COMPILATION_CACHE_DIR="${OUTPUT_ROOT}/jax_cache"
export RECOVAR_CUDA_LIB="${REPO_ROOT}/recovar/cuda/libcuda_backproject.so"
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" "${JAX_COMPILATION_CACHE_DIR}" \
  /scratch/gpfs/GILLES/mg6942/slurmo

PIXI_PY="${REPO_ROOT}/.pixi/envs/default/bin/python"
test -x "${PIXI_PY}"

"${PIXI_PY}" - <<'PY'
import pathlib
import jax
import recovar
import recovar.cuda_backproject as cuda_backproject
from recovar.relion_bind import _relion_bind_core

repo = pathlib.Path.cwd().resolve()
assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo) + "/")
assert ".pixi/envs/default/" in str(pathlib.Path(jax.__file__).resolve())
assert str(pathlib.Path(_relion_bind_core.__file__).resolve()).startswith(str(repo) + "/")
assert any(device.platform in {"gpu", "cuda"} for device in jax.devices())
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
print("VDAM parity provenance/GPU/CUDA-FFI gate passed", jax.default_backend(), jax.devices())
PY

"${PIXI_PY}" scripts/run_vdam_relion_parity_case.py \
  --case-id "${CASE_ID}" \
  --output-root "${OUTPUT_ROOT}" \
  --repo "${REPO_ROOT}"
