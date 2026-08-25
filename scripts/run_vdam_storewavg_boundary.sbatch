#!/usr/bin/env bash
#SBATCH --job-name=vdam-storewavg
#SBATCH --account=amits
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-storewavg-%j.out

set -euo pipefail

: "${NATIVE_DIRECTORY:?submit with NATIVE_DIRECTORY=/scratch/...}"
: "${NATIVE_IMAGE:?submit with NATIVE_IMAGE=/scratch/.../*nomask.bin}"
: "${RECOVAR_CAPTURE:?submit with RECOVAR_CAPTURE=/scratch/...npz}"
: "${OUTPUT_JSON:?submit with OUTPUT_JSON=/scratch/...json}"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ID="${SLURM_JOB_ID:-manual}"
cd "${REPO_ROOT}"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV JAX_PLATFORM_NAME
export PYTHONNOUSERSITE=1
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/${RUN_ID}/tmp"
export PIXI_HOME="/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/${RUN_ID}/pixi_home"
export RATTLER_CACHE_DIR="/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/${RUN_ID}/rattler_cache"
export RECOVAR_CUDA_LIB="${REPO_ROOT}/recovar/cuda/libcuda_backproject.so"
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" /scratch/gpfs/GILLES/mg6942/slurmo

PIXI_PY="${REPO_ROOT}/.pixi/envs/default/bin/python"
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
print("VDAM StoreWavg provenance/GPU/CUDA-FFI gate passed", jax.devices())
PY

ANALYZER_ARGS=(
  --native-directory "${NATIVE_DIRECTORY}"
  --native-image "${NATIVE_IMAGE}"
  --recovar-capture "${RECOVAR_CAPTURE}"
  --native-prefix "${NATIVE_PREFIX:-img0_part0_storeWavg_}"
  --physical-image-size "${PHYSICAL_IMAGE_SIZE:-128}"
  --current-size "${CURRENT_SIZE:-38}"
  --output-json "${OUTPUT_JSON}"
)
if [[ -n "${NATIVE_INVERSE_NOISE:-}" ]]; then
  ANALYZER_ARGS+=(--native-inverse-noise "${NATIVE_INVERSE_NOISE}")
fi
if [[ -n "${NATIVE_PROJECTOR_PREFIX:-}" ]]; then
  ANALYZER_ARGS+=(--native-projector-prefix "${NATIVE_PROJECTOR_PREFIX}")
fi
if [[ -n "${NATIVE_PARTICLE_BPREF_PREFIX:-}" ]]; then
  ANALYZER_ARGS+=(--native-particle-bpref-prefix "${NATIVE_PARTICLE_BPREF_PREFIX}")
fi
if [[ -n "${RECOVAR_ORIGINAL_INDEX:-}" ]]; then
  ANALYZER_ARGS+=(--recovar-original-index "${RECOVAR_ORIGINAL_INDEX}")
fi
if [[ -n "${RECOVAR_SCORE_DUMP:-}" ]]; then
  ANALYZER_ARGS+=(--recovar-score-dump "${RECOVAR_SCORE_DUMP}")
fi
"${PIXI_PY}" scripts/analyze_vdam_storewavg_boundary.py "${ANALYZER_ARGS[@]}"
