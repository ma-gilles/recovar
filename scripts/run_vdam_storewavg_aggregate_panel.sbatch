#!/usr/bin/env bash
#SBATCH --job-name=vdam-wavg-all
#SBATCH --account=amits
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-wavg-all-%j.out

set -euo pipefail

: "${NATIVE_DIRECTORY:?submit with NATIVE_DIRECTORY=/scratch/.../capture}"
: "${RELION_DATA_STAR:?submit with RELION_DATA_STAR=/scratch/...star}"
: "${RECOVAR_HALF1_DIRECTORY:?submit with RECOVAR_HALF1_DIRECTORY=/scratch/...}"
: "${RECOVAR_HALF2_DIRECTORY:?submit with RECOVAR_HALF2_DIRECTORY=/scratch/...}"
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
mapfile -t HALF1_CAPTURES < <(find "${RECOVAR_HALF1_DIRECTORY}" -maxdepth 1 -type f -name '*.npz' | sort)
mapfile -t HALF2_CAPTURES < <(find "${RECOVAR_HALF2_DIRECTORY}" -maxdepth 1 -type f -name '*.npz' | sort)
[[ "${#HALF1_CAPTURES[@]}" -eq 2 ]] || {
  echo "expected exactly two half-1 captures, found ${#HALF1_CAPTURES[@]}" >&2
  exit 2
}
[[ "${#HALF2_CAPTURES[@]}" -eq 2 ]] || {
  echo "expected exactly two half-2 captures, found ${#HALF2_CAPTURES[@]}" >&2
  exit 2
}

git rev-parse HEAD
git symbolic-ref --short HEAD || echo '<detached>'
git status --short --branch
git diff HEAD --stat
git diff HEAD | sha256sum
"${PIXI_PY}" - <<'PY'
import pathlib

import jax
import recovar
from recovar import cuda_backproject
from recovar.utils.parity_provenance import assert_parity_ancestors

repo = pathlib.Path.cwd().resolve()
assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo) + "/")
assert ".pixi/envs/default/" in str(pathlib.Path(jax.__file__).resolve())
assert jax.default_backend() == "gpu"
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
assert_parity_ancestors()
print("VDAM aggregate StoreWavg provenance/GPU/CUDA-FFI gate passed", jax.devices())
PY

"${PIXI_PY}" -m pytest -v \
  tests/unit/initial_model/test_vdam_storewavg_aggregate_panel.py \
  tests/unit/initial_model/test_vdam_storewavg_boundary.py \
  tests/unit/test_analyze_vdam_storewavg_panel.py

ANALYZER_ARGS=(
  --native-directory "${NATIVE_DIRECTORY}"
  --relion-data-star "${RELION_DATA_STAR}"
  --output-json "${OUTPUT_JSON}"
  --rotation-tolerance "${ROTATION_TOLERANCE:-1e-5}"
)
for path in "${HALF1_CAPTURES[@]}"; do
  ANALYZER_ARGS+=(--recovar-half1-capture "${path}")
done
for path in "${HALF2_CAPTURES[@]}"; do
  ANALYZER_ARGS+=(--recovar-half2-capture "${path}")
done
if [[ -n "${NATIVE_PRODUCTION_DIRECTORY:-}" ]]; then
  : "${RECOVAR_PRODUCTION_DIRECTORY:?set with NATIVE_PRODUCTION_DIRECTORY}"
  ANALYZER_ARGS+=(
    --native-production-directory "${NATIVE_PRODUCTION_DIRECTORY}"
    --recovar-production-directory "${RECOVAR_PRODUCTION_DIRECTORY}"
  )
fi

"${PIXI_PY}" -m scripts.analyze_vdam_storewavg_aggregate_panel "${ANALYZER_ARGS[@]}"
