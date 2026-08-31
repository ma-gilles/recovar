#!/usr/bin/env bash
#SBATCH --job-name=vdam-stable-fine
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-stable-fine-%j.out

set -euo pipefail

: "${OUTPUT_ROOT:?set a new SAFE_TO_DELETE output root}"
: "${PIXI_PY_OVERRIDE:?set the pinned pixi Python}"

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR}}
PIXI_PY=${PIXI_PY_OVERRIDE}
NVCC_PATH=${NVCC_PATH:-/usr/local/cuda-12.9/bin/nvcc}
CUDA_LIB_OVERRIDE=${CUDA_LIB_OVERRIDE:-}
RUN_GPU_PYTEST=${RUN_GPU_PYTEST:-1}
test ! -e "${OUTPUT_ROOT}"
test -x "${PIXI_PY}"
test -x "${NVCC_PATH}"

mkdir -p "${OUTPUT_ROOT}/provenance" "${OUTPUT_ROOT}/analysis" "${OUTPUT_ROOT}/build"
touch "${OUTPUT_ROOT}/SAFE_TO_DELETE"
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.50
export TMPDIR="${OUTPUT_ROOT}/tmp"
export PIXI_HOME="${OUTPUT_ROOT}/pixi_home"
export RATTLER_CACHE_DIR="${OUTPUT_ROOT}/rattler_cache"
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV

cd "${REPO_ROOT}"
git rev-parse HEAD > "${OUTPUT_ROOT}/provenance/repo_head.txt"
git status --short > "${OUTPUT_ROOT}/provenance/git_status.txt"
git diff --binary > "${OUTPUT_ROOT}/provenance/git_diff.patch"
sha256sum \
  recovar/cuda/cuda_backproject.cu \
  recovar/cuda_backproject.py \
  recovar/em/dense_single_volume/helpers/fourier_window.py \
  tests/unit/test_cuda_relion_fine_diff2.py \
  scripts/benchmark_vdam_stable_fine_window.py \
  scripts/run_vdam_stable_fine_window_gate.sbatch \
  > "${OUTPUT_ROOT}/provenance/source.sha256"
nvidia-smi -q > "${OUTPUT_ROOT}/provenance/nvidia_smi.txt"
env | LC_ALL=C sort > "${OUTPUT_ROOT}/provenance/environment.txt"

if [[ -n "${CUDA_LIB_OVERRIDE}" ]]; then
  test -s "${CUDA_LIB_OVERRIDE}"
  CUDA_LIB=${CUDA_LIB_OVERRIDE}
else
  CUDA_LIB="${OUTPUT_ROOT}/build/libcuda_backproject.so"
  export NVCC="${NVCC_PATH}"
  CUDA_LIB="${CUDA_LIB}" "${PIXI_PY}" - <<'PY' \
    > "${OUTPUT_ROOT}/analysis/build.stdout" 2> "${OUTPUT_ROOT}/analysis/build.stderr"
import os
from recovar import cuda_backproject

print(cuda_backproject.build_custom_cuda(os.environ["CUDA_LIB"], force=True))
PY
fi
sha256sum "${CUDA_LIB}" > "${OUTPUT_ROOT}/provenance/cuda_library.sha256"
export RECOVAR_CUDA_LIB="${CUDA_LIB}"
export RECOVAR_REQUIRE_CUSTOM_CUDA_FOR_TESTS=1

if [[ "${RUN_GPU_PYTEST}" == 1 ]]; then
  "${PIXI_PY}" -m pytest --run-gpu -q \
    tests/unit/test_cuda_relion_fine_diff2.py::test_relion_runtime_cutoff_fine_diff2_matches_static_paths_and_reuses_compile \
    | tee "${OUTPUT_ROOT}/analysis/gpu_pytest.txt"
fi

for order in runtime-first static-first; do
  "${PIXI_PY}" -m scripts.benchmark_vdam_stable_fine_window \
    --order "${order}" \
    --repeats 3000 \
    --output "${OUTPUT_ROOT}/analysis/${order}.json" \
    | tee "${OUTPUT_ROOT}/analysis/${order}.stdout"
done

touch "${OUTPUT_ROOT}/SUCCESS"
