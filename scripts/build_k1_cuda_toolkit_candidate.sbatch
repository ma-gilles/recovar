#!/usr/bin/env bash
#SBATCH --job-name=k1-cuda-toolkit-build
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00

set -euo pipefail
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${SOURCE:?SOURCE is required}"
: "${PYTHON:?PYTHON is required}"
: "${CUDA_MODULE:?CUDA_MODULE is required}"
: "${EXPECTED_HEAD:?EXPECTED_HEAD is required}"
: "${EXPECTED_SOURCE_SHA:?EXPECTED_SOURCE_SHA is required}"
: "${EXPECTED_LAUNCHER_SHA:?EXPECTED_LAUNCHER_SHA is required}"

LAUNCHER=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820/scripts/build_k1_cuda_toolkit_candidate.sbatch
CUDA_SOURCE=${SOURCE}/recovar/cuda/cuda_backproject.cu
CUDA_LIB=${SOURCE}/recovar/cuda/libcuda_backproject.so

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test "$(git -C "${SOURCE}" rev-parse HEAD)" = "${EXPECTED_HEAD}"
test "$(sha256sum "${CUDA_SOURCE}" | awk '{print $1}')" = "${EXPECTED_SOURCE_SHA}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA}"

mkdir -p "${RUN_ROOT}/provenance" "${RUNTIME_ROOT}/tmp" \
    "${RUNTIME_ROOT}/pixi_home" "${RUNTIME_ROOT}/rattler_cache"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME_ROOT}/tmp PIXI_HOME=${RUNTIME_ROOT}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/rattler_cache
source /etc/profile.d/modules.sh
set +u
module purge
module load "${CUDA_MODULE}"
set -u

command -v nvcc > "${RUN_ROOT}/provenance/nvcc_path.txt"
nvcc --version > "${RUN_ROOT}/provenance/nvcc_version.txt"
"${PYTHON}" -c 'import jax; print(jax.__file__)' > "${RUN_ROOT}/provenance/jax_path.txt"
git -C "${SOURCE}" status --short --branch > "${RUN_ROOT}/provenance/git_status_before.txt"
git -C "${SOURCE}" diff --binary > "${RUN_ROOT}/provenance/source_diff.patch"
sha256sum "${CUDA_SOURCE}" "${LAUNCHER}" > "${RUN_ROOT}/provenance/inputs.sha256"

PYTHON="${PYTHON}" make -C "${SOURCE}/recovar/cuda" clean all \
    > "${RUN_ROOT}/build.log" 2>&1
test -s "${CUDA_LIB}"
sha256sum "${CUDA_LIB}" > "${RUN_ROOT}/provenance/cuda_library.sha256"
if command -v cuobjdump >/dev/null 2>&1; then
    cuobjdump --dump-resource-usage "${CUDA_LIB}" \
        > "${RUN_ROOT}/provenance/cuda_resources.txt"
else
    printf 'cuobjdump unavailable under %s\n' "${CUDA_MODULE}" \
        > "${RUN_ROOT}/provenance/cuda_resources.txt"
fi
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
