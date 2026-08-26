#!/usr/bin/env bash
#SBATCH --job-name=k1-native-batch-invariant
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:15:00

set -euo pipefail
trap 'status=$?; printf "ERROR status=%d line=%d command=%q\n" "${status}" "${LINENO}" "${BASH_COMMAND}" >&2' ERR

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${REPO:?REPO is required}"
: "${EXPECTED_HEAD:?EXPECTED_HEAD is required}"
: "${EXPECTED_TEST_SHA256:?EXPECTED_TEST_SHA256 is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${CUDA_LIB:?CUDA_LIB is required}"
: "${EXPECTED_CUDA_LIB_SHA256:?EXPECTED_CUDA_LIB_SHA256 is required}"

TEST=${REPO}/tests/unit/test_cuda_relion_fine_diff2.py
LAUNCHER=${REPO}/scripts/run_k1_native_texture_batch_invariance.sbatch
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_HEAD}"
test "$(sha256sum "${TEST}" | awk '{print $1}')" = "${EXPECTED_TEST_SHA256}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = "${EXPECTED_CUDA_LIB_SHA256}"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/provenance" \
    "${RUNTIME_ROOT}/tmp" "${RUNTIME_ROOT}/pixi_home" \
    "${RUNTIME_ROOT}/rattler_cache" "${RUNTIME_ROOT}/jax_cache" \
    "${RUNTIME_ROOT}/pycache"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME_ROOT}/tmp PIXI_HOME=${RUNTIME_ROOT}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/rattler_cache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME_ROOT}/jax_cache
export PYTHONPYCACHEPREFIX=${RUNTIME_ROOT}/pycache
export RECOVAR_CUDA_LIB=${CUDA_LIB}
export RECOVAR_REQUIRE_CUSTOM_CUDA_FOR_TESTS=1

source /etc/profile.d/modules.sh
set +u
module purge
module load cudatoolkit/12.8
set -u
CUDA_HOME=/usr/local/cuda-12.8
export PATH=${CUDA_HOME}/bin:${PATH}
NVIDIA_ROOT=$(find "${PIXI_ENV}/lib" -maxdepth 3 -type d -path '*/site-packages/nvidia' -print -quit)
NVIDIA_LIBS=$(find "${NVIDIA_ROOT}" -type d -name lib | paste -sd: -)
export LD_LIBRARY_PATH=${NVIDIA_LIBS}:${CUDA_HOME}/targets/x86_64-linux/lib:${PIXI_ENV}/lib

git -C "${REPO}" status --short --branch > "${RUN_ROOT}/provenance/git_status.txt"
git -C "${REPO}" diff --binary > "${RUN_ROOT}/provenance/source_diff.patch"
sha256sum "${TEST}" "${LAUNCHER}" "${CUDA_LIB}" > "${RUN_ROOT}/provenance/inputs.sha256"
nvidia-smi > "${RUN_ROOT}/provenance/gpu.txt"
"${PYTHON}" -c 'import jax, recovar; print(jax.__file__); print(recovar.__file__); print(jax.devices())' \
    > "${RUN_ROOT}/provenance/imports_and_devices.txt"

cd "${REPO}"
"${PYTHON}" -m pytest -v --run-gpu \
    tests/unit/test_cuda_relion_fine_diff2.py::test_relion_coarse_native_texture_is_bitwise_batch_context_invariant \
    2>&1 | tee "${RUN_ROOT}/logs/pytest.log"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
