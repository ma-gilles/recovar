#!/usr/bin/env bash
#SBATCH --job-name=k1-native-fine-unit
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
: "${CUDA_LIB:?CUDA_LIB is required}"
: "${EXPECTED_HEAD:?EXPECTED_HEAD is required}"
: "${EXPECTED_CUDA_SHA:?EXPECTED_CUDA_SHA is required}"
: "${EXPECTED_LAUNCHER_SHA:?EXPECTED_LAUNCHER_SHA is required}"
: "${EXPECTED_TEST_SHA:?EXPECTED_TEST_SHA is required}"

LAUNCHER=${SOURCE}/scripts/run_k1_native_fine_cuda_unit.sbatch
TEST_FILE=${SOURCE}/tests/unit/test_cuda_relion_fine_diff2.py
test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test "$(git -C "${SOURCE}" rev-parse HEAD)" = "${EXPECTED_HEAD}"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA}"
test "$(sha256sum "${TEST_FILE}" | awk '{print $1}')" = "${EXPECTED_TEST_SHA}"

mkdir -p "${RUN_ROOT}/provenance" "${RUNTIME_ROOT}/tmp" \
    "${RUNTIME_ROOT}/pixi_home" "${RUNTIME_ROOT}/rattler_cache" \
    "${RUNTIME_ROOT}/jax_cache" "${RUNTIME_ROOT}/pycache"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME_ROOT}/tmp PIXI_HOME=${RUNTIME_ROOT}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/rattler_cache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME_ROOT}/jax_cache
export PYTHONPYCACHEPREFIX=${RUNTIME_ROOT}/pycache
export RECOVAR_CUDA_LIB=${CUDA_LIB}
source /etc/profile.d/modules.sh
set +u
module purge
module load cudatoolkit/12.8
set -u

PIXI_ENV=${SOURCE}/.pixi/envs/default
CUDA_ROOT=/usr/local/cuda-12.8
NVIDIA_ROOT=$(find "${PIXI_ENV}/lib" -maxdepth 3 -type d -path '*/site-packages/nvidia' -print -quit)
NVIDIA_LIBS=$(find "${NVIDIA_ROOT}" -type d -name lib | paste -sd: -)
export PATH=${CUDA_ROOT}/bin:${PATH}
export LD_LIBRARY_PATH=${NVIDIA_LIBS}:${CUDA_ROOT}/targets/x86_64-linux/lib:${PIXI_ENV}/lib

nvidia-smi --query-gpu=uuid,name,memory.total --format=csv,noheader \
    > "${RUN_ROOT}/provenance/gpu_${SLURM_JOB_ID}.txt"
"${PYTHON}" -c "import pathlib,recovar,jax; from recovar import cuda_backproject; root=pathlib.Path('${SOURCE}').resolve(); assert pathlib.Path(recovar.__file__).resolve().is_relative_to(root); assert '.pixi/envs/default/' in str(pathlib.Path(jax.__file__).resolve()); assert jax.default_backend() == 'gpu'; assert cuda_backproject.cuda_available()"
sha256sum "${CUDA_LIB}" "${LAUNCHER}" "${TEST_FILE}" \
    > "${RUN_ROOT}/provenance/inputs_${SLURM_JOB_ID}.sha256"

cd "${SOURCE}"
"${PYTHON}" -m pytest -q "${TEST_FILE}" \
    -k 'relion_fine_native_texture_is_bitwise_batch_context_invariant' \
    > "${RUN_ROOT}/pytest.log" 2>&1
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
