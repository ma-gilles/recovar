#!/usr/bin/env bash
#SBATCH --job-name=k1-ppref-repeat
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:20:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${NATIVE_PPREF_DIR:?NATIVE_PPREF_DIR is required}"
: "${PRIMARY_MAP:?PRIMARY_MAP is required}"
: "${REPEAT_MAP:?REPEAT_MAP is required}"
: "${RECOVAR_MAP:?RECOVAR_MAP is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_ANALYZER_SHA256:?EXPECTED_ANALYZER_SHA256 is required}"

IREF_PATH=${IREF_PATH:-}
IREF_LABEL=${IREF_LABEL:-native_in_memory}
[[ "${IREF_LABEL}" =~ ^[A-Za-z0-9._-]+$ ]]

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_native_units_instrumentation_20260826
PYTHON=${REPO}/.pixi/envs/default/bin/python
LAUNCHER=${REPO}/scripts/audit_k1_ppref_map_repeat_floor.sbatch
ANALYZER=${REPO}/scripts/analyze_k1_ppref_map_provenance.py
RELION_BIND=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET/relion_bind_build/shared
REPORT=${RUN_ROOT}/analysis/NATIVE_PPREF_MAP_REPEAT_FLOOR.json

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${ANALYZER}" | awk '{print $1}')" = "${EXPECTED_ANALYZER_SHA256}"
test "$(sha256sum "${RELION_BIND}/_relion_bind_core.cpython-311-x86_64-linux-gnu.so" | awk '{print $1}')" = 56acdddab6d25bd83aac5e338162c4819f85ec26f68511458618f143c2b604b7
test -s "${NATIVE_PPREF_DIR}/pass1_class0_ppref_real.bin"
test -s "${PRIMARY_MAP}" -a -s "${REPEAT_MAP}" -a -s "${RECOVAR_MAP}"
if test -n "${IREF_PATH}"; then
    test -s "${IREF_PATH}"
fi
test ! -e "${REPORT}"

mkdir -p "${RUN_ROOT}/analysis" "${RUN_ROOT}/provenance" \
    "${RUNTIME_ROOT}/tmp" "${RUNTIME_ROOT}/pixi_home" \
    "${RUNTIME_ROOT}/rattler_cache" "${RUNTIME_ROOT}/jax_cache" \
    "${RUNTIME_ROOT}/pycache"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME_ROOT}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/rattler_cache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME_ROOT}/jax_cache
export PYTHONPYCACHEPREFIX=${RUNTIME_ROOT}/pycache
export RECOVAR_RELION_BIND_BUILD_DIR=${RELION_BIND}

source /etc/profile.d/modules.sh
set +u
module purge
module load cudatoolkit/12.8
set -u
CUDA_ROOT=/usr/local/cuda-12.8
PIXI_ENV=${REPO}/.pixi/envs/default
NVIDIA_ROOT=$(find "${PIXI_ENV}/lib" -maxdepth 3 -type d -path '*/site-packages/nvidia' -print -quit)
NVIDIA_LIBS=$(find "${NVIDIA_ROOT}" -type d -name lib | paste -sd: -)
export PATH=${CUDA_ROOT}/bin:${PATH}
export LD_LIBRARY_PATH=${NVIDIA_LIBS}:${CUDA_ROOT}/targets/x86_64-linux/lib:${PIXI_ENV}/lib

cd "${REPO}"
nvidia-smi --query-gpu=uuid,name,memory.total --format=csv,noheader \
    > "${RUN_ROOT}/provenance/gpu_${SLURM_JOB_ID}.txt"
env | sort > "${RUN_ROOT}/provenance/environment_${SLURM_JOB_ID}.txt"
hash_inputs=("${LAUNCHER}" "${ANALYZER}"
    "${RELION_BIND}/_relion_bind_core.cpython-311-x86_64-linux-gnu.so"
    "${PRIMARY_MAP}" "${REPEAT_MAP}" "${RECOVAR_MAP}")
if test -n "${IREF_PATH}"; then
    hash_inputs+=("${IREF_PATH}")
fi
sha256sum "${hash_inputs[@]}" \
    > "${RUN_ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"

command=("${PYTHON}" -m scripts.analyze_k1_ppref_map_provenance \
    --ppref "${NATIVE_PPREF_DIR}" \
    --map "native_primary_numbered:${PRIMARY_MAP}:relion" \
    --map "native_repeat_numbered:${REPEAT_MAP}:relion" \
    --map "recovar_current:${RECOVAR_MAP}:recovar" \
    --output-json "${REPORT}")
if test -n "${IREF_PATH}"; then
    command+=(--iref "${IREF_LABEL}:${IREF_PATH}")
fi
"${command[@]}"

sha256sum "${REPORT}" > "${RUN_ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
