#!/usr/bin/env bash
#SBATCH --job-name=k1c10-scatter
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:10:00

set -euo pipefail
: "${RUN_ROOT:?set an empty run root}"
: "${RUNTIME_ROOT:?set an empty runtime root}"
: "${FACTOR_DIR:?set the RELION factor directory}"
: "${PASS2_DIR:?set the RECOVAR pass-2 directory}"
: "${CUDA_LIB:?set the candidate CUDA library}"
: "${EXPECTED_LAUNCHER_SHA256:?pin the launcher}"
: "${EXPECTED_ANALYZER_SHA256:?pin the analyzer}"
: "${EXPECTED_CUDA_LIB_SHA256:?pin the CUDA library}"
: "${EXPECTED_REPO_HEAD:?pin the repository commit}"

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default
LAUNCHER=${REPO}/scripts/run_k1_case10_bpref_scatter_geometry.sbatch
ANALYZER=${REPO}/scripts/analyze_k1_bpref_scatter_geometry.py
OUTPUT=${RUN_ROOT}/analysis/K1_CASE10_IT1_BPREF_SCATTER_GEOMETRY.json

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -d "${FACTOR_DIR}" -a -d "${PASS2_DIR}"
test -n "$(find "${FACTOR_DIR}" -maxdepth 1 -type f -name '*.bpre-v2.bin' -print -quit)"
test -n "$(find "${PASS2_DIR}" -maxdepth 1 -type f -name 'pass2_orig*_cs058.npz' -print -quit)"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${ANALYZER}" | awk '{print $1}')" = "${EXPECTED_ANALYZER_SHA256}"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = "${EXPECTED_CUDA_LIB_SHA256}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test ! -e "${OUTPUT}"
test -z "$(find "${RUNTIME_ROOT}" -mindepth 1 ! -name SAFE_TO_DELETE -print -quit)"
mkdir -p "${RUN_ROOT}/analysis" "${RUN_ROOT}/logs" "${RUN_ROOT}/provenance"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_*) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/rattler_cache
export PYTHONPYCACHEPREFIX=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pycache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/jax_cache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" \
    "${PYTHONPYCACHEPREFIX}" "${JAX_COMPILATION_CACHE_DIR}"

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
export RECOVAR_CUDA_LIB=${CUDA_LIB}

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | xargs)
case "${GPU_NAME}" in *H100*) ;; *) echo "Expected H100, got ${GPU_NAME}" >&2; exit 2 ;; esac
{
    echo "date=$(date -Is)"
    echo "job_id=${SLURM_JOB_ID}"
    echo "host=$(hostname)"
    echo "gpu_name=${GPU_NAME}"
    echo "repo_head=${EXPECTED_REPO_HEAD}"
    echo "factor_dir=${FACTOR_DIR}"
    echo "pass2_dir=${PASS2_DIR}"
    sha256sum "${LAUNCHER}" "${ANALYZER}" "${CUDA_LIB}"
} > "${RUN_ROOT}/provenance/PREFLIGHT_${SLURM_JOB_ID}.txt"

cd "${REPO}"
"${PYTHON}" -m scripts.analyze_k1_bpref_scatter_geometry \
    --factor-directory "${FACTOR_DIR}" \
    --pass2-directory "${PASS2_DIR}" \
    --physical-image-size 384 \
    --output-json "${OUTPUT}" \
    > "${RUN_ROOT}/logs/analyze-${SLURM_JOB_ID}.stdout"
sha256sum "${OUTPUT}" > "${RUN_ROOT}/provenance/science_outputs_${SLURM_JOB_ID}.sha256"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
