#!/usr/bin/env bash
#SBATCH --job-name=k1c10-primitive
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:15:00

set -euo pipefail
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${FACTOR_DIR:?FACTOR_DIR is required}"
: "${PASS2_DIR:?PASS2_DIR is required}"
: "${CONTRIBUTION_DIR:?CONTRIBUTION_DIR is required}"
: "${SELECTION_JSON:?SELECTION_JSON is required}"
: "${LIVE_SIGMA2_NPY:?LIVE_SIGMA2_NPY is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_ANALYZER_SHA256:?EXPECTED_ANALYZER_SHA256 is required}"
: "${EXPECTED_CUDA_LIB_SHA256:?EXPECTED_CUDA_LIB_SHA256 is required}"
: "${EXPECTED_SELECTION_SHA256:?EXPECTED_SELECTION_SHA256 is required}"
: "${EXPECTED_LIVE_SIGMA2_SHA256:?EXPECTED_LIVE_SIGMA2_SHA256 is required}"

ROOT=${RUN_ROOT}
RUNTIME=${RUNTIME_ROOT}
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default
LAUNCHER=${REPO}/scripts/run_k1_case10_it1_bpref_primitives_analysis.sbatch
ANALYZER=${REPO}/scripts/compare_k1_relion_recovar_bpref_primitives.py
CUDA_LIB=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case05_it1_bpref_fma_44d770de3_20260823T1938ET/cuda/libcuda_backproject.so
SOURCE_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET
DATA=${SOURCE_ROOT}/cases/10_high_res_anisotropic_100k_g384_radial_noise3_bf0/data
RELION_BIND=${RELION_BIND_OVERRIDE:-${SOURCE_ROOT}/relion_bind_build/shared}
OUTPUT=${ROOT}/analysis/K1_CASE10_IT1_BPREF_PRIMITIVES.json
DUMPS=${ROOT}/analysis/primitive_dumps

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test -d "${FACTOR_DIR}" -a -d "${PASS2_DIR}" -a -d "${CONTRIBUTION_DIR}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${ANALYZER}" | awk '{print $1}')" = "${EXPECTED_ANALYZER_SHA256}"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = "${EXPECTED_CUDA_LIB_SHA256}"
test "$(sha256sum "${SELECTION_JSON}" | awk '{print $1}')" = "${EXPECTED_SELECTION_SHA256}"
test "$(sha256sum "${LIVE_SIGMA2_NPY}" | awk '{print $1}')" = "${EXPECTED_LIVE_SIGMA2_SHA256}"
test ! -e "${OUTPUT}" -a ! -e "${DUMPS}"
mkdir -p "${ROOT}/analysis" "${ROOT}/logs" "${ROOT}/provenance"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_*) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/rattler_cache
export PYTHONPYCACHEPREFIX=${RUNTIME}/${SLURM_JOB_ID}/pycache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/jax_cache
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
export RECOVAR_RELION_BIND_BUILD_DIR=${RELION_BIND}

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | xargs)
case "${GPU_NAME}" in *H100*) ;; *) echo "Expected H100, got ${GPU_NAME}" >&2; exit 2 ;; esac
{
    echo "date=$(date -Is)"
    echo "job_id=${SLURM_JOB_ID}"
    echo "host=$(hostname)"
    echo "gpu_name=${GPU_NAME}"
    echo "factor_dir=${FACTOR_DIR}"
    echo "pass2_dir=${PASS2_DIR}"
    echo "contribution_dir=${CONTRIBUTION_DIR}"
    sha256sum "${LAUNCHER}" "${ANALYZER}" "${CUDA_LIB}" \
        "${SELECTION_JSON}" "${LIVE_SIGMA2_NPY}"
} > "${ROOT}/provenance/PREFLIGHT_${SLURM_JOB_ID}.txt"

cd "${REPO}"
"${PYTHON}" -m scripts.compare_k1_relion_recovar_bpref_primitives \
    --factor-directory "${FACTOR_DIR}" \
    --pass2-directory "${PASS2_DIR}" \
    --contribution-directory "${CONTRIBUTION_DIR}" \
    --selection-json "${SELECTION_JSON}" \
    --source-star "${DATA}/particles.star" \
    --relion-bind-directory "${RELION_BIND}" \
    --live-initial-sigma2 "${LIVE_SIGMA2_NPY}" \
    --dump-directory "${DUMPS}" \
    --output-json "${OUTPUT}" \
    > "${ROOT}/logs/analyze-${SLURM_JOB_ID}.stdout" \
    2> "${ROOT}/logs/analyze-${SLURM_JOB_ID}.stderr"
sha256sum "${OUTPUT}" "${DUMPS}"/*.npz > "${ROOT}/provenance/science_outputs_${SLURM_JOB_ID}.sha256"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
