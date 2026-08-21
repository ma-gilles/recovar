#!/usr/bin/env bash
#SBATCH --job-name=k1c26-ppref-wavg
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:05:00

set -euo pipefail
: "${RUN_ROOT:?set the native Wavg/PPref capture root}"
: "${RUNTIME_ROOT:?set an empty runtime root}"
: "${RECOVAR_CAPTURE_ROOT:?set the source-23 RECOVAR capture root}"
: "${EXPECTED_LAUNCHER_SHA256:?pin this launcher}"
: "${EXPECTED_ANALYZER_SHA256:?pin the analyzer}"
: "${EXPECTED_REPO_HEAD:?pin the repository commit}"
REC_CAPTURE_ROOT=${RECOVAR_CAPTURE_ROOT}

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default
LAUNCHER=${REPO}/scripts/run_k1_case26_ppref_wavg_analysis.sbatch
ANALYZER=${REPO}/scripts/analyze_k1_native_wavg_pixels.py
CASE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_current_pr_failure_gate_fd8e27b9_20260808T1515ET/scorecard_remaining5/cases/26_tiny_severe_1k_g128_radial_noise5_nonuniform_pct30_bf80
RELION_BIND=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_bpref_order_fix_20260810/relion_bind
CUDA_LIB=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_bpref_factor_boundary_7c3b5478_20260810T0610ET/small_control_retry2/cuda/libcuda_backproject.so
PPREF=${PPREF_OVERRIDE:-${RUN_ROOT}/ppref/ppref_iter001_rank-01_model000.bin}
OUTPUT=${RUN_ROOT}/analysis/K1_CASE26_SOURCE23_EXACT_PPREF_WAVG.json

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${ANALYZER}" | awk '{print $1}')" = "${EXPECTED_ANALYZER_SHA256}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${RELION_BIND}"/_relion_bind_core*.so | awk '{print $1}')" = 793919bdb09875304d630727f9930ca729e1d261f6a354cecce7221af728dced
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = d9a32bd168fe38e82c755eb54012007e0ffd621dc0c421d4ccedda8a9ac7d9ff
test -s "${PPREF}"
test ! -e "${OUTPUT}"
test -z "$(find "${RUNTIME_ROOT}" -mindepth 1 ! -name SAFE_TO_DELETE -print -quit)"
mkdir -p "${RUN_ROOT}/analysis" "${RUN_ROOT}/provenance"

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
    sha256sum "${LAUNCHER}" "${ANALYZER}" "${PPREF}" "${CUDA_LIB}"
} > "${RUN_ROOT}/provenance/PPREF_WAVG_ANALYSIS_PREFLIGHT_${SLURM_JOB_ID}.txt"

"${PYTHON}" "${ANALYZER}" \
    --recovar-pass2-dir "${REC_CAPTURE_ROOT}/pass2" \
    --native-wavg-dir "${RUN_ROOT}/capture" \
    --native-preprocess-dir "${RUN_ROOT}/capture" \
    --source-star "${CASE}/data/particles.star" \
    --relion-bind-dir "${RELION_BIND}" \
    --cuda-lib "${CUDA_LIB}" \
    --expected-particles 1 \
    --ppref-half2 "${PPREF}" \
    --output "${OUTPUT}" \
    > "${RUN_ROOT}/analysis/K1_CASE26_SOURCE23_EXACT_PPREF_WAVG.stdout"

sha256sum "${OUTPUT}" \
    > "${RUN_ROOT}/provenance/ppref_wavg_analysis_outputs_${SLURM_JOB_ID}.sha256"
touch "${RUN_ROOT}/PPREF_WAVG_ANALYSIS_SUCCESS_${SLURM_JOB_ID}"
