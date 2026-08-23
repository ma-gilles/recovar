#!/usr/bin/env bash
#SBATCH --job-name=k1c26-pre-stage
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_preprocess_stage_h100_20260811T0015ET/logs/run-%j.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_preprocess_stage_h100_20260811T0015ET/logs/run-%j.err
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:05:00

set -euo pipefail
: "${EXPECTED_LAUNCHER_SHA256:?submission must pin launcher hash}"
: "${EXPECTED_ANALYZER_SHA256:?submission must pin analyzer hash}"
: "${EXPECTED_CUDA_LIB_SHA256:?submission must pin CUDA library hash}"
: "${EXPECTED_REPO_HEAD:?submission must pin repository head}"

ROOT=${RUN_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_preprocess_stage_h100_20260811T0015ET}
RUNTIME=${RUNTIME_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case26_preprocess_stage_h100_20260811T0015ET}
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
PYTHON=${REPO}/.pixi/envs/default/bin/python
LAUNCHER=${REPO}/scripts/run_k1_case26_preprocess_stage_replay.sbatch
ANALYZER=${REPO}/scripts/analyze_k1_preprocess_stage_panel.py
CUDA_LIB=${CUDA_LIBRARY:-${REPO}/recovar/cuda/libcuda_backproject.so}
CAPTURE=${CAPTURE_DIR:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_native_preprocess_it1_panel_20260810T2340ET/capture}
EXPECTED_PARTICLES=${EXPECTED_PARTICLES:-17}

require_sha() {
    test "$(sha256sum "$2" | awk '{print $1}')" = "$1"
}

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
require_sha "${EXPECTED_LAUNCHER_SHA256}" "${LAUNCHER}"
require_sha "${EXPECTED_ANALYZER_SHA256}" "${ANALYZER}"
require_sha "${EXPECTED_CUDA_LIB_SHA256}" "${CUDA_LIB}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(find "${CAPTURE}" -maxdepth 1 -type f -name '*.preprocess-v1.bin' | wc -l)" -eq "${EXPECTED_PARTICLES}"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export RECOVAR_CUDA_LIB=${CUDA_LIB}
export TMPDIR=${RUNTIME}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/rattler_cache
export PYTHONPYCACHEPREFIX=${RUNTIME}/${SLURM_JOB_ID}/pycache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" "${PYTHONPYCACHEPREFIX}"

GPU_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | xargs)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | xargs)
case "${GPU_NAME}" in *H100*) ;; *) echo "Expected H100, got ${GPU_NAME}" >&2; exit 2 ;; esac
{
    echo "date=$(date -Is)"
    echo "job_id=${SLURM_JOB_ID}"
    echo "host=$(hostname)"
    echo "gpu_uuid=${GPU_UUID}"
    echo "gpu_name=${GPU_NAME}"
    echo "repo_head=${EXPECTED_REPO_HEAD}"
    sha256sum "${LAUNCHER}" "${ANALYZER}" "${CUDA_LIB}"
} > "${ROOT}/provenance/PREFLIGHT_${SLURM_JOB_ID}.txt"

run_mode() {
    local mode=$1
    shift
    "${PYTHON}" "${ANALYZER}" \
        --native-preprocess-dir "${CAPTURE}" \
        --expected-particles "${EXPECTED_PARTICLES}" \
        --output "${ROOT}/analysis/K1_CASE26_PREPROCESS_STAGE_${mode}.json" \
        --output-npz "${ROOT}/analysis/K1_CASE26_PREPROCESS_STAGE_${mode}.npz" \
        "$@" \
        > "${ROOT}/analysis/K1_CASE26_PREPROCESS_STAGE_${mode}.stdout"
}

IFS=, read -r -a replay_modes <<< "${REPLAY_MODES:-DEFAULT,DETERMINISTIC_LANE,NATIVE_ATOMIC}"
for mode in "${replay_modes[@]}"; do
    case "${mode}" in
        DEFAULT) run_mode DEFAULT ;;
        DETERMINISTIC_LANE) run_mode DETERMINISTIC_LANE --native-lane-reduction ;;
        NATIVE_ATOMIC) run_mode NATIVE_ATOMIC --native-atomic-reduction \
            --native-atomic-repeats "${NATIVE_ATOMIC_REPEATS:-0}" ;;
        *) echo "unknown preprocessing replay mode: ${mode}" >&2; exit 2 ;;
    esac
done

sha256sum "${ROOT}"/analysis/*.json > "${ROOT}/provenance/science_outputs_${SLURM_JOB_ID}.sha256"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
