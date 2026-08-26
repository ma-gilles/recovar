#!/usr/bin/env bash
#SBATCH --job-name=k1-live-fsc
#SBATCH --partition=cpu
#SBATCH --account=gilles
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=01:00:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${CASE_ROOT:?CASE_ROOT is required}"
: "${RELION_DIR:?RELION_DIR is required}"
: "${GT_VOLUME:?GT_VOLUME is required}"
: "${OUTPUT_TAG:?OUTPUT_TAG is required}"
: "${EXPECTED_HEAD:?EXPECTED_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_FSC_SCRIPT_SHA256:?EXPECTED_FSC_SCRIPT_SHA256 is required}"

case "${OUTPUT_TAG}" in
    *[!A-Za-z0-9_.-]*|'') echo "OUTPUT_TAG contains unsafe characters" >&2; exit 2 ;;
esac

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
PYTHON=${REPO}/.pixi/envs/default/bin/python
LAUNCHER=${REPO}/scripts/audit_k1_live_fsc_prefix.sbatch
FSC_SCRIPT=${REPO}/scripts/audit_k1_fsc_trajectory.py
OUTPUT=${RUN_ROOT}/output
ANALYSIS=${RUN_ROOT}/analysis_live_${OUTPUT_TAG}

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -d "${OUTPUT}/intermediates" -a -d "${RELION_DIR}" -a -s "${GT_VOLUME}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${FSC_SCRIPT}" | awk '{print $1}')" = "${EXPECTED_FSC_SCRIPT_SHA256}"
test ! -e "${ANALYSIS}"

mkdir -p "${ANALYSIS}" "${RUN_ROOT}/provenance" \
    "${RUNTIME_ROOT}/${SLURM_JOB_ID}/tmp" \
    "${RUNTIME_ROOT}/${SLURM_JOB_ID}/pixi_home" \
    "${RUNTIME_ROOT}/${SLURM_JOB_ID}/rattler_cache"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu
export TMPDIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/rattler_cache

cd "${REPO}"
"${PYTHON}" -m scripts.audit_k1_fsc_trajectory \
    --case-root "${CASE_ROOT}" --recovar-dir "${OUTPUT}" \
    --relion-dir "${RELION_DIR}" --gt-volume "${GT_VOLUME}" \
    --numbered-only --allow-incomplete \
    --output-json "${ANALYSIS}/fsc.json" \
    --output-markdown "${ANALYSIS}/fsc.md" \
    --output-shellwise-npz "${ANALYSIS}/fsc_shellwise.npz"

sha256sum "${LAUNCHER}" "${FSC_SCRIPT}" "${ANALYSIS}/fsc.json" \
    "${ANALYSIS}/fsc.md" "${ANALYSIS}/fsc_shellwise.npz" \
    > "${RUN_ROOT}/provenance/live_fsc_${OUTPUT_TAG}_${SLURM_JOB_ID}.sha256"
touch "${ANALYSIS}/SUCCESS_${SLURM_JOB_ID}"
