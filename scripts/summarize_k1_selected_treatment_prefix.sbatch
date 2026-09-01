#!/usr/bin/env bash
#SBATCH --job-name=k1-prefix-summary
#SBATCH --partition=cpu
#SBATCH --account=gilles
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00

set -euo pipefail

: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${CASE4_REPORT:?CASE4_REPORT is required}"
: "${CASE5_REPORT:?CASE5_REPORT is required}"
: "${CASE10_REPORT:?CASE10_REPORT is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_ANALYZER_SHA256:?EXPECTED_ANALYZER_SHA256 is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
PYTHON=${REPO}/.pixi/envs/default/bin/python
ANALYZER=${REPO}/scripts/summarize_k1_selected_treatment_prefix.py
LAUNCHER=${REPO}/scripts/summarize_k1_selected_treatment_prefix.sbatch

test -f "${OUTPUT_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -s "${CASE4_REPORT}" -a -s "${CASE5_REPORT}" -a -s "${CASE10_REPORT}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${ANALYZER}" | awk '{print $1}')" = "${EXPECTED_ANALYZER_SHA256}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"

mkdir -p "${OUTPUT_ROOT}/analysis" "${OUTPUT_ROOT}/provenance"
mkdir -p "${RUNTIME_ROOT}/tmp" "${RUNTIME_ROOT}/pixi_home" "${RUNTIME_ROOT}/rattler_cache"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1
export TMPDIR=${RUNTIME_ROOT}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/rattler_cache

cd "${REPO}"
"${PYTHON}" -m scripts.summarize_k1_selected_treatment_prefix \
  --case "4=${CASE4_REPORT}" \
  --case "5=${CASE5_REPORT}" \
  --case "10=${CASE10_REPORT}" \
  --output-json "${OUTPUT_ROOT}/analysis/k1_selected_treatment_prefix.json" \
  --output-markdown "${OUTPUT_ROOT}/analysis/k1_selected_treatment_prefix.md"

sha256sum "${ANALYZER}" "${LAUNCHER}" \
  "${CASE4_REPORT}" "${CASE5_REPORT}" "${CASE10_REPORT}" \
  "${OUTPUT_ROOT}/analysis/k1_selected_treatment_prefix.json" \
  "${OUTPUT_ROOT}/analysis/k1_selected_treatment_prefix.md" \
  > "${OUTPUT_ROOT}/provenance/summary_${SLURM_JOB_ID}.sha256"
touch "${OUTPUT_ROOT}/SUMMARY_SUCCESS_${SLURM_JOB_ID}"
