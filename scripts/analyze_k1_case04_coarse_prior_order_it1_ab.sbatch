#!/usr/bin/env bash
#SBATCH --job-name=k1-c04-prior-ab
#SBATCH --partition=cpu
#SBATCH --account=gilles
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=00:20:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${LEGACY_RUN_ROOT:?LEGACY_RUN_ROOT is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_ANALYZER_SHA256:?EXPECTED_ANALYZER_SHA256 is required}"

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_native_units_instrumentation_20260826
PYTHON=/usr/bin/python3
LAUNCHER=${REPO}/scripts/analyze_k1_case04_coarse_prior_order_it1_ab.sbatch
ANALYZER=${REPO}/scripts/analyze_k1_case04_coarse_prior_order_it1_ab.py
OLD_DIR=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_float64_radius_prefix2_retry3_20260827T0110ET/recovar/output/intermediates
CURRENT_DIR=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_top4_recovar_fresh_fine_retry1_20260827T0435ET/output/intermediates
LEGACY_DIR=${LEGACY_RUN_ROOT}/output/intermediates
OUTPUT=${RUN_ROOT}/analysis/K1_CASE04_COARSE_PRIOR_ORDER_IT1_AB.json

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -f "${LEGACY_RUN_ROOT}/SAFE_TO_DELETE"
test -s "${LEGACY_RUN_ROOT}/output/refinement_results.npz"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${ANALYZER}" | awk '{print $1}')" = "${EXPECTED_ANALYZER_SHA256}"
test ! -e "${OUTPUT}"

mkdir -p "${RUN_ROOT}/analysis" "${RUN_ROOT}/provenance" \
  "${RUNTIME_ROOT}/tmp" "${RUNTIME_ROOT}/pixi_home" \
  "${RUNTIME_ROOT}/rattler_cache"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1
export TMPDIR=${RUNTIME_ROOT}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/rattler_cache

cd "${REPO}"
"${PYTHON}" -m scripts.analyze_k1_case04_coarse_prior_order_it1_ab \
  --old-dir "${OLD_DIR}" \
  --current-dir "${CURRENT_DIR}" \
  --legacy-dir "${LEGACY_DIR}" \
  --output-json "${OUTPUT}" > "${RUN_ROOT}/analysis/stdout_${SLURM_JOB_ID}.log"

sha256sum "${LAUNCHER}" "${ANALYZER}" "${OUTPUT}" \
  > "${RUN_ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
