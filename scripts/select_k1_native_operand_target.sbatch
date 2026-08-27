#!/usr/bin/env bash
#SBATCH --job-name=k1-native-target
#SBATCH --partition=cpu
#SBATCH --account=gilles
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${STAGE_JOIN_JSON:?STAGE_JOIN_JSON is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_SELECTOR_SHA256:?EXPECTED_SELECTOR_SHA256 is required}"
: "${EXPECTED_BOUNDARY_ANALYZER_SHA256:?EXPECTED_BOUNDARY_ANALYZER_SHA256 is required}"
: "${EXPECTED_FACTOR_VALIDATOR_SHA256:?EXPECTED_FACTOR_VALIDATOR_SHA256 is required}"
: "${EXPECTED_SCORE_VALIDATOR_SHA256:?EXPECTED_SCORE_VALIDATOR_SHA256 is required}"

REPO=${REPO:-/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_native_units_instrumentation_20260826}
PYTHON=${PYTHON:-${REPO}/.pixi/envs/default/bin/python}
LAUNCHER=${REPO}/scripts/select_k1_native_operand_target.sbatch
SELECTOR=${REPO}/scripts/select_k1_native_operand_target.py
BOUNDARY_ANALYZER=${REPO}/scripts/analyze_k1_fine_score_boundary.py
FACTOR_VALIDATOR=${REPO}/scripts/validate_relion_bpref_factor_capture.py
SCORE_VALIDATOR=${REPO}/scripts/validate_relion_fine_score_capture.py
OUTPUT=${RUN_ROOT}/analysis/K1_CASE04_TOP1_NATIVE_OPERAND_TARGET.json

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -x "${PYTHON}"
test -s "${STAGE_JOIN_JSON}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${SELECTOR}" | awk '{print $1}')" = "${EXPECTED_SELECTOR_SHA256}"
test "$(sha256sum "${BOUNDARY_ANALYZER}" | awk '{print $1}')" = "${EXPECTED_BOUNDARY_ANALYZER_SHA256}"
test "$(sha256sum "${FACTOR_VALIDATOR}" | awk '{print $1}')" = "${EXPECTED_FACTOR_VALIDATOR_SHA256}"
test "$(sha256sum "${SCORE_VALIDATOR}" | awk '{print $1}')" = "${EXPECTED_SCORE_VALIDATOR_SHA256}"
test ! -e "${OUTPUT}"

mkdir -p "${RUN_ROOT}/analysis" "${RUN_ROOT}/logs" "${RUN_ROOT}/provenance"
mkdir -p "${RUNTIME_ROOT}/tmp" "${RUNTIME_ROOT}/pixi_home" \
  "${RUNTIME_ROOT}/rattler_cache" "${RUNTIME_ROOT}/pycache"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1
export TMPDIR=${RUNTIME_ROOT}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/rattler_cache
export PYTHONPYCACHEPREFIX=${RUNTIME_ROOT}/pycache

cd "${REPO}"
git rev-parse HEAD > "${RUN_ROOT}/provenance/repo_head_${SLURM_JOB_ID}.txt"
git status --short > "${RUN_ROOT}/provenance/repo_status_${SLURM_JOB_ID}.txt"
env | sort > "${RUN_ROOT}/provenance/environment_${SLURM_JOB_ID}.txt"
sha256sum "${LAUNCHER}" "${SELECTOR}" "${BOUNDARY_ANALYZER}" \
  "${FACTOR_VALIDATOR}" "${SCORE_VALIDATOR}" "${STAGE_JOIN_JSON}" \
  > "${RUN_ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"

start_epoch=$(date +%s)
"${PYTHON}" -m scripts.select_k1_native_operand_target \
  --stage-join-json "${STAGE_JOIN_JSON}" --output-json "${OUTPUT}" \
  > "${RUN_ROOT}/logs/select_${SLURM_JOB_ID}.stdout"
wall_s=$(( $(date +%s) - start_epoch ))

"${PYTHON}" - "${OUTPUT}" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
assert report["schema"] == "recovar.em.k1_native_operand_target.v1"
assert report["status"] == "complete"
assert report["stack_index_one_based"] == 31560
assert report["native_rotation_local"] >= 0
assert report["native_translation_id"] >= 0
PY

sha256sum "${OUTPUT}" > "${RUN_ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
printf '{"job_id":"%s","wall_s":%d,"stage_join_job":13031009}\n' \
  "${SLURM_JOB_ID}" "${wall_s}" > "${RUN_ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
