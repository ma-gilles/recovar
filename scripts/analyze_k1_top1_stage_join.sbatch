#!/usr/bin/env bash
#SBATCH --job-name=k1-top1-stage-join
#SBATCH --partition=cpu
#SBATCH --account=gilles
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=01:00:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${MANIFEST:?MANIFEST is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_ANALYZER_SHA256:?EXPECTED_ANALYZER_SHA256 is required}"
: "${EXPECTED_STAGE_ANALYZER_SHA256:?EXPECTED_STAGE_ANALYZER_SHA256 is required}"
: "${EXPECTED_MANIFEST_SHA256:?EXPECTED_MANIFEST_SHA256 is required}"

REPO=${REPO:-/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_native_units_instrumentation_20260826}
PYTHON=${PYTHON:-${REPO}/.pixi/envs/default/bin/python}
LAUNCHER=${REPO}/scripts/analyze_k1_top1_stage_join.sbatch
ANALYZER=${REPO}/scripts/analyze_k1_top1_stage_join.py
STAGE_ANALYZER=${REPO}/scripts/analyze_k1_fine_score_stages.py
OUTPUT=${RUN_ROOT}/analysis/K1_CASE04_TOP1_STAGE_JOIN.json

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -x "${PYTHON}"
test -s "${MANIFEST}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${ANALYZER}" | awk '{print $1}')" = "${EXPECTED_ANALYZER_SHA256}"
test "$(sha256sum "${STAGE_ANALYZER}" | awk '{print $1}')" = "${EXPECTED_STAGE_ANALYZER_SHA256}"
test "$(sha256sum "${MANIFEST}" | awk '{print $1}')" = "${EXPECTED_MANIFEST_SHA256}"
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
sha256sum "${LAUNCHER}" "${ANALYZER}" "${STAGE_ANALYZER}" "${MANIFEST}" \
  > "${RUN_ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"

start_epoch=$(date +%s)
"${PYTHON}" -m scripts.analyze_k1_top1_stage_join \
  --manifest "${MANIFEST}" --physical-image-size 256 --top-count 64 \
  --output-json "${OUTPUT}" \
  > "${RUN_ROOT}/logs/analyze_${SLURM_JOB_ID}.stdout"
wall_s=$(( $(date +%s) - start_epoch ))

"${PYTHON}" - "${OUTPUT}" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
assert report["schema"] == "recovar.em.k1_top1_stage_join.v1"
assert report["status"] == "complete"
assert report["target"]["source_row_zero_based"] == 31559
assert report["target"]["stack_index_one_based"] == 31560
assert report["stage_analysis"]["stack_index_one_based"] == 31560
PY

sha256sum "${OUTPUT}" > "${RUN_ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
printf '{"job_id":"%s","wall_s":%d,"native_job":13030505,"recovar_job":13030753}\n' \
  "${SLURM_JOB_ID}" "${wall_s}" > "${RUN_ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
