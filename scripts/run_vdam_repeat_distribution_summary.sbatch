#!/usr/bin/env bash
#SBATCH --job-name=vdam-repeat-distribution
#SBATCH --account=gilles
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-repeat-distribution-%j.out

set -euo pipefail

: "${PANEL_ROOT:?set the completed paired repeat-panel root}"
: "${CASE_ID:?set the frozen VDAM case id}"
: "${EXPECTED_ANALYZER_HEAD:?pin the analysis source head}"
: "${EXPECTED_SCIENCE_HEAD:?pin the trajectory source head}"

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
PIXI_PY=${PIXI_PY_OVERRIDE:-${REPO_ROOT}/.pixi/envs/default/bin/python}
VDAM_SCORECARD=${VDAM_SCORECARD:-${REPO_ROOT}/docs/math/vdam_k1_gui_default_full_suite_v1.json}
VDAM_REPEAT_COUNT=${VDAM_REPEAT_COUNT:-4}
ANALYSIS_TAG=$(git -C "${REPO_ROOT}" rev-parse --short=10 HEAD)
ANALYSIS_ROOT=${PANEL_ROOT}/analysis/${ANALYSIS_TAG}
MAP_REPORT=${ANALYSIS_ROOT}/repeat_map_distribution.json
MAP_SHELLS=${ANALYSIS_ROOT}/repeat_map_distribution_shells.npz
STATE_REPORT=${ANALYSIS_ROOT}/repeat_state_distribution.json
STATUS_REPORT=${ANALYSIS_ROOT}/repeat_distribution_status.json

test -f "${PANEL_ROOT}/SAFE_TO_DELETE"
test -x "${PIXI_PY}"
test -s "${VDAM_SCORECARD}"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_ANALYZER_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"
for repeat_index in $(seq 1 "${VDAM_REPEAT_COUNT}"); do
    repeat_root=${PANEL_ROOT}/repeat-$(printf '%02d' "${repeat_index}")/${CASE_ID}
    test -s "${repeat_root}/trajectory_audit.json"
    test -s "${repeat_root}/particle_state_trajectory_audit.json"
    test -s "${repeat_root}/run_provenance.json"
    test -s "${repeat_root}/paired_gpu_uuid.json"
    test -s "${repeat_root}/relion/relion.timing.json"
    test -s "${repeat_root}/recovar/recovar.timing.json"
    test "$("${PIXI_PY}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["git_head"])' "${repeat_root}/run_provenance.json")" = "${EXPECTED_SCIENCE_HEAD}"
done

if [[ -e "${ANALYSIS_ROOT}" ]]; then
    test -z "$(find "${ANALYSIS_ROOT}" -mindepth 1 -print -quit)"
else
    mkdir -p "${ANALYSIS_ROOT}"
fi

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1
export PYTHONPATH=${REPO_ROOT}
export TMPDIR=${ANALYSIS_ROOT}/runtime/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${ANALYSIS_ROOT}/runtime/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${ANALYSIS_ROOT}/runtime/${SLURM_JOB_ID}/rattler_cache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}"

set +e
"${PIXI_PY}" -m scripts.audit_vdam_repeat_panel \
    --scorecard "${VDAM_SCORECARD}" --case-id "${CASE_ID}" \
    --panel-root "${PANEL_ROOT}" --repeat-count "${VDAM_REPEAT_COUNT}" \
    --output-json "${MAP_REPORT}" --output-shells-npz "${MAP_SHELLS}"
map_status=$?
"${PIXI_PY}" -m scripts.audit_vdam_repeat_state_panel \
    --scorecard "${VDAM_SCORECARD}" --case-id "${CASE_ID}" \
    --panel-root "${PANEL_ROOT}" --repeat-count "${VDAM_REPEAT_COUNT}" \
    --output-json "${STATE_REPORT}"
state_status=$?
set -e

"${PIXI_PY}" - "${MAP_REPORT}" "${STATE_REPORT}" "${STATUS_REPORT}" \
    "${map_status}" "${state_status}" "${EXPECTED_SCIENCE_HEAD}" \
    "${EXPECTED_ANALYZER_HEAD}" "${SLURM_JOB_ID}" <<'PY'
import json
import sys
from pathlib import Path

map_report = json.loads(Path(sys.argv[1]).read_text())
state_report = json.loads(Path(sys.argv[2]).read_text())
status = {
    "schema": "recovar.vdam_repeat_distribution_job.v1",
    "result": "pass" if sys.argv[4:6] == ["0", "0"] else "fail",
    "map_result": map_report["result"],
    "state_result": state_report["result"],
    "map_exit_status": int(sys.argv[4]),
    "state_exit_status": int(sys.argv[5]),
    "science_head": sys.argv[6],
    "analyzer_head": sys.argv[7],
    "job_id": sys.argv[8],
}
Path(sys.argv[3]).write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
print(json.dumps(status, indent=2, sort_keys=True))
PY

sha256sum \
    "${MAP_REPORT}" "${MAP_SHELLS}" "${STATE_REPORT}" "${STATUS_REPORT}" \
    "${REPO_ROOT}/scripts/audit_vdam_repeat_panel.py" \
    "${REPO_ROOT}/scripts/audit_vdam_repeat_state_panel.py" \
    "${REPO_ROOT}/scripts/audit_vdam_candidate_state_envelope.py" \
    "${REPO_ROOT}/scripts/audit_vdam_sampling_trajectory.py" \
    "${REPO_ROOT}/scripts/run_vdam_repeat_distribution_summary.sbatch" \
    > "${ANALYSIS_ROOT}/evidence.sha256"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_ANALYZER_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"

if (( map_status != 0 || state_status != 0 )); then
    touch "${ANALYSIS_ROOT}/ANALYSIS_FAILED"
    exit 1
fi
touch "${ANALYSIS_ROOT}/ANALYSIS_COMPLETED"
