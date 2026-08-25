#!/usr/bin/env bash
#SBATCH --job-name=vdam-repeat-summary
#SBATCH --account=gilles
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-repeat-summary-%j.out

set -euo pipefail

: "${CASE_ID:?set the frozen VDAM case id}"
: "${OUTPUT_ROOT:?set the completed repeat panel root}"

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
PIXI_PY=${PIXI_PY_OVERRIDE:-${REPO_ROOT}/.pixi/envs/default/bin/python}
VDAM_SCORECARD=${VDAM_SCORECARD:-${REPO_ROOT}/docs/math/vdam_k1_gui_default_full_suite_v1.json}
VDAM_REPEAT_COUNT=${VDAM_REPEAT_COUNT:-4}
REPORT=${OUTPUT_ROOT}/repeat_panel_audit.json
SHELLS=${OUTPUT_ROOT}/repeat_panel_shellwise_fsc.npz

test -f "${OUTPUT_ROOT}/SAFE_TO_DELETE"
test -s "${VDAM_SCORECARD}"
test -x "${PIXI_PY}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"
for repeat_index in $(seq 1 "${VDAM_REPEAT_COUNT}"); do
    repeat_root=${OUTPUT_ROOT}/repeat-$(printf '%02d' "${repeat_index}")/${CASE_ID}
    test -s "${repeat_root}/trajectory_audit.json"
    test -s "${repeat_root}/run_provenance.json"
    test -s "${repeat_root}/paired_gpu_uuid.json"
done

if [[ -s "${REPORT}" && -s "${SHELLS}" ]]; then
    "${PIXI_PY}" - "${REPORT}" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
if report.get("schema") != "recovar.vdam_relion_repeat_panel.v2":
    raise SystemExit("existing repeat-panel report has the wrong schema")
print(json.dumps(report, indent=2, sort_keys=True))
raise SystemExit(0 if report.get("result") == "pass" else 1)
PY
fi

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1
export TMPDIR=${OUTPUT_ROOT}/summary_runtime/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${OUTPUT_ROOT}/summary_runtime/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${OUTPUT_ROOT}/summary_runtime/${SLURM_JOB_ID}/rattler_cache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}"

cd "${REPO_ROOT}"
"${PIXI_PY}" -m scripts.audit_vdam_repeat_panel \
    --scorecard "${VDAM_SCORECARD}" --case-id "${CASE_ID}" \
    --panel-root "${OUTPUT_ROOT}" --repeat-count "${VDAM_REPEAT_COUNT}" \
    --output-json "${REPORT}" --output-shells-npz "${SHELLS}"
