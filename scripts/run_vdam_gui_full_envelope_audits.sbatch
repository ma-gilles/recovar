#!/usr/bin/env bash
#SBATCH --job-name=vdam-gui-audit
#SBATCH --account=gilles
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --array=2-22%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-gui-audit-%A_%a.out

set -euo pipefail

: "${SCIENCE_ROOT:?set the completed GUI-default envelope science root}"
: "${SLURM_ARRAY_TASK_ID:?submit as a Slurm array}"

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
PIXI_PY=${PIXI_PY_OVERRIDE:-${REPO_ROOT}/.pixi/envs/default/bin/python}
VDAM_SCORECARD=${VDAM_SCORECARD:-${REPO_ROOT}/docs/math/vdam_k1_gui_default_full_suite_v1.json}
VDAM_REPEAT_COUNT=${VDAM_REPEAT_COUNT:-4}
EXPECTED_REPO_HEAD=${EXPECTED_REPO_HEAD:-}

if (( SLURM_ARRAY_TASK_ID < 2 || SLURM_ARRAY_TASK_ID > 22 )); then
    echo "VDAM envelope audit task must be in [2, 22]" >&2
    exit 2
fi
test -x "${PIXI_PY}"
test -s "${VDAM_SCORECARD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"
if [[ -n "${EXPECTED_REPO_HEAD}" ]]; then
    test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
fi

CASE_ID=$(printf 'vdam-gf%02d' "${SLURM_ARRAY_TASK_ID}")
PANEL_ROOT=${SCIENCE_ROOT}/${CASE_ID}
CANDIDATE_ROOT=${PANEL_ROOT}/repeat-01/${CASE_ID}
FIXTURE_DIR=${CANDIDATE_ROOT}/data
ANALYSIS_DIR=${PANEL_ROOT}/analysis/$(git -C "${REPO_ROOT}" rev-parse --short=10 HEAD)
test -f "${PANEL_ROOT}/SCIENCE_COMPLETED"
test -s "${PANEL_ROOT}/science_manifest.json"
test -s "${FIXTURE_DIR}/particles.star"
for repeat_index in $(seq 1 "${VDAM_REPEAT_COUNT}"); do
    repeat_root=${PANEL_ROOT}/repeat-$(printf '%02d' "${repeat_index}")/${CASE_ID}
    test -s "${repeat_root}/trajectory_audit.json"
    test -s "${repeat_root}/particle_state_trajectory_audit.json"
    test -s "${repeat_root}/run_provenance.json"
    test -s "${repeat_root}/paired_gpu_uuid.json"
done

PIXEL_SIZE=$("${PIXI_PY}" - "${FIXTURE_DIR}/particles.star" <<'PY'
import sys
from pathlib import Path

import starfile

value = starfile.read(Path(sys.argv[1]))
if not isinstance(value, dict):
    raise SystemExit("fixture STAR has no optics block")
optics = value.get("optics")
if optics is None or len(optics) == 0:
    raise SystemExit("fixture STAR optics block is empty")
matches = [column for column in optics.columns if str(column).lstrip("_") == "rlnImagePixelSize"]
if len(matches) != 1:
    raise SystemExit(f"expected one rlnImagePixelSize column, found {matches}")
values = sorted(set(float(item) for item in optics[matches[0]].tolist()))
if len(values) != 1 or values[0] <= 0:
    raise SystemExit(f"expected one positive pixel size, found {values}")
print(values[0])
PY
)

export CANDIDATE_ROOT PANEL_ROOT FIXTURE_DIR VDAM_SCORECARD CASE_ID PIXEL_SIZE
export VDAM_REPEAT_COUNT REPO_ROOT EXPECTED_REPO_HEAD PIXI_PY_OVERRIDE="${PIXI_PY}"
export ANALYSIS_DIR
bash "${REPO_ROOT}/scripts/run_vdam_candidate_envelope_audits.sbatch"

sha256sum \
    "${ANALYSIS_DIR}/candidate_native_envelope.json" \
    "${ANALYSIS_DIR}/candidate_native_envelope_shells.npz" \
    "${ANALYSIS_DIR}/candidate_state_envelope.json" \
    "${ANALYSIS_DIR}/candidate_envelope_status.json" \
    > "${ANALYSIS_DIR}/evidence.sha256"
