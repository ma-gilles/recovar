#!/usr/bin/env bash
#SBATCH --job-name=vdam-candidate-envelope
#SBATCH --account=gilles
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-candidate-envelope-%j.out

set -euo pipefail

: "${CANDIDATE_ROOT:?set the completed candidate root}"
: "${PANEL_ROOT:?set the completed native-repeat panel root}"
: "${FIXTURE_DIR:?set the frozen materialized fixture directory}"
: "${VDAM_SCORECARD:?set the frozen suite scorecard}"
: "${CASE_ID:?set the frozen case id}"
: "${PIXEL_SIZE:?set the particle pixel size in Angstrom}"

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
PIXI_PY=${PIXI_PY_OVERRIDE:-${REPO_ROOT}/.pixi/envs/default/bin/python}
VDAM_REPEAT_COUNT=${VDAM_REPEAT_COUNT:-4}
ANALYSIS_DIR=${ANALYSIS_DIR:-${CANDIDATE_ROOT}/analysis/native_envelope}
VDAM_MARK_CANDIDATE_COMPLETE=${VDAM_MARK_CANDIDATE_COMPLETE:-0}

test -x "${PIXI_PY}"
test -s "${VDAM_SCORECARD}"
test -f "${CANDIDATE_ROOT}/SAFE_TO_DELETE"
test -f "${PANEL_ROOT}/SAFE_TO_DELETE"
test -s "${FIXTURE_DIR}/fixture_materialization.json"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"
if [[ -n "${EXPECTED_REPO_HEAD:-}" ]]; then
    test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
fi
if [[ ! "${VDAM_REPEAT_COUNT}" =~ ^[0-9]+$ ]] || (( VDAM_REPEAT_COUNT < 2 )); then
    echo "VDAM_REPEAT_COUNT must be an integer of at least two" >&2
    exit 2
fi
case "${VDAM_MARK_CANDIDATE_COMPLETE}" in
    0|1) ;;
    *) echo "VDAM_MARK_CANDIDATE_COMPLETE must be 0 or 1" >&2; exit 2 ;;
esac

native_args=()
for repeat_index in $(seq 1 "${VDAM_REPEAT_COUNT}"); do
    native_root=${PANEL_ROOT}/repeat-$(printf '%02d' "${repeat_index}")/${CASE_ID}
    test -s "${native_root}/trajectory_audit.json"
    test -s "${native_root}/particle_state_trajectory_audit.json"
    test -s "${native_root}/run_provenance.json"
    test -s "${native_root}/paired_gpu_uuid.json"
    native_args+=(--native-root "${native_root}")
done

mkdir -p "${ANALYSIS_DIR}"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER JAX_PLATFORM_NAME JAX_PLATFORMS
export PYTHONNOUSERSITE=1
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${ANALYSIS_DIR}/runtime/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${ANALYSIS_DIR}/runtime/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${ANALYSIS_DIR}/runtime/${SLURM_JOB_ID}/rattler_cache
export JAX_COMPILATION_CACHE_DIR=${ANALYSIS_DIR}/runtime/${SLURM_JOB_ID}/jax_cpu_cache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" "${JAX_COMPILATION_CACHE_DIR}"

set +e
"${PIXI_PY}" -m scripts.audit_vdam_candidate_native_envelope \
    --scorecard "${VDAM_SCORECARD}" --case-id "${CASE_ID}" \
    --candidate-root "${CANDIDATE_ROOT}" "${native_args[@]}" \
    --fixture-dir "${FIXTURE_DIR}" \
    --output-json "${ANALYSIS_DIR}/candidate_native_envelope.json" \
    --output-shells-npz "${ANALYSIS_DIR}/candidate_native_envelope_shells.npz"
map_status=$?
"${PIXI_PY}" -m scripts.audit_vdam_candidate_state_envelope \
    --scorecard "${VDAM_SCORECARD}" --case-id "${CASE_ID}" \
    --candidate-root "${CANDIDATE_ROOT}" "${native_args[@]}" \
    --fixture-dir "${FIXTURE_DIR}" --pixel-size "${PIXEL_SIZE}" \
    --output-json "${ANALYSIS_DIR}/candidate_state_envelope.json"
state_status=$?
set -e

test -s "${ANALYSIS_DIR}/candidate_native_envelope.json"
test -s "${ANALYSIS_DIR}/candidate_native_envelope_shells.npz"
test -s "${ANALYSIS_DIR}/candidate_state_envelope.json"
"${PIXI_PY}" - "${ANALYSIS_DIR}" "${map_status}" "${state_status}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
status = {
    "schema": "recovar.vdam_candidate_envelope_job.v1",
    "map_exit_status": int(sys.argv[2]),
    "state_exit_status": int(sys.argv[3]),
}
status["result"] = "pass" if not status["map_exit_status"] and not status["state_exit_status"] else "fail"
(root / "candidate_envelope_status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
print(json.dumps(status, indent=2, sort_keys=True))
PY
if (( map_status != 0 || state_status != 0 )); then
    exit 1
fi
if [[ "${VDAM_MARK_CANDIDATE_COMPLETE}" == 1 ]]; then
    test -d "${CANDIDATE_ROOT}/provenance"
    sha256sum \
        "${ANALYSIS_DIR}/candidate_native_envelope.json" \
        "${ANALYSIS_DIR}/candidate_native_envelope_shells.npz" \
        "${ANALYSIS_DIR}/candidate_state_envelope.json" \
        "${ANALYSIS_DIR}/candidate_envelope_status.json" \
        > "${CANDIDATE_ROOT}/provenance/analysis_sha256.txt"
    touch "${CANDIDATE_ROOT}/COMPLETED"
fi
