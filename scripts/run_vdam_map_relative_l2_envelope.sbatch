#!/usr/bin/env bash
#SBATCH --job-name=vdam-map-l2-envelope
#SBATCH --account=gilles
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-map-l2-envelope-%j.out

set -euo pipefail

: "${CANDIDATE_ROOT:?set the completed candidate root}"
: "${PANEL_ROOT:?set the completed native-repeat panel root}"
: "${CASE_ID:?set the case directory name below each repeat}"
: "${OUTPUT_JSON:?set a new diagnostic report path}"
: "${EXPECTED_REPO_HEAD:?pin the diagnostic source head}"

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
PIXI_PY=${PIXI_PY_OVERRIDE:-${REPO_ROOT}/.pixi/envs/default/bin/python}
VDAM_REPEAT_COUNT=${VDAM_REPEAT_COUNT:-4}

test -x "${PIXI_PY}"
test -s "${CANDIDATE_ROOT}/trajectory_audit.json"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"
test ! -e "${OUTPUT_JSON}"
if [[ ! "${VDAM_REPEAT_COUNT}" =~ ^[0-9]+$ ]] || (( VDAM_REPEAT_COUNT < 2 )); then
  echo "VDAM_REPEAT_COUNT must be an integer of at least two" >&2
  exit 2
fi

native_args=()
for repeat_index in $(seq 1 "${VDAM_REPEAT_COUNT}"); do
  native_root=${PANEL_ROOT}/repeat-$(printf '%02d' "${repeat_index}")/${CASE_ID}
  test -s "${native_root}/trajectory_audit.json"
  native_args+=(--native-root "${native_root}")
done

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${TMPDIR:-/scratch/gpfs/GILLES/mg6942/tmp/${SLURM_JOB_ID}}
export PIXI_HOME=${PIXI_HOME:-/scratch/gpfs/GILLES/mg6942/pixi_home/${SLURM_JOB_ID}}
export RATTLER_CACHE_DIR=${RATTLER_CACHE_DIR:-/scratch/gpfs/GILLES/mg6942/rattler_cache/${SLURM_JOB_ID}}
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" "$(dirname "${OUTPUT_JSON}")"

"${PIXI_PY}" -m scripts.analyze_vdam_map_relative_l2_envelope \
  --candidate-root "${CANDIDATE_ROOT}" "${native_args[@]}" \
  --output-json "${OUTPUT_JSON}"

test -s "${OUTPUT_JSON}"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"
sha256sum "${OUTPUT_JSON}" \
  "${REPO_ROOT}/scripts/analyze_vdam_map_relative_l2_envelope.py" \
  "${REPO_ROOT}/scripts/run_vdam_map_relative_l2_envelope.sbatch" \
  > "${OUTPUT_JSON%.json}.sha256"
