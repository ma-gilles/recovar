#!/usr/bin/env bash
#SBATCH --job-name=vdam-repeat-panel
#SBATCH --account=amits
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-repeat-panel-%j.out

set -euo pipefail

: "${CASE_ID:?submit with CASE_ID and OUTPUT_ROOT}"
: "${OUTPUT_ROOT:?submit with CASE_ID and OUTPUT_ROOT}"

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
VDAM_SCORECARD="${VDAM_SCORECARD:-${REPO_ROOT}/docs/math/vdam_k1_long_trajectory_suite_v1.json}"
VDAM_REPEAT_COUNT="${VDAM_REPEAT_COUNT:-4}"
if [[ ! "${VDAM_REPEAT_COUNT}" =~ ^[0-9]+$ ]] || (( VDAM_REPEAT_COUNT < 2 )); then
  echo "VDAM_REPEAT_COUNT must be an integer of at least two" >&2
  exit 2
fi
PANEL_ROOT="${OUTPUT_ROOT}"
export REPO_ROOT VDAM_SCORECARD
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${PANEL_ROOT}/jax_cache}"
mkdir -p "${PANEL_ROOT}" "${JAX_COMPILATION_CACHE_DIR}"
touch "${PANEL_ROOT}/SAFE_TO_DELETE"

for repeat_index in $(seq 1 "${VDAM_REPEAT_COUNT}"); do
  repeat_root="${PANEL_ROOT}/repeat-$(printf '%02d' "${repeat_index}")"
  export CASE_ID OUTPUT_ROOT="${repeat_root}"
  set +e
  bash "${REPO_ROOT}/scripts/run_vdam_relion_parity_case.sbatch"
  repeat_status=$?
  set -e
  audit_path="${repeat_root}/${CASE_ID}/trajectory_audit.json"
  if (( repeat_status != 0 )) && [[ ! -s "${audit_path}" ]]; then
    echo "repeat ${repeat_index} failed before producing trajectory evidence" >&2
    exit "${repeat_status}"
  fi
done

export OUTPUT_ROOT="${PANEL_ROOT}"
PIXI_PY="${REPO_ROOT}/.pixi/envs/default/bin/python"
"${PIXI_PY}" "${REPO_ROOT}/scripts/audit_vdam_repeat_panel.py" \
  --scorecard "${VDAM_SCORECARD}" --case-id "${CASE_ID}" \
  --panel-root "${OUTPUT_ROOT}" --repeat-count "${VDAM_REPEAT_COUNT}" \
  --output-json "${OUTPUT_ROOT}/repeat_panel_audit.json" \
  --output-shells-npz "${OUTPUT_ROOT}/repeat_panel_shellwise_fsc.npz"
