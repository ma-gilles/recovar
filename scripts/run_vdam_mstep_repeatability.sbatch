#!/usr/bin/env bash
#SBATCH --job-name=vdam-mstep-repeat
#SBATCH --account=amits
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-mstep-repeat-%j.out

set -euo pipefail

: "${OUTPUT_ROOT:?submit with OUTPUT_ROOT=/scratch/...}"

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
VDAM_SCORECARD="${VDAM_SCORECARD:-${REPO_ROOT}/docs/math/vdam_relion_parity_scorecard_v1.json}"
CASE_ID="${CASE_ID:-vdam-01}"
PIXI_PY="${REPO_ROOT}/.pixi/envs/default/bin/python"
mkdir -p "${OUTPUT_ROOT}"
touch "${OUTPUT_ROOT}/SAFE_TO_DELETE"
BASE_OUTPUT_ROOT="${OUTPUT_ROOT}"

for arm in a b; do
  arm_root="${BASE_OUTPUT_ROOT}/${arm}"
  export OUTPUT_ROOT="${arm_root}/run"
  export REPO_ROOT VDAM_SCORECARD CASE_ID
  export RECOVAR_DEBUG_DUMP_DIR="${arm_root}/native_mstep"
  export RECOVAR_DEBUG_DUMP_MSTEP_ITER=1
  export RECOVAR_MSTEP_DUMP_DIR="${arm_root}/recovar_mstep"
  export RECOVAR_MSTEP_DUMP_ITER=1
  export RECOVAR_INITIAL_MODEL_ACCUM_DUMP_DIR="${arm_root}/recovar_accum"
  mkdir -p "${RECOVAR_DEBUG_DUMP_DIR}" "${RECOVAR_MSTEP_DUMP_DIR}" \
    "${RECOVAR_INITIAL_MODEL_ACCUM_DUMP_DIR}"
  bash "${REPO_ROOT}/scripts/run_vdam_relion_parity_case.sbatch"
done

"${PIXI_PY}" "${REPO_ROOT}/scripts/analyze_vdam_mstep_repeatability.py" \
  --native-a "${BASE_OUTPUT_ROOT}/a/native_mstep" \
  --native-b "${BASE_OUTPUT_ROOT}/b/native_mstep" \
  --recovar-a "${BASE_OUTPUT_ROOT}/a/recovar_mstep" \
  --recovar-b "${BASE_OUTPUT_ROOT}/b/recovar_mstep" \
  --output-json "${BASE_OUTPUT_ROOT}/mstep_repeatability.json"
