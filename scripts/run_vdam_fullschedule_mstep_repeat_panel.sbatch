#!/usr/bin/env bash
#SBATCH --job-name=vdam-mstep-repeat-panel
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-mstep-repeat-panel-%j.out

set -euo pipefail

: "${PANEL_ROOT:?set a new SAFE_TO_DELETE output root}"
: "${DATA_DIR:?set the frozen materialized VDAM fixture directory}"
: "${EXPECTED_REPO_HEAD:?pin the tracked source head}"
: "${EXPECTED_CUDA_SHA256:?pin the qualified RECOVAR CUDA binary}"
: "${EXPECTED_RELION_BIND_SHA256:?pin the qualified RELION binding binary}"
: "${VDAM_RELION_BINARY:?set the isolated patched non-MPI RELION binary}"
: "${RECOVAR_CUDA_LIB_SOURCE:?set the immutable qualified CUDA binary}"

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
PIXI_PY=${PIXI_PY_OVERRIDE:-${REPO_ROOT}/.pixi/envs/default/bin/python}
TARGET_ITERATION=${TARGET_ITERATION:-1}
NR_ITER_SCHEDULE=${NR_ITER_SCHEDULE:-200}
RELION_THREADS=${RELION_THREADS:-8}
REPORT=${PANEL_ROOT}/analysis/mstep_repeat_panel.json

test -f "${PANEL_ROOT}/SAFE_TO_DELETE"
test -x "${PIXI_PY}"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"
test "$(sha256sum "${RECOVAR_CUDA_LIB_SOURCE}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
test -z "$(find "${PANEL_ROOT}" -mindepth 1 ! -name SAFE_TO_DELETE -print -quit)"

gpu_uuid_before=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -n 1 | tr -d '[:space:]')
for arm in a b; do
  arm_root=${PANEL_ROOT}/${arm}
  mkdir -p "${arm_root}"
  touch "${arm_root}/SAFE_TO_DELETE"
  env \
    OUTPUT_ROOT="${arm_root}" \
    DATA_DIR="${DATA_DIR}" \
    EXPECTED_REPO_HEAD="${EXPECTED_REPO_HEAD}" \
    EXPECTED_CUDA_SHA256="${EXPECTED_CUDA_SHA256}" \
    EXPECTED_RELION_BIND_SHA256="${EXPECTED_RELION_BIND_SHA256}" \
    VDAM_RELION_BINARY="${VDAM_RELION_BINARY}" \
    RECOVAR_CUDA_LIB_SOURCE="${RECOVAR_CUDA_LIB_SOURCE}" \
    REPO_ROOT="${REPO_ROOT}" \
    PIXI_PY_OVERRIDE="${PIXI_PY}" \
    TARGET_ITERATION="${TARGET_ITERATION}" \
    NR_ITER_SCHEDULE="${NR_ITER_SCHEDULE}" \
    RELION_THREADS="${RELION_THREADS}" \
    RELION_VDAM_WORKER_TRACE_REPLAY="${RELION_VDAM_WORKER_TRACE_REPLAY:-0}" \
    RELION_VDAM_BLOCK_TRACE_CAPTURE="${RELION_VDAM_BLOCK_TRACE_CAPTURE:-${RELION_VDAM_BLOCK_TRACE_REPLAY:-0}}" \
    RELION_VDAM_BLOCK_TRACE_REPLAY="${RELION_VDAM_BLOCK_TRACE_REPLAY:-0}" \
    RELION_VDAM_BLOCK_TRACE_CAPACITY="${RELION_VDAM_BLOCK_TRACE_CAPACITY:-1000000}" \
    RELION_VDAM_BLOCK_TRACE_SM_COUNT="${RELION_VDAM_BLOCK_TRACE_SM_COUNT:-132}" \
    RECOVAR_VDAM_CANDIDATE_BLOCK_TRACE_CAPTURE="${RECOVAR_VDAM_CANDIDATE_BLOCK_TRACE_CAPTURE:-0}" \
    RECOVAR_VDAM_CANDIDATE_BLOCK_TRACE_CAPACITY="${RECOVAR_VDAM_CANDIDATE_BLOCK_TRACE_CAPACITY:-1000000}" \
    RECOVAR_VDAM_CANDIDATE_BLOCK_MAP_CAPTURE="${RECOVAR_VDAM_CANDIDATE_BLOCK_MAP_CAPTURE:-0}" \
    RECOVAR_VDAM_CANDIDATE_BLOCK_MAP_CAPACITY="${RECOVAR_VDAM_CANDIDATE_BLOCK_MAP_CAPACITY:-1000000}" \
    RECOVAR_RELION_VDAM_WORKER_REPLAY_TOPOLOGY="${RECOVAR_RELION_VDAM_WORKER_REPLAY_TOPOLOGY:-captured}" \
    bash "${REPO_ROOT}/scripts/run_vdam_fullschedule_mstep_boundary.sbatch"
  test -f "${arm_root}/RUN_SUCCESS_${SLURM_JOB_ID}"
  test "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -n 1 | tr -d '[:space:]')" = "${gpu_uuid_before}"
done

mkdir -p "$(dirname "${REPORT}")"
"${PIXI_PY}" -m scripts.analyze_vdam_mstep_repeat_panel \
  --arm-a "${PANEL_ROOT}/a" --arm-b "${PANEL_ROOT}/b" \
  --iteration "${TARGET_ITERATION}" --output-json "${REPORT}"
if [[ "${RECOVAR_VDAM_CANDIDATE_BLOCK_MAP_CAPTURE:-0}" = 1 ]]; then
  "${PIXI_PY}" -m scripts.analyze_vdam_mapped_block_chronology \
    --panel-root "${PANEL_ROOT}" \
    --output-json "${PANEL_ROOT}/analysis/mapped_block_chronology_panel.json"
  test -s "${PANEL_ROOT}/analysis/mapped_block_chronology_panel.json"
fi

test -s "${REPORT}"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"
test "$(sha256sum "${RECOVAR_CUDA_LIB_SOURCE}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
sha256sum "${REPORT}" \
  "${REPO_ROOT}/scripts/analyze_vdam_mstep_repeat_panel.py" \
  "${REPO_ROOT}/scripts/analyze_vdam_mapped_block_chronology.py" \
  "${REPO_ROOT}/scripts/run_vdam_fullschedule_mstep_boundary.sbatch" \
  "${REPO_ROOT}/scripts/run_vdam_fullschedule_mstep_repeat_panel.sbatch" \
  > "${PANEL_ROOT}/analysis/evidence.sha256"
