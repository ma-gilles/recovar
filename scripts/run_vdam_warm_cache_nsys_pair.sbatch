#!/usr/bin/env bash
#SBATCH --job-name=vdam-warm-nsys
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=01:15:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-warm-nsys-%j.out

set -euo pipefail

: "${OUTPUT_ROOT:?set a new SAFE_TO_DELETE pair root}"
: "${DATA_DIR:?set the frozen materialized fixture data directory}"
: "${RELION_OPTIMISER:?set the incoming RELION optimiser checkpoint}"
: "${VDAM_SCORECARD:?set the frozen scorecard}"
: "${CASE_ID:?set the frozen K=1 case id}"
: "${EXPECTED_REPO_HEAD:?pin the tracked source head}"
: "${EXPECTED_CUDA_SHA256:?pin the qualified RECOVAR CUDA binary}"
: "${EXPECTED_RELION_BIND_SHA256:?pin the qualified RELION binding binary}"

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
PIXI_PY=${PIXI_PY_OVERRIDE:-${REPO_ROOT}/.pixi/envs/default/bin/python}
TARGET_GPU_UUID=${TARGET_GPU_UUID:-}
TARGET_ITERATION=${TARGET_ITERATION:-80}
NR_ITER_SCHEDULE=${NR_ITER_SCHEDULE:-200}
TARGET_ORIGINAL_INDEX=${TARGET_ORIGINAL_INDEX:-1002}
TARGET_STACK_INDEX=${TARGET_STACK_INDEX:-1003}
TARGET_RELION_PERTURBATION_SOURCE=${TARGET_RELION_PERTURBATION_SOURCE:-}
VDAM_NSYS_BIN=${VDAM_NSYS_BIN:-/opt/nvidia/nsight-systems/2025.3.2/target-linux-x64/nsys}
VDAM_NSYS_DELAY_S=${VDAM_NSYS_DELAY_S:-120}
# The target exits before this ceiling in normal runs.  A 120-second trace can
# terminate an 80-iteration target before its required final checkpoint, so the
# warm-trajectory wrapper uses the already-qualified full-run window.
VDAM_NSYS_DURATION_S=${VDAM_NSYS_DURATION_S:-600}

test ! -e "${OUTPUT_ROOT}"
test -x "${PIXI_PY}"
test -x "${VDAM_NSYS_BIN}"
test -s "${DATA_DIR}/particles.star"
test -s "${RELION_OPTIMISER}"
test -s "${VDAM_SCORECARD}"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"

source "${REPO_ROOT}/scripts/vdam_gpu_selection.sh"
vdam_select_target_gpu "${TARGET_GPU_UUID}" 0
selected_gpu_uuid=${VDAM_SELECTED_GPU_UUID}

shared_cache=${OUTPUT_ROOT}_jax_cache
test ! -e "${shared_cache}"
mkdir -p "${OUTPUT_ROOT}/provenance" "${shared_cache}"
touch "${OUTPUT_ROOT}/SAFE_TO_DELETE" "${shared_cache}/SAFE_TO_DELETE"
git -C "${REPO_ROOT}" rev-parse HEAD > "${OUTPUT_ROOT}/provenance/repo_head.txt"
printf '%s\n' "${selected_gpu_uuid}" > "${OUTPUT_ROOT}/provenance/selected_gpu_uuid.txt"
printf '%s\n' "${VDAM_VISIBLE_GPU_UUIDS_CSV}" > "${OUTPUT_ROOT}/provenance/allocated_gpu_uuids.csv"
nvidia-smi -q > "${OUTPUT_ROOT}/provenance/nvidia_smi.txt"
env | LC_ALL=C sort > "${OUTPUT_ROOT}/provenance/submission_environment.txt"

export RECOVAR_INITIALMODEL_SKIP_EXPECTED_ACCURACY=0
export RECOVAR_INITIALMODEL_EXPECTED_ACCURACY_SUBPROCESS=0

for repeat in 1 2; do
  run_root=$(printf '%s/repeat-%02d' "${OUTPUT_ROOT}" "${repeat}")
  mkdir -p "${run_root}"
  touch "${run_root}/SAFE_TO_DELETE"
  nsys_output=
  if [[ "${repeat}" == 2 ]]; then
    nsys_output=${run_root}/nsight/vdam_warm
  fi
  printf '%02d\t%s\n' "${repeat}" "${nsys_output:-unprofiled}" \
    >> "${OUTPUT_ROOT}/provenance/execution_order.tsv"

  OUTPUT_ROOT="${run_root}" \
  DATA_DIR="${DATA_DIR}" \
  RELION_OPTIMISER="${RELION_OPTIMISER}" \
  REPO_ROOT="${REPO_ROOT}" \
  PIXI_PY_OVERRIDE="${PIXI_PY}" \
  EXPECTED_REPO_HEAD="${EXPECTED_REPO_HEAD}" \
  EXPECTED_CUDA_SHA256="${EXPECTED_CUDA_SHA256}" \
  EXPECTED_RELION_BIND_SHA256="${EXPECTED_RELION_BIND_SHA256}" \
  TARGET_GPU_UUID="${selected_gpu_uuid}" \
  TARGET_ITERATION="${TARGET_ITERATION}" \
  NR_ITER_SCHEDULE="${NR_ITER_SCHEDULE}" \
  VDAM_SCORECARD="${VDAM_SCORECARD}" \
  CASE_ID="${CASE_ID}" \
  TARGET_ORIGINAL_INDEX="${TARGET_ORIGINAL_INDEX}" \
  TARGET_RELION_PART_ID=-1 \
  TARGET_STACK_INDEX="${TARGET_STACK_INDEX}" \
  TARGET_RELION_PERTURBATION_SOURCE="${TARGET_RELION_PERTURBATION_SOURCE}" \
  CAPTURE_LOCAL_SCORE=0 \
  CAPTURE_FUSED_SCORES=0 \
  CAPTURE_COARSE_SCORE=0 \
  CAPTURE_NATIVE_REPLAY=0 \
  BOUNDARY_DETERMINISTIC_CUDA=0 \
  VDAM_JAX_COMPILATION_CACHE_DIR="${shared_cache}" \
  VDAM_NSYS_OUTPUT="${nsys_output}" \
  VDAM_NSYS_BIN="${VDAM_NSYS_BIN}" \
  VDAM_NSYS_DELAY_S="${VDAM_NSYS_DELAY_S}" \
  VDAM_NSYS_DURATION_S="${VDAM_NSYS_DURATION_S}" \
    bash "${REPO_ROOT}/scripts/run_vdam_first_state_boundary_capture.sbatch"

  vdam_verify_selected_gpu "${selected_gpu_uuid}"
done

unset RECOVAR_INITIALMODEL_SKIP_EXPECTED_ACCURACY
unset RECOVAR_INITIALMODEL_EXPECTED_ACCURACY_SUBPROCESS
touch "${OUTPUT_ROOT}/COMPLETED"
