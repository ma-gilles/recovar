#!/usr/bin/env bash
#SBATCH --job-name=vdam-graph-pair
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-graph-pair-%j.out

set -euo pipefail

: "${OUTPUT_ROOT:?set a new SAFE_TO_DELETE pair root}"
: "${DATA_DIR:?set the frozen materialized fixture data directory}"
: "${RELION_OPTIMISER:?set the incoming RELION optimiser checkpoint}"
: "${VDAM_SCORECARD:?set the frozen scorecard}"
: "${CASE_ID:?set the frozen K=1 case id}"
: "${EXPECTED_REPO_HEAD:?pin the tracked source head}"
: "${EXPECTED_CUDA_SHA256:?pin the candidate RECOVAR CUDA binary}"
: "${EXPECTED_RELION_BIND_SHA256:?pin the qualified RELION binding binary}"
: "${RECOVAR_CUDA_LIB_OVERRIDE:?set the unique CUDA binary from the focused gate}"
: "${TARGET_RELION_PERTURBATION_SOURCE:?pin the full-precision perturbation source}"

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
PIXI_PY=${PIXI_PY_OVERRIDE:-${REPO_ROOT}/.pixi/envs/default/bin/python}
TARGET_GPU_UUID=${TARGET_GPU_UUID:-}
TARGET_ITERATION=${TARGET_ITERATION:-80}
NR_ITER_SCHEDULE=${NR_ITER_SCHEDULE:-200}
TARGET_ORIGINAL_INDEX=${TARGET_ORIGINAL_INDEX:-1002}
TARGET_STACK_INDEX=${TARGET_STACK_INDEX:-1003}

for value_name in TARGET_ITERATION NR_ITER_SCHEDULE; do
  value=${!value_name}
  case "${value}" in
    ''|*[!0-9]*) echo "${value_name} must be a positive integer" >&2; exit 2 ;;
  esac
  test "${value}" -ge 1
done
test "${NR_ITER_SCHEDULE}" -ge "${TARGET_ITERATION}"
test ! -e "${OUTPUT_ROOT}"
test -x "${PIXI_PY}"
test -s "${DATA_DIR}/particles.star"
test -s "${RELION_OPTIMISER}"
test -s "${VDAM_SCORECARD}"
test -s "${RECOVAR_CUDA_LIB_OVERRIDE}"
test -s "${TARGET_RELION_PERTURBATION_SOURCE}"
test "$(sha256sum "${RECOVAR_CUDA_LIB_OVERRIDE}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
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
printf 'control\t0\ncandidate\t1\n' > "${OUTPUT_ROOT}/provenance/arms.tsv"

export RECOVAR_INITIALMODEL_SKIP_EXPECTED_ACCURACY=0
export RECOVAR_INITIALMODEL_EXPECTED_ACCURACY_SUBPROCESS=0
# Pin the accepted GF46 execution topology in the runner itself.  The graph
# flag below is the only intended difference between arms; do not depend on a
# submitter's interactive environment to reconstruct the control.
export RECOVAR_EXACT_LOCAL_SOURCE_BPREF_FUSED_SERIAL_PARTICLES=1
export RECOVAR_EXACT_LOCAL_SOURCE_BPREF_LAUNCH_SERIAL_ROTATIONS=1
export RECOVAR_VDAM_PRECOMPUTE_ORDERED_RESIDUALS=1
export RECOVAR_VDAM_FIXED_WARP_ORDER_SCATTER=1
export RECOVAR_K1_RELION_WAVG_SEQUENTIAL_CUDA=1
export RECOVAR_EXACT_LOCAL_BUCKET_RADIX=4
export RECOVAR_INITIAL_MODEL_PROFILE=1
unset RECOVAR_EXACT_LOCAL_SOURCE_BPREF_FUSED_SERIAL_ROTATIONS
unset RECOVAR_VDAM_PREPROJECT_PERSISTENT_ROTATIONS
unset RECOVAR_VDAM_PRECOMPUTE_PERSISTENT_RESIDUALS
unset RECOVAR_VDAM_EXACT_NATIVE_PTX
env | LC_ALL=C sort > "${OUTPUT_ROOT}/provenance/runtime_environment.txt"

# Warm the shared JAX executable palette in both orders, then time the graph
# before the control so cache placement and monotonic node temperature cannot
# make the candidate look artificially faster.
for phase_and_arm in \
  warmup:control \
  warmup:candidate \
  timed:candidate \
  timed:control; do
  phase=${phase_and_arm%%:*}
  arm=${phase_and_arm##*:}
  if [[ "${arm}" == control ]]; then
    graph_enabled=0
  else
    graph_enabled=1
  fi
  run_root=${OUTPUT_ROOT}/${phase}/${arm}
  mkdir -p "${run_root}"
  touch "${run_root}/SAFE_TO_DELETE"
  printf '%s\t%s\t%s\n' "${phase}" "${arm}" "${graph_enabled}" \
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
  RECOVAR_VDAM_ORDERED_SCATTER_CUDA_GRAPH="${graph_enabled}" \
    bash "${REPO_ROOT}/scripts/run_vdam_first_state_boundary_capture.sbatch"

  vdam_verify_selected_gpu "${selected_gpu_uuid}"
done

unset RECOVAR_INITIALMODEL_SKIP_EXPECTED_ACCURACY
unset RECOVAR_INITIALMODEL_EXPECTED_ACCURACY_SUBPROCESS
touch "${OUTPUT_ROOT}/COMPLETED"
