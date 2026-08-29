#!/usr/bin/env bash
#SBATCH --job-name=vdam-accuracy-ab
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-accuracy-ab-%j.out

set -euo pipefail

: "${OUTPUT_ROOT:?set a new SAFE_TO_DELETE panel root}"
: "${DATA_DIR:?set the frozen materialized fixture data directory}"
: "${RELION_OPTIMISER:?set the incoming RELION optimiser checkpoint}"
: "${VDAM_SCORECARD:?set the frozen scorecard}"
: "${CASE_ID:?set the frozen K=1 case id}"
: "${EXPECTED_REPO_HEAD:?pin the tracked source head}"
: "${EXPECTED_CUDA_SHA256:?pin the qualified RECOVAR CUDA binary}"
: "${EXPECTED_RELION_BIND_SHA256:?pin the qualified RELION binding binary}"
: "${TARGET_GPU_UUID:?pin the physical H100 used by the native envelope}"

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
PIXI_PY=${PIXI_PY_OVERRIDE:-${REPO_ROOT}/.pixi/envs/default/bin/python}
REPEATS=${REPEATS:-4}
TARGET_ITERATION=${TARGET_ITERATION:-4}
NR_ITER_SCHEDULE=${NR_ITER_SCHEDULE:-200}
TARGET_ORIGINAL_INDEX=${TARGET_ORIGINAL_INDEX:-285}
TARGET_STACK_INDEX=${TARGET_STACK_INDEX:-286}
TARGET_RELION_PERTURBATION_SOURCE=${TARGET_RELION_PERTURBATION_SOURCE:-}

case "${REPEATS}" in
  ''|*[!0-9]*) echo "REPEATS must be a positive integer" >&2; exit 2 ;;
esac
test "${REPEATS}" -ge 1
test ! -e "${OUTPUT_ROOT}"
test -x "${PIXI_PY}"
test -s "${DATA_DIR}/particles.star"
test -s "${RELION_OPTIMISER}"
test -s "${VDAM_SCORECARD}"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"

visible_gpu_uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
if [[ "${visible_gpu_uuid}" != "${TARGET_GPU_UUID}" ]]; then
  echo "VDAM_TARGET_GPU_MISS expected=${TARGET_GPU_UUID} observed=${visible_gpu_uuid}" >&2
  exit 75
fi

mkdir -p "${OUTPUT_ROOT}/provenance"
touch "${OUTPUT_ROOT}/SAFE_TO_DELETE"
git -C "${REPO_ROOT}" rev-parse HEAD > "${OUTPUT_ROOT}/provenance/repo_head.txt"
nvidia-smi -q > "${OUTPUT_ROOT}/provenance/nvidia_smi.txt"
env | LC_ALL=C sort > "${OUTPUT_ROOT}/provenance/submission_environment.txt"

for mode in baseline skip; do
  for repeat in $(seq 1 "${REPEATS}"); do
    run_root=$(printf '%s/%s/repeat-%02d' "${OUTPUT_ROOT}" "${mode}" "${repeat}")
    mkdir -p "${run_root}"
    touch "${run_root}/SAFE_TO_DELETE"
    if [[ "${mode}" == skip ]]; then
      export RECOVAR_INITIALMODEL_SKIP_EXPECTED_ACCURACY=1
    else
      export RECOVAR_INITIALMODEL_SKIP_EXPECTED_ACCURACY=0
    fi
    export RECOVAR_INITIALMODEL_EXPECTED_ACCURACY_SUBPROCESS=0

    OUTPUT_ROOT="${run_root}" \
    DATA_DIR="${DATA_DIR}" \
    RELION_OPTIMISER="${RELION_OPTIMISER}" \
    REPO_ROOT="${REPO_ROOT}" \
    PIXI_PY_OVERRIDE="${PIXI_PY}" \
    EXPECTED_REPO_HEAD="${EXPECTED_REPO_HEAD}" \
    EXPECTED_CUDA_SHA256="${EXPECTED_CUDA_SHA256}" \
    EXPECTED_RELION_BIND_SHA256="${EXPECTED_RELION_BIND_SHA256}" \
    TARGET_GPU_UUID="${TARGET_GPU_UUID}" \
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
    BOUNDARY_DETERMINISTIC_CUDA=1 \
      bash "${REPO_ROOT}/scripts/run_vdam_first_state_boundary_capture.sbatch"

    printf '%s\n' "${mode}" > "${run_root}/provenance/expected_accuracy_mode.txt"
  done
done

unset RECOVAR_INITIALMODEL_SKIP_EXPECTED_ACCURACY
unset RECOVAR_INITIALMODEL_EXPECTED_ACCURACY_SUBPROCESS
touch "${OUTPUT_ROOT}/COMPLETED"
