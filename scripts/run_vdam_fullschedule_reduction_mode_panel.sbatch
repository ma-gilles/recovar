#!/usr/bin/env bash
#SBATCH --job-name=vdam-reduction-panel
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
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-reduction-panel-%j.out

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
RANDOM_SEED=${RANDOM_SEED:-0}
TARGET_GPU_UUID=${TARGET_GPU_UUID:-}
REPORT_DIR=${PANEL_ROOT}/analysis
BOUNDARY_RUNNER=${REPO_ROOT}/scripts/run_vdam_fullschedule_mstep_boundary.sbatch
PAIR_ANALYZER=${REPO_ROOT}/scripts/analyze_vdam_mstep_repeat_panel.py

test -f "${PANEL_ROOT}/SAFE_TO_DELETE"
test -x "${PIXI_PY}"
test -s "${DATA_DIR}/particles.star"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"
test "$(sha256sum "${RECOVAR_CUDA_LIB_SOURCE}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
test -z "$(find "${PANEL_ROOT}" -mindepth 1 ! -name SAFE_TO_DELETE -print -quit)"

gpu_uuid_before=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -n 1 | tr -d '[:space:]')
if [[ -n "${TARGET_GPU_UUID}" && "${gpu_uuid_before}" != "${TARGET_GPU_UUID}" ]]; then
  echo "VDAM_TARGET_GPU_MISS expected=${TARGET_GPU_UUID} observed=${gpu_uuid_before}" >&2
  exit 75
fi
mkdir -p "${REPORT_DIR}"

run_repeat() {
  mode=$1
  repeat=$2
  arm_root=${PANEL_ROOT}/${mode}/${repeat}
  mkdir -p "${arm_root}"
  touch "${arm_root}/SAFE_TO_DELETE"

  mode_env=(
    env
    -u RECOVAR_RELION_X_HALF_BP_PER_PARTICLE_LAUNCH
    -u RECOVAR_RELION_X_HALF_SEQUENTIAL_TRANSLATION_REDUCTION
    -u RECOVAR_RELION_X_HALF_BP_FUSED_ATOMICS
    -u RECOVAR_RELION_X_HALF_BP_BLOCK_TOPOLOGY
  )
  case "${mode}" in
    production)
      ;;
    fused_block)
      mode_env+=(
        RECOVAR_RELION_X_HALF_BP_PER_PARTICLE_LAUNCH=1
        RECOVAR_RELION_X_HALF_BP_FUSED_ATOMICS=1
        RECOVAR_RELION_X_HALF_BP_BLOCK_TOPOLOGY=1
      )
      ;;
    sequential_fused_block)
      mode_env+=(
        RECOVAR_RELION_X_HALF_BP_PER_PARTICLE_LAUNCH=1
        RECOVAR_RELION_X_HALF_SEQUENTIAL_TRANSLATION_REDUCTION=1
        RECOVAR_RELION_X_HALF_BP_FUSED_ATOMICS=1
        RECOVAR_RELION_X_HALF_BP_BLOCK_TOPOLOGY=1
      )
      ;;
    *)
      echo "unknown reduction mode: ${mode}" >&2
      exit 2
      ;;
  esac

  "${mode_env[@]}" \
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
    RANDOM_SEED="${RANDOM_SEED}" \
    bash "${BOUNDARY_RUNNER}"
  test -f "${arm_root}/RUN_SUCCESS_${SLURM_JOB_ID}"
  test "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -n 1 | tr -d '[:space:]')" = "${gpu_uuid_before}"
}

IFS=, read -r -a reduction_modes <<< "${REDUCTION_MODES:-production,fused_block,sequential_fused_block}"
for mode in "${reduction_modes[@]}"; do
  for repeat in a b; do
    run_repeat "${mode}" "${repeat}"
  done
  "${PIXI_PY}" -m scripts.analyze_vdam_mstep_repeat_panel \
    --arm-a "${PANEL_ROOT}/${mode}/a" \
    --arm-b "${PANEL_ROOT}/${mode}/b" \
    --iteration "${TARGET_ITERATION}" \
    --output-json "${REPORT_DIR}/${mode}.json"
done

test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"
test "$(sha256sum "${RECOVAR_CUDA_LIB_SOURCE}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
printf '{"schema":"recovar.vdam_reduction_mode_panel.v1","job_id":"%s","physical_gpu_uuid":"%s","target_gpu_uuid":"%s","target_iteration":%d,"nr_iter_schedule":%d,"random_seed":%d,"modes":"%s"}\n' \
  "${SLURM_JOB_ID}" "${gpu_uuid_before}" "${TARGET_GPU_UUID}" "${TARGET_ITERATION}" "${NR_ITER_SCHEDULE}" \
  "${RANDOM_SEED}" "$(IFS=,; printf '%s' "${reduction_modes[*]}")" \
  > "${REPORT_DIR}/run.json"
sha256sum "${BOUNDARY_RUNNER}" "${PAIR_ANALYZER}" "${BASH_SOURCE[0]}" \
  "${REPORT_DIR}"/*.json > "${REPORT_DIR}/evidence.sha256"
touch "${PANEL_ROOT}/RUN_SUCCESS_${SLURM_JOB_ID}"
