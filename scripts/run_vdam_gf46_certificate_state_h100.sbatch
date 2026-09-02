#!/usr/bin/env bash
#SBATCH --job-name=vdam-cert-gf46
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-cert-gf46-%j.out

set -euo pipefail

: "${EXPECTED_REPO_HEAD:?pin the committed qualification head}"
: "${EXPECTED_REPO_TREE:?pin the committed qualification tree}"
: "${EXPECTED_SOURCE_MANIFEST_SHA256:?pin the selected source-manifest SHA-256}"
: "${TARGET_GPU_UUID:?pin the physical H100 UUID}"
: "${RECOVAR_PIXI_PY:?set the absolute pixi-environment Python executable}"

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
REPO_ROOT=$(realpath -e -- "${REPO_ROOT}")
RECOVAR_PIXI_PY=$(realpath -e -- "${RECOVAR_PIXI_PY}")
RESULT_ROOT=${VDAM_GF46_CERTIFICATE_RESULT_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf46_certificate_state_h100_${SLURM_JOB_ID}_${EXPECTED_REPO_HEAD:0:9}}
RESULT_ROOT=$(realpath -m -- "${RESULT_ROOT}")
RUNTIME_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/vdam_gf46_certificate_state_h100_${SLURM_JOB_ID}

[[ "${EXPECTED_REPO_HEAD}" =~ ^[0-9a-f]{40}$ ]]
[[ "${EXPECTED_REPO_TREE}" =~ ^[0-9a-f]{40}$ ]]
[[ "${EXPECTED_SOURCE_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]]
[[ "${TARGET_GPU_UUID}" =~ ^GPU-[0-9a-fA-F-]{36}$ ]]
test -x "${RECOVAR_PIXI_PY}"
test "$(git -C "${REPO_ROOT}" rev-parse --show-toplevel)" = "${REPO_ROOT}"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${REPO_ROOT}" rev-parse 'HEAD^{tree}')" = "${EXPECTED_REPO_TREE}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=all)"
test ! -e "${RESULT_ROOT}"
test -d "$(dirname "${RESULT_ROOT}")"
case "${RESULT_ROOT}/" in
  "${REPO_ROOT}/"*)
    echo "result root must be outside the repository: ${RESULT_ROOT}" >&2
    exit 2
    ;;
esac
test ! -e "${RUNTIME_ROOT}"

initial_allocation_spec=${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}}
source "${REPO_ROOT}/scripts/vdam_gpu_selection.sh"
vdam_assert_target_gpu_allocated "${TARGET_GPU_UUID}" "${initial_allocation_spec}"
vdam_select_target_gpu "${TARGET_GPU_UUID}" 0
selected_gpu_uuid=${VDAM_SELECTED_GPU_UUID}
test "${selected_gpu_uuid}" = "${TARGET_GPU_UUID}"
gpu_name=$(nvidia-smi -i "${selected_gpu_uuid}" --query-gpu=name --format=csv,noheader | sed -n '1p' | sed 's/[[:space:]]*$//')
[[ "${gpu_name}" == *H100* ]]

mkdir -p "${RUNTIME_ROOT}/jax_cache" "${RUNTIME_ROOT}/tmp"
touch "${RUNTIME_ROOT}/SAFE_TO_DELETE"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV JAX_PLATFORM_NAME
unset RECOVAR_CUDA_LIB RECOVAR_ENABLE_CUSTOM_CUDA RECOVAR_REQUIRE_CUSTOM_CUDA_FOR_TESTS
unset XLA_FLAGS
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${selected_gpu_uuid}
export JAX_ENABLE_X64=1
# RECOVAR initializes a CPU control device at import time.  Keep CUDA first so
# it remains the default backend while retaining that required CPU backend.
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=${RUNTIME_ROOT}/jax_cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export TMPDIR=${RUNTIME_ROOT}/tmp

actual_source_manifest_sha256=$(
  cd "${REPO_ROOT}"
  "${RECOVAR_PIXI_PY}" -m scripts.qualify_vdam_gf46_certificate_state_h100 \
    --print-source-manifest-sha256
)
test "${actual_source_manifest_sha256}" = "${EXPECTED_SOURCE_MANIFEST_SHA256}"

on_error() {
  local status=$?
  trap - ERR
  printf 'GF46 certificate-state H100 qualification failed: status=%s line=%s command=%s\n' \
    "${status}" "${BASH_LINENO[0]}" "${BASH_COMMAND}" >&2
  exit "${status}"
}
trap on_error ERR

cd "${REPO_ROOT}"
"${RECOVAR_PIXI_PY}" -m scripts.qualify_vdam_gf46_certificate_state_h100 \
  --output-root "${RESULT_ROOT}" \
  --expected-repo-head "${EXPECTED_REPO_HEAD}" \
  --expected-repo-tree "${EXPECTED_REPO_TREE}" \
  --expected-source-manifest-sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}" \
  --expected-gpu-uuid "${TARGET_GPU_UUID}" \
  --warmup-runs 2 \
  --timed-runs 5

vdam_verify_selected_gpu "${selected_gpu_uuid}"
test -f "${RESULT_ROOT}/SAFE_TO_DELETE"
test -f "${RESULT_ROOT}/COMPLETED"
test -f "${RESULT_ROOT}/SHA256SUMS"
test -f "${RESULT_ROOT}/SHA256SUMS.sha256"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${REPO_ROOT}" rev-parse 'HEAD^{tree}')" = "${EXPECTED_REPO_TREE}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=all)"

printf 'GF46 certificate-state H100 qualification complete: %s\n' "${RESULT_ROOT}"
