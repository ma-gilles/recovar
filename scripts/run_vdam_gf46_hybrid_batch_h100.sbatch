#!/usr/bin/env bash
#SBATCH --job-name=vdam-hybrid-gf46
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-hybrid-gf46-%j.out

set -euo pipefail
unset PYTHONOPTIMIZE

: "${EXPECTED_REPO_HEAD:?pin the committed hybrid qualification head}"
: "${EXPECTED_REPO_TREE:?pin the committed hybrid qualification tree}"
: "${EXPECTED_SOURCE_MANIFEST_SHA256:?pin the hybrid source-manifest SHA-256}"
: "${RECOVAR_PIXI_PY:?set the absolute pixi-environment Python executable}"

readonly CUDA_TOOLKIT=/usr/local/cuda-12.6
readonly CUDA_ARCH_FLAGS='-gencode arch=compute_90,code=sm_90'
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
REPO_ROOT=$(realpath -e -- "${REPO_ROOT}")
RECOVAR_PIXI_PY=$(realpath -e -- "${RECOVAR_PIXI_PY}")
RESULT_ROOT=${VDAM_GF46_HYBRID_RESULT_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf46_hybrid_batch_h100_${SLURM_JOB_ID}_${EXPECTED_REPO_HEAD:0:9}}
RESULT_ROOT=$(realpath -m -- "${RESULT_ROOT}")
RUNTIME_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/vdam_gf46_hybrid_batch_h100_${SLURM_JOB_ID}
CUDA_BINARY=${RUNTIME_ROOT}/libcuda_backproject.so
NVCC=${CUDA_TOOLKIT}/bin/nvcc

[[ "${EXPECTED_REPO_HEAD}" =~ ^[0-9a-f]{40}$ ]]
[[ "${EXPECTED_REPO_TREE}" =~ ^[0-9a-f]{40}$ ]]
[[ "${EXPECTED_SOURCE_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]]
test -x "${RECOVAR_PIXI_PY}"
test -x "${NVCC}"
test "$(git -C "${REPO_ROOT}" rev-parse --show-toplevel)" = "${REPO_ROOT}"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${REPO_ROOT}" rev-parse 'HEAD^{tree}')" = "${EXPECTED_REPO_TREE}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=all)"
test ! -e "${RESULT_ROOT}"
test -d "$(dirname "${RESULT_ROOT}")"
test ! -e "${RUNTIME_ROOT}"
case "${RESULT_ROOT}/" in
  "${REPO_ROOT}/"*)
    echo "result root must be outside the repository: ${RESULT_ROOT}" >&2
    exit 2
    ;;
esac

initial_allocation_spec=${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}}
source "${REPO_ROOT}/scripts/vdam_gpu_selection.sh"
mapfile -t initially_visible_gpu_uuids < <(vdam_visible_gpu_uuids)
if [[ "${#initially_visible_gpu_uuids[@]}" -ne 1 ]]; then
  echo "hybrid qualification requires exactly one cgroup-visible GPU" >&2
  exit 76
fi
target_gpu_uuid=${initially_visible_gpu_uuids[0]}
vdam_assert_target_gpu_allocated "${target_gpu_uuid}" "${initial_allocation_spec}"
vdam_select_target_gpu "${target_gpu_uuid}" 0
selected_gpu_uuid=${VDAM_SELECTED_GPU_UUID}
gpu_name=$(nvidia-smi -i "${selected_gpu_uuid}" --query-gpu=name --format=csv,noheader | sed -n '1p' | sed 's/[[:space:]]*$//')
[[ "${gpu_name}" == *H100* ]]

mkdir -p "${RUNTIME_ROOT}/jax_cache" "${RUNTIME_ROOT}/tmp"
touch "${RUNTIME_ROOT}/SAFE_TO_DELETE"
make -B -C "${REPO_ROOT}/recovar/cuda" \
  "PYTHON=${RECOVAR_PIXI_PY}" \
  "NVCC=${NVCC}" \
  "CUDA_ARCH=${CUDA_ARCH_FLAGS}" \
  "LIB=${CUDA_BINARY}" \
  > "${RUNTIME_ROOT}/cuda_build.stdout" \
  2> "${RUNTIME_ROOT}/cuda_build.stderr"
test -s "${CUDA_BINARY}"
nm -D --defined-only "${CUDA_BINARY}" > "${RUNTIME_ROOT}/cuda_binary_symbols.txt"
rg -q 'RelionCoarseDiff2RectangularF32' "${RUNTIME_ROOT}/cuda_binary_symbols.txt"
rg -q 'RelionCoarseDiff2RotationBlocksF32' "${RUNTIME_ROOT}/cuda_binary_symbols.txt"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV JAX_PLATFORM_NAME
unset RECOVAR_DISABLE_CUDA RECOVAR_CUDA_LIB RECOVAR_ENABLE_CUSTOM_CUDA
unset RECOVAR_REQUIRE_CUSTOM_CUDA_FOR_TESTS XLA_FLAGS
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=${REPO_ROOT}
export CUDA_VISIBLE_DEVICES=${selected_gpu_uuid}
export RECOVAR_CUDA_LIB=${CUDA_BINARY}
export RECOVAR_REQUIRE_CUSTOM_CUDA_FOR_TESTS=1
export RECOVAR_ENABLE_CUSTOM_CUDA=1
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=${RUNTIME_ROOT}/jax_cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export TMPDIR=${RUNTIME_ROOT}/tmp
export PATH=${CUDA_TOOLKIT}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_TOOLKIT}/lib64:${CUDA_TOOLKIT}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}

actual_source_manifest_sha256=$(
  cd "${REPO_ROOT}"
  "${RECOVAR_PIXI_PY}" -m scripts.qualify_vdam_gf46_hybrid_batch_h100 \
    --print-source-manifest-sha256
)
test "${actual_source_manifest_sha256}" = "${EXPECTED_SOURCE_MANIFEST_SHA256}"

on_error() {
  local status=$?
  trap - ERR
  printf 'GF46 hybrid H100 qualification failed: status=%s line=%s command=%s\n' \
    "${status}" "${BASH_LINENO[0]}" "${BASH_COMMAND}" >&2
  exit "${status}"
}
trap on_error ERR

cd "${REPO_ROOT}"
"${RECOVAR_PIXI_PY}" -m scripts.qualify_vdam_gf46_hybrid_batch_h100 \
  --output-root "${RESULT_ROOT}" \
  --expected-repo-head "${EXPECTED_REPO_HEAD}" \
  --expected-repo-tree "${EXPECTED_REPO_TREE}" \
  --expected-source-manifest-sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}" \
  --expected-gpu-uuid "${selected_gpu_uuid}" \
  --timed-runs 3

vdam_verify_selected_gpu "${selected_gpu_uuid}"
test -f "${RESULT_ROOT}/SAFE_TO_DELETE"
test -f "${RESULT_ROOT}/COMPLETED"
test -f "${RESULT_ROOT}/SHA256SUMS"
test -f "${RESULT_ROOT}/SHA256SUMS.sha256"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${REPO_ROOT}" rev-parse 'HEAD^{tree}')" = "${EXPECTED_REPO_TREE}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=all)"

printf 'GF46 hybrid H100 qualification complete: %s\n' "${RESULT_ROOT}"
