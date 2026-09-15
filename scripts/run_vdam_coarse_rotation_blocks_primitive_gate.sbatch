#!/usr/bin/env bash
#SBATCH --job-name=vdam-rot-blocks
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-rot-blocks-%j.out

set -euo pipefail
unset PYTHONOPTIMIZE

: "${EXPECTED_REPO_HEAD:?pin the committed primitive head}"
: "${EXPECTED_REPO_TREE:?pin the committed primitive tree}"
: "${PIXI_PY_OVERRIDE:?pin a qualified RECOVAR pixi interpreter}"
: "${VDAM_ROTATION_BLOCKS_RESULT_ROOT:?set a new disposable output root}"

readonly CUDA_TOOLKIT=/usr/local/cuda-12.6
readonly CUDA_ARCH_FLAGS='-gencode arch=compute_90,code=sm_90'
readonly CPU_TESTS=(
  tests/unit/test_cuda_lib_symbol_check.py
  tests/unit/test_cuda_relion_fine_diff2.py::test_relion_coarse_diff2_cuda_source_pins_production_topology
  tests/unit/test_cuda_relion_fine_diff2.py::test_relion_coarse_diff2_fails_closed_without_gpu
  tests/unit/test_cuda_relion_fine_diff2.py::test_relion_coarse_diff2_rotation_blocks_fails_closed_without_gpu
  tests/unit/test_cuda_relion_fine_diff2.py::test_relion_coarse_diff2_rotation_blocks_rejects_noninteger_ids
)
readonly GPU_TESTS=(
  tests/unit/test_cuda_relion_fine_diff2.py::test_relion_coarse_diff2_rectangular_matches_atomic_envelope
  tests/unit/test_cuda_relion_fine_diff2.py::test_relion_coarse_diff2_rotation_blocks_matches_atomic_envelope
)
readonly SOURCE_FILES=(
  recovar/cuda/Makefile
  recovar/cuda/cuda_backproject.cu
  recovar/cuda_backproject.py
  scripts/run_vdam_coarse_rotation_blocks_primitive_gate.sbatch
  tests/conftest.py
  tests/unit/test_cuda_lib_symbol_check.py
  tests/unit/test_cuda_relion_fine_diff2.py
  pixi.lock
  pixi.toml
  pyproject.toml
)

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
REPO_ROOT=$(realpath -e -- "${REPO_ROOT}")
PIXI_PY=$(realpath -e -- "${PIXI_PY_OVERRIDE}")
RESULT_ROOT=$(realpath -m -- "${VDAM_ROTATION_BLOCKS_RESULT_ROOT}")
PROVENANCE=${RESULT_ROOT}/provenance
RESULTS=${RESULT_ROOT}/results
RUNTIME=${RESULT_ROOT}/runtime
CUDA_BINARY=${RUNTIME}/libcuda_backproject.so
NVCC=${CUDA_TOOLKIT}/bin/nvcc

on_error() {
  local status=$?
  local message="VDAM rotation-block primitive gate failed: status=${status} line=${BASH_LINENO[0]} command=${BASH_COMMAND}"
  trap - ERR
  printf '%s\n' "${message}" >&2
  if [[ -d "${PROVENANCE}" ]]; then
    printf '%s\n' "${message}" > "${PROVENANCE}/failure.txt"
  fi
  exit "${status}"
}
trap on_error ERR

[[ "${EXPECTED_REPO_HEAD}" =~ ^[0-9a-f]{40}$ ]]
[[ "${EXPECTED_REPO_TREE}" =~ ^[0-9a-f]{40}$ ]]
[[ "${RESULT_ROOT}" = /* ]]
test ! -e "${RESULT_ROOT}"
test -d "$(dirname "${RESULT_ROOT}")"
case "${RESULT_ROOT}/" in
  "${REPO_ROOT}/"*)
    echo "result root must be outside the repository: ${RESULT_ROOT}" >&2
    exit 2
    ;;
esac
test -x "${PIXI_PY}"
test -x "${NVCC}"
test "$(git -C "${REPO_ROOT}" rev-parse --show-toplevel)" = "${REPO_ROOT}"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${REPO_ROOT}" rev-parse 'HEAD^{tree}')" = "${EXPECTED_REPO_TREE}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=all)"

mkdir "${RESULT_ROOT}"
mkdir -p "${PROVENANCE}" "${RESULTS}" "${RUNTIME}"
touch "${RESULT_ROOT}/SAFE_TO_DELETE"

(
  cd "${REPO_ROOT}"
  sha256sum "${SOURCE_FILES[@]}"
) > "${PROVENANCE}/source_manifest.sha256"
git -C "${REPO_ROOT}" rev-parse HEAD > "${PROVENANCE}/repo_head.txt"
git -C "${REPO_ROOT}" rev-parse 'HEAD^{tree}' > "${PROVENANCE}/repo_tree.txt"
git -C "${REPO_ROOT}" symbolic-ref --short HEAD > "${PROVENANCE}/repo_branch.txt"
git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=all > "${PROVENANCE}/repo_status.txt"
sha256sum "${PIXI_PY}" > "${PROVENANCE}/interpreter.sha256"
"${NVCC}" --version > "${PROVENANCE}/nvcc_version.txt"
printf '%s\n' "${CUDA_ARCH_FLAGS}" > "${PROVENANCE}/cuda_arch_flags.txt"
printf '%s\n' "${SLURM_JOB_ID}" > "${PROVENANCE}/slurm_job_id.txt"
printf '%s\n' "${SLURMD_NODENAME:-$(hostname -s)}" > "${PROVENANCE}/node.txt"
nvidia-smi --query-gpu=uuid,name,driver_version --format=csv,noheader \
  > "${PROVENANCE}/allocated_gpus.csv"
scontrol show job "${SLURM_JOB_ID}" > "${PROVENANCE}/slurm_job.txt"

make -B -C "${REPO_ROOT}/recovar/cuda" \
  "PYTHON=${PIXI_PY}" \
  "NVCC=${NVCC}" \
  "CUDA_ARCH=${CUDA_ARCH_FLAGS}" \
  "LIB=${CUDA_BINARY}" \
  > "${PROVENANCE}/cuda_build.stdout" \
  2> "${PROVENANCE}/cuda_build.stderr"
test -s "${CUDA_BINARY}"
sha256sum "${CUDA_BINARY}" > "${PROVENANCE}/cuda_binary.sha256"
nm -D --defined-only "${CUDA_BINARY}" \
  > "${PROVENANCE}/cuda_binary_symbols.txt"
rg -q 'RelionCoarseDiff2RectangularF32' "${PROVENANCE}/cuda_binary_symbols.txt"
rg -q 'RelionCoarseDiff2RotationBlocksF32' "${PROVENANCE}/cuda_binary_symbols.txt"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset RECOVAR_DISABLE_CUDA RECOVAR_CUDA_LIB RECOVAR_ENABLE_CUSTOM_CUDA
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=${REPO_ROOT}
export RECOVAR_CUDA_LIB=${CUDA_BINARY}
export RECOVAR_REQUIRE_CUSTOM_CUDA_FOR_TESTS=1
export RECOVAR_ENABLE_CUSTOM_CUDA=1
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=${RUNTIME}/jax_cache
export TMPDIR=${RUNTIME}/tmp
mkdir -p "${JAX_COMPILATION_CACHE_DIR}" "${TMPDIR}"

(
  cd "${REPO_ROOT}"
  JAX_PLATFORMS=cpu "${PIXI_PY}" -m pytest -q "${CPU_TESTS[@]}"
) > "${RESULTS}/cpu_tests.stdout" 2> "${RESULTS}/cpu_tests.stderr"

(
  cd "${REPO_ROOT}"
  "${PIXI_PY}" -m pytest -q --run-gpu "${GPU_TESTS[@]}" \
    --junitxml="${RESULTS}/gpu_tests.junit.xml"
) > "${RESULTS}/gpu_tests.stdout" 2> "${RESULTS}/gpu_tests.stderr"

test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${REPO_ROOT}" rev-parse 'HEAD^{tree}')" = "${EXPECTED_REPO_TREE}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=all)"
sha256sum "${RESULTS}/cpu_tests.stdout" "${RESULTS}/gpu_tests.stdout" \
  "${RESULTS}/gpu_tests.junit.xml" > "${PROVENANCE}/results.sha256"
touch "${RESULT_ROOT}/COMPLETED"
