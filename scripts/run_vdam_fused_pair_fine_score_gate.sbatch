#!/usr/bin/env bash
#SBATCH --job-name=vdam-pair-fine
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-pair-fine-%j.out

set -euo pipefail

: "${OUTPUT_ROOT:?set a fresh SAFE_TO_DELETE output root}"
: "${EXPECTED_REPO_HEAD:?pin the committed source head}"

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR}}
PIXI_PY=${PIXI_PY_OVERRIDE:-${REPO_ROOT}/.pixi/envs/default/bin/python}
CUDA_BUILD_TOOLKIT=${CUDA_BUILD_TOOLKIT:-/usr/local/cuda-12.6}
CUDA_RUNTIME_TOOLKIT=${CUDA_RUNTIME_TOOLKIT:-/usr/local/cuda-13.1}
CUDA_ARCH_FLAGS=${CUDA_ARCH_FLAGS:--gencode arch=compute_90,code=sm_90}
PROVENANCE=${OUTPUT_ROOT}/provenance
RESULTS=${OUTPUT_ROOT}/results
RUNTIME=${OUTPUT_ROOT}/runtime
CUDA_BINARY=${RUNTIME}/cuda/libcuda_backproject.so

test ! -e "${OUTPUT_ROOT}"
test -x "${PIXI_PY}"
test -x "${CUDA_BUILD_TOOLKIT}/bin/nvcc"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1)"

mkdir -p "${PROVENANCE}" "${RESULTS}" "${RUNTIME}/cuda"
touch "${OUTPUT_ROOT}/SAFE_TO_DELETE"
git -C "${REPO_ROOT}" rev-parse HEAD > "${PROVENANCE}/repo_head.txt"
git -C "${REPO_ROOT}" symbolic-ref --short HEAD > "${PROVENANCE}/repo_branch.txt"
git -C "${REPO_ROOT}" status --porcelain=v1 > "${PROVENANCE}/repo_status.txt"
printf '%s\n' "${SLURM_JOB_ID}" > "${PROVENANCE}/slurm_job_id.txt"
printf '%s\n' "${SLURMD_NODENAME:-$(hostname -s)}" > "${PROVENANCE}/node.txt"
nvidia-smi -q > "${PROVENANCE}/nvidia_smi.txt"
"${CUDA_BUILD_TOOLKIT}/bin/nvcc" --version > "${PROVENANCE}/nvcc_version.txt"
(
  cd "${REPO_ROOT}"
  sha256sum \
    recovar/cuda/cuda_backproject.cu \
    recovar/cuda_backproject.py \
    tests/unit/test_cuda_relion_fine_diff2.py \
    scripts/benchmark_vdam_fused_pair_fine_score.py \
    scripts/run_vdam_fused_pair_fine_score_gate.sbatch
) > "${PROVENANCE}/source.sha256"

make -B -C "${REPO_ROOT}/recovar/cuda" \
  PYTHON="${PIXI_PY}" \
  NVCC="${CUDA_BUILD_TOOLKIT}/bin/nvcc" \
  "CUDA_ARCH=${CUDA_ARCH_FLAGS}" \
  LIB="${CUDA_BINARY}" \
  > "${PROVENANCE}/cuda_build.stdout" \
  2> "${PROVENANCE}/cuda_build.stderr"
test -s "${CUDA_BINARY}"
sha256sum "${CUDA_BINARY}" > "${PROVENANCE}/cuda_library.sha256"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV JAX_PLATFORM_NAME
unset RECOVAR_DISABLE_CUDA RECOVAR_CUDA_LIB RECOVAR_ENABLE_CUSTOM_CUDA
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export RECOVAR_CUDA_LIB=${CUDA_BINARY}
export RECOVAR_REQUIRE_CUSTOM_CUDA_FOR_TESTS=1
export RECOVAR_ENABLE_CUSTOM_CUDA=1
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.80
export JAX_COMPILATION_CACHE_DIR=${RUNTIME}/jax_cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export TMPDIR=${RUNTIME}/tmp
export PIXI_HOME=${RUNTIME}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/rattler_cache
export PATH=${CUDA_RUNTIME_TOOLKIT}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_RUNTIME_TOOLKIT}/lib64:${CUDA_RUNTIME_TOOLKIT}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
mkdir -p "${JAX_COMPILATION_CACHE_DIR}" "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}"
touch "${JAX_COMPILATION_CACHE_DIR}/SAFE_TO_DELETE"

(
  cd "${REPO_ROOT}"
  "${PIXI_PY}" - <<'PY'
import jax
from recovar import cuda_backproject

assert jax.default_backend() == "gpu"
assert len(jax.devices("gpu")) == 1
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
print(jax.devices("gpu"))
PY
) > "${PROVENANCE}/runtime_gate.stdout" 2> "${PROVENANCE}/runtime_gate.stderr"

(
  cd "${REPO_ROOT}"
  "${PIXI_PY}" -m pytest -q --run-gpu \
    --junitxml="${RESULTS}/pytest.junit.xml" \
    tests/unit/test_cuda_relion_fine_diff2.py::test_relion_fused_translate_pairs_match_rectangular_tree_bitwise \
    tests/unit/test_cuda_relion_fine_diff2.py::test_relion_runtime_fused_translate_pairs_reuse_physical_compile \
    tests/unit/test_cuda_relion_fine_diff2.py::test_relion_runtime_pairs_fails_closed_without_gpu
) > "${RESULTS}/pytest.stdout" 2> "${RESULTS}/pytest.stderr"

(
  cd "${REPO_ROOT}"
  "${PIXI_PY}" -m scripts.benchmark_vdam_fused_pair_fine_score \
    --output "${RESULTS}/benchmark.json"
) > "${RESULTS}/benchmark.stdout" 2> "${RESULTS}/benchmark.stderr"

"${PIXI_PY}" - "${RESULTS}/benchmark.json" "${EXPECTED_REPO_HEAD}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
assert payload["schema"] == "recovar.vdam_fused_pair_fine_score_microbenchmark.v1"
assert payload["git_head"] == sys.argv[2]
assert payload["all_bitwise_equal"] is True
assert all(case["speedup"] > 0.0 for case in payload["cases"])
PY

test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1)"
touch "${OUTPUT_ROOT}/SUCCESS"
