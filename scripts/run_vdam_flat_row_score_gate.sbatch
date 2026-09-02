#!/usr/bin/env bash
#SBATCH --job-name=vdam-flat-row-gate
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --nodelist=della-h21g4
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-flat-row-gate-%j.out

set -euo pipefail

: "${VDAM_FLAT_ROW_GATE_ROOT:?set a fresh disposable output root}"
: "${EXPECTED_REPO_HEAD:?pin the committed source head}"
: "${EXPECTED_SOURCE_MANIFEST_SHA256:?pin the source/test manifest}"
: "${EXPECTED_NODE_NAME:?pin the Slurm node}"
: "${TARGET_GPU_UUID:?pin the physical H100 UUID}"
: "${VDAM_FLAT_ROW_METADATA_ROOT:?point at the captured local-layout metadata}"

ROOT=${VDAM_FLAT_ROW_GATE_ROOT}
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
PIXI_PY=${PIXI_PY_OVERRIDE:-${REPO_ROOT}/.pixi/envs/default/bin/python}
CUDA_BUILD_TOOLKIT=${CUDA_BUILD_TOOLKIT:-/usr/local/cuda-12.6}
CUDA_RUNTIME_TOOLKIT=${CUDA_RUNTIME_TOOLKIT:-/usr/local/cuda-13.1}
CUDA_ARCH_FLAGS=${VDAM_FLAT_ROW_CUDA_ARCH_FLAGS:--gencode arch=compute_90,code=sm_90}
PROVENANCE=${ROOT}/provenance
RUNTIME=${ROOT}/runtime
RESULTS=${ROOT}/results
CUDA_BINARY=${RUNTIME}/cuda/libcuda_backproject.so
EXACT_JSON=${RESULTS}/flat_row_exactness.json
TIMING_JSON=${RESULTS}/flat_row_timing.json
JOB_TMP_ROOT=/scratch/gpfs/GILLES/mg6942/tmp/${SLURM_JOB_ID}
JOB_PIXI_ROOT=/scratch/gpfs/GILLES/mg6942/pixi_home/${SLURM_JOB_ID}
JOB_RATTLER_ROOT=/scratch/gpfs/GILLES/mg6942/rattler_cache/${SLURM_JOB_ID}
SOURCE_FILES=(
  recovar/cuda/Makefile
  recovar/cuda/cuda_backproject.cu
  recovar/cuda_backproject.py
  recovar/em/dense_single_volume/helpers/flat_local_rows.py
  recovar/em/dense_single_volume/helpers/sparse_pass2_bucketed.py
  recovar/em/dense_single_volume/local_layout.py
  scripts/benchmark_vdam_flat_row_fine_score.py
  scripts/run_vdam_flat_row_score_gate.sbatch
  scripts/vdam_gpu_selection.sh
  tests/conftest.py
  tests/unit/test_cuda_relion_fine_diff2.py
  tests/unit/test_flat_local_rows.py
  pixi.lock
  pixi.toml
  pyproject.toml
)

on_error() {
  local status=$?
  local message="VDAM flat-row gate failed: status=${status} line=${BASH_LINENO[0]} command=${BASH_COMMAND}"
  trap - ERR
  printf '%s\n' "${message}" >&2
  if [[ -d "${PROVENANCE}" ]]; then
    printf '%s\n' "${message}" > "${PROVENANCE}/failure.txt"
  fi
  exit "${status}"
}
trap on_error ERR

write_source_manifest() {
  (
    cd "${REPO_ROOT}"
    sha256sum "${SOURCE_FILES[@]}"
  )
}

test ! -e "${ROOT}"
test -d "${VDAM_FLAT_ROW_METADATA_ROOT}"
test -x "${PIXI_PY}"
test -x "${CUDA_BUILD_TOOLKIT}/bin/nvcc"
test -d "${CUDA_RUNTIME_TOOLKIT}/lib64"
test "${SLURMD_NODENAME:-$(hostname -s)}" = "${EXPECTED_NODE_NAME}"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1)"

source "${REPO_ROOT}/scripts/vdam_gpu_selection.sh"
vdam_select_target_gpu "${TARGET_GPU_UUID}" 0
selected_gpu_uuid=${VDAM_SELECTED_GPU_UUID}
test "${selected_gpu_uuid}" = "${TARGET_GPU_UUID}"
gpu_name=$(nvidia-smi --id="${selected_gpu_uuid}" --query-gpu=name --format=csv,noheader | sed -n '1p' | sed 's/[[:space:]]*$//')
[[ "${gpu_name}" == *H100* ]]

mkdir -p "${PROVENANCE}" "${RUNTIME}/cuda" "${RESULTS}"
touch "${ROOT}/SAFE_TO_DELETE"
write_source_manifest > "${PROVENANCE}/source_manifest.sha256"
source_manifest_sha256=$(sha256sum "${PROVENANCE}/source_manifest.sha256" | awk '{print $1}')
test "${source_manifest_sha256}" = "${EXPECTED_SOURCE_MANIFEST_SHA256}"
git -C "${REPO_ROOT}" rev-parse HEAD > "${PROVENANCE}/repo_head.txt"
git -C "${REPO_ROOT}" symbolic-ref --short HEAD > "${PROVENANCE}/repo_branch.txt"
git -C "${REPO_ROOT}" status --porcelain=v1 > "${PROVENANCE}/repo_status.txt"
printf '%s\n' "${SLURM_JOB_ID}" > "${PROVENANCE}/slurm_job_id.txt"
printf '%s\n' "${SLURMD_NODENAME:-$(hostname -s)}" > "${PROVENANCE}/node.txt"
printf '%s\n' "${selected_gpu_uuid}" > "${PROVENANCE}/selected_gpu_uuid.txt"
printf '%s\n' "${gpu_name}" > "${PROVENANCE}/gpu_name.txt"
printf '%s\n' "${VDAM_VISIBLE_GPU_UUIDS_CSV}" > "${PROVENANCE}/allocated_gpu_uuids.csv"
printf '%s\n' "${CUDA_ARCH_FLAGS}" > "${PROVENANCE}/cuda_arch_flags.txt"
printf '%s\n' "${CUDA_RUNTIME_TOOLKIT}" > "${PROVENANCE}/cuda_runtime_toolkit.txt"
nvidia-smi -q > "${PROVENANCE}/nvidia_smi.txt"
env | LC_ALL=C sort > "${PROVENANCE}/submission_environment.txt"
"${CUDA_BUILD_TOOLKIT}/bin/nvcc" --version > "${PROVENANCE}/nvcc_version.txt"

build_started=$(date +%s.%N)
make -B -C "${REPO_ROOT}/recovar/cuda" \
  PYTHON="${PIXI_PY}" \
  NVCC="${CUDA_BUILD_TOOLKIT}/bin/nvcc" \
  "CUDA_ARCH=${CUDA_ARCH_FLAGS}" \
  LIB="${CUDA_BINARY}" \
  > "${PROVENANCE}/cuda_build.stdout" \
  2> "${PROVENANCE}/cuda_build.stderr"
build_finished=$(date +%s.%N)
test -s "${CUDA_BINARY}"
cuda_sha256=$(sha256sum "${CUDA_BINARY}" | awk '{print $1}')
printf '%s  %s\n' "${cuda_sha256}" "${CUDA_BINARY}" > "${PROVENANCE}/qualified_cuda.sha256"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV JAX_PLATFORM_NAME
unset RECOVAR_DISABLE_CUDA RECOVAR_CUDA_LIB RECOVAR_ENABLE_CUSTOM_CUDA
unset RECOVAR_REQUIRE_CUSTOM_CUDA_FOR_TESTS
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${selected_gpu_uuid}
export RECOVAR_EXPECTED_REPO_ROOT=${REPO_ROOT}
export RECOVAR_CUDA_LIB=${CUDA_BINARY}
export RECOVAR_REQUIRE_CUSTOM_CUDA_FOR_TESTS=1
export RECOVAR_ENABLE_CUSTOM_CUDA=1
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=${RUNTIME}/jax_cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export TMPDIR=${JOB_TMP_ROOT}
export PIXI_HOME=${JOB_PIXI_ROOT}
export RATTLER_CACHE_DIR=${JOB_RATTLER_ROOT}
export PATH=${CUDA_RUNTIME_TOOLKIT}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_RUNTIME_TOOLKIT}/lib64:${CUDA_RUNTIME_TOOLKIT}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
mkdir -p "${JAX_COMPILATION_CACHE_DIR}" "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}"
touch "${JAX_COMPILATION_CACHE_DIR}/SAFE_TO_DELETE"
touch "${TMPDIR}/SAFE_TO_DELETE" "${PIXI_HOME}/SAFE_TO_DELETE" "${RATTLER_CACHE_DIR}/SAFE_TO_DELETE"

(
  cd "${REPO_ROOT}"
  "${PIXI_PY}" - <<'PY'
import os
import pathlib

import jax
import recovar
from recovar import cuda_backproject

repo = pathlib.Path(os.environ["RECOVAR_EXPECTED_REPO_ROOT"]).resolve()
assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo) + "/")
assert jax.default_backend() == "gpu"
assert len(jax.devices("gpu")) == 1
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
print("VDAM flat-row runtime gate passed", jax.devices(), recovar.__file__)
PY
) > "${PROVENANCE}/runtime_gate.stdout" 2> "${PROVENANCE}/runtime_gate.stderr"
vdam_verify_selected_gpu "${selected_gpu_uuid}"

test_started=$(date +%s.%N)
(
  cd "${REPO_ROOT}"
  "${PIXI_PY}" -m pytest -q --run-gpu \
    --junitxml="${RESULTS}/pytest.junit.xml" \
    tests/unit/test_flat_local_rows.py \
    tests/unit/test_cuda_relion_fine_diff2.py::test_relion_fused_translate_cuda_source_pins_native_block_topology
) > "${RESULTS}/pytest.stdout" 2> "${RESULTS}/pytest.stderr"
test_finished=$(date +%s.%N)

exact_started=$(date +%s.%N)
(
  cd "${REPO_ROOT}"
  "${PIXI_PY}" -m scripts.benchmark_vdam_flat_row_fine_score \
    --metadata-root "${VDAM_FLAT_ROW_METADATA_ROOT}" \
    --iterations 60 --warmups 2 --exactness-only \
    --output "${EXACT_JSON}"
) > "${RESULTS}/exactness.stdout" 2> "${RESULTS}/exactness.stderr"
exact_finished=$(date +%s.%N)

timing_started=$(date +%s.%N)
(
  cd "${REPO_ROOT}"
  "${PIXI_PY}" -m scripts.benchmark_vdam_flat_row_fine_score \
    --metadata-root "${VDAM_FLAT_ROW_METADATA_ROOT}" \
    --iterations 20 40 60 80 --warmups 2 --repeats 5 \
    --output "${TIMING_JSON}"
) > "${RESULTS}/timing.stdout" 2> "${RESULTS}/timing.stderr"
timing_finished=$(date +%s.%N)

"${PIXI_PY}" - "${EXACT_JSON}" "${TIMING_JSON}" "${EXPECTED_REPO_HEAD}" <<'PY'
import json
import sys
from pathlib import Path

exact = json.loads(Path(sys.argv[1]).read_text())
timing = json.loads(Path(sys.argv[2]).read_text())
expected_head = sys.argv[3]
for payload in (exact, timing):
    assert payload["schema"] == "recovar.vdam_flat_row_fine_score_microbenchmark.v1", payload
    assert payload["git_head"] == expected_head, payload
    assert payload["all_bitwise_equal"] is True, payload
    assert all(case["outer_call_cardinality_equal"] for case in payload["cases"]), payload
assert exact["mode"] == "exactness_only", exact
assert timing["mode"] == "timed", timing
assert timing["all_pass_two_percent_call_gate"] is True, timing
assert timing["promotion_decision"] == "accept", timing
PY

test "$(sha256sum "${CUDA_BINARY}" | awk '{print $1}')" = "${cuda_sha256}"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1)"
write_source_manifest > "${PROVENANCE}/source_manifest.final.sha256"
cmp "${PROVENANCE}/source_manifest.sha256" "${PROVENANCE}/source_manifest.final.sha256"
vdam_verify_selected_gpu "${selected_gpu_uuid}"

"${PIXI_PY}" - "${PROVENANCE}/run.json" <<PY
import hashlib
import json
from pathlib import Path

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

payload = {
    "schema": "recovar.vdam_flat_row_score_slurm_gate.v1",
    "status": "passed",
    "passed": True,
    "job_id": "${SLURM_JOB_ID}",
    "git_head": "${EXPECTED_REPO_HEAD}",
    "node": "${SLURMD_NODENAME:-$(hostname -s)}",
    "gpu_uuid": "${selected_gpu_uuid}",
    "gpu_name": "${gpu_name}",
    "source_manifest_sha256": "${source_manifest_sha256}",
    "cuda_sha256": "${cuda_sha256}",
    "pytest_junit_sha256": sha256("${RESULTS}/pytest.junit.xml"),
    "exactness_sha256": sha256("${EXACT_JSON}"),
    "timing_sha256": sha256("${TIMING_JSON}"),
    "build_wall_s": float("${build_finished}") - float("${build_started}"),
    "focused_test_wall_s": float("${test_finished}") - float("${test_started}"),
    "exactness_wall_s": float("${exact_finished}") - float("${exact_started}"),
    "timing_wall_s": float("${timing_finished}") - float("${timing_started}"),
    "classification": "primitive_exactness_and_call_level_performance",
    "trajectory_promotion_allowed": False,
}
Path("${PROVENANCE}/run.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

touch "${ROOT}/COMPLETED"
printf 'VDAM flat-row gate completed: %s\n' "${ROOT}"
