#!/usr/bin/env bash
#SBATCH --job-name=vdam-coarse-ms-test
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
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-coarse-ms-test-%j.out

set -euo pipefail

: "${VDAM_COARSE_MULTISTREAM_TEST_ROOT:?set a new disposable output root}"
: "${EXPECTED_REPO_HEAD:?pin the tracked RECOVAR source head}"
: "${EXPECTED_SOURCE_MANIFEST_SHA256:?pin the source/test manifest}"
: "${EXPECTED_INTERPRETER_SHA256:?pin the pixi interpreter}"
: "${EXPECTED_NODE_NAME:?pin the Slurm node}"
: "${TARGET_GPU_UUID:?pin the physical H100 UUID}"
: "${PREBUILT_CUDA_BINARY:?set the exact binary built by the sealed first stage}"
: "${EXPECTED_CUDA_SHA256:?pin the prebuilt CUDA binary}"
: "${PRECOMPUTED_RESOURCE_REPORT:?set the login-node resource JSON}"
: "${EXPECTED_RESOURCE_REPORT_SHA256:?pin the resource JSON}"
: "${PRECOMPUTED_RESOURCE_DEMANGLED:?set the demangled cuobjdump report}"
: "${EXPECTED_RESOURCE_DEMANGLED_SHA256:?pin the demangled cuobjdump report}"

ROOT=${VDAM_COARSE_MULTISTREAM_TEST_ROOT}
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
PIXI_PY=${PIXI_PY_OVERRIDE:-${REPO_ROOT}/.pixi/envs/default/bin/python}
CUDA_BUILD_TOOLKIT=${CUDA_BUILD_TOOLKIT:-/usr/local/cuda-12.6}
CUDA_RUNTIME_TOOLKIT=${CUDA_RUNTIME_TOOLKIT:-/usr/local/cuda-13.1}
PROVENANCE=${ROOT}/provenance
RUNTIME=${ROOT}/runtime
CUDA_BINARY=${RUNTIME}/cuda/libcuda_backproject.so
TEST_NODES=(
  tests/unit/test_cuda_relion_fine_diff2.py::test_relion_coarse_vdam_multistream_skips_poisoned_padding_bitwise
  tests/unit/test_cuda_relion_fine_diff2.py::test_relion_coarse_vdam_multistream_atomic_stays_in_lane_envelope
)
SOURCE_FILES=(
  recovar/cuda/Makefile
  recovar/cuda/cuda_backproject.cu
  recovar/cuda_backproject.py
  recovar/em/scoring/significance.py
  scripts/vdam_gpu_selection.sh
  tests/conftest.py
  tests/unit/test_cuda_relion_fine_diff2.py
  scripts/run_vdam_coarse_multistream_gpu_gate.sbatch
  pixi.lock
  pixi.toml
  pyproject.toml
)

on_error() {
  local status=$?
  local message="VDAM coarse multistream GPU gate failed: status=${status} line=${BASH_LINENO[0]} command=${BASH_COMMAND}"
  printf '%s\n' "${message}" >&2
  if [[ -d "${PROVENANCE}" ]]; then
    printf '%s\n' "${message}" > "${PROVENANCE}/failure.txt"
  fi
  exit "${status}"
}
trap on_error ERR

test ! -e "${ROOT}"
test -x "${PIXI_PY}"
test "$(sha256sum "${PIXI_PY}" | awk '{print $1}')" = \
  "${EXPECTED_INTERPRETER_SHA256}"
test -x "${CUDA_BUILD_TOOLKIT}/bin/nvcc"
test -d "${CUDA_RUNTIME_TOOLKIT}/lib64"
test -s "${PREBUILT_CUDA_BINARY}"
test -s "${PRECOMPUTED_RESOURCE_REPORT}"
test -s "${PRECOMPUTED_RESOURCE_DEMANGLED}"
test "${SLURMD_NODENAME:-$(hostname -s)}" = "${EXPECTED_NODE_NAME}"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1)"
repo_tree=$(git -C "${REPO_ROOT}" rev-parse 'HEAD^{tree}')

initial_allocation_spec=${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}}
source "${REPO_ROOT}/scripts/vdam_gpu_selection.sh"
vdam_assert_target_gpu_allocated "${TARGET_GPU_UUID}" "${initial_allocation_spec}"
vdam_select_target_gpu "${TARGET_GPU_UUID}" 0
selected_gpu_uuid=${VDAM_SELECTED_GPU_UUID}
test "${selected_gpu_uuid}" = "${TARGET_GPU_UUID}"
gpu_name=$(nvidia-smi -i "${selected_gpu_uuid}" --query-gpu=name --format=csv,noheader | sed -n '1p' | sed 's/[[:space:]]*$//')
[[ "${gpu_name}" == *H100* ]]

mkdir -p "${PROVENANCE}" "${RUNTIME}/cuda"
touch "${ROOT}/SAFE_TO_DELETE"
(
  cd "${REPO_ROOT}"
  sha256sum "${SOURCE_FILES[@]}"
) > "${PROVENANCE}/source_manifest.sha256"
source_manifest_sha256=$(sha256sum "${PROVENANCE}/source_manifest.sha256" | awk '{print $1}')
test "${source_manifest_sha256}" = "${EXPECTED_SOURCE_MANIFEST_SHA256}"
git -C "${REPO_ROOT}" rev-parse HEAD > "${PROVENANCE}/repo_head.txt"
printf '%s\n' "${repo_tree}" > "${PROVENANCE}/repo_tree.txt"
git -C "${REPO_ROOT}" symbolic-ref --short HEAD > "${PROVENANCE}/repo_branch.txt"
git -C "${REPO_ROOT}" status --porcelain=v1 > "${PROVENANCE}/repo_status.txt"
git -C "${REPO_ROOT}" diff HEAD | sha256sum > "${PROVENANCE}/repo_diff.sha256"
sha256sum "${PIXI_PY}" > "${PROVENANCE}/interpreter.sha256"
printf '%s\n' "${SLURM_JOB_ID}" > "${PROVENANCE}/slurm_job_id.txt"
printf '%s\n' "${SLURMD_NODENAME:-$(hostname -s)}" > "${PROVENANCE}/node.txt"
printf '%s\n' "${selected_gpu_uuid}" > "${PROVENANCE}/selected_gpu_uuid.txt"
printf '%s\n' "${gpu_name}" > "${PROVENANCE}/gpu_name.txt"
printf '%s\n' "${VDAM_ALLOCATED_GPU_UUIDS_CSV}" > "${PROVENANCE}/allocated_gpu_uuids.csv"
printf '%s\n' "${VDAM_VISIBLE_GPU_UUIDS_CSV}" > "${PROVENANCE}/visible_gpu_uuids.csv"
nvidia-smi -i "${selected_gpu_uuid}" -q > "${PROVENANCE}/nvidia_smi.txt"
env | LC_ALL=C sort > "${PROVENANCE}/submission_environment.txt"
"${CUDA_BUILD_TOOLKIT}/bin/nvcc" --version > "${PROVENANCE}/nvcc_version.txt"
printf '%s\n' "${CUDA_RUNTIME_TOOLKIT}" \
  > "${PROVENANCE}/cuda_runtime_toolkit.txt"

build_started=$(date +%s.%N)
test "$(sha256sum "${PREBUILT_CUDA_BINARY}" | awk '{print $1}')" = \
  "${EXPECTED_CUDA_SHA256}"
test "$(sha256sum "${PRECOMPUTED_RESOURCE_REPORT}" | awk '{print $1}')" = \
  "${EXPECTED_RESOURCE_REPORT_SHA256}"
test "$(sha256sum "${PRECOMPUTED_RESOURCE_DEMANGLED}" | awk '{print $1}')" = \
  "${EXPECTED_RESOURCE_DEMANGLED_SHA256}"
cp --reflink=auto "${PREBUILT_CUDA_BINARY}" "${CUDA_BINARY}"
cp --reflink=auto "${PRECOMPUTED_RESOURCE_REPORT}" \
  "${PROVENANCE}/coarse_kernel_resources.json"
cp --reflink=auto "${PRECOMPUTED_RESOURCE_DEMANGLED}" \
  "${PROVENANCE}/cuda_resource_usage.demangled.txt"
build_finished=$(date +%s.%N)
test -s "${CUDA_BINARY}"
cuda_sha256=$(sha256sum "${CUDA_BINARY}" | awk '{print $1}')
test "${cuda_sha256}" = "${EXPECTED_CUDA_SHA256}"
printf '%s  %s\n' "${cuda_sha256}" "${CUDA_BINARY}" \
  > "${PROVENANCE}/qualified_cuda.sha256"
"${PIXI_PY}" - \
  "${PROVENANCE}/coarse_kernel_resources.json" \
  "${PROVENANCE}/cuda_resource_usage.demangled.txt" \
  "${EXPECTED_CUDA_SHA256}" \
  "${EXPECTED_RESOURCE_DEMANGLED_SHA256}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
resources = report["resources"]
assert report["binary_sha256"] == sys.argv[3], report
assert set(resources) == {"canonical", "atomic"}, resources
assert (
    resources["canonical"]["static_shared_bytes"]
    - resources["atomic"]["static_shared_bytes"]
) == 16 * 128 * 4, resources
assert (
    resources["atomic"]["registers_per_thread"]
    < resources["canonical"]["registers_per_thread"]
), resources
demangled_sha256 = hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest()
assert demangled_sha256 == sys.argv[4], demangled_sha256
assert report["demangled_resource_report_sha256"] == demangled_sha256, report
PY

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV JAX_PLATFORM_NAME
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
import os
import pathlib

import jax
import recovar
from recovar import cuda_backproject
from recovar.utils.parity_provenance import assert_parity_ancestors

repo = pathlib.Path(os.environ["RECOVAR_EXPECTED_REPO_ROOT"]).resolve()
assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo) + "/")
assert ".pixi/envs/default/" in str(pathlib.Path(jax.__file__).resolve())
assert jax.default_backend() == "gpu"
assert len(jax.devices("gpu")) == 1
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
assert pathlib.Path(os.environ["RECOVAR_CUDA_LIB"]).is_file()
assert_parity_ancestors()
print("VDAM coarse multistream focused GPU/runtime gate passed", jax.devices(), recovar.__file__)
PY
) > "${PROVENANCE}/runtime_gate.stdout" 2> "${PROVENANCE}/runtime_gate.stderr"
vdam_verify_selected_gpu "${selected_gpu_uuid}"

test_started=$(date +%s.%N)
(
  cd "${REPO_ROOT}"
  "${PIXI_PY}" -m pytest -v --run-gpu \
    --junitxml="${ROOT}/pytest.junit.xml" \
    "${TEST_NODES[@]}"
) > "${ROOT}/pytest.stdout" 2> "${ROOT}/pytest.stderr"
test_finished=$(date +%s.%N)

"${PIXI_PY}" - "${ROOT}/pytest.junit.xml" <<'PY'
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
suite = root if root.tag == "testsuite" else root.find("testsuite")
assert suite is not None
assert int(suite.attrib["tests"]) == 3
assert int(suite.attrib.get("failures", 0)) == 0
assert int(suite.attrib.get("errors", 0)) == 0
assert int(suite.attrib.get("skipped", 0)) == 0
PY

test "$(sha256sum "${CUDA_BINARY}" | awk '{print $1}')" = "${cuda_sha256}"
test "$(sha256sum "${PIXI_PY}" | awk '{print $1}')" = \
  "${EXPECTED_INTERPRETER_SHA256}"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${REPO_ROOT}" rev-parse 'HEAD^{tree}')" = "${repo_tree}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1)"
(
  cd "${REPO_ROOT}"
  sha256sum "${SOURCE_FILES[@]}"
) > "${PROVENANCE}/source_manifest.final.sha256"
cmp "${PROVENANCE}/source_manifest.sha256" "${PROVENANCE}/source_manifest.final.sha256"
vdam_verify_selected_gpu "${selected_gpu_uuid}"

"${PIXI_PY}" - "${PROVENANCE}/run.json" \
  "${SLURM_JOB_ID}" "${EXPECTED_REPO_HEAD}" "${selected_gpu_uuid}" \
  "${SLURMD_NODENAME:-$(hostname -s)}" "${gpu_name}" "${cuda_sha256}" \
  "${source_manifest_sha256}" "${build_started}" "${build_finished}" \
  "${test_started}" "${test_finished}" "${TEST_NODES[*]}" \
  "${repo_tree}" "${EXPECTED_INTERPRETER_SHA256}" <<'PY'
import json
import sys
from pathlib import Path

payload = {
    "schema": "recovar.vdam_coarse_multistream_focused_gpu_gate.v2",
    "classification": "performance_only_qualification",
    "job_id": sys.argv[2],
    "git_head": sys.argv[3],
    "git_tree": sys.argv[14],
    "interpreter_sha256": sys.argv[15],
    "gpu_uuid": sys.argv[4],
    "node": sys.argv[5],
    "gpu_name": sys.argv[6],
    "cuda_sha256": sys.argv[7],
    "source_manifest_sha256": sys.argv[8],
    "source_manifest_scope": "selected_high_risk_files",
    "cuda_stage_wall_s": float(sys.argv[10]) - float(sys.argv[9]),
    "test_wall_s": float(sys.argv[12]) - float(sys.argv[11]),
    "pytest_node": sys.argv[13],
    "tests": 3,
    "passed": 3,
    "science_promotion_allowed": False,
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

touch "${ROOT}/COMPLETED"
