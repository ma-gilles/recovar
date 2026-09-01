#!/usr/bin/env bash
#SBATCH --job-name=fixed-local-score-gate
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
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/fixed-local-score-gate-%j.out

set -euo pipefail

: "${FIXED_CAPACITY_LOCAL_SCORE_GATE_ROOT:?set a new disposable output root}"
: "${EXPECTED_REPO_HEAD:?pin the committed fixed-capacity score-gate head}"
: "${EXPECTED_SOURCE_MANIFEST_SHA256:?pin the source/test manifest}"
: "${EXPECTED_NODE_NAME:?pin the Slurm node}"
: "${TARGET_GPU_UUID:?pin the physical H100 UUID}"
: "${EXPECTED_FOCUSED_TEST_COUNT:?pin the exact focused pytest count}"

ROOT=${FIXED_CAPACITY_LOCAL_SCORE_GATE_ROOT}
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
PIXI_PY=${PIXI_PY_OVERRIDE:-${REPO_ROOT}/.pixi/envs/default/bin/python}
CUDA_BUILD_TOOLKIT=${CUDA_BUILD_TOOLKIT:-/usr/local/cuda-12.6}
CUDA_RUNTIME_TOOLKIT=${CUDA_RUNTIME_TOOLKIT:-/usr/local/cuda-13.1}
CUDA_ARCH_FLAGS=${FIXED_CAPACITY_CUDA_ARCH_FLAGS:--gencode arch=compute_90,code=sm_90}
PROVENANCE=${ROOT}/provenance
RUNTIME=${ROOT}/runtime
RESULTS=${ROOT}/results
CUDA_BINARY=${RUNTIME}/cuda/libcuda_backproject.so
HARNESS_OUTPUT=${RESULTS}/harness
HARNESS_RESULT=${HARNESS_OUTPUT}/gate_result.json
HARNESS_SCHEMA=recovar.fixed_capacity_local_score_gate.v1
FOCUSED_TEST=tests/unit/test_fixed_capacity_local_score_gate.py
JOB_TMP_ROOT=/scratch/gpfs/GILLES/mg6942/tmp/${SLURM_JOB_ID}
JOB_PIXI_HOME=/scratch/gpfs/GILLES/mg6942/pixi_home/${SLURM_JOB_ID}
JOB_RATTLER_CACHE=/scratch/gpfs/GILLES/mg6942/rattler_cache/${SLURM_JOB_ID}
SOURCE_FILES=(
  recovar/cuda/Makefile
  recovar/cuda/cuda_backproject.cu
  recovar/cuda_backproject.py
  recovar/utils/parity_provenance.py
  recovar/em/dense_single_volume/batch_planning.py
  recovar/em/dense_single_volume/fixed_capacity_local.py
  recovar/em/dense_single_volume/local_big_jit.py
  recovar/em/dense_single_volume/local_caches.py
  recovar/em/dense_single_volume/local_em_engine.py
  recovar/em/dense_single_volume/local_layout.py
  scripts/run_fixed_capacity_local_score_gate.py
  scripts/run_fixed_capacity_local_score_gate.sbatch
  scripts/vdam_gpu_selection.sh
  tests/conftest.py
  tests/unit/test_fixed_capacity_local_binding.py
  tests/unit/test_fixed_capacity_local_score_gate.py
  pixi.lock
  pixi.toml
  pyproject.toml
)

on_error() {
  local status=$?
  local message="fixed-capacity local score gate failed: status=${status} line=${BASH_LINENO[0]} command=${BASH_COMMAND}"
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
test -x "${PIXI_PY}"
test -x "${CUDA_BUILD_TOOLKIT}/bin/nvcc"
test -d "${CUDA_RUNTIME_TOOLKIT}/lib64"
[[ "${EXPECTED_FOCUSED_TEST_COUNT}" =~ ^[1-9][0-9]*$ ]]
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
git -C "${REPO_ROOT}" diff --binary --no-ext-diff HEAD -- | sha256sum \
  > "${PROVENANCE}/repo_diff.sha256"
printf '%s\n' "${SLURM_JOB_ID}" > "${PROVENANCE}/slurm_job_id.txt"
printf '%s\n' "${SLURMD_NODENAME:-$(hostname -s)}" > "${PROVENANCE}/node.txt"
printf '%s\n' "${selected_gpu_uuid}" > "${PROVENANCE}/selected_gpu_uuid.txt"
printf '%s\n' "${gpu_name}" > "${PROVENANCE}/gpu_name.txt"
printf '%s\n' "${VDAM_VISIBLE_GPU_UUIDS_CSV}" > "${PROVENANCE}/allocated_gpu_uuids.csv"
printf '%s\n' "${CUDA_ARCH_FLAGS}" > "${PROVENANCE}/cuda_arch_flags.txt"
printf '%s\n' "${CUDA_RUNTIME_TOOLKIT}" > "${PROVENANCE}/cuda_runtime_toolkit.txt"
printf '%s\n' "${EXPECTED_FOCUSED_TEST_COUNT}" > "${PROVENANCE}/focused_test_count.txt"
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
printf '%s  %s\n' "${cuda_sha256}" "${CUDA_BINARY}" \
  > "${PROVENANCE}/qualified_cuda.sha256"

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
export PIXI_HOME=${JOB_PIXI_HOME}
export RATTLER_CACHE_DIR=${JOB_RATTLER_CACHE}
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
from recovar.utils.parity_provenance import assert_parity_ancestors

repo = pathlib.Path(os.environ["RECOVAR_EXPECTED_REPO_ROOT"]).resolve()
assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo) + "/")
assert ".pixi/envs/default/" in str(pathlib.Path(jax.__file__).resolve())
assert jax.default_backend() == "gpu"
assert len(jax.devices("gpu")) == 1
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
cuda_binary = pathlib.Path(os.environ["RECOVAR_CUDA_LIB"]).resolve()
assert cuda_binary.is_file()
assert not str(cuda_binary).startswith(str(repo) + "/")
assert_parity_ancestors()
print("fixed-capacity local score GPU/runtime gate passed", jax.devices(), recovar.__file__)
PY
) > "${PROVENANCE}/runtime_gate.stdout" 2> "${PROVENANCE}/runtime_gate.stderr"
vdam_verify_selected_gpu "${selected_gpu_uuid}"

test_started=$(date +%s.%N)
(
  cd "${REPO_ROOT}"
  "${PIXI_PY}" -m pytest -vv --run-gpu \
    --junitxml="${RESULTS}/pytest.junit.xml" \
    "${FOCUSED_TEST}"
) > "${RESULTS}/pytest.stdout" 2> "${RESULTS}/pytest.stderr"
test_finished=$(date +%s.%N)

"${PIXI_PY}" - "${RESULTS}/pytest.junit.xml" "${EXPECTED_FOCUSED_TEST_COUNT}" <<'PY'
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
cases = root.findall(".//testcase")
assert len(cases) == int(sys.argv[2]), (len(cases), sys.argv[2])
for case in cases:
    assert case.find("failure") is None, ET.tostring(case, encoding="unicode")
    assert case.find("error") is None, ET.tostring(case, encoding="unicode")
    assert case.find("skipped") is None, ET.tostring(case, encoding="unicode")
PY

harness_started=$(date +%s.%N)
printf '%q ' "${PIXI_PY}" scripts/run_fixed_capacity_local_score_gate.py \
  --output-dir "${HARNESS_OUTPUT}" \
  --expected-repo-head "${EXPECTED_REPO_HEAD}" \
  --expected-gpu-uuid "${selected_gpu_uuid}" \
  > "${PROVENANCE}/harness_command.txt"
printf '\n' >> "${PROVENANCE}/harness_command.txt"
(
  cd "${REPO_ROOT}"
  "${PIXI_PY}" scripts/run_fixed_capacity_local_score_gate.py \
    --output-dir "${HARNESS_OUTPUT}" \
    --expected-repo-head "${EXPECTED_REPO_HEAD}" \
    --expected-gpu-uuid "${selected_gpu_uuid}"
) > "${RESULTS}/harness.stdout" 2> "${RESULTS}/harness.stderr"
harness_finished=$(date +%s.%N)
test -s "${HARNESS_RESULT}"

"${PIXI_PY}" - "${HARNESS_RESULT}" "${HARNESS_SCHEMA}" \
  "${EXPECTED_REPO_HEAD}" "${selected_gpu_uuid}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
assert payload["schema"] == sys.argv[2], payload
assert payload["classification"] == "correctness_only", payload
assert payload["passed"] is True, payload
assert payload["speed_claim_allowed"] is False, payload
assert payload["default_promotion_allowed"] is False, payload
assert payload["git_head"] == sys.argv[3], payload
assert payload["gpu_uuid"] == sys.argv[4], payload
PY

test "$(sha256sum "${CUDA_BINARY}" | awk '{print $1}')" = "${cuda_sha256}"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1)"
write_source_manifest > "${PROVENANCE}/source_manifest.final.sha256"
cmp "${PROVENANCE}/source_manifest.sha256" "${PROVENANCE}/source_manifest.final.sha256"
vdam_verify_selected_gpu "${selected_gpu_uuid}"

harness_result_sha256=$(sha256sum "${HARNESS_RESULT}" | awk '{print $1}')
junit_sha256=$(sha256sum "${RESULTS}/pytest.junit.xml" | awk '{print $1}')
"${PIXI_PY}" - "${PROVENANCE}/run.json" \
  "${SLURM_JOB_ID}" "${EXPECTED_REPO_HEAD}" "${selected_gpu_uuid}" \
  "${SLURMD_NODENAME:-$(hostname -s)}" "${gpu_name}" "${cuda_sha256}" \
  "${source_manifest_sha256}" "${build_started}" "${build_finished}" \
  "${test_started}" "${test_finished}" "${harness_started}" "${harness_finished}" \
  "${EXPECTED_FOCUSED_TEST_COUNT}" "${harness_result_sha256}" "${junit_sha256}" <<'PY'
import json
import sys
from pathlib import Path

payload = {
    "schema": "recovar.fixed_capacity_local_score_slurm_gate.v1",
    "classification": "correctness_only",
    "status": "passed",
    "passed": True,
    "job_id": sys.argv[2],
    "git_head": sys.argv[3],
    "gpu_uuid": sys.argv[4],
    "node": sys.argv[5],
    "gpu_name": sys.argv[6],
    "cuda_sha256": sys.argv[7],
    "source_manifest_sha256": sys.argv[8],
    "cuda_build_wall_s": float(sys.argv[10]) - float(sys.argv[9]),
    "focused_test_wall_s": float(sys.argv[12]) - float(sys.argv[11]),
    "harness_wall_s": float(sys.argv[14]) - float(sys.argv[13]),
    "focused_test_count": int(sys.argv[15]),
    "harness_result_sha256": sys.argv[16],
    "pytest_junit_sha256": sys.argv[17],
    "speed_claim_allowed": False,
    "default_promotion_allowed": False,
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

touch "${ROOT}/COMPLETED"
