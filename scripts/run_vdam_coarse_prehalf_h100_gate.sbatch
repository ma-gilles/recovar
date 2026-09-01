#!/usr/bin/env bash
#SBATCH --job-name=vdam-prehalf-h100
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-prehalf-h100-%j.out

set -euo pipefail

: "${EXPECTED_REPO_HEAD:?pin the committed runner head}"
: "${TARGET_GPU_UUID:?pin the physical H100 UUID}"

readonly PRIMITIVE_HEAD=3c1ab9990b33496697a894e617f81fc0b0a2d0d9
readonly PRIMITIVE_TREE=95c4a7cd4c40878beb87380e43d7e310748586fe
readonly BASELINE_BINARY=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_coarse_atomic_multistream_resources_129280420_20260831/libcuda_backproject.so
readonly BASELINE_SHA256=fcd9f03383ec42483016cf2b686adc998e61f1cc6fdf338b3a6713bbc3554d56
readonly CUDA_TOOLKIT=/usr/local/cuda-12.6
readonly CUDA_BINARY_TOOLS_TOOLKIT=/usr/local/cuda-13.1
readonly CUDA_ARCH_FLAGS='-gencode arch=compute_90,code=sm_90'
readonly FOCUSED_GPU_TEST=tests/unit/test_cuda_relion_fine_diff2.py::test_relion_coarse_vdam_prehalf_source_order_across_dispatchers

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
REPO_ROOT=$(realpath -e -- "${REPO_ROOT}")
PIXI_PY=${REPO_ROOT}/.pixi/envs/default/bin/python
NVCC=${CUDA_TOOLKIT}/bin/nvcc
CUOBJDUMP=${CUDA_BINARY_TOOLS_TOOLKIT}/bin/cuobjdump
RESULT_ROOT=${VDAM_PREHALF_RESULT_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_coarse_prehalf_h100_${SLURM_JOB_ID}}
RESULT_ROOT=$(realpath -m -- "${RESULT_ROOT}")
PROVENANCE=${RESULT_ROOT}/provenance
RESULTS=${RESULT_ROOT}/results
BUILD=${RESULT_ROOT}/build
CANDIDATE_BINARY=${BUILD}/libcuda_backproject.so
AUDIT_JSON=${RESULTS}/binary_audit.json
AUDIT_MARKDOWN=${RESULTS}/binary_audit.md
PYTEST_JUNIT=${RESULTS}/focused_gpu_pytest.junit.xml
BENCHMARK_JSON=${RESULTS}/gf46_default_geometry_benchmark.json
RUNTIME_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/${SLURM_JOB_ID}

SOURCE_FILES=(
  recovar/cuda/Makefile
  recovar/cuda/cuda_backproject.cu
  recovar/cuda/relion_coarse_diff2_projector_body.inc
  recovar/cuda_backproject.py
  scripts/audit_vdam_coarse_prehalf_binary.py
  scripts/benchmark_vdam_coarse_prehalf.py
  scripts/run_vdam_coarse_prehalf_h100_gate.sbatch
  scripts/vdam_gpu_selection.sh
  tests/conftest.py
  tests/unit/test_cuda_relion_fine_diff2.py
  tests/unit/initial_model/test_vdam_coarse_prehalf_binary_audit.py
  tests/unit/initial_model/test_vdam_coarse_prehalf_benchmark.py
  tests/unit/initial_model/test_vdam_coarse_prehalf_h100_runner.py
  pixi.lock
  pixi.toml
  pyproject.toml
)
PRIMITIVE_FILES=(
  recovar/cuda/Makefile
  recovar/cuda/cuda_backproject.cu
  recovar/cuda/relion_coarse_diff2_projector_body.inc
  recovar/cuda_backproject.py
  scripts/audit_vdam_coarse_prehalf_binary.py
  scripts/benchmark_vdam_coarse_prehalf.py
  tests/conftest.py
  tests/unit/test_cuda_relion_fine_diff2.py
  tests/unit/initial_model/test_vdam_coarse_prehalf_binary_audit.py
  tests/unit/initial_model/test_vdam_coarse_prehalf_benchmark.py
  pixi.lock
  pixi.toml
  pyproject.toml
)

on_error() {
  local status=$?
  local message="VDAM coarse prehalf H100 gate failed: status=${status} line=${BASH_LINENO[0]} command=${BASH_COMMAND}"
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

[[ "${EXPECTED_REPO_HEAD}" =~ ^[0-9a-f]{40}$ ]]
[[ "${TARGET_GPU_UUID}" =~ ^GPU-[0-9a-fA-F-]{36}$ ]]
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
test -x "${CUOBJDUMP}"
test -s "${BASELINE_BINARY}"
test "$(sha256sum "${BASELINE_BINARY}" | awk '{print $1}')" = "${BASELINE_SHA256}"
test "$(git -C "${REPO_ROOT}" rev-parse --show-toplevel)" = "${REPO_ROOT}"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=all)"
test "$(git -C "${REPO_ROOT}" rev-parse "${PRIMITIVE_HEAD}^{tree}")" = "${PRIMITIVE_TREE}"
git -C "${REPO_ROOT}" merge-base --is-ancestor "${PRIMITIVE_HEAD}" "${EXPECTED_REPO_HEAD}"
git -C "${REPO_ROOT}" diff --quiet "${PRIMITIVE_HEAD}" -- "${PRIMITIVE_FILES[@]}"

initial_allocation_spec=${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}}
source "${REPO_ROOT}/scripts/vdam_gpu_selection.sh"
vdam_assert_target_gpu_allocated "${TARGET_GPU_UUID}" "${initial_allocation_spec}"
vdam_select_target_gpu "${TARGET_GPU_UUID}" 0
selected_gpu_uuid=${VDAM_SELECTED_GPU_UUID}
test "${selected_gpu_uuid}" = "${TARGET_GPU_UUID}"
gpu_name=$(nvidia-smi -i "${selected_gpu_uuid}" --query-gpu=name --format=csv,noheader | sed -n '1p' | sed 's/[[:space:]]*$//')
[[ "${gpu_name}" == *H100* ]]

mkdir "${RESULT_ROOT}"
mkdir -p "${PROVENANCE}" "${RESULTS}" "${BUILD}"
touch "${RESULT_ROOT}/SAFE_TO_DELETE"

baseline_stat_before=$(stat -c '%d:%i:%s:%Y:%Z' "${BASELINE_BINARY}")
write_source_manifest > "${PROVENANCE}/source_manifest.sha256"
git -C "${REPO_ROOT}" rev-parse HEAD > "${PROVENANCE}/repo_head.txt"
git -C "${REPO_ROOT}" rev-parse 'HEAD^{tree}' > "${PROVENANCE}/repo_tree.txt"
git -C "${REPO_ROOT}" symbolic-ref --short HEAD > "${PROVENANCE}/repo_branch.txt"
git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=all > "${PROVENANCE}/repo_status.txt"
git -C "${REPO_ROOT}" diff --binary --no-ext-diff HEAD -- | sha256sum > "${PROVENANCE}/repo_diff.sha256"
printf '%s\n' "${PRIMITIVE_HEAD}" > "${PROVENANCE}/primitive_head.txt"
printf '%s\n' "${PRIMITIVE_TREE}" > "${PROVENANCE}/primitive_tree.txt"
printf '%s\n' "${SLURM_JOB_ID}" > "${PROVENANCE}/slurm_job_id.txt"
printf '%s\n' "${SLURMD_NODENAME:-$(hostname -s)}" > "${PROVENANCE}/node.txt"
printf '%s\n' "${selected_gpu_uuid}" > "${PROVENANCE}/selected_gpu_uuid.txt"
printf '%s\n' "${gpu_name}" > "${PROVENANCE}/gpu_name.txt"
printf '%s\n' "${VDAM_ALLOCATED_GPU_UUIDS_CSV}" > "${PROVENANCE}/allocated_gpu_uuids.csv"
printf '%s\n' "${VDAM_VISIBLE_GPU_UUIDS_CSV}" > "${PROVENANCE}/visible_gpu_uuids.csv"
printf '%s\n' "${CUDA_ARCH_FLAGS}" > "${PROVENANCE}/cuda_arch_flags.txt"
printf '%s\n' "${baseline_stat_before}" > "${PROVENANCE}/baseline_stat.txt"
sha256sum "${BASELINE_BINARY}" > "${PROVENANCE}/baseline_binary.sha256"
sha256sum "${PIXI_PY}" > "${PROVENANCE}/interpreter.sha256"
nvidia-smi -i "${selected_gpu_uuid}" -q > "${PROVENANCE}/nvidia_smi.txt"
"${NVCC}" --version > "${PROVENANCE}/nvcc_version.txt"
"${CUOBJDUMP}" --version > "${PROVENANCE}/cuobjdump_version.txt"
scontrol show job "${SLURM_JOB_ID}" > "${PROVENANCE}/slurm_job.txt"

{
  printf '%q ' make -B -C "${REPO_ROOT}/recovar/cuda" \
    "PYTHON=${PIXI_PY}" "NVCC=${NVCC}" "CUDA_ARCH=${CUDA_ARCH_FLAGS}" \
    "LIB=${CANDIDATE_BINARY}"
  printf '\n'
  printf '%q ' "${PIXI_PY}" scripts/audit_vdam_coarse_prehalf_binary.py \
    --baseline-binary "${BASELINE_BINARY}" --candidate-binary "${CANDIDATE_BINARY}" \
    --cuobjdump "${CUOBJDUMP}" --output-json "${AUDIT_JSON}" \
    --output-markdown "${AUDIT_MARKDOWN}"
  printf '\n'
  printf '%q ' "${PIXI_PY}" -m pytest -vv --run-gpu \
    --junitxml="${PYTEST_JUNIT}" "${FOCUSED_GPU_TEST}"
  printf '\n'
  printf '%q ' "${PIXI_PY}" scripts/benchmark_vdam_coarse_prehalf.py \
    --repo "${REPO_ROOT}" --output-json "${BENCHMARK_JSON}"
  printf '\n'
} > "${PROVENANCE}/commands.txt"

make -B -C "${REPO_ROOT}/recovar/cuda" \
  "PYTHON=${PIXI_PY}" \
  "NVCC=${NVCC}" \
  "CUDA_ARCH=${CUDA_ARCH_FLAGS}" \
  "LIB=${CANDIDATE_BINARY}" \
  > "${PROVENANCE}/cuda_build.stdout" \
  2> "${PROVENANCE}/cuda_build.stderr"
test -s "${CANDIDATE_BINARY}"
candidate_sha256=$(sha256sum "${CANDIDATE_BINARY}" | awk '{print $1}')
sha256sum "${CANDIDATE_BINARY}" > "${PROVENANCE}/candidate_binary.sha256"
printf 'after_build %s\n' "${candidate_sha256}" \
  > "${PROVENANCE}/candidate_sha256_checkpoints.txt"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV JAX_PLATFORM_NAME
unset RECOVAR_DISABLE_CUDA RECOVAR_CUDA_LIB RECOVAR_ENABLE_CUSTOM_CUDA
unset RECOVAR_REQUIRE_CUSTOM_CUDA_FOR_TESTS
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${selected_gpu_uuid}
export RECOVAR_EXPECTED_REPO_ROOT=${REPO_ROOT}
export RECOVAR_CUDA_LIB=${CANDIDATE_BINARY}
export RECOVAR_REQUIRE_CUSTOM_CUDA_FOR_TESTS=1
export RECOVAR_ENABLE_CUSTOM_CUDA=1
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=${RUNTIME_ROOT}/jax_cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export TMPDIR=${RUNTIME_ROOT}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/rattler_cache
mkdir -p "${JAX_COMPILATION_CACHE_DIR}" "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}"
touch "${RUNTIME_ROOT}/SAFE_TO_DELETE"
{
  printf 'EXPECTED_REPO_HEAD=%q\n' "${EXPECTED_REPO_HEAD}"
  printf 'TARGET_GPU_UUID=%q\n' "${TARGET_GPU_UUID}"
  printf 'CUDA_VISIBLE_DEVICES=%q\n' "${CUDA_VISIBLE_DEVICES}"
  printf 'RECOVAR_CUDA_LIB=%q\n' "${RECOVAR_CUDA_LIB}"
  printf 'RECOVAR_REQUIRE_CUSTOM_CUDA_FOR_TESTS=%q\n' \
    "${RECOVAR_REQUIRE_CUSTOM_CUDA_FOR_TESTS}"
  printf 'RECOVAR_ENABLE_CUSTOM_CUDA=%q\n' "${RECOVAR_ENABLE_CUSTOM_CUDA}"
  printf 'JAX_PLATFORMS=%q\n' "${JAX_PLATFORMS}"
  printf 'XLA_PYTHON_CLIENT_PREALLOCATE=%q\n' "${XLA_PYTHON_CLIENT_PREALLOCATE}"
  printf 'JAX_COMPILATION_CACHE_DIR=%q\n' "${JAX_COMPILATION_CACHE_DIR}"
  printf 'TMPDIR=%q\n' "${TMPDIR}"
  printf 'PIXI_HOME=%q\n' "${PIXI_HOME}"
  printf 'RATTLER_CACHE_DIR=%q\n' "${RATTLER_CACHE_DIR}"
  env | LC_ALL=C sort | sed -n '/^SLURM_/p'
} > "${PROVENANCE}/execution_environment.txt"

(
  cd "${REPO_ROOT}"
  "${PIXI_PY}" scripts/audit_vdam_coarse_prehalf_binary.py \
    --baseline-binary "${BASELINE_BINARY}" \
    --candidate-binary "${CANDIDATE_BINARY}" \
    --cuobjdump "${CUOBJDUMP}" \
    --output-json "${AUDIT_JSON}" \
    --output-markdown "${AUDIT_MARKDOWN}"
) > "${RESULTS}/binary_audit.stdout" 2> "${RESULTS}/binary_audit.stderr"

"${PIXI_PY}" - "${AUDIT_JSON}" "${BASELINE_SHA256}" "${candidate_sha256}" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
assert report["pass"] is True, report
assert report["artifacts"]["baseline_binary"]["sha256"] == sys.argv[2], report
assert report["artifacts"]["candidate_binary"]["sha256"] == sys.argv[3], report
assert report["default_enablement_allowed"] is False, report
assert report["production_wiring_evaluated"] is False, report
PY

(
  cd "${REPO_ROOT}"
  "${PIXI_PY}" - <<'PY'
import os
from pathlib import Path

import jax
import recovar
from recovar import cuda_backproject

repo = Path(os.environ["RECOVAR_EXPECTED_REPO_ROOT"]).resolve()
candidate = Path(os.environ["RECOVAR_CUDA_LIB"]).resolve()
assert str(Path(recovar.__file__).resolve()).startswith(str(repo) + "/")
assert ".pixi/envs/default/" in str(Path(jax.__file__).resolve())
assert jax.default_backend() == "gpu"
devices = jax.devices("gpu")
assert len(devices) == 1, devices
assert "H100" in str(devices[0].device_kind), devices[0]
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
assert Path(cuda_backproject._loaded_lib_path).resolve() == candidate
assert not str(candidate).startswith(str(repo) + "/")
print("VDAM coarse prehalf runtime gate passed", devices, recovar.__file__)
PY
) > "${PROVENANCE}/runtime_gate.stdout" 2> "${PROVENANCE}/runtime_gate.stderr"
vdam_verify_selected_gpu "${selected_gpu_uuid}"

(
  cd "${REPO_ROOT}"
  "${PIXI_PY}" -m pytest -vv --run-gpu \
    --junitxml="${PYTEST_JUNIT}" \
    "${FOCUSED_GPU_TEST}"
) > "${RESULTS}/focused_gpu_pytest.stdout" 2> "${RESULTS}/focused_gpu_pytest.stderr"

"${PIXI_PY}" - "${PYTEST_JUNIT}" <<'PY'
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
cases = root.findall(".//testcase")
assert len(cases) == 1, len(cases)
for case in cases:
    assert case.find("failure") is None, ET.tostring(case, encoding="unicode")
    assert case.find("error") is None, ET.tostring(case, encoding="unicode")
    assert case.find("skipped") is None, ET.tostring(case, encoding="unicode")
PY
test "$(sha256sum "${CANDIDATE_BINARY}" | awk '{print $1}')" = "${candidate_sha256}"
printf 'after_focused_gpu_test %s\n' "${candidate_sha256}" \
  >> "${PROVENANCE}/candidate_sha256_checkpoints.txt"
vdam_verify_selected_gpu "${selected_gpu_uuid}"

(
  cd "${REPO_ROOT}"
  "${PIXI_PY}" scripts/benchmark_vdam_coarse_prehalf.py \
    --repo "${REPO_ROOT}" \
    --output-json "${BENCHMARK_JSON}"
) > "${RESULTS}/gf46_benchmark.stdout" 2> "${RESULTS}/gf46_benchmark.stderr"

"${PIXI_PY}" - "${BENCHMARK_JSON}" "${EXPECTED_REPO_HEAD}" \
  "${candidate_sha256}" "${selected_gpu_uuid}" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
configuration = report["configuration"]
assert report["classification"] == "diagnostic_only_no_decision", report
assert configuration["profile"] == "gf46_it181_geometry", configuration
assert configuration["current_size"] == 128, configuration
assert configuration["model_max_r"] == 64, configuration
assert configuration["rotation_count"] == 36_864, configuration
assert configuration["translation_count"] == 29, configuration
assert configuration["physical_batch_size"] == 187, configuration
assert configuration["actual_batch_size"] == 187, configuration
assert configuration["repetitions"] == 5, configuration
assert configuration["seed"] == 46, configuration
assert configuration["prehalf_default"] is False, configuration
assert report["protocol"]["alternation"] == [False, True], report["protocol"]
assert set(report["apis"]) == {"serial", "multistream"}, report["apis"]
assert report["provenance"]["repository"]["head"] == sys.argv[2], report
assert report["provenance"]["repository"]["status_porcelain_v1"] == [], report
assert report["provenance"]["cuda"]["sha256"] == sys.argv[3], report
assert report["provenance"]["device"]["cuda_visible_devices"] == sys.argv[4], report
PY

test "$(sha256sum "${CANDIDATE_BINARY}" | awk '{print $1}')" = "${candidate_sha256}"
printf 'after_benchmark %s\n' "${candidate_sha256}" \
  >> "${PROVENANCE}/candidate_sha256_checkpoints.txt"
test "$(sha256sum "${BASELINE_BINARY}" | awk '{print $1}')" = "${BASELINE_SHA256}"
test "$(stat -c '%d:%i:%s:%Y:%Z' "${BASELINE_BINARY}")" = "${baseline_stat_before}"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=all)"
write_source_manifest > "${PROVENANCE}/source_manifest.final.sha256"
cmp "${PROVENANCE}/source_manifest.sha256" "${PROVENANCE}/source_manifest.final.sha256"
vdam_verify_selected_gpu "${selected_gpu_uuid}"

touch "${RESULT_ROOT}/COMPLETED"
(
  cd "${RESULT_ROOT}"
  find . -type f ! -name SHA256SUMS ! -name SHA256SUMS.sha256 -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum
) > "${RESULT_ROOT}/SHA256SUMS"
sha256sum "${RESULT_ROOT}/SHA256SUMS" > "${RESULT_ROOT}/SHA256SUMS.sha256"
find "${RESULT_ROOT}" -type f -exec chmod a-w {} +
find "${RESULT_ROOT}" -depth -type d -exec chmod a-w {} +

printf 'VDAM coarse prehalf H100 gate complete: %s\n' "${RESULT_ROOT}"
