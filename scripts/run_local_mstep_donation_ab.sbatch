#!/bin/bash
#SBATCH --job-name=local-donate-ab
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
#SBATCH --export=NIL
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/local-donate-ab-%j.out

set -euo pipefail

# This script accepts exactly two positional values: a reviewed launch
# manifest and its SHA-256.  Science paths, source/input hashes, GPU identity,
# node identity, runtime binding, and test count never come from the submission
# environment.  --export=NIL prevents BASH_ENV and toolchain/tuning variables
# from reaching /bin/bash before this code executes.
if (( $# != 2 )); then
  /usr/bin/printf 'usage: sbatch %s /absolute/launch_manifest.json SHA256\n' "$0" >&2
  exit 64
fi
readonly LAUNCH_MANIFEST_ARG=$1
readonly EXPECTED_LAUNCH_MANIFEST_SHA256=$2
export PATH=/usr/bin:/bin
export LANG=C
export LC_ALL=C
unset CDPATH ENV BASH_ENV GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE 2>/dev/null || true

readonly CUDA_BUILD_TOOLKIT=/usr/local/cuda-12.6
readonly CUDA_RUNTIME_TOOLKIT=/usr/local/cuda-13.1
readonly RELION_RUNTIME=/scratch/gpfs/GILLES/mg6942/recovar_dev/relion_k4_100k_dispatchv2_20260717
readonly MPI_ROOT=/usr/local/openmpi/cuda-12.6/4.1.6/gcc
# The runner resolves its expected preload path before comparing the sealed
# environment.  Seal that same resolved file into both sides of the contract.
readonly CUSPARSE_LIBRARY=$(/usr/bin/readlink -f -- \
  "${CUDA_BUILD_TOOLKIT}/lib64/libcusparse.so.12")
readonly CHECKPOINT_ITERATION=180
readonly PROFILED_ITERATION=181
readonly NR_ITER_SCHEDULE=200
readonly -a FOCUSED_TESTS=(
  tests/unit/initial_model/test_local_mstep_donation_ab.py
  tests/unit/initial_model/test_local_mstep_donation_input_closure.py
)

[[ "${EXPECTED_LAUNCH_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]]
[[ "${LAUNCH_MANIFEST_ARG}" = /* ]]
readonly LAUNCH_MANIFEST=$(/usr/bin/readlink -f -- "${LAUNCH_MANIFEST_ARG}")
test "${LAUNCH_MANIFEST}" = "${LAUNCH_MANIFEST_ARG}"
test -s "${LAUNCH_MANIFEST}"
test "$(/usr/bin/sha256sum "${LAUNCH_MANIFEST}" | /usr/bin/awk '{print $1}')" = \
  "${EXPECTED_LAUNCH_MANIFEST_SHA256}"

readonly PREARM_TMP=$(/usr/bin/mktemp -d "/tmp/recovar-donation-${SLURM_JOB_ID:?}.XXXXXX")
/usr/bin/python3 - "${LAUNCH_MANIFEST}" > "${PREARM_TMP}/manifest.values" <<'PY'
import json
import os
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text())
keys = {
    "schema", "output_root", "repo_root", "checkpoint_optimiser",
    "input_star", "data_dir", "particle_stacks", "expected_repo_head",
    "expected_source_manifest_sha256", "expected_input_manifest_sha256",
    "expected_node_name", "target_gpu_uuid", "relion_bind_binary",
    "expected_relion_bind_sha256", "expected_focused_test_count",
}
if not isinstance(payload, dict) or set(payload) != keys:
    raise SystemExit("launch manifest has missing or unexpected keys")
if payload["schema"] != "recovar.local_mstep_donation_launch_manifest.v1":
    raise SystemExit("launch manifest schema drifted")
for key, value in payload.items():
    if key in {"particle_stacks", "expected_focused_test_count"}:
        continue
    if not isinstance(value, str) or not value or any(c in value for c in "\0\r\n"):
        raise SystemExit(f"invalid scalar launch value: {key}")
for key in (
    "expected_source_manifest_sha256",
    "expected_input_manifest_sha256",
    "expected_relion_bind_sha256",
):
    if re.fullmatch(r"[0-9a-f]{64}", payload[key]) is None:
        raise SystemExit(f"invalid SHA-256: {key}")
if re.fullmatch(r"[0-9a-f]{40,64}", payload["expected_repo_head"]) is None:
    raise SystemExit("expected_repo_head must be a full commit ID")
if re.fullmatch(r"GPU-[0-9A-Fa-f-]+", payload["target_gpu_uuid"]) is None:
    raise SystemExit("target_gpu_uuid must be a physical GPU UUID")
count = payload["expected_focused_test_count"]
if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
    raise SystemExit("expected_focused_test_count must be positive")

strict_files = {
    "checkpoint_optimiser", "input_star", "relion_bind_binary",
}
strict_dirs = {"repo_root", "data_dir"}
for key in ("output_root", *sorted(strict_files | strict_dirs)):
    raw = Path(payload[key])
    if not raw.is_absolute():
        raise SystemExit(f"{key} is not absolute")
    resolved = raw.resolve(strict=key != "output_root")
    if resolved != raw:
        raise SystemExit(f"{key} is not canonical")
    if key in strict_files and (not resolved.is_file() or resolved.stat().st_size <= 0):
        raise SystemExit(f"{key} is not a non-empty regular file")
    if key in strict_dirs and not resolved.is_dir():
        raise SystemExit(f"{key} is not a directory")

particles = payload["particle_stacks"]
if not isinstance(particles, list) or not particles:
    raise SystemExit("particle_stacks must be a non-empty list")
resolved_particles = []
for raw_value in particles:
    if not isinstance(raw_value, str) or not raw_value or any(c in raw_value for c in "\0\r\n"):
        raise SystemExit("invalid particle-stack path")
    raw = Path(raw_value)
    resolved = raw.resolve(strict=True)
    if not raw.is_absolute() or raw != resolved:
        raise SystemExit("particle-stack paths must be absolute and canonical")
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise SystemExit("particle stack must be a non-empty regular file")
    resolved_particles.append(str(resolved))
if resolved_particles != sorted(set(resolved_particles)):
    raise SystemExit("particle stacks must be sorted and unique")

ordered = [
    payload["output_root"], payload["repo_root"], payload["checkpoint_optimiser"],
    payload["input_star"], payload["data_dir"], payload["expected_repo_head"],
    payload["expected_source_manifest_sha256"],
    payload["expected_input_manifest_sha256"], payload["expected_node_name"],
    payload["target_gpu_uuid"], payload["relion_bind_binary"],
    payload["expected_relion_bind_sha256"], str(count), str(len(resolved_particles)),
    *resolved_particles,
]
for value in ordered:
    print(value)
PY
mapfile -t MANIFEST_VALUES < "${PREARM_TMP}/manifest.values"
(( ${#MANIFEST_VALUES[@]} >= 15 ))
readonly ROOT=${MANIFEST_VALUES[0]}
readonly REPO_ROOT=${MANIFEST_VALUES[1]}
readonly CHECKPOINT=${MANIFEST_VALUES[2]}
readonly DATA_STAR=${MANIFEST_VALUES[3]}
readonly DATA_DIR=${MANIFEST_VALUES[4]}
readonly EXPECTED_REPO_HEAD=${MANIFEST_VALUES[5]}
readonly EXPECTED_SOURCE_MANIFEST_SHA256=${MANIFEST_VALUES[6]}
readonly EXPECTED_INPUT_MANIFEST_SHA256=${MANIFEST_VALUES[7]}
readonly EXPECTED_NODE_NAME=${MANIFEST_VALUES[8]}
readonly TARGET_GPU_UUID=${MANIFEST_VALUES[9]}
readonly RELION_BIND_BINARY_OVERRIDE=${MANIFEST_VALUES[10]}
readonly EXPECTED_RELION_BIND_SHA256=${MANIFEST_VALUES[11]}
readonly EXPECTED_FOCUSED_TEST_COUNT=${MANIFEST_VALUES[12]}
readonly PARTICLE_COUNT=${MANIFEST_VALUES[13]}
[[ "${PARTICLE_COUNT}" =~ ^[1-9][0-9]*$ ]]
(( ${#MANIFEST_VALUES[@]} == 14 + PARTICLE_COUNT ))
readonly -a PARTICLE_STACKS=("${MANIFEST_VALUES[@]:14}")

readonly PIXI_PY=${REPO_ROOT}/.pixi/envs/default/bin/python
readonly PROVENANCE=${ROOT}/provenance
readonly RUNTIME=${ROOT}/runtime
readonly RESULTS=${ROOT}/results
readonly RUNS=${ROOT}/runs
readonly CUDA_BINARY=${RUNTIME}/cuda/libcuda_backproject.so
readonly RELION_BIND_DIR=${RUNTIME}/relion_bind
readonly ANALYSIS=${RESULTS}/donation_ab_analysis.json
readonly JOB_TMP_ROOT=/scratch/gpfs/GILLES/mg6942/tmp/${SLURM_JOB_ID}
readonly JOB_PIXI_HOME=/scratch/gpfs/GILLES/mg6942/pixi_home/${SLURM_JOB_ID}
readonly JOB_RATTLER_CACHE=/scratch/gpfs/GILLES/mg6942/rattler_cache/${SLURM_JOB_ID}
readonly SEALED_HOME=${JOB_TMP_ROOT}/home
readonly SEALED_BUILD_HOME=${PREARM_TMP}/build_home
readonly SEALED_PROCESS_PATH=${CUDA_RUNTIME_TOOLKIT}/bin:${REPO_ROOT}/.pixi/envs/default/bin:/usr/bin:/bin
readonly SEALED_LD_LIBRARY_PATH=${RELION_RUNTIME}/system_dev_overlay/lib:${MPI_ROOT}/lib64:${CUDA_BUILD_TOOLKIT}/lib64:${CUDA_BUILD_TOOLKIT}/targets/x86_64-linux/lib:${CUDA_RUNTIME_TOOLKIT}/lib64:${CUDA_RUNTIME_TOOLKIT}/targets/x86_64-linux/lib
readonly -a SOURCE_FILES=(
  recovar/cuda/Makefile
  recovar/cuda/cuda_backproject.cu
  recovar/cuda_backproject.py
  recovar/em/local/local_big_jit.py
  recovar/em/local/local_em_engine.py
  recovar/em/vdam/dense_adapter.py
  recovar/em/vdam/driver.py
  recovar/em/vdam/iteration_loop.py
  recovar/em/vdam/schedules.py
  recovar/utils/parity_provenance.py
  recovar/commands/initial_model.py
  scripts/run_vdam_late_iteration_profile.py
  scripts/run_local_mstep_donation_ab.py
  scripts/analyze_local_mstep_donation_ab.py
  scripts/run_local_mstep_donation_ab.sbatch
  scripts/vdam_gpu_selection.sh
  tests/conftest.py
  tests/unit/initial_model/test_local_mstep_donation_ab.py
  tests/unit/initial_model/test_local_mstep_donation_input_closure.py
  pixi.lock
  pixi.toml
  pyproject.toml
)

on_error() {
  local status=$?
  local message="local M-step donation A/B failed: status=${status} line=${BASH_LINENO[0]} command=${BASH_COMMAND}"
  trap - ERR
  /usr/bin/printf '%s\n' "${message}" >&2
  if [[ -d "${PROVENANCE}" ]]; then
    /usr/bin/printf '%s\n' "${message}" > "${PROVENANCE}/failure.txt"
  fi
  exit "${status}"
}
trap on_error ERR

write_source_manifest() {
  (
    cd "${REPO_ROOT}"
    /usr/bin/sha256sum "${SOURCE_FILES[@]}"
  )
}

write_input_manifest() {
  local destination=$1
  local resolved_destination=$2
  (
    cd "${REPO_ROOT}"
    /usr/bin/env -i \
      HOME="${SEALED_BUILD_HOME}" PATH="${REPO_ROOT}/.pixi/envs/default/bin:/usr/bin:/bin" \
      LANG=C LC_ALL=C PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 PYTHONUTF8=1 \
      "${PIXI_PY}" - "${destination}" "${resolved_destination}" \
      "${CHECKPOINT}" "${DATA_STAR}" "${DATA_DIR}" "${PARTICLE_STACKS[@]}" <<'PY'
import sys
from pathlib import Path

from scripts.run_local_mstep_donation_ab import write_gf46_input_manifest

digest = write_gf46_input_manifest(
    Path(sys.argv[1]),
    Path(sys.argv[3]),
    Path(sys.argv[4]),
    Path(sys.argv[5]),
    [Path(value) for value in sys.argv[6:]],
    resolved_output=Path(sys.argv[2]),
)
print(digest)
PY
  )
}

test ! -e "${ROOT}"
test -x "${PIXI_PY}"
test -x "${CUDA_BUILD_TOOLKIT}/bin/nvcc"
test -d "${CUDA_RUNTIME_TOOLKIT}/lib64"
test -s "${CUSPARSE_LIBRARY}"
test "${SLURMD_NODENAME:?}" = "${EXPECTED_NODE_NAME}"
test "${CHECKPOINT_ITERATION}" -eq 180
test "${PROFILED_ITERATION}" -eq 181
test "${NR_ITER_SCHEDULE}" -eq 200
test "$(/usr/bin/git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(/usr/bin/git -C "${REPO_ROOT}" status --porcelain=v1)"
test "$(/usr/bin/sha256sum "${RELION_BIND_BINARY_OVERRIDE}" | /usr/bin/awk '{print $1}')" = \
  "${EXPECTED_RELION_BIND_SHA256}"

# The helper only observes Slurm's allocated devices and the manifest-pinned
# physical UUID.  PATH is already the fixed /usr/bin:/bin pre-arm path.
source "${REPO_ROOT}/scripts/vdam_gpu_selection.sh"
vdam_select_target_gpu "${TARGET_GPU_UUID}" 0
readonly selected_gpu_uuid=${VDAM_SELECTED_GPU_UUID}
test "${selected_gpu_uuid}" = "${TARGET_GPU_UUID}"
gpu_name=$(/usr/bin/nvidia-smi --id="${selected_gpu_uuid}" \
  --query-gpu=name --format=csv,noheader | /usr/bin/sed -n '1p' | /usr/bin/sed 's/[[:space:]]*$//')
[[ "${gpu_name}" == *H100* ]]

/usr/bin/mkdir -p "${PROVENANCE}" "${RUNTIME}/cuda" "${RELION_BIND_DIR}" "${RESULTS}" "${RUNS}"
/usr/bin/touch "${ROOT}/SAFE_TO_DELETE"
/usr/bin/cp --reflink=auto "${LAUNCH_MANIFEST}" "${PROVENANCE}/launch_manifest.json"
test "$(/usr/bin/sha256sum "${PROVENANCE}/launch_manifest.json" | /usr/bin/awk '{print $1}')" = \
  "${EXPECTED_LAUNCH_MANIFEST_SHA256}"
write_source_manifest > "${PROVENANCE}/source_manifest.sha256"
source_manifest_sha256=$(/usr/bin/sha256sum "${PROVENANCE}/source_manifest.sha256" | /usr/bin/awk '{print $1}')
test "${source_manifest_sha256}" = "${EXPECTED_SOURCE_MANIFEST_SHA256}"
/usr/bin/git -C "${REPO_ROOT}" rev-parse HEAD > "${PROVENANCE}/repo_head.txt"
/usr/bin/git -C "${REPO_ROOT}" symbolic-ref --short HEAD > "${PROVENANCE}/repo_branch.txt"
/usr/bin/git -C "${REPO_ROOT}" status --porcelain=v1 > "${PROVENANCE}/repo_status.txt"
/usr/bin/printf '%s\n' "${SLURM_JOB_ID}" > "${PROVENANCE}/slurm_job_id.txt"
/usr/bin/printf '%s\n' "${selected_gpu_uuid}" > "${PROVENANCE}/selected_gpu_uuid.txt"
/usr/bin/printf '%s\n' "${gpu_name}" > "${PROVENANCE}/gpu_name.txt"
/usr/bin/printf '%s\n' "${VDAM_VISIBLE_GPU_UUIDS_CSV}" > "${PROVENANCE}/allocated_gpu_uuids.csv"
/usr/bin/nvidia-smi -q > "${PROVENANCE}/nvidia_smi.txt"
/usr/bin/env | /usr/bin/sort > "${PROVENANCE}/prearm_environment.txt"

write_input_manifest "${PROVENANCE}/input_manifest.json" \
  "${PROVENANCE}/resolved_inputs.json" > "${PROVENANCE}/input_manifest.writer.stdout"
input_manifest_sha256=$(/usr/bin/sha256sum "${PROVENANCE}/input_manifest.json" | /usr/bin/awk '{print $1}')
test "${input_manifest_sha256}" = "${EXPECTED_INPUT_MANIFEST_SHA256}"

/usr/bin/mkdir -p "${SEALED_BUILD_HOME}"
/usr/bin/env -i HOME="${SEALED_BUILD_HOME}" PATH=/usr/bin:/bin LANG=C LC_ALL=C \
  /usr/bin/make -rR -B -C "${REPO_ROOT}/recovar/cuda" \
  PYTHON="${PIXI_PY}" \
  NVCC="${CUDA_BUILD_TOOLKIT}/bin/nvcc" \
  CUDA_ARCH='-gencode arch=compute_90,code=sm_90' \
  LIB="${CUDA_BINARY}" \
  > "${PROVENANCE}/cuda_build.stdout" \
  2> "${PROVENANCE}/cuda_build.stderr"
test -s "${CUDA_BINARY}"
cuda_sha256=$(/usr/bin/sha256sum "${CUDA_BINARY}" | /usr/bin/awk '{print $1}')
/usr/bin/printf '%s  %s\n' "${cuda_sha256}" "${CUDA_BINARY}" > "${PROVENANCE}/qualified_cuda.sha256"
/usr/bin/cp --reflink=auto "${RELION_BIND_BINARY_OVERRIDE}" "${RELION_BIND_DIR}/"
readonly RELION_BIND_BINARY=${RELION_BIND_DIR}/$(/usr/bin/basename "${RELION_BIND_BINARY_OVERRIDE}")
test "$(/usr/bin/sha256sum "${RELION_BIND_BINARY}" | /usr/bin/awk '{print $1}')" = \
  "${EXPECTED_RELION_BIND_SHA256}"
/usr/bin/sha256sum "${RELION_BIND_BINARY_OVERRIDE}" "${RELION_BIND_BINARY}" \
  "${CUSPARSE_LIBRARY}" "${CUDA_BINARY}" > "${PROVENANCE}/runtime_inputs.sha256"

readonly -a SEALED_COMMON_ENV=(
  HOME="${SEALED_HOME}"
  PATH="${SEALED_PROCESS_PATH}"
  LD_LIBRARY_PATH="${SEALED_LD_LIBRARY_PATH}"
  LD_PRELOAD="${CUSPARSE_LIBRARY}"
  LANG=C.UTF-8
  LC_ALL=C.UTF-8
  PYTHONHASHSEED=0
  PYTHONNOUSERSITE=1
  PYTHONUNBUFFERED=1
  PYTHONUTF8=1
  TMPDIR="${JOB_TMP_ROOT}"
  PIXI_HOME="${JOB_PIXI_HOME}"
  RATTLER_CACHE_DIR="${JOB_RATTLER_CACHE}"
  CUDA_VISIBLE_DEVICES="${selected_gpu_uuid}"
  CUDA_LAUNCH_BLOCKING=0
  RECOVAR_EXPECTED_REPO_ROOT="${REPO_ROOT}"
  RECOVAR_CUDA_LIB="${CUDA_BINARY}"
  RECOVAR_ENABLE_CUSTOM_CUDA=1
  RECOVAR_INITIAL_MODEL_PROFILE=1
  RECOVAR_RELION_BIND_BUILD_DIR="${RELION_BIND_DIR}"
  XLA_PYTHON_CLIENT_PREALLOCATE=false
  JAX_LOG_COMPILES=1
  JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
)
readonly PRECHECK_CACHE=${RUNTIME}/precheck_jax_cache
/usr/bin/mkdir -p "${SEALED_HOME}" "${JOB_TMP_ROOT}" "${JOB_PIXI_HOME}" \
  "${JOB_RATTLER_CACHE}" "${PRECHECK_CACHE}"
/usr/bin/touch "${SEALED_HOME}/SAFE_TO_DELETE" "${JOB_TMP_ROOT}/SAFE_TO_DELETE" \
  "${JOB_PIXI_HOME}/SAFE_TO_DELETE" "${JOB_RATTLER_CACHE}/SAFE_TO_DELETE"

(
  cd "${REPO_ROOT}"
  /usr/bin/env -i "${SEALED_COMMON_ENV[@]}" \
    JAX_COMPILATION_CACHE_DIR="${PRECHECK_CACHE}" "${PIXI_PY}" - <<'PY'
import os
import pathlib
import sys

import jax
import recovar
from recovar import cuda_backproject
from recovar.relion_bind import _relion_bind_core as relion_bind_core
from recovar.utils.parity_provenance import assert_parity_ancestors

repo = pathlib.Path(os.environ["RECOVAR_EXPECTED_REPO_ROOT"]).resolve()
pixi_env = (repo / ".pixi" / "envs" / "default").resolve()
assert pathlib.Path(recovar.__file__).resolve().is_relative_to(repo)
assert pathlib.Path(sys.prefix).resolve() == pixi_env
assert pathlib.Path(sys.executable).resolve().is_relative_to(pixi_env)
assert pathlib.Path(jax.__file__).resolve().is_relative_to(pixi_env)
assert jax.default_backend() == "gpu"
assert len(jax.devices("gpu")) == 1
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
assert callable(getattr(relion_bind_core, "vdam_rnd_unif_sequence", None))
assert_parity_ancestors()
print("local donation A/B GPU/runtime gate passed", jax.devices(), recovar.__file__)
PY
) > "${PROVENANCE}/runtime_gate.stdout" 2> "${PROVENANCE}/runtime_gate.stderr"
vdam_verify_selected_gpu "${selected_gpu_uuid}"

(
  cd "${REPO_ROOT}"
  /usr/bin/env -i "${SEALED_COMMON_ENV[@]}" \
    JAX_COMPILATION_CACHE_DIR="${PRECHECK_CACHE}" \
    "${PIXI_PY}" -m pytest -vv --run-gpu \
    --junitxml="${RESULTS}/pytest.junit.xml" "${FOCUSED_TESTS[@]}"
) > "${RESULTS}/pytest.stdout" 2> "${RESULTS}/pytest.stderr"
/usr/bin/python3 - "${RESULTS}/pytest.junit.xml" "${EXPECTED_FOCUSED_TEST_COUNT}" <<'PY'
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

/usr/bin/printf 'ordinal\trepeat\tarm\n' > "${PROVENANCE}/execution_order.tsv"

run_arm() {
  local ordinal=$1
  local repeat=$2
  local arm=$3
  local run_dir=${RUNS}/repeat-${repeat}/${arm}
  local output_dir=${run_dir}/result
  local sample_file=${run_dir}/nvidia_smi.tsv
  local cache_dir=${run_dir}/jax_cache
  local -a particle_args=()
  local particle
  for particle in "${PARTICLE_STACKS[@]}"; do
    particle_args+=(--expected-particle-stack "${particle}")
  done
  /usr/bin/mkdir -p "${run_dir}"
  test ! -e "${cache_dir}"
  /usr/bin/mkdir "${cache_dir}"
  test -z "$(/usr/bin/find "${cache_dir}" -mindepth 1 -print -quit)"
  /usr/bin/touch "${run_dir}/SAFE_TO_DELETE"
  /usr/bin/python3 - "${run_dir}/launch_contract.json" "${ordinal}" \
    "${repeat}" "${arm}" "${cache_dir}" "${input_manifest_sha256}" \
    "${EXPECTED_LAUNCH_MANIFEST_SHA256}" <<'PY'
import json
import sys
from pathlib import Path

payload = {
    "schema": "recovar.local_mstep_donation_launch.v1",
    "ordinal": int(sys.argv[2]),
    "repeat": sys.argv[3],
    "arm": sys.argv[4],
    "jax_cache_dir": str(Path(sys.argv[5]).resolve()),
    "jax_cache_initial_empty": True,
    "input_manifest_sha256": sys.argv[6],
    "launch_manifest_sha256": sys.argv[7],
    "checkpoint_iteration": 180,
    "profiled_iteration": 181,
    "nr_iter_schedule": 200,
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
  /usr/bin/printf '%s\t%s\t%s\n' "${ordinal}" "${repeat}" "${arm}" \
    >> "${PROVENANCE}/execution_order.tsv"
  /usr/bin/printf 'unix_ns\tmemory_mib\n' > "${sample_file}"
  (
    cd "${REPO_ROOT}"
    /usr/bin/env -i "${SEALED_COMMON_ENV[@]}" \
      JAX_COMPILATION_CACHE_DIR="${cache_dir}" \
      "${PIXI_PY}" -m scripts.run_local_mstep_donation_ab \
        --arm "${arm}" \
        --checkpoint-optimiser "${CHECKPOINT}" \
        --input-star "${DATA_STAR}" \
        --data-dir "${DATA_DIR}" \
        "${particle_args[@]}" \
        --output-root "${output_dir}" \
        --checkpoint-iteration "${CHECKPOINT_ITERATION}" \
        --nr-iter "${NR_ITER_SCHEDULE}" \
        --random-seed 29 \
        --image-batch-size 500 \
        --exact-local-bucket-radix 4 \
        --exact-local-physical-order-chunk-size 0 \
        --input-manifest "${PROVENANCE}/input_manifest.json" \
        --expected-input-manifest-sha256 "${input_manifest_sha256}" \
        --launch-manifest "${PROVENANCE}/launch_manifest.json" \
        --expected-launch-manifest-sha256 "${EXPECTED_LAUNCH_MANIFEST_SHA256}" \
        --expected-jax-cache-dir "${cache_dir}" \
        --expected-cuda-lib "${CUDA_BINARY}" \
        --expected-relion-bind-dir "${RELION_BIND_DIR}" \
        --expected-cusparse-library "${CUSPARSE_LIBRARY}" \
        --expected-repo-head "${EXPECTED_REPO_HEAD}" \
        --expected-gpu-uuid "${selected_gpu_uuid}"
  ) > "${run_dir}/runner.stdout" 2> "${run_dir}/runner.stderr" &
  local runner_pid=$!
  while /bin/kill -0 "${runner_pid}" 2>/dev/null; do
    local timestamp_ns
    local memory_mib
    timestamp_ns=$(/usr/bin/date +%s%N)
    memory_mib=$(/usr/bin/nvidia-smi --id="${selected_gpu_uuid}" \
      --query-gpu=memory.used --format=csv,noheader,nounits | \
      /usr/bin/sed -n '1p' | /usr/bin/tr -d '[:space:]')
    if [[ "${memory_mib}" =~ ^[0-9]+$ ]]; then
      /usr/bin/printf '%s\t%s\n' "${timestamp_ns}" "${memory_mib}" >> "${sample_file}"
    fi
    /usr/bin/sleep 0.05
  done
  wait "${runner_pid}"
  test -s "${output_dir}/donation_arm_summary.json"
  test -s "${output_dir}/TIMED_BEGIN"
  test -s "${output_dir}/TIMED_END"
  /usr/bin/find "${cache_dir}" -mindepth 1 -printf '%P\n' | /usr/bin/sort \
    > "${run_dir}/jax_cache_entries.final.txt"
  test -s "${run_dir}/jax_cache_entries.final.txt"
  vdam_verify_selected_gpu "${selected_gpu_uuid}"
}

# Cross order on one physical H100; every entry is a fresh Python process and
# has a private, initially empty persistent compilation cache.
run_arm 1 01 donated
run_arm 2 01 control
run_arm 3 02 control
run_arm 4 02 donated
run_arm 5 03 donated
run_arm 6 03 control

(
  cd "${REPO_ROOT}"
  /usr/bin/env -i "${SEALED_COMMON_ENV[@]}" \
    JAX_COMPILATION_CACHE_DIR="${PRECHECK_CACHE}" \
    "${PIXI_PY}" -m scripts.analyze_local_mstep_donation_ab \
    --root "${ROOT}" \
    --output "${ANALYSIS}" \
    --expected-repo-head "${EXPECTED_REPO_HEAD}" \
    --expected-gpu-uuid "${selected_gpu_uuid}" \
    --expected-input-manifest-sha256 "${input_manifest_sha256}" \
    --expected-launch-manifest-sha256 "${EXPECTED_LAUNCH_MANIFEST_SHA256}"
) > "${RESULTS}/analysis.stdout" 2> "${RESULTS}/analysis.stderr"
test -s "${ANALYSIS}"
/usr/bin/python3 - "${ANALYSIS}" "${EXPECTED_INPUT_MANIFEST_SHA256}" \
  "${EXPECTED_LAUNCH_MANIFEST_SHA256}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
assert payload["schema"] == "recovar.local_mstep_donation_ab_analysis.v4"
assert payload["passed"] is True
assert payload["cold_and_warm_are_separate_phase_matched_cohorts"] is True
assert payload["numeric_qualification_status"] in {"exact_pass", "inconclusive_3x3"}
if payload["numeric_qualification_status"] == "inconclusive_3x3":
    assert payload["numeric_qualification_allowed"] is False
    assert payload["same_job_preliminary_speed_signal_allowed"] is False
assert payload["same_declared_discrete_science_metadata"] is True
assert payload["arithmetic_changed"] is False
assert payload["production_xhalf_adjoint_program_observed"] is True
assert payload["alias_contract_passed"] is True
assert payload["unique_fresh_jax_cache_count"] == 6
assert payload["speed_claim_allowed"] is False
assert payload["memory_claim_allowed"] is False
assert payload["input_manifest_sha256"] == sys.argv[2]
assert payload["launch_manifest_sha256"] == sys.argv[3]
assert payload["default_promotion_allowed"] is False
PY

test "$(/usr/bin/sha256sum "${CUDA_BINARY}" | /usr/bin/awk '{print $1}')" = "${cuda_sha256}"
test "$(/usr/bin/git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(/usr/bin/git -C "${REPO_ROOT}" status --porcelain=v1)"
test "$(/usr/bin/sha256sum "${PROVENANCE}/launch_manifest.json" | /usr/bin/awk '{print $1}')" = \
  "${EXPECTED_LAUNCH_MANIFEST_SHA256}"
write_source_manifest > "${PROVENANCE}/source_manifest.final.sha256"
/usr/bin/cmp "${PROVENANCE}/source_manifest.sha256" "${PROVENANCE}/source_manifest.final.sha256"
write_input_manifest "${PROVENANCE}/input_manifest.final.json" \
  "${PROVENANCE}/resolved_inputs.final.json" > "${PROVENANCE}/input_manifest.final.writer.stdout"
/usr/bin/cmp "${PROVENANCE}/input_manifest.json" "${PROVENANCE}/input_manifest.final.json"
/usr/bin/cmp "${PROVENANCE}/resolved_inputs.json" "${PROVENANCE}/resolved_inputs.final.json"
test "$(/usr/bin/sha256sum "${PROVENANCE}/input_manifest.final.json" | /usr/bin/awk '{print $1}')" = \
  "${EXPECTED_INPUT_MANIFEST_SHA256}"
/usr/bin/sha256sum "${RELION_BIND_BINARY_OVERRIDE}" "${RELION_BIND_BINARY}" \
  "${CUSPARSE_LIBRARY}" "${CUDA_BINARY}" > "${PROVENANCE}/runtime_inputs.final.sha256"
/usr/bin/cmp "${PROVENANCE}/runtime_inputs.sha256" "${PROVENANCE}/runtime_inputs.final.sha256"
vdam_verify_selected_gpu "${selected_gpu_uuid}"
analysis_sha256=$(/usr/bin/sha256sum "${ANALYSIS}" | /usr/bin/awk '{print $1}')
junit_sha256=$(/usr/bin/sha256sum "${RESULTS}/pytest.junit.xml" | /usr/bin/awk '{print $1}')
/usr/bin/python3 - "${PROVENANCE}/run.json" "${SLURM_JOB_ID}" \
  "${EXPECTED_REPO_HEAD}" "${selected_gpu_uuid}" "${source_manifest_sha256}" \
  "${input_manifest_sha256}" "${EXPECTED_LAUNCH_MANIFEST_SHA256}" \
  "${cuda_sha256}" "${analysis_sha256}" "${junit_sha256}" <<'PY'
import json
import sys
from pathlib import Path

payload = {
    "schema": "recovar.local_mstep_donation_ab_slurm.v4",
    "classification": "diagnostic_performance_only",
    "job_id": sys.argv[2],
    "git_head": sys.argv[3],
    "gpu_uuid": sys.argv[4],
    "source_manifest_sha256": sys.argv[5],
    "input_manifest_sha256": sys.argv[6],
    "launch_manifest_sha256": sys.argv[7],
    "cuda_sha256": sys.argv[8],
    "analysis_sha256": sys.argv[9],
    "junit_sha256": sys.argv[10],
    "execution_order": [
        "donated-01", "control-01", "control-02",
        "donated-02", "donated-03", "control-03",
    ],
    "fresh_process_per_arm": True,
    "unique_fresh_jax_cache_per_arm": True,
    "checkpoint_iteration": 180,
    "profiled_iteration": 181,
    "nr_iter_schedule": 200,
    "same_physical_h100": True,
    "runtime_claim_scope": "same_job_preliminary_only_runtime_contents_not_archivally_hashed",
    "speed_claim_allowed": False,
    "memory_claim_allowed": False,
    "default_promotion_allowed": False,
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
/usr/bin/touch "${ROOT}/COMPLETED"
