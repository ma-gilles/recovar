#!/usr/bin/env bash
#SBATCH --job-name=vdam-hybrid-true200
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --nodelist=della-h19g3
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-hybrid-true200-%j.out

set -euo pipefail

if [[ "$#" -ne 0 ]]; then
  if [[ "$#" -ne 5 ]]; then
    echo "usage: $0 [ROOT LAUNCH_MANIFEST LAUNCH_SHA256 RESUME SENTINEL]" >&2
    exit 2
  fi
  VDAM_TRUE200_ROOT=$1
  VDAM_TRUE200_LAUNCH_MANIFEST=$2
  EXPECTED_LAUNCH_MANIFEST_SHA256=$3
  VDAM_TRUE200_RESUME=$4
  VDAM_TRUE200_SENTINEL=$5
fi

: "${VDAM_TRUE200_ROOT:?set a fresh disposable result root}"
: "${VDAM_TRUE200_LAUNCH_MANIFEST:?set the reviewed launch manifest}"
: "${EXPECTED_LAUNCH_MANIFEST_SHA256:?pin the reviewed launch manifest SHA-256}"
: "${SLURM_JOB_ID:?the true-200 harness must run under Slurm}"

ROOT=${VDAM_TRUE200_ROOT}
LAUNCH_MANIFEST=$(realpath "${VDAM_TRUE200_LAUNCH_MANIFEST}")
test -s "${LAUNCH_MANIFEST}"
test "$(sha256sum "${LAUNCH_MANIFEST}" | awk '{print $1}')" = "${EXPECTED_LAUNCH_MANIFEST_SHA256}"
REPO_ROOT=$(jq -er '.repo_root' "${LAUNCH_MANIFEST}")
FIXTURE=$(jq -er '.fixture_dir' "${LAUNCH_MANIFEST}")
NATIVE_ROOT=$(jq -er '.native_reference_root' "${LAUNCH_MANIFEST}")
PIXI_PY=$(jq -er '.files.interpreter.path' "${LAUNCH_MANIFEST}")
QUALIFIED_CUDA_OVERRIDE=$(jq -er '.files.qualified_cuda.path' "${LAUNCH_MANIFEST}")
RELION_BIND_BINARY_OVERRIDE=$(jq -er '.files.relion_bind.path' "${LAUNCH_MANIFEST}")
GPU_SELECTION_HELPER=$(jq -er '.files.gpu_selection_helper.path' "${LAUNCH_MANIFEST}")
CUSPARSE_LIBRARY=$(jq -er '.files.cusparse_library.path' "${LAUNCH_MANIFEST}")
EXPECTED_CUSPARSE_SHA256=$(jq -er '.files.cusparse_library.sha256' "${LAUNCH_MANIFEST}")
EXPECTED_OVERLAY_HEAD=$(jq -er '.expected_overlay_head' "${LAUNCH_MANIFEST}")
EXPECTED_PRODUCTION_CANDIDATE_HEAD=$(jq -er '.production_candidate_head' "${LAUNCH_MANIFEST}")
EXPECTED_SOURCE_MANIFEST_SHA256=$(jq -er '.source_manifest.sha256' "${LAUNCH_MANIFEST}")
EXPECTED_ACCEPTANCE_CONFIG_SHA256=$(jq -er '.files.acceptance.sha256' "${LAUNCH_MANIFEST}")
EXPECTED_ANALYZER_SOURCE_SHA256=$(jq -er '.files.analyzer.sha256' "${LAUNCH_MANIFEST}")
EXPECTED_HARNESS_SOURCE_SHA256=$(jq -er '.files.harness.sha256' "${LAUNCH_MANIFEST}")
EXPECTED_GPU_SELECTION_HELPER_SHA256=$(jq -er '.files.gpu_selection_helper.sha256' "${LAUNCH_MANIFEST}")
EXPECTED_NODE_NAME=$(jq -er '.expected_node_name' "${LAUNCH_MANIFEST}")
TARGET_GPU_UUID=$(jq -er '.target_gpu_uuid' "${LAUNCH_MANIFEST}")
PROVENANCE=${ROOT}/provenance
ATTEMPT_ID=${SLURM_JOB_ID}
ATTEMPT_PROVENANCE=${PROVENANCE}/attempts/${ATTEMPT_ID}
RUNTIME=${ROOT}/runtime
CUDA_BINARY=${RUNTIME}/cuda/libcuda_backproject.so
RELION_BIND_DIR=${RUNTIME}/relion_bind
ACCEPTANCE=$(jq -er '.files.acceptance.path' "${LAUNCH_MANIFEST}")
ANALYZER=$(jq -er '.files.analyzer.path' "${LAUNCH_MANIFEST}")
HARNESS=$(jq -er '.files.harness.path' "${LAUNCH_MANIFEST}")
RESOLVER=$(jq -er '.files.resolver.path' "${LAUNCH_MANIFEST}")
SCORECARD=$(jq -er '.files.scorecard.path' "${LAUNCH_MANIFEST}")
monitor_pid=
RESUME=${VDAM_TRUE200_RESUME:-0}
case "${RESUME}" in
  0|1) ;;
  *) echo "VDAM_TRUE200_RESUME must be 0 or 1" >&2; exit 2 ;;
esac
SENTINEL=${VDAM_TRUE200_SENTINEL:-0}
case "${SENTINEL}" in
  0|1) ;;
  *) echo "VDAM_TRUE200_SENTINEL must be 0 or 1" >&2; exit 2 ;;
esac
if [[ "${SENTINEL}" -eq 1 && "${RESUME}" -eq 1 ]]; then
  echo "the two-arm sentinel must use a fresh result root" >&2
  exit 2
fi

mapfile -t SOURCE_FILES < <(
  jq -er '.source_contract.files[]' "${ACCEPTANCE}" | LC_ALL=C sort
)

on_error() {
  local status=$?
  if [[ -n "${monitor_pid}" ]]; then
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
  fi
  if [[ -d "${ATTEMPT_PROVENANCE}" ]]; then
    set -o noclobber
    printf 'HARNESS_FAILURE status=%s line=%s command=%s\n' \
      "${status}" "${BASH_LINENO[0]}" "${BASH_COMMAND}" > "${ATTEMPT_PROVENANCE}/harness_failure.txt" || true
    set +o noclobber
  fi
  exit "${status}"
}

on_signal() {
  local signal=$1
  if [[ -n "${monitor_pid}" ]]; then
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
  fi
  if [[ -d "${ATTEMPT_PROVENANCE}" ]]; then
    set -o noclobber
    printf 'HARNESS_SIGNAL signal=%s job_id=%s\n' \
      "${signal}" "${SLURM_JOB_ID}" > "${ATTEMPT_PROVENANCE}/harness_signal.txt" || true
    set +o noclobber
  fi
  exit 143
}

write_marker_once() {
  local path=$1
  local message=$2
  test ! -e "${path}"
  set -o noclobber
  printf '%s\n' "${message}" > "${path}"
  set +o noclobber
}

if [[ "${RESUME}" -eq 0 ]]; then
  test ! -e "${ROOT}"
  mkdir -p "${ATTEMPT_PROVENANCE}"
  touch "${ROOT}/SAFE_TO_DELETE"
else
  test -d "${ROOT}"
  test -f "${ROOT}/SAFE_TO_DELETE"
  test ! -e "${ROOT}/COMPLETED"
  test -s "${ROOT}/provenance/run.json"
  test ! -e "${ATTEMPT_PROVENANCE}"
  mkdir -p "${ATTEMPT_PROVENANCE}"
fi
trap on_error ERR
trap 'on_signal TERM' TERM
trap 'on_signal INT' INT
test -x "${PIXI_PY}"
"${PIXI_PY}" "${RESOLVER}" validate \
  --manifest "${LAUNCH_MANIFEST}" \
  --expected-sha256 "${EXPECTED_LAUNCH_MANIFEST_SHA256}" \
  > "${ATTEMPT_PROVENANCE}/launch_manifest_validation.stdout" \
  2> "${ATTEMPT_PROVENANCE}/launch_manifest_validation.stderr"
test -s "${CUSPARSE_LIBRARY}"
test -d "${FIXTURE}"
test -d "${NATIVE_ROOT}"
test -s "${QUALIFIED_CUDA_OVERRIDE}"
test -s "${RELION_BIND_BINARY_OVERRIDE}"
test -s "${GPU_SELECTION_HELPER}"
test "$(sha256sum "${CUSPARSE_LIBRARY}" | awk '{print $1}')" = "${EXPECTED_CUSPARSE_SHA256}"
test "${SLURMD_NODENAME:-$(hostname -s)}" = "${EXPECTED_NODE_NAME}"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_OVERLAY_HEAD}"
git -C "${REPO_ROOT}" merge-base --is-ancestor \
  "${EXPECTED_PRODUCTION_CANDIDATE_HEAD}" "${EXPECTED_OVERLAY_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1)"
test "$(sha256sum "${QUALIFIED_CUDA_OVERRIDE}" | awk '{print $1}')" = \
  "$(jq -r '.qualified_candidate.cuda_sha256' "${ACCEPTANCE}")"
test "$(sha256sum "${RELION_BIND_BINARY_OVERRIDE}" | awk '{print $1}')" = \
  "$(jq -r '.qualified_candidate.relion_bind_sha256' "${ACCEPTANCE}")"
test "$(sha256sum "${PIXI_PY}" | awk '{print $1}')" = \
  "$(jq -r '.qualified_candidate.interpreter_sha256' "${ACCEPTANCE}")"
test "$(sha256sum "${GPU_SELECTION_HELPER}" | awk '{print $1}')" = \
  "${EXPECTED_GPU_SELECTION_HELPER_SHA256}"
test "$(sha256sum "${ACCEPTANCE}" | awk '{print $1}')" = \
  "${EXPECTED_ACCEPTANCE_CONFIG_SHA256}"
test "$(sha256sum "${ANALYZER}" | awk '{print $1}')" = \
  "${EXPECTED_ANALYZER_SOURCE_SHA256}"
test "$(sha256sum "${HARNESS}" | awk '{print $1}')" = \
  "${EXPECTED_HARNESS_SOURCE_SHA256}"
test "$(sha256sum "${SCORECARD}" | awk '{print $1}')" = \
  "$(jq -r '.case.scorecard_sha256' "${ACCEPTANCE}")"

source "${GPU_SELECTION_HELPER}"
initial_allocation_spec=${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}}
vdam_assert_target_gpu_allocated "${TARGET_GPU_UUID}" "${initial_allocation_spec}"
vdam_select_target_gpu "${TARGET_GPU_UUID}" 0
selected_gpu_uuid=${VDAM_SELECTED_GPU_UUID}
test "${selected_gpu_uuid}" = "$(jq -r '.native_reference.physical_gpu_uuid' "${ACCEPTANCE}")"
test "${VDAM_ALLOCATED_GPU_UUIDS_CSV}" = "${selected_gpu_uuid}"
printf '%s\n' "${initial_allocation_spec}" > "${ATTEMPT_PROVENANCE}/initial_allocation_spec.txt"
printf '%s\n' "${VDAM_ALLOCATED_GPU_UUIDS_CSV}" > "${ATTEMPT_PROVENANCE}/allocated_gpu_uuids.csv"
printf '%s\n' "${VDAM_VISIBLE_GPU_UUIDS_CSV}" > "${ATTEMPT_PROVENANCE}/visible_gpu_uuids.csv"
gpu_name=$(nvidia-smi -i "${selected_gpu_uuid}" --query-gpu=name --format=csv,noheader | sed -n '1p' | sed 's/[[:space:]]*$//')
[[ "${gpu_name}" == *H100* ]]
if [[ "${RESUME}" -eq 1 && ( -e "${ROOT}/SCIENCE_PASSED" || -e "${ROOT}/SCIENCE_FAILED" ) ]]; then
  terminal_quarantine=${ROOT}/failed_attempts/terminal-before-${SLURM_JOB_ID}
  test ! -e "${terminal_quarantine}"
  mkdir -p "${terminal_quarantine}"
  for marker in SCIENCE_PASSED SCIENCE_FAILED; do
    if [[ -e "${ROOT}/${marker}" ]]; then
      mv "${ROOT}/${marker}" "${terminal_quarantine}/${marker}"
    fi
  done
fi

mkdir -p "${PROVENANCE}" "${RUNTIME}/cuda" "${RELION_BIND_DIR}" "${ROOT}/runs"
if [[ "${RESUME}" -eq 0 ]]; then
  cp --reflink=auto "${QUALIFIED_CUDA_OVERRIDE}" "${CUDA_BINARY}"
  cp --reflink=auto "${RELION_BIND_BINARY_OVERRIDE}" "${RELION_BIND_DIR}/"
fi
RELION_BIND_BINARY=${RELION_BIND_DIR}/$(basename "${RELION_BIND_BINARY_OVERRIDE}")
test -s "${CUDA_BINARY}"
test -s "${RELION_BIND_BINARY}"
test "$(sha256sum "${CUDA_BINARY}" | awk '{print $1}')" = \
  "$(jq -r '.qualified_candidate.cuda_sha256' "${ACCEPTANCE}")"
test "$(sha256sum "${RELION_BIND_BINARY}" | awk '{print $1}')" = \
  "$(jq -r '.qualified_candidate.relion_bind_sha256' "${ACCEPTANCE}")"

test "$(sha256sum "${FIXTURE}/fixture_materialization.json" | awk '{print $1}')" = \
  "$(jq -r '.case.fixture_manifest_sha256' "${ACCEPTANCE}")"
test "$(sha256sum "${FIXTURE}/particles.star" | awk '{print $1}')" = \
  "$(jq -r '.case.particle_star_sha256' "${ACCEPTANCE}")"
test "$(sha256sum "${FIXTURE}/particles.128.mrcs" | awk '{print $1}')" = \
  "$(jq -r '.case.particle_stack_sha256' "${ACCEPTANCE}")"
test "$(sha256sum "${FIXTURE}/reference_gt_relion.mrc" | awk '{print $1}')" = \
  "$(jq -r '.case.gt_relion_mrc_sha256' "${ACCEPTANCE}")"

if [[ "${RESUME}" -eq 0 ]]; then
  cp --reflink=auto "${LAUNCH_MANIFEST}" "${PROVENANCE}/launch_manifest.json"
  (
    cd "${REPO_ROOT}"
    sha256sum "${SOURCE_FILES[@]}"
  ) > "${PROVENANCE}/source_manifest.sha256"
  jq -S '.science_contract' "${ACCEPTANCE}" > "${PROVENANCE}/science_contract.json"
  cp --reflink=auto "${PROVENANCE}/source_manifest.sha256" "${ATTEMPT_PROVENANCE}/source_manifest.sha256"
  cp --reflink=auto "${PROVENANCE}/science_contract.json" "${ATTEMPT_PROVENANCE}/science_contract.json"
else
  cmp "${PROVENANCE}/launch_manifest.json" "${LAUNCH_MANIFEST}"
  (
    cd "${REPO_ROOT}"
    sha256sum "${SOURCE_FILES[@]}"
  ) > "${ATTEMPT_PROVENANCE}/source_manifest.sha256"
  cmp "${PROVENANCE}/source_manifest.sha256" "${ATTEMPT_PROVENANCE}/source_manifest.sha256"
  jq -S '.science_contract' "${ACCEPTANCE}" > "${ATTEMPT_PROVENANCE}/science_contract.json"
  cmp "${PROVENANCE}/science_contract.json" "${ATTEMPT_PROVENANCE}/science_contract.json"
fi
source_manifest_sha256=$(sha256sum "${PROVENANCE}/source_manifest.sha256" | awk '{print $1}')
test "${source_manifest_sha256}" = "${EXPECTED_SOURCE_MANIFEST_SHA256}"
science_contract_sha256=$(sha256sum "${PROVENANCE}/science_contract.json" | awk '{print $1}')

git -C "${REPO_ROOT}" rev-parse HEAD > "${ATTEMPT_PROVENANCE}/repo_head.txt"
git -C "${REPO_ROOT}" symbolic-ref --short HEAD > "${ATTEMPT_PROVENANCE}/repo_branch.txt"
git -C "${REPO_ROOT}" status --porcelain=v1 > "${ATTEMPT_PROVENANCE}/repo_status.txt"
printf '%s\n' "${SLURM_JOB_ID}" > "${ATTEMPT_PROVENANCE}/slurm_job_id.txt"
printf '%s\n' "${SLURMD_NODENAME:-$(hostname -s)}" > "${ATTEMPT_PROVENANCE}/node.txt"
printf '%s\n' "${selected_gpu_uuid}" > "${ATTEMPT_PROVENANCE}/selected_gpu_uuid.txt"
env | LC_ALL=C sort > "${ATTEMPT_PROVENANCE}/submission_environment.txt"
nvidia-smi -i "${selected_gpu_uuid}" -q > "${ATTEMPT_PROVENANCE}/nvidia_smi.txt"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV JAX_PLATFORM_NAME \
  CUDA_TOOLKIT RELION_CUDA_TOOLKIT RELION_RUNTIME MPI_ROOT LD_PRELOAD LD_LIBRARY_PATH
export PATH
PATH=$(jq -er '.runtime_contract.fixed_science_environment.PATH' "${ACCEPTANCE}")
export LD_LIBRARY_PATH
LD_LIBRARY_PATH=$(jq -er '.runtime_contract.fixed_science_environment.LD_LIBRARY_PATH' "${ACCEPTANCE}")
export PYTHONNOUSERSITE
PYTHONNOUSERSITE=$(jq -er '.runtime_contract.fixed_science_environment.PYTHONNOUSERSITE' "${ACCEPTANCE}")
export PYTHONUNBUFFERED
PYTHONUNBUFFERED=$(jq -er '.runtime_contract.fixed_science_environment.PYTHONUNBUFFERED' "${ACCEPTANCE}")
ATTEMPT_RUNTIME=${RUNTIME}/attempts/${SLURM_JOB_ID}
export TMPDIR=${ATTEMPT_RUNTIME}/tmp
export PIXI_HOME=${ATTEMPT_RUNTIME}/pixi_home
export RATTLER_CACHE_DIR=${ATTEMPT_RUNTIME}/rattler_cache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}"
touch "${ATTEMPT_RUNTIME}/SAFE_TO_DELETE"
export CUDA_VISIBLE_DEVICES=${selected_gpu_uuid}
export JAX_PLATFORMS
JAX_PLATFORMS=$(jq -er '.runtime_contract.fixed_science_environment.JAX_PLATFORMS' "${ACCEPTANCE}")
export RECOVAR_EXPECTED_REPO_ROOT=${REPO_ROOT}
export RECOVAR_CUDA_LIB=${CUDA_BINARY}
export RECOVAR_RELION_BIND_BUILD_DIR=${RELION_BIND_DIR}
export RECOVAR_EM_RAW_IMAGE_CACHE
RECOVAR_EM_RAW_IMAGE_CACHE=$(jq -er '.runtime_contract.fixed_science_environment.RECOVAR_EM_RAW_IMAGE_CACHE' "${ACCEPTANCE}")
export RECOVAR_INITIAL_MODEL_PROFILE
RECOVAR_INITIAL_MODEL_PROFILE=$(jq -er '.runtime_contract.fixed_science_environment.RECOVAR_INITIAL_MODEL_PROFILE' "${ACCEPTANCE}")
export XLA_PYTHON_CLIENT_PREALLOCATE
XLA_PYTHON_CLIENT_PREALLOCATE=$(jq -er '.runtime_contract.fixed_science_environment.XLA_PYTHON_CLIENT_PREALLOCATE' "${ACCEPTANCE}")
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS
JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=$(jq -er '.runtime_contract.fixed_science_environment.JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS' "${ACCEPTANCE}")
mapfile -t FIXED_SCIENCE_ENVIRONMENT_ARGS < <(
  jq -er '.runtime_contract.fixed_science_environment | to_entries | sort_by(.key)[] | "\(.key)=\(.value)"' \
    "${ACCEPTANCE}"
)
test "${#FIXED_SCIENCE_ENVIRONMENT_ARGS[@]}" -gt 0

(
  cd "${REPO_ROOT}"
  env "${FIXED_SCIENCE_ENVIRONMENT_ARGS[@]}" PYTHONPATH="${REPO_ROOT}" "${PIXI_PY}" - "${ACCEPTANCE}" \
    "${ATTEMPT_PROVENANCE}/sealed_override_environment.json" <<'PY'
import json
import os
import sys
from pathlib import Path
from scripts.resolve_vdam_coarse_combined_true200_launch import _validate_override_environment

acceptance = json.load(open(sys.argv[1], encoding="utf-8"))
observed = _validate_override_environment(
    dict(os.environ), acceptance["runtime_contract"], include_science=True
)
Path(sys.argv[2]).write_text(json.dumps(observed, indent=2, sort_keys=True) + "\n")
PY
)

(
  cd "${REPO_ROOT}"
  env "${FIXED_SCIENCE_ENVIRONMENT_ARGS[@]}" LD_PRELOAD="${CUSPARSE_LIBRARY}" \
    PYTHONPATH="${REPO_ROOT}" "${PIXI_PY}" - <<'PY'
import pathlib
import jax
import recovar
import recovar.cuda_backproject as cuda_backproject
from recovar.relion_bind import _relion_bind_core
from recovar.utils.parity_provenance import assert_parity_ancestors

repo = pathlib.Path(__import__("os").environ["RECOVAR_EXPECTED_REPO_ROOT"]).resolve()
assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo) + "/")
assert jax.default_backend() == "gpu" and len(jax.devices("gpu")) == 1
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
assert callable(getattr(_relion_bind_core, "vdam_rnd_unif_sequence", None))
assert_parity_ancestors()
print("true-200 runtime gate passed", jax.devices(), recovar.__file__)
PY
) > "${ATTEMPT_PROVENANCE}/runtime_gate.stdout" 2> "${ATTEMPT_PROVENANCE}/runtime_gate.stderr"
vdam_verify_selected_gpu "${selected_gpu_uuid}"
"${PIXI_PY}" - "${ATTEMPT_PROVENANCE}/runtime_libraries.json" \
  "${PIXI_PY}" "${CUDA_BINARY}" "${RELION_BIND_BINARY}" "${CUSPARSE_LIBRARY}" <<'PY'
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

output = Path(sys.argv[1])
roots = [Path(value).resolve() for value in sys.argv[2:]]
paths = set(roots)
ldd_outputs = {}
for root in roots:
    text = subprocess.check_output(["ldd", str(root)], text=True, stderr=subprocess.STDOUT)
    if "not found" in text:
        raise SystemExit(f"unresolved runtime library for {root}: {text}")
    ldd_outputs[str(root)] = text
    for line in text.splitlines():
        match = re.search(r"(?:=>\s+)?(/[^\s(]+)", line)
        if match:
            paths.add(Path(match.group(1)).resolve())
rows = []
def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

for path in sorted(paths, key=str):
    if not path.is_file():
        raise SystemExit(f"runtime library is not a file: {path}")
    rows.append({"path": str(path), "sha256": sha256(path)})
payload = {
    "schema": "recovar.vdam_combined_true200_runtime_libraries.v1",
    "transitive_runtime_library_closure_claimed": False,
    "direct_and_ldd_visible_artifacts": rows,
    "ldd_output": ldd_outputs,
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
sha256sum "${ATTEMPT_PROVENANCE}/runtime_libraries.json" > \
  "${ATTEMPT_PROVENANCE}/runtime_libraries.sha256"
if [[ "${RESUME}" -eq 0 ]]; then
  cp --reflink=auto "${ATTEMPT_PROVENANCE}/runtime_libraries.json" \
    "${PROVENANCE}/runtime_libraries.json"
else
  cmp "${PROVENANCE}/runtime_libraries.json" \
    "${ATTEMPT_PROVENANCE}/runtime_libraries.json"
fi

RUN_LABELS=(
  control_serial_1 combined_candidate_1 combined_candidate_2 control_serial_2
  combined_candidate_3 control_serial_3 control_serial_4 combined_candidate_4
  control_serial_5 combined_candidate_5 combined_candidate_6 control_serial_6
)
RUN_WORKERS=(0 0 0 0 0 0 0 0 0 0 0 0)
RUN_ATOMIC=(0 0 0 0 0 0 0 0 0 0 0 0)
RUN_HYBRID=(0 1 1 0 1 0 0 1 0 1 1 0)
test "$(jq -c '.panel.execution_order' "${ACCEPTANCE}")" = \
  "$(printf '%s\n' "${RUN_LABELS[@]}" | jq -R . | jq -sc .)"
test "$(jq -c '.panel.coarse_multistream_workers' "${ACCEPTANCE}")" = \
  "$(printf '%s\n' "${RUN_WORKERS[@]}" | jq -R 'tonumber' | jq -sc .)"
test "$(jq -c '.panel.coarse_native_atomic_reduction' "${ACCEPTANCE}")" = \
  "$(printf '%s\n' "${RUN_ATOMIC[@]}" | jq -R 'tonumber' | jq -sc .)"
test "$(jq -c '.panel.coarse_gemm_hybrid' "${ACCEPTANCE}")" = \
  "$(printf '%s\n' "${RUN_HYBRID[@]}" | jq -R 'tonumber' | jq -sc .)"
RUN_LIMIT=${#RUN_LABELS[@]}
if [[ "${SENTINEL}" -eq 1 ]]; then
  RUN_LIMIT=$(jq -er '.panel.sentinel_prefix_arm_count' "${ACCEPTANCE}")
  test "${RUN_LIMIT}" -eq 2
  test "${RUN_HYBRID[0]}" -eq 0
  test "${RUN_HYBRID[1]}" -eq 1
fi

validate_completed_arm() {
  local label=$1
  local workers=$2
  local native_atomic=$3
  local hybrid=$4
  local require_marker=$5
  (
    cd "${REPO_ROOT}"
    env "${FIXED_SCIENCE_ENVIRONMENT_ARGS[@]}" PYTHONPATH="${REPO_ROOT}" \
      "${PIXI_PY}" - "${ROOT}" "${ACCEPTANCE}" "${label}" "${workers}" \
      "${native_atomic}" "${hybrid}" "${require_marker}" <<'PY'
import json
import sys
from pathlib import Path

from scripts.analyze_vdam_coarse_combined_true200 import (
    _arm_runtime,
    _load_json,
    _validate_arm_artifacts,
    _validate_arm_coarse_hybrid_execution,
    _validate_native_options,
)

root, acceptance_path = Path(sys.argv[1]), Path(sys.argv[2])
label = sys.argv[3]
workers, native_atomic, hybrid, require_marker = map(int, sys.argv[4:8])
acceptance = _load_json(acceptance_path, "acceptance contract")
run = _load_json(root / "provenance/run.json", "run provenance")
expected = {
    **run,
    "production_candidate_head": acceptance["qualified_candidate"]["production_head"],
}
trajectory = acceptance["trajectory"]
iterations = tuple(
    range(int(trajectory["first_iteration"]), int(trajectory["last_iteration"]) + 1)
)
artifact = _validate_arm_artifacts(
    root,
    label,
    workers,
    native_atomic,
    hybrid,
    iterations,
    trajectory["required_artifact_suffixes"],
    expected=expected,
    runtime_contract=acceptance["runtime_contract"],
    require_completion_marker=bool(require_marker),
)
_validate_native_options(
    root / "runs" / label / "output/run_native_options.json",
    acceptance["science_contract"]["definition"],
)
coarse_execution = _validate_arm_coarse_hybrid_execution(
    root,
    label,
    workers,
    native_atomic,
    hybrid,
    iterations[1:],
    cache_contract=acceptance["science_contract"][
        "hybrid_projection_cache_contract"
    ],
)
runtime = _arm_runtime(
    root,
    label,
    iterations,
    expected_gpu_uuid=acceptance["native_reference"]["physical_gpu_uuid"],
)
print(
    json.dumps(
        {
            "artifact": artifact,
            "coarse_hybrid_execution": coarse_execution,
            "runtime": runtime,
        },
        indent=2,
        sort_keys=True,
    )
)
PY
  ) > "${ATTEMPT_PROVENANCE}/${label}_validation.json"
}

if [[ "${RESUME}" -eq 0 ]]; then
  "${PIXI_PY}" - "${PROVENANCE}/run.json" "${ACCEPTANCE}" "${PROVENANCE}/source_manifest.sha256" \
  "${REPO_ROOT}" "${SCORECARD}" "${FIXTURE}" "${NATIVE_ROOT}" "${CUDA_BINARY}" \
  "${RELION_BIND_BINARY}" "${PIXI_PY}" "${CUSPARSE_LIBRARY}" "${selected_gpu_uuid}" \
  "${SLURMD_NODENAME:-$(hostname -s)}" "${gpu_name}" "${SLURM_JOB_ID}" \
  "${EXPECTED_PRODUCTION_CANDIDATE_HEAD}" "${science_contract_sha256}" \
  "${PROVENANCE}/launch_manifest.json" "${EXPECTED_LAUNCH_MANIFEST_SHA256}" \
  "${PROVENANCE}/runtime_libraries.json" "${VDAM_ALLOCATED_GPU_UUIDS_CSV}" \
  "${VDAM_VISIBLE_GPU_UUIDS_CSV}" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path

(output, acceptance_path, manifest_path, repo, scorecard, fixture, native, cuda, bind,
 interpreter, cusparse, gpu, node, gpu_name, job_id, production, science_sha, launch_manifest,
 launch_manifest_sha, runtime_libraries, allocated_gpu_uuids, visible_gpu_uuids) = sys.argv[1:]
acceptance = json.load(open(acceptance_path, encoding="utf-8"))
manifest = Path(manifest_path)
source_files = [line.split("  ", 1)[1] for line in manifest.read_text().splitlines()]
def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
payload = {
    "schema": "recovar.vdam_coarse_combined_true200_run.v3",
    "classification": "sealed_default_off_true200_candidate_gate",
    "job_id": job_id,
    "repo_root": str(Path(repo).resolve()),
    "git_head": subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip(),
    "production_candidate_head": production,
    "node": node,
    "gpu_uuid": gpu,
    "gpu_name": gpu_name,
    "allocated_gpu_uuids_csv": allocated_gpu_uuids,
    "visible_gpu_uuids_csv": visible_gpu_uuids,
    "cuda_path": str(Path(cuda).resolve()),
    "cuda_sha256": sha(cuda),
    "relion_bind_path": str(Path(bind).resolve()),
    "relion_bind_sha256": sha(bind),
    "interpreter_path": str(Path(interpreter).resolve()),
    "interpreter_sha256": sha(interpreter),
    "cusparse_path": str(Path(cusparse).resolve()),
    "cusparse_sha256": sha(cusparse),
    "scorecard_path": str(Path(scorecard).resolve()),
    "scorecard_sha256": sha(scorecard),
    "acceptance_path": str(Path(acceptance_path).resolve()),
    "acceptance_config_sha256": sha(acceptance_path),
    "analyzer_source_sha256": sha(Path(repo) / "scripts/analyze_vdam_coarse_combined_true200.py"),
    "harness_source_sha256": sha(Path(repo) / "scripts/run_vdam_coarse_combined_true200.sbatch"),
    "source_files": source_files,
    "source_manifest_sha256": sha(manifest),
    "science_contract_sha256": science_sha,
    "launch_manifest_path": str(Path(launch_manifest).resolve()),
    "launch_manifest_sha256": launch_manifest_sha,
    "runtime_libraries_path": str(Path(runtime_libraries).resolve()),
    "runtime_libraries_sha256": sha(runtime_libraries),
    "definition": acceptance["science_contract"]["definition"],
    "fixture_dir": str(Path(fixture).resolve()),
    "native_reference_root": str(Path(native).resolve()),
    "runtime_contract": acceptance["runtime_contract"],
    "execution_order": acceptance["panel"]["execution_order"],
    "coarse_multistream_workers": acceptance["panel"]["coarse_multistream_workers"],
    "coarse_native_atomic_reduction": acceptance["panel"]["coarse_native_atomic_reduction"],
    "coarse_gemm_hybrid": acceptance["panel"]["coarse_gemm_hybrid"],
    "fresh_process_and_jax_cache_per_arm": True,
    "only_configuration_delta": acceptance["science_contract"]["only_configuration_delta"],
    "initial_model_profile": True,
}
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
else
  (
    cd "${REPO_ROOT}"
    env "${FIXED_SCIENCE_ENVIRONMENT_ARGS[@]}" PYTHONPATH="${REPO_ROOT}" \
      "${PIXI_PY}" - "${ROOT}" "${ACCEPTANCE}" "${SCORECARD}" <<'PY'
import sys
from pathlib import Path
from scripts.analyze_vdam_coarse_combined_true200 import _load_json, _validate_run_provenance

root, acceptance_path, scorecard_path = map(Path, sys.argv[1:])
acceptance = _load_json(acceptance_path, "acceptance contract")
_validate_run_provenance(root, acceptance_path, scorecard_path, acceptance)
print("resume run provenance verified")
PY
  ) > "${ATTEMPT_PROVENANCE}/run_validation.stdout"
fi

completed_prefix=1

for index in "${!RUN_LABELS[@]}"; do
  if [[ "${index}" -ge "${RUN_LIMIT}" ]]; then
    break
  fi
  label=${RUN_LABELS[index]}
  workers=${RUN_WORKERS[index]}
  native_atomic=${RUN_ATOMIC[index]}
  hybrid=${RUN_HYBRID[index]}
  run_root=${ROOT}/runs/${label}
  output=${run_root}/output
  cache=${run_root}/jax_cache
  tmp=${run_root}/tmp
  if [[ -f "${run_root}/SCIENCE_COMPLETED" ]]; then
    test "${RESUME}" -eq 1
    test "${completed_prefix}" -eq 1
    validate_completed_arm "${label}" "${workers}" "${native_atomic}" "${hybrid}" 1
    continue
  fi
  completed_prefix=0
  if [[ -e "${run_root}" ]]; then
    test "${RESUME}" -eq 1
    mkdir -p "${ROOT}/failed_attempts"
    quarantine=${ROOT}/failed_attempts/${label}-before-${SLURM_JOB_ID}
    test ! -e "${quarantine}"
    mv "${run_root}" "${quarantine}"
  fi
  mkdir -p "${output}" "${cache}" "${tmp}"
  touch "${cache}/SAFE_TO_DELETE" "${tmp}/SAFE_TO_DELETE"
  "${PIXI_PY}" - "${run_root}/command.json" "${FIXTURE}/particles.star" \
    "${output}/run" "${FIXTURE}" "${ACCEPTANCE}" <<'PY'
import json, sys
from pathlib import Path
from scripts.run_vdam_relion_parity_case import build_recovar_command

target, input_star, output_prefix, fixture, acceptance_path = sys.argv[1:]
contract = json.load(open(acceptance_path, encoding="utf-8"))["science_contract"]
argv = build_recovar_command(
    input_star=Path(input_star), output_prefix=Path(output_prefix), fixture_dir=Path(fixture),
    definition=contract["definition"], image_batch_size=int(contract["image_batch_size"]),
)
Path(target).write_text(json.dumps({"argv": argv}, indent=2, sort_keys=True) + "\n")
PY
  mapfile -d '' -t command < <("${PIXI_PY}" - "${run_root}/command.json" <<'PY'
import json, sys
for value in json.load(open(sys.argv[1], encoding="utf-8"))["argv"]:
    sys.stdout.buffer.write(str(value).encode() + b"\0")
PY
  )
  science_environment_args=(
    "${FIXED_SCIENCE_ENVIRONMENT_ARGS[@]}"
    "LD_PRELOAD=${CUSPARSE_LIBRARY}"
    "PYTHONPATH=${REPO_ROOT}"
    "TMPDIR=${tmp}"
    "JAX_COMPILATION_CACHE_DIR=${cache}"
    "RECOVAR_COARSE_GAUSSIAN_GEMM_HYBRID=${hybrid}"
    "RECOVAR_COARSE_GAUSSIAN_GEMM_MACRO=${hybrid}"
    "RECOVAR_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE=${hybrid}"
    "RECOVAR_K1_COARSE_MULTISTREAM_WORKERS=${workers}"
    "RECOVAR_K1_COARSE_NATIVE_ATOMIC_REDUCTION=${native_atomic}"
  )
  (
    cd "${REPO_ROOT}"
    env "${science_environment_args[@]}" "${PIXI_PY}" - \
      "${run_root}/science_environment.json" "${ACCEPTANCE}" "${label}" "${SLURM_JOB_ID}" <<'PY'
import json
import os
import sys
from pathlib import Path

from scripts.resolve_vdam_coarse_combined_true200_launch import _science_environment_snapshot

target, acceptance_path, label, job_id = sys.argv[1:]
acceptance = json.load(open(acceptance_path, encoding="utf-8"))
payload = {
    "schema": "recovar.vdam_coarse_combined_true200_science_environment.v3",
    "label": label,
    "job_id": job_id,
    "environment": _science_environment_snapshot(
        dict(os.environ), acceptance["runtime_contract"]
    ),
}
Path(target).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
  )
  start_epoch=$(date +%s.%N)
  nvidia-smi -i "${selected_gpu_uuid}" \
    --query-gpu=timestamp,index,name,uuid,memory.used,memory.total,utilization.gpu \
    --format=csv -l 1 > "${run_root}/gpu_monitor.csv" &
  monitor_pid=$!
  trap - ERR
  set +e
  (
    cd "${REPO_ROOT}"
    /usr/bin/time -v -o "${run_root}/process.time" \
      env "${science_environment_args[@]}" \
      "${command[@]}"
  ) > "${run_root}/runner.stdout" 2> "${run_root}/runner.stderr"
  status=$?
  set -e
  trap on_error ERR
  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true
  monitor_pid=
  end_epoch=$(date +%s.%N)
  if [[ "${status}" -ne 0 ]]; then
    printf 'SCIENCE_EXECUTION_FAILURE status=%s job_id=%s\n' \
      "${status}" "${SLURM_JOB_ID}" > "${run_root}/science_execution_failure.${SLURM_JOB_ID}.txt"
    exit "${status}"
  fi
  test "$(find "${output}" -maxdepth 1 -type f -regextype posix-extended \
    -regex '.*/run_it[0-9]{3}_(class001\.mrc|data\.star|model\.star|recovar_meta\.json)' | wc -l)" -eq 804
  for iteration in $(seq 0 200); do
    tag=$(printf '%03d' "${iteration}")
    test -s "${output}/run_it${tag}_class001.mrc"
    test -s "${output}/run_it${tag}_data.star"
    test -s "${output}/run_it${tag}_model.star"
    test -s "${output}/run_it${tag}_recovar_meta.json"
  done
  (
    cd "${output}"
    find . -maxdepth 1 -type f -regextype posix-extended \
      -regex './run_it[0-9]{3}_(class001\.mrc|data\.star|model\.star|recovar_meta\.json)' \
      -printf '%f\n' | LC_ALL=C sort | xargs sha256sum
  ) > "${run_root}/artifact_manifest.sha256"
  "${PIXI_PY}" - "${run_root}/timing.json" "${start_epoch}" "${end_epoch}" <<'PY'
import json, sys
start, end = float(sys.argv[2]), float(sys.argv[3])
open(sys.argv[1], "w", encoding="utf-8").write(json.dumps({
    "start_epoch": start, "end_epoch": end, "external_wall_s": end - start, "exit_status": 0,
}, indent=2, sort_keys=True) + "\n")
PY
  "${PIXI_PY}" - "${run_root}/arm.json" "${PROVENANCE}/run.json" "${label}" "${workers}" \
    "${native_atomic}" "${hybrid}" "${output}" "${cache}" "${run_root}/science_environment.json" \
    "${SLURM_JOB_ID}" <<'PY'
import hashlib, json, sys
from pathlib import Path
(target, run_path, label, workers, native_atomic, hybrid, output, cache,
 science_environment, job_id) = sys.argv[1:]
run = json.load(open(run_path, encoding="utf-8"))
def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
keys = ("git_head", "production_candidate_head", "gpu_uuid", "node",
        "allocated_gpu_uuids_csv", "visible_gpu_uuids_csv", "cuda_sha256",
        "relion_bind_sha256", "interpreter_sha256", "cusparse_sha256", "scorecard_sha256",
        "acceptance_config_sha256", "analyzer_source_sha256", "science_contract_sha256")
payload = {key: run[key] for key in keys}
payload.update(schema="recovar.vdam_coarse_combined_true200_arm.v3", label=label,
               job_id=job_id,
               multistream_workers=int(workers), native_atomic_reduction=int(native_atomic),
               coarse_gemm_hybrid=int(hybrid),
               output_dir=str(Path(output).resolve()),
               jax_cache_dir=str(Path(cache).resolve()), exit_status=0, fresh_process=True,
               fresh_jax_cache=True, initial_model_profile=True,
               science_environment_sha256=sha256(science_environment))
Path(target).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
  (
    cd "${run_root}"
    printf '%s\n' \
      command.json timing.json gpu_monitor.csv process.time runner.stdout runner.stderr \
      arm.json science_environment.json output/run_native_options.json | \
      LC_ALL=C sort | xargs sha256sum
  ) > "${run_root}/runtime_artifact_manifest.sha256"
  validate_completed_arm "${label}" "${workers}" "${native_atomic}" "${hybrid}" 0
  write_marker_once "${run_root}/SCIENCE_COMPLETED" "job_id=${SLURM_JOB_ID}"
  test "$(sha256sum "${CUDA_BINARY}" | awk '{print $1}')" = \
    "$(jq -r '.qualified_candidate.cuda_sha256' "${ACCEPTANCE}")"
  test "$(sha256sum "${CUSPARSE_LIBRARY}" | awk '{print $1}')" = "${EXPECTED_CUSPARSE_SHA256}"
  vdam_verify_selected_gpu "${selected_gpu_uuid}"
done

if [[ "${SENTINEL}" -eq 1 ]]; then
  test "$(find "${ROOT}/runs" -mindepth 2 -maxdepth 2 -name SCIENCE_COMPLETED -type f | wc -l)" \
    -eq "${RUN_LIMIT}"
  (
    cd "${REPO_ROOT}"
    sha256sum "${SOURCE_FILES[@]}"
  ) > "${ATTEMPT_PROVENANCE}/source_manifest.final.sha256"
  cmp "${PROVENANCE}/source_manifest.sha256" \
    "${ATTEMPT_PROVENANCE}/source_manifest.final.sha256"
  test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_OVERLAY_HEAD}"
  test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1)"
  test "$(sha256sum "${CUDA_BINARY}" | awk '{print $1}')" = \
    "$(jq -r '.qualified_candidate.cuda_sha256' "${ACCEPTANCE}")"
  test "$(sha256sum "${CUSPARSE_LIBRARY}" | awk '{print $1}')" = \
    "${EXPECTED_CUSPARSE_SHA256}"
  vdam_verify_selected_gpu "${selected_gpu_uuid}"
  write_marker_once "${ROOT}/SENTINEL_PASSED" \
    "job_id=${SLURM_JOB_ID} arms=${RUN_LIMIT} classification=non_scoring_setup_sentinel"
  write_marker_once "${ROOT}/COMPLETED" "job_id=${SLURM_JOB_ID} status=sentinel"
  exit 0
fi

if [[ ! -e "${ROOT}/RUNS_COMPLETED" ]]; then
  write_marker_once "${ROOT}/RUNS_COMPLETED" "job_id=${SLURM_JOB_ID}"
else
  test "${RESUME}" -eq 1
fi
if [[ "${RESUME}" -eq 1 && -d "${ROOT}/analysis" ]] && \
   find "${ROOT}/analysis" -mindepth 1 -print -quit | read -r; then
  mkdir -p "${ROOT}/failed_attempts"
  analysis_quarantine=${ROOT}/failed_attempts/analysis-before-${SLURM_JOB_ID}
  test ! -e "${analysis_quarantine}"
  mv "${ROOT}/analysis" "${analysis_quarantine}"
fi
mkdir -p "${ROOT}/analysis"
trap - ERR
set +e
(
  cd "${REPO_ROOT}"
  env "${FIXED_SCIENCE_ENVIRONMENT_ARGS[@]}" PYTHONPATH="${REPO_ROOT}" \
    "${PIXI_PY}" "${ANALYZER}" \
    --root "${ROOT}" \
    --acceptance "${ACCEPTANCE}" \
    --scorecard "${SCORECARD}" \
    --fixture-dir "${FIXTURE}" \
    --native-reference-root "${NATIVE_ROOT}" \
    --output "${ROOT}/analysis/true200_analysis.json" \
    --markdown-output "${ROOT}/analysis/true200_analysis.md" \
    --shells-output "${ROOT}/analysis/true200_final_fsc_shells.npz"
)
analysis_status=$?
set -e
trap on_error ERR
if [[ "${analysis_status}" -eq 2 ]]; then
  write_marker_once "${ATTEMPT_PROVENANCE}/ANALYSIS_SETUP_FAILED" "job_id=${SLURM_JOB_ID}"
  exit 2
fi
test "${analysis_status}" -eq 0 -o "${analysis_status}" -eq 1
test -s "${ROOT}/analysis/true200_analysis.json"
test -s "${ROOT}/analysis/true200_analysis.md"
test -s "${ROOT}/analysis/true200_final_fsc_shells.npz"
sha256sum "${ROOT}/analysis/true200_analysis.json" \
  "${ROOT}/analysis/true200_analysis.md" \
  "${ROOT}/analysis/true200_final_fsc_shells.npz" > "${ATTEMPT_PROVENANCE}/sealed_analysis.sha256"
(
  cd "${REPO_ROOT}"
  sha256sum "${SOURCE_FILES[@]}"
) > "${ATTEMPT_PROVENANCE}/source_manifest.final.sha256"
cmp "${PROVENANCE}/source_manifest.sha256" "${ATTEMPT_PROVENANCE}/source_manifest.final.sha256"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_OVERLAY_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1)"
test "$(sha256sum "${CUSPARSE_LIBRARY}" | awk '{print $1}')" = "${EXPECTED_CUSPARSE_SHA256}"
vdam_verify_selected_gpu "${selected_gpu_uuid}"
if [[ "${analysis_status}" -eq 0 ]]; then
  write_marker_once "${ROOT}/SCIENCE_PASSED" "job_id=${SLURM_JOB_ID}"
else
  write_marker_once "${ROOT}/SCIENCE_FAILED" "job_id=${SLURM_JOB_ID}"
fi
write_marker_once "${ROOT}/COMPLETED" "job_id=${SLURM_JOB_ID} status=${analysis_status}"
exit "${analysis_status}"
