#!/usr/bin/env bash
#SBATCH --job-name=vdam-late-prof
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-late-prof-%j.out

set -euo pipefail

: "${VDAM_LATE_PROFILE_ROOT:?set a new disposable output root}"
: "${VDAM_LATE_PROFILE_CHECKPOINT:?set the exact run_it180_optimiser.star}"
: "${VDAM_LATE_PROFILE_DATA_STAR:?set the exact run_it180_data.star}"
: "${VDAM_LATE_PROFILE_DATA_DIR:?set the particle-stack directory}"
: "${EXPECTED_REPO_HEAD:?pin the tracked RECOVAR source head}"
: "${EXPECTED_RELION_SHA256:?pin the native RELION executable}"
: "${RELION_BIND_BINARY_OVERRIDE:?set the qualified RELION binding binary}"
: "${EXPECTED_RELION_BIND_SHA256:?pin the qualified RELION binding binary}"

ROOT=${VDAM_LATE_PROFILE_ROOT}
CHECKPOINT=${VDAM_LATE_PROFILE_CHECKPOINT}
DATA_STAR=${VDAM_LATE_PROFILE_DATA_STAR}
DATA_DIR=${VDAM_LATE_PROFILE_DATA_DIR}
PARTICLE_STACK=${VDAM_LATE_PROFILE_PARTICLE_STACK:-${DATA_DIR}/particles.128.mrcs}
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
PIXI_PY=${PIXI_PY_OVERRIDE:-${REPO_ROOT}/.pixi/envs/default/bin/python}
RELION_BINARY=${RELION_BINARY:-/scratch/gpfs/GILLES/mg6942/relion/build_patched/bin/relion_refine}
NSYS=${VDAM_NSYS_BIN:-/opt/nvidia/nsight-systems/2025.3.2/target-linux-x64/nsys}
TARGET_GPU_UUID=${TARGET_GPU_UUID:-}
CHECKPOINT_ITERATION=${CHECKPOINT_ITERATION:-180}
PROFILED_ITERATION=$((CHECKPOINT_ITERATION + 1))
NR_ITER_SCHEDULE=${NR_ITER_SCHEDULE:-200}
CUDA_TOOLKIT=${CUDA_TOOLKIT:-/usr/local/cuda-13.1}
RELION_CUDA_TOOLKIT=${RELION_CUDA_TOOLKIT:-/usr/local/cuda-12.6}
CUSPARSE_LIBRARY=${CUSPARSE_LIBRARY:-${RELION_CUDA_TOOLKIT}/lib64/libcusparse.so.12}
RELION_RUNTIME=${RELION_RUNTIME:-/scratch/gpfs/GILLES/mg6942/recovar_dev/relion_k4_100k_dispatchv2_20260717}
MPI_ROOT=${MPI_ROOT:-/usr/local/openmpi/cuda-12.6/4.1.6/gcc}
PROVENANCE=${ROOT}/provenance
RUNTIME=${ROOT}/runtime
NATIVE_PLAIN=${ROOT}/native_unprofiled
NATIVE_PROFILE=${ROOT}/native_profiled
RECOVAR_PROFILE=${ROOT}/recovar_profiled
NSIGHT=${ROOT}/nsight
CUDA_BINARY=${RUNTIME}/cuda/libcuda_backproject.so
RELION_BIND_DIR=${RUNTIME}/relion_bind
EXACT_LOCAL_BUCKET_RADIX=${EXACT_LOCAL_BUCKET_RADIX:-4}
EXACT_LOCAL_PHYSICAL_ORDER_CHUNK_SIZE=${EXACT_LOCAL_PHYSICAL_ORDER_CHUNK_SIZE:-0}
STABLE_FOURIER_WINDOW_SHAPES=${STABLE_FOURIER_WINDOW_SHAPES:-0}
AUDIT_RAW_IMAGE_CACHE=${AUDIT_RAW_IMAGE_CACHE:-0}
EXECUTION_CONTRACT=${VDAM_LATE_PROFILE_CONTRACT:-diagnostic}
IMAGE_SIZE=${VDAM_LATE_PROFILE_IMAGE_SIZE:-128}
HOST_ATTRIBUTION=${VDAM_LATE_PROFILE_HOST_ATTRIBUTION:-0}
REUSE_NATIVE_ROOT=${VDAM_LATE_PROFILE_REUSE_NATIVE_ROOT:-}
EXPECTED_REUSED_NATIVE_NSYS_SHA256=${EXPECTED_REUSED_NATIVE_NSYS_SHA256:-}

on_error() {
  local status=$?
  local message="VDAM late profiler failed: status=${status} line=${BASH_LINENO[0]} command=${BASH_COMMAND}"
  printf '%s\n' "${message}" >&2
  if [[ -d "${PROVENANCE}" ]]; then
    printf '%s\n' "${message}" > "${PROVENANCE}/failure.txt"
  fi
  exit "${status}"
}
trap on_error ERR

test ! -e "${ROOT}"
test -x "${PIXI_PY}"
test -x "${RELION_BINARY}"
test -x "${NSYS}"
test -x "${CUDA_TOOLKIT}/bin/nvcc"
test -s "${CUSPARSE_LIBRARY}"
test -s "${CHECKPOINT}"
test -s "${DATA_STAR}"
test -d "${DATA_DIR}"
test -s "${PARTICLE_STACK}"
test -s "${RELION_BIND_BINARY_OVERRIDE}"
[[ "${EXACT_LOCAL_BUCKET_RADIX}" =~ ^(2|4)$ ]]
[[ "${EXACT_LOCAL_PHYSICAL_ORDER_CHUNK_SIZE}" =~ ^[0-9]+$ ]]
[[ "${STABLE_FOURIER_WINDOW_SHAPES}" =~ ^[01]$ ]]
[[ "${AUDIT_RAW_IMAGE_CACHE}" =~ ^[01]$ ]]
[[ "${EXECUTION_CONTRACT}" =~ ^(diagnostic|default|all_optimized_q32)$ ]]
[[ "${IMAGE_SIZE}" =~ ^[0-9]+$ ]]
[[ "${HOST_ATTRIBUTION}" =~ ^[01]$ ]]
test "${IMAGE_SIZE}" -gt 0
test "${PROFILED_ITERATION}" -lt "${NR_ITER_SCHEDULE}"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"
test "$(sha256sum "${RELION_BINARY}" | awk '{print $1}')" = "${EXPECTED_RELION_SHA256}"
test "$(sha256sum "${RELION_BIND_BINARY_OVERRIDE}" | awk '{print $1}')" = \
  "${EXPECTED_RELION_BIND_SHA256}"
if [[ -n "${REUSE_NATIVE_ROOT}" ]]; then
  test -n "${EXPECTED_REUSED_NATIVE_NSYS_SHA256}"
  [[ "${EXPECTED_REUSED_NATIVE_NSYS_SHA256}" =~ ^[0-9a-f]{64}$ ]]
  test -d "${REUSE_NATIVE_ROOT}"
  test "${REUSE_NATIVE_ROOT}" != "${ROOT}"
fi

# Derive the candidate selector list from the same Python helper that performs
# the post-run metadata validation.  This keeps submission labels and runtime
# contracts from drifting apart.
PROFILE_CANDIDATE_ENV=()
if [[ "${EXECUTION_CONTRACT}" != diagnostic ]]; then
  mapfile -t PROFILE_CANDIDATE_ENV < <(
    cd "${REPO_ROOT}"
    "${PIXI_PY}" - <<'PY'
from scripts.run_vdam_late_iteration_profile import _all_optimized_q32_environment

for name, value in sorted(_all_optimized_q32_environment().items()):
    print(f"{name}={value}")
PY
  )
  test "${#PROFILE_CANDIDATE_ENV[@]}" -gt 0
fi
case "${EXECUTION_CONTRACT}" in
  diagnostic)
    ;;
  default)
    for assignment in "${PROFILE_CANDIDATE_ENV[@]}"; do
      unset "${assignment%%=*}"
    done
    STABLE_FOURIER_WINDOW_SHAPES=0
    ;;
  all_optimized_q32)
    for assignment in "${PROFILE_CANDIDATE_ENV[@]}"; do
      export "${assignment}"
    done
    STABLE_FOURIER_WINDOW_SHAPES=1
    ;;
esac

source "${REPO_ROOT}/scripts/vdam_gpu_selection.sh"
vdam_select_target_gpu "${TARGET_GPU_UUID}" 0
selected_gpu_uuid=${VDAM_SELECTED_GPU_UUID}

mkdir -p "${PROVENANCE}" "${RUNTIME}/cuda" "${RELION_BIND_DIR}" \
  "${NATIVE_PLAIN}" "${NATIVE_PROFILE}" "${NSIGHT}"
touch "${ROOT}/SAFE_TO_DELETE"
git -C "${REPO_ROOT}" rev-parse HEAD > "${PROVENANCE}/repo_head.txt"
git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no \
  > "${PROVENANCE}/repo_status.txt"
printf '%s\n' "${selected_gpu_uuid}" > "${PROVENANCE}/selected_gpu_uuid.txt"
printf '%s\n' "${VDAM_VISIBLE_GPU_UUIDS_CSV}" > "${PROVENANCE}/allocated_gpu_uuids.csv"
printf 'exact_local_bucket_radix=%s\nexact_local_physical_order_chunk_size=%s\n' \
  "${EXACT_LOCAL_BUCKET_RADIX}" "${EXACT_LOCAL_PHYSICAL_ORDER_CHUNK_SIZE}" \
  > "${PROVENANCE}/candidate_parameters.txt"
printf 'stable_fourier_window_shapes=%s\n' \
  "${STABLE_FOURIER_WINDOW_SHAPES}" \
  >> "${PROVENANCE}/candidate_parameters.txt"
printf 'audit_raw_image_cache=%s\n' \
  "${AUDIT_RAW_IMAGE_CACHE}" \
  >> "${PROVENANCE}/candidate_parameters.txt"
printf 'execution_contract=%s\nimage_size=%s\n' \
  "${EXECUTION_CONTRACT}" "${IMAGE_SIZE}" \
  >> "${PROVENANCE}/candidate_parameters.txt"
printf 'host_attribution=%s\n' "${HOST_ATTRIBUTION}" \
  >> "${PROVENANCE}/candidate_parameters.txt"
nvidia-smi -q > "${PROVENANCE}/nvidia_smi.txt"
env | LC_ALL=C sort > "${PROVENANCE}/submission_environment.txt"
"${NSYS}" --version > "${PROVENANCE}/nsys_version.txt"
"${CUDA_TOOLKIT}/bin/nvcc" --version > "${PROVENANCE}/nvcc_version.txt"

# Resolve the guarded K=1 Gaussian defaults before a CUDA build or either
# profiler capture.  In particular, the GEMM macro needs explicit opt-outs for
# the fused/canonical backends that are default-on for this continuation path.
(
  cd "${REPO_ROOT}"
  "${PIXI_PY}" - "${PROVENANCE}/profile_environment_preflight.json" \
    "${EXECUTION_CONTRACT}" <<'PY'
import json
import sys
from pathlib import Path

from scripts.run_vdam_late_iteration_profile import (
    _validate_profile_contract_environment,
    _validate_profile_environment,
)

report = _validate_profile_environment()
report["execution_contract"] = _validate_profile_contract_environment(sys.argv[2])
Path(sys.argv[1]).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, sort_keys=True))
PY
) > "${PROVENANCE}/profile_environment_preflight.stdout" \
  2> "${PROVENANCE}/profile_environment_preflight.stderr"

checkpoint_prefix=${CHECKPOINT%_optimiser.star}
CHECKPOINT_FILES=(
  "${CHECKPOINT}"
  "${checkpoint_prefix}_model.star"
  "${checkpoint_prefix}_data.star"
  "${checkpoint_prefix}_sampling.star"
  "${checkpoint_prefix}_class001.mrc"
  "${checkpoint_prefix}_1moment001.mrc"
  "${checkpoint_prefix}_1moment002.mrc"
  "${checkpoint_prefix}_2moment001.mrc"
  "${DATA_STAR}"
  "${PARTICLE_STACK}"
  "${RELION_BINARY}"
  "${RELION_BIND_BINARY_OVERRIDE}"
  "${CUSPARSE_LIBRARY}"
  "${REPO_ROOT}/scripts/run_vdam_late_iteration_profile.sbatch"
  "${REPO_ROOT}/scripts/run_vdam_late_iteration_profile.py"
  "${REPO_ROOT}/scripts/vdam_relion_one_iteration.gdb"
  "${REPO_ROOT}/scripts/summarize_vdam_nsys_sqlite.py"
)
for path in "${CHECKPOINT_FILES[@]}"; do
  test -s "${path}"
done
sha256sum "${CHECKPOINT_FILES[@]}" > "${PROVENANCE}/static_inputs.sha256"

NATIVE_NSYS_BASE=${NSIGHT}/native_it$(printf '%03d' "${PROFILED_ITERATION}")
if [[ -n "${REUSE_NATIVE_ROOT}" ]]; then
  source_native_report=${REUSE_NATIVE_ROOT}/nsight/native_it$(printf '%03d' "${PROFILED_ITERATION}").nsys-rep
  source_native_manifest=${REUSE_NATIVE_ROOT}/provenance/static_inputs.sha256
  source_native_profile=${REUSE_NATIVE_ROOT}/native_profiled
  source_native_plain=${REUSE_NATIVE_ROOT}/native_unprofiled
  test -s "${source_native_report}"
  test -s "${source_native_manifest}"
  [[ "$(cat "${REUSE_NATIVE_ROOT}/provenance/repo_head.txt")" =~ ^[0-9a-f]{40}$ ]]
  test ! -s "${REUSE_NATIVE_ROOT}/provenance/repo_status.txt"
  test "$(sha256sum "${source_native_report}" | awk '{print $1}')" = \
    "${EXPECTED_REUSED_NATIVE_NSYS_SHA256}"
  test -s "${source_native_plain}/run_it$(printf '%03d' "${PROFILED_ITERATION}")_data.star"
  test -s "${source_native_profile}/run_it$(printf '%03d' "${PROFILED_ITERATION}")_data.star"
  test ! -e "${source_native_plain}/run_it$(printf '%03d' "$((PROFILED_ITERATION + 1))")_data.star"
  test ! -e "${source_native_profile}/run_it$(printf '%03d' "$((PROFILED_ITERATION + 1))")_data.star"
  test "$(rg -c 'VDAM_GDB_FIRST_EXPECTATION' "${source_native_plain}/runner.stdout")" -eq 1
  test "$(rg -c 'VDAM_GDB_SECOND_EXPECTATION' "${source_native_plain}/runner.stdout")" -eq 1
  test "$(rg -c 'VDAM_GDB_CUDA_PROFILER_START status=0' "${source_native_profile}/runner.stdout")" -eq 1
  test "$(rg -c 'VDAM_GDB_CUDA_PROFILER_STOP status=0' "${source_native_profile}/runner.stdout")" -eq 1
  NATIVE_REUSE_INPUTS=(
    "${CHECKPOINT}"
    "${checkpoint_prefix}_model.star"
    "${checkpoint_prefix}_data.star"
    "${checkpoint_prefix}_sampling.star"
    "${checkpoint_prefix}_class001.mrc"
    "${checkpoint_prefix}_1moment001.mrc"
    "${checkpoint_prefix}_1moment002.mrc"
    "${checkpoint_prefix}_2moment001.mrc"
    "${DATA_STAR}"
    "${PARTICLE_STACK}"
    "${RELION_BINARY}"
    "${CUSPARSE_LIBRARY}"
    "${REPO_ROOT}/scripts/vdam_relion_one_iteration.gdb"
  )
  "${PIXI_PY}" - "${PROVENANCE}/reused_native_inputs.json" \
    "${source_native_manifest}" "${NATIVE_REUSE_INPUTS[@]}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
expected = {}
for line in manifest_path.read_text().splitlines():
    digest, path = line.split(maxsplit=1)
    expected[path] = digest
records = []
for raw_path in sys.argv[3:]:
    path = Path(raw_path)
    if raw_path not in expected:
        raise RuntimeError(f"reused native manifest omitted {raw_path}")
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    digest = checksum.hexdigest()
    if digest != expected[raw_path]:
        raise RuntimeError(f"reused native input changed: {raw_path}")
    records.append({"path": raw_path, "sha256": digest})
output.write_text(json.dumps({"inputs": records}, indent=2, sort_keys=True) + "\n")
PY
  cp --reflink=auto "${source_native_report}" "${NATIVE_NSYS_BASE}.nsys-rep"
  test "$(sha256sum "${NATIVE_NSYS_BASE}.nsys-rep" | awk '{print $1}')" = \
    "${EXPECTED_REUSED_NATIVE_NSYS_SHA256}"
  printf '%s\n' "${REUSE_NATIVE_ROOT}" > "${PROVENANCE}/reused_native_root.txt"
  sha256sum "${source_native_report}" > "${PROVENANCE}/reused_native_nsys.sha256"
fi

build_started=$(date +%s.%N)
make -C "${REPO_ROOT}/recovar/cuda" \
  PYTHON="${PIXI_PY}" \
  NVCC="${CUDA_TOOLKIT}/bin/nvcc" \
  CUDA_ARCH='-gencode arch=compute_90,code=sm_90' \
  LIB="${CUDA_BINARY}" \
  > "${PROVENANCE}/cuda_build.stdout" \
  2> "${PROVENANCE}/cuda_build.stderr"
build_finished=$(date +%s.%N)
"${PIXI_PY}" - "${build_started}" "${build_finished}" \
  "${PROVENANCE}/cuda_build_wall.json" <<'PY'
import json
import sys
from pathlib import Path

start, stop = float(sys.argv[1]), float(sys.argv[2])
Path(sys.argv[3]).write_text(json.dumps({"wall_s": stop - start}, indent=2) + "\n")
PY
test -s "${CUDA_BINARY}"
sha256sum "${CUDA_BINARY}" > "${PROVENANCE}/built_cuda.sha256"
cp --reflink=auto "${RELION_BIND_BINARY_OVERRIDE}" "${RELION_BIND_DIR}/"
RELION_BIND_BINARY=${RELION_BIND_DIR}/$(basename "${RELION_BIND_BINARY_OVERRIDE}")
test "$(sha256sum "${RELION_BIND_BINARY}" | awk '{print $1}')" = \
  "${EXPECTED_RELION_BIND_SHA256}"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${selected_gpu_uuid}
export CUDA_LAUNCH_BLOCKING=0
export RECOVAR_EXPECTED_REPO_ROOT=${REPO_ROOT}
export RECOVAR_CUDA_LIB=${CUDA_BINARY}
export RECOVAR_RELION_BIND_BUILD_DIR=${RELION_BIND_DIR}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=${RUNTIME}/jax_cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export TMPDIR=${RUNTIME}/tmp
export PIXI_HOME=${RUNTIME}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/rattler_cache
export PATH=${CUDA_TOOLKIT}/bin:${PATH}
export LD_LIBRARY_PATH=${RELION_RUNTIME}/system_dev_overlay/lib:${MPI_ROOT}/lib64:${RELION_CUDA_TOOLKIT}/lib64:${RELION_CUDA_TOOLKIT}/targets/x86_64-linux/lib:${CUDA_TOOLKIT}/lib64:${CUDA_TOOLKIT}/targets/x86_64-linux/lib
mkdir -p "${JAX_COMPILATION_CACHE_DIR}" "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}"
touch "${JAX_COMPILATION_CACHE_DIR}/SAFE_TO_DELETE"

(
  cd "${REPO_ROOT}"
  env LD_PRELOAD="${CUSPARSE_LIBRARY}" "${PIXI_PY}" - <<'PY'
import jax
import recovar
import recovar.cuda_backproject as cuda_backproject
from recovar.relion_bind import _relion_bind_core as relion_bind_core

assert recovar.__file__ is not None
assert any(device.platform in {"gpu", "cuda"} for device in jax.devices())
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
assert callable(getattr(relion_bind_core, "vdam_rnd_unif_sequence", None))
print("VDAM late profiler GPU/runtime gate passed", jax.devices(), recovar.__file__)
PY
) > "${PROVENANCE}/runtime_gate.stdout" 2> "${PROVENANCE}/runtime_gate.stderr"
vdam_verify_selected_gpu "${selected_gpu_uuid}"

native_plain_started=
native_plain_finished=
native_profile_started=
native_profile_finished=
if [[ -z "${REUSE_NATIVE_ROOT}" ]]; then
  NATIVE_ARGS=(
    "${RELION_BINARY}"
    --continue "${CHECKPOINT}"
    --o "${NATIVE_PLAIN}/run"
    --iter "${NR_ITER_SCHEDULE}"
    --pool 3
    --gpu 0
    --j 8
  )
  printf '%q ' gdb -q -batch -ex 'set $vdam_profile_cuda = 0' \
    -x "${REPO_ROOT}/scripts/vdam_relion_one_iteration.gdb" --args "${NATIVE_ARGS[@]}" \
    > "${PROVENANCE}/native_unprofiled_command.sh"
  printf '\n' >> "${PROVENANCE}/native_unprofiled_command.sh"
  native_plain_started=$(date +%s.%N)
  (
    cd "${DATA_DIR}"
    gdb -q -batch -ex 'set $vdam_profile_cuda = 0' \
      -x "${REPO_ROOT}/scripts/vdam_relion_one_iteration.gdb" \
      --args "${NATIVE_ARGS[@]}"
  ) > "${NATIVE_PLAIN}/runner.stdout" 2> "${NATIVE_PLAIN}/runner.stderr"
  native_plain_finished=$(date +%s.%N)
  test -s "${NATIVE_PLAIN}/run_it$(printf '%03d' "${PROFILED_ITERATION}")_data.star"
  test ! -e "${NATIVE_PLAIN}/run_it$(printf '%03d' "$((PROFILED_ITERATION + 1))")_data.star"
  test "$(rg -c 'VDAM_GDB_FIRST_EXPECTATION' "${NATIVE_PLAIN}/runner.stdout")" -eq 1
  test "$(rg -c 'VDAM_GDB_SECOND_EXPECTATION' "${NATIVE_PLAIN}/runner.stdout")" -eq 1

  NATIVE_PROFILE_ARGS=(
    "${RELION_BINARY}"
    --continue "${CHECKPOINT}"
    --o "${NATIVE_PROFILE}/run"
    --iter "${NR_ITER_SCHEDULE}"
    --pool 3
    --gpu 0
    --j 8
  )
  printf '%q ' "${NSYS}" profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
    --kill=none --force-overwrite=true --capture-range=cudaProfilerApi \
    --capture-range-end=stop -o "${NATIVE_NSYS_BASE}" \
    gdb -q -batch -ex 'set $vdam_profile_cuda = 1' \
    -x "${REPO_ROOT}/scripts/vdam_relion_one_iteration.gdb" \
    --args "${NATIVE_PROFILE_ARGS[@]}" > "${PROVENANCE}/native_nsys_command.sh"
  printf '\n' >> "${PROVENANCE}/native_nsys_command.sh"
  native_profile_started=$(date +%s.%N)
  (
    cd "${DATA_DIR}"
    "${NSYS}" profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
      --kill=none --force-overwrite=true --capture-range=cudaProfilerApi \
      --capture-range-end=stop -o "${NATIVE_NSYS_BASE}" \
      gdb -q -batch -ex 'set $vdam_profile_cuda = 1' \
      -x "${REPO_ROOT}/scripts/vdam_relion_one_iteration.gdb" \
      --args "${NATIVE_PROFILE_ARGS[@]}"
  ) > "${NATIVE_PROFILE}/runner.stdout" 2> "${NATIVE_PROFILE}/runner.stderr"
  native_profile_finished=$(date +%s.%N)
  test -s "${NATIVE_NSYS_BASE}.nsys-rep"
  test -s "${NATIVE_PROFILE}/run_it$(printf '%03d' "${PROFILED_ITERATION}")_data.star"
  test ! -e "${NATIVE_PROFILE}/run_it$(printf '%03d' "$((PROFILED_ITERATION + 1))")_data.star"
  test "$(rg -c 'VDAM_GDB_CUDA_PROFILER_START status=0' "${NATIVE_PROFILE}/runner.stdout")" -eq 1
  test "$(rg -c 'VDAM_GDB_CUDA_PROFILER_STOP status=0' "${NATIVE_PROFILE}/runner.stdout")" -eq 1
  vdam_verify_selected_gpu "${selected_gpu_uuid}"
fi

RECOVAR_NSYS_BASE=${NSIGHT}/recovar_it$(printf '%03d' "${PROFILED_ITERATION}")_warm
RECOVAR_NSYS_TRACE=cuda,nvtx,osrt
RECOVAR_NSYS_EXTRA=(--sample=none --cpuctxsw=none)
if [[ "${HOST_ATTRIBUTION}" = 1 ]]; then
  RECOVAR_NSYS_TRACE=cuda,nvtx,osrt,python-gil
  RECOVAR_NSYS_EXTRA+=(--python-sampling=true --python-sampling-frequency=1000)
fi
RECOVAR_COMMAND=(
  env "LD_PRELOAD=${CUSPARSE_LIBRARY}"
  "${PIXI_PY}" -m scripts.run_vdam_late_iteration_profile
  --checkpoint-optimiser "${CHECKPOINT}"
  --input-star "${DATA_STAR}"
  --data-dir "${DATA_DIR}"
  --output-root "${RECOVAR_PROFILE}"
  --checkpoint-iteration "${CHECKPOINT_ITERATION}"
  --nr-iter "${NR_ITER_SCHEDULE}"
  --random-seed 29
  --image-batch-size 500
  --image-size "${IMAGE_SIZE}"
  --exact-local-bucket-radix "${EXACT_LOCAL_BUCKET_RADIX}"
  --exact-local-physical-order-chunk-size "${EXACT_LOCAL_PHYSICAL_ORDER_CHUNK_SIZE}"
  --execution-contract "${EXECUTION_CONTRACT}"
  --cuda-profiler-range
)
if [[ "${STABLE_FOURIER_WINDOW_SHAPES}" = 1 ]]; then
  RECOVAR_COMMAND+=(--stable-fourier-window-shapes)
fi
if [[ "${AUDIT_RAW_IMAGE_CACHE}" = 1 ]]; then
  RECOVAR_COMMAND+=(--audit-raw-image-cache)
fi
printf '%q ' "${NSYS}" profile --trace="${RECOVAR_NSYS_TRACE}" \
  "${RECOVAR_NSYS_EXTRA[@]}" \
  --kill=none --force-overwrite=true --capture-range=cudaProfilerApi \
  --capture-range-end=stop -o "${RECOVAR_NSYS_BASE}" "${RECOVAR_COMMAND[@]}" \
  > "${PROVENANCE}/recovar_nsys_command.sh"
printf '\n' >> "${PROVENANCE}/recovar_nsys_command.sh"
recovar_profile_started=$(date +%s.%N)
(
  cd "${REPO_ROOT}"
  "${NSYS}" profile --trace="${RECOVAR_NSYS_TRACE}" \
    "${RECOVAR_NSYS_EXTRA[@]}" \
    --kill=none --force-overwrite=true --capture-range=cudaProfilerApi \
    --capture-range-end=stop -o "${RECOVAR_NSYS_BASE}" \
    "${RECOVAR_COMMAND[@]}"
) > "${RECOVAR_PROFILE}.stdout" 2> "${RECOVAR_PROFILE}.stderr"
recovar_profile_finished=$(date +%s.%N)
test -s "${RECOVAR_NSYS_BASE}.nsys-rep"
test -s "${RECOVAR_PROFILE}/profile_summary.json"
test -s "${RECOVAR_PROFILE}/warm/run_it$(printf '%03d' "${PROFILED_ITERATION}")_recovar_meta.json"
test ! -e "${RECOVAR_PROFILE}/warm/run_it$(printf '%03d' "$((PROFILED_ITERATION + 1))")_recovar_meta.json"
vdam_verify_selected_gpu "${selected_gpu_uuid}"

"${PIXI_PY}" - "${RECOVAR_PROFILE}/profile_summary.json" \
  "${PROVENANCE}/recovar_execution_contract.json" <<'PY'
import json
import sys
from pathlib import Path

profile = json.loads(Path(sys.argv[1]).read_text())
payload = {
    "mode": profile["execution_contract_mode"],
    "environment": profile["execution_contract_environment"],
    "cold": profile["cold"]["execution_contract"],
    "warm": profile["warm"]["execution_contract"],
}
if payload["mode"] != "diagnostic":
    for label in ("cold", "warm"):
        contract = payload[label]
        if not contract.get("profile_exact") or not contract.get("profile_checked"):
            raise RuntimeError(f"{label} execution contract is not exact")
Path(sys.argv[2]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
test -s "${PROVENANCE}/recovar_execution_contract.json"

for implementation in native recovar; do
  if [[ "${implementation}" == native ]]; then
    report=${NATIVE_NSYS_BASE}.nsys-rep
  else
    report=${RECOVAR_NSYS_BASE}.nsys-rep
  fi
  sqlite=${NSIGHT}/${implementation}.sqlite
  "${NSYS}" export --type=sqlite --force-overwrite=true --output="${sqlite}" "${report}"
  test -s "${sqlite}"
  for summary in cuda_gpu_kern_sum cuda_gpu_kern_gb_sum cuda_api_sum cuda_gpu_sum; do
    "${NSYS}" stats --report "${summary}" --format csv "${sqlite}" \
      > "${NSIGHT}/${implementation}_${summary}.csv"
    test -s "${NSIGHT}/${implementation}_${summary}.csv"
  done
  (
    cd "${REPO_ROOT}"
    "${PIXI_PY}" -m scripts.summarize_vdam_nsys_sqlite \
      --sqlite "${sqlite}" --output "${NSIGHT}/${implementation}_summary.json"
  )
done

"${PIXI_PY}" - "${PROVENANCE}/run.json" \
  "${SLURM_JOB_ID}" "${EXPECTED_REPO_HEAD}" "${selected_gpu_uuid}" \
  "${CHECKPOINT_ITERATION}" "${PROFILED_ITERATION}" "${NR_ITER_SCHEDULE}" \
  "${native_plain_started}" "${native_plain_finished}" \
  "${native_profile_started}" "${native_profile_finished}" \
  "${recovar_profile_started}" "${recovar_profile_finished}" \
  "${REUSE_NATIVE_ROOT}" "${EXPECTED_REUSED_NATIVE_NSYS_SHA256}" \
  "${HOST_ATTRIBUTION}" <<'PY'
import json
import sys
from pathlib import Path

reuse_root = Path(sys.argv[14]) if sys.argv[14] else None
if reuse_root is None:
    execution_order = [
        "native_unprofiled",
        "native_profiled",
        "recovar_cold",
        "recovar_warm",
    ]
    native_unprofiled_wall = float(sys.argv[9]) - float(sys.argv[8])
    native_profiled_wall = float(sys.argv[11]) - float(sys.argv[10])
    native_capture = {"mode": "captured_in_job"}
else:
    source_job_path = reuse_root / "provenance" / "slurm_job_id.txt"
    if source_job_path.is_file():
        source_job_id = source_job_path.read_text().strip()
    else:
        source_job_id = "unknown"
        submission_environment = (
            reuse_root / "provenance" / "submission_environment.txt"
        )
        if submission_environment.is_file():
            for line in submission_environment.read_text().splitlines():
                if line.startswith("SLURM_JOB_ID="):
                    source_job_id = line.split("=", 1)[1]
                    break
    execution_order = ["native_reused", "recovar_cold", "recovar_warm"]
    native_unprofiled_wall = None
    native_profiled_wall = None
    native_capture = {
        "mode": "reused",
        "source_root": str(reuse_root.resolve()),
        "source_job_id": source_job_id,
        "source_gpu_uuid": (
            reuse_root / "provenance" / "selected_gpu_uuid.txt"
        ).read_text().strip(),
        "nsys_sha256": sys.argv[15],
    }

payload = {
    "schema": "recovar.vdam_late_iteration_same_h100_gate.v1",
    "classification": "diagnostic_performance_only",
    "job_id": sys.argv[2],
    "git_head": sys.argv[3],
    "gpu_uuid": sys.argv[4],
    "checkpoint_iteration": int(sys.argv[5]),
    "profiled_iteration": int(sys.argv[6]),
    "nr_iter_schedule": int(sys.argv[7]),
    "execution_order": execution_order,
    "walls_s": {
        "native_unprofiled_process": native_unprofiled_wall,
        "native_profiled_process": native_profiled_wall,
        "recovar_profile_process": float(sys.argv[13]) - float(sys.argv[12]),
    },
    "native_capture": native_capture,
    "host_attribution": bool(int(sys.argv[16])),
    "timing_truth_allowed": not bool(int(sys.argv[16])),
    "native_stop_mechanism": "gdb second-expectation sentinel after exact profiled iteration",
    "science_promotion_allowed": False,
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=no)"
test "$(sha256sum "${RELION_BINARY}" | awk '{print $1}')" = "${EXPECTED_RELION_SHA256}"
test "$(sha256sum "${RELION_BIND_BINARY_OVERRIDE}" | awk '{print $1}')" = \
  "${EXPECTED_RELION_BIND_SHA256}"
touch "${ROOT}/COMPLETED"
