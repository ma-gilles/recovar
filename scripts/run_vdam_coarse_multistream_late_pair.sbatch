#!/usr/bin/env bash
#SBATCH --job-name=vdam-coarse-crossed
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --nodelist=della-h21g4
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=00:35:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-coarse-crossed-%j.out

set -euo pipefail

: "${VDAM_COARSE_MULTISTREAM_PAIR_ROOT:?set a new disposable output root}"
: "${VDAM_LATE_PROFILE_CHECKPOINT:?set the exact run_it180_optimiser.star}"
: "${VDAM_LATE_PROFILE_DATA_STAR:?set the exact run_it180_data.star}"
: "${VDAM_LATE_PROFILE_DATA_DIR:?set the particle-stack directory}"
: "${QUALIFIED_GPU_GATE_ROOT:?set the completed focused GPU gate root}"
: "${EXPECTED_CUDA_SHA256:?pin the focused-gate CUDA binary}"
: "${EXPECTED_REPO_HEAD:?pin the tracked RECOVAR source head}"
: "${EXPECTED_SOURCE_MANIFEST_SHA256:?pin the benchmark source manifest}"
: "${EXPECTED_INPUT_MANIFEST_SHA256:?pin the GF46 input manifest}"
: "${EXPECTED_NODE_NAME:?pin the Slurm node}"
: "${TARGET_GPU_UUID:?pin the physical H100 UUID}"
: "${RELION_BIND_BINARY_OVERRIDE:?set the qualified RELION binding binary}"
: "${EXPECTED_RELION_BIND_SHA256:?pin the qualified RELION binding binary}"
: "${EXPECTED_INTERPRETER_SHA256:?pin the pixi interpreter}"
: "${PRECOMPUTED_RESOURCE_REPORT:?set the login-node atomic resource JSON}"
: "${EXPECTED_RESOURCE_REPORT_SHA256:?pin the atomic resource JSON}"
: "${PRECOMPUTED_RESOURCE_DEMANGLED:?set the demangled cuobjdump report}"
: "${EXPECTED_RESOURCE_DEMANGLED_SHA256:?pin the demangled cuobjdump report}"

ROOT=${VDAM_COARSE_MULTISTREAM_PAIR_ROOT}
CHECKPOINT=${VDAM_LATE_PROFILE_CHECKPOINT}
DATA_STAR=${VDAM_LATE_PROFILE_DATA_STAR}
DATA_DIR=${VDAM_LATE_PROFILE_DATA_DIR}
PARTICLE_STACK=${VDAM_LATE_PROFILE_PARTICLE_STACK:-${DATA_DIR}/particles.128.mrcs}
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
PIXI_PY=${PIXI_PY_OVERRIDE:-${REPO_ROOT}/.pixi/envs/default/bin/python}
NSYS=${VDAM_NSYS_BIN:-/opt/nvidia/nsight-systems/2025.3.2/target-linux-x64/nsys}
CUDA_TOOLKIT=${CUDA_TOOLKIT:-/usr/local/cuda-13.1}
RELION_CUDA_TOOLKIT=${RELION_CUDA_TOOLKIT:-/usr/local/cuda-12.6}
CUSPARSE_LIBRARY=${CUSPARSE_LIBRARY:-${RELION_CUDA_TOOLKIT}/lib64/libcusparse.so.12}
RELION_RUNTIME=${RELION_RUNTIME:-/scratch/gpfs/GILLES/mg6942/recovar_dev/relion_k4_100k_dispatchv2_20260717}
MPI_ROOT=${MPI_ROOT:-/usr/local/openmpi/cuda-12.6/4.1.6/gcc}
CHECKPOINT_ITERATION=${CHECKPOINT_ITERATION:-180}
PROFILED_ITERATION=$((CHECKPOINT_ITERATION + 1))
NR_ITER_SCHEDULE=${NR_ITER_SCHEDULE:-200}
QUALIFIED_CUDA=${QUALIFIED_GPU_GATE_ROOT}/runtime/cuda/libcuda_backproject.so
PROVENANCE=${ROOT}/provenance
RUNTIME=${ROOT}/runtime
NSIGHT=${ROOT}/nsight
CUDA_BINARY=${RUNTIME}/cuda/libcuda_backproject.so
RELION_BIND_DIR=${RUNTIME}/relion_bind
FOCUSED_TEST_NODE=tests/unit/test_cuda_relion_fine_diff2.py::test_relion_coarse_vdam_multistream_atomic_stays_in_lane_envelope
SOURCE_FILES=(
  recovar/cuda/Makefile
  recovar/cuda/cuda_backproject.cu
  recovar/cuda_backproject.py
  recovar/em/scoring/significance.py
  recovar/em/classification/k_class.py
  recovar/commands/initial_model.py
  recovar/em/vdam/dense_adapter.py
  recovar/em/vdam/driver.py
  recovar/em/vdam/iteration_loop.py
  recovar/em/vdam/m_step.py
  recovar/em/vdam/schedules.py
  tests/conftest.py
  tests/unit/test_cuda_relion_fine_diff2.py
  scripts/run_ab_initio.py
  scripts/run_vdam_late_iteration_profile.py
  scripts/summarize_vdam_nsys_sqlite.py
  scripts/run_vdam_coarse_multistream_late_pair.sbatch
  scripts/vdam_gpu_selection.sh
  pixi.lock
  pixi.toml
  pyproject.toml
)

on_error() {
  local status=$?
  local message="VDAM coarse atomic late pair failed: status=${status} line=${BASH_LINENO[0]} command=${BASH_COMMAND}"
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
test -x "${NSYS}"
test -s "${CUSPARSE_LIBRARY}"
test -s "${CHECKPOINT}"
test -s "${DATA_STAR}"
test -d "${DATA_DIR}"
test -s "${PARTICLE_STACK}"
test -e "${QUALIFIED_GPU_GATE_ROOT}/COMPLETED"
test -s "${QUALIFIED_GPU_GATE_ROOT}/provenance/source_manifest.sha256"
test -s "${QUALIFIED_CUDA}"
test -s "${RELION_BIND_BINARY_OVERRIDE}"
test -s "${PRECOMPUTED_RESOURCE_REPORT}"
test -s "${PRECOMPUTED_RESOURCE_DEMANGLED}"
test "${SLURMD_NODENAME:-$(hostname -s)}" = "${EXPECTED_NODE_NAME}"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1)"
repo_tree=$(git -C "${REPO_ROOT}" rev-parse 'HEAD^{tree}')
test "$(sha256sum "${QUALIFIED_CUDA}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
test "$(sha256sum "${RELION_BIND_BINARY_OVERRIDE}" | awk '{print $1}')" = \
  "${EXPECTED_RELION_BIND_SHA256}"
test "$(sha256sum "${PRECOMPUTED_RESOURCE_REPORT}" | awk '{print $1}')" = \
  "${EXPECTED_RESOURCE_REPORT_SHA256}"
test "$(sha256sum "${PRECOMPUTED_RESOURCE_DEMANGLED}" | awk '{print $1}')" = \
  "${EXPECTED_RESOURCE_DEMANGLED_SHA256}"
(
  cd "${REPO_ROOT}"
  sha256sum -c "${QUALIFIED_GPU_GATE_ROOT}/provenance/source_manifest.sha256"
) > /dev/null

initial_allocation_spec=${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}}
source "${REPO_ROOT}/scripts/vdam_gpu_selection.sh"
vdam_assert_target_gpu_allocated "${TARGET_GPU_UUID}" "${initial_allocation_spec}"
vdam_select_target_gpu "${TARGET_GPU_UUID}" 0
selected_gpu_uuid=${VDAM_SELECTED_GPU_UUID}
test "${selected_gpu_uuid}" = "${TARGET_GPU_UUID}"
gpu_name=$(nvidia-smi -i "${selected_gpu_uuid}" --query-gpu=name --format=csv,noheader | sed -n '1p' | sed 's/[[:space:]]*$//')
[[ "${gpu_name}" == *H100* ]]

mkdir -p "${PROVENANCE}" "${RUNTIME}/cuda" "${RELION_BIND_DIR}" "${NSIGHT}"
touch "${ROOT}/SAFE_TO_DELETE"
cp --reflink=auto "${QUALIFIED_CUDA}" "${CUDA_BINARY}"
cp --reflink=auto "${RELION_BIND_BINARY_OVERRIDE}" "${RELION_BIND_DIR}/"
cp --reflink=auto "${PRECOMPUTED_RESOURCE_REPORT}" \
  "${PROVENANCE}/coarse_atomic_resources.json"
cp --reflink=auto "${PRECOMPUTED_RESOURCE_DEMANGLED}" \
  "${PROVENANCE}/cuda_resource_usage.demangled.txt"
RELION_BIND_BINARY=${RELION_BIND_DIR}/$(basename "${RELION_BIND_BINARY_OVERRIDE}")
test "$(sha256sum "${CUDA_BINARY}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
test "$(sha256sum "${RELION_BIND_BINARY}" | awk '{print $1}')" = \
  "${EXPECTED_RELION_BIND_SHA256}"
test "$(sha256sum "${PROVENANCE}/coarse_atomic_resources.json" | awk '{print $1}')" = \
  "${EXPECTED_RESOURCE_REPORT_SHA256}"
test "$(sha256sum "${PROVENANCE}/cuda_resource_usage.demangled.txt" | awk '{print $1}')" = \
  "${EXPECTED_RESOURCE_DEMANGLED_SHA256}"

"${PIXI_PY}" - "${PROVENANCE}/coarse_atomic_resources.json" \
  "${EXPECTED_CUDA_SHA256}" "${EXPECTED_RESOURCE_DEMANGLED_SHA256}" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
assert report["binary_sha256"] == sys.argv[2], report
assert report["demangled_resource_report_sha256"] == sys.argv[3], report
canonical = report["resources"]["canonical"]
atomic = report["resources"]["atomic"]
assert canonical == {"registers_per_thread": 56, "static_shared_bytes": 15232}
assert atomic == {"registers_per_thread": 48, "static_shared_bytes": 7040}
assert canonical["static_shared_bytes"] - atomic["static_shared_bytes"] == 8192
assert canonical["registers_per_thread"] - atomic["registers_per_thread"] == 8
PY

checkpoint_prefix=${CHECKPOINT%_optimiser.star}
INPUT_CANDIDATES=(
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
)
declare -A SEEN_INPUT_FILES=()
INPUT_FILES=()
for path in "${INPUT_CANDIDATES[@]}"; do
  test -s "${path}"
  resolved_path=$(realpath -e -- "${path}")
  if [[ -z "${SEEN_INPUT_FILES[${resolved_path}]+x}" ]]; then
    INPUT_FILES+=("${path}")
    SEEN_INPUT_FILES["${resolved_path}"]=1
  fi
done
test "${#INPUT_FILES[@]}" -eq "${#SEEN_INPUT_FILES[@]}"
sha256sum "${INPUT_FILES[@]}" > "${PROVENANCE}/input_manifest.sha256"
input_manifest_sha256=$(sha256sum "${PROVENANCE}/input_manifest.sha256" | awk '{print $1}')
test "${input_manifest_sha256}" = "${EXPECTED_INPUT_MANIFEST_SHA256}"
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
sha256sum "${CUDA_BINARY}" > "${PROVENANCE}/qualified_cuda.sha256"
sha256sum "${RELION_BIND_BINARY}" > "${PROVENANCE}/relion_bind.sha256"
nvidia-smi -i "${selected_gpu_uuid}" -q > "${PROVENANCE}/nvidia_smi.txt"
env | LC_ALL=C sort > "${PROVENANCE}/submission_environment.txt"
"${NSYS}" --version > "${PROVENANCE}/nsys_version.txt"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV JAX_PLATFORM_NAME
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${selected_gpu_uuid}
export RECOVAR_EXPECTED_REPO_ROOT=${REPO_ROOT}
export RECOVAR_CUDA_LIB=${CUDA_BINARY}
export RECOVAR_RELION_BIND_BUILD_DIR=${RELION_BIND_DIR}
export RECOVAR_EM_RAW_IMAGE_CACHE=off
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_LOG_COMPILES=1
export TMPDIR=${RUNTIME}/tmp
export PIXI_HOME=${RUNTIME}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/rattler_cache
export PATH=${CUDA_TOOLKIT}/bin:${PATH}
export LD_LIBRARY_PATH=${RELION_RUNTIME}/system_dev_overlay/lib:${MPI_ROOT}/lib64:${RELION_CUDA_TOOLKIT}/lib64:${RELION_CUDA_TOOLKIT}/targets/x86_64-linux/lib:${CUDA_TOOLKIT}/lib64:${CUDA_TOOLKIT}/targets/x86_64-linux/lib
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}"

(
  cd "${REPO_ROOT}"
  env LD_PRELOAD="${CUSPARSE_LIBRARY}" "${PIXI_PY}" - <<'PY'
import os
import pathlib

import jax
import recovar
import recovar.cuda_backproject as cuda_backproject
from recovar.relion_bind import _relion_bind_core as relion_bind_core
from recovar.utils.parity_provenance import assert_parity_ancestors

repo = pathlib.Path(os.environ["RECOVAR_EXPECTED_REPO_ROOT"]).resolve()
assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo) + "/")
assert ".pixi/envs/default/" in str(pathlib.Path(jax.__file__).resolve())
assert jax.default_backend() == "gpu"
assert len(jax.devices("gpu")) == 1
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
assert callable(getattr(relion_bind_core, "vdam_rnd_unif_sequence", None))
assert_parity_ancestors()
print("VDAM coarse atomic late-pair runtime gate passed", jax.devices(), recovar.__file__)
PY
) > "${PROVENANCE}/runtime_gate.stdout" 2> "${PROVENANCE}/runtime_gate.stderr"
vdam_verify_selected_gpu "${selected_gpu_uuid}"

(
  cd "${REPO_ROOT}"
  env LD_PRELOAD="${CUSPARSE_LIBRARY}" \
    JAX_COMPILATION_CACHE_DIR="${RUNTIME}/focused_jax_cache" \
    RECOVAR_REQUIRE_CUSTOM_CUDA_FOR_TESTS=1 \
    RECOVAR_ENABLE_CUSTOM_CUDA=1 \
    JAX_PLATFORMS=cuda,cpu \
    "${PIXI_PY}" -m pytest -v --run-gpu \
      --junitxml="${ROOT}/focused_pytest.junit.xml" \
      "${FOCUSED_TEST_NODE}"
) > "${ROOT}/focused_pytest.stdout" 2> "${ROOT}/focused_pytest.stderr"
test -s "${ROOT}/focused_pytest.junit.xml"
"${PIXI_PY}" - "${ROOT}/focused_pytest.junit.xml" <<'PY'
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
suite = root if root.tag == "testsuite" else root.find("testsuite")
assert suite is not None
assert int(suite.attrib["tests"]) == 1
assert int(suite.attrib.get("failures", 0)) == 0
assert int(suite.attrib.get("errors", 0)) == 0
assert int(suite.attrib.get("skipped", 0)) == 0
PY
vdam_verify_selected_gpu "${selected_gpu_uuid}"

RUN_LABELS=(
  canonical_serial_1
  atomic_serial_1
  atomic_multistream_1
  canonical_multistream_1
  canonical_multistream_2
  atomic_multistream_2
  atomic_serial_2
  canonical_serial_2
)
RUN_WORKERS=(0 0 8 8 8 8 0 0)
RUN_SINGLE_LANE=(0 0 0 0 0 0 0 0)
RUN_NATIVE_ATOMIC=(0 1 1 0 0 1 1 0)
test "${#RUN_LABELS[@]}" -eq "${#RUN_WORKERS[@]}"
test "${#RUN_LABELS[@]}" -eq "${#RUN_SINGLE_LANE[@]}"
test "${#RUN_LABELS[@]}" -eq "${#RUN_NATIVE_ATOMIC[@]}"
printf 'order\tlabel\tworkers\tsingle_lane_canonical\tnative_atomic_reduction\tnsys_base\n' \
  > "${PROVENANCE}/execution_order.tsv"
for index in "${!RUN_LABELS[@]}"; do
  label=${RUN_LABELS[index]}
  workers=${RUN_WORKERS[index]}
  single_lane=${RUN_SINGLE_LANE[index]}
  native_atomic=${RUN_NATIVE_ATOMIC[index]}
  run_root=${ROOT}/runs/${label}
  cache_root=${run_root}/jax_cache
  nsys_base=${NSIGHT}/${label}_it$(printf '%03d' "${PROFILED_ITERATION}")_warm
  mkdir -p "${run_root}" "${cache_root}"
  touch "${cache_root}/SAFE_TO_DELETE"
  printf '%d\t%s\t%s\t%s\t%s\t%s\n' \
    "$((index + 1))" "${label}" "${workers}" "${single_lane}" \
    "${native_atomic}" "${nsys_base}" \
    >> "${PROVENANCE}/execution_order.tsv"
  command=(
    env
    -u RECOVAR_RELION_COARSE_CANONICAL_REDUCTION
    "LD_PRELOAD=${CUSPARSE_LIBRARY}"
    "RECOVAR_K1_COARSE_MULTISTREAM_WORKERS=${workers}"
    "RECOVAR_K1_COARSE_SINGLE_LANE_CANONICAL=${single_lane}"
    "RECOVAR_K1_COARSE_NATIVE_ATOMIC_REDUCTION=${native_atomic}"
    "JAX_COMPILATION_CACHE_DIR=${cache_root}"
    "${PIXI_PY}" -m scripts.run_vdam_late_iteration_profile
    --checkpoint-optimiser "${CHECKPOINT}"
    --input-star "${DATA_STAR}"
    --data-dir "${DATA_DIR}"
    --output-root "${run_root}/profile"
    --checkpoint-iteration "${CHECKPOINT_ITERATION}"
    --nr-iter "${NR_ITER_SCHEDULE}"
    --random-seed 29
    --image-batch-size 500
    --exact-local-bucket-radix 4
    --exact-local-physical-order-chunk-size 0
    --cuda-profiler-range
  )
  printf '%q ' "${NSYS}" profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
    --kill=none --force-overwrite=true --capture-range=cudaProfilerApi \
    --capture-range-end=stop -o "${nsys_base}" "${command[@]}" \
    > "${PROVENANCE}/${label}_command.sh"
  printf '\n' >> "${PROVENANCE}/${label}_command.sh"
  /usr/bin/time -v -o "${run_root}/process.time" \
    "${NSYS}" profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
      --kill=none --force-overwrite=true --capture-range=cudaProfilerApi \
      --capture-range-end=stop -o "${nsys_base}" "${command[@]}" \
      > "${run_root}/runner.stdout" 2> "${run_root}/runner.stderr"
  test -s "${nsys_base}.nsys-rep"
  test -s "${run_root}/profile/profile_summary.json"
  test -s "${run_root}/profile/warm/run_it$(printf '%03d' "${PROFILED_ITERATION}")_recovar_meta.json"
  test ! -e "${run_root}/profile/warm/run_it$(printf '%03d' "$((PROFILED_ITERATION + 1))")_recovar_meta.json"
  test "$(sha256sum "${CUDA_BINARY}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
  vdam_verify_selected_gpu "${selected_gpu_uuid}"

  sqlite=${NSIGHT}/${label}.sqlite
  "${NSYS}" export --type=sqlite --force-overwrite=true --output="${sqlite}" "${nsys_base}.nsys-rep"
  test -s "${sqlite}"
  (
    cd "${REPO_ROOT}"
    "${PIXI_PY}" -m scripts.summarize_vdam_nsys_sqlite \
      --sqlite "${sqlite}" --output "${NSIGHT}/${label}_summary.json"
  )
  find "${cache_root}" -maxdepth 1 -type f -printf '%f\n' | LC_ALL=C sort \
    > "${run_root}/jax_cache_files.txt"
  printf '%s\n' "$(wc -l < "${run_root}/jax_cache_files.txt")" \
    > "${run_root}/jax_cache_file_count.txt"
done

"${PIXI_PY}" - "${PROVENANCE}/run.json" \
  "${SLURM_JOB_ID}" "${EXPECTED_REPO_HEAD}" "${selected_gpu_uuid}" \
  "${SLURMD_NODENAME:-$(hostname -s)}" "${gpu_name}" "${EXPECTED_CUDA_SHA256}" \
  "${source_manifest_sha256}" "${input_manifest_sha256}" \
  "${QUALIFIED_GPU_GATE_ROOT}" "${repo_tree}" \
  "${EXPECTED_RELION_BIND_SHA256}" "${EXPECTED_INTERPRETER_SHA256}" \
  "${EXPECTED_RESOURCE_REPORT_SHA256}" \
  "${EXPECTED_RESOURCE_DEMANGLED_SHA256}" <<'PY'
import json
import sys
from pathlib import Path

payload = {
    "schema": "recovar.vdam_coarse_atomic_multistream_crossed.v3",
    "classification": "diagnostic_performance_only",
    "job_id": sys.argv[2],
    "git_head": sys.argv[3],
    "git_tree": sys.argv[11],
    "relion_bind_sha256": sys.argv[12],
    "interpreter_sha256": sys.argv[13],
    "resource_report_sha256": sys.argv[14],
    "resource_demangled_sha256": sys.argv[15],
    "gpu_uuid": sys.argv[4],
    "node": sys.argv[5],
    "gpu_name": sys.argv[6],
    "cuda_sha256": sys.argv[7],
    "source_manifest_sha256": sys.argv[8],
    "source_manifest_scope": "selected_high_risk_files",
    "input_manifest_sha256": sys.argv[9],
    "qualified_gpu_gate_root": sys.argv[10],
    "execution_order": [
        "canonical_serial_1",
        "atomic_serial_1",
        "atomic_multistream_1",
        "canonical_multistream_1",
        "canonical_multistream_2",
        "atomic_multistream_2",
        "atomic_serial_2",
        "canonical_serial_2",
    ],
    "coarse_multistream_workers": [0, 0, 8, 8, 8, 8, 0, 0],
    "single_lane_canonical": [False] * 8,
    "native_atomic_reduction": [False, True, True, False, False, True, True, False],
    "focused_test_node": "tests/unit/test_cuda_relion_fine_diff2.py::test_relion_coarse_vdam_multistream_atomic_stays_in_lane_envelope",
    "raw_image_cache": "off",
    "exact_local_bucket_radix": 4,
    "exact_local_physical_order_chunk_size": 0,
    "acceptance_rule": {
        "mathematically_equivalent_reduction_required": True,
        "numeric_delta_within_repeat_envelope_required": True,
        "nondirectional_signed_delta_required": True,
        "repeat_growth_forbidden": True,
        "discrete_decisions_preserved_required": True,
        "basin_preserved_required": True,
        "material_end_to_end_runtime_win_required": True,
    },
    "science_promotion_allowed": False,
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${REPO_ROOT}" rev-parse 'HEAD^{tree}')" = "${repo_tree}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1)"
test "$(sha256sum "${CUDA_BINARY}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
test "$(sha256sum "${RELION_BIND_BINARY}" | awk '{print $1}')" = \
  "${EXPECTED_RELION_BIND_SHA256}"
test "$(sha256sum "${PIXI_PY}" | awk '{print $1}')" = \
  "${EXPECTED_INTERPRETER_SHA256}"
sha256sum -c "${PROVENANCE}/input_manifest.sha256" > /dev/null
(
  cd "${REPO_ROOT}"
  sha256sum "${SOURCE_FILES[@]}"
) > "${PROVENANCE}/source_manifest.final.sha256"
cmp "${PROVENANCE}/source_manifest.sha256" "${PROVENANCE}/source_manifest.final.sha256"
vdam_verify_selected_gpu "${selected_gpu_uuid}"
touch "${ROOT}/COMPLETED"
