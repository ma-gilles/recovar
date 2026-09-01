#!/usr/bin/env bash
#SBATCH --job-name=vdam-cache-abba
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
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-cache-abba-%j.out

set -euo pipefail
unset PYTHONOPTIMIZE

: "${VDAM_RAW_CACHE_ABBA_ROOT:?set a new disposable output root}"
: "${EXPECTED_REPO_HEAD:?pin the committed harness head}"
: "${EXPECTED_REPO_TREE:?pin the committed harness tree}"
: "${EXPECTED_SOURCE_MANIFEST_SHA256:?pin the benchmark source manifest}"

readonly SCIENCE_BASE_HEAD=77e09c292e438a265a6d157414b2a0fe525710e6
readonly SCIENCE_BASE_TREE=384448a3ce6639e571cad53d2b82c1b18df1979a
readonly QUALIFIED_GPU_GATE_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_coarse_prehalf_h100_13315513
readonly QUALIFIED_GATE_SHA256SUMS_SHA256=9ab443af3a90f63bc0fd6eeac90f9c15f84d7f667c6c19171682a11f0c168cc8
readonly QUALIFIED_CUDA=${QUALIFIED_GPU_GATE_ROOT}/build/libcuda_backproject.so
readonly EXPECTED_CUDA_SHA256=2af7bf1e4cbdc10705948d907c087d1662db612fe8d57362f1390033ac6c047b
readonly EXPECTED_INTERPRETER_SHA256=48556a44c0dd1570866beb838e6fcbea771bce93d413acd4197bb2f254b72d23
readonly RELION_BIND_SOURCE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_coarse_atomic_multistream_crossed_5a7ff8ad0_h21g4_20260901/runtime/relion_bind/_relion_bind_core.cpython-311-x86_64-linux-gnu.so
readonly EXPECTED_RELION_BIND_SHA256=9bbb1fb0ce6fa7ac816598ec521453515d163221642b916e5715bb2850798980
readonly NSYS=/opt/nvidia/nsight-systems/2025.3.2/target-linux-x64/nsys
readonly EXPECTED_NSYS_SHA256=9b32b4e9beee469bc8c26640db228b04583234505c315e23f7e622efd61a68ab
readonly CUDA_TOOLKIT=/usr/local/cuda-13.1
readonly RELION_CUDA_TOOLKIT=/usr/local/cuda-12.6
readonly CUSPARSE_LIBRARY=${RELION_CUDA_TOOLKIT}/lib64/libcusparse.so.12
readonly EXPECTED_CUSPARSE_SHA256=58ffc54edb1d007f56a1718aaadcb30f45bbf662f43515920ea8ff094304bdbf
readonly RELION_RUNTIME=/scratch/gpfs/GILLES/mg6942/recovar_dev/relion_k4_100k_dispatchv2_20260717
readonly MPI_ROOT=/usr/local/openmpi/cuda-12.6/4.1.6/gcc
readonly TARGET_NODE=della-h21g4
readonly TARGET_GPU_UUID=GPU-099c0d77-bb85-f2e9-f628-148b733c9176
readonly GF46_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_full_expansion_v3_984637b7d_87274be_20260826/vdam-gf46/repeat-01/vdam-gf46
readonly CHECKPOINT=${GF46_ROOT}/relion/run_it180_optimiser.star
readonly DATA_STAR=${GF46_ROOT}/relion/run_it180_data.star
readonly DATA_DIR=${GF46_ROOT}/data
readonly PARTICLE_STACK=${DATA_DIR}/particles.128.mrcs
readonly EXPECTED_INPUT_MANIFEST_SHA256=de224471a690d1faaae4067217dbcc90b632269d62b0b3372b20aafa69157d91
readonly EXPECTED_PARTICLE_STACK_SHA256=804af933bd315f41f0159f62e93867cf852d70cb29f2f27a525fb2fc3eb68ad9
readonly RAW_IMAGE_CACHE_MAX_GB=16
readonly RAW_IMAGE_CACHE_EXPECTED_BYTES=196608000
readonly EXPECTED_CACHE_MAPPING_SHA256=e02b912acbce6b05645063e4f0d43fdcb294ed3680917de7ac7cf71cfa9974b2
readonly EXPECTED_CACHE_LEAF_SELECTION_SHA256=e8c9ceaf5aacc63c25b4cdd8542592f9d58aff50e3e8fc6c55591d3d8f596562
readonly CHECKPOINT_ITERATION=180
readonly PROFILED_ITERATION=181
readonly NR_ITER_SCHEDULE=200

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
REPO_ROOT=$(realpath -e -- "${REPO_ROOT}")
PIXI_PY=${REPO_ROOT}/.pixi/envs/default/bin/python
ROOT=$(realpath -m -- "${VDAM_RAW_CACHE_ABBA_ROOT}")
PROVENANCE=${ROOT}/provenance
RUNTIME=${ROOT}/runtime
NSIGHT=${ROOT}/nsight
ANALYSIS=${ROOT}/analysis
CUDA_BINARY=${RUNTIME}/cuda/libcuda_backproject.so
RELION_BIND_DIR=${RUNTIME}/relion_bind
RELION_BIND_BINARY=${RELION_BIND_DIR}/$(basename "${RELION_BIND_SOURCE}")

SOURCE_FILES=(
  recovar/cuda/Makefile
  recovar/cuda/cuda_backproject.cu
  recovar/cuda/relion_coarse_diff2_projector_body.inc
  recovar/cuda_backproject.py
  recovar/data_io/image_loader.py
  recovar/data_io/staging.py
  recovar/em/dense_single_volume/batch_planning.py
  recovar/em/dense_single_volume/helpers/significance.py
  recovar/em/dense_single_volume/k_class.py
  recovar/em/initial_model/dense_adapter.py
  recovar/em/initial_model/driver.py
  recovar/em/initial_model/iteration_loop.py
  recovar/em/initial_model/m_step.py
  recovar/em/initial_model/schedules.py
  recovar/commands/initial_model.py
  scripts/run_ab_initio.py
  scripts/run_vdam_late_iteration_profile.py
  scripts/probe_vdam_raw_cache_memory.py
  scripts/summarize_vdam_nsys_sqlite.py
  scripts/analyze_vdam_raw_cache_abba.py
  scripts/run_vdam_raw_cache_abba.sbatch
  scripts/vdam_gpu_selection.sh
  tests/unit/test_image_loader.py
  tests/unit/initial_model/test_iteration_loop.py
  tests/unit/initial_model/test_vdam_late_iteration_profile.py
  tests/unit/initial_model/test_vdam_raw_cache_memory_probe.py
  tests/unit/initial_model/test_vdam_raw_cache_abba_analyzer.py
  tests/unit/initial_model/test_vdam_raw_cache_abba_runner.py
  pixi.lock
  pixi.toml
  pyproject.toml
)

BENCHMARK_DIFF_ALLOWLIST=(
  recovar/data_io/image_loader.py
  recovar/em/initial_model/iteration_loop.py
  scripts/analyze_vdam_raw_cache_abba.py
  scripts/probe_vdam_raw_cache_memory.py
  scripts/run_vdam_late_iteration_profile.py
  scripts/run_vdam_raw_cache_abba.sbatch
  tests/unit/test_image_loader.py
  tests/unit/initial_model/test_iteration_loop.py
  tests/unit/initial_model/test_vdam_late_iteration_profile.py
  tests/unit/initial_model/test_vdam_raw_cache_memory_probe.py
  tests/unit/initial_model/test_vdam_raw_cache_abba_analyzer.py
  tests/unit/initial_model/test_vdam_raw_cache_abba_runner.py
)

INPUT_FILES=(
  "${GF46_ROOT}/relion/run_it180_optimiser.star"
  "${GF46_ROOT}/relion/run_it180_model.star"
  "${GF46_ROOT}/relion/run_it180_data.star"
  "${GF46_ROOT}/relion/run_it180_sampling.star"
  "${GF46_ROOT}/relion/run_it180_class001.mrc"
  "${GF46_ROOT}/relion/run_it180_1moment001.mrc"
  "${GF46_ROOT}/relion/run_it180_1moment002.mrc"
  "${GF46_ROOT}/relion/run_it180_2moment001.mrc"
  "${PARTICLE_STACK}"
)

on_error() {
  local status=$?
  local message="VDAM raw-cache ABBA failed: status=${status} line=${BASH_LINENO[0]} command=${BASH_COMMAND}"
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

validate_benchmark_overlay() {
  local candidate
  local allowed
  local matched
  local -a changed=()
  mapfile -t changed < <(
    git -C "${REPO_ROOT}" diff --name-only --diff-filter=ACDMRTUXB \
      "${SCIENCE_BASE_HEAD}..${EXPECTED_REPO_HEAD}" | LC_ALL=C sort -u
  )
  test "${#changed[@]}" -gt 0
  for candidate in "${changed[@]}"; do
    matched=0
    for allowed in "${BENCHMARK_DIFF_ALLOWLIST[@]}"; do
      if [[ "${candidate}" == "${allowed}" ]]; then
        matched=1
        break
      fi
    done
    if [[ "${matched}" != 1 ]]; then
      echo "science-base overlay contains an unapproved benchmark path: ${candidate}" >&2
      return 2
    fi
  done
  printf '%s\n' "${changed[@]}"
}

verify_gate_and_runtime_inputs() {
  test -e "${QUALIFIED_GPU_GATE_ROOT}/COMPLETED"
  test -s "${QUALIFIED_GPU_GATE_ROOT}/SHA256SUMS"
  test "$(sha256sum "${QUALIFIED_GPU_GATE_ROOT}/SHA256SUMS" | awk '{print $1}')" = \
    "${QUALIFIED_GATE_SHA256SUMS_SHA256}"
  (
    cd "${QUALIFIED_GPU_GATE_ROOT}"
    sha256sum -c SHA256SUMS
  ) > /dev/null
  test "$(< "${QUALIFIED_GPU_GATE_ROOT}/provenance/repo_head.txt")" = "${SCIENCE_BASE_HEAD}"
  test "$(< "${QUALIFIED_GPU_GATE_ROOT}/provenance/repo_tree.txt")" = "${SCIENCE_BASE_TREE}"
  test "$(< "${QUALIFIED_GPU_GATE_ROOT}/provenance/node.txt")" = "${TARGET_NODE}"
  test "$(< "${QUALIFIED_GPU_GATE_ROOT}/provenance/selected_gpu_uuid.txt")" = "${TARGET_GPU_UUID}"
  test -s "${QUALIFIED_CUDA}"
  test "$(sha256sum "${QUALIFIED_CUDA}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
  test -s "${RELION_BIND_SOURCE}"
  test "$(sha256sum "${RELION_BIND_SOURCE}" | awk '{print $1}')" = \
    "${EXPECTED_RELION_BIND_SHA256}"
  test -x "${PIXI_PY}"
  test "$(sha256sum "${PIXI_PY}" | awk '{print $1}')" = \
    "${EXPECTED_INTERPRETER_SHA256}"
  test -x "${NSYS}"
  test "$(sha256sum "${NSYS}" | awk '{print $1}')" = "${EXPECTED_NSYS_SHA256}"
  test -s "${CUSPARSE_LIBRARY}"
  test "$(sha256sum "${CUSPARSE_LIBRARY}" | awk '{print $1}')" = \
    "${EXPECTED_CUSPARSE_SHA256}"
}

verify_frozen_inputs() {
  sha256sum -c "${PROVENANCE}/input_manifest.sha256" > /dev/null
  test "$(sha256sum "${PARTICLE_STACK}" | awk '{print $1}')" = \
    "${EXPECTED_PARTICLE_STACK_SHA256}"
}

[[ "${EXPECTED_REPO_HEAD}" =~ ^[0-9a-f]{40}$ ]]
[[ "${EXPECTED_REPO_TREE}" =~ ^[0-9a-f]{40}$ ]]
[[ "${EXPECTED_SOURCE_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]]
[[ "${ROOT}" = /* ]]
test ! -e "${ROOT}"
test -d "$(dirname "${ROOT}")"
case "${ROOT}/" in
  "${REPO_ROOT}/"*)
    echo "result root must be outside the repository: ${ROOT}" >&2
    exit 2
    ;;
esac
test "$(git -C "${REPO_ROOT}" rev-parse --show-toplevel)" = "${REPO_ROOT}"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${REPO_ROOT}" rev-parse 'HEAD^{tree}')" = "${EXPECTED_REPO_TREE}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=all)"
test "$(git -C "${REPO_ROOT}" rev-parse "${SCIENCE_BASE_HEAD}^{tree}")" = \
  "${SCIENCE_BASE_TREE}"
git -C "${REPO_ROOT}" merge-base --is-ancestor "${SCIENCE_BASE_HEAD}" "${EXPECTED_REPO_HEAD}"
benchmark_overlay=$(validate_benchmark_overlay)
for source_file in "${SOURCE_FILES[@]}"; do
  test -s "${REPO_ROOT}/${source_file}"
done
for input_file in "${INPUT_FILES[@]}"; do
  test -s "${input_file}"
done
verify_gate_and_runtime_inputs
test "${SLURMD_NODENAME:-$(hostname -s)}" = "${TARGET_NODE}"

initial_allocation_spec=${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}}
source "${REPO_ROOT}/scripts/vdam_gpu_selection.sh"
vdam_assert_target_gpu_allocated "${TARGET_GPU_UUID}" "${initial_allocation_spec}"
vdam_select_target_gpu "${TARGET_GPU_UUID}" 0
selected_gpu_uuid=${VDAM_SELECTED_GPU_UUID}
test "${selected_gpu_uuid}" = "${TARGET_GPU_UUID}"
gpu_name=$(nvidia-smi -i "${selected_gpu_uuid}" --query-gpu=name --format=csv,noheader | sed -n '1p' | sed 's/[[:space:]]*$//')
[[ "${gpu_name}" == *H100* ]]

mkdir "${ROOT}"
mkdir -p "${PROVENANCE}" "${RUNTIME}/cuda" "${RELION_BIND_DIR}" "${NSIGHT}" "${ANALYSIS}"
touch "${ROOT}/SAFE_TO_DELETE"
cp --reflink=auto "${QUALIFIED_CUDA}" "${CUDA_BINARY}"
cp --reflink=auto "${RELION_BIND_SOURCE}" "${RELION_BIND_BINARY}"
cp --reflink=auto "${QUALIFIED_GPU_GATE_ROOT}/SHA256SUMS" \
  "${PROVENANCE}/qualified_gate.SHA256SUMS"
cp --reflink=auto "${QUALIFIED_GPU_GATE_ROOT}/SHA256SUMS.sha256" \
  "${PROVENANCE}/qualified_gate.SHA256SUMS.sha256"
test "$(sha256sum "${CUDA_BINARY}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
test "$(sha256sum "${RELION_BIND_BINARY}" | awk '{print $1}')" = \
  "${EXPECTED_RELION_BIND_SHA256}"

sha256sum "${INPUT_FILES[@]}" > "${PROVENANCE}/input_manifest.sha256"
input_manifest_sha256=$(sha256sum "${PROVENANCE}/input_manifest.sha256" | awk '{print $1}')
test "${input_manifest_sha256}" = "${EXPECTED_INPUT_MANIFEST_SHA256}"
write_source_manifest > "${PROVENANCE}/source_manifest.sha256"
source_manifest_sha256=$(sha256sum "${PROVENANCE}/source_manifest.sha256" | awk '{print $1}')
test "${source_manifest_sha256}" = "${EXPECTED_SOURCE_MANIFEST_SHA256}"
verify_frozen_inputs

git -C "${REPO_ROOT}" rev-parse HEAD > "${PROVENANCE}/repo_head.txt"
git -C "${REPO_ROOT}" rev-parse 'HEAD^{tree}' > "${PROVENANCE}/repo_tree.txt"
git -C "${REPO_ROOT}" symbolic-ref --short HEAD > "${PROVENANCE}/repo_branch.txt"
git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=all > "${PROVENANCE}/repo_status.txt"
git -C "${REPO_ROOT}" diff --binary --no-ext-diff HEAD -- | sha256sum \
  > "${PROVENANCE}/repo_diff.sha256"
printf '%s\n' "${SCIENCE_BASE_HEAD}" > "${PROVENANCE}/science_base_head.txt"
printf '%s\n' "${SCIENCE_BASE_TREE}" > "${PROVENANCE}/science_base_tree.txt"
printf '%s\n' "${benchmark_overlay}" > "${PROVENANCE}/benchmark_overlay.txt"
printf '%s\n' "${SLURM_JOB_ID}" > "${PROVENANCE}/slurm_job_id.txt"
printf '%s\n' "${TARGET_NODE}" > "${PROVENANCE}/node.txt"
printf '%s\n' "${selected_gpu_uuid}" > "${PROVENANCE}/selected_gpu_uuid.txt"
printf '%s\n' "${gpu_name}" > "${PROVENANCE}/gpu_name.txt"
printf '%s\n' "${VDAM_ALLOCATED_GPU_UUIDS_CSV}" > "${PROVENANCE}/allocated_gpu_uuids.csv"
printf '%s\n' "${VDAM_VISIBLE_GPU_UUIDS_CSV}" > "${PROVENANCE}/visible_gpu_uuids.csv"
sha256sum "${PIXI_PY}" > "${PROVENANCE}/interpreter.sha256"
sha256sum "${CUDA_BINARY}" > "${PROVENANCE}/qualified_cuda.sha256"
sha256sum "${RELION_BIND_BINARY}" > "${PROVENANCE}/relion_bind.sha256"
sha256sum "${NSYS}" > "${PROVENANCE}/nsys.sha256"
sha256sum "${CUSPARSE_LIBRARY}" > "${PROVENANCE}/cusparse.sha256"
nvidia-smi -i "${selected_gpu_uuid}" -q > "${PROVENANCE}/nvidia_smi.txt"
"${NSYS}" --version > "${PROVENANCE}/nsys_version.txt"
scontrol show job "${SLURM_JOB_ID}" > "${PROVENANCE}/slurm_job.txt"
env | LC_ALL=C sort > "${PROVENANCE}/submission_environment.txt"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV JAX_PLATFORM_NAME
unset RECOVAR_EM_RAW_IMAGE_CACHE RECOVAR_EM_RAW_IMAGE_CACHE_MAX_GB
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${selected_gpu_uuid}
export RECOVAR_EXPECTED_REPO_ROOT=${REPO_ROOT}
export RECOVAR_CUDA_LIB=${CUDA_BINARY}
export RECOVAR_RELION_BIND_BUILD_DIR=${RELION_BIND_DIR}
export RECOVAR_REQUIRE_CUSTOM_CUDA_FOR_TESTS=1
export RECOVAR_ENABLE_CUSTOM_CUDA=1
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_LOG_COMPILES=1
export TMPDIR=${RUNTIME}/tmp
export RECOVAR_CACHE_DIR=
export PIXI_HOME=${RUNTIME}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/rattler_cache
export PATH=${CUDA_TOOLKIT}/bin:${PATH}
export LD_LIBRARY_PATH=${RELION_RUNTIME}/system_dev_overlay/lib:${MPI_ROOT}/lib64:${RELION_CUDA_TOOLKIT}/lib64:${RELION_CUDA_TOOLKIT}/targets/x86_64-linux/lib:${CUDA_TOOLKIT}/lib64:${CUDA_TOOLKIT}/targets/x86_64-linux/lib
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}"

(
  cd "${REPO_ROOT}"
  env LD_PRELOAD="${CUSPARSE_LIBRARY}" "${PIXI_PY}" - <<'PY'
import os
import sys
from pathlib import Path

import jax
import recovar
import recovar.cuda_backproject as cuda_backproject
from recovar.em.dense_single_volume.batch_planning import _em_raw_image_cache_mode
from recovar.relion_bind import _relion_bind_core as relion_bind_core
from recovar.utils.parity_provenance import assert_parity_ancestors

repo = Path(os.environ["RECOVAR_EXPECTED_REPO_ROOT"]).resolve()
cuda = Path(os.environ["RECOVAR_CUDA_LIB"]).resolve()
if sys.flags.optimize != 0:
    raise RuntimeError("runtime validation requires Python assertions")
assert str(Path(recovar.__file__).resolve()).startswith(str(repo) + "/")
assert ".pixi/envs/default/" in str(Path(jax.__file__).resolve())
assert jax.default_backend() == "gpu"
assert len(jax.devices("gpu")) == 1
assert "H100" in str(jax.devices("gpu")[0].device_kind)
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
assert Path(cuda_backproject._loaded_lib_path).resolve() == cuda
assert callable(getattr(relion_bind_core, "vdam_rnd_unif_sequence", None))
assert _em_raw_image_cache_mode() == "auto"
assert os.environ.get("RECOVAR_CACHE_DIR") == ""
assert_parity_ancestors()
print("VDAM raw-cache ABBA runtime gate passed", jax.devices(), recovar.__file__)
PY
) > "${PROVENANCE}/runtime_gate.stdout" 2> "${PROVENANCE}/runtime_gate.stderr"
vdam_verify_selected_gpu "${selected_gpu_uuid}"

RUN_LABELS=(cache_off_1 cache_auto_1 cache_auto_2 cache_off_2 cache_auto_3 cache_off_3 cache_off_4 cache_auto_4)
RUN_MODES=(off auto auto off auto off off auto)
test "${#RUN_LABELS[@]}" -eq "${#RUN_MODES[@]}"
printf 'order\tlabel\traw_image_cache_mode\traw_image_cache_max_gb\tworkers\tsingle_lane_canonical\tnative_atomic_reduction\tnsys_base\n' \
  > "${PROVENANCE}/execution_order.tsv"

for index in "${!RUN_LABELS[@]}"; do
  label=${RUN_LABELS[index]}
  mode=${RUN_MODES[index]}
  run_root=${ROOT}/runs/${label}
  cache_root=${run_root}/jax_cache
  nsys_base=${NSIGHT}/${label}_it181_warm
  mkdir -p "${run_root}" "${cache_root}"
  touch "${cache_root}/SAFE_TO_DELETE"
  printf '%d\t%s\t%s\t%s\t8\t0\t1\t%s\n' \
    "$((index + 1))" "${label}" "${mode}" "${RAW_IMAGE_CACHE_MAX_GB}" "${nsys_base}" \
    >> "${PROVENANCE}/execution_order.tsv"
  verify_frozen_inputs
  command=(
    env
    -u RECOVAR_RELION_COARSE_CANONICAL_REDUCTION
    "LD_PRELOAD=${CUSPARSE_LIBRARY}"
    "RECOVAR_CACHE_DIR="
    "RECOVAR_EM_RAW_IMAGE_CACHE=${mode}"
    "RECOVAR_EM_RAW_IMAGE_CACHE_MAX_GB=${RAW_IMAGE_CACHE_MAX_GB}"
    RECOVAR_K1_COARSE_MULTISTREAM_WORKERS=8
    RECOVAR_K1_COARSE_SINGLE_LANE_CANONICAL=0
    RECOVAR_K1_COARSE_NATIVE_ATOMIC_REDUCTION=1
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
    --audit-raw-image-cache
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
  test -s "${run_root}/profile/warm/run_it181_recovar_meta.json"
  test ! -e "${run_root}/profile/warm/run_it182_recovar_meta.json"

  "${PIXI_PY}" - "${run_root}/profile/profile_summary.json" \
    "${run_root}/cache_admission.json" "${label}" "${mode}" \
    "${RAW_IMAGE_CACHE_MAX_GB}" "${RAW_IMAGE_CACHE_EXPECTED_BYTES}" \
    "${PARTICLE_STACK}" "${EXPECTED_CACHE_MAPPING_SHA256}" \
    "${EXPECTED_CACHE_LEAF_SELECTION_SHA256}" <<'PY'
import json
import sys
from pathlib import Path

if sys.flags.optimize != 0:
    raise RuntimeError("cache-admission validation requires Python assertions")

(
    summary_path,
    output_path,
    label,
    mode,
    max_gb,
    expected_bytes,
    particle_stack,
    mapping_sha256,
    leaf_selection_sha256,
) = sys.argv[1:]
summary = json.loads(Path(summary_path).read_text())
max_gb = float(max_gb)
expected_bytes = int(expected_bytes)
assert summary["raw_image_cache_audit_enabled"] is True, summary
phases = {}
for phase in ("cold", "warm"):
    audit = summary[phase]["raw_image_cache_audit"]
    assert audit["mode"] == mode, audit
    assert float(audit["max_gb"]) == max_gb, audit
    events = audit["load_all_events"]
    if mode == "off":
        assert events == [], events
    else:
        assert mode == "auto", mode
        assert len(events) == 1, events
        event = events[0]
        assert event["loader_type"] == "recovar.data_io.image_loader.StarLoader", event
        assert event["num_images"] == 3000, event
        assert event["image_size"] == 128, event
        assert event["dtype"] == "<f4", event
        assert event["estimated_bytes"] == expected_bytes, event
        assert event["cached_before"] is False, event
        assert event["cached_after"] is True, event
        assert event["cached_nbytes"] == expected_bytes, event
        assert event["cached_shape"] == [3000, 128, 128], event
        assert event["cached_dtype"] == "<f4", event
        assert event["cached_c_contiguous"] is True, event
        assert event["cached_writeable"] is True, event
        assert event["loader_topology"] == {
            "mapped_rows": 3000,
            "mapped_files": [particle_stack],
            "mapped_file_count": 1,
            "mapping_unique_index_count": 3000,
            "mapping_min_index": 0,
            "mapping_max_index": 2999,
            "mapping_is_unique": True,
            "mapping_is_contiguous_set": True,
            "mapping_is_strictly_ascending": False,
            "mapping_mrc_indices_sha256": mapping_sha256,
            "leaf_loader_count": 1,
            "leaf_loaders": [
                {
                    "path": particle_stack,
                    "io_path": particle_stack,
                    "loader_type": "recovar.data_io.image_loader.MRCLoader",
                    "num_images": 3000,
                    "image_size": 128,
                    "dtype": "<f4",
                    "selection_indices_sha256": leaf_selection_sha256,
                }
            ],
            "leaf_cached_before": [False],
            "leaf_cached_after": [False],
        }, event
        assert float(event["elapsed_s"]) >= 0.0, event
        assert (
            event["current_rss_after_bytes"] - event["current_rss_before_bytes"]
            == event["current_rss_delta_bytes"]
        ), event
        assert (
            event["high_water_rss_after_bytes"] - event["high_water_rss_before_bytes"]
            == event["high_water_rss_delta_bytes"]
        ), event
        assert event["high_water_rss_delta_bytes"] >= 0, event
        assert event["high_water_rss_before_bytes"] >= event["current_rss_before_bytes"], event
        assert event["high_water_rss_after_bytes"] >= event["current_rss_after_bytes"], event
        for key in (
            "current_rss_before_bytes",
            "current_rss_after_bytes",
            "high_water_rss_before_bytes",
            "high_water_rss_after_bytes",
        ):
            assert event[key] >= 0, event
    phases[phase] = {
        "load_all_count": len(events),
        "load_all_events": events,
        "admitted": mode == "auto" and len(events) == 1,
    }
payload = {
    "schema": "recovar.vdam_raw_cache_admission.v2",
    "label": label,
    "mode": mode,
    "max_gb": max_gb,
    "expected_bytes": expected_bytes,
    "phases": phases,
}
Path(output_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

  sqlite=${NSIGHT}/${label}.sqlite
  "${NSYS}" export --type=sqlite --force-overwrite=true --output="${sqlite}" \
    "${nsys_base}.nsys-rep"
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
  test "$(sha256sum "${CUDA_BINARY}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
  test "$(sha256sum "${RELION_BIND_BINARY}" | awk '{print $1}')" = \
    "${EXPECTED_RELION_BIND_SHA256}"
  test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
  test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=all)"
  vdam_verify_selected_gpu "${selected_gpu_uuid}"
done

touch "${ROOT}/ARMS_COMPLETED"

memory_probe_command=(
  env
  JAX_PLATFORMS=cpu
  JAX_PLATFORM_NAME=cpu
  RECOVAR_CACHE_DIR=
  "${PIXI_PY}" -m scripts.probe_vdam_raw_cache_memory
  --input-star "${DATA_STAR}"
  --data-dir "${DATA_DIR}"
  --output-json "${PROVENANCE}/raw_cache_memory_probe.json"
  --comparison-batch-size 500
)
printf '%q ' "${memory_probe_command[@]}" \
  > "${PROVENANCE}/raw_cache_memory_probe_command.sh"
printf '\n' >> "${PROVENANCE}/raw_cache_memory_probe_command.sh"
(
  cd "${REPO_ROOT}"
  "${memory_probe_command[@]}"
) > "${PROVENANCE}/raw_cache_memory_probe.stdout" \
  2> "${PROVENANCE}/raw_cache_memory_probe.stderr"
test -s "${PROVENANCE}/raw_cache_memory_probe.json"
verify_frozen_inputs

"${PIXI_PY}" - "${PROVENANCE}/run.json" "${SLURM_JOB_ID}" \
  "${EXPECTED_REPO_HEAD}" "${EXPECTED_REPO_TREE}" "${source_manifest_sha256}" \
  "${input_manifest_sha256}" "${selected_gpu_uuid}" "${gpu_name}" <<'PY'
import json
import sys
from pathlib import Path

payload = {
    "schema": "recovar.vdam_raw_cache_abba.v1",
    "classification": "diagnostic_performance_only",
    "job_id": sys.argv[2],
    "git_head": sys.argv[3],
    "git_tree": sys.argv[4],
    "science_base_head": "77e09c292e438a265a6d157414b2a0fe525710e6",
    "science_base_tree": "384448a3ce6639e571cad53d2b82c1b18df1979a",
    "source_manifest_sha256": sys.argv[5],
    "source_manifest_scope": "selected_high_risk_files",
    "input_manifest_sha256": sys.argv[6],
    "particle_stack_sha256": "804af933bd315f41f0159f62e93867cf852d70cb29f2f27a525fb2fc3eb68ad9",
    "gpu_uuid": sys.argv[7],
    "gpu_name": sys.argv[8],
    "node": "della-h21g4",
    "qualified_gpu_gate_root": "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_coarse_prehalf_h100_13315513",
    "qualified_gate_sha256sums_sha256": "9ab443af3a90f63bc0fd6eeac90f9c15f84d7f667c6c19171682a11f0c168cc8",
    "cuda_sha256": "2af7bf1e4cbdc10705948d907c087d1662db612fe8d57362f1390033ac6c047b",
    "relion_bind_sha256": "9bbb1fb0ce6fa7ac816598ec521453515d163221642b916e5715bb2850798980",
    "interpreter_sha256": "48556a44c0dd1570866beb838e6fcbea771bce93d413acd4197bb2f254b72d23",
    "nsys_sha256": "9b32b4e9beee469bc8c26640db228b04583234505c315e23f7e622efd61a68ab",
    "cusparse_sha256": "58ffc54edb1d007f56a1718aaadcb30f45bbf662f43515920ea8ff094304bdbf",
    "execution_order": [
        "cache_off_1", "cache_auto_1", "cache_auto_2", "cache_off_2",
        "cache_auto_3", "cache_off_3", "cache_off_4", "cache_auto_4",
    ],
    "raw_image_cache_modes": ["off", "auto", "auto", "off", "auto", "off", "off", "auto"],
    "raw_image_cache_max_gb": 16.0,
    "raw_image_cache_expected_bytes": 196608000,
    "raw_image_cache_force_used": False,
    "checkpoint_iteration": 180,
    "profiled_iteration": 181,
    "nr_iter_schedule": 200,
    "random_seed": 29,
    "image_batch_size": 500,
    "coarse_multistream_workers": 8,
    "single_lane_canonical": False,
    "native_atomic_reduction": True,
    "exact_local_bucket_radix": 4,
    "exact_local_physical_order_chunk_size": 0,
    "science_promotion_allowed": False,
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

# A valid diagnostic NO_GO is analyzer exit 1.  Disable both errexit and the
# ERR trap while collecting that status; ``set +e`` alone does not suppress an
# ERR trap.  Restore fail-closed behavior immediately afterwards.
trap - ERR
set +e
(
  cd "${REPO_ROOT}"
  env JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu \
    "${PIXI_PY}" -m scripts.analyze_vdam_raw_cache_abba \
    --root "${ROOT}" --repo "${REPO_ROOT}" \
    --output-json "${ANALYSIS}/report.json" \
    --output-markdown "${ANALYSIS}/report.md"
) > "${ANALYSIS}/analyzer.stdout" 2> "${ANALYSIS}/analyzer.stderr"
analyzer_status=$?
set -e
trap on_error ERR
printf '%s\n' "${analyzer_status}" > "${ANALYSIS}/analyzer_status.txt"
if [[ "${analyzer_status}" != 0 && "${analyzer_status}" != 1 ]]; then
  analyzer_message="raw-cache ABBA analyzer failed with setup/runtime status ${analyzer_status}"
  printf '%s\n' "${analyzer_message}" >&2
  printf '%s\n' "${analyzer_message}" > "${PROVENANCE}/failure.txt"
  exit "${analyzer_status}"
fi
test -s "${ANALYSIS}/report.json"
test -s "${ANALYSIS}/report.md"

test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${REPO_ROOT}" rev-parse 'HEAD^{tree}')" = "${EXPECTED_REPO_TREE}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=all)"
test "$(validate_benchmark_overlay)" = "${benchmark_overlay}"
write_source_manifest > "${PROVENANCE}/source_manifest.final.sha256"
cmp "${PROVENANCE}/source_manifest.sha256" "${PROVENANCE}/source_manifest.final.sha256"
verify_frozen_inputs
verify_gate_and_runtime_inputs
test "$(sha256sum "${CUDA_BINARY}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
test "$(sha256sum "${RELION_BIND_BINARY}" | awk '{print $1}')" = \
  "${EXPECTED_RELION_BIND_SHA256}"
vdam_verify_selected_gpu "${selected_gpu_uuid}"

touch "${ROOT}/COMPLETED"
(
  cd "${ROOT}"
  find . -type f ! -name SHA256SUMS ! -name SHA256SUMS.sha256 -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum
) > "${ROOT}/SHA256SUMS"
sha256sum "${ROOT}/SHA256SUMS" > "${ROOT}/SHA256SUMS.sha256"
find "${ROOT}" -type f -exec chmod a-w {} +
find "${ROOT}" -depth -type d -exec chmod a-w {} +

printf 'VDAM raw-cache ABBA complete: %s (analyzer status %s)\n' \
  "${ROOT}" "${analyzer_status}"
if [[ "${analyzer_status}" == 1 ]]; then
  trap - ERR
  exit 1
fi
