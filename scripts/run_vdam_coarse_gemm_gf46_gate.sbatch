#!/usr/bin/env bash
#SBATCH --job-name=vdam-gemm-gf46
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
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-gemm-gf46-%j.out

set -euo pipefail
unset PYTHONOPTIMIZE

: "${VDAM_COARSE_GEMM_GF46_ROOT:?set a new disposable output root}"
: "${EXPECTED_REPO_HEAD:?pin the committed harness head}"
: "${EXPECTED_REPO_TREE:?pin the committed harness tree}"
: "${EXPECTED_SOURCE_MANIFEST_SHA256:?pin the benchmark source manifest}"

readonly QUALIFIED_GPU_GATE_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_coarse_prehalf_h100_13315513
readonly QUALIFIED_GATE_SHA256SUMS_SHA256=9ab443af3a90f63bc0fd6eeac90f9c15f84d7f667c6c19171682a11f0c168cc8
readonly QUALIFIED_CUDA=${QUALIFIED_GPU_GATE_ROOT}/build/libcuda_backproject.so
readonly EXPECTED_CUDA_SHA256=2af7bf1e4cbdc10705948d907c087d1662db612fe8d57362f1390033ac6c047b
readonly EXPECTED_INTERPRETER_SHA256=48556a44c0dd1570866beb838e6fcbea771bce93d413acd4197bb2f254b72d23
readonly RELION_BIND_SOURCE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_coarse_atomic_multistream_crossed_5a7ff8ad0_h21g4_20260901/runtime/relion_bind/_relion_bind_core.cpython-311-x86_64-linux-gnu.so
readonly EXPECTED_RELION_BIND_SHA256=9bbb1fb0ce6fa7ac816598ec521453515d163221642b916e5715bb2850798980
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
readonly RAW_TARGETS_CSV=1,2160
readonly EXPECTED_SELECTED_IDS_INT64_SHA256=c0199226ec7aa92f74a2fd66660e597fe44d73155db20f026b278840992a90be
readonly CHECKPOINT_ITERATION=180
readonly PROFILED_ITERATION=181
readonly NR_ITER_SCHEDULE=200
readonly MATERIAL_END_TO_END_RATIO=0.90

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
REPO_ROOT=$(realpath -e -- "${REPO_ROOT}")
PIXI_PY=${PIXI_PY_OVERRIDE:-${REPO_ROOT}/.pixi/envs/default/bin/python}
ROOT=$(realpath -m -- "${VDAM_COARSE_GEMM_GF46_ROOT}")
PROVENANCE=${ROOT}/provenance
RUNTIME=${ROOT}/runtime
ANALYSIS=${ROOT}/analysis
RAW_ROOT=${ROOT}/raw_score_diagnostic
RAW_DIAGNOSTIC_DIR=${RAW_ROOT}/artifacts
RAW_OUTPUT=${RAW_ROOT}/output
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
  recovar/em/helpers/batch_planning.py
  recovar/em/helpers/projection.py
  recovar/em/scoring/scoring.py
  recovar/em/scoring/significance.py
  recovar/em/classification/k_class.py
  recovar/em/vdam/dense_adapter.py
  recovar/em/vdam/driver.py
  recovar/em/vdam/iteration_loop.py
  recovar/em/vdam/m_step.py
  recovar/em/vdam/schedules.py
  recovar/commands/initial_model.py
  scripts/run_ab_initio.py
  scripts/run_vdam_late_iteration_profile.py
  scripts/analyze_vdam_coarse_multistream_late_pair.py
  scripts/analyze_vdam_coarse_gemm_gf46_gate.py
  scripts/run_vdam_coarse_gemm_gf46_gate.sbatch
  scripts/vdam_gpu_selection.sh
  tests/unit/test_coarse_gaussian_gemm_macro.py
  tests/unit/test_coarse_selector_audit.py
  tests/unit/initial_model/test_coarse_selector_audit_adapter.py
  tests/unit/initial_model/test_vdam_coarse_gemm_gf46_analyzer.py
  tests/unit/initial_model/test_vdam_coarse_gemm_gf46_runner.py
  pixi.lock
  pixi.toml
  pyproject.toml
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
  local message="VDAM coarse GEMM GF46 gate failed: status=${status} line=${BASH_LINENO[0]} command=${BASH_COMMAND}"
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

verify_frozen_inputs() {
  sha256sum -c "${PROVENANCE}/input_manifest.sha256" > /dev/null
  test "$(sha256sum "${PARTICLE_STACK}" | awk '{print $1}')" = \
    "${EXPECTED_PARTICLE_STACK_SHA256}"
}

verify_runtime_inputs() {
  test -e "${QUALIFIED_GPU_GATE_ROOT}/COMPLETED"
  test -s "${QUALIFIED_GPU_GATE_ROOT}/SHA256SUMS"
  test "$(sha256sum "${QUALIFIED_GPU_GATE_ROOT}/SHA256SUMS" | awk '{print $1}')" = \
    "${QUALIFIED_GATE_SHA256SUMS_SHA256}"
  (
    cd "${QUALIFIED_GPU_GATE_ROOT}"
    sha256sum -c SHA256SUMS
  ) > /dev/null
  test -s "${QUALIFIED_CUDA}"
  test "$(sha256sum "${QUALIFIED_CUDA}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
  test -s "${RELION_BIND_SOURCE}"
  test "$(sha256sum "${RELION_BIND_SOURCE}" | awk '{print $1}')" = \
    "${EXPECTED_RELION_BIND_SHA256}"
  test -x "${PIXI_PY}"
  test "$(sha256sum "${PIXI_PY}" | awk '{print $1}')" = \
    "${EXPECTED_INTERPRETER_SHA256}"
  test -s "${CUSPARSE_LIBRARY}"
  test "$(sha256sum "${CUSPARSE_LIBRARY}" | awk '{print $1}')" = \
    "${EXPECTED_CUSPARSE_SHA256}"
}

verify_repo_and_runtime() {
  test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
  test "$(git -C "${REPO_ROOT}" rev-parse 'HEAD^{tree}')" = "${EXPECTED_REPO_TREE}"
  test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=all)"
  verify_frozen_inputs
  test "$(sha256sum "${CUDA_BINARY}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
  test "$(sha256sum "${RELION_BIND_BINARY}" | awk '{print $1}')" = \
    "${EXPECTED_RELION_BIND_SHA256}"
  test "$(sha256sum "${PIXI_PY}" | awk '{print $1}')" = \
    "${EXPECTED_INTERPRETER_SHA256}"
  test "$(sha256sum "${CUSPARSE_LIBRARY}" | awk '{print $1}')" = \
    "${EXPECTED_CUSPARSE_SHA256}"
  vdam_verify_selected_gpu "${TARGET_GPU_UUID}"
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
for source_file in "${SOURCE_FILES[@]}"; do
  test -s "${REPO_ROOT}/${source_file}"
done
for input_file in "${INPUT_FILES[@]}"; do
  test -s "${input_file}"
done
verify_runtime_inputs
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
mkdir -p "${PROVENANCE}" "${RUNTIME}/cuda" "${RELION_BIND_DIR}" "${ANALYSIS}" "${RAW_OUTPUT}"
touch "${ROOT}/SAFE_TO_DELETE"
test ! -e "${RAW_DIAGNOSTIC_DIR}"
cp --reflink=auto "${QUALIFIED_CUDA}" "${CUDA_BINARY}"
cp --reflink=auto "${RELION_BIND_SOURCE}" "${RELION_BIND_BINARY}"
cp --reflink=auto "${QUALIFIED_GPU_GATE_ROOT}/SHA256SUMS" \
  "${PROVENANCE}/qualified_gate.SHA256SUMS"
cp --reflink=auto "${QUALIFIED_GPU_GATE_ROOT}/SHA256SUMS.sha256" \
  "${PROVENANCE}/qualified_gate.SHA256SUMS.sha256"

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
printf '%s\n' "${SLURM_JOB_ID}" > "${PROVENANCE}/slurm_job_id.txt"
printf '%s\n' "${TARGET_NODE}" > "${PROVENANCE}/node.txt"
printf '%s\n' "${selected_gpu_uuid}" > "${PROVENANCE}/selected_gpu_uuid.txt"
printf '%s\n' "${gpu_name}" > "${PROVENANCE}/gpu_name.txt"
printf '%s\n' "${VDAM_ALLOCATED_GPU_UUIDS_CSV}" > "${PROVENANCE}/allocated_gpu_uuids.csv"
printf '%s\n' "${VDAM_VISIBLE_GPU_UUIDS_CSV}" > "${PROVENANCE}/visible_gpu_uuids.csv"
sha256sum "${PIXI_PY}" > "${PROVENANCE}/interpreter.sha256"
sha256sum "${CUDA_BINARY}" > "${PROVENANCE}/qualified_cuda.sha256"
sha256sum "${RELION_BIND_BINARY}" > "${PROVENANCE}/relion_bind.sha256"
sha256sum "${CUSPARSE_LIBRARY}" > "${PROVENANCE}/cusparse.sha256"
nvidia-smi -i "${selected_gpu_uuid}" -q > "${PROVENANCE}/nvidia_smi.txt"
scontrol show job "${SLURM_JOB_ID}" > "${PROVENANCE}/slurm_job.txt"
env | LC_ALL=C sort > "${PROVENANCE}/submission_environment.txt"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV JAX_PLATFORM_NAME
unset RECOVAR_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_DIR
unset RECOVAR_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_ORIGINAL_INDICES
unset RECOVAR_COARSE_GAUSSIAN_GEMM_MACRO
unset RECOVAR_K1_COARSE_FUSED_PROJECTOR
unset RECOVAR_RELION_COARSE_CANONICAL_REDUCTION
unset RECOVAR_K1_COARSE_NATIVE_ATOMIC_REDUCTION
unset RECOVAR_K1_COARSE_PREHALF_WEIGHT
unset RECOVAR_K1_COARSE_SINGLE_LANE_CANONICAL
unset RECOVAR_K1_COARSE_MULTISTREAM_WORKERS
unset RECOVAR_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE
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
assert_parity_ancestors()
print("VDAM coarse GEMM GF46 runtime gate passed", jax.devices(), recovar.__file__)
PY
) > "${PROVENANCE}/runtime_gate.stdout" 2> "${PROVENANCE}/runtime_gate.stderr"
vdam_verify_selected_gpu "${selected_gpu_uuid}"

(
  cd "${REPO_ROOT}"
  env LD_PRELOAD="${CUSPARSE_LIBRARY}" "${PIXI_PY}" - \
    "${DATA_STAR}" "${CHECKPOINT}" "${PROVENANCE}/raw_target_preflight.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from recovar.data_io.starfile import read_star
from recovar.em.vdam.driver import (
    _micrograph_sort_order,
    _optics_group_indices,
    _relion_star_list_value,
)
from recovar.em.vdam.iteration_loop import select_subset_for_iter
from recovar.em.vdam.state import InitialModelState
from recovar.relion_bind import _relion_bind_core as relion_bind_core

EXPECTED_TARGETS = (1, 2160)
EXPECTED_POSITIONS = (72, 999)
EXPECTED_PART_IDS = (1, 2160)
EXPECTED_HALFSETS = (1, 0)
EXPECTED_SELECTED_IDS_INT64_SHA256 = (
    "c0199226ec7aa92f74a2fd66660e597fe44d73155db20f026b278840992a90be"
)
EXPECTED_PARTICLE_COUNT = 3000
EXPECTED_SUBSET_SIZE = 1000
EXPECTED_CHECKPOINT_ITERATION = 180
EXPECTED_RANDOM_SEED = 29
REQUESTED_IMAGE_BATCH_SIZE = 500

data_star = Path(sys.argv[1]).resolve(strict=True)
checkpoint = Path(sys.argv[2]).resolve(strict=True)
output = Path(sys.argv[3])
main_star, _ = read_star(str(data_star))
checkpoint_text = checkpoint.read_text()
checkpoint_iteration = _relion_star_list_value(
    checkpoint_text,
    "rlnCurrentIteration",
    int,
)
random_seed = _relion_star_list_value(checkpoint_text, "rlnRandomSeed", int)
subset_size = _relion_star_list_value(checkpoint_text, "rlnSgdSubsetSize", int)
nr_iter = _relion_star_list_value(checkpoint_text, "rlnNumberOfIterations", int)
if (
    len(main_star),
    checkpoint_iteration,
    random_seed,
    subset_size,
    nr_iter,
) != (
    EXPECTED_PARTICLE_COUNT,
    EXPECTED_CHECKPOINT_ITERATION,
    EXPECTED_RANDOM_SEED,
    EXPECTED_SUBSET_SIZE,
    200,
):
    raise RuntimeError("GF46 continuation preflight scalar state differs")

particle_order = _micrograph_sort_order(main_star)
if not np.array_equal(
    particle_order,
    np.arange(EXPECTED_PARTICLE_COUNT, dtype=np.int64),
):
    raise RuntimeError("GF46 Experiment read order differs from the frozen identity order")


def _fallback_forbidden(_seed: int):
    raise RuntimeError("native RELION shuffle binding was not used")


state = InitialModelState(
    iter=checkpoint_iteration,
    nr_iter=nr_iter,
    subset_size=subset_size,
)
selected_state = select_subset_for_iter(
    state,
    iter=checkpoint_iteration + 1,
    nr_particles=len(main_star),
    optics_group_by_particle=_optics_group_indices(main_star),
    rnd_unif_factory=_fallback_forbidden,
    random_seed=random_seed,
    do_grad=True,
    particle_order=particle_order,
)
selected = np.asarray(selected_state.subset_particle_ids, dtype=np.int64)
part_ids = np.asarray(selected_state.sorted_particle_part_ids[:subset_size], dtype=np.int64)
halfsets = np.asarray(selected_state.subset_halfset_ids, dtype=np.int64)
selected_digest = hashlib.sha256(
    np.ascontiguousarray(selected.astype("<i8", copy=False)).tobytes()
).hexdigest()
if selected.shape != (EXPECTED_SUBSET_SIZE,) or np.unique(selected).size != EXPECTED_SUBSET_SIZE:
    raise RuntimeError("GF46 selected subset is not a unique 1000-particle prefix")
if selected_digest != EXPECTED_SELECTED_IDS_INT64_SHA256:
    raise RuntimeError(
        "GF46 selected-particle hash differs: "
        f"{selected_digest} != {EXPECTED_SELECTED_IDS_INT64_SHA256}"
    )

positions = tuple(int(np.flatnonzero(selected == target)[0]) for target in EXPECTED_TARGETS)
target_part_ids = tuple(int(part_ids[position]) for position in positions)
target_halfsets = tuple(int(halfsets[position]) for position in positions)
requested_batch_indices = tuple(
    position // REQUESTED_IMAGE_BATCH_SIZE for position in positions
)
if positions != EXPECTED_POSITIONS:
    raise RuntimeError(f"GF46 target positions differ: {positions} != {EXPECTED_POSITIONS}")
if target_part_ids != EXPECTED_PART_IDS:
    raise RuntimeError(f"GF46 target part IDs differ: {target_part_ids} != {EXPECTED_PART_IDS}")
if target_halfsets != EXPECTED_HALFSETS or set(target_halfsets) != {0, 1}:
    raise RuntimeError(f"GF46 target pseudo-halfsets differ: {target_halfsets}")
if requested_batch_indices != (0, 1):
    raise RuntimeError(
        "GF46 targets no longer force two distinct requested image batches: "
        f"{requested_batch_indices}"
    )
if not callable(getattr(relion_bind_core, "vdam_randomise_particles_order", None)):
    raise RuntimeError("native RELION shuffle binding is unavailable")

payload = {
    "schema": "recovar.vdam_coarse_gemm_gf46_raw_target_preflight.v1",
    "checkpoint": str(checkpoint),
    "input_star": str(data_star),
    "checkpoint_iteration": checkpoint_iteration,
    "profiled_iteration": checkpoint_iteration + 1,
    "random_seed": random_seed,
    "native_shuffle_seed": random_seed + checkpoint_iteration + 1,
    "particle_count": len(main_star),
    "subset_size": subset_size,
    "selected_particle_ids_int64_sha256": selected_digest,
    "joint_halfset_particle_stream": True,
    "requested_image_batch_size": REQUESTED_IMAGE_BATCH_SIZE,
    "targets": [
        {
            "original_index": target,
            "selected_position": position,
            "part_id": part_id,
            "pseudo_halfset_id": halfset,
            "requested_batch_index": batch_index,
        }
        for target, position, part_id, halfset, batch_index in zip(
            EXPECTED_TARGETS,
            positions,
            target_part_ids,
            target_halfsets,
            requested_batch_indices,
            strict=True,
        )
    ],
}
with output.open("x", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY
) > "${PROVENANCE}/raw_target_preflight.stdout" \
  2> "${PROVENANCE}/raw_target_preflight.stderr"

raw_cache_root=${RAW_ROOT}/jax_cache
mkdir "${raw_cache_root}"
touch "${raw_cache_root}/SAFE_TO_DELETE"
raw_command=(
  env
  "LD_PRELOAD=${CUSPARSE_LIBRARY}"
  "RECOVAR_CACHE_DIR="
  RECOVAR_EM_RAW_IMAGE_CACHE=auto
  RECOVAR_EM_RAW_IMAGE_CACHE_MAX_GB=16
  RECOVAR_K1_COARSE_FUSED_PROJECTOR=0
  RECOVAR_RELION_COARSE_CANONICAL_REDUCTION=0
  RECOVAR_K1_COARSE_NATIVE_ATOMIC_REDUCTION=0
  RECOVAR_K1_COARSE_PREHALF_WEIGHT=0
  RECOVAR_K1_COARSE_SINGLE_LANE_CANONICAL=0
  RECOVAR_K1_COARSE_MULTISTREAM_WORKERS=0
  RECOVAR_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE=0
  RECOVAR_K1_COARSE_GAUSSIAN_FFI=1
  RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF=1
  RECOVAR_K1_RELION_EXACT_COARSE_OPERANDS=1
  RECOVAR_K1_RELION_F32_COARSE_SUPPORT=1
  RECOVAR_COARSE_GAUSSIAN_GEMM_MAX_PROJECTED_TRANSIENT_GB=2
  RECOVAR_COARSE_GAUSSIAN_GEMM_MACRO=1
  "RECOVAR_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_DIR=${RAW_DIAGNOSTIC_DIR}"
  "RECOVAR_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_ORIGINAL_INDICES=${RAW_TARGETS_CSV}"
  RECOVAR_INITIAL_MODEL_PROFILE=1
  "JAX_COMPILATION_CACHE_DIR=${raw_cache_root}"
  "${PIXI_PY}" -m scripts.run_ab_initio
  --i "${DATA_STAR}"
  --o "${RAW_OUTPUT}/run"
  --nr_iter 200
  --grad_write_iter 1
  --K 1
  --tau2_fudge 4
  --sym C1
  --do_run_C1 1
  --particle_diameter 200
  --random_seed 29
  --healpix_order 1
  --oversampling 1
  --offset_range 6
  --offset_step 2
  --padding_factor 1
  --image_batch_size 500
  --datadir "${DATA_DIR}"
  --gpu 0
  --require_custom_cuda
  --diagnostic_continue_optimiser "${CHECKPOINT}"
  --diagnostic_stop_after_iteration "${PROFILED_ITERATION}"
)
printf '%q ' "${raw_command[@]}" > "${PROVENANCE}/raw_score_diagnostic_command.sh"
printf '\n' >> "${PROVENANCE}/raw_score_diagnostic_command.sh"
(
  cd "${REPO_ROOT}"
  /usr/bin/time -v -o "${RAW_ROOT}/process.time" \
    "${raw_command[@]}" > "${RAW_ROOT}/runner.stdout" 2> "${RAW_ROOT}/runner.stderr"
)
test -s "${RAW_OUTPUT}/run_it181_recovar_meta.json"
test ! -e "${RAW_OUTPUT}/run_it182_recovar_meta.json"
test "$(find "${RAW_DIAGNOSTIC_DIR}" -maxdepth 1 -type f -name 'coarse_gemm_manifest_*.json' | wc -l)" -eq 1
test "$(find "${RAW_DIAGNOSTIC_DIR}" -maxdepth 1 -type f -name 'coarse_gemm_scope_*.json' | wc -l)" -eq 1
test "$(find "${RAW_DIAGNOSTIC_DIR}" -maxdepth 1 -type f -name 'coarse_gemm_ab_*.npz' | wc -l)" -eq 2
touch "${ROOT}/RAW_COMPLETED"
verify_repo_and_runtime

RUN_LABELS=(direct_1 gemm_1 gemm_2 direct_2)
RUN_GEMM=(0 1 1 0)
test "${#RUN_LABELS[@]}" -eq "${#RUN_GEMM[@]}"
printf 'order\tlabel\tgemm_macro\n' > "${PROVENANCE}/execution_order.tsv"
for index in "${!RUN_LABELS[@]}"; do
  label=${RUN_LABELS[index]}
  gemm=${RUN_GEMM[index]}
  run_root=${ROOT}/runs/${label}
  cache_root=${run_root}/jax_cache
  mkdir -p "${run_root}" "${cache_root}"
  touch "${cache_root}/SAFE_TO_DELETE"
  printf '%d\t%s\t%s\n' "$((index + 1))" "${label}" "${gemm}" \
    >> "${PROVENANCE}/execution_order.tsv"
  command=(
    env
    -u RECOVAR_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_DIR
    -u RECOVAR_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_ORIGINAL_INDICES
    "LD_PRELOAD=${CUSPARSE_LIBRARY}"
    "RECOVAR_CACHE_DIR="
    RECOVAR_EM_RAW_IMAGE_CACHE=auto
    RECOVAR_EM_RAW_IMAGE_CACHE_MAX_GB=16
    RECOVAR_K1_COARSE_FUSED_PROJECTOR=0
    RECOVAR_RELION_COARSE_CANONICAL_REDUCTION=0
    RECOVAR_K1_COARSE_NATIVE_ATOMIC_REDUCTION=0
    RECOVAR_K1_COARSE_PREHALF_WEIGHT=0
    RECOVAR_K1_COARSE_SINGLE_LANE_CANONICAL=0
    RECOVAR_K1_COARSE_MULTISTREAM_WORKERS=0
    RECOVAR_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE=0
    RECOVAR_K1_COARSE_GAUSSIAN_FFI=1
    RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF=1
    RECOVAR_K1_RELION_EXACT_COARSE_OPERANDS=1
    RECOVAR_K1_RELION_F32_COARSE_SUPPORT=1
    RECOVAR_COARSE_GAUSSIAN_GEMM_MAX_PROJECTED_TRANSIENT_GB=2
    "RECOVAR_COARSE_GAUSSIAN_GEMM_MACRO=${gemm}"
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
  )
  printf '%q ' "${command[@]}" > "${PROVENANCE}/${label}_command.sh"
  printf '\n' >> "${PROVENANCE}/${label}_command.sh"
  (
    cd "${REPO_ROOT}"
    /usr/bin/time -v -o "${run_root}/process.time" \
      "${command[@]}" > "${run_root}/runner.stdout" 2> "${run_root}/runner.stderr"
  )
  test -s "${run_root}/profile/profile_summary.json"
  test -s "${run_root}/profile/warm/run_it181_recovar_meta.json"
  test ! -e "${run_root}/profile/warm/run_it182_recovar_meta.json"
  test -z "$(find "${run_root}" -type f -name 'coarse_gemm*' -print -quit)"
  verify_repo_and_runtime
done
touch "${ROOT}/ARMS_COMPLETED"

write_source_manifest > "${PROVENANCE}/source_manifest.final.sha256"
cmp "${PROVENANCE}/source_manifest.sha256" "${PROVENANCE}/source_manifest.final.sha256"

"${PIXI_PY}" - "${PROVENANCE}/run.json" "${SLURM_JOB_ID}" \
  "${EXPECTED_REPO_HEAD}" "${EXPECTED_REPO_TREE}" "${source_manifest_sha256}" \
  "${input_manifest_sha256}" "${selected_gpu_uuid}" "${gpu_name}" <<'PY'
import json
import sys
from pathlib import Path

payload = {
    "schema": "recovar.vdam_coarse_gemm_gf46_gate.v1",
    "classification": "diagnostic_performance_only",
    "job_id": sys.argv[2],
    "git_head": sys.argv[3],
    "git_tree": sys.argv[4],
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
    "cusparse_sha256": "58ffc54edb1d007f56a1718aaadcb30f45bbf662f43515920ea8ff094304bdbf",
    "execution_order": ["direct_1", "gemm_1", "gemm_2", "direct_2"],
    "gemm_macro_modes": [0, 1, 1, 0],
    "raw_score_diagnostic_original_indices": [1, 2160],
    "raw_score_diagnostic_selected_positions": [72, 999],
    "raw_score_diagnostic_part_ids": [1, 2160],
    "raw_score_diagnostic_pseudo_halfset_ids": [1, 0],
    "raw_score_diagnostic_joint_halfset_particle_stream": True,
    "selected_particle_ids_int64_sha256": "c0199226ec7aa92f74a2fd66660e597fe44d73155db20f026b278840992a90be",
    "raw_score_diagnostic_timing_eligible": False,
    "timing_diagnostic_variables_unset": True,
    "checkpoint_iteration": 180,
    "profiled_iteration": 181,
    "nr_iter_schedule": 200,
    "random_seed": 29,
    "image_batch_size": 500,
    "raw_image_cache": "auto",
    "raw_image_cache_max_gb": 16.0,
    "exact_local_bucket_radix": 4,
    "exact_local_physical_order_chunk_size": 0,
    "material_end_to_end_ratio": 0.90,
    "science_promotion_allowed": False,
    "default_enablement_allowed": False,
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

trap - ERR
set +e
(
  cd "${REPO_ROOT}"
  env LD_PRELOAD="${CUSPARSE_LIBRARY}" JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu \
    "${PIXI_PY}" -m scripts.analyze_vdam_coarse_gemm_gf46_gate \
    --root "${ROOT}" --repo "${REPO_ROOT}" \
    --output-json "${ANALYSIS}/report.json" \
    --output-markdown "${ANALYSIS}/report.md"
) > "${ANALYSIS}/analyzer.stdout" 2> "${ANALYSIS}/analyzer.stderr"
analyzer_status=$?
set -e
trap on_error ERR
printf '%s\n' "${analyzer_status}" > "${ANALYSIS}/analyzer_status.txt"
if [[ "${analyzer_status}" != 0 && "${analyzer_status}" != 1 ]]; then
  analyzer_message="coarse GEMM GF46 analyzer failed with setup/runtime status ${analyzer_status}"
  printf '%s\n' "${analyzer_message}" >&2
  printf '%s\n' "${analyzer_message}" > "${PROVENANCE}/failure.txt"
  exit "${analyzer_status}"
fi
test -s "${ANALYSIS}/report.json"
test -s "${ANALYSIS}/report.md"

verify_repo_and_runtime
write_source_manifest > "${PROVENANCE}/source_manifest.final.sha256"
cmp "${PROVENANCE}/source_manifest.sha256" "${PROVENANCE}/source_manifest.final.sha256"
if [[ "${analyzer_status}" == 0 ]]; then
  touch "${ROOT}/PASSED"
fi
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

printf 'VDAM coarse GEMM GF46 gate complete: %s (analyzer status %s)\n' \
  "${ROOT}" "${analyzer_status}"
if [[ "${analyzer_status}" == 1 ]]; then
  trap - ERR
  exit 1
fi
