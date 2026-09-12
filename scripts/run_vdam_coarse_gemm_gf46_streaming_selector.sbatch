#!/usr/bin/env bash
#SBATCH --job-name=vdam-gf46-stream
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
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-gf46-stream-%j.out

set -euo pipefail
unset PYTHONOPTIMIZE

: "${VDAM_COARSE_GEMM_GF46_STREAM_ROOT:?set a new disposable output root}"
: "${EXPECTED_REPO_HEAD:?pin the committed streaming-diagnostic head}"
: "${EXPECTED_REPO_TREE:?pin the committed streaming-diagnostic tree}"
: "${EXPECTED_SOURCE_MANIFEST_SHA256:?pin the compact source manifest}"
: "${PIXI_PY_OVERRIDE:?pin the qualified RECOVAR pixi interpreter}"

readonly REQUIRED_INTEGRATION_COMMIT=56f2ee1e892925421fa2d807379062158f6eb1e3
readonly REQUIRED_PRE_PRIOR_V2_COMMIT=8b0d5bcacdda537337a4bae0f5b2ac242d796562
readonly QUALIFIED_GPU_GATE_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_coarse_rotation_blocks_h100_695a629fa_20260901
readonly QUALIFIED_GATE_HEAD=695a629fa70ee951734d98728bb3daffcc53bd88
readonly QUALIFIED_GATE_TREE=04a3ca49be3547cc8329d6c27c7b1aa2acf30c3e
readonly QUALIFIED_GATE_JOB_ID=13328717
readonly QUALIFIED_CUDA=${QUALIFIED_GPU_GATE_ROOT}/runtime/libcuda_backproject.so
readonly EXPECTED_CUDA_SHA256=664c741946c16df13b59ae41c6a98949c5463beddbe33193e66948ae87a91de0
readonly EXPECTED_INTERPRETER_SHA256=3e05f86750f7f3bdb3df1b956cd8cd9b3ae11c655d0455ca4c71b714bc8745d8
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
readonly EXPECTED_SELECTED_IDS_INT64_SHA256=c0199226ec7aa92f74a2fd66660e597fe44d73155db20f026b278840992a90be
readonly CHECKPOINT_ITERATION=180
readonly PROFILED_ITERATION=181
readonly NR_ITER_SCHEDULE=200
readonly EXPECTED_PARTICLE_COUNT=3000
readonly EXPECTED_SUBSET_SIZE=1000
readonly STREAM_TOPK=2048

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
REPO_ROOT=$(realpath -e -- "${REPO_ROOT}")
PIXI_PY=$(realpath -e -- "${PIXI_PY_OVERRIDE}")
ROOT=$(realpath -m -- "${VDAM_COARSE_GEMM_GF46_STREAM_ROOT}")
PROVENANCE=${ROOT}/provenance
RUNTIME=${ROOT}/runtime
ANALYSIS=${ROOT}/analysis
DIAGNOSTIC_ROOT=${ROOT}/streaming_selector_diagnostic
DIAGNOSTIC_DIR=${DIAGNOSTIC_ROOT}/artifacts
OUTPUT=${DIAGNOSTIC_ROOT}/output
CACHE_ROOT=${DIAGNOSTIC_ROOT}/jax_cache
CUDA_BINARY=${RUNTIME}/cuda/libcuda_backproject.so
RELION_BIND_DIR=${RUNTIME}/relion_bind
RELION_BIND_BINARY=${RELION_BIND_DIR}/$(basename "${RELION_BIND_SOURCE}")
SELECTED_IDS_BINARY=${PROVENANCE}/selected_particle_ids.i64le
IDENTITY_CERTIFICATE=${ANALYSIS}/streaming_identity_certificate.json

SOURCE_FILES=(
  recovar/cuda/Makefile
  recovar/cuda/cuda_backproject.cu
  recovar/cuda/relion_coarse_diff2_projector_body.inc
  recovar/cuda_backproject.py
  recovar/data_io/image_loader.py
  recovar/data_io/staging.py
  recovar/em/helpers/batch_planning.py
  recovar/em/scoring/coarse_gemm_streaming.py
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
  scripts/run_vdam_coarse_gemm_gf46_streaming_selector.sbatch
  scripts/vdam_gpu_selection.sh
  tests/unit/test_coarse_gaussian_gemm_macro.py
  tests/unit/test_coarse_gaussian_gemm_streaming.py
  tests/unit/initial_model/test_vdam_coarse_gemm_gf46_streaming_runner.py
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
  local message="VDAM GF46 streaming selector diagnostic failed: status=${status} line=${BASH_LINENO[0]} command=${BASH_COMMAND}"
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

write_runtime_manifest() {
  sha256sum \
    "${QUALIFIED_CUDA}" \
    "${RELION_BIND_SOURCE}" \
    "${PIXI_PY}" \
    "${CUSPARSE_LIBRARY}" \
    "${QUALIFIED_GPU_GATE_ROOT}/provenance/results.sha256"
}

verify_frozen_inputs() {
  sha256sum -c "${PROVENANCE}/input_manifest.sha256" > /dev/null
  test "$(sha256sum "${PARTICLE_STACK}" | awk '{print $1}')" = \
    "${EXPECTED_PARTICLE_STACK_SHA256}"
  test "$(sha256sum "${SELECTED_IDS_BINARY}" | awk '{print $1}')" = \
    "${EXPECTED_SELECTED_IDS_INT64_SHA256}"
  test "$(stat -c %s "${SELECTED_IDS_BINARY}")" -eq "$((EXPECTED_SUBSET_SIZE * 8))"
}

verify_runtime_inputs() {
  test -e "${QUALIFIED_GPU_GATE_ROOT}/COMPLETED"
  test "$(<"${QUALIFIED_GPU_GATE_ROOT}/provenance/repo_head.txt")" = \
    "${QUALIFIED_GATE_HEAD}"
  test "$(<"${QUALIFIED_GPU_GATE_ROOT}/provenance/repo_tree.txt")" = \
    "${QUALIFIED_GATE_TREE}"
  test "$(<"${QUALIFIED_GPU_GATE_ROOT}/provenance/slurm_job_id.txt")" = \
    "${QUALIFIED_GATE_JOB_ID}"
  (
    cd "${QUALIFIED_GPU_GATE_ROOT}"
    sha256sum -c provenance/results.sha256
  ) > /dev/null
  test -s "${QUALIFIED_CUDA}"
  test "$(sha256sum "${QUALIFIED_CUDA}" | awk '{print $1}')" = \
    "${EXPECTED_CUDA_SHA256}"
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
  git -C "${REPO_ROOT}" merge-base --is-ancestor \
    "${REQUIRED_INTEGRATION_COMMIT}" "${EXPECTED_REPO_HEAD}"
  git -C "${REPO_ROOT}" merge-base --is-ancestor \
    "${REQUIRED_PRE_PRIOR_V2_COMMIT}" "${EXPECTED_REPO_HEAD}"
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
git -C "${REPO_ROOT}" merge-base --is-ancestor \
  "${REQUIRED_INTEGRATION_COMMIT}" "${EXPECTED_REPO_HEAD}"
git -C "${REPO_ROOT}" merge-base --is-ancestor \
  "${REQUIRED_PRE_PRIOR_V2_COMMIT}" "${EXPECTED_REPO_HEAD}"
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
mkdir -p \
  "${PROVENANCE}" \
  "${RUNTIME}/cuda" \
  "${RELION_BIND_DIR}" \
  "${ANALYSIS}" \
  "${OUTPUT}" \
  "${CACHE_ROOT}"
touch "${ROOT}/SAFE_TO_DELETE" "${CACHE_ROOT}/SAFE_TO_DELETE"
test ! -e "${DIAGNOSTIC_DIR}"
cp --reflink=auto "${QUALIFIED_CUDA}" "${CUDA_BINARY}"
cp --reflink=auto "${RELION_BIND_SOURCE}" "${RELION_BIND_BINARY}"

sha256sum "${INPUT_FILES[@]}" > "${PROVENANCE}/input_manifest.sha256"
input_manifest_sha256=$(sha256sum "${PROVENANCE}/input_manifest.sha256" | awk '{print $1}')
test "${input_manifest_sha256}" = "${EXPECTED_INPUT_MANIFEST_SHA256}"
write_source_manifest > "${PROVENANCE}/source_manifest.sha256"
source_manifest_sha256=$(sha256sum "${PROVENANCE}/source_manifest.sha256" | awk '{print $1}')
test "${source_manifest_sha256}" = "${EXPECTED_SOURCE_MANIFEST_SHA256}"
write_runtime_manifest > "${PROVENANCE}/runtime_manifest.sha256"
runtime_manifest_sha256=$(sha256sum "${PROVENANCE}/runtime_manifest.sha256" | awk '{print $1}')

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
nvidia-smi -i "${selected_gpu_uuid}" -q > "${PROVENANCE}/nvidia_smi.txt"
scontrol show job "${SLURM_JOB_ID}" > "${PROVENANCE}/slurm_job.txt"
env | LC_ALL=C sort > "${PROVENANCE}/submission_environment.txt"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV JAX_PLATFORM_NAME
unset RECOVAR_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_DIR
unset RECOVAR_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_ORIGINAL_INDICES
unset RECOVAR_COARSE_GAUSSIAN_GEMM_STREAM_DIAGNOSTIC_DIR
unset RECOVAR_COARSE_GAUSSIAN_GEMM_STREAM_TOPK
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
assert callable(getattr(cuda_backproject, "relion_coarse_diff2_rotation_blocks_f32", None))
assert callable(getattr(relion_bind_core, "vdam_rnd_unif_sequence", None))
assert_parity_ancestors()
print("VDAM GF46 streaming selector runtime gate passed", jax.devices(), recovar.__file__)
PY
) > "${PROVENANCE}/runtime_gate.stdout" 2> "${PROVENANCE}/runtime_gate.stderr"
vdam_verify_selected_gpu "${selected_gpu_uuid}"

(
  cd "${REPO_ROOT}"
  env LD_PRELOAD="${CUSPARSE_LIBRARY}" "${PIXI_PY}" - \
    "${DATA_STAR}" "${CHECKPOINT}" "${PROVENANCE}/selection_preflight.json" \
    "${SELECTED_IDS_BINARY}" <<'PY'
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
selected_output = Path(sys.argv[4])
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
selected_bytes = np.ascontiguousarray(selected.astype("<i8", copy=False)).tobytes()
selected_digest = hashlib.sha256(selected_bytes).hexdigest()
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

with selected_output.open("xb") as stream:
    stream.write(selected_bytes)
payload = {
    "schema": "recovar.vdam_coarse_gemm_gf46_streaming_preflight.v1",
    "checkpoint": str(checkpoint),
    "input_star": str(data_star),
    "checkpoint_iteration": checkpoint_iteration,
    "profiled_iteration": checkpoint_iteration + 1,
    "random_seed": random_seed,
    "native_shuffle_seed": random_seed + checkpoint_iteration + 1,
    "particle_count": len(main_star),
    "subset_size": subset_size,
    "selected_particle_ids_int64_sha256": selected_digest,
    "selected_particle_ids_binary": selected_output.name,
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
) > "${PROVENANCE}/selection_preflight.stdout" \
  2> "${PROVENANCE}/selection_preflight.stderr"
verify_frozen_inputs

# Keep this one-transition command aligned with the diagnostic continuation in
# run_vdam_coarse_gemm_gf46_gate.sbatch.  Only the bounded all-particle
# streaming diagnostic replaces the old two-particle score-cube capture.
stream_command=(
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
  RECOVAR_COARSE_GAUSSIAN_GEMM_MACRO=1
  "RECOVAR_COARSE_GAUSSIAN_GEMM_STREAM_DIAGNOSTIC_DIR=${DIAGNOSTIC_DIR}"
  "RECOVAR_COARSE_GAUSSIAN_GEMM_STREAM_TOPK=${STREAM_TOPK}"
  RECOVAR_INITIAL_MODEL_PROFILE=1
  "JAX_COMPILATION_CACHE_DIR=${CACHE_ROOT}"
  "${PIXI_PY}" -m scripts.run_ab_initio
  --i "${DATA_STAR}"
  --o "${OUTPUT}/run"
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
printf '%q ' "${stream_command[@]}" > "${PROVENANCE}/streaming_diagnostic_command.sh"
printf '\n' >> "${PROVENANCE}/streaming_diagnostic_command.sh"
(
  cd "${REPO_ROOT}"
  "${stream_command[@]}" > "${DIAGNOSTIC_ROOT}/runner.stdout" \
    2> "${DIAGNOSTIC_ROOT}/runner.stderr"
)
test -s "${OUTPUT}/run_it181_recovar_meta.json"
test ! -e "${OUTPUT}/run_it182_recovar_meta.json"
test "$(find "${DIAGNOSTIC_DIR}" -maxdepth 1 -type f -name 'coarse_gemm_rescore_manifest_*.json' | wc -l)" -eq 1
test "$(find "${DIAGNOSTIC_DIR}" -maxdepth 1 -type f -name 'coarse_gemm_rescore_scope_*.json' | wc -l)" -eq 1
test "$(find "${DIAGNOSTIC_DIR}" -maxdepth 1 -type f -name 'coarse_gemm_rescore_*.npz' | wc -l)" -gt 0
test -z "$(find "${DIAGNOSTIC_DIR}" -maxdepth 1 -type f \( \
  -name 'coarse_gemm_ab_*.npz' -o \
  -name 'coarse_gemm_manifest_*.json' -o \
  -name 'coarse_gemm_scope_*.json' \
\) -print -quit)"

"${PIXI_PY}" - \
  "${DIAGNOSTIC_DIR}" \
  "${OUTPUT}/run_it181_recovar_meta.json" \
  "${SELECTED_IDS_BINARY}" \
  "${IDENTITY_CERTIFICATE}" \
  "${EXPECTED_SUBSET_SIZE}" \
  "${STREAM_TOPK}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar_bool(artifact, name: str) -> bool:
    require(name in artifact.files, f"streaming artifact lacks {name}")
    value = np.asarray(artifact[name])
    require(value.shape == (), f"streaming artifact {name} is not scalar")
    return bool(value.item())


def dotted_paths(value, prefix: str = "") -> list[str]:
    if not isinstance(value, dict):
        return [prefix] if prefix else []
    paths = []
    for key, child in value.items():
        child_prefix = f"{prefix}.{key}" if prefix else str(key)
        paths.extend(dotted_paths(child, child_prefix))
    return paths


diagnostic_dir = Path(sys.argv[1]).resolve(strict=True)
meta_path = Path(sys.argv[2]).resolve(strict=True)
selected_path = Path(sys.argv[3]).resolve(strict=True)
certificate_path = Path(sys.argv[4])
expected_count = int(sys.argv[5])
expected_topk = int(sys.argv[6])
expected_ids = np.fromfile(selected_path, dtype="<i8")
require(expected_ids.shape == (expected_count,), "preflight selected-id count differs")
require(np.unique(expected_ids).size == expected_count, "preflight selected IDs are not unique")

aggregate_paths = sorted(diagnostic_dir.glob("coarse_gemm_rescore_manifest_*.json"))
require(len(aggregate_paths) == 1, "streaming diagnostic requires exactly one aggregate manifest")
aggregate_path = aggregate_paths[0].resolve()
aggregate = json.loads(aggregate_path.read_text())
require(isinstance(aggregate, dict), "streaming aggregate must be a JSON object")
require(int(aggregate.get("schema_version", 0)) >= 2, "streaming aggregate predates dual-state v2")
require(aggregate.get("particle_count") == expected_count, "aggregate particle count differs")
require(aggregate.get("retained_topk") == expected_topk, "aggregate retained top-k differs")
require(aggregate.get("all_particles_captured_exactly_once") is True, "aggregate exactly-once seal is absent")
require(aggregate.get("stores_score_cube") is False, "aggregate claims a serialized score cube")
summary = aggregate.get("summary")
require(isinstance(summary, dict), "streaming aggregate summary is missing")
require(int(summary.get("schema_version", 0)) >= 2, "streaming summary predates dual-state v2")
require(summary.get("particle_count") == expected_count, "streaming summary particle count differs")
require(summary.get("retained_topk") == expected_topk, "streaming summary retained top-k differs")
for summary_section in (
    "all_candidate",
    "pre_prior_all_candidate",
    "support",
    "relion_nonzero_rescore",
    "relion_raw_max_rescore",
    "relion_rescore_union",
    "persistent_streaming_state_bytes",
):
    require(isinstance(summary.get(summary_section), dict), f"streaming summary lacks {summary_section}")
scope_records = aggregate.get("scope_records")
require(isinstance(scope_records, list) and len(scope_records) == 1, "joint stream must have one scope record")
expected_call_ids = aggregate.get("expected_call_ids")
require(isinstance(expected_call_ids, list) and len(expected_call_ids) == 1, "joint stream call contract differs")

scope_ids = []
artifact_ids = []
artifact_names = []
artifact_records = []
field_sets = []
for record in scope_records:
    require(isinstance(record, dict), "scope record is not an object")
    require(record.get("stores_score_cube") is False, "scope record claims a score cube")
    require(record.get("retained_topk") == expected_topk, "scope top-k differs")
    record_ids = record.get("original_indices")
    require(isinstance(record_ids, list), "scope record lacks original particle IDs")
    require(record.get("particle_count") == len(record_ids), "scope particle count differs")
    scope_ids.extend(int(value) for value in record_ids)
    names = record.get("artifact_paths")
    require(isinstance(names, list) and names, "scope record has no artifacts")
    for name in names:
        require(isinstance(name, str) and Path(name).name == name, "unsafe artifact basename")
        require(name not in artifact_names, f"duplicate artifact reference: {name}")
        artifact_names.append(name)
        path = (diagnostic_dir / name).resolve(strict=True)
        require(path.parent == diagnostic_dir, f"artifact escapes diagnostic root: {name}")
        require(path.match("coarse_gemm_rescore_*.npz"), f"unexpected artifact name: {name}")
        with np.load(path, allow_pickle=False) as artifact:
            fields = set(artifact.files)
            field_sets.append(fields)
            artifact_schema = str(np.asarray(artifact["schema"]).item())
            schema_prefix = "recovar.coarse_gemm_streaming_rescore.v"
            require(artifact_schema.startswith(schema_prefix), f"unexpected artifact schema: {name}")
            require(int(artifact_schema.removeprefix(schema_prefix)) >= 2, f"artifact predates dual-state v2: {name}")
            require("original_indices" in fields, f"artifact lacks identities: {name}")
            require("actual_image_count" in fields, f"artifact lacks image count: {name}")
            ids = np.asarray(artifact["original_indices"], dtype=np.int64)
            require(ids.ndim == 1, f"artifact identities are not one-dimensional: {name}")
            require(int(np.asarray(artifact["actual_image_count"]).item()) == ids.size, f"artifact image count differs: {name}")
            require(int(np.asarray(artifact["retained_topk"]).item()) == expected_topk, f"artifact top-k differs: {name}")
            require(scalar_bool(artifact, "paired_capture_active"), f"artifact capture is inactive: {name}")
            require(not scalar_bool(artifact, "clean_timing_eligible"), f"artifact is incorrectly timing eligible: {name}")
            require(not scalar_bool(artifact, "stores_score_cube"), f"artifact stores a score cube: {name}")
            require(not scalar_bool(artifact, "production_behavior_changed"), f"artifact changed production behavior: {name}")
            for dual_state_field in (
                "pre_prior_finite_pair_count",
                "pre_prior_all_candidate_max_abs_delta",
                "raw_max_error_safe_superset_count",
                "raw_max_error_safe_source_rotation_block_ids",
                "relion_rescore_source_rotation_block_union_ids",
                "posterior_streaming_state_bytes",
                "pre_prior_streaming_state_bytes",
                "persistent_streaming_state_bytes",
            ):
                require(dual_state_field in fields, f"artifact lacks v2 field {dual_state_field}: {name}")
            posterior_bytes = int(np.asarray(artifact["posterior_streaming_state_bytes"]).item())
            pre_prior_bytes = int(np.asarray(artifact["pre_prior_streaming_state_bytes"]).item())
            persistent_bytes = int(np.asarray(artifact["persistent_streaming_state_bytes"]).item())
            require(posterior_bytes > 0 and pre_prior_bytes > 0, f"artifact dual-state byte counts are empty: {name}")
            require(persistent_bytes == posterior_bytes + pre_prior_bytes, f"artifact dual-state byte accounting differs: {name}")
            artifact_ids.extend(int(value) for value in ids)
            artifact_records.append(
                {
                    "name": name,
                    "sha256": sha256(path),
                    "particle_count": int(ids.size),
                    "fields": sorted(fields),
                }
            )

expected_list = expected_ids.tolist()
require(scope_ids == expected_list, "scope records do not cover the selected stream in order")
require(artifact_ids == expected_list, "streaming artifacts do not cover the selected stream in order")
require(len(set(artifact_ids)) == expected_count, "streaming artifacts contain duplicate particle IDs")
actual_artifacts = sorted(path.name for path in diagnostic_dir.glob("coarse_gemm_rescore_*.npz"))
require(sorted(artifact_names) == actual_artifacts, "aggregate does not cover every streaming artifact")

metadata = json.loads(meta_path.read_text())
require(metadata.get("joint_halfset_particle_stream") is True, "continuation was not joint-halfset")
require(metadata.get("selected_particle_ids") == expected_list, "iteration metadata selected IDs differ")
recorded_path = Path(str(metadata.get("coarse_gaussian_gemm_stream_aggregate_manifest_path", ""))).resolve()
require(recorded_path == aggregate_path, "iteration metadata aggregate path differs")
require("coarse_gaussian_gemm_aggregate_manifest_path" not in metadata, "old score-cube aggregate was emitted")

field_union = sorted(set().union(*field_sets))
field_intersection = sorted(set.intersection(*field_sets))
summary_paths = sorted(dotted_paths(summary))
raw_or_pre_prior_fields = [
    name for name in field_union + summary_paths
    if "raw" in name.lower() or "pre_prior" in name.lower()
]
posterior_or_support_fields = [
    name for name in field_union + summary_paths
    if "posterior" in name.lower() or "support" in name.lower()
]
require(raw_or_pre_prior_fields, "dual-state v2 raw/pre-prior certificate fields are absent")
require(posterior_or_support_fields, "dual-state v2 posterior/support certificate fields are absent")
certificate = {
    "schema": "recovar.vdam_coarse_gemm_gf46_streaming_identity.v1",
    "classification": "diagnostic_only_timing_ineligible",
    "aggregate_manifest": {
        "path": str(aggregate_path),
        "sha256": sha256(aggregate_path),
    },
    "selected_particle_ids_int64_sha256": sha256(selected_path),
    "selected_particle_count": expected_count,
    "retained_topk": expected_topk,
    "all_particles_captured_exactly_once": True,
    "identity_order_matches_preflight": True,
    "stores_score_cube": False,
    "clean_timing_eligible": False,
    "artifact_count": len(artifact_records),
    "artifacts": artifact_records,
    "artifact_field_intersection": field_intersection,
    "artifact_field_union": field_union,
    "aggregate_summary_paths": summary_paths,
    "raw_or_pre_prior_fields": raw_or_pre_prior_fields,
    "posterior_or_support_fields": posterior_or_support_fields,
}
with certificate_path.open("x", encoding="utf-8") as stream:
    json.dump(certificate, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

test -s "${IDENTITY_CERTIFICATE}"
touch "${ROOT}/IDENTITY_VALIDATED"
verify_repo_and_runtime
write_source_manifest > "${PROVENANCE}/source_manifest.final.sha256"
cmp "${PROVENANCE}/source_manifest.sha256" "${PROVENANCE}/source_manifest.final.sha256"
write_runtime_manifest > "${PROVENANCE}/runtime_manifest.final.sha256"
cmp "${PROVENANCE}/runtime_manifest.sha256" "${PROVENANCE}/runtime_manifest.final.sha256"

aggregate_manifest=$(find "${DIAGNOSTIC_DIR}" -maxdepth 1 -type f \
  -name 'coarse_gemm_rescore_manifest_*.json' -print -quit)
"${PIXI_PY}" - \
  "${PROVENANCE}/run.json" \
  "${SLURM_JOB_ID}" \
  "${EXPECTED_REPO_HEAD}" \
  "${EXPECTED_REPO_TREE}" \
  "${source_manifest_sha256}" \
  "${input_manifest_sha256}" \
  "${runtime_manifest_sha256}" \
  "${selected_gpu_uuid}" \
  "${gpu_name}" \
  "${aggregate_manifest}" \
  "${IDENTITY_CERTIFICATE}" <<'PY'
import json
import sys
from pathlib import Path

payload = {
    "schema": "recovar.vdam_coarse_gemm_gf46_streaming_selector_run.v1",
    "classification": "diagnostic_only_timing_ineligible",
    "job_id": sys.argv[2],
    "git_head": sys.argv[3],
    "git_tree": sys.argv[4],
    "source_manifest_sha256": sys.argv[5],
    "source_manifest_scope": "compact_selected_high_risk_files",
    "input_manifest_sha256": sys.argv[6],
    "runtime_manifest_sha256": sys.argv[7],
    "gpu_uuid": sys.argv[8],
    "gpu_name": sys.argv[9],
    "node": "della-h21g4",
    "aggregate_manifest_path": str(Path(sys.argv[10]).resolve()),
    "identity_certificate_path": str(Path(sys.argv[11]).resolve()),
    "qualified_gpu_gate_root": "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_coarse_rotation_blocks_h100_695a629fa_20260901",
    "qualified_gate_job_id": 13328717,
    "cuda_sha256": "664c741946c16df13b59ae41c6a98949c5463beddbe33193e66948ae87a91de0",
    "relion_bind_sha256": "9bbb1fb0ce6fa7ac816598ec521453515d163221642b916e5715bb2850798980",
    "interpreter_sha256": "3e05f86750f7f3bdb3df1b956cd8cd9b3ae11c655d0455ca4c71b714bc8745d8",
    "cusparse_sha256": "58ffc54edb1d007f56a1718aaadcb30f45bbf662f43515920ea8ff094304bdbf",
    "particle_stack_sha256": "804af933bd315f41f0159f62e93867cf852d70cb29f2f27a525fb2fc3eb68ad9",
    "selected_particle_ids_int64_sha256": "c0199226ec7aa92f74a2fd66660e597fe44d73155db20f026b278840992a90be",
    "selected_particle_count": 1000,
    "all_particles_captured_exactly_once": True,
    "stream_retained_topk": 2048,
    "checkpoint_iteration": 180,
    "profiled_iteration": 181,
    "nr_iter_schedule": 200,
    "random_seed": 29,
    "image_batch_size": 500,
    "joint_halfset_particle_stream": True,
    "old_two_particle_score_cube_diagnostic_enabled": False,
    "trajectory_timing_arms_enabled": False,
    "clean_timing_eligible": False,
    "timing_policy": "paired all-particle direct-plus-GEMM capture is timing-ineligible",
    "science_promotion_allowed": False,
    "default_enablement_allowed": False,
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

(
  cd "${ROOT}"
  find streaming_selector_diagnostic -type f -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum
) > "${PROVENANCE}/diagnostic_manifest.sha256"

verify_repo_and_runtime
write_source_manifest > "${PROVENANCE}/source_manifest.final.sha256"
cmp "${PROVENANCE}/source_manifest.sha256" "${PROVENANCE}/source_manifest.final.sha256"
write_runtime_manifest > "${PROVENANCE}/runtime_manifest.final.sha256"
cmp "${PROVENANCE}/runtime_manifest.sha256" "${PROVENANCE}/runtime_manifest.final.sha256"
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

printf 'VDAM GF46 streaming selector diagnostic complete: %s\n' "${ROOT}"
