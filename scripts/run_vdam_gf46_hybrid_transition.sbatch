#!/usr/bin/env bash
#SBATCH --job-name=vdam-hybrid-180-181
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-hybrid-180-181-%j.out

set -euo pipefail
unset PYTHONOPTIMIZE

: "${EXPECTED_REPO_HEAD:?pin the committed transition harness head}"
: "${EXPECTED_REPO_TREE:?pin the committed transition harness tree}"
: "${EXPECTED_SOURCE_MANIFEST_SHA256:?pin the transition source manifest}"
: "${RECOVAR_PIXI_PY:?set the absolute pixi Python executable}"

readonly CUDA_TOOLKIT=/usr/local/cuda-12.6
readonly CUDA_ARCH_FLAGS='-gencode arch=compute_90,code=sm_90'
readonly CUSPARSE_LIBRARY=${CUDA_TOOLKIT}/lib64/libcusparse.so.12
readonly EXPECTED_CUSPARSE_SHA256=58ffc54edb1d007f56a1718aaadcb30f45bbf662f43515920ea8ff094304bdbf
readonly EXPECTED_INTERPRETER_SHA256=1e43e23601e6369d52fd56b0405882463297c9ba6ad5238b460da90db6771b9c
readonly RELION_BIND_SOURCE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_coarse_atomic_multistream_crossed_5a7ff8ad0_h21g4_20260901/runtime/relion_bind/_relion_bind_core.cpython-311-x86_64-linux-gnu.so
readonly EXPECTED_RELION_BIND_SHA256=9bbb1fb0ce6fa7ac816598ec521453515d163221642b916e5715bb2850798980
readonly RELION_RUNTIME=/scratch/gpfs/GILLES/mg6942/recovar_dev/relion_k4_100k_dispatchv2_20260717
readonly MPI_ROOT=/usr/local/openmpi/cuda-12.6/4.1.6/gcc
readonly GF46_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_full_expansion_v3_984637b7d_87274be_20260826/vdam-gf46/repeat-01/vdam-gf46
readonly CHECKPOINT=${GF46_ROOT}/relion/run_it180_optimiser.star
readonly DATA_STAR=${GF46_ROOT}/relion/run_it180_data.star
readonly DATA_DIR=${GF46_ROOT}/data
readonly PARTICLE_STACK=${DATA_DIR}/particles.128.mrcs
readonly EXPECTED_INPUT_MANIFEST_SHA256=de224471a690d1faaae4067217dbcc90b632269d62b0b3372b20aafa69157d91
readonly EXPECTED_PARTICLE_STACK_SHA256=804af933bd315f41f0159f62e93867cf852d70cb29f2f27a525fb2fc3eb68ad9
readonly EXPECTED_SELECTED_IDS_INT64_SHA256=c0199226ec7aa92f74a2fd66660e597fe44d73155db20f026b278840992a90be

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
REPO_ROOT=$(realpath -e -- "${REPO_ROOT}")
RECOVAR_PIXI_PY=$(realpath -e -- "${RECOVAR_PIXI_PY}")
ROOT=${VDAM_GF46_HYBRID_TRANSITION_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf46_hybrid_transition_${SLURM_JOB_ID}_${EXPECTED_REPO_HEAD:0:9}}
ROOT=$(realpath -m -- "${ROOT}")
PROVENANCE=${ROOT}/provenance
ANALYSIS=${ROOT}/analysis
RUNTIME=${ROOT}/runtime
CUDA_BINARY=${RUNTIME}/cuda/libcuda_backproject.so
RELION_BIND_DIR=${RUNTIME}/relion_bind
RELION_BIND_BINARY=${RELION_BIND_DIR}/$(basename "${RELION_BIND_SOURCE}")
NVCC=${CUDA_TOOLKIT}/bin/nvcc

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
  local message="GF46 hybrid transition failed: status=${status} line=${BASH_LINENO[0]} command=${BASH_COMMAND}"
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
    "${RECOVAR_PIXI_PY}" - "${REPO_ROOT}" <<'PY'
import sys
from pathlib import Path

from scripts.analyze_vdam_gf46_hybrid_transition import _source_manifest

manifest, _entries = _source_manifest(Path(sys.argv[1]))
sys.stdout.buffer.write(manifest)
PY
  )
}

verify_frozen_inputs() {
  sha256sum -c "${PROVENANCE}/input_manifest.sha256" > /dev/null
  test "$(sha256sum "${PARTICLE_STACK}" | awk '{print $1}')" = \
    "${EXPECTED_PARTICLE_STACK_SHA256}"
}

verify_repo_and_runtime() {
  test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
  test "$(git -C "${REPO_ROOT}" rev-parse 'HEAD^{tree}')" = "${EXPECTED_REPO_TREE}"
  test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=all)"
  verify_frozen_inputs
  test "$(sha256sum "${RECOVAR_PIXI_PY}" | awk '{print $1}')" = \
    "${EXPECTED_INTERPRETER_SHA256}"
  test "$(sha256sum "${RELION_BIND_BINARY}" | awk '{print $1}')" = \
    "${EXPECTED_RELION_BIND_SHA256}"
  test "$(sha256sum "${CUSPARSE_LIBRARY}" | awk '{print $1}')" = \
    "${EXPECTED_CUSPARSE_SHA256}"
  test -s "${CUDA_BINARY}"
  vdam_verify_selected_gpu "${selected_gpu_uuid}"
}

[[ "${EXPECTED_REPO_HEAD}" =~ ^[0-9a-f]{40}$ ]]
[[ "${EXPECTED_REPO_TREE}" =~ ^[0-9a-f]{40}$ ]]
[[ "${EXPECTED_SOURCE_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]]
test -x "${RECOVAR_PIXI_PY}"
test "$(sha256sum "${RECOVAR_PIXI_PY}" | awk '{print $1}')" = \
  "${EXPECTED_INTERPRETER_SHA256}"
test -x "${NVCC}"
test -s "${CUSPARSE_LIBRARY}"
test "$(sha256sum "${CUSPARSE_LIBRARY}" | awk '{print $1}')" = \
  "${EXPECTED_CUSPARSE_SHA256}"
test -s "${RELION_BIND_SOURCE}"
test "$(sha256sum "${RELION_BIND_SOURCE}" | awk '{print $1}')" = \
  "${EXPECTED_RELION_BIND_SHA256}"
test "$(git -C "${REPO_ROOT}" rev-parse --show-toplevel)" = "${REPO_ROOT}"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${REPO_ROOT}" rev-parse 'HEAD^{tree}')" = "${EXPECTED_REPO_TREE}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=all)"
test ! -e "${ROOT}"
test ! -e "${RUNTIME}"
test -d "$(dirname "${ROOT}")"
case "${ROOT}/" in
  "${REPO_ROOT}/"*)
    echo "result root must be outside the repository" >&2
    exit 2
    ;;
esac
for input_file in "${INPUT_FILES[@]}"; do
  test -s "${input_file}"
done

initial_allocation_spec=${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}}
source "${REPO_ROOT}/scripts/vdam_gpu_selection.sh"
mapfile -t initially_visible_gpu_uuids < <(vdam_visible_gpu_uuids)
if [[ "${#initially_visible_gpu_uuids[@]}" -ne 1 ]]; then
  echo "GF46 transition requires exactly one cgroup-visible H100" >&2
  exit 76
fi
target_gpu_uuid=${initially_visible_gpu_uuids[0]}
vdam_assert_target_gpu_allocated "${target_gpu_uuid}" "${initial_allocation_spec}"
vdam_select_target_gpu "${target_gpu_uuid}" 0
selected_gpu_uuid=${VDAM_SELECTED_GPU_UUID}
gpu_name=$(nvidia-smi -i "${selected_gpu_uuid}" --query-gpu=name --format=csv,noheader | sed -n '1p' | sed 's/[[:space:]]*$//')
[[ "${gpu_name}" == *H100* ]]

mkdir -p "${PROVENANCE}" "${ANALYSIS}" "${RUNTIME}/cuda" \
  "${RELION_BIND_DIR}" "${RUNTIME}/tmp"
touch "${ROOT}/SAFE_TO_DELETE"
sha256sum "${INPUT_FILES[@]}" > "${PROVENANCE}/input_manifest.sha256"
input_manifest_sha256=$(sha256sum "${PROVENANCE}/input_manifest.sha256" | awk '{print $1}')
test "${input_manifest_sha256}" = "${EXPECTED_INPUT_MANIFEST_SHA256}"
write_source_manifest > "${PROVENANCE}/source_manifest.sha256"
source_manifest_sha256=$(sha256sum "${PROVENANCE}/source_manifest.sha256" | awk '{print $1}')
test "${source_manifest_sha256}" = "${EXPECTED_SOURCE_MANIFEST_SHA256}"

make -B -C "${REPO_ROOT}/recovar/cuda" \
  "PYTHON=${RECOVAR_PIXI_PY}" \
  "NVCC=${NVCC}" \
  "CUDA_ARCH=${CUDA_ARCH_FLAGS}" \
  "LIB=${CUDA_BINARY}" \
  > "${RUNTIME}/cuda_build.stdout" \
  2> "${RUNTIME}/cuda_build.stderr"
test -s "${CUDA_BINARY}"
nm -D --defined-only "${CUDA_BINARY}" > "${RUNTIME}/cuda_binary_symbols.txt"
rg -q 'RelionCoarseDiff2RectangularF32' "${RUNTIME}/cuda_binary_symbols.txt"
rg -q 'RelionCoarseDiff2RotationBlocksF32' "${RUNTIME}/cuda_binary_symbols.txt"
cp --reflink=auto "${RELION_BIND_SOURCE}" "${RELION_BIND_BINARY}"
cuda_sha256=$(sha256sum "${CUDA_BINARY}" | awk '{print $1}')

git -C "${REPO_ROOT}" rev-parse HEAD > "${PROVENANCE}/repo_head.txt"
git -C "${REPO_ROOT}" rev-parse 'HEAD^{tree}' > "${PROVENANCE}/repo_tree.txt"
git -C "${REPO_ROOT}" symbolic-ref --short HEAD > "${PROVENANCE}/repo_branch.txt"
git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=all > "${PROVENANCE}/repo_status.txt"
git -C "${REPO_ROOT}" diff --binary --no-ext-diff HEAD -- | sha256sum \
  > "${PROVENANCE}/repo_diff.sha256"
printf '%s\n' "${SLURM_JOB_ID}" > "${PROVENANCE}/slurm_job_id.txt"
printf '%s\n' "${SLURMD_NODENAME:-$(hostname -s)}" > "${PROVENANCE}/node.txt"
printf '%s\n' "${selected_gpu_uuid}" > "${PROVENANCE}/selected_gpu_uuid.txt"
printf '%s\n' "${gpu_name}" > "${PROVENANCE}/gpu_name.txt"
printf '%s\n' "${VDAM_ALLOCATED_GPU_UUIDS_CSV}" > "${PROVENANCE}/allocated_gpu_uuids.csv"
printf '%s\n' "${VDAM_VISIBLE_GPU_UUIDS_CSV}" > "${PROVENANCE}/visible_gpu_uuids.csv"
sha256sum "${RECOVAR_PIXI_PY}" > "${PROVENANCE}/interpreter.sha256"
sha256sum "${CUDA_BINARY}" > "${PROVENANCE}/cuda.sha256"
sha256sum "${RELION_BIND_BINARY}" > "${PROVENANCE}/relion_bind.sha256"
sha256sum "${CUSPARSE_LIBRARY}" > "${PROVENANCE}/cusparse.sha256"
nvidia-smi -i "${selected_gpu_uuid}" -q > "${PROVENANCE}/nvidia_smi.txt"
scontrol show job "${SLURM_JOB_ID}" > "${PROVENANCE}/slurm_job.txt"
env | LC_ALL=C sort > "${PROVENANCE}/submission_environment.txt"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV JAX_PLATFORM_NAME
unset RECOVAR_DISABLE_CUDA RECOVAR_CUDA_LIB RECOVAR_ENABLE_CUSTOM_CUDA
unset RECOVAR_REQUIRE_CUSTOM_CUDA_FOR_TESTS XLA_FLAGS JAX_ENABLE_X64
unset RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT
unset RECOVAR_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_DIR
unset RECOVAR_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_ORIGINAL_INDICES
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=${REPO_ROOT}
export CUDA_VISIBLE_DEVICES=${selected_gpu_uuid}
export RECOVAR_EXPECTED_REPO_ROOT=${REPO_ROOT}
export RECOVAR_CUDA_LIB=${CUDA_BINARY}
export RECOVAR_RELION_BIND_BUILD_DIR=${RELION_BIND_DIR}
export RECOVAR_REQUIRE_CUSTOM_CUDA_FOR_TESTS=1
export RECOVAR_ENABLE_CUSTOM_CUDA=1
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export TMPDIR=${RUNTIME}/tmp
export RECOVAR_CACHE_DIR=
export PIXI_HOME=${RUNTIME}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/rattler_cache
export PATH=${CUDA_TOOLKIT}/bin:${PATH}
export LD_LIBRARY_PATH=${RELION_RUNTIME}/system_dev_overlay/lib:${MPI_ROOT}/lib64:${CUDA_TOOLKIT}/lib64:${CUDA_TOOLKIT}/targets/x86_64-linux/lib
export LD_PRELOAD=${CUSPARSE_LIBRARY}
mkdir -p "${PIXI_HOME}" "${RATTLER_CACHE_DIR}"

(
  cd "${REPO_ROOT}"
  "${RECOVAR_PIXI_PY}" - <<'PY'
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
assert jax.default_backend() == "gpu"
assert len(jax.devices("gpu")) == 1
assert "H100" in str(jax.devices("gpu")[0].device_kind)
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
assert Path(cuda_backproject._loaded_lib_path).resolve() == cuda
assert callable(getattr(relion_bind_core, "vdam_rnd_unif_sequence", None))
assert_parity_ancestors()
print("GF46 hybrid transition runtime gate passed", jax.devices(), recovar.__file__)
PY
) > "${PROVENANCE}/runtime_gate.stdout" 2> "${PROVENANCE}/runtime_gate.stderr"
verify_repo_and_runtime

RUN_LABELS=(audit_direct audit_hybrid direct_1 hybrid_1 hybrid_2 direct_2)
RUN_HYBRID=(0 1 0 1 1 0)
RUN_AUDIT=(1 1 0 0 0 0)
test "${#RUN_LABELS[@]}" -eq "${#RUN_HYBRID[@]}"
test "${#RUN_LABELS[@]}" -eq "${#RUN_AUDIT[@]}"
printf 'order\tlabel\thybrid\tsupport_audit\n' > "${PROVENANCE}/execution_order.tsv"

for index in "${!RUN_LABELS[@]}"; do
  label=${RUN_LABELS[index]}
  hybrid=${RUN_HYBRID[index]}
  audit=${RUN_AUDIT[index]}
  run_root=${ROOT}/runs/${label}
  cache_root=${run_root}/jax_cache
  mkdir -p "${run_root}" "${cache_root}"
  touch "${cache_root}/SAFE_TO_DELETE"
  printf '%d\t%s\t%s\t%s\n' "$((index + 1))" "${label}" "${hybrid}" "${audit}" \
    >> "${PROVENANCE}/execution_order.tsv"
  if [[ "${hybrid}" == 1 ]]; then
    macro=1
    projection_cache=1
  else
    macro=0
    projection_cache=0
  fi
  audit_env=(-u RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT)
  if [[ "${audit}" == 1 ]]; then
    audit_env=(RECOVAR_COARSE_SIGNIFICANCE_SUPPORT_AUDIT=1)
  fi
  command=(
    env
    -u RECOVAR_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_DIR
    -u RECOVAR_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_ORIGINAL_INDICES
    "${audit_env[@]}"
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
    RECOVAR_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE_MAX_GB=4
    RECOVAR_COARSE_GAUSSIAN_GEMM_HYBRID_BLOCK_CAPACITY=64
    "RECOVAR_COARSE_GAUSSIAN_GEMM_MACRO=${macro}"
    "RECOVAR_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE=${projection_cache}"
    "RECOVAR_COARSE_GAUSSIAN_GEMM_HYBRID=${hybrid}"
    "JAX_COMPILATION_CACHE_DIR=${cache_root}"
    "${RECOVAR_PIXI_PY}" -m scripts.run_vdam_late_iteration_profile
    --checkpoint-optimiser "${CHECKPOINT}"
    --input-star "${DATA_STAR}"
    --data-dir "${DATA_DIR}"
    --output-root "${run_root}/profile"
    --checkpoint-iteration 180
    --nr-iter 200
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
  verify_repo_and_runtime
  if [[ "${index}" == 1 ]]; then
    touch "${ROOT}/AUDITS_COMPLETED"
  fi
done
touch "${ROOT}/ARMS_COMPLETED"

write_source_manifest > "${PROVENANCE}/source_manifest.final.sha256"
cmp "${PROVENANCE}/source_manifest.sha256" "${PROVENANCE}/source_manifest.final.sha256"

"${RECOVAR_PIXI_PY}" - "${PROVENANCE}/run.json" "${SLURM_JOB_ID}" \
  "${EXPECTED_REPO_HEAD}" "${EXPECTED_REPO_TREE}" "${source_manifest_sha256}" \
  "${input_manifest_sha256}" "${cuda_sha256}" "${selected_gpu_uuid}" \
  "${gpu_name}" "${SLURMD_NODENAME:-$(hostname -s)}" <<'PY'
import json
import sys
from pathlib import Path

payload = {
    "schema": "recovar.vdam_gf46_hybrid_transition.v1",
    "classification": "diagnostic_one_transition_only",
    "job_id": sys.argv[2],
    "git_head": sys.argv[3],
    "git_tree": sys.argv[4],
    "source_manifest_sha256": sys.argv[5],
    "source_manifest_scope": "selected_high_risk_files",
    "input_manifest_sha256": sys.argv[6],
    "cuda_sha256": sys.argv[7],
    "gpu_uuid": sys.argv[8],
    "gpu_name": sys.argv[9],
    "node": sys.argv[10],
    "particle_stack_sha256": "804af933bd315f41f0159f62e93867cf852d70cb29f2f27a525fb2fc3eb68ad9",
    "relion_bind_sha256": "9bbb1fb0ce6fa7ac816598ec521453515d163221642b916e5715bb2850798980",
    "interpreter_sha256": "1e43e23601e6369d52fd56b0405882463297c9ba6ad5238b460da90db6771b9c",
    "cusparse_sha256": "58ffc54edb1d007f56a1718aaadcb30f45bbf662f43515920ea8ff094304bdbf",
    "cuda_toolkit": "/usr/local/cuda-12.6",
    "cuda_arch": "sm_90",
    "execution_order": ["audit_direct", "audit_hybrid", "direct_1", "hybrid_1", "hybrid_2", "direct_2"],
    "support_audit_arms": ["audit_direct", "audit_hybrid"],
    "timing_arms": ["direct_1", "hybrid_1", "hybrid_2", "direct_2"],
    "checkpoint_iteration": 180,
    "profiled_iteration": 181,
    "nr_iter_schedule": 200,
    "random_seed": 29,
    "image_batch_size": 500,
    "effective_significance_image_batch_size": 187,
    "selected_particle_ids_int64_sha256": "c0199226ec7aa92f74a2fd66660e597fe44d73155db20f026b278840992a90be",
    "material_warm_wall_ratio": 0.90,
    "support_audit_timing_eligible": False,
    "clean_timing_support_audit_unset": True,
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
    JAX_COMPILATION_CACHE_DIR= \
    "${RECOVAR_PIXI_PY}" -m scripts.analyze_vdam_gf46_hybrid_transition \
    --root "${ROOT}" --repo "${REPO_ROOT}" \
    --output-json "${ANALYSIS}/report.json" \
    --output-markdown "${ANALYSIS}/report.md"
) > "${ANALYSIS}/analyzer.stdout" 2> "${ANALYSIS}/analyzer.stderr"
analyzer_status=$?
set -e
trap on_error ERR
printf '%s\n' "${analyzer_status}" > "${ANALYSIS}/analyzer_status.txt"
if [[ "${analyzer_status}" != 0 && "${analyzer_status}" != 1 ]]; then
  printf 'hybrid transition analyzer setup failed: %s\n' "${analyzer_status}" \
    > "${PROVENANCE}/failure.txt"
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

printf 'GF46 hybrid transition complete: %s (analyzer status %s)\n' \
  "${ROOT}" "${analyzer_status}"
if [[ "${analyzer_status}" == 1 ]]; then
  trap - ERR
  exit 1
fi
