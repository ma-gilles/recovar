#!/usr/bin/env bash
#SBATCH --job-name=vdam-same-state
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
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-same-state-%j.out

set -euo pipefail
unset PYTHONOPTIMIZE

: "${VDAM_SAME_STATE_ROOT:?set a new disposable result root}"
: "${VDAM_SAME_STATE_LAUNCH_MANIFEST:?set a reviewed true-200 launch manifest}"
: "${EXPECTED_REPO_HEAD:?pin the committed diagnostic head}"

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
REPO_ROOT=$(realpath -e -- "${REPO_ROOT}")
ROOT=$(realpath -m -- "${VDAM_SAME_STATE_ROOT}")
LAUNCH_MANIFEST=$(realpath -e -- "${VDAM_SAME_STATE_LAUNCH_MANIFEST}")
CHECKPOINT_ITERATION=${CHECKPOINT_ITERATION:-34}
CANDIDATE_MODE=${CANDIDATE_MODE:-hybrid}
TARGET_GPU_UUID=${TARGET_GPU_UUID:-}
NOISE_SPLIT_DIAGNOSTICS=${VDAM_SAME_STATE_NOISE_SPLIT_DIAGNOSTICS:-0}
PIXI_PY=$(jq -er '.files.interpreter.path' "${LAUNCH_MANIFEST}")
RELION_BIND_SOURCE=$(jq -er '.files.relion_bind.path' "${LAUNCH_MANIFEST}")
CUSPARSE_LIBRARY=$(jq -er '.files.cusparse_library.resolved_path' "${LAUNCH_MANIFEST}")
EXPECTED_RELION_BIND_SHA256=$(jq -er '.files.relion_bind.sha256' "${LAUNCH_MANIFEST}")
EXPECTED_CUSPARSE_SHA256=$(jq -er '.files.cusparse_library.sha256' "${LAUNCH_MANIFEST}")
EXPECTED_INTERPRETER_SHA256=$(jq -er '.files.interpreter.sha256' "${LAUNCH_MANIFEST}")
FIXTURE=$(jq -er '.fixture_dir' "${LAUNCH_MANIFEST}")
CUDA_BUILD_TOOLKIT=${CUDA_BUILD_TOOLKIT:-/usr/local/cuda-12.6}
CUDA_ARCH_FLAGS=${VDAM_SAME_STATE_CUDA_ARCH_FLAGS:--gencode arch=compute_90,code=sm_90}
NVCC=${CUDA_BUILD_TOOLKIT}/bin/nvcc
ACCEPTANCE=${REPO_ROOT}/scripts/vdam_coarse_combined_true200_acceptance.json
PROVENANCE=${ROOT}/provenance
RUNTIME=${ROOT}/runtime
SCIENCE=${ROOT}/science
CUDA_BINARY=${RUNTIME}/cuda/libcuda_backproject.so
RELION_BIND_DIR=${RUNTIME}/relion_bind
RELION_BIND_BINARY=${RELION_BIND_DIR}/$(basename "${RELION_BIND_SOURCE}")
monitor_pid=

on_error() {
  local status=$?
  if [[ -n "${monitor_pid}" ]]; then
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
  fi
  if [[ -d "${PROVENANCE}" ]]; then
    printf 'VDAM_SAME_STATE_FAILURE status=%s line=%s command=%s\n' \
      "${status}" "${BASH_LINENO[0]}" "${BASH_COMMAND}" \
      > "${PROVENANCE}/failure.txt"
  fi
  exit "${status}"
}
trap on_error ERR

[[ "${CHECKPOINT_ITERATION}" =~ ^[1-9][0-9]*$ ]]
[[ "${NOISE_SPLIT_DIAGNOSTICS}" =~ ^[01]$ ]]
case "${CANDIDATE_MODE}" in
  hybrid)
    expected_arm_order=direct_1,hybrid_1,hybrid_2,direct_2
    ;;
  flat_rows)
    expected_arm_order=direct_1,flat_rows_1,flat_rows_2,direct_2
    ;;
  packed_projection)
    expected_arm_order=direct_1,packed_projection_1,packed_projection_2,direct_2
    ;;
  packed_deferred)
    expected_arm_order=direct_1,packed_deferred_1,packed_deferred_2,direct_2
    ;;
  hybrid_packed_deferred)
    expected_arm_order=direct_1,hybrid_packed_deferred_1,hybrid_packed_deferred_2,direct_2
    ;;
  stable_shapes)
    expected_arm_order=stable_off_1,stable_on_1,stable_on_2,stable_off_2
    ;;
  stable_flat_capacity)
    expected_arm_order=stable_flat_off_1,stable_flat_on_1,stable_flat_on_2,stable_flat_off_2
    ;;
  *)
    echo "unsupported CANDIDATE_MODE: ${CANDIDATE_MODE}" >&2
    exit 2
    ;;
esac
test ! -e "${ROOT}"
test -x "${PIXI_PY}"
test -x "${NVCC}"
test -s "${RELION_BIND_SOURCE}"
test -s "${CUSPARSE_LIBRARY}"
test -s "${FIXTURE}/particles.star"
test -s "${FIXTURE}/particles.128.mrcs"
test -s "${ACCEPTANCE}"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=all)"
test "$(sha256sum "${PIXI_PY}" | awk '{print $1}')" = "${EXPECTED_INTERPRETER_SHA256}"
test "$(sha256sum "${RELION_BIND_SOURCE}" | awk '{print $1}')" = \
  "${EXPECTED_RELION_BIND_SHA256}"
test "$(sha256sum "${CUSPARSE_LIBRARY}" | awk '{print $1}')" = \
  "${EXPECTED_CUSPARSE_SHA256}"

initial_allocation_spec=${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}}
source "${REPO_ROOT}/scripts/vdam_gpu_selection.sh"
vdam_select_target_gpu "${TARGET_GPU_UUID}" 0
selected_gpu_uuid=${VDAM_SELECTED_GPU_UUID}
vdam_assert_target_gpu_allocated "${selected_gpu_uuid}" "${initial_allocation_spec}"

mkdir -p "${PROVENANCE}" "${RUNTIME}/cuda" "${RELION_BIND_DIR}" \
  "${RUNTIME}/tmp" "${RUNTIME}/pixi_home" "${RUNTIME}/rattler_cache"
touch "${ROOT}/SAFE_TO_DELETE" "${RUNTIME}/tmp/SAFE_TO_DELETE"
cp --reflink=auto "${RELION_BIND_SOURCE}" "${RELION_BIND_BINARY}"
chmod 0755 "${RELION_BIND_BINARY}"

"${NVCC}" --version > "${PROVENANCE}/nvcc_version.txt"
make -B -C "${REPO_ROOT}/recovar/cuda" \
  "PYTHON=${PIXI_PY}" \
  "NVCC=${NVCC}" \
  "CUDA_ARCH=${CUDA_ARCH_FLAGS}" \
  "LIB=${CUDA_BINARY}" \
  > "${PROVENANCE}/cuda_build.stdout" \
  2> "${PROVENANCE}/cuda_build.stderr"
test -s "${CUDA_BINARY}"
cuda_sha256=$(sha256sum "${CUDA_BINARY}" | awk '{print $1}')
printf '%s  %s\n' "${cuda_sha256}" "${CUDA_BINARY}" \
  > "${PROVENANCE}/qualified_cuda.sha256"

git -C "${REPO_ROOT}" rev-parse HEAD > "${PROVENANCE}/repo_head.txt"
git -C "${REPO_ROOT}" rev-parse 'HEAD^{tree}' > "${PROVENANCE}/repo_tree.txt"
git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=all \
  > "${PROVENANCE}/repo_status.txt"
printf '%s\n' "${SLURM_JOB_ID}" > "${PROVENANCE}/slurm_job_id.txt"
printf '%s\n' "${SLURMD_NODENAME:-$(hostname -s)}" > "${PROVENANCE}/node.txt"
printf '%s\n' "${selected_gpu_uuid}" > "${PROVENANCE}/selected_gpu_uuid.txt"
printf '%s\n' "${VDAM_VISIBLE_GPU_UUIDS_CSV}" > "${PROVENANCE}/visible_gpu_uuids.csv"
printf '%s\n' "${CANDIDATE_MODE}" > "${PROVENANCE}/candidate_mode.txt"
printf '%s\n' "${NOISE_SPLIT_DIAGNOSTICS}" \
  > "${PROVENANCE}/noise_split_diagnostics.txt"
printf '%s\n' "${CUDA_ARCH_FLAGS}" > "${PROVENANCE}/cuda_arch_flags.txt"
nvidia-smi -i "${selected_gpu_uuid}" -q > "${PROVENANCE}/nvidia_smi.txt"
scontrol show job "${SLURM_JOB_ID}" > "${PROVENANCE}/slurm_job.txt"
sha256sum \
  "${REPO_ROOT}/scripts/run_vdam_hybrid_same_state_transition.py" \
  "${REPO_ROOT}/scripts/run_vdam_hybrid_same_state_transition.sbatch" \
  "${REPO_ROOT}/recovar/em/dense_single_volume/helpers/coarse_gemm_hybrid.py" \
  "${REPO_ROOT}/recovar/em/dense_single_volume/helpers/flat_local_rows.py" \
  "${REPO_ROOT}/recovar/em/dense_single_volume/helpers/significance.py" \
  "${REPO_ROOT}/recovar/em/dense_single_volume/local_big_jit.py" \
  "${REPO_ROOT}/recovar/em/dense_single_volume/local_em_engine.py" \
  "${REPO_ROOT}/recovar/em/initial_model/dense_adapter.py" \
  "${REPO_ROOT}/recovar/em/initial_model/driver.py" \
  "${REPO_ROOT}/recovar/em/initial_model/iteration_loop.py" \
  "${REPO_ROOT}/recovar/cuda/cuda_backproject.cu" \
  "${ACCEPTANCE}" "${LAUNCH_MANIFEST}" "${PIXI_PY}" "${CUDA_BINARY}" \
  "${RELION_BIND_BINARY}" "${CUSPARSE_LIBRARY}" "${FIXTURE}/particles.star" \
  "${FIXTURE}/particles.128.mrcs" > "${PROVENANCE}/static_inputs.sha256"

mapfile -d '' -t fixed_environment < <(
  jq -j '.runtime_contract.fixed_science_environment | to_entries[] | .key, "=", .value, "\u0000"' \
    "${ACCEPTANCE}"
)
diagnostic_environment=()
if [[ "${NOISE_SPLIT_DIAGNOSTICS}" = 1 ]]; then
  # This enables aggregate A2/XA returns without requesting per-particle
  # dumps, so the production local big-JIT path remains active.
  diagnostic_environment+=(
    "RECOVAR_NOISE_DEBUG_DUMP_DIR=${RUNTIME}/noise_split_enabled"
  )
fi
command=(
  env -i
  "${fixed_environment[@]}"
  "${diagnostic_environment[@]}"
  "CUDA_VISIBLE_DEVICES=${selected_gpu_uuid}"
  "JAX_COMPILATION_CACHE_DIR=${RUNTIME}/jax_cache"
  "LD_PRELOAD=${CUSPARSE_LIBRARY}"
  "PYTHONPATH=${REPO_ROOT}"
  "RECOVAR_CACHE_DIR="
  "RECOVAR_CUDA_LIB=${CUDA_BINARY}"
  "RECOVAR_EXPECTED_REPO_ROOT=${REPO_ROOT}"
  "RECOVAR_RELION_BIND_BUILD_DIR=${RELION_BIND_DIR}"
  "RECOVAR_COARSE_GAUSSIAN_GEMM_HYBRID=0"
  "RECOVAR_COARSE_GAUSSIAN_GEMM_MACRO=0"
  "RECOVAR_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE=0"
  "RECOVAR_INITIAL_MODEL_FLAT_LOCAL_ROWS=0"
  "RECOVAR_INITIAL_MODEL_STABLE_FLAT_ROW_CAPACITY=0"
  "RECOVAR_INITIAL_MODEL_PACKED_LOCAL_PROJECTION=0"
  "RECOVAR_INITIAL_MODEL_DEFER_PACKED_VDAM=0"
  "TMPDIR=${RUNTIME}/tmp"
  "PIXI_HOME=${RUNTIME}/pixi_home"
  "RATTLER_CACHE_DIR=${RUNTIME}/rattler_cache"
  "${PIXI_PY}" -m scripts.run_vdam_hybrid_same_state_transition
  --fixture-dir "${FIXTURE}"
  --acceptance-config "${ACCEPTANCE}"
  --output-root "${SCIENCE}"
  --checkpoint-iteration "${CHECKPOINT_ITERATION}"
  --image-batch-size 500
  --candidate-mode "${CANDIDATE_MODE}"
)
printf '%q ' "${command[@]}" > "${PROVENANCE}/command.sh"
printf '\n' >> "${PROVENANCE}/command.sh"
env | LC_ALL=C sort > "${PROVENANCE}/submission_environment.txt"

mkdir -p "${RUNTIME}/jax_cache"
touch "${RUNTIME}/jax_cache/SAFE_TO_DELETE"
nvidia-smi -i "${selected_gpu_uuid}" \
  --query-gpu=timestamp,index,name,uuid,memory.used,memory.total,utilization.gpu \
  --format=csv -l 1 > "${PROVENANCE}/gpu_monitor.csv" &
monitor_pid=$!
(
  cd "${REPO_ROOT}"
  /usr/bin/time -v -o "${PROVENANCE}/process.time" \
    "${command[@]}"
) > "${PROVENANCE}/runner.stdout" 2> "${PROVENANCE}/runner.stderr"
kill "${monitor_pid}" 2>/dev/null || true
wait "${monitor_pid}" 2>/dev/null || true
monitor_pid=

test -s "${SCIENCE}/report.json"
test "$(jq -r '.schema' "${SCIENCE}/report.json")" = \
  'recovar.vdam_hybrid_same_state_transition.v3'
test "$(jq -r '.checkpoint_iteration' "${SCIENCE}/report.json")" = \
  "${CHECKPOINT_ITERATION}"
test "$(jq -r '.candidate_mode' "${SCIENCE}/report.json")" = \
  "${CANDIDATE_MODE}"
test "$(jq -r '.arm_order | join(",")' "${SCIENCE}/report.json")" = \
  "${expected_arm_order}"
test "$(sha256sum "${CUDA_BINARY}" | awk '{print $1}')" = "${cuda_sha256}"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=all)"
vdam_verify_selected_gpu "${selected_gpu_uuid}"
touch "${ROOT}/COMPLETED"
(
  cd "${ROOT}"
  find . -type f ! -name SHA256SUMS ! -name SHA256SUMS.sha256 -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum
) > "${ROOT}/SHA256SUMS"
sha256sum "${ROOT}/SHA256SUMS" > "${ROOT}/SHA256SUMS.sha256"
printf 'VDAM same-state transition complete: %s\n' "${ROOT}"
