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
MIRRORED_INCREMENTAL_PANELS=${VDAM_SAME_STATE_MIRRORED_INCREMENTAL_PANELS:-0}
MIRRORED_HYBRID_IMAGE_BATCH_PANELS=${VDAM_SAME_STATE_MIRRORED_HYBRID_IMAGE_BATCH_PANELS:-0}
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
[[ "${MIRRORED_INCREMENTAL_PANELS}" =~ ^[01]$ ]]
[[ "${MIRRORED_HYBRID_IMAGE_BATCH_PANELS}" =~ ^[01]$ ]]
test "$((MIRRORED_INCREMENTAL_PANELS + MIRRORED_HYBRID_IMAGE_BATCH_PANELS))" -le 1
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
  packed_final_noise)
    expected_arm_order=direct_1,packed_final_noise_1,packed_final_noise_2,direct_2
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
  compact_posterior)
    expected_arm_order=direct_1,compact_posterior_1,compact_posterior_2,direct_2
    ;;
  compact_packed_deferred)
    expected_arm_order=direct_1,compact_packed_deferred_1,compact_packed_deferred_2,direct_2
    ;;
  all_optimized)
    expected_arm_order=direct_1,all_optimized_1,all_optimized_2,direct_2
    ;;
  exact_coarse_single_translate)
    expected_arm_order=single_translate_off_1,single_translate_on_1,single_translate_on_2,single_translate_off_2
    ;;
  exact_compact_preprocess)
    expected_arm_order=compact_preprocess_off_1,compact_preprocess_on_1,compact_preprocess_on_2,compact_preprocess_off_2
    ;;
  fused_pair_fine_score)
    expected_arm_order=pair_fine_off_1,pair_fine_on_1,pair_fine_on_2,pair_fine_off_2
    ;;
  *)
    echo "unsupported CANDIDATE_MODE: ${CANDIDATE_MODE}" >&2
    exit 2
    ;;
esac
if [[ "${MIRRORED_INCREMENTAL_PANELS}" = 1 ]]; then
  test "${CANDIDATE_MODE}" = packed_final_noise
  expected_arm_order=direct_oracle,abba_packed_deferred_1,abba_packed_final_noise_1,abba_packed_final_noise_2,abba_packed_deferred_2,baab_packed_final_noise_1,baab_packed_deferred_1,baab_packed_deferred_2,baab_packed_final_noise_2
fi
if [[ "${MIRRORED_HYBRID_IMAGE_BATCH_PANELS}" = 1 ]]; then
  test "${CANDIDATE_MODE}" = all_optimized
  expected_arm_order=direct_oracle,abba_batch110_1,abba_batch200_1,abba_batch200_2,abba_batch110_2,baab_batch200_1,baab_batch110_1,baab_batch110_2,baab_batch200_2
fi
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
printf '%s\n' "${MIRRORED_INCREMENTAL_PANELS}" \
  > "${PROVENANCE}/mirrored_incremental_panels.txt"
printf '%s\n' "${MIRRORED_HYBRID_IMAGE_BATCH_PANELS}" \
  > "${PROVENANCE}/mirrored_hybrid_image_batch_panels.txt"
printf '%s\n' "${CUDA_ARCH_FLAGS}" > "${PROVENANCE}/cuda_arch_flags.txt"
nvidia-smi -i "${selected_gpu_uuid}" -q > "${PROVENANCE}/nvidia_smi.txt"
scontrol show job "${SLURM_JOB_ID}" > "${PROVENANCE}/slurm_job.txt"
sha256sum \
  "${REPO_ROOT}/scripts/run_vdam_hybrid_same_state_transition.py" \
  "${REPO_ROOT}/scripts/run_vdam_hybrid_same_state_transition.sbatch" \
  "${REPO_ROOT}/recovar/em/dense_single_volume/helpers/coarse_gemm_hybrid.py" \
  "${REPO_ROOT}/recovar/em/dense_single_volume/helpers/oversampling.py" \
  "${REPO_ROOT}/recovar/em/dense_single_volume/helpers/flat_local_rows.py" \
  "${REPO_ROOT}/recovar/em/dense_single_volume/helpers/fourier_window.py" \
  "${REPO_ROOT}/recovar/em/dense_single_volume/helpers/significance.py" \
  "${REPO_ROOT}/recovar/em/dense_single_volume/helpers/sparse_pass2_bucketed.py" \
  "${REPO_ROOT}/recovar/em/dense_single_volume/local_backprojection.py" \
  "${REPO_ROOT}/recovar/em/dense_single_volume/local_big_jit.py" \
  "${REPO_ROOT}/recovar/em/dense_single_volume/local_em_engine.py" \
  "${REPO_ROOT}/recovar/em/initial_model/dense_adapter.py" \
  "${REPO_ROOT}/recovar/em/initial_model/driver.py" \
  "${REPO_ROOT}/recovar/em/initial_model/iteration_loop.py" \
  "${REPO_ROOT}/recovar/cuda/cuda_backproject.cu" \
  "${REPO_ROOT}/recovar/cuda_backproject.py" \
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
  "RECOVAR_COARSE_GAUSSIAN_GEMM_COMPACT_POSTERIOR=0"
  "RECOVAR_COARSE_GAUSSIAN_GEMM_MACRO=0"
  "RECOVAR_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE=0"
  "RECOVAR_INITIAL_MODEL_FLAT_LOCAL_ROWS=0"
  "RECOVAR_INITIAL_MODEL_STABLE_FLAT_ROW_CAPACITY=0"
  "RECOVAR_INITIAL_MODEL_PACKED_LOCAL_PROJECTION=0"
  "RECOVAR_INITIAL_MODEL_DEFER_PACKED_VDAM=0"
  "RECOVAR_INITIAL_MODEL_PACKED_FINAL_NOISE=0"
  "RECOVAR_SAME_STATE_GPU_MONITOR_CSV=${PROVENANCE}/gpu_monitor.csv"
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
if [[ "${MIRRORED_INCREMENTAL_PANELS}" = 1 ]]; then
  command+=(--mirrored-incremental-panels)
fi
if [[ "${MIRRORED_HYBRID_IMAGE_BATCH_PANELS}" = 1 ]]; then
  command+=(--mirrored-hybrid-image-batch-panels)
fi
printf '%q ' "${command[@]}" > "${PROVENANCE}/command.sh"
printf '\n' >> "${PROVENANCE}/command.sh"
env | LC_ALL=C sort > "${PROVENANCE}/submission_environment.txt"

mkdir -p "${RUNTIME}/jax_cache"
touch "${RUNTIME}/jax_cache/SAFE_TO_DELETE"
stdbuf -oL nvidia-smi -i "${selected_gpu_uuid}" \
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

find "${RUNTIME}/jax_cache" -maxdepth 1 -type f -name '*-cache' -printf '%f\n' \
  | sed -E 's/-[0-9a-f]{64}-cache$//' \
  | LC_ALL=C sort | uniq -c | sort -nr \
  > "${PROVENANCE}/jax_cache_name_counts.txt"
grep -E \
  'coarse|prepare_relion|assemble_coarse|update_logsumexp|posterior|translate_score|run_local_bucket_big_jit|fine_diff2_fused_translate_runtime_(flat_rows|pairs)|relion_vdam_mstep_fused_projector|gather|where|take_along_axis' \
  "${PROVENANCE}/jax_cache_name_counts.txt" \
  > "${PROVENANCE}/pass1_jax_cache_name_counts.txt" || true

test -s "${SCIENCE}/report.json"
test "$(jq -r '.schema' "${SCIENCE}/report.json")" = \
  'recovar.vdam_hybrid_same_state_transition.v3'
test "$(jq -r '.checkpoint_iteration' "${SCIENCE}/report.json")" = \
  "${CHECKPOINT_ITERATION}"
test "$(jq -r '.candidate_mode' "${SCIENCE}/report.json")" = \
  "${CANDIDATE_MODE}"
test "$(jq -r '.mirrored_incremental_panels' "${SCIENCE}/report.json")" = \
  "$([[ "${MIRRORED_INCREMENTAL_PANELS}" = 1 ]] && printf true || printf false)"
test "$(jq -r '.mirrored_hybrid_image_batch_panels' "${SCIENCE}/report.json")" = \
  "$([[ "${MIRRORED_HYBRID_IMAGE_BATCH_PANELS}" = 1 ]] && printf true || printf false)"
test "$(jq -r '.arm_order | join(",")' "${SCIENCE}/report.json")" = \
  "${expected_arm_order}"
if [[ "${CANDIDATE_MODE}" == compact_posterior || \
      "${CANDIDATE_MODE}" == compact_packed_deferred || \
      "${CANDIDATE_MODE}" == all_optimized ]] && \
   [[ "${MIRRORED_HYBRID_IMAGE_BATCH_PANELS}" = 0 ]]; then
  jq -e '.compact_science_contract.hard_exact_contract_passed == true' \
    "${SCIENCE}/report.json" > /dev/null
  for arm in direct_1 direct_2; do
    jq -e --arg arm "${arm}" \
      '.arms[$arm].execution_contract
       | .environment_exact == true
         and .profile_checked == true
         and .compact_requested == false
         and .compact_effective == false
         and .profile_exact == true' \
      "${SCIENCE}/report.json" > /dev/null
  done
  for arm in "${CANDIDATE_MODE}_1" "${CANDIDATE_MODE}_2"; do
    jq -e --arg arm "${arm}" \
      '.arms[$arm].execution_contract
       | .environment_exact == true
         and .profile_checked == true
         and .compact_requested == true
         and .compact_effective == true
         and .profile_exact == true
         and (.hybrid_profiles | length) > 0' \
      "${SCIENCE}/report.json" > /dev/null
  done
  if [[ "${CANDIDATE_MODE}" == compact_packed_deferred || \
        "${CANDIDATE_MODE}" == all_optimized ]]; then
    for arm in "${CANDIDATE_MODE}_1" "${CANDIDATE_MODE}_2"; do
      jq -e --arg arm "${arm}" \
        '.arms[$arm].execution_contract.packed_deferred_effective == true' \
        "${SCIENCE}/report.json" > /dev/null
    done
  fi
fi
if [[ "${CANDIDATE_MODE}" == all_optimized && \
      "${MIRRORED_HYBRID_IMAGE_BATCH_PANELS}" = 0 ]]; then
  for arm in direct_1 direct_2; do
    jq -e --arg arm "${arm}" \
      '.arms[$arm]
       | .execution_contract.all_optimized_profile_exact == true
         and .execution_contract.all_optimized.enabled == false
         and (.execution_contract.all_optimized.enabled_seams | length) == 0
         and (.execution_contract.all_optimized.disabled_seams | length) == 10
         and .execution_contract.all_optimized.packed_final_noise.enabled == false
         and .stable_fourier_window_contract.enabled == false
         and .stable_flat_capacity_contract == null' \
      "${SCIENCE}/report.json" > /dev/null
  done
  for arm in all_optimized_1 all_optimized_2; do
    jq -e --arg arm "${arm}" \
      '.arms[$arm]
       | .execution_contract.all_optimized_profile_exact == true
         and .execution_contract.all_optimized.enabled == true
         and (.execution_contract.all_optimized.enabled_seams | length) == 10
         and (.execution_contract.all_optimized.disabled_seams | length) == 0
         and .execution_contract.all_optimized.packed_final_noise.enabled == true
         and ([.execution_contract.all_optimized.packed_final_noise.profiles[].sum_packed_final_noise_rows] | all(. > 0))
         and (.execution_contract.all_optimized.local_profiles | length) > 0
         and .stable_fourier_window_contract.enabled == true
         and .stable_flat_capacity_contract.enabled == true
         and .stable_flat_capacity_contract.strict_reduction_count == 0' \
      "${SCIENCE}/report.json" > /dev/null
  done
fi
if [[ "${CANDIDATE_MODE}" == exact_coarse_single_translate ]]; then
  test "$(jq -r '.prewarm | keys | join(",")' "${SCIENCE}/report.json")" = \
    skip_off,skip_on
  test "$(jq -r '.same_state_contract.baseline_trajectory_backend' "${SCIENCE}/report.json")" = \
    all_optimized_exact_coarse_single_translate_off
  test "$(jq -r '.same_state_contract.candidate_backend' "${SCIENCE}/report.json")" = \
    all_optimized_exact_coarse_single_translate_on
  jq -e '
    .compact_science_contract.hard_exact_contract_passed == true
      and .exact_coarse_single_translate_atomic_contract_passed == true
      and .compact_science_contract.accumulator_repeat_envelope.all_cross_within_observed_repeat_envelope == true
      and .compact_science_contract.final_state_repeat_envelope.all_cross_within_observed_repeat_envelope == true
      and .same_state_contract.both_compared_backends_prewarmed == true
      and .same_state_contract.all_optimized_profile_fail_closed == true
      and .same_state_contract.exact_coarse_single_translate_profile_fail_closed == true
      and .exact_coarse_single_translate_runtime_contract.skip_off.repeat_count == 2
      and .exact_coarse_single_translate_runtime_contract.skip_on.repeat_count == 2
      and ([.exact_coarse_single_translate_runtime_contract.skip_off.measurements[],
            .exact_coarse_single_translate_runtime_contract.skip_on.measurements[]]
           | all(length == 2 and all(. > 0)))
      and .exact_coarse_single_translate_runtime_contract.isolated_environment_contract.all_other_environment_exact == true
      and .exact_coarse_single_translate_runtime_contract.isolated_environment_contract.profile_free_wall_timing == true
      and .exact_coarse_single_translate_runtime_contract.isolated_environment_contract.profile_counter_device_synchronization == false
      and .prewarm.skip_off.exact_coarse_single_translate.profile_exact == true
      and .prewarm.skip_off.exact_coarse_single_translate.expected_calls_per_batch == 2
      and .prewarm.skip_on.exact_coarse_single_translate.profile_exact == true
      and .prewarm.skip_on.exact_coarse_single_translate.expected_calls_per_batch == 1
  ' "${SCIENCE}/report.json" >/dev/null
  for arm in single_translate_off_1 single_translate_off_2; do
    jq -e --arg arm "${arm}" '
      .arms[$arm]
      | .candidate_enabled == false
        and .execution_contract.all_optimized.enabled == true
        and .execution_contract.compact_requested == true
        and .execution_contract.compact_effective == true
        and .execution_contract.exact_coarse_single_translate.enabled == false
        and .execution_contract.exact_coarse_single_translate.profile_enabled == false
        and .execution_contract.exact_coarse_single_translate.profile_checked == false
        and .execution_contract.exact_coarse_single_translate.profile_free_wall_timing == true
    ' "${SCIENCE}/report.json" >/dev/null
  done
  for arm in single_translate_on_1 single_translate_on_2; do
    jq -e --arg arm "${arm}" '
      .arms[$arm]
      | .candidate_enabled == true
        and .execution_contract.all_optimized.enabled == true
        and .execution_contract.compact_requested == true
        and .execution_contract.compact_effective == true
        and .execution_contract.exact_coarse_single_translate.enabled == true
        and .execution_contract.exact_coarse_single_translate.profile_enabled == false
        and .execution_contract.exact_coarse_single_translate.profile_checked == false
        and .execution_contract.exact_coarse_single_translate.profile_free_wall_timing == true
    ' "${SCIENCE}/report.json" >/dev/null
  done
  jq -e '
    [.prewarm.skip_off.exact_coarse_single_translate.profiles[]
     | .skip_generic_effective == false
       and .generic_assembly_count == .batch_count
       and .exact_assembly_count == .batch_count
       and .translate_score_call_count == (2 * .batch_count)] | all
  ' "${SCIENCE}/report.json" >/dev/null
  jq -e '
    [.prewarm.skip_on.exact_coarse_single_translate.profiles[]
     | .skip_generic_effective == true
       and .generic_assembly_count == 0
       and .exact_assembly_count == .batch_count
       and .translate_score_call_count == .batch_count] | all
  ' "${SCIENCE}/report.json" >/dev/null
fi
if [[ "${CANDIDATE_MODE}" == exact_compact_preprocess ]]; then
  test "$(jq -r '.prewarm | keys | join(",")' "${SCIENCE}/report.json")" = \
    preprocess_off,preprocess_on
  test "$(jq -r '.same_state_contract.baseline_trajectory_backend' "${SCIENCE}/report.json")" = \
    all_optimized_exact_compact_preprocess_off
  test "$(jq -r '.same_state_contract.candidate_backend' "${SCIENCE}/report.json")" = \
    all_optimized_exact_compact_preprocess_on
  jq -e '
    .compact_science_contract.hard_exact_contract_passed == true
      and .exact_compact_preprocess_atomic_contract_passed == true
      and .compact_science_contract.accumulator_repeat_envelope.all_cross_within_observed_repeat_envelope == true
      and .compact_science_contract.final_state_repeat_envelope.all_cross_within_observed_repeat_envelope == true
      and .same_state_contract.both_compared_backends_prewarmed == true
      and .same_state_contract.all_optimized_profile_fail_closed == true
      and .same_state_contract.exact_compact_preprocess_profile_fail_closed == true
      and .exact_compact_preprocess_runtime_contract.preprocess_off.repeat_count == 2
      and .exact_compact_preprocess_runtime_contract.preprocess_on.repeat_count == 2
      and ([.exact_compact_preprocess_runtime_contract.preprocess_off.measurements[],
            .exact_compact_preprocess_runtime_contract.preprocess_on.measurements[]]
           | all(length == 2 and all(. > 0)))
      and .exact_compact_preprocess_runtime_contract.isolated_environment_contract.all_other_environment_exact == true
      and .exact_compact_preprocess_runtime_contract.isolated_environment_contract.profile_free_wall_timing == true
      and .exact_compact_preprocess_runtime_contract.isolated_environment_contract.profile_counter_device_synchronization == false
      and .prewarm.preprocess_off.exact_compact_preprocess.profile_exact == true
      and .prewarm.preprocess_off.exact_compact_preprocess.expected_process_half_image_calls_per_batch == 2
      and .prewarm.preprocess_on.exact_compact_preprocess.profile_exact == true
      and .prewarm.preprocess_on.exact_compact_preprocess.expected_process_half_image_calls_per_batch == 1
  ' "${SCIENCE}/report.json" >/dev/null
  for arm in compact_preprocess_off_1 compact_preprocess_off_2; do
    jq -e --arg arm "${arm}" '
      .arms[$arm]
      | .candidate_enabled == false
        and .execution_contract.all_optimized.enabled == true
        and .execution_contract.compact_requested == true
        and .execution_contract.compact_effective == true
        and .execution_contract.exact_compact_preprocess.enabled == false
        and .execution_contract.exact_compact_preprocess.profile_enabled == false
        and .execution_contract.exact_compact_preprocess.profile_checked == false
        and .execution_contract.exact_compact_preprocess.profile_free_wall_timing == true
    ' "${SCIENCE}/report.json" >/dev/null
  done
  for arm in compact_preprocess_on_1 compact_preprocess_on_2; do
    jq -e --arg arm "${arm}" '
      .arms[$arm]
      | .candidate_enabled == true
        and .execution_contract.all_optimized.enabled == true
        and .execution_contract.compact_requested == true
        and .execution_contract.compact_effective == true
        and .execution_contract.exact_compact_preprocess.enabled == true
        and .execution_contract.exact_compact_preprocess.profile_enabled == false
        and .execution_contract.exact_compact_preprocess.profile_checked == false
        and .execution_contract.exact_compact_preprocess.profile_free_wall_timing == true
    ' "${SCIENCE}/report.json" >/dev/null
  done
  jq -e '
    [.prewarm.preprocess_off.exact_compact_preprocess.profiles[]
     | .exact_compact_preprocess_effective == false
       and .generic_score_preprocess_count == .batch_count
       and .exact_source_preprocess_count == .batch_count
       and .generic_ctf_evaluation_count == .batch_count
       and .generic_full_translation_count == .batch_count
       and .translate_score_call_count == .batch_count] | all
  ' "${SCIENCE}/report.json" >/dev/null
  jq -e '
    [.prewarm.preprocess_on.exact_compact_preprocess.profiles[]
     | .exact_compact_preprocess_effective == true
       and .generic_score_preprocess_count == 0
       and .exact_source_preprocess_count == .batch_count
       and .generic_ctf_evaluation_count == 0
       and .generic_full_translation_count == 0
       and .generic_fallback_policy == "raise_before_generic_score_fallback"
       and .translate_score_call_count == .batch_count] | all
  ' "${SCIENCE}/report.json" >/dev/null
fi
if [[ "${CANDIDATE_MODE}" == fused_pair_fine_score ]]; then
  test "$(jq -r '.prewarm | keys | join(",")' "${SCIENCE}/report.json")" = \
    pair_off,pair_on
  test "$(jq -r '.same_state_contract.baseline_trajectory_backend' "${SCIENCE}/report.json")" = \
    all_optimized_fused_pair_fine_score_off
  test "$(jq -r '.same_state_contract.candidate_backend' "${SCIENCE}/report.json")" = \
    all_optimized_fused_pair_fine_score_on
  jq -e '
    .compact_science_contract.hard_exact_contract_passed == true
      and .fused_pair_fine_atomic_contract_passed == true
      and .compact_science_contract.atomic_repeat_envelope_passed == true
      and .compact_science_contract.accumulator_repeat_envelope.all_cross_within_observed_repeat_envelope == true
      and .compact_science_contract.scalar_meta_repeat_envelope.all_cross_within_observed_repeat_envelope == true
      and .compact_science_contract.final_state_repeat_envelope.all_cross_within_observed_repeat_envelope == true
      and .same_state_contract.both_compared_backends_prewarmed == true
      and .same_state_contract.all_optimized_profile_fail_closed == true
      and .same_state_contract.fused_pair_fine_score_profile_fail_closed == true
      and .same_state_contract.persistent_cache_family_counts_recorded == true
      and .same_state_contract.gpu_hbm_samples_recorded == true
      and .gpu_memory.available == true
      and .gpu_memory.all_arms_sampled == true
      and .fused_pair_fine_runtime_contract.pair_off.repeat_count == 2
      and .fused_pair_fine_runtime_contract.pair_on.repeat_count == 2
      and .fused_pair_fine_runtime_contract.material_speedup_passed == true
      and .fused_pair_fine_runtime_contract.persistent_cache_contract.timed_arms_add_no_target_family_programs_after_prewarm == true
      and ([.fused_pair_fine_runtime_contract.pair_off.measurements[],
            .fused_pair_fine_runtime_contract.pair_on.measurements[]]
           | all(length == 2 and all(. >= 0)))
      and .fused_pair_fine_runtime_contract.isolated_environment_contract.all_other_environment_exact == true
      and .fused_pair_fine_runtime_contract.isolated_environment_contract.profile_free_wall_timing == true
      and .fused_pair_fine_runtime_contract.isolated_environment_contract.profile_counter_device_synchronization == false
      and .prewarm.pair_off.fused_pair_fine_score.profile_exact == true
      and .prewarm.pair_off.fused_pair_fine_score.enabled == false
      and .prewarm.pair_on.fused_pair_fine_score.profile_exact == true
      and .prewarm.pair_on.fused_pair_fine_score.enabled == true
      and .prewarm.pair_on.fused_pair_fine_score.compile_shape_capacity_family_count > 0
  ' "${SCIENCE}/report.json" >/dev/null
  for arm in pair_fine_off_1 pair_fine_off_2; do
    jq -e --arg arm "${arm}" '
      .arms[$arm]
      | .candidate_enabled == false
        and .execution_contract.all_optimized.enabled == true
        and .execution_contract.compact_requested == true
        and .execution_contract.compact_effective == true
        and .execution_contract.fused_pair_fine_score.enabled == false
        and .execution_contract.fused_pair_fine_score.profile_exact == true
        and .execution_contract.fused_pair_fine_score.compile_shape_capacity_family_count == 0
        and .persistent_cache.delta.no_new_target_family_programs == true
        and .gpu_memory.sample_count > 0
    ' "${SCIENCE}/report.json" >/dev/null
  done
  for arm in pair_fine_on_1 pair_fine_on_2; do
    jq -e --arg arm "${arm}" '
      .arms[$arm]
      | .candidate_enabled == true
        and .execution_contract.all_optimized.enabled == true
        and .execution_contract.compact_requested == true
        and .execution_contract.compact_effective == true
        and .execution_contract.fused_pair_fine_score.enabled == true
        and .execution_contract.fused_pair_fine_score.profile_exact == true
        and .execution_contract.fused_pair_fine_score.shared_compact_pair_order == true
        and .execution_contract.fused_pair_fine_score.pair_pixel_gathers_materialized == false
        and .execution_contract.fused_pair_fine_score.dense_posterior_order_restored == true
        and .execution_contract.fused_pair_fine_score.compile_shape_capacity_family_count > 0
        and .execution_contract.fused_pair_fine_score.sum_candidates > 0
        and .execution_contract.fused_pair_fine_score.sum_candidates <= .execution_contract.fused_pair_fine_score.sum_capacity
        and .execution_contract.fused_pair_fine_score.sum_capacity <= .execution_contract.fused_pair_fine_score.sum_dense_capacity
        and .persistent_cache.delta.no_new_target_family_programs == true
        and .gpu_memory.sample_count > 0
    ' "${SCIENCE}/report.json" >/dev/null
  done
fi
if [[ "${MIRRORED_HYBRID_IMAGE_BATCH_PANELS}" = 1 ]]; then
  test "$(jq -r '.prewarm | keys | join(",")' "${SCIENCE}/report.json")" = \
    batch110,batch200
  test "$(jq -r '.same_state_contract.baseline_trajectory_backend' "${SCIENCE}/report.json")" = \
    all_optimized_hybrid_image_batch_110
  test "$(jq -r '.same_state_contract.candidate_backend' "${SCIENCE}/report.json")" = \
    all_optimized_hybrid_image_batch_200
  test "$(jq -r '.same_state_contract.both_compared_backends_prewarmed' "${SCIENCE}/report.json")" = \
    true
  test "$(jq -r '.same_state_contract.direct_oracle_from_shared_checkpoint' "${SCIENCE}/report.json")" = \
    true
  jq -e '
    .hybrid_image_batch_science_contract
    | .hard_exact_contract_passed == true
      and .atomic_repeat_envelope_passed == true
      and (.exact_pair_checks | length) == 36
      and .accumulator_repeat_envelope.control_repeat_pair_count == 6
      and .accumulator_repeat_envelope.candidate_repeat_pair_count == 6
      and .accumulator_repeat_envelope.cross_pair_count == 16
      and .accumulator_repeat_envelope.all_cross_within_observed_repeat_envelope == true
      and .final_state_repeat_envelope.control_repeat_pair_count == 6
      and .final_state_repeat_envelope.candidate_repeat_pair_count == 6
      and .final_state_repeat_envelope.cross_pair_count == 16
      and .final_state_repeat_envelope.all_cross_within_observed_repeat_envelope == true
  ' "${SCIENCE}/report.json" >/dev/null
  jq -e '
    .hybrid_image_batch_runtime_contract
    | .batch110.repeat_count == 4
      and .batch200.repeat_count == 4
      and ([.batch110.measurements[], .batch200.measurements[]]
           | all(length == 4 and all(. > 0)))
  ' "${SCIENCE}/report.json" >/dev/null
  jq -e --arg env_name RECOVAR_COARSE_GAUSSIAN_GEMM_HYBRID_IMAGE_BATCH_SIZE '
    .arms.direct_oracle
    | .candidate_enabled == false
      and .backend_mode == "direct"
      and .execution_contract.environment_exact == true
      and (.execution_contract.requested_environment | has($env_name) | not)
      and (.execution_contract.effective_environment | has($env_name) | not)
      and .execution_contract.compact_requested == false
      and .execution_contract.compact_effective == false
      and .execution_contract.all_optimized_profile_exact == true
      and .execution_contract.all_optimized.enabled == false
      and (.execution_contract.all_optimized.disabled_seams | length) == 10
  ' "${SCIENCE}/report.json" >/dev/null
  for arm in \
    abba_batch110_1 abba_batch110_2 \
    baab_batch110_1 baab_batch110_2; do
    jq -e --arg arm "${arm}" '
      .arms[$arm]
      | .candidate_enabled == true
        and .backend_mode == "all_optimized"
        and .execution_contract.environment_exact == true
        and .execution_contract.effective_environment.RECOVAR_COARSE_GAUSSIAN_GEMM_HYBRID_IMAGE_BATCH_SIZE == "110"
        and .execution_contract.compact_requested == true
        and .execution_contract.compact_effective == true
        and .execution_contract.all_optimized_profile_exact == true
        and .execution_contract.all_optimized.enabled == true
        and (.execution_contract.all_optimized.enabled_seams | length) == 10
        and (.execution_contract.all_optimized.disabled_seams | length) == 0
        and .execution_contract.hybrid_image_batch.requested_batch_size == 110
        and .execution_contract.hybrid_image_batch.effective_batch_size == 110
        and .execution_contract.hybrid_image_batch.profile_exact == true
        and ([.execution_contract.hybrid_image_batch.profiles[]
              | .input_image_batch_size == 110
                and .requested_hybrid_image_batch_size == 110
                and .effective_image_batch_size == 110
                and .streamed_certificate_candidate_count_at_effective_batch == 24837120
                and .batch_count == 2
                and .selected_rescore_batch_count == 2
                and .fallback_batch_count == 0
                and .fallback_image_count == 0
                and .certificate_chunk_count_per_batch == 8
                and .certificate_chunk_rows == 4608
                and .selected_rescore_image_count == 200
                and .actual_image_batch_sizes == [110, 90]
                and .physical_image_batch_sizes == [110, 110]] | all)
    ' "${SCIENCE}/report.json" >/dev/null
  done
  for arm in \
    abba_batch200_1 abba_batch200_2 \
    baab_batch200_1 baab_batch200_2; do
    jq -e --arg arm "${arm}" '
      .arms[$arm]
      | .candidate_enabled == true
        and .backend_mode == "all_optimized"
        and .execution_contract.environment_exact == true
        and .execution_contract.effective_environment.RECOVAR_COARSE_GAUSSIAN_GEMM_HYBRID_IMAGE_BATCH_SIZE == "500"
        and .execution_contract.compact_requested == true
        and .execution_contract.compact_effective == true
        and .execution_contract.all_optimized_profile_exact == true
        and .execution_contract.all_optimized.enabled == true
        and (.execution_contract.all_optimized.enabled_seams | length) == 10
        and (.execution_contract.all_optimized.disabled_seams | length) == 0
        and .execution_contract.hybrid_image_batch.requested_batch_size == 500
        and .execution_contract.hybrid_image_batch.effective_batch_size == 200
        and .execution_contract.hybrid_image_batch.profile_exact == true
        and ([.execution_contract.hybrid_image_batch.profiles[]
              | .input_image_batch_size == 110
                and .requested_hybrid_image_batch_size == 500
                and .effective_image_batch_size == 200
                and .streamed_certificate_candidate_count_at_effective_batch == 45158400
                and .batch_count == 1
                and .selected_rescore_batch_count == 1
                and .fallback_batch_count == 0
                and .fallback_image_count == 0
                and .certificate_chunk_count_per_batch == 8
                and .certificate_chunk_rows == 4608
                and .selected_rescore_image_count == 200
                and .actual_image_batch_sizes == [200]
                and .physical_image_batch_sizes == [200]] | all)
    ' "${SCIENCE}/report.json" >/dev/null
  done
fi
if [[ "${MIRRORED_INCREMENTAL_PANELS}" = 1 ]]; then
  test "$(jq -r '.prewarm | keys | join(",")' "${SCIENCE}/report.json")" = \
    packed_deferred,packed_final_noise
  test "$(jq -r '.same_state_contract.baseline_trajectory_backend' "${SCIENCE}/report.json")" = \
    packed_deferred
  test "$(jq -r '.same_state_contract.both_incremental_backends_prewarmed' "${SCIENCE}/report.json")" = \
    true
  test "$(jq -r '.same_state_contract.direct_oracle_from_shared_checkpoint' "${SCIENCE}/report.json")" = \
    true
  test "$(jq -r '.arms.direct_oracle.backend_mode' "${SCIENCE}/report.json")" = \
    direct
  for arm in direct_oracle; do
    jq -e --arg arm "${arm}" '
      .arms[$arm].execution_contract
      | .profile_checked == true
        and .packed_final_noise.backend_mode == "direct"
        and .packed_final_noise.enabled == false
        and .packed_final_noise.profile_exact == true
        and ([.packed_final_noise.profiles[].sum_packed_final_noise_rows] | all(. == 0))
    ' "${SCIENCE}/report.json" >/dev/null
  done
  for arm in \
    abba_packed_deferred_1 abba_packed_deferred_2 \
    baab_packed_deferred_1 baab_packed_deferred_2; do
    jq -e --arg arm "${arm}" '
      .arms[$arm].execution_contract
      | .profile_checked == true
        and .packed_final_noise.backend_mode == "packed_deferred"
        and .packed_final_noise.enabled == false
        and .packed_final_noise.profile_exact == true
        and ([.packed_final_noise.profiles[].sum_packed_final_noise_rows] | all(. == 0))
    ' "${SCIENCE}/report.json" >/dev/null
  done
  for arm in \
    abba_packed_final_noise_1 abba_packed_final_noise_2 \
    baab_packed_final_noise_1 baab_packed_final_noise_2; do
    jq -e --arg arm "${arm}" '
      .arms[$arm].execution_contract
      | .profile_checked == true
        and .packed_final_noise.backend_mode == "packed_final_noise"
        and .packed_final_noise.enabled == true
        and .packed_final_noise.profile_exact == true
        and ([.packed_final_noise.profiles[].sum_packed_final_noise_rows] | all(. > 0))
    ' "${SCIENCE}/report.json" >/dev/null
  done
  for oracle_pair in \
    direct_oracle__vs__abba_packed_deferred_1 \
    direct_oracle__vs__abba_packed_final_noise_1; do
    jq -e --arg pair "${oracle_pair}" '
      .comparisons[$pair]
      | .support_audits.exact_equal
        and ([.estep_meta[].exact_equal] | all)
        and ([.particle_state[].exact_equal] | all)
        and ([.sampling_state[].exact_equal] | all)
        and ([
          .final_state.Mavg.exact_equal,
          .final_state.ave_Pmax.exact_equal,
          .final_state.pdf_class.exact_equal,
          .final_state.pdf_direction.exact_equal,
          .final_state.sigma2_offset.exact_equal,
          .final_state.tau2_class.exact_equal,
          .final_state.fsc_halves_class.exact_equal,
          .final_state.current_resolution.exact_equal,
          .final_state.current_size.exact_equal
        ] | all)
    ' "${SCIENCE}/report.json" >/dev/null
  done
fi
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
