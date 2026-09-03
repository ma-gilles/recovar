#!/usr/bin/env bash
#SBATCH --job-name=vdam-stable-traj
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --nodelist=della-h19g2
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-stable-traj-%j.out

set -euo pipefail

: "${VDAM_STABLE_TRAJECTORY_ROOT:?set a fresh disposable output root}"
: "${VDAM_STABLE_TRAJECTORY_REPO_ROOT:?set the stable-shape worktree}"
: "${VDAM_STABLE_TRAJECTORY_EXPECTED_HEAD:?pin the committed worktree HEAD}"
: "${VDAM_STABLE_TRAJECTORY_CUDA:?set the qualified stable-shape CUDA artifact}"
: "${VDAM_STABLE_TRAJECTORY_RELION_BIND:?set the sealed RELION binding artifact}"
: "${SLURM_JOB_ID:?this gate must run under Slurm}"

ROOT=${VDAM_STABLE_TRAJECTORY_ROOT}
REPO_ROOT=$(realpath "${VDAM_STABLE_TRAJECTORY_REPO_ROOT}")
EXPECTED_HEAD=${VDAM_STABLE_TRAJECTORY_EXPECTED_HEAD}
CUDA_SOURCE=$(realpath "${VDAM_STABLE_TRAJECTORY_CUDA}")
RELION_BIND_SOURCE=$(realpath "${VDAM_STABLE_TRAJECTORY_RELION_BIND}")
FIXTURE=${VDAM_STABLE_TRAJECTORY_FIXTURE:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_full_expansion_v3_984637b7d_87274be_20260826/vdam-gf46/repeat-01/vdam-gf46/data}
LAST_ITERATION=${VDAM_STABLE_TRAJECTORY_LAST_ITERATION:-50}
STABLE_FOURIER_WINDOW_SHAPES=${VDAM_STABLE_TRAJECTORY_STABLE_FOURIER_WINDOW_SHAPES:-1}
STABLE_FOURIER_WINDOW_QUANTUM=${VDAM_STABLE_TRAJECTORY_STABLE_FOURIER_WINDOW_QUANTUM:-8}
STABLE_FLAT_ROW_CAPACITY=${VDAM_STABLE_TRAJECTORY_STABLE_FLAT_ROW_CAPACITY:-0}
ACCEPTANCE=${REPO_ROOT}/scripts/vdam_coarse_combined_true200_acceptance.json
ANALYZER=${REPO_ROOT}/scripts/analyze_vdam_stable_shape_trajectory.py
GPU_SELECTION_HELPER=${REPO_ROOT}/scripts/vdam_gpu_selection.sh
PIXI_PY=${REPO_ROOT}/.pixi/envs/default/bin/python
CUSPARSE=/usr/local/cuda-12.6/targets/x86_64-linux/lib/libcusparse.so.12.5.4.2
CUDA_BINARY=${ROOT}/runtime/cuda/libcuda_backproject.so
RELION_BIND_DIR=${ROOT}/runtime/relion_bind
RELION_BIND_BINARY=${RELION_BIND_DIR}/$(basename "${RELION_BIND_SOURCE}")
monitor_pid=

on_error() {
  status=$?
  if [[ -n "${monitor_pid}" ]]; then
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
  fi
  if [[ -d "${ROOT}" ]]; then
    printf 'status=%s line=%s command=%s\n' \
      "${status}" "${BASH_LINENO[0]}" "${BASH_COMMAND}" > "${ROOT}/HARNESS_FAILED" || true
  fi
  exit "${status}"
}
trap on_error ERR

test ! -e "${ROOT}"
test -d "${REPO_ROOT}/.git" -o -f "${REPO_ROOT}/.git"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1)"
test -x "${PIXI_PY}"
test -s "${CUDA_SOURCE}"
test -s "${RELION_BIND_SOURCE}"
test -s "${FIXTURE}/particles.star"
test -s "${FIXTURE}/particles.128.mrcs"
test -s "${ACCEPTANCE}"
test -s "${ANALYZER}"
test -s "${GPU_SELECTION_HELPER}"
test "${LAST_ITERATION}" -ge 2
test "${LAST_ITERATION}" -le 200
case "${STABLE_FOURIER_WINDOW_SHAPES}" in
  0|1) ;;
  *)
    printf 'VDAM_STABLE_TRAJECTORY_STABLE_FOURIER_WINDOW_SHAPES must be 0 or 1, got %s\n' \
      "${STABLE_FOURIER_WINDOW_SHAPES}" >&2
    exit 2
    ;;
esac
case "${STABLE_FLAT_ROW_CAPACITY}" in
  0|1) ;;
  *)
    printf 'VDAM_STABLE_TRAJECTORY_STABLE_FLAT_ROW_CAPACITY must be 0 or 1, got %s\n' \
      "${STABLE_FLAT_ROW_CAPACITY}" >&2
    exit 2
    ;;
esac
if ! [[ "${STABLE_FOURIER_WINDOW_QUANTUM}" =~ ^[0-9]+$ ]] \
  || (( STABLE_FOURIER_WINDOW_QUANTUM < 2 )) \
  || (( STABLE_FOURIER_WINDOW_QUANTUM % 2 != 0 )); then
  printf 'VDAM_STABLE_TRAJECTORY_STABLE_FOURIER_WINDOW_QUANTUM must be an even integer >= 2, got %s\n' \
    "${STABLE_FOURIER_WINDOW_QUANTUM}" >&2
  exit 2
fi
if [[ "${STABLE_FOURIER_WINDOW_SHAPES}" == 0 && "${STABLE_FLAT_ROW_CAPACITY}" == 0 ]]; then
  printf 'candidate must enable at least one stable ABI\n' >&2
  exit 2
fi

mkdir -p "${ROOT}/provenance" "${ROOT}/runtime/cuda" "${RELION_BIND_DIR}" \
  "${ROOT}/runs" "${ROOT}/analysis"
touch "${ROOT}/SAFE_TO_DELETE"
cp --reflink=auto "${CUDA_SOURCE}" "${CUDA_BINARY}"
cp --reflink=auto "${RELION_BIND_SOURCE}" "${RELION_BIND_BINARY}"

source "${GPU_SELECTION_HELPER}"
vdam_select_target_gpu "" 0
selected_gpu_uuid=${VDAM_SELECTED_GPU_UUID}

git -C "${REPO_ROOT}" rev-parse HEAD > "${ROOT}/provenance/repo_head.txt"
git -C "${REPO_ROOT}" symbolic-ref --short HEAD > "${ROOT}/provenance/repo_branch.txt"
git -C "${REPO_ROOT}" status --porcelain=v1 > "${ROOT}/provenance/repo_status.txt"
sha256sum "${CUDA_BINARY}" "${RELION_BIND_BINARY}" "${ACCEPTANCE}" "${ANALYZER}" "$0" > \
  "${ROOT}/provenance/input_manifest.sha256"
printf '%s\n' "${SLURM_JOB_ID}" > "${ROOT}/provenance/slurm_job_id.txt"
printf '%s\n' "${SLURMD_NODENAME:-$(hostname -s)}" > "${ROOT}/provenance/node.txt"
printf '%s\n' "${selected_gpu_uuid}" > "${ROOT}/provenance/selected_gpu_uuid.txt"
printf '%s\n' "${STABLE_FOURIER_WINDOW_SHAPES}" > \
  "${ROOT}/provenance/stable_fourier_window_shapes.txt"
printf '%s\n' "${STABLE_FOURIER_WINDOW_QUANTUM}" > \
  "${ROOT}/provenance/stable_fourier_window_quantum.txt"
printf '%s\n' "${STABLE_FLAT_ROW_CAPACITY}" > \
  "${ROOT}/provenance/stable_flat_row_capacity.txt"
nvidia-smi -i "${selected_gpu_uuid}" -q > "${ROOT}/provenance/nvidia_smi_q.txt"
env | LC_ALL=C sort > "${ROOT}/provenance/submission_environment.txt"

mapfile -t FIXED_ENVIRONMENT < <(
  jq -er '.runtime_contract.fixed_science_environment | to_entries | sort_by(.key)[] | "\(.key)=\(.value)"' \
    "${ACCEPTANCE}"
)
COMMON_ENVIRONMENT=(
  "${FIXED_ENVIRONMENT[@]}"
  "LD_PRELOAD=${CUSPARSE}"
  "PYTHONPATH=${REPO_ROOT}"
  "RECOVAR_CUDA_LIB=${CUDA_BINARY}"
  "RECOVAR_EXPECTED_REPO_ROOT=${REPO_ROOT}"
  "RECOVAR_RELION_BIND_BUILD_DIR=${RELION_BIND_DIR}"
  "RECOVAR_COARSE_GAUSSIAN_GEMM_HYBRID=1"
  "RECOVAR_COARSE_GAUSSIAN_GEMM_MACRO=1"
  "RECOVAR_COARSE_GAUSSIAN_GEMM_PROJECTION_CACHE=1"
  "RECOVAR_INITIAL_MODEL_FLAT_LOCAL_ROWS=1"
  "RECOVAR_INITIAL_MODEL_PACKED_LOCAL_PROJECTION=1"
  "RECOVAR_INITIAL_MODEL_DEFER_PACKED_VDAM=1"
  "RECOVAR_INITIAL_MODEL_STABLE_FLAT_ROW_CAPACITY=0"
)

(
  cd "${REPO_ROOT}"
  env "${COMMON_ENVIRONMENT[@]}" "${PIXI_PY}" - <<'PY'
import jax
import recovar
import recovar.cuda_backproject as cuda_backproject

assert jax.default_backend() == "gpu", jax.devices()
assert len(jax.devices("gpu")) == 1, jax.devices("gpu")
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
print("stable-ABI trajectory runtime gate passed", jax.devices(), recovar.__file__)
PY
) > "${ROOT}/provenance/runtime_gate.stdout" 2> "${ROOT}/provenance/runtime_gate.stderr"

ARM_ORDER=(stable_off_1 stable_on_1 stable_on_2 stable_off_2)
for arm in "${ARM_ORDER[@]}"; do
  run_root=${ROOT}/runs/${arm}
  output=${run_root}/output
  cache=${run_root}/jax_cache
  tmp=${run_root}/tmp
  mkdir -p "${output}" "${cache}" "${tmp}"
  touch "${cache}/SAFE_TO_DELETE" "${tmp}/SAFE_TO_DELETE"
  command=(
    "${PIXI_PY}" -m scripts.run_ab_initio
    --i "${FIXTURE}/particles.star"
    --o "${output}/run"
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
    --datadir "${FIXTURE}"
    --gpu 0
    --require_custom_cuda
    --diagnostic_stop_after_iteration "${LAST_ITERATION}"
  )
  stable=0
  stable_flat=0
  stable_quantum=8
  if [[ "${arm}" == stable_on_* ]]; then
    stable=${STABLE_FOURIER_WINDOW_SHAPES}
    stable_quantum=${STABLE_FOURIER_WINDOW_QUANTUM}
    if [[ "${stable}" == 1 ]]; then
      command+=(--stable-fourier-window-shapes)
    fi
    if [[ "${STABLE_FLAT_ROW_CAPACITY}" == 1 ]]; then
      stable_flat=1
    fi
  fi
  "${PIXI_PY}" - "${run_root}/command.json" "${stable}" "${stable_flat}" \
    "${stable_quantum}" \
    "${command[@]}" <<'PY'
import json
import sys
from pathlib import Path

target, stable, stable_flat, stable_quantum, *argv = sys.argv[1:]
Path(target).write_text(json.dumps({
    "argv": argv,
    "stable_fourier_window_shapes": bool(int(stable)),
    "stable_flat_row_capacity": bool(int(stable_flat)),
    "stable_fourier_window_quantum": int(stable_quantum),
}, indent=2, sort_keys=True) + "\n")
PY
  science_environment=(
    "${COMMON_ENVIRONMENT[@]}"
    "RECOVAR_INITIAL_MODEL_STABLE_FLAT_ROW_CAPACITY=${stable_flat}"
    "RECOVAR_RELION_VDAM_STABLE_FOURIER_WINDOW_QUANTUM=${stable_quantum}"
    "TMPDIR=${tmp}"
    "JAX_COMPILATION_CACHE_DIR=${cache}"
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
      env "${science_environment[@]}" "${command[@]}"
  ) > "${run_root}/runner.stdout" 2> "${run_root}/runner.stderr"
  status=$?
  set -e
  trap on_error ERR
  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true
  monitor_pid=
  end_epoch=$(date +%s.%N)
  test "${status}" -eq 0
  "${PIXI_PY}" - "${run_root}/timing.json" "${start_epoch}" "${end_epoch}" <<'PY'
import json
import sys
from pathlib import Path

start, end = map(float, sys.argv[2:])
Path(sys.argv[1]).write_text(json.dumps({
    "start_epoch": start,
    "end_epoch": end,
    "external_wall_s": end - start,
    "exit_status": 0,
}, indent=2, sort_keys=True) + "\n")
PY
  expected_count=$((4 * (LAST_ITERATION + 1)))
  observed_count=$(find "${output}" -maxdepth 1 -type f -regextype posix-extended \
    -regex '.*/run_it[0-9]{3}_(class001\.mrc|data\.star|model\.star|recovar_meta\.json)' | wc -l)
  test "${observed_count}" -eq "${expected_count}"
  for iteration in $(seq 0 "${LAST_ITERATION}"); do
    tag=$(printf '%03d' "${iteration}")
    test -s "${output}/run_it${tag}_class001.mrc"
    test -s "${output}/run_it${tag}_data.star"
    test -s "${output}/run_it${tag}_model.star"
    test -s "${output}/run_it${tag}_recovar_meta.json"
  done
  (
    cd "${run_root}"
    sha256sum command.json timing.json gpu_monitor.csv process.time runner.stdout runner.stderr \
      output/run_native_options.json > runtime_manifest.sha256
  )
  printf 'job_id=%s stable=%s\n' "${SLURM_JOB_ID}" "${stable}" > "${run_root}/SCIENCE_COMPLETED"
  test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_HEAD}"
  test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1)"
done

trap - ERR
set +e
analyzer_args=(
  --root "${ROOT}"
  --acceptance "${ACCEPTANCE}"
  --last-iteration "${LAST_ITERATION}"
  --stable-fourier-window-quantum "${STABLE_FOURIER_WINDOW_QUANTUM}"
  --output "${ROOT}/analysis/report.json"
  --markdown-output "${ROOT}/analysis/report.md"
)
if [[ "${STABLE_FOURIER_WINDOW_SHAPES}" == 0 ]]; then
  analyzer_args+=(--no-stable-fourier-window-shapes)
fi
if [[ "${STABLE_FLAT_ROW_CAPACITY}" == 1 ]]; then
  analyzer_args+=(--stable-flat-row-capacity)
fi
(
  cd "${REPO_ROOT}"
  env "${COMMON_ENVIRONMENT[@]}" "${PIXI_PY}" "${ANALYZER}" \
    "${analyzer_args[@]}"
) > "${ROOT}/analysis/analyzer.stdout" 2> "${ROOT}/analysis/analyzer.stderr"
analysis_status=$?
set -e
trap on_error ERR
test "${analysis_status}" -eq 0 -o "${analysis_status}" -eq 1
test -s "${ROOT}/analysis/report.json"
test -s "${ROOT}/analysis/report.md"
sha256sum "${ROOT}/analysis/report.json" "${ROOT}/analysis/report.md" > \
  "${ROOT}/analysis/report.sha256"
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = "${EXPECTED_HEAD}"
test -z "$(git -C "${REPO_ROOT}" status --porcelain=v1)"
if [[ "${analysis_status}" -eq 0 ]]; then
  printf 'job_id=%s\n' "${SLURM_JOB_ID}" > "${ROOT}/SCIENCE_PASSED"
else
  printf 'job_id=%s\n' "${SLURM_JOB_ID}" > "${ROOT}/SCIENCE_FAILED"
fi
printf 'job_id=%s analysis_status=%s\n' "${SLURM_JOB_ID}" "${analysis_status}" > "${ROOT}/COMPLETED"
exit "${analysis_status}"
