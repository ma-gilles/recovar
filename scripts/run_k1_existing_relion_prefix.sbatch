#!/usr/bin/env bash
#SBATCH --job-name=k1-relion-prefix
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=04:00:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${CASE_SOURCE:?CASE_SOURCE is required}"
: "${CASE_NAME:?CASE_NAME is required}"
: "${SEED:?SEED is required}"

REPO=${REPO:-/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_norm_matrix_fb3995ca_20260821}
PYTHON=${PYTHON:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_normspectrum_fb3995ca_20260821T2025ET/venv/bin/python}
CUDA_LIB=${CUDA_LIB:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_normspectrum_fb3995ca_20260821T2025ET/cuda/libcuda_backproject.so}
RELION_BIND_DIR=${RELION_BIND:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_normspectrum_fb3995ca_20260821T2025ET/relion_bind_build/shared}
EXPECTED_HEAD=${EXPECTED_HEAD:-fb3995ca08192fe31956fc853a305eb94f1e0739}
EXPECTED_REPO_DIFF_SHA256=${EXPECTED_REPO_DIFF_SHA256:-}
EXPECTED_LAUNCHER_SHA256=${EXPECTED_LAUNCHER_SHA256:-}
MAX_ITER=${MAX_ITER:-2}
TARGET_SOURCE_ROWS=${TARGET_SOURCE_ROWS:-}
TARGET_CURRENT_SIZE=${TARGET_CURRENT_SIZE:-100}
TARGET_ITERATION=${TARGET_ITERATION:-2}
STOP_AFTER_TARGET=${STOP_AFTER_TARGET:-0}
PROJECTION_ROTATIONS=${PROJECTION_ROTATIONS:-}
PASSIVE_SIGNIFICANCE_DUMP=${PASSIVE_SIGNIFICANCE_DUMP:-0}
PASSIVE_STOP_AFTER_TARGET=${PASSIVE_STOP_AFTER_TARGET:-0}
ENABLE_EXACT_BPREF=${ENABLE_EXACT_BPREF:-0}
ENABLE_LIVE_INITIAL_NOISE=${ENABLE_LIVE_INITIAL_NOISE:-0}
ENABLE_NATIVE_TEXTURE=${ENABLE_NATIVE_TEXTURE:-0}
SPECTRUM_NORM=${SPECTRUM_NORM:-1}
NOISE_DEBUG_DUMP=${NOISE_DEBUG_DUMP:-0}
NORM_RESIDUAL_DUMP=${NORM_RESIDUAL_DUMP:-0}
NORM_RESIDUAL_TARGET_HALF=${NORM_RESIDUAL_TARGET_HALF:-}
STATE_SWAP_VARIANT=${STATE_SWAP_VARIANT:-}

case "${STOP_AFTER_TARGET}" in 0|1) ;; *) echo "STOP_AFTER_TARGET must be 0 or 1" >&2; exit 2 ;; esac
case "${PASSIVE_SIGNIFICANCE_DUMP}" in 0|1) ;; *) echo "PASSIVE_SIGNIFICANCE_DUMP must be 0 or 1" >&2; exit 2 ;; esac
case "${PASSIVE_STOP_AFTER_TARGET}" in 0|1) ;; *) echo "PASSIVE_STOP_AFTER_TARGET must be 0 or 1" >&2; exit 2 ;; esac
case "${ENABLE_EXACT_BPREF}" in 0|1) ;; *) echo "ENABLE_EXACT_BPREF must be 0 or 1" >&2; exit 2 ;; esac
case "${ENABLE_LIVE_INITIAL_NOISE}" in 0|1) ;; *) echo "ENABLE_LIVE_INITIAL_NOISE must be 0 or 1" >&2; exit 2 ;; esac
case "${ENABLE_NATIVE_TEXTURE}" in 0|1) ;; *) echo "ENABLE_NATIVE_TEXTURE must be 0 or 1" >&2; exit 2 ;; esac
case "${SPECTRUM_NORM}" in 0|1) ;; *) echo "SPECTRUM_NORM must be 0 or 1" >&2; exit 2 ;; esac
case "${NOISE_DEBUG_DUMP}" in 0|1) ;; *) echo "NOISE_DEBUG_DUMP must be 0 or 1" >&2; exit 2 ;; esac
case "${NORM_RESIDUAL_DUMP}" in 0|1) ;; *) echo "NORM_RESIDUAL_DUMP must be 0 or 1" >&2; exit 2 ;; esac
case "${STATE_SWAP_VARIANT}" in ""|all_relion) ;; *) echo "STATE_SWAP_VARIANT must be empty or all_relion" >&2; exit 2 ;; esac
if [[ -n "${NORM_RESIDUAL_TARGET_HALF}" && "${NORM_RESIDUAL_TARGET_HALF}" != 1 && "${NORM_RESIDUAL_TARGET_HALF}" != 2 ]]; then
  echo "NORM_RESIDUAL_TARGET_HALF must be 1 or 2" >&2
  exit 2
fi
if [[ -n "${NORM_RESIDUAL_TARGET_HALF}" && "${NORM_RESIDUAL_DUMP}" != 1 ]]; then
  echo "NORM_RESIDUAL_TARGET_HALF requires NORM_RESIDUAL_DUMP=1" >&2
  exit 2
fi
if [[ "${PASSIVE_STOP_AFTER_TARGET}" = 1 && "${PASSIVE_SIGNIFICANCE_DUMP}" != 1 ]]; then
  echo "PASSIVE_STOP_AFTER_TARGET requires PASSIVE_SIGNIFICANCE_DUMP=1" >&2
  exit 2
fi
if [[ -n "${TARGET_SOURCE_ROWS}" ]]; then
  [[ "${TARGET_CURRENT_SIZE}" =~ ^[0-9]+$ ]]
  [[ "${TARGET_ITERATION}" =~ ^[0-9]+$ ]]
  test "${TARGET_ITERATION}" -le "${MAX_ITER}"
elif [[ "${STOP_AFTER_TARGET}" = 1 || "${PASSIVE_SIGNIFICANCE_DUMP}" = 1 || "${NORM_RESIDUAL_DUMP}" = 1 || -n "${PROJECTION_ROTATIONS}" ]]; then
  echo "targeted dump settings require TARGET_SOURCE_ROWS" >&2
  exit 2
fi

ROOT=${RUN_ROOT}
RUNTIME=${RUNTIME_ROOT}
DATA=${CASE_SOURCE}/data
RELION=${CASE_SOURCE}/relion_ref
OUTPUT=${ROOT}/output
ANALYSIS=${ROOT}/analysis

test -f "${ROOT}/SAFE_TO_DELETE"
test -f "${RUNTIME}/SAFE_TO_DELETE"
test -x "${PYTHON}"
test -f "${CUDA_LIB}"
test -s "${DATA}/particles.star"
test -s "${DATA}/reference_gt.mrc"
test -s "${RELION}/run_it000_data.star"
test -s "${RELION}/run_it000_optimiser.star"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_HEAD}"
if [[ -n "${EXPECTED_REPO_DIFF_SHA256}" ]]; then
  test "$(git -C "${REPO}" diff --binary | sha256sum | awk '{print $1}')" = "${EXPECTED_REPO_DIFF_SHA256}"
else
  test -z "$(git -C "${REPO}" status --porcelain=v1)"
fi
if [[ -n "${EXPECTED_LAUNCHER_SHA256}" ]]; then
  test "$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
fi

mkdir -p "${ROOT}/logs" "${ROOT}/provenance" "${OUTPUT}/intermediates" "${ANALYSIS}"
if [[ -n "${TARGET_SOURCE_ROWS}" ]]; then
  mkdir -p "${ROOT}/pass1" "${ROOT}/pass2"
fi
mkdir -p "${RUNTIME}/tmp" "${RUNTIME}/pixi_home" "${RUNTIME}/rattler_cache" \
  "${RUNTIME}/pycache" "${RUNTIME}/jax_cache" "${RUNTIME}/cuda_cache"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
  case "${variable_name}" in RECOVAR_*|RELION_*) unset "${variable_name}" ;; esac
done < <(env)

export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME}/tmp
export PIXI_HOME=${RUNTIME}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/rattler_cache
export PYTHONPYCACHEPREFIX=${RUNTIME}/pycache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME}/jax_cache
export RECOVAR_CUDA_CACHE_DIR=${RUNTIME}/cuda_cache

source /etc/profile.d/modules.sh
set +u
module purge
module load cudatoolkit/12.8
set -u
PYTHON_REAL=$(readlink -f "${PYTHON}")
PIXI_ENV=$(cd "$(dirname "${PYTHON_REAL}")/.." && pwd)
NVIDIA_ROOT=$(find "${PIXI_ENV}/lib" -maxdepth 3 -type d -path '*/site-packages/nvidia' -print -quit)
NVIDIA_LIBS=$(find "${NVIDIA_ROOT}" -type d -name lib | paste -sd: -)
export LD_LIBRARY_PATH=${NVIDIA_LIBS}:/usr/local/cuda-12.8/targets/x86_64-linux/lib:${PIXI_ENV}/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=0
export RECOVAR_CUDA_LIB=${CUDA_LIB}
export RECOVAR_RELION_BIND_BUILD_DIR=${RELION_BIND_DIR}

# Match the fixed terminal matrix. The production source-faithful spectrum
# normalization and exact BPref defaults are intentionally not overridden.
export RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION=.40
export RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE=0
export RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT=0
export RECOVAR_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE=${ENABLE_NATIVE_TEXTURE}
export RECOVAR_K1_BPREF_EXECUTION_ORDER_CHUNK_SIZE=220
export RECOVAR_K1_RELION_POWERCLASS_SPECTRUM_NORM=${SPECTRUM_NORM}
unset RECOVAR_K1_RELION_EXACT_BPREF_OPERANDS
export RECOVAR_K1_RELION_LIVE_INITIAL_NOISE=${ENABLE_LIVE_INITIAL_NOISE}
if [[ "${ENABLE_EXACT_BPREF}" = 1 ]]; then
  export RECOVAR_K1_RELION_EXACT_BPREF_OPERANDS=1
  export RECOVAR_K1_RELION_EXACT_CTF_STAR=${DATA}/particles.star
fi
if [[ -n "${STATE_SWAP_VARIANT}" ]]; then
  export RECOVAR_STATE_SWAP_FORCE_FRESH_PARTICLE_ORDER=1
fi
if [[ "${NOISE_DEBUG_DUMP}" = 1 ]]; then
  mkdir -p "${ROOT}/noise_debug"
  export RECOVAR_NOISE_DEBUG_DUMP_DIR=${ROOT}/noise_debug
  export RECOVAR_NOISE_DEBUG_DUMP_ITERATION=0
fi
unset RECOVAR_FINAL_ALL_DATA_GRID_CORRECT RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER
if [[ -n "${TARGET_SOURCE_ROWS}" ]]; then
  export RECOVAR_SIGNIFICANCE_DUMP_DIR=${ROOT}/pass1
  export RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES=${TARGET_SOURCE_ROWS}
  export RECOVAR_SIGNIFICANCE_DUMP_CURRENT_SIZE=${TARGET_CURRENT_SIZE}
  export RECOVAR_SIGNIFICANCE_DUMP_ITERATION=${TARGET_ITERATION}
  if [[ -n "${PROJECTION_ROTATIONS}" ]]; then
    export RECOVAR_SIGNIFICANCE_DUMP_PROJECTION_ROTATIONS=${PROJECTION_ROTATIONS}
  fi
  if [[ "${PASSIVE_SIGNIFICANCE_DUMP}" = 1 ]]; then
    export RECOVAR_SIGNIFICANCE_DUMP_PASSIVE_CACHE=1
    if [[ "${PASSIVE_STOP_AFTER_TARGET}" = 1 ]]; then
      export RECOVAR_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET=1
    fi
  fi
  if [[ "${PASSIVE_STOP_AFTER_TARGET}" != 1 ]]; then
    export RECOVAR_PASS2_DUMP_DIR=${ROOT}/pass2
    export RECOVAR_PASS2_DUMP_ORIGINAL_INDICES=${TARGET_SOURCE_ROWS}
    export RECOVAR_PASS2_DUMP_CURRENT_SIZE=${TARGET_CURRENT_SIZE}
    export RECOVAR_PASS2_DUMP_ITERATION=${TARGET_ITERATION}
    export RECOVAR_PASS2_DUMP_CLASS=1
    if [[ "${NORM_RESIDUAL_DUMP}" != 1 ]]; then
      export RECOVAR_PASS2_DUMP_RAW_OPERANDS=1
    fi
    if [[ "${STOP_AFTER_TARGET}" = 1 ]]; then
      export RECOVAR_PASS2_DUMP_STOP_AFTER_TARGET=1
    fi
    if [[ "${NORM_RESIDUAL_DUMP}" = 1 ]]; then
      export RECOVAR_PASS2_DUMP_NORM_RESIDUAL_INPUTS=1
      export RECOVAR_PASS2_DUMP_NORM_RESIDUAL_ONLY=1
      export RECOVAR_PASS2_DUMP_NORM_RESIDUAL_STOP_AFTER_TARGET=1
      if [[ -n "${NORM_RESIDUAL_TARGET_HALF}" ]]; then
        export RECOVAR_PASS2_DUMP_TARGET_HALF=${NORM_RESIDUAL_TARGET_HALF}
      fi
    fi
  fi
fi

cd "${REPO}"
git rev-parse HEAD > "${ROOT}/provenance/repo_head_${SLURM_JOB_ID}.txt"
git status --short > "${ROOT}/provenance/repo_status_${SLURM_JOB_ID}.txt"
sha256sum "${BASH_SOURCE[0]}" "${CUDA_LIB}" "${DATA}/particles.star" \
  "${RELION}/run_it000_data.star" "${RELION}/run_it000_optimiser.star" \
  > "${ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"
env | sort > "${ROOT}/provenance/environment_${SLURM_JOB_ID}.txt"
nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv,noheader \
  > "${ROOT}/provenance/gpu_${SLURM_JOB_ID}.txt"

"${PYTHON}" - <<'PY'
from pathlib import Path
import jax
import recovar
from recovar import cuda_backproject

repo = Path.cwd().resolve()
assert Path(recovar.__file__).resolve().is_relative_to(repo)
assert len(jax.devices("gpu")) == 1 and "H100" in jax.devices("gpu")[0].device_kind.upper()
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
print(jax.devices())
PY

command=(
  "${PYTHON}" -m scripts.run_full_refinement
  --data_dir "${DATA}" --output "${OUTPUT}" --max_iter "${MAX_ITER}" --skip_final_iteration
  --healpix_order 3 --auto_local_healpix_order 4 --offset_range 3.0 --offset_step 1.0
  --adaptive_oversampling 1 --init_resolution 30.0
  --image_batch_size 187 --rotation_block_size 8192
  --seed "${SEED}" --perturb_seed "${SEED}" --particle_diameter_ang 200
  --tau2_fudge 1.0 --max_significants -1 --firstiter_cc --apply-initial-lowpass
  --image-fourier-backend relion_cuda
  --save_intermediates_dir "${OUTPUT}/intermediates" --save_intermediates_skip_unregularized
  --relion_half_sets "${RELION}/run_it000_data.star"
  --relion_optimiser "${RELION}/run_it000_optimiser.star"
  --relion_init_dir "${RELION}"
)
if [[ -n "${STATE_SWAP_VARIANT}" ]]; then
  command+=(
    --perturb_replay_relion_dir "${RELION}"
    --state-swap-target-relion-iteration "${TARGET_ITERATION}"
    --state-swap-variant "${STATE_SWAP_VARIANT}"
    --state-swap-replay-relion-references
  )
fi
printf '%q ' "${command[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
start_epoch=$(date +%s)
set +e
"${command[@]}" > "${ROOT}/logs/run_${SLURM_JOB_ID}.log" 2>&1
command_status=$?
set -e
wall_s=$(( $(date +%s) - start_epoch ))

! rg -q 'Traceback|RESOURCE_EXHAUSTED|Out of memory|accounting is inconsistent' "${ROOT}/logs/run_${SLURM_JOB_ID}.log"
if [[ "${PASSIVE_SIGNIFICANCE_DUMP}" = 1 && "${PASSIVE_STOP_AFTER_TARGET}" = 1 ]]; then
  IFS=',' read -r -a target_rows <<< "${TARGET_SOURCE_ROWS}"
  expected_count=${#target_rows[@]}
  test "${command_status}" -eq 0
  test "$(find "${ROOT}/pass1" -maxdepth 1 -name 'significance_*.npz' | wc -l)" -eq "${expected_count}"
  test -z "$(find "${ROOT}/pass2" -maxdepth 1 -type f -print -quit)"
  rg -q 'RECOVAR coarse-significance dump completed; stopping before pass-2/M-step work' \
    "${ROOT}/logs/run_${SLURM_JOB_ID}.log"
  sha256sum "${ROOT}"/pass1/*.npz \
    > "${ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
  printf '{"case":"%s","job_id":"%s","wall_s":%d,"max_iter":%d,"mode":"passive_coarse_projection_dump","target_source_rows":"%s","target_iteration":%d,"target_current_size":%d,"projection_rotations":"%s","exact_bpref":%s,"live_initial_noise":%s,"native_texture":%s,"spectrum_norm":%s}\n' \
    "${CASE_NAME}" "${SLURM_JOB_ID}" "${wall_s}" "${MAX_ITER}" "${TARGET_SOURCE_ROWS}" \
    "${TARGET_ITERATION}" "${TARGET_CURRENT_SIZE}" "${PROJECTION_ROTATIONS}" "${ENABLE_EXACT_BPREF}" "${ENABLE_LIVE_INITIAL_NOISE}" "${ENABLE_NATIVE_TEXTURE}" "${SPECTRUM_NORM}" \
    > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
  touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
  exit 0
fi
if [[ "${NORM_RESIDUAL_DUMP}" = 1 ]]; then
  IFS=',' read -r -a target_rows <<< "${TARGET_SOURCE_ROWS}"
  expected_count=${#target_rows[@]}
  test "${command_status}" -eq 0
  test "$(find "${ROOT}/pass2" -maxdepth 1 -name 'norm_residual_*.npz' | wc -l)" -eq "${expected_count}"
  rg -q 'requested fine-score boundary or norm/scale boundary' "${ROOT}/logs/run_${SLURM_JOB_ID}.log"
  sha256sum "${ROOT}"/pass2/norm_residual_*.npz \
    > "${ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
  printf '{"case":"%s","job_id":"%s","wall_s":%d,"max_iter":%d,"mode":"norm_residual_target_dump","target_source_rows":"%s","target_iteration":%d,"target_current_size":%d,"exact_bpref":%s,"live_initial_noise":%s,"spectrum_norm":%s}\n' \
    "${CASE_NAME}" "${SLURM_JOB_ID}" "${wall_s}" "${MAX_ITER}" "${TARGET_SOURCE_ROWS}" \
    "${TARGET_ITERATION}" "${TARGET_CURRENT_SIZE}" "${ENABLE_EXACT_BPREF}" "${ENABLE_LIVE_INITIAL_NOISE}" "${SPECTRUM_NORM}" \
    > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
  touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
  exit 0
fi
if [[ "${STOP_AFTER_TARGET}" = 1 ]]; then
  IFS=',' read -r -a target_rows <<< "${TARGET_SOURCE_ROWS}"
  expected_count=${#target_rows[@]}
  test "${command_status}" -eq 0
  test "$(find "${ROOT}/pass1" -maxdepth 1 -name 'significance_*.npz' | wc -l)" -eq "${expected_count}"
  test "$(find "${ROOT}/pass2" -maxdepth 1 -name 'pass2_*.npz' | wc -l)" -eq "${expected_count}"
  rg -q 'pass-2 operand dump completed; stopping at the requested fine-score boundary' \
    "${ROOT}/logs/run_${SLURM_JOB_ID}.log"
  sha256sum "${ROOT}"/pass1/*.npz "${ROOT}"/pass2/*.npz \
    > "${ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
  printf '{"case":"%s","job_id":"%s","wall_s":%d,"max_iter":%d,"mode":"matrix_faithful_target_dump","target_source_rows":"%s","target_iteration":%d,"target_current_size":%d,"exact_bpref":%s,"live_initial_noise":%s,"spectrum_norm":%s,"state_swap_variant":"%s"}\n' \
    "${CASE_NAME}" "${SLURM_JOB_ID}" "${wall_s}" "${MAX_ITER}" "${TARGET_SOURCE_ROWS}" \
    "${TARGET_ITERATION}" "${TARGET_CURRENT_SIZE}" "${ENABLE_EXACT_BPREF}" "${ENABLE_LIVE_INITIAL_NOISE}" "${SPECTRUM_NORM}" "${STATE_SWAP_VARIANT}" \
    > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
  touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
  exit 0
fi
test "${command_status}" -eq 0
test -s "${OUTPUT}/refinement_results.npz"

"${PYTHON}" -m scripts.audit_k1_fsc_trajectory \
  --case-root "${CASE_SOURCE}" \
  --recovar-dir "${OUTPUT}" \
  --relion-dir "${RELION}" \
  --gt-volume "${DATA}/reference_gt.mrc" \
  --numbered-only --allow-incomplete \
  --output-json "${ANALYSIS}/prefix_fsc.json" \
  --output-markdown "${ANALYSIS}/prefix_fsc.md" \
  --output-shellwise-npz "${ANALYSIS}/prefix_fsc_shellwise.npz"

"${PYTHON}" -m scripts.audit_k1_intermediate_trajectory \
  --case-root "${CASE_SOURCE}" \
  --recovar-dir "${OUTPUT}" \
  --relion-dir "${RELION}" \
  --output-json "${ANALYSIS}/prefix_intermediates.json" \
  --output-markdown "${ANALYSIS}/prefix_intermediates.md"

sha256sum "${OUTPUT}/refinement_results.npz" "${ANALYSIS}/prefix_fsc.json" \
  "${ANALYSIS}/prefix_fsc_shellwise.npz" "${ANALYSIS}/prefix_intermediates.json" \
  > "${ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
printf '{"case":"%s","job_id":"%s","wall_s":%d,"max_iter":%d,"mode":"matrix_faithful_existing_relion_prefix"}\n' \
  "${CASE_NAME}" "${SLURM_JOB_ID}" "${wall_s}" "${MAX_ITER}" \
  > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
