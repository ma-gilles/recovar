#!/usr/bin/env bash
#SBATCH --job-name=k1c7-it2-current
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=02:00:00

set -euo pipefail
: "${EXPERIMENT_ROOT:?set EXPERIMENT_ROOT}"
: "${EXPERIMENT_RUNTIME:?set EXPERIMENT_RUNTIME}"
: "${EXPECTED_HEAD:?set EXPECTED_HEAD}"
: "${EXPECTED_CUDA_SHA256:?set EXPECTED_CUDA_SHA256}"
: "${EXPECTED_LAUNCHER_SHA256:?set EXPECTED_LAUNCHER_SHA256}"
: "${EXPECTED_AUDIT_BUILDER_SHA256:?set EXPECTED_AUDIT_BUILDER_SHA256}"
: "${EXPECTED_REPO_DIFF_SHA256:?set EXPECTED_REPO_DIFF_SHA256}"
ARM=${ARM:-ffi}
case "${ARM}" in ffi|sincosf|per_image_fft|exact_operands|native_texture) ;; *) exit 2 ;; esac
TARGET_SOURCE_ROWS=${K1_CASE07_TARGET_SOURCE_ROWS:-}
PASSIVE_SIGNIFICANCE_DUMP=${K1_CASE07_PASSIVE_SIGNIFICANCE_DUMP:-0}
PASSIVE_STOP_AFTER_TARGET=${K1_CASE07_PASSIVE_STOP_AFTER_TARGET:-0}
KEEP_STACK_INDICES=${K1_CASE07_KEEP_STACK_INDICES:-}
PROJECTION_ROTATIONS=${K1_CASE07_PROJECTION_ROTATIONS:-}
EXACT_COARSE_COMPONENTS=${K1_EXACT_COARSE_COMPONENTS:-}
FRESH_PARTICLE_ORDER_SEED=${K1_CASE07_DIAGNOSTIC_FRESH_PARTICLE_ORDER_SEED:-}
PRESERVE_BPREF_PARTICLE_ORDER=${K1_CASE07_DIAGNOSTIC_PRESERVE_BPREF_PARTICLE_ORDER:-0}
RUN_FINAL=${K1_CASE07_RUN_FINAL:-0}
FINAL_REPLAY_FIELDS=${K1_CASE07_FINAL_REPLAY_FIELDS:-}
START_ITER=${K1_CASE07_START_ITER:-1}
INITIAL_HALF1_FT_NPZ=${K1_CASE07_INITIAL_HALF1_FT_NPZ:-}
INITIAL_HALF2_FT_NPZ=${K1_CASE07_INITIAL_HALF2_FT_NPZ:-}
FRESH_INITIAL_REFERENCE_MRC=${K1_CASE07_FRESH_INITIAL_REFERENCE_MRC:-}
case "${RUN_FINAL}" in 0|1) ;; *) echo "K1_CASE07_RUN_FINAL must be 0 or 1" >&2; exit 2 ;; esac
if [[ -n "${INITIAL_HALF1_FT_NPZ}" || -n "${INITIAL_HALF2_FT_NPZ}" ]]; then
  test -s "${INITIAL_HALF1_FT_NPZ}"
  test -s "${INITIAL_HALF2_FT_NPZ}"
fi
if [[ -n "${FRESH_INITIAL_REFERENCE_MRC}" ]]; then
  test -s "${FRESH_INITIAL_REFERENCE_MRC}"
  if [[ "${START_ITER}" != 0 || -n "${INITIAL_HALF1_FT_NPZ}" ]]; then
    echo "K1_CASE07_FRESH_INITIAL_REFERENCE_MRC requires a fresh run without Fourier overrides" >&2
    exit 2
  fi
fi
case "${START_ITER}" in
  0)
    MAX_ITER=${K1_CASE07_MAX_ITER:-3}
    if [[ ! "${MAX_ITER}" =~ ^[0-9]+$ || "${MAX_ITER}" -lt 1 || "${MAX_ITER}" -gt 15 ]]; then
      echo "K1_CASE07_MAX_ITER must be an integer in [1, 15]" >&2
      exit 2
    fi
    if [[ -n "${TARGET_SOURCE_ROWS}" ]]; then
      case "${MAX_ITER}" in
        1)
          TARGET_COARSE_SIZE=56
          TARGET_PASS2_SIZE=56
          TARGET_DUMP_ITERATION=1
          ;;
        3)
          TARGET_COARSE_SIZE=100
          TARGET_PASS2_SIZE=104
          TARGET_DUMP_ITERATION=3
          ;;
        *)
          echo "targeted dumps are qualified only for the one- or three-iteration boundaries" >&2
          exit 2
          ;;
      esac
    else
      TARGET_COARSE_SIZE=0
      TARGET_PASS2_SIZE=0
      TARGET_DUMP_ITERATION=
    fi
    RELION_ITERATION=${MAX_ITER}
    ;;
  1) RELION_ITERATION=2; TARGET_COARSE_SIZE=100; TARGET_PASS2_SIZE=100; MAX_ITER=1; TARGET_DUMP_ITERATION= ;;
  2) RELION_ITERATION=3; TARGET_COARSE_SIZE=100; TARGET_PASS2_SIZE=104; MAX_ITER=1; TARGET_DUMP_ITERATION= ;;
  *)
    if [[ ! "${START_ITER}" =~ ^[0-9]+$ || "${START_ITER}" -lt 1 || "${START_ITER}" -ge 15 ]]; then
      echo "K1_CASE07_START_ITER must be an integer in [0, 14]" >&2
      exit 2
    fi
    MAX_ITER=${K1_CASE07_MAX_ITER:-1}
    if [[ ! "${MAX_ITER}" =~ ^[0-9]+$ || "${MAX_ITER}" -lt 1 || $((START_ITER + MAX_ITER)) -gt 15 ]]; then
      echo "continuation MAX_ITER must be positive and end by physical iteration 15" >&2
      exit 2
    fi
    if [[ -n "${TARGET_SOURCE_ROWS}" ]]; then
      echo "targeted dumps are not qualified for generic continuation" >&2
      exit 2
    fi
    RELION_ITERATION=$((START_ITER + MAX_ITER))
    TARGET_COARSE_SIZE=0
    TARGET_PASS2_SIZE=0
    TARGET_DUMP_ITERATION=
    ;;
esac
if [[ "${RUN_FINAL}" = 1 && $((START_ITER + MAX_ITER)) -ne 15 ]]; then
  echo "K1_CASE07_RUN_FINAL requires a continuation ending at physical iteration 15" >&2
  exit 2
fi
if [[ "${RUN_FINAL}" != 1 && -n "${FINAL_REPLAY_FIELDS}" ]]; then
  echo "K1_CASE07_FINAL_REPLAY_FIELDS requires K1_CASE07_RUN_FINAL=1" >&2
  exit 2
fi
RELION_ITERATION_PADDED=$(printf '%03d' "${RELION_ITERATION}")

REPO=${K1_REPO_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810}
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default
CASE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fixedsuite_case07_tree_c74beea4_20260724T110500Z/cases/7_anisotropic_100k_g256_white_noise1_bf80
CUDA_LIB=${K1_CUDA_LIB:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_cuda126_compute80_ptx_e791e875_20260814T0948ET/cuda128_sm90_control/build/libcuda_backproject_cuda128_sm90.so}
RELION_BIND_BUILD_DIR=${K1_RELION_BIND_BUILD_DIR:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fixedsuite_case07_tree_c74beea4_20260724T110500Z/relion_bind_build/shared}
RELION_BIND_LIB=$(find "${RELION_BIND_BUILD_DIR}" -maxdepth 1 -type f -name '_relion_bind_core*.so' -print -quit)
LAUNCHER=${K1_LAUNCHER_PATH:-/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810/scripts/run_k1_case07_current_it2_boundary.sbatch}
AUDIT_BUILDER=${REPO}/scripts/build_em_particle_audit_archive.py

test -f "${EXPERIMENT_ROOT}/SAFE_TO_DELETE"
test -f "${EXPERIMENT_RUNTIME}/SAFE_TO_DELETE"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_HEAD}"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
test -n "${RELION_BIND_LIB}"
if [[ -n "${EXPECTED_RELION_BIND_SHA256:-}" ]]; then
  test "$(sha256sum "${RELION_BIND_LIB}" | awk '{print $1}')" = "${EXPECTED_RELION_BIND_SHA256}"
fi
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${AUDIT_BUILDER}" | awk '{print $1}')" = "${EXPECTED_AUDIT_BUILDER_SHA256}"
test "$(git -C "${REPO}" diff --binary | sha256sum | awk '{print $1}')" = "${EXPECTED_REPO_DIFF_SHA256}"
test -s "${AUDIT_BUILDER}"
mkdir -p "${EXPERIMENT_ROOT}"/{analysis,logs,output,provenance} \
  "${EXPERIMENT_RUNTIME}/${SLURM_JOB_ID}"/{tmp,pixi_home,rattler_cache,jax_cache,pycache}
touch "${EXPERIMENT_ROOT}/output/SAFE_TO_DELETE"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
  case "${variable_name}" in RECOVAR_*|RELION_BPRE_*|RELION_FINE_*|RELION_DISPATCH_LOG) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${EXPERIMENT_RUNTIME}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${EXPERIMENT_RUNTIME}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${EXPERIMENT_RUNTIME}/${SLURM_JOB_ID}/rattler_cache
export JAX_COMPILATION_CACHE_DIR=${EXPERIMENT_RUNTIME}/${SLURM_JOB_ID}/jax_cache
export PYTHONPYCACHEPREFIX=${EXPERIMENT_RUNTIME}/${SLURM_JOB_ID}/pycache

source /etc/profile.d/modules.sh
set +u
module purge
module load cudatoolkit/12.8
set -u
CUDA_HOME=/usr/local/cuda-12.8
export PATH=${CUDA_HOME}/bin:${PATH}
NVIDIA_ROOT=$(find "${PIXI_ENV}/lib" -maxdepth 3 -type d -path '*/site-packages/nvidia' -print -quit)
NVIDIA_LIBS=$(find "${NVIDIA_ROOT}" -type d -name lib | paste -sd: -)
export LD_LIBRARY_PATH=${NVIDIA_LIBS}:${CUDA_HOME}/targets/x86_64-linux/lib:${PIXI_ENV}/lib
export CUDA_VISIBLE_DEVICES=0

export RECOVAR_CUDA_LIB=${CUDA_LIB}
export RECOVAR_RELION_BIND_BUILD_DIR=${RELION_BIND_BUILD_DIR}
export RELION_SRC_DIR=/scratch/gpfs/GILLES/mg6942/relion/src
export RECOVAR_EXPECTED_REPO_ROOT=${REPO}
export RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION=0.40
export RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE=0
export RECOVAR_INITIAL_PROJECTOR_USE_REAL_REFERENCE=1
export RECOVAR_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN=4e-6
export RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT=0
export RECOVAR_K1_RELION_LIVE_INITIAL_NOISE=1
export RECOVAR_K1_RELION_EXACT_TRANSLATION_GRID=1
export RECOVAR_RELION_X_HALF_F32_FINE_POSTERIOR=1
export RECOVAR_K1_RELION_EXACT_BPREF_OPERANDS=1
export RECOVAR_K1_RELION_EXACT_CTF_STAR=${CASE}/data/particles.star
export RECOVAR_K1_COARSE_GAUSSIAN_FFI=1
export RECOVAR_K1_RELION_F32_COARSE_SUPPORT=1
if [[ "${ARM}" != ffi ]]; then export RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF=1; fi
if [[ "${ARM}" = per_image_fft ]]; then export RECOVAR_K1_RELION_PER_IMAGE_COARSE_FFT=1; fi
if [[ "${ARM}" = exact_operands || "${ARM}" = native_texture ]]; then
  export RECOVAR_K1_RELION_EXACT_COARSE_OPERANDS=1
fi
if [[ "${ARM}" = native_texture ]]; then
  export RECOVAR_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE=1
fi
if [[ -n "${EXACT_COARSE_COMPONENTS}" ]]; then
  export RECOVAR_K1_RELION_EXACT_COARSE_COMPONENTS=${EXACT_COARSE_COMPONENTS}
fi
if [[ -n "${TARGET_SOURCE_ROWS}" ]]; then
  mkdir -p "${EXPERIMENT_ROOT}/pass1" "${EXPERIMENT_ROOT}/pass2"
  export RECOVAR_SIGNIFICANCE_DUMP_DIR=${EXPERIMENT_ROOT}/pass1
  export RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES=${TARGET_SOURCE_ROWS}
  export RECOVAR_SIGNIFICANCE_DUMP_CURRENT_SIZE=${TARGET_COARSE_SIZE}
  if [[ -n "${PROJECTION_ROTATIONS}" ]]; then
    export RECOVAR_SIGNIFICANCE_DUMP_PROJECTION_ROTATIONS=${PROJECTION_ROTATIONS}
  fi
  if [[ -n "${TARGET_DUMP_ITERATION}" ]]; then
    export RECOVAR_SIGNIFICANCE_DUMP_ITERATION=${TARGET_DUMP_ITERATION}
  fi
  if [[ "${PASSIVE_SIGNIFICANCE_DUMP}" = 1 ]]; then
    export RECOVAR_SIGNIFICANCE_DUMP_PASSIVE_CACHE=1
    if [[ "${PASSIVE_STOP_AFTER_TARGET}" = 1 ]]; then
      export RECOVAR_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET=1
    else
      unset RECOVAR_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET
    fi
  else
    export RECOVAR_PASS2_DUMP_DIR=${EXPERIMENT_ROOT}/pass2
    export RECOVAR_PASS2_DUMP_ORIGINAL_INDICES=${TARGET_SOURCE_ROWS}
    export RECOVAR_PASS2_DUMP_CURRENT_SIZE=${TARGET_PASS2_SIZE}
    if [[ -n "${TARGET_DUMP_ITERATION}" ]]; then
      export RECOVAR_PASS2_DUMP_ITERATION=${TARGET_DUMP_ITERATION}
    fi
    export RECOVAR_PASS2_DUMP_CLASS=1
    export RECOVAR_PASS2_DUMP_RAW_OPERANDS=1
    export RECOVAR_PASS2_DUMP_STOP_AFTER_TARGET=1
  fi
fi
unset RECOVAR_FINAL_ALL_DATA_GRID_CORRECT RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER
if [[ "${RUN_FINAL}" = 1 ]]; then
  if [[ -z "${FINAL_REPLAY_FIELDS}" ]]; then
    # Candidate control: retain RECOVAR's complete last-numbered state.  The
    # unnumbered RELION optimiser still supplies the natural convergence event.
    export RECOVAR_FINAL_ALL_DATA_DISABLE_REPLAY_LAST_NUMBERED_STATE=1
  else
    # Focused state-group intervention.  The runner passes an explicit filtered
    # override, so automatic whole-state replay remains disabled by default.
    unset RECOVAR_FINAL_ALL_DATA_DISABLE_REPLAY_LAST_NUMBERED_STATE
  fi
fi

nvidia-smi -q > "${EXPERIMENT_ROOT}/provenance/nvidia_smi_${SLURM_JOB_ID}.txt"
git -C "${REPO}" status --short --branch > "${EXPERIMENT_ROOT}/provenance/git_status_${SLURM_JOB_ID}.txt"
env | sort > "${EXPERIMENT_ROOT}/provenance/environment_${SLURM_JOB_ID}.txt"
cd "${REPO}"
"${PYTHON}" -c "import pathlib,recovar,jax; from recovar import cuda_backproject; root=pathlib.Path.cwd().resolve(); assert pathlib.Path(recovar.__file__).resolve().is_relative_to(root); assert '.pixi/envs/default/' in str(pathlib.Path(jax.__file__).resolve()); assert len(jax.devices('gpu')) == 1 and 'H100' in jax.devices('gpu')[0].device_kind.upper(); assert cuda_backproject.cuda_available(); print(jax.devices())"

command=("${PYTHON}" -m scripts.run_multi_iter_parity
  --relion_dir "${CASE}/relion_ref" --data_star "${CASE}/data/particles.star"
  --iter "${START_ITER}" --max_iter "${MAX_ITER}" --continuous-relion-noise-state
  --image_batch_size 187 --rotation_block_size 8192 --image-fourier-backend relion_cuda
  --output_dir "${EXPERIMENT_ROOT}/output"
  --save_intermediates_dir "${EXPERIMENT_ROOT}/output/intermediates")
if [[ "${RUN_FINAL}" != 1 ]]; then
  command+=(--skip_final_iteration)
fi
if [[ -n "${FINAL_REPLAY_FIELDS}" ]]; then
  command+=(--final-replay-fields "${FINAL_REPLAY_FIELDS}")
fi
if [[ -n "${FRESH_PARTICLE_ORDER_SEED}" ]]; then
  command+=(--diagnostic-fresh-particle-order-seed "${FRESH_PARTICLE_ORDER_SEED}")
fi
if [[ "${PRESERVE_BPREF_PARTICLE_ORDER}" = 1 ]]; then
  command+=(--diagnostic-preserve-bpref-particle-order)
fi
if [[ -n "${INITIAL_HALF1_FT_NPZ}" ]]; then
  command+=(
    --initial-half1-ft-npz "${INITIAL_HALF1_FT_NPZ}"
    --initial-half2-ft-npz "${INITIAL_HALF2_FT_NPZ}"
  )
fi
if [[ -n "${FRESH_INITIAL_REFERENCE_MRC}" ]]; then
  command+=(--fresh-initial-reference-mrc "${FRESH_INITIAL_REFERENCE_MRC}")
fi
if [[ -n "${KEEP_STACK_INDICES}" ]]; then
  command+=(--keep_stack_indices "${KEEP_STACK_INDICES}")
fi
printf '%q ' "${command[@]}" > "${EXPERIMENT_ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${EXPERIMENT_ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
set +e
"${command[@]}" > "${EXPERIMENT_ROOT}/output/run.log" 2>&1
command_status=$?
set -e
if [[ -n "${TARGET_SOURCE_ROWS}" && "${PASSIVE_SIGNIFICANCE_DUMP}" = 1 && "${PASSIVE_STOP_AFTER_TARGET}" = 1 ]]; then
  IFS=',' read -r -a target_rows <<< "${TARGET_SOURCE_ROWS}"
  expected_count=${#target_rows[@]}
  test "$(find "${EXPERIMENT_ROOT}/pass1" -maxdepth 1 -name 'significance_*.npz' | wc -l)" -eq "${expected_count}"
  test "${command_status}" -eq 0
  rg -q 'RECOVAR coarse-significance dump completed; stopping before pass-2/M-step work' "${EXPERIMENT_ROOT}/output/run.log"
  sha256sum "${EXPERIMENT_ROOT}"/pass1/*.npz \
    > "${EXPERIMENT_ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
  touch "${EXPERIMENT_ROOT}/SUCCESS_${SLURM_JOB_ID}"
  exit 0
fi
if [[ -n "${TARGET_SOURCE_ROWS}" && "${PASSIVE_SIGNIFICANCE_DUMP}" != 1 ]]; then
  IFS=',' read -r -a target_rows <<< "${TARGET_SOURCE_ROWS}"
  expected_count=${#target_rows[@]}
  test "$(find "${EXPERIMENT_ROOT}/pass1" -maxdepth 1 -name 'significance_*.npz' | wc -l)" -eq "${expected_count}"
  test "$(find "${EXPERIMENT_ROOT}/pass2" -maxdepth 1 -name 'pass2_*.npz' | wc -l)" -eq "${expected_count}"
  test "${command_status}" -ne 0
  rg -q 'requested RECOVAR pass-2 dump target set was written' "${EXPERIMENT_ROOT}/output/run.log"
  sha256sum "${EXPERIMENT_ROOT}"/pass1/*.npz "${EXPERIMENT_ROOT}"/pass2/*.npz \
    > "${EXPERIMENT_ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
  touch "${EXPERIMENT_ROOT}/SUCCESS_${SLURM_JOB_ID}"
  exit 0
fi
test "${command_status}" -eq 0
if [[ "${RUN_FINAL}" = 1 ]]; then
  "${PYTHON}" - "${EXPERIMENT_ROOT}/output/refinement_results.npz" <<'PY'
import sys

import numpy as np

with np.load(sys.argv[1], allow_pickle=False) as result:
    assert bool(np.asarray(result["final_all_data_ran"]).reshape(()))
    assert not bool(np.asarray(result["final_all_data_grid_correct"]).reshape(()))
PY
  rg -q 'Replay override: RELION final optimiser convergence' "${EXPERIMENT_ROOT}/output/run.log"
  if [[ -z "${FINAL_REPLAY_FIELDS}" ]]; then
    rg -q 'final all-data skips automatic last-numbered RELION state replay' "${EXPERIMENT_ROOT}/output/run.log"
  else
    rg -q 'Diagnostic final-only RELION state substitution' "${EXPERIMENT_ROOT}/output/run.log"
  fi
fi
if [[ -n "${TARGET_SOURCE_ROWS}" ]]; then
  IFS=',' read -r -a target_rows <<< "${TARGET_SOURCE_ROWS}"
  expected_count=${#target_rows[@]}
  test "$(find "${EXPERIMENT_ROOT}/pass1" -maxdepth 1 -name 'significance_*.npz' | wc -l)" -eq "${expected_count}"
fi
"${PYTHON}" "${AUDIT_BUILDER}" --output-dir "${EXPERIMENT_ROOT}/output"
particle_audit_command=("${PYTHON}" -m scripts.audit_em_particle_state_distribution \
  --recovar-results "${EXPERIMENT_ROOT}/analysis/refinement_results_audit_source_order.npz" \
  --recovar-particles-star "${CASE}/data/particles.star" \
  --output-json "${EXPERIMENT_ROOT}/analysis/particle.json" \
  --output-npz "${EXPERIMENT_ROOT}/analysis/particle_arrays.npz" \
  --output-hash-manifest "${EXPERIMENT_ROOT}/analysis/particle.json.sha256")
for audit_iteration in $(seq 0 $((MAX_ITER - 1))); do
  relion_audit_iteration=$((START_ITER + audit_iteration + 1))
  relion_audit_iteration_padded=$(printf '%03d' "${relion_audit_iteration}")
  particle_audit_command+=(
    --relion-star "${relion_audit_iteration}:${CASE}/relion_ref/run_it${relion_audit_iteration_padded}_data.star"
    --recovar-iteration "${audit_iteration}"
  )
done
particle_audit_command+=(--relion-iteration-offset $((START_ITER + 1)))
"${particle_audit_command[@]}"
sha256sum "${EXPERIMENT_ROOT}/output/refinement_results.npz" \
  "${EXPERIMENT_ROOT}/analysis/particle.json" \
  "${EXPERIMENT_ROOT}/analysis/particle_arrays.npz" \
  > "${EXPERIMENT_ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
touch "${EXPERIMENT_ROOT}/SUCCESS_${SLURM_JOB_ID}"
