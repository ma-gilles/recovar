#!/usr/bin/env bash
#SBATCH --job-name=k1c22-order-it3
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=02:00:00

set -euo pipefail
: "${K1_CASE22_IT3_ROOT:?set the disposable experiment root}"
: "${K1_CASE22_IT3_RUNTIME:?set the disposable runtime root}"
: "${K1_CASE22_IT3_REPO:?set the frozen source snapshot}"
: "${EXPECTED_LAUNCHER_SHA256:?pin the launcher hash}"
: "${EXPECTED_REPO_HEAD:?pin the source snapshot head}"
: "${EXPECTED_REPO_DIFF_SHA256:?pin the source snapshot tracked diff}"
: "${EXPECTED_CUDA_SHA256:?pin the CUDA library}"

ROOT=${K1_CASE22_IT3_ROOT}
RUNTIME=${K1_CASE22_IT3_RUNTIME}
REPO=${K1_CASE22_IT3_REPO}
MAX_ITER=${K1_CASE22_MAX_ITER:-3}
PARITY_DUMP=${K1_CASE22_PARITY_DUMP:-0}
COMPACT_STACK1204_DUMP=${K1_CASE22_COMPACT_STACK1204_DUMP:-0}
STOP_AFTER_COMPACT_DUMP=${K1_CASE22_STOP_AFTER_COMPACT_DUMP:-0}
NORM_RESIDUAL_INPUTS=${K1_CASE22_NORM_RESIDUAL_INPUTS:-0}
POWERCLASS_SPECTRUM_NORM=${K1_CASE22_POWERCLASS_SPECTRUM_NORM:-0}
TRANSLATED_WAVG_NORM=${K1_CASE22_TRANSLATED_WAVG_NORM:-0}
WAVG_ATOMIC_SCALE_AA=${K1_CASE22_WAVG_ATOMIC_SCALE_AA:-0}
WAVG_ATOMIC_DIRECT_RESIDUAL=${K1_CASE22_WAVG_ATOMIC_DIRECT_RESIDUAL:-0}
FINE_ROTATION_EXECUTION_ORDER=${K1_CASE22_FINE_ROTATION_EXECUTION_ORDER:-0}
COARSE_STACK1204_DUMP=${K1_CASE22_COARSE_STACK1204_DUMP:-0}
GAUSSIAN_COARSE_STACK1204=${K1_CASE22_GAUSSIAN_COARSE_STACK1204:-0}
GAUSSIAN_SINCOSF_STACK1204=${K1_CASE22_GAUSSIAN_SINCOSF_STACK1204:-${GAUSSIAN_COARSE_STACK1204}}
EXACT_COARSE_STACK1204=${K1_CASE22_EXACT_COARSE_STACK1204:-0}
NOISE_UPDATE_DUMP=${K1_CASE22_NOISE_UPDATE_DUMP:-0}
PROJECTOR_DUMP=${K1_CASE22_PROJECTOR_DUMP:-0}
TARGET_STACK_ONE_BASED=${K1_CASE22_TARGET_STACK_ONE_BASED:-1204}
TARGET_STACKS_ONE_BASED=${K1_CASE22_TARGET_STACKS_ONE_BASED:-${TARGET_STACK_ONE_BASED}}
TARGET_STACKS_ONE_BASED=${TARGET_STACKS_ONE_BASED//:/,}
TARGET_ITERATION=${K1_CASE22_TARGET_ITERATION:-2}
TARGET_CURRENT_SIZE=${K1_CASE22_TARGET_CURRENT_SIZE:-60}
if (( TARGET_STACK_ONE_BASED < 1 )); then
    echo "K1_CASE22_TARGET_STACK_ONE_BASED must be positive" >&2
    exit 2
fi
if (( TARGET_ITERATION < 1 || TARGET_CURRENT_SIZE < 1 )); then
    echo "K1_CASE22_TARGET_ITERATION and K1_CASE22_TARGET_CURRENT_SIZE must be positive" >&2
    exit 2
fi
TARGET_ORIGINAL_INDICES=
IFS=',' read -r -a target_stack_values <<< "${TARGET_STACKS_ONE_BASED}"
for target_stack in "${target_stack_values[@]}"; do
    if [[ ! "${target_stack}" =~ ^[0-9]+$ ]] || (( target_stack < 1 )); then
        echo "K1_CASE22_TARGET_STACKS_ONE_BASED must be comma-separated positive integers" >&2
        exit 2
    fi
    target_original=$((target_stack - 1))
    TARGET_ORIGINAL_INDICES=${TARGET_ORIGINAL_INDICES:+${TARGET_ORIGINAL_INDICES},}${target_original}
done
if (( ${#target_stack_values[@]} != 1 )) && [[ "${COARSE_STACK1204_DUMP}" != 1 ]]; then
    echo "multiple target stacks are supported only by coarse-significance dump mode" >&2
    exit 2
fi
case "${NORM_RESIDUAL_INPUTS}" in
    0|1) ;;
    *) echo "K1_CASE22_NORM_RESIDUAL_INPUTS must be 0 or 1" >&2; exit 2 ;;
esac
case "${POWERCLASS_SPECTRUM_NORM}" in
    0|1) ;;
    *) echo "K1_CASE22_POWERCLASS_SPECTRUM_NORM must be 0 or 1" >&2; exit 2 ;;
esac
case "${TRANSLATED_WAVG_NORM}" in
    0|1) ;;
    *) echo "K1_CASE22_TRANSLATED_WAVG_NORM must be 0 or 1" >&2; exit 2 ;;
esac
case "${WAVG_ATOMIC_SCALE_AA}" in
    0|1) ;;
    *) echo "K1_CASE22_WAVG_ATOMIC_SCALE_AA must be 0 or 1" >&2; exit 2 ;;
esac
case "${WAVG_ATOMIC_DIRECT_RESIDUAL}" in
    0|1) ;;
    *) echo "K1_CASE22_WAVG_ATOMIC_DIRECT_RESIDUAL must be 0 or 1" >&2; exit 2 ;;
esac
case "${FINE_ROTATION_EXECUTION_ORDER}" in
    0|1) ;;
    *) echo "K1_CASE22_FINE_ROTATION_EXECUTION_ORDER must be 0 or 1" >&2; exit 2 ;;
esac
case "${NOISE_UPDATE_DUMP}" in
    0|1) ;;
    *) echo "K1_CASE22_NOISE_UPDATE_DUMP must be 0 or 1" >&2; exit 2 ;;
esac
case "${PROJECTOR_DUMP}" in
    0|1) ;;
    *) echo "K1_CASE22_PROJECTOR_DUMP must be 0 or 1" >&2; exit 2 ;;
esac
TARGET_ORIGINAL_INDEX=$((TARGET_STACK_ONE_BASED - 1))
PASS2_ROTATION_ROWS=${K1_CASE22_PASS2_ROTATION_ROWS-50293,50295}
if [[ "${PASS2_ROTATION_ROWS}" = none ]]; then
    PASS2_ROTATION_ROWS=
fi
PYTHON=${K1_CASE22_PYTHON:-/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810/.pixi/envs/default/bin/python}
PIXI_ENV=${K1_CASE22_PIXI_ENV:-/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810/.pixi/envs/default}
LAUNCHER=${REPO}/scripts/run_k1_case22_fresh_order_it3.sbatch
DATA=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_currenthead_25ab6e68_20260728T011020ET/cases/22_small_severe_outliers_3k_g128_radial_noise5_bf80/data
SEALED_CONTROL=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_bpref_factor_boundary_7c3b5478_20260810T0610ET/small_control_retry2/native_control/relion
CUDA_LIB=${K1_CASE22_CUDA_LIB:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_bpref_factor_boundary_7c3b5478_20260810T0610ET/small_control_retry2/cuda/libcuda_backproject.so}
RELION_BIND=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_bpref_order_fix_20260810/relion_bind
OUTPUT=${ROOT}/output

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${REPO}" diff --binary | sha256sum | awk '{print $1}')" = "${EXPECTED_REPO_DIFF_SHA256}"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
test -z "$(find "${OUTPUT}" -mindepth 1 -print -quit)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/rattler_cache
export PYTHONPYCACHEPREFIX=${RUNTIME}/${SLURM_JOB_ID}/pycache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/jax_cache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" \
    "${PYTHONPYCACHEPREFIX}" "${JAX_COMPILATION_CACHE_DIR}" "${ROOT}/provenance"

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
export RECOVAR_RELION_BIND_BUILD_DIR=${RELION_BIND}
export RELION_SRC_DIR=/scratch/gpfs/GILLES/mg6942/relion/src
export RECOVAR_EXPECTED_REPO_ROOT=${REPO}
export RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION=0.40
export RECOVAR_INITIAL_PROJECTOR_USE_REAL_REFERENCE=1
export RECOVAR_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN=4e-6
export RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT=0
export RECOVAR_K1_RELION_LIVE_INITIAL_NOISE=1
export RECOVAR_RELION_X_HALF_F32_FINE_POSTERIOR=1
export RECOVAR_K1_RELION_EXACT_BPREF_OPERANDS=1
export RECOVAR_K1_RELION_EXACT_CTF_STAR=${DATA}/particles.star
if [[ "${EXACT_COARSE_STACK1204}" = 1 ]]; then
    GAUSSIAN_COARSE_STACK1204=1
    GAUSSIAN_SINCOSF_STACK1204=1
fi
if [[ "${GAUSSIAN_COARSE_STACK1204}" = 1 ]]; then
    export RECOVAR_K1_COARSE_GAUSSIAN_FFI=1
    export RECOVAR_K1_RELION_F32_COARSE_SUPPORT=1
fi
if [[ "${GAUSSIAN_SINCOSF_STACK1204}" = 1 ]]; then
    export RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF=1
fi
if [[ "${EXACT_COARSE_STACK1204}" = 1 ]]; then
    export RECOVAR_K1_RELION_EXACT_COARSE_OPERANDS=1
fi
unset RECOVAR_K1_BPREF_EXECUTION_ORDER_LOCAL_FILE
unset RECOVAR_FINAL_ALL_DATA_GRID_CORRECT RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER
if [[ "${POWERCLASS_SPECTRUM_NORM}" = 1 ]]; then
    export RECOVAR_K1_RELION_POWERCLASS_SPECTRUM_NORM=1
fi
if [[ "${TRANSLATED_WAVG_NORM}" = 1 ]]; then
    export RECOVAR_K1_RELION_TRANSLATED_WAVG_NORM=1
fi
if [[ "${WAVG_ATOMIC_SCALE_AA}" = 1 ]]; then
    export RECOVAR_RELION_WAVG_ATOMIC_SCALE_AA=1
fi
if [[ "${WAVG_ATOMIC_DIRECT_RESIDUAL}" = 1 ]]; then
    export RECOVAR_RELION_WAVG_ATOMIC_SCALE_AA=1
    export RECOVAR_RELION_WAVG_ATOMIC_DIRECT_RESIDUAL=1
fi
if [[ "${FINE_ROTATION_EXECUTION_ORDER}" = 1 ]]; then
    export RECOVAR_RELION_FINE_ROTATION_EXECUTION_ORDER=1
fi
if [[ "${PARITY_DUMP}" = 1 ]]; then
    export RECOVAR_PARITY_DUMP_DIR=${ROOT}/parity
    export RECOVAR_PARITY_DUMP_VOLUME_DOWNSAMPLE=8
    mkdir -p "${RECOVAR_PARITY_DUMP_DIR}"
fi
if [[ "${NOISE_UPDATE_DUMP}" = 1 ]]; then
    export RECOVAR_NOISE_DEBUG_DUMP_DIR=${ROOT}/noise
    export RECOVAR_NOISE_DEBUG_DUMP_ITERATION=1
    mkdir -p "${RECOVAR_NOISE_DEBUG_DUMP_DIR}"
fi
if [[ "${PROJECTOR_DUMP}" = 1 ]]; then
    export RECOVAR_RELION_PROJECTOR_DUMP_DIR=${ROOT}/projector
    mkdir -p "${RECOVAR_RELION_PROJECTOR_DUMP_DIR}"
fi
if [[ "${COMPACT_STACK1204_DUMP}" = 1 && "${COARSE_STACK1204_DUMP}" = 1 ]]; then
    echo "compact pass-2 and coarse-significance dump modes are mutually exclusive" >&2
    exit 2
elif [[ "${COMPACT_STACK1204_DUMP}" = 1 ]]; then
    export RECOVAR_PASS2_DUMP_DIR=${ROOT}/pass2
    export RECOVAR_PASS2_DUMP_ORIGINAL_INDICES=${TARGET_ORIGINAL_INDEX}
    export RECOVAR_PASS2_DUMP_CURRENT_SIZE=${TARGET_CURRENT_SIZE}
    export RECOVAR_PASS2_DUMP_ITERATION=${TARGET_ITERATION}
    # The compact analyzer reconstructs raw diff2 from the selected projection,
    # image, correction, and high-resolution-tail operands.  Enabling the full
    # raw dump materializes every large bucket and can require >17 GiB for a
    # single diagnostic batch.
    export RECOVAR_PASS2_DUMP_RAW_OPERANDS=0
    if [[ "${NORM_RESIDUAL_INPUTS}" = 1 ]]; then
        export RECOVAR_PASS2_DUMP_NORM_RESIDUAL_INPUTS=1
        export RECOVAR_PASS2_DUMP_NORM_RESIDUAL_ONLY=1
        if [[ "${STOP_AFTER_COMPACT_DUMP}" = 1 ]]; then
            export RECOVAR_PASS2_DUMP_NORM_RESIDUAL_STOP_AFTER_TARGET=1
        fi
    fi
    if [[ -n "${PASS2_ROTATION_ROWS}" ]]; then
        export RECOVAR_PASS2_DUMP_ROTATION_ROWS=${PASS2_ROTATION_ROWS}
    else
        unset RECOVAR_PASS2_DUMP_ROTATION_ROWS
    fi
    # The ordinary pass-2 dump is emitted before the norm/scale operands. Use
    # a distinct stop flag in residual-input mode so both artifacts are written
    # before the focused run exits.
    if [[ "${STOP_AFTER_COMPACT_DUMP}" = 1 && "${NORM_RESIDUAL_INPUTS}" != 1 ]]; then
        export RECOVAR_PASS2_DUMP_STOP_AFTER_TARGET=1
    fi
    mkdir -p "${RECOVAR_PASS2_DUMP_DIR}"
elif [[ "${STOP_AFTER_COMPACT_DUMP}" = 1 ]]; then
    echo "K1_CASE22_STOP_AFTER_COMPACT_DUMP requires K1_CASE22_COMPACT_STACK1204_DUMP=1" >&2
    exit 2
elif [[ "${COARSE_STACK1204_DUMP}" = 1 ]]; then
    export RECOVAR_SIGNIFICANCE_DUMP_DIR=${ROOT}/coarse
    export RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES=${TARGET_ORIGINAL_INDICES}
    export RECOVAR_SIGNIFICANCE_DUMP_CURRENT_SIZE=${TARGET_CURRENT_SIZE}
    export RECOVAR_SIGNIFICANCE_DUMP_ITERATION=${TARGET_ITERATION}
    export RECOVAR_SIGNIFICANCE_DUMP_TARGET_HALF=1
    export RECOVAR_SIGNIFICANCE_DUMP_STOP_AFTER_TARGET=1
    mkdir -p "${RECOVAR_SIGNIFICANCE_DUMP_DIR}"
fi

mkdir -p "${OUTPUT}"
cd "${REPO}"
"${PYTHON}" - <<'PY'
from pathlib import Path
import jax
import recovar
from recovar import cuda_backproject
from recovar.relion_bind import _relion_bind_core as bind

root = Path.cwd().resolve()
assert Path(recovar.__file__).resolve().is_relative_to(root)
assert ".pixi/envs/default/" in str(Path(jax.__file__).resolve())
assert len(jax.devices("gpu")) == 1 and "H100" in jax.devices("gpu")[0].device_kind.upper()
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
assert hasattr(bind, "auto_refine_randomise_half_orders")
print(jax.devices())
print(bind.__file__)
PY

COMMAND=(
    "${PYTHON}" -m scripts.run_full_refinement
    --data_dir "${DATA}" --output "${OUTPUT}" --max_iter "${MAX_ITER}" --skip_final_iteration
    --healpix_order 3 --auto_local_healpix_order 4 --offset_range 3.0 --offset_step 1.0
    --adaptive_oversampling 1 --init_resolution 30.0 --image_batch_size 187
    --rotation_block_size 8192 --seed 1722 --perturb_seed 1722
    --particle_diameter_ang 200 --tau2_fudge 1.0 --max_significants -1
    --firstiter_cc --apply-initial-lowpass --image-fourier-backend relion_cuda
    --benchmark_ledger_json "${OUTPUT}/benchmark_ledger.json"
    --timing_dir "${OUTPUT}/timing"
    --save_intermediates_dir "${OUTPUT}/intermediates"
    --save_intermediates_skip_unregularized
    --relion_half_sets "${SEALED_CONTROL}/run_it000_data.star"
    --relion_optimiser "${SEALED_CONTROL}/run_it000_optimiser.star"
    --relion_init_dir "${SEALED_CONTROL}"
)
printf '%q ' "${COMMAND[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
nvidia-smi -q > "${ROOT}/provenance/nvidia_smi_${SLURM_JOB_ID}.txt"
sha256sum "${LAUNCHER}" "${CUDA_LIB}" "${RELION_BIND}"/_relion_bind_core*.so \
    > "${ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"

start=$(date +%s)
"${COMMAND[@]}" > "${OUTPUT}/run.log" 2>&1
wall_s=$(( $(date +%s) - start ))

if [[ "${STOP_AFTER_COMPACT_DUMP}" = 1 ]]; then
    printf -v target_original_label '%06d' "${TARGET_ORIGINAL_INDEX}"
    printf -v target_size_label '%03d' "${TARGET_CURRENT_SIZE}"
    compact_dump=${RECOVAR_PASS2_DUMP_DIR}/pass2_orig${target_original_label}_cs${target_size_label}.npz
    if [[ "${NORM_RESIDUAL_INPUTS}" = 1 ]]; then
        norm_residual_dump=${RECOVAR_PASS2_DUMP_DIR}/norm_residual_orig${target_original_label}_half1_cs${target_size_label}.npz
        chunked_scale_aa_dump=${RECOVAR_PASS2_DUMP_DIR}/scale_aa_chunked_orig${target_original_label}_half1_cs${target_size_label}.npz
        test -s "${norm_residual_dump}" || test -s "${chunked_scale_aa_dump}"
        rg -q 'requested fine-score or norm/scale boundary' "${OUTPUT}/run.log"
    else
        test -s "${compact_dump}"
        rg -q 'pass-2 operand dump completed; stopping at the requested fine-score boundary' \
            "${OUTPUT}/run.log"
    fi
    if [[ "${PROJECTOR_DUMP}" = 1 ]]; then
        test -s "${RECOVAR_RELION_PROJECTOR_DUMP_DIR}/iter001_half0_relion_projector_half.npz"
    fi
    printf '{"job_id":"%s","wall_s":%d,"max_iter":%d,"guard":"fresh_k1_only","compact_stack1204_dump":true,"norm_residual_inputs":%s,"projector_dump":%s,"target_stack_one_based":%d,"target_iteration":%d,"target_current_size":%d,"pass2_rotation_rows":"%s","stopped_after_dump":true}\n' \
        "${SLURM_JOB_ID}" "${wall_s}" "${MAX_ITER}" "${NORM_RESIDUAL_INPUTS}" "${PROJECTOR_DUMP}" "${TARGET_STACK_ONE_BASED}" \
        "${TARGET_ITERATION}" "${TARGET_CURRENT_SIZE}" "${PASS2_ROTATION_ROWS}" \
        > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
    touch "${ROOT}/DUMP_SUCCESS_${SLURM_JOB_ID}"
    exit 0
fi
if [[ "${COARSE_STACK1204_DUMP}" = 1 ]]; then
    printf -v target_iteration_label '%03d' "${TARGET_ITERATION}"
    printf -v target_size_label '%03d' "${TARGET_CURRENT_SIZE}"
    for target_original in ${TARGET_ORIGINAL_INDICES//,/ }; do
        printf -v target_original_label '%06d' "${target_original}"
        coarse_dump=${RECOVAR_SIGNIFICANCE_DUMP_DIR}/significance_orig${target_original_label}_it${target_iteration_label}_cs${target_size_label}.npz
        test -s "${coarse_dump}"
    done
    rg -q 'coarse-significance dump completed; stopping before pass-2/M-step work' \
        "${OUTPUT}/run.log"
    printf '{"job_id":"%s","wall_s":%d,"max_iter":%d,"guard":"fresh_k1_only","coarse_stack1204_dump":true,"target_stacks_one_based":"%s","target_iteration":%d,"target_current_size":%d,"gaussian_coarse":%s,"gaussian_sincosf":%s,"exact_coarse":%s,"stopped_after_dump":true}\n' \
        "${SLURM_JOB_ID}" "${wall_s}" "${MAX_ITER}" "${TARGET_STACKS_ONE_BASED}" "${TARGET_ITERATION}" "${TARGET_CURRENT_SIZE}" "${GAUSSIAN_COARSE_STACK1204}" "${GAUSSIAN_SINCOSF_STACK1204}" "${EXACT_COARSE_STACK1204}" \
        > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
    touch "${ROOT}/DUMP_SUCCESS_${SLURM_JOB_ID}"
    exit 0
fi

last_index=$((MAX_ITER - 1))
printf -v last_label '%03d' "${last_index}"
test -s "${OUTPUT}/intermediates/it${last_label}_half1_reg.mrc"
test -s "${OUTPUT}/intermediates/it${last_label}_half2_reg.mrc"
if [[ "${NOISE_UPDATE_DUMP}" = 1 ]]; then
    test -s "${RECOVAR_NOISE_DEBUG_DUMP_DIR}/recovar_noise_update_it002.npz"
fi
if [[ "${COMPACT_STACK1204_DUMP}" = 1 ]]; then
    printf -v target_original_label '%06d' "${TARGET_ORIGINAL_INDEX}"
    printf -v target_size_label '%03d' "${TARGET_CURRENT_SIZE}"
    compact_dump=${RECOVAR_PASS2_DUMP_DIR}/pass2_orig${target_original_label}_cs${target_size_label}.npz
    test -s "${compact_dump}"
    if [[ -n "${PASS2_ROTATION_ROWS}" ]]; then
        "${PYTHON}" - <<'PY' "${compact_dump}" "${PASS2_ROTATION_ROWS}"
import numpy as np
import sys

with np.load(sys.argv[1], allow_pickle=False) as payload:
    assert str(payload["schema"]) == "recovar.em.k1_pass2_selected_rotations.v1"
    expected = np.asarray([int(value) for value in sys.argv[2].split(",")], dtype=np.int64)
    assert np.array_equal(payload["rotation_rows_global"], expected)
PY
    else
        "${PYTHON}" - <<'PY' "${compact_dump}" "${TARGET_ITERATION}" "${TARGET_CURRENT_SIZE}"
import numpy as np
import sys

with np.load(sys.argv[1], allow_pickle=False) as payload:
    assert "schema" not in payload.files
    assert int(payload["iteration"]) == int(sys.argv[2])
    assert int(payload["current_size"]) == int(sys.argv[3])
    assert payload["scores_with_prior"].shape == payload["probs"].shape
    assert payload["candidate_mask"].shape == payload["probs"].shape
PY
    fi
    if [[ "${NORM_RESIDUAL_INPUTS}" = 1 ]]; then
        norm_residual_dump=${RECOVAR_PASS2_DUMP_DIR}/norm_residual_orig${target_original_label}_half1_cs${target_size_label}.npz
        test -s "${norm_residual_dump}"
    fi
fi
! rg -q 'Traceback|RESOURCE_EXHAUSTED|Out of memory|accounting is inconsistent' "${OUTPUT}/run.log"
rg -q 'Applied RELION fresh paired AutoRefine particle order with effective seed 1723' "${OUTPUT}/run.log"
rg -q 'preserving fresh RELION physical BPref particle order' "${OUTPUT}/run.log"
rg -q 'expected accuracy consumes an explicit physical trial order' "${OUTPUT}/run.log"
if [[ "${PARITY_DUMP}" = 1 ]]; then
    test -s "${RECOVAR_PARITY_DUMP_DIR}/iter_001.npz"
fi

printf '{"job_id":"%s","wall_s":%d,"max_iter":%d,"guard":"fresh_k1_only","compact_stack1204_dump":%s,"norm_residual_inputs":%s,"powerclass_spectrum_norm":%s,"translated_wavg_norm":%s,"wavg_atomic_scale_aa":%s,"wavg_atomic_direct_residual":%s,"parity_dump":%s,"noise_update_dump":%s}\n' \
    "${SLURM_JOB_ID}" "${wall_s}" "${MAX_ITER}" "${COMPACT_STACK1204_DUMP}" "${NORM_RESIDUAL_INPUTS}" "${POWERCLASS_SPECTRUM_NORM}" "${TRANSLATED_WAVG_NORM}" "${WAVG_ATOMIC_SCALE_AA}" "${WAVG_ATOMIC_DIRECT_RESIDUAL}" "${PARITY_DUMP}" "${NOISE_UPDATE_DUMP}" \
    > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
find "${OUTPUT}" -type f -print0 | sort -z | xargs -0 sha256sum \
    > "${ROOT}/provenance/science_outputs_${SLURM_JOB_ID}.sha256"
touch "${ROOT}/RUN_SUCCESS_${SLURM_JOB_ID}"
