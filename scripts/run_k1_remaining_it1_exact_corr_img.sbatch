#!/usr/bin/env bash
#SBATCH --job-name=k1-rem-it1-corr
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=00:45:00

# Bounded iteration-1/2 gate or autonomous terminal run for the fixed-suite K=1
# failures. The submitter supplies isolated roots, mode, and Slurm limits.
set -euo pipefail

: "${K1_CASE_INDEX:?set K1_CASE_INDEX to 4, 5, or 10}"
: "${RUN_ROOT:?set an isolated run root}"
: "${RUNTIME_ROOT:?set an isolated runtime root}"
: "${EXPECTED_REPO_HEAD:?submission must pin repository HEAD}"
: "${EXPECTED_SOURCE_DIFF_SHA256:?submission must pin the RECOVAR source diff}"
: "${EXPECTED_LAUNCHER_SHA256:?submission must pin this launcher}"
: "${EXPECTED_CUDA_LIB_SHA256:?submission must pin the CUDA library}"

MAX_ITER=${K1_MAX_ITER:-1}
SKIP_FINAL=${K1_SKIP_FINAL:-1}
BPREF_NOISE_OVERRIDE=${BPREF_NOISE_VARIANCE_OVERRIDE_PATH:-}
PASS2_DUMP_ORIGINAL_INDICES=${K1_PASS2_DUMP_ORIGINAL_INDICES:-}
BPREF_PARTICLE_POOL_SIZE=${K1_BPREF_PARTICLE_POOL_SIZE:-1}
BPREF_EXECUTION_ORDER_CHUNK_SIZE=${K1_BPREF_EXECUTION_ORDER_CHUNK_SIZE:-220}
ENABLE_LIVE_INITIAL_NOISE=${K1_ENABLE_LIVE_INITIAL_NOISE:-0}
ENABLE_NATIVE_TEXTURE=${K1_ENABLE_NATIVE_TEXTURE:-0}
FORCE_FULL_COARSE_GRID=${K1_FORCE_FULL_COARSE_GRID:-0}
SERIAL_PARTICLE_LAUNCHES=${K1_SERIAL_PARTICLE_LAUNCHES:-0}
NATIVE_FINE_SCORE_UNITS=${K1_NATIVE_FINE_SCORE_UNITS:-1}
TRANSLATION_ANGLE_SCALE=${K1_TRANSLATION_ANGLE_SCALE:-}
INTERNAL_FOURIER_DUMP_DIR=${K1_INTERNAL_FOURIER_DUMP_DIR:-}
INTERNAL_FOURIER_DUMP_ITERATION=${K1_INTERNAL_FOURIER_DUMP_ITERATION:-}
ENABLE_PARITY_DUMP=${K1_ENABLE_PARITY_DUMP:-}
if [[ -z "${ENABLE_PARITY_DUMP}" ]]; then
    if [[ "${MAX_ITER}" == 2 ]]; then
        ENABLE_PARITY_DUMP=1
    else
        ENABLE_PARITY_DUMP=0
    fi
fi
if [[ "${MAX_ITER}" != 1 && "${MAX_ITER}" != 2 && "${MAX_ITER}" != 999 ]]; then
    echo "K1_MAX_ITER must be 1, 2, or 999" >&2
    exit 2
fi
if [[ "${SKIP_FINAL}" != 0 && "${SKIP_FINAL}" != 1 ]]; then
    echo "K1_SKIP_FINAL must be 0 or 1" >&2
    exit 2
fi
if [[ "${ENABLE_LIVE_INITIAL_NOISE}" != 0 && "${ENABLE_LIVE_INITIAL_NOISE}" != 1 ]]; then
    echo "K1_ENABLE_LIVE_INITIAL_NOISE must be 0 or 1" >&2
    exit 2
fi
for binary_setting in ENABLE_NATIVE_TEXTURE FORCE_FULL_COARSE_GRID SERIAL_PARTICLE_LAUNCHES NATIVE_FINE_SCORE_UNITS; do
    if [[ "${!binary_setting}" != 0 && "${!binary_setting}" != 1 ]]; then
        echo "${binary_setting} must be 0 or 1" >&2
        exit 2
    fi
done
if [[ "${FORCE_FULL_COARSE_GRID}" == 1 && "${ENABLE_NATIVE_TEXTURE}" != 1 && "${SERIAL_PARTICLE_LAUNCHES}" != 1 ]]; then
    echo "K1_FORCE_FULL_COARSE_GRID requires native texture or serial particle launches" >&2
    exit 2
fi
if [[ "${ENABLE_PARITY_DUMP}" != 0 && "${ENABLE_PARITY_DUMP}" != 1 ]]; then
    echo "K1_ENABLE_PARITY_DUMP must be 0 or 1" >&2
    exit 2
fi
if [[ -n "${PASS2_DUMP_ORIGINAL_INDICES}" && "${MAX_ITER}" != 2 ]]; then
    echo "K1_PASS2_DUMP_ORIGINAL_INDICES requires K1_MAX_ITER=2" >&2
    exit 2
fi
if [[ -n "${INTERNAL_FOURIER_DUMP_DIR}" || -n "${INTERNAL_FOURIER_DUMP_ITERATION}" ]]; then
    if [[ -z "${INTERNAL_FOURIER_DUMP_DIR}" || -z "${INTERNAL_FOURIER_DUMP_ITERATION}" ]]; then
        echo "K1 internal Fourier dump directory and iteration must be supplied together" >&2
        exit 2
    fi
    if [[ "${INTERNAL_FOURIER_DUMP_DIR:0:1}" != / ]]; then
        echo "K1_INTERNAL_FOURIER_DUMP_DIR must be absolute" >&2
        exit 2
    fi
fi
case "${BPREF_PARTICLE_POOL_SIZE}" in
    ''|*[!0-9]*|0*) echo "K1_BPREF_PARTICLE_POOL_SIZE must be a canonical positive integer" >&2; exit 2 ;;
esac
case "${BPREF_EXECUTION_ORDER_CHUNK_SIZE}" in
    ''|*[!0-9]*|0*) echo "K1_BPREF_EXECUTION_ORDER_CHUNK_SIZE must be a canonical positive integer" >&2; exit 2 ;;
esac
if (( BPREF_PARTICLE_POOL_SIZE > 1 )) && [[ "${MAX_ITER}" != 1 ]]; then
    echo "K1_BPREF_PARTICLE_POOL_SIZE > 1 is qualified only for K1_MAX_ITER=1" >&2
    exit 2
fi

case "${K1_CASE_INDEX}" in
    4)
        CASE_NAME=4_high_noise_100k_g256_white_noise3_bf80
        SEED=1704
        ;;
    5)
        CASE_NAME=5_very_high_noise_100k_g256_white_noise10_bf80
        SEED=1705
        ;;
    10)
        CASE_NAME=10_high_res_anisotropic_100k_g384_radial_noise3_bf0
        SEED=1710
        ;;
    *)
        echo "K1_CASE_INDEX must be 4, 5, or 10" >&2
        exit 2
        ;;
esac

ROOT=${RUN_ROOT}
RUNTIME=${RUNTIME_ROOT}
REPO=${K1_REPO_OVERRIDE:-/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820}
FIXED_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET
CASE=${FIXED_ROOT}/cases/${CASE_NAME}
DATA=${CASE}/data
RELION=${CASE}/relion_ref
OUTPUT=${ROOT}/output
PIXI_ENV=${K1_PIXI_ENV_OVERRIDE:-${REPO}/.pixi/envs/default}
PYTHON=${PIXI_ENV}/bin/python
export PIXI_ENV
CUDA_LIB=${CUDA_LIB_OVERRIDE:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_preprocess_stage_split_20260821T1600ET/cuda/libcuda_backproject.so}
RELION_BIND=${FIXED_ROOT}/relion_bind_build/shared
LAUNCHER=${REPO}/scripts/run_k1_remaining_it1_exact_corr_img.sbatch

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test -z "$(find "${OUTPUT}" -mindepth 1 -print -quit)"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${REPO}" diff --binary -- recovar/em recovar/cuda/cuda_backproject.cu recovar/cuda_backproject.py scripts/run_full_refinement.py | sha256sum | awk '{print $1}')" = "${EXPECTED_SOURCE_DIFF_SHA256}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = "${EXPECTED_CUDA_LIB_SHA256}"
if [[ -n "${BPREF_NOISE_OVERRIDE}" ]]; then
    : "${EXPECTED_BPREF_NOISE_SHA256:?BPref noise override must pin its SHA-256}"
    test "${BPREF_NOISE_OVERRIDE:0:1}" = / -a -s "${BPREF_NOISE_OVERRIDE}"
    test "$(sha256sum "${BPREF_NOISE_OVERRIDE}" | awk '{print $1}')" = "${EXPECTED_BPREF_NOISE_SHA256}"
fi
if [[ -n "${INTERNAL_FOURIER_DUMP_DIR}" ]]; then
    mkdir -p "${INTERNAL_FOURIER_DUMP_DIR}"
    test -z "$(find "${INTERNAL_FOURIER_DUMP_DIR}" -mindepth 1 -print -quit)"
fi
RELION_BIND_LIB=$(find "${RELION_BIND}" -maxdepth 1 -type f -name '_relion_bind_core*.so' -print -quit)
test "$(sha256sum "${RELION_BIND_LIB}" | awk '{print $1}')" = 56acdddab6d25bd83aac5e338162c4819f85ec26f68511458618f143c2b604b7
STACK=$(find -L "${DATA}" -maxdepth 1 -type f -name 'particles.*.mrcs' -print -quit)
test -n "${STACK}" -a -s "${STACK}"
test -s "${DATA}/particles.star"
test -s "${DATA}/fixture_materialization.json"
test -s "${RELION}/run_it000_data.star" -a -s "${RELION}/run_it000_optimiser.star"
test -s "${RELION}/run_it001_half1_class001.mrc" -a -s "${RELION}/run_it001_half2_class001.mrc"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_*) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/rattler_cache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/jax_cache
export PYTHONPYCACHEPREFIX=${RUNTIME}/${SLURM_JOB_ID}/pycache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" \
    "${JAX_COMPILATION_CACHE_DIR}" "${PYTHONPYCACHEPREFIX}" "${OUTPUT}/timing"

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
export RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE=0
export RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT=0
export RECOVAR_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE=${ENABLE_NATIVE_TEXTURE}
if [[ "${FORCE_FULL_COARSE_GRID}" == 1 ]]; then
    export RECOVAR_K1_COARSE_GAUSSIAN_FORCE_FULL_ROTATION_GRID_DIAGNOSTIC=1
fi
if [[ "${SERIAL_PARTICLE_LAUNCHES}" == 1 ]]; then
    export RECOVAR_K1_COARSE_GAUSSIAN_SERIAL_PARTICLE_LAUNCHES_DIAGNOSTIC=1
fi
export RECOVAR_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN=4e-6
export RECOVAR_K1_RELION_LIVE_INITIAL_NOISE=${ENABLE_LIVE_INITIAL_NOISE}
export RECOVAR_RELION_X_HALF_F32_FINE_POSTERIOR=1
export RECOVAR_K1_RELION_POWERCLASS_SPECTRUM_NORM=1
export RECOVAR_K1_RELION_EXACT_BPREF_OPERANDS=1
export RECOVAR_K1_RELION_NATIVE_FINE_SCORE_UNITS=${NATIVE_FINE_SCORE_UNITS}
export RECOVAR_K1_RELION_EXACT_CTF_STAR=${DATA}/particles.star
export RECOVAR_K1_BPREF_EXECUTION_ORDER_CHUNK_SIZE=${BPREF_EXECUTION_ORDER_CHUNK_SIZE}
export RECOVAR_K1_RELION_X_HALF_BP_PARTICLE_POOL_SIZE=${BPREF_PARTICLE_POOL_SIZE}
if [[ -n "${TRANSLATION_ANGLE_SCALE}" ]]; then
    export RECOVAR_K1_RELION_TRANSLATION_ANGLE_SCALE=${TRANSLATION_ANGLE_SCALE}
fi
if [[ -n "${BPREF_NOISE_OVERRIDE}" ]]; then
    export RECOVAR_K1_BPREF_NOISE_VARIANCE_OVERRIDE_PATH=${BPREF_NOISE_OVERRIDE}
fi
if [[ -n "${INTERNAL_FOURIER_DUMP_DIR}" ]]; then
    export RECOVAR_K1_INTERNAL_FOURIER_DUMP_DIR=${INTERNAL_FOURIER_DUMP_DIR}
    export RECOVAR_K1_INTERNAL_FOURIER_DUMP_ITERATION=${INTERNAL_FOURIER_DUMP_ITERATION}
fi
if [[ "${MAX_ITER}" != 1 ]]; then
    export RECOVAR_SAVE_INTERMEDIATES_DIR=1
    export RECOVAR_SAVE_INTERMEDIATES_SKIP_UNREGULARIZED=1
fi
if [[ "${ENABLE_PARITY_DUMP}" == 1 ]]; then
    export RECOVAR_PARITY_DUMP_DIR=${ROOT}/parity
    export RECOVAR_PARITY_DUMP_VOLUME_DOWNSAMPLE=8
    mkdir -p "${RECOVAR_PARITY_DUMP_DIR}"
fi
if [[ -n "${PASS2_DUMP_ORIGINAL_INDICES}" ]]; then
    if [[ "${K1_CASE_INDEX}" != 10 ]]; then
        echo "Focused pass-2 capture is currently qualified only for case 10" >&2
        exit 2
    fi
    export RECOVAR_PASS2_DUMP_DIR=${ROOT}/pass2
    export RECOVAR_PASS2_DUMP_ORIGINAL_INDICES=${PASS2_DUMP_ORIGINAL_INDICES}
    export RECOVAR_PASS2_DUMP_CURRENT_SIZE=58
    export RECOVAR_PASS2_DUMP_ITERATION=2
    export RECOVAR_PASS2_DUMP_CLASS=1
    export RECOVAR_PASS2_DUMP_RAW_OPERANDS=1
    mkdir -p "${RECOVAR_PASS2_DUMP_DIR}"
fi
unset RECOVAR_FINAL_ALL_DATA_GRID_CORRECT RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER

nvidia-smi --query-gpu=uuid,name,memory.total --format=csv,noheader > "${ROOT}/provenance/gpu_${SLURM_JOB_ID}.txt"
env | sort > "${ROOT}/provenance/environment_${SLURM_JOB_ID}.txt"
git -C "${REPO}" status --short --branch > "${ROOT}/provenance/git_status_${SLURM_JOB_ID}.txt"
git -C "${REPO}" diff --binary -- recovar/em recovar/cuda/cuda_backproject.cu \
    recovar/cuda_backproject.py scripts/run_full_refinement.py \
    > "${ROOT}/provenance/recovar_source_diff_${SLURM_JOB_ID}.patch"

cd "${REPO}"
"${PYTHON}" - <<'PY'
import os
from pathlib import Path
import jax
import recovar
from recovar import cuda_backproject

root = Path.cwd().resolve()
assert Path(recovar.__file__).resolve().is_relative_to(root)
pixi_env = Path(os.environ["PIXI_ENV"]).resolve()
assert Path(jax.__file__).resolve().is_relative_to(pixi_env)
assert len(jax.devices("gpu")) == 1 and "H100" in jax.devices("gpu")[0].device_kind.upper()
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
PY

FINAL_ARGS=()
if [[ "${SKIP_FINAL}" == 1 ]]; then
    FINAL_ARGS+=(--skip_final_iteration)
fi
INTERMEDIATE_ARGS=()
if [[ "${MAX_ITER}" != 1 || -n "${INTERNAL_FOURIER_DUMP_DIR}" ]]; then
    mkdir -p "${OUTPUT}/intermediates"
    INTERMEDIATE_ARGS+=(
        --save_intermediates_dir "${OUTPUT}/intermediates"
        --save_intermediates_skip_unregularized
    )
fi
command=("${PYTHON}" -m scripts.run_full_refinement
    --data_dir "${DATA}" --output "${OUTPUT}" --max_iter "${MAX_ITER}"
    --healpix_order 3 --auto_local_healpix_order 4 --offset_range 3.0 --offset_step 1.0
    --adaptive_oversampling 1 --init_resolution 30.0 --image_batch_size 187
    --rotation_block_size 8192 --seed "${SEED}" --perturb_seed "${SEED}"
    --particle_diameter_ang 200 --tau2_fudge 1.0 --max_significants -1
    --firstiter_cc --apply-initial-lowpass --image-fourier-backend relion_cuda
    --benchmark_ledger_json "${OUTPUT}/benchmark_ledger.json"
    --timing_dir "${OUTPUT}/timing"
    --relion_half_sets "${RELION}/run_it000_data.star"
    --relion_optimiser "${RELION}/run_it000_optimiser.star"
    --relion_init_dir "${RELION}"
    "${INTERMEDIATE_ARGS[@]}"
    "${FINAL_ARGS[@]}")
printf '%q ' "${command[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
start_epoch=$(date +%s)
"${command[@]}" > "${OUTPUT}/run_full_refinement.log" 2>&1
wall_s=$(( $(date +%s) - start_epoch ))

test -s "${OUTPUT}/refinement_results.npz"
rg -q 'Refinement complete' "${OUTPUT}/run_full_refinement.log"
! rg -q 'Traceback|RESOURCE_EXHAUSTED|Out of memory|accounting is inconsistent' "${OUTPUT}/run_full_refinement.log"
if [[ "${ENABLE_PARITY_DUMP}" == 1 ]]; then
    for ((iteration = 1; iteration <= MAX_ITER; iteration++)); do
        printf -v iteration_tag '%03d' "${iteration}"
        test -s "${RECOVAR_PARITY_DUMP_DIR}/iter_${iteration_tag}.npz"
    done
fi
if [[ -n "${PASS2_DUMP_ORIGINAL_INDICES}" ]]; then
    "${PYTHON}" - "${RECOVAR_PASS2_DUMP_DIR}" \
        "${PASS2_DUMP_ORIGINAL_INDICES}" <<'PY'
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
expected = {int(value) for value in sys.argv[2].split(",") if value}
observed = set()
for path in sorted(root.glob("pass2_orig*_cs058.npz")):
    with np.load(path, allow_pickle=False) as archive:
        if "raw_operand_schema" not in archive:
            raise RuntimeError(f"raw score operands missing from {path}")
        observed.add(int(archive["original_index"]))
if observed != expected:
    raise RuntimeError(
        f"raw-score source rows {sorted(observed)} != expected {sorted(expected)}"
    )
PY
fi
find "${OUTPUT}" -type f -print0 | sort -z | xargs -0 sha256sum > "${ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
if [[ -n "${PASS2_DUMP_ORIGINAL_INDICES}" ]]; then
    find "${RECOVAR_PASS2_DUMP_DIR}" -type f -print0 \
        | sort -z | xargs -0 sha256sum >> "${ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
fi
if [[ -n "${INTERNAL_FOURIER_DUMP_DIR}" ]]; then
    printf -v internal_iteration_tag '%03d' "${INTERNAL_FOURIER_DUMP_ITERATION}"
    test -s "${INTERNAL_FOURIER_DUMP_DIR}/it${internal_iteration_tag}_half1_mean_ft.npz"
    test -s "${INTERNAL_FOURIER_DUMP_DIR}/it${internal_iteration_tag}_half2_mean_ft.npz"
    find "${INTERNAL_FOURIER_DUMP_DIR}" -type f -print0 \
        | sort -z | xargs -0 sha256sum >> "${ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
fi
printf '{"job_id":"%s","wall_s":%d,"case_index":%d,"max_iter":%d,"skip_final":%d,"live_initial_noise":%d,"native_texture":%d,"full_coarse_grid":%d,"serial_particle_launches":%d,"native_fine_score_units":%d,"translation_angle_scale":"%s","parity_dump":%d}\n' \
    "${SLURM_JOB_ID}" "${wall_s}" "${K1_CASE_INDEX}" "${MAX_ITER}" "${SKIP_FINAL}" \
    "${ENABLE_LIVE_INITIAL_NOISE}" "${ENABLE_NATIVE_TEXTURE}" "${FORCE_FULL_COARSE_GRID}" \
    "${SERIAL_PARTICLE_LAUNCHES}" "${NATIVE_FINE_SCORE_UNITS}" "${TRANSLATION_ANGLE_SCALE}" \
    "${ENABLE_PARITY_DUMP}" \
    > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
