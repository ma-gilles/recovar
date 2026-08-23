#!/usr/bin/env bash
#SBATCH --job-name=k1c04-norm-prefix
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

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${MAX_ITER:=2}"
: "${ROTATION_BLOCK_SIZE:=8192}"
: "${SPECTRUM_NORM:=1}"
: "${ENABLE_EXACT_BPREF:=1}"
: "${ENABLE_LIVE_INITIAL_NOISE:=1}"
: "${BPREF_PARTICLE_POOL_SIZE:=}"
: "${BPREF_EXECUTION_ORDER_CHUNK_SIZE:=}"
: "${BPREF_BLOCK_TOPOLOGY:=}"
: "${CASE_SOURCE:=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET/cases/4_high_noise_100k_g256_white_noise3_bf80}"
: "${SEED:=1704}"
: "${EXPECTED_REPO_HEAD:=}"
: "${EXPECTED_REPO_DIFF_SHA256:=}"
: "${EXPECTED_LAUNCHER_SHA256:=}"

for setting in SPECTRUM_NORM ENABLE_EXACT_BPREF ENABLE_LIVE_INITIAL_NOISE; do
    case "${!setting}" in
        0|1) ;;
        *) echo "${setting} must be 0 or 1" >&2; exit 2 ;;
    esac
done
if [[ -n "${BPREF_PARTICLE_POOL_SIZE}${BPREF_EXECUTION_ORDER_CHUNK_SIZE}${BPREF_BLOCK_TOPOLOGY}" ]]; then
    if [[ "${BPREF_PARTICLE_POOL_SIZE}:${BPREF_BLOCK_TOPOLOGY}" != "3:1" ]] \
        || (( BPREF_EXECUTION_ORDER_CHUNK_SIZE % BPREF_PARTICLE_POOL_SIZE != 0 )); then
        echo "particle-pool diagnostic requires pool size 3, a chunk size divisible by 3, and block topology 1" >&2
        exit 2
    fi
fi
if ! [[ "${ROTATION_BLOCK_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ROTATION_BLOCK_SIZE must be a positive integer" >&2
    exit 2
fi

ROOT=${RUN_ROOT}
RUNTIME=${RUNTIME_ROOT}
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
CASE=${CASE_SOURCE}
DATA=${CASE}/data
RELION=${CASE}/relion_ref
PYTHON=${REPO}/.pixi/envs/default/bin/python
LAUNCHER=${REPO}/scripts/run_k1_case04_norm_spectrum_prefix.sbatch
CUDA_LIB=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_preprocess_stage_split_20260821T1600ET/cuda/libcuda_backproject.so
RELION_BIND=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET/relion_bind_build/shared

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
[[ "${SEED}" =~ ^[0-9]+$ ]]
if [[ -n "${EXPECTED_REPO_HEAD}" ]]; then
    test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
fi
if [[ -n "${EXPECTED_REPO_DIFF_SHA256}" ]]; then
    test "$(git -C "${REPO}" diff --binary | sha256sum | awk '{print $1}')" = "${EXPECTED_REPO_DIFF_SHA256}"
fi
if [[ -n "${EXPECTED_LAUNCHER_SHA256}" ]]; then
    test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
fi
mkdir -p "${ROOT}/logs" "${ROOT}/output" "${ROOT}/parity" "${ROOT}/provenance"
mkdir -p "${RUNTIME}/tmp" "${RUNTIME}/pixi_home" "${RUNTIME}/rattler_cache" "${RUNTIME}/pycache" "${RUNTIME}/jax_cache"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_*) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME}/tmp
export PIXI_HOME=${RUNTIME}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/rattler_cache
export PYTHONPYCACHEPREFIX=${RUNTIME}/pycache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME}/jax_cache

source /etc/profile.d/modules.sh
set +u
module purge
module load cudatoolkit/12.8
set -u
PIXI_ENV=${REPO}/.pixi/envs/default
NVIDIA_ROOT=$(find "${PIXI_ENV}/lib" -maxdepth 3 -type d -path '*/site-packages/nvidia' -print -quit)
NVIDIA_LIBS=$(find "${NVIDIA_ROOT}" -type d -name lib | paste -sd: -)
export LD_LIBRARY_PATH=${NVIDIA_LIBS}:/usr/local/cuda-12.8/targets/x86_64-linux/lib:${PIXI_ENV}/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=0
export RECOVAR_CUDA_LIB=${CUDA_LIB}
export RECOVAR_RELION_BIND_BUILD_DIR=${RELION_BIND}
export RECOVAR_K1_RELION_LIVE_INITIAL_NOISE=${ENABLE_LIVE_INITIAL_NOISE}
export RECOVAR_K1_RELION_EXACT_BPREF_OPERANDS=${ENABLE_EXACT_BPREF}
export RECOVAR_K1_RELION_EXACT_CTF_STAR=${DATA}/particles.star
export RECOVAR_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN=4e-6
export RECOVAR_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE=0
export RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT=0
export RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION=.40
export RECOVAR_RELION_X_HALF_F32_FINE_POSTERIOR=1
export RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE=0
export RECOVAR_K1_RELION_POWERCLASS_SPECTRUM_NORM=${SPECTRUM_NORM}
if [[ -n "${BPREF_PARTICLE_POOL_SIZE}" ]]; then
    export RECOVAR_K1_RELION_X_HALF_BP_PARTICLE_POOL_SIZE=${BPREF_PARTICLE_POOL_SIZE}
    export RECOVAR_K1_BPREF_EXECUTION_ORDER_CHUNK_SIZE=${BPREF_EXECUTION_ORDER_CHUNK_SIZE}
    export RECOVAR_RELION_X_HALF_BP_BLOCK_TOPOLOGY=${BPREF_BLOCK_TOPOLOGY}
fi
export RECOVAR_PARITY_DUMP_DIR=${ROOT}/parity
export RECOVAR_PARITY_DUMP_TAG=68694
export RECOVAR_PARITY_DUMP_VOLUME_DOWNSAMPLE=8
unset RECOVAR_FINAL_ALL_DATA_GRID_CORRECT RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER

cd "${REPO}"
git rev-parse HEAD > "${ROOT}/provenance/repo_head_${SLURM_JOB_ID}.txt"
git status --short > "${ROOT}/provenance/repo_status_${SLURM_JOB_ID}.txt"
git diff --binary | sha256sum > "${ROOT}/provenance/repo_diff_${SLURM_JOB_ID}.sha256"
sha256sum "${CUDA_LIB}" "${DATA}/particles.star" "${RELION}/run_it000_optimiser.star" \
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
assert Path(jax.__file__).resolve().is_relative_to(repo / ".pixi/envs/default")
assert len(jax.devices("gpu")) == 1 and "H100" in jax.devices("gpu")[0].device_kind.upper()
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
print(jax.devices())
PY

command=(
    "${PYTHON}" -m scripts.run_full_refinement
    --data_dir "${DATA}" --output "${ROOT}/output" --max_iter "${MAX_ITER}" --skip_final_iteration
    --healpix_order 3 --auto_local_healpix_order 4 --offset_range 3.0 --offset_step 1.0
    --adaptive_oversampling 1 --init_resolution 30.0
    --image_batch_size 187 --rotation_block_size "${ROTATION_BLOCK_SIZE}"
    --seed "${SEED}" --perturb_seed "${SEED}" --particle_diameter_ang 200
    --tau2_fudge 1.0 --max_significants -1 --firstiter_cc --apply-initial-lowpass
    --image-fourier-backend relion_cuda
    --save_intermediates_dir "${ROOT}/output/intermediates"
    --save_intermediates_skip_unregularized
    --relion_half_sets "${RELION}/run_it000_data.star"
    --relion_optimiser "${RELION}/run_it000_optimiser.star"
    --relion_init_dir "${RELION}"
)
printf '%q ' "${command[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
start_epoch=$(date +%s)
"${command[@]}" > "${ROOT}/logs/run_${SLURM_JOB_ID}.log" 2>&1
wall_s=$(( $(date +%s) - start_epoch ))

! rg -q 'Traceback|RESOURCE_EXHAUSTED|Out of memory|accounting is inconsistent' "${ROOT}/logs/run_${SLURM_JOB_ID}.log"
for iteration in $(seq 1 "${MAX_ITER}"); do
    test -s "${ROOT}/parity/iter_$(printf '%03d' "${iteration}").npz"
done
test -s "${ROOT}/output/refinement_results.npz"
sha256sum "${ROOT}"/parity/iter_*.npz "${ROOT}/output/refinement_results.npz" \
    > "${ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
printf '{"job_id":"%s","wall_s":%d,"max_iter":%d,"rotation_block_size":%d,"mode":"spectrum_factorial","case_source":"%s","seed":%d,"spectrum_norm":%d,"exact_bpref":%d,"live_initial_noise":%d,"bpref_particle_pool_size":"%s","bpref_execution_order_chunk_size":"%s","bpref_block_topology":"%s"}\n' \
    "${SLURM_JOB_ID}" "${wall_s}" "${MAX_ITER}" "${ROTATION_BLOCK_SIZE}" "${CASE}" "${SEED}" \
    "${SPECTRUM_NORM}" "${ENABLE_EXACT_BPREF}" "${ENABLE_LIVE_INITIAL_NOISE}" \
    "${BPREF_PARTICLE_POOL_SIZE}" "${BPREF_EXECUTION_ORDER_CHUNK_SIZE}" "${BPREF_BLOCK_TOPOLOGY}" \
    > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
