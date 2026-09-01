#!/usr/bin/env bash
#SBATCH --job-name=k1c10-it7-tail-h1
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=00:30:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_SOURCE_DIFF_SHA256:?EXPECTED_SOURCE_DIFF_SHA256 is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_CUDA_LIB_SHA256:?EXPECTED_CUDA_LIB_SHA256 is required}"

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
FIXED=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET
CASE=${FIXED}/cases/10_high_res_anisotropic_100k_g384_radial_noise3_bf0
DATA=${CASE}/data
RELION=${CASE}/relion_ref
ROOT=${RUN_ROOT}
RUNTIME=${RUNTIME_ROOT}
OUTPUT=${ROOT}/output
PASS2=${ROOT}/pass2
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default
CUDA_LIB=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case10_exact_it15_final_20260823T0230ET/cuda/libcuda_backproject.so
RELION_BIND=${FIXED}/relion_bind_build/shared
LAUNCHER=${REPO}/scripts/run_k1_case10_it7_pose_tail_h1.sbatch

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test -z "$(find "${OUTPUT}" "${PASS2}" -mindepth 1 -print -quit 2>/dev/null || true)"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${REPO}" diff --binary -- recovar/em recovar/cuda/cuda_backproject.cu scripts/run_multi_iter_parity.py | sha256sum | awk '{print $1}')" = "${EXPECTED_SOURCE_DIFF_SHA256}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = "${EXPECTED_CUDA_LIB_SHA256}"
RELION_BIND_LIB=$(find "${RELION_BIND}" -maxdepth 1 -type f -name '_relion_bind_core*.so' -print -quit)
test "$(sha256sum "${RELION_BIND_LIB}" | awk '{print $1}')" = 56acdddab6d25bd83aac5e338162c4819f85ec26f68511458618f143c2b604b7
test -s "${RELION}/run_it006_optimiser.star" -a -s "${RELION}/run_it007_data.star"

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
    "${JAX_COMPILATION_CACHE_DIR}" "${PYTHONPYCACHEPREFIX}" \
    "${OUTPUT}/intermediates" "${PASS2}" "${ROOT}/logs" "${ROOT}/provenance"

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
export RECOVAR_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE=0
export RECOVAR_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN=4e-6
export RECOVAR_K1_BPREF_EXECUTION_ORDER_CHUNK_SIZE=220
export RECOVAR_K1_RELION_POWERCLASS_SPECTRUM_NORM=1
export RECOVAR_K1_RELION_EXACT_BPREF_OPERANDS=1
export RECOVAR_K1_RELION_EXACT_CTF_STAR=${DATA}/particles.star
export RECOVAR_K1_RELION_LIVE_INITIAL_NOISE=0
export RECOVAR_RELION_X_HALF_F32_FINE_POSTERIOR=1
export RECOVAR_K1_COARSE_GAUSSIAN_FFI=1
export RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF=1
export RECOVAR_K1_RELION_EXACT_COARSE_OPERANDS=1
export RECOVAR_K1_RELION_F32_COARSE_SUPPORT=1
export RECOVAR_PASS2_DUMP_DIR=${PASS2}
export RECOVAR_PASS2_DUMP_ORIGINAL_INDICES=35637
export RECOVAR_PASS2_DUMP_CURRENT_SIZE=146
export RECOVAR_PASS2_DUMP_ITERATION=6
export RECOVAR_PASS2_DUMP_CLASS=1
export RECOVAR_PASS2_DUMP_RAW_OPERANDS=1
export RECOVAR_PASS2_DUMP_STOP_AFTER_TARGET=1
unset RECOVAR_FINAL_ALL_DATA_GRID_CORRECT RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER

cd "${REPO}"
git status --short --branch > "${ROOT}/provenance/git_status_${SLURM_JOB_ID}.txt"
git diff --binary -- recovar/em recovar/cuda/cuda_backproject.cu scripts/run_multi_iter_parity.py \
    > "${ROOT}/provenance/source_diff_${SLURM_JOB_ID}.patch"
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
assert '.pixi/envs/default/' in str(Path(jax.__file__).resolve())
assert len(jax.devices('gpu')) == 1 and 'H100' in jax.devices('gpu')[0].device_kind.upper()
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
print(jax.devices())
PY

command=("${PYTHON}" -m scripts.run_multi_iter_parity
    --relion_dir "${RELION}"
    --data_star "${DATA}/particles.star"
    --iter 6 --max_iter 1
    --continuous-relion-noise-state
    --diagnostic-native-relion-particle-order-seed 1710
    --skip_final_iteration
    --image-fourier-backend relion_cuda
    --image_batch_size 187
    --rotation_block_size 8192
    --output_dir "${OUTPUT}"
    --save_intermediates_dir "${OUTPUT}/intermediates")
printf '%q ' "${command[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
start_epoch=$(date +%s)
set +e
"${command[@]}" > "${ROOT}/logs/run_${SLURM_JOB_ID}.log" 2>&1
command_status=$?
set -e
wall_s=$(( $(date +%s) - start_epoch ))

capture=${PASS2}/pass2_orig035637_cs146.npz
test "${command_status}" -ne 0
test -s "${capture}"
test "$(find "${PASS2}" -maxdepth 1 -type f -name 'pass2_orig*_cs146.npz' | wc -l)" -eq 1
rg -q 'Sparse K=1 pass-2 stop-after-dump requested .*target-set progress 1/1' \
    "${ROOT}/logs/run_${SLURM_JOB_ID}.log"
rg -q 'Pass2DumpComplete: requested RECOVAR pass-2 dump target set was written' \
    "${ROOT}/logs/run_${SLURM_JOB_ID}.log"
! rg -q 'RESOURCE_EXHAUSTED|Out of memory|accounting is inconsistent' \
    "${ROOT}/logs/run_${SLURM_JOB_ID}.log"
sha256sum "${capture}" > "${ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
printf '{"job_id":"%s","wall_s":%s,"source_row":35637,"physical_iteration":7}\n' \
    "${SLURM_JOB_ID}" "${wall_s}" > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
