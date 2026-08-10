#!/usr/bin/env bash
#SBATCH --job-name=k1c22-it2-coarse
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_coarse_recovar_20260810T1255ET/logs/run-%j.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_coarse_recovar_20260810T1255ET/logs/run-%j.err
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=00:12:00

set -euo pipefail
: "${EXPECTED_GIT_DIFF_SHA256:?submission must pin the dirty-tree diff}"
: "${EXPECTED_CUDA_SHA256:?submission must pin the CUDA library}"

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_coarse_recovar_20260810T1255ET
RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case22_it2_coarse_recovar_20260810T1255ET
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default
DATA=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_currenthead_25ab6e68_20260728T011020ET/cases/22_small_severe_outliers_3k_g128_radial_noise5_bf80/data
RELION=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_bpref_factor_boundary_7c3b5478_20260810T0610ET/small_control_retry2/native_capture/relion
CUDA_LIB=${REPO}/recovar/cuda/libcuda_backproject.so

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(git -C "${REPO}" rev-parse HEAD)" = fa60832431c638e969a0e08ef9de5b83191eca51
test "$(git -C "${REPO}" diff --binary | sha256sum | awk '{print $1}')" = "${EXPECTED_GIT_DIFF_SHA256}"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
test ! -e "${ROOT}/output" -a ! -e "${ROOT}/coarse"

mkdir -p "${ROOT}/logs" "${ROOT}/coarse" \
    "${RUNTIME}/${SLURM_JOB_ID}/tmp" "${RUNTIME}/${SLURM_JOB_ID}/pixi_home" \
    "${RUNTIME}/${SLURM_JOB_ID}/rattler_cache" "${RUNTIME}/${SLURM_JOB_ID}/jax_cache"
touch "${ROOT}/coarse/SAFE_TO_DELETE"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_BPRE_*|RELION_FINE_*|RELION_DISPATCH_LOG) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/rattler_cache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/jax_cache

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
export RECOVAR_RELION_BIND_BUILD_DIR=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fixedsuite_case07_tree_c74beea4_20260724T110500Z/relion_bind_build/shared
export RELION_SRC_DIR=/scratch/gpfs/GILLES/mg6942/relion/src
export RECOVAR_EXPECTED_REPO_ROOT=${REPO}
export RECOVAR_RELION_EM_BATCH_PROJECTION_FRACTION=0.40
export RECOVAR_FINAL_ALL_DATA_REPLAY_LAST_NUMBERED_STATE=0
export RECOVAR_INITIAL_PROJECTOR_USE_REAL_REFERENCE=1
export RECOVAR_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN=4e-6
export RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT=0
export RECOVAR_RELION_X_HALF_F32_FINE_POSTERIOR=1
export RECOVAR_K1_RELION_EXACT_BPREF_OPERANDS=1
export RECOVAR_K1_RELION_EXACT_CTF_STAR=${DATA}/particles.star
export RECOVAR_SIGNIFICANCE_DUMP_DIR=${ROOT}/coarse
export RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES=251,590,2123
export RECOVAR_SIGNIFICANCE_DUMP_CURRENT_SIZE=60
unset RECOVAR_FINAL_ALL_DATA_GRID_CORRECT RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER

cd "${REPO}"
"${PYTHON}" -c "import pathlib,recovar,jax; from recovar import cuda_backproject; repo=pathlib.Path.cwd().resolve(); assert pathlib.Path(recovar.__file__).resolve().is_relative_to(repo); assert '.pixi/envs/default/' in str(pathlib.Path(jax.__file__).resolve()); assert len(jax.devices('gpu')) == 1 and 'H100' in jax.devices('gpu')[0].device_kind.upper(); assert cuda_backproject.cuda_available(); print(jax.devices())"

start=$(date +%s)
"${PYTHON}" -m scripts.run_multi_iter_parity \
    --relion_dir "${RELION}" \
    --data_star "${DATA}/particles.star" \
    --iter 1 --max_iter 1 \
    --continuous-relion-noise-state \
    --skip_final_iteration \
    --image_batch_size 187 \
    --rotation_block_size 8192 \
    --image-fourier-backend relion_cuda \
    --keep_stack_indices 251,590,2123 \
    --output_dir "${ROOT}/output" \
    --save_intermediates_dir "${ROOT}/output/intermediates" \
    > "${ROOT}/logs/run_multi_iter.log" 2>&1
wall_s=$(( $(date +%s) - start ))

test "$(find "${ROOT}/coarse" -maxdepth 1 -name 'significance_*.npz' | wc -l)" -eq 3
! rg -q 'Traceback|RESOURCE_EXHAUSTED|Out of memory|accounting is inconsistent' "${ROOT}/logs/run_multi_iter.log"

{
    echo "job_id=${SLURM_JOB_ID}"
    echo "wall_s=${wall_s}"
    echo "repo_head=$(git -C "${REPO}" rev-parse HEAD)"
    echo "git_diff_sha256=${EXPECTED_GIT_DIFF_SHA256}"
    echo "cuda_sha256=${EXPECTED_CUDA_SHA256}"
    echo "RECOVAR_FINAL_ALL_DATA_GRID_CORRECT=unset"
    echo "RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER=unset"
    nvidia-smi --query-gpu=uuid,name --format=csv,noheader
    find "${ROOT}/coarse" -maxdepth 1 -name 'significance_*.npz' -print0 \
        | sort -z | xargs -0 sha256sum
} > "${ROOT}/provenance_${SLURM_JOB_ID}.txt"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
