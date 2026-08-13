#!/usr/bin/env bash
#SBATCH --job-name=k1c22-s1204-rec
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=00:20:00

set -euo pipefail
: "${K1_CASE22_STACK1204_ROOT:?set the disposable experiment root}"
: "${K1_CASE22_STACK1204_RUNTIME:?set the disposable runtime root}"
: "${EXPECTED_LAUNCHER_SHA256:?pin the launcher hash}"
: "${EXPECTED_REPO_HEAD:?pin the repository head}"
: "${EXPECTED_REPO_DIFF_SHA256:?pin the tracked diff}"
: "${EXPECTED_CUDA_SHA256:?pin the CUDA library}"

ROOT=${K1_CASE22_STACK1204_ROOT}
RUNTIME=${K1_CASE22_STACK1204_RUNTIME}
REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default
LAUNCHER=${REPO}/scripts/run_k1_case22_stack1204_recovar_capture.sbatch
CUDA_LIB=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_bpref_factor_boundary_7c3b5478_20260810T0610ET/small_control_retry2/cuda/libcuda_backproject.so
DATA=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_currenthead_25ab6e68_20260728T011020ET/cases/22_small_severe_outliers_3k_g128_radial_noise5_bf80/data
RELION=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_bpref_factor_boundary_7c3b5478_20260810T0610ET/small_control_retry2/native_capture/relion
OUTPUT=${ROOT}/recovar/output
PASS2=${ROOT}/recovar/pass2
TARGET_STACK_ONE_BASED=${K1_CASE22_TARGET_STACK_ONE_BASED:-1204}
TARGET_ORIGINAL_INDEX=$(( TARGET_STACK_ONE_BASED - 1 ))
INITIAL_HALF1_FT_NPZ=${K1_CASE22_INITIAL_HALF1_FT_NPZ:-}
INITIAL_HALF2_FT_NPZ=${K1_CASE22_INITIAL_HALF2_FT_NPZ:-}
INITIAL_NOISE_HALF1_NPY=${K1_CASE22_INITIAL_NOISE_HALF1_NPY:-}
INITIAL_NOISE_HALF2_NPY=${K1_CASE22_INITIAL_NOISE_HALF2_NPY:-}
NORMALIZATION_FACTOR_OVERRIDE=${K1_CASE22_NORMALIZATION_FACTOR_OVERRIDE:-}
if (( TARGET_STACK_ONE_BASED < 1 )); then
    echo "K1_CASE22_TARGET_STACK_ONE_BASED must be positive" >&2
    exit 2
fi
if [[ -n "${INITIAL_HALF1_FT_NPZ}" || -n "${INITIAL_HALF2_FT_NPZ}" ]]; then
    test -f "${INITIAL_HALF1_FT_NPZ}" -a -f "${INITIAL_HALF2_FT_NPZ}"
fi
if [[ -n "${INITIAL_NOISE_HALF1_NPY}" || -n "${INITIAL_NOISE_HALF2_NPY}" ]]; then
    test -f "${INITIAL_NOISE_HALF1_NPY}" -a -f "${INITIAL_NOISE_HALF2_NPY}"
fi

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${REPO}" diff --binary | sha256sum | awk '{print $1}')" = "${EXPECTED_REPO_DIFF_SHA256}"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
test -z "$(find "${OUTPUT}" -mindepth 1 -print -quit)"
test -z "$(find "${PASS2}" -mindepth 1 -print -quit)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
while IFS='=' read -r variable_name _; do
    case "${variable_name}" in RECOVAR_*|RELION_BPRE_*|RELION_FINE_*|RELION_DISPATCH_LOG) unset "${variable_name}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/rattler_cache
export PYTHONPYCACHEPREFIX=${RUNTIME}/${SLURM_JOB_ID}/pycache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/jax_cache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" "${PYTHONPYCACHEPREFIX}" "${JAX_COMPILATION_CACHE_DIR}" "${ROOT}/recovar/provenance"

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
export RECOVAR_K1_RELION_LIVE_INITIAL_NOISE=1
export RECOVAR_RELION_X_HALF_F32_FINE_POSTERIOR=1
export RECOVAR_K1_RELION_EXACT_BPREF_OPERANDS=1
export RECOVAR_K1_RELION_EXACT_CTF_STAR=${DATA}/particles.star
export RECOVAR_K1_COARSE_GAUSSIAN_FFI=${K1_CASE22_COARSE_GAUSSIAN_FFI:-1}
export RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF=${K1_CASE22_COARSE_GAUSSIAN_SINCOSF:-1}
export RECOVAR_K1_RELION_F32_COARSE_SUPPORT=${K1_CASE22_RELION_F32_COARSE_SUPPORT:-0}
export RECOVAR_PASS2_DUMP_DIR=${PASS2}
export RECOVAR_PASS2_DUMP_ORIGINAL_INDICES=${TARGET_ORIGINAL_INDEX}
export RECOVAR_PASS2_DUMP_CURRENT_SIZE=60
export RECOVAR_PASS2_DUMP_CLASS=1
export RECOVAR_PASS2_DUMP_RAW_OPERANDS=1
unset RECOVAR_FINAL_ALL_DATA_GRID_CORRECT RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER

nvidia-smi -q > "${ROOT}/recovar/provenance/nvidia_smi_${SLURM_JOB_ID}.txt"
"${PYTHON}" -c "import pathlib,recovar,jax; from recovar import cuda_backproject; root=pathlib.Path('${REPO}').resolve(); assert pathlib.Path(recovar.__file__).resolve().is_relative_to(root); assert '.pixi/envs/default/' in str(pathlib.Path(jax.__file__).resolve()); assert len(jax.devices('gpu')) == 1 and 'H100' in jax.devices('gpu')[0].device_kind.upper(); assert cuda_backproject.cuda_available(); print(jax.devices())"

command=(
    "${PYTHON}" -m scripts.run_multi_iter_parity
    --relion_dir "${RELION}" --data_star "${DATA}/particles.star"
    --iter 1 --max_iter 1 --continuous-relion-noise-state --skip_final_iteration
    --image_batch_size 187 --rotation_block_size 8192
    --image-fourier-backend relion_cuda --keep_stack_indices "${TARGET_ORIGINAL_INDEX}"
    --output_dir "${OUTPUT}" --save_intermediates_dir "${OUTPUT}/intermediates"
)
if [[ -n "${INITIAL_HALF1_FT_NPZ}" ]]; then
    command+=(
        --initial-half1-ft-npz "${INITIAL_HALF1_FT_NPZ}"
        --initial-half2-ft-npz "${INITIAL_HALF2_FT_NPZ}"
    )
fi
if [[ -n "${INITIAL_NOISE_HALF1_NPY}" ]]; then
    command+=(
        --initial-noise-half1-npy "${INITIAL_NOISE_HALF1_NPY}"
        --initial-noise-half2-npy "${INITIAL_NOISE_HALF2_NPY}"
    )
fi
if [[ -n "${NORMALIZATION_FACTOR_OVERRIDE}" ]]; then
    command+=(
        --normalization-factor-override
        "${TARGET_ORIGINAL_INDEX}:${NORMALIZATION_FACTOR_OVERRIDE}"
    )
fi
printf '%q ' "${command[@]}" > "${ROOT}/recovar/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/recovar/provenance/command_${SLURM_JOB_ID}.sh"
cd "${REPO}"
start=$(date +%s)
"${command[@]}" > "${OUTPUT}/run.log" 2>&1
printf '{"job_id":"%s","wall_s":%d}\n' "${SLURM_JOB_ID}" "$(( $(date +%s) - start ))" > "${ROOT}/recovar/provenance/walltime_${SLURM_JOB_ID}.json"

printf '{"job_id":"%s","target_stack_one_based":%d,"target_original_index":%d}\n' \
    "${SLURM_JOB_ID}" "${TARGET_STACK_ONE_BASED}" "${TARGET_ORIGINAL_INDEX}" \
    > "${ROOT}/recovar/provenance/target_${SLURM_JOB_ID}.json"
printf -v expected_pass2 'pass2_orig%06d_cs060.npz' "${TARGET_ORIGINAL_INDEX}"
test "$(find "${PASS2}" -maxdepth 1 -name "${expected_pass2}" | wc -l)" -eq 1
! rg -q 'Traceback|RESOURCE_EXHAUSTED|Out of memory|accounting is inconsistent' "${OUTPUT}/run.log"
find "${PASS2}" "${OUTPUT}" -type f -print0 | sort -z | xargs -0 sha256sum \
    > "${ROOT}/recovar/provenance/science_outputs_${SLURM_JOB_ID}.sha256"
touch "${ROOT}/recovar/SUCCESS_${SLURM_JOB_ID}"
