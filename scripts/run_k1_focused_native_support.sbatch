#!/usr/bin/env bash
#SBATCH --job-name=k1-focused-score
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:30:00

set -euo pipefail

: "${K1_FOCUSED_ROOT:?set the disposable result root}"
: "${K1_FOCUSED_RUNTIME:?set the disposable runtime root}"
: "${K1_FOCUSED_REPO:?set the RECOVAR checkout}"
: "${K1_FOCUSED_RECOVAR_RUN:?set the completed RECOVAR source run}"
: "${K1_FOCUSED_NATIVE_FACTOR:?set the native RELION factor capture}"
: "${K1_FOCUSED_NATIVE_FINE_SCORE:?set the native RELION score capture}"
: "${K1_FOCUSED_SOURCE_INDEX:?set the zero-based source index}"
: "${K1_FOCUSED_HALF:?set the one-based half}"
: "${EXPECTED_LAUNCHER_SHA256:?pin this launcher}"
: "${EXPECTED_DRIVER_SHA256:?pin the focused Python driver}"
: "${EXPECTED_REPO_HEAD:?pin the checkout HEAD}"
: "${EXPECTED_REPO_DIFF_SHA256:?pin the tracked checkout diff}"
: "${EXPECTED_CUDA_SHA256:?pin the CUDA library}"

ROOT=${K1_FOCUSED_ROOT}
RUNTIME=${K1_FOCUSED_RUNTIME}
REPO=${K1_FOCUSED_REPO}
PYTHON=${K1_FOCUSED_PYTHON:-${REPO}/.pixi/envs/default/bin/python}
PIXI_ENV=${K1_FOCUSED_PIXI_ENV:-${REPO}/.pixi/envs/default}
DRIVER=${REPO}/scripts/capture_k1_focused_native_support.py
LAUNCHER=${REPO}/scripts/run_k1_focused_native_support.sbatch
DATA=${K1_FOCUSED_DATA_DIR:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_currenthead_25ab6e68_20260728T011020ET/cases/22_small_severe_outliers_3k_g128_radial_noise5_bf80/data}
BOUNDARY_INDEX=${K1_FOCUSED_BOUNDARY_INDEX:-2}
CONSUMER_ITERATION=${K1_FOCUSED_CONSUMER_ITERATION:-3}
CURRENT_SIZE=${K1_FOCUSED_CURRENT_SIZE:-80}
CUDA_LIB=${K1_FOCUSED_CUDA_LIB:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_stack0079_production_raw_it3_67716bcb_20260813T1430ET/cuda/libcuda_backproject.so}
RELION_BIND=${K1_FOCUSED_RELION_BIND:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_bpref_order_fix_20260810/relion_bind}
OUTPUT=${ROOT}/pass2

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${DRIVER}" | awk '{print $1}')" = "${EXPECTED_DRIVER_SHA256}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${REPO}" diff --binary | sha256sum | awk '{print $1}')" = "${EXPECTED_REPO_DIFF_SHA256}"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
test -z "$(find "${OUTPUT}" -mindepth 1 -print -quit 2>/dev/null || true)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/rattler_cache
export PYTHONPYCACHEPREFIX=${RUNTIME}/${SLURM_JOB_ID}/pycache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/jax_cache
mkdir -p "${OUTPUT}" "${ROOT}/provenance" "${TMPDIR}" "${PIXI_HOME}" \
    "${RATTLER_CACHE_DIR}" "${PYTHONPYCACHEPREFIX}" "${JAX_COMPILATION_CACHE_DIR}"

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
: "${CUDA_VISIBLE_DEVICES:?Slurm must assign exactly one visible GPU}"

export RECOVAR_CUDA_LIB=${CUDA_LIB}
export RECOVAR_RELION_BIND_BUILD_DIR=${RELION_BIND}
export RELION_SRC_DIR=/scratch/gpfs/GILLES/mg6942/relion/src
export RECOVAR_EXPECTED_REPO_ROOT=${REPO}
unset RECOVAR_FINAL_ALL_DATA_GRID_CORRECT RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER

COMMAND=(
    "${PYTHON}" -m scripts.capture_k1_focused_native_support
    --data-dir "${DATA}"
    --recovar-run "${K1_FOCUSED_RECOVAR_RUN}"
    --boundary-index "${BOUNDARY_INDEX}"
    --consumer-iteration "${CONSUMER_ITERATION}"
    --source-index "${K1_FOCUSED_SOURCE_INDEX}"
    --half "${K1_FOCUSED_HALF}"
    --current-size "${CURRENT_SIZE}"
    --native-factor "${K1_FOCUSED_NATIVE_FACTOR}"
    --native-fine-score "${K1_FOCUSED_NATIVE_FINE_SCORE}"
    --output-dir "${OUTPUT}"
)
if [[ -n "${K1_FOCUSED_NATIVE_PPREF:-}" ]]; then
    COMMAND+=(--native-ppref "${K1_FOCUSED_NATIVE_PPREF}")
fi
if [[ -n "${K1_FOCUSED_NATIVE_PREPROCESS_CAPTURE:-}" ]]; then
    COMMAND+=(--native-preprocess-capture "${K1_FOCUSED_NATIVE_PREPROCESS_CAPTURE}")
fi
if [[ -n "${K1_FOCUSED_NATIVE_SCORING_NOISE:-}" ]]; then
    COMMAND+=(--native-scoring-noise "${K1_FOCUSED_NATIVE_SCORING_NOISE}")
fi
if [[ -n "${K1_FOCUSED_NATIVE_FINE_IMAGE_CAPTURE:-}" ]]; then
    COMMAND+=(--native-fine-image-capture "${K1_FOCUSED_NATIVE_FINE_IMAGE_CAPTURE}")
fi
if [[ -n "${K1_FOCUSED_NORMALIZATION_FACTOR_OVERRIDE:-}" ]]; then
    COMMAND+=(--normalization-factor-override "${K1_FOCUSED_NORMALIZATION_FACTOR_OVERRIDE}")
fi
if [[ "${K1_FOCUSED_NATIVE_LANE_SOFTMASK_REDUCTION:-0}" == 1 ]]; then
    COMMAND+=(--native-lane-softmask-reduction)
fi
if [[ "${K1_FOCUSED_NATIVE_ATOMIC_SOFTMASK_REDUCTION:-0}" == 1 ]]; then
    COMMAND+=(--native-atomic-softmask-reduction)
fi
printf '%q ' "${COMMAND[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
nvidia-smi -q > "${ROOT}/provenance/nvidia_smi_${SLURM_JOB_ID}.txt"
STATIC_INPUTS=(
    "${LAUNCHER}"
    "${DRIVER}"
    "${CUDA_LIB}"
    "${K1_FOCUSED_NATIVE_FACTOR}"
    "${K1_FOCUSED_NATIVE_FINE_SCORE}"
    "${RELION_BIND}"/_relion_bind_core*.so
)
if [[ -n "${K1_FOCUSED_NATIVE_PPREF:-}" ]]; then
    STATIC_INPUTS+=("${K1_FOCUSED_NATIVE_PPREF}")
fi
if [[ -n "${K1_FOCUSED_NATIVE_PREPROCESS_CAPTURE:-}" ]]; then
    STATIC_INPUTS+=("${K1_FOCUSED_NATIVE_PREPROCESS_CAPTURE}")
fi
if [[ -n "${K1_FOCUSED_NATIVE_SCORING_NOISE:-}" ]]; then
    STATIC_INPUTS+=("${K1_FOCUSED_NATIVE_SCORING_NOISE}")
fi
if [[ -n "${K1_FOCUSED_NATIVE_FINE_IMAGE_CAPTURE:-}" ]]; then
    STATIC_INPUTS+=("${K1_FOCUSED_NATIVE_FINE_IMAGE_CAPTURE}")
fi
sha256sum "${STATIC_INPUTS[@]}" \
    > "${ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"

start=$(date +%s)
cd "${REPO}"
"${COMMAND[@]}" > "${ROOT}/run.log" 2>&1
wall_s=$(( $(date +%s) - start ))

test -s "${OUTPUT}/FOCUSED_CAPTURE.json"
test -s "${OUTPUT}/pass2_orig$(printf '%06d' "${K1_FOCUSED_SOURCE_INDEX}")_cs$(printf '%03d' "${CURRENT_SIZE}").npz"
printf '{"job_id":"%s","wall_s":%d,"source_index":%d,"half":%d,"boundary_iteration":%d,"consumer_iteration":%d,"current_size":%d}\n' \
    "${SLURM_JOB_ID}" "${wall_s}" "${K1_FOCUSED_SOURCE_INDEX}" "${K1_FOCUSED_HALF}" \
    "${BOUNDARY_INDEX}" "${CONSUMER_ITERATION}" "${CURRENT_SIZE}" \
    > "${ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
touch "${ROOT}/FOCUSED_CAPTURE_SUCCESS_${SLURM_JOB_ID}"
