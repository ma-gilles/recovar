#!/usr/bin/env bash
#SBATCH --job-name=k1-c04-fine-cf
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=00:20:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${SOURCE:?SOURCE is required}"
: "${PYTHON:?PYTHON is required}"
: "${CUDA_LIB:?CUDA_LIB is required}"
: "${RELION_BIND:?RELION_BIND is required}"
: "${RECOVAR_CAPTURE:?RECOVAR_CAPTURE is required}"
: "${REFERENCE_MAP:?REFERENCE_MAP is required}"
: "${NATIVE_FACTOR:?NATIVE_FACTOR is required}"
: "${NATIVE_FINE_SCORE:?NATIVE_FINE_SCORE is required}"
: "${EXPECTED_HEAD:?EXPECTED_HEAD is required}"
: "${EXPECTED_CUDA_SHA:?EXPECTED_CUDA_SHA is required}"
: "${EXPECTED_LAUNCHER_SHA:?EXPECTED_LAUNCHER_SHA is required}"
: "${EXPECTED_ANALYZER_SHA:?EXPECTED_ANALYZER_SHA is required}"
: "${EXPECTED_RECOVAR_CAPTURE_SHA:?EXPECTED_RECOVAR_CAPTURE_SHA is required}"
: "${EXPECTED_REFERENCE_SHA:?EXPECTED_REFERENCE_SHA is required}"
: "${EXPECTED_NATIVE_FACTOR_SHA:?EXPECTED_NATIVE_FACTOR_SHA is required}"
: "${EXPECTED_NATIVE_FINE_SCORE_SHA:?EXPECTED_NATIVE_FINE_SCORE_SHA is required}"

LAUNCHER=${SOURCE}/scripts/run_k1_case04_top1_fine_counterfactual.sbatch
ANALYZER=${SOURCE}/scripts/analyze_k1_native_texture_fine_counterfactual.py
OUTPUT=${RUN_ROOT}/analysis/K1_CASE04_TOP1_FINE_COUNTERFACTUAL.json

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test "$(git -C "${SOURCE}" rev-parse HEAD)" = "${EXPECTED_HEAD}"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA}"
test "$(sha256sum "${ANALYZER}" | awk '{print $1}')" = "${EXPECTED_ANALYZER_SHA}"
test "$(sha256sum "${RECOVAR_CAPTURE}" | awk '{print $1}')" = "${EXPECTED_RECOVAR_CAPTURE_SHA}"
test "$(sha256sum "${REFERENCE_MAP}" | awk '{print $1}')" = "${EXPECTED_REFERENCE_SHA}"
test "$(sha256sum "${NATIVE_FACTOR}" | awk '{print $1}')" = "${EXPECTED_NATIVE_FACTOR_SHA}"
test "$(sha256sum "${NATIVE_FINE_SCORE}" | awk '{print $1}')" = "${EXPECTED_NATIVE_FINE_SCORE_SHA}"
test ! -e "${OUTPUT}"

mkdir -p "${RUN_ROOT}/analysis" "${RUN_ROOT}/provenance" \
  "${RUNTIME_ROOT}/tmp" "${RUNTIME_ROOT}/pixi_home" \
  "${RUNTIME_ROOT}/rattler_cache" "${RUNTIME_ROOT}/jax_cache" \
  "${RUNTIME_ROOT}/pycache"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME_ROOT}/tmp PIXI_HOME=${RUNTIME_ROOT}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/rattler_cache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME_ROOT}/jax_cache
export PYTHONPYCACHEPREFIX=${RUNTIME_ROOT}/pycache
export RECOVAR_CUDA_LIB=${CUDA_LIB}
export RECOVAR_RELION_BIND_BUILD_DIR=${RELION_BIND}

source /etc/profile.d/modules.sh
set +u
module purge
module load cudatoolkit/12.8
set -u

PIXI_ENV=${SOURCE}/.pixi/envs/default
CUDA_ROOT=/usr/local/cuda-12.8
NVIDIA_ROOT=$(find "${PIXI_ENV}/lib" -maxdepth 3 -type d -path '*/site-packages/nvidia' -print -quit)
NVIDIA_LIBS=$(find "${NVIDIA_ROOT}" -type d -name lib | paste -sd: -)
export PATH=${CUDA_ROOT}/bin:${PATH}
export LD_LIBRARY_PATH=${NVIDIA_LIBS}:${CUDA_ROOT}/targets/x86_64-linux/lib:${PIXI_ENV}/lib

nvidia-smi --query-gpu=uuid,name,memory.total --format=csv,noheader \
  > "${RUN_ROOT}/provenance/gpu_${SLURM_JOB_ID}.txt"
"${PYTHON}" -c "import pathlib,recovar,jax; from recovar import cuda_backproject; root=pathlib.Path('${SOURCE}').resolve(); assert pathlib.Path(recovar.__file__).resolve().is_relative_to(root); assert '.pixi/envs/default/' in str(pathlib.Path(jax.__file__).resolve()); assert jax.default_backend() == 'gpu'; assert cuda_backproject.cuda_available()"

git -C "${SOURCE}" status --short --branch > "${RUN_ROOT}/provenance/git_status_${SLURM_JOB_ID}.txt"
git -C "${SOURCE}" diff --binary > "${RUN_ROOT}/provenance/source_diff_${SLURM_JOB_ID}.patch"
sha256sum "${CUDA_LIB}" "${LAUNCHER}" "${ANALYZER}" \
  "${RECOVAR_CAPTURE}" "${REFERENCE_MAP}" "${NATIVE_FACTOR}" \
  "${NATIVE_FINE_SCORE}" > "${RUN_ROOT}/provenance/inputs_${SLURM_JOB_ID}.sha256"

cd "${SOURCE}"
"${PYTHON}" -m scripts.analyze_k1_native_texture_fine_counterfactual \
  --recovar-capture "${RECOVAR_CAPTURE}" \
  --reference-map "${REFERENCE_MAP}" \
  --native-factor "${NATIVE_FACTOR}" \
  --native-fine-score "${NATIVE_FINE_SCORE}" \
  --physical-image-size 256 --padding-factor 2 --projector-current-size 100 \
  --output-json "${OUTPUT}" > "${RUN_ROOT}/analysis/stdout_${SLURM_JOB_ID}.log"

sha256sum "${OUTPUT}" > "${RUN_ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
