#!/usr/bin/env bash
#SBATCH --job-name=k1-fine-sig
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:05:00

set -euo pipefail
: "${K1_FINE_SIGNIFICANCE_ROOT:?set a disposable run root}"
: "${K1_FINE_SIGNIFICANCE_RUNTIME:?set a disposable runtime root}"
: "${K1_FINE_SCORE_PATH:?set the native fine-score capture}"
: "${K1_FINE_FACTOR_PATH:?set the matching native BPref factor capture}"
: "${K1_FINE_CUDA_LIB:?set the pinned CUDA extension}"
: "${EXPECTED_LAUNCHER_SHA256:?pin the launcher}"
: "${EXPECTED_ANALYZER_SHA256:?pin the analyzer}"
: "${EXPECTED_REPO_HEAD:?pin the repository commit}"
: "${EXPECTED_REPO_DIFF_SHA256:?pin the tracked diff}"
: "${EXPECTED_FINE_SCORE_SHA256:?pin the fine-score capture}"
: "${EXPECTED_FACTOR_SHA256:?pin the BPref factor capture}"
: "${EXPECTED_CUDA_LIB_SHA256:?pin the CUDA extension}"

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_docs_20260810
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default
LAUNCHER=${REPO}/scripts/run_k1_native_fine_significance_replay.sbatch
ANALYZER=${REPO}/scripts/analyze_k1_native_fine_significance.py
ROOT=${K1_FINE_SIGNIFICANCE_ROOT}
RUNTIME=${K1_FINE_SIGNIFICANCE_RUNTIME}

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${ANALYZER}" | awk '{print $1}')" = "${EXPECTED_ANALYZER_SHA256}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(git -C "${REPO}" diff --binary | sha256sum | awk '{print $1}')" = "${EXPECTED_REPO_DIFF_SHA256}"
test "$(sha256sum "${K1_FINE_SCORE_PATH}" | awk '{print $1}')" = "${EXPECTED_FINE_SCORE_SHA256}"
test "$(sha256sum "${K1_FINE_FACTOR_PATH}" | awk '{print $1}')" = "${EXPECTED_FACTOR_SHA256}"
test "$(sha256sum "${K1_FINE_CUDA_LIB}" | awk '{print $1}')" = "${EXPECTED_CUDA_LIB_SHA256}"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/rattler_cache
export PYTHONPYCACHEPREFIX=${RUNTIME}/${SLURM_JOB_ID}/pycache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/jax_cache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" \
    "${PYTHONPYCACHEPREFIX}" "${JAX_COMPILATION_CACHE_DIR}" \
    "${ROOT}/analysis" "${ROOT}/provenance"

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
export RECOVAR_CUDA_LIB=${K1_FINE_CUDA_LIB}

nvidia-smi --query-gpu=name,uuid,compute_cap --format=csv,noheader \
    > "${ROOT}/provenance/gpu-${SLURM_JOB_ID}.txt"
case "$(cut -d, -f1 "${ROOT}/provenance/gpu-${SLURM_JOB_ID}.txt")" in
    *H100*) ;;
    *) echo "Expected H100" >&2; exit 2 ;;
esac

"${PYTHON}" "${ANALYZER}" \
    --fine-score "${K1_FINE_SCORE_PATH}" \
    --factor "${K1_FINE_FACTOR_PATH}" \
    --output "${ROOT}/analysis/native-fine-significance.json"
sha256sum "${ROOT}/analysis/native-fine-significance.json" \
    > "${ROOT}/provenance/output-${SLURM_JOB_ID}.sha256"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
