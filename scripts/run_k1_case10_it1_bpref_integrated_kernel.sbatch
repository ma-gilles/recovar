#!/usr/bin/env bash
#SBATCH --job-name=k1c10-bpref-integrated
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${CUDA_LIB:?CUDA_LIB is required}"
: "${FACTOR_DIRECTORY:?FACTOR_DIRECTORY is required}"
: "${NATIVE_DIRECTORY:?NATIVE_DIRECTORY is required}"

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
PYTHON=${REPO}/.pixi/envs/default/bin/python
ANALYZER=${REPO}/scripts/analyze_k1_bpref_integrated_kernel.py
OUTPUT=${RUN_ROOT}/analysis/K1_CASE10_IT1_BPREF_INTEGRATED_KERNEL.json

test -f "${RUN_ROOT}/SAFE_TO_DELETE"
test -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -x "${PYTHON}"
test -f "${CUDA_LIB}"
test -d "${FACTOR_DIRECTORY}"
test -d "${NATIVE_DIRECTORY}"
test ! -e "${OUTPUT}"
mkdir -p "${RUN_ROOT}/analysis" "${RUN_ROOT}/logs" "${RUN_ROOT}/provenance"
mkdir -p "${RUNTIME_ROOT}/tmp" "${RUNTIME_ROOT}/pixi_home" "${RUNTIME_ROOT}/rattler_cache"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME_ROOT}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/rattler_cache
export RECOVAR_CUDA_LIB=${CUDA_LIB}

cd "${REPO}"
nvidia-smi > "${RUN_ROOT}/provenance/nvidia_smi_${SLURM_JOB_ID}.txt"
git status --short --branch > "${RUN_ROOT}/provenance/repo_status_${SLURM_JOB_ID}.txt"
git diff --binary > "${RUN_ROOT}/provenance/git_diff_${SLURM_JOB_ID}.patch"
sha256sum "${CUDA_LIB}" "${ANALYZER}" \
  > "${RUN_ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"
"${PYTHON}" - <<'PY' > "${RUN_ROOT}/provenance/imports_${SLURM_JOB_ID}.txt"
import jax
import recovar
print("recovar", recovar.__file__)
print("jax", jax.__file__)
print("devices", jax.devices())
PY

"${PYTHON}" -m scripts.analyze_k1_bpref_integrated_kernel \
  --factor-directory "${FACTOR_DIRECTORY}" \
  --native-directory "${NATIVE_DIRECTORY}" \
  --repeats 2 \
  --output-json "${OUTPUT}" \
  > "${RUN_ROOT}/analysis/K1_CASE10_IT1_BPREF_INTEGRATED_KERNEL.stdout"

sha256sum "${OUTPUT}" > "${RUN_ROOT}/analysis/K1_CASE10_IT1_BPREF_INTEGRATED_KERNEL.sha256"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
