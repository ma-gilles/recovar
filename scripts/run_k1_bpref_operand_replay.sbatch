#!/usr/bin/env bash
#SBATCH --job-name=k1-bpref-replay
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:10:00

set -euo pipefail
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${EXPECTED_HEAD:?EXPECTED_HEAD is required}"
: "${EXPECTED_SOURCE_DIFF:?EXPECTED_SOURCE_DIFF is required}"
: "${EXPECTED_SCRIPT_SHA:?EXPECTED_SCRIPT_SHA is required}"
: "${EXPECTED_LAUNCHER_SHA:?EXPECTED_LAUNCHER_SHA is required}"
: "${EXPECTED_CUDA_SHA:?EXPECTED_CUDA_SHA is required}"

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
PYTHON=${REPO}/.pixi/envs/default/bin/python
SCRIPT=${REPO}/scripts/replay_k1_bpref_operands.py
LAUNCHER=${REPO}/scripts/run_k1_bpref_operand_replay.sbatch
CUDA_LIB=${REPO}/recovar/cuda/libcuda_backproject.so
NATIVE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_k1_bpref_operand_v2_d476e6_20260825T0110ET/native_prefix
REC1=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case10_it1_bpref_operand_v2_h1_567b22d4ea_20260825T0100ET/recovar/accumulator_delta/bpref_accumulator_delta_it001_h1_orig081970.npz
REC2=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case10_it1_bpref_operand_v2_h2_567b22d4ea_20260825T0100ET/recovar/accumulator_delta/bpref_accumulator_delta_it001_h2_orig036742.npz

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_HEAD}"
test "$(git -C "${REPO}" diff --binary -- recovar/em recovar/cuda/cuda_backproject.cu recovar/cuda_backproject.py scripts/run_full_refinement.py | sha256sum | awk '{print $1}')" = "${EXPECTED_SOURCE_DIFF}"
test "$(sha256sum "${SCRIPT}" | awk '{print $1}')" = "${EXPECTED_SCRIPT_SHA}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA}"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA}"
test -s "${REC1}" -a -s "${REC2}"

mkdir -p "${RUN_ROOT}/analysis" "${RUN_ROOT}/provenance" \
    "${RUNTIME_ROOT}/tmp" "${RUNTIME_ROOT}/pixi_home" \
    "${RUNTIME_ROOT}/rattler_cache" "${RUNTIME_ROOT}/pycache" \
    "${RUNTIME_ROOT}/jax_cache" "${RUNTIME_ROOT}/cuda_cache"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0
export TMPDIR=${RUNTIME_ROOT}/tmp PIXI_HOME=${RUNTIME_ROOT}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/rattler_cache
export PYTHONPYCACHEPREFIX=${RUNTIME_ROOT}/pycache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME_ROOT}/jax_cache
export RECOVAR_CUDA_CACHE_DIR=${RUNTIME_ROOT}/cuda_cache
export RECOVAR_CUDA_LIB=${CUDA_LIB}
source /etc/profile.d/modules.sh
set +u
module purge
module load cudatoolkit/12.8
set -u
CUDA_HOME=/usr/local/cuda-12.8
export PATH=${CUDA_HOME}/bin:${PATH}
NVIDIA_ROOT=$(find "${REPO}/.pixi/envs/default/lib" -maxdepth 3 -type d -path '*/site-packages/nvidia' -print -quit)
NVIDIA_LIBS=$(find "${NVIDIA_ROOT}" -type d -name lib | paste -sd: -)
export LD_LIBRARY_PATH=${NVIDIA_LIBS}:${CUDA_HOME}/targets/x86_64-linux/lib:${REPO}/.pixi/envs/default/lib

nvidia-smi --query-gpu=uuid,name,memory.total --format=csv,noheader > "${RUN_ROOT}/provenance/gpu.txt"
sha256sum "${SCRIPT}" "${LAUNCHER}" "${CUDA_LIB}" "${REC1}" "${REC2}" \
    > "${RUN_ROOT}/provenance/inputs.sha256"
cd "${REPO}"
"${PYTHON}" -m scripts.replay_k1_bpref_operands --native-directory "${NATIVE}" \
    --recovar-half1 "${REC1}" --recovar-half2 "${REC2}" \
    --output "${RUN_ROOT}/analysis/K1_BPREF_NATIVE_OPERAND_REPLAY.json"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
