#!/usr/bin/env bash
#SBATCH --job-name=k1-native-fine-replay
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:15:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${NATIVE_DIR:?NATIVE_DIR is required}"
: "${RECOVAR_CAPTURE:?RECOVAR_CAPTURE is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_ANALYZER_SHA256:?EXPECTED_ANALYZER_SHA256 is required}"
: "${EXPECTED_REPLAY_SHA256:?EXPECTED_REPLAY_SHA256 is required}"

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
PYTHON=${REPO}/.pixi/envs/default/bin/python
LAUNCHER=${REPO}/scripts/replay_k1_native_fine_operands_gpu.sbatch
ANALYZER=${REPO}/scripts/analyze_k1_native_fine_operand_boundary.py
REPLAY=${REPO}/scripts/replay_k1_native_fine_operands_gpu.py
CUDA_LIB=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_preprocess_stage_split_20260821T1600ET/cuda/libcuda_backproject.so
REPORT=${RUN_ROOT}/analysis/NATIVE_FINE_GPU_REPLAY.json

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test "$(git -C "${REPO}" rev-parse HEAD)" = 223e28fd5339eb037250e78384e52652d406fcf1
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${ANALYZER}" | awk '{print $1}')" = "${EXPECTED_ANALYZER_SHA256}"
test "$(sha256sum "${REPLAY}" | awk '{print $1}')" = "${EXPECTED_REPLAY_SHA256}"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = b5178b902aafaa945bf7271421edab28bd75f0291484006833e0da0ebc91494b
test -s "${NATIVE_DIR}/pass1_exp_Mweight_raw_preprior.bin"
test -s "${RECOVAR_CAPTURE}"
test ! -e "${REPORT}"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME_ROOT}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/rattler_cache
export JAX_COMPILATION_CACHE_DIR=${RUNTIME_ROOT}/jax_cache
export PYTHONPYCACHEPREFIX=${RUNTIME_ROOT}/pycache
export RECOVAR_CUDA_LIB=${CUDA_LIB}
export CUDA_VISIBLE_DEVICES=0

source /etc/profile.d/modules.sh
set +u
module purge
module load cudatoolkit/12.8
set -u
CUDA_HOME=/usr/local/cuda-12.8
PIXI_ENV=${REPO}/.pixi/envs/default
export PATH=${CUDA_HOME}/bin:${PATH}
NVIDIA_ROOT=$(find "${PIXI_ENV}/lib" -maxdepth 3 -type d -path '*/site-packages/nvidia' -print -quit)
NVIDIA_LIBS=$(find "${NVIDIA_ROOT}" -type d -name lib | paste -sd: -)
export LD_LIBRARY_PATH=${NVIDIA_LIBS}:${CUDA_HOME}/targets/x86_64-linux/lib:${PIXI_ENV}/lib

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | xargs)
case "${GPU_NAME}" in *H100*) ;; *) echo "Expected H100, got ${GPU_NAME}" >&2; exit 2 ;; esac
mkdir -p "${RUN_ROOT}/analysis" "${RUN_ROOT}/logs" "${RUN_ROOT}/provenance" \
  "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" \
  "${JAX_COMPILATION_CACHE_DIR}" "${PYTHONPYCACHEPREFIX}"
nvidia-smi -q > "${RUN_ROOT}/provenance/nvidia_smi_${SLURM_JOB_ID}.txt"
env | sort > "${RUN_ROOT}/provenance/environment_${SLURM_JOB_ID}.txt"
sha256sum "${LAUNCHER}" "${ANALYZER}" "${REPLAY}" "${CUDA_LIB}" \
  > "${RUN_ROOT}/provenance/static_inputs_${SLURM_JOB_ID}.sha256"

cd "${REPO}"
"${PYTHON}" - <<'PY'
from pathlib import Path

import jax
import recovar
from recovar import cuda_backproject

repo = Path.cwd().resolve()
assert Path(recovar.__file__).resolve().is_relative_to(repo)
assert ".pixi/envs/default/" in str(Path(jax.__file__).resolve())
assert len(jax.devices("gpu")) == 1 and "H100" in jax.devices("gpu")[0].device_kind.upper()
assert cuda_backproject.cuda_available(), cuda_backproject.cuda_unavailable_error()
print(jax.devices())
PY
"${PYTHON}" -m scripts.replay_k1_native_fine_operands_gpu \
  --native-dir "${NATIVE_DIR}" \
  --recovar-capture "${RECOVAR_CAPTURE}" --physical-image-size 384 \
  --output-json "${REPORT}"
sha256sum "${REPORT}" > "${RUN_ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
