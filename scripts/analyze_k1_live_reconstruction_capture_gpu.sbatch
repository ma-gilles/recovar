#!/usr/bin/env bash
#SBATCH --job-name=k1-live-recon-gpu
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=00:15:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${RECOVAR_RUN_ROOT:?RECOVAR_RUN_ROOT is required}"
: "${NATIVE_RUN_ROOT:?NATIVE_RUN_ROOT is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_ANALYZER_SHA256:?EXPECTED_ANALYZER_SHA256 is required}"

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_native_units_instrumentation_20260826
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default
LAUNCHER=${REPO}/scripts/analyze_k1_live_reconstruction_capture_gpu.sbatch
ANALYZER=${REPO}/scripts/analyze_k1_live_reconstruction_capture.py
NATIVE_STAGES=${NATIVE_RUN_ROOT}/reconstruct
NATIVE_RELION=${NATIVE_RUN_ROOT}/relion
ANALYSIS=${RUN_ROOT}/analysis

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -f "${RECOVAR_RUN_ROOT}/SAFE_TO_DELETE" -a -f "${NATIVE_RUN_ROOT}/SAFE_TO_DELETE"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${ANALYZER}" | awk '{print $1}')" = "${EXPECTED_ANALYZER_SHA256}"
test -s "${NATIVE_RUN_ROOT}/analysis/RECONSTRUCT_CAPTURE_VALIDATION.json"
for half in 1 2; do
    test -s "${RECOVAR_RUN_ROOT}/recovar/reconstruction_inputs/recovar_reconstruction_input_it001_half${half}.npz"
    test -s "${RECOVAR_RUN_ROOT}/recovar/premask/recovar_premask_it001_half${half}.npz"
done

mkdir -p "${ANALYSIS}" "${RUN_ROOT}/provenance" \
    "${RUNTIME_ROOT}/tmp" "${RUNTIME_ROOT}/pixi_home" \
    "${RUNTIME_ROOT}/rattler_cache" "${RUNTIME_ROOT}/pycache"
test -z "$(find "${ANALYSIS}" -mindepth 1 -print -quit)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
unset JAX_PLATFORMS JAX_PLATFORM_NAME RECOVAR_DISABLE_CUDA
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME_ROOT}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/rattler_cache
export PYTHONPYCACHEPREFIX=${RUNTIME_ROOT}/pycache

source /etc/profile.d/modules.sh
set +u
module purge
module load cudatoolkit/12.8
set -u
CUDA_HOME=/usr/local/cuda-12.8
NVIDIA_ROOT=$(find "${PIXI_ENV}/lib" -maxdepth 3 -type d -path '*/site-packages/nvidia' -print -quit)
NVIDIA_LIBS=$(find "${NVIDIA_ROOT}" -type d -name lib | paste -sd: -)
export LD_LIBRARY_PATH=${NVIDIA_LIBS}:${CUDA_HOME}/targets/x86_64-linux/lib:${PIXI_ENV}/lib:${LD_LIBRARY_PATH:-}

cd "${REPO}"
nvidia-smi -q > "${RUN_ROOT}/provenance/nvidia_smi_${SLURM_JOB_ID}.txt"
"${PYTHON}" - <<'PY'
from pathlib import Path
import jax
import recovar

repo = Path.cwd().resolve()
assert Path(recovar.__file__).resolve().is_relative_to(repo)
assert Path(jax.__file__).resolve().is_relative_to(repo / ".pixi/envs/default")
assert len(jax.devices("gpu")) == 1 and "H100" in jax.devices("gpu")[0].device_kind.upper()
PY

"${PYTHON}" -m scripts.analyze_k1_live_reconstruction_capture \
    --recovar-root "${RECOVAR_RUN_ROOT}/recovar" \
    --native-stage-dir "${NATIVE_STAGES}" \
    --native-relion-dir "${NATIVE_RELION}" \
    --output-json "${ANALYSIS}/K1_LIVE_RECONSTRUCTION_CAPTURE.json" \
    > "${ANALYSIS}/K1_LIVE_RECONSTRUCTION_CAPTURE.stdout"

sha256sum "${LAUNCHER}" "${ANALYZER}" \
    "${ANALYSIS}/K1_LIVE_RECONSTRUCTION_CAPTURE.json" \
    > "${RUN_ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
