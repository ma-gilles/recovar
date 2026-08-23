#!/usr/bin/env bash
#SBATCH --job-name=k1-direct-score
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00

set -euo pipefail
: "${K1_DIRECT_SCORE_ROOT:?set the disposable analysis root}"
: "${K1_DIRECT_SCORE_RUNTIME:?set the disposable runtime root}"
: "${K1_DIRECT_SCORE_REPO:?set the source checkout}"
: "${K1_DIRECT_SCORE_NATIVE_CAPTURE:?set the native fine operand capture}"
: "${K1_DIRECT_SCORE_RECOVAR_CAPTURE:?set the RECOVAR selected pass-2 capture}"
: "${K1_DIRECT_SCORE_CUDA_LIB:?set the pinned CUDA library}"
: "${EXPECTED_LAUNCHER_SHA256:?pin the launcher hash}"
: "${EXPECTED_ANALYZER_SHA256:?pin the analyzer hash}"
: "${EXPECTED_CUDA_SHA256:?pin the CUDA library hash}"

ROOT=${K1_DIRECT_SCORE_ROOT}
RUNTIME=${K1_DIRECT_SCORE_RUNTIME}
REPO=${K1_DIRECT_SCORE_REPO}
NATIVE_CAPTURE=${K1_DIRECT_SCORE_NATIVE_CAPTURE}
RECOVAR_CAPTURE=${K1_DIRECT_SCORE_RECOVAR_CAPTURE}
CUDA_LIB=${K1_DIRECT_SCORE_CUDA_LIB}
LAUNCHER=${REPO}/scripts/run_k1_direct_score_input_analysis.sbatch
ANALYZER=${REPO}/scripts/analyze_k1_direct_score_input_boundary.py
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default
REPORT=${ROOT}/analysis/K1_DIRECT_SCORE_INPUT_BOUNDARY.json

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test -s "${NATIVE_CAPTURE}" -a -s "${RECOVAR_CAPTURE}" -a -s "${CUDA_LIB}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${ANALYZER}" | awk '{print $1}')" = "${EXPECTED_ANALYZER_SHA256}"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"
test ! -e "${REPORT}"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/${SLURM_JOB_ID}/rattler_cache
export PYTHONPYCACHEPREFIX=${RUNTIME}/${SLURM_JOB_ID}/pycache
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" \
    "${PYTHONPYCACHEPREFIX}" "${ROOT}/analysis" "${ROOT}/provenance"

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
export CUDA_VISIBLE_DEVICES=0 RECOVAR_CUDA_LIB=${CUDA_LIB}

cd "${REPO}"
COMMAND=(
    "${PYTHON}" "${ANALYZER}"
    --capture "${NATIVE_CAPTURE}"
    --recovar "${RECOVAR_CAPTURE}"
    --physical-image-size 128
    --output-json "${REPORT}"
)
printf '%q ' "${COMMAND[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
nvidia-smi -q > "${ROOT}/provenance/nvidia_smi_${SLURM_JOB_ID}.txt"

"${COMMAND[@]}" > "${ROOT}/analysis/run_${SLURM_JOB_ID}.log" 2>&1
test -s "${REPORT}"
"${PYTHON}" - <<'PY' "${REPORT}"
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
assert report["schema"] == "recovar.em.k1_direct_score_input_boundary.v1"
assert report["status"] == "complete"
PY
sha256sum "${LAUNCHER}" "${ANALYZER}" "${CUDA_LIB}" \
    "${NATIVE_CAPTURE}" "${RECOVAR_CAPTURE}" "${REPORT}" \
    > "${ROOT}/provenance/science_outputs_${SLURM_JOB_ID}.sha256"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
