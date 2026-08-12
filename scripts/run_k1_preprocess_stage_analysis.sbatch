#!/usr/bin/env bash
#SBATCH --job-name=k1-preprocess-stage
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
: "${K1_PREPROCESS_ANALYSIS_ROOT:?set the disposable analysis root}"
: "${K1_PREPROCESS_ANALYSIS_RUNTIME:?set the disposable runtime root}"
: "${K1_PREPROCESS_ANALYSIS_REPO:?set the source checkout}"
: "${K1_PREPROCESS_NATIVE_DIR:?set the native preprocess capture directory}"
: "${K1_PREPROCESS_CUDA_LIB:?set the pinned CUDA library}"
: "${EXPECTED_LAUNCHER_SHA256:?pin the launcher hash}"
: "${EXPECTED_ANALYZER_SHA256:?pin the analyzer hash}"
: "${EXPECTED_CUDA_SHA256:?pin the CUDA library hash}"

ROOT=${K1_PREPROCESS_ANALYSIS_ROOT}
RUNTIME=${K1_PREPROCESS_ANALYSIS_RUNTIME}
REPO=${K1_PREPROCESS_ANALYSIS_REPO}
NATIVE_DIR=${K1_PREPROCESS_NATIVE_DIR}
CUDA_LIB=${K1_PREPROCESS_CUDA_LIB}
LAUNCHER=${REPO}/scripts/run_k1_preprocess_stage_analysis.sbatch
ANALYZER=${REPO}/scripts/analyze_k1_preprocess_stage_panel.py
PYTHON=${REPO}/.pixi/envs/default/bin/python
PIXI_ENV=${REPO}/.pixi/envs/default

test -f "${ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME}/SAFE_TO_DELETE"
test -s "${NATIVE_DIR}/part639_stack1574.preprocess-v1.bin" -a -s "${CUDA_LIB}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${ANALYZER}" | awk '{print $1}')" = "${EXPECTED_ANALYZER_SHA256}"
test "$(sha256sum "${CUDA_LIB}" | awk '{print $1}')" = "${EXPECTED_CUDA_SHA256}"

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
for mode in default native_lane native_atomic; do
    extra=()
    case "${mode}" in
        default) ;;
        native_lane) extra+=(--native-lane-reduction) ;;
        native_atomic) extra+=(--native-atomic-reduction --native-atomic-repeats 8) ;;
    esac
    command=(
        "${PYTHON}" "${ANALYZER}"
        --native-preprocess-dir "${NATIVE_DIR}"
        --expected-particles 1
        --output "${ROOT}/analysis/K1_PREPROCESS_STAGE_${mode}.json"
        --output-npz "${ROOT}/analysis/K1_PREPROCESS_STAGE_${mode}.npz"
        "${extra[@]}"
    )
    printf '%q ' "${command[@]}" > "${ROOT}/provenance/command_${SLURM_JOB_ID}_${mode}.sh"
    printf '\n' >> "${ROOT}/provenance/command_${SLURM_JOB_ID}_${mode}.sh"
    "${command[@]}" > "${ROOT}/analysis/run_${SLURM_JOB_ID}_${mode}.log" 2>&1
done

"${PYTHON}" - <<'PY' "${ROOT}/analysis"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for mode in ("default", "native_lane", "native_atomic"):
    report = json.loads((root / f"K1_PREPROCESS_STAGE_{mode}.json").read_text())
    assert report["schema"] == "recovar.em.k1_preprocess_stage_panel.v1"
    assert len(report["particles"]) == 1
PY
sha256sum "${LAUNCHER}" "${ANALYZER}" "${CUDA_LIB}" \
    "${NATIVE_DIR}/part639_stack1574.preprocess-v1.bin" \
    "${ROOT}"/analysis/K1_PREPROCESS_STAGE_*.json \
    "${ROOT}"/analysis/K1_PREPROCESS_STAGE_*.npz \
    > "${ROOT}/provenance/science_outputs_${SLURM_JOB_ID}.sha256"
touch "${ROOT}/SUCCESS_${SLURM_JOB_ID}"
