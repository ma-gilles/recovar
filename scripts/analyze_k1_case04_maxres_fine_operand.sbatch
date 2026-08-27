#!/usr/bin/env bash
#SBATCH --job-name=k1c04-maxres-op
#SBATCH --partition=cpu
#SBATCH --account=gilles
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:20:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${REPO:?REPO is required}"
: "${RECOVAR_CAPTURE:?RECOVAR_CAPTURE is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_ANALYZER_SHA256:?EXPECTED_ANALYZER_SHA256 is required}"
: "${EXPECTED_RECOVAR_CAPTURE_SHA256:?EXPECTED_RECOVAR_CAPTURE_SHA256 is required}"

PYTHON=${REPO}/.pixi/envs/default/bin/python
LAUNCHER=${REPO}/scripts/analyze_k1_case04_maxres_fine_operand.sbatch
ANALYZER=${REPO}/scripts/analyze_k1_fine_operand_tuple.py
REPORT=${RUN_ROOT}/analysis/K1_CASE04_ROW82009_MAXRES_FINE_OPERAND.json

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${ANALYZER}" | awk '{print $1}')" = "${EXPECTED_ANALYZER_SHA256}"
test "$(sha256sum "${RECOVAR_CAPTURE}" | awk '{print $1}')" = "${EXPECTED_RECOVAR_CAPTURE_SHA256}"
test "$(find "${RUN_ROOT}/capture" -maxdepth 1 -type f -name '*_stack82010_class1.fine-operand-v1.bin' | wc -l)" -eq 1
test ! -e "${REPORT}"

mkdir -p "${RUN_ROOT}/analysis" "${RUN_ROOT}/provenance" \
  "${RUNTIME_ROOT}/analysis_tmp" "${RUNTIME_ROOT}/analysis_pixi_home" \
  "${RUNTIME_ROOT}/analysis_rattler_cache" "${RUNTIME_ROOT}/analysis_pycache"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1
export TMPDIR=${RUNTIME_ROOT}/analysis_tmp
export PIXI_HOME=${RUNTIME_ROOT}/analysis_pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/analysis_rattler_cache
export PYTHONPYCACHEPREFIX=${RUNTIME_ROOT}/analysis_pycache

native_capture=$(find "${RUN_ROOT}/capture" -maxdepth 1 -type f \
  -name '*_stack82010_class1.fine-operand-v1.bin')
command=(
  "${PYTHON}" "${ANALYZER}"
  --capture "${native_capture}"
  --pass2 "${RECOVAR_CAPTURE}"
  --recovar-global-rotation 3
  --physical-image-size 256
  --output "${REPORT}"
)
printf '%q ' "${command[@]}" > "${RUN_ROOT}/provenance/analysis_command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${RUN_ROOT}/provenance/analysis_command_${SLURM_JOB_ID}.sh"
env | sort > "${RUN_ROOT}/provenance/analysis_environment_${SLURM_JOB_ID}.txt"

cd "${REPO}"
"${command[@]}" > "${RUN_ROOT}/analysis/run_${SLURM_JOB_ID}.log"
"${PYTHON}" - "${REPORT}" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
assert report["schema"] == "recovar.em.k1_fine_operand_tuple.v2"
assert report["identity"]["stack_index_one_based"] == 82010
assert report["identity"]["native_rotation_local"] == 3
assert report["identity"]["native_translation"] == 55
assert report["identity"]["recovar_rotation_row"] == 3
assert report["identity"]["recovar_translation_row"] == 55
PY
sha256sum "${LAUNCHER}" "${ANALYZER}" "${RECOVAR_CAPTURE}" \
  "${native_capture}" "${REPORT}" \
  > "${RUN_ROOT}/provenance/analysis_results_${SLURM_JOB_ID}.sha256"
touch "${RUN_ROOT}/ANALYSIS_SUCCESS_${SLURM_JOB_ID}"
