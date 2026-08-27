#!/usr/bin/env bash
#SBATCH --job-name=k1-c04-op-analyze
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
: "${PRODUCER_ROOT:?PRODUCER_ROOT is required}"
: "${PRODUCER_JOB:?PRODUCER_JOB is required}"
: "${RECOVAR_CAPTURE:?RECOVAR_CAPTURE is required}"
: "${EXPECTED_NATIVE_TRANSLATION:?EXPECTED_NATIVE_TRANSLATION is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_ANALYZER_SHA256:?EXPECTED_ANALYZER_SHA256 is required}"
: "${EXPECTED_OPERAND_VALIDATOR_SHA256:?EXPECTED_OPERAND_VALIDATOR_SHA256 is required}"
: "${EXPECTED_COMPARISON_HELPER_SHA256:?EXPECTED_COMPARISON_HELPER_SHA256 is required}"
: "${EXPECTED_RECOVAR_CAPTURE_SHA256:?EXPECTED_RECOVAR_CAPTURE_SHA256 is required}"

PYTHON=${REPO}/.pixi/envs/default/bin/python
LAUNCHER=${REPO}/scripts/analyze_k1_case04_top1_fine_operand.sbatch
ANALYZER=${REPO}/scripts/analyze_k1_fine_operand_tuple.py
OPERAND_VALIDATOR=${REPO}/scripts/validate_relion_fine_operand_capture.py
COMPARISON_HELPER=${REPO}/scripts/compare_k4_relion_recovar_fine_operands.py
REPORT=${RUN_ROOT}/analysis/K1_CASE04_TOP1_FINE_OPERAND_T${EXPECTED_NATIVE_TRANSLATION}.json

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -f "${PRODUCER_ROOT}/SUCCESS_${PRODUCER_JOB}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${ANALYZER}" | awk '{print $1}')" = "${EXPECTED_ANALYZER_SHA256}"
test "$(sha256sum "${OPERAND_VALIDATOR}" | awk '{print $1}')" = "${EXPECTED_OPERAND_VALIDATOR_SHA256}"
test "$(sha256sum "${COMPARISON_HELPER}" | awk '{print $1}')" = "${EXPECTED_COMPARISON_HELPER_SHA256}"
test "$(sha256sum "${RECOVAR_CAPTURE}" | awk '{print $1}')" = "${EXPECTED_RECOVAR_CAPTURE_SHA256}"
test "$(find "${PRODUCER_ROOT}/capture" -maxdepth 1 -type f -name '*_stack31560_class1.fine-operand-v1.bin' | wc -l)" -eq 1
test ! -e "${REPORT}"

mkdir -p "${RUN_ROOT}/analysis" "${RUN_ROOT}/provenance" \
  "${RUNTIME_ROOT}/tmp" "${RUNTIME_ROOT}/pixi_home" \
  "${RUNTIME_ROOT}/rattler_cache" "${RUNTIME_ROOT}/pycache"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1
export TMPDIR=${RUNTIME_ROOT}/tmp PIXI_HOME=${RUNTIME_ROOT}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/rattler_cache
export PYTHONPYCACHEPREFIX=${RUNTIME_ROOT}/pycache

native_capture=$(find "${PRODUCER_ROOT}/capture" -maxdepth 1 -type f \
  -name '*_stack31560_class1.fine-operand-v1.bin')
command=(
  "${PYTHON}" -m scripts.analyze_k1_fine_operand_tuple
  --capture "${native_capture}"
  --pass2 "${RECOVAR_CAPTURE}"
  --recovar-global-rotation 78712
  --physical-image-size 256
  --output "${REPORT}"
)
printf '%q ' "${command[@]}" > "${RUN_ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${RUN_ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
env | sort > "${RUN_ROOT}/provenance/environment_${SLURM_JOB_ID}.txt"

cd "${REPO}"
"${command[@]}" > "${RUN_ROOT}/analysis/stdout_${SLURM_JOB_ID}.log"
"${PYTHON}" - "${REPORT}" "${EXPECTED_NATIVE_TRANSLATION}" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
expected_translation = int(sys.argv[2])
identity = report["identity"]
assert report["schema"] == "recovar.em.k1_fine_operand_tuple.v2"
assert identity["stack_index_one_based"] == 31560
assert identity["native_rotation_local"] == 0
assert identity["recovar_rotation_row"] == 0
assert identity["native_translation"] == expected_translation
assert identity["recovar_translation_row"] == expected_translation
PY

sha256sum "${LAUNCHER}" "${ANALYZER}" "${OPERAND_VALIDATOR}" \
  "${COMPARISON_HELPER}" "${RECOVAR_CAPTURE}" "${native_capture}" \
  "${REPORT}" > "${RUN_ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
