#!/usr/bin/env bash
#SBATCH --job-name=k1-native-operands
#SBATCH --partition=cpu
#SBATCH --account=gilles
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=160G
#SBATCH --time=01:30:00

set -euo pipefail

trap 'status=$?; printf "ERROR status=%d line=%d command=%q\n" "${status}" "${LINENO}" "${BASH_COMMAND}" >&2' ERR

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${NATIVE_VERBOSE_ROOT:?NATIVE_VERBOSE_ROOT is required}"
: "${RECOVAR_CAPTURE:?RECOVAR_CAPTURE is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_ANALYZER_SHA256:?EXPECTED_ANALYZER_SHA256 is required}"
: "${EXPECTED_RECOVAR_CAPTURE_SHA256:?EXPECTED_RECOVAR_CAPTURE_SHA256 is required}"

REPO=${REPO:-/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820}
PYTHON=${PYTHON:-${REPO}/.pixi/envs/default/bin/python}
LAUNCHER=${REPO}/scripts/analyze_k1_native_verbose_recovar_operands.sbatch
ANALYZER=${REPO}/scripts/analyze_em_k1_native_fine_operands.py
FULL_IMAGE_SIZE=${K1_FULL_IMAGE_SIZE:-256}
CHUNK_SIZE=${K1_OPERAND_ANALYSIS_CHUNK_SIZE:-32}
RECOVAR_SCORE_UNITS=${K1_RECOVAR_SCORE_UNITS:-normalized}
OUTPUT=${K1_OPERAND_ANALYSIS_OUTPUT:-${RUN_ROOT}/analysis/K1_NATIVE_VERBOSE_VS_RECOVAR_OPERANDS.json}

case "${FULL_IMAGE_SIZE}" in ''|*[!0-9]*) echo "K1_FULL_IMAGE_SIZE must be a positive integer" >&2; exit 2 ;; esac
case "${CHUNK_SIZE}" in ''|*[!0-9]*) echo "K1_OPERAND_ANALYSIS_CHUNK_SIZE must be a positive integer" >&2; exit 2 ;; esac
test "${FULL_IMAGE_SIZE}" -gt 0 -a "${CHUNK_SIZE}" -gt 0
case "${RECOVAR_SCORE_UNITS}" in
    native|normalized) ;;
    *) echo "K1_RECOVAR_SCORE_UNITS must be native or normalized" >&2; exit 2 ;;
esac
test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -f "${NATIVE_VERBOSE_ROOT}/SAFE_TO_DELETE"
test -s "${RECOVAR_CAPTURE}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${ANALYZER}" | awk '{print $1}')" = "${EXPECTED_ANALYZER_SHA256}"
test "$(sha256sum "${RECOVAR_CAPTURE}" | awk '{print $1}')" = "${EXPECTED_RECOVAR_CAPTURE_SHA256}"
test ! -e "${OUTPUT}"

for artifact in \
    pass1_exp_Mweight_raw_preprior.bin \
    pass1_acc_rot_idx.bin \
    pass1_acc_trans_idx.bin \
    pass1_candidate_combined_log_prior.bin \
    pass1_candidate_weight_normalized.bin \
    pass1_candidate_in_reconstruction_set.bin \
    pass1_class0_fine_eulers.bin \
    pass1_class0_fine_ref_real.bin \
    pass1_class0_fine_ref_imag.bin \
    pass1_class0_fine_shifted_real.bin \
    pass1_class0_fine_shifted_imag.bin; do
    test -s "${NATIVE_VERBOSE_ROOT}/capture/${artifact}"
done
test "$(find "${NATIVE_VERBOSE_ROOT}/capture" -maxdepth 1 -type f -name 'pass1_img*_corr_img.bin' | wc -l)" -eq 1

mkdir -p "$(dirname "${OUTPUT}")" "${RUN_ROOT}/logs" "${RUN_ROOT}/provenance" \
    "${RUNTIME_ROOT}/tmp" "${RUNTIME_ROOT}/pixi_home" \
    "${RUNTIME_ROOT}/rattler_cache" "${RUNTIME_ROOT}/pycache"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE
export PYTHONNOUSERSITE=1
export TMPDIR=${RUNTIME_ROOT}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/rattler_cache
export PYTHONPYCACHEPREFIX=${RUNTIME_ROOT}/pycache

cd "${REPO}"
command=("${PYTHON}" -m scripts.analyze_em_k1_native_fine_operands
    --relion-dump-dir "${NATIVE_VERBOSE_ROOT}/capture"
    --recovar-pass2-npz "${RECOVAR_CAPTURE}"
    --full-image-size "${FULL_IMAGE_SIZE}"
    --chunk-size "${CHUNK_SIZE}"
    --recovar-score-units "${RECOVAR_SCORE_UNITS}"
    --output-json "${OUTPUT}")
printf '%q ' "${command[@]}" > "${RUN_ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${RUN_ROOT}/provenance/command_${SLURM_JOB_ID}.sh"
env | sort > "${RUN_ROOT}/provenance/environment_${SLURM_JOB_ID}.txt"
start_epoch=$(date +%s)
"${command[@]}" > "${RUN_ROOT}/logs/analyze_${SLURM_JOB_ID}.stdout" \
    2> "${RUN_ROOT}/logs/analyze_${SLURM_JOB_ID}.stderr"
wall_s=$(( $(date +%s) - start_epoch ))

test -s "${OUTPUT}"
"${PYTHON}" - "${OUTPUT}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
assert payload["status"] == "complete"
assert payload["candidate_keys_exact"] is True
assert payload["candidate_count"] > 0
PY
find "${NATIVE_VERBOSE_ROOT}/capture" -maxdepth 1 -type f -name '*.bin' -print0 \
    | sort -z | xargs -0 sha256sum > "${RUN_ROOT}/provenance/native_inputs_${SLURM_JOB_ID}.sha256"
sha256sum "${LAUNCHER}" "${ANALYZER}" "${RECOVAR_CAPTURE}" "${OUTPUT}" \
    > "${RUN_ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
printf '{"job_id":"%s","wall_s":%d,"full_image_size":%d,"chunk_size":%d,"recovar_score_units":"%s"}\n' \
    "${SLURM_JOB_ID}" "${wall_s}" "${FULL_IMAGE_SIZE}" "${CHUNK_SIZE}" \
    "${RECOVAR_SCORE_UNITS}" \
    > "${RUN_ROOT}/provenance/walltime_${SLURM_JOB_ID}.json"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
