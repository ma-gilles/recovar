#!/usr/bin/env bash
#SBATCH --job-name=k1-native-boundary
#SBATCH --partition=cpu
#SBATCH --account=gilles
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00

set -euo pipefail

trap 'status=$?; printf "ERROR status=%d line=%d command=%q\n" "${status}" "${LINENO}" "${BASH_COMMAND}" >&2' ERR

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${RELION_DUMP_DIR:?RELION_DUMP_DIR is required}"
: "${RECOVAR_CAPTURE:?RECOVAR_CAPTURE is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_ANALYZER_SHA256:?EXPECTED_ANALYZER_SHA256 is required}"
: "${EXPECTED_RECOVAR_CAPTURE_SHA256:?EXPECTED_RECOVAR_CAPTURE_SHA256 is required}"

REPO=${REPO:-/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_native_units_instrumentation_20260826}
PYTHON=${PYTHON:-${REPO}/.pixi/envs/default/bin/python}
LAUNCHER=${REPO}/scripts/analyze_k1_native_candidate_score_boundary.sbatch
ANALYZER=${REPO}/scripts/analyze_k1_native_candidate_score_boundary.py
OUTPUT=${OUTPUT:-${RUN_ROOT}/analysis/K1_NATIVE_CANDIDATE_SCORE_BOUNDARY.json}

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -s "${RECOVAR_CAPTURE}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${ANALYZER}" | awk '{print $1}')" = "${EXPECTED_ANALYZER_SHA256}"
test "$(sha256sum "${RECOVAR_CAPTURE}" | awk '{print $1}')" = "${EXPECTED_RECOVAR_CAPTURE_SHA256}"
test ! -e "${OUTPUT}"

for artifact in \
    pass1_class0_fine_eulers.bin \
    pass1_acc_rot_idx.bin \
    pass1_acc_trans_idx.bin \
    pass1_exp_Mweight_raw_preprior.bin \
    pass1_candidate_combined_log_prior.bin \
    pass1_candidate_weight_normalized.bin \
    pass1_candidate_in_reconstruction_set.bin; do
    test -s "${RELION_DUMP_DIR}/${artifact}"
done

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
command=("${PYTHON}" -m scripts.analyze_k1_native_candidate_score_boundary
    --relion-dump-dir "${RELION_DUMP_DIR}"
    --recovar-pass2-npz "${RECOVAR_CAPTURE}"
    --output-json "${OUTPUT}")
printf '%q ' "${command[@]}" > "${RUN_ROOT}/provenance/native_candidate_boundary_command_${SLURM_JOB_ID}.sh"
printf '\n' >> "${RUN_ROOT}/provenance/native_candidate_boundary_command_${SLURM_JOB_ID}.sh"
env | sort > "${RUN_ROOT}/provenance/native_candidate_boundary_environment_${SLURM_JOB_ID}.txt"
"${command[@]}" > "${RUN_ROOT}/logs/native_candidate_boundary_${SLURM_JOB_ID}.stdout" \
    2> "${RUN_ROOT}/logs/native_candidate_boundary_${SLURM_JOB_ID}.stderr"

test -s "${OUTPUT}"
"${PYTHON}" - "${OUTPUT}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
assert payload["status"] == "complete"
assert payload["candidate_set"]["common_count"] > 0
assert "first_nonidentical_boundary" in payload
PY
sha256sum "${LAUNCHER}" "${ANALYZER}" "${RECOVAR_CAPTURE}" "${OUTPUT}" \
    > "${RUN_ROOT}/provenance/native_candidate_boundary_results_${SLURM_JOB_ID}.sha256"
touch "${RUN_ROOT}/analysis/SUCCESS_NATIVE_CANDIDATE_BOUNDARY_${SLURM_JOB_ID}"
