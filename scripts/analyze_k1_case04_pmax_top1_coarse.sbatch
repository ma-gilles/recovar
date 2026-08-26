#!/usr/bin/env bash
#SBATCH --job-name=k1c04-pmax-coarse-audit
#SBATCH --partition=cpu
#SBATCH --account=gilles
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -euo pipefail

: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_ANALYZER_SHA256:?EXPECTED_ANALYZER_SHA256 is required}"

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
PYTHON=${REPO}/.pixi/envs/default/bin/python
LAUNCHER=${REPO}/scripts/analyze_k1_case04_pmax_top1_coarse.sbatch
ANALYZER=${REPO}/scripts/analyze_k1_native_coarse_boundary.py
NATIVE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_pmax_top1_native_567b22d4ea_20260825T0910ET
RECOVAR=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_pmax_top8_coarse_exactlib_567b22d4ea_20260825T0905ET
RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/k1_case04_pmax_top1_native_20260825T0910ET
PANEL=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_pmax_prefix2_exactnative_exactlib_567b22d4ea_20260825T0837ET/analysis/control_vs_exactlib_iter2/candidate_top24.json
OUT=${NATIVE}/analysis/P_MAX_TOP1_COARSE_BOUNDARY.json

test -f "${NATIVE}/SAFE_TO_DELETE" -a -f "${RECOVAR}/SAFE_TO_DELETE"
test -f "${RUNTIME}/SAFE_TO_DELETE" -a -s "${PANEL}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${ANALYZER}" | awk '{print $1}')" = "${EXPECTED_ANALYZER_SHA256}"
test ! -e "${OUT}"

source_row=$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["rows"][0]["source_row_zero_based"])' "${PANEL}")
printf -v source_label '%06d' "${source_row}"
recovar_capture=${RECOVAR}/pass1/significance_orig${source_label}_it002_cs100.npz
native_capture=$(find "${NATIVE}/capture" -maxdepth 1 -type f -name '*.coarse-v1.bin')
test -s "${recovar_capture}" -a -s "${native_capture}"
test "$(find "${NATIVE}/capture" -maxdepth 1 -type f -name '*.coarse-v1.bin' | wc -l)" -eq 1

mkdir -p "${NATIVE}/analysis" "${NATIVE}/provenance" "${NATIVE}/logs" \
    "${RUNTIME}/coarse_audit_tmp" "${RUNTIME}/coarse_audit_pixi_home" \
    "${RUNTIME}/coarse_audit_rattler_cache"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1
export TMPDIR=${RUNTIME}/coarse_audit_tmp
export PIXI_HOME=${RUNTIME}/coarse_audit_pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/coarse_audit_rattler_cache

cd "${REPO}"
"${PYTHON}" -m scripts.analyze_k1_native_coarse_boundary \
    --native "${native_capture}" \
    --recovar "${recovar_capture}" \
    --native-rotation 0 --native-translation 0 \
    --recovar-rotation 0 --recovar-translation 0 \
    --native-directions 768 --native-psi 48 \
    --output-json "${OUT}" \
    > "${NATIVE}/logs/analyze_pmax_top1_coarse_${SLURM_JOB_ID}.stdout"

"${PYTHON}" - "${OUT}" "${NATIVE}/analysis/P_MAX_TOP1_COARSE_BOUNDARY_SUMMARY.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
summary = {
    "schema": "recovar.em.k1_case04_pmax_top1_coarse_summary.v1",
    "status": "complete",
    "native_summary": report["native_summary"],
    "recovar_summary": report["recovar_summary"],
    "cutoff_boundary": report["cutoff_boundary"],
    "full_boundary": report["full_boundary"],
    "shared_best": report["shared_best"],
}
Path(sys.argv[2]).write_text(
    json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
)
PY

sha256sum "${LAUNCHER}" "${ANALYZER}" "${PANEL}" "${recovar_capture}" \
    "${native_capture}" "${OUT}" \
    "${NATIVE}/analysis/P_MAX_TOP1_COARSE_BOUNDARY_SUMMARY.json" \
    > "${NATIVE}/provenance/pmax_top1_coarse_analysis_${SLURM_JOB_ID}.sha256"
touch "${NATIVE}/P_MAX_TOP1_COARSE_ANALYSIS_SUCCESS_${SLURM_JOB_ID}"
