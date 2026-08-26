#!/usr/bin/env bash
#SBATCH --job-name=k1c04-pmax-fine-audit
#SBATCH --partition=cpu
#SBATCH --account=gilles
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=01:00:00

set -euo pipefail

: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_ANALYZER_SHA256:?EXPECTED_ANALYZER_SHA256 is required}"

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
PYTHON=${REPO}/.pixi/envs/default/bin/python
LAUNCHER=${REPO}/scripts/analyze_k1_case04_pmax_top1_fine.sbatch
ANALYZER=${REPO}/scripts/analyze_k1_fine_score_stages.py
NATIVE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_pmax_top1_native_567b22d4ea_20260825T0910ET
RECOVAR=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_pmax_top8_fine_exactlib_567b22d4ea_20260825T0912ET
RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/k1_case04_pmax_top1_native_20260825T0910ET
PANEL=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_pmax_prefix2_exactnative_exactlib_567b22d4ea_20260825T0837ET/analysis/control_vs_exactlib_iter2/candidate_top24.json
OUT=${NATIVE}/analysis/P_MAX_TOP1_FINE_SCORE_STAGES.json

test -f "${NATIVE}/SAFE_TO_DELETE" -a -f "${RECOVAR}/SAFE_TO_DELETE"
test -f "${RUNTIME}/SAFE_TO_DELETE" -a -s "${PANEL}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${ANALYZER}" | awk '{print $1}')" = "${EXPECTED_ANALYZER_SHA256}"
test ! -e "${OUT}"

source_row=$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["rows"][0]["source_row_zero_based"])' "${PANEL}")
printf -v source_label '%06d' "${source_row}"
recovar_capture=${RECOVAR}/pass2/pass2_orig${source_label}_cs100.npz
native_fine=$(find "${NATIVE}/capture" -maxdepth 1 -type f -name '*.fine-score-v1.bin')
native_factor=$(find "${NATIVE}/capture" -maxdepth 1 -type f -name '*.bpre-v2.bin')
test -s "${recovar_capture}" -a -s "${native_fine}" -a -s "${native_factor}"
test "$(find "${NATIVE}/capture" -maxdepth 1 -type f -name '*.fine-score-v1.bin' | wc -l)" -eq 1
test "$(find "${NATIVE}/capture" -maxdepth 1 -type f -name '*.bpre-v2.bin' | wc -l)" -eq 1

mkdir -p "${NATIVE}/analysis" "${NATIVE}/provenance" \
    "${RUNTIME}/fine_audit_tmp" "${RUNTIME}/fine_audit_pixi_home" \
    "${RUNTIME}/fine_audit_rattler_cache"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1
export TMPDIR=${RUNTIME}/fine_audit_tmp
export PIXI_HOME=${RUNTIME}/fine_audit_pixi_home
export RATTLER_CACHE_DIR=${RUNTIME}/fine_audit_rattler_cache
export RECOVAR_RELION_BIND_BUILD_DIR=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_prefetched_ddc9d5f54_20260820T1615ET/relion_bind_build/shared

cd "${REPO}"
"${PYTHON}" -m scripts.analyze_k1_fine_score_stages \
    --native-factor "${native_factor}" \
    --native-fine-score "${native_fine}" \
    --recovar-capture "${recovar_capture}" \
    --physical-image-size 256 --top-count 64 --output-json "${OUT}" \
    > "${NATIVE}/logs/analyze_pmax_top1_fine_${SLURM_JOB_ID}.stdout"

"${PYTHON}" - "${OUT}" "${NATIVE}/analysis/P_MAX_TOP1_FINE_SCORE_STAGES_SUMMARY.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
summary = {
    "schema": "recovar.em.k1_case04_pmax_top1_fine_summary.v1",
    "status": "complete",
    "first_exact_unequal_boundary": report["first_exact_unequal_boundary"],
    "stage_exact": report["stage_exact"],
    "native_active_count": report["native_active_count"],
    "recovar_active_candidate_count": report["recovar_active_candidate_count"],
    "native_significant_count": report["native_significant_count"],
    "recovar_significant_count": report["recovar_significant_count"],
    "support_native_only_count": report["support_native_only_count"],
    "support_recovar_only_count": report["support_recovar_only_count"],
    "comparisons": report["comparisons"],
}
Path(sys.argv[2]).write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
PY

sha256sum "${LAUNCHER}" "${ANALYZER}" "${PANEL}" "${recovar_capture}" \
    "${native_fine}" "${native_factor}" "${OUT}" \
    "${NATIVE}/analysis/P_MAX_TOP1_FINE_SCORE_STAGES_SUMMARY.json" \
    > "${NATIVE}/provenance/pmax_top1_fine_analysis_${SLURM_JOB_ID}.sha256"
touch "${NATIVE}/P_MAX_TOP1_FINE_ANALYSIS_SUCCESS_${SLURM_JOB_ID}"
