#!/usr/bin/env bash
#SBATCH --job-name=k1-fine-analyze
#SBATCH --partition=cpu
#SBATCH --account=gilles
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -euo pipefail

: "${NATIVE_ROOT:?NATIVE_ROOT is required}"
: "${RECOVAR_ROOT:?RECOVAR_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${RECOVAR_CAPTURE_NAME:?RECOVAR_CAPTURE_NAME is required}"
: "${PHYSICAL_IMAGE_SIZE:?PHYSICAL_IMAGE_SIZE is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"

REPO=${REPO:-/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820}
PYTHON=${PYTHON:-${REPO}/.pixi/envs/default/bin/python}
LAUNCHER=${REPO}/scripts/run_k1_fine_score_stage_analysis.sbatch
OUTPUT_STEM=${OUTPUT_STEM:-K1_FINE_SCORE_STAGES_RECOVAR}
TOP_COUNT=${TOP_COUNT:-25}
ANALYSIS=${NATIVE_ROOT}/analysis

test -f "${NATIVE_ROOT}/SAFE_TO_DELETE"
test -f "${RECOVAR_ROOT}/SAFE_TO_DELETE"
test -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test "${PHYSICAL_IMAGE_SIZE}" -gt 0
test "${TOP_COUNT}" -ge 2
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"

mkdir -p "${NATIVE_ROOT}/logs" "${ANALYSIS}" "${NATIVE_ROOT}/provenance"
mkdir -p "${RUNTIME_ROOT}/tmp" "${RUNTIME_ROOT}/pixi_home" "${RUNTIME_ROOT}/rattler_cache"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1
export TMPDIR=${RUNTIME_ROOT}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/rattler_cache
export RECOVAR_RELION_BIND_BUILD_DIR=${RECOVAR_RELION_BIND_BUILD_DIR:-/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_remaining3_normspectrum_fb3995ca_20260821T2025ET/relion_bind_build/shared}

fine_score=$(find "${NATIVE_ROOT}/capture" -maxdepth 1 -type f -name '*.fine-score-v1.bin')
factor=$(find "${NATIVE_ROOT}/capture" -maxdepth 1 -type f -name '*.bpre-v2.bin')
recovar_capture=${RECOVAR_ROOT}/pass2/${RECOVAR_CAPTURE_NAME}
test -n "${fine_score}" -a -n "${factor}"
test "$(find "${NATIVE_ROOT}/capture" -maxdepth 1 -type f -name '*.fine-score-v1.bin' | wc -l)" -eq 1
test "$(find "${NATIVE_ROOT}/capture" -maxdepth 1 -type f -name '*.bpre-v2.bin' | wc -l)" -eq 1
test -s "${recovar_capture}"

cd "${REPO}"
"${PYTHON}" -m scripts.analyze_k1_fine_score_stages \
  --native-factor "${factor}" \
  --native-fine-score "${fine_score}" \
  --recovar-capture "${recovar_capture}" \
  --physical-image-size "${PHYSICAL_IMAGE_SIZE}" \
  --top-count "${TOP_COUNT}" \
  --output-json "${ANALYSIS}/${OUTPUT_STEM}.json" \
  > "${NATIVE_ROOT}/logs/analyze_fine_score_${SLURM_JOB_ID}.stdout"

"${PYTHON}" - "${ANALYSIS}/${OUTPUT_STEM}.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
report = json.loads(path.read_text())
summary = {
    "schema": "recovar.em.k1_first_residual_summary.v1",
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
out = path.with_name(path.stem + "_SUMMARY.json")
out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

sha256sum "${LAUNCHER}" "${ANALYSIS}/${OUTPUT_STEM}.json" \
  "${ANALYSIS}/${OUTPUT_STEM}_SUMMARY.json" \
  > "${NATIVE_ROOT}/provenance/analysis_results_${SLURM_JOB_ID}.sha256"
touch "${NATIVE_ROOT}/ANALYSIS_SUCCESS_${SLURM_JOB_ID}"
