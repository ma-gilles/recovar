#!/usr/bin/env bash
#SBATCH --job-name=k1-coarse-analyze
#SBATCH --partition=cpu
#SBATCH --account=gilles
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -euo pipefail

: "${NATIVE_ROOT:?NATIVE_ROOT is required}"
: "${RECOVAR_CAPTURE:?RECOVAR_CAPTURE is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${NATIVE_ROTATION:?NATIVE_ROTATION is required}"
: "${NATIVE_TRANSLATION:?NATIVE_TRANSLATION is required}"
: "${RECOVAR_ROTATION:?RECOVAR_ROTATION is required}"
: "${RECOVAR_TRANSLATION:?RECOVAR_TRANSLATION is required}"
: "${NATIVE_DIRECTIONS:?NATIVE_DIRECTIONS is required}"
: "${NATIVE_PSI:?NATIVE_PSI is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"

REPO=${REPO:-/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820}
PYTHON=${PYTHON:-${REPO}/.pixi/envs/default/bin/python}
LAUNCHER=${REPO}/scripts/run_k1_native_coarse_stage_analysis.sbatch
OUTPUT_STEM=${OUTPUT_STEM:-K1_NATIVE_COARSE_BOUNDARY}

test -f "${NATIVE_ROOT}/SAFE_TO_DELETE"
test -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -s "${RECOVAR_CAPTURE}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"

mkdir -p "${NATIVE_ROOT}/analysis" "${NATIVE_ROOT}/logs" "${NATIVE_ROOT}/provenance"
mkdir -p "${RUNTIME_ROOT}/tmp" "${RUNTIME_ROOT}/pixi_home" "${RUNTIME_ROOT}/rattler_cache"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1
export TMPDIR=${RUNTIME_ROOT}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/rattler_cache

native_capture=$(find "${NATIVE_ROOT}/capture" -maxdepth 1 -type f -name '*.coarse-v1.bin')
test -n "${native_capture}"
test "$(printf '%s\n' "${native_capture}" | wc -l)" -eq 1

cd "${REPO}"
"${PYTHON}" -m scripts.analyze_k1_native_coarse_boundary \
  --native "${native_capture}" \
  --recovar "${RECOVAR_CAPTURE}" \
  --native-rotation "${NATIVE_ROTATION}" \
  --native-translation "${NATIVE_TRANSLATION}" \
  --recovar-rotation "${RECOVAR_ROTATION}" \
  --recovar-translation "${RECOVAR_TRANSLATION}" \
  --native-directions "${NATIVE_DIRECTIONS}" \
  --native-psi "${NATIVE_PSI}" \
  --output-json "${NATIVE_ROOT}/analysis/${OUTPUT_STEM}.json" \
  > "${NATIVE_ROOT}/logs/analyze_coarse_${SLURM_JOB_ID}.stdout"

sha256sum "${LAUNCHER}" "${native_capture}" "${RECOVAR_CAPTURE}" \
  "${NATIVE_ROOT}/analysis/${OUTPUT_STEM}.json" \
  > "${NATIVE_ROOT}/provenance/coarse_analysis_results_${SLURM_JOB_ID}.sha256"
touch "${NATIVE_ROOT}/COARSE_ANALYSIS_SUCCESS_${SLURM_JOB_ID}"
