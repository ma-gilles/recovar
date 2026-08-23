#!/usr/bin/env bash
#SBATCH --job-name=k1-target-matrix
#SBATCH --partition=cpu
#SBATCH --account=gilles
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -euo pipefail

: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_ANALYZER_SHA256:?EXPECTED_ANALYZER_SHA256 is required}"

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
PYTHON=${REPO}/.pixi/envs/default/bin/python
LAUNCHER=${REPO}/scripts/run_k1_target82009_factor_matrix.sbatch
ANALYZER=${REPO}/scripts/analyze_k1_target_factor_matrix.py
SPECTRUM=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_target82009_it2_matrixfaithful_fb3995ca_20260821T2255ET
EXACT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_target82009_spectrum_exact_it2_fb3995ca_20260821T2335ET
LIVE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_target82009_spectrum_live_it2_fb3995ca_20260821T2335ET
COMBINED=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_target82009_spectrum_combined_it2_fb3995ca_20260821T2335ET
BASELINE=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_target82009_baseline_no_spectrum_it2_fb3995ca_20260821T2355ET

test -f "${OUTPUT_ROOT}/SAFE_TO_DELETE"
test -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${ANALYZER}" | awk '{print $1}')" = "${EXPECTED_ANALYZER_SHA256}"
for root in "${SPECTRUM}" "${EXACT}" "${LIVE}" "${COMBINED}" "${BASELINE}"; do
  test -f "${root}/SAFE_TO_DELETE"
done

mkdir -p "${OUTPUT_ROOT}/analysis" "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/provenance"
mkdir -p "${RUNTIME_ROOT}/tmp" "${RUNTIME_ROOT}/pixi_home" "${RUNTIME_ROOT}/rattler_cache"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1
export TMPDIR=${RUNTIME_ROOT}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/rattler_cache

JSON=${OUTPUT_ROOT}/analysis/K1_TARGET82009_FACTOR_MATRIX.json
MARKDOWN=${OUTPUT_ROOT}/analysis/K1_TARGET82009_FACTOR_MATRIX.md
cd "${REPO}"
"${PYTHON}" -m scripts.analyze_k1_target_factor_matrix \
  --arm "baseline=${BASELINE}" \
  --arm "spectrum_only=${SPECTRUM}" \
  --arm "spectrum_exact_bpref=${EXACT}" \
  --arm "spectrum_live_noise=${LIVE}" \
  --arm "spectrum_exact_bpref_live_noise=${COMBINED}" \
  --output-json "${JSON}" --output-markdown "${MARKDOWN}" \
  > "${OUTPUT_ROOT}/logs/factor_matrix_${SLURM_JOB_ID}.stdout"

sha256sum "${LAUNCHER}" "${ANALYZER}" "${JSON}" "${MARKDOWN}" \
  > "${OUTPUT_ROOT}/provenance/factor_matrix_${SLURM_JOB_ID}.sha256"
touch "${OUTPUT_ROOT}/FACTOR_MATRIX_SUCCESS_${SLURM_JOB_ID}"
