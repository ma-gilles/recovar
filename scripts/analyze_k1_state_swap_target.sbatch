#!/usr/bin/env bash
#SBATCH --job-name=k1-state-swap-audit
#SBATCH --partition=cpu
#SBATCH --account=gilles
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${SCIENCE_JOB_ID:?SCIENCE_JOB_ID is required}"
: "${CAPTURE_PATH:?CAPTURE_PATH is required}"
: "${SIGNIFICANCE_CAPTURE_PATH:?SIGNIFICANCE_CAPTURE_PATH is required}"
: "${AUTONOMOUS_INTERMEDIATES_DIR:?AUTONOMOUS_INTERMEDIATES_DIR is required}"
: "${RECOVAR_PARTICLES_STAR:?RECOVAR_PARTICLES_STAR is required}"
: "${RELION_STAR:?RELION_STAR is required}"
: "${RECOVAR_ITERATION:?RECOVAR_ITERATION is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_ANALYZER_SHA256:?EXPECTED_ANALYZER_SHA256 is required}"

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_native_units_instrumentation_20260826
PYTHON=${REPO}/.pixi/envs/default/bin/python
LAUNCHER=${REPO}/scripts/analyze_k1_state_swap_target.sbatch
ANALYZER=${REPO}/scripts/analyze_k1_state_swap_target.py
OUTPUT=${OUTPUT_JSON:-${RUN_ROOT}/analysis/K1_STATE_SWAP_TARGET.json}

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -f "${RUN_ROOT}/SUCCESS_${SCIENCE_JOB_ID}"
test -s "${CAPTURE_PATH}" -a -s "${SIGNIFICANCE_CAPTURE_PATH}"
test -s "${RECOVAR_PARTICLES_STAR}" -a -s "${RELION_STAR}"
test -d "${AUTONOMOUS_INTERMEDIATES_DIR}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${ANALYZER}" | awk '{print $1}')" = "${EXPECTED_ANALYZER_SHA256}"
test ! -e "${OUTPUT}"

mkdir -p "${RUN_ROOT}/analysis" "${RUN_ROOT}/provenance" \
    "${RUNTIME_ROOT}/${SLURM_JOB_ID}/tmp" \
    "${RUNTIME_ROOT}/${SLURM_JOB_ID}/pixi_home" \
    "${RUNTIME_ROOT}/${SLURM_JOB_ID}/rattler_cache"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu
export TMPDIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/rattler_cache

cd "${REPO}"
"${PYTHON}" -m scripts.analyze_k1_state_swap_target \
    --capture "${CAPTURE_PATH}" \
    --significance-capture "${SIGNIFICANCE_CAPTURE_PATH}" \
    --autonomous-intermediates-dir "${AUTONOMOUS_INTERMEDIATES_DIR}" \
    --recovar-particles-star "${RECOVAR_PARTICLES_STAR}" \
    --relion-star "${RELION_STAR}" \
    --recovar-iteration "${RECOVAR_ITERATION}" \
    --output-json "${OUTPUT}" \
    > "${RUN_ROOT}/analysis/state_swap_target_${SLURM_JOB_ID}.stdout"

sha256sum "${LAUNCHER}" "${ANALYZER}" "${CAPTURE_PATH}" \
    "${SIGNIFICANCE_CAPTURE_PATH}" \
    "${RECOVAR_PARTICLES_STAR}" "${RELION_STAR}" "${OUTPUT}" \
    > "${RUN_ROOT}/provenance/state_swap_target_${SLURM_JOB_ID}.sha256"
touch "${RUN_ROOT}/analysis/SUCCESS_${SLURM_JOB_ID}"
