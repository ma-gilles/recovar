#!/usr/bin/env bash
#SBATCH --job-name=k1-live-particle
#SBATCH --partition=cpu
#SBATCH --account=gilles
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${CASE_SOURCE:?CASE_SOURCE is required}"
: "${RECOVAR_ITERATION:?RECOVAR_ITERATION is required}"
: "${PIXEL_SIZE_ANGSTROM:?PIXEL_SIZE_ANGSTROM is required}"
: "${ANALYSIS_TAG:?ANALYSIS_TAG is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_ANALYZER_SHA256:?EXPECTED_ANALYZER_SHA256 is required}"

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_native_units_instrumentation_20260826
PYTHON=${REPO}/.pixi/envs/default/bin/python
LAUNCHER=${REPO}/scripts/audit_k1_live_particle_state.sbatch
ANALYZER=${REPO}/scripts/audit_k1_live_particle_state.py
INTERMEDIATES=${RUN_ROOT}/output/intermediates
[[ "${RECOVAR_ITERATION}" =~ ^[0-9]+$ ]]
[[ "${ANALYSIS_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]
RELION_ITERATION=$((RECOVAR_ITERATION + 1))
printf -v RELION_TAG '%03d' "${RELION_ITERATION}"
RELION_STAR=${CASE_SOURCE}/relion_ref/run_it${RELION_TAG}_data.star
PARTICLES_STAR=${CASE_SOURCE}/data/particles.star
ANALYSIS=${RUN_ROOT}/analysis/live_particle_state_${ANALYSIS_TAG}

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -d "${INTERMEDIATES}"
test -s "${INTERMEDIATES}/it$(printf '%03d' "${RECOVAR_ITERATION}")_particle_state_half1.npz"
test -s "${INTERMEDIATES}/it$(printf '%03d' "${RECOVAR_ITERATION}")_particle_state_half2.npz"
test -s "${PARTICLES_STAR}" -a -s "${RELION_STAR}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${ANALYZER}" | awk '{print $1}')" = "${EXPECTED_ANALYZER_SHA256}"
test ! -e "${ANALYSIS}"

mkdir -p "${ANALYSIS}" "${RUN_ROOT}/provenance" \
    "${RUNTIME_ROOT}/${SLURM_JOB_ID}/tmp" \
    "${RUNTIME_ROOT}/${SLURM_JOB_ID}/pixi_home" \
    "${RUNTIME_ROOT}/${SLURM_JOB_ID}/rattler_cache"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu
export TMPDIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/rattler_cache

cd "${REPO}"
"${PYTHON}" -m scripts.audit_k1_live_particle_state \
    --intermediates-dir "${INTERMEDIATES}" \
    --recovar-particles-star "${PARTICLES_STAR}" \
    --relion-star "${RELION_STAR}" \
    --recovar-iteration "${RECOVAR_ITERATION}" \
    --pixel-size-angstrom "${PIXEL_SIZE_ANGSTROM}" \
    --output-json "${ANALYSIS}/particle_state.json" \
    > "${ANALYSIS}/particle_state.stdout"

sha256sum "${LAUNCHER}" "${ANALYZER}" "${PARTICLES_STAR}" "${RELION_STAR}" \
    "${INTERMEDIATES}/it$(printf '%03d' "${RECOVAR_ITERATION}")_particle_state_half1.npz" \
    "${INTERMEDIATES}/it$(printf '%03d' "${RECOVAR_ITERATION}")_particle_state_half2.npz" \
    "${ANALYSIS}/particle_state.json" \
    > "${RUN_ROOT}/provenance/live_particle_state_${ANALYSIS_TAG}_${SLURM_JOB_ID}.sha256"
touch "${ANALYSIS}/SUCCESS_${SLURM_JOB_ID}"
