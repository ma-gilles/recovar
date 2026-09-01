#!/usr/bin/env bash
#SBATCH --job-name=k1-terminal-state
#SBATCH --partition=cpu
#SBATCH --account=gilles
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=02:00:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${CASE_SOURCE:?CASE_SOURCE is required}"
: "${RECOVAR_ITERATION:?zero-based RECOVAR numbered iteration is required}"
: "${RELION_ITERATION:?one-based RELION numbered iteration is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"

REPO=${REPO:-/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820}
PYTHON=${REPO}/.pixi/envs/default/bin/python
LAUNCHER=${REPO}/scripts/audit_k1_terminal_state.sbatch
OUTPUT=${RUN_ROOT}/output
ANALYSIS=${RUN_ROOT}/analysis/terminal_state_it${RELION_ITERATION}
DATA=${CASE_SOURCE}/data
RELION=${CASE_SOURCE}/relion_ref
RELION_NUMBERED=${RELION}/run_it$(printf '%03d' "${RELION_ITERATION}")_data.star
RELION_FINAL=${RELION}/run_data.star

[[ "${RECOVAR_ITERATION}" =~ ^[0-9]+$ ]]
[[ "${RELION_ITERATION}" =~ ^[1-9][0-9]*$ ]]
test "$((RECOVAR_ITERATION + 1))" -eq "${RELION_ITERATION}"
test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -s "${OUTPUT}/refinement_results.npz"
test -s "${DATA}/particles.star"
test -s "${RELION_NUMBERED}" -a -s "${RELION_FINAL}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test ! -e "${ANALYSIS}/particle_state.json"

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
"${PYTHON}" -m scripts.audit_em_particle_state_distribution \
  --recovar-results "${OUTPUT}/refinement_results.npz" \
  --recovar-particles-star "${DATA}/particles.star" \
  --relion-star "${RELION_ITERATION}:${RELION_NUMBERED}" \
  --recovar-iteration "${RECOVAR_ITERATION}" \
  --relion-iteration-offset 1 \
  --relion-final-star "${RELION_FINAL}" \
  --output-json "${ANALYSIS}/particle_state.json" \
  --output-npz "${ANALYSIS}/particle_state_arrays.npz" \
  --output-hash-manifest "${ANALYSIS}/particle_state.sha256"

sha256sum "${LAUNCHER}" "${OUTPUT}/refinement_results.npz" \
  "${RELION_NUMBERED}" "${RELION_FINAL}" \
  "${ANALYSIS}/particle_state.json" "${ANALYSIS}/particle_state_arrays.npz" \
  > "${RUN_ROOT}/provenance/terminal_state_${SLURM_JOB_ID}.sha256"
touch "${ANALYSIS}/SUCCESS_${SLURM_JOB_ID}"
