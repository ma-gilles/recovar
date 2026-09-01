#!/usr/bin/env bash
#SBATCH --job-name=k1-prefix-audit
#SBATCH --partition=cpu
#SBATCH --account=gilles
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=01:00:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${CASE_SOURCE:?CASE_SOURCE is required}"
: "${BASELINE_PARTICLES:?BASELINE_PARTICLES is required}"
: "${BASELINE_FSC:?BASELINE_FSC is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
PYTHON=${REPO}/.pixi/envs/default/bin/python
LAUNCHER=${REPO}/scripts/audit_k1_existing_relion_prefix_movement.sbatch
OUTPUT=${RUN_ROOT}/output
ANALYSIS=${RUN_ROOT}/analysis
DATA=${CASE_SOURCE}/data
RELION=${CASE_SOURCE}/relion_ref

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -s "${OUTPUT}/refinement_results.npz"
test -s "${DATA}/particles.star"
test -s "${DATA}/reference_gt.mrc"
test -s "${RELION}/run_it001_data.star"
test -s "${RELION}/run_it002_data.star"
test -s "${BASELINE_PARTICLES}"
test -s "${BASELINE_FSC}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"

mkdir -p "${ANALYSIS}" "${RUN_ROOT}/provenance"
mkdir -p "${RUNTIME_ROOT}/tmp" "${RUNTIME_ROOT}/pixi_home" "${RUNTIME_ROOT}/rattler_cache"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1
export TMPDIR=${RUNTIME_ROOT}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/rattler_cache

cd "${REPO}"
"${PYTHON}" -m scripts.audit_k1_fsc_trajectory \
  --case-root "${CASE_SOURCE}" \
  --recovar-dir "${OUTPUT}" \
  --relion-dir "${RELION}" \
  --gt-volume "${DATA}/reference_gt.mrc" \
  --numbered-only --allow-incomplete \
  --output-json "${ANALYSIS}/prefix_fsc.json" \
  --output-markdown "${ANALYSIS}/prefix_fsc.md" \
  --output-shellwise-npz "${ANALYSIS}/prefix_fsc_shellwise.npz"

"${PYTHON}" -m scripts.audit_em_particle_state_distribution \
  --recovar-results "${OUTPUT}/refinement_results.npz" \
  --recovar-particles-star "${DATA}/particles.star" \
  --relion-star "1:${RELION}/run_it001_data.star" \
  --relion-star "2:${RELION}/run_it002_data.star" \
  --recovar-iteration 0 --recovar-iteration 1 --relion-iteration-offset 1 \
  --output-json "${ANALYSIS}/candidate_particle_state.json" \
  --output-npz "${ANALYSIS}/candidate_particle_state_arrays.npz" \
  --output-hash-manifest "${ANALYSIS}/candidate_particle_state.sha256"

"${PYTHON}" -m scripts.analyze_k1_autonomous_boundary_movement \
  --baseline-particles "${BASELINE_PARTICLES}" \
  --candidate-particles "${ANALYSIS}/candidate_particle_state_arrays.npz" \
  --baseline-fsc "${BASELINE_FSC}" \
  --candidate-fsc "${ANALYSIS}/prefix_fsc.json" \
  --particle-iteration 2 --relion-iteration 2 \
  --scope "K=1 matrix-faithful powerClass normalization prefix A/B" \
  --output-json "${ANALYSIS}/prefix_movement.json" \
  --output-markdown "${ANALYSIS}/prefix_movement.md"

sha256sum "${LAUNCHER}" "${OUTPUT}/refinement_results.npz" \
  "${ANALYSIS}/prefix_fsc.json" \
  "${ANALYSIS}/prefix_fsc_shellwise.npz" \
  "${ANALYSIS}/candidate_particle_state.json" \
  "${ANALYSIS}/candidate_particle_state_arrays.npz" \
  "${ANALYSIS}/prefix_movement.json" \
  > "${RUN_ROOT}/provenance/prefix_audit_${SLURM_JOB_ID}.sha256"
touch "${RUN_ROOT}/AUDIT_SUCCESS_${SLURM_JOB_ID}"
