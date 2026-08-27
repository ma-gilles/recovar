#!/usr/bin/env bash
#SBATCH --job-name=k1-final-merge-boundary
#SBATCH --partition=cpu
#SBATCH --account=gilles
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -euo pipefail

: "${REPO:?REPO is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${RELION_DIR:?RELION_DIR is required}"
: "${EXPECTED_HEAD:?EXPECTED_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_ANALYZER_SHA256:?EXPECTED_ANALYZER_SHA256 is required}"

PYTHON=${REPO}/.pixi/envs/default/bin/python
LAUNCHER=${REPO}/scripts/audit_k1_final_merge_boundary.sbatch
ANALYZER=${REPO}/scripts/analyze_k1_final_merge_boundary.py
OUTPUT=${RUN_ROOT}/output
ANALYSIS=${RUN_ROOT}/analysis_final_merge_boundary

test -f "${RUN_ROOT}/SAFE_TO_DELETE"
test -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -x "${PYTHON}"
for path in \
  "${OUTPUT}/final_half1_unfil.mrc" \
  "${OUTPUT}/final_half2_unfil.mrc" \
  "${OUTPUT}/final_merged.mrc" \
  "${RELION_DIR}/run_half1_class001_unfil.mrc" \
  "${RELION_DIR}/run_half2_class001_unfil.mrc" \
  "${RELION_DIR}/run_class001.mrc"; do
  test -s "${path}"
done
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_HEAD}"
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
"${PYTHON}" -m scripts.analyze_k1_final_merge_boundary \
  --recovar-half1 "${OUTPUT}/final_half1_unfil.mrc" \
  --recovar-half2 "${OUTPUT}/final_half2_unfil.mrc" \
  --recovar-merged "${OUTPUT}/final_merged.mrc" \
  --relion-half1 "${RELION_DIR}/run_half1_class001_unfil.mrc" \
  --relion-half2 "${RELION_DIR}/run_half2_class001_unfil.mrc" \
  --relion-merged "${RELION_DIR}/run_class001.mrc" \
  --output-json "${ANALYSIS}/merge_boundary.json" \
  --output-shellwise-npz "${ANALYSIS}/merge_boundary_shellwise.npz"

sha256sum "${LAUNCHER}" "${ANALYZER}" \
  "${OUTPUT}/final_half1_unfil.mrc" \
  "${OUTPUT}/final_half2_unfil.mrc" \
  "${OUTPUT}/final_merged.mrc" \
  "${RELION_DIR}/run_half1_class001_unfil.mrc" \
  "${RELION_DIR}/run_half2_class001_unfil.mrc" \
  "${RELION_DIR}/run_class001.mrc" \
  "${ANALYSIS}/merge_boundary.json" \
  "${ANALYSIS}/merge_boundary_shellwise.npz" \
  > "${RUN_ROOT}/provenance/final_merge_boundary_${SLURM_JOB_ID}.sha256"
touch "${ANALYSIS}/SUCCESS_${SLURM_JOB_ID}"
