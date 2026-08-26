#!/usr/bin/env bash
#SBATCH --job-name=k1-pre-tau2
#SBATCH --partition=cpu
#SBATCH --account=gilles
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:20:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${FT_Y_HALF1:?FT_Y_HALF1 is required}"
: "${FT_Y_HALF2:?FT_Y_HALF2 is required}"
: "${FT_CTF_HALF1:?FT_CTF_HALF1 is required}"
: "${FT_CTF_HALF2:?FT_CTF_HALF2 is required}"
: "${REFINEMENT_RESULTS:?REFINEMENT_RESULTS is required}"
: "${EXPECTED_HEAD:?EXPECTED_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_ANALYZER_SHA256:?EXPECTED_ANALYZER_SHA256 is required}"

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
PYTHON=${REPO}/.pixi/envs/default/bin/python
LAUNCHER=${REPO}/scripts/build_k1_pre_reconstruction_tau2.sbatch
ANALYZER=${REPO}/scripts/build_k1_pre_reconstruction_tau2.py
OUTPUT_DIR=${OUTPUT_DIR:-${RUN_ROOT}/analysis/pre_reconstruction_tau2}
ITERATION_INDEX=${ITERATION_INDEX:-0}
FULL_HALF_AXIS=${FULL_HALF_AXIS:-0}

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${ANALYZER}" | awk '{print $1}')" = "${EXPECTED_ANALYZER_SHA256}"
for artifact in "${FT_Y_HALF1}" "${FT_Y_HALF2}" "${FT_CTF_HALF1}" \
  "${FT_CTF_HALF2}" "${REFINEMENT_RESULTS}"; do
  test -s "${artifact}"
done
test ! -e "${OUTPUT_DIR}"

mkdir -p "${OUTPUT_DIR}" "${RUN_ROOT}/logs" "${RUN_ROOT}/provenance" \
  "${RUNTIME_ROOT}/${SLURM_JOB_ID}/tmp" \
  "${RUNTIME_ROOT}/${SLURM_JOB_ID}/pixi_home" \
  "${RUNTIME_ROOT}/${SLURM_JOB_ID}/rattler_cache"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu
export TMPDIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/rattler_cache

cd "${REPO}"
"${PYTHON}" -m scripts.build_k1_pre_reconstruction_tau2 \
  --ft-y-half1 "${FT_Y_HALF1}" --ft-y-half2 "${FT_Y_HALF2}" \
  --ft-ctf-half1 "${FT_CTF_HALF1}" --ft-ctf-half2 "${FT_CTF_HALF2}" \
  --refinement-results "${REFINEMENT_RESULTS}" \
  --iteration-index "${ITERATION_INDEX}" --full-half-axis "${FULL_HALF_AXIS}" \
  --output-dir "${OUTPUT_DIR}" \
  > "${RUN_ROOT}/analysis/pre_reconstruction_tau2_${SLURM_JOB_ID}.stdout"

test -s "${OUTPUT_DIR}/REPORT.json"
sha256sum "${LAUNCHER}" "${ANALYZER}" "${OUTPUT_DIR}/REPORT.json" \
  "${OUTPUT_DIR}/tau2_pre_reconstruction_half1.npy" \
  "${OUTPUT_DIR}/tau2_pre_reconstruction_half2.npy" \
  > "${RUN_ROOT}/provenance/pre_reconstruction_tau2_${SLURM_JOB_ID}.sha256"
touch "${OUTPUT_DIR}/SUCCESS_${SLURM_JOB_ID}"
