#!/usr/bin/env bash
#SBATCH --job-name=k1-c10-native-boundary
#SBATCH --partition=cpu
#SBATCH --account=gilles
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=01:00:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${NATIVE_VERBOSE_ROOT:?NATIVE_VERBOSE_ROOT is required}"
: "${NATIVE_PACKAGED_ROOT:?NATIVE_PACKAGED_ROOT is required}"
: "${RECOVAR_ROOT:?RECOVAR_ROOT is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
PYTHON=${REPO}/.pixi/envs/default/bin/python
LAUNCHER=${REPO}/scripts/analyze_k1_case10_native_verbose_same_input.sbatch
ANALYSIS=${RUN_ROOT}/analysis

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -f "${NATIVE_VERBOSE_ROOT}/SAFE_TO_DELETE"
test -f "${NATIVE_PACKAGED_ROOT}/SAFE_TO_DELETE"
test -f "${RECOVAR_ROOT}/SAFE_TO_DELETE"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"

mapfile -t native_factor < <(find "${NATIVE_PACKAGED_ROOT}/capture" -maxdepth 1 \
    -type f -name '*.bpre-v2.bin' -print)
mapfile -t native_fine < <(find "${NATIVE_PACKAGED_ROOT}/capture" -maxdepth 1 \
    -type f -name '*.fine-score-v1.bin' -print)
mapfile -t recovar_capture < <(find "${RECOVAR_ROOT}/pass2" -maxdepth 1 \
    -type f -name 'pass2_orig*_cs058.npz' -print)
test "${#native_factor[@]}" -eq 1
test "${#native_fine[@]}" -eq 1
test "${#recovar_capture[@]}" -eq 1

mkdir -p "${ANALYSIS}" "${RUN_ROOT}/provenance"
mkdir -p "${RUNTIME_ROOT}/tmp" "${RUNTIME_ROOT}/pixi_home" \
    "${RUNTIME_ROOT}/rattler_cache" "${RUNTIME_ROOT}/pycache"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1
export TMPDIR=${RUNTIME_ROOT}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/rattler_cache
export PYTHONPYCACHEPREFIX=${RUNTIME_ROOT}/pycache

output=${ANALYSIS}/K1_CASE10_ROW68955_NATIVE_VS_SAME_INPUT_FINE_OPERANDS.json
test ! -e "${output}"
cd "${REPO}"
"${PYTHON}" -m scripts.analyze_k1_native_fine_operand_boundary \
    --native-verbose-dir "${NATIVE_VERBOSE_ROOT}/capture" \
    --native-factor "${native_factor[0]}" \
    --native-fine-score "${native_fine[0]}" \
    --recovar-capture "${recovar_capture[0]}" \
    --physical-image-size 384 \
    --output-json "${output}" \
    > "${RUN_ROOT}/analysis/stdout_${SLURM_JOB_ID}.log"

sha256sum "${LAUNCHER}" "${native_factor[0]}" "${native_fine[0]}" \
    "${recovar_capture[0]}" "${output}" \
    > "${RUN_ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
