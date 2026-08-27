#!/usr/bin/env bash
#SBATCH --job-name=k1-ref-write
#SBATCH --partition=cpu
#SBATCH --account=gilles
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=01:00:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${RECOVAR_ACCUMULATOR:?RECOVAR_ACCUMULATOR is required}"
: "${RECOVAR_INTERMEDIATES:?RECOVAR_INTERMEDIATES is required}"
: "${NATIVE_RUN_ROOT:?NATIVE_RUN_ROOT is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_ANALYZER_SHA256:?EXPECTED_ANALYZER_SHA256 is required}"

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_native_units_instrumentation_20260826
PYTHON=${REPO}/.pixi/envs/default/bin/python
LAUNCHER=${REPO}/scripts/analyze_k1_recovar_reference_write_boundary.sbatch
ANALYZER=${REPO}/scripts/analyze_k1_recovar_reference_write_boundary.py
NATIVE_STAGES=${NATIVE_RUN_ROOT}/reconstruct
NATIVE_RELION=${NATIVE_RUN_ROOT}/relion
ANALYSIS=${RUN_ROOT}/analysis

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -f "${NATIVE_RUN_ROOT}/SAFE_TO_DELETE"
test -s "${RECOVAR_ACCUMULATOR}" -a -s "${RECOVAR_INTERMEDIATES}/it000_fsc.npy"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${ANALYZER}" | awk '{print $1}')" = "${EXPECTED_ANALYZER_SHA256}"
test -s "${NATIVE_RUN_ROOT}/analysis/RECONSTRUCT_CAPTURE_VALIDATION.json"
for half in 1 2; do
    rank=$(printf '%02d' "${half}")
    test "$(find "${NATIVE_STAGES}" -maxdepth 1 -name "reconstruct_rank${rank}_*_call0000_volume_after_gridding.bin" | wc -l)" -eq 1
    test -s "${NATIVE_RELION}/run_it001_half${half}_class001.mrc"
    test -s "${RECOVAR_INTERMEDIATES}/it000_half${half}_reg.mrc"
done

mkdir -p "${ANALYSIS}" "${RUN_ROOT}/provenance" \
    "${RUNTIME_ROOT}/tmp" "${RUNTIME_ROOT}/pixi_home" \
    "${RUNTIME_ROOT}/rattler_cache" "${RUNTIME_ROOT}/pycache"
test -z "$(find "${ANALYSIS}" -mindepth 1 -print -quit)"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=${RUNTIME_ROOT}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/rattler_cache
export PYTHONPYCACHEPREFIX=${RUNTIME_ROOT}/pycache

cd "${REPO}"
"${PYTHON}" - <<'PY'
from pathlib import Path
import jax
import recovar

repo = Path.cwd().resolve()
assert Path(recovar.__file__).resolve().is_relative_to(repo)
assert Path(jax.__file__).resolve().is_relative_to(repo / ".pixi/envs/default")
PY

"${PYTHON}" -m scripts.analyze_k1_recovar_reference_write_boundary \
    --recovar-accumulator "${RECOVAR_ACCUMULATOR}" \
    --recovar-fsc "${RECOVAR_INTERMEDIATES}/it000_fsc.npy" \
    --recovar-map-dir "${RECOVAR_INTERMEDIATES}" \
    --native-stage-dir "${NATIVE_STAGES}" \
    --native-relion-dir "${NATIVE_RELION}" \
    --voxel-size 2.125 \
    --ini-high-angstrom 30 \
    --fourier-mask-edge 2 \
    --particle-diameter-angstrom 200 \
    --solvent-mask-edge-pixels 5 \
    --output-json "${ANALYSIS}/K1_RECOVAR_REFERENCE_WRITE_BOUNDARY.json" \
    > "${ANALYSIS}/K1_RECOVAR_REFERENCE_WRITE_BOUNDARY.stdout"

sha256sum "${LAUNCHER}" "${ANALYZER}" "${RECOVAR_ACCUMULATOR}" \
    "${ANALYSIS}/K1_RECOVAR_REFERENCE_WRITE_BOUNDARY.json" \
    "${RECOVAR_INTERMEDIATES}/it000_fsc.npy" \
    "${RECOVAR_INTERMEDIATES}/it000_half1_reg.mrc" \
    "${RECOVAR_INTERMEDIATES}/it000_half2_reg.mrc" \
    > "${RUN_ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
find "${NATIVE_STAGES}" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum \
    > "${RUN_ROOT}/provenance/native_inputs_${SLURM_JOB_ID}.sha256"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
