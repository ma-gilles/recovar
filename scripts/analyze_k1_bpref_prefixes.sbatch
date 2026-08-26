#!/usr/bin/env bash
#SBATCH --job-name=k1-bpref-prefix-audit
#SBATCH --partition=cpu
#SBATCH --account=gilles
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=01:00:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${NATIVE_RUN_ROOT:?NATIVE_RUN_ROOT is required}"
: "${RECOVAR_HALF1_RUN_ROOT:?RECOVAR_HALF1_RUN_ROOT is required}"
: "${RECOVAR_HALF2_RUN_ROOT:?RECOVAR_HALF2_RUN_ROOT is required}"
: "${PREFIX_SELECTION_JSON:?PREFIX_SELECTION_JSON is required}"
: "${EXPECTED_SELECTION_SHA256:?EXPECTED_SELECTION_SHA256 is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
PYTHON=${REPO}/.pixi/envs/default/bin/python
LAUNCHER=${REPO}/scripts/analyze_k1_bpref_prefixes.sbatch
NATIVE=${NATIVE_RUN_ROOT}/native_prefix
RECOVAR_HALF1=${RECOVAR_HALF1_RUN_ROOT}/recovar/accumulator_delta
RECOVAR_HALF2=${RECOVAR_HALF2_RUN_ROOT}/recovar/accumulator_delta
STAGED_RECOVAR=${RUN_ROOT}/recovar_prefix
ANALYSIS=${RUN_ROOT}/analysis

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -f "${NATIVE_RUN_ROOT}/SAFE_TO_DELETE"
test -f "${RECOVAR_HALF1_RUN_ROOT}/SAFE_TO_DELETE"
test -f "${RECOVAR_HALF2_RUN_ROOT}/SAFE_TO_DELETE"
test -s "${PREFIX_SELECTION_JSON}"
test "$(sha256sum "${PREFIX_SELECTION_JSON}" | awk '{print $1}')" = "${EXPECTED_SELECTION_SHA256}"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(find "${NATIVE}" -maxdepth 1 -name '*.bin' | wc -l)" -eq 80
test "$(find "${RECOVAR_HALF1}" -maxdepth 1 -name 'bpref_accumulator_delta_it001_h1_*.npz' | wc -l)" -eq 10
test "$(find "${RECOVAR_HALF2}" -maxdepth 1 -name 'bpref_accumulator_delta_it001_h2_*.npz' | wc -l)" -eq 10

mkdir -p "${STAGED_RECOVAR}" "${ANALYSIS}" "${RUN_ROOT}/provenance" \
    "${RUNTIME_ROOT}/tmp" "${RUNTIME_ROOT}/pixi_home" \
    "${RUNTIME_ROOT}/rattler_cache" "${RUNTIME_ROOT}/pycache"
test -z "$(find "${STAGED_RECOVAR}" "${ANALYSIS}" -mindepth 1 -print -quit)"
find "${RECOVAR_HALF1}" "${RECOVAR_HALF2}" -maxdepth 1 \
    -name 'bpref_accumulator_delta_it001_h*.npz' -print0 \
    | while IFS= read -r -d '' path; do ln -s "${path}" "${STAGED_RECOVAR}/$(basename "${path}")"; done
test "$(find "${STAGED_RECOVAR}" -maxdepth 1 -type l | wc -l)" -eq 20

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1
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

"${PYTHON}" -m scripts.analyze_k1_bpref_prefixes \
    --selection "${PREFIX_SELECTION_JSON}" \
    --native-directory "${NATIVE}" \
    --recovar-directory "${STAGED_RECOVAR}" \
    --output-json "${ANALYSIS}/K1_BPREF_PREFIXES.json" \
    --output-markdown "${ANALYSIS}/K1_BPREF_PREFIXES.md"

sha256sum "${LAUNCHER}" "${PREFIX_SELECTION_JSON}" \
    "${ANALYSIS}/K1_BPREF_PREFIXES.json" \
    "${ANALYSIS}/K1_BPREF_PREFIXES.md" \
    > "${RUN_ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
find "${NATIVE}" -maxdepth 1 -name '*.bin' -print0 | sort -z | xargs -0 sha256sum \
    > "${RUN_ROOT}/provenance/native_inputs_${SLURM_JOB_ID}.sha256"
find -L "${STAGED_RECOVAR}" -maxdepth 1 -name '*.npz' -print0 | sort -z | xargs -0 sha256sum \
    > "${RUN_ROOT}/provenance/recovar_inputs_${SLURM_JOB_ID}.sha256"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
