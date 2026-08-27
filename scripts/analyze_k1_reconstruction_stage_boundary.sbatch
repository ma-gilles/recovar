#!/usr/bin/env bash
#SBATCH --job-name=k1-recon-boundary
#SBATCH --partition=cpu
#SBATCH --account=gilles
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=01:00:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${NATIVE_RUN_ROOT:?NATIVE_RUN_ROOT is required}"
: "${RECOVAR_RUN_ROOT:?RECOVAR_RUN_ROOT is required}"
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_ANALYZER_SHA256:?EXPECTED_ANALYZER_SHA256 is required}"

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_native_units_instrumentation_20260826
PYTHON=${REPO}/.pixi/envs/default/bin/python
LAUNCHER=${REPO}/scripts/analyze_k1_reconstruction_stage_boundary.sbatch
ANALYZER=${REPO}/scripts/analyze_k1_reconstruction_stage_boundary.py
INTERMEDIATES=${RECOVAR_RUN_ROOT}/recovar/output/intermediates
NATIVE_STAGES=${NATIVE_RUN_ROOT}/reconstruct
ANALYSIS=${RUN_ROOT}/analysis
ACCUMULATOR=${RUN_ROOT}/recovar_accumulator.npz

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -f "${NATIVE_RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RECOVAR_RUN_ROOT}/SAFE_TO_DELETE"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"
test "$(sha256sum "${ANALYZER}" | awk '{print $1}')" = "${EXPECTED_ANALYZER_SHA256}"
test "$(find "${RECOVAR_RUN_ROOT}" -maxdepth 1 -name 'SUCCESS_*' | wc -l)" -eq 1
test -s "${NATIVE_RUN_ROOT}/analysis/RECONSTRUCT_CAPTURE_VALIDATION.json"
for name in Ft_y_0 Ft_y_1 Ft_ctf_0 Ft_ctf_1 fsc; do
    test -s "${INTERMEDIATES}/it000_${name}.npy"
done
for stage in tau2 fweight_decentered fweight_regularized radavg_weight \
    fconv_divided volume_before_gridding volume_after_gridding; do
    test "$(find "${NATIVE_STAGES}" -maxdepth 1 -name "reconstruct_rank*_call0000_${stage}.bin" | wc -l)" -eq 2
done
"${PYTHON}" - "${NATIVE_RUN_ROOT}/analysis/RECONSTRUCT_CAPTURE_VALIDATION.json" <<'PY'
import json
import sys

records = json.load(open(sys.argv[1]))["records"]
assert len(records) == 14
assert len({record["name"] for record in records}) == 14
PY

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

"${PYTHON}" - "${INTERMEDIATES}" "${ACCUMULATOR}" <<'PY'
import sys
from pathlib import Path

import numpy as np

source = Path(sys.argv[1])
output = Path(sys.argv[2])
np.savez(
    output,
    grid_size=np.int64(256),
    current_size=np.int64(56),
    padding_factor=np.int64(2),
    volume_shape=np.asarray((256, 256, 256), dtype=np.int64),
    mstep_accumulator_shape=np.asarray((115, 115, 115), dtype=np.int64),
    Ft_y_0=np.load(source / "it000_Ft_y_0.npy", allow_pickle=False),
    Ft_y_1=np.load(source / "it000_Ft_y_1.npy", allow_pickle=False),
    Ft_ctf_0=np.load(source / "it000_Ft_ctf_0.npy", allow_pickle=False),
    Ft_ctf_1=np.load(source / "it000_Ft_ctf_1.npy", allow_pickle=False),
)
PY

"${PYTHON}" -m scripts.analyze_k1_reconstruction_stage_boundary \
    --recovar-accumulator "${ACCUMULATOR}" \
    --native-stage-dir "${NATIVE_STAGES}" \
    --native-call-index 0 \
    --output-json "${ANALYSIS}/K1_RECONSTRUCTION_STAGE_BOUNDARY.json" \
    > "${ANALYSIS}/K1_RECONSTRUCTION_STAGE_BOUNDARY.stdout"

sha256sum "${LAUNCHER}" "${ANALYZER}" "${ACCUMULATOR}" \
    "${ANALYSIS}/K1_RECONSTRUCTION_STAGE_BOUNDARY.json" \
    "${INTERMEDIATES}"/it000_{Ft_y_0,Ft_y_1,Ft_ctf_0,Ft_ctf_1,fsc}.npy \
    > "${RUN_ROOT}/provenance/results_${SLURM_JOB_ID}.sha256"
find "${NATIVE_STAGES}" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum \
    > "${RUN_ROOT}/provenance/native_inputs_${SLURM_JOB_ID}.sha256"
touch "${RUN_ROOT}/SUCCESS_${SLURM_JOB_ID}"
