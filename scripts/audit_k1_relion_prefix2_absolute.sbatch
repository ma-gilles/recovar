#!/usr/bin/env bash
#SBATCH --job-name=k1-prefix2-audit
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
: "${EXPECTED_REPO_HEAD:?EXPECTED_REPO_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"

REPO=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_test_gpu_binding_v2_20260820
PYTHON=${REPO}/.pixi/envs/default/bin/python
LAUNCHER=${REPO}/scripts/audit_k1_relion_prefix2_absolute.sbatch
OUTPUT=${RUN_ROOT}/output
PARITY=${RUN_ROOT}/parity
ANALYSIS=${RUN_ROOT}/analysis/prefix_it002
DATA=${CASE_SOURCE}/data
RELION=${CASE_SOURCE}/relion_ref

test -f "${RUN_ROOT}/SAFE_TO_DELETE" -a -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -s "${OUTPUT}/refinement_results.npz"
test -s "${OUTPUT}/intermediates/it000_half1_reg.mrc"
test -s "${OUTPUT}/intermediates/it000_half2_reg.mrc"
test -s "${OUTPUT}/intermediates/it001_half1_reg.mrc"
test -s "${OUTPUT}/intermediates/it001_half2_reg.mrc"
test -s "${PARITY}/iter_001.npz" -a -s "${PARITY}/iter_002.npz"
test -s "${DATA}/particles.star" -a -s "${DATA}/reference_gt.mrc"
test -s "${RELION}/run_it001_data.star" -a -s "${RELION}/run_it002_data.star"
test "$(git -C "${REPO}" rev-parse HEAD)" = "${EXPECTED_REPO_HEAD}"
test "$(sha256sum "${LAUNCHER}" | awk '{print $1}')" = "${EXPECTED_LAUNCHER_SHA256}"

mkdir -p "${ANALYSIS}" "${RUN_ROOT}/provenance"
mkdir -p "${RUNTIME_ROOT}/audit_tmp" "${RUNTIME_ROOT}/audit_pixi_home" \
    "${RUNTIME_ROOT}/audit_rattler_cache"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1
export TMPDIR=${RUNTIME_ROOT}/audit_tmp
export PIXI_HOME=${RUNTIME_ROOT}/audit_pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/audit_rattler_cache

cd "${REPO}"
"${PYTHON}" - <<'PY'
from pathlib import Path
import jax
import recovar

repo = Path.cwd().resolve()
assert Path(recovar.__file__).resolve().is_relative_to(repo)
assert Path(jax.__file__).resolve().is_relative_to(repo / ".pixi/envs/default")
PY

"${PYTHON}" -m scripts.audit_k1_fsc_trajectory \
    --case-root "${CASE_SOURCE}" \
    --recovar-dir "${OUTPUT}" \
    --relion-dir "${RELION}" \
    --gt-volume "${DATA}/reference_gt.mrc" \
    --numbered-only --allow-incomplete \
    --output-json "${ANALYSIS}/fsc.json" \
    --output-markdown "${ANALYSIS}/fsc.md" \
    --output-shellwise-npz "${ANALYSIS}/fsc_shellwise.npz"

"${PYTHON}" -m scripts.audit_em_particle_state_distribution \
    --recovar-results "${OUTPUT}/refinement_results.npz" \
    --recovar-particles-star "${DATA}/particles.star" \
    --relion-star "1:${RELION}/run_it001_data.star" \
    --relion-star "2:${RELION}/run_it002_data.star" \
    --recovar-iteration 0 --recovar-iteration 1 --relion-iteration-offset 1 \
    --output-json "${ANALYSIS}/particle_state.json" \
    --output-npz "${ANALYSIS}/particle_state_arrays.npz" \
    --output-hash-manifest "${ANALYSIS}/particle_state.sha256"

sha256sum "${LAUNCHER}" "${OUTPUT}/refinement_results.npz" \
    "${PARITY}/iter_001.npz" "${PARITY}/iter_002.npz" \
    "${ANALYSIS}/fsc.json" "${ANALYSIS}/fsc_shellwise.npz" \
    "${ANALYSIS}/particle_state.json" \
    "${ANALYSIS}/particle_state_arrays.npz" \
    > "${RUN_ROOT}/provenance/prefix2_absolute_audit_${SLURM_JOB_ID}.sha256"
touch "${RUN_ROOT}/PREFIX2_AUDIT_SUCCESS_${SLURM_JOB_ID}"
