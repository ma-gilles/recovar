#!/usr/bin/env bash
#SBATCH --job-name=k1-final-manifest-oracle
#SBATCH --partition=cpu
#SBATCH --account=gilles
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail

: "${REPO:?REPO is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${CURRENT_HALF1:?CURRENT_HALF1 is required}"
: "${CURRENT_HALF2:?CURRENT_HALF2 is required}"
: "${ORACLE_HALF1:?ORACLE_HALF1 is required}"
: "${ORACLE_HALF2:?ORACLE_HALF2 is required}"
: "${PARTICLE_STAR:?PARTICLE_STAR is required}"
: "${RELION_HALFSET_STAR:?RELION_HALFSET_STAR is required}"
: "${FRESH_ORDER_SEED:?FRESH_ORDER_SEED is required}"
: "${RELION_BIND_BUILD_DIR:?RELION_BIND_BUILD_DIR is required}"
: "${EXPECTED_HEAD:?EXPECTED_HEAD is required}"
: "${EXPECTED_LAUNCHER_SHA256:?EXPECTED_LAUNCHER_SHA256 is required}"
: "${EXPECTED_ANALYZER_SHA256:?EXPECTED_ANALYZER_SHA256 is required}"

PYTHON=${REPO}/.pixi/envs/default/bin/python
LAUNCHER=${REPO}/scripts/audit_k1_final_manifest_against_oracle.sbatch
ANALYZER=${REPO}/scripts/analyze_k1_final_manifest_ab.py
ANALYSIS=${RUN_ROOT}/analysis_final_manifest_oracle

test -f "${RUN_ROOT}/SAFE_TO_DELETE"
test -f "${RUNTIME_ROOT}/SAFE_TO_DELETE"
test -x "${PYTHON}"
test -s "${CURRENT_HALF1}" -a -s "${CURRENT_HALF2}"
test -s "${ORACLE_HALF1}" -a -s "${ORACLE_HALF2}"
test -s "${PARTICLE_STAR}" -a -s "${RELION_HALFSET_STAR}"
test -d "${RELION_BIND_BUILD_DIR}"
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
export RECOVAR_RELION_BIND_BUILD_DIR=${RELION_BIND_BUILD_DIR}
export TMPDIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/tmp
export PIXI_HOME=${RUNTIME_ROOT}/${SLURM_JOB_ID}/pixi_home
export RATTLER_CACHE_DIR=${RUNTIME_ROOT}/${SLURM_JOB_ID}/rattler_cache

cd "${REPO}"
"${PYTHON}" - <<'PY'
from pathlib import Path

import jax
import recovar

repo = Path.cwd().resolve()
recovar_path = Path(recovar.__file__).resolve()
jax_path = Path(jax.__file__).resolve()
expected_jax_root = (repo / ".pixi" / "envs" / "default").resolve()
if repo not in recovar_path.parents:
    raise SystemExit(f"RECOVAR provenance failure: {recovar_path} is outside {repo}")
if expected_jax_root not in jax_path.parents:
    raise SystemExit(f"JAX provenance failure: {jax_path} is outside {expected_jax_root}")
print(f"recovar={recovar_path}")
print(f"jax={jax_path}")
PY

"${PYTHON}" -m scripts.analyze_k1_final_manifest_ab \
  --control-half1 "${CURRENT_HALF1}" \
  --control-half2 "${CURRENT_HALF2}" \
  --candidate-half1 "${ORACLE_HALF1}" \
  --candidate-half2 "${ORACLE_HALF2}" \
  --candidate-particle-star "${PARTICLE_STAR}" \
  --relion-halfset-star "${RELION_HALFSET_STAR}" \
  --fresh-order-seed "${FRESH_ORDER_SEED}" \
  --output-json "${ANALYSIS}/manifest_ab.json"

sha256sum "${LAUNCHER}" "${ANALYZER}" \
  "${CURRENT_HALF1}" "${CURRENT_HALF2}" \
  "${ORACLE_HALF1}" "${ORACLE_HALF2}" \
  "${PARTICLE_STAR}" "${RELION_HALFSET_STAR}" \
  "${ANALYSIS}/manifest_ab.json" \
  > "${RUN_ROOT}/provenance/final_manifest_oracle_${SLURM_JOB_ID}.sha256"
touch "${ANALYSIS}/SUCCESS_${SLURM_JOB_ID}"
