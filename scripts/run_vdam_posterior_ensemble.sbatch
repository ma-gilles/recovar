#!/usr/bin/env bash
#SBATCH --job-name=vdam-post-ensemble
#SBATCH --account=amits
#SBATCH --partition=cryoem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-post-ensemble-%j.out

set -euo pipefail

: "${NATIVE_PANEL_ROOT:?set the root containing repeat-NN/capture directories}"
: "${NATIVE_REPEAT_COUNT:?set the native repeat count}"
: "${RELION_DATA_STAR:?set the matching RELION data STAR}"
: "${RECOVAR_HALF1_DIRECTORY:?set the half-1 contribution directory}"
: "${RECOVAR_HALF2_DIRECTORY:?set the half-2 contribution directory}"
: "${OUTPUT_JSON:?set a new output JSON path}"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ID="${SLURM_JOB_ID:-manual}"
cd "${REPO_ROOT}"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV JAX_PLATFORM_NAME JAX_PLATFORMS
export PYTHONNOUSERSITE=1
export TMPDIR="/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/${RUN_ID}/tmp"
export PIXI_HOME="/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/${RUN_ID}/pixi_home"
export RATTLER_CACHE_DIR="/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/${RUN_ID}/rattler_cache"
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" /scratch/gpfs/GILLES/mg6942/slurmo

PIXI_PY="${REPO_ROOT}/.pixi/envs/default/bin/python"
mapfile -t HALF1_CAPTURES < <(find "${RECOVAR_HALF1_DIRECTORY}" -maxdepth 1 -type f -name '*.npz' | sort)
mapfile -t HALF2_CAPTURES < <(find "${RECOVAR_HALF2_DIRECTORY}" -maxdepth 1 -type f -name '*.npz' | sort)
test "${#HALF1_CAPTURES[@]}" -eq 2
test "${#HALF2_CAPTURES[@]}" -eq 2
test "${NATIVE_REPEAT_COUNT}" -ge 2

git rev-parse HEAD
git symbolic-ref --short HEAD || echo '<detached>'
git status --short --branch
git diff HEAD --stat
git diff HEAD | sha256sum

"${PIXI_PY}" -m pytest -v tests/unit/initial_model/test_vdam_posterior_ensemble.py

ARGS=(
  --relion-data-star "${RELION_DATA_STAR}"
  --output-json "${OUTPUT_JSON}"
  --rotation-tolerance "${ROTATION_TOLERANCE:-1e-5}"
)
for repeat_index in $(seq 1 "${NATIVE_REPEAT_COUNT}"); do
  capture="${NATIVE_PANEL_ROOT}/repeat-$(printf '%02d' "${repeat_index}")/capture"
  test -d "${capture}"
  ARGS+=(--native-directory "${capture}")
done
for path in "${HALF1_CAPTURES[@]}"; do ARGS+=(--recovar-half1-capture "${path}"); done
for path in "${HALF2_CAPTURES[@]}"; do ARGS+=(--recovar-half2-capture "${path}"); done
"${PIXI_PY}" -m scripts.analyze_vdam_posterior_ensemble "${ARGS[@]}"
