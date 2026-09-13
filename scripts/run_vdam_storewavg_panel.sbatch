#!/usr/bin/env bash
#SBATCH --job-name=vdam-wavg-panel
#SBATCH --account=amits
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-wavg-panel-%j.out

set -euo pipefail

: "${NATIVE_DIRECTORY:?submit with NATIVE_DIRECTORY=/scratch/...}"
: "${SCORE_DIRECTORY:?submit with SCORE_DIRECTORY=/scratch/...}"
: "${RELION_DATA_STAR:?submit with RELION_DATA_STAR=/scratch/...star}"
: "${OUTPUT_JSON:?submit with OUTPUT_JSON=/scratch/...json}"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ID="${SLURM_JOB_ID:-manual}"
cd "${REPO_ROOT}"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV JAX_PLATFORM_NAME
export PYTHONNOUSERSITE=1
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/${RUN_ID}/tmp"
export PIXI_HOME="/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/${RUN_ID}/pixi_home"
export RATTLER_CACHE_DIR="/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/${RUN_ID}/rattler_cache"
mkdir -p "${TMPDIR}" "${PIXI_HOME}" "${RATTLER_CACHE_DIR}" /scratch/gpfs/GILLES/mg6942/slurmo

PIXI_PY="${REPO_ROOT}/.pixi/envs/default/bin/python"
"${PIXI_PY}" - <<'PY'
import pathlib

import jax
import recovar

repo = pathlib.Path.cwd().resolve()
assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo) + "/")
assert ".pixi/envs/default/" in str(pathlib.Path(jax.__file__).resolve())
assert jax.default_backend() == "gpu"
print("VDAM StoreWavg panel provenance/GPU gate passed", jax.devices())
PY

"${PIXI_PY}" -m scripts.analyze_vdam_storewavg_panel \
  --native-directory "${NATIVE_DIRECTORY}" \
  --score-directory "${SCORE_DIRECTORY}" \
  --relion-data-star "${RELION_DATA_STAR}" \
  --output "${OUTPUT_JSON}" \
  --rotation-tolerance "${ROTATION_TOLERANCE:-1e-5}"
