#!/usr/bin/env bash
# IgG-1D no-CTF sanity dataset: first 50 PDBs, B80, grid 256, 1M images,
# oracle pipeline/embedding, compute_state for target state 25, GT metrics.

#SBATCH --job-name=igg1d-noctf-n30-b80-1m
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=900G
#SBATCH --time=96:00:00
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/tmp/igg1d_first50_noctf_noise30_b80_1m_20260528/slurm/%x-%j.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/tmp/igg1d_first50_noctf_noise30_b80_1m_20260528/slurm/%x-%j.err

set -euo pipefail

WORKDIR="${WORKDIR:-/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar}"
ROOT="${ROOT:-/scratch/gpfs/CRYOEM/gilleslab/tmp/igg1d_first50_noctf_noise30_b80_1m_20260528}"
RUN_DIR="${RUN_DIR:-$ROOT/n01000000/runs/n01000000_seed0000}"
PDB_DIR="${PDB_DIR:-/home/mg6942/mytigress/cryobench2/IgG-1D/pdbs}"
PDB_GLOB="${PDB_GLOB:-0[0-4][0-9].pdb}"
GRID_SIZE="${GRID_SIZE:-256}"
VOXEL_SIZE="${VOXEL_SIZE:-1.25}"
N_STATES="${N_STATES:-50}"
N_IMAGES="${N_IMAGES:-1000000}"
NOISE_LEVEL="${NOISE_LEVEL:-30}"
RENDER_BFACTOR="${RENDER_BFACTOR:-80}"
TARGET_STATE="${TARGET_STATE:-25}"
ZDIM="${ZDIM:-4}"
AGENT_ID="igg1d-noctf-noise${NOISE_LEVEL}-b${RENDER_BFACTOR}-n${N_IMAGES}-${SLURM_JOB_ID:-manual}"

mkdir -p "$ROOT/slurm"
export TMPDIR="/scratch/gpfs/GILLES/mg6942/tmp/$AGENT_ID"
export PIXI_HOME="/scratch/gpfs/GILLES/mg6942/pixi_home/$AGENT_ID"
export RATTLER_CACHE_DIR="/scratch/gpfs/GILLES/mg6942/rattler_cache/$AGENT_ID"
mkdir -p "$TMPDIR" "$PIXI_HOME" "$RATTLER_CACHE_DIR" "$RUN_DIR"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WORKDIR"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR=cuda_malloc_async
export XDG_CACHE_HOME="$TMPDIR/xdg_cache"
export JAX_COMPILATION_CACHE_DIR="$TMPDIR/jax_cache"
mkdir -p "$XDG_CACHE_HOME" "$JAX_COMPILATION_CACHE_DIR"

cd "$WORKDIR"
module unload cudatoolkit/12.8 >/dev/null 2>&1 || true

echo "START=$(date -Is)"
echo "HOST=$(hostname)"
echo "JOB_ID=${SLURM_JOB_ID:-manual}"
echo "WORKDIR=$WORKDIR"
echo "ROOT=$ROOT"
echo "RUN_DIR=$RUN_DIR"
echo "PDB_DIR=$PDB_DIR"
echo "PDB_GLOB=$PDB_GLOB"
echo "GRID_SIZE=$GRID_SIZE"
echo "VOXEL_SIZE=$VOXEL_SIZE"
echo "N_STATES=$N_STATES"
echo "N_IMAGES=$N_IMAGES"
echo "NOISE_LEVEL=$NOISE_LEVEL"
echo "RENDER_BFACTOR=$RENDER_BFACTOR"
echo "TARGET_STATE=$TARGET_STATE"
echo "ZDIM=$ZDIM"
echo "COMMIT=$(git rev-parse HEAD)"
echo "BRANCH=$(git rev-parse --abbrev-ref HEAD)"
echo "STATUS=$(git status --short | tr '\n' ';')"
nvidia-smi || true

PIXI_PY="$(pixi run which python)"
if [[ ! -x "$PIXI_PY" ]]; then
  echo "Could not find pixi python" >&2
  exit 2
fi

if [[ ! -f recovar/cuda/libcuda_backproject.so ]]; then
  module load cudatoolkit/12.8 || true
  PYTHON="$PIXI_PY" make -C recovar/cuda clean all
  module unload cudatoolkit/12.8 || true
fi

"$PIXI_PY" - <<'PY'
import pathlib
import jax
import recovar

repo = pathlib.Path.cwd().resolve()
recovar_file = pathlib.Path(recovar.__file__).resolve()
jax_file = pathlib.Path(jax.__file__).resolve()
print("recovar.__file__=", recovar_file)
print("jax.__file__=", jax_file)
print("jax.devices()=", jax.devices())
assert str(recovar_file).startswith(str(repo) + "/")
assert ".pixi/envs/default/" in str(jax_file)
if not any(getattr(device, "platform", "") == "gpu" for device in jax.devices()):
    raise RuntimeError("JAX did not see a GPU")
PY

/usr/bin/time -v "$PIXI_PY" scripts/experiments/igg1d_oracle/run_igg1d_noctf_walkthrough.py \
  --output-dir "$RUN_DIR" \
  --pdb-dir "$PDB_DIR" \
  --render-glob "$PDB_GLOB" \
  --grid-size "$GRID_SIZE" \
  --render-voxel-size "$VOXEL_SIZE" \
  --voxel-size "$VOXEL_SIZE" \
  --render-bfactor "$RENDER_BFACTOR" \
  --n-states "$N_STATES" \
  --n-images "$N_IMAGES" \
  --noise-level "$NOISE_LEVEL" \
  --bfactor 0 \
  --zdim "$ZDIM" \
  --target-state "$TARGET_STATE" \
  --seed 0 \
  --lazy \
  --compute-state-save-all-estimates \
  --use-oracle-pipeline \
  --no-kernel-report

/usr/bin/time -v "$PIXI_PY" scripts/experiments/igg1d_oracle/plot_igg1d_noctf_oracle_metrics.py \
  --run-dir "$RUN_DIR" \
  --compute-dir "$RUN_DIR/07_compute_state" \
  --mask "$RUN_DIR/05_masks/focus_mask_moving.mrc" \
  --out-dir "$RUN_DIR/08_noctf_oracle_gt_weighted_report"

echo "END=$(date -Is)"
echo "RUN_DIR=$RUN_DIR"
echo "PIPELINE_DIR=$RUN_DIR/06_pipeline"
echo "COMPUTE_STATE_DIR=$RUN_DIR/07_compute_state"
echo "REPORT_DIR=$RUN_DIR/08_noctf_oracle_gt_weighted_report"
