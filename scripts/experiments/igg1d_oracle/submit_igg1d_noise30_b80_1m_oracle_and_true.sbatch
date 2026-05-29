#!/usr/bin/env bash
# IgG-1D 50-state synthetic experiment:
#   task 0: render first 50 PDBs, simulate 1M images, oracle pipeline, compute_state, plot metrics
#   task 1: wait for the dataset/masks, run true RECOVAR pipeline + compute_state in separate dirs

#SBATCH --job-name=igg1d-n30-b80-1m
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=900G
#SBATCH --time=96:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/%x-%A_%a.out
#SBATCH --error=/scratch/gpfs/GILLES/mg6942/slurmo/%x-%A_%a.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd -P)}"
WORKDIR="${WORKDIR:-$SUBMIT_DIR}"
ROOT="${ROOT:-/scratch/gpfs/CRYOEM/gilleslab/tmp/igg1d_first50_noise30_b80_1m_20260528}"
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

AGENT_ID="igg1d-n30-b80-1m-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export TMPDIR="${TMPDIR:-/scratch/gpfs/GILLES/mg6942/tmp/$AGENT_ID}"
export PIXI_HOME="${PIXI_HOME:-/scratch/gpfs/GILLES/mg6942/pixi_home/$AGENT_ID}"
export RATTLER_CACHE_DIR="${RATTLER_CACHE_DIR:-/scratch/gpfs/GILLES/mg6942/rattler_cache/$AGENT_ID}"
mkdir -p "$TMPDIR" "$PIXI_HOME" "$RATTLER_CACHE_DIR" "$RUN_DIR" /scratch/gpfs/GILLES/mg6942/slurmo

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR=cuda_malloc_async
export XDG_CACHE_HOME="$TMPDIR/xdg_cache"
export JAX_COMPILATION_CACHE_DIR="$TMPDIR/jax_cache"
mkdir -p "$XDG_CACHE_HOME" "$JAX_COMPILATION_CACHE_DIR"

cd "$WORKDIR"
module unload cudatoolkit/12.8 >/dev/null 2>&1 || true

echo "START=$(date -Is)"
echo "TASK=${SLURM_ARRAY_TASK_ID}"
echo "HOST=$(hostname)"
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
echo "WORKDIR_COMMIT=$(git rev-parse HEAD)"
echo "WORKDIR_BRANCH=$(git rev-parse --abbrev-ref HEAD)"
echo "WORKDIR_STATUS=$(git status --short | tr '\n' ';')"
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
print("recovar", recovar.__file__)
print("jax", jax.__file__)
print("devices", jax.devices())
assert pathlib.Path(recovar.__file__).resolve().is_relative_to(repo)
assert ".pixi/envs/default" in str(pathlib.Path(jax.__file__).resolve())
assert any(getattr(d, "platform", "") == "gpu" for d in jax.devices())
PY

wait_for_file() {
  local path="$1"
  local label="$2"
  for _ in $(seq 1 5760); do
    if [[ -e "$path" ]]; then
      echo "Found $label: $path"
      return 0
    fi
    echo "Waiting for $label: $path at $(date -Is)"
    sleep 30
  done
  echo "Timed out waiting for $label: $path" >&2
  return 1
}

run_oracle() {
  /usr/bin/time -v "$PIXI_PY" -m recovar.commands.spike_walkthrough \
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

  /usr/bin/time -v "$PIXI_PY" scripts/experiments/igg1d_oracle/plot_igg1d_oracle_metrics.py \
    --run-dir "$RUN_DIR" \
    --compute-dir "$RUN_DIR/07_compute_state" \
    --mask "$RUN_DIR/05_masks/focus_mask_moving.mrc" \
    --out-dir "$RUN_DIR/08_oracle_gt_weighted_report"
}

run_true_pipeline() {
  wait_for_file "$RUN_DIR/03_dataset/simulation_info.pkl" "simulation_info"
  wait_for_file "$RUN_DIR/03_dataset/particles.star" "particles.star"
  wait_for_file "$RUN_DIR/05_masks/focus_mask_moving.mrc" "focus mask"

  local true_pipeline="$RUN_DIR/06_pipeline_true"
  local true_compute="$RUN_DIR/07_compute_state_true"
  local target_point="$RUN_DIR/target_latent_point_true_pipeline_state0025.txt"

  if [[ ! -f "$true_pipeline/model/params.pkl" ]]; then
    /usr/bin/time -v "$PIXI_PY" -m recovar.commands.pipeline \
      "$RUN_DIR/03_dataset/particles.star" \
      --poses "$RUN_DIR/03_dataset/poses.pkl" \
      --ctf "$RUN_DIR/03_dataset/ctf.pkl" \
      -o "$true_pipeline" \
      --mask "$RUN_DIR/05_masks/volume_mask_union.mrc" \
      --focus-mask "$RUN_DIR/05_masks/focus_mask_moving.mrc" \
      --no-correct-contrast \
      --noise-model radial \
      --zdim "$ZDIM" \
      --lazy
  else
    echo "Reusing true pipeline output: $true_pipeline"
  fi

  "$PIXI_PY" - "$RUN_DIR" "$true_pipeline" "$target_point" "$TARGET_STATE" "$ZDIM" <<'PY'
import pickle
import sys
from pathlib import Path

import numpy as np

from recovar.output import output as o

run_dir = Path(sys.argv[1])
pipeline_dir = Path(sys.argv[2])
target_path = Path(sys.argv[3])
target_state = int(sys.argv[4])
zdim = int(sys.argv[5])

po = o.PipelineOutput(str(pipeline_dir))
latent = po.get("latent_coords")
zs = np.asarray(latent[zdim] if zdim in latent else latent[f"zdim_{zdim}"], dtype=np.float32)
with (run_dir / "03_dataset" / "simulation_info.pkl").open("rb") as handle:
    sim_info = pickle.load(handle)
states = np.asarray(sim_info["image_assignment"], dtype=np.int64).reshape(-1)
keep = states == target_state
if not np.any(keep):
    raise RuntimeError(f"No particles assigned to state {target_state}")
target = np.nanmean(zs[keep], axis=0, dtype=np.float64).astype(np.float32)
target_path.parent.mkdir(parents=True, exist_ok=True)
np.savetxt(target_path, target.reshape(1, -1))
print("wrote", target_path, "target", target)
PY

  if [[ ! -f "$true_compute/state000_unfil.mrc" ]]; then
    /usr/bin/time -v "$PIXI_PY" -m recovar.commands.compute_state \
      "$true_pipeline" \
      -o "$true_compute" \
      --latent-points "$target_point" \
      --Bfactor 0 \
      --n-bins 50 \
      --maskrad-fraction 20 \
      --n-min-particles 100 \
      --lazy \
      --save-all-estimates
  else
    echo "Reusing true compute_state output: $true_compute"
  fi
}

case "$SLURM_ARRAY_TASK_ID" in
  0) run_oracle ;;
  1) run_true_pipeline ;;
  *) echo "Unexpected task $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

echo "END=$(date -Is)"
