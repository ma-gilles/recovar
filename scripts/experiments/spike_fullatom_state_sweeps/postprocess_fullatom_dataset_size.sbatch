#!/usr/bin/env bash
# Postprocess a full-atom spike dataset-size sweep after RECOVAR finishes.

#SBATCH --job-name=spike-fa-post
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/%x-%j.out
#SBATCH --error=/scratch/gpfs/GILLES/mg6942/slurmo/%x-%j.err

set -euo pipefail

ROOT="${ROOT:?Set ROOT to the dataset-size sweep root}"
LABEL="${LABEL:-$(basename "$ROOT")}"
WORKDIR="${WORKDIR:-/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar}"
TARGET_STATE="${TARGET_STATE:-50}"
VOXEL_SIZE="${VOXEL_SIZE:-1.25}"
MASK="${MASK:-/scratch/gpfs/GILLES/mg6942/runs/spike_consistency_grid256_noise10_scaling_20260516/volumes_finished_dataset_size/mask_in_grid.mrc}"
MEAN_SUB_DIR="${MEAN_SUB_DIR:-$ROOT/mean_subtracted_metrics_20260518_r12}"

AGENT_ID="spike-fa-post-${SLURM_JOB_ID}"
export TMPDIR="/scratch/gpfs/GILLES/mg6942/tmp/${AGENT_ID}"
export PIXI_HOME="/scratch/gpfs/GILLES/mg6942/pixi_home/${AGENT_ID}"
export RATTLER_CACHE_DIR="/scratch/gpfs/GILLES/mg6942/rattler_cache/${AGENT_ID}"
mkdir -p "$TMPDIR" "$PIXI_HOME" "$RATTLER_CACHE_DIR"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "$WORKDIR"
PIXI_PY="$(pixi run which python)"

echo "start=$(date -Is)"
echo "root=$ROOT"
echo "label=$LABEL"
echo "mask=$MASK"
echo "target_state=$TARGET_STATE"
echo "python=$PIXI_PY"

"$PIXI_PY" scripts/experiments/spike_fullatom_state_sweeps/score_dataset_size_fsc_vs_gt.py \
  --root "$ROOT" \
  --label "$LABEL" \
  --target-state "$TARGET_STATE" \
  --mask "$MASK" \
  --voxel-size "$VOXEL_SIZE"

"$PIXI_PY" scripts/experiments/spike_fullatom_state_sweeps/score_mean_subtracted_dataset_size.py \
  --root "$ROOT" \
  --out-dir "$MEAN_SUB_DIR" \
  --target-state "$TARGET_STATE" \
  --voxel-size "$VOXEL_SIZE" \
  --locres-sampling-A 12 \
  --locres-radius-A 12 \
  --locres-max-points 400

"$PIXI_PY" scripts/experiments/spike_fullatom_state_sweeps/plot_mean_subtracted_minfreq_sweep.py \
  --metrics-dir "$MEAN_SUB_DIR" \
  --plot-min-frequency 0.05

echo "done=$(date -Is)"
