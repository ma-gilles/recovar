#!/usr/bin/env bash
# Postprocess a full-atom spike dataset-size sweep after RECOVAR finishes.

#SBATCH --job-name=spike-fa-post
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd -P)}"
USER_NAME="${USER:-$(id -un)}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/tmp/${USER_NAME}/recovar_spike}"
ROOT="${ROOT:?Set ROOT to the dataset-size sweep root}"
LABEL="${LABEL:-$(basename "$ROOT")}"
WORKDIR="${WORKDIR:-$SUBMIT_DIR}"
TARGET_STATE="${TARGET_STATE:-50}"
VOXEL_SIZE="${VOXEL_SIZE:-1.25}"
MASK="${MASK:-/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_direct_volume_shell_metrics_20260523/full_gt_vols_plus_masks_20260524/masks/broad_mask.mrc}"
MEAN_SUB_DIR="${MEAN_SUB_DIR:-$ROOT/mean_subtracted_metrics_20260518_r12}"

AGENT_ID="spike-fa-post-${SLURM_JOB_ID}"
export TMPDIR="${TMPDIR:-$SCRATCH_ROOT/tmp/${AGENT_ID}}"
export PIXI_HOME="${PIXI_HOME:-$SCRATCH_ROOT/pixi_home/${AGENT_ID}}"
export RATTLER_CACHE_DIR="${RATTLER_CACHE_DIR:-$SCRATCH_ROOT/rattler_cache/${AGENT_ID}}"
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
