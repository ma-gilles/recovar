#!/usr/bin/env bash
# Retry the IgG-1D true RECOVAR pipeline on the existing 1M/noise30/B80 dataset.
# This job does not stage the particle stack, does not use low-memory flags, and
# runs compute_state with the no-reg embedding once the true pipeline completes.

#SBATCH --job-name=igg1d-realpipe-noreg
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=900G
#SBATCH --time=96:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/%x-%j.out
#SBATCH --error=/scratch/gpfs/GILLES/mg6942/slurmo/%x-%j.err

set -euo pipefail

WORKDIR="${WORKDIR:-/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar}"
RUN_DIR="${RUN_DIR:-/scratch/gpfs/CRYOEM/gilleslab/tmp/igg1d_first50_noise30_b80_1m_20260528/n01000000/runs/n01000000_seed0000}"
ZDIM="${ZDIM:-4}"
TARGET_STATE="${TARGET_STATE:-25}"
TRUE_PIPELINE="${TRUE_PIPELINE:-$RUN_DIR/06_pipeline_true_noreg_retry_platformalloc}"
TRUE_COMPUTE="${TRUE_COMPUTE:-$RUN_DIR/07_compute_state_true_noreg_retry_platformalloc_state0025}"
TARGET_POINT="${TARGET_POINT:-$RUN_DIR/target_latent_point_true_pipeline_noreg_retry_state0025.txt}"
PLOT_DIR="${PLOT_DIR:-$RUN_DIR/08_true_pipeline_noreg_retry_state0025}"

AGENT_ID="igg1d-realpipe-noreg-${SLURM_JOB_ID:-manual}"
export TMPDIR="/scratch/gpfs/GILLES/mg6942/tmp/$AGENT_ID"
export PIXI_HOME="/scratch/gpfs/GILLES/mg6942/pixi_home/$AGENT_ID"
export RATTLER_CACHE_DIR="/scratch/gpfs/GILLES/mg6942/rattler_cache/$AGENT_ID"
mkdir -p "$TMPDIR" "$PIXI_HOME" "$RATTLER_CACHE_DIR" "$RUN_DIR" /scratch/gpfs/GILLES/mg6942/slurmo

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export RECOVAR_CACHE_DIR=
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
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
echo "RUN_DIR=$RUN_DIR"
echo "TRUE_PIPELINE=$TRUE_PIPELINE"
echo "TRUE_COMPUTE=$TRUE_COMPUTE"
echo "TARGET_POINT=$TARGET_POINT"
echo "ZDIM=$ZDIM"
echo "TARGET_STATE=$TARGET_STATE"
echo "COMMIT=$(git rev-parse HEAD)"
echo "BRANCH=$(git rev-parse --abbrev-ref HEAD)"
echo "RECOVAR_CACHE_DIR='${RECOVAR_CACHE_DIR}'"
echo "XLA_PYTHON_CLIENT_ALLOCATOR=$XLA_PYTHON_CLIENT_ALLOCATOR"
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

if [[ ! -f "$TRUE_PIPELINE/model/params.pkl" ]]; then
  /usr/bin/time -v "$PIXI_PY" -m recovar.commands.pipeline \
    "$RUN_DIR/03_dataset/particles.star" \
    --poses "$RUN_DIR/03_dataset/poses.pkl" \
    --ctf "$RUN_DIR/03_dataset/ctf.pkl" \
    -o "$TRUE_PIPELINE" \
    --mask "$RUN_DIR/05_masks/volume_mask_union.mrc" \
    --focus-mask "$RUN_DIR/05_masks/focus_mask_moving.mrc" \
    --no-correct-contrast \
    --noise-model radial \
    --zdim "$ZDIM" \
    --lazy
else
  echo "Reusing existing true pipeline: $TRUE_PIPELINE"
fi

"$PIXI_PY" - "$RUN_DIR" "$TRUE_PIPELINE" "$TARGET_POINT" "$TARGET_STATE" "$ZDIM" <<'PY'
import sys
from pathlib import Path
import numpy as np

run_dir = Path(sys.argv[1])
pipeline_dir = Path(sys.argv[2])
target_path = Path(sys.argv[3])
target_state = int(sys.argv[4])
zdim = int(sys.argv[5])

zs = np.load(pipeline_dir / "model" / f"zdim_{zdim}" / "latent_coords_noreg.npy").astype(np.float64)
states = np.load(run_dir / "03_dataset" / "state_assignment.npy").reshape(-1).astype(np.int64)
if zs.shape[0] != states.size:
    raise RuntimeError(f"latent/state length mismatch: {zs.shape[0]} vs {states.size}")
target = zs[states == target_state, :zdim].mean(axis=0)
target_path.parent.mkdir(parents=True, exist_ok=True)
np.savetxt(target_path, target.reshape(1, -1))
print("wrote no-reg target", target_path, target)
PY

if [[ ! -f "$TRUE_COMPUTE/state000_unfil.mrc" ]]; then
  /usr/bin/time -v "$PIXI_PY" -m recovar.commands.compute_state \
    "$TRUE_PIPELINE" \
    -o "$TRUE_COMPUTE" \
    --latent-points "$TARGET_POINT" \
    --Bfactor 0 \
    --n-bins 50 \
    --maskrad-fraction 20 \
    --n-min-particles 100 \
    --save-all-estimates \
    --kernel-regression-mode standard \
    --embedding-option cov_dist \
    --no-z-regularization \
    --lazy
else
  echo "Reusing existing true compute_state: $TRUE_COMPUTE"
fi

"$PIXI_PY" scripts/experiments/igg1d_oracle/plot_igg1d_oracle_metrics.py \
  --run-dir "$RUN_DIR" \
  --compute-dir "$TRUE_COMPUTE" \
  --mask "$RUN_DIR/05_masks/focus_mask_moving.mrc" \
  --out-dir "$PLOT_DIR" \
  --nearest-count 1000

echo "END=$(date -Is)"
echo "TRUE_PIPELINE=$TRUE_PIPELINE"
echo "TRUE_COMPUTE=$TRUE_COMPUTE"
echo "PLOT_DIR=$PLOT_DIR"
