#!/bin/bash
# Real RECOVAR pipeline.py + compute_state on existing full-atom noise30/B80
# 1M and 3M datasets. Uses one H100, the moving focus mask, and no low-memory
# pipeline flags.

#SBATCH --job-name=spfa-n30-real-pipe
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=900G
#SBATCH --time=72:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_true_pipeline_noise30_b80_h100_fullmem_20260528/slurm/%x-%A_%a.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_true_pipeline_noise30_b80_h100_fullmem_20260528/slurm/%x-%A_%a.err

set -euo pipefail

N_IMAGES_VALUES=(1000000 3000000)
N_IMAGES="${N_IMAGES_VALUES[$SLURM_ARRAY_TASK_ID]}"
N_LABEL="$(printf 'n%08d' "$N_IMAGES")"

WORKDIR="${WORKDIR:-/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar}"
SOURCE_ROOT="${SOURCE_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise30_b80_parallel_20260518}"
RUN_DIR="$SOURCE_ROOT/$N_LABEL/runs/${N_LABEL}_seed0000"
PIPELINE_DIR="$RUN_DIR/06_pipeline_true_recovar_h100_fullmem_movingfocus_20260527"
COMPUTE_STATE_DIR="$RUN_DIR/07_compute_state_true_recovar_h100_fullmem_movingfocus_zdim4_reg_lazy"
TARGET_POINT="$RUN_DIR/target_latent_point_true_recovar_h100_fullmem_movingfocus_zdim4_reg_state0050.txt"
LOG_ROOT="/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_true_pipeline_noise30_b80_h100_fullmem_20260528"
AGENT_ID="true-n30-full-${N_LABEL}-h100-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"

mkdir -p \
  "$LOG_ROOT/slurm" \
  "$LOG_ROOT/tmp/$AGENT_ID" \
  "$LOG_ROOT/pixi_home/$AGENT_ID" \
  "$LOG_ROOT/rattler_cache/$AGENT_ID" \
  "$LOG_ROOT/xdg_cache/$AGENT_ID"

cd "$WORKDIR"

source /usr/share/Modules/init/bash 2>/dev/null || true
module load cudatoolkit/13.2 >/dev/null 2>&1 || true

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TMPDIR="$LOG_ROOT/tmp/$AGENT_ID"
export PIXI_HOME="$LOG_ROOT/pixi_home/$AGENT_ID"
export RATTLER_CACHE_DIR="$LOG_ROOT/rattler_cache/$AGENT_ID"
export XDG_CACHE_HOME="$LOG_ROOT/xdg_cache/$AGENT_ID"
export JAX_COMPILATION_CACHE_DIR="$TMPDIR/jax_cache"
export RECOVAR_CUDA_LIB="$WORKDIR/recovar/cuda/libcuda_backproject.so"
mkdir -p "$TMPDIR" "$PIXI_HOME" "$RATTLER_CACHE_DIR" "$XDG_CACHE_HOME" "$JAX_COMPILATION_CACHE_DIR"

PIXI_PY="$(pixi run which python)"

echo "START=$(date -Is)"
echo "ARRAY_JOB_ID=$SLURM_ARRAY_JOB_ID"
echo "ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "N_IMAGES=$N_IMAGES"
echo "RUN_DIR=$RUN_DIR"
echo "PIPELINE_DIR=$PIPELINE_DIR"
echo "COMPUTE_STATE_DIR=$COMPUTE_STATE_DIR"
echo "COMMIT=$(git rev-parse HEAD)"
echo "BRANCH=$(git rev-parse --abbrev-ref HEAD)"
echo "STATUS=$(git status --short | tr '\n' ';')"
echo "XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION"
nvidia-smi || true

for required in \
  "$RUN_DIR/03_dataset/particles.star" \
  "$RUN_DIR/03_dataset/poses.pkl" \
  "$RUN_DIR/03_dataset/ctf.pkl" \
  "$RUN_DIR/03_dataset/state_assignment.npy" \
  "$RUN_DIR/05_masks/volume_mask_union.mrc" \
  "$RUN_DIR/05_masks/focus_mask_moving.mrc"
do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 2
  fi
done

if [[ ! -f "$RECOVAR_CUDA_LIB" ]]; then
  PYTHON="$PIXI_PY" make -C recovar/cuda clean all
fi

"$PIXI_PY" - <<'PY'
from pathlib import Path
import jax
import recovar
import recovar.cuda_backproject as cb

repo = Path.cwd().resolve()
recovar_file = Path(recovar.__file__).resolve()
jax_file = Path(jax.__file__).resolve()
print("recovar", recovar_file, flush=True)
print("jax", jax_file, flush=True)
print("devices", jax.devices(), flush=True)
print("cuda existing lib", cb._existing_lib_path(), flush=True)
assert str(recovar_file).startswith(str(repo) + "/")
assert ".pixi/envs/default/" in str(jax_file)
if not any(getattr(device, "platform", "") == "gpu" for device in jax.devices()):
    raise RuntimeError("JAX did not see a GPU")
PY

if [[ ! -f "$PIPELINE_DIR/model/params.pkl" ]]; then
  rm -rf "$PIPELINE_DIR"
  /usr/bin/time -v "$PIXI_PY" -m recovar.commands.pipeline \
    "$RUN_DIR/03_dataset/particles.star" \
    --poses "$RUN_DIR/03_dataset/poses.pkl" \
    --ctf "$RUN_DIR/03_dataset/ctf.pkl" \
    --mask "$RUN_DIR/05_masks/volume_mask_union.mrc" \
    --focus-mask "$RUN_DIR/05_masks/focus_mask_moving.mrc" \
    --keep-input-mask \
    --zdim 4 \
    --no-correct-contrast \
    --no-do-over-with-contrast \
    --noise-model radial \
    --uninvert-data false \
    --lazy \
    -o "$PIPELINE_DIR"
else
  echo "Reusing existing pipeline: $PIPELINE_DIR"
fi

"$PIXI_PY" - "$RUN_DIR" "$PIPELINE_DIR" "$TARGET_POINT" <<'PY'
from pathlib import Path
import pickle
import sys
import numpy as np

run = Path(sys.argv[1])
pipeline = Path(sys.argv[2])
target_path = Path(sys.argv[3])
params = pickle.load(open(pipeline / "model" / "params.pkl", "rb"))
if isinstance(params, dict) and params.get("is_oracle_pipeline_output", False):
    raise SystemExit("pipeline unexpectedly marked oracle")
coords = np.load(pipeline / "model" / "zdim_4" / "latent_coords.npy").astype(np.float64)
assignments = np.load(run / "03_dataset" / "state_assignment.npy").astype(np.int64).reshape(-1)
if coords.shape[0] != assignments.shape[0]:
    raise RuntimeError(f"Embedding/state assignment length mismatch: {coords.shape[0]} vs {assignments.shape[0]}")
state_mask = assignments == 50
if not np.any(state_mask):
    raise RuntimeError("No particles assigned to target state 50")
target = np.nanmean(coords[state_mask], axis=0).astype(np.float32).reshape(1, -1)
np.savetxt(target_path, target)
print("TARGET_POINT", target_path, target, flush=True)
PY

if [[ ! -f "$COMPUTE_STATE_DIR/state000_unfil.mrc" ]]; then
  rm -rf "$COMPUTE_STATE_DIR"
  /usr/bin/time -v "$PIXI_PY" -m recovar.commands.compute_state "$PIPELINE_DIR" \
    -o "$COMPUTE_STATE_DIR" \
    --latent-points "$TARGET_POINT" \
    --Bfactor 0 \
    --n-bins 50 \
    --maskrad-fraction 20 \
    --n-min-particles 100 \
    --save-all-estimates \
    --kernel-regression-mode standard \
    --embedding-option cov_dist \
    --lazy
else
  echo "Reusing existing compute_state: $COMPUTE_STATE_DIR"
fi

echo "DONE=$(date -Is)"
echo "PIPELINE_DIR=$PIPELINE_DIR"
echo "COMPUTE_STATE_DIR=$COMPUTE_STATE_DIR"
echo "TARGET_POINT=$TARGET_POINT"
