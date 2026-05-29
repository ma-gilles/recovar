#!/bin/bash
# Run compute_state from the ordinary RECOVAR pipeline embedding, zdim4 regularized.

#SBATCH --job-name=spfa-n30-pipe-z4-cs
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --time=48:00:00
#SBATCH --array=0-6%7
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_pipeline_zdim4_reg_compute_state_noise30_20260526/slurm/%x-%A_%a.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_pipeline_zdim4_reg_compute_state_noise30_20260526/slurm/%x-%A_%a.err

set -euo pipefail

N_IMAGES_VALUES=(10000 30000 100000 300000 1000000 3000000 10000000)
N_IMAGES="${N_IMAGES_VALUES[$SLURM_ARRAY_TASK_ID]}"
N_LABEL="$(printf 'n%08d' "$N_IMAGES")"

WORKDIR="${WORKDIR:-/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar}"
SOURCE_ROOT="${SOURCE_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise30_b80_parallel_20260518}"
LOG_ROOT="${LOG_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_pipeline_zdim4_reg_compute_state_noise30_20260526}"
RUN_DIR="${SOURCE_ROOT}/${N_LABEL}/runs/${N_LABEL}_seed0000"
PIPELINE_DIR="$RUN_DIR/06_pipeline"
TARGET_STATE="${TARGET_STATE:-50}"
TARGET_POINT="$RUN_DIR/target_latent_point_pipeline_zdim4_reg_state$(printf '%04d' "$TARGET_STATE").txt"
OUT_DIR="$RUN_DIR/07_compute_state_pipeline_zdim4_reg_lazy"
AGENT_ID="spfa-n30-pipe-z4-cs-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"

mkdir -p "$LOG_ROOT/slurm" "$LOG_ROOT/tmp/$AGENT_ID" "$LOG_ROOT/pixi_home/$AGENT_ID" "$LOG_ROOT/rattler_cache/$AGENT_ID" "$LOG_ROOT/xdg_cache/$AGENT_ID"

cd "$WORKDIR"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TMPDIR="$LOG_ROOT/tmp/$AGENT_ID"
export PIXI_HOME="$LOG_ROOT/pixi_home/$AGENT_ID"
export RATTLER_CACHE_DIR="$LOG_ROOT/rattler_cache/$AGENT_ID"
export XDG_CACHE_HOME="$LOG_ROOT/xdg_cache/$AGENT_ID"
export JAX_COMPILATION_CACHE_DIR="$TMPDIR/jax_cache"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

echo "START=$(date -Is)"
echo "ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID}"
echo "ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "N_IMAGES=$N_IMAGES"
echo "RUN_DIR=$RUN_DIR"
echo "PIPELINE_DIR=$PIPELINE_DIR"
echo "TARGET_POINT=$TARGET_POINT"
echo "OUT_DIR=$OUT_DIR"
echo "TARGET_STATE=$TARGET_STATE"
echo "COMMIT=$(git rev-parse HEAD)"
echo "BRANCH=$(git rev-parse --abbrev-ref HEAD)"
echo "STATUS=$(git status --short | tr '\n' ';')"
nvidia-smi || true

if [[ ! -f "$PIPELINE_DIR/model/params.pkl" ]]; then
  echo "Missing pipeline output: $PIPELINE_DIR/model/params.pkl" >&2
  exit 2
fi
if [[ ! -f "$PIPELINE_DIR/model/zdim_4/latent_coords.npy" ]]; then
  echo "Missing pipeline zdim4 coordinates: $PIPELINE_DIR/model/zdim_4/latent_coords.npy" >&2
  exit 2
fi
if [[ ! -f "$PIPELINE_DIR/model/zdim_4/latent_precision.npy" ]]; then
  echo "Missing pipeline zdim4 precision: $PIPELINE_DIR/model/zdim_4/latent_precision.npy" >&2
  exit 2
fi
if [[ ! -f "$RUN_DIR/03_dataset/state_assignment.npy" ]]; then
  echo "Missing state assignment: $RUN_DIR/03_dataset/state_assignment.npy" >&2
  exit 2
fi

PIXI_PY="$(pixi run which python)"
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

"$PIXI_PY" - "$RUN_DIR" "$PIPELINE_DIR" "$TARGET_POINT" "$TARGET_STATE" <<'PY'
from pathlib import Path
import json
import sys
import numpy as np

run = Path(sys.argv[1])
pipeline = Path(sys.argv[2])
target_path = Path(sys.argv[3])
target_state = int(sys.argv[4])

coords_path = pipeline / "model" / "zdim_4" / "latent_coords.npy"
precision_path = pipeline / "model" / "zdim_4" / "latent_precision.npy"
assignments_path = run / "03_dataset" / "state_assignment.npy"
assignments = np.load(assignments_path).astype(np.int64).reshape(-1)
coords = np.load(coords_path).astype(np.float64)
if coords.shape[0] != assignments.shape[0]:
    raise RuntimeError(f"Embedding/assignment length mismatch: {coords.shape[0]} vs {assignments.shape[0]}")
mask = assignments == target_state
if not np.any(mask):
    raise RuntimeError(f"No particles assigned to state {target_state}")
target = np.nanmean(coords[mask], axis=0, dtype=np.float64).astype(np.float32).reshape(1, -1)
np.savetxt(target_path, target)
meta = {
    "embedding_source": "ordinary RECOVAR pipeline",
    "pipeline": str(pipeline),
    "zdim": 4,
    "regularization": "reg",
    "embedding_option": "cov_dist",
    "coords_path": str(coords_path),
    "precision_path": str(precision_path),
    "state_assignment_path": str(assignments_path),
    "target_state": target_state,
    "target_state_count": int(np.sum(mask)),
    "target_point": target.reshape(-1).tolist(),
}
(run / f"pipeline_zdim4_reg_state{target_state:04d}_metadata.json").write_text(
    json.dumps(meta, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(meta, indent=2, sort_keys=True))
PY

if [[ -f "$OUT_DIR/state000_unfil.mrc" && "${OVERWRITE:-0}" != "1" ]]; then
  echo "Reusing completed compute_state at $OUT_DIR"
  exit 0
fi

cmd=(
  "$PIXI_PY" -m recovar.commands.compute_state "$PIPELINE_DIR"
  -o "$OUT_DIR"
  --latent-points "$TARGET_POINT"
  --Bfactor 0
  --n-bins 50
  --maskrad-fraction 20
  --n-min-particles 100
  --save-all-estimates
  --kernel-regression-mode standard
  --embedding-option cov_dist
  --lazy
)

echo "COMMAND=${cmd[*]}"
/usr/bin/time -v "${cmd[@]}"

echo "DONE=$(date -Is)"
echo "OUT_DIR=$OUT_DIR"
echo "TARGET_POINT=$TARGET_POINT"
