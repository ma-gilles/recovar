#!/usr/bin/env bash
#SBATCH --job-name=spike-fa-state-cs
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=900G
#SBATCH --time=72:00:00
#SBATCH --array=0-5%6
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/%x-%A_%a.out
#SBATCH --error=/scratch/gpfs/GILLES/mg6942/slurmo/%x-%A_%a.err

set -euo pipefail

N_IMAGES_VALUES_STR="${N_IMAGES_VALUES_STR:-10000 30000 100000 300000 1000000 3000000}"
read -r -a N_IMAGES_VALUES <<< "$N_IMAGES_VALUES_STR"
N_IMAGES="${N_IMAGES_VALUES[$SLURM_ARRAY_TASK_ID]}"
N_LABEL="$(printf 'n%08d' "$N_IMAGES")"

WORKDIR="${WORKDIR:-/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar}"
SOURCE_ROOT="${SOURCE_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise10_b100_parallel_20260516}"
STATE_ROOT="${STATE_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise10_b100_state0025_reuse_20260517}"
TARGET_STATE="${TARGET_STATE:-25}"
ZDIM="${ZDIM:-4}"
GPU_MEMORY="${GPU_MEMORY:-55}"

SOURCE_RUN="$SOURCE_ROOT/$N_LABEL/runs/${N_LABEL}_seed0000"
RUN_DIR="$STATE_ROOT/$N_LABEL/runs/${N_LABEL}_seed0000"
COMPUTE_STATE_DIR="$RUN_DIR/07_compute_state"
TARGET_PATH="$RUN_DIR/target_latent_point_state$(printf '%04d' "$TARGET_STATE").txt"
TARGET_STATE_LABEL="$(printf 's%04d' "$TARGET_STATE")"
AGENT_ID="spike-fa-${TARGET_STATE_LABEL}-cs-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"

export TMPDIR="/scratch/gpfs/GILLES/mg6942/tmp/${AGENT_ID}"
export PIXI_HOME="/scratch/gpfs/GILLES/mg6942/pixi_home/${AGENT_ID}"
export RATTLER_CACHE_DIR="/scratch/gpfs/GILLES/mg6942/rattler_cache/${AGENT_ID}"
mkdir -p "$TMPDIR" "$PIXI_HOME" "$RATTLER_CACHE_DIR" "$RUN_DIR"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR=cuda_malloc_async
export XDG_CACHE_HOME="$TMPDIR/xdg_cache"
export JAX_COMPILATION_CACHE_DIR="$TMPDIR/jax_cache"
mkdir -p "$XDG_CACHE_HOME" "$JAX_COMPILATION_CACHE_DIR"

cd "$WORKDIR"
module unload cudatoolkit/12.8 >/dev/null 2>&1 || true

echo "ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID}"
echo "ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "N_IMAGES=${N_IMAGES}"
echo "N_IMAGES_VALUES_STR=$N_IMAGES_VALUES_STR"
echo "TARGET_STATE=${TARGET_STATE}"
echo "HOST=$(hostname)"
echo "WORKDIR=$WORKDIR"
echo "SOURCE_RUN=$SOURCE_RUN"
echo "RUN_DIR=$RUN_DIR"
echo "START=$(date -Is)"
nvidia-smi || true

for required in \
  "$SOURCE_RUN/03_dataset/state_assignment.npy" \
  "$SOURCE_RUN/04_ground_truth/gt_vol$(printf '%04d' "$TARGET_STATE").mrc" \
  "$SOURCE_RUN/05_masks/volume_mask_union.mrc" \
  "$SOURCE_RUN/06_pipeline/model/zdim_${ZDIM}/latent_coords.npy"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required source artifact: $required" >&2
    exit 2
  fi
done

ln -sfn "$SOURCE_RUN/03_dataset" "$RUN_DIR/03_dataset"
ln -sfn "$SOURCE_RUN/04_ground_truth" "$RUN_DIR/04_ground_truth"
ln -sfn "$SOURCE_RUN/05_masks" "$RUN_DIR/05_masks"
ln -sfn "$SOURCE_RUN/06_pipeline" "$RUN_DIR/06_pipeline"
printf '%s\n' "$SOURCE_RUN" > "$RUN_DIR/source_run.txt"

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
import jax, pathlib, recovar
print("recovar", recovar.__file__)
print("jax", jax.__file__)
devices = jax.devices()
print("devices", devices)
assert pathlib.Path(recovar.__file__).resolve().is_relative_to(pathlib.Path.cwd().resolve())
assert ".pixi/envs/default" in str(pathlib.Path(jax.__file__).resolve())
if not any(getattr(device, "platform", "") == "gpu" for device in devices):
    raise RuntimeError(f"JAX did not see a GPU: {devices}")
PY

"$PIXI_PY" - "$SOURCE_RUN" "$TARGET_PATH" "$TARGET_STATE" "$ZDIM" <<'PY'
from pathlib import Path
import sys
import numpy as np

source_run = Path(sys.argv[1])
target_path = Path(sys.argv[2])
target_state = int(sys.argv[3])
zdim = int(sys.argv[4])

zs = np.load(source_run / "06_pipeline" / "model" / f"zdim_{zdim}" / "latent_coords.npy")
assignments = np.load(source_run / "03_dataset" / "state_assignment.npy").reshape(-1)
mask = assignments == target_state
if not np.any(mask):
    raise RuntimeError(f"No particles assigned to state {target_state} in {source_run}")
target = np.mean(zs[mask], axis=0, dtype=np.float64).astype(np.float32)
target_path.parent.mkdir(parents=True, exist_ok=True)
np.savetxt(target_path, target.reshape(1, -1))
print("target_path", target_path)
print("target_state_count", int(mask.sum()))
print("target_latent_point", target.tolist())
PY

if [[ -f "$COMPUTE_STATE_DIR/state000_unfil.mrc" && "${OVERWRITE:-0}" != "1" ]]; then
  echo "Compute_state already exists, skipping: $COMPUTE_STATE_DIR"
  exit 0
fi

/usr/bin/time -v "$PIXI_PY" -m recovar.commands.compute_state \
  "$SOURCE_RUN/06_pipeline" \
  --latent-points "$TARGET_PATH" \
  -o "$COMPUTE_STATE_DIR" \
  --Bfactor 0 \
  --n-bins 50 \
  --maskrad-fraction 20 \
  --n-min-particles 100 \
  --lazy \
  --gpu-memory "$GPU_MEMORY" \
  --save-all-estimates

echo "END=$(date -Is)"
find "$RUN_DIR" -maxdepth 3 -type f | sort
