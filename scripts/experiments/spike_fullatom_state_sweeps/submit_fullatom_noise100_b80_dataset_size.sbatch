#!/usr/bin/env bash
# Full-atom spike dataset-size sweep with noisier images and B80-rendered GT.
#
# This intentionally uses the same spike_consistency worktree/driver that
# generated the B100/noise10 dataset-size sweep so the output layout remains:
#
#   <BASE_ROOT>/nXXXXXXXX/runs/nXXXXXXXX_seed0000/
#     03_dataset/
#     04_ground_truth/
#     05_masks/
#     06_pipeline/
#     07_compute_state/
#
# Usage:
#   sbatch scripts/experiments/spike_fullatom_state_sweeps/submit_fullatom_noise100_b80_dataset_size.sbatch

#SBATCH --job-name=spike-fa-n100-b80
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=900G
#SBATCH --time=96:00:00
#SBATCH --array=0-4%5
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd -P)}"
USER_NAME="${USER:-$(id -un)}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/tmp/${USER_NAME}/recovar_spike}"
N_IMAGES_VALUES_STR="${N_IMAGES_VALUES_STR:-10000 30000 100000 300000 1000000}"
read -r -a N_IMAGES_VALUES <<< "$N_IMAGES_VALUES_STR"
N_IMAGES="${N_IMAGES_VALUES[$SLURM_ARRAY_TASK_ID]}"
N_LABEL="$(printf 'n%08d' "$N_IMAGES")"

WORKDIR="${WORKDIR:-$SUBMIT_DIR}"
NOISE_LEVEL="${NOISE_LEVEL:-100.0}"
RENDER_BFACTOR="${RENDER_BFACTOR:-80}"
BASE_ROOT="${BASE_ROOT:-$SCRATCH_ROOT/spike_fullatom_consistency_grid256_noise100_b80_parallel}"
SHARED_ROOT="${SHARED_ROOT:-$SCRATCH_ROOT/spike_fullatom_consistency_grid256_noise100_b80_shared}"
SHARED_RAW="$SHARED_ROOT/01_raw_volumes"
OUT_ROOT="$BASE_ROOT/$N_LABEL"
PDB_DIR="${PDB_DIR:-/projects/CRYOEM/singerlab/mg6942/spike_morph_pdbs}"
MASK="${MASK:-/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_direct_volume_shell_metrics_20260523/full_gt_vols_plus_masks_20260524/masks/broad_mask.mrc}"
AGENT_ID="spike-fa-noise${NOISE_LEVEL}-b${RENDER_BFACTOR}-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"

export TMPDIR="${TMPDIR:-$SCRATCH_ROOT/tmp/${AGENT_ID}}"
export PIXI_HOME="${PIXI_HOME:-$SCRATCH_ROOT/pixi_home/${AGENT_ID}}"
export RATTLER_CACHE_DIR="${RATTLER_CACHE_DIR:-$SCRATCH_ROOT/rattler_cache/${AGENT_ID}}"
mkdir -p "$TMPDIR" "$PIXI_HOME" "$RATTLER_CACHE_DIR" "$OUT_ROOT/shared/01_raw_volumes" "$SHARED_RAW"

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
echo "HOST=$(hostname)"
echo "WORKDIR=$WORKDIR"
echo "BASE_ROOT=$BASE_ROOT"
echo "OUT_ROOT=$OUT_ROOT"
echo "SHARED_RAW=$SHARED_RAW"
echo "PDB_DIR=$PDB_DIR"
echo "MASK=$MASK"
echo "NOISE_LEVEL=$NOISE_LEVEL"
echo "RENDER_BFACTOR=$RENDER_BFACTOR"
echo "START=$(date -Is)"
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

count_shared_raw() {
  find "$SHARED_RAW" -maxdepth 1 -name 'vol*.mrc' 2>/dev/null | wc -l
}

wait_for_shared_raw() {
  for _ in $(seq 1 720); do
    raw_count="$(count_shared_raw)"
    if [[ "$raw_count" -eq 100 ]]; then
      return 0
    fi
    echo "Waiting for shared B80 raw volumes: found $raw_count/100 at $(date -Is)"
    sleep 30
  done
  echo "Timed out waiting for shared raw volumes in $SHARED_RAW" >&2
  return 1
}

raw_count="$(count_shared_raw)"
if [[ "$raw_count" -ne 100 ]]; then
  LOCK_DIR="$SHARED_ROOT/render.lock"
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    cleanup_lock() {
      rmdir "$LOCK_DIR" 2>/dev/null || true
    }
    trap cleanup_lock EXIT
    echo "Rendering shared B80 raw volumes into $SHARED_ROOT"
    /usr/bin/time -v "$PIXI_PY" -m recovar.commands.render_spike_morph_volumes \
      "$SHARED_ROOT" \
      --pdb-dir "$PDB_DIR" \
      --grid-size 256 \
      --voxel-size 1.25 \
      --bfactor "$RENDER_BFACTOR"
    cleanup_lock
    trap - EXIT
  else
    wait_for_shared_raw
  fi
fi

raw_count="$(count_shared_raw)"
if [[ "$raw_count" -ne 100 ]]; then
  echo "Expected 100 shared raw volumes in $SHARED_RAW, found $raw_count" >&2
  exit 2
fi

ln -sfn "$SHARED_RAW"/vol*.mrc "$OUT_ROOT/shared/01_raw_volumes"/

/usr/bin/time -v "$PIXI_PY" -m recovar.commands.spike_consistency \
  --output-dir "$OUT_ROOT" \
  --mask "$MASK" \
  --pdb-dir "$PDB_DIR" \
  --grid-size 256 \
  --render-voxel-size 1.25 \
  --render-bfactor "$RENDER_BFACTOR" \
  --n-images-list "$N_IMAGES" \
  --noise-level "$NOISE_LEVEL" \
  --bfactor 0 \
  --zdim 4 \
  --seed 0 \
  --lazy \
  --low-memory-option \
  --pipeline-gpu-memory 55 \
  --compute-state-save-all-estimates

echo "END=$(date -Is)"
