#!/usr/bin/env bash
# PPCA pipeline + zdim4 no-reg compute_state for the 1M spike noise10/B100 dataset.

#SBATCH --job-name=spike-n10-ppca1m
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=900G
#SBATCH --time=72:00:00
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_true_pipeline_sweep_noise10_b100_dev2_20260529/slurm/%x-%j.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_true_pipeline_sweep_noise10_b100_dev2_20260529/slurm/%x-%j.err

set -euo pipefail

WORKDIR="${WORKDIR:-/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_kernel_dev2_rebase}"
ROOT="${ROOT:-/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_true_pipeline_sweep_noise10_b100_dev2_20260529}"
SOURCE_ROOT="${SOURCE_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise10_b100_parallel_20260516}"
N_IMAGES="${N_IMAGES:-1000000}"
N_LABEL="$(printf 'n%08d' "$N_IMAGES")"
RUN_DIR="$ROOT/$N_LABEL/runs/${N_LABEL}_seed0000"
SOURCE_RUN="$SOURCE_ROOT/$N_LABEL/runs/${N_LABEL}_seed0000"
PIPELINE_DIR="$RUN_DIR/06_pipeline_true_zdim4_ppca_no_keep_input_mask"
TARGET_ROOT="$RUN_DIR/targets_zdim4_ppca_noreg_mean_embedding"
COMPUTE_ROOT="$RUN_DIR/07_compute_state_zdim4_ppca_noreg"
METRICS_ROOT="$RUN_DIR/08_metrics_zdim4_ppca_noreg"
BROAD_MASK="${BROAD_MASK:-/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_direct_volume_shell_metrics_20260523/full_gt_vols_plus_masks_20260524/masks/broad_mask.mrc}"
TARGET_STATES_STR="${TARGET_STATES_STR:-0 25 50}"
PPCA_ITERS="${PPCA_ITERS:-20}"
PPCA_ZDIM="${PPCA_ZDIM:-4}"
AGENT_ID="spike-n10-ppca1m-${SLURM_JOB_ID:-manual}"

mkdir -p \
  "$ROOT/slurm" \
  "$ROOT/tmp/$AGENT_ID" \
  "$ROOT/pixi_home/$AGENT_ID" \
  "$ROOT/rattler_cache/$AGENT_ID" \
  "$ROOT/xdg_cache/$AGENT_ID" \
  "$TARGET_ROOT" \
  "$COMPUTE_ROOT" \
  "$METRICS_ROOT"

cd "$WORKDIR"
source /usr/share/Modules/init/bash 2>/dev/null || true
module load cudatoolkit/13.2 >/dev/null 2>&1 || true

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TMPDIR="$ROOT/tmp/$AGENT_ID"
export PIXI_HOME="$ROOT/pixi_home/$AGENT_ID"
export RATTLER_CACHE_DIR="$ROOT/rattler_cache/$AGENT_ID"
export XDG_CACHE_HOME="$ROOT/xdg_cache/$AGENT_ID"
export JAX_COMPILATION_CACHE_DIR="$TMPDIR/jax_cache"
export RECOVAR_CUDA_LIB="$WORKDIR/recovar/cuda/libcuda_backproject.so"
# Empty value disables transparent particle-stack staging; do not copy 1M images into TMPDIR.
export RECOVAR_CACHE_DIR=""
mkdir -p "$TMPDIR" "$PIXI_HOME" "$RATTLER_CACHE_DIR" "$XDG_CACHE_HOME" "$JAX_COMPILATION_CACHE_DIR"

PIXI_PY="$WORKDIR/.pixi/envs/default/bin/python"

cat <<INFO
START=$(date -Is)
WORKDIR=$WORKDIR
ROOT=$ROOT
SOURCE_ROOT=$SOURCE_ROOT
RUN_DIR=$RUN_DIR
SOURCE_RUN=$SOURCE_RUN
PIPELINE_DIR=$PIPELINE_DIR
COMPUTE_ROOT=$COMPUTE_ROOT
METRICS_ROOT=$METRICS_ROOT
BROAD_MASK=$BROAD_MASK
TARGET_STATES=$TARGET_STATES_STR
PPCA_ZDIM=$PPCA_ZDIM
PPCA_ITERS=$PPCA_ITERS
RECOVAR_CACHE_DIR=${RECOVAR_CACHE_DIR@Q}
COMMIT=$(git rev-parse HEAD)
BRANCH=$(git rev-parse --abbrev-ref HEAD)
STATUS=$(git status --short | tr '\n' ';')
INFO
nvidia-smi || true

for required in \
  "$SOURCE_RUN/03_dataset/particles.star" \
  "$SOURCE_RUN/03_dataset/poses.pkl" \
  "$SOURCE_RUN/03_dataset/ctf.pkl" \
  "$SOURCE_RUN/03_dataset/state_assignment.npy" \
  "$SOURCE_RUN/05_masks/volume_mask_union.mrc" \
  "$SOURCE_RUN/05_masks/focus_mask_moving.mrc" \
  "$SOURCE_RUN/04_ground_truth/gt_vol0050.mrc" \
  "$BROAD_MASK" \
  "$PIXI_PY"
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
    "$SOURCE_RUN/03_dataset/particles.star" \
    --poses "$SOURCE_RUN/03_dataset/poses.pkl" \
    --ctf "$SOURCE_RUN/03_dataset/ctf.pkl" \
    --mask "$SOURCE_RUN/05_masks/volume_mask_union.mrc" \
    --focus-mask "$SOURCE_RUN/05_masks/focus_mask_moving.mrc" \
    --zdim "$PPCA_ZDIM" \
    --no-correct-contrast \
    --no-do-over-with-contrast \
    --noise-model radial \
    --uninvert-data false \
    --lazy \
    --use-ppca \
    --ppca-zdim "$PPCA_ZDIM" \
    --ppca-em-iters "$PPCA_ITERS" \
    -o "$PIPELINE_DIR"
else
  echo "Reusing existing PPCA pipeline: $PIPELINE_DIR"
fi

"$PIXI_PY" - "$SOURCE_RUN" "$PIPELINE_DIR" "$TARGET_ROOT" "$TARGET_STATES_STR" <<'PY'
from pathlib import Path
import json
import sys
import numpy as np
from recovar.output import output as output_mod

source_run = Path(sys.argv[1])
pipeline_dir = Path(sys.argv[2])
target_root = Path(sys.argv[3])
states = [int(value) for value in sys.argv[4].split()]
po = output_mod.PipelineOutput(str(pipeline_dir))
keys = po.get_embedding_keys("latent_coords_noreg")
if 4 not in keys and "zdim_4" not in keys:
    raise RuntimeError(f"PPCA pipeline has no zdim4 noreg embedding; keys={keys}")
coords = np.asarray(po.get_embedding_component("latent_coords_noreg", 4), dtype=np.float64)
assignments = np.load(source_run / "03_dataset" / "state_assignment.npy").astype(np.int64).reshape(-1)
if coords.shape[0] != assignments.size:
    raise RuntimeError(f"Embedding/state assignment length mismatch: {coords.shape[0]} vs {assignments.size}")
manifest = {
    "source_run": str(source_run),
    "pipeline_dir": str(pipeline_dir),
    "coords_entry": "latent_coords_noreg",
    "precision_entry": "latent_precision_noreg",
    "zdim": 4,
    "targets": [],
}
for state in states:
    mask = assignments == state
    if not np.any(mask):
        raise RuntimeError(f"No particles assigned to state {state}")
    target = np.mean(coords[mask], axis=0, dtype=np.float64).astype(np.float32).reshape(1, -1)
    path = target_root / f"state{state:04d}.txt"
    np.savetxt(path, target)
    info = {
        "state": state,
        "target_state_count": int(mask.sum()),
        "target_path": str(path),
        "target_point": target.reshape(-1).astype(float).tolist(),
    }
    (target_root / f"state{state:04d}.json").write_text(json.dumps(info, indent=2) + "\n")
    manifest["targets"].append(info)
(target_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2), flush=True)
PY

for state in $TARGET_STATES_STR; do
  state4="$(printf '%04d' "$state")"
  target_path="$TARGET_ROOT/state${state4}.txt"
  out_dir="$COMPUTE_ROOT/state${state4}"
  metrics_dir="$METRICS_ROOT/state${state4}"

  if [[ -f "$out_dir/state000_unfil.mrc" && "${OVERWRITE_COMPUTE_STATE:-0}" != "1" ]]; then
    echo "Reusing compute_state: $out_dir"
  else
    rm -rf "$out_dir"
    cmd=(
      "$PIXI_PY" -m recovar.commands.compute_state
      "$PIPELINE_DIR"
      -o "$out_dir"
      --latent-points "$target_path"
      --Bfactor 0
      --n-bins 50
      --maskrad-fraction 20
      --n-min-particles 100
      --save-all-estimates
      --kernel-regression-mode standard
      --embedding-option cov_dist
      --no-z-regularization
      --lazy
    )
    echo "COMPUTE_STATE_COMMAND=${cmd[*]}"
    /usr/bin/time -v "${cmd[@]}"
  fi

  "$PIXI_PY" scripts/experiments/spike_fullatom_state_sweeps/plot_compute_state_shell_metrics.py \
    --run-dir "$SOURCE_RUN" \
    --compute-state-dir "$out_dir" \
    --target-state "$state" \
    --mask "$BROAD_MASK" \
    --voxel-size 1.25 \
    --out-dir "$metrics_dir"
done

"$PIXI_PY" scripts/experiments/spike_fullatom_method_benchmark/plot_ppca_1m_overlay_noise10.py \
  --root "$ROOT" \
  --source-root "$SOURCE_ROOT" \
  --ppca-run "$RUN_DIR" \
  --out-dir "$ROOT/09_ppca_1m_overlay_broadmask_soft_current" \
  --mask "$BROAD_MASK" \
  --states 0,25,50

cat > "$RUN_DIR/ppca_1m_manifest.txt" <<MANIFEST
source_run=$SOURCE_RUN
ppca_pipeline_dir=$PIPELINE_DIR
ppca_compute_root=$COMPUTE_ROOT
ppca_metrics_root=$METRICS_ROOT
target_root=$TARGET_ROOT
broad_mask=$BROAD_MASK
regularization=noreg
zdim=4
embedding_option=cov_dist
ppca_zdim=$PPCA_ZDIM
ppca_em_iters=$PPCA_ITERS
MANIFEST

echo "DONE=$(date -Is)"
echo "PIPELINE_DIR=$PIPELINE_DIR"
echo "COMPUTE_ROOT=$COMPUTE_ROOT"
echo "METRICS_ROOT=$METRICS_ROOT"
echo "OVERLAY_DIR=$ROOT/09_ppca_1m_overlay_broadmask_soft_current"
