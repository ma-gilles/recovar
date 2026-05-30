#!/usr/bin/env bash
# RECOVAR compute_state sweep for spike full-atom noise1/B80 datasets.
# Reuses existing real pipeline.py outputs, then evaluates states 0,25,50.

#SBATCH --job-name=spike-n1-recovar
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=900G
#SBATCH --time=72:00:00
#SBATCH --array=0-4%5
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise1_b80_20260530/slurm/%x-%A_%a.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise1_b80_20260530/slurm/%x-%A_%a.err

set -euo pipefail

WORKDIR="${WORKDIR:-/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_kernel_dev2_rebase}"
SOURCE_ROOT="${SOURCE_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise1_b80_true_oracle_sweep_20260527}"
BENCH_ROOT="${BENCH_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise1_b80_20260530}"
BROAD_MASK="${BROAD_MASK:-/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_direct_volume_shell_metrics_20260523/full_gt_vols_plus_masks_20260524/masks/broad_mask.mrc}"
N_IMAGES_VALUES_STR="${N_IMAGES_VALUES_STR:-30000 100000 300000 1000000 3000000}"
TARGET_STATES_STR="${TARGET_STATES_STR:-0 25 50}"
PIPELINE_RELPATH="${PIPELINE_RELPATH:-06_pipeline_true_recovar_h100_fullmem_movingfocus_20260527}"
read -r -a N_IMAGES_VALUES <<< "$N_IMAGES_VALUES_STR"

N_IMAGES="${N_IMAGES_VALUES[$SLURM_ARRAY_TASK_ID]}"
N_LABEL="$(printf 'n%08d' "$N_IMAGES")"
RUN_DIR="$SOURCE_ROOT/$N_LABEL/runs/${N_LABEL}_seed0000"
PIPELINE_DIR="$RUN_DIR/$PIPELINE_RELPATH"
OUT_ROOT="$BENCH_ROOT/recovar/$N_LABEL"
TARGET_ROOT="$OUT_ROOT/targets_zdim4_noreg_state_means"
COMPUTE_ROOT="$OUT_ROOT/compute_state_zdim4_noreg_focus"
METRICS_ROOT="$OUT_ROOT/metrics_zdim4_noreg_broadmask"
AGENT_ID="spike-n1-recovar-${N_LABEL}-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"

mkdir -p \
  "$BENCH_ROOT/slurm" \
  "$BENCH_ROOT/tmp/$AGENT_ID" \
  "$BENCH_ROOT/pixi_home/$AGENT_ID" \
  "$BENCH_ROOT/rattler_cache/$AGENT_ID" \
  "$BENCH_ROOT/xdg_cache/$AGENT_ID" \
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
export TMPDIR="$BENCH_ROOT/tmp/$AGENT_ID"
export PIXI_HOME="$BENCH_ROOT/pixi_home/$AGENT_ID"
export RATTLER_CACHE_DIR="$BENCH_ROOT/rattler_cache/$AGENT_ID"
export XDG_CACHE_HOME="$BENCH_ROOT/xdg_cache/$AGENT_ID"
export JAX_COMPILATION_CACHE_DIR="$TMPDIR/jax_cache"
export RECOVAR_CUDA_LIB="$WORKDIR/recovar/cuda/libcuda_backproject.so"
mkdir -p "$TMPDIR" "$PIXI_HOME" "$RATTLER_CACHE_DIR" "$XDG_CACHE_HOME" "$JAX_COMPILATION_CACHE_DIR"

PIXI_PY="$WORKDIR/.pixi/envs/default/bin/python"

echo "START=$(date -Is)"
echo "ARRAY_JOB_ID=$SLURM_ARRAY_JOB_ID"
echo "ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "N_IMAGES=$N_IMAGES"
echo "RUN_DIR=$RUN_DIR"
echo "PIPELINE_DIR=$PIPELINE_DIR"
echo "OUT_ROOT=$OUT_ROOT"
echo "BROAD_MASK=$BROAD_MASK"
echo "TARGET_STATES=$TARGET_STATES_STR"
echo "COMMIT=$(git rev-parse HEAD)"
echo "BRANCH=$(git rev-parse --abbrev-ref HEAD)"
echo "STATUS=$(git status --short | tr '\n' ';')"
nvidia-smi || true

for required in \
  "$RUN_DIR/03_dataset/particles.star" \
  "$RUN_DIR/03_dataset/poses.pkl" \
  "$RUN_DIR/03_dataset/ctf.pkl" \
  "$RUN_DIR/03_dataset/state_assignment.npy" \
  "$RUN_DIR/05_masks/volume_mask_union.mrc" \
  "$RUN_DIR/05_masks/focus_mask_moving.mrc" \
  "$PIPELINE_DIR/model/params.pkl" \
  "$PIPELINE_DIR/model/zdim_4/latent_coords_noreg.npy" \
  "$PIPELINE_DIR/model/zdim_4/latent_precision_noreg.npy" \
  "$BROAD_MASK"
do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 2
  fi
done

if [[ ! -x "$PIXI_PY" ]]; then
  echo "Missing pixi Python: $PIXI_PY" >&2
  exit 2
fi

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

"$PIXI_PY" - "$RUN_DIR" "$PIPELINE_DIR" "$TARGET_ROOT" "$TARGET_STATES_STR" <<'PY'
from pathlib import Path
import json
import pickle
import sys

import numpy as np

run = Path(sys.argv[1])
pipeline = Path(sys.argv[2])
target_root = Path(sys.argv[3])
states = [int(value) for value in sys.argv[4].split()]

params = pickle.load(open(pipeline / "model" / "params.pkl", "rb"))
if isinstance(params, dict) and params.get("is_oracle_pipeline_output", False):
    raise RuntimeError(f"Pipeline is marked oracle-generated: {pipeline}")

assignments = np.load(run / "03_dataset" / "state_assignment.npy").astype(np.int64).reshape(-1)
coords = np.load(pipeline / "model" / "zdim_4" / "latent_coords_noreg.npy").astype(np.float64)
if coords.shape[0] != assignments.size:
    raise RuntimeError(f"Embedding/state length mismatch: {coords.shape[0]} vs {assignments.size}")

manifest = {
    "run_dir": str(run),
    "pipeline_dir": str(pipeline),
    "coords": str(pipeline / "model" / "zdim_4" / "latent_coords_noreg.npy"),
    "precision": str(pipeline / "model" / "zdim_4" / "latent_precision_noreg.npy"),
    "regularization": "noreg",
    "target_states": states,
    "targets": [],
}
for state in states:
    mask = assignments == state
    if not np.any(mask):
        raise RuntimeError(f"No particles assigned to state {state}")
    target = np.mean(coords[mask], axis=0, dtype=np.float64).astype(np.float32).reshape(1, -1)
    path = target_root / f"state{state:04d}.txt"
    np.savetxt(path, target)
    target_info = {
        "state": state,
        "target_state_count": int(mask.sum()),
        "target_path": str(path),
        "target_point": target.reshape(-1).astype(float).tolist(),
    }
    (target_root / f"state{state:04d}.json").write_text(json.dumps(target_info, indent=2) + "\n")
    manifest["targets"].append(target_info)

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
    --run-dir "$RUN_DIR" \
    --compute-state-dir "$out_dir" \
    --target-state "$state" \
    --mask "$BROAD_MASK" \
    --voxel-size 1.25 \
    --out-dir "$metrics_dir"
done

cat > "$OUT_ROOT/manifest.txt" <<EOF
run_dir=$RUN_DIR
pipeline_dir=$PIPELINE_DIR
compute_root=$COMPUTE_ROOT
metrics_root=$METRICS_ROOT
target_root=$TARGET_ROOT
broad_mask=$BROAD_MASK
regularization=noreg
zdim=4
embedding_option=cov_dist
compute_state_mode=standard
states=$TARGET_STATES_STR
EOF

echo "DONE=$(date -Is)"
echo "OUT_ROOT=$OUT_ROOT"
echo "TARGET_ROOT=$TARGET_ROOT"
echo "COMPUTE_ROOT=$COMPUTE_ROOT"
echo "METRICS_ROOT=$METRICS_ROOT"
