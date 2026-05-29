#!/usr/bin/env bash
# Real RECOVAR pipeline on the 100k spike full-atom noise10/B100 dataset,
# followed by a small compute_state parameter sweep.

#SBATCH --job-name=spike-recovar-100k-sanity
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sanity_100k_noise10_b100_20260529/slurm/%x-%j.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sanity_100k_noise10_b100_20260529/slurm/%x-%j.err

set -euo pipefail

WORKDIR="${WORKDIR:-/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar}"
SOURCE_RUN="${SOURCE_RUN:-/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise10_b100_parallel_20260516/n00100000/runs/n00100000_seed0000}"
BENCH_ROOT="${BENCH_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sanity_100k_noise10_b100_20260529}"
RECOVAR_ROOT="$BENCH_ROOT/recovar"
PIPELINE_DIR="$RECOVAR_ROOT/06_pipeline_true_recovar_zdim1_4"
DATASET_DIR="$SOURCE_RUN/03_dataset"
MASK_DIR="$SOURCE_RUN/05_masks"
TARGET_STATES_STR="${TARGET_STATES:-0 25 50}"
VARIANTS_STR="${RECOVAR_VARIANTS:-zdim1_reg_covdist zdim1_noreg_covdist zdim4_reg_covdist zdim4_noreg_covdist zdim4_reg_dist}"
AGENT_ID="spike-recovar-100k-${SLURM_JOB_ID:-manual}"

mkdir -p "$BENCH_ROOT/slurm" "$RECOVAR_ROOT" \
  "/scratch/gpfs/GILLES/mg6942/tmp/$AGENT_ID" \
  "/scratch/gpfs/GILLES/mg6942/pixi_home/$AGENT_ID" \
  "/scratch/gpfs/GILLES/mg6942/rattler_cache/$AGENT_ID" \
  "/scratch/gpfs/GILLES/mg6942/xdg_cache/$AGENT_ID"

cd "$WORKDIR"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TMPDIR="/scratch/gpfs/GILLES/mg6942/tmp/$AGENT_ID"
export PIXI_HOME="/scratch/gpfs/GILLES/mg6942/pixi_home/$AGENT_ID"
export RATTLER_CACHE_DIR="/scratch/gpfs/GILLES/mg6942/rattler_cache/$AGENT_ID"
export XDG_CACHE_HOME="/scratch/gpfs/GILLES/mg6942/xdg_cache/$AGENT_ID"
export JAX_COMPILATION_CACHE_DIR="$TMPDIR/jax_cache"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

echo "START=$(date -Is)"
echo "JOB_ID=${SLURM_JOB_ID:-manual}"
echo "HOST=$(hostname)"
echo "WORKDIR=$WORKDIR"
echo "SOURCE_RUN=$SOURCE_RUN"
echo "BENCH_ROOT=$BENCH_ROOT"
echo "PIPELINE_DIR=$PIPELINE_DIR"
echo "TARGET_STATES=$TARGET_STATES_STR"
echo "RECOVAR_VARIANTS=$VARIANTS_STR"
echo "TMPDIR=$TMPDIR"
echo "COMMIT=$(git rev-parse HEAD)"
echo "BRANCH=$(git rev-parse --abbrev-ref HEAD)"
echo "STATUS=$(git status --short | tr '\n' ';')"
nvidia-smi || true

set +u
module load cudatoolkit/13.2 >/dev/null 2>&1 || module load cudatoolkit/12.8 >/dev/null 2>&1 || true
set -u
echo "NVCC=$(command -v nvcc || true)"
export RECOVAR_CUDA_LIB="$WORKDIR/recovar/cuda/libcuda_backproject.so"

pixi install
pixi run install-recovar
PIXI_PY="$(pixi run which python)"
"$PIXI_PY" -m pip uninstall -y recovar || true
"$PIXI_PY" -m pip install -e . --no-deps --no-build-isolation --ignore-installed
PYTHON="$PIXI_PY" make -C recovar/cuda clean all

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

for required in \
  "$DATASET_DIR/particles.star" \
  "$DATASET_DIR/poses.pkl" \
  "$DATASET_DIR/ctf.pkl" \
  "$DATASET_DIR/state_assignment.npy" \
  "$MASK_DIR/volume_mask_union.mrc" \
  "$MASK_DIR/focus_mask_moving.mrc"
do
  if [[ ! -f "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 2
  fi
done

if [[ ! -f "$PIPELINE_DIR/model/zdim_1/latent_coords.npy" \
   || ! -f "$PIPELINE_DIR/model/zdim_1/latent_coords_noreg.npy" \
   || ! -f "$PIPELINE_DIR/model/zdim_4/latent_coords.npy" \
   || ! -f "$PIPELINE_DIR/model/zdim_4/latent_coords_noreg.npy" \
   || "${OVERWRITE_PIPELINE:-0}" == "1" ]]; then
  rm -rf "$PIPELINE_DIR"
  pipeline_cmd=(
    "$PIXI_PY" -m recovar.commands.pipeline
    "$DATASET_DIR/particles.star"
    --poses "$DATASET_DIR/poses.pkl"
    --ctf "$DATASET_DIR/ctf.pkl"
    --mask "$MASK_DIR/volume_mask_union.mrc"
    --focus-mask "$MASK_DIR/focus_mask_moving.mrc"
    --keep-input-mask
    --zdim 1,4
    --no-correct-contrast
    --no-do-over-with-contrast
    --noise-model radial
    --uninvert-data false
    --lazy
    -o "$PIPELINE_DIR"
  )
  echo "PIPELINE_COMMAND=${pipeline_cmd[*]}"
  /usr/bin/time -v "${pipeline_cmd[@]}"
else
  echo "Reusing completed true pipeline: $PIPELINE_DIR"
fi

"$PIXI_PY" - "$SOURCE_RUN" "$PIPELINE_DIR" "$RECOVAR_ROOT" "$TARGET_STATES_STR" "$VARIANTS_STR" <<'PY'
from pathlib import Path
import json
import pickle
import sys

import numpy as np

source_run = Path(sys.argv[1])
pipeline = Path(sys.argv[2])
out_root = Path(sys.argv[3])
target_states = [int(x) for x in sys.argv[4].split()]
variants = sys.argv[5].split()

params = pickle.load(open(pipeline / "model" / "params.pkl", "rb"))
if isinstance(params, dict) and params.get("is_oracle_pipeline_output", False):
    raise RuntimeError(f"{pipeline} is marked as oracle-generated, expected true recovar pipeline")

assignments = np.load(source_run / "03_dataset" / "state_assignment.npy").astype(np.int64).reshape(-1)
manifest = {
    "source_run": str(source_run),
    "pipeline": str(pipeline),
    "state_assignment": str(source_run / "03_dataset" / "state_assignment.npy"),
    "target_states": target_states,
    "variants": variants,
    "targets": [],
}

for variant in variants:
    if variant.startswith("zdim1_"):
        zdim = 1
    elif variant.startswith("zdim4_"):
        zdim = 4
    else:
        raise ValueError(f"Unknown variant name: {variant}")
    regularization = "noreg" if "_noreg_" in variant else "reg"
    coords_name = "latent_coords_noreg.npy" if regularization == "noreg" else "latent_coords.npy"
    precision_name = "latent_precision_noreg.npy" if regularization == "noreg" else "latent_precision.npy"
    coords_path = pipeline / "model" / f"zdim_{zdim}" / coords_name
    precision_path = pipeline / "model" / f"zdim_{zdim}" / precision_name
    coords = np.load(coords_path).astype(np.float64)
    if coords.shape[0] != assignments.shape[0]:
        raise RuntimeError(f"{coords_path} length {coords.shape[0]} != assignments length {assignments.shape[0]}")
    for state in target_states:
        mask = assignments == state
        if not np.any(mask):
            raise RuntimeError(f"No particles assigned to state {state}")
        target = np.nanmean(coords[mask], axis=0, dtype=np.float64).astype(np.float32).reshape(1, -1)
        target_path = out_root / "targets" / variant / f"state{state:04d}.txt"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(target_path, target)
        metadata = {
            "variant": variant,
            "zdim": zdim,
            "regularization": regularization,
            "coords_path": str(coords_path),
            "precision_path": str(precision_path),
            "target_state": state,
            "target_state_count": int(mask.sum()),
            "target_point": target.reshape(-1).astype(float).tolist(),
            "target_path": str(target_path),
        }
        meta_path = target_path.with_suffix(".json")
        meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        manifest["targets"].append(metadata)

(out_root / "recovar_param_sanity_targets_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

for variant in $VARIANTS_STR; do
  extra_args=()
  case "$variant" in
    zdim1_reg_covdist)
      extra_args+=(--zdim1 --embedding-option cov_dist)
      ;;
    zdim1_noreg_covdist)
      extra_args+=(--zdim1 --no-z-regularization --embedding-option cov_dist)
      ;;
    zdim4_reg_covdist)
      extra_args+=(--embedding-option cov_dist)
      ;;
    zdim4_noreg_covdist)
      extra_args+=(--no-z-regularization --embedding-option cov_dist)
      ;;
    zdim4_reg_dist)
      extra_args+=(--embedding-option dist)
      ;;
    *)
      echo "Unknown RECOVAR variant: $variant" >&2
      exit 2
      ;;
  esac

  for state in $TARGET_STATES_STR; do
    state4="$(printf '%04d' "$state")"
    target_path="$RECOVAR_ROOT/targets/$variant/state${state4}.txt"
    out_dir="$RECOVAR_ROOT/compute_state/$variant/state${state4}"
    metrics_dir="$RECOVAR_ROOT/metrics/$variant/state${state4}"
    if [[ -f "$out_dir/state000_unfil.mrc" && "${OVERWRITE_COMPUTE_STATE:-0}" != "1" ]]; then
      echo "Reusing completed compute_state: $out_dir"
    else
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
        --lazy
      )
      cmd+=("${extra_args[@]}")
      echo "COMPUTE_STATE_COMMAND=${cmd[*]}"
      /usr/bin/time -v "${cmd[@]}"
    fi

    "$PIXI_PY" scripts/experiments/spike_fullatom_state_sweeps/plot_compute_state_shell_metrics.py \
      --run-dir "$SOURCE_RUN" \
      --compute-state-dir "$out_dir" \
      --target-state "$state" \
      --mask "$MASK_DIR/focus_mask_moving.mrc" \
      --voxel-size 1.25 \
      --out-dir "$metrics_dir"
  done
done

echo "DONE=$(date -Is)"
echo "PIPELINE_DIR=$PIPELINE_DIR"
echo "RECOVAR_ROOT=$RECOVAR_ROOT"
echo "TARGET_MANIFEST=$RECOVAR_ROOT/recovar_param_sanity_targets_manifest.json"
