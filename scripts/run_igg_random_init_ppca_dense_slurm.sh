#!/usr/bin/env bash
#SBATCH --job-name=igg-ppca-dense
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=250GB
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/igg-ppca-dense-%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:-/scratch/gpfs/GILLES/mg6942/tmp/ppca_igg_random_dense_20260505}"
GRID_SIZE="${GRID_SIZE:-64}"
N_IMAGES="${N_IMAGES:-5000}"
DENSE_N_IMAGES="${DENSE_N_IMAGES:-}"
NOISE_LEVEL="${NOISE_LEVEL:-0.1}"
SEED="${SEED:-20260505}"
IGG_VOLUME_PREFIX="${IGG_VOLUME_PREFIX:-/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_wt_contrast_multimask/_agent_scratch/pipeline_e2e/IgG-1D/vols/vol}"
IGG_VOLUME_GLOB="${IGG_VOLUME_GLOB:-/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_wt_contrast_multimask/_agent_scratch/pipeline_e2e/IgG-1D/vols/vol*.mrc}"
K_INIT="${K_INIT:-10}"
Q="${Q:-4}"
N_ITERS="${N_ITERS:-2}"
HEALPIX_ORDER="${HEALPIX_ORDER:-1}"
CURRENT_SIZE="${CURRENT_SIZE:-16}"
OFFSET_RANGE_PX="${OFFSET_RANGE_PX:-6}"
OFFSET_STEP_PX="${OFFSET_STEP_PX:-2}"
IMAGE_BATCH_SIZE="${IMAGE_BATCH_SIZE:-50}"
ROTATION_BLOCK_SIZE="${ROTATION_BLOCK_SIZE:-72}"
MSTEP_CHUNK_SIZE="${MSTEP_CHUNK_SIZE:-65536}"
COMPUTE_EMBEDDING="${COMPUTE_EMBEDDING:-1}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-100}"
DENSE_IMAGE_ARGS=()
RUN_N_IMAGES="$N_IMAGES"
if [ -n "$DENSE_N_IMAGES" ]; then
  DENSE_IMAGE_ARGS=(--n-images "$DENSE_N_IMAGES")
  RUN_N_IMAGES="$DENSE_N_IMAGES"
fi

AGENT_ID="igg_ppca_dense_${SLURM_JOB_ID:-manual}"
export TMPDIR="/scratch/gpfs/GILLES/mg6942/tmp/${AGENT_ID}"
export PIXI_HOME="/scratch/gpfs/GILLES/mg6942/pixi_home/${AGENT_ID}"
export RATTLER_CACHE_DIR="/scratch/gpfs/GILLES/mg6942/rattler_cache/${AGENT_ID}"
mkdir -p "$TMPDIR" "$PIXI_HOME" "$RATTLER_CACHE_DIR" /scratch/gpfs/GILLES/mg6942/slurmo "$DATA_ROOT"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=0

cd "$REPO_DIR"

echo "=== Repo provenance ==="
pwd
whoami
git rev-parse HEAD
git branch --show-current
git status --short

echo "=== Environment setup ==="
pixi install
PIXI_PY="$(pixi run which python)"
"$PIXI_PY" -m pip uninstall -y recovar || true
"$PIXI_PY" -m pip install -e . --no-deps --no-build-isolation --ignore-installed
PYTHON="$PIXI_PY" make -C recovar/cuda clean all
pixi run smoke-import-recovar
"$PIXI_PY" - <<'PY'
import pathlib
import jax
import recovar
repo = pathlib.Path.cwd().resolve()
assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo) + "/")
assert ".pixi/envs/default/" in str(pathlib.Path(jax.__file__).resolve())
print(jax.devices())
PY

DATASET_PARENT="${DATA_ROOT}/dataset_g${GRID_SIZE}_n${N_IMAGES}_noise${NOISE_LEVEL}_seed${SEED}"
DATASET_DIR="${DATASET_PARENT}/test_dataset"
if [ ! -f "${DATASET_DIR}/particles.star" ]; then
  echo "=== Generate IgG synthetic dataset ==="
  "$PIXI_PY" -m recovar.commands.make_test_dataset "$DATASET_PARENT" \
    --noise-level "$NOISE_LEVEL" \
    --n-images "$N_IMAGES" \
    --grid-size "$GRID_SIZE" \
    --volume-input "$IGG_VOLUME_PREFIX" \
    --seed "$SEED"
else
  echo "Reusing dataset at ${DATASET_DIR}"
fi

if [ -z "${INIT_NPZ:-}" ]; then
  INIT_DIR="${DATA_ROOT}/init_k${K_INIT}_q${Q}_g${GRID_SIZE}_seed${SEED}"
  INIT_NPZ="${INIT_DIR}/ppca_init.npz"
  if [ ! -f "$INIT_NPZ" ]; then
    echo "=== Prepare random-volume PPCA initializer ==="
    "$PIXI_PY" scripts/prepare_random_volume_ppca_init.py \
      --volume-glob "$IGG_VOLUME_GLOB" \
      --output-dir "$INIT_DIR" \
      --k "$K_INIT" \
      --q "$Q" \
      --target-grid-size "$GRID_SIZE" \
      --seed "$SEED" \
      --frame recovar
  else
    echo "Reusing PPCA init at $INIT_NPZ"
  fi
fi

RUN_DIR="${DATA_ROOT}/dense_ppca_k${K_INIT}_q${Q}_n${RUN_N_IMAGES}_hp${HEALPIX_ORDER}_cs${CURRENT_SIZE}_it${N_ITERS}"
mkdir -p "$RUN_DIR"

echo "=== Dense PPCA run ==="
"$PIXI_PY" scripts/run_ppca_dense_from_init_npz.py \
  --data-star "${DATASET_DIR}/particles.star" \
  --simulation-info "${DATASET_DIR}/simulation_info.pkl" \
  --init-npz "$INIT_NPZ" \
  --output-dir "$RUN_DIR" \
  --q "$Q" \
  --n-iters "$N_ITERS" \
  "${DENSE_IMAGE_ARGS[@]}" \
  --healpix-order "$HEALPIX_ORDER" \
  --offset-range-px "$OFFSET_RANGE_PX" \
  --offset-step-px "$OFFSET_STEP_PX" \
  --current-size "$CURRENT_SIZE" \
  --image-batch-size "$IMAGE_BATCH_SIZE" \
  --rotation-block-size "$ROTATION_BLOCK_SIZE" \
  --mstep-chunk-size "$MSTEP_CHUNK_SIZE" \
  --save-mrc

if [ "$COMPUTE_EMBEDDING" = "1" ]; then
  echo "=== Best-pose PPCA embedding ==="
  "$PIXI_PY" scripts/compute_ppca_best_pose_embedding.py \
    --data-star "${DATASET_DIR}/particles.star" \
    --simulation-info "${DATASET_DIR}/simulation_info.pkl" \
    --ppca-result-npz "${RUN_DIR}/final_ppca_dense.npz" \
    --output-dir "${RUN_DIR}/embedding_best_pose" \
    --healpix-order "$HEALPIX_ORDER" \
    --offset-range-px "$OFFSET_RANGE_PX" \
    --offset-step-px "$OFFSET_STEP_PX" \
    --current-size "$CURRENT_SIZE" \
    --image-batch-size "$EMBEDDING_BATCH_SIZE"
fi

echo "=== Completed ==="
echo "DATASET_DIR=$DATASET_DIR"
echo "INIT_NPZ=$INIT_NPZ"
echo "RUN_DIR=$RUN_DIR"
