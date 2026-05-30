#!/usr/bin/env bash
# Generate the missing 10k spike full-atom noise1/B80 dataset and true RECOVAR
# pipeline output. This intentionally avoids low-memory flags.

#SBATCH --job-name=spike-n1-10k-pipe
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=900G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise1_b80_20260530/slurm/%x-%j.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise1_b80_20260530/slurm/%x-%j.err

set -euo pipefail

WORKDIR="${WORKDIR:-/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_kernel_dev2_rebase}"
SOURCE_ROOT="${SOURCE_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise1_b80_true_oracle_sweep_20260527}"
SHARED_RAW="${SHARED_RAW:-/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise1_b80_true_oracle_sweep_20260527_shared/01_raw_volumes}"
PDB_DIR="${PDB_DIR:-/projects/CRYOEM/singerlab/mg6942/spike_morph_pdbs}"
N_IMAGES=10000
N_LABEL="$(printf 'n%08d' "$N_IMAGES")"
RUN_DIR="$SOURCE_ROOT/$N_LABEL/runs/${N_LABEL}_seed0000"
PIPELINE_DIR="$RUN_DIR/06_pipeline_true_recovar_h100_fullmem_movingfocus_20260530"
AGENT_ID="spike-n1-10k-pipe-${SLURM_JOB_ID:-manual}"
BENCH_ROOT="/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise1_b80_20260530"

mkdir -p \
  "$BENCH_ROOT/slurm" \
  "$BENCH_ROOT/tmp/$AGENT_ID" \
  "$BENCH_ROOT/pixi_home/$AGENT_ID" \
  "$BENCH_ROOT/rattler_cache/$AGENT_ID" \
  "$BENCH_ROOT/xdg_cache/$AGENT_ID" \
  "$RUN_DIR/01_raw_volumes" \
  "$SHARED_RAW"

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
echo "JOB_ID=${SLURM_JOB_ID:-manual}"
echo "RUN_DIR=$RUN_DIR"
echo "PIPELINE_DIR=$PIPELINE_DIR"
echo "SHARED_RAW=$SHARED_RAW"
echo "PDB_DIR=$PDB_DIR"
echo "COMMIT=$(git rev-parse HEAD)"
echo "BRANCH=$(git rev-parse --abbrev-ref HEAD)"
echo "STATUS=$(git status --short | tr '\n' ';')"
nvidia-smi || true

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

repo = Path.cwd().resolve()
recovar_file = Path(recovar.__file__).resolve()
jax_file = Path(jax.__file__).resolve()
print("recovar", recovar_file, flush=True)
print("jax", jax_file, flush=True)
print("devices", jax.devices(), flush=True)
assert str(recovar_file).startswith(str(repo) + "/")
assert ".pixi/envs/default/" in str(jax_file)
if not any(getattr(device, "platform", "") == "gpu" for device in jax.devices()):
    raise RuntimeError("JAX did not see a GPU")
PY

raw_count="$(find "$SHARED_RAW" -maxdepth 1 -name 'vol[0-9][0-9][0-9][0-9].mrc' 2>/dev/null | wc -l)"
if [[ "$raw_count" -ne 100 ]]; then
  echo "Rendering shared B80 raw volumes to $SHARED_RAW"
  /usr/bin/time -v "$PIXI_PY" -m recovar.commands.render_spike_morph_volumes \
    "$(dirname "$SHARED_RAW")" \
    --pdb-dir "$PDB_DIR" \
    --grid-size 256 \
    --voxel-size 1.25 \
    --bfactor 80
fi

raw_count="$(find "$SHARED_RAW" -maxdepth 1 -name 'vol[0-9][0-9][0-9][0-9].mrc' 2>/dev/null | wc -l)"
if [[ "$raw_count" -ne 100 ]]; then
  echo "Expected 100 raw volumes in $SHARED_RAW, found $raw_count" >&2
  exit 2
fi

ln -sfn "$SHARED_RAW"/vol*.mrc "$RUN_DIR/01_raw_volumes"/

if [[ ! -f "$RUN_DIR/03_dataset/particles.star" || "${OVERWRITE_DATASET:-0}" == "1" ]]; then
  /usr/bin/time -v "$PIXI_PY" -m recovar.commands.spike_walkthrough \
    --output-dir "$RUN_DIR" \
    --grid-size 256 \
    --render-voxel-size 1.25 \
    --voxel-size 1.25 \
    --render-bfactor 80 \
    --n-images "$N_IMAGES" \
    --noise-level 1 \
    --bfactor 0 \
    --zdim 4 \
    --target-state 50 \
    --seed 0 \
    --lazy \
    --pipeline-gpu-memory 80 \
    --compute-state-save-all-estimates \
    --no-kernel-report \
    --no-use-oracle-pipeline
else
  echo "Reusing existing 10k dataset: $RUN_DIR/03_dataset"
fi

if [[ ! -f "$PIPELINE_DIR/model/params.pkl" || "${OVERWRITE_PIPELINE:-0}" == "1" ]]; then
  rm -rf "$PIPELINE_DIR"
  /usr/bin/time -v "$PIXI_PY" -m recovar.commands.pipeline \
    "$RUN_DIR/03_dataset/particles.star" \
    --poses "$RUN_DIR/03_dataset/poses.pkl" \
    --ctf "$RUN_DIR/03_dataset/ctf.pkl" \
    --mask "$RUN_DIR/05_masks/volume_mask_union.mrc" \
    --focus-mask "$RUN_DIR/05_masks/focus_mask_moving.mrc" \
    --zdim 4 \
    --no-correct-contrast \
    --no-do-over-with-contrast \
    --noise-model radial \
    --uninvert-data false \
    --lazy \
    -o "$PIPELINE_DIR"
else
  echo "Reusing existing true pipeline: $PIPELINE_DIR"
fi

cat > "$RUN_DIR/noise1_b80_10k_true_pipeline_manifest.txt" <<EOF
run_dir=$RUN_DIR
pipeline_dir=$PIPELINE_DIR
n_images=$N_IMAGES
noise_level=1
render_bfactor=80
zdim=4
focus_mask=$RUN_DIR/05_masks/focus_mask_moving.mrc
volume_mask=$RUN_DIR/05_masks/volume_mask_union.mrc
low_memory_flags=none
EOF

echo "DONE=$(date -Is)"
echo "RUN_DIR=$RUN_DIR"
echo "PIPELINE_DIR=$PIPELINE_DIR"
