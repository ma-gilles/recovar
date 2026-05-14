#!/bin/bash -l
#SBATCH --job-name=sp128-noctf-n0p01-emb10-20k
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/spike128_rawidx_noctf_noise0p01_20k_emb10_%j.out
#SBATCH --error=/scratch/gpfs/GILLES/mg6942/slurmo/spike128_rawidx_noctf_noise0p01_20k_emb10_%j.err

set -euo pipefail
set -x

REPO="/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar"
RUN_DIR="/scratch/gpfs/GILLES/mg6942/runs/spike_grid128_box320_noctf_noise0p01_n20k_raw_index_embsigma10_20260514"
REFERENCE_RUN="/scratch/gpfs/GILLES/mg6942/runs/spike_grid128_box320_noise30_n300k_raw_index_gtnoisy_20260513"
LOG="${RUN_DIR}/run_raw_index_noctf_noise0p01_20k_embsigma10_standard_deconv_report.log"

mkdir -p "$RUN_DIR" /scratch/gpfs/GILLES/mg6942/slurmo
cd "$REPO"

export TMPDIR="/scratch/gpfs/GILLES/mg6942/tmp/${SLURM_JOB_ID}"
export PIXI_HOME="/scratch/gpfs/GILLES/mg6942/pixi_home/${SLURM_JOB_ID}"
export RATTLER_CACHE_DIR="/scratch/gpfs/GILLES/mg6942/rattler_cache/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$PIXI_HOME" "$RATTLER_CACHE_DIR"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORM_NAME=gpu

{
  echo "START $(date -Is)"
  echo "host=$(hostname)"
  echo "repo=$REPO"
  echo "commit=$(git rev-parse HEAD)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "run_dir=$RUN_DIR"
  echo "reference_run=$REFERENCE_RUN"
  echo "grid_size=128"
  echo "voxel_size=2.5"
  echo "n_states=100"
  echo "n_images=20000"
  echo "no_ctf=true"
  echo "image_noise_level=0.01"
  echo "stream_simulation=true"
  echo "compute_state_lazy=true"
  echo "particle_stack_dtype=float32"
  echo "pc_project=0"
  echo "true_embedding=state_index_scaled_to_previous_raw_index_embedding"
  echo "embedding_noise_std=10"
  echo "embedding_seed=20260513"
  nvidia-smi || true
} | tee "$LOG"

module load cudatoolkit/12.8 || true
pixi install 2>&1 | tee -a "$LOG"
pixi run install-recovar 2>&1 | tee -a "$LOG"
PIXI_PY="$(pixi run which python)"
"$PIXI_PY" -m pip uninstall -y recovar 2>&1 | tee -a "$LOG" || true
"$PIXI_PY" -m pip install -e . --no-deps --no-build-isolation --ignore-installed 2>&1 | tee -a "$LOG"

PY_SITE="$("$PIXI_PY" -c 'import site; print(site.getsitepackages()[0])')"
export LD_LIBRARY_PATH="$PY_SITE/nvidia/cuda_runtime/lib:$PY_SITE/nvidia/cuda_nvrtc/lib:$PY_SITE/nvidia/cublas/lib:$PY_SITE/nvidia/cusolver/lib:$PY_SITE/nvidia/cusparse/lib:$PY_SITE/nvidia/cudnn/lib:$PY_SITE/nvidia/cufft/lib:$PY_SITE/nvidia/nvjitlink/lib:$PY_SITE/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
PYTHON="$PIXI_PY" make -C recovar/cuda clean all 2>&1 | tee -a "$LOG"
export RECOVAR_CUDA_LIB="$REPO/recovar/cuda/libcuda_backproject.so"

"$PIXI_PY" - <<'PY' 2>&1 | tee -a "$LOG"
import pathlib
import jax
import recovar

repo = pathlib.Path.cwd().resolve()
recovar_file = pathlib.Path(recovar.__file__).resolve()
jax_file = pathlib.Path(jax.__file__).resolve()
print("recovar.__file__=", recovar_file)
print("jax.__file__=", jax_file)
print("jax.devices()=", jax.devices())
print("jax.default_backend()=", jax.default_backend())
assert str(recovar_file).startswith(str(repo) + "/")
assert ".pixi/envs/default/" in str(jax_file)
assert jax.default_backend() == "gpu"
PY

"$PIXI_PY" -m py_compile \
  recovar/commands/spike_walkthrough.py \
  recovar/commands/compute_state.py \
  recovar/commands/spike_kernel_report.py \
  _agent_scratch/run_spike128_raw_index_gtnoise.py

/usr/bin/time -v "$PIXI_PY" _agent_scratch/run_spike128_raw_index_gtnoise.py \
  --run-dir "$RUN_DIR" \
  --reference-raw-dir "${REFERENCE_RUN}/01_raw_volumes" \
  --reference-gt-metadata "${REFERENCE_RUN}/index_noisy_embedding_metadata.json" \
  --grid-size 128 \
  --voxel-size 2.5 \
  --n-states 100 \
  --n-images 20000 \
  --noise-level 0.01 \
  --noise-model radial1 \
  --no-ctf \
  --render-bfactor 100 \
  --target-state 50 \
  --embedding-seed 20260513 \
  --embedding-noise-std 10 \
  --stream-simulation \
  --compute-state-lazy \
  --crafted-mask /scratch/gpfs/GILLES/mg6942/uploads/mask_crafted.mrc \
  --crafted-mask-level 0.0126 \
  --crafted-mask-cosine-width 3 \
  2>&1 | tee -a "$LOG"

{
  echo "DONE $(date -Is)"
  echo "RUN_DIR=$RUN_DIR"
  echo "LOG=$LOG"
  echo "TARGET_VOL=${RUN_DIR}/04_ground_truth/gt_vol0050.mrc"
  echo "MASK=${RUN_DIR}/05_masks/mask_crafted_level0p0126_cosine3_128.mrc"
  echo "REPORT_OUT=${RUN_DIR}/08_kernel_report_mask_crafted_level0p0126_cosine3_lazy"
} | tee -a "$LOG"
