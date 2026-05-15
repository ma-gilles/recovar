#!/bin/bash
#SBATCH --job-name=sp128-n1-setup
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/spike128_ctf_noise1_setup_%j.out
#SBATCH --error=/scratch/gpfs/GILLES/mg6942/slurmo/spike128_ctf_noise1_setup_%j.err

set -euo pipefail

cd /scratch/gpfs/GILLES/mg6942/recovar_dev/recovar

export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export RECOVAR_CACHE_DIR=
export TMPDIR="/scratch/gpfs/GILLES/mg6942/tmp/${SLURM_JOB_ID}"
export PIXI_HOME="/scratch/gpfs/GILLES/mg6942/pixi_home/${SLURM_JOB_ID}"
export RATTLER_CACHE_DIR="/scratch/gpfs/GILLES/mg6942/rattler_cache/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$PIXI_HOME" "$RATTLER_CACHE_DIR" /scratch/gpfs/GILLES/mg6942/slurmo
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV

PYTHON=/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar/.pixi/envs/default/bin/python
RUN_DIR=/scratch/gpfs/GILLES/mg6942/runs/spike_grid128_box320_ctf_noise1_n200k_raw_index_embsigma316_20260515
EMB_SIGMA=316.22776601683796
TAG=noise1_d5_i2_q5_monomial_none
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/run_setup_${TAG}.log"

{
  date -Is
  "$PYTHON" - <<'PY'
import pathlib, recovar, jax
repo = pathlib.Path.cwd().resolve()
print("recovar.__file__=", recovar.__file__)
print("jax.__file__=", jax.__file__)
print("jax.devices=", jax.devices())
assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo) + "/")
assert ".pixi/envs/default/" in str(pathlib.Path(jax.__file__).resolve())
PY
  echo "RUN_DIR=$RUN_DIR"
  echo "EMB_SIGMA=$EMB_SIGMA"
  echo "TAG=$TAG"
  /usr/bin/time -v "$PYTHON" _agent_scratch/run_spike128_raw_index_gtnoise.py \
    --run-dir "$RUN_DIR" \
    --reference-raw-dir /scratch/gpfs/GILLES/mg6942/runs/spike_grid128_box320_noise30_n300k_raw_index_gtnoisy_20260513/01_raw_volumes \
    --reference-gt-metadata /scratch/gpfs/GILLES/mg6942/runs/spike_grid128_box320_noise30_n300k_raw_index_gtnoisy_20260513/index_noisy_embedding_metadata.json \
    --grid-size 128 \
    --voxel-size 2.5 \
    --n-states 100 \
    --n-images 200000 \
    --noise-level 1 \
    --noise-model radial1 \
    --render-bfactor 100 \
    --target-state 50 \
    --embedding-seed 20260513 \
    --embedding-noise-std "$EMB_SIGMA" \
    --stream-simulation \
    --compute-state-lazy \
    --crafted-mask /scratch/gpfs/GILLES/mg6942/uploads/mask_crafted.mrc \
    --crafted-mask-level 0.0126 \
    --crafted-mask-cosine-width 3 \
    --local-poly-output-tag "$TAG" \
    --local-poly-h-state-grid 1,2,4,8,16,32 \
    --local-poly-em \
    --local-poly-degree 5 \
    --local-poly-em-iterations 2 \
    --local-poly-em-quadrature 5 \
    --local-poly-basis monomial \
    --local-poly-pol-reg-type none \
    --skip-gt-polynomial-sweep
  date -Is
} > "$LOG" 2>&1
