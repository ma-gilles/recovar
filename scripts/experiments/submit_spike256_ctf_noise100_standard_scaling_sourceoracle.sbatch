#!/bin/bash
# Submit standard-only compute_state scaling runs for the 256-box spike
# source-oracle experiment. This is meant to compare image counts against the
# existing 1M standard result without running deconvolved or local-poly tails.
#
# Usage:
#   sbatch --parsable scripts/experiments/submit_spike256_ctf_noise100_standard_scaling_sourceoracle.sbatch

#SBATCH --job-name=sp256-n100-std-scale
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_grid256_box320_ctf_noise100_sourceoracle_standard_scaling_20260515/slurm/spike256_noise100_stdscale_%A_%a.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_grid256_box320_ctf_noise100_sourceoracle_standard_scaling_20260515/slurm/spike256_noise100_stdscale_%A_%a.err

set -euo pipefail
set -x

N_IMAGES_VALUES=(30000 100000 300000)
N_LABEL_VALUES=(30k 100k 300k)

N_IMAGES="${N_IMAGES_VALUES[$SLURM_ARRAY_TASK_ID]}"
N_LABEL="${N_LABEL_VALUES[$SLURM_ARRAY_TASK_ID]}"

BASE_ROOT="${BASE_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_grid256_box320_ctf_noise100_sourceoracle_standard_scaling_20260515}"
BASE="${BASE:-${BASE_ROOT}/n${N_LABEL}}"
RUN_DIR="${RUN_DIR:-${BASE}/run}"
WORKDIR="${WORKDIR:-/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar}"
PYTHON="${PYTHON:-${WORKDIR}/.pixi/envs/default/bin/python}"
REFERENCE_RAW_DIR="${REFERENCE_RAW_DIR:-/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_consistency_n10_1M_20260515/output/spike_consistency_grid256_noise10_n1M_bfac80_zdim4_job8271045/shared/01_raw_volumes}"
CRAFTED_MASK="${CRAFTED_MASK:-/scratch/gpfs/GILLES/mg6942/uploads/mask_crafted.mrc}"
LOG="${LOG:-${BASE}/spike256_noise100_sourceoracle_standard_n${N_LABEL}_job${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log}"

mkdir -p \
  "${BASE_ROOT}/slurm" \
  "${BASE}/tmp/${SLURM_JOB_ID}" \
  "${BASE}/pixi_home/${SLURM_JOB_ID}" \
  "${BASE}/rattler_cache/${SLURM_JOB_ID}" \
  "${BASE}/xdg_cache/${SLURM_JOB_ID}" \
  "$RUN_DIR"

cd "$WORKDIR"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORM_NAME=gpu
export RECOVAR_CACHE_DIR=
export TMPDIR="${BASE}/tmp/${SLURM_JOB_ID}"
export PIXI_HOME="${BASE}/pixi_home/${SLURM_JOB_ID}"
export RATTLER_CACHE_DIR="${BASE}/rattler_cache/${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${BASE}/xdg_cache/${SLURM_JOB_ID}"

{
  echo "START $(date -Is)"
  echo "STANDARD_ONLY_SCALING_RUN"
  echo "n_images=$N_IMAGES"
  echo "n_label=$N_LABEL"
  echo "host=$(hostname)"
  echo "workdir=$WORKDIR"
  echo "run_dir=$RUN_DIR"
  echo "reference_raw_dir=$REFERENCE_RAW_DIR"
  echo "crafted_mask=$CRAFTED_MASK"
  echo "tmpdir=$TMPDIR"
  echo "pixi_home=$PIXI_HOME"
  echo "rattler_cache=$RATTLER_CACHE_DIR"
  echo "xdg_cache_home=$XDG_CACHE_HOME"
  echo "commit=$(git rev-parse HEAD)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "git_status_short=$(git status --short | tr '\n' ';')"
  nvidia-smi || true
} | tee "$LOG"

source /usr/share/Modules/init/bash
module purge
module load cudatoolkit/12.9
echo "nvcc: $(which nvcc 2>/dev/null || echo MISSING)" | tee -a "$LOG"

pixi install 2>&1 | tee -a "$LOG"

"$PYTHON" - <<'PY' 2>&1 | tee -a "$LOG"
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

/usr/bin/time -v "$PYTHON" _agent_scratch/run_spike128_raw_index_gtnoise.py \
  --run-dir "$RUN_DIR" \
  --reference-raw-dir "$REFERENCE_RAW_DIR" \
  --grid-size 256 \
  --voxel-size 1.25 \
  --n-states 100 \
  --n-images "$N_IMAGES" \
  --noise-level 100 \
  --noise-model radial1 \
  --render-bfactor 100 \
  --target-state 50 \
  --embedding-source source_oracle \
  --embedding-seed 20260513 \
  --stream-simulation \
  --crafted-mask "$CRAFTED_MASK" \
  --crafted-mask-level 0.0126 \
  --crafted-mask-cosine-width 3 \
  --standard-only \
  --skip-gt-polynomial-sweep \
  2>&1 | tee -a "$LOG"

{
  echo "DONE $(date -Is)"
  echo "N_IMAGES=$N_IMAGES"
  echo "RUN_DIR=$RUN_DIR"
  echo "LOG=$LOG"
} | tee -a "$LOG"
