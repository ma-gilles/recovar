#!/usr/bin/env bash
#SBATCH --job-name=spike-n10-1M-tmp
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=900G
#SBATCH --time=22:00:00
#SBATCH --exclusive
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_consistency_n10_1M_20260515/slurm/spike_consistency_n10_1M_%j.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_consistency_n10_1M_20260515/slurm/spike_consistency_n10_1M_%j.err

set -euo pipefail
set -x

BASE="/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_consistency_n10_1M_20260515"
WORKDIR="/scratch/gpfs/GILLES/mg6942/recovar_wt_spike_consistency_20260514_104555"
SWEEP_ROOT="${BASE}/output/spike_consistency_grid256_noise10_n1M_bfac80_zdim4_job${SLURM_JOB_ID}"
MASK="/scratch/gpfs/GILLES/mg6942/runs/spike_grid128_box320_noctf_noise0p01_n200k_raw_index_embsigma100_20260514/05_masks/mask_crafted_level0p0126_cosine3_128.mrc"
LOG="${BASE}/spike_consistency_n10_1M_job${SLURM_JOB_ID}.log"

mkdir -p "${BASE}/slurm" "${BASE}/tmp/${SLURM_JOB_ID}" "${BASE}/pixi_home/${SLURM_JOB_ID}" \
  "${BASE}/rattler_cache/${SLURM_JOB_ID}" "${BASE}/xdg_cache/${SLURM_JOB_ID}" "$SWEEP_ROOT"

cd "$WORKDIR"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="${BASE}/tmp/${SLURM_JOB_ID}"
export PIXI_HOME="${BASE}/pixi_home/${SLURM_JOB_ID}"
export RATTLER_CACHE_DIR="${BASE}/rattler_cache/${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${BASE}/xdg_cache/${SLURM_JOB_ID}"
export JAX_PLATFORM_NAME=gpu
export RECOVAR_CACHE_DIR=

{
  echo "START $(date -Is)"
  echo "host=$(hostname)"
  echo "workdir=$WORKDIR"
  echo "sweep_root=$SWEEP_ROOT"
  echo "mask=$MASK"
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
PIXI_PY="$(pixi run which python)"

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

/usr/bin/time -v "$PIXI_PY" -m recovar.commands.spike_consistency \
  --output-dir "$SWEEP_ROOT" \
  --mask "$MASK" \
  --pdb-dir /scratch/gpfs/GILLES/mg6942/spike_pdb_motion \
  --grid-size 256 \
  --render-voxel-size 1.25 \
  --n-images-list 1000000 \
  --noise-level 10.0 \
  --bfactor 80 \
  --zdim 4 \
  --seed 0 \
  --lazy \
  --low-memory-option \
  2>&1 | tee -a "$LOG"

{
  echo "DONE $(date -Is)"
  echo "SWEEP_ROOT=$SWEEP_ROOT"
  echo "LOG=$LOG"
} | tee -a "$LOG"
