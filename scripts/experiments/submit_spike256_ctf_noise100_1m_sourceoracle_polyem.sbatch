#!/bin/bash
# Submit the 1M-image spike source-oracle/poly-EM experiment used for the
# deconvolved-kernel regression comparison.
#
# Usage:
#   sbatch --parsable scripts/experiments/submit_spike256_ctf_noise100_1m_sourceoracle_polyem.sbatch
#
# Optional overrides:
#   BASE=/scratch/gpfs/.../my_run WORKDIR=/path/to/recovar \
#     sbatch scripts/experiments/submit_spike256_ctf_noise100_1m_sourceoracle_polyem.sbatch

#SBATCH --job-name=KEEP-sp256-n100-1m
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=900G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_grid256_box320_ctf_noise100_n1m_sourceoracle_20260515/slurm/spike256_noise100_sourceoracle_%j.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_grid256_box320_ctf_noise100_n1m_sourceoracle_20260515/slurm/spike256_noise100_sourceoracle_%j.err

set -euo pipefail
set -x

BASE="${BASE:-/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_grid256_box320_ctf_noise100_n1m_sourceoracle_20260515}"
RUN_DIR="${RUN_DIR:-${BASE}/run}"
WORKDIR="${WORKDIR:-/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar}"
PYTHON="${PYTHON:-${WORKDIR}/.pixi/envs/default/bin/python}"
REFERENCE_RAW_DIR="${REFERENCE_RAW_DIR:-/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_consistency_n10_1M_20260515/output/spike_consistency_grid256_noise10_n1M_bfac80_zdim4_job8271045/shared/01_raw_volumes}"
CRAFTED_MASK="${CRAFTED_MASK:-/scratch/gpfs/GILLES/mg6942/uploads/mask_crafted.mrc}"
LOG="${LOG:-${BASE}/spike256_noise100_sourceoracle_1m_job${SLURM_JOB_ID}.log}"

mkdir -p \
  "${BASE}/slurm" \
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
  echo "ACTIVE_HARDENED_RERUN_DO_NOT_CANCEL_WITHOUT_CONFIRMING_WITH_MG"
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
  --n-images 1000000 \
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
  --local-poly-output-tag sourceoracle_noise100_d5_i2_q5_monomial_none \
  --local-poly-h-state-grid 1,2,4,8,16,32 \
  --local-poly-em \
  --local-poly-degree 5 \
  --local-poly-em-iterations 2 \
  --local-poly-em-quadrature 5 \
  --local-poly-basis monomial \
  --local-poly-pol-reg-type none \
  --skip-gt-polynomial-sweep \
  2>&1 | tee -a "$LOG"

{
  echo "DONE $(date -Is)"
  echo "RUN_DIR=$RUN_DIR"
  echo "LOG=$LOG"
} | tee -a "$LOG"
