#!/bin/bash
#SBATCH --job-name=spike-recovar-n30-score
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/%x-%j.out
#SBATCH --error=/scratch/gpfs/GILLES/mg6942/slurmo/%x-%j.err

set -euo pipefail

REPO=${REPO:-/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar}
AGENT_ID=${SLURM_JOB_ID:-manual}_spike_recovar_noise30_score
export TMPDIR=/scratch/gpfs/GILLES/mg6942/tmp/${AGENT_ID}
export PIXI_HOME=/scratch/gpfs/GILLES/mg6942/pixi_home/${AGENT_ID}
export RATTLER_CACHE_DIR=/scratch/gpfs/GILLES/mg6942/rattler_cache/${AGENT_ID}
mkdir -p "$TMPDIR" "$PIXI_HOME" "$RATTLER_CACHE_DIR" /scratch/gpfs/GILLES/mg6942/slurmo

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "$REPO"
PIXI_PY=$(.pixi/envs/default/bin/python -c 'import sys; print(sys.executable)')

"$PIXI_PY" scripts/experiments/spike_fullatom_method_benchmark/score_recovar_state_sweep_fsc_error.py \
  --title "RECOVAR noise30/B80" \
  --out-dir /scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise30_b80_20260528/recovar \
  --state-root 0=/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise30_b80_state0000_reuse_20260519 \
  --state-root 25=/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise30_b80_state0025_reuse_20260519 \
  --state-root 50=/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise30_b80_parallel_20260518
