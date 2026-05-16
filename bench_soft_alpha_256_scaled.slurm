#!/usr/bin/env bash
#SBATCH --job-name=sa256-scaled
#SBATCH --account=amits
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500GB
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/sa256-scaled-%j.out

set -euo pipefail
cd /scratch/gpfs/GILLES/mg6942/recovar_wt_agent_precond_study_20260330

export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="/scratch/gpfs/GILLES/mg6942/tmp/slurm_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
pixi run python -c "import jax; d=jax.devices(); print(f'GPU: {d}'); assert any('gpu' in str(x).lower() or 'cuda' in str(x).lower() for x in d)"
date
pixi run python bench_soft_alpha_256_scaled.py 2>&1
date
