#!/usr/bin/env bash
#SBATCH --job-name=soft-alpha
#SBATCH --account=amits
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500GB
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/soft-alpha-%j.out

set -euo pipefail

WORKDIR="/scratch/gpfs/GILLES/mg6942/recovar_wt_agent_precond_study_20260330"
cd "$WORKDIR"

export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="/scratch/gpfs/GILLES/mg6942/tmp/slurm_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi
echo ""

# Provenance + GPU check
pixi run python -c "
import pathlib, recovar, jax, sys
repo = pathlib.Path.cwd().resolve()
assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo)+'/'), 'WRONG recovar'
assert '.pixi/envs/default/' in str(pathlib.Path(jax.__file__).resolve()), 'WRONG jax'
devices = jax.devices()
print('ENV_OK')
print(f'GPU devices: {devices}')
assert any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices), 'NO GPU!'
print('GPU_OK')
"

date
pixi run python bench_soft_alpha.py 2>&1
date
echo "=== Done ==="
