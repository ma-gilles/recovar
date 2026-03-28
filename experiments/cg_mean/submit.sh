#!/usr/bin/env bash
#SBATCH --job-name=cg-mean-512
#SBATCH --account=amits
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500GB
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/cg-mean-512-%j.out
#SBATCH --exclusive

set -euo pipefail

WORKDIR="/scratch/gpfs/GILLES/mg6942/recovar_wt_agent_20260326_205254_9289"
EXPDIR="/scratch/gpfs/GILLES/mg6942/experiments/cg_mean"

cd "$WORKDIR"

export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="/scratch/gpfs/GILLES/mg6942/tmp/slurm_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV

pixi run python -c "
import pathlib, recovar, jax
repo = pathlib.Path.cwd().resolve()
assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo)+'/'), 'WRONG recovar'
assert '.pixi/envs/default/' in str(pathlib.Path(jax.__file__).resolve()), 'WRONG jax'
print('ENV_OK')
"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
pixi run python -c "import jax; print('JAX devices:', jax.devices())"

export CG_WORKDIR="$EXPDIR"
pixi run python "$EXPDIR/cg_mean_experiment.py" 2>&1

echo "DONE"
