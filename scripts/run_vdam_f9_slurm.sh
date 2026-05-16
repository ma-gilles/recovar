#!/usr/bin/env bash
#SBATCH --job-name=vdam-f9
#SBATCH --account=amits
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200GB
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-f9-%j.out
#SBATCH --exclusive

# F9: full 200-iter VDAM run against the 500-particle InitialModel fixture
# followed by FSC comparison against initial_model.mrc.
#
# Uses the VDAM-blend approach from F8 (plain EM + real-space step=0.5
# blend each iter). Tighter parity requires full VDAM reconstructGrad
# wiring which is future work.

set -euo pipefail

cd /scratch/gpfs/GILLES/mg6942/recovar_dev/recovar

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="/scratch/gpfs/GILLES/mg6942/tmp/slurm_${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR"

pixi run python scripts/run_vdam_f9_fixture.py \
    --nr-iter 200 \
    --blend-step 0.5 \
    --output-mrc /scratch/gpfs/GILLES/mg6942/_agent_scratch/vdam_f9_${SLURM_JOB_ID:-manual}_final.mrc
