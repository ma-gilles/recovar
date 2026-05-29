#!/bin/bash
#SBATCH --job-name=spike-3dflex-n10-sweep
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --array=0-5%6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/%x-%A_%a.out
#SBATCH --error=/scratch/gpfs/GILLES/mg6942/slurmo/%x-%A_%a.err

set -euo pipefail

REPO=${REPO:-/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar}
BASE_ROOT=${BASE_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise10_b100_parallel_20260516}
BENCH_ROOT=${BENCH_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise10_b100_20260528}
PROJECT=${PROJECT:-P587}
IMPORT_LANE=${IMPORT_LANE:-24hrs}
GPU_LANE=${GPU_LANE:-48hrs-a100}

N_VALUES=(10000 30000 100000 300000 1000000 3000000)
N_IMAGES=${N_VALUES[$SLURM_ARRAY_TASK_ID]}
LABEL=$(printf "n%08d" "$N_IMAGES")
SOURCE_RUN="$BASE_ROOT/$LABEL/runs/${LABEL}_seed0000"
THIS_BENCH="$BENCH_ROOT/$LABEL"

AGENT_ID=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_spike_3dflex_noise10_b100_${LABEL}
export TMPDIR=/scratch/gpfs/GILLES/mg6942/tmp/${AGENT_ID}
export PIXI_HOME=/scratch/gpfs/GILLES/mg6942/pixi_home/${AGENT_ID}
export RATTLER_CACHE_DIR=/scratch/gpfs/GILLES/mg6942/rattler_cache/${AGENT_ID}
mkdir -p "$TMPDIR" "$PIXI_HOME" "$RATTLER_CACHE_DIR" "$THIS_BENCH" /scratch/gpfs/GILLES/mg6942/slurmo

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# The modulefile references PS1; keep nounset off only while module loads.
set +u
module load cryosparc-tools/5.0.3
set -u
cd "$REPO"

python scripts/experiments/spike_fullatom_method_benchmark/submit_cryosparc_3dflex_100k.py \
  --source-run "$SOURCE_RUN" \
  --bench-root "$THIS_BENCH" \
  --project "$PROJECT" \
  --workspace-title "spike fullatom noise10 B100 ${LABEL} method sweep 20260528" \
  --import-lane "$IMPORT_LANE" \
  --gpu-lane "$GPU_LANE" \
  --submit
