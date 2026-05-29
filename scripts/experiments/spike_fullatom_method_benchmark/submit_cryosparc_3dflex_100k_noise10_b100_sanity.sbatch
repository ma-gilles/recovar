#!/usr/bin/env bash
# Submit a single 100k spike full-atom noise10/B100 3DFlex K=1/K=2 sanity sweep.

#SBATCH --job-name=spike-3dflex-100k-sanity
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sanity_100k_noise10_b100_20260529/slurm/%x-%j.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sanity_100k_noise10_b100_20260529/slurm/%x-%j.err

set -euo pipefail

WORKDIR="${WORKDIR:-/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar}"
SOURCE_RUN="${SOURCE_RUN:-/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise10_b100_parallel_20260516/n00100000/runs/n00100000_seed0000}"
BENCH_ROOT="${BENCH_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sanity_100k_noise10_b100_20260529}"
PROJECT="${PROJECT:-P587}"
IMPORT_LANE="${IMPORT_LANE:-24hrs}"
GPU_LANE="${GPU_LANE:-48hrs-a100}"
AGENT_ID="spike-3dflex-100k-${SLURM_JOB_ID:-manual}"

export TMPDIR="/scratch/gpfs/GILLES/mg6942/tmp/${AGENT_ID}"
export PIXI_HOME="/scratch/gpfs/GILLES/mg6942/pixi_home/${AGENT_ID}"
export RATTLER_CACHE_DIR="/scratch/gpfs/GILLES/mg6942/rattler_cache/${AGENT_ID}"
mkdir -p "$TMPDIR" "$PIXI_HOME" "$RATTLER_CACHE_DIR" "$BENCH_ROOT/slurm" "$BENCH_ROOT/cryosparc_3dflex"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

set +u
module purge
module load cryosparc-tools/5.0.3
set -u

cd "$WORKDIR"

echo "START=$(date -Is)"
echo "JOB_ID=${SLURM_JOB_ID:-manual}"
echo "HOST=$(hostname)"
echo "SOURCE_RUN=$SOURCE_RUN"
echo "BENCH_ROOT=$BENCH_ROOT"
echo "PROJECT=$PROJECT"
echo "IMPORT_LANE=$IMPORT_LANE"
echo "GPU_LANE=$GPU_LANE"
echo "TMPDIR=$TMPDIR"
echo "python=$(which python)"

python scripts/experiments/spike_fullatom_method_benchmark/submit_cryosparc_3dflex_100k.py \
  --source-run "$SOURCE_RUN" \
  --bench-root "$BENCH_ROOT" \
  --project "$PROJECT" \
  --workspace-title "spike fullatom noise10 B100 100k method sanity 20260529" \
  --import-lane "$IMPORT_LANE" \
  --gpu-lane "$GPU_LANE" \
  --submit

echo "DONE=$(date -Is)"
echo "MANIFEST=$BENCH_ROOT/cryosparc_3dflex/cryosparc_3dflex_jobs.json"
