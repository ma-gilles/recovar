#!/usr/bin/env bash
# Decode and score completed cryoDRGN models for each image count.

#SBATCH --job-name=spike-drgn-score
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-5%6
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise10_b100_20260528/slurm/%x-%A_%a.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise10_b100_20260528/slurm/%x-%A_%a.err

set -euo pipefail

N_IMAGES_VALUES=(10000 30000 100000 300000 1000000 3000000)
N_IMAGES="${N_IMAGES_VALUES[$SLURM_ARRAY_TASK_ID]}"
N_LABEL="$(printf 'n%08d' "$N_IMAGES")"

WORKDIR="${WORKDIR:-/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar}"
SOURCE_ROOT="${SOURCE_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise10_b100_parallel_20260516}"
BENCH_ROOT="${BENCH_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise10_b100_20260528}"
SOURCE_RUN="$SOURCE_ROOT/$N_LABEL/runs/${N_LABEL}_seed0000"
THIS_BENCH="$BENCH_ROOT/$N_LABEL"
EVAL_ROOT="$THIS_BENCH/evaluation"
AGENT_ID="spike-drgn-score-${N_LABEL}-${SLURM_ARRAY_JOB_ID:-manual}-${SLURM_ARRAY_TASK_ID:-0}"
DRGN_TMP_ROOT="${DRGN_TMP_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/tmp}"
DRGN_TMPDIR="$DRGN_TMP_ROOT/drgns_${SLURM_JOB_ID:-${SLURM_ARRAY_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}}"

mkdir -p "$BENCH_ROOT/slurm" "$BENCH_ROOT/tmp/$AGENT_ID" "$EVAL_ROOT" "$DRGN_TMPDIR"
export TMPDIR="$DRGN_TMPDIR"
export TMP="$DRGN_TMPDIR"
export TEMP="$DRGN_TMPDIR"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
set +u
module purge
module load cryodrgn/3.4.3
set -u

cd "$WORKDIR"

echo "START=$(date -Is)"
echo "HOST=$(hostname)"
echo "SOURCE_RUN=$SOURCE_RUN"
echo "THIS_BENCH=$THIS_BENCH"
echo "EVAL_ROOT=$EVAL_ROOT"
echo "N_IMAGES=$N_IMAGES"
echo "TMPDIR=$TMPDIR"
echo "python=$(which python)"
echo "cryodrgn=$(which cryodrgn)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true

python scripts/experiments/spike_fullatom_method_benchmark/evaluation_compute_mean_embeddings.py \
  --source-run "$SOURCE_RUN" \
  --bench-root "$THIS_BENCH" \
  --out-root "$EVAL_ROOT" \
  --methods cryodrgn

python scripts/experiments/spike_fullatom_method_benchmark/evaluation_decode_mean_embeddings.py \
  --bench-root "$THIS_BENCH" \
  --evaluation-root "$EVAL_ROOT" \
  --methods cryodrgn \
  --apix 1.25 \
  --overwrite

python scripts/experiments/spike_fullatom_method_benchmark/evaluation_score_decoded_volumes.py \
  --source-run "$SOURCE_RUN" \
  --bench-root "$THIS_BENCH" \
  --evaluation-root "$EVAL_ROOT" \
  --out-dir "$EVAL_ROOT/metrics"

echo "DONE=$(date -Is)"
echo "METRICS=$EVAL_ROOT/metrics/decoded_volume_metrics.csv"
