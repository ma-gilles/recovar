#!/usr/bin/env bash
# Single-dataset cryoDRGN sanity sweep on the 100k spike full-atom noise10/B100 dataset.

#SBATCH --job-name=spike-drgn-100k-sanity
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sanity_100k_noise10_b100_20260529/slurm/%x-%j.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sanity_100k_noise10_b100_20260529/slurm/%x-%j.err

set -euo pipefail

WORKDIR="${WORKDIR:-/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar}"
SOURCE_RUN="${SOURCE_RUN:-/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise10_b100_parallel_20260516/n00100000/runs/n00100000_seed0000}"
BENCH_ROOT="${BENCH_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sanity_100k_noise10_b100_20260529}"
DATASET="$SOURCE_RUN/03_dataset"
EVAL_ROOT="$BENCH_ROOT/evaluation"
DRGN_TMP_ROOT="${DRGN_TMP_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/tmp}"
DRGN_TMPDIR="$DRGN_TMP_ROOT/drgn100k_${SLURM_JOB_ID:-manual}"
ZDIMS_STR="${ZDIMS:-8 1}"
LABELS="${LABELS:-0,25,50}"

mkdir -p "$BENCH_ROOT/slurm" "$EVAL_ROOT" "$DRGN_TMPDIR"
export TMPDIR="$DRGN_TMPDIR"
export TMP="$DRGN_TMPDIR"
export TEMP="$DRGN_TMPDIR"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
set +u
module purge
module load cryodrgn/3.4.3
set -u

cd "$WORKDIR"

echo "START=$(date -Is)"
echo "JOB_ID=${SLURM_JOB_ID:-manual}"
echo "HOST=$(hostname)"
echo "WORKDIR=$WORKDIR"
echo "SOURCE_RUN=$SOURCE_RUN"
echo "BENCH_ROOT=$BENCH_ROOT"
echo "ZDIMS=$ZDIMS_STR"
echo "LABELS=$LABELS"
echo "TMPDIR=$TMPDIR"
echo "cryodrgn=$(which cryodrgn)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true

for required in "$DATASET/particles.256.mrcs" "$DATASET/poses.pkl" "$DATASET/ctf.pkl" "$DATASET/state_assignment.npy"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 2
  fi
done

for zdim in $ZDIMS_STR; do
  outdir="$BENCH_ROOT/cryodrgn/zdim${zdim}"
  mkdir -p "$outdir"
  if [[ -f "$outdir/weights.19.pkl" && "${OVERWRITE:-0}" != "1" ]]; then
    echo "Existing final checkpoint found, skipping train: $outdir/weights.19.pkl"
    continue
  fi
  cmd=(
    cryodrgn train_vae
    "$DATASET/particles.256.mrcs"
    --poses "$DATASET/poses.pkl"
    --ctf "$DATASET/ctf.pkl"
    --zdim "$zdim"
    --lazy
    --seed 0
    --num-epochs 20
    --num-workers 4
    --max-threads "${SLURM_CPUS_PER_TASK:-8}"
    -o "$outdir"
  )
  echo "TRAIN_COMMAND=${cmd[*]}"
  /usr/bin/time -v "${cmd[@]}"
done

python scripts/experiments/spike_fullatom_method_benchmark/evaluation_compute_mean_embeddings.py \
  --source-run "$SOURCE_RUN" \
  --bench-root "$BENCH_ROOT" \
  --out-root "$EVAL_ROOT" \
  --methods cryodrgn \
  --labels "$LABELS"

python scripts/experiments/spike_fullatom_method_benchmark/evaluation_decode_mean_embeddings.py \
  --bench-root "$BENCH_ROOT" \
  --evaluation-root "$EVAL_ROOT" \
  --methods cryodrgn \
  --apix 1.25 \
  --overwrite

python scripts/experiments/spike_fullatom_method_benchmark/evaluation_score_decoded_volumes.py \
  --source-run "$SOURCE_RUN" \
  --bench-root "$BENCH_ROOT" \
  --evaluation-root "$EVAL_ROOT" \
  --out-dir "$EVAL_ROOT/metrics"

echo "DONE=$(date -Is)"
echo "BENCH_ROOT=$BENCH_ROOT"
echo "METRICS=$EVAL_ROOT/metrics/decoded_volume_metrics.csv"
