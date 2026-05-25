#!/usr/bin/env bash
#SBATCH --job-name=spike-drgn-100k
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/%x_%A_%a.out
#SBATCH --error=/scratch/gpfs/GILLES/mg6942/slurmo/%x_%A_%a.err

set -euo pipefail

ZDIMS=(1 8)
ZDIM="${ZDIMS[$SLURM_ARRAY_TASK_ID]}"

SOURCE_RUN="${SOURCE_RUN:-/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise10_b100_parallel_20260516/n00100000/runs/n00100000_seed0000}"
BENCH_ROOT="${BENCH_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_benchmark_100k_20260517}"
OUTDIR="$BENCH_ROOT/cryodrgn/zdim${ZDIM}"
DATASET="$SOURCE_RUN/03_dataset"

export TMPDIR="${TMPDIR:-/scratch/gpfs/CRYOEM/gilleslab/tmp/cryodrgn_100k_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}}"
mkdir -p "$TMPDIR" "$OUTDIR" "$BENCH_ROOT/logs"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

set +u
module purge
module load cryodrgn/3.4.3
set -u

echo "job=${SLURM_JOB_ID:-unknown}"
echo "array_job=${SLURM_ARRAY_JOB_ID:-unknown}"
echo "task=${SLURM_ARRAY_TASK_ID:-unknown}"
echo "host=$(hostname)"
echo "start=$(date -Is)"
echo "source_run=$SOURCE_RUN"
echo "outdir=$OUTDIR"
echo "zdim=$ZDIM"
echo "cryodrgn=$(which cryodrgn)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true

if [[ -f "$OUTDIR/weights.19.pkl" && "${OVERWRITE:-0}" != "1" ]]; then
  echo "Existing final checkpoint found, skipping: $OUTDIR/weights.19.pkl"
  exit 0
fi

/usr/bin/time -v cryodrgn train_vae \
  "$DATASET/particles.256.mrcs" \
  --poses "$DATASET/poses.pkl" \
  --ctf "$DATASET/ctf.pkl" \
  --zdim "$ZDIM" \
  --lazy \
  --seed 0 \
  --num-epochs 20 \
  --num-workers 4 \
  --max-threads "${SLURM_CPUS_PER_TASK:-8}" \
  -o "$OUTDIR"

echo "done=$(date -Is)"
find "$OUTDIR" -maxdepth 1 -type f | sort

