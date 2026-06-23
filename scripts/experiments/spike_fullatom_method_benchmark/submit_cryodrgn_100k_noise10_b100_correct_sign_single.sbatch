#!/usr/bin/env bash
# Train one corrected-sign cryoDRGN model on the 100k spike full-atom noise10/B100 dataset.

#SBATCH --job-name=spike-drgn-correct-sign
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
ZDIM="${ZDIM:?Set ZDIM=1 or ZDIM=8}"
OUT_PARENT="${OUT_PARENT:-$BENCH_ROOT/cryodrgn_correct_sign}"
OUTDIR="$OUT_PARENT/zdim${ZDIM}"
DRGN_TMP_ROOT="${DRGN_TMP_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/tmp}"
DRGN_TMPDIR="$DRGN_TMP_ROOT/drgn100k_correct_${SLURM_JOB_ID:-manual}"

mkdir -p "$BENCH_ROOT/slurm" "$OUTDIR" "$DRGN_TMPDIR"
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
echo "OUTDIR=$OUTDIR"
echo "ZDIM=$ZDIM"
echo "TMPDIR=$TMPDIR"
echo "cryodrgn=$(which cryodrgn)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true

for required in "$DATASET/particles.256.mrcs" "$DATASET/poses.pkl" "$DATASET/ctf.pkl"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 2
  fi
done

cmd=(
  cryodrgn train_vae
  "$DATASET/particles.256.mrcs"
  --poses "$DATASET/poses.pkl"
  --ctf "$DATASET/ctf.pkl"
  --zdim "$ZDIM"
  --lazy
  --seed 0
  --num-epochs 20
  --num-workers 4
  --max-threads "${SLURM_CPUS_PER_TASK:-8}"
  --uninvert-data
  -o "$OUTDIR"
)
echo "TRAIN_COMMAND=${cmd[*]}"
/usr/bin/time -v "${cmd[@]}"

echo "DONE=$(date -Is)"
echo "OUTDIR=$OUTDIR"
