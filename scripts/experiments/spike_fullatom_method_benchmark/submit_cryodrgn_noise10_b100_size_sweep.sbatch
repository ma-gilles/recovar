#!/usr/bin/env bash
# cryoDRGN size/zdim sweep on existing spike full-atom noise10/B100 datasets.

#SBATCH --job-name=spike-drgn-n10-sweep
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=48:00:00
#SBATCH --array=0-11%12
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise10_b100_20260528/slurm/%x-%A_%a.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise10_b100_20260528/slurm/%x-%A_%a.err

set -euo pipefail

N_IMAGES_VALUES=(10000 30000 100000 300000 1000000 3000000)
ZDIMS=(1 8)
N_INDEX=$((SLURM_ARRAY_TASK_ID / ${#ZDIMS[@]}))
Z_INDEX=$((SLURM_ARRAY_TASK_ID % ${#ZDIMS[@]}))
N_IMAGES="${N_IMAGES_VALUES[$N_INDEX]}"
ZDIM="${ZDIMS[$Z_INDEX]}"
N_LABEL="$(printf 'n%08d' "$N_IMAGES")"

WORKDIR="${WORKDIR:-/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar}"
SOURCE_ROOT="${SOURCE_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise10_b100_parallel_20260516}"
BENCH_ROOT="${BENCH_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_noise10_b100_20260528}"
SOURCE_RUN="$SOURCE_ROOT/$N_LABEL/runs/${N_LABEL}_seed0000"
OUTDIR="$BENCH_ROOT/$N_LABEL/cryodrgn/zdim${ZDIM}"
DATASET="$SOURCE_RUN/03_dataset"
AGENT_ID="spike-drgn-n10-${N_LABEL}-z${ZDIM}-${SLURM_ARRAY_JOB_ID:-manual}-${SLURM_ARRAY_TASK_ID:-0}"
DRGN_TMP_ROOT="${DRGN_TMP_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/tmp}"
DRGN_TMPDIR="$DRGN_TMP_ROOT/drgn_${SLURM_JOB_ID:-${SLURM_ARRAY_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}}"

mkdir -p "$BENCH_ROOT/slurm" "$BENCH_ROOT/tmp/$AGENT_ID" "$OUTDIR" "$DRGN_TMPDIR"
# cryoDRGN/PyTorch multiprocessing opens AF_UNIX sockets under TMPDIR; keep the
# per-job path short enough to avoid the socket path limit.
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
echo "ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-manual}"
echo "HOST=$(hostname)"
echo "WORKDIR=$WORKDIR"
echo "SOURCE_RUN=$SOURCE_RUN"
echo "BENCH_ROOT=$BENCH_ROOT"
echo "OUTDIR=$OUTDIR"
echo "N_IMAGES=$N_IMAGES"
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
  --uninvert-data \
  -o "$OUTDIR"

echo "DONE=$(date -Is)"
echo "OUTDIR=$OUTDIR"
