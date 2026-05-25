#!/usr/bin/env bash
#SBATCH --job-name=spike-dyna-100k
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/%x_%A_%a.out
#SBATCH --error=/scratch/gpfs/GILLES/mg6942/slurmo/%x_%A_%a.err

set -euo pipefail

ZDIMS=(1 6)
ZDIM="${ZDIMS[$SLURM_ARRAY_TASK_ID]}"

SOURCE_RUN="${SOURCE_RUN:-/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise10_b100_parallel_20260516/n00100000/runs/n00100000_seed0000}"
BENCH_ROOT="${BENCH_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_benchmark_100k_20260517}"
OUTDIR="$BENCH_ROOT/dynamight/zdim${ZDIM}"
DYNAMIGHT_PYDEPS="${DYNAMIGHT_PYDEPS:-/scratch/gpfs/CRYOEM/gilleslab/tmp/relion_dynamight_pydeps_20260517}"

export TMPDIR="${TMPDIR:-/scratch/gpfs/CRYOEM/gilleslab/tmp/dynamight_100k_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}}"
mkdir -p "$TMPDIR" "$OUTDIR" "$BENCH_ROOT/logs"

unset PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export PYTHONPATH="$DYNAMIGHT_PYDEPS"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

set +u
module purge
module load relion/5.0.1/gcc-11.5.0-gpu
set -u

echo "job=${SLURM_JOB_ID:-unknown}"
echo "array_job=${SLURM_ARRAY_JOB_ID:-unknown}"
echo "task=${SLURM_ARRAY_TASK_ID:-unknown}"
echo "host=$(hostname)"
echo "start=$(date -Is)"
echo "source_run=$SOURCE_RUN"
echo "outdir=$OUTDIR"
echo "zdim=$ZDIM"
echo "dynamight=$(which dynamight)"
echo "python=$(which python)"
echo "dynamight_pydeps=$DYNAMIGHT_PYDEPS"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true

if [[ ! -d "$DYNAMIGHT_PYDEPS/click" ]]; then
  echo "Missing scratch click shim at $DYNAMIGHT_PYDEPS/click" >&2
  echo "Create it with: python3 -m pip install --no-deps --target $DYNAMIGHT_PYDEPS click==8.1.7" >&2
  exit 2
fi

if [[ -f "$OUTDIR/checkpoints/checkpoint_final.pt" && "${OVERWRITE:-0}" != "1" ]]; then
  echo "Existing final checkpoint found, skipping: $OUTDIR/checkpoints/checkpoint_final.pt"
  exit 0
fi

/usr/bin/time -v python "$SLURM_SUBMIT_DIR/scripts/experiments/spike_fullatom_method_benchmark/run_dynamight_optimize_direct.py" \
  --refinement-star-file "$SOURCE_RUN/03_dataset/particles.star" \
  --output-directory "$OUTDIR" \
  --initial-model "$SOURCE_RUN/06_pipeline/output/volumes/mean.mrc" \
  --mask-file "$SOURCE_RUN/05_masks/volume_mask_union.mrc" \
  --n-latent-dimensions "$ZDIM" \
  --n-gaussians "${N_GAUSSIANS:-8000}" \
  --initial-resolution "${INITIAL_RESOLUTION:-8}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --n-threads "${SLURM_CPUS_PER_TASK:-8}" \
  --n-workers 4 \
  --gpu-id 0

echo "done=$(date -Is)"
find "$OUTDIR" -maxdepth 2 -type f | sort | sed -n '1,200p'
