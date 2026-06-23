#!/usr/bin/env bash
# Train one zdim=1 corrected-sign cryoDRGN model for the spike full-atom
# noise10/B100 image-count sweep.

#SBATCH --job-name=spike-drgn-z1-n10
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_zdim1_correct_sign_noise10_b100_20260529/slurm/%x-%j.out
#SBATCH --error=/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_zdim1_correct_sign_noise10_b100_20260529/slurm/%x-%j.err

set -euo pipefail

N_IMAGES="${N_IMAGES:?Set N_IMAGES, e.g. 100000}"
WORKDIR="${WORKDIR:-/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar}"
SOURCE_ROOT="${SOURCE_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_consistency_grid256_noise10_b100_parallel_20260516}"
BENCH_ROOT="${BENCH_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_method_sweep_zdim1_correct_sign_noise10_b100_20260529}"
N_LABEL="$(printf 'n%08d' "$N_IMAGES")"
SOURCE_RUN="$SOURCE_ROOT/$N_LABEL/runs/${N_LABEL}_seed0000"
DATASET="$SOURCE_RUN/03_dataset"
OUTDIR="$BENCH_ROOT/$N_LABEL/cryodrgn/zdim1"
DRGN_TMP_ROOT="${DRGN_TMP_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/tmp}"
DRGN_TMPDIR="$DRGN_TMP_ROOT/drgn_z1_${N_LABEL}_${SLURM_JOB_ID:-manual}"

mkdir -p "$BENCH_ROOT/slurm" "$OUTDIR" "$DRGN_TMPDIR"

# cryoDRGN/PyTorch multiprocessing opens AF_UNIX sockets under TMPDIR. Keep
# this path short enough to avoid the socket path-length limit.
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

visible_gpus=1
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a cuda_visible <<< "$CUDA_VISIBLE_DEVICES"
  visible_gpus="${#cuda_visible[@]}"
elif [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  visible_gpus="$SLURM_GPUS_ON_NODE"
fi

extra_args=()
if [[ "${USE_MULTIGPU:-1}" == "1" && "$visible_gpus" -gt 1 ]]; then
  extra_args+=(--multigpu)
fi

echo "START=$(date -Is)"
echo "JOB_ID=${SLURM_JOB_ID:-manual}"
echo "HOST=$(hostname)"
echo "WORKDIR=$WORKDIR"
echo "SOURCE_RUN=$SOURCE_RUN"
echo "BENCH_ROOT=$BENCH_ROOT"
echo "OUTDIR=$OUTDIR"
echo "N_IMAGES=$N_IMAGES"
echo "ZDIM=1"
echo "TMPDIR=$TMPDIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "VISIBLE_GPUS=$visible_gpus"
echo "USE_MULTIGPU=${USE_MULTIGPU:-1}"
echo "SIGN_OPTION=--uninvert-data"
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

cmd=(
  cryodrgn train_vae
  "$DATASET/particles.256.mrcs"
  --poses "$DATASET/poses.pkl"
  --ctf "$DATASET/ctf.pkl"
  --zdim 1
  --lazy
  --seed 0
  --num-epochs 20
  --num-workers 4
  --max-threads "${SLURM_CPUS_PER_TASK:-8}"
  --uninvert-data
  "${extra_args[@]}"
  -o "$OUTDIR"
)
echo "TRAIN_COMMAND=${cmd[*]}"
/usr/bin/time -v "${cmd[@]}"

echo "DONE=$(date -Is)"
echo "OUTDIR=$OUTDIR"
