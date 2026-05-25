#!/usr/bin/env bash
#SBATCH --job-name=ma-fixed-b-ds
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -eo pipefail
set +u
module purge
module load relion/5.0.1/gcc-11.5.0-gpu
set -u

USER_NAME="${USER:-$(id -un)}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/tmp/${USER_NAME}/recovar_spike}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TORCH_HOME="${TORCH_HOME:-$SCRATCH_ROOT/modelangelo_torch_home}"
export TMPDIR="${TMPDIR:-$SCRATCH_ROOT/tmp/modelangelo_fixed_b_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}}"
mkdir -p "$TMPDIR" "$TORCH_HOME"

ROOT="${ROOT:?Set ROOT to the ModelAngelo experiment output directory}"
RUN_LIST="${RUN_LIST:?Set RUN_LIST to a TSV of label<TAB>run_dir}"
FASTA="${FASTA:-/scratch/gpfs/GILLES/mg6942/runs/spike_full_atom_morph_validation_20260516/spike_P0DTC2_2P_ectodomain_27_1147_trimer_ABC.fasta}"
ADHOC_BFAC="${ADHOC_BFAC:--100}"
ANGPIX="${ANGPIX:-1.25}"

LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$RUN_LIST")"
LABEL="${LINE%%$'\t'*}"
RUN_DIR="${LINE#*$'\t'}"
HALF1="$RUN_DIR/07_compute_state/state000_half1_unfil.mrc"
HALF2="$RUN_DIR/07_compute_state/state000_half2_unfil.mrc"
MASK="$RUN_DIR/05_masks/volume_mask_union.mrc"
POST="$ROOT/postprocess/$LABEL"
OUT="$ROOT/modelangelo/$LABEL"
MAP="$POST/postprocess.mrc"

rm -rf "$POST" "$OUT"
mkdir -p "$POST" "$OUT"

echo "host=$(hostname)"
echo "start=$(date -Is)"
echo "label=$LABEL"
echo "run_dir=$RUN_DIR"
echo "half1=$HALF1"
echo "half2=$HALF2"
echo "mask=$MASK"
echo "fasta=$FASTA"
echo "adhoc_bfac=$ADHOC_BFAC"
echo "post=$POST"
echo "out=$OUT"
echo "torch_home=$TORCH_HOME"
echo "tmpdir=$TMPDIR"
echo "model_angelo=$(which model_angelo)"
echo "relion_postprocess=$(which relion_postprocess)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true

relion_postprocess \
  --i "$HALF1" \
  --i2 "$HALF2" \
  --o "$POST/postprocess" \
  --angpix "$ANGPIX" \
  --mask "$MASK" \
  --adhoc_bfac "$ADHOC_BFAC" \
  --low_pass -1 \
  --force_mask

model_angelo build \
  --volume-path "$MAP" \
  --protein-fasta "$FASTA" \
  --output-dir "$OUT" \
  --device 0

echo "done=$(date -Is)"
find "$POST" "$OUT" -maxdepth 2 -type f | sort
