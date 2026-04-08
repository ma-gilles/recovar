#!/bin/bash
#SBATCH --job-name=relion-skipalign
#SBATCH --account=amits
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=300G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/%j-%x.out

# Diagnostic: run RELION's auto-refine with --skip_align so it does NOT
# do pose search at iter 1 — it just uses the GT poses from particles.star
# and does pure M-step reconstruction.
#
# This isolates RELION's M-step from its E-step. If RELION reaches
# similar resolution to recovar in this mode, then the ~6 A gap is
# entirely from RELION's E-step (pose search) being suboptimal on this
# benchmark. If RELION still loses by 6 A here, the gap is in the
# Wiener / projection / regularization steps.

set -eo pipefail

export PYTHONNOUSERSITE=1
export TMPDIR="/scratch/gpfs/GILLES/mg6942/tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" /scratch/gpfs/GILLES/mg6942/slurmo

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
unset LD_LIBRARY_PATH || true

DATA_DIR="${DATA_DIR:-/scratch/gpfs/GILLES/mg6942/tmp/em_profile/data}"
RELION_REF_DIR="${RELION_REF_DIR:-${DATA_DIR}/relion_ref_skipalign}"
RELION_RUN_PREFIX="${RELION_REF_DIR}/run"
RELION_MODULE="${RELION_MODULE:-relion/5.0.1/gcc-11.5.0-gpu}"
HEALPIX_ORDER="${HEALPIX_ORDER:-3}"
OFFSET_RANGE="${OFFSET_RANGE:-3.0}"
OFFSET_STEP="${OFFSET_STEP:-1.0}"
RELION_MPI_RANKS="${RELION_MPI_RANKS:-3}"

mkdir -p "$RELION_REF_DIR"

echo "=== Config ==="
echo "DATA_DIR=$DATA_DIR"
echo "OUTPUT=$RELION_REF_DIR"
test -f "${DATA_DIR}/reference_init_relion.mrc"

module load "$RELION_MODULE"
export CUDA_VISIBLE_DEVICES=0

cd "$DATA_DIR"
# REQUIRED RELION flags for any non-RELION-init benchmark:
#   --firstiter_cc — non-RELION init intensity-scale fix.
#     memory/feedback_relion_firstiter_cc_required.md
#   --ctf — RELION default is OFF. Without it, RELION reconstructs the
#     CTF-convolved volume (dark halo, ~18-22 A ceiling).
#     memory/feedback_relion_ctf_required.md
mpirun -n "$RELION_MPI_RANKS" relion_refine_mpi \
  --i particles.star \
  --ref reference_init_relion.mrc \
  --o "$RELION_RUN_PREFIX" \
  --auto_refine \
  --split_random_halves \
  --particle_diameter 200 \
  --ini_high 30 \
  --firstiter_cc \
  --ctf \
  --flatten_solvent \
  --zero_mask \
  --low_resol_join_halves 40 \
  --norm \
  --scale \
  --healpix_order "$HEALPIX_ORDER" \
  --offset_range "$OFFSET_RANGE" \
  --offset_step "$OFFSET_STEP" \
  --oversampling 1 \
  --pad 2 \
  --skip_align \
  --gpu 0 \
  --j 4

echo "=== Complete ==="
