#!/bin/bash
#SBATCH --job-name=relion-fixed-init
#SBATCH --account=amits
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=300G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/%j-%x.out

# Re-run RELION's auto-refine on the same benchmark but using the
# RELION-frame init MRC (reference_init_relion.mrc), so RELION doesn't
# refine into the antipode basin like the original reference run did
# (median pose error 133°, see commit history 2026-04-08).
#
# All other flags match the original reference run in
# scripts/run_relion_parity_benchmark_slurm.sh, so the resulting
# trajectories are directly comparable.

set -eo pipefail   # NOT -u: RELION's module load tickles unbound PS1

export PYTHONNOUSERSITE=1
export TMPDIR="/scratch/gpfs/GILLES/mg6942/tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" /scratch/gpfs/GILLES/mg6942/slurmo

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
unset LD_LIBRARY_PATH || true

DATA_DIR="${DATA_DIR:-/scratch/gpfs/GILLES/mg6942/tmp/em_profile/data}"
RELION_REF_DIR="${RELION_REF_DIR:-${DATA_DIR}/relion_ref_fixed_init}"
RELION_RUN_PREFIX="${RELION_REF_DIR}/run"
RELION_MODULE="${RELION_MODULE:-relion/5.0.1/gcc-11.5.0-gpu}"
HEALPIX_ORDER="${HEALPIX_ORDER:-3}"
OFFSET_RANGE="${OFFSET_RANGE:-3.0}"
OFFSET_STEP="${OFFSET_STEP:-1.0}"
RELION_MPI_RANKS="${RELION_MPI_RANKS:-3}"

mkdir -p "$RELION_REF_DIR"

echo "=== Verify RELION-frame init MRC exists ==="
test -f "${DATA_DIR}/reference_init_relion.mrc"
ls -la "${DATA_DIR}/reference_init_relion.mrc"

echo "=== Load RELION ==="
module load "$RELION_MODULE"
command -v relion_refine_mpi
command -v mpirun

export CUDA_VISIBLE_DEVICES=0

echo "=== Run RELION auto-refine with fixed init ==="
# Two RELION flags REQUIRED here:
#
# --firstiter_cc — the recovar prepare script writes the init via
# save_volume() which is in recovar's intensity convention, NOT RELION's.
# Without --firstiter_cc, RELION's iter-1 Bayesian E-step uses the wrong
# projection scale and the pose search collapses to a 2D-extruded basin
# (volume slices show horizontal stripes instead of a 3D ribosome,
# h1/h2 FSC plateaus at ~14-22 A but FSC vs GT stays at ~20 A regardless
# of particle count). See
# ~/.claude/projects/-home-mg6942/memory/feedback_relion_firstiter_cc_required.md
#
# --ctf — RELION's default is OFF. Without --ctf, RELION reconstructs
# the CTF-convolved volume directly: real-space dark halo / ringing
# around bright density, resolution plateau ~18-22 A regardless of
# particle count, radial power spectrum shows CTF oscillations and
# 1-2 orders of magnitude excess high-freq power vs GT. See
# ~/.claude/projects/-home-mg6942/memory/feedback_relion_ctf_required.md
cd "$DATA_DIR"
# Additional GUI-default flags (DEFAULT OFF on command line):
#   --flatten_solvent / --zero_mask — mask references and particle exterior
#   --low_resol_join_halves 40 — prevents h1/h2 divergence below 40 A
#   --norm / --scale — per-optics-group corrections (no-op for single group)
# See memory/feedback_relion_required_flags.md for the GUI-vs-CLI audit.
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
  --gpu 0 \
  --j 4

echo "=== RELION fixed-init run complete ==="
echo "Output: $RELION_REF_DIR"
ls -la "$RELION_REF_DIR" | head -20
