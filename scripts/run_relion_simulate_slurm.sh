#!/bin/bash
#SBATCH --job-name=relion-simulate
#SBATCH --account=amits
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/%j-%x.out

# Use relion_project to generate synthetic particles from the recovar GT
# volume (in RELION frame), using the existing particles.star poses + CTF
# parameters.
#
# This eliminates "recovar's simulator favors recovar's reconstructor" as
# a possible source of recovar's edge: the data is now produced by RELION's
# own forward model, scored by both pipelines on equal footing.

set -eo pipefail

export PYTHONNOUSERSITE=1
export TMPDIR="/scratch/gpfs/GILLES/mg6942/tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" /scratch/gpfs/GILLES/mg6942/slurmo

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
unset LD_LIBRARY_PATH || true

INPUT_DIR="${INPUT_DIR:-/scratch/gpfs/GILLES/mg6942/tmp/em_profile/data}"
OUT_DIR="${OUT_DIR:-/scratch/gpfs/GILLES/mg6942/tmp/em_profile/data_relionsim_5k}"
WHITE_NOISE="${WHITE_NOISE:-1.0}"
ANGPIX="${ANGPIX:-4.25}"
PARTICLE_DIAMETER="${PARTICLE_DIAMETER:-200}"

mkdir -p "$OUT_DIR"

echo "=== Inputs ==="
echo "GT vol  : $INPUT_DIR/reference_gt_relion.mrc"
echo "STAR    : $INPUT_DIR/particles.star"
echo "Out dir : $OUT_DIR"
echo "Noise σ : $WHITE_NOISE"
test -f "$INPUT_DIR/reference_gt_relion.mrc"
test -f "$INPUT_DIR/particles.star"

module load relion/5.0.1/gcc-11.5.0-gpu

# 1) Use relion_project to generate noise-free + noisy projections.
#    --ang particles.star  : poses + CTF from the existing STAR
#    --ctf                 : apply CTF
#    --add_noise           : add white Gaussian noise
#    --white_noise σ       : noise std (per pixel)
echo "=== Generating particles via relion_project ==="
cd "$OUT_DIR"
relion_project \
  --i "$INPUT_DIR/reference_gt_relion.mrc" \
  --ang "$INPUT_DIR/particles.star" \
  --o "$OUT_DIR/particles_relionsim" \
  --ctf \
  --add_noise \
  --white_noise "$WHITE_NOISE" \
  --angpix "$ANGPIX"

ls -la "$OUT_DIR"
echo "=== relion_project complete ==="
