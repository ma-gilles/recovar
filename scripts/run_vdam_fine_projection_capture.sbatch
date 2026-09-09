#!/usr/bin/env bash
#SBATCH --job-name=vdam-proj-cap
#SBATCH --account=amits
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-proj-cap-%j.out

set -euo pipefail

: "${RELION_CAPTURE:?set RELION_CAPTURE}"
: "${RECOVAR_SCORE_DUMP:?set RECOVAR_SCORE_DUMP}"
: "${COMPARISON_OUTPUT:?set COMPARISON_OUTPUT}"

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
cd "${REPO_ROOT}"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export RECOVAR_CUDA_LIB="${REPO_ROOT}/recovar/cuda/libcuda_backproject.so"

"${REPO_ROOT}/.pixi/envs/default/bin/python" \
  scripts/compare_vdam_fine_projection_capture.py \
  --relion-capture "${RELION_CAPTURE}" \
  --recovar-score-dump "${RECOVAR_SCORE_DUMP}" \
  --output "${COMPARISON_OUTPUT}"
