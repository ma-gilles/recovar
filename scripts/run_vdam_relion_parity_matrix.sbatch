#!/usr/bin/env bash
#SBATCH --job-name=vdam-fixed12
#SBATCH --account=amits
#SBATCH --partition=cryoem
#SBATCH --array=1-12%4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-fixed12-%A_%a.out

set -euo pipefail

: "${OUTPUT_ROOT:?submit with --export=ALL,OUTPUT_ROOT=/scratch/...}"
: "${SLURM_ARRAY_TASK_ID:?submit this script as a Slurm array}"

if (( SLURM_ARRAY_TASK_ID < 1 || SLURM_ARRAY_TASK_ID > 12 )); then
  echo "VDAM fixed-suite task must be in [1, 12], got ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
CASE_ID="$(printf 'vdam-%02d' "${SLURM_ARRAY_TASK_ID}")"
if [[ -z "${RECOVAR_VDAM_IMAGE_BATCH_SIZE:-}" ]]; then
  case "${CASE_ID}" in
    vdam-09) export RECOVAR_VDAM_IMAGE_BATCH_SIZE=200 ;;
    vdam-12)
      export RECOVAR_VDAM_IMAGE_BATCH_SIZE=25
      export RECOVAR_EXACT_LOCAL_SPARSE_BIG_JIT_MSTEP_MAX_GB=0
      ;;
  esac
fi
export REPO_ROOT CASE_ID OUTPUT_ROOT

exec bash "${REPO_ROOT}/scripts/run_vdam_relion_parity_case.sbatch"
