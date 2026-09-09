#!/usr/bin/env bash
#SBATCH --job-name=vdam-k1-robust
#SBATCH --account=amits
#SBATCH --partition=cryoem
#SBATCH --array=1-12%4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/vdam-k1-robust-%A_%a.out

set -euo pipefail

: "${OUTPUT_ROOT:?submit with --export=ALL,OUTPUT_ROOT=/scratch/...}"
: "${SLURM_ARRAY_TASK_ID:?submit this script as a Slurm array}"

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
VDAM_SCORECARD="${REPO_ROOT}/docs/math/vdam_k1_robustness_suite_v1.json"
if (( SLURM_ARRAY_TASK_ID < 1 || SLURM_ARRAY_TASK_ID > 12 )); then
  echo "VDAM robustness-suite task must be in [1, 12], got ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi
CASE_ID="$(printf 'vdam-b%02d' "${SLURM_ARRAY_TASK_ID}")"
export REPO_ROOT VDAM_SCORECARD CASE_ID OUTPUT_ROOT

exec bash "${REPO_ROOT}/scripts/run_vdam_relion_parity_case.sbatch"
