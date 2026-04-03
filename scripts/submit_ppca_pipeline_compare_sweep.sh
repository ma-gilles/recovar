#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/gpfs/GILLES/mg6942/recovar_wt_agent_precond_study_20260330}"
BASE_DIR="${BASE_DIR:-/home/mg6942/mytigress/cryobench2}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-/scratch/gpfs/GILLES/mg6942/ppca_pipeline_compare_${STAMP}}"
GRID_SIZE="${GRID_SIZE:-128}"
N_IMAGES="${N_IMAGES:-100000}"
NOISE_LEVEL="${NOISE_LEVEL:-1.0}"
ZDIM="${ZDIM:-10}"
PPCA_EM_ITERS="${PPCA_EM_ITERS:-20}"
SEED="${SEED:-42}"
GPU_GB="${GPU_GB:-40}"
FORCE="${FORCE:-0}"

DATASETS=(Ribosembly IgG-1D IgG-RL Tomotwin-100)
CONTRASTS=(0.0 0.3)

cd "$REPO_ROOT"

echo "Submitting PPCA pipeline comparison sweep"
echo "  REPO_ROOT=$REPO_ROOT"
echo "  BASE_DIR=$BASE_DIR"
echo "  RESULTS_ROOT=$RESULTS_ROOT"
echo "  GRID_SIZE=$GRID_SIZE N_IMAGES=$N_IMAGES NOISE_LEVEL=$NOISE_LEVEL"
echo "  ZDIM=$ZDIM PPCA_EM_ITERS=$PPCA_EM_ITERS SEED=$SEED GPU_GB=$GPU_GB"
echo

mkdir -p "$RESULTS_ROOT"
mkdir -p /scratch/gpfs/GILLES/mg6942/slurmo

MANIFEST="$RESULTS_ROOT/slurm_jobs.txt"
: > "$MANIFEST"

for dataset in "${DATASETS[@]}"; do
  safe_dataset="${dataset//[^A-Za-z0-9]/-}"
  for contrast in "${CONTRASTS[@]}"; do
    ctag="${contrast/./p}"
    job_name="pcmp-${safe_dataset}-c${ctag}"
    jid="$(
      sbatch --parsable \
        --job-name="$job_name" \
        --export=ALL,REPO_ROOT="$REPO_ROOT",BASE_DIR="$BASE_DIR",RESULTS_ROOT="$RESULTS_ROOT",DATASET="$dataset",GRID_SIZE="$GRID_SIZE",N_IMAGES="$N_IMAGES",NOISE_LEVEL="$NOISE_LEVEL",CONTRAST_STD="$contrast",ZDIM="$ZDIM",PPCA_EM_ITERS="$PPCA_EM_ITERS",SEED="$SEED",GPU_GB="$GPU_GB",FORCE="$FORCE" \
        scripts/ppca_pipeline_compare.sbatch
    )"
    echo "$jid $dataset contrast_std=$contrast" | tee -a "$MANIFEST"
  done
done

echo
echo "Submitted jobs:"
cat "$MANIFEST"
echo
echo "Manifest: $MANIFEST"
