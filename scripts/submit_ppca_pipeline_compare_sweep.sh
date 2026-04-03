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
COVARIANCE_GPU_GB="${COVARIANCE_GPU_GB:-$GPU_GB}"
PPCA_GPU_GB="${PPCA_GPU_GB:-$GPU_GB}"
COVARIANCE_LOW_MEMORY_OPTION="${COVARIANCE_LOW_MEMORY_OPTION:-0}"
COVARIANCE_VERY_LOW_MEMORY_OPTION="${COVARIANCE_VERY_LOW_MEMORY_OPTION:-0}"
PPCA_LOW_MEMORY_OPTION="${PPCA_LOW_MEMORY_OPTION:-0}"
PPCA_VERY_LOW_MEMORY_OPTION="${PPCA_VERY_LOW_MEMORY_OPTION:-0}"
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
echo "  COVARIANCE_GPU_GB=$COVARIANCE_GPU_GB PPCA_GPU_GB=$PPCA_GPU_GB"
echo "  COVARIANCE_LOW_MEMORY_OPTION=$COVARIANCE_LOW_MEMORY_OPTION COVARIANCE_VERY_LOW_MEMORY_OPTION=$COVARIANCE_VERY_LOW_MEMORY_OPTION"
echo "  PPCA_LOW_MEMORY_OPTION=$PPCA_LOW_MEMORY_OPTION PPCA_VERY_LOW_MEMORY_OPTION=$PPCA_VERY_LOW_MEMORY_OPTION"
echo

mkdir -p "$RESULTS_ROOT"
mkdir -p /scratch/gpfs/GILLES/mg6942/slurmo

MANIFEST="$RESULTS_ROOT/slurm_jobs.txt"
: > "$MANIFEST"
JOB_IDS=()

for dataset in "${DATASETS[@]}"; do
  safe_dataset="${dataset//[^A-Za-z0-9]/-}"
  for contrast in "${CONTRASTS[@]}"; do
    ctag="${contrast/./p}"
    job_name="pcmp-${safe_dataset}-c${ctag}"
    jid="$(
      sbatch --parsable \
        --job-name="$job_name" \
        --export=ALL,REPO_ROOT="$REPO_ROOT",BASE_DIR="$BASE_DIR",RESULTS_ROOT="$RESULTS_ROOT",DATASET="$dataset",GRID_SIZE="$GRID_SIZE",N_IMAGES="$N_IMAGES",NOISE_LEVEL="$NOISE_LEVEL",CONTRAST_STD="$contrast",ZDIM="$ZDIM",PPCA_EM_ITERS="$PPCA_EM_ITERS",SEED="$SEED",GPU_GB="$GPU_GB",COVARIANCE_GPU_GB="$COVARIANCE_GPU_GB",PPCA_GPU_GB="$PPCA_GPU_GB",COVARIANCE_LOW_MEMORY_OPTION="$COVARIANCE_LOW_MEMORY_OPTION",COVARIANCE_VERY_LOW_MEMORY_OPTION="$COVARIANCE_VERY_LOW_MEMORY_OPTION",PPCA_LOW_MEMORY_OPTION="$PPCA_LOW_MEMORY_OPTION",PPCA_VERY_LOW_MEMORY_OPTION="$PPCA_VERY_LOW_MEMORY_OPTION",FORCE="$FORCE" \
        scripts/ppca_pipeline_compare.sbatch
    )"
    JOB_IDS+=("$jid")
    echo "$jid $dataset contrast_std=$contrast" | tee -a "$MANIFEST"
  done
done

dep=$(IFS=:; echo "${JOB_IDS[*]}")
aggregate_jid="$(
  sbatch --parsable \
    --job-name="pcmp-aggregate" \
    --dependency="afterany:${dep}" \
    --export=ALL,REPO_ROOT="$REPO_ROOT",RESULTS_ROOT="$RESULTS_ROOT" \
    --account=gilles \
    --partition=cryoem \
    --ntasks=1 \
    --cpus-per-task=2 \
    --mem=32G \
    --time=01:00:00 \
    --output=/scratch/gpfs/GILLES/mg6942/slurmo/%x-%j.out \
    --wrap="cd '$REPO_ROOT' && export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false TMPDIR='/scratch/gpfs/GILLES/mg6942/tmp/\${SLURM_JOB_ID}' PIXI_HOME='/scratch/gpfs/GILLES/mg6942/pixi_home/\${SLURM_JOB_ID}' RATTLER_CACHE_DIR='/scratch/gpfs/GILLES/mg6942/rattler_cache/\${SLURM_JOB_ID}' && mkdir -p \"\$TMPDIR\" \"\$PIXI_HOME\" \"\$RATTLER_CACHE_DIR\" && ./.pixi/envs/default/bin/python -m recovar.ppca.summarize_pipeline_compare_sweep '$RESULTS_ROOT' --output-dir '$RESULTS_ROOT/aggregate'" \
)"
echo "$aggregate_jid aggregate results_root=$RESULTS_ROOT" | tee -a "$MANIFEST"

echo
echo "Submitted jobs:"
cat "$MANIFEST"
echo
echo "Manifest: $MANIFEST"
