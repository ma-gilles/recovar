#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/gpfs/GILLES/mg6942/recovar_wt_agent_precond_study_20260330}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-/scratch/gpfs/GILLES/mg6942/realdata_cov_ppca_compare_${STAMP}}"
ZDIM="${ZDIM:-10}"
PPCA_EM_ITERS="${PPCA_EM_ITERS:-20}"
GPU_GB="${GPU_GB:-40}"
ANALYZE_MODE="${ANALYZE_MODE:-umap}"

DATASETS=(10073 10076 10180 10345)
METHODS=(covariance ppca ppca_projected_covariance)

cd "$REPO_ROOT"
mkdir -p "$RESULTS_ROOT"
mkdir -p /scratch/gpfs/GILLES/mg6942/slurmo

MANIFEST="$RESULTS_ROOT/slurm_jobs.txt"
: > "$MANIFEST"

for dataset in "${DATASETS[@]}"; do
  for method in "${METHODS[@]}"; do
    jid="$(
      sbatch --parsable \
        --job-name="real-${dataset}-${method}" \
        --export=ALL,REPO_ROOT="$REPO_ROOT",DATASET_ID="$dataset",METHOD="$method",RESULTS_ROOT="$RESULTS_ROOT",ZDIM="$ZDIM",PPCA_EM_ITERS="$PPCA_EM_ITERS",GPU_GB="$GPU_GB",ANALYZE_MODE="$ANALYZE_MODE" \
        scripts/realdata_cov_ppca_focus_compare.sbatch
    )"
    echo "$jid dataset=$dataset method=$method" | tee -a "$MANIFEST"
  done
done

echo "Submitted real-data covariance/PPCA jobs:"
cat "$MANIFEST"
echo "Manifest: $MANIFEST"
