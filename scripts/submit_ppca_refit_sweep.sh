#!/bin/bash
set -euo pipefail

# Submit PPCA refit sweep: runs all new methods on all datasets
# Reuses existing PPCA results from the reference isolated run.

REPO_ROOT="${REPO_ROOT:-/scratch/gpfs/GILLES/mg6942/recovar_wt_ppca_refit}"
REFERENCE_ROOT="${REFERENCE_ROOT:-/scratch/gpfs/GILLES/mg6942/ppca_pipeline_compare_projcov_isolated_20260404_111212}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-/scratch/gpfs/GILLES/mg6942/ppca_refit_sweep_${STAMP}}"
ZDIM="${ZDIM:-10}"
BATCH_SIZE="${BATCH_SIZE:-128}"

# Datasets (no-contrast only)
DATASETS=(Ribosembly IgG-1D IgG-RL Tomotwin-100)
CONTRAST="0.0"

# Methods to run
# Post-processing (fast, ~10 min): refit_b, temperature_scalar, temperature_diag
# Iterative (slow, ~1-2 hrs): stiefel_ub, whitening_manifold_ub, coord_reg_grid, coord_reg_physical
METHODS=(refit_b temperature_scalar temperature_diag stiefel_ub whitening_manifold_ub coord_reg_grid coord_reg_physical)

if [ -n "${METHODS_CSV:-}" ]; then
  IFS=',' read -r -a METHODS <<< "$METHODS_CSV"
fi

cd "$REPO_ROOT"

echo "Submitting PPCA refit sweep"
echo "  REPO_ROOT=$REPO_ROOT"
echo "  REFERENCE_ROOT=$REFERENCE_ROOT"
echo "  RESULTS_ROOT=$RESULTS_ROOT"
echo "  DATASETS=${DATASETS[*]}"
echo "  METHODS=${METHODS[*]}"
echo "  ZDIM=$ZDIM BATCH_SIZE=$BATCH_SIZE"
echo

mkdir -p "$RESULTS_ROOT"
mkdir -p /scratch/gpfs/GILLES/mg6942/slurmo

MANIFEST="$RESULTS_ROOT/slurm_jobs.txt"
: > "$MANIFEST"
ALL_JOB_IDS=()

build_run_name() {
  local dataset="$1"
  printf '%s_g128_n100000_snr%s_c0p00_z%s_seed42' "$dataset" "${SNR:-1.0}" "$ZDIM"
}

for dataset in "${DATASETS[@]}"; do
  safe_dataset="${dataset//[^A-Za-z0-9]/-}"
  run_name="$(build_run_name "$dataset")"

  # PPCA result from reference run
  ppca_result_dir="${REFERENCE_ROOT}/${run_name}/ppca/result"
  if [ ! -f "$ppca_result_dir/model/params.pkl" ]; then
    echo "WARNING: Missing PPCA result for $dataset at $ppca_result_dir" >&2
    continue
  fi

  # Create dataset output directory
  dataset_dir="$RESULTS_ROOT/$run_name"
  mkdir -p "$dataset_dir"

  # Symlink reference data for scoring later
  ref_dir="${REFERENCE_ROOT}/${run_name}"
  for f in simulated_data halfsets.pkl volumes_fixed dataset_config.json; do
    src="$ref_dir/$f"
    dst="$dataset_dir/$f"
    if [ -e "$src" ] && [ ! -e "$dst" ]; then
      ln -s "$src" "$dst"
    fi
  done

  # Symlink existing method results for scoring
  for existing_method in covariance ppca ppca_projected_covariance; do
    src="$ref_dir/$existing_method"
    dst="$dataset_dir/$existing_method"
    if [ -d "$src" ] && [ ! -e "$dst" ]; then
      ln -s "$src" "$dst"
    fi
  done

  for method in "${METHODS[@]}"; do
    short_method="${method//_/-}"
    job_name="refit-${safe_dataset}-${short_method}"
    output_dir="$dataset_dir/${method}/result"

    # Skip if already completed
    if [ -f "$output_dir/model/params.pkl" ]; then
      echo "SKIP: $dataset $method (already exists)"
      continue
    fi

    jid="$(
      sbatch --parsable \
        --job-name="$job_name" \
        --export=ALL,REPO_ROOT="$REPO_ROOT",PPCA_RESULT_DIR="$ppca_result_dir",METHOD="$method",OUTPUT_DIR="$output_dir",ZDIM="$ZDIM",BATCH_SIZE="$BATCH_SIZE" \
        scripts/ppca_refit.sbatch
    )"
    ALL_JOB_IDS+=("$jid")
    echo "$jid $dataset method=$method output=$output_dir" | tee -a "$MANIFEST"
  done
done

if [ ${#ALL_JOB_IDS[@]} -eq 0 ]; then
  echo "No jobs to submit (all already completed)"
  exit 0
fi

# Submit scoring job that depends on all method jobs
dep=$(IFS=:; echo "${ALL_JOB_IDS[*]}")
score_jid="$(
  sbatch --parsable \
    --job-name="refit-score-all" \
    --dependency="afterany:${dep}" \
    --account=gilles \
    --partition=cryoem \
    --gres=gpu:1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem=500G \
    --time=04:00:00 \
    --exclusive \
    --output=/scratch/gpfs/GILLES/mg6942/slurmo/%x-%j.out \
    --export=ALL,REPO_ROOT="$REPO_ROOT",RESULTS_ROOT="$RESULTS_ROOT",REFERENCE_ROOT="$REFERENCE_ROOT",ZDIM="$ZDIM" \
    --wrap="cd '$REPO_ROOT' && export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false TMPDIR='/scratch/gpfs/GILLES/mg6942/tmp/\${SLURM_JOB_ID}' PIXI_HOME='/scratch/gpfs/GILLES/mg6942/pixi_home/\${SLURM_JOB_ID}' RATTLER_CACHE_DIR='/scratch/gpfs/GILLES/mg6942/rattler_cache/\${SLURM_JOB_ID}' && mkdir -p \"\$TMPDIR\" \"\$PIXI_HOME\" \"\$RATTLER_CACHE_DIR\" && unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV && ./.pixi/envs/default/bin/python -m recovar.ppca.score_refit_sweep '$RESULTS_ROOT' --reference-root '$REFERENCE_ROOT' --zdim $ZDIM --output-dir '$RESULTS_ROOT/aggregate'"
)"
echo "$score_jid scoring results_root=$RESULTS_ROOT" | tee -a "$MANIFEST"

echo
echo "Submitted ${#ALL_JOB_IDS[@]} method jobs + 1 scoring job"
echo "Manifest: $MANIFEST"
echo "Score job: $score_jid (depends on all method jobs)"
