#!/usr/bin/env bash
# Submit the full-atom spike experiment jobs.
#
# Typical usage after download_student_spike_experiment.sh:
#   source /scratch/gpfs/CRYOEM/gilleslab/tmp/$USER/spike_fullatom_student/student_spike_env.sh
#   $RECOVAR_CHECKOUT/scripts/experiments/spike_fullatom_state_sweeps/run_student_spike_experiment.sh smoke
#   $RECOVAR_CHECKOUT/scripts/experiments/spike_fullatom_state_sweeps/run_student_spike_experiment.sh full
#   $RECOVAR_CHECKOUT/scripts/experiments/spike_fullatom_state_sweeps/run_student_spike_experiment.sh postprocess

set -euo pipefail

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1

ACTION="${1:-smoke}"

RECOVAR_CHECKOUT="${RECOVAR_CHECKOUT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd -P)}"
RECOVAR_STUDENT_ROOT="${RECOVAR_STUDENT_ROOT:-/scratch/gpfs/CRYOEM/gilleslab/tmp/${USER}/spike_fullatom_student}"
PDB_DIR="${PDB_DIR:-/projects/CRYOEM/singerlab/mg6942/spike_morph_pdbs}"
DEFAULT_MASK="${DEFAULT_MASK:-/scratch/gpfs/CRYOEM/gilleslab/tmp/spike_fullatom_direct_volume_shell_metrics_20260523/full_gt_vols_plus_masks_20260524/masks/broad_mask.mrc}"

NOISE_LEVEL="${NOISE_LEVEL:-100.0}"
RENDER_BFACTOR="${RENDER_BFACTOR:-80}"
N_IMAGES_VALUES_STR="${N_IMAGES_VALUES_STR:-10000 30000 100000 300000 1000000}"

FULL_ROOT="${FULL_ROOT:-$RECOVAR_STUDENT_ROOT/spike_fullatom_consistency_grid256_noise100_b80}"
FULL_SHARED="${FULL_SHARED:-$RECOVAR_STUDENT_ROOT/spike_fullatom_consistency_grid256_noise100_b80_shared}"
SMOKE_ROOT="${SMOKE_ROOT:-$RECOVAR_STUDENT_ROOT/spike_smoke_noise100_b80}"
SMOKE_SHARED="${SMOKE_SHARED:-$RECOVAR_STUDENT_ROOT/spike_smoke_noise100_b80_shared}"

mkdir -p "$RECOVAR_STUDENT_ROOT/slurmo"
cd "$RECOVAR_CHECKOUT"

print_sweep_paths() {
  local root="$1"
  local shared="$2"
  local image_counts="$3"
  local count
  local run_label

  echo
  echo "Outputs will be written under:"
  echo "  $root"
  echo "Shared setup/cache output:"
  echo "  $shared"
  echo "Slurm logs:"
  echo "  $RECOVAR_STUDENT_ROOT/slurmo"
  echo "Per-size run directories:"
  for count in $image_counts; do
    printf -v run_label "n%08d" "$count"
    echo "  $root/$run_label/runs/${run_label}_seed0000"
  done
  echo
}

submit_compute_sweep() {
  local array_spec="$1"
  local root="$2"
  local shared="$3"
  local n_images="$4"
  sbatch \
    --array="$array_spec" \
    --output="$RECOVAR_STUDENT_ROOT/slurmo/%x-%A_%a.out" \
    --error="$RECOVAR_STUDENT_ROOT/slurmo/%x-%A_%a.err" \
    --export=ALL,WORKDIR="$RECOVAR_CHECKOUT",SCRATCH_ROOT="$RECOVAR_STUDENT_ROOT",BASE_ROOT="$root",SHARED_ROOT="$shared",PDB_DIR="$PDB_DIR",MASK="$DEFAULT_MASK",N_IMAGES_VALUES_STR="$n_images",NOISE_LEVEL="$NOISE_LEVEL",RENDER_BFACTOR="$RENDER_BFACTOR" \
    scripts/experiments/spike_fullatom_state_sweeps/submit_fullatom_noise100_b80_dataset_size.sbatch
}

resolve_plot100k_run_dir() {
  local state_file="07_compute_state/state000_unfil.mrc"
  if [[ -n "${PLOT_RUN_DIR:-}" ]]; then
    if [[ ! -f "$PLOT_RUN_DIR/$state_file" ]]; then
      echo "PLOT_RUN_DIR does not contain $state_file: $PLOT_RUN_DIR" >&2
      exit 2
    fi
    echo "$PLOT_RUN_DIR"
    return
  fi

  local default_run_dir="$FULL_ROOT/n00100000/runs/n00100000_seed0000"
  if [[ -f "$default_run_dir/$state_file" ]]; then
    echo "$default_run_dir"
    return
  fi

  local candidates=()
  while IFS= read -r state_path; do
    candidate="$(dirname "$(dirname "$state_path")")"
    candidates+=("$candidate")
  done < <(
    find "$RECOVAR_STUDENT_ROOT" -maxdepth 8 -type f \
      -path "*/n00100000/runs/n00100000_seed0000/$state_file" | sort
  )

  if [[ "${#candidates[@]}" -eq 1 ]]; then
    echo "${candidates[0]}"
    return
  fi

  if [[ "${#candidates[@]}" -eq 0 ]]; then
    echo "Could not find a completed 100k run dir with $state_file at $default_run_dir or under $RECOVAR_STUDENT_ROOT." >&2
  else
    echo "Found multiple completed 100k run dirs under $RECOVAR_STUDENT_ROOT:" >&2
    printf '  %s\n' "${candidates[@]}" >&2
  fi
  echo "Set PLOT_RUN_DIR=/path/to/n00100000_seed0000 or FULL_ROOT=/path/to/sweep-root and retry." >&2
  exit 2
}

case "$ACTION" in
  smoke)
    echo "Submitting one 10k smoke run"
    print_sweep_paths "$SMOKE_ROOT" "$SMOKE_SHARED" "10000"
    submit_compute_sweep "0-0" "$SMOKE_ROOT" "$SMOKE_SHARED" "10000"
    ;;
  full)
    read -r -a image_counts <<< "$N_IMAGES_VALUES_STR"
    last_index=$((${#image_counts[@]} - 1))
    echo "Submitting full sweep"
    echo "Image counts: $N_IMAGES_VALUES_STR"
    print_sweep_paths "$FULL_ROOT" "$FULL_SHARED" "$N_IMAGES_VALUES_STR"
    submit_compute_sweep "0-${last_index}%3" "$FULL_ROOT" "$FULL_SHARED" "$N_IMAGES_VALUES_STR"
    ;;
  postprocess)
    echo "Submitting postprocess"
    echo
    echo "Postprocess reads:"
    echo "  $FULL_ROOT"
    echo "Postprocess writes plots under:"
    echo "  $FULL_ROOT/plots"
    echo "Slurm logs:"
    echo "  $RECOVAR_STUDENT_ROOT/slurmo"
    echo
    sbatch \
      --output="$RECOVAR_STUDENT_ROOT/slurmo/%x-%j.out" \
      --error="$RECOVAR_STUDENT_ROOT/slurmo/%x-%j.err" \
      --export=ALL,WORKDIR="$RECOVAR_CHECKOUT",SCRATCH_ROOT="$RECOVAR_STUDENT_ROOT",ROOT="$FULL_ROOT",LABEL="$(basename "$FULL_ROOT")",TARGET_STATE=50,MASK="$DEFAULT_MASK" \
      scripts/experiments/spike_fullatom_state_sweeps/postprocess_fullatom_dataset_size.sbatch
    ;;
  plot100k)
    plot_run_dir="$(resolve_plot100k_run_dir)"
    echo "Plotting compute_state shell metrics"
    echo
    echo "Plot input run:"
    echo "  $plot_run_dir"
    echo "Plot output directory:"
    echo "  $plot_run_dir/plots/compute_state_shell_metrics"
    echo
    "$RECOVAR_CHECKOUT/.pixi/envs/default/bin/python" \
      scripts/experiments/spike_fullatom_state_sweeps/plot_compute_state_shell_metrics.py \
      --run-dir "$plot_run_dir"
    ;;
  *)
    echo "Unknown action: $ACTION" >&2
    echo "Use one of: smoke, full, postprocess, plot100k" >&2
    exit 2
    ;;
esac

echo "Logs: $RECOVAR_STUDENT_ROOT/slurmo"
echo "Queue: squeue -u $USER"
