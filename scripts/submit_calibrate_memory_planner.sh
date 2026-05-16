#!/usr/bin/env bash
# Submit the empirical memory-planner calibration sweep on Della.
#
# Each (command, grid, n_pcs, backend) cell becomes one Slurm job.
# After all cells finish, run:
#
#   pixi run python scripts/aggregate_memory_calibration.py
#
# to populate recovar/utils/memory_calibration_data.json.
#
# This is a manual one-shot whose output is checked into the repo;
# CI does not need to re-run it.
#
# compute_state calibration is not wired yet; leave COMMANDS=pipeline
# unless that branch is implemented in calibrate_memory_planner.py.

set -euo pipefail

WORKDIR="${WORKDIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
RUNS_ROOT="${RUNS_ROOT:-/scratch/gpfs/GILLES/mg6942/calibration_runs}"
DATASETS_ROOT="${DATASETS_ROOT:-/scratch/gpfs/GILLES/mg6942/calibration_datasets}"
N_IMAGES="${N_IMAGES:-20000}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-gilles}"
SLURM_PARTITION="${SLURM_PARTITION:-cryoem}"
TIME_LIMIT="${TIME_LIMIT:-2:00:00}"

mkdir -p "$RUNS_ROOT/cells" "$RUNS_ROOT/logs" "$RUNS_ROOT/sbatch"

GRIDS=(${GRIDS:-64 128 256})
N_PCS_LIST=(${N_PCS_LIST:-4 20 50 200})
BACKENDS=(${BACKENDS:-custom_cuda jax_fallback})
COMMANDS=(${COMMANDS:-pipeline})

submitted=0
for cmd in "${COMMANDS[@]}"; do
  for grid in "${GRIDS[@]}"; do
    for n_pcs in "${N_PCS_LIST[@]}"; do
      for backend in "${BACKENDS[@]}"; do
        cell_id="${cmd}_g${grid}_n${n_pcs}_${backend}"
        out_json="$RUNS_ROOT/cells/${cell_id}.json"
        if [ -f "$out_json" ]; then
          echo "skip (already done): $cell_id"
          continue
        fi
        sbatch_script="$RUNS_ROOT/sbatch/${cell_id}.sh"
        cat > "$sbatch_script" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=recovar-cal-${cell_id}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200GB
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=$RUNS_ROOT/logs/${cell_id}-%j.out
#SBATCH --exclusive

set -euo pipefail
cd "${WORKDIR}"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.90
export TMPDIR="/scratch/gpfs/GILLES/mg6942/tmp/calibrate_\${SLURM_JOB_ID}"
export PIXI_HOME="/scratch/gpfs/GILLES/mg6942/pixi_home/calibrate_\${SLURM_JOB_ID}"
export RATTLER_CACHE_DIR="/scratch/gpfs/GILLES/mg6942/rattler_cache/calibrate_\${SLURM_JOB_ID}"
export RECOVAR_CUDA_CACHE_DIR="/scratch/gpfs/GILLES/mg6942/recovar_cuda_cache/calibrate_\${SLURM_JOB_ID}"
mkdir -p "\$TMPDIR" "\$PIXI_HOME" "\$RATTLER_CACHE_DIR" "\$RECOVAR_CUDA_CACHE_DIR"
if [ -x /usr/local/cuda-12.8/bin/nvcc ]; then
  export CUDA_HOME=/usr/local/cuda-12.8
  export CUDA_PATH=/usr/local/cuda-12.8
  export NVCC=/usr/local/cuda-12.8/bin/nvcc
  export CUDACXX="\$NVCC"
  export PATH="/usr/local/cuda-12.8/bin:\$PATH"
elif ! command -v nvcc >/dev/null 2>&1; then
  echo "FAIL: nvcc not found; set NVCC/CUDACXX or install cudatoolkit/12.8"
  exit 1
fi

if [ "${backend}" = "custom_cuda" ]; then
  pixi run python -c "from recovar.cuda_backproject import cuda_available; raise SystemExit(0 if cuda_available() else 'custom CUDA unavailable')"
fi

pixi run python scripts/calibrate_memory_planner.py \\
    --command ${cmd} \\
    --grid-size ${grid} \\
    --n-pcs ${n_pcs} \\
    --backend ${backend} \\
    --n-images ${N_IMAGES} \\
    --dataset-root ${DATASETS_ROOT} \\
    --out-json ${out_json} \\
    --out-runs-root ${RUNS_ROOT}/runs
EOF
        chmod +x "$sbatch_script"
        job_id=$(sbatch --parsable "$sbatch_script")
        echo "submitted: $cell_id (job $job_id)"
        submitted=$((submitted+1))
      done
    done
  done
done

echo
echo "Submitted $submitted calibration jobs."
echo "Watch progress: squeue -u \$USER --name=recovar-cal-"
echo "When complete: pixi run python scripts/aggregate_memory_calibration.py"
