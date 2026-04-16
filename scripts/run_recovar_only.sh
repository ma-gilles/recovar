#!/bin/bash
#SBATCH --job-name=recovar-only
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=300G
#SBATCH --time=2:00:00
#SBATCH --exclude=della-h19g3
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/recovar-only-%j.out

set -euo pipefail

export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="/scratch/gpfs/GILLES/mg6942/tmp/${SLURM_JOB_ID}"
export PIXI_HOME="/scratch/gpfs/GILLES/mg6942/pixi_home/${SLURM_JOB_ID}"
export RATTLER_CACHE_DIR="/scratch/gpfs/GILLES/mg6942/rattler_cache/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$PIXI_HOME" "$RATTLER_CACHE_DIR" /scratch/gpfs/GILLES/mg6942/slurmo

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
unset LD_LIBRARY_PATH

REPO_DIR="${REPO_DIR:-/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar}"
DATA_DIR="${DATA_DIR:-/scratch/gpfs/GILLES/mg6942/tmp/em_profile/data}"
RELION_REF_DIR="$DATA_DIR/relion_ref_benchmark"
TAG="${TAG:-v26}"
OUTPUT_DIR="$DATA_DIR/runs/${TAG}_$(date +%Y%m%d_%H%M%S)"
MAX_ITER="${MAX_ITER:-3}"

cd "$REPO_DIR"
git rev-parse HEAD

pixi install
PIXI_PY="$(pixi run which python)"
"$PIXI_PY" -m pip uninstall -y recovar || true
"$PIXI_PY" -m pip install -e . --no-deps --no-build-isolation --ignore-installed
PYTHON="$PIXI_PY" make -C recovar/cuda clean all
pixi run smoke-import-recovar

mkdir -p "$OUTPUT_DIR"

echo "=== Run recovar full refinement in RELION mode ==="
"$PIXI_PY" scripts/run_full_refinement.py \
  --data_dir "$DATA_DIR" \
  --output "$OUTPUT_DIR" \
  --mode relion \
  --max_iter "$MAX_ITER" \
  --healpix_order 3 \
  --offset_range 3 --offset_step 1 \
  --adaptive_oversampling 1 \
  --adaptive_fraction 0.999 \
  --max_significants -1 \
  --adaptive_skip_threshold 0.5 \
  --relion_half_sets "$RELION_REF_DIR/run_it001_data.star"

echo "=== Compare recovar vs RELION reference ==="
"$PIXI_PY" scripts/compare_vs_relion.py \
  --our_results "$OUTPUT_DIR" \
  --relion_ref_npz "$DATA_DIR/relion_ref_npz_benchmark" \
  --relion_ref_star "$RELION_REF_DIR" \
  | tee "$OUTPUT_DIR/compare_report.txt"

echo "=== DONE: $OUTPUT_DIR ==="
