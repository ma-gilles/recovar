#!/bin/bash
#SBATCH --job-name=relion-parity
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/relion-parity-%j.out

set -euo pipefail

export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="/scratch/gpfs/GILLES/mg6942/tmp/${SLURM_JOB_ID}"
export PIXI_HOME="/scratch/gpfs/GILLES/mg6942/pixi_home/${SLURM_JOB_ID}"
export RATTLER_CACHE_DIR="/scratch/gpfs/GILLES/mg6942/rattler_cache/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$PIXI_HOME" "$RATTLER_CACHE_DIR" /scratch/gpfs/GILLES/mg6942/slurmo

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
unset LD_LIBRARY_PATH

REPO_DIR="${REPO_DIR:-/home/mg6942/recovar_dev/recovar}"
DATA_DIR="${DATA_DIR:-/scratch/gpfs/GILLES/mg6942/tmp/em_profile/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_DIR}/comparison_results}"
OUR_RESULTS_DIR="${OUR_RESULTS_DIR:-${DATA_DIR}/our_results_relion}"
RELION_REF_DIR="${RELION_REF_DIR:-${DATA_DIR}/relion_ref_benchmark}"
RELION_REF_NPZ_DIR="${RELION_REF_NPZ_DIR:-${DATA_DIR}/relion_ref_npz_benchmark}"
RELION_RUN_PREFIX="${RELION_RUN_PREFIX:-${RELION_REF_DIR}/run}"
RELION_MODULE="${RELION_MODULE:-relion/5.0.1/gcc-11.5.0-gpu}"
MAX_ITER="${MAX_ITER:-10}"
ADAPTIVE_OVERSAMPLING="${ADAPTIVE_OVERSAMPLING:-1}"
ADAPTIVE_FRACTION="${ADAPTIVE_FRACTION:-0.999}"
MAX_SIGNIFICANTS="${MAX_SIGNIFICANTS:--1}"
ADAPTIVE_SKIP_THRESHOLD="${ADAPTIVE_SKIP_THRESHOLD:--1}"
HEALPIX_ORDER="${HEALPIX_ORDER:-3}"
OFFSET_RANGE="${OFFSET_RANGE:-3.0}"
OFFSET_STEP="${OFFSET_STEP:-1.0}"
RELION_MPI_RANKS="${RELION_MPI_RANKS:-3}"

cd "$REPO_DIR"

echo "=== Repo provenance ==="
pwd
whoami
git rev-parse HEAD
git status --short

echo "=== Pixi setup ==="
pixi install
PIXI_PY="$(pixi run which python)"
"$PIXI_PY" -m pip uninstall -y recovar || true
"$PIXI_PY" -m pip install -e . --no-deps --no-build-isolation --ignore-installed
PYTHON="$PIXI_PY" make -C recovar/cuda clean all
pixi run smoke-import-recovar
"$PIXI_PY" -c "import pathlib,recovar,jax; repo=pathlib.Path.cwd().resolve(); assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo)+'/'); assert '.pixi/envs/default/' in str(pathlib.Path(jax.__file__).resolve())"

echo "=== Prepare benchmark dataset ==="
"$PIXI_PY" scripts/prepare_relion_parity_benchmark.py \
  --output-dir "$DATA_DIR" \
  --n-images 5000 \
  --grid-size 128 \
  --noise-level 1.0

mkdir -p "$RELION_REF_DIR" "$RELION_REF_NPZ_DIR" "$OUTPUT_DIR" "$OUR_RESULTS_DIR"

echo "=== Load RELION ==="
(
  export PS1="${PS1-}"
  module load "$RELION_MODULE"
  command -v relion_refine_mpi
  command -v mpirun

  export CUDA_VISIBLE_DEVICES=0

  if [ ! -f "${RELION_RUN_PREFIX}_optimiser.star" ] || \
     [ ! -f "$RELION_REF_DIR/run_class001.mrc" ] || \
     [ ! -f "$RELION_REF_DIR/run_half1_class001_unfil.mrc" ] || \
     [ ! -f "$RELION_REF_DIR/run_half2_class001_unfil.mrc" ]; then
    echo "=== Run RELION auto-refine ==="
    cd "$DATA_DIR"
    mpirun -n "$RELION_MPI_RANKS" relion_refine_mpi \
      --i particles.star \
      --ref reference_init.mrc \
      --o "$RELION_RUN_PREFIX" \
      --auto_refine \
      --split_random_halves \
      --particle_diameter 200 \
      --ini_high 30 \
      --healpix_order "$HEALPIX_ORDER" \
      --offset_range "$OFFSET_RANGE" \
      --offset_step "$OFFSET_STEP" \
      --oversampling 1 \
      --pad 2 \
      --gpu 0 \
      --j 4
  else
    echo "=== Reusing existing RELION output in $RELION_REF_DIR ==="
  fi
)

unset LD_LIBRARY_PATH

echo "=== Extract RELION reference ==="
"$PIXI_PY" scripts/extract_relion_reference.py \
  "$RELION_RUN_PREFIX" \
  "$RELION_REF_NPZ_DIR"

echo "=== Run recovar full refinement in RELION mode ==="
"$PIXI_PY" scripts/run_full_refinement.py \
  --data_dir "$DATA_DIR" \
  --output "$OUR_RESULTS_DIR" \
  --mode relion \
  --max_iter "$MAX_ITER" \
  --healpix_order "$HEALPIX_ORDER" \
  --offset_range "$OFFSET_RANGE" \
  --offset_step "$OFFSET_STEP" \
  --adaptive_oversampling "$ADAPTIVE_OVERSAMPLING" \
  --adaptive_fraction "$ADAPTIVE_FRACTION" \
  --max_significants "$MAX_SIGNIFICANTS" \
  --adaptive_skip_threshold "$ADAPTIVE_SKIP_THRESHOLD" \
  --relion_half_sets "${RELION_RUN_PREFIX}_it001_data.star"

echo "=== Compare recovar refinement against RELION reference ==="
"$PIXI_PY" scripts/compare_vs_relion.py \
  --our_results "$OUR_RESULTS_DIR" \
  --relion_ref_npz "$RELION_REF_NPZ_DIR" \
  --relion_ref_star "$RELION_REF_DIR" \
  | tee "$OUTPUT_DIR/compare_vs_relion_report.txt"

echo "=== Run head-to-head comparison (own FSC + oracle) ==="
"$PIXI_PY" scripts/run_comparison.py \
  --data_dir "$DATA_DIR" \
  --output "$OUTPUT_DIR" \
  --mode relion \
  --max_iter "$MAX_ITER" \
  --adaptive_oversampling "$ADAPTIVE_OVERSAMPLING" \
  --adaptive_fraction "$ADAPTIVE_FRACTION" \
  --max_significants "$MAX_SIGNIFICANTS" \
  --adaptive_skip_threshold "$ADAPTIVE_SKIP_THRESHOLD" \
  --healpix_order "$HEALPIX_ORDER" \
  --offset_range "$OFFSET_RANGE" \
  --offset_step "$OFFSET_STEP" \
  | tee "$OUTPUT_DIR/run_comparison_report.txt"

echo "=== Benchmark complete ==="
echo "Dataset: $DATA_DIR"
echo "RELION output: $RELION_REF_DIR"
echo "RELION NPZ: $RELION_REF_NPZ_DIR"
echo "Refinement output: $OUR_RESULTS_DIR"
echo "Comparison output: $OUTPUT_DIR"
