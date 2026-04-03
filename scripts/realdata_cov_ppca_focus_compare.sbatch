#!/bin/bash
#SBATCH --job-name=recovar-real
#SBATCH --account=gilles
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/%x-%j.out

set -euo pipefail

export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="/scratch/gpfs/GILLES/mg6942/tmp/${SLURM_JOB_ID}"
export PIXI_HOME="/scratch/gpfs/GILLES/mg6942/pixi_home/${SLURM_JOB_ID}"
export RATTLER_CACHE_DIR="/scratch/gpfs/GILLES/mg6942/rattler_cache/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$PIXI_HOME" "$RATTLER_CACHE_DIR"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV

REPO_ROOT="${REPO_ROOT:-/scratch/gpfs/GILLES/mg6942/recovar_wt_agent_precond_study_20260330}"
DATASET_ID="${DATASET_ID:?DATASET_ID not set}"
METHOD="${METHOD:?METHOD not set}"
RESULTS_ROOT="${RESULTS_ROOT:?RESULTS_ROOT not set}"
ZDIM="${ZDIM:-10}"
PPCA_EM_ITERS="${PPCA_EM_ITERS:-20}"
GPU_GB="${GPU_GB:-40}"
ANALYZE_MODE="${ANALYZE_MODE:-umap}"

cd "$REPO_ROOT"
PYTHON="$REPO_ROOT/.pixi/envs/default/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo "Missing pixi python at $PYTHON" >&2
  exit 1
fi

"$PYTHON" -c "import pathlib,recovar,jax; repo=pathlib.Path('$REPO_ROOT').resolve(); assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo)+'/'); assert '.pixi/envs/default/' in str(pathlib.Path(jax.__file__).resolve())"

DROPBOX_ROOT="/tigress/CRYOEM/singerlab/mg6942/RECOVAR_datasets"

case "$DATASET_ID" in
  10073)
    PARTICLES="/tigress/CRYOEM/singerlab/mg6942/10073/recovar_data/particles.256.mrcs"
    POSES="$DROPBOX_ROOT/10073/poses.pkl"
    CTF="$DROPBOX_ROOT/10073/ctf.pkl"
    MASK="$DROPBOX_ROOT/10073/mask.mrc"
    FOCUS_MASK="$DROPBOX_ROOT/10073/focus_mask.mrc"
    IND="$DROPBOX_ROOT/10073/ind.pkl"
    ;;
  10076)
    PARTICLES="/tigress/CRYOEM/singerlab/mg6942/10076/particles.256.mrcs"
    POSES="$DROPBOX_ROOT/10076/poses.pkl"
    CTF="$DROPBOX_ROOT/10076/ctf.pkl"
    MASK="$DROPBOX_ROOT/10076/mask.mrc"
    FOCUS_MASK="/scratch/gpfs/AMITS/mg6942/cryodrgn_empiar/empiar10076/inputs/recovar_masks/mask_10076.mrc"
    IND=""
    ;;
  10180)
    PARTICLES="/scratch/gpfs/AMITS/mg6942/cryodrgn_empiar/empiar10180/inputs/particles.256.mrcs"
    POSES="$DROPBOX_ROOT/10180/poses.pkl"
    CTF="$DROPBOX_ROOT/10180/ctf.pkl"
    MASK="$DROPBOX_ROOT/10180/mask.mrc"
    FOCUS_MASK="$DROPBOX_ROOT/10180/focus_mask.mrc"
    IND="$DROPBOX_ROOT/10180/filtered.ind.pkl"
    ;;
  10345)
    PARTICLES="/tigress/CRYOEM/singerlab/mg6942/10345/recovar_data/particles.256.mrcs"
    POSES="$DROPBOX_ROOT/10345/poses.pkl"
    CTF="$DROPBOX_ROOT/10345/ctf.pkl"
    MASK="$DROPBOX_ROOT/10345/mask.mrc"
    FOCUS_MASK="$DROPBOX_ROOT/10345/focus_mask.mrc"
    IND="$DROPBOX_ROOT/10345/ind.pkl"
    ;;
  *)
    echo "Unsupported DATASET_ID=$DATASET_ID" >&2
    exit 1
    ;;
esac

OUT_DIR="$RESULTS_ROOT/${DATASET_ID}/${METHOD}"
LOG_DIR="$RESULTS_ROOT/${DATASET_ID}/${METHOD}/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

PIPELINE_CMD=(
  "$PYTHON" -m recovar.commands.pipeline
  "$PARTICLES"
  -o "$OUT_DIR"
  --poses "$POSES"
  --ctf "$CTF"
  --mask "$MASK"
  --focus-mask "$FOCUS_MASK"
  --correct-contrast
  --zdim "$ZDIM"
  --gpu-gb "$GPU_GB"
  --lazy
)

if [ -n "$IND" ]; then
  PIPELINE_CMD+=(--ind "$IND")
fi

if [ "$METHOD" = "ppca" ]; then
  PIPELINE_CMD+=(--use-ppca --ppca-zdim "$ZDIM" --ppca-em-iters "$PPCA_EM_ITERS")
elif [ "$METHOD" = "ppca_projected_covariance" ]; then
  PIPELINE_CMD+=(--use-ppca --ppca-zdim "$ZDIM" --ppca-em-iters "$PPCA_EM_ITERS" --ppca-projected-covariance)
elif [ "$METHOD" != "covariance" ]; then
  echo "Unsupported METHOD=$METHOD" >&2
  exit 1
fi

ANALYZE_CMD=(
  "$PYTHON" -m recovar.commands.analyze
  "$OUT_DIR"
  --zdim "$ZDIM"
  --n-clusters 20
)

case "$ANALYZE_MODE" in
  umap)
    ANALYZE_CMD+=(--n-trajectories 0 --skip-centers)
    ;;
  full)
    ANALYZE_CMD+=(--n-trajectories 2)
    ;;
  *)
    echo "Unsupported ANALYZE_MODE=$ANALYZE_MODE" >&2
    exit 1
    ;;
esac

echo "Running pipeline: ${PIPELINE_CMD[*]}"
"${PIPELINE_CMD[@]}" 2>&1 | tee "$LOG_DIR/pipeline.log"

echo "Running analyze: ${ANALYZE_CMD[*]}"
"${ANALYZE_CMD[@]}" 2>&1 | tee "$LOG_DIR/analyze.log"
