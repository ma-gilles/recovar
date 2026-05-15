#!/bin/bash
#SBATCH --job-name=sp128-hard-lp
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=300G
#SBATCH --time=12:00:00
#SBATCH --array=0-17%8
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/spike128_ctf_noise0p1_basis_hard_%A_%a.out
#SBATCH --error=/scratch/gpfs/GILLES/mg6942/slurmo/spike128_ctf_noise0p1_basis_hard_%A_%a.err

set -euo pipefail

cd /scratch/gpfs/GILLES/mg6942/recovar_dev/recovar

export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export RECOVAR_CACHE_DIR=
export TMPDIR="/scratch/gpfs/GILLES/mg6942/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export PIXI_HOME="/scratch/gpfs/GILLES/mg6942/pixi_home/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export RATTLER_CACHE_DIR="/scratch/gpfs/GILLES/mg6942/rattler_cache/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR" "$PIXI_HOME" "$RATTLER_CACHE_DIR" /scratch/gpfs/GILLES/mg6942/slurmo
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV

PYTHON=/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar/.pixi/envs/default/bin/python
RUN_DIR=/scratch/gpfs/GILLES/mg6942/runs/spike_grid128_box320_ctf_noise0p1_n200k_raw_index_embsigma100_20260514

case "$SLURM_ARRAY_TASK_ID" in
  0)
    TAG="basis_hard_d5_i2_q7_monomial_none"
    EXTRA_ARGS="--local-poly-degree 5 --local-poly-em-iterations 2 --local-poly-em-quadrature 7 --local-poly-basis monomial --local-poly-pol-reg-type none"
    ;;
  1)
    TAG="basis_hard_d7_i1_q5_monomial_none"
    EXTRA_ARGS="--local-poly-degree 7 --local-poly-em-iterations 1 --local-poly-em-quadrature 5 --local-poly-basis monomial --local-poly-pol-reg-type none"
    ;;
  2)
    TAG="basis_hard_d7_i1_q7_monomial_none"
    EXTRA_ARGS="--local-poly-degree 7 --local-poly-em-iterations 1 --local-poly-em-quadrature 7 --local-poly-basis monomial --local-poly-pol-reg-type none"
    ;;
  3)
    TAG="basis_hard_d5_i1_q7_monomial_coeff_eta1"
    EXTRA_ARGS="--local-poly-degree 5 --local-poly-em-iterations 1 --local-poly-em-quadrature 7 --local-poly-basis monomial --local-poly-pol-reg-type coeff --local-poly-pol-reg-eta 1 --local-poly-pol-reg-power 2"
    ;;
  4)
    TAG="basis_hard_d5_i2_q5_monomial_coeff_eta1"
    EXTRA_ARGS="--local-poly-degree 5 --local-poly-em-iterations 2 --local-poly-em-quadrature 5 --local-poly-basis monomial --local-poly-pol-reg-type coeff --local-poly-pol-reg-eta 1 --local-poly-pol-reg-power 2"
    ;;
  5)
    TAG="basis_hard_d5_i2_q7_monomial_coeff_eta1"
    EXTRA_ARGS="--local-poly-degree 5 --local-poly-em-iterations 2 --local-poly-em-quadrature 7 --local-poly-basis monomial --local-poly-pol-reg-type coeff --local-poly-pol-reg-eta 1 --local-poly-pol-reg-power 2"
    ;;
  6)
    TAG="basis_hard_d7_i1_q5_monomial_coeff_eta1"
    EXTRA_ARGS="--local-poly-degree 7 --local-poly-em-iterations 1 --local-poly-em-quadrature 5 --local-poly-basis monomial --local-poly-pol-reg-type coeff --local-poly-pol-reg-eta 1 --local-poly-pol-reg-power 2"
    ;;
  7)
    TAG="basis_hard_d7_i1_q7_monomial_coeff_eta1"
    EXTRA_ARGS="--local-poly-degree 7 --local-poly-em-iterations 1 --local-poly-em-quadrature 7 --local-poly-basis monomial --local-poly-pol-reg-type coeff --local-poly-pol-reg-eta 1 --local-poly-pol-reg-power 2"
    ;;
  8)
    TAG="basis_hard_d5_i1_q7_monomial_deriv1_eta0p3"
    EXTRA_ARGS="--local-poly-degree 5 --local-poly-em-iterations 1 --local-poly-em-quadrature 7 --local-poly-basis monomial --local-poly-pol-reg-type deriv1 --local-poly-pol-reg-eta 0.3 --local-poly-pol-reg-power 2"
    ;;
  9)
    TAG="basis_hard_d5_i2_q5_monomial_deriv1_eta0p3"
    EXTRA_ARGS="--local-poly-degree 5 --local-poly-em-iterations 2 --local-poly-em-quadrature 5 --local-poly-basis monomial --local-poly-pol-reg-type deriv1 --local-poly-pol-reg-eta 0.3 --local-poly-pol-reg-power 2"
    ;;
  10)
    TAG="basis_hard_d5_i2_q7_monomial_deriv1_eta0p3"
    EXTRA_ARGS="--local-poly-degree 5 --local-poly-em-iterations 2 --local-poly-em-quadrature 7 --local-poly-basis monomial --local-poly-pol-reg-type deriv1 --local-poly-pol-reg-eta 0.3 --local-poly-pol-reg-power 2"
    ;;
  11)
    TAG="basis_hard_d7_i1_q5_monomial_deriv1_eta0p3"
    EXTRA_ARGS="--local-poly-degree 7 --local-poly-em-iterations 1 --local-poly-em-quadrature 5 --local-poly-basis monomial --local-poly-pol-reg-type deriv1 --local-poly-pol-reg-eta 0.3 --local-poly-pol-reg-power 2"
    ;;
  12)
    TAG="basis_hard_d7_i1_q7_monomial_deriv1_eta0p3"
    EXTRA_ARGS="--local-poly-degree 7 --local-poly-em-iterations 1 --local-poly-em-quadrature 7 --local-poly-basis monomial --local-poly-pol-reg-type deriv1 --local-poly-pol-reg-eta 0.3 --local-poly-pol-reg-power 2"
    ;;
  13)
    TAG="basis_hard_d5_i1_q7_monomial_deriv2_eta30"
    EXTRA_ARGS="--local-poly-degree 5 --local-poly-em-iterations 1 --local-poly-em-quadrature 7 --local-poly-basis monomial --local-poly-pol-reg-type deriv2 --local-poly-pol-reg-eta 30 --local-poly-pol-reg-power 2"
    ;;
  14)
    TAG="basis_hard_d5_i2_q5_monomial_deriv2_eta30"
    EXTRA_ARGS="--local-poly-degree 5 --local-poly-em-iterations 2 --local-poly-em-quadrature 5 --local-poly-basis monomial --local-poly-pol-reg-type deriv2 --local-poly-pol-reg-eta 30 --local-poly-pol-reg-power 2"
    ;;
  15)
    TAG="basis_hard_d5_i2_q7_monomial_deriv2_eta30"
    EXTRA_ARGS="--local-poly-degree 5 --local-poly-em-iterations 2 --local-poly-em-quadrature 7 --local-poly-basis monomial --local-poly-pol-reg-type deriv2 --local-poly-pol-reg-eta 30 --local-poly-pol-reg-power 2"
    ;;
  16)
    TAG="basis_hard_d7_i1_q5_monomial_deriv2_eta30"
    EXTRA_ARGS="--local-poly-degree 7 --local-poly-em-iterations 1 --local-poly-em-quadrature 5 --local-poly-basis monomial --local-poly-pol-reg-type deriv2 --local-poly-pol-reg-eta 30 --local-poly-pol-reg-power 2"
    ;;
  17)
    TAG="basis_hard_d7_i1_q7_monomial_deriv2_eta30"
    EXTRA_ARGS="--local-poly-degree 7 --local-poly-em-iterations 1 --local-poly-em-quadrature 7 --local-poly-basis monomial --local-poly-pol-reg-type deriv2 --local-poly-pol-reg-eta 30 --local-poly-pol-reg-power 2"
    ;;
  *)
    echo "Unknown array task $SLURM_ARRAY_TASK_ID" >&2
    exit 2
    ;;
esac

LOG="$RUN_DIR/run_hardcase_${TAG}.log"

{
  date -Is
  "$PYTHON" - <<'PY'
import pathlib, recovar, jax
repo = pathlib.Path.cwd().resolve()
print("recovar.__file__=", recovar.__file__)
print("jax.__file__=", jax.__file__)
print("jax.devices=", jax.devices())
assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo) + "/")
assert ".pixi/envs/default/" in str(pathlib.Path(jax.__file__).resolve())
PY
  echo "SWEEP_TAG=$TAG"
  echo "EXTRA_ARGS=$EXTRA_ARGS"
  /usr/bin/time -v "$PYTHON" _agent_scratch/run_spike128_raw_index_gtnoise.py \
    --run-dir "$RUN_DIR" \
    --reference-raw-dir /scratch/gpfs/GILLES/mg6942/runs/spike_grid128_box320_noise30_n300k_raw_index_gtnoisy_20260513/01_raw_volumes \
    --reference-gt-metadata /scratch/gpfs/GILLES/mg6942/runs/spike_grid128_box320_noise30_n300k_raw_index_gtnoisy_20260513/index_noisy_embedding_metadata.json \
    --grid-size 128 \
    --voxel-size 2.5 \
    --n-states 100 \
    --n-images 200000 \
    --noise-level 0.1 \
    --noise-model radial1 \
    --render-bfactor 100 \
    --target-state 50 \
    --embedding-seed 20260513 \
    --embedding-noise-std 100 \
    --stream-simulation \
    --compute-state-lazy \
    --crafted-mask /scratch/gpfs/GILLES/mg6942/uploads/mask_crafted.mrc \
    --crafted-mask-level 0.0126 \
    --crafted-mask-cosine-width 3 \
    --local-poly-output-tag "$TAG" \
    --local-poly-h-state-grid 1,2,4,8,16,32 \
    --local-poly-em \
    --skip-gt-polynomial-sweep \
    $EXTRA_ARGS
  date -Is
} > "$LOG" 2>&1
