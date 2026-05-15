#!/bin/bash
#SBATCH --job-name=sp128-basis-reg
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=300G
#SBATCH --time=12:00:00
#SBATCH --array=4-50%4
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/spike128_ctf_noise0p1_basis_reg_%A_%a.out
#SBATCH --error=/scratch/gpfs/GILLES/mg6942/slurmo/spike128_ctf_noise0p1_basis_reg_%A_%a.err

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

if [[ "$SLURM_ARRAY_TASK_ID" -eq 0 ]]; then
  TAG="basis_baseline_d5_i1_q5_mono_none"
  EXTRA_ARGS="--local-poly-degree 5 --local-poly-em-iterations 1 --local-poly-em-quadrature 5 --local-poly-basis monomial --local-poly-pol-reg-type none"
elif [[ "$SLURM_ARRAY_TASK_ID" -eq 1 ]]; then
  TAG="basis_baseline_d5_i1_q7_mono_none"
  EXTRA_ARGS="--local-poly-degree 5 --local-poly-em-iterations 1 --local-poly-em-quadrature 7 --local-poly-basis monomial --local-poly-pol-reg-type none"
elif [[ "$SLURM_ARRAY_TASK_ID" -eq 2 ]]; then
  TAG="basis_baseline_d5_i2_q5_mono_none"
  EXTRA_ARGS="--local-poly-degree 5 --local-poly-em-iterations 2 --local-poly-em-quadrature 5 --local-poly-basis monomial --local-poly-pol-reg-type none"
else
  BASES=(monomial weighted_cholesky)
  REGS=(coeff deriv1 deriv2)
  ETAS=(0 0.03 0.1 0.3 1 3 10 30)
  IDX=$((SLURM_ARRAY_TASK_ID - 3))
  ETA_IDX=$((IDX % 8))
  REG_IDX=$(((IDX / 8) % 3))
  BASIS_IDX=$((IDX / 24))
  BASIS=${BASES[$BASIS_IDX]}
  REG=${REGS[$REG_IDX]}
  ETA=${ETAS[$ETA_IDX]}
  ETA_TOKEN=${ETA//./p}
  TAG="basis_d5_i1_q5_${BASIS}_${REG}_eta${ETA_TOKEN}"
  EXTRA_ARGS="--local-poly-degree 5 --local-poly-em-iterations 1 --local-poly-em-quadrature 5 --local-poly-basis ${BASIS} --local-poly-pol-reg-type ${REG} --local-poly-pol-reg-eta ${ETA} --local-poly-pol-reg-power 2"
fi

LOG="$RUN_DIR/run_sweep_${TAG}.log"

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
} 2>&1
