#!/bin/bash
#SBATCH --job-name=parity-traj
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/parity-traj-%j.out
#SBATCH --error=/scratch/gpfs/GILLES/mg6942/slurmo/parity-traj-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#
# Template for a RELION-parity multi-iter run with per-iter perf monitoring.
#
# Copy this script and edit the WORKTREE / RELION_REF / DATA_STAR / OUT /
# DUMP_DIR / BASELINE / INIT_ITER / MAX_ITER variables before sbatching.
#
# What it does:
#   1. Sets up env (CUDA_VISIBLE_DEVICES=0, JAX cache, RECOVAR_PARITY_DUMP_DIR)
#   2. Launches scripts/run_multi_iter_parity.py via launch_with_perf_check.py
#   3. After EACH iter completes (i.e. each new iter_NNN.npz appears in DUMP_DIR),
#      runs scripts/parity/check_perf.py --single-iter N
#   4. Echoes the result to stdout (visible in the slurm log)
#
# By default REGRESSED iters only trigger a warning. To auto-cancel the job
# on regression, append --cancel-on-regression to the launcher invocation
# at the bottom of this script.

set -euo pipefail

# --- Edit these paths for your run ---
WORKTREE=/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_perf_baseline_${SLURM_JOB_ID:-manual}
RELION_REF=/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized/relion_ref_os0
DATA_STAR=/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized/particles.star
OUT=/scratch/gpfs/GILLES/mg6942/_agent_scratch/parity/recovar_${SLURM_JOB_ID:-manual}
DUMP_DIR="$OUT/parity_dump"
BASELINE="$WORKTREE/tests/baselines/parity/perf_baseline_5k_128_a100.json"
INIT_ITER=3
MAX_ITER=14

# --- Standard env setup ---
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
export TMPDIR="/scratch/gpfs/GILLES/mg6942/tmp/parity_${SLURM_JOB_ID:-manual}"
export JAX_COMPILATION_CACHE_DIR="/scratch/gpfs/GILLES/mg6942/jax_cache_a100"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export RECOVAR_PARITY_DUMP_DIR="$DUMP_DIR"
mkdir -p "$TMPDIR" "$DUMP_DIR" "$OUT" "$JAX_COMPILATION_CACHE_DIR"

cd "$WORKTREE"

# --- Sanity check env provenance ---
pixi run python -c "
import pathlib, recovar, jax
repo = pathlib.Path.cwd().resolve()
assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo)+'/'), f'WRONG recovar: {recovar.__file__}'
assert '.pixi/envs/default/' in str(pathlib.Path(jax.__file__).resolve()), f'WRONG jax: {jax.__file__}'
print('ENV_OK')
"

# --- Run with per-iter perf checking ---
INNER_CMD="pixi run python scripts/run_multi_iter_parity.py \
    --relion_dir $RELION_REF \
    --data_star $DATA_STAR \
    --iter $INIT_ITER --max_iter $MAX_ITER \
    --output_dir $OUT \
    --local_engine exact_v1"

pixi run python scripts/parity/launch_with_perf_check.py \
    --dump-dir "$DUMP_DIR" \
    --baseline "$BASELINE" \
    --poll-interval 30 \
    --cmd "$INNER_CMD"

# --- After completion, full-table check ---
echo "=== Final perf-check across all iters ==="
pixi run python scripts/parity/check_perf.py \
    --dump-dir "$DUMP_DIR" \
    --baseline "$BASELINE"
