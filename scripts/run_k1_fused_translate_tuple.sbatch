#!/bin/bash
#SBATCH --job-name=em-k1-fused-tuple
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --qos=della-cryoem
#SBATCH --constraint=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -euo pipefail

: "${RUN_ROOT:?set RUN_ROOT}"
: "${REPO:?set REPO}"
: "${CUDA_LIB:?set CUDA_LIB}"
: "${FINE_OPERAND:?set FINE_OPERAND}"
: "${FINE_SCORE:?set FINE_SCORE}"
: "${PASS2:?set PASS2}"

export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export TMPDIR="/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_fused_tuple_${SLURM_JOB_ID}/tmp"
export PIXI_HOME="/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_fused_tuple_${SLURM_JOB_ID}/pixi_home"
export RATTLER_CACHE_DIR="/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_fused_tuple_${SLURM_JOB_ID}/rattler_cache"
mkdir -p "$RUN_ROOT/provenance" "$RUN_ROOT/analysis" "$TMPDIR" "$PIXI_HOME" "$RATTLER_CACHE_DIR"
touch "$RUN_ROOT/SAFE_TO_DELETE"
touch "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_fused_tuple_${SLURM_JOB_ID}/SAFE_TO_DELETE"

cd "$REPO"
PYTHON="$REPO/.pixi/envs/default/bin/python"
test -x "$PYTHON"
test -f "$CUDA_LIB"
export RECOVAR_CUDA_LIB="$CUDA_LIB"

nvidia-smi > "$RUN_ROOT/provenance/nvidia_smi_${SLURM_JOB_ID}.txt"
sha256sum "$CUDA_LIB" "$FINE_OPERAND" "$FINE_SCORE" "$PASS2" \
  > "$RUN_ROOT/provenance/static_inputs_${SLURM_JOB_ID}.sha256"
git rev-parse HEAD > "$RUN_ROOT/provenance/git_head_${SLURM_JOB_ID}.txt"
git diff --binary > "$RUN_ROOT/provenance/git_diff_${SLURM_JOB_ID}.patch"
"$PYTHON" - <<'PY' > "$RUN_ROOT/provenance/imports_${SLURM_JOB_ID}.txt"
import jax
import recovar
print("recovar", recovar.__file__)
print("jax", jax.__file__)
print("devices", jax.devices())
PY

"$PYTHON" -m scripts.analyze_k1_fused_translate_tuple \
  --fine-operand "$FINE_OPERAND" \
  --fine-score "$FINE_SCORE" \
  --pass2 "$PASS2" \
  --recovar-global-rotation 161944 \
  --physical-image-size 128 \
  --output "$RUN_ROOT/analysis/FUSED_TRANSLATE_TUPLE.json" \
  | tee "$RUN_ROOT/analysis/FUSED_TRANSLATE_TUPLE.stdout"

touch "$RUN_ROOT/FUSED_TRANSLATE_TUPLE_SUCCESS_${SLURM_JOB_ID}"
