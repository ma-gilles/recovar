#!/bin/bash
#SBATCH --job-name=bench-5k-1iter
#SBATCH --account=amits
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/bench-5k-1iter-%j.out

# Fast head-to-head: 5k images × 1 iter, RELION GUI defaults vs recovar.
# RELION reference run already exists at $DATA_DIR/relion_ref_os0 with
# measured iter walls in run_itNNN_optimiser.star mtimes — we just measure
# recovar for the same data + 1 iter and print the comparison.

set -eo pipefail

DATA_DIR="${DATA_DIR:-/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized}"
RELION_REF="${RELION_REF:-$DATA_DIR/relion_ref_os0}"
OUT_DIR="${OUT_DIR:-/scratch/gpfs/GILLES/mg6942/_agent_scratch/bench_5k_2026_05_02/run_${SLURM_JOB_ID}}"

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_DIR"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="/scratch/gpfs/GILLES/mg6942/tmp/bench_5k_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$OUT_DIR"

pixi run python -c "
import pathlib, recovar, jax
print('jax devices:', jax.devices())
print('recovar:', pathlib.Path(recovar.__file__).parent)
"

echo "================================================================"
echo "Recovar refine_single_volume — 1 iter @ HEALPix order 2 (RELION-default)"
echo "================================================================"
T_START_REC=$(date +%s.%N)
pixi run python scripts/run_full_refinement.py \
  --data_dir "$DATA_DIR" \
  --output "$OUT_DIR/recovar_1iter" \
  --max_iter 1 \
  --healpix_order 3 \
  --adaptive_oversampling 1 \
  --offset_range 5.0 \
  --offset_step 1.0 \
  --init_resolution 30 \
  --image_batch_size 64 \
  --rotation_block_size 1000 \
  --tau2_fudge 4.0 \
  --perturb_factor 0.5 \
  --perturb_seed 42
T_END_REC=$(date +%s.%N)
RECOVAR_WALL=$(awk "BEGIN {print $T_END_REC - $T_START_REC}")

echo "================================================================"
echo "Comparing vs existing RELION 1-iter reference"
echo "================================================================"
pixi run python - <<PY
import json, pathlib
from datetime import datetime
import numpy as np

relion_dir = pathlib.Path("$RELION_REF")
out_dir    = pathlib.Path("$OUT_DIR")

# RELION per-iter wall: mtime gap between iter_000 and iter_001 optimisers.
def mtime(p): return p.stat().st_mtime
opt0 = relion_dir / "run_it000_optimiser.star"
opt1 = relion_dir / "run_it001_optimiser.star"
relion_iter1_wall = mtime(opt1) - mtime(opt0) if opt0.exists() and opt1.exists() else float("nan")

# Recovar wall = total subprocess runtime measured by the slurm shell.
recovar_total = float("$RECOVAR_WALL")
# Per-iter from results.npz if available.
recovar_results = list(out_dir.glob("**/results.npz"))
recovar_iter_walls = []
if recovar_results:
    npz = np.load(recovar_results[0], allow_pickle=True)
    walls = npz.get("wall_times", None)
    if walls is not None:
        recovar_iter_walls = [float(w) for w in np.asarray(walls)]

summary = {
    "dataset": str("$DATA_DIR"),
    "n_images_target": 5000,
    "image_size": 128,
    "iter_1_wall_seconds": {
        "RELION_GUI_default":      relion_iter1_wall,
        "recovar_refine_single":   recovar_iter_walls[0] if recovar_iter_walls else float("nan"),
        "recovar_total_subprocess": recovar_total,
    },
    "ratio_recovar_over_relion": (
        (recovar_iter_walls[0] / relion_iter1_wall) if recovar_iter_walls and relion_iter1_wall > 0 else None
    ),
    "generated_at": datetime.now().isoformat(),
}
out_path = out_dir / "perf_5k_1iter.json"
out_path.write_text(json.dumps(summary, indent=2))

# Pretty table.
print()
print("=" * 66)
print(f"{'5k images × 1 iter':<40} {'wall (s)':>14} {'rel.':>8}")
print("-" * 66)
print(f"{'RELION 3D auto-refine (GUI default)':<40} {relion_iter1_wall:>14.2f} {1.00:>8.2f}x")
if recovar_iter_walls:
    rel = recovar_iter_walls[0] / relion_iter1_wall if relion_iter1_wall > 0 else float('nan')
    print(f"{'recovar refine_single_volume':<40} {recovar_iter_walls[0]:>14.2f} {rel:>8.2f}x")
print(f"{'  (recovar full subprocess incl. setup)':<40} {recovar_total:>14.2f}")
print("=" * 66)
print(f"\\nWrote {out_path}")
PY

echo "=== DONE ==="
echo "Results: $OUT_DIR"
