#!/bin/bash
#SBATCH --job-name=autorefine-k1
#SBATCH --account=amits
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/autorefine-k1-%j.out
#SBATCH --exclusive

# k=1 full RELION-style auto-refine quality test on the prepared 100k_512
# single-class dataset. Drives recovar's refine_single_volume() through the
# full HEALPix-order ramp + adaptive oversampling + per-iter resolution
# advance, then we score the final half-maps against the GT volume.
#
# Defaults: 256³ working grid (downsampled from 512), 30 iters, adaptive
# oversampling on. To run at native 512 set DOWNSAMPLE_D="" before sbatch.

set -euo pipefail

DATA_DIR="${DATA_DIR:-/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_100k_512}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/gpfs/GILLES/mg6942/_agent_scratch/autorefine_k1_2026_05_02}"
MAX_ITER="${MAX_ITER:-30}"
HEALPIX_ORDER="${HEALPIX_ORDER:-4}"   # auto-refine ramps from order 1 up to this
ADAPTIVE_OVERSAMPLING="${ADAPTIVE_OVERSAMPLING:-1}"
ADAPTIVE_FRACTION="${ADAPTIVE_FRACTION:-0.999}"
DOWNSAMPLE_D="${DOWNSAMPLE_D:-256}"
INIT_RESOLUTION="${INIT_RESOLUTION:-25}"
IMAGE_BATCH_SIZE="${IMAGE_BATCH_SIZE:-32}"
ROTATION_BLOCK_SIZE="${ROTATION_BLOCK_SIZE:-500}"
TAU2_FUDGE="${TAU2_FUDGE:-4.0}"
PERTURB_FACTOR="${PERTURB_FACTOR:-0.5}"
PERTURB_SEED="${PERTURB_SEED:-42}"

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_DIR"

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="/scratch/gpfs/GILLES/mg6942/tmp/autorefine_k1_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$OUTPUT_ROOT" /scratch/gpfs/GILLES/mg6942/slurmo

# Provenance gate.
pixi run python -c "
import pathlib, recovar, jax
repo = pathlib.Path.cwd().resolve()
assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo)+'/'), 'WRONG recovar'
assert '.pixi/envs/default/' in str(pathlib.Path(jax.__file__).resolve()), 'WRONG jax'
print('ENV_OK', jax.devices())
"

OUTPUT_DIR="$OUTPUT_ROOT/run_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

DS_FLAG=""
if [ -n "$DOWNSAMPLE_D" ]; then
  DS_FLAG="--downsample_D $DOWNSAMPLE_D"
fi

pixi run python scripts/run_full_refinement.py \
  --data_dir "$DATA_DIR" \
  --output "$OUTPUT_DIR" \
  --max_iter "$MAX_ITER" \
  --healpix_order "$HEALPIX_ORDER" \
  --adaptive_oversampling "$ADAPTIVE_OVERSAMPLING" \
  --adaptive_fraction "$ADAPTIVE_FRACTION" \
  --init_resolution "$INIT_RESOLUTION" \
  --image_batch_size "$IMAGE_BATCH_SIZE" \
  --rotation_block_size "$ROTATION_BLOCK_SIZE" \
  --tau2_fudge "$TAU2_FUDGE" \
  --perturb_factor "$PERTURB_FACTOR" \
  --perturb_seed "$PERTURB_SEED" \
  $DS_FLAG

echo "=== Scoring vs GT ==="
pixi run python - <<PY
import json, sys
from pathlib import Path
import mrcfile, numpy as np
from recovar.reconstruction.regularization import get_fsc

out = Path("$OUTPUT_DIR")
data = Path("$DATA_DIR")
gt_path = data / "reference_gt.mrc"
init_path = data / "reference_init.mrc"
res_npz = list(out.glob("results.npz"))
if not res_npz:
    print("no results.npz", file=sys.stderr)
    sys.exit(1)
res = np.load(res_npz[0], allow_pickle=True)

# Load GT + downsample to match working grid.
with mrcfile.open(str(gt_path), permissive=True) as mrc:
    gt = mrc.data.copy().astype(np.float32)
    gt_voxel = float(mrc.voxel_size.x)
print("GT shape:", gt.shape, "voxel:", gt_voxel)

# Find final mean volume from refinement.
final_mrc = list(out.glob("iter_*_mean.mrc"))
final_mrc = sorted(final_mrc)[-1] if final_mrc else None
if final_mrc is None:
    print("no iter_*_mean.mrc; checking results.npz keys:", list(res.keys()))
    sys.exit(0)
with mrcfile.open(str(final_mrc), permissive=True) as mrc:
    pred = mrc.data.copy().astype(np.float32)
    pred_voxel = float(mrc.voxel_size.x)
print("pred shape:", pred.shape, "voxel:", pred_voxel)

# Crude downsample GT to match if needed.
if gt.shape[0] != pred.shape[0]:
    ratio = gt.shape[0] // pred.shape[0]
    if ratio > 1 and gt.shape[0] == pred.shape[0] * ratio:
        gt_ds = gt.reshape(pred.shape[0], ratio, pred.shape[0], ratio, pred.shape[0], ratio).mean(axis=(1,3,5))
    else:
        print(f"shape mismatch: GT {gt.shape} vs pred {pred.shape} — skipping FSC")
        sys.exit(0)
else:
    gt_ds = gt

vol_shape = pred.shape
fsc = np.asarray(get_fsc(pred.reshape(-1), gt_ds.reshape(-1), vol_shape))
fsc_area = float(np.mean(fsc))
fsc_at_05 = int(np.argmax(fsc < 0.5)) if (fsc < 0.5).any() else len(fsc)
fsc_at_0143 = int(np.argmax(fsc < 0.143)) if (fsc < 0.143).any() else len(fsc)
print(f"FSC@0.5  = shell {fsc_at_05}  ({pred_voxel*pred.shape[0]/max(fsc_at_05,1):.2f} A)")
print(f"FSC@0.143 = shell {fsc_at_0143}  ({pred_voxel*pred.shape[0]/max(fsc_at_0143,1):.2f} A)")
print(f"FSC area mean = {fsc_area:.4f}")

summary = {
    "fsc_area_mean": fsc_area,
    "fsc_at_05_shell": fsc_at_05,
    "fsc_at_0143_shell": fsc_at_0143,
    "voxel_size": pred_voxel,
    "grid_size": pred.shape[0],
    "fsc_at_05_angstrom": pred_voxel * pred.shape[0] / max(fsc_at_05, 1),
    "fsc_at_0143_angstrom": pred_voxel * pred.shape[0] / max(fsc_at_0143, 1),
}
with (out / "fsc_vs_gt.json").open("w") as fh:
    json.dump(summary, fh, indent=2)
print(f"wrote {out/'fsc_vs_gt.json'}")
PY

echo "=== Comparison vs RELION auto-refine ==="
RELION_REF_DIR="${RELION_REF_DIR:-$DATA_DIR/relion_ref_full_autorefine}"
GT_MRC="${GT_MRC:-$DATA_DIR/reference_gt.mrc}"
if [ -d "$RELION_REF_DIR" ] && [ -f "$GT_MRC" ]; then
  pixi run python scripts/compare_recovar_relion_autorefine.py \
    --mode k1 \
    --recovar-dir "$OUTPUT_DIR" \
    --relion-dir "$RELION_REF_DIR" \
    --gt-mrc "$GT_MRC" \
    --output "$OUTPUT_DIR/comparison.json"
else
  echo "Skipping comparison: RELION ref dir or GT mrc missing"
  echo "  RELION_REF_DIR=$RELION_REF_DIR"
  echo "  GT_MRC=$GT_MRC"
fi

echo "=== DONE ==="
echo "Results: $OUTPUT_DIR"
