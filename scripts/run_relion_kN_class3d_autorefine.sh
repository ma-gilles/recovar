#!/bin/bash
#SBATCH --job-name=relion-class3d-kN
#SBATCH --account=amits
#SBATCH --partition=cryoem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gpfs/GILLES/mg6942/slurmo/relion-class3d-kN-%j.out

# RELION Class3D multi-iter auto-refine reference for the k=N quality
# benchmark. Mirrors the auto-refine settings that produced the
# data_100k_512 single-class reference: HEALPix order ramp from 1 up,
# tau2_fudge=4, pad=2, particle_diameter=380A, MPI 3 ranks × 4 OMP.
#
# Output goes to $DATA_DIR/relion_class3d_k{K}_autorefine/ so the recovar
# k=N quality script can find it for side-by-side comparison.

set -euo pipefail

DATA_DIR="${DATA_DIR:-/scratch/gpfs/GILLES/mg6942/em_relion_proj/ribosembly_allk_g256_n100000_snr1_cubic}"
N_CLASSES="${N_CLASSES:-4}"
ITER="${ITER:-25}"
HEALPIX_ORDER="${HEALPIX_ORDER:-2}"
OFFSET_RANGE="${OFFSET_RANGE:-3}"
OFFSET_STEP="${OFFSET_STEP:-1}"
PARTICLE_DIAMETER="${PARTICLE_DIAMETER:-380}"
TAU2_FUDGE="${TAU2_FUDGE:-4}"
RELION_REFINE_MPI="${RELION_REFINE_MPI:-/scratch/gpfs/GILLES/mg6942/relion/build_patched/bin/relion_refine_mpi}"
RELION_MODULE="${RELION_MODULE:-relion/5.0.1/gcc-11.5.0-gpu}"

OUT_DIR="$DATA_DIR/relion_class3d_k${N_CLASSES}_autorefine"
mkdir -p "$OUT_DIR"

cd "$DATA_DIR"

# RELION needs an init reference STAR with K rows. The single-class
# Ribosembly dataset prep has K=16 references; we keep the first K.
INIT_STAR="$DATA_DIR/reference_init_classes_relion.star"
if [ ! -f "$INIT_STAR" ]; then
  echo "ERROR: $INIT_STAR not found; run the prepare script first." >&2
  exit 1
fi

# Trim to K classes if needed.
TRIMMED_INIT="$OUT_DIR/reference_init_first${N_CLASSES}.star"
python3 - <<PY
from pathlib import Path
src = Path("$INIT_STAR").read_text().splitlines()
out = []
in_loop = False
loop_count = 0
for line in src:
    if line.strip().startswith("loop_"):
        in_loop = True
        out.append(line)
        continue
    if in_loop and line.strip().startswith("_") :
        out.append(line)
        continue
    if in_loop and line.strip() and not line.strip().startswith("_"):
        if loop_count >= int("$N_CLASSES"):
            continue
        loop_count += 1
    out.append(line)
Path("$TRIMMED_INIT").write_text("\n".join(out))
print(f"wrote $TRIMMED_INIT with $N_CLASSES classes")
PY

module load "$RELION_MODULE"

mpirun -n 3 "$RELION_REFINE_MPI" \
  --i "$DATA_DIR/particles.star" \
  --ref "$TRIMMED_INIT" \
  --o "$OUT_DIR/run" \
  --iter "$ITER" \
  --tau2_fudge "$TAU2_FUDGE" \
  --particle_diameter "$PARTICLE_DIAMETER" \
  --K "$N_CLASSES" \
  --flatten_solvent \
  --zero_mask \
  --ctf \
  --norm \
  --scale \
  --sym C1 \
  --oversampling 1 \
  --healpix_order "$HEALPIX_ORDER" \
  --offset_range "$OFFSET_RANGE" \
  --offset_step "$OFFSET_STEP" \
  --pad 2 \
  --pool 3 \
  --dont_combine_weights_via_disc \
  --gpu 0:0:0 \
  --j 4 \
  2>&1 | tee "$OUT_DIR/run.log"

echo "=== DONE ==="
echo "Reference: $OUT_DIR"
