#!/usr/bin/env bash
# RELION CPU dump for particle 478 (1-indexed, = `478@particles.64.mrcs`).
# Run this AFTER recovar's per-pose dump completes; outputs go to a fresh
# dir ready for element-wise comparison with recovar's per-pose scores.

set -eu

DUMP_DIR="/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_codex_em_phase01_sparse_pass2_rebase_20260424/_agent_scratch/perpose_relion_p478"
mkdir -p "$DUMP_DIR"

cd /scratch/gpfs/GILLES/mg6942/em_relion_proj/data_tiny_parity

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PATH=/usr/local/openmpi/4.1.6/gcc/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/openmpi/4.1.6/gcc/lib64:/usr/local/cuda/lib64:/scratch/gpfs/GILLES/mg6942/relion/external/fftw/lib

export RELION_DUMP_DIR="$DUMP_DIR"
export RELION_DUMP_STACK_INDEX=478   # 1-indexed in RELION

echo "[$(date)] starting RELION CPU dump for particle 478"

# CPU mode (no --gpu). Uses --auto_iter_max 1 to limit to one iter so the
# dump captures iter-1 state before iter-2 overwrites.
mpirun -n 3 \
    -x RELION_DUMP_DIR -x RELION_DUMP_STACK_INDEX -x LD_LIBRARY_PATH \
    /scratch/gpfs/GILLES/mg6942/relion/build_patched/bin/relion_refine_mpi \
    --i particles.star --ref reference_init_relion.mrc \
    --o "$DUMP_DIR/run" \
    --auto_refine --split_random_halves \
    --particle_diameter 200 --ini_high 30 \
    --ctf --flatten_solvent --zero_mask \
    --low_resol_join_halves 40 --norm --scale \
    --healpix_order 3 --offset_range 3.0 --offset_step 1.0 \
    --oversampling 0 --pad 2 \
    --random_seed 1775685454 --j 1 \
    --auto_iter_max 1 || true   # auto_refine doesn't always honor; killable

echo "[$(date)] dump done; contents:"
ls -la "$DUMP_DIR" | head
