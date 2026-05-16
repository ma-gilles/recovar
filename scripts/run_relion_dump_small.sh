#!/usr/bin/env bash
# Run patched RELION on the small InitialModel fixture and write coherent
# dumps for both the iter-1 E-step posterior and the iter-1 M-step BPref.
# Both env vars must point to the same directory so subsequent probes
# (scripts/probe_estep_coherent.py) compare apples to apples.
#
# Requires:
#   - RELION built with docs/patches/relion_estep_dump.patch applied
#     (binary at /scratch/gpfs/GILLES/mg6942/relion/build_patched/bin/relion_refine)
#   - The small InitialModel fixture at PARTICLES_DIR
set -euo pipefail

DUMP_DIR="${DUMP_DIR:-/scratch/gpfs/GILLES/mg6942/_agent_scratch/relion_estep_dump_small}"
RUN_DIR="${RUN_DIR:-/scratch/gpfs/GILLES/mg6942/_agent_scratch/relion_estep_run_small}"
PARTICLES_DIR="${PARTICLES_DIR:-/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar/.tmp/slurm_7178672/pytest-of-mg6942/pytest-0/test_pipeline_spa_gpu0/gpu_spa/test_dataset}"
RELION="${RELION:-/scratch/gpfs/GILLES/mg6942/relion/build_patched/bin/relion_refine}"
SEED="${SEED:-1234567}"

mkdir -p "$DUMP_DIR" "$RUN_DIR"
touch "$DUMP_DIR/SAFE_TO_DELETE" "$RUN_DIR/SAFE_TO_DELETE"
rm -f "$DUMP_DIR"/p* "$DUMP_DIR"/pipe_it1_* "$DUMP_DIR"/mstep_it1_* "$DUMP_DIR"/bpref_c0_* "$DUMP_DIR"/iref_c0_* "$DUMP_DIR"/rg_* "$DUMP_DIR"/bootstrap_* 2>/dev/null || true
rm -rf "$RUN_DIR"/run* 2>/dev/null || true

cd "$PARTICLES_DIR"
RECOVAR_DEBUG_ESTEP_DIR="$DUMP_DIR" RECOVAR_DEBUG_DUMP_DIR="$DUMP_DIR" "$RELION" \
    --o "$RUN_DIR/run" --iter 1 --grad --denovo_3dref \
    --i particles.star --ctf --K 1 --sym C1 \
    --flatten_solvent --zero_mask \
    --dont_combine_weights_via_disc --no_parallel_disc_io \
    --pool 1 --pad 1 --particle_diameter 544 \
    --oversampling 1 --healpix_order 1 --offset_range 6 --offset_step 2 \
    --auto_sampling --tau2_fudge 4 --j 4 --random_seed "$SEED" \
    > "$RUN_DIR/stdout.log" 2>&1

echo "RELION run complete (seed=$SEED)."
echo "Dumps:"
ls "$DUMP_DIR"/p0_*.bin 2>/dev/null | head -5
ls "$DUMP_DIR"/pipe_it1_*.bin 2>/dev/null | head -3
echo "Perturbation: $(grep random_perturbation "$DUMP_DIR/p0_perturbation.txt" 2>/dev/null || echo n/a)"
