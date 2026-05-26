# RELION instrumentation patches

## relion_ml_optimiser_debug_dump.patch

Adds `RECOVAR_DEBUG_DUMP_DIR`-gated dumps to
`/scratch/gpfs/GILLES/mg6942/relion/src/ml_optimiser.cpp` at two points:

- **Bootstrap reconstruct** (line 3264, `setSigmaNoiseEstimatesAndSetAverageImage`):
  dumps `wsum_model.BPref[iclass].data / .weight` and `mymodel.Iref[iclass]`
  before/after the reconstruct call. Also dumps `wsum_model.current_size`
  trace at the bootstrap entry point (line ~2946).

- **Iter-1 VDAM M-step** (line ~5234, `reconstructGrad` branch): dumps
  `Iref[iclass]` before + after `reconstructGrad`, the BPref accumulator
  `data / weight`, and a meta file with `grad_current_stepsize`,
  effective stepsize, `tau2_fudge_factor`, `min_resol_shell`,
  BackProjector `pad_size / r_max / skip_gridding`.

## How to apply

```bash
cd /scratch/gpfs/GILLES/mg6942/relion
git apply /scratch/gpfs/GILLES/mg6942/recovar_dev/recovar/docs/patches/relion_ml_optimiser_debug_dump.patch
cd build_patched && make -j16
```

## How to run

```bash
export RECOVAR_DEBUG_DUMP_DIR=/scratch/gpfs/GILLES/mg6942/_agent_scratch/relion_debug_dump
mkdir -p $RECOVAR_DEBUG_DUMP_DIR
/scratch/gpfs/GILLES/mg6942/relion/build_patched/bin/relion_refine \
    --o out/run --iter 1 --grad --denovo_3dref --i particles.star \
    [...exact fixture args...] --random_seed 1776701668
```

The dumps are consumed by
`tests/unit/initial_model/test_bootstrap_iref_fixture.py::test_bootstrap_iref_matches_fresh_relion_dump`
which achieves **CC = 0.999313** vs same-build RELION (machine-precision
parity modulo FFTW planner non-determinism).

## File format

Each `.bin` dump is:

```
int64 nz
int64 ny
int64 nx
<nz * ny * nx elements>   # complex128 (if 16 bytes/elem) or float64 (if 8)
```

Python reader: see `tests/unit/initial_model/test_bootstrap_iref_fixture.py::_read_binary_dump`.

## Why not vendor the patched RELION

Keeping the patch as a reviewable diff means:
1. Regular RELION releases can be tracked and the patch rebased.
2. We don't ship a fork — users apply the patch locally if they need to
   regenerate dumps (e.g. for a new fixture or different RELION version).
3. Keeps the diff minimal and easy to read: dump code is env-gated, off
   by default.
