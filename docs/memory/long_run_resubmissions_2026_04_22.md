# Long-Run Resubmissions: 2026-04-22

## Purpose

Resubmit the long RECOVAR-vs-RELION quality runs that were not usable as
final evidence:

- 11-iteration 5k replay on the current speed-optimized branch state
- 20k benchmark at noise `1.0`
- 20k benchmark at noise `0.1`

This note is the single audit trail for the reruns: old failure mode,
new command/job metadata, and final outcome.

## Failure Modes Being Fixed

### 11-iteration 5k replay

- Old job: `7219478`
- Old log:
  `/scratch/gpfs/GILLES/mg6942/slurmo/parity-5k11-v1-7219478.out`
- Problem:
  timed out on the old local-search path and later iterations expanded
  into pathological multi-million-rotation unions, so it is not
  representative of the current branch state.

### 20k benchmark jobs

- Old job (`noise=1.0`): `7227985`
- Old log (`noise=1.0`):
  `/scratch/gpfs/GILLES/mg6942/slurmo/parity-20k-bench-7227985.out`
- Old job (`noise=0.1`): `7228047`
- Old log (`noise=0.1`):
  `/scratch/gpfs/GILLES/mg6942/slurmo/parity-20k-n0p1-7228047.out`
- Problem:
  `scripts/run_comparison.py` defaulted to `DATA_DIR/relion_ref`, while
  the benchmark harness writes RELION output under
  `DATA_DIR/relion_ref_benchmark`. The comparison phase therefore failed
  with a missing `run_it001_data.star` even though RELION output itself
  existed.

## Code Fix Used For Resubmission

- `scripts/run_comparison.py`
  - added `--relion_ref_dir`
  - autodetects `relion_ref_benchmark` when `relion_ref` is absent
- `scripts/run_relion_parity_benchmark_slurm.sh`
  - now passes `--relion_ref_dir "$RELION_REF_DIR"` explicitly
  - default `REPO_DIR` resolves from the script location so the current
    checkout is used by default

## Resubmissions

### 11-iteration 5k replay

- Git commit:
  `PENDING`
- Command / script:
  `PENDING`
- Job id:
  `PENDING`
- Output dir:
  `PENDING`
- Log:
  `PENDING`
- Status:
  `PENDING`
- Final metrics:
  - half-map FSC `0.143`: `PENDING`
  - merged-vs-GT FSC `0.5`: `PENDING`
  - RECOVAR-vs-RELION pose gap: `PENDING`
  - wall time: `PENDING`

### 20k benchmark, noise `1.0`

- Git commit:
  `PENDING`
- Command / script:
  `PENDING`
- Job id:
  `PENDING`
- Data dir:
  `PENDING`
- Log:
  `PENDING`
- Status:
  `PENDING`
- Final metrics:
  - half-map FSC `0.143`: `PENDING`
  - merged-vs-GT FSC `0.5`: `PENDING`
  - RECOVAR-vs-RELION pose gap: `PENDING`
  - wall time: `PENDING`

### 20k benchmark, noise `0.1`

- Git commit:
  `PENDING`
- Command / script:
  `PENDING`
- Job id:
  `PENDING`
- Data dir:
  `PENDING`
- Log:
  `PENDING`
- Status:
  `PENDING`
- Final metrics:
  - half-map FSC `0.143`: `PENDING`
  - merged-vs-GT FSC `0.5`: `PENDING`
  - RECOVAR-vs-RELION pose gap: `PENDING`
  - wall time: `PENDING`
