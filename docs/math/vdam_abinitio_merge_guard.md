# VDAM / InitialModel Merge Guard

Use this guard before and after merging EM, VDAM, or PPCA branches that may
touch InitialModel, dense EM, or GT-evaluation code.

## CPU Gate

```bash
pixi run test-vdam-abinitio-merge-guard
```

This runs:

- compile checks for the ab-initio evaluator and merge-guard tests
- VDAM ab-initio contract tests
- the full `tests/unit/initial_model/` unit suite
- `scripts/run_em_fast_guard.sh`

The guard writes a JSON ledger under:

`/scratch/gpfs/GILLES/mg6942/_agent_scratch/vdam_abinitio_merge_guard_<timestamp>_<pid>/vdam_abinitio_merge_guard_summary.json`

The ledger records commit, branch, dirty status, recovar/JAX provenance, GPU
snapshot, command return codes, log paths, and walltime for each command. Keep
the pre-merge and post-merge ledgers and compare command status plus elapsed
time before accepting the merged result.

## GPU Gate

```bash
pixi run test-vdam-abinitio-merge-guard-gpu
```

This runs the CPU gate plus:

- `tests/integration/test_em_parity_fast.py`
- `scripts/extract_em_parity_tables.py --tier fast`

Use a local GPU only for short checks after `nvidia-smi` confirms availability.
Use Slurm if the queue or machine state makes the run non-interactive.

## Fast Planning / Smoke

To verify the command plan without running tests:

```bash
pixi run python scripts/run_vdam_abinitio_merge_guard.py --tier all --quick --dry-run
```

## Contracts Protected

The merge guard intentionally fails if any merge drops:

- `--gt_align`, `--gt_align_healpix_order`, or `--gt_align_max_shell` wiring in
  `scripts/run_multi_iter_parity.py`
- the same GT-alignment wiring in `scripts/postprocess_multi_iter_gt.py`
- standalone `scripts/evaluate_ab_initio_gt.py` support for RELION-frame
  InitialModel outputs
- default GT alignment order `2`, score shell `8`, or the order-2 rotation-grid
  count of `4608`
- raw and aligned GT metric separation

These checks are behavior-oriented, not ancestry-oriented, so they still work
after rebases or non-squash merges.
