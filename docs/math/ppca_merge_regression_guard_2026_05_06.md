# PPCA Merge Regression Guard - 2026-05-06

This branch contains dense PPCA refinement foundation work that is likely to
conflict with adjacent EM, VDAM, and PPCA-refinement branches. During merges,
protect the following invariants before resuming feature work.

## Correctness Guards

Run these after each merge step:

```bash
pixi run python -m pytest \
  tests/unit/ppca_refinement/test_pose_score_grid_recovery.py \
  tests/unit/ppca_refinement/test_fused_dense_refinement.py \
  tests/unit/ppca_refinement/test_dense_dataset_iteration.py \
  -q
```

Then run the whole PPCA refinement unit group:

```bash
pixi run python -m pytest tests/unit/ppca_refinement -q
```

These tests pin:

- PPCA pose-marginal score and augmented moments against an independent NumPy
  reference, including pose-prior axis order.
- q=0 and W=0 homogeneous score behavior.
- exact HEALPix-grid rotation recovery.
- dense translation phase sign and indexing.
- dense dataset block construction through `iter_dense_ppca_dataset_blocks`.
- fused dense pass-2 backprojection against a slow pose-loop reference.
- rotation-block split invariance for dataset-backed dense PPCA EM.
- dense/local all-retained support equivalence.

## Performance Guard

Save a pre-merge baseline:

```bash
mkdir -p /scratch/gpfs/GILLES/mg6942/tmp/ppca_merge_guard
pixi run python scripts/benchmark_ppca_fused_block_guard.py \
  --output /scratch/gpfs/GILLES/mg6942/tmp/ppca_merge_guard/pre_merge_fused_block.json
```

After each merge, rerun:

```bash
pixi run python scripts/benchmark_ppca_fused_block_guard.py \
  --output /scratch/gpfs/GILLES/mg6942/tmp/ppca_merge_guard/post_merge_fused_block.json
```

Compare `median_elapsed_s`, `min_elapsed_s`, and `mean_elapsed_s`. Also compare
`rhs_abs_checksum`, `lhs_abs_checksum`, `pmax_mean`, and `logZ_mean`; those
should remain stable for the same JAX/backend configuration.

## Merge Stop Rules

Stop and debug before continuing if any of these change without an intentional
reason:

- Dense PPCA score/moment NumPy-reference test fails.
- Best rotation or translation recovery changes on HEALPix-grid tests.
- Fused dense block no longer matches the slow pose-loop reference.
- Dense PPCA results depend on `rotation_block_size`.
- Dense/local all-retained support equivalence fails.
- The fused-block benchmark slows materially on the same node/backend, or its
  checksums drift.

Do not restart native PPCA VDAM work until these guards pass on the merged
branch.
