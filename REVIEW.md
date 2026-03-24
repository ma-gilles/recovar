# Code Review Guidelines

## Must check

- **No baseline modifications**: Files under `tests/baselines/` must never be modified, regenerated, or overwritten. They are ground truth from the published recovar code.
- **No tolerance widening**: Changes to `_TOL`, `tol_frac`, `HIGH_VARIANCE_TOKENS`, or `ABSOLUTE_TOLERANCE_FLOORS` are not allowed without explicit approval.
- **Numerical stability**: Changes to covariance estimation, embedding, or eigendecomposition must be validated against regression baselines. Half-set cross-validation involves near-cancellation — small FP errors amplify ~1e6x.
- **CUDA coordinate conventions**: k0=row, k1=col in CUDA; coord[0]=col, coord[1]=row in JAX meshgrid. Any coordinate changes must pass `test_cuda_jax_equivalence.py`.
- **Volume normalization**: Eigenvectors must use `load_u_real_for_metrics()`, never load from MRC directly.
- **Test visibility**: Regression tests must print comparison tables via `logging` (not `print`), and save scores to JSON.

## Should check

- New pipeline code has corresponding test coverage (unit or integration).
- GPU memory usage doesn't regress (check batch sizing logic).
- `ForwardModelConfig` changes don't cause unnecessary JIT recompilation.
- New CLI flags are documented in the command's argparse help.

## Skip

- Formatting in files not touched by the PR (formatting converges gradually).
- Notebook cell outputs.
- Generated files under `data-*/` or `scripts/output/`.
