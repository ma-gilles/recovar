# EM Module Developer Guide

## Architecture

```
em/
├── core.py                  # Cross-correlation, dot products, probability utils
├── e_step.py                # E_with_precompute (full E-step with batching)
├── m_step.py                # M_with_precompute, sum_up_images_fixed_rots_eqx
├── iterations.py            # E_M_batches_2 orchestrator, split_E_M_v2
├── states.py                # EMState, SGDState, HeterogeneousEMState
├── sampling.py              # HEALPix rotation grids, translation grids
├── heterogeneity.py         # Low-rank heterogeneity EM (H/B matrices, PCA)
└── dense_single_volume/     # Clean extraction of the dense homogeneous path
    ├── types.py             # DensePoseGrid, DenseEMPlan, MeanStats
    ├── plan.py              # Centralized memory planner
    ├── projection_cache.py  # Forward slice precomputation
    ├── posterior.py          # E-step wrapper
    ├── accumulate.py         # M-step wrapper
    ├── solver.py             # RELION-style Wiener solve wrapper
    ├── engine.py             # Original orchestrator (wraps old code path)
    ├── engine_fused.py       # Fused E+M engine (2.5× faster)
    └── engine_v2.py          # Two-pass JIT engine with blockwise normalization
```

## Key Computations

### E-step cross-term (the expensive GEMM)

```
cross[i,r,t] = -2 Re <S_t(CTF·y_i/σ²), P_r μ>
```

Implemented as: create n_trans shifted copies of each image, flatten to
`(n_img × n_trans, N)`, one GEMM against `(N, n_rot)` projections.
Code: `core.py:82` (`compute_dot_products_eqx`).

The n_trans factor inflates the GEMM but enables 200× better data reuse vs
FFT-based cross-correlation. See `docs/math/translation_handling_analysis.md`.

### M-step accumulation

```
Ft_y += Σ_{i,t} γ_{i,r,t} · P_r*(S_t* CTF·y_i/σ²)
```

The sum over images and translations is done by one GEMM BEFORE backprojection:
`P @ shifted_images → (n_rot, N)`, then adjoint_slice_volume.
Code: `m_step.py:117` (`sum_up_images_fixed_rots_eqx`).

### Translation handling

Two methods exist (see `docs/math/translation_handling_analysis.md`):
- **GEMM** (default): explicit phase-shifted copies + matmul. Best for batched rotations.
- **FFT**: `iFFT(conj(img) · proj)` cross-correlation. Best for single-rotation refinement.

GEMM wins by 33× for the dense grid because it reads input data once for all rotations.
FFT wins by 2× per single rotation but cannot batch across rotations efficiently.

## Performance Status (as of 2026-03-31)

Benchmarked on A100-80GB, 5000 images, 128px, order 3 (36,864 rotations), 7×7 translations:

| Engine | Time | vs old |
|---|---|---|
| Old (E_with_precompute + M_with_precompute) | 68s | 1× |
| engine_fused.py | 26s | 2.6× |
| engine_v2.py | 29s | 2.3× |
| Half-spectrum GEMMs (benchmarked, not integrated) | 19s | 3.6× |

RELION 5.0.1 on same hardware/data: ~163s per iteration (includes CPU M-step + overhead).

### Known optimization opportunities (in priority order)

1. **Half-spectrum GEMMs**: operate on N_half=8320 instead of N=16384. Demonstrated 1.7× speedup. Not yet integrated into the engines.

2. **Fourier cropping to current resolution**: RELION uses `current_size` to crop images to the current FSC resolution. At early iterations this is 50×+ fewer pixels. This is the single biggest gap vs RELION.

3. **Two-pass adaptive oversampling**: coarse angular search → prune to significant weights → fine search. Reduces effective orientations per image from 36K to ~100-500.

4. **Significant weight pruning**: only top-K orientations per image get the expensive fine-resolution evaluation.

## Testing

- `tests/unit/test_dense_em_equivalence.py` — 12 numerical equivalence tests pinning all refactored functions
- `tests/unit/test_dense_em_plan.py` — 5 planner tests
- Run `pixi run test-fast` (2454 tests) before pushing
- Run `./scripts/run_tests_parallel.sh long-test` via Slurm before PR

## Rules

- NEVER widen test tolerances or modify baselines without explicit approval
- NEVER modify `heterogeneity.py` — it has separate owners
- EMState delegates to `dense_single_volume/` for the homogeneous path; SGDState and HeterogeneousEMState still call the old functions directly
- `split_E_M_v2` accesses `state.Ft_y` and `state.Ft_CTF` after `finish_up_M_step` — these attributes must be preserved
- All GPU work via Slurm for real jobs; login GPUs for quick benchmarks only
