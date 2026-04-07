# EM Module Developer Guide

## RELION Volume Convention (READ THIS FIRST)

recovar and RELION use different 3D coordinate frames for real-space
volumes:
```python
vol_recovar = -np.transpose(vol_relion, (2, 1, 0))   # negate + swap X↔Z
```

**Pinned by tests/unit/test_relion_volume_convention.py — do NOT remove
these helpers without updating that test.**

### Canonical helpers (in `recovar/utils/helpers.py`)
- `load_mrc(path)` / `write_mrc(path, vol)` — for **recovar / cryosparc /
  cryoDRGN** MRCs. Round-trip safe.
- `load_relion_volume(path)` — load a **RELION** MRC and return it
  already in recovar's frame. **Use this when comparing RELION outputs
  against recovar outputs.**
- `relion_volume_to_recovar(vol)` / `recovar_volume_to_relion(vol)` —
  explicit frame conversion (the operation is its own inverse).
- `R_to_relion(R)` / `R_from_relion(euler)` — rotation Euler conversion.
  These are **CORRECT and intentionally paired with the volume
  transpose**. The negation in the volume convention cancels out at the
  projection step. Do NOT "fix" them. See issue #86.

### One-liner for FSC against a RELION reference
```python
from recovar.utils.helpers import load_relion_volume, load_mrc
relion_ref = load_relion_volume("relion_output/run_class001.mrc")
recovar_vol = load_mrc("recovar_output/final_merged.mrc")
# both are now in the same frame; FSC is meaningful
```

### History
The helpers were added in commit `7df73fa` (2026-04-01 11:00) and
removed in commit `4703c634` (2026-04-01 12:08, "revert helpers.py to
clean origin/dev state") **without updating this guide**. The result was
a year of intermittent confusion: every maintainer who tried to follow
this guide hit `ImportError` and gave up. Restored by commit on
2026-04-07 along with `tests/unit/test_relion_volume_convention.py` to
prevent the same drift.

## Active Development Plan

**Read `docs/math/plan_relion_parity.md` before making any changes to this module.**

The plan describes a 7-phase effort to bring this module to RELION feature parity.
All new work targets `dense_single_volume/engine_v2.py`. Do not modify the legacy
`core.py`/`m_step.py` path unless needed for shared utilities. Do not modify
`heterogeneity.py` (separate owners).

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
