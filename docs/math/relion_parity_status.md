# RELION-Parity EM: Status & Handoff (2026-04-02)

## Branch & Location
- Branch: `claude/em-relion-parity` on `ma-gilles/recovar`
- Worktree: `/scratch/gpfs/GILLES/mg6942/recovar_wt_merge/`
- Pushed to origin

## What exists

### New files
| File | Purpose |
|------|---------|
| `recovar/em/dense_single_volume/convergence.py` | RefinementState, convergence detection, angular refinement logic |
| `recovar/em/dense_single_volume/refine.py` (mode="relion") | Full RELION-parity refinement loop |
| `recovar/em/sampling.py` (+193 lines) | `get_rotation_grid_at_order`, `get_local_rotation_grid_fast` (HEALPix query_disc) |
| `recovar/reconstruction/regularization.py` (+138 lines) | `compute_data_vs_prior`, `resolution_from_data_vs_prior`, `compute_current_size_relion` |
| `recovar/reconstruction/relion_functions.py` | `zero_pad_fourier_volume` (BROKEN), `padding_factor` param added |
| `docs/math/plan_relion_parity_v2.md` | 1564-line implementation plan |
| `tests/unit/test_convergence.py` | 68 tests |
| `tests/unit/test_resolution_criterion.py` | 21 tests |
| `tests/unit/test_refine_relion_mode.py` | 7 smoke tests |

### Reference documents (READ THESE)
| Document | What it contains |
|----------|------------------|
| `/scratch/gpfs/GILLES/mg6942/relion5_auto_refine_algorithm.md` | 554-line exhaustive RELION 5 algorithm reference from C++ source |
| `/scratch/gpfs/GILLES/mg6942/tmp/em_relion_proj/audit_report.md` | 20-item gap analysis between our code and RELION |
| `docs/math/plan_relion_parity_v2.md` | Full implementation plan with phases C1-C9 and S1-S7 |
| `recovar/em/CLAUDE.md` | EM module developer guide |

## Best results achieved

| Configuration | Time | FSC 0.143 | AUC | Notes |
|---------------|------|-----------|-----|-------|
| **Capped local search (20K), no adaptive OS** | **773s** | **14.3 A** | **0.359** | Best speed/quality |
| No local search | ~900s | 11.8 A | 0.488 | Best quality (overfits?) |
| RELION | 1217s | 18.8 A | 0.332 | Reference |

All on 128px, 5000 images, proper half-set splits, RELION-projected data.

## What works
1. **Convergence-driven loop** — data_vs_prior resolution, angular refinement triggers
2. **Fourier windowing** — 95% reduction at early iters (cs=32), 30s vs 130s
3. **Local angular search** — fast HEALPix query_disc (0.01s), Gaussian prior weighting, 20K cap
4. **Half-set gold-standard FSC** — proper independent half-maps
5. **tau2/Wiener regularization** — formula matches RELION exactly
6. **Adaptive oversampling** — two-pass coarse/fine (pass 2 now executes)

## What's broken / missing (priority order)

### 1. `zero_pad_fourier_volume` is broken (HIGH)
- Location: `relion_functions.py`
- Symptom: CC drops from 0.97 to 0.25 with padding_factor=2
- Effect: Without padding, our FSC drops gradually (interpolation artifacts) instead of RELION's sharp cutoff
- Currently disabled: `PADDING_FACTOR = 1` in refine.py
- Fix: Debug the frequency mapping in the zero-padding function

### 2. No per-image local search (HIGH for speed at order 5+)
- RELION processes ~500 orientations per image individually
- We take the UNION of all images' local neighborhoods → grid grows to 20K-200K
- At order 5+, the full grid (2.4M rotations) OOMs; we cap at order 4
- Fix: Either process images one at a time (like RELION) or batch by similar best orientation

### 3. Adaptive oversampling is inefficient at early iterations (MEDIUM)
- At early iters, 86% of coarse orientations are significant → pass 2 evaluates nearly the full grid
- Makes it 2x slower than single-pass with no quality benefit
- Fix: Skip adaptive OS when significance fraction > 50%, or use a smaller coarse grid

### 4. Noise estimation still hard-assignment (MEDIUM)
- We use best-orientation residuals; RELION uses posterior-weighted
- Biases noise low → under-regularization → overfitting
- Our noise fix (all images both half-sets) helped but isn't enough
- Fix: Accumulate weighted residuals during M-step (Plan phase C3, Strategy A)

### 5. No masked alignment / unmasked reconstruction split (MEDIUM)
- RELION uses masked images for E-step, unmasked for M-step
- We use unmasked for both
- For particle_diameter = box_size (our test), effect is minimal
- For real data with smaller particles, this matters

### 6. FSC curve shape doesn't match RELION (the core issue)
- Our FSC vs GT drops gradually from shell 1
- RELION's FSC stays at 1.0 through shell 25 then drops sharply
- Root causes: no padding (#1) + different noise/regularization (#4)

## Test dataset
- RELION-projected images: `/scratch/gpfs/GILLES/mg6942/tmp/em_relion_proj/relion_proj.mrcs`
- RELION refine output: `/scratch/gpfs/GILLES/mg6942/tmp/em_relion_proj/relion_refine/`
- GT volume: `/scratch/gpfs/GILLES/mg6942/tmp/em_relion_proj/gt_vol.mrc`
- Poses/CTF: `/scratch/gpfs/GILLES/mg6942/tmp/em_128/poses.pkl`, `ctf.pkl`
- Sim info: `/scratch/gpfs/GILLES/mg6942/tmp/em_128/simulation_info.pkl`
- 5000 images, 128px, voxel_size=4.25 A, noise_std=0.54

## How to run
```bash
cd /scratch/gpfs/GILLES/mg6942/recovar_wt_merge
CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONNOUSERSITE=1 \
  pixi run python -u your_script.py
```

Example refine call:
```python
result = refine_single_volume(
    [half1, half2], init.reshape(-1), nv, sv, rotations, translations, 'linear_interp',
    max_iter=20, image_batch_size=500, rotation_block_size=5000,
    adaptive_oversampling=1, mode='relion',
    init_healpix_order=3, max_healpix_order=4,
)
```

## Running tests
```bash
pixi run python -m pytest tests/unit/test_convergence.py tests/unit/test_resolution_criterion.py \
  tests/unit/test_refine_relion_mode.py tests/unit/test_em_core.py tests/unit/test_em_states.py \
  tests/unit/test_fourier_window.py tests/unit/test_half_spectrum_em.py -v --tb=short
```
