# EM Module Developer Guide

## Active Agent Instructions

This `CLAUDE.md` file is the canonical EM agent guide for both Claude and
Codex. `recovar/em/AGENTS.md` only points back here to avoid duplicated
instructions. Do not discard or replace the existing instructions below;
they remain active.

Current RELION-parity branch: `claude/relion-parity-local-search-fix`.

Goal: perfect quality parity with RELION and near RELION speed parity for
the full end-to-end dense single-volume EM iteration.

Method: aggressively compare RECOVAR and RELION source code, and add gated
dump hooks to both codebases when source reading is not decisive. The local
RELION checkout is `/scratch/gpfs/GILLES/mg6942/relion`; the patched build is
`/scratch/gpfs/GILLES/mg6942/relion/build_patched`.

Do not solve parity by parameter tuning. For parity fixes, identify the
RELION source behavior, metadata value, or dump-level mismatch first, then
encode the same behavior in RECOVAR with a targeted test.

Required dump coverage for deep parity work includes raw E-step scores,
posterior probabilities, every attempted pose in full-grid pass 1,
adaptive/oversampled pass 2, and local search, best poses after each pass,
angle/translation priors, support masks, noise accumulators, `Ft_y`,
`Ft_ctf`, maps, FSC, tau2, data-vs-prior, current size, and resolution
state.

Use targeted EM tests and targeted RELION replay Slurm jobs during
iteration. Do not run the full RECOVAR-wide test suite for normal EM parity
work unless explicitly requested or preparing a PR that requires it.

Keep algorithmic parity changes separate from performance changes. Batching,
caps, memory layout, and scheduling changes are performance-only until output
equivalence is proven against the old path.

Current measured baselines, hardware, Slurm job IDs, artifacts, and open
parity gaps are tracked in
`docs/math/relion_parity_current_status_2026_04_25.md`. Update that doc
whenever a new replay result, source-code finding, or dump comparison changes
the state of the investigation.

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

## RELION Reference Runs — REQUIRED FLAGS (READ BEFORE BENCHMARKING)

We lost a week of the 2026-04 RELION-parity benchmark to RELION flags
that default to off on the **command line** but are always set by the
**GUI**. NEVER trust `relion_refine --help` for "what should I pass";
the help shows C++ defaults (mostly wrong). The authoritative source is
`relion/src/pipeline_jobs.cpp::initialiseAutorefineJob()` (≈line 4126).

After auditing every `parser.checkOption(...)` call in
`relion/src/ml_optimiser.cpp::parseInitial()` against the GUI's
auto-refine job constructor, **7 flags need to be added** to any CLI
invocation of `relion_refine_mpi --auto_refine`:

### Critical (silently wrong reconstruction without these)

| Flag | CLI default | What breaks |
|---|---|---|
| `--ctf` | **OFF** | RELION reconstructs the CTF-convolved volume. Dark halo / ringing in real space, resolution plateaus ~18-22 A regardless of particle count, radial power spectrum has CTF oscillations + excess high-freq power. |
| `--firstiter_cc` | **OFF** | When init is non-RELION (recovar / cryosparc / EMD), iter-1 Bayesian E-step uses the wrong intensity scale. Pose search collapses to a 2D-extruded "stripes" basin. |

### Quality (degraded convergence / artifacts without these)

| Flag | CLI default | What it does |
|---|---|---|
| `--flatten_solvent` | OFF | Masks reconstructed reference outside the particle to zero. GUI: "Always flatten the solvent" (`pipeline_jobs.cpp:4461`). Without it, references contain noise outside the particle that pollutes projections. |
| `--zero_mask` | OFF | Masks particle exterior to **zero** instead of random noise. GUI default true for SPA. Random-noise fill (CLI default) introduces correlated Fourier components into alignment. |
| `--low_resol_join_halves 40` | -1 (off) | Below 40 Å, the two random half-reconstructions share orientation statistics instead of being independent. GUI: "Always join low-res data" (`pipeline_jobs.cpp:4509`). Prevents h1/h2 divergence at low SNR. |

### No-op for single-optics-group, but set for parity

| Flag | CLI default | What it does |
|---|---|---|
| `--norm` | OFF | Per-optics-group normalisation correction. NO-OP if all particles share an optics group. Important for any real-data benchmark with multiple sessions. GUI: hardcoded (`pipeline_jobs.cpp:4510`). |
| `--scale` | OFF | Per-optics-group intensity scale correction. Same NO-OP / multi-optics caveat as `--norm`. GUI: hardcoded same line. |

**Diagnostic before any "RELION volume looks wrong" investigation:**
```bash
grep -E "_rlnDoCorrectCtf|_rlnRefsAreCtfCorrected|_rlnDoNormCorrection|_rlnDoScaleCorrection" \
     <relion_run_dir>/run_it000_optimiser.star
# _rlnDoCorrectCtf      0   ←  YOU FORGOT --ctf  (silently wrong reconstruction)
# _rlnDoCorrectCtf      1   ←  ok
```

**Canonical RELION-parity invocation** (also in
`scripts/run_relion_parity_benchmark_slurm.sh`):
```bash
mpirun -n 3 relion_refine_mpi \
  --i particles.star \
  --ref reference_init_relion.mrc \   # RELION-frame init via write_relion_mrc
  --o run \
  --auto_refine --split_random_halves \
  --particle_diameter 200 --ini_high 30 \
  --ctf \                              # required (default off!)
  --firstiter_cc \                     # required for non-RELION init
  --flatten_solvent \                  # GUI always sets this
  --zero_mask \                        # GUI default true
  --low_resol_join_halves 40 \         # GUI always sets this
  --norm --scale \                     # GUI always sets these
  --healpix_order 3 --offset_range 3 --offset_step 1 \
  --oversampling 1 --pad 2 --gpu 0 --j 4
```

**To verify the CTF-trap diagnosis** (debugging only — does NOT change
recovar's production code path), disable CTF in recovar with
`scripts/debug_recovar_no_ctf.py` and confirm it produces the same dark
halo + power-spectrum signature as RELION-no-ctf. The script
`scripts/diagnose_relion_no_ctf_quantitative.py` makes a falsifiable
prediction: `|F[vol_relion]|^2 / |F[vol_GT]|^2 ≈ <CTF^2(k)>` if RELION
ran without `--ctf`. Both curves overlap shell-by-shell when the
diagnosis is correct.

## RELION's iter-1 ave_Pmax = 1.0 is a binarization artifact, NOT inference

When diffing recovar vs RELION per-iter, **ignore the iter-1 Pmax gap**.
At iter 1 with `--firstiter_cc` (or `--always_cc`), RELION executes a
literal winner-take-all binarization (`ml_optimiser.cpp:7775-7803`):

```cpp
if ((iter == 1 && do_firstiter_cc) || do_always_cc) {
    // Binarize the squared differences array to skip marginalisation
    // Find best CC, set its weight to 1.0, all others to 0.0
    ...
    DIRECT_A1D_ELEM(exp_Mweight, myminidx) = 1.;
}
```

So `ave_Pmax = 1.0` is by construction at iter 1, not because RELION's
inference is "sharper". The CC scoring (line 7414) is also scale-invariant,
specifically to absorb intensity-scale mismatch from non-RELION init
volumes. **Do not add a hard-CC iter-1 path to recovar's `_run_relion_iteration_loop`**
to match this number — it's RELION's hack, not its model.

The compounding effect on iter 2+ via the iter-1 volume IS real, though:
RELION's iter-1 hard-assigned reconstruction is tighter than recovar's
soft-Bayesian one, so iter 2's Bayesian E-step starts from a sharper
volume and gets sharper posteriors. This persists for ~6 iters before
both pipelines converge to similar Pmax.

See `~/.claude/projects/-home-mg6942/memory/feedback_relion_iter1_hard_cc_is_not_parity_bug.md`
for the full forensic write-up.

See `~/.claude/projects/-home-mg6942/memory/feedback_relion_required_flags.md`,
`feedback_relion_ctf_required.md`, and
`feedback_relion_firstiter_cc_required.md` for the full forensic
write-ups.

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
All new work targets `dense_single_volume/em_engine.py`. Do not modify the legacy
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
├── noise.py                 # RELION-parity noise estimation
├── regularization.py        # tau2 prior, FSC, Wiener regularization
├── heterogeneity.py         # Low-rank heterogeneity EM (H/B matrices, PCA)
└── dense_single_volume/     # Dense homogeneous refinement (RELION-parity)
    ├── iteration_loop.py            # Core loop: refine_single_volume, _run_relion_iteration_loop
    ├── em_engine.py         # Two-pass JIT engine: E-step scoring + M-step accumulation
    └── helpers/  # Supporting modules (black-box from algorithm perspective)
        ├── types.py         # MeanStats, RelionStats, NoiseStats, EMProfileStats
        ├── convergence.py   # Angular/translational convergence detection
        ├── oversampling.py      # Two-pass adaptive oversampling (significance pruning)
        ├── fourier_window.py# Fourier cropping to current resolution
        ├── local_search.py  # Local search helper functions
        ├── orientation_priors.py # Prior construction for RELION mode
        ├── resolution.py   # Initialization / coarse-size helpers
        └── significance.py  # Batched significance computation
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
| em_engine.py | 29s | 2.3× |
| Half-spectrum GEMMs (benchmarked, not integrated) | 19s | 3.6× |

RELION 5.0.1 on same hardware/data: ~163s per iteration (includes CPU M-step + overhead).

### Known optimization opportunities (in priority order)

1. **Half-spectrum GEMMs**: operate on N_half=8320 instead of N=16384. Demonstrated 1.7× speedup. Not yet integrated into the engines.

2. **Fourier cropping to current resolution**: RELION uses `current_size` to crop images to the current FSC resolution. At early iterations this is 50×+ fewer pixels. This is the single biggest gap vs RELION.

3. **Two-pass adaptive oversampling**: coarse angular search → prune to significant weights → fine search. Reduces effective orientations per image from 36K to ~100-500.

4. **Significant weight pruning**: only top-K orientations per image get the expensive fine-resolution evaluation.

### Known correctness debt (post-parity)

1. **Half-spectrum Hermitian weights (TODO RELION-parity-debt)**: RELION sums
   over the rfft half-image with weight=1 for ALL pixels. The mathematically
   correct approach uses weight=2 for interior frequencies (which have a
   conjugate partner) and weight=1 for DC/Nyquist (self-conjugate). RELION's
   approach computes ~half the true Gaussian log-likelihood, making posteriors
   softer than the true Bayesian posterior. The MAP orientation is unchanged
   (same ranking). We match RELION for parity; switching to correct weights
   (`make_half_image_weights`) is a post-parity improvement that would sharpen
   posteriors and may improve convergence speed. Code location:
   `em_engine.py:run_em()` where `half_spectrum_scoring=True` sets
   `half_weights = ones(...)`.

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
