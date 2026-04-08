# PPCA ab-initio v0 — Plan & Spec

**Status:** draft, pre-implementation
**Owner:** ma-gilles / mg6942
**Branch:** `claude/ppca-abinitio-v0`, branched from `claude/em-relion-parity`
**Module root:** `recovar/em/ppca_abinitio/`
**Companion docs:** `docs/math/plan_relion_parity.md`, `docs/math/plan_ab_initio_relion_parity.md`, `recovar/em/CLAUDE.md`

This document is the authoritative spec for the **direct PPCA ab-initio**
experiments. It is the contract that the implementation must satisfy.
Subsequent code, tests, and PR descriptions should reference section
numbers from here. Do not silently deviate; if something below is wrong,
update this doc in the same PR that changes behavior.

---

## 1. Goal and non-goals

### 1.1 Goal

Build, in **strict stages**, the simplest possible direct-PPCA ab-initio
path in RECOVAR, and answer three questions in order:

1. **Score test.** Does the marginalized PPCA score rank the true
   pose/translation better than the homogeneous score on heterogeneous
   data?
2. **Mean test.** Given a decent init, does a PPCA-informed loop improve
   the mean and the low-rank subspace beyond what the homogeneous loop
   does?
3. **Bootstrap test.** Once both work on synthetic fixed-grid data, can
   the loop be bootstrapped truly ab-initio (random `W`, or atlas-PCA
   `W` from a coarse external `K`-class run)?

These are *separate* questions and the project must keep them
separate. Conflating them is the dominant historical failure mode.

### 1.2 Explicit non-goals for v0

The following are **out of scope** until each preceding stage is green:

- adaptive / coarse-to-fine pose search (no `get_local_rotation_grid_fast`)
- FSC-driven resolution schedule, `current_size` cropping
- noise re-estimation between iterations
- half-set splitting, FSC-based prior updates
- outlier model
- writing back hard pose assignments into the dataset object
- spectrum (`s`) updates
- exact soft M-step with full second moments — v0 uses generalized EM

If a candidate change touches any of the above, it belongs in a later
phase, not v0.

---

## 2. Repository policy

- **Branch:** `claude/ppca-abinitio-v0`, off `claude/em-relion-parity`.
  Do not target `dev` directly. Do not rebase onto `dev` until the
  parent parity branch lands.
- **Module:** all new code under `recovar/em/ppca_abinitio/`. Do **not**
  edit `recovar/em/heterogeneity.py`, `recovar/em/iterations.py`, or
  `recovar/em/states.py` in this branch (they have separate owners and
  are evolving fast).
- **Tests:** `tests/ppca_abinitio/`.
- **Scripts:** `scripts/ppca_abinitio/` (experiment entrypoints, plotting).
- **Shared helpers:** if a helper from `e_step.py` / `heterogeneity.py`
  must be reused, **import it**, do not copy or refactor it inside this
  branch. Refactors of shared code go through a separate PR against the
  parity branch.

---

## 3. Existing code we reuse (with file:line anchors)

The following pieces already exist on `claude/em-relion-parity` and are
the substrate for v0. Each is referenced by section in the staging plan
below.

| Purpose | File | Symbol |
|---|---|---|
| Dense E-step over (rot, trans) with optional low-rank | `recovar/em/e_step.py:27` | `E_with_precompute(..., u=None, s=None)` |
| Marginalized low-rank score (H, b construction) | `recovar/em/heterogeneity.py:44` | `compute_little_H_b` |
| Marginalized low-rank score (full path) | `recovar/em/heterogeneity.py:84` | `compute_bHb_terms` |
| Equinox version of above | `recovar/em/heterogeneity.py:159` | `compute_bHb_terms_eqx` |
| Mean accumulation (M-step) | `recovar/em/m_step.py:224` | `M_with_precompute` |
| Per-batch accumulation kernel | `recovar/em/m_step.py:118` | `sum_up_images_fixed_rots_eqx` |
| HEALPix rotation grid | `recovar/em/sampling.py:37` | `get_rotation_grid` |
| HEALPix grid by order | `recovar/em/sampling.py:379` | `get_rotation_grid_at_order` |
| Translation grid | `recovar/em/sampling.py:75` | `get_translation_grid` |
| Index → matrix | `recovar/em/sampling.py:85` | `rotation_indices_to_matrices` |
| Reference loop pattern (do **not** reuse directly) | `recovar/em/iterations.py` | `E_M_batches_2`, `split_E_M_v2` |
| Heterogeneous state object (reference, not target) | `recovar/em/states.py:143` | `HeterogeneousEMState` |
| RELION ↔ recovar volume / rotation conversion | `recovar/utils/helpers.py` | `load_relion_volume`, `relion_volume_to_recovar`, `R_to_relion`, `R_from_relion` |

**Why `HeterogeneousEMState` is not the target.** It uses the low-rank
model in the E-step but updates heterogeneity via covariance-column
estimation + projected covariance solves
(`heterogeneity.compute_both_H_B`, `pca_by_projected_covariance`). That
is a fundamentally different M-step from direct PPCA. Treat it as a
reference, not as the class to extend.

---

## 4. Model

### 4.1 Generative model

For image `i` with pose / translation hidden state
`g = (r, t) ∈ G`, low-rank dimensionality `q`, and per-image noise
covariance `Σ_i`:

```
y_i  =  A_g μ  +  A_g U α_i  +  ε_i
α_i  ~  N(0, diag(s))
ε_i  ~  N(0, Σ_i)
```

Stored as `(μ, U, s)` to stay compatible with the existing low-rank
scorer:
- `μ` ∈ `(volume_size,)` — flat centered FT volume (recovar convention,
  see `recovar/CLAUDE.md`).
- `U` ∈ `(q, volume_size)` — orthonormal columns after re-gauging.
- `s` ∈ `(q,)` — non-negative latent variances, descending.

Internally it is often clearer to think in terms of
`W = U · diag(s)^{1/2}`, so the image covariance for fixed `g` is

```
Σ_{i,g}^{img}  =  Σ_i  +  A_g W W^T A_g^T  =  Σ_i  +  A_g U diag(s) U^T A_g^T.
```

### 4.2 Pose-marginalized score

```
p(y_i)  =  Σ_{g ∈ G}  π_g  N(y_i ;  A_g μ ,  Σ_i + A_g W W^T A_g^T).
```

For v0, `π_g = |G|^{-1}` (uniform pose prior).

### 4.3 Latent posterior

For each `(i, g)`, with `Σ_i^{-1}` diagonal in Fourier space:

```
H_{i,g}   =  diag(1/s)  +  U_g^H Σ_i^{-1} U_g                   ∈ R^{q×q}
b_{i,g}   =  U_g^H Σ_i^{-1} (y_i - μ_g)                          ∈ C^{q}
m_{i,g}   =  E[α_i | y_i, g]              =  H_{i,g}^{-1} b_{i,g}
C_{i,g}   =  E[α_i α_i^H | y_i, g]        =  H_{i,g}^{-1} + m_{i,g} m_{i,g}^H
```

with `μ_g = A_g μ`, `U_g = A_g U`. The pose responsibility is
`γ_{i,g} ∝ π_g · p(y_i | g)`.

### 4.4 Translation-independence of `H`

`H_{i,g}` depends on the image-specific CTF and on `r` (through
`U_g = A_r U`), but **not** on the translation index `t`, because
translation acts as a unitary phase in Fourier space and cancels in
`U^H Σ^{-1} U`. This is exactly how
`heterogeneity.compute_little_H_b` (`recovar/em/heterogeneity.py:44`) is
written, and it is the reason the small (q × q) Cholesky cost is
amortized across translations. **The new helper must preserve this
structure.** Materializing `H_{i,r,t}` would inflate cost by `n_trans`.

---

## 5. Fixed-grid v0 — design contract

### 5.1 Discretization

- Rotations: HEALPix order **2** or **3**, no oversampling.
- Translations: integer grid, `max_shift ∈ {1, 2}` pixels, step `1`.
- All synthetic data is generated **on the same grid** that inference
  uses. There is no off-grid pose noise in v0.

### 5.2 Frozen quantities (v0)

`Σ_i` (noise variance), regularization strength on `μ`, latent
dimensionality `q`, the grid `G`, the interpolation type (`linear_interp`),
and the support mask are all **fixed** for the entire run. The loop
must not mutate any of them.

### 5.3 Self-contained loop

The v0 loop is a fresh, ~100-line orchestrator in
`recovar/em/ppca_abinitio/loop.py`. It does **not** call
`E_M_batches_2` or `split_E_M_v2` (those couple to half-sets, hard pose
writeback, FSC priors). Per-iteration accumulators are **explicitly
zeroed** at the top of each iteration. No cross-iteration hidden state
beyond `(μ, U, s)` and a small metrics dict.

---

## 6. Module layout

```
recovar/em/ppca_abinitio/
    __init__.py
    types.py            # PPCAInit, FixedGridSpec, PPCABatchPosterior, PPCAConfig, PPCAMetrics
    grid.py             # build_fixed_grid(): wraps sampling.get_rotation_grid_at_order + get_translation_grid
    synthetic.py        # make_synthetic_fixed_grid_dataset(...)
    posterior.py        # score_and_posterior_moments_eqx(...) — Section 7
    mean_update.py      # mean-only updates (v0 + v1, Section 8.2)
    factor_update.py    # generalized-EM W/U updates + re-gauge (Section 8.3)
    loop.py             # run_score_diagnostic, run_fixed_grid_ppca_gem
    init.py             # truth-perturbed, random-lowpass, atlas-PCA initializers
    atlas.py            # K-class volume alignment + PCA compression (Phase 3)
    metrics.py          # hidden-state, mean, subspace, embedding, optimization metrics
    relion_io.py        # thin wrappers over recovar.utils.helpers for K-class import (Phase 3)
```

```
tests/ppca_abinitio/
    test_posterior_small.py                 # tiny brute-force ground truth
    test_score_matches_existing_e_step.py   # parity vs heterogeneity.compute_bHb_terms_eqx
    test_gauge_fix_invariance.py
    test_fixed_grid_synthetic_score_gain.py # Stage 1A
    test_mean_only_update.py                # Stage 1B
    test_factor_update_smoke.py             # Stage 1C
    test_atlas_pca_init.py                  # Phase 3
```

```
scripts/ppca_abinitio/
    run_score_diagnostic.py
    run_fixed_grid_ppca.py
    run_atlas_bootstrap.py
```

---

## 7. Posterior helper — first new math object

This is the central new function. It returns score and posterior
moments in one pass over the batch.

### 7.1 Signature

```python
@eqx.filter_jit
def score_and_posterior_moments_eqx(
    config: ForwardModelConfig,
    mean_projections,   # (n_rot, image_size)             complex
    u_projections,      # (n_rot, q, image_size)          complex
    s,                  # (q,)                            real, > 0
    batch,              # (n_img, image_size)             complex
    translations,       # (n_trans, 2)                    real
    ctf_params,
    noise_variance,
) -> PPCABatchPosterior:
    ...
```

### 7.2 Returned dataclass

```python
@dataclass
class PPCABatchPosterior:
    log_scores:    jnp.ndarray  # (n_img, n_rot, n_trans)   marginalized log p(y_i | g)
    log_resp:      jnp.ndarray  # (n_img, n_rot, n_trans)   normalized log γ_{i,g}
    post_mean:     jnp.ndarray  # (n_img, n_rot, n_trans, q)  m_{i,g}
    post_Hinv:     jnp.ndarray  # (n_img, n_rot, q, q)        H_{i,r}^{-1}, translation-independent
```

**Why `post_Hinv` and not the full `C_{i,g}`.** `H_{i,r}^{-1}` does not
depend on `t`, so storing one `(q, q)` per `(i, r)` is `n_trans×`
cheaper than materializing `C` per `(i, r, t)`. The M-step can form
`C = Hinv + m m^H` on the fly when it needs the second moment.

### 7.3 Internal implementation

1. Build `mean_projections` and `u_projections` once per iteration (the
   loop, not the helper, owns this — same pattern as `E_with_precompute`).
2. Form `H = diag(1/s) + U_r^H Σ^{-1} U_r` per `(i, r)` — reuse the
   exact recipe in `compute_UPLambdainvPU`
   (`recovar/em/heterogeneity.py:24`). Do not duplicate the math —
   import it.
3. Form `b_{i,r,t}` per `(i, r, t)` — reuse `compute_bLambdainvPU_terms`
   (`recovar/em/heterogeneity.py:127`). Same rule: import, don't copy.
4. Solve `m = H^{-1} b` and `bHinvb = b^H m` via a single Cholesky per
   `(i, r)`, vmapped across the batch — same structure as
   `compute_bHb_terms_eqx` (`recovar/em/heterogeneity.py:159`).
5. Build `log_scores` from `bHinvb` and `log det H` plus the
   homogeneous-residual term so that the result is the **full**
   marginal log-likelihood up to constants that do not depend on `g`.
6. Normalize within each image to obtain `log_resp`.

### 7.4 Parity test (mandatory)

`tests/ppca_abinitio/test_score_matches_existing_e_step.py` must show
that, on a small synthetic batch:

```
log_scores from score_and_posterior_moments_eqx(...)
   ==
log_scores produced inside E_with_precompute(..., u=U, s=s)
   (after subtracting any image-only normalizing constants)
```

up to `1e-6` relative tolerance in float64. This is the *contract* that
locks the new helper to the existing low-rank scorer. If the test fails,
the new helper is wrong; do not relax the tolerance.

### 7.5 Brute-force test (mandatory)

`tests/ppca_abinitio/test_posterior_small.py` must, for `q ≤ 3`,
`n_rot ≤ 4`, `n_trans ≤ 4`, `image_size ≤ 32`, compare `m`, `Hinv`, and
`log_scores` against a direct numpy implementation that materializes
`Σ_{i,g}^{img}` and inverts it densely. Tolerance `1e-6` (float64).

---

## 8. M-step ladder

### 8.1 Parameter discipline

| Stage | μ updated? | U updated? | s updated? | Method |
|---|---|---|---|---|
| 1A | ✗ | ✗ | ✗ | score-only diagnostic |
| 1B | ✓ | ✗ | ✗ | mean-only loop, fixed `W` |
| 1C | ✓ | ✓ (GEM) | ✗ | generalized-EM `W` update + re-gauge |
| 1D | ✓ | ✓ (soft M-step) | ✗ | full second moments, ECM |
| 2 | ✓ | ✓ | ✗ | true ab-initio, external mean bootstrap |
| 3 | ✓ | ✓ | possibly ✓ | atlas-PCA bootstrap |

`s` is frozen until 1D is green. Reason: spectrum and subspace
identifiability problems are easy to confuse, and the latter is the
question we care about.

### 8.2 Mean update

#### 8.2.1 Mean update v0 (Stage 1B)

Reuse `M_with_precompute` (`recovar/em/m_step.py:224`) and
`sum_up_images_fixed_rots_eqx` (`recovar/em/m_step.py:118`) **as-is**,
but feed them PPCA pose responsibilities `γ_{i,g}` from
`score_and_posterior_moments_eqx`. **Ignore** the latent correction
term — i.e. backproject `y_i`, not `y_i - A_g U m_{i,g}`.

This is intentionally not the exact M-step. Its job is to answer the
diagnostic question: do PPCA-shaped responsibilities give a better
mean than homogeneous ones, even before residualization?

#### 8.2.2 Mean update v1 (after 1B is green)

Backproject the **residualized** image:

```
y_i^res(g) = y_i - A_g U m_{i,g}
```

This is the first PPCA-correct mean update. Implement it as a thin
wrapper around the same accumulation kernel, with the residual computed
inside the per-batch closure so we never hold all `(i, r, t)` residuals
in memory.

### 8.3 Factor update

#### 8.3.1 W update v0 — generalized EM (Stage 1C)

**Objective.** With `μ` fixed (or only updated by mean-update v0/v1),
take K (≈3) gradient steps on the expected complete-data negative
log-likelihood w.r.t. `W`, using the posterior moments
`(γ, m, Hinv)` from the current E-step:

```
L(W) = Σ_{i,g} γ_{i,g} ·
        E_{α | y_i, g} [ ‖ y_i − A_g μ − A_g W α ‖_{Σ_i^{-1}}^2 ]
       + ‖ W ‖_{prior}^2
```

The expectation is closed-form because
`E[α α^H] = Hinv + m m^H`.

**Implementation rule.** Use `jax.value_and_grad` over a small JIT'd
closure that takes `(W, batch_state)` and returns scalar `L`. Do not
hand-derive the gradient. The gradient is non-trivial and easy to get
wrong; the autodiff version is the source of truth.

**Gauge fix (mandatory after every step).** PPCA factors are unique
only up to a `q × q` orthogonal rotation, and the loop will drift
without an explicit re-gauge:

1. Form `W ∈ C^{volume_size × q}`.
2. Thin SVD: `W = U_w Σ_w V_w^H`.
3. Set `U ← U_w`, `s ← Σ_w^2` (descending).
4. Rotate any cached posterior means: `m ← V_w m`.

After re-gauging, columns of `U` are orthonormal in the
volume-mass-weighted inner product the rest of the codebase uses.

#### 8.3.2 W update v1 — full soft M-step (Stage 1D)

Closed-form weighted ridge solve using full second moments
`C = Hinv + m m^H`. Only attempt this **after** 1C is observed to
behave: subspace angles improve over iterations, the embedding metric
does not collapse, and the generalized-EM objective is monotone or
nearly so.

---

## 9. Synthetic data harness

`recovar/em/ppca_abinitio/synthetic.py:make_synthetic_fixed_grid_dataset`.

### 9.1 Inputs

- `mu_true` — flat centered-FT volume in recovar convention.
- `U_true` — `(q, volume_size)`, orthonormal columns.
- `s_true` — `(q,)`, descending positive.
- `n_images`, RNG seed.
- `FixedGridSpec` (rotations, translations).
- noise level (`Σ_i = σ^2 I` is fine for v0).
- CTF distribution (existing recovar CTF generator is fine).

### 9.2 Output

A `CryoEMDataset` (or a duck-typed equivalent that
`E_with_precompute` accepts) plus a "ground truth bundle":

```python
@dataclass
class GroundTruth:
    g_true:     np.ndarray  # (n_images,)  index into G
    r_true_idx: np.ndarray  # (n_images,)  rotation index
    t_true_idx: np.ndarray  # (n_images,)  translation index
    alpha_true: np.ndarray  # (n_images, q) latent coords
    mu_true:    np.ndarray
    U_true:     np.ndarray
    s_true:     np.ndarray
```

### 9.3 Ground-truth source

For v0, hand-built low-resolution `(μ, U)` is sufficient (e.g. a small
Gaussian blob mean plus 2–3 sinusoidal PCs). Do **not** rely on a
RECOVAR run for ground truth in v0 — that introduces a chicken-and-egg
dependency.

### 9.4 Initialization controls

`recovar/em/ppca_abinitio/init.py` exposes:

- `init_truth_perturbed(gt, eps_mu, eps_U)` — Stage 1B/1C positive control
- `init_random_lowpass(volume_shape, q, cutoff)` — Stage 1C negative control
- `init_from_external_mean(path)` — Phase 2
- `init_from_aligned_class_atlas(volumes, weights, q)` — Phase 3

---

## 10. Metrics

`recovar/em/ppca_abinitio/metrics.py`. Metrics are grouped by purpose
and reported separately. Do not collapse them into a single number.

### 10.1 Hidden-state / alignment

- `top1_acc` — fraction of images whose argmax `g` equals `g_true`.
- `true_state_mass` — mean of `γ_{i, g_true(i)}`.
- `true_state_rank` — mean rank of the true `g` in the per-image score.
- `angular_error_deg` — angular distance between `R_argmax` and
  `R_true` (use existing `_angular_distance_matrices`,
  `recovar/em/sampling.py:409`).
- `translation_error_px`.

### 10.2 Mean

- `fsc(mu_est, mu_true)` and the half-bit / 0.143 thresholds.
- Fourier-norm error.

### 10.3 Subspace

- Principal angles between `span(U_est)` and `span(U_true)`.
- Projector Frobenius error
  `‖ P_{U_est} − P_{U_true} ‖_F`.

### 10.4 Spectrum (only after Stage 1D)

- Relative error of `s_est`.
- Total low-rank variance ratio.

### 10.5 Embedding (two flavors, both reported)

- **Oracle embedding error.** Use `m_{i, g_true(i)}` and compare to
  `α_true_i` after orthogonal Procrustes. This isolates the latent
  posterior from pose errors.
- **Marginal embedding error.** Use
  `α̂_i = Σ_g γ_{i,g} m_{i,g}` and compare to `α_true_i` after
  orthogonal Procrustes. This is the metric a downstream user would
  see.

### 10.6 Optimization

- Mean log-likelihood per image, per iteration.
- Posterior entropy over poses (per image, mean across batch).
- For Stage 1C: monotonicity flag for the GEM objective.

---

## 11. Staging plan and exit criteria

The loop must not advance to the next stage until the previous one
satisfies its exit criterion on the synthetic harness.

### 11.1 Stage 1A — score-only diagnostic

**Implement.** `run_score_diagnostic(dataset, init, grid)` that calls
`E_with_precompute` twice — once with `u=None`, once with `u=U`,
`s=s` — and reports Section 10.1 metrics for both.

**Exit criterion.** PPCA scoring beats the homogeneous baseline on
**at least one** alignment metric, on truth-perturbed init **and** on
random-lowpass init for the heterogeneous synthetic dataset. If neither
beats the baseline, **stop and debug** — there is no reason to build a
PPCA M-step.

### 11.2 Stage 1B — mean-only PPCA loop

**Implement.** `run_fixed_grid_ppca_gem(..., update_mu=True,
update_factor=False)` using mean-update v0 (Section 8.2.1) and the new
posterior helper.

**Exit criterion.** From a truth-perturbed init, the mean-FSC trajectory
of the PPCA loop is **strictly above** the mean-FSC trajectory of the
homogeneous loop after ≥3 iterations on a heterogeneous synthetic
dataset. The subspace and `s` are still frozen at this stage.

### 11.3 Stage 1C — generalized-EM `W` update

**Implement.** mean-update v1 (Section 8.2.2) plus W-update v0
(Section 8.3.1) plus re-gauge.

**Exit criterion.** All of:

1. The mean does not collapse (mean-FSC does not regress past the
   init's value).
2. Subspace principal angles improve over iterations from
   truth-perturbed init.
3. Marginal embedding error improves over iterations from
   truth-perturbed init.
4. With `s` frozen, the per-iteration GEM objective is monotone within
   `1e-3` relative noise (small non-monotonicity from finite-step
   gradient updates is acceptable).
5. Random-lowpass init does not blow up (no NaN, no `s` explosion via
   re-gauge).

### 11.4 Stage 1D — full soft M-step

Implement only after Stage 1C exit criteria are green and stable across
≥3 distinct synthetic seeds. No exit criterion is committed yet — that
will be added in a follow-up edit to this doc when 1C results are in.

### 11.5 Phase 2 — external mean bootstrap

**Goal.** Initialize `μ` from an external homogeneous ab-initio
(RELION / cryoSPARC / a homogeneous RECOVAR run, in that order of
preference because of pose convention complexity), initialize `W`
random-lowpass and small, and run Stage 1C.

**Pose-convention guard.** Any externally produced volume must be
brought into recovar convention via
`recovar/utils/helpers.load_relion_volume` or
`relion_volume_to_recovar` **before** PCA, FSC, or projection. Do not
re-derive this transform — it is pinned by
`tests/unit/test_relion_volume_convention.py` and explained in
`recovar/em/CLAUDE.md`. The negation in `vol_recovar = -np.transpose(...)`
is paired with `R_to_relion` / `R_from_relion`; do not "fix" either
half independently.

**Exit criterion.** PPCA loop produces a non-trivial subspace (subspace
angles vs the highest-likelihood subspace from a longer reference run
< 60° on at least one component) without `μ` collapsing. This is a
stress test, not the final initializer.

### 11.6 Phase 3 — K-class atlas-PCA bootstrap

**Pipeline.**

1. Run a coarse external `K`-class ab-initio (RELION
   `relion_refine --K 4` or similar) on a synthetic or benchmark
   heterogeneous dataset.
2. Convert each `V_k` to recovar frame via `load_relion_volume`.
3. Align them in recovar's frame. For v0 a global least-squares
   alignment to the K=1 mean is enough; rigid Procrustes is fine.
4. Compute atlas mean `μ_0 = Σ_k π_k V_k`, deviations `D_k = V_k − μ_0`.
5. PCA / SVD over `{D_k}` (stack as columns) to obtain `U_0`, `s_0`.
6. Run the Stage-1C loop from `(μ_0, U_0, s_0)`.

**Exit criterion.** Loop is stable, mean and subspace metrics are not
worse than Phase 2 on the same dataset, and the subspace recovers a
direction that is *not* in the column span of the K-class deviation
basis (proof that PPCA is doing more than reproducing the input atlas).

### 11.7 Phase 4 — dataset sweep + faster E-step

Out of scope for v0. Tracked here for completeness:

- CryoBench sweep on the cluster path the user provided.
- Coarse-to-fine pose search using
  `get_local_rotation_grid_fast` (`recovar/em/sampling.py:544`) and
  `get_oversampled_translation_grid` (`recovar/em/sampling.py:305`).

---

## 12. Concrete API

```python
# types.py
@dataclass
class PPCAInit:
    mu: jnp.ndarray   # (volume_size,) complex
    U:  jnp.ndarray   # (q, volume_size) complex
    s:  jnp.ndarray   # (q,) real, descending

@dataclass
class FixedGridSpec:
    rotations:     jnp.ndarray   # (n_rot, 3, 3) real
    translations:  jnp.ndarray   # (n_trans, 2)  real
    log_prior:     jnp.ndarray | None = None  # (n_rot * n_trans,) or None for uniform

@dataclass
class PPCAConfig:
    n_iters:        int
    update_mu:      bool = True
    update_factor:  bool = False
    update_s:       bool = False
    factor_inner_steps: int = 3
    factor_lr:      float = 1e-2
    seed:           int = 0

@dataclass
class PPCABatchPosterior:
    log_scores:  jnp.ndarray  # (n_img, n_rot, n_trans)
    log_resp:    jnp.ndarray  # (n_img, n_rot, n_trans)
    post_mean:   jnp.ndarray  # (n_img, n_rot, n_trans, q)
    post_Hinv:   jnp.ndarray  # (n_img, n_rot, q, q)

# grid.py
def build_fixed_grid(healpix_order: int, max_shift: int, shift_step: int = 1) -> FixedGridSpec: ...

# synthetic.py
def make_synthetic_fixed_grid_dataset(
    mu_true, U_true, s_true, grid: FixedGridSpec, n_images: int,
    noise_sigma: float, ctf_distribution, seed: int,
) -> tuple[CryoEMDataset, GroundTruth]: ...

# posterior.py
def score_and_posterior_moments_eqx(
    config, mean_projections, u_projections, s, batch, translations,
    ctf_params, noise_variance,
) -> PPCABatchPosterior: ...

# loop.py
def run_score_diagnostic(
    dataset, init: PPCAInit, grid: FixedGridSpec, gt: GroundTruth,
) -> PPCAMetrics: ...

def run_fixed_grid_ppca_gem(
    dataset, init: PPCAInit, grid: FixedGridSpec, cfg: PPCAConfig,
    gt: GroundTruth | None = None,
) -> tuple[PPCAInit, list[PPCAMetrics]]: ...

# init.py
def init_truth_perturbed(gt, eps_mu: float, eps_U: float) -> PPCAInit: ...
def init_random_lowpass(volume_shape, q: int, cutoff_px: int, seed: int) -> PPCAInit: ...
def init_from_external_mean(mu_path: str, q: int, cutoff_px: int, seed: int) -> PPCAInit: ...
def init_from_aligned_class_atlas(
    volumes: list[jnp.ndarray], weights: jnp.ndarray, q: int,
) -> PPCAInit: ...
```

---

## 13. Test plan (must exist before any large run)

| Test | Stage | Purpose |
|---|---|---|
| `test_posterior_small.py` | pre-1A | Brute-force ground truth for `m`, `Hinv`, `log_scores` (q ≤ 3, image_size ≤ 32) |
| `test_score_matches_existing_e_step.py` | pre-1A | Parity vs `compute_bHb_terms_eqx` to `1e-6` in float64 |
| `test_gauge_fix_invariance.py` | pre-1C | Re-gauge preserves `W W^H` and `log_scores` |
| `test_fixed_grid_synthetic_score_gain.py` | 1A | PPCA score beats homogeneous on at least one metric |
| `test_mean_only_update.py` | 1B | Mean-FSC trajectory dominates the homogeneous loop |
| `test_factor_update_smoke.py` | 1C | No NaN, monotone-ish GEM objective, subspace angles improve |
| `test_atlas_pca_init.py` | Phase 3 | Atlas-PCA initializer produces a valid `(μ, U, s)` |

All tests must run on CPU in float64 in under a minute each so they
fit in `pixi run test-fast`. The synthetic problems are tiny by
construction (image_size ≤ 32, n_images ≤ 64).

---

## 14. Known failure modes and mitigations

1. **Spectrum collapse.** `s` shrinks. Mitigation: freeze `s` until
   1D; report subspace and embedding metrics separately so a poor
   embedding scale is not misread as a wrong subspace.
2. **Mean absorbs heterogeneity.** Weak / bad `W` causes the mean
   update to average heterogeneity into `μ`. Mitigation: Stage 1A
   isolates this by freezing `μ`. If 1A is green but 1B regresses,
   the mean update is the culprit, not the score.
3. **Random `W` destabilizes pose search** in Phase 2. Mitigation:
   keep `W` small in norm and aggressively low-pass filtered. Phase 2
   is a stress test, not the intended initializer.
4. **Hidden coupling to existing EM state.** `EMState` /
   `HeterogeneousEMState` accumulate cross-iteration statistics that
   are easy to depend on without noticing. Mitigation: v0 loop is
   self-contained; do not import from `iterations.py` or `states.py`.
5. **Premature local refinement.** Switching to coarse-to-fine before
   the dense path is green makes failures unattributable. Mitigation:
   no local grid utilities until Phase 4.
6. **RELION pose / volume convention drift.** Any external map must go
   through `load_relion_volume`; any external rotation must go through
   `R_from_relion`. The transform's negation is intentional — see
   `recovar/em/CLAUDE.md` and `recovar/CLAUDE.md`. Pinned by
   `tests/unit/test_relion_volume_convention.py`.

---

## 15. PR / validation policy for v0

Per `CLAUDE.md`, no PR for this work is "ready" until:

1. `pixi run test-fast` is green, including the new
   `tests/ppca_abinitio/` tests.
2. `./scripts/run_tests_parallel.sh long-test` is green and the
   summary log has been read.
3. Quality and performance regression tables (per the format in
   `CLAUDE.md`) are in the PR body.
4. The PR description references the section of this doc that the
   code implements, and **explicitly states which exit criterion
   from Section 11 the code satisfies.**

A PR that touches PPCA-ab-initio code without satisfying the matching
Section 11 exit criterion should not be merged.

---

## 16. Open questions (to be resolved as we go)

- Concrete choice of CTF distribution and `σ^2` for the synthetic
  harness — placeholder until Stage 1A is wired up.
- Whether the GEM gradient step needs a small ridge on `W` for
  numerical stability with `q > 5`.
- Whether atlas alignment in Phase 3 needs RELION-style `flatten_solvent`
  treatment of class volumes before PCA.
- Whether the marginal embedding metric should be reported with or
  without re-gauging applied to `m` before Procrustes.

When any of these is resolved, update this doc in the same PR.
