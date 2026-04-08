# PPCA ab-initio v0 — Plan & Spec

**Status:** post-review draft, pre-implementation
**Owner:** ma-gilles / mg6942
**Branch:** `claude/ppca-abinitio-v0`
**Parent commit:** `3a39533212c7955d59507473544fa821c9b4eb6e` on
`claude/em-relion-parity`. Do not silently rebase past this commit;
when a rebase is needed, update this line in the same commit and
audit any imported helpers that moved.
**Module root:** `recovar/em/ppca_abinitio/`
**Companion docs:** `docs/math/plan_relion_parity.md`,
`docs/math/plan_relion_parity_v2.md`, `recovar/em/CLAUDE.md`,
`docs/math/review_ppca_abinitio_v0_codex_20260408.md`

This document is the authoritative spec for the **direct PPCA ab-initio**
experiments. It is the contract that the implementation must satisfy.
Subsequent code, tests, and PR descriptions should reference section
numbers from here. Do not silently deviate; if something below is
wrong, update this doc in the same PR that changes behavior.

This is a **post-review** revision. Two adversarial critiques (one
from the original author, one from an independent reviewer in
`docs/math/review_ppca_abinitio_v0_codex_20260408.md`) have been
folded in. Where the two critiques disagreed, the rationale for the
choice is documented in Section 0.

---

## 0. Resolved decisions and prerequisite audit results

### 0.1 Math / dtype audit results (do not redo)

The two prerequisite tests committed on this branch
(`tests/ppca_abinitio/test_compute_bHb_terms_correctness.py` and
`tests/ppca_abinitio/test_compute_bHb_terms_dtype.py`) have already
established the following. New code may rely on them; new audits of
the same properties are not needed.

1. **`recovar/em/heterogeneity.py:84:compute_bHb_terms` is
   mathematically correct** to `rtol=1e-10` against an independent
   dense numpy reference. The function returns
   `b^H H^{-1} b - log det H`, which the caller subtracts from the
   homogeneous squared residual. The local variable named
   `log_det_H` is actually `-log det H`; the final value is correct
   despite the misnomer.

2. **Float64 propagation through the low-rank scorer is clean**
   end-to-end. With complex128 / float64 inputs,
   `compute_UPLambdainvPU`, `compute_bLambdainvPU_terms`, and
   `compute_bHb_terms` all return float64 with no hidden downcast.
   The reason production runs at float32 is solely that
   `CryoEMDataset.dtype` defaults to `np.complex64` and
   `E_with_precompute` creates `u_projections` as `complex64`
   explicitly. The function itself is dtype-agnostic.

3. **`recovar/em/heterogeneity.py:159:compute_bHb_terms_eqx` is
   dead code** with a stale `logger.warning("Make sure this is
   correct...")`. It is never called from production. The legacy
   `compute_bHb_terms` is the parity oracle for this spec. Cleanup
   of the dead `_eqx` twin is filed against the parity branch
   separately.

### 0.2 Open-question decisions (committed)

These were left open in the previous draft and are now resolved:

| # | Question | Decision |
|---|---|---|
| Q1 | Is Stage 1B a graduation gate or only a debugging ablation? | **Ablation only.** Stage 1C subsumes it; do not let 1B graduate the project. |
| Q2 | "Frozen `s`" — strict `U`-only updates, or allow `s` to drift? | **Strict `U`-only.** `s` is literally constant during 1C. The W = U·diag(s)^{1/2} re-SVD pattern is dropped. |
| Q3 | Should `HeterogeneousEMState` be a required baseline at 1C and Phase 2? | **Yes, required.** If direct PPCA cannot beat or match the existing in-tree heterogeneous learner on the same fixed-grid harness, the project does not justify itself. *Not* a baseline at 1A — the scorer there is identical. |
| Q4 | Is HEALPix order 3 required for v0? | **No, order 2 only.** Order 3 is deferred to Phase 4 (post-streaming-API). |
| Q5 | Continuous heterogeneity only, or also mixtures / multimodal? | **Continuous only.** PPCA on multimodal data is an approximation and out of scope until v1; explicitly stated as a non-goal. |
| Q6 | Held-out validation split + null family in v0? | **Mandatory.** Synthetic data is split into train/val seeds; primary metrics are reported on validation only. The null family (`s_true=0` or `q_true=0`) is a graduation gate at every stage. |
| Q7 | External atlas: RELION only, or also cryoSPARC? | **RELION only in v0.** cryoSPARC import is its own convention/scale audit and lives in v1. |

### 0.3 Reviewer-driven structural changes

The independent review surfaced four structural problems that
required more than localized edits. They are baked into the rest of
this document:

- **The PPCA gauge group is real orthogonal `O(q)`, not complex
  unitary `U(q)`.** A complex SVD of `W` is not a valid gauge fix;
  it can leave the learned columns outside the real-volume Fourier
  subspace (i.e. they no longer correspond to FTs of real volumes).
  Section 8.3 is rewritten around a real-volume parameterization.

### 0.3.1 User-driven correction (2026-04-08): half-volume layout

A subsequent user correction replaced the original "enforce
Hermitian symmetry on full-volume FTs" design with the **half-volume
rfft layout** that the rest of the codebase already uses
(`recovar/em/dense_single_volume/engine_v2.py`). Key consequences:

- `μ` and each row of `U` are stored in `(N0, N1, N2//2+1)`
  rfft-packed half-volume layout (flattened to
  `half_volume_size = N0 * N1 * (N2//2+1)`).
- The Hermitian-symmetry constraint is **structural**, not
  enforced. There is no `enforce_real_volume_ft` step; the layout
  simply does not represent the redundant half of the spectrum.
  Any complex array of the right rfft shape decodes via
  `recovar.core.fourier_transform_utils.get_idft3_real` to a
  real-valued volume by construction.
- Slicing and backprojection use `slice_volume(...,
  half_volume=True, half_image=True)` and the matching
  `adjoint_slice_volume(...)` (`recovar/core/slicing.py:222, 331`).
- **Inner products on the half-spectrum require an rfft Hermitian
  weight**: `2` for interior packed-axis columns (which represent
  conjugate pairs), `1` on the DC column (`kx=0`) and the Nyquist
  column (`kx=N2/2`, even N only). With these weights,
  `Σ_k w(k) Re(conj(a_half) b_half) = Re<a_full, b_full>` exactly.
  The 3D weight is provided by
  `recovar.em.ppca_abinitio.half_volume.make_half_volume_weights`,
  matching the 2D `engine_v2.make_half_image_weights` recipe.
- Real-space orthonormalization on the rows of a `(q, N_half)`
  matrix is done via Cholesky-whitening of the **weighted** Gram
  `(U * w) @ U^H / N_full` (see
  `real_volume_orthonormalize_half`).
- **The dense `(n_img, n_rot, n_trans, q)` posterior tensor is a
  memory dead-end** and silently locks out significant-weight
  pruning, local-grid search, and the streaming pattern that
  `recovar/em/dense_single_volume/engine_v2.py:500-540` already
  uses. Section 6 and Section 7 are rewritten around a streaming
  block iterator.
- **Stage 11.1 ("PPCA score beats homogeneous on at least one
  metric")** is the multiple-comparisons trap. Section 11 is
  replaced with the reviewer's pre-registered-primary-metric
  staging plan, with explicit null-family and misspecified-family
  gates.
- **`fsc(mu_est, mu_true)` is not a split-map FSC** and the
  half-bit / 0.143 thresholds do not apply. Section 10 is corrected
  to use `oracle_fsc_gt` with non-thresholded summary stats.

---

## 1. Goal and non-goals

### 1.1 Four separate claims (do not conflate)

Build, in **strict stages**, the simplest possible direct-PPCA ab-initio
path in RECOVAR, and answer four questions in order:

1. **Oracle score helps.** Given the *true* `(μ, U, s)`, does the
   marginalized PPCA score rank the true pose / translation better
   than the homogeneous score on heterogeneous data — and **not**
   on homogeneous (null) data?
2. **Residualized mean helps.** With a fixed PPCA factor, does
   residualized mean reconstruction beat the homogeneous mean
   reconstruction loop on the same synthetic family?
3. **Factor learning helps.** With a truth-perturbed init and `s`
   frozen, does updating `U` improve the subspace and the marginal
   embedding without destabilizing the score or the mean?
4. **Bootstrap reaches the basin.** From a non-oracle initializer
   (external homogeneous mean; or atlas-PCA), does the loop reach
   a usable basin?

These are *separate* questions and the project must keep them
separate. Conflating them is the dominant historical failure mode.
Random-`W` from-scratch bootstrap is a **stress test only**, not a
graduation path; it lives in its own lane and never gates the
project.

### 1.2 Explicit non-goals for v0

Out of scope until each preceding stage is green:

- adaptive / coarse-to-fine pose search
- HEALPix order > 2 (per Q4)
- multimodal / discrete-class heterogeneity (per Q5)
- FSC-driven resolution schedule, `current_size` cropping
- noise re-estimation between iterations
- half-set splitting and gold-standard FSC (no half-bit / 0.143
  thresholds anywhere in v0)
- outlier model
- writing back hard pose assignments into the dataset object
- spectrum (`s`) updates
- exact soft M-step with full second moments (deferred to 1D)
- cryoSPARC import (per Q7)

If a candidate change touches any of the above, it belongs in a
later phase, not v0.

---

## 2. Repository policy

- **Branch:** `claude/ppca-abinitio-v0`, branched off
  `claude/em-relion-parity` at commit
  `3a39533212c7955d59507473544fa821c9b4eb6e`. Pinned. Rebases must
  bump the pin in the same commit and re-audit imported helpers.
- **Module:** all new code under `recovar/em/ppca_abinitio/`.
- **Tests:** unit tests in `tests/ppca_abinitio/`. Stage gate
  experiments are **scripts**, not tests; see Section 13.
- **Scripts:** `scripts/ppca_abinitio/` (experiment entrypoints,
  plotting, JSON outputs).
- **Shared-helper edits.** Do not fork local copies of helpers
  inside `ppca_abinitio/`. If a function in
  `recovar/em/heterogeneity.py` or `recovar/em/e_step.py` needs a
  new optional kwarg with default-preserving behavior, that change
  goes through a separate narrow PR against the parity branch with
  its own parity test pinning the old behavior. Do not edit
  `recovar/em/heterogeneity.py`, `recovar/em/iterations.py`, or
  `recovar/em/states.py` *behaviorally* in this branch.

---

## 3. Existing code we reuse (with file:line anchors)

The following pieces already exist on the pinned parent commit and
are the substrate for v0. Each is referenced by section in the
staging plan.

| Purpose | File | Symbol |
|---|---|---|
| Dense E-step over (rot, trans) with optional low-rank | `recovar/em/e_step.py:27` | `E_with_precompute(..., u=None, s=None)` |
| Marginalized low-rank score (H, b construction) | `recovar/em/heterogeneity.py:44` | `compute_little_H_b` |
| Marginalized low-rank score — **production oracle** | `recovar/em/heterogeneity.py:84` | `compute_bHb_terms` |
| Per-image dot products | `recovar/em/core.py:82` | `compute_dot_products_eqx` |
| CTF-projected norms | `recovar/em/core.py` | `compute_CTFed_proj_norms_eqx` |
| Mean accumulation (M-step) | `recovar/em/m_step.py:224` | `M_with_precompute` |
| Per-batch accumulation kernel | `recovar/em/m_step.py:118` | `sum_up_images_fixed_rots_eqx` |
| RELION-style mean post-process solve | `recovar/reconstruction/relion_functions.py` | `post_process_from_filter` |
| HEALPix grid by order | `recovar/em/sampling.py:379` | `get_rotation_grid_at_order` |
| Translation grid | `recovar/em/sampling.py:75` | `get_translation_grid` |
| Index → matrix | `recovar/em/sampling.py:85` | `rotation_indices_to_matrices` |
| Angular distance utility | `recovar/em/sampling.py:409` | `_angular_distance_matrices` |
| Streaming-engine pattern (mirror this) | `recovar/em/dense_single_volume/engine_v2.py:500-540` | `engine_v2` blockwise loop |
| Reference loop pattern (do **not** reuse directly) | `recovar/em/iterations.py` | `E_M_batches_2`, `split_E_M_v2` |
| Existing heterogeneous learner — **baseline at 1C / Phase 2 only** | `recovar/em/states.py:143` | `HeterogeneousEMState` |
| RELION ↔ recovar volume / rotation conversion | `recovar/utils/helpers.py` | `load_relion_volume`, `relion_volume_to_recovar`, `R_to_relion`, `R_from_relion` |

Note: `compute_bHb_terms_eqx` (`recovar/em/heterogeneity.py:159`)
is **dead code** in this branch and is *not* the parity oracle.
Do not reference it from new code or new tests.

**Why `HeterogeneousEMState` is not the implementation target,
but is a baseline.** Its E-step is exactly the production low-rank
scorer (`recovar/em/states.py:180-191`) — so as a Stage 1A score
baseline it is a no-op (same scorer). But its M-step is
covariance-column estimation with a projected covariance solve
(`recovar/em/states.py:194-284`,
`recovar/em/heterogeneity.py:322-685`), which is a fundamentally
different way of learning the subspace from direct PPCA. That makes
it the right *learning-stage* baseline at 1C and Phase 2: if direct
PPCA cannot beat or at least match it, the project does not justify
itself.

---

## 4. Model

### 4.1 Generative model

For image `i` with pose / translation hidden state
`g = (r, t) ∈ G`, low-rank dimensionality `q`, and per-image noise
covariance `Σ_i`:

```
y_i  =  CTF_i · A_g (μ + U α_i)  +  ε_i
α_i  ~  N(0, diag(s))
ε_i  ~  N(0, Σ_i)
```

`α_i` is **real-valued**. This is non-negotiable: the latent prior
is `N(0, diag(s))` over `R^q`, and that fixes the gauge group as
real orthogonal `O(q)`. Any update step or gauge fix that produces
complex `α` or applies a complex unitary to `U` is wrong.

### 4.2 Representation contract (load-bearing)

`μ` and each row of `U` correspond to **real-space 3D volumes**.
They are stored in **half-volume rfft layout**: a flat complex128
vector of length `half_volume_size = N0 * N1 * (N2//2 + 1)`. The
half-volume layout makes Hermitian symmetry structural — the
redundant half of the spectrum is simply not stored. There is no
projection-back step.

This means:

- The layout is the same one `recovar/em/dense_single_volume/`
  uses, produced by
  `recovar.core.fourier_transform_utils.get_dft3_real` and
  inverted by `get_idft3_real`. Any complex array of the right
  rfft shape decodes to a real-valued volume by construction.
- Slicing: use
  `recovar.core.slicing.slice_volume(..., half_volume=True,
  half_image=True)` (`recovar/core/slicing.py:222`). Backprojection:
  `adjoint_slice_volume(..., half_volume=True, half_image=True)`
  (`recovar/core/slicing.py:331`).
- "Orthonormal rows of `U`" means orthonormal under the
  **real-space** inner product on the decoded real volumes, NOT
  the bare complex inner product on the half-spectrum vectors.
  Parseval gives the relationship: with rfft Hermitian weights
  `w(k)` (2 for interior packed-axis columns, 1 for DC and
  Nyquist columns), real-space inner product
  `<a, b>_real = (1/N_full) · Σ_k w(k) Re[conj(a_half) b_half]`.
  Real-space orthonormality is therefore
  `Re[(U * w) @ U^H] / N_full = I_q`.
- The gauge group is real orthogonal `O(q)`. The gauge fix is
  Cholesky-whitening of the weighted Gram, implemented in
  `recovar/em/ppca_abinitio/half_volume.py:real_volume_orthonormalize_half`.

Stored shapes:

- `μ` ∈ `(half_volume_size,)`, complex128, rfft-packed.
- `U` ∈ `(q, half_volume_size)`, complex128, rfft-packed; rows
  real-space orthonormal under the weighted inner product.
- `s` ∈ `(q,)`, non-negative, descending, **real**.
- The full real-space `volume_shape = (N0, N1, N2)` is carried as
  a static field on `PPCAInit`, since the half size alone does
  not determine the full size unambiguously when `N2` is odd.

### 4.3 FFT noise scale contract

RECOVAR stores Fourier images in the **native unnormalized** FFT
convention (see `recovar/reconstruction/noise.py:797-800`). This
means a real-space image with per-pixel variance `σ²` has Fourier
coefficients with per-pixel variance `σ² · prod(image_shape)`.

The synthetic harness must specify which units `noise_variance`
lives in. v0 specifies it as **Fourier-space variance per pixel**,
matching the convention used by `compute_little_H_b`. Real-space
white noise of per-pixel variance `σ²_real` is converted to the
synthetic `noise_variance` array as
`noise_variance[k] = σ²_real · prod(image_shape)` (constant in `k`
for white noise). This is pinned by
`tests/ppca_abinitio/test_fft_noise_scale_contract.py`.

### 4.4 Pose-marginalized score

```
p(y_i)  =  Σ_{g ∈ G}  π_g  N(y_i ;  CTF_i · A_g μ ,  Σ_i + (CTF_i · A_g U) diag(s) (CTF_i · A_g U)^H ).
```

For v0, `π_g = |G|^{-1}` (uniform pose prior). The plumbing for a
non-uniform prior exists in the API (`FixedGridSpec.log_prior`)
even though v0 leaves it flat.

### 4.5 Latent posterior

For each `(i, g)`:

```
H_{i,r}   =  diag(1/s)  +  (CTF_i A_r U)^H Σ_i^{-1} (CTF_i A_r U)     ∈ R^{q×q}
b_{i,g}   =  (CTF_i A_r U)^H Σ_i^{-1} (S_t y_i  −  CTF_i A_r μ)        ∈ R^{q}
m_{i,g}   =  H_{i,r}^{-1} b_{i,g}                                       ∈ R^{q}
```

`b` and `m` are real-valued because all inputs are Hermitian-
symmetric and the inner product of two Hermitian-symmetric vectors
is real. (This is a precondition the implementation must verify, not
an assumption it can rely on.)

### 4.6 Translation-independence of `H`

`H_{i,r}` does not depend on the translation index `t`. Translation
acts as a phase factor `exp(-i 2π k·t)` on the image, which is
absorbed into `b` but cancels in `(CTF·U)^H Σ^{-1} (CTF·U)`. This
is exactly how `compute_little_H_b` (`recovar/em/heterogeneity.py:44`)
is written, and it is the reason the small `(q × q)` Cholesky cost
is amortized across translations. The new helper must preserve this
structure: storing `H_{i,r,t}` would inflate cost by `n_trans`.

**Caveat.** The translation-independence of `H` is valid only when
`Σ_i` is diagonal in Fourier space and translation-invariant. v0
explicitly disables image-space soft masks during PPCA scoring
(unlike the parity branch's E-step). Re-enabling them in a later
phase requires either (a) restoring the translation-dependent `H`
or (b) developing a mask-aware score path.

---

## 5. Fixed-grid v0 — design contract

### 5.1 Discretization

- Rotations: HEALPix order **2** only in v0. Order 3+ is Phase 4.
- Translations: integer grid, `max_shift ∈ {1, 2}` pixels, step `1`.
- The **matched-grid** synthetic family generates data on the same
  grid that inference uses. The **misspecified** family does not
  (Section 9.3).

### 5.2 Frozen quantities (v0)

`Σ_i` (noise variance), regularization strength on `μ`, latent
dimensionality `q`, the grid `G`, the interpolation type
(`linear_interp`), the support mask, the spectrum `s`, and dtype
(`complex128` / `float64`) are all **fixed** for the entire run.
The loop must not mutate any of them.

### 5.3 Self-contained loop

The v0 loop is a fresh, ~150-line orchestrator in
`recovar/em/ppca_abinitio/loop.py`. It does **not** call
`E_M_batches_2` or `split_E_M_v2`. Per-iteration accumulators are
**explicitly zeroed** at the top of each iteration. No
cross-iteration hidden state beyond `(μ, U, s)` and a small metrics
dict written to JSON.

The mean update path **must** call the same post-processing solve
as the homogeneous baseline:

```python
Ft_y, Ft_ctf = M_with_precompute(...)
mu_next = relion_functions.post_process_from_filter(
    dataset, Ft_ctf, Ft_y, tau=mean_variance, disc_type=disc_type
).reshape(-1)
```

Otherwise the comparison "PPCA loop vs homogeneous loop" measures
the post-processing solve, not PPCA.

---

## 6. Module layout

```
recovar/em/ppca_abinitio/
    __init__.py
    types.py            # PPCAInit (half-volume layout), FixedGridSpec, PPCAConfig, PosteriorStats, PosteriorBlock
    half_volume.py      # rfft Hermitian weights, radial band-limit, real_volume_orthonormalize_half, decoders
    grid.py             # build_fixed_grid (order 2 only in v0)
    posterior.py        # score_and_posterior_moments_eqx + streaming block iterator (half-volume / half-image)
    synthetic.py        # 5 synthetic families (Section 9.3)
    metrics.py          # hidden-state, mean (oracle_fsc_gt), subspace, embedding, optimization
    init.py             # init_truth_perturbed, init_random_lowpass, init_from_external_mean
    factor_update.py    # U-only updates with weighted real-space orthonormalization (Section 8.3)
    mean_update.py      # residualized mean update calling post_process_from_filter
    loop.py             # run_score_diagnostic, run_fixed_grid_ppca
    atlas.py            # K-class volume alignment + atlas PCA (Phase 3)
    relion_io.py        # thin wrappers over recovar.utils.helpers for K-class import
```

`half_volume.py` replaces the earlier `real_volume.py` design (full
volume + Hermitian projection), per Section 0.3.1.

```
tests/ppca_abinitio/
    test_compute_bHb_terms_correctness.py        # already committed (audit P1)
    test_compute_bHb_terms_dtype.py              # already committed (audit P2)
    test_score_matches_e_step_residual_ref.py    # production-path parity
    test_fft_noise_scale_contract.py             # FFT unit pin
    test_half_volume.py                          # rfft weights, weighted Gram orthonormalization, decoders
    test_posterior_brute_force.py                # m, Hinv, log_scores vs dense reference
    test_posterior_calibration.py                # 90% ellipsoid coverage at true pose
    test_posterior_real_valued.py                # post_mean is real for real-volume inputs
```

```
scripts/ppca_abinitio/
    run_stage_0b_oracle_score.py
    run_stage_1a_factor_perturbation.py
    run_stage_1b_residualized_mean.py
    run_stage_1c_factor_learning.py
    run_phase_2_external_mean_bootstrap.py
    run_phase_3_atlas_bootstrap.py
```

Stage gates are scripts that emit JSON, not pytest items. Pytest is
for math and invariants only; this matches the marker discipline in
`tests/CLAUDE.md`.

---

## 7. Posterior helper — first new math object

This is the central new function.

### 7.1 Signature (streaming)

```python
def score_and_posterior_moments_eqx(
    config: ForwardModelConfig,
    mean_projections,   # (n_rot, image_size)             complex128, Hermitian-symmetric
    u_projections,      # (n_rot, q, image_size)          complex128, Hermitian-symmetric
    s,                  # (q,)                            float64, > 0
    batch,              # (n_img, image_size)             complex128, Hermitian-symmetric
    translations,       # (n_trans, 2)                    float64
    ctf_params,
    noise_variance,     # (image_size,)                   float64, in Fourier units
    *,
    rot_block_size: int | None = None,
    trans_block_size: int | None = None,
) -> PosteriorStats:
    ...
```

`score_and_posterior_moments_eqx` is the **fully-materialized**
form: it returns `PosteriorStats` with full `(n_img, n_rot,
n_trans)` `log_scores` and `log_resp`, full `(n_img, n_rot, n_trans,
q)` `post_mean`, and full `(n_img, n_rot, q, q)` `post_Hinv`. It is
intended only for tiny CPU tests and Stage 0A correctness checks.

For real workloads, use:

```python
def iter_posterior_blocks(
    config, mean_projections, u_projections, s, batch, translations,
    ctf_params, noise_variance, *, rot_block_size, trans_block_size,
) -> Iterator[PosteriorBlock]:
    ...
```

which yields `(rot_block, trans_block, log_scores_block,
post_mean_block, post_Hinv_block)` per `(rot_block, trans_block)`
pair. M-step accumulators consume the block iterator and never
materialize the full posterior tensor. This mirrors
`recovar/em/dense_single_volume/engine_v2.py:500-540`.

### 7.2 Returned dataclasses

```python
@dataclass
class PosteriorStats:
    log_scores:  jnp.ndarray  # (n_img, n_rot, n_trans)   float64
    log_resp:    jnp.ndarray  # (n_img, n_rot, n_trans)   float64, image-normalized
    post_mean:   jnp.ndarray  # (n_img, n_rot, n_trans, q) float64
    post_Hinv:   jnp.ndarray  # (n_img, n_rot, q, q)       float64

@dataclass
class PosteriorBlock:
    rot_slice:   slice
    trans_slice: slice
    log_scores:  jnp.ndarray  # (n_img, len(rot_slice), len(trans_slice))
    post_mean:   jnp.ndarray  # (n_img, len(rot_slice), len(trans_slice), q)
    post_Hinv:   jnp.ndarray  # (n_img, len(rot_slice), q, q)
```

`post_Hinv` is translation-independent within each rotation block
(see Section 4.6), so its `n_trans` axis is dropped on purpose.
`C = Hinv + m m^T` is formed on the fly inside the M-step where
needed; v0 never holds the full second-moment tensor.

### 7.3 Internal implementation

1. Build `mean_projections` and `u_projections` once per outer
   iteration (the loop, not the helper, owns this — same pattern
   as `E_with_precompute`).
2. Form `H_{i,r} = diag(1/s) + U_r^H Σ^{-1} U_r` per `(i, r)` —
   reuse `compute_UPLambdainvPU` (`recovar/em/heterogeneity.py:24`)
   by import. Do not duplicate the math.
3. Form `b_{i,r,t}` per `(i, r, t)` — reuse
   `compute_bLambdainvPU_terms` (`recovar/em/heterogeneity.py:127`)
   by import.
4. Solve `m = H^{-1} b` and `bHinvb = b^H m` via a single Cholesky
   per `(i, r)`, vmapped across the batch. Same structure as
   `compute_bHb_terms` (`recovar/em/heterogeneity.py:84`).
5. Build `log_scores` from `bHinvb` and `log det H` plus the
   homogeneous-residual term so that the result is the **full**
   marginal `log p(y_i | g)` up to constants that do not depend
   on `g`.
6. Normalize within each image to obtain `log_resp`.

The streaming iterator does the same work but materializes only
one `(rot_block, trans_block)` slice at a time and yields it.

### 7.4 Mandatory tests

| Test | Asserts |
|---|---|
| `test_posterior_brute_force.py` | For `q ≤ 3`, `image_size ≤ 32`, the helper's `log_scores`, `m`, and `Hinv` agree with a dense Σ_y reference computed in float64 numpy to `rtol=1e-10`. |
| `test_score_matches_e_step_residual_ref.py` | The helper's `log_scores` (float64) match the score *actually* assembled inside `E_with_precompute` from `compute_dot_products_eqx`, `compute_CTFed_proj_norms_eqx`, and `compute_bHb_terms`, after subtracting per-image constants. `rtol=1e-6`. |
| `test_posterior_calibration.py` | For `q=2`, `n_img=256`, synthetic data drawn from the model, the empirical 90% ellipsoid coverage of `(α_true - m)^T H (α_true - m)` lies in `[0.85, 0.95]` at the **true** pose. This is the only test that proves `m` and `Hinv` are real posteriors, not just Cholesky outputs that happen to give a correct score. |
| `test_posterior_real_valued.py` | For inputs derived from real-space volumes, `max(abs(imag(post_mean))) < 1e-10`. This verifies the Hermitian-symmetry chain holds end-to-end. |

If any of these fail, do not run any stage experiments.

---

## 8. M-step ladder

### 8.1 Parameter discipline

| Stage | μ | U | s | Method |
|---|---|---|---|---|
| 0A | ✗ | ✗ | ✗ | posterior helper correctness only |
| 0B | ✗ | ✗ | ✗ | oracle score diagnostic, true `(μ, U, s)` |
| 1A | ✗ | ✗ | ✗ | non-oracle score stress test |
| 1B (ablation) | ✓ | ✗ | ✗ | residualized mean only, fixed `W` |
| 1C | ✓ | ✓ (gradient) | ✗ | factor learning, `s` strictly fixed |
| 1D | ✓ | ✓ (soft M-step) | ✗ | full second moments, ECM |
| Phase 2 | ✓ | ✓ | ✗ | external homogeneous mean bootstrap |
| Phase 3 | ✓ | ✓ | possibly ✓ | atlas-PCA bootstrap |

`s` is **literally constant** through 1C. Per Q2, the
re-SVD-and-rebind pattern from the previous draft is dropped: it is
not "frozen `s`". Stage 1D is where `s` becomes free.

### 8.2 Mean update

#### 8.2.1 Mean update v0 — debugging ablation only

Unresidualized mean update (PPCA responsibilities, homogeneous
backprojection) is implementable as a debugging knob in
`mean_update.py`, but **it is not a stage gate**. Per Q1, Stage 1B
graduates only on the residualized form.

#### 8.2.2 Mean update v1 — residualized (Stage 1B and onward)

Backproject the **residualized** image:

```
y_i^res(g) = y_i - CTF_i · A_g · U · m_{i,g}
```

The residual is computed inside the per-batch closure that consumes
`iter_posterior_blocks` so we never hold all `(i, r, t)` residuals
in memory.

The post-processing solve is:

```python
Ft_y, Ft_ctf = M_with_precompute(... feeding y_i^res(g) ...)
mu_next = relion_functions.post_process_from_filter(
    dataset, Ft_ctf, Ft_y, tau=mean_variance, disc_type=disc_type
).reshape(-1)
# In half-volume layout there is no Hermitian-projection step.
# The post-processing solve is performed in whichever layout the
# rest of the parity branch uses; convert to half-volume via
# full_volume_to_half_volume before storing back into PPCAInit.mu.
mu_next_half = ftu.full_volume_to_half_volume(mu_next, dataset.volume_shape)
```

The PPCA loop and the homogeneous loop being compared **must use
the same `tau` and the same `disc_type`** so the comparison is
PPCA vs no-PPCA, not configuration-vs-configuration.

### 8.3 Factor update (Stage 1C, real-volume parameterization)

#### 8.3.1 What is being updated

Only `U`. `s` is constant. `μ` may continue to update via
Section 8.2.2 in the same outer iteration.

#### 8.3.2 Objective

With `μ` and `s` held fixed, take K (≈3) gradient steps on the
expected complete-data negative log-likelihood w.r.t. `U`, using
the posterior moments `(γ, m, Hinv)` from the current E-step:

```
L(U) = Σ_{i,g} γ_{i,g} ·
        E_{α | y_i, g} [ ‖ y_i − CTF_i A_g μ − CTF_i A_g U α ‖_{Σ_i^{-1}}^2 ]
       + λ · ‖ U ‖_{prior}^2
```

with `E[α α^T] = Hinv + m m^T` (real, not complex). `λ` is the
ridge constant fixed in Section 8.3.4.

**Implementation rule.** Use `jax.value_and_grad` over a small
JIT'd closure that takes `U` and returns scalar `L`. Do not
hand-derive the gradient.

**Memory rule.** The closure consumes posterior **blocks**
(`PosteriorBlock`), not full posterior tensors. The accumulator
adds block contributions into running `(volume_size, q)` arrays.

#### 8.3.3 The update step (half-volume layout)

```python
U_raw  = U - lr * grad_U                                          # free gradient step (half-volume)
U_band = radial_band_limit_half(U_raw, volume_shape, k_max)       # optional low-pass mask
U      = real_volume_orthonormalize_half(U_band, weights, N_full) # weighted O(q) gauge fix
```

There is **no Hermitian-projection step** (per Section 0.3.1).
With `U` stored in half-volume rfft layout, the redundant half of
the spectrum is not represented at all, so it cannot be violated.
Any complex array of the right rfft shape decodes via
`get_idft3_real` to a real volume by construction.

The remaining steps:

1. **Gradient step** is on the half-spectrum complex array. The
   gradient itself comes from autodiff over a closure that
   consumes `(U_half, posterior_block)` and returns the
   complete-data NLL. Autodiff respects the half-volume
   parameterization.
2. **`radial_band_limit_half`** zeroes voxels with radial
   frequency above `k_max` to prevent `U` from learning
   high-frequency noise where `Σ^{-1}` is small and `H` is
   ill-conditioned. v0 uses `k_max = grid_size // 4`. The radial
   index respects the half-volume layout (centered `kz, ky`,
   packed `kx`).
3. **`real_volume_orthonormalize_half`** is Cholesky-whitening of
   the rfft-weighted Gram `Re[(U * w) @ U^H] / N_full`, where
   `w = make_half_volume_weights(volume_shape)`. It is the
   correct real `O(q)` gauge fix and is **not** a complex thin
   SVD.

After this chain, the rows of `U`:
- correspond to real 3D volumes by layout (no representation
  outside the Hermitian subspace);
- have no energy above `k_max`;
- are orthonormal in the real-space inner product on the decoded
  volumes.

**Mandatory test** (`tests/ppca_abinitio/test_half_volume.py`,
already committed):
- `test_orthonormalized_rows_decode_to_orthonormal_real_volumes`
  decodes `U_orth` via `get_idft3_real` and verifies the resulting
  real volumes are row-orthonormal under the standard real-space
  inner product.
- `test_weighted_half_inner_product_equals_full_inner_product`
  pins the rfft Hermitian weights against the full-spectrum
  reference.

#### 8.3.4 Ridge constant

`λ = 1e-4 · trace(Σ^{-1})`. This is **not** an open question; it is
required for numerical stability in the ill-conditioned regime
discussed in Audit 2 and is set to a value small enough to be
inert in the well-conditioned regime.

#### 8.3.5 W update v1 — full soft M-step (Stage 1D)

Closed-form weighted ridge solve using full second moments
`C = Hinv + m m^T`. Implement only after Stage 1C is observed to
behave on the synthetic harness (Section 11.5).

---

## 9. Synthetic data harness

`recovar/em/ppca_abinitio/synthetic.py`.

### 9.1 Core API

```python
def make_synthetic_fixed_grid_dataset(
    family: SyntheticFamily,           # one of A..E (Section 9.3)
    mu_true, U_true, s_true,
    grid: FixedGridSpec,
    n_images_train: int,
    n_images_val: int,
    noise_variance: jnp.ndarray,        # FOURIER units, see Section 4.3
    ctf_distribution: CTFDistribution,
    seed: int,
) -> tuple[CryoEMDataset, GroundTruth, ValSplit]: ...
```

`ValSplit` is a `(train_indices, val_indices)` namedtuple. **All
learning-stage primary metrics must be reported on `val_indices`
only.** Per Q6 this is mandatory in v0.

### 9.2 Ground-truth bundle

```python
@dataclass
class GroundTruth:
    g_true:     np.ndarray  # (n_images,)  combined index into G
    r_true_idx: np.ndarray  # (n_images,)  rotation index
    t_true_idx: np.ndarray  # (n_images,)  translation index
    alpha_true: np.ndarray  # (n_images, q) real latent coords
    contrast_true: np.ndarray | None  # (n_images,) — only set for family D
    mu_true:    np.ndarray
    U_true:     np.ndarray
    s_true:     np.ndarray
```

### 9.3 Five required synthetic families

| ID | Name | Heterogeneity | Pose | Contrast | Purpose |
|---|---|---|---|---|---|
| A | **Null** | `s_true = 0` (or `q_true = 0`) | matched-grid | uniform | Negative control: PPCA must show no gain. |
| B | **Matched-grid heterogeneous** | continuous low-rank | matched-grid | uniform | Primary positive control. |
| C | **Misspecified pose** | as B | rotation jitter 1–2°, translation jitter 0.25–0.5 px **off-grid** | uniform | Stress test: PPCA must not collapse, pose error must not be absorbed into `W`. |
| D | **Per-particle contrast** | as B | matched-grid | `c_i ∈ [0.8, 1.2]`, see `recovar/data_io/cryoem_dataset.py:932-945` | Tests whether the first PC absorbs contrast variation rather than structure. |
| E (optional) | **CTF-zero-localized heterogeneity** | low-rank energy concentrated near first CTF zero shell for a subset of particles | matched-grid | uniform | Tests CTF-zero unidentifiability. May be deferred. |

Stages 0B, 1A, 1B, 1C are gated against **A and B at minimum**.
Stage 1C must additionally pass on C (no-collapse) and D (no-
contrast-absorption). Family E is reported but not gating in v0.

### 9.4 Ground-truth source

Hand-built low-resolution `(μ, U)` is sufficient (e.g. a small
Gaussian blob mean plus 2–3 spatially localized PCs constructed in
real space and then FT'd through the canonical helpers). Do **not**
construct `U` directly in Fourier space — that bypasses the real-
volume invariant.

### 9.5 Initialization controls (`init.py`)

- `init_truth_perturbed(gt, eps_mu, eps_U)` — Stage 1B/1C positive
  control. Perturbations are added in real space and re-encoded
  via `real_volume_to_half`, so the half-volume layout's structural
  Hermitian symmetry is preserved automatically.
- `init_random_lowpass(volume_shape, q, k_max, seed)` — stress
  control. Generates real-space volumes, FTs them, band-limits.
- `init_from_external_mean(mu_path, q, k_max, seed)` — Phase 2.
  Pose convention is handled by `relion_io.py`.
- `init_from_aligned_class_atlas(volumes, weights, q)` — Phase 3.

---

## 10. Metrics

`recovar/em/ppca_abinitio/metrics.py`. Metrics are grouped by
purpose, reported separately, and subject to the rule:
**every stage has exactly one pre-registered primary metric** that
decides graduation. Other metrics are reported as context, not
gates.

### 10.1 Hidden-state / alignment

- `top1_acc` — fraction of images whose argmax `g` equals `g_true`.
- **`true_state_mass`** — mean of `γ_{i, g_true(i)}`. Primary
  metric for score stages.
- `true_state_rank`.
- `angular_error_deg` (uses `_angular_distance_matrices`,
  `recovar/em/sampling.py:409`).
- `translation_error_px`.

### 10.2 Mean

- **`oracle_fsc_gt(mu_est, mu_true)`** — explicitly named to
  distinguish from gold-standard split-map FSC. **Do not use
  half-bit / 0.143 thresholds**; they are valid only for split-map
  FSC, not oracle FSC against ground truth.
- **`fourier_relative_error(mu_est, mu_true)`** — primary metric
  for mean stages: `‖μ_est − μ_true‖ / ‖μ_true‖` in Fourier norm.
- Shell-averaged correlation curve (reported but not gating).

### 10.3 Subspace

- Principal angles between `span(U_est)` and `span(U_true)`.
- **`projector_frobenius_error`** — `‖P_{U_est} − P_{U_true}‖_F`.
  Primary metric for factor stages. Gauge-invariant.

### 10.4 Spectrum (only after Stage 1D)

- Relative error of `s_est`.
- Total low-rank variance ratio.

### 10.5 Embedding

- **Oracle embedding error.** Use `m_{i, g_true(i)}` and compare
  to `α_true_i` after orthogonal Procrustes (real-orthogonal, not
  complex unitary). Isolates the latent posterior from pose errors.
- **Marginal embedding error.** Use `α̂_i = Σ_g γ_{i,g} m_{i,g}`
  and compare to `α_true_i` after orthogonal Procrustes.

### 10.6 Optimization

- Mean log-likelihood per image, per iteration.
- Posterior entropy over poses.
- For Stage 1C: monotonicity flag for the GEM objective (allowed
  relative slack `1e-3`).

### 10.7 Reporting rules

- Report the primary metric first, with mean and 95% bootstrap CI
  over images, for each seed.
- Report secondary metrics, but they do not decide stage
  graduation.
- Any FSC against ground truth is labeled `oracle_fsc_gt`. No
  half-bit / 0.143 language until actual half maps exist.
- Validation-only metrics are required for all learning stages
  (Stage 1B, 1C, 1D, Phase 2, Phase 3).

---

## 11. Staging plan and exit criteria

The loop must not advance to the next stage until the previous one
satisfies its exit criterion on the synthetic harness. For every
stage:

- a fixed train/validation split of images;
- at least 3 RNG seeds;
- one heterogeneous synthetic family (B by default) and one
  homogeneous-null family (A);
- one pre-registered primary metric;
- the same metric reported on validation only for any stage that
  learns parameters from the train split.

### 11.0 Common evaluation protocol

**Synthetic families.** A (null), B (matched-grid heterogeneous),
C (misspecified pose), D (per-particle contrast). See Section 9.3.

**Primary metrics by stage.**

- Score stages (0B, 1A): validation `true_state_mass`.
- Mean stages (1B): validation `fourier_relative_error(μ)`.
- Factor stages (1C, Phase 2, Phase 3): validation
  `projector_frobenius_error`.

**Reporting rules.** See Section 10.7.

### 11.1 Stage 0A — posterior helper correctness

**Implement.** `score_and_posterior_moments_eqx` and
`iter_posterior_blocks` in `posterior.py`.

**Required tests** (all four mandatory):

1. `test_posterior_brute_force.py` — brute-force parity for
   `log_scores`, `m`, and `Hinv`.
2. `test_score_matches_e_step_residual_ref.py` — production-score
   parity, against the assembly inside `E_with_precompute`, **not**
   against dead `_eqx` code.
3. `test_posterior_calibration.py` — 90% ellipsoid coverage in
   `[0.85, 0.95]`.
4. `test_posterior_real_valued.py` — `imag(post_mean) < 1e-10`
   for real-volume inputs.

**Exit criterion.** All four tests pass. If any fail, do not run
stage experiments.

### 11.2 Stage 0B — oracle-score falsification

**Implement.** `run_score_diagnostic(...)` with no parameter
updates. Use the *true* `(μ, U, s)` for the PPCA branch and the
same `μ_true` with `u=None` for the homogeneous branch.

**Primary metric.** Validation `true_state_mass`.

**Exit criterion.** All of:

1. On family **B**, PPCA improves validation `true_state_mass`
   over homogeneous by an amount whose 95% bootstrap CI excludes
   zero, for **all 3 seeds**.
2. On family **A** (null), the absolute change in validation
   `true_state_mass` is `≤ 0.01` for all 3 seeds.
3. On family **C** (misspecified pose), the PPCA advantage is
   reduced but its sign does not flip and its magnitude is at
   least 25% of the family-B advantage on the same seed.

If this stage fails, **stop the project**. There is no reason to
implement a PPCA M-step if oracle factors do not help the score
under modest model mismatch.

### 11.3 Stage 1A — non-oracle score stress test

**Implement.** Re-run `run_score_diagnostic(...)` with two
factor initializations:

- truth-perturbed `U` (positive control),
- random-lowpass `U` (negative control / stress test).

**Primary metric.** Validation `true_state_mass`.

**Exit criterion.**

- Truth-perturbed init must preserve a positive PPCA-over-
  homogeneous score gain on family B with 95% bootstrap CI
  excluding zero.
- Random-lowpass init is **not** a graduation requirement. It is
  reported. Failure here is informative but does not block
  progress; success here does not mean the bootstrap problem is
  solved.

### 11.4 Stage 1B — residualized mean-only loop (graduation gate)

**Implement.** `run_fixed_grid_ppca(..., update_mu=True,
update_factor=False)` using the **residualized** mean update
(Section 8.2.2) and the same `post_process_from_filter` solve as
the homogeneous baseline.

**Primary metric.** Validation `fourier_relative_error(μ)`.

**Exit criterion.** All of:

1. From truth-perturbed init on family B, the PPCA loop improves
   the primary metric over the homogeneous loop after 8 iterations,
   for all 3 seeds.
2. Validation `true_state_mass` at the final iteration is not
   worse than the initialization by more than 0.01 absolute.
3. On family **A**, PPCA is not better than homogeneous by more
   than noise. If it is, the residualization is explaining
   nuisance structure rather than heterogeneity, and 1B fails.

The unresidualized v0 mean update is allowed as a debugging
ablation in `mean_update.py` but is **not** a graduation gate (Q1).

### 11.5 Stage 1C — fixed-spectrum factor learning

**Implement.** `run_fixed_grid_ppca(..., update_mu=True,
update_factor=True, update_s=False)` with the U-only real-volume
update chain from Section 8.3.3 and ridge constant from 8.3.4.
`s` is **literally constant** (Q2).

**Primary metric.** Validation `projector_frobenius_error`.

**Required baseline.** `HeterogeneousEMState` on the same
`(family, seed, init)` triple, reported alongside the PPCA loop
for every primary and secondary metric (Q3).

**Exit criterion.** All of:

1. Final `projector_frobenius_error` improves over the
   initialization for all 3 seeds on family B.
2. Oracle embedding error improves over the initialization for
   all 3 seeds on family B.
3. Validation `true_state_mass` does not regress relative to
   Stage 1B by more than 0.01 absolute.
4. No NaN/Inf is produced; the factor-update projection chain
   preserves the real-volume invariant (`imag_energy_fraction <
   1e-8` after every step).
5. The train-side generalized-EM objective is non-decreasing up
   to relative slack `1e-3`.
6. On family **A**, none of (1)–(3) shows PPCA-over-homogeneous
   gain — i.e. the loop is not learning structure on null data.
7. On family **C**, the factor learns the same subspace as on
   family B up to a tolerance of `0.1` in projector Frobenius
   error. Pose jitter must not be absorbed into `W`.
8. On family **D**, the first PC's overlap with the contrast
   direction in image space is `< 0.3`. Contrast variation must
   not be absorbed into `U`.
9. PPCA loop matches or beats `HeterogeneousEMState` on the
   primary metric on at least family B for all 3 seeds.

Random-lowpass init remains a stress test only.

### 11.6 Stage 1D — full soft M-step

Implement only after Stage 1C is stable across the 3 seeds and
all 9 sub-criteria above. Before enabling 1D, update this spec
with the closed-form solve, the memory plan, the primary metric,
and the null-family behavior expected from the full second-moment
path.

### 11.7 Phase 2 — external-mean bootstrap

**Goal.** Test basin-of-attraction reachability from a non-oracle
mean.

**Initialization order** (lowest convention risk first):

1. homogeneous RECOVAR mean (no axis flip, no Euler conversion);
2. RELION mean converted with `load_relion_volume`;
3. cryoSPARC mean — **not in v0** (Q7).

Random-lowpass `U` is a negative-control stress test only, never
the main initializer.

**Pose-convention guard.** Any externally produced volume must be
brought into recovar convention via
`recovar/utils/helpers.load_relion_volume` or
`relion_volume_to_recovar` **before** PCA, FSC, or projection. Do
not re-derive this transform — it is pinned by
`tests/unit/test_relion_volume_convention.py` and explained in
`recovar/em/CLAUDE.md`.

**Primary metric.** Final validation `projector_frobenius_error`.

**Required baseline.** `HeterogeneousEMState` from the same
external mean.

**Exit criterion.** Starting from the external mean and a non-
oracle factor init, the final projector error and final mean
error are within a pre-declared tolerance band of the Stage 1C
truth-perturbed result on the same synthetic family, **and** the
direct PPCA result matches or beats the `HeterogeneousEMState`
result on at least one of `{projector_error, mean_error}`. If
random-lowpass fails, document the failure; do not rebrand it as
a partial success.

### 11.8 Phase 3 — K-class atlas bootstrap

**Pipeline.**

1. Run an external `K`-class ab-initio (RELION).
2. Convert every class volume to recovar frame via
   `load_relion_volume`.
3. Align each class volume with an **explicit rotational search
   and a handedness check**. Rigid Procrustes on voxel vectors is
   not sufficient; alignment is its own non-trivial 3D-3D
   registration problem.
4. Normalize shell-wise amplitude before PCA.
5. Form atlas mean `μ_0 = Σ_k π_k V_k` and deviations
   `D_k = V_k − μ_0`.
6. Use at most `K - 1` atlas-derived directions as claimed atlas
   PCs. If `q > K - 1`, the extra directions are auxiliary random
   directions and **must be labeled as such** in metrics output.
7. Run the Stage 1C loop from `(μ_0, U_0, s_0)`.

**Primary metric.** Final validation `projector_frobenius_error`
relative to the atlas initializer.

**Required baseline.** `HeterogeneousEMState` from the same
atlas init.

**Exit criterion.** The PPCA loop improves both `projector_error`
and `fourier_relative_error(μ)` relative to the raw atlas
initializer, without destabilizing `true_state_mass`. Do not
require a vague "found a direction outside the atlas span"
proof; require a measurable improvement over the atlas itself.

### 11.9 Phase 4 — dataset sweep + faster E-step

Out of scope for v0. Tracked here for completeness:

- HEALPix order 3+, requires `iter_posterior_blocks` to scale
  beyond CPU-only sizes.
- CryoBench sweep on the cluster path provided by the user.
- Coarse-to-fine pose search using
  `get_local_rotation_grid_fast` and
  `get_oversampled_translation_grid`.

---

## 12. Concrete API

```python
# types.py

@dataclass
class PPCAInit:
    mu: jnp.ndarray   # (half_volume_size,) complex128, rfft-packed half-volume
    U:  jnp.ndarray   # (q, half_volume_size) complex128, rfft-packed half-volume, real-orthonormal rows
    s:  jnp.ndarray   # (q,) float64, descending
    volume_shape: tuple  # (N0, N1, N2) — full real-space volume shape, static field

@dataclass
class FixedGridSpec:
    rotations:        jnp.ndarray   # (n_rot, 3, 3) float64
    translations:     jnp.ndarray   # (n_trans, 2) float64
    log_prior_rot:    jnp.ndarray | None = None  # (n_rot,) or None for uniform
    log_prior_trans:  jnp.ndarray | None = None  # (n_trans,) or None for uniform

@dataclass
class PPCAConfig:
    n_iters:             int
    update_mu:           bool = True
    update_factor:       bool = False
    update_s:            bool = False  # forbidden < 1D
    factor_inner_steps:  int = 3
    factor_lr:           float = 1e-2
    ridge_lambda:        float = 1e-4  # see Section 8.3.4
    rot_block_size:      int = 256
    trans_block_size:    int = 16
    seed:                int = 0

@dataclass
class PosteriorStats:
    log_scores:  jnp.ndarray  # (n_img, n_rot, n_trans) float64
    log_resp:    jnp.ndarray  # (n_img, n_rot, n_trans) float64
    post_mean:   jnp.ndarray  # (n_img, n_rot, n_trans, q) float64
    post_Hinv:   jnp.ndarray  # (n_img, n_rot, q, q) float64

@dataclass
class PosteriorBlock:
    rot_slice:   slice
    trans_slice: slice
    log_scores:  jnp.ndarray
    post_mean:   jnp.ndarray
    post_Hinv:   jnp.ndarray

# half_volume.py
def make_half_volume_weights(volume_shape) -> jnp.ndarray: ...
def half_volume_radial_index(volume_shape) -> jnp.ndarray: ...
def radial_band_limit_half(v_flat_half, volume_shape, k_max) -> jnp.ndarray: ...
def real_volume_orthonormalize_half(U_flat_half, weights, volume_size, *, ridge=1e-12) -> jnp.ndarray: ...
def half_real_space_gram(U_flat_half, weights, volume_size) -> jnp.ndarray: ...
def half_to_real_volume(v_flat_half, volume_shape) -> jnp.ndarray: ...
def real_volume_to_half(real_vol, volume_shape) -> jnp.ndarray: ...

# grid.py
def build_fixed_grid(healpix_order: int, max_shift: int, shift_step: int = 1) -> FixedGridSpec: ...

# synthetic.py
class SyntheticFamily(Enum):
    NULL = "A"
    MATCHED_GRID_HET = "B"
    MISSPECIFIED_POSE = "C"
    PER_PARTICLE_CONTRAST = "D"
    CTF_ZERO_HET = "E"

def make_synthetic_fixed_grid_dataset(
    family: SyntheticFamily,
    mu_true, U_true, s_true,
    grid: FixedGridSpec,
    n_images_train: int, n_images_val: int,
    noise_variance, ctf_distribution, seed: int,
) -> tuple[CryoEMDataset, GroundTruth, ValSplit]: ...

# posterior.py
def score_and_posterior_moments_eqx(
    config, mean_projections, u_projections, s, batch, translations,
    ctf_params, noise_variance,
) -> PosteriorStats: ...

def iter_posterior_blocks(
    config, mean_projections, u_projections, s, batch, translations,
    ctf_params, noise_variance, *, rot_block_size: int, trans_block_size: int,
) -> Iterator[PosteriorBlock]: ...

# loop.py
def run_score_diagnostic(
    dataset, init: PPCAInit, grid: FixedGridSpec, gt: GroundTruth,
    val_split: ValSplit,
) -> PPCAMetrics: ...

def run_fixed_grid_ppca(
    dataset, init: PPCAInit, grid: FixedGridSpec, cfg: PPCAConfig,
    gt: GroundTruth | None = None, val_split: ValSplit | None = None,
) -> tuple[PPCAInit, list[PPCAMetrics]]: ...

# init.py
def init_truth_perturbed(gt, eps_mu: float, eps_U: float) -> PPCAInit: ...
def init_random_lowpass(volume_shape, q: int, k_max: int, seed: int) -> PPCAInit: ...
def init_from_external_mean(mu_path: str, q: int, k_max: int, seed: int) -> PPCAInit: ...
def init_from_aligned_class_atlas(volumes, weights, q: int) -> PPCAInit: ...
```

---

## 13. Test plan

### 13.1 Unit tests (`tests/ppca_abinitio/`, `unit` marker)

These pin math and invariants. They run in `pixi run test-fast`.

| Test | Purpose | Status |
|---|---|---|
| `test_compute_bHb_terms_correctness.py` | Audit P1: brute-force parity for `compute_bHb_terms` to `rtol=1e-10`. | committed |
| `test_compute_bHb_terms_dtype.py` | Audit P2: float64 propagation through the existing scorer. | committed |
| `test_score_matches_e_step_residual_ref.py` | Production-score parity: assemble the score from `compute_dot_products_eqx + compute_CTFed_proj_norms_eqx − compute_bHb_terms`, compare to a brute-force reference. Pinned against the actual production assembly. | committed |
| `test_fft_noise_scale_contract.py` | Pin the FFT-unit convention so `noise_variance`, synthetic `σ²`, and learned `s` live on the same scale. | committed |
| `test_half_volume.py` | rfft Hermitian weights match Parseval against full-spectrum, weighted Gram orthonormalization produces row-orthonormal real volumes (verified via `get_idft3_real`), span preservation, ridge-stable on rank-deficient input. | committed |
| `test_posterior_brute_force.py` | New posterior helper agrees with dense `Σ_y` reference for `q ≤ 3`, `image_size ≤ 32`. | TODO |
| `test_posterior_calibration.py` | 90% ellipsoid coverage at the true pose lies in `[0.85, 0.95]`. | TODO |
| `test_posterior_real_valued.py` | `imag(post_mean) < 1e-10` for real-volume inputs. | TODO |

All unit tests run in float64 on CPU and complete in seconds, not
minutes.

### 13.2 Stage gate experiments (`scripts/ppca_abinitio/`)

Stage gates are scripts that emit JSON summaries, not pytest items.
They are run by hand or via Slurm; their output JSON is what
graduates a stage. A lightweight `integration`-marked test may
verify that the script *runs* and produces the expected keys, but
does not re-run the experiment as part of `test-fast`.

| Script | Stage | Inputs | Output JSON keys |
|---|---|---|---|
| `run_stage_0b_oracle_score.py` | 0B | family ∈ {A, B, C}, 3 seeds | `true_state_mass_ppca`, `true_state_mass_homog`, `bootstrap_ci`, per-family-per-seed |
| `run_stage_1a_factor_perturbation.py` | 1A | family B, 3 seeds, init ∈ {truth-perturbed, random-lowpass} | as above |
| `run_stage_1b_residualized_mean.py` | 1B | family ∈ {A, B}, 3 seeds, 8 iterations | per-iter `fourier_relative_error_mu_val`, `true_state_mass_val` |
| `run_stage_1c_factor_learning.py` | 1C | family ∈ {A, B, C, D}, 3 seeds | per-iter `projector_frobenius_error_val`, contrast-overlap, `HeterogeneousEMState` baseline |
| `run_phase_2_external_mean_bootstrap.py` | Phase 2 | external mean source, family B | as 1C |
| `run_phase_3_atlas_bootstrap.py` | Phase 3 | RELION K-class output dir | as 1C plus atlas alignment audit |

---

## 14. Known failure modes and mitigations

1. **Spectrum collapse** (in 1D). Mitigation: 1D is not v0.
2. **Mean absorbs heterogeneity.** Mitigation: Stage 0B with frozen
   `μ` is the canary; if 0B is green but 1B regresses, the mean
   update is the culprit, not the score.
3. **Random `W` destabilizes pose search.** Mitigation: random init
   is a stress test, not a graduation path.
4. **Hidden coupling to existing EM state.** Mitigation: v0 loop
   is self-contained; do not import from `iterations.py` or
   `states.py`.
5. **Premature local refinement.** Mitigation: dense order-2 grid
   only in v0.
6. **RELION pose / volume convention drift.** Mitigation: pinned
   helpers in `recovar/utils/helpers.py`, pinned by
   `tests/unit/test_relion_volume_convention.py`.
7. **Off-grid pose error absorbed into `W`.** Stage 1C exit
   criterion 7 (family C) is the explicit gate.
8. **Per-particle contrast absorbed into the first PC.** Stage 1C
   exit criterion 8 (family D) is the explicit gate.
9. **CTF-zero unidentifiability.** Family E is a reported but
   non-gating probe. Symptom: `U` learns directions whose energy
   is dominated by frequencies near a CTF zero.
10. **Overconfident posteriors on null data.** Family A and the
    posterior calibration test (`test_posterior_calibration.py`)
    are the gates. Symptom: PPCA `true_state_mass` exceeds
    homogeneous on null data.
11. **Class-pose / handedness ambiguity in Phase 3.** Mitigation:
    Phase 3 alignment must include an explicit handedness check;
    rigid Procrustes is not sufficient (Section 11.8).
12. **`U` columns leave the real-volume Fourier subspace under
    autodiff.** Mitigation: half-volume rfft layout makes Hermitian
    symmetry structural — the redundant half is not stored, so it
    cannot be violated. Pinned by `test_half_volume.py`.
13. **Float32 silent downcast.** Mitigation: dtype contract on
    `score_and_posterior_moments_eqx` enforces complex128 / float64
    on entry; pinned by `test_compute_bHb_terms_dtype.py` and the
    new `test_posterior_brute_force.py`.
14. **Image-space soft mask breaks `H` translation-independence.**
    Mitigation: v0 disables the mask in PPCA scoring (Section 4.6
    caveat); re-enabling is a v1 task.

---

## 15. PR / validation policy for v0

### 15.1 Code-validation requirements (per `CLAUDE.md`)

Per `CLAUDE.md`, no PR for this work is "ready" until:

1. `pixi run test-fast` is green, including the new
   `tests/ppca_abinitio/` unit tests.
2. `./scripts/run_tests_parallel.sh long-test` is green and the
   summary log has been read.
3. Quality and performance regression tables (per the format in
   `CLAUDE.md`) are in the PR body.

These rules are inherited unchanged. They cover **code** validation:
"this PR does not break the rest of the repo."

### 15.2 Scientific-claim requirements (additional)

For any PR that claims a stage gate has been passed, the PR body
must additionally include:

1. **Pre-registered primary metric** for that stage (Section
   10.7 / 11.x).
2. **Null-family result** showing no PPCA gain on family A.
3. **Held-out evaluation** — all numbers reported on validation
   indices, not training.
4. **`HeterogeneousEMState` baseline** for any 1C / Phase 2 /
   Phase 3 claim.
5. **Synthetic family explicitly named** (A through E).
6. **Random seeds explicitly named** (≥ 3).
7. **Reference to the section of this doc** that the code
   implements, and the matching exit criterion sub-clause.

A PR that misses any of these for a stage-gate claim is not
"ready" no matter how green the long-test suite is.

---

## 16. Open questions (not yet resolved)

The previous draft's open questions are now resolved in Section
0.2. Genuinely open questions remaining:

- **Phase 3 alignment algorithm.** "Explicit rotational search +
  handedness check" is the requirement. The specific algorithm
  (FFT cross-correlation in real space? gradient descent on a
  Lie-algebra parameterization?) is open and will be settled in
  `atlas.py` when Phase 3 is reached.
- **CTF distribution for synthetic data.** Defocus uniform in
  `[1, 3]` µm with no astigmatism is the placeholder; the final
  choice will be set when `synthetic.py` is committed.
- **Ridge constant numerical floor.** `λ = 1e-4 · trace(Σ^{-1})`
  is the placeholder. The final value will be set when 1C runs
  produce ill-conditioned cases.
- **Stage 1D primary metric.** Will be added when 1C is green.

When any of these is resolved, update this doc in the same PR.
