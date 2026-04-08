# Review of `docs/math/plan_ppca_abinitio_v0.md`

## Headline issues

- The staging logic is not falsifiable enough. Section 11 currently allows progression on weak or tautological wins, uses multiple-comparisons exit criteria, and evaluates on the same synthetic family that leaks the intended answer back into the score. I would not start implementation until the primary metric, null control, and held-out protocol are fixed.
- The planned posterior API is already architecturally wrong for anything beyond toy order-2 runs. Materializing `log_scores`, `log_resp`, `post_mean`, and `post_Hinv` as dense batch tensors locks out significant-weight pruning, local-grid search, and blockwise normalization from day one, even though `recovar/em/dense_single_volume/engine_v2.py:500-545` already shows the right streaming pattern.
- The factor-update story is internally inconsistent. Section 8 says "`s` is frozen", then updates free `W`, then re-SVDs `W` and resets `s ← Σ_w^2` (`docs/math/plan_ppca_abinitio_v0.md:359-390`). That is not "frozen `s`". On top of that, a complex SVD gives a unitary gauge, while the PPCA latent prior is real-Gaussian, so the proposed gauge fix is not even the right invariance class.
- The spec does not define how direct PPCA stays inside RECOVAR's real-volume / Hermitian-valid subspace. `μ` and `U` are stored as full centered Fourier volumes (`docs/math/plan_ppca_abinitio_v0.md:118-123`), but there is no projection step that enforces conjugate symmetry after gradient updates. Without that, the algorithm can optimize objects that are not valid Fourier transforms of real 3D densities.
- The synthetic harness is too friendly and too incomplete. Same-grid poses, no off-grid error, no null-heterogeneity control, no held-out split, no contrast nuisance, and no shell-localized CTF stress means a clean-looking result could still be entirely about the synthetic setup rather than about PPCA helping cryo-EM alignment.
- The mean-stage metric is not honest as written. Section 10.2 uses `fsc(mu_est, mu_true)` together with half-bit / 0.143 thresholds (`docs/math/plan_ppca_abinitio_v0.md:466-469`). Those thresholds are for split-map FSC, not oracle FSC against ground truth. This will overstate progress and conflicts with the repo's gold-standard FSC logic in `recovar/reconstruction/homogeneous.py:65-147` and `recovar/reconstruction/regularization.py:683-745`.
- The spec is too dismissive of the existing heterogeneous path. `HeterogeneousEMState` should not be extended into direct PPCA, but excluding it from the experimental baseline set is a mistake. It is the only existing heterogeneity-learning path in-tree and should be part of the comparison story for the learning stages.

## Per-section critique

### Section 1

- Load-bearing: the author is right to separate score, mean, and bootstrap questions (`docs/math/plan_ppca_abinitio_v0.md:21-35`). That separation is the strongest part of the plan.
- What's wrong: the "bootstrap test" still mixes two very different questions: random-`W` recovery and atlas-assisted initialization (`docs/math/plan_ppca_abinitio_v0.md:30-32`). Those are not one stage. One is a negative-control stress test; the other is a plausible initializer. Also, "half-set splitting" being out of scope (`docs/math/plan_ppca_abinitio_v0.md:41-48`) is acceptable for v0 implementation, but then Section 10 cannot use split-map FSC thresholds as if gold-standard logic existed.
- What to change: rewrite the top-level questions as four separate claims: "oracle score helps", "residualized mean helps", "factor learning helps", and "bootstrap reaches the basin". Move random-`W` to a stress-test lane that cannot graduate the project on its own. Rename any ground-truth-vs-estimate FSC metric to `oracle_fsc_gt` and remove half-bit / 0.143 language until there are actual half maps.

### Section 2

- Load-bearing: keeping the new work isolated under `recovar/em/ppca_abinitio/` and importing shared helpers rather than copying math is correct (`docs/math/plan_ppca_abinitio_v0.md:60-69`).
- What's wrong: the branch ancestry is not pinned even though the parent is a moving target. The current worktree shows the exact merge-base is `3a39533212c7955d59507473544fa821c9b4eb6e`, but the spec only says "off `claude/em-relion-parity`" (`docs/math/plan_ppca_abinitio_v0.md:57-59`). Also, one of the listed companion docs does not exist in this branch: `docs/math/plan_ab_initio_relion_parity.md` (`docs/math/plan_ppca_abinitio_v0.md:7`). Finally, "do not edit existing files" is too rigid for a plan that already depends on imported shared helpers and almost certainly needs a shared kwarg or utility before Phase 4.
- What to change: record the parent commit hash explicitly in the spec. Remove or correct the missing companion-doc path. Replace the "do not edit" rule with "shared-helper edits require a separate narrowly scoped PR against the parity branch; do not fork local copies inside `ppca_abinitio/`."

### Section 3

- Load-bearing: pinning to `E_with_precompute`, `compute_little_H_b`, `compute_bHb_terms`, `M_with_precompute`, and the grid helpers is the right anchor to current behavior (`docs/math/plan_ppca_abinitio_v0.md:73-100`).
- What's wrong: the spec is right that `HeterogeneousEMState` is not the target algorithm, but wrong to imply it should mostly disappear from the experimental plan. `HeterogeneousEMState.E_step` is exactly the same low-rank scorer (`recovar/em/states.py:180-191`), so it is not a distinct Stage 1A baseline if `(μ,U,s)` are matched. But for learning stages it matters a lot: it already couples the same scorer to covariance-column accumulation (`recovar/em/states.py:194-236`, `recovar/em/heterogeneity.py:322-430`, `recovar/em/heterogeneity.py:550-685`) and then extracts a subspace via projected covariance solves (`recovar/em/states.py:243-284`). Direct PPCA would gain explicit latent posteriors and a generative objective; it would lose the non-PPCA covariance-estimation path and its weaker assumptions about latent distribution.
- What to change: state this explicitly. Recommendation: do not use `HeterogeneousEMState` as a separate Stage 1A score baseline, because the scorer is the same. Do use it as a Stage 1C / Phase 2 baseline on synthetic data for mean error, projector error, stability, and runtime. If direct PPCA cannot beat or at least match the existing heterogeneous learner on the same fixed-grid harness, there is no implementation case.

### Section 4

- Load-bearing: Section 4.4 is correct under the current legacy scorer. `compute_little_H_b` and `compute_dot_products_eqx` both call `process_images(..., apply_image_mask=False)` (`recovar/em/heterogeneity.py:71-76`, `recovar/em/core.py:82-93`), so the current `H_{i,r}` amortization over translations is valid for the pinned path.
- What's wrong: the model section omits the representation invariants that matter in cryo-EM. `μ` and `U` are described as flat centered Fourier volumes, but there is no statement that they must be valid Fourier transforms of real 3D volumes, i.e. obey conjugate symmetry and have real DC/Nyquist coefficients. That omission becomes fatal once Section 8 proposes gradient updates on `W`. There is also a scale ambiguity the spec never nails down: `Σ_i` lives in RECOVAR's unnormalized FFT convention, so the absolute scale of `s` is not portable unless the synthetic harness uses the same noise convention (`recovar/reconstruction/noise.py:797-800`). Finally, if the latent prior is real-Gaussian, then the legitimate gauge group is real orthogonal, not arbitrary complex unitary.
- What to change: add an explicit representation contract. Example text: "`μ` and each column of `U` must correspond to a real-space volume. After every update, project them back to the real-volume Fourier subspace by enforcing conjugate symmetry in centered FT layout and zeroing imaginary parts at self-conjugate frequencies." Also state that all synthetic `noise_variance` values are in RECOVAR Fourier units, not real-space variance units. If the implementation wants a free gauge, parameterize in real-space volumes and only transform to Fourier for slicing; then the gauge really is orthogonal.

### Section 5

- Load-bearing: the insistence on a fresh self-contained loop and dense global grids before local search is correct (`docs/math/plan_ppca_abinitio_v0.md:183-190`).
- What's wrong: the fixed-grid contract is too favorable to PPCA. Same-grid synthetic generation and inference (`docs/math/plan_ppca_abinitio_v0.md:171-175`) removes exactly the kind of model error that a low-rank heterogeneity term can silently absorb: off-grid pose error, centering error, and residual contrast mismatch. A score improvement on this setup is a positive control, not evidence that PPCA will help on realistic alignment.
- What to change: keep the same-grid case as the first positive control only. Before Stage 1A can graduate, add one mandatory misspecification stressor: either rotation jitter of 1 to 2 degrees and translation jitter of 0.25 to 0.5 px at generation time, or a held-out evaluation on such data. If PPCA only wins when the truth lives exactly on the inference grid, the right conclusion is "not robust enough", not "stage passed".

### Section 6

- Load-bearing: the module split is sensible, especially the intent to separate synthetic generation, posterior computation, initialization, and metrics.
- What's wrong: `PPCABatchPosterior` bakes in full materialization of all posterior tensors (`docs/math/plan_ppca_abinitio_v0.md:256-268`). That silently commits the implementation to a dense-batch API that is already obsolete relative to the rest of the EM code. `engine_v2` was written specifically to avoid materializing the full `(n_images, n_rot, n_trans)` tensor (`recovar/em/dense_single_volume/engine_v2.py:500-510`). The proposed API would make significant-weight pruning, local-search unions, and two-pass normalization all much harder later.
- What to change: split the API into a streaming core and an optional materializer. For example, `posterior.py` should expose a block iterator that yields `(rot_block, trans_block, log_score_block, post_mean_block, post_Hinv_block)` and an accumulator interface for mean/factor updates. A separate debug-only helper can materialize the full tensors for tiny CPU tests.

### Section 7

- Load-bearing: reusing the production `H` and `b` math rather than rewriting it is correct, and the parity-vs-production requirement is exactly the right instinct.
- What's wrong: three separate issues. First, the spec still points at `compute_bHb_terms_eqx` as a meaningful reference (`docs/math/plan_ppca_abinitio_v0.md:84`, `docs/math/plan_ppca_abinitio_v0.md:281-282`, `docs/math/plan_ppca_abinitio_v0.md:679`). In this branch that function is dead code with a stale warning; the production contract is `compute_bHb_terms` as used by `E_with_precompute` (`recovar/em/e_step.py:109-153`). Second, the return type duplicates information: `log_scores` and `log_resp` are both dense, and `post_mean` is the largest tensor in the entire design. Third, the plan has no posterior-calibration test. Score parity does not prove that `m` and `Hinv` are meaningful posteriors.
- What to change: make the parity oracle the actual production path. A concrete test should compute the legacy reference as

```python
resid = compute_dot_products_eqx(...)
resid += compute_CTFed_proj_norms_eqx(...)[..., None]
resid -= compute_bHb_terms(...)
score_ref = -0.5 * resid
score_new = posterior.log_scores
score_ref -= score_ref.max(axis=(1, 2), keepdims=True)
score_new -= score_new.max(axis=(1, 2), keepdims=True)
np.testing.assert_allclose(score_new, score_ref, rtol=1e-6, atol=1e-8)
```

Use `image_shape=(6,6)`, `q=2`, `n_rot=3`, `n_trans=5`, `n_img=4`, complex128/float64, and fixed RNG seed. Add a second deterministic test for posterior calibration at the true pose: `q=2`, `n_img=256`, evaluate the true-pose Mahalanobis distances `(alpha_true - m)^T H (alpha_true - m)` and assert the empirical 90% ellipsoid coverage lies in `[0.85, 0.95]`. If this fails, the posterior moments are not trustworthy even if the score parity test passes.

### Section 8

- Load-bearing: the ladder from fixed score to mean to factor learning is directionally right.
- What's wrong: Stage 1B is the first major scientific trap. `M_with_precompute` only accumulates `Ft_y` and `Ft_ctf` (`recovar/em/m_step.py:224-298`); the actual mean update happens later via `relion_functions.post_process_from_filter` (`recovar/em/states.py:52-55`, `recovar/em/states.py:238-241`). Section 8 never specifies whether the PPCA and homogeneous loops use the same post-processing solve and the same mean prior. More importantly, Stage 1B gates progress on a deliberately wrong M-step (`docs/math/plan_ppca_abinitio_v0.md:334-342`). A success there is uninterpretable. Section 8.3 then compounds the problem by saying "`s` is frozen" while updating free `W` and re-SVDing it into a new `s` (`docs/math/plan_ppca_abinitio_v0.md:326-327`, `docs/math/plan_ppca_abinitio_v0.md:384-387`).
- What to change: Stage 1B should be demoted to a debugging ablation, not a graduation stage. If you keep it, it must use the exact same mean post-processing as the homogeneous baseline:

```python
Ft_y, Ft_ctf = M_with_precompute(...)
mu_next = relion_functions.post_process_from_filter(
    dataset, Ft_ctf, Ft_y, tau=mean_variance, disc_type=disc_type
).reshape(-1)
```

For Stage 1C, if `s` is truly frozen, update `U` only and rebuild `W = U diag(s_fixed)^{1/2}` after every step. The update loop should be:

```python
U_raw = U - lr * grad_U
U_sym = enforce_real_volume_ft(U_raw, volume_shape)
U_band = radial_project(U_sym, k_max)
U = orthonormalize_real_volume_columns(U_band)
W = U * jnp.sqrt(s_fixed)[None, :]
```

Do not use a complex thin SVD as the gauge fix. It violates the latent-model assumptions and can leave the learned columns outside the real-volume subspace. Add a unit test with `volume_shape=(8,8,8)`, `q=3`, one synthetic gradient step, and the assertion that the inverse FFT of every updated column has imaginary-energy fraction `< 1e-10`.

### Section 9

- Load-bearing: avoiding chicken-and-egg dependence on an existing RECOVAR run is right.
- What's wrong: the harness is underspecified exactly where cryo-EM experiments fail in practice. There is no held-out split. There is no homogeneous null. There is no contrast nuisance even though the dataset object supports mutable per-image contrast (`recovar/data_io/cryoem_dataset.py:932-945`). There is no FFT-scale contract for the noise model even though RECOVAR's Fourier units are not RELION's (`recovar/reconstruction/noise.py:797-800`). There is no CTF stressor that places heterogeneity near information-poor shells. And the hand-built sinusoidal low-rank basis is likely to be much easier than realistic localized density motion.
- What to change: define at least four synthetic families before implementation starts. Family A: homogeneous null (`q_true=0` or `s_true=0`). Family B: continuous low-rank heterogeneity concentrated in low frequencies. Family C: same as B but with off-grid pose jitter. Family D: same as B but with per-image contrast scales `c_i ∈ [0.8, 1.2]`. Family E, if time allows: heterogeneity energy concentrated near the first CTF zero for a subset of particles. Also define a train/validation split at dataset construction time and report stage metrics on validation only. Finally, specify the noise contract explicitly: either synthesize noise in RECOVAR Fourier units, or generate in real space and convert using the exact `(H * W)^2` scale factor.

### Section 10

- Load-bearing: keeping alignment, mean, subspace, embedding, and optimization metrics separate is correct.
- What's wrong: the mean metric is named and interpreted incorrectly. `fsc(mu_est, mu_true)` is an oracle-vs-ground-truth comparison, not a split-half FSC, so half-bit and 0.143 thresholds are not honest here (`docs/math/plan_ppca_abinitio_v0.md:466-469`). The metric set also lacks a pre-registered primary metric for each stage, which is why Section 11 devolves into "beats on at least one metric". Finally, embedding metrics need canonicalization: projector error is gauge-invariant, but componentwise embeddings are not.
- What to change: replace Section 10.2 with `oracle_fsc_gt(mu_est, mu_true)` and a non-thresholded summary such as shell-averaged FSC or Fourier relative error. Reserve half-map FSC thresholds for actual half-set reconstructions. Pre-register one primary metric per stage: `true_state_mass_val` for score stages, Fourier relative mean error for mean stages, and projector Frobenius error for factor stages. For embedding metrics, always apply orthogonal Procrustes on the real latent coordinates before reporting errors.

### Section 11

- Load-bearing: the author is trying to enforce stage gates before coding ahead. That is the right discipline.
- What's wrong: the current gates do not answer the questions they claim to answer. Stage 1A can pass by chance or by overfitting to an overly favorable synthetic family. Stage 1B can pass for the wrong reason because the mean update is intentionally wrong. Stage 1C uses truth-perturbed init only and does not require held-out improvement. Phase 2 defines a "stress test" with no pass/fail meaning. Phase 3 asks for a proof obligation ("recovers a direction not in the atlas span") that is both harder than needed and poorly defined.
- What to change: replace Section 11 entirely. A drop-in rewrite is below.

Alternative Stage 1A design A: fixed-pose posterior calibration. Use the true pose only, score the latent posterior under the true `(μ,U,s)`, and test calibration of `m` and `Hinv`. Pros: small, sensitive, isolates the low-rank image model from search. Cons: it does not test whether PPCA helps pose ranking.

Alternative Stage 1A design B: heterogeneity-strength sweep on held-out images. Use the true `(μ,U,s)` and the full pose grid, vary latent amplitude across a small sweep that includes the homogeneous null, and measure `Δ true_state_mass_val = PPCA - homogeneous`. Pros: directly falsifies the claim that marginalized PPCA helps alignment; also tells you the minimum heterogeneity strength where it helps. Cons: still synthetic and still uses oracle factors.

Alternative Stage 1A design C: homogeneous-residual rescue. First fit a homogeneous mean, then compare whether PPCA reorders the same candidate poses toward truth when scored with a fixed `U` estimated from residual structure. Pros: closer to the real intended use. Cons: depends on how `U` is initialized, so it is harder to interpret as an early falsifier.

My recommendation is A plus B before any M-step code, then move to residualized mean learning.

### Section 12

- Load-bearing: the typed API sketch is useful.
- What's wrong: `FixedGridSpec.log_prior` exists (`docs/math/plan_ppca_abinitio_v0.md:615-620`) but the posterior signature ignores priors. `PPCABatchPosterior` duplicates dense arrays that should never be required at once. And the API is still built around a fixed dense Cartesian grid, which will make future local-search unions awkward.
- What to change: define separate rotation and translation log priors up front, even if they are flat in v0. Replace `PPCABatchPosterior` with a small `PosteriorStats` object plus optional block outputs. Add a future-proof candidate-grid abstraction now: a dense fixed grid is one implementation of `CandidateGrid`, not the only one.

### Section 13

- Load-bearing: the brute-force posterior test and production-parity test absolutely should exist before any larger run.
- What's wrong: the stage-gate experiments do not belong in `test-fast` as minute-long CPU unit tests (`docs/math/plan_ppca_abinitio_v0.md:676-688`). That conflicts with the spirit of `tests/CLAUDE.md`: unit tests should be deterministic and focused, while stage experiments are scientific regressions or integration runs. The table also still points to `compute_bHb_terms_eqx`, which is the wrong oracle for this branch.
- What to change: keep only deterministic math and convention tests in `tests/ppca_abinitio/` under the `unit` marker. Move Stage 1A/1B/1C experiment gates to `scripts/ppca_abinitio/` with fixed seeds, JSON output, and an optional lightweight `integration` test that only checks the script runs and emits the expected keys. Concrete new unit tests I would require before implementation:

`test_score_matches_e_step_residual_ref.py`
: `image_shape=(6,6)`, `q=2`, `n_rot=3`, `n_trans=5`, `n_img=4`, float64; assert score parity to `rtol=1e-6`, `atol=1e-8`.

`test_posterior_mean_is_real_on_real_volume_inputs.py`
: same setup, but with Fourier data generated from real-space volumes; assert `max(abs(imag(post_mean))) < 1e-10`.

`test_real_volume_symmetry_projection.py`
: `volume_shape=(8,8,8)`, `q=3`; after one projected factor step, inverse-FFT imaginary-energy fraction `< 1e-10`.

`test_fft_noise_scale_contract.py`
: white-noise images of size `16x16`, known real-space variance `σ^2`; after `process_images`, empirical mean Fourier power is within 5% of `σ^2 * (16 * 16)^2`.

`test_null_heterogeneity_no_gain.py`
: as a small script-level regression, homogeneous synthetic data should show `|Δ true_state_mass| <= 0.01`.

### Section 14

- Load-bearing: the section is right to call out mean-absorbs-heterogeneity and premature local refinement.
- What's wrong: the missing failure modes are more serious than the listed ones. There is no mention of off-grid pose error being absorbed into `W`; no contrast or per-particle scale nuisance; no CTF-zero unidentifiability; no overconfident posteriors on null data; no handedness / class-pose ambiguity for Phase 3 atlas volumes.
- What to change: add those failure modes explicitly, with at least one synthetic family or validation metric attached to each. If the spec names a failure mode without assigning a diagnostic to it, it is not really mitigated.

### Section 15

- Load-bearing: the repo-wide validation policy is correctly inherited.
- What's wrong: it blurs code-validation requirements and scientific-stage requirements. The long-test suite is mandatory for a PR, but it is not the thing that proves the PPCA experiments mean what they claim to mean.
- What to change: keep the existing PR rules, but add a second checklist for scientific claims: primary metric, null control, held-out evaluation, HeterogeneousEMState baseline where relevant, and exact synthetic family used.

### Section 16

- Load-bearing: it is good that open questions are written down instead of hidden.
- What's wrong: the biggest unresolved questions are missing. The spec does not ask whether `alpha` is treated as real or complex, whether order 3 is allowed before the posterior API streams, whether the synthetic harness includes held-out validation, or whether Stage 1B is allowed to stop being a gate.
- What to change: add those questions now. They are architectural, not incidental.

## Issues the author missed

- The plan still leans on the wrong parity oracle. `compute_bHb_terms_eqx` is dead code in this branch, but the spec and test table still mention it (`docs/math/plan_ppca_abinitio_v0.md:84`, `docs/math/plan_ppca_abinitio_v0.md:281-282`, `docs/math/plan_ppca_abinitio_v0.md:679`). The only parity target that matters is the score actually used inside `E_with_precompute`.
- The gauge problem is worse than "SVD breaks Hermitian symmetry". A complex SVD produces a complex unitary right factor. If the latent prior is `alpha ~ N(0, diag(s))` over real coordinates, that is not an allowed gauge transformation. This is a PPCA-modeling bug, not just a representation bug.
- Oracle FSC vs ground truth is being given split-map FSC semantics. That is a methodological error, not just a wording issue. Using half-bit / 0.143 on `fsc(mu_est, mu_true)` will make the experiment look much more mature than it is.
- The synthetic harness has no held-out split. Without one, any Stage 1C claim is training-set-only. On synthetic data this is easy to fix, so there is no excuse for not doing it.
- The plan ignores per-particle amplitude/contrast nuisance even though RECOVAR exposes it directly (`recovar/data_io/cryoem_dataset.py:932-945`). A direct PPCA model without a scale nuisance can easily spend its first factor explaining contrast variation rather than structure.
- The noise model scale is unspecified in RECOVAR units. Because RECOVAR stores Fourier images in the native unnormalized FFT convention (`recovar/reconstruction/noise.py:797-800`), the same symbol `σ^2` means different numbers in real-space and Fourier-space descriptions. If that scale is not fixed in the spec, `s_true` and `s_init` are not comparable between experiments.
- The Phase 2 initializer ordering is backwards from an interpretability standpoint. `docs/math/plan_ppca_abinitio_v0.md:555-558` prefers RELION or cryoSPARC means before a homogeneous RECOVAR mean. For a clean algorithmic bootstrap study, the same-codebase homogeneous mean is the lowest-convention-risk input and should come first.
- The atlas pipeline ignores handedness and shell-wise scale normalization. `load_relion_volume` fixes the coordinate frame, not the fact that independent class volumes can differ by handedness, rotation, mask boundary, and amplitude scale before PCA.
- The current plan makes it too easy for `W` to learn residual pose derivatives. On same-grid data with no pose jitter this will look like success. Mild off-grid jitter is the smallest relevant cryo-EM falsifier for that failure mode.
- The stage experiments are planned as fast CPU tests. That is a testing-architecture smell and conflicts with `tests/CLAUDE.md`'s marker separation. Unit tests should pin math and invariants; scientific stage gates should be scripts or integration tests with saved metrics.

## Revised Section 11

The text below is the replacement I would use for Section 11.

---

## 11. Staging plan and exit criteria

The loop must not advance to the next stage until the previous one
satisfies its exit criterion on the synthetic harness. For every stage
below, use:

- a fixed train/validation split of images;
- at least 3 RNG seeds;
- one heterogeneous synthetic family and one homogeneous-null family;
- one pre-registered primary metric;
- the same metric reported on validation only for any stage that learns
  parameters from the train split.

### 11.0 Common evaluation protocol

**Synthetic families.**

1. **Null family.** `s_true = 0` (or `q_true = 0`), same pose/CTF/noise
   distribution as the heterogeneous run.
2. **Matched-grid heterogeneous family.** Low-rank continuous
   heterogeneity with known `(μ_true, U_true, s_true)`.
3. **Misspecified heterogeneous family.** Same as (2), but generation
   includes either rotation jitter of 1 to 2 degrees or translation
   jitter of 0.25 to 0.5 px that is not present in the inference grid.

**Primary metrics by stage.**

- Score stages: validation `true_state_mass`.
- Mean stages: validation Fourier relative error of `μ` against
  `μ_true` (report `oracle_fsc_gt` as a secondary curve only).
- Factor stages: projector Frobenius error
  `||P_{U_est} - P_{U_true}||_F`.

**Reporting rules.**

- Report the primary metric first, with mean and 95% bootstrap CI over
  images, for each seed.
- Report secondary metrics, but secondary metrics do not decide stage
  graduation.
- Any FSC against ground truth must be labeled `oracle_fsc_gt`; do not
  use half-bit or 0.143 thresholds unless actual half maps are
  reconstructed.

### 11.1 Stage 0A — posterior helper correctness

**Implement.** `score_and_posterior_moments_eqx(...)`.

**Required tests.**

1. Brute-force parity for `log_scores`, `m`, and `Hinv` on tiny dense
   problems (`q <= 3`, `n_rot <= 4`, `n_trans <= 4`, `image_size <= 32`,
   float64).
2. Production-score parity against the score assembled from
   `compute_dot_products_eqx`, `compute_CTFed_proj_norms_eqx`, and
   `compute_bHb_terms`, not against dead helper code.
3. Real-volume posterior invariant: with synthetic data generated from
   real-space volumes, the posterior mean coordinates are real to
   numerical tolerance.

**Exit criterion.** All required tests pass. If any fail, do not run
stage experiments.

### 11.2 Stage 0B — oracle-score falsification

**Implement.** `run_score_diagnostic(...)` with no parameter updates.
Use the true `(μ_true, U_true, s_true)` for the PPCA branch and the same
`μ_true` with `u=None` for the homogeneous branch.

**Primary metric.** Validation `true_state_mass`.

**Exit criterion.** All of:

1. On the matched-grid heterogeneous family, PPCA improves validation
   `true_state_mass` over homogeneous by a positive amount whose 95%
   bootstrap CI excludes zero for all 3 seeds.
2. On the null family, the absolute change in validation
   `true_state_mass` is <= 0.01 for all 3 seeds.
3. On the misspecified heterogeneous family, the PPCA advantage does
   not disappear entirely. A smaller gain is acceptable; zero or sign
   reversal is not.

If this stage fails, stop the project. There is no reason to implement a
PPCA M-step if the oracle factors do not help the score under modest
model mismatch.

### 11.3 Stage 1A — non-oracle score stress test

**Implement.** Re-run `run_score_diagnostic(...)` with one misspecified
factor initialization:

- truth-perturbed `U` as a positive control;
- random-lowpass `U` as a negative control.

**Primary metric.** Validation `true_state_mass`.

**Exit criterion.**

- Truth-perturbed init must preserve a positive PPCA-over-homogeneous
  score gain on the matched-grid heterogeneous family.
- Random-lowpass init is **not** a graduation requirement. It is a
  stress test only. A failure here is informative; a success here does
  not mean the bootstrap problem is solved.

### 11.4 Stage 1B — residualized mean-only loop

**Implement.** `run_fixed_grid_ppca_gem(..., update_mu=True,
update_factor=False)` using the residualized mean update
`y_i - A_g U m_{i,g}` and the same post-processing solve as the
homogeneous baseline.

**Primary metric.** Validation Fourier relative error of `μ`.

**Exit criterion.** All of:

1. From truth-perturbed init, the PPCA loop improves the primary metric
   over the homogeneous loop for all 3 seeds on the matched-grid
   heterogeneous family.
2. The validation `true_state_mass` at the final iteration is not worse
   than the initialization by more than 0.01 absolute.
3. On the null family, the PPCA loop is not better than homogeneous by
   more than noise-level fluctuations. If it is, the factor correction is
   likely explaining nuisance structure rather than heterogeneity.

Mean-update v0 without residualization may still be implemented as a
debugging ablation, but it must not be a stage gate.

### 11.5 Stage 1C — fixed-spectrum factor learning

**Implement.** Factor updates with truly fixed `s`. Update `U` only,
project back to the real-volume Fourier subspace after every step, apply
the agreed radial band-limit / support mask, and re-orthonormalize.

**Primary metric.** Projector Frobenius error.

**Secondary metrics.** Validation `true_state_mass`, oracle embedding
error, validation mean error, and the train-side generalized-EM
objective.

**Exit criterion.** All of:

1. Final projector error improves over the initialization for all 3
   seeds on the matched-grid heterogeneous family.
2. Oracle embedding error improves over the initialization for all 3
   seeds.
3. Validation `true_state_mass` does not regress relative to Stage 1B by
   more than 0.01 absolute.
4. No NaN/Inf is produced, and the factor-update projection preserves
   the real-volume invariant to numerical tolerance.
5. The train-side generalized-EM objective is non-decreasing up to a
   relative tolerance of `1e-3`.

Random-lowpass init remains a stress test only. Report it, but do not
use it as the sole success path.

### 11.6 Stage 1D — full soft M-step

Implement only after Stage 1C is stable across the 3 seeds above.
Before enabling this stage, update this spec with:

- the exact closed-form solve,
- the memory plan,
- the primary metric,
- and the null-family behavior expected from the full second-moment path.

### 11.7 Phase 2 — external-mean bootstrap

**Goal.** Test basin-of-attraction reachability from a non-oracle mean.

**Initialization order.**

1. homogeneous RECOVAR mean;
2. RELION mean converted with `load_relion_volume`;
3. cryoSPARC mean after an explicit frame and scale audit.

Use random-lowpass `U` only as a negative-control stress test, not as
the main initializer.

**Primary metric.** Final projector error on synthetic data.

**Exit criterion.** Starting from the external mean and a non-oracle
factor init, the final projector error and final mean error are within a
pre-declared tolerance band of the Stage 1C truth-perturbed result on
the same synthetic family. If random-lowpass fails, document the
failure; do not rebrand it as a partial success.

### 11.8 Phase 3 — K-class atlas bootstrap

**Pipeline.**

1. Run an external `K`-class ab-initio.
2. Convert every class volume to RECOVAR frame.
3. Align each class volume with an explicit rotational search and a
   handedness check; rigid Procrustes on voxel vectors is not sufficient.
4. Normalize shell-wise amplitude before PCA.
5. Form the atlas mean and deviations.
6. Use at most `K - 1` atlas-derived directions as claimed atlas PCs.
   If `q > K - 1`, the extra directions are auxiliary random directions
   and must be labeled as such.
7. Run the Stage 1C loop.

**Primary metric.** Final projector error relative to the atlas
initializer.

**Exit criterion.** On the same dataset, the PPCA loop improves both the
projector error and the mean error relative to the raw atlas initializer
without destabilizing the score stage. Do not require a vague proof that
"a direction outside the atlas span" was found; require a measurable
improvement over the atlas itself.

### 11.9 Phase 4 — dataset sweep + faster E-step

Out of scope for v0 implementation. Before entering Phase 4, the
posterior API must already support streaming/blockwise accumulation so
that local-grid search and significant-weight pruning can be added
without rewriting the whole PPCA stack.

---

## Recommended prerequisite work

- Add a production-score parity test that compares the new posterior helper against the score assembled from the exact legacy E-step pieces, not against `compute_bHb_terms_eqx`.
- Add a real-volume / Hermitian-invariant test for any factor-update projection routine. This is a correctness prerequisite, not an optimization nice-to-have.
- Add an FFT-noise-scale contract test so that synthetic `σ^2`, Fourier `noise_variance`, and reported `s` all live on the same scale.
- Add a null-heterogeneity script experiment before any heterogeneity-positive experiment. If PPCA looks helpful on a homogeneous dataset, the whole methodology is suspect.
- Add a held-out split to the synthetic harness and require all learning-stage exit metrics to be reported on validation only.
- Add one off-grid misspecification stress run before any claim that a Stage 1A or 1C result predicts real-data usefulness.
- Add an atlas-import audit script that checks frame conversion, handedness choice, and shell-wise scale normalization before Phase 3 exists on paper.
- Keep stage-gate science runs as scripts or integration runs with JSON output; keep `tests/ppca_abinitio/` focused on math and invariants.

## Open questions to escalate to the user

- Is Stage 1B allowed to stop being a graduation stage and become a debugging ablation only? I think it should.
- For "frozen `s`", do you want true fixed-spectrum learning of `U` only, or are you willing to admit that Stage 1C jointly updates `(U,s)`? The current text tries to do both.
- Should `HeterogeneousEMState` be required as a baseline for Stage 1C and Phase 2? My recommendation is yes.
- Is order 3 actually required for v0, or can Stage 1 be declared order-2-only until the posterior API streams instead of materializing full tensors?
- Are the Phase 2/3 bootstrap claims meant to cover only continuous heterogeneity, or also mixtures / class-like data? If mixtures are in scope, the plan needs an explicit statement that PPCA is only an approximation there.
- Are you willing to add a held-out validation split and a homogeneous null family even though the current draft tries to keep v0 minimal? Without those, the stage exits are not honest enough.
- For external atlas volumes, is RELION the only intended source, or must cryoSPARC imports be supported in the same phase? That choice changes the required convention and scale audits.
