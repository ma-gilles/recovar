# PPCA refinement contracts

This package refines a pose-marginalized PPCA model initialized from aligned
K-class or GT volumes. It is not a native PPCA InitialModel/VDAM controller.
The [package reading guide](../../recovar/em/ppca_refinement/README.md) maps
entry points and data flow. Historical experiments do not qualify this source.

## Implementation owners

- [engine.py](../../recovar/em/ppca_refinement/engine.py) owns shared dense/local
  scoring and accumulation. The independent score/moment formulation and
  augmented solve live in `recovar/ppca/`; preserve the `q=0` and `W=0` limits.
- [ppca_bridge.py](../../recovar/em/ppca_refinement/ppca_bridge.py) feeds pose/Pmax
  diagnostics through the shared `RefinementState` controller. K-class execution
  lives in [classification/k_class.py](../../recovar/em/classification/k_class.py).
- [local_dataset.py](../../recovar/em/ppca_refinement/local_dataset.py) consumes
  `LocalHypothesisLayout`; local priors and pruning constrain support without
  changing the PPCA score. Do not introduce a parallel local-search layout.
- The model is `[mu, W_1, ..., W_q]`, with real latent `z ~ N(0, I)`.
  The joint M-step retains mean/loading and loading/loading cross terms.
  Mean regularization uses its own RELION-style prior; `W_prior` is variance-like.
  Frequency chunking avoids materializing the whole unpacked augmented LHS.

## Diagnostic dense-runner controls

The dense NPZ runner provides explicit diagnostic controls:

1. `--current-size-schedule` applies a fixed per-iteration Fourier window such
   as `32,48,64,64,64`. This is frequency/current-size marching for controlled
   experiments, not a substitute for the halfset FSC gate.
2. `--freeze-mean-iters` keeps `mu` fixed for the first N iterations and solves
   only the conditional W equation
   `(A_WW + W_precision) W = rhs_W - A_Wmu mu_fixed`.
3. GT-derived mean/W priors default to raw half-Fourier shell power
   (`--gt-prior-box-power 0`). The legacy `N^2`-divided setting is preserved
   as an explicit diagnostic option because it over-shrinks W in the dense
   augmented M-step.

These controls separate pose/search failures from loading and regularization
failures. They do not replace the halfset resolution gate.

## Objective accounting

Implementation: [diagnostics.py](../../recovar/em/ppca_refinement/diagnostics.py).

The runner reports three objective families separately:

1. `log_likelihood` / `logZ_mean` are the E-step pose evidence for the current
   scoring model.
2. `mstep_objective_input_*`, `mstep_objective_solved_*`, and
   `mstep_objective_output_*` are fixed-statistics augmented quadratic
   lower-bound terms for one M-step. These values are comparable within one
   iteration only.
3. `legacy_logZ_plus_input_prior` is retained for old summaries but is not an
   EM objective, because it mixes E-step log evidence with a prior penalty from
   a different fixed-statistics quadratic.

The regression guard checks that `mstep_objective_solved_delta_per_image` is
non-negative up to numerical tolerance. It does not require the postprocessed
scoring model to improve the M-step objective, because RELION-style masking,
background fill, and grid correction are explicit heuristics outside the
augmented solve. Clean algorithmic experiments should use
`--postprocess-strategy none`; heuristic scoring-model experiments should track
`mstep_objective_postprocess_delta_per_image` as a separate diagnostic.

## Bootstrap initialization

Implementation: [initialization.py](../../recovar/em/ppca_refinement/initialization.py).

The bootstrap initializer consumes already aligned or explicitly
alignment-checked class/GT volumes:

1. load K-class or GT volumes in a declared frame (`recovar`, `relion`, or
   `fourier`);
2. align volumes before PCA when an alignment callback is supplied;
3. compute `mu_init` as the weighted mean;
4. compute `W_init = U_q sqrt(lambda_q)` so `W W*` reconstructs the intended
   weighted centered-volume covariance;
5. keep the latent prior identity and never hide eigenvalues in `z`;
6. fail loudly rather than silently guessing GT frame, Fourier normalization,
   mask, or amplitude scale.

The first implementation includes an identity alignment path and explicit
frame-conversion hooks. Production volume alignment remains a separate task.

## Known image-scale correction

Implementation: [dense_dataset.py](../../recovar/em/ppca_refinement/dense_dataset.py).

Synthetic fixtures can include a known per-image signal scale
`per_image_contrast`. When this is explicitly enabled, the PPCA score treats
the projection model as `s_i A_i [mu, W]`:

1. image cross/RHS terms are multiplied by `s_i`;
2. CTF-squared, template quadratic, and augmented LHS terms are multiplied by
   `s_i^2`;
3. `y_norm` is unchanged because it is pose/model independent.

This is not a production substitute for estimating scale/contrast on real
data. It is a synthetic debugging control that prevents PPCA loadings from
absorbing nuisance contrast while we test the pose and heterogeneity update
itself. Dense and exact-local paths now use the same optional correction, and
the all-retained local-support unit test checks parity with dense when the
scale correction and W-score tempering are both active.

## Post-solve mask/grid heuristic

Implementation: [postprocess.py](../../recovar/em/ppca_refinement/postprocess.py).

The current default scoring model uses the post-solve PPCA reference:

1. `mu` gets the RELION-style real-space soft mask, background fill, and
   optional grid correction;
2. each `W` column gets the same soft support but with zero background;
3. after low-resolution/current-size solves, postprocessed `mu/W` are
   bandlimited back to the active reconstruction radius;
4. postprocessed `W` is shell-power capped so real-space masking cannot invent
   covariance power in shells where the solved `W` or GT-derived shell prior has
   little support.

This remains a flagged heuristic, not a masked PCG objective. The cap is
deliberately variance-scale preserving: it bounds the postprocessed
`sum_k |W[xi,k]|^2` shell average by the solved pre-mask value, keeping the
latent prior identity and avoiding the failure mode where a W mask creates a
huge prior penalty at low current size. A future production strategy should
move masking/preconditioning into the objective rather than applying it after
the augmented solve.

## Halfset resolution gate

Dense and local iteration publication share
[`_finish_refinement_iteration`](../../recovar/em/ppca_refinement/refinement_loop.py);
the gate itself remains in [`evaluate_halfset_resolution_gate`](../../recovar/em/ppca_refinement/schedule.py).

Resolution/current-size growth is allowed only when all first-pass checks pass:

1. K-class schedule says growth is otherwise allowed;
2. current and previous best pose ids are stable under a strict fraction
   threshold;
3. halfset means are aligned before comparison;
4. the halfset mean FSC or supplied equivalent supports the proposed
   `current_size`;
5. no frame/sign/drift diagnostic fails;
6. an optional minimum-pmax threshold passes when requested; pmax alone never
   permits growth. `logZ_mean` is diagnostic, not a gate in this function.

For this pass, "mean of halfset distributions" means the halfset mean volumes
`mu_half[0]` and `mu_half[1]`. Loading subspace agreement is reported as a
diagnostic where available, with sign flips treated as stable when the span and
scale are unchanged.

## Validation inventory

These focused checks cover the contracts below; their presence is not a claim of current-source scientific qualification:

1. `tests/unit/ppca_refinement/test_augmented_mstep.py`
   verifies joint augmented normal equations and `q=0` homogeneous reduction.
2. `tests/unit/ppca_refinement/test_dense_q0_parity.py`
   verifies the dense block uses the same PPCA score expression and that `q=0`
   / `W=0` reduce to homogeneous scoring up to documented constants.
3. `tests/unit/ppca_refinement/test_kclass_initialization.py`
   verifies weighted mean/PCA covariance scaling, no double RELION conversion,
   and synthetic PC recovery up to sign/subspace rotation.
4. `tests/unit/ppca_refinement/test_ppca_schedule.py` and
   `test_halfset_resolution_gating.py` verify negative resolution-growth cases:
   FSC good but poses changed, poses stable but halfset means misaligned, and
   sign-flipped `W` with the same subspace remains stable.
5. `pixi run test-em-fast-guard` remains the EM-scoped guard for dense/local
   infrastructure health; no full RECOVAR long suite is run for this task.
6. `tests/unit/ppca_refinement/test_dense_dataset_iteration.py` verifies the
   dataset-backed dense iteration, exact-local all-retained parity, and dense
   and exact-local refinement loop positive/negative resolution-gating cases.
   It also guards the low-current-size default postprocess so W shell-power
   capping prevents post-solve masking from increasing the W prior penalty.
7. `tests/unit/ppca_refinement/test_kclass_fixture_validation.py` verifies the
   real-fixture validator on RELION-frame K-class MRCs and RECOVAR-frame GT
   MRCs.
8. `scripts/run_ppca_dense_fixture_smoke.py` verifies a capped real-fixture
   dense PPCA E/M iteration produces finite `mu/W`, pose diagnostics, and best
   pose arrays from K-class-initialized PPCA.
9. `tests/unit/ppca_refinement/test_ppca_postprocess.py` verifies mean
   background fill, W zero-mask behavior, Fourier bandlimiting, and the
   shell-power cap.
10. `tests/unit/ppca_refinement/test_synthetic_recovery_checker.py` verifies
    the embedding/recovery diagnostics and guards the exact+HEALPix candidate
    source used to distinguish pose-grid misses from PPCA algebra failures.

## Historical evidence and limits

The [original branch plan](https://github.com/ma-gilles/recovar-experiments/blob/025f7726cc2b1b92cb870b5f306c534528166f94/docs/math/ppca_angle_refine_plan_2026_05_05.md)
preserves all synthetic recovery measurements, old branch/package names and
proposed next work. Correlation and embedding diagnostics in that history are
not substitutes for current FSC/GT qualification.

`scripts/validate_ppca_kclass_initialization_fixture.py` checks declared volume
frames, weighted covariance scale and path-versus-preloaded loading.
`scripts/run_ppca_dense_fixture_smoke.py` exercises a capped real-fixture
iteration; it does not establish source-level RELION replay parity. Production
alignment, masked/preconditioned PCG, and full real-data dense-to-local
qualification remain separate work; archival does not mark them complete.

The completed May bandwidth experiment (scripts, negative results and plot) is
preserved in the [experiment archive](https://github.com/ma-gilles/recovar-experiments/tree/545be1075145aa93c8da2a13c8dd5526aa45f51d/recovar/scripts/experiments/ppca_init_bandwidth_2026_05).
Its historical workdir defaults and external data requirements remain recorded there.
