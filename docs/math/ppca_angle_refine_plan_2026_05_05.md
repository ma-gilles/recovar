# PPCA angle-refine EM on K-class bootstrap plan (2026-05-05)

This branch starts from `origin/codex/kclass-class3d-mstep-20260505`
(`42c83ff4` after rebasing onto the current target) and ports only durable refinement-first pieces from
`origin/claude/ppca-refine-mature-pipeline`. The target is not native PPCA
VDAM. InitialModel / VDAM / K-class outputs are bootstrap inputs for poses and
aligned class volumes; the PPCA control plane remains a mature refinement-EM
loop.

## What is being ported

The old branch's durable pieces are:

1. `recovar.ppca.pose_marginal`: no-contrast per-pose PPCA score and
   augmented moments, with the required `q=0` and `W=0` homogeneous limits.
2. `recovar.ppca.pose_accumulators` and `recovar.ppca.pc_prior_config`: small
   data containers that document axis order, latent-prior identity, and
   `W_prior` as a variance-like loading prior.
3. A focused augmented M-step wrapper, rewritten here as a direct
   per-frequency normal-equation solve for the first dense loop. Production
   PCG and multimask generalization remain separate numerical improvements.
4. Dense block score/diagnostic helpers that reuse the same PPCA score
   function as future sparse/local paths. The implementation includes
   prepared-block fused volume backprojection and a dataset-backed dense
   two-pass EM iteration that normalizes across all rotation blocks before
   streaming pass-2 backprojection accumulators.
5. A dense halfset refinement loop that calls the dataset-backed PPCA
   iteration, compares halfset means at the proposed resolution, and advances
   `current_size` only when best-pose stability, halfset agreement, and the
   production K-class schedule bridge all pass.
6. An exact-local PPCA path that consumes `LocalHypothesisLayout` directly,
   applies local rotation/translation priors and sample masks as support-only
   constraints, and reuses the same fused PPCA score/backprojection algebra.
7. A PPCA/K-class schedule bridge in
   `recovar/em/dense_single_volume/iteration_loop.py` that feeds PPCA
   best-pose/Pmax diagnostics through the production `RefinementState` update
   logic, so HEALPix/local-search/convergence state evolves with the same
   controller as K-class refinement.
8. The mean-prior split from the PPCA guide: component 0 will use the
   RELION/K=1/K-class mean prior, while components 1..q use `W_prior`.

## What is being discarded or rewritten

The following old-branch pieces are intentionally not carried forward:

1. Any reference to stale K-class files such as `dense_k_class_engine.py` or
   `local_k_class_engine.py`. The live K-class orchestration is
   `recovar/em/dense_single_volume/k_class.py`, with dense and exact-local
   helpers under the same package.
2. PPCA InitialModel / VDAM controller logic, pseudo-halfset VDAM schedules,
   BPref-specific adaptive scaling, or native PPCA VDAM accumulators.
3. Simple pmax-only resolution growth. PPCA resolution growth must be gated by
   best-pose stability and halfset mean agreement.
4. A post-Wiener halfset combiner as the final gold-standard mechanism. Halfset
   mean comparison is a gate, not a substitute for halfset state.
5. Random local-search perturbation builders that bypass
   `LocalHypothesisLayout`.
6. Placeholder scalar EMA noise updates and stale old-branch long-suite
   checklists that conflict with `recovar/em/CLAUDE.md`.

## K-class reuse boundary

PPCA should reuse K-class schedule mechanics wherever they are about pose and
compute structure:

1. dense pass layout and K-class normalization semantics from
   `recovar/em/dense_single_volume/k_class.py` and its dense helpers;
2. `current_size`, HEALPix order, adaptive oversampling, and translation
   evolution from the mature EM loop;
3. `LocalHypothesisLayout` and exact-local bucketing when the schedule switches
   to local search;
4. diagnostics compatible with K-class: logZ, pmax, best pose, significant
   pose counts, per-iteration pose changes, and current-size trajectories.

PPCA differs from K-class in model state and halfset acceptance:

1. the scoring model is `[mu, W_1, ..., W_q]` with latent `z ~ N(0, I)`;
2. the M-step solves the augmented `[mu, W]` normal equations jointly, including
   off-diagonal mean/loading and loading/loading terms. The first direct solve
   can chunk over frequencies so the unpacked `[q+1, q+1]` LHS is not
   materialized for the whole volume at once;
3. halfset gating compares the mean of each halfset PPCA distribution,
   primarily `mu_half[0]` and `mu_half[1]`, after alignment;
4. `W` sign/subspace ambiguity is diagnostic-only in the first controller and
   must not block a valid same-span sign flip.

## Current masking / grid-correction policy

This branch currently uses an explicit post-solve heuristic, not the final
masked-objective PPCA strategy:

1. the augmented `[mu, W]` M-step itself remains unmasked and frequency-local;
2. component 0 uses the K=1/K-class RELION-style mean tau denominator;
3. after the solve, the default scoring model applies RELION-style soft
   background-fill masking and grid correction to `mu`;
4. after the solve, the default scoring model soft-masks every `W` column with
   zero background outside the support and applies the same grid correction;
5. diagnostics include
   `heuristic_post_solve_mask_grid_correction_not_masked_pcg_objective` so this
   is not confused with a principled native PPCA objective.

The intended replacement is a masked/preconditioned PCG formulation closer to
the non-refinement PPCA solver, where the mask and preprocessing operators are
part of the optimization problem. The mean may still need an optional
RELION-style post-solve background-fill step because RELION's K=1 masking is a
heuristic reference-preparation step rather than a clean EM objective term.

## Diagnostic dense-runner controls

The dense NPZ runner has two deliberately explicit controls for debugging
failure modes before the production PPCA auto-refine controller exists:

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

These controls are intended to separate pose/search failures from PPCA loading
and regularization failures. They should be removed or folded into a proper
stateful schedule once the full dense refinement loop owns resolution growth.

## Objective accounting

The runner now reports three objective families separately:

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

The bootstrap initializer will consume already aligned or explicitly
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

## Real fixture validation

The first real-data bootstrap check is
`scripts/validate_ppca_kclass_initialization_fixture.py`. It consumes the
`summary.json` emitted by `scripts/run_k_class_parity.py`, loads the selected
K-class class maps in a declared frame, initializes PPCA, and writes a compact
JSON report with:

1. class weights and `q`;
2. K-class volume, `mu`, and `W` scale diagnostics;
3. covariance trace parity between the weighted centered class maps and
   `W W*`;
4. a path-vs-preloaded loader-frame check to catch accidental double
   RELION-to-RECOVAR conversion;
5. optional CryoBench/PDB manifest comparison against GT mean and loading
   subspace.

The GT comparison is diagnostic-only by default. In the current K=4 5k
CryoBench/PDB fixture, one-iteration K-class maps have much lower covariance
scale than the GT volumes, so GT subspace agreement is not a safe bootstrap
gate until a mature multi-iteration or aligned InitialModel fixture is used.

`scripts/run_ppca_dense_fixture_smoke.py` is the first real-fixture dense PPCA
one-iteration smoke. It reuses the fixture validator for `mu/W`, loads the
RELION model noise spectrum, builds a capped dense pose grid, and runs one
dataset-backed dense PPCA M-step on a particle subset. The default smoke is
deliberately capped and uses RECOVAR's generic HEALPix grid unless the optional
RELION-bind orientation helper is available; it validates the PPCA dense data
path, not source-level RELION replay parity.

## Synthetic recovery diagnostics

The synthetic checker now reports both map-space and embedding-space recovery:

1. pose error against simulator rotations/translations;
2. mean-map correlation and `W` subspace agreement against simulator-scaled
   class-volume PCA;
3. embedding subspace and linear `R^2` against GT class-volume PC scores;
4. assignment one-hot `R^2`, which is useful when the simulator uses discrete
   classes rather than native continuous PPCA latents;
5. per-image contrast `R^2`, to catch the failure mode where PPCA loadings
   explain contrast instead of conformation.

The current 5k, 128-box, noise=1 ribosome-class fixture shows robust pose
behavior with HEALPix3 and GT mean/random shell-power `W` warm start, but not
full heterogeneity recovery from random `W`: pose median error is about 3.9 deg
and assignment one-hot mean `R^2` is about 0.18 after eight clean iterations.
With exact simulator poses and GT `W` initialization, assignment mean `R^2`
improves to about 0.52 and contrast `R^2` stays low, so contrast is not the
dominant failure mode. With exact poses but random `W`, known synthetic
per-image contrast scaling is critical. On the 128-box noise=0.01 fixture, the
same GT-mean/random-shell-power-`W` warm start recovers both embedding
directions when the known contrast scale is included in the score/LHS/RHS
algebra (`linear_r2_estimated_z_to_gt_pcs` about `[0.99, 0.96]`,
assignment-one-hot mean `R^2` about `0.98`, contrast `R^2` about `0.001`).
Without that scale correction, one recovered direction mostly tracks image
contrast (`contrast_r2` about `0.64`).

The current off-grid HEALPix tests separate pose and heterogeneity failures:

1. HEALPix3 with random `W` and known contrast scale has median pose error
   about 3.8 deg and does not recover the heterogeneity subspace.
2. Adding the exact simulator rotations to the same HEALPix3 distractor set
   gives exact poses and substantially better recovery (`assignment` mean
   `R^2` about `0.82`), so the score algebra is not fundamentally broken.
3. HEALPix4 lowers the median pose error to about 1.9 deg but still loses the
   second PC from random `W`; a denser global grid alone is too expensive and
   not reliable enough.
4. A 20k-image HEALPix3 run initialized with GT `W` took about 533 s for six
   iterations and still had median rotation error about 3.9 deg. The mean
   recovered well (`mu` correlation about `0.96`), but heterogeneity remained
   weak (`embedding R^2` about `[0.56, 0.005]`, assignment mean `R^2` about
   `0.31`). More images alone do not compensate for this pose-grid error.

This makes exact-local pose refinement the next production blocker. PPCA needs
to inherit K-class local neighborhoods and translation priors before the
random-W path can be judged on realistic off-grid images.

At higher noise, exact-pose candidate sets are not enough with only 1k images:
noise=0.1 keeps poses exact but random-W recovery is weak (first embedding
direction `R^2` about `0.77`, second about `0.06`), while noise=1.0 breaks pose
search entirely under a 1k simulator-rotation candidate set. Exact-pose-support
5k-image diagnostics separate the regimes:

1. noise=0.1 recovers from GT mean/random shell-power `W` when poses are nailed
   (`embedding R^2` about `[0.97, 0.82]`, assignment mean `R^2` about `0.90`,
   contrast `R^2` about `0.002`);
2. noise=1.0 remains weak even with exact-pose support and 5k images
   (`embedding R^2` about `[0.37, 0.001]`, assignment mean `R^2` about `0.20`);
3. the GT-model embedding upper bound at noise=1.0 is also weak
   (`embedding R^2` about `[0.32, 0.12]` at `current_size=64`), so that SNR is
   not a useful near-term PPCA refinement target for per-image embeddings
   without stronger preprocessing, more signal, or a different fixture.

## Known image-scale correction

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

Resolution/current-size growth is allowed only when all first-pass checks pass:

1. K-class schedule says growth is otherwise allowed;
2. current and previous best pose ids are stable under a strict fraction
   threshold;
3. halfset means are aligned before comparison;
4. the halfset mean FSC or supplied equivalent supports the proposed
   `current_size`;
5. no frame/sign/drift diagnostic fails;
6. pmax/logZ diagnostics are good enough, but never sufficient by themselves.

For this pass, "mean of halfset distributions" means the halfset mean volumes
`mu_half[0]` and `mu_half[1]`. Loading subspace agreement is reported as a
diagnostic where available, with sign flips treated as stable when the span and
scale are unchanged.

## First acceptance tests

The first deliverable stops after these CPU-safe gates:

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

## Deferred next tasks

1. Production halfset/FSC resolution controller using real aligned halfmaps.
2. End-to-end real-data exercise that switches from dense PPCA to exact-local
   PPCA when the bridge's `RefinementState.do_local_search` becomes true.
