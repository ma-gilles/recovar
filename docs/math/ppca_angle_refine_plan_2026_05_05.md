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
7. `tests/unit/ppca_refinement/test_kclass_fixture_validation.py` verifies the
   real-fixture validator on RELION-frame K-class MRCs and RECOVAR-frame GT
   MRCs.
8. `scripts/run_ppca_dense_fixture_smoke.py` verifies a capped real-fixture
   dense PPCA E/M iteration produces finite `mu/W`, pose diagnostics, and best
   pose arrays from K-class-initialized PPCA.

## Deferred next tasks

1. Production halfset/FSC resolution controller using real aligned halfmaps.
2. End-to-end real-data exercise that switches from dense PPCA to exact-local
   PPCA when the bridge's `RefinementState.do_local_search` becomes true.
