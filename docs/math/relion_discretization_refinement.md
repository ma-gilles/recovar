# RELION Discretization And Refinement: Exact Behavior, recovar Status

Date: 2026-04-03

## Goal

This document is about *behavioral parity*, not about "good enough"
refinement quality.

For RELION auto-refine parity, the target is:

1. The same hidden-state grid at each iteration.
2. The same posterior weights over that grid.
3. The same sufficient statistics emitted by the E-step.
4. The same control decisions for `current_resolution`, `current_size`,
   angular refinement, local search, and convergence.
5. Only after that, a faster implementation.

If recovar gives a better-looking map or a faster FSC rise while using a
different grid, different priors, different residual term, or different
noise/tau updates, that is still a parity failure.

## Scope

This document covers the single-class, single-volume, homogeneous
`relion_refine --auto_refine` path used for the projected-image benchmark.

It focuses on:

- discretization of directions, in-plane angles, and translations
- adaptive oversampling
- local angular refinement
- masked vs unmasked image handling
- the per-iteration resolution / image-size control loop
- what recovar has already implemented
- what is still missing

## Source Basis

RELION references are from `/home/mg6942/myscratch/relion/src/`, mainly:

- `ml_optimiser.cpp`
- `ml_optimiser_mpi.cpp`
- `healpix_sampling.cpp`
- `backprojector.cpp`
- `projector.cpp`
- `fftw.h`

recovar references are mainly:

- `recovar/em/dense_single_volume/iteration_loop.py`
- `recovar/em/dense_single_volume/em_engine.py`
- `recovar/em/dense_single_volume/adaptive.py`
- `recovar/em/dense_single_volume/convergence.py`
- `recovar/em/sampling.py`
- `recovar/reconstruction/regularization.py`
- `recovar/reconstruction/relion_functions.py`

## RELION Refinement State

RELION's refinement loop is not just "run EM, then increase resolution".
It carries a set of state variables from one iteration to the next.

The most important ones for the discretization / search path are:

- `current_resolution`
- `current_size`
- `healpix_order`
- `psi` sampling implied by the current order and oversampling level
- translation search range and step
- `ave_Pmax`
- `incr_size`
- `has_high_fsc_at_limit`
- `nr_iter_wo_resol_gain`
- `nr_iter_wo_large_hidden_variable_changes`
- `has_fine_enough_angular_sampling`
- `do_use_all_data`

Two details matter a lot:

1. `current_size` for iteration `n` is derived from *iteration `n-1`* state.
2. `incr_size` and `has_high_fsc_at_limit` are sticky, monotone control
   variables in the RELION 5 source tree.

More specifically, in the MPI auto-refine path RELION raises `incr_size` to at
least `fsc0143 - fsc05 + 5`, and `has_high_fsc_at_limit` latches once FSC at
`current_size / 2 - 1` exceeds `0.2`. So parity needs the exact latch/update
rule, not just the idea that both helpers are "sticky."

## RELION Hidden-State Discretization

RELION searches over a Cartesian product of:

- direction / orientation samples
- in-plane `psi` samples
- translations

For single-particle refinement, the practical hidden state for one image is:

`(idir, ipsi, itrans)`

where:

- `idir` indexes the HEALPix direction grid
- `ipsi` indexes the in-plane angle grid associated with that direction level
- `itrans` indexes the translation grid

The important point is that RELION's "orientation index" is not just one flat
rotation matrix table in spirit. It is structured as:

- a direction lattice from HEALPix
- an in-plane lattice for `psi`
- optional oversampled child states of those coarse parents

That structure is important because adaptive oversampling and local search act
on the *parent coarse samples*, not on an unrelated flat list of arbitrary
rotation matrices.

In the actual RELION code this structure is explicit.

The coarse flattened hidden-state index is built as:

`ipos = iclass * NrDir * NrPsi * NrTrans + idir * NrPsi * NrTrans + ipsi * NrTrans + itrans`

in `HealpixSampling::getPositionSamplingPoint(...)`.

For oversampled children, RELION does not build a new unrelated global hidden
state. It expands a coarse parent position as:

`ipos_over = ipos * nr_over_orient * nr_over_trans + nr_over_trans * iover_rot + iover_trans`

in `HealpixSampling::getPositionOversampledSamplingPoint(...)`.

That means all bookkeeping in RELION is parent-relative:

- first choose a coarse `(idir, ipsi, itrans)`
- then enumerate oversampled orientation children `iover_rot`
- then enumerate oversampled translation children `iover_trans`

## RELION Direction Grid

The direction grid comes from HEALPix.

At a given order:

- RELION enumerates coarse direction pixels
- each pixel is paired with a set of `psi` values
- the total orientation table is therefore "direction x psi"

Two consequences:

1. Coarse adaptive significance is over *samples*, not just directions.
2. Oversampling is defined relative to the parent direction and parent `psi`
   bin, not by snapping to a separate global finer table.

## RELION `psi` Discretization

`psi` is not treated as an afterthought. It is a separate coarse lattice that
is refined during oversampling and local search.

The key RELION behavior is:

- oversampled `psi` children are midpoint samples inside the parent `psi` bin
- they are not simply "take the next global `psi` grid and use the nearest row"

This matters because a global finer table and midpoint children are not the
same discretization, even if they look numerically similar.

For parity, the child `psi` values must be generated from the parent coarse
sample exactly the way RELION does it.

In RELION's source this is `HealpixSampling::pushbackOversampledPsiAngles(...)`.
For oversampling order `k`, the number of `psi` children is `2^k`, and the
child centers are:

`psi_parent - 0.5 * psi_step + (0.5 + i) * psi_step / 2^k`

for `i = 0, ..., 2^k - 1`.

For 3D refinement, one coarse orientation sample therefore expands into:

- `4^k` HEALPix direction children
- `2^k` `psi` children
- total orientation oversampling factor `8^k`

## RELION Translation Grid

Translations are part of the hidden-state lattice, not a separate postprocess.

RELION:

- evaluates coarse translations in pass 1
- identifies significant coarse `(rotation, translation)` samples
- oversamples translations in pass 2 as well

This means a parity implementation cannot:

- prune only rotations and forget which translations were significant
- oversample only orientation while leaving translation on the coarse grid

The translation parent-child mapping has to be preserved through adaptive pass 2.

Another exact lattice detail is that RELION's coarse translation table is a
disk, not a full square box. In `HealpixSampling::setTranslations(...)` the SPA
loop keeps only offsets satisfying

`xoff^2 + yoff^2 <= offset_range^2 + 0.001`

So for `--offset_range 3 --offset_step 1`, the coarse grid has `29`
translations, not `49`. recovar's `get_translation_grid()` already matches
that disk-shaped sampling.

RELION also ties the *translation-prior width* in the score path to the active
search range. In `convertAllSquaredDifferencesToWeights(...)`:

- if `offset_range_x > 0`, it sets
  `sigma_offset_x^2 = offset_range_x^2 / 9`
- likewise for y and z
- only when there is no explicit search range does it fall back to
  `mymodel.sigma2_offset`

So during a global search with `--offset_range 3 --offset_step 1`, the score
path effectively uses a 3-sigma Gaussian with `sigma = 1 px`, not simply the
`--offset` parameter value. That exact branch is part of RELION parity.

RELION's translation children are also midpoint samples inside the parent cell,
generated by `HealpixSampling::getTranslationsInPixel(...)`.

For 2D refinement and oversampling order `k`:

- there are `2^k` x-children
- there are `2^k` y-children
- total translation oversampling factor is `4^k`

The child enumeration order matters for exact indexing and comparison. In the
RELION loops, x is the outer loop and y is the inner loop.

## RELION Global Search

Early in refinement, RELION performs a global search:

- coarse direction grid
- coarse `psi` grid
- coarse translation grid
- full posterior over all coarse hidden states

This search is still subject to:

- masked-image likelihood
- the current `current_size`
- the current noise spectrum
- the current tau prior
- the current scale / norm state

So "global search" does not mean "uninformative posterior". A flat posterior
in the first iteration is usually evidence that the scoring path is wrong.

There is also an important prior detail here that is easy to miss from the
high-level descriptions. Even when RELION is in `orientational_prior_mode ==
NOPRIOR`, the score path is not literally flat over directions. It still uses
the learned per-class directional distribution `mymodel.pdf_direction[iclass]`
inside `convertAllSquaredDifferencesToWeights(...)`, then mean-normalizes that
distribution over the currently searched orientation subset. In other words,
"no local orientation prior" in RELION still includes a learned global
direction prior. recovar currently does not carry an equivalent
posterior-weighted `pdf_direction` update outside local search.

## RELION Adaptive Oversampling

This is the core search refinement step.

RELION's logic is:

1. Evaluate all coarse hidden states.
2. Convert diff2 to posterior weights.
3. For each particle / metadata row, sort positive coarse weights, walk down
   until cumulative weight exceeds `adaptive_fraction * exp_sum_weight`, apply
   `maximum_significants`, and then keep every coarse sample with weight at
   least that threshold.
4. Expand only those coarse parents into oversampled child states.
5. Re-evaluate only those children at finer angular / translational sampling.

Important parity properties:

- in the SPA benchmark, significance is effectively per image because there is
  one image per particle; in the general RELION code it is tracked per
  `part_id` / metadata row
- significance is over flattened `(rotation, translation)` samples
- pass 2 children depend on the exact parent coarse sample set for that image
- pass 2 may therefore have a different candidate count for each image
- tie handling matters: the retained set is "thresholded and capped," not
  literally the mathematically smallest set reaching the target mass

One benchmark-specific detail matters here. The projected-image RELION
reference run in
`/scratch/gpfs/GILLES/mg6942/tmp/em_profile/data/relion_ref/run_optimiser.star`
stores:

- `_rlnAdaptiveOversampleFraction = 0.999`
- `_rlnMaximumSignificantPoses = -1`

In RELION, `maximum_significants <= 0` means "uncapped". Earlier recovar
benchmark runs were still forcing `max_significants=500`, so the observed
`nr_significant_samples` saturation at `500` was partly a harness mismatch
rather than pure posterior mismatch.

Another exact detail is that, when adaptive oversampling is enabled and
`strict_highres_exp <= 0`, RELION uses:

- `image_coarse_size` in pass 1
- `image_current_size` in pass 2

This switch is wired into the expectation loop, not bolted on later. In the
source tree that behavior is visible both in
`MlOptimiser::updateImageSizeAndResolutionPointers()` and in the expectation
path that chooses coarse vs fine image windows depending on
`exp_current_oversampling`.

There is one more nuance here: the scoring/precalculation path may use
`image_coarse_size`, but weighted sums for reconstruction are always prepared on
`image_current_size`. With `strict_highres_exp > 0`, both scoring passes stay on
`image_coarse_size`.

RELION also evaluates the translation prior only on the coarse translation
lattice and then reuses that coarse `pdf_offset` value for all oversampled
translation children. That is an intentional RELION approximation, so a more
"exact" per-child translation prior would actually break parity.

The coarse-grid translation prior is then mean-normalized over the sampled
coarse translation lattice before it is multiplied into the hidden-state
weight. That mean normalization is constant for one metadata row, so it does
not change the within-row posterior shape, but it *does* matter for exact
log-evidence / metadata parity.

This is why a union-of-significant-rotations approximation is not parity,
even if it is much easier to batch.

## RELION Local Search

Auto-refine often enters local search only after global refinement has
stabilized, but that is not an absolute rule. If the starting HEALPix order is
already at or above RELION's local-search threshold, local-search mode can be
active from iteration 0.

Local search is not "take a smaller global grid". It is:

- centered on the previous best pose for each particle / metadata row
- governed by Gaussian priors in orientation space
- still evaluated on RELION's HEALPix / `psi` child geometry

The previous best hidden state matters because each image gets its own local
neighborhood.

For parity, the implementation must carry:

- the exact previous best pose per image
- the exact local prior width / cutoff behavior
- the exact child-generation semantics from that pose
- sigma-cutoff truncation of the prior support
- separate normalization of direction and `psi` priors
- symmetry-mate handling
- nearest-sample fallback when the truncated set would otherwise be empty

Using one shared union of local candidates for many images is a performance
approximation, not exact RELION behavior.

The trigger for entering finer angular sampling and eventually local searches is
also not arbitrary. RELION estimates angular and translational accuracies in
`MlOptimiser::calculateExpectedAngularErrors(...)`, then uses those estimates in
`MlOptimiser::updateAngularSampling(...)` to decide whether to:

- keep the current exhaustive grid
- refine the HEALPix order / translation step
- or, once the configured threshold is reached, switch into local-search mode

So local search is coupled to:

- estimated angular accuracy
- estimated translational accuracy
- the number of iterations without resolution gain
- the number of iterations without large hidden-variable changes

Matching only the local grid geometry without matching those trigger conditions
is still not full RELION parity.

## RELION Masked Scoring vs Unmasked Reconstruction

RELION explicitly uses two different image paths in one iteration:

1. Unmasked Fourier images for reconstruction.
2. Masked or noise-filled Fourier images for scoring and noise estimation.

This split lives in `getFourierTransformsAndCtfs()`:

- `exp_Fimg_nomask`: unmasked image FT
- `exp_Fimg`: masked image FT

The consequence is:

- posterior weights are computed from masked data
- backprojection is accumulated from unmasked data

RELION also couples a third bookkeeping path to the masked images:

- shell-wise masked residual power for `sigma2_noise`
- masked-image high-frequency residual power above `current_size`
- full masked-image shell spectrum `exp_power_img` used to extend
  `sigma2_noise` beyond the current Fourier window

This is a large correctness detail. If one code path uses the same image for
both, both the posterior and the reconstruction drift from RELION.

The mask geometry itself also matters. RELION does not use a generic
"soft circular window near the box edge". In the SPA auto-refine path it
applies `softMaskOutsideMap(img, particle_diameter / (2 * angpix),
width_mask_edge)` in real space. For the projected-image benchmark here,
that means:

- `particle_diameter = 200 A`
- `angpix = 4.25 A/px`
- mask radius `= 23.53 px`
- `width_mask_edge = 5 px`

recovar's default image backend mask was previously a generic
`window_mask(D, 0.85, 0.99)`, which for `D=128` keeps signal out to roughly
`54 px` radius and therefore includes far more background than RELION in the
likelihood path. The benchmark harness now overrides the dataset scoring mask
with the RELION-style particle-diameter mask before refinement starts.
General recovar RELION mode still needs a clean way to source these mask
parameters outside the benchmark harness.

## RELION `highres_Xi2`

When `current_size < ori_size`, RELION does not simply discard high-frequency
image power.

Instead it:

- computes the masked-image Fourier power above the optics-group remap of
  `current_size` on the image grid
- stores it as `exp_highres_Xi2_img`
- adds `exp_highres_Xi2_img / 2` to every candidate `diff2`
- and also stores the full masked-image per-shell spectrum `exp_power_img` for
  later noise bookkeeping

This keeps the likelihood normalized correctly while only the low-frequency
window participates in orientation discrimination.

That term is one reason RELION can restrict search to low frequencies without
changing the total log-likelihood inconsistently.

## RELION `Pmax` And Significant Samples

After evaluating candidate diff2 values, RELION converts them to posterior
weights and stores:

- total posterior mass
- significant coarse sample set
- `Pmax` per particle / metadata row

In the SPA projected benchmark that reduces to one image per particle, so the
per-row and per-image views coincide.

The exact RELION bookkeeping is more specific than that summary:

1. `MlOptimiser::convertAllSquaredDifferencesToWeights(...)` first constructs
   coarse-grid priors on orientation and translation.
2. It normalizes those priors by their coarse-grid means:
   - `pdf_orientation /= pdf_orientation_mean`
   - `pdf_offset /= pdf_offset_mean`
3. It then forms the unnormalized hidden-state weight
   `pdf_orientation * pdf_offset * exp(-(diff2 - exp_min_diff2))`.
4. It sums those raw weights into `exp_sum_weight`.
5. It sorts the positive weights, walks down from largest to smallest until the
   cumulative sum exceeds `adaptive_fraction * exp_sum_weight`, and only then
   applies `maximum_significants` if `maximum_significants > 0`.
6. The threshold is therefore a weight value, not a fixed count. RELION keeps
   every sample with weight `>= significant_weight`, so ties at the threshold
   are intentionally included.

That exact "threshold-and-tie" behavior matters. It is why parity code should
not think in terms of "keep the smallest top-k set" unless it also reproduces
RELION's threshold semantics.

There is also an important distinction between raw and normalized weights:

- significance is decided on the raw weights
- inside `storeWeightedSums(...)`, RELION then divides each retained weight by
  `exp_sum_weight`
- `METADATA_PMAX` stores the maximum of those normalized retained weights

So in the single-particle SPA case:

`Pmax = exp_max_weight / exp_sum_weight`

where both quantities already include the orientation and translation priors.

RELION also stores the per-particle log-likelihood contribution as

`dLL = log(exp_sum_weight) - exp_min_diff2 - logsigma2`

where `logsigma2` is the shell-summed Gaussian normalization term for the
current masked-image Fourier window. That is another piece of metadata parity
that depends on the exact same score / noise bookkeeping.

`Pmax` is not just a reporting metric. In RELION 5 its direct control-flow role
is image-size growth:

- image-size growth

Its effect on convergence or angular refinement is indirect, through the fact
that image-size growth changes the downstream refinement trajectory.

If `Pmax` is wrong, `current_size` can still look right for one iteration by
accident while the rest of the refinement trajectory is still wrong.

`ave_Pmax` is also source-coupled to that same metadata path. RELION does not
recompute it from a separate summary pass. In `storeWeightedSums(...)` it adds
the metadata-row `Pmax` values into `wsum_model.ave_Pmax`, and in
`maximizationOtherParameters()` it divides by the total row weight sum. For the
single-particle benchmark that reduces to "average over all particle rows", but
the implementation still matters because it uses the exact same winning-state
metadata as the rest of the EM loop.

## RELION Weighted Sums And Shell Statistics

The central structural fact about RELION is that search, metadata updates, and
M-step statistics are coupled inside one weighted pass.

In `MlOptimiser::storeWeightedSums(...)`, every retained hidden state
contributes its normalized posterior weight to all of the following at once:

- masked-image residual power per shell for `sigma2_noise`
- high-resolution residual fill via `exp_power_img` above `current_size`
- unmasked weighted image sums for reconstruction
- weighted CTF sums / reconstruction weights
- class and direction prior accumulators
- translation-prior accumulators and `sigma2_offset`
- `Pmax` and best-pose metadata
- `dLL` metadata
- scale and norm correction sums

This is why RELION's shell statistics stay self-consistent. The same posterior
weights that decide the best pose also drive:

- `sigma2_noise`
- `tau2`
- `data_vs_prior`
- `current_resolution`
- next iteration's `current_size`

If recovar reconstructs only `Ft_y` and `Ft_ctf` during the E/M pass, then
estimates noise or tau2 later from hard assignments or a different batch path,
that is not a small approximation. It changes the actual sufficient statistics.

## Numerical Stability In recovar's Score Path

One recovar-specific failure mode turned out to be purely numerical rather than
conceptual.

The Gaussian negative log-likelihood for one image/candidate contains a term
proportional to:

- `||y||^2 / sigma^2`

That term is:

- image-specific
- constant across all `(rotation, psi, translation)` candidates for that image

So mathematically it cancels out of:

- posterior weights
- `Pmax`
- significant-sample pruning
- hard assignments

However, if that constant is inserted directly into every float32 candidate
score, it can dominate the rotation-dependent terms by many orders of
magnitude. That is exactly what happened in recovar's benchmark path:

- the constant score offset was around `1e8`
- the candidate-to-candidate differences were tiny by comparison
- float32 rounded all candidates to the same score
- the posterior collapsed to nearly uniform

The correct implementation strategy is:

1. omit image-constant score terms from the relative E-step score tensor
2. do logsumexp / softmax / significance on the relative scores only
3. add the constant back *only* when reporting absolute log-evidence values

As of 2026-04-03, recovar now follows that strategy in
`em_engine.py`. This was a major prerequisite for any meaningful adaptive
search comparison, because before that fix the code could not even represent
the posterior sharply enough for significance pruning to work.

## RELION Resolution And Image-Size Loop

The control path is:

1. Reconstruction and FSC update.
2. `data_vs_prior_class` update from FSC-derived SSNR in split-half auto-refine.
3. `current_resolution` update from the shell where `data_vs_prior` drops below 1.
4. Next iteration's `current_size` update from:
   - previous `current_resolution`
   - previous `ave_Pmax`
   - sticky `incr_size`
   - sticky `has_high_fsc_at_limit`

The important subtlety is that `current_size` is a lagged control variable.
The size used for iteration `n` is based on the statistics from iteration `n-1`.

## What recovar Now Matches

As of 2026-04-03, recovar has real progress on the discretization / refinement
side.

### Implemented or close to implemented

- Half-spectrum GEMM E-step / M-step engine.
- Fourier-windowed low-resolution search.
- RELION-style `current_size` bootstrap for the benchmark case.
- RELION-mode coarse-grid initialization now regenerates pass-1 rotations from
  the coarse `healpix_order` state instead of accidentally inheriting a finer
  caller-supplied grid.
- RELION-mode adaptive pass 1 now uses the actual RELION particle diameter for
  `image_coarse_size` instead of falling back to the full box size. On the 128px
  projected benchmark that changes the order-2 coarse size from about `20 px`
  to the RELION-like `52 px`.
- Sticky `incr_size` / `has_high_fsc_at_limit` growth-state handling.
- Padded reconstruction path with `padding_factor=2`.
- Oversampled `psi` midpoint children instead of pretending pass-2 children are
  just rows of the next global table.
- Exact sparse adaptive pass 2 over per-image significant coarse samples.
- Exact dense oversampled pass when every coarse sample is significant.
- RELION-mode skip of adaptive pass 2 when the significant fraction is high.
- RELION-style uncapped significant-sample mode (`max_significants <= 0`).
- RELION-mode per-image `Pmax` plumbing from `run_em`.
- RELION-style translation prior in the score path, including reuse of the
  coarse translation prior on oversampled translation children.
- Masked-scoring / unmasked-reconstruction split in `em_engine.run_em()`.
- Numerical-stability fix for E-step scores: image-constant `||y||^2` terms are
  omitted from relative scores and added back only for absolute log-evidence
  reporting.
- RELION-style initial `sigma2_noise` estimation in the benchmark harness from
  particle power spectra instead of a flat unit spectrum.
- RELION-mode `ave_Pmax` / convergence bookkeeping now aggregates both half-sets
  instead of only half-set 0.
- RELION-mode now feeds back a learned global direction prior outside local
  search, using coarse-grid posterior mass accumulated from the active
  orientation grid rather than oversampled child-grid indices.
- Clean focused validation on a free GPU for the RELION-mode / noise /
  regularization stack.

### Important benchmark improvements already observed

On the 5k-image projected benchmark with RELION half-set split:

- iteration-1 `current_size` now matches RELION at `56`
- the coarse translation grid size matches RELION's disk sampling at `29`
  offsets for `offset_range=3`, `offset_step=1`
- the RELION-mode loop now starts pass 1 on the intended coarse HEALPix order
  instead of accidentally starting on the oversampled fine order
- after the 2026-04-03 score-stability fix, the patched real-size run stopped
  treating every coarse candidate as equally probable: half-set 0 dropped to
  `23088 / 36864` significant coarse rotations with median
  `nr_significant_samples/image = 500` instead of keeping all
  `36864 * 29 = 1,069,056` coarse samples per image
- after switching the benchmark harness from a flat initial noise spectrum to a
  RELION-style particle-spectrum estimate, iteration-1 pass 1 still did not
  match RELION, but it remained in the same non-degenerate regime: the coarse
  posterior was pruned instead of collapsing to the full grid
- after removing the non-RELION `max_significants=500` cap and rerunning the
  realistic 5k-image benchmark, half-set 0 pass 1 dropped to
  `570 / 4608` significant coarse rotations with median
  `nr_significant_samples/image = 4`, which is much closer to RELION's
  iteration-1 distribution than the old capped benchmark
- after enabling the learned global direction prior and rerunning for two
  iterations, iteration 2 now exercises the intended RELION-like prior path,
  but it also shows the remaining blocker clearly: once recovar's shell
  statistics drive `current_size` down to `26` instead of RELION's `30`, the
  search broadens again and sparse pass 2 effectively becomes dense
  (`median local rotations ≈ 4032`, `median valid candidates/image ≈ 456k`)

That means the top-level control flow and parts of the discretization path are
closer than they were at the start of this effort.

## What recovar Still Does Not Match

The remaining blockers are now more about *posterior correctness and sufficient
statistics* than about gross control flow.

### 1. Posterior weights are still not RELION-correct

Observed on the benchmark:

- before the 2026-04-03 stability fix, our iteration-1 `Pmax` effectively
  collapsed to the uniform `1 / (n_rot * n_trans)` limit because float32 lost
  all candidate score differences
- before the 2026-04-03 coarse-grid fix, pass 1 was also incorrectly running on
  the fine order-3 grid while the state machine claimed coarse order 2
- before the 2026-04-03 particle-diameter fix, pass 1 also computed
  `image_coarse_size` from the full box width instead of RELION's actual
  particle diameter, which made the coarse pass much blurrier than RELION
- after those fixes, the posterior is no longer exactly uniform, but it still
  does not match RELION's shape
- before the uncapped benchmark rerun, the comparison was polluted by a
  non-RELION `max_significants=500` harness cap
- after enabling RELION-style uncapped significance, the median significant
  count moved dramatically closer to RELION, so the remaining mismatch is
  smaller and more clearly about the score/statistics path itself
- in the latest uncapped run, our iteration-1 mean `Pmax` is actually larger
  than RELION's (`0.8433` vs `0.5901`), so the remaining mismatch is not simply
  "too flat" anymore
- in that same uncapped run, our `nr_significant_samples` distribution is
  `min/median/mean/max = 1 / 4 / 5.92 / 53`, while RELION's iteration-1
  reference is `1 / 7 / 8.63 / 44`
- in the 2-iteration run with learned global direction priors enabled, the
  search path is correct enough to expose the next blocker: iteration 2
  `current_size` drops to `26` vs RELION `30`, `nr_significant_samples`
  jumps to `min/median/mean/max = 14199 / 14256 / 14267 / 14322` vs RELION
  `8920 / 8920 / 8874 / 8920`, and `ave_Pmax` collapses to `0.0000` vs
  RELION `0.0002`

That means the hidden-state weights themselves are still wrong, even when the
search lattice and `current_size` are closer.

### 2. `highres_Xi2` is still only partially matched

recovar now scores masked images and therefore carries the masked-image norm
through the E-step, which is directionally correct.

But it still does *not* emit RELION's explicit:

- `exp_power_img`
- `exp_highres_Xi2_img`
- weighted high-frequency residual accumulators

So the bookkeeping is still not identical to RELION, especially for the noise
update path.

### 3. Noise estimation is still not RELION's weighted residual update

RELION uses posterior-weighted masked residual spectra accumulated during the
E-step.

recovar still does not emit the equivalent shell accumulators directly from the
engine, so `sigma2_noise` is not yet driven by the same statistics.

### 4. Tau2 / `data_vs_prior` are still downstream approximations

RELION's split-half auto-refine path sets:

- `data_vs_prior = FSC / (1 - FSC)` up to whole-map correction
- `tau2 = data_vs_prior * sigma2_out`

with `sigma2_out` coming from actual reconstruction weights.

recovar still does not fully rebuild that path from the exact reconstruction
statistics emitted during the current iteration.

### 5. Local search is not yet full RELION parity

The code now has much better scaffolding for per-image local search, but full
parity still requires:

- exact per-image parent-state carryover
- exact per-image prior application
- exact RELION child-generation semantics for local neighborhoods
- exact RELION trigger conditions from angular-accuracy / convergence logic
- exact metadata / convergence coupling

### 6. Global orientation and translation priors are still partial rather than exact

recovar now carries a RELION-style translation prior in the hidden-state score
path and reuses the coarse prior value on oversampled translation children, as
RELION does. It also matches RELION's score-path search-range branch where a
finite translation search uses `sigma_offset = offset_range / 3`.

What is still missing is the full RELION coupling between:

- RELION's damped / normalized `pdf_direction` update outside local search
- prior normalization on the coarse grid
- metadata-carried previous offsets and priors
- weighted prior updates in the same `storeWeightedSums()` path
- downstream convergence and reporting fields

## Status Matrix

### Discretization / search path

- Direction grid from HEALPix semantics: partial
- `psi` midpoint child generation: implemented
- Parent-relative hidden-state indexing semantics: partial
- Translation oversampling in pass 2: implemented
- Disk-shaped coarse translation lattice: implemented
- Per-image significant coarse samples: implemented
- Per-image sparse adaptive pass 2: implemented
- RELION uncapped `maximum_significants <= 0`: implemented
- Exact per-image local search parity: partial
- Global direction prior parity outside local search: partial
- Translation prior parity: partial

### Scoring path

- Masked scoring images: implemented
- Unmasked reconstruction images: implemented
- RELION particle-diameter scoring mask in benchmark harness: implemented
- Image-constant score-term handling: implemented
- RELION-style initial sigma2 estimate in benchmark harness: implemented
- RELION-style translation prior in score path: partial / implemented for the
  benchmarked SPA path
- Explicit `highres_Xi2` bookkeeping: partial
- Exact RELION diff2 / prior path: partial
- Exact `Pmax` / significant-sample parity: partial / still missing in practice
- Exact `dLL` / log-likelihood metadata parity: missing

### M-step / control path

- Sticky image-size growth state: implemented
- Exact `sigma2_noise` weighted residual path: missing
- Exact FSC-derived `data_vs_prior`: partial
- Exact FSC-derived tau2 from reconstruction weights: partial
- Final joined-halves RELION iteration: missing

## recovar File Map

### `recovar/em/dense_single_volume/em_engine.py`

This is the critical fast path.

It currently handles:

- blockwise score normalization
- half-spectrum projections
- optional windowed low-resolution scoring
- masked scoring and unmasked reconstruction split
- numerically stable relative-score evaluation for `Pmax` / softmax
- per-image posterior-max statistics

It does *not yet* emit all RELION sufficient statistics, especially:

- weighted residual shell sums for noise
- explicit high-frequency tail statistics
- scale and norm accumulators
- weighted direction sums needed for RELION's learned global `pdf_direction`

### `recovar/em/dense_single_volume/iteration_loop.py`

This is the RELION-mode control loop.

It currently handles:

- `current_size` bootstrap and updates
- adaptive pass 1 / pass 2 control
- local-search scaffolding
- convergence-state updates

It still approximates parts of RELION because the engine does not yet return
all necessary statistics.

### `recovar/em/dense_single_volume/adaptive.py`

This is where adaptive oversampling now most closely mirrors RELION.

It currently supports:

- exact sparse pass 2 over per-image significant coarse samples
- dense exact pass when all coarse samples are significant
- translation oversampling
- RELION-style child orientation generation from parent samples

### `recovar/em/sampling.py`

This file contains the orientation-grid semantics.

This is where the important "RELION child sample geometry" logic lives:

- parent sample decomposition
- oversampled child generation
- midpoint `psi` handling
- mapping between flat indices and orientation matrices

## Immediate Next Work

The next highest-leverage parity tasks are:

1. Make the E-step emit exact weighted residual shell accumulators.
2. Emit explicit `highres_Xi2` / `exp_power_img`-equivalent statistics.
3. Rebuild `sigma2_noise` from those weighted residuals.
4. Rebuild `tau2` and `data_vs_prior` from actual reconstruction weights and FSC.
5. Then revisit `Pmax` and significant-sample parity with the corrected stats path.

At this point the main mismatch is no longer "wrong `current_size` logic".
It is "wrong posterior weights and downstream shell statistics".

## Practical Interpretation

If a future run shows:

- `current_size` matches RELION
- but `Pmax` is near zero
- and median significant samples are huge

then the problem is almost certainly in one of:

- masked/noise likelihood bookkeeping
- priors
- residual normalization
- noise spectrum
- tau / `data_vs_prior` feedback

not in the HEALPix discretization itself.

That is the current situation.
