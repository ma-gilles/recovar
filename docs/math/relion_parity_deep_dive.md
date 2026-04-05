# RELION Auto-Refine Parity Deep Dive

Date: 2026-04-03

See also:

- `docs/math/relion_discretization_refinement.md` for the detailed
  discretization / adaptive-refinement / local-search behavior map.

Current high-level status as of 2026-04-03:

- Iteration-1 `current_size` on the 5k projected benchmark now matches
  RELION at `56`.
- The adaptive search path is materially closer to RELION than before:
  sparse per-image pass 2 exists, dense-exact pass 2 exists when all coarse
  samples are significant, and RELION-mode can skip pass 2 when it is pure
  overhead.
- The E-step score path no longer adds the image-constant `||y||^2 / sigma^2`
  term into every float32 candidate score. That constant now stays outside the
  relative softmax scores and is only added back for absolute log-evidence
  reporting, which fixed a major posterior-flattening precision bug.
- The benchmark harness now seeds iteration 1 with a RELION-style initial
  `sigma2_noise` estimate from particle power spectra rather than a flat unit
  spectrum.
- The benchmark harness and significance code now support RELION's uncapped
  `_rlnMaximumSignificantPoses = -1` mode instead of forcing
  `max_significants=500`. On the realistic 5k-image rerun, iteration-1 pass 1
  for half-set 0 dropped to `570 / 4608` significant coarse rotations with
  median `nr_significant_samples/image = 4`, which is much closer to RELION's
  iteration-1 distribution than the earlier capped comparison.
- RELION-mode now uses the same search-range-driven translation-prior width as
  RELION's score path: when a finite offset search range is active, the Gaussian
  prior width is `offset_range / 3` rather than the raw `--offset` value.
- RELION-mode now aggregates `ave_Pmax` and convergence bookkeeping across both
  half-sets instead of incorrectly using only half-set 0.
- RELION-mode now feeds back a learned global direction prior outside local
  search, using coarse-grid posterior mass accumulated from the active
  orientation grid rather than from oversampled child indices.
- The main remaining blocker is the scoring / statistics path:
  the search lattice is much closer, but the weighted residual / tau /
  `data_vs_prior` path is still not exact RELION, and on the latest uncapped
  benchmark the posterior is now slightly too sharp rather than too flat
  (`Pmax` above RELION while the final map agreement is still poor).
- On the completed uncapped 5k-image iteration-1 benchmark, recovar is now in
  the right search regime but still not in the right posterior regime:
  `ave_Pmax = 0.8433` vs RELION `0.5901`, and
  `nr_significant_samples` is `min/median/mean/max = 1 / 4 / 5.92 / 53` vs
  RELION `1 / 7 / 8.63 / 44`.
- On the completed 2-iteration benchmark with the learned global direction
  prior enabled, iteration 2 now follows the intended code path, but the run
  demonstrates that the remaining blocker is upstream of the search lattice:
  recovar chooses `current_size = 26` vs RELION `30`, iteration-2
  `nr_significant_samples` blows back up to `14267` mean vs RELION `8874`, and
  `ave_Pmax` collapses to `0.0000` vs RELION `0.0002`. That is consistent with
  the weighted shell-statistics / noise / tau path still being wrong enough to
  poison the next iteration's search state.

## Goal

The goal is not "better than RELION on this dataset" and not "similar overall
trajectory." The goal is:

1. Reproduce RELION 5 `relion_refine --auto_refine` behavior for single-class,
   single-volume homogeneous refinement as closely as possible.
2. Match the per-iteration control flow, sufficient statistics, per-image
   metadata, spectral updates, and final maps.
3. Only after that, optimize the implementation so it is faster and/or uses
   less memory than RELION while preserving the same outputs.

If recovar resolves earlier, gets a smoother FSC, or looks "better" than
RELION before matching RELION's internal behavior, that is still a parity
failure.

## Audit Basis

This audit was built from:

- `docs/math/plan_relion_parity_v2.md`
- `docs/math/prior_noise_audit.md`
- `/scratch/gpfs/GILLES/mg6942/tmp/em_relion_proj/audit_report.md`
- `/scratch/gpfs/GILLES/mg6942/relion5_auto_refine_algorithm.md`
- RELION 5 source in `/home/mg6942/myscratch/relion/src/`
- The recovar RELION-mode path centered on:
  - `recovar/em/dense_single_volume/refine.py`
  - `recovar/em/dense_single_volume/engine_v2.py`
  - `recovar/em/dense_single_volume/adaptive.py`
  - `recovar/em/dense_single_volume/convergence.py`
  - `recovar/reconstruction/regularization.py`
  - `recovar/reconstruction/noise.py`
  - `recovar/reconstruction/relion_functions.py`

The comparison target is the clean `origin/claude/em-relion-parity` logic, not
the local experimental fixes that may exist in other working trees.

## Function Map

| RELION function | Role in RELION | recovar counterpart | Status | Main mismatch |
| --- | --- | --- | --- | --- |
| `MlOptimiser::iterate()` | Top-level auto-refine loop | `refine.py::_refine_relion_mode()` | Partial | Same overall loop shape, but final joined-halves iteration, metadata propagation, and exact convergence behavior are missing. |
| `updateCurrentResolution()` | Convert `data_vs_prior` to current resolution | `regularization.compute_data_vs_prior()` + `resolution_from_data_vs_prior()` | Partial | RELION uses true reconstruction weights from the backprojector; we use shell averages from our accumulated `Ft_ctf`, which are only exact if upstream stats are exact. |
| `updateImageSizeAndResolutionPointers()` | Grow `current_size`, set coarse/fine image sizes | `compute_current_size_relion()` + `compute_coarse_image_size()` + `quantize_current_size()` | Partial | Growth rule exists, but quantization is still not exact RELION behavior and the final `do_use_all_data` Nyquist iteration is missing. |
| `getFourierTransformsAndCtfs()` | Image preprocessing, masking, priors, high-res residual term | `engine_v2._preprocess_batch()` plus parts of `refine.py` | Partial | recovar now has masked-scoring and unmasked-reconstruction paths, the benchmark harness now applies RELION's particle-diameter mask geometry, and RELION mode now uses the actual particle diameter in the adaptive coarse-size formula. It still does not emit RELION's explicit `exp_power_img` / `exp_highres_Xi2_img` bookkeeping or the full metadata-driven prior state. |
| `precalculateShiftedImagesCtfsAndInvSigma2s()` | Shifted masked/unmasked images, resized CTFs, inverse sigma2 | No direct equivalent | Missing | recovar recomputes a simpler path on the fly and does not expose the same sufficient statistics. |
| `getAllSquaredDifferences()` | Compute likelihood terms over hidden states | `engine_v2.run_em_v2()` and `refine._compute_significance_batched()` | Partial | No exact `highres_Xi2`; current-size truncation differs; benchmark mask geometry is now closer, but the general RELION-mode path still does not source RELION mask parameters automatically and the exact score scale still differs. |
| `convertAllSquaredDifferencesToWeights()` | Convert diff2 to posterior weights, keep significant samples, set `Pmax` | `run_em_v2()` softmax plus `adaptive.find_significant_mask()` | Partial | recovar now persists per-image `Pmax`, supports uncapped significance, carries RELION-style translation priors in the score path, and reuses a learned global direction prior outside local search. It still lacks RELION's exact prior damping/normalization and the full metadata-coupled prior update path. |
| `storeWeightedSums()` | Weighted backprojection, noise accumulation, scale/norm stats, metadata update | `run_em_v2()` plus post-hoc updates in `refine.py` | Partial | RELION accumulates many coupled statistics in one pass: masked residual spectra, unmasked backprojection, `Pmax`, `dLL`, scale/norm sums, and prior updates. recovar still reconstructs mainly `Ft_y` and `Ft_ctf`, then approximates the rest later. |
| `maximizationOtherParameters()` | Update `sigma2_noise`, `ave_Pmax`, scale, norm, priors | `noise.estimate_noise_level_no_masks()` + `compute_relion_prior()` + convergence state updates | Missing/partial | `ave_Pmax` is now aggregated from both half-sets, but noise, scale, norm, and prior updates are still not derived from the same weighted sums RELION uses. |
| `BackProjector::updateSSNRarrays()` | Update `tau2`, `sigma2_ref`, `data_vs_prior`, Fourier coverage | `compute_relion_prior()` and `compute_data_vs_prior()` | Partial | recovar uses different denominators and does not currently mirror the exact shell statistics from the actual reconstruction weights. |
| `BackProjector::reconstruct()` | Apply Wiener term and reconstruct at pad 2 | `relion_functions.post_process_from_filter_v2()` | Partial | The solver is close, but parity still depends on exact padding, exact input weights, and exact tau2. |
| `updateAngularSampling()` | Refine angular and translational search grids | `convergence.refine_angular_sampling()` | Partial | Broadly similar formulas, but our stability signals are approximate and local search behavior is not per-image. |
| `checkConvergence()` | Decide final joined-halves iteration | `convergence.check_convergence()` | Partial | Our convergence counters are based on simplified assignment changes and we do not execute RELION's final "all data to Nyquist" iteration. |

## What recovar already has

These pieces are good building blocks and should be preserved:

- A fast half-spectrum GEMM engine in `engine_v2.py`.
- Coordinate-preserving Fourier windowing for `current_size`.
- RELION-style reconstruction helpers in `relion_functions.py`.
- A RELION-mode top-level loop in `_refine_relion_mode()`.
- Local-search and adaptive-oversampling scaffolding.
- RELION reference extraction utilities in `scripts/extract_relion_reference.py`.
- Comparison helpers in `tests/integration/test_relion_comparison.py`.

The problem is not the absence of a RELION-mode scaffold. The problem is that
the scaffold does not yet preserve the same sufficient statistics and control
signals that RELION uses internally.

## Current Parity Blockers

### 1. `run_em_v2()` does not emit the statistics RELION actually optimizes

RELION's `storeWeightedSums()` simultaneously accumulates:

- weighted image backprojections from unmasked images
- weighted residual power for `sigma2_noise`
- per-group scale correction numerators and denominators
- norm-correction terms
- `Pmax`
- per-image best pose metadata
- class and direction weighted sums

Our `run_em_v2()` mainly returns:

- `Ft_y`
- `Ft_ctf`
- hard assignments

Everything else is approximated afterward from a different computation path.
This is the single biggest structural reason we do not match RELION.

### 2. The masked/unmasked split is implemented, but `highres_Xi2` bookkeeping is still missing

RELION explicitly uses:

- masked image Fourier data for alignment and likelihood
- unmasked image Fourier data for reconstruction

recovar now mirrors that split in `engine_v2.py`, which was a necessary parity
step. The remaining mismatch is that RELION also stores:

- `exp_power_img`
- `exp_highres_Xi2_img`

and folds `exp_highres_Xi2_img / 2` into every candidate diff2. recovar still
truncates the search window without emitting the same explicit high-frequency
tail statistics.

### 3. Adaptive oversampling is much closer, but not the whole story

RELION significance is per image and per hidden sample:

- coarse pass over all coarse states
- find significant `(rotation, translation)` samples per image
- generate oversampled children only for each image's significant coarse states

recovar currently does all of the following instead:

- keeps per-image significant coarse `(rotation, translation)` samples
- can run an exact sparse per-image pass 2
- can run a dense exact oversampled pass when every coarse sample is significant
- supports uncapped `maximum_significants <= 0` mode

The remaining gaps are:

- local search is still not exact RELION per-image metadata parity
- the sparse evaluator is expensive and not yet grouped/bucketed efficiently
- the posterior/statistics path after pass 2 is still not RELION-exact

### 4. No exact `highres_Xi2`

RELION stores the high-frequency residual contribution above `current_size` and
adds `exp_highres_Xi2_img / 2` to every diff2. That keeps the total
log-likelihood consistent while alignment is restricted to lower frequencies.

recovar truncates the Fourier window and currently ignores this term. The
result is different absolute log-evidence, different `dLL`, and different
noise-shell bookkeeping. Because `exp_highres_Xi2_img` is image-constant across
all candidates for one particle / metadata row, it does not by itself change
the within-row posterior shape or `Pmax`.

### 5. Noise update is fundamentally different

RELION noise is posterior-weighted and derived from the same weighted residuals
used during `storeWeightedSums()`.

recovar's `estimate_noise_level_no_masks()` is:

- hard-assignment based
- run after the EM step
- not masked/unmasked aware
- not derived from the actual posterior weights

This affects `sigma2_noise`, scale, tau2, and then the next iteration's
likelihoods.

### 6. Tau2 and `data_vs_prior` are still downstream approximations

RELION's `updateSSNRarrays()` derives reconstruction noise from the actual
backprojected weight array, then combines it with FSC to update `tau2` and
`data_vs_prior`.

recovar is closer than it used to be because
`compute_relion_prior_from_reconstruction_stats()` now uses the current
backprojected reconstruction weights. The remaining gap is that the upstream
noise / weighted-shell statistics are still not the exact RELION ones, so the
tau2 and `data_vs_prior` curves remain downstream approximations.

### 7. `ave_Pmax` is better, but metadata parity is still incomplete

RELION stores `Pmax` per particle and aggregates `ave_Pmax` directly from the
posterior weights.

recovar now carries per-image `Pmax` out of the engine and aggregates
`ave_Pmax` across both half-sets, which fixes an earlier bookkeeping bug.
What is still missing is the rest of the RELION metadata/statistics coupling:

- `dLL = log(sum_weight) - min_diff2 - logsigma2`
- `current_size` growth is not trustworthy
- convergence signals drift
- local search and refinement decisions drift

### 8. Priors are incomplete

RELION uses:

- orientation prior (`pdf_direction` and local Gaussian priors)
- translation prior (`pdf_offset`)
- local-search priors centered on previous best assignments

recovar now has a RELION-style translation prior in the score path and reuses
the coarse translation prior value on oversampled translation children. It also
matches RELION's score-path `sigma_offset = offset_range / 3` branch when a
finite search range is active.

What is still missing is:

- RELION's learned global direction prior `pdf_direction` outside local search
- RELION's coarse-grid prior normalization by `pdf_orientation_mean` and
  `pdf_offset_mean`
- exact metadata-driven orientation priors from previous best poses
- weighted prior updates in the same path that updates noise and reconstruction

### 9. `current_size` is still not exact RELION behavior

RELION can use any even `current_image_size` consistent with the model and
optics-group remapping. recovar still routes the result through
`quantize_current_size()`, which means the current-size trajectory can differ
even if the underlying resolution criterion is correct.

### 10. Convergence logic is only approximately RELION

RELION's convergence logic depends on:

- resolution stalls
- hidden-variable changes in angles and offsets
- whether angular sampling is already fine enough
- a final joined-halves, full-resolution iteration

recovar currently:

- tracks simplified assignment change as equality of coarse rotation indices
- uses approximate translation-change logic
- does not execute the final joined-halves RELION iteration

## Gaps That Matter Less For The Current Target

These are still relevant, but they should come after the blockers above for the
single-class, single-optics-group, homogeneous parity target:

- Multi-body refinement details
- Multiple optics groups
- Symmetry-specific subtleties when the target dataset is C1
- Helical and subtomogram-specific branches

They should not be the first thing we touch for the current projected-image
benchmark.

## Items That Are Not Parity Work

These may be useful later, but they should not lead the roadmap:

- Skipping adaptive oversampling when the significant fraction is high
- Grouping similar orientations to reduce memory if it changes which hidden
  states are evaluated
- Any heuristic that improves FSC or runtime by changing the search path before
  we have matched RELION's search path

They belong after the exact path exists and can be compared against it.

## Detailed Plan

### Phase 0: Lock the measurement loop

Objective: make parity measurable before changing more math.

Deliverables:

- A single benchmark entry point that runs recovar RELION-mode on the projected
  128px dataset and writes per-iteration outputs in a RELION-comparable format.
- A comparison script or test that consumes:
  - recovar outputs
  - `scripts/extract_relion_reference.py` outputs
- A report that includes, at minimum, per iteration:
  - `current_image_size`
  - resolution shell / Angstrom resolution
  - `tau2`
  - `sigma2_noise`
  - `data_vs_prior`
  - `Pmax`
  - `nr_significant_samples`
  - best angles and offsets

Implementation notes:

- Reuse `scripts/extract_relion_reference.py`.
- Reuse the helper comparisons in `tests/integration/test_relion_comparison.py`.
- Stop accepting "same ballpark" assertions in parity tests.

Exit criteria:

- We can run one command and get a machine-readable diff against RELION for each
  iteration.

### Phase 1: Make the engine return RELION-style sufficient statistics

Objective: stop approximating RELION's maximization inputs after the fact.

Implementation:

- Add a structured result object for the RELION-mode E/M pass, for example:
  `EMIterationStats`.
- Extend `run_em_v2()` so it can return:
  - `Ft_y`
  - `Ft_ctf`
  - per-image `log_Z`
  - per-image `Pmax`
  - per-image best hidden-state index
  - per-image significant coarse sample count
  - shell-wise weighted residual sums for noise
  - shell-wise weighted `ctf2` sums
  - scale-correction XA and AA sums
  - norm-correction accumulator
- Keep this streaming and blockwise. Do not materialize the full
  `(n_images, n_rot, n_trans)` tensor.

Files:

- `recovar/em/dense_single_volume/engine_v2.py`
- `recovar/em/dense_single_volume/types.py` or a new result container module

Exit criteria:

- RELION-mode no longer needs `ave_Pmax = 0.5`.
- RELION-mode no longer calls a separate hard-assignment noise estimator.

### Phase 2: Implement exact masked/unmasked preprocessing and `highres_Xi2`

Objective: match RELION's per-image preprocessing contract.

Implementation:

- Change preprocessing so each batch yields:
  - masked Fourier images for scoring
  - unmasked Fourier images for reconstruction
  - current-size windowed views
  - high-frequency residual term above `current_size`
- Replace the current full-image `batch_norm` handling with RELION-equivalent
  low-frequency plus `highres_Xi2` accounting.
- Carry `highres_Xi2` into the score normalization path.

Files:

- `recovar/em/dense_single_volume/engine_v2.py`
- potentially a new helper module if preprocessing becomes too large

Exit criteria:

- The likelihood path uses the same frequency split as RELION.
- `Pmax` and significant-count mismatches drop materially even before fixing
  pass 2.

### Phase 3: Replace union-of-significant pass 2 with per-image sparse pass 2

Objective: match RELION's adaptive oversampling semantics exactly enough to
remove the largest search-path mismatch.

Implementation:

- Keep significance over coarse `(rotation, translation)` samples, not
  rotations only.
- For each image, carry a compact representation of significant coarse samples.
- Generate oversampled child states only for that image's significant coarse
  states.
- Evaluate pass 2 sparsely per image or per group of identical parent samples.
- Preserve translation oversampling in pass 2.
- Keep uncapped significant-sample mode available for exact RELION benchmarks.
- Leave the "skip pass 2 when dense" shortcut only as a separate speed-mode
  option, not as the default parity benchmark path.

Files:

- `recovar/em/dense_single_volume/adaptive.py`
- `recovar/em/dense_single_volume/engine_v2.py`
- `recovar/em/dense_single_volume/refine.py`

Exit criteria:

- We can run order 5+ local-search iterations without the current union OOM
  failure mode.
- `rlnNrOfSignificantSamples` distributions track RELION closely.

### Phase 4: Implement exact priors in the E-step

Objective: make the hidden-state weights match RELION, not just the diff2 core.

Implementation:

- Tighten the existing translation prior path so it matches RELION exactly,
  including `offset_range / 3` sigma selection when a finite search range is
  active and coarse-grid mean normalization.
- Replace the current union-based local rotation prior with a per-image prior
  centered on each image's previous best pose.
- Preserve RELION's coarse-vs-fine handling of translational priors.
- Carry exact `Pmax` and metadata updates from the winning hidden state.

Files:

- `recovar/em/dense_single_volume/engine_v2.py`
- `recovar/em/dense_single_volume/refine.py`
- `recovar/em/dense_single_volume/convergence.py`

Exit criteria:

- Pose differences to RELION shrink without changing the reconstruction solver.

### Phase 5: Move noise, scale, norm, and `Pmax` updates inside the RELION-mode M-step

Objective: reproduce `storeWeightedSums()` and `maximizationOtherParameters()`
behavior rather than approximating it afterward.

Implementation:

- Delete or bypass `noise.estimate_noise_level_no_masks()` in RELION mode.
- Update `sigma2_noise` from the shell accumulators returned by the E/M engine.
- Add scale-correction XA/AA accumulation and update.
- Add norm-correction accumulation and update if needed for the target dataset.
- Derive `ave_Pmax` directly from accumulated per-image `Pmax`.

Files:

- `recovar/em/dense_single_volume/refine.py`
- `recovar/reconstruction/noise.py`
- new helper module if the update logic becomes large

Exit criteria:

- RELION mode no longer depends on hard assignments for its noise update.
- `sigma2_noise` trajectories compare shell-by-shell to RELION.

### Phase 6: Rebuild tau2 and `data_vs_prior` from the actual reconstruction weights

Objective: match `BackProjector::updateSSNRarrays()` semantics.

Implementation:

- Base reconstruction-noise estimates on the actual Fourier weight array coming
  out of the RELION-mode backprojection path.
- Update `tau2` from FSC and reconstruction noise using RELION's
  oversampling-corrected formulas.
- Compute `data_vs_prior` from the same shell statistics.
- Ensure CTF-premultiplied correction paths are correct if the dataset needs
  them.

Files:

- `recovar/reconstruction/regularization.py`
- `recovar/reconstruction/relion_functions.py`
- `recovar/em/dense_single_volume/refine.py`

Exit criteria:

- `tau2`, `reference_sigma2`, and `data_vs_prior` curves match RELION to within
  tight shell-wise tolerances.

### Phase 7: Make reconstruction exact

Objective: ensure the maps are being reconstructed the same way once the
statistics are right.

Implementation:

- Land the `zero_pad_fourier_volume()` fix on the actual parity branch.
- Run RELION mode with padding factor 2 in the parity path.
- Verify `post_process_from_filter_v2()` against RELION's
  `BackProjector::reconstruct()` inputs and outputs.
- Remove any remaining `current_size` quantization that is not required by the
  math.

Files:

- `recovar/reconstruction/relion_functions.py`
- `recovar/em/dense_single_volume/refine.py`
- tests for padding and reconstruction

Exit criteria:

- FSC shape against RELION stops showing the gradual-padding artifact.

### Phase 8: Make convergence and the final iteration exact

Objective: match RELION's auto-refine control plane.

Implementation:

- Replace the current assignment-change heuristic with RELION-like changes in
  optimal orientations and offsets.
- Use exact `ave_Pmax` and exact resolution stalls to drive angular-step
  refinement.
- Execute the final joined-halves, full-resolution iteration after convergence.
- Save final metadata in a RELION-comparable form.

Files:

- `recovar/em/dense_single_volume/convergence.py`
- `recovar/em/dense_single_volume/refine.py`

Exit criteria:

- `current_image_size`, `healpix_order`, and termination iteration match
  RELION's trajectory.

### Phase 9: Turn parity into a hard gate

Objective: stop regressions.

Implementation:

- Replace `tests/integration/test_full_refinement.py` "same ballpark" checks
  with a real parity benchmark.
- Add a dedicated integration test, for example
  `tests/integration/test_relion_parity_benchmark.py`.
- Save recovar iteration outputs in a format that matches the extracted RELION
  reference fields.

Suggested checks:

- exact `current_image_size` per iteration
- shell-wise `sigma2_noise` relative error
- shell-wise `tau2` / `reference_sigma2` relative error
- `nr_significant_samples` distribution match
- `Pmax` match
- angle and offset agreement
- final merged-map FSC to RELION merged map

Exit criteria:

- A parity benchmark fails loudly whenever behavior diverges.

### Phase 10: Only then optimize

Objective: become faster than RELION without changing outputs.

Safe targets after parity exists:

- Efficient per-image sparse batching for pass 2
- Better GPU tiling and projection reuse
- Optional "skip adaptive when almost everything is significant" mode, but only
  behind a separate flag or after proving identical outputs
- Similar-state grouping only if it is mathematically exact for the evaluated
  hidden states

## Recommended Immediate Next Actions

If we are executing this plan from the current codebase, the next concrete
sequence should be:

1. Put the zero-padding fix on the actual parity branch.
2. Refactor `run_em_v2()` so RELION mode receives structured E/M statistics
   instead of only `Ft_y`, `Ft_ctf`, and hard assignments.
3. Finish the remaining exact score bookkeeping around masked/unmasked
   preprocessing and `highres_Xi2`.
4. Group or bucket the per-image sparse pass-2 evaluator so the exact search
   path stays fast on realistic runs.
5. Move `sigma2_noise`, `Pmax`, and scale/norm updates into the RELION-mode
   iteration loop.
6. Rebuild tau2 from the actual reconstruction weights.
7. Replace the permissive integration tests with a real parity gate.

## Non-Negotiable Success Criteria

We should consider RELION parity achieved only when all of the following are
true on the projected 128px benchmark:

- `current_image_size` trajectory matches exactly.
- Termination iteration matches.
- Shell-wise `sigma2_noise` and `tau2` match to tight tolerance with no
  systematic drift.
- `Pmax` and `nr_significant_samples` track RELION closely enough that the
  distributions are visually and numerically aligned.
- Best-angle and best-offset differences are small and stable across
  iterations.
- Final half maps and merged map match RELION up to the expected numerical
  tolerance of the different implementation substrate.

Until those conditions hold, speed work is secondary.

## RELION Design Quirks and Non-Obvious Choices

This section documents several RELION design decisions that appear arbitrary or
surprising but have concrete reasons behind them. Understanding these is
important for parity work because naively "fixing" any of them would break the
match.

### 1. DC pixel exclusion from scores but inclusion in noise

RELION sets `Minvsigma2[0] = 0` to exclude the DC (zero-frequency) component
from likelihood scores, making the alignment algorithm invariant to additive
constants in the images. However, DC is included in the noise estimation path:
`storeWeightedSums` accumulates `|diff|^2` at `ires=0` into the noise sum.

This means DC noise can be very large without affecting the posterior, but it
inflates the overall noise estimate. The large DC noise (0.059 vs ~0.00004 at
other shells) is an artifact of the masking: `softMaskOutsideMap` zeros outside
the particle diameter, reducing the DC of the image, but the reference
projection still has full DC. The resulting large `sigma2_noise[0]` is harmless
because the DC shell is never used for scoring, but it is visible in diagnostic
output and can be confusing.

### 2. Prior mean normalization

In `convertAllSquaredDifferencesToWeights`, RELION divides `pdf_orientation` by
`pdf_orientation_mean` and `pdf_offset` by `pdf_offset_mean`. This "extra
normalization" prevents the priors from dominating the likelihood at early
iterations when the model is poor. It is not standard Bayesian EM -- it is a
heuristic that adjusts the effective prior strength.

Without this normalization, the prior contribution scales with the number of
candidate orientations and offsets, which can overwhelm the (poor) likelihood
at early iterations. The mean normalization makes the effective prior
contribution O(1) regardless of the grid size.

### 3. Half-complex scoring convention

RELION sums `0.5 * |diff|^2 * Minvsigma2` over the FFTW half-complex
representation (independent modes only). The factor 0.5 and the half-complex
sum together give the correct negative log-likelihood for real-valued images.

Using the full spectrum with Hermitian weights (as recovar originally did)
gives 2x the score, which is mathematically equivalent after softmax but
produces exponentially more peaked posteriors. This is because the softmax
temperature is implicitly set by the score scale: `exp(-2 * score)` is much
more peaked than `exp(-score)`. For parity, recovar must use the same scoring
convention as RELION.

### 4. `highres_Xi2` as a score baseline

RELION initializes each `diff2` with `highres_Xi2 / 2` (the power of the image
beyond `current_size`). This is a per-image constant that cancels in the
posterior softmax but is needed for:

- (a) correct absolute log-likelihood monitoring
- (b) high-frequency fill in noise estimation

Without it, the noise estimate at high frequencies is zero instead of the image
power. The `highres_Xi2` term does not affect alignment (it is constant across
all candidate poses for a given image) but is essential for consistent
bookkeeping across resolution shells.

### 5. `sigma2_fudge` fixed at 1.0 for auto-refine

For classification, RELION uses `sigma2_fudge = 4` to inflate the noise and
make classification more robust. For auto-refine (single class), it is always
1.0. This is just a temperature parameter on the likelihood.

The classification fudge factor makes posteriors broader (less confident), which
prevents premature class collapse. For single-class auto-refine, there is no
classification step, so the fudge factor is unnecessary and would only slow
convergence.

### 6. Noise estimation includes model error

At early iterations, RELION's posterior-weighted `sigma2_noise` includes model
error (imperfect reference). This is deliberate -- it prevents the likelihood
from being too sharp when the model is poor. As the model improves, the noise
estimate decreases.

In contrast, an initial estimate from particle power spectra reflects only true
noise, not model error. RELION's approach creates a natural annealing schedule:
early iterations have inflated noise (broad posteriors, robust alignment), and
later iterations have lower noise (sharp posteriors, precise alignment). This is
one reason why RELION's noise generally starts high and decreases, rather than
staying flat or oscillating.
