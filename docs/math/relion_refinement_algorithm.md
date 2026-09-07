# RELION-style refinement: algorithm and code map

This page describes RECOVAR's current dense-volume refinement implementation,
including its K-class and exact local-search routes. Function names identify
implementation owners; line numbers are deliberately omitted because code moves
as it is cleaned up. The [codebase map](../development/codebase.md) covers the
rest of RECOVAR.

This is an implementation guide, not a claim of complete RELION parity.
[Current EM status](../development/em_status.md) records accepted checks,
failed comparisons and pending qualification. Historical RELION source locations
in code comments refer to the source used for those comparisons.

## 1. Controller and state

[`refine_single_volume`](../../recovar/em/dense_single_volume/iteration_loop.py)
accepts two half-set datasets, initial Fourier volumes, noise and signal priors,
and refinement settings. Despite its historical name, it supports `n_classes > 1`.
[`RefinementOptions`](../../recovar/em/dense_single_volume/refinement_options.py)
groups the settings; supplied option fields override the corresponding individual
arguments.

The controller, `_run_relion_iteration_loop`, coordinates:

1. Sampling and reference preparation for the current iteration.
2. Scoring each half through `_score_half_dense` or `_score_half_local`.
3. Reconstruction, low-frequency accumulator joining, and updates to noise,
   normalization, scale and translation-prior statistics.
4. Resolution and hidden-variable tracking for the next sampling decision.
5. An optional final all-data expectation and reconstruction.

The order of individual updates within these stages matters for trajectory
comparisons. Replay/oracle inputs can replace selected state boundaries;
results from those modes must remain distinguishable from autonomous refinement.
Their implementation belongs to
[`relion_replay.py`](../../recovar/em/dense_single_volume/relion_replay.py).

[`score_outputs.py`](../../recovar/em/dense_single_volume/score_outputs.py)
defines the controller's scoring payloads:

- `HalfScoreResult` holds one half's accumulators, assignments and statistics.
- `PerHalfOutputs` holds two half-set slots per field. Class axes live inside
  those slots; a class axis is not a half-set axis. The halves may contain
  different numbers of images.

These containers store existing references. They do not copy arrays, normalize
precision or release device storage. The controller owns buffer lifetime.
Accumulator shape and half-spectrum axis metadata must travel with the arrays.

## 2. Sampling grids and units

[`sampling.py`](../../recovar/em/sampling.py) owns rotation and translation grids,
Euler conversions, oversampled children and perturbations. For the full C1 grid,
`rotation_grid_n_in_planes` and `rotation_grid_size` give

```text
n_directions = 12 * 4**order
n_psi        = 6 * 2**order
n_rotations  = n_directions * n_psi
angular_step = 360 / (6 * 2**order) degrees
```

| Base order | Directions | Psi steps | Rotations | Angular step |
| --- | ---: | ---: | ---: | ---: |
| 1 | 48 | 12 | 576 | 30° |
| 2 | 192 | 24 | 4,608 | 15° |
| 3 | 768 | 48 | 36,864 | 7.5° |
| 4 | 3,072 | 96 | 294,912 | 3.75° |

`get_relion_rotation_grid` constructs the grid from the RELION binding.
RECOVAR's grid indexing is psi-slow and direction-fast. Preserve index order
when comparing hard assignments; equal sets of rotations are insufficient.
The source Euler and matrix precision can also matter at score ties.

For oversampling level `s`, `get_oversampled_rotation_grid_from_samples`
generates `4**s` direction children and `2**s` psi children per rotation parent:
`8**s` rotations. `get_oversampled_translation_grid` generates `4**s` children
per 2D translation parent. At `s=1`, one rotation/translation parent therefore
has up to **8 × 4 = 32 pose children**, before support restrictions. Here `s` is
an oversampling level; `K` below is the number of classes.

Translation grids and stored offsets use pixels unless an argument explicitly
names Angstroms. Local prior sigmas are stored in radians; angular sampling
and RELION Euler metadata use degrees. `relion_angular_sampling_deg` includes
oversampling when requested.

`advance_relion_perturbation` and the seed/replay helpers determine the scalar
sampling perturbation. `apply_relion_rotation_perturbation` and
`apply_relion_translation_perturbation` apply it to their respective grids.
Preserve the seed, iteration number, unperturbed grid and perturbation together
when reproducing a scoring boundary.

## 3. Gaussian scoring and posterior normalization

For image `i`, class `c`, rotation `r` and translation `t`, write the predicted
Fourier image as `a = S_t C_i P_r mu_c`. Ignoring constants common to all
hypotheses for that image, the Gaussian score is

```text
residual = sum_k w[k] * |y_i[k] - a[k]|² / sigma²_i[k]
         = data + cross + norm
cross    = -2 * sum_k w[k] * Re(conj(y_i[k]) * a[k]) / sigma²_i[k]
norm     = sum_k w[k] * |a[k]|² / sigma²_i[k]
score    = -0.5 * (cross + norm)
```

The image-only `data` term cancels when normalizing pose probabilities within
a common scoring convention. Evidence comparisons and external class
normalizers must account for the same omitted offset. Image normalization,
group scale, CTF and noise factors are applied by the preprocessing/scoring
path; the formula above is schematic about where those factors are stored.

Let `log_prior` include the applicable class, direction and translation priors.
The full posterior is

```text
log_weight[i,c,r,t] = score[i,c,r,t] + log_prior[i,c,r,t]
log_Z[i]           = logsumexp over all allowed (c,r,t) of log_weight[i,c,r,t]
gamma[i,c,r,t]     = exp(log_weight[i,c,r,t] - log_Z[i])
```

Priors belong inside the normalization. In K-class EM, independently
normalizing each class's pose distribution would discard class probabilities.
Support pruning and first-iteration winner selection are additional policies;
retained M-step mass need not equal the full posterior mass.

The implementation owners are:

| Work | Owner |
| --- | --- |
| Image/CTF/noise preparation and translation phases | [`preprocessing.py`](../../recovar/em/dense_single_volume/helpers/preprocessing.py), `preprocess_batch` and `preprocess_batch_firstiter_cc` |
| Projection and projection-dependent residual statistics | [`projection.py`](../../recovar/em/dense_single_volume/helpers/projection.py), `compute_projections_block` and `compute_relion_projector_projections_block` |
| Gaussian and normalized-CC block scores | [`scoring.py`](../../recovar/em/dense_single_volume/helpers/scoring.py), `_score_rotation_block` and `_e_step_block_scores_windowed` |
| Priors, candidate masks and class/external-normalizer constraints | [`score_constraints.py`](../../recovar/em/dense_single_volume/helpers/score_constraints.py), `DenseScoreConstraints` |
| Scoring weights for the selected Fourier convention | [`half_spectrum.py`](../../recovar/em/dense_single_volume/helpers/half_spectrum.py), `make_scoring_half_image_weights` |

The half-image layout has `H * (W//2 + 1)` entries. RELION half-sum scoring and
Hermitian full-image inner-product weights are separate conventions. Gaussian
RELION scoring masks redundant centered `kx=0` rows; normalized-CC callers can
retain them. Changing these weights is a numerical change, not a missing
optimization to enable during cleanup.

## 4. Dense, adaptive and local execution

There are two different uses of “two pass.” Keep them separate when profiling
or comparing intermediate results.

**Blockwise normalization within one grid.**
[`em_engine.run_em`](../../recovar/em/dense_single_volume/em_engine.py) processes
image batches and rotation blocks. Its first sweep collects normalization and
best-pose statistics; its second sweep recomputes scores for accumulation.
`_update_logsumexp` and `_merge_block_logsumexp` in `helpers/scoring.py` combine
block normalizers. An external `normalization_log_evidence` can normalize this
class against a joint class/pose distribution. `score_only` skips accumulation;
other options can skip negligible second-sweep blocks or use fused execution.
The complete image × rotation × translation score tensor is not required.

**Adaptive coarse-to-fine search.**
[`k_class.py`](../../recovar/em/dense_single_volume/k_class.py) owns
`run_dense_k_class_em` and `run_dense_k_class_em_adaptive`.
[`significance.py`](../../recovar/em/dense_single_volume/helpers/significance.py)
computes joint coarse class/pose evidence and significant support, including
K=1 routed through the class-aware implementation.
[`oversampling.py`](../../recovar/em/dense_single_volume/helpers/oversampling.py)
owns cumulative-mass selection and coarse/fine mappings. Significance selects
rotation/translation pairs; it is not simply an independent probability cutoff
on every orientation.

Fine execution can use dense or sparse routes.
[`sparse_pass2_bucketed.py`](../../recovar/em/dense_single_volume/helpers/sparse_pass2_bucketed.py)
owns bucketed and compact-pair scoring, posterior reconstruction policies and
accumulation, including `compute_k_class_pass2_stats_sparse_fused`.
The support representation, execution buckets and float32 posterior policy
are part of the comparison contract. Preserving only final MAP assignments
does not establish equivalent soft M-step contributions.

**Exact local search.**
[`local_search_iteration.py`](../../recovar/em/dense_single_volume/local_search_iteration.py)
constructs per-image neighborhoods, applies the batch budget and dispatches
`local_em_engine.run_local_em_exact` or `k_class.run_local_k_class_em`.
[`local_layout.py`](../../recovar/em/dense_single_volume/local_layout.py)
builds the per-image hypothesis layout. This route does not use the retired
sort-and-split union helper formerly described on this page.

`sampling.get_local_rotation_grid_fast` implements C1 factored direction/psi
priors. Viewing directions use the third **row** of the RELION rotation matrix.
Direction and psi cutoffs use their respective sigmas; psi width must not widen
the direction cone. Neighborhoods follow previous best poses, while M-step
weights inside the neighborhood remain soft unless a winner-selection policy
is active. This limits exploration of separated modes; it does not prove that
a particle can never leave its initial neighborhood over later iterations.

**Fourier windows and performance.**
[`fourier_window.py`](../../recovar/em/dense_single_volume/helpers/fourier_window.py)
defines `FourierWindowSpec` and the score/projection window mappings.
`current_size` is an image diameter in pixels. Window shape, pixel order and
redundant-axis treatment depend on the scoring route. Smaller windows reduce
operand sizes, but neither the layout size nor a GEMM formulation establishes
a fixed speedup. Use paired measurements under the
[benchmark contract](../development/benchmarks.md).

## 5. Accumulation, reconstruction and parameter updates

Conceptually, each class accumulates a weighted-image numerator and a
CTF/noise precision denominator:

```text
Ft_y[c]   = sum_(i,r,t) gamma[i,c,r,t] * P_r* (conj(S_t C_i) y_i / sigma²_i)
Ft_ctf[c] = sum_(i,r,t) gamma[i,c,r,t] * P_r* (|C_i|² / sigma²_i)
mu_c      ≈ Ft_y[c] / (Ft_ctf[c] + prior_precision[c])
```

These equations omit layout, interpolation, normalization and padding details.
`P_r*` inserts a 2D slice into the 3D accumulator. Dense accumulation belongs to
`em_engine._dense_mstep_block` and the scoring/adjoint helpers; local and sparse
routes have their own implementations. `Ft_y` is complex. `Ft_ctf` represents
real weights, although some return layouts store it in a complex array.

[`half_volume_mstep.py`](../../recovar/em/dense_single_volume/helpers/half_volume_mstep.py)
owns packed-half conventions, the Hermitian `x=0` plane and conversions to
public layouts. Do not assume all accumulators have the full native volume
shape: padding and current-size backprojector grids change their dimensions.

[`mean_helpers.py`](../../recovar/em/dense_single_volume/mean_helpers.py) owns
`compute_unregularized_halfmaps_and_align_signs`,
`_reconstruct_and_postprocess_means`, `update_posterior_noise_variance`,
`update_relion_norm_scale_corrections` and `update_c1_sigma_offset_from_posterior`.
These updates consume posterior-weighted residual and moment statistics as
well as accumulators. The input noise representation can be a per-pixel array
or separate half-set inputs; radial statistics and group corrections have
explicit conversion/update paths.

[`regularization.py`](../../recovar/reconstruction/regularization.py) owns FSC,
tau2 and data/prior helpers. `compute_data_vs_prior` uses shell-average weight
**multiplied by** tau2, tau2 fudge and the padding-volume correction; it is not
`Ft_ctf / tau2`. The controller's K1 scheduling path also uses
`_k1_data_vs_prior_for_scheduling`; the generic weight-based helper is not an
exhaustive description of its resolution policy.
[`relion_reconstruct`](../../recovar/reconstruction/relion_functions.py) applies
the actual regularized reconstruction and postprocessing conventions.

During numbered split-half iterations, `join_halves_at_low_resolution` averages
**accumulators**, then writes the result into both halves inside the join sphere.
It does not average already reconstructed maps. The effective joining
resolution is the larger Angstrom value of the configured threshold (40 Å by
default) and the available current resolution. Thus “low resolution” means
low Fourier frequency, not spatial wavelengths smaller than 40 Å.

## 6. Sampling transitions and convergence

[`convergence.py`](../../recovar/em/dense_single_volume/helpers/convergence.py)
owns `RefinementState`, `update_refinement_state`, `update_angular_sampling`,
`refine_angular_sampling` and `check_convergence`.

Convergence requires the latched `has_fine_enough_angular_sampling` flag,
sufficient resolution stall and sufficiently stable hidden-variable changes.
When per-particle change tracking has not been populated, the implementation
uses its assignment-counter fallback. Reaching `max_healpix_order` prevents
further grid growth; it does **not** establish convergence.

At the sampling transition, the old effective angular step is compared with
75% of the measured angular accuracy. The controller latches the fine-enough
flag at that boundary. Recomputing it from a newly refined grid would move the
convergence decision. `refine_angular_sampling` increases the order, updates
translation range/step, resets the change counters and activates local search
at `auto_local_healpix_order`. Its local sigma is
`2 * radians(new_angular_step / 2**adaptive_oversampling)`.

[`expected_accuracy.py`](../../recovar/em/dense_single_volume/helpers/expected_accuracy.py)
owns the RELION-style accuracy trial calculation. The approximate posterior
helper `calculate_expected_angular_errors` is a different route. Likewise,
`mean_helpers._relion_optimizer_average_pmax` uses the split-half optimizer's
mass normalization, rather than just averaging all recorded Pmax values.

## 7. Final output and validation boundaries

When `_should_run_final_all_data_iteration` allows it, the controller performs a
final expectation at full image size and reconstructs from the combined
half-set accumulators. This is distinct from low-frequency joining during
numbered iterations. `skip_final_iteration`, convergence state and the explicit
final-iteration policies affect whether it runs.

The return dictionary includes `mean`, per-half/class products, assignments,
`fsc`, `convergence_state` and trajectories. `final_all_data_ran` identifies the
final route. `fsc` is the last numbered-iteration FSC; `final_all_data_fsc` is a
separate field when final all-data processing runs. Consumers should inspect
the actual returned fields instead of assuming one fixed four-item tuple.

First-iteration normalized-CC scoring and winner-take-all reconstruction are
implemented policies. They are not an unimplemented parity gap. Similarly,
no universal gridding-equivalence or speedup claim follows from this map.
Use [EM status](../development/em_status.md) for evidence tied to exact source,
inputs and hardware. The recorded fixed-input backprojection repeatability
finding is one reason to retain intermediate posterior and accumulator checks
alongside final-map, accuracy, wall-time and memory comparisons.

Hierarchical candidate propagation and multiple local-search centers remain
future engine design questions. They would change search support, state and
memory requirements and need their own scientific validation after this cleanup.
