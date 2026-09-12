# EM implementation reference

Start with the [workflow map](codebase.md) to select the correct controller.
Read the relevant owner below for the current task; this is implementation
reference, not a requirement to load every module description on each resume.
The [EM contract](../../recovar/em/AGENTS.md) remains authoritative for scientific
rules and the [current status](em_status.md) identifies reviewed source/evidence.

## Dense and local EM ownership

Pose-stack preparation for convergence belongs to
[`helpers.convergence.concatenate_pose_stacks_or_none`](../../recovar/em/helpers/convergence.py).
The iteration controller supplies precision and logging context and retains the
four current/previous rotation/translation call sites. Empty half-sets, missing
poses, malformed-shape warnings and concatenation ownership are preserved.
The completed state also owns `fraction_changed`; the controller reuses it for
history instead of repeating the assignment reduction after replay overrides.
The assignment metric takes only the two index stacks and translation count;
it compares decoded rotation indices, with no angular-distance threshold.
`convergence.concatenate_assignments` and `concatenate_assignments_or_none` join
both half-sets' int32 assignment indices for that comparison; the strict form
is used for the current iteration and the tolerant form for a previous
iteration that may not have recorded assignments. The optimizer Pmax
normalization mass comes from `mean_helpers._relion_pmax_normalization_mass_per_half`:
Class3D uses each half's retained M-step posterior mass, K=1 the half's noise
`sumw`, and `_relion_optimizer_average_pmax` divides half 1's Pmax sum by it.

Learned direction priors belong to
[`mean_helpers.update_learned_direction_priors`](../../recovar/em/refinement/mean_helpers.py).
K=1 collapses each half's rotation posterior at the order used for scoring and
skips a half whose prior cannot form a RELION log prior, with the warning routed
through the controller logger; K-class combines both halves' per-class posteriors
on the exhaustive grid for global scoring only and stores an independent copy per
half. The controller decides when both posteriors are present, derives the K=1
order from its sampling state and supplies the grid sizes, so the controller's
sampling policy stays the only source of grid geometry. `RefinementHistory`
owns the float64 snapshot copies of the rotation posteriors and of the learned
priors (class 0 per half for K-class); the controller passes the live lists.
Controller tests that stub the collapse step patch `mean_helpers`.

Snapshot initialization of those priors belongs to
`orientation_priors.initial_direction_priors_from_snapshot`: a RELION restart
carries the previous iteration's `pdf_direction` per half (per class for K-class),
normalized to the scoring dtype with the HEALPix order inferred from its length.
Scoring with those priors follows RELION through one owner,
[`orientation_priors.relion_direction_log_priors_for_half`](../../recovar/em/helpers/orientation_priors.py),
which both the regular iterations and the final all-data pass call per half.
RELION (`ml_optimiser.cpp`) multiplies orientation weights by the class's
`pdf_direction` value at the sampled direction only in `NOPRIOR` mode, so local searches
get no direction prior; `initialisePdfDirection` resets every class to an even
distribution on a sampling change, so a prior at another HEALPix order is not
used; RELION keeps one prior per class and copies class 0 to all classes when
seeding K references, so a K-class run that only holds a shared prior applies
it to every class; and each half scores with its own model, including the
joined final iteration. Sealed captured sampling expands the prior onto the
captured direction rows the scorer uses (`_sealed_direction_log_prior`, now
also owned here); otherwise the canonical sample ordering is used. The controller
supplies the scoring order, the sealed state only when the scored grid is the
sealed grid, and its logger. Adopting one rule changed three edge cases that
previously differed between the passes: the final pass now applies a shared prior
to every class, K-class priors follow sealed rows, and a stale class prior with a
matching shared prior now yields identical per-class rows instead of one shared
vector. [`test_direction_log_prior_owner.py`](../../tests/unit/test_direction_log_prior_owner.py)
pins each rule.
`orientation_priors.relion_local_search_sigmas` owns the local-search
orientational prior widths (configured widths kept, psi falling back to rot,
twice the oversampled angular step when unset; `updateAngularSampling`) for
both passes ([`test_local_search_sigma_owner.py`](../../tests/unit/test_local_search_sigma_owner.py)).

The [refinement controller](../../recovar/em/refinement/iteration_loop.py)
owns iteration history, half-set dispatch, sampling updates, convergence and
finalization scheduling/state mutation. Its module-level helpers
`_relion_mstep_source_eulers` and `_perturbed_trial_grid` hold the two sampling rules
both passes share: the exact M-step rotations are seeded from the sealed grid's own
angles or RELION's canonical grid at the perturbation order (the scoring grid's
angles when the row counts differ), and one RELION `SamplingPerturbation` rotates
the trial orientations, rebuilds the M-step rotations and shifts the translation
grid. `_initial_coarse_grids` materializes the first exhaustive grid from a
sealed capture, a caller translation table or the RELION translation grid, and
`_relion_base_translation_grid` is the only unperturbed translation-grid
construction in the controller; [`test_initial_coarse_grid_owner.py`](../../tests/unit/test_initial_coarse_grid_owner.py)
pins both. `_ExpectedAccuracyInputs` bundles the run-constant inputs of RELION's
expected-accuracy estimation, `_expected_accuracy_class_ids` gives the half-1 class
labels, and `_estimate_half1_expected_accuracy` is the one call site shared by both
passes ([`test_expected_accuracy_inputs_owner.py`](../../tests/unit/test_expected_accuracy_inputs_owner.py)).
`_advance_relion_perturbation` advances RELION's SamplingPerturbation to an iteration
(seeded `random_seed + iteration`, or the run's generator) for both passes
([`test_perturbation_advance_owner.py`](../../tests/unit/test_perturbation_advance_owner.py)).
The controller's adaptive and single-pass dense half-scoring calls share one keyword
set (`dense_half_kwargs`); only the adaptive branch adds its pass-1 grid and
batch/size overrides ([`test_dense_half_kwargs_owner.py`](../../tests/unit/test_dense_half_kwargs_owner.py)). `_exact_local_fine_grid` materializes RELION's fine local-search grid
once with its perturbation and exact M-step rotations, and
`_local_search_mstep_rotations` reuses or rebuilds the M-step matrices of a
scoring grid; the final pass sizes its parent pass with
`resolution.relion_local_pass1_current_size` only under adaptive oversampling
([`test_local_fine_grid_owner.py`](../../tests/unit/test_local_fine_grid_owner.py)).
They stay in the controller module because the sampling primitives they
call are the ones controller tests substitute. Its exact local-search stage is implemented in
[`local_search_iteration`](../../recovar/em/local/local_search_iteration.py).
That module builds local pose neighborhoods, asks
[`batch_planning`](../../recovar/em/helpers/batch_planning.py) for
batch sizes, calls the selected kernel and returns `_LocalSearchIterationResult`
with named accumulators, pose fields, statistics and optional class summaries.
The controller reads those fields directly.
The exact sparse pass-2 kernels in
[`sparse_pass2_bucketed`](../../recovar/em/sparse_pass2/sparse_pass2_bucketed.py)
reproduce RELION's CUDA `powerClass` through one operand owner:
`_relion_powerclass_packed_image` (RELION's unshifted `Faux` layout and
amplitude convention), `_relion_powerclass_operands` (CUDA shell map and pixel
validity) and `_relion_powerclass_native_spectrum_highres` (native atomics),
shared by the JAX reproductions and the native wrappers
([`test_powerclass_operand_owner.py`](../../tests/unit/test_powerclass_operand_owner.py)).
`_relion_powerclass_noise_terms` selects the `highres_Xi2` and high-shell norm
terms a sparse pass-2 batch needs for both sparse scorers
([`test_powerclass_noise_terms_owner.py`](../../tests/unit/test_powerclass_noise_terms_owner.py)).
`_sparse_pass2_window_setup` builds the forward-model configuration, score and
reconstruction windows, RELION x-half reconstruction indices and the windowed-prepare
decision for both sparse scorers
([`test_sparse_pass2_window_setup_owner.py`](../../tests/unit/test_sparse_pass2_window_setup_owner.py)).
In [`local_score_pass`](../../recovar/em/local/local_score_pass.py),
`_support_from_local_probs` is the one reconstruction-support rule (full-sort
significance or per-image threshold, else the rotation mask) used by every fused
score pass ([`test_local_support_owner.py`](../../tests/unit/test_local_support_owner.py)).
In [`k_class`](../../recovar/em/classification/k_class.py),
`_override_class_assignments_with_coarse_winner` applies RELION's coarse-grid
binarization to a pass-2 result (winning class, that class's fine pose, decoded
best-pose details) for both adaptive pass-2 paths
([`test_kclass_results_owner.py`](../../tests/unit/test_kclass_results_owner.py)).

Replay and finalization have separate selection and mutation boundaries:

| Responsibility | Owner | Inputs and preserved behavior |
| --- | --- | --- |
| Final-pass admission and gridding selector | [`finalization_policy.py`](../../recovar/em/refinement/finalization_policy.py) | Receives convergence/cap state and the controller logger. Reads diagnostic flags when called; does not mutate refinement state. |
| Replay numbering and cutoff | [`relion_replay.py`](../../recovar/em/diagnostics/relion_replay.py) | `_numbered_relion_iteration` maps restart-local indices; `_native_sampling_boundary_for_iteration` checks cutoff and sealed state. The controller retains scheduling. |
| Numbered optimiser accuracy override | `relion_replay.read_optimiser_accuracy_replay` | Selects this iteration's numbered optimiser STAR when replay is active and unsealed; finite RELION rotation/translation accuracies replace the reported and convergence accuracies. Read or parse failures warn and keep values assigned before the failure. Returns `OptimiserAccuracyReplay`; the controller passes its metadata to `apply_optimiser_convergence_replay` after the state update. |
| Final override selection | `relion_replay._select_final_replay_override` | Receives the requested index, explicit override, recorded history and its already-computed presence flag. Returns an index and the original override object; no copying or state updates. |
| Final reference substitution | `relion_replay._prepare_final_replay_references` | Validates source iteration, K1 restriction, two-map count and shapes in order; casts each map to its half's dtype. With no maps, returns the original reference list. |
| Applying selected state | [`iteration_loop.py`](../../recovar/em/refinement/iteration_loop.py) | Retains sigma, pose, corrections, noise and direction-prior updates in their original order, including casts and half-set handling. |

Read `_should_run_final_all_data_iteration` in decision order: forced-cap mode
rejects the extra pass first; otherwise convergence admits it. Without
convergence, the optional after-cap diagnostic can admit K1 only when the cap
has been reached. K-class still requires convergence. Gridding correction
currently defaults **off**; the strict-parity target specifies on. Resolving
that discrepancy is a separate scientific change, not part of extraction.

Replay admission is also explicit. An initial-only override does not activate
numbered final replay. An explicit final diagnostic override wins, even an
empty dictionary; otherwise automatic replay requires numbered overrides or
its force flag and must not be disabled. Selection clamps to the final stored
slot when needed. A `None` slot in a nonempty history raises an error; an absent
history logs that no override exists. The selected dictionary retains identity.
These final-selection diagnostics use the controller logger passed by the caller.

The replay owner also validates physical BPref ordering for fresh/imported/sealed
boundaries, selects final sampling STARs and checks required files. Numerical
grid construction after selection remains in the controller. Diagnostic map
loading includes half/class and shared-class fallback files, with unchanged
casts and Fourier/frame conversion; these map-loading messages use the replay
module's logger. Review [replay-state tests](../../tests/unit/test_relion_replay_state.py)
and [controller tests](../../tests/unit/test_refine_relion_mode.py) for selection
identity, missing-slot errors, cutoff behavior and cold-start finalization.

[`mean_helpers`](../../recovar/em/refinement/mean_helpers.py) owns two
M-step boundaries that the regular iterations and the final all-data pass used
to repeat inline. `join_half_accumulators_at_low_resolution` applies RELION's
`--low_resol_join_halves` to the K=1 half accumulators before the Wiener solve;
the join radius is capped by the last recorded shell resolution, a non-positive
recorded shell leaves it uncapped, and without history a finite state resolution
is used. `_class_tau2_from_iref_power_spectrum`, `_class_tau2_update_details`
and `_stack_class_tau2_update_details` produce the Class3D per-class tau2
volume, the RELION- and RECOVAR-frame shells, data-vs-prior and the stacked host
detail record with `fsc_shells` left `None`. The controller keeps the enabling
conditions, the replay `class_tau2` branch, the round/floor weight statistics
and the dump calls, and passes the regular or final accumulators, current size
and layout explicitly. Both owners reach `regularization` through the module
attribute, so monkeypatched controller tests keep working. Source guards in
[`test_dense_iteration_loop_merge_guards.py`](../../tests/unit/test_dense_iteration_loop_merge_guards.py)
check that the controller no longer calls those regularization functions
directly; [`test_class_tau2_lowres_join_owner.py`](../../tests/unit/test_class_tau2_lowres_join_owner.py)
pins the argument layout, dtypes and record layout.

The local kernel returns `LocalEMResult` from
[`helpers.types`](../../recovar/em/helpers/types.py):
`Ft_y`, `Ft_ctf`, `hard_assignments`, `stats`, optional best-pose fields,
`noise_stats`, `profile` and `significant_counts`. All sixteen return-flag
combinations have the same field layout; disabled fields are `None`. The
local-search wrapper and K-class orchestration read these fields directly;
the positional packer and both decoders are removed. The result stores array
references without copying or synchronizing them.

Requesting reconstruction probabilities or sample IDs enables the engine's
profile that carries those captures. The wrapper still exposes a profile only
when requested, and copies its dictionary before adding wrapper timings.
Significant counts retain their own field even when that internal profile is
hidden. `tests/unit/test_local_search_result_contract.py` covers this routing,
plus K2/exact-K4 pose, noise and class-summary settings. Returned arrays retain
their layouts, dtypes and identities; saved refinement field names are unchanged.

Per-half dispatch belongs to
[`half_scoring`](../../recovar/em/dense/half_scoring.py).
Its dense and local adapters prepare engine arguments, retain adaptive/first-CC
routing, and write class/pose fields into the caller-owned `PerHalfOutputs`.
`_dense_uses_adaptive_engine` states the engine rule for K=1 and K-class scoring:
RELION's `storeWeightedSums` accumulates the group-scale `XA`/`AA` sums and the
norm-correction residuals in every pass, so scoring with RELION scale groups uses
the adaptive/sparse engine at the requested oversampling order, including 0, where
its single coarse pass on the current grid is RELION's single pass; positive
oversampling always keeps the two-pass adaptive expectation (pass 1 at the
current size when no reduced coarse size exists); the direct dense engine serves
only runs without scale groups at oversampling 0
([`test_dense_scale_group_routing.py`](../../tests/unit/test_dense_scale_group_routing.py)).
`half_scoring._adaptive_pass2_grids` materializes the perturbed coarse grid, the
oversampled children with parent maps, the fine M-step rotations and the coarse
translation phase source for both routes
([`test_adaptive_pass2_grids_owner.py`](../../tests/unit/test_adaptive_pass2_grids_owner.py)).
`half_scoring._adaptive_engine_shared_kwargs` holds the keywords both routes pass
identically to `run_dense_k_class_em_adaptive` (noise accumulation, RELION's adaptive
fraction, the fine M-step rotations pruned only for sparse pass 2); the K=1 call adds
significance skipping, the diagnostic float64 pass 2 and the host-double coarse
translation phases, and the K-class call plans its own batches. `k_class._sparse_pass2_selected`
reads the `RECOVAR_K1_DENSE_PASS2` / `RECOVAR_K_CLASS_DENSE_PASS2` diagnostic switches
for the three adaptive call sites, and `_coarse_pose_assignments` collapses fine pose
assignments onto the coarse grid when a fine pass ran
([`test_adaptive_engine_call_owner.py`](../../tests/unit/test_adaptive_engine_call_owner.py)).
In [`local_em_engine`](../../recovar/em/local/local_em_engine.py), the
exact-local BPref contribution capture binds its fixed operands once per run through
`_exact_local_bpref_capture_static_kwargs` (raw batch data, CTF parameters, image masks
and shadow comparisons recorded as absent; padding factors, x-half layout, adjoint radius,
window indices and shapes from the M-step geometry) and derives each captured bucket's
candidate mask and prior-free scores through `_bpref_capture_priors`; the fused and big-JIT
capture sites pass only their per-bucket operands
([`test_bpref_capture_operands_owner.py`](../../tests/unit/test_bpref_capture_operands_owner.py)).
[`heterogeneity._fixed_rotation_covariance_images`](../../recovar/em/reference/heterogeneity.py) accumulates the
fixed-rotation covariance-column update in image space (right-hand side and normal
operator per rotation) for both the Equinox and the classic accumulator, which only
convert to half images and back-project. In [`initial_model.layout`](../../recovar/em/vdam/layout.py),
`_centered_bpref_sources` validates and centers the data/weight cubes once for the
dense and the RELION-x-half BPref converters, and `_bpref_slab_outputs` applies RELION's
double-precision cast and denormal-weight clamp
([`test_covariance_rhs_and_bpref_source_owner.py`](../../tests/unit/test_covariance_rhs_and_bpref_source_owner.py)).
[`local_debug._requested_dump_rows`](../../recovar/em/diagnostics/local_debug.py) decides once
whether a local debug dump writes anything (dump directory, pending original image ids,
requested current sizes and iterations) and which bucket rows it covers; the fused-posterior,
score and noise-component dump writers only serialize the selected rows
([`test_debug_dump_rows_owner.py`](../../tests/unit/test_debug_dump_rows_owner.py)).
[`state_swap_runtime._apply_state_swap_probe`](../../recovar/em/diagnostics/state_swap_runtime.py)
returns a `_StateSwapValues` named tuple (current size, maps, tau2, noise, poses, sigma offset and
direction priors in the controller's unpacking order); the unchanged value is built once from the
inputs and returned by both early exits
([`test_state_swap_values_owner.py`](../../tests/unit/test_state_swap_values_owner.py)).
[`k_class._PerClassSubsetResults`](../../recovar/em/classification/k_class.py) collects the
per-class outputs of the dense and sparse firstiter-CC global-winner subset passes in class
order: a class without images gets zero accumulators, `-inf` best scores and zero posteriors,
and a scored class has its subset accumulators, statistics, noise and best poses expanded to
the full image axis. Each route states whether it hosts the appended accumulators
([`test_kclass_results_owner.py`](../../tests/unit/test_kclass_results_owner.py)).
[`scoring._e_step_block_score_components`](../../recovar/em/scoring/scoring.py)
computes the two HIGHEST-precision GEMMs every dense scorer is built from (the cross term
`-2 Re(conj(shifted) . proj_weighted)` and the model energy `ctf2_over_nv . proj_abs2`);
the residual, windowed, normalized-CC and coarse Gaussian scorers only combine them
([`test_score_components_owner.py`](../../tests/unit/test_score_components_owner.py)).
[`projection._relion_projector_fftw_block`](../../recovar/em/helpers/projection.py)
projects one rotation block through RELION's Projector onto the clamped `2 r_max` (or
requested) square with the scorer rotations transposed at the handoff; the centered-row
projector reorders its rows and the indexed projector gathers its pixels from that block
([`test_projector_fftw_block_owner.py`](../../tests/unit/test_projector_fftw_block_owner.py)).
In [`em_engine.run_em`](../../recovar/em/dense/em_engine.py) the per-batch scoring
operands (windowed shifted images and weights, batch norm, half weights, batch and
translation counts, shapes, score mode and precision policy) are bound once per batch as
`score_block_kwargs`; the pass-1 and pass-2 rotation-block scorers add only their block's
projections ([`test_em_engine_score_block_kwargs.py`](../../tests/unit/test_em_engine_score_block_kwargs.py)).
[`pass2_diagnostics._optional_operand_row_fields`](../../recovar/em/diagnostics/pass2.py)
captures one image's optional RELION score operands for the K=1 pass-2 dump, recording
operands the caller did not supply as absent (empty arrays of the capture dtype, or NaN for
the per-image normalization factor and batch corrections); the selected-rows and
effective-grid schemas both write these fields
([`test_pass2_dump_operand_fields_owner.py`](../../tests/unit/test_pass2_dump_operand_fields_owner.py)).
[`sparse_pass2_bucketed._gaussian_algebraic_score_terms`](../../recovar/em/sparse_pass2/sparse_pass2_bucketed.py)
computes the historical algebraic Gaussian scores of one bucket before candidate masking
(HIGHEST-precision weighted cross einsum and projection norm, prior-free and prior-added
scores); the production algebraic scorer and its components variant only apply their masks
([`test_gaussian_algebraic_terms_owner.py`](../../tests/unit/test_gaussian_algebraic_terms_owner.py)).
[`types.sparse_pass2_result`](../../recovar/em/helpers/types.py) assembles the
sparse pass-2 return tuple: the six accumulator and pose outputs first, then each requested
optional entry in a fixed order (RELION statistics, the score-only log partition function,
merged noise statistics, source Euler angles). Callers unpack by position, so an omitted
entry shifts the ones after it; the bucketed pass keeps the historical rule that the score
log-Z is emitted only alongside the statistics
([`test_sparse_pass2_result_tuple_owner.py`](../../tests/unit/test_sparse_pass2_result_tuple_owner.py)).
In [`significance`](../../recovar/em/scoring/significance.py),
`_coarse_gaussian_ffi_default` applies the fresh-InitialModel coarse Gaussian FFI
default only when the supplied RELION projector operands exist; a dense pass
without a projector keeps the JAX coarse path and an explicit environment request
still fails closed.
`preprocessing.uses_relion_cuda_image_preprocessing` (with `relion_preprocess_backend`,
which follows subset parents) is the one detection of RELION's CUDA image path; the
local engine, the InitialModel adapter and the controller's early fresh-K=1 check use
it ([`test_relion_cuda_preprocess_owner.py`](../../tests/unit/test_relion_cuda_preprocess_owner.py)).
The controller calls its two BPref-scoped entry points and retains iteration
scheduling, state transitions, reconstruction and device-buffer lifetime.
[`scoring_policy`](../../recovar/em/dense/scoring_policy.py)
owns shared padding/window constants, the import-time static kwargs object,
and the existing call-time environment selectors. Their override precedence,
invalid-value handling and float32 defaults are preserved. The controller and
scorers share the same kwargs object; policy imports do not load engine modules.
These owners have no imports back into the controller. Log messages are unchanged,
with namespaces following the owner of each moved function.

The dense single-class kernel is
[`em_engine.run_em`](../../recovar/em/dense/em_engine.py).
It returns `DenseEMResult` from
[`helpers.types`](../../recovar/em/helpers/types.py), with
named `mean`, `hard_assignments`, `Ft_y`, `Ft_ctf`, `stats`, `noise_stats` and
`profile` fields. Optional outputs are `None` when their existing flags are
disabled; changing flags no longer changes tuple positions. The container does
not copy arrays. Controller and K-class callers read these fields directly.
The local single-class kernel is
[`local_em_engine.run_local_em_exact`](../../recovar/em/local/local_em_engine.py).

[`local_batch_planning`](../../recovar/em/local/local_batch_planning.py)
owns exact-local row limits, environment overrides, automatic boosts and memory
probes. The engine applies these policies at the same dispatch boundaries;
reporting imports the planning owner directly. Layout padding stays in
`local_layout`. The existing device-memory query/cache behavior is preserved,
including the all-device `nvidia-smi` query; it is not a visibility-aware probe.
[`k_class`](../../recovar/em/classification/k_class.py) supplies dense,
adaptive and local K-class orchestration.
[`k_class_inputs`](../../recovar/em/classification/k_class_inputs.py) owns
class-axis validation, shared/per-class array selection and local prior layouts.
It imports no execution engines; engine-specific keyword filtering stays in `k_class`.
[`k_class_results`](../../recovar/em/classification/k_class_results.py) owns
the shared result type, joint result assembly and host/device publication.
Accumulator offloading and scheduling stay in the orchestrator.
Class evidence and posterior mass
must be handled at the K-class level, not inferred from independently normalized
single-class probabilities.

Fixed-capacity call selection and validation belong to
[`fixed_capacity_local.py`](../../recovar/em/local/fixed_capacity_local.py),
with the sealed plan/operand/hypothesis binding types. The engine delegates those
checks at the same pre-JIT boundaries. Callers use the general call-index API;
the test-only call-0 wrappers are removed. Canonical byte/dtype checks, poisoned-tail
rejection and authoritative dataset fetch order remain mandatory. Bucket geometry
and the dtype-preserving adjoint rotation accessor belong to
[`local_layout.py`](../../recovar/em/local/local_layout.py).
Both owners import independently of execution modules.

Scale-group ID validation and full-axis sizing have one host owner,
[`helpers/scale_groups.py`](../../recovar/em/helpers/scale_groups.py).
Local EM, both sparse scorers and the K-class subset router use it. Explicit
counts retain groups absent from a class subset; missing IDs disable engine
scale-statistics allocation, while routing still retains an explicit count.
Empty ID arrays retain the existing one-group convention. Engine callers check
the flattened image axis; the router has no image-count constraint. The helper
preserves existing casts and errors and imports independently of execution.

External normalization inputs are prepared by
[`helpers/normalization_inputs.py`](../../recovar/em/helpers/normalization_inputs.py).
`prepare_local_normalization_inputs` returns named log-Z, log-evidence, Pmax and
reconstruction-threshold arrays. It owns the local modes' validation order and
exclusivity; sparse and K-class callers reuse only optional F64 image-vector
conversion and retain their different semantic checks. This owner prepares inputs;
posterior arithmetic and normalization kernels remain at their execution sites.
It preserves input strides and avoids copying already suitable F64 arrays.

Local projector slab normalization has one owner,
[`projector_preparation.prepare_local_projector_slab`](../../recovar/em/refinement/projector_preparation.py).
Bucket projection, packed-noise projection and the main BigJIT path accept the
same three-dimensional slab or singleton class axis. The helper preserves JAX
dtype conversion and path-specific errors. Radius requirements, pixel selection,
interpolation, masking and projection execution stay with the callers.

Within `local_em_engine._project_local_bucket`, backend selection is separate
from shared result assembly. RELION/indexed compact rows use the same score and
reconstruction gathers; full outputs use the existing window selectors. Weighting
and precision conversion have one call site. Optional reconstruction and native
projection arguments remain explicit, with no additional result wrapper.
`local_em_engine._accumulate_packed_noise_chunk` owns the per-chunk noise shell,
norm-residual and group-scale accumulation of the exact local M-step for both
packed projection sources
([`test_packed_noise_chunk_owner.py`](../../tests/unit/test_packed_noise_chunk_owner.py)).

Sealed VDAM worker and block-chronology replay lives in
[`helpers/vdam_replay.py`](../../recovar/em/diagnostics/vdam_replay.py).
It owns NPZ schema validation, four cached loaders, stack-ID joins, worker/launch
ordering, iteration selectors and physical-row gathers. The local engine calls
this owner directly while retaining kernel execution and the capture call site.
Candidate block-map publication and its binary schema share this owner; the CLI
reader imports the schema while retaining validation/sealing and versions 1/2
compatibility. Replay defaults, caches, stable ordering and error behavior are
preserved; importing the helper does not initialize the execution engine.

The exact coarse Gaussian path in
[`helpers/significance.py`](../../recovar/em/scoring/significance.py)
passes its existing host pixel indices to the shared source-precision CTF owner
[`helpers/relion_ctf.py`](../../recovar/em/relion/relion_ctf.py).
Coarse, local and sparse scoring call that owner directly; its single process
cache and native binding remain independent of the execution engines.
That loader gathers each cached CTF row before stacking and device placement;
index order and duplicates are preserved. Omitting pixel indices retains the
full-grid contract and cache. Only scoring operands are compacted: full-image
powerClass inputs, source precision, scale correction and padding semantics
remain intact. The [current evidence](em_status.md) separates operand equivalence
and the allocation microbenchmark from pending full-runtime/trajectory checks.

The production fine-grid significance mask is lazy: `_ClassFineGridSignificanceMask`
and `_PerClassFineGridSignificanceMask` generate only the requested image/rotation
block. The materialized NumPy comparison lives in
[`tests/helpers/fine_grid_significance_reference.py`](../../tests/helpers/fine_grid_significance_reference.py).
It has no production callers and retains a separate mask-building algorithm
for checking lazy blocks and explicit/complement coarse support.

[`score_outputs`](../../recovar/em/dense/score_outputs.py) owns
the scoring containers and class/coarse-grid result adapters. It also owns
optional half-accumulator combination, shape/axis resolution and profile-row
recording. The controller retains scheduling and device-buffer offloading.
`HalfScoreResult` carries one halfset's common scoring output.
`PerHalfOutputs` owns separate two-slot lists for a scoring phase; slot 0/1
always selects the halfset, including for class-related fields. Image arrays
retain each halfset's local order and size. The K-class adapters populate class
assignments, posterior summaries and per-class noise statistics separately from
`update_from`. That method preserves existing optional pose fields when the
new result omits them, while always replacing accumulator-layout metadata.

Local-search dependencies are imported from their owners. Tests that replace a
kernel for a local dispatch check patch its binding in `local_search_iteration`.
`half_scoring` has its own active `build_local_hypothesis_layout` binding for
adaptive parent-layout construction. Tests of whole-controller dispatch patch
engine bindings in `half_scoring`; tests of local chunk execution patch
`local_search_iteration`. A shared sizing function mocked across both the
controller and scorer must be patched at both consumers. Patch the call site
exercised by the test; do not add reverse imports to preserve an old monkeypatch
location.

[`diagnostics.iteration`](../../recovar/em/diagnostics/iteration.py) owns the
half-selection policy for terminating significance/noise captures and numbered
BPref device captures, together with iteration dump writers. The controller
calls those selectors at the same dispatch boundaries and retains diagnostic
completion/stop control. Selector errors and log messages are unchanged; their
logger keeps its historical `dense_single_volume.debug_dumps` namespace. Capture state and counters belong to the
separate diagnostic owner below.

[`diagnostics.reconstruction`](../../recovar/em/diagnostics/reconstruction.py)
serializes the K-class current-size/M-step, tau2-update and final BPref NPZ
captures. The refinement controller retains the environment gates and call
boundaries; writers preserve historical fields, casts and optional entries.

[`helpers.coarse_score_diagnostics`](../../recovar/em/diagnostics/coarse_score_diagnostics.py)
owns host NumPy summaries of direct/GEMM score deltas, ULPs, winner margins,
support changes, repeated runs and scale panels, plus the qualification decision.
It also validates selector audits, hashes exact support and attaches diagnostic
profiles to results. Scoring, K-class/InitialModel controllers and reporting
scripts import these helpers directly; saved-audit validation does not require
loading `significance`. `significance` calls this owner directly. JAX scoring/M-step kernels remain in
`helpers.scoring`; diagnostic imports do not initialize those execution modules.

[`diagnostics.pass2`](../../recovar/em/diagnostics/pass2.py)
owns K1/K-class score dumps and target-row selection, including staging effective K-class raw operands after
scoring. It reads the shared numbered-half context from `bpref_diagnostics`;
it does not import sparse scoring. The scorer retains scheduling and numerical
operand preparation, calling the capture helpers at their original boundaries. Capture schemas, casts and reduction order are unchanged.
[`diagnostics.norm_scale`](../../recovar/em/diagnostics/norm_scale.py) owns
normalization-residual and chunked scale-AA captures. The diagnostics package
itself imports no engines or capture modules; import specific writers directly.

K1 and fused K-class preprocessing captures share
`bpref_diagnostics.build_bpref_preprocess_capture`. Callers retain the raw
preprocessing tuple to preserve operand lifetime and invoke the builder at the
original capture gate; schema, defaults, masks and dtype casts stay unchanged.

[`helpers.bpref_diagnostics`](../../recovar/em/diagnostics/bpref_diagnostics.py)
owns the numbered-half capture context, contribution and membership counters,
membership selectors/rotation-mass writers, device-panel state, capture validation
and artifact writers shared by sparse and exact-local EM. Fused K-class capture-row
materialization lives here beside the signature/shadow consumer; compact-pair
expansion, selected-row order and reconstruction-posterior fields are preserved. The controller and replay scripts set and clear that context through this
owner. Sparse scoring retains candidate planning, numerical kernels and live
accumulation; it asks the diagnostic owner for scoped capture decisions. The
diagnostic module has no direct import of sparse scoring or the iteration
controller. The package initializer exposes options and sampling/statistics
helpers; import K-class execution from `k_class.py` and result assembly/types
from `k_class_results.py`. The historical result type alias in `k_class` preserves
pickle compatibility; importing the new result owner does not load engines.
Standalone helper imports do not load dense/local engines or sparse scoring.
Tests replace capture functions and state at this owner, including optional
native signature panels. Dump schemas, precision, counter order and error
behavior remain unchanged. The boolean parser is shared through
`helpers.env_flags.parse_env_flag`; file identities use `utils.file_hash`.

Strict local capacity/packing selectors call `helpers.env_flags.parse_env_binary_flag`
directly. It accepts only `0` and `1` after stripping whitespace, defaults to
false when unset, and rejects blank or textual boolean values. Its behavior
differs from the permissive diagnostic parser above; do not interchange them
during structural cleanup. The engine retains the order of reads and mode checks.

Unused constant copies in `iteration_loop` have also been retired. Batch and
raw-image-cache limits, first-iteration reconstruction caps, dense K-class
hypothesis budgets and adaptive pass plans belong to `batch_planning`.
`firstiter_cc` constructs the first-iteration coarse/fine grids; the fine-grid
precomputation limit belongs to `local_search_iteration`. Their values and
environment overrides are unchanged.

The PPCA schedule bridge and its dense/local wrappers are imported from
[`ppca_bridge`](../../recovar/em/ppca_refinement/ppca_bridge.py).
Their unused controller re-exports have been retired. Helper-only callers also
import sign alignment and combined noise statistics from `mean_helpers`, rotation
metadata from `relion_metadata`, and replay iteration mapping from `relion_replay`.

[`relion_normalization`](../../recovar/em/relion/relion_normalization.py)
owns per-image norm and per-group scale formulas and their result type. It
depends on NumPy/JAX, not the controller, mean reconstruction or follower
dispatch. The controller retains state installation and temporary lifetimes;
`relion_worker_scale` handles follower-specific corrections. The seven formula
tests live in `tests/unit/test_relion_normalization.py` and import this owner
directly. The old `mean_helpers` normalization exports are removed.

Dense and local scoring share `orientation_priors.relion_translation_prior_center`;
the duplicate `relion_local_translation_prior_center` entry point has been removed.
`orientation_priors.relion_half_translation_prior_inputs` builds one half-set's
score and sigma-offset prior centers, the zero-centered cold-start engine
center and the prior translation grid for both the regular iterations and the
final all-data pass; the controller keeps the search base, the per-half sigma
offset and the dense-only score log-prior call. The local adapter receives
its own center array, and the engine center aliases the sigma center.
Both use `(prior - rounded_old_offset) / pixel_size`, as before. The separate
`relion_sigma_offset_prior_center` serves sufficient statistics and keeps its
pixel-space formula without that division.

Split local bucket preparation belongs to
[`local_preprocessing.prepare_local_bucket`](../../recovar/em/local/local_preprocessing.py).
It owns mask/cache selection, CTF weighting, translation operands and batch norms;
`local_big_jit` retains the compiled preprocessing primitive and fused kernel.
Masked/unmasked reconstruction share one exact BPref translation operation.
Callers and operand-capture tests import the preparation owner directly. The
local engine retains execution scheduling and one final `LocalEMResult` assembly;
profile construction and synchronization run only when requested.

Raw and processed-image cache limits belong to
[`local_caches.py`](../../recovar/em/local/local_caches.py).
Bounded RELION projection caches belong to
[`local_projection_cache.py`](../../recovar/em/local/local_projection_cache.py):
budget parsing, stable bucket sorting/grouping, rotation-ID mapping and chunked
projection construction share that owner. `plan_cache` returns a
`ProjectionCachePlan` containing the ordered buckets, groups and capacity
metadata. Positive requested capacity still sorts buckets even when the group
limit subsequently rejects caching. The local engine retains eligibility,
layout-ID storage, buffer construction, timing, group advancement and release. Only mapped
valid rows may be consumed; unused capacity keeps its existing uninitialized
padding. Cache consumers and tests import the owner directly, without engine
re-exports. Importing it does not initialize execution controllers.
Profile fields and `LocalBucketProgress` belong to
[`local_timing.py`](../../recovar/em/local/local_timing.py).
The reporter owns progress counters, environment cadence and log formatting;
the engine marks completed buckets and forces the final message at the original
execution sites. Importing this owner does not load execution modules.
Their unused local-engine re-exports have been removed. Tests import cache
limit names directly from their owner.

The former runtime `compute_e_step_weights` API had only test consumers.
Its materialized dense posterior implementation is preserved in
[`tests/helpers/dense_posterior_reference.py`](../../tests/helpers/dense_posterior_reference.py).
The adaptive-oversampling tests still compare its complete posterior with active
significance paths. This reference keeps separate orchestration but shares
production preprocessing/scoring kernels; it does not independently validate
those kernels. Production significance belongs to `helpers/significance.py`.

Pure convergence-policy cases live in `tests/unit/test_convergence.py`; collecting
this module does not import the iteration controller. Full iteration smoke tests
remain in `test_refine_relion_mode.py`.

The first-iteration winner-take-all dispatcher lives with its grid builder in
`firstiter_cc.py`. It calls the batch planner and K-class engine directly for
both K=1 and K-class scoring. The controller supplies its logger and chooses
whether the batch clamp also updates the caller’s argument dictionary.

Precision selectors belong to `helpers/dtype_policy.py`. The controller passes
its existing static argument mapping; diagnostic iteration selection still
reads the environment at call time. Moving the selectors does not evaluate a
second set of import-time defaults or change any selected dtype.

Angular-grid order policies belong to `helpers/convergence.py`: exhaustive-grid
capping, final parent/fine orders, perturbation order and direction-prior order.
The controller supplies the active state and captured final-sampling metadata;
the helpers preserve their distinct order choices.

For an extraction, identify the actual boundary first: array layout, casts,
reduction order, JIT scope, device placement, buffer ownership and returned
statistics. Preserve those contracts during structural cleanup. The
[EM development guide](../../recovar/em/AGENTS.md) and
[mathematical algorithm map](../math/relion_refinement_algorithm.md) describe the
validation ladder and scientific state transitions.

## Diagnostics and reusable evidence

Diagnostic scripts compare specific captures, layouts and policies. Similar
names or similar-looking reductions are insufficient evidence of duplication.
Keep an independent numerical reference separate from the implementation it tests.

[`tests/helpers/sparse_pass2_test_support.py`](../../tests/helpers/sparse_pass2_test_support.py)
retains six former runtime helpers used only by tests: cached/packed scoring
variants, a lane-tree wrapper, pair normalization and materialized noise-row
gathers. These share production primitives and are comparison utilities, not
independent numerical oracles. The tests keep their separate NumPy references;
production imports no test helpers.

Common transport and command mechanics have narrow owners:

- [`file_hash.sha256_file`](../../recovar/utils/file_hash.py) hashes files in
  8 MiB blocks for RECOVAR-dependent diagnostics. A hash alone does not make a
  mutable file immutable or validate a manifest.
- [`json_utils.to_jsonable`](../../recovar/utils/json_utils.py) converts NumPy
  values, paths and nested containers. Finite-value and report acceptance rules
  remain the caller's responsibility.
- [`scripts.file_hash.sha256_file`](../../scripts/file_hash.py) supplies the same
  8 MiB hashing contract to standalone diagnostics without importing RECOVAR or
  JAX. Keep the runtime helper in the installed package and this script helper
  usable through both direct entry points and package imports.
- [`scorecard_cli`](../../scripts/scorecard_cli.py) supplies print/write/check
  handling for compatible historical scorecards. Each renderer retains its own
  fixed case inventory, validator and Markdown format. These CLIs remain usable
  without importing the scientific environment.
- [`scorecard_validation`](../../scripts/scorecard_validation.py) validates the
  paired baseline/treatment cases shared by four historical EM scorecards.
  Their fixed inventories, Markdown rendering and refusal to overwrite an
  existing report remain in the individual scripts.

Use `python -m scripts.<name>` from the checkout for diagnostics that import
other script modules. Some older direct-file entry points still fail their
imports; the current review records that debt rather than treating failed help
commands as successful checks.

The [benchmark contract](benchmarks.md) defines source, fixture, library,
quality and performance evidence. [Current EM status](em_status.md) separates
the selected source from historical results and records open qualification gaps.
Historical scorecards describe their pinned runs; they do not qualify a new
checkout merely because the same report can still be rendered.

### Replay state diagnostics

`helpers/state_swap_probe.py` owns the supported component variants and CLI
validation. It can enumerate variants without importing the refinement
controller. `helpers/state_swap_runtime.py` owns snapshot copying, map-amplitude
scaling and restoration of the selected components. The controller still owns
when the snapshot is taken and applies it after the RELION replay override.

Snapshots preserve the existing ownership contract: array inputs are copied,
while `state_fields` is a shallow copy of `state.__dict__`. Changing that
ownership, the ordered return tuple or the restoration sequence requires its
own behavior review. The in-memory scoring-state inventory and overwrite guard are owned by
`diagnostics/frozen_boundary.py`, alongside the sealed-boundary loader.
The controller takes and checks those snapshots at the existing boundaries.

Captured sampling grids belong to `diagnostics/relion_replay.py`, which
also applies replay state overrides. Its helpers construct Euler/translation
grids, canonical coarse rotation IDs and direction log priors directly from
sealed sampling metadata. They preserve the recorded direction/psi order and
convert translations from Angstroms to pixels using the supplied voxel size.
The controller selects when to use these grids.

Refinement now receives one `RefinementOptions` container. Its groups own
scheduling, adaptive search, parity behavior, local search, class setup, replay,
diagnostics and batching. `refinement_options.with_validated_sampling_schedule`
owns explicit current-size/HEALPix schedule admission and shallow option copies;
the entry point invokes it before starting the loop. Option construction does
not trigger these checks. `helpers/iteration_history.py` owns the per-iteration
history lists and their established result-dictionary keys. Its noise/tau2 recorder
also owns host diagnostic formatting: it prepares all float64 shell fields
before appending, retains aliases to existing float64 shell arrays, and stacks
the halves into a fresh array. `mean_helpers._noise_radial_history` constructs
the radial history used by initialization and replay; pixel-noise normalization
and estimation remain with their callers. The controller
still chooses when each snapshot is recorded.

Precision is explicit at extracted boundaries: replay grids, resolution
curves and scoring-output adapters receive the caller's dtype. These helpers
do not import the controller to discover runtime settings. PR180's numerical
changes and their qualification state are tracked on the
[EM status page](em_status.md).

`helpers/convergence.py` owns angular-refinement state transitions, including
validation and application of explicit HEALPix schedules used for oracle runs.
The controller selects the iteration's requested order; the convergence helper
advances through the existing angular and translation updates without coarsening
an active state.
It also owns the approximate-accuracy convergence gate and its environment
overrides. The controller supplies its logger so malformed-override warnings
keep their existing routing.

`helpers/resolution.py` owns current-size growth inputs and first-iteration
resolution rules: the inclusive FSC/data-vs-prior boundary, raw versus corrected
K1 scheduling, the initial high-resolution cutoff, and the tau2 reporting taper.
It also owns expectation-boundary coarse sizing: replay selects the incoming
HEALPix order, and local pass 1 sizes its Fourier window from that order while
child expansion retains the updated order. Callers import these policies directly.
Initial FSC/low-pass resolution seeding also lives here, with explicit FSC dtype
and the shared ini_high shell calculation. The controller retains replay/FSC/
low-pass precedence; seeding preserves input copies and state-assignment order.
The controller retains their timing within the refinement loop. The pure
scheduling cases live in `tests/unit/test_resolution_scheduling.py`; the
reconstruction/taper ordering check remains with the controller tests.

`refinement/projector_preparation.py` prepares RELION reference slabs
for the controller's scoring calls. It owns native reference conversion, cache
keys and files, optional dumps, and validation of captured projector geometry.
`relion_replay.py` retains the captured-state type and parser; the controller
selects the native or captured path and passes the resulting slabs to scoring.

Captured sampling and projector-state tests live in
`tests/unit/test_relion_replay_state.py`. They exercise the replay and projector
owners directly, including immutable copied arrays and suppression of external
metadata reads. End-to-end controller behavior remains in
`test_refine_relion_mode.py`; capture-file parsing remains in
`test_relion_projector_capture.py`.

Import execution entry points explicitly from their owners:

```python
from recovar.em.refinement.iteration_loop import refine_single_volume
from recovar.em.classification.k_class_results import KClassEMResult
from recovar.em.classification.k_class import (
    run_dense_k_class_em,
    run_local_k_class_em,
)
```

The package initializer does not re-export these names. The CPU fast guard
checks that importing replay, normalization, projector, result and diagnostic
helpers leaves the controller, K-class orchestration, dense/local engines and
sparse scoring unloaded. Existing callers already import from these owners;
the definitions and their serialized module identities are unchanged.

## Ground-truth reporting

[`vdam/gt_registration.py`](../../recovar/em/vdam/gt_registration.py)
owns the optional CPU rigid fitter and immutable fit-once transform. The existing
[`gt_metrics.py`](../../recovar/em/vdam/gt_metrics.py) keeps its legacy
rotation-only alignment API and result type. The reporting CLI
[`evaluate_ab_initio_gt.py`](../../scripts/evaluate_ab_initio_gt.py) opts into the
new fitter or applies a saved transform without fitting. [The reporting guide](gt_reporting.md)
explains geometry, common-frame comparisons and limitations. These are diagnostic
reporting tools; E/M execution, precision and quality gates are independent.
