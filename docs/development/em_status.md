# Current EM development scope

Current decisions belong here; update this page when a decision changes, not for
every test or publication. Detailed receipts belong behind links. The
[previous page is preserved byte-for-byte](em_cleanup_history_20260910_a2ab056cb.md)
at `a2ab056cb`, including one paragraph per earlier checkpoint; its historical
next actions are superseded here.

## Milestone and invariants

Complete RECOVAR cleanup **EM first, GUI excluded**, before new-engine work:
remove proven dead/duplicate code, clarify owners/APIs and establish reproducible
synthetic, real and exactly-K4 accuracy/performance benchmarks. The goal remains
incomplete. See [cleanup plan](cleanup_plan.md), [codebase map](codebase.md),
[benchmark contract](benchmarks.md) and [EM operating rules](../../recovar/em/AGENTS.md).

Structural changes preserve defaults, casts/reductions/JIT order, buffer lifetime,
non-EM APIs, saved formats and independent references. Canonical source Euler
angles and host pixel geometry stay metadata; derive computation arrays from them.
Double is diagnostic, not a production remedy or proof of noise. No tolerance or
baseline changes. User priority is short-prefix parity then final FSC, with up to
2× native runtime provisionally; this does not waive accuracy or completion gates.

## Source and ownership

- Primary: `/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_structural_cleanup_20260907`,
  branch `codex/integrate-pr180`, published tip `a2ab056cb5dfac9b1152692f15a4ce683f9fc9b2` (2026-09-10).
  Recheck HEAD, diff and untracked files; earlier trajectories do not qualify it.
- **em_clean is sole integrator/publisher**, [draft PR179](https://github.com/ma-gilles/recovar/pull/179)
  on pinned PR158 base `44d770de3f9336ab2f3f6a34203394bae8d1aeed`.
  [Compact handoff](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/pr179_coordination/CURRENT_TASK.md)
  and [ownership board](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/pr179_coordination/README.md)
  govern assignments. Consult peer status for current numerical work and jobs;
  historical paused states do not establish current ownership or availability.
- Preserve frozen checkouts, jobs, inputs and binaries. No shared RELION writer
  lock is granted here. em_clean jobs are listed in
  [status/em_clean.json](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/pr179_coordination/status/em_clean.json);
  peer job state must be checked before acting, not inferred from historical
  records. Never duplicate peer work.
  Leave local GPU0 free; only immediately idle GPUs1–3 by UUID; respect Slurm visibility.

## Engineering work and recent evidence

Each row is one commit on draft PR179 with an exact old/new comparison, the CPU
fast guard and, where the controller path changed, the 502-case controller
panel (one GPU-only skip). "Structural" rows change no arithmetic; "fix" and
"semantics" rows are labeled and kept in their own commits. Receipts hold the
case counts, provenance and limits.

| Commit | Change | Kind | Receipt |
| --- | --- | --- | --- |
| see receipt | remove the unused `convergence.SIGMA_CUTOFF` and seven unreferenced helpers in EM parity/diagnostic scripts; records the pre-existing sealed static-argument drift in `run_local_mstep_donation_ab.py` | dead code, token scan | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/dead_script_helpers_20260911/result.json) |
| see receipt | `preprocessing.uses_relion_cuda_image_preprocessing` / `relion_preprocess_backend` own the RELION CUDA preprocessing detection (local engine and InitialModel adapter shared two inline copies); the controller fails closed at setup when the fresh K=1 defaults run without it; the fast tier's K1 cold start uses the production `relion_cuda` backend | fix (fast tier K1 cold start) + structural | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_cuda_preprocess_owner_20260911/result.json) |
| see receipt | remove unused EM APIs whose only callers were their own tests: `oversampling.compute_pass2_stats` (296 lines), `sampling.get_healpix_children`/`get_oversampled_rotation_grid`, `shape_buckets.ShapeBucket`/`dense_shape_bucket`/`local_shape_bucket`, `resolution.should_skip_adaptive_pass2`, `fourier_window.make_frequency_radius_map_half`, `flat_local_rows.gather_flat_local_rows`, with their tests | dead code, reference scan | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/dead_em_apis_20260911/result.json) |
| see receipt | `sparse_pass2_bucketed._sparse_pass2_window_setup` builds the forward model, Fourier windows, x-half reconstruction indices and windowed-prepare decision for both sparse pass-2 scorers | structural, 32 output/call/log-identical | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/sparse_pass2_window_setup_owner_20260911/result.json) |
| see receipt | `k_class._override_class_assignments_with_coarse_winner` replaces pass-2 class assignments with the coarse global winner for both adaptive K-class pass-2 paths | structural, 8 replace-kwargs cases | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/kclass_coarse_override_owner_20260911/result.json) |
| see receipt | the fused abs2-on-demand local score pass derives its reconstruction support through the existing `_support_from_local_probs` owner instead of an inline copy | structural, bitwise on CPU | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/local_support_owner_20260911/result.json) |
| see receipt | the adaptive and single-pass dense half-scoring calls share one keyword set (`dense_half_kwargs`, 41 shared, 8 adaptive-only) with one post-call block and the single-pass manifest dump | structural, exec-equivalent branches incl. manifest bytes | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/dense_half_kwargs_owner_20260910/result.json) |
| see receipt | `_advance_relion_perturbation` advances RELION's SamplingPerturbation (seeded `random_seed + iteration` or generator path) for both passes | structural, 36 value/type/log-identical | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/perturbation_advance_owner_20260910/result.json) |
| see receipt | `_ExpectedAccuracyInputs`, `_expected_accuracy_class_ids`, `_estimate_half1_expected_accuracy` share RELION's expected-accuracy inputs and half-1 class labels between both passes | structural, 32 kwargs-identical | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/expected_accuracy_inputs_owner_20260910/result.json) |
| `a2ab056cb` | `half_scoring._adaptive_pass2_grids` builds RELION's two-pass trial grids for the K=1/K-class routes and the pose-grid rebuild (3 sites) | structural, 8+4 bit-exact | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/adaptive_pass2_grids_owner_20260910/result.json) |
| `05227488b` | `sparse_pass2_bucketed._relion_powerclass_noise_terms` selects `highres_Xi2`/norm terms for both sparse scorers | structural, 16 wiring | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/powerclass_noise_terms_owner_20260910/result.json) |
| `583d8bd37` | `local_em_engine._accumulate_packed_noise_chunk` owns per-chunk noise/norm/scale accumulation (two 42-line bodies) | structural, 8 wiring | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/packed_noise_chunk_owner_20260910/result.json) |
| `5618216f3` | one operand owner for the four RELION `powerClass` reproductions (`_relion_powerclass_packed_image`, `_operands`, `_native_spectrum_highres`) | structural, 56 bit-exact + 6 error paths | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/powerclass_operand_owner_20260910/result.json) |
| `e5dbad66d` | remove three unreferenced helpers; records the pre-existing `picked_frequencies` undefined name | dead code | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/dead_em_helpers_20260910/result.json) |
| `4ec1778cf` | K-class scoring at positive oversampling always keeps RELION's two-pass expectation (no direct-engine fallback at full coarse size) | semantics, 4 decision rows | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/kclass_adaptive_two_pass_20260910/result.json) |
| `f5b624f35` | `orientation_priors.relion_local_search_sigmas` owns local-search prior widths for both passes | structural, 144 exact | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/local_search_sigma_owner_20260910/result.json) |
| `bd602096a` | fresh-InitialModel coarse Gaussian FFI default only with its RELION projector operands (`_coarse_gaussian_ffi_default`) | fix (fast tier K1 cold start) | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/fast_tier_repairs_20260910/result.json) |
| `b6db2f8ef` | K-class scale groups route through the accumulating engine at oversampling 0 (`_dense_uses_adaptive_engine`) | fix (fast tier strict K-class) | same receipt |
| `da7222f08` | `_exact_local_fine_grid`, `_local_search_mstep_rotations`, final-pass `relion_local_pass1_current_size` shared by both passes | structural, 37 exact | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/local_fine_grid_owner_20260910/result.json) |
| `67ee54ab1` | final-pass local pass-1 size only under parent expansion (previously `UnboundLocalError`; RELION keeps `coarse_size == current_size`) | fix, AST guard | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/final_local_pass1_size_fix_20260910/result.json) |
| `593b6adda` | `_initial_coarse_grids` and `_relion_base_translation_grid` own the controller's coarse trial grids (7 sites) | structural, 40+64 exact | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/initial_coarse_grid_owner_20260910/result.json) |
| `d9a23ceb1` | fast tier K4 cases take a dispatch-capable oracle/schedule through `EM_PARITY_FAST_K4_*`; 5k/128 dispatch oracle 13711156 | fixture | [note](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_5k128_dispatch_oracle_20260910/NOTE.md) |
| `56bf027a4` | K=1 scale groups route through the adaptive engine at oversampling 0 instead of raising | fix (user-delegated) | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k1_scale_group_routing_20260910/result.json) |
| `9870438cd`, `b8e97709b` | `_relion_cuda_corr_img_from_native_noise_variance` accepts `output_dtype`; K1 completion launcher uses `relion_cuda` images | fix | [K1 run](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_completion_k1_9870438cd_h100_20260910) |
| `cd119c900` | `orientation_priors.initial_direction_priors_from_snapshot` | structural, 16 exact | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/snapshot_direction_prior_owner_20260910/result.json) |
| `40f4ec622` | `_relion_mstep_source_eulers`, `_perturbed_trial_grid` shared by both passes | structural, 40 exact | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/trial_grid_perturbation_owner_20260910/result.json) |
| `ebf607858` | `orientation_priors.relion_direction_log_priors_for_half` (RELION `pdf_direction` rules; user decision) | semantics, 64/72 exact + 8 classified | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/direction_log_prior_owner_20260910/result.json) |
| `4a2a33de6` | `convergence.concatenate_assignments*`, `mean_helpers._relion_pmax_normalization_mass_per_half` | structural, 48 exact | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/convergence_input_owners_20260910/result.json), [asymmetry review](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/convergence_input_owners_20260910/prescoring_direction_prior_asymmetry_review.md) |
| `2a81aeafb` | `mean_helpers.update_learned_direction_priors`; history owns snapshot copies | structural, 768 exact | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/direction_prior_learning_owner_20260910/result.json) |
| `50b4986e0` | half join + Class3D tau2 (`mean_helpers`), `relion_replay.read_optimiser_accuracy_replay`, `orientation_priors.relion_half_translation_prior_inputs` | structural, 24/240/576 exact | [receipts](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/class_tau2_lowres_join_owner_20260910/result.json) |

Earlier batches (reference/alias cleanup, dead Fourier chain, replay selection,
finalization policy, PPCA helper removal, particle-state reporting, batch-plan
logging, convergence and sampling/initialization batches, canonical Euler/pixel
repairs) keep their paragraphs in the
[preserved page](em_cleanup_history_20260910_a2ab056cb.md#engineering-work-and-recent-evidence).

### Qualification state (frozen checkpoints, H100)

- **Fast parity tier** on frozen `d9a23ceb1` (job 13711329, with the 5k/128
  dispatch-oracle K4 fixture): 4 of 7 pass. K1 replay and perturbation replay
  pass (the K=1 routing repair holds); K-class cold start and strict oversampled
  cold start pass with the new fixture. K-class replay keeps its known |ΔPmax|
  0.999999 failure. K1 cold start and strict K-class cold start failed on the
  two gaps repaired in `bd602096a` and `b6db2f8ef`. The rerun on frozen
  `bd602096a` (job 13713500) passes 5 of 7: both strict K-class cold starts now
  pass, so the K-class routing repair is validated on the GPU; the K-class
  replay case keeps its known failure; the K1 cold start now fails deeper, in
  the sparse pass-2 bucket preparation ("exact RELION BPref operands require
  RELION CUDA preprocessing"), because the K=1 fresh-refinement defaults assume
  the `relion_cuda` image backend while the test uses the script's `host_numpy`
  default (the K1 completion launcher already passes `relion_cuda`). Earlier tier runs 13704244
  (`400ad81e4`, 6 of 7 failed) remain preserved.
- **K1 100k/256 completion** on frozen `9870438cd` (job 13709837, exclusive
  H100, `relion_cuda` images) **completed without qualifying**: 17 iterations
  in 24,147 s against RELION's 12,695 s (1.90× wall; 4.14 vs 7.88 images/s),
  never converged (RELION converged at iteration 15), so the final all-data
  pass was skipped and the summarizer (job 13709838) reports `failed` with no
  RECOVAR-vs-GT FSC. Resolution stalled at 8.77 Å from iteration 14 while the
  average Pmax fell to 0.12 at HEALPix order 7; global iterations 2–7 took
  45–60 min each (E-step dominated), local iterations 7–14 min. This is a
  quality failure and a runtime failure of that frozen checkpoint; the
  trajectory diagnosis belongs to the numerical workstream. A diagnostic
  comparison of the pre-final maps with the summarizer's own metrics gives
  RECOVAR merged-vs-GT FSC AUC 0.403 and mean FSC over shells 1–8 of 0.77,
  against RELION's final map at 0.491 and 0.996, and RECOVAR-vs-RELION low-shell
  FSC 0.78: the maps differ already at low resolution, consistent with the
  collapsing Pmax, so this is not a final-pass artifact.
  [Summary](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_completion_k1_9870438cd_h100_20260910/summary.md),
  [diagnostic FSC](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_completion_k1_9870438cd_h100_20260910/diagnostic_prefinal_fsc.json).
  Attempts 13704245 (`host_numpy` backend) and 13709226 (`output_dtype`
  TypeError) are preserved as failed.
- **Exactly-K4 100k/256**: the dispatch-logging RELION oracle 13708102 completed
  (1 h 15 min, schema-2 log with 1.5 M rows); the schema-3 schedule
  (oracle_id `a220abb55f4`) was built by 13711316, whose launcher step could not
  resolve the RELION GPU module on a CPU node and was re-run from the login
  node. The RECOVAR run (13712371/13712372/13712373, frozen `400ad81e4`) is
  queued. Attempt 13704399 (legacy dispatch log) is preserved as failed.
- No gate has moved. Moving HEAD is qualified only by CPU comparisons and the
  controller panel; production-F32 quality, convergence/finalization and
  matched-GPU runtime remain open (next section).

## Unresolved validation gates

**Moving HEAD is not scientifically or performance qualified.** Map passes below
apply only to their frozen source and fixture; strict state is a separate gate.

| Evidence | Established result and remaining limit |
| --- | --- |
| [Frozen4f9 synthetic K1, 3k/128, full200](evidence/vdam-full200-4f9-20260909/README.md), job13653485 | All201 map gates pass: minimum cross-AUC .9997253945, worst GT delta −.000251216. Strict state: 3,726 coarse-count mismatches (first32), selected Pmax gaps from61, maximum .699197, late pose/origin differences. Missing fine support/margins; no noise waiver |
| [Frozen5ca9 real10076 K1, 10k/256, prefix20](evidence/vdam-canonical-pixel-prefix20-20260910/README.md), job13664081 | All21 cross-map gates pass, minimum .9999968978; map0 byte-exact. 57 count differences from3; 294 selected-row Pmax gaps >=.001 from13. No GT, timing ratio or full200 acceptance |
| [Frozen5ca9 real10076 full200](evidence/vdam-real-full200-5ca9-20260910/README.md), job13664965 | All four final cross-engine AUCs .968354–.974833 fail .999; native/native .975107 also fails. Twelve raw-map AUCs at100/200 independently exact; no GT/repeat-band waiver. Descriptive process ratios1.277–1.447×; not current-source acceptance |
| [Frozen5ca9 real10076 K1 InitialModel, 100k/256, full200](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real100k_admission_20260910/review.json), job13683192 | Saved201 summary independently recomputed; worst115 and final200 raw-map AUCs exact. Minimum 0.920691882,140/201 below .999, final .993747847: fail. Paired process ratio 2.148386× exceeds provisional2×. Raw state at21 already has a count difference; at32 Pmax gap .02103 and2 count differences. No exact-through31, chaos/noise, GT or moving-source acceptance; full native/input closure remains open |
| [K4 saved comparisons](benchmarks.md#k4-audit-integrity-and-reviewed-saved-comparisons--september-10), reported5ca9, 20 iterations | 3,200 saved curves/classes rechecked. Original synthetic minimum .99768012 and real .64899880 fail. Synthetic repeat passes; closest real repeat .99833338 still fails at20. Source/build and raw-map admission incomplete; native variation is not a waiver |
| [Robustness screening](benchmarks.md#robustness-gt-curve-review--september-10) | 64 GT integrals rechecked; original13/100,22/200,32/200 fail −.002 screening. Case22 fails8/12 repeat comparisons. Independent per-map alignments are not the prespecified shared-transform GT gate; source/build admission remains open |

[E6 row942 coarse-cap review](evidence/vdam-coarse-cap-tie-20260910/README.md)
confirms measured0–3 ULP competing scores and exact threshold/support agreement
in six instrumented candidate histories. This local near-tie evidence does not
establish the incoming-state cause, native matched-input parity or a trajectory
waiver; six-digit scalar equality in the producer report is corrected.

[Saved-state repeat audit](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/same_state_admission_20260910/result.json)
independently checks nine candidate metadata comparisons from reported5ca9/job13665172.
All200 support counts match in each comparison; t12 row2765 differs in Pmax by
1.3709068e-6, and saved noise/power/BPref summaries differ. This is not exact
full-state agreement or proof of an M-step-only cause; native inputs/maps and
complete build provenance are outside this audit.

Rejected explanations: canonical pixel narrowing was a demonstrated metadata bug
([scalar causal gate](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_real10076_pixel_prefix4_20260909/RESULTS.md)),
not a reason to promote arithmetic to double. Case22 candidate repeats are **not
byte-identical**: 97,376 final voxels differ. Correlation/repeat-band summaries
cannot close FSC or discrete gates. No deterministic-accumulation rewrite follows
without actual score/support evidence. Counter187 is
[monitor-only in fixed200 InitialModel](evidence/vdam-full200-4f9-20260909/late_counter_scope.md),
not an ordinary auto-refine/K4 waiver; four native adaptive fields are uncaptured.

The dense/local fast guard now rejects undefined names before JAX startup.
A broader static scan found existing `picked_frequencies` use-before-assignment
in `recovar/em/heterogeneity.py:971`; outside this guard's scope, unfixed.
A [caller audit](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/legacy_pca_callers_20260910/result.json)
finds no repository caller of that78-line legacy function and no package export.
The main PCA pipeline uses a distinct implementation. An unapplied removal
candidate is preserved; external/dynamic uses remain unverified.
[Guard check and exact finding](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/undefined_name_guard_20260910/result.json).

Historical failures stay open: API13641893 has6 failures (older13634313:12),
normalization13636581 has4 GPU bytewise failures, PR180 CPU has25 failures, and
K1 matched-noise replay has6 Pmax failures with incomplete margins/oracle identity.
K4 job13560356 failed2:0 at10/class2. Partial repairs do not qualify those panels.
[Complete failure ledger](em_cleanup_history_20260909_f0a8804e2.md#unresolved-validation-gates).
The optional M-step rotation override cleared by class-prior layout remains a
[separate correctness question](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k_class_input_owner_20260909/result.json).
Buffer lifetime changes likewise require measured evidence, not cleanup assumptions.

Completion still needs current-source robustness, real confirmation, production-F32
inventory, >=100k/256 K1 and exactly-K4 matched-GPU pairs, Hungarian per-class
results, convergence/finalization and shared downstream checks. Keep
[quantitative gates](../math/em_parity_program.md) unchanged. Preserve the reviewed
final-grid-correction default; its strict-target discrepancy needs separate qualification.

## Frozen jobs and representative performance

Frozen4f9 full200 H100: 452.295/303.905s = **1.4883×**, one3k/128 pair with asymmetric
harness/I/O and unmeasured contention/memory. Older8ab100k A100 paired ratio **1.831839×**
predates compact CTF and is quality-unqualified. Reported K4 slowdowns (~20× synthetic,
~8.9× real) need source-closed timing review. The completed K1 100k/256 run on
frozen `9870438cd` (H100, job 13709837) took 24,147 s for 17 non-converging
iterations against RELION's 12,695 s for its converged auto-refinement
(1.90× wall, 4.14 vs 7.88 images/s; global iterations 45–60 min, local 7–14 min);
it is a runtime measurement of a run that failed quality admission, not a
qualified ratio. None establishes moving-tip performance.
For newer pending evidence consult [VDAM status](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/pr179_coordination/status/vdam.json);
this page does not schedule or authorize duplicate runs.

## Next action and efficient execution

Admit the queued fast-tier rerun (13713500), the K1 completion (13709837) and the
exactly-K4 completion (13712372) against the [benchmark contract](benchmarks.md)
when they finish, recording every result including failures. Between results,
continue one bounded structural package at a time from the cleanup plan.
Remaining candidates after the September 10–11 packages: a named result type
for the four positional big-JIT output layouts unpacked in `local_em_engine`
and returned by `local_big_jit` (design change at the hottest kernel boundary),
unifying the K=1 and K-class adaptive routes in `half_scoring`, the diagnostic
capture keyword lists in `local_em_engine`, seven forwarding aliases found by
the wrapper scan (`sampling.get_rotation_grid_at_order` has 44 callers), and
the kept test-facing helpers listed in the dead-API receipt. Each package keeps
an exact old/new comparison,
the CPU guard, the controller panel when the controller path changes, and one
combined validation/publication per package. Keep first-divergence numerical
diagnosis and remaining real-full200/source/build admission separate; peer
summary labels are not acceptance.

Use [the workflow](agent_workflow.md), existing `scripts/em_work_package.py` receipts,
focused CPU tests and one combined validation/publication per package. Reuse the
[verified native CPU binding](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/native_binding_cpu_restore_20260909/result.json)
only after checking its source/dependency/loaded-file pins; do not rebuild it implicitly.

## Agent efficiency package — September 9

Direct Astra medium remains the default for this workstream. Built-in delegation
is disabled; isolated Terra smokes passed but the reviewed small-task pilot used
1.625× input and 1.667× elapsed time versus direct Astra. No token savings proved.
Compact handoffs, scripted receipts and batched publication continue. No automatic
wakeup/model-polling promise. [Setup, measured limits and recovery](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/pr179_coordination/delegated/README.md).
