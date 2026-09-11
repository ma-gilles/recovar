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
| `11bc4f0c2` | EM handoff integrated: the RELION projector crop is sized from the particle-image window (not the model sphere) and local-search rotation ids and hard assignments are int64; a numerical fix that is a no-op when the optics pixel equals the model pixel (all synthetic fixtures unchanged); three projection-cache test callers migrated to the peer's `rows_for_bucket` contract | peer numerical fix | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_crop_fix_integration_20260911/result.json) |
| `8125096d1` | `types.sparse_pass2_result` owns the sparse pass-2 return tuple and the order of its optional entries (statistics, score log-Z, noise, source Eulers); the two pass-2 functions had four branches building it by concatenation, and the historical rule that the score log-Z rides with the statistics is now stated once | structural | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/sparse_pass2_result_tuple_owner_20260911/result.json) |
| `96bcc0990` | `local_big_jit._LocalBigJitCore` names the 22-value core that every big-JIT result carries; the four producer sites build it and `local_em_engine` unpacks it once (core, then the deferred / source-VDAM / M-step-tensor extras per layout) instead of four positional 22-name unpacks; the wire format stays a plain tuple for the fixed-capacity scan carry window and the BPref transaction queue | structural | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/local_big_jit_result_layout_owner_20260911/result.json) |
| `f3a091a09` | `types.read_sparse_pass2_result` reads the positional sparse pass-2 tuple next to its builder and returns `SparsePass2Output` (absent optional entries `None`); the two index-walking consumers in `k_class` now use it | structural | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/sparse_pass2_result_reader_owner_20260911/result.json) |
| `4486149b1` | `k_class._PerClassResults` collects the per-class outputs of the dense and local full-image K-class runners (accumulators, int32 assignments, statistics, noise, best poses, Euler angles/profiles, new means) in class order, replacing two hand-maintained sets of eight parallel lists | structural | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/kclass_per_class_results_owner_20260911/result.json) |
| see receipt | `k_class._pass2_support_log_args` owns the 21 support-statistics values the three adaptive pass-2 routing logs print after their own leading arguments | structural | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/kclass_support_log_args_owner_20260911/result.json) |
| see receipt | `run_local_em_exact` converts each bucket's translation log-prior, rotation mask and sample mask to device arrays once before its fused-score chain instead of in every branch (six call sites plus the deferred reconstruction mask); AST identical after substituting the bindings back | structural | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/local_bucket_operands_owner_20260911/result.json) |
| see receipt | `local_em_engine._relion_local_projector_flat` owns the RELION-projector block (radius check, slab, output size, pixel selection, disk mask, texture/floorf quirks) shared by the bucket and packed-noise projections; each caller keeps only its pixel-selection rule | structural | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/local_relion_projector_flat_owner_20260911/result.json) |
| see receipt | `local_em_engine._packed_reconstruction_rows` owns the gather-and-zero of packed reconstruction rows (take along the packed indices, zero the padding rows) that eleven sites repeated; jaxpr identical to the inline sequence | structural | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/local_packed_rows_owner_20260911/result.json) |
| see receipt | `sparse_pass2_bucketed._pass2_window_setup` and `_fine_translation_prior_2d` own the window/precision setup and the fine translation-prior expansion that both bucketed pass-2 entry points repeated verbatim | structural | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/pass2_window_setup_owner_20260911/result.json) |
| see receipt | `local_em_engine._packed_bucket_rotations` owns the host packed-rotation gathers (scoring and M-step rotations along the packed take indices, device copies of the indices and pack mask) that four sites repeated (the source-VDAM float32 cast is a keyword) | structural | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/local_packed_rotations_owner_20260911/result.json) |
| see receipt | the deferred exact-noise core and kernel calls in `run_local_em_exact` share one binding of their eight common keywords instead of two hand-maintained copies; AST keyword mapping identical | structural | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/deferred_noise_shared_kwargs_owner_20260911/result.json) |
| `3a3adc0f8` | `sparse_pass2_bucketed._gaussian_algebraic_score_terms` owns the batched algebraic Gaussian score terms (weighted cross einsum, projection norm, prior-free and prior-added scores) shared by the production algebraic scorer and its components variant; traced programs identical | structural | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/gaussian_algebraic_terms_owner_20260911/result.json) |
| `a0b0a8141` | `pass2_diagnostics._optional_operand_row_fields` owns the ten optional RELION operand captures of the K=1 pass-2 dump (absent operands recorded as empty arrays or NaN); the selected-rows and effective-grid schemas shared two 57-line blocks | structural | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/pass2_dump_operand_fields_owner_20260911/result.json) |
| `4e025b481` | the dense engine binds its per-batch score-block keywords once (`score_block_kwargs`) after the last scoring-operand rebinding and passes them to both the pass-1 and pass-2 `_score_rotation_block` calls, which keep only their block projections | structural | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_engine_score_block_kwargs_20260911/result.json) |
| `9af19840f` | `projection._relion_projector_fftw_block` owns the RELION Projector call shared by the centered-row and indexed centered-row projectors (projector size clamp, scorer-rotation transpose, FFTW-row block); traced programs identical | structural | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/projector_fftw_block_owner_20260911/result.json) |
| `1e46d90d7` | `scoring._e_step_block_score_components` is the one owner of the cross/model-energy GEMM pair; the half residual scorer and both normalized-CC scorers no longer re-derive it (traced programs identical, ten HIGHEST-precision GEMM sites become four) | structural | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/score_components_owner_20260911/result.json) |
| `9e8ad731a` | `k_class._PerClassSubsetResults` assembles the per-class accumulators, assignments, statistics, noise and best poses of the dense and sparse firstiter-CC global-winner subset passes (empty-class zero fill and subset expansion were two 40-line pairs); the routes' host/device placement of empty-class zeros and engine outputs is recorded and unchanged | structural | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/kclass_subset_results_owner_20260911/result.json) |
| `65a7a9e5d` | `state_swap_runtime._StateSwapValues` names the fourteen iteration-state values the RELION replay override hands back; the probe builds the unchanged value once and returns it from both early exits (three positional 14-value returns and their helper removed) | structural | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/state_swap_values_owner_20260911/result.json) |
| `673eb84af` | `local_debug._requested_dump_rows` owns the dump gate (directory, pending ids, requested current sizes and iterations) and the bucket-row selection shared by the fused-posterior, score and noise-component dump writers | structural | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/debug_dump_rows_owner_20260911/result.json) |
| `4473f73ea` | `heterogeneity._fixed_rotation_covariance_images` owns the fixed-rotation covariance right-hand-side and normal-operator images shared by the Equinox and classic accumulators (two 40-line bodies; traced programs identical); `initial_model.layout._centered_bpref_sources` and `_bpref_slab_outputs` own the BPref source validation and the double-precision denormal clamp shared by the dense and RELION-x-half converters | structural | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/covariance_rhs_and_bpref_source_owner_20260911/result.json) |
| `e5029adb4` | `local_em_engine._exact_local_bpref_capture_static_kwargs` owns the twenty-eight fixed operands of the exact-local BPref contribution capture (absent raw data/CTF/mask fields and the run's M-step geometry, bound once when capture is active) and `_bpref_capture_priors` owns the candidate mask and prior-free scores; the fused and big-JIT capture sites keep only their per-bucket operands | structural | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/bpref_capture_static_kwargs_owner_20260911/result.json) |
| `164000d6b` | one owner for the adaptive engine call shared by the K=1 and K-class dense routes (`_adaptive_engine_shared_kwargs`), the sparse/dense pass-2 environment switch (`k_class._sparse_pass2_selected`, three inline reads) and the coarse pose collapse (`_coarse_pose_assignments`); six duplicated grid locals removed; route-specific keywords stay explicit | structural | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/adaptive_engine_call_owner_20260911/result.json) |
| `9e93d7657` | remove the uncalled legacy `heterogeneity.estimate_principal_components` (with its undefined `picked_frequencies`), a shadowing re-import, two unused `e_step` imports and two unused package re-exports; the EM package outside PPCA refinement is now free of ruff F findings | dead code | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/legacy_pca_removal_20260911/result.json) |
| `3e9acee61` | remove eighteen unused local assignments (ruff F841) across the EM package, including two windowed translation-phase tables the single-class sparse scorer computed at setup and never read | dead code | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/unused_locals_20260911/result.json) |
| `67b184b39` | remove the unused `convergence.SIGMA_CUTOFF` and seven unreferenced helpers in EM parity/diagnostic scripts; records the pre-existing sealed static-argument drift in `run_local_mstep_donation_ab.py` | dead code, token scan | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/dead_script_helpers_20260911/result.json) |
| `dd7d9b218` | `preprocessing.uses_relion_cuda_image_preprocessing` / `relion_preprocess_backend` own the RELION CUDA preprocessing detection (local engine and InitialModel adapter shared two inline copies); the controller fails closed at setup when the fresh K=1 defaults run without it; the fast tier's K1 cold start uses the production `relion_cuda` backend | fix (fast tier K1 cold start) + structural | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_cuda_preprocess_owner_20260911/result.json) |
| `74993efa9` | remove unused EM APIs whose only callers were their own tests: `oversampling.compute_pass2_stats` (296 lines), `sampling.get_healpix_children`/`get_oversampled_rotation_grid`, `shape_buckets.ShapeBucket`/`dense_shape_bucket`/`local_shape_bucket`, `resolution.should_skip_adaptive_pass2`, `fourier_window.make_frequency_radius_map_half`, `flat_local_rows.gather_flat_local_rows`, with their tests | dead code, reference scan | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/dead_em_apis_20260911/result.json) |
| `f55fa5f3d` | `sparse_pass2_bucketed._sparse_pass2_window_setup` builds the forward model, Fourier windows, x-half reconstruction indices and windowed-prepare decision for both sparse pass-2 scorers | structural, 32 output/call/log-identical | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/sparse_pass2_window_setup_owner_20260911/result.json) |
| `49e49ba1d` | `k_class._override_class_assignments_with_coarse_winner` replaces pass-2 class assignments with the coarse global winner for both adaptive K-class pass-2 paths | structural, 8 replace-kwargs cases | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/kclass_coarse_override_owner_20260911/result.json) |
| `6a04c9afa` | the fused abs2-on-demand local score pass derives its reconstruction support through the existing `_support_from_local_probs` owner instead of an inline copy | structural, bitwise on CPU | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/local_support_owner_20260911/result.json) |
| see receipt | the adaptive and single-pass dense half-scoring calls share one keyword set (`dense_half_kwargs`, 41 shared, 8 adaptive-only) with one post-call block and the single-pass manifest dump | structural, exec-equivalent branches incl. manifest bytes | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/dense_half_kwargs_owner_20260910/result.json) |
| `607c4439a` | `_advance_relion_perturbation` advances RELION's SamplingPerturbation (seeded `random_seed + iteration` or generator path) for both passes | structural, 36 value/type/log-identical | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/perturbation_advance_owner_20260910/result.json) |
| `a93a6d84c` | `_ExpectedAccuracyInputs`, `_expected_accuracy_class_ids`, `_estimate_half1_expected_accuracy` share RELION's expected-accuracy inputs and half-1 class labels between both passes | structural, 32 kwargs-identical | [receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/expected_accuracy_inputs_owner_20260910/result.json) |
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
  default (the K1 completion launcher already passes `relion_cuda`). The rerun on frozen
  `dd7d9b218` (job 13726940, della-h20g2 H100) ran six cases before the 60-minute job
  limit killed it (TIMEOUT at 1:05:30) during the strict oversampled K-class cold
  start: K1 replay passes (half correlation 0.99994, |ΔPmax| 1.9e-4, 63 s), K1
  perturbation replay passes (0.99963/0.99957, |ΔPmax| 0.003, 567 s), the K-class cold
  start (worst Hungarian class correlation 0.99984, 241 s) and the strict K-class cold
  start pass again, K-class replay keeps its known failure (|ΔPmax| 1.0,
  class-assignment accuracy 0.545, per-class map correlation 0.991/0.991), and the K1
  cold start now reaches its quality gates and fails them: half correlations
  0.9947/0.9944 against the 0.999 gate, iteration-3 average Pmax 0.887 against RELION's
  0.965 (gap 0.078, gate 0.01), sigma-offset carry-over passes (4.50 Å at iteration 2),
  and the run took 2444 s against the test's ~5-minute budget with the production
  `relion_cuda` fresh-K=1 bundle, which is why the strict oversampled K-class cold start
  (passing on `bd602096a`) was cut off. Not admitted; the K1 cold-start quality and
  runtime gap joins the K1 completion failure in the numerical workstream's hand-off
  ([record](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_fast_tier_dd7d9b218_h100_20260911/admission_record.json)).
  The rerun on frozen `11bc4f0c2` (job 13737449, della-h21g4 H100) completed all seven
  cases in 3537 s: 5 pass (K1 replay 0.999944/|ΔPmax| 1.9e-4, K1 perturbation replay,
  K-class cold start, strict K-class cold start, strict oversampled K-class cold start
  worst class 0.99997 in 190 s), K-class replay keeps its known failure (|ΔPmax| 1.0,
  class correlation 0.991), and the K1 cold start fails the same gates as on
  `dd7d9b218` (half correlations 0.994696/0.994417 vs 0.999; Pmax 0.8866 vs 0.9647;
  2294 s). Not admitted; unchanged from the numerical hand-off.
  Earlier tier runs 13704244 (`400ad81e4`, 6 of 7 failed) remain preserved.
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
  node. The RECOVAR run (13712371/13712372/13712373, frozen `400ad81e4`,
  same H100 node as the oracle) completed 15 iterations in 28037 s and is
  **not accepted**: mean class GT FSC-AUC 0.265754 vs RELION 0.266272
  (delta -0.000519, gate tolerance 0.0001; all four Hungarian-matched classes
  0.00015-0.00101 behind with identical FSC 0.5/0.143 shells), while
  particle-level parity is the best recorded (class agreement 0.990, poses
  within 1 deg 0.978, translations within 1 px 0.980, map corr vs RELION
  >= 0.9998) and wall is 6.20x RELION (4525 s; target <= 2x; sparse K-class
  pass-2 is 85 % of iteration wall). Full record in
  [em_parity_best_metrics.md](../math/em_parity_best_metrics.md#2026-09-11-k4-structural-cleanup-400ad81e4-100k256)
  and `em_completion_k4_400ad81e4_h100_20260910/{summary.md,admission_record.json}`.
  The four class populations are near balanced, so a population-weighted class
  mean does not change that verdict, and neither engine converged by iteration 15,
  so neither side has a final all-data pass. The summarizer (13712373) exits 2 for
  the failed quality gate; its K=1 fields report missing only because the launch was
  K4-only. The quality deficit goes to the numerical workstream and the runtime to
  the performance workstream, where the K4 compile-glue rounds integrated on
  September 11 (absent from this frozen source) and the stable-shape design item
  both apply.
  Attempt 13704399 (legacy dispatch log) is preserved as failed.
- **EM workstream A/B on the crop fixes** (13735554 base `a0b0a8141`, 13735555
  `7a3fdb665`; `pr179_crop_fixes_validation_20260911`): both jobs spent their
  60-minute limit in the per-job CUDA build and ended FAILED before any parity
  case ran; no A/B result exists yet (resubmitted as 13737347). The fast tier on
  frozen `11bc4f0c2` (13737449) is recorded above.
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

The dense/local fast guard rejects undefined names before JAX startup. The
`picked_frequencies` use-before-assignment that a broader scan found in
`recovar/em/heterogeneity.py` lived in the uncalled legacy
`estimate_principal_components`; after the
[caller audit](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/legacy_pca_callers_20260910/result.json)
and a fresh token scan found no caller, that function was removed on
September 11 ([receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/legacy_pca_removal_20260911/result.json)).
The main PCA pipeline uses a distinct implementation.
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

## VDAM insight port to the EM path

Disposition of the twelve insights in the September 11 VDAM handoff. "Lands with
the commits" means the shared code came in with the cherry-picks and needs no
separate EM change.

| # | insight | disposition |
| --- | --- | --- |
| a | staging copies every stack to a GPFS `TMPDIR`, network to network, and leaves it behind | **Ported.** `data_io.staging` now declines the `TMPDIR` fallback on a network filesystem and logs why; an explicit `RECOVAR_CACHE_DIR` is still honored anywhere. Verified here: the EM work runtime root reports gpfs and is refused, `/tmp` (xfs) and `/dev/shm` (tmpfs) still stage. The completion harness already disabled staging for this reason. |
| b | `RECOVAR_PREREAD_IMAGES` removes per-iteration subset re-reads | **Ported.** The completion harness sets it by default with the loader's 64 GB per-file cap; the 100k/256 stacks are about 26 GB. The EM K1 100k pair will be measured with it. |
| c | the local engine fetches each subset a second time after pass 1 | **Recorded, not changed.** The handoff rates it about 2 s per iteration once the preread is on; it stays on the cleanup candidate list rather than being folded into this batch. |
| d | per-bucket eager glue costs ~137 XLA programs per bucket shape | **Lands with the commits** (both glue rounds). EM K4 cold and warm walls are being re-measured. |
| e | remaining K4 compile cost: current-size changes and per-iteration bucket shapes recompile every pixel-dimensioned program | **Design item, not started.** Extending `--stable-fourier-window-shapes` to K>1 and stabilizing shapes in every K-class bucket planner is a separate change with its own qualification; recorded here as the next performance step for K4. |
| f | determinism opt-ins give bitwise same-state K1 maps | **Lands with the commits.** Use `RECOVAR_EM_DETERMINISTIC_REDUCTIONS=1` with `RECOVAR_RELION_WAVG_DETERMINISTIC_ROTATION_SUM=1` for EM same-state bitwise checks. |
| g | K4 base runs are not bitwise-repeatable even with the opt-ins | **Recorded.** Any K4 equivalence claim in this workstream is band-level, never bitwise, until the x-half BPref atomics are covered. |
| h | the fixture generator writes a CTF block RELION rejects for no-CTF cells | **Already handled here, no change.** The K=1 robustness matrix writes a sanitized identity-CTF STAR through `scripts/make_relion_identity_ctf_star.py` (gated by `EM_K1_NOCTF_RELION_USE_CTF`, default on) and the K-class matrix drops `--ctf` for those cases. Both predate the handoff's source, so the VDAM runner can reuse either instead of excluding the cells. |
| i | the K-class GT scorer reports a plain mean over classes | **Ported.** `evaluate_kclass_gt.py` takes `--class_population` per class and reports population-weighted means beside the plain ones. |
| j | `test_relion_cuda_powerclass_norm_units_preserve_divide_before_square` fails on CPU-only environments | **Did not reproduce; not marked GPU-only.** It passes on this branch under `JAX_PLATFORMS=cpu` with `CUDA_VISIBLE_DEVICES` empty, with and without the prebuilt native binding. Handed back rather than weakening the test. |
| k | a fresh worktree's source mtimes are newer than a copied CUDA library, so the loader rebuilds it in place | **Adopted as practice.** The validation runners `touch` the pinned library before each arm and record its sha256. A loader guard that refuses to overwrite a pinned `RECOVAR_CUDA_LIB` remains an open suggestion. |
| l | late-phase per-iteration times were bimodal purely from I/O | **Ported.** Completion jobs now echo staging, preread and compilation-cache state with the other provenance, so a wall is read together with its I/O placement. |

## VDAM end-to-end status (carried from the VDAM workstream)

Integrated on September 11 from the VDAM handoff
([document](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/pr179_coordination/handoffs/vdam_to_em_clean_integration_and_em_port_20260911.md)):
six determinism opt-ins, two K4 fused pass-2 compile-glue rounds and the host-memory
particle preread, cherry-picked in the order the handoff gives and reconciled with the
cleanup owners ([receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_integration_20260911/result.json)).
The four ladder commits the handoff marks as reverted were skipped. The table below is the
VDAM workstream's own end-to-end evidence, reproduced verbatim and refreshed on every
publish; its rows are VDAM results, not EM-path results, and the EM path's own rows stay
in the sections above.

Gate = "inside the RELION run-to-run band" (RELION ×2 / recovar ×2, same seed, same GPU class) unless a fixed rule is stated; both engines are chaotic after
iteration ~30–70 so fixed map tolerances are not meaningful. All rows at source 5ca9c8fff unless noted; K4 GT-AUC is population-weighted (plain means are
dominated by empty classes).

| test (end to end) | metric | gate | current value | status | receipt / jobs |
|---|---|---|---|---|---|
| K1 synthetic, 20 library cells (5k/128, one 256²; 2 no-CTF cells excluded: RELION rejects the fixture CTF) | it000 map exact; GT FSC-AUC it100/it200 vs RELION; cross-AUC | exact; inside band (seed sweep if single pair ambiguous) | 20/20 exact; 17 within ±0.003, 3 inside band, case 22 by seed sweep | pass | `em_work/codex/vdam_synthetic_k1_matrix_5ca9c8fff_20260910/RESULTS.md` |
| K1 real 10k/256 (EMPIAR-10076 subset), natural 200 | cross-AUC vs RELION, 4-arm band | inside band | in band | pass | `vdam_k1_10k_cachewarm_5ca9c8fff_20260910`, integrated_full200 roots |
| K1 real 100k/256, natural 200 | cross-AUC it200 vs RELION ×2; min over checkpoints | inside band (RELION-vs-RELION 0.9945 / 0.9222 min) | 0.9931–0.9953 / 0.9217 min | pass | `vdam_real10076_100k_repeat_5ca9c8fff_20260910`, `vdam_k1_100k_preread_20260910` (jobs 13683192, 13688031/2, 13712451) |
| K1 fixed-state replays (real 10k, t=20…58; 100k t30/t31) | Pmax gap, significant counts, pose flips | ≤1.6e-4, 0, 0 | ≤1.6e-4, 0, 0 | pass | `vdam_real100k_onestep_replay_…`, case22_replays |
| exactly-K4 5k/128, existing fixture, natural 200 | Hungarian matched class AUC, assignment agreement, populations, weighted GT-AUC vs 4-arm band | inside band | 70 % class 0.63/0.70 vs RELION (band 0.35–0.94); populations 0.707/0.293 (band 0.70/0.30) | pass | `vdam_k4_synthetic_full200_repeat_5ca9c8fff_20260910`, `vdam_k4_full200_glue_cf8778730_20260910` |
| exactly-K4, 5 new fixtures (noise 3, radial noise, Kent, head-heavy, Kent+offsets), 2 pairs each | same | inside band; seed sweep if ambiguous | recovar-vs-RELION distances = RELION-vs-RELION in 5/5; weighted GT-AUC in band 4/5; radial fixture: behind at seed 29, equal/ahead at seeds 30–32 | pass | `vdam_k4_fixture_matrix_5ca9c8fff_20260910/RESULTS.md` (jobs 13710571–8, 13711152/3, 13717088–97, 13723830–7) |
| **perf** K1 real 100k/256 wall | recovar / RELION, same H100 class, shared nodes | ≤2× provisional; goal ≈1× | **3874 s / 3524–3589 s = 1.08–1.10×** with preread (was 2.15–2.54×) | pass | `vdam_k1_100k_preread_20260910/RESULTS.md` (8f348b05a) |
| **perf** K1 real 10k/256 wall | same | ≤2× | 676 s warm cache / 608–662 s = 1.1× (850 s cold) | pass | `vdam_k1_10k_cachewarm_5ca9c8fff_20260910` |
| **perf** exactly-K4 5k/128 nr_iter 20 wall | same | ≤2× | cold 800 s / 39 s (5ca9c8fff); warm cache 205 s (5.2×); glue rounds: 20 cold iterations 545 s vs 800 s | **fail** (compile-bound; design item below) | `vdam_k4_synthetic_cachewarm_…`, `vdam_k4glue_20260910/RESULTS.md` |
| **perf** exactly-K4 5k/128 natural 200 wall | same | ≤2× | 6108 s (glue) / 932–1086 s = 5.6–6.6× (was 6900–9725 s) | **fail** | `vdam_k4_full200_glue_cf8778730_20260910/RESULTS.md` |
| determinism (K1 same-state, opt-ins) | bitwise map repeat | bitwise | bitwise ×2 (122 s; 77 s with fusion autotuner off) | pass | `vdam_detred_samestate_t3_4e5407be_20260910` |
| determinism (K4 same-state) | bitwise map repeat | bitwise | not bitwise (x-half BPref atomics not covered), Δ 1.5e-8 | open | `vdam_k4glue_20260910/RESULTS.md` (job 13713415) |

**Validation of the handoff's two GPU checks on the published source** (frozen
`97f6d6b33`, source-identical to `11bc4f0c2`; root
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_port_gpu_validation_97f6d6b33_20260911`,
H100 della-h19g1, no other job of this workstream on the node, XLA persistent caches
off as in the VDAM harness). Exactly-K4 5k/128 nr_iter 20 (job 13740118): cold 533 s
wall, warm cache 143 s, 5964 cached programs, cold-vs-warm iteration-20 class maps
within 1e-6 (float32 band) — better than the table's 800 s / 205 s at `5ca9c8fff`
and level with the glue rounds' 545 s, still 13.7×/3.7× RELION's 39 s (fail, the
compile-bound design item stands). K1 real 10k/256 natural 200 with
`RECOVAR_PREREAD_IMAGES=1` (job 13740476): fill 1302 s, warm cache 967 s, 201 maps
each, fill-vs-warm map correlation 0.9963 at it100 / 0.9985 at it200 (chaotic band) —
**worse** than the table's 850 s / 676 s and 2.0×/1.5× RELION's 608–662 s, with
5142 compiled programs against 2401 at `5ca9c8fff`. The doubled program count is
the lead: an attribution pair with the per-iteration profile runs the same fill arm
on `97f6d6b33` (13742057) and on `4d569d27a` (13742058, VDAM integrated, before the
EM projector-crop port) to separate the crop port from the VDAM integration.
Earlier attempts of this pair (13735912…13739747) failed on environment only:
missing RELION binding / FFTW CMake paths, the CUDA 12.8 toolkit's `nvlink` against
pixi's 12.9 `ptxas`, and `JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=all` breaking warm
arms at kernel launch; all archived under `attempt*` in the root.

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

The fast-tier rerun on frozen `dd7d9b218` (job 13726940) and the exactly-K4
completion (13712372: GT FSC-AUC gate failed by -0.000519, 6.20x RELION wall)
and the fast tier on frozen `11bc4f0c2` (13737449: 5 pass, 2 known failures)
and the VDAM-handoff GPU validation pair on the published source (K4 20-iteration
cold 533 s / warm 143 s; K1 10k/256 preread 1302 s / 967 s with 5142 programs vs
2401 at `5ca9c8fff` — attribution pair 13742057/13742058 running) are recorded
above. Between results,
continue one bounded structural package at a time from the cleanup plan.
Remaining candidates after the September 10–11 packages (J through UU; the big-JIT
core layout and the sparse pass-2 tuple reader landed as TT and UU): the 35-line bucket-pipeline blocks repeated inside
`sparse_pass2_bucketed` and the 33-line blocks inside `local_big_jit` (hot paths,
exact comparison would need GPU-shaped fixtures; the dump-writer operand fields,
the `_score_rotation_block` keyword sets and the K-class per-class lists landed as
QQ, PP and VV); and the K=1/K-class route asymmetries recorded in the
adaptive-engine-call receipt (coarse translation phases, significance skipping and
the diagnostic float64 pass 2 are K=1-only). Two findings from the duplicate scan are deliberately not packages. The long
positional parameter lists shared by the big-JIT kernels in `local_big_jit` are a
JIT boundary, not duplicated logic; collapsing them into containers would change
the static/dynamic argument structure of the hottest kernels and is out of scope
for structural cleanup. The two bucketed pass-2 entry points resolve their fine
translation grid differently and that difference is numerical, not structural:
`compute_pass2_stats_sparse_bucketed` builds the oversampled grid from the
host-precision coarse array and validates supplied overrides, while
`compute_k_class_pass2_stats_sparse_fused` builds it from the score-dtype cast and
performs no override validation. Sharing an owner would either change K-class
numbers or add a validation flag, so the divergence is recorded here for the
numerical workstream rather than merged. The global-winner summary writer and
its analysis validator repeat the semantics contract on purpose (independent
check) and stay; the seven forwarding aliases found by the wrapper scan are
intentional public names or test patch points and stay. Each package keeps
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
