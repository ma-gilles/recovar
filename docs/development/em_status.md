# Current EM development scope

The [projector dtype history audit](em_projector_dtype_history_20260913.md)
identifies7e9e5c3c9 as the producer cast removal: actual before/after function
bodies reproduce C64→C128 and lost texture eligibility. This is source-level
attribution, not a historical whole-run timing claim. Matched parent/repaired
three-iteration K1 job13829869 and K4 job13832403 are terminal. K1 regression
does not exercise the affected PPref path; K4 early state and all four
cross-native AUCs improve, but class1 and iteration3 still miss gates.
[Corrected canonical report, coverage limits and short-profile outcome](em_projector_quality_profile_20260913.md).
Full quality and general speed remain open; next separate real-input preparation
from the bounded warm profile rather than repeat expensive controller startup.

## Private projector consumer repair — September 13

Private candidate on shared parent `37faa4998` restores the explicit
float32 projector upload boundary, without the separate half-staging speed
patch. [Boundary policy and tests](em_projector_consumer_precision.md) preserve
native host preparation and independent double diagnostics. Three focused
pre-wiring failures reproduced the missing dispatch policy; 81 focused cases
and the unchanged 90-case CPU fast guard pass after wiring. These are CPU
dtype/dispatch checks, not GPU score/support, FSC, trajectory or speed admission.
The EM-clean lead remains sole shared integrator; current-source matched GPU
qualification remains open. A matched real10073 one-particle replay now records
warm calls34.484s inherited-C128 versus3.011s repaired-C64 (11.45× in this
scope), with the difference concentrated in the sparse-pass wrapper. Significant
count changes9184→9194; Pmax changes1.72e-5. No trajectory/FSC or general speed
admission. [Frozen replay, numerical differences and limitations](em_projector_real_replay_20260913.md).
No existing jobs or sealed libraries were changed.

Current decisions belong here; update this page when a decision changes, not for
every test or publication. Detailed receipts belong behind links. The
[previous page is preserved byte-for-byte](em_cleanup_history_20260910_a2ab056cb.md)
at `a2ab056cb`, including one paragraph per earlier checkpoint; its historical
next actions are superseded here.

## Geometry validation and compact dtype repair — September 13

The [zero-oversampling geometry correction](../math/zero_coarse_geometry.md)
at `c5559c7aa` passed matched H100 job **13816465**: one three-iteration K1
regression, zero skips, 25m36s. Source, inputs, harness and loaded-binary checks
pass. Against controller 13814090, first-iteration Pmax RMSE improves from
0.0000146878 to 0.0000090670 and significant-count mismatches fall from 1454
to 1121. Later count mismatches are 658/9 versus 654/9; strict state remains open.
Signed merged cross-engine FSC-AUC is 0.999619283464, a change of +3.4481e-8.
[Canonical reports and matched command comparison](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_zero_geometry_integration_20260913/result.json).

Both compact CUDA guards 13815504/a38 and13816668/bdee failed: compact execution
sent float64 probabilities to the F32-only native dual sums, after the rectangular
reference completed. Separate repair `58aa58131` checks actual per-class operand
dtypes and retains the existing precision-preserving reduction for unsupported
types. Profiles distinguish requested from executed native dispatch. CPU checks
pass; unchanged tiny K2 guard **13817137** completed0:0 on frozen58aa in19s,
with one test and zero skips. Independent source, harness, manifest, loaded-library
and JUnit checks pass. This closes that dispatch regression only, not current-head,
exactly-K4, full-trajectory or speed qualification.
[Terminal GPU review](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/compact_native_dtype_repair_20260913/gpu_terminal_review.json).

VDAM row selection, native binary replay readers and artifact profiling now
share their existing owners. Two diagnostic analyzers also share one FFT row
mapping, with its explicit expected-array reference preserved. Duplicate tests
were removed only where inputs, implementation and assertions matched.
[Replay-reader checks](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_native_replay_reader_20260913/result.json),
[artifact checks](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_artifact_profile_cleanup_20260913/result.json),
[FFT diagnostic checks](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_fft_replay_layout_cleanup_20260913/result.json).

All ten same-source VDAM controls and twelve older/newer-base bisections are
terminal. Seven of ten controls miss an original mean or weighted-FSC band.
The older base has four of six arms inside both bands; the newer base has three.
Their only source difference changes generic shared-spectrum noise operands from
float64-promoting division to a reciprocal cast to the accumulation dtype before
multiplication. Effective paired environments differ only in runtime paths;
declared CUDA libraries hash identically. Current lead source retains generic
division. These observations do not qualify either precision or the optional speed
changes: same-source replicate variability and original acceptance bands remain.
The existing radial-knob matrix continues; private speed integration is held.
[Completed base and source review](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_capacity_noise_review_20260913/base_source_environment_review.json),
[all terminal base metrics](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_capacity_noise_review_20260913/base_all_terminal.json),
[terminal control review](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_capacity_noise_review_20260913/control_matrix_terminal.json).

Earlier controller, startup-noise, projection-radius and prior corrections are
recorded with their frozen-source evidence in the [published checkpoint history](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/development/em_status.md).
Those results do not qualify the current source.

## Milestone and invariants

Complete RECOVAR cleanup **EM first, GUI excluded**, before new-engine work:
remove proven dead/duplicate code, clarify owners/APIs and establish reproducible
synthetic, real and exactly-K4 accuracy/performance benchmarks. The goal remains
incomplete. See [cleanup plan](cleanup_plan.md), [codebase map](codebase.md),
[benchmark contract](benchmarks.md) and [EM operating rules](../../recovar/em/AGENTS.md).

Structural changes preserve defaults, casts/reductions/JIT order, buffer lifetime,
non-EM APIs/formats and independent references. EM/VDAM APIs, CLI interfaces and
historical Python object names need not remain compatible. Prefer less code and
clear ownership over wrappers or consolidation that introduces complex branching.
Keep maintained validation tools; preserve obsolete experiments by their last
usable commit and external evidence, then remove their tools and redundant tests.
Canonical source Euler
angles and host pixel geometry stay metadata; derive computation arrays from them.
Double is diagnostic, not a production remedy or proof of noise. No tolerance or
baseline changes. User priority is short-prefix parity then final FSC, with up to
2× native runtime provisionally; this does not waive accuracy or completion gates.

The CUDA implementation is also in the cleanup scope; it does not replace the
EM/VDAM implementation and test cleanup. Trace kernel, FFI, test and private
replay consumers before retirement. Preserve the shared heterogeneity backend
and diagnostic precision variants. The [CUDA audit inventory](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/cuda_cleanup_audit_20260912/)
and [first dead-kernel receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/cuda_unused_batch_project_20260912/result.json)
record the current source-specific evidence.

## Source and ownership

- Primary: `/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_structural_cleanup_20260907`,
  branch `codex/integrate-pr180`; current published identity is in the
  [lead status](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/pr179_coordination/status/em_clean.json).
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

## Engineering history and current architecture

`recovar/em` is the implementation root. EM and VDAM share scoring, local search
and accumulation owners; optional captures and replay readers live in diagnostics,
with reconstruction boundary calls remaining in their production owners. Native
M-step replay is separated from VDAM reconstruction and still counted by the size
gate. The explicit `pass2_engine` option owns routing; legacy environment and
per-half-summary Pmax controller fallbacks are retired.
VDAM startup uses the loaded image backend directly and requires the native
particle shuffle; obsolete mask-backend and Python-shuffle fallbacks are retired.
The independent shuffle reference remains in `tests/helpers/vdam.py`.
The E-step requires the sampling state already created by startup or continuation;
its unused no-state branch is retired. Gradient mode directly controls pseudo-halfsets.
The private configuration builder requires iteration-owned priors, sigma offset,
pass-1 order and dataset geometry. Reference and momentum arrays are required
state inputs; initialization and checkpoint loading supply them explicitly.
[Shuffle evidence](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_shuffle_contract_cleanup_20260913/result.json)
and [backend evidence](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_image_backend_contract_cleanup_20260913/result.json)
record unchanged numerical calls, surviving coverage and the failing size gate.

The [published engineering history](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/development/em_status.md#engineering-work-and-recent-evidence)
preserves the takeover's original failures, their repairs, deleted-file evidence,
CUDA build comparisons and all earlier checkpoint measurements. Current work and
live job ownership are in the coordination board above; historical next actions
are superseded by that board.

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
`recovar/em/reference/heterogeneity.py` lived in the uncalled legacy
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
`RECOVAR_PREREAD_IMAGES=1`: a first pair (job 13740476) measured fill 1302 s / warm
967 s with 5142 compiled programs against the table's 850 s / 676 s and 2401, and
an attribution set (13742057/13742058/13745309) showed the same slow profile on
`97f6d6b33`, on `4d569d27a` (before the EM projector-crop port) and on the VDAM
commit `5ca9c8fff` itself — the driver copied from the parallel em_clean session
carried the VDAM candidate CLI but none of the gate contract's candidate
environment (`native_noise_full200_8ab1a44be_13628018/gf43/candidate_1/environment.json`:
62 `RECOVAR_*`/`JAX_*` K1 opt-ins such as the coarse GEMM hybrid and projection
cache, exact-local BPref packing/capacity/transaction, host-plan CUDA, noise pixel
capacity/CUDA, packed local projection and stable flat rows), so every arm ran the
engine at defaults. With that environment applied (`vdam_candidate_env.json`, job
13746584) the published source measures fill 827 s / warm cache 627 s, 2427 programs,
expectation 576 s over 200 iterations — level with the table's 850 s / 676 s, 2401
programs, 582 s at `5ca9c8fff`. Verdict: parity with the carried table; the opt-ins
are environment-only forks the VDAM workstream owns. Earlier attempts of the pair
(13735912…13739747) failed on environment only: missing RELION binding / FFTW CMake
paths, the CUDA 12.8 toolkit's `nvlink` against pixi's 12.9 `ptxas`, and
`JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=all` breaking warm arms at kernel launch;
all archived under `attempt*` in the root.

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

Compact native weighted sums now consume the source-ordered pair list directly
(`2e6601b28`), preserving float32 probability rejection and both complex value
precisions. The existing dense wrapper and its JIT boundary remain unchanged.
50 CPU tests and 45 builder-contract cases passed. H100 job **13827004** passed
all nine GPU cases without skips: raw sums match dense native bitwise, while
metadata reductions retain the previously reviewed ULP bound. Source, harness
and the loaded library were verified
([GPU receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/pair_sparse_sums_integration_20260913/gpu_terminal_review.json)).
Full trajectories and matched speed remain open.

Recent structural cleanup consolidated the compact probability scatter, removed
235 unused test lines while retaining all 175 test functions, moved direct Wavg
norm serialization into its existing diagnostics owner, and consolidated
translation-spacing inference in sampling. The per-image reference remains
independent, and caller precision casts are unchanged. The last change passed
90 EM fast-guard cases, 32 layout cases (three required a separate pinned-binding
rerun), and 40 exact spacing comparisons
([spacing receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/translation_spacing_owner_20260913/result.json)).
The diagnostic move preserved all 36 compared payloads and passed 33 focused
checks ([diagnostic receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/sparse_norm_diagnostic_owner_20260913/result.json)).
Continue controller cleanup and individually reviewed VDAM integrations under
the full coordination queue; these checks do not close scientific acceptance.

The user retained the fused compact-pair scoring route. It is integrated at
`6458ee41d` into the sparse scoring owner, without a new experimental switch.
Supported float32 GPU compact pairs use the fused kernel; CPU and diagnostic
double routes retain their existing implementations. Masked pairs return positive
infinity and the CUDA kernel groups 16 independent pairs per block. The scorer
matches the gathered CUDA reference bitwise; its known few-ULP differences from
the JAX emulation remain a numerical-policy distinction, not strict parity.
191 CPU tests passed; the one CUDA-only skip was covered by frozen H100 job
**13826292**, which passed all 16 GPU cases without skips in 20 seconds. Tests
cover partial groups, masking, posterior ties, runtime compilation and the compact
versus rectangular x-half path. Source, harness and loaded library were verified
([GPU receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/compact_fused_scoring_integration_20260913/gpu_terminal_review.json)).
This qualifies the focused checks only; full trajectories and matched speed remain open.

The tiny dtype-repair CUDA guard is complete on its frozen source. Continue
strict K1 state closure and the existing peer attribution jobs before advancing
the scientific ladder in the coordination queue. Do not restart completed checks
or duplicate live candidates. The user approved replacing the historical VDAM
size cap after the growth audit. The [responsibility budgets](codebase.md#vdam-code-budgets)
count 8,630 lines against 8,850 combined, including all shared extractions.
Option validation now lives on `NativeInitialModelOptions`: 32 controller lines
removed, four net lines added, 164 focused tests and 582 exact option comparisons
passed ([receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_option_validation_owner_20260913/result.json)).
This budget revision does not close the remaining cleanup or scientific gates.
The unreachable VDAM fine-prior fallback was removed with 223 focused tests
passing and 75 unchanged routing comparisons
([cleanup receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_dead_prior_fallback_20260913/result.json)).
The bounded v6 native capture passed live rank/library and closed-payload checks.
The subsequent six-call rectangular scorer check passed on those captured inputs,
but the hybrid support still differs: 52,884 versus 52,883 selected candidates.
The next private capture targets the full candidate population for one particle
using actual RECOVAR production inputs; its worker still requires qualification.
Neither bounded result establishes completed-refinement or strict-state acceptance
([primitive receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/native_rectangular_terminal_lead_review_20260913/result.json)).

Preserve the distinct numerical routes while reducing structural overhead. Big-JIT
argument lists define static/dynamic compilation boundaries. Bucketed and fused
K-class pass 2 construct fine translation grids with different precision and
validation; unifying them is numerical work. Keep independent summary validators
and references independent.

The four live traced SGD variants passed the one-row trace comparison in H100
job13814262 on frozen2b27201e2: traced/untraced outputs were bitwise equal and
source/harness/library checks passed. Full replay and trajectory qualification
remain open. [Trace evidence](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/cuda_live_trace_variants_20260913/result.json).

Use [the workflow](agent_workflow.md), existing `scripts/em_work_package.py` receipts,
focused checks and cohesive publication batches. Preserve the existing agent
assignments and session model choices. Build and identify native libraries
explicitly before GPU qualification; frozen libraries remain immutable.

## Agent efficiency package — September 9

Direct Astra medium remains the default for this workstream. Built-in delegation
is disabled; isolated Terra smokes passed but the reviewed small-task pilot used
1.625× input and 1.667× elapsed time versus direct Astra. No token savings proved.
Compact handoffs, scripted receipts and batched publication continue. No automatic
wakeup/model-polling promise. [Setup, measured limits and recovery](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/pr179_coordination/delegated/README.md).

Independent CPU replay of that capture now reproduces all 512 lane slots and
6,496 running values using the existing RECOVAR direct-square arithmetic.
Lane accumulation explains the three traced support differences, but correcting
those scores exposes another cutoff difference. Full support remains open;
qualification of the existing rectangular scorer is being prepared. See the
[attribution review](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/native_v6_attribution_lead_review_20260913/result.json).

The accumulated parity notebook is now a [history index](../math/relion_parity_agent_notes.md)
with 216 links to the exact published findings. This removes 13,579 historical
lines from the working tree without discarding the scientific record.
