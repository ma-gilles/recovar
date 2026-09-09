# VDAM precision investigation — September 9, 2026

This dated evidence record preserves the reviewed private `907b02ce` /
`fe8472947` investigation. It does not qualify newer source or authorize
precision adoption. The [current EM status](em_status.md) owns the next check
and current integration decisions. Relative scratch paths below use
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/`; source-review paths use
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/hia_source_review_20260906/`.

VDAM's separate two-path full-float32-product candidate `907b02ce` is now frozen
and reviewed: exactly two `HIGHEST` contraction keywords differ in production;
all 32 recorded H100 cases pass (13638400), after two expected-red cases
(13638324). Native-cutoff candidate replay 13638431 is diagnostic only. See
`handoffs/em_clean_vdam_full_float32_review_20260909.json`. Prepared-state jobs
13638710/13638814 on that private source now preserve complete incoming/candidate
inputs and particle decisions/Pmax. Noise/BPref arrays vary even within a policy;
the first panel's one-ULP support-sum policy association did not reproduce in the
second. Only the rectangle-power helper executes in this replay; the atomic
fallback has focused-test coverage only. These are not full-state equivalence,
trajectory or runtime acceptance.

VDAM's private composition `fe8472947` carries the same `907b02ce` patch atop
compact-CTF merge `5a39eab29`; only two documentation files separate that parent
from the assigned `cbff0b092` base. The candidate applies cleanly to the current
primary, but is **not adopted**. H100 13639506 completed four natural 200-iteration
old/new/new/old trajectories (0:0, 1,819 s). Whole-process old times are
449.269/454.861 s and new times 456.089/457.051 s: **1.009965×** by mean,
a small-fixture diagnostic only. The completed 804-file/201-checkpoint metadata
ledger is schema-valid but fails numeric comparison at 199 iterations (2–200).
Old-policy repeats first differ in pose at 32 and resolution shell at 45;
new-policy repeats first differ in pose at 52 and keep the same resolution shell.
Their winning margins and convergence implications remain unresolved.

The completed producer FSC review reports old-repeat minimum cross-FSC-AUC
0.969668 at iteration 155, versus 0.998219 for new repeats at 86. The two paired
old/new minima are 0.969722 and 0.998219. These are cross-map diagnostics.
The registered-GT follow-up below now covers these checkpoints; historical
native source-to-binary closure remains missing.
The candidate cross-native minima span 0.998196–0.998767 at iteration 86,
versus a native-repeat minimum of 0.9997155. The second old-policy arm is closer
to both native endpoints than either candidate arm: no uniform candidate win.
The producer reports exact independent agreement for all 15 terminal AUCs and
shell curves; em_clean's review here is the pinned report, not a fresh curve
recomputation. See `handoffs/vdam_full201_fsc_terminal_20260909.json` and
`vdam_f32_full200_20260909/fsc_trajectory_figure/fsc_trajectories.png` in scratch.
em_clean verified the review's seven artifact hashes, without independently
recomputing its arrays. No full-state, trajectory or runtime acceptance follows.
See `vdam_f32_full200_20260909/analysis_integrated_v2_review/result_summary.json`
under the scratch artifact root; review receipt is
`dense_firstiter_arguments_20260909/full200_peer_report_review.json` under the
source-review root below.

Prepared E/M job 13639984 failed in diagnostic callable serialization before
science E/M completion; the failed evidence remains. VDAM's artifact-only v2
repair completed as H100 13640121 (0:0, 145 s) on frozen fe847. Report review
verified five artifact hashes and four completion receipts, without independently
recomputing the numeric array comparisons. It reports exact incoming/compact-CTF
operands and local decisions, but coarse scores vary up to 1.220703125e-4,
including same-policy repeats. Accumulators and six post-state arrays vary;
crossed maxima are not uniformly bounded by the two same-policy comparisons.
Single-boundary cross-map FSC near one does not establish trajectory quality.

Follow-up raw FFI job **13640613** completed on frozen fe847. Independent CPU
recomputation of all 16 saved `(200,576,29)` float32 outputs verifies variation
in all 15 comparisons with repeat 0: maximum 1.220703125e-4, p95 3.0517578125e-5.
All 200 pre-prior raw winners remain exact across 16 runs; the minimum represented
score margin is 0.0013885498046875. Thirteen manifest pins, 17 native source
inputs in both checkouts, the library hash and terminal receipt were checked.
The frozen harness compiles the raw scorer once and uses distinct retained
output buffers; it reports exact negation/max controls on one fixed output.

The captured route is the shared-pretranslated direct float32 FFI, whose CUDA
source merges lane sums with atomics. Recompilation and the later Wavg product
intervention are not necessary for this observed variation. Actual atomic order
was not recorded, and these are raw pre-prior margins, not posterior margins or
evidence for later trajectory flips. No arithmetic change or quality acceptance
follows. The [durable audit archive](evidence/vdam-coarse-repeat-20260909/README.md) preserves
metrics, hashes, the exact CPU audit script and its reproduction command. Original
outputs are under `vdam_coarse_atomic_repeat_20260909` in scratch.

Paired-history H100 job **13641091** completed two fresh old-policy prefixes on
frozen fe847, through M31 and ordinary E32 plus three prepared repeats, stopping
before M32. Both choose rotation111738 for particle1367; signed competing margins
(best163798 minus best111738) are −0.0009765625 and −0.000244140625. Full target
candidate geometry/order agrees (3,360 cells, 256 finite). Target debug payloads
are exact within each history, and all 200 published decisions/Pmax agree across
ordinary/clean/debug calls. Whole E outputs still vary in 17 accumulator/noise
leaves. Incoming model, momentum, noise and priors differ before target preparation;
the cross-history score gap0.00115966796875 is not a same-input arithmetic bound.
Neither fresh history reproduces the original F200 pose flip, whose saved inputs
are unavailable. Reconstructed raw scores are not an independent arithmetic trace.

The bounded producer RNG audit found no scientific consumer of the differing
Python/legacy NumPy globals under these options; target preparation/E32 preserves
them within each history. Native RNG state and independent generators are outside
those snapshots. No global-seeding patch or automatic tie classification follows.
em_clean verified 11 report/manifest hashes, two terminal child receipts and
summary margin arithmetic; it did not independently recompute snapshot arrays or
repeat the full RNG reachability audit. See board handoff
`vdam_paired_history_terminal_20260909.json` and review/reproduction script under
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/hia_source_review_20260906/paired_history_review_20260909/`.
That prefix audit is now complete: all 64 metadata files for checkpoints 0–31
have matching schemas. Initial stored map arrays are exact; small accumulator,
noise and offset-sum differences appear at iteration 1, followed by support mass
at 2. Significant counts first differ at 18 for particle 1290 (36 versus 35);
no saved pose change occurs through 31. Full incoming E1 operands were not saved,
so initial-map equality does not establish identical E1 inputs or identify the
first divergent operation. The support-count tie remains unclassified.

The producer's momentum audit reports matching update formulas and no near-zero
square-root denominator (minimum 0.98634); the ten largest Fourier-cell differences
account for 98.68% of squared second-moment difference. Neither observation proves
the cause or its effect on poses. em_clean checked both pinned reports and the
prefix CPU completion receipt, without recomputing arrays or independently auditing
the full momentum formula. See `handoffs/vdam_prefix_first_divergence_20260909.json`
and `pass2_raw_schema_20260909/prefix_review.json` under the source-review root.
The registered-GT v2 diagnostic is complete: four planted controls plus native/null
optimizer checks pass. Fitting on native_1 final only, translation raises its
GT-AUC from 0.126986 to 0.388388 at fixed rotation. The same frozen transform was
then applied to all six histories and 201 checkpoints: 1,206 existing maps, no
refits or new scientific/GPU runs. The producer reports 443.88 CPU seconds, stable
source/input pins and all six terminal curves exact against the final-map pilot.
Registration controls validate this artifact, not a globally optimal or unique
frame; a final-map transform can miss earlier frame motion.

em_clean independently recomputed all 2,412 all-shell/held-out AUC values from the
saved curves (maximum discrepancy 1.12e-16) and verified all eight RECOVAR/native
trajectory summaries plus seven artifact hashes. This does not repeat fitting,
FFT computation or full source/native/MRC provenance closure. The unchanged
all-shell GT deficit condition is −0.002: old1 first fails at 155, reaching
−0.004248842 versus native1; it fails at 42 checkpoints versus native1 and 30
versus native2. New1, new2 and old2 remain within that condition at all 201
checkpoints against both native repeats. Their minima versus native1 are
−0.000508038, −0.001345653 and −0.000507449 respectively. Terminal deficits are
−0.000294190, −0.001214108 and −0.000312159; old1 ends at −0.002285102.
There is no uniform policy win across these two repeats.

Held-out shells 9–62 are descriptive: the immutable analyzer incorrectly attached
the all-shell threshold to them and native-repeat summaries. The preserved raw
report is superseded by `report_scoped.json`, which removes only those labels.
No scientific threshold changed. Review/reproduction script and receipt:
`pr179_current_api_20260909/gt_full201_review.{py,json}` under the source-review
root; board handoffs `vdam_registered_GT_terminal_20260909.json` and
`vdam_registered_GT_full201_20260909.json`. Strict accumulator/support/pose,
cross-engine FSC, convergence, robust K1/K4 and current-source runtime gates
remain open. Precision907 stays private; no production/native source change is
assigned.

The reviewed runs use **float32 scoring with an existing double-precision
M-step**. Effective precision is stage-specific:

| Stage in the reviewed composition | Effective execution | Evidence/scope |
| --- | --- | --- |
| Captured coarse scoring/projector operands | float32/complex64 | Captured-path evidence, not a claim about every intermediate or route |
| Candidate Wavg products | Two `HIGHEST` contraction keywords on float32 products | Same output dtypes; no M-step precision change |
| JAX VDAM M-step | float64/complex128 numerical computation in both arms | Explicit device casts and host call in [relion_vdam_mstep.py](../../recovar/em/dense_single_volume/helpers/relion_vdam_mstep.py); actual M executed in 13640121 |

The M helper is byte-identical in shared source and frozen fe847 (SHA-256
`0ba75b68202380372e0ffbf777e23bf5b1a82145437723a519429b7507524f2c`).
This is more than metadata precision and is not complete float32 M qualification.
No existing arithmetic is changed or newly accepted by this reporting correction;
the intended production-float32 goal remains. See board handoff
`vdam_effective_precision_boundary_20260909.json` and review receipt
`effective_precision_reporting_20260909/review.json` under the source-review root.
Warm prepared E is 268.953 → 284.035 ms,
**1.056075× (+5.61%)** in this small preloaded-data panel; no full-runtime
acceptance. Only rectangle power executes; full local score surfaces are absent.
VDAM's completed registered-GT diagnostic is reviewed above; both Wavg
helpers remain fixed. Preserve fe847 and its original evidence; no shared
precision adoption, new arithmetic change or duplicate GPU job. See
`handoffs/em_clean_prepared_em_composition_review_20260909.json` and the producer
`vdam_wavg_composition_v2_20260909/{RESULTS.md,result_summary.json}` under the
scratch artifact root. These runs precede newer structural cleanup and do not
qualify the current primary.

The separate907 correctness review now explicitly incorporates the completed
registered-GT trajectory measurement through board addendum
`handoffs/em_clean_precision907_registered_gt_review_20260909.json`.
This closes the missing registered-GT measurement for the reviewed fixture:
both candidates and old2 satisfy the unchanged all-shell delta condition at every
checkpoint against both native repeats; old1 first fails at155 and reaches
−0.004248842. It does not close strict cross-FSC, state/support/pose, convergence,
source/native provenance, broader K1/K4 quality or representative runtime gates.
No uniform policy win or adoption follows. The independent saved-curve audit
and its exact scope remain recorded above; no new fit, FFT or GPU run was made.

A later oracle inventory closes recorded source/build identity for private full
non-MPI RELION executable `6c54d2ac…`, built in 13636510. It does not close the
historical `2d070d64…` binary used above. Integrator review verified 54 material
hashes across the inventory and reporting proposal, including the source archive
file, build manifests/receipts and executable; it did not independently extract
all 1507 archive members or rehash all 13034 external dependency entries.
Allocation runtime dependencies and actual scientific outcomes remain separate.
See `handoffs/em_clean_reporting_oracle_review_20260909.json` for the exact scope.
The new four-arm source-closed diagnostic 13642331 is terminal 0:0 on one H100;
its quality analysis is separate, and its fe847 candidate still has the existing
F64/C128 M-step. Earlier failed launcher 13642161 remains preserved. None of this
adopts precision907 or establishes current-source quality/runtime acceptance.

## Source-closed four-arm panel

Job **13642331** completed all four 200-iteration arms on the same physical H100,
using frozen `fe8472947` and the source-closed `6c54d2ac…` native build. The new
analysis covers 804 maps at 201 checkpoints with one prespecified transform fitted
to this panel's native 1 final map. These are different histories and a different
native build from the six-history analysis above.

Integrator CPU review independently reintegrates **2,814 full-shell FSC curves
and 1,608 held-out curves**, exactly matching the saved AUC arrays and all 201
streamed records. All four candidate/native comparisons, six raw-pair summaries,
three repeat summaries and failure lists agree exactly. The all-shell calculation
excludes DC and integrates over the canonical normalized shell axis; held-out
shells remain descriptive, without an added acceptance threshold.

| Registered GT comparison | Minimum AUC delta | Iteration | Final delta | Iterations below −0.002 |
| --- | ---: | ---: | ---: | ---: |
| Candidate 1 − native 1 | −0.000627249 | 91 | +0.001609063 | 0 |
| Candidate 1 − native 2 | −0.000032548 | 53 | +0.000817972 | 0 |
| Candidate 2 − native 1 | −0.001293697 | 155 | +0.002292598 | 0 |
| Candidate 2 − native 2 | **−0.002423452** | **155** | +0.001501506 | **1** |

All four cross-engine raw-FSC comparisons also have values below the existing
0.999 condition: first failures are at 78 against native 1 and80 against native 2.
Candidate 2/native minima reach 0.97076 and 0.97049. Better final registered-GT
values do not override the transient GT failure or cross-engine FSC failures.
Precision907 remains private and unaccepted.

The independent review checks nine named artifact hashes before/after, including
the saved arrays, summary, transform, streamed records and timing/completion
receipts. It compares the producer's 8,712-file before/after manifests as records;
it does **not** independently rehash those 8,712 inputs, reload the 804 MRCs, rerun
FFT/FSC generation, refit GT, or rerun optimizer controls. Those calculations and
source/dependency checks remain producer evidence. The producer's separate STAR
audit reports sampling-accuracy divergence at 20, candidate 2 resolution divergence
at 145 and current-size divergence at 146. Saved STARs permit partial state checks;
raw score margins, native selected IDs and candidate momentum checkpoints are
missing, so they cannot establish complete state or tie-aware parity.

| Whole-child timing | Seconds |
| --- | ---: |
| Native1 | 305.405011 |
| Candidate 1 | 448.153169 |
| Candidate 2 | 449.220601 |
| Native2 | 305.476140 |

The independently recomputed geometric RECOVAR/native ratio is **1.468982×**
(46.90% slower) on this 3k/128 K1 fixture. Times match the terminal panel receipt,
which records one physical H100 UUID. They include cold compilation/JIT and
output I/O; candidate processes additionally include their existing import,
pin and output-hashing checks. Filesystem/page caches were not flushed. This
is not representative 100k timing, peak-memory qualification or current-primary
performance evidence. Candidate scoring/projector operands are F32/C64; the
existing numerical M remains F64/C128.

Reproduce the read-only CPU audit from the primary checkout:

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONNOUSERSITE=1 \
  .pixi/envs/default/bin/python \
  /scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/hia_source_review_20260906/source_closed_saved_curve_review_20260909/review.py
```

Review scope/result: `source_closed_saved_curve_review_20260909/{scope,review}.json`
under the source-review root. Producer artifacts:
`vdam_source_closed_pair_analysis_20260909/results_v3/` under scratch;
summary SHA-256 `ca6034718e379b7d9e48988a07820d03cdc2fb6d4b5e6aa70d14e3b21bc83e0b`.
Board acknowledgement: `handoffs/em_clean_source_closed_saved_curve_review_20260909.json`.
No new GPU job, source change, baseline update or tolerance change was made.

## Saved-spectrum resolution boundary

The follow-up on frozen fe847 examines the active, non-final K1 resolution rule
around both candidate-2 departures. Both rules stop at the first shell with
SSNR below 1, retain the preceding shell and enforce minimum shell 5. Equality
at 1 passes. With the recorded increment 10 and no high-FSC expansion, next E-size
is twice the sum of the selected shell and 10. The first event and the event next to the
GT failure are distinct:

| Post-M iteration | Candidate 2 SSNR, shell 27 | Native 2 SSNR, shell 27 | Next E iteration | Candidate 2/native2 size |
| --- | ---: | ---: | ---: | --- |
| 145 | 0.998513566527 | 1.007064 | 146 | 72 / 74 |
| 154 | 0.998925307446 | 1.000503 | 155 | 72 / 74 |

The formula here is `2 * (shell + 10)`, using shell 26 or 27. Native values are
serialized with six decimals, candidate values with 12 significant digits; none
of the decisive values is rounded to 1. At147 and156 the saved sizes agree again.
Different spectra explain the branch choices with the same rule. This does not
identify the spectra's earlier cause, establish a general convergence rule match,
or prove that the smaller window caused the iteration-155 registered-GT loss.

Integrator review independently verifies **166 named artifact/source hashes**,
all **72 serialized SSNR scans** (four arms, 139–156), **68 following-size links**,
and the recorded three probes immediately below/at/above 1. It does not rerun the
native executable, E/M, or the producer's full source-manifest guard. Eight
counter-prediction differences at 140/150 across all four arms remain in the
receipt: the limited replay omits sampling resets. Both sources contain those
resets, but the complete sampling update was not replayed. No counter mismatch
was silently discarded or reclassified as roundoff.

Artifact: `vdam_source_closed_pair_analysis_20260909/resolution_boundary145/`.
Failed initial native/candidate STAR-schema attempts remain preserved there.
Independent review and script:
`normalization_input_owner_20260909/resolution_review.{json,py}` under the
source-review root; run the script with the primary pixi Python and CPU visibility.
Board receipt: `handoffs/em_clean_resolution_boundary_review_20260909.json`.
Outcome SHA-256: `e1f6f668d67263b5dd4724a40bc398a1656860fec0b0bac8c761afca762635da`.
The next scientific question is the upstream SSNR construction on matched state;
no numerical change, new GPU job, precision907 or F32 M adoption follows.


## Private F32 M inverse-FFT DC follow-up

The private fixed-M experiment `1b4f2adee` exposed an imaginary DC component
(about 0.00011390) at the inverse FFT boundary. NVIDIA documents that C2R input
must be Hermitian; violating this requirement leaves the result undefined.
See the [cuFFT transform contract](https://docs.nvidia.com/cuda/cufft/index.html#fourier-transform-types).
This requirement alone does not prove the cause of a measured map difference.
The peer subsequently ran saved-tail raw, DC-cleared and Hermitian-projected
controls, native CPU compatibility checks, and a corrected full-M replay.

Private correction `b17913a91`, separate from the explicit F32 capability commit,
clears only the imaginary centered DC immediately before the inverse FFT in both
M precisions. It retains real DC, all other Fourier coefficients and the earlier
moments/priors/accumulators. This is not a general Hermitian-boundary repair.
At the initial receipt review neither commit was adopted. The later authorized
integration below preserves the inherited F64 numerical M default.

Integrator review independently checked 53 named material hashes before/after,
the clean private source identities and nine changed-file hashes across the DC
and later route handoffs. All six old/new prepared-input pairs agree exactly,
including array dtype/shape/bytes and serialized scalars. Four saved map arrays
reproduce the following descriptive numerical comparisons:

| Saved-state comparison | Relative L2 map difference | Maximum absolute difference |
| --- | ---: | ---: |
| Original F32 versus original F64 | 6.928921219355242e-5 | 9.380339799081039e-7 |
| DC-corrected F32 versus corrected F64 | 2.3645888560075493e-7 | 5.055884433335933e-8 |
| Original versus corrected F64 | 2.1884647693715523e-16 | 5.551115123125783e-17 |

These are fixed-input numerical diagnostics, not map-quality gates. Independent
relative-norm recomputation differs from the producer's analysis by 4.46e-19 and
4.09e-28 in the latter two rows; maximum absolute differences are exact. Initial
review-script attempts asserted relative-norm agreement at 1e-12 and failed.
They remain preserved; the final report records these recomputation differences
without adding or changing a scientific tolerance.

The two saved native CPU maps are byte-identical over all 2,097,152 F64 voxels
when only gradient imaginary DC is removed. The integrator verified those files;
it did not rerun native code, E/M or FFTs. The peer reports 173 CPU passes,
four GPU tests passing after their CPU skips, guard38 and 18 corrected full-M
calls. Full peer source/dependency manifests and all replay outputs were not
independently re-audited in this receipt review.

A separate same-reference CPU check on serialized iteration-153 maps matches
native/JAX projector power at roughly 1e-16 relative L2. The incoming maps
reproduce the shell-27 prior-power difference. Stored MRCs were rounded to F32;
this does not replay full-precision in-memory references or identify their first
upstream divergence. That report was read and hashed, not independently rerun.
It does not connect the DC finding to the source-closed iteration-155 GT failure.

The later private `bae959dab` adds an explicit CLI-to-M F32 route. Peer H100 job
13644423 completed two updates with exact pose/support decisions, Pmax changes
up to 1.78e-5 and verified F32 M/state/mask boundaries. Other numerical stages
remain higher precision. Its focused44 and guard38 pass; three broader native
projector failures reproduce on its base and remain recorded. This smoke is
not trajectory acceptance or a benchmark. Full200 job13644924 is running on
that frozen source; preserve its source/native pins and avoid duplicate jobs.
The route has only source-identity/receipt review here, not integration approval.
At this initial receipt review, precision907, the F32 capability, DC fix and
route were private and unadopted. The subsequent integration decision follows.

Independent review and reproduction: run
`mstep_dc_receipt_review_20260909/review.py` under the source-review root with
primary pixi Python, `CUDA_VISIBLE_DEVICES=''`, `JAX_PLATFORMS=cpu` and one
OpenBLAS thread. Exact findings/pins are in `review.json`; peer artifacts are
under `em_work/codex/vdam_mstep_f32_replay_20260909/`. Coordination receipt:
`handoffs/em_clean_mstep_dc_receipt_review_20260909.json`.


## Authorized M capability and DC integration

Following explicit user authorization, merge **`29d7e38a500892b61e28579f251653bf0bef591b`**
incorporates capability `1b4f2adee` and DC correction `b17913a91` onto shared
`c9bc5ab0b`. Both original commits remain in history. Exactly four files change:
`helpers/relion_projector_setup.py`, `helpers/relion_vdam_mstep.py` and the two
new `initial_model/test_device_m_step_{precision,real_dc}.py` tests. Each merged
file is byte-identical to the reviewed handoff. Core, reconstruction, native
binding and existing affected tests are unchanged since the candidate base49ef.
No CLI route, precision907, baseline or non-EM behavior change is included.

The device/host M APIs accept explicit `compute_dtype`; inherited F64 remains
the default. Requested F32 uses an uncorrected F32 projector/FFT, rejects silent
native-double fallback for FFT grids below16 and preserves authoritative tau2.
M shell geometry keeps the existing F64 rounding rule. Persistent publication,
solvent masking, corrected projector refresh and other numerical stages retain
their existing precision. This capability does not establish an all-F32 pipeline.
The separately approved DC correction is active for both working precisions.

Integration validation on primary:

- Pre-merge control:156 existing native transaction/projector cases pass.
- Merged source:173 CPU cases pass in18.55s; four GPU cases explicitly deselected.
  These comprise the same156 existing cases plus17 new precision/DC checks.
- CPU/import guard:38 pass in43.44s. Both integrated panels preserve identical
  complete source manifests and the pinned native library hash.
- Existing A100 analytical tests:4 pass, zero skips. Integrator verified XML,
  both changed source/test hashes and the exact dirty precommit diff that became
  b179. This is reused private-source evidence, not a fresh primary GPU panel.
  The earlier18-call saved-state replay remains separately bounded evidence.

The first control attempt failed at fixture setup because the generic cleanup
wrapper's older native binding lacked `vdam_m_step_transaction`:19 passes and
137 setup errors. That output remains intact. A task-specific wrapper selects
existing pinned binding`6fefa350…` from
`vdam_source_closed_pair_20260909_v2/native/relion_bind/`; it does not rebuild or
mutate either binding. The corrected control and merged panels pass unchanged
assertions. This environment repair does not erase prior numerical failures.

Reproduce with the primary checkout and
`hia_source_review_20260906/mstep_dc_integration_20260909/run_checks.sh`:

```bash
bash "$REVIEW/run_checks.sh" <fresh-label> \
  tests/unit/initial_model/test_device_m_step_transaction.py \
  tests/unit/test_relion_projector_setup.py \
  tests/unit/initial_model/test_device_m_step_precision.py \
  tests/unit/initial_model/test_device_m_step_real_dc.py -m 'not gpu'
bash "$REVIEW/run_checks.sh" <fresh-guard-label> --fast-guard
```

Here `REVIEW` is the absolute source-review directory ending in
`mstep_dc_integration_20260909`. Scope, exact commands, checks and pins are in
`{scope,merge,validation,gpu_evidence_review,environment_repair}.json` there.
CPU logs/XML live under `em_work/codex/pr180_integration_20260908/` in
`mstep_dc_control_20260909`, `mstep_dc_control_v2_20260909`,
`mstep_dc_integrated_20260909` and `mstep_dc_guard_20260909`.

No new GPU or Slurm job was launched. Private routebae959's smoke13644423 and
full200job13644924 do not qualify this exact shared composition. Strict state,
pose/tie, trajectoryGT, current100k performance and exactK4/real-data completion
gates remain open. Capability/DC source admission does not mean those gates
passed. Precision907 and the CLI route were unmerged at that checkpoint; the
subsequent route integration is recorded below. Frozen peer sources, existing
failed runs and native dependencies remain untouched.

## Explicit CLI M precision route integration

User-authorized merge **`cb897cda5d917af37f0542ed741ac8b3de943c73`** preserves
original commit **`bae959dabe465dcb7e7d6d3aee586ed2039a9203`** on primary
`a4cb21490`. All five handoff files match exactly: `scripts/run_ab_initio.py`,
InitialModel `driver.py`, `iteration_loop.py`, `m_step.py`, and new
`test_mstep_precision_route.py`. Their preimages match candidate parentb179.
Precision907 remains excluded.

Select the capability explicitly with:

```bash
pixi run python scripts/run_ab_initio.py <existing-run-arguments> \
  --mstep-backend jax --mstep-compute-dtype float32
```

The driver converts six M-owned state fields once after bootstrap or continuation:
Iref, Igrad1/2, sigma2_class, data_vs_prior_class and fourier_coverage_class.
The transaction verifies F32/C64 output and publication, leaves other class slots
and incoming state untouched, and preserves authoritative tau2 exactly. Post-M
mask multiplication uses the selected dtype. Native backend, replay/dump routes,
missing transaction certificates and silent native-double fallback are rejected
when incompatible with the explicit F32 request. The inherited F64 default stays
unchanged. Tests include actual CPU M execution and K4 class-slot ownership;
this is not K4 trajectory qualification.

Bootstrap, corrected E projector/FFT and tau2 preparation, E normalization/noise,
BPref export and M shell geometry still include higher-precision numerical work.
Native initialization certificates evaluate the selected incoming moments,
including their explicit initial quantization. This is a bounded M capability,
not an all-F32 pipeline or a remedy established for earlier trajectory failures.

Primary validation uses pinned existing native binding `6fefa350…`:

| Check | Result |
| --- | --- |
| Existing driver/iteration/M callers before merge | 211 passed, 6.21 s |
| Same callers plus32 new route cases after merge | 243 passed, 6.38 s; no skips |
| CPU/import guard on identical merged source | 38 passed, 42.49 s |
| Private H100 CLI smoke13644423 | Two actual updates per precision; terminal0:0, 78 s allocation |

Integrator review independently checks the recorded dtype/state-stage inventory,
bootstrap equality, exact tau2 receipts and dependency path/hash equality; it
reintegrates all three saved FSC curves and compares actual saved pose/support
and Pmax metadata. Thirty-five named hashes are stable. Cross-arm FSC-AUC is
1 at initialization, 0.9999999997857151 at1 and 0.9999999997278919 at2. Both updates
preserve the six recorded pose/support fields, with maximum Pmax differences
1.52587890625e-5 and1.7762184143066406e-5. Continuous state differs. The observer's
cast/input-ownership attestations are saved receipts, not independent reconstruction
of omitted runtime arrays. This review does not repeat the producer's full2306-pin
inventory or execute E/M, FFT, native code or GPU work.

The earlier peer broader panel's three native-projector failures also reproduce
on unchangedb179; they remain unresolved and are not part of the passing243-case
inventory. Private H100 evidence matches the transferred route and named helpers,
but primary has newer structural changes elsewhere, so it does not establish
current-primary end-to-end acceptance. Full-200 job 13644924 is terminal 0:0 and
independently reviewed in the
[separate F32 M full200 record](vdam_f32_m_full200_review_20260909.md). All four
registered-GT conditions pass; all four strict cross-engine FSC histories fail.
Full-F32, strict state/pose/tie, current-primary quality, representative 100k runtime
and exactK4/real-data gates remain open.

Evidence root:
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/hia_source_review_20260906/mstep_route_integration_20260909/`.
Read `scope.json`, `merge.json`, `validation.json`, `gpu_receipt_review.json` and
`review_receipts.py`. Three initial receipt-review attempts are preserved: raw
ldd text differed in ASLR addresses, then independent AUC reducers differed in
DC exclusion and normalization order. The repaired review compares exact dependency
path/hash maps and the existing normalized-axis, DC-excluded AUC definition;
no scientific assertion, producer output or tolerance was relaxed.

Reproduce CPU checks using the existing task wrapper:

```bash
REVIEW=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/hia_source_review_20260906/mstep_dc_integration_20260909
bash "$REVIEW/run_checks.sh" <fresh-label> \
  tests/unit/initial_model/test_native_driver.py \
  tests/unit/initial_model/test_iteration_loop.py \
  tests/unit/initial_model/test_m_step.py \
  tests/unit/initial_model/test_native_m_step_transaction.py \
  tests/unit/initial_model/test_mstep_precision_route.py
bash "$REVIEW/run_checks.sh" <fresh-guard-label> --fast-guard
```

Exact commands, import provenance and XML/logs are under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/pr180_integration_20260908/`
in `mstep_route_{control,integrated,guard}_20260909/`. No source or library changed
during either integrated check; frozen peer sources and jobs were not modified.
