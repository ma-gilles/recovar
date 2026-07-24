# RECOVAR / RELION EM Parity Scorecard

**K=1 fixed-suite score: 25 / 34 passing (34 / 34 evaluated; 31 / 34 intermediate-topology passes).**

Suite: `k1-gui-grid0-local-highshell-full34` (version 1; denominator frozen at 34).
Frozen case-definition SHA-256: `9e3f2cb7192eb2cbf8a50181cf47de8562adfb98734bab05a736fb7d4d404fc1`.

A checked box means the complete autonomous FSC/FSC-AUC trajectory contract passed. Unchecked cases remain in the denominator. New diagnostics do not enter this suite; changing the case set or scientific definitions requires a new suite version.

The artifact-pinned fixture manifest is checked into the repository and binds all 34 cases (470,170,958,467 bytes) to exact file sizes and SHA-256 digests. Manifest SHA-256: `422a79a0a7703d92f9777266e8c34ccd3a7cf5963b354e57a7d9a18f227babee`. Regenerated inputs are non-scoring replicates.

Acceptance uses shellwise FSC and normalized FSC-AUC, exact schedule/topology, convergence/finalization semantics, same-physical-GPU RELION/RECOVAR pairs, grid correction unset/off, and no forced K-class-like finalization. Correlation is not computed or gated.

Evidence snapshot: `em_k1_gui_grid0_local_highshell_full34_superseding_ledger_v6`, generated `2026-07-24T01:04:11.826284+00:00`, JSON SHA-256 `32c6512a8507f7b17a59d0be527fa5c9609067e0d8f598a2d108bed9a3fc8a56`.
Progress: +5 passing cases since the first frozen snapshot; +2 since the previous snapshot.

| Done | Case | Fixture | Trajectory | Topology | Final cross-engine FSC-AUC | Final GT delta | Jobs |
|---|---|---|---|---|---:|---:|---|
| [x] | `k1-01` | `baseline_100k_g256_white_noise1_bf80` | pass | pass | 0.998379294 | +0.008163340 | science 11384176; trajectory 11384362; intermediate 11384363 |
| [x] | `k1-02` | `more_images_200k_g256_white_noise1_bf80` | pass | pass | 0.998574606 | +0.005625197 | science 11501888; trajectory 11501907; intermediate 11501907 |
| [ ] | `k1-03` | `more_images_300k_g256_white_noise1_bf80` | fail | fail | 0.998786703 | +0.005373891 | science 11384178; trajectory 11384366; intermediate 11384367 |
| [ ] | `k1-04` | `high_noise_100k_g256_white_noise3_bf80` | fail | pass | 0.991556309 | +0.003869282 | science 11384179; trajectory 11384368; intermediate 11384369 |
| [ ] | `k1-05` | `very_high_noise_100k_g256_white_noise10_bf80` | fail | pass | 0.985743479 | +0.000544950 | science 11384180; trajectory 11384370; intermediate 11384371 |
| [x] | `k1-06` | `noctf_control_100k_g256_white_noise3_bf80` | pass | pass | 0.997522945 | +0.005563842 | science 11384181; trajectory 11384372; intermediate 11384373 |
| [ ] | `k1-07` | `anisotropic_100k_g256_white_noise1_bf80` | fail | fail | 0.843316945 | +0.006670338 | science 11384182; trajectory 11384374; intermediate 11384375 |
| [x] | `k1-08` | `anisotropic_high_noise_100k_g256_white_noise3_bf80` | pass | pass | 0.996260789 | +0.001007928 | science 11384183; trajectory 11384376; intermediate 11384377 |
| [x] | `k1-09` | `high_res_near_nyquist_100k_g384_white_noise1_bf0` | pass | pass | 0.995510893 | +0.003664545 | science 11432807; trajectory 11454201; intermediate 11432810 |
| [ ] | `k1-10` | `high_res_anisotropic_100k_g384_radial_noise3_bf0` | fail | pass | 0.983006504 | +0.000128347 | science 11421265; trajectory 11454202; intermediate 11421267 |
| [x] | `k1-11` | `small_baseline_3k_g128_white_noise1_bf80` | pass | pass | 0.998515876 | +0.019981607 | science 11384186; trajectory 11384382; intermediate 11384383 |
| [x] | `k1-12` | `small_very_high_noise_3k_g128_white_noise10_bf80` | pass | pass | 0.998135578 | +0.002422120 | science 11384187; trajectory 11384384; intermediate 11384385 |
| [x] | `k1-13` | `small_anisotropic_3k_g128_white_noise3_bf80` | pass | pass | 0.997569995 | +0.011223852 | science 11385531; trajectory 11385557; intermediate 11385558 |
| [x] | `k1-14` | `small_noctf_3k_g128_white_noise3_bf80` | pass | pass | 0.997775943 | +0.017579713 | science 11385532; trajectory 11385559; intermediate 11385560 |
| [x] | `k1-15` | `small_outliers_3k_g128_pct20_noise1_bf80` | pass | pass | 0.998431014 | +0.019058454 | science 11385533; trajectory 11385561; intermediate 11385562 |
| [x] | `k1-16` | `small_anisotropic_outliers_3k_g128_pct25_noise3_bf80` | pass | pass | 0.996556471 | +0.008243245 | science 11385534; trajectory 11385563; intermediate 11385564 |
| [x] | `k1-17` | `small_extra_particles_3k_g128_noise1_bf80` | pass | pass | 0.998794039 | +0.016078006 | science 11385535; trajectory 11385565; intermediate 11385566 |
| [x] | `k1-18` | `small_contrast_noise_scale_3k_g128_noise1_bf80` | pass | pass | 0.998712222 | +0.014739591 | science 11385536; trajectory 11385567; intermediate 11385568 |
| [x] | `k1-19` | `small_image_offset_3k_g128_noise1_bf80` | pass | pass | 0.998259358 | +0.020106905 | science 11385537; trajectory 11385569; intermediate 11385570 |
| [x] | `k1-20` | `small_high_res_radial_3k_g256_noise3_bf0` | pass | pass | 0.998129368 | +0.001149427 | science 11498687; trajectory 11498738; intermediate 11498738 |
| [x] | `k1-21` | `small_kent_angles_3k_g128_white_noise3_bf80` | pass | pass | 0.998345537 | +0.010110173 | science 11385539; trajectory 11385573; intermediate 11385574 |
| [ ] | `k1-22` | `small_severe_outliers_3k_g128_radial_noise5_bf80` | fail | fail | 0.825938890 | -0.000351848 | science 11385540; trajectory 11385575; intermediate 11385576 |
| [x] | `k1-23` | `small_noctf_radial_3k_g128_noise3_bf80` | pass | pass | 0.998342408 | +0.012298496 | science 11501524; trajectory 11501622; intermediate 11501622 |
| [ ] | `k1-24` | `small_kent_outliers_3k_g128_pct20_noise3_bf80` | fail | pass | 0.994805104 | +0.008173298 | science 11385542; trajectory 11385579; intermediate 11385580 |
| [x] | `k1-25` | `tiny_baseline_1k_g128_white_noise3_bf80` | pass | pass | 0.998192576 | +0.009181804 | science 11385543; trajectory 11385581; intermediate 11385582 |
| [ ] | `k1-26` | `tiny_severe_1k_g128_radial_noise5_nonuniform_pct30_bf80` | fail | pass | 0.954913646 | +0.010098947 | science 11385544; trajectory 11385583; intermediate 11385585 |
| [x] | `k1-27` | `small_extreme_outliers_3k_g128_pct70_noise1_bf80` | pass | pass | 0.998332271 | +0.010086417 | science 11385545; trajectory 11385587; intermediate 11385588 |
| [x] | `k1-28` | `small_kent_extra_offset_3k_g128_noise3_bf80` | pass | pass | 0.998534963 | +0.016603039 | science 11384203; trajectory 11384427; intermediate 11384428 |
| [x] | `k1-29` | `small_low_noise_3k_g128_white_noise0p2_bf80` | pass | pass | 0.998867525 | +0.014987020 | science 11384204; trajectory 11384429; intermediate 11384430 |
| [x] | `k1-30` | `small_low_noise_kent_3k_g128_white_noise0p2_bf80` | pass | pass | 0.998823366 | +0.013967656 | science 11384205; trajectory 11384433; intermediate 11384434 |
| [x] | `k1-31` | `mid_10k_g128_white_noise1_bf80` | pass | pass | 0.998725941 | +0.016924536 | science 11384206; trajectory 11384436; intermediate 11384437 |
| [ ] | `k1-32` | `mid_10k_kent_g128_radial_noise3_bf80` | fail | pass | 0.974500501 | +0.004132488 | science 11384207; trajectory 11384438; intermediate 11384439 |
| [x] | `k1-33` | `max_images_400k_g128_white_noise1_bf80` | pass | pass | 0.999734254 | +0.000244294 | science 11508260; trajectory 11508286; intermediate 11508286 |
| [x] | `k1-34` | `max_images_400k_g128_radial_noise3_nonuniform_bf80` | pass | pass | 0.995757412 | +0.002869240 | science 11384210; trajectory 11384443; intermediate 11384444 |

## Progress history

| Snapshot | Date (UTC) | Commit boundary | Passed | Δ passed | Failed | Not evaluated/error |
|---|---|---|---:|---:|---:|---:|
| `strict-k1-v1-old-head-20260721` | 2026-07-21T04:33:00.281935+00:00 | `ac5177d2b0cd` | 20 | — | 12 | 2 |
| `strict-k1-v3-20260721` | 2026-07-21T10:35:40.626248+00:00 | `ac5177d2b0cd`, `9d1722781e1d` | 21 | +1 | 13 | 0 |
| `strict-k1-v4-20260722` | 2026-07-22T15:57:09.593124+00:00 | `ac5177d2b0cd`, `9d1722781e1d`, `6ddd094011db` | 22 | +1 | 12 | 0 |
| `strict-k1-v5-20260722` | 2026-07-22T19:00:51.329249+00:00 | `ac5177d2b0cd`, `9d1722781e1d`, `6ddd094011db`, `ab52b1ff4038` | 23 | +1 | 11 | 0 |
| `strict-k1-v6-20260724` | 2026-07-24T01:04:11.826284+00:00 | `ac5177d2b0cd`, `9d1722781e1d`, `6ddd094011db`, `ab52b1ff4038`, `84143872a517`, `a2be302cdc08` | 25 | +2 | 9 | 0 |

<!-- BEGIN MANUAL POST-SNAPSHOT DIAGNOSTICS -->
## Post-snapshot fixed-fixture intervention diagnostics

These rows use frozen case bytes but do not rewrite the immutable snapshot
above.  A failing intervention remains unchecked and does not change the
25/34 score.

| Done | Case | Commit/intervention | Trajectory | Topology | Final cross-engine FSC-AUC | Final GT delta | Jobs |
|---|---|---|---|---|---:|---:|---|
| [ ] | `k1-03` | `84143872`; unchanged fixed-suite science with 36-hour budget after the prior 24-hour timeout | pending | pending | — | — | science 11553236; strict audit 11553237 |
| [ ] | `k1-04` | `c74beea4`; direct-real initial projector + bounded firstiter top-2 tree rescore | pending | pending | — | — | setup 11563826; science 11563827; summary 11563828; strict audit 11563842 |
| [ ] | `k1-05` | `c74beea4`; identical case-4 intervention, frozen-fixture generalization | pending | pending | — | — | setup 11564052; science 11564053; summary 11564054; strict audit 11564062 |
| [ ] | `k1-24` | `b826bc52`; direct-real initial projector + bounded firstiter top-2 tree rescore | fail | pass | 0.994801463 | +0.008173125 | setup 11562037; science 11562038; summary 11562039; strict audit 11562082 |

The case-4 intervention is motivated by the complete same-H100
first-iteration coarse-grid diagnostic `11562639`, which completed `0:0`.
All 1,069,056 RELION and RECOVAR candidate identities agree.  The aligned
score correlation is `0.9999999999954908`, and the centered score difference
has p95 absolute `5.1409006e-7` and maximum absolute `1.4603138e-6`.
RELION's two best hypotheses have exactly equal float32 scores
(`0.2807506024837494`), while RECOVAR separates them by only
`1.7881393432617188e-7` and selects the opposite 150.7523-degree winner.
This supports the bounded RELION 128-lane re-reduction; it is not yet a
fixed-suite pass.  Comparison JSON SHA-256:
`2e3368c5c03db4d0eea9519c746be6c4d4b26f8b8b0f11e98420ee6d878ebcdd`.

The case-5 arm applies exactly the same bounded intervention to an independent
frozen fixture, without a case-5-derived code or threshold change.  Its
accepted baseline particle audit contains three first-iteration assignment
exceptions (original indices 26055, 93729, and 95412), all translation-only at
the reporting tolerance.  It is a generalization check, not a new metric row;
the frozen score remains unchanged while its strict auditors are pending.
On integration head `2dfafb5a`, the exact direct-real projector, bounded
top-two replacement, and Slurm forwarding/scope tests pass 3/3.  The JUnit
SHA-256 is
`83356757653924ee61b1a3bda00a737c356ce2026dbe2d5c0d6d287707df610c`.

The case-24 intervention is effectively exact for the first three numbered
maps (merged cross-engine FSC-AUC `0.999999999973`,
`0.999999999903`, `0.999999999901`).  The first material map drift is
iteration 4.  Particle-state audit localizes its seed to one 2.125 Angstrom
translation decision at iteration 3.  Patched RELION operand replay `11562574`
completed with exact 64/64 fine support, exact 12/12 reconstruction support,
posterior correlation `0.999999998039`, and the same translation-59 winner as
RECOVAR.  That replay used RELION source `f2c1a3`, not the installed stock
`d476e6`, so its one-float32-unit raw-score gap is mapping evidence rather than
an exact stock score oracle.  Same-node/same-GPU no-dump job `11562830`
completed `0:0`; both installed stock and the older patched binary chose
translation 59.  The immediately adjacent installed-stock replicate in
source-exact probe `11563252` chose translation 57.  The two stock runs have
identical iteration-1 state and, through iteration 3, merged cross-run FSC-AUC
of at least `0.999999999992`; only the iteration-3 x translation differs.
This directly classifies the hard winner as launch-sensitive at the numerical
boundary.  Job `11563252` completed `0:0`: its source-exact d476e6 active
capture chose translation 59 and measured translation 59 versus 57 at exactly
one float32 raw-score ULP (`0.0001220703125`) and
`3.9343036e-5` posterior.  Its dormant-instrumentation arm chose 57.  This is
oracle-stability telemetry, not a new strict pass; the fixed denominator and
score remain unchanged.

## K=1 frozen case-22 combined-intervention rerun

Case 22 is one of the two remaining exact-topology failures.  Its known
iteration-2 numerical butterfly is driven by the incoming reference, so the
same bounded direct-real-reference and first-iteration top-two reduction-tree
intervention under test on cases 4/5 is now running against the immutable
3,000-particle/grid-128 fixture.  The clean detached source is
`b1d444270de89a4ede0868fe0e39954d012fd593`.

Setup `11566711` completed, science `11566712` is running, summary
`11566713` is dependency-gated, and the unchanged strict FSC/topology audit is
`11566739`.  Grid correction and forced final all-data after maximum iteration
are unset.  The durable root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fixedsuite_case22_tree_b1d44427_20260724T142000Z`.

The first two RECOVAR boundaries now rule out that intervention as the cause
of case 22's butterfly.  The top-two rescore changed zero of 3,000 winners.
Iteration-1 merged new-versus-old FSC-AUC is `0.999999999984`, and CPU audit
`11567536` finds iteration-2 merged new-versus-old FSC-AUC
`0.999999999942`.  Iteration-2 new-versus-RELION FSC-AUC is
`0.999993352100` versus the frozen old row's `0.999993352125`; GT FSC-AUC is
also effectively unchanged (`0.215965204` versus `0.215965248`, with RELION
at `0.216062652`).  The divergence therefore begins downstream of the
firstiter direct-reference/top-two path.  The autonomous science and strict
audit have now completed as a strict failure.  Science `11566712` wrote all
artifacts and returned the launcher's expected quality-failure exit `2:0`;
the dependency-impossible zero-runtime audit `11566739` was replaced by direct
unchanged terminal audit `11567655`.

The terminal FSC audit fails first at numbered iteration 9
(`0.989830716627 < 0.995`), while iteration 10 is `0.997438778645` and final
merged cross-engine FSC-AUC is `0.826260991659`.  Final merged GT delta is
`-0.000434619684`.  Exact topology has 10 RECOVAR versus 11 RELION numbered
iterations; iteration 9 differs at current size `72` versus `70` and HEALPix
order `4` versus `5`, with the order mismatch continuing through iteration
10.  Both unchanged auditors return status 2.  This confirms the existing
strict/topology failure and does not change the frozen score of 25/34.

CPU scheduler-boundary audit `11568050` localizes the topology split without
supporting a scheduler override.  Iteration 8 still has equal size 70/order 4
topology, but RELION remains at shell 19 (`28.631579` Angstrom; resolution
stall counter 2), while RECOVAR crosses to shell 20 (`27.20` Angstrom; counter
0).  At iteration 9, both measured angular accuracies say the 1.875-degree
grid is not fine enough (`2.479` degrees RELION, `2.469` RECOVAR), but only
RELION's accumulated stall state is ready to advance.  The size/order split is
therefore a downstream amplification of the map/FSC-shell butterfly.

## K=1 frozen case-7 firstiter generalization

Frozen case 7 tests the same intervention against a longer 100,000-particle
fixture whose known shell-20 scheduler split amplifies accumulated drift.
Clean detached source is `c74beea47a1a91a723ab2c99f961b8a70483c34c`, and
the immutable fixture-manifest SHA-256 is
`422a79a0a7703d92f9777266e8c34ccd3a7cf5963b354e57a7d9a18f227babee`.
Setup `11567460` completed; science `11567461`, summary `11567462`, and
independent strict FSC/topology audit `11567496` are queued.  The audit is
fail-closed behind successful science and independently checks source/import
provenance.  The durable run root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fixedsuite_case07_tree_c74beea4_20260724T110500Z`.
This is a pending diagnostic, not a score change.

## K=1 final-only family diagnostic

Frozen case 10 already passes all 15 numbered FSC and exact-topology gates but
fails only the final merged cross-engine FSC-AUC (`0.983006503534 < 0.995`).
CPU audit `11566606` completed `0:0` and passed exact iteration-15 schedule,
convergence, finalization, and 100,000-particle image-identity gates.  From the
last numbered state (current size 68) to final all-data (full grid 384), the
fraction of particles within 0.5 degrees changes only
`91.501% -> 91.457%`; angular-error p95 changes
`0.990475839 -> 1.001330626` degrees.  The fraction within 0.5 Angstrom
translation changes `92.650% -> 92.531%`, with p95 fixed at
approximately `0.708333` Angstrom.  Pmax absolute-error p95 changes only
`0.006589644 -> 0.006748276`.

Despite that nearly stationary particle-state tail, merged cross-engine
FSC-AUC falls from `0.999967227122` at iteration 15 to `0.983006503534` at
final all-data.  Case 10 therefore joins cases 4/5/24/26/32: the shared
full-grid final expectation amplifies an inherited pose/reference/posterior
tail rather than creating a new final pose-writeback or scheduling mismatch.
The audit JSON and aligned-array NPZ SHA-256 values are
`eb0ac4bcbecbfa4d9333ececc6340ed5fb4dfedb6c19745472b205b5b2582dbd`
and `c44b7a4a85af54413c04f50572a6f0b805dfa5f31c63542a395926c3e8c1bab0`.
The durable root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case10_last_final_audit_20260724T140800Z`.
This is localization evidence, not a new checkbox; the frozen score remains
25/34.

Complete exact-identity audits `11571320_0` and `11571320_1` extend this
classification across every numbered state for frozen cases 4 and 5.  The
intervention leaves only two material case-4 iteration-1 exceptions (original
indices `5234` and `72654`) and three translation-only case-5 exceptions
(`38594`, `65070`, and `93729`); Pmax and significant-support arrays are
otherwise exact at that boundary.  The greater-than-0.1-degree pose tails then
grow from `1 -> 6,017` and `0 -> 7,269` particles by the last numbered state,
while last-numbered to final changes are small.  This confirms that final
full-grid reconstruction amplifies an inherited first-iteration
winner/reference/posterior butterfly.  H100 array `11571746` and fail-closed
analysis `11571905` reproduce all five exceptions and capture all 1,069,056
coarse scores for each target.  Case-4 particles `5234` and `72654` have
native float32 top-two margins `3.27826e-7` and `2.38419e-7`; case-5 particle
`93729` has margin `4.61936e-7`.  All three are inside the bounded `4e-6`
rescore band, but tree rescore leaves their native winners unchanged.
Case-5 particles `38594` and `65070` instead have margins `9.89005e-4` and
`1.83816e-3`, far outside that band.  This rules out a global threshold
increase.  Exact fine-pose-to-coarse-grid mapping further separates the
boundary: both case-4 targets and case-5 particle `93729` map to RECOVAR's
coarse runner-up, while case-5 particles `38594` and `65070` already use
RELION's coarse parent and therefore diverge inside fine pass 2.  The mapping
JSON SHA-256 is
`6ad97a96805f77d67af78418e1460239ff06f4d46336aa9cd1032ce08d371cf5`.
Same-physical-H100 patched-RELION full-grid discriminator `11572062` runs for
particle `5234`; fine-pass capture `11572658` is queued for particles `38594`
and `65070`.  The frozen score remains 25/34 strict, 31/34 topology, and
34/34 evaluated.

## K=4 physical-GPU trajectory diagnostic

This diagnostic is not part of the frozen K=1 denominator.  It compares two
otherwise identical RECOVAR K=4 trajectories that ran on different physical
A100 GPUs.  It uses only shellwise FSC and normalized FSC-AUC; correlation is
not computed.  The tiny iteration-1 numerical envelope amplifies into discrete
assignment and schedule-state differences:

| Numbered iteration | Worst map FSC-AUC defect | Half-1 assignment mismatches | Half-1 coarse mismatches | Noise relative L2 | Tau2 relative L2 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.000000063 | 0 / 100000 | 0 / 100000 | 0.000000000 | 0.000000000 |
| 4 | 0.000009934 | 4 / 100000 | 2 / 100000 | 0.000000054 | 0.000000135 |
| 5 | 0.000122543 | 44 / 100000 | 28 / 100000 | 0.000000407 | 0.000026835 |
| 7 | 0.000637164 | 420 / 100000 | 220 / 100000 | 0.000001806 | 0.000082195 |
| 9 | 0.001751329 | 908 / 100000 | 501 / 100000 | 0.000004783 | 0.000338354 |

CPU audit job `11564864` completed `0:0`.  The JSON artifact SHA-256 is
`afac855a3d7423998d14082be429f8243f247a6ab880f95b497c099c2d20ac00`;
the shellwise NPZ SHA-256 is
`ad1adfea9a3fc164f6c8f3671a615ac5cb90475ae5b58e8daf3e502d93b679ee`.
This establishes why a strict K=4 acceptance comparison must bind RELION
control, RELION capture, and RECOVAR to one physical GPU UUID.

The cross-A100 recovery job `11561204` reached numbered iteration 10 but
failed before completing the 96-particle capture.  The failure was a
diagnostic invariant bug, not OOM: class-2 capture rejected a particle whose
jointly normalized K-class posterior legitimately had zero reconstruction
rows in class 2.  Commit `9dcd709b` preserves such a particle's native launch
and ownership manifest, writes a schema-valid zero-contributor shard, and
keeps finite/nonnegative operand, exact-zero omission, and WTA upper-bound
checks fail-closed.  The targeted device-signature/canonical-replay suite is
92/92 passing.

The vulnerable same-GPU graph (`11564419`, `11564442`, `11564443`) was
canceled before the known iteration-10 failure; both auditors had zero
runtime.  Its replacement is a fresh non-resumed one-allocation graph at
`9dcd709b`: science `11565045`, zero-contributor-aware vector audit
`11565121`, and independent scalar audit `11565131`.  The initially submitted
auditors (`11565048`, `11565050`) were canceled at zero runtime after preflight
found the same invalid per-class nonempty assumption in their comparators.
The graph reruns both RELION arms and RECOVAR sequentially on one physical
A100 and does not promote outputs from any canceled graph or auditor.

The replacement RELION control/capture pair passed its fail-closed inertness
gate.  Dispatch logs are bitwise equal; all 96 particles were captured with
the expected 48/48 MPI ownership split; and the four class-map normalized
FSC-AUC values are at least `0.999999992551`.  The accepted inertness JSON
SHA-256 is
`9c1cf28f563d0f4a4e9e202cf6d0a3af1847012d43c45c3089d8e6fafc5c85f5`.
RECOVAR then passed checkout, CUDA-device, and RELION-binding provenance gates
on the same allocation.  Science and both dependent auditors remain pending;
this does not change the frozen K=1 score.

Frozen case 5 has also reached its first RECOVAR boundary under the unchanged
direct-real-reference plus bounded `4e-6` top-two intervention.  The rescore
changed four of 100,000 winners (`3/1` by half), which cannot be exactly the
accepted baseline exception set (`1/2` by half).  CPU FSC audit `11567287`
and the superseding within-physical-GPU-pair audit `11568205` show that
iteration-1 merged cross-engine FSC-AUC improves from `0.999999999628` to
`0.999999999798`.  The older cross-GPU comparison (`0.999999997357`) is not
used for the causal delta.  Merged GT FSC-AUC moves from
`0.103266845326` toward `0.103266884384`; the matched RELION values are
`0.103266904607` and `0.103266904543`.  This is positive first-boundary
evidence, not a full-case pass; science `11564053` and strict audit
`11564062` remain active or dependency-gated.

Iteration 2 independently preserves the cross-engine gain.  CPU audit
`11567559` reports merged new-versus-RELION FSC-AUC `0.999999972228`, versus
the matched old pair's `0.999999951558`; half-1/half-2 improve from
`0.999999933191`/`0.999999886717` to
`0.999999992764`/`0.999999904988`.  New merged GT FSC-AUC is
`0.107758163608` versus matched RELION `0.107757661217`; the old pair is
`0.107757134405` versus `0.107756806991`, so GT closeness does not improve.
This remains intermediate evidence; the immutable score does not change
before terminal science and both strict auditors pass.

The gain also survives the first two large-grid transitions.  CPU audit
`11567932` reports iteration-3 merged cross-engine FSC-AUC improving from
the matched old pair's `0.999999237418` to `0.999999428673`, and iteration 4
from `0.999998069731` to `0.999998178270`.  Audit `11568517` reports that
iteration 5 improves from `0.999971221351` to `0.999986014860`, about a
two-fold smaller FSC defect.  At iteration 6, audit `11568660` records the
first reversal: cross-engine FSC-AUC changes from `0.999989489556` old to
`0.999988729729` new, about a 7.2% larger defect, while GT closeness improves.
Iteration 7 improves again, from `0.999985251367` to `0.999986793254`
(about a 1.12-fold smaller defect), while GT closeness worsens.  RELION
cross-run FSC-AUC is only `0.999984664365` at that boundary, so the late
old/new delta is not isolated from physical-GPU/run drift.  The intervention
does not dominate the old trajectory at every boundary.  These are still
numbered-boundary diagnostics, not a terminal acceptance.

Iteration 8 improves only slightly, from `0.999976386019` to
`0.999977086595` (about a 1.03-fold smaller defect), while GT closeness
improves.  RELION cross-run FSC-AUC is `0.999973486836`, again comparable to
the within-pair defects.  This late delta is observational rather than a
causal acceptance signal.

Iterations 9--12 remain far above the numbered `0.995` gate, with new
cross-engine FSC-AUC `0.999968659886`, `0.999962386862`,
`0.999956585373`, and `0.999956018939`.  Relative to the matched old pair,
iterations 9--10 improve negligibly, while 11--12 worsen by about `8.8%` and
`1.9%` in FSC-defect terms.  RELION cross-run FSC-AUC is
`0.9999499`--`0.9999649`, comparable to these defects, and GT closeness
improves at all four boundaries.  Terminal same-GPU acceptance remains the
only promotion gate.

Iterations 13--15 remain in the same run-sensitive regime.  Their new
cross-engine FSC-AUC values are `0.999954301471`, `0.999954396064`, and
`0.999954431570`, about `5.5%`, `2.6%`, and `2.3%` worse in FSC-defect terms
than the matched old pair.  Iteration 16 reverses that small loss:
`0.999955965898` new versus `0.999954256087` old, about a `1.04`-fold smaller
defect.  GT closeness improves at all four boundaries; at iteration 16 the
new RECOVAR/RELION GT FSC-AUC difference is only `0.000000703`.  RELION
cross-run FSC-AUC remains about `0.999950` throughout.  Array audit
`11569878` and iteration-16 audit `11569901` completed `0:0`; all 16 numbered
boundaries pass.  The terminal all-data boundary does not: strict audit
`11564062` fails only final merged cross-engine FSC-AUC,
`0.985721587320 < 0.995`, while half-1/half-2 are
`0.989930339640`/`0.988255553139` and the GT delta is positive
(`+0.000330734`).  Exact topology passes.  The frozen old final value is
`0.985743479037`, so the intervention does not improve the terminal result.
Accepted within-pair terminal audit `11570953` further reports RELION
cross-run final FSC-AUC only `0.986463892284`, comparable to the failed
within-pair value; terminal old/new deltas are not causal intervention
measurements.  The frozen score therefore remains unchanged.

Shell-profile audit `11569181` localizes `94.6%` of the iteration-6 negative
AUC delta to shells 1--64, with the largest losses at shells 53--56 rather
than the high-shell tail.  RELION cross-run FSC at those four shells is only
`0.998565`--`0.999542`, while neither within-pair curve has any shell below
`0.995`.  This is a run-sensitive mid-shell butterfly, not evidence for a
new arithmetic or scheduler patch.

Frozen case 4 has now reached its first RECOVAR boundary under the same
intervention.  The bounded rescore changed six of 100,000 winners (`2/4` by
half).  CPU FSC audit `11567836` reports iteration-1 merged
new-versus-RELION FSC-AUC `0.999999999398`, versus the frozen old trajectory's
`0.999999987721`; half-1/half-2 improve within their matched pairs from
`0.999999978736`/`0.999999984761` to
`0.999999999712`/`0.999999998283`.  New merged GT FSC-AUC is
`0.104211187030`, much closer to RELION's `0.104211182503` than the old
trajectory's `0.104211286116`.  Iteration 1 keeps the expected size 56,
30.22 Angstrom boundary and chooses size 100 for iteration 2, matching the
frozen trajectory.  Within-pair audit `11568204` confirms that the improvement
is not cross-GPU RELION drift.  At iteration 2, merged cross-engine FSC-AUC
improves from `0.999999203271` to `0.999999845857` (about a five-fold smaller
defect), although GT closeness worsens.  Audit `11568516` shows that iteration
3 improves from `0.999993389833` to `0.999997980427`, about a 3.27-fold
smaller defect, while GT closeness improves again.  Iteration 4 also improves
from `0.999976663509` to `0.999991309525`, about a 2.69-fold smaller defect,
with a strong GT-closeness gain.  Iteration 5 improves cross-engine FSC-AUC
from `0.999946344632` to `0.999972054601` (about a 1.92-fold smaller defect),
but GT closeness worsens sharply.  Iteration 6 improves cross-engine FSC-AUC
from `0.999924548651` to `0.999953210339` (about a 1.61-fold smaller defect)
and improves GT closeness.  Iterations 7--9 continue the matched improvement:
new versus old cross-engine FSC-AUC is
`0.999933403200` versus `0.999904546987`,
`0.999910731841` versus `0.999883427749`, and
`0.999867699585` versus `0.999830531488`, respectively, or about
`1.43`-, `1.31`-, and `1.28`-fold smaller FSC defects.  GT closeness worsens
at iterations 7--8 and improves at iteration 9.  RELION cross-run FSC-AUC
falls to `0.9999127`--`0.9999609`, so these late deltas remain observational.
Audits `11569878`, `11569970`, and `11570004` completed `0:0`.  This is mixed
numbered-boundary evidence.  Iterations 10--17 also improve the matched FSC
defect by `1.28`--`1.36` fold and preserve convergence at numbered iteration
17; GT closeness improves at six of those eight boundaries.

The terminal gate still fails.  Strict audit `11563842` reports half-1,
half-2, and merged cross-engine FSC-AUC
`0.994737855585`, `0.993581124654`, and `0.992965911620`; only the final
merged gate fails, the GT delta is positive (`+0.003915953`), and exact
topology passes.  This improves the frozen old final value
`0.991556308523`, and final particle-state diagnostics improve as well:
mean pose error `0.2730` to `0.1967` degrees and mean translation error
`0.02733` to `0.02197` pixels.  However, accepted within-pair terminal audit
`11570953` measures RELION cross-run final FSC-AUC only `0.994964023912`,
already below the gate.  The terminal improvement is therefore observational
under substantial run/GPU butterfly amplification and cannot promote case 4.

The same-GPU K=4 science job `11565045` has completed iterations 1--10 with
sizes `38,38,42,56,60,62,68,70,72,74`, resolutions
`60.44,49.45,30.22,27.20,25.90,22.67,21.76,20.92,20.15,19.43` Angstrom,
and iteration-10 Pmax `0.915066`; the unconverged trajectory continues into
iteration 11 at size 76 under its configured 15-iteration ceiling.  This
exact size/resolution topology matches both prior corrected `c390f8bf`
diagnostic trajectories through this boundary.
Independent early map audit `11569628` passes iteration 7 with identity class
matching, minimum classwise cross-engine FSC-AUC `0.996806796`, and worst GT
FSC-AUC delta `-0.000076974`, inside the unchanged `0.995/-0.002` gates.
The formerly decisive iteration-8 boundary now also passes in independent
audit `11570025`: identity class assignments, classwise cross-engine FSC-AUC
`0.997311835`, `0.996462452`, `0.996514489`, and `0.997541298`, and worst GT
delta `-0.000029845`.  The accepted stdout SHA-256 is
`9835a0e24404f29ccda4ca28c9d1991cafd03c0ace6a85b0944214e9f5fda72f`.
Independent fail-closed audit `11573095` finds that iteration 11 has crossed
the strict map gate: identity class assignments remain exact and all GT
deltas are within `-0.000073031`, but class-2/class-3 cross-engine FSC-AUC is
`0.994509131`/`0.994150545`, below `0.995`; class 1/4 remains
`0.995505295`/`0.995592423`.  Iterations 9 and 10 are being audited to locate
the first failing boundary.
Class agreement remains unmeasured until the terminal result exists.
Vector audit `11565121` and scalar audit `11565131` remain dependency-gated,
so this is not yet K=4 acceptance.
<!-- END MANUAL POST-SNAPSHOT DIAGNOSTICS -->

## Non-scoring regenerated-data diagnostics

These runs exercise the same parameter definitions with newly generated particle bytes. They are useful robustness evidence but never change the fixed-suite score.

| Case | Trajectory | Topology | Final cross-engine FSC-AUC | Final GT delta | Jobs |
|---|---|---|---:|---:|---|
| `k1-20` | fail | pass | 0.994327835 | +0.001101584 | science 11497499; audit 11498100 |
| `k1-22` | fail | fail | 0.825960160 | -0.000306153 | science 11497555; audit 11497575 |
| `k1-23` | pass | pass | 0.997483478 | +0.012306248 | science 11497556; audit 11497576 |
| `k1-24` | fail | pass | 0.989305857 | +0.008729054 | science 11497557; audit 11497577 |

Generate this PR-ready table with:

```bash
pixi run python scripts/summarize_em_relion_parity_scorecard.py
```

Verify that the checked scorecard, frozen snapshot, and marked live-diagnostics
appendix are current with:

```bash
pixi run python scripts/summarize_em_relion_parity_scorecard.py \
  --check docs/math/em_relion_parity_scorecard.md
```

After a terminal strict auditor passes, build a fail-closed candidate
superseding ledger with `--proposal-output`. The command validates the
the pinned fixture-manifest bytes and re-hashes every materialized byte, clean source and
submitted job/case-table identity, same physical GPU, autonomous
FSC/topology audits, convergence/finalization contract, and evidence
hashes. It never mutates the checked scorecard. For example:

```bash
pixi run python scripts/summarize_em_relion_parity_scorecard.py \
  --proposal-previous-ledger /absolute/path/to/current-ledger.json \
  --proposal-ledger-schema em_k1_gui_grid0_local_highshell_full34_superseding_ledger_v7 \
  --proposal-generated-utc 2026-07-24T13:00:00+00:00 \
  --proposal-status-note "Case k1-NN passed immutable strict evidence." \
  --proposal-evidence 'k1-NN|/absolute/path/to/case-root|SCIENCE_JOB|AUDIT_JOB' \
  --proposal-output /absolute/path/to/proposed-ledger.json
```

Launch a scoring rerun with `--scorecard`. This fail-closed mode requires the
checked-in fixture manifest/root pair and forces autonomous RELION pairing,
per-iteration RECOVAR maps, grid correction off, and valid convergence-only
finalization. For example:

```bash
EM_K1_MATRIX_FIXTURE_MANIFEST="$PWD/docs/math/em_relion_parity_fixture_manifest_v2.json" \
EM_K1_MATRIX_FIXTURE_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex \
EM_K1_MATRIX_CASES=2,3 \
./scripts/run_em_k1_robustness_matrix_slurm.sh --scorecard
```
