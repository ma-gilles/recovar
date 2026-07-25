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
| [ ] | `k1-04` | `c74beea4`; direct-real initial projector + bounded firstiter top-2 tree rescore | fail | pass | 0.992965912 | +0.003915953 | setup 11563826; science 11563827; summary 11563828; strict audit 11563842 |
| [ ] | `k1-04` | `161cb18f`; same intervention + exact RELION 128-add coarse CC and direction-major tie order | fail | fail | 0.992294244 | +0.003751261 | setup 11579502; science 11579503; summary 11579504; strict audit 11579539 |
| [ ] | `k1-05` | `c74beea4`; identical case-4 intervention, frozen-fixture generalization | fail | pass | 0.985721587 | +0.000330734 | setup 11564052; science 11564053; summary 11564054; strict audit 11564062 |
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

The completed exact-128-add arm does not promote case 4.  Science `11579503`
completed `0:0` on H100 `della-h20g5`; summary `11579504` completed `0:0`,
and the fail-closed audit `11579539` correctly exited `1:0`.  The numbered
trajectory remains extremely close through iteration 17 (merged
cross-engine FSC-AUC `0.999664883`), but the independently reconstructed
final merged map reaches only `0.992294244`, below the frozen `0.995`
threshold.  The topology audit also finds one schedule mismatch at iteration
15 (`RELION current_size=154`, `RECOVAR current_size=156`).  RECOVAR remains
better against ground truth at the final merged boundary
(`0.352136260` versus `0.348384999`, delta `+0.003751261`) and runs in
`7,969.70` seconds versus RELION's `16,053` seconds (`2.0143x` speed ratio),
so the rejection is specifically parity shape/topology, not final quality or
performance.  Relative to the prior bounded-tree arm (`0.992965912`), exact
128-add/tie handling changes final cross-engine FSC-AUC by `-0.000671668`;
it is therefore not a production fix.  FSC and topology report SHA-256 values
are `56994ba7e843b0245ca31671d64a60f6fc4ab747d150d6542bfe809ec79f733f`
and `df1aad317d46e15d79a8ece0413cdfb2e533a69e4426e0f14ec5360705667ef1`.
The frozen snapshot remains unchanged.

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
Same-physical-H100 patched-RELION full-grid discriminator `11572062` completed
for particle `5234` with exact identity across all 1,069,056 candidates.
RELION assigns the RECOVAR winner `(32933, 24)` and its own mapped winner
`(33690, 20)` the same float32 normalized-CC score,
`0.27847832441329956`.  RECOVAR instead assigns
`0.27847859263420105` and `0.2784782648086548`, respectively, splitting the
RELION tie by the previously measured `3.278255e-7` and selecting the other
coarse parent.  Across the full centered score grid, absolute difference p95
is `3.688037e-7` and maximum is `1.147389e-6`.  This rejects missing candidate
support at the case-4 first boundary and localizes the exception to
sub-micro score arithmetic/tie preservation.  Analysis JSON SHA-256 is
`1a834a5f9cfdc67899f79485d6c467860f8a170f109555567fdf6169e70d2d12`;
the exact RELION dump manifest SHA-256 is
`d558418952fe8f9a1a791ca8fbca54ca6d0bc7c61e1f4d4081273025efc5b80c`.
An exact FFTW-order operand replay further closes preprocessing: RECOVAR's
positive score weight matches RELION `corr_img` at relative L2 `2.41567e-7`,
and the combined image/CTF/translation operand matches at relative L2
`2.80247e-7` to `2.91223e-7` across all 29 translations.  The two cross-winner
translations are `2.82062e-7` and `2.88089e-7`.  This eliminates image
preprocessing, CTF weighting, translation phase, and window ordering as the
material source, leaving projected-reference generation or score
operand/reduction arithmetic.  Operand-report SHA-256 is
`f7258bfe7ac859b4499d6166ab78b597ad7c5183b333fcbdce6555eb0272530a`.
An independent production-CUDA replay now closes the projected-reference
branch too.  With RELION's persisted `PPref` and its eight fine Euler
matrices, RECOVAR's texture projector is bitwise equal to all 51,968 complex
pixels (eight orientations, 32 hypotheses, 1,624 pixels each): relative L2
and maximum absolute error are both exactly zero.  The result JSON and
analyzer SHA-256 values are
`9cb5f3407b44e137e20d86bb727015d55c7c168935a1b916420f37626059c10e`
and
`d23ff58c162b0fceccdda22125b17e495756c341317472f6b47f36f58cf23f95`.
For identical reference/Euler inputs, projector implementation is therefore
not the case-4 limiter; the remaining first-boundary branch is Euler
construction/handoff or fused normalized-CC arithmetic/reduction.

Synchronized RELION coarse-component job `11577336` and dependent audit
`11577341` close that remaining case-4 branch.  RELION's dumped numerator and
norm are bitwise identical to the RECOVAR hybrid projection/image replay for
both cross-winner hypotheses.  The CUDA source has all 128 threads atomically
add the same reduced score divided by 128; replaying those 128 sequential
float32 additions produces `0.27847832441329956` for both candidates,
bitwise equal to RELION.  RELION's direction-major flat keys are `943626` and
`928339`, so the exact tie selects RECOVAR pose ID `977030`, the observed
RELION winner.  Across all 1,069,056 candidates, the atomic replay has zero
maximum absolute score error; 1,069,055 are bitwise equal and the sole
remaining bit mismatch is positive versus negative zero.

The production fix reproduces that atomic accumulation and tie order only
inside the bounded first-iteration normalized-CC top-two replay.  A frozen
A100 replay of the captured case-4 components selects pose `977030` and is
stored under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_atomic_fix_replay_20260724T225000Z`.
Component-result, analyzer, frozen-replay JSON, and replay-script SHA-256
values are
`65b9a8a74a581eb504a012251dc1a67f046e2abc6b7ef89820104dfeb54ea874`,
`f6a4e991deff3b3cfcd72cba565fd5ee1fa3f35b43528840d4b866888e4a49a8`,
`f5a685c4ec943c81029372cb208fb733ff28fe8fdf0825b8c6378edce0e3f54c`,
and
`a82f8d6ca89b3f7693576a823ba6e8ed45d3bf3140925f92dec4abcbce4ed522`.
This is an exact arithmetic/winner regression, not a correlation acceptance
or a completed autonomous trajectory.  The frozen score remains 25/34 strict,
31/34 topology, and 34/34 evaluated until a fixed-fixture rerun passes the
unchanged FSC/FSC-AUC and topology gates.

Same-physical-H100 fine-pass capture `11572658` completed `0:0` for particles
`38594` and `65070` on the required physical GPU
`GPU-0d7b80c7-fef8-e346-6332-de36ae1af518`.  Both RELION fine candidates are
present in RECOVAR's support.  Particle `38594` selects RELION's candidate
exactly, with a native float32 top-two margin of `1.4901161e-8`.  Particle
`65070` has an exact float32 tie between flat candidates `332` and `333`;
RELION selects `333`, while RECOVAR's first-index `argmax` selects `332`.
The candidates share the same rotation and differ by one fine translation
step (`1.0624999` Angstrom), exactly explaining the remaining reported
translation exception.  Because this capture did not record RELION's two raw
fine scores or compact-candidate order, the remaining discriminator is
fine-score arithmetic versus fine-candidate tie ordering; missing support and
a score-margin threshold are excluded.
Fine-summary JSON SHA-256 is
`019d3111c6eda111080bd2e87a81832971d4128535f2a3718bb7352fd452897f`;
the two captured fine-panel SHA-256 values are
`c024a27a8b2f8071a1015e845ed28a938e6d7b3ece309a8789d07b702fddbeb6`
and
`f4e57638c96361f1040374827342a97866b802276810dca61b2ba21f16bee18d`.
This is localization evidence, not an FSC checkbox.  A passive RELION capture
of the two raw fine costs and their compact indices is required before a
production tie rule is justified.  Exact-physical-H100 discriminator
`11602720` is submitted for that purpose, with fail-closed launcher SHA-256
`94db8675962d37e1ab28cb2a20a95605bec7a31682fedaa4c118aa7d43cbc4b8`.
Before accepting the fine comparison, it requires exact identity-aligned
stock-versus-dump-enabled RELION iteration-1 poses, translations, class, Pmax,
and significant-support counts for all 100,000 particles, then maps the
eight-rotation fine panels by Euler matrices.  Superseded pending jobs
`11602588` and `11602654` were cancelled before execution.
The frozen score remains
25/34 strict, 31/34 topology, and 34/34 evaluated.

## K=4 fixed backend-trajectory baseline

K=4 uses a separate fixed 15-iteration, four-class trajectory denominator; it
does not alter the K=1 score above.  Checked snapshot
`k4-host-ac5177d2-20260719` records:

| Gate | Passing | Total |
|---|---:|---:|
| Direct per-class FSC-AUC at `0.995` | 40 | 60 |
| Iterations passing all four classes | 9 | 15 |
| Exact control topology | 1 | 1 |

The per-iteration direct-pass vector is
`[4,4,4,4,4,4,4,4,4,3,0,1,0,0,0]`; minimum cross-engine FSC-AUC is
`0.9912957080903252`, minimum RECOVAR-minus-RELION GT FSC-AUC is
`-0.0001556260741278903`, and minimum class agreement is `0.9932`.  The
machine-readable baseline is
`docs/math/em_k4_backend_trajectory_baseline_v1.json` (SHA-256
`7ad897000cdbcd0d4342bf5db36a6c56da004a31720e2de02fd8322055d1e41c`).

Same-physical-A100 host-versus-RELION-CUDA science `11600592` and dependent
audit `11600593` are the candidate trajectory for the next snapshot.  As an
early health check, its completed host arm passes 24/24 fixed-host comparisons
through iteration 6 at the same `0.995` gate, with identity class permutation
and minimum FSC-AUC `0.9977914887513855`.  The partial report SHA-256 is
`11f8c708ca4ec71aa43341e3c068de0095c2c36e7c281e1da908b250e06a3d1f`.
This cross-run diagnostic does not promote the candidate; promotion requires
the complete same-GPU backend audit.

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
on the same allocation.  At submission, science and both dependent auditors
were pending; this did not change the frozen K=1 score.

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
evidence, not a full-case pass.  Both jobs subsequently completed; the
terminal strict failure is recorded below.

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

The same-GPU K=4 science job `11565045` completed all 15 configured numbered
iterations without convergence.  It therefore correctly skipped forced final
all-data and persisted the last numbered half-average as the final maps.
Iterations 1--10 used
sizes `38,38,42,56,60,62,68,70,72,74`, resolutions
`60.44,49.45,30.22,27.20,25.90,22.67,21.76,20.92,20.15,19.43` Angstrom,
and iteration-10 Pmax `0.915066`.  This
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
`0.995505295`/`0.995592423`.  Independent array audit `11573422` passes
iterations 9 and 10 with identity assignments and respective minimum
cross-engine FSC-AUC `0.995753485` and `0.995367310`, proving iteration 11 is
the first failing boundary.
Shellwise localization `11573825` shows that the iteration-11 failure is
outside the active reconstruction support: cross-engine FSC-AUC remains
`0.999753711`--`0.999866473` through the reported resolution and
`0.998362350`--`0.998989750` through the current-size radius, while more than
`91.2%` of each class's positive shellwise deficit lies beyond that radius.
The unchanged full-grid gate remains failed; this diagnostic does not promote
K=4 or authorize a threshold change.
Independent fail-closed array audit `11575379` extends the trajectory through
iterations 12--13.  Identity class assignments remain exact and worst GT
deltas are only `-0.000110190`/`-0.000036050`, but minimum classwise
cross-engine FSC-AUC falls to `0.994672195` and `0.992658054`.  Both tasks
therefore exit `2:0` at the unchanged full-grid gate.  This continued
cross-engine divergence does not indicate a GT-quality collapse and does not
promote K=4.
Independent audit `11577443` extends the same gate through iteration 14.
Identity class assignment remains exact; classwise cross-engine FSC-AUC is
`0.994832774,0.993014076,0.992320840,0.995034148`, so the task exits `2:0`.
All GT deltas remain inside the unchanged quality gate, with worst magnitude
only `0.000095231`.  The strict full-grid trajectory therefore remains red
without evidence of GT-quality loss.
Independent audit `11577956` extends the trajectory through terminal numbered
iteration 15.  Identity map assignment remains exact, but all four classwise
cross-engine FSC-AUC values fail:
`0.994459232,0.993069734,0.992039376,0.994497731`.  GT deltas remain small
(`+0.000115765,-0.000113423,+0.000041339,+0.000018044`), so this is continued
trajectory divergence rather than GT-quality collapse.
Same-UUID vector/scalar auditors `11577999`/`11578000` complete `0:0`, but
their 56,720-versus-111 count compared RELION's complete candidate table to
RECOVAR's positive contributors and is superseded.  Corrected contributor
audit `11580683` compares emitted contributors on both sides: RELION has 120
versus RECOVAR's 111, 13 RECOVAR particles have no class-2 contributor, and
no contributor rotation matches within `1e-6`.  That job fixes the earlier
candidate/contributor category error, but its cross-engine support mismatch
is not accepted parity evidence: the RELION arm was restarted from iteration
9 and used sampling perturbation `-0.12306`, while RECOVAR correctly followed
the uninterrupted oracle at `+0.096421`.
Candidate-parent audit `11580995` is rejected and superseded.  It additionally
treated RELION direction-major and RECOVAR psi-major integer indices as one
coordinate system without validating them against the captured matrices.
Independent Slurm audit `11581784` runs the geometry-gated v2 auditor at
commit `0b5182b5`.  It validates both conventions independently
(RELION max-abs `5.0664e-7`, RECOVAR `1.7881e-7`) and then fails closed with
`incomparable_sampling_perturbation_precludes_cross_engine_support_claim`;
the perturbation delta is `0.219481`.  Therefore its former 7,090/6,857/1,677
parent counts and 14/96 contributor-retention claim are descriptive artifacts
of an invalid comparison, not a localization of the K=4 parity gap.  The
rejected v1 report SHA-256 is
`d90f970ddb98c1c31ab9de4c18949fce20e3150581c66d7945b4d4e143bbd508`;
the corrected v2 report SHA-256 is
`077a611d1a6025834316b41d3522efea1d008a3ecbbb0a0f645c3402902e5486`.
Matched-restart replay `11582127` then evaluates RECOVAR at the same
iteration-10 sampling perturbation, `-0.12306`.  Its E/M boundary capture is
complete for all 96 selected particles in 95 shards, but the launcher exits
`1:0` after capture because the optional map diagnostic passes current-size
RELION x-half accumulators to a full-layout reconstruction helper.  The
capture is therefore sealed separately and is not represented as a complete
map replay.  Independent CPU audit `11583809` completes `0:0` in 14 seconds.
The geometry-gated v3 audit finds exact agreement for every particle: 7,090
coarse parents, all 56,720 fine candidates, and all 120 positive
fine-rotation contributors match.  Both matrix gates pass (RELION maximum
absolute error `5.0664e-7`, RECOVAR `0`) and the perturbation delta is exactly
zero.  This rules out coarse selection, oversampled candidate generation, and
rotation-level significance masking at the matched boundary.

The next prescatter comparison matches all 120 contributor rotations and
localizes the first discrete difference to the outer current-size pixel
support.  Only 5/96 particles have exact reached-pixel sets; across 257,589
union entries there are 130 RELION-only and 128 RECOVAR-only pixels, and all
258 one-sided pixels are on shell 37, the exact current-size radius.  On the
common support, median per-particle relative L2 is `7.2088e-7` for complex
data and `3.8432e-7` for real weight.  The v3 support report, prescatter JSON,
and prescatter array SHA-256 values are
`aeb6f14c03da5c44fead5b3e63efd94b9423133cc1af6990910302f46e1fceb0`,
`234bc4871b17dd1cdf1c4eaa7754d56999c63e6b2944c624506e3b69633893b2`,
and
`54670c115c15062756e96adec94a255fcb24c0f2c706879bb7758593cfc75c4c`.
The durable run/runtime roots are
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_matched_restart_boundary_replay_0f5f1404_20260725T012000Z`
and
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k4_it10_matched_restart_boundary_replay_0f5f1404_20260725T012000Z`;
both contain `SAFE_TO_DELETE`.  K=4 remains red and the frozen K=1 score stays
25/34 strict, 31/34 topology, and 34/34 evaluated.

The preceding “matched-restart” interpretation is superseded by a
full-precision perturbation audit.  RELION's continuation from
`run_it009` uses live perturbation `-0.12305957078933716`, reconstructed from
random seed `1778628798` with restart state iteration 9; the sampling STAR
rounds that value to `-0.12306`.  Replay `11582127` and audit `11583809` fed
the rounded value to RECOVAR and compared the two rounded metadata values, so
their zero perturbation delta was not an exact arithmetic match.  The
`4.2921066e-7` perturbation error produces approximately `2e-5` degree Euler
shifts and explains the 258 shell-37 one-sided pixels.

Device substitution `11584294` had already shown that replacing only the
RECOVAR matrices with captured RELION matrices restores all 120 rotation rows,
all 96 particle pixel sets, and zero one-sided pixels.  Matrix-origin probe
`11584445` completes `0:0` in 40 seconds (maximum RSS 836,700 KiB) and rejects
three rounded-input approximations: the closest host path still has 1,009 of
1,080 float32 entries different and maximum absolute error `5.0664e-7`.
Repeating the host-grid generation with the seed-exact restart perturbation
matches all 1,080 captured float32 entries bit-for-bit.  Therefore the
outer-shell support mismatch is a rounded replay-input artifact, not a
RECOVAR matrix-construction or scatter-predicate defect.  The checked replay
harness now defaults to seed-exact perturbations, requires an explicit restart
boundary for continuations, and has a five-matrix frozen bit-pattern
regression.  A new exact-input boundary replay is required before re-accepting
the contributor/prescatter localization.  No frozen case is promoted: the
score remains 25/34 strict, 31/34 topology, and 34/34 evaluated.

Seed-exact boundary replay `11584817` and independent audit `11585023`
supersede that rounded-input result.  The live perturbation is exactly
`-0.12305957078933716`; all 96 coarse-parent sets, all 96 fine-candidate
sets, and all 96 positive-contributor rotation sets match, covering all 120
class-2 contributors.  Both engines' captured matrices match their
independently reconstructed geometry with maximum absolute error `0`, and
all 96 reached-pixel sets now match with zero one-sided pixels.  This closes
candidate topology, contributor topology, matrix construction, and scatter
support at the exact iteration-10 boundary.

The value audit `11586748` then tests whether the remaining prescatter
difference is one scalar posterior-mass normalization per contributor.
Geometry is qualified for all 120/120 contributors and all 257,461 reached
pixels.  Only 109/120 contributors pass the predeclared `1e-5` scalar gate
(`90.8333%`, below the predeclared `95%` causal threshold).  Complex-data
scalar-fit residuals have median `3.55733e-7` and maximum `4.22837e-5`;
weight residuals have median `3.21426e-7` and maximum `6.96009e-7`.
The data/weight scalar relative difference is at most `5.65098e-6`.
The sealed classification is therefore
`pixel_varying_source_difference_not_explained_by_per_rotation_scalar`:
the remaining K=4 boundary mismatch is in a pixel-varying source operand,
not support, scatter geometry, or a single per-rotation posterior scalar.

The durable run/runtime roots are
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_seedexact_restart_boundary_replay_f58a29ae_20260725T011349Z`
and
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k4_it10_seedexact_restart_boundary_replay_f58a29ae_20260725T011349Z`;
both contain `SAFE_TO_DELETE`.  The contributor-support JSON, scalar JSON,
and scalar NPZ SHA-256 values are
`c80906e9afe1e269c30c5e100e358e9a79615fef6df380809e5786e5fbed5075`,
`9009b415e84f1e7771c9fe7d124738d9e2d3e735c7f2877a0211952a9811214e`,
and
`83ab74e6c590087e9cf5e919fe80f0416d7350ac7addc6c33a7637f27a41b9b8`.
The checked replay audit now fails closed on any rounded perturbation.
K=4 remains red, so the frozen score remains 25/34 strict, 31/34 topology,
and 34/34 evaluated.

A full terminal
FSC/class-assignment audit `11578043` completed the complete 15-iteration
trajectory in `01:01:29` with 4,767,900 KiB maximum RSS.  Its expected `2:0`
exit is a scientific gate failure, not a missing-product failure: all 15
numbered boundaries, the non-converged final-map policy, and all 100,000
particle assignments were evaluated.  Iterations 1--10 pass; the earliest
failure is iteration 11 class 2 at cross-engine FSC-AUC `0.994509131`
(class 3 is `0.994150545`).  Terminal classwise cross-engine FSC-AUC is
`0.994459232,0.993069734,0.992039376,0.994497731`.  Identity assignment is
preserved and every GT delta stays inside the unchanged `-0.002` gate.  The
JSON and shellwise NPZ SHA-256 values are
`02615753e8bb20df95673a6aa45fe374111b28aac819115d09d44d429bec2288`
and `c676d662c3a1204d5ff2710d710dd1fd6bd3288169c3d3e49373d2beb008db4e`.
K=4 remains red, and no frozen K=1 score change is claimed.

Native reconstruction substitution localizes that red boundary upstream of
reconstruction.  Audit jobs `11580327`/`11580360` converted the saved
iteration-11 odd `155^3` RECOVAR accumulators into RELION's explicit
`155x155x78` BPref layout, applied the qualified `-256^2`, `256^4`, and
`1/256^4` data/weight/tau2 frame conversions, and called RELION's compiled
`BackProjector::reconstruct` with its Class3D default `skip_gridding=true`.
Native-RELION versus saved-RECOVAR FSC-AUC is
`0.999999953,0.999999944,0.999999945,0.999999935` for classes 1--4.
Substituting those maps changes their cross-engine FSC-AUC by only
`+4.27e-8,+6.83e-8,+3.72e-8,+7.51e-8`; the class-2/class-3 failures remain
`0.994509199`/`0.994150582`.  The non-default iterative-preweight arm
`11580236` is worse (`0.993845903` for class 2) and is rejected.  Therefore
the material iteration-11 difference is already present in accumulated
support/statistics, not RECOVAR's reconstruction or solvent postprocessing.

The seed-exact K=4 factor panel now closes the next upstream boundary without
changing the frozen K=1 score.  Passive RELION factor capture passed formal
control/capture map inertness for all four classes at FSC-AUC
`0.999999992492`--`0.999999995085`.  A100 comparison job `11590986` then
matched 17 particles, 25 contributor rotations, and 53 accepted hypotheses
with bitwise-exact contributor geometry and exact accepted-translation
support.  RECOVAR's factor replay matches its captured production source at
relative L2 `7.34e-8`--`8.29e-8`.

Aggregate RELION/RECOVAR relative L2 is `2.83e-7` for CTF, `3.15e-8` for
inverse noise, `3.16e-8` for translation increments, `8.28e-5` for posterior,
`8.42e-5` for the complex term, `8.42e-5` for the real weight term, and
`4.24e-5` after contributor source summation.  Standalone processed-image and
weighted-CTF residuals are larger (`0.00662` and `0.00648`) but arise from
opposing normalization/correction placement and mostly cancel in the term.
This localizes the first material exact-support difference to posterior
weight arithmetic, not geometry, pixel support, CTF, inverse noise,
translation increments, or scatter.  Comparison JSON SHA-256 is
`e70f404a25c4a43fc768d12a6ee507a61ab9d39e348f527d6d1caffbbe1d590a`;
formal inertness JSON SHA-256 is
`365e85fa249defb07b05f5676462cd4d83811aae59c6b95a585dbfa49ee29fe6`.

The predeclared RELION-posterior counterfactual `11591141` is decisive.
Substituting only the captured RELION posterior on the exact same 53
hypotheses reduces complex-term relative L2 from `8.4172e-5` to
`3.6280e-7`, real-weight relative L2 from `8.4192e-5` to `3.4732e-7`, and
source-sum relative L2 from `4.2365e-5` to `3.6284e-7`.  This removes
`99.9981%`/`99.9983%` of term/weight residual energy and `99.9927%` after
summation.  Posterior construction is therefore the causal remaining factor
boundary. Counterfactual JSON SHA-256 is
`e526fdb5b49f4675393b65512864f772be88580a37f1c1a25a8e08b0621d68d4`.

Posterior decomposition `11591351` moves that boundary one step upstream.
Raw accepted exp(50)-frame weights differ at relative L2 `1.0195e-4`, while
the 17 all-support exp(50)-frame normalizers differ at `7.3824e-5`; both feed
the normalized `8.2828e-5` posterior residual.  RELION raw-log-weight versus
RECOVAR shifted-score absolute differences have median `2.4407e-4`, p95
`4.8832e-4`, and maximum `4.8835e-4`.  This rejects a divide-only
normalization defect and identifies fine score argument/reduction arithmetic
before exponentiation as the next boundary. Decomposition JSON SHA-256 is
`33a6a98d17f3c84ff55c406d4ab49c8d5c337189aa24d668ed14121fccbfea61`.
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
