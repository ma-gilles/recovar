# EM evidence inventory, 2026-09-01

This is the compact coverage index for the EM evidence checked into the pull
request. It distinguishes accepted registry records, supporting calibration
evidence, rejected diagnostics, and runnable but unfinished gates. It indexes
completed artifacts through the sign-fixed EMPIAR-10076 three-seed
independent-half diagnostic at RECOVAR commit `7136e5c8d`; live or later runs
must be sealed separately before this inventory can promote them.

The machine registry currently contains six single-run entries, nine campaign
records, and one synthetic negative diagnostic. The exact filenames are pinned
by `tests/unit/test_validate_em_benchmark_registry.py`, while
`scripts/validate_em_benchmark_registry.py` validates every accepted or
synthetic-negative JSON and fails closed on an unknown diagnostic family.
Eight real-data diagnostic schemas are deliberately routed outside the
accepted registry and are never counted as accepted results. The InitialModel,
offset-prior, and historical selected-fine ledgers have dedicated whole-ledger
validators.

## Accepted synthetic K=4 single-run entries

All six entries retain source and executable provenance, input and artifact
hashes, exact commands, Slurm accounting, permutation-aware class quality,
class populations, and measured performance. A `SCIENCE_EQUIVALENT` result
does not erase a frozen formal failure.

| Record | Workload | Formal / science | Performance disposition |
| --- | --- | --- | --- |
| `k4-ribosembly-100k256-ac5177d2-a100` | Ribosembly, 100k, box 256, C1, 15 iterations | FAIL / PASS | Same-A100 wall and RSS are formal; RELION HBM was not sampled and is explicitly null. |
| `k4-ribosembly-100k256-1b9209cd8-h100` | Same frozen 100k fixture on the candidate | FAIL / PASS | H100 RECOVAR values are retained, but the RELION comparator is the A100 oracle; the cross-model speed value is diagnostic only. |
| `k4-ribosembly-10k128-white1-uniform-0050dc54f-h100` | Ribosembly, white noise, uniform | PASS / PASS | Matched same-H100 wall/HBM; RECOVAR 1199 s / 17087 MiB, RELION 122 s / 79563 MiB. |
| `k4-ribosembly-10k128-radial3-nonuniform-linear-0050dc54f-h100` | Ribosembly, radial noise, nonuniform poses, linear weights | PASS / PASS | Matched same-H100 wall/HBM; RECOVAR 1621 s / 33495 MiB, RELION 143 s / 79559 MiB. |
| `k4-ribosembly-10k128-radial3-nonuniform-outliers20-0050dc54f-h100` | Previous case with 20% outliers | FAIL / PASS | Matched same-H100 wall/HBM; RECOVAR 1674 s / 33495 MiB, RELION 143 s / 79559 MiB. |
| `k4-igg-10k128-white1-uniform-0050dc54f-h100` | IgG-1D, white noise, uniform | FAIL / PASS | Matched same-H100 wall/HBM; RECOVAR 1343 s / 17087 MiB, RELION 117 s / 79561 MiB. Both engines have the recorded near-collapse. |

The historical 100k records explicitly retain missing common-mask half-map
FSC and missing RELION HBM rather than treating either as zero. The four 10k
pilots stop at a nonconverged five-iteration cap and are not substitutes for
the 100k/256 completion workload.

## Accepted synthetic K=4 campaigns

Campaign JSON stores one row per case rather than only an aggregate. Each row
retains its generator/configuration, source inputs, final class quality,
occupancy, wall/HBM, and Slurm ReqTRES/AllocTRES.

| Campaign | Cases | Frozen / science outcomes | Performance coverage |
| --- | ---: | --- | --- |
| `k2-ribosembly-three-seed-0d85b576b-h100` | 3 | 3 PASS / 3 PASS | Three matched-H100 wall/HBM pairs. |
| `k8-ribosembly-three-seed-0d85b576b-h100` | 3 | 3 FAIL / 3 PASS | Three matched-H100 wall/HBM pairs; strict failures preserve small map/assignment drift. |
| `k16-ribosembly-three-seed-0d85b576b-h100` | 3 | 3 FAIL / 3 PASS | Three matched-H100 wall/HBM pairs; all direct/GT class cells pass and strict failures are assignment-only. |
| `k4-expanded14-3466e7a32-h100` | 14 | 6 PASS, 7 FAIL, 1 NOT_EVALUABLE / 11 PASS, 2 BOUNDARY, 1 UNRESOLVED | Both wall/HBM values are present for all 14 same-H100 cases, with limitations retained for non-evaluable endpoints. |
| `k4-exact-input-invariance-91e8a30f4-h100` | 9 | 3 PASS, 6 FAIL / 3 PASS, 6 UNRESOLVED | All nine rows retain timing/HBM, but only the three producer rows admit a matched-hardware comparison; consumers reuse the sealed RELION oracle. |
| `k4-c4-three-seed-c75cbfffc-h100` | 3 | 3 PASS / 3 PASS | Three matched-H100 wall/HBM pairs. |
| `k4-d4-three-seed-c75cbfffc-h100` | 3 | 3 PASS / 3 PASS | Three matched-H100 wall/HBM pairs. |
| `k4-o-three-seed-22efd8065-h100` | 3 | 3 PASS / 3 PASS | Three matched-H100 wall/HBM pairs. |
| `k4-i1-three-seed-22efd8065-h100` | 3 | 3 PASS / 3 PASS | Three matched-H100 wall/HBM pairs. |

The K=2/K=8/K=16 extension is documented in
`kclass_k2_k8_k16_multiseed_20260902.md`. The C4, D4, O, and I1 records are
the checked-in multi-seed rotational-
symmetry evidence. The no-CTF cases 30, 35, and 36 are not positive campaign
results: all nine RELION replicates had a zero-mass class before RECOVAR ran.
Their sealed RELION-only provenance and performance are retained in
`diagnostics/k4-noctf-collapse-cases30-35-36-h100.json` with
`EXCLUDED_FROM_ACCEPTED_RESULTS` disposition.

## K=1 evidence

### Synthetic

K=1 has extensive unit, fast-parity, long-run, and historical trajectory
coverage, but there is no current-source K=1 single-run or campaign JSON under
this schema-v1 registry. `k1_box800_memory_qualification_20260901.md` is a
qualified memory-boundary report, not a full-dataset resolution or matched
cross-engine quality/performance result. It now also records fresh local
score-only job `13363559`: the compact K=1 planner replaced the false
81.92-GB persistent estimate with 4.02 GB, completed on one H100 with a
38,059-MiB sampled peak, and released the separate full-dataset job
`13363818`.  Commit `f91c73f29` then removed an unused 31.25-GiB padded mean
from the supplied-projector local path.  Matched job `13366512` preserved all
audited plans and support counts while reducing peak HBM to 20,123 MiB and
wall from 650.344 to 124.897 s.  Full-particle checkpoint-seeded job
`13368258` then crossed both local passes for all 15,258 particles in one half
at `current_size=414`, completing in 512.723 s with a 20,125-MiB whole-job and
15,305-MiB local-window peak.  It is a score-only execution confirmation, so a
future K=1 completion record must be added rather than inferring registry
admission from any memory gate.  Commit `4749f6ad9` additionally donates the
consumed exact-local x-half accumulators.  Matched `current_size=498` jobs
`13369477/13369478` preserve the checked numerical values and pointer ownership
contract while reducing HBM from 11,991 to 6,279 MiB.  The 5,712-MiB reduction
matches the exact 5,710.607529-MiB numerator/weight pair to sampling
resolution; this is a memory-path gate, not a final reconstruction.

The associated full-particle compact-planner job `13363818` subsequently
reproduced iteration 11 at 3.46 A and reduced the size-564 fine loop to 1,263
chunks, but still failed naturally after 1,000 chunks in a transient RELION
texture allocation.  Peak sampled HBM was 79,259 MiB; host cgroup memory
events remained zero.  This is retained as a negative full-scale boundary,
not an accepted completion.  It defines the boundary used by the subsequent
persistent-texture lifetime and arithmetic gates below.

Persistent-texture commit `530eba051` now passes the exact box-800,
current-size-564 allocation/lifetime gate. Nonexclusive H100 job `13379238`
runs both a normal 256-rotation compiled bucket and a forced 512-rotation split
bucket against one shared `(1131,1131,566)` complex64 PPref owner, updates the
full x-half accumulators, closes the owner, and completes `0:0`. Its 100-ms HBM
peak is 39,033 MiB, 40,226 MiB below the old failure sample. This admits the
memory boundary, not final trajectory quality. Parent commit `737018067`
separately removes a 15.258789-GiB box-800 host snapshot copy while preserving
device offload and non-owning-view isolation.

Independent nonexclusive H100 job `13379318` supplies the same full-size PPref
with 2,556,626 deterministic nonzero voxels and seven nonidentity rotations.
Persistent dynamic-handle and historical transient-texture projections are
bitwise identical (`max_abs_diff=0.0`, common output SHA-256
`dd30338fbbe1a02b10003f95cf5e19a4ac407c4195f1d03c6c54835d346593b4`).
It requested and received exactly one H100, four CPUs, and 80 GiB, completed
`0:0` in 19 s, and is retained at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k1_persistent_texture_arithmetic_20260903`.

At the clean integrated run's numbered checkpoint 1, commit `37f640c7e`
reproduces the corrected no-padding run's controller and next-size decision,
all 30,515 coarse and fine particle assignments, rotation/translation grids,
complete saved particle state, and FSC curve bit-for-bit. The maps are not
bitwise equal: their relative-L2 differences are `7.96e-8` and `7.91e-8`,
with a maximum observed floating-array difference of `1.03e-7`. This is exact
discrete/controller/FSC equivalence with very small floating drift, not a
strict floating-equivalence claim. Iteration wall time falls from 999.8 to
795.6 s (20.42%) at essentially unchanged sampled peak HBM. The hardened
semantic JSON and Markdown are sealed at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/live_run_monitor/empiar10202_set6_checkpoint1_corrected_vs_ptex_20260903T0450`
with SHA-256
`432277018307aedd8bb1ff27fdad50c690f95f1dea8237b2d21619c5220a99ff`
and
`a259db52a9634d71cf6b879af7ff6691cc8fff02c06432b77309f23a1706ea1b`.

The same comparison is no longer exact at checkpoint 2: fine assignments
differ for 4/15,258 and 18/15,257 particles, while the controller path and
FSC crossings remain equal. Half-map relative-L2 drift is
`1.17e-5/3.99e-5`, correlations remain above `0.9999999991`, and FSC RMSE is
`2.16e-5`. The checked wording is therefore sparse/science-small drift, not
strict equivalence. Its semantic JSON and manifest are retained at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/live_run_monitor/empiar10202_set6_checkpoint2_corrected_vs_ptex_20260903T0509`
with SHA-256
`a0194ef444fc9f8e6809c959a3a1b7cfd746f009107d87ff730370e1eaa2b747`
and
`397247a65cd97789db0d31506f40a4fa373a0d48cf32c00f49eb7651d8955f22`.

Corrected full-particle job `13376414` subsequently completes the historical
box-800/current-size-564 failure boundary in both halves. Numbered iteration
12 saves both maps and particle-state archives after `1266/1266` and
`1263/1263` fine chunks, including every 512-rotation split bucket, and peaks
at 47,275 MiB sampled HBM versus 79,259 MiB before the old failure. Its
numbered iteration-11 maps also pass the direct shared-frame RELION scorecard:
both half-map FSC curves cross 0.143 at shell 250 (`2.5215997696 A`), merged
cross-engine FSC-AUC is `0.9911398`, half-map cross-engine AUC is
`0.9868918/0.9869988`, and half-FSC RMSE is `0.0050346`. No fitted operation
is used. This is an interim matched-iteration high-resolution result because
the trajectory is still advancing, not a final convergence record. CPU audit
job `13380196` and its exact-resource provenance are retained at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/live_run_monitor/empiar10202_set6_it11_corrected_direct_fsc_20260903T0501`;
the result JSON and verified manifest SHA-256 values are
`a89bf4b106e2f6943e3ea001e022afd910f4fe8d6d6d371bbace74eb3b1936f2`
and
`ae7b2157bc951fb31dc472b396828f933638bf0d62edddc75174052a94c0219d`.

The supporting publication-mask replication uses the same corrected maps,
the sealed mask, and the same RELION phase-randomization policy and seed as
the matched control. Both engines cross corrected-masked FSC 0.143 at shell
248 (`2.5419352516 A`); resolved-band AUC differs by `0.0009232` and RMSE is
`0.0119231`. It is explicitly not an acceptance or rescue metric. Exact CPU
job `13380452` and the command, mask, input, executable, environment, and
accounting seals are retained at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/live_run_monitor/empiar10202_set6_it11_corrected_masked_fsc_20260903T0518`.
The result JSON and validated manifest SHA-256 values are
`6799bfefff1ec370828ec292be0dd69d799df83dda7eda3ed10e0f80566bd7a2`
and
`a748c022552f5b3e9bd34db2dd14a91dc37143097a074774024440006c78d72d`.

The same corrected run reaches a stronger matched numbered-iteration-14
checkpoint: both RECOVAR and RELION cross direct unmasked half-map FSC 0.143
at shell 262 (`2.4061066504 A`). Resolved-band direct cross-engine FSC-AUC is
`0.9875827` merged and `0.9816448/0.9830144` for halves 1/2; half-FSC RMSE is
`0.0059807`, and all primary gates pass without a fitted operation. This is
still intermediate evidence, not the final-convergence record. CPU job
`13380640`, the exact allocation and inputs, and a verified manifest are
sealed at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/live_run_monitor/empiar10202_set6_it14_corrected_direct_fsc_20260903T0529`;
the result JSON and manifest SHA-256 values are
`eaae20795685282f13814a9ee9ccbeb41ec9c73a7eb52a0b57201f921c90881b`
and
`5b141e0e92265881cd36744d9274c128c341ea9b02f29b4c0465843b70cf6fe9`.

The same iteration-14 maps cross the matched corrected-masked FSC threshold at
shell 258 (`2.4434106295 A`) for both engines under the unchanged publication
mask, phase-randomization threshold, and seed. Masked resolved-band AUC differs
by `0.0018079` and RMSE is `0.0071828`; this is supporting-only and cannot
replace the direct unmasked result. CPU job `13380912` and its verified
manifest are retained at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/live_run_monitor/empiar10202_set6_it14_corrected_masked_fsc_20260903T0546`.
The result JSON and manifest SHA-256 values are
`04bdfb5fe7eee859dedb8bb87f38fecd12277a5d339c0d06ad7596c4f5b5fc3e`
and
`9962a2000bf8f86e63c01131e5174aae7ef5a061f2f9c7e9a967d4afcaeddf89`.

The corrected trajectory then reaches matched numbered iteration 16. Both
RECOVAR and RELION cross direct unmasked half-map FSC 0.143 at shell 281
(`2.2434161651 A`). Resolved shells 1--280 have merged/half-1/half-2
cross-engine FSC-AUC `0.9858187/0.9796343/0.9803080`; the half-FSC RMSE is
`0.0051946`, the half-band AUC difference is `0.0005905`, and all primary
gates pass without a fitted operation. This is still an intermediate
checkpoint. CPU job `13381233` used exact
`ReqTRES=AllocTRES=cpu=8,mem=80G,node=1,billing=20`; its result JSON and
verified manifest SHA-256 values are
`c78932d1de8c6d0295820570c3e571365841e80178cabb9bbdd84e0f3106343c`
and
`bdb19f45bbf1ce1cea3382855184893fb509ee196315e23fa34e73c9e1f8229e`.
The complete record is retained at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/root/empiar10202_set6_it16_corrected_direct_fsc_20260903T100833Z`.

The corresponding iteration-16 masked-support check uses the exact sealed
publication mask and phase-randomization policy. Its corrected-masked curves
first remain below FSC 0.143 at shell 280 for RECOVAR (`2.251428 A`) and shell
277 for RELION (`2.275812 A`); over common shells 1--276, masked AUC differs
by `0.0008457` and RMSE is `0.0053812`. The raw curves reproduce the direct
shell-281 crossing. Masked FSC remains supporting only. CPU job `13381344`
used exact `ReqTRES=AllocTRES=cpu=4,mem=64G,node=1,billing=16`; its comparison
JSON and verified manifest SHA-256 are
`f72404609a235f0cb2a09df9d0bae3dc4e23d87d907944892e7ea9e9ec8abaa7`
and
`97fe931cd7fa13d0da05861059b5053723c14dce10a11e812680227757f260f0`.
The complete record is retained at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/root/empiar10202_set6_it16_corrected_masked_fsc_20260903T101521Z`.

At corrected full-particle checkpoint `it006` (numbered iteration 7), the
release, compact, and no-padding trajectories retain identical saved FSC
crossings (0.5 at shell 150; 0.143 at shell 183). Corrected-versus-release map
correlations are 0.9999608/0.9999576 and relative L2 differences are
0.00884/0.00919, while pose-assignment agreement is 97.80%/97.70%. This is
scientifically close map/FSC evidence but a strict execution-equivalence
failure. The compact three-way record is sealed at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/live_run_monitor/empiar10202_set6_checkpoint7_threeway_20260903T0435`;
its JSON and Markdown SHA-256 values are
`4dc0a318e2f3e332306fb52de28f63cabd472cbda3caaa9f7659d5655b64fc92`
and `a6427feb0c12db158b2384f4e3cb2846b57243279a82c8fa86359c498b6ded75`.

The K=4 compact-pair threshold campaign is closed under a separately frozen
two-tier contract. Three matched default-512/threshold-128 seed pairs preserve
every controller decision and all 120,000 saved class assignments while
reducing sparse-group wall time by 12.77--17.23% without sampled-HBM growth.
The formal tier rejects all three pairs, and the scientific-equivalence tier
rejects one of three because seed 42002 changes one iteration-8 pose by a
47.37-degree physical rotation. The aggregate rule is an all-pair conjunction,
with no averaging. The production default therefore remains 512. The compact
ledger and its dedicated fail-closed validator retain the formal and science
outcomes separately.

`k1_empiar10202_checkpoint_equivalence_20260902.md` records the first five
matching checkpoints from the full-particle box-800 I1 control and compact
candidate.  Checkpoint 0 is execution-exact.  At checkpoints 1 through 4,
strict execution equivalence rejects small accumulated pose/state differences,
while both FSC thresholds still cross at identical shells.  At checkpoint 3,
minimum pose agreement is `0.9971818`; at checkpoint 4 it is `0.9941666`, map
correlation is `0.9999975`, and FSC-curve RMSE is `3.95167e-4`.  Both FSC
threshold crossings remain identical through checkpoint 4.
This is in-progress same-engine trajectory evidence, not a final
RECOVAR-versus-RELION admission.

### Real particles

`docs/math/em_k1_realdata_science_equivalence_scorecard_v1.json` is the frozen
quality scorecard. It hashes the FSC artifacts and producer/collector
references for EMPIAR-10073, 10345, and 10097 and fixes the EMPIAR-10202 set-6
I1 contract. These calibration rows are supporting evidence, not entries in
the PR scoring denominator.

| Dataset | Frozen unmasked result | Corrected masked 0.143 resolution, RECOVAR / RELION | Registry/performance status |
| --- | --- | --- | --- |
| EMPIAR-10073 | Within-engine half-map calibration PASS; raw cross-engine route qualified | 4.156 A / 4.092 A | Quality artifacts and rerun commands are sealed; uniform source-tree, wall, HBM, and MaxRSS fields are incomplete. |
| EMPIAR-10345 | Within-engine half-map calibration PASS; raw cross-engine route qualified | 5.240 A / 5.240 A | Quality artifacts and rerun commands are sealed; uniform source-tree, wall, HBM, and MaxRSS fields are incomplete. |
| EMPIAR-10097 | Within-engine half-map calibration PASS; raw and allowed proper-rigid cross-engine routes remain unqualified | 5.782 A / 5.684 A | Quality artifacts and rerun commands are sealed; uniform source-tree, wall, HBM, and MaxRSS fields are incomplete. Masking does not rescue the frozen cross-engine failure. |
| EMPIAR-10202 set 6, I1 | RELION complete at 2.511554 A unmasked; RECOVAR pending at the frozen subject commit | RECOVAR pending / RELION 2.122559 A | Partial RELION jobs and hashes are sealed. No two-engine final quality or performance record exists yet. |

The common-mask artifacts for 10073, 10345, and 10097 are now complete and
hashed. They remain supporting-only because each mask was derived from the
RELION merged map. The reason these cases are not registry entries is no longer
missing mask provenance; it is the incomplete uniform producer/source/Slurm
and performance envelope, plus the frozen pending 10202 RECOVAR arm.

## Real K=4 diagnostics

No real-data K=4 result is admitted. The checked evidence is nevertheless
repeatable and records useful negative performance without turning a failed
quality pair into a formal speed comparison.

| Evidence | Scope | Quality disposition | Performance disposition |
| --- | --- | --- | --- |
| `diagnostics/real-kclass-initialmodel-20260901.json` | Three 10k-particle, eight-iteration C1 pairs over EMPIAR-10076 and 10345 | All three fail FSC/assignment; two also retain the class-2 collapse | Per-engine wall/HBM/RSS retained as diagnostics; formal ratios are null. |
| `diagnostics/real-kclass-offset-prior-fullpairs-92438c285-20260901.json` | Two clean-source same-H100 pairs, one per dataset | Exact raw iteration-1 labels but failing post-update maps; the later shared-200 audit supersedes the earlier M-step-first interpretation by proving a pre-reconstruction candidate/score gap | Native wall/HBM/RSS retained; raw 31.59x and 26.90x RECOVAR/RELION wall ratios are diagnostic only. |
| `diagnostics/real-kclass-selected-fine-10076-20260901.json` | Two-particle iteration-1 selected-fine capture made before `d1f2f9f93` | Historical support/prior discriminator only. Its large translation-prior delta is from the subsequently fixed InitialModel offset-prior arithmetic and is not a current-source causal boundary. | No final performance claim. A dedicated validator freezes the strict claim boundary and compact artifact hashes. |
| `real_k4_shared200_causal_replay.md` | Frozen 200-particle, four-class iteration-1 replay; source `24317e40c`, job 13322235 | Strict gate FAIL, but 199/200 hard assignments agree and correctly framed per-class map FSC-AUC is 0.604--0.907. Candidate topology/raw scores already differ before reconstruction. | Six native arms plus one RECOVAR arm are bundled, so the 469 s allocation is causal evidence, not a formal speed ratio. |
| `real_k4_native_coarse_score_boundary_20260902.md` and `diagnostics/real-k4-native-coarse-score-a32cccccb-20260902.json` | Native RELION and RECOVAR coarse surfaces for 16 frozen shared-200 probes, 1,069,056 candidates; job 13330906 | Diagnostic PASS: priors are excluded causally and the first material support difference is the raw likelihood/`diff2` surface. This is not a final K=4 quality admission. | Capture/control replay used one H100 for 35 s; timing is instrumentation qualification, not an engine speed comparison. |
| `real_k4_native_coarse_component_boundary_20260902.md` and `diagnostics/real-k4-native-coarse-components-8f9ebc9-20260902.json` | Paired native norm/cross components and RECOVAR full component surfaces for the same 16 probes; jobs 13332998 and 13332392 | Diagnostic PASS: a native cross-only swap restores 11/16 exact supports versus 5/16 for RECOVAR and 5/16 for a native-norm-only swap. The cross term is the dominant first likelihood component boundary. | One-H100 capture/replay timing is diagnostic only. The earlier in-kernel observer is retained as rejected evidence because it changed production results. |
| `real_k4_native_coarse_operand_boundary_20260902.md` and `diagnostics/real-k4-native-coarse-operands-b061776-20260902.json` | Historical three-repeat operand localization plus current-source rounded-shell/Euler replay; jobs 13334583--13334585, 13336787, 13336962, and 13337519 | Diagnostic PASS and bounded boundary closure: native projected references localized the old defect; commit `5f74755c2` then yields zero projected-reference/Euler error and exact support for 16/16 probes and 1,336/1,336 selected candidates. This is not a final K=4 refinement admission. | Replay state is bitwise repeatable; eight GPU-atomic maps have minimum FSC-AUC `0.999999997335` and maximum relative L2 `1.57406e-7`. Timings remain diagnostic, not an engine speed comparison. Job 13334206 remains rejected because synchronous capture perturbed production results. |
| `real_kclass_halfmap_refinement.md`, first-iteration native score boundary | Passive native RELION coarse/fine scores and RECOVAR surfaces for 16 frozen EMPIAR-10076 particles; 1,069,056 coarse candidates; H100 job 13346151 | Diagnostic PASS: commit `4a91369a3` preserves only `run_it000` origins during the fresh Class3D global search. Winner classes, global poses, all per-class poses, fine parents/winners, and integer pre-shifts are exact across the panel. This closes the first-iteration score boundary, not final K=4 quality. | RECOVAR wall 80.4 s and peak HBM 33,423 MiB for the containing 5,000-particle iteration. Full-surface minimum correlation `0.999999999996`; worst fine-score absolute error `7.45e-8`. Sealed JSON SHA-256 `977d115f1593`. |
| `diagnostics/real-k4-native-signfix-causal-7136e5c8d-20260902.json` | Seed-42001 two-iteration EMPIAR-10076 half-1 causal A/B after all three controls independently logged the same class-3 sign flip | Causal PASS: commit `7136e5c8d` restores class-3 iteration-1 FSC-AUC from -0.990489 to +0.990489 and iteration-2 occupancy from 0.0012 to 0.0368 versus RELION 0.0382. Iteration-2 class agreement is 0.942, so this is not a final K=4 admission. | H100 job 13348468 completed in 9m22s, exit 0, exact one-GPU allocation, peak HBM 33,465 MiB. Full JSON SHA-256 `25f3a772fc31`. |
| `diagnostics/real-k4-pilot10k-multiseed-stability-7136e5c8d-20260902.json` | Three seeds, two immutable 5,000-particle halves per seed, eight K=4 iterations per engine | All three prospective gates remain rejected. Median paired masked half-map FSC-AUC delta is -0.00054, but seed-42001 class 3 is -0.11925. Same-seed cross-engine labels and final maps are substantially closer than either engine is across seeds; for every class, same-seed map FSC-AUC minimum 0.8352--0.9527 exceeds within-engine cross-seed maximum 0.6440--0.8782. A cross-CPU replay retains every discrete decision with maximum FSC-AUC drift 0.003631 and minimum separation margin 0.07447 under the checked 0.005/0.05 semantic contract. This establishes a seed-sensitive local-optimum boundary without rescuing the failures. | Same-H100 serial measurements across six halves: RECOVAR median 1,280.24 s / 33,478 MiB; RELION 380.76 s / 79,588 MiB. Map analysis job 13353269 and semantic replay job 13354170 completed on exact 4-CPU/64-GiB resources. Failed job 13353671 is retained as an overstrict byte-comparison harness failure after complete analysis, not a science failure. |
| `diagnostics/real-k4-10073-native10k-fixed8-seed42001-827aedd66-20260903.json` | EMPIAR-10073 native-grid Tier-A calibration, seed 42001, two immutable 5,000-particle halves, eight K=4 iterations per engine | Complete early-trajectory rejection: all engines and audits completed, no class collapsed, and every same-half cross-engine FSC-AUC is at least 0.94756, but three merged-map, two class-2 half-map, and both assignment gates reject. RECOVAR still had 93.24%/97.16% assignments changing, so this is not a final K=4 verdict. | Same-H100 serial walls are 144.26/145.27 s for RELION and 1,571.56/1,656.84 s for RECOVAR; peaks are 79,591/79,593 and 33,493/33,497 MiB. All four engine runs plus setup consumed 1.1039 H100-hours. |
| `k4_validation_matrix.md`, exact-local CUDA x-half gate | Three-image, four-class direct exact-local invariant plus numbered-wrapper seam test; H100 job `13366865`, source `f91c73f29`, fix `43a924358` | Direct PASS: every jointly normalized K=4 class accumulator matches its independent K=1 replay; worst relative L2 is `4.18270e-8` for `Ft_ctf` and `1.36490e-9` for `Ft_y`. The gate exposed and the fix closes missing class-projector/`r_max` forwarding in numbered local refinement. This is execution evidence, not a final real-data K=4 admission. | One H100, four CPUs, 32 GiB host memory, 18 s, exact requested/allocated resources. Result SHA-256 `be52382cc1b`. |
| `k4_validation_matrix.md`, supplied-projector local partition gate | Three-image, four-class real local engine with distinct RELION projectors, irregular supports, nonuniform priors, two translations, corrections/pre-shifts, and `(3,8)` versus `(1,1)` image/rotation partitioning; commit `b4cc56162` | CPU PASS: all discrete outputs are exact, every class retains posterior mass, every class reports the RELION projector path, and f32/fp64 comparisons satisfy their frozen tolerances. This is execution evidence, not a final real-data K=4 admission. | Focused test 31.62 s; three-test sibling set 32.54 s. CPU timing is test cost, not an engine performance comparison. |
| `real_kclass_halfmap_refinement.md` | Independent-half refinement launcher/runbook and full three-seed interpretation | Infrastructure complete for EMPIAR-10076; no accepted completed pair | Native-grid execution remains blocked by the rejected pilot gate. |

InitialModel emits one class map rather than independently refined half maps,
so those precursor diagnostics cannot support final resolution. The completed
independent-half pilot supplies Hungarian per-class FSC, populations, a common
mask, seed stability, and same-hardware performance, but it fails the frozen
quality and assignment thresholds. It therefore remains outside real K=4
admission.

## Remaining coverage gaps

1. Seal at least one current-source K=1 synthetic completion record with final
   FSC/FSC-AUC and matched performance.
2. Convert each finished K=1 real calibration into a separate registry entry
   only after source tree, exact producer commands, Slurm allocation, wall,
   HBM, and MaxRSS are complete. Keep 10097's cross-engine route unqualified
   unless a new frozen gate passes.
3. Finish and seal the RECOVAR EMPIAR-10202 arm before making any target-grid
   high-resolution or two-engine performance claim.
4. The real K=4 firstiter score boundary and post-reconstruction sign boundary
   are repaired at commits `4a91369a3` and `7136e5c8d`. The three-seed
   independent-half rerun is complete but rejected. Particle-state trajectory
   and final-map cross-seed discriminators show that both engines enter shared
   seed-sensitive local optima; the weakest same-seed map match still exceeds
   the strongest within-engine cross-seed match for every class. The map result
   also passes a cross-CPU semantic reproduction contract that freezes all
   discrete decisions while allowing at most 0.005 FSC-AUC fit drift. Retain the
   seed-42001 weak-class outlier and frozen gate failures, but do not spend more
   pilot effort treating it as a deterministic RECOVAR-only defect. Do not
   promote the result to native-grid acceptance.
5. K=2, K=8, and K=16 now have accepted three-seed schema-v1 synthetic
   campaigns. Real-particle K>1 acceptance remains open; do not generalize
   synthetic science equivalence to real-data class recovery.

## Validation and replay

The ordinary checks do not rehash large external stacks:

```bash
pixi run python scripts/validate_em_benchmark_registry.py
pixi run pytest tests/unit/test_validate_em_benchmark_registry.py
pixi run python scripts/validate_em_real_kclass_diagnostics.py
pixi run pytest tests/unit/initial_model/test_validate_em_real_kclass_diagnostics.py
pixi run python scripts/validate_em_real_kclass_offset_prior_fullpairs.py
pixi run pytest tests/unit/initial_model/test_validate_em_real_kclass_offset_prior_fullpairs.py
pixi run python scripts/validate_em_real_kclass_selected_fine.py
pixi run pytest tests/unit/initial_model/test_validate_em_real_kclass_selected_fine.py
pixi run python scripts/summarize_em_k1_realdata_science_equivalence.py --verify-calibrations --verify-masked-support --verify-target-partial --check-markdown
```

Use each validator's explicit `--verify-files` mode only when resealing or
re-auditing external evidence. No command in this inventory submits Slurm.

## Historical baseline left unchanged

`tests/baselines/em_parity_completion_k4_100k256.json` remains the historical
correlation-based completion guard from job 8290126. It was not rewritten with
new FSC evidence. Current scientific and performance claims belong in the
versioned registry and dedicated diagnostic ledgers above.
