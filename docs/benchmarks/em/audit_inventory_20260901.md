# EM evidence inventory, 2026-09-01

This is the compact coverage index for the EM evidence checked into the pull
request. It distinguishes accepted registry records, supporting calibration
evidence, rejected diagnostics, and runnable but unfinished gates. It indexes
completed artifacts through the sign-fixed EMPIAR-10076 three-seed
independent-half diagnostic at RECOVAR commit `7136e5c8d`; live or later runs
must be sealed separately before this inventory can promote them.

The machine registry currently contains six single-run entries, six campaign
records, and one synthetic negative diagnostic. The exact filenames are pinned
by `tests/unit/test_validate_em_benchmark_registry.py`, while
`scripts/validate_em_benchmark_registry.py` validates every accepted or
synthetic-negative JSON and fails closed on an unknown diagnostic family.
Eight real-data diagnostic schemas are deliberately routed outside the
accepted registry and are never counted as accepted results. The InitialModel
and offset-prior ledgers have dedicated whole-ledger validators; the
selected-fine record's missing equivalent validator is listed as an open
evidence gap below.

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
| `k4-expanded14-3466e7a32-h100` | 14 | 6 PASS, 7 FAIL, 1 NOT_EVALUABLE / 11 PASS, 2 BOUNDARY, 1 UNRESOLVED | Both wall/HBM values are present for all 14 same-H100 cases, with limitations retained for non-evaluable endpoints. |
| `k4-exact-input-invariance-91e8a30f4-h100` | 9 | 3 PASS, 6 FAIL / 3 PASS, 6 UNRESOLVED | All nine rows retain timing/HBM, but only the three producer rows admit a matched-hardware comparison; consumers reuse the sealed RELION oracle. |
| `k4-c4-three-seed-c75cbfffc-h100` | 3 | 3 PASS / 3 PASS | Three matched-H100 wall/HBM pairs. |
| `k4-d4-three-seed-c75cbfffc-h100` | 3 | 3 PASS / 3 PASS | Three matched-H100 wall/HBM pairs. |
| `k4-o-three-seed-22efd8065-h100` | 3 | 3 PASS / 3 PASS | Three matched-H100 wall/HBM pairs. |
| `k4-i1-three-seed-22efd8065-h100` | 3 | 3 PASS / 3 PASS | Three matched-H100 wall/HBM pairs. |

The C4, D4, O, and I1 records are the checked-in multi-seed rotational-
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
cross-engine quality/performance result. A future K=1 completion record must
be added rather than inferring registry admission from those tests.

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
| `diagnostics/real-kclass-selected-fine-10076-20260901.json` | Two-particle iteration-1 selected-fine capture made before `d1f2f9f93` | Historical support/prior discriminator only. Its large translation-prior delta is from the subsequently fixed InitialModel offset-prior arithmetic and is not a current-source causal boundary. | No final performance claim. This routed record does not yet have a standalone whole-ledger validator. |
| `real_k4_shared200_causal_replay.md` | Frozen 200-particle, four-class iteration-1 replay; source `24317e40c`, job 13322235 | Strict gate FAIL, but 199/200 hard assignments agree and correctly framed per-class map FSC-AUC is 0.604--0.907. Candidate topology/raw scores already differ before reconstruction. | Six native arms plus one RECOVAR arm are bundled, so the 469 s allocation is causal evidence, not a formal speed ratio. |
| `real_k4_native_coarse_score_boundary_20260902.md` and `diagnostics/real-k4-native-coarse-score-a32cccccb-20260902.json` | Native RELION and RECOVAR coarse surfaces for 16 frozen shared-200 probes, 1,069,056 candidates; job 13330906 | Diagnostic PASS: priors are excluded causally and the first material support difference is the raw likelihood/`diff2` surface. This is not a final K=4 quality admission. | Capture/control replay used one H100 for 35 s; timing is instrumentation qualification, not an engine speed comparison. |
| `real_k4_native_coarse_component_boundary_20260902.md` and `diagnostics/real-k4-native-coarse-components-8f9ebc9-20260902.json` | Paired native norm/cross components and RECOVAR full component surfaces for the same 16 probes; jobs 13332998 and 13332392 | Diagnostic PASS: a native cross-only swap restores 11/16 exact supports versus 5/16 for RECOVAR and 5/16 for a native-norm-only swap. The cross term is the dominant first likelihood component boundary. | One-H100 capture/replay timing is diagnostic only. The earlier in-kernel observer is retained as rejected evidence because it changed production results. |
| `real_k4_native_coarse_operand_boundary_20260902.md` and `diagnostics/real-k4-native-coarse-operands-b061776-20260902.json` | Historical three-repeat operand localization plus current-source rounded-shell/Euler replay; jobs 13334583--13334585, 13336787, 13336962, and 13337519 | Diagnostic PASS and bounded boundary closure: native projected references localized the old defect; commit `5f74755c2` then yields zero projected-reference/Euler error and exact support for 16/16 probes and 1,336/1,336 selected candidates. This is not a final K=4 refinement admission. | Replay state is bitwise repeatable; eight GPU-atomic maps have minimum FSC-AUC `0.999999997335` and maximum relative L2 `1.57406e-7`. Timings remain diagnostic, not an engine speed comparison. Job 13334206 remains rejected because synchronous capture perturbed production results. |
| `real_kclass_halfmap_refinement.md`, first-iteration native score boundary | Passive native RELION coarse/fine scores and RECOVAR surfaces for 16 frozen EMPIAR-10076 particles; 1,069,056 coarse candidates; H100 job 13346151 | Diagnostic PASS: commit `4a91369a3` preserves only `run_it000` origins during the fresh Class3D global search. Winner classes, global poses, all per-class poses, fine parents/winners, and integer pre-shifts are exact across the panel. This closes the first-iteration score boundary, not final K=4 quality. | RECOVAR wall 80.4 s and peak HBM 33,423 MiB for the containing 5,000-particle iteration. Full-surface minimum correlation `0.999999999996`; worst fine-score absolute error `7.45e-8`. Sealed JSON SHA-256 `977d115f1593`. |
| `diagnostics/real-k4-native-signfix-causal-7136e5c8d-20260902.json` | Seed-42001 two-iteration EMPIAR-10076 half-1 causal A/B after all three controls independently logged the same class-3 sign flip | Causal PASS: commit `7136e5c8d` restores class-3 iteration-1 FSC-AUC from -0.990489 to +0.990489 and iteration-2 occupancy from 0.0012 to 0.0368 versus RELION 0.0382. Iteration-2 class agreement is 0.942, so this is not a final K=4 admission. | H100 job 13348468 completed in 9m22s, exit 0, exact one-GPU allocation, peak HBM 33,465 MiB. Full JSON SHA-256 `25f3a772fc31`. |
| `diagnostics/real-k4-pilot10k-multiseed-stability-7136e5c8d-20260902.json` | Three seeds, two immutable 5,000-particle halves per seed, eight K=4 iterations per engine | All three prospective gates remain rejected. Median paired masked half-map FSC-AUC delta is -0.00054, but seed-42001 class 3 is -0.11925. Same-seed cross-engine labels and final maps are substantially closer than either engine is across seeds; for every class, same-seed map FSC-AUC minimum 0.8352--0.9527 exceeds within-engine cross-seed maximum 0.6440--0.8782. This establishes a seed-sensitive local-optimum boundary without rescuing the failures. | Same-H100 serial measurements across six halves: RECOVAR median 1,280.24 s / 33,478 MiB; RELION 380.76 s / 79,588 MiB. Map analysis job 13353269 completed on exact 4-CPU/64-GiB resources. |
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
   the strongest within-engine cross-seed match for every class. Retain the
   seed-42001 weak-class outlier and frozen gate failures, but do not spend more
   pilot effort treating it as a deterministic RECOVAR-only defect. Do not
   promote the result to native-grid acceptance.
5. Add a dedicated whole-ledger validator for the selected-fine diagnostic or
   migrate it into a versioned diagnostic schema before relying on it as more
   than causal evidence.
6. The runnable robustness matrix includes K=2, K=8, and K=16, but no accepted
   schema-v1 result for those K values is checked in. Do not generalize the K=4
   campaigns to them.

## Validation and replay

The ordinary checks do not rehash large external stacks:

```bash
pixi run python scripts/validate_em_benchmark_registry.py
pixi run pytest tests/unit/test_validate_em_benchmark_registry.py
pixi run python scripts/validate_em_real_kclass_diagnostics.py
pixi run pytest tests/unit/initial_model/test_validate_em_real_kclass_diagnostics.py
pixi run python scripts/validate_em_real_kclass_offset_prior_fullpairs.py
pixi run pytest tests/unit/initial_model/test_validate_em_real_kclass_offset_prior_fullpairs.py
pixi run python scripts/summarize_em_k1_realdata_science_equivalence.py --verify-calibrations --verify-masked-support --verify-target-partial --check-markdown
```

Use each validator's explicit `--verify-files` mode only when resealing or
re-auditing external evidence. No command in this inventory submits Slurm.

## Historical baseline left unchanged

`tests/baselines/em_parity_completion_k4_100k256.json` remains the historical
correlation-based completion guard from job 8290126. It was not rewritten with
new FSC evidence. Current scientific and performance claims belong in the
versioned registry and dedicated diagnostic ledgers above.
