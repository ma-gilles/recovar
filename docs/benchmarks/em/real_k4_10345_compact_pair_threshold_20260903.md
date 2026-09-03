# EMPIAR-10345 native-grid K=4 compact-pair threshold audit

The compact-pair route is faster than rectangular execution for small K=4
support buckets, but changing their grouping can change floating-point
reduction order. This real-data audit asks whether lowering the production
minimum pair-bucket size from 512 preserves an eight-iteration trajectory,
not merely whether a two-iteration timing sample looks unchanged.

## Threshold 1: rejected

The first candidate repeats RECOVAR half 1 of the sealed EMPIAR-10345
native10k seed-42001 run with the sole override
`RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE=1`. The control and
candidate otherwise use identical 5,000 particles, four starting maps,
dispatch replay, seed, binary, source commit `2f6759608`, and science options.

The thresholds were frozen and hashed while GPU job `13376529` was running.
Acceptance required exact controller arrays, assignments, and support counts;
pose-coordinate drift at most 1e-6; Pmax drift at most 1e-5; final-map signed
non-DC FSC-AUC at least 0.99999 and relative L2 at most 0.001; at least 5%
sparse bucket-group speedup; and no more than 5% sampled-HBM growth.

| Metric | Control | Threshold 1 | Result |
| --- | ---: | ---: | --- |
| Sparse bucket-group wall | 1109.3 s | 959.0 s | 13.55% faster, pass |
| Sampled peak HBM | 33,789 MiB | 33,781 MiB | pass |
| Assignment mismatches, iterations 1--7 | 0 | 0 | pass |
| Assignment mismatches, iteration 8 | 0 | 7/5,000 | fail |
| Minimum final-map FSC-AUC | 1 | 0.9992810281 | fail |
| Maximum final-map relative L2 | 0 | 0.0047774016 | fail |

The maps remain scientifically close, but the prospective behavior-equivalence
gate is deliberately stricter: threshold 1 is classified
`REJECTED_FAST_BUT_TRAJECTORY_NONEXACT` and is not made the production
default. Less aggressive thresholds are evaluated under the same frozen
contract rather than weakening this gate after seeing the result.

## Thresholds 128 and 256: historical-control result

The same eight-iteration half-1 experiment was then repeated with thresholds
128 and 256. Their acceptance documents were sealed before either run
completed and retained the threshold-1 contract unchanged. Both candidates
again passed the performance and sampled-HBM gates but failed strict
trajectory equivalence against the original default-512 control.

| Metric | Original default 512 | Threshold 128 | Threshold 256 |
| --- | ---: | ---: | ---: |
| Sparse bucket-group wall | 1109.3 s | 925.1 s | 944.5 s |
| Sparse-wall ratio | 1 | 0.833949 | 0.851438 |
| Sampled peak HBM | 33,789 MiB | 33,781 MiB | 33,783 MiB |
| Assignment mismatches | 0 | 3 | 1 |
| Significant-count mismatches | 0 | 68 | 45 |
| Minimum final-map FSC-AUC | 1 | 0.9999027508 | 0.9995443157 |
| Maximum final-map relative L2 | 0 | 0.0020626852 | 0.0035064463 |

Threshold 128 was 16.61% faster in the measured sparse phase and threshold
256 was 14.86% faster. These historical-control comparisons still classify
both runs as formal rejects; that classification is preserved below even
after calibrating the control's own repeatability.

## Fresh default-512 repeatability calibration

A fresh default-512 run with the environment override unset was launched from
the same source, input, dispatch replay, and science command. It did not pass
the deliberately exact historical-control contract either: it produced three
iteration-8 assignment changes, 69 significant-count changes, minimum map
FSC-AUC 0.9999027483, and maximum relative L2 0.0020627594. Its endpoint is
nearly identical to threshold 128's endpoint, including the same three
historical-control assignment changes and the same pose tails. This proves
that the original gate is stricter than the observed repeatability of the
RECOVAR route itself; it does not retroactively turn any failed gate into a
pass.

The direct candidate-versus-fresh-default contract was separately frozen from
the published RELION repeat floor before its computation. Threshold 128
preserved all controller arrays, all 40,000 particle/iteration assignments,
and every Euler and translation. It reduced sparse wall time by 15.21% with
unchanged HBM; maximum map relative L2 was only
`3.1916274e-6`. Nevertheless, its minimum direct signed non-DC FSC-AUC was
`0.999999909144`, below the prospective `0.99999999` gate, so its formal
decision remains `REJECT`. Threshold 256 was decisively separated from the
fresh default by two assignments, minimum FSC-AUC `0.999564619529`, and
maximum relative L2 `0.00341399785`.

Threshold 128 is therefore retained as a scientifically negligible,
performance-positive seed-42001 result, not by itself as a changed production
default. Threshold 256 is closed.

## Three-seed threshold-128 decision: rejected

Two additional contemporaneous default-512/threshold-128 pairs used seeds
42002 and 42003. Their contract was sealed before any of the four GPU runs
completed. It retains the formal FSC/L2 thresholds and adds a separately
labeled science-equivalence tier requiring exact controller arrays, exact
class assignments, exact poses, no class collapse, final-map FSC-AUC at least
0.999999, final-map relative L2 at most 0.0001, at least 5% sparse-phase
speedup, and at most 5% sampled-HBM growth. Seed 42001's science label is
explicitly retrospective; seeds 42002 and 42003 are prospective. The aggregate
is an all-pair conjunction and never averages metrics across seeds.

| Seed | Formal | Science | Assignment mismatches | Poses exact | Min map FSC-AUC | Max map relL2 | Sparse-wall ratio | HBM ratio |
| ---: | :---: | :---: | ---: | :---: | ---: | ---: | ---: | ---: |
| 42001 | reject | accept (retrospective) | 0/40,000 | yes | 0.999999909144 | 3.19163e-6 | 0.847860 | 0.999645 |
| 42002 | reject | reject (prospective) | 0/40,000 | **no** | 0.999999792004 | 8.83407e-5 | 0.872308 | 0.999763 |
| 42003 | reject | accept (prospective) | 0/40,000 | yes | 0.999999909518 | 8.62948e-7 | 0.827732 | 0.999586 |

Seed 42002 remains identical through iteration 7. At iteration 8, exactly one
of 5,000 Euler rows changes (source row 3280), from
`[-142.99463, 150.59065, 2.5717983]` to
`[13.690024, 161.02734, 157.18436]` degrees. This is not merely a different
Euler representation: the corresponding physical rotation matrices differ by
47.374945 degrees. Translations, class assignments, controller state, and
class occupancy remain exact. The maps are nearly indistinguishable, but the
predeclared exact-pose science gate fails.

The aggregate outcome is formal `REJECT` (0/3 pairs) and scientific-equivalence
`REJECT` (2/3 pairs). Consequently, threshold 128 is not made the production
default; the minimum compact-pair bucket size remains 512. This is deliberately
the conservative interpretation: speed and endpoint-map similarity cannot be
averaged across seeds to erase an observed trajectory change.

## Reproduction and audit trail

The complete candidate, predeclared thresholds, launchers, logs, saved states,
maps, and analysis are under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_threshold1_full8_10345_native10k_seed42001_2f6759608_20260903`.
GPU job `13376529` requested and received exactly one H100, eight CPUs, and
128 GiB, completing in 24m41s. CPU audit job `13377361` requested and received
exactly four CPUs and 64 GiB, and exited `3:0` solely to encode the frozen
rejection. Neither job requested exclusive access; both run and runtime roots
have `SAFE_TO_DELETE`.

The predeclared threshold JSON has SHA-256
`af98d2e4bb0969d9bb1a0b6f411628514902d1afdad6bfac3261f6103164b52c`.
The complete audit JSON has SHA-256
`80089b0a075fced90161e3ea274bf5ce6eaa0d14060ee7329f847e0c8f60b534`.
The compact checked-in record is
`diagnostics/real-k4-10345-native10k-compact-pair-threshold1-2f6759608-20260903.json`.

Threshold-128 GPU job `13377631` requested and received exactly one H100,
eight CPUs, and 128 GiB, and completed `0:0` in 23m55s. Its intentionally
rejecting CPU audit was job `13378106`. The complete run is under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_threshold128_full8_10345_native10k_seed42001_2f6759608_20260903`;
its predeclared threshold and audit JSON hashes are
`d69d2c8de31a9fb7c7cfcae436f5d7970b7345a55271233c3dfd52b2f73828c3`
and `982b0a18cfde84651e4220603918aec25996935517795b3920167389b209743b`.

Threshold-256 GPU job `13377656` used the same exact resource shape and
completed `0:0` in 24m11s; rejecting CPU audit job `13378158` completed in 18s.
Its run root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_threshold256_full8_10345_native10k_seed42001_2f6759608_20260903`;
the corresponding threshold and audit hashes are
`23e1e77c0466384db9ed2fdf59d90963e7a8ea29fab062640e686c212c85a618`
and `3b0ebf5dade6401ca054ec85abaa7e61e34659564e3b44bc4d30b758ccdc8fa2`.

Fresh-default GPU job `13377738` also requested and received exactly one
H100, eight CPUs, and 128 GiB and completed `0:0` in 26m36s. Its strict audit
job `13378278` exited `3:0` solely to encode `NOT_REPEATABLE`. The run root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_default512_full8_10345_native10k_seed42001_2f6759608_20260903`;
the predeclared threshold and audit hashes are
`1c3ca53145254cc558930fc7a80663e6b1d207555de17c6b43202455bfaf7411`
and `a487a19f21fab28b6821c2d6dccdf0a2ae243016308d878cc54637f7fd7582ec`.

Direct comparison audit job `13378363` requested and received four CPUs and
64 GiB and exited `3:0` because neither candidate passed every prospective
gate. Its complete root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_threshold128_256_vs_default512_direct_10345_native10k_seed42001_2f6759608_20260903`.
The frozen contract and result JSON hashes are
`21ff779ce5327f132bb026f98f839a7fa8264a35246f3536e2d0c02f0f6d5848`
and `360af00952259ce6f31564db8da204d4946f08767f7fe3260456f6433282b8e1`.
All jobs were nonexclusive, and every disposable run and runtime root carries
`SAFE_TO_DELETE`.

The valid additional GPU pairs are jobs `13378734` (seed 42002, default 512,
26m28s), `13378735` (seed 42002, threshold 128, 24m10s), `13378736` (seed
42003, default 512, 26m11s), and `13378737` (seed 42003, threshold 128,
23m02s). Each requested and received exactly one H100, eight CPUs, and 128 GiB
without `--exclusive`. Pair-audit jobs `13378859`, `13378860`, and `13378861`
requested and received four CPUs and 64 GiB; aggregate job `13378887` requested
and received one CPU and 4 GiB. Their `3:0` exits are expected encodings of the
frozen rejection, not compute failures.

The complete campaign root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_threshold128_default512_seedpair_42002_42003_10345_native10k_2f6759608_20260903`.
The attempt-2 contract, pair JSONs for seeds 42001--42003, aggregate JSON, and
aggregate Markdown have SHA-256
`a38953a6a8b1d7365d9d790f694fbed80f1fb572714be85a5eb9219f6b413e7f`,
`60684e10414fd83109f4aa807a33c4baeb313338dd8adae2b29b9c3fb6bf45a8`,
`29f2b481b4d428d83168f13548e033a0454d233387946d68ec8aecf878e3ec63`,
`3c8ca12be45a97562b6680a8ebfc678286104d7893942bece42d131ea1bc5992`,
`19a341f196c955ba6fdb19ac975caee412c66ccec88cdfab5719858527807c96`,
and `6d98231b644796c035c8ac59a8b41ea39d139ed55ac7ff29e0ee4b48a7cf29e0`,
respectively. Failed jobs `13378526`--`13378529` are retained and explicitly
excluded: the first launcher accidentally combined seed-42002/42003 commands
with a seed-42001 dispatch schedule and failed before iteration 1. The exclusion
record's SHA-256 is
`47bc86779437b604fc7f5fc22525a8551e87ed506edb65cfdff88bc926659980`.

The reusable reproduction tools are
`scripts/audit_em_kclass_default_threshold_pair.py` and
`scripts/aggregate_em_kclass_default_threshold_pairs.py`; their unit tests
cover state mutation, no-collapse checks, formal/science separation, incomplete
logs, conjunction without averaging, and duplicate-label rejection.
