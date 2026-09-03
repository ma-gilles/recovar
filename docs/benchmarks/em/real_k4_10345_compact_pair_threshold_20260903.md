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
