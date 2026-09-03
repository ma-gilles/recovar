# EMPIAR-10345 native-grid K=4 three-seed diagnostic

Three independent `native10k-256` runs now evaluate RECOVAR and RELION on the
same EMPIAR-10345 particles, halves, four starting maps, and stochastic seed.
Each seed uses 5,000 particles per independent half, eight C1 iterations, and
serial same-H100 execution. The source is commit `2f6759608`, and the frozen
seeds are 42001, 42002, and 42003.

All six RECOVAR and all six RELION half refinements completed normally. The
three qualification jobs ended with exit code 3 only because their
predeclared science gates rejected the result. None had a compute failure,
OOM, traceback, missing class, or ambiguous class permutation.

## Aggregate result

The strongest discriminator compares seed-matched engines with each engine
across seeds:

| Metric | Seed-matched RECOVAR vs RELION | Within one engine across seeds |
| --- | ---: | ---: |
| Full unmasked non-DC class-map FSC-AUC | 0.98530--0.99314 (mean 0.98902) | 0.62790--0.82231 (mean 0.72049) |
| Hard-assignment agreement, half 1 | 0.9614--0.9634 (mean 0.9627) | 0.5900--0.6516 (mean 0.6299) |
| Hard-assignment agreement, half 2 | 0.9616--0.9730 (mean 0.9669) | 0.5868--0.6522 (mean 0.6214) |

For every one of the four classes, the minimum seed-matched cross-engine map
agreement exceeds the maximum within-engine cross-seed agreement. The same
strict separation holds for assignments in both halves. This is strong
evidence that RECOVAR and RELION follow the same seed-dependent basin and that
the remaining endpoint difference is a shared local-mode boundary, rather
than a RECOVAR-specific class reconstruction defect.

The common-mask evidence is closer still. Across all 12 seed/class cells,
registered merged cross-engine FSC-AUC is 0.997499--0.998972 (mean 0.998388).
RECOVAR-minus-RELION masked half-map FSC-AUC has median +0.000519 and range
-0.013877 to +0.005097.

## Admission decision

This evidence does not rewrite the gate after seeing the outcome. All three
per-seed prospective gates remain rejected: the 0.99 hard-assignment
requirement fails, some unmasked merged-map cells fall just below 0.99, and
seed 42001 class 4 has a -0.013877 masked half-map AUC delta. The aggregate is
therefore a complete diagnostic with classification
`SHARED_SEED_SENSITIVE_LOCAL_OPTIMUM_BOUNDARY`, not an accepted Tier-6
registry entry.

An independent same-input, same-seed RELION repeat is being used to measure
same-engine nondeterminism directly. Until that finishes, this report makes no
claim that every residual 3--4% assignment difference is unavoidable.

## Performance and reproduction

Across the six half runs, RELION takes 141.25--147.48 s (mean 143.14 s) and
RECOVAR takes 1,532.41--1,708.03 s (mean 1,603.23 s). Sampled peak HBM is
79,573--79,585 MiB for RELION and 33,789--33,795 MiB for RECOVAR. Quality and
performance remain separate decisions.

The seed roots are:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_halfmap_10345_native10k_seed42001_2f6759608_20260903`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_halfmap_10345_native10k_seed42002_2f6759608_20260903`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_halfmap_10345_native10k_seed42003_2f6759608_20260903`

Their setup/qualification jobs are `13371068/13371069`,
`13374047/13374048`, and `13374045/13374046`. Reproduce each seed with the
`--dataset 10345 --profile native10k-256 --seed <seed>` command in
`real_kclass_halfmap_refinement.md`, always using a fresh output root.

CPU aggregate job `13375384` requested and received exactly four CPUs and
64 GiB, completed in 6m46s with exit 0, and produced:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_halfmap_10345_native10k_multiseed_2f6759608_20260903/analysis/assignment_and_quality_stability.json`
  (SHA-256 `cf47aeb5b6f1e4901d58b68d6e0a810abaea9f9bca1a1d97aa86d0154aed9066`);
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_halfmap_10345_native10k_multiseed_2f6759608_20260903/analysis/map_stability.json`
  (SHA-256 `4b9288f1ade5726d53bde8dc3a4e41ee78fa1a722a23dc60dffc290350558a9b`).

The compact checked-in record is
`docs/benchmarks/em/diagnostics/real-k4-10345-native10k-multiseed-2f6759608-20260903.json`.
