# Measured coarse-cap tie: real10076 E6, row942

Frozen5ca9, six instrumented candidate repeats, job13666912. This is a local
score/support observation, not native matched-operand or trajectory acceptance.

All200 selected rows have identical support sets across the six arms at each
iteration1–5. Exactly row942 differs at6. Its competing coarse IDs7206/9868 sit
at ranks100/101 under max_significants=100. One score ULP is1.52587890625e-5.

| Arm | Score7206 − score9868 (ULPs) | Published support size | Membership |
| --- | ---: | ---: | --- |
| cap1 | 3 | 100 | 7206 only |
| cap2 | 0 | 101 | both |
| cap3 | 0 | 101 | both |
| cap4 | -1 | 100 | 9868 only |
| cap5 | 2 | 100 | 7206 only |
| cap6 | 2 | 100 | 7206 only |

For all six saved score rows, selecting scores >= the100th-largest score exactly
reproduces the published support IDs. Strict-greater would exclude the boundary;
blindly retaining only100 would discard one member of the exact ties. A float64
mass-sum diagnostic places the uncapped0.999 mass cutoff at634 in every arm;
that diagnostic does not propose production double arithmetic.

The original report overstates scalar equality: thresholds range
7.690334519118791e18–7.690451617107149e18, which differ at six significant digits.
Pmax ranges0.1808100789785385–0.1808115541934967 (spread1.4752149582e-6).
The competing scores themselves are approximately−192.995;−182.274 is the
row raw maximum, not either competing score. These corrections do not remove
the demonstrated cutoff near tie.

**Admitted:** a measured near-tie support decision in these instrumented candidate
histories. **Not proved by this capture:** its incoming-state cause, M-step
nondeterminism as the sole cause, native candidate geometry/scores on identical
inputs, no E-step bugs, a50/50 population rate, or acceptance of later map/state
gates. The observer reads host-return values; no intervention in its source is
reported, but instrumentation and different incoming histories remain limitations.
The failed real-full200, robustness and K4 conditions remain failed.

[Exact values and hashes](https://github.com/ma-gilles/recovar/blob/5c07fc3169d636dc3775b2bb41ae4c76d6829611/docs/development/evidence/vdam-coarse-cap-tie-20260910/result.json),
[portable row scores/support](https://github.com/ma-gilles/recovar/blob/5c07fc3169d636dc3775b2bb41ae4c76d6829611/docs/development/evidence/vdam-coarse-cap-tie-20260910/row942_scores_and_support.npz),
[review script](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/coarse_cap_tie_review_20260910/review.py),
[CPU command/receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/coarse_cap_tie_review_20260910/verification/receipt.json).
All declared launch pins and the six capture/index files plus36 metadata files
were hashed before/after. This does not provide complete native/input build
closure. No source, baseline, tolerance, GPU workload or shared binary changed.
