# GF46 integrated-hybrid checkpoint sealed diagnostic — job 13343577

> **Overall: FAIL, with the failure confined to two analyzer boundaries.**
> All six GPU arms completed and the result was sealed. The feature remains
> default-off and the frozen parity scores are unchanged.

## At a glance

| Gate | Result | Status |
|---|---:|---:|
| Persisted cutoff/discrete/STAR/model identity | exact across audit and clean arms | PASS |
| Hash-only ordered support rule | direct includes one row-412 tie; hybrid excludes it | FAIL |
| Model-state pooled repeat envelope | all 15 hybrid-repeat and 36 crossed pairs | PASS |
| Map pooled repeat envelope | only normalized scale drift `2.21e-11 > 1.75e-11` | FAIL |
| Median warm wall | `21.384448 -> 5.297722 s` (**4.037x**) | PASS |
| Median expectation | `20.641061 -> 4.442341 s` (**4.646x**) | PASS |
| Median pass 1 | `18.790628 -> 2.286312 s` (**8.219x**) | PASS |
| Median peak RSS | `3.339 -> 3.518 GiB` (`+5.36%`) | INFO |

Relative L2, max absolute map delta, and signed map bias all passed. The one
failed map metric is far below float32 resolution. The superseding analyzer
therefore reuses the already predeclared true-200 normalized numerical floor
of `4 * eps(float32) = 4.76837158203125e-7`; max-absolute and signed-bias
limits remain derived from direct repeats.

The hash-only audit could localize the support difference to image row 412
and prove that every serialized cutoff count was exact, but could not prove
which ID differed. The superseding diagnostic stores the exact selected IDs
(about 5,900 integers total), validates their existing row and aggregate
hashes, and accepts only pairwise-nested inclusive-tie expansions whose common
core covers the exact cutoff rank.

## Provenance

- Slurm: `13343577`, state `FAILED 1:0` (semantic diagnostic failure), elapsed
  `00:06:52` on `della-h19g1`.
- GPU: `GPU-75c2d200-95d1-ef57-fb52-1698386c756c` (H100).
- Source/tree: `0056f3e3957e51b3538c5d7d9630e38e95a3f869` /
  `db54b57c87db0a12f36a00355d50a2db1748af98`.
- Source/input manifests: `94a5eb68655bf5d02c3718d98f40dadf8588635fea0e90d144a1f815c91733b4` /
  `de224471a690d1faaae4067217dbcc90b632269d62b0b3372b20aafa69157d91`.
- Sealed root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf46_hybrid_transition_13343577_0056f3e39`.

This diagnostic does not qualify a long trajectory, default enablement, or a
change to frozen correctness **2/20** or runtime **0/20**.
