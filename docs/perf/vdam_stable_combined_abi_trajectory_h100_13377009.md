# VDAM stable Fourier + flat-row ABI trajectory gate — H100 job 13377009

## Decision

**HOLD default-off.** The combined stable-ABI policy is a material performance
win, reducing median fresh-process wall time by 25.90% through iteration 50,
but it does not pass the predeclared strict trajectory-science gate. Both
candidate repeats enter the same alternate hard-assignment basin at iteration
35. A flat-row-only ABBA is required to separate the high-value fixed-`B*R`
row ABI from the lower-value Fourier-window bucketing policy.

The Slurm job is recorded as `FAILED (1:0)` because the completed analyzer
correctly returned status 1 for the science rejection. All four scientific
runs and the report completed.

## Setup

| Field | Value |
|---|---|
| Source commit | `f5792e9b0af6a70f13cb63a546d594013c774f16` |
| Job / node | `13377009` / `della-h19g2` |
| Physical GPU | H100 `GPU-e2c3190a-9599-15f7-a19c-7ae55e4e0a85` |
| Panel | stable off / on / on / off |
| Isolation | four fresh Python processes and four fresh JAX caches |
| Trajectory | K=1 GF46, iterations `0 -> 50`, 19 logical Fourier sizes |
| Candidate-only deltas | stable Fourier-window shapes and fixed flat-row capacity `Q=B*R` |
| CUDA artifact SHA-256 | `b4d5a24d679123faf29d438accb2a64fea6da60840be07c1e920928f72dec21d` |
| RELION binding SHA-256 | `9bbb1fb0ce6fa7ac816598ec521453515d163221642b916e5715bb2850798980` |
| Analyzer report SHA-256 | `a39f95b6b0f56b95b1f7e91bd6531f9b7c1a4d582c37e94503cf9656c13400e0` |

The stable policy reduced the 19 logical Fourier sizes to eight physical size
classes. Ordinary packed execution exercised 152 calls with `Q < B*R`; both
candidate runs instead used the fixed rectangular row ABI on every call.

## Performance

| Metric | Ordinary median | Stable median | Change |
|---|---:|---:|---:|
| End-to-end wall | 513.649 s | 380.602 s | **-25.90%** |
| Expectation stage | 479.404 s | 347.992 s | **-27.41%** |
| Peak GPU memory | 17,587 MiB | 17,585 MiB | -0.01% |
| Profiled local EM | 360.709 s | 227.037 s | -37.06% |
| Local big-JIT buckets | 135.706 s | 90.579 s | -33.25% |
| Deferred local noise | 122.309 s | 62.132 s | -49.20% |
| Local packing | 30.252 s | 18.306 s | -39.49% |

As an independent shape-churn proxy, each ordinary fresh cache contains 4,774
compiled cache objects versus 3,283 for each stable run (-31.23%). The
`run_local_bucket_big_jit` entries fall from 87 to 49 (-43.68%). Cache-object
counts are not kernel timings, but they agree with the measured compile-heavy
stage reductions and with the original ABI-churn diagnosis.

## Science result

The strict report contains 52 exact-state and 285 fixed-bound failures. The
important pairwise chronology is more informative than the aggregate count:

| Pair | First hard-state difference | Repeat behavior through iteration 50 |
|---|---|---|
| ordinary 1 vs ordinary 2 | iteration 48 | one pose differs at iterations 48 and 50 |
| stable 1 vs stable 2 | none | every predeclared discrete field is exact |
| ordinary vs stable | iteration 35 | two rotation IDs / poses first differ; the basin separation persists |

The stable repeats are therefore deterministic in the checked hard state and
substantially tighter in their map trajectory: whole-trajectory map RMS
normalized L2 is `8.89e-7` between stable repeats versus `7.51e-6` between
ordinary repeats. Cross-mode RMS is about `9.05e-5`, with a maximum checkpoint
distance of `3.10e-4`. This is consistent with a mathematically equivalent
shape/reduction-order perturbation accumulating into another basin, but it is
not strict direct-path identity and cannot promote the policy by itself.

A post-hoc diagnostic at iteration 50 found that the stable pair was very
slightly closer than the ordinary pair to both the frozen RELION map
(`0.7951329` versus `0.7951238` median FSC-AUC) and ground truth (`0.0074307`
versus `0.0074222`). This is not a predeclared acceptance result and does not
override the failed gate; it only rules out an obvious quality collapse at the
first 50 iterations.

## Next gate

Job `13378054` runs the same four-fresh-process panel with stable Fourier
shapes disabled and only fixed `Q=B*R` enabled. This identifies whether the
25.90% gain and the iteration-35 basin choice both come from the packed-row
ABI. The complete optimized stack remains fail-closed until that isolation and
the subsequent repeat-controlled `0 -> 200` RELION-quality gate complete.

No broad RECOVAR suite was run for this performance experiment.
