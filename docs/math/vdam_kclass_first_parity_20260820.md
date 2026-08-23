# VDAM K-class first parity boundary (2026-08-20)

## Contract

K-class InitialModel comparisons use a Hungarian class assignment selected by
normalized non-DC FSC-AUC.  Each matched class must have FSC-AUC at least
`0.999`; assigned-particle labels must agree at least `0.995` after applying
the same permutation.  Map correlation is neither computed nor gated.
Iteration zero has no assigned particles in RELION and is therefore map-only.

Both programs run sequentially on one physical Slurm GPU.  The reference is
the clean pinned RELION `f2c1a384400aec37dc6805856a5ba645650a44f1`
executable with SHA-256
`08d9151c976ec51e664060992db62d929e47282e0d8481d977f64e32f07fca39`.

## First K=2 result

The 5,000-particle, 128-pixel two-state PDB fixture first exposed a strict
CUDA preprocessing boundary failure: the split local-exact path did not pass
explicit identity normalization factors and integer shifts after those
operations had already been handled by its caller.  The backend correctly
failed closed.  The fixed boundary supplies typed identity operands only to
the strict RELION CUDA backend; general backends retain the prior path.

Same-GPU paired job `12767153` then passed iterations 0, 1, and 2:

| Iteration | Minimum matched FSC-AUC | Assignment agreement |
|---:|---:|---:|
| 0 | 1.000000000 | not assigned |
| 1 | 0.999995378 | 1.0000 |
| 2 | 0.999993747 | 0.9984 |

At iteration 2, eight of 5,000 hard class labels differ (four in each
direction), while both class maps remain above `0.9999937`.  This residual is
kept visible rather than described as exact particle-state equality.

The paired CLI wall times were 8.93 seconds for RELION and 51.83 seconds for
RECOVAR, a `5.80x` ratio.  Comparable-runtime work therefore remains open.
The immutable diagnostic report is:

`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k2_paired_evidence_retry_20260820T1745Z/pair/pair_report.json`

with SHA-256
`4c773b4fe1ef73e984e4afdf843b46cf68408603d861a618f8b2c3ea1d059724`.

The next qualification boundary is the clean-source K=2 default trajectory
at checkpoints 0, 1, 2, 4, and 8, followed by parameter variants and K=4.

## Current qualification tracker (2026-08-22)

The K-class path now passes the complete eight-iteration default trajectory
for both K=2 and K=4.  K=2 also passes the first parameter panel.  K=4 passes
the repeatability, 25-iteration, 10,000-particle real-data, and
100,000-particle/256-pixel scale gates.  The default audits compare every
written iteration 0 through 8 and the long audit compares every written
iteration 0 through 25.  All audits require exact artifact topology and retain
the fixed `0.999` per-class FSC-AUC and `0.995` class-assignment thresholds.

| Case | Result | Minimum matched FSC-AUC | Minimum assignment accuracy | RELION wall time | RECOVAR wall time | Ratio |
|---|---|---:|---:|---:|---:|---:|
| K=2 default | PASS | 0.9999995522 | 1.0000 | 21.36 s | 119.97 s | 5.62x |
| K=2 default, native dual replay | PASS | 0.9999999999 | 1.0000 | 14.34 s | 122.67 s | 8.55x |
| K=4 default, clean post-fix | PASS | 0.9999999972 | 1.0000 | 17.34 s | 156.06 s | 9.00x |
| K=4 default, native dual replay | PASS | 0.9999999996 | 1.0000 | 17.34 s | 144.00 s | 8.31x |
| K=4, 25 iterations | PASS | 0.9999999679 | 1.0000 | 41.47 s | 402.48 s | 9.71x |
| K=4, real 10076, 10,000 particles | PASS | 0.9999987694 | 0.9988 | 71.65 s | 705.67 s | 9.85x |
| K=4, 100,000 particles, 256 pixels | PASS | 0.9999851527 | 0.9998 | 267.06 s | 1960.97 s | 7.34x |
| K=2 seed 7 | PASS | 0.9999999996 | 1.0000 | 25.65 s | 313.97 s | 12.24x |
| K=2 Healpix 0 | PASS | 0.9999999995 | 1.0000 | 20.19 s | 366.77 s | 18.16x |
| K=2 tau2 fudge 2 | PASS | 0.9999999999 | 1.0000 | 28.59 s | 228.09 s | 7.98x |
| K=2 tau2 fudge 8 | PASS | 0.9999999999 | 1.0000 | 28.71 s | 217.94 s | 7.59x |
| K=2 offset range/step 4/1 | PASS | 0.9999999999 | 1.0000 | 28.93 s | 322.38 s | 11.14x |
| K=2 diameter 140 A | PASS | 0.9999999898 | 1.0000 | 28.59 s | 304.26 s | 10.64x |
| K=2 Healpix 2 | PASS | 0.9999996951 | 1.0000 | 30.76 s | 246.61 s | 8.02x |
| K=2 oversampling 2 | PASS | 0.9999999987 | 1.0000 | 31.44 s | 434.77 s | 13.83x |
| K=2 oversampling 0, latest EM stack | PASS | 0.9999999993 | 0.9998 | 28.61 s | 87.99 s | 3.07x |

The default and first parameter runs are Slurm jobs `12777536`, `12777537`,
`12777748`, `12777749`, `12778397`, `12778401`, `12778803`, `12778815`,
`12784677`, and `12784678`.  Job `12787780` is the clean post-rebase K=2
oversampling-zero qualification on top of the current PR 158 EM stack.  The
clean current-head K=4 default and long gates are jobs `12798176` and
`12798175`; the real, scale, and repeatability gates are `12796578`,
`12796579`, and `12796580`.  The K=2 oversampling-zero report is:

`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k2_os0_3cd31e91/pair/pair_report.json`

Correctness is qualified for this panel, but runtime is not: RECOVAR remains
3.07x to 18.16x slower on the small paired jobs and 7.34x slower on the frozen
scale gate.  The remaining tracker is:

- rerun the K=2 default and nonzero-oversampling panel after every material EM
  stack update;
- rerun the qualified K=4 parameter, scale, repeatability, 25-iteration, and
  real-particle panels after every accepted runtime change;
- profile the qualified fused compact-pair route and remove recompilation and
  per-iteration launch overhead without changing the parity thresholds;
- keep the public `recovar initial_model` CLI and GUI defaults as the single
  configurable entry point while exposing the important sampling, class,
  mask/CTF, batching, and diagnostic controls.

## Current runtime boundary

The clean default is dominated by fused compact-pair pass 2, not by projector
construction, artifact writing, or host staging.  Group-timed job `12798234`
attributes about 94% of the RECOVAR wall to the expectation step.  Fine score
and M-step/noise reductions dominate within each group.  A previously seen
shape is fast in the second pseudo-half, while the first encounter of a new
pair-bucket/image-batch/window shape costs seconds.  This identifies changing
JAX executable shapes and their first launch as the principal small-case
overhead.  Raw diff2 host staging is only hundredths of a second per call and
is not a useful optimization target.

A fresh-cache compile audit (`12801524`) and one-iteration cProfile audit
(`12801764`) confirmed this boundary.  The profiled process created 1,982
backend executables and spent about 95.2 seconds in backend compilation;
bucket I/O preparation and indexed gathers were the next material Python/JAX
costs.  Removing generic float64 posterior normalization that the guarded
RELION float32 path immediately discarded is therefore accepted.  Same-node
job `12801958` completed in 150 seconds versus 154 seconds for control job
`12798234`; its eight-iteration trajectory retained minimum FSC-AUC
`0.9999999973`, exact assignments, and exact artifact topology.  The focused
unit file passed 169 tests with one skip.

The persistent-cache write policy is also narrowed from every executable to
executables taking at least 10 ms to compile.  The prior zero-second policy had
grown the default cache to 382,428 files.  A staged threshold qualification
retained 8,162 files at 10 ms and its complete warm K=4 replay (`12805441`)
finished in 150 seconds, equal to accepted job `12801958`.  Its eight-iteration
trajectory audit (`12805777`) passed with exact artifact topology, exact class
assignments, and minimum matched FSC-AUC `0.9999999996`.  Higher thresholds
lost too much reuse: a 100 ms cold fill (`12804725`) took 559 seconds and its
warm replay was only halfway through after 191 seconds (`12804793`); a 50 ms
fill took 400 seconds (`12805102`) and its warm replay was only through
iteration 6 after 220 seconds (`12805118`).  Explicit user cache settings
continue to override the default threshold.

The following same-contract experiments were rejected rather than promoted:

- Increasing `--image-batch-size` from 500 to 2500 slowed RECOVAR from
  `169.02 s` to `188.39 s` in controlled jobs `12800289` and `12800287`.
  Both trajectories passed, at runtime ratios `7.57x` and `8.61x`.
- Reusing compact noise sums took `177.8 s` and reduced the minimum trajectory
  FSC-AUC to `0.999979` in job `12798272`.
- Aggressive 64-image tail inflation produced a `10.37x` runtime ratio and
  minimum FSC-AUC `0.999979` in job `12798383`.
- Grouping pseudo-halves preserved the trajectory (`0.9999999996` minimum
  FSC-AUC, exact assignments) but slowed RECOVAR to `320.5 s` versus RELION's
  `22.9 s` in job `12799750`.
- Decomposing arbitrary image chunks into a power-of-two shape palette caused
  several cold shapes at once and had not completed iteration 1 after 75
  seconds.  Diagnostic job `12800685` was cancelled at 88 seconds and the
  source experiment was removed.
- Padding every semantic row count to a multiple of eight did not amortize
  the added work.  Diagnostic job `12801069` reached a 33.5-second final
  iteration before the experiment was cancelled and removed.
- A native indexed compact-pair scorer eliminated the large pair-by-pixel JAX
  gathers and was bitwise-equal to the existing CUDA reduction in focused GPU
  tests (`12803423`).  Its complete trajectory was also unchanged (minimum
  FSC-AUC `0.9999999996`, exact assignments), but shape-specific cold
  compilation raised end-to-end time from 150 to 157 seconds in job
  `12803430`.  The source experiment was removed; its audit was job `12803697`.
- Routing compact noise and norm-residual statistics through the existing
  combined JAX graph passed three focused math/path tests (`12804345`), but the
  enlarged graph spent more than 220 seconds compiling before reaching the
  first sparse pass.  Job `12804329` was cancelled after the one-time CUDA
  rebuild plus 235 seconds of candidate execution, and the route change was
  removed.
- Replacing row-mapped RELION posterior CUDA primitives with batched
  exponentiation, CUB sort/scan, and division was bitwise-equal in four H100
  tests (`12806770`).  After a cold build/fill (`12806860`), its complete warm
  replay (`12807074`) still took 151 seconds versus the 150-second control;
  summed iteration time improved by only 0.44 seconds.  The three-API CUDA
  surface was removed because that 0.3% internal gain did not improve
  end-to-end wall time.

The accepted next step is one shared native dual weighted-sum boundary.  It
keeps reconstruction and masked-noise contractions independent, supports the
production `complex64` and `complex128` paths, and skips exactly-zero posterior
weights before loading translated pixels.  Representative H100 tests are
bitwise-equal to the two existing JAX matmuls in both precisions (`12808038`,
three tests).  The opt-in K=4 run `12808109` completed in 142 seconds and its
eight-checkpoint audit `12808277` passed with exact assignments and minimum
FSC-AUC `0.9999999996`.  The promoted default replay `12808596` completed in
144 seconds versus the accepted 150-second control; audit `12808723` passed
with exact assignments and minimum FSC-AUC `0.9999999996`.  Summed iteration
time fell from 135.26 to 126.84 seconds in the controlled run.  The K=1 default
replay `12808650` also passed at minimum cross-engine FSC-AUC
`0.9999999999`, and K=2 replay `12808673` passed with exact assignments and
minimum FSC-AUC `0.9999999999`.  Explicit environment overrides can still
disable or force the native boundary; by default it is restricted to the
exact RELION Gaussian, x-half, noise-accumulating CUDA contract.

A non-sparse prototype of the same boundary is rejected: job `12807978`
retained the trajectory (audit `12808240`, minimum FSC-AUC `0.9999999996`,
exact assignments) but took 179 seconds because every output thread loaded
all 116 translated pixels, including zero-posterior rows.

The next implementation boundary is therefore a broader native fused
score/posterior/noise path with a deliberately small executable-shape palette,
or ahead-of-time/persistent compilation reuse across recurring bucket shapes.
Either candidate must beat the clean 500-image control including cold startup,
preserve the unchanged FSC/assignment gates, and then pass the real and
100,000-particle scale gates before becoming the default.
