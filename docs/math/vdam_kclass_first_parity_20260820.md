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
| K=4 default, clean native-dual qualification | PASS | 0.9999999972 | 1.0000 | 17.24 s | 144.14 s | 8.36x |
| K=4, 25 iterations | PASS | 0.9999999679 | 1.0000 | 41.47 s | 402.48 s | 9.71x |
| K=4, 25 iterations, clean native-dual qualification | PASS | 0.9999998963 | 1.0000 | 42.55 s | 387.07 s | 9.10x |
| K=4, real 10076, 10,000 particles | PASS | 0.9999987694 | 0.9988 | 71.65 s | 705.67 s | 9.85x |
| K=4, real 10076, fresh-reference native-dual replay | FAIL | 0.9999625893 | 0.9949 | 156.85 s | 544.00 s | 3.47x |
| K=4, real 10076, frozen-oracle native-dual replay | PASS | 0.9999987693 | 0.9988 | frozen | 502.13 s | not comparable |
| K=4, 100,000 particles, 256 pixels | PASS | 0.9999851527 | 0.9998 | 267.06 s | 1960.97 s | 7.34x |
| K=4, 100,000 particles, 256 pixels, native dual | PASS | 0.9999996968 | 1.0000 | 268.79 s | 1590.93 s | 5.92x |
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

Moving the unique compact-pair scatter into that native boundary is also
rejected.  Its representative `complex64` and `complex128` dense-scratch and
weighted-sum outputs were bitwise-equal to JAX (`12809345`), and the complete
trajectory remained exact (`12809513`, minimum FSC-AUC `0.9999999996`, exact
assignments).  However, the full run `12809374` took 166 seconds versus the
144--147-second accepted default replays.  Materializing the dense scratch as
an observable custom-call output prevented the useful producer/consumer
fusion and increased synchronization and weighted-sum traffic, so the
prototype was removed.

The next implementation boundary is therefore a broader native fused
score/posterior/noise path with a deliberately small executable-shape palette,
or ahead-of-time/persistent compilation reuse across recurring bucket shapes.
Either candidate must beat the clean 500-image control including cold startup,
preserve the unchanged FSC/assignment gates, and then pass the real and
100,000-particle scale gates before becoming the default.

## Clean native-dual qualification and immutable-reference policy (2026-08-23)

The isolated clean-head qualification at `4c908639` passed the default, long,
and scale gates in jobs `12809684`, `12809685`, and `12809687`.  The scale run
is the material performance result: RECOVAR fell from `1960.97 s` to
`1590.93 s`, and the same-GPU ratio fell from `7.34x` to `5.92x`, while minimum
FSC-AUC improved to `0.9999996968` and minimum hard-class agreement to
`0.99999`.

The fresh real-data pair `12809686` exposed a reference-repeatability problem,
not a hidden candidate-state change.  RECOVAR's hard-class arrays are exactly
the same as the earlier passing run at every written iteration.  The two
independent eight-thread RELION references differ in 47 of 10,000 final class
labels; the four stable candidate/reference disagreements and those 47 native
repeat disagreements are disjoint, producing the observed 51-label
`0.9949` result.  Both map trajectories remain above the fixed `0.999` FSC-AUC
gate.  Two one-thread RELION repeats in job `12811072` reduce the final native
label variation to one particle but do not make the GPU maps bytewise
deterministic, confirming that GPU accumulation, not only host threading,
belongs to the oracle variance.

The pair harness therefore supports `--reference-pair-report`.  This mode
validates the complete scientific command, fixture hashes, executable hash,
checkpoints, and unchanged `0.999`/`0.995` thresholds before reusing immutable
RELION artifacts.  It deliberately records no runtime ratio.  Fresh
same-physical-GPU pairs remain the performance protocol, while correctness
replays use one frozen oracle so a runtime change cannot pass or fail because
RELION generated a different atomic-reduction realization.  Native
repeatability remains a separate visible audit; no threshold or baseline is
changed.

The first clean frozen-oracle replay is job `12812053` at `40f355d6`.  It
passes every checkpoint with minimum FSC-AUC `0.9999987693` and minimum
assignment accuracy `0.9988095`.  Its report explicitly records
`reference_mode=frozen_pair_report`, `runtime_comparable=false`, and a null
runtime ratio.  RECOVAR itself took `502.13 s`; that standalone duration is
not divided by the historical oracle duration.

## Fused sparse M-step diagnostic (2026-08-23)

A broader native M-step prototype kept the compact posterior internal to one
custom call: it allocated private dense scratch, scattered each unique compact
pair once, ran the accepted ordered reconstruction/noise contractions, and
returned only the five M-step statistics consumed by Python.  Focused H100
tests covered `complex64`/`complex128` images and `float32`/`float64` CTF
weights; the final test job `12812588` passed all seven cases.  An initial
production run (`12812361`) also exposed and closed the missing `float64` CTF
contract before the timed replay.

The complete retry `12812440` took 142 seconds, indistinguishable from the
accepted native-dual run rather than faster than it.  Its audit `12812586`
passed every checkpoint with exact class assignments, exact artifact topology,
and minimum FSC-AUC `0.9999999996`.  Thus the private scratch did preserve the
scientific trajectory, but allocation, scatter, and extra native reductions
only moved work across the boundary; they did not reduce end-to-end runtime.
The prototype is retained as a diagnostic commit/revert on the isolated probe
branch and is not promoted.  Runtime work should remain focused on the much
larger scoring/posterior/noise path and executable-shape/compilation costs.

A proposed skip of class-local fine log-Z reductions is rejected after tracing
the live normalization contract.  Focused external-evidence tests passed
`4/4` (`12813292`) and the K=4 trajectory stayed exact (`12813192`, minimum
FSC-AUC `0.9999999996`), but the timed default does not supply
`normalization_log_evidence`; it supplies RELION's float32 `sum_weight` through
a separate input.  The changed condition therefore did not execute in jobs
`12812973`/`12813191`, and the apparent `140.60 s` warm result is ordinary
run-to-run variation rather than attributable speedup.  Moreover, when
external absolute evidence is supplied, class-local log-Z remains observable
in the returned per-class statistics.  The diagnostic patch was reverted and
no performance claim is made from these jobs.

## Whole-run GPU trace (2026-08-23)

Nsight Systems job `12813510` traced the accepted clean K=4 default over all
eight iterations.  The complete CUDA-kernel table accounts for only about
`5.25 s` of device work.  The largest kernels were indexed backprojection
(`32.5%` combined), coarse projector/diff2 (`22.7%` combined), posterior CUB
radix sort (`7.4%`), the accepted dual weighted sums (`5.0%`), and compact
fine diff2 (`2.7%`).  Optimizing any one of those kernels cannot explain or
close a roughly 140-second process runtime.

The launch/API topology is instead dominant: the trace observes 303,975
runtime-level `cudaLaunchKernel` calls, 205,588 driver-level `cuLaunchKernel`
calls, 222,569 device-to-device copies, 33,024 stream synchronizations, and
5,478 fatbinary loads.  These API layers overlap and must not be summed as
independent launches, but they expose the executable/dispatch fragmentation.
The posterior alone launched 20,800 exponentiation kernels, 20,800 division
kernels, 20,800 scans, and more than 53,000 radix-sort kernels.  Nsight may
disable some XLA command-buffer execution while profiling, so the trace is not
a timing baseline; its kernel totals and call multiplicities are the causal
result.  The next runtime boundary must collapse complete bucket-group graphs
and shape-specific executable/module loads, not micro-optimize another device
kernel.

The matching pinned-RELION trace (`12813996`) rules out raw launch count as
the explanation by itself.  RELION performs about `6.01 s` of GPU kernel work,
very close to RECOVAR's `5.25 s`, while issuing roughly 581,000 runtime-level
kernel launches and 795,000 stream synchronizations—more than RECOVAR in the
profile.  It nevertheless completes the unprofiled pair in about 17 seconds.
RELION loads 704 CUDA modules and drives work from eight host threads; RECOVAR
loads 5,478 modules and constructs/deserializes shape-specific XLA executables
through serial Python bucket boundaries.  Summed CUDA-API time is not directly
comparable because RELION's calls overlap across threads.  The actionable gap
is therefore executable construction/loading and serial host orchestration,
not device arithmetic or the mere existence of many CUDA launches.

An isolated-cache control quantifies the first-run penalty directly.  Job
`12814306` populated 8,102 cache files and took `523.91 s`, versus roughly
144 seconds after the shape executables are warm.  Per-iteration times ranged
from 37.95 to 118.33 seconds as new current-size/bucket shapes appeared.
Forcing eight-way LLVM module compilation (`12814307`) was rejected and
cancelled: iteration 1 regressed from `57.69 s` to `62.66 s`, because these
modules are too small for within-module splitting to amortize its overhead.

Holding pass-2 at the full 8,320-pixel half spectrum (`12814575`) was also
rejected.  It attempted to reuse one pixel shape across iterations, but the
extra arithmetic and the remaining current-size/bucket axes made iteration 1
slower (`60.01` versus `57.69 s`) and iteration 2 substantially slower
(`62.95` versus `52.17 s`); it was cancelled during an already-slower
iteration 3.  A useful executable palette must therefore preserve bounded
windowed work and make variable scientific extents runtime metadata inside a
broader native boundary, rather than padding JAX arrays to the full image.

A broader joint diff2/posterior diagnostic then moved the per-image global
minimum, exact float32 prior-addition order, class concatenation, exponentiation,
CUB sort/scan, adaptive pruning, and normalization into one custom call.  Its
focused H100 test was bitwise equal to the decomposed chain for both pruning
modes (`12815392`, 5 passed), and a full K=4 run plus trajectory audit passed
(`12815620`, `12815824`; minimum FSC-AUC `0.9999999971882169`, assignment
accuracy `1.0`).  It was nevertheless rejected and reverted: warm wall was
`142.05 s`, while an isolated-cache run (`12815943`) took `523.18 s` versus
the `523.91 s` control.  The control and diagnostic iteration boundaries were
also effectively identical.  Posterior/scoring assembly is therefore not the
dominant compilation boundary; the next probe must collapse the much larger
set of eager bucket preparation and M-step primitives rather than adding a
posterior-only FFI.

A one-iteration fresh-cache compile trace (`12816179`) makes that boundary
concrete.  JAX reported 1,975 XLA compilations totaling `87.58 s` (summed
compiler durations; the logging run itself took `138.85 s`).  The largest
named totals were `compute_noise_block` (`5.08 s`), `gather` (`5.04 s`), the
native compact weighted-sum wrapper (`4.71 s`), `_reduce_sum` (`4.51 s`),
`compute_norm_residual_per_image` (`4.40 s`), `_where` (`4.34 s`), and
`broadcast_in_dim` (`4.08 s`).  There were 292 broadcast compilations, 133
gathers, 118 where operations, 104 multiplies, and 100 adds in this single
iteration.  In contrast, all ten fine-posterior compilations summed to only
`3.09 s`.  The next runtime target is therefore a whole bucket-group
preparation/M-step boundary (or an equivalently coarse JIT/native region),
with runtime extents, rather than another isolated scoring primitive.

Disabling compact active-row extraction was tested as a synchronization
ablation because it removes data-dependent host materialization at the cost of
more dense GPU arithmetic.  The trajectory remained qualified (`12816709`;
minimum FSC-AUC `0.9999999971650473`, assignment accuracy `1.0`), but a
same-node warm pair (`12816795`) was conclusive: active rows took `139.30 s`
and dense rows took `139.38 s`.  The saved synchronization and added dense work
cancel, so the existing active-row default remains unchanged.

Warm bucket-group timing (`12816984`, one iteration, `71.11 s` total) locates
the remaining pass-2 wall more precisely.  The six K=4 compact-pair groups
took `49.23 s`: `13.67 s` scoring, `7.83 s` preparation, and `20.14 s`
M-step/noise, with the latter containing `7.34 s` weighted sums and `11.35 s`
noise accumulation.  Fetch/build/stats were small.  Merely retaining all four
raw diff2 partitions on device to remove their D2H/H2D staging preserved
science (`12817189`; minimum FSC-AUC `0.9999999995961244`, assignment `1.0`)
but regressed warm wall to `142.38 s` (`12817124`, versus the same-node
`139.30 s` control).  That diagnostic was reverted: overlapping K raw tensors
adds memory pressure without removing the downstream orchestration.  A useful
native boundary must consume score operands through weighted/noise statistics,
not just relocate raw intermediates.

## Fused M-step/noise promotion boundary (2026-08-23)

The accepted next boundary combines the two native compact weighted-sum calls
with the dense residual-noise and norm reduction in one JIT region.  It also
keeps the active-row mask on device and reuses the resulting shell sums in the
later noise update.  The path is deliberately narrow: it requires native dual
weighted sums, exact RELION Gaussian scoring, x-half M-step accumulation,
noise accumulation, the residual-term formulation, and no competing compact
noise reuse.  `RECOVAR_SPARSE_KCLASS_FUSED_MSTEP_NOISE=0` remains an explicit
diagnostic opt-out.

Focused H100 tests established exact equality for all seven returned arrays
(`12817496`, 8 passed), and the primary implementation passed its focused GPU
suite (`12818513`, 4 passed).  The following end-to-end gates used the same
RELION trajectory audit as the earlier parity work:

| Gate | Result | RECOVAR wall | Comparison |
| --- | --- | ---: | --- |
| K=4 promoted default, environment unset (`12819567`) | minimum FSC-AUC `0.9999999972918416`, assignment `1.0` | `132.58 s` | both halfsets recorded `sparse_kclass_fused_mstep_noise: true` |
| K=4 same-node A/B (`12817941`) | minimum FSC-AUC `0.9999999996693576`, assignment `1.0` | `133.97 s` | `6.60%` faster than the `143.43 s` control |
| K=4 isolated cache (`12818120`) | minimum FSC-AUC `0.9999999972527686`, assignment `1.0` | `497.82 s` | `4.8-5.0%` faster than the established isolated controls |
| K=2 warm (`12818196`) | minimum FSC-AUC `0.9999999994495288`, assignment `1.0` | `112.95 s` | `7.92%` faster than the prior native-dual baseline |
| K=4 parameter matrix (`12818160`) | all 9 cases passed | `112.83-308.33 s` | diameter, healpix, offset, oversampling, seed, and tau2-fudge variants |
| K=4 repeatability (`12818404`) | RECOVAR repeat minimum FSC-AUC `0.9999999972264677`, assignment `1.0` | `132.14-133.65 s` | both cross-engine audits passed |
| K=4, 25 iterations, warm (`12818736`) | minimum FSC-AUC `0.999999899069889`, assignment `1.0` | `354.33 s` | `12.0%` faster than the prior qualified run |
| 10076, 10k particles (`12818403`) | minimum FSC-AUC `0.9999979947608962`, assignment `0.997619` | `536.08 s` | `24.0%` faster than the prior qualified run |
| 100k particles at 256 px (`12818405`) | minimum FSC-AUC `0.9999981078201251`, assignment `0.99999` | `1474.53 s` | `24.8%` faster than the prior qualified run |

The first long-trajectory invocation paid a new-shape compilation cost
(`471.66 s`, `12818402`), while its warm replay produced the improvement above.
The isolated-cache gate nevertheless improved, so the boundary does not trade
warm speed for a general cold-start regression.  Direct-scatter, posterior-only
FFI, raw-diff2 retention, and dense-active-row alternatives remain rejected;
none of those diagnostic implementations is present in the promoted path.

## Post-promotion host/runtime probes (2026-08-23)

The promoted default's one-iteration stage profile (`12819946`) took
`48.43 s`.  Its warm second half spent `11.9 s` in compact pass 2: `2.98 s`
scoring, `4.55 s` M-step/noise, `1.70 s` preparation, and `0.97 s` image
fetching across the three pair-size groups.  The matching Python profile
(`12820244`) took `56.91 s` under profiling and found 1,851 top-level JAX
cache misses.  They accounted for `22.15 s` cumulatively, including `11.18 s`
lowering, `5.06 s` compile/cache loading, and `7.86 s` waiting on internal
locks.  Compact pass 2 accounted for `36.58 s`, initial-state construction
`7.56 s`, and coarse significance `6.74 s`.  This confirms that the remaining
gap is fragmented host/XLA orchestration across several stages, not one
dominant GPU kernel.

Several apparent ways to reduce that fragmentation were measured and rejected:

| Probe | Result | Reason rejected |
| --- | ---: | --- |
| 1 GiB projection-gather cap (`12820124`) | `82.53 s`; warm pass 2 `23.5 s` | split execution into 72-75 buckets |
| 4 GiB cap (`12820075`) | `68.09 s`; warm pass 2 `16.8 s` | fewer buckets but more padded rotation work |
| 8 GiB cap (`12820074`) | `70.45 s`; warm pass 2 `25.9 s` | still greater padding and adjoint work |
| 2048-row bucket quantum (`12820193`) | `64.50 s`; warm pass 2 `14.1 s` | extra executable shapes outweighed reduced padding |
| Image power inside the fused M-step graph (`12820428`) | `65.58 s`; warm pass 2 `17.1 s` | enlarged graph increased compile and memory pressure |
| Image-power-only JIT (`12820650`) | `53.12 s`; warm pass 2 `12.6 s` | synchronization/shape overhead exceeded saved dispatch |
| Residual-subtraction JIT (`12820809`) | `49.13 s`; warm pass 2 `12.0 s` | statistically neutral to the `48.43 s` control |
| Residual subtraction inside fused M-step/noise (`12821679`, `12821729`) | `123.76 s` without a barrier; `122.96 s` with an exactness barrier | warm pass 2 regressed from `11.40 s` to `32.17 s`; the unbarriered graph also changed five shell sums by up to `1.91e-6` (`12821680`), while the barrier restored exactness (`12821713`) but not runtime |
| Exact/per-class rotation-signature grouping (`12822101`, `12822200`) | cancelled after only 7 chunks in `49 s` and 9 chunks in `57 s` | rare one-particle and class-specific shapes expanded the compile palette |
| Joint-p95 two-palette grouping (`12822398`, `12822531`, `12822806`) | one iteration `49.57 s`; default-8 cold `379.07 s`, warm `120.26 s` | reduced compact chunks from 86 to 38 and warm pass 2 from `11.60 s` to `8.86 s`, but data-dependent remainder batches recompiled every iteration; science passed at minimum FSC-AUC `0.9999999995298583` and exact assignments |

The EM-style coarse score/prior/evidence JIT boundary also preserved the full
trajectory (`12821223`; minimum FSC-AUC `0.9999792267663643`, assignment
`0.9998`) but took `133.47 s`, versus `132.58 s` for the promoted default.
It was therefore reverted along with the smaller diagnostic JITs.  The next
runtime boundary must remove a larger serial host region while retaining the
current memory planner; simply wrapping individual eager regions in JIT or
changing bucket size does not improve end-to-end wall time.

The promoted source also completed the full fast suite after these runtime
probes (`12819918`): `7002 passed, 109 skipped` in `45:32`.  Every rejected
probe above was isolated to the diagnostic worktree; none is present in the
tracking branch.

The two-palette result identifies the next actionable boundary despite being
rejected as an implementation: its warm default-8 replay was `9.3%` faster
than the promoted `132.58 s` run.  The EM stack already has a static image-axis
padding contract (`pad_batch_data_ctf_and_valid_mask` plus unpadded
postprocessing in `local_em_engine.py`).  The next VDAM probe should reuse that
contract so common, remainder, and outlier chunks keep a fixed batch dimension
from the first trajectory, rather than compiling each data-dependent remainder
shape.
