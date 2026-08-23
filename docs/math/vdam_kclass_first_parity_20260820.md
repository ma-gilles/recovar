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
| Joint-p95 two-palette grouping (`12822398`, `12822531`, `12822806`) | one iteration `49.57 s`; default-8 isolated `379.07 s`, warm `120.26 s` | reduced compact chunks from 86 to 38 and warm pass 2 from `11.60 s` to `8.86 s`; science passed at minimum FSC-AUC `0.9999999995298583` and exact assignments |

The EM-style coarse score/prior/evidence JIT boundary also preserved the full
trajectory (`12821223`; minimum FSC-AUC `0.9999792267663643`, assignment
`0.9998`) but took `133.47 s`, versus `132.58 s` for the promoted default.
It was therefore reverted along with the smaller diagnostic JITs.  The next
runtime boundary must remove a larger serial host region while retaining the
current memory planner; simply wrapping individual eager regions in JIT or
changing bucket size does not improve end-to-end wall time.

The promoted source also completed the full fast suite after these runtime
probes (`12819918`): `7002 passed, 109 skipped` in `45:32`.

The EM-style fixed-image-axis follow-up was then tested in three progressively
narrower forms.  Padding the complete sparse pass (`12823711`, `12823776`)
took `74.04 s` for one iteration and `306.43 s` for default-8.  Deriving the
fixed capacity only from the memory planner (`12824076`, `12824126`) took
`79.51 s` and `289.28 s`.  Padding only scoring/posterior and removing dummy
rows before the scientific reductions (`12824324`, `12824380`) took `87.40 s`
and `364.99 s`.  All three full trajectories converged to the same weaker
boundary (minimum FSC-AUC about `0.99997923`, assignment `0.9998`), so every
padding variant was rejected and removed.

The no-padding two-palette grouping was reassessed against the appropriate
isolated and warm controls.  Its `379.07 s` isolated run is about `24%` faster
than the accepted implementation's `497.82 s` isolated run, and its `120.26 s`
warm replay is `9.3%` faster than the accepted `132.58 s` run, while preserving
the substantially stronger `0.9999999995298583` full-trajectory FSC boundary
and exact assignments.  A default-route probe (`12824580`) passed one-iteration
parity at minimum FSC-AUC `0.9999999962612028`, assignment `1.0`.

Full qualification supports making the palette the default.  The cold,
concurrently executed 25-iteration pair (`12824959`) remained scientifically
strong at minimum FSC-AUC `0.9999998919425003` and exact assignments.  A
source-matched explicit-library warm replay (`12825285`) took `344.07 s`,
versus `354.33 s` for the accepted planner.  The 10k real-data pair
(`12825021`) passed at `0.9999620906` / `0.9955` and took `453.62 s`, versus
the prior `536.08 s`.  Most importantly, the 100k-particle/256-pixel gate
(`12825039`) passed at `0.9999981063` / `0.99999` and fell from `1474.53 s`
to `1083.94 s` (runtime ratio `4.32x`).  All nine K=4 parameter cases passed;
the K=2 default passed at `0.9999999999` / `1.0`; and the repeatability audit
showed RECOVAR repeat-vs-repeat `0.9999999996` / `1.0` (the weaker fresh
cross-engine repeat was caused by the independently measured RELION reference
variation).

The initial fast-suite failures were an invalid CPU-node qualification whose
traceback was a missing `nvidia-smi`, not a planner regression.  The palette is
therefore default-on in the tracking branch; setting
`RECOVAR_SPARSE_KCLASS_GROUP_PAIR_BUCKETS_BY_ROTATION_SIGNATURE=0` restores the
prior planner.  Fixed image-axis padding remains rejected and absent.

The warm long-run artifact was subsequently audited at all 26 written
checkpoints, not only at the summary boundary.  Every checkpoint passed, with
minimum FSC-AUC `0.9999998915607998` at iteration 25 and exact hard-class
assignments and topology throughout.  Expanding the common palette cutoff
from joint p95 to p100 was rejected in job `12826199`: it changed the common
high-rotation group from 23 chunks at batch 72 to 98 chunks at batch 18 plus
tails (110 signatures total), increasing fragmentation instead of removing
it.  The accepted joint-p95 source-matched one-iteration replay (`12826387`)
took `43.92 s` and `43.05 s` on its two passes.

## InitialModel allocator qualification and deterministic fallback (2026-08-23)

An eight-iteration profile (`12826561`) isolated an iteration-8 allocator
pressure spike: expectation time reached `28.93 s`, including `13.05 s` and
`8.72 s` in the two fused sparse pass-2 halves, while all non-expectation
stages remained near one second per iteration.  On the same H100,
`TF_GPU_ALLOCATOR=cuda_malloc_async` reduced end-to-end wall from `124.65 s`
to `120.12 s` (`12826886`).  The scoped command-bootstrap implementation then
completed in `119.18 s` (`12827578`, `4.4%` faster than control); its full
trajectory audit passed with minimum FSC-AUC `0.999999999601246` and exact
assignments.  The scale gate rejected making this the default: 100k/256 job
`12828312` exceeded 33 minutes in RECOVAR without reaching iteration 8, more
than twice the prior qualified `1083.94 s` wall.  InitialModel therefore keeps
the existing allocator by default.  An existing caller `TF_GPU_ALLOCATOR`
still wins, and `RECOVAR_INITIAL_MODEL_CUDA_ALLOCATOR=cuda_malloc_async`
selects the scoped experimental allocator before JAX initialization.  The
resolved allocator is recorded in both dry-run and native-options JSON.

The exact pre-allocator fast suite (`12825815`) found one intermittent
CUDA-disabled test difference: two of five image-power shells differed by at
most `0.00048828`.  The test explicitly disables RECOVAR's custom CUDA path,
but its fallback shell binning still lowered to unordered GPU scatter atomics.
The fallback now performs a fixed per-shell reduction only when
`RECOVAR_DISABLE_CUDA=1`; production CUDA execution is unchanged.  The
original test plus 20 repetitions passed (`12828060`), and the complete
sparse-pass2 test file passed with `173 passed, 1 skipped` (`12828073`).

## Default-GUI robustness trajectory expansion (2026-08-23)

The original fixed-12 scorecard already covered baseline, very high noise,
anisotropic poses, no-CTF, contrast/noise-scale variation, image offsets,
severe outliers, high resolution, 10k, and production-scale inputs.  A new
independent 12-cell suite, `vdam-k1-robustness-v1`, reuses the remaining
immutable EM fixtures and keeps the same default K=1 InitialModel parameters.
Every cell runs a complete eight-iteration RELION/RECOVAR pair on one physical
GPU and audits checkpoints 0/1/2/4/8.  It adds 20%, 25%, 30%, and 70% outlier
fractions; junk particles; uniform, anisotropic, and Kent pose distributions;
white, radial, low, and high noise; no-CTF; combined contrast/shift stress;
small-N; and a 10k Kent/radial cell.

Slurm array `12831190` passed all 12 cells from source head `4c3d44ed`.  Across
all cases and checkpoints, the minimum cross-engine signed FSC-AUC was
`0.9999999916047712` (gate `0.999`), and the worst RECOVAR-minus-RELION GT
FSC-AUC delta was `-1.3592646956467336e-6` (gate `-0.002`).  Schedules and
artifact topology were exact throughout.  The immutable aggregate report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k1_robustness_4c3d44ed_20260823/robustness_suite_summary.json`
(SHA-256 `1e2f68e21a0e2fe5c7416b59eea7148bdde938610f2ff904091f6f776aee41a8`).

The small-cell runtime ratios ranged from `3.84x` to `11.70x`.  These jobs make
startup/JIT overhead visible rather than establishing scale throughput, so
runtime remains open and is evaluated separately on the 100k/256 paired gate.
Per the EM-only validation contract, the unrelated repo-wide long-test shards
were stopped and will not be used as VDAM acceptance evidence.

Four complementary stress cells were then extended to 25 complete iterations
in suite `vdam-k1-robustness-long-v1`: anisotropic poses with 25% outliers,
Kent poses with 20% outliers, 70% outliers, and the 10k Kent/radial-noise
fixture.  Slurm array `12831621` passed all four cells from source head
`f783c977` at checkpoints 0/1/2/4/8/12/16/20/25.  The minimum cross-engine
FSC-AUC was `0.999994364355`, and the worst GT FSC-AUC delta was
`-6.356e-6`.  The aggregate report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k1_robustness_long_f783c977_20260823/robustness_long_suite_summary.json`
(SHA-256 `6ba86baab4873c8347ca744f040896a57a70078ba1a1c8a312e03f14495a133d`).

The final default-allocator 100k/256 K=4 pair (`12830078`) also passed all
nine checkpoints with exact assignments and minimum per-class FSC-AUC
`0.9999995483248948`.  Its runtime was not acceptable: RECOVAR took
`1952.44 s` versus RELION's `266.60 s` (`7.32x`).  This is materially slower
than the earlier qualified `1083.94 s` RECOVAR run on the same fixture, so
large-scale throughput is both the remaining product gap and a variability
problem.  Correctness is not the blocker at this boundary.

## 100k lazy-I/O runtime probes (2026-08-23)

The next isolated probes tested whether the scale gap came primarily from
bucket-ordered particle reads.  Each one-iteration job used the same
100k-particle, 256-pixel, K=4 fixture and default scientific settings.

| Probe | Slurm job | Iteration wall | Result |
| --- | --- | ---: | --- |
| Unmodified lazy control | `12829578` | `909.88 s` | reference |
| Sort each bucket fetch into source order | `12829984` | `861.05 s` | rejected: only `5.4%` faster and changes accumulation order |
| Direct indexed source fetch | `12831416` | `811.37 s` | rejected: `10.8%` faster is insufficient for the product runtime target |
| Direct fetch plus bulk MRCS memmap gather | `12831942` | `956.81 s` | rejected: `5.2%` slower than control |
| Monotonic/coalesced MRC reads with exact output-order restore | `12835500` | stopped after half 1 | rejected: first-half pass 2 regressed from `325.1 s` to `406.4 s` (`+25.0%`) |
| Existing eager mode (`--no-lazy`) | `12829869` | `424.32 s` | diagnostic only: substantially faster for iteration 1 but high host-memory cost |

The direct, memmap, and monotonic-read implementations were reverted in the
isolated runtime worktree after measurement and were never promoted to the
tracking branch.  The monotonic reader kept returned arrays in the exact
requested order and passed all 54 focused image-loader tests, but scattering
the sorted reads back into that order cost more than the seek locality saved
on the locally staged stack.  Its dependent science audit was canceled because
the performance gate had already failed.
The complete eight-iteration eager run (`12830515`) took `1406 s` wall and
peaked at `94,388,912 KiB` RSS.  Its eight profiled iterations summed to
`1330.67 s`; the final iteration alone took `423.19 s`.  Against the matched
RELION wall of `266.60 s`, this is still `5.27x`, and the memory footprint is
not suitable as the default GUI behavior.  Its independent all-checkpoint
trajectory audit (`12832441`) passed iterations 0 through 8 with minimum
per-class FSC-AUC `0.9999995412448371`, exact assignments, and exact artifact
topology.  Eager loading therefore remains an explicit user option rather than
a runtime fix: the science is sound, but the scale/runtime tradeoff is not.

These measurements narrow the next implementation target: preserve lazy
startup and bounded memory, but cache the much smaller current-size
preprocessed Fourier representation used repeatedly by coarse/fine bucket
passes.  Any such cache must first prove exact trajectory parity and a clear
end-to-end gain in the isolated runtime worktree before promotion.

### Rejected full-resolution host-cache probes

Two follow-up probes tested whether retaining full-resolution particle and CTF
arrays on the host could remove repeated lazy reads.  They were implemented
only in the isolated runtime worktree and were never promoted.

| Probe | Slurm jobs | Iteration wall | Peak RSS | Science audit |
| --- | --- | ---: | ---: | --- |
| Cache only the fine pass | `12832948`, `12832949` | `656.73 s` | `69,987,588 KiB` | **fail** at iteration 1: minimum FSC-AUC `0.22945024013005907`; sampled assignments `1.0` |
| Share one cache across coarse and fine passes | `12834534`, `12834535` | `567.30 s` | `89,282,364 KiB` | **fail** at iteration 1: minimum FSC-AUC `0.2294502570722795`; sampled assignments `1.0` |

The shared-cache implementation first passed a complete small K=4 trajectory
(`12834179`) at checkpoints 0 through 8 with minimum FSC-AUC
`0.999999999568837`, exact assignments, and exact artifact topology.  The
100k/256 audit nevertheless demonstrated that this small test was not a
sufficient scale qualification.  Deterministic 256-row image/CTF sentinels
also matched fresh source reads, which localizes the failure beyond simple
cache indexing or sampled pose/class selection and into scale-dependent image
or reconstruction state.  The full-resolution cache is rejected regardless of
that root cause: the shared variant still takes `2.13x` RELION's complete
eight-iteration wall for only one RECOVAR iteration and reaches `89.28 GB`
RSS, close to eager mode's `94.39 GB`.

The next runtime candidate must remain bounded-memory and must avoid retaining
the full-resolution raw dataset.  The leading options are a scientifically
complete current-size processed-Fourier cache (including every normalization
and noise operand needed to preserve the RELION trajectory) or ordered,
bounded asynchronous prefetch.  Each candidate will be rejected immediately
unless it passes a small complete trajectory followed by the independent
100k/256 iteration-1 audit before any longer scale run.

### Rejected bounded bucket-prefetch probe

A one-bucket asynchronous prefetch kept exact requested particle order and at
most one additional batch resident.  Its small complete K=4 trajectory
(`12836093`) passed checkpoints 0/1/2/4/8 with minimum FSC-AUC
`0.999979226703295`, minimum assignment accuracy `0.9998`, and exact artifact
topology.  Wall time improved only from `428.12 s` to `418.79 s` (`2.2%`).
The scale job `12836388` then made the rejection unambiguous: its first
50,000-particle pass-2 half took `369.88 s`, versus the frozen `325.1 s`
control (`+13.8%`).  It was canceled during half two after `13:43` and peaked
at `35,129,516 KiB` RSS; dependent audit `12836389` was canceled.  The probe
was reverted in the isolated runtime worktree and is not present here.

## Real-data expansion and configurable surfaces (2026-08-23)

The second frozen real-data case uses 10,000 balanced EMPIAR-10073 particles.
The source is a legacy single-table STAR, so fixture preparation now promotes
its microscope metadata to a RELION 3.1 optics table.  The first promoted
fixture exposed an important conversion bug: legacy pixel origins were left
in deprecated columns.  RELION ignored those columns once optics existed and
searched around zero, while RECOVAR preserved centers as large as 55 pixels.
Job `12836331` therefore matched bootstrap (`0.9999999667` FSC-AUC) but
diverged at iteration 1 (`0.6215650`) and later.  This is an understood fixture
contract failure, not an accepted parity result.

The immutable v3 fixture converts legacy pixel origins to active Angstrom
columns using the derived `1.400011 A/px` optics value and resets stale
posteriors from the prior refinement.  It retains the identical balanced
source-index selection.  Its STAR SHA-256 is
`eb15efcb63d3496c6a7b39e966ec3a0d992b78223c7b11ef1c5cbdd61355ee1a`;
the source-index file SHA-256 remains
`4417136987d8c2501bcac53b795d25aa7821d0b4f2a09ac91d385d8224a15bea`.
Corrected pair `12836889` then passed the map and subset contracts at every
checkpoint: minimum FSC-AUC was `0.9999998165326929`, and the iteration-1
200-particle subset was exact.  Winning poses/translations were exact at
iterations 1, 2, and 4.  The strict overall result remains **FAIL** because
iteration 8 has one of 1,000 selected particles displaced by one pixel and a
different particle reaches `0.043531` absolute Pmax error (limit `0.01`).
RECOVAR took `197 s` versus RELION's `22 s` (`8.95x`).  The audit now derives
visited topology from the frozen selected identities and RELION's selected
output prefix, rather than stale positive class labels retained on unvisited
legacy rows.  This leaves a narrow iteration-8 point-reference winning-state/
Pmax boundary to diagnose; no threshold was relaxed.

The first matched full-from-seed RELION repeat (`12837509`) shows that both
10073 exceptions are inside RELION's own repeatability boundary.  Relative to
the canonical RELION run, the repeat changes exactly the same
`71231@particles.256.mrcs` winner by one `1.40001 A` translation pixel, and
the largest Pmax change (`0.043372`) occurs on the same
`73054@particles.256.mrcs` particle flagged by the RECOVAR comparison.  The
repeat map FSC-AUC is `0.9999998168949595`.  The repeated RELION translation
is the RECOVAR-selected translation to output precision.  This result does not
retroactively weaken the point-reference audit: it establishes that a robust
real-data gate must separately report exact point-reference agreement and
agreement within a measured RELION self-repeatability envelope.

A third real-data fixture is now frozen from the 64,174-particle filtered
EMPIAR-10345 STAR and its canonical 256-pixel stack.  The source contains
complete pose/CTF metadata but no halfset or pixel-size column.  Fixture
preparation therefore requires explicit opt-in to deterministic balanced
synthetic halves and records the explicit canonical CTF pixel size
`1.345 A/px`.  The 10,000-particle STAR SHA-256 is
`e5d9f77ff38d0e5137412892e7cc7591ba09265fb928b649cdeab58208a540f5`;
the selected-index file SHA-256 is
`9f812a7bfd6bb9dd071786143a501c6803f6c05541faee36c7d6e07f0aa787a3`.
Strict pair `12837220` passed bootstrap and iterations 1, 2, and 4 with exact
selected-particle topology and zero winning pose/translation mismatches. At
iteration 8, 999 of 1,000 selected particles still agree, but
`3138@particles.256.mrcs` selects a different pose and translation and reaches
`0.014321` absolute Pmax error. The minimum map FSC-AUC remains
`0.9999974063783914`, so the map contract passes while the particle-state
contract correctly keeps the overall result at **FAIL**. RECOVAR took `178 s`
versus RELION's `22 s` (`8.09x`). Together with the single late boundary in
EMPIAR-10073, this independently reproduces a rare iteration-8 decision-boundary
problem on two real datasets; the next diagnostic captures the exact pass-2
operands for those particles rather than weakening the state thresholds.
Unlike 10073, the matched full-from-seed RELION repeat (`12837510`) is exact in
all 1,000 selected poses and translations at iteration 8; maximum Pmax drift is
only `0.000097`, and repeat map FSC-AUC is `0.99999999805772`.  The 10345
particle therefore remains a genuine RECOVAR-vs-RELION fine-score boundary,
not a reference-repeatability exception.

The public CLI and GUI continue to resolve defaults from one
`GuiInitialModelDefaults` object.  The previously hard-coded scientific
schedule is now configurable end to end with `--grad-ini-frac`,
`--grad-fin-frac`, `--grad-em-iters`, `--stepsize`, and `--mu`; the GUI shows
the same controls under Advanced InitialModel controls.  Defaults remain the
RELION GUI values `0.3`, `0.2`, `0`, `0.5`, and `0.9`.  Focused CLI, schedule,
fixture, and GUI command-builder validation passed 79 tests, followed by the
two direct schedule-routing tests.  Frontend compilation could not run in the
current worktree because Node/npm is not installed; no repo-wide RECOVAR or
GUI acceptance suite was substituted for the VDAM-focused validation.
