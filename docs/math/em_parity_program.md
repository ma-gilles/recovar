# EM / RELION Parity Program Board

This file preserves quantitative gates and historical program records. The
active cleanup milestone and next check live in
[the EM status page](../development/em_status.md). Keep permanent rules in
`recovar/em/AGENTS.md`, detailed dated evidence in
`docs/math/relion_parity_agent_notes.md`, and completion records in
`docs/math/em_parity_best_metrics.md`. Historical next actions below are not
current task instructions.

## Objective

First achieve near-perfect RELION quality parity for supplied-map K=1
auto-refine and K=4 3D classification. Then optimize to near RELION speed while
holding the accepted quality checkpoint. Treat native InitialModel/VDAM parity
as the next product milestone rather than mixing it into the first closure.

## Current VDAM quality priority — September 9

The user provisionally accepts up to **2× RELION runtime** while prioritizing
quality: first establish short-iteration matched-input score/posterior/state
parity, then evaluate final shellwise FSC/FSC-AUC against GT and RELION across
the required scope, including Hungarian-matched K4 and robustness. Once their
causes and numerical bounds are established, late discrete trajectory differences
are diagnostic; unexplained mismatches are not excused as noise. Exact parity
remains preferable. No numerical tolerance or baseline is changed, and the
long-term speed objective remains. This allowance is not a measurement of
current-source representative speed or a completed quality gate. See the
[user-policy handoff](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/pr179_coordination/handoffs/vdam_quality_priority_20260909.json).

## VDAM active experiment — 2026-08-20

### 2026-09-01 fixed-capacity call-0 correctness gate

The first shared fixed-capacity executor gate is correctness-only and remains
default-off.  It compares the mature, explicitly disabled-selector, and fixed
call-0 arms on the same deterministic K=1 score-only call through the single
shared `run_local_bucket_big_jit` wrapper.  The captured boundary includes the
complete prepared operand set, scores, `log_Z`, best score/argmax, maximum
posterior, posterior masses, exact sample/rotation support masks, significant
counts, row counts, hard assignments, and public `RelionStats`.  Both float32
and float64 companion lanes run in alternating same-process repeats, followed
by an independent uninstrumented production-topology comparison.

Because the present seam substitutes byte-validated operands immediately
before the identical numeric call, its acceptance contract is bitwise exact;
nonzero drift is unexplained and fails.  A future reduction-order optimization
may use small nonzero envelopes only after a separate production speed gate
shows a material win, and only with exact discrete decisions/support plus
repeat-bounded, non-growing float32 drift and a float64 companion at least
three orders tighter.  This gate cannot claim speed or promote a default.

The gate audit exposed that `donate_argnums=(4, 5)` donated correction and
projection-mean inputs even though the implementation comment names the
loop-carried `Ft_y`/`Ft_ctf` accumulators.  Commit `13bfcce4a` corrects the
mapping to signature positions `(7, 8)` and adds structural caller guards.
The old mapping emitted an unusable donated-buffer warning in the CPU dry run;
the corrected mapping does not.  Independent review returned GO at sealed head
`d880da3d0`.  H100 job `13288282` then passed `7/7` focused tests and all eight
captured plus twelve uninstrumented production comparisons on
`della-h21g4` / `GPU-099c0d77-bb85-f2e9-f628-148b733c9176`.  Every reported
score, centered-score, `log_Z`, best-score, posterior, and posterior-mass
delta is exactly zero in both precision lanes; all discrete outputs and
prepared operand bytes are exact.  The final source manifest remained
`986f6c733672425e87c8de6b8c7dec18e5d4085c663145d5e2510af6d0a72e6c`.
This closes the first correctness gate only.  A donation performance claim,
if any, requires a separate same-binary/toggle or crossed-commit H100 M-step
runtime and peak-memory gate; fixed-capacity speed and default promotion remain
unqualified.

### 2026-09-03 stable coarse logical-prefix gate

Validation scope is performance-only and default-off.  Stable coarse square
capacities remove recompilations across changing logical Fourier windows, but
the CUDA scorer must still execute only RELION's logical pixel prefix.  H100
trajectory job `13395892` established the opportunity and rejected the first
implementation scientifically: checkpoint construction through iteration 47
fell from `269.649 s` to `239.072 s` (`-11.34%`), while one of 200 particles
selected an adjacent translation and eight significant counts moved by one.
This result remains an explicit performance win and hard-state failure.

The first runtime-prefix primitive test (`13397454`) found `31--38%` of raw
score words different, at most two float32 ULPs, despite skipping every padded
pixel.  ABBA job `13397690` localized the change to native atomic admission:
the runtime scorer itself moved within the same two-ULP band, while posterior,
cutoff, support, significant count, and best pose were wordwise exact.  A
compact-row experiment then showed that even the accepted static scorer has
multiple legal raw-score realizations.  In split-kernel ABBA job `13398261`,
the two crossed static/runtime pairs were bitwise identical while both
static/static and runtime/runtime pairs differed by at most two ULPs.  Raw
score bitwise equality is therefore not a valid gate for this native-atomic
primitive.

The fail-closed replacement does not use a tolerance or an empirical repeat
diameter.  The default VDAM geometry maps 128 threads over 29 translations,
giving exactly four active lane partials for each score; inactive threads add
exact zero.  Enumerating all `4!` serialized atomic orders is therefore the
complete legal arithmetic set.  Focused H100 job `13398365` passes both the
rectangular and selected-source-16 scorers against that exhaustive set and
keeps posterior, cutoff, support, significant count, and argmin wordwise
exact.  The active next gate is the existing shared-state
off/on/on/off iteration-47 transition.  No fresh trajectory may start until
that gate preserves every hard decision/support field and bounds continuous
state by the established native-repeat contract; runtime must also retain the
compile-shape win after compact-row packing cost.

### 2026-09-03 selected-source and grouped-BPref performance decisions

Validation scope is performance-only and default-off.  The selected-source
coarse CUDA primitive is scientifically usable but is not a material runtime
optimization.  Same-process H100 job `13416867` prewarmed the selected-source
and mature static-full fused routes, then ran four balanced repeats of each at
the real GF46 iteration-34-to-35 `T=37` boundary.  The corrected analyzer
separately proves the executed coarse geometry (`logical/physical=50/64`,
`R=4608`, `T=37`, 200 particles) and the emitted next schedule (`size=68`,
HEALPix order 2, `R=36864`, `T=148`).  Routing, instrumentation, and every
hard-state field are exact.  Continuous map/statistic differences remain at
repeat scale and pass the existing two-times static-control plus
`4*float32-epsilon` policy; the stricter one-times diagnostic remains visible.

The isolated selected kernel is `2.346x` faster in dense geometry and `1.334x`
in compact geometry, but projection-cache, certificate, selector, and assembly
overhead consume nearly all of that gain.  Dense expectation changes from
`1.29746 s` to `1.26727 s` (`2.33%` faster), while its directly measured
cache-plus-hybrid seam is `7.60%` slower than the direct full-fused seam.
Compact expectation changes from `1.23086 s` to `1.17383 s` (`4.63%` faster),
and its direct seam improves only `1.34%`.  Peak HBM changes by only `+28 MiB`
and `-4 MiB`.  This misses the predeclared 10% materiality gate.  Do not run a
selected-source `0 -> 50` trajectory, compose the branch, or promote a default
unless a later topology removes the fixed overhead and first passes a new
focused materiality gate.

The first bounded/no-denominator grouped-BPref repair is also a review NO-GO
before GPU execution.  Although it reuses the mature shared VDAM FFI and its
focused helper tests pass, its advertised 1 GiB estimator omits wrapper-side
stable image/CTF/noise buffers, generated rotation/replay operands, and native
staging.  It also materializes the incoming bucket before flushing a previous
near-cap group, so actual live memory can be the prior group plus the next
bucket.  Fusion preserves logical source order but changes cross-bucket CUDA
stream/atomic chronology; numerical equivalence must therefore use the
repeat-controlled policy, never an exact-chronology claim.  Fusion-build time
is missing from accounted EM time.  The active repair must establish a true
wrapper-aware live-memory/preflush contract and production-stream tests before
any H100 gate.

The active performance hypothesis returns to execution topology: compare the
mature supplied-map EM controller and persistent RELION CUDA lifecycle against
InitialModel's remaining warm host/dispatch boundaries, and require a
code-supported lever with at least a 10% end-to-end upper bound before another
trajectory.  BPref-only fusion, selected-source scoring, ordinary callable
caching, larger stable quanta, and small kernel-tail changes are already below
that bar or rejected.

### 2026-09-03 x-half 80M-to-160M capacity escalation preregistration

The valid warm iteration-45 Nsight capture rules out resource caching alone as
the next 10% lever.  Across 11 x-half callbacks, CUDA array allocation/free,
stream creation/destruction, texture-object creation/destruction, and all but
the one lazy `cudaMalloc` outlier total only about 5--6 ms.  The matched warm
trajectory needs 6.297 s, or about 126 ms per iteration, to improve by 10%.
Any native session experiment must therefore aggregate useful work or remove
the surrounding dispatch topology; merely retaining textures, streams, or
linear scratch is insufficient.

The next bounded experiment instead escalates the already measured shared
EM/VDAM x-half work-unit capacity.  Job `13260950` changed the row-pixel cap
from 40M to 80M, reduced the frozen iteration-80 topology from five buckets to
three, and improved median full `0 -> 80` wall from `310.949` to `287.444 s`
(`8.18%`) with flat observed HBM.  The preregistered follow-up compares 80M
control against 160M candidate in the same warmed A/B/B/A harness.  At the
sealed iteration-80 layout, 160M reaches the existing upstream hypothesis cap
and predicts two buckets (`266 + 94` particles) instead of three
(`150 + 150 + 60`); larger row-pixel values cannot reduce that topology
without separately changing the upstream safety cap.

This is a performance-only rung.  Advance 160M to the repaired four-repeat
science oracle only if the median incremental speedup is at least 2%, both
paired speedups are at least 1%, the predicted three-to-two topology executes,
no allocation/error path occurs, and peak HBM does not materially rise.  A 2%
increment compounds with the sealed 40M-to-80M result to more than 10%; a miss
stops row-pixel-cap escalation.  The historical two-repeat trajectory analyzer
is retained diagnostically but cannot promote science or alter the frozen v3
score.

### 2026-08-31 late-trajectory one-iteration performance gate

Validation scope is diagnostic/performance-only: it cannot promote science,
change the frozen VDAM quality denominator, or change a production default.
The frozen `vdam-gf46` run at RECOVAR commit `984637b7d` took `4388.59 s`
versus native RELION's `480.65 s` (`9.13x`).  The gap becomes largest after
the search changes from HEALPix order 2 (`36,864` fine rotations) to order 3
(`294,912` fine rotations) while the gradient subset grows from 360 to 1000
particles.  At iteration 180, RECOVAR's checkpoint-mtime interval is about
`38.8 s`, while RELION reports `4.138 s` for expectation; at iteration 200
the corresponding values are `25.9 s` and `3.468 s`.

The candidate performance hypothesis is excessive host orchestration around
the order-3 search: signature/bucket/chunk JAX and FFI calls, materializations,
and synchronization boundaries may keep RECOVAR from matching RELION's
persistent CUDA execution context.  This remains a hypothesis until the
kernel/API trace measures invocation counts and GPU idle time.  In particular,
the iteration-180 model's prior mode is 1 but its rot/tilt/psi widths are all
zero; both engines therefore enumerate the full direction/psi grid.  The gap
must not be attributed to RELION using a local angular cone.  The bounded discriminator starts native
RELION and RECOVAR from the same hash-pinned `run_it180` optimiser/model/data/
sampling state, executes exactly iteration 181 on the same H100, and records
wall time, RECOVAR stage timers, and Nsight CUDA-kernel/API/NVTX summaries.
The runner must fail closed on a missing gradient moment, mismatched schedule,
dirty source, binary/hash drift, GPU UUID drift, or any output other than the
single requested next iteration.  No 200-iteration trajectory or broad test
suite is part of this gate.  The implementation target, if the trace supports
the hypothesis, is a persistent macro-batched CUDA path that retains images,
projector/cache state, candidate buffers, posterior reductions, and BPref
accumulators across a pool of particles with one or a few launches rather than
per-signature/per-chunk dispatch.

### 2026-08-31 late-GF46 controller-topology discriminator

The matched late-state profile rules out the shared fine-score CUDA primitive
as the cause of the remaining runtime gap.  At GF46 iteration 150 the exact
fine-diff2 kernel launches twice and consumes `27.44 ms`, while
`local.run_local_em_exact` consumes `9.796 s`.  Two changing
`jit_run_local_bucket_big_jit` programs compile for `5.114 s`, or `52.2%` of
that local wall.  InitialModel's run-global radix-4 bucket unification pads
captured layouts by `3.55x`--`16.0x`; exact per-size execution avoids that work
but creates `67`--`120` changing shape/launch groups and is therefore not a
candidate default.  The same trace observes `1,840` raw stack reads for `920`
active particles because pass 1 and pass 2 load independently; mature EM's
persistent raw-loader cache is not invoked by InitialModel.

The active bounded hypothesis is that the mature supplied-map controller's
physical-order macro-batching policy can remove most VDAM-specific shape churn
without changing candidate arithmetic or BPref particle order.  The first
candidate will extract one shared consecutive padded-batch planner, replace
InitialModel's run-global maximum with chunk-local radix-2 maxima aligned to
the native pool-of-three order, and reuse EM's persistent raw-image loader
cache.  Admission requires identical candidate support, physical order, and
best state at a fixed late checkpoint.  Floating-point accumulator differences
may be nonzero only when they are bounded, repeatable, within the native-repeat
envelope, and do not alter the trajectory basin; bitwise equality is not a
requirement for a material speedup.  The candidate must also materially reduce
unique XLA programs, padded rows, raw reads, and same-H100 steady-state wall
time.  The existing CUDA scorers/posterior/Wavg kernels remain authoritative.
No broad RECOVAR suite or long trajectory is part of this discriminator.

Current continuation (2026-08-24): K=1 GUI-default qualification is running
from immutable production head `1e499798c`.  Completed 200-iteration cases
`vdam-gf01`--`vdam-gf11` all fail the unchanged `0.999` cross-engine FSC-AUC
gate, first at iterations 31, 82, 68, 72, 33, 72, 57, 93, 35, 73, and 93.  Their
minimum cross-engine FSC-AUC values are `0.66914`,
`0.68797`, `0.55649`, `0.49680`, `0.66096`, `0.53190`, `0.57190`, `0.94109`,
`0.04739`, `0.72508`, and `0.61528`.  Independent
native-RELION triplets show that `vdam-gf02`, `vdam-gf03`, and `vdam-gf05`
leave their sampled native-repeat envelopes; `vdam-gf03`, `vdam-gf04`, and
`vdam-gf06`, `vdam-gf07`, `vdam-gf09`, and `vdam-gf11` also miss the `-0.002`
GT-delta gate.  These are active parity
failures, not tolerance candidates.  A bounded-memory change carried the
20,000-particle severe-outlier/radial-noise `vdam-gf20` trajectory beyond its
former iteration-110 OOM boundary through all 200 iterations at about 12.5 GB
RSS.  Its frozen audit is a real parity failure: first failure at iteration 30
(`0.99127` cross-engine FSC-AUC), minimum/final cross-engine FSC-AUC `0.15622`,
and final RECOVAR-minus-RELION GT FSC-AUC `-0.01207`.  Runtime remains
an independent failure: `vdam-gf06`, `vdam-gf09`, `vdam-gf10`, and
`vdam-gf11` take `5.61x`, `6.51x`, `2.91x`, and `4.29x` RELION wall time,
respectively, while the
20,000-particle `vdam-gf20` case takes `7.07x`.  The earlier completed small
cases span `2.36--2.88x`.

### 2026-08-24 VDAM float32 fine-posterior default boundary

The complete eight-repeat, 200-particle native ensemble invalidated the earlier
two-repeat classification of 33 stable scoring defects: pooled RECOVAR
posterior error is within the native repeat diameter in both halves, with
`99.43%`/`99.62%` of coordinates inside the native envelope.  Two genuine
stable-repeat outliers, `144@1127` and `179@115`, then closed projected
references, score weights, priors, raw-particle preprocessing/translation,
fine diff2, and the final float32 log-weight table to the live RECOVAR scores.

The first BPref captures appeared to expose missing production wiring:
InitialModel's local big-JIT diagnostic saved float64 reconstruction
probabilities while the already qualified EM CUDA posterior remained behind a
default-off switch.  Focused H100 jobs `12910179` and `12910180` passed 14/14
tests and replayed the saved candidate scores through RELION's exact float32
`expf`, Policy800 CUB sort/scan, divide, and significance path.  For both
stable outliers, raw weights, sum weight, threshold, normalized posterior,
reconstruction mask, and pruned reconstruction probabilities are bitwise
identical to native RELION (2,720/2,720 and 512/512 values).  Padding the
candidate tables to 118,784 slots does not change any native result, so
candidate order/zero padding are excluded at this boundary.

The bounded production change makes the source-matched float32 posterior the
default for K=1 RELION x-half reconstruction while retaining the environment
switch as an explicit `0` rollback.  A follow-up trace found that the
big-JIT M-step was using the new float32 tensor, but its observational capture
incorrectly rebuilt `reconstruction_probs` from the generic float64
`debug_probs`.  Commit `3188c7e95` makes the capture rebuild the same exact
tensor from its returned score boundary and fail closed if the mask differs.
H100 unit job `12910923` passes 2/2 focused tests.  Fresh production captures
`12911029` and `12911030` now retain float32 reconstruction probabilities in
both halves.  Independent audits `12911086` and `12911087` pass 16/16 tests
each and prove the live InitialModel probabilities and masks bitwise exact to
the stable native outliers: 2,720/2,720 and 512/512 values, zero error.

The posterior boundary is therefore closed in production, but the whole
iteration-1 M-step is not: fresh-repeat raw accumulator relative-L2 remains
about `8.91e-6`/`1.00e-5` for data and `2.04e-6`/`2.49e-6` for weights.  The
accepted-posterior iteration-2 discriminator `12911356` completed `0:0` in
63 seconds against the identical frozen native data/component capture used by
the earlier control.  It leaves the causal raw-noise error effectively unchanged:
`+1.60060e-5` becomes `+1.59974e-5` (only `0.054%` smaller).  The AA and XA
signed errors remain `-3.87171e-6` and `-9.98757e-6`.  Its initially reported
support-mass improvement was later superseded because that diagnostic mixed
the generic posterior with the production mask; corrected job `12912049`
measures the native-dtype reconstruction tensor and gives `+2.28470e-5`.
Together with the direct operand decomposition, the unchanged AA/XA boundary
confirms that the remaining iteration-2 cutoff/noise failure is propagated
iteration-1 reference/BPref arithmetic rather than the accepted iteration-1
posterior correction.  The next causal gate returns to that production
accumulator/reference boundary; one
representative 0..200 trajectory follows only after a bounded discriminator
materially closes it, before promotion to the full 22-cell trajectory matrix.
No generic RECOVAR suite is part of this gate.

The source-faithful fused residual/scatter arm was then rebased onto the exact
posterior head in an isolated worktree (`a1c05c69f`).  Build/provenance job
`12911670` completed `0:0`, and focused H100 job `12911766` passed all five
selected source-order, scatter-boundary, routing, and posterior-interaction
tests.  Matched M-step job `12911801` rejects the combination.  Against the
same native accumulator capture, the fused and ordinary exact-posterior
accumulators are effectively indistinguishable: half-1 data relative-L2 is
`9.24987e-6` versus `9.24949e-6`, and weight is `2.30800e-6` versus
`2.30983e-6`.  The fused reconstructed reference is worse (`2.51023e-6`
versus `2.22717e-6`).  Frozen-native cutoff reruns `12912049` and `12912050`
likewise reduce the shell-15 raw-noise signed error by only `1.75107e-7`, from
`+1.60217e-5` to `+1.58466e-5` (`1.09%`).  This is not material closure, so
the fused arm is rejected without a 200-iteration trajectory and remains
unpushed.

Those reruns also corrected an observational gap in the cutoff harness.  The
score dump previously stored only the generic float64 posterior plus the
production reconstruction mask, so its reported support mass did not measure
the float32 M-step tensor.  Commit `0065204d2` records
`reconstruction_probs` at native dtype and makes the analyzer prefer it.
Focused Slurm job `12911970` passes 4/4 tests; Ruff on the touched files,
py_compile, and diff checks pass.  Direct AA/XA/noise terms were already
production values and are unchanged by this diagnostic correction.

Commit `e17ca882e` then makes the boundary harness's native thread count
explicit and records it in provenance.  Focused H100 guard job `12912220`
passes.  Independent full-schedule `--j 1` iteration-1 jobs `12912266` and
`12912267` both complete `0:0` in 38--40 seconds, but one host thread does not
make the native GPU accumulator deterministic.  Native repeat relative-L2 is
`8.91445e-6`/`8.32194e-6` for data, `2.02839e-6`/`1.86395e-6` for weights,
and `1.33450e-6` for the reconstructed reference.  The corresponding RECOVAR
repeat distances are only `1.91104e-7`/`2.13117e-7`,
`1.05739e-7`/`1.14044e-7`, and `5.61788e-7`.  Each paired cross-engine distance
is `0.77--1.36x` its native-repeat distance.  Host thread count is therefore
rejected as the missing mechanism: the remaining iteration-1 BPref distance
is already at the device-side native atomic/reduction-order envelope, even
under `--j 1`.  Exact equality to one arbitrary native accumulator realization
is not a valid point gate; the next causal gate must test whether matching the
native repeat distribution, rather than one draw, is sufficient to preserve
the iteration-2 cutoff and long trajectory.

The source-faithful fused arm also fails that repeat-distribution gate.  After
qualified rebuild `12912585`, fresh full-schedule repeat `12912674` completes
`0:0` in 52 seconds and is paired with the earlier fused capture `12911801`.
Native repeat relative-L2 is `9.71544e-6`/`9.82039e-6` for data and
`2.45064e-6`/`2.50817e-6` for weights, whereas fused RECOVAR remains much more
deterministic at `1.72101e-7`/`1.52556e-7` and
`7.96698e-8`/`7.88732e-8`.  Reconstructed-reference repeat distances are
`1.92968e-6` native and `6.36606e-7` fused.  Its cross-engine distances remain
`0.79--1.30x` one native-repeat distance, but fusion does not reproduce
RELION's device-order distribution.  The arm remains rejected without a
200-iteration run.  The next bounded implementation gate is the still-unmatched
native accumulator storage topology: RELION atomically updates three disjoint
real, imaginary, and weight arrays, while the candidate's complex accumulator
interleaves real and imaginary values in one `float2` allocation.

That storage-layout gate is also null.  Isolated commit `975500ffc` splits the
fused accumulator into RELION-shaped real, imaginary, and weight allocations,
aliases all three through the FFI, and recombines the data losslessly after the
kernel.  H100 build `12913028` completes `0:0`; focused job `12913108` passes
4/4 source, numerical-interior, native-y-boundary, and routing tests.  Paired
full-schedule jobs `12913115`/`12913122` both complete `0:0`.  Candidate repeat
relative-L2 remains only `1.52327e-7`/`1.87188e-7` for data and
`8.24795e-8`/`8.99974e-8` for weights, versus native
`9.33658e-6`/`1.07903e-5` and `2.43710e-6`/`2.12937e-6`.  Cross-engine
distances remain `0.82--1.15x` one native-repeat distance.  Disjoint storage
does not reproduce native atomic-order variance, so the commit remains
unpushed and receives no trajectory.  The next implementation boundary is the
native kernel's resource/occupancy topology (nine shared Euler values, inline
projection, and exact compiled control flow), not host threads, source
statement order, fused atomics, particle launch order, or output allocation
layout.

Source statement order alone is now rejected as sufficient.  Isolated commit
`99681a33b` completed all 200 frozen `vdam-gf01` iterations (Slurm
`12879549_1`): first strict failure is iteration 73, minimum/final cross-engine
FSC-AUC is `0.82343`/`0.98281`, and runtime is `3.94x` RELION.  Its GT-quality
gate remains within tolerance (minimum delta `-0.00153`), but hard-state
divergence begins at iteration 24 and reaches all 1000 particles by iteration
200.

The active numerical hypothesis is the remaining kernel boundary: RELION
forms the VDAM residual and scatters BPref in the same per-particle
`cuda_kernel_backproject3D_SGD<DATA3D=false>` launch.  Isolated commit
`45f794c62` fuses those operations, retains VDAM's native FFTW Nyquist and
negative-y/x=0 semantics rather than inheriting the generic EM scatter's
boundary convention, and removes the complex residual intermediate.  Focused
CUDA build `12883311` and source/interior/native-boundary GPU gates `12883340`
pass.  Executable inspection stopped the first full submission `12883623_1`
at iteration 73 because its sparse buckets had never selected the fused
target; those maps are quarantined as a source-order repeat.  Follow-up commit
`f98530a42` carries raw VDAM operands through the established physical-particle
sparse packing boundary.  Bounded job `12885370_1` completes with exact
iteration-1 particle state and three compiled fused-target executables; the
discriminating full frozen trajectory is `12885473_1`.  Executable inspection
confirms three compiled `cuda_relion_vdam_mstep_fused_x_half` targets.  Its
partial audit remains essentially exact through the old iteration-31 failure
boundary (cross-engine FSC-AUC `0.999999999878`) with zero divergent particle
states at iterations 1, 8, 16, 20, 24, and 31.  The complete audit rejects
fusion alone: the first strict map failure is iteration 73 (`0.99878735`, GT
delta `-0.00062945`), the minimum cross-engine FSC-AUC is `0.82180983` at
iteration 127, and a separate minimum GT delta of `-0.00236361` at iteration
142 also fails.  Final cross-engine FSC-AUC is `0.979021996`; RECOVAR takes
`2922.56` seconds versus RELION's `794.48` seconds (`3.68x`).  A second
isolated worktree tests
the remaining shared boundary by projecting the reference inline in the same
per-particle CUDA launch.  Isolated commit `c80a1b754` passes its CUDA build
(`12887299`), focused zero-projector GPU equivalence gate (`12887554`), and
bounded iteration-1 diagnostic (`12887823`: exact particle state,
cross-engine FSC-AUC `0.999999999959`, GT delta `-1.50e-8`).  Its full frozen
gf01 discriminator is Slurm `12887981_1`.  The executable cache contains the
inline-projector target, and its complete 0--200 audit rejects this variant.
Its first strict failure is iteration 31 (cross-engine FSC-AUC `0.99756656`,
GT delta `-0.00068389`), materially earlier than the fused preprojected
control's iteration-73 failure.  Minimum/final cross-engine FSC-AUC is
`0.66418465`/`0.85468779`; minimum GT delta is `-0.00496323`; hard-state
divergence begins at iteration 19.  RECOVAR takes `2931.63` seconds versus
RELION's `810.62` seconds (`3.62x`).  Inline projection therefore worsens both
trajectory stability and quality without resolving runtime.  The
experimental commits remain unpushed.

The authoritative full-schedule boundary is the sealed job `12869234`, not a
continuation.  It used RELION's true 200-iteration schedule and exact
iteration-32 optimiser plus the pinned `0.322510` perturbation.  Pose and
translation assignments match for all 3,000 particles through iteration 32;
iteration 33 first differs only for `1003@particles.128.mrcs` (pose error
`3.75` degrees, translation error `3.00` Angstrom, Pmax absolute error about
`2e-5`).  Its native and candidate posterior supports are exactly the same
298 tuples.  The native top-pair log odds are `+0.0004577651`, while RECOVAR
reverses their order at `-0.0001525879` (error `-0.000610353`).  Replaying
only the live noise-derived score weights moves that pair by
`+0.00048828125`; replaying only the live reference moves it by
`+0.0001220703125`; the live image contributes exactly zero to the pair.
Incoming iteration-32 `sigma2_noise` relative L2 is `1.994818e-5`, and the
top-pair raw-score residual is `-0.0005912781`.  Noise accumulation is
therefore the dominant causal defect, with a smaller inherited reference-map
contribution.  This also explains why fused scatter and inline projection do
not close the trajectory.

The earliest deterministic precursor is already visible at iteration 2,
shell 15: native raw numerator `0.124609` versus candidate
`0.1246245131`, a `+1.55e-5` error, while the candidate sum-weight ratio is
`0.999999589`.  Independent native repeat `12890438` follows the same true
200-iteration schedule.  Audit `12890696` confirms all 3,000 poses and
translations are identical between the two native runs through iterations 1
and 2.  Their iteration-2 shell-15 raw totals are identical at the native
capture's printed precision; across per-particle direct-residual rows the
relative L2 is `4.80e-7`, but signed shell-15 differences cancel to less than
`1.1e-19`.  The RECOVAR numerator error is consequently outside the observed
native-repeat floor.  A source-order Wavg experiment (`4a7aa71f`, unpushed)
passes its two focused ordering tests (`12891138`), but the paired iteration-2
jobs reject it as a repair: shell-15 raw numerator is `0.1246246435` in
`12891483`, versus `0.1246247254` for the same-head control `12891607` and
`0.124609` natively.  It removes only `8.20e-8`, about `0.52%` of the error,
so no 200-iteration trajectory was spent on it.

The causal search has moved one boundary earlier.  Native StoreWavg particle
`1140` already has a fine-posterior relative L2 error of `4.44e-7` at
iteration 1.  Focused operand audit `12891847` covers all 480 fine candidates
and 596 score pixels.  Candidate rotations, projected references, and score
weights are bit-exact; native raw `diff2` is reproduced exactly by both the
native-shift and fused-translation replays.  Replacing only the image operand
reproduces the live centered score residual (RMS `3.10e-5`, maximum
`1.25e-4`), while reference-only and weight-only replays are zero.  The
iteration-1 target is therefore the corrected particle-image formation before
fine scoring, not Wavg particle reduction or another projector/scatter
composition.  A diagnostic-only capture of that operand is the next bounded
experiment.

That bounded capture is now complete.  Diagnostic commit `5df279bdb` passes
the focused dump and big-JIT plumbing tests, then true-schedule iteration-1
job `12892595` records the pre-correction FFT, pixel correction, and corrected
score image for particle `1140`.  GPU audit `12892638` finds the live internal
product bit-exact at all 596 score pixels, but only 486/596 final corrected
pixels match native RELION (relative L2 `2.58e-8`, maximum `1.22e-4`).
Reconstructing RELION's XFLOAT correction from its captured RFLOAT CTF, with
the established RECOVAR sign conversion, leaves 114 one-ULP differences
(relative L2 `3.86e-8`, maximum `9.54e-7`).  The pre-correction and correction
errors partially cancel, so changing the reciprocal path alone is not yet an
accepted fix.

The same sealed native artifact already contains the missing uncorrected
post-optics Fourier image, so no rebuilt RELION executable is required.
Direct comparison finds 115/596 pre-correction score pixels unequal
(relative L2 `2.61e-8`, maximum `1.53e-5`).  Audit `12893369` makes the
upstream boundary exact: normalized/shifted real pixels match 16,384/16,384;
native pre- and post-optics Fourier arrays match 760/760; and applying
RECOVAR's FFT to the captured native masked real image matches native
760/760.  The current deterministic block-first soft mask instead matches
only 2,454/16,384 masked pixels (relative L2 `1.21e-9`), with its background
mean two float32 ULP below native; that alone becomes a `2.92e-8` Fourier
error.  The first causal operation is therefore the CUDA soft-mask background
reduction.  Private build attempts `12893019`, `12893150`, `12893173`,
`12893203`, and `12893255` all stopped during CMake dependency discovery and
produced no executable or scientific evidence; that branch is abandoned.
Failed setup jobs `12893490`, `12893600`, `12893669`, `12893744`, and
`12894645` produced no science.  Valid same-H100 panels `12893945` and
`12894797`, with audits `12894421`, `12894863`, and `12895553`, establish the
repeat boundary.  For exact iteration-1 normalized input, stock RELION spans
15 float32 ULP in the soft-mask background.  The deterministic block-first
value is inside that range and one ULP from its nearest sampled value.  Across
all 480 candidate scores, RECOVAR's nearest centered native RMS is `1.5012e-5`
versus native/native maximum `2.7816e-5`; 296 candidates are inside the
coordinatewise native envelope and 184 remain outside.  Native atomics add
schedule dependence without guaranteeing a closer trajectory, so no
production topology change is accepted.  The next discriminator is the
aggregate iteration-2 noise update under deterministic-lane and native-atomic
modes.

The aggregate discriminator rejects those modes.  Isolated commit
`f35844a9a` passes 4/4 focused routing tests; jobs `12896342`--`12896345`
measure shell-15 raw-numerator errors `+1.5602497e-5` (block-first),
`+1.5520540e-5` (native-lane), and `+1.5639750e-5` / `+1.5539167e-5`
(native-atomic repeats).  The best change is only `8.20e-8` or 0.53 percent,
so no 200-iteration candidate is warranted.  Controlled same-GPU gf01 repeat
panel `12880351` independently first fails candidate-repeat/native-repeat
equivalence at iteration 34; its worst repeat-floor margin is `-0.0452569`,
and the iteration-200 candidate/native repeat floors are `0.8348111` and
`0.8753012`.  RECOVAR repeat instability is therefore materially larger than
stock RELION's sampled repeat spread.

The 200-particle production big-JIT cutoff audit is complete for iteration 1
(Slurm `12897360`, commit `f343c5bb3`).  At shell 19, RECOVAR's direct
residual sums to `0.1293679055` versus native `0.1293675770`, only
`+3.28e-7`.  Its `AA` is lower for all 200 particles (sum error `-5.77e-8`),
while `XA` and inferred image-power sum errors are `+3.05e-7` and `+9.96e-7`;
those terms substantially cancel in the coupled direct residual.  Retained
support mass has relative L2 `2.90e-7`.  The known `+1.55e-5` iteration-2
shell-15 aggregate defect is therefore not already present at comparable
scale in iteration 1.  Job `12897360`
captured all 200 production dumps before a post-analysis module-invocation
setup error; the preserved artifacts analyzed successfully, and `d8314fb27`
repairs the runner without changing science.

The exact iteration-2 panel `12897664` completes cleanly and reproduces the
material shell-15 defect: direct-residual sum error is `+1.6006e-5`.  Its
`AA` error is `-3.8716e-6` and is negative for 197/200 particles; its `XA`
error is `-9.9872e-6`.  Because the coupled residual is `AA - 2*XA`, the
cross term contributes about `+1.9974e-5` and dominates, partly offset by
`AA`.  Inferred image-power error is only `-8.64e-8`.  Per-particle direct
error correlates `-0.9979` with `XA` error, while the one `2.11e-5`
support-mass outlier contributes only `-1.10e-7` direct error.  The first
material aggregate boundary is therefore the posterior-weighted
image/reference cross term, with a smaller systematic reference-power
deficit—not image-power formation, soft masking, or total support mass.  The
next bounded experiment replays native versus candidate posteriors against
the same captured operands for the largest `XA` contributors before changing
production arithmetic.

The replay closes that question.  Serial full-schedule native panel
`12899028` captures the eight largest contributors without the shared-prefix
race and reproduces every iteration-1/2 hard particle state.  Candidate versus
native posterior changes on identical native operands sum to only `4.65e-11`
for `XA` and `2.58e-12` for `AA`.  Substituting the production RECOVAR
reference projection accounts for `-2.2125802e-6` and `-5.6053283e-7`, while
the remaining replay residuals are `-3.63e-13` and `-4.00e-15`.  The
iteration-2 cutoff error is therefore propagated reference-state error.  The
active boundary moves back to the iteration-1 BPref/M-step accumulator, not
posterior formation or Wavg image/CTF/translation/reduction arithmetic.

The full iteration-1 M-step pair (`12901975`) makes that boundary quantitative.
Incoming `Igrad1` for both pseudo-halfsets and `Igrad2` are bitwise exact, while
the first nonexact state is the raw BPref accumulator: data relative L2 is
`9.0248e-6`/`9.8267e-6` and weight relative L2 is
`1.9306e-6`/`1.8982e-6`.  Independent native repeat `12902211` spans comparable
data differences (`8.406e-6`/`1.071e-5`), so magnitude alone cannot identify a
stable candidate defect.  Complete candidate contribution captures
`12904449` and `12904491` contain all `20,856`/`27,264` active rows.  A native
top-eight StoreWavg operand panel (`12906791`, `12906964`, and
`12907034`--`12907045`) has exact support for every particle; substituting the
candidate posterior into native image/CTF/translation/projector operands
closes data rows to `7.52e-8`--`1.24e-7` and weight rows to
`6.74e-8`--`9.69e-8`.  Fused StoreWavg arithmetic is therefore excluded for
those sources.

Native job `12907252` extends that panel to every one of the 200 selected
particles in one three-second, one-thread, true-200-iteration-schedule capture.
It records exactly 200 posterior tables, 200 rotation/translation/CTF bundles,
and 200 uniquely named accelerated unmasked Fourier images; the comma-truncated
predecessor `12907201` is rejected.  Complete aggregate replay `12907855`
passes 22/22 focused tests and covers all 200 identities with no incomplete
capture, no support mismatch, and no posterior argmax mismatch.  The native
versus candidate pre-scatter data error is `1.06453e-5`/`1.02516e-5`; using the
candidate posterior with otherwise native operands reduces it to
`9.75268e-8`/`9.76784e-8`.  Through the same RELION-double scatter, data closes
from `9.08620e-6`/`9.00707e-6` to `7.97858e-8`/`8.31516e-8`, and weight closes
to `1.47e-8`/`1.52e-8`.  This excludes residual formation and scatter as the
source of the panel gap and localizes it to posterior history.  Because this
panel and the saved production accumulator are different native repeats, it
does not yet distinguish a stable RECOVAR posterior error from native
accelerator variability.  The next exact discriminator is one uninterrupted
native iteration-1 run that captures both all 200 StoreWavg rows and its own
two raw BPref accumulators; the runner now has a fail-closed combined-capture
mode for that experiment.

That matched capture is complete.  Native job `12908104` records all 200
particle operands and both raw pseudo-halfset accumulators in one uninterrupted
six-second iteration-1 run.  Aggregate job `12908179` then uses the
corresponding same-run candidate accumulator for each half and passes 23/23
focused tests.  The posterior-induced shared-scatter gap and the actual
production-CUDA gap have data cosines `0.98359`/`0.97997` and weight cosines
`0.98292`/`0.97525`.  Projection onto the production gap accounts for
`96.66%`/`96.17%` of data and `96.33%`/`94.91%` of weight; the orthogonal
component is `17.73%`/`19.54%` and `18.03%`/`21.52%`, respectively.  Posterior
history is therefore the dominant causal component of the raw accumulator
error.  The residual is now explicitly bounded to shared-double versus
production-CUDA scatter topology.  Before changing scoring arithmetic, the
next cheapest experiment compares the complete candidate posterior against
both already captured 200-particle native repeats to determine whether it is a
stable out-of-repeat error or part of RELION's accelerated repeat spread.

That repeat comparison is complete in clean CPU-only Slurm job `12908333`
(4/4 focused tests).  All 200 identities close with zero support or argmax
mismatch.  Pooled candidate posterior error is `1.51727e-5`/`1.38283e-5`,
versus native-repeat error `1.47685e-5`/`1.72423e-5`; 57/100 and 67/100
candidate particles lie within their corresponding native-repeat distance.
Magnitude alone is therefore not a stable out-of-repeat discriminator.  The
residual direction is different: candidate-versus-native-A has cosine only
`0.47655`/`0.48200` with the native-B-versus-native-A residual and an
orthogonal component of `90.32%`/`70.27%` of the native-repeat norm.  More
importantly, 23/100 and 14/100 particles are bitwise identical across the two
native posterior captures but differ in RECOVAR.  The active bounded analysis
then compared pre-threshold normalized raw weights and centered log weights for
the same complete panel.  That split is decisive: raw-weight support is exact
for all 200 particles, while pooled pre-threshold errors reproduce the final
posterior errors (`1.51691e-5`/`1.38295e-5`).  Centered log-weight RMS is
`3.20382e-5`/`3.22615e-5`; 20/100 and 13/100 particles have exactly identical
normalized raw weights in both native repeats but nonexact candidate weights.
Thus exponentiation, normalization, and significance truncation are excluded
for 33 stable identities: their difference already exists in centered scoring
log weights.  The next bounded experiment captures detailed scoring operands
for the largest stable identities (`138`, `82`, `171`, and `125`, with matched
controls) and replays image, reference, CTF/noise, translation, and reduction
one at a time.  No production arithmetic change is justified before that
operand split.

The four-source operand panel and broader repeat calibration supersede that
two-repeat inference.  Provenance typo job `12908715` stopped before science;
corrected native panel `12908730` and GPU audits `12908752`--`12908755`,
`12908853`--`12908856`, and `12909056`--`12909059` all complete.  Across parts
138, 82, 171, and 125, projected references, score weights, and
orientation/translation log priors are exact.  Candidate likelihood-score RMS
is `1.54e-5`--`3.04e-5`.  Raw-particle replay `12909124`/`12909176` closes the
part-82 live centered score exactly after normalization, block-first soft
masking, FFT, correction, translation, and the fused scorer; only a constant
high-resolution offset remains.  Thus the saved preweighted debug image was
not an exact fused-score input and its apparent topology component is rejected.

Eight independent native part-82 captures (`12909205`) show why two repeats
were insufficient.  Native top-pair log odds span `6.109e-5`; RECOVAR is inside
that range and only `2.999e-8` from a sampled native value.  Its nearest
centered-score RMS is `2.953e-6`, versus native/native maximum `3.217e-5`, and
85.4% of coordinates lie inside the native envelope.  The deterministic
block-first background is one of the observed native values; native backgrounds
span four float32 ULP.

The complete eight-repeat, 200-particle, one-thread native ensemble
(`12909370`) and audit `12909383` now provide the authoritative iteration-1
calibration.  Pooled candidate posterior error is inside the native pair range
for both halves: nearest/max ratios `0.7459`/`0.7566`, with 99.43%/99.62% of
posterior coordinates inside the native envelope and 94/100 and 95/100
particles individually inside the native maximum.  Centered log-score pooled
nearest/max is `1.0118`/`0.9489`; the half-1 excess is only 1.18%, while 93/100
and 92/100 particles pass individually.  Raw support is exact for every
particle.  Therefore no global iteration-1 scoring change is justified.  The
active stable outliers are native part 144/original 1127 and native part
179/original 115 (zero sampled native score spread but nonzero candidate
error), followed by parts 2 and 3.  Detailed raw-particle replays for those
outliers are the next bounded discriminator.  Partial eight-thread panel
`12909277` is rejected because its marker raced 32 asynchronous dumps; it
contains only 168/200 particles and is excluded.

Capture submissions `12889423` and `12889446` remain rejected by fail-closed
provenance/native-state gates and provide no parity evidence.  Fresh paired
full-schedule job `12889537_1` follows a different native trajectory by
iteration 33 (its target posterior has 1,280 rather than 298 candidates), so
it is retained only as an independent 0--200 repeat, not as causal evidence
for the sealed particle-1003 boundary.

In parallel, the immutable 22-case
matrix, severe-memory case, and controlled same-GPU repeat panel continue as
evidence collection; generic RECOVAR full/long tests are deliberately outside
this EM-only validation scope.

The native InitialModel/VDAM implementation checkpoint is
`5a4c57839e50a46e47b2d25efb8d55744db04871`, on top of PR #158 head
`b10412ca`.  Iteration-0 bootstrap state is exact, and identity-aligned
iteration-1 particle pose/translation state is effectively exact, but the
frozen tiny trajectory still misses the map gates (iteration-1 cross FSC-AUC
`0.8274473`, GT delta `-0.0140749`).

The particle-0 projected-reference hypothesis is now rejected as the dominant
iteration-1 cause.  Against the current native RELION StoreWavg project panel,
VDAM's reference has the expected `-N^2` frame/normalization conversion.  The
converted reference differs by relative L2 `1.0028e-3`, but substituting the
native reference changes the actual gradient numerator by only `5.6474e-5`.
Replaying RELION's sequential translation accumulation with the current VDAM
reference changes that numerator by only `7.5372e-8`.  The previously isolated
scatter boundary remains machine precision (`4.78e-15` data, `2.87e-15`
weight).  Therefore the next measurable hypothesis is that the first material
iteration-1 divergence is in aggregate subset/pseudo-halfset routing or the
VDAM moment/reconstruction update, not the shared fine posterior, image/CTF
operands, projected-reference subtraction, or scatter arithmetic.

The apparent subset-routing mismatch was a row-coordinate mistake and is
rejected.  RELION writes its data STAR in lexicographic Experiment order, so
RELION output row `0..199` is not input particle-table row `0..199`.
Comparing stable `_rlnImageName` identities proves that the existing RECOVAR
and RELION iteration-1 subsets are exactly equal.  A table-order
counterfactual (Slurm job `12669580`, stopped after the result was known)
makes 79 identities differ in each direction and moves iteration-1
cross-engine FSC-AUC only from `0.8274473` to `0.8313125`, while worsening the
GT delta slightly to about `-0.0143244`.  The lexicographic Experiment-order
implementation is retained.  The frozen trajectory auditor now compares
iteration-1 subsets by `_rlnImageName`, preventing either input-row/output-row
coordinate system from passing accidentally.  The next measurable boundary
is pseudo-halfset membership after the already exact particle-identity,
posterior/scatter, and bootstrap controls.

The apparent pseudo-halfset failure was another unsupported inference and is
rejected.  A position-parity counterfactual made the nominal counts `100/100`,
but direct native-vs-RECOVAR BPref errors remained essentially unchanged and
the frozen trajectory did not improve.  Slurm job `12669867` produced
iteration-1 cross-engine FSC-AUC `0.8247451` and GT delta `-0.0137202`; the
minimum through iteration 8 was `0.6359682`.  The production global-particle
parity routing is restored.

The first direct aggregate boundary remains BPref.  Against native RELION's
captured iteration-1 arrays, production RECOVAR half-0/half-1 BPref data have
relative L2 `1.33211`/`1.33062` and cosine `0.10345`/`0.09585`; weights have
relative L2 `0.78793`/`0.83375`.  The position-parity counterfactual changes
these only marginally.  Downstream replay is not the cause: feeding RELION's
native post-`applyMomenta` data, moment-noise power, reference, and exact
parameters into the shared binding reproduces native `reconstructGrad` at
`7.10e-8` relative L2.

The particle-0 StoreWavg boundary is now closed with the actual unmasked
reconstruction image.  A first replay using RELION's masked scoring image was
invalid and produced numerator cosine `0.48068`; the analyzer now rejects that
operand by construction.  RELION's accelerated path does not expose the
unmasked Fourier image in the shared binary, so Slurm job `12671650` captured
the same pre-StoreWavg operand through the already compiled CPU hook.  Using
that image with the authoritative GPU posterior/projector gives numerator
relative L2 `0.0077293`, cosine `0.9999703`, and denominator relative L2
`0.0074343`, cosine `0.9999725` (analysis job `12671684`).  This rejects
per-particle image, posterior, CTF/noise, residual subtraction, and scatter
arithmetic as the dominant particle-0 cause.

The stable-identity StoreWavg panel also agrees.  For input rows `0`, `100`,
`277`, and `999` (native part IDs `0`, `4`, `199`, and `3`), numerator relative
L2 is `0.00774`, `0.00701`, `0.00449`, and `0.00987`, with cosines from
`0.999955` to `0.999992`; denominator results are comparable.  Slurm jobs
`12671889`--`12671892` used RECOVAR contribution captures from jobs `12671818`
and `12671819` and unmasked-image capture `12671784`.

That panel accidentally sampled only identities for which input-row parity and
RELION internal `part_id` parity agree.  Directly decoding RELION's frozen
iteration-1 `sorted_idx` and mapping through lexicographic Experiment order
shows `101/200` selected identities have different parities.  RELION source is
explicit that StoreWavg routes by internal `part_id % 2`; current RECOVAR routes
by original input-row parity.  The earlier position-parity trajectory is still
valid negative whole-run evidence, but it cannot override this demonstrated
first-boundary semantic mismatch while other aggregate defects remain.  The
active bounded experiment is therefore one deliberately mismatched identity:
native part ID `2` / input row `99`.  Capture its unmasked StoreWavg rows and
show both their arithmetic agreement and their opposite current half routing
before restoring Experiment-position parity as an independently tested fix.

That mismatched-identity experiment passed its arithmetic controls and failed
the routing control exactly as predicted.  Native part ID `2` / input row `99`
has numerator relative L2 `0.003556`, cosine `0.999994`, and denominator
relative L2 `0.001540`, cosine `0.999999` (Slurm job `12672040`), but the
production capture records it in RECOVAR half 2 while RELION routes even
internal part ID `2` to half 1.  RECOVAR now carries Experiment-position parity
alongside the lexicographic row mapping through shuffle, prefix selection, and
optics stable-sort.  Unit coverage pins the distinction.  The earlier
position-parity trajectory (`12669867`) remains evidence that this necessary
semantic correction does not by itself close aggregate or map parity; after
the focused validation ladder, the next boundary is native-vs-RECOVAR partial
accumulator sums in identical Experiment particle order.

A direct replay rejects missing x=0 Hermitian enforcement as the aggregate
cause.  Applying the existing shared RELION half-volume Hermitian helper to the
corrected-position RECOVAR accumulator worsens relative error and leaves data
cosines at only `0.1441`/`0.1136` and weight cosines at `0.6769`/`0.6530`.
Exhaustive simple axis swaps, flips, conjugation, and circular y/z shifts also
fail (best weight cosine below `0.717`; data below `0.17`).  No production
Hermitian or layout change is justified by that evidence.

The one-particle post-scatter boundary is also closed.  Slurm job `12672415`
moved the already matched particle-99/native-part-ID-2 rows through RELION's
CPU BackProjector in the same native layout.  Data relative L2 is `0.003511`
with cosine `0.9999940`; weight relative L2 is `0.001431` with cosine
`0.9999992`.  The scatter primitive and its native BPref layout are therefore
not the aggregate cause.

That closure exposed a separate adapter boundary.  The shared RELION x-half
M-step returns a public full cube by expanding native `(z,y,xhalf)` storage and
transposing it to RECOVAR `(x,y,z)`.  InitialModel then feeds that public cube
to the generic centered-slab converter without undoing the transpose and also
applies a projector-frame z flip.  A synthetic valid-Hermitian round trip has
relative L2 `1.3923` and cosine `0.02118`, which is the same signature as the
full aggregate mismatch.  Transposing the public cube back to `(z,y,x)` before
extracting the positive-x slab is exact (`0.0` relative L2).  The active
bounded fix is an explicit RELION-x-half-public-to-BPref inverse in the VDAM
adapter, with a lossless round-trip unit test; the unrelated generic dense
converter remains unchanged.  That fix is now implemented at `5a4c5783` and
the focused layout/adapter tests are exact.  The frozen `vdam-08` autonomous
trajectory (source EM case `k1-25`) passes unchanged gates in Slurm job
`12672666`: cross-engine FSC-AUC is `0.99999964`, `0.99999931`, `0.99918555`,
and `0.99900277` at iterations 1/2/4/8, respectively.  The worst RECOVAR-minus-
RELION GT FSC-AUC is `-8.23e-6`, compared with the fixed `-0.002` gate.  This
is the first accepted end-to-end VDAM checkpoint.  The active validation is
the frozen fixed-12 matrix, using the same per-case runner and at most four
simultaneous one-GPU Slurm tasks.

The first fixed-12 matrix round is Slurm array `12672742`.  The early cases
confirm the intended late-trajectory sensitivity: `vdam-03`, `vdam-04`,
`vdam-05`, and `vdam-07` pass; `vdam-01`, `vdam-02`, `vdam-06`, and `vdam-10`
miss only the fixed cross-engine FSC-AUC gate while retaining acceptable GT
quality.  `vdam-09` did not reach an audit because its high-resolution local
E-step exhausted the 40 GB GPU at the default 500-image batch.  `vdam-11`
completed and misses only the fixed cross-engine gate (minimum FSC-AUC
`0.99691967`) while retaining acceptable GT quality (minimum delta
`-0.00018535`).  `vdam-12` independently exhausted the same 40 GB GPU boundary
at the default batch.  Batch 200 closes the resource boundary for `vdam-09`,
which then completes with minimum cross-engine FSC-AUC `0.99734426` and no
negative GT delta.  For `vdam-12`, batch 200 reduces the failed allocation from
`37.54` GiB to `29.33` GiB but still exhausts the device once resident buffers
are included.  Batch 100 still requests `29.27` GiB, showing that the fused
sparse-M-step tensor rather than the outer image batch controls this boundary.
The runner records its resource/execution overrides; the matrix pins `vdam-09`
to batch 200 and routes `vdam-12` through the shared exact deferred packed-
M-step fallback by setting the sparse big-JIT tensor cap to zero.  The first
deferred attempt reaches a later batch-dependent shifted-reconstruction
tile and exhausts memory at 6.40 GiB with batch 100.  Pinning only `vdam-12`
to batch 25 while retaining the deferred path completes in Slurm job
`12673810` and passes every unchanged checkpoint: minimum cross-engine FSC-AUC
`0.99931708`, minimum GT delta `-1.60e-7`.  The complete SHA-bound scorecard is
now 6/12 passing with 12/12 evaluated.  Passing cases are `vdam-03`,
`vdam-04`, `vdam-05`, `vdam-07`, `vdam-08`, and `vdam-12`; all six failures
retain acceptable GT quality and fail only the strict cross-engine FSC-AUC
gate.  Evidence is recorded in
`docs/math/vdam_relion_parity_evidence_ledger_20260820_layoutfix.json`.

The first remaining autonomous divergence is now identity-localized rather
than inferred from maps.  The reusable STAR-to-STAR diagnostic
`scripts/audit_vdam_particle_state_trajectory.py` aligns every row by exact
`rlnImageName` and computes geodesic pose error, translation error, and
absolute Pmax error without correlation or an acceptance gate.  For
`vdam-01`, iteration 1 has exact pose/translation winners for all 3,000
particles.  Iteration 2 has only six divergent identities:
`1016@particles.128.mrcs`, `108@particles.128.mrcs`,
`1085@particles.128.mrcs`, `1130@particles.128.mrcs`,
`1137@particles.128.mrcs`, and `1171@particles.128.mrcs`.  The divergent count
then grows to 104/182/359 at iterations 3/4/8, while Pmax MAE grows from
`5.61e-5` at iteration 1 to `4.49e-4`, `0.01305`, `0.02358`, and `0.03399`.
The next scientific boundary is the iteration-2 score/support decision for
those six immutable original image identities; map tolerances remain frozen.

## Mode Contract

- **Production precision (user decision 2026-09-08):** float32 remains the
  intended EM execution path. Double precision is diagnostic only, used to
  distinguish roundoff from implementation bugs with matched inputs and state.
  It is not a substitute for a corrected and qualified float32 implementation.
  Final K1/K4 quality and performance evidence must use the production path;
  preserve deliberate higher-precision host/metadata operations. See the
  [mandatory precision policy](../../recovar/em/AGENTS.md).
- **Strict oracle:** the default during parity closure; pinned RELION GUI
  behavior and full iteration trajectory, including `firstiter_cc` hard-winner
  semantics.
- **Quality:** a later opt-in during parity closure; an intentional RELION
  difference is acceptable only when named and FSC/FSC-AUC against GT is
  neutral or better.
- **Performance:** exact accepted quality behavior with timing instrumentation;
  no algorithmic approximation without separate quality qualification.

## Reproducing the RELION dispatch-v2 oracle

Strict K-class replay needs RELION's authoritative mapping from sorted particle
position to MPI follower and original particle ID.  The diagnostic patch is
versioned at
`docs/patches/relion_dispatch_log_schema_v2_d476e6f.patch` (SHA-256
`6987c5ce397cbdd98835682cf1481a150c38c48cda621e006341d01a77e11c11`).
Apply it only to RELION base
`d476e6f6a4f1f37627c06ace5227fc374c0c2b05`:

```bash
test "$(git -C "$RELION_SRC" rev-parse HEAD)" = \
  d476e6f6a4f1f37627c06ace5227fc374c0c2b05
git -C "$RELION_SRC" apply \
  "$RECOVAR_SRC/docs/patches/relion_dispatch_log_schema_v2_d476e6f.patch"

source /etc/profile.d/modules.sh
module purge
module load relion/5.0.1/gcc-11.5.0-gpu
cmake --fresh -S "$RELION_SRC" -B "$RELION_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$(command -v gcc)" \
  -DCMAKE_CXX_COMPILER="$(command -v g++)" \
  -DMPI_C_COMPILER="$(command -v mpicc)" \
  -DMPI_CXX_COMPILER="$(command -v mpicxx)" \
  -DCUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.6 \
  -DCUDA_ARCH=80 -DGUI=OFF -DBUILD_TESTS=OFF
cmake --build "$RELION_BUILD" --target refine_mpi --parallel 8
strings "$RELION_BUILD/bin/relion_refine_mpi" \
  | grep -Fx RELION_DISPATCH_LOG_SCHEMA_V2
```

The qualified Della build used GCC 11, OpenMPI 4.1.6, CUDA 12.6, and RELION's
existing FFTW installation.  Set `RELION_DISPATCH_LOG` for a one-iteration
K-class smoke with the same fixture, MPI follower count, pool size, and seed as
the intended replay.  The leader writes the marker followed by five integer
columns:

```text
# RELION_DISPATCH_LOG_SCHEMA_V2
2 iteration follower_rank sorted_position original_part_id
```

Require the marker, then use the RECOVAR builder as the smoke validator; it
rejects non-v2 rows and non-bijective sorted positions or original IDs:

```bash
test "$(head -n 1 "$RELION_DISPATCH_LOG")" = \
  '# RELION_DISPATCH_LOG_SCHEMA_V2'
pixi run python -m scripts.build_relion_dispatch_schedule \
  --dispatch-log "$RELION_DISPATCH_LOG" \
  --output "$ORACLE_DIR/dispatch_schedule.npz" \
  --oracle-dir "$ORACLE_DIR" --n-particles "$N_PARTICLES" \
  --n-followers "$N_FOLLOWERS" --pool-size "$POOL_SIZE" \
  --random-seed "$RANDOM_SEED"
```

The hook is inert when `RELION_DISPATCH_LOG` is unset.  Keep the patch and
RELION source identity in run provenance; do not substitute a legacy
four-column range capture.

## Current State — 2026-07-17

Authoritative clean candidate checkout:
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_em_parity_20260711/recovar`

Current integrated implementation checkpoint:
`dcd1aa07c54a087631f7bfd706439b64b20cdcfb`
on `codex/em-parity-checkpoint-20260711`.

Immutable broad-candidate checkpoint:
`a6d1d086d81fe7d2be863c50bad33c7ea85e0b7f` on
`codex/em-parity-checkpoint-20260711`. The original dirty checkout remains
unchanged at
`/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_em_min_deferred_abs2_20260709_1745`.

Base HEAD before this board was created:
`4fba8f48a00ca7820a763e7ba41dac4a5a8d8242` on
`codex/em-deferred-bigjit-abs2-min-20260709` with a large dirty candidate stack.
Every new run must record a fresh diff SHA-256 and untracked manifest.

Known evidence:

- K=1 100k/256 map quality is excellent: merged RECOVAR-vs-RELION correlation
  `0.999571`, FSC-AUC `0.994387`, and RECOVAR GT FSC-AUC is `+0.009573` above
  RELION. RECOVAR is about `1.40x` RELION wall time on the recorded run.
- Current-head K=1 fixed-fixture boundary-replay job `11144457` matches all
  ten numbered RELION-seeded transitions, the exact current-size schedule, convergence at
  iteration 10, and the valid converged final-all-data path. Independent
  shellwise recomputation puts every numbered half/merged FSC-AUC above
  `0.9999985`; final merged RECOVAR-vs-RELION FSC-AUC is `0.998450626`, and
  RECOVAR final GT FSC-AUC exceeds RELION by `0.019912496`. This closes the
  fixed 3k/128 per-iteration boundary gate. It does **not** close the autonomous
  trajectory gate: `scripts/run_multi_iter_parity.py` injects RELION particle,
  noise, direction-prior, and optimiser-control state at each iteration.
- Current-head autonomous case-20 job `11197313` closes that small-trajectory
  qualification: exact current-size schedule and iteration-11 convergence,
  every numbered half/merged cross FSC-AUC at least `0.999986`, and final
  merged cross FSC-AUC `0.997634`.  RECOVAR final GT FSC-AUC is `+0.001144`
  above RELION.  The science command completed successfully; the recorded
  Slurm exit 2 is only a noncanonical-layout error in the post-run generic
  summarizer.
- K=4 100k/256 map quality is close/better by GT FSC-AUC, but particle-level
  state parity is incomplete: recorded class agreement `0.89025`, pose within
  5 degrees `0.71669`, translation within 1 px `0.77529`. Runtime is `2.181x`
  RELION; sparse K-class pass 2 dominates the completed iteration wall.
- Exact local x-half current/full BPref microbatching now survives the recorded
  3k/128 stress case without OOM. The conservative cap is validated for that
  fixture, not yet a universal optimal cap.
- The earlier 3k/128 final-state replay defect and iteration-1 reconstruction
  boundary are repaired. The current accepted evidence is FSC/FSC-AUC based;
  legacy map correlations are diagnostic only. The next K=1 work is the
  predefined boundary-replay robustness matrix while an autonomous cold-start
  trajectory is qualified separately, followed by 10k, real-particle, and
  100k/256 validation. The existing real EMPIAR-10076 failure is already localized to
  iteration-3 low-shell PPref formation before amplification at the
  iteration-8 global-to-local transition.
- Real-10076 iteration-1 BPref factor replay identified a production local-EM
  weighted-sum precision defect. Full float32 products reduce the frozen
  RELION numerator gap from `2.07567e-4` to `3.39177e-7` (`611.97x`).
  Same-A100 control/fix maps have minimum FSC-AUC `0.9999999863`, and the fix
  slightly improves both half-map and merged RELION FSC-AUC. This closes the
  one-iteration causal boundary only; full real-data trajectory parity and a
  clean warmed performance comparison remain open.

Canonical evidence and paths are in `docs/math/relion_parity_agent_notes.md`
and `docs/math/em_parity_best_metrics.md`.

## Quantitative Gates

These are program gates, not arbitrary test tolerances. Change them only by an
explicit user decision.

### Fixed-state arithmetic

- score/Pmax p95 absolute gap `<=1e-4` where RELION GPU arithmetic permits;
- maximum gap `<1e-3` unless a CPU/double adjudication explains it;
- exact best pose/class/translation agreement when the winning margin is above
  the numerical band; near-tie flips require candidate score/posterior evidence
  that the inputs agree within the numerical contract;
- no systematic drift by half, class, shell, pass, or candidate count.

### K=1 supplied-map quality

- merged RECOVAR-vs-RELION FSC-AUC `>=0.995`;
- RECOVAR GT FSC-AUC no worse than RELION by more than `0.002`;
- shellwise FSC curves and the established FSC score/resolution summaries
  versus both GT and RELION show no unexplained systematic deficit;
- strict-mode per-iteration state differences are arithmetic-level after the
  first-iteration policy is matched;
- convergence iteration and final all-data path agree exactly.

Map correlation is recorded only as a weak diagnostic. It is never a K=1
quality gate and cannot override the FSC/FSC-AUC decision in either direction.

### K=4 supplied-map quality

- every Hungarian-matched RECOVAR-vs-RELION class FSC-AUC `>=0.995`, with
  shellwise FSC curves and established FSC score/resolution summaries reported
  per class;
- per-class GT FSC-AUC no worse than RELION by more than `0.002` without a
  documented quality-mode improvement;
- class agreement `>=99%`, with pose/translation distributions reported per
  class and no collapsed/minority class;
- convergence/finalization semantics agree.

Map correlation is recorded only as a weak diagnostic. It is never a K=4
quality gate, and class averaging must not hide a poor per-class FSC result.

### Performance

- quality freeze first, then intermediate K=4 target `<=1.5x` RELION and K=1
  target `<=1.2x`;
- completion target K=4 `<=1.2x` and K=1 `<=1.1x` on the same pinned hardware;
- report compilation separately and include end-to-end time, per-stage time,
  throughput, and peak memory.

## Milestones And Exit Criteria

1. **Freeze reproducible oracle and candidate.** Create a clean checkpoint or
   reviewable logical commit series from the current stack; pin RELION build,
   fixtures, commands, and hardware. Exit when any result can be reproduced
   from immutable identities.
2. **Close K=1 strict trajectory parity.** Implement or qualify strict
   `firstiter_cc` semantics, compare every state boundary, and match
   convergence/finalization. Exit when K=1 gates pass across small robustness
   cells, at least one real-particle confirmation, and the 100k/256 completion
   case.
3. **Close K=4 quality and state parity.** Find first divergence before final
   maps, repair class/pose/translation trajectory and finalization, and cover
   class imbalance/noise/CTF stress. Exit when K=4 gates pass.
4. **Freeze quality checkpoint.** Tag/commit the accepted behavior and lock a
   reproducible K=1/K=4 benchmark matrix. No performance patch proceeds when
   its quality comparison is missing.
5. **Optimize K=4.** Attack measured sparse pass-2 and M-step/noise bottlenecks
   one at a time with output equivalence tests.
6. **Optimize K=1.** Reduce pass-2/local overhead, compilation, and memory
   traffic while retaining the quality checkpoint.
7. **Expand scope.** Native InitialModel/VDAM, broader distributions, larger
   boxes/counts, and additional GPU architectures after supplied-map closure.

## Active Milestone

Milestone 1 now has an immutable local checkpoint. The active scientific target
is Milestone 2; future changes must be small logical commits on top of the
checkpoint.

### Next experiment

Authoritative checkpoint on 2026-07-31 at 04:53 ET:

- The published PR lineage includes strict-audit code through
  `e87565c5d02c5ff3cd7d035c8d225e9047ae13a2` and its NaN-accounting
  documentation through `39c0cd1af3aa5bee768cea314e830e2f159bff9f`.
  Fixed metrics remain K=1
  `28/34` strict FSC/FSC-AUC, `32/34` topology, and `34/34` evaluated;
  K=4 remains `41/60` direct per-class FSC-AUC and `9/15` all-class
  iterations. The separate non-scoring K=4 exact-device causal boundary
  remains `2/4`.
- K=4 job `11820573` completed the full 100,000-particle four-class sparse
  E/M pass and wrote the targeted pass-2, contribution, and device-signature
  artifacts, then failed after capture when its current-size `79^3`
  accumulator reached generic final reconstruction. The recapture audit
  rejected a science conclusion because neither independent recapture was
  globally repeatable against the sealed source. At the V17 representative,
  continuing through contribution reproduced the native value by one ULP,
  while stopping after pass 2 reproduced the sealed value.
- Commit `d1fb8e52` adds an explicit contribution-stop boundary so the
  diagnostic exits only after the contribution and requested device signature
  exist, without falling into final reconstruction. Same-observer retry
  `11822111` received GPUs `41:00.0` and `C1:00.0`, excluding the pinned
  `81:00.0` UUID, and failed closed with exit `42:0` after three seconds,
  before import or science. Existing operand comparator `11822185` is
  `DependencyNeverSatisfied` and remains untouched.
- The retry launcher's value-equality checks do not literally prove the stated
  bitwise contract for signed zeros or distinct NaN payloads. Commit
  `72a0a396` adds a strict per-element byte auditor with structured incomplete
  and mismatch reports; `e87565c5` separates value inequality, paired NaNs,
  NaN-payload bytes, signed-zero bytes, and strict byte mismatches. The
  corrected historical recapture counts are `18,397`, `12,029`, and `16,531`
  strict raw byte mismatches, not `253,501`, `247,133`, and `251,635`; each
  larger value included 235,104 paired NaNs whose bytes are identical. CPU
  audit `11822605` completed successfully and wrote an explicit incomplete,
  rejected `0/3` report because all three producer archives are absent. Its
  report SHA-256 is
  `7e0db0f81931ec54bde2a55fee274303fc77247e722980818d024af700b366d9`.
- Isolated retry `11823179` requests the three A100 slots not held by the
  long-lived index-0 job, still fails closed unless the pinned UUID is present,
  and exposes only that UUID to the process. It is pending at zero elapsed for
  resources. A normalized launcher audit classifies the retry as
  `allocation_and_provenance_only_change`: after replacing only fixed
  job/launcher names, exclusive roots, and the predeclared two-to-three-GPU
  allocation/count gate, the launcher bytes are identical and the
  science-relevant difference count is zero. Its JSON SHA-256 is
  `9b21a05bf8b037ae13be5c0059e6899c6dca2e84f742c1a527395f4f55f66571`.
  Strict `afterany` audit `11823392`, frozen at `e87565c5`, is
  dependency-pending and will produce complete mismatch accounting or an
  explicit missing-artifact rejection.
- Independent fallback `11825092` removes the pinned-node allocation
  bottleneck without changing or superseding those jobs. It received one A100
  on `della-l07g4` and runs the frozen continue-through reference,
  contribution-stop arm, strict three-archive byte audit, conditional native
  RELION representative capture, and conditional operand comparator
  sequentially on that same physical GPU. Native and comparator stages are
  skipped unless strict repeatability passes `3/3`. The outer launcher
  SHA-256 is
  `5bd5d6e07a6bd054bc63e9cbf4d4eba4ef76f2b33c60ff3c47b8173640b3c165`;
  run root is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_sameallocation_reference_stop_native_20260731T0515ET`.
  A pre-result static review found that this first launcher still requires its
  new reference archives to have the historical hashes. Job `11825092`
  completed outcome-safe after `43:22`: the new reference hashes were
  `027d0aae614476485434fd7132ca885496a1bc78476b24247130a8efb3ed7518`,
  `17c2d4f97787f82578d8086ebc73f48cdd424632ea221c2d245ef78c78d4e236`,
  and
  `d63bcdbf2545ee99caa59a1bdbd24a5bf68327f3273e86d15a51167d00824d98`,
  so the historical-hash preflight rejected before the stop arm. Its strict
  report is explicitly incomplete and rejected `0/3` because the three
  observed archives do not exist.
  Corrected retry `11825133` instead exports the immediately preceding
  same-allocation reference hashes into the stop arm. It started independently
  on `della-l07g5`; outer launcher SHA-256 is
  `74402fb99f412dbc7f6bf2b50e093b17a1d544e08fa599c85efee4b83f6d751e`.
  Its reference hashes are
  `5504a5015c4e83300282f87f7ab77466544fbb17d6be8948548a725845ff12a9`,
  `98c8642d7b85645f6416aa834eef931d3561e3db651111cd5d22cbd6ff7e5c0b`,
  and
  `66effbd5a2a4bb649cfa418f9302e3349cb8d9c7276d548c54ae9009633a5803`.
  At 06:03:49 ET its contribution-stop arm passed preflight with those exact
  dynamic pins and the same physical UUID
  `GPU-803dc869-2e74-273c-1df4-08adbc94e1b3`.
- The submitted K=1 live-versus-serialized low-shell noise-state restart
  chain is closed without science evidence. Iteration-0 owner `11785428`
  and iteration-1 owner `11785547` each rejected an allocation whose GPU
  UUIDs did not match the sealed target, exiting `42:0` after one and two
  seconds. Robust auditors `11791340` and `11791341` and pair auditor
  `11791712` remain dependency-pending and cannot produce an admissible
  result from these failed owners. No job was modified.
- Independent job `11836574` removes that exact-UUID scheduling confound. It
  runs a fresh control and both serialized restart points sequentially on one
  scheduler-selected A100, with all three score captures and map products
  qualified inside the same allocation. It started on `della-l07g4` at
  10:44:22 ET from clean detached source `684bf67a`; no existing job was
  changed.
- The pending K=4 chain is not yet a quality result. Ingest retry-2's strict
  K=4 repeatability report before accepting an operand classification. The
  K=1 chain above produced no trajectory to ingest. Any K=4 production
  proposal still requires the frozen multistratum repeat and a
  same-physical-GPU end-to-end Hungarian per-class FSC/FSC-AUC comparison.

The PR-ready fixed metrics can be regenerated from the versioned, checked
scorecards with one deterministic command:

```bash
pixi run python scripts/report_em_parity_progress.py --format markdown
```

## 2026-07-31 exact-device K=1 case-22 restart chain closes pre-science

Read-only Slurm and artifact inspection closes the previously pending
case-22 serialized-restart chain without a scientific result. Job `11785428`
started on `della-l07g3` at `09:32:32 ET` and failed `42:0` after one second.
Job `11785547` started on the same node at `09:32:36 ET` and failed `42:0`
after two seconds. Both stderr files contain only:

```text
EXACT_GPU_REJECT target=GPU-6b5da455-0f76-eeaa-6041-ec8df42a2e8a
```

The allocation table recorded
`GPU-a1bb1fb4-d5e3-1c72-3382-63f6032e9fc6` and
`GPU-eb1c5b04-20c1-b6c9-16e6-b3dc87905bd7`; neither is the sealed target.
The two stderr files are byte-identical with SHA-256
`8da423abfecb9225eb023bf60afc16dd06ccb5e049501714e67b5be062e29424`.
The allocation-table SHA-256 is
`6cd75ee73c9728dda471e67db9564565ea6a4b87da8a5b8d4e4c4534a2f0349e`.
No restart trajectory, score report, or map report was written.

Dependency-bound auditors `11791340`, `11791341`, and `11791712` therefore
cannot qualify this chain. They remain untouched in Slurm with unsatisfied
dependencies. This is a provenance/device rejection, not evidence for or
against the serialized-noise hypothesis, and it cannot change a fixed
scorecard. K=1 remains `28/34` strict, `32/34` topology, and `34/34`
evaluated; K=4 remains `41/60` direct and `9/15` all-class; the separate
K=4 causal boundary remains `2/4`. Correlation was not computed. No Codex
process or Slurm job was killed, signalled, suspended, cancelled,
reprioritized, requeued, held, released, or otherwise altered.

### Same-allocation K=1 case-22 replacement gate

Commit `684bf67ab318d200f512636cd670ab9cc06e0e38` extends the
serialized-restart score analyzer with an optional explicit fresh-capture
tuple: capture directory, component-validation report, and direct-operand
validation report must all be supplied together or the analyzer fails
closed. The original parent-hash-bound mode remains unchanged. A real
14-particle replay against the sealed case-22 capture reproduced the legacy
classification, fixed metric, and all particle records exactly while
changing only the fresh-capture provenance mode. The score/map/pair unit gate
passes `25/25`.

Independent Slurm job `11836574` started at `10:44:22 ET` on
`della-l07g4`, A100 UUID
`GPU-2a189b71-ee86-c95f-7bc0-93e83a0812e4`. It runs three arms
sequentially under that one allocation:

1. fresh RELION from the immutable particle STAR and initial map through
   physical iteration 2;
2. restart from the fresh arm's serialized iteration-0 optimiser;
3. restart from the fresh arm's serialized iteration-1 optimiser.

The two restart score gates each retain `14` fixed particles. The two map
gates each retain half 1, half 2, and merged FSC-AUC (`3` products), with
strictly positive parity improvement and nonnegative GT delta required for
acceptance. The two-arm pair classifier retains all four outcomes. No
threshold, scale, sign, shell boundary, or correlation is fit.

The launcher and predeclaration SHA-256 values are
`582017d939310f0a0f38444329b87951241e43bdb5f6019c7e22ebf102fef85f`
and
`113047b75820df2f433ff62712d628223f8e0cae4bc90f45962f9bf5e4c7e6b0`.
Run and runtime roots are
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_sameallocation_restart_pair_684bf67a_20260731T1039ET`
and
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case22_sameallocation_restart_pair_684bf67a_20260731T1039ET`;
both contain `SAFE_TO_DELETE`. This remains non-scoring until all arms and
fixed analyzers complete. K=1 remains `28/34` strict, `32/34` topology, and
`34/34` evaluated; K=4 remains `41/60` direct and `9/15` all-class. No
existing process or Slurm job was modified or interrupted.

#### Iteration-0 continuation repair

Job `11836574` completed and sealed only its fresh arm, then failed naturally
before restart science. Its direct-operand and preprocessing capture gates
passed `14/14`, but restart-0 entered RELION's fresh-run initialization
because the serialized iteration number was zero and attempted to load an
empty reference filename. It wrote no restart capture, score/map report, pair
report, or completion ledger. This is orchestration/provenance evidence, not a
result for or against the serialized-restart hypothesis.

Private diagnostic RELION source commit
`ed53c60d83125902c456b7fc5461c78c3966b306` repairs only this boundary. It
distinguishes `--continue` from a true fresh run; preserves already loaded
data, references, sampling, and leader/follower data-versus-prior arrays;
avoids a second pixel-to-Angstrom sampling conversion and reference low-pass;
and restores the internal zero current-resolution sentinel that the
iteration-0 model serializes as `inf` Angstrom. Because RELION does not
serialize its first-iteration controls, restart-0 also restates
`--firstiter_cc --ini_high 30 --pool 3`. The pinned executable SHA-256 is
`a274dda1b0b40478ddd7f2b81d144bec20510db369225f365f1be7d27ac45309`.

The exact final binary passed a local pool-matched A100 qualification at the
iteration-1 boundary:

| Fixed control | Result |
| --- | ---: |
| particle identity/subset/group/class/pose/translation | exact `3,000/3,000` |
| sampling STAR | exact |
| half-1 fresh/restart FSC-AUC | `0.9999999999870963` |
| half-2 fresh/restart FSC-AUC | `0.9999999999864432` |
| merged fresh/restart FSC-AUC | `0.9999999999911986` |
| minimum non-DC shell FSC | `0.9999999998870037` |

Norm/likelihood differences are at most one six-decimal STAR unit.
Correlation was forbidden and not computed. A patched-versus-original
iteration-1 no-op control produced identical non-path tables and
bitwise-identical half maps, bounding the repair to iteration zero.

Corrected independent job `11838510` started at `12:05:45 ET` on
`della-l07g4`. It reruns fresh, restart-0, and restart-1 sequentially on one
A100 and adds fail-closed checks for the fresh iteration-0 sentinel/schedule,
the fresh-equivalent restart-0 iteration-1 topology, all `3,000` exact
particle state rows, exact sampling, and exact non-path optimiser state before
iteration-2 evidence is admissible. Launcher and predeclaration SHA-256 are
`70401a2e7b82df4c7bedd0972570f76b98783cf47ecf32c2cb38d8bb1c206b02`
and
`da9bb7e0054f244ac352a0e703a6724e1f5513eb3eac31b7feca784ad542fb15`.
The job is non-scoring while active; all fixed scorecards remain unchanged.

A read-only live check then found one remaining command-state confound before
acceptance. At the completed fresh/restart-0 iteration-1 boundary, all
`3,000` fixed particle state rows, sampling, first-iteration topology, and
zero no-resolution-gain count match. The sole non-path optimiser difference
is `_rlnNumberOfIterations: 2` fresh versus `999` restart: RELION's
continuation parser uses its command-line default unless
`--auto_iter_max` is restated. Job `11838510` is therefore
non-qualifying and remains untouched to finish naturally.

Independent job `11839040` explicitly pins `--auto_iter_max 2` on both
restart arms and retains the exact non-path optimiser equality gate.
Restart-0 uses
`--firstiter_cc --ini_high 30 --auto_iter_max 2 --pool 3`; restart-1 uses
`--auto_iter_max 2 --pool 3`. Launcher and predeclaration SHA-256 are
`6ef245eb663ae55eac0d6e4eae07de42f131713929056c1266d3b4620e74ff01`
and
`f8a6f1867d271dd69d489940f108f8311ed03aef599e3fe669acfc734e625ad2`.
It is pending A100 resources and remains non-scoring.

The consolidated report also names the exact remaining K=1 strict/topology
cases, every failed K=4 iteration/class cell, and the remaining non-scoring
K=4 causal cases. The K=4 class checklist, including all 60 measured FSC-AUC
values and checkmarks, can be validated independently:

```bash
pixi run python scripts/summarize_em_k4_class_fsc_auc_scorecard.py --check
```

The reporter validates the frozen K=1 scorecard and fixture manifest, the
fixed K=4 trajectory snapshot, its 60-cell class-level scorecard, and the
separate K=4 causal scorecard before emitting any count. These gap lists
cannot silently drift from the fixed denominators. The reporter cannot modify
a scorecard or authorize a production change.

Authoritative status on 2026-07-17:

- The same-A100 real-10076 K=1 run matches all 18 forced RELION numbered
  sampling/size boundaries, but its cross-engine map trajectory leaves the
  native repeat envelope at iteration 7. Forced scheduling prevents the
  autonomous iteration-8 collapse, yet forced merged FSC-AUC falls to
  `0.9745783` by iteration 16. This is a real accumulated state mismatch, not
  merely the discrete schedule branch.
- RECOVAR's autonomous trajectory advances HEALPix orders 4, 5, and 6 two
  iterations early and finalizes after numbered iteration 16. The historical
  `7f142d5f` post-cap interpretation was later invalidated by direct source
  audit: RELION has no convergence or sampling boundary after `iter > nr_iter`.
  Commit `607e4344` removes that synthetic boundary. The real-data termination
  difference remains an upstream accumulated map/schedule mismatch; do not
  treat the trajectory as accepted.
- The sealed iteration-2-to-3 operand decomposition is closed without a
  production candidate. It compares all 15 recurrent `>0.1`-degree tail rows
  with 15 deterministic matched controls under exact UID, support, geometry,
  and same-GPU control gates. Native common-prior operand TV is not enriched in
  the tail (median `8.9293e-5` tail versus `1.1321e-4` control; 7/15 paired
  tail values are larger). Canonical float64-from-captured-float32 TV is also
  similar (`4.0843e-6` versus `4.2158e-6`; 10/15 larger). No unit-aligned
  single- or two-field reference/image-weight/score-weight swap passes the
  pre-registered native-and-float64 movement and repeat-envelope gates.
  Classification is `unresolved_combined`; no production change or controlled
  substitution is authorized. Move to a compact full-10,000-particle
  score/posterior distribution diagnostic from a completely sealed
  uninterrupted RELION pre-iteration-3 boundary; do not resume serial
  particle tracing or pixel-operand capture. A restarted iteration-2 boundary
  is inadmissible because it changes the perturbation and can overwrite the
  half-2 follower noise state.
- The RECOVAR side of that diagnostic is implemented through `dcd1aa07`: an
  env-off-inert production score/posterior tap, bounded raw shards with strict
  readback/manifests and per-half identity closure, and an atomic captured
  RELION `Projector::data` replay contract. The full science launch remains
  blocked on a compact RELION live-state capture with corrected device Euler
  copies and complete rank/optics/metadata/control state.
- The K=4 100k/256 compact-score memory run is progressing without OOM on an
  A100; the dependent strict Hungarian FSC/FSC-AUC and state audit remains the
  quality gate.

Priority order: finish and validate the compact uninterrupted RELION live
boundary, then capture sharded full-dataset score/posterior arrays and compare
RECOVAR against that exact boundary. Treat independent RELION runs as whole-run
controls unless every boundary byte matches. Keep capped runs fail-closed
without a synthetic terminal boundary; permit an aggregate
boundary substitution only if the full-dataset evidence selects a systematic
source; then use full FSC/FSC-AUC trajectories to repair the remaining real
K=1 drift and accept or repair the running K=4 trajectory. Avoid further
serial particle tracing unless the aggregate evidence identifies a systematic
subgroup.

Current status on 2026-07-16: the eight-case autonomous K=1 robustness matrix
passes every FSC/FSC-AUC trajectory, schedule, convergence, and finalization
gate.  The remaining K=1 completion cells are the running 100k/256 trajectory
and the real-10076 repeat-qualified production-boundary confirmation.  The
bounded exact raw-diff2 cache has passed its frozen-boundary and same-A100
repeat controls and is integrated as a performance-only optimization.  Cache
and recompute are bitwise identical at the changed boundary; all 3,000 poses,
translations, and hard assignments remain exact, direct map FSC-AUC is at
least `0.999999999845`, and the clean controls improve wall time by
`10.7--15.8%`.

The local weighted-sum precision repair is integrated at `94b8f2b2`. Its
sealed same-A100 diagnostic is accepted as a causal production bug fix, not
as trajectory completion or performance evidence. A corrected 100k/256 full
trajectory is running under Slurm job `11288959`, followed by FSC/FSC-AUC
audit job `11288960`. Compare its complete schedule, convergence,
finalization, cross-engine FSC-AUC, and GT FSC-AUC before expanding the patch
to alternate dense M-step routes.

K=4 case 11 now has a recurrent aggregate iteration-1 membership boundary:
RELION and RECOVAR agree for 9,999/10,000 particles in a same-A100 six-arm
control, and zero-based particle 7915 is the sole recurrent class transfer.
A same-A100 class-routing intervention changes exactly that particle and
restores all four RELION-vs-RECOVAR class-map FSC-AUC values to at least
`0.9999999671`.  The global-winner boundary is therefore causal and a
full-class pre-scatter capture is unnecessary.  The remaining discriminator
is a frozen canonical float64/complex128 replay of that one decision, with
native float32 closure controls, to classify operand generation versus
reduction/order/precision.  Do not return to serial particle tracing or
unstable iteration-3+ cliffs.

Current status on 2026-07-14: the seven-case immutable K=1 robustness matrix
at detached commit `f0ef1f0c6c231ff1f9183371d235e0b37a15b825` matches every
RELION current-size schedule and convergence iteration.  The previously
systematic final-map offset is now localized to an explicit output behavior,
not trajectory noise: the quality-oriented RECOVAR final path leaves RELION's
radial sinc-squared gridding correction disabled.  Applying the exact
`padding_factor=2` correction post hoc raises final RECOVAR-vs-RELION FSC-AUC
from `0.997395--0.998870` to `0.999605--0.9999997`.  Corrected RECOVAR-minus-
RELION GT FSC-AUC is between `-9.33e-6` and `+1.03e-5` in all seven cases.

The next K=1 identity target is severe case 26, whose corrected final cross
FSC-AUC remains `0.999605407`; the other six corrected cases are at least
`0.999961340`.  Its first autonomous departure is now localized before
reconstruction to the iteration-1 accelerated BPref accumulator.  Iteration-1
hard-WTA poses, translations, and Pmax are exact, but the matched RECOVAR
post-x0 versus RELION pre-lowres-join BPref numerator/weight relative-L2
residual is `3e-6--6e-6`.  Do not compare against RELION pre-reconstruct here;
that boundary is already after the 40-Angstrom half join.
Three same-H100 RELION captures vary by only `1.0e-8--1.3e-8`, so this residual
is reproducible code arithmetic rather than atomic-order noise.  The device
capture now identifies and fixes the first cause: RECOVAR added the integer
BPref origin before extracting float32 interpolation fractions, whereas
RELION extracts the fractions first.  The captured p8494 boundary becomes
bitwise exact for support, coordinates, all eight indices, and coefficients
after the arithmetic-order fix.  Rerun the trajectory gates; do not add a
score tie-break.

The explicit production grid-on case-25 diagnostic completed
as repaired jobs `11194076--11194077`: it matched RELION's current-size
schedule and iteration-8 convergence, logged radial correction enabled, and
reached final RECOVAR-vs-RELION FSC-AUC `0.999961353`.  Its RECOVAR and RELION
GT FSC-AUCs are `0.317329223` and `0.317318952` (delta `+1.0271e-5`), confirming
the post-hoc boundary through the real finalization path.  Keep
`RECOVAR_FINAL_ALL_DATA_GRID_CORRECT` unset/off outside named strict-parity
diagnostics because the current GUI-quality default remains grid-off.  In
parallel, retain the exact-Gaussian diagnostic conclusion and close the
case-20 accelerated preprocessing
boundary.  Typed `image_fourier_backend="relion_cuda"` at commit `bdda53c4`
now reproduces the source-faithful float32 normalization, zero-fill
translation, 128-by-128 CUDA background reduction, `sqrtf`/`cospif` mask, and
JAX/cuFFT window.  On both A100 and H100, captured particles 365 and 469 reach
bit-exact 65536-pixel masks and 1300/1300 Fourier windows within RELION's own
unordered atomic background envelope.  The existing `host_numpy` default is
unchanged.  Next validate fixed-state score arrays and then the full case-20
trajectory with explicit `relion_cuda`; the captured operand gate alone is not
an end-to-end quality claim.

The replacement exact fine-Gaussian reducer is integrated provisionally as the
default float32 Gaussian path.  A same-A100 paired iteration-2 run changed only
`RECOVAR_DISABLE_RELION_EXACT_FINE_GAUSSIAN=1`: mean Pmax error against RELION
fell from `2.78821e-5` to `1.99846e-5`, rows above `1e-4` fell from 371 to 168,
and wall time fell from `1065.49` to `538.72` seconds.  Iteration-1 state was
bitwise equal; iteration-2 merged cross-engine FSC-AUC was non-regressing
(`0.999999981423` to `0.999999981578`).  This real-data fixture has no GT, so
the result is not a GT-quality claim.  Commit `49e8f416` adds fail-closed
routing, removes unsafe fallback behavior, and restores the relevant tests.
The integration remains experimental until full K=1 and K=4 trajectory gates
pass; revert it if either quality gate regresses.

After those K=1 gates, advance to K=4 quality/state parity.  Performance is
not yet accepted: the broad matrix measures RECOVAR at `1.64--5.89x` RELION
wall time with approximately `97.6%` of an H100's memory occupied.

The current clean autonomous K=1 gate is job `11151255` at commit
`5a5769df37e49674c118697f60e73cbdd706b880`.  All ten numbered iterations
match RELION's current-size schedule `[56,56,66,68,80,80,80,80,80,80]`,
healpix schedule `[3,3,3,4,4,5,5,6,6,6]`, convergence at iteration 10, and
the single final Nyquist branch.  Every numbered FSC-AUC and GT gate passes.
The autonomous final merged map remains below the strict cross gate at
`0.986771`, although its GT FSC-AUC is better than RELION by `+0.020688`.
An exact-RELION-iteration-10 control on the same HEAD (job `11151769`) passes
the final cross gate at `0.998457`, localizing the failure to accumulated
trajectory state rather than final reconstruction mechanics.

The expected-accuracy mismatch is closed. RELION excludes the redundant
packed-FFTW `x=0, y<0` column from `Mresol`; the binding counted it. Adding the
same exclusion changes the exact first-100 result from `1.844` degrees /
`1.6915` Angstrom to RELION's exact `1.858` / `1.6915`. A Nyquist guard alone
does not change the mismatch. Same-process jobs `11152475`, `11152727`, and
`11152933` also reject particle order, CPU/GPU `PPref`, serialization,
anisotropic magnification, and scale-difference transforms. The audit is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_ppref_cpu_ab_20260713_192524/ROOT_CAUSE_REPORT.md`.

Autonomous A100 job `11153043` verifies the fixed iteration-2 accuracy and the
same exact size/order/convergence schedule. Its final merged cross FSC-AUC is
`0.986985443`, still below `0.995`, while RECOVAR GT FSC-AUC is `0.671500068`
versus RELION `0.650834886`. Expected accuracy was therefore a real boundary
bug but is not the final-map cause.

The exact-final accumulator factorial (jobs `11151900`, `11151905`, `11152058`,
and `11152064_0`-`11152064_3`) independently shows that the residual is
posterior/adjoint accumulation, not tau2: replacing only the accumulator raises
FSC-AUC from `0.995640221` to `0.998443887`, while replacing only tau2 changes
it to `0.995652710`. The final-state factorial now localizes the material
autonomous failure. Holding the RELION iteration-10 state except for one
component gives cross FSC-AUC `0.995216198` for RECOVAR poses alone and
`0.996873606` for the RECOVAR map alone; using both gives `0.993407704`.
RECOVAR noise/tau2 and direction-prior substitutions remain near `0.9934`.
Substituting RECOVAR's image/group-scale corrections collapses the result to
`0.986822205`, nearly reproducing the autonomous failure.

RECOVAR's scale array is exactly one because the driver reads group IDs only
from input `particles.star`, where they are absent; RELION's supplied
`run_it000_data.star` contains 3,000 groups derived from 3,000 micrographs.
Restore those groups by image identity, preserve the full group count in both
halves, then enforce RELION's additional `data_vs_prior > 3` shell mask when
collecting scale `XA/AA`.

The complete 3k/128 strict firstiter A/B is finished. CUDA texture projection
matches all 3,000 RELION iter-1 orientations exactly. Correlation remains a
weak diagnostic rather than a quality gate; iter-1 and all later maps are
judged only by shellwise FSC, FSC-AUC, and FSC-derived score/resolution.

The iter-1 accumulator boundary is classified. BPref complex averages and
weights agree at arithmetic level for typical coordinates, with small outer
shell/error tails. All five materially different winners have complete
coarse score comparisons, and both same-parent fine flips have coherent fine
comparisons. Every flip is a demonstrated numerical tie; no unexplained
support, frame, score, or accumulator mismatch remains. Retain the iter-1
correlation as diagnostic context, but judge the map only by its shellwise FSC,
FSC-AUC, and FSC score/resolution evidence.

The ten-iteration trajectory is complete and its numbered states are stable.
The stale final-state fix is validated: final all-data now uses RELION
`run_it010` without fallback, improving true-final FSC-AUC from `0.980260` to
`0.991498`, while the fixed-state numbered iter-10 map has diagnostic
correlation `0.999995`. Final particle medians are effectively exact, but
angular p95 is `0.631` degrees and Pmax correlation is only `0.7434` with mean
absolute gap `0.0423`. This localizes the active hypothesis to final
fine-posterior/support or its BPref accumulation, not convergence history,
numbered reconstruction, or output grid correction.

The final BPref and score boundary is classified. RELION's matched final dump
uses 24 fine rotations for a representative same-pose/Pmax outlier, while the
RECOVAR default expands 1,392; candidate counts are 169 versus 4,926 positive
and Pmax is `0.8333` versus `0.2519`. The existing pruned-parent path restores
24 rotations, 156 positive candidates, Pmax `0.8445`, and 6 retained samples
versus RELION's 5. This improves true-final FSC-AUC from `0.991498` to
`0.994527` grid-off and canonical 63-shell FSC-AUC `0.995784` with strict
RELION grid correction. Grid-on RECOVAR-vs-GT is only `0.000887` below RELION,
within the `0.002` gate. This small-cell final map therefore passes the FSC
quality contract; correlation is diagnostic only.

RELION pruned-parent support is now the K=1 local adaptive default, with
full-parent retained as an explicit diagnostic override. The four-way final
cross-replay shows tau2 substitution changes FSC-AUC by less than `1e-5`, while
substituting the RELION BPref accumulator raises strict-oracle FSC-AUC from
`0.997003` to `0.999684`. This exonerates final tau2/Wiener reconstruction and
localizes the remaining measurable high-shell residual to BPref accumulation.

Full clean A100 trajectory job `10990444` completed the exact ten-iteration
schedule `[56,56,66,68,80,80,80,80,80,80]`, convergence at iteration 10,
and final all-data branch. The earlier H100 request `10989654` never started
and was replaced because the pinned RELION oracle ran on A100. Numbered iter-10
RECOVAR-vs-RELION FSC-AUC is `0.999324`, and its GT FSC-AUC delta is only
`-0.000037`, but the free-trajectory unnumbered final FSC-AUC falls to
`0.988116`. This fails strict final map parity even though RECOVAR remains
better against GT (`0.669009` versus RELION `0.650835`). Do not launch the
robustness matrix yet.

Final-only job `10992173` enters the final pass directly from the saved free
iter-10 half maps and exactly reproduces `0.988115`, proving no hidden state
history after iter 10. Merged-reference diagnostic `10992266` is worse at
`0.981072`, falsifying early half-reference joining. Exact-RELION-iter1 seeded
job `10992371` reaches final FSC-AUC `0.994488`; iter-1 ties explain most but
not all of the free residual.

The current strict path now also matches RELION's separate joined-final noise
semantic: both particle halves use half-1 `sigma2_noise` in the
post-convergence K=1 all-data E-step. Exact dumped operands then match a representative RELION
posterior within the fixed-state numerical contract. Fixed-final job
`10994996` still reaches only grid-off FSC-AUC `0.994497`, while RECOVAR GT
FSC-AUC remains better (`0.669846` versus `0.650835`). Matched numbered iter-10
BPref job `10996603` passes its patched-oracle FSC gate and localizes a small
half-2 difference to four missing sub-winner significant samples, not noise,
half joining, mapping, or winner poses. Exact-RELION-iter10 final-only job
`10997070` remains at FSC-AUC `0.994501`, falsifying the tiny numbered-map
difference as the remaining final limiter. Its full final Pmax mean/p95/max
absolute errors are `0.0282/0.0898/0.4592`; original particle 428 / RELION
stack 429 is now the worst case.

The matched stack-429 operand replay has now classified that boundary. RELION
zeros the redundant `kx=0, ky<0` rows in its non-redundant half-plane, while
RECOVAR's full-size local likelihood counted those conjugate rows a second
time. Applying the RELION axis mask offline changes the parent Pmax from
`0.657449` to `0.812126` versus RELION `0.811969`, and restores exactly the 10
parent pairs retained by RELION. In particular it restores the otherwise
missing `(RELION rotation 140, coarse translation 4)` pair. Expanding that
pair and applying the same mask changes the final fine Pmax from `0.173354` to
`0.628355` versus RELION `0.628361`. The shared fine-candidate posterior L1
error falls from `0.3785` to `0.0288` even before the restored parent pair is
added. A centralized scoring-weight fix and focused unit regression are now
present in the dirty candidate.

Exact-RELION-iter10 final-only A100 job `11001328` qualifies the axis-mask
patch. Canonical RECOVAR-vs-RELION FSC-AUC is `0.997302`; the minimum non-DC
shell FSC is `0.995021`. RECOVAR-vs-GT FSC-AUC is `0.670396`, which is
`+0.019561` above RELION, and its FSC=0.5 crossing is one shell better (41
versus 40). RECOVAR is lower than RELION against GT in only three very-low
frequency shells, with worst delta `-0.000266`, inside the arithmetic band;
the other 59 non-identical shells are higher. The fixed-final K=1 small-cell
quality gate therefore passes without grid correction.

Clean free-trajectory A100 job `11002266` reproduces the RELION current-size
schedule `[56,56,66,68,80,80,80,80,80,80]`, convergence at iteration 10, and
the final all-data branch, but the unnumbered final remains below gate at
RECOVAR-vs-RELION FSC-AUC `0.990397`. This improves the pre-mask free result
`0.988116`, while RECOVAR-vs-GT remains better (`0.669518` versus `0.650835`).
Numbered merged FSC-AUC is already `0.997007` at iteration 2 and rises to
`0.999339` at iteration 10. Do not launch robustness.

Exact-RELION-iter1 seed job `11007539` runs numbered iterations 2--10 plus
final and passes at canonical RECOVAR-vs-RELION FSC-AUC `0.997271`.
RECOVAR-vs-GT FSC-AUC is `0.670338` versus RELION `0.650835`. This closes the
later-trajectory hypothesis and localizes the remaining free residual to the
iteration-1 boundary.

The first free run with the Gaussian redundant-axis fix exposed a scoped
regression: 198/3000 iteration-1 orientations and 219/3000 translations no
longer matched. RELION normalized-CC scores every pixel in its rectangular
first-iteration FFTW crop; only Gaussian likelihood scoring removes centered
`kx=0, ky<0` redundant rows. A score-mode-specific correction retains all CC
rows while preserving the qualified Gaussian mask. A100 job `11013677`
restores byte-identical coarse and fine hard assignments to the prior exact
texture run: every orientation matches RELION and only the established
0.5-pixel translation tie remains. Job `11013457` was an infrastructure-only
failure (`CUDA_ERROR_NO_DEVICE`) before science on `della-l07g3`.

Clean score-mode-scoped free-trajectory job `11014763` reproduces RELION's
current-size schedule `[56,56,66,68,80,80,80,80,80,80]`, convergence at
iteration 10, and final all-data path, but final canonical
RECOVAR-vs-RELION FSC-AUC is only `0.990351`. RECOVAR remains better against
GT (`0.669412` versus `0.650835`). Exact poses therefore do not by themselves
close the iteration-1 seed error.

The first divergent iteration-1 state is now identified. RECOVAR calculated
tau2 before applying the `firstiter_cc --ini_high` cutoff and retained nonzero
shells 20--28. RELION applies a squared raised-cosine taper to tau2 and
data-vs-prior after first-iteration reconstruction. The candidate implements
that source-matched taper. It also matches the pinned RELION accelerated GPU
build's single-precision `XFLOAT` BPref accumulator; the earlier float64
one-particle comparison used RELION's CPU/double backprojector and did not
represent the production oracle.

One-iteration A100 job `11021943` gives tau2 shell 18 `106.849670`, shell 19
`0.0235179`, and shells 20 onward zero, versus RELION `106.808`, approximately
`0.0235`, and zero. Merged iteration-1 RECOVAR-vs-RELION FSC-AUC over RELION's
supported shells 1--18 improves from `0.996052` to `0.998430`; shell 18
improves from `0.908735` to `0.948464`.

Clean A100 job `11023037` completes in `579` seconds with the exact schedule,
iteration-10 convergence, and final all-data path. Final canonical
RECOVAR-vs-RELION FSC-AUC improves from `0.990351` to `0.994646`, narrowly
missing the unchanged `0.995` gate; RECOVAR-vs-GT remains better at `0.670285`
versus `0.650835`. Numbered merged-map FSC-AUC is already `0.997721` at
iteration 2 and `0.999746` at iteration 10.

Post-rotation-only cutoff job `11025153` is a null result: supported-shell
iteration-1 FSC-AUC remains `0.9984304911` and shell 18 remains `0.948464267`.
The candidate is reverted. Downsampled BPref shell sums are already effectively
exact, including shell 18, which moves the first residual after reconstruction.

Source inspection identifies an ordering mismatch. RELION reapplies the
`ini_high` Fourier low-pass inside maximization, then calls real-space
`solventFlatten` from the outer iteration loop. RECOVAR did those operations
in reverse order. They do not commute because the final real-space mask
reintroduces a small high-shell tail. The next cheapest experiment corrects
that order and reruns one iteration.

One-iteration A100 job `11025949` confirms the fix. Canonical full-shell
RECOVAR-vs-RELION FSC-AUC is `0.999538`; supported-shell 1--18 FSC-AUC is
`0.999930`, shell 18 is `0.998800`, and the minimum non-DC shell is
`0.996857`.

Clean boundary-replay A100 job `11026304` passes the small-cell fixed-transition gate.
It completes in `579` seconds with the exact current-size schedule, convergence
at iteration 10, and final all-data path. Final RECOVAR-vs-RELION FSC-AUC is
`0.997260`; minimum non-DC shell FSC is `0.994984`, fifth percentile is
`0.995371`, and the last-ten-shell minimum is `0.996734`. RECOVAR-vs-GT
FSC-AUC is better (`0.670484` versus `0.650835`); only GT shells 1--3 are
lower, with worst delta `-0.000266`, well inside the `0.002` gate.

The active milestone now advances to K=1 robustness: run source-matched
RECOVAR/RELION pairs across high noise, nonuniform/Kent angles, no CTF,
outliers, contrast/noise-scale variation, and translation stress before the
10k/real/100k confirmations. Any failing cell returns to first-divergence
debugging; K=4 remains gated on this K=1 robustness step.

The eight-cell 3k/128 robustness matrix is now closed on the intended final
product. Jobs `11027056`--`11027063` established six direct passes and exact
convergence/finalization. The apparent failures in heterogeneous
contrast/noise-scale cases 18 and 22 were a parity-harness reporting bug:
`run_multi_iter_parity.py` discarded the joined all-data reconstruction in
`result["mean"]` and instead averaged the two separately Wiener-regularized
half reconstructions. Those operations are not equivalent. Production
`run_full_refinement.py` already saved the joined reconstruction correctly.
Commit `f91ba865` fixes the harness and adds a regression.

Focused A100 validations `11032906` and `11032907` pass after that correction.
Case 18 has RECOVAR-vs-RELION FSC-AUC `0.995571`, minimum non-DC shell FSC
`0.988215`, and GT FSC-AUC `0.765648` versus RELION `0.751884`. Case 22 has
RECOVAR-vs-RELION FSC-AUC `0.996966`, minimum non-DC shell FSC `0.991276`, and
GT FSC-AUC `0.335789` versus RELION `0.326059`. Their worst shellwise GT deltas
are only `-0.000632` and `-0.000188`, respectively. Thus all eight robustness
cells pass the aggregate FSC-AUC, GT-quality, convergence, and finalization
gates. The active K=1 step advances to a 10k intermediate-scale confirmation,
then a characterized real-particle case and the pinned 100k/256 completion
case; K=4 remains gated until those K=1 confirmations pass.

The two-cell 10k/128 intermediate-scale matrix is also closed for map quality,
convergence, finalization, memory, and matched-A100 timing. Immutable setup,
case, and summary jobs `11033444`--`11033447` ran from commit `3cbfd9ea` under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_mid10k_strict_retry_20260711_235500`
with separate marked runtime scratch at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_mid10k_strict_retry_20260711_235500`.
The summary job passed its exact-HEAD and clean-worktree gates.

Uniform/white case 31 matches RELION convergence at iteration 13 and the final
all-data branch. Final RECOVAR-vs-RELION FSC-AUC is `0.998723`; minimum non-DC
shell FSC is `0.997648`, fifth percentile is `0.997682`, and the last-ten-shell
minimum is `0.997653`. RECOVAR-vs-GT FSC-AUC is `0.818646` versus RELION
`0.801765`, with worst shellwise GT delta only `-0.000453`. RECOVAR refinement
wall is `1009.8` seconds versus RELION `1702` seconds on matched A100s, a
RECOVAR/RELION ratio of `0.593`.

Kent/radial-noise-3 case 32 matches RELION convergence at iteration 11 and the
final all-data branch. Final RECOVAR-vs-RELION FSC-AUC is `0.998250`; minimum
non-DC shell FSC is `0.996442`, fifth percentile is `0.996871`, and the
last-ten-shell minimum is `0.997256`. RECOVAR-vs-GT FSC-AUC is `0.272194`
versus RELION `0.268373`. The localized GT-shell swing at shells 3--5 is not a
map-parity deficit: RECOVAR-vs-RELION FSC is `0.999743` or better over shells
1--5, and RECOVAR is better in aggregate and through the later signal-bearing
shells. RECOVAR refinement wall is `950.9` seconds versus RELION `1154`
seconds, a ratio of `0.824`.

The remaining 10k state tail is an explicit diagnostic, not a map-quality
failure. Final pose p95 is arithmetic-level in both cells and every pose is
within 5 degrees, but case 31 Pmax p95/max absolute gaps are
`0.01166/0.254996` and case 32 gaps are `0.002476/0.044970`. Case 32 has 70
adjacent-fine-grid pose flips above 1 degree and three translation differences
above 0.5 pixel. Before the real-particle and pinned 100k gates, adjudicate
representative particles with fixed-state RECOVAR score/posterior dumps and an
uninterrupted instrumented RELION run; continuation dumps are forbidden by the
previously demonstrated finalization-state confound.

The first characterized real-particle gate is open and currently fails strict
parity. A deterministic 10k-particle EMPIAR-10076 subset (seed `20260712`,
exactly 5000 particles per half) lives under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_10k_fixture_20260712`
with a `SAFE_TO_DELETE` marker and manifest. The shared RECOVAR mean volume is
used only as the common initializer; there is no pseudo-GT, so the quality
gate is RECOVAR-vs-RELION shellwise FSC/FSC-AUC plus half-map and state parity.

The first A100 pair exposed a real-subset indexing bug before RECOVAR science.
`half1_idx` and `half2_idx` are row positions in the subset STAR, but replay
overrides treated them as original stack IDs. Synthetic contiguous fixtures
hid the bug. Commit `c3b3a27e` maps each input row through `rlnImageName`,
validates missing and duplicate stack IDs, and adds a shuffled non-contiguous
regression. The full override unit file passes (`38 passed`). Corrected
RECOVAR-only job `11039455` completed from that commit on the same A100 model
as RELION; failed jobs `11039371` and `11039400` were pre-science
`CUDA_ERROR_NO_DEVICE` failures on unhealthy `della-l07g3`, which is excluded
from the corrected runs.

RELION converges at numbered iteration 17 and performs the final all-data
branch; RECOVAR reproduces the same control decision and final branch. The
underlying real-data state is not yet strict, however. Iteration-2 mean Pmax is
`0.0716` in RECOVAR versus `0.112067` in RELION, accompanied by a transient
support tail: median fine support is 32 rotations, but a few particles retain
up to 200704 rotations. Iteration 2 takes `899.3` seconds. The tail collapses
by iteration 3 and mean Pmax nearly recovers by iteration 4 (`0.6624` versus
RELION `0.663427`), local iterations take 31--94 seconds, and final mean Pmax
matches (`0.243046` versus `0.243119`). Per-particle final state still fails:
mean/p95/max absolute Pmax gaps are `0.082570/0.226190/0.570284`; pose p95 is
`0.759678` degrees and translation p95 is `0.641700` pixel.

Final map parity fails and must not be waived by the high correlation. Final
RECOVAR-vs-RELION FSC-AUC is `0.863672` while correlation is `0.988477`.
Truncated FSC-AUC is `0.998450` through shell 16, `0.992968` through shell 32,
and `0.964539` through shell 64, showing a smooth signal-band phase loss rather
than a single corrupt shell. RECOVAR wall time is `2375.7` seconds versus
RELION `1330` seconds; the ratio is `1.786`, dominated by the iteration-2
support transient. Artifacts and summary are under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_10k_strict_20260712_010000`
(`summary_retry.md` and `summary_metrics_retry.json`). The next quality
hypothesis is an iteration-2 firstiter-CC support/normalization mismatch:
select the largest per-particle Pmax/support residuals, freeze the same RELION
state and maps, and compare raw scores, priors, support masks, log-normalizers,
and winners before making performance changes.

Fixed-state jobs refine that boundary further. One-iteration replay job
`11041292` starts from RELION iteration-1 maps and state and produces
iteration-2 FSC-AUC `0.999321`, mean Pmax `0.112001` versus RELION `0.112067`,
mean absolute per-particle Pmax gap `0.000494`, and pose agreement through p99.
Its mean fine support is only `298.5/322.9` rotations per half, versus
`1180.9/1112.4` in the boundary-replay trajectory. Thus iteration-2 scoring is not the
primary source; it amplifies a preceding map difference. Direct iteration-1
job `11041546` proves the first divergence: Pmax is exactly 1 for every
particle, poses/translations match through p99, but the iteration-1 merged-map
FSC-AUC is only `0.988635`. The active hypothesis is therefore iteration-1
M-step support/weights, BPref accumulation, or reconstruction/filtering.
Compare the saved RECOVAR `Ft_y`, `Ft_ctf`, regularized/unregularized maps,
tau2, and noise under `iter1_map_diag/output/intermediates` against an
uninterrupted, identity-validated RELION iteration-1 BPref dump.

That accumulator boundary is now closed against the exact real-data oracle.
An isolated RELION `d476e6f` build adds a single environment-gated dump after
MPI combination, symmetry, and the 40-A low-resolution half join but before
reconstruction.  Installed-d476 stop-one versus the uninterrupted oracle,
installed versus patched with the environment unset, and patched environment
off versus on all pass at half-map FSC-AUC above `0.9999999996`, minimum
non-DC shell FSC above `0.9999999940`, maximum real-space delta
`1.8626451e-9`, and bit-exact Pmax/pose/translation arrays for all 10,000
particles.  Jobs `11042379`, `11042605`, and CPU analysis retry `11043664`
produced the qualified dump under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_d476_bpref_real10076_identity_20260712_020200`.

The RECOVAR and RELION post-join BPref accumulators are close but not
identical: complex-average coordinate relative-error medians are about
`2.0e-4`/`2.2e-4` for halves 1/2 (p95 `0.0449`/`0.0569`), while weight
medians are `3.9e-6`/`4.8e-6` (p95 `0.0063`/`0.0091`).  The coordinate frame
is unambiguous: permutation `(1,0,2)`, signs `(1,1,1)`, complex sign `-1`.
However, decisive cross-replay jobs `11043878` and corrected `11043975`
exonerate this residual.  Replaying the saved RECOVAR accumulators reproduces
the saved RECOVAR merged map at FSC-AUC `0.99999998`; replacing only the
accumulators with exact RELION values changes the replay by FSC-AUC only
`~7e-6` (`0.9999930` between replacement maps) and changes the comparison to
RELION only from `0.9886340` to `0.9886556`.  The real-data iteration-1
failure is therefore downstream of accumulation, in reconstruction or
post-processing.  Stage ablation job `11044058` confirms that solvent masking
is required and grid-on is the closer branch, but neither the raw,
initial-low-pass, masked, nor grid on/off RECOVAR variants closes the gap.
The next boundary is an identity-validated RELION dump immediately after
reconstruction, after the iteration-1 low-pass, and after solvent flattening.

Exact stage-map job `11044321` closes that boundary.  The stage hook is
observationally inert: patched environment-off versus environment-on half-map
FSC-AUC is above `0.99999999982`, minimum non-DC shell FSC is above
`0.9999999968`, and all particle/model arrays are exact.  RELION's captured
post-solvent maps are bit-exact with its written iteration-1 maps.  With exact
RELION accumulators, however, the first mismatch is already the raw
reconstruction: supported-shell FSC-AUC is only about `0.6003/0.6009` for the
two halves and `0.5968` merged.  The supported post-low-pass comparison rises
to about `0.9596`, while the mask turns the phase error into the familiar
full-shell final FSC-AUC `0.988656`.  Thus neither low-pass nor solvent-mask
ordering is the primary source.

The initial hypothesis that RECOVAR omitted RELION's default iterative
preweighting is false.  Exact-d476 binding probe `11045912` proves that RELION
5.0.1 defaults to `skip_gridding=true`; its native closed-form skip branch,
using the pre-filter RELION tau operand, reproduces the captured
post-reconstruct maps at supported-shell FSC-AUC approximately
`0.99999999999999`, full FSC-AUC above `0.99999999996`, relative L2 about
`1.2e-7`, and maximum real-space delta about `6.3e-9`.  Setting zero
preweight iterations with `skip_gridding=false` is identical, while enabling
ten iterative preweight steps is slightly worse and substituting the saved
post-hoc filtered tau is materially worse.

Source audit then identifies the smallest correction.  RECOVAR currently
applies `_firstiter_cc_ini_high_tau2_taper` to
`mean_signal_variance_per_half` before reconstruction.  RELION reconstructs
with the untapered tau, calls `initialLowPassFilterReferences` on the map, and
only then tapers tau2/data-vs-prior for the model/reporting state; its source
explicitly notes that those tapered values are not used in calculations.
Preserve untapered per-half tau for reconstruction, create the tapered state
copy afterward, and retain the existing post-reconstruction map low-pass and
solvent mask.  Do not add iterative preweighting or a C++ production solver
unless the corrected direct-JAX stage replay still leaves a measured gap.

The corrected direct-JAX stage replay and production iteration now pass.
Checkout-bound A100 replay `11046061` uses exact RELION accumulators plus
RECOVAR-computed untapered tau and reaches merged post-reconstruct
supported-shell FSC-AUC `0.999998526`, active post-low-pass FSC-AUC
`0.999999998`, and final solvent-masked full FSC-AUC `0.999999993` with
minimum non-DC FSC `0.999999844`.  Patched production job `11046453` reruns
the real 10,000-particle first iteration end to end and reaches merged
FSC-AUC `0.999991962`, minimum non-DC FSC `0.999984135`, and half-map
FSC-AUC `0.999984760/0.999991332`; Pmax is bit-exact for all particles and
pose/translation p95 errors remain at the numerical-noise scale.  Earlier
job `11046279` is rejected because direct script execution resolved submodules
from a different editable checkout; the corrected runner uses module execution
and asserts the exact `iteration_loop.py` and CUDA wrapper paths.

Full patched trajectory `11046636` matches RELION's convergence at numbered
iteration 16 and executes the final all-data branch, but correctly fails the
map gate: merged FSC-AUC is `0.859718` and half-map FSC-AUCs are
`0.835102/0.843042` despite diagnostic correlation `0.988109`.  The first
post-fix divergence is not iteration-2 scoring.  Map-only cross-replay
`11047995` starts from the corrected RECOVAR iteration-1 half maps plus exact
RELION iteration-1 state and obtains iteration-2 mean Pmax `0.112050` versus
`0.112067`, merged-map FSC-AUC `0.999299`, pose p95 zero at printed precision,
and translation p95 zero at printed precision.

The failed full trajectory did not start from the same iter-0 particle state.
Its iteration-1 mean angular error is about 92 degrees and translation error
about 11.7 pixels; tau2 already differs by up to 59.3% across active shells.
`run_full_refinement --relion_init_dir` loaded iter-0 noise/tau but omitted the
run_it000 particle pre-centering offsets (about 8.5 pixels mean absolute
component), previous orientations, image/scale corrections, and direction
prior.  Commit `a530ec6f` reuses the typed replay loader to install that
complete run_it000 state in override slot 0.  Unit replay tests pass `8/8`.
One-iteration A100 gate `11048426` then reaches merged FSC-AUC `0.999991954`,
half-map FSC-AUCs `0.999984759/0.999991342`, minimum non-DC merged FSC
`0.999984127`, and exact Pmax/rotation/translation arrays relative to the
qualified corrected runner.  Two-iteration A100 gate `11048692` then closes
the next handoff: half-map FSC-AUCs are `0.999051092/0.999183696`, merged
FSC-AUC is `0.999318831`, and minimum non-DC shell FSC is at least
`0.975355`.  Mean Pmax is `0.1120446` versus RELION `0.1120673`; pose and
translation p95 errors are at printed numerical precision, with rare discrete
tail changes retained for tie-aware inspection.  Corrected full real-data job
`11049135` now runs from clean commit `2e3cc620` and must match numbered
iteration-16 convergence plus the normal final all-data branch before this
trajectory is closed.

Job `11049135` matches that control flow exactly but does not close map parity:
it converges at numbered iteration 16, runs the normal iteration-17 all-data
branch with grid correction off, and reaches merged FSC-AUC `0.974017` plus
half-map FSC-AUCs `0.947021/0.944168`.  This is much better than the stale
cold-start run's merged `0.859718`, but still fails the strict `0.995` gate.
The first material post-iteration-2 drift is projector-dependent.  Starting
from the qualified iteration-2 maps and exact RELION state, iteration-3 job
`11050464` with CUDA texture interpolation gives merged FSC-AUC `0.992097`,
Pmax MAE `0.012945`, and mean Pmax gap `+0.000802`.  Same-A100 manual/JAX
projection job `11050495` improves those to `0.995407`, `0.006198`, and
`+0.000189`, respectively, while moving the pose/translation tails closer to
RELION.  Commit `db814243` therefore makes the manual projector the parity
default and keeps texture interpolation as the explicit
`RECOVAR_RELION_PROJECTOR_TEXTURE_INTERP=1` diagnostic.  Full manual-projector
trajectory job `11050804` is the next gate.

Source audit finds the largest texture-path defect: RELION clips each
projection to the smaller of the PPref/model radius and the current image
radius, while RECOVAR texture projection previously enforced only the PPref
radius.  Commit `81681151` adds the missing current-image disk mask without
changing the manual default.  K=4 probe `11051135` moves first-iteration
occupancies from the broken `[2708,1768,2405,3119]` to
`[3216,1710,2200,2874]`, toward RELION's `[3369,1828,2045,2758]`, but does not
close the residual.  Keep texture opt-in until its remaining even-box
Nyquist/coordinate arithmetic is isolated.

K=1 iteration-3 probe `11051461` confirms that the mask removes most of the
texture regression: merged FSC-AUC improves from `0.992097` to `0.994951`,
mean Pmax gap improves from `+0.000802` to `-0.000006`, and Pmax MAE improves
from `0.012945` to `0.007292` in 212 seconds.  The manual path remains slightly
better in map FSC-AUC (`0.995407`) and Pmax MAE (`0.006198`) but takes 357
seconds.  Quality therefore stays on manual projection while the fixed texture
path becomes the leading later performance candidate.

The paired full trajectories close the projector-only hypothesis but not the
real-data quality gate.  Manual job `11050804` and current-radius-masked texture
job `11051785` both reproduce RELION's complete current-size/healpix schedule,
converge at numbered iteration 16, and execute the normal iteration-17
all-data branch with grid correction off.  Manual final half-map FSC-AUCs are
`0.950676/0.948478` and merged FSC-AUC is `0.977975`; masked texture gives
`0.950195/0.945893` and `0.977663`.  Both therefore fail the immutable `0.995`
map gate.  Manual uses 3402 seconds externally versus 2077 seconds for texture
on the same A100 model (peak GPU memory `41156/41150` MiB).  The tiny manual
FSC advantage keeps it as the strict quality default; texture remains the
qualified speed diagnostic, not an accepted quality replacement.

The cumulative residual is a sparse early hypothesis tail, not a mismatch in
the averaged control trajectory or internal gold-standard FSC.  Against each
numbered RELION model, RECOVAR's mean internal half-map FSC differs by at most
`4.99e-4` through iteration 16 and the shellwise mean absolute difference is at
most `0.001073`.  At iterations 1--3, pose p95 remains at numerical precision,
but the fraction above one degree is `0.40%`, `1.09%`, and `1.04%`; the
corresponding mean pose errors are `0.0282`, `0.2255`, and `0.2741` degrees.
The tail grows through the global iterations and is then partly recovered by
local search.  Final manual all-data pose mean/p95 is `0.1650/0.6396` degrees
and translation mean/p95 is `0.0465/0.4538` pixel.  This is too large to waive
as a discrete tie without score evidence even though the majority is exact.

The first tie adjudication targets fixture row 1474,
`19638@particles.256.mrcs`: it is arithmetic-level at iteration 1 but selects
a pose 175.55 degrees away and a translation 2.24 pixels away at iteration 2,
while RECOVAR/RELION Pmax is only `0.040652/0.039912`.  The initial f2c-based
dump job `11053175` is rejected because its newer binary does not reproduce
the d476 oracle.  An exact-source d476 binary was then built with a minimal
score hook (build job `11053568`, binary SHA-256
`f1b27fe6472dac204b579d6163e2c8a0edcb0d6f0ad5904e87fce0078fd339cb`).
Its enabled job `11053649` also fails the mandatory same-binary observational
inertness gate against dump-disabled control `11053938`: schedules match and
iteration-2 map FSC-AUC remains above `0.9999999984`, but target rows 1474 and
5550 flip by `127.78` and `172.19` degrees, row 5550's class field is corrupted,
and one non-target shift changes.  Therefore every score/posterior array from
that hook and the cancelled six-particle RELION panel `11054149` is
scientifically inadmissible.  The audit is in
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_iter2_min_score_d476_20260712_073000/dump_instrumentation_inert_audit.json`.

The independently valid translation-coordinate audit resolves an apparent
candidate mismatch.  RELION pre-shifts by
`B = round_away_from_zero(old_absolute_px)`, scores a relative translation
`t`, then stores `new_absolute_px = B + t`; the inverse is `t = new - B`, not
`new - old`.  Dump-disabled control `11053938` therefore maps row 1474 to fine
translation 56 and row 5550 to 34.  The free texture row-5550 winner maps
exactly to fine translation 95 rather than being one pixel off.  This formula
and the STAR-derived evidence remain admissible, but any accompanying RELION
log-weight comparisons remain rejected until a redesigned hook passes the
enabled/disabled control.  Evidence is in
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_iter2_scoretail_fixedstate_20260712_072000/translation_coordinate_mapping.md`.

RECOVAR exact-state panel `11054150` completes six independently verified
fixture rows on an A100 and provides a fail-closed target set for the eventual
inert RELION comparison.  Its Pmax/top-two log-margin pairs are row 5550
`0.043996/0.072384`, row 5504 `0.003155/0.022116`, row 3102
`0.061285/0.012997`, row 394 `0.116131/0.013222`, row 2813
`0.375569/1.75604`, and row 7710 `0.364002/0.130013`.  Canonical
`source_indices.npy` and the fixture STAR independently confirm every
fixture-row-to-stack-image mapping.  The prepared comparator intentionally
raises until an admissible RELION dump is supplied.  Summary artifacts are in
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_iter2_recovar_panel6_20260712_075000`.

The controlled RECOVAR projector A/B on the exact RELION iteration-1 state
(texture job `11054266`; manual job `11054283`, cancelled after both requested
dumps were complete) selects the same candidate with both arithmetic paths for
both tail rows: `[289443,56]` for row 1474 and `[162979,34]` for row 5550.
Manual versus
texture posterior L1 is `0.008714`/`0.004191`; centered pre-prior score
difference p95 is `0.02995`/`0.02588`, and projection relative L2 is
`0.000889`/`0.000790`.  Thus projector arithmetic perturbs the controlled
surface but does not itself flip these two winners; free-trajectory flips
require amplification through the preceding map/state trajectory.  This is a
RECOVAR-only diagnostic and does not substitute for the rejected RELION score
comparison.

Pixel attribution rules out the remaining texture boundary hypotheses.  In
the same two exact-state dumps, shells 1--10 contain `97.1%/95.7%` of the
manual-versus-texture projection-difference energy, while boundary shell 46
contains only `0.0181%/0.0278%` and all coordinates outside radius 46 are zero
in both paths.  Removing the Nyquist row/column leaves `99.90%/98.46%` of the
candidate score-delta RMS.  The texture residual is therefore interior
interpolation arithmetic concentrated at low frequency, not another
support-radius or even-box boundary error.  The reproducible CPU audit is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_iter2_recovar_manual_p1474_p5550_20260712_075700/projector_pixel_audit.json`.

The first exact-d476 hook failure is fully explained: `ihidden_overs` is
authoritative only in host memory, but the hook copied its uninitialized CUDA
buffer back over that mapping before fine winner selection.  A host-only
replacement removes all extra GPU copies and syncs.  Cross-node jobs
`11054698/11054699` no longer corrupt either target and keep iteration-2 map
FSC-AUC above `0.9999999985`, but naturally differ at one non-target
translation, 13 significant counts, and Pmax by at most `0.000404`.  Same-node
sequential pair `11054601` likewise preserves both targets and all angles/classes
but differs at two non-target one-step translations and 17 significant counts,
so its strict fail-closed audit remains false.  Row triangulation shows the
enabled translations match both the original oracle and clean installed-binary
repeat `11055156`; the disabled control is the outlier.  The clean repeat itself
has iteration-2 Pmax delta at most `0.000126`, 17 one-count differences, no
translation differences, and map FSC-AUC above `0.9999999988`.  These establish
the natural RELION numerical envelope but do not waive the dump gate.

The v3 hook buffers host snapshots and defers all file writes until the
unconditional post-expectation MPI barrier, removing I/O from the
particle-scoring loop.  Same-node continuation gate `11055736` runs disabled
then enabled from the identical installed iteration-1 state and
passes the calibrated admissibility gate: iteration-2 map FSC-AUC is
`0.99999999855/0.99999999886`, minimum non-DC FSC is above `0.9999999826`, no
translation or class differs, maximum angle delta is `3.42e-6` degrees, Pmax
delta is at most `0.000117`, and nine significant counts differ by one.  Both
target states are exact, their Pmax deltas are `1.4e-5`, schedules are exact,
and all 26 dump files are present.  These changes are no larger than clean
installed-repeat `11055156`, so
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_iter2_inert_dump_control_20260712_080200/v3_exact_inertness_audit.json`
sets `score_arrays_admissible=true` for hook inertness on that continuation.
Failed precursor `11055708` stopped before scoring because of the wrong
continuation working directory and has no scientific output.

However, `--continue run_it001_optimiser.star` does not replay the uninterrupted
installed iteration-2 trajectory: the v3 disabled continuation versus oracle
map FSC-AUC is only `0.86405/0.86135` and many particle states differ, likely
because continuation does not restore the same perturbation/RNG sequence.
Thus `11055736` qualifies the hook but its score arrays are not the oracle
surface and must not be compared to RECOVAR's exact-oracle dumps.  A cold v3
two-iteration dump run must first reproduce the uninterrupted oracle target
states and candidate grid.

Cold v3 job `11055888` restores the correct `+0.405200` perturbation and
qualifies the target score surfaces.  Both row-1474 and row-5550 discrete
states match the uninterrupted oracle; target Pmax is exact and within
`2e-6`, respectively.  All 10,000 Euler/class arrays match to a maximum
`2.96e-6` degrees, with one known non-target one-step translation and 14
one-count differences from rebuilt-binary numerical variation.  Half-map
FSC-AUC is `0.9999999093/0.99999999865`, minimum non-DC FSC is at least
`0.9999996170`, and the 26 target arrays are admissible under the separate v3
inertness gate.  Qualification is in
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_iter2_inert_dump_control_20260712_080200/v3_cold_oracle_qualification.json`.

The admissible score comparison reverses the projector conclusion inferred
from final-map FSC alone.  Texture arithmetic is much closer to RELION on both
controlled oracle surfaces.  For row 1474, texture versus manual posterior L1
is `0.000291/0.008132`, centered pre-prior p95 absolute error is
`0.000626/0.029760`, and support symmetric difference is `0/6`.  For row 5550,
the corresponding values are `0.001086/0.004452`, `0.005816/0.025355`, and
`9/5`.  RELION, texture, and manual choose the same controlled winners
`[289443,56]` and `[162979,34]`; texture Pmax and top-two margins are also
closer.  Manual's tiny full-trajectory FSC advantage is therefore a
compensating error, not evidence of closer E-step arithmetic.  Do not optimize
or retain manual as the strict default without reconciling this score boundary.
The full comparison is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_iter2_inert_dump_control_20260712_080200/v3_cold_oracle_score_comparison.json`.

The qualified cold-v3 six-particle panel `11056138` extends that conclusion.
Five rows choose the same winner, with shared-surface posterior L1 between
`0.000704` and `0.002227` and common-centered pre-prior score p95 error between
`0.00262` and `0.00678`.  Row 2813 is the decisive non-tie: RELION chooses
`[289965,48]` at Pmax `0.116213` with top-two log margin `0.184082`, while the
texture RECOVAR path chooses `[284536,4]` at Pmax `0.375569` with margin
`1.75604`.  Only `49.80%` of RECOVAR's mass lies on RELION candidates, although
the shared-candidate centered score p95 error is only `0.003590`.  The bug is
therefore the coarse-parent support supplied to fine pass 2, not a numerical
tie or fine texture-score arithmetic.  Qualification and comparison are in
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_iter2_v3_panel6_20260712_085000`.

RECOVAR support audits localize the mismatch one pass earlier.  For row 5550,
RELION has 4,192 fine candidates and RECOVAR has 4,160; the 64 RELION-only
candidates are exactly two 32-child coarse parents `(22971,19)` and
`(22972,28)`, while RECOVAR adds parent `(2404,9)`.  RECOVAR ranks those parent
pairs 130, 131, and 126 around a rank-129 significance boundary, proving a
coarse-score ordering difference rather than a fine-expansion bug.  The row
2813 projector A/B is stronger: manual coarse projection selects RELION's exact
15 parent pairs and exact 480 fine candidates; texture swaps RELION parent
`(36245,8)` for `(35567,1)`.  Manual fine arithmetic is less accurate than
texture, so strict scoring requires a hybrid: manual supplied-PPref projection
for global coarse significance and texture supplied-PPref projection for fine
pass 2.

Minimal RELION v4.1 support probes passed the calibrated inertness gate
`11056477`; cold oracle run `11056533` preserves perturbation `+0.405200`, the
target state, and Pmax, with half-map FSC-AUC
`0.99999990925/0.99999999862`.  However, the hook queried RECOVAR psi-major
rotation IDs directly in RELION's pixel-major mask.  Its original three-zero
interpretation is coordinate-wrong and rejected.  Applying
`rel_rot=(rec_rot % 768)*48 + rec_rot//768`, the admissible v3 candidate arrays
establish membership `[1,1,0]` for `(22971,19)`, `(22972,28)`, and `(2404,9)`.
V4.1 therefore qualifies only the hook's inertness, not those requested support
booleans.  Audits are
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_iter2_v41_probe_gate_20260712_092500/v41_inertness_audit.json`
and
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_iter2_v41_probe_cold5550_20260712_092500/cold_probe_qualification.json`.

Commit `536d6bd9` implements an explicit typed coarse-projector choice and
bypasses the texture selector at the manual supplied-PPref leaf; native
RECOVAR projection behavior and fine pass 2 remain unchanged.  The first four
apparent hybrid retries (`11056492`, `11056559`, `11056716`, and `11056983`)
were invalid diagnostics because `python scripts/run_multi_iter_parity.py`
loaded EM submodules from the stale editable checkout
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_em_parity_100k_20260712`.
They are texture-identical and must not be cited as tests of the patch.  The
runner now invokes `python -m scripts.run_multi_iter_parity`, isolates its
per-job Python bytecode cache, and asserts the concrete iteration-loop,
K-class, and significance module paths.

Import-bound hybrid job `11057315` is the accepted row-2813 gate.  Its coarse
dump is bit-exact to every array in the qualified manual baseline, including
`normalization_log_z=65.99702880122135`, inclusion of coarse candidate
`1051113`, and exclusion of texture-only `1031444`.  Fine candidate and
reconstruction supports exactly match RELION (`480/480` candidates and
`189/189` reconstruction hypotheses), the winner is exactly `[289965,48]`,
and Pmax is `0.1162683021` versus RELION `0.1162131086` (gap `5.52e-5`).
Posterior L1 is `0.0007255`; common-centered pre-prior score p95/max error is
`0.00359195/0.00455151`.  The job was intentionally cancelled after both dumps
completed.  Machine-readable qualification is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_global_pass1_bound_import_assert_20260712_115000/hybrid_row2813_qualification.json`.

Import-bound hybrid panel `11057457` completes the six-target fixed-state gate;
the job was intentionally cancelled after all 12 requested dumps completed.
All six winners now exactly match RELION.  Candidate symmetric differences
versus RELION change from texture to hybrid as follows: row 394 `0→32`, row
2813 `64→0`, row 3102 `32→0`, row 5504 `3392→512`, row 5550 `96→64`, and row
7710 `64→0`.  Reconstruction-support differences are `5,0,0,139,5,0` in the
same order.  Pmax absolute gaps range from `3.04e-6` to `8.78e-4`; common-score
p95 errors remain `0.00274–0.00675`.  Thus the hybrid closes the known non-tie
winner bug and improves five support sets, but manual coarse support is not
universally exact.  Machine-readable comparison is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_iter2_hybrid_panel6_20260712_121000/hybrid_panel6_relion_comparison.json`.

The exact-texture Euler hypothesis remains open, but the first NumPy float32
proxy is now qualified as inexact.  Valid typed-texture row-2813 job `11058151`
moves centered coarse scores by RMS `0.02421` and moves 66.2% of hypotheses
toward the manual score surface, yet leaves the wrong texture parent swap
unchanged.  RELION source audit subsequently found that the proxy precomputed
`float32(pi)/float32(180)`, whereas the active CUDA kernel evaluates
`angle*float32(pi)/float32(180)` and may contract later operations.  Correcting
only that operation order changes 30,582 of 36,864 matrices (entry max
`6.56e-7`).  Therefore this job rejects only the NumPy proxy, not exact RELION
device Euler arithmetic.  Full audit:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_relion_euler_exact_audit_20260712_132500/AUDIT.md`.
Corrected multiply-then-divide typed-texture job `11058615` is also support-
negative: its significant mask is bit-identical to the accepted texture
control and retains the same row-2813 swap.  CUDA 12.6 and 12.8 produce
identical normalized sm80 SASS for the RECOVAR texture kernel, ruling out the
compiler-version hypothesis.  Pinned RELION jobs `11058907/11058986` then
dumped the exact device matrices.  They differ from RECOVAR's default matrices
in 257,338 of 327,024 active entries (p95 `2.38e-7`, max `6.85e-7`), but exact-
table injection job `11059563` leaves the texture support mask bit-identical
and worsens the decisive preference from `-0.00972366` to `-0.01270676`.
Exact Euler arithmetic is therefore ruled out causally.  Qualification:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_row2813_exact_relion_euler_texture_retry_20260712_154000/qualification.json`.

Raw-map PPref provenance is ruled out causally by corrected job `11058908`.
Bypassing the RECOVAR Fourier round trip from the matching raw `run_it001`
half maps leaves the accepted-control support mask unchanged; centered
with-prior RMS is only `1.70553e-5`, and the decisive pair preference moves
`-2.28882e-5` away from RELION/manual.  Qualification is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_row2813_rawmrc_it001_ppref_texture_coarse_20260712_111000/qualification.json`.
The preceding job `11058764` used off-state `run_it002` maps and is explicitly
invalid, not evidence.

Typed-texture six-row discriminator `11058143` was cancelled after all six
coarse dumps completed.  Across the six full 1,069,056-hypothesis arrays,
manual and texture coarse supports differ from RELION by 19 and 18 parents,
respectively, and differ from each other by 25.  Neither projector is globally
exact.  Row 2813 has the decisive RELION/manual parent `(36245,8)` versus
texture `(35567,1)` swap; row 394 is exact under texture but gains one manual
parent; row 5504 improves from 16 manual to 14 texture differences but changes
the support count from RELION 6509 to 6501; row 5550 is identical between both
RECOVAR projectors and retains a two-parent swap versus RELION.  Machine-readable
evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_iter2_texture_coarse_panel6_20260712_104500/texture_coarse_discriminator.json`.

Import-bound full 10k hybrid trajectory job `11057493` completed all 16
numbered iterations plus the converged final all-data iteration.  It terminates
at iteration 16, matching RELION.  The final merged-map RELION FSC-AUC is
`0.980495` (shells 1--16 mean FSC `0.999772`); half-map FSC-AUC is
`0.953110/0.949810`.  Final all-data pose error is `0.150624` degrees mean and
translation error `0.0438001` pixels mean.  RECOVAR wall time is 2021.1 s
versus RELION 1330 s (`1.51963x`), with sparse pass 2 consuming 862.42 s.
Summary:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_10k_hybrid_full_20260712_121500/summary.md`.

The even-size Nyquist-coordinate mismatch is causal and fixed by `39dc2ce2`.
RELION keeps the two surviving coarse-disk endpoints as `(+N/2,0)` and
`(0,+N/2)`, while RECOVAR sampled them with negative Nyquist coordinates and
then relabelled the output.  Row-2813 positive-Nyquist job `11059889` flips
exactly the two discrepant hypotheses.  Six-row panel `11059982` plus stable
row-7710 retry `11060171` match every direct RELION `(rotation,translation)`
support set exactly: counts `71,15,67,6509,131,24`, each with symmetric
difference zero.  Corrected RELION texture projection is now the default for
both coarse and fine supplied-PPref scoring; environment value `0` retains the
manual/JAX diagnostic fallback.  Qualification:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_iter2_nyquist_texture_panel6_20260712_153500/nyquist_panel6_qualification.json`.

Corrected-texture full 10k job `11060805` completes all 16 numbered
iterations, converges on the same iteration as RELION, and executes the normal
final all-data branch.  More importantly, its complete `(current_size,
resolution shell)` trajectory exactly matches the authoritative RELION model
STAR files.  At iteration 12 both select shell 32 (13.10 A): RELION FSC at
shells 32/33 is `0.512015/0.497433`, corrected RECOVAR is
`0.511150/0.498921`, while the older hybrid's `0.512818/0.500107` incorrectly
selects shell 33.  The apparent corrected-versus-RELION iteration-12 mismatch
was a diagnostic error caused by treating the older RECOVAR replay log as the
oracle.  Resolution audits must derive RELION's shell from each model STAR
`_rlnSsnrMap` (threshold 1, equivalently FSC 0.5), not another RECOVAR log.
Evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_iter12_schedule_audit_20260712_170000/REPORT.md`.

The corrected run's final merged-map RELION FSC-AUC is `0.979511`, with
half-map FSC-AUCs `0.953044/0.948438`.  This is slightly below the older
hybrid's `0.980495` and `0.953110/0.949810`; the loss is high-frequency
(shells 97--126 mean delta `-0.00283`) while shells 1--16 are unchanged to
`2e-6`.  This does not invalidate the corrected projector: the corrected run
is closer in the full control schedule and the six direct coarse support sets
are exact.  It instead leaves a downstream score/reconstruction arithmetic
boundary open.  Wall time improves from 2021.1 s to 1917.75 s, but remains
`1.44192x` RELION.  Summary:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_10k_corrected_texture_full_retry_20260712_121000/summary.md`.

Global-corrected-texture/manual-fine trajectory `11062767` is rejected.  It
diverges from the authoritative RELION SSNR schedule at iterations 12 and 14,
has merged/half FSC-AUC `0.979389` and `0.951552/0.947366`, and worsens final
pose/translation error to `0.157554` degrees and `0.0462333` pixels.  Its
3339.83 s wall time is `2.51115x` RELION and 74.2% slower than corrected
texture throughout.  Manual fine projection is therefore neither a quality
nor a speed solution; the older hybrid's small final FSC advantage is a
compensating error.  Summary:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_10k_global_texture_manual_fine_20260712_124110/summary.md`.

The admissible direct fine-projection operand gate localizes the remaining
score residual to Euler construction.  RELION operand job `11066282` preserves
all discrete arrays/support and shows that using RELION projections with the
saved RECOVAR image plus RELION's 128-lane reduction matches raw RELION scores
at the independent-rerun floor (centered p95 `0.000244141`, max `0.000488281`).
Using RECOVAR projections gives p95 `0.00341797`, and projection substitution
alone produces the same p95.  Exact device Euler job `11068941` finds zero of
480 candidate matrix rows bit-exact (matrix max absolute delta `5.96e-8`).
Injecting those exact matrices into the RECOVAR texture projector in GPU job
`11071482` reduces correct-half projection p95 error from `4.825e-5` to
`4.915e-7` (about 98x) and reduces score error back to the RELION rerun floor.

The cause is not a different Euler convention or missing RELION binding.
RELION's live iteration-2 perturbation is `0.4052000939846039`, while the
sampling STAR serializes only `0.405200`.  Seed-exact job `11075288` recovers
the live value from `_rlnRandomSeed`, makes all 480/480 effective candidate
matrices byte-identical to the accelerated RELION dump, and retains projection
p95 `4.915e-7`.  Commit `3917aa67` makes this the typed `auto` behavior,
verifies consistency with the rounded STAR, provides explicit `seed_exact`
and `star` modes, and falls back to STAR precision only when seed provenance
is unavailable.  Evidence root:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_row2813_relion_fine_euler_dump_20260712_132000`.

Six-target fixed-state job `11076618` validates the combined positive-Nyquist
and seed-exact path.  All six fine winners are exact.  Five targets have exact
candidate and reconstruction support; row 5504 differs by one coarse parent at
a `2.68e-10` weight cutoff and three reconstruction samples with probability
gaps at most `1.82e-11`, so every discrete difference is a demonstrated
threshold tie.  Centered fine-score p95 is `2.69e-4`--`5.96e-4`, at the
independent accelerated-RELION rerun floor, and Pmax gaps are at most
`8.70e-4`.  Machine-readable evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_iter2_fullpert_nyquist_panel6_20260712_193000/fullpert_panel6_relion_comparison.json`.

Clean combined full-trajectory job `11079185` nevertheless fails the strict
10k map gate.  It reproduces the full authoritative `(current_size,
resolution shell, HEALPix order)` schedule, converges at iteration 16, and
runs final all-data with parent/fine orders 6/7.  Final merged RELION FSC-AUC
is `0.978500`, versus `0.979511` for rounded replay; half-map FSC-AUC is
`0.950958/0.947709`.  The loss is high-frequency: new-minus-rounded merged
mean FSC is `-1.6e-6` at shells 1--16, `-1.03e-3` at 33--64, and `-2.40e-3`
at 65--96, with worst shell delta `-0.00580`.  Runtime is 1984.0 s, or
`1.4917x` RELION.  Full precision is closer in pose/Pmax through iteration 2;
iteration 4 is the first net particle-state worsening, after tiny prior-state
differences are amplified by rare global score ties.  Iteration-1 diagnostic
`11084547` rules out an early reconstruction bug: merged map FSC-AUC is
`0.999996565`, minimum non-DC shell FSC `0.999993239`, and all 10k Pmax values
are exactly one.  Evidence roots:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_10k_fullpert_finalorder_20260712_193000` and
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_iter1_fullprecision_boundary_20260712_151500`.

Four-iteration numbered-map job `11084946` reproduces `11079185` poses and
translations byte-exactly and localizes the visible early map residual to the
first shell outside current-size signal support.  Full-box merged FSC-AUC
falls from `0.9999966` at
iteration 1 to `0.9994003`, `0.9959409`, and `0.9905003` at iterations 2--4;
the minimum shell is exactly the current-size boundary (47, 61, and 62).
Through RELION's authoritative signal shell, however, merged FSC-AUC remains
`0.999999759`, `0.999996900`, `0.999983765`, and `0.999938423`, with minimum
shell FSC at least `0.999751`.  Thus no material signal-band reconstruction
drift precedes the iteration-4 particle flips.

A direct current-size-edge audit rules out a crop-origin or Nyquist-plane
defect.  The edge planes contain only
`0.073%`--`7.23%` of residual energy at iterations 2--4, fitted shifts are
below `0.001` pixel.  Uninterrupted raw-BPref job `11087020` is a
near-authoritative numerical oracle: half-map FSC-AUC against the installed
iteration 2 is `0.99999991/0.999999998`, all angles and X translations are
exact, and only two Y translations differ by one grid step at ties.  After
frame scaling, RELION-versus-RECOVAR BPref numerator relative L2 is
`1.77%/0.94%` by half while weight is `0.338%/0.200%`, but causal
cross-substitution remains unqualified because the first reconstruction probe
used `skip_gridding=False` and later probes paired the wrong BPref/map files.
The inherited claim that a RELION solver raised shell-47 FSC to `0.999147`
has no reproducible script or result artifact and is withdrawn.

Identical-input reconstruction is instead closed: on both RELION and RECOVAR
BPref operands, RECOVAR's wrapper and RELION's real `skip_gridding=True`
binding agree at current-support FSC-AUC above `0.9999999999999` and shell-47
FSC above `0.999999999996` after the documented frame transform.  The source
audit nevertheless finds three independent correctness defects in RECOVAR's
current-size wrapper: packed-half tau shells round padded radius before
division and then round again (mislabeling `388168/1643720` supported
iteration-2 voxels), numerator decenter incorrectly excludes the exact-radius
sphere that RELION includes, and the MAP prior needs its own strict-radius
support.  These fixes require a fresh four-iteration trajectory gate; they are
not yet a production-quality closure.  Evidence roots:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_iter4_fullprecision_boundary_20260712_153000` and
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_relion_iter2_bpref_dump_fullscratch_20260712_171500`.

Four-iteration validation job `11089339` retains those corrections.  It runs
from clean commit `fa597a61`, reproduces exact sizes `[48,92,120,122]`, and
completes in 937.6 seconds.  Merged authoritative current-support FSC-AUC at
iterations 2--4 improves from `0.999698/0.998491/0.996590` to
`0.999968/0.999711/0.999096`; the corresponding minimum support-shell FSC
improves from `0.982863/0.910681/0.800538` to
`0.999275/0.993299/0.992760`.  Iteration-4 full-box FSC-AUC rises from
`0.990577` to `0.998519`, and the two half-map edge shells rise from about
`0.789/0.791` to `0.993/0.992`.  Iteration-3 Pmax MAE falls from `0.003763`
to `0.002787`, pose outliers fall from 40 to 29, and translation outliers from
99 to 59.  Iteration-4 Pmax MAE also improves (`0.009267` to `0.008367`),
although sparse angle ties leave mean pose error slightly worse; aggregate
map quality, not correlation, is the retention gate.  Report:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fa597a61_iter4_validation_20260712_171214/VALIDATION_REPORT.md`.

Full corrected 10k job `11090698` also retains commit `fa597a61`.  It
reproduces the exact 16-iteration size/HEALPix schedule, converges at iteration
16, and runs final all-data with parent/fine orders 6/7, seed-exact
perturbation, and grid correction off.  Final merged FSC-AUC versus RELION is
`0.978674`, up from `0.978500`; half-map FSC-AUCs are
`0.952292/0.946336` versus `0.950958/0.947709`.  Final pose mean/p95 improves
from `0.16561/0.64240` to `0.15986/0.63442` degrees and translation mean from
`0.04928` to `0.04656` pixels, while Pmax MAE is essentially flat/slightly
worse (`0.033922` to `0.033939`).  Runtime is 2045.7 seconds, 3.11% slower
than the prior run and `1.538x` RELION.  The fixture contains no GT volume, so
no GT FSC claim is made.  The earliest remaining material boundary is the
iteration-8 global-to-local HEALPix-4 transition: minimum support FSC drops
from about `0.9766` to `0.9144`, pose p95 leaves the numerical floor for about
`2.39` degrees, and Pmax MAE reaches `0.0837`.  Next work must compare local
parent/fine scoring and support at that fixed transition, not revisit the now
closed identical-input reconstruction wrapper.  Report:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fa597a61_full10k_trajectory_20260712_173845/FINAL_REPORT.md`.

The fixed serialized RELION iteration-7 to iteration-8 replay exposes where
the global-to-local transition amplifies a residual, but it is not an
identical-input oracle for RELION's uninterrupted in-memory projector state.
Aggregate replay
job `11092142` starts from RELION iteration-7 maps and particle state, uses
the exact iteration-8 current size 122, parent/fine orders 4/5, and seed-exact
perturbation `-0.360924143344`.  It leaves `8802/10000` particle rotations at
the numerical floor, but the remaining 1198 differ by at least one degree;
pose p95 is `2.2536` degrees, translation p95 is `0.5` pixel, and mean Pmax is
`0.636480` versus RELION `0.628746`.  Its nonzero Slurm exit is an
instrumentation-gate failure after all science completed: the dump filter used
absolute iteration 8 while the one-step replay labels its runtime loop as
iteration 1.  Setup-only jobs `11091925` and `11091997` are also rejected
(CUDA library-path contamination and an unsupported CLI argument,
respectively).  Neither produced a scientific result.

Corrected dump job `11092382` completes with exit 0 and all twelve requested
particle surfaces.  Exact winner-to-grid mapping shows two distinct,
non-numerical failure modes.  For particles 5727 and 932, RELION's winning
coarse rotation parent is absent from RECOVAR pass 2; for particle 9887 the
rotation parent is present but its required coarse translation pair is
masked.  Regenerating the complete order-4 neighborhood and order-5 children
recovers all three RELION rotations, ruling out parent/child enumeration.
Within the serialized-state replay, their first divergence is pass-1 coarse
scoring, normalization, or the 0.999 significance selection.  Conversely,
RELION's uninterrupted winners for particles
3758, 4321, 5375, and 9826 are present but lose on RECOVAR's total score by
`30.306/0.893/0.528/1.367`; saved-operand recomputation agrees with the live
scores within about `0.002`, so reduction rounding cannot explain those
margins.  Particle 3758 is a material raw data-term inversion (`30.720`),
whereas the other three are flipped by RECOVAR's orientation/translation
priors after the RELION winner has the better RECOVAR raw term.  A cold
instrumented RELION iteration-8 run is therefore required to distinguish
coarse support, raw projection/image operands, and prior arithmetic directly;
continuation dumps remain inadmissible.  Evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fa597a61_fixed_rel7_to8_localdump_20260712_183500/retry3/SUPPORT_AUDIT.md`
and
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fa597a61_fixed_rel7_to8_localdump_20260712_183500/retry3/PASS2_SCORE_AUDIT.md`.

Matched RECOVAR coarse-table job `11092553` completes in 5:40 and proves
that the parent-support result is not a dump-mapping artifact.  For all twelve
particles, the coarse `reconstruction_sample_mask` pairs exactly equal the
fine table's finite candidates after collapsing each 8-by-4 child block back
to its parent.  The required RELION winner pairs for particles 5727, 932, and
9887 rank only 17, 13, and 3 in RECOVAR versus retained support sizes 6, 7,
and 1; their score gaps from the RECOVAR winner are `-12.8455`, `-3.98770`,
and `-14.23038`, and their posterior-to-cutoff ratios are
`2.64e-6`, `1.85e-2`, and `6.60e-7`.  These are material coarse score or
probability-table mismatches, not 0.999-threshold ties.  Particle 3758 gives
the complementary boundary: its RELION winner parent is RECOVAR's coarse
rank-1 pose with posterior `0.992`, so its `30.720` raw inversion arises only
after fine-child projection/scoring.  Evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fa597a61_fixed_rel7_to8_localdump_20260712_183500/retry4/coarse_dump/COARSE_DUMP_AUDIT.md`.

Cold operand-enabled RELION job `11092529` is rejected by the mandatory
observational-inertness gate.  It reproduces the exact iteration-1 through
iteration-8 schedule and perturbations, produces the expected 180 files, and
maps every requested particle ID correctly, but its iteration-8 half-map
FSC-AUC against the installed uninterrupted oracle is only
`0.985700/0.993777` with minimum non-DC FSC `0.963725/0.981571`.  It also
changes 684 particle angles above `1e-4` degrees, 818 translations, 1968
significant-sample counts, and Pmax by up to `0.68007`.  The Slurm job's
nonzero exit is the intended fail-closed post-science result (after correcting
an analysis-only STAR parser type assumption).  All 180 score/projection
files are quarantined and must not be used.  A redesigned minimal hook must
pass an enabled-versus-disabled cold control before direct RELION score
comparison resumes.  Gate:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_relion_it8_cold_score_oracle_20260712_184953/cold_it8_inertness_gate.json`.

The attempted score-only V3 pair `11092663/11092664` is also quarantined
before score use because its advertised build path had been mutated in place.
The genuinely qualified V3 binary from job `11055888` had SHA-256
`68982a12...` and produced 13 files per target; the current path instead had
SHA-256 `f77efbf3...`, contained the projection-operand hook, and produced 15
files per target.  The mismatch is scientifically visible: the same-binary
disabled run remains at HEALPix order 3 for iteration 8 while the enabled and
installed runs advance to order 4; enabled-versus-disabled half-map FSC-AUC is
only `0.706429/0.702250`, all 10,000 poses differ, and 8030 significant-count
rows differ.  No accepted marker exists.  Future instrumented builds must use
an immutable new build root and verify both source-diff and binary hashes
before submission.  Quarantine record:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_relion_it8_v3_cold_pair_20260712_191034/paired_cold_it8_qualification.json`.

Rebuilding the true V3 patch in a new immutable source root verifies exact
commit `d476e6f`, patch and full-diff SHA-256 `58579a10...`, the original
toolchain, and absence of all projection-operand code.  Build job `11093013`
correctly notes that byte identity cannot survive the path change: RELION
embeds absolute source/build paths in its executable.  A same-binary cold pair
is the stronger scientific test.  Jobs `11093087/11093088` produce the
expected 156/0 manifests and exact target mapping, but V3 is not inert through
iteration 8: the disabled run remains at HEALPix order 3 while enabled and
installed advance to order 4.  Enabled-versus-disabled half-map FSC-AUC is
`0.708193/0.703562`, with all 10,000 poses, 6292 shifts, and 7983 significant
counts differing.  Enabled-versus-installed FSC-AUC is only
`0.984972/0.992932`.  V3 was qualified for iteration 2 only and must not be
extrapolated to the later adaptive transition; its iteration-8 fine tables
are quarantined.  Gate:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_relion_it8_v3_immutable_pair_20260712_193549/paired_cold_it8_qualification.json`.

Fixed-transition projector A/B jobs `11093381/11093382` initially reveal a
routing defect rather than a scientific hybrid result.  The supplied-PPref
local bucket, packed-noise, and projection-cache helpers discarded their
explicit `projection_relion_texture_interp` argument and fell back to the
process-wide environment.  Consequently A was texture/texture and exactly
reproduced baseline, while B was manual/manual.  Manual/manual does not change
particles 3758, 5727, 932, or 9887 and slightly worsens mean angle,
translation, and Pmax parity; its FSC-AUC gain is only `1.47e-5`.  Their Slurm
`FAILED/1` states are a bad post-science regular-expression gate; both science
outputs are complete.  Report:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fa597a61_rel7to8_projector_ab_20260712_190500/A_B_REPORT.md`.

The production routing fix forwards the explicit selector through all three
supplied-projector paths, pins local parent pass 1 to manual interpolation,
and lets fine pass 2 follow the switchable texture default.  Validation job
`11093570` completes with exit 0 and logs the intended route for both halves.
It is exactly equal to the fixed baseline in every saved pose, translation,
Pmax, and per-half fine significant count; the four target particles remain
unchanged.  FSC-AUC changes by only `-5.22e-8` and maximum shell FSC by
`2.17e-6`.  The fix is retained as a correctness/configuration repair, but
manual-versus-texture selection is not the iteration-8 cause.  Report:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_local_hybrid_routing_validation_20260712_201000/VALIDATION_REPORT.md`.

An unmodified stock `relion_project` audit resolves the apparent particle-3758
fine-score contradiction and changes the causal boundary.  Projecting the two
candidate poses from the serialized iteration-7 half-map matches RECOVAR's
saved projection operands at relative L2 `0.000891/0.000832`.  Combining those
stock projections with RECOVAR's saved image/CTF/noise operands reproduces the
same material raw/total preference for RECOVAR's winner:
`30.7315/30.3182` versus RECOVAR `30.7196/30.3062`.  The opposite half-map
still prefers the RECOVAR winner by `18.98` raw-score units, stock CTF agrees
after the paired sign convention, and stock CTF-subtracted residuals also
favor the RECOVAR winner.  RECOVAR is therefore self-consistent with stock
RELION for the serialized map.  The uninterrupted RELION winner depends on
live reconstruction/projector state not recoverable from its written
map/model/data/sampling files, exactly as the failed continuation gates imply.
Iteration 8 amplifies an earlier map/projector-state difference; it does not
establish a local E-step bug on identical inputs.  Return the trajectory trace
to pre-iteration-8 reconstruction/projector formation.  Evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_it8_p3758_stock_project_20260712_203000/STOCK_PROJECTOR_AUDIT.md`.

The tempting indexed-backprojection coordinate-order explanation is rejected.
RELION rotates integer Fourier coordinates before multiplying by padding while
RECOVAR's CUDA kernel multiplies first, but padding factor 2 is exact binary
scaling: five million float32 coordinate cases are bit-exact.  The initial
dot-then-padding A100 probe used the wrong default box size and is invalid as
quantitative evidence; corrected box-256 pre-scatter and scatter probes instead
show common-mode data/weight behavior and do not support the coordinate-order
hypothesis.  The diagnostic source change was reverted.  The remaining
credible BPref mechanism is global translated complex numerator accumulation
and atomic ordering, which is more cancellation-sensitive than the positive
CTF-squared weight.

Reciprocal iteration-3 map-splice array `11086240_[0-3]` proves that low
shells through authoritative signal shell 29 dominate the iteration-4
particle divergence.  The authoritative-map control A and REL-low/REC-high C
retain `9787/10000` and `9895/10000` exact joint winners relative to the
boundary-replay RECOVAR-map B trajectory, while REC-low/REL-high D retains
`9887/10000` of B's
winners.  Iteration-4 merged FSC-AUC through shell 61 groups independently as
A/C (`0.998992/0.998972`) versus B/D (`0.998272/0.998286`).  A matches RELION
joint winners within `1e-4` for `9996/10000`; B reproduces the boundary-replay trajectory
for all six selected targets and `9997/10000` joint winners.  The Slurm array
is recorded as `FAILED/1` only because its post-run assertion requested local
fused-posterior dumps during a global sparse pass-2, where that hook is not
active; all science artifacts are complete.  This rules out iteration-4
sampling, scoring, and BPref as the first boundary and moves the trace to
iteration-3 low-shell PPref formation.  Evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it3_it4_ppref_shell_cross_20260712_160000/ANALYSIS_SUMMARY.txt`.

Clean 100k hybrid scale job `11058928` completes 16 numbered iterations plus
final in 2:35:23 without OOM, but its final pass incorrectly repeats numbered
parent/fine orders 6/7.  RELION's unnumbered `run_sampling.star` advances the
final parent to 7, so adaptive oversampling requires fine order 8.  Commit
`892c85e0` makes final sampling metadata authoritative while preserving the
state-order fallback.  Final-only job `11074230` and combined seed-exact retry
`11083758` validate parent/fine 7/8.  The final map passes strongly:
RECOVAR-vs-RELION FSC-AUC is `0.996184`, and RECOVAR-vs-GT is `0.497383`
versus RELION `0.490627`.  Pmax improves materially but remains non-parity:
RECOVAR mean `0.099121379` versus RELION `0.118882262`, correlation `0.9158`.
Seed-exact perturbation changes the mean only `1.1e-6`; 80,513 same-winner
particles retain a `-0.01956` mean gap.  This localizes the residual to
posterior denominator/support geometry, not sampling order, perturbation,
winner pose, OOM, or final-map quality.  Evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_100k_final_only_seedexact_hp8_combined_retry_20260712_150000/PMax_RESIDUAL_AUDIT.md`.

Next gates: validate the three source-derived current-size corrections in a
fresh four-iteration trajectory before another full 10k trajectory.  The
identical-input reconstruction gate is already closed; trajectory FSC/FSC-AUC
and particle-state changes decide whether these corrections are retained and
whether the smaller BPref numerator residual becomes the next boundary.
Attempt `11058781` was rejected before scoring because its CUDA output path
was shared.  The cold RELION dump `11084550` and stock continuation
`11085341` are also rejected oracles because their trajectories diverge from
the installed authoritative iteration 4.

The continuation-oracle path is closed rather than weakened.  Final
stored-accuracy job `11086683` exactly preserves the installed iteration-4
sampling order and full perturbation and retains the serialized iteration-3
accuracy (`2.006` degrees, `1.498312` Angstrom), yet still has half-map
FSC-AUC `0.9999892/0.9988276`, 20 angular differences above one degree, 581
translation differences, and 241 significant-sample-count mismatches.  Its
90 score dumps fail the unchanged gate and remain quarantined.  Therefore
RELION's serialized optimiser/data/model/sampling STARs omit process-history
state needed for an exact continuation oracle; no further ad hoc continuation
overrides are warranted.  Gate:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it4_authoritative_preserve_accuracy6_20260712_231500/authoritative_it4_parity_gate.json`.

The first strict 100k/256 completion attempt `11036541` reaches numbered
iteration 12, then fails in the local parent score-only big-JIT.  The failing
shape is 168 images by 198 rotations by 9 translations by 12,861 score pixels.
Its dense float32 residual is 14.34 GiB before compiler overhead and XLA asks
for 16.34 GiB.  The previous microbatch planner bounded only
`images * rotations`, so its automatic image-batch boost admitted this tensor
without charging translations or score pixels.  Commit `48e712f1` adds a
score-only runtime cap based on free allocator bytes, translations, score
pixels, float32 width, and measured 1.25x live-tensor headroom.  Explicit
overrides remain authoritative and M-step batching is unchanged.

Same-H100 diagnostic `11047386` jumps directly from RELION iteration 11 to the
exact failed iteration-12 state.  The cap selects 69 images per 198-rotation
parent bucket (`13,669` rows), processes all 49,913 half-set particles in
27.7 seconds, then completes the fine score-only pass in 26.0 seconds; the job
finishes successfully in 1:56 with no OOM.  Full clean 100k trajectory job
`11047558` was cancelled as scientifically stale after the missing run_it000
cold state was identified.  Corrected job `11049164` was relaunched from
clean commit `2e3cc620`, includes `--relion_init_dir`, and retains the validated
runtime score-tile cap.  Its artifact root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_100k_coldstart_scorecap_ready_20260712_051319`.

Job `11049164` was subsequently cancelled at 41:31 after the same-A100
iteration-3 projector A/B proved that its texture-on trajectory was
scientifically stale.  Do not restart 100k until the manual-projector full real
trajectory closes or identifies the next boundary.

K=4 also localizes the texture regression at its first global Class3D step.
Texture-on job `11049571` gives only 45.24% class agreement and per-class
FSC-AUC `0.3520-0.4439`.  Texture-off job `11049966` restores 99.82% class
agreement, pose p95 `1.42e-4` degrees, translation p95 `5.68e-6` pixels, and
exact Pmax; per-class map FSC-AUC improves to `0.9927-0.9967`, so its remaining
map/reconstruction tail is still open.  Class3D retains a fresh global first
search rather than replaying run_it000 input orientations (`2e0e25ab`); those
orientations are input metadata, not AutoRefine-style local-search centers.

In parallel only when authorized: audit K=4 per-iteration dumps to identify the
first class/pose/state divergence; do not optimize sparse pass 2 until that
quality boundary is known.

## Decision Log

Resolved with the user on 2026-07-11:

- RELION GUI semantics are the default until strict parity closes. Major
  behaviors remain typed, switchable options so later grid-correction,
  angle-refinement, and other scientific ablations do not require rewrites.
- Full K=1 trajectory parity closes before K=4; supplied-map EM closes before
  native InitialModel/VDAM.
- Discrete comparisons are tie-aware. A winner flip is acceptable only when
  underlying scores/posteriors prove a numerical near-tie. Convergence and
  finalization are expected to match exactly.
- K=1 closure spans multiple seeds, white/colored noise, CTF/no-CTF,
  uniform/preferred angular distributions, contrast/noise-scale variation,
  translation stress, and junk/outliers, followed by a well-characterized
  real-particle confirmation and a 100k/256 completion run.
- At scale, compare complete aggregate iteration state plus stratified score
  surfaces, automatically dumping every mismatch for full investigation.
- RECOVAR and RELION timing pairs use the same GPU model. Any available cluster
  GPU class is valid. Up to four local GPUs may be used for short checks, but
  only after confirming each selected device is idle.
- Up to three subagents may perform independent bounded investigations, with
  one writer per source area and the primary agent owning integration.
- Preserve the existing dirty candidate and create a separate clean local
  checkpoint/logical commit history. Do not push without separate approval.
- Long-lived EM development checkouts use
  `/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/`; bulky run outputs and runtime
  caches remain under `/scratch/gpfs/CRYOEM/gilleslab/em_work/`.

Record each resolved decision here with date, rationale, and effect on gates.

## Experiment Record Template

```text
Date / hypothesis:
Mode: strict | quality | performance
Commit / branch / dirty SHA-256 / untracked manifest:
RELION commit/build / command / MPI / GPU:
Fixture / seed / particle count / box / K:
RECOVAR command and environment overrides:
Slurm jobs / node / logs / artifact root / SAFE_TO_DELETE:
First divergence boundary:
Quality metrics and deltas:
Performance metrics and deltas:
Result: supported | falsified | inconclusive
Regression added:
Next cheapest discriminating experiment:
```

## Queue Discipline

Run multiple independent, decision-bearing diagnostics in parallel when that
reduces parity-debug latency; large Slurm queues are acceptable. Keep only one
writer per source area, use matched GPU models for timing A/Bs, and cancel jobs
as soon as their premise becomes stale. A run without a predeclared decision
it can change should not be submitted. Negative and rejected results must be
recorded so future agents do not repeat or accidentally cite them.

## 2026-07-12 Iteration-2 BPref Arithmetic Trace

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L2761). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-13 Matched BPref Operand And First-Boundary Trace

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L2830). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-13 Boundary-Enriched Production Scatter Trace

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L3034). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-13 autonomous K=1 native normalization result

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L3139). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-14 Case-22 TF32 Translation-Phase Root Cause

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L3150). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-14 Case-16 retained-posterior norm denominator

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L3197). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-14 Case-22 firstiter-CC reduction order

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L3251). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-14 Case-16 final SamplingPerturbation order

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L3378). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-14 Case-13 final per-half noise bug

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L3405). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-14 Clean-head case-20 scale result

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L3493). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-14 Explicit RELION CUDA preprocessing runtime integration

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L3571). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-14 Case-26 native WAVG/atomic discriminator rejected

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L3597). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-14 Case-26 paired raw operands close the accumulation branch

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L3625). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-14 Explicit RELION CUDA full-trajectory gate

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L3662). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-14 Serialized-sigma discriminator proves the case-26 boundary

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L3690). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-15 K=1 real-particle full trajectory and exact-reference boundary

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L3718). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-15 K=4 iteration-3 cliff and rejected causes

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L3768). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-15 Targeted posterior discriminators

The K=1 iteration-12 fused-posterior implementation is not the local-search
cause.  H100 jobs `11201155` and `11201156` independently capture the fused
and forced-materialized fine paths with exact incoming RELION references.  The
captures pass their instrumentation gates: merged map FSC-AUC is
`0.999999994760/0.999999994665`, p05 non-DC FSC is
`0.999999962714/0.999999961869`, and all Pmax differences from the undumped
exact-reference control are zero.  For their four shared target particles,
candidate rotations, translations, parent/child identities, masks, support,
scores, log normalizers, posteriors, Pmax, and winners are bitwise identical.

The remaining K=1 difference is structural relative to RELION, not a close
tie.  For fixture index 6536 (STAR 85521), RELION's winner is absent from
RECOVAR's finite fine support and the closest same-rotation RECOVAR candidate
is separated by score `7.659`.  For fixture index 4194 (STAR 54772), RELION's
translation is RECOVAR rank 4 with score gap `0.926178`; RECOVAR's close top
pair does not contain the RELION winner.  Fixture indices 8421 and 9640
choose the exact same winner in both programs, but RELION/RECOVAR Pmax are
`0.364345/0.982648` and `0.380272/0.998304`, with RECOVAR top-two gaps
`4.71278` and `6.64923`.

The parent-to-fine expansion is also exact and is no longer a candidate cause.
For all four targets, the finite fine mask is the exact 32-child expansion of
the significant parent cells, with no candidate-ID, rotation, translation,
parent-child, or mask mismatch.  The divergence is already present in the
parent or fine score surface relative to RELION.  For fixture 6536, RELION's
winner belongs to parent rotation 193770 / translation 13, which RECOVAR
scores but prunes at the parent boundary: it is parent rank 2 with posterior
`0.00075794` and score gap `7.18414`, while translation 14 alone is retained.
For fixture 4194 the RELION parent is retained, but its fine translation falls
to RECOVAR rank 4 with posterior `0.118761` and score gap `0.926178`.
Fixtures 8421 and 9640 retain the same final winner but have substantially
over-concentrated RECOVAR posteriors.  These are structural score/posterior
differences, not discrete tie-breaking or fused-kernel behavior.  Instrumented
RELION iteration-12 candidate captures localize them to parent scoring size as
described below.

K=1 capture evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it12_targeted_capture_bf49f93f_20260715_012401/fused_vs_fine_shared_comparison.json`.
Parent-expansion evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it12_targeted_capture_bf49f93f_20260715_012401/parent_to_fine_support_comparison.json`.

The instrumented RELION target captures further localize this to fine
hypothesis/support construction, before priors or posterior normalization.
The capture binary is not globally instrumentation-inert, so these arrays are
used only after target-level qualification against the forced-perturbation
no-dump control.  Targets 85521, 54772, and 126792 pass that target gate; target
110844 differs only in Pmax by `1e-5` with identical winner, pose, shift, and
significant count and remains explicitly marked failed-closed.  Orientation
and offset log-priors agree within `1.43e-6` and `4.77e-7`.  Conditional on the
common finite support, posterior total-variation distance is only
`2.72e-5`, `2.50e-7`, `1.87e-4`, and `1.17e-6`.  The support itself is not the
same: RELION/common/RECOVAR candidate counts are `128/32/32`, `128/64/160`,
`384/160/192`, and `128/32/32`; RELION assigns only `0.2260`, `0.99998`,
`0.36875`, and `0.38080` probability to the common support.  The large Pmax
differences are therefore caused by absent/excluded hypotheses, not by prior
or normalization arithmetic on a shared hypothesis set.

The parent-support cause is the ordering of local angular refinement and
Fourier-size selection.  RELION iteration 12 enters `expectation()` with
sampling order 3, computes its pass-1 parent image size from the old 7.5-degree
sampling (`56` pixels), and only then updates the sampling order to 4 for the
current local parent grid and order-5 fine children.  RECOVAR updated the order
first and recomputed the parent image size from 3.75 degrees, scoring at `110`
pixels.  RELION consequently selects `4/4/12/4` parents for the four targets,
while RECOVAR selects `1/5/6/1`; both expand every selected parent into exactly
8 rotations by 4 translations.  Aligned parent scores across the wrong
56-versus-110 Fourier bands have post-common-shift p95 residuals of
approximately `27.19/26.71/8.85/16.47`, while inferred combined-prior
residuals remain below `6.87e-5`; this is not a tie or prior effect.

RELION/RECOVAR hypothesis-alignment evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it12_relion_target_capture_20260715_022636/analysis/relion_recovar_posterior_alignment.json`.
Parent support-rule audit:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it12_relion_target_capture_20260715_022636/analysis/parent_support_rule_audit.json`.

The K=4 iteration-2 outliers are instead inherited amplification from tiny
reference drift.  With exact standard RELION iteration-1 references, corrected
target-qualified RELION captures and RECOVAR have candidate-support Jaccard
`1.0` for every class of original particles 2907 and 8083: 3,488/3,488 and
3,168/3,168 total candidates, including identical reconstruction support and
all eight classwise top keys.  Combined-prior error is at most `9.54e-7`,
centered score-with-prior p95 is `7.34e-5--1.15e-4` (worst maximum
`0.001005`), and posterior L1 after common renormalization is
`7.51e-6--1.90e-5`.  No score, prior, support, or posterior behavior mismatch
exists at this matched iteration-2 boundary.  The intrinsic K=4 investigation
therefore remains at iteration 3, where exact iteration-2 reference replay did
not close the trajectory cliff.

K=4 score/support evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_orig2907_8083_recovar_exactref_pass2_h100_20260715_020900/analysis/score_support_no_correlation.json`.

The K=4 iteration-3 raw-score cliff is caused by a group-scale state mismatch.
For original particle 6388, RELION scores with runtime scale `0.972485065`
while its rank-1 post-M-step model STAR serializes approximately `1.315036`;
RECOVAR's exact replay used the serialized value.  A clean H100 one-factor A/B
changes only that particle's scoring scale.  The serialized-scale arm retains
the wrong class-3 branch, class masses approximately
`[0, 0.0001786, 0.9998214, 0]`, support Jaccard
`0.351/0.357/0.446`, and centered score-with-prior mean/p95/max error
`16.655/31.314/43.710`.  The runtime-scale arm restores support Jaccard `1.0`
and every classwise top key; RECOVAR class masses become
`[0, 0.0166850, 0.3179502, 0.6653648]` versus RELION
`[0, 0.0166907, 0.3178691, 0.6654402]`, with the exact class-4 winner.
Centered score-with-prior mean/p95/max error falls to
`0.001392/0.003004/0.005733`.  This is causal behavioral evidence, not a
numerical tie or downstream support defect.

A per-rank iteration-2 state dump identifies the underlying RELION behavior.
The piecewise `MlWsumModel::pack` path sizes the group-scale XA/AA payload from
the one optics group instead of the 10,000 particle groups.  Only group 0 is
MPI-combined; among groups 1--9999, 5,027 have rank-1-only statistics and 4,972
have rank-2-only statistics, with no overlap or both-zero group.  For target
group 5989, rank 1 has raw XA/AA scale `1.348988547` and normalizes it to
`1.314142312`, while rank 2 has zero AA, substitutes the default scale 1, and
normalizes it to `0.973957723`.  The writer model matches rank 1 within
`5e-7`; the particle's next E-step can use rank 2's live state instead.  Strict
n=3 parity therefore requires follower-local scale vectors and exact
iteration-to-iteration particle ownership; a single global or rank-1
serialized scale vector cannot reproduce RELION.

K=4 scale A/B evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it3_orig6388_runtime_scale_ab_h100_20260715_024500/runtime_scale/analysis/score_support_no_correlation.json`.
K=4 per-rank scale-state evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_relion_scale_state_rank_audit_h100_20260715_034500/analysis/scale_rank_state_no_correlation.json`.

## 2026-07-15 local-size and continuation-noise fixes

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L3957). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-15 dynamic MPI dispatch correction

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L3986). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-15 Scoped K=1 BPref capture and K=4 particle-3591 boundary

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L4481). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-15 K=4 projector-fix three-iteration trajectory

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L4582). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-15 K=4 score-capture quarantine and projection-cache discriminator

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L4631). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-15 Projection-cache exoneration and K=1 full trajectory

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L4676). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-15 Corrected K=1 converged-state replay

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L4775). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-15 Same-GPU RELION trajectory repeat envelope

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L4806). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-15 K=1 iteration-2 exact-reference boundary

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L4837). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-15 K=4 particle-5993 frozen-boundary provenance correction

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L4867). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-15 K=4 frozen cutoff precision classification

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L4890). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-15 K=1 current-engine coarse cutoff capture

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L4913). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-15 K=1 iteration-2 passive cross-engine closure

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L4935). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-15 K=1 trajectory repeat-envelope classification

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L4999). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-16 K=1 exact-rotation trajectory and next boundary

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L5022). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-16 K=1 recurrent score and BPref boundary classification

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L5054). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-16 K=1 eight-case robustness trajectory gate

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L5130). Historical next actions are superseded by the current EM status and coordination queue.

## Aggregate particle-state diagnostic

Use `scripts/audit_em_particle_state_distribution.py` after full runs to align
particles by exact `rlnImageName` and compare Pmax, significant-support, pose,
translation, and K-class distributions across every available numbered
iteration.  When a same-physical-GPU RELION repeat is supplied, the report also
measures RECOVAR errors relative to that numerical control envelope.  An
independent repeat pair can be supplied with `--relion-control-reference-star`
and `--relion-control-star`, so the control envelope need not share the
cross-engine reference arm.  K-class agreement is Hungarian-matched once per
iteration and that fixed mapping is used for every subgroup, while retaining
raw label agreement and the full
confusion matrix.  Intermediate gates use these exact/array metrics; map-quality
gates remain FSC/FSC-AUC only.  Escalate from this aggregate report to a
particle capture only when it identifies a systematic cohort or an FSC
trajectory localizes a reproducible boundary.  Use repeated
`--recovar-iteration` arguments when only an explicit boundary subset has all
required state arrays; the default remains fail-closed rather than silently
omitting missing support.  Without that explicit selection, a RELION particle
STAR is required for every numbered RECOVAR iteration; omitted middle or
trailing iterations are errors.

The same auditor reports numbered current-size, resolution, HEALPix, Pmax,
expected-accuracy, and assignment-change scalars plus convergence and final
all-data topology. RECOVAR `pixel_resolutions` are converted from shell index
to Angstrom and compared with RELION `model_classes.rlnEstimatedResolution`;
`model_general.rlnCurrentResolution` is reported separately as the inherited
scheduling resolution. RELION's converged unnumbered optimiser value
`rlnCurrentIteration=-1` resolves to the highest preceding numbered state.
Final Pmax/pose/translation arrays are compared when present, while unavailable
final support or class arrays remain explicitly not measured. The CLI writes a
compact aligned-array NPZ and SHA-256 manifest beside the JSON. It stays
diagnostic by default; thresholds and exact schedule/convergence gates apply
only when explicitly requested. Correlation is neither computed nor used.

## 2026-07-16 K=4 strict Class3D trajectory closure

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L5192). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-16 K=1 exact-BPref robustness closure and performance boundary

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L5226). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-16 real-data repeat-control adjudication

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L5273). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-16 authoritative K=4 incoming-reference substitutions

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L5390). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-16 K=4 stock-RELION repeat calibration

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L5423). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-16 K=1 local significant-count semantics

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L5456). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-16 K=4 heterogeneous robustness expansion

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L5497). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-16 K=4 significant-count tie metadata

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L5591). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-16 K=4 case-11 firstiter winner boundary

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L5602). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-16 real-10076 iteration-2 ordinary BPref classification

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L5698). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-16 K=1 bounded raw-diff2 reuse closure

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L5725). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-16 K=4 particle-7916 precision classification

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L5759). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-16 real-10076 aggregate pre-scatter classification

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L5782). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-16 real-10076 aggregate pre-scatter substitution

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L5818). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-16 real-10076 stack-111721 coarse boundary audit

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L5843). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-16 real-10076 BPref factor and GEMM-precision closure

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L5868). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-16 K=1 100k scale and convergence qualification

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L5930). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-17 sealed native restart at the 100k expected-accuracy boundary

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L5988). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-17 real-10076 hidden-change tail and schedule intervention

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L6023). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-17 complete iteration-1 contribution replay

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L6098). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-17 same-A100 live real-10076 trajectory

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L6130). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-17 real-10076 iteration-2-to-3 aggregate score boundary

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L6162). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-17 real-10076 fixed-UID score reduction classification

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L6203). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-17 K=4 100k/256 memory-cap acceptance

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L6239). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-17 host-matrix incoming-boundary and tail-enrichment diagnostics

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L6262). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-17 native texture-context closure on an uninterrupted boundary

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L6324). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-17 sealed Iref-to-projector replay

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L6372). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-17 full-10k live-boundary closure

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L6407). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-18 frozen-boundary and robustness launch blocks

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L6528). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-18 frozen-boundary v2 and robustness pre-acceptance findings

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L6574). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-18 frozen projector causal result and corrected robustness launch

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L6645). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-18 K=4 mixed reduction-mode oracle invalidation

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L6695). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-18 frozen projector source/construction closure

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L6742). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-18 robustness case-11 acceptance-contract failure

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L6782). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-18 physical iteration-2 native-BPref factorial

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L6859). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-18 K=4 GUI first-iteration resolution-state repair

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L6915). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-18 split-half optimizer-Pmax workflow correction

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L7029). Historical next actions are superseded by the current EM status and coordination queue.

## Fixed-arm frozen-boundary diagnostic

The reusable fail-closed fixed real-10076 K=1 physical-it2 diagnostic arm,
deterministic finalizer, source ownership, and captured-Iref lineage contract are documented in
[`frozen_boundary_v3.md`](frozen_boundary_v3.md). Schema v2 remains historical
and cannot support the fixed-arm claim. Schema v3 seals an explicitly
enumerated reconstructed-projector diagnostic state; it does not claim identity
to RELION's full in-memory physical iteration.

## 2026-07-19 reusable exact-local BPref accumulator replay

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L7060). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-19 K=1 iteration-5 resident-state causal localization

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L7086). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-19 K=1 iteration-4 incoming-map amplification

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L7123). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-19 case-8 low-memory full-trajectory qualification

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L7223). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-19 capped convergence and autonomous termination classification

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L7252). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-19 recurrent K=1 final-boundary family localization

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L7322). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-19 same-GPU K=4 production-versus-float64 trajectory

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L7374). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-19 K=1 native RELION pre-scatter operand localization

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L7408). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-19 case-25 incoming-reference null interventions

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L7450). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-19 case-20 aggregate operand-factor localization

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L7485). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-19 case-25 accumulated non-reference state factorial

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L7522). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-19 factor-capture postflight contract repair

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L7555). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-19 case-20 variable-support factor-panel closure

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L7577). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-19 case-20 exact-state iteration-2/3 M-step closure

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L7611). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-20 case-20 physical-iteration-3 resident-state factorial

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L7639). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-20 case-20 remaining-state split audit correction

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L7672). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-20 case-20 physical-iteration-3 exact-state posterior panel

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L7703). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-20 case-20 exact-state coarse-parent cutoff localization

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L7739). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-20 case-20 native RELION coarse-score operand closure

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L7784). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-20 case-10 x-half acceptance OOM classification

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L7846). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-20 K=4 continuation rejection and live-factor replacement

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L7870). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-20 adaptive pass-1 CUDA scorer-matrix closure

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L7959). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-20 K=1 300k-particle case-3 acceptance

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L8070). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-20 current-head K=4 three-iteration replay

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L8082). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-20 current-head autonomous case-20 closure

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L8130). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-20 current-head autonomous case-26 replay

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L8173). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-20 inclusive current-size boundary correction

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L8215). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-20 active full case-33 and g384 acceptance chains

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L8317). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-20 late acceptance checkpoints and case-7 retry

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L8451). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-21 old-head full-34 durable negative closure

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L8526). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-21 case-33 iteration-6 FSC checkpoint

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L8560). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-21 case-9 low-cap arm crosses the old OOM boundary

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L8582). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-21 case-10 terminal half-1 memory checkpoint

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L8611). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-21 case-9 iteration-11 FSC checkpoint

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L8632). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-21 case-9 cap decomposition and case-33 seal repair

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L8658). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-21 case-33 iteration-11 FSC checkpoint

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L8686). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-21 case-9 terminal half-1 memory checkpoint

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L8707). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-21 case-9 science job completes with favorable terminal FSC-AUC

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L8730). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-21 case-33 complete numbered trajectory

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L8763). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-21 case-33 terminal rejection and state localization

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L8798). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-21 case-10 low-cap completion rejects terminal parity

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L8842). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-21 local mirrors freeze case-9 acceptance and case-10 final-only rejection

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L8882). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-21 case-7 panel confirms upstream state/reference locus

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L8926). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-21 case-10 final-transition FSC decomposition

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L9214). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-21 active case-7 state-component discriminator

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L9249). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-21 case-7 exact control is reproducible across H100 allocations

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L9275). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-21 historical K=1 v2 matrix sealed fail-closed

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L9302). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-21 strict K=1 v3 audit graph repaired fail-closed

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L9329). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-21 strict K=1 v3 matrix is structurally sealed and parity-failing

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L9365). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-21 case-7 capture-target observer effect and clean-rerun gate

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L9399). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-21 clean case-7 component and population results

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L9438). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-21 K=1 residual-panel live trajectory checkpoint

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L9481). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-21 K=1 clean residual panel reaches terminal classification

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L9518). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-21 current-head case-2 strict-boundary closure

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L9576). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-21 significance-capture boundary gate

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L9599). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-21 uninterrupted K=4 class-2 pre-scatter diagnostic

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L9659). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-30 case-22 corr_img inverse-noise factorial

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L10106). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-25 live K=4 preprocessing discriminator

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L10442). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-25 seed-exact K=4 factor-panel localization

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L10995). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-26 full K=4 backend trajectory accepts `relion_cuda`

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L11242). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-26 canonical case 3 advances K=1 to 26/34

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L11290). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-26 exact case-5 discrepancy classifier queued

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L11326). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-26 current-head fixed case 32 submitted

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L11351). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-26 fixed-metric and current-head verification checkpoint

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L11401). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-26 K=4 backend particle-state classification

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L11434). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-26 case 32 completes and localizes the final miss to pose state

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L11479). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-26 case 32 firstiter coarse boundary closes causally

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L11548). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-26 case 32 passes autonomously and advances K=1 to 27/34

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L11598). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-26 K=4 fine-score diagnostics gain a fixed repeatability floor

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L11651). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-26 exact-H100 K=1 capture rejects on its target translation

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L11783). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-27 K=4 numerator boundary: data score, not `expf` or priors

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L11846). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-27 production RELION translation-score FFI closes sealed targets

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L11904). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-27 exact translation scoring advances fixed K=1 to 28/34

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L11955). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-27 exact translation scoring extended through local search

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L11998). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-27 case-4 source effect sealed; local qualification queued

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L12045). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-27 exact-device trajectories pass their first affected boundaries

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L12222). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-27 shared outlier gate isolated from fixed-noise generation

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L12262). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-27 exact-device audit and shared shell-0 diagnostic

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L12290). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-27 sealed case-4 final particle-state transition

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L12365). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-28 one-GPU outlier repeatability and round-2 junk locus

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L12406). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-28 exact-A100 K=4 live checkpoint through iteration 11

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L12496). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-28 current-head fixed case-22 qualification

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L12608). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-28 frozen case-22 current-versus-b1d source audit

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L12654). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-12 case-22 direction-mass causal propagation

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L12693). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-28 native RELION C++ FSC rules out scheduler emulation

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L12706). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-28 joined BackProjector localizes case 22 descriptively

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L12747). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-28 case-22 pre-scatter discriminator submitted

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L12792). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-28 authoritative exact-A100 K=4 audit is unchanged

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L12830). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-28 exact-device native repeat accepts the BPref residual

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L12869). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-28 case-22 pre-scatter and full-trajectory closure

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L12913). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-28 same-A100 FSC establishes the physical-iteration-2 boundary

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L12956). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-28 bounded iteration-2 cohort replaces infeasible full capture

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L13005). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-28 RELION threshold substitution does not close case-22 membership

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L13058). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-28 bounded pre-scatter capture and iteration-3 coverage audit

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L13108). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-28 complete iteration-3 panel localizes candidate-grid and significance gaps

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L13163). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-28 iteration-3 one-particle candidate/significance discriminator

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L13227). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-29 iteration-3 state swap makes incoming maps causal

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L13258). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-29 corrected K=4 fixed-state target exposes a score tie

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L13308). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-29 case-22 shellwise map amplitude transfers coarse support

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L13440). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-29 unified fixed K=1/K=4 scorecard

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L13493). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-29 K=1 map-amplitude trajectory localization

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L13519). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-29 reference-2 tau2 substitution moves case 22 upstream

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L13560). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-29 case-22 coarse membership boundary is raw-score sensitive

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L13601). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-29 case-22 captured norm/cross operand gate

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L13698). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-30 case-22 preprocessing-boundary qualification

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L13864). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-30 case-22 post-optics score-transfer boundary

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L13957). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-30 case-22 pixel-correction / corr_img factorial

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L14003). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-30 case-22 corr_img conditioning audit

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L14045). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-30 case-22 inverse-noise shell partition

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L14075). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-30 case-22 serialized-restart score and map gates

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L14121). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-30 case-26 double cross-half M-step diagnostic

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L14172). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-30 predeclared matched-head case-26 precision factorial

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L14222). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-30 K=4 exact-device dependency retry

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L14317). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-30 active K=1/K=4 causal gates

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L14384). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-31 case-22 same-A100 serialized-restart result

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L15304). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-31 K=1 continuation resolution-initializer discriminator

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L15367). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-07-31 K=1 continuation sampling-perturbation boundary

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L15441). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-01 K=1 live binary64-noise counterfactual

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L15608). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-01 K=4 deterministic contribution-repeatability candidate

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L15654). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-01 K=1 exact initial-noise counterfactual rejection

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L15699). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-01 K=4 deterministic soft-mask full quality acceptance

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L15756). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-02 K=1 restart particle-order causal closure

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L15840). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-04 exact-device K=4 contribution retry interface correction

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L15890). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-04 K=4 device-signature fused-mode gate correction

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L15920). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-04 corrected K=4 downstream source binding

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L15944). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-04 K=4 fine-score FMA boundary closed locally

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L15973). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-09 K=1 first-divergence program supersedes broad hypotheses

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L16027). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-09 case-4 physical-iteration-2 coarse boundary localized

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L16085). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-09 case-10 exact boundary rejects a common radial-scorer fix

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L16126). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-10 case-26 RELION-CUDA radial scorer is negative

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L16162). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-10 case-10 coarse-parent support localization

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L16194). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-10 case-10 translation lattice boundary candidate

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L16246). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-10 case-22 H100 code-generation target is falsified

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L16271). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-10 case-10 executable translation-grid equivalence

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L16300). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-10 all case-10 panel translations require the outer grid

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L16318). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-11 case-26 iteration-1 BPref atomic boundary closure

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L16337). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-11 case-26 iteration-2 fine-score pixel-weight boundary

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L16383). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-12 case-22 first normalization boundary and next gate

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L16508). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-12 case-22 normalization propagation and direction-prior root

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L16534). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-12 case-22 iteration-3 parent flip localized to pixel-weight state

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L16559). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-13 case-22 direct Wavg noise factorial

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L16679). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-13 masked-Wavg first-divergence closure

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L16902). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-14 case-7 iteration-1 raw BPref localization

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L16966). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-15 case-7 exact-perturbation population and first score boundary

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L17103). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-15 case-7 live coarse translation and lane boundary

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L17320). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-15 case-7 live coarse operand and source-map closure

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L17970). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-15 case-7 iteration-1 pre-scatter population closure

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L18064). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-15 case-7 native-texture coarse boundary

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L18125). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-15 RELION coarse launch-scope population and trajectory gates

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L18175). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-20 case-7 coarse-CC source-expression closure

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L18520). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-20 fresh-K=1 physical order and bounded mixed-support propagation

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L18569). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-21 case-4 first cutoff error is one coarse raw score

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L18748). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-22 02:10 EDT — native full-coarse boundary and case-10 optics remap

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L19630). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-22 02:50 EDT — treatment prefix result and live operand capture

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L19750). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-22 03:30 EDT — case-10 result and normalization localization

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L19852). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-22 04:15 EDT — float64 state falsified; exact norm summands queued

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L19994). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-22 05:00 EDT — target norm closes; half-average boundary isolated

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L20075). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-22 06:00 EDT — continuation quantization invalidates the apparent factor root

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L20222). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-22 06:40 EDT — fresh normalization closes; row-68694 localizes to the map

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L20305). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-22 06:45 EDT — row-68694 first divergence is the iteration-1 reference

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L20425). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-22 07:20 EDT — particle-pool reduction is not the remaining K=1 fix

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L20514). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-22 09:00 EDT — unrounded live initial noise closes the first K=1 BPref boundary

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L20562). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-22 09:35 EDT — guarded promotion and case-10 memory gate

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L20650). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-22 09:55 EDT — exact compact global projection gate

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L20694). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-22 10:20 EDT — case-5 live-noise rejection and stopped state discriminator

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L20730). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-22 10:40 EDT — live-noise falsification and case-5 local fine-path localization

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L20783). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-22 13:35 EDT — case-5 live lane closes reduction ambiguity

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L21159). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-22 14:15 EDT — exact case-5 PPref gate launched

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L21213). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-22 15:10 EDT — case-5 root moves upstream to the iteration-1 map

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L21260). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-22 16:20 EDT — first unequal iteration-1 boundary is half-1 BPref

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L21320). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-22 16:55 EDT — order is exact; raw-BPref localization is rejected

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L21399). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-22 18:35 EDT — native score confirms a real split; production adapter omitted exact fine CC

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L21452). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-22 18:50 EDT — exact corr_img fix closes the complete iteration-1 map boundary

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L21494). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-22 19:40 EDT — autonomous case-10 iteration-1 FSC gate passes

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L21599). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-22 19:55 EDT — case-5 iteration-2 controller boundary remains exact

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L21636). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-22 20:00 EDT — case-10 iteration-2 controller boundary remains exact

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L21650). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-22 20:10 EDT — exact corr_img does not close the case-10 iteration-2 residual

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L21663). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-22 20:44 EDT — case-10's first row-12334 mismatch is one omitted coarse parent

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L21746). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-22 20:55 EDT — native case-10 coarse boundary captured

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L21788). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-22 21:24 EDT — production coarse capture falsifies the omitted-parent diagnosis

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L21826). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-22 21:48 EDT — production fine route is topology- and support-exact

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L21876). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-22 22:06 EDT — case-10 remains green through physical iteration 4

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L21909). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-22 22:38 EDT — case-10 remains green through iteration 5; targeted suite passes

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L21949). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-23 00:25 EDT — native-unit corr_img cast boundary is exact but secondary

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L21977). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-23 02:05 EDT — fresh fine boundary separates trajectory state from scorer arithmetic

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L22024). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-23 02:18 EDT — case 10 is now localized to the final-only boundary

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L22076). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-23 02:48 EDT — case-10 last-numbered particle state exposes a five-percent pose tail

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L22122). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-23 04:18 EDT — iteration-2 XA/AA continuation is a falsified production boundary

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L22183). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-23 04:28 EDT — exact iteration-15 state clears the case-10 final path

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L22243). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-23 04:46 EDT — case-10 pose-tail onset follows angular-grid refinement

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L22296). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-23 05:55 EDT — fresh native XA/AA and coarse support localize the first discrete mismatch

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L22335). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-23 06:35 EDT — the first material case-10 boundary is the fresh iteration-1 map

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L22426). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-23 07:28 EDT — five iteration-1 winners explain 94.6% of the case-10 projector gap

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L22481). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-23 08:05 EDT — projected-reference split closes, and the replay is reclassified

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L22567). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-23 08:58 EDT — exact model sampling closes active PPref, exposing one score tie

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L22625). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-23 10:16 EDT — first production defect is a missing projector boundary row

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L22687). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-23 10:42 EDT — crop correction closes all 100,000 case-10 iteration-1 assignments

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L22751). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-23 11:08 EDT — checkpoint pushed and fixed-prefix validation active

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L22808). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-23 11:26 EDT — all three production iteration-1 map gates pass

[Preserved experiment record](https://github.com/ma-gilles/recovar/blob/00f319ca0e74816f3a4edb58d126bd14ede7e8ae/docs/math/em_parity_program.md#L22857). Historical next actions are superseded by the current EM status and coordination queue.

## 2026-08-26/27 double-precision float32/complex64-forcing audit (`double_parity` branch)

Separate track from the K=1 case-22 chain above: a user-directed audit of
`recovar/em/` for places that hardcode `float32`/`complex64` regardless of
`RECOVAR_USE_FLOAT64_SCORING`/`RECOVAR_USE_FLOAT64_PROJECTIONS`, cross-checked
against RELION's own `ACC_DOUBLE_PRECISION` C++/CUDA source (newly available
this session at `/gpfs/gibbs/project/lederman/ry295/relion`, branch
`recovar_em_patch`). Full detail, RELION ground-truth findings, and a
prioritized backlog of ~120 further sites are in
`docs/math/relion_parity_agent_notes.md`'s 2026-08-26 "(continued 3)" entry
-- summary here per this file's own update contract:

- **Fixed and CPU-tested** (9 source files): cross-iteration pose/translation
  state recurrence (the highest-impact category -- confirmed via RELION's
  `exp_metadata`/`EMDL_DOUBLE`, never narrowed, but recovar was flooring it to
  float32 every iteration), direction/translation-prior upstream narrowing,
  `DenseScoreConstraints`, the local-search hot-path bucket-packing function
  (`local_layout.py:bucket_local_hypothesis_layout`, was defeating all
  upstream local-search double-precision threading), and
  `mean_helpers.py:update_relion_norm_scale_corrections`.
- **Not yet run**: the GPU fast parity tier
  (`pixi run test-em-parity-fast`) and any FSC/quality gate against this
  fix batch -- this is the mandatory next step before claiming the fixes are
  quality-neutral-or-positive, and before any further fixing in this track.
  Per the validation ladder, `pixi run test-em-fast-guard` (16/16) and four
  directly-relevant CPU unit-test files were run instead as the cheap rung;
  GPU parity was explicitly deferred per the user's "prioritize + backlog"
  scope decision for this large a sweep.
- **Backlog** (not yet fixed, prioritized P0/P1/P2 with file:line citations
  in the notes entry): `local_em_engine.py` (largest remaining item, dozens
  of sites in the default exact-local engine), the `sparse_pass2_bucketed.py`
  "RELION-GPU exact diff2 (Gaussian)" scoring family (same bug class as the
  already-fixed CC scorer, commit `8fc84499`, but currently only reachable
  when `use_float64_scoring=False` so not itself corrupting double-precision
  runs today -- open policy question: extend it or accept the algebraic path
  as the double-precision default), the sparse pass-2 translation/prior
  chain, remaining `local_layout.py` functions, `preprocessing.py`,
  `image_shifts.py`, `projection.py`, `k_class.py:_assemble_result`,
  `significance.py` output buffers, and `iteration_loop.py`'s
  `current_translations` per-iteration scoring grid (flagged high-impact but
  needs adjudication first -- a comment in that code offers a possible
  intentional rationale for the asymmetry with `current_rotations`).
- **Open question, not resolved**: an existing docstring in
  `relion_metadata.py:_relion_rotation_grid_float32` claims RELION Euler
  angles are always float32; RELION source read this session
  (`EMDL_DOUBLE`/`std::vector<double>` MetaDataTable storage,
  `rot_angles`/`tilt_angles` as `std::vector<RFLOAT>`) appears to contradict
  this. Left unresolved and Euler-angle sites left untouched this session
  pending either the missing justification or a correction.

### 2026-08-27 continuation: local_em_engine.py (largest P0 item) fixed;
### GPU/RELION parity check blocked by cluster resources this session

8 more files fixed (`local_em_engine.py`, `local_big_jit.py`, `k_class.py`,
`significance.py`, `preprocessing.py`, `image_shifts.py`, `projection.py`,
`sparse_pass2_bucketed.py`) -- `local_em_engine.py` was the P0 backlog's
largest single remaining item. Full detail in
`docs/math/relion_parity_agent_notes.md`'s 2026-08-27 entry, including a
real JAX scatter-add dtype-mismatch bug this round's own testing caught
and fixed (several `array.at[idx].add(value)` calls needed an explicit
`value.astype(target.dtype)` once an upstream forced-float32 cast was
correctly removed -- `.at[].add()` requires an exact dtype match, unlike
plain `+`/assignment).

**GPU/RELION parity still not obtained.** Four CPU-only Slurm attempts (up
to 300G requested) all hit `OUT_OF_MEMORY` at the same point regardless of
allocation size -- likely `recovar.utils.helpers.get_gpu_memory_total`'s
CPU-mode fallback reading the physical node's total RAM via
`psutil.virtual_memory().available` rather than any cgroup-limited amount,
which may be worth its own bug report/fix independent of this audit. A GPU
job queued but stayed `PENDING` on `QOSMaxCpuPerUserLimit` (this account
already had several other jobs running); not pursued further this session.
**No FSC/RELION-comparison quality claim is made for this round's fixes
(or the prior round's) -- only CPU-regression-clean, individually
RELION-source-justified.** This remains the mandatory first step of any
follow-up session before further fixing, per the validation ladder.

### 2026-08-27 continuation 2: reference-volume dtype fixed (user-reported);
### first GPU/RELION-oracle comparison this session obtained, resolving the
### "GPU/RELION parity check blocked" item above

User reported the actual volume being projected during EM scoring was still
`complex64`, correctly hypothesizing the float32-on-disk MRC read was the
root cause and that RELION widens to double after reading. Confirmed via
source read and fixed two narrow-then-widen bugs: `scripts/run_full_
refinement.py`'s initial-volume loading (all three branches: frozen-
boundary, K=1, K-class), and `recovar/reconstruction/relion_functions.py:
_pad_volume_for_projection_host` (the host-side FFT fallback for grids
`>=~293^3` at `padding_factor=2` -- was unconditionally forcing complex64
every iteration, independent of any upstream fix). Full detail, RELION
source citations, and the diagnostic trail in `docs/math/relion_parity_
agent_notes.md`'s 2026-08-27 "round 3" entry. Committed as `423b8d32`.

**GPU/RELION parity obtained this session**, per the user's instruction to
use the `gpu` Slurm partition (not `gpu_devel`, which round 2 found stuck
pending on a CPU-quota QOS limit) with GPU support loaded via `.vscode/
load_env.sh`'s exact module sequence. First attempt landed on a node
(`r816u35n07`) with a broken CUDA driver (`nvidia-smi` OK, but `cuInit`-
level `CUDA_ERROR_UNKNOWN`) that silently fell back to CPU and then hit the
still-unfixed `get_gpu_memory_total` CPU-RAM-autodetection bug flagged in
the entry above; root-caused via a dedicated diagnostic job (confirmed
node-specific, not a module-loading issue) and worked around with
`--exclude=r816u35n07` plus a fail-fast GPU-visibility assertion for future
jobs. Two successful runs against `relion_em_test_double_seeded` (igg_1d
K=1 `--firstiter_cc`) on a V100 node:

- `--max_iter 1`: double-precision merged `corr=0.999999`, FSC-AUC
  `0.999935`; float32 control merged `corr=0.999999`, FSC-AUC `0.999940`.
- `--max_iter 2` (needed to actually exercise the `init_vol_ft` fix, since
  `--firstiter_cc` iteration 1 uses an already-correct separate handoff):
  double-precision merged `corr=0.999999`, FSC-AUC `0.999591`; float32
  control merged `corr=0.999999`, FSC-AUC `0.999592`.

Both precision modes clear the `>=0.999` parity gate with no regression
from the fix, at this fixture's small (128^3) scale. This is the first
actual RELION-compared quality evidence obtained in this `double_parity`
audit track (rounds 1-2's fixes were CPU-regression-tested only, per their
own entries' explicit caveats) -- it validates only this round's two fixes
plus, incidentally, that the accumulated rounds 1-2 fixes (also active via
`RECOVAR_USE_FLOAT64_SCORING=1` in the same runs) don't regress quality
either, though rounds 1-2 were not isolated/attributed individually in
this comparison.

**Still open** (unchanged from the entry above except as noted): the P0/P1/
P2 backlog list is untouched this round (out of scope -- this round was
scoped to the user's specific volume-precision report); `get_gpu_memory_
total`'s CPU-RAM bug is still unfixed (caused this round's first GPU-job
attempt to crash once it silently fell back to CPU -- confirmed unrelated
to precision correctness); bug #2's fix (`_pad_volume_for_projection_host`)
has no GPU-scale empirical validation, only the CPU unit test, since this
fixture (128^3) is well under its `>=293^3` host-path threshold.

### 2026-08-28 correction: round 3's GPU validation didn't exercise its own
### fix (run_multi_iter_parity.py has an independent, unfixed init_volume
### path); fixed and re-validated genuinely this time

User pushed back on round 3's approach (wanted the fix moved into
`_run_relion_iteration_loop`), which on investigation surfaced two real
problems, not just a design preference -- full detail in `docs/math/
relion_parity_agent_notes.md`'s 2026-08-28 entry:

1. `_run_relion_iteration_loop` receives `init_volume` already in Fourier
   space; `jnp.fft` computes at whatever dtype the real-space array had
   going in, so casting inside the iteration loop after the FFT already
   ran (in the caller) cannot recover precision -- moving the fix there as
   literally requested would have been a no-op.
2. `scripts/run_multi_iter_parity.py` (the script this session's own GPU
   validation jobs used) and `scripts/run_comparison.py` build
   `init_volume` independently and call `refine_single_volume` directly,
   bypassing `run_full_refinement.py` (and its `423b8d32` fix) entirely.
   **Round 3's GPU validation numbers were real but did not exercise the
   fix they were reported alongside** -- both the "double-precision" and
   "control" runs were internally float32 the whole time for this
   specific volume-loading step.

Presented this to the user with three options; chose to keep `423b8d32`'s
fix as-is and additionally apply the identical real-space-before-FFT
pattern directly in `run_multi_iter_parity.py` (not `run_comparison.py`
this round -- open gap). Fixed, verified via a standalone dtype check
against the real oracle MRC (confirmed genuine float32->complex64 vs
float64->complex128 behavior, not another no-op), and re-ran the GPU
validation (job 60357525, this time actually exercising the fix):
double-precision and float32-control merged `corr=0.999999`, FSC-AUC
`0.999591` for both -- clears the parity gate, no regression, though at
this small fixture's scale the two runs' numbers converge closely enough
that the GPU comparison itself isn't strong evidence either way; the
standalone dtype check is the real evidence the fix works. Also fixed an
unrelated environment gap this surfaced along the way: recovar's custom
CUDA extension's lazy `make`-based build needs `nvcc`, which `.vscode/
load_env.sh`'s `module unload CUDA` removes from `PATH` -- worked around
by building the extension explicitly once with CUDA still loaded (job
60357511) before any runtime job unloads it.

Committed as `bada1a2e`. **`scripts/run_comparison.py` still has the same
unfixed independent init_volume path** -- open for a future session.

### 2026-08-31 dense single-volume double-precision audit

A complete static/call-chain audit of `recovar/em/**`
removed additional live narrowing at shared stats, pass-1 priors,
correction/pre-shift, global/replay translation, dense/local noise, sparse
pass-2 geometry/prior/output, and K-class fallback boundaries. CPU fast guard
passes 16/16. GPU job `60368743` on the `gpu` partition confirmed an idle A100
and JAX GPU visibility, but all seven fast-parity cases skipped because this
cluster lacks their `/scratch/gpfs/GILLES/mg6942` fixtures. The integrated
batch is therefore not yet GPU quality-qualified. Detailed classifications
and intentionally retained boundaries are in `relion_parity_agent_notes.md`.

Clean three-iteration GPU replay `60374001` subsequently exercised the exact
seeded dataset and requested command. Its residual iteration-3 parameter gaps
were unchanged at displayed precision after the dtype audit. The next phase is
therefore an implementation/routing investigation, starting with a complete
inventory and controlled ablation of RELION-parity environment gates.

### Active hypothesis: ACC_DOUBLE coordinate flooring

The first controlled implementation-gap experiment tests
`RECOVAR_RELION_ACC_DOUBLE_FLOORF_QUIRK=1`. RELION's accelerated double path
still narrows interpolation coordinates before `floorf`, whereas RECOVAR's
complex128/manual projector otherwise retains double coordinates. The
disproof criterion is simple: if the iteration-1 shell/sigma residuals and
iteration-2 Pmax/direction-prior residuals are unchanged versus GPU baseline
`60374001`, reject this gate and move to fine-score execution ordering.

Result: rejected by GPU job `60374288`. All displayed per-iteration fields and
direction-prior differences were unchanged; saved state was identical apart
from negligible reconstruction/noise roundoff and timing/provenance.

### Active hypothesis: fine-rotation execution order

Test `RECOVAR_RELION_FINE_ROTATION_EXECUTION_ORDER=1` independently. It keeps
the candidate set fixed but orders fine rotations by RELION parent execution
order, potentially changing near-tie/reduction behavior. Reject if Pmax,
direction-prior, and pose/state arrays remain unchanged versus `60374001`.

Result: rejected by GPU job `60374344`. Pmax moved only at `1e-16`; all
displayed parameters and direction priors were unchanged.

### Active hypothesis: iteration-3 reference map

Substitute only RELION's reference maps at iteration 3 using the existing
fail-closed replay diagnostic. If the iteration-3 Pmax residual closes, trace
the iteration-2 BPref/reconstruction boundary. If it remains, localize inside
iteration-3 significance and fine Gaussian scoring.

Result: confirmed by GPU job `60374396`. Iteration-3 `ave_Pmax` improved from
`0.931704` to `0.932146` (RELION displays `0.9322`), and direction-prior
relative L1 improved from `3.73e-4/4.14e-4` to `2.12e-4/1.18e-4`.

### Active hypothesis: iteration-2 reconstruction boundary

Capture iteration-2 post-join BPref accumulators, full-precision pre-mask
Wiener maps, and post-mask per-iteration maps in one baseline run. Compare the
post-mask maps directly to RELION and decompose global/shell scaling; use the
pre-mask/BPref boundary to decide whether the next oracle instrumentation must
target accumulation or reconstruction/postprocessing.

Result: GPU job `60374462` captured the boundary. Iteration-2 post-mask maps
already differ from RELION by scale-fitted relative L2 `1.28e-3` (half 1) and
`2.70e-3` (half 2), while optimal global scale is `1.000013/0.999996`.
Therefore the causal reference mismatch is structured rather than a global
normalization error. Raw iteration-2 BPref and full-precision pre-mask dumps
are preserved under
`/home/ry295/palmer_scratch/tmp/recovar_em_it2_recon_60374462`.

### Active hypothesis: exact fine Gaussian scoring route

As a causal diagnostic, keep float64 projections and double x-half M-step but
disable float64 scoring so the existing RELION-exact direct diff2/minimum
route is exercised in iterations 2-3. Improvement in the iteration-2 map and
iteration-3 Pmax will justify implementing the same direct ordering in
float64; no improvement rejects scorer ordering as the map cause.

Result: rejected by GPU job `60374528`. The exact float32 direct-diff2 route
left iteration-2 map relative L2 at `1.28e-3/2.70e-3` and the displayed
iteration-3 Pmax/direction-prior residuals unchanged.

### Active hypothesis: inherited iteration-1 reference error

Replay RELION reference maps only at iteration 2, capture the resulting
iteration-2 post-mask maps, and compare them with RELION. A large reduction
means the iteration-2 map gap is inherited from iteration 1; persistence
places it within iteration-2 E/M accumulation or reconstruction.

Result: GPU job `60374590` reduced the iteration-2 scale-fitted map relative
L2 from `1.28e-3/2.70e-3` to `9.94e-4/1.24e-3`. Thus the first-iteration map
accounts for a substantial fraction of the later residual, especially in
half 2, while an approximately `1e-3` iteration-2 transition residual remains.

### Production exact first-iteration fine CC routing

The production K=1 first-iteration sparse pass did not request the literal
RELION fine normalized-CC scorer, even though its coarse probe did. The route
is now wired for K=1 and guarded by a focused unit test. GPU job `60374655`
showed this is a null attribution for the seeded fixture: iteration-2 maps and
iteration-3 Pmax, sigma-offset, and direction-prior residuals were unchanged.
The correction is retained as a source-faithful routing fix, but it does not
explain the measured map gap.

### Sparse image-power precision boundary

The sparse M-step still narrowed the support-weighted image-power vector and
per-image norm sum to float32 before shell/noise and host-float64 accumulation.
Those reductions now preserve the producer dtype; a focused float64 dtype test
passes. GPU job `60374777` confirms the change is live in the noise trajectory
(`noise_radial_iter_001` changes by up to `3.67e-2`, relative L2 `4.14e-8`),
but the reconstruction/Pmax effect is only roundoff (`ave_Pmax` `1.1e-16`,
final-map relative L2 below `6e-15`). It is a valid precision fix, not the
cause of the remaining parity gap.

### First-iteration pre-join BPref boundary

An isolated double-precision RELION build dumped each half's raw `BPref.data`
and `BPref.weight` immediately before `joinTwoHalvesAtLowResolution`; its
iteration-1 maps are byte-identical to the saved oracle. RECOVAR job `60374928`
captured the matching boundary. Both halves have identical support topology
(`support_jaccard=1`, zero mismatched coordinates), proving the gap is not
particle/pixel membership. After matched frame conversion/downsampling, half 1
has numerator/denominator relative L2 `8.97e-3/2.55e-3`; half 2 has
`4.53e-2/1.64e-2`. The asymmetry is explained largely by the known half-2
near-tie pose/translation flips, while radial denominator shell-sum residuals
remain only approximately `1e-6` to `4e-4` through shells 1-10. Thus the join
is not the first divergent operation: raw weighted-image and weight
backprojection inputs already differ, with the numerator the stronger signal.

### Sampling perturbation precision

The iteration loop still narrowed perturbed rotation matrices, working Eulers,
and translation grids to float32. RELION uses RFLOAT for perturbation and
working sampling coordinates, so these now follow the active scoring dtype in
both ordinary and final-all-data paths. Default float32 behavior is retained.
Focused tests pass 42/42. Fresh pre-join job `60375148` changed `Ft_y/Ft_ctf`
only at relative `1.8e-15` to `3.5e-15`, and full GPU job `60375147` left the
reported three-iteration trajectory unchanged. This is a real double-mode
contract fix but a null attribution here because a separate preserved-float64
M-step rotation path was already active and the winning candidates did not
change.

Follow-up found that this first patch was incomplete: the unperturbed Euler
grid had already been narrowed in `_relion_rotation_grid_float32` (and in the
sealed/final-grid variants) before reaching the newly dtype-aware perturbation
helper. That array is a working RFLOAT operand because RELION perturbs it and
reconstructs scoring matrices from it; it is not merely public STAR metadata.
The base Euler and translation grids now preserve the active dtype from their
point of construction.

This correction is causal. GPU job `60375338` reduced iteration-1 pre-join
numerator/denominator relative L2 from `8.97e-3/2.55e-3` to
`7.18e-3/2.55e-3` in half 1 and from `4.53e-2/1.64e-2` to
`5.66e-3/2.34e-3` in half 2, with support still exact. Full three-iteration
job `60375361` improved iteration-3 optimizer Pmax from `0.9317` to `0.9320`
versus RELION `0.9322`, sigma-offset mean from `1.4772` to `1.4768` versus
`1.4769`, and direction-prior relative L1 from `3.73e-4/4.14e-4` to
`2.23e-4/4.01e-4`. Iteration-1 direction priors and sigma offsets are now
exact at the report's precision. The large half-2 improvement confirms that
the earlier narrowing changed near-tie fine-pose winners.

A matched-particle operand check separately showed that RECOVAR's complex64
preprocessed Fourier image differs from RELION double by only `3.45e-8`
relative L2, close to RELION-to-complex64 quantization alone (`2.56e-8`).
That boundary is therefore not the source of the percent-scale pre-join gap.

The RELION projector builder also contained an unconditional complex64 cast on
`Projector::data`; this was removed so the binding's complex128 result reaches
double projection unchanged. Combined-tree GPU job `60382385` reproduces the
improved three-iteration trajectory above. Fresh pre-join job `60382549`
reproduces numerator/denominator relative L2 `7.18e-3/2.55e-3` (half 1) and
`5.66e-3/2.34e-3` (half 2). A complete iteration-1 particle audit now shows
exact Pmax and significant-support arrays, with maximum pose error
`1.52e-5` degrees and maximum translation error `2.42e-6` angstrom. Therefore
the remaining pre-join residual is no longer attributable to discrete pose,
translation, posterior-support, or particle/pixel membership differences; it
is inside the per-hypothesis projection/CTF/weighted-backprojection arithmetic.

### Double fused-scatter boundary result

The RELION fused x-half CUDA target now has separate compiled
complex64/float32 and complex128/float64 specializations selected by the FFI
handler from the operand dtype. Its upstream sequential translation reduction
likewise carries in RELION `XFLOAT` precision rather than forcing float32. With
the double specialization active, shells 15--21 of the raw iteration-1 BPref
match native RELION at roughly `1e-7`--`1e-6` relative L2. The residual is
concentrated at shells 22--23, especially the exact outer cutoff.

Two correctness issues in that path were also removed: the templated radius
predicate retained a float temporary, and already-host-inverted M-step matrices
were numerically inverted again by the generic scorer transform. After both
fixes, job `60382778` measures numerator/denominator relative L2
`9.21e-3/2.53e-3` (half 1) and `6.55e-3/2.25e-3` (half 2); shell 23 alone is
`5.58e-2/1.54e-2` and `3.95e-2/1.40e-2`. This does not yet improve the full
numerator metric, but it sharply localizes the next implementation check to the
exact device Euler matrix and spherical-cutoff membership. It argues against a
broad scatter/interpolation mismatch because the interior is already nearly
closed.

### Replay cutoff and serialized initialization boundary

The `--replay-override-max-iter` cutoff now changes both replay sources at the
same physical-iteration boundary: it stops explicit per-iteration model/data
overrides and returns sampling, expected-accuracy, and convergence transitions
to RECOVAR ownership. Focused cutoff tests pass 11/11. Jobs comparing cutoff 0
and cutoff 1 are nevertheless identical through three numbered iterations
(apart from approximately `1e-14` GPU scheduling noise). This is expected for
the cold start used here: both runs bootstrap from `it000`, and RELION's first
numbered sampling state is reproduced exactly from its optimizer seed. The
cutoff does not, and cannot, undo precision already lost in the mandatory
initial `it000` snapshot.

Native RELION job `60426716` captured the full binary64 fine-score operands for
particles 255, 311, 544, 652, and 719 at iteration 2. RECOVAR has exactly the
same candidate topology, fine translation angles, and Euler matrices. Using
the captured native image, noise/CTF weight, and projector in RECOVAR's packed
candidate order reproduces RELION's centered raw `diff2` with standard
deviation `4.2e-14`--`8.8e-14` and maximum error `1.2e-13`--`2.1e-13`.
Conversely, rebuilding each score from RECOVAR's own dumped operands reproduces
RECOVAR's stored score bit-for-bit. This closes candidate generation, fine
geometry, translation, projector interpolation, pixel order, direct `diff2`,
and the 256-lane reduction tree as possible algorithmic sources.

The remaining ordinary-replay centered score residual is
`1.5e-4`--`5.3e-4` RMS for the five-particle panel. Operand substitution
localizes it: the native projector or corrected image alone has little effect,
whereas substituting the native noise/CTF weight reduces the residual to
`2.2e-5`--`4.0e-5`; substituting all native operands reaches the numerical
closure above. RELION retains the initial double-precision noise spectrum in
memory, but `run_it000_half{1,2}_model.star` serializes `rlnSigma2Noise` with
six decimal places. The resulting shellwise relative error is approximately
`1e-4` and explains nearly all of the observed weight discrepancy. Because
first-iteration CC skips the first noise update, that rounded initialization
continues to affect the second iteration and can flip later near-tie poses.

This evidence classifies the remaining cutoff-run divergence as a serialized
initial-state precision/provenance boundary, rather than a remaining scoring
algorithm or forced-float32 mismatch. A meaningful uninterrupted double-parity
comparison must either replay a captured full-precision startup state or
recompute RELION's startup noise with source-identical reductions, without a
STAR round trip.

End-to-end validation job `60427343` initialized RECOVAR from the captured
full-precision startup noise and repeated the iteration-2 five-particle panel.
The centered raw-score RMS gap fell from `1.5e-4`--`5.3e-4` to
`4.7e-7`--`2.1e-6`, while posterior relative L2 fell to
`7.2e-8`--`3.7e-7`; all candidate and reconstruction masks remained exact.
The remaining tiny residual is compatible with the other serialized inputs
and accumulated arithmetic, while the approximately 70x--1100x improvement
directly confirms startup-noise rounding as the dominant apparent divergence.

### Default x-half accumulator dtype boundary

The double-parity work exposed a default-mode regression when the scoring
pipeline supplied complex128 reconstruction rows to the independently
configured complex64/float32 x-half BPref accumulator. The fused CUDA wrapper
correctly rejects mixed specializations, but its per-particle caller had
assumed that scoring and M-step precision always matched. Reduced data and
weight rows are now converted once to the selected accumulator dtype before
the particle launch loop. This retains float32 production behavior, supports
float64 scoring with a float32 M-step, and preserves the opt-in double M-step.

Focused CPU tests pass 10/10, including a regression with complex128/float64
rows and complex64/float32 accumulators. Default-precision GPU job `60441355`
completed the replayed first iteration without the dtype exception; poses and
translations matched the RELION boundary exactly. Job `60441232` was an
infrastructure-only false start because this cluster does not expose the
generic `/scratch/gpfs` runtime root.

### Dataset and CTF evaluation precision boundary

The double-scoring scripts previously changed the dataset/backend output dtype
only after `load_dataset` had already rounded CTF parameters, poses, and
translations to float32. CTF frequency grids were also constructed in float32,
and several dense scoring paths evaluated the CTF before casting its result to
float64. The dataset loader now has an explicit complex64/complex128 precision
contract that propagates through the image source/backend, metadata, subsets,
and independent reloads. Frequency grids and CTF evaluation use the parameter
dtype, and dense preprocessing, significance, local scoring, and sparse pass-2
cast CTF parameters before evaluation. The complex64 default remains unchanged.

The seeded particle STAR stores CTF fields as decimal text (generally six
digits after the decimal point). RECOVAR's STAR reader already parsed those
fields as float64, and RELION registers defocus metadata as `EMDL_DOUBLE` and
uses double `RFLOAT` unless built with `RELION_SINGLE_PRECISION`; the premature
RECOVAR loader cast therefore discarded real source precision. On this fixture,
rounding the loaded CTF metadata through float32 changes a defocus value by as
much as `9.375e-4` Angstrom.

Focused precision tests pass 7/7 and the CPU EM fast guard passes 16/16. GPU
job `60477812` completed the three-iteration float64 replay on an A100 80 GB.
Controlled job `60477853` repeated it with only the CTF metadata rounded through
float32. The full-precision and ablation runs selected exactly the same poses
and translations in all three iterations. Their BPref numerator and denominator
differ by only about `3.1e-8`/`1.9e-8` relative L2 in iteration 1 and
`6.8e-8`/`5.1e-8` by iteration 3; merged GT correlations differ by at most
`3.8e-10`. Relative-L2 gaps to RELION in `tau2` and `sigma2` change by less
than approximately `5e-9` absolute between the two arms. Thus this is a valid
precision correction, but it is not the dominant cause of the remaining
`tau2`, `sigma2`, or iteration-3 near-tie pose discrepancy.
