<!-- frozen-vdam-parity-scorecard-v3 -->
### Frozen VDAM / InitialModel RELION parity scorecard

This PR carries a fixed-denominator K=1 full-trajectory suite modeled on the
EM PR scorecard. The v3 denominator is **20 cases**, every case spans numbered
iterations **0--200**, and adding or changing cases requires a new suite
version. The checked definition SHA-256 is
`9842b2c9cb7646d75127541801ef5982ed19e4a80485f9ce586ceabdb3ed0091`.

| Fixed K=1 v3 suite | Passed | Evaluated | Denominator | Live science |
|---|---:|---:|---:|---:|
| All quality/state/schedule gates | **2** | 19 | 20 | 20 complete, 0 running, 0 queued |
| Comparable same-H100 runtime | **0** | 19 | 20 | measured range: **5.89--11.58x RELION** |

Progress against the unchanged strict denominator is **0 -> 2 accepted
trajectories**. Earlier expansion v2 remains a separate regression track at
**6/15 accepted** and cannot change the v3 score.

> **Status: draft, not merge-ready.** K=1 correctness is the active gate.
> Runtime, K>1, real-data, and final CLI/GUI qualification follow only after
> the K=1 0--200 suite has no unexplained failures.

| Executive readout | Current evidence |
|---|---|
| Verdict | **Not merge-ready**: K=1 quality is 2/20 and runtime is 0/19 |
| Newly closed boundary | **GF38 iteration-3 controller passes** all active fields against 4/4 native repeats |
| Earliest active science blockers | GF46 coarse cutoff @4; GF47 particle posterior @58; GF53 final matrix audit |
| Live qualification | GF47 exact boundary capture `13017324`; GF38 immutable 0--200 `13017334`; GF53 audit `12999424_53` |
| Publication policy | Science fixes remain local/unpushed; this PR publishes only the live evidence dashboard |

Last scientific update: **2026-08-26 20:47 ET**

Tracking branch: `codex/vdam-relion-parity-20260820`

Base: PR #158 (shared supplied-map EM machinery)

Policy: focused VDAM tests and frozen trajectories only; **no generic RECOVAR
full/long suite** is being run for this campaign.

## At a glance

| Gate | Current result | Required to close |
|---|---|---|
| Frozen K=1 v3 quality | **2/20 pass**, 19/20 evaluated | 20/20 accepted with no unexplained case |
| Frozen K=1 v3 production | **20/20 complete**; only GF53 audit running | 20/20 science artifacts sealed; all audits terminal |
| Earlier 0--200 expansion | **6/15 accepted**, 9 classified failures; GF42 remains outside the sealed count | Every failure repaired and requalified |
| First causal production fix | **151x lower iteration-1 map error**; composed GF38 iteration-3 controller now passes 4/4 native repeats; focused guards 12/12 | Promote fresh immutable 0--200 trajectory and classify the former state @27/map @60 boundaries |
| Checkpoint-state fix | GF47 iteration-2 pose match **93.8% -> 100%** in sealed replay; focused guards 2/2 | Close the replacement trajectory's first particle split @58 |
| Controller-precision fix | GF47 iteration-10 controller passes 4/4 native repeats; composed trajectory delays schedule failure to 59 | Capture and repair the particle-2411 posterior boundary @58 |
| Runtime | **0/19 comparable**; current range **5.89--11.58x RELION** | Comparable same-H100 wall time |
| K>1 / real data | Existing short K=2/K=4 panels pass; final campaign deliberately deferred | Requalify after K=1 closes |
| CLI / GUI | Unified backend/default contract exists | Final defaults and important controls requalified |

### Gate breakdown

| Strict v3 gate | Pass | Fail | Pending | Readout |
|---|---:|---:|---:|---|
| Map envelope | 5 | 14 | 1 | GF44/GF45/GF47/GF59/GF62 pass |
| Particle-state envelope | 6 | 13 | 1 | GF43/GF44/GF45/GF47/GF49/GF62 pass |
| Pre-divergence schedule | 12 | 7 | 1 | failures: GF47/GF48/GF52/GF56/GF60/GF61/GF62 |
| All quality/state/schedule gates | **2** | **17** | **1** | accepted: **GF44, GF45** |
| Comparable runtime | **0** | **19** | **1** | 5.89--11.58x RELION |

### Latest change

GF38's composed-head iteration-3 discriminator `13014075_3` is terminal and
passes the complete active controller envelope against all four native
repeats. The formerly failing `optimal_offset_change` moved from
`3.303406559990986` to `3.303406454212302`; RELION serializes `3.303406`, so
the error is now `4.5421e-7`, inside the unchanged `5.1e-7` gate. Current
resolution/size, Healpix order, translation topology, offset range/step,
orientational-prior mode, and perturbation all pass. The only mismatch is
`nr_iter_without_resolution_gain`, which is diagnostic-only at this boundary
and cannot affect the active decision. The sealed report SHA-256 is
`e18cfa5b556b3ef86409d5e5c48d19e25e1c3c552fd94012838133b6e17b7545`.
This closes the short controller discriminator without promoting the older
0--200 trajectory; the next GF38 gate is a fresh immutable composed-head
trajectory and full audit.

GF47's apparent first particle-state split at iteration 2 is now localized to
checkpoint serialization, not the E-step. RECOVAR and RELION select the exact
same 200 particle identities, and native changes exactly those identities.
RECOVAR nevertheless rewrote 186 particles visited only by iteration 1 back to
their input-STAR poses because its `visited` mask was reset to the current
subset. Local unpushed commit `9685e9317` preserves cumulative visitation.
Replaying the sealed iteration-1/2 metadata changes pose match from **93.8% to
100%** (maximum angular difference `9.58e-6` degrees) while translation match
remains 100%. Focused guards pass 2/2. Exact-H100 replacement trajectory array
`13011552`, task 1, wrote all 201 checkpoints, but it is diagnostic-only: the
runner correctly failed its final provenance check because the worktree
advanced from `9685e9317` to `0a001923e` during execution. The RECOVAR process
itself exited 0. No score is promoted from that mixed-lifetime artifact.

GF47 also had a distinct pre-branch controller miss at iteration 10. RELION's
GPU search grid is float32, but persistent particle origins and hidden-variable
monitoring use CPU `RFLOAT` double precision. RECOVAR incorrectly reused the
float32 search translations for both roles. Local unpushed commit `0a001923e`
keeps the GPU grid float32 while preserving float64 metadata translations.
Sealed replay moves the iteration-10 range from `14.077556407` to
`14.077556605`, which serializes to RELION's `14.077557` and falls inside the
unchanged `5.1e-7` gate. Focused guards pass 6/6. Exact-H100 task
`13011940_1` completed on the target GPU; its calibrated iteration-10 mode
matches all four native repeats across every active controller check. The
report SHA-256 is `fea5a1824e0f29a4f516098ba0d58ab7715440a15add2d614afda2c9342fb5bc`.
The composed head is now isolated in an immutable worktree. Two replacement
attempts were rejected before promotion: `13013006` failed its target preflight
from a wrong working directory, while `13013240_1` detected a private CUDA
binary rebuilt from stale timestamps and was canceled. Fresh exact-H100 array
`13014045`, target task 1, completed all 201 checkpoints in 2,291 seconds
against the original four-repeat native panel from immutable head `0a001923e`.
Its private CUDA digest remained the qualified `87274beac3a7...`. Frozen
candidate-envelope audit `13015426` is now terminal and does **not** accept the
replacement: map first fails at iteration 77, particle state at 58, and
schedule at 59. This is a substantial delay from the original schedule miss at
iteration 10 and confirms both fixes are active, but it is not a parity pass;
no replacement score is promoted. The first state split is one selected
particle, `2411@particles.128.mrcs` (source row 2410), which chooses a
different orientation while retaining the exact native translation. Its
RELION internal identity has been reconstructed and checked through every
sealed subset permutation as `part_id=1570`. Exact native/candidate
iteration-58 capture array `13017324` is submitted against that internal ID on
the original physical H100; the earlier stack-index-only capture is rejected
because it produced no native score artifact.

GF46's iteration-4 particle split is now causally localized. On the 3,200
native-supported fine hypotheses, centered score relative-L2 is
`1.4234e-6`, and both engines choose the same winner. RECOVAR nevertheless
retains eight candidate-only fine rotations carrying 1.81% posterior mass;
all eight descend from one extra coarse parent. The native coarse cutoff keeps
100 hypotheses, while RECOVAR kept 101 because omitting RELION's common
per-image `min_diff2` term changed float32 cancellation in
`score + (50 - maximum)` and collapsed ranks 100/101 into a false tie. Local
unpushed commit `a8af8b28a` restores that absolute score frame. Offline replay
reproduces the native 100-hypothesis support exactly, and the focused
coarse-posterior suite passes 6/6. The exact-H100 live discriminator
`13016080` completed successfully at the process/provenance level but did
**not** repair the scientific boundary: the same one particle and eight fine
rotations remain outside native support. The offset makes the aggregate coarse
sum match the offline replay (`1.9297355e23`), but live score variation shrinks
the rank-100/101 gap from native `3.8147e-6` to `2.8610e-6`; RELION's final
`score + (50 - maximum)` float32 addition still rounds those two candidates to
one weight. The correction is therefore partial and unpromoted. Placement
attempts `13015679`/`13015770`/`13015922`/`13015995`/`13016022` all exited
before science (head-SHA or frozen-GPU misses); none are counted as parity
outcomes. Strict production gates remain unchanged.

Follow-up projector/reduction analysis `13016764` is terminal. Replaying the
captured RECOVAR projector reproduces production centered `diff2` within
`3.8147e-6` maximum, while native versus RECOVAR production differs by up to
`1.2398e-5` (p95 `5.2452e-6`). A fused native-texture projector intervention
does not change the rank-100/101 gap or support. The remaining defect is thus
localized to exact coarse score accumulation/rounding rather than the sampled
geometry, posterior rule, or texture interpolation. No tie relaxation is
being introduced.

The first exact production defect remains repaired locally at unpushed commit
`6387ff7c9`: the oversampling-zero big-JIT preserves RELION's retained coarse
posterior mass. Exact-H100 iteration-1 evidence improves posterior relative-L2
**258x**, noise relative-L2 **343x**, and map relative-L2 **151x**. Full 0--200
science task `13008435` completed all 201 checkpoints naturally in 2,152
seconds on the reference H100. Authoritative audit `13010186` is terminal and
does **not** pass: schedule first misses at iteration 3, one particle first
misses at 27, and the map first misses at 60. At iteration 3, every other active
controller field matches all four native repeats; `optimal_offset_change`
differs by only `5.60e-7` from RELION's serialized value, just outside the
unchanged `5.1e-7` gate. A short composed-head discriminator is queued as
`13014075`; no tolerance has been changed.

That short discriminator is now terminal and closes the schedule boundary:
`13014075_3` passes every active iteration-3 controller field against all four
native repeats. Because the earlier 0--200 run predates the composed
metadata-precision correction, its iteration-27 state and iteration-60 map
failures are not promoted as failures of the composed head. A new immutable
0--200 qualification is required.

### What is still failing

| Failure class | Current evidence | Next closure gate |
|---|---|---|
| Map/particle parity | GF43, GF46, GF48--GF58 and GF61 include classified map/state failures; GF49 and GF59 each pass one primary gate but fail another; composed GF47 now first splits at particle 2411 / iteration 58 | classify earliest boundary and repair without changing gates |
| Controller topology | GF38 iteration-3 is closed on the composed head; GF47 now first misses at 59 after its particle split; GF48/GF52/GF56/GF62 retain frozen strict failures | reproduce RELION schedule decisions exactly and requalify full trajectories |
| Runtime | every audited v3 case is 5.89--11.58x slower | profile only after a repaired trajectory passes end to end |
| Coverage | 19/20 v3 cases audited; all 20 science trajectories complete; only GF53 auditing | finish the final dependent audit |

No tolerance, baseline, or denominator has been widened to obtain a pass.
Native-repeat envelopes are used only when RELION itself branches; a candidate
must still remain inside the measured native quality/state envelope and obey
the pre-divergence controller topology.

## Current 20-case K=1 matrix

Every row is K=1 with GUI defaults, iterations **0--200**, four native RELION
repeats on the same physical H100, and an independent RECOVAR perturbation
seed. The audit checks maps, GT quality, particle state, schedule topology,
artifact topology, and wall time.

| Case | Seed | Distribution / stress | Science | Map | Particle | Schedule | Runtime | Overall |
|---|---:|---|---|---|---|---|---:|---|
| GF43 | 29 | baseline, uniform, white noise | seeded run complete | fail @146 | pass | pre-split pass | 7.88x | **FAIL: map/runtime** |
| GF44 | 29 | anisotropic, outliers, high noise | complete | pass | pass | pre-split pass | 8.03x | **ACCEPTED quality; runtime open** |
| GF45 | 29 | Kent, outliers, high noise | seeded run complete | pass | pass | pre-split pass | 9.33x | **ACCEPTED quality; runtime open** |
| GF46 | 29 | anisotropic, severe outliers, radial/high noise | complete | fail @20 | fail @4 | pre-split pass | 9.40x | **FAIL** |
| GF47 | 29 | extreme outliers, uniform, white noise | complete | pass | pass | fail @10 | 6.42x | **FAIL: controller/runtime** |
| GF48 | 29 | very-high noise, uniform, white noise | complete | fail @45 | fail @30 | fail @10 | 6.84x | **FAIL** |
| GF49 | 29 | low noise, uniform | complete | fail @115 | pass | pre-split pass | 11.07x | **FAIL: map/runtime** |
| GF50 | 29 | low noise, Kent | complete | fail @41 | fail @40 | pre-split pass | 11.58x | **FAIL** |
| GF51 | 29 | no CTF, radial noise | complete | fail @74 | fail @39 | pre-split pass | 5.89x | **FAIL** |
| GF52 | 29 | Kent, junk particles, translations | complete | fail @40 | fail @40 | fail @40 | 8.20x | **FAIL** |
| GF53 | 29 | high resolution, radial noise | complete | pending | pending | pending | pending | audit running |
| GF54 | 29 | midscale, Kent, radial noise | complete | fail @45 | fail @30 | pre-split pass | 7.81x | **FAIL** |
| GF55 | 101 | anisotropic, outliers, high noise | complete | fail @46 | fail @40 | pre-split pass | 7.98x | **FAIL** |
| GF56 | 101 | Kent, outliers, high noise | complete | fail @45 | fail @29 | fail @30 | 6.92x | **FAIL** |
| GF57 | 101 | anisotropic, severe outliers, radial/high noise | complete | fail @44 | fail @11 | pre-split pass | 10.40x | **FAIL** |
| GF58 | 101 | extreme outliers, uniform, white noise | complete | fail @94 | fail @48 | pre-split pass | 6.60x | **FAIL** |
| GF59 | 101 | very-high noise, uniform, white noise | complete | pass | fail @30 | pre-split pass | 6.86x | **FAIL: particle/runtime** |
| GF60 | 101 | low noise, uniform | complete | fail @42 | fail @40 | fail @20 | 8.73x | **FAIL** |
| GF61 | 101 | low noise, Kent | complete | fail @41 | fail @40 | fail @40 | 6.40x | **FAIL** |
| GF62 | 101 | Kent, junk particles, translations | complete | pass | pass | fail @20 | 7.21x | **FAIL: controller/runtime** |

The earlier expansion's accepted rows are GF27 (70% outliers), GF29
(low-noise uniform poses after input-orientation seeding), GF35
(`tau2_fudge=2`), GF36 (very-high noise with tau2 variants), GF37 (Healpix 2),
and GF40 (translation range/step 8/2). Its failures are retained as regression
targets; they are not removed from the program when a newer seed matrix is
added.

## Repaired first K=1 boundary: GF38 oversampling-zero posterior mass

The first map drift is no longer an undifferentiated reconstruction failure.
The causal ladder is now:

| Boundary | Observation | Classification |
|---|---|---|
| Iteration-0 maps and iteration-1 incoming moments | bitwise exact over 1,064,960 complex values | input state is not the cause |
| Raw BPref | data rel-L2 0.00263/0.00276; weight rel-L2 0.00162/0.00145 | first nonexact M-step artifact |
| Particle-0 hypotheses | identical 17 retained hypotheses | support/geometry are not the cause |
| Posterior normalization | RECOVAR mass 1.0000000137 vs RELION 0.9986371638 | coarse denominator differs by about 0.136% |
| Native `corr_img` factorial | coarse-score RMS 0.00449659 -> 0.0000362248 (**124x reduction**) | dominant error is the incoming radial score weight |
| Translation/accumulation-expression A/B | no improvement | coarse translation and written reduction expression rejected |
| Shell analysis | within-shell ratios constant to ~4e-8; shell means 0.99991797--1.00019730 | radial inverse-noise construction/state is the current locus |
| Fresh iteration-1 capture | native-vs-candidate score RMS 0.0000327728; native `corr_img` does not improve it | initial score weights are already correct to the residual floor |
| First noise update | iteration-0 noise rel-L2 9.19e-8; iteration-1 rel-L2 1.592e-4 | mismatch is introduced by the update, not initialization |
| High-shell denominator | native inferred sum-weight 199.6766 vs candidate metadata 200.0 | candidate renormalizes retained posterior mass to one |
| Production boundary | fallback local E-step accepts coarse Pmax; big-JIT never receives it | selected os0 hypotheses are silently renormalized only in production |
| Bounded correction | target posterior rel-L2 1.883e-3 -> 7.309e-6; noise rel-L2 1.592e-4 -> 4.645e-7 | cause and first update are repaired |
| Iteration-1 map | scaled rel-L2 2.759e-4 -> 1.822e-6 (**151x**) | promoted full-trajectory discriminator is justified |

The fresh full-schedule discriminator closes the initialization question.
Iteration-1 scoring is already at the same roughly `3.3e-5` residual floor
whether candidate or native `corr_img` is replayed; the 124x native-weight
rescue appears only after later state updates.  The production big-JIT omitted
the coarse Pmax normalization that its fallback path already honored, so the
100 selected os0 hypotheses were renormalized to unit mass.

Local correction `6387ff7c9` threads that normalization through the big-JIT.
Exact-GPU task `13007504_1` changes aggregate `noise_sumw` from `200.0` to
`199.676544`, within `6.1e-5` of the independently inferred native value.
The target posterior mass is `0.99811662` versus native `0.99811831`, with
identical support and argmax.  Iteration-1 noise relative-L2 improves 343x and
the map improves 151x. This implementation remains isolated and unpushed.
Arrays `13007637` and `13008037` exited 75 before science because Slurm
assigned non-reference GPUs. Task `13008278_1` reached the reference GPU but
was canceled before science when a stale-timestamp check attempted to rebuild
the shared qualified CUDA binary; the shared digest remained unchanged. Safe
replacement science task `13008435` used a private, read-only, digest-pinned
CUDA copy and completed all numbered iterations 0--200 with exit 0 in 2,152
seconds. Dependent audit `13010186` is terminal: the large iteration-1 defect
is fixed, but the trajectory still fails schedule at 3, particle state at 27,
and map at 60 against the existing four-repeat native envelope. This is a
partial causal repair, not an accepted trajectory.

Key sealed evidence:

- M-step boundary: `mstep_boundary.json`, SHA-256 `96302c189a16b463768efedd3cdd4ec35941265b09e1fea787a0bb39af120052`
- Particle posterior: `storewavg_particle0_posterior.json`, SHA-256 `2020b2260deeb25485c46d8c72e5855880820d1032962d195a8111c37966539d`
- Native-correlation factorial: `coarse_projector_boundary.json`, SHA-256 `2ccace9f0bc98b9acd5a5f075fd4d3195931569fce0f9ebcbbe88880aa8640af`
- Fresh iteration-1 score/noise discriminator: `coarse_projector_boundary.json`, SHA-256 `24b2a0e393ebf707cb538afcef6e3ecdc0f09ebbb4506a97ed029055a91ec863`
- Corrected posterior discriminator: `analysis_fresh_fused_posterior.json`, SHA-256 `6b005700b90ea2fe0cc802b6e45e332882118df782580fd2c711105c465455e3`

## Runtime

| Case | RELION median | RECOVAR | Ratio |
|---|---:|---:|---:|
| GF51 no-CTF radial | 281.2 s | 1655.6 s | **5.89x** |
| GF61 low-noise Kent, seed 101 | 588.2 s | 3764.1 s | **6.40x** |
| GF47 extreme outliers | 299.1 s | 1920.9 s | **6.42x** |
| GF58 extreme outliers, seed 101 | 289.5 s | 1911.3 s | **6.60x** |
| GF48 very-high noise | 286.8 s | 1962.6 s | **6.84x** |
| GF59 very-high noise, seed 101 | 286.1 s | 1963.1 s | **6.86x** |
| GF56 Kent/outliers, seed 101 | 291.2 s | 2016.2 s | **6.92x** |
| GF38 oversampling zero, partial repair | 308.4 s | 2152.0 s | **6.98x** |
| GF62 Kent/junk/translations, seed 101 | 319.6 s | 2302.9 s | **7.21x** |
| GF54 midscale Kent/radial | 351.4 s | 2745.5 s | **7.81x** |
| GF43 baseline, seeded repair | 297.3 s | 2343.2 s | **7.88x** |
| GF55 anisotropic/outliers, seed 101 | 378.3 s | 3018.6 s | **7.98x** |
| GF44 anisotropic/outliers | 286.2 s | 2298.7 s | **8.03x** |
| GF52 Kent/junk/translations | 323.3 s | 2651.3 s | **8.20x** |
| GF60 low-noise uniform, seed 101 | 403.4 s | 3520.9 s | **8.73x** |
| GF45 Kent/outliers, seeded repair | 297.4 s | 2775.8 s | **9.33x** |
| GF46 severe radial/outliers | 466.8 s | 4388.6 s | **9.40x** |
| GF57 severe radial/outliers, seed 101 | 480.6 s | 4998.7 s | **10.40x** |
| GF49 low-noise uniform | 406.6 s | 4501.6 s | **11.07x** |
| GF50 low-noise Kent | 436.3 s | 5053.6 s | **11.58x** |

Correctness remains the first gate. Runtime work will start from a sealed
accepted K=1 trajectory so performance changes cannot hide scientific drift.

## Live work and next gates

| Priority | Work | Slurm / state | Exit condition |
|---:|---|---|---|
| 1 | Classify the next GF47 boundary after composed repair | particle `2411@...`, source row 2410, verified RELION `part_id=1570`; exact capture array `13017324` submitted on the original H100 | identify the first posterior/support/score difference at iteration 58 before another 0--200 promotion |
| 2 | Close GF46 coarse score-spacing residual | local head `a8af8b28a`; focused guards 6/6; exact-H100 discriminator `13016080` terminal but still one particle outside native support | restore the native rank-100/101 score gap, then exact iteration-4 support/state and full 0--200 audit |
| 3 | Promote repaired GF38 | composed-head task `13014075_3` passes every active iteration-3 controller field against 4/4 native repeats; immutable 0--200 array `13017334` submitted | fresh trajectory must pass schedule/state/map audits |
| 4 | Finish v3 matrix | 20/20 science complete; GF53 audit running; 19/20 terminal | every row terminal and sealed |
| 5 | Seeded GF29 / GF43 / GF45 calibrated audits | GF29 and GF45 pass; GF43 fails only map at 146 | retain exact accepted/failed outcomes |
| 6 | GF41 authoritative re-audit | `12999430` terminal: map pass, particle fail | retain as classified repair target |
| 7 | Local corrections | `6387ff7c9` posterior mass; `9685e9317` cumulative checkpoint state; `0a001923e` metadata precision; `a8af8b28a` coarse min-diff score frame | full trajectories pass before implementation is published |
| 8 | Runtime, K>1, real-data, CLI/GUI finalization | deferred behind K=1 | comparable runtime and zero suite failures |

Implementation and diagnostic experiments remain in isolated local worktrees.
Only tracking/evidence commits are pushed to this PR while the science change
is unsettled; implementation changes are not pushed without approval.

## Validation policy

- Focused unit/source guards for the VDAM code being changed.
- Same-GPU native-repeat panels and all 201 numbered trajectory checkpoints.
- FSC/FSC-AUC map gates; no correlation fallback.
- Particle-state, controller/schedule, topology, provenance, and runtime gates.
- Exact source, CUDA, RELION binding, executable, fixture, GPU UUID, command,
  job ID, report path, and SHA-256 recorded for every promoted result.
- No generic RECOVAR full or long suite for this campaign, by explicit project
  direction.

Detailed chronological evidence remains in
`docs/math/em_parity_program.md`; this dashboard is the concise PR scorecard.
