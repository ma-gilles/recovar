<!-- frozen-vdam-parity-scorecard-v3 -->
### Frozen VDAM / InitialModel RELION parity scorecard v3

**K=1 fixed-suite score: 2 / 20 strict trajectories passing (20 / 20
evaluated).**

**K=1 same-H100 runtime score: 0 / 20 comparable (RECOVAR is currently
4.91--11.58x RELION).**

Suite: `vdam-k1-gui-default-full20` (version 3; denominator frozen at 20).
Frozen case-definition SHA-256:
`9842b2c9cb7646d75127541801ef5982ed19e4a80485f9ce586ceabdb3ed0091`.

| Fixed K=1 v3 suite | Strict pass | Map pass | Particle pass | Schedule pass | Evaluated |
|---|---:|---:|---:|---:|---:|
| GUI/default, iterations 0--200 | **2/20** | **5/20** | **6/20** | **13/20** | **20/20** |

Progress against the unchanged denominator is **0 -> 2 strict passes**. A
checked case means that its complete map, particle-state, and pre-divergence
schedule contract passed; unchecked cases remain in the denominator. No
tolerance, baseline, case, or acceptance definition was changed to obtain a
pass.

The separate v2 expansion snapshot remains **6/15 accepted**. It covers
additional parameter stresses, is retained as a regression track, and cannot
change the frozen v3 score. K>1, real-data, and runtime are likewise separate
gates and cannot inflate K=1 correctness.

The checked-in scorecard, evidence, validator, and tests are:

- `docs/math/vdam_relion_parity_scorecard_v1.json`
- `docs/math/vdam_relion_parity_scorecard.md`
- `docs/math/vdam_relion_parity_dashboard.md`
- `docs/math/vdam_relion_parity_evidence_ledger_20260824_exactfine.json`
- `scripts/summarize_vdam_relion_parity_scorecard.py`
- `tests/unit/initial_model/test_vdam_relion_parity_scorecard.py`

Current diagnostic work is **non-scoring** until a fresh immutable candidate
passes all 201 checkpoints against the frozen same-physical-H100 native
envelope. Every promoted result records source, executable, CUDA and RELION
binding digests, fixture, GPU UUID, command, Slurm jobs, report paths, and
SHA-256 values.

| Current readout | Evidence | Decision |
|---|---|---|
| Frozen score | **2/20** strict; **0/20** runtime | draft, not merge-ready |
| Latest closed boundary | native trace-operation-order panel `13037659` completed both exact frozen seed-29 arms in 78 s | exact `GPU-6222...`; reference **fails** at `1.423x / 1.469x`, so matching the 40-register class is insufficient and no trajectory is active |
| Failed setup, non-scoring | attempted 0--200 job `13036861` exited after 25 s before checkpoint 0 | seed-0 schedule could not join seed-29 selected particles; no science/audit result |
| GF47 serial float32 | repeat spread falls sharply; full job `13025432` completed 201 checkpoints | audit `13026777` fails particle @58, schedule @59, map @79; runtime **8.75x** native |
| GF47 binary64 accumulator | repeats are bitwise exact | rejected: reference error is **5.60--5.96x** its native floor |
| GF47 reverse float32 order | panel `13026879`, audit `13026880` terminal | reference is inside its fresh native floor, but post-second-moment error is **2.90--3.57x** the floor; no promotion |
| GF47 native `Igrad2` oracle | panel `13027533` terminal | second moment becomes exact, but reference changes only to **1.03x / 0.72x** its native floor; not the dominant missing operation |
| GF47 H100 SM-strided float32 | panel `13028122` terminal | post-second-moment ratio improves to **1.56--1.68x** from **2.75--3.10x**; reference is **0.961--0.963x** its native floor |
| GF47 SM132 full qualification | science `13028371` completed 201 checkpoints; audit `13029112` is terminal fail-closed | first failures: schedule @58, particle @61, map @80; runtime **9.23x** native; score unchanged |
| GF47 iteration-58 score capture | exact-GPU job `13029200` completed from the same immutable head/binary | native-4 mode beats the native-1 runner-up by `0.002014`, entirely in raw data score; posterior/tie-break semantics are rejected |
| GF47 native H100 block chronology | build `13031123`, exact-GPU science/audit `13031189` terminal pass | 51,888 blocks / 200 launches / all 132 SMs sealed; measured start order predicts first atomics at `0.999914` rank correlation while SM132 is only `0.046226` |
| GF47 captured block-start replay | H100 CUDA gate `13031919` passes 18/18; same-arm panel `13032413` completes in 80 s | **not promoted**: mixed reference ratios `1.760x / 0.908x`; candidate-kernel chronology is the next discriminator |
| GF47 passive candidate chronology | H100 gate `13032739` passes 22/22; panels `13032901` and `13033223` complete in 78/80 s | 199,168 candidate blocks versus 51,888 native; both contain exactly 27,221 contributing blocks, matched for all 200 particles |
| GF47 mapped logical chronology | map seals and analyzer pass in both `13033223` arms | all 27,221 atomics biject; global order is close, but per-particle start/first-atomic rank is only `0.149--0.170` / `0.397--0.406`; test native order inside one concurrent grid next |
| GF47 concurrent native-grid replay | final H100 gate `13033838` passes 23/23; panel `13033864` completes in 80 s | exact full native logical grid and contributor bijections, yet reconstructed-reference and post-second-moment gates fail; static logical permutation is rejected |
| GF47 physical scheduler repeat | sealed from both `13033864` arms | median physical start-order repeat is only `0.319` with p10 `-0.100`; median contributing physical-index overlap is `0.328`; a reusable static scheduler permutation is rejected |
| GF47 exact-native-cardinality replay | H100 gate `13034888` passes 24/24; traced panel `13034951` completes in 76 s | exact 48,824-block native grids and 26,241 contributors per arm; mixed reference/post-second ratios fail the promotion floor |
| GF47 untraced exact-grid control | panel `13034998` completes in 74 s | both reference arms fail at `1.485x / 1.306x`; tracing perturbs scheduling but does not explain the residual |
| GF47 CUDA resource topology | sealed `cuobjdump` audit of the exact binaries | candidate SGD uses **48 registers / 1,060 B shared** per block; native 2D-data SGD uses **40 registers / 48 B shared**; compile-time trace/order specialization is next |
| GF47 compile-time SGD specialization | H100 gate `13035150` passes 24/24; all specializations compile at 40 registers | non-target panel `13035208` passes reference at `0.666x / 0.668x`, but target-GPU `13035456` fails at `1.353x / 1.272x`; not robust |
| GF47 exact physical identity grid | local `1459edb1c`; H100 gate `13035817` passes 25/25; exact-GPU panel `13035899` completes | every raw data/weight half is inside the native-repeat magnitude, but reference is `1.668x / 1.346x` and post-second is `1.182x / 1.888x`; row indirection rejected |
| GF47 native per-worker issue order | sealed trace comparison | all eight worker chains are already exact, with zero inversions; this axis is closed |
| GF47 CUDA toolchain / device code | gate `13036222` passes 25/25; exact-GPU panel `13036273` completes | CUDA-12.6/PTX matching improves reference to `1.439x / 1.221x` and post-second to `1.052x / 0.196x`, but compiler target alone is insufficient |
| GF47 native trace instruction shape | seed-0 `13036723` passes, exact frozen seed-29 `13037011` fails | seed dependence is decisive: reference changes from `0.923x / 0.740x` to `2.841x / 2.355x`; no promotion |
| GF47 captured order + trace shape | native-order local `64db9df71`; H100 gate `13037580` passes 28/28; exact-GPU panel `13037659` completes | trace specialization reaches **40 registers / 40 B shared** versus native **40 / 48**, but reference fails at **1.423x / 1.469x**; register pressure alone is rejected |

> **Status: draft, not merge-ready.** K=1 correctness is the active gate.
> Runtime optimization starts from a sealed passing trajectory; K>1,
> real-data, and final CLI/GUI qualification follow K=1 closure.

#### Frozen case checkboxes

A check means the complete strict map, particle-state, and pre-divergence
schedule contract passes. Runtime remains open for every row.

|  |  |  |  |  |
|---|---|---|---|---|
| [ ] GF43 | [x] GF44 | [x] GF45 | [ ] GF46 | [ ] GF47 |
| [ ] GF48 | [ ] GF49 | [ ] GF50 | [ ] GF51 | [ ] GF52 |
| [ ] GF53 | [ ] GF54 | [ ] GF55 | [ ] GF56 | [ ] GF57 |
| [ ] GF58 | [ ] GF59 | [ ] GF60 | [ ] GF61 | [ ] GF62 |

Last scientific update: **2026-08-27 09:06 ET**

Tracking branch: `codex/vdam-relion-parity-20260820`

Base: PR #158 (shared supplied-map EM machinery)

Policy: focused VDAM tests and frozen trajectories only; **no generic RECOVAR
full/long suite** is being run for this campaign.

## Program board

This is the same fixed-scorecard convention as the supplied-map EM PR. Green
means a sealed scoring result; yellow is live or diagnostic-only; red is an
accepted failure. A successful short replay never changes the 20-case score.

| Program gate | Score / evidence | State | Promotion criterion |
|---|---:|:---:|---|
| K=1 full-trajectory correctness | **2/20** | 🔴 | 20/20 sealed quality/state/schedule passes |
| K=1 audit completeness | **20/20** | 🟢 | every frozen v3 row is terminal and sealed |
| K=1 same-H100 runtime | **0/20**, 4.91--11.58x | 🔴 | comparable to RELION after correctness closes |
| K>1 | short K=2/K=4 diagnostics pass | ⚪ | new fixed full-trajectory suite after K=1 |
| Real data | deferred | ⚪ | fixed RELION/RECOVAR datasets after K>1 |
| Unified CLI / GUI | backend/default contract exists | 🟡 | expose and requalify important controls |

### Active boundary board

| Priority | Case / first boundary | What is proved now | Live decisive evidence | Score impact |
|---:|---|---|---|---|
| 1 | GF47 repeatability starts @1 | exact seed/counts/workers/order, CUDA-12.6/PTX, trace shape, and 40-register kernel class are now combined | `13037659`: reference **1.423x / 1.469x**; raw boundaries straddle the native floor and noise is `2.745x / 3.858x` | no score change; close the remaining shared-memory/source and physical launch-scheduling differences, then require two-arm robustness |
| 2 | GF46 coarse cutoff @4 | support error is one rank-100/101 float32 score-spacing decision; geometry, posterior rule, and texture interpolation are rejected | fused-CUDA lane-partial capture is next | none; current fix remains partial |
| 3 | GF38 accuracy controller @20 | iteration-3 controller is closed; fresh 0--200 science completed in 2,110 s | audit `13018631` fails schedule @20, particle @27, map @60 | repair the iteration-20 accuracy fields, then rerun 0--200 |
| 4 | Frozen v3 matrix | all 20 science runs and audits are terminal | **2 accepted / 18 failed / 0 pending** | every failed row remains an explicit repair target |

### Gate breakdown

| Strict v3 gate | Pass | Fail | Pending | Readout |
|---|---:|---:|---:|---|
| Map envelope | 5 | 15 | 0 | GF44/GF45/GF47/GF59/GF62 pass |
| Particle-state envelope | 6 | 14 | 0 | GF43/GF44/GF45/GF47/GF49/GF62 pass |
| Pre-divergence schedule | 13 | 7 | 0 | failures: GF47/GF48/GF52/GF56/GF60/GF61/GF62 |
| All quality/state/schedule gates | **2** | **18** | **0** | accepted: **GF44, GF45** |
| Comparable runtime | **0** | **20** | **0** | 4.91--11.58x RELION |

<details>
<summary><strong>Detailed causal evidence, implementation checkpoints, and rejected attempts</strong></summary>

### Latest change

GF47 same-physical-H100 serial-orientation panel `13024626` completed both
repeats in 94 seconds from local unpushed commit `86324a705`. The diagnostic
keeps the production direct-CUDA arithmetic statements but launches each of
the 576 orientation blocks in deterministic order for each particle. It
reduces RECOVAR repeat spread from the production panel by **4.66x** for raw
BPref data, **4.89x** for BPref weight, **86x** for `mom1_noise_power`, and
**11.9x** for the reconstructed reference. The reconstructed-reference
cross-engine errors (`1.63e-6`, `1.49e-6`) now bracket the paired native-repeat
floor (`1.54e-6`), while RECOVAR's own repeat spread is `1.38e-8`. This is the
first positive GF47 causal intervention: within-particle orientation-block
atomic interleaving drives the repeatability failure. A systematic
post-second-moment mismatch remains (`2.75--3.10x` the native envelope), so
the short panel does not change the score.

Frozen 0--200 candidate job `13025432` completed all 201 checkpoints from
local unpushed head `ad0573df5` against the existing four-run GF47 native
envelope. It was pinned
to the original node and physical GPU, the frozen v3 scorecard, qualified CUDA digest `c39994b6e42a...`,
RELION binding digest `fcbb2a8356c2...`, and worker-schedule digest
`fedf84049b0b...`. The runner now fail-closes on those controls, explicitly
restores the diagnostic topology after its environment scrub, and submits the
full map/state/schedule envelope audit automatically after science. No generic
RECOVAR full or long suite was involved. Placement attempts `13024925`, `13024949`,
`13024950`, `13025040`, and `13025075` exited 75 before science on nonmatching
GPUs; `13025129` reached the correct GPU but rejected the older default
scorecard before science. They are infrastructure/provenance gates, not parity
outcomes.

The first correctly pinned science attempt, `13025193`, wrote iterations 0
and 1 and then failed closed before iteration 2 because the v2 worker trace
contains only the 200 particles selected at iteration 1. The
`single_rotation` topology intentionally discards captured owners and sends
every later particle to lane 0, so local commit `ad0573df5` still validates
the trace schema but no longer requires later subsets to join to iteration-1
IDs. Focused schedule tests pass, and replacement `13025432` crossed the
previous iteration-2 stop. It also crossed the former iteration-58 numerical
butterfly in the direct-map metric: relative-L2 at iterations 57, 58, and 59
is respectively `0.000296`,
`0.000522`, and `0.000710`, versus native repeat diameters `0.0258`, `0.0266`,
and `0.0273`. Those are 1.15%, 1.96%, and 2.60% of the native direct-map
diameter; the strict FSC/FSC-AUC audit below still controls promotion.

The matching partial particle audit is now decisive and prevents promotion.
At iteration 58, 199/200 active particles match at least one of the four
native states, but `2411@particles.128.mrcs` matches none. The candidate
orientation is `(-140.184378, 99.449195, 161.165527)` with Pmax `0.214351`;
all four native repeats select `(-143.051050, 101.809809, 161.281477)` with
Pmax spanning `0.127128--0.257153`. Translation is exactly the same
`(0.536160, -3.713840)` on both sides. This reproduces the older divergent
mode exactly: deterministic serial orientation-block ordering repairs
repeatability but does not repair the stable trajectory branch.

Deferred audit `13026777` is now terminal and fail-closed with exit `1:0`.
Particle state first fails at iteration 58 on that same particle; schedule
first fails at 59 because `optimal_offset_change` matches no native repeat;
the strict map gate first fails at 79 and fails 122/201 checkpoints. At 79,
the candidate-to-best-native FSC-AUC is `0.998834761`, below the unchanged
`0.999` cross-engine gate; its GT nondegradation check still passes. The run
took 2,616 seconds versus the 299.1-second native median (**8.75x**).
Map/state/status report SHA-256 values are `319ca5bf5230...`,
`1ba614b21916...`, and `e5b8bba0df32...`. The score remains 2/20.

Binary64-accumulator discriminator `13026518` and dependent aggregate audit
`13026519` are terminal from local unpushed commit `75acbdef2`. Projection,
translation, residual, scatter coefficients, and final output dtype remain
RELION float32; only the accumulator storage is binary64 until the kernel
finishes. Both RECOVAR repeats are bitwise identical at every measured stage,
including the reconstructed reference. That stability is scientifically
rejected because the systematic result is worse: raw BPref-data error is
`1.29--1.36e-5` versus native repeat floor `9.37e-6`, `mom1_noise_power` is
`5.45--6.55e-5` versus `1.11e-5`, and reference error is `9.19--9.77e-6`
versus `1.64e-6`. RELION's float32 rounding order is therefore part of the
effective target; replacing it with the exact high-precision sum cannot solve
GF47. The sealed report SHA-256 is `4f3c1e780fda...`. The next bounded family
is deterministic float32 orientation order, beginning with reverse order.

That reverse-order discriminator is now terminal. Paired panel `13026879`
completed in 95 seconds and dependent aggregate audit `13026880` completed in
3 seconds from local unpushed commit `21a0546a8`. Both RECOVAR repeats remain
stable (`1.42e-8` reconstructed-reference repeat spread). Cross-engine raw
BPref data and weight stay near or inside the fresh paired-native floor, and
the reconstructed-reference errors (`2.19e-6`, `1.73e-6`) are below that
panel's `2.95e-6` native-repeat floor. The leading systematic boundary does
not improve: post-second-moment errors are **2.90x** and **3.57x** the native
floor, versus **2.75x** and **3.10x** for ascending serial order. Reverse
orientation order is therefore rejected as a promotion. Report SHA-256 is
`843ee68ac2b7...`. Arbitrary order search will not be scored; the next bounded
test must model a concrete native CUDA scheduling topology.

Paired native-second-moment oracle `13027533` completed in 96 seconds from
local unpushed commit `7137ed541`. It replaces only iteration-1 `Igrad2_post`
with the exact paired-native buffer; all E-step, raw BPref, first-moment,
reconstruction, and controller work remains RECOVAR. The intended boundary
becomes exact in both arms, but reconstructed-reference errors move only from
the ascending serial control's `1.06x` / `0.97x` native-floor ratios to
`1.03x` / `0.72x`. This does not identify `Igrad2` as the dominant missing
iteration-1 operation, so the substitution is not promoted. Report SHA-256 is
`94004dec050f...`; the local oracle remains diagnostic-only and unpushed.

The concrete H100 scheduling discriminator is positive. Local unpushed commit
`da5a287b1` serializes the unchanged RELION float32 orientation kernel in
132-SM-strided queue order (`sm`, `sm + 132`, ...), matching the hardware
topology of the frozen H100 SXM panel rather than sweeping arbitrary
permutations. Focused CPU/source guards pass 3/3; CUDA/FFI build job `13027944`
passes 1/1 in 166.44 seconds. The qualified CUDA SHA-256 is
`bc2630113e01...`. Paired panel `13028122` completes in 97 seconds: raw BPref,
first moment, noise, and reference remain at or inside their paired-native
floors; reconstructed-reference ratios are `0.963x` and `0.961x`. The leading
post-second-moment ratios improve from the ascending serial control's
`2.75x` / `3.10x` to `1.68x` / `1.56x`. Report SHA-256 is
`ed0e98a7b146...`. This passed the short promotion criterion, so digest-pinned
full task `13028369_2` (Slurm job `13028371`) ran against the unchanged
four-repeat GF47 envelope on the original physical H100. Provenance records
both `SLURM_JOB_GPUS=2` and UUID `GPU-6222c402...`, exactly matching the
accepted native panel. Pre-science attempts `13028205`, `13028298`,
`13028317`, and `13028342` respectively rejected a missing isolated Python
override and three wrong GPU UUIDs; array tasks 1 and 3 also rejected wrong
UUIDs in seconds, and pending tasks 4--8 were canceled as soon as task 2
acquired the target. They are infrastructure gates, not parity results.

The full science job completed all 201 checkpoints in 2,762 seconds. Automatic
audit `13029112` is terminal fail-closed with exit `1:0`. Schedule first fails
at iteration 58: offset range (`9.211616345` Angstrom) matches repeat 1 while
offset change (`2.878784465` Angstrom) matches repeat 4, so no single complete
native schedule matches. Accuracy fields are correctly inactive there.
Particles remain inside the four-repeat envelope through iteration 60, then
first fail at 61 on `604@particles.128.mrcs` (one unmatched active particle;
maximum nearest-native pose error `3.82999` degrees). Thus SM132 shifts the
serial particle boundary from 58 to 61, but does not close it.

The strict map boundary likewise moves only one checkpoint, from 79 to 80.
Iteration 79 improves from the serial candidate's failing `0.998834761` to
`0.999157509`, but iteration 80 falls to `0.998955489` and 121/201 checkpoints
fail. The minimum best-native FSC-AUC is `0.996981405`. GT nondegradation
passes all 201 checkpoints under the frozen `-0.002` gate; its minimum delta is
`-0.001315197`. Runtime is **9.23x** the 299.1-second native median. Map, state,
status, and completion report SHA-256 values are `0815b7504371...`,
`5fac40a308ba...`, `e5b8bba0df32...`, and `00cf819a5158...`. The frozen score
remains 2/20.

Exact-GPU iteration-58 capture job `13029200` then reused the EM coarse,
operand, and fused-posterior capture machinery for source row 2706
(`2707@particles.128.mrcs`). It reproduces the production winner exactly and
changes Pmax by only `1.71e-4`. The native-repeat-4 mode is first with posterior
`0.284572` and total score `-5.022034`; the native-repeat-1 mode is second with
posterior `0.283999` and total score `-5.024048`. Their `0.002014` gap is
entirely raw data score; rotation and translation priors cancel. A broad
epsilon tie-break is therefore rejected. Fused, local-score, and coarse-capture
SHA-256 values are `272f6f04fe40...`, `284c819b43a0...`, and
`ef98e6702817...`. The next bounded discriminator is passive native H100
orientation-block chronology followed by replay, not a posterior-rule change.

That passive H100 discriminator is now sealed. Isolated RELION source head
`b115ff523` records, for every iteration-1 SGD orientation block, launch
sequence, internal particle, worker/class, orientation row, SM ID, block start,
first real atomic, block end, and an explicit atomic-free flag. The buffer is
allocated on the main thread before OpenMP launch and written only after every
device bundle synchronizes. RECOVAR validator head `84159acfd` rejects wrong
magic/version/size, truncation or trailing bytes, missing launches, invalid
workers/classes/SMs, incomplete orientation bijections, impossible timestamp
order, unknown flags, and zero first-atomic timestamps unless the post-barrier
atomic-free flag proves that the block accumulated nothing. Focused validator
and orchestration guards pass **11/11**; no generic RECOVAR suite was run.

The first trace build `13030554` compiled and sealed the schema. Exact-GPU job
`13030681` then completed 201 checkpoints and proved the instrumentation inert
at iteration 1 (FSC-AUC `0.999999999948`, exact poses/translations), but its
validator correctly rejected 24,667 legitimate blocks without a pass-0
atomic. Corrected job `13031011` was stopped after the live validator proved
those blocks can remain atomic-free across all passes. These are fail-closed
instrumentation outcomes, not parity failures. The final schema marks that
state explicitly. Node-local build job `13031123` reduced CMake generation
from roughly seven minutes on GPFS to **0.3 s** and completed the full build in
94 s. Qualified executable SHA-256 is `e9b6fe53b66f...`.

Final exact-physical-H100 job `13031189` completed all 201 native checkpoints
and every automatic audit in 328 s (`wall_s=317` for RELION itself, **1.06x**
the 299.1-second native median). The capture seals **51,888** records from
exactly 200 particle launches, all eight workers, and SM IDs 0--131. Of these,
27,221 blocks perform real atomics and 24,667 are explicitly atomic-free.
Iteration-1 map FSC-AUC is `0.999999999947`; all 3,000 poses and translations
match exactly; maximum Pmax difference is `1.7e-5`. Raw trace, sealed NPZ,
map audit, particle audit, and worker schedule SHA-256 values are
`d155936e868c...`, `1b99fb3c9db8...`, `15054ef9ba2b...`,
`4e10f32a0e4d...`, and `aeab283d529b...`.

The chronology rejects the SM132 heuristic decisively. Within-particle native
first-atomic order has weighted rank correlation only `0.046226` with SM132
and only `0.86698%` exact positions. Measured block-start order, however,
predicts global first-atomic order at rank correlation `0.9999138`. The next
bounded implementation therefore joins captured internal particle IDs through
the sealed worker trace and replays measured block-start order. It does not
widen a tolerance, change posterior selection, or promote the score before a
fresh repeat panel and unchanged 0--200 audit pass.

That bounded replay is now closed without promotion. Local unpushed science
head `1e23e356d` joins each native arm's v2 worker schedule to its v1 block
chronology, retains RELION's full local orientation grid (including
zero-posterior rows), and validates that RECOVAR's static bucket contains the
complete native eight-row-padded prefix before launching CUDA. Focused H100
gate `13031919` compiled the FFI and passed **18/18** tests in 62 seconds;
qualified CUDA SHA-256 is `3c2858a8c5ac...`. Precursor panels `13032103` and
`13032307` failed closed before producing a parity result: the first exposed
nonmatching packed cardinality, while the second proved that blindly
truncating native rows would discard **1,696 real atomic blocks** across 31
particles. Those findings drove the full-grid/prefix guard rather than a
permissive fallback.

Same-H100 two-arm panel `13032413` then completed in 80 seconds. Each arm
captured and sealed its own **51,888** native blocks before RECOVAR replayed
them, so no chronology was borrowed across native runs. Raw accumulator
data/weight and first-moment errors are generally at or inside the paired
native floor, but the decisive reconstructed-reference ratios are mixed at
**1.760x** and **0.908x**; the previously qualified SM132 panel was
**0.963x / 0.961x**. Post-second-moment ratios are `1.223x / 1.597x`.
Therefore measured native start order replayed as serial one-block launches is
not equivalent to RELION's one concurrent grid and is rejected. Report,
arm-A chronology, and arm-B chronology SHA-256 values are
`3fcb737b9ac1...`, `60c14c993a97...`, and `beccfa4a1003...`. No 0--200 job
was submitted and the frozen score remains 2/20.

The passive candidate-kernel discriminator is now sealed. Local unpushed head
`8536145f1` adds opt-in timestamps and identity fields to the production
concurrent kernel without changing its arithmetic. H100 build/FFI job
`13032739` passes **22/22** focused tests; qualified CUDA SHA-256 is
`d17a5a231653...`. Same-H100 repeat panel `13032901` completes in 78 seconds.
Each RECOVAR arm launches **199,168** physical blocks versus RELION's
**51,888**, but both execute exactly **27,221** real-atomic blocks. The atomic
count matches native for every one of the 200 particles; RECOVAR's remaining
171,947 blocks are explicit padding/no-ops. That rules out extra mathematical
contributions and makes a logical-row join mandatory before comparing order.

Local heads `bbc300aeb` and `601280906` therefore capture and fail-closed seal
the compact candidate-row to native orientation-row map, then analyze only
corresponding atomic work. The writer/validator, chronology, map, and panel
guards pass **25/25** plus **4/4** focused tests; no generic suite was run.
Mapped same-H100 panel `13033223` completes both arms in 80 seconds and proves
a **27,221/27,221** bijection: every candidate atomic maps to one native
atomic, every native atomic maps back, all 199,168 candidate rows are present,
and all 171,947 candidate padding flags agree with the device trace.

The mapped chronology localizes the remaining scheduler error. Candidate to
native global block-start rank correlation is `0.997236` / `0.997404`, and
global first-atomic rank correlation is `0.997277` / `0.997450`. Within each
particle, however, weighted start-rank correlation is only `0.149469` /
`0.170114`, and first-atomic correlation is only `0.396589` / `0.405685`.
Exact SM identity is not stable even natively and is not used as a promotion
target. The unchanged numerical gate remains mixed: reconstructed-reference
ratios are **1.006x / 1.593x** the paired native floor; post-second-moment
ratios are **1.080x / 0.479x**. Map arm-A, map arm-B, mapped chronology, and
repeat-panel SHA-256 values are `25bc30a8ea74...`, `95b826d2dc39...`,
`5ffe63c259fe...`, and `6a2697717136...`. No 0--200 run was submitted and the
frozen score remains 2/20.

The concurrent-grid discriminator is now terminal. CUDA gates `13033534` and
`13033737` passed 23/23 focused tests while two science submissions failed
closed before producing parity evidence: `13033678` exposed an obsolete FFI
geometry guard, and `13033769` exposed a double application of the captured
row offset. Neither failed submission is counted as a numerical result. Final
H100 gate `13033838` passed **23/23** tests from local unpushed head
`cdc4fb9d7`; qualified CUDA SHA-256 is `fccff56db4f7...`.

Same-H100 panel `13033864` then completed both arms in 80 seconds. Its v2 map
seals all **204,800** physical candidate blocks: **51,888** map bijectively to
the complete native logical grid, including **27,221** contributing and
**24,667** atomic-free native rows; the remaining **152,912** candidate blocks
are proven padding. This closes row identity and cardinality of mathematical
contributors, but not execution chronology. Reconstructed-reference errors
are **1.185x / 1.224x** the paired native floor and post-second-moment errors
are **2.304x / 1.879x**. Raw data, weight, and first-moment ratios straddle the
floor (`0.913x--1.098x`). The mechanism therefore fails the unchanged short
promotion gate, no 0--200 run was submitted, and the frozen score remains
2/20. Repeat-report, mapped chronology, map-A, and map-B SHA-256 values are
`9bcb6779853a...`, `a8a620b24346...`, `2ec174ac5661...`, and
`0f032803333a...`.

The physical scheduler audit rules out precomputing one inverse permutation.
When physical block IDs are recovered from append-order trace records and
timestamp ties retain midranks, the two candidate arms have median block-start
rank correlation only **0.319** (p10 **-0.100**, p90 **0.981**). The median
Jaccard overlap of physical IDs assigned contributing logical rows is only
**0.328**, and their first-atomic repeat correlation is **0.178**. Thus the
remaining error is dynamic launch/SM scheduling, not a missing logical row or
a reusable static block order.

The next bounded discriminator removes the **152,912** extra physical padding
blocks and launches each particle with its sealed native grid cardinality
while retaining the exact native logical-row permutation, worker ownership,
and one concurrent grid per particle. This tests the remaining launch-wave
topology without changing any contributing operand or arithmetic. Only a
fresh repeat panel that beats the unchanged native-floor gate can advance to
the fixed 201-checkpoint trajectory.

Local unpushed head `bf968e904` implements that discriminator as the opt-in
`captured_native_grid` topology. It threads a fail-closed per-particle launch
count through the JAX/CUDA FFI, trims trace and logical-map records to the
actually launched grid, and leaves the default production topology unchanged.
Focused CPU/source checks pass **16/16** plus **4/4** scheduler-analyzer tests.
H100 CUDA/FFI qualification job `13034888` completes in 55 seconds with
**24/24** focused checks; qualified CUDA SHA-256 is `32a082cc18b3...`.

Traced same-H100 panel `13034951` then completes both arms in 76 seconds. Each
arm seals exactly **48,824** native and candidate physical/logical blocks,
including **26,241** bijective contributors. Removing the fixed 1,024-block
padding improves one arm decisively, but the other remains outside the
unchanged floor: reconstructed-reference ratios are **0.736x / 1.414x** and
post-second-moment ratios are **0.596x / 1.473x**. Candidate repeat reference
spread (`1.256e-6`) is below native repeat (`3.182e-6`), so simple candidate
instability is not the full explanation. The traced mechanism is rejected,
no 0--200 run was submitted, and the score remains 2/20. Repeat and mapped
chronology SHA-256 values are `35bfc48afb22...` and `4d02fec3399a...`.

Candidate tracing performs a stream synchronization plus device-to-host append
after every particle. That is a direct perturbation of the cross-stream launch
topology under test. Panel `13034998` therefore repeats the exact-native-grid
mechanism from the same immutable head and qualified CUDA with candidate trace
and block-map capture disabled. The complete traced `13034951` panel remains
the row/cardinality proof. The untraced control completes in 74 seconds and
rejects tracing as the repair: reconstructed-reference ratios worsen to
**1.485x / 1.306x**. Post-second moment is **0.837x / 1.039x**; raw data and
weight ratios reach `1.03--1.98x`, with the half-1 residual dominating. No
0--200 run was submitted. Repeat-report SHA-256 is `eb95ebcdeaef...`.

The exact-binary CUDA resource audit now exposes a concrete launch-topology
mismatch beneath the logical grid. `cuobjdump --dump-resource-usage` reports
the RECOVAR float32 SGD specialization at **48 registers and 1,060 bytes of
shared memory per block**. RELION's actual 2D-data/3D-reference SGD
specializations use **40 registers and 48 bytes shared**. Both launch 128
threads, so the register difference changes resident-block occupancy and is a
direct mechanism for the poor within-particle scheduler match. Sealed RECOVAR
and RELION resource reports have SHA-256 `df3672f27bc9...` and
`f265aa38df54...`.

The next bounded implementation compile-time-specializes inactive candidate
trace and captured-order paths, then inspects the resulting binary before any
science run. It advances only if the untraced exact-grid specialization closes
the native 40-register occupancy class (or a separately justified equivalent)
and subsequently beats the unchanged two-arm numerical floor.

Local head `f1dce072f` completes that specialization. H100 CUDA/FFI gate
`13035150` passes **24/24** focused tests in 56 seconds; qualified CUDA SHA-256
is `6bf04699b95a...`. All float32 captured/untraced SGD variants now compile at
**40 registers**, matching the native occupancy-limiting register count.
Specialized resource-report SHA-256 is `1558e8249de9...`.

The first same-H100 untraced panel `13035208` completes in 74 seconds and beats
the reference floor in both arms at **0.666x / 0.668x**. That is a valid short
promotion on `GPU-9f98...`, but the frozen GF47 envelope was produced on
`GPU-6222...`; no trajectory was launched from a different physical GPU.
Node-pinned panel `13035316` then showed that `della-h19g4` exposes multiple
H100 UUIDs. It received `GPU-9904...` and gave mixed reference ratios
`1.040x / 0.547x`, so it was retained only as non-target diagnostic evidence.

Commit `04727ef61` adds a fail-before-output physical-UUID guard to the repeat
panel. Guarded attempt `13035437` and fan-out attempts `13035455`/`13035457`
exited in 2--4 seconds with code 75 on wrong GPUs; pending `13035458` was
cancelled once the target allocation started. Job `13035456` then ran both
arms on the exact frozen `GPU-6222c402...` and completed in 76 seconds. Its
reference ratios are **1.353x / 1.272x** and post-second ratios are
**0.580x / 1.165x**. The target-GPU gate therefore fails, no 0--200 run was
submitted, and the score remains 2/20. Target report SHA-256 is
`e066e5607fb6...`.

The remaining static launch difference is the captured-order lookup itself.
Native RELION launches identity physical block IDs; RECOVAR currently assigns
captured logical rows to those IDs. The next bounded topology keeps the sealed
native per-particle grid cardinalities and worker ownership but launches
identity physical rows through the matched 40-register specialization. This
removes non-coalesced row indirection without changing operands or arithmetic.

Local commit `1459edb1c` implements that identity-grid topology. Focused CPU
validation passes **19/19** and both Slurm runners pass `bash -n`. The first
GPU submission `13035791` failed closed in three seconds on a mistyped source
digest; corrected H100 gate `13035817` passes **25/25** in 57 seconds and
qualifies CUDA SHA-256 `8fe9d5bf10a8...`. Setup attempts
`13035856`--`13035858` failed before allocation evidence because the disposable
root marker had not yet been created. After that fail-closed precondition was
fixed, UUID-guard attempts `13035898` and `13035900` exited 75 in four seconds
on wrong GPUs, while `13035899` acquired the exact frozen `GPU-6222c402...`.

Target panel `13035899` completes both arms in 77 seconds. All four raw
data/weight ratios are inside the native-repeat magnitude in both arms
(`0.654x`--`0.942x`), as are all four first-moment ratios
(`0.625x`--`0.978x`). The decisive reconstructed-reference ratios nevertheless
fail at **1.668x / 1.346x**; post-second moment fails at
**1.182x / 1.888x**, and noise power at **1.954x / 1.288x**. No trajectory was
submitted. Report SHA-256 is `597b0fd78de3...`.

Identity block rows therefore improve raw-error magnitude but do not reproduce
native error structure after reconstruction. A direct traced-panel comparison
confirms that the candidate issue sequence already matches RELION exactly in
each of the eight worker lanes: **0 inversions** in all eight chains. This is
consistent with sealed owner replay `13023005`; particle issue rank is not a
new discriminator and no redundant implementation is warranted.

The remaining static code-generation contract is materially different.
Instrumented RELION job `13031123` was built with CUDA **12.6** and
`CUDA_ARCH=80`; its executable embeds seven `sm_80` cubins and seven
`compute_80` PTX images. H100 cannot execute `sm_80` cubins directly and uses
the embedded PTX path. RECOVAR gate `13035817` was built with CUDA **13.1** as
native `sm_90` SASS and embeds no PTX. Thus the two nominally matched kernels
still reach H100 through different compiler and driver-JIT paths. The next
bounded control builds RECOVAR with CUDA 12.6 and the same `sm_80` plus
`compute_80` fatbinary contract, audits its selected resources, and applies
the unchanged exact-GPU two-arm gate before any 0--200 run.

Local commit `3d23f803d` parameterizes and seals the gate compiler and target.
CUDA-12.6 qualification `13036222` passes **25/25** in 63 seconds, emits the
same `sm_80` cubin plus `compute_80` PTX route as RELION, and qualifies CUDA
SHA-256 `92d4a12751f4...`. The candidate identity specialization changes to
**40 registers / 36 B shared**, removing the CUDA-13.1 1,024-byte overhead and
approaching native **40 / 48**. UUID guards `13036272` and `13036274` exit 75
on wrong GPUs; `13036273` runs both arms on exact `GPU-6222c402...` in 77 s.

The compiler/JIT match is beneficial but not a repair. Reconstructed-reference
ratios improve from `1.668x / 1.346x` to **1.439x / 1.221x**. Post-second
moment improves to **1.052x / 0.196x**, while raw data/weight ratios span
`1.018x`--`1.355x`. Both reference arms still fail, so no trajectory is
submitted. Report SHA-256 is `77d45023b5bc...`.

The residual 12-byte shared-memory/source-shape difference is explained by the
instrumented native kernel: the fresh native arms execute timer/trace writes
and a shared first-atomic claim, whereas the untraced candidate compiles that
path away. The next bounded control preserves native counts, identity rows,
worker chains, CUDA 12.6, and PTX JIT while executing the native trace
instruction shape into device-only scratch. It omits candidate host copies and
append overhead, so it tests the kernel instruction/scheduling effect alone.

Local commits `23b8d1519` and `07ce1826b` implement that device-only trace
shape and confine it to the sealed first iteration. Focused CPU validation
passes **15/15** plus **12/12**; matched-toolchain H100 gates `13036648` and
`13036828` pass **26/26**. The active trace specialization compiles at
48 registers / 40 B shared versus native 40 / 48, so short promotion does not
yet waive the resource-clean confirmation requirement.

Exact-GPU seed-0 panel `13036723` completes in 76 seconds and initially meets
the short trigger at **0.923x / 0.740x** reconstructed-reference ratios. Its
report SHA-256 is `713030746ae3...`. That panel did not export the frozen
case's `RANDOM_SEED=29`; the boundary runner therefore used seed 0. Attempted
0--200 job `13036861` correctly fails before checkpoint 0 because the seed-0
schedule cannot join the frozen seed-29 selected particles. It is a setup
rejection, not a trajectory outcome.

Exact-case replacement panel `13037011` exports `RANDOM_SEED=29`, runs both
arms on target `GPU-6222c402...`, and completes in 87 seconds. Its native grid
contains 51,888 blocks rather than seed 0's 48,824. Reconstructed-reference
ratios fail at **2.841x / 2.355x** and noise power at
**4.230x / 3.237x**, although post-second passes at `0.506x / 0.859x`.
Seed-0 promotion is revoked; no 0--200 run is active. Report SHA-256 is
`2cef2fb30e7b...`.

Local commit `58469769a` combines those two individually insufficient axes:
the sealed native logical row permutation for the 51,888-block grid and the
device-only native trace instruction shape. It retains CUDA 12.6, PTX JIT,
native counts, workers, and the exact frozen seed. H100 qualification job
`13037149` passes **27/27** and seals CUDA SHA-256 `1f53804c5192...`.

Exact-target panel `13037200` then completes both arms in 83 seconds on
`GPU-6222c402...`. Post-second-moment ratios pass at **0.296x / 0.790x**, but
the reconstructed reference is mixed at **1.783x / 0.943x**. Raw
data/weight ratios span `0.720x`--`1.100x`; first-moment half 0 remains just
outside the native floor at `1.019x / 1.119x`. Capturing the logical row order
therefore helps one repeat but does not remove physical scheduling sensitivity.
The two-arm promotion gate rejects it, and no 0--200 trajectory is submitted.
The report SHA-256 is `32e3b477a2e4...`; sealed arm-A schedule and chronology
SHA-256 values are `a02c779f173f...` and `f16a14512623...`.

Source comparison then exposes a concrete trace-order difference: native
RELION reads the first-atomic timer before interpolation indices and
coefficients become live, while the candidate did so after computing them.
Local commit `64db9df71` moves the instruction to the native point and adds a
source-order guard. Focused CPU validation passes **16/16**; matched-toolchain
H100 gate `13037580` passes **28/28** and seals CUDA SHA-256
`90fab33796ae...`. The captured-order + trace specialization drops from 48 to
**40 registers**, matching native's register class; shared allocation remains
40 B versus native 48 B. Setup-only jobs `13037521` and `13037545` fail before
compilation because of a bad head digest and Slurm comma parsing respectively;
they contain no science result.

Exact-target seed-29 panel `13037659` completes both arms in 78 seconds. The
reconstructed-reference ratios are **1.423x / 1.469x**, post-second is
`1.103x / 0.940x`, and noise power is `2.745x / 3.858x`. Thus the corrected
source order and 40-register kernel alter the scheduler outcome but do not
close parity robustly. No trajectory is submitted. Report SHA-256 is
`bcf56394c2dd...`; sealed arm-A schedule and chronology SHA-256 values are
`def306c6a832...` and `c29c4fb11611...`.

GF47 same-physical-H100 panel `13024070` completed all four fresh-native arms
in 147 seconds from local unpushed commit `d8faaea77`. Two default controls and
two exact-owner replays with eight concurrent host issuers ran on GPU
`GPU-9f98ccbf...`. The default cross-engine raw half-0 BPref data errors are
`8.19e-6` and `9.10e-6` against a `9.09e-6` paired native-repeat floor; its
reconstructed-reference errors are `1.66e-6` and `1.80e-6` against a
`1.98e-6` native floor. The owner/concurrency replay remains within that native
floor at the reference (`1.71e-6`, `1.83e-6` versus `1.94e-6`) but is much less
repeatable: RECOVAR reconstructed-reference repeat spread grows from
`1.64e-7` to `8.99e-7` (**5.49x**), raw BPref-data spread grows from
`1.93e-7` to `6.46e-7` (**3.35x**), BPref-weight spread grows from `9.83e-8`
to `3.35e-7` (**3.41x**), and `mom1_noise_power` spread grows from `4.93e-7`
to `1.25e-5` (**25.3x**). Concurrent owner replay is therefore rejected as a
production fix, and the scheduler branch is closed. The next bounded work is
VDAM-specific deterministic direct-kernel accumulation followed by the
post-BPref momentum/reconstruction amplifier. Report SHA-256 values are
`eca6f5e5667f...` (control) and `325347e3b87d...` (replay). No score is
promoted.

GF47 exact worker-owner replay `13023005` completed successfully in 39 seconds
on one H100 from local unpushed RECOVAR commit `be58f92d7` and instrumented
RELION source `2359a63b`. Trace-v2 records both RELION's internal particle ID
and the zero-based stack-image ID; its 200 stack IDs match RECOVAR's selected
set exactly. Every one of the eight per-worker particle sequences also matches
RECOVAR's replay sequence exactly. The sealed trace and schedule SHA-256 values
are `d8608c5db639...` and `5b51f8add792...`. Despite that exact ownership and
within-worker order, replay does not improve the paired iteration-1 boundary:
raw half-0 BPref data relative-L2 is `9.60e-6`, weight is `2.22e-6`,
`mom1_noise_power` is `1.78e-5`, and the reconstructed reference is
`1.81e-6`. Those values are outside or at the worse edge of the two production
controls (`7.18--8.97e-6`, `1.62--1.77e-6`, `4.84--7.76e-6`, and
`1.54--1.56e-6`, respectively). Static ownership/stream-chain mismatch is
therefore rejected. RELION still launches those exact chains from eight
concurrent CPU workers, while RECOVAR issued them from one controller thread;
that host-issue concurrency is the next and last scheduling discriminator.
No score is promoted. The failed four-second predecessor `13022326` is an
instrumentation-only schema rejection before RECOVAR science and is not a
parity outcome.

GF47 same-H100 reduction-mode panel `13020545` completed in 217 seconds from
local unpushed source `bdef96dcc`. Each of production, fused/block topology,
and sequential-translation plus fused/block topology ran twice, with a fresh
native RELION iteration-1 control for every repeat. Production reproduces the
boundary: its reconstructed reference differs between repeats at relative-L2
`2.5452e-7`. Fused/block increases that spread to `3.7124e-7`, and adding
sequential translation reduction increases it to `7.2414e-7`; neither generic
EM arm improves repeatability and both are rejected. The production raw BPref
data/weight accumulator spread is only `0.77--1.08x` the paired native-repeat
floor, while reconstructed-reference spread is `1.14--1.15x` and
`mom1_noise_power` spread is `1.92--3.09x`; the first strong amplification is
therefore after BPref accumulation. A source and call-route audit corrects the
earlier hypothesis-stream interpretation: RELION loops translations
sequentially inside each orientation/pixel thread and then performs one fused
eight-neighbour scatter. RECOVAR's active VDAM-specific direct kernel already
implements that statement order; the generic sparse-reduction toggles do not
replace this route. The next causal discriminator is RELION's dynamic
task-to-worker CUDA-stream assignment versus RECOVAR's static `particle % 8`
lane assignment, followed by the post-BPref momentum update. The sealed report
SHA-256 values are `fceae7f316d0...` (production), `ef7abe8b86a8...`
(fused/block), and `a435ed456c7f...` (sequential/fused/block). No score is
promoted.

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
sealed subset permutation as `part_id=1570`; the same reconstruction returns
GF46's independently successful native `part_id=2067` exactly. Continuation
capture `13017324_1` is nevertheless rejected: RELION does not serialize its
hidden shuffled order into the iteration-57 optimiser, so the replay leaves
particle 2411 unchanged at iteration 58 and cannot reproduce the sealed
boundary. Fresh-from-zero native task `13017550` reran the original
200-iteration command and seed on the original physical H100 and reproduced
the frozen native row exactly: Euler angles
`(-143.05105, 101.809809, 161.281477)`, origin
`(0.536160, -3.713840)`, and Pmax `0.159542`, inside the four-repeat native
Pmax range `0.127128--0.257153`. The first valid candidate task `13017675`
and repeat task `13018070` both select the same native orientation and
translation to STAR precision, with Pmax `0.215385` and `0.215386`.
The older uninterrupted 0--200 candidate selected
`(-140.184378, 99.449195, 161.165527)` instead. GF47 is therefore classified
as a repeatability/numerical-butterfly boundary, not a stable missing-geometry
case. A third science process, task `13018390`, again wrote the native pose but
Pmax `0.280921`; it is rejected because its shared CUDA source changed after
the job had copied the qualified binary, and the final provenance check
correctly exited 1. The verified `87274be...` binary is now sealed in a
private read-only source. Direct placement `13018863` exited 75 before science
because Slurm assigned another GPU on the correct node; replacement array
`13018950`, target task `13018955`, passed every provenance check and
reproduced the divergent orientation exactly, with Pmax `0.214371`. This
confirms both native-pose and divergent-pose modes on the same physical H100,
head, fixture, command, and immutable CUDA bytes. Neither mode changes the
frozen score.

The fully valid candidate captures provide the backward causal boundary.
Iteration 0 particle state and maps are bitwise equal. At iteration 1, particle
state remains exact but the native-pose/divergent-pose maps differ at
relative-L2 `2.4740e-7`; the first
nonexact metadata fields are GPU-accumulated `wsum_img_power`,
`wsum_sigma2_noise`, and class/half BPref weight sums. Pmax first differs at
iteration 2, one pose first differs at iteration 34, and map relative-L2
reaches `2.9470e-4` at iteration 57 and `5.2665e-4` at iteration 58. The
iteration-58 score-table differences are
therefore inherited amplification. The new production locus is the
iteration-1 accumulator reduction order/repeatability, where the supplied-map
EM machinery can be reused, not a one-particle fine-posterior formula.

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

Corrected diagnostic runner commit `e0600f351` permits the isolated worktree
to use the already-qualified shared pixi environment. Replacement job
`13018487` completed with exit 0 and saved the requested projected image,
shifted image, pixel weights, and compact-index map. It also proves that this
dump is not sufficient for the final atomic-order question: at rank 101
(`flat=1957`) the preprojected replay is `9.4236393`, while both production and
the fused replay using RECOVAR's exact projector are `9.4890919`. Enumerating
lane orders from the preprojected operands would therefore answer the wrong
kernel path. The next bounded instrument is a passive four-lane partial dump
inside the fused projector kernel for only ranks 100/101; no production
arithmetic is changed by that capture.

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

That qualification is now terminal. Science `13017334` completed all 201
checkpoints in 2,110 seconds; manual replacement audit `13018631` was required
because the array-level dependency saw the expected non-target GPU exits.
The audit fails schedule first at iteration 20 because both accuracy fields
match no native repeat, particle state first at iteration 27 on
`1166@particles.128.mrcs`, and the map gate first at iteration 60. The map at
60 still matches a native map mode (`0.9999999765` best-native FSC-AUC), but
its GT FSC-AUC is `0.00200234` below the best native repeat, just outside the
unchanged `0.002` nondegradation gate. The repair therefore moves the schedule
boundary from 3 to 20; it does not close GF38.
The full map/state report SHA-256 values are `99d2405c534c...` and
`07067bdb76f6...`.

</details>

### What is still failing

| Failure class | Current evidence | Next closure gate |
|---|---|---|
| Map/particle parity | GF43, GF46, GF48--GF58 and GF61 include classified map/state failures; GF49 and GF59 each pass one primary gate but fail another; GF47 has a divergent full-run mode at particle 2411 / iteration 58 despite two exact short replays | make the GF47 mode repeatable, then classify or eliminate the earliest boundary without changing gates |
| Controller topology | GF38 iteration-3 is closed but accuracy fields first leave the native envelope at 20; the divergent GF47 full mode first misses at 59 after its particle split; GF48/GF52/GF56/GF62 retain frozen strict failures | reproduce RELION schedule decisions exactly and requalify full trajectories |
| Runtime | every audited v3 case is 4.91--11.58x slower | profile only after a repaired trajectory passes end to end |
| Coverage | **20/20 v3 cases audited**; 2 accepted, 18 failed, 0 pending | keep the denominator frozen while repairing every failed row |

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
| GF53 | 29 | high resolution, radial noise | complete | fail @44 | fail @40 | pre-split pass | 4.91x | **FAIL: map/particle/runtime** |
| GF54 | 29 | midscale, Kent, radial noise | complete | fail @45 | fail @30 | pre-split pass | 7.81x | **FAIL** |
| GF55 | 101 | anisotropic, outliers, high noise | complete | fail @46 | fail @40 | pre-split pass | 7.98x | **FAIL** |
| GF56 | 101 | Kent, outliers, high noise | complete | fail @45 | fail @29 | fail @30 | 6.92x | **FAIL** |
| GF57 | 101 | anisotropic, severe outliers, radial/high noise | complete | fail @44 | fail @11 | pre-split pass | 10.40x | **FAIL** |
| GF58 | 101 | extreme outliers, uniform, white noise | complete | fail @94 | fail @48 | pre-split pass | 6.60x | **FAIL** |
| GF59 | 101 | very-high noise, uniform, white noise | complete | pass | fail @30 | pre-split pass | 6.86x | **FAIL: particle/runtime** |
| GF60 | 101 | low noise, uniform | complete | fail @42 | fail @40 | fail @20 | 8.73x | **FAIL** |
| GF61 | 101 | low noise, Kent | complete | fail @41 | fail @40 | fail @40 | 6.40x | **FAIL** |
| GF62 | 101 | Kent, junk particles, translations | complete | pass | pass | fail @20 | 7.21x | **FAIL: controller/runtime** |

GF53 closes the last pending v3 row. Audit `12999424_53` is terminal: schedule
topology passes, particle state first fails at iteration 40, and the map first
fails at iteration 44. Its 256-pixel runtime is 2,751.4 s versus a 560.5 s
four-repeat RELION median (`4.91x`). Map/state report SHA-256 values are
`19640c79540e...` and `bc505e3476f9...`.

The earlier expansion's accepted rows are GF27 (70% outliers), GF29
(low-noise uniform poses after input-orientation seeding), GF35
(`tau2_fudge=2`), GF36 (very-high noise with tau2 variants), GF37 (Healpix 2),
and GF40 (translation range/step 8/2). Its failures are retained as regression
targets; they are not removed from the program when a newer seed matrix is
added.

<details>
<summary><strong>GF38 causal ladder and sealed evidence</strong></summary>

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

</details>

<details>
<summary><strong>Per-case same-H100 runtime table</strong></summary>

## Runtime

| Case | RELION median | RECOVAR | Ratio |
|---|---:|---:|---:|
| GF53 high-resolution radial | 560.5 s | 2751.4 s | **4.91x** |
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

</details>

## Live work and next gates

| Priority | Work | Slurm / state | Exit condition |
|---:|---|---|---|
| 1 | Close GF47 seed-29 short gate | native trace-order `13037659` matches the 40-register class but fails reference at `1.423x / 1.469x`; no trajectory is active | close the remaining shared-memory/source and physical launch-scheduling differences; pass two exact-GPU arms before any 201-checkpoint run |
| 2 | Close GF46 coarse score-spacing residual | local science head `a8af8b28a`; focused guards 6/6; operand job `13018487` proves preprojected operands cannot answer the fused-kernel lane-order question | capture the fused ranks-100/101 four-lane partials passively, restore native support, then requalify iteration 4 and 0--200 |
| 3 | Repair GF38's replacement boundary | composed-head 0--200 task `13017334` completed in 2,110 s; audit `13018631` fails schedule @20, particle @27, map @60 | close iteration-20 accuracy rotation/translation, then rerun 0--200 |
| 4 | Frozen v3 matrix | **20/20 terminal: 2 accepted, 18 failed, 0 pending**; GF53 fails particle @40 and map @44 while schedule passes | retain every failure as a repair target |
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
