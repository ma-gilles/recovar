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
| Latest closed boundary | GF38 iteration-3 controller matches 4/4 native repeats | GF38 now fails later: schedule @20, particle @27, map @60 |
| GF47 serial float32 | repeat spread falls sharply; full job `13025432` completed 201 checkpoints | audit `13026777` fails particle @58, schedule @59, map @79; runtime **8.75x** native |
| GF47 binary64 accumulator | repeats are bitwise exact | rejected: reference error is **5.60--5.96x** its native floor |
| GF47 reverse float32 order | panel `13026879`, audit `13026880` terminal | reference is inside its fresh native floor, but post-second-moment error is **2.90--3.57x** the floor; no promotion |
| GF47 native `Igrad2` oracle | panel `13027533` terminal | second moment becomes exact, but reference changes only to **1.03x / 0.72x** its native floor; not the dominant missing operation |
| GF47 H100 SM-strided float32 | panel `13028122` terminal | post-second-moment ratio improves to **1.56--1.68x** from **2.75--3.10x**; reference is **0.961--0.963x** its native floor |
| Live GF47 SM132 boundary | full task `13028369_2` (`13028371`) has crossed iteration 161 on recorded GPU index 2 / UUID `GPU-6222...` | particles pass @58/@59 and strict map parity now passes @79 (`0.999157509`, formerly `0.998834761`); schedule still fails @58 as a cross-repeat mixture, so the score remains unchanged pending all 201 checkpoints and the sealed audit |

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

Last scientific update: **2026-08-27 02:39 ET**

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
| 1 | GF47 repeatability starts @1 | a 132-SM-strided float32 order cuts the post-second-moment floor ratio about 44% while preserving reference parity | full task `13028369_2` repairs the serial particle split @58 and strict map failure @79, but its live calibrated audit still finds a whole-native-schedule failure @58 | no score change; let 0--200 finish, seal every gate, then localize the now-narrower schedule boundary |
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
`ed0e98a7b146...`. This passes the short promotion criterion, so digest-pinned
full array task `13028369_2` (Slurm job `13028371`) is running against the
unchanged four-repeat GF47 envelope on the original physical H100. Provenance
records both `SLURM_JOB_GPUS=2` and UUID `GPU-6222c402...`, exactly matching
the accepted native panel. Pre-science attempts `13028205`, `13028298`,
`13028317`, and `13028342` respectively rejected a missing isolated Python
override and three wrong GPU UUIDs; array tasks 1 and 3 also rejected wrong
UUIDs in seconds, and pending tasks 4--8 were canceled as soon as task 2
acquired the target. They are infrastructure gates, not parity results. The
run has crossed iteration 99. A calibrated non-scoring live audit at iterations
58--59 shows a genuine boundary improvement: every active particle now matches
at least one native state at both checkpoints, including the particle that the
serial trajectory missed at 58, and the complete schedule matches native repeat
1 at 59. Iteration 58 still matches no *single complete* native schedule. Its
offset range (`9.211616345` Angstrom) matches repeat 1 (`9.211616`), while its
offset-change value (`2.878784465` Angstrom) matches repeat 4 (`2.878784`);
no single repeat matches both active fields. Rotation/translation accuracy is
intentionally outside this checkpoint's gate because accuracy estimation is
not active yet. Matching individual active fields across different repeats is
not accepted. This is an early failure localization, not a scoring audit. The
frozen score remains 2/20 until all 201 checkpoints and the automatic sealed
audit are terminal.

The old strict map boundary is independently repaired at iteration 79. The
SM132 candidate's best-native FSC-AUC is `0.999157509`, above the unchanged
`0.999` gate and above the serial candidate's failing `0.998834761`; its
candidate-minus-best-native GT FSC-AUC is `-0.000534989`, which also passes the
unchanged `-0.001` nondegradation gate. This checkpoint is positive live
evidence only: the iteration-58 schedule failure still prevents promotion.

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
| 1 | Close GF47 repeatability before another trajectory | H100 SM-strided panel `13028122` improves the leading residual; full task `13028369_2` has crossed iteration 161 and repairs particle parity @58/@59 plus strict map parity @79, but live schedule parity still fails @58 | finish all 201 checkpoints and the unchanged automatic audit; then repair the sealed first boundary without changing the gate |
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
