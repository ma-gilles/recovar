<!-- frozen-vdam-parity-scorecard-v3 -->
### Frozen VDAM / InitialModel RELION parity scorecard v3

Suite: `vdam-k1-gui-default-full20` (version 3; denominator frozen at 20).
Frozen case-definition SHA-256:
`9842b2c9cb7646d75127541801ef5982ed19e4a80485f9ce586ceabdb3ed0091`.

| Fixed K=1 v3 gate | Current | Required | Status |
|---|---:|---:|:---:|
| Complete strict trajectories | **2/20** | 20/20 | 🔴 |
| Map envelope | **5/20** | 20/20 | 🔴 |
| Particle-state envelope | **6/20** | 20/20 | 🔴 |
| Pre-divergence schedule | **13/20** | 20/20 | 🔴 |
| Same-H100 runtime | **0/20** comparable; 4.91--11.58x RELION | 20/20 comparable | 🔴 |
| Terminal sealed audits | **20/20** | 20/20 | 🟢 |

### Why VDAM currently trails supplied-map EM

The headline scores are not a like-for-like measure of shared E-step maturity.
Supplied-map EM's frozen K=1 suite is currently **31/34**, not 34/34; its K=4
trajectory passes every class at **9/15** iterations, and its best current
large default-like K=1 timing is about **1.40x RELION**. VDAM's 2/20 suite
starts before a stable reference exists and fail-closes across as many as 201
self-updating checkpoints, particle state, controller state, and runtime. The
VDAM headline also includes a strict native-repeat particle-state envelope;
the EM headline is primarily an FSC/topology trajectory contract, so the two
fractions must not be read as the same test with different implementations.

| Axis | Supplied-map EM | VDAM / InitialModel | Consequence |
|---|---|---|---|
| Starting reference | Stable supplied map | Bootstrapped from the particles | VDAM must reproduce the initial basin as well as refinement. |
| Feedback | Scores a comparatively stable reference | Every small M-step difference becomes the next E-step input | Native-scale atomic variation can cross a later adaptive cutoff and avalanche. |
| Shared implementation | Authoritative significance, sparse pass-2, local-refinement, layout, posterior, and expected-accuracy code | Imports those same functions by object identity | The current gap is not a duplicate VDAM scorer implementation. |
| Algorithm-specific work | Conventional EM reconstruction/update path | InitialModel SGD BPref accumulation, gradient moments, pseudo-halfsets, reconstruction, and bootstrap/controller schedule | These pieces cannot simply call the supplied-map EM M-step because RELION uses a different update algorithm. |
| Current causal boundary | Several difficult noise/topology cases still remain | Raw GPU BPref accumulation realization | Replaying RELION raw BPref makes a paired 60-iteration VDAM trajectory effectively exact (`2.46e-15` output-reference relative L2, zero particle/operative-schedule divergences). |
| Runtime posture | Mature production path; best large run about 1.40x RELION | Reference-faithful and heavily guarded path; 4.91--11.58x | Correctness-first diagnostics and non-fused host work must be removed or optimized after a repeat-robust K=1 trajectory is sealed. |

The practical interpretation is therefore: the mature EM scoring machinery
has already been reused, and the remaining K=1 VDAM correctness problem is
narrow but unusually sensitive. The production task is to make the shared
BPref accumulation land reproducibly inside RELION's native distribution;
then qualify the feedback trajectory. The score must not be improved by
copying EM code, weakening particle gates, or selecting one lucky CUDA
realization.

The first launch-synchronized candidate closes GF46 twice through iteration
20 on the same physical H100, but fails the long gate. With the existing
shared `--deterministic-cuda` control (`CUDA_LAUNCH_BLOCKING=1`), jobs
`13116813 / 13117293` retain exact sampled particle/controller parity and map
FSC-AUC floors `0.9999999999565 / 0.9999999999567` through iteration 20.
Full job `13117709` first differs at one particle in iteration 33, grows to
178 particles by iteration 57, first fails the map gate at iteration 57
(`0.993022975`), and first differs in the particle-derived
`optimal_offset_change` controller statistic at iteration 33. Accuracy and
grid fields follow at iterations 40 and 52.
Launch synchronization is therefore a useful discriminator, not a production
correction; the raw-BPref accumulator remains the active causal boundary.

### At a glance: progress, failure, and next gate

| Status | Question | Readout |
|:---:|---|---|
| 🟢 | What improved? | Host-visible CUDA launch synchronization closes GF46 in two independent 0--20 runs on the exact same H100. Both retain **3,000/3,000** exact particle states at every sampled checkpoint, every controller/sampling field matches RELION, and the map FSC-AUC floors are `0.9999999999565 / 0.9999999999567`. |
| 🟢 | What is closed? | The GF46 discrepancy is not a VDAM-local formula, geometry, projector, prior, posterior, or significance bug: both observed rank-100/101 score gaps are exactly reachable from the same captured shared-scorer lanes. The architecture guard also proves InitialModel imports EM's authoritative significance, sparse pass-2, local-refinement, and layout functions by object identity. Native-posterior replay separately restores raw-BPref width to **0.81--1.07x native**. |
| 🔴 | What still fails? | The frozen score remains **2/20 strict** and runtime remains **0/20** at **4.91--11.58x** RELION. Synchronization delays GF46's first particle departure from iteration 4 to iteration 33, but it does not prevent the long feedback avalanche: the particle-derived optimal-offset statistic also differs at 33, 178 particles differ and map FSC-AUC is `0.993022975` by 57, and accuracy/grid fields follow at 40/52. |
| 🟡 | Why can both be true? | A synchronized 20-iteration run is repeat-robust locally, but it is not a proof about 200 feedback steps. Residual raw-BPref/map bias remains below the particle gate through iteration 32, then one particle switches at 33 and the difference compounds. This is the same sensitivity that the raw-BPref oracle removes; global launch ordering alone does not reproduce RELION's accumulator distribution. |
| 🔴 | First systematic departure | In synchronized full job `13117709`, `2538@particles.128.mrcs` first switches pose by 180 degrees and translation by `1.412063 A` at iteration 33. The divergence grows 1 -> 2 -> 7 -> 25 -> 69 -> 178 particles at iterations 33/37/42/48/52/57. |
| ➡️ | What is next? | Preserve `13117709` as a complete negative trajectory; focused job `13118385` will capture particle 2538's coarse/fine/fused boundary at iteration 33. Then implement and qualify the shared repeat-robust raw-BPref accumulator against RELION's native-repeat envelope. Re-enter @4/@20/@33/@40/@52/@57 before another 0--200 run; do not change the default from synchronization. |
| 🟢 | What finished? | Synchronized short jobs `13116158 / 13116369` pass through iteration 4; full-prefix jobs `13116813 / 13117293` pass through iteration 20. InitialModel still imports the authoritative EM significance, sparse pass-2, local-refinement, layout, posterior, and expected-accuracy machinery by object identity. |
| ⚪ | Score impact | Diagnostic-only: frozen score remains **2/20** and runtime remains **0/20**. No case, tolerance, denominator, or existing acceptance rule changed. |

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

<details>
<summary><strong>Full terminal diagnostic history (jobs, evidence, and decisions)</strong></summary>

| Current readout | Evidence | Decision |
|---|---|---|
| Frozen score | **2/20** strict; **0/20** runtime | draft, not merge-ready |
| Shared canonical reduction qualification | production `b90f22b1c`; build/test `13107529 / 13107668`; live 0--4 `13107898`; live 0--20 `13108111` | iteration-4 causal boundary is closed in production; zero particle-state divergence through iteration 19; frozen score unchanged pending full case qualification |
| GF46 iteration-20 controller boundary | candidate keeps sentinel accuracy `0 deg / 999 A`; native reports `2.391 deg / 1.88275 A`; candidate/native translation grids become `3.0 A / 116` versus `2.824125 A / 148` | shared E-step remains exonerated; seed all particle orientations from STAR, reuse the EM expected-accuracy implementation, and rerun the focused 0--20 gate |
| GF46 seeded-accuracy 0--20 qualification | science/audit `13108972`; map minimum FSC-AUC `0.999999972881`; iteration-20 accuracy/grid exact; strict particle first differs @4 and optimal-offset change @18 | controller correction is retained; run repeat-distribution and exact particle-286 cutoff capture before any frozen-score promotion |
| GF46 shared-wrapper repeat | science/audit `13109614`; same physical H100; map minimum FSC-AUC `0.999999969015`; identical particle-286 pose/Pmax split @4 | the departure is systematic rather than ordinary candidate-repeat width |
| GF46 expected-accuracy skip discriminator | science/audit `13110612`; expected-accuracy fields intentionally fail, maps pass at minimum FSC-AUC `0.999999999966`; all **3,000/3,000** particle states exact through @4 | expected-accuracy execution is causal; its scientific outputs are not the cause because the operative early grid is unchanged |
| GF46 production-projector chronology | science `194025e94`; focused guards pass; live `13111188`; maps pass at minimum FSC-AUC `0.999999998390`; sampling audit green; particle-286 split unchanged | RELION-order production projector preparation and E-step reuse are retained, but construction order alone is rejected as sufficient; exact particle capture supersedes this hypothesis |
| GF46 particle-286 exact boundary | fresh native `13111783` captured at @4 and was cancelled after the target; candidate `13111784`; native 100 parents versus candidate 101; one candidate-only parent `(67,14)`; centered total-score max delta `2.48e-5`; posterior TV `9.14e-7` | fine scoring and posterior normalization are downstream; the expected-accuracy-induced iteration-start reference residue crosses one adaptive coarse cutoff |
| GF46 expected-accuracy process isolation | shared science `33a96843b`; focused exact-result and strict-flag tests pass; 0--4 `13112338`; 0--20 `13112549`; zero particle or sampling failures; minimum map FSC-AUC `0.999999999952`; wall time 199 s | process-global side-effect leakage is causal and fresh-process 0--20 is closed |
| GF46 persistent CUDA-worker discriminator | science `9b95bc513`; live `13112995`; worker retained a second 550 MiB CUDA context; particle first differs @4, schedule first differs @18; maps pass at minimum FSC-AUC `0.999999969002`; wall time 512 s | persistent CUDA context lifetime is causal; do not promote the CUDA-backed worker |
| GF46 persistent CPU-worker discriminator | science `9b798db08`; exact direct/isolated focused test passes; live/audits `13113892`; no second CUDA context; particle first differs @4, schedule first differs @18; maps pass at minimum FSC-AUC `0.999999969011`; wall time 186 s | any persistent helper retains causal RELION state; CUDA context lifetime is not the full cause; revert worker reuse |
| GF46 default fresh-isolation qualification | production `9f2b77986`; default-contract focused test passes; live/audits `13114264`; particle first differs @4, schedule first differs @18; maps pass at minimum FSC-AUC `0.999999968982`; wall time 290 s (provenance-only typo submission `13114234` failed before science) | fresh CUDA-capable child is not robust across hosts; do not promote to 0--200 |
| GF46 fresh CPU-only isolation qualification | science `6ea5ac521`; exact direct/isolated result and parent-environment test passes; science/audits `13115118`; particle first differs @4, schedule first differs @18; maps pass at minimum FSC-AUC `0.999999969009`; wall time 195 s; filesystem-only attempt `13115035` cancelled before science | fresh CPU-only process boundary is rejected as sufficient; commits reverted, diagnostic helper retained |
| GF46 one-particle BPref bucket discriminator | shared-engine control `RECOVAR_K1_BPREF_EXECUTION_ORDER_CHUNK_SIZE=1`; science/audits `13115513`; exact same H100 as the systematic default red run; particle still differs @4; sampling passes; maps pass at minimum FSC-AUC `0.999999998391`; wall time 59 s through @4 | cross-particle atomics inside the default 220-particle bucket are not the sole cause; move inside the per-particle hypothesis scatter/reduction |
| Latest short boundary | materialized-order + native-epilogue panel `13038186` completed both exact frozen seed-29 arms in 90 s | exact `GPU-6222...`; reference **passes** at `0.978x / 0.967x`, earning full-trajectory qualification but not a score change |
| Latest full qualification | science `13038307` completed all 201 checkpoints; audit `13040047` is terminal fail-closed | first failures: schedule @58, particle @61, map @80; runtime **7.79x** native; score remains **2/20** |
| First-particle cause | exact it61 capture `13042355`; H100 operand replay `13043203` completed | **iteration-start map state**, not support, priors, image preprocessing, projector construction, noise weighting, or posterior math |
| Reference-clamp discriminator | exact-H100 task `13044790_2` / science step `13045526` completed 61 checkpoints in 377 s | **0 particle failures through it61**; all 61 replayed maps bitwise exact; operative schedule fields match |
| Late M-step operand boundary | exact-H100 task `13047664_1` / internal job `13047681` completed through it60 in 404 s | **0 particle failures through it60**; incoming `Igrad2` is the leading open state at `1.606e-3` relative L2, versus `2.88e-6--8.57e-6` for fresh raw BPref terms |
| Native `Igrad2` oracle at it60 | exact-H100 task `13048344_1` completed through it60 in 403 s | post-second moment becomes bitwise exact and reference error falls **262.0x**, from `1.038e-4` control to `3.963e-7`; diagnostic-only |
| Trajectory-wide native `Igrad2` oracle | exact-H100 task `13049505_1` completed 60 live-map iterations in 409 s | sampled map gate **passes** at minimum FSC-AUC `0.999999999900`; strict particle/schedule state still first departs @34, so no promotion |
| Trajectory-wide native `Igrad1 + Igrad2` oracle | exact-H100 task `13051779_1` / internal job `13051846` completed 60 live-map iterations in 427 s | all carried/post moment buffers and noise power are bitwise exact; reference error falls to `1.613e-8` and sampled maps pass at `0.999999999972`, but one particle/schedule choice departs @44; raw BPref is now the first non-exact stage |
| Trajectory-wide native raw-BPref oracle | exact-H100 task `13052694_1` / internal job `13052695` completed 60 live-map iterations in 430 s | raw data/weights and every downstream M-step operand become bitwise exact without moment/reference replay; an inherited `1.635e-8` reference residue remains and one paired translation departs @34; fresh paired RELION also leaves the old four-repeat state envelope, so no promotion |
| RELION float schedule precision | local science `7324440e2`; exact-H100 iteration-1 task `13054825_2` and iteration-60 task `13054923_2` / internal job `13054925` completed | matching RELION's float `x/a/b/scale` removes the reconstruction residue: it60 reference input/output are `2.527e-15 / 2.460e-15`; sampled FSC-AUC and assignment are `1.0`, with zero particle or operative-schedule divergences through 60; diagnostic-only, score unchanged |
| Float-schedule production discriminator | exact-H100 task `13056615_2` / internal job `13056622` completed 60 no-oracle iterations in 402 s; audit `13056914` completed | sampled maps still pass through 60 at minimum FSC-AUC `0.999997662`, but two particle states first split @58 and operative `optimal_offset_change` / `offset_range` split @58 / @60; it60 raw-BPref avalanche confirms accumulation order is the remaining production boundary |
| Fresh paired iteration-58 boundary | exact-H100 task `13058221` completed from local science `834b78c54`; contribution bundle and paired state audit are sealed | raw BPref remains non-exact, but the old particles 2411/2707 match exactly and only moving identity `46@particles.128.mrcs` differs among 3,000; native-vs-native repeat counts are 4/460/503/5, rejecting a fixed particle-specific correction |
| Stable-identity StoreWavg capture | exact-H100 `13059422_1` / internal job `13059423` completed in 390 s; posterior replay `13060135` completed | immutable stack 2707 resolved to run-local part 1898; all 128 candidate rotations occur natively, eight extra native rotations have exactly zero posterior, and all 116 translations map identically; this native run landed in a distant 462-particle mode, so its posterior difference is non-causal |
| State-aligned stable-identity capture | exact-H100 `13060276_2` / internal job `13060278` completed in 394 s; posterior replay `13060652` completed | paired map replay restores 3,000/3,000 particle states; stack 2707 posterior support is exact at 174 cells and its residual is only `1.39e-5` relative L2, rejecting support/normalization as the dominant aggregate BPref error |
| State-aligned StoreWavg operand split | exact-H100 `13061856_2` / internal job `13061877` completed in 389 s; split analyzer `13062288` completed | 3,000/3,000 states match; same-posterior translated image, denominator, and particle BPref close at `1.06e-7`, `7.17e-8`, and `1.05e-7`; live posterior explains the remaining `1.65e-5` particle residual, leaving aggregate multi-particle atomic order open |
| Isolated native particle accumulator | diagnostic H100 `13063621` completed in 406 s; analyzer `13063960` completed in 8 s | 3,000/3,000 states match; RECOVAR inline versus zeroed native-GPU particle BPref closes at data/weight relative L2 `5.43e-6 / 5.42e-6` with cosine `0.999999999985`; isolated scatter is closed and shared multi-particle accumulation is the first open production operation |
| Eight-thread round-robin discriminator | full-trajectory `13064681_0`, iteration-scoped `13064951_1`, and fixed-map `13065180_0` all completed in 381--384 s | enabling round-robin from iteration 1 changes the incoming trajectory; target-only replay with exact maps keeps 3,000/3,000 poses/translations but worsens raw data/weight relative L2 by about 12--13% in both halves, rejecting generic concurrency as the correction |
| Exact dynamic-worker discriminator | local science `a608d4eb9`; 34 focused tests pass; exact-H20g1 `13065579` queued behind a four-GPU reservation and available-H100 `13065713` queued | capture all native iteration-58 owner assignments, validate the schedule, and replay it only at iteration 58; no score impact until a complete trajectory passes |
| Exact dynamic-worker discriminator | available-H100 job `13065789` completed in 382 s with 200 selected particles, eight workers, and 25 particles per worker | 3,000/3,000 states match, but raw data/weight relative L2 remain `8.28e-6 / 7.75e-6` and `2.91e-6 / 2.70e-6`; exact owner identity is rejected as sufficient |
| Native iteration-58 block chronology | job `13065992` completed in 379 s; 200 launches, 5,720 blocks, all 132 SMs, and eight 25-particle worker chains sealed | host launch sequence predicts first block start / first atomic order at rank correlation `0.999994 / 0.999992`; particle ID itself is only `0.0523`, making measured global issue order the next causal axis |
| Captured global particle-issue replay | local science `deea2f427`; 68 focused tests pass; paired H100 job `13066700` completed in 390 s | 3,000/3,000 states match at iterations 57/58. Raw data becomes `7.14e-6 / 7.60e-6` and weight `2.50e-6 / 2.52e-6`: a **1.9--14.2%** improvement over exact-owner replay, but still above the isolated-particle floor; order alone is not sufficient and is not promoted |
| Serializing candidate/native chronology | local science `15a685150`; compile-level gate passes 58 focused tests; paired traced job `13068548` completes in 402 s | all 200 launch sequences and workers match exactly; first-start/atomic ranks match at `0.999994 / 0.999992`, but this diagnostic trace serialized candidate launches and reported a `0.0665x` span; the later passive trace supersedes its duration estimate while preserving the order evidence |
| Passive target-58 candidate chronology | local science `2edca4d23`; job `13069372` completed in 397 s | raw data remains `7.05e-6 / 7.06e-6` and weights `2.71e-6 / 2.40e-6`, consistent with the untraced floor; the passive candidate start span is `3,360,320` versus native `319,222,848` cycles (`0.0105266x`), so the earlier serializing trace overstated candidate duration |
| Captured launch-timing replay | local science `a43d6d073`; 26 focused tests pass; H100 build SHA `48f17d85...`; job `13070107` completed in 393 s | candidate/native start-span ratio is `0.99981485` with exact sequence/owners and rank `0.999994 / 0.999992`, but raw data worsens to `1.027e-5 / 1.009e-5` and weights to `4.50e-6 / 4.24e-6`; matching elapsed launch timing is decisively rejected as sufficient |
| Cross-realization native-count setup, non-scoring | local science `4cabaaf07`; jobs `13070381 / 13070382` stopped fail-closed before candidate accumulation | captured native block counts did not form a proven prefix of the current candidate posterior because schedule/chronology came from a different RELION realization; no science result and no tolerance was loosened |
| Same-arm native-count discriminator | runner fix `35025330f`; jobs `13070934 / 13070935` completed in 384/382 s | exact grid cardinality is not sufficient: issue+count raw data `9.12e-6 / 8.50e-6`, weights `3.35e-6 / 2.93e-6`; timing+count data `9.21e-6 / 8.09e-6`, weights `3.74e-6 / 2.79e-6`. Both chronology seals match their own RELION arm; no promotion |
| Within-particle native-order discriminator | local science `92214d9ec`; 52 focused tests pass; jobs `13071700 / 13071701` completed in 392/381 s | no promotion: issue+grid raw data `8.35e-6 / 1.03e-5`, weights `3.13e-6 / 3.93e-6`; timing+grid data `7.89e-6 / 8.76e-6`, weights `2.90e-6 / 3.11e-6`. Within-particle atomic medians remain `0.50 / 0.643`; a static block permutation does not control physical admission |
| Native physical-order repeatability | pairwise comparison of the four same-arm iteration-58 RELION seals from `13070934/35` and `13071700/01` | native-vs-native within-particle atomic-rank medians span `0.50--0.72`; candidate-vs-native is already inside that stochastic range, rejecting physical block order as a stable parity target |
| Production native-repeat qualification | science `f5e7d74ad`; jobs `13072500--13072503`; aggregate SHA `9d8f48f54e43...` | **32/32** raw data/weight comparisons are inside native repeat variability; cross/native ratio median `0.00203x`, p90 `0.01161x`, max `0.01283x`. Raw formula boundary is statistically green; repeated 0--200 trajectory qualification is next |
| Fresh GF47 full-repeat panel, interim 1/4 | science `f5e7d74ad`; job `13073643`; repeat 1 sealed all 201 checkpoints | point audit fails: one particle first differs @27 but rejoins @28; map first fails @75, minimum FSC-AUC `0.756025`, minimum GT delta `-0.0031605`; runtime `2151.9 / 350.3 s = 6.14x`. Distribution classification waits for all four repeats; score unchanged |
| Repeat-distribution analyzer compatibility | analysis `663575212`; focused sampling/state tests pass; legacy backtest `13076002`; live dependent audit `13075917` | old metadata without the newly recorded orientation-prior fields now fails closed in a complete report instead of raising `KeyError`; analysis-only, no acceptance change |
| Fresh GF47 full-repeat panel, interim 2/4 | science `f5e7d74ad`; job `13073643`; repeat 2 sealed all 201 checkpoints | point audit fails particle/map @4/13, minimum FSC-AUC `0.515764`, minimum GT delta `-0.006336`; runtime `1476.3 / 256.8 s = 5.75x`. The particle departure persists and expands, but distribution validity remains separate from point pairing |
| Same-GPU 2x2 state distribution | interim audit `13076823`; analyzer `dca338dcf` | strict particle support first fails @27 for one active identity; schedule candidate validity passes through @62, while reverse native coverage first fails @4 because both RECOVAR repeats select native mode 1 and neither samples native mode 2. Final 4x4 panel decides sampling insufficiency versus support bias |
| Joint schedule-distribution diagnostic | analysis `471b8cb1` plus paired-provenance compatibility `dca338dcf`; focused CPU jobs `13076563 / 13076678` pass 19/19 and 18/18 | preserves exact categorical modes, scales each continuous field by the nearest same-mode RELION repeat radius without changing frozen tolerances, and reports candidate validity separately from reverse native coverage; non-scoring |
| Fresh GF47 full-repeat panel, interim 3/4 | science `f5e7d74ad`; job `13073643`; repeat 3 sealed all 201 checkpoints | point audit first fails particle/map @71/115, minimum FSC-AUC `0.897837`, minimum GT delta `-0.003684`; runtime `1440.7 / 259.8 s = 5.55x`. Native repeat 3 exactly matches the candidate particle @27 and schedule @70 that the 2x2 panel called unsupported |
| Same-GPU 3x3 state distribution | interim audit `13077636`; analyzer `ac79e3348` | adding native 3 moves first unsupported particle state to @58 for candidates 1--2 and @104 for candidate 3; schedule candidate validity first fails @71. Reverse coverage still first fails @4 because all three candidates occupy native modes 1/3 and miss native mode 2 |
| Fresh GF47 full-repeat panel, final 4/4 | science `f5e7d74ad`; job `13073643`; four pairs sealed all 201 checkpoints | repeat 4 first fails particle/map @27/75, minimum FSC-AUC `0.753370`, minimum GT delta `-0.002603`, runtime `1445.7 / 257.4 s = 5.62x`. Four-run runtime ratio is median `5.68x`, p90 `6.03x`; point parity fails in every repeat |
| Same-GPU 4x4 early state distribution | state audit `13078873`; analyzer `ac79e3348` | candidate 1--2 first leave native particle support @58 and candidates 3--4 @104; strict schedule first fails @104 for candidates 1/4 and @71 for 2/3. At @4 all candidates match native modes 1/3/4 but none covers native mode 2, so reverse coverage remains red |
| Iteration-58 particle-1021 score boundary | exact-H100 capture `13078467`; analyzer `f4305b23f` | RECOVAR repeats split 2--2 between two orientations. The captured RELION-1/3/4 mode beats the RECOVAR-1/2 mode by `0.0001220703125`: raw score `+0.4185452461`, rotation prior `-0.4184231758`, translation prior `0`. This rejects an exact-tie/tie-break cause and localizes the live discriminator to near-cancelled fine scoring |
| Independent native/candidate particle-1021 score boundary | exact-H100 capture `13078944`; analyzer `306a2a7a5` | RECOVAR repeats the same winner with half the prior margin, `0.00006103515625`; native RELION's same-mode margin is `0.0059356689453125`. Candidate-minus-native gap components are raw score `-0.003300667`, rotation prior `-0.002573967`, translation prior `0`, proving the inherited boundary is not fine-score arithmetic alone |
| Twenty-repeat iteration-4 candidate panel | exact-H100 job `13078176`; analysis `ef8fd3dfc`; all 20 runs complete | every candidate is valid and matches native repeats 1/3/4, but all 20 miss native repeat 2 by the same one particle. Reverse coverage fails `0/20`; particle `1085@particles.128.mrcs` differs by `116.39 deg / 8.5 A`. Systematic early mode bias supersedes the sampling-insufficiency theory |
| Four-by-four map/state distribution | CPU audit `13077225`; analyzer `ac79e3348` | terminal scientific fail, not an infrastructure crash: map native-radius first fails @1 and schedule reverse coverage first fails @4; every candidate misses native repeat-2's early mode while later candidate-validity failures occur @71/@104 |
| Iteration-4 particle-1085 cutoff capture | exact-H100 paired capture `13079981`; native repeat-2 optimiser resumed from iteration 3 | both engines contain the same 16,704 hypotheses and top key `[556,14]`; native retains rank-10 key `[175,8]` because top-9 cumulative mass is `0.9989997162`, while RECOVAR prunes it at `0.9990066439`. The omitted parent seeds the `116.39 deg / 8.5 A` fine branch |
| Native/RECOVAR PPref coarse replay | exact-H100 job `13080743`; analyzer `51ff050eb`; setup-only attempt `13080653` had no science | native PPref reduces centered global score max from `0.0383606` to `0.00016785`; RECOVAR PPref reproduces production within `0.0000610`. On the exact winner/cutoff pair, native PPref replay misses native by only `0.0000610`, while RECOVAR PPref is exact to production |
| Serialized-map to PPref boundary | analyzer `1fd5617d4`; 10/10 focused map/projector tests; report SHA `93f6035fefc2...` | native serialized-map rebuild is bit-exact over all 26,011 PPref cells; RECOVAR rebuild closes at `5.107e-9` relative L2; cross-engine PPref is `5.774e-5`, a `11,306x` separation. First open state is the serialized iteration-3 map at `6.292e-5` relative L2 |
| Live native-PPref E-step intervention | exact-H100 job `13081790`; science `3c896257b`; analyzer `d83d1df7f`; report SHA `7240ff9d866d...` | replacing only iteration-4 PPref restores 10 coarse parents including causal index `5083`, preserves winner `16138`, and restores native repeat-2 particle state within `2.49e-6 deg / 1.0e-6 A`; untreated control is `116.39 deg / 8.5 A` away. Incoming PPref/map state is causally sufficient; E-step implementation is exonerated |
| Four-by-four direct map relative-L2 panel | CPU jobs `13082447--13082450`; analyzers `a8e43bb55 / 120f044d5`; aggregate SHA `058289d6e6e6...` | all 804 candidate checkpoints are inside the native four-repeat diameter; worst candidate/native-envelope ratio is `0.60616` at iteration 1. Candidate repeat spread is only `0.21659x / 0.003857x / 0.000721x` native at iterations 1/2/4; nearest-native coverage contracts from repeats 1/3 at iteration 1 to repeat 3 only at iteration 4. Magnitude is statistically green, reverse mode coverage is not |
| Exact embedded RELION PTX discriminator | science `e8253c02c`; exact-H100 job `13083892` completed two seed-0 arms through iteration 4 in 126 s; M-step report SHA `56ef5db37763...`; PTX/binary SHA `ff2db0da734b... / 2b7646cb708d...` | ABI/path is validated, but exact native device arithmetic does **not** restore mode coverage. Candidate map spread is `1.32750e-7`, only `0.000643975x` the four-repeat native diameter `2.06142e-4`; both candidates are nearest native repeat 3 and each differs from native repeat 2 by the same one particle. Exact PTX/kernel arithmetic is rejected as sufficient; host/process CUDA execution topology is next |
| CUDA context + fresh-process discriminator | context job `13084397`; clean-process science `c0c44dbe7`, exact-H100 job `13084892`, report SHA `49f7e9a47a99...` | runtime device flags, primary-context state, stream flags/priorities, and priority range all match. Two fresh native CUDA processes still give only `0.001359x` native map spread and preserve mismatch vector `[0,1,0,0]`; ordinary process/context history is rejected as sufficient |
| Exact same-stream Wavg predecessor | science `e5e969f30`; smoke `13086035`; exact-H100 trajectory `13086085` completed two arms through iteration 4 in 142 s; M-step report SHA `71360285e7e7...`; binary SHA `2baafe8c239a...` | exact RELION `Wavg<REFCTF,REF3D,!DATA3D,256>` is queued immediately before exact BPref on each worker stream. Candidate spread remains `2.41166e-7`, only `0.00116990x` the native diameter; arms are nearest native repeats 1 and 3 and both preserve mismatch vector `[0,1,0,0]`. Same-stream Wavg scheduling is rejected as sufficient |
| Native Wavg/BPref host interval | RELION source `d50fcb8`; qualified binary SHA `83577092ed05...`; exact-H100 continuation `13086700` targets repeat-2 iteration-4 particle 1085 | Wavg enqueue is `4,082 ns`, Wavg return to BPref call entry is `233 ns`, BPref setup/enqueue is `3,613 ns`, and the launch runs on host worker 4. These intervals are passive diagnostics, not a correction |
| Candidate pre-BPref interval | plumbing fix `08359b2b4`; exact-H100 `13087189` completed 0--4 in 72 s; setup-only `13087180` ran no science | target traces by iteration are intrinsic `193/182/209/178 ns` and pre-launch effective `292/230/263/216 ns`. The trace ends at candidate `cuLaunchKernel` entry while RELION's `233 ns` ends at wrapper entry, so the values are not directly comparable; complete enqueue/return columns are required |
| Exact 233 ns delay panel | exact-H100 `13087232` completed both 0--4 arms in 141 s | no promotion: candidate repeat distance is `1.09946e-7` while native repeat distance is `6.27760e-5`; paired cross/native ratios are `0.009502 / 0.999882`. The delay does not restore the alternate native basin and is rejected |
| Exact iteration-4 worker-owner replay | exact-H100 `13087339`; arm A has a terminal per-arm seal, arm B has a complete boundary report but no terminal marker and is excluded | sealed arm A captures and replays its own native owners yet remains `6.27832e-5` from the alternate native reference. The target owner varies across arms (worker 1 versus 2) without tracking the native branch. Exact owner identity is rejected as sufficient at the causal iteration-4 boundary |
| Complete launch-interval trace | local science `2ac3dc735`; focused host-replay guards `7/7`, Ruff and diff checks clean | adds candidate Wavg enqueue, pre-BPref intrinsic/effective gap, BPref enqueue, and Wavg-return-to-BPref-return columns. CUDA qualification and exact-H100 measurement are the active gate; no frozen score impact |
| Complete exact-PTX launch intervals | exact-H100 `13087597` completed 0--4 in 71 s; binary SHA `b3b3bdbd2dca...` | candidate iteration-4 Wavg/BPref enqueue is `3,102 / 7,802 ns` and Wavg-return-to-BPref-return is `8,131 ns`, versus native `4,082 / 3,613 / 3,846 ns`. The dynamic Driver API has a real iteration-4 host-latency spike |
| Registered Runtime BPref launch | local science `b60152741`; exact-H100 trace `13087838` and two-arm panel `13087881`; binary SHA `41a2fe4d31c2...`; focused guards `8/8` | Runtime BPref removes the host spike (`2,828 ns`, total `3,072 ns`) but does not restore native mode coverage. Candidate repeat map distance is `7.38888e-8` versus native `6.27404e-5`; paired ratios are `0.005026 / 1.000159`. Launch API affects timing but is rejected as the scientific cause |
| Full native BPref operand/prestate capture | RELION source `56e94b3`; build `13088114`; repeat-2 continuation `13088385`; binary SHA `7ce5e00a2c20...` | all 192 target files exist, including six immutable raw operand arrays, 13 launch scalars, and real/imag/weight accumulator prestate. The job was intentionally cancelled after iteration 4 had completed instead of wasting the GPU through iteration 200; the capture is diagnostic and non-scoring |
| Matched fixed-posterior native/replay panel | exact-H100 job `13095568`; science `85cd685f3`; report SHA `6eca689fc7b5...` | four native arms and four private/shared replay arms ran sequentially in one allocation on one physical H100. Fixed replay widths were only `0.0139--0.0321x` native for private and `0.126--0.338x` for shared, showing that frozen operands under-represent the native distribution but not identifying why |
| Matched native-posterior replay panel | exact-H100 job `13097692`; science `eeeceb368`; report/evidence SHA `901255a8a74a... / c0657f213e4b...` | all 200 native posterior panels were replayed fail-closed with zero support, Euler, or launch-count mismatches. Private/native BPref diameter ratios are `1.0367 / 0.8222 / 1.0076 / 0.8097`; shared/native ratios are `1.0384 / 0.8249 / 1.0739 / 0.8288` for data h0/h1 and weight h0/h1. Native posterior realization explains the missing width; upstream workload injection is retired |
| Matched live-posterior repeat panel | exact-H100 jobs `13100126 / 13101940`; analysis/science `4d6c88e53 / 90bf177df`; sealed report/evidence SHA `4468fefe7fa4... / 568c74f5eadd...` | native posterior diameter is `1.72558e-5` and centered-score RMS is `3.19258e-5`; all four live RECOVAR posterior and score panels are byte-identical with zero support drift. Enabling EM's source-faithful native-atomic soft-mask reduction leaves both widths exactly zero, rejecting preprocessing atomic order as sufficient and localizing the next fixed-state audit to the shared fine-score input/reduction boundary |
| Paired candidate host-input capture | science `9b26c2ae7`; exact-H100 `13088492` completed in 70 s; CUDA SHA `41a2fe4d31c2...` | 14/14 callback inputs are preserved with stable original stack IDs. Particle 1085 resolves uniquely to callback 12 row 96. Native/candidate launch geometry is `8 / 128` orientation blocks, and native prestate is worker-local while the current candidate inputs expose only two shared half accumulators. Exact candidate prelaunch state and iteration-1 aligned-state diffs remain open |
| Exact-GPU setup misses, non-scoring | `13088398 / 13088453` | both exited fail-closed at time zero with code 75 after receiving `GPU-9f98...` instead of frozen `GPU-75c2...`; no science ran. Slot holder `13088479` selected the non-target GPU for 45 s, allowing `13088492` to start on the target, then was cancelled |
| Native-radius map diagnostic | analysis `ac79e334`; focused job `13077189` passes 13/13; real-data audit `13077282` | candidate support passes 200/201 checkpoints; only @1 misses by `8.75e-13`. At @200, candidate/reverse native-radius margins are `+0.1991 / +0.0248`; remaining combined failures are reverse coverage and paired GT nondegradation, not candidate map support |
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
| GF47 materialized order + native trace epilogue | local `374b24b7f`; H100 gate `13038159` passes 29/29; exact-GPU panel `13038186` and full science `13038307` complete | short reference passes at **0.978x / 0.967x** and all raw accumulators pass; audit `13040047` first fails schedule @58, particle @61, map @80 |

</details>

> **Status: draft, not merge-ready.** K=1 correctness is the active gate.
> Runtime optimization starts from a sealed passing trajectory; K>1,
> real-data, and final CLI/GUI qualification follow K=1 closure.

#### Frozen K=1 case matrix

This mirrors the supplied-map EM scorecard: every fixed case stays visible,
including failures. A checked row passes map, particle-state, and
pre-divergence schedule gates; runtime remains open for every row.

| Done | Case | Seed | Distribution / stress | Map | Particle | Schedule | Runtime | Overall |
|---|---|---:|---|---|---|---|---:|---|
| [ ] | GF43 | 29 | baseline, uniform, white noise | fail @146 | pass | pass | 7.88x | **FAIL: map/runtime** |
| [x] | GF44 | 29 | anisotropic, outliers, high noise | pass | pass | pass | 8.03x | **QUALITY PASS; runtime open** |
| [x] | GF45 | 29 | Kent, outliers, high noise | pass | pass | pass | 9.33x | **QUALITY PASS; runtime open** |
| [ ] | GF46 | 29 | anisotropic, severe outliers, radial/high noise | fail @20 | fail @4 | pass | 9.40x | **FAIL** |
| [ ] | GF47 | 29 | extreme outliers, uniform, white noise | pass | pass | fail @10 | 6.42x | **FAIL: controller/runtime** |
| [ ] | GF48 | 29 | very-high noise, uniform, white noise | fail @45 | fail @30 | fail @10 | 6.84x | **FAIL** |
| [ ] | GF49 | 29 | low noise, uniform | fail @115 | pass | pass | 11.07x | **FAIL: map/runtime** |
| [ ] | GF50 | 29 | low noise, Kent | fail @41 | fail @40 | pass | 11.58x | **FAIL** |
| [ ] | GF51 | 29 | no CTF, radial noise | fail @74 | fail @39 | pass | 5.89x | **FAIL** |
| [ ] | GF52 | 29 | Kent, junk particles, translations | fail @40 | fail @40 | fail @40 | 8.20x | **FAIL** |
| [ ] | GF53 | 29 | high resolution, radial noise | fail @44 | fail @40 | pass | 4.91x | **FAIL** |
| [ ] | GF54 | 29 | midscale, Kent, radial noise | fail @45 | fail @30 | pass | 7.81x | **FAIL** |
| [ ] | GF55 | 101 | anisotropic, outliers, high noise | fail @46 | fail @40 | pass | 7.98x | **FAIL** |
| [ ] | GF56 | 101 | Kent, outliers, high noise | fail @45 | fail @29 | fail @30 | 6.92x | **FAIL** |
| [ ] | GF57 | 101 | anisotropic, severe outliers, radial/high noise | fail @44 | fail @11 | pass | 10.40x | **FAIL** |
| [ ] | GF58 | 101 | extreme outliers, uniform, white noise | fail @94 | fail @48 | pass | 6.60x | **FAIL** |
| [ ] | GF59 | 101 | very-high noise, uniform, white noise | pass | fail @30 | pass | 6.86x | **FAIL: particle/runtime** |
| [ ] | GF60 | 101 | low noise, uniform | fail @42 | fail @40 | fail @20 | 8.73x | **FAIL** |
| [ ] | GF61 | 101 | low noise, Kent | fail @41 | fail @40 | fail @40 | 6.40x | **FAIL** |
| [ ] | GF62 | 101 | Kent, junk particles, translations | pass | pass | fail @20 | 7.21x | **FAIL: controller/runtime** |

Last scientific update: **2026-08-28 18:34 ET**

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
| 1 | GF46 raw-BPref feedback boundary | launch synchronization closes two repeats through @20 but fails the long gate: first particle and optimal-offset statistic @33, accuracy @40, grid @52, map @57. The residual grows from one switched particle to 178, so global synchronization is rejected as sufficient | @4 `13116158 / 13116369`; @20 `13116813 / 13117293`; failing 0--200 `13117709`; queued @33 capture `13118385`; current science `23150e50c` | implement a shared repeat-robust BPref accumulator, require native-envelope raw buffers and exact @4/@20/@33/@40/@52/@57 gates, then rerun 0--200; no score change |
| 2 | GF47 systematic mode boundary @4, particle 1085 | exact E-step/operand/support chain, terminal 4x8 audit, and native-posterior replay close iteration-1 raw-BPref distribution scale. Eight native repeats choose particle modes 5:3 while four candidates choose 4:0; the decisive boundary is the rank-10 adaptive-support parent inherited through iteration-3 map/PPref | native expansion `13091586--13091618`; 4x8 audit `13092340`; matched fixed-posterior/native-posterior panels `13095568 / 13097692`; replay science `eeeceb368` | reuse the same corrected shared coarse path, then qualify 0--4/0--20 before 0--200 |
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

Exact-H100 replay `13105728` closes GF46's first clean candidate-support
departure. Shared diagnostic commit `f2a041d9d` captures the 128 pre-atomic
lanes from the same fused projector kernel used in production; it does not
reimplement the scorer. With 29 translations, four lanes contribute to each
candidate. The rank-100 parent has one legal float32 total. The rank-101 parent
has exactly two totals, `9.489091873` and `9.489092827`, separated by one ULP.
RECOVAR's production outcome is the lower total and RELION's observed
rank-100/rank-101 gap is exactly the higher legal outcome. Thus the formula,
geometry, projector, priors, and posterior rule are all closed at this boundary.

Applying a fixed lane order to the captured lanes, then preserving RELION's
absolute `min_diff2` score frame, reproduces the complete native threshold mask:
100 significant candidates, zero extras, zero omissions, and zero mismatches
across all 16,704 candidates. Order `(0,2,1,3)` reproduces the current extra
rank-101 parent; canonical `(0,1,2,3)` and two other legal orders reproduce the
native mask exactly. The next change must therefore live in the shared fused
coarse scorer/posterior path: deterministic canonical reduction composed with
the already tested min-diff frame. It remains diagnostic until GF46 passes
0--4 and 0--20 and then the unchanged frozen matrix. Report/NPZ SHA values are
`d95bcc46d927... / 9a13139e72a4...`; no score or tolerance changed.

The shared-code audit is also explicit now. Commit `21ac2c872` adds an object-
identity invariant showing that InitialModel directly imports EM's
significance, sparse adaptive pass-2, local K-class refinement, and hypothesis
layout implementations. The focused identity test passes. The broader
refactor-invariant file still reports existing size-budget debt in
`dense_adapter.py`, `m_step.py`, and the package total plus a loaded-login-node
cold-import timeout; those are maintenance failures, not duplicate scorer
implementations, and their limits were not raised.

Matched exact-H100 job `13097692` closes the iteration-1 raw-BPref width
question. Four native RELION arms and four private/shared replay arms ran
sequentially in one allocation on physical GPU `GPU-1fdb3b99...`. Each replay
used its corresponding repeat's actual native posterior for all **200**
particles. Validation found zero support-coordinate, Euler, or launch-count
mismatches. The pooled normalized native posterior itself varies at
`1.40e-5--1.62e-5` relative L2; per-particle maximum pairwise variation has
median `1.90e-5` and maximum `5.89e-5`.

That posterior variation restores the complete raw-BPref distribution scale.
Private/native maximum-diameter ratios are
`1.0367 / 0.8222 / 1.0076 / 0.8097`, and shared/native ratios are
`1.0384 / 0.8249 / 1.0739 / 0.8288`, ordered as data half 0/1 and weight half
0/1. Report SHA is `901255a8a74a...`; the complete evidence manifest is
`c0657f213e4b...`. Science commit `eeeceb368` adds the fail-closed posterior
mapping and matched harness. Its focused validation is **18/18**, with Ruff,
shell syntax, and diff checks clean.

The fixed-posterior matched panel `13095568` remains an important negative
control: its private widths were only `0.0139--0.0321x` native and shared
widths `0.126--0.338x`. Diagnostic job `13096599` is excluded from the
posterior-causality claim because its first implementation remapped native
Euler/support topology but accidentally retained RECOVAR posterior weights;
that mistake was corrected and guarded in `eeeceb368`. The combined evidence
means no artificial delay, upstream workload injection, or stochastic repair
is justified. Native posterior realization—not omitted GPU work—caused the
previously missing replay width.

The active boundary therefore returns to the systematic iteration-4 mode
bias. The rectangular audit `13092340` remains terminal: all four candidate
maps are inside the eight-native direct map envelope at **201/201** checkpoints,
but candidate repeats sample only the common iteration-4 mode. For particle
1085, RELION retains rank-10 coarse parent `[175,8]` at cumulative mass
`0.9989997162`, while RECOVAR prunes it at `0.9990066439`. Live native-PPref
intervention `13081790` restores that parent and the alternate state, so the
downstream E-step is exonerated. The next implementation work explicitly
reuses EM's source-ordered coarse scorer, float32 posterior normalization,
adaptive-support cutoff, and fused M-step primitives instead of adding
VDAM-only duplicate math.

Complete iteration-1 native BPref capture `13090875` seals all **200**
particles, not only the causal target. Every active candidate Euler/support row
maps bit-exactly into its native panel. Native launch widths span 8--744 rows;
the candidate buckets preserve all active mass after replaying native owners,
counts, Euler rows, and physical panels. Worker-private replay `13090954`,
global launch-order replay `13091237`, and exact Wavg predecessor `13091268`
do not improve the raw boundary.

The unconfounded shared-accumulator arm `13091437` is decisive. Relative to
the production shared result, exact native geometry changes data residuals by
only `1.000593x / 1.000173x` and weight residuals by
`1.004817x / 1.001302x` in halves 0/1. That scale is below independent
RELION-to-RELION accelerated-GPU variation. The deterministic E-step,
operand/support, BPref formula, and static topology boundary is therefore
closed; no production change is promoted from these diagnostic arms.

Four additional exact-H100 RELION trajectories (`13091586 / 13091616 /
13091617 / 13091618`) then make the open failure concrete. At iteration 4,
particle `1085@particles.128.mrcs` occupies the common state in native repeats
1/3/4/6/7 and the alternate state in repeats 2/5/8. All four RECOVAR
candidates occupy the common state. An interim direct state check gives zero
candidate-to-eight-native mismatches at iteration 4, but reverse native mode
coverage still fails; candidate repeats 1/2 again leave the expanded particle
support at iteration 58. The sealed rectangular audit is job `13092340` from
analysis commit `dfb0b1cfa`; it is non-scoring while live.

The exact replay repeatability discriminator is now complete. Its initial
fixed-posterior arms correctly showed shared accumulation wider than private,
but the matched native panel proved that both controls were artificially
narrow because each repeat reused one frozen posterior. Replaying each native
posterior restores both private and shared widths to the native range. The
worker-private arm remains a topology control; neither shared topology nor
upstream device workload is promoted as a repair. Artificial noise, seed
tuning, workload injection, and gate widening remain out of scope.

Complete exact-PTX trace `13087597` resolves the interval mismatch.  At
iteration 4, RECOVAR's Wavg enqueue is `3,102 ns`, its dynamic Driver API
BPref call is `7,802 ns`, and Wavg-return-to-BPref-return is `8,131 ns`.
RELION's corresponding values are `4,082 / 3,613 / 3,846 ns`.  Registered
Runtime-launch discriminator `b60152741` removes that host spike: exact-H100
trace `13087838` measures `2,828 ns` for BPref and `3,072 ns` for the complete
post-Wavg interval.

The controlled two-arm science result rejects launch API as the missing
scientific cause.  Runtime panel `13087881` completes both arms in 141 seconds.
RECOVAR maps are only `7.38888e-8` apart while native maps span
`6.27404e-5`; the paired cross/native ratios are `0.005026 / 1.000159`.
Thus Runtime versus Driver launching materially changes host latency but does
not make RECOVAR sample the alternate native mode.  The next fail-closed gate
byte-seals every immutable BPref launch operand, the accumulator state visible
at enqueue, and stream/event dependencies at iteration 4 and then iteration 1.

RELION launch-gap job `13086700` and candidate trace job `13087189` close the
simple host-delay question without changing the frozen score.  RELION repeat 2
measures `4,082 / 233 / 3,613 ns` for Wavg enqueue, Wavg-return-to-BPref-call,
and BPref setup/enqueue.  RECOVAR's first trace measures only to the lower-level
driver-call entry (`193 / 182 / 209 / 178 ns` intrinsically over iterations
1--4), so the apparent `178 < 233 ns` comparison is not definitionally valid.
Local science `2ac3dc735` now records matching complete enqueue and return
intervals.  Its focused host-replay guards pass `7/7`; Ruff and diff checks are
clean.

The intervention result is nevertheless negative.  Exact-H100 panel
`13087232` replays a 233 ns pre-launch interval in both arms.  RECOVAR repeats
remain only `1.09946e-7` apart while paired native repeats span `6.27760e-5`;
the candidates stay in the common native-1/3/4 basin and do not follow the
alternate native-2 basin.  Per-arm captured worker replay `13087339/a` is also
negative at `6.27832e-5` reference relative L2 even though its own native
worker assignment is replayed exactly.  Arm B wrote a complete boundary report
but lacks the required terminal marker and is excluded from sealed evidence.
Simple host delay and worker identity are therefore rejected as sufficient.

Exact-H100 job `13086085` completed two seed-0 iteration-4 arms in 142 seconds
from science head `e5e969f30`. The opt-in discriminator loads both RELION's
exact 27-argument VDAM SGD entry and its exact CTF-corrected 3D-reference,
2D-data Wavg entry from the embedded PTX. It queues Wavg immediately before
BPref on the same worker stream with no intervening synchronization, matching
RELION's concrete launch chain. Smoke job `13086035` validates the `sm_90`
wrapper and exact-PTX path. The PTX and qualified CUDA artifact SHA-256 values
are `ff2db0da734b...` and `2baafe8c239a...`.

The result rejects the same-stream Wavg predecessor as sufficient. Candidate
maps are `2.411662e-7` apart, only **`0.001169902x`** the four-native-repeat
diameter `2.061421e-4`. The two arms are nearest native repeats 1 and 3, but both
still differ from native repeat 2 only at particle 1085, giving mismatch vector
`[0,1,0,0]`. The sealed M-step panel is
`analysis/mstep_repeat_panel.json` with SHA-256 `71360285e7e7...`.

This follows two other negative host-layer discriminators. Context job
`13084397` found matching runtime device flags, primary-context state, stream
flags/priorities, and priority range. Clean-process exact-PTX job `13084892`
gave only **`0.001359x`** native map spread and the same `[0,1,0,0]` state
vector. Ordinary context/process history and the known Wavg scheduling edge are
therefore closed. The next bounded discriminator seals all per-particle launch
operands and event/synchronization edges on both implementations, then replays
only the first actual difference. No 0--20 or 0--200 qualification is warranted
until a focused 0--4 intervention restores native-repeat-2 coverage.

The preceding CPU jobs `13082447--13082450` completed four trajectory-wide
direct-map analyses from analyzer head `d83d1df7f`; aggregate analyzer
`120f044d5` measured both candidate-to-native distance and candidate-repeat
spread without alignment or scale fitting. All **804/804** candidate checkpoints
are inside the four-repeat native relative-L2 diameter. The worst ratio is
`0.606156` at iteration 1, so a gross map-error magnitude is rejected.

The repeat distribution is nevertheless under-dispersed. At iteration 1 the
maximum candidate-candidate distance is only `0.216588x` the native diameter,
and nearest-native coverage contains repeats 1 and 3 but misses 2 and 4. The
spread ratio falls to `0.003857x` at iteration 2 and `0.000721x` at iteration
4, where every candidate is nearest native repeat 3. Aggregate report
`vdam_gf47_map_l2_4x4_d83d1df7f_20260828/panel_120f044d5.json` has SHA
`058289d6e6e6...`; its four input report SHAs and all source roots are sealed.
Focused map/intervention validation passes **13/13**, followed by **4/4**
spread-specific tests after the extension; no generic RECOVAR suite ran. The
active target is now native accumulation distribution width and reverse mode
coverage, not lower direct relative-L2 error.

### Live causal intervention

Exact-H100 job `13081790` completed the live iteration-4 causal intervention
from science commit `3c896257b`. RECOVAR owned iterations 1--3 and every other
operation; only the incoming iteration-4 PPref was replayed from RELION repeat
2. The intervention restores all 10 coarse parents, including causal cutoff
parent `5083`, and preserves winner `16138`. Particle
`1085@particles.128.mrcs` moves from the untreated control mode
(`116.3878 deg / 8.499999 A` from native repeat 2) to the native-2 state within
`2.4855e-6 deg / 9.999999e-7 A`; Pmax differs by `7e-6` at STAR precision.

Fail-closed analyzer `d83d1df7f` binds the result to job `13081790`, physical
GPU `GPU-75c2d200...`, science commit, replay NPZ SHA `9075af389a7b...`, CUDA
SHA `48f17d85d197...`, RELION binding SHA `77ac98f16ae9...`, source STARs,
capture NPZ, and runner log. Its report is
`vdam_gf47_it4_native_ppref_live_3c896257b_20260828/ppref_live_intervention_d83d1df7f.json`
with SHA `7240ff9d866d...`. Four focused analyzer tests pass; no generic RECOVAR
suite ran. This closes the E-step response. The matched posterior panel later
closed raw-BPref width as a missing-distribution cause, leaving the
source-faithful coarse-score/normalization/cutoff path as the active K=1
implementation boundary.

### Earlier production-accumulator qualification

The production iteration-58 raw-accumulator boundary now passes an independent
native-repeat gate. Science commit `f5e7d74ad` adds a fail-closed multi-panel
summarizer and focused tests. Same-H100 jobs `13072500--13072503` completed
four panels (eight fresh arms) in 14:13--15:05. All **32/32** raw data/weight
comparisons satisfy `cross-engine relative L2 <= native-repeat relative L2`;
the ratio distribution is minimum `1.699e-5`, median `2.028e-3`, p90
`1.161e-2`, and maximum `1.283e-2`. The four panel-report SHA-256 values are
`0ee17564f5e6...`, `79935bac4c2b...`, `defb77ff9737...`, and
`f2415f9c6910...`; aggregate report SHA-256 is `9d8f48f54e43...` under
`vdam_gf47_it58_production_repeat_panel_92214d9ec_20260827/analysis/`.

This rejects further static atomic-order or SGD instruction-shape imitation as
the next justified experiment. It does not promote the frozen score: the next
gate is repeated production 0--200 trajectories, comparing cross-engine
schedule, particle, map, and runtime distributions against RELION repeats.
Focused validation is **9/9** repeat-panel tests plus Ruff and Python compile;
no generic RECOVAR full/long suite ran.

The iteration-58 failure is now known to move between native realizations.
Local science commit `834b78c54` fixes the BPref boundary analyzer so a target
iteration reads the matching `run_itNNN_*` native accumulators instead of the
iteration-1 filenames. Focused validation passes 10 analyzer tests plus the
full-schedule merge guard; no generic RECOVAR suite ran. Exact-H100 task
`13058221` then completed a fresh paired iteration-58 boundary from the same
seed, fixture, physical GPU, CUDA binary, and RELION binding.

The paired candidate matches fresh RELION for 2,999 of 3,000 particles. The
two previously identified failures, stack indices 2411 and 2707, now have
identical poses and translations; candidate/native Pmax differs by only
`2.35e-4` and `3e-6`. The sole state divergence is instead
`46@particles.128.mrcs`, with `31.17` degrees and `3.354` Angstrom error.
Fresh RELION versus frozen native repeats has 4, 460, 503, and 5 divergent
particles, respectively. This rejects a fixed particle-specific formula or
tie-break correction and exposes a moving atomic-order boundary.

The corresponding aggregate raw-BPref relative-L2 errors are
`3.1698e-4 / 6.0430e-4` for data and `1.1769e-4 / 2.3361e-4` for weight.
The contribution bundle captures 168 active rows for original indices 2410
and 2706. Report SHA-256 values are `b6d2545050ea...` for the BPref boundary,
`adea186bce19...` for the M-step, `502f5b878f1f...` for paired particle state,
and `9f48b2e8758f...` for the native-repeat envelope. Local science commit
`e4b53c26a` adds immutable stack-index targeting to the full-schedule runner;
exact-H100 job `13059422` is the stable-identity native StoreWavg capture.
It completed as task `13059422_1` (internal job `13059423`) in 390 seconds and
resolved immutable stack index 2707 to run-local part 1898. Candidate-to-native
geometry is closed: all 128 RECOVAR rotations exist exactly in RELION; eight
additional native rotations carry exactly zero posterior; and all 116
translations map one-to-one in the same order after the expected `-2*pi/128`
phase conversion (maximum error `1.49e-8`). Posterior-only replay `13060135`
then completed on the same H100.

That pair cannot support a causal posterior conclusion because this fresh
RELION realization landed in the distant native mode: its paired trajectory
has 462 divergent particles at iteration 58, matching the scale of frozen
repeat 2 rather than the near-parity `13058221` realization. Local science
commits `fe8524e79` and `3f88bed7c` make zero-mass native rotation padding
explicit and add a fail-closed posterior-only replay.

Exact-H100 task `13060276_2` (internal job `13060278`) then completed a
state-aligned trajectory in 394 seconds by replaying its own paired native
references while leaving RECOVAR posterior and BPref live. At iteration 58 all
3,000 particle poses and translations match; maximum errors are `9.96e-6`
degrees and `7.07e-6` Angstrom, and Pmax p95 is `1.8e-5`. Posterior replay
`13060652` finds 174 positive cells in both engines with zero support
mismatches. The full posterior residual is only `1.3945e-5` relative L2,
`1.7704e-5` L1, and `4.1425e-6` maximum absolute error; RECOVAR's independent
RELION-f32 replay is identical to its live posterior. Particle and posterior
report SHA-256 values are `f1aac0526a88...` and `b2a6d003a30d...`.

Local science commit `9390617e7` adds fail-closed loading of RELION's exact
StoreWavg inverse-noise operand. Exact-H100 task `13061856_2` (internal job
`13061877`) then repeats the reference-aligned iteration-58 boundary with
large production score operands enabled. All 3,000 particle states match;
stack 2707 has 155 positive cells in both engines with zero support mismatch
and a live-posterior residual of `1.7985e-5` relative L2.

Split analyzer `13062288` closes the translated `Fimg * CTF / sigma2` operand
at `1.0576e-7`, `CTF^2 / sigma2` at `7.1656e-8`, and the same-posterior
per-particle BPref volume after RELION scatter at `1.0488e-7` relative L2.
Using each engine's live posterior changes the particle volume by only
`1.6505e-5`. The operand and state report SHA-256 values are
`3767d6c5ccc7...` and `d042e893e6c2...`.

This closes immutable identity, rotation/translation topology, retained
support, posterior normalization, unmasked image preprocessing, translation
phase, CTF/noise weighting, and effective per-particle row generation as the
aggregate raw-BPref cause. The leading open boundary is multi-particle CUDA
scatter/atomic arrival order. The next bounded capture compares RECOVAR's
inline particle accumulator to a zeroed isolated native CUDA
`AccBackprojector`; the score remains **2/20**.

The inherited reference-map residue is closed to compiler last bits. RELION's
`updateStepSize()` stores `x`, `a`, `b`, and the sigmoid `scale` as 32-bit
`float`; RECOVAR had preserved the sigmoid scale in binary64. Local science
commit `7324440e2` reproduces the native rounding boundary in the Python
schedule and RELION parity helper. Focused Slurm job `13054781` passes
**101/101** relevant schedule/binding tests; two unrelated pre-existing LOC
budget checks were deliberately excluded, and no generic RECOVAR suite ran.

On the exact physical H100, iteration-1 task `13054825_2` makes every captured
input bitwise exact and reduces reconstructed-reference relative L2 from
`6.238e-9` to `3.098e-16`. The trajectory-wide raw-BPref oracle then completes
60 iterations in task `13054923_2` (internal job `13054925`, 407 seconds).
At iteration 60 every raw/downstream M-step operand is bitwise exact; the
incoming and outgoing references are only `2.527e-15` and `2.460e-15` from
paired RELION. Sampled FSC-AUC and assignment accuracy are both **`1.0`**,
particle audit finds zero pose/translation divergences through all 60
iterations, and every operative schedule field matches. Only inactive
accuracy placeholders differ from iteration 1. M-step, map, particle, and
sampling SHA-256 values are `e8bcf79e5378...`, `a177dfe30d8a...`,
`fa8af2a7aa58...`, and `5171c389de7a...`. This is a causal oracle rather than
a frozen scoring run, so the score remains **2/20**.

The corresponding no-oracle production task `13056615_2` (internal job
`13056622`) completed 60 iterations in 402 seconds. Sampled maps still pass
through iteration 60 at minimum FSC-AUC `0.999997662`, but paired particle
state first splits at iteration 58 for particles 2411 and 2707. Operative
`optimal_offset_change` splits at the same checkpoint and `offset_range`
follows at iteration 60. By iteration 60 the two raw-BPref data halves have
amplified to `8.404e-2 / 7.817e-3` relative L2 and the output reference to
`1.894e-3`; the raw-replay arm makes those operands exact and the reference
`2.460e-15`. Production M-step, map, particle, and sampling SHA-256 values are
`f249dad471d7...`, `53c1c09177f4...`, `bf1a4803134b...`, and
`0ddaeb5e1b8a...`. This leaves the already localized CUDA raw-accumulation
order at the iteration-58 close-score pair as the first production boundary;
a fresh 0--200 scoring qualification waits for that repair.

Before the float-schedule fix, raw BPref was sufficient to close the entire
downstream M-step state, but a smaller inherited reference-map residue
remained. Local science commit
`9459a205b` adds fail-closed complex-data and real-weight replay for both
pseudo-halfsets. Exact-H100 task `13052694_1` (internal job `13052695`)
completed 60 live-map iterations in 430 seconds on `GPU-6222...`; neither
first/second moments nor references were replayed.

At iteration 60, raw BPref data and weights, post-reweight data, incoming and
post-update first moments, incoming and post-update second moment,
`mom1_noise_power`, and post-`applyMomenta` data are all bitwise exact. The
only non-exact M-step stage is the reference entering reconstruction at
`1.635e-8` relative L2; reconstruction preserves rather than amplifies it at
`1.613e-8`. Sampled checkpoints through iteration 60 pass at minimum FSC-AUC
**`0.999999999972`** and class-assignment accuracy `1.0`.

One paired 1.5-Angstrom translation still departs at iteration 34, with
`optimal_offset_change` changing at the same checkpoint. A sampled comparison
against the original four native repeats accepts the candidate particle state
through iteration 44 and finds two unmatched active particles at iterations
58 and 60. Crucially, the freshly paired RELION trajectory itself has the
same two unmatched identities and identical per-repeat mismatch counts at
both checkpoints; its schedule also enters a mode absent from the old panel.
The old four-repeat envelope therefore does not span current native
nondeterminism and cannot turn this diagnostic into a score change. The next
bounded capture is iteration 1, which will split the `1.6e-8` residue between
initialization/frame conversion and recurrent `reconstructGrad` arithmetic.
M-step, sampled-map, particle, sampling, and sampled-envelope SHA-256 values
are `9b7e6d7a2fb2...`, `33a3a8073e57...`, `43040f4c040b...`,
`10e6c662bafc...`, and `aafc6544cc8f...`. The score remains **2/20**.

The complete gradient-moment trajectory is now closed, leaving raw BPref as
the first non-exact causal boundary. Local science commit `e7f30119a` adds a
fail-closed paired native `Igrad1` replay for both pseudo-halfsets and composes
it with the already qualified `Igrad2` replay. Exact-H100 array task
`13051779_1` (internal Slurm job `13051846`) completed 60 live-map iterations
in 427 seconds on `GPU-6222...`; native RELION stopped exactly at iteration 60
and the RECOVAR handoff recorded 0 MiB GPU use.

At iteration 60, incoming and post-update `Igrad1` for both halfsets,
`Igrad2`, and `mom1_noise_power` are all bitwise exact. The paired reference
entering reconstruction is `1.635e-8` relative L2 from RELION and the output
is `1.613e-8`, compared with `2.485e-6` for the `Igrad2`-only trajectory.
Sampled checkpoints 1, 4, 8, 16, 32, 58, and 60 all pass, with minimum
FSC-AUC **`0.999999999972`** and class-assignment accuracy `1.0`.

Strict state is still not exact: one neighboring pose/translation decision
first differs at iteration 44, and `optimal_offset_change` first differs at
the same checkpoint. This is ten iterations later than the `Igrad2`-only
oracle. All other operative schedule fields remain matched. The first
non-exact iteration-60 M-step operands are now raw BPref data/weights at
`2.72e-6--8.30e-6` relative L2; their post-reweight data remains
`9.14e-6--1.01e-5`. This closes the moment, noise-power, and reconstruction
implementations for the residual and makes trajectory-wide raw BPref replay
the next discriminator. M-step, sampled-map, particle, and sampling report
SHA-256 values are `db50b145c6fd...`, `4c5db6168396...`,
`536f3fdbbdcb...`, and `3187579d7400...`. The score remains **2/20**.

The second-moment diagnosis now survives a live-map trajectory intervention,
but it is not sufficient for strict state parity. Local science commit
`4f7a7cfdc` adds a fail-closed templated replay mode that captures native
`Igrad2_post` at every M-step and replays only that buffer in RECOVAR.
Exact-H100 task `13049505_1` completed iterations 1--60 in 409 seconds with
**no native reference-map replay**. `Igrad2` is bitwise exact throughout. At
iteration 60 the live candidate reference entering and leaving reconstruction
is only `2.068e-6` and `2.485e-6` relative L2 from paired RELION. The sampled
map gate passes at every checkpoint (1, 4, 8, 16, 32, 58, 60), with minimum
FSC-AUC **`0.999999999900`** and class assignment `1.0`.

Strict state remains open. Against the paired native run, one 1.5-Angstrom
translation choice first differs at iteration 34 and makes
`optimal_offset_change` depart at the same checkpoint; a four-repeat
diagnostic has unmatched close-tie particle states by iteration 38. At
iteration 60 the carried first moments are still `1.21e-5--1.26e-5` relative
L2 from native even though the second moment is exact. Thus trajectory-wide
`Igrad2` replay closes the dominant map-quality error, while the residual
first-moment/raw-score path is still large enough to change brittle particle
decisions. M-step, particle, sampling, and sampled-map report SHA-256 values
are `b59f55608d78...`, `397801edd0dc...`, `013a98856e59...`, and
`6f2c71afaf87...`. The score remains **2/20**.

The leading late M-step operand now passes a one-variable causal oracle.
Exact-H100 task `13048344_1` repeated the native-map-clamped iteration-60
boundary and replaced only RECOVAR's computed `Igrad2_post` with the paired
native buffer. The intended operand becomes bitwise exact. Reconstructed-map
relative L2 falls from the state-aligned control's `1.038433e-4` to
`3.963338e-7`: a **262.0x improvement** and **99.62% reduction**. Raw BPref,
both first moments, `applyMomenta`, reference input, and `reconstructGrad`
remain live RECOVAR work. The M-step report SHA-256 is `2a6bf2742934...`.

This independent oracle repeat is diagnostic rather than scoring: one of
3,000 particles takes a 1.5-Angstrom neighboring translation from iteration
34 onward (with a second transient mismatch at iterations 46--48), even
though native references are replayed. That does not explain away the oracle:
the exact second-moment substitution still removes 99.62% of the final map
error while those other differences remain live. It does mean the next gate
must replay the accumulated second-moment trajectory against the *same sealed
native realization* used by the zero-particle-failure control, not compare two
independent native runs. Frozen acceptance remains **2/20**.

The late pre-failure GF47 M-step is now split on a paired, state-aligned
trajectory. Local science commit `1afccd18c` extends the bounded runner to
reuse the qualified worker schedule and materialized native block chronology,
pin the physical H100, replay each freshly generated native reference, and
dump the unmodified iteration-60 M-step operands before the diagnostic clamp.
Exact-H100 array task `13047664_1` (internal Slurm job `13047681`) stopped the
native step exactly at iteration 60 and completed the candidate and analyzer
in 404 seconds. All 60 replayed maps are bitwise exact to that paired native
run. The particle audit has **zero divergent particles at all 60 checkpoints**
and minimum pose/translation match fractions of `1.0`; every operative
sampling field also matches.

The leading open input is now the carried gradient state, not the newly
computed particle state. At iteration 60, incoming `Igrad1` differs by
`3.33e-6--7.94e-6` relative L2 and incoming `Igrad2` by **`1.606e-3`**.
Fresh raw BPref data/weight differences are only `2.88e-6--8.57e-6`, while
`reconstructGrad` produces a `1.038e-4` reference difference. The paired
reference entering reconstruction is already aligned to `2.54e-8`. This
closes the black-box reconstruction boundary to accumulated gradient/momentum
state, with second moment the largest measured operand. M-step, particle, and
sampling report SHA-256 values are `b33fafa1f8ab...`, `0acdd8f3694f...`, and
`c2dd22ec46c0...`.

Three preceding placement/orchestration attempts (`13046699`, `13047060`,
`13047397`) are explicitly non-scoring. They exposed an array-specific Slurm
step-ID mismatch that left native RELION alive beside RECOVAR and caused
artificial OOMs. Commit `1afccd18c` makes numeric-step resolution
format-independent and fail-closed; the valid arm records native step
`13047664_1.0` terminal before RECOVAR and 0 MiB used at the handoff. The
frozen score remains **2/20** and no acceptance rule changed.

The inherited-map diagnosis now survives a trajectory-level intervention.
Local science commit `76ce63d7e` adds a fail-closed diagnostic that replaces
only each post-M-step reference from an explicit native-map template; all
RECOVAR E-step, posterior, particle, sampling, momentum, noise, and controller
state remains live. Exact-physical-H100 task `13044790_2` (science step
`13045526`) completed iterations 1--61 in 377 seconds. All 61 replayed map
arrays are bitwise equal to sealed native repeat 1, and the particle audit has
**zero divergent particles at every checkpoint**. Minimum pose and translation
match fractions are both `1.0`; at iteration 61 their maximum errors are only
`9.96e-6` degrees and `7.07e-6` Angstrom. The failed target
`604@particles.128.mrcs` returns to the native pose
`(28.248272, 65.322480, 116.388917)` with Pmax `0.493008`.

Every operative sampling field matches through iteration 61. The generic
sampling audit reports only the known inactive iteration-1 accuracy fields;
current size/resolution, Healpix order, offset range/step, optimal-offset
change, perturbation, update decision, prior mode, and translation topology
have no mismatch. Three additional same-state, same-H100 native continuations
also choose the same target pose; the candidate pose is not a missing sample
from the original four-repeat envelope. Particle, sampling, and fused-capture
SHA-256 values are `0e2037c782f7...`, `6d2d53585437...`, and
`c3211dd8559d...`. This closes the causal branch to the reconstruction-produced
reference trajectory. The next bounded work splits raw BPref accumulation,
gradient moments, and `reconstructGrad` at the latest pre-failure M-step. The
frozen score remains **2/20** and no acceptance rule changed.

The first GF47 particle failure now has a closed score-level explanation.
Exact same-physical-H100 job `13042355` continued sealed native repeat 1 and
replayed the RECOVAR prefix through iteration 61 for
`604@particles.128.mrcs`. All 32 fine candidates, their rotation matrices,
reconstruction masks, and centered orientation/translation priors match.
RELION ranks keys `(0, 121)` and `(2, 121)` with a top-two log-odds margin of
only `+0.000427`; RECOVAR reverses that margin to `-0.016907`. The posterior
change is therefore caused by the raw fine score, not candidate selection or
normalization.

Pinned H100 operand-substitution replay `13043203` then isolated that raw-score
difference. Substituting RECOVAR's reference projection into the native replay
changes the same top-two margin by `+0.018005`; substituting RECOVAR's image
changes it by exactly zero, and substituting RECOVAR's weight changes it by
only `-0.000671`. Rebuilding the projector from the sealed native it60 map is
bitwise exact to the native frozen texture, while rebuilding from the RECOVAR
it60 map reproduces the RECOVAR capture to `4.84e-9` relative L2. The first
open causal boundary is therefore the **iteration-start map state**. Fine
support, priors, image preprocessing, projector construction, score topology,
noise weighting, and posterior math are closed for this failure. The fused
posterior and operand reports have SHA-256 values `a8807cf6c8b7...` and
`6c9d7fed74cf...`. No acceptance threshold or frozen score changed.

The exact frozen GF47 materialized-order/native-epilogue trajectory is now
terminal. Science job `13038307` produced all 201 checkpoints in 2,330 seconds
on the frozen physical H100 from immutable local head `374b24b7f`; audit job
`13040047` then failed closed at schedule iteration 58, particle iteration 61,
and map iteration 80. Checkpoints 0--79 pass the map envelope, which moves the
first map failure one iteration later than serial-float32 job `13025432`, but
does not satisfy the unchanged full-trajectory contract. Runtime is **7.79x**
the sealed 299.1-second native run. The map, state, and terminal report SHA-256
values are `f5f4275a889c...`, `e24718f3c69d...`, and `e5b8bba0df32...`.
The frozen score remains **2/20**.

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

PTX comparison then identifies a native-inert instruction still present in
the captured candidate: every block loads
`rotation_replay_order[blockIdx.x]`, whereas native RELION indexes its already
ordered rows directly by `blockIdx.x`. Local commit `48ac6e358` materializes
posterior and rotation rows into captured physical order before launch, then
uses the identity-grid specialization with the native counts/workers/trace
shape. Focused CPU validation passes **17/17**; H100 gate `13037927` passes
**29/29** and seals CUDA SHA-256 `3b387a3ca00b...`.

Exact-target seed-29 panel `13037971` completes both arms in 86 seconds and is
the closest robust result so far. Every raw data and weight half is inside the
paired native-repeat floor. First moment also passes both halves; reconstructed
reference improves to **1.132x / 1.033x**. Post-second remains mixed at
`1.083x / 0.386x`, and noise power remains outside at `3.050x / 1.971x`.
Because both reference arms must be at or below `1.0x`, this near-pass is not
promoted and no trajectory is submitted. Report SHA-256 is
`41db56dff8bb...`; sealed arm-A schedule and chronology SHA-256 values are
`d6e588170311...` and `088212599123...`.

Native RELION's trace epilogue tests the shared first-atomic claim, sets the
no-atomic flag, and only then reads the final global timer. The candidate had
instead read the global trace record and written the end timer first. Local
commit `374b24b7f` composes the exact native epilogue and a source-order guard
on top of materialized rows. Focused CPU validation passes **15/15**; H100
gate `13038159` passes **29/29** and seals CUDA SHA-256 `13698b431b36...`.

Exact-target seed-29 panel `13038186` completes both arms in 90 seconds and
passes the predeclared reference gate at **0.978x / 0.967x**. All four raw
data/weight accumulators pass; noise is near the floor at `1.093x / 1.029x`.
Post-second remains outside at `2.266x / 3.295x`, so the short result is a
trajectory trigger rather than acceptance evidence. Report SHA-256 is
`ddf8f4b7ecdf...`; arm-A schedule and chronology SHA-256 values are
`55b286f192f1...` and `d8638d621b7b...`.

Pinned 0--200 job `13038307` is running on the same physical H100 from
immutable head `374b24b7f`, CUDA `13698b431b36...`, RELION binding
`fcbb2a8356c2...`, and those exact seed-29 schedule/chronology seals. Audit is
deferred until all 201 checkpoints exist. Wrong-GPU fanout `13038308` exits 75
before output creation, and redundant `13038309` is canceled. The frozen score
cannot change until the terminal map, particle-state, and schedule audit.

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
| 1 | Qualify the shared GF46 coarse correction | exact fused lanes and native mask are sealed by `13105728`; canonical order plus `a8af8b28a`'s min-diff frame gives 100 native members with zero mismatches offline | compose both changes only in the shared EM/VDAM path; pass exact-H100 0--4 and 0--20 before considering a full trajectory |
| 2 | Close GF47's first systematic production departure | iteration 4 is inside the eight-native state union; jobs `12909370 / 12909383 / 13092340 / 13097692` close the iteration-1 posterior envelope and downstream response. Four full repeats first leave native particle support at **58/58/104/104** | retest the same boundary with the corrected shared coarse path; if it remains, resume EM's earliest-unequal-state ladder on the sealed iteration-58 map/score/prior boundary |
| 3 | Repair GF38's replacement boundary | composed-head 0--200 task `13017334` completed in 2,110 s; audit `13018631` fails schedule @20, particle @27, map @60 | close iteration-20 accuracy rotation/translation, then rerun 0--200 |
| 4 | Frozen v3 matrix | **20/20 terminal: 2 accepted, 18 failed, 0 pending**; GF53 fails particle @40 and map @44 while schedule passes | retain every failure as a repair target |
| 5 | Seeded GF29 / GF43 / GF45 calibrated audits | GF29 and GF45 pass; GF43 fails only map at 146 | retain exact accepted/failed outcomes |
| 6 | GF41 authoritative re-audit | `12999430` terminal: map pass, particle fail | retain as classified repair target |
| 7 | Local corrections | `6387ff7c9` posterior mass; `9685e9317` cumulative checkpoint state; `0a001923e` metadata precision; `a8af8b28a` coarse min-diff score frame; shared lane capture `f2a041d9d`; reuse guard `21ac2c872` | full trajectories pass before implementation is published |
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
