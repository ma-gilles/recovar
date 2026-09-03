# RECOVAR / RELION VDAM parity dashboard

> **Authoritative release status: NOT READY.** Frozen v3 K=1 correctness is
> **2 / 20** and frozen runtime parity is **0 / 20**. The only accepted v3
> cases are `vdam-gf44` and `vdam-gf45`. Diagnostics, performance primitives,
> the legacy v2 expansion, K>1, and real data cannot change those scores.

## Live engineering snapshot — 2026-09-03

> **LATEST — STATIC GEOMETRY:** Comparable H100 job `13393276` removes **41
> XLA compilations (-9.43%)** and **0.677 s of recorded compile-miss time
> (-3.00%)** from the optimized iteration-48 replay by moving immutable
> half-spectrum geometry to one shared host plan. Cold wall improves 0.51%; a
> single warm replay improves 11.18%. All **23 / 23** deterministic science
> fields remain exact and both cold/warm execution contracts pass. The strict
> atomic diagnostic is only **14 / 21** inside the two-repeat envelope, with
> misses at existing CUDA-reduction scale (maximum normalized L2 `1.65e-7`),
> so it is recorded but not promoted. [Focused report.](../perf/vdam_static_half_geometry_h100_13393276.md)
>
> **LIVE RESULT:** The default-off compact-posterior + packed/deferred stack now
> makes the exact-state iteration-48 transition **2.583x faster**
> (`5.966 -> 2.310 s`, job `13377600`): pass 1 is **7.891x** faster, pass 2 is
> **1.269x** faster, the coarse score table is **36x smaller**, and every pose,
> translation, class, support ID, posterior maximum, particle state, and
> sampling state is exact. Continuous state remains below `4*float32` epsilon.
> The fixed packed-row ABI separately passed its exact same-state gate
> (`13376686`). Combined with stable Fourier shapes, its four-fresh-process
> trajectory is **25.90% faster** through iteration 50 (`13377009`), but both
> optimized repeats reproducibly choose an alternate hard-assignment basin at
> iteration 35, so strict trajectory science remains on hold. Flat-row-only
> isolation `13378054` recovers **9.37%** fresh wall and **9.58%** expectation
> time, but candidate hard paths split at iterations 29/35 and it remains
> default-off. This assigns most of the remaining combined cold gain to Fourier
> shape classes rather than fixed rows. The complete nine-seam stack passes
> its exact-state iteration-48 gate (`13378175`) at **2.553x** warmed speed:
> pass 1 is **8.124x** faster and pass 2 is **1.204x** faster. Fine
> pretranslation is rejected as
> immaterial (`+0.05%` warm wall). Final-support noise packing completed
> `13377626`: it improves warm pass
> 2 by about 15.7% but is neutral alone end to end, so its value must be measured
> on top of the compact stack; direct-oracle job `13378581` is active. A shared
> flat-row-scatter JIT boundary is under fresh-process gate `13378757`, while a
> projector-storage ABI proof targets the remaining compile churn. The last
> full `0 -> 200` result remains the prior
> **1.639x** speedup with a one-pose iteration-48 basin split. Frozen scores
> remain **2/20 correctness, 0/20 runtime** until a full scoring rerun passes.

| Signal | Status | Evidence / next decision |
|---|---|---|
| Release score | **NOT READY — correctness 2 / 20; runtime 0 / 20** | Frozen v3 is unchanged. Component gates and diagnostics cannot inflate it. |
| Shared static half-spectrum geometry | **23 / 23 EXACT / 41 FEWER COMPILES** | Comparable optimized H100 replay `13393276` changes stderr compile count `435 -> 394`, recorded miss time `22.596 -> 21.918 s`, cold wall `30.069 -> 29.915 s`, and warm wall `5.144 -> 4.569 s`. Strict atomic diagnostics are only 14/21 at nondeterministic scale and are not promoted; [report](../perf/vdam_static_half_geometry_h100_13393276.md). |
| Full combined GF46 sentinel | **COMPLETE — 1.639x FASTER / NON-SCORING DIVERGENCE** | Job `13369646` completed both `0 -> 200` arms: wall `2810.434 -> 1714.290 s` (-39.00%), expectation `2718.193 -> 1621.030 s` (-40.37%), peak RSS `17673 -> 17643 MiB`. First hard split is one pose at iteration 48. Both final maps remain inside the broad native-repeat quality envelope, but two arms cannot establish basin equivalence. |
| Same-state iteration 35 | **EXACT DECISIONS / ATOMIC-SCALE CONTINUOUS NOISE** | Job `13358712` deep-copied one exact live iteration-34 state into an ABBA panel. Every particle/pose/translation/class/posterior/significance field and all 200 exact support-ID rows agree across direct and hybrid; aggregate support SHA-256 is identical. Cross-backend reconstruction deltas are the same scale as direct/direct and hybrid/hybrid atomic-repeat noise. |
| Combined same-state iteration 35 | **QUALITY + MATERIAL RUNTIME PASS** | Job `13368042` enables both optimized seams. All decisions/support/state are exact; maps/noise remain repeat-scale. Warm wall improves `2.603267 -> 1.654054 s` (**36.46%**), expectation **39.85%**, pass 1 **45.33%**, and pass 2 **42.70%**. Full trajectory remains open. |
| Combined same-state iteration 48 | **EXACT DECISIONS / 2.521x WARM SPEED** | Job `13372936` starts all four arms from the exact iteration-47 state. All discrete state and support audits are exact. Cross accumulator L2 is at most `9.63e-8` versus `9.16e-8` direct/direct. Warm wall is `8.142330 -> 3.229593 s`; pass 1 is 4.469x and pass 2 is 1.261x faster. This localizes the sentinel split to accumulated roundoff, not a reproducible transition error. |
| Stable shape trajectory | **HOLD — 2.07% WALL GAIN / STRICT SCIENCE FAIL** | Job `13372996` completed four fresh-process trajectories through iteration 50. Fine pass 2 improves 22.82%, but wall improves only 2.07%. The baseline OFF/OFF pair itself splits discretely at iteration 35, while one ON arm remains discrete-identical to OFF1 through iteration 50. This rules out a deterministic toggle-only displacement but cannot exclude a variance effect or qualify promotion. |
| Shared local projection cache | **REJECTED — FINE POSES CHANGE** | Job `13373715` is repeatable within each mode but changes 21/200 fine pose assignments, nine rotation IDs, and 33 translation coordinates across modes from the exact same state. Cross accumulator L2 reaches `0.1838` versus about `9e-8` repeat noise. The approximate rotation-ID cache aliases oversampled exact matrices. |
| Validity-aware packed fine CUDA | **RETAIN — BITWISE / 3.233x KERNEL-LOCAL** | Jobs `13374637`/`13374638` preserve same-state science. Dedicated ABBA job `13375396` keeps all active values bitwise exact, writes exact `+inf` for all 2,631,104 invalid outputs, and changes four-call wall `0.038706 -> 0.011973 s` (-69.07%). The absolute 26.7 ms saving makes this an enabler, not the gap closer. Exact integrated CUDA artifact job `13376273` also passes. |
| Stable packed-row ABI (`Q=B*R`) | **25.90% COMBINED TRAJECTORY SPEED / STRICT SCIENCE HOLD** | Job `13376686` keeps the same-state transition exact. Combined stable-ABI job `13377009` cuts fresh-process wall `513.649 -> 380.602 s` and expectation 27.41%, with 31.23% fewer JAX cache objects. Flat-row-only job `13378054` assigns 9.37% wall gain to fixed rows and confirms that strict hard-state divergence remains. |
| Fixed flat-row ABI isolation | **9.37% WALL GAIN / STRICT SCIENCE FAIL** | Job `13378054` disables Fourier bucketing and changes only `Q` to `B*R`. Wall changes `512.628 -> 464.613 s`, expectation `479.372 -> 433.451 s`, and big-JIT variants fall `87 -> 48/49`. Ordinary repeats retain exact hard state; candidate repeats first split at iteration 29 and candidate 2 first differs from ordinary at iteration 35. Keep default-off; [report](../perf/vdam_stable_flat_only_trajectory_h100_13378054.md). |
| Pretranslated fine scorer | **REJECTED — NO END-TO-END VALUE** | Job `13377045` keeps science exact, but warm wall changes `2.478951 -> 2.480252 s` (`+0.0525%`) and saves only `0.560 ms` inside big-JIT. The added tensor/memory path is not being integrated or exposed in the GUI. |
| Compact hybrid posterior + packed fine path | **QUALITY PASS / 2.583x WARM SPEED** | Job `13377600` keeps every hard decision and support ID exact with zero fallback. Warm wall changes `5.966168 -> 2.309737 s` (-61.29%); pass 1 is 7.891x and pass 2 is 1.269x faster. Only 0.2595% of full candidates are rescored, using a 42.109 MiB table instead of 1,515.938 MiB. Integrated default-off; full `0 -> 200` remains required. |
| Fully optimized nine-seam stack | **EXACT DECISIONS / 2.553x WARM SPEED** | Job `13378175` composes compact posterior, packed/deferred fine work, and both stable ABIs from one exact iteration-47 state. All crossed decisions/support/state are exact; continuous deltas stay below `4*float32` epsilon. Warm wall changes `5.923261 -> 2.320158 s`, pass 1 is 8.124x and pass 2 is 1.204x faster. Stable shapes add no warm gain; their separate cold-trajectory basin split remains the promotion blocker. |
| Final-support packed noise | **SCIENCE-SAFE / DIRECT ORACLE ACTIVE** | Job `13377626` keeps decisions/support/state exact and continuous deltas near `1e-8`. Warm pass 2 changes `1.311271 -> 1.105007 s` (-15.73%), while standalone wall changes `5.880235 -> 5.964363 s` (+1.43%). Dense scalar order is now preserved while pixel-heavy work stays packed; in-memory mature-EM oracle job `13378581` is the acceptance gate. |
| Output pseudo-halfsets | **FIXED / 78 TARGETED TESTS PASS** | Commit `a8ce0bb52` writes `_rlnRandomSubset` from RELION InitialModel part-id parity, mapped back to input STAR rows, and overwrites stale labels. `tests/unit/initial_model/test_native_driver.py`: `78 passed`. |
| Root cause closed | **SHARED EM/DIRECT OPERANDS** | The pre-fix hybrid cache used `mask_current_image_disk=False` while mature EM/direct used `True`. Commit `e9a8e8256` now routes both through one shared projection helper; the previous 405 pose-assignment mismatches fell to **0 / 1,000**. |
| Performance decomposition | **BOTH PASSES NOW MATERIAL** | Across the prior 200-iteration hybrid, coarse pass 1 improved `1787.367 -> 493.504 s` (**3.62x**) while pass 2 stayed flat. Combined job `13368042` now improves the same-state pass 1 **45.33%** and pass 2 **42.70%**. The combined full-trajectory effect is not measured yet. |
| Cold-runtime diagnosis | **COMPILATION / CONTROLLER DOMINATED** | In the stable GF46 trajectory, 32/50 iterations create new executables and consume `317.39 / 348.37 s` of expectation time; the 18 zero-new-cache iterations total only `30.98 s`. GPU monitoring is 73.5% zero-utilization samples and 3.96% mean utilization. Current work stabilizes shared scatter and projector ABIs rather than tuning already-small kernels. |
| Packed fine-row scorer | **EXACT PRIMITIVE; PRODUCTION HOLD** | Primitive job `13361586` is bitwise exact and cuts score/posterior call time by 32.77--54.69% across GF46 iterations 20/40/60/80. Same-state production job `13362546` preserves every discrete decision/support row and stays inside the atomic repeat envelope, but warm pass 2 changes `1.219563 -> 1.223849 s` and whole-iteration wall `2.302587 -> 2.395506 s`. Default remains off. |
| Packed projection | **SCIENCE DECISIONS EXACT; 2.35% WARM GAIN** | Job `13363465` projects 38,016 packed rows instead of 62,208 padded rows: pass 2 `1.183086 -> 1.142239 s`, big JIT `1.023525 -> 0.998770 s`, and whole iteration `2.091715 -> 2.042595 s`. All decisions/support are exact; one tiny derived-state tail is 2.37x the largest repeat amplitude, so this remains a default-off component for the combined gate. |
| Final-support VDAM deferral | **QUALITY PASS / PERFORMANCE HOLD / TRAJECTORY OPEN** | Jobs `13367167` and `13367508` reuse the exact 1,868-pixel packed projection made by mature EM scoring, scatter it into the original dense layout, and call shared EM denominator/noise reductions. All decisions/support are exact; raw A2/XA and maps are repeat-scale; final noise is about `3e-8` normalized L2. Clean warm wall improves **4.09%**, narrowly below the 5% component threshold. Default remains off pending the combined trajectory gate. |
| Numerical classification | **MATHEMATICALLY EQUIVALENT REPEAT-SCALE NOISE** | Reprojection, not dense reduction order, caused the stable error. Exact projection reuse removes it; remaining float32 reduction/CUDA atomic variation is stable and nondirectional at this boundary. Full-trajectory basin stability is still required. |
| Focused regression | **INTEGRATED SEAMS PASS** | The stable-shape integration slice passes `365 / 365` CPU tests after building the optional RELION binding in isolated scratch; 16 GPU-only tests are intentionally skipped locally. The compact/stable/CUDA-focused slice then passes `244 / 244`, and the composed harness passes `22 / 22`; Ruff, py_compile, shell syntax, and diff checks pass. No broad RECOVAR suite was run. |

### Immediate queue

1. Continue compile-boundary attribution from the now-host-planned static
   geometry: the remaining cold profile is 394 compilations / 22.72 seconds,
   led by the coarse certificate, local big JIT, and eager controller
   primitives. Require exact hard state and repeat-controlled atomics for each
   retained boundary.
2. Finish the literal mature-EM direct-oracle packed-noise gate `13378581` and
   the shared scatter-boundary fresh-process gate `13378757`; retain only
   oracle-exact seams with measured cold or composed value.
3. Prove and gate a shared stable RELION-projector storage/radius ABI. Shape
   attribution shows 73.5% zero-GPU-util samples and 317/348 seconds of
   expectation time in iterations that create new executables.
4. Then run a fresh
   repeat-controlled `0 -> 200` sentinel for the qualified complete
   stack, with direct/direct variability, FSC/scale, selector/fallback, memory,
   compile-count, and runtime audits. A two-arm basin split cannot update a score.
5. Expand across outliers, pose/noise distributions, scale, parameters, and
   long trajectories. K>1 and real data remain later independent gates.

## At a glance

| Track | Score or status | Current boundary |
|---|---:|---|
| Frozen v3 K=1 correctness | **2 / 20** | Release gate; unchanged. |
| Frozen v3 runtime | **0 / 20** | Independent release gate; unchanged. |
| Legacy v2 expansion | **6 / 15** | Regression track only; no v3 score impact. |
| Current correctness work | **SAME-STATE EXACT / REPEAT-CONTROLLED BASIN GATE OPEN** | Job `13372936` rules out a reproducible combined-backend error at the sentinel's first hard split, iteration 48. The remaining question is whether full-run divergence exceeds direct/direct stochastic basin spread. |
| Current performance work | **25.90% FRESH 0->50 / 2.553x COMPLETE-STACK TRANSITION** | Combined stable ABIs cut fresh trajectory wall 25.90% but choose an alternate basin at iteration 35, so remain default-off. Flat rows alone account for 9.37% wall / 9.58% expectation and also fail strict trajectory identity. Job `13378175` proves the complete nine-seam stack exact for iteration 47->48: 8.124x pass 1, 1.204x pass 2, 2.553x wall. Direct-oracle packed noise and shared scatter JIT gates are active. |
| K>1 | **UNQUALIFIED** | Separate gate after K=1 closure. |
| Real data | **NOT SCORED** | Separate confirmation gate; no release claim. |

## Frozen v3 K=1 scorecard

The denominator is the fixed 20-case, iteration 0--200 v3 suite. Strict
correctness is the conjunction of map, particle-state, and pre-divergence
schedule gates. Runtime is scored independently. Component and coverage rows
are useful diagnostics, not additional passing cases.

| Gate | Passed | Evaluated | Denominator | Role |
|---|---:|---:|---:|---|
| K=1 strict full-trajectory correctness | **2** | 20 | 20 | `release_gate` |
| Map trajectory | **5** | 20 | 20 | `diagnostic_component` |
| Particle-state trajectory | **6** | 20 | 20 | `diagnostic_component` |
| Pre-divergence schedule | **13** | 20 | 20 | `diagnostic_component` |
| Runtime within 1.10x RELION | **0** | 20 | 20 | `independent_release_gate` |
| Complete terminal audits | **20** | 20 | 20 | `coverage` |

No diagnostic below may add a pass, replace a failed v3 case, or change either
release denominator.

## Non-scoring v2 expansion

| Track | Result | Role | Frozen v3 impact |
|---|---:|---|---:|
| `legacy_parameter_expansion_v2` | **6 / 15** | `regression` | **none** |

The v2 result is retained for historical regression coverage. It is neither a
subtotal nor evidence that the v3 score is 8/35.

## Correctness: current hybrid-GEMM boundary

The integrated shared-EM hybrid is **4.47x faster** in the sealed GF46
one-transition comparison and **1.84x faster** over the complete 200-iteration
trajectory. Standalone GEMM remains unsafe as the sole scorer: it changed
downstream discrete state and escaped direct-repeat envelopes. The accepted
design uses GEMM only to pre-screen, then direct-rescores every certified
source block capable of changing RELION's float32 raw winner, offset,
posterior, support, or selected state.

The complete direct/hybrid run first differs discretely at iteration 35. Job
`13358712` replayed that exact transition from one shared in-memory state and
found no score, support, or downstream decision difference whatsoever. The
remaining continuous differences match ordinary CUDA reconstruction-atomic
repeat noise. The current blocker is therefore full-trajectory basin stability,
not a known hybrid candidate omission.

| Evidence | Status | Read / decision |
|---|---|---|
| Explicit componentwise `abs2`, job `13327200` | **NO GO** | `4.394x` warm coarse comparison, but discrete state and map/model envelopes failed. Complex-absolute lowering is not the cause. |
| Full promoted operands / FP64, job `13327874` | **NO GO** | `4.512x` warm coarse comparison, but the same meaningful state and map/model failures remained. Precision promotion is not the fix. |
| Selected 16-rotation-block direct primitive, job `13328717` | **PRIMITIVE PASS** | CPU contract checks plus full-rectangular and selected-block H100 atomic-envelope checks passed. Default remains off; this is the direct-rescore building block, not a production selector. |
| Dual raw/post streaming certificate at `6e4e0ae65cba3cd0f86febdfe319e747ced07d97` | **INTEGRATED** | v2 artifacts separately preserve pre-prior/raw-max and posterior/support state, enforce byte accounting, and fail closed when either certificate family is absent. |
| All-1000 GF46 diagnostic, job `13329608` | **COMPLETE / NON-SCORING** | Exact H100 run completed in 69 s from clean source `6e4e0ae65cba3cd0f86febdfe319e747ced07d97`; diagnostic-only and timing-ineligible. |
| All-1000 promoted-FP64 diagnostic, job `13330442` | **COMPLETE / NON-SCORING** | Worst observed score delta fell from `3.3125` to `1.5`, but the critical particle-1933 near tie and particle-636 support surplus remained; source-block topology was unchanged. |
| Fused FP64 interval scorer and immutable topology seam, commit `d810048f1` | **FOCUSED CPU PASS** | Exact-rational stored-score enclosure, raw/post compact reduction, overflow/underflow fail-close, lookup ownership/tamper checks, and mature-EM GEMM reuse pass. Default-off; GPU lowering and candidatewise coverage are still gates. |
| Native source16 FP32/FTZ audit, commit `a0a9c248f` | **BINARY ARITHMETIC AUDITED** | The exact selected-block kernel has no local FTZ operations and uses FP32 FTZ/RN atomics in every native `sm_80`, `sm_86`, `sm_89`, `sm_90`, `sm_100`, and `sm_120` cubin. The source-specific additive envelope, first-overflow argument, provenance, and invalidation rules are durable. This does not yet qualify the complete hybrid. |

### All-1000 selector result: job `13329608`

| Boundary | Result |
|---|---|
| All-candidate and pre-prior error coverage | **1,000 / 1,000** particles over **1,069,056,000** finite pairs; 0 nonfinite; `Emax=3.3125`, RMS `0.05108`, signed mean `+0.00789`. |
| Raw winner / support decisions | Winner mismatch **1 / 1,000** (`particle 1933`, tiny margins); support mismatch **1 / 1,000**, one extra and **0 false negatives** (`particle 636`). |
| Raw-max safe direct blocks | **1,000 / 1,000** covered; at most 3 source-16 blocks. |
| Posterior TopK=2048 safe blocks | **727 / 1,000** covered. Every covered particle needs at most 6 source-16 blocks (median / p95 / max = `6 / 6 / 6`). |
| Diagnosis | Pair retention saturates before block capacity. The next selector is **per-source-block maxima**, followed by direct rescoring with certified fallback. |

Score impact: **none**. The run does not change frozen correctness **2 / 20**,
frozen runtime **0 / 20**, production defaults, or trajectory qualification.

Promoted FP64 does not make standalone GEMM safe.  Job `13330442` paired all
1,069,056,000 finite candidates again: RMS delta fell to `0.03593`, signed
mean to `-0.00058`, and pair-TopK coverage rose only from 727 to 732 particles.
Every covered posterior union still used at most six source-16 blocks.  FP64
is retained only as a potentially tighter center for the formal interval
certificate; exact selected-block direct rescoring remains mandatory.

## Runtime and performance lanes

The frozen runtime score remains **0 / 20**. A component speedup is not an
end-to-end VDAM runtime pass, and a fast numerically unsafe arm is rejected
before timing can authorize promotion.

### Retained or accepted for narrower use

| Lane | Evidence | Narrow decision |
|---|---|---|
| Cache-only arm | `13277456` | Retain for repeat/scale study; not promoted. |
| Flat-row scorer | `13361586`, `13362546` | Bitwise primitive and live science pass, but production warm big-JIT is neutral and whole iteration is 4.04% slower. Keep default off; extend packing through projection before another promotion gate. |
| Packed projection | `13363465` | Exact decisions/support and a 2.35% warm whole-iteration gain. Retain default-off and combine with final-support noise/M-step deferral before promotion. |
| Final-support VDAM deferral | `13367167`, `13367508` | Exact scoring-projection reuse restores A2/XA and final noise to repeat scale with exact decisions/support. Clean warm pass 2 improves 12.51%, big JIT 24.43%, and wall 4.09%. Advance default-off to combined hybrid/trajectory gates. |
| Combined coarse + fine path | `13368042` | Exact decisions/support/state and repeat-scale continuous state; warm whole iteration improves 36.46%, pass 1 45.33%, and pass 2 42.70%. Advance default-off to the full-trajectory sentinel. |
| Stable packed-row ABI | `13376686` | Exact same-state decisions/support and repeat-scale continuous state. Warm-neutral, but converts 78 observed `Q` values to mature EM's fixed `B*R` ABI; fresh-process trajectory `13377009` is the decision gate. |
| Fixed flat-row ABI-only trajectory | `13378054` | Wall improves 9.37% and expectation 9.58%, but both candidate trajectories leave the ordinary hard path (iterations 29/35) and one later changes the schedule. Keep default-off; use only as a compile-shape diagnostic. |
| Compact posterior + packed/deferred fine path | `13377600` | Exact hard state/support, zero fallback, 36x smaller coarse score table, and `2.583x` warm wall speed. Advance default-off to the all-optimization same-state and full-trajectory gates. |
| Final-support packed noise | `13377626` | Exact discrete/support state and 15.73% faster pass 2, but one-float32-ULP retained-mass drift and 1.43% slower standalone wall. Keep default-off; first preserve dense scalar order and isolate incremental timing. |
| Segmented fixed-capacity scan | `13360809` | Exact shared-EM execution boundary; performance rejected as the gap-closing lever (maximum 1.176x on 16 tiny calls, slower at two calls). |
| Stable fine window | `13264981`, `13265301` | Exact primitive; forecast only, default off. |
| Shared eight-stream coarse scheduler | `13279168`, `13279367` | Math accepted; performance hold below the runtime target. |
| Native-atomic T=29 plus eight streams | `13281836`, `13283759` | One-iteration math/runtime pass; default off pending no-growth and trajectory gates. |
| Selected-block direct rescore | `13328717` | Qualified primitive for the hybrid; not wired into production. |
| Shared GF46 projection-cache builder | `13332001` | Full `(1,36864,5100)` complex64 destination, verified donation/zero XLA insert temporary, exact sentinels, no OOM. Uniform 4608 rows removes the tail executable. Infrastructure only. |

### Rejected or do not promote

| Lane | Evidence | Decision |
|---|---|---|
| Physical-order chunking | `13277457` | Rejected; slower cold and neutral/slower warm. |
| Batched CUB trajectory | `13268653` | Rejected after state/schedule escapes and map-envelope failures. |
| 80M x-half cap | `13260950`, `13265965` | Rejected; causal operand equivalence was not proved. |
| Direct x-half implementation | `13281684`, `13282815` | Math accepted but performance rejected at only 2.12% warm-wall gain. |
| Shared posterior executor | `13280796`, `13281970` | Math accepted but performance rejected; posterior kernel time regressed. |
| Explicit `abs2` GEMM | `13327200` | Correctness no-go; do not promote. |
| FP64 GEMM operands | `13327874` | Correctness no-go; do not promote. |
| Pretranslated fine scorer | `13377045` | Scientifically equivalent but warm wall is 0.0525% slower and big-JIT saves only 0.560 ms. Do not integrate or expose. |

## K>1

K>1 is **unqualified** and has no v3 score impact. Do not use K>1 diagnostics
to close the K=1 gate or report a combined percentage. Its first admissible
gate requires a separately frozen suite, class matching, per-class FSC/FSC-AUC,
state/convergence checks, and a same-GPU runtime comparison.

## Real data

Real data is **not scored** and has no v3 score impact. A representative,
well-characterized real-particle run is a confirmation after synthetic K=1
correctness and stable hybrid behavior, not a substitute for either frozen
release gate.

## Next gates

1. Finish the in-memory mature-EM oracle for packed final-support noise and the
   shared scatter JIT fresh-cache trajectory.
2. Stabilize the shared RELION projector storage/radius shape after proving
   exact central overlap for every GF46 logical-to-physical size pair.
3. Integrate only qualified gates, then run a same-GPU repeat-controlled
   combined `0 -> 200` panel with basin,
   FSC/scale, selector/fallback, memory, and runtime audits.
4. Then rerun the frozen K=1 trajectory suite and the expanded outlier,
   pose/noise-distribution, scale, parameter, and long-trajectory matrix.
5. Keep frozen v3 at 2/20 and runtime at 0/20 unless a separately reviewed
   scoring rerun updates the authoritative scorecard. Only then open K>1 and
   real-data gates as independent tracks.

## Evidence and reproducibility

| Evidence | Immutable provenance |
|---|---|
| Frozen v3 dashboard snapshot | Commit `6f10c2d3f075654a94caa6e44249a5b1275b48f6`; `docs/math/vdam_relion_parity_scorecard_v3.json` SHA-256 `9ee5ccea3af8e26ef75303fa75314b0168960380e3fdbcb042403b2c4dd2ea50`. |
| Frozen v3 suite definition | `docs/math/vdam_k1_full_trajectory_expansion_v3.json` SHA-256 `9842b2c9cb7646d75127541801ef5982ed19e4a80485f9ce586ceabdb3ed0091`. |
| Primary / superseding v3 science | Primary commit `984637b7db95f1ca6f5800c08ea14c1e32c82c2e`; GF43/GF45 superseding commit `580477763f0f95f028841b074210c4eba34fd24b`. |
| Explicit-abs2 rejection | Source `9bddc776a598436c7b7440394298c548e3579aa0`; job `13327200`; sealed decision `75828be7569b2a44f671660a5ecc39e42c5d141e`. |
| FP64 rejection | Source `cf0db35e469203e5f78131bbf34bdd66c71a94ed`; job `13327874`; sealed decision `89aa5c21f07c8e088b66a1e429831acac3efb879`. |
| Selected-block primitive | Source `695a629fa70ee951734d98728bb3daffcc53bd88`; job `13328717`; report commit `56f2ee1e892925421fa2d807379062158f6eb1e3`. |
| Dual-certificate integration | Streaming base `0b574de9029c71de5e7ef3b14785c07f203d20f5`; pre-prior certificate `8b0d5bcacdda537337a4bae0f5b2ac242d796562`; runner `74d1eb60de5c3a437df232f7b60efa6b89db4d2f`; pinned head `6e4e0ae65cba3cd0f86febdfe319e747ced07d97`. |
| All-1000 diagnostic | Job `13329608`; clean source `6e4e0ae65cba3cd0f86febdfe319e747ced07d97`; H100 `GPU-099c0d77-bb85-f2e9-f628-148b733c9176`; identity certificate SHA-256 `3dfd92da58365c33388eea7071f1814be1561a5c7557615b52a9e263ce7e230f`; aggregate manifest SHA-256 `a78674eb7c342d16f624fe3d68b6a3c2ea868a6ca880a6c4f972d728f02a2a3c`. |
| All-1000 promoted-FP64 diagnostic | Job `13330442`; clean source `2c1a9e299e40563f2f5058dce490231c39fb7a12`; H100 `GPU-099c0d77-bb85-f2e9-f628-148b733c9176`; identity certificate SHA-256 `3a666ea737cedc2aabbaa033a1a226058004ae9983adeb4d3c05f1b95551d5ce`; aggregate manifest SHA-256 `faa883a1c6fa756adf679c7138c4b38e7135f8cc0aaac89c2d181b7d10f0a754`. |
| Certified scorer/topology foundation | Commit `d810048f1`; focused scorer/topology `40 / 40`; combined scorer/selector/streaming/macro regression `87 / 87` before the topology-only seam, followed by the 40-test seam rerun. |
| Native source16 FP32/FTZ audit | Commit `a0a9c248f`; audited library SHA-256 `92c3c098995ed89f32c4d258936363402c0f1c1d0945f9c452d79f9eb1b5dd9f`; artifact manifest SHA-256 `c7c7dac19cad2b117d130ae34a70cf4066a37231d15775c1ffe424ab6e158db1`; [report](vdam_coarse_source16_fp32_audit_20260902.md). |
| Shared projection cache | Source `43402732cb169ab5d91d90b10262a35ba99edae4`; job `13332001`; H100 `GPU-2ee3da91-970a-6714-84df-530aefe04a08`; result SHA-256 `576384429d01fbe2d86a81c13a7519c484490e5ac62bb8bc871387df653023c2`; [report](../perf/vdam_projection_cache_h100_13332001.md). |
| Integrated hybrid H100 boundary | Source `b611aeff15004a07d6d2a1ec590bc5b97180cb5b`; job `13341618`; H100 `GPU-2ee3da91-970a-6714-84df-530aefe04a08`; artifact-manifest SHA-256 `fec2ec40f98057a428799a472ea36908725c24ed54041d21ceacd99cf0ef325a`; [report](../perf/vdam_gf46_hybrid_batch_h100_13341618.md). |
| GF46 integrated-hybrid one-transition seal | Source `85c4b13bf5f7b975ea4504750fe8495556e6030c`; job `13344325`; H100 `GPU-75c2d200-95d1-ef57-fb52-1698386c756c`; report JSON SHA-256 `10649aa2916269335f7615d399c0a3388bf722a49a0223b6f2acc7a863535351`; [report](../perf/vdam_gf46_hybrid_transition_h100_13344325.md). |
| GF46 integrated-hybrid full trajectory | Harness source `c5910f956db6c8a0706d91acfb78b456dbdce64f`, production candidate `e0c1d1746570e64bad3618b09a50be31b79ebe60`, job `13354357`, H100 `GPU-97adb339-219f-d72d-11c9-74dc92fcff8c`, source manifest SHA-256 `9f728182a2c963974697ca06f52db9804282cbe42714c0c1bc27f9c2c575f048`. |
| GF46 same-state iteration-35 boundary | Source `2fc852da7a408d32dd0142fb177371de0407c552`, job `13358712`, H100 `GPU-9f98ccbf-3c62-c54f-7409-7eb58845ad4a`, report JSON SHA-256 `a3809404cf10f5c4dd473c3a09795189cc07c543fb3b7d9d89b8fc892c3e7153`; [report](../perf/vdam_hybrid_same_state_it35_h100_13358712.md). |
| Shared fixed-capacity multi-call seam | Source `231a0191cc17615fe97eef9a3894b93a824f5a64`, job `13359718`, H100 `GPU-099c0d77-bb85-f2e9-f628-148b733c9176`, source manifest SHA-256 `eb668b8bb28adb7ade5aefb9ec73f8bb8bfc85bc76eb5209035838286a528c3f`, result SHA-256 `96374d0c934f870f6f20ef0edf38c3ab4dc924fcea5cf0879e3abce19c2fc840`; [report](../perf/fixed_capacity_local_multicall_h100_13359718.md). |
| Segmented fixed-capacity scan | Source `560c1a6c754328140613fa5a2e3aea7b3e17afe3`, job `13360809`, H100 `GPU-099c0d77-bb85-f2e9-f628-148b733c9176`, run JSON SHA-256 `4035dfc21008b751d9c9cd9b905cfcf49bf6e90c7de0dda3ae380b2edb6c207c`; [report](../perf/fixed_capacity_segmented_scan_h100_13360809.md). |
| Packed fine-row primitive | Source `bc800e4fa2e5a607a6a4ecd480a174e0ca5ca628`, job `13361586`, H100 `GPU-099c0d77-bb85-f2e9-f628-148b733c9176`, run JSON SHA-256 `69866a5f13f64c3c4cb13dfc12647d635883188296abf23404d1ba5f65667ece`; [report](../perf/vdam_flat_row_score_h100_13361586.md). |
| Packed fine rows at production seam | Source `1de2e224b2aed0659cf954e4763fb1ef260355f5`, job `13362546`, H100 `GPU-0d7b80c7-fef8-e346-6332-de36ae1af518`, report JSON SHA-256 `0e255292085570ed3ee92a08450da083664bd31dd3794ecbae6dc4aaaf6536ee`; [report](../perf/vdam_flat_rows_same_state_it35_h100_13362546.md). |
| Packed projection at production seam | Source `cb5855d8367b452d48b6765f7e9265ed8d076ca4`, job `13363465`, H100 `GPU-0d7b80c7-fef8-e346-6332-de36ae1af518`, report JSON SHA-256 `825791a60b2e35168b807020b6bc7bc32bb149b5c82fe9c7b3be45ecb7de855c`; [report](../perf/vdam_packed_projection_same_state_it35_h100_13363465.md). |
| Final-support VDAM deferral | Source `fea9ae567e732d013860737e2cf5f2022c548818`, job `13364363`, H100 `GPU-0d7b80c7-fef8-e346-6332-de36ae1af518`, report JSON SHA-256 `55bb5760935c5ea1314ad9591558f8ba281f71f88141201e8b58c18383459afe`; [report](../perf/vdam_packed_deferred_same_state_it35_h100_13364363.md). |
| Dense-order final-support diagnostic | Source `0e1b4355c85b024d45b4782bdca6f7eba2656223`, job `13366669`, H100 `GPU-235ec3bc-ca9f-1c0e-88eb-c8b37c5e0480`, report JSON SHA-256 `ec6ded886dff2465a536b2c6c987092d91d40beac68b3cade57eac712764cf78`; [report](../perf/vdam_packed_deferred_dense_order_it35_h100_13366669.md). |
| Exact scoring-projection reuse | Source `cf9791d35e2b97cd5b64426aab33b749e3b50ce3`, diagnostic job `13367167`, clean job `13367508`, H100s `GPU-990435ac-e5fe-18d9-c741-59b8fd9c9439` and `GPU-5297e2fc-3064-625f-a65a-9db11614d705`, report JSON SHA-256s `1a02a96533fbd1b7d9c111da056aef41f35ead8cbc032e05b578d8f6e10c7313` and `6a979206fdb062678862d378d25516093bdad3fe83d41e63af21eadc6633ad08`; [report](../perf/vdam_packed_deferred_projection_reuse_it35_h100_13367167_13367508.md). |
| Combined hybrid + packed-deferred transition | Source `209aae4593f0a040090e150482f9d44dfa57e79f`, job `13368042`, H100 `GPU-0d7b80c7-fef8-e346-6332-de36ae1af518`, report JSON SHA-256 `302b222ebc223d580314012e6cc423f467981d96f24d2bd6bc0aff278ae5ebe1`; [report](../perf/vdam_hybrid_packed_deferred_same_state_it35_h100_13368042.md). |
| Combined full sentinel | Source `10cd188edc4d81d038e582c277c65d60396368d1`, job `13369646`, H100 `GPU-97adb339-219f-d72d-11c9-74dc92fcff8c`; two-arm setup/performance sentinel only, wall `2810.434 -> 1714.290 s`. |
| Combined iteration-48 causal gate | Source `10cd188edc4d81d038e582c277c65d60396368d1`, job `13372936`, H100 `GPU-e2c3190a-9599-15f7-a19c-7ae55e4e0a85`, report JSON SHA-256 `e734d48748cc6ae41f3851df5878b8928456d3b32db31fd39511b0927c5cec5e`; [report](../perf/vdam_combined_same_state_it48_h100_13372936.md). |
| Stable Fourier-shape trajectory | Source `47190851099ba608d60114de1df0624e9efe8e1a`, job `13372996`, H100 `GPU-ddb1592d-744e-ea56-d0a3-aec6e7c97d10`, analyzer report SHA-256 `037cdda817bb5914d6836aaf305ea2a6383539035620909fef602c586b316149`; [report](../perf/vdam_stable_shapes_trajectory_h100_13372996.md). |
| Shared local-projection-cache rejection | Source `8816d487e76540690e8e38cfa12bd014ba5d856f`, job `13373715`, H100 `GPU-0d7b80c7-fef8-e346-6332-de36ae1af518`, report SHA-256 `0ec47650576126f977380235595402e5766cf43671c856e665fa86ce7256c557`; [report](../perf/vdam_local_projection_cache_it48_h100_13373715.md). |
| Validity-aware packed fine rows | Source `4dc4a79f7c5f6f31d7a8e04bfe43339b700b48c6`, same-state jobs `13374637`/`13374638`, dedicated timing job `13375396`, H100 `GPU-75c2d200-95d1-ef57-fb52-1698386c756c`, timing manifest SHA-256 `d89361d388017f55ec77f9948673c6de6307594ab4f55edde697b4a88b2e5481`; [science report](../perf/vdam_flat_row_validity_it48_h100_13374637.md), [timing report](../perf/vdam_flat_row_invalid_early_exit_h100_13375396.md). Integrated source `9ae038377778419d5ead8f96c5eb75753807fd8d`, job `13376273`, CUDA SHA-256 `870ac72044ef78bcccd1ab695b9991b088cc42512231d1a3cafc000ea32ea777`. |
| Stable packed-row ABI | Source `2957ae6a4e4d81ca9c5524871f4e9609c2d9d39d`, job `13376686`, H100 `GPU-1fdb3b99-e7ff-fe6d-4f59-9d2cc85fa319`, report JSON SHA-256 `1eb3d9a92aa9886c7a08cdca91ae9f06899b4a2ab845f8cbfaf2d5a6bf5a8b42`; [report](../perf/vdam_stable_flat_capacity_same_state_it48_h100_13376686.md). |
| Combined stable-ABI trajectory | Source `f5792e9b0af6a70f13cb63a546d594013c774f16`, job `13377009`, H100 `GPU-e2c3190a-9599-15f7-a19c-7ae55e4e0a85`, report JSON SHA-256 `a39f95b6b0f56b95b1f7e91bd6531f9b7c1a4d582c37e94503cf9656c13400e0`; [report](../perf/vdam_stable_combined_abi_trajectory_h100_13377009.md). |
| Fixed flat-row ABI-only trajectory | Run source `64d6433c41a87370a70d779415b5df95ca53a188`, diagnostic analyzer `d3e29dafe`, job `13378054`, H100 `GPU-e2c3190a-9599-15f7-a19c-7ae55e4e0a85`, report JSON SHA-256 `265166b28c0268bdec081e5ba27c1e3954e22ee0541bbab276a44d45d3541f54`; [report](../perf/vdam_stable_flat_only_trajectory_h100_13378054.md). |
| Compact posterior + packed/deferred transition | Source `17ec60ca5dcd5d3b67e956876d9df07ed2a6345d`, job `13377600`, H100 `GPU-0d7b80c7-fef8-e346-6332-de36ae1af518`, science report SHA-256 `3e7b0b0df783adc2ce8c64ddd3e2860ea2f0c87908ca6cc5c938694f7dc62b25`; [report](../perf/vdam_compact_packed_same_state_it48_h100_13377600.md). |
| Fully optimized nine-seam transition | Source `0729473a1a752f1474edd666ae0740ef692049ed`, job `13378175`, H100 `GPU-ef985070-011e-0782-6f0a-94b053dcc120`, science report SHA-256 `5976788bbb57f4dbe88726e2f315050fd305e8ebf1dc39648d3ece6246057f54`; [report](../perf/vdam_all_optimized_same_state_it48_h100_13378175.md). |
| Fine-pretranslation rejection | Source `79375ada3`, job `13377045`, H100 on `della-h19g3`; immutable decision report commit `b7e1f1da1`, default-off and not integrated. |

Job `13329608` artifacts are under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_coarse_gemm_gf46_stream_v2_6e4e0ae65_h21g4_20260901/`
and are marked `SAFE_TO_DELETE`.

## Code references

- `recovar/em/dense_single_volume/helpers/coarse_gemm_streaming.py:CoarseGemmStreamingState`
- `recovar/em/dense_single_volume/helpers/coarse_gemm_streaming.py:summarize_coarse_gemm_streaming_state`
- `recovar/em/dense_single_volume/helpers/coarse_gemm_streaming.py:aggregate_coarse_gemm_streaming_summaries`
- `recovar/em/dense_single_volume/helpers/coarse_gemm_hybrid.py:CoarseGemmCertificateTopology`
- `recovar/em/dense_single_volume/helpers/coarse_gemm_hybrid.py:coarse_gemm_direct_f32_ftz_envelope_and_range`
- `recovar/em/dense_single_volume/helpers/projection_cache.py:build_projection_cache`
- `recovar/em/dense_single_volume/helpers/scoring.py:_relion_coarse_gaussian_gemm_update_certificate_state`
- `recovar/em/dense_single_volume/helpers/scoring.py:_relion_coarse_diff2_rotation_blocks_from_topology_f32`
- `recovar/em/dense_single_volume/helpers/significance.py:_score_relion_coarse_gaussian_gemm_macro`
- `recovar/em/dense_single_volume/helpers/significance.py:_seal_coarse_gaussian_gemm_streaming_scope`
- `recovar/em/initial_model/dense_adapter.py:_initial_model_coarse_gemm_diagnostic_scopes`
- `recovar/cuda_backproject.py:relion_coarse_diff2_rotation_blocks_f32`
- `recovar/cuda/cuda_backproject.cu:launch_relion_coarse_diff2_rotation_blocks_f32`
- `scripts/run_vdam_coarse_gemm_gf46_streaming_selector.sbatch`
- `scripts/run_vdam_coarse_rotation_blocks_primitive_gate.sbatch`
