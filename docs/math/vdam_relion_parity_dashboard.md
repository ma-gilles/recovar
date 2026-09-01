# RECOVAR / RELION VDAM parity dashboard

> **Authoritative v3 status — NOT READY.** Strict K=1 correctness is **2 / 20** and runtime parity is **0 / 20**. The only accepted cases are `vdam-gf44, vdam-gf45`.
>
> This page is generated from the frozen 20-case, iteration 0--200 scorecard. Scheduler diagnostics, the legacy v1/v2 tracks, K>1, and real data cannot change this score.

## At a glance

| Axis | Frozen score | Current read |
|---|---|---|
| K=1 correctness | **2/20** | Unchanged; accepted cases are `vdam-gf44, vdam-gf45`. |
| Runtime | **0/20** | Unchanged; observed suite range 4.91--11.58x. Promotion requires a large reproducible gain without instability or quality loss. |
| Performance lanes | non-scoring | Cache-only is retained for repeat/scale; chunking is rejected. The shared eight-stream coarse scheduler is mathematically accepted and measurably faster, but held below the runtime target. The 65--128 single-lane specialization is inapplicable to GF46's actual T=29 coarse call; the clean tracking-derived native-atomic T=29 batched-lane port cuts warm wall about 9%, but its frozen two-repeat maximum-envelope smoke gate fails narrowly. The separate 8+8 panel passes its numerical and material-runtime gates after an independently reviewed serializer-only recovery. The path remains default-off and no long trajectory is authorized until a new sealed true-200 harness exists. |
| Numerical policy | non-scoring | Mathematical equivalence is required; bitwise identity is not a universal requirement. Tiny differences are acceptable only as stable, unbiased, non-growing, repeat-bounded noise that preserves discrete choices, basin, and final quality, and only with a material, large, reproducible end-to-end runtime advantage. Unstable numerics fail. |
| EM reuse | shared production primitives | The remaining boundary is execution topology/variability, not duplicate projector or scorer math. The default-off shared fixed-capacity score seam is at Phase 1e (`793e3bb12a`; 146 passed; CPU review GO). Correctness-only H100 job `13288282` passed 7/7 focused tests plus 8 captured and 12 production comparisons exactly; speed/default remain unqualified. |
| Later gates | separate | K>1 remains unqualified; real data remains unscored. |

## Current focus

| Evidence | Result | What it rules out | Explicit next gate |
|---|---|---|---|
| Same-H100 GF46 it180->181 factorial `13276891/13276923/13277456/13277457` | **CACHE-ONLY RETAINED FOR REPEAT/SCALE; CHUNKING REJECTED.** Cache-only warm wall -5.27% and cold wall -1.24%, with +0.365 GiB HWM; all tracked discrete invariants exact. | Padding reduction alone does not close the gap: chunk-only was +14.09% cold and +0.23% warm. | Repeat cache-only across scales; do not promote it yet. |
| Production coarse trace `13275886` | **SCHEDULING/OVERLAP IS THE MEASURED GAP.** RECOVAR coarse union 4.355 s versus native 2.752 s, despite 3.881 ms per RECOVAR slot versus 3.925 ms per native particle. | The shared production kernel is not slower per unit; six serial padded batches lose to native RELION's eight-stream overlap. | Default-off K=1 multistream scheduling around the accepted shared production coarse kernel, preserving its texture math and canonical reduction; do not substitute the rejected diagnostic native-texture path. |
| Shared coarse multistream primitive `13279168` + crossed ABBA `13279367` | **MATH ACCEPTED; PERFORMANCE HOLD.** Warm expectation 9.104->8.155 s (-10.42%) and pass 1 6.266->5.493 s (-12.35%); every tracked discrete state is exact and map/BPref deltas remain at repeat scale. | Scheduling is only part of the gap: the per-particle RECOVAR kernel is 49.8% slower than native and carries 8192 excess shared bytes plus 56 versus 48 registers/thread. | The sealed T=29 factorial 13280655 supersedes the active_lanes=1 hypothesis for GF46. Compare the already-existing shared native-RELION atomic reduction against canonical reduction at T=29, then cross the winner with the accepted eight-stream scheduler under the stable numerical-equivalence policy. |
| T=29 applicability factorial `13280655` | **NEGATIVE / REDIRECTED.** All 3000/3000 winners are exact and map rel-L2 is at most `3.170e-09`, but all arms ran the generic kernel because the live coarse operand has 29 translations. | The earlier 116 count was oversampled/fine, not coarse; active-lanes=1 cannot accelerate GF46. | Benchmark the already-existing shared native-RELION atomic lane admission at T=29: 48 registers and 7040 B shared versus canonical's 56 registers and 15232 B. If materially faster, qualify atomic plus eight streams against native/control repeat envelopes before considering a T=29 four-lane canonical shuffle. |
| Clean tracking native-atomic + coarse batched lanes `13285438/13285647` | **MATERIAL RUNTIME PASS; FROZEN TWO-REPEAT NUMERIC GATE FAIL; DEFAULT OFF.** Warm wall -9.03%, expectation -10.26%, pass 1 -10.97%, coarse sum -10.14%, and coarse union -10.83%; launches are 48 instead of 6 serial launches. | All 3,000 STAR rows and discrete metadata are exact, with negligible bias and scale drift, but max cross rel-L2 exceeds the within-repeat maximum by 16.30% cold and 7.66% warm. The immutable strict report remains FAIL (`b46b54d168c9cf4febf7fb6527d2633e7c8c4b83f4569f0b3cfd2d2e801f9506`). | Keep the frozen n=2 strict result visibly failed. Build and independently review a same-binary 4+4 true-200 no-growth, basin, final-quality, and end-to-end runtime harness. Do not submit the old 20-iteration ABBA scaffold and do not enable the default. |
| Replicated batched-lane panel `13286397` | **ALL PREDECLARED NUMERICAL AND RUNTIME GATES PASS AFTER INDEPENDENTLY REVIEWED SERIALIZER-ONLY RECOVERY; DEFAULT OFF.** All 16 fresh-process arms completed and all 3,000 STAR rows/discrete outputs are exact. Warm max rel-L2 is `2.357e-09`, joint p is `0.3164`, cache-amplification p is `0.6327`, and median warm wall changes -9.32% (36.301->32.919 s). | The original Slurm wrapper failed only while JSON-serializing two NumPy booleans after RUNS_COMPLETED; the sealed analyzer stayed byte-identical at `9f1010ab02dfa07649ccea8cf93f2be1007d323b4d724fb4dce400868e07142c` and the original acceptance contract is `4f266733a4a7ffbc79c06d340918a4389c79d7c5cf34401031d07a64c38f258d`. The original failure/absent COMPLETED marker remain preserved. Recovered JSON SHA-256 `66d52235cf429f952b5db5bf087fdd72f51cc82210abbd8dcd494eaedf12aaf1`; recovery provenance SHA-256 `4d74c2840631b57840ec34d53547c35752f26f467566eca79b3e600e83a4e3a2`. The frozen n=2 rejection is not overwritten. | The reusable trajectory scaffold is only 2+2 through iteration 20 and its analyzer is unsealed; build and review a same-binary 4+4 true-200 drift, basin, quality, and runtime gate before submission. |
| Same-binary ABBA `13271166` | **NUMERICALLY EQUIVALENT; END-TO-END GAIN IMMATERIAL**; zero particle-state/schedule escapes; relative-L2 map differences remain ~1e-7 and warm speedup is `1.0091x` | All four arms loaded CUDA SHA `6210cdb1cc97aa72fbdf80b36b501ad48c8d1d1e4866f4a2c11889076e1bff53`; different CUDA libraries are not the cause, and the strict two-control map-diameter flag alone is not a scientific rejection. | Run multi-repeat stability/equivalence checks for drift growth, bias, variance, state-escape rate, basin changes, and final quality; seek a different optimization if end-to-end gain remains near 1.009x. |

## Late-iteration same-H100 factorial

GF46 iteration 180->181 ran at source `f61808a0e6` on `della-h21g4` / `GPU-099c0d77-bb85-f2e9-f628-148b733c9176`. This is a one-transition diagnostic gate only; it cannot change frozen correctness **2/20** or runtime **0/20**, and no production default is authorized.

| Arm / job | Cache / radix / chunk | Cold wall | Warm wall | Warm expectation | Warm HWM | Numerical read | Decision |
|---|---|---:|---:|---:|---:|---|---|
| A / `13276891` | off / 4 / 0 | 26.637 s (control) | 13.904 s (control) | 8.899 s (control) | 3.556 GiB (control) | discrete exact; control repeat map rel-L2 `2.19155e-09` | **CONTROL** |
| B / `13276923` | auto / 2 / 220 | 30.646 s (+15.05%) | 13.021 s (-6.35%) | 8.311 s (-6.60%) | 3.959 GiB (+11.35%) | discrete exact; map rel-L2 `2.44339e-09` | **HOLD / REJECT AS DEFAULT** |
| C / `13277456` | auto / 4 / 0 | 26.306 s (-1.24%) | 13.172 s (-5.27%) | 8.132 s (-8.62%) | 3.920 GiB (+0.365 GiB) | discrete exact; map rel-L2 `2.033e-09`, within repeat envelope | **RETAIN FOR REPEAT/SCALE; NOT PROMOTED** |
| D / `13277457` | off / 2 / 220 | 30.388 s (+14.09%) | 13.936 s (+0.23%) | 8.866 s (-0.37%) | 3.620 GiB (+1.80%) | discrete exact; map rel-L2 `2.249e-09` | **REJECT** |

Reference job `13275886` measured native RELION at 5.240 s process / 4.334 s expectation, versus RECOVAR 14.714 s warm wall / 9.751 s expectation. Nsight resolves the coarse path as `6_serial_padded_batches_on_one_stream` versus `1000_per_particle_launches_over_8_streams`. The shared RECOVAR coarse kernel is slightly faster per executed slot; the remaining gap is serial scheduling, padding, and lost overlap rather than different projector math.

The numerical gate is scientific, not bitwise. Mathematical equivalence and stable, unbiased, non-growing repeat-bounded noise are mandatory. Discrete changes are measured, not universally forbidden: rare marginal changes may be accepted only when consistent with control/native repeat variability, remain in the same basin, and cause no material final-quality loss. A slight quality change inside that stable envelope is acceptable only when paired with a large runtime gain. This factorial's exact tracked decisions are strong evidence, not the universal acceptance definition.

## Performance lanes

| Bucket | Lane | Evidence | Explicit next gate |
|---|---|---|---|
| **ACCEPTED PRIMITIVE ONLY** | Flat-row scorer `13266322/13266460` | Active raw/dense scores, posterior 6/6, poisoned tail, and call count are bitwise; isolated combined-call reduction 32.88%, 35.77%, 54.53%, 32.68%. Default-off. | Wire behind an explicit default-off control, include shared compact-pair packing/projection cost, and repeat the bitwise live-call boundary before any short trajectory. |
| **ACCEPTED PRIMITIVE ONLY** | Stable fine window `13264981/13265301` | Logical [68, 70, 72] under physical 72 is bitwise; compile identities 3->1; 5.2--5.4% is forecast-only. Default-off. | Integrate score, Wavg, and BPref physical shapes together, poison-test inactive tails, and measure a live exact boundary before a trajectory. |
| **REJECTED** | Batched CUB trajectory `13268653` | State escapes it4/p285, it16/p2902, it18/p902; schedule escape it18; 21 candidate map checkpoints outside, worst ratio `87.9460303805`. Cold/warm speedups `1.0247x`/`1.0517x` are non-scoring. Raw report SHA-256 `a67f6c969e84da096c70d88219ddb4e6962ecd13266814743d992099be7b172d`. | Rejected. Revisit only after a causal fix preserves the scalar scan/reduction chronology and passes the same fail-closed ABBA trajectory gate. |
| **NUMERICALLY EQUIVALENT / E2E INCONCLUSIVE** | Elementwise primitive `13269547` + trajectory `13269681` | Primitive is bitwise at it20/40/60/80 and 7.1839--31.6257x faster; trajectory has zero state/schedule escapes and only roundoff-scale terminal relative-L2 `5.452e-07`. Raw report SHA-256 `42544acfe0ae193022808abdbdf56639f418f6102d4e66b9a44ddc1a0aa1ff56`. | Run a multi-repeat equivalence panel measuring drift growth, bias, variance, discrete state-escape rate, basin changes, and final quality. Promote only with a large reproducible end-to-end gain. |
| **NUMERICALLY EQUIVALENT / E2E IMMATERIAL** | Same-binary causal `13271166` | Zero state/schedule escapes; roundoff-scale relative-L2 differences at it2/3 with identical CUDA SHA. Cold/warm speedups `1.0762x`/`1.0091x`; the warm gain is immaterial. Raw report SHA-256 `ccb9d9cc4f4ee949aabbfa2c6045aea5b6c2007bcdbcd871e0e1df246d0c3db0`. | Run multi-repeat stability/equivalence checks for drift growth, bias, variance, state-escape rate, basin changes, and final quality; seek a different optimization if end-to-end gain remains near 1.009x. |
| **REJECTED** | 80M x-half `13260950/13265965` | Iteration-1 topology is identical but 3/3 artifacts differ; causal projection effect was not proved. Prior 7.56% wall reduction is unusable. | Do not revisit without one-process replay from a byte-identical frozen incoming state, resetting accumulators between 40M and 80M arms. |
| **MATH ACCEPTED / PERFORMANCE HOLD** | Shared 8-stream coarse scheduler `13279168/13279367` | All tracked discrete state is exact; warm expectation improves 10.42%, but coarse union remains 3.885 s versus the 3.0 s gate. | The sealed T=29 factorial 13280655 supersedes the active_lanes=1 hypothesis for GF46. Compare the already-existing shared native-RELION atomic reduction against canonical reduction at T=29, then cross the winner with the accepted eight-stream scheduler under the stable numerical-equivalence policy. |
| **REJECTED FOR GF46 / RETAINED PRIMITIVE** | Single-lane coarse `13280613/13280655` | Primitive 2/2 passes, but GF46 is T=29; requested-vs-generic coarse union changed only 0.0030%. No discrete/basin effect. | Benchmark the already-existing shared native-RELION atomic lane admission at T=29: 48 registers and 7040 B shared versus canonical's 56 registers and 15232 B. If materially faster, qualify atomic plus eight streams against native/control repeat envelopes before considering a T=29 four-lane canonical shuffle. |
| **CLEAN PORT: STRICT N=2 FAIL / INDEPENDENTLY REVIEWED REPLICATED RECOVERY PASS** | Native atomic + coarse batched lanes `13285438/13285647/13286397` | The tracking-derived shared kernel improves warm wall 9.03%, expectation 10.26%, GPU union 10.45%, and coarse union 10.83%. All 3000/3000 STAR rows/discrete outputs are exact, but the frozen two-repeat max-envelope rule fails narrowly. The recovered 8+8 panel passes with -9.32% median warm wall and warm max rel-L2 `2.357e-09` after independent serializer-recovery review GO. Prior experimental lane-only job `13284897` remains supporting, non-production evidence. Immutable clean-port report SHA-256 `b46b54d168c9cf4febf7fb6527d2633e7c8c4b83f4569f0b3cfd2d2e801f9506`. | Keep the frozen n=2 strict result visibly failed. Build and independently review a same-binary 4+4 true-200 no-growth, basin, final-quality, and end-to-end runtime harness. Do not submit the old 20-iteration ABBA scaffold and do not enable the default. |
| **MATH ACCEPTED / PERFORMANCE REJECTED** | Direct RELION x-half BPref `13281684/13282815` | Actual CUDA K=1/K=3 primitive outputs are bitwise; crossed GF46 decisions are exact and map/BPref remain in repeat noise. Finalize improves 85.40%, but warm wall improves only 2.12%. `13281543` is invalid collection-only; jobs `13281914/13281950/13282022` stopped before science. | Keep default-off and do not spend a trajectory on this implementation; preserve it as a qualified primitive for a future larger fused finalization redesign. |
| **MATH ACCEPTED / PERFORMANCE REJECTED** | Shared posterior executor `13280796/13281970` | Every discrete result is exact and map rel-L2 is at most `2.348e-09`; warm wall improves only 3.37% while posterior-kernel time regresses 36.86%. | Keep default-off and do not spend a trajectory on this implementation. Use the sealed native/RECOVAR profiles to identify a larger remaining execution-topology boundary. |
| **SEALED PROFILE / LARGE LEVER IDENTIFIED** | Native-vs-RECOVAR decomposition | RECOVAR summed GPU kernel work is only 3.05% above native, but overlap is 1.108 versus 1.651; measured excess idle is 3.344 s (22.54% of warm wall). Six serial coarse kernels underlap, while 1000 per-image launches inflate coarse work 35.06%. | Extend the mature shared EM planner/executor with approximately eight persistent coarse-grained batched lanes and pipeline data/layout preparation through coarse, fine, and finalization stages. Preserve shared math and avoid a VDAM-only executor. Report SHA-256 `079f52c02b20128902e99461770092fbadd6aa35f30b7c0b1ad209b53ff3658b`. |
| **H100 SCORE CORRECTNESS PASS / RUNTIME UNQUALIFIED** | Shared fixed-capacity local executor | `local.run_local_em_exact` accounts for 53.547 of 57.833 s XLA compile (92.6%). Big-JIT plus fused x-half alone consume 37.404 s. The compile-only forecast is 10.1% of the 423 s run, before packed-work savings. Phase 1e `793e3bb12a` seals the default-off shared call-0 score seam through one mature EM/VDAM numeric wrapper. H100 job `13288282` on `della-h21g4` / `GPU-099c0d77-bb85-f2e9-f628-148b733c9176` passed 7/7 focused tests and all 8+12 float32/float64 comparisons: every score, centered-score, logZ, best-score, Pmax, posterior, and mass delta is exactly zero; support/discrete state, significant counts `[2, 4, 4]`, and row count `6` are exact. This is correctness-only: no speed or default promotion. | Run an independently reviewed same-binary/crossed H100 donation runtime and peak-memory A/B; then widen the shared fixed executor beyond call 0. The correctness-only score gate authorizes neither a speed claim nor default promotion. Source manifest `986f6c733672425e87c8de6b8c7dec18e5d4085c663145d5e2510af6d0a72e6c`; CUDA `948a728b98e2d38c882a6832abba991cbbcb4ae87474b849f109166dd7158db6`; diagnostics `f9b73b6facd9f74b41c4e7c76a46f6b47fb6d45734cc032f5a07f8fcadf64d25`; gate JSON `065e2901accefe57e59e61e6048a3ae5e66d929e4ce470527c640bec19849f7f`; JUnit `96dc875b31f7c0a5bbe61f3344cd9f07745a92bd0b1b8ee8ddb5408185d2082f`. Sealed analysis SHA-256 `fd33935f77c275615249e6149abfdf9abf50d4b1c8463aa4dcbe11f2ea176def`; runtime report SHA-256 `5ffa9ae565466aa924eccd1a836a3a5af1613df40cf4c584b344ec3dd4494165`. |

Invalid jobs `13270868`, `13270984`, `13285416`, and `13285596` stopped before candidate science and are not evidence. All listed lanes are diagnostic and default-off/unwired, with **no impact** on frozen correctness 2/20 or runtime 0/20. Forced nondefault native-texture path remains rejected after 120/120 strict artifacts differed.

### Gate progression

| Gate | Status | Evidence |
|---|---|---|
| `13252518`: 20-iteration `vdam-gf46` | **PARTICLE TRAJECTORY EXACT** | All 3,000/3,000 pose/translation states match at every iteration; first divergence is `null`; requested/effective Wavg=`true`, radix=`4`. Wall 177 s; pre-artifact 158.846 s. 177 s is below the 182 s accepted short baseline but above the 169.40 s prior exact-control result; these are cross-run H100 observations, not a paired speed result. Evidence: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf46_typed_runtime20_6f39ad52e_20260831/provenance/particle_state_audit_it001_020.json`. |
| `13253088` + audit `13256248`: 80-iteration typed gate | **MIXED: POLICY/MAP PASS; PARTICLE FAIL@37; RUNTIME INCONCLUSIVE** | Typed policy **PASS 80/80 in both arms**. Same-GPU map **PASS 81/81**; active-particle **FAIL@37 — OPEN**; runtime **INCONCLUSIVE**. Original profiled repeat stopped at iteration 64 (invalid harness); corrected job `13254470` completed, with direct map/particle FAIL@4, but is cross-GPU diagnostic only. Frozen scores stay 2/20 correctness and 0/20 runtime. Evidence SHA-256: policy `0c6c60df5c841ebbdbd54d67df681d6a5c2c1229d0657315d4add08d663f28f5`, runtime `5ffa9ae565466aa924eccd1a836a3a5af1613df40cf4c584b344ec3dd4494165`, map `4a2fa726dd2ee7e491983cd67e6212211dc8022aa89a9510e89495035ea1010a`, particle `a7772c546379ac414a511fbcd15907144de264da18677ad6de41c8bec03c47ba`. |

## Primary panels

| Gate | Passed | Evaluated | Denominator | Role |
|---|---:|---:|---:|---|
| K=1 strict full-trajectory correctness | **2** | 20 | 20 | `release_gate` |
| Map trajectory | **5** | 20 | 20 | `diagnostic_component` |
| Particle-state trajectory | **6** | 20 | 20 | `diagnostic_component` |
| Pre-divergence schedule | **13** | 20 | 20 | `diagnostic_component` |
| Runtime within 1.10x RELION | **0** | 20 | 20 | `independent_release_gate` |
| Complete terminal audits | **20** | 20 | 20 | `coverage` |

Strict correctness is the conjunction of map, particle-state, and pre-divergence schedule gates. Runtime is an independent release gate; diagnostics have no score impact.

## Frozen v3 case matrix

| Case | Seed | Stress | Map | Particle | Schedule | Runtime | Strict | Evidence |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `vdam-gf43` | 29 | baseline; uniform poses; white noise | FAIL@146 | PASS | PASS | 7.88x FAIL | FAIL | `gf43_seeded` |
| `vdam-gf44` | 29 | anisotropic poses; outliers; high noise | PASS | PASS | PASS | 8.03x FAIL | PASS | `v3_original` |
| `vdam-gf45` | 29 | Kent poses; outliers; high noise | PASS | PASS | PASS | 9.33x FAIL | PASS | `gf45_seeded` |
| `vdam-gf46` | 29 | anisotropic poses; severe outliers; radial/high noise | FAIL@20 | FAIL@4 | PASS | 9.40x FAIL | FAIL | `v3_original` |
| `vdam-gf47` | 29 | extreme outliers; uniform poses; white noise | PASS | PASS | FAIL@10 | 6.42x FAIL | FAIL | `v3_original` |
| `vdam-gf48` | 29 | very-high noise; uniform poses; white noise | FAIL@45 | FAIL@30 | FAIL@10 | 6.84x FAIL | FAIL | `v3_original` |
| `vdam-gf49` | 29 | low noise; uniform poses | FAIL@115 | PASS | PASS | 11.07x FAIL | FAIL | `v3_original` |
| `vdam-gf50` | 29 | low noise; Kent poses | FAIL@41 | FAIL@40 | PASS | 11.58x FAIL | FAIL | `v3_original` |
| `vdam-gf51` | 29 | no CTF; radial noise | FAIL@74 | FAIL@39 | PASS | 5.89x FAIL | FAIL | `v3_original` |
| `vdam-gf52` | 29 | Kent poses; junk particles; translations | FAIL@40 | FAIL@40 | FAIL@40 | 8.20x FAIL | FAIL | `v3_original` |
| `vdam-gf53` | 29 | high resolution; radial noise | FAIL@44 | FAIL@40 | PASS | 4.91x FAIL | FAIL | `v3_original` |
| `vdam-gf54` | 29 | midscale; Kent poses; radial noise | FAIL@45 | FAIL@30 | PASS | 7.81x FAIL | FAIL | `v3_original` |
| `vdam-gf55` | 101 | anisotropic poses; outliers; high noise | FAIL@46 | FAIL@40 | PASS | 7.98x FAIL | FAIL | `v3_original` |
| `vdam-gf56` | 101 | Kent poses; outliers; high noise | FAIL@45 | FAIL@29 | FAIL@30 | 6.92x FAIL | FAIL | `v3_original` |
| `vdam-gf57` | 101 | anisotropic poses; severe outliers; radial/high noise | FAIL@44 | FAIL@11 | PASS | 10.40x FAIL | FAIL | `v3_original` |
| `vdam-gf58` | 101 | extreme outliers; uniform poses; white noise | FAIL@94 | FAIL@48 | PASS | 6.60x FAIL | FAIL | `v3_original` |
| `vdam-gf59` | 101 | very-high noise; uniform poses; white noise | PASS | FAIL@30 | PASS | 6.86x FAIL | FAIL | `v3_original` |
| `vdam-gf60` | 101 | low noise; uniform poses | FAIL@42 | FAIL@40 | FAIL@20 | 8.73x FAIL | FAIL | `v3_original` |
| `vdam-gf61` | 101 | low noise; Kent poses | FAIL@41 | FAIL@40 | FAIL@40 | 6.40x FAIL | FAIL | `v3_original` |
| `vdam-gf62` | 101 | Kent poses; junk particles; translations | PASS | PASS | FAIL@20 | 7.21x FAIL | FAIL | `v3_original` |

## Active boundary diagnostics

These are scheduler/causal diagnostics, not v3 score entries. INVALID attempts and expected hypothesis rejections have no score impact.

| Job | Role | Status | Scientific outcome | Scheduler state | Score impact | Interpretation |
|---:|---|---|---|---|---|---|
| `13206294` | `diagnostic` | **DIAGNOSTIC** | PASS | expected-contract-failure | none | Science candidate is green through iteration 4; the wrapper is expected to exit red after asserting the paired contract. |
| `13207483` | `diagnostic` | **INVALID** | INVALID | cancelled | none | Cache miss under a newer source; no causal scientific result. |
| `13207996` | `diagnostic` | **INVALID setup** | INVALID | failed-setup | none | Target GPU mismatch invalidated setup. |
| `13208089` | `diagnostic` | **INVALID setup** | INVALID | failed-setup | none | Missing worktree RELION binding invalidated setup. |
| `13208186` | `diagnostic` | **INVALID** | INVALID | cancelled | none | Cancelled before science because duplicate-cache GPFS setup was too slow; no causal result. |
| `13208265` | `diagnostic` | **INVALID** | INVALID | cancelled | none | Cancelled before science because duplicate-cache GPFS setup was too slow; no causal result. |
| `13208734` | `diagnostic` | **INVALID** | INVALID | cancelled | none | A warm-cache arm compiled critical science keys; cancelled INVALID with no parity inference. |
| `13208735` | `diagnostic` | **INVALID** | INVALID | cancelled | none | A warm-cache arm compiled critical science keys; cancelled INVALID with no parity inference. |
| `13209422` | `diagnostic` | **INVALID** | INVALID | cancelled | none | fresh_a science PASS through iteration 4 (audit_status 0; cache 0->435), but warm80_a added 435 entries to the sealed 5037-entry cache, including critical science keys; cancelled at 2:23 INVALID with no parity inference. |
| `13210232` | `diagnostic` | **EXPECTED FAIL-CLOSED** | HYPOTHESIS REJECTED | failed | none | cold_a PASS and warm_a PASS with pair_a 0->4823 then byte-stable 4823->4823; cold_b FAIL@4 and warm_b FAIL@4 with pair_b 0->4676 then byte-stable 4676->4676. Wrapper FAILED as an expected fail-closed hypothesis rejection, not a new scorecard failure. |
| `13211317` | `diagnostic` | **INVALID/SUPERSEDED** | INVALID | failed | none | Prior ordered-shell attempt is INVALID/SUPERSEDED by valid same-cache job 13211719; it carries no parity inference or score impact. |
| `13211719` | `diagnostic` | **EXPECTED HYPOTHESIS REJECTION** | HYPOTHESIS REJECTED | failed | none | Valid A/B execution used one canonical cache: A populated 0->377 and A-after/B-before/B-after were byte-identical. Repeatability failed scientifically: one float32 ULP first appears in an E-step posterior scalar, followed by ordered-noise and both-half BPref differences; the target stack stayed exact. |
| `13212500` | `diagnostic` | **INVALID HARNESS** | INVALID | cancelled | none | Cancelled before science after the loader treated the qualified CUDA library as stale and launched make/nvcc against that source artifact. The library bytes stayed unchanged, but a source-side .build.lock was created. INVALID HARNESS: no A/B result, runtime result, or promotion. |
| `13254010` | `diagnostic` | **EXPECTED FAIL-CLOSED** | HYPOTHESIS REJECTED | failed | none | EXPECTED FAIL-CLOSED: the warm candidate preserved 20/20 particle states but was slower (+4.1% wall; +7.3% pass 1), then changed its sealed cache (1277->1280 files). The 128-tail palette is rejected, default-off, and cannot be promoted. |
| `13256612` | `diagnostic` | **MICROGATE ONLY** | BITWISE PASS MICROCASE | completed | none | SUBORDINATE MICROGATE: bitwise in the active-3 / batch-500 CUDA microcase and 23.91x faster there only. Full paired job 13257087 supersedes it; there is no default change, promotion, or frozen-score impact. |
| `13257087` | `diagnostic` | **VALID SCIENCE FAIL / DO NOT PROMOTE** | HYPOTHESIS REJECTED | failed | none | VALID SCIENCE FAIL / DO NOT PROMOTE: this forced the nondefault native-texture path, while the accepted warm80 run is dominated by the production relion_coarse_diff2_projector_f32_kernel. All 20 particle pose/translation schedules match in both pairs, but 120/120 strict artifacts differ amid comparable within-arm repeat drift; the 1.00695x (~0.69%) median wall gain is noise-scale. |
| `13257182` | `diagnostic` | **VALID EXACTNESS FAIL** | HYPOTHESIS REJECTED | failed | none | VALID EXACTNESS FAIL: the representative active-200 / batch-500 microgate was 2.317x faster but active rows were not bitwise equal. This reinforces the full paired DO NOT PROMOTE decision. |

Job `13209422` warm-cache additions included: `run_local_bucket_big_jit`, `relion_coarse_diff2_projector_f32`, `coarse posterior`, `relion_vdam_mstep_fused_projector_x_half`. Evidence: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf46_jax_cache_history_profilematched_39a38f9d6_20260830/analysis/arms.tsv`.

### Pair-stable cache result: job `13210232`

| Pair | Cold arm | Warm arm | Cache transitions | Warm bytes |
|---|---:|---:|---|---:|
| `pair_a` | `cold_a` PASS | `warm_a` PASS | 0->4823; 4823->4823 | byte-stable |
| `pair_b` | `cold_b` FAIL@4 | `warm_b` FAIL@4 | 0->4676; 4676->4676 | byte-stable |

Particle `286@particles.128.mrcs` has the exact historical graph-pair red pose. The wrapper **FAILED as an expected fail-closed hypothesis rejection**; this is not a new scorecard failure.

Evidence: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf46_jax_cache_history_samepath_ee673be1f_20260830/analysis/cache_history_summary.json` (SHA-256 `cb13a0d710936a6234dfa242e392b021d0b7f52565eb287b0d32dd14fb8a4782`).

**Conclusion:** long-run cache reuse/deserialization is not necessary. Pair-stable independently compiled cache outcomes narrow the unresolved boundary to **compile or autotune variant versus runtime reduction**. Prior ordered-shell attempt `13211317` is **INVALID/SUPERSEDED** by the valid result below.

### Valid same-cache ordered-shell result: job `13211719`

| Cache snapshot | Files | Manifest SHA-256 |
|---|---:|---|
| A before | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| A after | 377 | `2703b7d6fdbc0d329407e20574a90791a15af59c6dc882bc2e617069e28ef3d5` |
| B before | 377 | `2703b7d6fdbc0d329407e20574a90791a15af59c6dc882bc2e617069e28ef3d5` |
| B after | 377 | `2703b7d6fdbc0d329407e20574a90791a15af59c6dc882bc2e617069e28ef3d5` |

Both arms used one canonical cache; B added 0 files and changed 0 files. A-after, B-before, and B-after are byte-stable.

| Earliest / downstream boundary | Nonexact extent | Maximum absolute difference | Exact companion |
|---|---:|---:|---|
| E-step `max_posterior_per_image` | particle ID `2896` / selected index `178` | 1.1641532182693481e-10 (1 float32 ULP) | pose / rotation / translation / class exact |
| ordered image power | 27/65 shells | 512.0 | - |
| ordered sigma numerator | 1/65 shells | 0.00390625 | - |
| final noise | 24/65 shells | 0.001562500001455192 | - |
| live BPref | both halves; 12 fields | h0 0.0234375; h1 0.0107421875 | target stack `286` exact |

The Slurm `FAILED 1:0` terminal state is an **expected scientific-gate hypothesis rejection**: both arms completed, the execution and cache proof are valid, and this is not a frozen-score failure.

Evidence: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf46_ordered_noise_shell_samecache_3b5afd98e_20260830T1847ET/analysis/repeatability.json` (SHA-256 `a3c544602c9845e7c514393e6e20b47d153d23ec8b8757c0a60d1b17d5fe2619`); `analysis/jax_cache_validation.json` (SHA-256 `da25be1dbd38b4940fa5e673081136c808d8240fb9e939fd6863408fdaa24794`); log SHA-256 `fedbccb441958eb01a4609c6e7ea12d062d5724afcb706c7a86c04869780dbb4`.

**Conclusion:** identical persistent-cache bytes do not guarantee exact ordered-shell replay. The first captured upstream difference is one float32 ULP in an E-step posterior scalar; discrete selection is still exact before differences become visible in ordered noise and both-half BPref.

### Invalid speed-gate attempt: job `13212500`

Cancelled after 00:04:23 before any A/B science or timing result. The loader treated the qualified CUDA library as stale and launched `make`/`nvcc` against the source artifact. Its bytes remained unchanged at SHA-256 `a548e44d81adcad7d0356ad369d8cfd23aae7404c1383b1ca2cf85967e77241b`, but it created `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_ordered_scatter_graph_gate_6b5e6568a_20260830/.build.lock`. The attempt is **INVALID HARNESS** and authorizes no 80-iteration promotion.

Evidence: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf01_sig_bucket_ab_73945d69f_20260830T1900ET/trials/cold/control/runner.log` (SHA-256 `693ca77cb8a01ad7dc78b282df19f7df8c5ef4bc624ad2c9f99ef365b508fc04`); Slurm log SHA-256 `f39303615d8f692e242f0df8116649139e9d47d4bc78ad5186b2a2f82c47eed3`.

### Warm H100 profile: job `13248509`

Iterations 47--80: 240.36 s wall, 44.58 s kernels, 35.899 s coarse projector, 57.83 s XLA compile, and 31.90 s dataset getitem. The profile points to execution topology and shape churn; full artifact paths and hashes remain bound in the JSON ledger.

### Engineering decision ledger

| Track | Decision | Evidence |
|---|---|---|
| shared EM arithmetic primitives | **KEEP** | Coarse scoring, compact planning/fine posterior, Wavg, radix buckets, and ordered accumulation already use the mature EM implementations. |
| ordered-scatter CUDA Graph | **QUALIFIED CANDIDATE** | Job 13203664 was quality-neutral (80/80 particles; 81/81 maps) and reduced wall 293->291 s. |
| eager shared raw-image cache | **REJECTED** | Exact-control wall regressed 169.40->172.20 s (+1.7%); 9cb34ddf2 was reverted by 274e4062d. |
| inline indexed fine projection | **REJECTED** | Job 13249200 was bitwise exact but 2--4x slower because it repeated projections. |
| float32 coarse scorer | **REJECTED** | 788,541/802,681 coarse scans changed; commit 31953d remains a regression-only prototype. |
| typed Wavg/radix defaults | **KEEP; POLICY + MAP PASS; PARTICLE OPEN** | Audit job 13256248 confirms typed policy PASS 80/80 in both arms and the same-GPU map envelope PASS 81/81; the active-particle envelope first fails at iteration 37, so correctness remains open. |
| coarse 128-tail palette | **REJECTED; DEFAULT OFF** | Job 13254010 failed closed after the sealed candidate cache changed; its completed warm arm was slower (+4.1% wall and +7.3% pass 1). |
| dynamic coarse-tail mask | **REJECTED; DO NOT PROMOTE** | The 23.91x active-3 microgate was superseded by full paired job 13257087: only ~0.69% median wall gain on a forced nondefault native-texture path, 120/120 strict artifact mismatches, and no production-kernel benefit. |
| runtime profiler toggle | **KEEP UNSET IN PRODUCTION** | Same-GPU crossed job 13258895 favored profiler-off in both pairs (+14.37% and +0.95% cost when on) with exact 20/20 pose/translation trajectories. Shared node contention invalidates a precise magnitude claim. |
| bounded pass-1 to pass-2 raw cache | **EXACT BUT BOUNDED; DEFAULT OFF** | Job 13260861 is bitwise exact in 15/15 sampled blocks and saves ~0.061 s per 200-row two-pass slice, only ~4.9 s over 80 iterations; the first clean full pair was 1.45% slower. |
| shared coarse-projection cache | **EXACT BUT IMMATERIAL; NOT INTEGRATED** | Jobs 13261042/13261159 are bitwise exact; job 13261339 crosses over near batch 45 and reaches only 1.01115x at batch 200. GF46 has one batch per pass, so its end-to-end ceiling is below 0.52%. |
| x-half 80M outer projection batch | **REJECTED / CAUSAL PROOF UNQUALIFIED** | Exclusive same-H100 job 13260950 reduced median wall by 7.56%, but fail-closed operand job 13265965 found different iteration-1 artifacts before the two caps predict different topology. Commit 732868abb records that no causal x-half invariant was established; no split was implemented or promoted. |
| literal pool-preserving local buckets | **REJECTED BEFORE GPU PAIR** | Job 13262146 proves exact source chronology and 49.5--82.2% fewer logical padded rows, but raises calls from 3--7 to 67--120; earlier exact, less fragmented variants regressed 31--36%. Pursue macro-packed rows inside unchanged outer calls instead. |
| call-neutral flat-row fine scorer | **QUALIFIED MICROBENCHMARK; DEFAULT OFF** | Jobs 13266322/13266460 preserve active raw scores, dense scores, all six shared-posterior outputs, poisoned-tail no-op behavior, and outer-call cardinality bitwise. GF46 iteration-20/40/60/80 score-plus-posterior calls are 32.88%, 35.77%, 54.53%, and 32.68% lower, but projection/packing and trajectory costs are not measured. |
| stable physical fine-window shapes | **EXACT PRIMITIVE; FORECAST ONLY; DEFAULT OFF** | Jobs 13264981/13265301 prove logical sizes 68/70/72 bitwise under one physical size 72 and reduce compile identities from three to one. The GF46 29-to-14 signature and 5.2--5.4% net-wall figures are forecasts; score, Wavg, and BPref shapes are not yet integrated together and no trajectory ran. |
| batched CUB sort/scan scratch reuse | **REJECTED; TRAJECTORY SCIENCE FAIL; DEFAULT OFF** | The primitive gate in job 13267397 was downstream-exact and 20.4--26.8% faster, but fail-closed ABBA trajectory job 13268653 rejects promotion: candidate repeat 1 escapes the particle-state envelope at iterations 4/16/18 (particles 285/2902/902), changes current_changes_optimal_offsets_angstrom at iteration 18, and reaches a worst map/control-diameter ratio of 87.94603038052571. The 1.0247x cold and 1.0517x warm observations are non-scoring because correctness failed. |
| batched posterior elementwise exponentiate/divide | **NUMERICALLY EQUIVALENT; END-TO-END GAIN IMMATERIAL/INCONCLUSIVE; DEFAULT OFF** | Primitive job 13269547 is bitwise exact at GF46 iterations 20/40/60/80 and 7.1839--31.6257x faster at the isolated warmed exponentiate-plus-divide boundary. Jobs 13269681/13271166 have zero particle-state and schedule escapes. The strict two-control map-diameter diagnostic flags roundoff-scale relative-L2 differences, but the same-binary rerun moves them to candidate repeat 2 at iterations 2/3 while all four arms use identical CUDA bytes, supporting control-scale numerical variability rather than a basin change. Same-binary warm gain is only 1.0091x, so there is no large reproducible end-to-end advantage and no promotion. |

## Shared EM implementation

| Component | Shared with EM | Qualification |
|---|---:|---|
| coarse projector/scorer | yes | production shared primitive |
| compact active-row planner, fine scorer, and posterior | yes | production shared primitive |
| sequential weighted-average accumulation | yes | production shared primitive |
| radix buckets | yes | production shared primitive |
| ordered-scatter CUDA Graph | yes | candidate is quality-neutral in paired diagnostics |

VDAM calls the shared EM primitives above; it does not carry duplicate projector, scoring, posterior, or ordered-accumulation algorithms.

## Interface and secondary gates

CLI and GUI both default to `relion_fast`; `reference` remains diagnostic. Current typed runtime-control integration `6f39ad52e` passed 18/18 focused checks and defaults sequential CUDA Wavg to `true` and exact-local radix to `4`. K>1 remains unqualified.

| Track | Result | Role | v3 score impact |
|---|---:|---|---:|
| `legacy_parameter_expansion_v2` | 6/15 | `regression` | none |
| `k_greater_than_one` | unqualified | `separate_gate` | none |
| `real_data` | not scored | `separate_gate` | none |

## Next gates

1. Keep the clean tracking-derived batched-lane port default-off. Its 9.03% warm-wall gain is material, but immutable job 13285647 remains a frozen two-repeat strict FAIL. The separately predeclared 8+8 panel `13286397` passes after independently reviewed serializer-only recovery at `f8cea5d209`. Build and seal a same-binary 4+4 true-200 no-growth, basin, final-quality, and end-to-end runtime harness; do not submit the inadequate old 20-iteration ABBA scaffold.
2. Keep direct x-half default-off: it is mathematically qualified and makes finalization 85.40% faster, but finalization is too small and warm wall improves only 2.12%. Preserve the primitive for a future larger fused finalization redesign; do not spend a trajectory on it alone.
3. Keep the shared posterior executor rejected: it is mathematically qualified but saves only 3.37% warm wall while its posterior kernels regress 36.86%.
4. Continue the sealed profile's larger shared lever from default-off Phase 1e `793e3bb12a`. Correctness-only H100 gate `13288282` is independently reviewed PASS. Next run a separately reviewed same-binary/crossed donation runtime and peak-memory A/B, then widen the shared executor beyond call 0. Target the measured 3.344 s excess idle and 53.547 s local compile churn while preserving shared EM/VDAM math and call chronology; do not claim speed or promote the default from the correctness gate.
5. Repeat cache-only arm C across seeds, scales, and representative trajectory checkpoints. Track the 0.365 GiB HWM cost and promote only if the cold/warm gain is reproducible; keep physical-order chunking and the combined B arm out of the production default.
6. Keep the profiler unset. Wire the qualified flat-row scorer behind an explicit default-off typed control, reuse shared compact-pair packing/projection, and time the complete live call. Repeat the raw, dense-score, six-posterior, poisoned-tail, and outer-call exactness audit before any trajectory.
7. Integrate stable physical score, Wavg, and BPref shapes together while retaining the logical cutoff as the runtime bound. Poison-test padded tails and replace the 5.2--5.4% forecast with a live exact compile-amortized trajectory measurement; the isolated runtime-bound BPref primitive is numerically qualified but 14.8--17.7% slower per steady call, so it cannot stand alone.
8. Keep batched CUB and the 80M x-half cap rejected. Keep elementwise default-off and unpromoted: it is numerically equivalent, but the same-binary 1.0091x warm result is immaterial and the isolated primitive speedup cannot authorize promotion.
9. Keep the audited typed Wavg/radix defaults. Preserve the active-particle FAIL@37 boundary, but defer its arithmetic investigation while the explicitly requested performance-first phase is active.
10. Do not revive the rejected 128-tail palette, float32 scorer, eager whole-stack raw cache, shared coarse-projection cache, literal pool-per-call layout, physical-order chunking, or dynamic tail mask without new evidence. After speed closes, isolate correctness and then expand the frozen K=1 matrix before K>1 or real-data promotion.

## Evidence and reproducibility

Frozen suite definition: `docs/math/vdam_k1_full_trajectory_expansion_v3.json` (`9842b2c9cb7646d75127541801ef5982ed19e4a80485f9ce586ceabdb3ed0091`).

- `v3_original` (primary_scientific_evidence): `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_full_expansion_v3_984637b7d_87274be_20260826` at `984637b7db95f1ca6f5800c08ea14c1e32c82c2e`
- `gf43_seeded` (superseding_scientific_evidence): `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf43_full_seeded_accuracy_580477763_87274be_20260826` at `580477763f0f95f028841b074210c4eba34fd24b`
- `gf45_seeded` (superseding_scientific_evidence): `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf45_full_seeded_accuracy_580477763_87274be_20260826` at `580477763f0f95f028841b074210c4eba34fd24b`

Detailed chronological notes remain in `docs/math/em_parity_program.md`; they are not the score source.

Regenerate or validate this dashboard with:

```bash
pixi run python scripts/report_vdam_parity_progress.py --check
```
