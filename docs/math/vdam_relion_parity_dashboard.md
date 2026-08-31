# RECOVAR / RELION VDAM parity dashboard

> **Authoritative v3 status — NOT READY.** Strict K=1 correctness is **2 / 20** and runtime parity is **0 / 20**. The only accepted cases are `vdam-gf44, vdam-gf45`.
>
> This page is generated from the frozen 20-case, iteration 0--200 scorecard. Scheduler diagnostics, the legacy v1/v2 tracks, K>1, and real data cannot change this score.

## Current action

| Gate | Result | What it means now |
|---|---|---|
| Typed Wavg/radix policy | **PASS 80/80 in both arms** | Requested/effective sequential Wavg=`true`; radix=`4`. |
| Same-GPU map envelope | **PASS 81/81** | Minimum best-native FSC AUC `0.9999885424`; no checkpoint outside. |
| Active-particle envelope | **FAIL@37 — OPEN** | 35/80 checkpoints fail; 360/360 active particles are unmatched at iteration 80. |
| Runtime | **INCONCLUSIVE** — 0/20 | No candidate has a qualified trajectory/runtime pair. Flat-row and batched-CUB calls are exact at their downstream gates and materially faster only at call level; stable windows remain forecast-only. |
| Immediate work | **WIRE FLAT ROW DEFAULT-OFF** | Include shared packing/projection cost in one live call, then gate the same active outputs. Replay captured batched-CUB operands, integrate stable score/Wavg/BPref shapes together, then run focused trajectory A/B gates. |

## At a glance

| Axis | Authoritative state | Current engineering read |
|---|---|---|
| K=1 correctness | **2/20** strict cases pass | Frozen; no recent diagnostic changes this score. |
| Runtime | **0/20** cases meet 1.10x; observed 4.91--11.58x | Flat-row score-plus-posterior calls are 32.68--54.53% lower and batched CUB is 20.4--26.8% lower, but full-stage, memory, and trajectory runtime remain unmeasured. |
| EM reuse | Shared production arithmetic | The remaining gap is execution topology, not duplicate scoring math. |
| Later gates | K>1 unqualified; real data not scored | Kept separate until K=1 correctness and runtime close. |

## Runtime workboard

| Lane | State | Exactness | Performance readout | Next gate |
|---|---|---|---|---|
| Flat-row scorer (`13266322/13266460`) | **QUALIFIED MICROBENCH; DEFAULT OFF** | Active raw, dense scores, posterior 6/6, poison tail, and calls bitwise | Combined call reduction at it20/40/60/80: 32.88%, 35.77%, 54.53%, 32.68%; packing/projection not timed | Default-off live call with packing/projection, then exact boundary; no trajectory yet. |
| Stable fine window (`13264981/13265301`) | **EXACT PRIMITIVE; FORECAST ONLY** | Logical [68, 70, 72] under physical 72 bitwise; compile identities 3->1 | GF46 signatures 29->14; forecast 5.2--5.4% only | Integrate score/Wavg/BPref shapes together; no trajectory yet. |
| Batched CUB (`13266477/13266811/13267179/13267397`) | **DOWNSTREAM EXACT; TRAJECTORY NEXT; DEFAULT OFF** | Sort, threshold index/value, support mask, and n-significant bitwise at all four shapes; raw scan is natively variable | 20.4--26.8% lower; minimum 1.2565x | Captured operands, then control/control/candidate trajectory A/B; no promotion yet. |
| 80M x-half (`13260950/13265965`) | **REJECTED / UNQUALIFIED** | iteration-1 topology is identical but 3/3 artifacts already differ; causal projection effect not proved | Prior same-H100 wall was 7.56% lower, but unusable for promotion | Revisit only with one-process replay from a byte-identical frozen state. |

All four lanes are diagnostic, default-off/unwired, and have **no impact** on frozen correctness 2/20 or runtime 0/20.

### Gate progression

| Gate | Status | Evidence |
|---|---|---|
| `13252518`: 20-iteration `vdam-gf46` | **PARTICLE TRAJECTORY EXACT** | All 3,000/3,000 pose/translation states match at every iteration; first divergence is `null`; requested/effective Wavg=`true`, radix=`4`. Wall 177 s; pre-artifact 158.846 s. 177 s is below the 182 s accepted short baseline but above the 169.40 s prior exact-control result; these are cross-run H100 observations, not a paired speed result. Evidence: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf46_typed_runtime20_6f39ad52e_20260831/provenance/particle_state_audit_it001_020.json`. |
| `13253088` + audit `13256248`: 80-iteration typed gate | **MIXED: POLICY/MAP PASS; PARTICLE FAIL@37; RUNTIME INCONCLUSIVE** | Original profiled repeat stopped at iteration 64 (invalid harness); corrected job `13254470` completed, with direct map/particle FAIL@4, but is cross-GPU diagnostic only. Frozen scores stay 2/20 correctness and 0/20 runtime. Evidence SHA-256: policy `0c6c60df5c841ebbdbd54d67df681d6a5c2c1229d0657315d4add08d663f28f5`, runtime `5ffa9ae565466aa924eccd1a836a3a5af1613df40cf4c584b344ec3dd4494165`, map `4a2fa726dd2ee7e491983cd67e6212211dc8022aa89a9510e89495035ea1010a`, particle `a7772c546379ac414a511fbcd15907144de264da18677ad6de41c8bec03c47ba`. |

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

## Speed snapshot

| Experiment | Timing readout | Exactness / scope | Decision |
|---|---|---|---|
| typed warm80 audit `13256248` | cold 704 vs 644 s (+9.3%); warm 415 vs 423 s (-1.9%) | Same hard state only through iterations 1--36; warm is cross-GPU. | **INCONCLUSIVE — no speed claim** |
| 128-tail palette `13254010` | warm wall +4.1%; pass 1 +7.3% | 20/20 particle states, then sealed cache changed 1277->1280 files. | **REJECTED / DEFAULT OFF** |
| dynamic tail mask `13257087` | 65.1455 vs 65.5981 s = 1.00695x | Forced nondefault native-texture path; 120/120 strict artifacts differ. Microgates: active-3 23.91x bitwise; active-200 2.317x not bitwise. | **VALID SCIENCE FAIL / DO NOT PROMOTE** |
| shared coarse-projection cache `13261042 / 13261159 / 13261300 / 13261339` | exact at every tested batch; `1.01115x` at batch 200 | GF46 has 1 image batch per pass; loose overall ceiling `0.52%`, observed iteration contribution about `0.10%`; retains `0.99--1.5 GiB`. | **EXACT BUT IMMATERIAL; NOT INTEGRATED** |
| x-half projection batch `13260950` | median 310.949 -> 287.444 s (7.56% lower); both pairs faster | Peak memory unchanged, but operand job 13265965 found iteration-1 trajectory differences before cap topology differs; no causal x-half invariant. | **REJECTED / CAUSAL PROOF UNQUALIFIED** |
| literal pool-local buckets `13262146` | logical padded rows -49.5% to -82.2% | Exact source chronology, but calls [7, 3, 4, 5] -> [67, 67, 67, 120]; earlier less-fragmented exact variants were 31--36% slower. | **REJECTED BEFORE GPU PAIR** |
| ordered-scatter CUDA Graph `13203664` | 291 vs 293 s (-0.68%) | particles 80/80; maps 81/81. | **QUALIFIED CANDIDATE** |

The x-half pair measured a runtime opportunity but failed the required causal invariant: all three iteration-1 artifacts differ even though both caps predict identical iteration-1 topology. Commit `732868abb` rejects the lane as unqualified; it does not attribute later projector differences to the cap. None of these diagnostics can change the frozen correctness or runtime panels.

Palette evidence: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf46_sig_bucket_ab_4846bd5c5_20260831/FAILED.json` (SHA-256 `91754058b814633ea39fd3f8f958a3765d48bd21575cc69be489765657da92af`); Slurm log `324bb62a9e257fb8b9871462d99bcb2cbda9750a021db6809b63f85ea088fc7a`. Tail-pair evidence: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf46_coarse_tail_pair_51357fbec_20260831/pair_summary.json` (SHA-256 `fe00d3936fefa751a676ee8b3b52b262770e50e865d2716a1512a26076d4b1ba`); Slurm log `8e27a0f48d42c072604b326d4ee96e6052b1a794715795f5a9662e5f17f38699`.

### Warm H100 profile: job `13248509`

Iterations 47--80: 240.36 s wall, 44.58 s kernels, 35.899 s coarse projector, 57.83 s XLA compile, and 31.90 s dataset getitem. The profile points to execution topology and shape churn; full artifact paths and hashes remain bound in the JSON ledger.

### Archived speed gates

| Gate | Compact decision |
|---|---|
| profiler `13258895` | Keep unset; both crossed pairs favored off, but contention forbids a magnitude claim; 20/20 particle checkpoints exact. |
| pass-to-pass raw cache `13260861` | 15/15 blocks exact, but only ~4.9 s ceiling and the full pair was +1.45% slower; default off. |
| shared coarse cache `13261042/13261159/13261339` | Exact but only 1.01115x at batch 200 and <0.52% end-to-end ceiling; not integrated. |
| x-half `13260950/13265965` | 7.56% lower wall, but preboundary state was noninvariant; rejected/unqualified at `732868abb`. |
| literal pool `13262146` | Rows fell 49.5--82.2%, but calls rose to [67, 67, 67, 120]; rejected before GPU pair. |

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
| batched CUB sort/scan scratch reuse | **DOWNSTREAM-EXACT MICROGATE; TRAJECTORY NEXT; DEFAULT OFF** | Job 13267397 preserves all sorts, threshold indices/values, support masks, and significant counts bitwise at the four GF46 shapes while reducing call time 20.4--26.8% (minimum 1.2565x). Iteration-20 normalized drift stays inside scalar self-repeat variability; iterations 40/60/80 are exact through normalized/reconstruction weights. Captured operands and a control/control/candidate trajectory A/B remain required before promotion. |

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

1. Keep the profiler unset. Wire the qualified flat-row scorer behind an explicit default-off typed control, reuse shared compact-pair packing/projection, and time the complete live call. Repeat the raw, dense-score, six-posterior, poisoned-tail, and outer-call exactness audit before any trajectory.
2. Advance batched CUB to actual captured posterior operands, then a control/control/candidate trajectory A/B. Job 13267397 already preserves sort, threshold, support, and significant-count outputs bitwise and reduces call time 20.4--26.8%; keep it default-off until both higher gates pass.
3. Integrate stable physical score, Wavg, and BPref shapes together while retaining the logical cutoff as the runtime bound. Poison-test padded tails and replace the 5.2--5.4% forecast with a live exact boundary measurement before any trajectory.
4. Keep the 80M x-half cap rejected. Job 13265965 did not provide a causal invariant, so do not implement the accumulation split without a one-process replay from byte-identical frozen incoming state.
5. Keep the audited typed Wavg/radix defaults. Preserve the active-particle FAIL@37 boundary, but defer its arithmetic investigation while the explicitly requested performance-first phase is active.
6. Do not revive the rejected 128-tail palette, float32 scorer, eager raw cache, shared coarse-projection cache, literal pool-per-call layout, or dynamic tail mask without new evidence. After speed closes, isolate correctness and then expand the frozen K=1 matrix before K>1 or real-data promotion.

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
