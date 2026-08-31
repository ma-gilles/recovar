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
| Runtime | **INCONCLUSIVE** | Cold and warm observations conflict; the corrected warm retry is cross-GPU and trajectories differ. No speedup or regression claim. |
| Immediate work | **QUALIFY TOPOLOGY CANDIDATES** | Keep profiling and bounded raw reuse off; test x-half sizing, shared coarse-projection reuse, and pool-preserving local buckets. |

## At a glance

| Axis | Authoritative state | Current engineering read |
|---|---|---|
| K=1 correctness | **2/20** strict cases pass | Frozen; no recent diagnostic changes this score. |
| Runtime | **0/20** cases meet 1.10x; observed 4.91--11.58x | Warm80 timing is INCONCLUSIVE; exact-local topology, pass-1 orchestration, and repeated input work dominate the remaining wall gap. |
| EM reuse | Shared production arithmetic | The remaining gap is execution topology, not duplicate scoring math. |
| Later gates | K>1 unqualified; real data not scored | Kept separate until K=1 correctness and runtime close. |

### Gate progression

| Gate | Status | Evidence |
|---|---|---|
| `13252518`: 20-iteration `vdam-gf46` | **PARTICLE TRAJECTORY EXACT** | All 3,000/3,000 pose/translation states match at every iteration; first divergence is `null`; requested/effective Wavg=`true`, radix=`4`. Wall 177 s; pre-artifact 158.846 s. 177 s is below the 182 s accepted short baseline but above the 169.40 s prior exact-control result; these are cross-run H100 observations, not a paired speed result. Evidence: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf46_typed_runtime20_6f39ad52e_20260831/provenance/particle_state_audit_it001_020.json`. |
| `13253088` + audit `13256248`: 80-iteration typed gate | **MIXED: POLICY/MAP PASS; PARTICLE FAIL@37; RUNTIME INCONCLUSIVE** | Original profiled repeat stopped at iteration 64 (invalid harness); corrected job `13254470` completed, with direct map/particle FAIL@4, but is cross-GPU diagnostic only. Frozen scores stay 2/20 correctness and 0/20 runtime. |

### Typed warm80 audit evidence

| Artifact | Path | SHA-256 |
|---|---|---|
| `typed_policy` | `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf46_typed_runtime_warm80_6f39ad52e_20260831/repeat-01/analysis/typed_runtime_gate/typed_runtime_policy_it001_080.json` | `0c6c60df5c841ebbdbd54d67df681d6a5c2c1229d0657315d4add08d663f28f5` |
| `runtime` | `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf46_typed_runtime_warm80_6f39ad52e_20260831/repeat-01/analysis/typed_runtime_gate/runtime_profile_comparison.json` | `5ffa9ae565466aa924eccd1a836a3a5af1613df40cf4c584b344ec3dd4494165` |
| `same_gpu_map` | `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf46_typed_runtime_warm80_6f39ad52e_20260831/repeat-01/analysis/typed_runtime_gate/map_typed_r01_native4_envelope_it000_080.json` | `4a2fa726dd2ee7e491983cd67e6212211dc8022aa89a9510e89495035ea1010a` |
| `same_gpu_particle` | `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf46_typed_runtime_warm80_6f39ad52e_20260831/repeat-01/analysis/typed_runtime_gate/particle_typed_r01_native4_envelope_it001_080.json` | `a7772c546379ac414a511fbcd15907144de264da18677ad6de41c8bec03c47ba` |
| `retry_map` | `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf46_typed_runtime_warm80_6f39ad52e_20260831/repeat-01/analysis/typed_runtime_gate/map_typed_r02_native4_envelope_it000_080.json` | `473b650ddf4f9a261514529f14d54fd6306ddb1580a600fbce203254e09f977c` |
| `retry_particle` | `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf46_typed_runtime_warm80_6f39ad52e_20260831/repeat-01/analysis/typed_runtime_gate/particle_typed_r02_native4_envelope_it001_080.json` | `36b0da26550a6e98d2eb0917cb0b12cdc62bca5b8843abe72f4458d9bb77d0f7` |
| `exit_statuses` | `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf46_typed_runtime_warm80_6f39ad52e_20260831/repeat-01/analysis/typed_runtime_gate/cpu_audit_exit_statuses.txt` | `a29bc69e9e1fa0ec5de5211e8b1cbe1cdb673816c8d1a58deb15fe36ba2b05e0` |
| `slurm_log` | `/scratch/gpfs/GILLES/mg6942/slurmo/vdam-typed80-audit-13256248.out` | `0004a1cf9d67b1638883ba3a1f55edffa6cb265d10971db637c8a89ed3ca44f3` |

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
| ordered-scatter CUDA Graph `13203664` | 291 vs 293 s (-0.68%) | particles 80/80; maps 81/81. | **QUALIFIED CANDIDATE** |

The current warm80 comparison cannot establish a speedup or regression. The tail-mask microbenchmark does not target the dominant accepted production coarse kernel and is superseded by the full paired gate. None of these diagnostics can change the frozen correctness or runtime panels.

Palette evidence: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf46_sig_bucket_ab_4846bd5c5_20260831/FAILED.json` (SHA-256 `91754058b814633ea39fd3f8f958a3765d48bd21575cc69be489765657da92af`); Slurm log `324bb62a9e257fb8b9871462d99bcb2cbda9750a021db6809b63f85ea088fc7a`. Tail-pair evidence: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf46_coarse_tail_pair_51357fbec_20260831/pair_summary.json` (SHA-256 `fe00d3936fefa751a676ee8b3b52b262770e50e865d2716a1512a26076d4b1ba`); Slurm log `8e27a0f48d42c072604b326d4ee96e6052b1a794715795f5a9662e5f17f38699`.

### Warm H100 profile: job `13248509`

| Profile slice | Time | Readout |
|---|---:|---|
| iterations 47--80 | 240.36 s wall / 44.58 s kernels | GPU kernels explain only part of wall time. Artifact: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf46_warm_nsys_v3_6b5e6568a_20260831`. |
| shared coarse projector | 35.899 s | 80.53% of GPU kernel time, but only a bounded fraction of end-to-end wall; not the bulk gap. |
| exact-local / pass 1 wall | 130.09 / 71.39 s | Globally padded rectangular scheduling is the main topology target. |
| XLA compile ranges | 57.83 s | Shape-policy stability matters before arithmetic changes. |
| dataset getitem / disk read | 31.90 / 25.36 s | Input latency is material, but the exact eager cache experiment regressed. |
| fine fused / Wavg | 1.897 / 0.556 s | Already small; optimize only without repeating projections. |

### Production profiler toggle: job `13258895`

| Crossed pair | Profiler off | Profiler on | Readout |
|---|---:|---:|---|
| pair 1 | 80.067 s | 91.570 s | on +14.37% |
| pair 2 | 121.944 s | 123.108 s | on +0.95% |

All 20/20 pose/translation checkpoints match across arms. Keep profiling unset in production. The late-arm input/pass-1 slowdown confounds the exact magnitude, so the observed median ratio is diagnostic only. Evidence: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf46_profile_toggle_pair_52c38c85b_20260831/pair_summary.json` (SHA-256 `2eab2b313f7484407ed639d97c149f7afc8cefba1cb19a490bf8103ea316f778`).

### Bounded pass-to-pass raw cache: job `13260861`

| Iteration sample | Control slice | Cached slice | Change |
|---:|---:|---:|---:|
| 1 | 0.1473 s | 0.0859 s | -41.7% |
| 9 | 0.1462 s | 0.0856 s | -41.4% |
| 20 | 0.1469 s | 0.0859 s | -41.5% |

All 15/15 raw, metadata, and masked/unmasked CUDA-preprocess blocks are bitwise exact. The absolute ceiling is only about 4.9 s over 80 iterations, while the first clean full pair was +1.45% slower. **Default off; not promoted.** Evidence: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_between_pass_cache_microbench_gf46_90ce10d61_20260831_retry02/summary.json` (SHA-256 `a9423c50358c80b8aacb87d9ae61f9596785f80284c16e16be49c8ef6cc9619e`).

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

1. Keep the profiler unset and finish the exact-local x-half bucket-size sweep. Require exact particle states/maps, the RELION envelope, repeatable warmed timing, and arm-resolved peak memory.
2. Keep bounded pass-1 to pass-2 raw reuse default-off. Audit whether consecutive RELION pools of three can use pool-local radix buckets without changing particle order or accumulation chronology; run GPU science only if the layout-only padded-row reduction is material.
3. Qualify shared coarse-projection reuse only if its standalone output is strictly bitwise and its end-to-end production gain is measurable; the shared coarse primitive is not the bulk wall-time gap.
4. Keep the audited typed Wavg/radix defaults. Preserve the active-particle FAIL@37 boundary, but defer its arithmetic investigation while the explicitly requested performance-first phase is active.
5. Do not revive the rejected 128-tail palette, float32 scorer, eager raw cache, exact-local projection cache, or nondefault dynamic tail mask without new evidence. After speed closes, isolate correctness and then expand the frozen K=1 matrix before K>1 or real-data promotion.

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
