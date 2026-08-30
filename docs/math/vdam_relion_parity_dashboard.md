# RECOVAR / RELION VDAM parity dashboard

> **Authoritative v3 status — NOT READY.** Strict K=1 correctness is **2 / 20** and runtime parity is **0 / 20**. The only accepted cases are `vdam-gf44, vdam-gf45`.
>
> This page is generated from the frozen 20-case, iteration 0--200 scorecard. Scheduler diagnostics, the legacy v1/v2 tracks, K>1, and real data cannot change this score.

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

## Speed snapshot

Ordered-scatter CUDA Graph candidate `6b5e6568a` (paired job `13203664`) ran in 291 s versus 293 s (-0.68%). Ordered backprojection fell from 39.447 s to 36.660 s (-7.07%).

Quality-neutral candidate/control checks: particles 80/80; maps 81/81. The separate native-particle envelope remains 23/80. This performance snapshot cannot change the frozen correctness or runtime panels.

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

CLI and GUI both default to `relion_fast`. The `reference` mode is diagnostic. The typed policy is `36103aaa2` with 28/28 focused checks; K>1 remains unqualified.

| Track | Result | Role | v3 score impact |
|---|---:|---|---:|
| `legacy_parameter_expansion_v2` | 6/15 | `regression` | none |
| `k_greater_than_one` | unqualified | `separate_gate` | none |
| `real_data` | not scored | `separate_gate` | none |

## Current hypothesis and next gate

Job 13211719 rejects same-cache reuse as sufficient for exact ordered-shell repeatability: the earliest captured difference is one float32 ULP in an E-step max-posterior scalar.

Ordered-shell head `3b5afd98e` follows pre-diagnostic head `94bc7d890`; pair-stable head `ee673be1f` follows prior profile-matched head `39a38f9d6` (harness fix `381bf7949`). Evidence: **A populated the canonical cache 0->377; A-after, B-before, and B-after share manifest SHA-256 2703b7d6fdbc0d329407e20574a90791a15af59c6dc882bc2e617069e28ef3d5 with no B additions or changes.** Cache identity is proven. Pose, rotation, translation, and class selection remain exact while the one-ULP posterior difference feeds ordered noise and both-half BPref differences. The next discriminator remains compile/autotune variant versus runtime reduction at or before this E-step scalar. No cache-disable or production arithmetic change is authorized by this snapshot.

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
