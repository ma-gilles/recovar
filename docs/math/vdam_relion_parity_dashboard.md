# RECOVAR / RELION VDAM parity dashboard

> **Authoritative release status: NOT READY.** Frozen v3 K=1 correctness is
> **2 / 20** and frozen runtime parity is **0 / 20**. The only accepted v3
> cases are `vdam-gf44` and `vdam-gf45`. Diagnostics, performance primitives,
> the legacy v2 expansion, K>1, and real data cannot change those scores.

## Live engineering snapshot — 2026-09-02

| Signal | Status | Evidence / next decision |
|---|---|---|
| Release score | **NOT READY — correctness 2 / 20; runtime 0 / 20** | Frozen v3 is unchanged. Component gates and diagnostics cannot inflate it. |
| Corrected GF46 checkpoint | **COMPLETE DATA GREEN / SEALED RERUN NEXT** | Job `13343052`, frozen iteration `180 -> 181`: zero novel hybrid support rows, exact cutoff/discrete/STAR/model identities, and every hybrid map/model delta inside the pooled direct-repeat numerical envelope. The job stopped only in its obsolete post-analyzer and is not promoted as sealed evidence. |
| Root cause closed | **SHARED EM/DIRECT OPERANDS** | The pre-fix hybrid cache used `mask_current_image_disk=False` while mature EM/direct used `True`. Commit `e9a8e8256` now routes both through one shared projection helper; the previous 405 pose-assignment mismatches fell to **0 / 1,000**. |
| Speed result | **4.38x WARM TRANSITION SPEEDUP** | Clean audit-off ABBA medians: wall `21.267882 -> 4.852816 s`; expectation `20.640134 -> 4.172020 s` (**4.95x**); pass 1 `18.787344 -> 2.309944 s` (**8.13x**); pass 2 is unchanged (`1.110826 -> 1.121583 s`). Peak RSS `3.325 -> 3.508 GiB`. |
| Numerical classification | **CONTROL-BOUNDED FP32 NOISE** | Direct itself changed one inclusive threshold-tie support at row 412 while persisted cutoff counts and downstream discretes stayed exact. Hybrid introduced no support outcome outside the two direct observations. Across six direct and six hybrid executions, all 15 hybrid-repeat and 36 crossed map/model pairs pass the mature EM-style `2x` pooled-control envelope. |
| Focused regression | **49 / 49 PASS** | Hybrid significance, InitialModel diagnostic propagation, shared projection operands, and transition analyzer/harness tests pass. No broad RECOVAR suite was run. |

### Immediate queue

1. Rerun the same pinned GF46 direct/hybrid audit plus clean ABBA with the corrected v2 analyzer and seal the result.
2. If the sealed rerun reproduces this boundary, run the representative repeat/no-growth trajectory.
3. Then rerun frozen K=1 full trajectories and expand across outliers, pose/noise distributions, scale, parameters, and long trajectories. K>1 and real data remain separate later gates.

## At a glance

| Track | Score or status | Current boundary |
|---|---:|---|
| Frozen v3 K=1 correctness | **2 / 20** | Release gate; unchanged. |
| Frozen v3 runtime | **0 / 20** | Independent release gate; unchanged. |
| Legacy v2 expansion | **6 / 15** | Regression track only; no v3 score impact. |
| Current correctness work | **CORRECTED CHECKPOINT DATA GREEN / SEAL OPEN** | Job `13343052`: exact downstream state and zero novel hybrid support outcomes after sharing mature EM/direct operands; analyzer-only superseding rerun pending. |
| Current performance work | **4.38x CHECKPOINT SIGNAL / SEAL OPEN** | Clean ABBA wall `21.267882 -> 4.852816 s`; pass 1 is 8.13x faster and pass 2 unchanged. Frozen runtime stays 0/20 pending trajectory scoring. |
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

## Correctness: current hybrid-GEMM blocker

The expanded shared-EM GEMM path is roughly **4.5x faster** in the sealed GF46
one-transition coarse comparisons, but it is unsafe as the sole scorer:
standalone GEMM arms changed downstream discrete state and escaped the direct
repeat map/model envelopes. A valid hybrid must use GEMM only to pre-screen,
then direct-rescore every source block capable of changing RELION's float32 raw
winner, offset, posterior, support, or selected state.

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
| Flat-row scorer | `13266322`, `13266460` | Bitwise primitive pass; default off pending full packing/projection cost. |
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

1. On all 1,000 frozen GF46 iteration-181 particles, compare the integrated
   hybrid against direct scoring for conservative block coverage, fallbacks,
   exact candidate support/winners, particle state, and map/model envelopes.
2. Measure direct/hybrid cold and warm wall on the same pinned H100, with
   diagnostics disabled in timing arms and selector, projection/cache,
   exact-rescore, posterior, and fallback costs reported separately.
3. If that one-transition gate passes with a material end-to-end gain, run a
   representative repeat/no-growth trajectory and multi-dataset basin gate.
4. Then run the frozen K=1 trajectory suite and the expanded outlier,
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
