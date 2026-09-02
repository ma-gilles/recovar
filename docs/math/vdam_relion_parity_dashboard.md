# RECOVAR / RELION VDAM parity dashboard

> **Authoritative release status: NOT READY.** Frozen v3 K=1 correctness is
> **2 / 20** and frozen runtime parity is **0 / 20**. The only accepted v3
> cases are `vdam-gf44` and `vdam-gf45`. Diagnostics, performance primitives,
> the legacy v2 expansion, K>1, and real data cannot change those scores.

## At a glance

| Track | Score or status | Current boundary |
|---|---:|---|
| Frozen v3 K=1 correctness | **2 / 20** | Release gate; unchanged. |
| Frozen v3 runtime | **0 / 20** | Independent release gate; unchanged. |
| Legacy v2 expansion | **6 / 15** | Regression track only; no v3 score impact. |
| Current correctness work | **BLOCKED** | GEMM is a fast pre-screen, not a safe scorer by itself; direct rescoring must certify both raw/pre-prior and posterior/support decisions. |
| Current performance work | **DIAGNOSTIC** | Selected-block direct rescoring is qualified as a primitive, but no hybrid production path or end-to-end runtime result is accepted. |
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

1. Replace pair-only TopK retention with per-source-block maxima while keeping
   the dual raw/pre-prior and posterior/support v2 certificate fail-closed.
2. On all 1,000 GF46 particles, prove conservative block coverage or trigger
   full-direct fallback; direct-rescore the certified union with the qualified
   job-`13328717` primitive and recompute RELION float32 offset/posterior/support.
3. Require exact or repeat-envelope-safe one-transition decisions and maps
   before measuring a paired same-H100 hybrid runtime. Report selector,
   projection/packing, direct-rescore, and fallback costs separately.
4. If the one-transition gate passes with a material end-to-end gain, run a
   representative repeat/no-growth trajectory and multi-dataset basin gate.
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

Job `13329608` artifacts are under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_coarse_gemm_gf46_stream_v2_6e4e0ae65_h21g4_20260901/`
and are marked `SAFE_TO_DELETE`.

## Code references

- `recovar/em/dense_single_volume/helpers/coarse_gemm_streaming.py:CoarseGemmStreamingState`
- `recovar/em/dense_single_volume/helpers/coarse_gemm_streaming.py:summarize_coarse_gemm_streaming_state`
- `recovar/em/dense_single_volume/helpers/coarse_gemm_streaming.py:aggregate_coarse_gemm_streaming_summaries`
- `recovar/em/dense_single_volume/helpers/significance.py:_score_relion_coarse_gaussian_gemm_macro`
- `recovar/em/dense_single_volume/helpers/significance.py:_seal_coarse_gaussian_gemm_streaming_scope`
- `recovar/em/initial_model/dense_adapter.py:_initial_model_coarse_gemm_diagnostic_scopes`
- `recovar/cuda_backproject.py:relion_coarse_diff2_rotation_blocks_f32`
- `recovar/cuda/cuda_backproject.cu:launch_relion_coarse_diff2_rotation_blocks_f32`
- `scripts/run_vdam_coarse_gemm_gf46_streaming_selector.sbatch`
- `scripts/run_vdam_coarse_rotation_blocks_primitive_gate.sbatch`
