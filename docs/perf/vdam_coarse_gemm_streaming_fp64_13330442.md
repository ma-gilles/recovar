# VDAM GF46 all-particle promoted-FP64 streaming diagnostic

Status: **DIAGNOSTIC ONLY / NO PRODUCTION PROMOTION**.  Promoted FP64 remains
default-off.  This run measures whether the higher-precision expanded-square
surface materially shrinks the conservative direct-rescore set; it does not
qualify standalone GEMM scoring, trajectory correctness, or runtime parity.

## Immutable run

- Slurm job: `13330442` (`COMPLETED`, exit `0:0`, wall `59 s`)
- Node / GPU: `della-h21g4` / pinned H100
  `GPU-099c0d77-bb85-f2e9-f628-148b733c9176`
- Source / tree: `2c1a9e299e40563f2f5058dce490231c39fb7a12` /
  `1209819d79e9650f33656436f1ae0a66477d1c37`
- Source-manifest SHA-256:
  `7d66670221449b7162162beeff2f6552b614471f0d996f9787dd3c4818547f01`
- Frozen transition: GF46 iteration `180 -> 181`, K=1, all 1,000 selected
  particles, 1,069,056 candidates per particle
- Result root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_coarse_gemm_gf46_stream_fp64_v2_2c1a9e299_h21g4_20260901`
- Identity-certificate SHA-256:
  `3a666ea737cedc2aabbaa033a1a226058004ae9983adeb4d3c05f1b95551d5ce`
- Aggregate-manifest SHA-256:
  `faa883a1c6fa756adf679c7138c4b38e7135f8cc0aaac89c2d181b7d10f0a754`

The root contains `SAFE_TO_DELETE`, `IDENTITY_VALIDATED`, and `COMPLETED` and
was sealed read-only by the runner.

## Result

| Boundary | Promoted-FP64 result | Float32 all-1,000 reference |
|---|---:|---:|
| Finite paired candidates | `1,069,056,000`; 0 nonfinite | same |
| Maximum absolute direct delta | `1.5` | `3.3125` |
| Weighted RMS delta | `0.03593275` | `0.05107886` |
| Signed mean delta | `-0.00057747` | `+0.00789002` |
| Raw safe blocks | median `1`, p95 `2`, max `3` | median `1`, p95 `2`, max `3` |
| Pair-TopK posterior coverage | `732 / 1,000` | `727 / 1,000` |
| Covered posterior/union blocks | median / p95 / max `6 / 6 / 6` | `6 / 6 / 6` |
| Support | one false positive, zero false negatives | same |

The critical near tie is **not repaired** by FP64.  Original particle `1933`
still chooses direct candidate `1052758` while the promoted macro surface ties
at its maximum and reports candidate `98994`; the direct winner margin is only
`0.000732421875`.  Original particle `636` retains the same winner but has one
extra macro-support candidate and zero omissions.  Exact selected-block direct
rescoring is therefore still required.

The old pair-retention state still saturates: TopK=2048 certifies only 732
particles.  This is not a source-block-capacity failure.  Every covered
posterior union fits in six 16-rotation blocks, and the raw track fits in at
most three.  Complete per-source-block interval maxima remain the appropriate
production selector.

## Decision

The empirical selector topology is effectively unchanged by promotion because
RELION's 138-score posterior span dominates the observed score-error reduction.
FP64 is nevertheless still a viable *certificate arithmetic* candidate: the
sealed clean timing gate in job `13327874` measured a 4.512x coarse comparison
speedup, and a formal interval around the live FP64 expanded score avoids the
large cancellation term in a conservative FP32-GEMM error bound.  The next
gate must compare formal FP32 and FP64 interval widths and total hybrid wall
time, including projection, interval reduction, selected-block direct rescore,
and fallback rate.  Neither arithmetic mode may be promoted from this run.

Frozen v3 K=1 correctness remains `2 / 20`, and frozen runtime remains
`0 / 20`.
