# VDAM final-support dense-order diagnostic — H100 job 13366669

## Decision

Reject this revision at the noise-quality gate, while retaining its default-off
implementation as evidence for the next operand-reuse revision. Scattering the
489 projected final-support rows back into the original 62,208-row dense layout
and then calling the mature EM reductions preserves all discrete results and is
fast when warm. It does not restore direct noise statistics. Aggregate A2/XA
diagnostics show that both operands already differ, so dense reduction shape is
not the root cause.

## Qualification

| Field | Value |
|---|---|
| Source | `0e1b4355c85b024d45b4782bdca6f7eba2656223` |
| Slurm | `13366669` (`COMPLETED`, exit `0:0`, elapsed `00:05:53`) |
| Hardware | `della-h19g2`, NVIDIA H100 80GB HBM3, `GPU-235ec3bc-ca9f-1c0e-88eb-c8b37c5e0480` |
| Boundary | One exact in-memory GF46 iteration-34 state, iteration `34 -> 35` |
| Panel | direct / packed deferred / packed deferred / direct |
| Diagnostics | Aggregate `wsum_noise_a2` and `wsum_noise_xa`; production big JIT retained |
| Peak monitored GPU memory | 17,579 MiB across the complete ABBA process |

The candidate remains guarded by the flat-row, packed-projection, exact/source-
faithful RELION x-half BPref, and
`RECOVAR_INITIAL_MODEL_DEFER_PACKED_VDAM=1` switches. All are off by default.

## Science result

- Every tracked pose, translation, class, posterior, significance, particle-
  state, sampling-state, and complete support-audit field is exactly equal.
- Reconstruction/state differences other than noise stay within the ordinary
  direct/direct CUDA atomic repeat envelope. For example, candidate/direct
  `Igrad2` normalized L2 is `2.66561e-6`, versus `2.66568e-6` direct/direct.
- Candidate repeat stability is good: final `sigma2_noise` normalized L2 is
  `9.21e-9`, comparable to `7.44e-9` direct/direct.
- Candidate/direct final `sigma2_noise` normalized L2 is still `1.89114e-5`,
  with maximum absolute delta `1.34949e-7`.

The split sufficient statistics isolate the failed boundary:

| Raw statistic, direct 2 vs candidate 2 | Normalized L2 | Maximum absolute delta | Direct/direct normalized L2 |
|---|---:|---:|---:|
| `wsum_noise_a2` | `2.62281e-4` | `264,768` | `8.26745e-8` |
| `wsum_noise_xa` | `2.13288e-4` | `303,584` | `7.57088e-8` |
| `wsum_sigma2_noise` | `3.67604e-4` | `579,792` | `9.69146e-8` |
| `wsum_img_power` | `1.33033e-7` | `1,024` | `1.03259e-7` |

Because A2 and XA both move while image power, posterior decisions, support,
and dense topology remain at repeat scale, the previous zero-row-reduction
theory is rejected. The remaining difference is upstream in the separately
materialized projection and/or denominator operands. The direct and candidate
already use the same denominator-only CUDA primitive, while packed-projection
job `13363465` retained repeat-scale noise when it consumed the projection made
inside the scoring JIT. The strongest next test is therefore to return and
reuse that exact packed union projection instead of reprojecting final support.

## Runtime result

| Metric | Warm direct | Warm packed deferred | Change |
|---|---:|---:|---:|
| Whole iteration | 2.105864 s | 2.060737 s | **-2.14%** |
| Shared local EM | 1.208894 s | 0.942125 s | **-22.07%** |
| Local big JIT | 1.080519 s | 0.700809 s | **-35.14%** |
| Deferred local noise | 0.000000 s | 0.111254 s | +0.111254 s |
| Final accumulator | 0.015201 s | 0.014561 s | -4.21% |

The candidate scores 38,016 packed rows rather than 62,208 dense rows and
backprojects 489 nonzero posterior rows. Its first invocation compiled the new
dense-order noise boundary; only the warmed ABBA pair is timing evidence. The
noise failure prevents promotion regardless of the speed result.

## Provenance

- Report JSON SHA-256:
  `ec6ded886dff2465a536b2c6c987092d91d40beac68b3cade57eac712764cf78`
- Qualified CUDA checksum-file SHA-256:
  `ff62e734a1fa98ad6fac10b7c2579da327aa53236a1c9351b166ae5461b62bd4`
- Artifact manifest SHA-256:
  `631387c14121d42e7c428fc712181790acc2a7049c0d06bf44a02389a04febb9`
- Static-input checksum-file SHA-256:
  `f513c2e152a70b44dc5ec82778e379b9b9ad1ddae52393ba6ec97548e162d009`
- Disposable artifact root:
  `/scratch/gpfs/GILLES/mg6942/vdam_runs/vdam_packed_deferred_dense_reduce_same_state_it34_0e1b4355c_20260903T012005Z`
