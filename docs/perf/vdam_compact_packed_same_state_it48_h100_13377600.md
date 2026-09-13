# VDAM compact-posterior + packed-deferred iteration-48 gate — H100 job 13377600

## Decision

Pass the one-transition composition boundary and advance the default-off
compact-posterior plus packed/deferred candidate to a fresh full `0 -> 200`
trajectory sentinel. Starting from one exact in-memory GF46 iteration-47
state, every crossed comparison retained exact support, pose, translation,
class, maximum-posterior, particle-state, and sampling-state results. The
warmed combined arm reduced whole-transition wall time by 61.29%, coarse pass
1 by 87.33%, and fine pass 2 by 21.21%.

This does not promote or default-enable either path. A one-transition panel
cannot establish accumulated basin stability or full-trajectory runtime.

## Qualification

| Field | Value |
|---|---|
| Source | `17ec60ca5dcd5d3b67e956876d9df07ed2a6345d` |
| Source tree | `0116e5b6fa4a85f4b4e698fe17c455136c2d2961` |
| Slurm | `13377600` (`COMPLETED`, exit `0:0`, elapsed `00:08:23`) |
| Hardware | `della-h19g1`, NVIDIA H100 80GB HBM3, `GPU-0d7b80c7-fef8-e346-6332-de36ae1af518` |
| Excluded node | `della-h19g2` |
| Boundary | One exact in-memory GF46 iteration-47 state, iteration `47 -> 48` |
| Panel | direct / compact+packed / compact+packed / direct |
| Fixture | Frozen GF46 launch manifest used by the earlier true-200 sentinel |
| Diagnostics | Complete support IDs enabled in every transition arm; no score dumps |
| Peak monitored HBM | 17,595 MiB for the whole direct-first ABBA process |
| Maximum host RSS | 8,175,580 KiB (`/usr/bin/time -v`) |

The harness explicitly set all seven candidate switches to zero in both
controls and required them to be one in both candidate arms. Every arm
reported an exact requested/effective environment match. Both candidates
reported the fixed-capacity source-16 compact score layout, no expanded GEMM
scores, zero fallback batches, flat local rows, packed local projection,
deferred packed VDAM, and reuse of the flat score projection. Both controls
published no hybrid profile.

## Exact science result

- All four crossed direct/candidate comparisons pass the hard exact contract:
  selected particle IDs, rotation/pose/translation/class decisions,
  significant counts, complete per-image support IDs, particle state, and
  sampling state are exact.
- Every arm has support SHA-256
  `955d3e2d023968e189c58411a751779d82aaa9a3c10fdd151578beb31dbd010c`.
- `max_posterior_per_image` is bitwise equal for all 200 selected particles in
  every crossed comparison.
- The largest crossed accumulator normalized L2 is `8.90001e-8`; the worst
  cross/maximum-within-arm-repeat ratio is `1.061x`.
- The largest crossed final-state normalized L2 is `3.17352e-8`. The maximum
  reconstructed-reference (`Iref`) normalized L2 is `5.19938e-11`.

The deliberately strict, unscaled observed-repeat check flags all four
accumulator fields and five final-state fields because one crossed sample can
slightly exceed the maximum of only one repeat from each arm. Its largest raw
ratio is `2.956x` for `sigma2_noise`, whose absolute normalized L2 is only
`1.67137e-9`. Every continuous delta is well below the mature
`4*float32-epsilon = 4.76837e-7` numerical floor, while every discrete and
support-bearing result is exact. This is accepted as one-step mathematically
equivalent numerical noise; the full trajectory remains the required growth
and basin test.

## Runtime result

Only the second warmed arm of each backend is compared.

| Metric | Warm direct | Warm compact+packed | Change | Speedup |
|---|---:|---:|---:|---:|
| Whole transition | 5.966168 s | 2.309737 s | **-61.29%** | **2.583x** |
| Coarse pass 1 | 3.884599 s | 0.492254 s | **-87.33%** | **7.891x** |
| Fine pass 2 | 1.320671 s | 1.040594 s | **-21.21%** | **1.269x** |

The first candidate arm took 17.296377 s because it compiled the new combined
shapes; the repeated candidate is the timing arm. Relative to the preceding
compact-only gate, the combined arm improves raw warmed wall time by another
7.63% and fine-pass time by 17.85%. Normalizing each run to its own direct
control gives an 8.59% additional whole-transition reduction and a 20.29%
additional fine-pass reduction. The pass-1 difference between those separate
jobs is small hardware/run variability and is not attributed to the packed
path.

## Compact table result

The production profile used `R=36,864`, `T=49`, padded image lanes `B=110`
over two batches, source-block capacity `Q=64`, and no fallback.

| Quantity | Value |
|---|---:|
| Active exact source-16 blocks | 1,196 |
| Maximum selected blocks per image | 16 |
| Active exact candidates | 937,664 |
| Full candidates for 200 real images | 361,267,200 |
| Active exact fraction | 0.259549% |
| Fixed-capacity compact table | 11,038,720 float32 values / 42.109 MiB |
| Equivalent padded dense table | 397,393,920 float32 values / 1,515.938 MiB |
| Compact/dense table fraction | 2.777778% |
| Dense-to-compact table reduction | **36.0x** |

The monitored HBM peak is not arm-specific because the direct arm runs first
and the JAX allocator may retain its buffers. The table capacities prove the
score-table reduction; the process peak must not be presented as compact-only
memory.

## Provenance

- Science report SHA-256:
  `3e7b0b0df783adc2ce8c64ddd3e2860ea2f0c87908ca6cc5c938694f7dc62b25`
- Rebuilt CUDA binary SHA-256:
  `10282a2cb09f156fcf5946c40827caca1a5dae3da52883b92ca25f6ae1022d1e`
- Qualified-CUDA checksum-file SHA-256:
  `91b6c91e35f2054925f066b403d9a2e51e3a5b9efa618be45ea7ec95c802194f`
- Static-input checksum-file SHA-256:
  `a0c853e30735dce312a8e8ace6bbdbaee698b6f3a846406c4a2ee9309937beed`
- Frozen launch-manifest SHA-256:
  `e197dd9bcf326b8cc8d22f3330d3a39f17405cb635afd619e277f457a0f920f5`
- Artifact manifest SHA-256:
  `d76c5d844a9d14778403e237b94d36d6e42764b41a9499d834feb2964089956b`
- Sealed artifact-manifest checksum-file SHA-256:
  `5dbd25c1f962240cb6154349765b2607968d39865fcfe760aabbe7ce1f299a33`
- Empty worktree-status SHA-256:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Disposable immutable artifact root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_compact_packed_same_state_it47_17ec60ca5_20260903T062731Z`
