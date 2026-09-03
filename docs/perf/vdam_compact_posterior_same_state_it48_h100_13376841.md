# VDAM compact-posterior iteration-48 gate — H100 job 13376841

## Decision

Pass the one-transition exact-science and performance boundary and advance the
default-off compact posterior to a combined compact-plus-packed gate and a
fresh full-trajectory sentinel.  Starting from one exact in-memory GF46
iteration-47 state, every direct/compact crossed comparison retained exact
support, pose, translation, class, maximum-posterior, particle-state, and
sampling-state results.  The warmed compact arm reduced coarse pass 1 by
87.88% and whole-transition wall time by 57.65%.

This does not promote or default-enable the path.  A one-transition panel
cannot establish accumulated basin stability, and this panel compares the
combined certified-hybrid-plus-compact path to direct scoring rather than
isolating compact versus the former dense-hybrid posterior.

## Qualification

| Field | Value |
|---|---|
| Source | `f833ec2b42ea11e26a34c9449952ced0f971cf6a` |
| Source tree | `16855156eff7186f4b5f0e32970befbcd7dc6a9b` |
| Slurm | `13376841` (`COMPLETED`, exit `0:0`, elapsed `00:08:00`) |
| Hardware | `della-h19g3`, NVIDIA H100 80GB HBM3, `GPU-1fdb3b99-e7ff-fe6d-4f59-9d2cc85fa319` |
| Boundary | One exact in-memory GF46 iteration-47 state, iteration `47 -> 48` |
| Panel | direct / compact posterior / compact posterior / direct |
| Fixture | Frozen GF46 launch manifest used by the earlier true-200 sentinel |
| Diagnostics | Complete support IDs enabled in every transition arm; no score dumps |
| Peak monitored HBM | 17,595 MiB for the whole direct-first ABBA process |
| Maximum host RSS | 8,163,396 KiB (`/usr/bin/time -v`) |

The harness explicitly set the compact environment to zero in both controls
and required it to be one together with the hybrid, macro, and projection
cache environments in both candidates.  Every arm reported an exact
requested/effective environment match.  Both candidate profiles reported the
fixed-capacity source-16 layout, no expanded GEMM scores, and zero fallback
batches; both controls published no hybrid profile.

## Exact science result

- All four crossed direct/compact comparisons pass the hard exact contract:
  selected particle IDs, rotation/pose/translation/class decisions,
  significant counts, complete per-image support IDs, particle state, and
  sampling state are exact.
- Every arm has support SHA-256
  `8aa7d543c837a0079b7a50b76863b962308b5a6b8d89135eef82071d82290780`.
- `max_posterior_per_image` is bitwise equal for all 200 selected particles in
  every repeat and crossed comparison.
- Continuous accumulator deltas remain at CUDA-repeat scale.  The maximum
  crossed accumulator normalized L2 is `8.79e-8`; the worst cross/maximum-
  within-arm-repeat ratio is `1.093x`.
- Continuous final-state deltas remain small: maximum crossed normalized L2 is
  `1.05e-7`, and the reconstructed-reference (`Iref`) maximum is `7.40e-11`.
  The worst raw cross/repeat ratio is `1.936x` for `sigma2_class`.

The deliberately strict, unscaled observed-repeat check flags three of four
accumulator fields and five final-state fields because their largest crossed
sample is slightly larger than the maximum of only one direct and one compact
repeat.  All normalized deltas are nevertheless below the mature
`4*float32-epsilon = 4.768e-7` numerical floor and below two times their pooled
maximum within-arm repeat.  Crossed accumulator-weight signed means vary in
sign rather than showing a consistent directional drift.  This is accepted as
one-step mathematically equivalent numerical noise, but the full trajectory
remains the required growth/basin test.

## Runtime result

Only the second warmed arm of each backend is compared.

| Metric | Warm direct | Warm compact | Change | Speedup |
|---|---:|---:|---:|---:|
| Whole transition | 5.904510 s | 2.500615 s | **-57.65%** | **2.361x** |
| Coarse pass 1 | 3.866524 s | 0.468535 s | **-87.88%** | **8.252x** |
| Fine pass 2 | 1.281364 s | 1.266655 s | -1.15% | 1.012x |

The first candidate arm took 8.544557 s because it compiled the new compact
shape; the repeated candidate is the timing arm.  The direct repeated arm
reused the direct shapes compiled by the first control.

## Compact table result

The production profile used `R=36,864`, `T=49`, padded image lanes `B=110`
over two batches, source-block capacity `Q=64`, and no fallback.

| Quantity | Value |
|---|---:|
| Active exact source-16 blocks | 1,197 |
| Maximum selected blocks per image | 17 |
| Active exact candidates | 938,448 |
| Full candidates for 200 real images | 361,267,200 |
| Active exact fraction | 0.259766% |
| Fixed-capacity compact table | 11,038,720 float32 values / 42.109 MiB |
| Equivalent padded dense table | 397,393,920 float32 values / 1,515.938 MiB |
| Compact/dense table fraction | 2.777778% |
| Dense-to-compact table reduction | **36.0x** |

The monitored HBM peak is not an arm-specific allocation measurement: the
direct arm runs first, and the JAX allocator can retain its buffers.  Therefore
the table byte counts prove the score-table reduction, while the 17,595 MiB
process peak must not be presented as compact-only memory.

## Provenance

- Science report SHA-256:
  `c29ae861e1ea934a4393e76a7ce5cabab51dd0923cdce9f9726fab1b27a9cb52`
- Rebuilt CUDA binary SHA-256:
  `1bd298dd262adf73fdec84a25d88ee058d9ca5e82013e39af606f143b2de3c3d`
- Qualified-CUDA checksum-file SHA-256:
  `665d4bf52fac66c7ff578ff7c6f446213b6d46e9f55a5786beebe33a186fd46f`
- Static-input checksum-file SHA-256:
  `6ad6fba9e8b846a1ce53070112fef99f6b1d59f2d9390eb514665ba1278f18d2`
- Frozen launch-manifest SHA-256:
  `e197dd9bcf326b8cc8d22f3330d3a39f17405cb635afd619e277f457a0f920f5`
- Artifact manifest SHA-256:
  `4cbb7b5a07d3e9cb4128c29f6c9677c44a8ee00c2705ed8ad1f43bfb9fe3ba9a`
- Sealed artifact-manifest checksum-file SHA-256:
  `a10d24f0abf8ec52dc88a8da8e34a88b4bf8a6b32ac7f67500cbd18d74ab65e1`
- Empty worktree-status SHA-256:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Disposable immutable artifact root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_compact_posterior_same_state_it47_f833ec2b4_20260903T060543Z`
