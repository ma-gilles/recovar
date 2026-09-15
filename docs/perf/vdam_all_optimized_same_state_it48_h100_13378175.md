# VDAM fully optimized iteration-48 gate — H100 job 13378175

## Decision

Pass the one-transition composition boundary for the complete default-off
performance stack. Starting from one exact in-memory GF46 iteration-47 state,
all four crossed direct/candidate comparisons retain exact selected particles,
poses, translations, classes, posterior maxima, significant counts, complete
support IDs, particle state, and sampling state. The warmed candidate reduces
whole-transition wall time by 60.83%, coarse pass 1 by 87.69%, and fine pass 2
by 16.93%.

This is a diagnostic composition pass, not a promotion. Stable Fourier and
fixed-row shapes are warm-neutral at this boundary; their intended benefit is
reducing cold cross-iteration compilation. Their separate fresh-process
trajectory first enters a reproducible alternate hard-assignment basin at
iteration 35, so a repeat-controlled full trajectory remains mandatory.

## Qualification

| Field | Value |
|---|---|
| Source | `0729473a1a752f1474edd666ae0740ef692049ed` |
| Source tree | `12a6e93abeb06c7a59857f34c6f13df6e39d2f6e` |
| Slurm | `13378175` (`COMPLETED`, exit `0:0`, elapsed `00:08:24`) |
| Hardware | `della-h19g3`, NVIDIA H100 80GB HBM3, `GPU-ef985070-011e-0782-6f0a-94b053dcc120` |
| Excluded node | `della-h19g2` |
| Boundary | One exact in-memory GF46 iteration-47 state, iteration `47 -> 48` |
| Panel | direct / all-optimized / all-optimized / direct |
| Peak monitored HBM | 17,595 MiB for the whole direct-first ABBA process |
| Maximum host RSS | 8,128,948 KiB (`/usr/bin/time -v`) |

The harness proves all nine named seams disabled in both controls and enabled
in both candidates: certified coarse hybrid, coarse GEMM macro batching,
coarse projection cache, compact posterior, flat local rows, packed local
projection, packed VDAM deferral, stable Fourier window shapes, and stable
flat-row capacity. It fails closed on any requested/effective environment or
profile mismatch.

At this transition, the stable Fourier plan maps logical size 84 to physical
size 88 and logical 2,835 reconstruction pixels to 3,105 physical pixels. The
fixed-row path executes five chunks of 12,288 rows, exactly equal to each
chunk's `B*R` capacity. The compact coarse path has zero fallbacks and stores
11,038,720 float32 candidates (42.109 MiB) instead of the equivalent
397,393,920-candidate padded dense table (1,515.938 MiB), a 36x reduction.

## Exact science result

- Every crossed direct/candidate pair passes the hard exact contract.
- All arms publish the same complete-support SHA-256 prefix `7615d7e4`.
- Maximum crossed accumulator normalized L2 is `2.33984e-7`.
- Maximum crossed final-state normalized L2 is `1.20901e-7`.
- Maximum crossed reconstructed-reference (`Iref`) normalized L2 is
  `1.31512e-10`.

The deliberately strict two-repeat envelope rejects four accumulator fields
and six continuous final-state fields. The largest crossed continuous delta is
still below `4*float32-epsilon = 4.76837e-7`, while all support-bearing and
discrete results are bitwise exact. This is accepted as one-step
mathematically equivalent numerical noise; only a trajectory can determine
whether it accumulates into a different basin.

## Runtime result

Only the second warmed arm of each backend is compared.

| Metric | Warm direct | Warm all-optimized | Change | Speedup |
|---|---:|---:|---:|---:|
| Whole transition | 5.923261 s | 2.320158 s | **-60.83%** | **2.553x** |
| Coarse pass 1 | 3.853030 s | 0.474271 s | **-87.69%** | **8.124x** |
| Fine pass 2 | 1.302402 s | 1.081844 s | **-16.93%** | **1.204x** |

The first all-optimized arm takes 20.030360 s while compiling its shape set;
the first direct arm takes 13.976958 s. The warmed candidate is effectively
equal to the preceding compact-posterior plus packed/deferred result
(2.309737 s). Thus stable shapes should be judged on fresh trajectories and
compile counts, not credited with a warmed kernel speedup.

## Provenance

- Science report SHA-256:
  `5976788bbb57f4dbe88726e2f315050fd305e8ebf1dc39648d3ece6246057f54`
- Sealed aggregate checksum-file SHA-256:
  `c0eeb39937e145843cf020bf685d8b03b2bdde580292540a1d9c8e09ad712ee4`
- Static-input checksum-file SHA-256:
  `184c1fe73c840b6a7574af30ac03c134bbe9b44b5b813e0a9a07d55a05d2f931`
- Qualified-CUDA checksum-file SHA-256:
  `57aa74a849cdac0eabfea614c884d60f6b2a89f3ae0b14a812022e2512108d0d`
- Artifact manifest SHA-256: `f41d6a99ea6c95741bcf46844cde576e5f525c2977bf115c3f48d43e9658873f`
- Empty worktree-status SHA-256:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Disposable immutable artifact root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_all_optimized_same_state_it47_0729473a1_20260903T065746Z`
