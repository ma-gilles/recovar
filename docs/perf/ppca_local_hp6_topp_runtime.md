# PPCA HP6 top-p local pose-scoring runtime — 2026-05-15

Operational note: on the K=4 100k 256² Ribosembly completion fixture, the
PPCA dense → OS → HP6 top-p (p=4) local pose-scoring pipeline at
`current_size=256, q=4, sigma_rot=2°, sigma_psi=2°, healpix_order_local=6` is
forced to `image_batch_size=1`. This file documents why and what to change.

## Measured bottleneck

`recovar/em/ppca_refinement/local_dataset.py:1038` allocates

    proj_aug = _project_local_augmented(...)   # (B, R, q+1, n_pixels) complex64

For this run:

- `B = image_batch_size`
- `R = bucket_rotation_count` (per-image HP6 neighborhood at sigma_rot=2°,
  sigma_cutoff=3 ≈ 1700–2900 rotations; padded by
  `_exact_bucket_rotation_size` to multiples of 64 → 1728, 1792, 1856, …, 2880)
- `q + 1 = 5`
- `n_pixels = current_size * (current_size // 2 + 1) = 33024` for box 256

Single proj_aug at `B=2, R=1856` = **4.9 GiB** (complex64). XLA peaks at
3-5× proj_aug while scoring (the inline comment near
`PPCA_LOCAL_IMAGE_BATCH_CEIL` in the same file documents this empirically).
On the running K=4 100k/256² job:

- `--image-batch-size 4` (original sbatch) OOM'd at allocation 8.57 GiB
  → BFC after dense+OS state was already partly pinned.
- `--image-batch-size 2` OOM'd at allocation 17.07 GiB (fresh BFC heap, but
  the XLA scratch was bigger than the post-dense run).
- `--image-batch-size 1` fits at 19–36 GiB resident on A100-80GB/H100-80GB
  but processes **100k buckets** one image at a time. GPU util drops to
  33–60% because launch overhead dominates over compute.

End-to-end stage timings (image_batch=1):

| stage                          | wall    |
|--------------------------------|---------|
| dense HP3 score (full grid)    | 3 h 41 m |
| adaptive OS HP4 pass-2         | 8.1 min |
| HP6 top-p p=4 pose-only        | 61 min   |
| EM iter (HP6 top-p + M-step)   | ≥ 2 h    |

## Why none of the existing knobs help

- `--rotation-block-size` only changes `engine_cap = max(64, min(rbs, 1024))`,
  which only affects buckets *under* engine_cap (power-of-2 rounding). For
  R≈1700 the large-bucket path applies and rounds to multiples of 64
  regardless.
- `--max-hypotheses-per-microbatch` caps `max_images = max(1, min(image_batch_size,
  max_hypotheses // bucket_size))`. With bucket=1856 the natural cap
  is 35; the user-supplied `image_batch_size` is the binding constraint.
- `RECOVAR_LOCAL_BUCKET_QUANTUM=128` rounds buckets up to larger multiples →
  more padding, no speedup here.
- `RECOVAR_LOCAL_BUCKET_UNIFY=1` pins all buckets to the worst-case
  `bucket_size = max ≈ 2880`. Saves a few JIT compiles but raises peak memory
  ~55%. Net wash for this neighborhood-distribution.
- `RECOVAR_PPCA_LOCAL_TARGET_ROW_PIXELS` only takes effect when
  `--max-hypotheses-per-microbatch` is unset; CLI explicitly sets it here.

## Status: rotation-chunked pose-only path implemented (env-var opt-in)

`RECOVAR_PPCA_LOCAL_R_CHUNK_SIZE=<N>` enables a Python-side rotation-chunked
wrapper around `score_local_pose_ppca_bucket`. Each bucket's R dim is tiled
into chunks of size N; per-chunk diagnostics are aggregated into a single
:class:`PosteriorDiagnostics`. Tested at parity (commit pending) via
`tests/unit/ppca_refinement/test_local_pose_score_rotation_chunked.py` —
all of `logZ`, `best_*`, `top_*`, `pmax`, `top_posterior` match the one-shot
kernel exactly on synthetic data. `rotation_posterior_sums` is set to zero
(only used in EM M-step; M-step accumulator uses a different code path
unaffected by this knob). `n_significant_per_image` is a conservative
over-estimate (sum of per-chunk counts under per-chunk normalization).

Recommended starting value for HP6 top-p (p=4) at box 256², q=4:
``RECOVAR_PPCA_LOCAL_R_CHUNK_SIZE=256`` combined with
``--image-batch-size 8``. Peak proj_aug per chunk is then
``8 × 256 × 5 × 33024 × 8 = 2.7 GiB`` (well under 80 GiB even with 3-5×
XLA scratch). This is the **pose-only** path only; the EM M-step
accumulator (`run_local_ppca_fused_em_iteration` →
`_accumulate_local_ppca_fused_stats`) has its own memory profile and
still requires ``image_batch_size=1`` at this neighborhood scale until
a similar refactor lands there.

## Real fix: rotation-tiled score kernel

Make `score_local_pose_ppca_bucket` (and `score_local_pose_ppca_bucket_with_moments`
for the EM path) process the R dimension as a `lax.scan` over fixed-size
chunks. Peak per-bucket memory drops to `chunk × q+1 × n_pixels` per image
regardless of the per-image neighborhood size.

### Streaming aggregation across rotation chunks

The score kernel currently materializes `score: (B, T, R)` in one shot.
With chunking, each chunk produces `chunk_score: (B, T, chunk_R)`. Carry
state needed for streaming aggregation:

| Output field                  | Carry / aggregation strategy                                              |
|-------------------------------|----------------------------------------------------------------------------|
| `logZ` `(B,)`                 | streaming logsumexp: keep `(running_max, running_sum_exp)` per image       |
| `pmax` `(B,)`                 | `exp(global_max_score - logZ)` — `global_max_score` falls out of streaming |
| `best_log_score_per_image`    | same `global_max_score`                                                    |
| `best_rotation_idx`           | streaming argmax with chunk-offset, store `(best_score, best_flat_idx)`    |
| `best_translation_idx`        | same                                                                       |
| `top_rotation_idx` (top-K)    | streaming top-K: concat `(carry_K, chunk_K)` → re-`lax.top_k`              |
| `top_translation_idx`         | same, paired with rotation indices                                         |
| `top_log_score_per_image`     | from streaming top-K                                                       |
| `top_posterior_per_image`     | `exp(top_log_score - logZ)` after global `logZ`                            |
| `n_significant_per_image`     | needs two passes (collect chunk scores, then re-count vs global logZ) OR keep a conservative chunk-sum (over-estimate) |
| `rotation_posterior_sums`     | unused in pose-only path; in EM the M-step needs accurate per-pose gamma → not affected by this chunking strategy (M-step uses moments kernel) |

`n_significant` is exact via a two-pass scan: pass 1 collects per-chunk
score statistics for streaming logZ + top-K + best; pass 2 re-iterates
chunks with the now-known `logZ` to count `score > logZ + log(threshold)`.

### Suggested signature

```python
@partial(jax.jit, static_argnames=("rotation_chunk_size", "significance_threshold", "top_pose_count"))
def score_local_pose_ppca_bucket_scan(
    Y1, proj_aug, ctf2_over_noise, y_norm, pose_log_prior,
    *, rotation_chunk_size: int, significance_threshold: float = 1e-3, top_pose_count: int = 1,
) -> PosteriorDiagnostics:
    ...
```

Caller dispatch in `_score_local_ppca_pose_diagnostics`:

```python
chunk = int(os.environ.get("RECOVAR_PPCA_LOCAL_R_CHUNK_SIZE", "0"))
if chunk > 0 and chunk < block.proj_aug.shape[1]:
    posterior = score_local_pose_ppca_bucket_scan(
        block.Y1, block.proj_aug, block.ctf2_over_noise, block.y_norm, block.pose_log_prior,
        rotation_chunk_size=chunk,
        top_pose_count=top_pose_candidate_count(pose_selection, candidate_count),
    )
else:
    posterior = score_local_pose_ppca_bucket(...)  # existing path
```

### Expected gain

At `chunk_size=256, image_batch=8`:

- proj_aug peak per chunk = `8 × 256 × 5 × 33024 × 8 = 2.7 GiB`
- XLA scratch (3-5×) = 8–14 GiB → fits in 80 GB with mu/W/dataset state.
- Buckets reduced from 100k (batch=1) → 12.5k (batch=8) → ~4× fewer kernel
  launches; each launch processes 8× more images.
- Net speedup target: **3-5× faster HP6 pose-only stage** (61 min → 12-20
  min), comparable EM iter speedup.

### Risks

- Streaming top-K must be tested against the one-shot top-K result on
  small synthetic data — element ordering matters when ties exist.
- `n_significant_per_image` exactness requires the two-pass scan.
- Adding `rotation_chunk_size` to `static_argnames` triggers a JIT recompile
  per chunk size value used in a session; the value should be a single
  process-wide constant (env var fixes this).

### Out-of-scope alternatives (deeper refactor)

- Split mu and W projection into two passes (q+1=5 → q+1=1 + q=4), each
  with its own scoring kernel that reuses cached y/ctf state. ~5× memory
  reduction in proj_aug, allows even larger image_batch_size. Doubles
  projection compute. ~1-2 days work.
- Run `_project_augmented_half_volumes` with R-tiled rotation matrices on
  the CUDA side (`recovar.cuda_backproject.batch_project`). The kernel
  already supports arbitrary R; the JAX wrapper would tile externally.
  Eliminates the (R, q+1, n_pixels) tensor materialization entirely; the
  kernel writes directly into a small running accumulator.

## Reproducing the bottleneck without burning a multi-hour run

```python
import numpy as np
from recovar.em.sampling import get_relion_rotation_grid, build_local_search_grid_metadata
from recovar.em.ppca_refinement.highres_refinement import build_top_p_local_hypothesis_layout
from recovar.em.dense_single_volume.local_layout import bucket_local_hypothesis_layout

# Synthesize 100 top-p centers at a HP4 grid
center_grid = np.asarray(get_relion_rotation_grid(4), dtype=np.float32)
target_grid = np.asarray(get_relion_rotation_grid(6), dtype=np.float32)
metadata = build_local_search_grid_metadata(6)
n = 100
top_rot = np.zeros((n, 4), dtype=np.int64)
top_trans = np.zeros((n, 4), dtype=np.int64)
top_rot[:, 1:] = -1  # only 1 valid pose per image
top_trans[:, 1:] = -1
top_mats = np.tile(center_grid[0:1, None], (n, 4, 1, 1))
layout = build_top_p_local_hypothesis_layout(
    top_rot, top_trans,
    center_rotation_grid=center_grid,
    top_rotation_matrices=top_mats,
    center_translation_grid=np.asarray([[0.0, 0.0]], dtype=np.float32),
    target_rotation_grid=target_grid,
    healpix_order=6,
    translations=np.asarray([[0.0, 0.0]], dtype=np.float32),
    sigma_rot=np.deg2rad(2.0), sigma_psi=np.deg2rad(2.0),
    sigma_offset_angstrom=3.0, voxel_size=1.0, grid_metadata=metadata,
)
buckets = bucket_local_hypothesis_layout(layout, image_batch_size=1, rotation_block_size=512, max_hypotheses_per_microbatch=65536)
print(len(buckets), 'buckets; unique bucket_rotation_count:', sorted({b.bucket_rotation_count for b in buckets}))
# → ~100 buckets, bucket sizes in {1728, 1792, 1856}
```
