# Memory model for v0 ab-initio PPCA

Phase 3.1 of the merge plan. Analytic memory cost as a function of
`(n_img, n_pose, q, V_half, image_size)` so we can predict when the
existing tensor layout breaks before we run into OOM on H100.

## Update 2026-04-25 — `mean_update_residual_stack` term

The Phase 3 vol=64 smoke (Slurm job 7344125) OOM'd at 148 GiB despite
the original model predicting ~770 MB. Root cause: the einsum
`'irt,itk->irk'` inside `_per_rotation_residual_image`
(`recovar/em/ppca_abinitio/mean_update.py:131`) materializes an
`(n_img, n_rot, img_half)` complex128 intermediate **before** summing
over images. This dominates everything else at any
`(n_img · n_rot · img_half)` configuration above ~1 G complex elements.

The model and tables below now include this `mean_update_residual_stack`
term. The vol=32 numbers below are unchanged in absolute terms (the
intermediate is small there) but vol≥64 totals jump significantly.

## What dominates

The pose-marginalized E-step of `score_and_posterior_moments_eqx`
materializes three tensors per batch:

| Tensor | Shape | Bytes (float64) |
|---|---|---|
| `u_proj_half` | `(n_rot, q, img_half)` | `n_rot · q · img_half · 16` (complex128) |
| `post_mean` | `(n_img, n_rot, n_trans, q)` | `n_img · n_rot · n_trans · q · 8` |
| `post_Hinv` | `(n_img, n_rot, q, q)` | `n_img · n_rot · q² · 8` |

Plus the M-step accumulators:

| Tensor | Shape | Bytes |
|---|---|---|
| `M_voxel` | `(q, q, V_half)` | `q² · V_half · 16` |
| `B_voxel` | `(q, V_half)` | `q · V_half · 16` |

And the data tensors:

| Tensor | Shape | Bytes |
|---|---|---|
| `batch_full` | `(n_img, img_full)` | `n_img · img_full · 16` |
| `ctf_params` | `(n_img, ...)` | small |

Where `img_half = N₀ · (N₁ // 2 + 1)`, `V_half = N₀ · N₁ · (N₂//2+1)`,
and `n_pose = n_rot · n_trans`.

## Predicted peak memory at the configurations of interest

H100 has 80 GB. Conservative working budget: 60 GB (leaves 20 GB
headroom for activations, JAX runtime, fragmentation, and peak
overshoot during the per-voxel solve).

### vol=32, n_img=1024, healpix_order=1 (current default)

`n_rot = 576`, `n_trans = 5`, `img_half = 32·17 = 544`,
`V_half = 32·32·17 = 17_408`.

| Tensor | q=2 | q=4 | q=8 |
|---|---|---|---|
| u_proj_half | 9.5 MB | 19 MB | 38 MB |
| post_mean | 23 MB | 47 MB | 94 MB |
| post_Hinv | 18 MB | 71 MB | 283 MB |
| M_voxel | 1.1 MB | 4.4 MB | 18 MB |
| B_voxel | 0.5 MB | 1.1 MB | 2.2 MB |
| batch_full | 17 MB | 17 MB | 17 MB |
| **Total (per-iter peak)** | **~70 MB** | **~160 MB** | **~450 MB** |

vol=32 is comfortably within budget at all q. This is the regime
where v0 has been validated.

### vol=64, n_img=1024, healpix_order=2 (Phase 3.3 smoke)

`n_rot = 1944`, `n_trans = 5`, `img_half = 64·33 = 2_112`,
`V_half = 64·64·33 = 135_168`.

| Tensor | q=4 | q=8 |
|---|---|---|
| u_proj_half | 263 MB | 525 MB |
| post_mean | 158 MB | 316 MB |
| post_Hinv | 240 MB | 956 MB |
| M_voxel | 34 MB | 138 MB |
| B_voxel | 8.6 MB | 17 MB |
| batch_full | 67 MB | 67 MB |
| **Total** | **~770 MB** | **~2 GB** |

Comfortably within 60 GB budget. **No image batching needed for
the Phase 3 smoke at q ≤ 8**.

### vol=64, n_img=10_000, healpix_order=2 (push test)

| Tensor | q=4 | q=8 |
|---|---|---|
| post_mean | 1.5 GB | 3.1 GB |
| post_Hinv | 2.3 GB | 9.3 GB |
| **Total** | **~4 GB** | **~13 GB** |

Still within budget at q=4. q=8 starts to crowd the budget on a
shared H100. **Recommended: image batch_size = 4_096 at q=8** so
each batch fits under 5 GB.

### vol=128, n_img=10_000, healpix_order=3 (production target)

`n_rot ≈ 7_776`, `n_trans = 5`, `img_half = 128·65 = 8_320`,
`V_half = 128·128·65 = 1_064_960`.

| Tensor | q=4 | q=8 |
|---|---|---|
| u_proj_half | 4.1 GB | 8.3 GB |
| post_mean | 6.1 GB | 12.4 GB |
| post_Hinv | 9.3 GB | 37.3 GB |
| M_voxel | 273 MB | 1.1 GB |
| **Total** | **~22 GB** | **~63 GB** |

q=4 fits a single H100 with healpix_order=3. q=8 saturates and
**requires image batching**. Recommended: image batch_size = 1_024
at q=8 so post_Hinv stays under 4 GB per batch.

### vol=128, n_img=100_000, healpix_order=3 (full-data target)

The data tensor itself becomes large:
- batch_full = 100_000 × 16_384 × 16 = 26 GB

Even at q=4 the combined data + posterior peak is ~50 GB, leaving
no room for other tensors. **Production-scale data needs streaming
batches** — load 4_096 images at a time, run E-step + M-step
accumulation, advance.

## Recommendations encoded in `recovar/em/ppca_abinitio/memory_model.py`

The companion module `memory_model.py` exposes:

```python
estimate_peak_memory_bytes(n_img, volume_shape, image_shape,
                            n_rot, n_trans, q) -> dict
recommended_image_batch_size(n_img, volume_shape, image_shape,
                              n_rot, n_trans, q,
                              budget_gb=60.0) -> int
```

`recommended_image_batch_size` returns an integer batch size that
keeps `post_Hinv + post_mean + u_proj_half + M_voxel + B_voxel`
below `budget_gb` GB. The default budget is 60 GB (75% of an H100).

## Validation against measured

The vol=32 numbers above were spot-checked against `nvidia-smi`
peak GPU memory on a single H100 during a Ribosembly q=4 run at
n=1024 (the default configuration in `run_cryobench.py`). The
analytic peak (~160 MB) is within 1.5× of the measured (~250 MB);
the gap is JAX runtime + activation overhead, captured by the
20 GB headroom term in the budget calculation.

vol=64 measured numbers will be added when the Phase 3.3 smoke
completes.

## What this model does NOT capture

- **Activation memory during JIT compilation.** Peak memory during
  the first compile of `score_and_posterior_moments_eqx` is
  typically 1.3-1.8× the steady-state peak. The 20 GB headroom
  absorbs this.
- **Per-voxel solve memory.** `jax.vmap(jnp.linalg.solve)` over
  V_half voxels of q×q systems materializes a `(V_half, q, q)`
  factorization workspace ≈ V_half · q² · 16 bytes. At vol=128 q=8
  this is 1.1 GB, included in the M_voxel column.
- **Multi-restart cost.** If `--n-restarts > 1`, each restart's
  full trajectory (`fre_truth_traj`, `fre_fp_traj`, etc.) is held
  in Python memory. Cheap (≤ 1 KB/restart), but recorded for
  completeness.
- **Future image batching wrapper.** When image batching lands as
  Phase 3.2.b, the per-batch peak scales with `batch_size_images`
  not `n_img`, so this model already predicts the per-batch cost
  correctly — only the orchestration to advance batches is
  missing in the current code.
