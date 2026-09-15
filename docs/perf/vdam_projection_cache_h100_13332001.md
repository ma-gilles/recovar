# VDAM shared projection-cache H100 qualification: job 13332001

Status: **infrastructure primitive passed; not an end-to-end runtime or science
qualification**.

The shared builder assembled the full GF46-shaped complex64 destination
`(1, 36864, 5100)` on one H100.  This qualification used a deterministic
zero-pattern projection callback with boundary sentinels; it validates cache
allocation/insertion, aliasing, memory planning, and chunk-shape behavior.  It
does not measure the real projector or certify VDAM results.

## Result

| Arm | Chunk shapes | Compiled executables | Compile total | Full build wall | Conservative peak |
|---|---|---:|---:|---:|---:|
| 4992 | `4992` plus tail `1920` | 2 | 0.304090 s | 0.184620 s | 2,039,992,320 B |
| 4608 | eight equal `4608` chunks | 1 | 0.048230 s | 0.114919 s | 1,998,766,080 B |

Both arms made eight callback calls, completed without OOM, and reproduced
the first row, final row, and every chunk-boundary sentinel exactly.  Every
compiled insert had parameter-0 input/output aliasing, with output and alias
size exactly `1,504,051,200` bytes and XLA temporary size zero.

The durable structural decision is to use 4608 rows for the GF46 production
experiment: `36864 = 8 * 4608`, so it removes the final 1920-row executable.
The observed 84.14% lower cold compile total and 37.75% lower one-shot build
wall are useful discrimination evidence, not a throughput forecast.  The
allocator watermark was cumulative across arms, so only the conservative plan
supports the stated 41,226,240-byte peak reduction.

## Provenance

- Slurm job: `13332001`, completed `0:0` in 11 s on `della-h19g1`
- GPU: H100 80 GB HBM3, UUID
  `GPU-2ee3da91-970a-6714-84df-530aefe04a08`
- Python/JAX/jaxlib: `3.11.13` / `0.9.0.1` / `0.9.0.1`
- CUDA PJRT/driver: `12090` / `610.57.04`
- immutable source commit: `43402732cb169ab5d91d90b10262a35ba99edae4`
- source tree: `9fcebb83cfe1d0d6a8bcec87c427d4229e9ecd81`
- result JSON SHA-256:
  `576384429d01fbe2d86a81c13a7519c484490e5ac62bb8bc871387df653023c2`
- artifact root (contains `SAFE_TO_DELETE`):
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_projection_cache_h100_20260902_43402732_uniform4608`

Reproduce while the pinned worktree remains available:

```bash
sbatch /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_projection_cache_h100_20260902_43402732_uniform4608/qualify_projection_cache_chunks_h100.sbatch
```

Superseded job `13331902` was intentionally cancelled after the uniform-4608
arm was requested and is not evidence.

## Code references

- `recovar/em/dense_single_volume/helpers/projection_cache.py:plan_projection_cache`
- `recovar/em/dense_single_volume/helpers/projection_cache.py:build_projection_cache`
- `recovar/em/dense_single_volume/helpers/projection_cache.py:_write_projection_cache_rows`
- `tests/unit/test_projection_cache.py`
