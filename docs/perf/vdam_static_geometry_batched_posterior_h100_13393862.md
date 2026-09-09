# VDAM static geometry + batched posterior — H100 job 13393862

## Decision

Retain the composition of shared host-planned half-spectrum geometry and the
shared EM/VDAM batched exact-posterior CUDA primitives.  Against the comparable
static-geometry-only replay, batching removes exactly 2,040 GPU kernel launches
and improves warm wall by 4.02%, while all 23 deterministic science fields stay
exact.  Against the pre-composition baseline, the two changes together improve
warm wall by 14.75%, reduce the Nsight capture span by 13.91%, and remove 2,410
kernel launches.

This is a focused iteration-48 composition gate, not a trajectory or default
promotion.  The strict atomic-repeat diagnostic is 12/21; all misses remain at
the existing nondeterministic CUDA-reduction scale, with maximum normalized L2
`1.53e-7`.  The separate fresh trajectory remains the science authority.

## Comparable performance

Jobs `13393276` and `13393862` replay the same GF46 iteration-47 checkpoint on
the same H100 node (`della-h19g1`) and clean source `21402b859`.  Both use the
`all_optimized_q32` contract, stable Fourier quantum 32, static host geometry,
the same native RELION capture, and cold compile attribution.  The only
candidate change is
`RECOVAR_RELION_BATCHED_POSTERIOR_PRIMITIVES=0 -> 1`.

| Metric | Static only `13393276` | Static + batched `13393862` | Change |
|---|---:|---:|---:|
| Warm profiled wall | 4.569264 s | 4.385716 s | **-0.183549 s (-4.02%)** |
| Cold profiled wall | 29.914641 s | 28.945001 s | -0.969640 s (-3.24%) |
| Nsight capture span | 1.507044 s | 1.453764 s | **-0.053279 s (-3.54%)** |
| GPU kernel launches | 13,233 | 11,193 | **-2,040 (-15.42%)** |
| GPU kernel work | 119.381 ms | 117.986 ms | -1.395 ms (-1.17%) |
| stderr XLA compilations | 394 | 394 | 0 |
| recorded compile misses | 389 | 389 | 0 |

Cold wall and compile time are diagnostic because attribution is enabled.  The
compile counts are identical; the observed compile-time reduction (`22.717 ->
21.977 s`) is ordinary run variation and is not attributed to posterior
batching.  The exact 2,040-launch reduction matches the dedicated posterior
ABBA result and is the causal performance signal.

Relative to original job `13391819`, before static host geometry and posterior
batching were composed:

| Metric | Original `13391819` | Combined `13393862` | Change |
|---|---:|---:|---:|
| Warm profiled wall | 5.144385 s | 4.385716 s | **-14.75% (1.173x)** |
| Cold profiled wall | 30.068897 s | 28.945001 s | -3.74% |
| Nsight capture span | 1.688715 s | 1.453764 s | **-13.91%** |
| GPU kernel launches | 13,603 | 11,193 | **-2,410 (-17.72%)** |
| stderr XLA compilations | 435 | 394 | **-41 (-9.43%)** |
| recorded compile misses | 429 | 389 | **-40 (-9.32%)** |

The combined RECOVAR capture still spans 1.454 s versus 0.852 s for the reused
native RELION capture.  RECOVAR performs only 118.0 ms of GPU kernel work
versus RELION's 220.5 ms, so the remaining gap is controller/launch idle rather
than heavier GPU mathematics.

## Science and execution contracts

| Gate | Result |
|---|---|
| Required exact metadata | **12/12 exact** |
| Required exact scalar state | **10/10 exact** |
| Numeric data STAR content | **exact** |
| Combined deterministic fields | **23/23 exact** |
| Optimized execution contract | **cold exact; warm exact** |
| Atomic repeat envelope | **12/21 inside; diagnostic only** |

Particle IDs, pose/rotation assignments, translations, classes, posterior
maxima, significance counts, direction posterior sums, halfset streams,
resolution, sampling state, and numeric data STAR content are bitwise exact
across static-only cold/warm and combined cold/warm outputs.

The nine envelope misses are `wsum_img_power`, `wsum_sigma2_noise`,
`class_bpref_weight_sums`, their applicable halfset summaries,
`class001_volume`, and numeric model STAR content.  The largest cross/repeat
ratio is 1.571x; the largest normalized L2 delta is `1.531e-7`, and the map
delta is `3.064e-9`.  These are repeat-scale atomic reductions and do not
authorize a trajectory promotion.

The posterior path is shared with mature EM: the VDAM coarse and fine callers
route through `relion_cuda_f32_coarse_posterior` and
`relion_cuda_f32_fine_posterior`, which use the same batched exact CUDA
exponentiate/sort-scan/divide primitives.  This result therefore reduces code
duplication as well as launch overhead.

## Provenance

- Slurm: `13393862`, `COMPLETED`, `00:01:39`, `della-h19g1`.
- Candidate source: `21402b8592f9cc77ae837ef08a583b026c135fc0`.
- Candidate root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_static_geometry_batched_cold_21402b859_20260903T174715Z`.
- Static-only root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_static_geometry_runtime_stack_21402b859_20260903T1734Z`.
- Original root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_fused_half_expand_cold_71513bc86_20260903T1742Z`.
- Checkpoint optimiser SHA-256:
  `addfec8d36581f9c2e02504b29bbca979f9222132eb1a60ccd534a0ed3564e25`.
- Reused native Nsight SHA-256:
  `6d13813ddf22e6062163470978d8f2bd8e41f1fcacd23cc58c27e29bcc1b7ca4`.

The submission is the command recorded for job `13393276`, with a fresh
result root and only:

```bash
RECOVAR_RELION_BATCHED_POSTERIOR_PRIMITIVES=1 \
  sbatch scripts/run_vdam_late_iteration_profile.sbatch
```

Rebuild compile attribution with:

```bash
pixi run python -m scripts.analyze_vdam_cold_compile_attribution \
  --root /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_static_geometry_batched_cold_21402b859_20260903T174715Z \
  --output /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_static_geometry_batched_cold_21402b859_20260903T174715Z/provenance/cold_compile_attribution.json \
  --top 100
```
