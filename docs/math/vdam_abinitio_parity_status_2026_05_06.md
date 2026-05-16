# VDAM / InitialModel ab-initio parity status - 2026-05-06

This document records the current ab-initio / VDAM parity checkpoint on branch
`codex/vdam-after-merge-20260506`.  The active goal remains near-perfect RELION
InitialModel parity, with `0.999` map parity as the short-term target at least
at iter 1 and then across full end-to-end schedules.

## Current commits

- `4fc63f1a` - `Add native VDAM InitialModel long guard`
- `1c8da2ef` - integration base `codex/em-vdam-ppca-integration-final-20260506`
- `23a51dcf` - `em-initialmodel: match relion seed-zero subset order`
- `8fd73603` - `em-initialmodel: apply relion bootstrap postprocess`
- `64595e15` - `em-initialmodel: add aligned GT benchmark metrics`
- `8b4a736f` - `em-initialmodel: use finer aligned GT benchmark search`
- `58bedd35` - `em-initialmodel: document vdam parity state`

## Evaluation convention

InitialModel output MRCs use the RELION output frame.  For ad-hoc direct Python
comparisons against recovar-frame arrays, load both RELION and native
InitialModel MRCs with:

```python
from recovar.utils import helpers
vol = helpers.load_relion_volume("run_it008_class001.mrc")
```

Using `helpers.load_mrc()` on these InitialModel MRCs applies the wrong
frame/sign for this specific output path and flips the direct correlation sign.
For older 8-iter runs below:

- `load_relion_volume(native_it008)` vs `load_relion_volume(RELION_it008)`:
  `CC = 0.996258`
- `load_mrc(native_it008)` vs `load_relion_volume(RELION_it008)`:
  `CC = -0.996258`

## Current K=1 50k / 256-pixel checkpoint after integration merge

This is the current production-scale native VDAM InitialModel status on
`codex/vdam-after-merge-20260506`, after rebasing on the EM/VDAM/PPCA
integration branch.

### 2026-05-12 integration branch schedule check

On integration branch `codex/vdam-initial-volume-parity` at base commit
`5e90b143d7b6176c50dec8bec2a3633385e09cb5` plus the inclusive VDAM sigmoid
schedule fix, the same 50k/256 K=1 `nr_iter=8` run reproduces the accepted
May 6 parity band.

RECOVAR run:

`/scratch/gpfs/GILLES/mg6942/_agent_scratch/codex_vdam_initial_parity_20260512/k1_50k8_fix_8146395`

Slurm/log:

- Job: `8146395`
- Log: `/scratch/gpfs/GILLES/mg6942/slurmo/vdam-k1-50k8-fix-8146395.out`
- Wall time: `426.75 s`

The first failed comparison in this session used `--nr_iter 3` against the
8-iteration RELION reference. That mismatched VDAM schedules
(`1833/3417/5000` selected particles instead of RELION's
`250/250/1200` for iters 1-3) and is not a parity result.

Schedule source fix:

- RELION's short 8-iteration InitialModel reference uses the inclusive
  in-between span for sigmoid schedules. Stepsize/tau2 at iters 1-4 are
  `0.899960/1.000003`, `0.896040/1.029703`, `0.700000/3.970297`,
  `0.503960/3.999997`.
- RECOVAR now uses `grad_inbetween_iter - 1` in the VDAM stepsize and tau2
  sigmoid lengths, while leaving the subset-size ramp unchanged.

Direct native-vs-RELION map parity, loading both maps with
`helpers.load_relion_volume()`:

| Iteration | CC vs RELION | Scale | Residual/std | Delta vs May 6 status |
|---:|---:|---:|---:|---|
| 1 | `0.997919083` | `1.020772` | `0.064479` | same (`+0.000003`) |
| 2 | `0.999005145` | `1.036257` | `0.044595` | same (`+0.000006`) |
| 3 | `0.998927025` | `1.026496` | `0.046312` | not previously listed |
| 4 | `0.998764692` | `1.022317` | `0.049690` | same (`+0.000259`) |
| 5 | `0.998483144` | `1.019283` | `0.055058` | not previously listed |
| 6 | `0.998080996` | `1.016842` | `0.061922` | not previously listed |
| 7 | `0.997549550` | `1.014715` | `0.069964` | not previously listed |
| 8 | `0.996847267` | `1.012668` | `0.079344` | same (`+0.000032`) |

Conclusion: the integration branch had not regressed the accepted 50k/256
K=1 VDAM parity after the schedule fix. The remaining iter-8 gap to the strict
`0.999` target is still the known multi-iteration state-evolution gap, not a
post-merge regression.

Native run:

`/scratch/gpfs/GILLES/mg6942/_agent_scratch/native_vdam_solventfix_nr8_50k256_20260506_234219_27563`

RELION InitialModel reference:

`/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_50k_256_normalized/relion_initialmodel_k1_it008_write1`

Command shape:

```bash
pixi run python scripts/run_ab_initio.py \
  --i /scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_50k_256_normalized/particles.star \
  --datadir /scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_50k_256_normalized \
  --o <scratch>/out/run \
  --nr_iter 8 \
  --K 1 \
  --sym C1 \
  --particle_diameter 200 \
  --tau2_fudge 4 \
  --random_seed 0 \
  --healpix_order 1 \
  --oversampling 1 \
  --offset_range 6 \
  --offset_step 2 \
  --image_batch_size 16 \
  --rotation_block_size 256 \
  --padding_factor 1 \
  --eager_images
```

Direct native-vs-RELION map parity, loading both maps with
`helpers.load_relion_volume()`:

| Iteration | CC vs RELION | Scale | Residual/std |
|---:|---:|---:|---:|
| 1 | `0.997915757` | `1.020716` | `0.064530` |
| 2 | `0.998998945` | `1.033677` | `0.044734` |
| 3 | `0.998605486` | `0.983939` | `0.052793` |
| 4 | `0.998506149` | `0.980544` | `0.054639` |
| 8 | `0.996815765` | `0.986096` | `0.079739` |

The large 50k/256 iter-1 regression was missing RELION's post-M-step
`solventFlatten()` before writing iteration artifacts.  A RELION debug dump
showed that multiplying `mstep_it1_c0_iref_after.bin` by the centered
spherical raised-cosine solvent mask reproduces RELION's written
`run_it001_class001.mrc` at `CC=1.0`.  The native driver now applies the same
post-M-step hook when `do_solvent` / `--flatten_solvent` is enabled.

Before this fix, the schedule-matched direct native-vs-RELION iter-1 map CC on
the same fixture was approximately `0.902`.  The corrected 8-iteration run
reaches `0.997916` at iter 1 and `0.998999` at iter 2, but it still misses the
strict `0.999` direct-map target by iter 8.  The remaining gap is therefore in
multi-iteration E-step / BPref accumulator evolution, not in the final
artifact frame or a missing solvent mask.

The long guard intentionally still requires direct native-vs-RELION iter-8 map
CC `>= 0.999`; do not relax this threshold.  It is the active target.

## Alignment-aware GT metric

Ab-initio maps have arbitrary global orientation and may require a handedness
choice before comparing to simulation GT.  The benchmark now has opt-in
alignment-aware metrics:

```bash
pixi run python scripts/run_multi_iter_parity.py \
  --relion_dir <relion_dir> \
  --data_star <particles.star> \
  --iter <start_iter> --max_iter <n> \
  --gt_volume <reference_gt.mrc> \
  --output_dir <out> \
  --gt_align
```

Defaults:

- `--gt_align_healpix_order 2`
- `--gt_align_max_shell 8`
- mirror/handedness test enabled
- global sign flip disabled
- merged-map alignment only unless `--gt_align_all_series` is set

The order-2 default is intentionally one HEALPix order finer than the native
InitialModel run that produced the current K=1 maps (`--healpix_order 1`
with `--oversampling 1`).  The scorer Fourier-crops the low-shell box before
the rotation sweep, then rotates the full map once for the winning orientation.

Standalone InitialModel MRCs can be evaluated with:

```bash
pixi run python scripts/evaluate_ab_initio_gt.py \
  --volume <run_itNNN_class001.mrc> \
  --label run_itNNN_class001 \
  --gt_volume <reference_gt.mrc> \
  --volume_frame relion \
  --gt_frame recovar \
  --gt_align \
  --output_npz <out>/abinitio_gt_metrics.npz \
  --output_json <out>/abinitio_gt_metrics.json
```

Use `--volume_frame relion` for native `scripts/run_ab_initio.py` outputs,
because those MRCs are intentionally written through `write_relion_mrc()`.

## Current K=1 500-particle / 64-pixel end-to-end checkpoint

Fixture:

`/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar/.tmp/slurm_7178672/pytest-of-mg6942/pytest-0/test_pipeline_spa_gpu0/gpu_spa/test_dataset/particles.star`

Run outputs:

- native: `/scratch/gpfs/GILLES/mg6942/_agent_scratch/codex_initialmodel_iter8_20260506_021545/recovar`
- RELION: `/scratch/gpfs/GILLES/mg6942/_agent_scratch/codex_initialmodel_iter8_20260506_021545/relion`

Native-vs-RELION map correlation:

| Iteration | CC vs RELION |
|---:|---:|
| 1 | `0.999907707` |
| 4 | `0.999059067` |
| 5 | `0.998943858` |
| 6 | `0.998615233` |
| 7 | `0.997849287` |
| 8 | `0.996257990` |

The iter-1 target is effectively met on this fixture.  The full 8-iter
trajectory still drifts below the `0.999` target, so the remaining parity work
is multi-iteration state evolution rather than one-step output plumbing.

## Order-2 aligned GT FSC for the same checkpoint

GT target:

- weighted simulation mean with state counts `[104, 103, 193]`
- outlier count `100`
- voxel size `8.5 A`

Alignment:

- `healpix_order=2`
- `4608` rotations
- score shells `<= 8`
- mirror enabled
- sign disabled

| Map | Raw mean FSC 1-8 | Aligned mean FSC 1-8 | Aligned mean FSC 1-16 | Aligned FSC<0.5 | Aligned FSC<0.143 |
|---|---:|---:|---:|---:|---:|
| RECOVAR it001 | `0.216041` | `0.332996` | `0.168525` | shell `3` / `181.33 A` | shell `6` / `90.67 A` |
| RECOVAR it008 | `0.315276` | `0.437997` | `0.216559` | shell `5` / `108.80 A` | shell `7` / `77.71 A` |
| RELION it008 | `0.334579` | `0.462366` | `0.234475` | shell `5` / `108.80 A` | shell `7` / `77.71 A` |

Interpretation:

- RECOVAR reaches the same aligned `FSC<0.143` shell as RELION by it8 on this
  small fixture.
- RECOVAR still trails RELION in low-shell FSC amplitude:
  `0.024369` lower over shells 1-8 and `0.017916` lower over shells 1-16.
- This is consistent with the native-vs-RELION map CC drift by it8.

## Regression guards added

The aligned-GT evaluator is covered by unit tests that lock:

- exact recovery for a known grid rotation
- handedness handling when mirror testing is enabled
- no hidden global sign flip unless explicitly enabled
- `-1` sentinel for FSC threshold not crossed
- low-shell Fourier crop shape and self-correlation
- default alignment order `2`, max score shell `8`, and order-2 rotation count
- order-2 alignment materially improves over a too-coarse order-0 grid for a
  synthetic map rotated by an order-2 grid orientation

Relevant command:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
  pixi run python -m pytest -v tests/unit/initial_model/test_gt_metrics.py
```

## Larger fixture queued next

The available production-scale K=1 fixture is:

`/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_50k_256_normalized`

It contains:

- `particles.star`
- `particles.256.mrcs`
- `reference_gt.mrc`
- `reference_init.mrc`
- `simulation_info.pkl`
- RELION reference trajectory under `relion_ref_os0/` through iteration 14

Use Slurm for this tier.  Do not run project-wide `long-test` for EM-only
changes; use EM-scoped long parity or a dedicated InitialModel job.

The immediate larger stress case is a short VDAM InitialModel run on the full
50k / 256-pixel stack:

```bash
pixi run python scripts/run_ab_initio.py \
  --i /scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_50k_256_normalized/particles.star \
  --o <scratch>/recovar/run \
  --nr_iter 3 \
  --K 1 \
  --sym C1 \
  --particle_diameter 200 \
  --tau2_fudge 4 \
  --random_seed 0 \
  --healpix_order 1 \
  --oversampling 1 \
  --offset_range 6 \
  --offset_step 2 \
  --image_batch_size 128 \
  --rotation_block_size 5000 \
  --padding_factor 1
```

Then evaluate all saved classes with `scripts/evaluate_ab_initio_gt.py` using
the command pattern above.  This tests more images and higher resolution than
the current 500-particle / 64-pixel checkpoint while staying focused on the
ab-initio VDAM path.
