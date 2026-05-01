# RELION Parity Agent Notes

This document holds detailed EM/RELION parity notes that used to live in
`recovar/em/CLAUDE.md`. Keep `CLAUDE.md` short and put dated findings,
benchmark details, and diagnostic recipes here or in the current-status doc.

## Current Status Documents

- `docs/math/relion_parity_current_status_2026_04_25.md` tracks measured
  baselines, hardware, Slurm job IDs, artifacts, and open parity gaps.
- `docs/math/relion_initial_model_em_parity_conventions.md` is the canonical
  checklist for the InitialModel-vs-normal-EM convention fixes: adaptive
  support, RELION projector frame, FFT/noise scales, translation-prior units,
  centered rows, and BPref frame conversion.
- `docs/math/relion_parity_roadmap_2026_04_27.md` tracks milestone ordering
  for pass-2 routing, convergence, initialization, large-run reruns, cleanup,
  K-class refinement, and ab-initio work.
- Update the current-status doc whenever a new replay result, source-code
  finding, or dump comparison changes the state of the investigation.

## Active Parity State

Known low-priority boundary issue: the best one-iteration native half-volume
M-step replay matches RELION assignments and maps (`Pmax` mean abs `3.5e-5`,
exact poses/translations, final map corr `0.999996`) and matches BPref through
`rpad<=52` at `~1e-4`, but shell 26/27 BPref boundary voxels still differ.
Do not spend more time on this outermost-shell scatter mismatch unless later
end-to-end parity points back to it.

Particle 933 at iter 2 is a boundary-stress case. It remains a large Pmax
outlier even when RECOVAR and RELION use the same two rotation/translation
candidates and priors. Direct projection of the RELION half-map through
RECOVAR's projector reproduces RELION fine-reference projections to `~1e-7`,
so projection/scoring is not the root cause. The score gap is driven by
high-shell map residuals, mainly shells 26-28 at `current_size=58`. Explicitly
zeroing projection/image pixels in those shells does not reproduce RELION and
should not be used as a fix. Use less boundary-dominated particles for the next
M-step/tau2/noise parity trace.

2026-04-27 tau2/noise update: RECOVAR mirrors RELION's per-half
`BackProjector::updateSSNRarrays` ordering. FSC is shared across halves, but
each half's sigma2/tau2/data-vs-prior comes from that half's own BPref weight,
not the average of the two halves. On the 5k/128 replay, this closes the broad
shell 14-34 tau2/sigma2 mismatch; only outer support shell 35 remains.

2026-04-27 convergence update: replay/refine convergence state must not start
from sentinels. RELION replay initializes from the previous
`run_itNNN_optimiser.star` and `run_itNNN_half1_model.star`, including current
resolution, no-resolution-gain count, no-large-hidden-variable-change count,
smallest change trackers, and optimiser accuracy estimates. Non-replay RELION
mode seeds starting current resolution from `init_fsc` or `ini_high`.

## Recent Source Findings

The RELION accelerated path uses `Projector::initialiseData(current_size)` with
`r_max=current_size/2`, CUDA texture linear interpolation, direct diff2
scoring, and FFTW-style centered complex image FFTs. RECOVAR RELION-parity
refinement routes dense/local EM projection helpers through the RELION
texture-interpolation projector directly.

RELION M-step parity depends on these source details:

- `BackProjector::getDownsampledAverage` uses RELION `ROUND`
  (round-half-away-from-zero), not NumPy banker rounding.
- `BackProjector::getLowResDataAndWeight` /
  `setLowResDataAndWeight` join low-resolution half accumulators by squared
  radius `k*k+i*i+j*j <= ROUND(padding_factor * lowres_r_max)^2`, not by
  rounded shell labels.
- `BackProjector::updateSSNRarrays` averages Fourier weight voxels only with
  `r2 < ROUND(r_max * padding_factor)^2`, where `r_max=current_size/2`.
- `BackProjector::calculateDownSampledFourierShellCorrelation` bins by
  `ROUND(R)`, but first skips exact native radii with `R > r_max`.

FSC timing fix, commit `5097ded6`: RELION computes the current iteration FSC
from M-step BPref accumulators before `updateSSNRarrays`
(`ml_optimiser_mpi.cpp:4031, 4091`; `backprojector.cpp:1044`). RECOVAR uses a
hybrid FSC choice in `iteration_loop.py`: previous-iter FSC by default, and
current-iter fresh FSC only when `max(|prior_fsc|) < 1e-3` to handle cold start
from `init_fsc=zeros`. See
`docs/math/relion_updateSSNR_algorithm_2026_04_25.md`.

## Recent Replay Checkpoints

Tiny 1k / 64^3 replay with automatic defaults:
`_agent_scratch/20260426_tiny1k_auto_parity_15715`, local A100, 69.5s,
mean abs Pmax `3.68e-5`, max abs Pmax `8.70e-4`, exact pose parity,
recovar-vs-RELION map corr `0.999964`.

RECOVAR-side float64 replay:
`_agent_scratch/20260426_tiny1k_float64_replay_25714`. The p668 pre-prior
score deltas remained `[-4.60e-4, 0, -2.02e-4, +6.47e-5, -2.65e-4,
-1.52e-4, +8.40e-5]`, so this is likely RELION accelerated float32/texture
arithmetic unless a RELION CPU/double dump proves otherwise.

Tiny 1k / 64^3 5-iteration replay after M-step/FSC fixes:
`_agent_scratch/codex_tiny5_joinboundary_20260426_105052_10436`, local A100
GPU 2. Final recovar-vs-RELION half-map corr: half1 `0.999970`, half2
`0.999969`. Pmax mean abs gaps by iter: `3.53e-5`, `9.32e-3`, `6.22e-3`,
`4.77e-3`, `6.09e-3`.

Tiny 2-iteration replay with `updateSSNRarrays`/FSC boundary fixes:
`_agent_scratch/codex_pmax_sentinels_fsc_rmax_20260426_185332_27278`, local
A100 GPU 3, `JAX_ENABLE_X64=1`. Final recovar-vs-RELION corr `0.999998`;
iter-1 tau2 shell 28 matches RELION zero support. Iter-2 Pmax remained open:
mean abs `0.005059`, max abs `0.276175`, corr `0.957262`.

5k / 128^3 long end-to-end baseline:
`_agent_scratch/long_end2end_parity_20260426_182134`, Slurm job `7383509`,
A100 node `della-l07g4`, elapsed `2075.6s`, branch
`claude/relion-parity-local-search-fix`, commit
`949ab6b84a40bab5011024689c15492414c4e6ce`. Final half-map corr vs RELION:
half1 `0.996346`, half2 `0.996437`. Pmax mean abs gaps by RELION iter 2-9:
`0.00109`, `0.00634`, `0.00654`, `0.00620`, `0.01383`, `0.01920`,
`0.03473`, `0.04248`.

## Diagnostic Harness

Dense/local EM fast guardrail:

```bash
pixi run test-em-fast-guard
```

It uses tiny synthetic/unit fixtures, defaults to CPU (`JAX_PLATFORMS=cpu`),
and should finish in under about 60 seconds. For local GPU execution, check
`nvidia-smi` first and run:

```bash
EM_FAST_GUARD_BACKEND=gpu pixi run test-em-fast-guard
```

`recovar/em/dense_single_volume/parity_dump.py` is an env-gated per-iteration
dump writer. Set `RECOVAR_PARITY_DUMP_DIR=<path>` to write `iter_NNN.npz`
files with metrics, per-half arrays, accumulators, map/FSC/noise state, and
timings when stage timers are wired. It has zero overhead when unset.

RELION dump and comparison scripts:

- `scripts/parity/dump_relion_iter.py`
- `scripts/parity/compare_dumps.py`

Pre-prior RECOVAR per-pose dump:

```bash
RECOVAR_DEBUG_PER_POSE_DUMP_DIR=<dir>
RECOVAR_DEBUG_PER_POSE_DUMP_TARGET=<image_idx>
RECOVAR_DEBUG_PER_POSE_DUMP_PREPRIOR=1
```

This captures scores before prior addition for apples-to-apples comparison
with RELION `exp_Mweight_diff2.bin`.

Use `--adaptive_fraction 1.0` to disable sparse pass-2 significance pruning
and route through the full-grid branch in `_run_relion_iteration_loop`. This is
useful for isolating sparse-vs-dense normalization differences.

## Fixtures And Performance Traps

Tiny fixture for fast debug:
`/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_tiny_parity/`.
It has 1k particles, 64^3 box, and 16 RELION iterations at
`relion_ref_os0/`. Most microtests should use this with `--max_particles 100`
for sub-30-second iterations.

Do not use the 5k/128 fixture for iterative debugging. The cold sparse-pass-2
path can take around 50 minutes there if bucketed batching does not activate.

Sparse-pass-2 perf trap: `compute_pass2_stats_sparse` used to call
`run_em(image_batch_size=1, ...)` in a per-image Python loop, causing 5000 JIT
compiles per iter on the 5k fixture. Shape-bucketed batching landed in commits
`66989c86` and `12f1a7c3` via `helpers/sparse_pass2_bucketed.py`. If logs show
`[NOISE-DIAG] sumw=1` per batch, the bucketed path likely did not activate;
check `helpers/oversampling.py:compute_pass2_stats_sparse`.

For the standard 5k replay starting from RELION iteration 3 and comparing
through RELION iteration 14, use:

```bash
scripts/run_multi_iter_parity.py --iter 3 --max_iter 11
```

`--max_iter` is the number of emitted RECOVAR iterations, not the final RELION
iteration number. Passing `--max_iter 14` asks for metadata through
`run_it017_*`.

After replay, run `scripts/parity/check_perf.py` on the dump directory against
`tests/baselines/parity/perf_baseline_5k_128_a100.json`. Warning-level perf
output is still useful and should be reported with the dump path.

## RELION Volume Convention

RECOVAR and RELION use different 3D coordinate frames:

```python
vol_recovar = -np.transpose(vol_relion, (2, 1, 0))
```

Canonical helpers in `recovar/utils/helpers.py`:

- `load_mrc(path)` / `write_mrc(path, vol)` for RECOVAR, cryoSPARC, and
  cryoDRGN-frame MRCs.
- `load_relion_volume(path)` for RELION MRCs before comparison with RECOVAR
  output.
- `relion_volume_to_recovar(vol)` / `recovar_volume_to_relion(vol)` for
  explicit frame conversion.
- `R_to_relion(R)` / `R_from_relion(euler)` for rotation conversion. These are
  intentionally paired with the volume transpose. Do not change them casually.

For FSC against a RELION reference:

```python
from recovar.utils.helpers import load_relion_volume, load_mrc

relion_ref = load_relion_volume("relion_output/run_class001.mrc")
recovar_vol = load_mrc("recovar_output/final_merged.mrc")
```

The convention is pinned by `tests/unit/test_relion_volume_convention.py`.

## RELION Reference Flags

RELION command-line defaults differ from GUI auto-refine defaults. The
authoritative source is `relion/src/pipeline_jobs.cpp::initialiseAutorefineJob`
and `relion/src/ml_optimiser.cpp::parseInitial()`.

Add these flags to `relion_refine_mpi --auto_refine` CLI parity invocations:

- `--ctf`
- `--firstiter_cc`
- `--flatten_solvent`
- `--zero_mask`
- `--low_resol_join_halves 40`
- `--norm`
- `--scale`

Diagnostic:

```bash
grep -E "_rlnDoCorrectCtf|_rlnRefsAreCtfCorrected|_rlnDoNormCorrection|_rlnDoScaleCorrection" \
  <relion_run_dir>/run_it000_optimiser.star
```

If `_rlnDoCorrectCtf` is `0`, the run forgot `--ctf` and the reconstruction is
silently wrong for parity.

Canonical shape of a RELION parity invocation:

```bash
mpirun -n 3 relion_refine_mpi \
  --i particles.star \
  --ref reference_init_relion.mrc \
  --o run \
  --auto_refine --split_random_halves \
  --particle_diameter 200 --ini_high 30 \
  --ctf --firstiter_cc --flatten_solvent --zero_mask \
  --low_resol_join_halves 40 --norm --scale \
  --healpix_order 3 --offset_range 3 --offset_step 1 \
  --oversampling 1 --pad 2 --gpu 0 --j 4
```

## RELION Iter-1 Pmax

RELION iter-1 `ave_Pmax = 1.0` with `--firstiter_cc` or `--always_cc` is a
winner-take-all binarization artifact in `ml_optimiser.cpp:7775-7803`, not
Bayesian inference. The CC scoring path is scale-invariant to absorb intensity
scale mismatch from non-RELION init volumes.

Do not add a hard-CC iter-1 path to RECOVAR's `_run_relion_iteration_loop` just
to match this number. The compounding effect on iter 2+ via the iter-1 volume
is real, so compare downstream behavior explicitly.

## Architecture Notes

Dense homogeneous RELION-parity code lives in `dense_single_volume/`:

- `iteration_loop.py`: `refine_single_volume`, `_run_relion_iteration_loop`
- `em_engine.py`: two-pass JIT engine for E-step scoring and M-step
  accumulation
- `helpers/types.py`: stats containers
- `helpers/convergence.py`: angular/translational convergence detection
- `helpers/oversampling.py`: two-pass adaptive oversampling
- `helpers/fourier_window.py`: Fourier cropping to current resolution
- `helpers/local_search.py`: local search helpers
- `helpers/orientation_priors.py`: RELION-mode prior construction
- `helpers/resolution.py`: initialization and coarse-size helpers
- `helpers/significance.py`: batched significance computation

Older shared EM files:

- `core.py`: cross-correlation, dot products, probability utils
- `e_step.py`: `E_with_precompute`
- `m_step.py`: `M_with_precompute`, `sum_up_images_fixed_rots_eqx`
- `iterations.py`: `E_M_batches_2`, `split_E_M_v2`
- `states.py`: `EMState`, `SGDState`, `HeterogeneousEMState`
- `sampling.py`: HEALPix and translation grids
- `noise.py`: RELION-parity noise estimation
- `regularization.py`: tau2 prior, FSC, Wiener regularization
- `heterogeneity.py`: low-rank heterogeneity EM; separate owners

Key computations:

- E-step cross-term: `cross[i,r,t] = -2 Re <S_t(CTF*y_i/sigma^2), P_r mu>`.
  The GEMM path creates shifted images and multiplies against projections for
  dense-grid reuse.
- M-step accumulation: `Ft_y += sum gamma * P_r*(S_t* CTF*y_i/sigma^2)`.
  The image/translation sum is done by GEMM before backprojection.
- Translation handling has GEMM and FFT paths. GEMM is best for dense batched
  rotations; FFT is useful for single-rotation refinement.

## Performance Notes

Historical A100-80GB benchmark, 5000 images, 128 px, order 3, 7x7
translations:

| Engine | Time | vs old |
|---|---:|---:|
| Old `E_with_precompute` + `M_with_precompute` | 68s | 1x |
| `engine_fused.py` | 26s | 2.6x |
| `em_engine.py` | 29s | 2.3x |
| Half-spectrum GEMMs | 19s | 3.6x |

High-priority optimization themes:

1. Fourier cropping to current resolution.
2. Two-pass adaptive oversampling.
3. Significant-weight pruning.

RELION half-spectrum Hermitian weights: RELION sums over the rfft half-image
with weight 1 for all pixels. That is not the mathematically exact full
Gaussian likelihood, but RECOVAR matches RELION for parity through
`make_scoring_half_image_weights(..., relion_half_sum=True)`.
