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

2026-07-25 K=4 restart-precision finding: never use the rounded
`_rlnSamplingPerturbInstance` from a sampling STAR as the arithmetic input to
a strict boundary replay.  For the iteration-10 continuation from
`run_it009`, the STAR says `-0.12306`, while RELION's seed-exact live value is
`-0.12305957078933716` for random seed `1778628798` and restart state
iteration 9.  The `4.2921066e-7` difference is enough to move outer
current-size support decisions.  With the rounded value, 1,009/1,080 captured
matrix entries differ and 258 one-sided pixels appear on shell 37; with the
seed-exact value, RECOVAR's existing host-grid path matches all 1,080
captured float32 entries bit-for-bit.  `scripts/run_k_class_parity.py` now
defaults to seed-exact recovery and requires
`--perturb-restart-state-iteration` for an explicit continuation.  Use
`--perturb-replay-precision star` only as a diagnostic.  Slurm provenance:
matrix substitution `11584294`, matrix-origin probe `11584445`, run root
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_relion_matrix_device_substitution_7eb5e1be_20260725T014500Z`,
runtime root
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k4_it10_relion_matrix_device_substitution_7eb5e1be_20260725T014500Z`;
both roots contain `SAFE_TO_DELETE`.

2026-06-30 old-offset rounding finding: RELION's SPA accelerated path applies
`my_old_offset.selfROUND()` before scoring and writeback, and `ROUND(x)` is
half-away-from-zero (`(x > 0) ? int(x + 0.5) : int(x - 0.5)`). RECOVAR had
used NumPy `rint` for the dense/local `relion_translation_search_base` and
InitialModel native pre-shift/update paths, which is banker rounding at
half-integers. Adaptive translation children commonly produce `.5` offsets, so
the mismatch can deterministically move the next local search center by one
pixel. Use `relion_round_away_from_zero` for RELION old-offset image
pre-shifts and absolute translation writeback, and do the rounding decision
before any float32 downcast of STAR-parsed offsets; keep NumPy rounding only
for unrelated radial shells or already-integral validation.

2026-06-29 K-class score-dump finding: in non-split MPI Class3D/K-class runs
(`_rlnDoSplitRandomHalves 0`), follower runtime state can disagree with
`run_itNNN_model.star` for per-group scale corrections. In the K=4 case8
10k/128 stack-2347 diagnostic, six identical RELION reruns with the same seed
scored the same dumped particle on either MPI rank 1 or rank 2. Rank-1 dumps
used runtime scale around `0.756-0.757`; rank-2 dumps used runtime scale around
`0.998-0.999`. Only rank 1 writes the numbered model file in this mode, and
`scale_correction` is not broadcast to all followers in the same way as
`Iref`, `data_vs_prior_class`, `fourier_coverage_class`, and `sigma2_class`.
Exact score-surface replay from RELION dumps must therefore use
`pass0_img0_scale_correction.bin` for the dumped particle, not just the scale
in `run_itNNN_model.star`. After applying that runtime scale and remapping
RECOVAR rotation order to RELION order (`pixel * n_psi + psi`), matched
RECOVAR-vs-RELION score surfaces for both scale branches had the same top pose
`(class=1, rot=210, trans=7)`, correlation `~0.998`, and zero-intercept slope
`~1.02`. Artifacts:
`/scratch/gpfs/CRYOEM/gilleslab/_agent_scratch/em_k4_case8_rankscale_replicates_20260629_191257`
and
`/scratch/gpfs/CRYOEM/gilleslab/_agent_scratch/em_k4_case8_matched_recovar_replays_20260629_191832`.

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

## 2026-06-29 K=1 Fine-Pass M-Step Pruning

RELION accumulates noise/norm residuals from weighted squared differences only
over retained fine-pass significant samples. Therefore the image-power term in
`|I - A|^2 = |I|^2 + |A|^2 - 2 Re(A*I)` must be multiplied by the retained
posterior support mass for each image. K-class fused sparse pass-2 already did
this; K=1 sparse pass-2 was still adding full image power while A2/XA used the
pruned support. That creates a systematic reconstruction/norm bias: poses and
RECOVAR-vs-RELION map FSC can remain very close, but GT FSC-AUC can trail by a
repeatable small amount on cases such as anisotropic 100k case7.

The contract is now pinned in
`tests/unit/test_sparse_pass2_bucketed_perf.py::test_k1_relion_fine_mstep_prune_weights_image_power_by_retained_mass`
and the rotation-chunked path is covered by
`test_sparse_pass2_rotation_chunking_matches_unchunked_windowed_path`.

## 2026-06-30 Exact RELION Projector Scoring

A consistent K=1 quality/Pmax gap traced to projection, not priors, rotation
order, translation priors, or Fourier support. Dumped RELION `PPref` projected
through RECOVAR's exact `relion_project_half` matched RELION fine reference
rows at about 5e-4 relative error, while RECOVAR's padded Fourier slicer was
about 2-3% off for the same target particle. The important detail is that
RELION's `PPref` is cropped to `current_size`; projecting it as a full-box
projector is wrong. The exact path now projects at `2*r_max` and scatters the
cropped FFTW rows into RECOVAR's full-box centered half-spectrum layout before
the usual Fourier window gather.

RELION-mode adaptive and local scoring now build `Projector::data` slabs from
the current half-map references and pass them through coarse significance,
sparse pass-2, fused K-class sparse pass-2, and local search. InitialModel
RELION-projector-frame E-steps use the exact projector by default, with
`RECOVAR_INITIAL_MODEL_EXACT_RELION_PROJECTOR=0` retained only as a diagnostic
opt-out.

Regression coverage:

- `tests/unit/test_relion_project_half.py::test_centered_row_projector_scatters_cropped_ppref_into_full_box`
- `tests/unit/test_em_kclass_merge_guards.py::test_sparse_pass2_preserves_relion_projector_api_and_forwarding`
- `tests/unit/initial_model/test_dense_adapter.py::test_resolve_class_inputs_relion_projector_uses_exact_path_by_default`

## 2026-06-30 K=1 RELION X-Half M-Step BPref Layout

K=1 adaptive RELION mode now defaults to RELION x-half BPref-layout
accumulation for the fine-pass M-step. The old native half-volume path
systematically underweighted low shells in case32 accumulator comparisons
against RELION (`shell1` about 0.82x, `shell2` about 0.91x). With x-half
enabled, the same iter-1/iter-2 shell weight sums match RELION at about 1.0x
across shells.

The diagnostic opt-out is `RECOVAR_K1_RELION_X_HALF_MSTEP=0`; launcher dry-run
coverage records that env so old-path comparisons remain reproducible. The
sparse pass-2 logs now distinguish `RELION x-half BPref-layout` from native
half-volume/full-volume accumulation.

This fix does not by itself prove full quality parity. Case32 x-half probes
still show a structured complex-average residual after the global sign/scale
conversion (`~0.30` median coordinate avg relerr at iter 1 and `~0.58` at iter
2), while shell weight sums are fixed. Coordinate-frame scans confirm the
RELION `k,i,j -> RECOVAR i,k,j` mapping is the best mapping, so the residual is
not a comparison-axis artifact.

Follow-up bisection records the actual M-step reconstruction-window operands in
sparse pass-2 dumps: `shifted_recon`, `ctf2_over_nv_recon`, and
`recon_window_indices`. These are distinct from the score-window operands when
`score_pixels != recon_pixels`, and are required to decide whether the remaining
gap comes from candidate probabilities/support or from adjoint/BPref
contribution layout.

Regression coverage:

- `tests/unit/test_em_kclass_merge_guards.py::test_k1_relion_x_half_mstep_defaults_on_with_escape_hatch`
- `tests/unit/test_run_em_k1_robustness_matrix_slurm.py::test_k1_relion_x_half_mstep_diagnostic_env_is_forwarded`
- `tests/unit/test_em_kclass_merge_guards.py::test_sparse_pass2_dump_writes_score_and_recon_operand_arrays`
- `tests/unit/test_compare_iter1_bpref_accum.py::test_coordinate_mapping_scan_ranks_expected_kij_to_ikj_mapping_first`

## 2026-06-30 K=1 Local-Search X-Half M-Step Propagation

The late K=1 case11/case32 tau2 drift was not random numerical noise. In both
repro cases RECOVAR and RELION shell weights matched exactly before local
search, then the first local M-step switched back to native half-volume
backprojection and introduced an almost uniform factor-of-two weight inflation.
Case32 matched through iter 5, then showed median `avg_weight` ratio
`1.964x` and median `sigma2` ratio `0.509x` for iterations 6-11. Case11 showed
the same pattern starting at iter 4 (`avg_weight` about `1.98x`, `sigma2`
about `0.50x`). The maps remained highly correlated with RELION, so map FSC
alone did not expose the accumulator contract regression.

Post-fix Slurm validation confirms the same shell contract on an independent
very-high-noise case. Job `10459657` (`case12`, 3k particles, grid 128,
white-noise scale 10, uniform poses) ran after the local-search x-half
propagation change and logged both:
`RELION local K=1 M-step: using x-half BPref-layout backprojection` and
`Exact local M-step: using RELION x-half BPref-layout backprojection`. Its
iteration-by-iteration audit against RELION half1 model stars gives median
RECOVAR/RELION `sigma2` ratio `1.000` and median `avg_weight` ratio `1.000`
for every iteration, including all local-search iterations. The pre-fix
controls in root
`/scratch/gpfs/CRYOEM/gilleslab/_agent_scratch/em_k1_tau2_axis_patch_quality_20260630_033329`
end at `0.504/1.984` for case11 and `0.509/1.964` for case32, while the
post-fix case12 in root
`/scratch/gpfs/CRYOEM/gilleslab/_agent_scratch/em_k1_broad_quality_20260630_023700`
ends at `1.000/1.000`.

Follow-up small-axis cases show the factor-of-two bug is gone, but parity is
not perfect. In the same broad root, patched case13 (nonuniform poses, white
noise 3) ends at median half1 `sigma2/avg_weight = 0.996184/1.003830`;
patched case16 (nonuniform poses plus 25 percent outliers, white noise 3)
ends at `0.996739/1.003271`; patched case15 (20 percent outliers, white noise
1, uniform poses) ends cleanly at `0.999976/1.000024` but has a transient
iteration-10 half1 mismatch of `1.018498/0.981838` before returning to parity
on iterations 11-12. The transient is not a simple adjacent-RELION-iteration
comparison offset. A targeted case13 BPref/updateSSNR diagnostic rerun was
queued as Slurm jobs `10465057` (setup), `10465058` (case), and `10465059`
(summary) under
`/scratch/gpfs/CRYOEM/gilleslab/_agent_scratch/em_k1_case13_bpref_diag_20260630_064945`.
Use those dumps to decide whether the remaining sub-percent residual comes
from candidate/support differences, local-search probability mass, or a shell
count/sum detail.

Post-fix small-case quality is generally RECOVAR-positive but still not strict
RELION equality. RECOVAR-vs-ground-truth FSC-AUC deltas vs RELION were
`+0.002413` (case12), `+0.010014` (case13), `+0.012095` (case15), and
`+0.002600` (case16). RECOVAR wall time remained slower on these small Slurm
jobs, with RECOVAR/RELION wall ratios roughly `3.3x-4.6x`; the E-step/local
search dominates those timings.

`_score_half_local` now enables the same default-on K=1 RELION x-half
BPref-layout M-step as adaptive sparse pass-2 and records
`mstep_full_half_axis=0` so tau2 shell stats use the RELION x-half axis after
public full-volume expansion. The default is conditional on custom CUDA being
available because the x-half indexed adjoint is CUDA-only; CPU/JAX-fallback
unit runs keep the native path unless `RECOVAR_K1_RELION_X_HALF_MSTEP=1` is
explicitly forced. K-class local search keeps the K-class full-volume/default
contract and does not use this K=1 x-half path.

Regression coverage:

- `tests/unit/test_dense_iteration_loop_merge_guards.py::test_k1_local_search_keeps_relion_x_half_mstep_contract`
- `tests/unit/test_dense_iteration_loop_merge_guards.py::test_k1_local_search_passes_relion_x_half_mstep`

## 2026-06-30 K=1 Local-Search Direction Prior Contract

The remaining case13/case16 sub-percent tau2 residual is consistent enough to
treat as a parity bug, not numerical noise. A source-level comparison found
that RELION's `convertAllSquaredDifferencesToWeights` uses
`mymodel.pdf_direction` only in the `NOPRIOR` branch. Local angular searches
run in the explicit rot/tilt/psi prior branch and score
`exp_directions_prior * exp_psi_prior` instead. RECOVAR was still threading the
learned global `pdf_direction` prior into K=1 local-search parent layouts and
also into the parent pass call, which can perturb both pass-1 significant
support and final local weights on anisotropic pose distributions.
For case13, the RELION half-model `pdf_direction` is far from flat once local
search starts: iter-6/iter-9 have 578-702 nonzero directions out of 3072 and
nonzero max/min log-spans of about 13-16 nats. Routing that as an extra local
prior can therefore hard-mask or strongly bias candidates that RELION still
scores under the explicit local Gaussian prior.

`_score_half_local` now forces the local rotation-log-prior input to `None`;
the only angular prior in local search is the image-specific local direction
and psi prior already stored on the `LocalHypothesisLayout`. The iteration
loop also skips learned global/per-class direction-prior construction while
`use_local` is true, and the final all-data local pass does the same. Dense
global/exhaustive scoring still uses `pdf_direction` when RELION is in the
`NOPRIOR` branch.

Regression coverage:

- `tests/unit/test_refine_relion_mode.py::test_score_half_local_parent_layout_ignores_global_rotation_prior_for_adaptive_pass2`
- `tests/unit/test_dense_iteration_loop_merge_guards.py::test_k1_local_search_does_not_score_learned_global_direction_prior`

## 2026-06-30 Local Packed-Noise Projection Tail Chunking

Broad-matrix severe-outlier cases exposed a robustness failure separate from
the quality-prior bug. Case26 (`1k`, radial noise 5, nonuniform poses,
30 percent outliers) OOMed at local iteration 7 after one image expanded to
250,560 fine local candidates; case22 (`3k`, radial noise 5, nonuniform poses,
50 percent outliers) reached final all-data local scoring and OOMed on a
4.59 GiB packed projection/mask allocation. Both failures kept the exact
candidate set and probabilities intact up to the packed local noise projection;
the issue was materializing all packed reconstruction rows for the noise/norm
residual in one JAX block.

`run_local_em_exact` now chunks deferred packed noise projections by a
row-pixel budget (`RECOVAR_EXACT_LOCAL_PACKED_NOISE_TARGET_ROW_PIXELS`,
default `64_000_000`) and accumulates identical shell noise sums and per-image
norm residuals across chunks. This targets severe outlier/local-search tails
without pruning support or changing RELION probability semantics. Patched
Slurm retries were queued as case26 jobs `10466597`/`10466598`/`10466599` under
`/scratch/gpfs/CRYOEM/gilleslab/_agent_scratch/em_k1_local_noise_chunk_case26_20260630_001`
and case22 jobs `10466632`/`10466633`/`10466634` under
`/scratch/gpfs/CRYOEM/gilleslab/_agent_scratch/em_k1_local_noise_chunk_case22_20260630_001`.

Regression coverage:

- `tests/unit/test_refine_relion_mode.py::test_packed_local_noise_projection_chunk_rows_env`
- `tests/unit/test_refine_relion_mode.py::test_exact_local_noise_projection_chunks_packed_tail`

## 2026-06-30 Final All-Data Gridding Correction Default

The small K=1 robustness rows run with RELION-style final output gridding
correction enabled had repeatable negative RECOVAR-vs-RELION GT FSC-AUC
residuals, despite final RECOVAR-vs-RELION map AUC near `0.975-0.991`.
Case11 repeated at `-0.001385` and `-0.001377`, making the residual much
larger than rerun jitter. Post-hoc replay of the dumped final all-data BPref
accumulators isolated the effect to final reconstruction/postprocessing: using
the same `Ft_y/Ft_ctf/tau2` but disabling final gridding correction improved
GT FSC-AUC on every completed small case:

| Case | grid on GT AUC | grid off GT AUC | off-vs-on |
|---|---:|---:|---:|
| 11 baseline 3k | 0.631926 | 0.651514 | +0.019588 |
| 12 very high noise 3k | 0.187979 | 0.190340 | +0.002361 |
| 13 anisotropic 3k | 0.292691 | 0.303600 | +0.010910 |
| 14 no CTF 3k | 0.543126 | 0.560412 | +0.017286 |
| 15 20% outliers 3k | 0.591349 | 0.609959 | +0.018610 |
| 16 anisotropic + outliers 3k | 0.275118 | 0.283227 | +0.008109 |

The grid-off maps remain close to RELION final maps (`RECOVAR-vs-RELION`
FSC-AUC about `0.974-0.990` in those cases), but are consistently better
against the simulator ground truth. Treat the RELION-corrected convention as a
diagnostic map-parity mode, not the default quality path.

`RECOVAR_FINAL_ALL_DATA_GRID_CORRECT` now defaults to disabled for the
RECOVAR GUI-quality path after the June 2026 robustness matrix showed the
RELION-style final output correction consistently lowered GT FSC-AUC on the
small synthetic cases. Set it to `1` only for diagnostics that intentionally
compare the RELION-corrected final-map convention.

Direct binding comparison against `Projector::griddingCorrect` confirms
RECOVAR's radial trilinear correction uses the same origin/radius convention
as RELION for even and odd boxes, so the post-correction sub-0.001 residual is
not explained by the gridding-correction kernel itself.

Regression coverage:

- `tests/unit/test_refine_relion_mode.py::test_final_all_data_grid_correct_env_defaults_to_relion_parity`
- `tests/unit/test_dense_iteration_loop_merge_guards.py::test_final_all_data_grid_correction_defaults_to_relion_parity`
- `tests/unit/test_relion_functions.py::test_gridding_correct_matches_relion_binding`

### 2026-07-19 strict-parity correction

An exact same-boundary Case 25 capture superseded the grid-off parity default.
Replaying RELION's joined BPref accumulator and tau2 through RECOVAR with the
RELION padding factor reproduced RELION's captured final map at FSC-AUC
`0.999999999999910` with gridding correction on, versus `0.998191610326802`
with it off.  RECOVAR's own accumulator and tau2 with correction on reached
FSC-AUC `0.999998729433245` against the uninterrupted RELION final map.
Therefore strict RELION parity now defaults the final correction on;
`RECOVAR_FINAL_ALL_DATA_GRID_CORRECT=0` preserves the earlier grid-off quality
ablation explicitly.

## 2026-06-30 K-Class Final All-Data After Max-Iter Guard

RELION's final all-data iteration is a post-convergence step, not a generic
post-`max_iter` step. The K=4 case5 default run
`/scratch/gpfs/CRYOEM/gilleslab/_agent_scratch/em_kclass_case5_case7_defaults_20260629_073653`
did not converge by `max_iter=8` and therefore kept the last numbered maps:
RECOVAR-vs-GT FSC-AUC `0.193188` versus RELION `0.195526`, a meaningful but
small default-path residual of `-0.002338`.

Forcing `RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER=1` on the same K=4 case family
is qualitatively wrong after non-convergence: the forced-final diagnostic
completed but dropped to mean GT FSC-AUC `0.123809` versus RELION `0.195449`,
a loss of about `0.072`. This is far beyond numerical noise and should not be
used as a GUI/default quality path.

`_should_run_final_all_data_iteration` now ignores
`RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER=1` for K-class after max-iter
exhaustion while preserving K-class final all-data after actual convergence
and preserving the K=1 diagnostic override.

Regression coverage:

- `tests/unit/test_refine_relion_mode.py::test_final_all_data_after_max_iter_env_defaults_to_disabled`
- `tests/unit/test_refine_relion_mode.py::TestRelionModeSmokeTest::test_relion_final_iteration_supports_k_class`
- `tests/unit/test_refine_relion_mode.py::TestRelionModeSmokeTest::test_relion_final_iteration_k_class_adaptive_uses_sparse_pass2_route`
- `tests/unit/test_refine_relion_mode.py::TestRelionModeSmokeTest::test_relion_mode_does_not_finalize_after_max_iter_exhaustion`

## 2026-06-30 Exact RELION `ini_high` Low-Pass

The remaining target-particle score mismatch in case32 was not caused by CTF
preprocessing, image scaling, RELION grid order, or the `Projector::project`
port. RELION image operands matched RECOVAR after the known `N^2`/`N^4`
conversions, and projecting RELION's dumped `PPref` through the RECOVAR port
reproduced RELION scores to the expected small residual. The mismatch came
from how RECOVAR built `PPref`: `scripts/run_full_refinement.py` and the
iter-1 CC post-reconstruction path called the generic
`recovar.heterogeneity.locres.low_pass_filter_map`, while RELION's
`initialLowPassFilterReferences` uses a real-space `rfftn` plus a two-shell
cosine edge from `ml_optimiser.cpp`/`WIDTH_FMASK_EDGE`.

On case32 target original index 7, old RECOVAR `Projector::data` differed from
RELION `pass1_class0_ppref` at coefficient p95 `1.32e-4`, max `1.75e-3`
after scale fit. Replacing the startup and iter-1 low-pass with the exact
`initial_low_pass_filter_references` helper reduced that to p95 `5.1e-9`,
max `1.6e-7`. The full 1,069,056-candidate score-table comparison against the
RELION part-specific ACC dump improved from p95 `1.89e-3`, max `5.46e-3`, and
a top-pose mismatch (`RECOVAR [6978,16]`, RELION `[7972,15]`) to p95
`1.03e-4`, max `2.79e-4`, and matching top pose `[7972,15]`.

Regression coverage:

- `tests/unit/test_em_parity_lowpass_and_tau2_fudge.py::test_apply_initial_lowpass_helper_calls_exact_relion_lowpass`
- `tests/unit/test_em_parity_lowpass_and_tau2_fudge.py::test_mean_helpers_initial_lowpass_matches_exact_relion_helper`

## 2026-06-30 Local Pass2 Retained-Mass Statistics

The repeatable case11/case19 K=1 residual after the low-pass and final-output
fixes is larger than rerun jitter, so treat it as a bug signal rather than
numerical noise until proven otherwise. RELION source inspection ruled out two
tempting explanations:

- `convertAllSquaredDifferencesToWeights` intentionally computes
  `pdf_offset` on the coarse translation sample and shares it across
  oversampled translation children.
- The significant-support threshold sorts strictly positive posterior weights,
  stops on `frac_weight > adaptive_fraction * exp_sum_weight`, and
  `storeWeightedSums` reconstructs weights with `weight >= significant_weight`.

The actionable mismatch found in the RECOVAR local adaptive path was
statistics accounting after pass2 support truncation. RELION normalizes the
full pass2 posterior for evidence and best-pose reporting, but the M-step,
noise sums, and rotation posterior accumulated into the next iteration only
see the retained `storeWeightedSums` support. RECOVAR already used the
retained posterior for the actual backprojection/noise M-step, but local
postprocessing still defaulted rotation-posterior statistics to the full
posterior. `stats_use_reconstruction_probs` now lets the local K=1 adaptive
pass2 path report retained-mass statistics while keeping pose argmax and
evidence on the full posterior.

Validation in flight:

- case11 support-ablation job `10488054` under
  `/scratch/gpfs/CRYOEM/gilleslab/_agent_scratch/em_case11_local_support_ablation_20260630_200201`
  showed that broadening parent support flips/overshoots the delta, so the
  default bug is not simply "support too narrow".
- patched case11 replay job `10488575` is queued under
  `/scratch/gpfs/CRYOEM/gilleslab/_agent_scratch/em_case11_float64norm_replay_20260630_203735`
  and installs the current worktree at job start.

## 2026-06-30 Final All-Data Sampling STAR Precedence

The retained-mass statistics patch did not explain the deterministic case11
K=1 quality gap. A matched-particle RELION final all-data dump later showed
that the earlier sampling-precedence conclusion was backwards: RELION's final
`run_data.star` translations match the unnumbered `run_sampling.star`
`SamplingPerturbInstance=+0.461207`, while the last numbered
`run_it010_sampling.star` held `+0.182315`. Choosing the numbered state shifts
the entire fine translation grid by about `0.109 px` per axis and changes the
local parent support away from RELION's final output.

Do this oracle against the 36-point local fine grid, not the 9-point parent
translation grid. For the matched dump root
`/scratch/gpfs/CRYOEM/gilleslab/_agent_scratch/em_case11_relion_dump_stack418_iter11_20260630_205040`,
particle `418@particles.128.mrcs` has final `run_data.star` origin
`[-1.257140, 1.133486] A`. The fine child grid from `run_sampling.star`
contains `[-1.257139, 1.133486] A` at child index 5 (`~1.2e-6 A` away);
the nearest last-numbered fine child is still `0.323 A` away. A parent-grid
only check gives the wrong conclusion.

Final all-data replay now searches sampling metadata in this order:

- `run_it{final_relion_iteration}_sampling.star`
- `run_sampling.star`
- `run_it{last_numbered_iteration}_sampling.star`

The intermediate last-numbered replay job `10491368` completed successfully under
`/scratch/gpfs/CRYOEM/gilleslab/_agent_scratch/em_case11_final_sampling_fix_replay_20260630_223207`.
It records `final_all_data_sampling_star_source=last-numbered` and improved
GT FSC-AUC, but reduced RECOVAR-vs-RELION map FSC-AUC and no longer matches
RELION's final STAR metadata. Treat that result as a quality-oriented
diagnostic, not the parity target.

A strict replay with the corrected precedence was submitted as Slurm job
`10495659` under
`/scratch/gpfs/CRYOEM/gilleslab/_agent_scratch/em_case11_final_sampling_relion_parity_20260701_001600`.
A second strict replay against the matched RELION dump root's own `relion_ref`
was submitted as Slurm job `10495933` under
`/scratch/gpfs/CRYOEM/gilleslab/_agent_scratch/em_case11_matched_relion_replay_20260701_002301`.
Use this one for `storeWavg` score-surface comparison.
Pending broad robustness jobs were temporarily held so `10495933` can start
before lower-value queued diagnostics; release helper job `10496281`
(`afterany:10495933`) releases those holds.
Job `10495933` failed before the final E-step because it started from a stale
source state where the fused sparse K-class pass-2 helper did not accept
`group_ids`. The current import signature now includes `group_ids`, and
`tests/unit/test_sparse_pass2_bucketed_perf.py::test_fused_sparse_k_class_pass2_matches_existing_two_pass_path`
passes with that path. Retry job `10496685` was submitted under
`/scratch/gpfs/CRYOEM/gilleslab/_agent_scratch/em_case11_matched_relion_replay_retry_20260701_004209`;
release helper job `10496686` (`afterany:10496685`) releases the temporary
holds after the retry exits.
To get the retry through Della's user job-count limit, stale monitor jobs were
cancelled, current monitor job `10494240` was cancelled temporarily, and the
just-started broad case5 job `10493669` was cancelled after `requeuehold` was
rejected by Slurm. Resubmit/release broad coverage after the strict replay
result.

Regression coverage:

- `tests/unit/test_dense_iteration_loop_merge_guards.py::test_final_all_data_sampling_replay_prefers_final_sampling_star_before_last_numbered`

Cross-case/source-code check on 2026-07-01:

- A scan of 120 recent RELION `relion_ref/run_sampling.star` roots found that
  all 120 differed from their last numbered `run_itNNN_sampling.star` in
  `SamplingPerturbInstance`; the sampled roots usually had identical
  `OffsetRange`, `OffsetStep`, `HealpixOrder`, and perturbation factor. This
  makes final sampling replay a deterministic, dataset-wide parity issue rather
  than random numerical noise.
- RELION source check:
  `/scratch/gpfs/GILLES/mg6942/relion/src/healpix_sampling.cpp` applies
  `random_perturbation * offset_step / pixel_size` to every oversampled
  translation child in `HealpixSampling::getTranslationsInPixel`.
- RELION source check:
  `/scratch/gpfs/GILLES/mg6942/relion/src/ml_optimiser.cpp` computes
  `pdf_offset` at the coarse parent translation
  `old_offset + sampling.translations_[itrans]`, while the likelihood loop
  scores the oversampled children. RECOVAR's coarse-prior/fine-likelihood
  wiring matches this source path; the active suspect for the case11 final
  score mismatch remains the final sampling source/perturbation unless the
  matched `storeWavg` comparison says otherwise.

## 2026-06-30 Post-Fix Reconstruction/E-Step Split

The post-fix case11 replay reproduces its dumped final BPref reconstruction
exactly under `scripts/replay_final_bpref_dump.py --tau2-fsc-mode whole`
(`replay_minus_dumped_tau2_prior_shells_absmax=0`). Diagnostic variants did
not improve RELION map parity:

- `--tau2-fsc-mode half` lowered RECOVAR-vs-RELION final-map FSC-AUC from
  `0.969922` to `0.965232` and lowered GT FSC-AUC from `0.636909` to
  `0.634788`.
- `--grid-correct off` improved GT FSC-AUC to `0.656683` but slightly lowered
  RECOVAR-vs-RELION map FSC-AUC to `0.969152`.

So the remaining map/pose-probability divergence is not explained by final
tau2 conversion or gridding correction alone. The active diagnostic target is
the final all-data E-step/support: job `10493556` was submitted to regenerate a
matched RELION stack-index-418 candidate dump from the original case11
`run_it009_optimiser.star` path. Broader post-fix validation is also queued:
K=1 jobs `10493667`/`10493668`/`10493669`/`10493671`/`10493672`/`10493673`/
`10493674` cover 100k/400k, SNR, anisotropy, and radial/nonuniform pose
stress, while K-class jobs `10493694` through `10493705` cover K=2/4/8/16,
Ribo/IgG, 10k/20k/50k, class balance, Kent/nonuniform poses, and outliers.

## 2026-06-30 Post-Fix Support Investigation

The first RELION continuation dump job, `10493556`, is not an apples-to-apples
case11 final-state dump: continuing from `run_it009_optimiser.star` without
the original perturbation state regenerated `run_it010_sampling.star` with a
different perturbation (`-0.357630` instead of original `+0.182315`). Its
candidate mismatch against RECOVAR is therefore useful only as a support-scale
diagnostic, not as quality-parity evidence. In that non-matched dump, RELION
exposed 51,840 fine denominator candidates and 35 reconstruction candidates
for stack index 418, while the strict RECOVAR pruned-parent dump for original
index 417 exposed 160 finite pass2 candidates and 8 reconstruction samples.
The mismatch is support/state scale, not small floating-point jitter.

Two follow-up jobs are the current apples-to-apples checks:

- seeded RELION dump `10493964` under
  `/scratch/gpfs/CRYOEM/gilleslab/_agent_scratch/em_case11_relion_matched_seeded_dump_20260630_2342`,
  using `--random_seed 1711`;
- RECOVAR replay against the unseeded RELION continuation state `10494067`
  under
  `/scratch/gpfs/CRYOEM/gilleslab/_agent_scratch/em_case11_recovar_unseeded_replay_20260630_2346`,
  using `RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT=0` and the same RELION
  continuation directory.

The post-fix GUI/default K=1 matrix has completed five small cases so far, all
with RECOVAR above RELION versus ground truth:

| Case | Axis | RECOVAR GT FSC-AUC | RELION GT FSC-AUC | Delta |
|---|---|---:|---:|---:|
| 11 | baseline 3k, white noise 1 | 0.659862 | 0.633303 | +0.026559 |
| 12 | very high noise 10 | 0.191153 | 0.187878 | +0.003274 |
| 13 | anisotropic, noise 3 | 0.300240 | 0.292026 | +0.008214 |
| 15 | 20% outliers | 0.618386 | 0.592797 | +0.025589 |
| 21 | Kent angles, noise 3 | 0.383742 | 0.376168 | +0.007574 |

This does not prove perfect RELION parity: RECOVAR-vs-RELION map FSC is lower
on the harder axes, and the exact E-step/support contract remains under
investigation. A dedicated 24h monitor, job `10494240`, writes rolling reports
to
`/scratch/gpfs/CRYOEM/gilleslab/_agent_scratch/em_broad_postfix_monitor_20260630_234720/report`
for the current post-fix, broad K=1, K-class, and pruned-parent roots.

## 2026-07-01 Case32 Broad-Score Particle Association Trap

The apparent case32 broad first-iteration score mismatch for RECOVAR original
index `6756` was a RELION dump association error, not a projection/scoring
bug. RELION job `10502855` regenerated the dump under
`/scratch/gpfs/CRYOEM/gilleslab/_agent_scratch/em_case32_cc_components_20260701_032855`
with `RELION_ACC_DUMP_CC_COMPONENTS=1`. The generic `pass*_img0_*` operands
and component arrays corresponded to `img0_part6398`, while a separate
part-specific score table existed for `img0_part6756`.

After applying the RELION pixel-major to RECOVAR psi-major rotation mapping
(`relion_n_psi=48`), RECOVAR's full significance score table matches
`img0_part6398_pass1_class0_pass1` with correlation `0.999999825` and max
centered score error `1.86e-4`; its top key is identical. The same RECOVAR dump
compared to `img0_part6756_pass1_class0_pass1` has correlation `0.231`, which
is the false mismatch. Use the updated
`scripts/compare_relion_recovar_estep_dump.py` prefix rankings before drawing
conclusions from multi-particle RELION ACC dump directories.

## 2026-07-01 Deep-Bug Monitor and Iter-Cap Probe

The old broad monitor job `10494240` was cancelled at `2026-07-01T00:43:13`,
so the current rolling monitor is Slurm job `10503652` under
`/scratch/gpfs/CRYOEM/gilleslab/_agent_scratch/em_broad_deepbug_monitor_20260701_035613`.
It writes `report/latest_report.md` every 10 minutes through
`2026-07-02T03:57:30-04:00` and tracks both the original broad roots and the
active retry roots.

A manual refresh before restarting the monitor found one completed negative
delta in the current post-fix matrix: K-class case2
`ribo_k4_10k_g128_white_noise1_uniform` was `-0.00136568` FSC-AUC vs RELION.
That row reached `max_iter=5` without convergence, skipped RECOVAR final
all-data, and therefore compares RECOVAR last-numbered class maps to RELION's
iteration-5 Class3D maps. Prior case5 evidence shows forcing final all-data
after K-class max-iter exhaustion is qualitatively wrong, so the active
hypothesis is an iteration-cap/convergence comparison artifact, not a final
all-data switch bug.

Focused probe:

- root:
  `/scratch/gpfs/CRYOEM/gilleslab/_agent_scratch/em_kclass_case2_itercap_probe_20260701_035928`
- setup job `10504002` completed successfully
- K-class case job `10504003` is queued with `EM_KCLASS_MATRIX_MAX_ITER=10`
  and `EM_KCLASS_MATRIX_TIME_LIMIT=10:00:00`
- summary job `10504004` depends on the case job
- dedicated monitor job `10504230` writes
  `/scratch/gpfs/CRYOEM/gilleslab/_agent_scratch/em_kclass_case2_itercap_probe_monitor_20260701_040202/report/latest_report.md`
  through `2026-07-02T04:02:35-04:00`

Interpretation rule for this probe: if the `max_iter=10` run converges or
improves the delta materially, treat the original `-0.00136568` as an
iteration-cap artifact. If it remains negative with converged RECOVAR output
and matched final/all-data status, resume dump-level comparison for the first
divergent K-class iteration.

The refreshed broad monitor also surfaced a concrete robustness bug in high-res
K=1 case10 `high_res_anisotropic_100k_g384_radial_noise3_bf0`: job `10484517`
failed in the first sparse pass-2 M-step while enforcing RELION's x=0 Hermitian
plane on a 771^3 BPref grid. JAX attempted a transient full packed-volume
device update and OOMed on a 3.42 GiB allocation. The fix is to make host-side
x=0 plane enforcement a memory-safety default for large RELION half-volume
grids, independent of the diagnostic native-half repack switch. Explicit
`RECOVAR_RELION_X_HALF_HOST_X0=0` still opts out.

Regression coverage:

- `tests/unit/test_half_volume_mstep.py::test_relion_x_half_host_x0_threshold_is_memory_safety_default`
- `tests/unit/test_half_volume_mstep.py::test_enforce_half_volume_x0_uses_host_path_for_large_grids`

Focused case10 retry:

- root:
  `/scratch/gpfs/CRYOEM/gilleslab/_agent_scratch/em_k1_case10_host_x0_retry_20260701_040933`
- setup job `10504407` completed successfully
- RECOVAR-only high-res case job `10504408` is queued with
  `EM_K1_MATRIX_MAX_ITER=15`
- summary job `10504409` depends on the case job
- monitor job `10504450` writes
  `/scratch/gpfs/CRYOEM/gilleslab/_agent_scratch/em_k1_case10_host_x0_retry_monitor_20260701_041207/report/latest_report.md`
  through `2026-07-02T04:12:49-04:00`

## 2026-07-11 K=1 X-Half Full-BPref Microbatch Memory Guard

The exact-local high-memory auto-boost was unsafe when the M-step retained a
RELION x-half BPref accumulator.  Small current-size and full-resolution
BPref grids leave materially less transient memory than the score-only path,
and the boosted cap produced late local-search OOMs on the 3k stress cases.
The exact-local cap now keeps a conservative x-half-specific bound unless an
explicit diagnostic environment override is supplied.  Unit coverage pins
both full-BPref and current-size BPref cases.

End-to-end validation:

- source commit: `4fba8f48a00ca7820a763e7ba41dac4a5a8d8242`
- dirty diff SHA-256: `65ecc89d87ad6c79b0febccaa399edbc85a5cf68b94190cfc28fbe7026597221`
- Slurm jobs: setup `10977258`, case `10977259`, summary `10977260`
- root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case18_xhalf_current_bpref_cap_retry_20260711_071154`
- case: 3,000 particles, 128 box, white noise 1, per-particle
  `noise_scale_std=0.5`, `contrast_std=0.5`
- status: completed without OOM in 2,295 s total case wall; RECOVAR refinement
  itself reported 2,272.2 s for 16 iterations
- quality: RECOVAR-vs-GT FSC-AUC `0.752256`, RELION-vs-GT `0.733599`
  (`+0.018657`, better); RECOVAR-vs-RELION merged FSC-AUC `0.976233`
  and correlation `0.995063` (same-quality parity)
- final particle metrics: pose mean/p95 `0.376/0.718 deg`, translation
  mean/p95 `0.0166/0.1500 px`; free-trajectory Pmax mean absolute gap
  `0.0931`

This validates the memory guard without trading away map quality.  Keep the
x-half cap conservative by default; larger caps remain performance
experiments, not quality defaults.

## 2026-07-11 100k K=1 Free-Trajectory Pmax Gap Is State-History, Not E-Step Arithmetic

The optimized 100k/256 K=1 run (Slurm `10940895`) completed ten iterations in
11,398.2 s and retained strong map parity: RECOVAR-vs-RELION merged FSC-AUC
`0.994387`, correlation `0.999571`, and RECOVAR-vs-GT FSC-AUC `0.456776`
versus RELION `0.447203` (`+0.009573`, better).  Its final free-trajectory
particle Pmax comparison was much weaker: correlation `0.405970`, mean
absolute gap `0.178171`, although poses and translations remained close.

An iteration scan localized the gap to iteration 2, before local search:
Pmax correlation `0.348878` and mean absolute gap `0.187896`.  The original
interpretation that RECOVAR did not emulate RELION's iter-1 winner-take-all
path was superseded by the result-assembly audit below.

A fixed-state replay then selected the two worst free-trajectory particles
from each half-set (`37899,43649,58806,78500`) and replayed RELION it001 to
it002 with RELION maps, noise, norm/scale corrections, priors, poses,
translations, sampling perturbation, and current size.  The same particles
had free-trajectory absolute Pmax errors of `0.768-0.844`; under fixed-state
replay they had:

- mean absolute Pmax error `0.000309`
- maximum absolute Pmax error `0.000917`
- Pmax correlation `1.000000`
- exact RECOVAR-vs-RELION poses and translations

Replay provenance:

- Slurm job `10978841` on H100 `della-h19g2`, completed in 208 s
- root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fixedstate_it2_pmax_20260711_081114`
- command: `scripts/run_multi_iter_parity.py --iter 1 --max_iter 1
  --keep_stack_indices 78500,43649,58806,37899 --image_batch_size 4
  --rotation_block_size 512 --skip_final_iteration`
- RECOVAR pass-2 dumps:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fixedstate_it2_pmax_20260711_081114/dumps`

This falsifies the late fused/split score kernel and pass-2 arithmetic as the
cause of the large free-trajectory Pmax gap.  A separate fused-vs-split dump
for 178,176 identical candidates had maximum score difference `4.88e-4`,
maximum posterior difference `2.75e-5`, and identical best pose/translation.
Use fixed-state replay for arithmetic-level posterior claims and map/pose/GT
metrics for free-trajectory quality claims.

## 2026-07-11 K=1 Firstiter-CC Pmax Assembly Bug

Two independent source audits and the recorded 100k run showed that RECOVAR
already executed normalized-CC scoring, hard-winner reconstruction, and the
best-coarse winner-subset pass-2 path.  However, top-level K-class result
assembly combined coarse-pass log evidence with fine-pass best scores and
recomputed `Pmax = exp(fine_best - coarse_logZ)`.  Those values are from
different score surfaces, so the 100k strict run produced iter-1 Pmax
min/mean/max `2.79e-7 / 2.03e-6 / 1.46e-5` and logged `ave_Pmax=0.0000`.
RELION's corresponding model STAR reports exactly `1.000000` because its
firstiter-CC path binarizes the winning weight.

The fix adds an explicit firstiter winner-take-all contract to
`k_class._assemble_result` and sets Pmax to one for every valid fine-pass
winner.  It changes only reported state; the existing fine-pass `Ft_y` and
`Ft_CTF` accumulators already passed through unchanged.  Therefore this fixes
the earliest proven trajectory mismatch in Pmax scheduling/convergence but
does not by itself prove that iter-1 reconstruction state caused all of the
iter-2 boundary-replay gap.

Regression and validation:

- failing-before/passing-after test:
  `tests/unit/test_firstiter_cc_batch_budget.py::test_firstiter_winner_take_all_assembly_reports_unit_pmax_across_score_normalizations`
- full focused file: 11 passed
- `pixi run test-em-fast-guard`: passed
- no GPU or Slurm validation yet

The canonical 5k/128 fixture omits `--firstiter_cc` and is not a valid strict
iter-1 oracle.  Use the complete 3k/128 strict fixture at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_small_stress_relion_20260711_042025_22010/cases/11_small_baseline_3k_g128_white_noise1_bf80`
for the immediate A/B and generate a pinned strict 5k fixture later.

`scripts/run_multi_iter_parity.py` now exposes typed
`--firstiter-cc-mode {auto,on,off}` control. `auto` is the default and follows
the pinned optimiser STAR command; `on` forces strict iter-0 semantics for a
diagnostic, and `off` is the explicit ablation path. Unit coverage validates
oracle-on/oracle-off auto selection, both overrides, and rejects forcing the
mode after iter 0. The focused harness file passes 11/11 tests.

## 2026-07-11 Strict Firstiter-CC A/B and Remaining Iter-1 Map Gap

Slurm job `10986138` ran three sequential one-iteration controls on H100
`della-h21g2` from the same complete 3k/128 strict fixture. The run root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_firstiter_pmax_ab_20260711_122212`
and is marked `SAFE_TO_DELETE`. The tested source was commit `b044c0a3`.
Two earlier attempts, `10986075` and `10986106`, failed before refinement due
to missing cuSPARSE runtime paths and then a missing RELION binding; they are
setup failures and contain no scientific result.

The strict `--firstiter-cc-mode on` run now matches RELION's entire iter-1
Pmax vector exactly: mean, median, and maximum absolute gap are zero for all
3,000 particles, and both report `ave_Pmax=1`. The explicit off ablation
reports `ave_Pmax=0.893242`. Strict iter-1 pose/translation assignments are
also close: angular mean `0.0111` degrees with 99.9 percent within 5 degrees,
and translation mean `0.0003` pixels with p99 zero. The few discrete outliers
still require score-margin adjudication before they can be called numerical
ties.

The Pmax repair does not close the full iter-1 state. The strict merged map
correlation against RELION is `0.995764` (half correlations
`0.995723/0.995755`), recorded only as a weak diagnostic. RECOVAR and RELION reach
the same FSC<0.143 shell 20; RECOVAR-vs-GT map correlation is `0.712883`
versus RELION `0.711143`, so this is a strict-parity failure rather than an
observed GT-quality regression. By contrast, fixed-state RELION-it1 to
iter-2 replay gives map correlation `1.000000`, pose mean gap `0.0059`
degrees, translation mean gap `0.0002` pixels, and Pmax mean gap about
`2.97e-5`. This localizes the remaining material difference to the iter-1
history, before the later fixed-state arithmetic.

The strict run preserved its post-join iter-1 `Ft_y` and `Ft_ctf` arrays under
`strict_on/intermediates/it000_Ft_*`. The next discriminating experiment is
therefore a matched patched-RELION iter-1 raw/downsampled BPref dump, followed
by coordinate- and shell-level comparison with
`scripts/compare_iter1_bpref_accum.py`. Do not extend the boundary-replay trajectory
until this accumulator boundary is classified.

## 2026-07-11 Iter-1 BPref and Tie-Aware Winner Adjudication

The matched patched-RELION M-step dump ran as Slurm job `10986571` on the
same H100 node (`della-h21g2`) as the strict RECOVAR A/B. Job `10986554`
failed before RELION due an MPI-slot allocation mistake and is setup-only.
The successful run root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_firstiter_bpref_dump_20260711_123707`
and is marked `SAFE_TO_DELETE`. The patched RELION binary SHA-256 is
`206d6fbb7e3840549ea49adb4154c313dcb7b36a709c902a4969ee6d03cc6a33`.
Its iter-1 maps match the original d476e6 oracle with correlation `1.0` and
scaled relative error about `3e-8`; all 3,000 Pmax values match exactly.
The local RELION changes therefore do not contaminate this oracle boundary.

At all 47,209 logical BPref coordinates per half, the expected
RELION `(k,i,j)` to RECOVAR `(i,k,j)` mapping ranks first. Median relative
complex-average errors are `6.74e-5/9.39e-5` for halves 1/2, and median
weight errors are `2.30e-6/3.23e-6`. Complex-average cosine similarities are
`0.999939/0.999808`; weight cosine similarities are `0.999985/0.999928`.
Shell weight ratios are effectively one except the outer `r=28` boundary
(`0.9982/0.9983`). The larger half-2 error tail coincides with four of the
five materially different WTA assignments.

Tie-aware score adjudication used jobs `10986838` (matched RELION broad/fine
tables), `10987192` (five RECOVAR full coarse tables), and `10987411`
(coherent RELION fine table for original index 2693), all on H100
`della-h21g2`. Jobs `10986788` and `10987024` were cancelled/failed diagnostic
setup attempts: the former used original indices instead of RELION's
micrograph-sorted internal part IDs, and the latter selected a dump hook that
is bypassed by the joint strict-firstiter coarse scorer. Their outputs are not
scientific evidence. The complete artifact root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_firstiter_tie_scores_20260711_124711`
and is marked `SAFE_TO_DELETE`.

For original indices `431,1087,1280,1794,2693`, all 1,069,056 coarse
candidates match one-to-one. RECOVAR-vs-RELION centered score p95 gaps are
`6.98e-5` to `8.60e-5`, and maximum gaps are `1.71e-4` to `2.08e-4`.
Coarse winners are identical for 431 and 2693. The other three winner flips
are genuine near ties: RELION's preference is `2.41e-5` to `4.96e-5`, while
RECOVAR's opposite preference is `2.26e-6` to `1.38e-5`. Particle 431's
32-candidate fine surfaces correlate `0.99999961`; its opposite winner
preferences are `7.54e-6` and `1.45e-5`. Particle 2693's fine surfaces
correlate `0.99999972`, with centered p95/max gaps `5.61e-5/6.24e-5` and
opposite preferences `4.83e-6/2.87e-5`.

Thus every iter-1 discrete difference is explained by underlying score arrays
inside the established GPU numerical band. This satisfies the user's
tie-aware decision contract. The iter-1 map residual is a hard-WTA
amplification of five qualified ties plus arithmetic-level scatter tails, not
an unexplained support, coordinate, score, or accumulator bug. Continue with
the repaired strict boundary-replay trajectory and require convergence/finalization to
match exactly.

## 2026-07-11 Strict Full Trajectory and Final-Oracle Boundary

Slurm job `10987518` completed the strict 3k/128 trajectory on H100
`della-h21g2` from commit `966fe8fe`. The artifact root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_strict_full_trajectory_20260711_131542`
and is marked `SAFE_TO_DELETE`. RECOVAR matches RELION's ten numbered
iterations, convergence at iteration 10, current sizes
`[56,56,66,68,80,80,80,80,80,80]`, and the post-convergence all-data branch.
Numbered RECOVAR-vs-RELION map correlations are `0.995764, 0.999775,
0.999872, 0.999857, 0.999831, 0.999885, 0.999889, 0.999900, 0.999908,
0.999912`; the iter-1 exception is the already-qualified WTA tie boundary.

The harness initially mislabeled final parity by comparing RECOVAR's
post-convergence all-data result against RELION's numbered iter-10 half maps.
The semantically correct oracle is the unnumbered RELION
`run_class001.mrc`. Against it, the RECOVAR final merged map has correlation
`0.987438` and FSC-AUC `0.980260`. RECOVAR remains better against GT on this
fixture (`0.654772` versus RELION `0.633303` FSC-AUC), so this is a strict
finalization-parity failure rather than a quality failure. Post-hoc RELION
radial grid correction changes parity only to correlation `0.989735` and
FSC-AUC `0.981087`, falsifying grid correction as the main residual.

The first final-only state bug is explicit in `run.log`: final all-data
requested replay state index 10 but fell back to index 9. The parity harness
allocated only `max_iter` overrides and therefore never loaded RELION's
last-numbered `run_it010` particle/model state for final scoring. The focused
fix allocates the additional final seed state, selects the true unnumbered
final map when `final_all_data_ran`, records canonical final correlation and
FSC-AUC in the NPZ/JSON ledger, and preserves final poses, translations, Pmax,
sampling, tau2, FSC, and grid metadata for subsequent boundary comparisons.

The focused validation first ran as job `10988194`. Its scientific pass and
1.2 GB final BPref dump completed, but a reporting-only `KeyError` occurred
because the new unnumbered final-map branch still tried to add nonexistent
RELION half maps to the GT table. A focused regression now covers that branch.
Clean retry job `10988298` completed on H100 `della-h21g2` in `3m50s` from
commit `44c6fc1b`; its root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_final_state10_replay_retry_20260711_134820`
and is marked `SAFE_TO_DELETE`.

The retry log shows final replay `previous_state_index=1` without fallback;
for this `--iter 9 --max_iter 1` replay, index 1 is RELION `run_it010`. The
numbered iter-10 fixed-state map correlation is `0.999995`. Correct final-state
seeding improves the true unnumbered final FSC-AUC from `0.980260` to
`0.991498`; correlation improves from `0.987438` to `0.988789`. Grid-on
postprocessing reaches only correlation `0.991226` and FSC-AUC `0.992144`, so
it remains a secondary output convention rather than the main residual.

Against RELION `run_data.star`, all 3,000 RECOVAR final particles have median
angular/translation errors of about `6e-6` degrees and `1.4e-7` pixels, but
the tails are material: angular p95/p99 are `0.631/0.937` degrees and
translation p95/p99 are `0.196/0.391` pixels. Pmax correlation is `0.7434`,
with mean/p95/max absolute gaps `0.0423/0.1468/0.6415`. Both programs move by
similar amounts from their iter-10 state to final, so the final policy is
broadly correct; the remaining first suspect is fine posterior/support and
its joined BPref accumulation. The next experiment is a matched patched-RELION
final BPref dump, not another boundary-replay trajectory.

## 2026-07-11 Final Local Pass-2 Parent-Support Bug

Matched final diagnostics used RELION jobs `10988528`, `10988659`, `10988810`,
`10988828`, and `10988885`. Job `10988659` established unpatched H100 rerun
jitter (`0.99999943` final-map correlation, Pmax mean gap `3.07e-4`). A full
patched trajectory drifted beyond that, so it was rejected as the strict
accumulator oracle. Final-only continuation `10988885`, forced to the oracle
`run_sampling.star` perturbation `+0.461207`, matches the original final map at
`0.9999847` and supplies the accepted BPref dump. Jobs `10988810` and
`10988828` were setup/non-matched sampling controls, not strict evidence.

The accepted half-boundary comparison shows RECOVAR already differs before
joining: complex scale-fitted L2 errors are `17.9%/18.4%`, and weight errors
are `5.58%/5.72%`. RELION's joined BPref is exactly the sum of its two halves,
exonerating the join. Matched score jobs `10989018`, `10989019`, and retry
`10989166` targeted original index 2219, whose winner pose is effectively
identical but whose Pmax differs materially. RELION uses 24 rotations x 36
translations, 169 positive candidates, 5 retained samples, and Pmax `0.8333`.
Default RECOVAR uses 1,392 rotations, 4,926 positive candidates, 30 retained,
and Pmax `0.2519`. Materialized and fused RECOVAR posterior dumps agree.

Pruned-parent validation job `10989301` restores the exact 24-rotation support,
156 positive candidates, 6 retained samples, and Pmax `0.8445`. True-final
FSC-AUC improves from `0.991498` to `0.994526` grid-off. The legacy grid-on
report was `0.995366`, but it integrated two incomplete Nyquist-edge shells.
Under the canonical 63-shell contract used by `regularization.get_fsc_gpu`,
grid-on RECOVAR-vs-RELION FSC-AUC is `0.995784` and passes the `0.995` gate.
Grid-on RECOVAR-vs-GT FSC-AUC is `0.649948` versus RELION `0.650835`, a
`-0.000887` delta within the `-0.002` gate. Diagnostic correlation is not an
acceptance criterion. The correctness change makes pruned-parent support the
default and retains full-parent as an explicit diagnostic override.

## 2026-07-11 Canonical FSC Contract and Final Cross-Replay

Commit `f80ac8ad` makes the K=1 replay/completion and K=4 evaluation reporters
use the same canonical shell range as `regularization.get_fsc_gpu`: for a
128-cube, shells 0 through 62, with DC excluded from normalized FSC-AUC. The
previous NumPy reporters included shells 63 and 64 even though they are
incomplete Nyquist-edge spheres. Focused tests pin the 63-shell result. Commit
`b95099b3` also promotes GT FSC-AUC into the trajectory NPZ/JSON ledger so GT
correlation cannot become the headline merely because FSC-AUC was omitted.

Canonical saved-map audit of job `10989301` gives grid-off
RECOVAR-vs-RELION FSC-AUC `0.994527` and grid-on `0.995784`; all grid-on
cross-FSC shells exceed `0.989`. Against GT, grid-off RECOVAR is substantially
better than RELION (`0.669908` versus `0.650835`), while strict grid-on differs
by only `-0.000887`. The grid-on FSC residual is concentrated at high
frequency: shells 41 through 62 contribute about 69 percent of its AUC
deficit. Radial amplitude rolloff is recorded as tau2/filter telemetry only;
it cannot pass or fail map quality because FSC is invariant to a positive
per-shell scale.

Commit `e766a2ef` adds a four-way final reconstruction replay using the saved
RECOVAR and matched RELION BPref/tau2 dumps. The artifact root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_final_cross_replay_20260711_150100`
and is marked `SAFE_TO_DELETE`. Against the matched continuation oracle, FSC-AUC
is `0.997036` for RECOVAR accumulator plus RECOVAR tau2, `0.997044` for
RECOVAR accumulator plus RELION tau2, `0.999756` for RELION accumulator plus
RECOVAR tau2, and `0.999750` for RELION accumulator plus RELION tau2. Against
the original strict oracle the corresponding values are `0.997003`,
`0.997010`, `0.999684`, and `0.999678`. Tau2 substitution is negligible and
the RECOVAR reconstruction/filter implementation is near exact when fed the
RELION accumulator. The remaining measurable high-shell residual is therefore
in BPref accumulation. This fixed-state final replay passes the FSC map gates,
but it does not qualify the boundary-replay trajectory documented below.

## 2026-07-11 Full Free-Trajectory Final Sensitivity

Clean A100 job `10990444` ran from commit `f80ac8ad` at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_strict_full_pruned_20260711_143632`
with a `SAFE_TO_DELETE` marker. The original H100 request `10989654` was
cancelled before execution because the RELION oracle was produced on A100.
The accepted job matches the ten numbered current sizes
`[56,56,66,68,80,80,80,80,80,80]`, convergence at iteration 10, and the
final all-data branch. Canonical numbered iter-10 RECOVAR-vs-RELION FSC-AUC is
`0.999324`; RECOVAR and RELION iter-10 GT FSC-AUC values differ by only
`-0.000037`. The unnumbered free-trajectory final nevertheless falls to
RECOVAR-vs-RELION FSC-AUC `0.988116`. RECOVAR final GT FSC-AUC remains better
(`0.669009` versus `0.650835`), but strict final parity fails.

The corresponding fixed RELION-seeded final job `10989301` has FSC-AUC
`0.994527`; its final map and the free result have mutual FSC-AUC `0.992955`.
Their final Pmax vectors differ by mean/p95/max `0.0470/0.1413/0.4505`, while
final tau2 and FSC shell arrays differ much less. Diagnostic-only commit
`7367ee61` adds zero-numbered-iteration final replay from supplied RECOVAR-frame
half maps. A100 job `10992173`, rooted at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_final_reference_sensitivity_20260711_152100`,
enters finalization from the saved free iter-10 half maps and reproduces
FSC-AUC `0.988115`. Therefore the saved half references fully explain the
free/fixed difference; no hidden optimizer state after iter 10 is required.

Merged-reference diagnostic A100 job `10992266`, rooted at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_final_merged_reference_20260711_152900`,
scores both halves against the merged free iter-10 reference and worsens final
FSC-AUC to `0.981072`. Early reference joining is not RELION's missing final
semantic. Job `10992371` now starts from exact RELION iter-1 half maps and runs
iterations 2 through 10 plus final. It decides whether the five already
qualified iter-1 WTA near-ties are sufficient to seed the amplified final
residual or whether a later numbered boundary also contributes.

## 2026-07-11 Texture Projection, Scatter, and Final Noise Boundary

A100 job `10992371` completed from exact RELION iter-1 half maps. It matched
the numbered schedule and convergence but reached final RECOVAR-vs-RELION
FSC-AUC `0.994488`, so qualified iter-1 ties explain most of the free-run
loss but are not the sole final residual.

RELION's first-iteration CUDA projector uses single-precision texture
interpolation. Direct texture probe job `10993092` flips all three previously
different coarse winners to RELION's choices; production-helper validation
job `10993348` matches the direct texture projections exactly. A100 iter-1
job `10993443` then matches all 3,000 orientations exactly. The strict
projector path now defaults to this CUDA texture arithmetic with an explicit
diagnostic switch. Fixed-final A100 job `10993646` nevertheless remains at
FSC-AUC `0.994526`, classifying texture projection as a real early-score fix
but not the final-map limiter.

One-particle scatter job `10994016`, rooted at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_single_particle_bpref_scatter_20260711_164500`,
feeds identical pre-reduced rows to RELION's C++ backprojector and RECOVAR's
CUDA x-half scatter. In float64, data and weight relative L2 errors are
`1.90e-14` and `1.83e-14`; float32 is materially worse (`7.17e-3` and
`1.30e-2`). Thus the production double scatter is correct and forcing float
accumulation is rejected. Array correlations in that diagnostic are not
quality evidence.

Verbose matched RELION final job `10994150` provides exact fine Euler,
projection, translated-image, CTF/noise, raw-cost, and posterior operands for
original particle 2219. The 24 fine rotation matrices map one-to-one with
maximum Frobenius error `7.45e-9`; translations are exact. The old pruned
RECOVAR dump had 156 positive/6 retained samples and Pmax `0.844463` versus
RELION's 169 positive/5 retained and `0.833306`. Texture replay jobs `10994391`
(setup failure), `10994428`, and `10994464` show exact RELION projections alone
do not close the old posterior.

The shared multiplicative mismatch in both shifted-image and CTF-squared
operands identifies a final joined-noise semantic: particle 2219 is in half 2,
but RELION scores both joined final particle halves with the half-1
`sigma2_noise`. Shell 4 is the clearest example: `1268.288 / 1312.464 =
0.96634`, matching the measured operand ratio `0.96635`. An exact-operand
replay with half-1 noise gives Pmax `0.833344` versus RELION `0.833306` and
candidate-posterior L1 error `7.7e-5`. Numbered split-half iterations still
retain separate noise spectra; only K=1 post-convergence all-data scoring
uses the joined first-half spectrum.

Fixed-final A100 job `10994996`, rooted at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_final_firsthalf_noise_20260711_165000`,
completed with canonical grid-off RECOVAR-vs-RELION FSC-AUC `0.994497`, still
below the `0.995` gate. RECOVAR-vs-GT FSC-AUC is `0.669846` versus RELION
`0.650835`. The new particle-2219 CTF/noise and shifted-image operands match
RELION at relative L2 `8.5e-7` and `2.6e-5`, but production Pmax is `0.816094`.
Injecting the exact RELION reference projection raises it to `0.833168`,
localizing the remaining amplification to the tiny numbered iter-10 map/
projector input difference. The next experiment is a matched patched-RELION
numbered iter-10 BPref dump from `run_it009`, compared against job `10994996`'s
saved numbered accumulators. Do not rerun the boundary-replay trajectory until that
earlier accumulator boundary is classified.

## 2026-07-11 Numbered Iter-10 and Exact-Reference Final Boundary

Matched patched-RELION continuation job `10996603` ran numbered iteration 10
only from `run_it009_optimiser.star` on A100 `della-l08g6`, using the exact
`+0.182315` sampling perturbation. Its root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_relion_iter10_bpref_20260711_170000`
and is marked `SAFE_TO_DELETE`. Patched-versus-original oracle FSC-AUC is
`0.9999987/0.9999966` for halves 1/2, with minimum non-DC shell FSC
`0.9999908/0.9999810`, so the raw BPref dump passes its oracle gate. Earlier
attempt `10996494` was cancelled after it started iteration 11 because its
continuation omitted `--auto_iter_max 10`; its products are preserved for the
audit trail but are not acceptance evidence.

At the expected coordinate mapping, numbered RECOVAR-versus-RELION median
complex-average errors are `0.001478/0.007461` and median weight errors are
`0.000230/0.003955` for halves 1/2. The low-resolution half join is exact,
per-half noise agrees at roughly `3e-6` median relative error, and winner
pose/Pmax errors are comparable. The first half-2-only difference is four
fewer significant samples (`15738` versus RELION `15742`; half 1 is exactly
`15501`). Thus this is sub-winner support/posterior arithmetic, not a half
swap, joined-noise error, coordinate mapping, or winner-pose bug.

Exact-reference A100 final-only job `10997070`, rooted at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_final_exact_iter10_firsthalf_noise_20260711_171000`
and marked `SAFE_TO_DELETE`, starts directly from the original RELION
iteration-10 half maps while retaining pruned-parent support, texture
projection, and joined half-1 final noise. It reaches final
RECOVAR-versus-RELION FSC-AUC `0.994501`, effectively unchanged from
`10994996` (`0.994497`), and therefore falsifies the hypothesis that the tiny
numbered-map difference causes the remaining gate miss. RECOVAR-versus-GT
FSC-AUC remains better (`0.669848` versus RELION `0.650835`).

Exact references improve particle 2219 from Pmax `0.816094` to `0.833537`
versus RELION `0.833306`, proving that particle is no longer representative.
Across all particles, however, final Pmax mean/p95/max absolute errors remain
`0.0282/0.0898/0.4592`. The worst current case is original index 428 / RELION
stack 429 (`0.173354` versus `0.632525`). The joined final accumulator still
differs from the accepted matched RELION pre-reconstruct accumulator by
`11.64%` complex-data and `3.20%` weight relative L2. The next experiment is
an exact matched RECOVAR/RELION final candidate and StoreWavg operand dump for
stack 429, followed by identical-row replay if its pre-scatter operands agree.
Do not launch the boundary-replay trajectory or robustness matrix until this final
support/BPref boundary is classified and the fixed-final FSC gate passes.

## 2026-07-11 Stack-429 Redundant-Axis Scoring Boundary

The matched final dumps under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_final_particle429_20260711_172000`
identify a full-size half-spectrum scoring mismatch. On interior `kx>0`
pixels, the RELION fine projections, shifted image, and CTF/noise operand match
the corresponding RECOVAR operands at arithmetic level after the documented
FFT/layout scaling. On the `kx=0` axis, RELION zeros `ky<0` rows because they
are redundant conjugates; RECOVAR retained and scored those rows. The Nyquist
boundary row is retained by RELION and is not part of the mask.

An offline replay using
`recovar_parent3/scores/local_score_it002_image_428_exact10_final429.npz`
changes the RECOVAR parent Pmax from `0.657449` to `0.812126`, versus RELION
`0.811969`. The 0.999 adaptive support then contains the same 10 parent pairs
as RELION, including local parent 83 / RELION rotation 140 at coarse
translation 4 (`0.000488` RECOVAR versus `0.000486` RELION). Without the mask,
that pair ranked below RECOVAR's cutoff at only `0.000090`; its absence removed
the fine candidates that later dominate the posterior.

Replaying the saved fine operands with the axis mask and those restored 32
children gives Pmax `0.628355` versus RELION `0.628361`, with the same winner.
On the old shared support alone, the mask reduces posterior L1 error from
`0.3785` to `0.0288`; the residual is dominated by the parent pair that was
absent from the original fine dump. This demonstrates that the apparent
translation-support mismatch is a downstream consequence of the same parent
score bug, not a second layout-routing defect.

The candidate fix is centralized in
`recovar/em/dense_single_volume/helpers/half_spectrum.py`: RELION scoring
weights are one except for packed rows `1:H/2` at column zero, which are zero.
Non-RELION Hermitian weights are unchanged. The next validation is an
exact-RELION-iter10 final-only A100 replay; quality remains gated only by
shellwise FSC, canonical FSC-AUC, and FSC-derived score/resolution.

Qualification job `11001328` completed on A100 `della-l07g2` in `136` seconds
with exit code zero. Its root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_final_exact_iter10_x0mask_20260711_184500`
and contains `SAFE_TO_DELETE`. Provenance is HEAD
`4e0d1a9b94460b80f1dbd63e3fcd48e3fd6a400d`, dirty SHA-256
`84e56685113c0b5e5865a08ce76e6accf734cd2465b85666215bd7632e3574ba`,
with only `scripts/probe_firstiter_texture_ties.py` untracked.

The fixed-final canonical RECOVAR-vs-RELION FSC-AUC is `0.997302`, up from
`0.994501`. The minimum non-DC shell FSC is `0.995021`, the fifth percentile
is `0.995382`, and the minimum over the last ten shells is `0.996824`.
RECOVAR-vs-GT FSC-AUC is `0.670396` versus RELION `0.650835` (delta
`+0.019561`); the FSC=0.5 crossing is shell 41 versus 40. Only GT shells 1--3
are lower for RECOVAR, with deltas `-0.000016`, `-0.000266`, and `-0.000172`;
no shell is lower by `0.002`. The fixed-final small-cell FSC contract passes.
The next qualification is a free ten-iteration trajectory, not the robustness
matrix.

## 2026-07-11 Axis-Mask Free-Trajectory Requalification

Clean A100 job `11002266` completed in `1201` seconds on `della-l07g2` from
commit `0b2d8cd1feaec40e539e02fd9aa32f1d8357d287`. Its marked root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_strict_free_x0mask_20260711_175100`.
It reproduces the numbered current-size schedule
`[56,56,66,68,80,80,80,80,80,80]`, convergence at iteration 10, and the
post-convergence Nyquist all-data path using state 10 and half-1 joined noise.

The final RECOVAR-vs-RELION FSC-AUC is `0.990397`, improved from the earlier
free result `0.988116` but still below the `0.995` gate. RECOVAR-vs-GT remains
better at `0.669518` versus RELION `0.650835`. The numbered merged maps are
already close: FSC-AUC is `0.997007` at iteration 2, `0.998487` at iteration
4, `0.999179` at iteration 8, and `0.999339` at iteration 10. The iteration-10
minimum non-DC shell FSC is `0.997415`. Thus the strict final sensitivity is
again much larger than the numbered-map residual.

The next single hypothesis is that the remaining free residual is seeded by
the iteration-1 reconstruction/BPref boundary. Re-run the exact-RELION-iter1
seed replay from the current checkpoint through iterations 2--10 and final.
Do not launch the robustness matrix unless that boundary is closed and the
boundary-replay trajectory passes.

## 2026-07-11 Iteration-1 Score-Geometry Scope

Exact-RELION-iter1 seed A100 job `11007539`, rooted at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_exact_iter1_seed_x0mask_20260711_182500`,
completed the exact iterations 2--10 schedule and convergence/final path. Its
final canonical RECOVAR-vs-RELION FSC-AUC is `0.997271`; RECOVAR-vs-GT is
`0.670338` versus RELION `0.650835`, with FSC=0.5 crossing shell 41 versus 40.
The remaining free-run failure is therefore seeded at iteration 1, not in a
later local-search boundary.

The post-axis-mask free run revealed 198/3000 angular and 219/3000 translation
mismatches at iteration 1 (angular mean `0.607718` degrees, p95 `3.750014`;
translation mean `0.048068` pixels, p95 `0.499999`). The pre-mask texture run
`10993443` had matched all orientations and retained only one qualified
0.5-pixel translation tie. Its iter-1 BPref accumulator relative-L2 errors
against RELION were `0.004621/0.001104` (complex/weight) for half 1 and
`0.004162/0.001139` for half 2; the globally masked run regressed to
`0.093928/0.054039` and `0.086122/0.044691`.

The cause is score geometry. The current-size-56 Gaussian radial window
contains none of the redundant-axis rows, while the normalized-CC rectangular
crop contains 27. RELION's first-iteration CUDA CC kernel scores all pixels in
that rectangle; only Gaussian likelihood scoring drops centered
`kx=0, ky<0`. The helper now exposes that distinction, and dense, K-class
significance, and both sparse pass-2 CC callers retain the full rectangle.
Gaussian dense/local/sparse callers retain the previously qualified mask.

One-iteration A100 qualification job `11013677` completed in `112` seconds on
`della-l08g6` under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_firstiter_cc_axis_scope_20260711_184500`.
Its coarse and fine hard-assignment arrays are byte-identical to job
`10993443`; all 3,000 orientations match RELION within Euler serialization
precision, and the same single 0.5-pixel translation tie remains. Job
`11013457` failed before RECOVAR with `CUDA_ERROR_NO_DEVICE` on
`della-l07g3` and is infrastructure-only. The next qualification is a clean
free ten-iteration trajectory; robustness remains gated on its canonical
FSC-AUC and shellwise FSC result.

## 2026-07-11 Iteration-1 Tau2 and Accelerated-BPref Precision

Clean score-mode-scoped free-trajectory job `11014763`, rooted at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_strict_free_cc_axis_scope_20260711_185000`
and marked `SAFE_TO_DELETE`, completed on A100 `della-l08g6` from commit
`47665b0fd1f94496da63d7202760b36a6ffcaa4f`. It exactly reproduces the
current-size schedule `[56,56,66,68,80,80,80,80,80,80]`, convergence at
iteration 10, and final all-data state/noise path. Final canonical
RECOVAR-vs-RELION FSC-AUC is `0.990351`, below the `0.995` gate, while
RECOVAR-vs-GT remains better (`0.669412` versus `0.650835`). Restoring exact
iteration-1 poses therefore did not restore the complete iteration-1 state.

Raw accumulator comparison localizes almost all weight-difference energy to
the padded outer boundary: more than 99.9 percent lies at padded radii 52--57.
Inside radius 52, weight relative-L2 error is approximately `9.4e-6` in each
half; after downsampling it is only `3--5e-6` through shell 27 and rises to
`0.00594/0.00621` at the shell-27/28 boundary. Half 1's remaining interior
complex residual contains the established 0.5-pixel translation tie; half 2
is otherwise clean.

The earlier one-particle conclusion that float accumulation was incorrect
compared RECOVAR CUDA against RELION's C++ CPU/double backprojector. The pinned
production GPU build has `DoublePrec_ACC=OFF`, so its accelerated `XFLOAT` is
single precision. The strict x-half path now allocates complex64/float32 BPref
data/weight arrays by default, retains an explicit double diagnostic switch,
casts the CPU-constructed orthonormal inverse to accelerator precision, and
evaluates the rotated radius cutoff in that same precision. One-iteration job
`11021943`, rooted at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_firstiter_acc_float_rot_tau_20260711_193500`
and marked `SAFE_TO_DELETE`, reduces downsampled-through-shell-28 BPref
data/weight relative-L2 errors to `0.004184/0.000949` for half 1 and
`0.003502/0.000902` for half 2. This is a modest boundary improvement, not by
itself a map-quality conclusion.

The larger proven state error was tau2. RECOVAR retained shell 18 `292.929`,
shell 19 `196.997`, and nonzero shells 20--28 because it saved tau2 calculated
before the first-iteration high-resolution filter. RELION source applies the
`--ini_high` raised-cosine mask to tau2 and data-vs-prior after
first-iteration reconstruction, squaring the mask for tau2. The source-matched
candidate produces shell 18 `106.849670`, shell 19 `0.0235179`, and zero from
shell 20 onward, versus RELION `106.808`, approximately `0.0235`, and zero.
Two setup jobs preserve the audit trail: `11021427` exposed a missing import,
and `11021652` exposed the need to clamp cube-corner radial indices; neither
produced final science.

Using the canonical Fourier-shell computation directly on the saved half maps,
job `11021943` improves merged iteration-1 RECOVAR-vs-RELION FSC-AUC over
supported shells 1--18 from `0.996052` to `0.998430`. Shell 18 improves from
`0.908735` to `0.948464`; half-map FSC-AUC values are `0.998428` and
`0.998427`. These FSC results, rather than map correlation, support the
candidate. The next discriminating experiment is a clean ten-iteration free
trajectory with the combined first-iteration state fix. Robustness remains
blocked until the full strict FSC, GT, shellwise, convergence, and finalization
gates pass.

Clean boundary-replay A100 job `11023037` subsequently completed in `579`
seconds on `della-l08g6` from commit
`a614969a29fd90535aba4b73fed093839ba29390`. Its marked root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_strict_free_tau_acc_20260711_200500`.
It matches the ten-step current-size schedule, convergence at iteration 10,
and final all-data path. Final canonical RECOVAR-vs-RELION FSC-AUC improves
from `0.990351` to `0.994646` but remains `0.000354` below the fixed gate.
RECOVAR-vs-GT is `0.670285` versus RELION `0.650835`. Numbered merged-map
RECOVAR-vs-RELION FSC-AUC is `0.997721` at iteration 2, `0.999213` at
iteration 4, `0.999686` at iteration 8, and `0.999746` at iteration 10.

RELION's accelerated CUDA `backproject2Dto3D` source shows only one sphere
test after rotating the source coordinate. RECOVAR also tested the unrotated
radius. Post-rotation-only A100 job `11025153`, rooted at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_firstiter_postrot_only_20260711_202000`,
falsifies that difference as the limiter: supported-shell iteration-1 FSC-AUC
is unchanged at `0.9984304911`, and shell 18 is unchanged at `0.948464267`.
The source-cutoff candidate is reverted.

Offline comparison of the new float32 accumulators against the existing
patched-RELION downsampled BPref dumps shows effectively exact shell sums.
At shell 18 both complex-average power and weight ratios print as `1.0000` for
both halves. Across 47,209 coordinates, weight median relative error is about
`1.76e-6`; complex-average median errors are `1.30e-5` and `1.22e-5` for the
two halves. The first meaningful residual is therefore after BPref
downsampling, not in its shell-aggregate input.

RELION ordering is explicit in source: `maximizationOtherParameters` calls
`initialLowPassFilterReferences`, then the outer iteration loop calls
`solventFlatten`. RECOVAR applied solvent flattening before the iter-1
Fourier low-pass. These operations do not commute because a final real-space
mask creates a deterministic Fourier tail. The active candidate swaps only
that first-iteration postprocessing order and updates the existing event-order
regression. One-iteration shellwise FSC must improve before another full
trajectory.

One-iteration A100 job `11025949`, rooted at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_firstiter_lowpass_before_mask_20260711_203000`
and marked `SAFE_TO_DELETE`, confirms the source-matched order. It completed
in `84` seconds on `della-l07g2` with dirty diff SHA-256
`208149fee341b071c06209412624c60a8fae2fecc4cae62f07fa13fb5841579f`.
Canonical full-shell RECOVAR-vs-RELION FSC-AUC rises from `0.298879` in the
wrong-order one-iteration run to `0.999538`. More directly, supported-shell
1--18 FSC-AUC rises from `0.998430` to `0.999930`; shell 18 rises from
`0.948464` to `0.998800`. The minimum non-DC shell FSC is `0.996857` at shell
19 and the fifth percentile is `0.998724`. This shellwise evidence closes the
iteration-1 reconstruction/postprocessing boundary. The next qualification is
a clean ten-iteration boundary replay; robustness remains gated on its final
FSC-AUC, shellwise, GT, convergence, and finalization results.

Clean ten-step boundary-replay A100 job `11026304`, rooted at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_strict_free_filter_order_20260711_204500`
and marked `SAFE_TO_DELETE`, completed in `579` seconds on `della-l08g7` from
commit `52d8b599254a713c115f052314f283abd6e0fb4c`. It exactly reproduces the
current-size schedule `[56,56,66,68,80,80,80,80,80,80]`, converges at
iteration 10, and enters the final all-data path using state 10 and half-1
joined noise.

Final canonical RECOVAR-vs-RELION FSC-AUC is `0.997260`, above the `0.995`
gate. Minimum non-DC shell FSC is `0.994984` at shell 27, fifth percentile is
`0.995371`, and the minimum over the last ten shells is `0.996734`; there is
no unexplained systematic shellwise deficit. RECOVAR-vs-GT FSC-AUC is
`0.670484` versus RELION `0.650835`, with FSC=0.5 crossings at shells 41 and
40. Only GT shells 1--3 are lower for RECOVAR, by `-0.000016`, `-0.000266`,
and `-0.000171`, all well inside the `0.002` gate. Correlation remains
diagnostic only and is not used in this acceptance.

This is the accepted 3k/128 white-noise supplied-map K=1 strict checkpoint.
The next phase is the predefined small robustness matrix with RELION generated
and run per case: high noise, nonuniform and Kent angle distributions, no CTF,
outliers, contrast/noise-scale variation, and image-offset stress. Failures
must be localized at the earliest trajectory boundary before proceeding to
10k, real-particle, 100k/256, or K=4 evidence.

## 2026-07-13 Exact M-step Geometry and Remaining 10k Iteration-1 Tie

The strict 10k/128 iteration-1 investigation closes the production
rotation-to-scatter geometry boundary without claiming complete BPref or
trajectory parity.  The active branch is
`codex/em-parity-checkpoint-20260711`; the exact-head full gate used commit
`1e8ad088b9d2c7837308056a105763508dfca165`.

RELION constructs the M-step inverse matrix on the CPU in double precision,
using its explicit 3x3 cofactor, determinant, and division order, before
casting the matrix for the CUDA backprojector.  RECOVAR now carries a separate
host-generated M-step rotation stream while leaving projection, scoring,
posterior, and winner selection on the existing score matrices.  Global
sparse pass 2 and local adjoint plumbing preserve the two streams separately.
The backprojection coordinate expressions also use RELION's operand order so
the CUDA compiler contracts the same multiply-add pairs.  These changes are
commits `e5bcb00e`, `07562137`, `59ac0b35`, `62940cc0`, `e1415768`, and
`1e8ad088`.

Production signature job `11137420` completed on an A100 in 131 seconds.  Its
marked root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_mstep_fma_signature_gate_20260713_142542`.
Across 128 particles, the RELION-host, RECOVAR-host, and CUDA-consumed rotation
matrices have zero uint32 mismatches.  Cutoff, fold, interpolation base, and
neighbor mismatch counts are all zero.  Raw-coordinate, radius-squared, and
interpolation-coefficient relative L2 errors are zero; Fweight relative L2 is
`3.53404e-7`.  Deterministic accumulator relative L2 is `2.91911e-7` for
half 1 and `2.70024e-7` for half 2.  The JSON and report are
`audit/production_mstep_signature_audit.json` and
`audit/production_mstep_signature_report.md` under that root.

The full exact-head A100 gate is Slurm job `11137642`, rooted at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_10k128_it1_hostrot_fma_gate_20260713_142749`
and marked `SAFE_TO_DELETE`.  It hash-verifies the immutable fresh RELION maps
and pre-reconstruct BPref arrays from job `11130337`.  The merged
RECOVAR-versus-RELION normalized FSC-AUC is `0.999999985292`; half-map values
are `0.999999952467` and `0.999999996172`.  RECOVAR merged GT FSC-AUC is
`0.227550394410` versus RELION `0.227551223235`.  RECOVAR's GT FSC=0.5 and
FSC=0.143 resolutions are `29.575925` and `27.475383` Angstrom, versus
RELION's `29.576021` and `27.475483` Angstrom.  Correlation is auxiliary and
does not participate in this gate.

The initially unexplained supported BPref numerator/weight relative-L2 errors
were `0.0116863180/0.00257571901` for half 1 and
`0.000804700285/0.0000836767217` for half 2.  Shells 15--28 were nearly closed
in half 2, while the half-1 numerator error grew to about 2.8 percent at the
outer supported shells.  Comparing saved winners to RELION by numeric
`rlnImageName` localized the material asymmetry to one of 10,000 particles:
zero-based original index 4394 in half 1.  RECOVAR chose psi
`-50.585873` degrees and translation `[0.13351604, 0.63351607]` pixels;
RELION chose the neighboring psi `-46.835870` degrees and translation
`[0.133516, 0.133516]` pixels.  All half-2 winners and all other material
half-1 winners agreed, apart from STAR/float32 serialization noise.

The exact score adjudication is rooted at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_it1_p4394_score_audit_20260713_150500`
and marked `SAFE_TO_DELETE`.  Corrected RELION capture job `11140143` maps
RELION's internally sorted particle id 3774 to original image
`4395@particles.128.mrcs`; the earlier direct-id assumption was wrong and its
RELION target is excluded.  All `1,069,056` coarse hypotheses map one-to-one.
The centered complete-score-surface absolute gap has p95 `2.259016e-5` and
maximum `8.808076e-5`, both below the previously accepted 3k numerical band.
RECOVAR ranks coarse candidates `(rotation 31658, translation 15)` and
`(32426, 14)` first/second by `3.516674e-6`; RELION ranks the same pair in the
opposite order by `2.712011e-6`.  The candidate-specific residual difference
is exactly the sum of those opposite margins and the flip-identity residual is
zero.  This qualifies the discrete winner difference as numerical noise in
the underlying scores, rather than a support, mapping, or tie-order bug.

Production CUDA causal-replay job `11141660` replaced only particle 4394's
RECOVAR winner contribution with its RELION winner contribution.  The half-1
numerator relative-L2 residual fell from `0.0116863180` to `0.0014925369`
(`87.23%` reduction), and the weight residual fell from `0.0025757190` to
`0.0002483004` (`90.36%` reduction).  This causally attributes the asymmetric
raw BPref tail to the qualified coarse near-tie and its different downstream
fine support.  No production tie-breaking patch is justified.  The
hash-pinned classification is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_it1_p4394_score_audit_20260713_150500/particle_4394_classification.json`.

Commit `e12b230a` completes the separate host M-step rotation generator path
for numbered K=1/K=4 local search and K=1 final local all-data.  It preserves
the public float32 scoring Euler API, derives adjoint matrices from binding
source-precision float64 Euler rows, and requests aligned host matrices for
lazy parent-expanded/adaptive local layouts.  Ten focused tests pass.  The
full `test_refine_relion_mode.py` result was 281 passed and five failed; all
five failures reproduced unchanged on exact base `1e8ad088`, so none is
introduced by this slice.  The current-head full-trajectory validation is
recorded below.  Do not infer robustness, scale, real-data, or K=4 quality
parity from the completed iteration-1 classification alone.

## 2026-07-13 Current-Head K=1 Per-Iteration Boundary-Replay Acceptance

Fail-closed A100 job `11144457` completed `0:0` in 12 minutes 10 seconds at
commit `ef2dbd065812bafd3e31ba7863f4a2975414c249`.  Its root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_3k128_full_trajectory_preflight_20260713_144434`
and is marked `SAFE_TO_DELETE`.  The runner required a clean exact head, the
local M-step generator commit `e12b230a` in ancestry, and the SHA-256-qualified
particle-4394 score classification.  It also hash-verified all 95 immutable
RELION oracle inputs and kept `RECOVAR_FINAL_ALL_DATA_GRID_CORRECT` and
`RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER` unset.

Semantic audit after acceptance established that this runner is not an
autonomous trajectory. `scripts/run_multi_iter_parity.py` loads a
`replay_iteration_overrides` entry from each RELION `run_itNNN_data.star`,
model, and optimiser state, injecting previous poses, corrections, noise,
direction priors, sigma-offset, and convergence control before every numbered
step. Historical job `11026304` used the same mechanism despite its old
"free" label. These runs strongly qualify every fixed transition and the
finalization boundary, but they do not prove that RECOVAR's own evolving state
stays on the RELION trajectory without intervention. The autonomous
cold-start gate remains open and must omit per-iteration replay overrides and
current-size oracles.

The exact current-size schedule is `[56,56,66,68,80,80,80,80,80,80]`.
RECOVAR and RELION converge at iteration 10, after which RECOVAR enters the
valid converged final-all-data Nyquist path exactly once.  An independent
NumPy FFT/shell-binning audit, which does not call RECOVAR FSC routines,
recomputes every numbered half-map and merged curve.  Its worst numbered shell
FSC is `0.999992610469` at iteration 10, half 2, shell 41.  The worst numbered
half/merged normalized FSC-AUC is `0.999998585696`, and numbered merged GT
FSC-AUC deltas range from `-5.306682553e-6` to `+4.833533575e-6`.

The final merged RECOVAR-vs-RELION normalized FSC-AUC is `0.998450626094`.
The minimum final shell FSC is `0.997581338511` at shell 51; low, middle, and
high frequency-band means are `0.999657940515`, `0.997986171031`, and
`0.997799875262`.  No final shell is below `0.995`, `0.99`, or `0.95`, and
there is no isolated or coherent shellwise collapse.  Final RECOVAR GT
FSC-AUC is `0.670747381970` versus RELION `0.650834885635`, a delta of
`+0.019912496335`; FSC=0.5 crosses at shell 41 for RECOVAR and shell 40 for
RELION.  Correlation is not used in this decision.

The automated report, independent audit, saved shell arrays, and signed-off
manual review are `TRAJECTORY_FSC_REPORT.md`,
`independent_audit_job_11144457.json`, `trajectory_fsc_arrays.npz`, and
`MANUAL_SHELLWISE_REVIEW.md` under the run root.  This closes the fixed
3k/128 white-noise K=1 per-iteration boundary-replay gate at the current head.
The autonomous trajectory, K=1 robustness, scale, and real-particle quality
gates remain open; K=4 strict quality follows those. The run is not a speed
comparison.
# 2026-07-13: autonomous K=1 final gap and expected-accuracy/PPref boundary

- Clean candidate: `5a5769df37e49674c118697f60e73cbdd706b880` on
  `codex/em-parity-checkpoint-20260711`.
- Autonomous job `11151255` passes all ten numbered schedule, convergence,
  shellwise FSC-AUC, and GT gates.  The one final Nyquist map fails only the
  cross-FSC-AUC gate: `0.986771` versus required `0.995`; GT FSC-AUC delta is
  `+0.020688` in RECOVAR's favor.  Evidence root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_autonomous_coarsehvc_currenthead_20260713_183403/`.
- Exact iteration-10 final-only job `11151769` passes at cross FSC-AUC
  `0.998457`, with RECOVAR GT FSC-AUC `0.670751` versus RELION `0.650835`.
  Evidence root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_final_exact_iter10_currenthead_20260713_190800/`.
- Binding replay on the exact first 100 IDs gives `1.844` degrees and
  `1.6915` Angstrom; authoritative RELION gives `1.858` and `1.6915`.
  `RELION_SINGLE_PRECISION` is not causal: the oracle log records
  `BASE=double` and CPU double.  The binding is also double.
- RELION builds `PPref` during expectation setup through the CUDA gridding,
  padding, and `cufftExecD2Z` branch when `--gpu 0` is active, then consumes
  that double `PPref` in the CPU expected-error loop.  RECOVAR's binding
  suppresses CUDA and uses CPU FFTW.  GPU job `11152404` rejects that backend
  as causal: JAX/cuFFT and CPU FFTW `PPref` agree at relative L2 `3.67e-16`,
  and both return `1.844` / `1.6915`.
- Same-process RELION job `11152475` reproduces the authoritative
  `1.858` / `1.6915` and directly confirms first vector IDs
  `[2313,2343,2409,806,815]`.  The remaining expected-accuracy audit is the
  live in-memory state versus serialized MRC/STAR inputs, not particle order
  or the FFT backend.
- A naive CPU continuation job `11152191` is a negative control, not evidence:
  RELION re-randomized particle order and reported `2.058` degrees /
  `1.72975` Angstrom before intentionally failing on a relative stack path.
  Do not compare that aggregate to the original-process values.
- Exact-final raw BPref audit root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_it2_particle_replace_20260713_165017/exact_final_bpref_relion/`.
  Accumulator substitution raises replay FSC-AUC from `0.995640221` to
  `0.998443887`; tau2 substitution alone reaches only `0.995652710`.
- Autonomous pose comparison is exact at iteration 1 except for the qualified
  single translation near-tie.  Rotational discrepancies above `0.1` degree
  then grow from 3 particles at iteration 2 to 110 at iteration 3, 277 at
  iteration 4, and 1,050 at iteration 10.  The next final-map discriminator is
  a final-state map/pose factorial, with FSC/FSC-AUC as its acceptance metric.

# 2026-07-13: expected-accuracy closure and correction-state root cause

- RELION's `Mresol` excludes the redundant packed `x=0, y<0` column. The
  expected-accuracy binding did not. The exact exclusion changes
  `1.844/1.6915` to RELION's `1.858/1.6915`; the Nyquist guard is independently
  null. Jobs `11152475`, `11152727`, and `11152933` completed `0:0`. Full
  report:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_ppref_cpu_ab_20260713_192524/ROOT_CAUSE_REPORT.md`.
- Corrected autonomous job `11153043` completed `0:0` in `00:09:49` on A100.
  It preserves the exact current-size, HEALPix, convergence-at-10, and final
  Nyquist schedule. Final merged cross FSC-AUC is `0.986985443`; RECOVAR GT
  FSC-AUC is `0.671500068` versus RELION `0.650834886`.
- Exact-state factorial evidence identifies image/group-scale corrections as
  the material final-state component. RECOVAR poses alone pass at
  `0.995216198`; its map alone passes at `0.996873606`; both give
  `0.993407704`. Adding RECOVAR corrections gives `0.986822205`; adding
  noise/tau2 and direction priors changes only the fourth-to-sixth decimal.
  Evidence root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_autonomous_final_state_factorial_20260713_194341/`.
- The immediate cause of scale arrays staying exactly one is disabled native
  group statistics: `/data/particles.star` has no `rlnGroupNumber`, but the
  supplied RELION `run_it000_data.star` has 3,000 singleton groups generated
  from unique micrographs. Map those rows by `rlnImageName`; never index the
  lexicographically reordered RELION table with dataset-order half indices.
  Preserve all 3,000 group slots in both half models, including absent groups.
- After group statistics are enabled, match RELION's scale-statistic support:
  collect `XA/AA` only where the iteration-start class
  `data_vs_prior[ires] > 3`. This was the remaining pre-fix discrepancy.
  Correlation is not an acceptance metric for any of these decisions.

# 2026-07-13: autonomous K=1 native-correction acceptance

- RECOVAR now maps the supplied RELION data STAR to dataset order by exact
  `rlnImageName`, preserves all 3,000 model-group slots in both halves, and
  accumulates group XA/AA only on the iteration-start `data_vs_prior > 3`
  shells. Dense fallbacks fail closed rather than silently dropping native
  scale statistics. The parity NPZ records raw XA/AA, group identities/counts,
  and post-update normalization state.
- Two-iteration A100 job `11154785` completed `0:0` in `00:02:36`. At
  iteration 2, group IDs and per-group particle counts match RELION exactly.
  Group-scale median absolute errors are `3.456e-5` and `5.354e-6` by half;
  relative-L2 errors are `0.001254` and `0.000359`. After the exact `128^2`
  unit conversion, average norm errors are `1.770e-5` and `1.512e-5`.
  Evidence:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_native_scale2iter_20260713_210956/iter2_correction_comparison.json`.
- Autonomous A100 job `11154968` completed `0:0` in `00:10:17`. It matches
  the exact schedule `[56,56,66,68,80,80,80,80,80,80]`, converges after
  numbered iteration 10, and enters one valid Nyquist all-data pass. All
  numbered FSC gates pass; their minimum half/merged normalized FSC-AUC is
  `0.999874272`.
- Final merged RECOVAR-versus-RELION normalized FSC-AUC is `0.997935505`,
  improving the pre-fix `0.986985443` failure. The minimum non-DC shell FSC is
  `0.995978466` at shell 60; low/mid/high band means are
  `0.999652168/0.997622576/0.996633032`. No shell is below `0.995`.
  RECOVAR GT FSC-AUC is `0.670747694` versus RELION `0.650834886`, delta
  `+0.019912809`. Correlation was forbidden and was not computed.
- Automated report and manual shellwise sign-off:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_autonomous_native_scale_20260713_212530/particleids_exact_accuracy_fsc_job_11154968.json`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_autonomous_native_scale_20260713_212530/MANUAL_SHELLWISE_REVIEW.md`.
- This closes the single 3k/128 white-noise autonomous K=1 fixture. Robustness,
  scale, real-particle, and K=4 quality gates remain open.

# 2026-07-14: case-22 translation-phase precision closure

- Case-22 exact per-pixel RELION CUDA capture jobs `11164971` and `11165146`
  completed on an A100. Exact unit mapping and hybrid scores isolate the
  first-iteration normalized-CC winner flips to RECOVAR's weighted translated
  image; projection interpolation, CTF, and score reduction are null.
- Fitting the complex-image residual to a Fourier phase ramp yields effective
  translation errors that match TF32-rounded float32 shifts. For example,
  `-2.0479863` becomes `-2.0488281` (delta `-0.000841856`) and `1.9520137`
  becomes `1.9521484` (delta `+0.0001347065`). Removing the ramp leaves only
  `1.24e-7`--`1.47e-7` weighted phase RMS.
- Commit `c741faee266f243877f43ecb457a9debd22e6bcf` sets
  `jax.lax.Precision.HIGHEST` on generic, full-half-table, and indexed-half-
  table candidate phases. Commit `b658bd8d12bac32a72040d309bad6f259a8e2f87`
  applies the same contract to the separate per-image Fourier pre-shift phase.
  Compiled-jaxpr regressions prevent silent TF32 reintroduction in all four
  paths.
- A100 job `11177896` restores both RELION winners. Full score-field RMS falls
  by about 8--9x to `1.6304e-6` and `1.7883e-6`; the four corrected target
  score gaps are between `-7.45e-7` and `+5.66e-7`.
- Focused tests: 42 passed across core geometry, RELION E3 shifts, all four EM
  phase paths, and pose/translation score recovery.
- Canonical evidence is under
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_early_score_audit_20260713_235725/`.
  The next gate is the autonomous case-22 trajectory at the exact clean head,
  judged by shellwise FSC/FSC-AUC plus schedule/convergence/finalization. Track
  phase-generation timing because the generic precision change is cross-cutting.

# 2026-07-14: first phase-corrected robustness cell accepted

- Clean-head A100 job `11178306_1` completes case 15 (3k/128, 20% outliers,
  noise scale 1) in `00:12:42` at commit `6604b129`.
- All 12 numbered per-half/merged FSC-AUC gates pass. The exact RELION size
  schedule is `[56,56,64,66,66,66,68,68,78,82,82,82]`; the old run used 76
  instead of 78 at iteration 9. Convergence and the single valid final
  all-data path now match exactly.
- Final merged cross-FSC-AUC is `0.996927042`, and RECOVAR GT FSC-AUC exceeds
  RELION by `+0.019123653`. The final non-DC shell minimum is `0.993255592`,
  fifth percentile `0.993867482`, and no shell is below `0.99`. Manual review
  accepts the predefined gate while retaining the shallow high-shell tail as
  a final-boundary diagnostic.
- The phase timing audit finds no slowdown: for the same 16 ledger entries,
  total phase time changes from `0.483892` to `0.479649` seconds and the
  post-first median from `0.006245` to `0.006081` seconds.
- Evidence and manual sign-off:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_robust_phase_precision_20260714_143051/15_small_outliers_3k_g128_pct20_noise1_bf80/`.

# 2026-07-14: case-16 norm `sum_weight` bug and plumbing validation

- Exact RELION-state iteration-3 replay matches all 3,000 hard poses. The
  autonomous five-particle divergence is therefore upstream state drift, not
  an unexplained scorer discrepancy.
- RELION accumulates one unweighted updated `normcorr` per particle but divides
  by retained significant posterior `sum_weight`. RECOVAR divided by image
  count. In addition, sparse adaptive K=1 correctly produced retained mass,
  but `_assemble_result` replaced it with the single class responsibility
  sum, exactly `N`.
- The production fix preserves `NoiseStats.sumw` for K=1 and uses it as the
  norm denominator. Unit coverage independently guards the denominator formula
  and the K=1 aggregation boundary. The EM fast guard passes 16/16.
- A100 jobs `11183647` (default posterior) and `11183648` (diagnostic float32
  posterior) both complete successfully and agree. Half-set retained masses
  are `1468.013404` and `1529.966030`; internal average norms improve to
  `5212.8874` and `5212.7233`. The diagnostic posterior mode is not needed.
- This closes the formula and routing bug only. Do not call case 16 accepted
  until the canonical autonomous trajectory passes shellwise FSC/FSC-AUC,
  schedule, convergence, and finalization gates.
- Durable source/factorial report:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_robust_phase_precision_20260714_143051/16_small_anisotropic_outliers_3k_g128_pct25_noise3_bf80/audit_case16_divergence_20260714/exact_iter3_score_20260714_145500/analysis/NORM_SUM_WEIGHT_ROOT_CAUSE.md`.
- Validation root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/case16_norm_sumw_fix_validation_20260714_171639/`.

# 2026-07-14: case-16 autonomous numbered trajectory accepted; final branch open

- Clean A100 job `11184169` at `d685ba36` completes in `00:08:58` and exactly
  matches RELION's 11-iteration size/order schedule and convergence boundary.
- Numbered merged cross-FSC-AUC is
  `1.000000000, .999999999, .999999997, .999999998, .999999998,
  .999999999, .999999999, .999999999, .999999999, .999998470,
  .999987621`. Numbered GT FSC-AUC differences are at most `4.01e-5`.
- Hard rotation/translation mismatch counts by iteration are
  `0/0, 0/0, 1/1, 0/0, 1/1, 1/0, 1/0, 0/0, 2/0, 28/15, 25/19`.
  The first difference is a diffuse close choice, while map FSC remains
  effectively exact; do not require brittle discrete tie identity.
- The final all-data cross-FSC-AUC is only `0.743531728` (RECOVAR GT
  `0.293003`, RELION GT `0.238181`). This is a real unresolved final-branch
  mismatch despite the exact control trajectory. Decompose final poses,
  half accumulators, joined FSC/tau2, and final reconstruction before changing
  numbered EM.
- Audit:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/case16_norm_sumw_fix_validation_20260714_171639/autonomous_default_commit_d685/analysis/TRAJECTORY_AUDIT.md`.

# 2026-07-14: case-22 normalized-CC complex reduction boundary

- Same-A100 RELION/RECOVAR capture proves original particle 1552 sees the
  same coarse parent and the same ordered 32 fine candidates. RELION chooses
  `(r1,t89)` over `(r4,t88)` by `4.76837e-7`; RECOVAR's complex c64
  `dot_general` reverses them by `5.96046e-8`.
- The reference norm is bit-identical. The discrepancy is only the complex
  numerator reduction. Explicit float32 real/imaginary products plus
  float32 `reduce_sum` recover RELION's winner with a `4.47035e-7` margin.
- The production patch changes only normalized firstiter-CC pass-2 cross
  terms in bucketed, cached-single, and compact-pair routes. Norm contractions
  remain unchanged. Focused reduction tests plus the EM fast guard pass.
- A100 job `11185051` matches all 3,000 canonical iteration-1 Euler/origin
  decisions with zero threshold failures and restores the target particle.
  Do not accept or commit solely from this boundary: require the autonomous
  case-22 FSC/FSC-AUC trajectory, schedule, convergence, finalization, and
  end-to-end timing.
- Capture:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it1_particle1553_capture_20260714_160000/ARITHMETIC_REPORT.md`.
- Validation:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_explicit_cross_it1_validation_20260714_174500/`.

# 2026-07-14: post-firstiter robustness boundaries

- Case 22 clean-head A100 job `11185459` proves the firstiter reduction fix is
  exact at iteration 1 and keeps numbered merged cross-FSC-AUC at least
  `0.99887014` through iteration 8. It still chooses 72 instead of 70 at
  iteration 9, converges at 9 instead of 11, and fails final cross-FSC-AUC at
  `0.8245735`. Continue at the ordinary Gaussian scorer; do not undo the
  exact 3,000-pose firstiter fix.
- Case 20 same-H100 job `11185799` converges at the same iteration 11 and has
  near-exact numbered FSC, but differs in current size at iterations 8 and 10
  and fails final cross-FSC-AUC at `0.9851148007`. Its first hard differences
  are two iteration-2 Gaussian pass-2 decisions.
- Case 11 same-A100 job `11185798` passes the strict robustness gate; retain
  its full audit alongside cases 20 and 22 under
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_cleanhead_68a3f9e6_20260714_180000/`.

# 2026-07-14: case-16 native final perturbation-order bug

- Exact RELION iteration-11 state replay gives final cross-FSC-AUC `0.997899`.
  Autonomous references plus exact RELION metadata give `0.997343`; neither
  final machinery nor the half references explain the autonomous `0.743532`.
- Native autonomous finalization used the exhaustive order 3 angular step for
  SamplingPerturbation. RELION final `sampling.star` uses active local order 4.
  Both observed 36,864-rotation manifests match their canonical construction
  exactly and differ by a common `0.682834` degree right rotation.
- `_native_final_perturbation_healpix_order` now selects active
  `state.healpix_order` for local search and preserves exhaustive-grid order
  for global search. Focused merge guards pass `23/23`; EM fast guard passes
  `16/16`. Require corrected autonomous case-16 FSC/FSC-AUC before acceptance.

# 2026-07-14: corrected case-13 classification and integrated validation

- Retract the earlier claim that particle 1682 demonstrated an intrinsic
  Gaussian coarse-scorer mismatch. That analysis used zero-based RECOVAR
  index 1682 as if it were RELION's one-based stack index 1682. The corrected
  target is `1683@particles.128.mrcs`. Its measured score gap is explained by
  a real workflow bug: the final joined E-step duplicated half-1 noise for
  both random subsets. Half-2 noise substitution leaves only a qualified
  float32 arithmetic residual. Commit `24c5157f` fixes the per-half routing.
- Clean immutable A100 job `11190363` at `d07915fa` matches case-13's exact
  nine-iteration size schedule and convergence boundary. Worst numbered
  merged cross-FSC-AUC is `0.999999970691`; final joined cross-FSC-AUC is
  `0.997779297632`. Final merged GT FSC-AUC is `0.312357369405` for RECOVAR
  versus `0.301136422552` for RELION. Correlation was not computed.
- Correctly indexed matched-grid captures distinguish numerical ties from
  remaining behavior. Final particle 2701 has an exact RELION top tie and a
  centered score residual of only `0.003113` maximum (`0.000855` RMS).
  Final particle 2828 has the same robust winner and the same few-ULP maximum;
  its uninterrupted `0.142079` Pmax gap is not reproduced by the replay.
- Retract the iteration-9 particle-1466 structured-residual claim. The
  restarted RELION diagnostic broadcast rank-1 `sigma2_noise` to the subset-2
  follower during MPI initialization, while RECOVAR retained the correct
  half-2 curve. The contaminated `0.446167/0.180758` maximum/RMS and winner
  flip are not an uninterrupted trajectory comparison. Fixed-state A100 job
  `11192981` proves both effective curves match half 2 (RELION/RECOVAR relative
  RMS `2.75e-6/3.00e-6`) and reduces the all-RECOVAR-operand residual to
  maximum/RMS `0.001709/0.000237`, with the same `(17,43)` winner and Pmax
  delta `-8.8e-8`. Final particle 188 remains the next separate localization
  target: 128 candidates/four parents, `0.187515/0.065117`, agreeing replay
  winner, and no reproduction of the original hard-pose boundary.
- Mid-trajectory RELION restart captures are fail-closed unless they run
  uninterrupted or record the particle random subset and shellwise prove that
  `CTF^2 * group_scale^2 / corr_img` matches that subset's previous-iteration
  model STAR. Do not infer formula parity from a restarted score dump without
  this state gate.
- RELION array `11191084_[0-2]` completed. Task `11191084_3` exited 1 because
  a `6e-7` check was stricter than five-decimal STAR serialization; its
  iteration-9 dump was independently hash-checked and perturbation-checked at
  `3e-6`, then marked `PASS_RELION_EXACT_PERTURB_DUMP_POSTVALIDATED`. Do not
  describe the whole array as completed. RECOVAR `11191497_1` completed the
  fine capture; `11191497_0` requested a parent/probe hook absent from the
  numbered path and produced no scientific artifact.
- Durable joint audit:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case13_targeted_scores_d079_20260714_201200/JOINT_RELION_RECOVAR_SCORE_AUDIT.md`.
- RELION postvalidation details and exact hashes:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case13_targeted_relion_scores_d476_exactperturb_20260714_204000/RELION_EXACT_PERTURB_POSTVALIDATION.md`.
- Corrected fixed-state job `11192981`, audit, machine-readable shells, and
  logs:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case13_it9_z1466_fixed_half_noise_20260714_211456/FIXED_STATE_OPERAND_AUDIT.md`,
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case13_it9_z1466_fixed_half_noise_20260714_211456/operand_substitution_audit.json`,
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case13_it9_z1466_fixed_half_noise_20260714_211456/logs/run_11192981.out`, and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case13_it9_z1466_fixed_half_noise_20260714_211456/logs/run_11192981.err`.

# 2026-07-14: case-22 iteration-2 qualified numerical butterfly

- RECOVAR debug iteration names are zero-based here: `it000` is the
  post-iteration-1 reference consumed by iteration 2. Using `it001` would be
  a post-iteration-2 look-ahead and is invalid for this comparison.
- Original particle 1203 is a true one-float32-ULP boundary. The native
  RECOVAR post-iteration-1 reference gives exact full-grid trees
  `130.3276519775`/`130.3276672363`, so the native candidate wins by one ULP.
  Replacing only that reference with RELION's post-iteration-1 reference gives
  `130.3275909424`/`130.3275756836`, so the RELION candidate wins by one ULP.
  This preserves a causal role for the small accumulator/BPref-to-reference
  perturbation even though it is globally almost invisible by FSC.
- Isolated RELION `AccProjectorKernel` capture for the same RELION reference
  closes the missing projector factor. On the 1,461 `corr_img>0` pixels, the
  RELION-versus-RECOVAR projector relative L2 is `6.20e-9` for the native
  candidate and `1.05e-6` for the RELION candidate; the corresponding
  corr-weighted values are `2.68e-8` and `7.28e-6`. The apparent full-grid
  relative L2 near `0.5` is confined to 399 zero-weight pixels. RELION and
  RECOVAR projectors produce bit-identical 256-lane float32 trees and the same
  one-ULP RELION winner on the identical active operands.
- RELION is numerically nondeterministic below that decision boundary: the
  captured high-resolution Xi2 half-constant varied from float32 bits
  `0x3ce5b2d8` through `0x3ce5b2db`. Subtracting it reproduces the exact trees,
  and adding it leaves both raw candidate scores and their margin unchanged.
  Two cross-device A100 repeats and two serial repeats on physical GPU UUID
  `GPU-4bccbe72-c64a-5f5f-1fa8-ecf0bf6acf37` all select the RELION candidate
  by exactly one ULP. The same-device map FSC-AUC is
  `0.9999999999988929`; no correlation metric was computed.
- Classification: qualified numerical butterfly, not an intrinsic active-grid
  projector or high-resolution-constant bug. Do not add a tie-break patch for
  this particle. Continue upstream at the accumulation/reference arithmetic,
  and retain the autonomous case-22 FSC/FSC-AUC failure as unresolved.
- Projector audit:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_particle1203_factorial_20260714_201047/relion_fine_projection_capture/fine_projection_comparison.json`.
- Self-jitter FSC audit:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_particle1203_self_jitter_20260714_202700/self_jitter_audit.json`.

# 2026-07-14: broad K=1 final-map gridding-correction boundary

The immutable seven-case K=1 matrix at detached commit
`f0ef1f0c6c231ff1f9183371d235e0b37a15b825` completed as Slurm jobs
`11192470--11192477`.  All cases match RELION's complete current-size schedule
and converge on the same numbered iteration.  The numbered and final joined
FSC curves already localize the systematic output residual after convergence.

Applying RECOVAR's source-derived RELION radial sinc-squared correction
`_gridding_correct_trilinear_np(map, 128, 2)` to the existing final joined map
gives the following canonical non-DC FSC-AUC results.  These use
`regularization.get_fsc_gpu` and the normalized trapezoid convention used by
`run_multi_iter_parity.py`; no correlation metric is used.

| case | cross, grid off | cross, grid on | REC GT, grid on | RELION GT | GT delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| 21 | 0.998350520 | 0.999998612 | 0.388073525 | 0.388074334 | -0.000000809 |
| 23 | 0.998368802 | 0.999991464 | 0.441903887 | 0.441903639 | +0.000000248 |
| 24 | 0.998345948 | 0.999998285 | 0.351268986 | 0.351266822 | +0.000002164 |
| 25 | 0.998163030 | 0.999961340 | 0.317328521 | 0.317318189 | +0.000010333 |
| 26 | 0.997395347 | 0.999605407 | 0.211284140 | 0.211293473 | -0.000009333 |
| 29 | 0.998869708 | 0.999999692 | 0.842265375 | 0.842262860 | +0.000002515 |
| 30 | 0.998823366 | 0.999999673 | 0.767898858 | 0.767898392 | +0.000000466 |

This closes the systematic RECOVAR-better-than-RELION GT FSC-AUC offset as an
explicit gridding-correction behavior difference, not numerical noise.  The
remaining corrected identity target is severe case 26.

The end-to-end grid-on confirmation completed in repaired jobs
`11194076--11194077` under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case25_grid_on_7ee984cb_20260714_213300`.
Case 25 matched the RELION current-size schedule and converged at numbered
iteration 8.  The final path explicitly logged radial gridding correction
enabled.  Canonical normalized non-DC FSC-AUC is `0.999961352645` versus the
RELION final map; RECOVAR-vs-GT is `0.317329222939` and RELION-vs-GT is
`0.317318951714` (delta `+1.02712e-5`).  This agrees with the post-hoc result
and establishes the systematic seven-case final-map offset as an actual
output-option boundary.  Keep the quality-path default unchanged outside
named strict-parity diagnostics.

Matrix evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_broad_f0ef1f0c_immutable_20260714_211000/k1_robustness_matrix_summary.json`.

# 2026-07-14: case-20 exact-Gaussian paired control rejects a causal quality claim

- Isolated candidate `7ad2526da4ce41a0a5b32736b41b9d23c08d81f4`
  implements RELION's float32 fine-Gaussian diff2 tree, common minimum, prior
  addition order, and high-resolution tail.  It remains unmerged.
- Same-H100 job `11194268` ran exact then algebraic modes sequentially with the
  same immutable checkout and replay inputs.  Both modes used current sizes
  `[56,56,52,52,50,50,50,52,50,52,50]`, converged at iteration 11, and ran
  the converged final-all-data path.
- The exact/algebraic maps already differ at iteration 1, where normalized CC
  bypasses the Gaussian scorer.  Their iteration-1 merged FSC-AUC is
  `0.999999999952`; this is the sequential reconstruction/atomic jitter floor,
  not a scorer effect.  Numbered exact-vs-RELION minus algebraic-vs-RELION
  FSC-AUC changes sign and remains between approximately `-1.04e-9` and
  `+3.54e-9`.
- Raw saved-final exact/algebraic RECOVAR-vs-RELION FSC-AUC is
  `0.997633868321/0.997633784286` (exact `+8.40e-8`).  After applying the
  explicit RELION radial sinc-squared final boundary it is
  `0.999502423092/0.999502801876` (exact `-3.79e-7`).  Exact also changes raw
  and strict-grid GT FSC-AUC by `-3.51e-8` and `-2.02e-8`.  There is no
  consistent FSC/FSC-AUC quality benefit above the jitter floor.
- External wall is `776` seconds exact versus `742` seconds algebraic
  (`+4.58%`); sampled peak GPU memory is `41245/42049` MiB.  Production did
  not save bounded raw hypothesis-score arrays, so this trajectory cannot
  promote the feature from array-level diagnostic to default behavior.
- Focused exact-score tests passed, including the captured particle-469
  one-ULP operation-order boundary, the CUDA-tree NumPy oracle within one ULP,
  and array-equal dense/cached/compact routes.  Additional local validation:
  `tests/unit/test_sparse_pass2_relion_diff2_tree.py` passed `16/16`, and the
  existing K-class fused-vs-two-pass node passed `1/1`.
- Conclusion and FSC-only artifacts:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_exact_gaussian_ab_7ad2526d_20260714_215100/PAIRED_AUDIT_CONCLUSION.md`,
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_exact_gaussian_ab_7ad2526d_20260714_215100/audit_exact_vs_algebraic/trajectory_fsc_audit.json`, and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_exact_gaussian_ab_7ad2526d_20260714_215100/audit_exact_vs_algebraic/shellwise_fsc_curves.npz`.

# 2026-07-16: replacement exact fine-Gaussian reducer integrated provisionally

- Commit `0cacb891` reintroduces the replacement RELION CUDA-style float32 fine
  `diff2` reduction.  Commit `49e8f416` adds routing guards, fail-closed sparse
  behavior, and restored parity checks.  It remains provisional pending the
  production-size K=1 and K=4 trajectory gates below.
- Same-A100 iteration-2 A/B evidence changed only that disable flag.  Exact
  reduced mean Pmax error against RELION by `28.3248%`
  (`2.78821e-5 -> 1.99846e-5`), reduced rows above `1e-4` from 371 to 168, and
  reduced wall time by `49.439%` (`1065.49 -> 538.72` seconds).  Iteration-1
  state was bitwise equal and iteration-2 cross-engine merged FSC-AUC changed
  by only `+1.55e-10`.  The real fixture has no GT.
- The candidate uses the RELION 256-lane/tree reduction, zero-gap
  full-grid topology, and one common class-by-pose minimum.  K>1 refuses an
  unsafe per-class fallback, and host staging is capped by
  `RECOVAR_SPARSE_KCLASS_RAW_HOST_STAGING_MAX_BYTES` (8 GiB by default).
- Sealed evidence:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k1_exact_gaussian_it2_ab_f62eb37e_20260716_091600/analysis/SEALED_ANALYSIS.json`.
  Its SHA-256 is
  `bc1e27f16437913e8e2c081016a838d0a05b7b6539adcacb07f3d9ff4cc1a993`.
  Rerun the K=1 matrix at the integrated commit; do not treat K=4 as proven
  until production-size K=1 and K=4 trajectory runs pass.  Revert the
  integration if either quality gate regresses.

# 2026-07-14: case-26 first real divergence is accelerated BPref arithmetic

- Iteration-1 hard-WTA E-step decisions are exact: every pose, translation,
  and Pmax equals RELION.  Matched patched-RELION and RECOVAR H100 captures
  place the first non-bitwise boundary in the accelerated BPref accumulator,
  before low-resolution joining or reconstruction.  The exact comparison is
  RECOVAR post-x0 against RELION pre-lowres-join.  RELION pre-reconstruct is
  already after the 40-Angstrom half join and is not the same boundary.
- On radius `<56`, RECOVAR-vs-RELION relative-L2 is `5.663e-6/2.993e-6`
  (numerator/weight) for half 1 and `6.141e-6/2.914e-6` for half 2.  Scale fits
  near `1+2e-7` do not improve the residual.  The resulting iteration-1 maps
  still have half-map FSC-AUC `0.9999999566/0.9999999528` against RELION, but
  that tiny map difference is causally material later.
- Three same-H100 RELION controls, jobs `11194466`, `11195696`, and `11195697`,
  vary by only `1.0e-8--1.3e-8` relative-L2 in raw BPref.  Their map FSC-AUC is
  1.0 to printed precision with minimum shell FSC at least `0.99999982`.
  RECOVAR's `3e-6--6e-6` residual is `250--600x` larger and cannot be
  classified as RELION atomic nondeterminism.
- Production sparse-pass job `11195695` closes both iteration-3 pose changes.
  Particle 207 has identical 352-candidate support: the RELION map selects
  `(fine89300,t54)` by `0.002684`, while the RECOVAR map selects
  `(fine89302,t55)` by `0.001415`, a `3.689` degree change.  Particle 236's
  common candidates have a RELION margin of only `0.000881`; the RECOVAR map
  shifts their relative order by about `0.006386` and selects a candidate
  `26.039` degrees away with margin `0.005505`.  Its 32 RELION-only support
  candidates are irrelevant to the winner.
- The causal chain is therefore BPref accumulation residual -> tiny
  post-iteration-1 map residual -> ordinary Gaussian score-surface reorder ->
  later hard pose changes.  Do not patch tie-breaking or sparse candidate
  reporting.  The next discriminating experiment is a same-H100
  per-particle/prefix accumulator comparison to identify the first differing
  contribution, scatter coordinate, or arithmetic operation.
- Audit root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_earliest_score_audit_20260714_214916`.

# 2026-07-14: source-faithful accelerated image preprocessing integrated

- Commit `bdda53c47cc6426ea7b816fc8335606236304c60` adds explicit typed
  `image_fourier_backend="relion_cuda"`; `host_numpy` remains the default and
  `jax_gpu` remains the FFT-only diagnostic.  Unsupported CPU, dtype, shift,
  multiplier, mask-geometry, or custom-CUDA configurations fail closed.
- The CUDA FFI preserves RELION's stored float32 normalization and zero-fill
  translation stages, then launches exactly 128 blocks of 128 threads for the
  background reduction, atomically accumulates 128 lanes, performs the two CUB
  sums and float32 division, and uses CUDA `sqrtf`/`cospif` for the cosine fill.
  JAX/cuFFT performs the already-qualified centered rFFT and current-size
  window afterward.
- H100 job `11195746` and A100 job `11195763` each ran 100 repeats for captured
  particles 365 and 469.  Normalized/shifted pixels are bit exact 65536/65536
  on every launch.  Both particles reach a fully bit-exact 65536-pixel mask
  and bit-exact 1300/1300 current-size-50 Fourier window on both architectures.
  Worst mask relative-L2 is approximately `8e-9--1.1e-8`, explained by the
  deliberately RELION-compatible unordered inter-block atomic background
  addition; the captured RELION background occurs repeatedly on every GPU.
- Main-checkout validation built
  `recovar/cuda/libcuda_backproject.so` from commit `bdda53c4` and passed the
  focused GPU suite `72/72` on local A100 GPU 1 with checkout-bound RECOVAR and
  Pixi JAX imports.  The isolated branch also passed five CPU strict-routing
  tests, the 16-test EM fast guard, and selected Ruff checks.
- This closes the captured preprocessing operand boundary, not case-20
  trajectory quality.  The next gate is fixed-state score/logZ/posterior array
  comparison followed by a same-GPU full trajectory using explicit
  `relion_cuda`; judge maps only with FSC/FSC-AUC and GT FSC summaries.
- Full audit and reproduction commands:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/case20_rel_preprocess_boundary_20260714_220000/AUDIT.md`.

# 2026-07-14: case-26 strict scatter topology is not causal

- Same-H100 job `11195923` repeated iteration 1 with block topology,
  per-particle launches, fused real/imaginary/weight atomics, and sequential
  float32 translation reduction all enabled.  The correct comparison remains
  RECOVAR post-x0 versus RELION pre-lowres-join.
- Strict residuals are half-1 `5.797640e-6/3.010545e-6` and half-2
  `5.992257e-6/2.930651e-6` for numerator/weight.  Recomputed baseline job
  `11195621` is `5.798215e-6/3.010871e-6` and
  `5.993527e-6/2.931076e-6`.  The full strict topology improves only
  `0.01--0.02%`; launch grouping and fused atomic order are not causal.
- The residual is widespread and unbiased: only about `0.5%` of supported
  numerator components and `1.2%` of weights are bit equal; median final-value
  distances are 65--67 ULP for numerator components and 24--25 ULP for
  weights, while total supported weight is conserved within about `2.5e-8`.
- RELION constructs translated complex values and Fweight inside `BP.cuh`,
  including factor placement and `sincosf`, before neighbor atomics.  RECOVAR
  precomputes phases and reduces translations into `summed`/`ctf_probs` before
  the CUDA scatter.  RELION job `11195897` and RECOVAR job `11195923` captured
  complete value signatures for all 1000 particles.  Compare these pre-atomic
  rows and then bisect deterministic per-particle prefixes; do not revisit
  launch topology unless those operands match.

# 2026-07-14: adaptive `relion_cuda` plumbing fixed and runtime-qualified

- Commit `241db84d` wires typed float32 normalization and int32 integer-shift
  operands through both adaptive coarse significance loops and sparse pass 2.
  CUDA consumes unshifted real images and downstream code applies only the
  remaining group scale, so normalization is not doubled.  Host defaults are
  unchanged and unsupported operands still fail closed.
- Main-checkout focused guard: `67/67` passed.  H100 job `11196916` completed
  both halves of iteration 1 in 62.5 science seconds without OOM or routing
  failure.
- Direct iteration-1 non-DC sign-invariant FSC-AUC versus RELION is
  `0.999999999500` merged and `0.999999999448/0.999999999451` by half.  GT
  FSC-AUC is `0.098134227300` RECOVAR versus `0.098134193656` RELION.  The
  maps differ only by the known arbitrary global sign at this gate; no
  correlation metric is used.
- Evidence:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_relion_cuda_aeb337df_iter1_20260714_232010/`.

# 2026-07-14: native WAVG/atomic emulation is a decisive null

- The corrected two-particle adapter (`11196729`) separates centered source
  indices from RELION FFTW scatter coordinates and proves exact WTA 1/0
  posterior semantics.  The initial scrambled-index run `11196641` is
  inadmissible scientific evidence but retained as an audit failure.
- All-1000 H100 job `11196772` changes the case-26 BPref numerator/weight
  residual by only `0.007--0.046%`, with mixed signs.  Final residuals remain
  approximately `5.78e-6/5.99e-6` for numerator and
  `2.99e-6/2.91e-6` for weight, versus RELION repeat jitter near `1e-8`.
  Reject and do not merge this diagnostic path.
- Geometry, launch topology, translation-loop placement, factor placement,
  and atomic ordering are closed.  The remaining upstream pre-atomic value
  boundary is approximately `1e-6`.  In exact WTA, Fweight contains no image
  or phase term, so compare raw CTF and Minvsigma2 first, then Fimg and
  translation vectors.  RELION raw capture `11197096` and RECOVAR raw capture
  `11197128` are active.
- Machine audit and report:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_earliest_score_audit_20260714_214916/native_wavg_all1000_audit_11196772.json`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_earliest_score_audit_20260714_214916/native_wavg_all1000_report_11196772.md`.

# 2026-07-14: case-26 raw operands explain the residual

- Same-H100 jobs `11197096`/`11197128` establish a one-to-one mapping for all
  1,227 active pixels.  Image and CTF residuals after exact convention
  conversion are only `1.5e-7` and `2.2--2.6e-7`.  Translation coefficients
  differ by at most one float32 ULP over all candidates; both actual WTA
  winners and their 1,227 phase arguments are bit-exact, so phase is not part
  of the residual.
- Minvsigma2 is the largest raw difference at `9.062e-7`; combined pre-atomic
  data and weight differ at `1.0--1.5e-6` and `9.8e-7`.  This fully explains
  the prior value-signature gap.  Do not revisit scatter layout, native WAVG,
  or atomic topology absent contradictory evidence.
- RECOVAR is bit-exact to the rounded iteration-0 model STAR.  Fresh RELION is
  bit-exact to its repeatable, higher-precision in-memory bootstrap noise and
  does not re-read the STAR before iteration 1.  This is a harness/state
  boundary, not a noise-formula bug.  For strict underlying-array comparison,
  restart RELION from the serialized state or capture/feed its full-precision
  state; otherwise allow only the quantified near-tie consequences.
- Audit:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_earliest_score_audit_20260714_214916/case26_paired_raw_operand_audit_11197096_11197128.json`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_earliest_score_audit_20260714_214916/case26_paired_raw_operand_audit_11197096_11197128.md`.

# 2026-07-14: explicit `relion_cuda` full trajectory clears case 20

- H100 job `11197313` completes 11 numbered iterations plus the valid final
  all-data pass in 638.1 science seconds, with 42,339 MiB peak device memory.
  Current sizes and convergence at iteration 11 match RELION exactly.
- Numbered merged/half cross FSC-AUC minima are `0.999988902` and
  `0.999986015`; numbered GT FSC-AUC deltas stay within
  `[-1.89e-6,+1.118e-5]`.  Final merged cross FSC-AUC is `0.997634223`, and
  RECOVAR final GT FSC-AUC is `0.085752642` versus RELION `0.084608151`.
- Particle arrays have small genuine mid-trajectory differences, not only
  ties or serialization noise, but the errors contract by iteration 11 and do
  not perturb schedule, convergence, or FSC quality.
- Slurm exit 2 is a post-science generic-summarizer layout error.  The science
  command exited zero; use the direct FSC-only audit:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_relion_cuda_aeb337df_full_20260714_233032/full_trajectory_fsc_audit.json`.

# 2026-07-14: serialized sigma makes the strict state boundary exact

- H100 job `11198286` uses a fresh RELION run plus `--sigma`; do not use
  `--continue run_it000_optimiser.star`, because RELION disables firstiter-CC
  on continuation.
- Minvsigma2 becomes 1,227/1,227 bit-exact to RECOVAR for both audited
  particles, from 298/1,227 in the retained in-memory run.  Sampling, WTA
  winners, runtime layout, and iteration-0 maps remain exact.
- The residual composite values after exact noise alignment are still about
  `1e-6`, now fully localized to normal float32 FFT/CTF arithmetic.  Iteration-1
  maps change only at non-DC FSC-AUC approximately `1-1.6e-11`.
- The first paired audit's reconstructed phase metric is superseded by
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_earliest_score_audit_20260714_214916/case26_paired_raw_operand_audit_phase_correction_addendum_11197096_11197128.json`;
  actual winning phase coefficients and arguments are bit-exact.
- Machine results:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_serialized_sigma_discriminator_20260714_234311/analysis/serialized_sigma_discriminator.json`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_serialized_sigma_discriminator_20260714_234311/analysis/serialized_sigma_operand_metrics.json`.

# 2026-07-16: RELION-exact CUDA scorer rotations close the K=1 iteration-2 cutoff

> Superseded on 2026-07-17 for accelerated EM: the captured CUDA arithmetic
> was reproduced correctly, but it was attached to the wrong RELION call
> path. See the host-generated scorer-matrix classification below.

- A same-A100, cross-engine operand capture localized the earliest material
  iteration-2 difference to scorer rotation generation.  RELION constructs
  float32 Euler matrices with device `sincosf` and explicitly evaluates the
  perturbation product in float32; the prior RECOVAR path used host float64
  trigonometry followed by a float32 cast.  The authoritative classification
  is `rotation operand generation/precision`, not reduction noise.
- A passive counterfactual using the captured RELION matrices reduced maximum
  projection error by `937.5x` and the critical rank-48/rank-47 score-gap error
  by `757x`.  Its device matrices matched all 36 captured RELION float32 values
  bitwise.  Instrumentation inertness passed; map checks used FSC/FSC-AUC only.
- Production now has a result-only CUDA FFI for RELION's Euler construction and
  optional `A @ R` product.  Scorer matrices use this device result, while the
  separate M-step inverse-matrix path and Euler metadata remain unchanged.
- Same-physical-A100 production job `11249915` completed successfully.  For
  original particle 7881 it changed the ordinary iteration-2 coarse support
  from 48 to RELION's 49 significant parents.  The production capture is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k1_exact_rotation_production_it2_20260716_011500/significance/significance_orig007881_it002_cs056.npz`
  with SHA-256
  `364413a24a8c5f72ae788ed20994ae732b30e71c3a0107cac2b6301e43dab97c`.
- This closes the targeted cutoff boundary, not full-trajectory acceptance.
  The next gate is an uninterrupted K=1 trajectory with per-iteration arrays,
  schedule/convergence, and map FSC/FSC-AUC.  Before scale/performance work,
  batch or cache local per-image scorer-matrix FFI calls and decide whether the
  strict custom-CUDA requirement should remain parity-mode-only.
- Authoritative analysis root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k1_it2_gaussian_cross_engine_analyzer_20260715_225010/analysis_v2_authoritative`.

# 2026-07-16: exact scorer rotations restore K=1 convergence, but residual quality differences remain

- Same-physical-A100 job `11250218`, at commit `d302a760`, completed all 16
  numbered iterations and converged at iteration 16, exactly matching RELION's
  convergence boundary.  The valid post-convergence all-data pass ran once,
  with `RECOVAR_FINAL_ALL_DATA_GRID_CORRECT` unset/off and the forced after-max
  path unset.  Science/external/Slurm wall times were 1,699.2/1,764/1,930
  seconds (the last includes audits), sampled peak device memory was 34,463
  MiB, and all 338 sealed science artifacts re-hash successfully.
- The rotation fix improves all 48 numbered half1/half2/merged FSC-AUC checks
  against the previous trajectory.  Representative half-map changes are iteration 2
  `0.999993119/0.999998046` to `0.999999848/0.999999180`, iteration 7
  `0.992872475/0.993256130` to `0.994870962/0.997492541`, and iteration 9
  `0.987277787/0.987370194` to `0.989869677/0.993149298`.  The corrected run's
  merged FSC-AUC remains above 0.995 through iteration 8, then falls below the
  strict gate at iteration 9 before recovering above 0.995 by iteration 14.
- Grid-off final merged FSC-AUC is `0.989787314`.  Do not causally compare this
  final map to the prior grid-on trajectory; the numbered maps and the frozen
  iteration-2 counterfactual provide the admissible fix evidence.  Final
  minimum non-DC FSC is `0.968028261`.
- The earliest remaining continuous-array failure is iteration-2 Pmax:
  absolute-error p95 `1.1868439569e-4`, maximum `1.4683530460e-2`.  Original
  particle 8240 is the persistent maximum and was almost unchanged by the
  rotation fix; particle 257 is the largest newly exposed regression.  Capture
  and adjudicate their underlying candidate scores before interpreting later
  discrete pose or support differences.
- Current-policy classification is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k1_exact_rotation_fulltraj_d302a760_20260716_012500/analysis/trajectory_classification_current_grid_policy.json`
  (SHA-256 `653d4af535573d3af6bc61f8e0a8745b06aed1c52565a6e0fd6f33cb9b89d107`).
  The older classifier's requirement that final gridding correction be on is
  stale and must not be used for the current GUI-quality policy.  Map gates use
  FSC/FSC-AUC only; no correlation metric is used.

# 2026-07-16: remaining iteration-2 residuals inherit the iteration-1 map

- A same-code, same-physical-A100 two-arm run compared the autonomous state
  with exact RELION iteration-1 reference maps while holding iteration-2
  images, candidate geometry, priors, translations, CTF, and support inputs
  fixed.  Particle 257 changes from 15 to 16 coarse parents because cumulative
  mass at rank 15 moves from `0.9990000254` to `0.9989998528`, only
  `1.73e-7` across the `0.999` cutoff.  Its winner remains exactly `312367`.
  Particle 8240 retains 20 parents and winner `88406`; its change is continuous.
  The map substitution reduces their previously measured Pmax errors by
  `53.18x` and `143.71x`, respectively.  The remaining causal boundary is
  therefore iteration-1 map generation, not iteration-2 cutoff or tie-breaking.
- Against the frozen RELION raw-BPref dump, current commit `d302a760` reduces
  supported data relative L2 by `137.8x/14.7x` and weight relative L2 by
  `5553x/6631x` for halves 1/2 versus the older boundary.  Current weight
  residuals are `9.14e-7/9.06e-7`; data residuals are
  `1.2639e-4/1.0816e-3`, above same-code repeat controls of about `5.7e-8`.
  Iteration-1 map FSC-AUC is `0.999999988768/0.999999954399`.
- Exactly one of 10,000 iteration-1 translations differs by more than
  `1e-4` pixels: original particle 8494 in half 2 selects an adjacent
  x-translation displaced by `0.49999974` pixels.  Capture its complete
  firstiter-CC candidate surface before attributing the larger half-2 data
  residual.  The smaller continuous residual still requires the existing
  canonical BPref contribution replay to separate operand generation,
  geometry, and accumulation precision/order.
- Authoritative paired artifacts are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k1_it2_residual_dualarm_d302a760_20260716_024500/analysis/dualarm_significance_comparison.json`
  (SHA-256 `5a3edcfd251837ddfb0f5ba9d9960296fc680b058d4d07003e8d4936b63c2772`)
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k1_it2_residual_dualarm_d302a760_20260716_024500/analysis/it1_bpref_current_vs_relion.json`
  (SHA-256 `8964832f0b399de806d66526c638e6db39d56dcc43c7646d5abf7b5841fada14`).
  Map quality uses FSC/FSC-AUC only.

# 2026-07-16: the iteration-1 translation outlier is a numerical near tie

- Particle 8494's two competing firstiter-CC coarse translations differ by
  exactly one float32 ULP (`2.98e-8`).  The coarse rotation is identical, and
  replaying the exact RELION fine support makes RECOVAR select the RELION
  winner.  This is numerical tie sensitivity, not a geometry or support-rule
  mismatch.
- Replacing only that translation in the production device-derived BPref
  contribution reduces the half-2 raw-data relative-L2 residual from
  `1.08160e-3` to `1.25343e-4`, an `88.4%` reduction; the
  translation-invariant weight is unaffected.
- The production GPU path reaches 897 pixels while the double-precision C++
  oracle reaches 896.  The sole difference is a radius-boundary pixel whose
  production float32 radius is inside (`r^2=2303.999759`) while the C++ double
  radius is outside (`r^2=2304.000138`).  All 896 common pixels agree.  Removing
  that pixel reduces the apparent panel data residual from `1.35819e-2` to
  `6.21507e-6`, so this closes a diagnostic-oracle mismatch, not a production
  bug.
- Genuine downstream float64/complex128 recomputation from the captured raw
  float32 image changes the final unmasked operand by only `4.39994e-7` in
  relative L2.  The actual RELION CUDA `storeWeightedSums` capture then finds
  exact support, coordinates, all eight indices, and Hermitian flags.  Data
  and weight operands agree at `3.61e-7` and `3.92e-7` relative L2, but RECOVAR
  coefficient relative L2 is `4.7538e-6` versus RELION's `3.5699e-8`
  canonical envelope.
- Cause and fix: RECOVAR added the integer BPref origin before extracting the
  float32 fractions; RELION extracts `floorf`/the fractions first.  Commit
  `65587ea5` matches that arithmetic order.  The patched p8494 replay is
  bitwise exact across all 897 support pixels, coordinates, eight indices, and
  coefficients.  Focused validation passed 4 GPU tests and 39 CPU tests.
- Evidence:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k1_it1_p8494_neartie_capture_d302a760_20260716_024559/analysis/p8494_score_boundary_report_v2/report.json`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k1_it1_p8494_device_capture_fix_fb4e6b73_20260716_082324/analysis/continuous_residual_localization/report.json`.
  Patched validation:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k1_it1_storewavg_device_capture_fb4e6b73_20260716_092618/analysis/patched_boundary_validation.json`
  (SHA-256 `96834e9758fd4e0cbab19415cc1dc36ec89dd9ba77338a5cad11f86ab690e758`).

# 2026-07-16: particle 1491 classifies the recurrent iteration-2 score boundary

- A paired same-physical-A100 capture freezes the earliest recurrent
  iteration-2 boundary.  Every one of RELION's 36,336 coarse rotations maps
  exactly to the transpose of a unique RECOVAR rotation; translation order,
  the coarse winner, and all 30 coarse parent rotations agree.  RELION's
  metadata threshold rank is 173, but its `>=` tie expansion evaluates 174
  hypotheses while RECOVAR evaluates 173.  The sole RELION-only coarse pair
  creates 32 fine descendants with posterior mass `0.004659639`.
- The support difference dominates the visible posterior error.  All-support
  L1 is `9.31920e-3`; on independently renormalized shared support, L1 is
  `2.48768e-4`, relative L2 is `1.88068e-4`, and Pmax error is `5.79469e-5`.
  RELION's full-grid 256-lane replay on the restored support is slightly worse
  than that shared-support production comparison (L1 `3.01462e-4`, Pmax error
  `9.87415e-5`), so fine reduction topology is not the first cause.
- Same-physical-GPU float64 scoring leaves support unchanged and reduces
  centered coarse-score RMS against RELION only from `1.68293e-4` to
  `1.64606e-4` (`2.19%`).  A preliminary complex128-projector-plus-float64
  scoring arm also leaves the boundary unchanged and is much farther from
  RELION; because it ran on another physical GPU, do not use its aggregate
  effects causally without a same-GPU repeat.  These broad controls did not
  classify the boundary and motivated the exact coarse pass-1 capture below.
- The exact coarse contribution capture now resolves this specific pair.
  Reference, weight, and shifted-image operands agree below `7.11e-7`
  relative L2.  The production cross-program candidate residual is
  `-3.62396e-4`, while float64 arithmetic from the captured operands reduces
  it to `+7.663e-6` direct and `+4.456e-6` decomposed.  The matched-prior
  float32 replay envelope `[-2.3079e-4,+1.3542e-4]` spans the tie.  Classify
  particle 1491 as numerical operand/reduction sensitivity, not a formulation
  bug; do not generalize that conclusion to all particles.
- Candidate `7ad2526d` remains unmerged: it does not fix the first cause, its
  earlier paired trajectory had no consistent FSC-AUC improvement, and it cost
  `4.58%` wall time.
- Evidence:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k1_it2_p1491_paired_a92c35ef_20260716_081502/analysis/p1491_coarse_boundary.json`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k1_it2_p1491_paired_a92c35ef_20260716_081502/SCIENCE_COMPLETE.txt`.
  The final replay seal is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/p1491_coarse_operand_replay_20260716_094000/analysis/FINAL_SEAL.json`
  (SHA-256 `a4e559c3bc5a2f378d9f2af37ddb2e5348cf630afe529c2a0e736b742d37b274`).

# 2026-07-16: aggregate iteration-2 differences do not isolate a subgroup

- Across all 10,000 exactly aligned particles, RECOVAR-versus-baseline-RELION
  absolute Pmax error has mean `2.7855e-5`, p95 `8.9897e-5`, p99
  `1.66165e-4`, and maximum `1.33390e-3`; the paired RELION/RELION control is
  `6.2361e-6`, `3.0e-5`, `4.9e-5`, and `6.38e-4`, respectively.
- Signed RECOVAR error is unbiased (mean `1.80e-8`, median `-1.30e-7`, positive
  fraction `0.4963`).  Both halves behave similarly.  Error changes smoothly
  with Pmax, support size, and defocus rather than concentrating in a discrete
  population; support count differs for 43/10,000 particles versus 14/10,000
  in the RELION control.
- This does not classify the remaining difference as reduction noise.  It only
  fails to isolate a systematic subgroup and therefore ends serial
  particle-by-particle debugging unless later aggregate evidence identifies
  one.  Continue with exact coarse operand/formulation replay, controlled
  iteration-boundary substitutions, complete FSC/FSC-AUC trajectories,
  robustness/scale/real-data gates, and then K=4.
- Report:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k1_it2_p1491_paired_a92c35ef_20260716_081502/analysis/aggregate_it2_pmax_support_distribution.json`.

# 2026-07-16: the current K=1 small robustness matrix passes full trajectories

- All eight 3k/128 cases pass the numbered FSC/FSC-AUC audit and terminate on
  the same numbered iteration as RELION (`9--15`).  Minimum numbered merged
  cross-engine FSC-AUC is `0.999838371`; minimum RECOVAR-minus-RELION merged GT
  FSC-AUC is `-0.000279612`.
- With `RECOVAR_FINAL_ALL_DATA_GRID_CORRECT` unset/off, final merged
  cross-engine FSC-AUC is `0.997233874--0.998704958`, while RECOVAR final GT
  FSC-AUC exceeds RELION by `+0.007711695--+0.020127071` in all cases.  This is
  the intended GUI-quality output policy, not a reason to turn grid correction
  on.
- RECOVAR remains `1.41--2.47x` slower.  The severe-outlier case is `2.18x`:
  iteration 2 retains about `100M` hypotheses per half and takes `1285.9`
  seconds, while subsequent iterations contract to tens of seconds.  Preserve
  the completed quality artifact before testing performance changes.
- Evidence:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_robust_current_65d2c3f1_20260716_091500/`
  (case jobs `11264619--11264626`, summary `11264627`, trajectory audit
  `11265312`).

# 2026-07-16: real-10076 approaches the observed repeat scale late

- In 44 of 48 calibrated iteration-1--16 half1/half2/merged comparisons, the
  worse of the two RECOVAR-versus-corresponding-RELION FSC-AUC deficits
  exceeds the single observed RELION-A/B deficit.  Half 1 at iterations
  14--16 and merged iteration 16 are within that empirical control scale.
- Iteration-1 particle 8494 is closed as a one-float32-ULP coarse tie; it does
  not support a fine-scorer bug.  Iteration-2 particle 8240 is different: one
  coarse `0.999`-support swap creates 32 different fine descendants and the
  RELION-only branch holds `0.0777750465` posterior mass.
- Same-GPU complex128/float64 source scoring preserves RECOVAR's p8240
  support.  Frozen factor substitution localizes the score gap to projected
  reference generation; image, CTF/noise, score formulation, reduction order,
  priors, fine reduction, and exact raw rotation matrices are excluded.
  Production-device capture now proves coordinates, coefficients, eight
  corner indices, and conjugation flags bitwise exact.  All 20,064 native
  corner values differ (`1.806e-7` RMS), and the captured projection reproduces
  ordinary RECOVAR scoring bitwise.  A serialized RELION PPref replay through
  the identical RECOVAR staging path is within `2.20e-11` RMS at corners and
  `1.19e-11` RMS after hardware projection.  Staging and interpolation geometry
  are excluded.  The serial p8240 diagnostic stops at the upstream PPref
  grid-value boundary; do not trace its preceding construction input unless
  an aggregate audit identifies a systematic cohort.  Continue with
  distribution-level PPref-grid comparisons and controlled boundary
  substitutions.
- The capture is inert at minimum half/merged FSC-AUC `0.99999997795` on A100
  UUID `GPU-64011c8c-bd98-eb41-2c46-dd201730ef64`.
- Treat the one repeat as an empirical same-model scale, not a confidence
  interval.  Do not gate iteration 17 or final maps against it because the
  control trajectories terminate differently.  Final RELION A/B FSC-AUC is
  `0.946113`; final RECOVAR A/B is `0.948730`.
- Evidence:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_dual_replay_2e40e614_20260716_131000/analysis/real10076_completed_dual_repeat_envelope_v1.json`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it2_p8240_capture_505af690_20260716_124319/distribution_substitution_v1/FINAL_SEAL.txt`
  (manifest SHA-256
  `b92c370d690a77ed1d03be0cb7727d69d0339cfc1dffda3edf5de76e2c50a76f`).

# 2026-07-16: authoritative incoming-reference K=4 discriminators

- Hardened same-A100 job `11273615` changes only the incoming case-8
  iteration-4 reference.  Class-2 iteration-5 direct FSC-AUC rises from
  `0.978234446` to `0.999999996`; the substituted arm's minimum over all
  classes is `0.999999946`.  The same UUID
  `GPU-a1de512c-f178-a5e1-6c95-c54c6d07c9f3` is recorded at all boundaries.
- The substituted arm has exact class agreement and no iteration-5 support
  count differences.  The visible cliff is inherited map-state amplification,
  not an iteration-5 E/M formulation error.  The earlier source remains open.
- In case 2, hardened job `11273364` closes iteration 2--5 when each previous
  RELION reference is supplied: iteration-5 minimum direct FSC-AUC is
  `0.999999974`, class agreement is `1.0`, and the minimum GT FSC-AUC delta is
  `-6.57e-7`.  Focus further work on the earlier reconstructed-reference
  boundary, not later particle scoring.
- A float64 M-step perturbation worsens case-8 class 2 to `0.658960344`; this
  demonstrates numerical sensitivity but is not canonical cross-program
  float64 closure.
- Evidence:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_case8_it5_relref_ab_uuidfix_03c0969b_20260716_131422/analysis/ab_summary.json`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_case2_relref_ab_03c0969b_hardened_20260716_131700/analysis/ab_summary.json`.

# 2026-07-16: late K=4 case-2/case-8 gaps are inside stock RELION instability

- Same-A100 stock RELION repeats are effectively identical through iteration
  2, then become chaotic.  Case 8 reaches matched-class repeat FSC-AUC minima
  `0.756449`, `0.377412`, and `0.223965` at iterations 3--5; iteration-5 class
  agreement is `0.8880`.  Case 2 reaches `0.900719`, `0.893286`, and `0.863441`.
- RECOVAR/RELION is substantially closer than RELION/RELION at the visible
  late boundaries: case-8 minority class is `0.978234`; case-2 iteration-4 is
  `0.991908` with agreement `0.9914`, versus native-repeat class values
  `0.893286--0.904928` and agreement `0.9019`.
- Dispatch schedule non-rank columns are exact, while runtime follower owners
  differ for `25,468/50,000` case-8 and `19,712/50,000` case-2 rows.  Treat
  previous-reference substitution as a sensitivity control, not proof of a
  reconstruction bug.  Do not pursue those late cliffs unless a difference
  exceeds the case-specific stock-repeat envelope; keep early stable arrays,
  convergence, and FSC/FSC-AUC trajectory gates.
- Evidence:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_case8_relion_repeat_full5_uuidfix_20260716_141807/analysis/relion_repeat_full5.json`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_case2_relion_repeat_full5_uuidfix_20260716_144006/analysis/relion_repeat_full5.json`.

# 2026-07-16: K=1 local runs recorded the wrong pass's significant count

- RELION's `_rlnNrOfSignificantSamples` is the first/coarse-pass count.
  RECOVAR serialized the fine M-step support count after local search began,
  causing 2,691/3,000 count differences at iteration 4 and a mean absolute
  error of `75.834` by iteration 10.
- Full trajectory job `11275201` closes the parent-pass serialization bug:
  four iterations match all 3,000 counts, and only 16 particle-iteration
  residuals remain elsewhere, each exactly one count.  Iteration 4 drops from
  2,691 mismatches to 2; iteration-10 mean absolute error drops from
  `75.8343` to `0.002333`.
- Production now counts the explicit retained parent-pass index lists for
  serialization and the approximate-accuracy diagnostic.  It does not expose
  the fine engine count as RELION metadata, and non-adaptive local search
  reports this field unavailable.  Fine support still controls all M-step
  arrays and statistics; the change is not a map-quality intervention.
- Targeted guards pass, including valid local-GPU composite coverage of 363
  tests.  Replay the exact extraction refinement as a final runtime guard.
- The remaining 16 targets are upstream coarse-support boundaries, not a
  serialization or fine-M-step-support error.  They are materialized for the
  next direct-score replay.
- Evidence:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_sigcount_fulltraj_52178ed3_20260716_135924/analysis/REPORT_v2.md`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_sigcount_fulltraj_52178ed3_20260716_135924/analysis/residual_coarse_count_particles_v2.tsv`.
  The manifest SHA-256 is
  `190ace970540dd6ad2eb7a35627a2d8e66345fba1f0a3cf536df0c77ba3ee803`.

# 2026-07-16: K=4 case 11 has a stable first-map boundary and unstable late trajectory

- Hardened same-A100 case 12 (30k Tomotwin, white noise, uniform classes) passes
  in job `11274946`.  Iteration direct FSC-AUC minima are `0.999820693`,
  `0.999157434`, and `0.996495854`; minimum assignment agreement is `0.9986`,
  and worst GT FSC-AUC delta is `-6.28e-5`.
- Case 11 (10k IgG, white noise, uniform classes, 20% outliers) diverges by
  iteration 3: class-2 direct FSC-AUC is `0.9735133506`.  Its earliest
  nonordinary distribution boundary is iteration 2, with 9,999 Pmax
  differences, 1,166 support differences, and 24 class mismatches.
- Exact incoming-reference A/Bs localize this causally.  Supplying RELION's
  iteration-2 maps only for scoring iteration 3 raises the minimum direct
  FSC-AUC from `0.973512763` to `0.999999632`.  Supplying RELION's iteration-1
  maps only for scoring iteration 2 raises iteration-2 minimum direct FSC-AUC
  from `0.998533895` to `0.999999956` and prevents the iteration-3 cliff
  (`0.999924228`, class agreement `1.0`).  The broad drift begins at the
  iteration-1 reconstructed-reference boundary, not later scoring.
- Particle 7915 is the sole iteration-1 label mismatch and is an exact
  one-float32-ULP score tie at the same pose.  This closes the discrete label
  decision as numerical but does not explain every aggregate map operand.
- Same-A100 stock RELION job `11277907` proves the first map boundary exceeds
  native variation: iteration-2 repeat minimum FSC-AUC is `0.9999999975`, with
  exact class agreement and support sizes, versus RECOVAR/RELION
  `0.998533895`.  This is a real stable reconstructed-reference parity target,
  but it is not yet classified as an algorithm bug versus numerical
  representation/reduction sensitivity.
  RELION itself bifurcates at iteration 3 (minimum FSC-AUC `0.725582397`, class
  agreement `0.9719`), so the RECOVAR/RELION iteration-3 cliff is inside the
  native nonlinear envelope and is not a separate defect.
- Freeze the iteration-1 reconstruction boundary and compare production order,
  canonical order, and recomputed float64/complex128 operands.  Continue with
  aggregate arrays and FSC/FSC-AUC; do not resume serial particle debugging.
- Evidence:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_case11_it3_relref_ab_b5dd574a_20260716_141000/analysis/ab_summary.json`,
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_case11_it2_relref_ab_b5dd574a_20260716_143300/analysis/ab_summary.json`,
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_case11_relion_repeat_full3_hardened_20260716_145000/analysis/relion_repeat_full3.json`,
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_case11_p7915_rec_capture_b5dd574a_20260716_142900/analysis.json`.

# 2026-07-16: K=4 tied cutoff counts are metadata-only

- RELION writes the pre-tie cutoff rank while retaining all threshold ties in
  pass 2.  RECOVAR wrote the expanded support cardinality.
- K=4 significance helpers now return the existing cutoff rank only when
  requested.  Metadata uses that rank; masks, posterior weights, and Ft_y/Ft_CTF
  remain unchanged.  Default helper tuple arities and firstiter count `1` are
  preserved.  The integrated affected-module run passes 74 tests.

# 2026-07-16: corrected K=1 parent-count replay passes its A/B/A control

- Same-A100 job `11278391` ran parent-count control A1, exact extracted-count B,
  and parent-count control A2 for ten iterations on the same physical GPU.  The
  exact source semantics and serialized count aliases match after applying the
  recorded half-order-to-image-order permutation; deterministic state is exact.
- The original post-hoc analyzer compared half-order counts directly with
  image-order metadata and therefore failed for an ordering artifact.  Preserve
  that failure, but use the sealed corrected report.  It classifies the count
  fix as inside the same-GPU parent-repeat envelope: total RELION support-count
  residuals are `20/15/15` for A1/B/A2 and summed Pmax MAE is
  `0.00200546/0.00189107/0.00189499`.
- Over `136,688,904` accumulator values, A1-versus-A2 RMS is `8.4373e-5`, while
  B-versus-A2 RMS is `5.5372e-6`.  This is an aggregate state guard for the
  metadata fix, not a general quality claim; map statements remain gated by
  FSC/FSC-AUC.
- Evidence:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_sigcount_parent_exact_parent_aba_52178ed3_8fd704ab_20260716_152025/analysis/parent_exact_parent_same_gpu_aba.json`
  (SHA-256
  `9acd943ef0d6ba8cd11fdaba982c48fcd6fb1b58bb5652c80e3e29c7f0d3580d`),
  with analyzer seal
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_sigcount_parent_exact_parent_aba_52178ed3_8fd704ab_20260716_152025/provenance/analyzer_revision_v3.json`
  (SHA-256
  `720f3551a7d3e18df92fda8dfefb7b00a72e402e3cbfa23ceb67231a9e259b3e`).

# 2026-07-16: exact fine-Gaussian arithmetic is the dominant case-22 cost

- Same-A100 exact/algebraic/exact job `11274919` completed all three arms on
  UUID `GPU-5e619c2e-82b4-ff79-cbcb-ab29514a9f30`.  Exact A/C environments
  are identical; algebraic B adds only
  `RECOVAR_DISABLE_RELION_EXACT_FINE_GAUSSIAN=1`.  Source, native-library, and
  input manifests reverify.
- Iteration-2 wall is `3401.041/1265.114/3398.249` seconds for A/B/C.  Relative
  to the exact-control mean, algebraic is `2.687x` faster; external wall is
  `2.575x` faster and the aggregate two-half sparse M-step is `2.728x` faster.
  Exact C/A iteration-2 timing ratio is `0.999179`, so this penalty is stable.
- Exact support equality is not a valid gate: iteration-2 by-image mismatches
  are `27/33/25` for A-B/A-C/B-C, and the B-versus-best-exact support RMS is
  smaller than the native A-C repeat.  All 88 comparable bucket identities
  happen to match and only those identities enter matched-bucket timing.
- This is performance evidence only.  Although all support arrays are inside
  the exact-repeat RMS envelope, only 9/13 continuous-state and 2/8 accumulator
  arrays are.  No maps were loaded and no FSC/FSC-AUC was computed; a separate
  FSC/FSC-AUC quality experiment is required before enabling algebraic scoring.
- Evidence:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_exact_algebraic_exact_aba_fc70abc3_retry_dedicated_cuda_20260716_135000/analysis/aba_performance_analysis_v2.json`
  (SHA-256
  `eff0f31da149369a3fd50bff16751ef2c229e9ed37407a0e987fca51ab787016`),
  with sealed provenance in
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_exact_algebraic_exact_aba_fc70abc3_retry_dedicated_cuda_20260716_135000/analysis/aba_performance_analysis_v2.seal.json`
  (SHA-256
  `57621195df54a846875273d9ba6337f797c3ea9e91aeda53cc1976cccebbd24c`).

# 2026-07-16: K=4 float64 source/geometry sensitivity is validated but not causal

- The frozen case-11 iteration-1 RECOVAR capture is inert by FSC/FSC-AUC: its
  per-class capture/control map FSC-AUC is
  `0.9999999748/0.9999999925/0.9999999925/0.9999999782`.  Exact hard and coarse
  assignments are unchanged.  This does not close the cross-engine boundary;
  class-2 RECOVAR-versus-RELION direct FSC-AUC remains `0.999330787`, far
  outside the stock RELION repeat value `0.9999999975`.
- Commit `a8b8bd995f941f81a9d65e09c36b913ef06c13ce` adds a fail-closed production
  float32 source control before any high-precision interpretation.  Across 12
  capture shards, all 72 active, signature, and control-repeat array
  comparisons are bitwise equal.  Thus the frozen raw images, normalization,
  integer shifts, cuFFT/JAX CTF and phase path, posterior reduction, and source
  row interpretation reproduce the captured complex64/float32 operands.
- With that control closed, recomputed float64/complex128 geometry changes
  `21,885` support decisions and `32` target indices over `50,195,816`
  contributions.  Per-class target mismatches are `16/0/0/16`; support
  mismatches are `9,425/1,881/1,730/8,849`; Hermitian-fold flags never differ.
  Genuine float64 source relative-L1 differences are approximately
  `1.94e-7--1.97e-7` for data and `1.58e-7--1.61e-7` for weights.
- This validates a real precision-sensitive source/geometry boundary, but it
  does not make it the cause of the class-2 map gap.  The capture does not
  contain RELION's complete contribution list or RECOVAR's production atomic
  schedule.  Next capture RELION's full 847-particle iteration-1 class-2
  pre-scatter boundary and compare both engines in a common deterministic
  float64 geometry/reduction replay.  Do not resume serial particle tracing.
- Sealed summary:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k4_case11_it1_frozen_boundary_20260716/capture_jobs/joblocal_a100gpu1_b1ce7242_20260716_1620/analysis_followup_7877b2b3_source_control/all_class_summary.json`
  (SHA-256
  `9d78b69479bd541e544409e24854fd18892d25bf0fd47cc7039e5d196f8fe66f`).
  The manifest is `artifacts.sha256` (SHA-256
  `c3b012271a1fad632386acbb231a0e845a0ae0bc3f2400c3bc232086d0b672f9`)
  and passes `sha256sum -c`.

# 2026-07-16: K=4 particle 7915 routing causally closes the first-map boundary

- A sealed three-arm intervention ran stock RELION, RECOVAR control, and
  RECOVAR with only original zero-based particle 7915 forced from class 1 to
  class 0, serially on the same A100.  RELION and the override have identical
  class counts `[4293,846,797,4064]` and exact membership for all 10,000
  particles.  Control differs from RELION only at particle 7915.
- The intervention raises zero-based class-0 RELION/RECOVAR FSC-AUC from
  `0.9998860754` to `0.9999999661`, removing `99.970257%` of its FSC-AUC
  defect.  Class 1 rises from `0.9993307839` to `0.9999999884`, removing
  `99.998274%`.  Classes 2 and 3 remain at the native numerical floor, with
  defect ratios `0.997275` and `1.003540`.  Map conclusions use shellwise FSC
  and FSC-AUC; correlation is not a gate.
- The analyzer's initial assumption that the two unaffected classes must be
  bitwise identical was rejected.  GPU atomic repeat/order variation makes
  that too strict: affected/unaffected accumulator residual ratios are at
  least `121,698x` for `Ft_y` and `45,685x` for `Ft_ctf`.  These measured-floor
  ratios classify this intervention; the descriptive `100x` fail-close is not
  a general program tolerance or confidence interval.
- Therefore the sole recurrent first-iteration class-routing decision is
  causally sufficient for essentially the entire affected first-map gap.  No
  backprojection-kernel defect is supported at this boundary.  RELION raw
  accumulators were not captured, so this conclusion is map-level causal
  closure, not a cross-engine claim of bitwise accumulator equality.
- Continue with the integrated offset-free all-particle winner summary to
  classify why the score boundary occurs.  If a smaller replay is still
  needed, select an aggregate near-boundary subgroup and use canonical
  float32 plus recomputed float64/complex128 controls; do not resume serial
  particle-by-particle debugging.
- Sealed evidence:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k4_particle7915_causal_mstep_a100_20260716_184912/analysis/causal_mstep_report_v1.json`
  (SHA-256
  `564fa793a617303556432fd2f60157d0c208d69473e965d2368b2e4f4062fccd`),
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k4_particle7915_causal_mstep_a100_20260716_184912/analysis/causal_mstep_shellwise_fsc_v1.npz`
  (SHA-256
  `945dd9b3cc59025fcf766ab88747c1d158ca160dcd3f866a930baa13c0c5e0bb`),
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k4_particle7915_causal_mstep_a100_20260716_184912/provenance/completion_seal_v1.json`
  (SHA-256
  `cd3970196647ea6ae59a996883ed7b553b1dc12b660a0529789d4f35b396b8f1`).
# 2026-07-16: bounded exact raw-diff2 reuse is parity-safe at the frozen boundary

- The sealed case-22 A1 iteration-2 aggregate replay compares cache OFF twice,
  cache ON, and reversed-input cache OFF. After canonical particle ordering,
  saved score/log-evidence/log-Z/Pmax arrays, support counts, assignments, best
  rotations/translations, and rotation-posterior sums are bitwise identical.
- All six merged-map pairings have complex128-input normalized FSC-AUC
  `0.999999999997208--0.999999999997247`, or defect
  `2.753e-12--2.792e-12`. Cache ON is no worse than same-GPU OFF/OFF repeat
  and input-order controls; the residual accumulator/map variation is GPU
  reduction noise. Correlation is not an acceptance metric.
- Cache ON is `9.7%` faster than the mean same-order OFF controls (`75.07` vs
  `83.10` seconds) only on this bounded high-support probe. Do not quote it as
  a full-iteration or full-run speedup.
- Production commit `7e48bcd85f735548f4d39ba1d5cc856581d5d8a2` admits the cache only within
  512 MiB, 1% physical-device, 25% physical-free, and 25% JAX-allocator-free
  caps. Any unavailable/nonpositive memory observation fails closed; set
  `RECOVAR_SPARSE_PASS2_EXACT_RAW_DIFF2_CACHE_MAX_BYTES=0` to disable it.
- Frozen source was `d4bc78fbfe976287ad86af507aaa9ec4ae8ab71e` on GPU
  `GPU-dc6576aa-e1e4-6055-4a5e-d0fa809f3983`. The pinned CUDA library is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_raw_diff2_reuse_aba_hardened_frozen_20260716_153618/native/optimized/libcuda_backproject.so`
  (`206c3d486d738b9c40a872cc47cee2be34499559dfd59fd4e0a5bea414c12ae3`),
  and the RELION binding is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_raw_diff2_reuse_aba_hardened_frozen_20260716_153618/relion_bind_build/shared/_relion_bind_core.cpython-311-x86_64-linux-gnu.so`
  (`1e9f0cf04f254e00abb5f742b74ed09c50c0a13ee883fe628613320e2fd755b6`).
- Authoritative artifacts are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_raw_diff2_boundary_probe_20260716_184500/artifacts/FINAL_REPORT_v2.md`
  (`6c74453da31015d5b109c3d6e750063a455a46ba94af6137307409e91daaf537`),
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_raw_diff2_boundary_probe_20260716_184500/artifacts/analysis_v2.json`
  (`62abe127d20fb28b11ef1a3dd66757856c6a18f0dd5dbcc57170a0cd6b98fb5a`),
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_raw_diff2_boundary_probe_20260716_184500/boundary/manifest.json`
  (`7b062d5d8126f74fb9d8969b39791cf7e9a5ce4dfb97be02297a5c6f50c0d320`),
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_raw_diff2_boundary_probe_20260716_184500/provenance/artifact_manifest.sha256`
  (`7737332f72c5ae8981b2fbf73222105cff6a323d514fe199d93d1f1e2504894b`),
  and `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_raw_diff2_boundary_probe_20260716_184500/SEALED_v2.txt`
  (`dc164525bf61fd4d7a9a915a4fe2f13631503252dc7b18ff561429354b62c779`).
  The allocator guard probe is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_raw_diff2_allocator_probe_7e48bcd8_20260716_164400/allocator_probe.log`
  (`19c74ab7657b560e06fccc5373beded700a1e5e54a04ec3e6c308b14eba759f8`).
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_raw_diff2_boundary_probe_20260716_184500/artifacts/analysis_v1.json`
  (`65dc3de3e6c3b47fdc3b31d7b8a47aa1ed03c019ccd5bfbb393af9707047b302`)
  is superseded because it integrated float32 FSC values above one; its mode
  arrays were overwritten by the independent v2 repeat and are not sealed.

# 2026-07-16: current-head K=1 robustness and 100k replay trajectories pass map-quality gates

- The paired same-GPU replay-controlled matrix at detached commit `f1c83011`
  completed cases 15 (20% outliers), 21 (Kent angles), 23 (no-CTF radial
  noise), and 32 (10k Kent/radial). All numbered schedules, convergence
  iterations, and final-all-data topology match exactly.
- Across 47 numbered boundaries, the minimum half-or-merged cross-engine
  FSC-AUC is `0.9998758717` and the worst RECOVAR-minus-RELION merged GT
  FSC-AUC is `-7.0143e-5`. All four final RECOVAR GT FSC-AUC values exceed
  RELION by `+0.003859` to `+0.019029`. Final cross-engine FSC-AUC values near
  `0.9983` retain the intentional grid-off GUI output policy and are not a
  numbered-trajectory failure. Correlation is not an acceptance metric.
- The corrected aggregate state auditor also passes exact schedule,
  convergence, and finalization for all four cases. It identifies recurring
  low/intermediate-Pmax and support-quantile cohorts, without a persistent
  half-set or defocus pattern. Preserve those aggregate cohorts; do not resume
  serial particle tracing.
- The 100k/256 replay-controlled run `11268911` converged with RELION at
  iteration 14 and passed all numbered FSC gates: minimum merged cross-engine
  FSC-AUC `0.999960405`, minimum half-or-merged `0.999918721`, and worst GT
  FSC-AUC delta `-3.0788e-5`. The final RECOVAR GT FSC-AUC exceeds RELION by
  `+0.0080925`.
- In the same A100 allocation, recorded external wall time is `8094` seconds
  for RECOVAR and `30745` seconds for RELION (`3.798x` speed ratio), with peak
  monitor values `34597 MiB` and `80053 MiB`, respectively. This is scale and
  performance evidence for the replay-controlled path, not autonomous parity.
- The historical 100k artifact contains per-particle support counts only for
  iterations 1--5. Pmax, pose, translation, maps, convergence, and final state
  remain available through iteration 14. Audits must report late support as
  not measured rather than infer it. This run predates commit `8fd704ab`, and
  current-head runs serialize the corrected parent-pass counts. The complete
  Pmax/pose audit finds a broad mid-trajectory Pmax residual (iteration-10
  median `0.003039`, p95 `0.01480`, p99 `0.02850`), while measured support
  differences through iteration 5 are sparse one-count tails and pose/shift
  deviations have tiny p95/p99 values with isolated large maxima.
- Evidence:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_robust_subset_f1c83011_20260716_223000/trajectory_matrix_summary.json`
  (`7bdc4e41a779509298557e41ae869836c5331091048d6d87a1324691c3d0b20f`),
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_robust_subset_f1c83011_20260716_223000/aggregate_state_audit_20260717/matrix_summary.json`
  (`18d91f23dfdc58851d2d36e5d9bbad423eb8da70c2a4d22b85437115a01cd335`),
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_scale100k_505af690_20260716_112000/cases/1_baseline_100k_g256_white_noise1_bf80/trajectory_analysis/k1_scale_acceptance.json`
  (`2c0c4de857b509ffcc56fb4caea7ea263775a549710fa3dbe58cadc5974923be`),
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_scale100k_505af690_20260716_112000/aggregate_state_audit_100k_20260717/full_pmax_pose_all14_support_available5/report.json`
  (`6603cabc389ea6a72d660fabb7c69119f2997fe4e09defa105e81017864fed6f`).

# 2026-07-16: case-21 posterior residual exceeds the same-A100 RELION repeat envelope

- Same-allocation RELION-A/RELION-B/RECOVAR job `11290745` closes the repeat
  control for the recurrent case-21 state boundary. The RELION repeat's minimum
  numbered merged FSC-AUC is `0.999999999934`.
- At iteration 5, RECOVAR-versus-RELION-A mean absolute Pmax error is
  `0.00495290`, versus `7.899e-6` for RELION-B versus RELION-A (`627x`); `79.73%`
  of RECOVAR rows exceed the RELION repeat maximum. At iteration 6 the values
  are `0.00541547` versus `1.32467e-5` (`408.8x`), with `97.97%` exceeding the
  repeat maximum. Significant-support counts differ for `31/90` RECOVAR rows
  at iterations 5/6 and `0/0` RELION-repeat rows.
- Across iterations 2--11, mean absolute Pmax ratios are `34x--627x`. This is a
  systematic state residual beyond native same-GPU repeat variation, even
  though maps, convergence, and final GT FSC-AUC pass strongly.
- Do not yet label the residual an algorithm bug. Freeze one aggregate
  iteration-5/6 boundary over the flagged Pmax/support cohorts and compare
  production-order float32, canonical-order float32, captured-cast float64,
  and genuinely recomputed float64/complex128 operands. Classify operand
  generation, geometry, ordering, or precision before changing production
  math. Do not use bitwise discrete decisions or particle-by-particle tracing
  as gates.
- Evidence:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case21_repeat_control_32ac19dc_20260716_221646/cases/21_small_kent_angles_3k_g128_white_noise3_bf80/repeat_control_analysis/repeat_control_summary.json`
  (`60ecf20439fa87a7964cd09903294d288cbfc563168003b26d79c8c4e5ac652c`).

# 2026-07-16: robustness launchers now record the cgroup-visible physical GPU

- Numeric `SLURM_JOB_GPUS` values name host-physical indices, while
  `nvidia-smi` is commonly remapped to visible index zero inside a Slurm GPU
  cgroup. The K=1 and K-class launchers incorrectly queried the numeric host
  index in the remapped namespace and could accept `No devices were found` as
  a UUID.
- Commits `565f4363` and `b42c185a` instead require exactly one visible
  `GPU-*` UUID and cross-check Slurm only when Slurm itself supplies a UUID.
  Real allocations with `SLURM_JOB_GPUS=1` and `2` now record valid physical
  UUIDs. The invalid pre-science jobs were cancelled and preserved as
  superseded infrastructure evidence.

# 2026-07-16: anisotropic-outlier, high-resolution, and low-noise K=1 gates pass

- Same-GPU science jobs `11291242`, `11291425`, and `11291427`, followed by
  fail-closed audit jobs `11292006`, `11292008`, and `11292010`, cover 25%
  anisotropic outliers, grid-256 high-resolution data without B-factor
  attenuation, and low-noise data. All three FSC/FSC-AUC and aggregate-state
  audits pass.
- RECOVAR and RELION converge at the same boundary and run the valid final
  all-data step: iteration 11 for the anisotropic-outlier and high-resolution
  cases, and iteration 12 for the low-noise case. Numbered topology and image
  identity alignment are complete.
- Minimum numbered merged cross-engine FSC-AUC is respectively
  `0.9999999973`, `0.9999806461`, and `0.9999969065`. Worst numbered
  RECOVAR-minus-RELION merged GT FSC-AUC is `-9.45e-7`, `-1.63e-6`, and
  `-3.89e-6`. Final merged cross-engine FSC-AUC is `0.998032399`,
  `0.997737292`, and `0.998869473`, while final RECOVAR GT FSC-AUC exceeds
  RELION by `+0.008378607`, `+0.001147312`, and `+0.014986853`. Map quality is
  assessed only with shellwise FSC/FSC-AUC; correlation is not computed.
- Evidence:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_robust_expansion_audit_cb83d1b9_20260717_031000/cases/16_small_anisotropic_outliers_3k_g128_pct25_noise3_bf80/k1_fsc_trajectory.json`
  (`a7d252923c10af7c7504ac37b15334a2141b577942fa893f25c2de0a024c07a9`),
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_robust_expansion_audit_cb83d1b9_20260717_031000/cases/20_small_high_res_radial_3k_g256_noise3_bf0/k1_fsc_trajectory.json`
  (`d869b0b3cdad9f7525da2d2cd3960c63431cb7f1b5e98915c75e6ae67706f7bb`),
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_robust_expansion_audit_cb83d1b9_20260717_031000/cases/29_small_low_noise_3k_g128_white_noise0p2_bf80/k1_fsc_trajectory.json`
  (`1a60e0ae0b7be85a017cddca99cccfb5a8b121eff54311a1cc433e9abb8deaa5`).

# 2026-07-16: HIGHEST BPref precision passes the 100k trajectory and final gates

- The corrected replay-controlled 100k/256 run at commit `94b8f2b2` completed
  as science job `11288959`, followed by FSC/FSC-AUC audit `11288960` and
  complete intermediate-topology audit `11291973`. RECOVAR and RELION converge
  exactly at iteration 14, and RECOVAR runs the valid converged final all-data
  step at Nyquist with grid correction unset/off.
- Across 14 numbered boundaries, the minimum merged cross-engine FSC-AUC is
  `0.9999621181` and the worst RECOVAR-minus-RELION merged GT FSC-AUC is
  `-2.81585e-5`. Final merged cross-engine FSC-AUC is `0.9986929591`;
  RECOVAR-versus-GT is `0.5444889741`, RELION-versus-GT is `0.5363817908`, and
  the delta is `+0.0081071833`. Map acceptance uses only shellwise FSC and
  FSC-AUC.
- The complete 14-boundary intermediate audit passes required-artifact,
  finite-array, shape, and topology checks. Its continuous residual magnitudes
  remain diagnostic rather than tolerance-gated; the HIGHEST-precision change
  is not evidence that posterior-state differences are all resolved.
- Evidence:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_scale100k_highest_94b8f2b2_20260716T211300Z/analysis/k1_fsc_trajectory.json`
  (SHA-256
  `c2c05e5ffc89805900eecbedd4f2d86c6335b23d3edc9efd77ed974d3a889669`)
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_scale100k_highest_94b8f2b2_20260716T211300Z/analysis/intermediate_trajectory_beeb8230_after11288959/k1_intermediate_trajectory.json`
  (SHA-256
  `bd75044371aaa424da97301444cdafa5b54df9006ec7830ff3ab82ea36e7b6ec`).

# 2026-07-16: low-contrast/noise-scale K=1 robustness gate passes

- Same-GPU science job `11291424` and audits `11292007` cover the
  3k-particle grid-128 low-contrast/noise-scale case. RECOVAR and RELION
  converge exactly at iteration 16 and both take the converged final all-data
  branch; FSC/FSC-AUC and aggregate particle-state gates pass.
- The minimum numbered merged cross-engine FSC-AUC is `0.9999629449`, and the
  worst numbered RECOVAR-minus-RELION merged GT FSC-AUC is `-4.87703e-6`.
  Final merged cross-engine FSC-AUC is `0.9987484518`, while RECOVAR final GT
  FSC-AUC exceeds RELION by `+0.0146267707`. Correlation is not computed or
  used as an acceptance metric.
- Evidence:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_robust_expansion_audit_cb83d1b9_20260717_031000/cases/18_small_contrast_noise_scale_3k_g128_noise1_bf80/k1_fsc_trajectory.json`
  (SHA-256
  `344d00edfa1dd11395ec665491461ca7ae6ae05404e4636b5fed46edc41cf1a8`)
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_robust_expansion_audit_cb83d1b9_20260717_031000/cases/18_small_contrast_noise_scale_3k_g128_noise1_bf80/particle_state_distribution.json`
  (SHA-256
  `f8faee082bfb96ef0a9917cce1004e2da9d5d6d1db4cb660880adc195cddf082`).
# 2026-07-16: aggregate case-21 precision controls bound the score residual

- Corrected CPU audit job `11292961` verifies exact candidate geometry in all
  186 frozen iteration-5/6 captures. Float32 candidate-order replay changes
  Pmax by at most `1.19209e-7`; float64 replay removes that order effect to the
  floating-point floor.
- Same-boundary float64 scoring changes Pmax by about `1.1e-5--1.3e-5` on
  average and at most `6.54459e-5`, with zero changes to either saved coarse
  pass-1 support or captured fine pass-2 reconstruction support. This is far
  below the aggregate RECOVAR/RELION mean Pmax residual of about
  `0.00495--0.00542`.
- This rules out ordinary scoring precision and candidate reduction order as
  the dominant aggregate cause. Upstream per-pixel operand-generation
  precision remains unresolved because the iteration-5 capture widens already
  narrowed operands and the iteration-6 capture narrows operands and omits
  complete pixel identities. Do not resume serial particle tracing without a
  systematic aggregate subgroup.
- Evidence:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case21_it56_precision_32ac19dc_20260716_225200/analysis/precision_score_order_analysis_v1.json`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case21_it56_precision_32ac19dc_20260716_225200/analysis/precision_state_map_analysis_v1.json`.
  Intermediate arrays use exact/array metrics; map conclusions use FSC and
  FSC-AUC, never correlation.

# 2026-07-17: real-10076 current-size difference is upstream of saved BPref aggregates

- Autonomous job `11291277` matches the fresh RELION current-size and search
  schedule through numbered iteration 15. At that boundary RECOVAR shell 33
  FSC is `0.5011810064`, selecting resolution shell 33 and iteration-16 size
  130; RELION shell 33 FSC is `0.495847`, selecting shell 32 and size 128. Both
  take the same high-FSC/Pmax aggressive-growth branch. The increment-state
  difference is noncausal, so do not patch the current-size formula.
- The saved-aggregate precision auditor reproduces the production FSC curve
  bitwise, then independently varies aggregate-to-native and FSC-shell
  reductions over float32/float64 and canonical/reverse orders. Its complete
  shell-33 control range is `7.7486e-7`; every control remains above `0.5` by
  at least `0.00118053`. The nearest same-GPU RELION control is `0.00299301`
  away, `6276.8x` the maximum downstream deviation from the saved result.
- Classification is `unresolved_upstream_of_saved_aggregate`. The failed run
  has no complete per-contribution operands, device geometry/signatures,
  genuine complex128/float64 recomputation inputs, or RELION raw accumulator,
  so operand generation, geometry, and GPU atomic ordering remain open. A
  genuine frozen iteration-15 capture is required before changing production
  backprojection math.
- The old run subsequently exposed a separate final-all-data cold-start replay
  bug. Commit `719ad930` is the causal repair and its focused unit tests pass;
  production validation job `11292212` is running independently.
- Evidence:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it015_saved_aggregate_v2final_20260717T040109Z/shell33_saved_aggregate_audit.json`
  (SHA-256
  `1c465954ee152b4bd2e5aa4d57ddc2449c5e488ee83b43c13d29076dd8d25a03`)
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it15_shell33_fsc_precision_20260717T040500Z/EVIDENCE.md`
  (SHA-256
  `31827b2287952b293c888d3f8edca251d2c26b6213fa60bdc3efb9f14ce9480b`).

# 2026-07-17: severe-outlier K=1 robustness gate passes

- Same-GPU science job `11291426` and audit job `11292009` cover the
  3k-particle radial-noise-5 severe-outlier case. RECOVAR and RELION have the
  exact numbered schedule, converge at iteration 11, and complete the valid
  converged final-all-data step. FSC/FSC-AUC and aggregate-state audits pass.
- Minimum numbered half-or-merged cross-engine FSC-AUC is `0.9996428687`;
  worst numbered merged RECOVAR-minus-RELION GT FSC-AUC is `-0.0003160482`.
  Final merged cross-engine FSC-AUC is `0.9975098418`, and RECOVAR final GT
  FSC-AUC exceeds RELION by `+0.0098185505`. Correlation is not computed.
- Evidence:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_robust_expansion_audit_cb83d1b9_20260717_031000/cases/22_small_severe_outliers_3k_g128_radial_noise5_bf80/k1_fsc_trajectory.json`
  (SHA-256
  `55d5913dfcd278b59937ded628c7b70df8d61c34d7eb91ce2044fa73711eeaca`)
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_robust_expansion_audit_cb83d1b9_20260717_031000/cases/22_small_severe_outliers_3k_g128_radial_noise5_bf80/particle_state_distribution.json`
  (SHA-256
  `cdc7a53b21c2c47b382184562ed152973007bcb6c40e92b7c93123d0ae40e302`).

# 2026-07-17: real-10076 iteration-2 half-2 anomaly was replay noise semantics

- RELION MPI initializes and broadcasts follower rank 1's half-1 noise once at
  process start, then uninterrupted numbered maximizations update the two
  follower-local half spectra independently. With `--firstiter_cc`, iteration
  2 is the first boundary at which those saved half spectra differ.
- Same-A100 sequential arms from the numbered iteration-2 state isolate the
  effect. Restart-faithful half-1 broadcast gives half-2 map FSC-AUC
  `0.9989539270` and mean absolute Pmax error `0.0039184317`; uninterrupted
  half-specific noise gives `0.9999999425` and `0.0000823915`. Half-1 Pmax is
  bitwise unchanged between arms, and its arm-to-arm map FSC-AUC is
  `0.9999930328`.
- This causally explains the previous half-2 one-step substitution anomaly as
  a restart-versus-uninterrupted diagnostic mismatch, not a production half-2
  defect. The fixed uninterrupted RELION target came from another A100, so the
  target comparison is supporting evidence rather than final same-physical-GPU
  cross-engine acceptance. Map conclusions use only FSC/FSC-AUC.
- Science job `11293728` and audit job `11293729` completed successfully at
  RECOVAR commit `2ec77532`. Evidence:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it2_noise_semantics_2ec77532_20260717T040151Z/noise_semantics_analysis_v1.json`
  (SHA-256
  `4edde2570ad8833e925b923c2ab5c7bb5773bcbc0a5b6ca471388b43f7dfaeb1`).

# 2026-07-17: real-10076 finalization passes, but autonomous parity fails

- Same-GPU job `11292212` ran stock RELION and RECOVAR sequentially on
  physical A100 UUID `GPU-a1bb1fb4-d5e3-1c72-3382-63f6032e9fc6`. Both
  science commands exited zero. RECOVAR converged after numbered split-half
  iteration 16, ran final all-data label 17, and completed both halves plus
  Nyquist reconstruction. This independently validates the commit-`719ad930`
  cold-start finalization repair. The Slurm `FAILED` state is a wrapper-only
  stale grep for the old convergence log wording, not a science failure.
- Perfect autonomous parity nevertheless fails. The paired trajectories end
  at RECOVAR numbered 16/final 17 and RELION numbered 18/final 19. Existing
  same-UUID stock-RELION A/B controls both end at numbered 16/final 17, so
  convergence topology is repeat sensitive, but this does not absorb the
  stable early cross-engine residual.
- The earliest FSC-array difference beyond both engine controls is numbered
  iteration 1, shell 20: RECOVAR `0.7094454765` versus RELION `0.709487`, an
  absolute residual of `4.1523468e-5`; both repeat controls are exactly zero
  at that boundary. The first half-map FSC-AUC below `0.995` occurs at
  iteration 6 and the first merged-map gate at iteration 7. Final merged
  cross-engine FSC-AUC is `0.798012166`, versus `0.967954843` between the
  same-UUID RELION repeats.
- The iteration-15 shell-33/current-size branch remains unresolved and repeat
  sensitive: independent RECOVAR and RELION trajectories cross the `0.5`
  threshold in both directions. Do not patch that discrete branch. The next
  causal boundary is a paired iteration-1 contribution capture/replay,
  beginning with FSC generation and aggregate Pmax/pose/translation arrays.
- Evidence:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_parity_synthesis_719ad930_20260717T003600Z/synthesis.json`
  (SHA-256
  `3c9264755b1823f207037d3e3168a2b219a886bba510f8be06ed2e29a151b780`)
  and its verified `MANIFEST.sha256` (SHA-256
  `6d00da0be159d946d45dfccc8c8c069161b7b3b201d97e6edec8ef344c834810`).
  Intermediate comparisons use exact/array metrics; map gates use only
  shellwise FSC and FSC-AUC.

# 2026-07-17: iteration-2 PPref grid boundary is inherited from stack 111721

- A complete-grid offline replay sends paired control and HIGHEST iteration-1
  half-2 maps through the same RELION `Projector::computeFourierTransformMap`
  binding, then compares all `187 x 187 x 94` iteration-2 PPref values and all
  20,064 p8240-accessed corners with the sealed RELION target.
- HIGHEST removes only `1.00210%` of the full-grid residual L2 and `1.29897%`
  at the accessed corners; `98.9979%` of the full-grid residual remains. The
  default-A100 GEMM defect is real and its production repair remains justified,
  but it is not the material cause of this PPref boundary.
- A stronger common-weight/tau substitution closes the apparent upstream
  ambiguity. Replacing the 4,999-particle systematic RELION source bucket
  while leaving stack 111721 unchanged removes only `1.44088%` of residual
  energy. Replacing only stack 111721 removes `98.4884%`; replacing all RELION
  sources removes `99.999786%`. The p8240 PPref-grid symptom is therefore
  overwhelmingly inherited from the already-classified stack-111721/original-
  particle-8494 float32 near-tie translation-child decision, not a broad PPref
  generation defect. Do not resume serial particle tracing.
- Control/fixed maps and the RELION target use physical A100 UUID
  `GPU-64011c8c-bd98-eb41-2c46-dd201730ef64`, but the RELION target comes from
  the earlier sealed p8240 allocation. This is a bounded causal discriminator,
  not a same-allocation acceptance gate. No map-quality claim or correlation
  metric is used.
- Evidence:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_highest_ppref_grid_replay_20260717T063000Z/analysis/aggregate_ppref_grid_highest_replay_v1.json`
  (SHA-256
  `c755aeb9f3e0ee0092029639dfb23c6ba1cd7880844a30b0acb5ca1dc0372a04`),
  with a verified three-entry manifest at
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_highest_ppref_grid_replay_20260717T063000Z/provenance/SHA256SUMS`
  (SHA-256
  `f2451ff49c494affc090c8de066828e18a37b4316ae08e617da88cc475efa79c`).

# 2026-07-17: autonomous 100k K=1 fails convergence and quality parity

- Provenance-gated science job `11290560` ran commit `32ac19dc` on 100,000
  particles. RECOVAR reached its numbered-iteration cap at 17 without
  convergence and correctly skipped final all-data. The reference RELION run
  has 16 numbered split-half iterations followed by a converged final-all-data
  pass, so strict convergence topology fails.
- The pinned-commit summary job `11295485` sealed the failure. RECOVAR's
  pre-final merged GT FSC-AUC is `0.403209`, versus RELION final GT FSC-AUC
  `0.490627`, a delta of `-0.0874175`. The maps being pre-final versus final is
  explicitly recorded, but it is itself a consequence of the convergence
  mismatch and does not qualify parity.
- A visible scheduling amplification occurs after numbered iteration 15.
  RECOVAR recomputes `acc_rot=0.623` degrees and advances from HEALPix order 6
  to 7; RELION reports about `0.625` degrees on entry to iteration 16, stays at
  order 6, reaches stall
  counters `(resolution=2, hidden-variable=2)` at numbered iteration 16, and
  enters final all-data. Because the two maps and expected-accuracy operands
  have already diverged, do not patch this threshold or scheduler branch. The
  causal next step is an aggregate iteration-15 boundary factorial over the
  common first-100 expected-accuracy trials, substituting map, poses, and noise
  between engines. Do not resume serial particle tracing.
- RELION's `0.627` value stored in `run_it015_optimiser.star` was computed on
  entry to iteration 15 and must not be compared to RECOVAR's entry-to-16
  `0.623` value. The matched RELION entry-to-16 log value is about `0.625`.
- This autonomous failure does not contradict the replay-controlled 100k gate
  above: that experiment supplied the RELION trajectory boundary and therefore
  did not test autonomous schedule evolution.
- Evidence:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_completion_autonomous_fulltraj_32ac19dc_20260716_221339/analysis/pinned32ac_summary/summary_metrics.json`
  (SHA-256
  `8a29aee3d038a7831cd0c531591bd405647d4b0f53e99a538f48e9cb7be56f7b`)
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_completion_autonomous_fulltraj_32ac19dc_20260716_221339/analysis/pinned32ac_summary/FINAL_MANIFEST.sha256`
  (SHA-256
  `7a6cc169c9002c8860dd8f42ba2c670a975181f189ed8e09cd0736577bc7d902`).
  Map acceptance uses FSC/FSC-AUC only.

# 2026-07-17: autonomous 100k K=1 common numbered maps pass

- Numbered-only FSC job `11295651` compares the 16 boundaries common to the
  autonomous RECOVAR trajectory and RELION. Both audit views deliberately
  exclude final products because RECOVAR did not converge or run final
  all-data, while RELION did.
- All numbered map gates pass. The minimum merged cross-engine FSC-AUC is
  `0.9993081862` at iteration 16, the minimum half-map cross-engine FSC-AUC is
  `0.9988855748` at iteration 16 half 2, and the worst RECOVAR-minus-RELION
  merged GT FSC-AUC is `-0.0001185833` at iteration 9.
- Therefore the strict autonomous failure is convergence/finalization
  topology, not a collapse of common numbered-map quality. Keep the late
  expected-accuracy boundary open; do not compare RECOVAR pre-final products
  to RELION's final all-data map as if they were equivalent products.
- Clean isolated aggregate state job `11295840` independently fails exactly
  four strict checks: iteration-16 HEALPix order, convergence iteration,
  convergence flag, and final-all-data presence. Current size and HEALPix
  order match through iteration 15. Pose-error p95 remains about `3e-5`
  degrees through iteration 15, while Pmax residuals become broad at the
  local-refinement transition
  (iteration-8 mean absolute `0.0485663`, p95 `0.140763`). At the iteration-16
  grid split, mean absolute Pmax error is `0.315048` and pose-error p95 is
  `0.527618` degrees. Treat the earlier posterior calibration residual as an
  aggregate open diagnostic even though the numbered map-quality gates pass.
- Evidence:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_completion_autonomous_fulltraj_32ac19dc_20260716_221339/analysis/autonomous_trajectory_fsc_v3/k1_fsc_trajectory_common16.json`
  (SHA-256
  `f164d21a7be793f122962a871c238006327cb66bcc7423e473205745a09669c5`)
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_completion_autonomous_fulltraj_32ac19dc_20260716_221339/analysis/autonomous_trajectory_fsc_v3/FINAL_MANIFEST.sha256`
  (SHA-256
  `ab9e55f9bfc881b5e1c1add803723b0e4df3919da02394b54ce88d9fa6a33490`).
  State evidence is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_completion_autonomous_fulltraj_32ac19dc_20260716_221339/analysis/autonomous_trajectory_state_v4_clean/particle_state_distribution_common16.json`
  (SHA-256
  `5acfe87fa2994fda52adc6fdcbce7cf548d2e1c6863a0d93b3da4325b8c32b3e`)
  with manifest
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_completion_autonomous_fulltraj_32ac19dc_20260716_221339/analysis/autonomous_trajectory_state_v4_clean/FINAL_MANIFEST.sha256`
  (SHA-256
  `f3b8a12e243fa07c1b6d1facc1a7166d27d8d4ddae388440ddba059eb8037eeb`).
  Map acceptance uses FSC/FSC-AUC only.

# 2026-07-17: 100k expected-accuracy serialized-operand factorial is null

- CPU job `11295812` freezes the post-iteration-15 half-1 boundary and uses
  one exact 100-particle trial set for all 16 substitutions of serialized
  reference map, Euler poses, radial noise, and current image size. Every arm,
  including all-RECOVAR and all-RELION serialized inputs, returns exactly
  `acc_rot=0.6230000000000006` degrees and
  `acc_trans=0.6353750000000011` Angstrom. The map, Euler, and noise arrays are
  non-identical by SHA-256; current size is identically `188` in both engines.
- RELION cadence is explicit: `0.627` in `run_it015_optimiser.star` belongs to
  entry to iteration 15. The matched uninterrupted native entry-to-iteration-16
  value is about `0.625`, stored in `run_it016_optimiser.star`. Thus the
  all-RELION serialized binding replay still misses native RELION by about
  `0.002` degrees.
- The remaining aggregate fork is live double-precision in-memory state versus
  serialized/reloaded state, or a semantic/unbound-input mismatch between the
  standalone binding and native MPI `calculateExpectedAngularErrors`. Both CPU
  paths use double `RFLOAT`, so this is not currently classified as generic
  float32 noise. The next discriminator must preserve the exact original
  first-100 trial identities across a disposable native RELION restart; a
  normal restart with re-randomized particle order is invalid evidence.
- Evidence:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_100k_expected_accuracy_factorial_20260717T053100Z/RESULT.md`
  (SHA-256
  `2be72d7fd9142e5ea5a2cb21a796a338681865dd0ab6bf1be3c718070affd5d7`),
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_100k_expected_accuracy_factorial_20260717T053100Z/analysis/expected_accuracy_factorial_v1.json`
  (SHA-256
  `d45d66455bf2223018bbb72595e327c0a066916c72ce3588b4aa1f7ca453ada9`),
  and verified manifest
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_100k_expected_accuracy_factorial_20260717T053100Z/provenance/FINAL_MANIFEST.sha256`
  (SHA-256
  `65190c94a159e325e8e2a2aba87bf1991faa559ebf6b6f7f63843d6fa93c4971`).

# 2026-07-17: real-10076 same-GPU accumulator precision is not causal

- Science job `11293740` runs 15-iteration f32/c64 and f64/c128 accumulator
  arms sequentially on physical A100
  `GPU-8a30ed71-3361-7198-deac-61f8598401b7`; audit job `11294350` completes
  and verifies all manifests. This A/B widens already-produced operands and
  accumulators, not genuine upstream operand generation.
- At iteration 1 shell 20, f32 and f64 are exactly
  `0.7094454765319824`, both `-4.15234680176e-5` from same-UUID RELION. Thus
  accumulator widening removes none of the earliest residual. The f64 arm
  instead has its largest active-curve residual `0.00107424949` at shell 24.
- f32 matches the same-UUID RELION current-size schedule through all 15
  numbered iterations. f64 first differs from both at iteration 13, so the
  late f64 trajectory sensitivity is not evidence for a production precision
  change. No production change is justified.
- Next classify the frozen iteration-1 reduction boundary with identity-complete
  original/canonical order replays, captured-cast f64, and genuine upstream
  f64/c128 operand recomputation. Use exact arrays for intermediates and
  FSC/FSC-AUC for maps; do not resume serial particle tracing.
- Evidence:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_bpref_f32_f64_samegpu_ab_719ad930_20260717T001600Z/analysis/samegpu_precision_audit.json`
  (SHA-256
  `c3a40e1e94fa3173da2892896129c378c0fe8cc3ed25199ba0bc596d97653453`)
  and verified analysis manifest
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_bpref_f32_f64_samegpu_ab_719ad930_20260717T001600Z/analysis/ANALYSIS_ARTIFACTS.sha256`
  (SHA-256
  `4128f0446d2890acab508a00f4052f4e711d296ca57765da009fd2a10859f4f9`).
  Correlation is not computed.

# 2026-07-17: autonomous 100k terminal product alignment

- CPU job `11295847` separates RECOVAR numbered indices 15 and 16, RELION
  numbered iteration 16, RELION's final unfiltered half-map average, and its
  final merged map. This corrects the earlier invalid direct comparison of a
  RECOVAR numbered product with a RELION final product.
- RECOVAR index 15 versus RELION numbered iteration 16 has merged FSC-AUC
  `0.9993081862`. Even RECOVAR's unnecessary extra index 16 remains close to
  RELION numbered iteration 16 at `0.9985865016`, although its GT FSC-AUC
  falls from `0.4109877057` to `0.4032090802`.
- RELION numbered iteration 16 versus RELION's own final half-map average has
  FSC-AUC only `0.5307371223`; RECOVAR index 15 versus the same final product
  is `0.5309642252`. The previously reported approximately `0.531` terminal
  cross-engine value is therefore a product-type/finalization boundary, not
  evidence that RECOVAR's numbered map collapsed.
- The strict outer verdict job `11295841` remains failed for exactly the
  17-versus-16 numbered count, iteration-16 HEALPix order, convergence
  iteration/flag, and final-all-data presence. Its common-prefix FSC gate is
  independently passing. Evidence:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_completion_autonomous_fulltraj_32ac19dc_20260716_221339/analysis/autonomous_terminal_alignment_v1/terminal_alignment.json`
  (SHA-256
  `8108303af00f015b49e046471c617d82b94488ae21288b1302fb80bcd9b67f21`)
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_completion_autonomous_fulltraj_32ac19dc_20260716_221339/analysis/autonomous_trajectory_strict_verdict_v1/strict_verdict.json`
  (SHA-256
  `966395436f457e57ccb92095e4cc145ffd15c39956d213b6c46e1f202c585acd`).
  Map quality uses FSC/FSC-AUC only.

# 2026-07-17: sealed 100k native-restart expected-accuracy classification

- The decisive native RELION control is sealed under
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_100k_relion_restart_accuracy_oracle_20260717T054518Z`
  (`SAFE_TO_DELETE`). Build job `11296008` and native MPI restart job
  `11296009` completed `0:0`. The restart exits immediately after the
  iteration-16 expected-accuracy calculation, before expectation.
- Ordinary continuation order is invalid evidence because a new RELION
  process reshuffles the particle order. This control instead restores the
  exact original first 100 trial IDs after that shuffle. The IDs are unique,
  all belong to half 1, and their complete STAR row identities agree between
  iterations 0 and 15. The ID file SHA-256 is
  `84262018ebd56268dfb8cfa1e674e97b09f8d9a712f967df39b2f3c0a0e6190a`;
  the canonical int64 ID-sequence SHA-256 is
  `0d4dc2a259d594b2bc656fc763c8a41413c78e5595e2043c36f8947f9388142a`.
  The identity proof is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_100k_relion_restart_accuracy_oracle_20260717T054518Z/analysis/original_trial_identity_proof.json`
  (SHA-256
  `068e5c8955b1ca34a695f92490493bccb3654d972f2c8b94374dcd778a274fbc`).
- With those exact identities, native restart reports `0.623` degrees and
  `0.635375` Angstrom. The independently reduced 100-row per-trial CSV gives
  `0.6230000000000006` and `0.6353750000000011`, exactly reproducing the
  serialized standalone all-RELION replay and not uninterrupted native
  entry-to-iteration-16 (`0.625` degrees and `0.6375` Angstrom). This
  exonerates the standalone binding at this boundary. The unresolved source
  is live in-memory RELION state versus checkpoint write/reload, including
  the possibility of mutation-before-write state that is not serialized.
- This approximately `0.002`-degree terminal difference does not explain the
  much earlier real-10076 schedule split and must not redirect that aggregate,
  distribution-level investigation. No production patch is justified by this
  terminal diagnostic.
- Sealed results are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_100k_relion_restart_accuracy_oracle_20260717T054518Z/analysis/native_restart_accuracy_v1.json`
  (SHA-256
  `4249a06cff0fb63884fa7f68079b56a52fa6a0d1ae90c095bc5ecf1e57e587fd`)
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_100k_relion_restart_accuracy_oracle_20260717T054518Z/restart/per_trial_errors.csv`
  (SHA-256
  `d29e4b460c5562a0328f4fb5ed6806c138bc4408c6c765e95bb224aa68d91da4`).

# 2026-07-17: real-10076 HEALPix timing explains a repeat-sensitive map fork

- Science job `11294177` ran four ten-iteration RECOVAR arms sequentially on
  one physical A100: two autonomous controls, a RELION HEALPix-order oracle,
  and a full RELION HEALPix/current-size oracle. All four science commands
  completed. The job's final `FAILED` state is only the original audit
  wrapper's checkout-import error; isolated CPU audit job `11296645` completed
  with exit code zero and verified the science artifacts.
- Both autonomous controls use HEALPix orders
  `[3,3,3,3,3,3,3,4,4,4]`. Their merged-map A/B FSC-AUC remains at least
  `0.996052964` through iteration 10, so their common refinement at iteration
  8 is stable within the RECOVAR repeat. For the paired primary RELION target,
  holding only its HEALPix schedule `[3,3,3,3,3,3,3,3,3,4]` raises the
  minimum iteration-8--10 merged FSC-AUC from `0.726125984` to `0.978128435`.
  Also forcing its current sizes raises the minimum only to `0.978681776`;
  current size is therefore a marginal secondary effect for this target.
- This does not make the primary target's iteration-10 refinement a unique
  RELION truth. Two independent RELION controls ran sequentially on another
  physical A100; both refine at iteration 8 with the same schedule and
  perturbations as autonomous RECOVAR. Against those targets the result
  reverses: the
  autonomous minimum is `0.976245891`/`0.976847520`, while the HEALPix-oracle
  minimum is `0.727216099`/`0.727921637`. RELION's own same-GPU repeat reaches
  a minimum full-trajectory FSC-AUC of `0.954744985`. The HEALPix branch
  causally explains the late map bifurcation, but the branch itself is within
  RELION's nonlinear repeat behavior and is not a systematic RECOVAR defect.
  Freeze reduction inputs/order before requiring one of these two
  RELION-realized branches.
- Schedule matching does not close the remaining quality residual. Using the
  same merged-map statistic, native RELION repeat A/B FSC-AUC is `0.996451`,
  `0.993440`, and `0.990081` at iterations 8--10, above RECOVAR's
  approximately `0.976` against either early repeat at iteration 10. The gap
  grows gradually from iteration 2 and is not labeled numerical noise.
- The scheduler implementation is not the causal defect. The tests in
  `tests/unit/test_convergence.py` reproduce RELION's refinement at iteration
  10 when fed RELION's recorded hidden-change scalars and RECOVAR's refinement
  at iteration 8 when fed RECOVAR's scalars. Do not patch the 3-percent
  threshold or stall counter.
- Hardware qualification: the intervention arms share physical A100 UUID
  `GPU-f3e946...`, the primary target pair shares `GPU-a1bb1...`, and native
  RELION repeats share `GPU-bd720...`. Direct intervention-to-target metrics
  therefore use the same A100-80 model but not the same physical UUID. A
  live-RELION-derived same-allocation full trajectory remains required for
  cross-engine acceptance.
- The scalar difference is concentrated in a rare pose tail. Across the
  primary target, RECOVAR's iteration-6--7 mean angular change is
  `4.834649267` degrees versus RELION's `4.743451024`. Only `518/10000`
  particles differ by more than `0.1` degree, while the largest 1 percent
  accounts for `92.2177%` of the absolute cross-engine change difference.
  The median and p95 cross-engine pose errors remain approximately `1.18e-5`
  and `3.58e-5` degrees, so percentile summaries alone hide the subgroup.
  The same classification holds against both independent RELION repeat arms
  and the second RECOVAR control. The subgroup is enriched for lower Pmax.
  An identity-fixed trajectory audit shows no subgroup pose errors above
  `0.1` degree at iterations 1--2, then `1`, `10`, `42`, `218`, and `353` of
  the 518 identities at iterations 3--7. The cohort explains `90.2869%` of
  hidden-change disagreement at the iteration-3--4 boundary and `99.9745%`
  at iteration 6--7. This localizes the fan-out to iteration 3 onward; the
  repeat-envelope discriminator below selects its earliest systematic
  precursor.
- Comparing those identities with RELION's A/B control arrays isolates one
  systematic precursor before that ten-particle fan-out. Fixture row `5676`
  (image `73773`, half 1) is still close at iteration 2: Pmax differs by
  `2.93813e-6` and pose by `7.37e-6` degrees. At iteration 3 its Pmax differs
  by `0.190258046` and its pose by `4.717064956` degrees, while the RELION A/B
  Pmax difference for the same identity is only `0.000176`, a ratio of
  `1081.01`. The other nine first-diverging identities remain pose-exact at
  iteration 3 and fan out at iteration 4. Prioritize a frozen iteration-2--3
  scorer/operand capture for row `5676`; treat iteration-3--4 as propagation
  until that precursor is classified.
- Primary evidence is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_sampling_oracle_ab_ff2ed9d3_20260717T044116Z/analysis/schedule_oracle_ab_v3.json`
  (SHA-256
  `de3b602ddaa8d141495546825d23e65c1b33b1afaeb6700593358bc090f36295`)
  with verified manifest
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_sampling_oracle_ab_ff2ed9d3_20260717T044116Z/provenance/science_artifacts_v3.sha256`
  (SHA-256
  `6b520df3de8b081f08e080349552c3380932855cf6f9a98b054587f788f27ede`).
  Tail evidence is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_sampling_oracle_ab_ff2ed9d3_20260717T044116Z/analysis/hidden_change_it006_it007_v1/hidden_change_distribution.json`
  (SHA-256
  `ccb20a0aaffbc73ab9211a546dbc3ea1a9a479b0e85a02d5837004fbc8b16e32`)
  with verified manifest SHA-256
  `133eb60f481f150f1778c340dac5772f7eb801587d1672f6307f65dd6afe9ae1`.
  The three-target schedule manifest is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_sampling_oracle_ab_ff2ed9d3_20260717T044116Z/analysis/SCHEDULE_ORACLE_SHA256SUMS`
  (SHA-256
  `d87634cddeb640567f3df1943192861d418f1d74db9b0dff7ff1bc660f6b6c8a`).
  RELION self-repeat evidence is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_relion_repeat_envelope_a10080_retry4_prepared_20260715_195515/analysis/repeat_envelope.json`
  (SHA-256
  `abcb1fafbbc090f96c5d2927e012f272e2eeecbdba5cb3da286fd787b91a4805`).
  Its identical-statistic merged-map control is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_relion_repeat_envelope_a10080_retry4_prepared_20260715_195515/analysis/native_repeat_merged_fsc_v1.json`
  (SHA-256
  `d6eba533c9d11b127855ad10dba4c1f3cb9f201df4ea0a0ff115b482a27e90c7`).
  The fixed-cohort trajectory is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_sampling_oracle_ab_ff2ed9d3_20260717T044116Z/analysis/fixed_hidden_tail_trajectory_v1/fixed_hidden_tail_trajectory.json`
  (SHA-256
  `c9eeaf67952cb50a7c923c6dd29877600bc07e0040abaae6725d15d731ef2dc4`)
  with verified manifest SHA-256
  `8d4c0a4ace586db948b5df527c23505e95edb9fb0a8e1ea70b3b5e7973174c31`.
  Its RELION-repeat discriminator is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_sampling_oracle_ab_ff2ed9d3_20260717T044116Z/analysis/fixed_tail_repeat_envelope_v1/fixed_tail_repeat_envelope.json`
  (SHA-256
  `8a3a46406f1e34ecd7f0b3e9bc4ed0e8135c8676912d6d167c9ce99f0f6786d9`)
  with verified manifest SHA-256
  `1838fbf33e0155a4744ae5750d1bb94505ee7889ebcf837a7b599c3f927420d1`.
  Map acceptance uses FSC/FSC-AUC only; correlation is not computed.

# 2026-07-17: iteration-2-to-3 aggregate residual is in raw scoring

- The sealed 32-row production panel has exact RELION/RECOVAR candidate UID
  support. Native-order posterior replay closes within TV `7.29e-7`. A
  canonical-float64 factorial gives median prior-only TV `7.13e-8` versus
  raw-score-only TV `1.5357e-4`; do not pursue priors or normalization as the
  leading cause. Particle 2449 has the only M-step support exchange, one of
  144 hypotheses, and does not define a systematic subgroup.
- High-precision RECOVAR score-operand job `11305358` completed all 32 rows.
  Median common-support TV is `1.57054e-4` for production RECOVAR versus
  RELION float32-origin, `1.28238e-3` for high-precision RECOVAR versus
  RELION, and `1.27341e-3` for RECOVAR production versus high precision.
  Zero rows improve. Particles 6007 and 1012 each substitute 32 UIDs; 30/32
  supports remain exact.
- Classification: this is precision sensitivity, not float64 cross-engine
  closure. RELION GPU operands remain float32-origin and the high-precision
  arm changes pass-1 membership. Do not call the smaller production residual
  numerical noise yet. Next use a fixed-production-UID all-32 operand
  cross-swap and reduction-order factorial; do not return to serial particles.
- Evidence is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it23_aggregate_panel_586f7fb4_20260717T093000Z/analysis/score_factorial.json`
  (SHA-256
  `4075c4dfb2a65782de52c75f992e85be0cf7f9b22e0c9a24bb73c6880b1df7d0`)
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it23_genuine_f64_20260717T145500Z/analysis/genuine_f64_vs_f32_origin.json`
  (SHA-256
  `b54aa3a5221145ce1b6df204f3c5c3197b4230560f39fe3955b624c8b3c6c955`).
  Intermediate metrics are exact arrays and posterior distances; map quality
  remains FSC/FSC-AUC only, with no correlation.

# 2026-07-17: fixed-UID score reduction is on the float32 sensitivity scale

- Sealed job `11306470` passes all 32 closure rows and 17,216 exact five-field
  candidate UIDs. RECOVAR's native tree reproduces captured production scores
  exactly; adjusted float64 direct/algebraic closure is at most `4.38e-12`.
- Against deterministic `math.fsum`, float32 pairwise reduction gives median
  posterior TV `7.74e-5` and RELION's 256-lane tree gives `7.39e-5`. The
  high-accuracy projected-reference swap is smaller at median `1.014e-5`; the
  image/CTF/noise/scale swap is `6.17e-7`.
- Classification: the observed production score gap is on the float32
  reduction-sensitivity scale, but a native cross-engine order mismatch is not
  yet proved. Capture one immutable common contribution list and replay both
  native schedules plus a shared canonical schedule before changing
  production. Do not return to serial particle tracing.
- Evidence is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it23_aggregate_panel_586f7fb4_20260717T093000Z/analysis/score_reduction_factorial.json`
  (SHA-256
  `271dbf2fbc84d99659971f5517d8dfe834afe94615a28931f80d1359fc36bb72`)
  with verified manifest
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it23_aggregate_panel_586f7fb4_20260717T093000Z/provenance/score_reduction_factorial.sha256`
  (SHA-256
  `6cde8dd18e8961c9d48a1a60c78392e1d030f842b4c962ab0059662b6f516a10`).
  No correlation is used.

# 2026-07-17: K=4 100k/256 compact-score memory fix passes two iterations

- Science job `11304416` completed every compact/rectangular group in
  iterations 1 and 2, including the former 24,576 and 94,208 OOM boundaries.
  Cumulative sampled A100 memory is 33,361 MiB of 81,920 MiB, with no OOM,
  allocator, resource-exhaustion, or traceback signature.
- Iteration 1 finished in 4,710.3 seconds; iteration 2 finished in 2,727.6
  seconds and advanced to iteration 3 at RELION-derived current size 60. Both
  boundaries wrote finite class/half maps and finite state arrays.
- This accepts memory safety, not scientific parity. Keep strict class-matched
  FSC/FSC-AUC audit `11304830` as the quality gate after the science trajectory
  under
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_100k_memcap_fce9ee48_20260717T132717Z`.

# 2026-07-17: real-10076 full trajectory separates schedule and state drift

- Same-A100 science job `11307959` completed four K=1 arms: two native RELION
  repeats, autonomous RECOVAR, and RECOVAR forced to RELION's numbered
  schedule. Audit job `11307960` intentionally exits 2 because the strict map
  and state gates find real mismatches. The two RELION arms match all 18
  numbered sampling/size boundaries and converge during expectation 19.
- Autonomous RECOVAR advances HEALPix orders 4, 5, and 6 two iterations early
  and finalizes after numbered iteration 16. Forced scheduling avoids the
  iteration-8 map collapse, but it does not close the accumulated state gap:
  forced merged RECOVAR-vs-RELION FSC-AUC is `0.9947079` at iteration 7 and
  `0.9745783` at iteration 16. The native RELION-repeat envelope is
  `0.9999709` and `0.9929671` at those boundaries. Autonomous final merged
  FSC-AUC is `0.778789`.
- Direct arrays show gradual drift before the schedule branch. At iteration 2,
  214/10,000 significant-support rows differ and mean absolute Pmax error is
  `3.6e-5`; by iteration 7 the corresponding values are 1,393/10,000 and
  `0.03318`. The early HEALPix transition then produces median angle and
  translation errors of `2.37` degrees and `0.819` Angstrom at iteration 8.
- The historical post-cap diagnosis was later invalidated by direct RELION
  source audit. RELION checks convergence only at the top of a loop iteration
  satisfying `iter <= nr_iter`; it does not run a sampling or convergence
  boundary after the cap. Commit `607e4344` removes RECOVAR's synthetic
  post-cap check. The forced-arm termination difference must therefore be
  traced to the earlier accumulated map and adaptive-schedule state.
- Evidence is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_current_head_fulltraj_15a6d355_20260717T190000Z/analysis/strict_acceptance.json`
  (SHA-256 `7dcd2071ed58fddefa046dc22299d1249c9d3ac351399a27d794456c9ee054be`),
  `strict_acceptance_shellwise.npz` (SHA-256
  `d69242e48b38d07e3b8d424a604fbad80dc3ad1f57fc15a835b6f06356d538a6`),
  `particle_state_distribution.json` (SHA-256
  `8c75e971812c3d8b3cb62257a6abbc9141037d9c423e74f67da28fe16bd5541e`),
  and `dynamic_schedule_trajectory.json` (SHA-256
  `3a20a0165c87a25e424a8ad1de59361559cb1bf7280055c7333fdeb0367cb28c`).
  Map acceptance uses FSC/FSC-AUC only; correlation is not computed.

# 2026-07-17: common contribution replay localizes iteration-3 scoring to operands

- Fresh same-GPU RELION control/capture job `11311539` and replay job
  `11312187` completed `0:0` on A100 UUID
  `GPU-803dc869-2e74-273c-1df4-08adbc94e1b3`. The replay passes every
  capture, geometry, native-closure, and repeatability gate for 32 rows,
  17,216 exact five-field candidate UIDs, and 126,021,120 float32 pixel
  contributions.
- The instrumented RELION capture, the earlier sealed RELION panel, and
  RECOVAR have exact candidate support for all 32 rows. The independent
  env-off RELION control differs only for particle 5676, where it omits 32
  capture candidates carrying `0.9958962` capture posterior mass. This is
  recorded as a native threshold cliff, not used as an exact-support gate;
  control/control arithmetic is evaluated only on exact shared UIDs.
- Both native schedule implementations give identical results on each
  immutable contribution list, so a cross-engine reduction-order mismatch is
  rejected. RELION and RECOVAR per-pixel terms are not bitwise common. Native
  common-prior posterior TV has median `1.2986e-4`, p95 `3.9140e-4`, and
  maximum `5.3648e-4`. Shared canonical float64 reduces these to median
  `3.6555e-5`, p95 `3.3187e-4`, and maximum `5.1888e-4`, but does not close
  them. The earliest systematic classification is therefore operand
  generation with an additional float32 order/precision contribution.
- Next capture the exact native projected-reference, shifted-image, and
  score-weight operands for the same aggregate rows. Do not change production
  until that decomposition distinguishes an algorithmic operand mismatch from
  expected float32 interpolation/phase noise.
- Sealed report:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it23_common_contrib_20260717T180500Z/analysis/common_contribution_replay.json`
  (SHA-256 `39ca0755d1b0e100205b812020819b838ab223d6a1b6a3765ef964475be2eee1`).
  Its verified top manifest is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it23_common_contrib_20260717T180500Z/provenance/common_contribution_replay.sha256`
  (SHA-256 `c596e13b22afd61b7db79a104d5814edb8cb8c97a74c1e3c9c26ffd66250be85`).
  Intermediate evidence uses exact arrays and posterior distances; map quality
  remains FSC/FSC-AUC only and correlation is not computed.

# 2026-07-17: accelerated scorer matrices come from RELION's host inverse path

- Source tracing and a fresh native capture correct the earlier CUDA-matrix
  interpretation. RELION's accelerated fine expectation and weighted-sum
  paths both call host `generateEulerMatrices(..., inverse=true)` in RFLOAT
  precision, cast the result to XFLOAT, and then copy it to the device. The
  CUDA `make_eulers_3D` helper remains a valid isolated arithmetic diagnostic,
  but it is not the production matrix generator for this EM path.
- All 1,120 unique captured fine matrices differ from RECOVAR's former
  device-reconstructed matrices; 7,822/10,080 entries differ, usually by a
  few float32 ULP. On 32 particles and 17,216 exact five-field candidate UIDs,
  the rotation handoff accounts for canonical-float64 posterior TV median
  `3.90795e-5` (p95 `3.31169e-4`). The remaining native-texture arithmetic
  residual has median `1.41812e-5` (p95 `5.81312e-5`) and remains a separate
  follow-up.
- The candidate host-double inverse generator matches the native capture
  bitwise: zero differences across 10,080 unique and 154,944
  candidate-weighted float32 matrix entries. Replaying the frozen PPref/helper
  produces bitwise-identical references and zero canonical posterior TV
  versus the captured-native-matrix arm for all 32 particles. Production EM
  now uses this host path for both scoring and backprojection; the CUDA helper
  is diagnostic only. Full-trajectory FSC/FSC-AUC remains the acceptance gate.
- Evidence is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it23_common_contrib_20260717T180500Z/analysis/native_rotation_texture_discriminator.json`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it23_common_contrib_20260717T180500Z/analysis/candidate_rotation_generator_validation.json`,
  with verified manifest
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it23_common_contrib_20260717T180500Z/provenance/candidate_rotation_generator_validation.sha256`.
  Jobs `11314778` and `11315399` completed `0:0`. Intermediate gates use exact
  arrays and posterior TV; map acceptance uses FSC/FSC-AUC only and correlation
  is not computed.

# 2026-07-17: host-matrix full trajectory preserves early drift and fails strict acceptance

- Commit `8c83202739a31251bc6c10be834f237732e879d3` uses RELION's
  host-generated inverse rotation matrices for scoring and backprojection.
  Science job `11315888` completed `0:0` in 58:29, with stock RELION followed
  by RECOVAR on physical A100 UUID
  `GPU-928b9735-3919-8a8c-41b9-a7ca7b41017b`. Both engines completed a valid
  final all-data reconstruction; grid correction remained off. RELION had 18
  numbered iterations, while RECOVAR converged after numbered iteration 16.
  Existing sealed same-UUID RELION repeats also realize numbered-16 and
  numbered-18 branches, so this terminal topology difference is inside the
  observed RELION repeat envelope. It is still an exact strict-gate failure,
  but is not by itself classified as a deterministic RECOVAR defect.
- Before that nonlinear branch, the merged cross-engine FSC-AUC trajectory is
  `0.999999484051`, `0.999996539405`, `0.999757808900`,
  `0.998859025561`, `0.996646508538`, `0.994605851609`, and
  `0.991747711667` at iterations 1--7. Iteration 6 is the first merged map
  below the `0.995` acceptance threshold. The host rotation handoff therefore
  does not close the stable early aggregate drift.
- Late merged FSC-AUC is `0.914395202504` at iteration 15,
  `0.915809586413` at iteration 16, and `0.795975565806` after final
  all-data. The shellwise failures are broad: at the `0.995` criterion,
  iteration 15 and 16 each fail 105/127 shells, continuously from shell 22
  through 126; final fails 108/127 shells, continuously from shell 19 through
  126. At the stronger `0.9` criterion, the corresponding failure counts are
  44, 37, and 83 shells. Final half-1 and half-2 FSC-AUC are
  `0.791102509518` and `0.781113535114`.
- The exact-identity iteration-6--7 hidden-change audit aligns all 10,000 rows
  by unique `rlnImageName`. RECOVAR and RELION mean angular change are
  `4.731405544` and `4.750984210` degrees; the signed RECOVAR-minus-RELION
  difference has mean `-0.019578666` degrees but median only
  `-2.20724e-8`. The absolute difference has median `2.44246e-6`, p95
  `0.371592`, and maximum `115.549846` degrees. An evidence-defined
  `>0.1`-degree subgroup contains 584/10,000 particles, and the largest one
  percent accounts for `91.55596%` of the total absolute difference. This is
  a systematic rare tail rather than a broad pose shift. Five-field candidate
  UID evidence remains the separate frozen scorer-panel diagnostic; numbered
  STAR histories expose particle identities, not candidate support.
- Audit job `11315889` exited `2:0` after producing and hashing all requested
  artifacts. This nonzero exit is the expected strict acceptance result
  (`map_status=2`, `particle_state_status=1`), not a harness failure. The state
  audit verified exact `rlnImageName` topology with no missing matched
  RECOVAR iteration, then failed the requested exact schedule/convergence
  gates. Duplicate audit job `11318600` was cancelled as stale after these
  complete artifacts were verified.
- Evidence root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_host_matrix_fulltraj_commit_pending_a100_20260717T192457Z`.
  The main report is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_host_matrix_fulltraj_commit_pending_a100_20260717T192457Z/analysis/trajectory_acceptance.json`
  (SHA-256
  `81b67289629668e192abf3b176ff8d353f83b037369776b9f5763fe7068dfa7d`),
  with shellwise curves
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_host_matrix_fulltraj_commit_pending_a100_20260717T192457Z/analysis/trajectory_shellwise_fsc.npz`
  (SHA-256
  `5e880b4edf98db1841b5800ebd4bfe6ad5f9d14ee1add248fc029bc1579af1b7`).
  The particle-state report
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_host_matrix_fulltraj_commit_pending_a100_20260717T192457Z/analysis/particle_state_distribution.json`
  and compact arrays
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_host_matrix_fulltraj_commit_pending_a100_20260717T192457Z/analysis/particle_state_distribution_arrays.npz`
  have SHA-256
  `af6144f82a086310327a277a3fe2b19c036f250693c50ffd12985280972655ed`
  and `c32fe2878ec5bea03c5906aeeeb476773dde1f58bd52de1e4064d72e6b38f8aa`.
  The hidden-change report is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_host_matrix_fulltraj_commit_pending_a100_20260717T192457Z/analysis/hidden_change_it006_it007.json`
  (SHA-256
  `1dd1834877f8087bb1cab30b791af37715e0cd914e05c4f99df9a445d0598c81`),
  with arrays
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_host_matrix_fulltraj_commit_pending_a100_20260717T192457Z/analysis/hidden_change_it006_it007_arrays.npz`
  (SHA-256
  `f3b524c8280ec49ced2994e0d1c7cdd9b9c3207cbde1253db5819a35f81b19cd`).
  Intermediate comparisons use exact identity-aligned arrays; map quality uses
  shellwise FSC and FSC-AUC only. Correlation is not computed.

# 2026-07-17: incoming iteration-3 A/B improves aggregate state but not support cutoffs

- Same-allocation job `11318372` ran parent `7f142d5f`, host `8c832027`, and a
  second host control sequentially on physical A100 UUID
  `GPU-8a30ed71-3361-7198-deac-61f8598401b7`, starting from the complete exact
  RELION iteration-2 boundary.  The host controls have bitwise-identical Pmax,
  pose, translation, and significant-count arrays.  Their merged self-map
  FSC-AUC is `1.000000027180`; the worst parent/host merged FSC-AUC is
  `0.999999970436`.
- The source-matched host inverse matrices improve RELION agreement.  Absolute
  Pmax median/p95/mean changes from `6.21627e-5/2.22109e-4/8.54588e-5` to
  `4.95276e-5/1.74502e-4/6.70980e-5`.  Angular mean/p95 changes from
  `1.33237e-5/2.94768e-5` to `5.17333e-6/1.01020e-5` degrees.  Merged map
  FSC-AUC versus RELION changes from `0.999998435067` to `0.999998625826`.
- Significant-count cutoff parity moves in the opposite direction.  After
  scattering the serialized half-order arrays to exact input identities, the
  parent has 3/10,000 mismatches and each host arm has 9/10,000.  All six new
  host decisions worsen exact agreement by one count, with signed changes
  `+1,+1,+1,-1,+1,-1`.  Do not describe the host handoff as improving support
  parity and do not revert it: the aggregate direct-array and FSC-AUC evidence
  improves, while these six cutoff decisions localize remaining score/texture
  arithmetic.
- The A/B was clean and built at the named commits, but the shared host
  worktree advanced through a documentation-only commit during the serial
  arms.  Hashed production sources and built artifacts remained unchanged,
  but the job did not reassert immutable HEAD before every arm.  Treat this as
  a causal boundary diagnostic rather than exact-clean-runtime acceptance.
- Evidence:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_incoming_it3_rotation_ab_7f_vs_8c_retry1_20260717T203134Z/analysis/it3_rotation_ab.json`
  (SHA-256
  `feb3571d41fdf1277c77f68f9a9c03601781a6f6859101278f023e8d4cea72dd`)
  and `it3_rotation_ab_fsc_curves.npz` (SHA-256
  `619969d0059c07c879ec4d02daf21b4e152657ab7b200f91cba7b510c64f04de`).

# 2026-07-17: adjacent-boundary enrichment identifies an early recurrent tail

- The existing particle-state distribution auditor now relates exact
  `rlnImageName`-aligned significant-count mismatch and exact top-5%-absolute
  Pmax delta at boundary `t` to a `>0.1`-degree cross-engine pose tail at
  `t+1`.  It reports contingency counts, conditional rates, enrichment, and
  capture fraction for all 15 consecutive numbered boundaries.  Zero
  denominators are null and named explicitly.  This is diagnostic only: no
  correlation is computed and no quality gate was added.
- Iteration 2 to 3 is the clearest early recurrent boundary.  Significant-count
  mismatch exposure has contingency counts `7/556/8/9429`
  (exposed-and-tail/exposed-only/tail-only/neither), tail rates
  `1.2433%` exposed versus `0.0848%` unexposed, `14.6667x` enrichment, and
  7/15 capture.  The top-5% Pmax exposure has counts `5/495/10/9490`, rates
  `1.0%` versus `0.1053%`, `9.5x` enrichment, and 5/15 capture.
- Significant-count enrichment across boundaries 3-to-4 through 6-to-7 is
  `7.1413x`, `3.7132x`, `3.2311x`, and `3.1111x`; the corresponding capture
  fractions are `6.41%`, `14.0%`, `22.56%`, and `31.06%`.  This is a real
  distribution-level subgroup signal, but it is not a complete predictor.
  At iteration 7 to 8 all 10,000 particles exceed the pose threshold after the
  sampling branch, making both exposed and unexposed rates 1 and enrichment
  exactly 1.  Later broad-divergence ratios approach 1.  Iteration-1 Pmax
  deltas are all zero at the top-5% cutoff, so that deterministic tie-selected
  row is not interpreted.
- Audit job `11319327` completed `0:0`; its internal state status is the
  expected strict failure 1 because the sealed trajectory still fails exact
  schedule/convergence.  All 15 diagnostic boundaries and all 45 boolean
  artifact masks were validated.  JSON:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_tail_enrichment_0fa05894_20260717T204707Z/analysis/particle_state_distribution_tail_enrichment.json`
  (SHA-256
  `73df4fab3130179c8abee447635fe944439b2827714ec4851ab186342d1a6cef`);
  arrays:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_tail_enrichment_0fa05894_20260717T204707Z/analysis/particle_state_distribution_tail_enrichment_arrays.npz`
  (SHA-256
  `6d76535d88f33be7fc11249177c64dc99699c0e529c75cedcbb04e65af468757`).
- Use this to prioritize aggregate score/posterior subgroups and controlled
  iteration-boundary substitutions.  Do not resume serial particle tracing.
  Map quality remains gated only by shellwise FSC/FSC-AUC.

Code references:

- `scripts/audit_em_particle_state_distribution.py:_cross_iteration_tail_enrichment`
- `tests/unit/test_audit_em_particle_state_distribution.py:test_cross_iteration_tail_enrichment_uses_exact_aligned_state_and_is_diagnostic`

# 2026-07-17: matched-tail operand decomposition does not select a source

- The frozen cohort contains all 15 exact iteration-2-to-3 `>0.1`-degree pose
  tail rows and 15 deterministic matched controls. Half assignment is exactly
  10/5 in each cohort; significant-count exposure is 7/15 and top-5% absolute
  Pmax exposure is 5/15 in each cohort. Accepted local run `91721004` executed
  a fresh RELION control, instrumented RELION capture, and current-head RECOVAR
  replay sequentially on physical GPU UUID
  `GPU-dc6576aa-e1e4-6055-4a5e-d0fa809f3983`. The ownership monitor passed,
  all 30 rows have exact candidate support, and all capture manifests verify.
- The version-2 common-contribution replay keeps bitwise score closure as a
  diagnostic rather than a selector. It passes 30/30 local numerical gates:
  10,912 candidate scores contain one nonzero replay residual, exactly one
  float32 ULP at the canceled input scale, pooled p99.9 is zero, and its
  induced posterior TV is below the row's independent native-versus-canonical
  float64 control. The term self-closure adjudication also passes 30/30 rows.
  RECOVAR has 1,584/62,733,088 non-bitwise terms (`2.525e-5`), all within the
  independent RELION expression envelope and at most two output-float32 ULP;
  the maximum induced posterior TV is `6.92e-16`. This is classified as
  float32 term-formation rounding, not an algorithmic mismatch.
- The tail is not enriched for the remaining operand residual. Native
  common-prior posterior TV medians are `8.9293e-5` for tail rows and
  `1.13209e-4` for controls; the paired median tail-minus-control value is
  `-3.13513e-5`, with only 7/15 tail values larger. Shared canonical
  float64-from-captured-float32 medians are `4.08429e-6` and `4.21585e-6`;
  the paired median is `8.72021e-7`, with 10/15 tail values larger. The
  same-GPU control/capture envelope has pooled median TV `1.05062e-4` and
  maximum `2.23009e-4`.
- Raw mixed-source factorial arms are unit-mismatch controls and are not
  selectable. In the unit-aligned factorial, none of the single-field arms
  `CRR`, `RCR`, or `RRC`, nor the two-field arms `CCR`, `CRC`, or `RCC`, passes
  the pre-registered native-float32 and canonical-float64 effect, movement,
  pair-count, and repeat-envelope gates. Successful sealed selector job
  `11323681` therefore reports `classification=unresolved_combined`,
  `selected_arm=null`, `production_change_authorized=false`, and
  `substitution_launch_authorized=false`. No substitution was launched.
  Duplicate job `11323757` stopped at the immutable-output precondition and
  did not overwrite the sealed reports.
- Do not continue serial particle or pixel-operand capture. The next K=1
  diagnostic is a compact full-10,000-particle score/posterior distribution at
  the exact iteration-2 boundary, compared with a same-GPU RELION repeat.
  Preserve five-field candidate UIDs, centered scores, and posterior weights
  in bounded append-only shards with per-particle offsets so common-support
  posterior TV, centered residuals, and exclusive posterior mass can be
  recomputed directly. Atomic finalization, shard checksums, exact-identity and
  topology gates, inode/byte estimates, and a same-GPU RELION repeat are
  required. Do not scale the panel's per-field small-file layout to 10,000
  particles. Derive candidate/support counts, best/runner-up gaps, log
  normalizer/Pmax, entropy/effective support/top-k mass, significant
  threshold/count, and winner state, then report global and pre-registered
  half/defocus/Pmax/support strata. Map acceptance remains shellwise
  FSC/FSC-AUC only and correlation is not computed.

Sealed evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it23_tail_operand_panel_e1a7c87c_20260717T210105Z/analysis/common_contribution_replay_v2.json` (SHA-256 `f6ef5d14163e6b1cf6ffb2a6f8cfc5522dcf264e82af69a6c00f42100b3f6591`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it23_tail_operand_panel_e1a7c87c_20260717T210105Z/analysis/term_self_closure_v2.json` (SHA-256 `efe095f46bc788d806fd4eec173d124ae490bad4b00488730dda8cb59358873a`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it23_tail_operand_panel_e1a7c87c_20260717T210105Z/analysis/tail_operand_selector_v2.json` (SHA-256 `8362f79bff43b71fe89fc8ed25ba31ef6bfb91bdc8c7d7428e38cd95174bd105`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it23_tail_operand_panel_e1a7c87c_20260717T210105Z/provenance/selector_v2_outputs_11323681.sha256`

# 2026-07-17: full-cohort capture plumbing and exact-boundary correction

- The proposed five-arm iteration-2 restart panel is rejected. A stock RELION
  restart advances the sampling perturbation to `-0.24547913670539856`, not the
  uninterrupted iteration-3 value `-0.30032598972320557`, and process-start
  MPI initialization overwrites the half-2 follower noise curve with rank-1
  state. Matching only the perturbation scalar therefore cannot establish an
  exact boundary. The authoritative source must be an uninterrupted RELION A
  run with its complete live pre-iteration-3 state sealed in place. Separate
  uninterrupted B/C runs are whole-run capture controls unless their complete
  boundary bytes independently match A.
- Existing `pass0_plan_eulers.bin` files are also invalid boundary evidence.
  The diagnostic hook read device-backed `AccProjectorPlan::eulers` through
  its host pointer without `cpToHost`; byte inspection found zeros, non-finite-
  scale garbage, and magnitudes near `1e35`. A replacement hook must copy after
  the producer stream completes and fail unless every matrix is finite,
  orthogonal, and determinant `+1`. The production trajectory itself is not
  implicated; this is a diagnostic-capture bug.
- Commit `b11e4a88` adds an atomic captured-projector replay contract. Both
  half-set `Projector::data` slabs must be complex64 and arrive together with
  their `r_max`, class count, current size, explicit padding factor, unpadded
  volume shape, and lowercase source-manifest SHA-256. The replay rejects
  partial, non-finite, wrong-dtype, wrong-shape, or live-geometry-mismatched
  state instead of silently rebuilding a projector from a resident half-map.
- Commit `1a3a9d24` adds a K=1 compact tap after the normal production
  score-to-posterior normalization. Its environment gate is independent of
  `_pass2_dump_requested_for_bucket`, so enabling it does not select the
  materialized diagnostic scorer. It preserves native scores, posteriors,
  priors, support/significance, exact rotation/translation geometry, log-Z,
  Pmax, and winner state in bounded node-local shards. Commits `5b225f2c`,
  `2d3e01a5`, and `dcd1aa07` add strict readback, immutable manifests,
  particle/candidate caps, and exact per-half identity-set sealing. A capture
  with swapped halves cannot pass by matching only the global identity union.
- CPU checks pass: the sparse pass-2 plus compact-capture suite is `28/28` in
  `52.39 s`; replay-related tests are `24/24`; projector corruption/geometry
  cases are `7/7`; and the final hardened compact-capture suite is `9/9`.
  These qualify plumbing only. No full-10,000-particle science job or
  production algorithm change is authorized until the RELION live-state hook
  emits bounded compact shards, corrected pass-0 geometry, half-specific
  noise/scale/metadata, and an immutable verified manifest that RECOVAR can
  consume exactly.

The pre-registered no-launch plan is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it23_full10k_score_posterior_9b3c737c_20260717T231518Z/plan/RUN_PLAN.md`.
Intermediate comparisons remain exact-array/score/posterior diagnostics;
map-quality acceptance remains shellwise FSC/FSC-AUC only, never correlation.

# 2026-07-18: autonomous matrix exposes final-boundary and schedule subfamilies

- The autonomous current-head K=1 matrix root is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_current_head_autonomous_full34_fulltraj_paired_5592a009_20260718T135707Z`.
  Completed strict passes currently include cases 11, 15, 16, 17, 19, 21,
  and 27. Case 27, with 70% extreme outliers, passes all ten numbered maps
  (minimum merged FSC-AUC `0.999983260752`), intermediate topology, and final
  merged FSC-AUC `0.996766096548`; its RECOVAR-minus-RELION merged GT FSC-AUC
  is `+0.010652529734`. High outlier fraction alone therefore does not explain
  the residual failures.
- Cases 12, 13, 14, 24, and 25 have matching numbered topology but fail only
  the final product. Their final merged cross-engine FSC-AUC values are
  `0.955972641` (original case 12), `0.993514345`, `0.994251441`,
  `0.993444834`, and `0.993954551`. Case 14's value is from the fixed no-CTF
  rerun below. Do not weaken the `0.995` final gate; use controlled final-only
  substitutions and aggregate score/posterior diagnostics.
- Case 13's final-only factorial selects inherited poses/translations. A
  native repeat remains near `0.99349`; RELION final sampling/sigma remains
  near `0.99348`; substituting only the RELION last-numbered poses/translations
  raises final merged FSC-AUC to `0.996893352093`. Repeat variation is about
  `2.6e-5`, more than 100 times smaller than the pose rescue. The affected
  pose state is a roughly 1--2% tail rather than a broad shift.
- Case 12's strict four-arm factorial is under
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case12_final_map_factorial_f7091660_20260718T105300Z`.
  Native repeat and RELION-reference-only arms pass the numbered repeat gate
  and finish at `0.955972997009` and `0.966164947053`. Pose-only finishes at
  `0.986305085277` but its independent numbered repeat misses the deliberately
  tight `0.99999` control gate. The combined pose-plus-reference arm passes
  the numbered gate at `0.999999996372`, passes intermediate topology, and
  raises final merged FSC-AUC to `0.9952237096`. Thus both inherited pose and
  half-reference state are causal in this very-high-noise final boundary.
- Case 18 has a real resolution-schedule split. At iteration 11 shell 34,
  RELION FSC/DVP is `0.494797/0.979401`, while RECOVAR is
  `0.501733541/1.006958246`. That changes iteration-12 current size from 98 to
  100, advances RECOVAR to Healpix 6 one iteration early, and explains the
  later one-numbered-iteration convergence mismatch. Fix the earlier map/FSC
  state, not the downstream convergence counters.
- Case 20's iteration-8 size split is likewise real. At iteration 7 shell 16,
  RELION FSC/DVP is `0.500753/1.003016`; RECOVAR is
  `0.499684155/0.998737395`, producing current size 52 versus 50. Replaying
  the saved RECOVAR BPref in production float32, canonical/reverse order, and
  canonical/reverse float64 changes FSC by only `5.32e-8`, roughly 17,900
  times smaller than the engine gap. This excludes downstream reduction order,
  precision, growth formula, and quantization; trace upstream operands,
  geometry, or accumulated search state at aggregate/distribution level.

# 2026-07-18: identity-CTF expected accuracy fixed; final gap remains separate

- Commit `422ec992` reads RELION's authoritative `rlnDoCorrectCtf`, maps the
  data-STAR CTF rows into expected-accuracy order, and avoids constructing a
  RELION `CTF` when correction is disabled. Previously the simulator's
  identity sentinel `Q0=-1` reached `CTF::setValues` and threw, leaving
  infinite expected accuracy, runaway Healpix refinement, and no convergence.
- The authoritative fixed trajectory root is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case14_noctf_accuracy_fix_25a9837d_20260718T150519Z/case14_small_noctf_3k_g128_white_noise3_bf80_local_gpu2_authoritative_20260718T152451Z`.
  RELION and RECOVAR now both converge at numbered iteration 11; intermediate
  topology and particle distributions pass; all numbered maps pass with
  minimum merged FSC-AUC `0.999592170105`. The final merged FSC-AUC remains
  `0.994251441431`, with GT delta `+0.017572017703`, so the targeted no-CTF bug
  is closed while case 14 remains in the common final-boundary workstream.
- Combined focused validation is 94 passing unit/binding tests. A deliberately
  stale pre-fix shared binding fails the new negative-Q0 regression, while the
  rebuilt binding used by the trajectory passes; retain this provenance
  distinction in future test reports.

# 2026-07-18: corrected autonomous K=4 first map failure is iteration 8

- The corrected autonomous K=4 100k root is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_100k_gui_cc_nodisc_autonomous_c390f8bf_20260718T110620Z`.
  Iterations 1--7 pass strict schedule, dispatch, convergence, class matching,
  and per-class FSC gates; iteration 7's minimum matched class FSC-AUC is
  `0.995999436847`. Iteration 8 is the first map-quality failure, not the first
  exact scalar-state difference.
- Iteration 8 identity-matched class FSC-AUC is
  `[0.996440627563, 0.995262998531, 0.994601216732, 0.996054978465]`.
  Schedule, current size, resolution, dispatch ownership, priors, convergence,
  and class matching remain exact.
- The original iteration-8 audit compared RECOVAR's optimizer `ave_Pmax` with
  the arithmetic mean of RELION's per-particle Pmax column. That is not
  RELION's optimizer state. The authoritative `rlnAveragePmax` in
  `run_it008_model.star` is `0.922993`, versus RECOVAR `0.922998`; the delta is
  `5e-6` and both display as `0.9230`. RELION's model scalar and particle-column
  mean differ by `6.90e-5` to `4.07e-4` across iterations 2--15. The corrected
  v3 audit therefore has exactly one failure: class-3 full-grid FSC-AUC. The
  source contract is `ml_optimiser.cpp:5294` for aggregation and line 5689 for
  scheduling.
- With the model scalar used consistently, the earliest exact optimizer-Pmax
  mismatch in the pre-fix trajectory is iteration 2: RECOVAR `0.069386` versus
  RELION `0.069454`, delta `-6.8e-5`. Pre-fix RECOVAR-minus-RELION deltas for iterations 3--9 are
  `[-3.1e-5, -6.11e-4, -1.49e-4, -3.61e-4, -1.46e-4, +5e-6, -1.71e-4]`.
  These values do not change the already-matched size/convergence decisions,
  but they are real scalar-state differences and must remain visible. The
  particle-column mean is a separate distribution diagnostic, not a substitute
  scheduling oracle.
- A subsequent source audit found that RECOVAR's own pre-fix scalar was also
  the wrong optimizer quantity: it averaged raw Pmax over both halves and
  divided by particle count. RELION divides half 1's Pmax sum by half 1's
  retained M-step posterior mass, then broadcasts rank 1's scalar to both
  halves (`ml_optimiser.cpp:5294`, `ml_optimiser_mpi.cpp:4133-4138`). Commit
  `8e7ce8af` implements that exact workflow, records the denominator trajectory,
  and passes asymmetric-half regression tests. The correction does not explain
  the iteration-8 map failure: the compared Pmax values remain on the same side
  of every active current-size threshold and no earlier convergence split occurs.
- Class 3 remains above `0.9999250` normalized FSC-AUC through RELION's
  reported shell 26 and `0.9986630` through the current-size radius 35; its
  beyond-radius FSC-AUC is `0.9930675`. Shells 27--35 contain about `1.83e-4`
  of the map's Fourier energy and shells 36--126 about `1.10e-5`, but the
  predefined FSC gate weights shells rather than energy and remains unchanged.
  The two RECOVAR half-map payload arrays are exactly equal for every class;
  differing MRC hashes for classes 3 and 4 are header-only.
- Corrected evidence is
  `analysis/corrected_trajectory_iteration_008_audit_v3.json` under the run
  root (SHA-256 prefix `a278e11e`) and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it8_aggregate_boundary_audit_20260718T155100Z/aggregate_iteration_008_audit_v2.json`
  (SHA-256 prefix `eb5db2c1`). Localize the iteration-7-to-8 class-3 aggregate
  boundary; do not lower the `0.995` map gate and do not use the particle-column
  mean as a scheduling oracle.
## 2026-07-18 frozen-boundary v3 test-device incident

A broad `pixi run pytest` command was started without forcing the CPU. JAX
initialized local CUDA and briefly allocated about 41 GB on physical GPU 0,
contaminating the concurrently running K1 case-19 measurement. PID 835601
exited and case 19 was scheduled for retry. All subsequent v3 tests use both
`CUDA_VISIBLE_DEVICES=''` and `JAX_PLATFORMS=cpu`; v3 science launch remains on
hold pending schema/source-closure review.

## 2026-07-18 frozen-boundary v3 provenance scope

The source capture provenance records instrumented RELION commit `d5398ed`
(`Isolate oversized parity candidate shards`), binary SHA-256 prefix
`916a301b`, RELION 5.0.1, CUDA 12.6, GCC 11, and an A100-SXM4-80GB with UUID
`GPU-add27088-5e0d-a3a0-eb77-c7c8ed03881f`. The proposed `d476e6f` value is a
declared clean base, not the verified instrumented capture commit. Schema v3
therefore labels command/base-build fields as declared and explicitly marks
source/runtime hardware-toolchain identity cross-device-unverified. Until a
future arm seals both endpoints, this harness cannot classify residuals as
same-device equivalence or numerical noise.

# 2026-07-19: K=4 same-GPU float64 is trajectory-sensitive but not a parity fix

- Science job `11361629` ran production and genuine float64 sequentially on
  A100 UUID `GPU-27d0dd53-0c19-7be3-82f4-eaba66bb35aa`; audit job `11374498`
  verified exact current-size/order topology through eight numbered global
  iterations. The exact-local normalization path is not entered.
- Production first fails the direct per-class RELION map gate at iteration 8,
  class 3, with FSC-AUC `0.994738857112`. Float64 reaches
  `0.994700770508`, worsening the gap by `3.81e-5`; it is not an authorized
  production repair. RECOVAR-minus-RELION GT FSC-AUC remains within roughly
  `1.21e-4` in both arms.
- Precision is nevertheless a real trajectory perturbation: production versus
  float64 direct map FSC-AUC has minimum `0.997201977521`, hard-class
  agreement has minimum `0.99729`, Pmax MAE has maximum `0.00546038`, exact
  significant-count fraction has minimum `0.7157`, and class-mass relative L2
  has maximum `0.000341639`.
- This is a pre-local-fix `c390f8bf` precision diagnostic, not current-head
  quality acceptance. Continue at upstream accumulated reference/posterior
  state and retain the `0.995` per-class map gate.
- Sealed evidence:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_100k_samegpu_prod_f64_it8_c390f8bf_20260719T002650Z/analysis/samegpu_pair_audit_v1/FINAL_AUDIT_SEAL.json`
  (SHA-256 `522505c2c16e1642db4b24a0b7b23bd36bf7e42074aa2794eb3142f5b5335673`);
  rebuilt 25-entry manifest SHA-256
  `71f6c9464a5341ddf74e4b779d0c64ed20cb6a36f854fda3816891228f1fa973`.
  Correlation is not computed.

# 2026-07-19: native case-20 pre-scatter residual is operand generation

- The physical-iteration-4/half-1 RELION pre-scatter capture is inert on the
  same A100 used by RECOVAR: captured versus stock RELION merged FSC-AUC is
  `0.999999999710`; RECOVAR versus captured RELION is `0.999997240709`.
- Exact common-support matching covers 9,169 rotations and 9,716,168 RELION
  pixel rows. Rotation matrices are bit-exact (`max_abs=0`), all emitted rows
  lie inside the RECOVAR window, and every row has positive RECOVAR weight.
  Only three particles differ in contributor membership; aggregate positive
  counts are 9,171 RECOVAR and 9,172 RELION.
- Native common-row operand relative L2 is `0.0238642` for data and `0.0221334`
  for weight. Explicit RECOVAR downcasts leave those discrepancies unchanged,
  rejecting complex128/float64-versus-complex64/float32 representation alone.
- One common canonical complex128/float64 replay makes geometry-only
  accumulators exactly equal and produces map FSC-AUC `1.0`. Operand-only
  replay retains data/weight accumulator relative L2 `0.0221842/0.00564523`
  and map FSC-AUC `0.999792368507`. The RELION inputs remain promoted captured
  float32 values, not genuine recomputed float64 operands.
- Classification: upstream contribution membership plus operand generation,
  not scatter geometry or reduction order. Next compare posterior,
  scale/correction, CTF, and shifted-image factors distributionally on the
  full common subset; do not chase individual particles.
- Sealed result:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_it4_relion_prescatter_local_same_rec_gpu_cap1tb_retry_20260719T091600Z/analysis/SEALED_RESULTS_V1.json`
  (SHA-256 `cefa46b5ab1947859d6c3086d6ae782a1efca6d9997c8765b11a07c2b055695c`).
  Full comparison SHA-256:
  `8e7a9e0a747bae5e3b420682fe1198deb31d5b9bff4a5a7ada1fd4c530058c0b`.
  FSC/FSC-AUC only; correlation is not computed.

# 2026-07-19: case-25 references alone do not reproduce autonomous drift

- In a strict replay with exact per-iteration RELION non-reference state,
  substituting RELION half references only at scoring iteration 7 changes
  cross-engine merged FSC-AUC from `0.999999999868` to
  `0.999999999899`; direct arm FSC-AUC is `0.999999999894`. Pmax p95 is
  `1.404e-4`, support counts and pose/translation p95 are exact. The effect
  reverses at iteration 8 and does not propagate.
- A scoring-iteration-2 reference-only probe changes cross-engine FSC-AUC by
  `+1.925e-11`, direct arm FSC-AUC is `0.999999999925`, and the particle-state
  effect remains control-scale and non-propagating through iteration 3.
- Autonomous iteration 7 is materially larger (`0.999995965764` map FSC-AUC,
  Pmax p95 `0.0120212`). Exact non-reference state closes it by four to eight
  orders while resident RECOVAR references remain, so accumulated
  pose/posterior/correction state interacting with references is required.
  References alone are not causal in this exact context.
- Next use a bounded aggregate state/reference factorial at the first
  autonomous boundary; do not chase individual particles.
- Report:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case25_it7_relref_ab_20260719T111500Z/CASE25_REFERENCE_SUBSTITUTION_CLASSIFICATION.md`
  (SHA-256 `949e13f9d241709edd651ac3da68709a4fc8cb2b05f9b7c6123de36710cc3ce5`),
  with 11/11 manifest entries verified. FSC/FSC-AUC only; correlation is not
  computed.

# 2026-07-19: case-20 residual is concentrated in posterior support

- Across all 9,169 common half-1 rotations and 9,716,168 pixel rows, removing
  one fitted scalar per rotation lowers complex-data relative L2 from
  `0.0238642` to `0.00145241`; a pixelwise weight-ratio control gives the same
  residual. Posterior/contributor mass is dominant over pixelwise factors.
- The top `0.1%` of rotations carry `0.999996` of residual energy. Stack 1969
  alone carries `0.993009` of data and `0.996270` of weight residual energy
  and has RECOVAR/RELION contributor counts `3/2`.
- RECOVAR retains rotations 117318, 117319, and 119525 with masses
  `0.00793385`, `0.70557328`, and `0.28549708`. RELION's oversample 5 exactly
  matches rotation 119525 and carries `0.9971559`; the dominant 117319 child
  comes from a parent absent from RELION's selected fine list.
- Audit pass-1 parent/significance and pass-2 child scores with f32/f64 controls;
  do not change the global threshold or reduction topology.
- Evidence:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_it4_relion_prescatter_local_same_rec_gpu_cap1tb_retry_20260719T091600Z/analysis/common_operand_upstream_decomposition_v1.json`
  (SHA-256 `7929a9e14c3f311200a279af4706c272d27cf76d3ab85a7745fae2ae4b100041`).

# 2026-07-19: case-25 accumulated non-reference state is causal

- A qualified autonomous-prefix repeat envelope through iteration 6 has
  maximum map `1-FSC-AUC` `7.11e-11`, Pmax p95 `4.18e-5`, exact support, and
  zero pose/translation p95.
- At iteration 7, exact non-reference state closes cross-RELION merged FSC-AUC
  to `0.999999999507`; accumulated RECOVAR non-reference state remains at
  `0.999999104446` even with exact RELION references. All RECOVAR state is
  `0.999999105247`, and the two accumulated-state arms compare at
  `0.999999999848`.
- The first singleton results reject poses alone (`0.999999999901`) and show
  image/scale as a minority contributor (`0.999999952094`). Complete the
  tau/noise, direction-prior, sigma-offset, scheduler/state, and leave-one-out
  arms before selecting a source change.
- Sealed parent report:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case25_it7_autonomous_prefix_factorial_20260719T095858Z/CASE25_IT7_FACTORIAL_CLASSIFICATION.md`
  (SHA-256 `5f6419542f9cfae362a159862146bea29c7edce77e61b7b9774d042e8f6f9295`).
  Map gates use FSC/FSC-AUC only.

# 2026-07-19: factor-v2 postflight permits particle-local support

- The directory validator falsely required identical fine-orientation count,
  significance threshold, and posterior normalizer across selected particles.
  Those are particle-local in adaptive refinement. Validate each against its
  own arrays and reserve cross-particle equality for run-wide geometry/policy.
- The generic inertness audit was hard-coded to iteration 1 and one reference
  key schema. It now accepts an explicit iteration and both sealed reference
  layouts.
- The real 32-particle panel validates at orientation counts 8--104 and
  accepted counts 11--345. Same-A100 iteration-4 half-map capture/control
  FSC-AUC is `0.999999999358`/`0.999999998802`; all accumulator comparisons
  remain inside the independent repeat envelope.
- Reports:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_it4_bpref_factor_v2_panel32_same_gpu_20260719T101000Z/analysis/relion_factor_validation_v2.json`
  (SHA-256 `62c3d9c965dbab59ae9ab184563b788522fe11ea8e691bd94e30a114127f0a4b`)
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_it4_bpref_factor_v2_panel32_same_gpu_20260719T101000Z/analysis/capture_inertness_v2.json`
  (SHA-256 `532c3a5a7bbdfb6411f81cc9aeda8dc7c574f5555890e1f97b609b259699dbfb`).

# 2026-07-19: RELION unfiltered halves retain the real-space corner mask

- RELION's `BackProjector::reconstruct(do_map=false)` omits the tau2 prior,
  but `windowToOridimRealSpace` still always calls `softMaskOutsideMap`.
  RECOVAR had explicitly disabled that mask for solvent-FSC inputs and final
  `*_unfil.mrc` products.
- Replaying the exact captured RELION half accumulators with RECOVAR's mask
  disabled gives FSC-AUC `0.8384024` and `0.8352665`; enabling the mask gives
  FSC-AUC `1.0` for both halves (minimum shell FSC at least `0.9999999`).
- The exact RELION stage oracle also shows decentered weights are bit-exact,
  radial floors agree within `1.5e-14`, and divided Fourier data agree within
  `1.5e-17`, localising the difference solely to the missing corner mask.
- Evidence root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case25_relion_final_halves_continue_20260719T145200Z`.

# 2026-07-20: K=1 post-M-step state dropped the current-size boundary shell

- Case 33 first differs in current-size topology at numbered iteration 3:
  RECOVAR uses `[56, 68, 98]`, while RELION uses `[56, 68, 100]`.  At the
  preceding size-68 boundary, RECOVAR FSC shell 34 is `0.9487659` and its
  derived data-vs-prior is `18.51823997`; RELION records FSC `0.948759` and
  `rlnSsnrMap=18.515718`.  Shell 35 is unavailable in RELION.
- The old K=1 post-M-step code zeroed data-vs-prior from
  `current_size // 2`.  Pinned RELION commit
  `f2c1a384400aec37dc6805856a5ba645650a44f1` preserves the boundary and
  zeros only from `current_size / 2 + 1`; see
  `/scratch/gpfs/GILLES/mg6942/relion/src/ml_optimiser_mpi.cpp` in the
  gold-standard FSC truncation.  Commit `7f5f7584` implements that inclusive
  contract for corrected K=1 scheduling, K=1 post-M-step state, and K-class
  state through one helper.
- Deterministic saved-FSC replay changes exactly the disputed case-33
  decision: the old state resolves at shell 33 and schedules 98; the corrected
  state resolves at shell 34 and schedules 100.  A matrix-wide replay finds
  11 affected decisions in seven cases.  Only later decisions are causal:
  cases 2 and 3 schedule 162 rather than RELION's 164 from size 100, and case
  33 schedules 98 rather than RELION's 100 from size 68.  Cases 1, 6, 29, and
  30 are unchanged because `--firstiter_cc` supplies the first-iteration
  ini-high resolution state.
- Clean detached-checkout Slurm replay `11438037` completed `0:0` on H100
  `della-h20g2` in 4,265 seconds of science wall time.  Its fail-closed JSON
  records RECOVAR and RELION current sizes `[56, 68, 100]` and
  `matches_relion=true`; iteration 2 reports shell 34 and schedules
  `raw=100`, `quantized=100`.  This accepts the boundary schedule, not full
  case-33 FSC-AUC.  Grid correction and forced after-max final were unset.
  Audit:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case33_boundary3_currenthead_7f5f7584_20260720T204500Z/audit/schedule_replay.json`
  (SHA-256 `a525d47a2d6c7ee02900fb57fbae2ce5aa4f6e7ab69f4a2c9ff908da716a4ea0`).
- Focused matrix-boundary validation is 5 passed.  The complete EM-targeted
  unit selection is 345 passed in 1,151.42 seconds.  JUnit SHA-256 values are
  `b430642d54cef0b0018b7a0a7d33bc0f270b3c8632d9fcad555f7eb7652b98fc`
  for the focused matrix selection and
  `8bcd27209d6286f101b65cf61307a51dba765003e66c1557c114913f90a50593`
  for the full targeted selection.
- Evidence:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_case33_boundary_fix_20260720T203400/TEST_AUDIT.md`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_case33_boundary_fix_20260720T203400/MATRIX_BOUNDARY_REPLAY.md`.

# 2026-07-20: case-7 prefix drift is an ambiguous-particle trajectory

- A read-only audit aligns all 100,000 particles by exact `rlnImageName` over
  the topology-identical RELION iterations 1--11. Current size and HEALPix
  order are exact throughout the selected prefix.
- The particle state is not exact. Support-count mismatches increase from 72
  at iteration 2 to 536 at iteration 4, 2,166 at iteration 10, and 2,731 at
  iteration 11. Pmax absolute p95 increases from `0.000455` to `0.007820`,
  `0.028327`, and `0.033425`. Pose-error incidence above 0.1 degree increases
  from `0.020%` to `0.140%`, `0.912%`, and `1.105%`.
- At iteration 11, pose p99 is `1.844918` degrees and translation p99 is
  `0.973781 A`. The tail is half-symmetric: pose-error incidence is `1.0957%`
  for half 1 and `1.1143%` for half 2. This rejects a one-half ownership or
  dispatch artifact.
- The tail is concentrated in RELION `Pmax < 0.5` particles. Their pose-error
  incidence rises from `1.079%` at iteration 4 to `4.256%` at iteration 10
  and `5.033%` at iteration 11; higher-Pmax iteration-11 cohorts are only
  `0.044%`--`0.373%`. The largest absolute Pmax-error 5% at iteration 10 is
  not predictive of the next pose tail (`0.976x` enrichment), while a support
  mismatch is only weakly predictive (`1.609x`).
- The iteration-11 FSC threshold crossing is therefore a downstream amplifier
  of a broad low-confidence posterior/support trajectory, not a scheduler or
  shell-reduction bug. Keep FSC=0.5 and the sampling schedule unchanged. The
  next fixed-state capture should stratify `Pmax < 0.5` iteration-10 particles
  into pose-tail and pose-stable controls and compare candidate score margins,
  priors, posterior normalizers, and support cutoffs.
- Report:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_case7_prefix_particle_audit_20260720T213000/case7_prefix_particle_audit.json`
  (SHA-256 `af79c2598ad46b6f2176b57645acc2f16f440ead6a1267774950838a6a424852`).
  Interpretation:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_case7_prefix_particle_audit_20260720T213000/INTERPRETATION.md`
  (SHA-256 `3ce5039efcaa3812ca1e8692c6f756559bbc7044c0609976fdc48850666f7bb3`).
  Correlation was not computed; this audit is diagnostic/non-gating.

# 2026-07-20: case-7 physical-iteration-11 stratified discriminator prepared

- Active hypothesis before submission: accumulated incoming state/reference
  changes the physical-iteration-11 local score surface for ambiguous
  particles. The alternative is pure same-input GPU near-tie arithmetic.
- The deterministic panel contains six low-Pmax pose-tail particles and six
  Pmax/support-matched pose-stable controls from each half, 24 total. The two
  arms share one source commit, fixture, exact current-size/HEALPix schedule,
  H100 allocation, and production significance/local-posterior capture path.
- Slurm job `11439493` is running the resident and exact-RELION arms
  sequentially on H100 `della-h20g3`. It imports RECOVAR from detached commit
  `77bcf3bd7f45760ab0671c4883d91a453d58113a`, uses an isolated CUDA artifact,
  leaves grid correction unset, and skips final all-data.
- Exact state/reference substitution closing the tail/control margin split
  will retain the locus in accumulated upstream state/reference. The same
  split in both arms will instead direct the next audit to common-input score
  arithmetic.
- Selection:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_it11_stratified_posterior_77bcf3bd_20260720T214500Z/selection/panel24.json`
  (SHA-256 `0f441257af9b1152d6bf1eb2126960479826656bfafa3da1b0fb90b514d4dd2b`).

# 2026-07-20: active full case-33 and g384 fail-closed acceptance

- Full case-33 science job `11440100` runs clean pushed commit `7605c1b0` on
  H100 `della-h20g2`, reusing the immutable 400k fixture/RELION oracle. It is
  autonomous to convergence, with grid correction, forced after-max final,
  and forced current sizes unset. Iteration 1 completed in 832.4 seconds and
  scheduled `56 -> 68` exactly. Iteration 2 completed in 1,234.1 seconds,
  retained `res_shell=34`, and scheduled the causal `68 -> 100` boundary
  exactly (`raw=100`, `quantized=100`). This accepts the corrected autonomous
  schedule boundary; full convergence/final FSC/FSC-AUC acceptance remains
  pending.
- A read-only provisional audit of complete numbered rows 1--2 reports merged
  cross-engine FSC-AUC `0.999999999967`/`0.999999989583` and merged GT deltas
  `+1.5702e-8`/`+8.6502e-8`.  Its nonzero status is the expected incomplete
  `RECOVAR=2 RELION=14` topology, not a numeric gate failure.  Method note:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/k1_case33_provisional_it2_20260720T225300/PROVISIONAL_AUDIT.md`
  (SHA-256 `d070a5c58af2b49d87d85d09adaadbb709d583d4b7f0baf16a790b2d9eed84d1`;
  exact input-manifest SHA-256
  `36974a77ecfe360e756fca54864ce08e72db12182b156b4cb868c50a3b8be6a4`).
- Case-33 dependent audit `11440102` gates shellwise FSC/FSC-AUC and
  intermediate topology. Sealer `11440295` additionally requires exact Git,
  H100, convergence, final-path, grid-off, exit-status, and submission-hash
  provenance. Run root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case33_full_currenthead_7605c1b0_20260720T222000Z`.
- Case 9 science/audits/sealer are `11432807`, `11432810`/`11432811`, and
  `11440427`; case 10 uses `11421265`, `11421266`/`11421267`, and `11440428`.
  The common `afterany` sealer fail-closes on all upstream exit states, FSC
  gates, shellwise-curve readability, topology, convergence-only final,
  grid-off execution, clean source provenance, and exact paired GPU UUID.
- Read-only real validation on completed 400k case 34 passed: 162 shellwise
  curves reopened, final merged cross-engine FSC-AUC `0.9957574121481196`,
  and GT FSC-AUC delta `+0.0028692403080708972`. Sealer root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k1_single_case_acceptance_sealer_20260720`.
  Python SHA-256:
  `78a768df2eec4da35bb5a6b30963ae78d9f006317397287cf44fdd4bf9a7563c`.
- These jobs are active or dependent, not accepted results. Map quality uses
  shellwise FSC/FSC-AUC only; correlation is not computed.

# 2026-07-21: old-head full-34 matrix is durably rejected

- Durable-ledger job `11385656` completed `0:0` for all 34 old-head
  (`ac5177d2`) rows. It records 31 completed/three failed science jobs, 20
  trajectory passes/12 failures/two errors, and 25 intermediate passes/seven
  topology mismatches/two errors.
- Original sealer `11385657` failed before execution because its submission
  omitted the pinned script hash. Repair `11444630` then exposed a stale
  pre-graph-repair summary ID in the launcher. Graph-repaired job `11444736`
  binds the actual summary `11385655` and ledger `11385656`; its status `2`
  is the intended scientific rejection, not an infrastructure failure.
- Canonical seal SHA-256 is
  `819c532884408cca35de9ea3ed43c0e516d2be822d23cb7bd14d76c54da9d9e2`;
  ledger SHA-256 is
  `b79c18e5feb368782ef3a9fd439413bc3d1f890bcf19d1f97dc23037d09d97f1`.
- Cases 2, 3, and 33 carry exactly the old dropped-boundary-shell topology
  error fixed by `7f5f7584`. The other old-head failures remain negative
  evidence and are not silently attributed to that fix. Cases 9 and 10 are
  being rerun under separate fail-closed low-memory acceptance chains.
- Full audit trail:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_guigrid_localhighshell_full34_autonomous_ac5177d2_20260719T174000Z/provenance/FULL34_DURABLE_NEGATIVE_ACCEPTANCE_20260721T0045-0400.md`.
  Grid correction and forced after-max finalization were unset; correlation
  was not computed.

# 2026-07-21: case-33 iteration 6 remains inside FSC gates

- Science job `11440100` completed iteration 6 at size 128/8.63 A and entered
  iteration 7 at the same size, exactly matching RELION topology.
- Provisional half-1/half-2/merged cross-engine FSC-AUC is
  `0.9999992518331595`/`0.9999993072698271`/`0.9999996330509999`; merged GT
  delta is `-4.163994294370532e-6`, and worst merged non-DC shell FSC is
  `0.9999969085087222` at shell 62.
- Status 2 is solely incomplete live topology (`6` versus `14` numbered
  rows). All numerical rows pass. The 25-input manifest SHA-256 is
  `6dd1be6ba086f11ae81a679eee5141c583c13cf24bda637d4bc8530136304e14`.
- Sealed root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/k1_case33_provisional_it6_20260721T004800`.
  Terminal FSC/FSC-AUC acceptance remains pending; correlation was not
  computed.

# 2026-07-21: case-9 low-cap retry crosses iteration-11 OOM

- Default-memory job `11415206` failed in iteration-11 half 2 after selecting
  cap 8707 and requesting 8.24 GiB. Low-cap retry `11432807` completed the
  same half at cap 3879, completed iteration 11 in 419.1 seconds, and entered
  iteration 12 at size 212.
- Across the ten complete shared numbered rows, iteration-10 merged
  default-versus-low-cap FSC-AUC is `0.9998737350071399`, worst merged shell
  FSC is `0.9991935318010615`, hard-assignment mismatch is 2039/100000, and
  combined-noise relative L2 is `1.2366320110209453e-5`.
- Ten-row runtime sums differ by `+0.2182%` across different H100 UUIDs. The
  successful arm bundles `cuda_malloc_async`, 64M row pixels, and a 2 GiB
  large-JIT cap, so this is not single-control attribution.
- Sealed root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/case9_lowcap_prefix10_compare_20260721T005600`;
  200-input manifest SHA-256
  `35487dd427726a6ba09dc843f4db136f23e2d7ca2a47ee3ec9317e062ffd85ff`.
  Terminal FSC/FSC-AUC gates remain pending; correlation was not computed.

# 2026-07-21: case-10 completes terminal half-1 M-step

- Science job `11421265` completed final all-data half 1 at size 384 on A100
  `GPU-4bccbe72-c64a-5f5f-1fa8-ecf0bf6acf37` and proceeded to half 2.
- The `(771, 771, 771)` x-half BPref M-step ran at tail cap 60, completed
  49,878 chunks/49,933 particles in 4,964.2 seconds, then completed host
  Hermitian enforcement and both accumulator repacks. Total half-1 wall time
  was 6,693.4 seconds.
- The half-1 manifest SHA-256 is
  `fead22d62f6e7302e7b931f3e10269f6df87fd65382cf7ccb3a26b43de9502b6`.
  Durable checkpoint:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case10_xhalf_tail_lowcap_accept_9d172278_20260720T164557Z/provenance/CASE10_TERMINAL_HALF1_MSTEP_CHECKPOINT_20260721T0113-0400.md`.
- This accepts the bundled low-cap half-1 memory boundary only. Half 2,
  terminal FSC/FSC-AUC, and the fail-closed sealer remain pending; correlation
  was not computed.

# 2026-07-21: case-9 iteration 11 passes FSC gates

- The first numbered maps past default job `11415206`'s OOM boundary pass:
  half-1/half-2/merged cross-engine FSC-AUC is
  `0.9997908657983370`/`0.9998123118852400`/`0.9998832836013437`, and merged
  GT FSC-AUC delta is `-0.000014794080971658463`.
- Worst merged non-DC shell FSC is `0.9992117910499290` at shell 103. All 11
  frozen prefix rows pass; status 2 is only incomplete 11-versus-16 topology.
- The first live-directory staging attempt failed closed when iteration 12
  arrived. The admissible run explicitly froze 22 RECOVAR maps and seals 45
  consumed GT/RECOVAR/RELION inputs with manifest SHA-256
  `abcf195e90278997f890948fbbaf70bb3626fec9ba0e787abb905905f2de96ca`.
- Sealed root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/k1_case9_provisional_it11_20260721T010500`;
  seal-manifest SHA-256
  `ac9c9d440cdc540d7ec318ef9fd38e98162c374800652dc7f64a8df0a2426e05`.
  Terminal acceptance and single-control attribution remain pending;
  correlation was not computed.

# 2026-07-21: case-9 cap attribution remains factorial

- Exact source-function caps at the failed iteration-11 half-2 geometry are
  8,707 (190M/4GiB), 3,879 (64M/4GiB), 4,353 (190M/2GiB), and 3,879
  (64M/2GiB). The 64M knob alone reproduces the successful bundle's cap; the
  2GiB-only arm remains 12.2% larger.
- Both single knobs reduce the failed arithmetic cap, but neither has a GPU
  completion result. Do not attribute the bundled success to one knob or
  change production defaults before terminal acceptance.
- Result root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/case9_cap_decomposition_20260721T012100`;
  JSON SHA-256
  `521d9bc9a94bd8eb97d2c272b6090bc55bdc2a44aff6a44608a130df2668c73c`.
- Case-33 sealer `11440295` was also repaired from `afterok:11440102` to
  `afterany:11440102`, ensuring its existing fail-closed logic executes on a
  negative audit. No science or threshold changed. Dependency-audit SHA-256:
  `083f41579db12f73efe2bc923c75802aed3120c893a1917cad2ba217007bcf25`.

# 2026-07-21: case-33 iteration 11 remains effectively exact

- Iteration-11 half-1/half-2/merged cross-engine FSC-AUC is
  `0.9999997772584983`/`0.9999998224078882`/`0.9999998956685751`; merged GT
  FSC-AUC delta is `-0.000007236484065309412`.
- Worst merged non-DC shell FSC is `0.9999990820988931` at shell 62. All 11
  frozen numbered rows pass; status 2 is only incomplete 11-versus-14
  topology.
- Exact 45-input manifest SHA-256:
  `6b4520d7ff872606ae060da476aafd37d95e269f7206febf2c1b9fb13cee9795`.
  Sealed root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/k1_case33_provisional_it11_20260721T012700`.
- Terminal convergence/final FSC/FSC-AUC remains pending; correlation was not
  computed.

# 2026-07-21: case-9 completes terminal half-1 M-step

- Science job `11432807` converged after numbered iteration 16 and entered
  final all-data without a forced after-max override.
- On H100 `GPU-9f98ccbf-3c62-c54f-7409-7eb58845ad4a`, the full-Nyquist
  half-1 score pass completed 16,607 chunks/49,820 particles in 451.7 seconds.
- The `(771, 771, 771)` x-half M-step used tail cap 735, completed 5,648
  chunks/49,820 particles in 310.8 seconds, then completed host Hermitian
  enforcement and both accumulator repacks. Total half-1 wall time was 857.3
  seconds.
- Manifest SHA-256:
  `7c911e120b4e49ceedd7307657bc84357768709179b3aaa85b3c41dd6415af1d`.
  Durable checkpoint:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case9_bucketcap_accept_9d172278_20260720T095000Z/provenance/CASE9_TERMINAL_HALF1_MSTEP_CHECKPOINT_20260721T0151-0400.md`.
  Checkpoint SHA-256:
  `ca933eab4a07b240c0f287fd0ef1bf5ffa40799ba53561dc53de17b6d00035e8`.
- This is a bundled low-cap memory-boundary acceptance only; terminal
  FSC/FSC-AUC, the sealer, and single-control attribution remain pending.
  Correlation was not computed.

# 2026-07-21: case-7 selected tail closes directly to RELION under exact state

- A read-only exact-identity comparison closes the missing target check in the
  iteration-11 panel. In the updated `77bcf3bd` resident arm, 11/12 selected
  tail rotations and 7/12 translations differ from RELION by more than
  `0.001`; the exact-state/reference arm places all 12 rotations and all 12
  translations within `0.001`.
- Tail median absolute Pmax error contracts `107.515x`, from
  `0.02562694075012209` to `0.00023835652923584472`; the maximum contracts
  `150.670x`, from `0.15097260182571415` to `0.0010020107574463255`.
- The closure is panel-specific. Across all 100,000 particles, the exact arm
  still has rotation-geodesic p95 `1.8445827782316297` degrees and Pmax
  absolute-error p95 `0.2693974259247777`. Do not infer a general replay fix
  or relax a gate.
- Durable note:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_it11_stratified_posterior_77bcf3bd_20260720T214500Z/provenance/CASE7_EXACT_ARM_RELION_TARGET_CLOSURE_20260721T0408-0400.md`
  (SHA-256
  `4d2f28a593e0ae9e9104f54275ed2ab96fdbd4a71ff4ed91e798f707b3710823`).
- Active same-H100 state-component job `11449766` passed source/import/GPU
  preflight and entered the intended sparse iteration-1 path. Its scratch-only
  96-input analyzer passed a completed-capture smoke audit. Grid correction
  and forced after-max finalization are unset; correlation is not computed.

# 2026-07-21: case-7 exact-state benefit is real but partial at full population

- Across all 100,000 exact identities, exact incoming state/reference reduces
  the 0.1-degree rotation tail from 5,793 to 5,064: 961 resident tails close,
  232 matched particles open, and 4,832 persist. The 0.1-pixel translation
  tail falls from 6,230 to 4,963 (1,471 close, 204 open).
- Absolute Pmax error improves for 71,678 particles; the median contracts from
  `0.015977736459732023` to `0.006330115291595495`. Half 1 has 490 closed
  versus 106 opened rotation tails; half 2 has 471 closed versus 126 opened,
  rejecting a half-specific response.
- Rotation closure concentrates at uncertain RELION Pmax: the
  `0.25 <= Pmax < 0.5` cohort closes 667 and opens 59 tails, whereas
  `Pmax >= 0.75` is nearly neutral at 79 closed versus 84 opened.
- This supports the active pose-versus-map state split but is not a production
  replay fix. Analyzer/JSON SHA-256 values are
  `3af8b2b83400dee2d68395fd903098e9c83fcb8e762f77143b6a7faea62721f7`
  and
  `74161261039659161767bebee41c2898f9eb960a26a8f32fd9570d3a55c4db33`.
  Durable note:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_it11_stratified_posterior_77bcf3bd_20260720T214500Z/provenance/CASE7_FULL_POPULATION_STATE_REFERENCE_SPLIT_20260721T0419-0400.md`
  (SHA-256
  `1453b5632771f04d7201a3d36ffeb98e013c88fcebc4b76059049946035c681b`).
  Correlation was not computed; this remains diagnostic and non-gating.

# 2026-07-21: case-7 persistent winners are almost entirely unchanged by replay

- Only 1,203/100,000 stored rotation Euler rows change between resident and
  exact-state/reference arms. All 961 closed and all 232 opened outcomes change;
  4,822/4,832 persistent tails and all 93,975 stable rotations are identical.
- Exact state/reference therefore fails to dislodge the resident winner for
  99.793% of the persistent cohort, although its median arm-to-arm absolute
  Pmax change is `0.01084938645362854`.
- V2 analyzer/JSON SHA-256 values:
  `db28bf1c9372a4a8e4af5d95f76be0eea6d39fe0dc90aab45199fadad6523faf`,
  `bb0c54c8280931b4858c74a3531be0d09e85add9c5f61d24e1fa84414dd0718e`.
  Durable note:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_it11_stratified_posterior_77bcf3bd_20260720T214500Z/provenance/CASE7_FULL_POPULATION_ARM_WINNER_RESPONSE_20260721T0435-0400.md`
  (SHA-256
  `017eadfa1b1f0ea98f9096908f3f680f78ef4a380419509845030e57e042a8c6`).
- Residual posterior science job `11451167` is running on one H100 with 48
  balanced persistent/opened targets and controls; replacement audit job
  `11452890` uses `afterany` and fails closed on the 96-input target-rank v2
  schema. Original audit `11451209` was canceled without running because its
  queued Slurm snapshot was still v1. Run root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_it11_residual_posterior_77bcf3bd_20260721T042500Z`.
- This is diagnostic/non-gating; correlation was not computed and FSC/FSC-AUC
  remains the map-quality gate.

# 2026-07-21: current-head case-20/case-33 replays are cross-allocation evidence

- Case 20's canonical paired UUID is
  `GPU-b9c5d089-cde3-7f8b-717b-6f61c49ef1ae`, while current-head job
  `11435532` ran RECOVAR on
  `GPU-2ee3da91-970a-6714-84df-530aefe04a08`.
- Case 33's canonical paired UUID is
  `GPU-2ced982b-7cc9-32c2-a413-a600b1c00a1f`, while current-head job
  `11440100` ran RECOVAR on
  `GPU-49c1a223-be61-858b-49d8-d8b0347ac252`.
- Their final merged cross-engine FSC-AUC values are
  `0.9977609799825519` and `0.9727626280594356`; their GT FSC-AUC deltas are
  positive, but the runs do not satisfy the strict same-physical-GPU ledger
  invariant.
- Do not use these RECOVAR-only replays as silent case-20/case-33 replacement
  rows. Keep them as explicitly labelled immutable-oracle diagnostics unless
  paired RELION reruns are produced on the exact allocations.
- Historical v2 ledger jobs `11409642`/`11409643` remain preserved and may
  fail closed on obsolete case-9/case-10 roots. A future strict v3 may replace
  only with the newer verified same-GPU case-9/case-10 pairs.
- Durable note:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_full34_superseding_v2_9d172278_20260720T072521Z/provenance/CURRENTHEAD_REPLAY_GPU_PROVENANCE_BOUNDARY_20260721T0445-0400.md`.
  SHA-256:
  `a5ab9bdb05c2ed600d632405d0b1d7c35e0c291d96c9ffddd62a085eb2e40901`.

# 2026-07-21: v3 admits only newer same-GPU case-9/case-10 roots

- Eligibility passes for case 9 (`GPU-9f98ccbf-...`) and case 10
  (`GPU-4bccbe72-...`), with exact RELION/RECOVAR UUID triples and clean
  `9d172278` science provenance.
- Exact config/generation/CTF/pose artifacts and normalized STAR identities
  pass. The 17-particle samples have maximum absolute delta `4.7683716e-7`
  and relative L2 below `3.09e-8`; reference deltas are below `1.12e-8`.
- Cases 20 and 33 remain labelled `diagnostic_only_cross_allocation` and are
  not replacement rows.
- Eligibility JSON:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_full34_superseding_v3_33ff4287_20260721T044600Z/provenance/REPLACEMENT_ELIGIBILITY_AUDIT.json`
  (SHA-256
  `81b0d743ca3a6c0217598a31cf9f5105b19ea9f7d67f4ea82126c57616bbc174`).
- Do not build the v3 aggregate until independent jobs `11432810`,
  `11432811`, `11421266`, and `11421267` are terminal and consumed.
- Submitted builder `11452420` waits `afterany` on those four audits; sealer
  `11452421` waits `afterany:11452420`. Preterminal self-test resolved 34 rows,
  with only cases 9/10 structurally incomplete due to pending audits.
- Job registry:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_full34_superseding_v3_33ff4287_20260721T044600Z/JOB_REGISTRY.md`
  (SHA-256
  `77842f24be89f021d3f95ed6c0f7b89b15a534f1e2369003edbe99fa911f18fd`).

# 2026-07-21: residual panel now distinguishes support loss from score loss

- Analyzer v2 checks the immutable RELION iteration-11 target pose directly
  in each captured candidate set, including posterior mass/rank and
  reconstruction-support membership.
- The 0.001-degree/0.001-pixel target matcher aligns STAR rows by image
  identity. Functional smoke image 11540 proves a target-present score-loss
  path: resident ranks the target second and selects a 1.875-degree neighbor;
  exact state/reference promotes the same target to rank 1. The resident
  target/winner posterior ratio is `0.9696178825` (log gap `0.03085322066`).
- Analyzer/audit-launcher SHA-256 values:
  `5e3479b827df78dab166455ab9a2a72503d6d4db3884961fcb136b1ee181ac56`,
  `86615bdfdcc042d351d34778e8d27d3cd7e831c44c14778c65ab6282d2e992b5`.
- Amendment note:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_it11_residual_posterior_77bcf3bd_20260721T042500Z/provenance/ANALYZER_V2_AMENDMENT_20260721T0455-0400.md`
  (SHA-256
  `696740a5ec112d77f2beec6c1aa061bfeada05d9904fa4281b088d861552b011`).
- This remains diagnostic/non-gating; correlation is not computed and map
  quality remains FSC/FSC-AUC.

# 2026-07-21: completed case-7 tail mixes support loss and score loss

- In the completed 12-tail panel, exact state/reference contains and ranks the
  RELION target first for all 12.
- Resident state has one rank-1 target, seven target-present rank-2 near ties
  (target/winner ratios `0.8657448936`--`0.9859998237`), and four absent
  targets.
- Three absent targets miss only one `0.0835`-pixel translation-child step;
  one misses the rotation by `1.844585` degrees. Exact replay repairs both
  local search support and relative scoring.
- All 12 stable controls contain/rank the target first in both arms;
  candidate-support Jaccard is 1 and median posterior TV is `0.00905826257`,
  versus tail median TV `0.05901374049`.
- Analyzer/JSON SHA-256 values:
  `4ab6c895ed626e1949e423835ba67e117727a43777413b9cefc18426d98e42ed`,
  `2c779f535b372221e6739a8550c355d32f3fd8ed1d73de346f978a74e2271743`.
- Durable note:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_it11_stratified_posterior_77bcf3bd_20260720T214500Z/provenance/CASE7_COMPLETED_PANEL_RELION_TARGET_RANK_20260721T0512-0400.md`
  (SHA-256
  `8180615c7d242f192e9b9e7a6997ae46f4ae6c483e10c25e07a979933f2ced8e`).
- Do not implement a single support-only or scoring-only production change
  from this mixed cohort. Component and residual jobs remain decisive.

# 2026-07-21: state-component audit upgraded and correctly resubmitted as v3

- Science job `11449766` was not changed. Before any iteration-11 captures
  existed, its analyzer was extended to test every arm against the immutable
  RELION target pose.
- In addition to TV, winner displacement, and support Jaccard, the audit now
  reports target presence/support membership, rank, mass, target/winner
  posterior ratio, nearest-support displacement, and winner displacement.
- This directly tests `restore_recovar_poses` for support/centering loss and
  `restore_recovar_maps` for relative score/rank loss. It does not infer those
  mechanisms from arm-to-arm TV alone.
- V3 predeclares separate subcohorts from the completed resident panel: seven
  rank-2 score losses, four absent-target support losses, one rank-1 tail, and
  twelve stable controls.
- `scontrol write batch_script` proved pending audit `11450599` retained the
  stale v1 submission snapshot. It was canceled without running and replaced
  by verified v3 audit `11452889` with `afterany:11449766`.
- The same check found residual audit `11451209` stale at v1. It was canceled
  without running and replaced by verified v2 audit `11452890` with
  `afterany:11451167`. Neither science job was interrupted.
- Target tolerances are `0.001` degree and `0.001` pixel; the STAR is aligned
  by particle identity. A completed-capture target-summary smoke passed.
- Analyzer/subtype/helper/STAR/audit/contract SHA-256 values:
  `b5ea0a03c451c6d7aaf85d2a1c0961b207f3027add92be7be8dddf6f62a99309`,
  `02d94ce3d78b559bcdac3dab255789d5df12ae0ec9252fca3c6a0c83d7204f52`,
  `5e3479b827df78dab166455ab9a2a72503d6d4db3884961fcb136b1ee181ac56`,
  `022865cdc40d4d4c5813078d81f6f421f2f54949d04e4762498659ce271a9b55`,
  `1c3c5395694d34000e276bfaf8a18273287a01a8184790f9c4a7c78862046826`,
  `566a3e0da1d08b83119fe52411388bbfbf64b542725b163de54b884751439f65`.
- Original v2 amendment:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_it11_state_component_77bcf3bd_20260721T034700Z/provenance/ANALYZER_V2_AMENDMENT_20260721T0518-0400.md`
  (SHA-256
  `0658833c449617880621f9e1e249d8cf4e9048a0a279d6f49c06da0a67e1e412`).
- V3 supersession note:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_it11_state_component_77bcf3bd_20260721T034700Z/provenance/AUDIT_SUPERSESSION_V3_20260721T0526-0400.md`
  (SHA-256
  `8d94944cc288520de7340c5673c2ceb9eb96df48e53a113e6044af827d8f9d74`).
- Residual v2 supersession note:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_it11_residual_posterior_77bcf3bd_20260721T042500Z/provenance/AUDIT_SUPERSESSION_V2_20260721T0526-0400.md`
  (SHA-256
  `5b89f65cc77923b9292cb346fc47360fd4dbb8b184a33f7395b2b0560b3416bb`).
- This remains diagnostic/non-gating; map quality remains FSC/FSC-AUC.

# 2026-07-21: exact case-7 control closes cross-H100 reproducibility

- State-component job `11449766` completed its `all_relion_repeat` arm on
  `della-h19g1`, H100 UUID
  `GPU-2ee3da91-970a-6714-84df-530aefe04a08`. Independent prior exact job
  `11442740` ran on `della-h20g3`, H100 UUID
  `GPU-2dcba0de-4bea-ece2-85aa-34ebe8d3d949`.
- The new repeat contains/ranks the immutable RELION target first and includes
  it in reconstruction support for all 24 images, including each predeclared
  tail subtype.
- Prior exact versus repeated exact has support Jaccard `1/1/1`, posterior TV
  `0/0/0`, zero winner displacement, and identical latent/physical winners
  for 24/24 images under every audited representation.
- This rejects allocation/GPU-specific arithmetic as the case-7 panel cause.
  It does not yet choose maps versus poses; wait for both remaining component
  arms and the independent residual panel before touching production logic.
- New capture-manifest/refinement/wall-time SHA-256 values:
  `31cf0d14eb831eb9c022658ad0af85a137f658f92041bb21bf599b82e66f5a0b`,
  `c4978e29cbdef4e69a10fd4fc2c50ba2fa62196da9ee5f4f0fb5607d06eb51ef`,
  `781979ef966a6bff2c42914efcf42646e38a98a39dd895310f6614e67f7cb521`.
- Durable note:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_it11_state_component_77bcf3bd_20260721T034700Z/provenance/ALL_RELION_REPEAT_CROSS_ALLOCATION_RESULT_20260721T0551-0400.md`
  (SHA-256
  `8f145d75670f08e0136d253bafdc5d43630d84e31f46ed882983067857b3a45d`).
- This remains diagnostic/non-gating; correlation was not computed.

# 2026-07-21: historical K=1 v2 seal is terminal and fail-closed

- Oversized, never-started sealer `11409643` was canceled and replaced by
  verified one-CPU/4-GiB/10-minute job `11453977`; no science input or sealer
  logic changed.
- `11453977` ran on `della-i13n21` for two seconds with 56012 KiB maximum RSS.
  Exit `2:0` is the expected failing-seal result; stderr is empty and every
  artifact in the generated output manifest revalidates.
- The 34-case seal reports structural `fail` and parity `fail`. Only cases 9
  and 10 lack structural closure in this historical ledger. Earliest parity
  failure is case 2 iteration 3, merged GT FSC-AUC delta
  `-0.003274589 < -0.002`.
- This closes v2 as historical evidence only. The strict v3 replacement graph
  remains authoritative once its independent case-9/case-10 audits finish.
- Terminal JSON/Markdown/manifest SHA-256 values:
  `804ade09bfb022887cb9c6045d615127b818dff394315297b4f79a43c1dcef52`,
  `9f842fd4421dc7fd2990d7a6c01aee64fd23ee13d2799d4fd48b296a3c613de3`,
  `5a35ec9b360fddea4708439677bc6511d177cb86374d83c5f6a445f74d543a5a`.
- Durable scheduler/result note:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_full34_superseding_v2_9d172278_20260720T072521Z/provenance/SEALER_RIGHTSIZE_RESUBMISSION_20260721T0556-0400.md`
  (SHA-256
  `e4cc94f33a99976ecb51bdd7b8fdbdf1c0d6e1b25be66f8796d8dd637a5524c1`).
- Correlation was not computed or gated.

# 2026-07-21: strict K=1 v3 graph repaired after audit OOM/role discovery

- Case-9 intermediate `11432810` and case-10 intermediate `11421267`
  completed at 543884 and 622232 KiB MaxRSS. Their complete trajectory peers
  `11432811` and `11421266` hit the 8-GiB cgroup and ended
  `OUT_OF_MEMORY 0:125`.
- Audit inspection also proved the pre-existing case-9 registry fields were
  reversed: `11432810` is intermediate, while `11432811` was trajectory.
- Builder `11452420` and sealer `11452421` were canceled before running. The
  old builder accepted any terminal state and therefore was not safe against
  OOM plus stale output files.
- New trajectory retries are `11454201` (case 9) and `11454202` (case 10),
  each one CPU/32 GiB/30 minutes. The corrected registry binds case-9
  intermediate `11432810` and case-10 intermediate `11421267`.
- Corrected builder `11454286` waits on both retries; sealer `11454287` waits
  on the builder. Aggregate launchers are one CPU/8 GiB/30 minutes and their
  submitted snapshots directly verify the manifest gates.
- Builder outcome validation now rejects infrastructure states including OOM
  while permitting successful `0:0` and intentional fail-closed `2:0` exits.
- Corrected preterminal closure is 32/34, with only the two running trajectory
  retries incomplete. Static and eligibility manifests both pass.
- Durable notes:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_full34_superseding_v3_33ff4287_20260721T044600Z/provenance/V3_GRAPH_REPAIR_20260721T0610-0400.md`
  (SHA-256
  `b3dd960f88c70cd9f4caeb07ebec3d730ef465a5259de4cfedd9726cc3ad648a`)
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_full34_superseding_v3_33ff4287_20260721T044600Z/provenance/AUDIT_RESOURCE_RIGHTSIZE_20260721T0601-0400.md`
  (SHA-256
  `7036091ba125a8f98d13d2ee5e9bdb124f6ca5724ec9a4fcf6b3c44a011a7936`).
- No science artifact or FSC/FSC-AUC threshold changed; correlation remains
  uncomputed and ungated.

# 2026-07-21: strict K=1 v3 reaches terminal structural closure

- The preceding active graph is superseded. Case-9 trajectory `11454201`
  completed `0:0` in 28m11s at 8669880 KiB MaxRSS and passes final merged
  cross-engine FSC-AUC at `0.9955108928134183`.
- Case-10 trajectory `11454202` produced complete evidence and intentionally
  exited `FAILED 2:0` in 27m58s at 8665864 KiB. Intermediate status is pass;
  final merged cross-engine FSC-AUC is `0.9830065035340728 < 0.995`, while
  the final merged GT delta is positive `0.00012834695727731438`.
- The first corrected aggregate pair `11454286`/`11454287` was canceled
  without running while the near-limit trajectory results were inspected.
- Fresh builder `11454959` completed `0:0` and resolves 34/34 structurally
  complete cases. Sealer `11454960` intentionally exits `2:0`: structural
  `pass`, parity `fail`, earliest parity failure case 2 iteration 3
  (`-0.003274589 < -0.002` merged GT FSC-AUC delta).
- Both terminal artifact manifests independently revalidate with every entry
  `OK`.
- Ledger JSON/Markdown/manifest SHA-256:
  `5393ee8f1549ccce6dbf7befec7c14f66d58d16b6196ccb52eef8a70e8ddf26f`,
  `76036d28f410b55ce2a9bd5a30f524cc921d327339e9fb319af0fc80a0a74d4f`,
  `c0827618c6550e5b15eae94291985173ce9b14c97dc2beb55ea7ba801d5675ee`.
- Seal JSON/Markdown/manifest SHA-256:
  `897a21e317eb5fd77aeaf715736332c8ed0f76dcf2e3199ca8e407e425b73a51`,
  `7cc531b9fd92ca7a93590bec3cc097fd5f5652cb722ae208cdb777e4261ff3d5`,
  `3b582c96b65a23888909c219ab2a5fe419da726d8d00acaf34c8bde057e8df10`.
- Updated graph/resource/registry note SHA-256:
  `de96b2aa7437b1f996cfafe770ff19ef0ebe47c032277841f89ea6c012288984`,
  `29f6c623cb9222f3c4075fb289f7ed0d6f2e492ad04d31eb152346865b66df4d`,
  `84f711fec024ce5f77008233a9699e0e96bc63fb46ab36214437f1c04216d813`.
- This closes the evidence graph, not parity. Do not weaken the FSC/FSC-AUC
  gate or alter production logic from the aggregate seal alone; the running
  case-7 component/residual discriminators remain the next evidence gate.

# 2026-07-21: quarantine case-7 capture-target observer effect

- Exact-local parent pass 1 uses `score_only=True`, but its retained support
  drives fine pass 2. The debug target-only optimization therefore filtered a
  science-critical call and made refinement results depend on the capture
  identity list.
- The disjoint 24- and 48-target runs retained different parent-bucket sets
  despite identical source, science inputs, seed, schedule, and byte-identical
  stripped CUDA libraries. They changed 534/100000 Euler rows; rotation tails
  were 5793 versus 5620 and translation tails 6230 versus 6030.
- Supersede the prior full-population resident/exact interpretation. The
  derived residual `persistent`/`opened` panel is invalid and must be rebuilt
  from an unfiltered run. Canonical K=1 matrix science without capture
  variables is unaffected.
- Component/residual jobs `11449766`/`11451167` were canceled after 03:00:59
  and 02:24:49. Their never-started audits `11452889`/`11452890` were also
  canceled. All partial output roots have prominent quarantine markers.
- The code fix makes `RECOVAR_LOCAL_SCORE_DUMP_TARGET_ONLY=1` explicit opt-in.
  Unset/default now executes all science buckets while materializing only the
  requested target dump artifacts. Focused capture/debug tests pass 7 tests
  with 329 deselected.
- Authoritative note:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_it11_stratified_posterior_77bcf3bd_20260720T214500Z/provenance/CAPTURE_TARGET_PARENT_PASS_OBSERVER_EFFECT_20260721T0654-0400.md`
  (SHA-256
  `312594539d2b6932c86aadb727afe885417282c509e5be63ab2652b689435671`).
- Do not submit a replacement residual panel until its cohort is selected from
  a clean unfiltered full-population output. Map-quality acceptance remains
  FSC/FSC-AUC; correlation was not computed.

# 2026-07-21: clean case-7 residual panel is mixed support and score evidence

- Clean science/audit jobs `11477130`/`11477132` completed `0:0`. Both arms
  sealed 24 iteration-11 captures; the v2 audit consumed all 48 and passed the
  predeclared 8/8/4/4 persistent-target/control/opened-target/control split.
- Physical iterations 2--6 have half-1 resident/exact sparse work
  `120380/120382`, `87241/87249`, `89247/89219`, `94623/94609`, and
  `95642/95586`; half 2 is `120780/120781`, `87758/87759`, `89913/89921`,
  `94887/94879`, and `95976/95876`. The split changes sign and stays at
  size-16/32 bucket boundaries before both arms enter the same local schedule.
- Persistent targets have candidate-support Jaccard one and identical
  arm-to-arm physical winners for 8/8. Exact state moves median RELION-target
  posterior mass from `0.36655533` to `0.39146566` and median target/winner
  ratio from `0.93263455` to `0.99932882`, but seven remain rank-2 near ties.
  Persistent divergence is therefore relative score order within available
  support.
- Opened controls remain stable. All four opened targets change winner by
  median `1.8614835` degrees. Two retain identical candidate support and become
  rank-2 near ties; two have support Jaccard `0.5`, including one absent target
  and one target demoted to rank 3. This cohort mixes support and scoring and
  does not justify a single production edit.
- Audit JSON/Markdown SHA-256 values are
  `b3fc8255366b66e17ae9149c456d3fa82ce9c2320fbe08dc120aebfe7cc498f1`
  and
  `2fbed88f8fe8abf9d45c638bfd0cdc0ba53bb2d698ce25fe78eb9b8f69d654e1`.
  Root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_it11_residual_clean_fa0c93fc_20260721T082000Z`.
- The audit is diagnostic/non-gating; correlation was not computed and
  FSC/FSC-AUC remains the map-quality gate.

# 2026-07-21: K=4 failure-only audit preserves a completed target boundary

- Full science `11480333` and success audit `11480664` remain authoritative.
  Capture-instrumented RELION control runtime projects beyond the science
  job's immutable 16-hour limit, which Slurm would not extend in place.
- Failure-only audit `11481766` waits on `afternotok:11480333`. It can seal a
  separately named partial report only after verifying the complete RECOVAR
  iteration-10 target boundary, all frozen identities and hashes, and inert
  RELION inputs.
- Its status is explicitly
  `target_boundary_complete_terminal_trajectory_incomplete`; it cannot create
  `SCIENCE_COMPLETE` or satisfy the full audit. Missing target-boundary input
  fails closed. No K=4 production change is authorized by this recovery path.

# 2026-07-21: live K=4 iteration-10 gate invalidates the frozen panel

- Uninterrupted science job `11480333` reached clean-control iteration 10.
  Only 72/96 frozen targets remain in class 2; full class counts are
  7/72/14/3 and all six 16-particle categories contain at least one off-class
  identity. The largest loss is 11/16 in the predeclared
  RELION-class-2/RECOVAR-disagreement category.
- Gate JSON SHA-256 is
  `69f8358e26255d93131d92dce6d51220d8a2a2a2662c05b6c88cc17005296b5a`;
  its input STAR SHA-256 is
  `a044f3a98457954730a47f2fcabad3a45b87d8336d5fedf10fee501a39ab13d5`.
  Sidecar and exact schema/count/hash assertions pass.
- The job remains untouched and must finish the clean control, then fail its
  existing authoritative 96/96 gate before passive capture. Audit `11480664`
  cannot run after success; failure-only audit `11481766` must itself fail
  closed because the capture boundary is absent.
- Live-control versus older-canonical iteration-10 FSC-AUC is 0.954572533,
  0.949992474, 0.941647625, and 0.944810030 for classes 1--4. This is
  RELION-build trajectory sensitivity, not RECOVAR evidence. Diagnostic JSON
  SHA-256 is
  `01bd7b83bf99fe0f4e34751912199421ec9e679142a7fea8b936d257dfa7cc6f`;
  correlation was not computed.
- Replace the frozen cohort with a predeclared deterministic post-control
  selector: verify live source identities and follower ownership, seal an
  exact class-2 panel on the same allocation, then start passive capture and
  RECOVAR. Request a longer allocation. Do not edit K=4 production arithmetic
  from this negative gate alone.

# 2026-07-21: dynamic K=4 panel replacement is submitted with an import gate

- Static reservoir/selector SHA-256 values are
  `ad96d6e0112f8b0f7aa000e720ab5a38345d06939daa334d5a2530525c6ba648`
  and
  `c7f9854abf93314f5ecf0044a5d50888ea9d70567abf281611e4b6d92dfe76a1`.
  Two current-control dry runs deterministically produce 96 live class-2
  targets, 16/category and 48/48 by live follower owner.
- Jobs `11484384`/`11484385` were canceled at 1m40s/0s before selection or
  capture. From RELION's data directory the pixi Python resolved `recovar` to
  an unrelated editable checkout, violating the checkout-binding provenance
  gate. Preserve that partial root only as quarantined orchestration history.
- Fresh science/audit jobs `11484481`/`11484482` use root
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_dynamic_class2_prescatter_ac5177d2_20260722T015934Z`.
  The 24-hour launcher binds and asserts source `77a348c5` before control
  iteration 1 and again before live selection. Launcher SHA-256 values are
  `4896aac254895a9bb5828fe4badbf42b7060b0d8cbd21397cd1309d172dbbd04`
  and
  `7e33e576c477184fc0cc9a2f051c36ad8eece29bce76134483cb68ae1992cd12`.
- Science `11484481` is running on `della-l08g5`. Its pre-control gate resolves
  RECOVAR under the exact `77a348c5` source and JAX under the pixi environment;
  audit `11484482` remains dependency-gated.
- The submitted panel is still unknown by design. Only the new clean control's
  iteration-10 class winners and dispatch owners can resolve it; capture fails
  closed until the resulting manifest passes. No production edit is implied.
- Primary audit `11484482` remains the geometry/support authority. Secondary
  scalar audit `11484846` waits on `afterok:11484482` and cannot bypass it.
  Comparator/launcher SHA-256 values are
  `9a6a3b650ead568c07833a00f6c102784be2cd6b687616d84e4cf56a0b9c969f`
  and
  `48a66034daf483096ed9012bc0ce1172ad61263636f72e81e2659ae510bb22ba`.
- The scalar discriminator predeclares aggregate relative-L2 `1e-6`, then
  per-rotation positive real data/weight scale residual, phase, and scale
  consistency thresholds `1e-5`, with 95% valid/compatible fractions. A pass
  is compatible with posterior mass but is not causal proof. Synthetic
  exact-scalar, pixel-varying, complex-phase, and zero-reference checks pass.
- Current focused posterior/pruning baseline passes 7 tests with 152
  deselected under `JAX_PLATFORMS=cpu`. Do not combine the existing float32
  posterior and joint-pruning helpers unless the qualified capture supports it.

# 2026-07-21: replacement K=4 control bifurcates after iteration 2

- The v2 identity audit hashes immutable per-iteration dispatch slices, not the
  growing full log. At iteration 2, comparison with capture-capable control
  `11480333` gives 100,000/100,000 matching class and Euler assignments.
- MPI follower ownership differs for 49,780 particles, showing the agreement
  is not an artifact of identical follower partitions. Absolute Pmax
  difference has median zero, p95 `5e-6`, p99 `9e-6`, and maximum `1.21e-4`.
  Only two translations differ, each by one 2.125 A pixel.
- At iteration 3, 5,352 class labels, 12,383 Euler tuples, and 11,655
  translations differ. Classwise map FSC-AUC is 0.991678, 0.991111, 0.987813,
  and 0.971361. The same-binary cross-A100 trajectory has therefore already
  bifurcated despite its nearly exact iteration-2 maps.
- Iteration-2/iteration-3 state JSON SHA-256 values are
  `9343094d372a994e7c950f6f162d099a761413d6cb476e36a40849f3334bd0e1`
  and
  `00bd3a367356f5e40ad089468ca73c417d5df8537c019889d40200fa02025216`;
  iteration-3 map JSON/v2 helper SHA-256 values are
  `5a139b5e7d42a12084bb35f7bee717b3401f32bba407083328d17f1aec9701d6`
  and
  `bf0d2708498c3c4287093bc8727cd623bbbe47b728657b49dceecb6f70738f90`.
- This is descriptive/non-gating and strengthens the requirement for live
  same-allocation panel selection and passive capture. Correlation was not
  computed.

# 2026-07-22: frozen K=4 panel graph is terminally fail-closed

- Science `11480333` ended `FAILED 1:0` after 5h45m38s. The clean RELION
  process first returned zero and sealed all 48 iteration-0--15 STAR boundary
  files; the enclosing job then raised the expected 24-particle class-gate
  exception.
- Terminal verifier `11485567` completed `0:0` in 4 seconds. It confirms the
  exact 7/72/14/3 live counts, zero RELION/RECOVAR capture files, and absent
  science/capture-audit completion markers. JSON SHA-256 is
  `0ccb9180f5c37ebd78ac92d6c267c0db0d5a1e3e0ef7987bb189fd9f0fc926e7`.
- Oversized, never-started salvage `11481766` was canceled at zero runtime.
  Right-sized replacement `11487432` used the same pinned launcher/comparator
  and failed `1:0` in 3 seconds at the known-absent inertness prerequisite,
  creating no partial result. Salvage-rejection JSON/helper SHA-256 values are
  `b9db836e284423846918676259a7fa7e89e7f4310f887f07f8220cafe428e2b9`
  and
  `3f6cc93165f7b55df85547af8816ce2d04cc8a13641d87e7b5e175b9bce3b8a5`.
- Success audit `11480664` was canceled at zero runtime after becoming
  `DependencyNeverSatisfied`. The frozen-panel recovery graph is closed; it
  does not authorize a production edit. Correlation was not computed.

# 2026-07-22: authoritative dynamic K=4 class-2 panel is sealed

- Corrected science `11484481` completed its clean 15-iteration RELION control
  with status zero in 16,597 seconds on
  `GPU-3bae32ea-7500-d97f-68d3-b73eaf826482`. It sealed all 48
  optimiser/data/model STAR files and exactly 1,500,000 dispatch rows.
- The official selector manifest passes `sha256sum -c`. Its panel has 96
  unique live class-2 targets, 16/category and 48/48 by normalized follower
  owner. Live class counts are 25,166/23,728/24,845/26,261; live candidate
  pools are 22,215 agreement and 74 disagreement.
- Panel JSON/manifest/identity SHA-256 values are
  `7cf6ed42934460c9540b4f6a66238921e99b3b665117a5a88a66930836ab68f7`,
  `bb86ac1c3f61cb1d14e9314f9bdfb60e6a4abd09becdab55149cfaf656e66262`,
  and
  `b1c85f635cc342aded1cbe95ffac9d99e0b9ed5afb432dd85f40f9b0e0d085be`.
- Original/stack selected-index arrays are byte-identical to the isolated
  iteration-10 preview and retain SHA-256 values
  `48058423d876305cf72c23260e514a5ca982508c2791b43234e51f7a8671b489`
  and
  `d3900f382b275f529ec4365232e105a819138f2add51d1eb2a77b5a457e11105`.
- The official manifest sealed at 02:39:21 EDT and the first passive-capture
  artifact followed at 02:39:48. Science remains active; primary/scalar audits
  `11484482`/`11484846` remain dependency-gated. This authorizes the capture,
  not a production source edit.

# 2026-07-22: passive K=4 capture passes its early repeat envelope

- Iteration 1 is exact for all 100,000 class/Euler/translation/Pmax rows and
  dispatch owners; classwise capture/control FSC-AUC is at least
  `0.999999999278`.
- At iteration 2, dispatch owners and classes remain exact. Pmax absolute p95
  is `4e-6`; one diffuse-posterior Euler tuple and two translations choose
  alternate grid winners. One translation identity is the same near tie seen
  between independent clean controls.
- Classwise iteration-2 FSC-AUC is
  `0.999999984390/0.999999983143/0.999999981554/0.999999973606`, passing the
  predeclared `0.999999` map threshold. The corrected classification is
  FSC-pass with repeat-scale particle near ties, not bitwise particle replay.
- Iteration-1 state/map JSON SHA-256 values are
  `e37e3c81dc1b3ec9c08500e87a20da9fc6b5ff0dd52a4e7dc6951a91a2660626`
  and
  `127b659cc72069c26938041cc7beeff6279422859423a7db2993f0e4096462fe`;
  iteration-2 values are
  `adf3866ba64744090d28216bbe883d840560fa3f39328a7a2bf815a1cef4ea5d`
  and
  `a7ef6939bb0bb73e6e00176a7ebace85cce06514f3be9914c2906671fa2e85b0`.
- An operator-side exact-particle assertion exited nonzero only after both
  reports were sealed; science was unaffected. The formal iteration-10
  capture validator remains unchanged and authoritative. Correlation is not
  computed; no production edit is authorized.

# 2026-07-22: passive K=4 iteration 3 raises an inertness warning

- Dispatch owners, all 100,000 class labels, and schedule scalars remain exact;
  one Euler tuple and four translations differ.
- Classwise capture/control FSC-AUC is
  `0.999998957108/0.999999983889/0.999999972165/0.999999951124`. Class 1 is
  `4.29e-8` below the same `0.999999` numeric threshold used by the formal
  target validator; classes 2--4 remain above it.
- State/map JSON SHA-256 values are
  `b865b1cd370a3b042b3e407c239083d12c61d99b91ec828f9329d16c2f564f81`
  and
  `3a7c02a08e3fba57f1ece0a11106fb899cdd981bebec3df236c5e4e55baebdfc`.
- This is an early warning, not a relocated gate. Only iteration 10 contains
  the target capture and is formally authoritative. Science continues
  unchanged and must fail closed there if inertness rejects. Correlation is
  not computed; no production edit is authorized.

# 2026-07-22: passive K=4 warning amplifies through iteration 5

- Iteration 4 retains exact dispatch ownership but differs in 1 class label,
  9 Euler tuples, and 10 translations. Classwise capture/control FSC-AUC is
  `0.999987631111/0.999999768543/0.999992547236/0.999999858478`; classes 1
  and 3 are below `0.999999`.
- Iteration 5 retains an exact 100,000-row dispatch slice at SHA-256
  `8a5336e4ab89461ad4b5a9b9261d54c74dfdecaf342a5a5bc38c7ae736b44e96`,
  while the state difference grows to 8 class labels, 32 Euler tuples, and 43
  translations. Pmax absolute p95 is `0.001055` and maximum is `0.522492`.
- Iteration-5 classwise FSC-AUC is
  `0.999933587444/0.999934151441/0.999866285411/0.999992450129`; all four
  classes are below threshold. Iteration-4 state/map JSON SHA-256 values are
  `6453998fc03bb3ded0627e1fa8aab7e301d64c7d1250342b3158cc2cc8879e40`
  and
  `06a000dc22ba18c8e007ad95bcd7229da0a4ffa5dc9df6d8fe75ebd7b78deb95`;
  iteration-5 values are
  `45afd7b63fc3ce2d6d5a5b92614dc2fb96fe21ef0cd4745193e02908dc7f3036`
  and
  `3bd501e9ccd8e373b0e0b172d640c63b15a2a0eaad742a9b36c7b6339912d374`.
- This makes iteration-10 rejection likely but does not replace the formal
  target gate. The unsubmitted contingency restarts clean control and passive
  capture from the exact clean iteration-9 optimiser, with exact clean
  dispatch/order replay in both arms. RECOVAR can still use the original full
  clean run for iteration-0 state, optimiser metadata, and dispatch schedule;
  the restart capture is an independent comparison operand. Running science
  remains untouched and no production edit is authorized.

# 2026-07-22: full-start iteration-10 capture rejects; restart gate submitted

- The authoritative iteration-10 boundary sealed at 06:20:46 EDT. Capture and
  control have exact 100,000-row dispatch slices, follower ownership
  50,140/49,860, row SHA-256
  `00059e382a1a4888275fe43d801e463529f0ae07bd046cc748f06400e855fb76`,
  source-order SHA-256
  `759d64f245c4c8ffcce4c527e990c115391d6607271d4e7f75da5550c9324534`,
  and perturbation `0.096421` in both arms.
- Classwise capture/control FSC-AUC is
  `0.998247194648/0.997600525037/0.997363409604/0.998221443830`, all below
  the declared `0.999999` threshold; relative L2 is
  `0.0046840/0.00508357/0.00520915/0.00375230`. Native capture validation
  independently rejects with `ValueError: missing MPI rank identity`.
- The sealed rejection JSON SHA-256 is
  `3d5134197e3071ce5074d75bc45f5fdfcada794eacbe4bfc227e5552e14ae789`.
  Correlation was not computed.
- The restart contingency was released only after that JSON existed. Science
  job `11492718` began on A100 node `della-l07g2` at 06:24:07 EDT; primary and
  scalar after-ok audits are `11492719` and `11492720`. Both RELION arms
  restart from the immutable clean iteration-9 optimiser with exact clean
  iteration-10 dispatch/order replay. RECOVAR remains gated on capture
  inertness plus closure to the original clean trajectory.
- Original full-start science `11484481` remains untouched. No production
  edit is authorized until restart capture and dependent geometry/scalar
  audits qualify an evidence-backed mismatch.

# 2026-07-22: restart capture rank provenance corrected

- Original capture artifacts encode MPI rank as unsigned `-1`: the diagnostic
  helper read only `OMPI_COMM_WORLD_RANK`, but direct Slurm `srun` supplies
  `SLURM_PROCID`. Thus restart graph `11492718`/`11492719`/`11492720` was
  canceled after 6m39s/0s/0s before capture or RECOVAR could run. Original
  science `11484481` was untouched.
- Diagnostic RELION commit
  `4ab53edf206e9cafd993484a92eccd77e828c497` adds a strict Slurm-rank fallback
  with fail-closed validation and no reconstruction/arithmetic change. The
  isolated replacement binary SHA-256 is
  `dad0ff14a1478b22b1f3ba9acc93934341aaf7b8750205a606256f8c990ce475`.
- Replacement science `11492933` began on A100 node `della-l07g7` at 06:41:03
  EDT after an independent three-task rank probe passed exactly `0,1,2`.
  Primary/scalar audits are `11492934`/`11492935`.
- Preflight-only attempt `11492919` first failed in three seconds because
  launcher-owned empty child directories had been created too early. It wrote
  no science; `11492920`/`11492921` never ran. Removing only those empty
  children restored the unchanged launcher contract.
- The active graph retains the same immutable clean iteration-9 restart, exact
  iteration-10 dispatch and particle-order replay, sealed panel,
  original-clean closure, and same-A100 RECOVAR gates. No RECOVAR production
  edit is authorized before its dependent geometry/scalar evidence.

# 2026-07-22: restart continuation uses absolute iteration bound

- Rank-corrected `11492933` sealed unset-control iteration 10 but then entered
  expectation iteration 11. On continuation, RELION retains the optimiser's
  stored `nr_iter=15`; `--auto_iter_max 10` alone does not override that loop
  bound. It was canceled after 21m25s before capture/RECOVAR, and audits
  `11492934`/`11492935` never ran.
- `parseContinue` directly overrides `nr_iter` through `--iter`. Fresh science
  `11493435` therefore uses `--iter 10 --auto_iter_max 10` for both arms and
  fails closed on any iteration-11 log line or optimiser output.
- `11493435` began on A100 node `della-l07g2` at 07:04:48 EDT after task-rank
  probe `0,1,2` and panel/import gates passed. Primary/scalar audits are
  `11493436`/`11493437`.
- The fresh root reuses no partial control/capture/RECOVAR output. Original
  full-start science `11484481` remains independent and untouched; no RECOVAR
  production edit is authorized before the new dependent evidence.

# 2026-07-22: capture cap bound to sealed 48/48 ownership

- Absolute-bound `11493435` completed unset control in 787 seconds and stopped
  at iteration 10. Capture then failed after 23 valid-rank artifacts because
  `MAX_PARTICLES_PER_RANK=96` doubled the diagnostic file-size estimate versus
  the sealed 48/48 follower-owner panel. It produced no temporary files,
  OOM, RECOVAR output, or dependent audit; `11493436`/`11493437` never ran.
- The fresh launcher sets the exact per-follower completeness cap to 48 while
  retaining 96 expected particles, two followers, and the 64 GiB ceiling.
  Full 4,608-orientation worst-case storage for all 96 targets is
  49,785,899,520 bytes (46.37 GiB), below that ceiling.
- Fresh science `11494295` began on A100 node `della-l07g2` at 07:29:29 EDT
  after rank `0,1,2`, panel, import, and 48/48 ownership gates passed.
  Primary/scalar audits are `11494296`/`11494297`.
- Absolute iteration 10, exact dispatch/order replay, inertness,
  original-clean closure, and same-A100 RECOVAR gates are unchanged. No
  production edit is authorized before the dependent evidence.

# 2026-07-22: K=4 owner-cap failure exposes an internal-part-ID join bug

- Restart-pair science job `11494295` completed its exact iteration-10 control
  in 13m42s, then failed closed after sealing 92/96 capture artifacts when MPI
  rank 1 attempted its 49th target under the exact cap of 48. RECOVAR and
  audits `11494296`/`11494297` did not run.
- All 92 headers prove `dispatch_owner(part_id) == mpi_rank - 1`, while none
  has `part_id + 1 == stack_index`. The v1 selector incorrectly treated the
  dispatch log's `source_index` as canonical stack/source row instead of
  RELION's internal zero-based `part_id` after STAR reordering. Its apparent
  48/48 owner split was therefore not a runtime split.
- The v2 selector joins canonical identity to the live data-STAR row/internal
  `part_id` through `rlnImageName`, then indexes dispatch ownership. The new
  panel verifies 48/48 exactly, 0 owner mismatches, 96 unique internal IDs,
  and 0/96 accidental `part_id + 1 == stack_index` identities. Six categories
  remain fixed at 16 targets each.
- Failed-root classification:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_restart_pair_rankfix_absiter10_owner48_class2_prescatter_ac5177d2_20260722T072700Z/provenance/FAILURE_PART_ID_OWNER_JOIN_20260722.md`.
- Corrected preflight:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_restart_pair_rankfix_absiter10_partowner48_class2_prescatter_ac5177d2_20260722T080000Z/provenance/PREFLIGHT_PART_ID_OWNER48_20260722.md`.
- This is harness rejection, not parity evidence. Correlation was not
  computed, and no production EM source changed.

# 2026-07-22: internal-part-owner K=4 graph submitted

- Corrected science job `11494895` uses the v2 panel's explicit
  `rlnImageName -> RELION internal part_id -> dispatch owner` join. Primary
  and scalar after-ok audits are `11494896` and `11494897`.
- The sealed panel contains 96 unique identities/internal IDs, six fixed
  categories of 16, and exact owner counts 48/48. All 96 declared owners
  match dispatch ownership, while 0/96 internal IDs equal the old assumed
  stack-row identity.
- The science launcher SHA-256 is
  `c4876b6c0cdcafb4a42539382bd39c131c55bcfa94f79289f0e9622990054e40`;
  the panel JSON SHA-256 is
  `65a50e9b6428d3b4176cc7d5f69233460a38375fc8a3108c5adae242fc192d9b`.
- Full submission provenance is at
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_restart_pair_rankfix_absiter10_partowner48_class2_prescatter_ac5177d2_20260722T080000Z/provenance/SUBMISSION_20260722.md`.
- The stable K=1 scorecard remains the progress baseline at 21/34 strict
  trajectory passes and 27/34 topology passes; K=4 evidence is tracked as a
  separate diagnostic until its suite is mature enough to freeze.

# 2026-07-22: K=4 panel rebound to the actual restart boundary

- Science `11494895` completed both RELION restart arms `0:0`, but the wrapper
  exited `1:0` before RECOVAR because its standalone validator could not
  import the pinned checkout's `scripts` package. Audits `11494896` and
  `11494897` never ran and were canceled.
- The run emitted 95 structurally valid class-2 artifacts at owners 48/47.
  The missing target was class 2 in the original uninterrupted iteration-10
  STAR but class 4 in both restart arms. Only 78/96 old-panel targets were
  class-2 winners at the restart boundary, so the old panel was scientifically
  aligned to the wrong boundary even though its owner join was correct.
- Passive capture remained inert: classwise capture/control FSC-AUC is
  `0.999999995192/0.999999994390/0.999999994139/0.999999992466`, all above
  `0.999999`. Control/capture class, Euler, and translation winners are exact
  for all 100,000 particles. This remains harness evidence, not parity
  evidence; RECOVAR did not run.
- The v3 panel requires class 2 in both completed restart arms, verifies their
  internal part-ID order, then balances dispatch ownership 48/48. It retains
  six categories of 16 from qualified pools of 20,385 agreements and 63
  RELION-class-2/RECOVAR disagreements.
- Fresh science `11495311` and after-ok audits `11495312`/`11495313` use the
  corrected import contract and repeat-qualified panel. The science launcher
  SHA-256 is
  `ca8ebc48a23900b9d387cf63eeac49c52dd9edd8333ff8660fee32f514c68763`;
  panel JSON SHA-256 is
  `69119b4ed6ab6af53477beb6d76a4ced350bf115c09bfe088b39ba6aa20e6973`.
- Failure classification:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_restart_pair_rankfix_absiter10_partowner48_class2_prescatter_ac5177d2_20260722T080000Z/provenance/FAILURE_RESTART_BOUNDARY_PANEL_AND_IMPORT_20260722.md`.
- Fresh submission provenance:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_restart_repeatqual_owner48_class2_prescatter_ac5177d2_20260722T124500Z/provenance/SUBMISSION_20260722.md`.

# 2026-07-22: accepted K=4 restart pair salvaged past a stale oracle gate

- Science `11495311` completed RELION control/capture `0:0` in
  `00:13:35/00:14:39` on the same A100 and emitted exactly 96 class-2
  pre-scatter artifacts with follower ownership 48/48.
- Passive capture passed. Classwise capture/control FSC-AUC is
  `0.999999995149/0.999999994637/0.999999994225/0.999999992497`; maximum
  absolute map difference is `1.49e-8` for every class.
- The wrapper then rejected the accepted pair before RECOVAR because it
  incorrectly gated restart maps and sampling against the uninterrupted
  iteration-10 oracle. The uninterrupted perturbation is `+0.096421`; the
  prior qualification restart and both fresh arms use `-0.12306`. Same-class
  map FSC-AUC near `0.27` is descriptive across those different hypothesis
  grids, not a restart-inertness gate.
- The clean oracle remains authoritative for dispatch replay: its exact
  100,000-record iteration-10 slice matches the restart control at SHA-256
  `00059e382a1a4888275fe43d801e463529f0ae07bd046cc748f06400e855fb76`.
- Canceled never-runnable audits `11495312/11495313`. Continuation science
  `11495747`, primary audit `11495748`, and scalar audit `11495749` reuse a
  pinned digest of the accepted maps, STARs, dispatch logs, and inertness
  report; RELION is not recomputed. The continuation passed the corrected
  dispatch, qualification-perturbation, 96-identity, CUDA-build, and A100
  import gates before starting RECOVAR at commit `77a348c5`.
- Failure provenance:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_restart_repeatqual_owner48_class2_prescatter_ac5177d2_20260722T124500Z/provenance/FAILURE_STALE_CLEAN_ORACLE_GATE_20260722.md`.
- Continuation provenance:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_restart_repeatqual_owner48_class2_prescatter_ac5177d2_20260722T124500Z/provenance/RESUME_SUBMISSION_20260722.md`.

# 2026-07-22: superseded full-start K=4 graph is terminal

- Original full-start science `11484481` ended `FAILED 1:0` after `09:51:51`
  on `della-l08g5`. Its RELION control and passive-capture arms completed, but
  the wrapper stopped in the previously identified validator import path with
  `ModuleNotFoundError: No module named 'scripts'`.
- That graph had already failed its authoritative iteration-10 inertness gate:
  classwise capture/control FSC-AUC was
  `0.998247194648/0.997600525037/0.997363409604/0.998221443830`, below the
  predeclared `0.999999` threshold for every class. RECOVAR therefore did not
  run, and dependent audits `11484482`/`11484846` were canceled at zero
  runtime.
- The terminal log is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_dynamic_class2_prescatter_ac5177d2_20260722T015934Z/logs/science_11484481.out`.
  The restart-qualified continuation `11495747` and audits
  `11495748`/`11495749` are the only live authoritative comparison graph.

# 2026-07-22: fixed-suite boundary cases are being requalified at PR HEAD

- The checked-in v1 K=1 scorecard remains frozen at 34 cases and its latest
  accepted measurement remains 21/34 strict trajectory passes and 27/34
  exact intermediate-topology passes.  That snapshot is conservative: its
  source heads are `ac5177d2` and `9d172278`, while the inclusive current-size
  boundary correction is commit `7f5f7584` on the current PR branch.
- Clean pushed HEAD `3dd664c8` therefore launched paired same-allocation
  RELION/RECOVAR replacements for only the three old rows causally tied to
  that correction: cases 2, 3, and 33.  Setup is `11497146`; science jobs are
  `11497147`, `11497148`, and `11497149`; the matrix summary is `11497150`.
  Each science job runs both engines sequentially on one physical H100 and
  fails closed on a GPU-UUID change.
- The first queued summary/audits (`11497150` and
  `11497210`/`11497211`/`11497212`) were canceled before execution after the
  audit-trail commit advanced the shared worktree HEAD.  Their replacements
  are pinned to the detached `3dd664c8` audit checkout at
  `/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k1_requal_audit_3dd664c8_20260722`.
  Replacement summary is `11497305`; independent after-any shellwise
  FSC/FSC-AUC plus exact-topology audits are `11497302`, `11497303`, and
  `11497304`.  A successful process exit alone
  cannot replace a scorecard row.  Grid correction and forced final all-data
  after non-convergence are unset; correlation is not computed or gated.
- Run root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fixedsuite_boundary_requal_3dd664c8_20260722T111900Z`.
  Runtime root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_fixedsuite_boundary_requal_3dd664c8_20260722T111900Z`.
  Both contain `SAFE_TO_DELETE`.  The submission environment SHA-256 is
  `39e2b522290df7d247449ddad97ff34650c91f8d5e323b5b4682d0b69e7279c0`;
  the replacement audit-wrapper SHA-256 is
  `51e0c3a0dacde0deb85f09c5ac3009c2b713c28d4daf535c5bd6e4e8e36ae9d7`;
  the replacement summary-wrapper SHA-256 is
  `2fbb72cbc9cfaf62665a23282db3bb89211e6fb51c05689b5b580b6bf68a82af`.

# 2026-07-22: case 20 enters strict same-GPU requalification

- A prior current-head case-20 replay already closes the implementation
  behavior: all 11 current sizes match RELION, numbered merged FSC-AUC stays
  at least `0.999999998577`, and final merged FSC-AUC improves from the old
  row's `0.986023998270` to `0.997760979983`.  It remains ineligible for the
  fixed score only because RECOVAR and the immutable RELION oracle ran on
  different physical GPUs.
- Detached clean source `3dd664c8` therefore launched a fresh paired run in
  one H100 allocation.  Setup/science/summary jobs are
  `11497498`/`11497499`/`11497500`; after-any FSC/FSC-AUC and topology audit
  is `11497513`.  No score changes until that audit accepts both contracts.
- Run root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fixedsuite_case20_samegpu_3dd664c8_20260722T113100Z`.
  Runtime root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_fixedsuite_case20_samegpu_3dd664c8_20260722T113100Z`.
  Both have `SAFE_TO_DELETE`; the submission and audit-wrapper SHA-256 values
  are `dd0d2d84342edfa0b7b4a27a5463c6038bf2d05a6024a220e7ac2214b1205403`
  and `a296476a1b4ab518e044e742e33206b949acc838c9bad08a1b5821b8524bc6a9`.

# 2026-07-22: small old-head failures enter fixed-suite requalification

- The adaptive-pass-1 RELION CUDA-matrix correction (`db1bf391`) postdates
  the `ac5177d2` source used by five inexpensive failing fixed-suite rows.
  Clean detached `3dd664c8` therefore launched paired same-H100 reruns for
  cases 22, 23, 24, 26, and 32 without changing their definitions or gates.
- Setup is `11497554`; science jobs are `11497555` through `11497559` in
  case order; summary is `11497560`.  Corresponding after-any FSC/FSC-AUC and
  exact-topology audits are `11497575` through `11497579`.
- Run root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fixedsuite_smallfail_requal_3dd664c8_20260722T114000Z`.
  Runtime root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_fixedsuite_smallfail_requal_3dd664c8_20260722T114000Z`.
  Both contain `SAFE_TO_DELETE`; submission and audit-wrapper SHA-256 values
  are `7691322c849470eac74ebf7f64b890d6068f6be4f47cd6d540855b4875b46314`
  and `fca17ca100a2fe5df8707a8b5e440c4962fb16d7fdc7a2d45f67de23ebe03105`.

# 2026-07-22: frozen scorecard v5 and active acceptance graph

- Frozen snapshot `strict-k1-v5-20260722` accepts 23/34 strict trajectory
  cases and 29/34 exact intermediate topologies.  The first frozen snapshot
  was 20/34 and 27/34; the denominator did not change.  Evidence ledger
  `em_k1_gui_grid0_local_highshell_full34_superseding_ledger_v5` has SHA-256
  `11a82951cadd2ccd2123904345090bec45352dd5f1884e0ae25fdf35d0695311`.
- Case 23 exact-fixture science `11501524` and audit `11501622` passed.  Final
  merged cross-engine FSC-AUC is `0.9983424084`; RECOVAR-minus-RELION merged
  GT FSC-AUC is `+0.0122984963`; the exact 13-iteration schedule/topology
  passed.  Run root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fixedsuite_case23_exacttrajectory_ab52b1ff_20260722T175000Z`.
- Cases 2/3 exact-fixture science `11501888`/`11501889` uses detached commit
  `84143872a5179b5567dc8e43fb81b985d7beb37d`, whose ancestry contains the
  inclusive boundary-shell fix `7f5f7584` and CUDA adaptive-pass-1 fix
  `db1bf391`.  The falsifiable gate is whether iteration 3 now selects current
  size 164, not the old RECOVAR 162, and whether strict trajectory auditors
  `11501907`/`11501908` then pass.  Run root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fixedsuite_cases02_03_strict_84143872_20260722T180000Z`.
- Case 24 uses clean integration commit
  `6235fb035380b2ecb18851a65cd9729a6d4de868`: setup `11504822`, science
  `11504823`, summary `11504824`, and strict audit `11504831`.  The old exact
  run passed topology but missed the final merged cross gate by
  `0.000194896`.  Run and runtime roots are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fixedsuite_case24_strict_6235fb03_20260722T191500Z`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_fixedsuite_case24_strict_6235fb03_20260722T191500Z`;
  both contain `SAFE_TO_DELETE`.
- K=4 same-A100 job `11503805` runs source
  `7cd1aa4b13a543f7283e1490607ca3603b646611`.  It reuses accepted immutable
  RELION pair SHA-256
  `7a69dc75ec4e9375dcad915fafd7c06e70baabcad6d11389a9b9914f312aab06`
  and targets iteration 10, half 1, class 2, and the frozen 96-particle panel.
  The scoped fused-pass-2 capture is observational; authoritative M-step
  arrays are unchanged.  Run root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_fusedcapture_7cd1aa4b_20260722T183935Z`;
  runtime root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k4_it10_fusedcapture_7cd1aa4b_20260722T183935Z`;
  both contain `SAFE_TO_DELETE`.
- Integration validation at clean `6235fb03` is recorded under
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_parity_longtest_6235fb03_20260722T190500Z`.
  Jobs are smoke/downstream/SPA/ET `11504666`--`11504669`, remaining groups
  `11504670`--`11504674`, summary `11504675`, and corrected unit rerun
  `11504718`.  Original unit `11504665` is infrastructure invalid: it reached
  collection without the optional RELION binding and produced 19 import
  errors.  The rerun uses the accepted unchanged binding and overwrites that
  XML.  No push or PR score claim is authorized until every required group
  passes.

# 2026-07-22: case 24 localizes from final FSC to an iteration-2 pose decision

- The immutable replacement at integration source
  `a2be302cdc08a59f0937e61e3ad92b72e939ccd7` completed as science job
  `11507875`; strict FSC/topology auditor `11507904` rejected the row.  The
  fixed score therefore remains 23/34 strict and 29/34 topology.
- All 12 numbered RECOVAR maps match the 12 RELION maps and the exact
  convergence/finalization topology.  The worst numbered merged cross-engine
  FSC-AUC is `0.999807820661` at iteration 12.  Only the final merged map
  fails: `0.991502719959 < 0.995`; its RECOVAR-minus-RELION GT FSC-AUC is
  `+0.008628902885`.  This is a scientific failure, not an infrastructure
  failure, and the threshold was not widened.
- A full input-identity particle audit shows that the final all-data pass does
  not create a new pose tail.  At numbered iteration 12, 392/3000 particles
  already differ by more than 0.5 degrees, versus 391/3000 in the final pass;
  the per-particle angular errors correlate at `0.9987`.  The final angular
  median is `4.56e-6` degrees and p95 is `1.15775` degrees.
- The first greater-than-five-degree split appears at iteration 2 for original
  index `2767`, RELION one-based stack image `2768`, half 1.  RECOVAR/RELION
  Pmax is `0.385798991/0.385362` and both engines retain two significant
  samples, but their selected poses differ by `9.18615` degrees.  The particle
  is one of the simulator's injected outliers.  This makes the iteration-2
  candidate-score/tie boundary the earliest evidence-backed target; no
  production arithmetic change is authorized before its operands are
  compared.
- Diagnostic-only integration commit `9565b8a1`
  forwards the already implemented pass-1 significance dump controls through
  the K=1 Slurm launcher and adds a focused dry-run test.  The complete
  launcher test file passes 39/39.  The first queued setup `11509108` failed
  its source-HEAD provenance gate before science after its non-immutable
  worktree advanced; blocked jobs `11509109`/`11509110`/`11509172` were
  canceled at zero runtime.  Immutable detached-`0da399c4` replacement
  setup/science/summary jobs are `11509611`/`11509612`/`11509613`; after-any
  RELION-versus-RECOVAR operand comparison is `11509654`.
- Immutable result root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fixedsuite_case24_strict_a2be302c_20260722T204500Z`.
  Focused capture root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case24_it2_particle2767_capture_0da399c4_20260722T213929Z`.
  Both corresponding runtime roots and run roots contain `SAFE_TO_DELETE`.

# 2026-07-22: replacement unit group is green; shared long gate remains red

- Replacement unit job `11507920` completed on `della-l09g4` in `00:50:03`.
  The JUnit result is 5,639 tests: 5,586 passed, 53 skipped, 0 failed, and
  0 errors.  This closes the environment-dependent pixi-path assertion
  exposed by `11504718` and validates commit `a2be302c`.
- Every other integration group except the cryo-ET outlier regression is
  green.  The outlier failure reproduces on clean `dev2`, but the repository
  policy still makes the complete mandatory gate red.  Therefore the local
  parity commits are not pushed and the draft PR is updated with evidence
  instead.
- Durable validation root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_parity_longtest_6235fb03_20260722T190500Z`.
  The completed-unit result is appended to
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_parity_longtest_6235fb03_20260722T190500Z/provenance/SUBMISSION_11504665_11504675.md`.

# 2026-07-22: case 26 passes every numbered map and fails only final Nyquist

- Detached-`a2be302c` science `11508258` completed in `00:23:35`.  The strict
  FSC audit rejects only final merged cross-engine FSC-AUC
  `0.963324126445 < 0.995`; the 11 numbered maps pass, exact topology passes,
  and the final RECOVAR-minus-RELION GT FSC-AUC is `+0.009268703399`.
- The scheduled CPU auditor is `11508284`.  While it waited for priority, the
  identical two checked-in audit modules were run against the immutable case
  output.  Their statuses are FSC `2` (scientific threshold failure) and
  topology `0` (pass), so this is not an infrastructure failure and does not
  change the frozen 23/34 strict, 29/34 topology score.
- A complete particle identity join compares numbered iteration 11 and final
  all-data.  The fraction within 0.5 degrees changes only from 835/1000 to
  829/1000; angular median is `5.63e-6`/`5.69e-6` degrees and p95 is
  `2.34529`/`2.38760` degrees.  The full-Nyquist failure inherits the numbered
  pose tail, consistent with the case-24 evidence, rather than identifying a
  distinct final-join arithmetic defect.
- FSC/topology/particle JSON/particle NPZ SHA-256 values are
  `107a6983aa496346e50e875196c45fbb673e8a51187828b2758db6962285227e`,
  `3362fb2e785a42922b5d98414fca05c26f5ae6b04029c91e04224394e4851d2b`,
  `3de8993c743c3015838286148a7920d1b7116e796a22ebdc2a98d4e94dfd7d60`,
  and `2aa103ee09442d4a4fd00cdc38ec0aa89d0e4aaca586582189f37d43cf9258f6`.
- Durable case root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fixedsuite_unresolved_a2be302c_20260722T205500Z/cases/26_tiny_severe_1k_g128_radial_noise5_nonuniform_pct30_bf80`.

# 2026-07-24: cases 4/5 first broaden at scoring iteration 2

- The complete exact-identity audits show that RECOVAR iteration 0 / RELION
  iteration 1 has exactly equal Pmax and significant-support arrays for all
  100,000 particles in both cases.  Only 9 case-4 and 3 case-5 particles
  exceed 0.01 Angstrom translation error.
- At the next aligned iteration, all 100,000 Pmax values are non-identical.
  Pmax absolute p95 is `5.23240e-4` for case 4 and `1.13895e-4` for case 5;
  significant support differs for 331 and 207 particles, and the maximum of
  the greater-than-0.5-degree pose count and greater-than-0.01-Angstrom
  translation count is 60 and 25 particles.  This is the earliest broad
  continuous posterior split.
- Same-GPU case-4 counterfactual `11558427` runs through iteration 2 with
  exact incoming RELION replay state in both arms.  The only arm difference is
  resident RECOVAR versus exact RELION iteration-1 half maps at scoring
  iteration 2.  The output is identity-audited against RELION iteration 2.
- Initial `11558403` failed in three seconds in an over-strict JAX symlink-path
  assertion before science.  It is not evidence.  Replacement `11558427`
  passed clean-source, checkout-bound import, pixi-JAX, and one-CUDA-device
  gates on `della-h20g2`.
- Run root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_it2_relref_counterfactual_5cb01ec1_20260724T040135Z`.
- The frozen score remains 25/34 strict, 31/34 exact topology, and 34/34
  evaluated until a fixed case passes its accepted audit.

# 2026-07-24: case 4 iteration-2 exact-map intervention is mixed

- Same-H100 science `11558427` completed resident-map and exact-RELION-map
  arms sequentially on
  `GPU-49c1a223-be61-858b-49d8-d8b0347ac252`.  Both arms share exact incoming
  RELION non-map state; only the scoring-iteration-2 half references differ.
- Exact non-map state alone is nearly null: mean/p95 absolute-Pmax-error
  ratios versus autonomous are `0.951815`/`0.989488`, with support mismatches
  331 to 323.
- Exact maps are the leading component but miss the predeclared dominance
  gate: conditional mean/p95 ratios are `0.256586`/`0.189190`, support
  mismatches fall 323 to 92, angular `<=0.5` degrees improves
  `0.99958 -> 0.99992`, and translation `<=0.01` Angstrom improves
  `0.99947 -> 0.99983`.
- Combined exact state and maps produce ratios `0.244222`/`0.187201` and
  support 331 to 92.  The accepted classification is `mixed`: most of the
  broad residual is inherited through the iteration-1 half maps, while a
  smaller identical-input scorer/candidate residual remains.
- The GPU wrapper's two arm runs and hash-pinned reports completed, then its
  obsolete `status == "pass"` assertion rejected the auditor's current
  `"complete"` status.  Recovery audit `11559766` independently verified
  both report/array manifests and completed `0:0`; superseded
  dependency-never-satisfied job `11558553` was canceled at zero runtime.
- Accepted classification SHA-256:
  `0496eb4b4247a308f9ab3012ed2fc97389da2ae5a2271f0eb179bde9d0e18a3f`.
  Corrected audit-launcher SHA-256:
  `b712c20bb4edb6e9bcd13f141e2aa6892c5fb2d953afe9f7413b1feb24dde933`.
- Next discriminator: matched iteration-1 native RECOVAR and passive RELION
  BPref/reconstruction capture plus a narrow iteration-2 residual-candidate
  audit.  Do not change the `0.999` significance threshold.
- Frozen score remains 25/34 strict, 31/34 exact topology, and 34/34
  evaluated.

# 2026-07-24: case 4 iteration-1 BPref localizes upstream of reconstruction

- Same-H100 science `11559949` completed the RECOVAR native/pre-join/post-join
  accumulator capture followed by passive RELION BPref/stage capture on
  `GPU-8fdb5482-ff52-be6a-c41a-cda8af052492`.  Independent H100 audit
  `11559964` completed `0:0`.
- The patched RELION control is inert: 100,000/100,000 poses, translations,
  Pmax values, and support counts are exactly equal.  Captured-versus-oracle
  half-map FSC-AUC is `0.9999999999965`/`0.9999999999966`; through-shell-28
  FSC-AUC is `1.0` to displayed precision.
- RECOVAR native post-x0 versus public pre-join is bit-exact for numerator and
  weight in both halves.  Native layout conversion is closed.
- The first nonzero cross-engine boundary is the joined BPref.  Numerator
  relative L2 is `0.00187373`/`0.00300368`; weight relative L2 is
  `0.000269398`/`0.000458921`.
- RECOVAR reconstruction of the captured RELION BPref matches RELION's
  post-reconstruct maps at FSC-AUC `0.999999999699`/`0.999999999634`.
  Derived tau2 matches the first 18 RELION model shells within relative L2
  `2.07e-7`/`2.08e-7`.  Final flattened-map FSC-AUC is at least
  `0.999999999984`.
- Reconstruction, tau2, and post-processing are rejected as causes.  The
  remaining target is accumulation: determine how much of the residual is
  explained by the nine known first-iteration winner exceptions, then audit
  matched-winner M-step/backprojection arithmetic if needed.
- Accepted audit SHA-256:
  `7cf9a60c6fa824a43603d3c095462a40848c697247d1ca32c00910ff671cd13c`.
  Science marker SHA-256:
  `81094fc19ee61b6b329eac7ff87318a170b6d54eecce2f17289b3b721a3fa920`.
- Run root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_it1_native_relion_bpref_8a3737af_20260724T050935Z`.
- No reconstruction, tau2, significance-threshold, or unconditional map
  substitution patch is authorized.  Frozen score remains 25/34 strict,
  31/34 exact topology, and 34/34 evaluated.

# 2026-07-24: nine first-iteration winners explain the case-4 BPref gap

- Same-H100 contribution capture `11561082` completed `0:0` in `00:14:44` on
  `GPU-9f98ccbf-3c62-c54f-7409-7eb58845ad4a`.  It captured all nine known
  winner exceptions and produced bit-exact captured-winner controls for both
  half-set accumulator replays.
- A second same-GPU intervention retained the exact captured RECOVAR
  image/CTF/scatter operands and changed only the nine WTA rotations and
  translations to their RELION values.  Its GPU replay completed before
  wrapper `11561160` rejected an obsolete cross-run bit-exact aggregate gate.
  CPU recovery audit accepted the fresh aggregate envelope, whose relative
  L2 is at most `6.37e-8`.
- The intervention reduces numerator relative L2 from
  `0.0018737330 -> 1.9293688e-6` in half 1 and
  `0.0030036818 -> 3.0769983e-5` in half 2.  Weight relative L2 falls from
  `0.00026939846 -> 8.4532979e-7` and
  `0.00045892132 -> 7.1740414e-6`.  The worst arm removes `0.99975563` of
  residual energy.
- Therefore the case-4 accumulator mismatch is attributable to the nine
  discrete iteration-1 winner choices, not general reconstruction or
  M-step/backprojection arithmetic.  Original indices `6322` and `60368`
  have large rotation changes (`150.7523` and `165.1183` degrees); for `6322`
  the RELION winner is absent from RECOVAR's selected eight-rotation fine
  subset.  The next target is their firstiter coarse/global score grid and
  winner routing.
- Accepted audit:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_it1_winner9_contributions_608c509d_20260724T064102Z/analysis/case04_winner9_contribution_scale_audit.json`
  (SHA-256
  `6fe333a95b07495185d95103c8b1e70d0c1c9d91cca5dfdd705d14839e3ab553`).
  Pre-science `11561065` failed only its JAX symlink path assertion; it is not
  scientific evidence.
- No production patch is authorized from this result.  Frozen score remains
  25/34 strict, 31/34 exact topology, and 34/34 evaluated.

# 2026-07-24: case 5 tests the case-4 intervention unchanged

- The accepted case-5 baseline audit has three first-iteration assignment
  exceptions: original indices `26055`, `93729`, and `95412`.  All are
  translation-only at the reporting tolerance.
- Frozen-fixture setup `11564052` completed `0:0`; science `11564053`, matrix
  summary `11564054`, and unchanged strict FSC/topology audit `11564062` test
  the same direct-real initial projector plus bounded `4e-6` top-two tree
  rescore used for case 4.
- Science is source-bound to clean detached commit `c74beea4` and physical
  H100 `GPU-49c1a223-be61-858b-49d8-d8b0347ac252`.  Grid correction and forced
  final-all-data after non-convergence are unset.
- This is an independent generalization arm, not a metric update.  Frozen
  score remains 25/34 strict, 31/34 exact topology, and 34/34 evaluated while
  its auditors are pending.

# 2026-07-24: case 24 iteration-1 winner and maps close under combined opt-ins

- Reduction-only full replay at source `9abd79fb` examined 3,000 images,
  found four top-two margins within `4e-6`, and changed zero winners.  The
  existing RELION 128-lane/tree rescore is therefore necessary for the
  target pair but is not sufficient with RECOVAR's live initial projector.
- The live initial projector differs from captured RELION `Projector::data`
  at relative L2 `2.75962e-7`.  The discrepancy is reproduced by the
  complex64 Fourier-to-real roundtrip used before initial projector
  construction.  Case input `reference_init_relion.mrc` is exactly the
  negative of `reference_init.mrc`, so this is not an input-map mismatch.
- Commit `a521cfb6` adds the default-off
  `RECOVAR_INITIAL_PROJECTOR_USE_REAL_REFERENCE=1` path.  Only the initial
  projector consumes the preserved float64 real low-pass result; resident
  Fourier state and subsequent iterations are unchanged.  The saved
  complex64 projector residual drops `10.82x` to `2.55164e-8`.
- The combined local A100 run uses both
  `RECOVAR_INITIAL_PROJECTOR_USE_REAL_REFERENCE=1` and
  `RECOVAR_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN=4e-6`, with
  `RECOVAR_FINAL_ALL_DATA_GRID_CORRECT` unset/off.  Half 1 has 1/1
  ambiguity/change; half 2 has 3/0.  Original index 1901 changes from
  `(16550, 14)` to RELION `(16551, 14)`.
- Cross-engine iteration-1 FSC-AUC improves from
  `0.999998394276/0.999999741584/0.999999380590` to
  `0.999999999956/0.999999999955/0.999999999972` for
  half1/half2/merged.  These are non-DC FSC metrics over 62 finite shells;
  correlation is not used.
- Accepted diagnostic:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case24_tree_top2_58457106_20260724T033852Z/case24_combined_projector_tree_intervention_audit.json`
  (SHA-256
  `7343196ea7ca9643bb586ca97159564badc85687102a3f4c179269b6729f1502`).
  Run root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case24_tree_top2_58457106_20260724T033852Z/combined_a521cfb6`.
- Canceled pending Slurm attempts `11561699` and `11561811` had zero runtime
  and are not evidence.  Next run the complete fixed case and apply the
  unchanged strict FSC/topology audit.  Until it passes, frozen snapshot
  `strict-k1-v6-20260724` remains 25/34 strict, 31/34 exact topology, and
  34/34 evaluated.

# 2026-07-24: case 24 is exact through iteration 3; the remaining split starts with one translation

- The complete artifact-pinned case-24 rerun at commit `b826bc52` used both
  default-off interventions, grid correction unset/off, and no forced
  final-all-data iteration.  Setup `11562037`, science `11562038`, and summary
  `11562039` completed `0:0`; strict audit `11562082` returned the scientific
  failure `final merged cross-engine FSC-AUC 0.9948014630931105 < 0.995`.
  The intermediate topology audit passed all 12 numbered iterations.
- Numbered merged cross-engine FSC-AUC is
  `0.999999999973`, `0.999999999903`, and `0.999999999901` at iterations
  1--3.  It first moves materially at iteration 4 (`0.999999304336`) and
  reaches `0.999926660877` at iteration 12.  Final half1/half2/merged values
  are `0.999040692683`, `0.995870180922`, and `0.994801463093`; final GT
  FSC-AUC delta is `+0.008173125002`.
- A sealed two-iteration A100 capture shows that the former particle-2767
  split is closed.  RECOVAR and the accepted patched RELION capture have
  identical 64/64 fine support, identical 13/13 reconstruction support, the
  same local winner `(6, 52)`, probability correlation
  `0.999999992880`, and common-renormalized probability L1
  `0.0001021239235`.  The centered total-score maximum error is
  `3.77953e-4`, versus the old two-candidate pre-prior margin discrepancy of
  approximately `0.10411`.
- Exact pose comparison over all 3,000 particle identities finds zero
  rotation differences above `0.1` degrees through iteration 4.  Translations
  are exact through iteration 2.  At iteration 3, only original index `2332`
  differs above `0.01` Angstrom: RECOVAR and RELION have the same rotation,
  while the x translation differs by one fine step (`0.5` pixel =
  `2.125` Angstrom).  The mismatch persists at iteration 4; iteration 5 is
  the first point with two rotation differences above one degree.
- The iteration-3 RECOVAR capture reproduces the split.  Its two leading fine
  rows `(1, 59)` and `(1, 57)` are both in fine and reconstruction support,
  carry posterior `0.3224018721` and `0.3223231703`, and differ by only
  `0.000244140625` in both total and pre-prior score.  The original stock
  RELION trajectory stores translation `57`, while RECOVAR stores translation
  `59`.
- Same-node/same-physical-GPU patched RELION replay `11562574` completed
  `0:0` in `00:06:24` on `della-l07g6`, using the fixed-case GPU UUID
  `GPU-6a3cea75-90ac-d3de-7c1a-a8158412a9f4`.  Its 64/64 fine support and
  12/12 reconstruction support are exact against RECOVAR; posterior
  correlation is `0.999999998039`, common-renormalized L1 is
  `6.0464626e-5`, centered pre-prior maximum error is
  `3.662109375e-4`, and centered total-score maximum error is
  `4.884e-4`.  Both engines choose `(1, 59)`.
- Binary/source audit subsequently found that replay `11562574` used patched
  RELION `5.0.1-commit-f2c1a3` (binary SHA-256
  `d3447c820511b3dc1bb0fd9969323800c192bb3a9c6e6a0367b22a85b3fde689`),
  whereas the installed stock oracle is `5.0.1-commit-d476e6` (binary
  SHA-256
  `92cf3ba54038d5e162e238b952fe88f1414f440d4e6cba23bc4b097428087b4a`).
  The source commits differ only by the relax-symmetry parser addition, but
  the replay is not an instrumentation-only rebuild of the exact stock
  source.  Its arithmetic remains useful localization evidence, not an exact
  stock score capture.
- The replay also stores x origin `3.168496` Angstrom
  (`0.74552849` pixel), exactly the translation-59 pose.  This rules out a
  RECOVAR winner-to-pose mapping error: RELION's accelerated path stores the
  rounded previous integer-pixel offset plus the selected oversampled
  translation.  The stock trajectory's translation-57 choice is therefore a
  score near-tie, not a serialization discrepancy.  The patched replay's
  translation-59 versus translation-57 posterior gap is only
  `3.93436e-5`; their raw diff2 values differ by one float32 unit
  (`1442.527099609375` versus `1442.5272216796875`).
- Winner-stability probe `11562830` completed `0:0` in `00:12:34`.  It ran
  installed stock d476e6 and patched f2c1a3 sequentially with every dump
  variable unset on the fixed-case node and physical GPU; both stored
  translation `59`.  The immediately adjacent stock-d476e6 arm in job
  `11563252`, on the same node/GPU and with the same early-iteration science
  arguments, stored translation `57`.
- Direct comparison of those two installed-stock runs is exact for all
  iteration-1 pose/Pmax/support fields.  At iteration 2 the poses and support
  remain exact while 1,538/3,000 serialized Pmax values differ by at most
  `6.4e-5`.  At iteration 3 Euler angles and support remain exact, exactly one
  x translation differs by `2.125` Angstrom, and 1,897/3,000 Pmax values
  differ by at most `6.1e-5`.  Half1/half2/merged cross-run FSC-AUC is
  `0.999999999999/0.999999999998/0.999999999999` at iteration 1,
  `0.999999999985/0.999999999985/0.999999999993` at iteration 2, and
  `0.999999999985/0.999999999985/0.999999999992` at iteration 3.
- This is direct stock-repeat evidence that the one-ULP hard winner is
  launch-sensitive.  The two probe jobs differ in job-local scratch identity
  and preceding process history; the accepted full run also used a different
  MPI temporary root.  The result is therefore not a claim of byte-identical
  launch context, but it rejects a deterministic translation-57 stock target
  and rejects forcing RECOVAR to one side of the numerical boundary.
- Source-exact d476e6 score probe `11563252` completed `0:0` in `00:18:37`.
  Its installed-stock and dormant-instrumentation arms both chose
  translation `57`; the narrow active-capture arm chose translation `59`.
  The source is exact d476e6 plus a hash-pinned four-file patch, although the
  rebuilt executable is explicitly not byte-identical to the installed
  binary.
- The exact-source active capture records raw scores
  `1442.52734375` for `(rot_idx=1, trans_idx=59)` and
  `1442.5274658203125` for `(rot_idx=1, trans_idx=57)`.  Their difference is
  exactly one float32 ULP (`0.0001220703125`).  Normalized posteriors are
  `0.322370919140121` and `0.322331576104412`, a
  `3.93430357086353e-5` gap.
- Dormant versus active instrumented runs retain exact Euler angles and
  support counts through iteration 3, differ in only that one x translation,
  and have merged FSC-AUC `1.000000000000`, `0.999999999993`, and
  `0.999999999992` at iterations 1--3.  Since installed stock independently
  selected both translations across the adjacent jobs, this difference does
  not establish a causal dump effect.  It captures the numerical boundary
  and supports no production arithmetic or tie-break change.
- Fixed-case root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fixedsuite_case24_combined_b826bc52_20260724T082100Z`.
  Particle-2767 capture:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case24_it2_combined_b826bc52_20260724T084954Z`.
  Particle-2332 capture and replay:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case24_it3_p2332_combined_b826bc52_20260724T085831Z`.
  Winner-stability probe:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case24_it3_relion_winner_probe_20260724T092606Z`.
  Source-exact score probe:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case24_it3_d476_score_probe_20260724T094334Z`.
- This is a localization result, not a strict pass.  Frozen snapshot
  `strict-k1-v6-20260724` remains 25/34 strict, 31/34 exact topology, and
  34/34 evaluated.

# 2026-07-24: case 4 particle 6322 is an exact RELION coarse-score tie

- Same-H100 full-grid job `11562639` completed `0:0` in `01:24:39` on
  `GPU-f6cfb4eb-6f8b-0df7-4ec9-8ec065affa8f`.  It ran RELION and RECOVAR
  sequentially from fixed case-4 bytes at iteration 1/current size 56.
- The fail-closed comparison asserts 1,069,056 candidates per engine,
  1,069,056 common identities, Jaccard 1.0, and no duplicate or engine-only
  keys.  Aligned score correlation is `0.9999999999954908`; centered
  RECOVAR-minus-RELION score difference has mean `-3.1083669e-7`, p95
  absolute `5.1409006e-7`, and maximum absolute `1.4603138e-6`.
- RELION selects mapped key `(20057, 8)` and RECOVAR selects `(25798, 0)`.
  Both have the exact same top RELION float32 score
  `0.2807506024837494`; the exact RELION tie count is two.  RECOVAR scores
  the pair `0.2807507812976837` and `0.28075096011161804`, separating them
  by only `1.7881393432617188e-7`.
- The earlier 150.7523-degree winner exception is therefore localized to
  coarse float32 reduction of an exact RELION tie.  This supports the
  existing bounded 128-lane top-two re-reduction and lower-pose-ID tie
  resolution.  It rejects broad scoring, projector, reconstruction, tau2,
  and posterior-threshold changes.
- The official comparison JSON SHA-256 is
  `2e3368c5c03db4d0eea9519c746be6c4d4b26f8b8b0f11e98420ee6d878ebcdd`.
  Durable result:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_it1_p6322_coarsegrid_7a4120c1_20260724T031102Z/provenance/RESULT_11562639.md`.
- Frozen-fixture case-4 intervention jobs are setup `11563826`, science
  `11563827`, summary `11563828`, and unchanged strict audit `11563842`.
  They use clean detached source `c74beea4`, the preserved real initial
  projector, tree-rescore margin `4e-6`, grid correction unset/off, and no
  forced final-all-data iteration.
- CPU targeted tests initially exposed backend contraction beyond a one-ULP
  NumPy logical replay.  Exact H100 test `11563940` passed the CUDA one-ULP
  contract (`1 passed in 2.68s`); the backend-aware CPU gate and related
  tests pass `13 passed in 7.89s`.
- The fixed score remains 25/34 strict, 31/34 exact topology, and 34/34
  evaluated until the new science and both strict audits pass.

# 2026-07-24: K=4 recovery auditors canceled on physical-GPU mismatch

- Accepted RELION control/capture and superseded RECOVAR science used physical
  A100 `GPU-803dc869-2e74-273c-1df4-08adbc94e1b3`.  RECOVAR-only recovery
  `11561204` uses `GPU-6ec3d0a5-efc4-2f4c-fa73-7d76b911a412`.
- The replacement audit launchers verified only recovery walltime UUID versus
  recovery preflight UUID.  They omitted the required equality check against
  the reused RELION runtime UUID, so pending jobs `11561345` and `11561350`
  were canceled at zero runtime.
- The two RECOVAR trajectories differ materially by numbered iteration 8:
  half/class shellwise FSC-AUC spans
  `0.998882600598`--`0.999075807124`; fine/coarse assignments differ
  `705/100000` and `366/100000`; noise and tau2 relative L2 are
  `4.18455764e-6` and `1.71844644e-4`.
- Recovery science remains a valid cross-A100 diagnostic but cannot support
  same-physical-GPU K=4 acceptance.  Any replacement must run all three arms
  sequentially in one allocation and assert UUID equality before scientific
  comparison.
- Durable note:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_fusedcapture_74f89c60_20260724T014000Z/provenance/RECOVERY_UUID_INVALIDATION_20260724.md`.
- Correct all-in-one replacement jobs are science `11564419`, vector audit
  `11564442`, and scalar audit `11564443`.  Science and both audit launchers
  independently bind the fresh root; science checks the three runtime
  walltime UUIDs and the primary audit repeats the triplet gate.  Runtime
  preflight binds physical A100
  `GPU-6f45f415-9d0b-d562-9ff3-c9fb7bc53aa7`.

# 2026-07-24: fixed-suite snapshot proposals are fail-closed and reproducible

- `scripts/summarize_em_relion_parity_scorecard.py --proposal-output` now
  constructs a candidate superseding ledger only after validating the frozen
  case definition, the exact checked v2 manifest bytes
  (`422a79a0a7703d92f9777266e8c34ccd3a7cf5963b354e57a7d9a18f227babee`),
  every re-hashed materialized fixture byte, clean science source, exact
  agreement between the requested science job and both `submission.env` and
  `selected_cases.tsv`, same physical RELION/RECOVAR GPU, autonomous
  trajectory mode, terminal FSC/FSC-AUC and exact-topology audit status,
  convergence-only final all-data, grid correction off, and SHA-256
  identities for all audit products.
- The proposal binds the exact current parent-ledger SHA-256 and requires a
  monotonically newer ledger schema.  It refuses to overwrite an existing
  output and never mutates the checked scorecard.  A human-reviewed
  scorecard/history update therefore remains a separate explicit step.
- A historical real-evidence replay against the pre-v6 scorecard reproduced
  the accepted case-2 and case-33 v6 update objects exactly (2/2).  The
  immutable v6 ledger remains
  `32c6512a8507f7b17a59d0be527fa5c9609067e0d8f598a2d108bed9a3fc8a56`.
- CPU Slurm replay `11567015` completed `0:0` in 82 seconds after reading and
  re-hashing the 78.6 GB of materialized case-2/case-33 particle stacks.  It
  exercised the combined byte-hash and submission/case-table job binding at
  clean source `93abcb91` and again reproduced both v6 update objects exactly.
  Its report and launcher SHA-256 values are
  `e75c27ebb0aae11453dbc5017cb2edc0778d2ef859745e6c1b873d34cbf8ffca`
  and `b13a1d3e37765291fbfa03256664a2a43238d5f4704ff38cefa97bde074676ac`;
  stderr is empty.
- Focused validation is 15/15 passing; Ruff formatting/lint and mypy are
  clean; the generated scorecard check passes.  The new regressions mutate a
  materialized fixture without changing its size, alter the manifest with a
  structurally valid SHA-256, and confirm that proposal validation rejects
  both identity changes and a science job not bound to the exact
  submission/case-table row.
- Original proposal/replay logs remain under
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_scorecard_proposal_20260724`.
  Re-hash validation is under
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_scorecard_materialized_rehash_20260724T142500Z`;
  its latest passing `validation_manifest_pin.log` and
  `pytest_manifest_pin.xml` SHA-256 values are
  `6f88b63a26e825fb986f953a358546d6d31b10ae2c8b9297986b9f7d0118e688`
  and `244230e24d7f092d9f61943319794e47d97e6a4c12dc3de3b6964aa0cf10d03c`.
- This tooling does not change the scientific numerator.  Snapshot
  `strict-k1-v6-20260724` remains 25/34 strict, 31/34 exact topology, and
  34/34 evaluated while cases 3/4/5 and K=4 auditors remain active.

# 2026-07-24: case 10 confirms the inherited full-grid final-only family

- CPU Slurm job `11566606` completed `0:0` in 43 seconds with 780256 KiB
  maximum RSS.  It audited frozen case 10's last numbered state and final
  all-data state by exact image identity at clean source `162ee03a`.
- Exact schedule/convergence gates pass: RELION and RECOVAR both converge at
  iteration 15, final all-data is present, and iteration 15 matches current
  size 68 and HEALPix order 5.  The final all-data expectation then uses the
  full 384 grid.
- Last-numbered versus final fractions within 0.5 degrees are
  `91.501% -> 91.457%`; angular p95 is
  `0.990475839 -> 1.001330626` degrees.  Fractions within 0.5 Angstrom
  translation are `92.650% -> 92.531%`, and Pmax absolute-error p95 is
  `0.006589644 -> 0.006748276`.
- The particle-state tail is therefore nearly stationary while merged
  cross-engine FSC-AUC falls from `0.999967227122` to `0.983006503534`.
  This confirms that the full-grid final expectation amplifies inherited
  pose/reference/posterior state; it does not support a final writeback,
  scheduler, grid-correction, or threshold change.
- JSON/NPZ SHA-256 values are
  `eb0ac4bcbecbfa4d9333ececc6340ed5fb4dfedb6c19745472b205b5b2582dbd`
  and `c44b7a4a85af54413c04f50572a6f0b805dfa5f31c63542a395926c3e8c1bab0`.
  Launcher SHA-256 is
  `5e5a271f88b23570d418733fae27c19a04fb5dcba6404f2701d15e5b6271e5d0`.
  Durable run/runtime roots are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case10_last_final_audit_20260724T140800Z`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case10_last_final_audit_20260724T140800Z`.
- This diagnostic does not change snapshot `strict-k1-v6-20260724`: 25/34
  strict, 31/34 exact topology, 34/34 evaluated.

# 2026-07-24: frozen case 22 tests the bounded firstiter intervention

- Case 22 remains strict-fail and exact-topology-fail.  Its known iteration-2
  numerical butterfly is reference-driven, making it the smaller independent
  generalization target for the direct-real-reference plus bounded top-two
  RELION reduction-tree intervention already under test on cases 4/5.
- Clean detached source is
  `b1d444270de89a4ede0868fe0e39954d012fd593`; the fixed fixture-manifest
  SHA-256 is
  `422a79a0a7703d92f9777266e8c34ccd3a7cf5963b354e57a7d9a18f227babee`.
  Submission worktree status and diff are clean.
- Slurm graph is setup `11566711`, science `11566712`, summary `11566713`,
  and unchanged strict FSC/topology audit `11566739`.  The audit is
  `afterok:11566712`; no failed or partial science can change the score.
- The full autonomous run sets
  `RECOVAR_INITIAL_PROJECTOR_USE_REAL_REFERENCE=1` and
  `RECOVAR_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN=4e-6`.  It leaves
  `RECOVAR_FINAL_ALL_DATA_GRID_CORRECT` and
  `RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER` unset and explicitly disables
  last-numbered-state replay for the final expectation.
- Setup/science/audit launcher SHA-256 values are
  `c1f50a8095bd107b733a4c3f25046a4c8e28c3d8718366e39bd2ac41c3ce9b62`,
  `ae103ec9f899d37b39f4046541562060ef26b2a2bac350be92347f5d1558146a`,
  and `43d4429bc73766f122578a52d9489ae1634f2614dd12fe4962bdcd2fd2a4cd58`.
  The durable submission note SHA-256 is
  `f6bcba2f5d1c7718e73f3b56558a06e3495b5e405df05f67ffd1f362c4088335`.
- Run/runtime roots are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fixedsuite_case22_tree_b1d44427_20260724T142000Z`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_fixedsuite_case22_tree_b1d44427_20260724T142000Z`;
  both contain `SAFE_TO_DELETE`.
- This is a pending full-case generalization arm, not a score change.

# 2026-07-24: case 5 gains are non-monotonic through iteration 7

- Frozen-fixture science `11564053` completed RECOVAR iteration 1 with the
  RELION schedule boundary: current size 56, 30.22 Angstrom resolution, Pmax
  1.0, and next size 56.  The direct float64 real-reference handoff was active.
- The bounded `4e-6` top-two tree rescore examined all 100,000 particles and
  changed four winners: half 1 examined 50,059, found 75 ambiguous, and changed
  three; half 2 examined 49,941, found 79 ambiguous, and changed one.  The
  accepted baseline's three assignment exceptions have half-set distribution
  one/two, so the intervention's three/one changes cannot be exactly that
  previously reported exception set.
- CPU FSC audit `11567287` completed `0:0` in 52 seconds with 2,104,548 KiB
  maximum RSS.  It correctly measured the new same-GPU pair, but its old
  trajectory was compared to the new run's RELION maps.  Superseding
  within-pair audit `11568205` shows numbered iteration-1 merged
  RECOVAR-versus-RELION FSC-AUC improving from `0.9999999996276892` to
  `0.9999999997976555`.  Half-1/half-2 improve from
  `0.9999999992895403`/`0.9999999994962860` to
  `0.9999999997257697`/`0.9999999995988572`.
- Merged GT FSC-AUC moves from `0.10326684532558558` to
  `0.10326688438429595`, toward RELION's `0.103266904543226`.  The metric is
  shellwise FSC/FSC-AUC only; correlation was not computed.
- CPU FSC audit `11567559` completed `0:0` in 68 seconds with 2,222,748 KiB
  maximum RSS at numbered iteration 2.  The within-pair cross-engine
  improvement persists: merged FSC-AUC moves from old
  `0.9999999515577382` to new `0.9999999722278509`; half-1/half-2 values
  move from `0.9999999331905616`/`0.9999998867172372` to
  `0.9999999927643750`/`0.9999999049878318`.
- Merged iteration-2 GT FSC-AUC is `0.1077581636081282`, versus old
  `0.10775713440494669`.  Their matched RELION values are
  `0.10775766121678897` and `0.10775680699055788`, respectively, so GT
  closeness is slightly worse even though cross-engine FSC-AUC improves.
  Only the terminal strict trajectory can decide acceptance.
- CPU FSC audit `11567932` completed `0:0` in 112 seconds with 2,197,784 KiB
  maximum RSS and shows that the gain survives iterations 3 and 4.
  Iteration-3 merged new-versus-RELION FSC-AUC is
  `0.9999994286732671`, versus the matched old pair's
  `0.9999992374179649`; iteration 4 is `0.9999981782699359`, versus old
  `0.9999980697306444`.
- Iteration-3 GT FSC-AUC is `0.10909314872619791`, versus old
  `0.10909390516870988` and RELION `0.10908983473527732`.  Iteration-4 GT
  FSC-AUC is `0.10907400326716823`, versus old `0.10906393775269568`.
  Matched RELION is `0.10908983473527732` new versus
  `0.10909082348729723` old at iteration 3, and `0.10908224561458033` new
  versus `0.10907050037496863` old at iteration 4.  GT closeness is therefore
  slightly worse at both later boundaries.
- Audit stdout and launcher SHA-256 values are
  `2e0e47f426284ed533f4b48e6b945a9bc715866ea0ec6d13efeb5c231339d890`
  and `05b06cfba9446797f738e179734069d4c028b359b642898a979ac91b6e2f1091`.
  Runtime root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case5_iter1_analysis_20260724`.
  Pre-audit `11567277` failed before comparison on a missing `scripts` import
  path and produced no scientific result.
- Iteration-2 stdout, analysis script, and launcher SHA-256 values are
  `35be2613fbac7120b73e0f4b3e06c33df8cae8847f7f995e47e1fab8b484be3c`,
  `e0946cd286ff10d282e6707dbc4a3dcf9fa3fc69cf04a7e171531b9aa49a4924`,
  and `6af89219f662eae944811cdb678dd8910fb10b2d6e5c9f7c905404c0e2f608b7`.
  Runtime root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case5_iter2_analysis_20260724`.
- Iteration-3/4 stdout, analysis script, and launcher SHA-256 values are
  `f604d258fac128f92f2f5aca30cdeb8964a9c95b57821d22d414b1a4c4041565`,
  `1d0919f24b3130d4a9e7e7a702201cc1854b389636fc553ac51f99b373054404`,
  and `dc213e2fabad826e7a3e13001abe683cc5d9da3c810fe6f31674a65bcbfb8ad3`.
  Runtime root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case5_iter3_4_analysis_20260724`.
- Superseding within-pair audit `11568205` completed `0:0` in 265 seconds
  with 2,227,060 KiB maximum RSS.  It binds the new pair to physical GPU
  `GPU-49c1a223-be61-858b-49d8-d8b0347ac252` and the old pair to
  `GPU-ab7221db-5a74-4e07-9521-0c63530c053d`, compares each RECOVAR
  trajectory only to its co-allocated RELION trajectory, and separately
  measures RELION cross-run drift.  Its stdout SHA-256 is
  `4de004bec44639b7aa627ac0f9fb999ee92135c0f1084c4a4fb54f91c01ec2b7`;
  shared analysis-script SHA-256 is
  `74c611696cd553c7526f2c170d2cf4db192af90bddb36eda035b737b0173a236`.
  Runtime root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_intervention_within_pair_v2_20260724`.
- Iteration 5 completed at current size 110, 23.65 Angstrom resolution, Pmax
  `0.445909`, and next size 110.  Within-pair audit `11568517` completed
  `0:0` in 85 seconds with 2,225,104 KiB maximum RSS.  Merged cross-engine
  FSC-AUC improves from `0.9999712213514595` to `0.9999860148597615`,
  reducing the FSC defect by about 2.06-fold after the smaller iteration-4
  gain.  Half-1/half-2 improve from
  `0.9999445061818316`/`0.9999416130352303` to
  `0.9999753442588191`/`0.9999702087213115`.
- Iteration-5 GT FSC-AUC is `0.1090079564338494` versus matched new RELION
  `0.10899736116509263`; the old pair is `0.10900573347159732` versus
  `0.10900394979588701`.  GT closeness worsens despite the cross-engine gain.
  Audit stdout SHA-256 is
  `de0c6f4e976ca4a304cac962474785fb71bb8afd3184460106c6c211644c2301`.
  Shared v3 analysis-script SHA-256 is
  `3e67f3de7212c9dd29438eb2aa58a4b8a9d993a04e4ba34836cdafa3dd3db9bb`;
  runtime root is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_intervention_within_pair_v3_20260724`.
- Iteration 6 completed at current size 110, 23.65 Angstrom resolution, Pmax
  `0.455194`, and next size 110.  Within-pair audit `11568660` completed
  `0:0` in 72 seconds with 2,019,104 KiB maximum RSS and records the first
  cross-engine reversal: merged FSC-AUC changes from old
  `0.9999894895556242` to new `0.9999887297291508`, increasing the FSC defect
  by about 7.2%.  Half-1/half-2 similarly change from
  `0.9999836453293489`/`0.9999775623781483` to
  `0.9999825011727032`/`0.9999761687835128`.
- GT closeness moves in the opposite direction at iteration 6: new
  RECOVAR/RELION FSC-AUC are
  `0.1084485613781282`/`0.1084563187806185`, versus old
  `0.10845076403471834`/`0.1084638385936106`.  This boundary demonstrates why
  neither an intermediate cross-engine improvement nor a GT improvement alone
  can promote the fixed case.  Audit stdout SHA-256 is
  `9d42362e4bf7d6d7a4bd484cee57fd023aff1b8593a0684ae3d068efffd04c25`;
  launcher SHA-256 is
  `935b6f22394aab69b5457f9c5a689b9a907fc10461ca08d268d198f4c1a8963d`.
- Iteration 7 completed at current size 110, 24.73 Angstrom resolution, Pmax
  `0.459120`, and next size 108.  Audit `11569008` completed `0:0` in
  71 seconds with 2,020,800 KiB maximum RSS.  Cross-engine FSC-AUC improves
  again from old `0.9999852513666029` to new `0.9999867932543595`, about a
  1.12-fold smaller defect, while GT closeness worsens: new
  RECOVAR/RELION are `0.10801188894332099`/`0.10799429622435769`, versus old
  `0.1079925250128447`/`0.10798164272984055`.
- RELION new-versus-old FSC-AUC is `0.9999846643652367` at iteration 7, so
  native physical-GPU/run drift is comparable to the within-pair
  cross-engine defects.  The small late old/new delta is observational and
  cannot by itself establish intervention causality.  Terminal same-GPU
  strict acceptance remains authoritative.  Audit stdout SHA-256 is
  `4cec2f90b40bf1d5d26bce43a0e427d2662063d6b3e1ef49e5d82a192be5789b`;
  launcher SHA-256 is
  `34fde3a4485f151d1b833e9df409b4a5ec9b86b5867fece44be4f1542e4dd47c`.
- Shell-profile audit `11569181` completed `0:0` in 31 seconds with
  2,018,944 KiB maximum RSS.  At iteration 6, the full AUC delta is
  `-7.598264734065552e-7`; shells 1--64 contribute
  `-7.190593914567778e-7` (`94.6%`), while shells 64--126 contribute only
  `-4.076708185163369e-8`.  The largest losses are at shells 53--56
  (`-1.78610e-5` to `-3.51685e-5`).
- RELION new-versus-old FSC at shells 53--56 is only
  `0.999542386862`, `0.999101729683`, `0.998700527702`, and
  `0.998564842746`.  Neither old nor new within-pair curve has any shell below
  `0.995`; their minimum non-DC values are `0.999922172364` and
  `0.999928750100`.  The observed reversal is therefore a run-sensitive
  mid-shell butterfly rather than a high-shell collapse or evidence for a
  new production arithmetic/scheduler patch.
- At iteration 7, only 16 of 126 finite non-DC shells have negative
  new-minus-old delta, and the low/high contributions to the positive AUC
  delta are `+1.1915097656998164e-6` and
  `+3.503779907609861e-7`.  Audit stdout, analysis-script, and launcher
  SHA-256 values are
  `3ad632ad35f3389dd5d83336a4fdd04898892f339685557fad94682de93bf6e7`,
  `c888cb76bd570696b592dc067fae3279642732e032dc262d1fdb08179a776951`,
  and `a1f8f006706e205a8c056b89c7f3f7610f2620caca831b7f9d6c53e180ed61ee`.
  Runtime root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case5_late_shell_profile_20260724`.
- Iteration 8 completed at current size 108, 24.73 Angstrom resolution, Pmax
  `0.463344`, and next size 108.  Audit `11569355` completed `0:0` in
  69 seconds with 2,018,832 KiB maximum RSS.  Merged cross-engine FSC-AUC
  improves only from old `0.9999763860186208` to new
  `0.9999770865953913`, about a 1.03-fold smaller defect.
- Iteration-8 GT closeness improves: new RECOVAR/RELION FSC-AUC are
  `0.10773940054308494`/`0.10772735678433272`, versus old
  `0.1077489879376356`/`0.10772242824743296`.  RELION cross-run FSC-AUC is
  `0.9999734868363788`, comparable to both cross-engine defects, so the small
  old/new gain is observational rather than causal.  Audit stdout and
  launcher SHA-256 values are
  `20619a56023f0bb95ac0ec8fb4edab98a8d0fa09bb40f251eee20ff130b7f2fe`
  and
  `7586d4c40c0aed0417c37e563e4d061fb2ad08e55acc0ff981e52ecf25cf650d`.
- Array audit `11569647` completed all four late case-5 boundaries with
  `0:0` task exits.  Iterations 9--12 have new/old cross-engine FSC-AUC:
  `0.9999686598862774`/`0.9999686414562661`,
  `0.9999623868618940`/`0.9999610817486346`,
  `0.9999565853728390`/`0.9999604022763192`, and
  `0.9999560189390468`/`0.9999568196477043`.
- The corresponding RELION cross-run values are
  `0.9999649148988634`, `0.9999540334545888`,
  `0.9999537767945256`, and `0.9999499270733202`.  Thus iterations 9--10
  improve only negligibly, while iterations 11--12 increase the FSC defect
  by about `8.8%` and `1.9%`; all deltas are observational within comparable
  run drift and all cross-engine values remain far above `0.995`.
- GT closeness improves at every one of iterations 9--12.  New
  RECOVAR-minus-RELION absolute differences are
  `1.21549e-5`, `2.14918e-5`, `2.80035e-5`, and `5.55775e-6`, versus old
  `3.23313e-5`, `3.59787e-5`, `4.56079e-5`, and `4.13603e-5`.
  Audit stdout SHA-256 values for array tasks 1--4 are
  `33e4464d2078c87cfc8e204ea7e0840e47d15506136ce5597b44641ec9868267`,
  `cd8bf250b9e415a315bef8a3e4cd7a7454200a8fff7d20463b76ac4530e12204`,
  `fdb625113a27095bb6431dbc4b39060bab248546642a17410ce1640fbcf90e9a`,
  and `4b186783f5c59866152b4004bcff5b212449e1f69a13130662768a7ad246a3a2`.
- This is a positive first-boundary generalization result, not a full-case
  acceptance.  Science `11564053` and strict audit `11564062` subsequently
  completed with a terminal strict failure and exact-topology pass, so
  snapshot `strict-k1-v6-20260724` remains 25/34 strict, 31/34 exact topology,
  and 34/34 evaluated.

# 2026-07-24: case 4 first three boundaries improve under the intervention

- Frozen-fixture science `11563827` completed RECOVAR iteration 1 at current
  size 56, 30.22 Angstrom resolution, Pmax 1.0, and next size 100.  That
  boundary and size decision match the frozen trajectory.
- The bounded `4e-6` top-two rescore examined all 100,000 particles and
  changed six winners: half 1 examined 50,371, found 81 ambiguous, and
  changed two; half 2 examined 49,629, found 87 ambiguous, and changed four.
- CPU FSC audit `11567836` completed `0:0` in 64 seconds with 2,221,000 KiB
  maximum RSS.  Iteration-1 merged RECOVAR-versus-RELION FSC-AUC improves
  from the frozen old trajectory's `0.9999999877211377` to
  `0.9999999993977797`.  Half-1/half-2 improve from
  `0.9999999787358119`/`0.9999999847609748` to
  `0.9999999997117701`/`0.9999999982826421`.
- New merged GT FSC-AUC is `0.104211187030167`, versus old
  `0.10421128611604787` and RELION `0.10421118250310028`.  Both the
  cross-engine and GT comparisons therefore move strongly toward RELION.
- Iteration 2 retains the RELION schedule at current size 100,
  20.92 Angstrom resolution, and next size 116.  CPU audit `11568148`
  completed `0:0` in 66 seconds with 2,221,636 KiB maximum RSS.  The
  within-pair merged FSC-AUC improves from `0.9999992032709832` to
  `0.9999998458568423`, reducing the defect by about five-fold.
- Iteration-2 GT FSC-AUC is `0.1940580090487044` versus matched new RELION
  `0.1940532218186498`; the old pair is `0.19405475510892947` versus
  `0.19405276175018424`.  GT closeness worsens despite the cross-engine gain,
  so this remains a terminal-audit question.
- Audit stdout, analysis script, and launcher SHA-256 values are
  `08a5e44687d21407c3f417459856e195a8cf392f140fc1d30eedc90c71315868`,
  `1ee32ce3e04d6cbf6eb5bd95ffc86d8a9899fb7f26818cdac99366b8f120756c`,
  and `e52a5b43b421d4adde71db333866d8f19e5a98e704526ffea4eb5496f3108909`.
  Runtime root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case4_iter1_analysis_20260724`;
  it contains `SAFE_TO_DELETE` and durable submission metadata.
- Iteration-2 audit stdout, analysis script, and launcher SHA-256 values are
  `35f12a29427e0a8943d91c3eb8458aa63efe4e196e73c1d1ef4283e04558a1ea`,
  `50b8f6bea95ceb98249c7edd89a48fa0a7a825f85348afe77d21a4f657516d65`,
  and `14d305c4b52d9f92a68942fbfd25fbceb6c2b8fc5d0580ba1462a6316e54b46c`.
  Runtime root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case4_iter2_analysis_20260724`.
- Superseding within-pair audit `11568204` completed `0:0` in 146 seconds
  with 2,018,552 KiB maximum RSS.  It binds the new pair to physical GPU
  `GPU-9f98ccbf-3c62-c54f-7409-7eb58845ad4a` and the old pair to
  `GPU-529cbb83-7457-2191-767f-7b3c1a8276c3`.  RELION cross-run merged
  FSC-AUC is `0.9999999999982520` at iteration 1 and
  `0.9999999954796889` at iteration 2.  Stdout SHA-256 is
  `643080d82e3f888092d2df83e17a0ff31072e83a1b0204ab9cfcebd0c95c4435`.
- Iteration 3 completed at current size 116, 16.00 Angstrom resolution, Pmax
  `0.600891`, and next size 132.  Within-pair audit `11568516` completed
  `0:0` in 80 seconds with 2,195,316 KiB maximum RSS.  Merged cross-engine
  FSC-AUC improves from `0.9999933898334938` to `0.9999979804273882`,
  reducing the FSC defect by about 3.27-fold.  GT FSC-AUC also moves closer:
  new RECOVAR/RELION are `0.24646245364215238`/`0.24645781869245548`,
  while old RECOVAR/RELION are
  `0.2464467656324268`/`0.24645908446286174`.
- Iteration-3 audit stdout SHA-256 is
  `d930102385a38a34258685b7c0b7a3e88ceaccfd9f16261e6be3f59122a21dc3`;
  shared v3 analysis-script SHA-256 is
  `3e67f3de7212c9dd29438eb2aa58a4b8a9d993a04e4ba34836cdafa3dd3db9bb`.
  Runtime root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_intervention_within_pair_v3_20260724`.
- Iteration 4 completed at current size 132, 15.54 Angstrom resolution, Pmax
  `0.659163`, and next size 134.  Audit `11568956` completed `0:0` in
  72 seconds with 2,034,360 KiB maximum RSS.  Merged cross-engine FSC-AUC
  improves from old `0.9999766635093335` to new `0.9999913095253029`,
  reducing the FSC defect by about 2.69-fold.
- Iteration-4 GT closeness improves strongly: new RECOVAR/RELION FSC-AUC are
  `0.27128037702419755`/`0.27128960665779256`, while old RECOVAR/RELION are
  `0.27121078968800255`/`0.2712957537838481`.  RELION cross-run FSC-AUC is
  `0.9999983950905497`.  Audit stdout SHA-256 is
  `8cbccbbff28ae26c09f98793d9eb2a3b1f56e59abb051399681b1a7ecd72018c`;
  launcher SHA-256 is
  `0486343bf9ca4e2ef87704935caba5aac17458884d9b8137bd76f376758d5f91`.
- Iteration 5 completed at current size 134, 15.11 Angstrom resolution, Pmax
  `0.689431`, and next size 136.  Audit `11569324` completed `0:0` in
  78 seconds with 2,224,940 KiB maximum RSS.  Merged cross-engine FSC-AUC
  improves from old `0.9999463446317820` to new `0.9999720546013090`,
  reducing the defect by about 1.92-fold.
- GT closeness reverses strongly at iteration 5: new RECOVAR/RELION FSC-AUC
  are `0.279917626085553`/`0.27985627767746785`, versus old
  `0.2798082576477501`/`0.27981080861832763`.  The cross-engine gain alone
  therefore cannot establish quality acceptance.  RELION cross-run FSC-AUC
  is `0.9999901012568425`.  Audit stdout and launcher SHA-256 values are
  `e7badba2501aad48d8357976920e5b295803e7dc1ee8e27534eec0c30eae07b1`
  and
  `f115f4e6dddd6b8b377806b31a71717863f657dcb94f91816bd96b4407028c27`.
- Iteration 6 completed at current size 136, 15.11 Angstrom resolution, Pmax
  `0.710076`, and next size 136.  Array task 0 under job `11569647` completed
  `0:0` in 70 seconds with 2,024,964 KiB maximum RSS.  Merged cross-engine
  FSC-AUC improves from old `0.9999245486508258` to new
  `0.9999532103394377`, about a 1.61-fold smaller defect.
- Iteration-6 GT closeness improves: new RECOVAR/RELION are
  `0.2851142404037022`/`0.2850663249147656`, versus old
  `0.2851382201712396`/`0.28500954818472085`.  RELION cross-run FSC-AUC is
  `0.9999753761283718`.  Audit stdout SHA-256 is
  `9e295a46ef1c9273da94b5f5d05377d9d4307ca1718d65d0154217edb0f5cb99`;
  shared array launcher SHA-256 is
  `ae530abefc4e06706b17ca9a77cb6fad794af7f9ff5bbd81cc187af29cd569c2`.
- This is strong first-boundary evidence, not a full-case acceptance.
  Science `11563827` and strict audit `11563842` subsequently completed with a
  terminal strict failure and exact-topology pass, so the fixed score remains
  25/34.

# 2026-07-24: K=4 same-GPU trajectory reaches iteration 7

- Source-bound science `11565045` runs commit
  `9dcd709b56a28a6f361806b57f5b20aaad3ebeed` on physical A100
  `GPU-6f45f415-9d0b-d562-9ff3-c9fb7bc53aa7` after the accepted inert RELION
  control/capture pair.
- RECOVAR completed iterations 1--4 with current sizes
  `38,38,42,56`, resolutions `60.44,49.45,30.22,27.20` Angstrom, and Pmax
  `1.0,0.069455872,0.265973233,0.597798849`; iteration 5 starts at size 60.
  That schedule exactly matches the three prior diagnostic trajectories
  through this boundary.  Their iteration-4 Pmax values span
  `0.597791747`--`0.597796751`, placing the current value within
  `7.102e-6`.
- Iteration 5 completed at current size 60, 25.90 Angstrom resolution, Pmax
  `0.823400272`, and HEALPix order 1; the dynamic scheduler selected size 62
  for iteration 6.  Both prior corrected `c390f8bf` diagnostics have exactly
  the same size/resolution/order boundary and rounded Pmax `0.8224`.
- Iteration 6 completed at current size 62, 22.67 Angstrom resolution, Pmax
  `0.920524`, and HEALPix order 1; the scheduler selected size 68 for
  iteration 7.  At this boundary the vector and scalar auditors were still
  dependency-gated, so this was trajectory telemetry rather than an
  acceptance claim.
- Iteration 7 completed at current size 68, 21.76 Angstrom resolution, Pmax
  `0.909746`, and HEALPix order 1; the scheduler selected size 70 for
  iteration 8.  This size/resolution/order boundary matches the corrected
  `c390f8bf` trajectory, whose first strict map failure was only iteration 8.
- Iterations 8--10 completed at sizes `70,72,74`, resolutions
  `20.92,20.15,19.43` Angstrom, and Pmax `0.9237,0.9470,0.915066`.
  Iteration 10 remained unconverged after 2,326.8 seconds and selected size
  76 for iteration 11.  The job was configured for 15 numbered iterations,
  so this is a continuing trajectory rather than a terminal iteration-10
  result.
- Iteration 11 completed at size 76, 18.76 Angstrom resolution, Pmax
  `0.958718`, and HEALPix order 1 after 2,037.8 seconds.  It remained
  unconverged and selected size 78 for iteration 12.
- Independent CPU map audit `11573095` failed closed (`2:0`) in 377 seconds
  with identity class assignment.  Classwise cross-engine FSC-AUC is
  `0.995505295,0.994509131,0.994150545,0.995592423`; classes 2 and 3 are
  below the unchanged `0.995` gate.  GT deltas are
  `+6.74435e-5,-7.30310e-5,-3.62876e-5,-6.19497e-5`, all well within the
  `-0.002` gate.  This is a trajectory-parity failure rather than a GT-quality
  collapse.
- Iteration-11 audit JSON, analyzer, and launcher SHA-256 values are
  `2fd75c707c39ececc5ac54e270e010995d966c51f60e0759754e52a1209c40d4`,
  `d74fdc8b400f16707959fdf2a45b654a9991408cc2dbf207752d2010ba2a5236`,
  and
  `228abf5d78b93ca34d2eae986f0c9686fe1271535c3f2ec5e0fb3603cbbb9cbc`.
  Run/runtime roots are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_iteration11_early_fsc_20260724`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k4_iteration11_early_fsc_20260724`;
  both contain `SAFE_TO_DELETE`.
- Independent fail-closed array audit `11573422` brackets the first map failure
  exactly.  Iteration 9 passes with identity assignment, classwise
  cross-engine FSC-AUC
  `0.996581803,0.996373177,0.995753485,0.996810831`, and worst GT delta
  `-5.72845e-5`.  Iteration 10 passes with identity assignment, classwise
  cross-engine FSC-AUC
  `0.996370254,0.995522852,0.995367310,0.996596692`, and worst GT delta
  `-6.58008e-5`.  The two tasks completed `0:0` in 300/236 seconds with
  4,400,824/4,610,984 KiB maximum RSS.  Therefore numbered iteration 11 is
  the first strict cross-engine map failure in this same-physical-GPU
  trajectory.
- Iteration-9/10 analysis JSON SHA-256 values are
  `a6b70467b3f651b9cae1df5983ce9065ea6a654452bc00ef4a6a60b53b492daf`
  and
  `9e50e515a4215817e6c71163cb29f23fe1b423ea9b7d122f9a8462c466ddc1c4`.
  Shared analyzer/launcher SHA-256 values are
  `f97ef19fb74727aa8e076adb5321bde83ed21b2d6891ef5690b51d3ed4a49d06`
  and
  `34a95a5a38eab1064b3aaa73f1651569b75cc9883a8925b730da8c8d3b758858`.
  Run/runtime roots are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_iteration9_10_early_fsc_20260724`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k4_iteration9_10_early_fsc_20260724`;
  both contain `SAFE_TO_DELETE`.
- CPU shellwise localization `11573825` completed `0:0` in 432 seconds with
  4,674,968 KiB maximum RSS and preserved every full-grid FSC value from the
  accepted iteration-10/11 audits.  At iteration 11, classwise cross-engine
  FSC-AUC through RELION's reported shell 29 is
  `0.999817891,0.999753711,0.999770535,0.999866473`; through the current-size
  radius 38 it is
  `0.998743644,0.998416953,0.998362350,0.998989750`.  The full non-DC
  `0.995` gate still fails for classes 2/3 because the beyond-radius band is
  `0.992837146/0.992348888`.
- The fraction of each class's positive shellwise FSC deficit beyond the
  current-size radius is `91.55%,91.29%,91.54%,93.08%`.  Iteration-10 to
  iteration-11 mean FSC changes through the old radius are only
  `-0.000198260,-0.000231292,-0.000275279,-0.000197622`, versus
  `-0.001132232,-0.001333115,-0.001602355,-0.001333367` beyond the new
  radius.  This localizes the strict failure to the low-energy full-grid
  tail/post-processing boundary and does not support changing K-class
  scoring, support, or reconstruction arithmetic from this result.
- Shell-localization JSON, shellwise NPZ, analyzer, and launcher SHA-256 values
  are
  `93dae31fefe09ca8dc58f388ac1e548aa3bbdac511b29263a1f19f3277b03b64`,
  `c9f38013d40a5d74b9c97c25687f61bb23529e63ee2f76726ee7007599eda2ef`,
  `140d3efb9b16e9b3edf53cde5f235b506c464eb5b6f9cecf6bc3ff0b2d165d2c`,
  and
  `8f472bca57a05ba71ed06bc92e5c4ae302b429071cdb5d17b4c6318ee609e07c`.
  Run/runtime roots are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_iteration10_11_shellwise_localization_20260724`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k4_iteration10_11_shellwise_localization_20260724`;
  both contain `SAFE_TO_DELETE`.
- Independent fail-closed array audit `11575379` extends the numbered map
  trajectory through iterations 12--13.  Iteration 12 classwise
  cross-engine FSC-AUC is
  `0.995634590,0.994695012,0.994672195,0.995469749`; iteration 13 is
  `0.995436809,0.993535520,0.992658054,0.994713432`.  Identity assignments
  remain exact at both boundaries.
- Worst GT deltas are only `-0.000110190` and `-0.000036050`, respectively,
  inside the unchanged `-0.002` quality gate.  Both tasks fail closed
  (`2:0`) solely because the full-grid cross-engine gate remains below
  `0.995`; this is continued trajectory divergence, not a GT-quality collapse.
  The tasks ran 254/233 seconds with 4,495,748/4,514,984 KiB maximum RSS.
- Iteration-12/13 analysis JSON SHA-256 values are
  `441102f6bf3ba6b348a70e51be2aaac2922c57ed8d3f4454e9ebf8e266dd244b`
  and
  `6b48248176f485eb9eed92fa5013e243fdfccc78d25c901325e093c20375738f`.
  Shared analyzer/launcher SHA-256 values are
  `f97ef19fb74727aa8e076adb5321bde83ed21b2d6891ef5690b51d3ed4a49d06`
  and
  `89d4a21250eab4c97e7f2e594e5e0f9c0daa0c445789e55603042e591b6b10bd`.
  Run/runtime roots are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_iteration12_13_fsc_20260724T160500Z`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k4_iteration12_13_fsc_20260724T160500Z`;
  both contain `SAFE_TO_DELETE`.
- Independent fail-closed audit `11577443` extends the map trajectory through
  numbered iteration 14.  Identity class assignment remains exact.
  Classwise cross-engine FSC-AUC is
  `0.994832774,0.993014076,0.992320840,0.995034148`; classes 1--3 fail the
  unchanged `0.995` gate, so the job exits `2:0`.
- RECOVAR-minus-RELION GT FSC-AUC deltas are
  `+0.000034991,+0.000095231,-0.000011025,-0.000010560`, all far inside the
  unchanged `-0.002` gate.  The iteration-14 failure is continued strict
  cross-engine full-grid divergence rather than GT-quality loss.  The task
  ran 233 seconds with 4,610,596 KiB maximum RSS.
- Iteration-14 analysis JSON/stdout, shared analyzer, and launcher SHA-256
  values are
  `21771a98e8a20addc93e6abe2c106391f97bddccc5a85803a12c29b0d1d57ded`,
  `f97ef19fb74727aa8e076adb5321bde83ed21b2d6891ef5690b51d3ed4a49d06`,
  and
  `100357b0fb9c59cba86da3908fcd1c37a3ec305af5038ea9f3a29c5478e15e8f`.
  Run/runtime roots are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_iteration14_fsc_20260724T172000Z`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k4_iteration14_fsc_20260724T172000Z`;
  both contain `SAFE_TO_DELETE`.
- Science `11565045` completed all 15 configured iterations `0:0` in
  `09:03:48` with 60,094,852 KiB maximum RSS.  It remained unconverged and
  correctly skipped forced final all-data.  Independent iteration-15 audit
  `11577956` fails closed (`2:0`) in 252 seconds with 4,611,312 KiB maximum
  RSS.
  Identity map assignment remains exact, while classwise cross-engine
  FSC-AUC is
  `0.994459232,0.993069734,0.992039376,0.994497731`; all four classes fail
  the unchanged `0.995` gate.  GT deltas are
  `+0.000115765,-0.000113423,+0.000041339,+0.000018044`, all inside the
  unchanged `-0.002` gate.
- Iteration-15 analysis JSON, shared analyzer, and launcher SHA-256 values are
  `0a159a05f05f4fc641e6d76bf5796d120268ebb92bf7a3dfcf7cec3030e42e1f`,
  `f97ef19fb74727aa8e076adb5321bde83ed21b2d6891ef5690b51d3ed4a49d06`,
  and
  `7f80134f31ec1a15134af28a9bf06bd04b938e22971743e1987af76d2606ce7c`.
  Run/runtime roots are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_iteration15_fsc_20260724T213615Z`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k4_iteration15_fsc_20260724T213615Z`;
  both contain `SAFE_TO_DELETE`.
- The original vector audit `11565121` exposed two auditor-only assumptions:
  it admitted the native panel NPZ into the scatter-signature set and then
  assumed at least one exact cross-engine rotation match.  Corrected,
  hash-pinned vector/scalar audits `11577999`/`11578000` complete `0:0` in
  11/10 seconds.  Both classify
  `rotation_support_difference_precedes_operand_value_comparison`: across
  the exact 96-particle class-2 panel, RELION contains 56,720 prescatter
  rotations versus 111 RECOVAR contributor rotations, 13 RECOVAR particles
  have zero class-2 contributors, and no rotation matches within `1e-6`.
  Operand/scalar metrics are deliberately unqualified.
- Vector/scalar JSON SHA-256 values are
  `ed96c45cdc8c142fd854bce894aacd276e26f4d3a9e8bdc4d4e14b73703d9ae9`
  and
  `166b7aae8066f94617576ba14d86249631ab3b606f898f420bdaf800869f3ab6`.
  Comparator SHA-256 values are
  `ca4335f09cfdb84a1eb514ac36f2df5b3aa291afaf4175442bc63b1b99b8567a`
  and
  `5e0c0199eff97b4d28ca09ca3f96f448c054a55f98db581cb931b89b8bbf10f1`;
  launcher SHA-256 values are
  `f61d3d254d7514bb3b86e2ad5999f3ce2849e90a3ad393db81c2388aaef9de7b`
  and
  `d9a194920e9aea6c8330fd2a602aea5f81c728f58c0ac38b1e85a0f798845a88`.
  Full terminal FSC/class-assignment audit `11578043` subsequently completed
  from
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_terminal_full_audit_20260724T214500Z`;
  its complete red-gate result is recorded below.
- The 56,720-versus-111 interpretation above is superseded: the RELION
  `.bpre-v1.bin` `rotations` array is the complete candidate table, while
  only `orientation_local` values referenced by emitted rows are positive
  class contributors.  It was therefore invalid to compare that whole table
  against RECOVAR's positive contributors.
- Corrected contributor audit `11580683` completed `0:0` in 9 seconds with
  308,704 KiB maximum RSS.  It compares the 120 RELION emitted contributors
  against 111 RECOVAR contributors, retains all 13 explicit RECOVAR
  zero-contributor particles, and finds zero exact per-particle rotation
  matches at `1e-6`.  Pixel/value comparison remains deliberately
  unqualified.  This fixes the candidate-versus-contributor category error,
  but it does not localize the parity gap because the restarted RELION capture
  and uninterrupted RECOVAR capture used different sampling perturbations.
  The corrected report, comparator, and launcher SHA-256 values are
  `c6205255a5fb7c7ce1e8f4d376d0d054b0a079975369a1316d6424b4aee873ea`,
  `ed4eb83131d7be96006c48add1055b06f5623e740a2670b3510e8f2d4e82a3b2`,
  and
  `9794560e6ca9c0384d1879770deee622c0a4565fd9b505acd7e076546537d301`.
  Preliminary job `11580595` failed before audit work because its import
  provenance was checked before changing into the exact source checkout;
  preliminary `11580606` exposed and led to removal of a vacuous pixel
  support pass when no rotations aligned.
- Slurm audit `11580995` and its
  `coarse_parent_support_difference_precedes_fine_scoring` classification are
  rejected.  The v1 auditor treated RELION direction-major
  `orientation_class_key` and RECOVAR psi-major
  `active_global_rotation_indices` as one index convention without a matrix
  gate.  Its report, auditor, unit-test, and launcher SHA-256 values, retained
  only as rejected provenance, are
  `d90f970ddb98c1c31ab9de4c18949fce20e3150581c66d7945b4d4e143bbd508`,
  `e65ee28bad9c5239cf69b300c489176c9b6f361356da65cfc0f5c84f53bbacc0`,
  `971cec81f630c2b904f9de9c0b59c02534ad69abcfbb362084ef89aef78bf9f5`,
  and
  `934059d9fc630f5299001f61dbb20555bc68de475816d524dab73464fc35b2d5`.
  Run/runtime roots are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_candidate_parent_audit_f48bcbc0_20260724T232300Z`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k4_it10_candidate_parent_audit_f48bcbc0_20260724T232300Z`;
  both contain `SAFE_TO_DELETE`.
- The v2 auditor now generates each engine's expected fine grid, validates
  captured matrices before interpreting integers, converts RELION
  direction-major parents into canonical psi-major identities, and refuses a
  localization when sampling perturbations differ.  Independent Slurm job
  `11581784`, pinned to commit `0b5182b5`, completed `0:0` in 13 seconds with
  686,404 KiB maximum RSS.  The full 96-particle CPU audit validates RELION
  geometry to max-abs `5.0664e-7` and RECOVAR geometry to `1.7881e-7`, then
  reports `status=invalid_comparison` and
  `incomparable_sampling_perturbation_precludes_cross_engine_support_claim`.
  Restarted RELION used `-0.12306`; uninterrupted RECOVAR used `+0.096421`,
  exactly matching the uninterrupted RELION oracle's iteration-10 sampling
  state.  The `0.219481` perturbation delta invalidates the former
  7,090/6,857/1,677 parent-overlap and 14/96 contributor-retention claims as
  parity evidence.  The corrected report SHA-256 is
  `077a611d1a6025834316b41d3522efea1d008a3ecbbb0a0f645c3402902e5486`.
  Its run/runtime roots are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_candidate_support_geometrygate_slurm_0b5182b5_20260725T005000Z`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k4_it10_candidate_support_geometrygate_slurm_0b5182b5_20260725T005000Z`;
  both contain `SAFE_TO_DELETE`.
- Map-only early audit `11569628` completed `0:0` in 217 seconds with
  4,398,616 KiB maximum RSS.  Identity matching is selected for both
  RECOVAR-to-RELION and matched-pair-to-GT assignments.  Classwise
  cross-engine FSC-AUC is
  `0.9975035093,0.9968067962,0.9969715869,0.9980573808`; GT deltas are
  `+1.74559e-5,-7.69739e-5,+4.69329e-5,-3.82329e-5`.  All map/GT gates pass
  at unchanged thresholds `0.995/-0.002`.
- Pre-audit `11569464` produced no scientific report because it attempted to
  load terminal `refinement_results.npz` before science had written it.
  The accepted rerun deliberately reports class agreement unavailable;
  dependent terminal auditors remain authoritative.  Accepted stdout,
  analysis-script, and launcher SHA-256 values are
  `6b71b4315a59893ad1cec90a3c96543e690c289d9d5cfdaa5e389204e846a944`,
  `81d8528b63888b31d69ac1b90989de34b33887d95cc8f987830a246aa6d41fcd`,
  and `b3df857073af97cf86cc4913abbf3b3409a37379b073e05b5a8407d3d4a80384`.
  Runtime root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k4_iteration7_early_fsc_20260724`.
- The numbered full-grid gate remains red from iteration 11 onward.  No K=4
  acceptance or K=1 score change is claimed.

# 2026-07-24: case 22 rules out the firstiter intervention

- Frozen case-22 science `11566712` completed the first two RECOVAR
  boundaries.  Iteration 1 matches the RELION bootstrap schedule at current
  size 56, 30.22 Angstrom resolution, Pmax 1.0, and next size 60.  The
  bounded top-two rescore examined all 3,000 particles, found one ambiguous
  particle, and changed no winners.
- Iteration 1 is effectively unchanged: merged new-versus-old FSC-AUC is
  `0.9999999999842022`, and merged new-versus-RELION FSC-AUC is
  `0.9999999999730657` versus old `0.9999999999732839`.
- CPU FSC audit `11567536` completed `0:0` in 25 seconds with 681,148 KiB
  maximum RSS at iteration 2.  The intervention remains a byte-neighbor of
  the frozen old trajectory (`new-versus-old` merged FSC-AUC
  `0.9999999999416288`) and does not close the known RELION butterfly:
  new-versus-RELION is `0.9999933521000186` versus old
  `0.9999933521252655`.  Merged GT FSC-AUC is
  `0.21596520383729964` versus old `0.21596524819482166` and RELION
  `0.21606265237491196`.
- The direct-real-reference plus bounded first-iteration top-two hypothesis
  is therefore rejected for case 22.  Its divergence begins downstream of
  this path; the autonomous run and strict audit remain useful terminal
  evidence but cannot validate this intervention as the cause.
- Iteration-2 audit stdout, analysis script, and launcher SHA-256 values are
  `7b40aa847778dcee2604b2dd9256ab9b676538f9291f6ebe32e4c0ef4e7679dd`,
  `557c8dde84869a462426ead58456e8c009825a935632b86a7123722f6eb9bff3`,
  and `ef9ad571708fd2bceebd1fd2bd3775f90dd724c89efc311270f2f40be81f431a`.
  Runtime root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case22_iter2_analysis_20260724`.
- Full science `11566712` completed all artifacts but returned the launcher's
  expected scientific exit `2:0` because its embedded quality summary failed.
  RECOVAR converged after 10 numbered iterations versus RELION's 11 and then
  ran one convergence-valid final all-data iteration with grid correction
  off.  The original `afterok` audit `11566739` was dependency-impossible and
  was canceled at zero runtime.
- Unchanged terminal audit `11567655` was therefore run directly against the
  complete artifacts.  It failed closed (`1:0`) in 43 seconds with 763,960
  KiB maximum RSS: FSC audit status 2 and exact-topology audit status 2.
  Numbered iteration 9 is the first FSC failure
  (`0.9898307166274213 < 0.995`); numbered iteration 10 recovers to
  `0.9974387786446363`, but final merged cross-engine FSC-AUC is only
  `0.8262609916592335`.
- Exact topology fails at iteration 9 on current size
  (`RELION=70`, `RECOVAR=72`) and HEALPix order (`5` versus `4`), retains the
  HEALPix mismatch at iteration 10, and has numbered counts 11 versus 10.
  Final merged GT FSC-AUC delta is `-0.00043461968374969295`.
- Terminal FSC JSON, shellwise NPZ, and topology JSON SHA-256 values are
  `b304bcce0ad99684a73b65d2fc4e249ce2b1931b9db72c76737c172ee48385fe`,
  `ce0f797fb7408205941183de744acb99e76241c7966257a033dc946d8ee2f450`,
  and `5883f1bab7565b05c89f4047159ea5661e67002a181b8fa27bf7ec7790b8b4ae`.
  Audit stdout SHA-256 is
  `0543e42fbe870b0742cebeeaa6b67153f9c52f29e24e9e521ea250d1536604ba`.
- CPU scheduler-boundary audit `11568050` completed `0:0` in 3 seconds with
  53,552 KiB maximum RSS and localizes the topology split.  Iteration 8 still
  has equal size 70/order 4 topology, but RELION remains at shell 19
  (`28.631579` Angstrom, resolution-stall counter 2) while RECOVAR crosses to
  shell 20 (`27.20` Angstrom, counter 0).
- At iteration 9 the independently measured angular accuracies are
  `2.479` degrees for RELION and `2.469` degrees for RECOVAR.  For both,
  the 1.875-degree effective step fails the fine-enough test.  Only RELION's
  prior resolution-stall state is ready to advance, producing size/order
  `70/5` versus `72/4`.  This rejects a scheduler threshold/oracle patch: the
  topology failure amplifies the upstream map/FSC-shell butterfly.
- Scheduler audit stdout, analysis script, and launcher SHA-256 values are
  `69d75f19f1887630ac54f09a09b374757de47090f5910f40652c2f8403916f09`,
  `d7e8d06f2b39876672e5a9fbc9a1216ed34dcd7d458519d2e3ad1f194c3250c2`,
  and `cca79e35937a10573e02e298d5ea5c075dbdae9466beeb9f4390f2ebed9db529`.
  Runtime root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case22_scheduler_boundary_20260724`.
- Snapshot `strict-k1-v6-20260724` remains 25/34 strict, 31/34 exact
  topology, and 34/34 evaluated.

# 2026-07-26: exact case-5 result classification is pinned and queued

- Integration commit
  `6c483fba9f779533d169a65d67b867b90a443235` adds
  `scripts/classify_relion_recovar_fine_top_discrepancy.py`.  The classifier
  requires exact Euler-matrix matching for original particle `65070`,
  current-size `56`, and the eight exact iteration-1 particle-state fields
  for all `100000` stock-versus-dump-enabled RELION rows.
- The causal rule is exact and predeclared: if both engines assign exactly
  equal raw pre-prior scores to the two cross-winner candidates, the
  discrepancy is `compact_candidate_tie_order`; if either engine
  distinguishes them, it is `fine_score_arithmetic`.  The report explicitly
  sets `scorecard_change_admissible=false`; it is not an FSC checkbox.
- Classifier/test SHA-256 values are
  `9ca93a25c2b795bc10384bd664d4c3ca30a366e66b752c7734c687969971e976`
  and
  `e1582846e8078b29d292ecc7777ffdd00b51ccfc1ba4b1105b3ddbb7744a5442`.
  Targeted classifier plus comparator validation passes 31/31 tests, and
  scoped Ruff passes.
- Clean detached audit source is
  `/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_fine_classifier_6c483fba_20260726`
  at the pinned commit above.  Dependent CPU audit `11633508` is queued as
  `afterok:11602720`; launcher SHA-256 is
  `22b5f65320d797bf42d461411a0649adce06427ae1c97c3b22f2ab1827b1fd19`.
  Its run root is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case05_p65070_exact_relion_fine_20260725T083000ET`,
  and its runtime root will be
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case05_p65070_fine_classification_11633508`;
  both are marked or required `SAFE_TO_DELETE`.
- The checked metric remains `strict-k1-v7-20260726`: 26/34 strict
  FSC/FSC-AUC passes, 32/34 exact-topology passes, and 34/34 evaluated.

# 2026-07-26: immutable current-head case-32 chain submitted

- The eight remaining strict failures were ranked for a cheap independent
  current-head check. Case 24 is only `0.000195` below the FSC gate, but its
  exact stock RELION winner is already proved launch-sensitive at a one-ULP
  boundary; it is not an admissible tie-force target. Case 32 has exact old
  topology, only 10,000 particles at grid 128, and no current-head fixed-input
  rerun. Its checked final merged cross-engine FSC-AUC is
  `0.97450050098333`.
- Clean detached source
  `/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_case32_current_a03c9fd1_20260726`
  at `a03c9fd1359c47e69f16904935e6cb755d078b18` submitted setup
  `11633606`, same-A100 autonomous science `11633607`, summary `11633608`,
  and unchanged strict FSC/topology audit `11633676`
  (`afterok:11633607`).
- The run/runtime roots are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fixedsuite_case32_current_a03c9fd1_20260726T124000ET`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_fixedsuite_case32_current_a03c9fd1_20260726T124000ET`;
  both contain `SAFE_TO_DELETE`.
- Setup/science/summary launcher SHA-256 values are
  `ea333ecbb1f672a528a9b6222b93381cc8ac4a12404c217b027718934bee003c`,
  `02ffc2b7b2c58fe51e38f34d073785247b72a2df69ed4b9cf6c73237880ce74c`,
  and
  `473d05b6e38d5d4f8685269277833876da2762ddefc4703aa9aa022add8407de`.
  Strict-audit launcher SHA-256 is
  `0f98d966de8eeb8fc66ed062d5da32939b31232007c915501ee516e804738718`;
  accepted FSC/intermediate auditor SHA-256 values remain
  `2154c7f11519dee1756b24342bb962b35501bf3202c50bf1e9eac5267dd2e515`
  and
  `eb160be9c13762aae67a92aa2b21243d339d09175d14bd503196fa02bd138bba`.
  Pending setup allocation `11633606` was reduced from 64G/1h to 16G/10m
  after the matching prior setup completed in 26 seconds at 2,838,076 KiB
  peak RSS.  This is a scheduler-only update; launcher and fixture bytes are
  unchanged. The adjusted setup completed `0:0` in 32 seconds on
  `della-r3c2n1`. Science `11633607` started immediately on `della-l07g6`,
  passed the clean-source gate, and recorded physical A100 UUID
  `GPU-bd720f2f-c28a-09c0-d51e-d08b1897125a`.
  Read-only exact-identity particle-state audit `11633818` is queued
  `afterok:11633607` to report every numbered and final Pmax, support, pose,
  translation, class, and convergence distribution. It cannot change the
  scorecard. Launcher/auditor SHA-256 values are
  `59b0654de139681209d70add155b2abce5c516c074a6cef04b3a59ffcb681488`
  and
  `ae6e67c0c20385aa3740facd673b1c222f343c99ba2074ede1a04a32a11750cb`.
  Frozen fixture-manifest SHA-256 is
  `422a79a0a7703d92f9777266e8c34ccd3a7cf5963b354e57a7d9a18f227babee`.
- Scorecard mode keeps grid correction unset, final numbered-state replay
  off, forced finalization absent, exact fixture bytes, per-iteration maps,
  and no diagnostic scoring/support overrides. No score changes at
  submission; unchanged FSC/FSC-AUC and topology audits must pass first.
- Stale case-3 audit `11501908`, permanently held by
  `afterok:11501889(failed)`, was cancelled at
  `2026-07-26T12:48:16-04:00`. It could never run and is not part of a
  current evidence chain.

# 2026-07-26: canonical case 3 promotes fixed snapshot v7

- Canonical fixed-fixture science `11587631` and read-only strict audit
  `11632847` completed `0:0` from clean detached source
  `4c8b043a9b80ff12441e36f5a77c6e9f1896197b`.
- All 17 numbered rows pass FSC/FSC-AUC and exact intermediate topology.
  Worst numbered merged cross-engine FSC-AUC is `0.9999619013267681` at
  iteration 10.  Final merged cross-engine FSC-AUC is
  `0.9987827326111832`; RECOVAR-minus-RELION GT FSC-AUC delta is
  `+0.0054263318347904654`.
- The run converged at iteration 17, ran final all-data only after
  convergence, and kept `RECOVAR_FINAL_ALL_DATA_GRID_CORRECT` unset/off.
- FSC, topology, and shellwise evidence SHA-256 values are
  `0c5b3eccf9324b8c6aece1dcba3f920e49ef0da05eafa074fcc9124bf72fa2de`,
  `7e47bb0cdb3e488fcbc72cdcba9df7673989ed7cf5bc095238e4e6eddd72dbd7`,
  and
  `b8358785fd84ff970b4cd4f97483cf98e93f35a6a01d6906abadc0841f59e2bc`.
- Proposal `11633116` failed closed on an audit-log filename lacking the
  literal frozen ID `k1-03` and emitted no ledger.  Byte-identical hard-link
  aliases supplied the required provenance name without modifying audit
  bytes.  Replacement `11633309` completed `0:0` and re-hashed all
  470,170,958,467 pinned fixture bytes.
- Accepted v7 ledger SHA-256 is
  `55fb5042a3768c5d44b89aef72412682c6ebad2d832ba3c2a1b02a6a491c7d8e`.
  Durable proposal root is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_scorecard_v7_case03_d27d397c_20260726T114800ET`.
- Snapshot `strict-k1-v7-20260726` is 26/34 strict, 32/34 exact topology,
  and 34/34 evaluated.  Denominator, definitions, manifest, and thresholds
  are unchanged.
- Exact-H100 discriminator `11602720` remains the next K=1 causal gate; it
  cannot move the frozen score without a complete strict fixed-fixture pass.

# 2026-07-25: K4 candidate completes host arm and holds an 11.33% live speedup

- Same-physical-A100 science `11600592` completed all 15 `host_numpy`
  iterations in `30089` seconds without a forced final-all-data iteration.
  Its `relion_cuda` arm had completed nine numbered iterations at
  `2026-07-25T17:57:22-04:00`; dependent CPU audit `11600593` remains
  `afterok:11600592`.
- The first nine same-GPU timing pairs total
  `16597.981476306915` seconds for `host_numpy` and
  `14716.971381187439` seconds for `relion_cuda`, a
  `-11.332764154512143%` change.  `relion_cuda` is faster in all nine paired
  iterations.  This is a live performance diagnostic, not a quality
  acceptance result.
- The fixed-host FSC health check remains 24/24 direct comparisons through
  iteration 6 at `0.995`, identity permutation, and minimum FSC-AUC
  `0.9977914887513855`.  The complete candidate denominator remains 60 direct
  class checks across 15 iterations and will be emitted by `11600593`.
- The explicit fine-Euler matrix matcher used by exact-H100 case-5
  discriminator `11602720` was revalidated with its focused direct,
  transposed, class-specific, and compact-row matching tests.  Command
  `.pixi/envs/default/bin/python -m pytest -q
  tests/unit/test_compare_relion_recovar_estep_dump.py` passes 27/27 (one
  pre-existing NumPy invalid-divide warning); targeted Ruff also passes.  Job
  `11602720` remains pending for the exact physical H100 UUID with scheduler
  estimate `2026-07-28T10:31:11`.

# 2026-07-25: live JAX/cuFFT preprocessing reduces the K=4 score residual

- Same-A100 job `11598766` completed `0:0` in `00:30:22` from clean commit
  `dd6d4063774e36136bf9551ee828d3e113f46974`. Host NumPy and JAX/cuFFT ran
  sequentially on physical GPU
  `GPU-6f45f415-9d0b-d562-9ff3-c9fb7bc53aa7`.
- The pinned boundary is original index 42987, class 2, current size 74,
  global rotation 2956, translations 56--59. Fine translations, rotation
  indices, parent map, candidate mask, and both priors are exact across arms.
- Host centered residual L2 against the accepted passive RELION CUDA scores
  is `4.8828125e-4`; JAX is `2.44140625e-4`. Residual energy falls exactly
  75%. On the three production-exact RELION translations, L2 falls from
  `3.9867997e-4` to `1.9933999e-4`, also 75% energy removal.
- Host/JAX bounded wall times are 915/886 seconds, so this diagnostic has no
  measured speed penalty. The host class-2 NPZ is byte-identical across three
  independent captures.
- Accepted report:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_preprocess_live_pair_retry_f2ccc270_20260725T043000ET/analysis/LIVE_PREPROCESS_RELION_COMPARISON.json`
  (SHA-256
  `348c462c40c62f4b5a3b83de42fbaee81adf984a8318b2c59db1c7e0da685a74`).
- This proves a live causal preprocessing effect but does not authorize a
  default change. Carry JAX/cuFFT through a K=4 assignment and FSC/FSC-AUC
  trajectory gate, and keep the fully derived RELION-CUDA backend as a
  separate live discriminator.
- Diagnostic-only failures `11597063`, `11597459`, and `11598131` are
  preserved in the same run audit. None produced a paired science result or
  changes the fixed score.

# 2026-07-25: fully derived RELION-CUDA preprocessing confirms the live branch

- Same-A100 job `11599918` completed `0:0` in `00:26:46` from clean commit
  `ede6df86c2644e07de1fec8c30acc7657821e6db`. Host NumPy and `relion_cuda`
  ran sequentially on physical GPU
  `GPU-2f2a8197-bcc8-ec41-fc6f-dfb2b5aaf4fa`.
- Exact topology and the pinned class-2 candidate boundary match the earlier
  live discriminator. Host versus `relion_cuda` residual L2 is
  `4.8828125e-4` versus `2.44140625e-4` over all four candidates and
  `3.9867997e-4` versus `1.9933999e-4` over production-exact translations
  56/58/59. Both scopes remove exactly 75% of residual energy.
- Bounded wall time is 878/722 seconds, making `relion_cuda` 156 seconds
  (`17.77%`) faster on the same A100. This is a stop-after-capture runtime,
  not an end-to-end K=4 performance claim.
- The selected centered signature equals the earlier JAX/cuFFT result, but the
  full panels are not bitwise equal: 315/544 finite pre-prior scores differ,
  with maximum absolute difference `2.44140625e-4`. A trajectory-level
  assignment and FSC/FSC-AUC gate is still required before a default change.
- Accepted report:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_preprocess_relioncuda_pair_ede6df86_20260725T043500ET/analysis/LIVE_PREPROCESS_RELION_CUDA_COMPARISON.json`
  (SHA-256
  `fdaa3280131683974e4f446fd04ff0cb1cec42345859ddc5db982f07b5fbce37`).
  Host and `relion_cuda` class-2 artifact SHA-256 values are
  `ddc8d65de595699107b1e946f0fbe1dcb61d39d43191e67a3a364c6ac863a844`
  and
  `fe802d6aa1bf4d560acbe0ba5aa0a9c5531a810b39a266aa330991a8be9b22df`.
- Run and runtime roots contain `SAFE_TO_DELETE`; grid correction and forced
  final-after-max were unset. Snapshot `strict-k1-v6-20260724` remains 25/34
  strict, 31/34 exact topology, and 34/34 evaluated.

# 2026-07-25: full K=4 preprocessing trajectory gate launched

- Same-A100 science `11600592` runs complete 15-iteration `host_numpy` and
  `relion_cuda` K=4 trajectories sequentially from clean detached commit
  `4181d340997e548af36c6458cce825e133dba95a`. CPU audit `11600593` has
  dependency `afterok:11600592`.
- Both arms use the immutable accepted RELION 15-iteration trajectory and its
  exact dispatch schedule. Grid correction and forced final all-data after
  non-convergence are unset.
- The fixed comparison reports direct FSC-AUC gates passed out of 60
  `(iteration, class)` checks at threshold `0.995`, iterations with all four
  classes passing out of 15, minimum class agreement, minimum GT FSC-AUC
  delta, exact topology, and same-GPU wall time. Correlation is not used.
- Checked snapshot `k4-host-ac5177d2-20260719` fixes the production-host
  baseline at 40/60 direct checks and 9/15 all-class iterations, with exact
  topology. `docs/math/em_k4_backend_trajectory_baseline_v1.json` stores the
  per-iteration count vector and accepted evidence hashes.
- `scripts/compare_k4_backend_trajectories.py` makes this count reproducible
  from the two standard FSC/topology audits and rejects cross-GPU inputs. It
  also directly compares saved host/`relion_cuda` assignments after applying
  each audit's RECOVAR-to-RELION class permutation at every iteration.
- Run root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_full15_host_relioncuda_samegpu_4181d340_20260725T051500ET`.
  Runtime root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k4_full15_host_relioncuda_samegpu_4181d340_20260725T051500ET`.
  Both contain `SAFE_TO_DELETE`.
- This pending K=4 metric does not alter `strict-k1-v6-20260724`: 25/34
  strict, 31/34 exact topology, and 34/34 evaluated.

# 2026-07-26: full K=4 backend trajectory accepts `relion_cuda`

- Same-physical-A100 science `11600592` completed both 15-iteration,
  100,000-particle, grid-256 K=4 arms from clean detached commit
  `4181d340997e548af36c6458cce825e133dba95a`.
- Exact control topology passes for `host_numpy` and `relion_cuda`.  Neither
  arm forced final all-data after non-convergence; grid correction was unset.
- At the fixed `0.995` direct per-class FSC-AUC gate, host passes 40/60 and
  `relion_cuda` passes 41/60.  Both retain 9/15 all-four-class iterations.
  Their per-iteration pass vectors are
  `[4,4,4,4,4,4,4,4,4,3,0,1,0,0,0]` and
  `[4,4,4,4,4,4,4,4,4,3,0,2,0,0,0]`.
- Candidate minimum cross-engine FSC-AUC, GT delta, and RELION class
  agreement are `0.990091127730`, `-0.000352907281`, and `0.99245`, versus
  host `0.989158631903`, `-0.000409355343`, and `0.99175`.
  Direct host-versus-candidate class agreement after independently audited
  RELION permutations is at least `0.99413`.
- Wall time is 30,089/26,921 seconds.  `relion_cuda` is 3,168 seconds or
  10.5288% faster on the same physical A100.
- Comparison JSON SHA-256 is
  `bca250d659c2ccbf5dc752cb876ecf35efe34447d03bd12850f92be86fc1cedd`.
  Checked snapshot `docs/math/em_k4_backend_trajectory_snapshot_v2.json`
  preserves the immutable old baseline and records the new 41/60 score.
- Audit `11600593` completed all scientific reports but failed during sealing
  because `AUDIT_SHA256SUMS.partial` included itself before being renamed.
  The repair used a temporary file outside `analysis`, excluded both manifest
  names, pinned the decisive artifact hashes, and verified all 24 entries.
  Repaired-manifest and repair-provenance SHA-256 values are
  `b5c45ccad205f271a91c0d1fe2a7f068e5674f2833bf26686a55f3dea815099b`
  and
  `859e36235bff8c8df3b5688a47165242c5f96dfbc7c560e8913edebff0e5a9f3`.
- This K=4 result supports the accepted backend snapshot but not a silent
  global default change: the shared default also controls K=1 and non-CUDA
  audit paths that this paired experiment did not cover.

# Current K=4 status (2026-07-25): fine operands close to shifted-image preprocessing

This status summary precedes the detailed capture and factor audit log below.

- RELION diagnostic commit
  `96387461fdaa18e4d23d4dbc57477039e3145b77` captures the complete
  per-pixel fine-score operand and 256-lane reduction for stack 42988,
  particle 36655, class 2, rotation-local 124, translations 56--59.
  Build `11594507` and paired same-A100 science `11594695` completed `0:0`.
  Capture/control class-map FSC-AUC is
  `0.999999992455`--`0.999999995250`; dispatch and particle fields are exact.
- Artifact
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_fineoperand_capture_9638746_20260725T055500Z/capture/factors/part36655_stack42988_class2.fine-operand-v1.bin`
  has SHA-256
  `a81cf6c18e9ce47864c119ae3d827e3aeb64121bf8d071e01176e4bc350e1102`.
  Production CUDA evaluates
  `fmaf(diff_real, diff_real, roundf(diff_imag * diff_imag))` before the
  separate `0.5*corr` multiply. Passive replay is bitwise exact for
  translations 56, 58, and 59; translation 57 is one float32 ULP above
  production (`2255.6376953125` versus `2255.637451171875`).
- Comparator commits `d300a49d`, `ce0c9554`, `fe5ad4d6`, `c9e5430b`, and
  `2544c33a` make the audit production-faithful: apply the score-path DC mask,
  align the global Fourier sign before single-operand substitutions, remove
  particle-common score offsets, replay the captured `dataset_native`
  background-fill backend, and compare directly with saved RECOVAR candidate
  scores. Superseded posthoc jobs `11595717` and `11595857` remain preserved
  as audit history; they exposed the DC/sign/backend reconstruction mistakes
  rather than production defects.
- Final local A100 replay uses
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_fineoperand_analysis_2544c33a_20260725T031500ET`
  and runtime
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k4_it10_fineoperand_analysis_2544c33a_20260725T031500ET`,
  both marked `SAFE_TO_DELETE`. It pins physical GPU
  `GPU-c6d48651-75fd-c644-a83f-3879c0a58186`, integration commit
  `2544c33a880cf8f7926247fc7f2b0ac81d399048`, and comparator SHA-256
  `3964bba854f418e9457b7f35ed30c6c4aab4991f51f7029bf152edcc8753968e`.
- RELION reference versus sign-aligned RECOVAR projection is bitwise exact.
  Both DC weights are zero. Remaining `corr` relative L2 is
  `3.4613115e-7`, with maximum scaled absolute difference
  `8.3673513e-11`. Sign-aligned shifted-image relative L2 is
  `6.3840108e-5` and is the strongest centered component: 72.7% residual
  energy removed over all four candidates and 75% on the three
  production-exact candidates.
- The complete backend-faithful replay reproduces RECOVAR's saved centered
  data-score residual for the three production-exact candidates with maximum
  absolute error `2.7105054e-20`. The all-four residual is isolated to the
  known translation-57 one-ULP passive replay mismatch. Final comparison
  SHA-256 is
  `5c7d3c625a659c3a23983038a4be52f5b925873805f2b34626f530b959adaa74`;
  validator SHA-256 is
  `676d5b2d98ed1e3990f88ac327b42c6bb69853c532502cd526e8d77561b357d6`.
- The next bounded discriminator is a saved-panel comparison of the native
  background-fill/host-FFT score image against RECOVAR's existing
  RELION-CUDA preprocessing path. No projection, DC, prior, exponent,
  posterior-normalization, significance, factor-placement, or reduction patch
  is supported.
- The frozen metric remains `strict-k1-v6-20260724`: 25/34 strict, 31/34 exact
  topology, and 34/34 evaluated.

# 2026-07-25: fine-score capture moves K=4 to projection/residual operands

- RELION commit `05398d236147eb71ce7fbbb60c635f2e8c012746`
  passively captures selected-stack pre-exponent data/prior/combined scores,
  shifted arguments, production expf outputs, and exact sparse identities.
  Build `11591782` completed `0:0`.
- Paired control/capture `11591945` ran sequentially on physical A100
  `GPU-ed3fe7be-abe7-7c79-06da-bc76e74d6025` and sealed 17 factor plus 17
  fine-score sidecars. Its wrapper failed only the superseded assumption that
  all active hypotheses have positive post-exponent weight. RELION clamps
  shifted float32 score `< -88` to zero; the corrected validator accepts all
  46,208 candidates and exactly 43,842 such underflows.
- Post-hoc job `11593544` completed `0:0`. Control/capture dispatch and final
  particle fields are exact; all class maps pass inertness at FSC-AUC
  `0.999999992596`--`0.999999995235`. Score/shift algebra has zero maximum
  error and non-underflow expf has maximum relative error
  `2.384185791015625e-07`.
- On 17 particles, 25 exact contributor rotations, and 108 active
  hypotheses, fine post-exponent weights equal downstream factor weights
  bitwise. The combined centered score residual has relative L2
  `1.2579036e-5`; its data component is `1.7222145e-5`, while orientation
  and translation priors are only `7.9367489e-7` and `5.5783039e-8`.
- A data-component substitution removes `0.999735251703` of combined-score
  residual energy. Prior substitutions do not. Score-shape job `11593681`
  completed `0:0`; a best per-particle scalar offset removes only
  `0.548987582535` of data-residual energy and leaves maximum absolute
  candidate-varying residual `0.000335693359`.
- The next bounded K=4 discriminator is a matched-candidate capture of the
  fine projected reference/per-pixel diff2 operands and reduction. Do not
  patch priors, expf, posterior normalization, support, factor placement, or
  a per-particle offset.
- Evidence root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_finescore_capture_05398d2_20260725T043428Z`.
  Validation, inertness, decomposition, and score-shape report SHA-256 values
  are `a4778489664f5d67aff151f3a6f72b3c38764d91febbedea975d67321de91a06`,
  `9f0ab46b7ffbd63061c7c41f2fde3fe3daceb15cff80ec1d78201ec7727954cf`,
  `591e5ddfa4ed0c725cc18fa7d7ecc17ea9eef79de893a4866b166b2d8304f834`,
  and `4502b45fff0b04232d37e67df0bfbe2b7646f09c027674cff8abf2df446bd9bd`.

# 2026-07-25: exact-128-add case-4 arm fails the frozen gate

- Science `11579503` (`COMPLETED 0:0`, H100 `della-h20g5`), summary
  `11579504` (`COMPLETED 0:0`), and fail-closed strict audit `11579539`
  (`FAILED 1:0`) complete the clean `161cb18f` intervention arm.
- Numbered iteration 17 remains close at merged cross-engine FSC-AUC
  `0.999664883`, but the final merged map is `0.992294244`, below the frozen
  `0.995` requirement. The topology audit also finds iteration-15
  `current_size` 154 versus 156.
- RECOVAR final GT FSC-AUC is `0.352136260` versus RELION `0.348384999`,
  delta `+0.003751261`. Wall times are `7,969.70` and `16,053` seconds,
  making RECOVAR `2.0143x` faster. The arm is rejected specifically on parity
  shape/topology.
- Exact 128-add/tie handling worsens final cross-engine FSC-AUC by
  `-0.000671668` relative to the prior bounded-tree arm, so it is not a
  production fix. RECOVAR converged at iteration 17; final all-data is valid,
  and grid correction/forced after-max remain unset.
- FSC/topology report SHA-256 values are
  `56994ba7e843b0245ca31671d64a60f6fc4ab747d150d6542bfe809ec79f733f`
  and `df1aad317d46e15d79a8ece0413cdfb2e533a69e4426e0f14ec5360705667ef1`.
- Frozen score remains 25/34 strict, 31/34 exact topology, and 34/34
  evaluated.

# 2026-07-26: K=4 passive-capture variability is calibrated, not bitwise

- RELION source `29a64a3f578ce999a3bc1f1ae56588be03870b9a` moves only
  fine-score capture serialization after the existing all-class production
  barrier. Build `11642549` completed `0:0`; binary SHA-256 is
  `4cb3aad7b7314ad38e409028c1ceef5e04fa15e12dd16bc9e14af239736154ff`.
- Same-A100 control/capture `11642770` completed both science arms and all 24
  selected artifacts, then correctly failed its older bitwise all-field
  wrapper. Pose, translation, class, dispatch, and perturbation are exact;
  Pmax mismatch count/max are `12434 / 1.83e-4`, significant-count mismatch
  is 3, and all class-map FSC-AUC values exceed `0.9999999921`.
- Same-A100 uninstrumented control/control `11642928` has capture variables
  unset in both arms but independently produces Pmax mismatch count/max
  `12485 / 1.74e-4` and five significant-count mismatches. Pose,
  translation, class, dispatch, and perturbation remain exact, and all map
  FSC-AUC values exceed `0.9999999926`. Exact Pmax/significant equality is
  therefore a native-repeat property failure, not evidence that the passive
  capture perturbs scientific state.
- The two complete passive panels have exact geometry, candidate topology,
  and winners for 12/12 targets. Their centered combined residual energy,
  L2, and maximum absolute value over 24,800 active candidates are
  `1.224850926333829e-4`, `0.011067298343922193`, and
  `4.961985462159646e-4`.
- RECOVAR commits `8d2afd15` and `8f92455c` make the repeatability floor
  reusable and bind all calibration reports fail closed. No tolerance or
  fixed score changed. The pending host/RELION-CUDA panel must complete
  exact 48+48 artifacts and manifest verification before the three-way
  classification is admissible.
- Initial preprocessing pair `11641724` completed the 48-file host arm in
  6,082 seconds. Its first two JAX buckets were only 9--10% faster, below the
  23% speedup required to fit the original three-hour allocation. An
  attempted scheduling-only extension was denied; the job was deliberately
  canceled after `01:47:52`, with partial evidence preserved but
  non-admissible. Fresh full-pair retry `11645269` started from the same
  clean `300c6e90` payload on `della-l07g2` with a six-hour limit and new
  `SAFE_TO_DELETE` run/runtime roots.
- The retry host arm completed `48/48` files in 6,193 seconds.  Fail-closed
  analyzer commit `d9369301` compares it with the independent completed host
  arm: every score/topology field and all 12 predicted classes are exact;
  `44/48` NPZs are byte/array exact.  The remaining posterior-only residual
  has maximum absolute delta `7.771561172376096e-16`, L2
  `7.900650490391253e-16`, and energy `6.242027817131954e-31`.
  Classification is `exact_score_topology_posterior_roundoff_only`, remains
  non-admissible for scorecard changes, and is sealed at report SHA-256
  `cac77dc1fd6847193f9bae8c5d4a1b2d50b632c795328ef8e151d403d276dbdc`.
- The fresh in-job JAX arm and dependency-gated calibrated three-way job
  `11646634` remain the only admissible host/JAX result.
- Pair job `11645269` completed `0:0` in `03:22:48`, with all `48+48`
  outputs and a passing full manifest.  Dependency analyzer `11646634`
  completed `0:0` in six seconds.  Both preprocessing paths match all 12
  RELION CUDA winners.  RELION-CUDA removes `10.0487%` of raw-data residual
  energy and `9.9113%` of combined residual energy, but the removed energies
  are only `0.6626x` and `0.5676x` the fixed repeatability floor.
- The formal classification is
  `relion_cuda_preprocessing_reduction_is_within_capture_repeatability_floor`;
  report SHA-256 is
  `96767afe79dbd33300ca683cf76a5b6d3b948b6a84756c14ce4517bf43f4c24b`.
  It does not justify a default change or scorecard promotion.
- Post-job manifest replay found that the in-job manifest retained an
  ephemeral Slurm spool-script pathname.  The original is preserved; a
  persistent manifest binds the hash-identical saved launcher, analyzer, and
  report and passes independently.  Its SHA-256 is
  `df7b7c28879ae9b834a624a8f57d77e9a913ced6b211b683a5695459c0f80550`.
  This is a provenance-only repair.
- Integrated replay found approximately `1e-18` process/thread variation in
  BLAS-backed residual-energy reductions.  Commit `547b2e3d` uses
  order-stable `math.fsum` for this diagnostic only and adds a regression
  forbidding `numpy.vdot`.  Real one-thread and four-thread replays are
  byte-identical at SHA-256
  `53b746a1cae413398bc9f3ce5c9cec1d9985a715652427dc8a1883b280f1083e`,
  with unchanged within-floor classification and 12/12 winners.
- Identity-bound cohort analyzer commit `839b40b6` classifies the predeclared
  4/4/4 panel as `heterogeneous_cohort_effect_without_robust_reduction`.
  Corrected targets improve only to `0.9339x/0.8298x` their data/combined
  floors; persistent targets worsen by `-0.0145x/-0.0612x`, and introduced
  targets by `-0.2859x/-0.3568x`.  No cohort clears its own floor.  Report
  SHA-256 is
  `468a313879f7436b1c960c2f37edd7d48f77328ac124744a5b8c53c501f74bdd`.

## 2026-07-25 seed-exact K=4 accepted-hypothesis factor closure

- RELION factor-capture build `11587833` completed from RELION commit
  `a9ae8d2dd24704d7de52940fbc832fab1029a268`.  The diagnostic allocates
  expensive per-pixel term rows only for live accepted hypotheses while
  preserving the complete orientation/hypothesis metadata table.
- Parent capture job `11587967` completed both RELION control and capture
  science arms (`00:28:22` and `00:25:35`) and finalized all 17 mixed-rank
  factor files.  Its wrapper exit `1:0` is non-scientific: it required
  byte-identical repeat-scale data-STAR columns.  Formal class-map inertness
  passes at FSC-AUC `0.999999992492`--`0.999999995085`.
- Postflight validation binds 17 files with accepted-hypothesis counts
  `[2,2,5,2,4,3,4,2,3,3,2,2,4,4,3,4,4]` and heterogeneous rank/orientation
  ownership.  Validation and inertness SHA-256 values are
  `0833e750bf9109d3cbe7881477143e6a622bd8714c9b20b448dd75612603fd7b`
  and
  `365e85fa249defb07b05f5676462cd4d83811aae59c6b95a585dbfa49ee29fe6`.
- Final A100 factor comparison `11590986` completed `0:0` in 14 seconds from
  RECOVAR commit `0f6356803166b9f9c0e7e17bf1e7af4d39fd3768`.
  The panel has 17 particles, 25 exact contributor rotations, and 53 accepted
  hypotheses. Geometry and accepted translation support are exact.
- The comparator reconstructs the stored dataset-native production path,
  including zero-filled integer pre-shifts, host packed FFT, image correction
  in the numerator, and scale correction squared in the weight.  Its replay
  matches captured RECOVAR `active_summed` at relative L2
  `7.34e-8`--`8.29e-8`, so the decomposition is internally closed.
- Aggregate RELION/RECOVAR relative L2 is: CTF `2.8273e-7`, inverse noise
  `3.1549e-8`, translation increments `3.1649e-8`, posterior `8.2828e-5`,
  complex term `8.4172e-5`, real weight term `8.4192e-5`, and contributor
  source sum `4.2365e-5`.  Processed FFT and weighted CTF individually differ
  by `0.0066233` and `0.0064766` because the engines place the per-particle
  normalization/correction factor on opposite operands; the product mostly
  cancels.
- The next discriminator is a fixed-posterior counterfactual at these same 53
  accepted hypotheses.  It should reuse RELION posteriors while preserving
  each engine's image/CTF/noise/phase operands to test whether posterior
  arithmetic explains the remaining `8.42e-5` term/weight residual.
- Comparison JSON:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_seedexact_factor_capture_a9ae8d2_20260724T224000ET/analysis/K4_RELION_RECOVAR_FACTOR_COMPARISON.json`
  (SHA-256
  `e70f404a25c4a43fc768d12a6ee507a61ab9d39e348f527d6d1caffbbe1d590a`).
  The run and runtime roots both contain `SAFE_TO_DELETE`.
- This is a K=4 localization diagnostic only.  Frozen snapshot
  `strict-k1-v6-20260724` remains 25/34 strict, 31/34 exact topology, and
  34/34 evaluated.

The predeclared posterior-only counterfactual is complete. A100 job
`11591141` ran the immutable 17-particle/25-contributor/53-hypothesis panel
from commit `f52a8bfc` and completed `0:0` in 18 seconds. Replacing only
RECOVAR's posterior with the captured RELION posterior reduces aggregate
relative L2 as follows:

| Operand | Production posterior | RELION-posterior counterfactual | Residual-energy removal |
|---|---:|---:|---:|
| complex term | `8.4171546e-5` | `3.6279787e-7` | `0.99998142` |
| real weight term | `8.4192179e-5` | `3.4731528e-7` | `0.99998298` |
| contributor source sum | `4.2365267e-5` | `3.6284445e-7` | `0.99992665` |

This causally closes factor placement and identifies posterior construction
as the remaining source. The next audit must reconstruct posterior weights
from captured RELION `diff2`/normalization fields and RECOVAR candidate
score/log-normalizer operands before proposing a production change.
Counterfactual JSON SHA-256 is
`e526fdb5b49f4675393b65512864f772be88580a37f1c1a25a8e08b0621d68d4`;
completion marker SHA-256 is
`138b8490bff01a4233379b2dbe52418fc47c8ad26c3890f18b8035a7f5bdff5d`.
The frozen score remains 25/34 strict, 31/34 exact topology, and 34/34
evaluated.

Posterior score/normalizer decomposition `11591351` completed `0:0` in 30
seconds from commit `1df53190`.  On the fixed accepted support:

- raw exp(50)-frame posterior weight relative L2 is `1.0194764e-4`;
- all-support exp(50)-frame normalizer relative L2 is `7.3824062e-5`;
- normalized posterior relative L2 is `8.2827810e-5`;
- RELION raw-log-weight versus RECOVAR shifted-score absolute residual has
  median `2.4406874e-4`, p95 `4.8831519e-4`, and maximum `4.8834586e-4`.

Thus both the accepted weight numerator and global normalizer differ; a
divide-only posterior patch is rejected. The next exact-boundary capture
should record RELION's pre-exponent fine `diff2` plus orientation/translation
prior terms for these identities, then compare them with RECOVAR's captured
preprior and combined scores before changing exponentiation or pruning.
Decomposition JSON SHA-256 is
`33a6a98d17f3c84ff55c406d4ab49c8d5c337189aa24d668ed14121fccbfea61`;
completion marker SHA-256 is
`f706b25d226e69ccaae2c8f1831f49329eac25742473659452af706d0ba37912`.

# 2026-07-25: seed-exact K4 replay closes topology and scatter support

- Seed-exact restart replay `11584817` uses the live iteration-10 sampling
  perturbation `-0.12305957078933716`, not the rounded sampling-STAR value.
  Its expected post-capture reconstruction-layout failure is sealed as
  `capture_complete_post_capture_reconstruction_failed`; all boundary
  artifacts required by the independent audits are complete.
- Independent audit `11585023` completed `0:0` in 13 seconds.  All 96/96
  coarse-parent sets, fine-candidate sets, and positive-contributor rotation
  sets match exactly.  The 120 class-2 contributors have exact rotation
  identities, both independent geometry gates have maximum absolute error
  zero, and all 96 reached-pixel sets match with zero one-sided pixels.
- Value audit `11586748` completed `0:0` in 12 seconds.  It restricts RELION
  to the exact contributor rotations before fitting one positive real scalar
  independently to data and weight.  All 120 fits are geometry-qualified,
  but only 109/120 (`90.8333%`) pass the predeclared `1e-5` scalar gate,
  below the predeclared 95% causal threshold.  Complex-data fit residual
  median/maximum are `3.55733e-7`/`4.22837e-5`; weight residual
  median/maximum are `3.21426e-7`/`6.96009e-7`.  The sealed classification is
  `pixel_varying_source_difference_not_explained_by_per_rotation_scalar`.
- This rules out exact-input topology, contributor masking, matrix
  construction, outer-shell inclusion, scatter support, and a single
  posterior normalization scalar.  The remaining branch is the
  pixel-varying translated-image/CTF/noise factorization before scatter.
- Durable run/runtime roots are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_seedexact_restart_boundary_replay_f58a29ae_20260725T011349Z`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k4_it10_seedexact_restart_boundary_replay_f58a29ae_20260725T011349Z`;
  both contain `SAFE_TO_DELETE`.  Contributor-support JSON, scalar JSON, and
  scalar NPZ SHA-256 values are
  `c80906e9afe1e269c30c5e100e358e9a79615fef6df380809e5786e5fbed5075`,
  `9009b415e84f1e7771c9fe7d124738d9e2d3e735c7f2877a0211952a9811214e`,
  and
  `83ab74e6c590087e9cf5e919fe80f0416d7350ac7addc6c33a7637f27a41b9b8`.
- A deterministic 17-particle factor panel covers all 11 non-scalar
  contributors plus six category controls.  Selection and launcher SHA-256
  values are
  `23254206a4c90ef76a3e73b2d9323f27d68be5c213104cd112486706e7c92158`
  and
  `39cfafafcb1f10387b8f4a5e7fe556a7684051df9235c373e24b481a697e7da7`.
  RELION control/capture job `11586985` is active; its run/runtime roots are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_seedexact_factor_capture_ebd0852_20260725T021500Z`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k4_it10_seedexact_factor_capture_ebd0852_20260725T021500Z`,
  both marked `SAFE_TO_DELETE`.
- Snapshot `strict-k1-v6-20260724` remains 25/34 strict, 31/34 exact
  topology, and 34/34 evaluated.

# 2026-07-24: matched K4 restart excludes candidate and contributor support

- Matched-restart replay `11582127` uses the RELION iteration-9 state and the
  exact iteration-10 perturbation `-0.12306` for both engines.  Grid correction
  is unset/off.  The A100 job captures all 96 selected particles in 95 shards
  and records 121,777,840 source values.  It exits `1:0` only after the E/M
  capture, when an optional map diagnostic tries to interpret current-size
  RELION x-half accumulators as a full 512-cube Fourier layout.  The capture
  validator passes independently; its report SHA-256 is
  `d6209de63c8933d735b80068b4c12fcd9b0cd04eb085a632736ffddac7323bf2`.
  The run is explicitly sealed as
  `capture_complete_post_capture_reconstruction_failed`, not as a complete
  map replay.
- The original dependent audit `11582290` was cancelled because the science
  launcher's post-capture failure made its `afterok` dependency impossible.
  Replacement CPU audit `11583809` completed `0:0` in 14 seconds with 743,696
  KiB maximum RSS.  Both engines' rotation matrices pass the generated-grid
  gate, and the sampling perturbation delta is zero.
- The checked v3 identity audit reports exact support for all 96 particles:
  7,090/7,090 coarse parents, 56,720/56,720 fine candidates, and 120/120
  positive contributor rotations.  Every per-particle set is exact.  The
  scientific classification is
  `candidate_and_rotation_contributor_support_exact`; K4's later trajectory
  gap therefore does not begin in coarse selection, fine candidate
  generation, or rotation-level significance masking.  The report SHA-256 is
  `aeb6f14c03da5c44fead5b3e63efd94b9423133cc1af6990910302f46e1fceb0`.
- The matched prescatter operand comparator validates all 120 contributor
  rotations but finds exact reached-pixel support for only 5/96 particles.
  There are 130 RELION-only and 128 RECOVAR-only pixels across 257,589 union
  entries.  An independent enumeration shows all 258 are on shell 37, exactly
  the current-size radius.  On intersecting support, median per-particle
  relative L2 is `7.2088207e-7` for data and `3.8432100e-7` for weight; p95 is
  `2.2622767e-4` and `2.2589895e-4`.  The first discrete mismatch is therefore
  the outer-shell backprojection inclusion boundary, before accumulation or
  reconstruction.  The report/array/comparator SHA-256 values are
  `234bc4871b17dd1cdf1c4eaa7754d56999c63e6b2944c624506e3b69633893b2`,
  `54670c115c15062756e96adec94a255fcb24c0f2c706879bb7758593cfc75c4c`,
  and
  `ed4eb83131d7be96006c48add1055b06f5623e740a2670b3510e8f2d4e82a3b2`.
- Durable run/runtime roots are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_matched_restart_boundary_replay_0f5f1404_20260725T012000Z`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k4_it10_matched_restart_boundary_replay_0f5f1404_20260725T012000Z`;
  both contain `SAFE_TO_DELETE`.
- Next discriminator: recompute the 120 contributor rows with RELION's exact
  captured float32 fine-rotation matrices.  If the 130/128 shell-37
  asymmetry collapses, the causal branch is fine-matrix construction/handoff;
  otherwise compare the two engines' outer-radius inclusion predicates
  directly.  No scorecard checkbox is promoted: the frozen score remains
  25/34 strict, 31/34 topology, and 34/34 evaluated.

# 2026-07-24: complete K=4 trajectory confirms a first failure at iteration 11

- Independent full-trajectory audit `11578043` evaluated all 15 numbered
  K=4 boundaries and the non-converged final-map policy.  The expected
  scientific-gate exit is `2:0`; runtime was `01:01:29` and maximum RSS was
  4,767,900 KiB.
- Iterations 1--10 pass the unchanged `0.995` cross-engine FSC-AUC gate.  The
  earliest failures are iteration 11 classes 2 and 3 at `0.994509131` and
  `0.994150545`.  Terminal classwise values are
  `0.994459232,0.993069734,0.992039376,0.994497731`.
- Identity map and particle-class assignments remain preserved.  Every GT
  FSC-AUC delta remains inside the unchanged `-0.002` gate, so the red result
  is cross-engine trajectory divergence rather than a GT-quality collapse.
- Audit JSON and shellwise NPZ are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_terminal_full_audit_20260724T214500Z/analysis/k4_terminal_fsc_trajectory.json`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_terminal_full_audit_20260724T214500Z/analysis/k4_terminal_fsc_shellwise.npz`,
  with SHA-256 values
  `02615753e8bb20df95673a6aa45fe374111b28aac819115d09d44d429bec2288`
  and
  `c676d662c3a1204d5ff2710d710dd1fd6bd3288169c3d3e49373d2beb008db4e`.
  Run and runtime roots contain `SAFE_TO_DELETE`.
- K=4 remains red.  Snapshot `strict-k1-v6-20260724` remains 25/34 strict,
  31/34 exact topology, and 34/34 evaluated.

# 2026-07-24: native RELION reconstruction localizes K=4 divergence upstream

- `scripts/audit_k4_native_bpref_reconstruct.py` converts the saved
  iteration-11 odd `155^3` public accumulator into RELION's explicit
  `155x155x78` x-half BPref layout.  An independent round trip through
  RECOVAR's production x-half expansion is bitwise exact.  The qualified
  frame conversions are data `*-256^2`, weight `*256^4`, and tau2 `/256^4`.
- RELION Class3D defaults to `skip_gridding=true`.  Native binding jobs
  `11580327` and `11580360` complete `0:0` in 15/40 seconds with maximum RSS
  2,769,368/2,834,344 KiB.  Native-RELION reconstruction versus the saved
  RECOVAR map has classwise FSC-AUC
  `0.999999953,0.999999944,0.999999945,0.999999935`.
- Substituting the native maps changes cross-engine FSC-AUC by only
  `+4.27e-8,+6.83e-8,+3.72e-8,+7.51e-8`; the iteration-11 class-2/class-3
  failures remain `0.994509199`/`0.994150582`.  The accepted classification
  is `remaining_k4_gap_precedes_reconstruction`.
- Non-default `skip_gridding=false` arm `11580236` completes `0:0` in 18
  seconds but worsens class-2 cross-engine FSC-AUC to `0.993845903`; it is
  rejected.  Preflight-only job `11580190` failed before reconstruction on
  an unresolved `.pixi` symlink assertion.  Superseded pending jobs
  `11579928`/`11580144` were canceled at zero runtime.
- Run/runtime roots are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it11_native_reconstruct_20260724T224111Z`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k4_it11_native_reconstruct_20260724T224111Z`;
  both contain `SAFE_TO_DELETE`.  Classwise audit JSON SHA-256 values are
  `2c8568f8f88b5c43ec36863e05fb2a034f1237254160972838f58f5a7d260254`,
  `2885cdd8b21c8003578a835ad9a34ad812170e5215e5e98da185d209879b7f86`,
  `96e61737cb6366026155e49e892f5715f4d415ffeb0a09d5f253e2bc4e1889f9`,
  and
  `599ae299cb0ed343355310ba594f4448f8a5b89d5a72d887c9c2453652e211b7`.
  This diagnostic rejects a reconstruction/postprocessing patch and does not
  alter the red K=4 gate or frozen K=1 score.

# 2026-07-24: RELION's 128 atomic additions close the case-4 coarse tie

- Synchronized same-H100 RELION component job `11577336` completed `0:0` in
  `01:04:58` with 6,416,520 KiB maximum RSS.  Dependent fail-closed audit
  `11577341` completed `0:0` in 13 seconds with 637,100 KiB maximum RSS.
  The captured numerator and norm are bitwise equal to RECOVAR's hybrid
  projected-reference/image replay for both cross-winner candidates:
  `(0.09698139131069183,0.12128135561943054)` for pose `955081` and
  `(0.09905915707349777,0.12653392553329468)` for pose `977030`.
- RELION source `cuda_kernel_diff2_CC_coarse` has every one of its 128
  threads atomically add the same reduced normalized-CC contribution divided
  by 128.  A direct ratio retains a five-ULP candidate split.  Replaying all
  128 sequential float32 additions collapses both scores to identical bits:
  `0.27847832441329956`, uint32 `1049531574`, exactly matching the captured
  RELION scores.
- RELION resolves that tie in direction-major flat order.  RECOVAR pose IDs
  `955081,977030` map to RELION keys `943626,928339`; the smaller key selects
  pose `977030`, which is the observed RELION winner.  Across the complete
  1,069,056-candidate grid, atomic replay has zero maximum absolute error and
  1,069,055 bitwise matches.  The only bit mismatch is `+0.0` versus `-0.0`.
  The accepted classification is
  `relion_atomic_accumulation_closes_cross_winner_tie`.
- Production scoring now reproduces the 128 sequential additions, and exact
  ties use RELION's direction-major coarse key.  The change is confined to
  the existing bounded first-iteration normalized-CC top-two rescore.
  Independent frozen-component regressions pin the two component pairs,
  uint32 score bits, tie keys, and winner.
- A production A100 replay using the persisted case-4 capture selects pose
  `977030` with tied score bits.  Run/runtime roots are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_atomic_fix_replay_20260724T225000Z`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_atomic_cc_fix_20260724T224500Z`;
  both contain `SAFE_TO_DELETE`.  Frozen replay JSON/script SHA-256 values are
  `f5a685c4ec943c81029372cb208fb733ff28fe8fdf0825b8c6378edce0e3f54c`
  and
  `a82f8d6ca89b3f7693576a823ba6e8ed45d3bf3140925f92dec4abcbce4ed522`.
- Component result/analyzer SHA-256 values are
  `65b9a8a74a581eb504a012251dc1a67f046e2abc6b7ef89820104dfeb54ea874`
  and
  `f6a4e991deff3b3cfcd72cba565fd5ee1fa3f35b43528840d4b866888e4a49a8`.
  RELION weight, norm, and captured-score SHA-256 values are
  `fcaad1ed992d3721f976ac79c3de21ccd9131e7b7a936463c11adb14425cc224`,
  `a418076d3303e80162b1a0df9b8031ff4107376fa5dd3901256d0a563e0478b7`,
  and
  `4340b5b7b015ab1f9a94b73b4c069b0d5d95273e0973f0e28b682be8397186e9`.
  RELION component and runtime roots are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_p5234_cc_components_sync_20260724T165500Z`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case04_p5234_cc_components_sync_20260724T165500Z`;
  both contain `SAFE_TO_DELETE`.
- Euler-operand diagnostic `11578677` was canceled before execution at zero
  runtime because this exact component/source closure made it unnecessary.
- Clean detached commit `161cb18f8989d8e83320d539d35a12f597d32ea6`
  now owns an autonomous frozen-fixture case-4 graph: setup `11579502`
  completed `0:0`; science `11579503` is running; summary `11579504` and
  independent strict FSC/topology audit `11579539` are dependency-gated.
  The fixture-manifest SHA-256 is
  `422a79a0a7703d92f9777266e8c34ccd3a7cf5963b354e57a7d9a18f227babee`;
  grid correction is off, forced final-after-max is unset, and the bounded
  margin remains `4e-6`.
- Run/runtime roots are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fixedsuite_case04_atomic_161cb18f_20260724T231500Z`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_fixedsuite_case04_atomic_161cb18f_20260724T231500Z`;
  both contain `SAFE_TO_DELETE`.  The independent strict-audit launcher
  SHA-256 is
  `fb6fe4695a0aff0dede1f5cede72956ed237f912c43b4bbd47dc8251bb50aaf1`.
  The frozen score remains 25/34 strict, 31/34 exact topology, and 34/34
  evaluated until this rerun passes unchanged gates.

# 2026-07-24: frozen case 7 tests firstiter generalization

- Case 7 is a strict-FSC failure with exact topology already passing.  Its
  known shell-20 scheduler split is a downstream amplifier, so it is an
  independent long-fixture test of the same direct-real-reference plus
  bounded `4e-6` top-two intervention that improved case 5's first boundary.
- Clean detached source is
  `c74beea47a1a91a723ab2c99f961b8a70483c34c`; fixture-manifest SHA-256 is
  `422a79a0a7703d92f9777266e8c34ccd3a7cf5963b354e57a7d9a18f227babee`.
  Grid correction and forced final all-data after maximum iteration are
  unset, and the run uses autonomous RELION plus RECOVAR trajectories.
- Slurm graph is setup `11567460` (completed `0:0`), science `11567461`,
  summary `11567462`, and independent strict FSC/topology audit `11567496`.
  The audit is `afterok:11567461` and independently checks clean source and
  RECOVAR import provenance before both fail-closed auditors.
- Strict-audit, science, setup, and summary launcher SHA-256 values are
  `896a42f5f887ae0d9f5e4a9079bd83c944a43e551f676c60d03a5d15726b02d6`,
  `2e220fa559acf9f0af52921d81fe44f95c63434683161236e524772100a5c2e7`,
  `5afbde5f1067035a57a9d6ae41f0439e17cc5a4fc7f2d55bcb915f4338dfaa48`,
  and `638a00c242dbd18e7ff4f8937c5901a3dabbf58a7711a9eae33a56f289909a94`.
- Run/runtime roots are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fixedsuite_case07_tree_c74beea4_20260724T110500Z`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_fixedsuite_case07_tree_c74beea4_20260724T110500Z`;
  both contain `SAFE_TO_DELETE`.  This pending diagnostic does not change
  the fixed score.

# 2026-07-24: cases 4/5 accumulate their terminal pose tails from iteration 1

- Exact-image-identity particle-state audits `11571320_0` and `11571320_1`
  completed `0:0` in 230 and 214 seconds.  They cover every numbered state
  plus final all-data for 100,000 particles in frozen cases 4 and 5.  Both
  pass the exact numbered-iteration mapping, schedule, convergence, and
  convergence-valid finalization gates; the distributions are diagnostic and
  do not replace FSC/FSC-AUC acceptance.
- After the direct-real projector plus bounded `4e-6` tree intervention,
  case 4 iteration 1 has only two material particle exceptions: original
  index `5234` differs by `7.440216` degrees and `2.375820` Angstrom, while
  index `72654` differs only by `1.062504` Angstrom.  Case 5 has three
  translation-only exceptions: indices `38594` and `65070` differ by
  `1.062500` Angstrom and index `93729` by `2.125000` Angstrom.  Pmax and
  significant-support counts are exactly equal for all 100,000 particles in
  both cases at this first boundary.
- The greater-than-0.1-degree tail then grows continuously rather than
  appearing at final writeback.  Case 4 has `1,10,42,149,296,450,628,1609`
  tail particles over iterations 1--8, reaches 6,017 at iteration 17, and
  6,398 at final.  Case 5 has `0,11,76,192,505,843,1231,1601` over
  iterations 1--8, reaches 7,269 at iteration 16, and 7,421 at final.
  Significant-support mismatches likewise grow from zero at iteration 1 to
  4,456/8,493 at iteration 8 and 18,460/33,904 at the last numbered state
  for cases 4/5.
- Large Pmax-delta cohorts and support mismatches are enriched for the next
  iteration's tail after the initial exact boundary, but neither captures the
  majority of future tail particles.  This is consistent with a small
  first-iteration winner cohort seeding a broad, low-confidence posterior
  butterfly rather than a significance-threshold defect.
- Last-numbered to final particle changes remain small.  Case 4 changes from
  93.983% to 93.602% within 0.5 degrees and from 96.789% to 96.427% within
  0.5 Angstrom.  Case 5 changes from 92.731% to 92.579% within 0.5 degrees
  and from 95.625% to 95.404% within 0.5 Angstrom.  This independently
  confirms the inherited full-grid final-only family and rejects a final
  pose-writeback, grid-correction, scheduler, or threshold patch.
- Case-4 JSON/array SHA-256 values are
  `d68cf1f4ce2fa60664205cfb907cfbc67c3f991fcd308c02514d4c083e25c43b`
  and
  `fc4a164013e8a65a37f043e1db88415bdfd4221c615bd416c14f94c1724a1adc`.
  Case-5 values are
  `9a1875bfc1d7c98703dbc862fba37164aee840ca6dfc18d99c7fb0f31552eb5e`
  and
  `961f99e11f25559b7bfce75533bca69909d9830927a67099f7392ac3e8d5b472`.
  Durable run/runtime roots are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case45_full_particle_audit_20260724T134800Z`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case45_full_particle_audit_20260724T134800Z`.
- H100 array `11571746` completed both tasks `0:0` in 19:13 and 20:47.
  Fail-closed CPU analysis `11571905` completed `0:0` in 73 seconds and
  reproduces all five material exceptions against the original paired RELION
  iteration 1.  Each target capture contains the complete 1,069,056-entry
  coarse grid; exact image identity, Pmax, and support still pass.
- Case-4 particles `5234` and `72654` have native float32 top-two margins
  `3.278255e-7` and `2.384186e-7`.  Case-5 particle `93729` has margin
  `4.619360e-7`.  All three fall inside the bounded `4e-6` rescore band, but
  rescore leaves the native winner unchanged.  In contrast, case-5 particles
  `38594` and `65070` have margins `9.890050e-4` and `1.838163e-3`, so their
  translation exceptions are not threshold-edge decisions.  A global
  threshold increase is therefore rejected.
- Accepted summary JSON SHA-256 is
  `7a3731aa12504c4947c778fcf74571c084a71fb87b10eaef045d20596c42ace3`.
  Case-4 target NPZ SHA-256 values are
  `a6f0407a53e4e473e76ca0964cd82fd38f8f6bdd6d69ee6906f12f94e9a5ac77`
  and
  `17433056e94e0208ebefb00e3fd5d07b8eca64c24590693afaef04e7b2e61a64`;
  case-5 values are
  `56ad98e5311b7858d86249132e8850022958ae04bd10141b8e3f0e0771730d8e`,
  `1c341e4e29ff2893260d7c8f15b771d8a15a42d45567eaa4a6bcdd0954dbba16`,
  and
  `95f879451d2e37e5bbd2637d7edc4e33e61522e39fd8ee83a8dad6a3781fe4d9`.
  The run/runtime roots are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case45_remaining_it1_capture_20260724T140000Z`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case45_remaining_it1_capture_20260724T140000Z`;
  both contain `SAFE_TO_DELETE`.  Launcher SHA-256 is
  `775b0e7b74b3d18288839d4008a7cde1cb1f492d6efc36609c16deebe42aceb4`.
- Exact identity-aligned mapping of paired RELION fine poses back to the
  captured coarse grids identifies three runner-up-parent misses: case-4
  particle `5234` maps from native flat index `955081` to rank-2 `977030`,
  particle `72654` from `118097` to rank-2 `118092`, and case-5 particle
  `93729` from `510106` to rank-2 `510101`.  Case-5 particles `38594` and
  `65070` already select RELION's nearest coarse parents exactly
  (`917524` and `148012`), so their one-fine-step translation exceptions are
  downstream inside pass 2 rather than coarse-tree threshold decisions.
  Mapping JSON SHA-256 is
  `6ad97a96805f77d67af78418e1460239ff06f4d46336aa9cd1032ce08d371cf5`;
  analyzer SHA-256 is
  `f8312e9ab72b4008f33bb9e50f55cdaaaa5e9a946b82e12430a7c70dca0f75f8`.
- Same-physical-H100 patched-RELION job `11572062` completed `0:0` in
  1:05:38 for particle `5234`, the only target with a material rotation
  difference.  It reserved all four GPUs on `della-h21g2` and selected capture
  UUID `GPU-24350de1-cbbd-8567-62d2-db825502511b`.  Candidate sets are exactly
  identical: 1,069,056 common candidates, no engine-only candidates, no
  duplicates, and Jaccard `1.0`.  RELION assigns the two cross-winner keys
  `(32933, 24)` and `(33690, 20)` the same float32 normalized-CC score,
  `0.27847832441329956`.  RECOVAR assigns them
  `0.27847859263420105` and `0.2784782648086548`, respectively.  The resulting
  `3.278255e-7` split is exactly the previously captured native winner margin.
  Full-grid centered-score absolute difference p95 is `3.688037e-7`, maximum
  is `1.147389e-6`, and there are no active rotation or translation priors.
  This rules out missing candidate support and localizes the case-4
  first-boundary exception to sub-micro normalized-CC arithmetic/tie
  preservation.  Analysis JSON SHA-256 is
  `1a834a5f9cfdc67899f79485d6c467860f8a170f109555567fdf6169e70d2d12`;
  exact RELION dump-manifest SHA-256 is
  `d558418952fe8f9a1a791ca8fbca54ca6d0bc7c61e1f4d4081273025efc5b80c`.
  Its run/runtime roots are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_p5234_exact_relion_grid_20260724T142000Z`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case04_p5234_exact_relion_grid_20260724T142000Z`;
  both contain `SAFE_TO_DELETE`.  Launcher SHA-256 is
  `3499d355e54270d7be3eecb12a22509280784a1eb25ac491649aecba3f6bf694`.
- Exact FFTW-order replay of the immutable pass-0 operands closes the
  non-reference side of that tie.  RECOVAR `ctf2_data * 256^4` matches RELION
  `corr_img` at relative L2 `2.4156677e-7` and relative maximum
  `6.4483828e-7`.  RELION's actual coarse-kernel factorization,
  `Fimg_corrected * corr_img * translation_phase`, matches RECOVAR
  `-shifted_data * 256^2` at relative L2
  `2.8024704e-7/2.8735136e-7/2.9122280e-7` min/median/max across all 29
  translations.  The RELION-winner translation 20 and RECOVAR-winner
  translation 24 are `2.8206154e-7` and `2.8808909e-7`.  This eliminates image
  preprocessing, CTF weighting, translation phases, and FFTW window order as
  material causes; the remaining branch is projected-reference generation or
  score operand/reduction arithmetic.  Report and analyzer SHA-256 values are
  `f7258bfe7ac859b4499d6166ab78b597ad7c5183b333fcbdce6555eb0272530a`
  and
  `68dbe30dcf577026ce381f214663054cf3671ec6068b34ce2223193978f55471`.
  Exact-UUID component job `11574764` is pending on `della-h21g2`; it enables
  only RELION's built-in coarse normalized-CC numerator/norm dump.  Run and
  runtime roots are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_p5234_cc_components_20260724T154500Z`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case04_p5234_cc_components_20260724T154500Z`;
  both contain `SAFE_TO_DELETE`.  Corrected launcher SHA-256 is
  `8af43e740c9faf9ccc5f23f107846a76e00a3808c8316581c4c99eb138b0d4f7`.
  Pending job `11574731` was cancelled before execution after preflight found
  that its derived launcher had not yet isolated the runtime cache.
- Production-CUDA projection replay with the persisted RELION `PPref` and
  its eight fine Euler matrices is bitwise exact for all 51,968 complex
  pixels (eight orientations, 32 hypotheses, 1,624 pixels each).  Relative
  L2 and maximum absolute error are both exactly zero.  This rules out the
  texture projector implementation when its reference/Euler inputs agree
  and narrows the case-4 coarse tie to Euler construction/handoff or fused
  normalized-CC arithmetic/reduction.  Result/analyzer SHA-256 values are
  `9cb5f3407b44e137e20d86bb727015d55c7c168935a1b916420f37626059c10e`
  and
  `d23ff58c162b0fceccdda22125b17e495756c341317472f6b47f36f58cf23f95`.
  The result is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_p5234_cc_components_20260724T154500Z/analysis/RECOVAR_RELION_FINE_PROJECTION.json`.
- Same-physical-H100 fine-pass capture `11572658` completed `0:0` in 13:53
  for case-5 particles `38594` and `65070` on `della-h19g1`, using the
  required capture UUID
  `GPU-0d7b80c7-fef8-e346-6332-de36ae1af518`.  Both RELION fine candidates
  are present in RECOVAR's support.  Particle `38594` selects RELION's
  candidate exactly, with native top-two margin `1.4901161e-8`.  Particle
  `65070` has an exact float32 tie between flat candidates `332` and `333`;
  RELION selects `333`, whereas RECOVAR's first-index `argmax` selects `332`.
  They share rotation index `2` and differ by one fine translation step,
  `1.0624999` Angstrom.  This closes missing support and a margin threshold
  for both targets.  Because this capture does not contain RELION's two raw
  fine scores or compact-candidate order, the remaining discriminator is
  fine-score arithmetic versus fine-candidate tie ordering.  A passive RELION
  capture is required before changing production tie behavior.  Fine-summary
  JSON SHA-256 is
  `019d3111c6eda111080bd2e87a81832971d4128535f2a3718bb7352fd452897f`;
  target-panel SHA-256 values are
  `c024a27a8b2f8071a1015e845ed28a938e6d7b3ece309a8789d07b702fddbeb6`
  and
  `f4e57638c96361f1040374827342a97866b802276810dca61b2ba21f16bee18d`.
  Its run/runtime roots are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case05_finepass_targets_20260724T143000Z`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case05_finepass_targets_20260724T143000Z`;
  both contain `SAFE_TO_DELETE`.  Launcher SHA-256 is
  `0f26924928b50971b81e4f19578503da14674911d1a5d350439c55fa9eba8a2c`;
  corrected analyzer SHA-256 is
  `1b9a202192b00399b733d5e150c5ad647f47b0d82fdd69ed2113ee107d3fb674`.
- Passive RELION discriminator `11602720` is submitted on the same required
  H100 UUID.  It dumps RELION's pass-2 raw costs, compact rotation/translation
  indices, and oversampled hidden IDs for particle `65070`, then compares
  them against the accepted RECOVAR fine panel.  Before comparison, it
  requires exact identity-aligned stock-versus-dump-enabled RELION
  iteration-1 poses, translations, class, Pmax, and significant-support
  counts for all 100,000 particles, then maps the eight-rotation fine panels
  by Euler matrices.  Superseded pending jobs `11602588` and `11602654` were
  cancelled at `00:00:00` before execution; neither consumed GPU time.  The
  accepted pending launcher's run/runtime roots are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case05_p65070_exact_relion_fine_20260725T083000ET`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case05_p65070_exact_relion_fine_20260725T083000ET`;
  both contain `SAFE_TO_DELETE`.  The fail-closed launcher SHA-256 is
  `94db8675962d37e1ab28cb2a20a95605bec7a31682fedaa4c118aa7d43cbc4b8`.
- Snapshot `strict-k1-v6-20260724` remains 25/34 strict, 31/34 exact
  topology, and 34/34 evaluated.

## 2026-07-26 13:11 ET — fixed-metric/PR checkpoint

- Draft PR `ma-gilles/recovar#158` top-level metrics were updated in place to
  K=1 `26/34` strict, `32/34` exact topology, `34/34` evaluated, plus the
  separate K=4 `41/60` direct checks and `9/15` all-class iterations.  The
  remote branch head remains `6ddd0940`; no push occurred.
- The GitHub connector fetched the full body, required exactly one match for
  each of three stale strings, replaced only those strings, and verified the
  returned body.  Body length changed from 62,757 to 63,027 characters;
  GitHub recorded update time `2026-07-26T17:10:05Z`.
- Clean local head `5573b82b` passed checkout provenance, scorecard Markdown
  freshness, all `15/15` scorecard unit tests, scoped Ruff, and
  `git diff --check`.
- Exact commands are recorded in `docs/math/em_parity_program.md`.  Runtime
  root
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/scorecard_check_5573b82b_20260726T131100ET`
  is marked `SAFE_TO_DELETE`.
- The first freshness attempt omitted the required argument to `--check` and
  exited `2` before checking; the corrected invocation passed.  This was a
  command-shape error only.
- Case-32 science `11633607` remained healthy and running on A100
  `della-l07g6`; summary `11633608`, strict FSC/topology audit `11633676`,
  and particle-state audit `11633818` remained dependency-gated.

## 2026-07-26 13:27 ET — K=4 particle-state backend audit

- Read-only audits of all 15 × 100,000-particle states complete for the
  accepted host NumPy and `relion_cuda` arms from same-A100 job `11600592`.
  Both JSON/NPZ manifests verify.
- `relion_cuda` improves versus RELION in 14/15 Pmax-p95 rows, 15/15
  within-0.5-degree rows, and 14/15 within-0.5-Angstrom rows.  Exact support
  improves in 13 rows; class agreement improves in 10.
- At first FSC failure iteration 10 / class 2, Pmax absolute p95 improves
  `0.046972850489 -> 0.045644822556`, exact support agreement
  `0.58654 -> 0.59872`, and the 0.5-degree/Angstrom fractions
  `0.98718/0.97792 -> 0.98751/0.97872`.  Class agreement stays `0.99520`.
- In the exact 23,607-particle RELION-class-2 cohort, exact support agreement
  improves `0.494683780235 -> 0.511924429195` and class mismatches fall
  `177 -> 172`.
- Classification is
  `particle_state_improves_but_fixed_fsc_gate_remains_red`; checked K=4
  metrics remain `41/60` direct and `9/15` all-class iterations.
- Sealed report SHA-256:
  `b73ee37312c6ba2da9fe5b8a0139362a99a7582c0388ea2da1486218516cb620`.
  A balanced 12-particle persistent/corrected/introduced target panel has
  SHA-256
  `5cd036e6a1b834cb59c310b073ef5404efd63701454b49a5a2b55d297c14e8dd`.
  Full absolute paths and the seven-entry verifying manifest are recorded in
  `docs/math/em_parity_program.md`.

## 2026-07-26 15:00 ET — case-32 final-boundary causal closure

- Authoritative current-head science `11633607` completed `0:0`; strict audit
  `11633676` passed exact 11-vs-11 topology and all numbered FSC rows but
  intentionally failed the final merged gate:
  `0.9741320103734208 < 0.995`.
- Final GT FSC-AUC remains favorable:
  RECOVAR `0.2725652720757713`, RELION `0.2684149206256512`, delta
  `+0.0041503514501201`.  The frozen score therefore remains `26/34`, not
  because map quality regressed but because autonomous cross-engine parity is
  still below its fixed gate.
- Particle audit `11633818` identifies the first material pose split at
  numbered iteration 1 even though all Pmax and support counts match exactly.
  Original particle `3047` differs in rotation and translation; particle
  `6122` differs by one 2.125-Angstrom translation step.  The rare tail grows
  through iterations 2--5 and broadens when local search begins at iteration
  6.
- Diagnostic commit `ce03b5ad` adds paired, finite, exact-complex Fourier
  initial-reference inputs to `scripts/run_multi_iter_parity.py`.  It is
  opt-in only; default behavior is unchanged.  Targeted tests pass `33/33`
  and scoped Ruff passes.
- Exact-Fourier final-only jobs `11634911` and `11635037` both completed
  `0:0`.  Merged-reference scoring fails at `0.955455` and is rejected as a
  changed-semantics confounder.  Exact half-specific autonomous references
  plus exact RELION boundary state pass at `0.996194`.  Classification
  SHA-256 is
  `ba6f0c3755aba47fd462e500f3617d75d5b9f8b4bc14cf7fad1e13e70e08d79f`.
- Autonomous four-arm array `11634985` reran all 11 numbered iterations and
  substituted state only at the valid converged final boundary.  Final merged
  FSC-AUC is `0.995948034` for poses, `0.978981647` for references,
  `0.998080700` for poses+references, and `0.998072198` for all state.
  Poses are the minimal sufficient group; references have a secondary
  interaction; remaining scalar/correction/sampling state is immaterial at
  the current precision.
- All four source/output manifests verify.  The sealed factorial report is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case32_final_boundary_factorial_ce03b5ad_20260726T142000ET/FACTORIAL_CLASSIFICATION.md`,
  SHA-256
  `51825cb4e9ee4de8cf31315df835a00563c388348c43bf831d5cb0b9095e3a68`.
- Fixed metrics remain K=1 `26/34` strict, `32/34` exact topology,
  `34/34` evaluated and K=4 `41/60` direct, `9/15` all-class iterations.
  Diagnostic replay does not check a case.
- Next: capture/compare the first-iteration winner-score path for exact
  case-32 identities `3048@particles.128.mrcs` and
  `6123@particles.128.mrcs`.  Do not change tie ordering speculatively while
  passive exact-H100 case-5 job `11602720` remains pending.

## 2026-07-26 15:42 ET — case-32 iteration-1 winner path causally closed

- Passive target-3047 science from `11635535` is inert for all eight audited
  fields over all 10,000 particles.  Its complete coarse grid matches
  RECOVAR, but float32 reduction order reverses a roughly `1e-7` competition.
  Classification SHA-256:
  `1a2cc42e09f533bcca6c35a38a1bc38f4f476aa7400123d3e7eb653e07c5c395`.
- Passive target-6122 continuation `11636715` completed `0:0` on A100.
  RELION makes the two competing translations bitwise tied and selects the
  native earlier entry; RECOVAR's generic reduction favors the other by
  `2.980232238769531e-07`.  All 109 dump artifacts and all-particle
  inertness verify.  Classification SHA-256:
  `a31ffc747071d9e4a5390e73f76b43c8cba56bac0fab5ad1d0e93b47c21e7958`.
- Diagnostic commit `ee6bd7a5` makes pass-0/pass-1 selection explicit.
  Focused tests pass `28/28`; Ruff and `git diff --check` pass.
- Causal intervention `11637193` completed `0:0` in 1:53 on A100.  Exact
  RELION tree replay examines four half-2 near ties, makes one exact tie, and
  changes exactly two winners.  The resulting all-particle audit has exact
  Pmax/support and zero rotation or translation errors above `0.5`.
  Classification SHA-256:
  `50785e48e7e289ff5f4db93c146e3a30fe711de0d9a23b8c9ec6c62a9940bb4b`.
- Do not promote the fixed score yet.  Full autonomous FSC/FSC-AUC job
  `11635967` remains the unchanged map-quality gate; K=1 stays `26/34`
  strict, `32/34` topology, `34/34` evaluated, and K=4 stays `41/60`
  direct, `9/15` all-class.

## 2026-07-26 16:51 ET — case-32 promotion and K=1 score 27/34

- Canonical case-32 science `11635967` completed `0:0` in 56:45 on A100
  UUID `GPU-3b10bc7d-5485-6b3a-5607-da203ef39bd3`, matching the paired
  RELION UUID exactly.  Source was clean
  `916ab17a4c8040786bea6517b12c8746ae399d65`; grid correction and forced
  after-max finalization were unset.
- The bounded firstiter intervention examined four ambiguous half-2
  particles, made one exact tie, and changed exactly two winners.  The run
  then stayed effectively exact through all 11 numbered iterations.
- Worst numbered merged cross-engine FSC-AUC is `0.9999999719776043`;
  worst numbered GT delta is `-0.000002075771598608611`.  Final merged
  cross-engine FSC-AUC is `0.9982743466036096`, with RECOVAR/RELION GT
  values `0.2722232856252235/0.2683738524304857` and delta
  `+0.0038494331947377947`.
- Convergence occurred after numbered iteration 11, enabling a valid final
  all-data pass.  Final grid correction is `False`.
- Strict FSC/topology audit `11638090` completed `0:0`; all 11 numbered
  pairs and complete final products pass with no topology or numeric-artifact
  failures.  FSC/topology/shellwise SHA-256 values are
  `4a23da9f4ea335f27be1f24b518bf7a909480f867bc115542cbf67bf81964966`,
  `9ad2459ea736d44de87ef228ceca213eea6af1d8bb9d143e1db610cbfbc571c1`,
  and
  `2c3ee1b733e64b7f16a0edd69ef1411cfa34f00b4b5f1c29284d3ab33e198c6b`.
- Proposal `11639159` completed `0:0` and produced fail-closed v8 ledger
  SHA-256
  `13c7cf50de11d6819dda2cf0320915973183f09865e4b96cc8fcb04e6f005412`.
  Fixed K=1 metrics are now `27/34` strict, `32/34` exact topology, and
  `34/34` evaluated.  K=4 remains `41/60` direct and `9/15` all-class.
- Integration commits `bc4bde14`, `ce466fea`, and `59430e5a` promote the
  explicit comparator pass selector, documentation, and K=1-only
  `firstiter_cc` RELION defaults.  Explicit environment overrides win; K>1
  and non-firstiter behavior are unchanged.
- Integrated focused validation passes `56/56`; the EM fast guard, scoped
  Ruff, and `git diff --check` pass.  A first pytest invocation used the
  wrong class-qualified node ID, collected zero tests, and was corrected
  before any push.
- Full run root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fixedsuite_case32_projector_tree_916ab17a_20260726T150318ET`.
  Proposal root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_scorecard_v8_case32_916ab17a_20260726T162000ET`.
  Both are marked `SAFE_TO_DELETE`.

## 2026-07-26 20:55 ET — case-5 exact-H100 capture rejected and audited

- Job `11641917` completed both arms on H100 UUID
  `GPU-9f98ccbf-3c62-c54f-7409-7eb58845ad4a`, then intentionally exited
  `1:0` because strict capture inertness failed.
- All 109 dump files and seven canonical static inputs verify; no temporary
  dump files remain.  Both perturbations are exactly `-0.19536`.
- Seven of eight particle fields are exact over all 100,000 particles.
  Exactly one OriginY value differs, and it is the requested dump target
  `65071@particles.256.mrcs`: `0.116118 -> -0.946380` Angstrom, one
  `1.062498`-Angstrom fine step.
- Both half-map FSC-AUC values remain effectively exact at
  `0.9999999998941168` and `0.9999999999549709`, above the unchanged
  `0.999999` gate.
- The scratch v1 report's `68561@...` target label was a positional-row
  bookkeeping error; the full comparison and rejection were identity-aligned
  and correct.
- New identity-safe v3 analyzer requires the exact image identity, verifies
  its stack-prefix/original-index relation and expected particle count,
  reports bounded identity-specific mismatches, binds input hashes, and exits
  nonzero on rejection.  Focused validation passes `7/7`; Ruff, compile,
  checkout provenance, and
  `git diff --check` pass.
- Identity-safe report SHA-256:
  `4e36627a0c82b867c8428982d040e6de7c711afa1f51456c12b15738c173425c`.
  Failed-run audit:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case05_p65070_relion_inertness_h100_20260726T183000ET/provenance/FAILED_11641917.md`.
- Formal tie classification maps control to `[2, 101]` and capture to
  adjacent `[2, 100]`.  Captured RELION scores both exactly
  `0x1.95f52e0000000p-3`; RECOVAR scores both exactly
  `0x1.95f52c0000000p-3`.
- Classification is `observer_sensitive_exact_tie_winner_flip`: support and
  geometry are present, but the passive observer changes which exact-tie
  translation RELION serializes.  The rejected capture cannot qualify a
  scorecard row.
- Classifier tests pass `6/6`; direct CLI, Ruff, compile, checkout provenance,
  and `git diff --check` pass.  Classification SHA-256:
  `b7463ab0562f20863c650b70a18c495d235e44127256b0543693a3e7ecbccb4f`.
- Fixed metrics remain K=1 `27/34` strict, `32/34` topology, `34/34`
  evaluated and K=4 `41/60` direct, `9/15` all-class.

## 2026-07-27 K=4 posterior arithmetic remains cohort-dependent

Commit `e3008148` adds a fail-closed analyzer for the same immutable 12-target
iteration-10/class-2 panel.  It reconstructs RELION's normalized class-2
posterior directly from the captured fine-score `expf(50)` weights and the
float32 all-class weight normalizer retained in geometry-only BPref header
field 26.  For each RECOVAR preprocessing arm, it reconstructs the same frame
from the dumped all-class global Pmax, verifies the full probability mass and
candidate mapping, then evaluates two exact counterfactuals: replace only the
numerator with RELION's captured numerator, or replace only the normalizer
with RELION's captured normalizer.

The 24,800-candidate result is
`heterogeneous_posterior_arithmetic_response`.  The corrected cohort reduces
posterior residual energy from `1.4081601055419687e-9` to
`1.3970069295873553e-9`, but the removed energy is only
`0.34267176467456306x` its independent capture-repeatability floor.  Replacing
the numerator removes `73.59%`/`74.61%` of host/RELION-CUDA residual energy,
whereas replacing only the normalizer removes `9.30%`/`8.55%`.

The persistent cohort worsens slightly from `3.496417328756203e-14` to
`3.5273299236380804e-14`.  RELION-CUDA halves the cohort normalizer-relative
L2 from `2.2724876984742913e-6` to `1.1417529957183727e-6`, but the numerator
branch remains and accounts for about `56%` of removable energy.  The
introduced cohort also worsens, from `6.7663420684124795e-9` to
`6.786395560910647e-9`; its numerator and normalizer components counteract,
so replacing only the normalizer increases rather than decreases the
residual.

One-thread and four-thread replays are byte-identical at SHA-256
`3ecd85fc80b44e9f9c452eb13510f5dc474b2952a1140d029a2919896d388003`.
The report remains `scorecard_change_admissible=false`, computes no
correlation, and supports neither a preprocessing nor a posterior-normalizer
default change.  Continue at upstream numerator score arithmetic for the
persistent and introduced identities while retaining the normalizer as a
separate persistent-cohort branch.  The report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_preprocess_panel12_retry6h_300c6e90_20260726T200000ET/analysis/PANEL12_POSTERIOR_DECOMPOSITION_e3008148_T1.json`.

## 2026-07-27 K=4 numerator localizes before exponentiation

Commit `4f97ccbb` adds a second fail-closed analyzer for the same immutable
12-target, 24,800-candidate panel.  It binds the posterior-decomposition
report and every target artifact, maps the exact RELION candidate topology,
then compares the production numerator with two float32 score replays:
RECOVAR scores through RELION's shift/underflow/`expf` frame and RELION's own
captured shifted scores through the same replay.  It separately decomposes
the centered pre-exponent score into data score, orientation prior, and
translation prior.

All three cohorts and both preprocessing backends classify as
`numerator_residual_localized_upstream_of_exponentiation_to_data_score`.
Replacing the score with RELION's captured score removes
`0.9999994332053838`--`0.9999999999774101` of raw-numerator residual energy.
RECOVAR's posterior-derived numerator differs from its float32 score replay
by only `2.905210056316014e-8`--`3.065659800113664e-7` of production
residual energy.  Data-score substitution is strongest in all six
cohort/backend cells and removes `0.7908083535518164`--
`0.8879393494510351` of centered combined-score residual energy.
Orientation/translation-prior substitutions are negligible.

One-thread and four-thread reports are byte-identical at SHA-256
`1d5ee73794a6ff7498002634153d085d8faa11eb1f643c2694a2d0312c685a9e`.
The nine-entry manifest independently passes and has SHA-256
`6348ea9a14afce9f7d923155ccde70ea756b91f885ee46cdd6cf8d81f526ac67`.
The canonical report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_preprocess_panel12_retry6h_300c6e90_20260726T200000ET/analysis/PANEL12_NUMERATOR_BOUNDARY_4f97ccbb_T1.json`;
the provenance note is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_preprocess_panel12_retry6h_300c6e90_20260726T200000ET/provenance/NUMERATOR_BOUNDARY_4f97ccbb.md`.

This rejects exponentiation, posterior normalization, and prior handling as
the leading numerator cause.  Continue before prior addition in the native
data-score path, beginning with persistent stack identity 64843 and
introduced stack identity 42824.  The diagnostic is
`scorecard_change_admissible=false`; fixed K=1 remains `27/34` strict,
`32/34` topology, `34/34` evaluated, and fixed K=4 remains `41/60` direct
and `9/15` all-class.

## 2026-07-27 production CUDA translation-score boundary closes exactly

- Native operand replay identifies RELION CUDA `sincosf(x*tx + y*ty)` plus
  explicit real/imaginary products as a required direct-score primitive.
  The implementation routes only exact-Gaussian score input through the new
  `RelionTranslateScoreF32` FFI; reconstruction/M-step phases remain JAX.
- Same-A100 job `11655670` completed `0:0` in 1m29s. The FFI symbol built and
  38/38 focused CUDA/build/reduction/routing tests passed.
- Production sealed replay is exact for old stack 42988 (4/4 raw scores),
  current stack 64843 (10/10 replay-exact candidates, improved from 9/10),
  and current stack 42824 (2/2, unchanged).
- Gate:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_phase_ffi_gpu_13840a20_20260727T022224ET/analysis/PRODUCTION_FFI_SEALED_REPLAY_GATE.json`,
  SHA-256
  `fd0239f788916c9be3e1f025a9982c93de025315616d0d1a6a433f3ea93993fb`.
  Exact-candidate report SHA-256:
  `0c1dda8d8765f466ad87c30e8f3ecc384c2647227a60c6f820491835beff420f`.
- Repository-owned sealed eight-pixel A100 fixture job `11655745` passes
  bitwise. CPU-focused tests pass 19/19 and the EM fast guard passes 16/16.
- Grid correction and forced after-max finalization were unset. Map
  acceptance remains FSC/FSC-AUC only; no correlation is computed.
- This closes the exact score boundary but is not yet a scorecard promotion.
  Launch the fixed K=4 trajectory and affected K=1 fixed cases. Until those
  finish, K=1 remains `27/34` strict, `32/34` topology, `34/34` evaluated,
  and K=4 remains `41/60` direct, `9/15` all-class.

## 2026-07-27 fixed case 24 passes and advances K=1 to 28/34

- Canonical science `11655858` completed `0:0` in 17m07s on H100 UUID
  `GPU-9f98ccbf-3c62-c54f-7409-7eb58845ad4a`, exactly matching RELION.
  Source was clean detached
  `31c4a0ca203b70211f4d8586d044c94fca9fc037`.
- The fixed fixture manifest remained byte-identical at SHA-256
  `422a79a0a7703d92f9777266e8c34ccd3a7cf5963b354e57a7d9a18f227babee`;
  grid correction was unset/default-off and forced final all-data after
  nonconvergence was unset.
- Both engines have 12 numbered iterations with exact current-size trajectory
  `[56,56,56,56,56,56,56,58,58,60,60,60]`, converge at iteration 12,
  and enter final all-data only after convergence.
- Independent FSC/topology audit `11655936` completed `0:0`.  Final merged
  cross-engine FSC-AUC is `0.998090087202717` (frozen prior:
  `0.9948051037935267`), RECOVAR-minus-RELION merged GT FSC-AUC is
  `+0.00828011468926404`, and the worst numbered cross-engine FSC-AUC is
  `0.9999978940386508`.  The accepting audit computes no correlation.
- The audit manifest is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fixedsuite_case24_phaseffi_31c4a0ca_20260727T025000ET/cases/24_small_kent_outliers_3k_g128_pct20_noise3_bf80/trajectory_analysis/AUDIT_SHA256SUMS`,
  SHA-256
  `144bbdde8ce8a5aae01785d77b2f3fa837b9c604a4fe787bdc578ee76b66bc61`.
- The promoter initially failed closed because the sealed standard-output
  basename omitted literal `k1-24`.  A byte-identical hard-link alias was
  added and recorded at
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fixedsuite_case24_phaseffi_31c4a0ca_20260727T025000ET/provenance/AUDIT_LOG_ALIAS_REPAIR.md`
  (SHA-256
  `5207329cb1ab9c743b3571c08aae21b0cab3aff27a2e4bb97b255e8a865f1c33`);
  no audit bytes changed.
- The validated v9 ledger is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fixedsuite_case24_phaseffi_31c4a0ca_20260727T025000ET/provenance/em_k1_gui_grid0_local_highshell_full34_superseding_ledger_v9.json`,
  SHA-256
  `9cedb043dded9e5a2020cf53c413c4e1da366f4a6a2c54127347f8572a9ed7b3`.
- Fixed metrics are now K=1 `28/34` strict FSC/FSC-AUC, `32/34` exact
  topology, and `34/34` evaluated.  K=4 remains `41/60` direct and `9/15`
  all-class pending jobs `11655922` and `11655923`.

## 2026-07-27 exact-device K=1/K=4 live checkpoint and publication gate

- Exact-H100 K=1 science `11675461` passed the target UUID and checkout
  provenance gates.  Numbered iterations 1--7 preserve the prior control
  prefix; the new CUDA `sincosf` local translation-score path is active for
  both halves from iteration 8.  The resolution trajectory through completed
  iteration 11 is
  `[30.22,20.92,16.00,15.54,15.11,15.11,15.11,13.95,13.60,13.27,12.65]`
  A.  The run remains nonconverged and dependent FSC/FSC-AUC audit
  `11675472` remains pending.
- Exact-A100 K=4 science `11683600` passed the two-device allocation gate and
  selected required UUID
  `GPU-5e619c2e-82b4-ff79-cbcb-ab29514a9f30` as JAX's sole device.
  Iterations 1--6 complete at resolutions `60.44`, `49.45`, `30.22`,
  `27.20`, `25.90`, and `22.67` A.  Iteration-6 Pmax is `0.920704949` and
  occupancies are `0.2594/0.2109/0.2716/0.2582`; iteration 7 is active at
  current size 68 and hardened audit `11683764` remains pending.
- Non-scoring CPU job `11689329` applies the unchanged direct FSC-AUC gate to
  the 24 completed K=4 class/iteration checks with hash-bound inputs.  It is
  an early progress checkpoint only and cannot replace the 60-check audit.
- K=1 run root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_local_phaseffi_f5729c1b_20260727T143000ET`.
  K=4 run root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_full15_phaseffi_exactgpu_retry1_31c4a0ca_20260727T040500ET`.
- Shared cryo-ET outlier control `11686265` at pre-fixed-fixture source
  `f7148598` fails seven of 12 metrics, proving that the integration-branch
  failure predates `8fa5b02e..9a034097`.  The fixed integration fixture at
  `9a034097` improves this to eight of 12.
- Control `11686278` reuses the exact particle stack that passes 12/12 on
  GitHub `dev`.  Integration source passes 11/12 on those identical bytes;
  only round-2 particle precision is red (`0.1507` versus `0.1598`, `-5.7%`
  against the unchanged `-5%` gate).  The remaining sensitivity is
  downstream of fixture generation.  No tolerance or baseline was changed.
- Checkout-bound fixed-stream/CLI tests pass `27/27`.  Because the shared
  scientific gate is still red and rebasing this 1,467-commit integration
  history onto `github/dev` conflicts immediately, no push is permitted.
- This checkpoint is live evidence only.  Fixed metrics remain K=1 `28/34`
  strict FSC/FSC-AUC, `32/34` topology, `34/34` evaluated, and K=4 `41/60`
  direct, `9/15` all-class.

## 2026-07-27 exact-H100 local-score qualification is sealed but remains red

- K=1 science `11675461` completed `0:0` in `02:22:28` on exact H100 UUID
  `GPU-9f98ccbf-3c62-c54f-7409-7eb58845ad4a`, from clean detached source
  `f5729c1b59a7e658fd4bbbd00696191fa7fbb9e9`.  It converged autonomously
  after numbered iteration 17 and then ran one valid non-forced final
  all-data pass.  Grid correction remained unset/off.
- Independent CPU audit `11675472` completed `0:0` in `00:24:47`.  It emitted
  non-fatal warnings while rejecting incompatible shared CPU AOT cache
  entries, then completed fresh computation.  Both its inner and outer
  SHA-256 manifests replay exactly.
- All 17 numbered topology comparisons pass.  The worst numbered merged
  RECOVAR-to-RELION FSC-AUC is `0.9996855218427677` at iteration 17; the
  intermediate/topology audit status is `pass`.
- Final merged RECOVAR-to-RELION FSC-AUC is
  `0.9925486313265427`, an improvement of
  `+0.0009923228039618` over frozen case-4 score
  `0.9915563085225809`, but it remains below the immutable `0.995` gate.
  Final half-1 and half-2 cross-engine FSC-AUC values are
  `0.9944517322023848` and `0.9929489817788204`.
- RECOVAR final merged GT FSC-AUC is `0.35217657068255603`; RELION is
  `0.34833496994794966`; delta `+0.0038416007346063763` passes the
  `-0.002` gate.
- The fail-closed qualification is `status=complete`,
  `strict_gate_pass=false`, and `scorecard_change_admissible=false`.
  Trajectory JSON SHA-256 is
  `15e53e5c8b41698b6fc903b908de5717d242c17275a14dbb934f4bee2b52bd54`;
  intermediate JSON is
  `d61babc8d334670ee8e253ffbab64325ba01a3c1039c9161e43ca83282dc6c45`;
  qualification JSON is
  `232f4a19572a8e6c5eff6776c4ff92f1b1ab323c345be58fc456562b7b2151ea`;
  outer audit manifest is
  `c43f218e1cffaba77279f3739a32608dd4ea609c29a871357839ec399b3b77eb`.
- The complete evidence root is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_local_phaseffi_f5729c1b_20260727T143000ET`.
  No correlation was computed.  K=1 remains `28/34` strict,
  `32/34` topology, and `34/34` evaluated.

## 2026-07-27 reused-fixture memory control rejects `.90`; shell-0 control runs

- Integration reused-fixture job `11686796` retained the exact passing-dev
  particle stack but set `XLA_PYTHON_CLIENT_MEM_FRACTION=.90`.  It restored
  the authoritative-dev memory plan while worsening integration from
  `11/12` to `8/12` fixed metrics.
- Round-2 failures are image F1 `0.4717` versus `0.5049`, image precision
  `0.3109` versus `0.3407`, particle F1 `0.2496` versus `0.2755`, and
  particle precision `0.1426` versus `0.1598`.  All round-1 metrics and both
  round-2 recalls pass.  The round-2 combined inlier set shrinks from
  `4615/727` junk-or-anomaly detections on clean dev to `5184/823` on this
  integration control.  Restoring `.90` is rejected.
- The next isolated hypothesis changes only generic
  `regularization.get_fsc_gpu` shell 0 from the RELION convention `1.0` back
  to historical `fsc[1]`.  EM-specific RELION FSC helpers remain unchanged.
  Clean diagnostic commit is
  `47adbdda56e36a8f7e3364d089da845bf2635c10`; focused CPU FSC checks pass
  `2/2` with one GPU-only skip.
- Exact reused-fixture Slurm control `11688427` completed `0:0` in
  `00:18:39` with the same `.90` memory fraction and particle SHA-256
  `3aa1a5e41277b0d77ef84c910e0a9092fc7ae3bcf8bac8dd246fa24e182b2510`.
  All 12 fixed quality metrics pass.  The previously red round-2 image F1,
  image precision, particle F1, and particle precision improve to
  `0.510272443`, `0.345897669`, `0.280392157`, and `0.163055872`,
  all above their unchanged baselines.
- Integration commit
  `39a8bf1ec8a2f61e49ce2bddb8150a162513f7da` restores shell-1 extension
  only for generic multi-shell FSC.  Explicit RELION FSC helpers retain
  `FSC[0]=1`.  A one-shell shape guard was added after the first full
  regularization test run failed at the new boundary; the complete rerun
  passes `31/31` CPU tests with three GPU-only skips.
- Exact committed-source Slurm qualification `11688855` failed `1:0` after
  completing both rounds.  Four of 12 fixed metrics fail: round-2 image
  F1/precision are `0.464914930/0.304496901`, and particle F1/precision are
  `0.243197279/0.138431752`.  Run root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/outlier_fsc0_integration_39a8bf1e_20260727T224500ET`.
  Publication and push remain closed.
- The passing isolated and failing integration sources share runtime base
  `34872f38`; the one-shell guard is their only tracked runtime difference,
  and round-1 half-set artifacts are byte-identical.  The custom CUDA
  libraries differ in bytes.  Diagnostic `ee8892af6816ef06b55de5cf0a78c1b81a3b3400`
  adds only the guard to the passing checkout while holding its environment,
  CUDA library, fixture, and memory plan fixed.  Exact job `11689434` is
  running.

## 2026-07-27 sealed K=1 numbered-to-final particle-state audit

- The repository's standard exact-identity auditor compared RECOVAR local
  iteration 16 with physical RELION iteration 17 and also compared the valid
  converged final all-data states.  All 100,000 `rlnImageName` identities,
  iteration mapping, convergence at 17, and finalization topology pass.
- At iteration 17, rotation/translation within 0.1 degrees/Angstrom are
  `93.738%` / `94.561%`; final values are `93.207%` / `94.499%`.
  The 0.1-degree tail has 4,963 new, 4,432 resolved, and only 1,830
  persistent identities.  The translation tail has 2,708 new, 2,646
  resolved, and 2,793 persistent identities.
- Mean absolute Pmax error increases from `0.0193198831` to
  `0.0203614786`.  This supports a small final particle-decision degradation
  but not a final-writeback patch: the tail largely turns over, earlier
  full-trajectory evidence localizes its origin to iteration 1, and the new
  local-score source is worse than its same-source current control.
- JSON/NPZ SHA-256 values are
  `eb0095699a224d4f10f156476df35915dc2fae54052f1d9739849f68c1802bfa`
  and
  `92c4116eb0b287e225572fef51beaf15183e48bc0662f13fb93a52ffc21def83`.
  The replay manifest SHA-256 is
  `cf38b25e58586f7292d39d079e06d53b2f38cd5cc3d8ee908ed215683fbefc7d`.
  Artifacts are under
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_local_phaseffi_f5729c1b_20260727T143000ET/analysis/particle_state_transition`.
- The standard auditor performs no correlation and correctly converts
  RECOVAR translations into RELION's Angstrom units.  The earlier ad hoc
  mixed-unit translation comparison must not be cited.  This diagnostic
  leaves K=1 at `28/34` strict, `32/34` topology, and `34/34` evaluated.

## 2026-07-28 outlier repeatability and first-round-only junk candidate

- One-A100 paired job `11690043` held source `70faf942`, fixture, node,
  CUDA library, `.90` memory plan, and visible-device count fixed.  Repeat A
  passed the unchanged cryo-ET baseline; repeat B failed four round-2
  precision/F1 gates.
- Round-1 image-inlier Jaccard is `0.991140`; round-2 image and particle
  inlier Jaccards collapse to `0.333093/0.333333`.
- Custom CUDA and forced JAX backprojection probes are both non-bitwise at
  the same approximately `2.6e-7` maximum relative-L2 scale.  The quality
  variability is not custom-CUDA-specific.
- Sweep `11691075` shows all 143 true particle outliers were already removed
  before round 2 in both repeats.  Round-2 anomaly and all tested junk
  variants add zero true positives.  Current junk detection adds 730/772
  false positives.
- Isolated commit `77c2578d` runs junk detection in round 1 only by default,
  retains later anomaly/contrast detection, and provides
  `--junk-detection-every-round` for legacy behavior.
- First qualification `11691123` failed before science on the CUDA
  source-age gate.  Retry `11691182` used a byte-identical runtime copy of
  pinned library SHA-256
  `414cfd5412d9b2aa9039dd845e608b24aab0f7c68690baee47a90854d02da56b`
  and completed `0:0` in `00:29:43`.
- Both fixed-fixture cryo-ET repeats pass the unchanged regression in
  `885.516/889.116` seconds.  Round 1 requests junk; round 2 neither requests
  nor emits it.  Particle recall is `1.0` throughout, and round-2 incremental
  particle false positives are `101/117` with zero new true positives.
- Qualification JSON SHA-256 is
  `051f137ae2885f5d929832bc0fe726b58fa7ad0a310e875af1e0e368484812e6`.
  Fresh-fixture SPA pair `11691615` completed `0:0` in `00:27:33`; both
  unchanged long SPA regressions pass in `845.743/796.761` seconds on one
  manifest-sealed fixture.  The fixture manifest replays exactly.
- Both SPA schedule audits prove round 1 requested junk detection, round 2
  did not request it, and round 2 emitted no junk output.  Round-2 recall is
  `0.989333/0.992000`; cross-repeat image-inlier Jaccard is `0.987130` after
  round 1 and `0.961610` after round 2.
- SPA qualification JSON SHA-256 is
  `0cef0a901c8a5b23d755d778f2c4b6b3be038c657e892d29899ad5de60f2a043`;
  sealed output-manifest SHA-256 is
  `820172a5c92e96627c222367a761cf6e6f2d24c584f683d452d9a60848c9f495`.
- The combined cryo-ET, fresh-SPA, and focused-unit evidence admits local
  integration.  Cherry-pick `78eafa9a` applies the candidate to the
  integration branch; checkout-bound focused units pass `36/36` in
  `11.89` seconds and the worktree is clean.
- The change remains unpushed.  Current integration is 68 commits behind
  `github/dev`; repository policy requires rebase and a green parallel
  long-test before publication.  No tolerance, baseline, or scorecard
  changed.

## 2026-07-28 K=4 live checkpoint through iteration 11

- Exact-A100 science `11683600` remains healthy on required UUID
  `GPU-5e619c2e-82b4-ff79-cbcb-ab29514a9f30`.
- Iterations 7/8/9 complete at current sizes `68/70/72`, resolutions
  `21.76/20.92/20.15` A, and Pmax
  `0.909274072/0.923462127/0.946993273`.
- Iteration-9 occupancies are
  `0.2542/0.2345/0.2533/0.2580`; fraction changed is `0.9944`,
  rotation delta `14.313` degrees, translation delta `1.372 A`, and
  convergence remains false.
- Iteration 10 completed at current size 74, resolution `19.43 A`, Pmax
  `0.915517102`, and occupancies
  `0.2522/0.2361/0.2483/0.2633`.  Fraction changed is `0.8970`, rotation
  delta `14.238` degrees, translation delta `1.381 A`, class delta zero, and
  convergence remains false.
- Iteration 11 completed in `1737.4` seconds at current size 76, resolution
  `18.76 A`, Pmax `0.958589825`, and occupancies
  `0.2536/0.2380/0.2454/0.2630`.  Fraction changed is `0.7029`, rotation
  delta `14.538` degrees, translation delta `1.471 A`, class delta zero, and
  convergence remains false.  Timing artifact SHA-256 is
  `bba6c9dda14d185bd1ebe3a89aa4ba66df3521c13d300cb881d3d6dffbdd87`.
- Iteration 12 completed in `2169.9` seconds at current size 78, resolution
  `17.55 A`, Pmax `0.914282132`, and occupancies
  `0.2498/0.2407/0.2433/0.2662`.  Fraction changed is `0.9677`, rotation
  delta `14.275` degrees, translation delta `1.703 A`, class delta zero, and
  convergence remains false.  Timing artifact SHA-256 is
  `c0393ed66d338981db1e9af2ab7757fb631128f79e840373a3aabe7706ea8bbe`.
- Iteration 13 completed in `1990.6` seconds at current size 82, resolution
  `17.00 A`, Pmax `0.932549437`, and occupancies
  `0.2512/0.2425/0.2423/0.2640`.  Fraction changed is `0.9170`, rotation
  delta `13.803` degrees, translation delta `1.074 A`, class delta zero, and
  convergence remains false.  Timing artifact SHA-256 is
  `27bc27d69d921c7ffe5cc81a5749aeb52524a42d60726f5193834e74e4f14b96`.
- Iteration 14 completed in `1993.85` seconds at current size 84,
  resolution `16.48 A`, Pmax `0.945461285`, and occupancies
  `0.2510/0.2435/0.2393/0.2662`.  Fraction changed is `0.9841`, rotation
  delta `14.342` degrees, translation delta `1.280 A`, class delta zero, and
  convergence remains false.  Timing artifact SHA-256 is
  `cdd16b1c0f1ab33419629dc5d29d82f0164b4b4ebc0f2f688b507aaf21d35c6a`.
  Iteration 15 is active at current size 86.
- Non-scoring partial audit `11689329` completed `0:0`: `24/24` direct
  class/iteration FSC-AUC checks pass through iteration 6.  Minimum
  cross-engine FSC-AUC is `0.9967517990550623`; minimum GT delta is
  `-0.00010904820468227161`; both manifests replay exactly.  Summary SHA-256:
  `f80154d314285030bf48b40c72830fb9b673b018c83219e3ccde1b3773675491`.
- Incremental non-scoring audit `11691438` completed `0:0` in `00:23:12`.
  All `36/36` direct checks pass through iteration 9 and all nine boundaries
  pass all four classes.  Minimum cross-engine FSC-AUC is
  `0.9951820759211527`; minimum GT delta remains
  `-0.00010904820468227161`; both manifests replay exactly.  Summary
  SHA-256 is
  `ed56b552709fbb3725a668a5499d393c8926b3087753547064a8dc94422a0ca8`.
- Incremental non-scoring audit `11692389` completed `0:0` in `00:09:09`
  and extends the sealed checkpoint through iteration 11.  Combined direct
  passes are `39/44`, with per-boundary counts
  `[4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 0]`; nine of eleven boundaries pass all
  four classes.  Minimum cross-engine FSC-AUC is `0.9937084319428828`, and
  minimum GT delta remains `-0.00010904820468227161`.  The prior candidate
  has the same `3/4` and `0/4` pass pattern at iterations 10 and 11, while
  current-to-prior FSC-AUC at iteration 11 is at least
  `0.9959507809432725`; this does not identify a phase-FFI-specific
  late-iteration regression.  Both manifests replay exactly; summary
  SHA-256 is
  `2c16d5fb1533307b5c8fc17d53bfbb53c98ceff47d3d4abf06b2ece688f77ada`.
- One-boundary audit `11693520` completed `0:0` in `00:15:26` and extends
  the sealed non-scoring checkpoint through iteration 12.  Combined direct
  passes are `41/48`, with per-boundary counts
  `[4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 0, 2]`; nine of twelve boundaries pass
  all four classes.  Iteration 12 current-to-RELION FSC-AUC is
  `[0.995542481877, 0.994333504958, 0.994270495619, 0.995039176358]`.
  The prior candidate also passes `2/4`; current-to-prior FSC-AUC is at
  least `0.995963684284`, and current minimum GT delta
  `-0.000161770278` is better than prior `-0.000185617138`.  Both manifests
  replay exactly; summary SHA-256 is
  `ed5dabe7e1c7612e9b96a9ca8e41ac4db4d91763519c209fadf2368251ee3713`.
- Iteration-13 audit `11693828` completed `0:0` in `00:09:02`.  The sealed
  non-scoring checkpoint is now `41/52` direct checks with per-boundary
  counts `[4,4,4,4,4,4,4,4,4,3,0,2,0]`; `9/13` boundaries pass all four
  classes.  Iteration-13 current-to-RELION FSC-AUC is
  `[0.994935696247, 0.993232184859, 0.993238960707, 0.994116711636]`, or
  `0/4` passes.  Current-to-prior FSC-AUC is at least `0.995082984803`;
  the iteration's minimum GT delta is `-0.000052124212`, and the combined
  minimum remains `-0.000161770278`.  Both manifests replay exactly and
  independent assertions pass.  Summary JSON SHA-256 is
  `61d33c63914144a205a0f859280da8032d9ab8a4d5450f9cc50a5c49c8b3b7de`.
  Full audit `11683764` remains dependency-held.
- Iteration-14 audit `11694587` completed `0:0` in `00:07:08`.  The sealed
  checkpoint is now `41/56` direct checks with per-boundary counts
  `[4,4,4,4,4,4,4,4,4,3,0,2,0,0]`; `9/14` boundaries pass all four
  classes.  Iteration-14 current-to-RELION FSC-AUC is
  `[0.993698869720, 0.992272292001, 0.992156740849, 0.994615471123]`, or
  `0/4`.  Current-to-prior FSC-AUC is
  `[0.994798488062, 0.994343994904, 0.994058684129, 0.995839429300]`.
  Iteration minimum GT delta is `-0.000124827448`; the combined minimum
  remains `-0.000161770278`.  Both manifests replay exactly and independent
  assertions pass.  Input/output/summary SHA-256 values are
  `cd6a1b3457799e7306b9588850edcd9367a4b2a621860eb1ea68f46213c42ef7`,
  `a1d770931efbe95d088400d32d1c4615471066ffc2a8bb972ec139f0e0bffa35`,
  and `2664e241e3466edadcb50dcc77bc2b1a55707a53fc7b3a72cce8488556cb34f4`.
- Fixed metrics remain K=1 `28/34` strict, `32/34` topology, `34/34`
  evaluated, and K=4 `41/60` direct, `9/15` all-class.

## 2026-07-28 current-head fixed case-22 qualification

- Current local integration source `25ab6e68` contains the later exact local
  score-translation path but had not been qualified on frozen topology
  failure case 22.  A clean detached source with the unchanged fixed fixture
  therefore runs stock RELION and RECOVAR sequentially on one H100.
- Run root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_currenthead_25ab6e68_20260728T011020ET`.
  Both run and runtime roots contain `SAFE_TO_DELETE`.
- Setup `11691762` completed `0:0`; science `11691763` completed both stock
  RELION and RECOVAR reconstructions on `della-h19g3`, then intentionally
  exited `2:0` after `00:52:56` when the unchanged quality gate failed.
  Summary `11691764` and strict independent FSC/topology audit `11691796`
  also failed closed as intended.
- Fixture-manifest SHA-256 is
  `422a79a0a7703d92f9777266e8c34ccd3a7cf5963b354e57a7d9a18f227babee`.
  Grid correction and forced after-max finalization are unset/off.  The run
  is autonomous, scorecard-mode, and makes no scheduler override.  No
  external first-iteration override is supplied: the checked-in K=1
  `firstiter_cc` production defaults provide direct real-reference projector
  handoff and coarse-tree top-two rescore margin `4e-6`.
- RECOVAR/RELION used the same H100 UUID
  `GPU-ef985070-011e-0782-6f0a-94b053dcc120`.  RECOVAR merged-vs-GT
  FSC-AUC is `0.325628942674`; RELION is `0.326048370042`, a
  `-0.000419427367` delta outside the fixed `0.0001` qualification
  tolerance.  Final merged direct FSC-AUC is `0.826067895374`.
- Current size and HEALPix order match through iteration 8.  At iteration 9,
  RELION advances to size 70/order 5 while RECOVAR uses size 72/order 4;
  merged direct FSC-AUC falls to `0.989828446`.  RECOVAR converges after 10
  numbered iterations and RELION after 11.  This reproduces the previously
  classified upstream half-map FSC/resolution boundary split; do not force
  the scheduler or add an iteration.
- Summary, FSC-trajectory, and intermediate JSON SHA-256 values are
  `07df0b9b43d06d8c47269d2059392cf0fef1f8fda6865b1a9a7d5d321e74a105`,
  `eb0d187cb10e2008ce380332bbe116feb7f43df9cb024b8424644a7a063de170`,
  and `7acab11acc004fbf42f1e5239839931c53f5af10b47c50e093ca2df3567ea2d1`.
  The result is not score-admissible.  Fixed K=1 remains `28/34` strict,
  `32/34` topology, and `34/34` evaluated.
- Do not source-bisect against the older same-label `fc70abc3` eight-case
  run.  Its generated particle stack SHA-256 is
  `adc8404ccbc12f53ccfb9cd09ffdf9cdf49a006369b223983f9a11b6fde57e1a`,
  while the immutable scorecard stack is
  `804af933bd315f41f0159f62e93867cf852d70cb29f2f27a525fb2fc3eb68ad9`.
  Initial/GT references, STAR metadata, simulation info, and generation
  config also differ; only CTF and pose pickle hashes match.  That passing
  trajectory is a regenerated replicate, not a code-regression baseline for
  frozen case 22.

## 2026-07-28 frozen case-22 current-versus-b1d source audit

- A non-scoring source-effect audit compares `b1d44427` and `25ab6e68` on
  the exact same immutable case-22 fixture.  Both materializations bind the
  same ten files and fixture-manifest SHA-256
  `422a79a0a7703d92f9777266e8c34ccd3a7cf5963b354e57a7d9a18f227babee`;
  all 51 input-manifest entries replay exactly.
- All ten numbered pose and translation arrays are exact.  Significant
  counts differ by one for five particles at iteration 2 and one at
  iteration 6.  Maximum numbered Pmax absolute delta is `0.001735568047`.
- Minimum current-versus-old numbered merged FSC-AUC is
  `0.999999999776`; maximum internal half-map FSC delta is
  `3.943219781e-6`.  Iteration-8 shell-20 moves only
  `+3.576278687e-7`, from `0.501799643040` to `0.501800000668`, and
  therefore remains above RELION's `0.499048` value.
- Final all-data differs at exactly one pose/translation row, input index 15
  (`16@particles.128.mrcs`): one `0.352941155`-pixel translation step and
  `1.851751704` degrees rotation.  Final merged current-versus-old FSC-AUC
  is `0.999999971570`.
- Classification is
  `current_vs_b1d_numbered_pose_translation_exact_with_support_mass_changes`.
  Changes after `b1d44427` do not explain or repair the frozen upstream
  shell/topology failure.
- Run root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_current_vs_b1d_fixedfixture_audit_20260728T030000ET`.
  Run and runtime roots contain `SAFE_TO_DELETE`.  The local CPU audit
  completed `0:0` in `00:22.88`, maximum RSS `767104 KiB`.
- Report/input-manifest/output-manifest SHA-256 values are
  `9a2d226e9d758134d7716b1f7d3a2d4698f8c7f14a0e43ce9422f54132560d49`,
  `5b898012527f2c4fb8c9a2bf57d0b2c3e5d79693057e0b9b03e09ec7176b8589`,
  and `8b8731865d484e49f943b12b9b8dd4a812f3e980ea684a3548aa9fab21c94233`.
- Fixed K=1 stays `28/34` strict, `32/34` topology, and `34/34`
  evaluated.

## 2026-07-28 native RELION C++ FSC equivalence

- A binding-only `compute_fsc_from_bpref` diagnostic routes already
  accumulated compact-half data/weight arrays through RELION's native
  `getDownsampledAverage` and shell-FSC methods. It changes no production EM
  path.
- Synthetic same-operand equivalence passes; the final full focused binding
  file passes `16/16` in `26.04 s`. The four focused production-helper tests
  also pass in `10.26 s`.
- On frozen case-22 RECOVAR index 7 / physical RELION iteration 8, native
  C++ computes shell-20 FSC `0.501799971753`. The NumPy scheduler helper and
  both stored RECOVAR curves are `0.501800000668`, only
  `2.891466166e-8` away.
- Native C++ and NumPy are both above `0.5`; stock RELION is `0.499048`.
  The accepted classification is
  `native_relion_fsc_loop_confirms_recovar_accumulator_shell_split`.
  FSC-emulation arithmetic is not causal; the remaining locus is upstream
  accumulated half-map content.
- The corrected CPU audit completed `0:0` in `00:03.35`, maximum RSS
  `730828 KiB`. The first attempt failed before science because flattened
  accumulator arrays had not been reshaped; its output is rejected.
- Run/runtime roots are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_fsc_cpp_equivalence_e32cb3c9_20260728T032500ET`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case22_fsc_cpp_equivalence_e32cb3c9_20260728T032500ET`;
  both contain `SAFE_TO_DELETE`.
- Report/curves/input-manifest/output-manifest SHA-256 values are
  `0d10c9566aeba5e466588d044069d1faaee1e34fc00876566822ce6643a02f66`,
  `b3bf67b4ab2c7f7648aca41a990a063b27179521424005adf104ede24fc08034`,
  `ae2ac2ffa59f28886847bb871c81f456d72d0a1d5890e68f3dfbc98246abea9b`,
  and `5bdf89ae286af51a7e7dd7289c8ee162c3648908e50f78f1c2a9b1c2d48ebc70`.
- This is non-scoring. Fixed K=1 remains `28/34` strict, `32/34`
  topology, and `34/34` evaluated.

## 2026-07-28 frozen case-22 joined BackProjector audit

- A passive RELION diagnostic captures the joined, current-size
  BackProjector immediately before SSNR reconstruction at physical iteration
  8.  Capture-off and capture-on arms use the same patched RELION binary,
  immutable frozen case-22 fixture, and H100 UUID
  `GPU-1fdb3b99-e7ff-fe6d-4f59-9d2cc85fa319`; the capture is therefore
  qualified against a same-device control before comparing accumulators.
- Science `11695312` completed `0:0` in `00:18:28` on `della-h19g3`;
  corrected audit `11695904` completed `0:0` in `00:00:24`.  Minimum
  control/capture and capture/frozen-oracle half-map FSC-AUC values are
  `0.9999999526633185` and `0.9999999999441119`.  Map, pose, translation,
  optimiser, support-count, Pmax, sampling, and numeric-state envelopes pass.
- At current size 70, the physical-iteration-8 RELION-versus-RECOVAR
  accumulator relative L2 values are `0.1490804186` and `0.1509590837` for
  the half-1/half-2 complex data arrays, and `0.0634602585` and
  `0.0851252945` for the corresponding weight arrays.
- The sealed classification is
  `capture_and_oracle_within_fixed_repeat_envelopes_accumulator_comparison_descriptive`.
  Causal attribution to the accumulator boundary is deliberately false:
  this run has only one dump-on RELION accumulator, so it does not establish
  the native exact-device accumulator repeatability envelope.
- Run root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it8_joined_bpref_d476_capture_20260728T035235ET`.
  Its runtime root is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case22_it8_joined_bpref_d476_capture_20260728T035235ET`.
  Both contain `SAFE_TO_DELETE`; science-input, science-output, and
  audit-output manifests replay exactly.
- Audit JSON/Markdown SHA-256 values are
  `ea740197a3dc3407d6e813b967c42edcf5cb23746d44df9c4b788bc6b35fc4d4`
  and
  `296adc344ec7f9a91dbb8bceb41e7fccc762759f63e2a7e9673e6af73f00e2d4`.
  Science-input, science-output, and audit-output manifest SHA-256 values are
  `677c030d80d678cda714bbf03ed98ac20d2c1ca0e88229fe847534004d3df8af`,
  `246d4eb5d6fff4a911e660f60e3a9b9b1dfb702cf19ca70693d444372dec7f76`,
  and
  `754c9396d576abf7edf2836fc929fffb81ecfca5dc2693afc0d43fe6e0de7b50`.
- This is non-scoring.  The fixed K=1 score remains `28/34` strict,
  `32/34` topology, and `34/34` evaluated.  The next admissible discriminator
  is a second dump-on RELION iteration-8 accumulator on the exact same
  physical H100, followed by a common pre-scatter operand capture only if
  the cross-engine residual exceeds that native repeat envelope.

## 2026-07-28 exact-UUID BackProjector repeat accepts the residual

- The predeclared same-device native-repeat discriminator completed on the
  exact original capture H100 UUID
  `GPU-1fdb3b99-e7ff-fe6d-4f59-9d2cc85fa319`.  Science `11696239`
  completed `0:0` in `00:09:05` on `della-h19g3`; dependent audit
  `11696241` completed `0:0` in `00:00:12`.
- Identity/topology and all map repeat envelopes pass.  Minimum
  capture/repeat half-map FSC-AUC is `0.9999999993731247`.
- Native-repeat relative L2 is `2.1839458038e-5/3.4281907171e-5` for
  half-1/half-2 complex data and
  `1.2488977869e-5/9.8865142204e-6` for weights.  The corresponding
  cross-engine values are `0.1490804186/0.1509590837` and
  `0.0634602585/0.0851252945`, exceeding native repeat by factors from
  `4403.4622` to `8610.2435`.
- Classification is
  `cross_engine_accumulator_residual_exceeds_two_x_native_repeat`.
  The case-22 iteration-8 accumulated-content mismatch is therefore outside
  native RELION repeat variation.  This still does not separate
  particle/posterior state from backprojection arithmetic; a matched
  pre-scatter operand capture is the next causal boundary.
- Run/runtime roots are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it8_bpref_repeat_exactuuid_20260728T042700ET`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case22_it8_bpref_repeat_exactuuid_20260728T042700ET`;
  both contain `SAFE_TO_DELETE`.  All science and audit manifests replay
  exactly.
- Audit JSON/Markdown SHA-256 values are
  `6c0e0c371337e365093978036ca32bc182ba52c0eefe1544b4ac879d24ec4e7a`
  and
  `d11eb78bec5f0a9c9e0a025b957cf3369c12b0a575ca2955782647b87514fa24`.
  Science-input, science-output, and audit-output manifest SHA-256 values are
  `5d3e09f8a5d03d3b822720b886d8d3e9f497be290d6952e00ef3feecb698b28e`,
  `a48ae2cd942960dc533835de5f3f4a95f7a445dfe6446f23cbee381e73022463`,
  and
  `5bd4e12f21da9d30c3366898a8d328af75bc81fa9cd4c078be7693b8e278e0ed`.
- This is non-scoring; fixed K=1 remains `28/34` strict, `32/34`
  topology, and `34/34` evaluated.

## 2026-07-28 case-22 iteration-8 pre-scatter run is live

- Science `11696749` and after-success audit `11696750` are submitted on
  Della.  One physical A100 runs capture-off RELION, capture-on RELION, and
  RECOVAR sequentially through the frozen case-22 physical-iteration-8
  boundary.  The RECOVAR target is half 1, class 1, current size 70.
- The auditor was corrected before submission for exact-local soft-posterior
  semantics: particles can have a variable number of positive contributor
  rotations.  It performs unique rotation-matrix matching at tolerance
  `1e-6`, records membership and oversampled-identity differences, and
  compares native operands only on strict RELION-emitted radius-supported
  rows.  It makes only a one-sided claim that emitted RELION rows have
  positive RECOVAR weight; RECOVAR-only diagnostic-window pixels and device
  scatter geometry are outside this boundary.
- The predeclared operand separation threshold is relative L2 `1e-3`,
  over 29 times the largest exact-device native joined-BackProjector repeat.
  Capture inertness remains FSC/FSC-AUC gated; correlation is not computed.
- The corrected auditor reproduces the sealed case-20 control: 9,169 common
  contributors, 9,716,168 rows, data relative L2
  `0.023864238968028115`, and weight relative L2
  `0.022133396289162388`.
- Run/runtime roots are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it8_prescatter_operands_a100_20260728T045805ET`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case22_it8_prescatter_operands_a100_20260728T045805ET`;
  both contain `SAFE_TO_DELETE`.
- Science launcher, audit launcher, and auditor SHA-256 values are
  `e15a8a178d194a85150b7ecdfe3a9d985d8d138da96bb4ac96c30adc62b9b79d`,
  `26e1177478a213232bc3fcc4ce066fca0166d9fdc35a59325bc9e75a7e5f601d`,
  and
  `85dedfea4cc2f1b81c63cb84e55ed03b32bee6ce6100caa3f5079b4c53f8eb2a`.
  This live run is non-scoring; fixed K=1 remains `28/34` strict,
  `32/34` topology, and `34/34` evaluated.

## 2026-07-28 exact-A100 K=4 full audit is terminal

- Authoritative audit `11695141` completed `0:0` on `della-h16n3` in
  `01:33:24`, with maximum RSS `4675620K`.  Its science input is terminal
  job `11683600` on exact A100 UUID
  `GPU-5e619c2e-82b4-ff79-cbcb-ab29514a9f30`.
- Classification is `fixed_score_unchanged`. Exact topology and hardware
  gates pass, but no scorecard change is admissible.
- Terminal K=4 is `41/60` direct class checks and `9/15` all-class
  iterations. Per-iteration direct passes are
  `[4,4,4,4,4,4,4,4,4,3,0,2,0,0,0]`; minimum cross-engine FSC-AUC is
  `0.9908413238810354`, and minimum GT FSC-AUC delta is
  `-0.00021150154889848505`.
- Grid correction was unset. The run remained nonconverged and did not force
  final all-data.
- The hardware-matched source-effect report gives minimum current-to-prior
  FSC-AUC `0.9931728696060352`, maximum defect
  `0.006827130393964764`, and fine/coarse assignment mismatch totals
  `16387/7756`.
- All persistent science outputs and all topology/source-effect/audit input
  and output manifests replay exactly.  The already-documented expired Slurm
  spool launcher is the only non-persistent first science-manifest entry;
  replay begins at entry two and the durable launcher is independently
  pinned.
- Fixed-score JSON, source-effect JSON, audit-output-manifest, and
  audit-input-manifest SHA-256 values are
  `1767cca23378fc1ba36353564b47e4200dcfef896bea07e18270991fe0da09dd`,
  `0ff52feb98f4e2e7104c7414a7bdce68f36ccb9baba6ae96ac57609c90407cf8`,
  `740df77aec7dc442edea0fdf730fd1a9efa0abca083003f4c6aaa2b32b800e58`,
  and
  `fc54120e547fab451eb700d26ecd0c937324e0978d44c11a9b099657d1421960`.
- Run/runtime roots are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_full15_phaseffi_exactgpu_retry1_31c4a0ca_20260727T040500ET`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k4_full15_phaseffi_exactgpu_retry1_31c4a0ca_20260727T040500ET`;
  both contain `SAFE_TO_DELETE`.

## 2026-07-28 case-22 iteration-8 membership is state-driven

- Science `11696749` completed `0:0` on exact A100 UUID
  `GPU-eb1c5b04-20c1-b6c9-16e6-b3dc87905bd7` in `01:29:33`; its
  fail-closed audit `11696750` completed `0:0` in `00:03:31`. Capture
  inertness passes with minimum FSC-AUC `0.9999999881069878`. Grid
  correction was unset, the run remained nonconverged, and final all-data
  was not forced.
- The original operand report localizes before scatter:
  `52310` RECOVAR positive contributors versus `52909` RELION-emitted
  contributors, `49678` exact common rotation matrices, and matched-operand
  relative L2 `0.19790170746442434 / 0.14848135432332882` for data/weight.
  Every RELION-emitted source row has positive RECOVAR weight.
- Corrected CPU job `11701525` completed `0:0` in `00:00:45`, maximum RSS
  `492040K`. Its classification is
  `candidate_grid_and_significance_membership_differences`.
- Across 1490 half-1 particles, RELION/RECOVAR expose `79872/79424`
  candidate matrices. `75440` match exactly with zero matrix error;
  `4432/3984` are engine-only. Candidate sets are exact for `945/1490`
  particles.
- Positive contributors contain `49678` both-positive exact matches.
  Matched candidates add `1385` RELION-only and `1002` RECOVAR-only
  positives; unmatched candidates add `1846/1630` positives. Positive sets
  are exact for `547/1490` particles. RECOVAR-only unmatched positives have
  reconstruction-mass median `0.0005053590`, p95 `0.0522131718`, and
  maximum `0.9668054618`.
- Exact sorted stack-identity sets supersede the original order-sensitive
  array report. The original comparison of RELION oversampled-child
  identities with `RECOVAR_global_index % 8` is invalid because RECOVAR
  addresses the global fine grid; captured rotation matrices are the
  authoritative identity gate.
- Membership JSON/NPZ SHA-256 values are
  `c3065418f4bed2c3f7e9ca7edace2604d5135ffa60d860dc1e7d74bd65f67ef0`
  and
  `2f414fedc9c5c053ae1fd2862250b6a7214cf318092a01f47971b6354b82d488`.
  Input/output manifest SHA-256 values are
  `9188d41b0a98d12d9ab4b527c3fca03ede8025857de6e95c86f7c1725ad6e39f`
  and
  `0f25a35153a67a8ad2f302dbb0486c5488e6cdcd97e31b40ed998397bdf1787d`.
  All manifests replay exactly.
- The complete run/audit roots are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it8_prescatter_operands_a100_20260728T045805ET`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it8_membership_audit_20260728T074000ET`.
  Their runtime roots and run roots contain `SAFE_TO_DELETE`.

## 2026-07-28 case-22 trajectory moves the next capture to iteration 3

- The standard exact-image-identity particle-state auditor now covers all
  physical iterations 1 through 8 from the same pre-scatter run. It is
  diagnostic/non-gating and computes no correlation.
- Physical iteration 1 is arithmetic-level: rotation p95
  `9.241452606e-6` degrees, Pmax p95 zero, and exact support counts for all
  3000 particles.
- Physical iteration 2 retains arithmetic-level poses/Pmax but support counts
  differ for `1101/3000` particles. Those count differences do not enrich
  iteration-3 pose tails (`1.043955256x`).
- Physical iteration 3 is the first meaningful posterior/state boundary:
  Pmax p95 is `0.0361012003` and 61 particles exceed `0.1` degree. Its top
  5% Pmax deltas enrich physical-iteration-4 pose tails by
  `3.034722222x`.
- Physical iteration 4 is the first systematic pose boundary: rotation p95
  `1.846612276` degrees and `167/3000` particles above `0.1` degree. The
  tail grows to `440/3000` and p95 `2.415704723` degrees by iteration 8.
- JSON/NPZ SHA-256 values are
  `40d5dc418de55508471ae190ed98613d23879e7ed40e0d78e0c6e64fad8f158e`
  and
  `6ae16cc60b602281db31f2237428b29461d321a9fe0ca7afa446e00e4d031d1d`.
  Input/output manifest SHA-256 values are
  `6573f1b74e3ab3cc12c0743baa53ffc326e4a8c72422b99da26dc161415b8cb4`
  and
  `c6d38e6f7ad32bfa4eb15844e16504b91bb9c091e6f34a3256ee6206f95c642e`.
  All manifests replay exactly. The evidence root is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it1_it8_particle_state_trajectory_20260728T075200ET`.
- Same-allocation A100 science `11702248` now targets physical iteration 3,
  half 1, class 1, current size 80; after-success audit `11702340` is
  dependency-held. Control RELION, captured RELION, and RECOVAR run
  sequentially on one A100. Launcher/auditor/membership-analyzer SHA-256
  values are
  `006e89bc582cb7cd70c409d73769bd1e41f1d3616cd89d6f9d8439655dc26a3b`,
  `5624b8a272edfe7db17603150204207d4264f0751bf39241ffa829fe0e89fcc6`,
  and
  `b548aeb34632834260550bf794ed858b7699314337a0e4d90bb791a37d2edda6`.
  Run/runtime roots are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it3_prescatter_operands_a100_20260728T080200ET`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case22_it3_prescatter_operands_a100_20260728T080200ET`;
  both contain `SAFE_TO_DELETE`.
- These diagnostics are non-scoring. Fixed K=1 remains `28/34` strict,
  `32/34` exact topology, and `34/34` evaluated; K=4 remains `41/60`
  direct and `9/15` all-class.

## 2026-07-28 same-A100 FSC moves the earliest map boundary to iteration 2

- The checked-in FSC-only trajectory auditor was run against the sealed
  physical A100 products from science `11696749`. The direct first attempt
  failed closed because the diagnostic has only one RECOVAR final product;
  the accepted invocation uses a read-only numbered-map view and evaluates
  all 8/8 numbered iterations with complete topology.
- Physical iteration 1 remains effectively exact: merged cross-engine
  FSC-AUC `0.9999999999734349` and RECOVAR-minus-RELION merged GT FSC-AUC
  `+1.0720109161477254e-08`.
- Physical iteration 2 is the earliest non-negligible map boundary: merged
  cross-engine FSC-AUC `0.9999933517250045` and merged GT delta
  `-9.736476152685802e-05`. Iteration 3 is already downstream, at
  `0.9997433171093425` and `-0.0003074829002873425`.
- The same-A100 JSON/Markdown/shellwise-NPZ SHA-256 values are
  `551b63534bd7b5ca4ee74060cec29dbfe57c66122a9df221fec35b8ae51d32df`,
  `baebad3464069d07d75901fe1a712409bb264d2bf2ad78bc64ce3f205ce7a5d1`,
  and
  `84fa1d225527ff3e1f63106f7e9c2cc84ead5a70bc83371c17259d3372f37c0e`.
  The output manifest replays exactly.
- The sealed audit root is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it1_it8_samea100_fsc_20260728T081952ET`;
  its runtime root is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case22_it1_it8_samea100_fsc_20260728T081952ET`.
  Both contain `SAFE_TO_DELETE`.
- The earlier physical-iteration-3 capture remains useful for measuring
  downstream amplification as science/audit `11702248`/`11702340`.
  The earliest bounded pre-scatter discriminator is now physical iteration 2,
  half 1, class 1, current size 60. Hash-pinned same-allocation A100
  science/audit `11702643`/`11702647` are running from
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_prescatter_operands_a100_20260728T082538ET`.
  Science launcher, target auditor, and current membership-analyzer SHA-256
  values are
  `acedc3f794c5f45842de62f7b055adbb7c0920b4d86a4a423444d412502590a4`,
  `63a9270be0c1855a6ef9a326ae799cc1e49a52bc7252e1ae27ffe3c075f88a5d`,
  and
  `839585d3e4f38a480e9caca64298bbf0dd765b9f11d7a61d2805bd69e5b00fd2`.
- These diagnostics remain non-scoring. Fixed K=1 is `28/34` strict,
  `32/34` topology, and `34/34` evaluated; fixed K=4 is `41/60` direct and
  `9/15` all-class.

## 2026-07-28 iteration-2 capture is bounded to a frozen support cohort

- Original iteration-3 science `11702248` failed closed at the passive 1 TB
  envelope before RECOVAR; dependency audit `11702340` was canceled.
  Retry science/audit `11702824`/`11702825` retain the exact capture semantics
  and use a predeclared 5 TB all-particle envelope.
- Original iteration-2 science `11702643` and audit `11702647` were retired
  before capture/RECOVAR evidence because full-particle passive panels were
  deterministically impractical. This is not a science failure.
- New `scripts/select_k1_bpref_support_cohort.py` binds the sealed trajectory,
  canonical stack order, and sealed RELION `part_id`/MPI-rank headers. Its
  rank-1 cohort has 64 particles: 2 deepest (`<=-3`), 16 `-2`, 24 `-1`, and
  22 exact-support controls. Canonical row SHA-256 is
  `07901c4f17e9e13d878f9341fe6293a9f2968673c77784ae176d45c017b90c18`.
- A separate diagnostic RELION tree filters explicit `part_id` values before
  passive allocation. Diff SHA-256 is
  `82e79e3e07079e553280e2089d2fc5c4887fb43a27c032ee6df3228eb789bd21`;
  the fresh CUDA-12.6 `sm_80` binary SHA-256 is
  `53a59a64aad8011de26a820ca9b9ae76ea7bc3e8ffb9319f518391951d82dd66`.
  Production backprojection is untouched. The exact diff is now checked in
  as
  `docs/patches/relion_bpref_prescatter_part_id_filter_bc319d0.patch`;
  a fresh detached `bc319d0` apply check reproduces the pinned diff hash.
- Subset science `11703645` is live with after-success audit `11703646`.
  Run/runtime roots are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_subset_prescatter_operands_a100_20260728T085159ET`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case22_it2_subset_prescatter_operands_a100_20260728T085159ET`;
  both contain `SAFE_TO_DELETE`.
- The fixed targeted suite passes 63/63 tests, including the exact
  diagnostic-patch byte/guard check; Ruff and `git diff --check`
  pass. Live diagnostics are non-scoring. Fixed K=1 remains `28/34` strict,
  `32/34` topology, `34/34` evaluated; fixed K=4 remains `41/60` direct and
  `9/15` all-class.

## 2026-07-28 threshold substitution remains residual and iteration-2 probe fails closed

- `scripts/analyze_k1_bpref_contributor_membership.py` now decodes captured
  RELION `significant_weight`/`weight_norm` float32 fields and applies their
  normalized ratio to RECOVAR's saved pre-pruning maximum posterior on common
  rotation matrices.
- Sealed iteration-8 native common-candidate mismatches are `1385/1002`
  RELION-only/RECOVAR-only. Under the RELION threshold they are `1426/940`,
  so schema v2 classifies
  `common_candidate_significance_gap_persists_under_relion_threshold`.
  Candidate counts remain `79872/79424` with `75440` common matrices. This
  rules out a scalar threshold substitution as the sole cause without
  selecting a production patch.
- The read-only replay root is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it8_relion_threshold_substitution_v2_20260728T093500ET`;
  runtime is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case22_it8_relion_threshold_substitution_v2_20260728T093500ET`.
  Both contain `SAFE_TO_DELETE`. JSON/NPZ SHA-256 values are
  `7f76cdde7df92eb04f36f1b38c21df02bdf5c6dbd9d2ccbc1dc9fbca6427989a`
  and
  `e419dd42c50d557003cff688d1d81dd1bc012d86e2deae2bab9367f5144f44d5`.
- Iteration-2 subset science `11703645` completed control RELION but failed
  before its first capture artifact with CUDA `invalid argument` at the
  passive launch line 226 and exit `134:0`. RECOVAR did not run and no FSC
  result exists. The stranded audit `11703646` was canceled. This is a
  diagnostic-harness failure; the next retry is a bounded launch-dimension
  probe.
- The fixed targeted suite passes 64/64 tests; Ruff, `git diff --check`, and
  the exact patch-byte hash guard pass. All evidence here is non-scoring.
  Fixed K=1 remains `28/34` strict, `32/34` topology, `34/34` evaluated;
  fixed K=4 remains `41/60` direct, `9/15` all-class.

## 2026-07-28 iteration-3 recovery localizes a diagnostic coverage gap

- Recovery audit `11706550` completed `0:0` in `00:02:06` with maximum RSS
  `10286288K`.  The original science `11702824` completed all control,
  capture, and three-iteration RECOVAR science before its wrapper rejected
  1,475 contribution identities against an expected 1,490.  Capture
  inertness passes at minimum FSC-AUC `0.9999999999673241`; all 3,000
  RELION artifacts validate and no correlation metric is used.
- The common 1,475-particle panel contains 116,488/115,880
  RELION/RECOVAR candidates and 114,368 exact matrix matches.  Matched
  positive-membership differences are 507/411, unmatched positive
  differences are 339/108, and data/weight pre-scatter relative L2 values
  are `0.08984580198575254/0.06786875013555252`.
- RECOVAR's authoritative iteration log accounts for all 1,490 half-1
  images.  The contribution artifacts account for exactly the first four
  bucket groups, 1,233+143+70+29 = 1,475 images.  The omitted 15 are exactly
  the rotation-chunked 4,096-rotation group (13) and 8,192-rotation group
  (2).  This supersedes the initial `RELION-only particle` interpretation:
  it is diagnostic coverage, not missing science execution.
- `sparse_pass2_bucketed.py` now captures rotation-chunked contribution
  operands in authoritative global rotation order and calls the existing
  writer once per bucket.  The targeted inertness test covers ordinary and
  passive-shadow modes under both default and opt-in float32 fine posterior,
  and requires the complete returned production tree to be array-exact with
  capture disabled.
- The earlier iteration-2 CUDA `invalid argument` is independently localized
  to a 10,131,532,800-byte monolithic passive allocation for 145,568
  orientations.  Bounded science `11706338` completed `0:0` in `00:09:34`
  with a 512 MiB cap and 19 ordered chunks; CPU structural audit `11706638`
  completed `0:0` in `00:00:32`, validating 182,140,981 emitted rows.
- The exact tested RELION diff is checked in at
  `docs/patches/relion_bpref_prescatter_chunked_capture_bc319d0.patch`.
  Its SHA-256 is
  `1a9680d93ae6ab0577a7901999dca464c7929ed10b36c36744fc87672889668f`,
  and it leaves production backprojection untouched.
- Recovery and chunk-probe run roots are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it3_prescatter_operands_a100_retry1_20260728T083648ET`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_subset_chunked_probe_a100_20260728T101500ET`;
  both run/runtime pairs contain `SAFE_TO_DELETE`.  The next gate is a
  hash-pinned complete 1,490-particle RECOVAR recapture and replay against
  the sealed RELION iteration-3 artifacts.
- This remains non-scoring.  Fixed K=1 is `28/34` strict, `32/34` topology,
  `34/34` evaluated; fixed K=4 is `41/60` direct and `9/15` all-class.

## 2026-07-28 complete case-22 iteration-3 contributor audit

- Science `11707749` completed `0:0` in `01:06:35` on A100 UUID
  `GPU-77778a79-1ab4-832d-c0af-8c521897325f`, using clean RECOVAR commit
  `7c55b2a5b25afd80ce88a9778bf03424a3b27f1e`.
- The capture has 35 artifacts and exactly 1,490 unique particle identities,
  including all seven 4,096-rotation shards and both 8,192-rotation shards.
  Maximum actual rotation count is 5,864.  Coverage JSON SHA-256 is
  `6dfbb71af4464e70b07aa62b9d4b0a6f7c9838989836e2ea08ddf151dd0da075`.
- First audit `11707938` failed closed before analysis on a stale pre-import
  CUDA checksum after the science provenance gate rebuilt the library under
  CUDA 12.8.  The failure record SHA-256 is
  `3cf7197e2255f407fe72a524922febc6b587d298c46b9c5266d7a1ca4e106983`.
  Corrected audit `11709617` used the runtime-rebuilt input manifest and
  completed `0:0` in `00:03:34`.
- Runtime-input, science-output, and audit-output manifests have SHA-256
  `6b9ad474875bdcb7e1f2d96f6d161c9abcd4c37104f973ffa60b0525ef7bcc28`,
  `75d95f07f38df8864a52e614a450f82ae9e66d184a464792f1bfaa4a1489208a`,
  and
  `5fe9a2d63f89f0fbad5d74c66e05c8cf52492f1c0ac07af974e020d980790d87`;
  all replay exactly.
- Capture inertness passes at minimum FSC-AUC `0.9999999999673241`; no
  correlation is computed.  Exact RELION/RECOVAR particle sets are
  `1490/1490`, with zero whole-particle omissions.
- Candidate totals are `147608/146456`, exact matrix matches `144808`, and
  engine-only candidates `2800/1648`.  Candidate sets are exact for
  `1157/1490` particles.
- Positive totals are `50538/50069`, exact both-positive rotations `49479`,
  engine-only matched positives `652/479`, and engine-only unmatched
  positives `407/111`.  Positive sets are exact for `977/1490` particles.
  All 1,490 RECOVAR reconstruction thresholds are positive.
- Matched-contributor data/weight pre-scatter relative L2 values are
  `0.08982922016327309/0.067862978074296`.  The complete classification is
  `candidate_grid_and_significance_membership_differences`, not a missing
  chunked-execution or scatter-only gap.
- Membership JSON/NPZ SHA-256 values are
  `292323c5098f13324cd4fbc25c82bbaff6ae0fdb3343e9be7680cf9bb996bfd1`
  and
  `d45548bd939510995e57b6b5d7a67a324b554ff0c5cce48e733d635bc239a3a6`;
  pre-scatter JSON/NPZ values are
  `99a6b9436ded451243120d230e9f5c4099c4f6f82a94b5ca3918b7a46e039b4f`
  and
  `7b2fe7f99efe963a64012bd6845b3d8e7b808b9ce8b6dd392ad00bd59a6fa3aa`.
- Run/runtime roots are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it3_recovar_chunked_capture_7c55b2a5_20260728T110100ET`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case22_it3_recovar_chunked_capture_7c55b2a5_20260728T110100ET`;
  both contain `SAFE_TO_DELETE`.  Terminal record SHA-256 is
  `108ccacf4cc23b7dc5244550cd175db6074a5006a81f7570d41542e8e300ee1c`.
- This is non-scoring.  Fixed K=1 remains `28/34` strict (`82.4%`),
  `32/34` topology, and `34/34` evaluated; fixed K=4 remains `41/60` direct
  and `9/15` all-class.  Next inspect physical-iteration-2/3 candidate-grid
  construction and significance support before selecting a production fix.

## 2026-07-29 case-22 incoming-map state is causally sufficient

- Exact-A100 state-swap science `11719941` completed `0:0` in `03:10:00`.
  The physical-iteration-3 target uses exact RELION non-map state in every
  arm; only the incoming half maps differ.
- All RELION state reproduces `8/8` coarse parents with Jaccard `1.0`.
  Restoring only RECOVAR incoming maps produces `7/8`, Jaccard `0.875`, and
  the same single missing RELION parent.
- Centered raw-score residual is max/RMS `0.003311/0.000573` for all RELION
  and `9.83622/2.16229` with RECOVAR maps.  The accepted classification is
  `incoming_recovar_reference_maps_are_sufficient_for_target_support_divergence`.
- Summary, all-RELION, and RECOVAR-map SHA-256 values are
  `b3f0d7b9fcf9b005ecda3d121ff1d0eec5534af984d69aa31510a8351a5ecb37`,
  `bbca2f3c32672bee042a442785d187ad6e62c4bd9d1e0cf64bf02eb43083f7e4`,
  and
  `fe648c5802ff3712d92cac89f74608e9032f34f05caa5b8a6c5cc77998a362a9`.
- Across the 24 shared target fine rotations, RECOVAR-map projections have
  RMS-amplitude ratio `1.013936581207998` relative to RELION.  One global
  least-squares scale reduces relative L2 from `0.014153213970763244` to
  `0.002449800018841169`.
- The active non-scoring hypothesis is that this amplitude difference alone
  transfers the `8 -> 7` support decision.  The bounded falsifier is a
  reciprocal global/shellwise map-amplitude state-swap factorial under the
  same exact non-map state.  No production patch or scorecard promotion is
  admissible from this diagnostic alone.
- The sealed run root is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it3_maps_stateswap_190195e2_20260728T162431ET`
  and contains `SAFE_TO_DELETE`.  Grid correction and forced final all-data
  were unset.

## 2026-07-29 K=4 target audit corrected to the metadata frame

- Exact-A100 capture `11746808` completed `0:0` in `00:36:01` at source
  `31c4a0ca`, target UUID
  `GPU-5e619c2e-82b4-ff79-cbcb-ab29514a9f30`, physical iteration 2,
  current size 38.  The 109,184-candidate artifact SHA-256 is
  `3c4c566b6f2fce613f4d5869d2d3ccf53a2bcd1b3c26e5a32138588464049485`.
- Audit `11746841` used a v1 analyzer that directly compared relative
  pass-2 shifts with absolute metadata offsets.  Preserve its output as
  superseded provenance; do not use its `third_winner` classification.
- Commit `72f21482` converts candidates using RELION's written-metadata rule:
  `round_away_from_zero(previous_absolute) + relative`.  Compile, Ruff,
  `git diff --check`, and 2/2 focused unit tests pass.
- Corrected audit `11757252` completed `0:0` in `00:03:02`.  Incoming
  `[-3.6088611765, -0.6088611765]` pixels gives search base `[-4, -1]`.
  Relative phase/RELION candidates `[2.041065693, 0.041065693]` and
  `[3.041065693, 0.041065693]` therefore map to absolute
  `[-1.958934307, -0.958934307]` and
  `[-0.958934307, -0.958934307]`.
- Those two candidates have bitwise-equal raw score, prior, total score, and
  probability.  The first-index tie break selects phase-away index 80 over
  native RELION index 82.  The audit initially labeled the phase path causal,
  but the pre-phase control below rejects that label.  Preserve
  `fixed_relion_state_phaseffi_reproduces_away_winner__phase_score_path_is_causal`
  only as superseded provenance.  Its JSON SHA-256 is
  `0be2a5608cc0dc27b3ecd7bb683438f14d70fb4ce3e81706042a9f0b3cc6aa8d`.
- Same-device pre-phase control `11757378` completed `0:0` in `00:35:28`
  against parent `4181d340`.  It changes 20,875 of 109,184 background
  scores, with maximum absolute difference `0.0001220703125`, but leaves
  the target 80/82 pair bitwise tied and selects index 80 in both
  implementations.  The superseding classification is
  `phaseffi_changes_background_scores_but_not_target_tie__prior_phase_causal_label_rejected`.
  Classification, validation, capture, and completion SHA-256 values are
  `b4dafc3fa9d8b970122a706dec62c1e353545faa395a5e492a2f818e7ab62a63`,
  `16b281455854c17bb8b5009eddd7a5866f2e15a224f3cf9c74bea942473c03e1`,
  `459b9ebaa709768c194a2a1d50a0ebd21f3c59c2e6488aec535f4f8ee1165f34`,
  and
  `a4520e9e78f9df32e6ebab4048c854f2ae80c8ab6466030eafb83c11a355b923`.
  Its run root is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_orig53722_prephase_control_4181d340_20260729T094500ET`;
  runtime is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k4_it2_orig53722_prephase_control_4181d340_20260729T094500ET`;
  both contain `SAFE_TO_DELETE`.  Launcher SHA-256 is
  `8028dd3b61c995134878b6587fccd36bf1572f2bf050a9329d31981381b59505`.
- Native same-A100 job `11759665` failed `1:0` only after atomically sealing
  bounded score and operand artifacts.  Both operand rows replay that
  process exactly.  Fine-score and fine-operand SHA-256 values are
  `20ddb722cc747477babf580128ffaabd5ed9a5c1d38b8b967c9feba34b19b6f3`,
  and
  `0b3e3d78c85af6431e26e5c9ee379f4eb10b1891d31890c961e37d31f97d18c5`.
  Its later failure was an unrelated dense BPref-factor cap.
- Superseding provenance audit rejects these artifacts for authoritative
  iteration-2 comparison.  Fresh `--continue run_it001` used restart
  perturbation `-0.2149905264377594`, not the uninterrupted authoritative
  value `+0.27053284645080566`.  The resulting
  `-0.9710467457771301`-pixel grid displacement exactly matches the native
  capture.  The native `0.30828857421875` target raw-diff2 separation
  therefore describes only the restarted control state.
- Jobs `11759666` and `11759668` were canceled without allocation.
  Replacement `11760373` passed preflight but was canceled after
  `00:13:40`, before target capture, once the state mismatch was proven.
  Its native-operand conclusion is rejected.
- Default-off, fail-closed RELION perturbation override commits `57c0082`
  and `6982c77` require exact iteration/value input and print the applied
  value at full precision.  Build `11761492` completed `0:0`; binary SHA-256
  is
  `c761b5660cfd84e4960f95f62b01fb23bccbbb9caba8fe388b80e383acd00a74`.
- Same-model A100 control `11761710` replayed exact uninterrupted
  perturbation `+0.27053284645080566`.  Its RELION command completed and
  sealed all captures; the wrapper later exited `2:0` only because it used
  an obsolete fine-score-validator CLI.  Hash-pinned recovery passes, and
  the native translation grid matches within
  `3.725290298461914e-09` pixels.
- Transposed native matrices form a bitwise-exact bijection over all 2,968
  RECOVAR rotations.  Native row 1210 maps to target RECOVAR row 2626;
  the previously compared native row 2626 maps to RECOVAR row 210.  After
  mapping, support is exact at 109,184/109,184 (Jaccard 1.0), winner
  `(2626, 80)` is exact, maximum-tie key sets match, and translations 80/82
  are bitwise tied in native raw-diff2 with bitwise-exact cross-engine target
  scores.  Background combined-score maximum absolute difference is
  `0.0001220703125`.
- Superseding same-model classification is
  `authoritative_state_native_and_recovar_target_match_after_exact_rotation_permutation__prior_native_operand_boundary_rejected`.
  Comparison and recovery-seal SHA-256 values are
  `9b7f7020160fd38ec25f6be7d06e08b3ed06f061fed7c8417ae9e6dc2b28e39f`
  and
  `a163ab2f9abd6ba1e83d97506065d9f54692c65b67d0cd927d8fcea3f43c7f08`.
  Root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_native_same_model_control53722_6982c77_20260729T114500ET`.
  Exact-UUID confirmation replacement `11762553` remains pending, so this
  control is non-scoring and not exact-device authoritative evidence.
- This is non-scoring; fixed K=4 remains `41/60` direct and `9/15`
  all-class.  No correlation was computed, grid correction was unset, and
  forced final all-data was unset.

## 2026-07-29 case-22 shellwise map-amplitude factorial closes causality

- Same-A100 science `11748501` ran both arms at source `dd1eb519` on UUID
  `GPU-83a2fe0e-5ca2-bfe7-65cd-fdf081753bf8`.  Both arms and both
  analyzers completed; each arm recorded exit status zero.  Slurm reports
  `FAILED 1:0` after `03:17:37` only because the launcher subsequently hit
  the already documented malformed telemetry regex.  Arm wall times are
  5,938 and 5,723 seconds, and both capture manifests validate.
- Shell-scaling RECOVAR maps toward RELION restores exact coarse support:
  8/8, Jaccard 1.0.  Half-map scale ranges are
  `[0.986042384, 0.996838563]` and
  `[0.986044505, 0.997335876]`; relative L2 falls from
  `0.0116260905/0.0116277740` to `0.00272557521/0.00271663428`.
  Report SHA-256 is
  `590785315d330bc6c8ff30a88f17ee3d0d07f97bcc386eabbb2e9bfd1a8badd2`.
- The reciprocal RELION-to-RECOVAR shell scaling produces 7/8 support,
  Jaccard 0.875, with the single RELION parent 10538 absent.  Its scale
  ranges are `[1.00316035, 1.01415136]` and
  `[1.00286905, 1.0141491]`.  Cross-prior replay does not restore the
  parent.  Report SHA-256 is
  `6004b3856fb4f0408d7c19cec7b61e53594ac7041ed0ce00d7d143e2c6602927`.
- Accepted target classification is
  `shellwise_map_amplitude_is_sufficient_to_transfer_target_support`.
  This is a one-target causal result, not authorization for a generic
  production correction.
- Recovery `11750619` failed closed after the diagnostic checkout advanced
  from `dd1eb519` to `72f21482`.  Local recovery passed against clean
  detached checkout
  `/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_case22_map_factorial_recovery_dd1eb519_20260729`
  at the immutable science commit.  All seven completion-manifest entries
  replay exactly.  Summary and manifest SHA-256 values are
  `6a83f2b1d3c8e0ba459b81636d822fe5e673dfa10327bb1eff0e974c01a4811a`
  and
  `06f7cd4859186af5efcb7d00c67ba31e1c5187fc6aec4e697612b0008b1dbc2a`.
  Duplicate recovery `11757584` was canceled pending after local success.
- Run/runtime roots are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it3_map_amplitude_shell_factorial_dd1eb519_20260729T062927ET`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case22_it3_map_amplitude_shell_factorial_dd1eb519_20260729T062927ET`;
  both contain `SAFE_TO_DELETE`.  Grid correction and forced final all-data
  were unset; no correlation was computed.
- This remains non-scoring.  Fixed K=1 is `28/34` strict, `32/34`
  topology, and `34/34` evaluated; K=4 is `41/60` direct and `9/15`
  all-class.

## 2026-07-29 unified fixed scorecard checkpoint

- `scripts/summarize_em_relion_parity_scorecard.py` now pins, validates, and
  renders the accepted K=4 trajectory snapshot alongside the existing K=1
  fixed suite.  K=4 is displayed as 15 checked iteration rows with
  per-row class-pass counts.
- Denominators, thresholds, and evidence are unchanged: K=1 is `28/34`
  strict, `32/34` topology, `34/34` evaluated; K=4 is `41/60` direct and
  `9/15` all-class.  Snapshot SHA-256 is
  `bc10d0555488b22f0bc8d54afe5afc5288064ddb4708bd1c75f3b55dd4c0060a`.
- Freshness, compile, checkout/JAX provenance, scoped Ruff,
  `git diff --check`, and 23/23 focused scorecard/K=4 tests pass.  Runtime
  root
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/k4_scorecard_unification_3d39a434_20260729T122500ET`
  contains `SAFE_TO_DELETE`.
- Exact-UUID native confirmation `11762553` remains resource-pending on
  `della-l07g2`, with predicted start `2026-07-30T11:21:32`.  No score or
  production behavior changed.

## 2026-07-29 K=1 map-amplitude trajectory rejects a generic correction

- `scripts/analyze_em_k1_map_amplitude_trajectory.py` now reproduces the
  state-swap amplitude boundary from saved maps: RECOVAR
  `it(N-1)_halfH_reg.mrc` is paired with RELION
  `run_itN_halfH_class001.mrc`, using the same positive least-squares factors
  on rounded unshifted Fourier shells.  It records input hashes, normalized
  L2 before/after scaling, and shell factors without using correlation.
- Slurm audit `11763032` completed `0:0` in `00:00:48` on A100 UUID
  `GPU-de47e784-b81f-4a55-bb6f-099142193ae3` at source `78ffa37d`.
  It compared references 1--3 for current failing case 04, current failing
  case 22, and passing case 24.
- Case 22 alone develops a material reference-2 amplitude difference.
  Its half-map global RECOVAR-to-RELION factors are
  `0.988980770/0.988963664`; shell medians are
  `0.992036998/0.991904587`.  Shell scaling reduces relative L2 from
  `0.011916629/0.011924214` to `0.002662961/0.002651342`, explaining
  `77.6534%/77.7651%`.
- Failing case 04 is not an amplitude analogue: reference-2 global factors
  are `0.999984324/0.999984622`, and shell scaling explains only
  `0.9139%/1.1405%` of already-small `0.000130738/0.000117392` residuals.
  Passing case 24 remains near its MRC round-trip floor through reference 3
  (`0.78e-6` to `1.45e-6` relative L2, global factors within
  `5.96e-7` of one).
- At case-22 reference 3, shell scaling explains only `1.9971%/2.0950%`,
  so the later difference is no longer predominantly amplitude-only.  The
  bounded classification is
  `case22_reference2_amplitude_bias_is_case_specific__generic_map_rescaling_rejected`.
  This supports localizing the case-22 reference-2 reconstruction state,
  not applying an unconditional production rescale.
- Output JSON SHA-256 is
  `5de29b7b315c91402d87b9f7f67789627c78324344e34976574f615210f68c9c`;
  launcher SHA-256 is
  `aa76fa0e5b688fad2072f552b9579431fbe6198add812f507a07e95009dae306`.
  Run and runtime roots are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_map_amplitude_trajectory_78ffa37d_20260729T124500ET`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_map_amplitude_trajectory_78ffa37d_20260729T124500ET`;
  both contain `SAFE_TO_DELETE`.  Submission `11763030` was canceled
  without allocation after its expanded commit gate was found incorrect;
  the corrected launcher was rehashed before `11763032`.
- This is non-scoring.  Fixed K=1 remains `28/34` strict, `32/34`
  topology, and `34/34` evaluated; fixed K=4 remains `41/60` direct and
  `9/15` all-class.  Grid correction and forced final all-data were unset.

## 2026-07-29 case-22 reference-2 tau2 substitution is rejected

- The existing intermediate-state auditor passes over the complete sealed
  trajectory and shows that reference-2 RECOVAR tau2 differs from RELION
  half-1 tau2 by relative L2 `0.0250963025`, while gold-standard FSC differs
  by only `4.6824490e-5`.  That made tau2 a plausible explanation for the
  reference-2 map-amplitude bias, but not a causal result.
- `scripts/analyze_em_k1_tau2_substitution.py` holds each saved RECOVAR
  numerator/weight accumulator fixed and replays the numbered Wiener solve
  with either the stored RECOVAR tau2 or the corresponding RELION half tau2.
  FSC/FSC-AUC is primary; normalized L2 and shell-amplitude fits are
  secondary; correlation is absent.  A stored-tau replay must pass
  FSC-AUC `>=0.99999` and relative L2 `<=0.001` against the saved RECOVAR
  map before substitution is interpreted.
- Hash-pinned Slurm job `11763790` completed `0:0` in `00:00:20` on A100
  UUID `GPU-6b5da455-0f76-eeaa-6041-ec8df42a2e8a` at source `53bf0ac0`.
  Stored-tau replay FSC-AUC is `0.999995624/0.999994780` for halves 1/2,
  so both integrity gates pass.
- RELION tau2 raises cross-engine FSC-AUC from
  `0.999985231` to `0.999992977` in half 1 and from
  `0.999991860` to `0.999993414` in half 2, but it leaves the amplitude
  residual essentially intact.  Relative L2 changes only
  `0.011884866 -> 0.011884164` and
  `0.011985030 -> 0.011891107`, explaining
  `0.005899%` and `0.783669%`, respectively, versus the predeclared 50%
  causal gate.
- The accepted classification is
  `relion_tau2_rejected_as_map_residual_cause`.  Together with the
  reference-2 support-count differences on `1101/3000` particles, this
  moves the next discriminator upstream to the saved BPref numerator/weight
  source and its contributor membership, not a per-half tau2 default or a
  generic map rescale.
- Output and launcher SHA-256 values are
  `8c190d438deed4b7f68ab3b4db5b55d19512670d34696f662ace40f9ad51467f`
  and
  `346d35587dfefccb6b28897f9ea8b848636b5687c777369dd9bc01f0bfea25df`.
  Run/runtime roots are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_tau2_substitution_53bf0ac0_20260729T130000ET`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_tau2_substitution_53bf0ac0_20260729T130000ET`;
  both contain `SAFE_TO_DELETE`.  Grid correction and forced final all-data
  were unset.
- This is non-scoring.  Fixed K=1 remains `28/34` strict, `32/34`
  topology, and `34/34` evaluated; fixed K=4 remains `41/60` direct and
  `9/15` all-class.

## 2026-07-29 case-22 physical-iteration-2 BPref accumulator is causal

- `scripts/analyze_em_k1_bpref_boundary.py` compares RECOVAR's versioned
  post-join accumulator against RELION's passive downsampled average/FSC
  dumps, with an independent RELION-repeat map gate.  FSC/FSC-AUC is primary;
  complex normalized L2, positive amplitude fits, weight, and support are
  secondary.  Correlation is absent.
- Same-A100 job `11764048` completed both RELION and RECOVAR two-iteration
  science arms on `della-l08g5`, UUID
  `GPU-a20700a1-ed8d-42b4-3a83-38d3a8d7e57b`.  Slurm reports
  `FAILED 1:0` after `01:07:10`, maximum RSS `25670888K`, only because the
  wrapper expected `run_it001_half{1,2}_class001.mrc` after a run without
  `save_intermediates_dir`.  RECOVAR had already saved the corresponding
  non-converged numbered maps as `final_half{1,2}.mrc`, both versioned BPref
  dumps, results, and timing.  Final all-data did not run; grid correction
  and forced after-max finalization were unset.
- The diagnostic RELION repeat qualifies against the prior exact trajectory:
  half-map FSC-AUC is `0.9999998648859472/0.9999998500826959`.
  RECOVAR-versus-repeat maps are already biased at reference 2, with
  FSC-AUC `0.9999931029101212/0.9999934355897221` and normalized L2
  `0.011627993100749703/0.011629974090418277`.
- Post-join RECOVAR-versus-RELION complex-average FSC-AUC is
  `0.9999961774540641/0.9999962019771084`; normalized L2 is
  `0.01281965364812333/0.012812025855309038`.  Positive global scales
  `0.9886418590267985/0.9886450238112716` explain
  `55.6266381%/55.6834080%`; weight relative L2 is
  `0.001932232251382977/0.001953730974117599`, and support Jaccard is
  exactly `1.0`.
- `scripts/analyze_em_k1_bpref_substitution.py` decodes RELION's raw
  `[k,i,j>=0]` double-precision BPref storage, converts it to RECOVAR's
  centered `[j,i,k]` cube with the pinned `N^2/N^4` unit conversion, and
  completes the negative half by Hermitian symmetry.  The independently
  dumped RELION average self-replays at
  `9.3259147436e-17/9.3361588114e-17` relative L2, with exactly zero weight
  error.  The discrete real-map frame requires an explicit reported `-1`
  sign; no amplitude is fitted for replay integrity.
- The causal factorial uses identical RELION tau2 in both reconstruction
  arms.  RECOVAR accumulators retain FSC-AUC
  `0.9999929843572196/0.9999933517302643` and L2
  `0.0115961094347795/0.011597638611648597`.  RELION accumulator
  substitution reaches FSC-AUC
  `0.9999999999934455/0.9999999999904421` and L2
  `6.548196000620455e-7/1.912523539572997e-6`, explaining
  `99.9943531%/99.9835094%` of the map residual.
- The accepted classification is
  `relion_bpref_accumulator_explains_majority_of_map_residual`.  Together
  with the rejected tau2 substitution, this establishes that the
  reference-2 amplitude bias is carried by numerator/weight content.  It
  does not yet separate candidate membership, posterior mass, or scatter
  arithmetic; the physical-iteration-2 support-count differences on
  `1101/3000` particles remain the next bounded upstream discriminator.
- Run/runtime roots are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_bpref_boundary_8004e667_20260729T125500ET`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case22_it2_bpref_boundary_8004e667_20260729T125500ET`;
  both contain `SAFE_TO_DELETE`.  Boundary, substitution, recovery-output
  manifest, and recovery-record SHA-256 values are
  `dbbee34d5409338f1fc05a6b30e55f7355cbfd3f7699da46b3d6de9169a4a48f`,
  `d02c15b54db82fe05ab552f628a98380ee11d077434665b1dc38c07829535675`,
  `3ad13d220b6730634fb195bbca93c5fcc3f43ed6620f65e8fb87696b621b5b7f`,
  and
  `390bef7c3041d7af2af6971e1dc9573f83b3b1e7af7548beac0ff14a6837bc7d`.
- This is non-scoring.  Fixed K=1 remains `28/34` strict (`82.4%`),
  `32/34` topology, and `34/34` evaluated; fixed K=4 remains `41/60`
  direct and `9/15` all-class.  Exact-UUID K=4 confirmation `11762553`
  remains resource-pending.

## 2026-07-29 case-22 membership and raw coarse-score classification

- Fixed membership denominator: candidate sets `54/64`, positive-rotation
  sets `30/64`, significant-sample counts `5/64`, reconstruction-mass gate
  `64/64`.  Strict compact particle pass is `4/64`.
- The 10 candidate-mismatch particles contain 13 complete adaptive-parent
  differences: 11 RELION-only groups / 88 fine rotations and 2
  RECOVAR-only groups / 16 fine rotations.  Membership and parent reports
  have SHA-256 values
  `09b4cb69e585d2d0907541e407e386b3fe695d69206331d4714a3e79a46bbdc1`
  and
  `95080a4596dc1cc9bdf4a92f4aad398dd46ab9b8b06953eeea748a761d400460`.
- Passive RELION coarse pass-1 capture validates 14/14 artifacts,
  14,966,784 candidates, 463,933 significant samples, and exact
  `768 x 48 x 29` topology.  The zero CUDA `op.significant_weight` is a
  sentinel; validation additionally proves the saved mask is the exact
  monotone top-weight set inferred from its minimum selected weight.
- Original `11775061` completed RELION and the 14 artifacts, then failed
  only on the obsolete literal-cutoff validator.  Exact-UUID recovery
  `11775556` completed `0:0` in `00:50:46` on
  `GPU-6b5da455-0f76-eeaa-6041-ec8df42a2e8a`, validated RECOVAR `14/14`,
  remained non-converged after iteration 2, and skipped final all-data.
- Capture inertness passes all `6/6` fixed FSC-AUC comparisons.  Minimum
  FSC-AUC is `0.9999999998255802` versus the fixed `0.999999` threshold.
  Inertness, RELION-validation, completion, and output-manifest SHA-256
  values are
  `8589d908efea92cf2166a19750e7d49758541b11c3291080f568c046177201e5`,
  `b79c80d1aa1de880688fee0cbfa1ad44a6fcb50c48877bc10e7876eb69081a83`,
  `32ef903d336ab4b917bdc232532d746b3b98e9f0a71bbf6f596b4d40ef1556a1`,
  and
  `92a07769a416f808a020218b46520d6d73773ea2a09b3a15ec43e26895a759f8`.
- Commit `489827a4` fixes the analysis-only parent identity conversion:
  cohort IDs are RELION direction-major keys, whereas compared surfaces
  are already RECOVAR psi-major.  Output retains both the RELION key and
  canonical row.  Focused validator/analyzer tests pass `14/14`.
- Fixed boundary metric: expected side `13/13`, exact prior support
  `13/13`, raw target-parent arithmetic `12/13`, with-prior arithmetic
  `11/13`, mismatch posterior-TV `10/10`, control exact parent sets `4/4`.
  One target raw parent exceeds centered-score p95 `1e-4`; the other 12
  membership flips remain below that raw gate.  Classification:
  `candidate_parent_difference_originates_in_raw_coarse_scores`.
- Boundary JSON SHA-256 is
  `ecd032e6768f8788238439abb3e61e12bbef90e3f55fca397bac57e6c1d85ed6`;
  analyzer SHA-256 is
  `08a3417495f7c0959b092b34e8276cc31706723e5153dd0c7c5cb368710ecfd3`.
  No correlation is used.
- Run/runtime roots are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_coarse_pass1_0506806c_20260729T212000ET`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case22_it2_coarse_pass1_recovery_473cfca6_20260729T211000ET`;
  both contain `SAFE_TO_DELETE`.
- Exact-device K=4 replacement `11774193` failed closed `42:0` in eight
  seconds because its two allocated A100s omitted required UUID
  `GPU-5e619c2e-82b4-ff79-cbcb-ab29514a9f30`; no science ran.  Retry only
  after allocation state changes.
- Non-scoring: K=1 remains `28/34` strict, `32/34` topology, `34/34`
  evaluated; K=4 remains `41/60` direct and `9/15` all-class.

## 2026-07-30 case-22 inverse-noise shell partition

- The fixed threshold-`0.01` shell intervention has dominance vector
  `14,0,14,14,0` for actual RELION, RECOVAR-all, RELION inverse noise on all
  shells, RELION inverse noise on shells 1--4, and RELION inverse noise on
  shells 5+.
- Median centered-energy removal is `85.2128%`, `-0.1184%`, `85.1714%`,
  `85.1685%`, and `-0.1286%`, respectively.  Valid-pixel fraction is
  `0.9911019849`--`1.0`, median `0.9958932238`.
- The bound model STAR serializes scored shells 1--4 with six fixed decimal
  places and crosses to scientific notation at shell 5.  RELION source
  computes `1 / sigma2_noise[ires_remapped]`, copies it into the float
  `corr_img`, then applies CTF-squared and scale-squared.
- Classification:
  `inverse_noise_residual_is_confined_to_star_fixed_decimal_shells_1_through_4`.
  Report/completion SHA-256 values are
  `9d6b8cf39c9abe21c71d5c3d0dc0ef73b381566b439328748d92c32efa473073`
  and
  `ee851c8c1f19cc625f0de6b4869da9337fd75d117bc51aeebaecfa13d0f113ed`.
  The focused CPU gate passes 62/62 tests.
- Exact-device Slurm job `11785170` restarts from the serialized
  `run_it000_optimiser.star`, captures iteration-2 preprocessing and coarse
  operands, and writes under
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_serialized_it000_restart_e9ce2357_20260730T0500ET`.
  It was pending for resources at submission.  No existing process or job
  was modified.
- Non-scoring: K=1 remains `28/34` strict, `32/34` topology, `34/34`
  evaluated; K=4 remains `41/60` direct and `9/15` all-class.

## 2026-07-30 case-22 corr_img conditioning audit

- Commit parent `d096e436` localizes the fixed 14-particle `corr_img`
  factorial to inverse noise, but that inference divides by effective CTF
  squared.  The conditioning audit freezes thresholds `0`, `0.001`, `0.003`,
  and `0.01` and substitutes factors only where both implementations exceed
  the threshold.  It keeps actual RELION correction in excluded pixels.
- At every threshold, actual RELION and RECOVAR-CTF-scale-only are dominant
  `14/14`; RECOVAR-inverse-noise-only and both-factor arms are `0/14`.
  At threshold `0.01`, valid pixel fraction is
  `0.9911019849`--`1.0`, median `0.9958932238`.
- Classification is
  `inverse_noise_attribution_is_stable_above_fixed_effective_ctf_thresholds`.
  No scale, sign, correlation, or fitted threshold is used.  Report and
  completion SHA-256 values are
  `00336d64aabf166082860c6d62721128a6eddbd782dd99da742161e2d1234e12`
  and
  `4ca7fd037b49569e1a2ea87d19595ab922145643352191d31d89975780b8259e`.
- Run/runtime roots are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_corr_img_conditioning_final_d096e436_20260730T0435ET`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case22_corr_img_conditioning_final_d096e436_20260730T0435ET`;
  both contain `SAFE_TO_DELETE`.
- Non-scoring: K=1 remains `28/34` strict, `32/34` topology, `34/34`
  evaluated; K=4 remains `41/60` direct and `9/15` all-class.

## 2026-07-30 case-22 corr_img inverse-noise localization

- Exact iteration-1 half-1 model/data STAR files are hash-bound.  Stack,
  model-group, scale, half-set, and all parent artifact identities fail
  closed.  Cohort scale corrections are exactly 1.0 for 14/14.
- Actual RELION is dominant 14/14 (median energy removal `85.2128%`).
  RECOVAR CTF-times-scale squared only remains dominant 14/14
  (`85.1697%`).
- RECOVAR inverse noise only is dominant 0/14 (median `+0.0490%`); both
  RECOVAR factors are dominant 0/14 (median `-0.1187%`).
- Effective CTF-times-scale relative-L2 min/median/max is
  `1.1315e-7` / `2.0487e-7` / `4.1429e-7`; inverse-noise relative L2 is
  `1.3689e-6` / `1.8371e-6` / `5.7002e-5`.
- Classification:
  `raw_coarse_residual_is_inverse_noise_weight_dominated_not_ctf_scale_squared`.
  Next compare RELION `local_Minvsigma2` against RECOVAR noise-shell
  expansion and float precision/order.
- Report/completion SHA-256 values are
  `f4d2ccff415e9187d0cf79f7a9afff041175335d006b8a3228b3ad5c9014ae31`
  and
  `2d22cc68790985f73eac470399575d1913b06f6ba46b57bb197b004ebaf43e29`;
  analyzer/test SHA-256 values are
  `e709937ad3882a0a50469234023219db2ab5eb668eea9b85c72c616b34e1b2c5`
  and
  `01d7f7f89b17a8eb40f3a570d3dc05537123e67a6129e776a3da4dcc1db6e955`.
- Run root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_corr_img_factorial_3c75802d_20260730T041547ET`;
  runtime root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case22_corr_img_factorial_3c75802d_20260730T041547ET`.
  Both contain `SAFE_TO_DELETE`; short CPU analysis, no Slurm job.
- Expanded focused CPU gate: 118 passed, 10 GPU-only skipped.  Scoped Ruff,
  scorecard freshness, and `git diff --check` pass.
- Non-scoring: K=1 remains `28/34` strict, `32/34` topology, `34/34`
  evaluated; K=4 remains `41/60` direct and `9/15` all-class.

## 2026-07-30 case-22 score-transfer 2x2 factorial

- The predeclared arms are actual RELION, RECOVAR pixel correction only,
  RECOVAR `corr_img` only, and both RECOVAR operands.  The fixed denominator
  is 14 and the strict-majority threshold remains greater than 0.5.
- Actual RELION is dominant 14/14, median energy removal `85.2128%`.
  RECOVAR pixel correction only remains dominant 14/14, median `85.2310%`.
- RECOVAR `corr_img` only is dominant 0/14, median `-0.1187%`; replacing
  both operands is dominant 0/14, median `-0.0384%`.
- Median base relative L2 is `1.2231e-6`, `1.4537e-6`, `4.8813e-7`, and
  `3.1160e-7` for those four arms, respectively.
- Classification:
  `raw_coarse_residual_is_corr_img_score_weight_dominated_not_pixel_correction`.
  The next split is within `buildCorrImage`: `Minvsigma2`, CTF-squared, and
  scale-squared values/order.
- Report/completion SHA-256 values are
  `a64967d5a860e929ba37c65773d513f49ef59bedce9f92457ee14a9ccee7c7f4`
  and
  `ecbf32f4c37260bb261547bffad8e8c1728b508085c48ca8b15a56c13a26a1ad`;
  analyzer/test SHA-256 values are
  `6b978df78ff0819cb83c9c3a4981efe87b5433e19e8d40931ed940e23f2baeee`
  and
  `a30515cc377db3dc810d14a51523aee209454fd07f48df0fac38aac3e520bbea`.
- Run root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_score_transfer_factorial_f896a838_20260730T040552ET`;
  runtime root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case22_score_transfer_factorial_f896a838_20260730T040552ET`.
  Both contain `SAFE_TO_DELETE`; short local CPU analysis, no Slurm job.
- Expanded focused CPU gate: 112 passed, 10 GPU-only skipped.  Scoped Ruff,
  scorecard freshness, import provenance, and `git diff --check` pass.
- Non-scoring: K=1 remains `28/34` strict, `32/34` topology, `34/34`
  evaluated; K=4 remains `41/60` direct and `9/15` all-class.

## 2026-07-30 case-22 post-optics score-transfer localization

- Joined the qualified preprocessing capture, qualified production-operand
  capture, and RECOVAR score-component capture on the frozen 14-particle
  cohort.  No new RELION patch or science rerun was required.
- The fixed analyzer uses scale-sensitive relative L2 and centered
  residual-energy removal only.  It fits no scale/sign and computes no
  correlation.
- The actual RELION live weighted base is strict-majority dominant 14/14;
  energy removal is `63.6834%`--`92.9979%`, median `85.2128%`.
- Applying RECOVAR's CTF/noise score transfer to the same captured RELION
  post-optics image is dominant 0/14; energy removal is
  `-0.4150%`--`+1.6616%`, median `-0.0384%`.
- The hybrid base passes the fixed `1e-6` material gate 14/14.  Hybrid
  min/median/max relative L2 is
  `2.2121e-7` / `3.1160e-7` / `5.6403e-7`; actual RELION live-base
  min/median/max is
  `3.5885e-7` / `1.2231e-6` / `2.5990e-6`.
- Classification:
  `raw_coarse_residual_is_postoptics_score_weight_transfer_dominated_not_preprocessing`.
  The next bounded factorial is inside the paired pixel-correction /
  `buildCorrImage` transfer.
- Authoritative run root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_postoptics_score_transfer_e33642f1_20260730T035421ET`;
  runtime root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case22_postoptics_score_transfer_e33642f1_20260730T035421ET`.
  Both contain `SAFE_TO_DELETE`.  The analysis was a short local CPU replay
  over sealed exact-device artifacts, so there is no new Slurm job ID.
- Report/completion SHA-256 values are
  `2fb1f36603f9d52a8d47db010dc5492c56051c1debc5e48799373142b86fdd80`
  and
  `fec8a8438be0f494fec7b94ac6962e658fbf5eabff314ff368e012b73f7cd0bc`;
  analyzer/test SHA-256 values are
  `29977484d0c516933d0311ec6a28e427fadd22c8e121c65fc266fa092ef8191d`
  and
  `558c8eac5b7361251c9de7ae350fabf4ceb4aa013113b8ed88f042df9c7ddf0c`.
- Focused CPU gate: 107 passed, 10 GPU-only skipped.  Scoped Ruff,
  scorecard freshness, import provenance, and `git diff --check` pass.
- Non-scoring: K=1 remains `28/34` strict, `32/34` topology, `34/34`
  evaluated; K=4 remains `41/60` direct and `9/15` all-class.

## 2026-07-30 case-22 preprocessing-boundary qualification

- The live-operand factorial makes the base corrected image the next causal
  boundary: 14/14 particles are strict-majority dominated by that operand,
  while translation phase is 0/14.  Existing captures start after
  preprocessing, so a production change is not yet justified.
- Frozen additive patch 0006 captures seven passive boundaries for the same
  explicit cohort: raw input; normalized/rounded-shifted real; unmasked
  Fourier before/after optics; masked real; and masked Fourier before/after
  optics.  It reads no score, weight, reference, model, or map buffer.
- The environment contract requires a non-empty explicit particle-ID list,
  physical iteration, maximum particles per rank, expected followers, and
  total byte cap.  Unsupported 3D, tomo, multibody, helical, random-mask, or
  separate-reconstruction-image modes fail closed.  Existing artifacts are
  never overwritten.
- Patch SHA-256 is
  `a655a40e561167d1b39f1157d3ac3754751ac87e06448b3b5133bbca799517b4`.
  Forward and reverse application checks pass.  The CUDA 12.6/OpenMPI 4.1.6
  incremental build passes; patched source/binary SHA-256 values are
  `6513f8a0dab566544b44ff117e0017dfbb5df2a466a4280df022a1d60ed92d7d`
  and
  `982c15cfcdce94823c471228edef47839fa7d239ccac166c4ba66c829cd1f6ba`.
- The fail-closed Python schema validator and static passive-patch guards pass
  within a 76/76 focused capture/parity preflight gate.  Scoped Ruff, import
  provenance, and `git diff --check` pass.
- Runtime gate root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_relion_preprocess_capture_gate_20260730T030000ET`;
  it contains `SAFE_TO_DELETE`.  The external build root remains
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_k1_case22_operand_capture_build_ac12ca86_20260729T234000ET`.
- Exact-device science job `11783432` completed `0:0` in `00:14:15` on
  `della-l07g3`, using required UUID
  `GPU-6b5da455-0f76-eeaa-6041-ec8df42a2e8a`.  The validator passes 14/14
  complete seven-stage captures, exact physical iteration 2, real shape
  `1x128x128`, Fourier shape `1x60x31`, and zero temporary artifacts.
  Completion and validator SHA-256 values are
  `ede90fa9953ed2b1e9c82395371ec04736d08c41f58232b88f24d6f6177a709e`
  and
  `a77a08eb8da9289f4694d18941286d2a8500c7611a53c993926a027138eda0c1`.
- The unchanged-map gate passes 3/3: half-1, half-2, and merged non-DC
  FSC-AUC are `0.9999999999801346`, `0.9999999999797552`, and
  `0.9999999999616417`; minimum non-DC FSC is above `0.99999999960`.
  Inertness report SHA-256 is
  `2fd093ccc2aa4b0ef3a4f94b094e92aca720dd46108b1acce12cadd3bd31400d`.
- Dependent analyzer job `11783563` completed `0:0` in seven seconds on the
  same node and physical GPU.  The fixed 14/14 denominator has zero material
  gaps at normalization, unmasked FFT, masking, masked FFT, and post-optics
  under scale-sensitive relative-L2 threshold `5e-7`.  Raw disk,
  normalized real, and unmasked Fourier data are bitwise equal 14/14.
  Masked-real relative-L2 is `5.9075e-10`--`1.5918e-8`; masked-Fourier
  relative-L2 is `2.2456e-8`--`1.1845e-7`.  Optics correction is bitwise
  inert 14/14.
- Three strict replays keep normalized real bitwise exact 14/14.
  Masked-real replay maxima are `1.3361e-8`, `1.5444e-8`, and
  `1.3361e-8`; the non-bitwise atomic-reduction floor remains below the
  fixed material threshold.  No fitted scale, sign, or correlation is used.
- Classification is
  `all_preprocessing_boundaries_within_fixed_material_threshold`.  Analysis
  report/completion SHA-256 values are
  `e04fb43bdea0790d049284970ac9b50d9608e9e8ea1dfa659552cd09c00cdea2`
  and
  `5787d13d706959404aea2c01fba12319f5d6a84b88eefbfa43c7d1da2e1a2b80`.
  Analyzer/test SHA-256 values are
  `4a45b7546887e803dfeb7c315b66493f50ac9520f6985377db48aa99e41c7333`
  and
  `e8d4f64fb43fea898cb33e6a6f06b445bdd653afe270d0b58ab4e3b9b38e059e`.
- Analyzer current-size Fourier mapping is bitwise equal to RELION's
  `windowFourierTransform` for 14 synthetic 128-to-60 spectra; the exact
  binding test passes 9/9 and the frozen closing capture/parity gate passes
  83/83.  The external binding and build-log SHA-256 values are
  `b1ff4e99217665b56f96a59ade957a4cbf87b978ddefe7237ad08200858e995d`
  and
  `8a06d4c92717a94527470fae8bb843e7eb7fd044894d15a9779b30896eda88f2`.
- Science/analysis roots are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_preprocess_boundary_9521fbac_20260730T033000ET`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_preprocess_analysis_9521fbac_20260730T025500ET`;
  runtime roots with the same run IDs live under
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/`.  All four roots
  contain `SAFE_TO_DELETE`.
- Exact same-device evidence rules out stack loading, normalization, rounded
  shifts, zero-mask application, unmasked/masked current-size FFT
  layout/units, and optics correction as a material source of the case-22
  raw coarse residual.  Continue downstream of post-optics `op.Fimg` at
  current-size score-input/pixel correction, `buildCorrImage`, or the
  live-factorial conversion/interaction.  No production change is proposed.
- Non-scoring: K=1 remains `28/34` strict, `32/34` topology, `34/34`
  evaluated; K=4 remains `41/60` direct and `9/15` all-class.

## 2026-07-29 case-22 captured norm/cross qualification

- RELION job `11777114` completed the full exact-device science and sealed
  14/14 schema-v2 artifacts, but its recovered fixed validator rejected the
  diagnostic arrays: replay p95 `13/14`, replay maximum `14/14`, and
  reference-norm translation invariance `1/14`.  The 3/3 half/merged
  shellwise FSC-AUC inertness gate passes with minimum above
  `0.99999999996`.
- The rejected capture is map-safe but not classification-ready.  The
  reference norm varies by `9.5367e-7`--`7.6294e-6` across translation
  threads because the passive float32 component reduction follows different
  pixel partitions.  No gate was relaxed.
- RECOVAR job `11777337` completed `0:0` in `00:54:10`, with peak RSS
  `28234296K`, and sealed all 14 fixed component dumps.  Replay passes both
  fixed gates `14/14`; map inertness passes `3/3` with half-1, half-2, and
  merged FSC-AUC `0.9999999998350682`, `0.9999999998308815`, and
  `0.9999999998928133`.  Provisional analysis reports cross-engine closure
  `14/14`, cross-term dominance `14/14`, reference-norm dominance `0/14`,
  and cross-residual rotation dominance `14/14`.  A positive scale fitted
  independently per rotation removes a majority of cross-residual energy
  for only `2/14`, rejecting simple per-rotation amplitude as a cohort-wide
  explanation.
  The fail-closed classification remains `component_capture_not_qualified`
  because its RELION input is rejected.  Output SHA-256 is
  `694bb6d14e842e232a374a2f151f2f15fa40f95b971cd9a09125b50efe78bcf1`.
- A default-off FP64 component accumulator was built while leaving production
  float32 diff2 unchanged.  Binary SHA-256 is
  `9ffb9eee44254f5c17664595f0247f71cfba2d95ce449745ba70af2f8ac64f9d`;
  dependency job `11778245` failed preflight `1:0` in two seconds because the
  launcher referenced a nonexistent Python.  It wrote zero capture and zero
  RELION artifacts.  The failure is sealed in
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_diff2_components_fp64_1487550f_20260729T231000ET/provenance/PREFLIGHT_FAILURE_11778245.md`.
  Corrected exact-device job `11779197` completed `0:0` in `00:19:04`, with
  peak RSS `3003528K`, on required UUID
  `GPU-6b5da455-0f76-eeaa-6041-ec8df42a2e8a`.
- Job `11779197` improves reference-norm translation invariance from `1/14`
  to `14/14` and passes replay maximum `14/14`, but its unchanged replay-p95
  gate passes only `12/14`.  `part1480_stack2330` and
  `part2277_stack348` each measure `6.103515625e-5` against the fixed `5e-5`
  gate.  It is formally rejected.  Its half-1, half-2, and merged map
  inertness passes `3/3` at FSC-AUC `0.9999999999800995`,
  `0.9999999999799074`, and `0.9999999999605521`.
- The paired float32/FP64 audit reports zero bitwise-identical production
  raw-score particles, but centered inter-run p95 differences at most one
  image-constant float32 ULP for `14/14`; FP64 replay p95 values are integral
  ULPs for `14/14`.  The FP64 diagnostic separately expands norm and cross
  terms while production diff2 retains the original float32
  squared-difference path, so its classification is
  `fp64_capture_rejected_expanded_component_arithmetic_does_not_replay_production_float32_diff2`.
  No gate is relaxed and no correlation is computed.  Script and output JSON
  SHA-256 values are
  `221206e203acfecef2f98fabeda174394a116cbda710b41b7ae0e858464546ff`
  and `bcf4fd5fffa7ecac660128c48f90bde87668004504601004035ef95385f635f0`.
- The additive live-operand patch is frozen at SHA-256
  `a00ad73ac496be4b2cc0513ee7aa2fd0dd8de137927db66b4d80420d0b06ad1e`
  and its tested CUDA 12.6/OpenMPI 4.1.6 binary at
  `c54597d3e8ee23181f50bfcb510c83645e48ac76794e49a54b9abecd0959449a`.
  Science was withheld because the component prerequisite is rejected.
- Schema-v2 patch 0004 captures the exact GPU `translatePixel` outputs and
  CUDA coarse topology so the validator can replay the original production
  squared-difference arithmetic directly.  It does not depend on algebraic
  norm/cross closure for qualification, while retaining independent
  reference/cross replay checks.  Production p95/max gates remain
  `5e-5`/`5e-4`.  Patch and rebuilt binary SHA-256 values are
  `3d090744381306bdccc3be641834909286355f2bc15abc707053ad48d95f3b21`
  and `2e415f5c982773bdf4e33bf4d44933cc6307fd8f096b5a7e58b73e97318d54f8`.
  Exact-device job `11780231` sealed all 14/14 component and 14/14 operand
  artifacts plus the complete iteration-2 RELION output, then failed the
  operand validator.  Its passive projected-reference kernel used the correct
  device Euler array, but its metadata loop serialized an unsynchronised
  host-side buffer containing finite garbage up to `2.8283851e38`.
- Recovered half-1, half-2, and merged map inertness for job `11780231`
  passes 3/3 at FSC-AUC `0.9999999999798187`, `0.9999999999793953`, and
  `0.9999999999599425`.  An unaffected-operand audit also found the validator
  must preserve RELION's CUDA guard
  `tid / translation_num < block_size / translation_num`.  With that exact
  guard, centered production-diff2 replay passes p95 and maximum 14/14 at
  no more than `3.0517578125e-5`, below the unchanged gates.  This remains
  diagnostic because the Euler payload is invalid.
- Patch 0005 copies each nine-value Euler matrix directly from device memory.
  Patch and corrected binary SHA-256 values are
  `c7a27cb9467103b4cea840ce7a36c9bfd11ad1a46263f824d2baa5d04d8f5e0c`
  and `ce07fc71246d382e4630a3e36dc41004f2e29cc07dd834155efa4ecfc5da9374`.
  Seven schema tests pass.
- Clean exact-device job `11781751` completed `0:0` in `00:14:28`, peak RSS
  `2973352K`, on required UUID
  `GPU-6b5da455-0f76-eeaa-6041-ec8df42a2e8a`.  It seals all 14/14
  component and operand artifacts with no temporaries.  Operand reference,
  cross p95/max, and centered direct-diff2 p95/max gates pass 14/14.  The
  largest production replay p95/max error is `3.0517578125e-5`.
- Half-1, half-2, and merged capture inertness passes 3/3 at FSC-AUC
  `0.9999999999799831`, `0.9999999999798452`, and
  `0.9999999999613778`.  The direct operand capture is qualified; the
  independent algebraic component parent remains rejected 12/14 and is not
  used to qualify it.
- The first live-reference counterfactual output is superseded because it
  omitted the fixed RELION-to-RECOVAR projection conversion.  Direct closure
  identifies `-(128**2) = -16384`; after conversion the captured references
  match RECOVAR references to relative L2 `3.9095116e-6`.
- The corrected projected-reference-only intervention removes
  `-7.0922%` to `+2.6216%` of centered residual energy, median `-2.4833%`.
  Zero of 14 particles exceed the unchanged strict-majority threshold.
  Classification is
  `live_projected_reference_rejected_as_raw_coarse_residual_cause`.
  Report/analyzer SHA-256 values are
  `1e6d9524cca750b7d2dd25ed2566dc5b0eeff0ac3ba8498fba49c093edd1c408`
  and `28a857030ed6a0130e670b8ce7ec3ea3ecf7847eb7a22faf49b76b585cb18a95`.
  No correlation is computed; this remains non-scoring.
- The fixed 2^3 live-operand factorial identifies the shifted image as
  strict-majority dominant for 14/14 particles (median centered-energy
  removal `85.2109%`); correction alone is 0/14, and every pairwise arm
  containing the shifted image plus the all-live arm is 14/14.  Decomposing
  the shifted input identifies the base corrected image as 14/14 (median
  `85.2128%`) and translation phase as 0/14 (median `-0.0804%`).
  RELION/RECOVAR base-image relative L2 is `3.5885e-7` to `2.5990e-6`,
  while recovered-base variation across translations is only `1.2846e-7`
  to `1.5597e-7`.  Classification is
  `raw_coarse_residual_is_live_base_corrected_image_dominated_not_reference_correction_or_translation_phase`.
  Factorial report/analyzer SHA-256 values are
  `d33970bdec969c0eba9d26c705c61ef4308b6b4f3eb65cb89f184dc849a71a32`
  and `7aea14db989088281109f954e8a992b549e03e799c41270be48dad2c98fc66be`.
  The frozen focused parity gate passes 94/94 tests; scoped Ruff, scorecard
  freshness, import provenance, and `git diff --check` pass.  The first
  freshness invocation used the nonexistent `--check-freshness` spelling;
  the repository-supported `--check docs/math/em_relion_parity_scorecard.md`
  invocation passes.  This is a non-scoring causal boundary, not yet a
  production fix.
- Non-scoring: K=1 remains `28/34` strict, `32/34` topology, `34/34`
  evaluated; K=4 remains `41/60` direct and `9/15` all-class.

## 2026-07-29 case-22 raw-score component decomposition

- Commit `206fadf3` adds a fixed two-way additive energy decomposition over
  exact common finite-prior translation rows.  Rotation dominance is defined
  as a rotation-only centered-energy fraction strictly above `0.5`; no
  correlation is used.
- Rotation-only residual energy dominates `14/14` particles, `10/10`
  mismatches, and `4/4` controls.  Its range is
  `0.6942752958`--`0.8778484909`.
- Translation-only energy is `0.0031623918`--`0.0472093274`; interaction
  energy is `0.1189891173`--`0.2585153768`.
- Classification:
  `raw_coarse_score_residual_is_translation_independent_rotation_dominated`.
  This rejects translation-only offset/grid effects as the majority residual
  but does not yet distinguish projection norm from a rotation-dependent
  cross term.
- Output JSON SHA-256 is
  `a829fcdae2945534a901aff08024e3d650b98fc83e20a5f2995ecddee7e3e025`;
  analyzer SHA-256 is
  `cdda611ee61714fe4493b7a103475a4fb1ee78ba2a85be833567258e3fd9a40c`.
- Next bounded capture: split coarse raw diff2 into projection-norm and
  image/reference cross-term components on the same 14-particle cohort.
- Non-scoring: K=1 remains `28/34` strict, `32/34` topology, `34/34`
  evaluated; K=4 remains `41/60` direct and `9/15` all-class.

## 2026-07-30 case-26 fine source and causal-chain closure

- Exact H100 capture job `11788465` completed `0:0` in `00:29:23` on
  `della-h21g1`, UUID
  `GPU-aafe72bb-2a51-7aa0-076f-51a017cacc7d`.  The source commit was
  `34770d8241f13c27922baff22138624fa9119e3f`; grid correction and forced
  after-max finalization were unset.  Run and runtime roots are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_it3_target206_capture_34770d82_20260730T0840ET`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_case26_it3_target206_capture_34770d82_20260730T0840ET`;
  both contain `SAFE_TO_DELETE`.
- The fixed target is RECOVAR original index `206`, RELION stack index `207`,
  physical iteration `3`, current size `66`.  RELION's top key is `(36,54)`
  and RECOVAR's is `(38,55)`.  Candidate support and rotation identities are
  exact and the passive capture-inertness gate passes `3/3`.
- The `2^3` projection/shifted-image/correction operand factorial closes both
  production margins within `5e-5`.  Swapping only the projected reference
  changes the first-minus-second margin from approximately `+0.0034` to
  `-0.00135`; projection-norm change dominates the cross-term change.  The
  accepted classification is
  `fine_winner_flip_is_projected_reference_determined`.  Operand report
  SHA-256 is
  `060b844629caed30b80efa9172f08d3224478aed2412144bcf6919fbb9d4d7e5`.
- Exact RELION map-to-PPref reconstruction and RECOVAR production CUDA
  texture replay close at relative L2 below `1.2e-8`; frozen RELION PPref
  projects bitwise exactly to the captured RELION references.  The live
  cross-engine projection relative L2 is `0.00248934748`, more than
  `225,000x` the replay floor.  RELION iteration-2 half 1 versus RECOVAR
  iteration-1 half 1 map FSC-AUC is `0.999999687660`.  The accepted
  classification is
  `fine_projection_difference_is_iteration_start_map_state`; texture
  coordinates/interpolation and both serialized map-to-PPref paths are
  closed.  Source-boundary report SHA-256 is
  `b3911e287982db224d3f9206add0b2d99832504e48bf896d82702d1c1b13d771`.
- Commit `168a2cbe` adds a hash-pinned temporal gate over the fixed FSC
  trajectory, particle state, operand/source reports, and x-half precision
  factorial.  Physical iteration `1` has merged cross-engine FSC-AUC
  `0.999999999968`, zero support mismatches, and zero hard pose outliers.
  Iteration `2` has FSC-AUC `0.999999735312`, `87/1000` support-count
  mismatches (each at most one candidate), Pmax maximum absolute error
  `3.5221e-6`, and zero angular/translation outliers above
  `0.01 degree`/`0.01 Angstrom`.  Iteration `3` has FSC-AUC
  `0.999999155258`, `165/1000` support-count mismatches, two angular
  outliers, and two translation outliers.  The fixed onset order is therefore
  map `1` -> support `2` -> hard pose `3`.
- The temporal classification is
  `iteration_map_divergence_precedes_support_then_hard_pose_divergence__fine_path_inherits_iteration_start_map_state`.
  This does not justify a fine-projection interpolation change.  The matched
  x-half double-precision intervention remains rejected: it introduces
  `3` numbered failures versus `0` for control and changes final cross-engine
  FSC-AUC by `-0.0812706848`.
- The causal-chain report is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_it3_target206_capture_34770d82_20260730T0840ET/analysis/case26_causal_chain_v1.json`,
  SHA-256
  `d960b3171caaa5b493f0ac20f864f33b7801f958e9e07501a752dc4120c7bbb2`.
  The focused gate passes `53/53`; scoped Ruff, Python compilation,
  `git diff --check`, and Wheels run `30548434124` are recorded separately.
- Non-scoring: K=1 remains `28/34` strict, `32/34` topology, and `34/34`
  evaluated; K=4 remains `41/60` direct and `9/15` all-class.

## 2026-07-30 case-10 final full-grid FSC-deficit localization

- `scripts/analyze_em_k1_final_grid_fsc_deficit.py` replays the trajectory
  auditor's signed, normalized non-DC FSC-AUC exactly and partitions its
  defect by trapezoid segments at the last numbered reconstruction radius.
  It fits no scale/sign and computes no correlation.
- Case 10's last numbered iteration has current size `68`, hence radius `34`;
  the final expectation spans shells through `190`.  Final half-1, half-2,
  and merged FSC-AUC values replay exactly as `0.985843446538`,
  `0.985323765470`, and `0.983006503534`.
- The fraction of final FSC-AUC defect beyond radius `34` is
  `95.1891%`, `95.1018%`, and `95.8232%` for half 1, half 2, and merged.
  Relative to numbered iteration 15, the total defect is amplified
  `278.796x`, `270.634x`, and `518.523x`.  All products pass the frozen
  gates of strictly more than `95%` outside-radius defect and more than
  `250x` amplification.
- Restricted final FSC-AUC through radius `34` is
  `0.996099410176`, `0.995882846490`, and `0.995934905111`, so both halves
  and merged pass the unchanged `0.995` parity gate inside the numbered
  radius.  The complementary radius-34-to-190 values are
  `0.983673915769`, `0.983090113715`, and `0.980271649354`, and all fail.
- Classification:
  `final_full_grid_fsc_deficit_is_over_95pct_outside_last_numbered_radius`.
  This quantitatively localizes the failed final boundary to frequencies
  introduced beyond the converged numbered radius; it does not support a
  final pose-writeback, scheduler, or grid-correction change.
- The short CPU analysis root is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case10_final_grid_deficit_d5a947f8_20260730T0955ET`;
  it contains `SAFE_TO_DELETE`.  The v2 report SHA-256 is
  `7fc2ad34d2e7e7f865b53a3e645a85a44548110ab5d5e7c05fb3e5d4ca96d0ff`.
  Focused validation passes `14/14`; scoped Ruff and `git diff --check`
  pass.
- Non-scoring: K=1 remains `28/34` strict, `32/34` topology, and `34/34`
  evaluated; K=4 remains `41/60` direct and `9/15` all-class.

## 2026-07-30 K=4 exact-device target-score attribution

- Authoritative native science job `11787017` completed `0:0` in
  `00:10:50` on `della-l07g2`.  The allocation contains the required A100
  UUID `GPU-5e619c2e-82b4-ff79-cbcb-ab29514a9f30`; the completion seal
  SHA-256 is
  `d2685dd64f9f04a1748735952a8f8e8900c5fa1ba6dab6a8934a2178e66beab2`.
  Grid correction and forced after-max finalization were unset.
- Bitwise rotation permutation is bijective over all `2,968` rows.  Native
  and RECOVAR support is exact at `109,184/109,184`, Jaccard `1.0`; the
  winning key, two-way maximum tie set, and target translation tie at IDs
  `80` and `82` are exact.  This closes the decision topology.
- The target orientation and translation prior operands are bitwise exact in
  both engines.  The remaining target combined-score offset is
  `3.0517578125e-5` RECOVAR minus RELION.  The v2 analyzer decomposes it
  exactly, with zero residual: `1.8596649169921875e-5` is the shared
  data-then-prior path contribution from the unequal pre-prior values and
  `1.1920928955078125e-5` is RELION's production float32 operation-order
  contribution.
- Both captured RELION totals replay bitwise from
  `((orientation_prior + translation_prior) + min_diff2) - raw_diff2`.
  Both RECOVAR totals replay bitwise after converting the dumped pre-prior
  residual to float32 and adding the two priors.  Classification:
  `exact_device_target_absolute_score_offset_is_preprior_plus_float32_order_and_decision_inert`.
- The first dependent CPU audit, job `11787139`, failed `1:0` before
  analysis because its launcher invoked the analyzer by file path and could
  not import the `scripts` package.  It produced no report.  The repaired
  hash-pinned module-form audit, job `11790393`, completed `0:0` in
  `00:00:04` on `della-h12n17`.
- The official v3 report is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_native_authoritative_perturb53722_6982c77_20260729T113000ET/analysis/EXACT_DEVICE_NATIVE_SCORE_AUDIT_V3.json`,
  SHA-256
  `d7eea0c74e16ecc5139ab57ff12658b746b18ef89a8efb3704ca2cdd5bd0a75c`.
  Commit `d766e693` adds the target attribution gate; commit `f24c45fd`
  adds the full-table decomposition.  Focused tests pass `19/19`, and
  scoped Ruff, Python compilation, scorecard freshness, and
  `git diff --check` pass.
- Across all `109,184` active candidates, the telescoping decomposition
  closes with exactly zero residual.  Pre-prior/data-path differences account
  for `79.6643880%` of component L1, RELION float32 operation order for
  `19.6706644%`, RECOVAR dump replay residual for `0.4727976%`, orientation
  prior for `0.1447619%`, and translation prior for `0.0473881%`.
  Classification:
  `global_absolute_score_residual_is_preprior_data_path_dominated_with_exact_telescoping_closure`.
- The global bitwise score table is still not exact (`100,852/109,184`
  combined-score mismatches, maximum absolute difference
  `1.52587890625e-4`), so this is a non-scoring target-boundary closure, not
  a parity-score promotion or a production-kernel change.
- Non-scoring: K=1 remains `28/34` strict, `32/34` topology, and `34/34`
  evaluated; K=4 remains `41/60` direct and `9/15` all-class.

## 2026-07-30 K=4 authoritative pre-prior representative

- The v5 exact-device analyzer adds a deterministic identity for the maximum
  absolute pre-prior/data-path component: select the maximum absolute
  float32 component, then the lowest native candidate index on an exact tie.
  Candidate index, native rotation, mapped RECOVAR rotation, and translation
  are validated as aligned arrays; candidate indices must be unique.
- Replaying the sealed job-`11787017` inputs selects native candidate `587`,
  native rotation `24`, mapped RECOVAR rotation `1072`, and translation
  `103`.  Its RECOVAR-minus-native pre-prior component is
  `+1.64031982421875e-4`, from native `-19.790740966796875` to RECOVAR
  `-19.790576934814453`.
- The v5 report preserves exact support at `109,184/109,184`, the exact
  winner and maximum-tie topology, the zero-residual telescoping
  classification, `100,852/109,184` combined-score mismatches, and maximum
  combined-score absolute difference `1.52587890625e-4`.
- V3/V4 replay exposed CPU reduction-order drift below `3e-22` in
  `np.linalg.norm` relative-L2 fields.  V5 replaces only those diagnostic
  reductions with `math.fsum` over float64 squares in fixed C order.
  Complete replays with BLAS/OpenMP thread counts `1` and `8` are
  byte-identical.  All non-reduction report fields match v4 exactly; the
  three relative-L2 changes are below `3e-22`.
- The two byte-identical non-scoring reports are
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_native_representative_v5_bedc7761_20260730T1518ET/analysis/EXACT_DEVICE_NATIVE_SCORE_AUDIT_V5_THREADS1.json`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_native_representative_v5_bedc7761_20260730T1518ET/analysis/EXACT_DEVICE_NATIVE_SCORE_AUDIT_V5_THREADS8.json`,
  SHA-256
  `2a8c1ebd2798a3e567528f221888f0f8b5be691775cf4b0a205fb502ea55db24`.
  The run and runtime roots contain `SAFE_TO_DELETE`.  Focused validation
  passes `60/60`; scoped Ruff, Python compilation, scorecard freshness, and
  `git diff --check` pass.
- V6 adds a threshold-free decision-context summary without changing any
  score or classification.  The maximum pre-prior component candidate `587`
  is ranked `94,016/109,184` in both engines, lies `19.7204285` native and
  `19.7203064` RECOVAR log-score units below the captured-class maximum,
  and carries normalized within-class score mass
  `6.2955431e-11` and `6.2963506e-11`, respectively.
- Across the complete aligned captured-class table, float64 normalization of
  the captured float32 combined scores gives total variation
  `6.808398793631863e-6` and maximum absolute mass delta
  `3.956962159670785e-7`.  That maximum occurs at native candidate `46,671`,
  native rotation `1210`, mapped RECOVAR rotation `2626`, and translation
  `83`.  The scope is explicitly one captured class, not a complete K=4
  posterior, map-quality gate, or scorecard metric.
- V6 thread-count replays are byte-identical at
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_native_score_mass_v6_efef644a_20260730T1530ET/analysis/EXACT_DEVICE_NATIVE_SCORE_AUDIT_V6_THREADS1.json`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_native_score_mass_v6_efef644a_20260730T1530ET/analysis/EXACT_DEVICE_NATIVE_SCORE_AUDIT_V6_THREADS8.json`,
  SHA-256
  `50d1f1f76a323e0e7f88cbabeadb6d6265d3cbe0bf6749b1d45803f92cfcea78`.
  The broader K=4/native-operand/frozen-scorecard gate passes `99/99`.
- This localizes the next raw-operand comparison without choosing a
  production fix or changing a fixed denominator.  K=1 remains `28/34`
  strict (`82.4%`), `32/34` topology, and `34/34` evaluated; K=4 remains
  `41/60` direct (`68.3%`) and `9/15` all-class.

## 2026-07-30 case-22 live-versus-serialized noise-state boundary

- The qualified shell-partition parent already establishes that physical
  iteration-2 raw coarse-score residuals are inverse-noise dominated and
  confined to shells 1--4.  Its fixed 14-particle report SHA-256 remains
  `9d6b8cf39c9abe21c71d5c3d0dc0ef73b381566b439328748d92c32efa473073`.
- `scripts/analyze_em_k1_noise_serialization_boundary.py` recovers the
  effective shell noise independently from the sealed RELION live
  `corr_img`/effective-CTF operands and RECOVAR `ctf2_data`/effective-CTF
  operands.  It binds the exact model STAR tokens, keeps the fixed
  effective-CTF threshold `0.01`, evaluates all 14 particles, and fits no
  scale, sign, threshold, or correlation.
- On fixed-decimal STAR shells 1--4, RECOVAR's maximum absolute difference
  from the serialized token is at most `2.4296e-10`.  Live RELION's minimum
  absolute difference is `3.5756e-7`, `6.3893e-8`, `3.0616e-7`, and
  `4.0229e-7`, respectively.  The minimum live-versus-serialized closeness
  advantage is `419.81x`; the other shells reach `1,303x`, `1,472x`, and
  `16,593x`.
- Scientific-notation shell 5 is the fixed control.  RECOVAR and live RELION
  both close to the serialized token within `2.8211e-11` and `2.4431e-10`,
  respectively.  Maximum within-shell variation is below `3.545e-9` across
  every engine/shell product, ruling out a poorly conditioned pixel subset.
- Classification:
  `recovar_score_weight_matches_serialized_star_noise_while_live_relion_retains_pre_serialization_shells_1_to_4`.
  This identifies a live-versus-serialized state boundary, not a RECOVAR
  production-kernel fix.  The next causal gate remains the already queued
  exact-device RELION restart from the same serialized model STAR:
  primary `11785170`, iteration-0 retry `11785428`, and direct iteration-1
  restart `11785547`, with robust/pair audits `11791339`--`11791341` and
  `11791711`--`11791712`.
- The deterministic report is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_noise_serialization_boundary_313803ae_20260730T1430ET/analysis/NOISE_SERIALIZATION_BOUNDARY_V1.json`,
  SHA-256
  `07791270eec260bce623a39a6520cfb76f3a504db2ee40c4d7dc17f07fc09818`.
  Focused validation passes `81/81` CPU tests with 10 GPU-only skips; scoped
  Ruff, Python compilation, scorecard freshness, deterministic byte replay,
  and `git diff --check` pass.
- Non-scoring: K=1 remains `28/34` strict (`82.4%`), `32/34` topology, and
  `34/34` evaluated; K=4 remains `41/60` direct and `9/15` all-class.

## 2026-07-30 K=4 normalized-score-mass strata

- V7 partitions the V6 within-captured-class candidate-level total variation
  by exact mapped RECOVAR rotation and translation identity.  Each stratum
  uses `math.fsum` over the aligned candidate order, retains absolute
  candidate deltas without within-stratum cancellation, and ranks by
  descending TV contribution then ascending integer identity.  No threshold,
  fit, correlation, or scorecard gate is introduced.
- The rotation partition closes the full
  `6.808398793631863e-6` total variation with exactly zero replay residual
  across `2,968` groups.  Mapped RECOVAR rotation `2626` (native rotation
  `1210`) is rank 1: its 80 candidates contribute
  `1.1236180279902226e-6`, or `16.50341089069551%`, and its marginal
  normalized mass shifts by `+5.042772359288761e-7`.
- The translation partition also closes with zero replay residual across 104
  groups.  Translation `82` ranks first at
  `6.813629772775691e-7` (`10.007683126829646%`), translation `83` ranks
  second at `5.840084557443876e-7` (`8.577765102282661%`), and translation
  `80` ranks third at `5.798000211833489e-7`
  (`8.515952704263687%`).  Their marginal normalized-mass shifts are
  `-3.929162811263973e-8`, `+4.197402946491178e-7`, and
  `-3.289001381136056e-7`, respectively.
- The selected rotation is the fixed K=4 target rotation; translations
  `80,82` are already covered by the queued immutable raw/operand capture.
  Translation `83` contains the largest individual mass delta but remains a
  predeclared follow-up only after the official `80,82` pair qualifies.
  No additional science job was submitted.
- Thread-count-1 and thread-count-8 reports are byte-identical at
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_native_score_mass_strata_v7_9fefefec_20260730T1610ET/analysis/EXACT_DEVICE_NATIVE_SCORE_AUDIT_V7_THREADS1.json`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_native_score_mass_strata_v7_9fefefec_20260730T1610ET/analysis/EXACT_DEVICE_NATIVE_SCORE_AUDIT_V7_THREADS8.json`,
  SHA-256
  `b58e1848842f97178f8dd535ee8dd484eeef4d959d4a7d3794fa12d08f5e7373`.
  Removing only the V7 schema and strata field makes the report exactly equal
  to V6.
- This remains a descriptive one-class diagnostic, not a complete K=4
  posterior or FSC/FSC-AUC claim.  Fixed metrics remain K=1 `28/34` strict,
  `32/34` topology, and `34/34` evaluated; K=4 `41/60` direct and `9/15`
  all-class.

## 2026-07-30 active K=4 Slurm validity audit

- `scontrol write batch_script` reproduces exact submitted launcher SHA-256
  values for raw jobs `11790517`, `11793813`, and `11796622` and operand jobs
  `11790787`, `11793814`, and `11796623`.  The current launcher files retain
  those hashes, and the pinned RELION binary retains SHA-256
  `c761b5660cfd84e4960f95f62b01fb23bccbbb9caba8fe388b80e383acd00a74`.
- Filtering the immutable submission environments shows all three raw jobs
  carry their required `EXPECTED_LAUNCHER_SHA256`.  Operand job `11790787`
  carries both its exact launcher hash and the required binary hash, so it is
  potentially admissible after its exact-GPU and empty-output-owner gates.
- Operand jobs `11793814` and `11796623` carry only their launcher hashes.
  Both submitted scripts use `set -euo pipefail` and require
  `EXPECTED_BINARY_SHA256` at line 18.  They must therefore fail closed before
  source, binary, GPU, import, or science operations and cannot produce
  admissible evidence.
- The only potentially admissible dependent pair audits are `11795302`
  (`11790517 + 11790787`), `11795304`
  (`11793813 + 11790787`), and `11799807`
  (`11796622 + 11790787`).  Audits `11795303`, `11795305`, `11796769`, and
  `11799808` depend on an operand job that cannot complete successfully.
- All science jobs still had zero elapsed time and every intended output root
  was empty at `2026-07-30T16:19:49-04:00`.  Three valid raw routes and one
  valid operand owner already cover the fixed target, so no additional
  science job was submitted.
- The read-only audit is
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_active_job_validity_b8303926_20260730T1620ET/provenance/ACTIVE_K4_JOB_VALIDITY.md`,
  SHA-256
  `3e36504a797a226404259443bede9f671e04907ba0977af72fad39baf8acf7da`.
  No process or job was killed, signalled, suspended, cancelled,
  reprioritized, or otherwise altered.

## 2026-07-30 K=4 marginal score-mass survival

- V8 answers the predeclared decision-relevance question using the complete
  V7 mapped-rotation and translation partitions.  Within each stratum it sums
  signed RECOVAR-minus-native normalized-mass deltas with `math.fsum`; it then
  computes one-half the fixed-order sum of absolute stratum deltas.  No
  threshold, fit, correlation, or pass/fail classification is introduced.
- The unchanged candidate-level total variation is
  `6.808398793631863e-6`.  Across all 2,968 rotation groups, marginal TV is
  `1.8554921714932912e-6`: `27.25298895870787%` survives and
  `72.74701104129214%` cancels within rotations.
- Across all 104 translation groups, marginal TV is
  `2.056175817468767e-6`: `30.20057842957115%` survives and
  `69.79942157042885%` cancels within translations.  Translation marginal TV
  exceeds rotation by `2.0068364597547566e-7`, or only
  `10.815655762857054%` relative to the rotation marginal.
- Most candidate-level mass movement therefore cancels under either
  marginal.  The modest translation excess supports completing the already
  queued target-translation operand comparison but does not identify a
  production cause, justify another science job, or replace a full K=4
  posterior/FSC-AUC gate.
- Thread-count-1 and thread-count-8 reports are byte-identical at
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_native_marginal_score_mass_v8_ff7dff79_20260730T1635ET/analysis/EXACT_DEVICE_NATIVE_SCORE_AUDIT_V8_THREADS1.json`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_native_marginal_score_mass_v8_ff7dff79_20260730T1635ET/analysis/EXACT_DEVICE_NATIVE_SCORE_AUDIT_V8_THREADS8.json`,
  SHA-256
  `e9f7178a9280f9e7010dc7f7d92d883b8000ce1dc0e5e22f8dd49b89a73fda39`.
  Removing only the V8 schema and ten new marginal/cancellation fields
  reproduces V7 exactly.
- This remains non-scoring: K=1 is `28/34` strict, `32/34` topology, and
  `34/34` evaluated; K=4 is `41/60` direct and `9/15` all-class.

## 2026-07-30 K=4 exact-device raw/operand owner-pair result

- Raw job `11790517` completed on the pinned A100.  Its class-1 pass-2
  capture SHA-256 is
  `ccbdc9040da463f479784e3ad270fd76bb5817006742f43c96f9b053bf9d6eef`;
  all `109,184` saved RECOVAR score rows replay bitwise.
- Native operand retry `11812925` completed after the missing pinned
  `7a7ea9ba` source worktree was restored.  Operand SHA-256 is
  `93322e2b98ca11e626f178007f39cf8d6137655fdffd5239907cd2321459270f`.
  Target translations `80,82` both reproduce native production raw cost
  `501.4734191894531` bitwise.
- Dependency-bound audit job `11812941` accepts only that raw/operand pair.
  Its fixed causal-boundary metric is `2/4` passed, `4/4` evaluated:
  native target operand replay and fixed target raw diff2 pass; global raw
  diff2 and global combined score fail.
- Classification:
  `global_raw_and_score_paths_differ_but_fixed_target_closes`.  Completion:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_owner_pair_raw11790517_operand11812925_9959cc8a_20260730T2042ET/provenance/ANY_OWNER_PAIR_AUDIT_COMPLETE.json`,
  SHA-256
  `963e9b6b315368ae9a8201b73624163129f92e5626fff4089ad8fe3ce6516552`.
- Declared gates reproduce exactly locally and under Slurm.  Do not claim
  whole-report byte identity: three secondary raw-report CPU norm reductions
  differ only in their final printed digits.
- Complete active-table raw mismatch is `25,877/109,184`, maximum absolute
  delta `0.0001220703125`; common minima differ by one float32 bit.
- This is non-scoring.  Fixed metrics remain K=1 `28/34` strict,
  `32/34` topology, and `34/34` evaluated; K=4 `41/60` direct and `9/15`
  all-class.

## 2026-07-30 K=4 raw-mismatch strata V17

- Predeclared raw delta:
  `float64(recovar_float32) - float64(native_float32)`.  Partition bitwise
  mismatches by mapped rotation and translation; require exact `math.fsum`
  replay; rank by descending raw-delta L1 then ascending identity.
- Predeclared representative: leading rotation, then largest absolute delta,
  then lowest native candidate index.  This candidate is the sole next
  operand target.
- Predeclaration:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_raw_mismatch_strata_v17_ed49174d_20260730T2055ET/provenance/PREDECLARED_DIAGNOSTIC.md`,
  SHA-256
  `7630a6919bfae22fe167f284fc5c018460c811af2da5247a4c0dc1eb84f0e51d`.
- Thread-count-1 and thread-count-8 reports are byte-identical, SHA-256
  `950c90703340a0e68e5181efe7c2fcb0985f75af22e93744f07864ff56523100`.
  Canonical stratification SHA-256 is
  `9399373da53b64221dc515b2fecfcff04d36207c2c1e2901dbbaf335dfdbb5e8`.
- Both partitions replay signed delta `+0.708465576171875` and raw-delta L1
  `1.282928466796875` exactly.
- Leading rotation is mapped RECOVAR `954` / native `1738`: `39/96`
  mismatches and raw-delta L1 `0.00152587890625`, only
  `0.11893717738290636%` of the global L1.  Rotation top-ten concentration is
  `1.0609196222555246%`.
- Translation `80` ranks first at `2.8521135136420944%`; translation top-ten
  concentration is `24.09191465068151%`.
- Predeclared representative: native candidate `66561`, RECOVAR rotation
  `954` / native `1738`, translation `13`; RELION raw
  `516.3260498046875`, RECOVAR raw `516.3261108398438`, positive one-ULP
  delta `0.00006103515625`.
- Next bounded step: capture and compare that representative's exact native
  and RECOVAR operands.  V17 does not authorize a production fix or
  scorecard change; paired causal metric remains `2/4`, and fixed K1/K4
  metrics remain unchanged.

## 2026-07-30 K=4 translation-marginal owner predeclaration

- V10 will partition score-mass deltas by mapped rotation within translations
  `78,83,76,80,82`, rank complete rotation-owner components, quantify
  top-1/top-3/top-10 concentration and cancellation, and report target
  rotation `2626` (native `1210`) explicitly.
- Each translation's signed owner sum and marginal-TV contribution must
  reproduce V9 exactly; mapped/native owner identities must be one-to-one.
- The result cannot change the frozen scorecard, establish a full K=4
  posterior or FSC/FSC-AUC result, authorize a fix, or authorize another job.
- Predeclaration:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_translation_rotation_owners_v10_103287c8_20260730T170538ET/provenance/PREDECLARED_DIAGNOSTIC.md`.

## 2026-07-30 K=4 translation-marginal rotation owners

- V10's rotation-owner sums reproduce V9 exactly for all five selected
  translations, with zero residual and no missing translation or target
  rotation.
- Fixed rotation row `2626` (native `1210`) ranks first for V9 leaders
  `78`, `83`, and `76`, carrying `33.259304651582394%`,
  `33.877610167708705%`, and `44.25147352892708%` of their respective
  within-translation rotation-component TV.  Each signed component aligns
  with the corresponding net marginal.
- The same rotation ranks second for queued `80` (`12.324818327390429%`) and
  third for queued `82` (`10.487699164186956%`), but both signed components
  oppose their negative net translation marginals.
- Rotation cancellation is `45.719471498772635%`, `64.0638512575893%`,
  `57.59548740051366%`, `71.63676042626443%`, and
  `97.11668894385543%` for translations `78,83,76,80,82`, respectively.
  This explains why target `82` is candidate-level rank 1 yet marginal rank
  21.
- The queued pair therefore probes the same dominant rotation family, but
  cannot by itself establish the leading marginal cause.  No new job is
  authorized before a current official pair qualifies.
- Thread-count-1 and thread-count-8 reports are byte-identical, SHA-256
  `89519c9121107a284da09b1bc95a46f422c74068d07c0b25b5fe7181afba7aed`.
  Removing only the V10 schema and owner report reproduces V9 exactly.
- Fixed metrics remain K=1 `28/34` strict, `32/34` topology, and `34/34`
  evaluated; K=4 `41/60` direct and `9/15` all-class.

## 2026-07-30 K=4 normalized-component owner predeclaration

- V16 partitions each of V15's five normalized-mass component paths by all
  `2,968` mapped rotations and all `104` translations.
- Every partition flattened in ascending identity and aligned-candidate order
  must replay candidate-level L1 and signed sum exactly; separately summed
  rounded group summaries report their deterministic residual.  Ranking is
  descending group L1 then ascending identity, with separate marginal path-TV
  ranking and top-1/top-3/top-10 concentration.
- Fixed selections are target rotation `2626`, translations
  `78,83,76,80,82`, and queued pair `80,82`.
- The question is whether the already queued exact-device targets are leading
  owners of V15's normalized pre-prior and operation-order paths.
- Thread-count-1 and thread-count-8 reports must be byte-identical; removing
  only V16 must reproduce V15 exactly.
- This threshold-free one-class diagnostic cannot identify a raw operand,
  authorize a fix or new science job, change a scorecard, establish a full
  K=4 posterior, or establish FSC/FSC-AUC parity.
- Predeclaration:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_normalized_component_owners_v16_2a3ea6eb_20260730T192200ET/provenance/PREDECLARED_DIAGNOSTIC.md`.

## 2026-07-30 K=4 normalized-component owner result

- V16 reproduces all ten component/identity L1 totals exactly; flattened
  signed/L1 replay is exact and the maximum rounded group-summary signed
  residual is `9.430725965891257e-23`.
- Target rotation `2626` / native `1210` ranks first for normalized pre-prior
  L1 (`15.914497159294227%`) and operation-order L1
  (`21.359635465703442%`).  It is also rank 1 for translation prior, rank 2
  for orientation prior, and rank 3 for dump replay.
- Across complete translation strata, translation `82` ranks first for
  pre-prior (`9.276317250945769%`) and operation order
  (`11.83464100928627%`).  Translation `80` ranks fourth
  (`7.620610932989319%`) and second (`9.837478302874318%`).
- Queued `80,82` jointly cover `16.89692818393509%` of pre-prior L1 and
  `21.67211931216059%` of operation-order L1 across all rotations.
- Rotation marginals retain `33.63192464592954%` of pre-prior path TV and
  `53.69220327612797%` of operation-order path TV; translation marginals
  retain `29.95038619965972%` and `16.11634049688868%`.
- The rankings support the existing exact-device target but do not enlarge
  its captured intersection: V15's target rotation plus queued translations
  still cover only `1.4207897957134247%` of whole-class pre-prior L1 and
  `5.961174442750557%` of operation-order L1.
- Thread-count-1 and thread-count-8 reports are byte-identical, SHA-256
  `f36653a8febc4c40094fb2df35cebeff1fde5a1ec51a021ccf2d99ca5ce48d18`.
  Removing only V16 reproduces V15 exactly.
- Fixed metrics remain K=1 `28/34` strict, `32/34` topology, and `34/34`
  evaluated; K=4 `41/60` direct and `9/15` all-class.

## 2026-07-30 K=4 whole-class normalized-mass predeclaration

- V15 independently normalizes all six frozen V14 score stages across the
  complete aligned `109,184`-candidate captured-class table.
- Consecutive normalized-mass differences telescope native operation order,
  pre-prior data path, orientation prior, translation prior, and dump replay
  into the final RECOVAR-minus-native mass delta.
- Candidate closure, exact replay of V6 L1/TV, component signed/L1/TV,
  component-L1 share, cross-component cancellation, and deterministic rank
  are fixed before evaluation.
- Target rotation `2626`, translations `78,83,76,80,82`, and queued pair
  `80,82` receive explicit component coverage.
- Thread-count-1 and thread-count-8 reports must be byte-identical; removing
  only V15 must reproduce V14 exactly.
- This threshold-free one-class diagnostic cannot identify a raw operand,
  authorize a fix or new science job, change a scorecard, establish a full
  K=4 posterior, or establish FSC/FSC-AUC parity.
- Predeclaration:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_normalized_mass_components_v15_cd5fb93f_20260730T185600ET/provenance/PREDECLARED_DIAGNOSTIC.md`.

## 2026-07-30 K=4 whole-class normalized-mass result

- V15 closes all `109,184` candidate telescopes exactly and exactly replays
  V6 normalized-mass L1 `1.3616797587263726e-5` and TV
  `6.808398793631863e-6`; component signed-sum residual is zero.
- Pre-prior remains rank 1 at `64.19279653823473%` of component L1 and
  native operation order ranks second at `34.95575434713458%`.
  Translation prior, orientation prior, and dump replay contribute
  `0.5773417612457405%`, `0.23042706980106985%`, and
  `0.0436802835838883%`.
- Total component L1 is `2.4180473788687152e-5`; cross-component
  cancellation before final normalized-mass L1 is
  `43.68680404585641%`.
- Target rotation `2626` covers `15.914497159294227%` of whole-class
  pre-prior L1 and `21.359635465703442%` of operation-order L1.  Fixed
  translations `78,83,76,80,82` cover `10.139516134320438%` and
  `11.843135526808614%`, respectively.
- Queued `80,82` cover `1.4207897957134247%` of whole-class pre-prior L1
  and `5.961174442750557%` of operation-order L1.  Both are
  operation-order dominated after independent whole-class normalization,
  reflecting global-normalizer cancellation rather than a raw-operand
  classification.
- Thread-count-1 and thread-count-8 reports are byte-identical, SHA-256
  `c9cb273b903b4f972807f0075a216e7984ccb9138c4eef4ecdd64bc92f1cbc48`.
  Removing only V15 reproduces V14 exactly.
- Fixed metrics remain K=1 `28/34` strict, `32/34` topology, and `34/34`
  evaluated; K=4 `41/60` direct and `9/15` all-class.

## 2026-07-30 K=4 target-rotation component predeclaration

- V11 selects rotation row `2626` / native `1210` at translations
  `78,83,76,80,82` and serializes all five existing telescoping score
  components, exact closure, component-L1 shares, scores, score masses, and
  prior bitwise equality.
- Components are ranked by descending absolute value then ascending name;
  missing candidates are explicit.
- The diagnostic cannot establish raw-operand cause, authorize a production
  fix or new job, change the scorecard, or establish full K=4 posterior or
  FSC/FSC-AUC parity.  The queued `80,82` operand audit remains causal.
- Predeclaration:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_target_rotation_component_v11_cd654d16_20260730T172015ET/provenance/PREDECLARED_DIAGNOSTIC.md`.

## 2026-07-30 K=4 target-rotation component result

- All five selected candidates are present.  Each telescoping closure residual
  and RECOVAR dump-replay residual is exactly zero.
- Orientation priors are bitwise identical for every candidate.  Translation
  priors differ bitwise for `78,76`, but their translation-prior component is
  zero after float32 addition, as it is for `83,80,82`; those raw differences
  are decision-inert at the serialized combined-score stage.
- Leader `83` has combined-score delta `+6.103515625e-5`: pre-prior data path
  `+4.9114227294921875e-5` (`80.46875%` of component L1) plus native
  operation order `+1.1920928955078125e-5` (`19.53125%`).
- Queued `80` and `82` each have combined-score delta
  `+3.0517578125e-5`: pre-prior data path
  `+1.8596649169921875e-5` (`60.9375%`) plus operation order
  `+1.1920928955078125e-5` (`39.0625%`).
- Leaders `78` and `76` each have zero local combined-score delta because
  pre-prior `-1.430511474609375e-5` exactly cancels operation order
  `+1.430511474609375e-5`.  Their negative normalized mass deltas therefore
  come from the global softmax normalization shift, not a local score
  mismatch at those candidates.
- V11 localizes the nonzero target-rotation mismatch for `83,80,82` to the
  pre-prior path with reinforcing native operation order.  It supports the
  queued raw-operand audit's scope but cannot identify a raw operand,
  authorize a fix or new job, or establish posterior/FSC-AUC parity.
- Thread-count-1 and thread-count-8 reports are byte-identical, SHA-256
  `e3a8237af66333b06e2da2430d0b5cb4ac8f8822ecd93c9a3c40247310eb8993`.
  Removing only the V11 schema and component report reproduces V10 exactly.
- Fixed metrics remain K=1 `28/34` strict, `32/34` topology, and `34/34`
  evaluated; K=4 `41/60` direct and `9/15` all-class.

## 2026-07-30 K=4 softmax partition-contribution predeclaration

- V12 uses one shared cross-engine maximum and scalar
  `math.exp(score - shared_reference)` weights to partition the native versus
  RECOVAR softmax-denominator shift by aligned candidate.
- Fixed-order sums replay the two partitions, signed delta, candidate
  contribution sum, absolute contribution total, and cancellation.
- Absolute contributions are ranked by descending magnitude then native
  candidate identity.  Top-1/top-3/top-10 concentration and fixed target
  rotation `2626` / native `1210` at translations `78,83,76,80,82` are
  explicit.
- The threshold-free question is whether the global normalization movement
  affecting zero-local-score candidates `78,76` comes primarily from the
  same selected target-rotation set or from other captured identities.
- This cannot identify a raw operand, authorize a fix or new job, change the
  scorecard, or establish full K=4 posterior/FSC-AUC parity.
- Predeclaration:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_partition_contributions_v12_d8b71453_20260730T173157ET/provenance/PREDECLARED_DIAGNOSTIC.md`.

## 2026-07-30 K=4 softmax partition-contribution result

- Candidate contributions replay partition delta
  `+0.00105344617606562` with residual
  `-4.5775015722338e-16`.  Absolute contribution L1 is
  `0.0011348867684725596`, signed cancellation is
  `7.176098503337935%`, and the log-partition ratio is
  `+2.432914160221955e-5`.
- Fixed target-rotation translations `80`, `82`, and `83` rank first, second,
  and third, with absolute shares `2.689000640086723%`,
  `2.689000640086723%`, and `2.5103701171896114%`.
- Fixed candidates `78` and `76` contribute exactly zero.  Their complete
  modeled log-normalized-mass ratio is therefore the global normalization
  term `-2.432914160221955e-5`.
- The selected five cover `7.888371397363057%` of absolute partition
  movement.  Top-1/top-3/top-10 concentration is
  `2.689000640086723%` / `7.888371397363057%` /
  `16.73773931344828%`; the aggregate shift is diffuse despite the selected
  targets owning the three strongest individual contributions.
- Seven of the top ten contributors use mapped rotation `2626`; three use
  rotation `947`.  The queued target family is prominent but the fixed set
  cannot alone explain the shared normalizer.
- Thread-count-1 and thread-count-8 reports are byte-identical, SHA-256
  `fbb54f53f4da3a4a6428fd988752b6716214a752ef356461cda2af3f703737ca`.
  Removing only the V12 schema and partition report reproduces V11 exactly.
- This remains descriptive: no production change or new job is authorized,
  and fixed K=1/K=4 scorecards are unchanged.

## 2026-07-30 K=4 partition rotation-family predeclaration

- V13 groups all V12 shared-reference candidate weight deltas by mapped
  RECOVAR rotation and one-to-one native rotation.
- Each family reports signed and absolute contribution, cancellation, global
  share, and deterministic rank.  Top-1/top-3/top-10 concentration and
  signed/absolute replay are fixed before evaluation.
- Target rotation `2626` / native `1210` is explicit and is further grouped
  by translation, including fixed translations `78,83,76,80,82`.
- The threshold-free question is whether the complete target family
  dominates aggregate partition movement and how much the fixed translations
  cover, given that V12 found seven target-family candidates in its top ten
  but only `7.888371397363057%` global coverage for the fixed five.
- This cannot identify a raw operand, authorize a fix or new job, change the
  scorecard, or establish full K=4 posterior/FSC-AUC parity.
- Predeclaration:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_partition_rotation_families_v13_75cf45b6_20260730T174915ET/provenance/PREDECLARED_DIAGNOSTIC.md`.

## 2026-07-30 K=4 partition rotation-family result

- All `2968` rotation families replay the V12 signed and absolute candidate
  totals exactly, with zero residual.
- Target rotation `2626` / native `1210` ranks first at
  `19.13906864978897%` of global absolute partition movement.  Rotation
  `947` ranks second at `10.906452683930534%`.
- Family top-1/top-3/top-10 concentration is
  `19.13906864978897%` / `34.95688802532688%` /
  `55.30006470453038%`.  The target is the largest family but not a majority;
  the aggregate shift remains multi-family.
- Target-family signed contribution is `+0.00021664740697420007` against
  absolute `0.0002172067577153348`, only
  `0.25751995334684086%` cancellation.
- Within the target family, queued translations `80` and `82` rank first and
  second at `14.049798813571693%` each; translation `83` ranks third at
  `13.116469579188697%`.
- Fixed translations `78,83,76,80,82` cover
  `41.216067206332085%` of target-family absolute movement because `78,76`
  contribute zero.  The queued pair alone covers `28.099597627143386%`.
- Translation top-1/top-3/top-10 concentration within the family is
  `14.049798813571693%` / `41.216067206332085%` /
  `77.26998635364659%`.
- The queued pair therefore probes the two strongest contributors inside the
  globally leading family, but cannot alone explain the complete target
  family or multi-family normalization shift.  No new job or production
  change is authorized.
- Thread-count-1 and thread-count-8 reports are byte-identical, SHA-256
  `ceb413ca73e1322eb4a1c2f5dd386e71fc608537b59e7b74d65e966d49c37c8e`.
  Removing only the V13 schema and family report reproduces V12 exactly.
- Fixed metrics remain K=1 `28/34` strict, `32/34` topology, and `34/34`
  evaluated; K=4 `41/60` direct and `9/15` all-class.

## 2026-07-30 K=4 target-family weight-component predeclaration

- V14 exponentiates all six existing score stages for every candidate in
  target rotation `2626` / native `1210` under the unchanged V12 shared
  reference.
- Consecutive weight differences telescope native operation order, pre-prior
  data path, orientation prior, translation prior, and dump replay directly
  into the V13 target-family partition movement.
- Complete family replay, candidate closure, component signed/absolute
  contribution, cancellation, nonzero count, component-L1 share, and rank
  are fixed before evaluation.
- Fixed translations `78,83,76,80,82` and queued pair `80,82` receive
  explicit component coverage.
- The threshold-free question is whether nonlinear weighting preserves the
  selected-candidate pre-prior dominance across all 80 target-family
  candidates.
- This cannot identify a raw operand, authorize a fix or new job, change the
  scorecard, or establish full K=4 posterior/FSC-AUC parity.
- Predeclaration:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_target_family_weight_components_v14_043570b8_20260730T180418ET/provenance/PREDECLARED_DIAGNOSTIC.md`.

## 2026-07-30 K=4 target-family weight-component result

- V14 reproduces V13 target-family signed weight
  `+0.00021664740697420007` and absolute weight
  `0.0002172067577153348`.  All 80 candidate telescopes close exactly;
  component signed-sum residual is `-2.710505431213761e-20`.
- Pre-prior data path ranks first at `68.76001917139446%` of family component
  L1.  Its absolute contribution is `0.00018345801368107415`, signed
  contribution `+0.00014445558412118684`, and cancellation
  `21.259594376558366%`.
- Native float32 operation order ranks second at
  `31.20013120567811%`, with absolute `0.00008324480078044925`, signed
  `+0.00007229814528800239`, and cancellation
  `13.149956982079503%`.
- Translation prior contributes only `0.03984962292742545%`: six candidates
  sum to `-1.0632243498917863e-7`.  Orientation prior and dump replay are
  exactly zero.
- Total component L1 is `0.0002668091368965126`; cross-component
  cancellation is `18.800604246836106%`.
- Fixed translations cover `42.410355764206203%` of pre-prior absolute
  movement and `56.57078288375883%` of operation-order movement.  Queued
  `80,82` cover `20.273277661972602%` and `28.63995373494699%`,
  respectively, and are positive in both.
- Translations `78,76` retain exact component cancellation; `83,80,82`
  remain pre-prior dominated.
- Nonlinear weighting therefore preserves pre-prior-first ordering across
  the complete leading family, but does not identify its raw operand or
  authorize a fix/new job.
- Thread-count-1 and thread-count-8 reports are byte-identical, SHA-256
  `446c321d17d81a7e3cf48bf29a37a520aa5bc514616f7b5c9a8300a183ceb226`.
  Removing only the V14 schema and telescope reproduces V13 exactly.
- Fixed metrics remain K=1 `28/34` strict, `32/34` topology, and `34/34`
  evaluated; K=4 `41/60` direct and `9/15` all-class.

## 2026-07-30 K=4 marginal-TV concentration predeclaration

- V9 will rank every V8 rotation and translation marginal by
  `0.5 * abs(marginal_mass_delta_recovar_minus_native)`, breaking exact ties
  by ascending integer identity.
- It will report deterministic top-1/top-3/top-10 concentration, exact replay
  of complete marginal TV, and combined marginal-TV coverage for the
  pre-existing selected strata, including queued translations `80,82`.
- The result is threshold-free and scoped to one captured class.  It cannot
  change the frozen scorecard, establish a full K-class posterior or
  FSC/FSC-AUC result, authorize a production fix, or authorize another
  science job before an existing official `80,82` pair qualifies.
- Predeclaration:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_marginal_tv_concentration_v9_9d33f058_20260730T164706ET/provenance/PREDECLARED_DIAGNOSTIC.md`.

## 2026-07-30 K=4 marginal-TV concentration result

- V9 replays the V8 rotation and translation marginal TV exactly with zero
  residual.  Rotation top-1/top-3/top-10 concentration is
  `19.242801636833587%` / `38.42723815170505%` /
  `56.42859023722319%`; translation is `13.523267080529192%` /
  `31.97759071909936%` / `68.60886905129614%`.
- Rotation row `951` (native `1655`) ranks first; fixed target row `2626`
  (native `1210`) ranks second and covers `13.588772932494678%` of rotation
  marginal TV.
- Translation `78` ranks first, `83` second, and `76` third.  Queued target
  `80` ranks fourth at `7.997860283137037%`; queued target `82`, despite
  ranking first by candidate-level TV in V7, ranks twenty-first after
  cancellation at `0.9554539981169818%`.
- The immutable queued `80,82` pair covers only `8.953314281254018%` of
  translation marginal TV.  This shows why candidate-level target priority
  is not equivalent to marginal decision relevance, but does not supersede
  the already queued causal operand audit or authorize a new job.
- Thread-count-1 and thread-count-8 reports are byte-identical, SHA-256
  `b04fb7b598a5699f7323d396d564a4660466473f9780e3984aba513215c2002c`.
  Removing only the V9 schema and declared fields reproduces V8 exactly.
- This remains non-scoring: K=1 is `28/34` strict, `32/34` topology, and
  `34/34` evaluated; K=4 is `41/60` direct and `9/15` all-class.

## 2026-07-30 22:23 ET — fixed K=4 causal metric and representative chain

- Commit `fe0560f6bafcb87f15460b07d7d7ce508e4dbec7` checks in a
  versioned, fixed-denominator K=4 causal-boundary metric.  It is `2/4`
  passed and `4/4` evaluated: native target operand replay and fixed target
  raw diff2 at translations `80,82` pass; complete active-table raw diff2
  and combined score fail.  The validator freezes case identity/order,
  owner jobs, physical GPU, grid/finalization policy, classification,
  summary, checkmarks, and all six evidence digests.  It cannot change the
  K=1 or K=4 FSC/FSC-AUC scorecards.
- Native representative job `11813655` completed `0:0` on the pinned A100.
  Its completion SHA-256 is
  `58ea7b485e49219668b6211296684924173a6a5f5d2220e7c2b78044f89dbec6`;
  operand SHA-256 is
  `e9e848ed4f37e143e9b318e80a41df754b27b39bdd05363602917f44c07887b6`.
  Native candidate `66561`, native rotation `1738`, mapped RECOVAR rotation
  `954`, and translation `13` replay raw `516.3260498046875` bitwise.
  V17 records RECOVAR raw `516.3261108398438`, a positive one-ULP delta.
- Initial contribution job `11813772` failed closed before capture because
  the device-signature arm lacked its required per-particle-launch flag.
  Its root remains untouched.  Corrected isolated retry `11814215` supplies
  exactly the four required device-signature flags and is resource-pending.
  Comparator `11814470` depends on it and outcome-safe classifier
  `11814788` depends on the comparator.
- Pre-result audit found that centering a single counterfactual value always
  gives zero energy, making the previous top-level label arbitrary.
  Commit `39327674f142115badced374aa7c073a2b8381f7` uses centered
  attribution only for at least two informative candidates, raw attribution
  for one informative candidate, and `no_nonzero_fine_operand_residual` for
  a true zero residual.  Commit
  `5bd5a54c980404071833ee13035d214a663e32d7` adds a compatibility auditor
  that emits a separate V2 classification without rewriting immutable GPU
  evidence.
- Validation at `5bd5a54c` passed import provenance, Python compilation,
  scoped Ruff, scorecard render freshness, `git diff --check`, and
  `134/134` targeted K4/scorecard tests.  GitHub Actions Wheels run
  `30598039332` completed successfully for that exact commit.  No tolerance,
  baseline, fixed denominator, or map gate changed; no correlation metric
  was used.
- A read-only static preflight on 2026-07-30 revalidated launcher SHA-256
  values `a80b9706680a34f8b4f3f4b6ee69fd8236e4ed720e643826559c0325fcfe3493`,
  `16c07b5370cacae7969256efae32b38735391e6d56641a84315baac12e8e637d`,
  and
  `66b2d262c913265a2231fe111b605e98e730a00a0db5414d55071a005709df71`.
  All three parse with `bash -n`.  The K=4 science checkout remains clean at
  `ec68f651a4408ed14ed7ebce0ddf3d54a74e0d41`, and all exclusive output
  roots remain empty.
- Before job `11814788` produces a component label, the next possible
  multistratum cohort is predeclared from V17 rotation-L1 ranks 1--3:
  candidate `66561` (`+1` ULP), candidate `62317` (`+2` ULP), and
  candidate `63564` (`-2` ULP).  The predeclaration SHA-256 is
  `2c2dba8ef2336152b2fe81ccda3f7deca578faee56f0627bdd9f742113088fff`.
  It requires raw majority and informative centered-cohort agreement before
  carrying a component hypothesis forward; otherwise the result remains
  unresolved.  No follow-up science job is authorized before the current
  chain completes.
- Provenance:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_rawrep_rec954_contrib_retry1_ec68f651_20260730T2138ET/provenance/STATIC_PREFLIGHT_AUDIT_20260730T2217ET.md`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_rawrep_classification_v2b_audit11814470_5bd5a54c_20260730T2205ET/provenance/MULTISTRATUM_FOLLOWUP_PREDECLARATION.md`.
- Current fixed metrics remain K=1 `28/34` strict (`82.4%`), `32/34`
  topology, and `34/34` evaluated; K=4 `41/60` direct (`68.3%`) and
  `9/15` all-class; K=4 causal boundary `2/4`.
- No Codex process or Slurm job was killed, signalled, suspended, cancelled,
  reprioritized, or otherwise altered during this audit.

## 2026-07-30 22:37 ET — corrected K=1 serialized-restart owner chain

- Read-only inspection of Slurm's immutable submitted scripts supersedes the
  earlier owner/admissibility conclusion.  Job `11785170` is pinned to
  launcher SHA-256
  `0ba39a6e298c0f504f46d53126ae3f52df05f2c8e51299136d7870fa359780f8`
  and analysis HEAD `e9ce23576f27946fa8b762f1c8855207cf088bce`,
  while the referenced on-disk launcher and analysis checkout are now
  `e8166b3cf067ffe819d6955696e185b453c574bfe3073d4779155426b21631f7`
  and `81af6687a6f0fbf2efc54dc1edf64cc2803894d6`.  Its fail-closed
  preflight must stop before science.  Dependent jobs `11791339` and
  `11791711` cannot yield admissible evidence.
- The unique potentially admissible K=1 chain is iteration-0 science
  `11785428` followed by robust FSC/FSC-AUC auditor `11791340`,
  iteration-1 science `11785547` followed by robust auditor `11791341`,
  and pair auditor `11791712`.  The valid science launchers and analysis
  bytes match their immutable submitted values.  The analysis worktree is
  currently clean and both exclusive science output roots are empty.
- Each robust arm has a fixed denominator: `14/14` serialized-restart score
  dominance, `14/14` absolute score-gate passes, strict parity FSC-AUC
  improvement for half 1, half 2, and merged (`3/3`), and nondegraded GT
  FSC-AUC for the same maps (`3/3`).  The pair has exactly two arms and
  classifies all four pass/fail combinations.  No tolerance is fitted and
  no correlation metric is used.
- This causal chain cannot change the frozen K=1 scorecard directly.  Fixed
  metrics remain `28/34` strict (`82.4%`), `32/34` topology, and `34/34`
  evaluated; K=4 remains `41/60` direct (`68.3%`), `9/15` all-class, and
  `2/4` on the separate non-scoring causal boundary.
- Full audit and checksum:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_parity_monitor_fe0560f6_20260730T2142ET/provenance/ACTIVE_K1_RESTART_VALIDITY_V2_20260730T2237ET.md`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_parity_monitor_fe0560f6_20260730T2142ET/provenance/ACTIVE_K1_RESTART_VALIDITY_V2_20260730T2237ET.md.sha256`.
- No Codex process or Slurm job was killed, signalled, suspended, cancelled,
  reprioritized, or otherwise altered during this audit.

## 2026-07-30 23:00 ET — K=4 exact component-tie fail-closed guard

- A pre-result audit of the V17 representative ingestion path found that an
  exact tie in component energy-removed fractions still selected the first
  component by dictionary order.  This is distinct from the already fixed
  single-value centering problem: both raw one-candidate and centered
  multi-candidate counterfactuals can have informative nonzero energy while
  two or more substitutions attain the same exact maximum.
- The current comparator serializes the complete exact-maximizer set and
  whether it is unique.  Classification names a component only for a unique
  exact maximum; otherwise it emits
  `multiple_fine_operand_components_tie_for_largest_raw_single_substitution_effect`
  or the corresponding centered classification.
- The current compatibility reclassifier reconstructs the exact-maximizer set
  from the three fixed component records.  It leaves
  `selected_component=null`, lists `tied_components`, and sets
  `component_attribution_resolved=false` on a tie.  It also fails closed when
  the duplicated top-level target L2, informative flag, or strongest fraction
  disagrees with the component records.  No fitted tolerance, scale, sign,
  map surrogate, or correlation is used.
- Already submitted outcome-safe classifier `11814788` remains isolated on
  its clean, hash-pinned `5bd5a54c` source.  It was not modified or replaced.
  Its eventual report may carry a component hypothesis only after a
  current-head tie-aware replay confirms a unique exact maximum.  A tie is an
  unresolved result, not authorization for the frozen three-candidate
  follow-up.
- Import provenance, parity ancestry, scoped Ruff, Python compilation,
  scorecard-render freshness, `git diff --check`, and the complete affected
  K=4 causal/scorecard unit slice pass.  The fixed denominator is `110/110`,
  including raw and centered exact-tie cases.
- This is diagnostic-report hardening only.  Fixed metrics remain K=1
  `28/34` strict (`82.4%`), `32/34` topology, and `34/34` evaluated; K=4
  remains `41/60` direct (`68.3%`), `9/15` all-class, and `2/4` on the
  separate non-scoring causal boundary.
- No Codex process or Slurm job was killed, signalled, suspended, cancelled,
  reprioritized, or otherwise altered during this audit.

## 2026-07-30 23:29 ET — consolidated fixed-metric reporter

- `scripts/report_em_parity_progress.py` provides one deterministic,
  machine-readable and PR-ready view of all six fixed panels already used in
  progress updates.
- It calls the existing validators for the frozen K=1 scorecard, artifact
  manifest, K=4 trajectory snapshot, and separate K=4 causal-boundary
  scorecard before emitting any result.
- The unchanged output is K=1 `28/34` strict FSC/FSC-AUC, `32/34` topology,
  and `34/34` evaluated; K=4 `41/60` direct per-class FSC-AUC, `9/15`
  all-class iterations, and `2/4` on the non-scoring causal boundary.
- The reporter serializes exact source SHA-256 values and K=1 strict history
  `20 → 21 → 22 → 23 → 25 → 26 → 27 → 28`.  It cannot change a scorecard
  or authorize a production change.
