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
- Diagnostic-only commit `b86fb607cd147e71a2069c97403ab6d23c4482fe`
  forwards the already implemented pass-1 significance dump controls through
  the K=1 Slurm launcher and adds a focused dry-run test.  The test result is
  2 passed, 37 deselected.  Exact-fixture same-GPU setup/science/summary jobs
  are `11509108`/`11509109`/`11509110`; after-any RELION-versus-RECOVAR
  operand comparison is `11509172`.
- Immutable result root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fixedsuite_case24_strict_a2be302c_20260722T204500Z`.
  Focused capture root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case24_it2_particle2767_capture_b86fb607_20260722T212227Z`.
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
