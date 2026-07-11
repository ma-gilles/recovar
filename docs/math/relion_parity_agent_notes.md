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

- `tests/unit/test_refine_relion_mode.py::test_final_all_data_grid_correct_env_defaults_to_quality_mode`
- `tests/unit/test_dense_iteration_loop_merge_guards.py::test_final_all_data_grid_correction_defaults_to_quality_mode`
- `tests/unit/test_relion_functions.py::test_gridding_correct_matches_relion_binding`

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
iter-2 free-trajectory gap.

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
correlation against RELION is only `0.995764` (half correlations
`0.995723/0.995755`), below the `0.9995` K=1 gate. RECOVAR and RELION reach
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
`scripts/compare_iter1_bpref_accum.py`. Do not extend the free trajectory
until this accumulator boundary is classified.
