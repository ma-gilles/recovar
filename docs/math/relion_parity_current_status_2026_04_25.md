# RELION-parity current status — updated 2026-04-27

If you are starting a new debugging session, read the short handoff first:
`docs/math/relion_parity_quick_handoff_2026_04_27.md`.

This file is the long audit trail and contains stale/failed experiments as
well as current results.

## Branch
`claude/relion-parity-local-search-fix`. The current pushed baseline before
the selected-only fine-grid speed fix is `2e2d8301`; this document entry is
updated with the selected-only fix in the same branch.

## 2026-04-27 5k/128 timing and parity baselines

Dataset:
`/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized/particles.star`.
RELION reference:
`/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized/relion_ref_os0`.
This fixture has 5,000 particles, box size 128, pixel size 4.25 A, and
half-sets of 2,515 and 2,485 particles.

### Per-half tau2/noise source fix

RELION calls `BackProjector::updateSSNRarrays` independently for each half-map
backprojector. The gold-standard FSC is shared, but the shell-wise `sigma2`,
`tau2`, and `data_vs_prior` ingredients use each half's own BPref Fourier
weights outside joined low-resolution shells. RECOVAR previously passed the
average of half1/half2 weights into the tau2 update, which created an apparent
noise/tau2 gap after the low-resolution join boundary.

RECOVAR now computes tau2 once per half from that half's own `Ft_ctf`, keeps the
legacy saved `tau2_*_iter` arrays aligned to RELION half1 model.star for diff
scripts, and reconstructs each half with its own tau2 radial prior. The shared
`mean_variance` remains the average of the two half priors for the next E-step.

Focused 5k replay artifact:
`_agent_scratch/full5k_tau2_halfspecific_replay_20260427_1706`. The replay
emitted one RECOVAR iteration in 235.2 s.

| Metric vs RELION half1 it002 model.star | Before per-half tau2 | After per-half tau2 |
|---|---:|---:|
| sigma2 max / mean relative gap, shells 1..35 | 2.7015e-2 / 5.067e-3 | 2.1887e-2 / 6.335e-4 |
| tau2 max / mean relative gap, shells 1..35 | 1.7935e-2 / 4.433e-3 | 4.540e-4 / 6.120e-5 |
| FSC max / mean absolute gap, shells 1..35 | 2.541e-3 / 4.483e-4 | 2.541e-3 / 4.483e-4 |
| Pmax mean abs / max abs | 7.5e-5 / 1.03e-3 | 7.5e-5 / 1.03e-3 |
| pose / translation agreement | exact | exact |
| map corr RECOVAR vs RELION | 0.999995 | 0.999995 |

Shells 14-34 now match RELION essentially exactly for tau2/sigma2; the
remaining max gap is shell 35 only. Treat shell 35 as the same outer support
boundary issue described below unless a later replay shows it drives
end-to-end drift.

### Convergence-state initialization source fix

RELION does not initialize auto-refine convergence against an infinite previous
resolution. When replaying from a saved RELION iteration, the starting state is
the previous `run_itNNN_optimiser.star` plus `run_itNNN_half1_model.star`:
`rlnCurrentResolution`, `rlnNumberOfIterWithoutResolutionGain`,
`rlnNumberOfIterWithoutChangingAssignments`,
`rlnSmallestChangesOrientations`, `rlnSmallestChangesOffsets`, and
`rlnSmallestChangesClasses`. The 5k/128 replay exposed this mismatch:
starting from RELION it001, RELION it002 has
`rlnNumberOfIterWithoutResolutionGain=1`, while RECOVAR had logged
`stalls: resol=0` because `state.current_resolution` started at `inf`.

RECOVAR now initializes replay convergence state from those RELION optimiser
and model STAR fields. For non-replay RELION-mode runs, the initial
`current_resolution` is seeded from `init_fsc`/`init_current_size`, or from
`ini_high` for a cold first iteration. This fix targets stop-criterion parity;
it does not change E-step scoring, M-step accumulators, tau2, or noise.

### Local-search `sigma2_offset` posterior handoff fix

RELION updates `mymodel.sigma2_offset` from the posterior-weighted sufficient
statistic accumulated in `storeWeightedSums`:
`sum weight * pixel_size^2 * (prior - old_offset - sampled_translation)^2`,
then normalizes by `2 * sum_weight` for 2D SPA and clamps to
`min_sigma2_offset=2 A^2`. RECOVAR already accumulated the same statistic in
the dense and exact-local EM engines when `translation_prior_centers` was
provided.

The local-search branch had a handoff bug: it passed `trans_prior_center` as
the local translation prior used for score normalization, but did not pass that
same center through as `translation_prior_centers` to the exact local engine.
As a result, late local-search iterations produced `wsum_sigma2_offset=0` and
fell back to hard-assignment offset changes, visible in logs as
`C1 fallback: sigma_offset updated ... from hard assignments`.

RECOVAR now forwards `translation_prior_centers=trans_prior_center` through
the exact local-search path. Targeted validation:
`_agent_scratch/local_sigma_offset_posterior_smoke2_20260427_174016_14328`
ran `--iter 4 --max_iter 2 --max_particles 20 --local_engine exact_v1`.
The second emitted iteration entered local search and logged:
`C1: sigma_offset updated 1.414 A from posterior variance`, not the fallback.
This smoke is intentionally only a handoff/path validation; the 20-particle
subset does not represent end-to-end map parity.

### Full 13-row end-to-end replay after pre-shift/pass-2 fixes

Artifact:
`_agent_scratch/long_5k_exactlocal_2e2d8301_local_20260427_093922`.
Command used `--iter 1 --max_iter 13 --skip_final_iteration
--local_engine exact_v1 --local_search_profile off` on one visible A100
(`jax_devices=["cuda:0"]`, `CUDA_VISIBLE_DEVICES=1`). RECOVAR emitted 13 rows
matching RELION it002 through it014 and did not stop at 8 rows. Total wall time
was 2,956.8 s.

| Rec iter | RELION it | wall s | cs | pixres A | hp | REC Pmax | REL Pmax | mean abs | p99 abs | max abs | Pmax corr | pose mean/max deg | trans mean/max px | map corr R-vs-L | GT corr R/L | FSC0.143 R/L |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 302.8 | 70 | 25.0 | 3 | 0.646336 | 0.646337 | 7.53e-05 | 3.45e-04 | 1.03e-03 | 1.000000 | 5.76e-06/1.90e-05 | 3.20e-07/5.58e-07 | 0.999965 | 0.939964/0.939879 | 36/36 |
| 2 | 3 | 77.4 | 82 | 24.0 | 3 | 0.964185 | 0.964374 | 6.60e-03 | 1.04e-01 | 3.03e-01 | 0.973796 | 7.89e-02/15.12 | 8.30e-07/8.63e-07 | 0.999940 | 0.948235/0.948359 | 41/41 |
| 3 | 4 | 71.5 | 80 | 24.0 | 3 | 0.974087 | 0.973470 | 7.31e-03 | 1.51e-01 | 4.45e-01 | 0.940930 | 7.93e-02/15.18 | 4.11e-07/4.83e-07 | 0.999939 | 0.946547/0.946794 | 40/40 |
| 4 | 5 | 69.8 | 80 | 24.0 | 3 | 0.973225 | 0.972628 | 7.77e-03 | 1.60e-01 | 3.67e-01 | 0.940812 | 1.03e-01/14.78 | 5.81e-06/5.85e-06 | 0.999942 | 0.946122/0.946271 | 40/40 |
| 5 | 6 | 176.8 | 80 | 25.0 | 4 | 0.928715 | 0.928947 | 1.69e-02 | 1.81e-01 | 4.40e-01 | 0.950317 | 7.67e-02/6.75 | 5.44e-08/1.64e-06 | 0.999933 | 0.962108/0.962210 | 41/41 |
| 6 | 7 | 179.2 | 82 | 25.0 | 4 | 0.948387 | 0.949317 | 1.53e-02 | 2.01e-01 | 4.60e-01 | 0.927246 | 1.08e-01/7.12 | 1.52e-04/3.78e-01 | 0.999902 | 0.961636/0.961718 | 42/41 |
| 7 | 8 | 197.9 | 82 | 26.0 | 5 | 0.884212 | 0.885416 | 3.21e-02 | 2.99e-01 | 6.17e-01 | 0.909393 | 1.01e-01/4.79 | 1.23e-04/3.05e-01 | 0.999866 | 0.965130/0.965239 | 42/42 |
| 8 | 9 | 182.4 | 84 | 26.0 | 5 | 0.894761 | 0.896648 | 3.67e-02 | 3.34e-01 | 6.71e-01 | 0.868923 | 1.16e-01/4.10 | 2.44e-04/3.05e-01 | 0.999838 | 0.965878/0.965953 | 43/43 |
| 9 | 10 | 211.7 | 84 | 26.0 | 6 | 0.693342 | 0.692447 | 5.82e-02 | 3.38e-01 | 5.68e-01 | 0.893892 | 1.24e-01/2.95 | 1.51e-03/2.91e-01 | 0.999858 | 0.965971/0.966049 | 43/43 |
| 10 | 11 | 187.4 | 84 | 26.0 | 6 | 0.693309 | 0.694100 | 6.57e-02 | 3.71e-01 | 5.48e-01 | 0.866578 | 1.36e-01/3.54 | 3.50e-04/2.91e-01 | 0.999874 | 0.966651/0.966737 | 43/43 |
| 11 | 12 | 643.7 | 84 | 26.0 | 7 | 0.315508 | 0.316854 | 3.31e-02 | 1.59e-01 | 4.79e-01 | 0.874782 | 1.13e-01/2.08 | 2.30e-04/2.87e-01 | 0.999905 | 0.966608/0.966694 | 43/43 |
| 12 | 13 | 301.5 | 84 | 26.0 | 7 | 0.306719 | 0.307071 | 3.09e-02 | 1.37e-01 | 2.98e-01 | 0.900640 | 9.74e-02/2.00 | 9.79e-04/2.87e-01 | 0.999922 | 0.966065/0.966154 | 43/43 |
| 13 | 14 | 353.6 | 84 | 26.0 | 7 | 0.323451 | 0.324587 | 2.98e-02 | 1.32e-01 | 4.16e-01 | 0.909574 | 8.65e-02/1.18 | 1.17e-04/2.87e-01 | 0.999928 | 0.966522/0.966607 | 43/43 |

Final map parity: half1 corr vs RELION `0.999884`, half2 corr vs RELION
`0.999883`; merged GT corr RECOVAR/RELION `0.966522/0.966607`; merged
FSC<0.143 shell RECOVAR/RELION `43/43`.

Timing interpretation before the selected-only fine-grid fix: iteration 11
(`hp=7`) took 643.7 s because exact local materialized the full perturbed fine
grid (about 151M rotations) before selecting per-image neighborhoods. GPU was
mostly idle during this CPU/RAM-heavy materialization step. The selected-only
fix in this branch now selects canonical fine-grid IDs first and constructs
only the selected perturbed rotation matrices, matching RELION's
`selectOrientationsWithNonZeroPriorProbability` before `SamplingPerturbation`
ordering. Targeted unit gate after the fix: `141 passed` for
`tests/unit/test_convergence.py tests/unit/test_refine_relion_mode.py`.

Focused post-fix HP7 replay:
`_agent_scratch/fixed_hp7_selected_codex_hp7_selected_4fbba3e2_20260427_105105`.
Command used `--iter 11 --max_iter 1 --skip_final_iteration --local_engine
exact_v1 --local_search_profile off` on one A100. The selected-only path
logged `Using selected-only fine local-search grid: order=7 (150994944
rotations)` and completed the iteration in 391.0 s, with total script wall time
412.9 s. This is a 39% iteration-wall improvement vs the previous 643.7 s HP7
row while preserving fixed-state parity:

| RELION it | REC Pmax | REL Pmax | mean abs | p99 abs | max abs | Pmax corr | pose mean/max deg | trans max px | map corr R-vs-L | GT corr R/L | FSC0.143 R/L |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 0.316846 | 0.316854 | 2.51e-4 | 1.11e-3 | 2.55e-3 | 0.999992 | 2e-4/0.5534 | 0.0 | 0.999953 | 0.966635/0.966694 | 43/43 |

### Full 8-row end-to-end trajectory

Artifact:
`_agent_scratch/long_end2end_parity_20260426_182134/metrics_summary.json`.
This run predates the 2026-04-27 accelerated `pdf_offset` translation-prior
fix, so use it as the current complete 5k timing/performance baseline, not as
the final post-fix parity target. It ran on four visible GPUs
(`cuda:0`, `cuda:1`, `cuda:2`, `cuda:3`) and completed 8 emitted iterations in
2,075.6 s. The requested command used `--max_iter 13 --skip_final_iteration`,
but RECOVAR converged after 8 emitted rows; rows after that in older reports
were RELION-only reference rows.

| Rec iter | RELION it | wall s | cs | pixres A | hp | REC Pmax | REL Pmax | mean abs | p99 abs | max abs | Pmax corr | pose mean/max deg | trans mean px | map corr R-vs-L | GT corr R/L | FSC0.143 R/L |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 249.9 | 70 | 25.0 | 3 | 0.646471 | 0.646337 | 0.0010897 | 0.000476866 | 0.461337 | 0.997458 | 0.0262/19.2 | 0.0072 | 0.999971 | 0.940109/0.939879 | 36/36 |
| 2 | 3 | 84.3 | 82 | 24.0 | 3 | 0.964680 | 0.964374 | 0.0063416 | 0.102881 | 0.333938 | 0.975304 | 0.0839/11.2 | 0.0068 | 0.999957 | 0.948338/0.948359 | 41/41 |
| 3 | 4 | 76.7 | 80 | 24.0 | 3 | 0.973809 | 0.973470 | 0.00653746 | 0.1238 | 0.464203 | 0.951907 | 0.0748/13.2 | 0.0056 | 0.999967 | 0.946738/0.946794 | 40/40 |
| 4 | 5 | 68.8 | 80 | 24.0 | 3 | 0.973248 | 0.972628 | 0.00620259 | 0.111142 | 0.466193 | 0.960749 | 0.0667/14.8 | 0.00481 | 0.999969 | 0.946217/0.946271 | 40/40 |
| 5 | 6 | 211.8 | 80 | 25.0 | 4 | 0.930572 | 0.928947 | 0.0138341 | 0.143274 | 0.434564 | 0.963276 | 0.0584/8.03 | 0.000258 | 0.999937 | 0.962095/0.962210 | 41/41 |
| 6 | 7 | 204.6 | 82 | 25.0 | 4 | 0.954587 | 0.949317 | 0.0191984 | 0.246349 | 0.495957 | 0.882088 | 0.0783/7.96 | 0.0017 | 0.999915 | 0.961587/0.961718 | 42/41 |
| 7 | 8 | 591.9 | 82 | 26.0 | 5 | 0.885187 | 0.885416 | 0.0347312 | 0.318182 | 0.532186 | 0.895687 | 0.102/5.58 | 0.000734 | 0.999869 | 0.965139/0.965239 | 42/42 |
| 8 | 9 | 586.7 | 84 | 26.0 | 5 | 0.899029 | 0.896648 | 0.0424789 | 0.366282 | 0.573372 | 0.833259 | 0.122/5.26 | 0.00104 | 0.999820 | 0.965880/0.965953 | 43/43 |

Interpretation: map quality remains close to RELION through all emitted rows,
but per-particle Pmax parity degrades in later rows. The latest source-level
work traced part of this to RECOVAR using the wrong RELION local-translation
prior center; see the post-fix fixed-state baseline below.

### Post-fix fixed-state RELION it008 replay

Artifact:
`_agent_scratch/recovar_it008_5k_priorfix_20260427_031136`.
This is the current best post-fix fixed-state baseline. It replays one
iteration from RELION it008 on one A100 80GB using `--iter 7 --max_iter 1
--skip_final_iteration --local_engine exact_v2` (`exact_v2` currently falls
back to exact_v1).

| Run | RELION it | wall s | cs | pixres A | GPU | REC Pmax | REL Pmax | mean abs | median abs | p99 abs | max abs | pose mean/max deg | trans mean px | map corr R-vs-L | GT corr R/L | FSC0.143 R/L |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| post-fix fixed-state | 8 | 239.8 | 82 | 26.0 | A100 80GB | 0.885424 | 0.885416 | 0.00012301 | 1.87452e-05 | 0.00104297 | 0.0266694 | 5.669e-06/2.035e-05 | 1.302e-06 | 0.999952 | 0.965163/0.965239 | 42/42 |

Local-search profile for this fixed-state replay:

| Half | images | EM time s | chunks | union rows | unique global rotations | duplicate rotation factor |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2,515 | 121.7 | 2,515 | 3,571,740 | 1,842,518 | 1.939 |
| 2 | 2,485 | 99.7 | 2,485 | 3,530,076 | 1,829,936 | 1.929 |

The residual fixed-state gap after this fix is now score-level, not assignment
level. The current worst particle is original index 4603: RECOVAR and RELION
choose matching pose/translation to numerical noise, but RECOVAR is still
about 0.0267 higher in Pmax because the top-vs-second candidate score gap is
too sharp by about 0.136 log units. Continue by comparing RELION `fine_ref`,
`fine_shifted`, and `corr_img` against RECOVAR's `debug_proj_weighted`,
`debug_shifted_score`, and `debug_ctf2_over_nv` for that particle.

### Integer old-offset pre-shift source fix

Source/dump comparison for particle 4603 showed candidate translation phase,
projection interpolation, CTF/noise weights, candidate support, and priors were
already matching. The remaining score gap came from the image side: RELION
rounds `old_offset` and applies it to the real-space image with a zero-filled
integer translate before FFT (`TranslateAndNormCorrect` / `cuda_kernel_translate2D`),
whereas RECOVAR had modeled that same integer pre-centering as a circular
Fourier phase shift.

RECOVAR now applies RELION-style zero-filled real-space integer pre-shifts for
integral `image_pre_shifts` on real-space batches, and keeps the legacy
Fourier-phase path only for non-integral or already-Fourier test inputs.

Focused validation:
`_agent_scratch/focused_it008_stack4603_zerofill_20260427_090904`, local A100,
`--iter 7 --max_iter 1 --skip_final_iteration --keep_stack_indices 4603
--local_engine exact_v2`.

| Case | RECOVAR Pmax | RELION Pmax | abs gap | pose/trans |
|---|---:|---:|---:|---|
| before zero-filled pre-shift | 0.606924 | 0.580255 | 0.026669 | exact |
| after zero-filled pre-shift | 0.580947 | 0.580255 | 0.000692 | exact |

The p4603 top-vs-second score-gap excess dropped from about `0.136` log units
to about `0.022` log units, and the RECOVAR-vs-RELION weighted shifted-image
relative residual dropped from `6.47e-2` to `3.55e-3`. Remaining p4603 gap is
near the RELION accelerated float32/texture arithmetic band and should be
checked in the next full fixed-state 5k replay before pursuing another
single-particle micro-fix.

### Pass-2 routing source fix

The RELION refinement path no longer calls the grouped/bucketed sparse pass-2
helper from `iteration_loop.py`. Normal adaptive sparse pass 2 and the os0
global-significant-support replay branch now both route through
`_run_sparse_pass2_local_search_iteration()` and the exact local engine. The os0
branch keeps the RELION denominator contract by passing the full coarse-grid
`normalization_log_z` from pass 1, and keeps offset-noise accounting by passing
translation prior centers into exact local pass 2.

Targeted coverage:
`tests/unit/test_refine_relion_mode.py -k 'global_significant_support or
sparse_pass2_local_search_matches_per_image_reference or
routes_sparse_adaptive_pass2 or skips_pass2_when_significance_fraction_is_high
or uses_dense_exact_pass2'`.

### Post-fix forced replay timing-only run

Artifact:
`_agent_scratch/long_5k_force13_priorfix_20260427_033007/slurm-7393757.out`.
This run used `--force_max_iter_after_convergence` to avoid early convergence,
but it failed after emitted iteration 7 because the runner tried to read
missing `run_it015_sampling.star`. It is useful as a post-fix timing trace
only; it did not write final comparison tables.

| Rec iter | wall s | cs | pixres A | hp | ave Pmax | res A | note |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 161.9 | 82 | 26.0 | 5 | 0.8854 | 20.92 | post-fix forced replay, no final parity table |
| 2 | 141.4 | 84 | 26.0 | 5 | 0.8965 | 20.92 | post-fix forced replay, no final parity table |
| 3 | 173.4 | 84 | 26.0 | 6 | 0.6926 | 20.92 | post-fix forced replay, no final parity table |
| 4 | 147.9 | 84 | 26.0 | 7 | 0.6939 | 20.92 | post-fix forced replay, no final parity table |
| 5 | 378.5 | 84 | 26.0 | 7 | 0.3155 | 20.92 | post-fix forced replay, no final parity table |
| 6 | 223.5 | 84 | 26.0 | 7 | 0.3070 | 20.92 | post-fix forced replay, no final parity table |
| 7 | 223.0 | 84 | 26.0 | 7 | 0.3241 | 20.92 | post-fix forced replay, no final parity table |

## 2026-04-26 M-step/FSC update

Two RELION source details were matched in RECOVAR's M-step/FSC path:

- `BackProjector::getDownsampledAverage` uses RELION `ROUND`
  (round-half-away-from-zero), not NumPy banker rounding.
- `BackProjector::getLowResDataAndWeight` / `setLowResDataAndWeight` apply
  `--low_resol_join_halves` with squared radius
  `k*k+i*i+j*j <= ROUND(padding_factor * lowres_r_max)^2`, not rounded
  shell labels. The old rounded-shell mask over-joined boundary voxels near
  the 40 Å cutoff.

Targeted tests:

- `.pixi/envs/default/bin/python -m pytest tests/unit/test_regularization.py -k 'relion_fsc_from_backprojector' tests/unit/test_relion_functions.py -k 'join_halves_at_low_resolution'`
- Result: `2 passed, 43 deselected`.

Tiny 1k / 64³ 5-iteration replay after both source fixes:
`_agent_scratch/codex_tiny5_joinboundary_20260426_105052_10436`
(`CUDA_VISIBLE_DEVICES=2`, local A100 80GB).

Compared to the previous zero-prior/source-FSC replay
`_agent_scratch/codex_tiny5_zeroprior_20260426_095658_21371`, the RELION
M-step downsampled FSC gap improved:

| Comparison | Old mean abs / max abs | New mean abs / max abs |
|---|---:|---:|
| RECOVAR it000 vs RELION M-step call0001, shells 14-27 | 8.81e-3 / 3.36e-2 | 4.25e-3 / 1.01e-2 |
| RECOVAR it001 vs RELION M-step call0002, shells 14-27 | 1.02e-2 / 5.64e-2 | 2.53e-3 / 7.18e-3 |

Trajectory metrics also improved:

| Metric | Old | New |
|---|---:|---:|
| Final half1 corr vs RELION it006 | ~0.99994 | 0.999970 |
| Final half2 corr vs RELION it006 | ~0.99994 | 0.999969 |
| Per-iter recovar-vs-RELION map corr | 0.999964, 0.999963, 0.999958, 0.999961, 0.999962 | 0.999972, 0.999975, 0.999973, 0.999976, 0.999973 |
| Pmax mean abs gaps by iter | 3.68e-5, 1.16e-2, 9.57e-3, 6.78e-3, 9.04e-3 | 3.53e-5, 9.32e-3, 6.22e-3, 4.77e-3, 6.09e-3 |

Current standing: first E-step score/Pmax parity is arithmetic-level; the
remaining multi-iteration gap is now concentrated in residual M-step/tau2/noise
differences after shell 16 and later local posterior outliers. Continue by
dump-comparing `Ft_y`, `Ft_ctf`, downsampled averages, tau2/data-vs-prior, and
noise for the first particle/iteration where Pmax exceeds the `~1e-4`
arithmetic target.

### Known BPref outer-shell boundary mismatch

Do not reopen the native half-volume BPref outer-shell investigation unless a
later end-to-end replay shows map/pose/Pmax drift that points back to this
boundary. The current best one-iteration replay
`_agent_scratch/codex_exact_recon_proj27_1iter_20260426_161253_14230`
matches RELION at arithmetic level for assignments and maps:

- Pmax mean abs `3.5e-5`, median `8e-6`, max `8.4e-4`, corr `1.0`.
- Rotation and translation assignments are exact vs RELION for all particles.
- Final recovar-vs-RELION map corr is `0.999996`.
- BPref data/weight errors are `~1e-4` through `rpad<=52`, but the last
  reconstruction boundary shells still differ:
  shell 26 data/weight `4.85e-2 / 2.71e-2`, shell 27
  `1.75e-1 / 1.26e-1`.

Source/dump checks already falsified likely causes: rounded/no-DC M-step
support worsened inner-shell parity, sparse rotation float64 did not change the
gap, and a RELION-style `Mweight > 0` data guard was a no-op for this fixture.
Interpretation: this is most likely a padding/interpolation boundary-shell
scatter detail at the edge of the reconstructed support. Treat it as a known
low-priority boundary issue while first pursuing full end-to-end parity in
scores, poses, maps, tau2, noise, and runtime.

### p933 boundary-stress score decomposition — 2026-04-26

Latest two-iteration replay after the compact BPref boundary patch:
`_agent_scratch/codex_compactbounds_p933_20260426_175041_32725`.

Iteration 1 remains arithmetic-level parity: RECOVAR and RELION have
`ave_Pmax=0.863804`, mean abs Pmax gap `3.5e-5`, max `8.36e-4`, and exact
pose/translation assignments.

Iteration 2 improves globally but still has a boundary-stress outlier:
RECOVAR `ave_Pmax=0.978371`, RELION `0.977890`, mean abs Pmax gap
`4.94e-3`, max `0.280959`. The worst particle is original index 933:
RECOVAR `Pmax=0.949866`, RELION `Pmax=0.668907`, with pose and translation
matching to numerical noise.

For p933, the candidate set and priors match the RELION dump:

| Quantity | RECOVAR | RELION |
|---|---:|---:|
| fine candidates | `(rot 0, trans 14)`, `(rot 1, trans 14)` | same |
| rotation priors | `[-6.987382, -6.755034]` | `[-6.987013, -6.755066]` |
| RELION residual gap | n/a | candidate 1 better by `0.468994` |
| RECOVAR pre-prior score gap | candidate 1 better by `2.708965` | n/a |

Direct projector isolation is decisive: projecting RELION's own iter-2
half-map through RECOVAR's RELION-mode projector (`current_size=58`,
`max_r=29`, texture interpolation) reproduces RELION fine references at
RMS `~7e-8`, max `~4e-7`, corr `>0.99999999999`. Therefore p933 is not a
candidate enumeration, normalization, prior, or projection bug when the input
map is RELION's map.

Shell replacement/zeroing diagnostics show the outlier is driven by RECOVAR's
high-shell map coefficients, not by a simple missing-boundary zero:

| Diagnostic | p933 candidate-1 posterior |
|---|---:|
| RELION fine refs / RELION map projected in RECOVAR | `~0.6688` |
| RECOVAR current projection | `~0.9500` |
| replace RECOVAR projection shells `<=27` with RELION | `0.7396` |
| replace shells `<=28` with RELION | `0.6702` |
| replace all shells `<=29` with RELION | `0.6688` |
| zero RECOVAR projection shells `>=26` | `0.9197` |
| zero RECOVAR projection shells `26-28` | `0.9276` |
| zero RECOVAR projection shells `>=24` | `0.6360` |

Conclusion: p933 should remain a regression/stress case, but it should not be
the only driver for the next fix. The next useful trace should use particles
whose gaps are less dominated by the support boundary, and should compare
candidate enumeration, raw residual scores, `Ft_y`, `Ft_ctf`, tau2, noise, and
regularized map coefficients before changing algorithmic behavior.

## 2026-04-26 source-level projector/scoring finding

RELION's accelerated path uses three arithmetic details that must be active
for close E-step parity:

- `Projector::initialiseData(current_size)` sets `r_max = current_size / 2`
  (`projector.cpp:59-68`). For the tiny fixture pass-2 case,
  `current_size=54`, so the projector support radius is `27`, not the
  generic image default `64//2 - 1 = 31`.
- `AccProjectorKernel` samples the Fourier reference through CUDA texture
  objects with `cudaFilterModeLinear` (`acc_projector_impl.h:96` and
  `acc_projectorkernel_impl.h:133-145`).
- RELION's accelerated score kernel computes diff2 directly, and image
  preprocessing follows RELION/FFTW-style centered complex FFTs.

RECOVAR `mode="relion"` now defaults these on unless explicitly overridden:

- `RECOVAR_RELION_DIRECT_DIFF2_SCORING=1`
- `RECOVAR_RELION_TEXTURE_INTERP=1`
- `RECOVAR_RELION_NUMPY_IMAGE_FFT=1`

The windowed EM paths also pass `max_r=float(current_size // 2)` into
projection calls.

## Latest targeted validation: tiny 1k / 64³, start at RELION iter 1

Command shape:
`scripts/run_multi_iter_parity.py --relion_dir .../data_tiny_parity/relion_ref_os0 --data_star .../particles.star --iter 1 --max_iter 1 --skip_final_iteration`

Hardware: local A100 80GB via `CUDA_VISIBLE_DEVICES=1`. Full RECOVAR tests
were intentionally not run; this is an EM-only parity replay.

| Run | Key flags | Wall time | mean abs Pmax | max abs Pmax | mean gap | pose parity |
|---|---|---:|---:|---:|---:|---|
| `_agent_scratch/20260426_tiny1k_directdiff2_maxr27_13954` | direct diff2, no texture, no NumPy FFT | 80.6s | 1.3267e-3 | 1.8571e-2 | -6.06e-5 | one 7.5° outlier |
| `_agent_scratch/20260426_tiny1k_texture_maxr27_31065` | direct diff2 + texture | 78.3s | 5.3374e-5 | 6.713e-4 | +1.68e-6 | exact poses |
| `_agent_scratch/20260426_tiny1k_texture_numpy_maxr27_11236` | direct diff2 + texture + NumPy FFT | 71.9s | 3.6759e-5 | 8.6967e-4 | +1.92e-6 | exact poses |
| `_agent_scratch/20260426_tiny1k_auto_parity_15715` | automatic RELION-mode defaults | 69.5s | 3.6759e-5 | 8.6967e-4 | +1.92e-6 | exact poses |

Particle 668, the tracked failure case:

- With texture + `max_r=27`, RECOVAR's `proj_half` is bit-identical to the
  previous best projection dump (`relerr_to_old=0.0`).
- With texture + NumPy FFT, p668 `Pmax=0.638272786` vs RELION `0.638316`
  (gap about `-4.3e-5`).
- The remaining p668 pre-prior score deltas in RECOVAR row order are
  `[-4.60e-4, 0, -2.02e-4, +6.47e-5, -2.65e-4, -1.52e-4, +8.40e-5]`.

Current status: pose parity is exact on this one-iteration tiny replay and
map parity remains high (`recovar-vs-RELION corr=0.999964`). Remaining
Pmax differences are at the `1e-5` to `1e-4` level for most particles, with
the worst current outlier at original particle 374. Next debugging should
dump particle 374 and compare image/projection/diff2 subterms, not tune
parameters.

## Numerical triage target

RELION accelerated GPU scoring uses float32/texture arithmetic in several
places even when dumps are written as doubles. For this branch, use `~1e-4`
as the practical arithmetic-parity target for pre-prior per-pose scores and
Pmax when poses are identical. Gaps near this level are not automatically
bugs. Escalate `>=1e-3` score/Pmax gaps, any pose flip, or multi-iteration
drift. If a gap is ambiguous, rerun the RECOVAR side with float64 scoring
and obtain a RELION CPU/double or `ACC_DOUBLE_PRECISION` dump for the same
particle/candidate set before changing parameters.

### Float64 RECOVAR scoring check — 2026-04-26

Paired targeted replays on local A100s:

| Run | Scoring precision | Wall time | mean abs Pmax | max abs Pmax | mean gap | pose parity |
|---|---|---:|---:|---:|---:|---|
| `_agent_scratch/20260426_tiny1k_float64_replay_25714` | RECOVAR float64 scoring | 104.0s | 3.7e-5 | 8.70e-4 | +1.92e-6 | exact poses |
| `_agent_scratch/20260426_tiny1k_float32_replay_27227` | forced RECOVAR float32 scoring | 109.9s | 4.5e-5 | 7.67e-4 | -7.42e-7 | exact poses |

For p668, RECOVAR float64 did not remove the remaining shifted pre-prior
score gap vs RELION:
`[-4.60e-4, 0, -2.02e-4, +6.47e-5, -2.65e-4, -1.52e-4, +8.40e-5]`.

Interpretation: the current p668 residual is not caused by RECOVAR-side
float32 reduction alone. It is still consistent with RELION accelerated
float32/texture arithmetic, but the definitive check is a RELION CPU/double
or `ACC_DOUBLE_PRECISION` dump for the same active candidates. That check is
delegated separately.

## Measured gaps (latest validation runs)

### 2026-04-25 21:14–22:12 multi-iter trajectory on 5k/128 fixture (commit 949ab6b8)

| Iter | recovar pmax | RELION pmax | gap_abs | gap_rel |
|------|--------------|-------------|---------|---------|
| 0→1  | 0.042146 | 0.042136 | **+1.1e-5** | +0.025% |
| 1→2  | 0.645229 | 0.646337 | -1.1e-3 | -0.17% |
| 2→3  | 0.965861 | 0.964374 | +1.5e-3 | +0.15% |
| 3→4  | 0.974609 | 0.973470 | +1.1e-3 | +0.12% |
| 4→5  | 0.973150 | 0.972628 | +5.2e-4 | +0.05% |

**Final iter-5 metrics on 5k/128:**
- recovar_reg corr_vs_gt = 0.946049, RELION = 0.946271 (gap 2.2e-4)
- recovar-vs-RELION volume corr = **0.9993**
- pose error (recovar vs RELION): mean 0.45°, p99 9.4°
- 99.5% of poses agree within 1px translation

### Earlier (stale) measurements

| Test | Gap | Status |
|---|---:|---|
| 5k iter 13→14 (codex's canonical late-iter) | **-1.07e-4** | ✓ codex gold magnitude restored |
| Tiny cold-start iter 1 (default) | -17.6% (stale) | ✗ doc was wrong; actual now -0.08% |

The "-17.6% iter-1 deficit" reported in the previous version of this doc
was MISLEADING. It came from per-pose dump comparisons where the recovar
dump was captured AFTER the prior was added (`em_engine.py:1527`) but
RELION's `exp_Mweight_diff2.bin` is captured BEFORE the prior (`ml_optimiser.cpp:8450`).
The actual iter-1 ave_pmax gap on tiny is -0.08% (1k/64) and +0.025%
(5k/128) — at machine precision.

A pre-prior dump option was added (`RECOVAR_DEBUG_PER_POSE_DUMP_PREPRIOR=1`)
for future apples-to-apples comparisons.

## Fixes landed today

1. **Sparse-pass-2 shape bucketing** (commits `66989c86`, `12f1a7c3`) —
   replaced per-image Python loop with shape-bucketed batched evaluation.
   Iter-1 cold compile went from 50+min → ~80s on 5k.

2. **FSC reordering — RELION-exact ordering with hybrid prior fallback**
   (commit `5097ded6`). RELION's `ml_optimiser_mpi.cpp:4031, 4091` computes
   the CURRENT iter's FSC from M-step BPref accumulators BEFORE
   `updateSSNRarrays`. Recovar previously used `fsc_history[-1]` /
   `init_fsc` (PREVIOUS iter's FSC), which at cold start meant
   `init_fsc=zeros` → tau2 ≈ 0 at iter 1, then iter-2's tau2 derived from
   poorly-regularized iter-1 FSC (≈ 0.999) → tau2 amplifies 1e6× → 662×
   volume amplification → ave_Pmax collapse to 0.

   Hybrid choice: prev-iter FSC by default (preserves the late-iter
   cancellation; codex gold gap = -5.7e-5), current-iter fresh FSC ONLY
   when `max(|prior_fsc|) < 1e-3` (cold-start fallback). Algorithm doc:
   `docs/math/relion_updateSSNR_algorithm_2026_04_25.md`.

## Investigation results — iter-1 deficit (NOT YET RESOLVED)

Two-phase empirical investigation (Phase B: parity_dump + per-particle
compare; Phase C: single-particle diff² dump from RELION's CPU code path).

### What's confirmed

- iter-1 deficit is **uniform across particles** (per-particle correlation
  0.94 with RELION). Systematic offset, not pose-search failure.
- recovar's score formula is mathematically equivalent to RELION's diff²
  formula, modulo a per-particle constant `0.5 * exp_highres_Xi2 + 0.5 *
  sum |Fimg|² * Minvsigma2` that cancels in the softmax.
- For ONE particle (particle 0), recovar reproduces RELION's per-particle
  pmax to 4 decimals (0.6047 vs 0.6049) — so the per-particle formula match
  is OK in this case.
- Translation prior in
  `orientation_priors.py:make_relion_translation_log_prior` is dimensionally
  correct (`log_prior = -0.5 * sqdist_ang / sigma²` where `sqdist_ang =
  (translations_pixel * voxel_size)²` is in Å²). Tests with artificial
  scaling factors compensate but make formulas dimensionally wrong.

### What was attempted and DIDN'T close the gap

- **Bug #1 hypothesis** (Phase C agent): use sigma_offset from iter-(K-1)
  model.star (=10Å for cold start) instead of iter-K (=2.106Å). Per RELION
  source `ml_optimiser.cpp:5791-5808`, sigma2_offset only updates AT END of
  M-step, so iter-K E-step uses iter-(K-1) value. **Empirical result on
  tiny/1000 iter 0→1**: gap went from -17.6% to **-59.3%** (much worse).
  RELION-exact source-reading but doesn't compose with the rest of recovar's
  iter-1 code. Reverted.

- **Bug #2 hypothesis** (Phase C agent): add `pixel_size²` factor in
  `orientation_priors.py:51` per RELION's `tdiff2 *= pixel_size²`.
  Dimensional analysis shows recovar's formula was already RELION-equivalent
  (translations in pixels × voxel_size = Å, then squared). The agent's patch
  added an extra `voxel_size²` factor, dimensionally wrong. Reverted.

- **Both fixes together** (sigma=10 + extra voxel_size²): gap -12.4%,
  partially compensating each other but not actually correct.

### Best known config

`--adaptive_fraction 1.0` alone (no source patches) gives gap **-6.8%** on
tiny/1000 iter 0→1. This points the residual ~7pp gap (after sparse-pass-2
is bypassed) at something in the sparse pass-2 normalization itself or in
the score-tensor logsumexp.

### Hypotheses still open

1. Sparse-pass-2 logsumexp uses a different candidate set than RELION's
   full-grid sum, contributing a normalization-constant offset of ~log(0.93)
   = -0.073 in log_Z. The af=1.0 result removes ~11pp, supporting this.
2. Some other multiplicative scale in the per-pose diff² that wasn't isolated
   for the SAME particle that recovar handles correctly. Need single-particle
   diff² dump from a "deficit particle" (where recovar pmax << RELION pmax),
   not just particle 0 which happens to match.

## Open follow-up branches (waiting for parity bug to fully close)

| PR | Branch | Title | Tracked in |
|---|---|---|---|
| #118 | `claude/dense-cleanup-relion-only` | -1302 LOC dense_single_volume cleanup | #114 |
| #119 | `claude/parity-perf-baseline` | per-stage timers + check_perf.py | #115 |
| #120 | `claude/parity-quality-baseline` | fast (~5 min) parity test suite | #116 |

The actual parity bug fix is tracked in #117.

## Diagnostic harness available for next session

- `recovar/em/dense_single_volume/parity_dump.py` — env-gated per-iter dump
- `scripts/parity/dump_relion_iter.py` — RELION reference dumper
- `scripts/parity/compare_dumps.py` — per-iter parity report
- RELION patched binary at `/scratch/gpfs/GILLES/mg6942/relion/build_patched/bin/relion_refine_mpi`
  with `RELION_DUMP_DIR` infra (CPU dump path; GPU bypasses it)
- Tiny fixture at `/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_tiny_parity/`
  for sub-30-second iters with `--max_particles 100`

## Slurm / GPU artifacts (today's work)

- Worktrees with intermediate state (under SAFE_TO_DELETE):
  - `recovar_iter1_debug_130128/` — Phase A/B/C agent
  - `recovar_bisect_parity_*/` — bisect agent worktrees (general + codex-specific)
  - `recovar_iter1_diff_*/` — Phase C empirical dump comparison
  - `recovar_perf_baseline_*`, `recovar_parity_quality_baseline_*`,
    `recovar_sparse_pass2_fix_*`, `recovar_dense_cleanup_*` — agent branches
- Run logs: `/tmp/p2p4_v2.log`, `/tmp/late_v3.log`, `/tmp/iter1fix.log`,
  `/tmp/sigma_only.log`, `/tmp/af1.log`, etc.
- RELION dump artifacts:
  `recovar_iter1_debug_130128/_agent_scratch/relion_dump_C1_p0/`
  (one-particle iter-1 RELION reference)

## New finding (post-revert): deficit is roughly multiplicative

**Sorted-distribution comparison** (recovar's iter1_full_af1 dump vs RELION
iter-1 data.star, 992 particles, both halves, with `--adaptive_fraction=1.0`
on recovar):

| quantile | rec_h1 | rel_h1 | gap_h1 | rec_h2 | rel_h2 | gap_h2 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.071 | 0.076 | -0.005 | 0.061 | 0.067 | -0.007 |
| 0.50 | 0.198 | 0.208 | -0.011 | 0.184 | 0.206 | -0.022 |
| 0.95 | 0.516 | 0.544 | -0.028 | 0.516 | 0.564 | -0.048 |
| 0.99 | 0.706 | 0.741 | -0.036 | 0.702 | 0.757 | -0.055 |

Pattern: ratio ≈ 0.93 across the distribution → recovar's pmax is
multiplicatively suppressed by ≈ exp(-0.073). That's a constant log_Z
offset of ~+0.073 (recovar's log_Z is ~7% bigger than RELION's), meaning
recovar's softmax is wider — extra weight scattered into off-MAP poses.

**The bug is in the dense `run_em` path, not specifically in sparse
pruning**: even with `--adaptive_fraction=1.0` (which bypasses
`compute_pass2_stats_sparse` entirely and routes through dense `run_em`),
the 7% deficit persists.

A constant log_Z offset is consistent with:
- A pose-independent normalization factor recovar applies but RELION doesn't
  (or vice versa) — appears in log_Z but cancels in best_log_score, giving
  smaller pmax.
- The candidate-set size (number of poses entering the partition function)
  differs by ~exp(0.073) ≈ 7% extra candidates in recovar's sum.

## Recommended next actions

1. **Per-pose diff² dump for a "deficit particle"** (where recovar pmax <<
   RELION pmax). Particle 0 happens to match RELION's pmax to 4 decimals
   (0.6047 vs 0.6049), so it can't reveal a uniform multiplicative
   deficit. Pick a particle where recovar pmax ≈ 0.05 but RELION pmax ≈
   0.5 (there are 153 such on tiny/1000), instrument recovar's
   `_e_step_block_scores_*` to dump per-pose diff² for that particle, and
   compare element-wise with RELION's `exp_Mweight_diff2.bin` from the
   patched binary CPU dump (`RELION_DUMP_DIR` + `RELION_DUMP_STACK_INDEX=N`).
   The (rot, trans)-dependent deviation that averages to zero for
   particle 0 must appear for the deficit particle.
2. **Count the candidate-set size**: if recovar's logsumexp covers ~7%
   more candidates than RELION's, the gap is explained. Check how many
   poses RELION's `exp_thisimage_sumweight` actually sums over (it skips
   poses with `weight == 0` after `exp_min_diff2` subtraction —
   `ml_optimiser.cpp:8736-8772`). Recovar may not skip the same set.
3. **Open issue/branch for the residual iter-1 gap** as next work — bug
   tracker reference: this status doc + algorithm doc
   `docs/math/relion_updateSSNR_algorithm_2026_04_25.md`.
4. **Once iter-1 is closed**, the cold-start trajectory should track
   RELION within < 1% across all iters (per the `--iter 2` test which gave
   +0.27% at iter 4 when starting from RELION's iter-2 state).

## Mid-session candidates that turned out NOT to be the bug

For future agents debugging the same gap, ruled-out candidates with evidence:

- **sigma_offset source mismatch** (Phase C "Bug #1"): RELION uses
  iter-(K-1) pre-update value; recovar reads iter-K post-update. Empirical
  test forcing sigma=10Å made gap WORSE (-59%). Ruled out as primary cause.
- **`pixel_size²` factor in translation prior** (Phase C "Bug #2"):
  dimensional analysis shows recovar's formula was already RELION-equivalent.
  Ruled out.
- **`ini_high` low-pass filter mismatch**: filter is gated by
  `relion_firstiter_cc_this_iter` which is False on this fixture; the
  init reference is already smooth above shell 10. Ruled out.
- **noise_variance N⁴ scaling**: algebraically correct (cancels FFT
  normalization). Ruled out.
- **Per-particle pose mismatch**: recovar identifies the same MAP pose as
  RELION for ≥ 95% of particles (Eulers within 1°). Ruled out.

The diagnostic harness (parity_dump + dump_relion_iter + compare_dumps +
RELION patched binary) is in place for the next agent to do (1) directly.

## 2026-04-26 update: tau2/FSC boundary source fixes and long-run baseline

### Source-level fixes made

Two more RELION boundary semantics were matched in RECOVAR:

- `BackProjector::updateSSNRarrays` computes shell-wise reconstruction
  sigma/tau only over padded voxels satisfying
  `r2 < ROUND(r_max * padding_factor)^2`, with
  `r_max=current_size/2`. RECOVAR now passes the current `r_max` into
  `compute_relion_tau2_from_weights` and excludes weights outside that
  support.
- `BackProjector::calculateDownSampledFourierShellCorrelation` first skips
  exact native radii `R > r_max`, then bins surviving voxels by `ROUND(R)`.
  RECOVAR previously admitted the outer half of rounded boundary shells; it
  now applies the exact-radius gate before shell binning.

Targeted tests:

- `.pixi/envs/default/bin/python -m pytest tests/unit/test_regularization.py -k 'relion_fsc_from_backprojector or tau2_from_weights'`
- Result: `6 passed, 19 deselected`.

### Tiny 1k / 64³ replay after the fix

Artifact:
`_agent_scratch/codex_pmax_sentinels_fsc_rmax_20260426_185332_27278`
(local A100 GPU 3, `JAX_ENABLE_X64=1`).

Direct accumulator diagnostics against the previous sentinel run:

| shell | old tau | new tau | RELION tau*N^4 | note |
|---:|---:|---:|---:|---|
| 27 | 9.3655 | 9.5955 | 8.7625 | FSC boundary gate moves shell 27 toward RELION but does not fully close it |
| 28 | 494.05 | 1e-16 | 0.0 | fixed: false shell-28 support removed |

Replay metrics:

| Iter | Mean Abs Pmax Gap | Max Abs Pmax Gap | Corr | Pose status |
|---:|---:|---:|---:|---|
| 1 | 3.5e-5 | 8.36e-4 | 1.000000 | exact |
| 2 | 0.005059 | 0.276175 | 0.957262 | top outliers mostly same pose/translation |

Final map parity improved:

- `recovar-vs-RELION` final corr `0.999998` (was `0.999992` before this
  boundary fix).
- GT map quality remains same shell-level result as RELION:
  merged FSC<0.5 shell 22 and FSC<0.143 shell 27.

Top iter-2 Pmax outliers after the fix:

| particle | RECOVAR | RELION | gap |
|---:|---:|---:|---:|
| 933 | 0.945082 | 0.668907 | +0.276175 |
| 725 | 0.624497 | 0.878554 | -0.254057 |
| 843 | 0.634158 | 0.820172 | -0.186014 |
| 612 | 0.714027 | 0.538589 | +0.175438 |
| 412 | 0.572801 | 0.426704 | +0.146097 |

Pose check for these particles: angular and translation differences are
numerical-noise level for p933, p725, p843, p612, p412, p924, p672, p371,
p825, p566, p274, p668, p375, and p607. Particle p88 is a real pose outlier
(`8.51 deg`). Therefore the dominant residual at iter 2 is confidence/score
gap on the same pose, not missing candidate enumeration.

### 5k / 128³ long end-to-end run

Artifact:
`_agent_scratch/long_end2end_parity_20260426_182134`
(Slurm job `7383509`, A100 node `della-l07g4`, elapsed `2075.6s`, branch
`claude/relion-parity-local-search-fix`, commit
`949ab6b84a40bab5011024689c15492414c4e6ce`).

Final map parity:

- half1 corr vs RELION: `0.996346`
- half2 corr vs RELION: `0.996437`

Pmax trajectory:

| RELION iter | RECOVAR Pmax | RELION Pmax | Mean Abs Gap | Max Abs Gap | Corr |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.646471 | 0.646337 | 0.001090 | 0.461337 | 0.997458 |
| 3 | 0.964680 | 0.964374 | 0.006342 | 0.333938 | 0.975304 |
| 4 | 0.973809 | 0.973470 | 0.006537 | 0.464203 | 0.951907 |
| 5 | 0.973248 | 0.972628 | 0.006203 | 0.466193 | 0.960749 |
| 6 | 0.930572 | 0.928947 | 0.013834 | 0.434564 | 0.963276 |
| 7 | 0.954587 | 0.949317 | 0.019198 | 0.495957 | 0.882088 |
| 8 | 0.885187 | 0.885416 | 0.034731 | 0.532186 | 0.895687 |
| 9 | 0.899029 | 0.896648 | 0.042479 | 0.573372 | 0.833259 |

Pose trajectory remains much closer than Pmax:

| RELION iter | angular mean/p99/max deg | trans mean/p99/max px |
|---:|---:|---:|
| 2 | 0.0262 / 0.0000 / 19.2185 | 0.0072 / 0.0000 / 1.0000 |
| 3 | 0.0839 / 0.0642 / 11.2321 | 0.0068 / 0.0000 / 1.0000 |
| 4 | 0.0748 / 0.0000 / 13.1587 | 0.0056 / 0.0000 / 1.0000 |
| 5 | 0.0667 / 0.0001 / 14.7755 | 0.0048 / 0.0000 / 1.0000 |
| 6 | 0.0584 / 3.7174 / 8.0290 | 0.0003 / 0.0000 / 0.5351 |
| 7 | 0.0783 / 3.9671 / 7.9590 | 0.0017 / 0.0000 / 0.5351 |
| 8 | 0.1018 / 2.3575 / 5.5774 | 0.0007 / 0.0000 / 0.3053 |
| 9 | 0.1223 / 2.3925 / 5.2568 | 0.0010 / 0.0000 / 0.3053 |

Next trace target: use RELION `RELION_DUMP_BPREF=1` / `RECOVAR_MSTEP_DUMP_DIR`
or a direct in-memory RECOVAR dump to compare the exact map/Fourier
coefficients used for scoring, not post-written MRC projections. Saved MRCs
are close enough for map correlation, but not exact enough to diagnose
sub-unit score gaps on ambiguous two-candidate particles.

## 2026-04-26 late update: fixed matched-row reporting and early convergence cause

### Why `--max_iter 13 --skip_final_iteration` produced only 8 RECOVAR rows

`--max_iter` is an upper bound in RELION-mode RECOVAR replay. The iteration
loop stops when `state.has_converged` is true unless fixed-length diagnostics
are explicitly requested. The previous report incorrectly asked the diff script
for `max_iter + 1` rows even when RECOVAR emitted fewer rows, so RELION-only
iters 10-14 appeared without matching RECOVAR metrics.

Harness fixes:

- `scripts/run_multi_iter_parity.py` now records `completed_iterations` and
  calls `diff_relion_recovar_per_iter.py` only through the emitted RECOVAR
  iteration count.
- `scripts/diff_relion_recovar_per_iter.py` now caps output to
  RELION-init plus matched RECOVAR rows and prints an explicit note when fewer
  rows were emitted than requested.
- `refine_single_volume(..., force_max_iter_after_convergence=True)` can be
  used for fixed-length diagnostics after convergence, but should not be used
  to hide a real convergence-control mismatch.

Validation on the 5k long artifact:

- Artifact:
  `_agent_scratch/long_end2end_parity_20260426_182134/diff_matched_rows_after_harness_fix.txt`
- New report note:
  `recovar emitted 8 iteration rows; showing RELION init + matched rows only
  (requested 14).`

### Root cause of premature convergence at the 8th replay step

The row-count mismatch exposed a real control-flow parity bug. RECOVAR stopped
after its 8th replay step because its local-search `acc_rot` estimate was
stale and too large. At HEALPix order 5, this made
`effective_step < 0.75 * acc_rot` true, so the convergence check considered
angular sampling fine enough. RELION did not.

Measured against RELION `run_it009_optimiser.star`:

- Stale RECOVAR-style control value: `acc_rot ~= 6.1237 deg`
- RELION source-of-truth field: `_rlnOverallAccuracyRotations = 1.047 deg`
- At HEALPix order 5, `effective_step = 1.875 deg`
- With stale `6.1237`, `1.875 < 0.75 * 6.1237` -> falsely fine enough.
- With RELION `1.047`, `1.875 < 0.75 * 1.047` is false -> continue.

Fix:

- Added `read_relion_optimiser_metadata()` in `recovar/em/sampling.py`.
- In RELION replay mode, `iteration_loop.py` now reads
  `run_it{init + iteration + 1}_optimiser.star` and overrides
  `iter_acc_rot` with `_rlnOverallAccuracyRotations`.
- It also reads `_rlnOverallAccuracyTranslationsAngst` and passes the value
  converted to pixels as `acc_trans` for sampling-state parity.

Validation:

- `py_compile`:
  `.pixi/envs/default/bin/python -m py_compile recovar/em/sampling.py recovar/em/dense_single_volume/iteration_loop.py`
- Unit:
  `.pixi/envs/default/bin/python -m pytest tests/unit/test_convergence.py -q`
  -> `56 passed`.
- Direct control check:
  stale `acc_rot=6.1237` gives `fine=True, converged=True`;
  RELION `acc_rot=1.047` gives `fine=False, converged=False`.

### M-step BPref boundary status

A RELION BPref dump was compared against RECOVAR native half-volume M-step
accumulators using the correct axis/scaling map:

- RELION BPref loop coords `(z, y, x)` map to RECOVAR axes `(x, y, z)`.
- `RECOVAR Ft_y = - RELION bpref_data / 4096` for `N=64`.
- `RECOVAR Ft_ctf = RELION bpref_weight / 4096^2`.

With that mapping, half2 post-lowres-join accumulator parity is essentially
exact through shell 26. The remaining mismatch is concentrated at the cutoff
band for `rmax=27`:

- Shells 20-25: weight/data correlations are effectively `1.0`, relative
  errors around `1e-4` to `2e-4`.
- Shell 26: weight relative error `7.76e-4`, data relative error `1.48e-3`.
- Shell 27: sparse/outlier-heavy boundary disagreement; inside exact
  `r <= 27`, weight relative error `0.072`, data relative error `0.124`.
- Shell 28 is outside exact support and should not be used as a parity target
  for this BPref comparison.

Interpretation: the M-step is not broadly wrong. The remaining accumulator
gap is a cutoff-boundary scatter issue around radii `26.5-27.5`, and most
voxels in that band still match. It can still flip Pmax for ambiguous
particles because the score margin is small, but it should be investigated as
boundary arithmetic/source-window semantics, not as a normalization or prior
bug.

### Replay local-search sigma must follow RELION `sampling.star`

The optimiser-accuracy fix let RECOVAR continue beyond the old false
convergence point, but the next replay attempt exposed a second control-flow
parity bug: local-search sigma was not refreshed when RELION advanced to finer
HEALPix orders during replay. RECOVAR kept the stale hp4 local prior sigma
(`7.5 deg`) when RELION had already advanced to hp5/hp6. That inflated the
local-search cone and caused the previous hp6 replay attempt to hit a GPU OOM.

RELION writes the source of truth in each `run_it*_sampling.star`:

- hp4: `_rlnPsiStep = 3.75`; local sigma should be `2 * psi_step = 7.5 deg`.
- hp5: `_rlnPsiStep = 1.875`; local sigma should be `3.75 deg`.
- hp6: `_rlnPsiStep = 0.9375`; local sigma should be `1.875 deg`.

Fix:

- `read_relion_sampling_metadata()` now reads `_rlnPsiStep`.
- In RELION replay mode, `iteration_loop.py` updates `state.sigma_rot` and
  `state.sigma_psi` to `2 * _rlnPsiStep` whenever local search is active.

Validation on a 200-particle control replay:

- Artifact:
  `_agent_scratch/codex_sigma_replay_subset_20260426_235523_2471036`.
- Command shape:
  `scripts/run_multi_iter_parity.py --max_iter 9 --max_particles 200
  --skip_final_iteration --local_engine exact_v2`.
- The run emitted 9 RECOVAR rows, reached hp6, and completed without the
  previous hp6 allocation failure.
- Logs show the required replay transitions:
  hp4 -> hp5 changed local sigma `7.500/7.500 deg -> 3.750 deg`
  from RELION `psi_step=1.875`, and hp5 -> hp6 changed
  `3.750/3.750 deg -> 1.875 deg` from `psi_step=0.9375`.

Important caveat: the 200-particle replay is only a control-flow check. After
the first M-step, it reconstructs from 200 particles while the RELION maps and
metadata are from the full 5000-particle run, so post-iter-1 map quality and
Pmax gaps from that run are not valid quality-parity metrics. The fair
full-data 5k artifact before the convergence fix already showed strong parity
through RELION iter 9:

- recovar-vs-RELION map correlation stayed `0.99982` or better through iter 9.
- RECOVAR/RELION GT map correlations matched to about `1e-4`.
- Pose deltas stayed small: mean angular error `<= 0.1223 deg` and mean
  translation error `<= 0.0072 px` through iter 9.

The first fair fixed-length 5k replay was submitted as Slurm job `7391440`:

- Output dir:
  `_agent_scratch/long_5k_force13_scores_20260427_000907`.
- Command shape:
  `scripts/run_multi_iter_parity.py --max_iter 13 --skip_final_iteration
  --force_max_iter_after_convergence`.
- Goal: verify matched rows through the higher iteration range instead of
  treating RELION-only rows as parity failures.
- It also enables `RECOVAR_LOCAL_SCORE_DUMP_GLOBAL_INDICES=1325,659,1865`
  at `current_size=84` to capture exact-local raw scores for high-iteration
  Pmax outliers whose poses/translations are already matched.
- This job predates the 2026-04-27 local-prior metadata fix below, so use it
  for row-count/control-flow coverage, but do not treat local-search Pmax gaps
  from it as current conclusions. It was cancelled after reaching RECOVAR
  iteration 10 because it had imported the stale local-prior code and already
  showed a bad iter-9 ave_Pmax (`0.7164`).

Replacement fixed-length 5k replay after the local-prior fix:

- Slurm job: `7392673`.
- Output dir:
  `_agent_scratch/long_5k_force13_priorfix_20260427_0158`.
- Command shape:
  `scripts/run_multi_iter_parity.py --max_iter 13 --skip_final_iteration
  --force_max_iter_after_convergence --local_engine exact_v2`.
- Local score dumps enabled:
  `RECOVAR_LOCAL_SCORE_DUMP_GLOBAL_INDICES=659,1325,1865`,
  `RECOVAR_LOCAL_SCORE_DUMP_CURRENT_SIZE=82,84`,
  `RECOVAR_LOCAL_SCORE_DUMP_ITERATION=1,7,8,9,10,11,12,13`.
- Status: cancelled as stale after the later translation-prior pixel-unit
  bug was found. Do not use this job for current parity conclusions; relaunch
  fixed-length 5k replay from the grid-order/translation-prior code.

### Late local-search Pmax denominator hypothesis

Source reading in RELION `src/acc/acc_ml_optimiser_impl.h` and
`src/acc/acc_helper_functions_impl.h` shows that final local-search Pmax is:

- `Pmax = op.max_weight / op.sum_weight`.
- In the fine pass, `op.sum_weight` is the sum over `PassWeights.weights`.
- `PassWeights` is restricted by `FPCMasks`, and `FPCMasks` is built by
  `makeJobsForDiff2Fine()` from coarse significant hidden
  orientation/translation samples (`op.Mcoarse_significant`), not simply from
  every pose in a Gaussian local-prior cone.

Current RECOVAR exact-local normalization in `local_score_pass.py` computes
`log_Z` over every pose in the per-image `LocalHypothesisLayout`. That layout
is source-matched to RELION's local prior support only when its metadata is
built from the canonical factorized Healpix/psi axes
(`selectOrientationsWithNonZeroPriorProbability`: direction and psi within
`3 * sigma`). It is not yet proven to match RELION's final
`FPCMasks/PassWeights` membership. Therefore, if the fixed sigma/accuracy run
still shows large Pmax gaps while hard poses and translations match, the next
debug target is candidate-set/denominator parity:

- Compare RECOVAR raw scores for the selected image against RELION dumped
  `PassWeights`.
- Check whether the winning raw residual matches but RECOVAR and RELION sum
  over different candidate sets.
- If candidate sets differ, implement the RELION two-stage local membership
  path rather than changing priors, thresholds, or other parameters.

Debug plumbing update:

- `RECOVAR_LOCAL_SCORE_DUMP_CURRENT_SIZE` now accepts comma/space-separated
  sizes.
- `RECOVAR_LOCAL_SCORE_DUMP_ITERATION` can target emitted RECOVAR iteration
  numbers directly, avoiding the ambiguity that multiple local-search
  iterations can share the same `current_size`.
- Score dumps are written as
  `local_score_itNNN_image_<original_index>.npz` and include
  `debug_iteration`, `current_size`, raw scores, total scores, posteriors,
  priors, candidate masks, reconstruction masks, and local rotation IDs.

### 2026-04-27 local-search prior metadata fix

The first high-iteration focused failure case was particle 659 at the RELION
iter-8 -> iter-9 local E-step. Before the fix, RECOVAR had the same best
pose/translation and the same four active candidate raw residuals as RELION,
but Pmax differed because the local rotation priors were normalized on the
wrong grid:

- RELION source behavior: `selectOrientationsWithNonZeroPriorProbability()`
  evaluates direction and psi priors on the canonical factorized HEALPix
  direction array and the canonical psi array, normalizes those priors
  separately, and only then `SamplingPerturbation` changes the concrete
  orientations scored in the accelerated kernels.
- Old RECOVAR behavior: local-search metadata was built from the already
  perturbed full Euler table. That table no longer factorizes by direction and
  psi, so RECOVAR normalized a single full-grid prior over all candidate
  rotations.
- Fix: both grouped and exact-v1 local-search paths now call
  `build_local_search_grid_metadata(healpix_order)` for canonical factorized
  prior/layout metadata while still passing the perturbed rotation matrices to
  the scoring engine.

Dump evidence:

- RELION forced-sampling dump:
  `_agent_scratch/relion_dump_stack659_it008_forced_sampling_20260427/dump`.
- Forced RELION env values matched the original run:
  `_rlnSamplingPerturbInstance=-0.11871`, offset range `4.430302`, offset step
  `1.297312`, psi step `1.875`.
- RELION candidate denominator count for particle 659 was `4`; the candidates
  matched RECOVAR's top four, all at translation index `18`.
- RELION and RECOVAR raw residual deltas matched at arithmetic level; the
  prior/logit mismatch caused the Pmax gap.
- RELION candidate priors for the four candidates were approximately
  `[-5.626974, -5.267148, -8.407798, -5.112989]`, with normalized weights
  `[0.039600, 0.882781, 0.071479, 0.006182]`.

Focused validation after the fix:

- Artifact:
  `_agent_scratch/focused_it008_priorfix_3img_20260427`.
- Command shape:
  `scripts/run_multi_iter_parity.py --iter 7 --max_iter 1
  --keep_stack_indices 659,1325,1865 --local_engine exact_v2
  --skip_final_iteration`.
- The selected 3-image replay gave ave_Pmax RECOVAR `0.765738` vs RELION
  `0.765523`, mean abs Pmax gap `2.54e-4`, max abs gap `6.53e-4`, and
  Pmax correlation `0.999997`.
- Particle 659 is now closed at arithmetic level: RECOVAR `0.8829320669` vs
  RELION `0.882882`, gap about `5e-5`.
- Particle 1325 is also arithmetic-level: RECOVAR `0.7443317771` vs RELION
  `0.744390`, gap about `-5.8e-5`.
- Particle 1865 remains a small near-target residual: RECOVAR
  `0.6699510217` vs RELION `0.669298`, gap `6.53e-4`. Pose/translation still
  match exactly; inspect raw residual arithmetic before changing any
  algorithmic behavior.

Targeted regression test:

- `.pixi/envs/default/bin/python -m pytest
  tests/unit/test_refine_relion_mode.py::test_run_local_search_iteration_exact_engine_uses_factorized_prior_metadata_for_perturbed_grid`
  verifies that a perturbed full Euler table would be `"full"` metadata, but
  `_run_local_search_iteration_exact_v1()` still passes RELION-style
  `"factorized"` metadata to the local layout builder while scoring perturbed
  rotations.

Focused one-particle diagnostic path:

- Artifact:
  `_agent_scratch/focused_it008_priorfix_659_retry4_20260427_015232`.
- This selected only particle 659, so half2 was empty; map/tau2/noise metrics
  from that run are not meaningful. It is useful only as a tight E-step/Pmax
  diagnostic.
- Result: RECOVAR `0.8829320669` vs RELION `0.882882`, gap
  `5.0067e-5`, with exact pose and translation parity.
- Supporting fixes: empty per-half image/scale correction logging no longer
  calls min/max on empty arrays; empty half-sets now produce zero
  accumulators/empty stats; empty translation pre-shift bases normalize to
  shape `(0, 2)`.

### 2026-04-27 translation-prior units and translation-index ordering

Particle 1865 exposed a second local-search issue after the factorized prior
metadata fix. RELION's forced dump for the iter-8 local E-step included a
one-step offset candidate at translation index `11` with offset prior
`-7.599875`. RECOVAR initially assigned that candidate a prior near `-137`,
which suppressed its posterior mass.

Root cause and fix:

- `_replay_prior_translations` was built from raw RELION
  `_rlnOffsetRange/_rlnOffsetStep` Angstrom values.
- `make_relion_translation_log_prior()` expects pixel-valued translations and
  internally applies RELION's Angstrom conversion plus the source-code
  `pixel_size²` sharpening.
- The replay path now converts range/step to pixels before calling
  `get_translation_grid()`.

Validation artifacts:

- Pixel-unit validation:
  `_agent_scratch/focused_it008_translationpriorfix_3img_20260427_021012`.
- Grid-order validation:
  `_agent_scratch/focused_it008_gridorder_translationpriorfix_3img_20260427_021817`.

Focused 3-particle result after both fixes:

- Particle 659: RECOVAR `0.8828698993`, RELION `0.882882`, gap
  `-1.21e-5`.
- Particle 1325: RECOVAR `0.7439696193`, RELION `0.744390`, gap
  `-4.20e-4`.
- Particle 1865: RECOVAR `0.6689230800`, RELION `0.669298`, gap
  `-3.75e-4`.
- Pose and translation assignments are exact for all three selected
  particles.

Translation-grid ordering:

- RELION `HealpixSampling::setTranslations()` loops with `x` outer and `y`
  inner.
- RECOVAR's `get_translation_grid()` now uses the same order. This does not
  change the hypothesis set or selected pose, but it makes `itrans` labels
  source-identical in dumps.
- `tests/unit/test_relion_bind/test_s2_translations.py` now checks RELION
  and RECOVAR translation grid order directly.

Forced RELION p1865 candidate check:

- RELION forced dump:
  `_agent_scratch/relion_dump_stack1865_it008_forced_sampling_20260427b/dump`.
- After the grid-order fix, RECOVAR `trans[11]` is
  `[-0.3414861, -0.0362362]` with prior `-7.599875`, matching RELION's
  candidate.
- Candidate posterior for this off-center sample is RECOVAR `0.001279457`
  vs forced RELION `0.001303128`.
- The dominant five forced RELION candidate weights match RECOVAR within the
  expected small residuals; the original RELION optimiser Pmax for p1865
  (`0.669298`) is closer to RECOVAR (`0.668923`) than the forced continuation
  dump (`0.672077`), so treat the forced dump as candidate/source evidence,
  not as the authoritative original Pmax target.

Replacement fixed-length 5k replay from current code:

- Slurm job: `7393212`.
- Output dir:
  `_agent_scratch/long_5k_force13_gridorder_translationpriorfix_20260427_0222`.
- Log:
  `_agent_scratch/long_5k_force13_gridorder_translationpriorfix_20260427_0222/slurm-7393212.out`.
- Command shape:
  `scripts/run_multi_iter_parity.py --max_iter 13 --skip_final_iteration
  --force_max_iter_after_convergence --local_engine exact_v2`.
- Local score dumps enabled for particles `659,1325,1865` at current sizes
  `82,84` and emitted iterations `1,7,8,9,10,11,12,13`.

### Focused RELION-state iter-9 E-step result

To separate E-step/local-denominator errors from accumulated M-step state
drift, a focused replay was run directly from RELION iteration 8 into the
RELION iteration 9 E-step:

- Artifact:
  `_agent_scratch/focused_it009_scores_20260427_003738`.
- Command shape:
  `scripts/run_multi_iter_parity.py --iter 8 --max_iter 1
  --keep_stack_indices 1325,659,1865 --local_engine exact_v2
  --skip_final_iteration`.
- The run used RELION iter-8 maps/noise/metadata and dumped RECOVAR
  exact-local scores for iter 9.

Selected-particle Pmax comparison:

- Particle 659: RECOVAR `0.9997831583`, RELION `0.999776`, gap `7.2e-6`.
- Particle 1325: RECOVAR `0.9977205992`, RELION `0.997665`, gap `5.6e-5`.
- Particle 1865: RECOVAR `0.4278309345`, RELION `0.421312`, gap `6.5e-3`.

Update after the 2026-04-27 translation-prior and grid-order fixes:

- Artifact:
  `_agent_scratch/focused_it009_gridorder_translationpriorfix_3img_20260427_022452`.
- Particle 659: RECOVAR `0.9997761846`, RELION `0.999776`, gap
  `1.85e-7`.
- Particle 1325: RECOVAR `0.9976645708`, RELION `0.997665`, gap
  `-4.29e-7`.
- Particle 1865: RECOVAR `0.4215057194`, RELION `0.421312`, gap
  `1.94e-4`.
- Mean Pmax gap over these three particles is `6.4e-5`, max abs gap is
  `1.94e-4`, and pose/translation assignments are exact.
- Conclusion: with RELION's iter-8 state, local E-step/Pmax parity through
  RELION iter 9 is now near target; investigate full-run divergence as
  accumulated state drift, not local candidate enumeration.

The same selected particles in the full RECOVAR replay at `current_size=84`
had much larger score-landscape differences despite the same best
pose/translation IDs:

- Particle 1325: full replay Pmax `0.831682` vs focused RELION-state Pmax
  `0.997721`; best rotation/translation matched.
- Particle 1865: full replay Pmax `0.961688` vs focused RELION-state Pmax
  `0.427831`; best rotation/translation matched.
- Particle 659 stayed matched (`0.999614` full replay vs `0.999783`
  focused).

Interpretation: for these late-iteration examples, exact-local E-step
normalization and pose enumeration are not the dominant failure when RECOVAR
is given RELION's state. The large full-replay Pmax gaps are caused by
accumulated state drift before RELION iter 9, most likely in the M-step/map
regularization/noise/tau path. Map comparison of RECOVAR `it006_half*_reg.mrc`
against RELION `run_it008_half*_class001.mrc` after sign alignment shows
global correlation about `0.99980`, but shell-relative map errors rise from
about `0.3-0.7%` in low/mid shells to about `3%` near shell 40. Those small
map differences are enough to sharpen or flatten ambiguous local posterior
landscapes.

Next target: trace the state update from the previous matched E-step into the
regularized half-map used at RELION iter 9. Compare `Ft_y`, `Ft_ctf`, tau2,
noise, gridding correction, and regularized map shells for iter 6 -> 8 before
spending more effort on local candidate-set parameters.

### 2026-04-27 fixed-length row counts and accelerated `pdf_offset` units

The `--max_iter 13 --skip_final_iteration` report that showed only eight
RECOVAR rows but RELION-only rows through later iterations was a reporting/run
control issue, not proof of late-iteration parity by itself:

- `max_iter` is an upper bound; RECOVAR stopped after convergence.
- `skip_final_iteration` only skips RELION's final all-data/Nyquist iteration.
- Older reports still printed RELION rows past the RECOVAR emitted rows.
- For fixed-length diagnostics, use `--force_max_iter_after_convergence` and
  only compare RELION rows with matching RECOVAR rows.

The previous forced 13-iteration Slurm job `7393212` is not a valid parity
result. It failed after two emitted RECOVAR iterations with CUDA illegal
address cleanup errors:

- Output dir:
  `_agent_scratch/long_5k_force13_gridorder_translationpriorfix_20260427_0222`.
- Log:
  `_agent_scratch/long_5k_force13_gridorder_translationpriorfix_20260427_0222/slurm-7393212.out`.

Particle 256 then exposed a source-level accelerated-path unit mismatch in
RELION's translation prior:

- RELION dump path:
  `_agent_scratch/relion_dump_stack257_it008_20260427_030029/dump`.
- RECOVAR focused pre-fix path:
  `_agent_scratch/focused_it008_stack256_20260427_025207`.
- RECOVAR focused post-fix path:
  `_agent_scratch/focused_it008_stack256_priorfix_20260427_030957`.

RELION `acc_ml_optimiser_impl.h` builds `pdf_offset` from
`old_offset + sampling.translations - prior`. The `sampling.translations`
array is in Angstrom-space and `getTranslationsInPixel()` divides by
`my_pixel_size` for projection shifts. RECOVAR scores in the pixel-shift
coordinate system, so the prior center must be `(prior - ROUND(old_offset)) /
pixel_size`, while the image pre-shift remains `ROUND(old_offset)`.

Validation for particle 256:

- RELION dump `pass1_pdf_offset` matched the patched RECOVAR prior center with
  max abs error `1.6e-5`.
- Focused post-fix Pmax: RECOVAR `0.3514427`, RELION comparison row
  `0.349307`, gap `0.002136`.
- Pose/translation parity: angular error `8.3e-6` deg, translation error
  `9.2e-7` px.
- Before the fix this same particle had RECOVAR Pmax `0.943291` vs RELION
  `0.349307` because RECOVAR centered the prior one pixel too far in `y`.

Full 5k single-iteration replay from RELION iter 7 -> iter 8 after the fix:

- Artifact:
  `_agent_scratch/recovar_it008_5k_priorfix_20260427_031136`.
- Command shape:
  `scripts/run_multi_iter_parity.py --iter 7 --max_iter 1
  --skip_final_iteration --local_engine exact_v2`.
- Ave Pmax: RECOVAR `0.8854237`, RELION `0.8854160`, gap `7.7e-6`.
- Pmax abs diff: mean `1.23e-4`, median `1.87e-5`, p99 `1.04e-3`,
  max `0.02667`, correlation `0.9999954`.
- Pose parity: full-angle mean `5.7e-6` deg, max `2.0e-5` deg;
  translation mean `1.3e-6` px, max `1.3e-6` px.
- Map parity vs RELION: recovar-vs-RELION corr `0.999952`; GT corr
  RECOVAR `0.965163` vs RELION `0.965239`; both have FSC 0.143 shell `42`.

The remaining largest iter-8 fixed-state Pmax outlier is particle 4603:

- Focused RECOVAR artifact:
  `_agent_scratch/focused_it008_stack4603_priorfix_20260427_031644`.
- RELION dump:
  `_agent_scratch/relion_dump_stack4604_it008_20260427_031754/dump`.
- RECOVAR Pmax `0.606924`, RELION comparison row `0.580255`, gap
  `0.02667`.
- Pose/translation parity is exact to numerical noise.
- Candidate support and priors match RELION. RECOVAR's raw fixed-pose score
  gap between the top two orientations is about `0.136` log units sharper
  than RELION's, so the remaining error is a fixed-candidate scoring/projection
  detail, not pruning or probability normalization.
- A square scoring-window experiment worsened this particle (`0.613056`), so
  do not switch `RELION_FOURIER_WINDOW_SQUARE` to true based only on the
  dumped square operand size. RELION dumps square-sized Fourier operands, but
  the effective scoring support is not simply all square pixels.
- Debug-only RECOVAR operand dumps can be enabled with
  `RECOVAR_LOCAL_SCORE_DUMP_OPERANDS=1`; this adds `debug_shifted_score`,
  `debug_ctf2_over_nv`, `debug_proj_weighted`, and
  `debug_proj_abs2_weighted` arrays to the targeted local-score NPZ. Use this
  only for a small number of particles.

Forced higher-iteration replay now running after the prior-center fix:

- Slurm job: `7393757`.
- Output dir:
  `_agent_scratch/long_5k_force13_priorfix_20260427_033007`.
- Command shape:
  `scripts/run_multi_iter_parity.py --iter 7 --max_iter 13
  --force_max_iter_after_convergence --skip_final_iteration
  --local_engine exact_v2`.
- Status when launched: first emitted iteration completed in `161.9s` with
  iter-8 ave Pmax `0.8854`; job was entering emitted iteration 2
  (`current_size=84`) on `della-h20g2`.
