# EM Parity Best Metrics Ledger

This file tracks accepted best RECOVAR-vs-RELION EM parity runs for the
completion benchmark described in `recovar/em/CLAUDE.md`.

Use this for substantive "done" checks, not every edit-cycle test. The target
completion benchmark is both K=1 and K=4 on at least 100k particles with at
least 256x256 images, compared against RELION for accuracy and speed.

## Update Rules

- Append every completion benchmark attempt with date, commit, worktree, Slurm
  job IDs, fixture, hardware, exact commands, logs, and artifacts.
- Keep the accepted best run table current. If a new run is mixed, record it as
  mixed rather than replacing the accepted best wholesale.
- Every metric delta must say better, worse, or same. Use "same" only for
  noise-level movements that do not change the conclusion.
- Do not record only improved metrics. Include all available quality and
  performance metrics from the run.
- Accuracy and speed are separate axes. A faster run with worse quality, or a
  more accurate run that is slower, is a mixed result unless the tradeoff was
  explicitly intended.

## Accepted Best Runs

K=4 has one accepted >=100k, >=256x256 completion benchmark. K=1 is still
pending in this ledger.

| Case | Date | Commit | Fixture | Particles | Box | RELION baseline | RECOVAR run | Accuracy status | Speed status | Notes |
|------|------|--------|---------|-----------|-----|-----------------|-------------|-----------------|--------------|-------|
| K=1 | pending | pending | pending | >=100k | >=256 | pending | pending | pending | pending | Populate after the first accepted completion benchmark. |
| K=4 | 2026-05-13 | 93a7a365 | ribosembly_k4_g256_n100000_completion_20260512_171123 | 100k | 256 | relion_class3d_k4_it015_clean9d9 iter 1 | recovar_class3d_k4_it001_scoreprep_dirty9cf5f876 | same vs prior accepted quality | better, 1.87x RELION iter time | First accepted K=4 speed run; dirty validation tree was committed unchanged as 93a7a365. |

## Required Metric Template

Use this template for each new completion benchmark.

### YYYY-MM-DD `<short-name>`

Run metadata:

- Commit:
- Branch:
- Worktree:
- Dirty state:
- Fixture:
- Particle count:
- Box size:
- K:
- Initial/reference maps:
- RELION command/log:
- RECOVAR command/log:
- Slurm job IDs:
- Hardware:
- Artifacts:

Quality comparison:

| Metric | Previous best | Current | Delta | Status |
|--------|---------------|---------|-------|--------|
| final_half1_corr_vs_RELION | pending | pending | pending | pending |
| final_half2_corr_vs_RELION | pending | pending | pending | pending |
| merged_corr_vs_RELION | pending | pending | pending | pending |
| recovar_corr_vs_GT | pending | pending | pending | pending |
| relion_corr_vs_GT | pending | pending | pending | pending |
| FSC_0.5_shell_RECOVAR | pending | pending | pending | pending |
| FSC_0.143_shell_RECOVAR | pending | pending | pending | pending |
| Pmax_gap_RECOVAR_minus_RELION | pending | pending | pending | pending |
| Pmax_correlation | pending | pending | pending | pending |
| pose_angle_error_vs_RELION | pending | pending | pending | pending |
| translation_error_vs_RELION | pending | pending | pending | pending |
| K4_class_assignment_or_map_match | pending | pending | pending | pending |

Performance comparison:

| Metric | Previous best | Current | Delta | Status |
|--------|---------------|---------|-------|--------|
| RECOVAR_end_to_end_walltime | pending | pending | pending | pending |
| RELION_end_to_end_walltime | pending | pending | pending | pending |
| RECOVAR_per_iteration_walltime | pending | pending | pending | pending |
| RELION_per_iteration_walltime | pending | pending | pending | pending |
| RECOVAR_images_per_second | pending | pending | pending | pending |
| RELION_images_per_second | pending | pending | pending | pending |
| RECOVAR_peak_gpu_memory | pending | pending | pending | pending |
| RELION_peak_gpu_memory | pending | pending | pending | pending |

Conclusion:

- Overall status:
- Better metrics:
- Worse metrics:
- Same metrics:
- Accepted as new best:

### 2026-05-13 `k4-firstiter-score-probe-speed`

Run metadata:

- Commit: 93a7a365 (`Speed up first-iter K-class score probes`). The Slurm
  validation ran from dirty HEAD 9cf5f876 with the same code changes, then the
  code was committed unchanged.
- Branch: `codex/em-firstiter-kclass-perf-20260513`
- Worktree: `/scratch/gpfs/GILLES/mg6942/recovar_wt_em_completion_9d9ce981_20260512_1926`
- Dirty state during benchmark: `em_engine.py`, `k_class.py`,
  `test_half_spectrum_em.py`, `test_k_class_joint_semantics.py`
- Fixture: `/scratch/gpfs/GILLES/mg6942/em_relion_proj/ribosembly_k4_g256_n100000_completion_20260512_171123`
- Particle count: 100000
- Box size: 256
- K: 4
- Initial/reference maps: `relion_class3d_k4_it015_clean9d9`
- RELION log: `relion_class3d_k4_it015_clean9d9/run.log`
- RECOVAR command/log:
  `/scratch/gpfs/GILLES/mg6942/_agent_scratch/em_perf_profile_20260513_9cf5f876/k4_iter1_profile_wtaskip_100k_256.sh`,
  `recovar_class3d_k4_it001_scoreprep_dirty9cf5f876/run.log`
- Slurm job IDs: RECOVAR 8177614; earlier obsolete profiles 8175369,
  8176853 cancelled, 8177242 cancelled
- Hardware: RECOVAR H100 80GB (`della-h21g4`); RELION memory log reports
  A100 80GB, so absolute RECOVAR-vs-RELION speed is hardware-mixed.
- Artifacts:
  `recovar_class3d_k4_it001_scoreprep_dirty9cf5f876/benchmark_ledger.json`,
  `recovar_class3d_k4_it001_scoreprep_dirty9cf5f876/k4_recovar_gt_eval.json`,
  `recovar_class3d_k4_it001_scoreprep_dirty9cf5f876/gpu_memory.csv`

Quality comparison:

| Metric | Previous best | Current | Delta | Status |
|--------|---------------|---------|-------|--------|
| K4_mean_FSC_1_8_vs_GT | none | 0.995477 | new | accepted |
| K4_per_class_FSC_1_8_vs_GT | none | 0.9961, 0.9952, 0.9955, 0.9951 | new | accepted |
| K4_per_class_FSC_1_16_vs_GT | none | 0.9024, 0.9095, 0.9032, 0.9245 | new | accepted |
| K4_shell_at_FSC_0.5 | none | 19, 19, 19, 20 | new | accepted |
| K4_shell_at_FSC_0.143 | none | 20, 20, 20, 20 | new | accepted |
| K4_map_corr_vs_previous_winner_subset | none | 1.000000, 0.999999967, 1.000000, 1.000000 | new | same |
| Pmax_gap_RECOVAR_minus_RELION | not captured | not captured | n/a | not measured in this full-refine profile |
| pose_angle_error_vs_RELION | not captured | not captured | n/a | not measured in this full-refine profile |
| translation_error_vs_RELION | not captured | not captured | n/a | not measured in this full-refine profile |

Performance comparison:

| Metric | Previous best | Current | Delta | Status |
|--------|---------------|---------|-------|--------|
| RELION_iter1_expectation_plus_maximization | none | 694.752s | new | baseline |
| RECOVAR_iter1_walltime | none | 1296.811s | new | accepted |
| RECOVAR_vs_RELION_iter1_ratio | none | 1.87x | new | accepted, under 2x target |
| RECOVAR_script_walltime | none | 1375s | new | accepted; includes setup/staging |
| RECOVAR_images_per_second_iter | none | 77.1 img/s | new | accepted |
| RELION_images_per_second_iter | none | 143.9 img/s | new | baseline |
| RECOVAR_peak_gpu_memory | none | 41.2 GB | new | accepted |
| RELION_peak_gpu_memory | none | 79.7 GB | new | baseline |
| RECOVAR_half1_coarse_fine | none | 359.2s / 285.0s | new | accepted |
| RECOVAR_half2_coarse_fine | none | 352.2s / 280.3s | new | accepted |

Conclusion:

- Overall status: accepted as the first K=4 100k/256 speed/quality benchmark.
- Better metrics: new RECOVAR path is 1296.8s for iter 1, 1.87x RELION and
  below the 2x target; it improves the previous RECOVAR winner-subset profile
  from 1608.8s to 1296.8s.
- Worse metrics: none versus the prior RECOVAR quality run; absolute
  RECOVAR-vs-RELION speed is hardware-mixed (RECOVAR H100, RELION A100).
- Same metrics: GT FSC quality and class ordering match the prior accepted
  winner-subset output within numerical noise.
- Accepted as new best: yes for K=4; K=1 completion benchmark is still pending.
