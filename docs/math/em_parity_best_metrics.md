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

No accepted >=100k, >=256x256 completion benchmark has been recorded in this
ledger yet.

| Case | Date | Commit | Fixture | Particles | Box | RELION baseline | RECOVAR run | Accuracy status | Speed status | Notes |
|------|------|--------|---------|-----------|-----|-----------------|-------------|-----------------|--------------|-------|
| K=1 | pending | pending | pending | >=100k | >=256 | pending | pending | pending | pending | Populate after the first accepted completion benchmark. |
| K=4 | pending | pending | pending | >=100k | >=256 | pending | pending | pending | pending | Populate after the first accepted completion benchmark. |

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
