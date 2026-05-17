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

Scope contract: only ``>=100k, >=256x256`` runs (both K=1 and K=4) qualify
as completion-benchmark evidence here. Smaller fixtures live in
``tests/baselines/em_parity_*`` and are listed separately for reference.

| Case | Date | Commit | Fixture | Particles | Box | RELION baseline | RECOVAR run | Accuracy status | Speed status | Notes |
|------|------|--------|---------|-----------|-----|-----------------|-------------|-----------------|--------------|-------|
| K=1 os=0 strict | 2026-05-16 | a2108b77 + tau2_fudge fix (uncommitted) | pdb_k1_g256_n100000_completion_20260512_171123 (noise=0.001 bf=0) | 100k | 256³ | relion_autorefine_k1_it015_os0_bayes_clean9d9, wall ~17h | job 8280489, wall 74398s = 20:39h | **machine-precision parity** (recovar↔RELION FSC = 1.0 in first 30 shells; merged corr 0.999802) — same | 1.21× slower — worse | Strict no-oversampling regime; noise=0.001 fixture saturates vs GT (both RELION and recovar 0.7829 corr vs GT) |
| K=4 Class3D | 2026-05-16 | a2108b77 + tau2_fudge fix (uncommitted) | ribosembly_k4_g256_n100000_completion_20260512_171123 (noise=1.0 bf=80) | 100k | 256³ | relion_class3d_k4_it015_clean9d9, wall 2h09m | job 8290126, wall 23133s = 6h26m | mean per-class corr **0.9943**, worst 0.9934, no class permutation; FSC vs RELION never drops below 0.143 — close to parity but **recovar outperforms RELION ~1-2Å @0.5 vs GT** (parity smell — investigate post-processing) | 3.0× slower — worse | Realistic-noise fixture; recovar map quality matches but slightly diverges from RELION's exact output |
| K=1 os=1 (GUI default, noise=0.001) | 2026-05-16 (TIMEOUT) | a2108b77 + tau2_fudge fix | pdb_k1_g256_n100000_completion_20260512_171123 | 100k | 256³ | relion_autorefine_k1_it015_os1_redo (8290127 partial, 16 iters, res=4.28A converged) | job 8312443 TIMEOUT after 12h05m, only reached iter 4 at 4.81A | partial: iter 4 corr_vs_GT pending — incomplete | iter 3→4 took 3h15m, iter 5 stalled >7h before timeout | adaptive 2-pass too slow at large current_size; not viable as completion benchmark. K=1 noise=0.001 row stands; use K=1 noise=1.0 row as canonical 100k256 K=1 best |
| K=1 os=1 (noise=1.0 bf=80) | 2026-05-16 | 8ca4ddc0 (dev2 post-merge, EM tau2/LP fixes in HEAD ancestry) | pdb_k1_g256_n100000_noise1_bf80_20260516 (generated 2026-05-16, job 8313939) | 100k | 256³ | job 8314160, wall 12675s = 3h31m, 16 iters | job 8314161, wall 24779s = 6h53m, 15 iters | merged corr vs RELION it015 = **0.9995**, vs it016 = 0.9993 (h1=0.9995, h2=0.9994); recovar vs GT = 0.7552, RELION it015 vs GT = 0.7531 (recovar marginally better) — same | 1.95× slower — worse | Realistic-noise K=1 completion benchmark. Half-map parity at machine precision; tau2_fudge model.star read + ini_high LP filter active. New accepted best for K=1 noise=1.0. |

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
