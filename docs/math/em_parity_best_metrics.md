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
- Shellwise FSC, FSC-AUC, and established FSC score/resolution summaries versus
  GT and RELION are the only map-quality gates. Correlation values in legacy
  rows are retained as weak historical diagnostics and cannot accept or reject
  a run under the current contract.

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

## Recent Completion Attempts

### 2026-06-29 `k4-currentfix-100k256`

Run metadata:

- Commit: `4fba8f48a00ca7820a763e7ba41dac4a5a8d8242`
- Branch: `codex/em-relion-combined-candidate-20260627_0054`
- Worktree: `/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_em_relion_combined_candidate_20260627_0054`
- Dirty state: broad EM candidate branch with tracked edits plus untracked EM benchmark scripts/tests; run artifacts recorded under CRYOEM scratch.
- Fixture: `/scratch/gpfs/GILLES/mg6942/em_relion_proj/ribosembly_k4_g256_n100000_completion_20260512_171123`
- Particle count: 100000
- Box size: 256
- K: 4
- RELION baseline: `/scratch/gpfs/GILLES/mg6942/em_relion_proj/ribosembly_k4_g256_n100000_completion_20260512_171123/relion_class3d_k4_it015_clean9d9`
- RECOVAR command/log: `EM_COMPLETION_SCRATCH_DIR=/scratch/gpfs/CRYOEM/gilleslab/_agent_scratch/em_k4_completion_currentfix_20260628_202610 ./scripts/run_em_completion_bench_slurm.sh --k4-only`; log `/scratch/gpfs/CRYOEM/gilleslab/_agent_scratch/em_k4_completion_currentfix_20260628_202610/k4_100k256_recovar/run_full_refinement.log`
- Slurm job IDs: setup `10388780`, K4 `10388781`, summary `10388782`
- Hardware: single visible GPU on `della-l08g2`, exclusive Slurm node, peak RECOVAR GPU memory 32.52 GiB.
- Artifacts: `/scratch/gpfs/CRYOEM/gilleslab/_agent_scratch/em_k4_completion_currentfix_20260628_202610/summary.md`, `summary_metrics.json`, `SAFE_TO_DELETE`

Quality comparison:

| Metric | Previous accepted/reference | Current | Delta | Status |
|--------|-----------------------------|---------|-------|--------|
| mean class FSC-AUC vs GT, RECOVAR | pending in ledger | 0.284755 | +0.022584 vs RELION 0.262172 | better |
| mean class FSC-AUC vs GT, RELION | 0.262172 baseline | 0.262172 | same baseline | same |
| mean class corr RECOVAR vs RELION | 0.9943 (2026-05 ledger) | 0.994365 | +0.0001 | same |
| K4 class assignment agreement vs RELION | pending | 0.89025 | not directly comparable | incomplete particle-level parity |
| pose rotation within 5 deg vs RELION | pending | 0.71669 | not directly comparable | incomplete particle-level parity |
| translation within 1 px vs RELION | pending | 0.77529 | not directly comparable | incomplete particle-level parity |
| final all-data comparison | unavailable | unavailable; run stopped at max_iter without convergence | missing | incomplete |

Performance comparison:

| Metric | Previous/reference | Current | Delta | Status |
|--------|--------------------|---------|-------|--------|
| RECOVAR walltime vs recent active accepted run | 15372s | 16952.2s | +1580.2s, +10.3% | worse |
| RECOVAR walltime vs 2026-05 ledger K4 | 23133s | 16952.2s | -6180.8s, -26.7% | better |
| RELION walltime | 7771s | 7771s | same | same |
| RECOVAR/RELION wall ratio | 1.98x recent active run, 3.0x 2026-05 ledger | 2.181x | worse vs recent, better vs ledger | mixed |
| Sparse K-class pass-2 total | pending | 12453.5s, 73.7% of completed iteration wall | confirms bottleneck | worse |
| K-class group M-step noise stats | pending | 4388.4s, 35.2% of group wall | confirms bottleneck | worse |

Conclusion:

- Overall status: correctness OK by GT FSC-AUC, but not accepted as a speed best.
- Better metrics: RECOVAR GT FSC-AUC exceeds RELION by 0.022584; speed improves over the stale 2026-05 ledger.
- Worse metrics: walltime is 10.3% slower than the newer 15,372s active accepted K4 run and still 2.18x RELION.
- Same metrics: map-level RECOVAR-vs-RELION class corr remains approximately 0.994.
- Accepted as new best: no; keep drilling K4 sparse pass-2 speed without sacrificing the GT FSC-AUC gate.

### 2026-07-10 `k1-current-full-speed-100k256`

Run metadata:

- Commit: `4fba8f48a00ca7820a763e7ba41dac4a5a8d8242`
- Branch: `codex/em-deferred-bigjit-abs2-min-20260709`
- Dirty worktree fingerprint: `ebe9a7c99ea904bc1450c9958a8023efbb93d632afafd70ccf328cbe11d0a404`
  (diff SHA-256 `2aed215d19adfc78e81fa2525e0d83f402aed9c433af800b74e00e0208681860`)
- Worktree:
  `/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_em_min_deferred_abs2_20260709_1745`
- Fixture: `baseline_100k_g256_white_noise1_bf80`, 100,000 particles,
  256 box, white noise 1, B-factor 80
- RECOVAR Slurm job: `10940895`, H100, ten iterations
- RECOVAR log:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case1_current_full_speed_rerun_20260710_1235/logs/case1_current_full_speed_rerun_10940895.out`
- Artifacts:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case1_current_full_speed_rerun_20260710_1235/case1_baseline_100k_g256_white_noise1_bf80`
- RELION baseline wall: 8,135 s; RECOVAR wall: 11,398.2 s

Quality comparison:

| Metric | Previous/reference | Current | Delta | Status |
|--------|--------------------|---------|-------|--------|
| RECOVAR merged GT FSC-AUC | RELION `0.447203` | `0.456776` | `+0.009573` | better |
| merged RECOVAR-vs-RELION FSC-AUC | current strict gate `0.995` | `0.994387` | `-0.000613` | below current strict gate; GT quality better |
| merged RECOVAR-vs-RELION correlation | `0.9995` | `0.999571` | `+0.000071` | same |
| pose angle mean / p95 | pending | `0.4228 / 1.3072 deg` | new evidence | close |
| translation mean / p95 | pending | `0.0715 / 0.2250 px` | new evidence | close |
| free-trajectory Pmax corr / mean abs gap | prior fixed-state arithmetic target `~1e-4` | `0.4060 / 0.1782` | worse | trajectory-history mismatch |
| fixed-state it001->it002 worst-particle Pmax | pending | corr `1.0`, mean/max abs `0.000309/0.000917` | new evidence | arithmetic parity |

Performance comparison:

| Metric | Previous accepted K=1 noise-1 | Current | Delta | Status |
|--------|-------------------------------|---------|-------|--------|
| RECOVAR walltime | 24,779 s | 11,398.2 s | `-13,380.8 s` (`-54.0%`) | better |
| RELION walltime | 12,675 s | 8,135 s | different run/hardware | not directly comparable |
| RECOVAR/RELION wall ratio | `1.95x` | `1.40x` | `-0.55x` | better |
| mean RECOVAR iteration wall | pending | 1,139.2 s | new evidence | incomplete comparison |

Conclusion:

- Overall status: mixed but strong. GT quality is better, but strict
  RECOVAR-vs-RELION FSC-AUC is just below the current gate; the
  speed ratio improved from `1.95x` to `1.40x`; this historical run predates
  the later strict iter-1 CC/Pmax parity fix, so its free-trajectory Pmax is not
  current correctness evidence.
- Better metrics: GT FSC-AUC and end-to-end speed ratio.
- Worse metrics: free-trajectory Pmax agreement.
- Diagnostic only: merged map correlation remains about `0.9995`.
- Accepted as new best: speed evidence yes; not a wholesale replacement for
  the accepted K=1 row because this run stopped at its ten-iteration cap
  without final all-data output and uses a different RELION baseline wall.

### 2026-07-13 `k1-current-head-fixed-3k128-boundary-replay`

Run metadata:

- Commit: `ef2dbd065812bafd3e31ba7863f4a2975414c249`
- Branch: `codex/em-parity-checkpoint-20260711`
- Worktree:
  `/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_em_parity_20260711/recovar`
- Fixture: `11_small_baseline_3k_g128_white_noise1_bf80`, 3,000 particles,
  box 128, seed 1711
- K: 1 supplied-map strict per-iteration boundary replay, ten numbered
  RELION-seeded transitions plus converged final all-data
- Slurm job: `11144457`, A100, `COMPLETED 0:0`, elapsed 12m10s
- Artifacts:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_3k128_full_trajectory_preflight_20260713_144434`
  (`SAFE_TO_DELETE` present)

Quality comparison:

| Metric | Required/reference | Current | Delta | Status |
|--------|--------------------|---------|-------|--------|
| Exact current-size schedule | `[56,56,66,68,80,80,80,80,80,80]` | exact | 0 | same/pass |
| Convergence/finalization | iter 10, converged final-all-data once | exact | 0 | same/pass |
| Worst numbered half/merged FSC-AUC | `>=0.995` | `0.999998586` | `+0.004998586` margin | pass |
| Worst numbered shell FSC | no unexplained deficit | `0.999992610` | arithmetic-level | pass |
| Worst numbered merged GT FSC-AUC delta | `>=-0.002` | `-0.000005307` | `+0.001994693` margin | pass |
| Final merged RECOVAR-vs-RELION FSC-AUC | `>=0.995`; prior accepted 3k `0.997260` | `0.998450626` | `+0.001190626` vs prior | better/pass |
| Final minimum shell FSC | no unexplained deficit | `0.997581339` at shell 51 | all shells above gate | pass |
| Final GT FSC-AUC, RECOVAR/RELION | RECOVAR no worse by `0.002` | `0.670747382 / 0.650834886` | `+0.019912496` | better/pass |
| Final GT FSC=0.5 crossing | RELION shell 40 | RECOVAR shell 41 | +1 shell | better |

Conclusion:

- Overall status: accepted for the fixed 3k/128 K=1 per-iteration boundary
  gate. This is not an autonomous trajectory: the runner injects RELION state
  at every numbered iteration.
- Better metrics: final cross FSC-AUC improves over the prior accepted small
  checkpoint, and RECOVAR GT FSC-AUC/resolution is higher than RELION.
- Worse metrics: none against the quality contract.
- Same metrics: exact schedule, convergence, and finalization semantics.
- Scope limit: not a completion-benchmark, speed row, or autonomous-trajectory
  qualification; robustness, scale, real-particle, 100k/256, and K=4 evidence
  remain open.

### 2026-07-14 `k1-case16-autonomous-robustness`

Run metadata:

- Commit: `a78ec7c0bbdf69548aa16e9dea44efcd6a1712a9`
- Fixture: `16_small_anisotropic_outliers_3k_g128_pct25_noise3_bf80`,
  3,000 particles, box 128
- K: 1 supplied-map autonomous full trajectory and converged final all-data
- Slurm job: `11186872`, A100, `COMPLETED 0:0`, elapsed 9m48s
- Artifacts:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/case16_norm_sumw_fix_validation_20260714_171639/autonomous_final_hp_fix_commit_a78ec7c0`
  (`SAFE_TO_DELETE` present)

Quality comparison:

| Metric | Required/reference | Current | Delta | Status |
|--------|--------------------|---------|-------|--------|
| Exact current-size/order schedule | RELION 11-iteration schedule | exact | 0 | same/pass |
| Convergence/finalization | iter 11, final-all-data once | exact | 0 | same/pass |
| Worst numbered merged cross FSC-AUC | `>=0.995` | `0.999987620` | `+0.004987620` margin | pass |
| Final merged cross FSC-AUC | `>=0.995` | `0.996711421` | `+0.001711421` margin | pass |
| Final minimum non-DC shell FSC | no unexplained deficit | `0.991779912` | shell 47 | pass |
| Final GT FSC-AUC, RECOVAR/RELION | RECOVAR no worse by `0.002` | `0.293003022 / 0.284665626` | `+0.008337396` | better/pass |

The previously reported final FSC-AUC `0.743531728` was invalid: it used the
average of RELION's unfiltered half BackProjectors as the final oracle. The
authoritative post-convergence all-data oracle is the unnumbered joined
`run_class001.mrc`. The corrected audit computes FSC/FSC-AUC only and is stored
at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/case16_norm_sumw_fix_validation_20260714_171639/autonomous_final_hp_fix_commit_a78ec7c0/analysis/FINAL_HP_AUDIT.md`.

### 2026-07-14 `k1-case13-integrated-per-half-noise`

Run metadata:

- Commit: `d07915fad1c0e3761d75ba3603cc60455d0a01bb`
- Branch: `codex/em-parity-checkpoint-20260711`
- Worktree:
  `/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_case13_d07915fa_immutable_20260714`
- Fixture: `13_small_anisotropic_3k_g128_white_noise3_bf80`, 3,000 particles,
  box 128
- K: 1 supplied-map full trajectory and converged final all-data
- Slurm job: `11190363`, A100, `COMPLETED 0:0`, elapsed `00:07:15`
- Artifacts:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case13_integrated_d07915fa_immutable_retry_20260714_201100`
  (`SAFE_TO_DELETE` present)

Quality comparison:

| Metric | Required/reference | Current | Delta | Status |
|--------|--------------------|---------|-------|--------|
| Exact current-size schedule | RELION `[56,56,48,48,48,48,48,48,48]` | exact | 0 | same/pass |
| Convergence/finalization | iteration 9, final-all-data once | exact | 0 | same/pass |
| Worst numbered merged cross FSC-AUC | `>=0.995` | `0.999999970691` | `+0.004999970691` margin | pass |
| Final merged cross FSC-AUC | `>=0.995` | `0.997779297632` | `+0.002779297632` margin | pass |
| Final merged GT FSC-AUC, RECOVAR/RELION | RECOVAR no worse by `0.002` | `0.312357369405 / 0.301136422552` | `+0.011220946854` | better/pass |
| Repeatability, numbered merged FSC-AUC | independent identical commit | maximum delta `8.9e-11` | arithmetic-level | pass |
| Repeatability, final merged FSC-AUC | independent identical commit | delta `1.70e-8` | arithmetic-level | pass |

Conclusion:

- Aggregate case-13 quality trajectory: accepted by FSC/FSC-AUC, exact
  schedule, convergence, and finalization semantics. Correlation was not
  computed.
- The per-half final-noise workflow bug is closed by commit `24c5157f`.
- Arithmetic parity is not fully closed: matched-grid iteration-9 particle
  1466 and final particle 188 retain structured score residuals. These are
  localization diagnostics and do not override the FSC/FSC-AUC acceptance,
  but they remain active parity work.
- FSC-only audit:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case13_integrated_d07915fa_immutable_retry_20260714_201100/integrated_fsc_audit.json`.

### 2026-07-14 `k1-case22-iteration2-numerical-butterfly`

Run metadata:

- Commit documented: `f0ef1f0c6c231ff1f9183371d235e0b37a15b825`
- Fixture: `22_small_severe_outliers_3k_g128_radial_noise5_bf80`,
  3,000 particles, box 128, K=1 supplied-map iteration-2 diagnostic
- RELION capture jobs: `11191426` projector capture; cross-device repeats
  `11191025_[0-1]`; same-device serial repeats `11191385`
- Analysis jobs: `11191947` exact projector/score replay; `11191779`
  FSC-only jitter audit
- Hardware: A100; strict serial gate used physical UUID
  `GPU-4bccbe72-c64a-5f5f-1fa8-ecf0bf6acf37`
- Artifacts:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_particle1203_factorial_20260714_201047/relion_fine_projection_capture`
  and
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_particle1203_self_jitter_20260714_202700`
  (`SAFE_TO_DELETE` markers present)

Quality/localization comparison:

| Metric | Required/reference | Current | Status |
|--------|--------------------|---------|--------|
| Exact REL projector raw-score replay | captured RELION raw scores | bit-identical for both candidates | pass |
| Identical-reference REL vs REC projector winner | same active operands | same REL candidate by one float32 ULP | pass/localized |
| Same-device repeat winner | stable decision | same REL candidate by one ULP in both serial runs | pass |
| Same-device post-iteration-1 map FSC-AUC | arithmetic-level repeatability | `0.9999999999988929` | pass |
| Eight-map RELION self FSC-AUC | arithmetic-level repeatability | `0.9999999999982315` to `0.9999999999993140` | pass |
| Maximum RELION self-FSC per-shell spread, non-DC | arithmetic-level repeatability | `1.2258e-11` | pass |
| Native REC reference vs RELION references FSC-AUC | localization only | about `0.999999999632` | globally FSC-inert |

Conclusion:

- This is a local qualified numerical-butterfly classification, not an
  aggregate case-22 quality acceptance. Correlation was not computed.
- Active projector differences and the jittering high-resolution constant are
  score-inert at the exact 256-lane float32 reduction. Replacing only the
  iteration-1 reference flips the candidate from a one-ULP RECOVAR winner to
  a one-ULP RELION winner. Therefore accumulator/BPref-to-reference arithmetic
  remains causally relevant despite near-unity FSC/FSC-AUC.
- Do not add a tie-break workaround. Continue the autonomous case-22 parity
  investigation upstream at accumulation/reference arithmetic and retain the
  existing full-trajectory FSC/FSC-AUC failure.
- Exact score audit:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_particle1203_factorial_20260714_201047/relion_fine_projection_capture/fine_projection_comparison.json`.
- FSC-only jitter audit:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_particle1203_self_jitter_20260714_202700/self_jitter_audit.json`.

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
| final_half1_FSC_AUC_vs_RELION | pending | pending | pending | pending |
| final_half2_FSC_AUC_vs_RELION | pending | pending | pending | pending |
| merged_FSC_AUC_vs_RELION | pending | pending | pending | pending |
| recovar_FSC_AUC_vs_GT | pending | pending | pending | pending |
| relion_FSC_AUC_vs_GT | pending | pending | pending | pending |
| minimum_non_DC_shell_FSC_vs_RELION | pending | pending | pending | pending |
| FSC_0.5_shell_RECOVAR | pending | pending | pending | pending |
| FSC_0.143_shell_RECOVAR | pending | pending | pending | pending |
| Pmax_gap_RECOVAR_minus_RELION | pending | pending | pending | pending |
| Pmax_absolute_error_distribution | pending | pending | pending | pending |
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
