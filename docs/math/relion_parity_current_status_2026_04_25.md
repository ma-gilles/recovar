# RELION-parity current status — 2026-04-25 (end of session)

## Branch
`claude/relion-parity-local-search-fix` HEAD = `affbf9fa` (pushed)

## Measured gaps (latest validation runs)

| Test | Gap | Status |
|---|---:|---|
| 5k iter 13→14 (codex's canonical late-iter) | **-1.07e-4** | ✓ codex gold magnitude restored |
| Tiny cold-start iter 2 (was 0.0002 collapse) | -2% | ✓ collapse FIXED |
| Tiny cold-start iter 1 (default) | -17.6% | ❌ deeper bug, see below |
| Tiny cold-start iter 1 with `--adaptive_fraction=1.0` | **-6.8%** | best known config |
| Tiny iter 4 starting from `--iter 2` | +0.27% | ✓ near-perfect when iter-1 bypassed |

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
