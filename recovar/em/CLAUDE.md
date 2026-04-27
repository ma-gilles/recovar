# EM Module Developer Guide

## Active Agent Instructions

This `CLAUDE.md` file is the canonical EM agent guide for both Claude and
Codex. `recovar/em/AGENTS.md` only points back here to avoid duplicated
instructions. Do not discard or replace the existing instructions below;
they remain active.

Current RELION-parity branch: `claude/relion-parity-local-search-fix`.

Goal: perfect quality parity with RELION and near RELION speed parity for
the full end-to-end dense single-volume EM iteration.

Method: aggressively compare RECOVAR and RELION source code, and add gated
dump hooks to both codebases when source reading is not decisive. The local
RELION checkout is `/scratch/gpfs/GILLES/mg6942/relion`; the patched build is
`/scratch/gpfs/GILLES/mg6942/relion/build_patched`.

Do not solve parity by parameter tuning. For parity fixes, identify the
RELION source behavior, metadata value, or dump-level mismatch first, then
encode the same behavior in RECOVAR with a targeted test.

Numerical triage target: generally aim for `~1e-4` score/Pmax accuracy in
RELION accelerated GPU parity, because stable float32/texture arithmetic can
leave residual per-pose score and Pmax differences at that level. Treat exact
pose agreement plus Pmax/score gaps near `1e-4` as arithmetic-level parity
unless a source/dump comparison shows otherwise. Escalate gaps at `1e-3` or
larger, pose flips, or systematic multi-iteration drift. If unsure, rerun the
RECOVAR side with float64 scoring (`JAX_ENABLE_X64=1` and do not set
`RECOVAR_RELION_FLOAT32_SCORING=1`) and, when needed, obtain a RELION
CPU/double or `ACC_DOUBLE_PRECISION` dump for the same particle/candidate set
before changing algorithmic behavior.

Future EM parity agents should use this as the default numeric contract: do
not chase bitwise equality against RELION GPU/texture arithmetic, but do chase
all reproducible gaps beyond the `~1e-4` arithmetic band with source-level and
dump-level evidence.

Required dump coverage for deep parity work includes raw E-step scores,
posterior probabilities, every attempted pose in full-grid pass 1,
adaptive/oversampled pass 2, and local search, best poses after each pass,
angle/translation priors, support masks, noise accumulators, `Ft_y`,
`Ft_ctf`, maps, FSC, tau2, data-vs-prior, current size, and resolution
state.

Use targeted EM tests and targeted RELION replay Slurm jobs during
iteration. Do not run the full RECOVAR-wide test suite for normal EM parity
work unless explicitly requested or preparing a PR that requires it.

For EM development, ignore the typical root-level/full RECOVAR test-suite
requirement from broader repo instructions. Those tests are irrelevant to this
RELION-parity work and should not be launched by default. Run only targeted EM
unit tests, focused replay scripts, and selected dump-comparison jobs unless
the user explicitly asks for the full RECOVAR suite.

Keep algorithmic parity changes separate from performance changes. Batching,
caps, memory layout, and scheduling changes are performance-only until output
equivalence is proven against the old path.

Current measured baselines, hardware, Slurm job IDs, artifacts, and open
parity gaps are tracked in
`docs/math/relion_parity_current_status_2026_04_25.md`. Update that doc
whenever a new replay result, source-code finding, or dump comparison changes
the state of the investigation.

The compact roadmap and GitHub issue map for the next EM parity milestones is
`docs/math/relion_parity_roadmap_2026_04_27.md`. Read it before starting new
implementation work so pass-2 routing, convergence, initialization, large-run
reruns, cleanup, K-class refinement, and ab-initio work stay ordered.

Known low-priority boundary issue: the best one-iteration native half-volume
M-step replay matches RELION assignments and maps (`Pmax` mean abs `3.5e-5`,
exact poses/translations, final map corr `0.999996`) and matches BPref through
`rpad<=52` at `~1e-4`, but shell 26/27 BPref boundary voxels still differ.
Do not spend more time on this outermost-shell scatter mismatch unless later
end-to-end parity points back to it. Details and falsified hypotheses are in
`docs/math/relion_parity_current_status_2026_04_25.md`.

Related 2026-04-26 boundary-stress case: particle 933 at iter 2 remains a
large Pmax outlier even when RECOVAR and RELION use the same two
rotation/translation candidates and the same priors. Directly projecting the
RELION half-map through RECOVAR's projector reproduces RELION's fine-reference
projections to `~1e-7`, so projection/scoring is not the root cause. The score
gap is driven by high-shell map residuals, mainly shells 26-28 at
`current_size=58`. Explicitly zeroing projection/image pixels in those shells
does not reproduce RELION and should not be used as a fix. Keep p933 as a
boundary stress test, but choose less boundary-dominated particles for the next
M-step/tau2/noise parity trace.

2026-04-27 tau2/noise update: RECOVAR now mirrors RELION's per-half
`BackProjector::updateSSNRarrays` ordering. The FSC is shared across halves,
but each half's sigma2/tau2/data-vs-prior comes from that half's own BPref
weight, not the average of the two halves. On the 5k/128 replay, this closes
the broad shell 14-34 tau2/sigma2 mismatch; only the outer support shell 35
remains, consistent with the known boundary issue above.

2026-04-27 convergence update: replay/refine convergence state must not start
from sentinel values. For RELION replay, initialize from the previous
`run_itNNN_optimiser.star` and `run_itNNN_half1_model.star`, including current
resolution, no-resolution-gain count, no-large-hidden-variable-change count,
smallest change trackers, and optimiser accuracy estimates. For non-replay
RELION mode, seed the starting current resolution from `init_fsc` or `ini_high`.

## Recent Fixes & Active Parity Gaps (updated 2026-04-26)

### 2026-04-26 projector/scoring parity update

Source-level RELION comparison showed the accelerated path uses
`Projector::initialiseData(current_size)` with `r_max=current_size/2`, CUDA
texture linear interpolation, direct diff2 scoring, and FFTW-style centered
complex image FFTs. RECOVAR `mode="relion"` now enables these defaults unless
explicitly overridden:

- `RECOVAR_RELION_DIRECT_DIFF2_SCORING=1`
- `RECOVAR_RELION_TEXTURE_INTERP=1`
- `RECOVAR_RELION_NUMPY_IMAGE_FFT=1`

Latest tiny 1k / 64³ replay with automatic defaults:
`_agent_scratch/20260426_tiny1k_auto_parity_15715` on local A100, 69.5s,
mean abs Pmax `3.68e-5`, max abs Pmax `8.70e-4`, exact pose parity, and
recovar-vs-RELION map corr `0.999964`.

RECOVAR-side float64 replay did not remove the residual p668 score gap:
`_agent_scratch/20260426_tiny1k_float64_replay_25714` had p668 pre-prior
score deltas `[-4.60e-4, 0, -2.02e-4, +6.47e-5, -2.65e-4, -1.52e-4,
+8.40e-5]`. This is now treated as likely RELION accelerated float32/texture
arithmetic until a RELION CPU/double or `ACC_DOUBLE_PRECISION` dump proves
otherwise. Current worst Pmax outlier on that replay is original particle
374 (`-8.70e-4`) with exact pose parity.

### 2026-04-26 M-step/FSC source fixes

RELION M-step parity now depends on two exact source details:

- `BackProjector::getDownsampledAverage` uses RELION `ROUND`, i.e.
  round-half-away-from-zero, while NumPy `rint` uses banker rounding. RECOVAR
  must use RELION rounding when mapping padded backprojector voxels to the
  native downsampled half-complex grid.
- `BackProjector::getLowResDataAndWeight` / `setLowResDataAndWeight` join
  low-resolution half accumulators by squared radius
  `k*k+i*i+j*j <= ROUND(padding_factor * lowres_r_max)^2`, not by rounded
  shell labels. Rounded shell labels over-join boundary voxels near the
  `--low_resol_join_halves 40` cutoff and inflate FSC at shells 14-15 on the
  64³ tiny fixture.
- `BackProjector::updateSSNRarrays` averages only Fourier weight voxels with
  `r2 < ROUND(r_max * padding_factor)^2`, where
  `r_max=current_size/2`. Do not shell-average padded weights outside this
  support. On the tiny 64³ iter-1 accumulator this removes the false shell-28
  tau2 leak (`494.05 -> 1e-16`) and matches RELION's zero shell-28 support.
- `BackProjector::calculateDownSampledFourierShellCorrelation` bins by
  `ROUND(R)`, but first skips exact native radii with `R > r_max`. Do not
  include the outer half of a rounded shell merely because `ROUND(R) <=
  r_max`; this affects boundary shells such as shell 27 at `current_size=54`.

Latest tiny 1k / 64³ 5-iteration replay after those fixes:
`_agent_scratch/codex_tiny5_joinboundary_20260426_105052_10436` on local
A100 GPU 2. Final recovar-vs-RELION half-map corr: half1 `0.999970`, half2
`0.999969`; per-iteration recovar-vs-RELION map corr: `0.999972`,
`0.999975`, `0.999973`, `0.999976`, `0.999973`. Pmax mean abs gaps by iter:
`3.53e-5`, `9.32e-3`, `6.22e-3`, `4.77e-3`, `6.09e-3`. The remaining
multi-iter gap is above pure float32 noise and should be traced from the
remaining M-step/tau2/noise differences, not by tuning parameters.

Latest 2-iteration tiny replay with the `updateSSNRarrays`/FSC boundary fixes:
`_agent_scratch/codex_pmax_sentinels_fsc_rmax_20260426_185332_27278`, local
A100 GPU 3, `JAX_ENABLE_X64=1`. Map parity improved to final
recovar-vs-RELION corr `0.999998`, and iter-1 tau2 shell 28 now matches
RELION zero support. Iter-2 Pmax is still not closed: mean abs `0.005059`,
max abs `0.276175`, corr `0.957262`; top outliers mostly have identical
RELION/RECOVAR best pose and translation to numerical noise, so the residual
is posterior-confidence/score-gap, not pose enumeration.

Latest 5k / 128³ long end-to-end baseline:
`_agent_scratch/long_end2end_parity_20260426_182134`, Slurm job `7383509`,
A100 node `della-l07g4`, elapsed `2075.6s`, branch
`claude/relion-parity-local-search-fix`, commit
`949ab6b84a40bab5011024689c15492414c4e6ce`. Final half-map corr vs RELION:
half1 `0.996346`, half2 `0.996437`. Pmax mean abs gaps by RELION iter 2-9:
`0.00109`, `0.00634`, `0.00654`, `0.00620`, `0.01383`, `0.01920`,
`0.03473`, `0.04248`; pose mean angular gaps remain small (`0.026-0.122`
deg), with outlier pose flips but most particles matching.

### Fast diagnostic harness (use these for every parity session)

- **`recovar/em/dense_single_volume/parity_dump.py`** — env-gated per-iter
  dump. Set `RECOVAR_PARITY_DUMP_DIR=<path>` and the iteration loop writes
  one `iter_NNN.npz` per RELION iter index containing per-iter metrics
  (`ave_pmax`, `current_size`, `sigma_offset`, `random_perturbation`,
  `fsc`, `sigma2_noise`), per-half (`max_posterior`, `hard_assignment`,
  `coarse_hard_assignment`, `log_evidence`, `best_log_score`, `wsum_*`,
  `Ft_y_total/max/size`, `Ft_ctf_total/max/size`, `mean_real_ds`,
  `unreg_mean_real_ds`, `best_eulers_total`, `best_translations_total`),
  plus per-stage timings (`wall_time_s`, `stage_seconds_e_step`, etc.)
  when `start_iteration` / `mark_stage` are wired. Zero overhead when
  env unset.
- **`scripts/parity/dump_relion_iter.py`** — dump RELION's reference
  `run_itNNN_*.star` + half maps into the SAME schema for direct
  comparison.
- **`scripts/parity/compare_dumps.py`** — per-iter parity report
  (`ave_Pmax` gap, `vol_corr`, `sigma_offset` gap, `sigma2_noise`
  ratios, per-particle pose distance, first-divergence-iter assessment).

### Tiny fixture for FAST debug

`/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_tiny_parity/`:
1k particles, 64³ box, 16 RELION iters at `relion_ref_os0/`. Most parity
microtests should use this + `--max_particles 100` for sub-30-second iters.
**Do not use the 5k/128 fixture for iterative debugging** — the per-image
sparse-pass-2 path makes iter-1 a 50-min cold compile there.

### Sparse-pass-2 perf trap

`compute_pass2_stats_sparse` previously had a per-image Python for-loop
calling `run_em(image_batch_size=1, …)`, causing 5000 separate JIT
compiles per iter on the 5k fixture. Fix landed in commits `66989c86`
+ `12f1a7c3`: shape-bucketed batching via
`helpers/sparse_pass2_bucketed.py`. **If you see `[NOISE-DIAG] sumw=1`
per batch in logs, the bucketed path didn't activate** — check the
dispatch in `helpers/oversampling.py:compute_pass2_stats_sparse`.

### FSC timing fix (commit `5097ded6`)

RELION computes the CURRENT iter's FSC from M-step BPref accumulators
BEFORE `updateSSNRarrays` (`ml_optimiser_mpi.cpp:4031, 4091`;
`backprojector.cpp:1044`). Recovar previously used the PREVIOUS iter's
FSC (`fsc_history[-1]` or `init_fsc`), which at cold start meant
`init_fsc=zeros` → tau2 ≈ 0 at iter 1, then iter-2's tau2 derived from
a poorly-regularized iter-1 FSC (≈0.999) → tau2 amplifies 1e6× → 662×
volume amplification → ave_Pmax collapse to 0.

Current code (`iteration_loop.py:2845-2914`) uses **hybrid FSC choice**:
- prev-iter FSC by default (preserves the documented late-iter parity
  cancellation; codex gold gap = -5.7e-5 on 5k iter 13→14)
- current-iter fresh FSC ONLY when `max(|prior_fsc|) < 1e-3`
  (cold-start fallback for `init_fsc=zeros`)

Algorithm doc at
`docs/math/relion_updateSSNR_algorithm_2026_04_25.md`.

### `adaptive_fraction=1.0` trick

Set `--adaptive_fraction 1.0` to disable sparse pass-2 significance
pruning and route through the full-grid `else` branch in
`_run_relion_iteration_loop`. Useful for isolating sparse-vs-dense
normalization differences during E-step diffs. On the tiny fixture this
moved iter-1 ave_pmax from 0.20 → 0.23 (closer to RELION's 0.24),
indicating ~11pp of the iter-1 gap lives in the sparse-pass-2 logsumexp
normalization.

### Parity status (updated 2026-04-25 22:12, commit `949ab6b8`)

**Multi-iter trajectory on 5k/128 (cold start, default flags):**

| Iter | recovar pmax | RELION pmax | gap_abs | gap_rel |
|------|--------------|-------------|---------|---------|
| 0→1  | 0.042146 | 0.042136 | **+1.1e-5** | +0.025% |
| 1→2  | 0.645229 | 0.646337 | -1.1e-3 | -0.17% |
| 2→3  | 0.965861 | 0.964374 | +1.5e-3 | +0.15% |
| 3→4  | 0.974609 | 0.973470 | +1.1e-3 | +0.12% |
| 4→5  | 0.973150 | 0.972628 | +5.2e-4 | +0.05% |
| 5→6  | 0.928100 | 0.928428 | -3.3e-4 | -0.04% |
| 6→7  | 0.956000 | 0.950240 | +5.8e-3 | +0.61% |
| 7→8  | 0.889200 | 0.883678 | +5.5e-3 | +0.62% |

(iters 6-8 from second 15-iter run that was killed for time; iters 1-5
from first 5-iter run; both reproducibly match at 1e-3 magnitude)

**Per-iter regularized volume corr vs RELION half1 (5k/128):**

| Iter | corr_reg |
|------|----------|
| 1    | 0.999805 |
| 2    | 0.999706 |
| 3    | 0.999630 |
| 4    | 0.999543 |
| 5    | 0.999155 |

**Final iter-5 metrics:**
- `recovar_reg corr_vs_gt = 0.946049`, `RELION = 0.946271` (gap 2.2e-4)
- recovar-vs-RELION volume corr = **0.9993**
- pose error vs RELION: mean **0.45°**, p99 9.4°
- 99.5% of poses agree within 1px translation

### Pre-prior dump option (added 2026-04-25)

The old per-pose comparison was misleading: recovar's `RECOVAR_DEBUG_PER_POSE_DUMP_DIR`
captured `scores` AFTER prior addition (`em_engine.py:1527`) while RELION's
`exp_Mweight_diff2.bin` is captured BEFORE prior (`ml_optimiser.cpp:8450`).
This made the apparent gap ~200× worse than reality.

For apples-to-apples diff² comparison, set:
```
RECOVAR_DEBUG_PER_POSE_DUMP_DIR=<dir>
RECOVAR_DEBUG_PER_POSE_DUMP_TARGET=<image_idx>
RECOVAR_DEBUG_PER_POSE_DUMP_PREPRIOR=1
```
to also dump scores BEFORE prior addition (`em_engine.py:1481-1506`).
Files are named `target<idx>_block<b>_preprior.npy`.

### Stale measurements (kept for history)

| Test | Gap | Status |
|---|---:|---|
| 5k iter 13→14 (codex's canonical late-iter) | -1.07e-4 | ✓ matches gold magnitude |
| Tiny cold-start iter 1 (default) | (was -17.6%, ACTUAL +0.025% on 5k / -0.08% on tiny) | ✓ resolved; doc was misleading |

### Open follow-up branches

- **#118** `claude/dense-cleanup-relion-only` — drops
  `legacy_iteration_loop.py` + dead exports (-1302 LOC)
- **#119** `claude/parity-perf-baseline` — perf-baseline JSONs +
  `check_perf.py` + per-stage timers
- **#120** `claude/parity-quality-baseline` — fast (~5 min) parity
  quality test suite

All three gated on the parity bug fully closing first; rebase + merge
sequence tracked in issues #114, #115, #116, #117.

---

## RELION Volume Convention (READ THIS FIRST)

recovar and RELION use different 3D coordinate frames for real-space
volumes:
```python
vol_recovar = -np.transpose(vol_relion, (2, 1, 0))   # negate + swap X↔Z
```

**Pinned by tests/unit/test_relion_volume_convention.py — do NOT remove
these helpers without updating that test.**

### Canonical helpers (in `recovar/utils/helpers.py`)
- `load_mrc(path)` / `write_mrc(path, vol)` — for **recovar / cryosparc /
  cryoDRGN** MRCs. Round-trip safe.
- `load_relion_volume(path)` — load a **RELION** MRC and return it
  already in recovar's frame. **Use this when comparing RELION outputs
  against recovar outputs.**
- `relion_volume_to_recovar(vol)` / `recovar_volume_to_relion(vol)` —
  explicit frame conversion (the operation is its own inverse).
- `R_to_relion(R)` / `R_from_relion(euler)` — rotation Euler conversion.
  These are **CORRECT and intentionally paired with the volume
  transpose**. The negation in the volume convention cancels out at the
  projection step. Do NOT "fix" them. See issue #86.

### One-liner for FSC against a RELION reference
```python
from recovar.utils.helpers import load_relion_volume, load_mrc
relion_ref = load_relion_volume("relion_output/run_class001.mrc")
recovar_vol = load_mrc("recovar_output/final_merged.mrc")
# both are now in the same frame; FSC is meaningful
```

## RELION Reference Runs — REQUIRED FLAGS (READ BEFORE BENCHMARKING)

We lost a week of the 2026-04 RELION-parity benchmark to RELION flags
that default to off on the **command line** but are always set by the
**GUI**. NEVER trust `relion_refine --help` for "what should I pass";
the help shows C++ defaults (mostly wrong). The authoritative source is
`relion/src/pipeline_jobs.cpp::initialiseAutorefineJob()` (≈line 4126).

After auditing every `parser.checkOption(...)` call in
`relion/src/ml_optimiser.cpp::parseInitial()` against the GUI's
auto-refine job constructor, **7 flags need to be added** to any CLI
invocation of `relion_refine_mpi --auto_refine`:

### Critical (silently wrong reconstruction without these)

| Flag | CLI default | What breaks |
|---|---|---|
| `--ctf` | **OFF** | RELION reconstructs the CTF-convolved volume. Dark halo / ringing in real space, resolution plateaus ~18-22 A regardless of particle count, radial power spectrum has CTF oscillations + excess high-freq power. |
| `--firstiter_cc` | **OFF** | When init is non-RELION (recovar / cryosparc / EMD), iter-1 Bayesian E-step uses the wrong intensity scale. Pose search collapses to a 2D-extruded "stripes" basin. |

### Quality (degraded convergence / artifacts without these)

| Flag | CLI default | What it does |
|---|---|---|
| `--flatten_solvent` | OFF | Masks reconstructed reference outside the particle to zero. GUI: "Always flatten the solvent" (`pipeline_jobs.cpp:4461`). Without it, references contain noise outside the particle that pollutes projections. |
| `--zero_mask` | OFF | Masks particle exterior to **zero** instead of random noise. GUI default true for SPA. Random-noise fill (CLI default) introduces correlated Fourier components into alignment. |
| `--low_resol_join_halves 40` | -1 (off) | Below 40 Å, the two random half-reconstructions share orientation statistics instead of being independent. GUI: "Always join low-res data" (`pipeline_jobs.cpp:4509`). Prevents h1/h2 divergence at low SNR. |

### No-op for single-optics-group, but set for parity

| Flag | CLI default | What it does |
|---|---|---|
| `--norm` | OFF | Per-optics-group normalisation correction. NO-OP if all particles share an optics group. Important for any real-data benchmark with multiple sessions. GUI: hardcoded (`pipeline_jobs.cpp:4510`). |
| `--scale` | OFF | Per-optics-group intensity scale correction. Same NO-OP / multi-optics caveat as `--norm`. GUI: hardcoded same line. |

**Diagnostic before any "RELION volume looks wrong" investigation:**
```bash
grep -E "_rlnDoCorrectCtf|_rlnRefsAreCtfCorrected|_rlnDoNormCorrection|_rlnDoScaleCorrection" \
     <relion_run_dir>/run_it000_optimiser.star
# _rlnDoCorrectCtf      0   ←  YOU FORGOT --ctf  (silently wrong reconstruction)
# _rlnDoCorrectCtf      1   ←  ok
```

**Canonical RELION-parity invocation** (also in
`scripts/run_relion_parity_benchmark_slurm.sh`):
```bash
mpirun -n 3 relion_refine_mpi \
  --i particles.star \
  --ref reference_init_relion.mrc \   # RELION-frame init via write_relion_mrc
  --o run \
  --auto_refine --split_random_halves \
  --particle_diameter 200 --ini_high 30 \
  --ctf \                              # required (default off!)
  --firstiter_cc \                     # required for non-RELION init
  --flatten_solvent \                  # GUI always sets this
  --zero_mask \                        # GUI default true
  --low_resol_join_halves 40 \         # GUI always sets this
  --norm --scale \                     # GUI always sets these
  --healpix_order 3 --offset_range 3 --offset_step 1 \
  --oversampling 1 --pad 2 --gpu 0 --j 4
```

**To verify the CTF-trap diagnosis** (debugging only — does NOT change
recovar's production code path), disable CTF in recovar with
`scripts/debug_recovar_no_ctf.py` and confirm it produces the same dark
halo + power-spectrum signature as RELION-no-ctf. The script
`scripts/diagnose_relion_no_ctf_quantitative.py` makes a falsifiable
prediction: `|F[vol_relion]|^2 / |F[vol_GT]|^2 ≈ <CTF^2(k)>` if RELION
ran without `--ctf`. Both curves overlap shell-by-shell when the
diagnosis is correct.

## RELION's iter-1 ave_Pmax = 1.0 is a binarization artifact, NOT inference

When diffing recovar vs RELION per-iter, **ignore the iter-1 Pmax gap**.
At iter 1 with `--firstiter_cc` (or `--always_cc`), RELION executes a
literal winner-take-all binarization (`ml_optimiser.cpp:7775-7803`):

```cpp
if ((iter == 1 && do_firstiter_cc) || do_always_cc) {
    // Binarize the squared differences array to skip marginalisation
    // Find best CC, set its weight to 1.0, all others to 0.0
    ...
    DIRECT_A1D_ELEM(exp_Mweight, myminidx) = 1.;
}
```

So `ave_Pmax = 1.0` is by construction at iter 1, not because RELION's
inference is "sharper". The CC scoring (line 7414) is also scale-invariant,
specifically to absorb intensity-scale mismatch from non-RELION init
volumes. **Do not add a hard-CC iter-1 path to recovar's `_run_relion_iteration_loop`**
to match this number — it's RELION's hack, not its model.

The compounding effect on iter 2+ via the iter-1 volume IS real, though:
RELION's iter-1 hard-assigned reconstruction is tighter than recovar's
soft-Bayesian one, so iter 2's Bayesian E-step starts from a sharper
volume and gets sharper posteriors. This persists for ~6 iters before
both pipelines converge to similar Pmax.

See `~/.claude/projects/-home-mg6942/memory/feedback_relion_iter1_hard_cc_is_not_parity_bug.md`
for the full forensic write-up.

See `~/.claude/projects/-home-mg6942/memory/feedback_relion_required_flags.md`,
`feedback_relion_ctf_required.md`, and
`feedback_relion_firstiter_cc_required.md` for the full forensic
write-ups.

### History
The helpers were added in commit `7df73fa` (2026-04-01 11:00) and
removed in commit `4703c634` (2026-04-01 12:08, "revert helpers.py to
clean origin/dev state") **without updating this guide**. The result was
a year of intermittent confusion: every maintainer who tried to follow
this guide hit `ImportError` and gave up. Restored by commit on
2026-04-07 along with `tests/unit/test_relion_volume_convention.py` to
prevent the same drift.

## Active Development Plan

**Read `docs/math/plan_relion_parity.md` before making any changes to this module.**

The plan describes a 7-phase effort to bring this module to RELION feature parity.
All new work targets `dense_single_volume/em_engine.py`. Do not modify the legacy
`core.py`/`m_step.py` path unless needed for shared utilities. Do not modify
`heterogeneity.py` (separate owners).

## Architecture

```
em/
├── core.py                  # Cross-correlation, dot products, probability utils
├── e_step.py                # E_with_precompute (full E-step with batching)
├── m_step.py                # M_with_precompute, sum_up_images_fixed_rots_eqx
├── iterations.py            # E_M_batches_2 orchestrator, split_E_M_v2
├── states.py                # EMState, SGDState, HeterogeneousEMState
├── sampling.py              # HEALPix rotation grids, translation grids
├── noise.py                 # RELION-parity noise estimation
├── regularization.py        # tau2 prior, FSC, Wiener regularization
├── heterogeneity.py         # Low-rank heterogeneity EM (H/B matrices, PCA)
└── dense_single_volume/     # Dense homogeneous refinement (RELION-parity)
    ├── iteration_loop.py            # Core loop: refine_single_volume, _run_relion_iteration_loop
    ├── em_engine.py         # Two-pass JIT engine: E-step scoring + M-step accumulation
    └── helpers/  # Supporting modules (black-box from algorithm perspective)
        ├── types.py         # MeanStats, RelionStats, NoiseStats, EMProfileStats
        ├── convergence.py   # Angular/translational convergence detection
        ├── oversampling.py      # Two-pass adaptive oversampling (significance pruning)
        ├── fourier_window.py# Fourier cropping to current resolution
        ├── local_search.py  # Local search helper functions
        ├── orientation_priors.py # Prior construction for RELION mode
        ├── resolution.py   # Initialization / coarse-size helpers
        └── significance.py  # Batched significance computation
```

## Key Computations

### E-step cross-term (the expensive GEMM)

```
cross[i,r,t] = -2 Re <S_t(CTF·y_i/σ²), P_r μ>
```

Implemented as: create n_trans shifted copies of each image, flatten to
`(n_img × n_trans, N)`, one GEMM against `(N, n_rot)` projections.
Code: `core.py:82` (`compute_dot_products_eqx`).

The n_trans factor inflates the GEMM but enables 200× better data reuse vs
FFT-based cross-correlation. See `docs/math/translation_handling_analysis.md`.

### M-step accumulation

```
Ft_y += Σ_{i,t} γ_{i,r,t} · P_r*(S_t* CTF·y_i/σ²)
```

The sum over images and translations is done by one GEMM BEFORE backprojection:
`P @ shifted_images → (n_rot, N)`, then adjoint_slice_volume.
Code: `m_step.py:117` (`sum_up_images_fixed_rots_eqx`).

### Translation handling

Two methods exist (see `docs/math/translation_handling_analysis.md`):
- **GEMM** (default): explicit phase-shifted copies + matmul. Best for batched rotations.
- **FFT**: `iFFT(conj(img) · proj)` cross-correlation. Best for single-rotation refinement.

GEMM wins by 33× for the dense grid because it reads input data once for all rotations.
FFT wins by 2× per single rotation but cannot batch across rotations efficiently.

## Performance Status (as of 2026-03-31)

Benchmarked on A100-80GB, 5000 images, 128px, order 3 (36,864 rotations), 7×7 translations:

| Engine | Time | vs old |
|---|---|---|
| Old (E_with_precompute + M_with_precompute) | 68s | 1× |
| engine_fused.py | 26s | 2.6× |
| em_engine.py | 29s | 2.3× |
| Half-spectrum GEMMs (benchmarked, not integrated) | 19s | 3.6× |

RELION 5.0.1 on same hardware/data: ~163s per iteration (includes CPU M-step + overhead).

### Known optimization opportunities (in priority order)

1. **Half-spectrum GEMMs**: operate on N_half=8320 instead of N=16384. Demonstrated 1.7× speedup. Not yet integrated into the engines.

2. **Fourier cropping to current resolution**: RELION uses `current_size` to crop images to the current FSC resolution. At early iterations this is 50×+ fewer pixels. This is the single biggest gap vs RELION.

3. **Two-pass adaptive oversampling**: coarse angular search → prune to significant weights → fine search. Reduces effective orientations per image from 36K to ~100-500.

4. **Significant weight pruning**: only top-K orientations per image get the expensive fine-resolution evaluation.

### Known correctness debt (post-parity)

1. **Half-spectrum Hermitian weights (TODO RELION-parity-debt)**: RELION sums
   over the rfft half-image with weight=1 for ALL pixels. The mathematically
   correct approach uses weight=2 for interior frequencies (which have a
   conjugate partner) and weight=1 for DC/Nyquist (self-conjugate). RELION's
   approach computes ~half the true Gaussian log-likelihood, making posteriors
   softer than the true Bayesian posterior. The MAP orientation is unchanged
   (same ranking). We match RELION for parity; switching to correct weights
   (`make_half_image_weights`) is a post-parity improvement that would sharpen
   posteriors and may improve convergence speed. Code location:
   `em_engine.py:run_em()` where `half_spectrum_scoring=True` sets
   `half_weights = ones(...)`.

## Testing

- `tests/unit/test_dense_em_equivalence.py` — 12 numerical equivalence tests pinning all refactored functions
- `tests/unit/test_dense_em_plan.py` — 5 planner tests
- Run `pixi run test-fast` (2454 tests) before pushing
- Run `./scripts/run_tests_parallel.sh long-test` via Slurm before PR

## Rules

- NEVER widen test tolerances or modify baselines without explicit approval
- NEVER modify `heterogeneity.py` — it has separate owners
- EMState delegates to `dense_single_volume/` for the homogeneous path; SGDState and HeterogeneousEMState still call the old functions directly
- `split_E_M_v2` accesses `state.Ft_y` and `state.Ft_CTF` after `finish_up_M_step` — these attributes must be preserved
- All GPU work via Slurm for real jobs; login GPUs for quick benchmarks only
