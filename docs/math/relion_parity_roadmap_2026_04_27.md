# RELION parity roadmap - 2026-04-27

This is the local planning entry point for the dense single-volume EM parity
branch. The detailed audit trail remains in
`docs/math/relion_parity_current_status_2026_04_25.md`; the short handoff is
`docs/math/relion_parity_quick_handoff_2026_04_27.md`.

## Active branch

- Repo: `/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_codex_em_phase01_sparse_pass2_rebase_20260424`
- Branch: `claude/relion-parity-local-search-fix`
- Current pushed checkpoint: `7141f632`
- RELION source: `/scratch/gpfs/GILLES/mg6942/relion`
- RELION patched build: `/scratch/gpfs/GILLES/mg6942/relion/build_patched`

## Operating rules

- Goal: perfect RELION quality parity and near RELION speed parity for the
  full end-to-end dense single-volume EM iteration.
- Use RELION source and dump-level comparisons as authority. Do not tune
  parameters until source behavior and intermediate values explain the gap.
- For normal EM parity work, do not launch the full RECOVAR-wide test suite.
  Use targeted EM unit tests, fixed-state replays, dump comparisons, and
  selected Slurm parity runs.
- Keep algorithmic parity changes separate from performance changes. A speed
  refactor is not complete until it proves output equivalence to the old path.

## Priority plan

### Phase 1: close current single-class fixed-state parity

1. Close fixed-state E-step score/posterior outliers.
   - Issue: https://github.com/ma-gilles/recovar/issues/127
   - Current target: original particle `4603` in the 5k normalized fixture at
     RELION it008.
   - Next comparison: RELION `fine_ref`, `fine_shifted`, `corr_img` vs
     RECOVAR `debug_proj_weighted`, `debug_shifted_score`,
     `debug_ctf2_over_nv` for the six active candidates.

2. Route sparse pass 2 through the local-search iteration machinery.
   - Issue: https://github.com/ma-gilles/recovar/issues/121
   - Reason: the group-union/bucketed sparse pass-2 path is a major speed trap
     and is another source of normalization/candidate-set divergence.
   - The grouped-union path is deprecated for RELION refinement. Do not add new
     callers, do not use it as a fallback for parity runs, and delete it once
     exact/local pass-2 routing has replacement tests and large-run coverage.
   - Gate: old-path vs new-path equivalence on tiny fixed-state tests before
     treating any timing improvement as real.

3. Match RELION auto_refine convergence and finalization exactly.
   - Issue: https://github.com/ma-gilles/recovar/issues/122
   - This must fix early RECOVAR stopping and the forced-run failure that
     requested `run_it015_sampling.star` when RELION refs stopped at it014.
   - Include `do_join_random_halves`, `do_use_all_data`,
     final all-data/Nyquist iteration, `--max_iter`, and
     `--skip_final_iteration` semantics.

4. Match RELION initialization from iteration 1.
   - Issue: https://github.com/ma-gilles/recovar/issues/123
   - Current best parity often starts at iter 2 because the RELION initial
     volume/noise/current-size state is not matched exactly.

### Phase 2: rerun end-to-end and make regression checks durable

5. Rerun the 5k/128 end-to-end trajectory on the current branch after Phase 1.
   - Required metrics: Pmax, candidate score residuals, pose/translation
     parity, map correlation, GT correlation, FSC, tau2/noise, current size,
     convergence state, wall time, stage timings, memory, commit, Slurm IDs.
   - Save the result in the current-status doc and use it as the next baseline.

6. Port PR #119 and PR #120 on top of the current branch.
   - Existing issues: https://github.com/ma-gilles/recovar/issues/115 and
     https://github.com/ma-gilles/recovar/issues/116
   - Do this after the parity path is stable enough that the baseline checks
     encode desired behavior instead of preserving known bugs.

7. Rerun large particle-count and large image-size matrices on the current
   branch, then fix CUDA/batching failures.
   - Issue: https://github.com/ma-gilles/recovar/issues/124
   - Existing related issue: https://github.com/ma-gilles/recovar/issues/113
   - Prior large runs were from an older snapshot and are not final evidence.

### Phase 3: cleanup once behavior is pinned

8. Rebuild the PR #118 cleanup on top of the current branch.
   - Existing issue: https://github.com/ma-gilles/recovar/issues/114
   - Only delete code once the main RELION refinement branch and parity tests
     prove the code is unused.
   - Include the grouped-union local-search/pass-2 code in the deletion list
     after issue #121 lands. The RELION refinement branch should use exact/local
     candidate layouts only.

9. Address code TODOs that do not change RELION parity.
   - Defer TODOs that alter math, metadata, priors, support masks, or
     convergence until a source-level RELION comparison says they are safe.

### Phase 4: extend RELION parity scope

10. Implement RELION-style K-class refinement.
    - Issue: https://github.com/ma-gilles/recovar/issues/125
    - Build on the single-class parity path. Do not fork a separate algorithm.

11. Implement RELION-style ab-initio refinement.
    - Issue: https://github.com/ma-gilles/recovar/issues/126
    - Start only after single-class and K-class parity are stable.

## Parallel work split

The following investigations are independent enough to run in parallel while
one engineer continues the fixed-state score trace:

- Pass-2 routing: inspect how to replace sparse group-union pass 2 with
  `_run_local_search_iteration()` while preserving candidate enumeration.
- Convergence/finalization: source-map RELION `checkConvergence`,
  `do_join_random_halves`, `do_use_all_data`, and final iteration behavior.
- PR #119/#120 transplant: compare current branch to the PR heads and identify
  which scripts/tests/baselines can be cherry-picked safely.
- Large-run failures: parse the old 100k/box192 logs and propose current-branch
  reruns plus CUDA/batching reproducers.

Record each result in this roadmap or the current-status doc before starting
implementation, so the next agent does not repeat the same source audit.

## Read-only investigation results - 2026-04-27

### Pass-2 routing

Non-local adaptive pass 2 still routes through `compute_pass2_stats_sparse()`
from `iteration_loop.py`, with a second sparse caller in the global-significant
support path. The current bucketed implementation builds per-image oversampled
children and a sparse `(rotation, translation)` mask in
`helpers/sparse_pass2_bucketed.py`.

The key design mismatch is that `_run_local_search_iteration()` / exact local
search currently builds Gaussian neighborhoods from prior poses, while adaptive
pass 2 must score arbitrary pass-1 significant coarse samples and their
oversampled children. `score_local_bucket()` currently accepts only a
per-rotation mask; pass 2 needs an optional `(B, R, T)` candidate mask.

Minimal implementation:

1. Add a pass-2 layout builder beside `LocalHypothesisLayout` that reuses the
   current per-image pass-2 input preparation.
2. Extend `LocalHypothesisLayout`, `LocalBucketSpec`, and `score_local_bucket()`
   to accept an optional `(B, R, T)` candidate mask.
3. Extend `run_local_em_exact()` to return the pass-2 best rotation,
   translation, and child bookkeeping needed by the iteration loop.
4. Replace the adaptive sparse caller with `_run_local_search_iteration(...,
   local_engine="exact_v1", pass2_layout=...)`; do not route through
   `grouped_union`.
5. Convert the global-significant-support path only after adding support for
   its externally supplied `normalization_log_z`.

After this lands, grouped-union should have no RELION-refinement callers. Keep
it only long enough for migration/equivalence tests, then remove it rather than
maintaining it as an alternate backend.

Relevant tests: update `tests/unit/test_refine_relion_mode.py` assertions that
patch `compute_pass2_stats_sparse`, and compare new local-exact pass 2 against
the current per-image reference in `tests/unit/test_sparse_pass2_bucketed_parity.py`.

### Convergence and finalization

RELION checks convergence at the top of each loop before the E-step. The exact
condition is:

`has_fine_enough_angular_sampling && nr_iter_wo_resol_gain >= 1 &&
(auto_ignore_angle_changes || nr_iter_wo_large_hidden_variable_changes >= 1)`.

On convergence, RELION sets `has_converged=true`,
`do_join_random_halves=true`, and `do_use_all_data=true`. The final iteration is
not a separate post-loop phase; it is the next normal loop iteration after those
flags are set. `do_use_all_data` forces `current_size=ori_size`.

Important RECOVAR mismatches:

- RECOVAR currently checks convergence at the end of a normal iteration and has
  a separate final block.
- RECOVAR can run that final block after plain `max_iter` exhaustion; RELION
  does not.
- RECOVAR's final block scores against averaged half maps. RELION still scores
  each random half against its own half map and only joins weighted sums after
  expectation/FSC.
- RECOVAR treats `healpix_order >= max_healpix_order` as fine enough; RELION's
  condition is based on angular accuracy / maximum angular sampling.

Minimal implementation:

1. Move finalization into the RELION-mode loop as a pre-iteration control flag.
2. If previous state converged or RELION-style `force_converge` is requested,
   run exactly one Nyquist/current-size-full iteration with
   `do_join_random_halves=True` and `do_use_all_data=True`, then stop.
3. During final E-step, score half 1 from half 1 map and half 2 from half 2
   map; join `Ft_y`/`Ft_ctf` only after both halves finish.
4. Do not run finalization merely because `max_iter` was reached.
5. Tests should stub `run_em` to prove final iteration uses half maps,
   max-iter-without-convergence does not finalize, convergence triggers one
   Nyquist joined iteration, and final comparison uses unsuffixed RELION
   `run_*` files rather than a non-existent `run_itNN+1`.

### PR 119 / 120 transplant

PR #119 (`claude/parity-perf-baseline`) is effectively already present on this
branch. The add-only files are byte-identical, and the stage-marker commit is
superseded by current branch work. Do not cherry-pick #119.

PR #120 (`claude/parity-quality-baseline`) should be ported manually, not
cherry-picked wholesale. Copy only add-only quality tooling:

- `scripts/parity/check_parity.py`
- `scripts/parity/compare_dumps.py`
- `scripts/parity/dump_relion_iter.py`
- `scripts/parity/extract_quality_table.py`
- `scripts/parity/negative_test.py`
- `scripts/parity/populate_baseline.py`
- `tests/baselines/parity/quality_baseline_5k_128.json`
- `tests/parity/*`

Small wiring hunks to port manually: add `_agent_scratch/` to `.gitignore`, add
`test-parity-fast` and `test-parity-smoke` tasks to `pixi.toml`, and add the
pytest marker `parity` to `tests/conftest.py`.

Do not port PR #120's core EM edits wholesale. They are stale versus the
current local-search and translation-prior fixes, especially in
`orientation_priors.py`, `iteration_loop.py`, `local_em_engine.py`,
`local_layout.py`, `parity_dump.py`, and `tests/unit/test_refine_relion_mode.py`.

After transplant, run `git diff --check`, `pixi run test-parity-smoke`,
`pixi run python scripts/parity/check_parity.py --help`, and targeted EM tests.

### Large runs

Old large runs are not final evidence because they launched from
`/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_longrun_snapshot_20260426_191122`,
not the current branch.

The 100k run `7385332` was still running at the last audit. Its slowness is
caused by sparse pass 2 degenerating to one image per bucket: logs show roughly
`50002 images -> 50002 buckets`, median local rotations about `3500-3850`, and
pass-2 half times around `64-78 min`. This is directly tied to issue #121.

The box192 substitute run `7385331` failed after completing global iterations
1-4, during iteration 5 local search at `current_size=124`, HP4, `294912`
rotations, and 13 translations. The stack ends near
`compute_reconstruction_support -> find_significant_mask`, but the
`CUDA_ERROR_ILLEGAL_ADDRESS` likely surfaced asynchronously from an earlier GPU
kernel. It was not host OOM.

Current-branch rerun plan:

1. Rebuild CUDA and use a fresh JAX cache.
2. Rerun current branch after pass-2 routing is fixed.
3. If box192 still fails, run an iter-5-only repro with
   `CUDA_LAUNCH_BLOCKING=1`.
4. Isolate with `RECOVAR_RELION_TEXTURE_INTERP=0` and
   `RECOVAR_DISABLE_CUDA=1` on a smaller/subset repro to distinguish texture
   projector issues from indexed backproject/local-support issues.
