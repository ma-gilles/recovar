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
