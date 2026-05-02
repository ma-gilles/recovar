# EM Module Agent Guide

This is the compact operating guide for agents working in `recovar/em`.
Longer parity history and detailed run notes live in
`docs/math/relion_parity_agent_notes.md`.

## Startup

- `recovar/em/AGENTS.md` is a loader only; keep durable EM guidance here.
- Use the current checked-out task branch. Do not switch to historical branch
  names unless the user explicitly asks for a baseline comparison.
- Before implementation work, read the relevant code/tests plus:
  - `docs/math/relion_parity_current_status_2026_04_25.md`
  - `docs/math/relion_parity_roadmap_2026_04_27.md`
  - `docs/math/relion_parity_agent_notes.md` for detailed parity notes
- At task start, classify the work as docs-only, unit-test repair,
  algorithmic parity, performance-only, or PR preparation. Let that determine
  validation scope.

## Worktree provenance gate — MANDATORY

Before running any parity test, replay, or benchmark, print and verify:
```bash
git -C "$WORKDIR" rev-parse HEAD
git -C "$WORKDIR" symbolic-ref --short HEAD || echo "<detached>"
git -C "$WORKDIR" status --porcelain
```
A worktree directory name does NOT prove which branch is checked out — a
directory named `recovar_wt_parity_branch_*` may have been switched to an
unrelated branch by an earlier session. **`recovar_wt_parity_branch_20260502`
once held `claude/relion-parity-local-search-fix` (which lacks `7834dc0b`)
and produced "broken parity" results until the worktree was switched to a
branch that actually contains the parity commits.**

Parity claims must cite the commit hash, not the branch name. Branch tips
move; commits are immutable. The five commits below are the load-bearing
parity fixes — any worktree missing one of them in `HEAD`'s ancestry will
fail the replay test:
- `7834dc0b` current_size off-by-one + circular Fourier window
- `5f21574a` float64 scoring + current_size replay from model.star
- `0650b550` image pre-centering, normcorr, prior sign, float64 logsumexp
- `b125883f` shell-mapping at pf=2
- `0903a64c` pf³ tau2 correction + join radius scaling

`scripts/run_multi_iter_parity.py` now prints a provenance banner with
HEAD/branch/dirty state and asserts the required parity commits are
ancestors of HEAD. If it exits 2, the broken-looking parity is not a
regression — it's the wrong branch.

## Non-Negotiables

- Goal: perfect quality parity with RELION and near RELION speed parity for
  dense single-volume EM refinement.
- Do not solve parity by parameter tuning. Identify the RELION source behavior,
  metadata value, or dump-level mismatch first, then encode the same behavior
  in RECOVAR with a targeted test.
- Keep algorithmic parity changes separate from performance changes. Batching,
  caps, memory layout, and scheduling changes are performance-only until output
  equivalence is proven against the old path.
- Never widen test tolerances or edit baselines without explicit approval.
- Do not modify `heterogeneity.py`; it has separate owners.
- Preserve public/shared state contracts: `split_E_M_v2` reads `state.Ft_y`
  and `state.Ft_CTF` after `finish_up_M_step`.

## Numeric Contract

- For RELION accelerated GPU parity, `~1e-4` score/Pmax accuracy is usually
  arithmetic-level parity when pose agreement is exact.
- Escalate gaps at `1e-3` or larger, pose flips, or systematic multi-iteration
  drift.
- If unsure, rerun RECOVAR with float64 scoring (`JAX_ENABLE_X64=1` and the
  current float64 replay/refine option). When needed, obtain a RELION CPU,
  double, or `ACC_DOUBLE_PRECISION` dump for the same particle/candidate set.
- Do not chase bitwise equality against RELION GPU texture arithmetic. Do
  chase reproducible gaps beyond the `~1e-4` band with source-level and
  dump-level evidence.

## Evidence Policy

For deep parity work, dump enough state to identify first divergence:

- raw E-step scores and posterior probabilities
- every attempted pose in full-grid pass 1, adaptive/oversampled pass 2, and
  local search
- best poses after each pass, priors, masks, noise accumulators, `Ft_y`,
  `Ft_ctf`, maps, FSC, tau2, data-vs-prior, current size, and resolution state

For parity claims, report fixture, particle count, box size, commit/branch,
hardware, exact command, output directory, Slurm job ID if applicable, and
quantitative deltas.

## Environment

Use pixi, not a pre-existing conda environment. Run tests and scripts through
`pixi run ...` or `.pixi/envs/default/bin/python`.

For GPU tests, bind imports and rebuild CUDA FFI first:

```bash
pixi run install-recovar
PIXI_PY="$(pixi run which python)"
PYTHON="$PIXI_PY" make -C recovar/cuda clean all
```

Then run a provenance gate that checks:

- `recovar.__file__` is inside the checkout
- JAX imports from `.pixi/envs/default`
- `jax.devices()` sees the selected GPU
- `recovar.cuda_backproject.cuda_available()` is true

Set `RECOVAR_DISABLE_CUDA=1` only when intentionally testing the CPU/JAX
fallback.

## Testing

**Scope for EM/refinement/initial-volume work: never run the full RECOVAR
project test suite.** It is hours long, dominated by SPA/ET metrics,
PCS/locres, downstream pipelines, and outlier regressions that have nothing
to do with the EM engine. Running it on this scope is a waste of time and
GPU hours.

The following commands are **forbidden** for EM-scoped agents and must not
appear in any pre-PR checklist:

- `./scripts/run_tests_parallel.sh long-test`
- `./scripts/run_tests_parallel.sh full`
- `pixi run test-full`
- `pytest --long-test` (without an `-m em`-style filter)
- `pixi run python scripts/extract_regression_tables.py` (those tables are
  for SPA/ET pipeline regressions, not EM parity)

The "MANDATORY long-test" rule in the root `CLAUDE.md` applies to PRs that
touch `heterogeneity/`, `reconstruction/` (outside of EM use), `commands/`,
`output/`, or `data_io/` core paths. **EM-only PRs must use the EM-scoped
test set defined below**, plus parity-replay and full-refinement smoke
tests. If a change touches both EM and non-EM code, talk to the user before
expanding scope.

### EM-scoped test set (the only tests you run for EM work)

Quick guard (login node, no GPU, ~30s):
```bash
pixi run test-em-fast-guard
```

Focused unit tests for dense/local/helper refactors:
```bash
pixi run pytest \
  tests/unit/test_refine_relion_mode.py \
  tests/unit/test_dense_big_jit.py \
  tests/unit/test_fourier_window.py \
  tests/unit/test_half_spectrum_em.py \
  tests/unit/test_sparse_pass2_bucketed_perf.py \
  tests/unit/test_k_class_joint_semantics.py \
  tests/unit/test_local_parity_analysis.py \
  tests/unit/test_parse_relion_dump_dir.py \
  tests/unit/initial_model/
```

Parity replay + full-refinement smoke (require the 5k 128² normalized
fixture under `/scratch/gpfs/GILLES/mg6942/em_relion_proj/`):
```bash
# Fast tier — ~3 min on a single GPU. Run before every push.
pixi run python scripts/run_multi_iter_parity.py \
  --relion_dir /scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized/relion_ref_os0 \
  --data_star  /scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized/particles.star \
  --iter 3 --max_iter 1 \
  --gt_volume /scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized/reference_gt.mrc \
  --output_dir /scratch/gpfs/GILLES/mg6942/_agent_scratch/em_parity_check
```
Pass criterion: `Final half[12] vs RELION it004 corr ≥ 0.999`. The script
prints the provenance banner and asserts the parity-fix commits are in
HEAD's ancestry; if it exits 2, the worktree is on the wrong branch.

When the new EM-parity regression tests land
(`tests/integration/test_em_parity_fast.py` and
`tests/long_test/test_em_parity_long.py`), run those via Slurm before
shipping. They are scoped to EM only — do not pull in `--long-test` flags
that would sweep the rest of the suite.

Use local GPUs only for short checks after `nvidia-smi` confirms an idle
device. Use Slurm for any test ≥ a few minutes or any multi-iter parity
job. Keep all scratch under `/scratch/gpfs/GILLES/mg6942/`.

## RELION Conventions

RECOVAR and RELION use different real-space volume frames:

```python
vol_recovar = -np.transpose(vol_relion, (2, 1, 0))
```

Use `recovar.utils.helpers.load_relion_volume(path)` for RELION-produced MRCs
before FSC/comparison against RECOVAR output. Use `load_mrc` / `write_mrc` for
RECOVAR, cryoSPARC, and cryoDRGN-frame MRCs. Do not "fix" `R_to_relion` or
`R_from_relion`; they are intentionally paired with the volume convention.
`tests/unit/test_relion_volume_convention.py` pins this.

For RELION CLI parity, do not trust `relion_refine --help` defaults. Include
the GUI-equivalent flags from `pipeline_jobs.cpp::initialiseAutorefineJob()`:

```bash
--ctf --firstiter_cc --flatten_solvent --zero_mask \
--low_resol_join_halves 40 --norm --scale
```

RELION iter-1 `ave_Pmax = 1.0` with `--firstiter_cc` is a hard winner-take-all
cross-correlation artifact, not Bayesian inference. Do not add that path to
RECOVAR merely to match the iter-1 number.

## Active EM Pointers

- Dense homogeneous RELION-parity work lives under `dense_single_volume/`.
- New dense/refine work should target `dense_single_volume/em_engine.py` and
  helpers unless a shared utility really belongs elsewhere.
- `EMState` delegates to `dense_single_volume/` for the homogeneous path.
  `SGDState` and `HeterogeneousEMState` still call older functions directly.
- The old `core.py` / `m_step.py` path is still shared; change it only with
  targeted equivalence tests.

## Active Cleanup Ledger

Do not delete these TODOs or cleanup notes unless the implementation actually
removes the debt and the summary names the replacement.

- Apply the same cleanup pattern to `local_em_engine.py` / `local_big_jit.py`
  after the dense path is cleaned up; most dense EM readability TODOs have a
  local-engine analogue.
- Next local-engine cleanup should split `run_local_em_exact` into named
  bucket helpers with a small result/context object. Keep big-JIT and
  split/debug paths behaviorally identical while removing duplicated
  correction, projection, M-step, and noise setup blocks.
- Implement K-class EM for both dense and local engines after the single-class
  TODO cleanup is stable, with near RELION K-class end-to-end parity as the
  acceptance target.

## Useful Artifacts

- Current status: `docs/math/relion_parity_current_status_2026_04_25.md`
- Roadmap and issue map: `docs/math/relion_parity_roadmap_2026_04_27.md`
- Detailed agent notes: `docs/math/relion_parity_agent_notes.md`
- Algorithm note: `docs/math/relion_updateSSNR_algorithm_2026_04_25.md`
- Tiny parity fixture: `/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_tiny_parity/`
- RELION checkout: `/scratch/gpfs/GILLES/mg6942/relion`
- Patched RELION build: `/scratch/gpfs/GILLES/mg6942/relion/build_patched`

## Delivery

End every task with:

- what changed and key files modified
- exact test commands run, with Slurm job IDs/logs if applicable
- how to reproduce results
- `git status`
- `git diff --stat`
