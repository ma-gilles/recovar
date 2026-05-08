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

EM-parity regression tests (locked-down kernel-level RELION parity):

Fast tier (~5–10 min on a single GPU; required before any EM-only PR push):
```bash
pixi run test-em-parity-fast
# Equivalent: pixi run python -m pytest -v -s --run-slow --run-integration \
#   --run-gpu tests/integration/test_em_parity_fast.py
```
Tests `test_em_parity_fast_k1_replay` (5k 128² K=1 iter 3→4 vs RELION
it004) and `test_em_parity_fast_kclass_replay` (5k 128² K=2 iter 0→1 vs
RELION Class3D run_it001). Both check the worktree provenance gate first
via `recovar.utils.parity_provenance.assert_parity_ancestors`. Quality
ledgers are written to `tests/baselines/em_parity_quality_fast_ledger_*.json`.

EM-long tier (~2–4 hr per case; submit via Slurm only):
```bash
./scripts/run_em_parity_long_slurm.sh           # submit and exit
./scripts/run_em_parity_long_slurm.sh --watch   # submit and tail logs
```
This runs `test_em_parity_long_k1_full` (50k 256² K=1 15-iter auto-refine
parity), prepares/reuses the RELION `--grad --denovo_3dref` K=1 InitialModel
reference, runs `test_em_parity_long_k1_native_initialmodel_quality`
(50k 256² K=1 8-iter native VDAM cold-start quality), and runs
`test_em_parity_long_kclass_full` (50k 256² K=4 15-iter) on parallel GPU jobs.
It writes the combined report to the Slurm scratch dir.

The EM-long tier uses the dedicated `--em-parity-long` pytest flag and
the `em_parity_long` marker so it stays disjoint from the project-wide
`--long-test` (which is forbidden for EM-only PRs).

For PR descriptions on EM-only branches, paste the EM-parity tables:
```bash
pixi run python scripts/extract_em_parity_tables.py            # both tiers
pixi run python scripts/extract_em_parity_tables.py --tier fast
```
Do NOT use `scripts/extract_regression_tables.py` (that one is for the
SPA/ET pipeline regressions — different scope).

Use local GPUs only for short checks after `nvidia-smi` confirms an idle
device. Use Slurm for any test ≥ a few minutes or any multi-iter parity
job. Keep all scratch under `/scratch/gpfs/GILLES/mg6942/`.

## EM Benchmark Dataset Generation

For high-resolution EM/RELION parity or speed benchmarks, do not generate
particles from the bundled `recovar/assets/vol*.mrc` density maps. Those assets
are `64^3` legacy fixtures; upsampling them to 128/256/384 only zero-pads
Fourier space and cannot test near-Nyquist refinement.

Benchmark datasets intended to validate resolution must use the PDB/mmCIF path:

1. start from atomic coordinates, preferably `recovar/assets/5nrl_atoms.npz` or
   an explicit PDB/mmCIF supplied for the experiment;
2. generate a target-grid scattering-potential volume at the desired voxel size
   and grid size;
3. optionally apply an explicit, recorded B-factor; use `Bfactor=0` or very
   small values for near-Nyquist simulator sanity checks;
4. simulate particles from those target-grid volumes at the same resolution.

The existing reference implementation is
`scripts/prepare_pdb_k2_relion_parity_benchmark.py`, which uses
`recovar.simulation.trajectory_generation.generate_trajectory_volumes` before
calling `simulator.generate_synthetic_dataset`. For K=1 benchmarks, create or
use the analogous PDB-generated single-state script; do not use
`recovar.commands.make_test_dataset` unless the task is explicitly a low-res
legacy fixture test.

Always record in the run ledger: source PDB/mmCIF/NPZ path, grid size, voxel
size, B-factor, noise model/level, class distribution, seed, and whether RELION
normalization was applied.

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

In RECOVAR runner scripts, `--healpix_order` must mean the RELION coarse
pass-1 HEALPix order (`Oversampling=0` in RELION logs). Adaptive oversampling
then evaluates pass 2 at `healpix_order + adaptive_oversampling`. Do not encode
`--healpix_order` as the finest oversampled order; that silently under-samples
pass 1 by 8x per oversampling level and can look like a high-resolution quality
regression.

RELION iter-1 `ave_Pmax = 1.0` with `--firstiter_cc` is a hard winner-take-all
cross-correlation artifact, not Bayesian inference. Do not add that path to
RECOVAR merely to match the iter-1 number.

## `use_global_significant_support` path is BROKEN — keep `adaptive_fraction=1.0` for now

`recovar/em/dense_single_volume/iteration_loop.py:2414`'s
`use_global_significant_support` flag triggers a different M-step path
at iter 2+ when `state.adaptive_oversampling == 0` and
`adaptive_fraction < 1.0` and `iteration > 0`. This routes the M-step
through `_run_sparse_pass2_local_search_iteration` (the local-search
exact-engine path) instead of the dense engine.

**That path produces backprojection accumulators that are ~10⁴× off-scale
from the equivalent dense-engine output.** Even though both Ft_y and
Ft_ctf scale together (so Wiener output partially cancels), mid-shell
amplitudes drift, and the iter-2 reconstruction develops a catastrophic
FSC cliff at the low_resol_join_halves boundary. On the 5k 128² K=1
fixture, this dropped final corr_vs_GT to 0.65-0.72 (vs RELION 0.96).

**Workaround**: pass `--adaptive_fraction 1.0` (default in
`scripts/run_full_refinement.py` since this commit). Restores corr_vs_GT
to 0.964, FSC<0.5 = 14.32 Å (matches RELION's 0.960 / 15.11 Å).

**Real fix needed**: trace the BP scale discrepancy in
`_run_sparse_pass2_local_search_iteration` vs the dense engine. The two
paths are supposed to be equivalent. Memory note from a previous
session (`project_use_global_significant_support_path.md`) flagged this
as a known regression on the rebase branch — confirmed today.

Don't accidentally re-enable this path by setting
`--adaptive_fraction 0.999` (RELION's GUI default) without first fixing
the BP scale.

## RELION GUI defaults for `tau2_fudge` (memorize this — easy to get wrong)

RELION's `--tau2_fudge` default depends on JOB TYPE. The GUI passes
different values depending on which job the user clicks. Use these for
parity defaults in recovar's CLI scripts:

| RELION job | GUI default `T` (`--tau2_fudge`) | When recovar should match |
|------------|----------------------------------|---------------------------|
| **3D Auto-refine** (`relion_refine --auto_refine`) | **1.0** — GUI does NOT pass `--tau2_fudge`, so binary default `1.0` (ml_optimiser.cpp:878) is used | K=1 EM refinement, single-volume work |
| **3D Classification** (`relion_refine --K N`) | **4.0** | K-class EM, multi-class work |
| **2D Classification** | **2.0** | (n/a here) |
| **InitialModel (VDAM)** | **4.0** | InitialModel work |
| **MultiBody** | **4.0** (cryo-EM) / **1.0** (cryo-ET) | MultiBody refinement |

References: `pipeline_jobs.cpp::initialiseClass2DJob` (T=2),
`initialiseClass3DJob`/`initialiseInimodelJob` (T=4),
`initialiseMultibodyJob` (`default_T = (is_tomo) ? 1 : 4`),
`initialiseAutorefineJob` + `getCommandsAutorefineJob` (no
`--tau2_fudge` line → binary default 1.0).

**This was a real parity bug in `scripts/run_full_refinement.py`** (the
K=1 EM script): default was 4.0 with a comment claiming it matched
auto-refine. It did not — auto-refine uses 1.0. The 4× over-regularization
of `tau2` produced 17% high-shell amplitude excess at shell 18 and 8.4×
at the cs/2 boundary in iter-1 reconstructions, which compounded into
the iter-2+ FSC cliff. Fixed in commit (set default=1.0) on
`claude/em-quality-parity`.

When in doubt: **check RELION's `_rlnTau2FudgeFactor` field in
`run_it{NNN}_half1_model.star`** — that's the value RELION actually used,
regardless of what the binary default or your CLI flag says.

## Patching RELION for parity dumps — use the shared build, NEVER fork

There is **one** RELION checkout for parity work:

- Source: `/scratch/gpfs/GILLES/mg6942/relion/`
- Patched build: `/scratch/gpfs/GILLES/mg6942/relion/build_patched/`
- MPI binary: `/scratch/gpfs/GILLES/mg6942/relion/build_patched/bin/relion_refine_mpi`
- mpirun: `/usr/local/openmpi/4.1.6/gcc/bin/mpirun`

**Do NOT clone a fresh copy of RELION** to add dump statements. Every fresh
clone fragments the patch surface, makes it impossible to know which dumps
exist where, and burns a multi-hour rebuild. Edit the shared source and
rebuild the shared `build_patched/` instead.

When you need a new dump (e.g. for a new diagnostic):

1. Add the dump to the shared source under `recovar_mstep_dump_*` env-gated
   blocks (search `RECOVAR_MSTEP_DUMP_DIR` and `recovar_mstep_dump_enabled`
   in `backprojector.cpp` for the pattern). Keep all dumps env-gated so the
   default build is unaffected.
2. Rebuild with `cmake --build /scratch/gpfs/GILLES/mg6942/relion/build_patched -j 16`.
3. Document the new env var (purpose, when it fires, output format) in
   `docs/math/relion_parity_agent_notes.md`.
4. Commit your source-side change in the RELION repo (or note it in your
   PR description if RELION isn't a git repo for the user).

When activating dumps for a run, ALWAYS write to `~/myscratch/tmp/<run-name>/`
or `_agent_scratch/<run-name>/` per the global scratch rules — never write
RELION dumps next to the parity fixtures, and never write them inside the
RELION source/build tree.

If a previous agent already wired the exact dump you need (check the
`RECOVAR_*_DUMP_*` env vars in `backprojector.cpp`, `ml_optimiser.cpp`,
`healpix_sampling.cpp`), reuse it — don't duplicate.

## Reference RELION command for the K=1 5k 128² parity fixture

The deleted-and-regenerated fixture used:
```bash
mpirun -np 3 \
  /scratch/gpfs/GILLES/mg6942/relion/build_patched/bin/relion_refine_mpi \
  --i particles.star --ref reference_init_relion.mrc \
  --o relion_ref_os0/run --auto_refine --split_random_halves \
  --particle_diameter 544 --ini_high 30 --ctf --flatten_solvent --zero_mask \
  --low_resol_join_halves 40 --norm --scale --iter 8 --healpix_order 3 \
  --offset_range 3.0 --offset_step 1.0 --oversampling 0 --pad 2 \
  --gpu 1 --j 4 --random_seed 1775735620
```
`--split_random_halves` requires MPI (`relion_refine_mpi` + ≥3 processes:
1 master + 1 worker per half). `--random_seed 1775735620` matches the
seed in the historical fixture so half-set assignments are reproducible.

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
