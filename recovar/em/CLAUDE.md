# EM / RELION Parity Operating Contract

This file contains durable rules for work under `recovar/em/`. Current program
state and the next experiment live in `docs/math/em_parity_program.md`; detailed
findings live in `docs/math/relion_parity_agent_notes.md`; accepted completion
runs live in `docs/math/em_parity_best_metrics.md`.

`recovar/em/AGENTS.md` and `recovar/em/CLAUDE.md` must remain byte-for-byte
identical. After editing either, mirror the change and verify:

```bash
cmp recovar/em/CLAUDE.md recovar/em/AGENTS.md
```

## North Star

Achieve near-perfect RELION quality parity for K=1 auto-refine and K=4 3D
classification, then approach RELION wall time without losing quality. Quality
is the first gate; speed work begins only from a quality-accepted checkpoint.

During parity closure, RELION semantics are the default execution behavior.
Maintain two explicit behaviors so later scientific improvements remain easy
to test:

- **strict oracle mode** reproduces the pinned RELION GUI workflow, including
  historical behaviors such as `--firstiter_cc`, so state trajectories can be
  compared iteration by iteration;
- **quality mode** is a later opt-in during parity closure and may intentionally
  differ only when the difference is named, tested, and shown by FSC/FSC-AUC
  against ground truth to be neutral or better.

Major RELION behaviors such as grid correction, angle refinement, and
first-iteration policy must be typed configuration/CLI options with
RELION-compatible defaults, not environment-variable-only forks.

Never describe an intentional difference as strict parity. Do not tune
parameters until outputs happen to agree; identify RELION source behavior,
metadata, or the first dump-level divergence and encode it in a targeted test.

## Start Or Resume

At task start, after context compaction, before choosing tests, before a Slurm
submission, and before declaring completion:

1. Re-read this file and state the validation scope: docs-only, diagnostic,
   algorithmic quality, performance-only, or PR preparation.
2. Read `docs/math/em_parity_program.md`. Read only the relevant sections of
   `docs/math/relion_parity_agent_notes.md` and source/tests for the active item.
3. Print the immutable worktree provenance:

   ```bash
   git rev-parse HEAD
   git symbolic-ref --short HEAD || echo '<detached>'
   git status --short --branch
   git diff HEAD --stat
   git diff HEAD | sha256sum
   ```

4. Confirm the active checkout contains the required parity ancestors through
   `recovar.utils.parity_provenance` or `scripts/run_multi_iter_parity.py`.
5. Select exactly one measurable hypothesis and the cheapest experiment that
   could disprove it. Update the program board before switching hypotheses.

A directory or branch name is not provenance. Every result must cite the HEAD
commit and, for a dirty tree, its diff SHA-256 plus an untracked-file manifest.
Never overwrite, reset, stage, or commit unrelated user changes.

## Investigation Loop

Use this order for quality bugs:

1. Find the first divergent iteration, half, class, particle, pass, and state
   field. Do not debug only the final map.
2. Replay the same fixed RELION state and candidate set. Compare raw scores,
   probabilities, best pose/class/translation, and accumulators.
3. If fixed-state arithmetic agrees, move one state boundary earlier. Treat
   the issue as trajectory history rather than changing the E-step kernel.
4. Confirm the relevant behavior in RELION source or an env-gated RELION dump.
5. Add a focused regression that fails for the demonstrated reason.
6. Make the smallest correctness change, rerun the focused case, then climb
   the validation ladder.
7. Record the result, including null or negative findings, before moving on.

Keep algorithmic changes separate from performance changes. A batching,
microbatch-cap, scheduling, layout, fusion, or precision change is
performance-only until equivalence against the accepted path is demonstrated.

For deep parity work, capture enough state to locate first divergence: raw
scores and posterior probabilities, all pass-1/pass-2/local candidates, best
pose/class/translation after each pass, priors, masks, noise accumulators,
`Ft_y`, `Ft_CTF`, BPref data/weight, maps, FSC, tau2, data-vs-prior,
current-size/resolution state, convergence state, and stage timings.

## Numeric And Quality Contract

- RELION accelerated-GPU score/Pmax differences around `1e-4` are normally
  arithmetic-level parity. Escalate reproducible gaps at `1e-3` or systematic
  multi-iteration drift.
- Discrete decisions are tie-aware: require exact agreement when the winning
  margin is safely above the numeric error band. A near-tie flip is acceptable
  only after the underlying candidate scores/posteriors are shown to agree
  within that band. Never dismiss a flip as “numerical” without this evidence.
- Convergence iteration and finalization path must match exactly. Because they
  average over many particles, a mismatch is presumed to be a bug until strong
  evidence proves otherwise.
- Use float64 replay or RELION CPU/double dumps to adjudicate unclear numeric
  gaps. Do not chase bitwise equality with GPU texture arithmetic.
- Shellwise FSC curves, FSC-AUC, and the established FSC score/resolution
  summaries against GT and RELION are the only map-quality gates. Mean map
  correlation is a weak diagnostic only and must never pass, fail, or override
  a quality decision, even when it appears numerically excellent.
- K=4 comparisons require Hungarian class matching and per-class results;
  never hide a poor class in the mean.
- Report uncertainty and missing cells. “Not measured” is not “same.”

The current quantitative milestone gates are defined in
`docs/math/em_parity_program.md`. Changing a gate requires an explicit user
decision; never widen a tolerance or edit a baseline just to pass.

## Validation Ladder

Use the cheapest sufficient rung and advance only after it passes:

1. one/few-particle fixed-state dump replay;
2. focused unit test for the changed helper/path;
3. CPU fast guard: `pixi run test-em-fast-guard`;
4. GPU fast parity: `pixi run test-em-parity-fast`;
5. 5k/128 end-to-end K=1 or K-class smoke;
6. 10k-50k robustness cells at 128/256;
7. 100k/256 K=1 and K=4 completion pair, with RECOVAR and RELION for each pair
   run on the same GPU model.

During normal iteration, run the whole fast parity tier at most once every 3-4 hours
unless fixing that tier, changing its path, or doing final validation.
Prefer the directly affected test between tier runs.

For EM-only work, do **not** run repo-wide full/long suites or SPA/ET table
extraction. Forbidden by default:

- `pixi run test-full`
- `./scripts/run_tests_parallel.sh long-test`
- `./scripts/run_tests_parallel.sh full`
- unfiltered `pytest --long-test`
- `scripts/extract_regression_tables.py`

If a change crosses into shared `commands/`, `data_io/`, `output/`, or
non-EM reconstruction/heterogeneity behavior, ask the user before expanding
validation. Never modify `heterogeneity.py` for an EM-only task.

The EM long tier is Slurm-only:

```bash
./scripts/run_em_parity_long_slurm.sh
```

Completion evidence must use both K=1 and K=4 (exactly K=4, not a proxy), at least 100k particles,
at least 256x256 images, identical inputs/seeds/initial maps/masks,
and the same GPU class for RECOVAR and RELION. Completion runs are milestone
evidence, not edit-loop tests.

## Environment, GPU, And Scratch

Use pixi only. Before tests or jobs:

```bash
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
PIXI_PY="$(pixi run which python)"
"$PIXI_PY" -m pip install -e . --no-deps --no-build-isolation --ignore-installed
PYTHON="$PIXI_PY" make -C recovar/cuda clean all
"$PIXI_PY" -c "import pathlib,recovar,jax; repo=pathlib.Path.cwd().resolve(); assert str(pathlib.Path(recovar.__file__).resolve()).startswith(str(repo) + '/'); assert '.pixi/envs/default/' in str(pathlib.Path(jax.__file__).resolve()); print(jax.devices())"
```

Before a short local GPU check, run `nvidia-smi` and do not use a device already
used by another person or process. Use at most three idle local GPUs in total.
Use Slurm for multi-iteration, long, or contention-sensitive GPU work; cluster
jobs may be submitted broadly and allowed to queue. Compare RECOVAR and RELION
on the same GPU model within each timing pair; no single GPU architecture is
the universal oracle. Every sbatch job must set
`PYTHONNOUSERSITE=1`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, unset contaminating
Python/conda variables, and create per-job runtime roots:

```bash
RUN_ID="${SLURM_JOB_ID:-manual}"
export TMPDIR="/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/$RUN_ID/tmp"
export PIXI_HOME="/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/$RUN_ID/pixi_home"
export RATTLER_CACHE_DIR="/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/$RUN_ID/rattler_cache"
mkdir -p "$TMPDIR" "$PIXI_HOME" "$RATTLER_CACHE_DIR"
```

Put bulky disposable runs under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/<dated-run-name>/` and create a
`SAFE_TO_DELETE` marker at the run root. Do not put long-lived matrices under
the shared `_agent_scratch` roots. Keep long-lived EM source checkouts under
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/`, not the quota-constrained
GILLES project filesystem. Preserve curated fixtures in place.

## RELION Oracle Rules

- Pin and record the RELION source commit, patched-build identity, complete
  command, STAR metadata, GPU model, MPI layout, and seed. Do not trust help
  text for GUI defaults; inspect `pipeline_jobs.cpp` and output model STARs.
- Fail closed on mid-trajectory restarted per-half captures. RELION MPI
  initialization can broadcast rank-1 `sigma2_noise` to every follower and
  overwrite a loaded half-2 curve. Either capture the trajectory
  uninterrupted or record the target random subset and prove shellwise that
  `CTF^2 * group_scale^2 / corr_img` matches that subset's
  previous-iteration model STAR before attributing any score difference.
- Use the shared env-gated dump build under
  `/scratch/gpfs/GILLES/mg6942/relion/build_patched/`; do not create another
  RELION clone. Coordinate before editing or rebuilding this shared resource.
- Load RELION MRCs with `recovar.utils.helpers.load_relion_volume`; the frame
  convention is `vol_recovar = -transpose(vol_relion, (2, 1, 0))`.
- `--healpix_order` means the coarse pass-1 order. Adaptive oversampling is
  applied after it.
- Auto-refine uses `tau2_fudge=1`; 3D classification and InitialModel use 4.
  Verify `_rlnTau2FudgeFactor` in the model STAR.
- GUI auto-refine includes `--firstiter_cc`. Strict oracle mode must reproduce
  its hard winner and pass-2 routing semantics. Quality mode may differ only as
  an explicit, measured policy decision.
- Current-size BPref half joins use the explicit RELION padding factor.
- K-class quality claims use the RELION x-half/current-size BPref path. Native
  half-volume K-class accumulation is diagnostic unless explicitly selected.
- Do not force K-class final-all-data after non-convergence. Keep
  `RECOVAR_FINAL_ALL_DATA_GRID_CORRECT` unset/off except for a named diagnostic.
- Preserve shared contracts: `split_E_M_v2` reads `state.Ft_y` and
  `state.Ft_CTF` after `finish_up_M_step`.

Detailed source findings and dump variables belong in
`docs/math/relion_parity_agent_notes.md`, not in this contract.

## Benchmark Design And Reporting

High-resolution completion fixtures must come from target-grid PDB/mmCIF
scattering-potential volumes, not upsampled legacy 64^3 assets. Record source
coordinates, grid/voxel size, B-factor, noise model/level, CTF, class balance,
angle distribution, contrast/noise-scale variation, translations, outliers,
normalization, and seed.

Use `scripts/prepare_pdb_k1_relion_sanity_benchmark.py` for the canonical K=1
fixture and `scripts/prepare_cryobench_pdb_multiclass_relion_parity_benchmark.py`
for the canonical K=4 fixture. A K=15 run is useful stress coverage but is not the K=4
completion gate.

Broad quality claims require a matrix across dataset family, SNR/noise model,
K, class balance, uniform/preferred orientation distributions, CTF/no-CTF,
contrast/noise scale, translations, junk/outliers, seed, grid size, and
particle count. Use small cells to find failures; reserve 100k/256 runs for
milestone confirmation. Close synthetic K=1 trajectory parity first, then run
at least one well-characterized real-particle confirmation before K=4.

Complete aggregate state is compared every iteration. Candidate score surfaces
may use stratified sampling at scale, but automatically dump and investigate
every particle with a discrete, posterior, or convergence-relevant mismatch.

Every reported run includes:

- commit, dirty fingerprint, exact commands and environment overrides;
- fixture and RELION oracle identities;
- Slurm job IDs, node/GPU, logs, artifact root, `SAFE_TO_DELETE` status;
- FSC/FSC-AUC versus GT and RELION, Pmax, pose/translation, and K=4 class
  metrics as applicable;
- end-to-end and per-stage time, throughput, peak memory, compilation/warmup
  treatment, batch/microbatch sizes;
- comparison to the accepted run with every delta labeled better, worse, or same;
  use mixed or not measured only when no single directional label is valid.

Append detailed findings to `docs/math/relion_parity_agent_notes.md`; update
`docs/math/em_parity_program.md` when the active conclusion or next action
changes; update `docs/math/em_parity_best_metrics.md` only for completion
attempts. Do not paste large run histories into this file.

## Subagents And Ownership

When the user permits parallel agent work, follow `recovar/em/SUBAGENTS.md`.
The primary agent owns integration, the program board, and final claims.
Subagents get bounded hypotheses and disjoint write scopes; shared source and
the shared RELION build have one writer at a time.

## Delivery

End each task with the outcome, files changed, exact tests and Slurm job IDs,
reproduction commands, absolute artifact/log paths, unresolved risks, and the
current `git status --short --branch` plus `git diff HEAD --stat`. Do not claim
completion while required jobs are pending or while a quality gate is missing.
