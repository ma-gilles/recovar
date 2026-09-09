# EM validation and oracle runbook

Read the section needed for the current test, submission or scientific review.
The [EM contract](../../recovar/em/AGENTS.md) contains always-applicable rules.
These procedures remain mandatory when their scope applies; moving them here
does not waive a gate. Current work is in [EM status](em_status.md).

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

If the task includes shared `commands/`, `data_io/`, `output/`, reconstruction
or heterogeneity behavior, use the applicable shared validation as well. Existing
user authorization for that scope covers its necessary checks. Ask only if the
proposed work introduces a new objective not already authorized. Keep shared
scientific changes separate from EM-only fixes.

The EM long tier is Slurm-only:

```bash
./scripts/run_em_parity_long_slurm.sh
```

Completion evidence must use both K=1 and K=4 (exactly K=4, not a proxy), at least 100k particles,
at least 256x256 images, identical inputs/seeds/initial maps/masks,
and the same GPU class for RECOVAR and RELION. Completion runs are milestone
evidence, not edit-loop tests.

## Environment, GPU, And Scratch

Use the frozen pixi environment and import-provenance checks in
[CONTRIBUTING.md](../../CONTRIBUTING.md). Select CPU or allocated GPU visibility
before Python imports. Build custom CUDA explicitly for GPU qualification and
protect the recorded binary from runtime rebuilds as described there.

Before a short local GPU check, run `nvidia-smi` and do not use a device already
used by another person or process. On the user's four-GPU development machine,
leave physical GPU 0 free and use only idle physical GPUs 1, 2 or 3, at most three
in total. Set visibility by GPU UUID before Python imports or pytest collection.
On Slurm nodes, preserve the scheduler's allocation.
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
- Do not force K-class final-all-data after non-convergence. The strict-parity
  target specifies final gridding correction on. The reviewed PR158 source
  actually defaults it off; preserve that implementation during cleanup and
  record the effective setting. Resolving this scientific-policy discrepancy
  requires a separate, explicitly qualified change. Do not label the off path
  as satisfying the on-policy contract.
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

Keep the active conclusion, evidence state and next check in
`docs/development/em_status.md`. Preserve detailed dated evidence in the linked
program/notes archives; update `docs/math/em_parity_best_metrics.md` only for
completion attempts. Do not paste large run histories into this contract.

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
