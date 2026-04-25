# RELION Parity Current Status - 2026-04-25

This is the live branch status for dense single-volume EM RELION parity.
Historical context remains in `docs/math/relion_parity_benchmark_results.md`
and `docs/math/plan_relion_parity_v3.md`.

## Goal

Reach perfect parity with RELION in quality and near parity with RELION in
speed for the full end-to-end dense single-volume EM iteration.

The method is not parameter tuning. The method is aggressive source-level and
dump-level comparison between RECOVAR and RELION until every meaningful
intermediate agrees or the exact source-level reason for disagreement is known.

## Code And Data Locations

- RECOVAR worktree:
  `/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_codex_em_phase01_sparse_pass2_rebase_20260424`
- RECOVAR branch: `claude/relion-parity-local-search-fix`
- Local RELION source: `/scratch/gpfs/GILLES/mg6942/relion`
- Local patched RELION build:
  `/scratch/gpfs/GILLES/mg6942/relion/build_patched`
- Dataset:
  `/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized`
- RELION reference:
  `/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized/relion_ref_os0`
- Particles STAR:
  `/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized/particles.star`
- Ground-truth volume:
  `/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized/reference_gt.mrc`

## Current Branch Finding

RELION local-search pre-centering uses rounded integer-pixel previous offsets
for the search base. RECOVAR now uses `np.rint(previous_best_translations)` in
`relion_translation_search_base`.

This is an algorithmic parity change. It is covered by targeted tests in
`tests/unit/test_refine_relion_mode.py`.

## Current Quality Baselines

All jobs below ran on Della `cryoem` H100 80GB nodes through
`_agent_scratch/slurm/run_relion_parity_replay.sbatch`.

### One-Step Replay From RELION State

Run: RELION `iter 7 -> 8`, exact local search, `--skip_final_iteration`.

- Slurm job: `7347494`
- Log: `/scratch/gpfs/GILLES/mg6942/slurmo/recovar-late-replay-7347494.out`
- Output:
  `_agent_scratch/20260425_roundbase_125922_it007_008_exact_v1`
- Wall time trajectory: `[544.2338]` seconds
- Elapsed: `545.2472` seconds
- pmax mean RECOVAR: `0.885159865`
- pmax mean RELION: `0.885415984`
- pmax mean gap: `-0.000256119`
- pmax absolute mean gap: `0.005608471`
- pmax max absolute gap: `0.400822024`
- pmax correlation: `0.995279345`
- pose agreement vs RELION: angle mean `0.0129 deg`, max `3.0570 deg`;
  translation mean `0.0001 px`, max `0.3052 px`
- map quality: RECOVAR merged corr vs GT `0.965897`, RELION merged corr vs
  GT `0.965239`, RECOVAR-vs-RELION corr `0.999842`

Interpretation: one-step local replay from RELION state is tight enough that
the remaining multi-iteration drift is probably state carry-over, not gross
local support or scoring failure.

### Accumulated Full Trajectory

Run: RELION `iter 5 -> 8`, three RECOVAR iterations, exact local search,
`--skip_final_iteration`.

- Slurm job: `7348129`
- Log: `/scratch/gpfs/GILLES/mg6942/slurmo/recovar-late-replay-7348129.out`
- Output:
  `_agent_scratch/20260425_roundbase_fulltraj_currentfsc_131945_it005_008_exact_v1`
- Elapsed: `881.6116` seconds
- Wall time trajectory: `[164.7189, 149.1715, 566.9484]` seconds

| Recovar iter | RELION iter | RECOVAR pmax | RELION pmax | Mean gap | Abs mean gap | Corr |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6 | 0.929320273 | 0.928947427 | 0.000372846 | 0.001893066 | 0.998693 |
| 1 | 7 | 0.950245669 | 0.949316976 | 0.000928693 | 0.015577593 | 0.924363 |
| 2 | 8 | 0.889424854 | 0.885415984 | 0.004008870 | 0.034262965 | 0.900393 |

Final iter pose agreement vs RELION:

- full angle mean `0.0879 deg`, max `4.6778 deg`
- view direction mean `0.0727 deg`, max `4.6100 deg`
- in-plane mean `0.0338 deg`, max `2.0526 deg`
- translation mean `0.0003 px`, max `0.3053 px`

Final iter map quality:

- RECOVAR merged corr vs GT: `0.965798`
- RELION merged corr vs GT: `0.965239`
- RECOVAR-vs-RELION map corr: `0.999754`

Interpretation: map and pose parity are very close, but per-particle pmax
parity is not perfect in the accumulated trajectory. The next work should
focus on state carry-over between iterations: noise, tau2, priors, current
size, direction prior update/remap, scale/norm updates, and the exact
candidate sets per image.

## Performance Baselines

Performance is secondary to quality parity. These numbers are useful to notice
regressions, but a speed change is not a parity fix.

### Accepted Current-Path Speed Baseline

Current conservative exact-local path uses one image per exact bucket.

- Full `iter 5 -> 8` baseline: job `7348129`
- H100 80GB elapsed: `881.6116` seconds
- Per-iteration wall times: `[164.7189, 149.1715, 566.9484]` seconds

### Performance-Only Batching Experiment

An exact-local multi-image batching experiment was run and then removed from
the branch so it would not distract from algorithmic parity.

- One-step cap-8 job: `7349029`
- Output:
  `_agent_scratch/20260425_exact_batchcap_134630_it007_008_cap8`
- One-step wall time: `459.9079` seconds versus `544.2338` seconds for the
  conservative path
- pmax metrics were numerically unchanged at the reported precision
- Full cap-8 job: `7349897`
- Output:
  `_agent_scratch/20260425_defaultcap8_fulltraj_140918_it005_008_exact_v1`
- Full elapsed: `633.6340` seconds
- Full wall trajectory: `[84.1317, 61.2642, 487.4657]` seconds

Interpretation: batching likely helps speed without changing output, but it is
performance-only and should be reintroduced only after algorithmic parity is
settled or in a clearly separate patch.

## Required Deep Dump Comparison

The next durable debugging step is a matched RECOVAR/RELION dump path. The
dump should compare the same global image IDs across both codebases and include
all relevant iteration/pass metadata.

Minimum dump coverage:

- every raw E-step score for each attempted pose
- every posterior probability inferred from those scores
- every attempted pose in the full HEALPix pass
- every attempted pose in adaptive/oversampled pass 2
- every attempted pose in local search
- best rotation and translation after each pass
- rotation priors, direction priors, translation priors, local support masks
- `Ft_y`, `Ft_ctf`, weighted sums, and noise accumulators
- `sigma2_noise`, `sigma2_offset`, norm and scale corrections
- half maps, merged maps, regularized maps, unregularized maps
- FSC, tau2, data-vs-prior, current size, and resolution state

It is acceptable to edit both codebases for this:

- RELION dump hooks should live in the local checkout under
  `/scratch/gpfs/GILLES/mg6942/relion` and be built into `build_patched`.
- RECOVAR dump hooks may be added behind environment variables or debug flags,
  including inside `refine_single_volume` / `_run_relion_iteration_loop`.
- Dumps should be gated and off by default.
- Dumps should identify iteration, half-set, pass name, global image ID, local
  image index, units, and grid metadata.

## Open Questions

- Why does accumulated pmax drift by iter 2 when one-step `iter 7 -> 8` from
  RELION state is tight?
- Are RECOVAR and RELION using exactly the same candidate pose sets in every
  pass and local search bucket?
- Are posterior probabilities identical after subtracting any constant score
  offsets?
- Are noise, tau2, data-vs-prior, and current-size updates identical between
  iterations?
- Are norm and scale corrections updated and applied at the exact same point
  in the iteration?

## Update Policy

Update this file whenever:

- a new source-code finding changes the parity hypothesis
- a new dump comparison identifies an exact mismatch
- a replay job establishes a new baseline
- a speed experiment establishes a safe performance baseline
- an old result is invalidated
