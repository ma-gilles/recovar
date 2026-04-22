# Local Search Init-Eulers Checkpoint (2026-04-22)

## What was wrong

The hp4 replay was still paying a dense full-grid bootstrap on the first
local-search iteration even after the chunking and block-size fixes.

Root cause:
- `run_multi_iter_parity.py` already loaded the starting RELION Euler angles
  for each half-set.
- `refine_single_volume()` did not accept an initial
  `previous_best_rotation_eulers` argument.
- The first RECOVAR replay iteration therefore entered hp4 with
  `previous_best_rotation_eulers=[None, None]` and fell back to the dense
  `294912`-rotation search before exact local search.

## Code changes

- `recovar/em/dense_single_volume/refine.py`
  - added `init_previous_best_rotation_eulers`
  - seeded `previous_best_rotation_eulers` from that init argument before the
    RELION-mode loop starts
- `scripts/run_multi_iter_parity.py`
  - now extracts the starting RELION Euler angles from the input `run_itXXX_data.star`
  - passes them as `init_previous_best_rotation_eulers=[euler_h1, euler_h2]`
- `tests/unit/test_refine_relion_mode.py`
  - added `test_init_previous_best_rotation_eulers_seed_first_local_iteration`

## Validation

Focused regression:

- `env CUDA_VISIBLE_DEVICES='' pixi run pytest tests/unit/test_refine_relion_mode.py -k 'local_search_rotation_block_size or local_search_engine_rotation_block_size or pad_local_search_rotations or first_local_iteration_uses_previous_best_rotations_without_dense_bootstrap or init_previous_best_rotation_eulers_seed_first_local_iteration or local_search_uses_negative_rounded_previous_offsets_for_translation_prior' -v`
  - result: `7 passed, 33 deselected in 11.51s`

Focused retained suite:

- `env CUDA_VISIBLE_DEVICES='' pixi run pytest tests/unit/test_em_iterations_sampling.py tests/unit/test_refine_relion_mode.py tests/unit/test_run_multi_iter_parity.py tests/unit/test_adaptive_oversampling.py -v`
  - result: `96 passed, 5 warnings in 115.98s`

## Short replay benchmark

Benchmark command family:

- `scripts/run_multi_iter_parity.py`
- 5k dataset
- start from RELION iter `5`
- `max_iter=2`
- `max_particles=64`

### Before (`v6`)

Artifacts:
- log: `/scratch/gpfs/GILLES/mg6942/tmp/codex_cleanup_perf_20260421_220629_4867_perf/hp4_sub064_2iter_local_chunk1_v6.log`
- wall time: `/scratch/gpfs/GILLES/mg6942/tmp/codex_cleanup_perf_20260421_220629_4867_perf/hp4_sub064_2iter_local_chunk1_v6_time.txt`

Measured:
- script-level `Completed 2 iterations in 346.0s`
- wrapper wall time `real 379.45`
- iteration 1 time `317.7s`
- iteration 2 time `25.7s`

First hp4 step still showed dense bootstrap:
- `sumw=32 n_rot=294912 use_window=True`

### After (`v8`)

Artifacts:
- log: `/scratch/gpfs/GILLES/mg6942/tmp/codex_cleanup_perf_20260421_220629_4867_perf/hp4_sub064_2iter_local_chunk1_v8.log`
- wall time: `/scratch/gpfs/GILLES/mg6942/tmp/codex_cleanup_perf_20260421_220629_4867_perf/hp4_sub064_2iter_local_chunk1_v8_time.txt`

Measured:
- script-level `Completed 2 iterations in 139.8s`
- wrapper wall time `real 162.55`
- iteration 1 time `56.9s`
- iteration 2 time `16.8s`

The first hp4 step now starts in exact local search immediately:
- first `engine_v2` line uses `Using rotation log-prior: 1416 rotations (per-image)`
- no dense `n_rot=294912` bootstrap appears before local search

### Speedup summary

Compared with `v6` on the same 64-particle 2-step hp4 replay:

- iteration 1: `317.7s -> 56.9s` (`5.58x` faster)
- iteration 2: `25.7s -> 16.8s` (`1.53x` faster)
- 2-step script runtime: `346.0s -> 139.8s` (`2.48x` faster)
- wrapper wall time: `379.45s -> 162.55s` (`2.33x` faster)

## Important caveat

The short replay still ends with the explicit final all-data Nyquist iteration,
which is expected to use the dense full grid. The remaining dense
`n_rot=294912` call in `v8` is from that final iteration, not from the hp4
replay path.

## Next step

Run the same fixed code on the full 5k replay with `--skip_final_iteration` to
measure the real hp4 local-search timing on the actual benchmark path and
compare it directly against RELION's matched replay-stage timings.
