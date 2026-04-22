## Local Search Chunking Checkpoint (2026-04-22)

### What changed

- Switched exact local-search grouping in `recovar/em/dense_single_volume/refine.py` from chunked unions of up to `64` images to per-image exact chunks.
- Added `_local_search_rotation_block_size()` so the per-image exact path still reuses a small set of JAX shapes by padding local neighborhoods to power-of-two block sizes up to the existing block cap.
- Fixed RELION view-direction conventions in the selector/runtime/helpers:
  - viewing direction is the third **row** of the rotation matrix, not the third column
  - `rotation_indices_to_matrices()` now interprets HEALPix pixels in RELION's NEST ordering
  - parity pose metrics in `scripts/run_multi_iter_parity.py` now use row-based view/in-plane axes
  - `make_relion_direction_log_prior()` now uses the corrected row-based view direction

### Why the chunking change is exact

Each image still sees the same exact local candidate set it had before. The change only stops unrelated images from inflating each other's union-of-neighborhoods. No heuristic pruning was introduced.

### Candidate-count evidence on the saved 5k hp4 replay state

Source state:
- prior rotations: `_agent_scratch/multi_iter_full_hp4_v4_childfix_gt/intermediates/it001_best_rotation_eulers_half1.npy`
- fine grid: `_agent_scratch/multi_iter_full_hp4_v4_childfix_gt/intermediates/it002_effective_rotations.npy`
- `sigma_rot = sigma_psi = 7.5 deg`, `sigma_cutoff = 3.0`

Measured with the corrected selector (`get_local_rotation_grid_fast`) on 2515 half-set images:

| chunk size | chunks | mean union rotations | total union rotations | total rotation-image evaluations |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2515 | 1406.7 | 3,537,768 | 3,537,768 |
| 2 | 1258 | 2805.0 | 3,528,637 | 7,055,906 |
| 4 | 629 | 5584.3 | 3,512,518 | 14,045,982 |
| 8 | 315 | 11038.7 | 3,477,177 | 27,796,966 |
| 16 | 158 | 21599.4 | 3,412,712 | 54,550,222 |
| 32 | 79 | 41657.7 | 3,290,956 | 104,981,029 |
| 64 | 40 | 76294.2 | 3,051,767 | 194,172,293 |

The important number is the last column. Moving from `64` to `1` cuts exact local-search candidate work by about `54.9x` on this replay state.

### Validation completed on the corrected code

- `pixi run pytest tests/unit/test_em_iterations_sampling.py::test_local_rotation_grid_fast_uses_exact_prior_rotation_angles tests/unit/test_em_iterations_sampling.py::test_local_rotation_grid_fast_respects_provided_perturbed_grid tests/unit/test_em_iterations_sampling.py::test_perturbed_rotation_grid_metadata_reuses_precomputed_rotations tests/unit/test_em_iterations_sampling.py::test_local_rotation_grid_fast_full_mode_matches_reference_loop -v`
- `pixi run pytest tests/unit/test_refine_relion_mode.py::TestRelionModeSmokeTest::test_make_relion_direction_log_prior_matches_canonical_grid_indices tests/unit/test_refine_relion_mode.py::TestRelionModeSmokeTest::test_make_relion_direction_log_prior_tracks_perturbed_view_directions tests/unit/test_run_multi_iter_parity.py -v`
- `env CUDA_VISIBLE_DEVICES='' pixi run pytest tests/unit/test_em_iterations_sampling.py tests/unit/test_refine_relion_mode.py tests/unit/test_run_multi_iter_parity.py tests/unit/test_adaptive_oversampling.py -v`
- `pixi run python -m py_compile recovar/em/sampling.py recovar/em/dense_single_volume/refine.py scripts/run_multi_iter_parity.py tests/unit/test_em_iterations_sampling.py tests/unit/test_refine_relion_mode.py`

### Benchmark runs

Old pre-chunking 64-image 2-step replay:
- output: `/scratch/gpfs/GILLES/mg6942/tmp/codex_cleanup_perf_20260421_220629_4867_perf/hp4_sub064_2iter_local`
- total wall time: `853.46 s`

Current clean reruns on the corrected code:
- 2-step replay:
  - output: `/scratch/gpfs/GILLES/mg6942/tmp/codex_cleanup_perf_20260421_220629_4867_perf/hp4_sub064_2iter_local_chunk1_v3`
  - log: `/scratch/gpfs/GILLES/mg6942/tmp/codex_cleanup_perf_20260421_220629_4867_perf/hp4_sub064_2iter_local_chunk1_v3.log`
- 1-step replay baseline:
  - output: `/scratch/gpfs/GILLES/mg6942/tmp/codex_cleanup_perf_20260421_220629_4867_perf/hp4_sub064_1iter_v1`
  - log: `/scratch/gpfs/GILLES/mg6942/tmp/codex_cleanup_perf_20260421_220629_4867_perf/hp4_sub064_1iter_v1.log`

The 1-step run isolates the unchanged full-grid first iteration. The local-search phase cost will be estimated as:

`2-step total wall - 1-step total wall`

once both reruns complete.
