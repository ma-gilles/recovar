# Minimal Parity Workflow After Cleanup

This branch now keeps only the RECOVAR surface needed for the current
RELION-parity workflow:

- simulate benchmark data
- run refinement
- run `scripts/run_multi_iter_parity.py`
- inspect parity with `scripts/diff_relion_recovar_per_iter.py`

Supported entrypoints:

- Simulate / prepare benchmark data:
  `pixi run python scripts/prepare_relion_parity_benchmark.py --output-dir <dir> --n-images <N> --grid-size <N> --noise-level <x>`
- Run refinement:
  `CUDA_VISIBLE_DEVICES=<gpu> XLA_PYTHON_CLIENT_PREALLOCATE=false pixi run python scripts/run_full_refinement.py --data_dir <dir> --output <dir>`
- Run multi-iteration parity replay:
  `CUDA_VISIBLE_DEVICES=<gpu> XLA_PYTHON_CLIENT_PREALLOCATE=false pixi run python scripts/run_multi_iter_parity.py --relion_dir <relion_run_dir> --data_star <particles.star> --iter <start_iter> --max_iter <n> --output_dir <dir>`

Retained validation commands:

- `pixi run python -m py_compile recovar/em/dense_single_volume/__init__.py recovar/em/dense_single_volume/engine_v2.py recovar/em/dense_single_volume/refine.py scripts/prepare_relion_parity_benchmark.py scripts/run_full_refinement.py scripts/run_multi_iter_parity.py scripts/diff_relion_recovar_per_iter.py scripts/postprocess_multi_iter_gt.py`
- `pixi run pytest tests/unit/test_refine_relion_mode.py tests/unit/test_run_multi_iter_parity.py tests/unit/test_adaptive_oversampling.py -v`

Cleanup decisions:

- The initial-model / ab-initio stack is intentionally removed from this branch.
- The replay-forensics and parity-debug dump scripts are intentionally removed.
- `scripts/postprocess_multi_iter_gt.py` is retained because `run_multi_iter_parity.py`
  calls it directly.
