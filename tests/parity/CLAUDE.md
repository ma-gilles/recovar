# Parity quality+perf baseline tests

## Overview

`tests/parity/` is the fast-feedback regression suite for the RELION-parity
work. One pytest invocation, ~5 minutes on warm JAX cache, single GPU. Its
job: catch quality, correctness, or wall-time regressions on every commit
to a parity branch — so you don't discover them an hour into a long run.

```bash
pixi run test-parity-fast       # selects -m parity, runs everything in tests/parity/
```

## What it tests

Each scenario in `tests/baselines/parity/quality_baseline_5k_128.json`:

1. Runs `scripts/run_multi_iter_parity.py` with `RECOVAR_PARITY_DUMP_DIR`
   active so `recovar.em.dense_single_volume.parity_dump` writes
   `iter_NNN.npz` snapshots.
2. Diffs the recovar dump against the matching RELION reference dump
   (`_agent_scratch/parity/relion/iter_NNN.npz`).
3. Runs `scripts/parity/check_parity.py` which reads the baseline JSON's
   `expected_metrics` block and emits OK / WARN / REGRESSED per metric.
4. Test fails if any metric is REGRESSED.

Metrics covered:
- `ave_pmax`                                  — match `_rlnAveragePmax` from `model.star`
- `pp_hard_assign_match_lt_5deg_rate`         — fraction of particles whose best Euler is within 5deg of RELION's
- `pp_hard_assign_match_lt_1deg_rate`         — same, < 1deg
- `vol_corr_half1` / `vol_corr_half2`         — real-space volume correlation against RELION half-maps
- `sigma2_noise_ratio_half{1,2}_med`          — recovar / RELION noise-shell ratio (median over shells)
- `sigma_offset_a`                            — translation prior std (Angstroms)
- `wall_time_s`                               — per-iter elapsed time (recorded by `parity_dump.start_iteration` + `mark_stage`)

Per-stage wall-time fields (`stage_seconds_e_step`, `_recon`, `_fsc`,
`_noise_update`, `_convergence`) are also written by the dump for ad-hoc
investigation; the default checker compares total `wall_time_s`.

## Workload choice — why iter7→8

Iter 0→1 cold-compile is too slow (>30 min) for a CI-style suite.
Iter 0→14 is too slow (>2 h). Mid-trajectory single-iter replay (init
from RELION's `iter_007` state, run one more iter) reproduces the
`local_search=True` code path with `current_size=82` and
`healpix_order=5` — that path uses the static-shape
`local_em_engine.py` buckets and is the dominant workload in the
late-iter parity regime. With a warm JAX cache it lands in ~5 min on
A100; ~15 min cold the first time.

Default scenarios are chosen to:
- exercise the local search engine (most parity-sensitive code path)
- exercise both the `e_step` accumulator and the per-half M-step
  Wiener regularization
- complete in a single iter (so total runtime stays under the 5-min
  budget per scenario)

## Updating the baseline after an intentional change

1. Run the scenario locally, capture the new metrics:
   ```bash
   RECOVAR_PARITY_DUMP_DIR=/tmp/dump \
     CUDA_VISIBLE_DEVICES=2 \
     pixi run python scripts/run_multi_iter_parity.py \
       --relion_dir /scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized/relion_ref_os0 \
       --data_star /scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized/particles.star \
       --iter 7 --max_iter 1 --skip_final_iteration \
       --local_engine grouped_union --output_dir /tmp/out
   ```
2. Read out the metrics:
   ```bash
   pixi run python scripts/parity/check_parity.py \
     --baseline tests/baselines/parity/quality_baseline_5k_128.json \
     --scenario iter7_to_8_grouped_union \
     --recovar-dump /tmp/dump/iter_008.npz \
     --relion-dump _agent_scratch/parity/relion/iter_008.npz
   ```
3. Edit the baseline JSON, updating `expected_metrics` for the affected
   scenario, and append to `source_runs` an entry of the form
   `branch=<name> sha=<HEAD> date=<YYYY-MM-DD>` plus a short note about
   why the metrics shifted.
4. Commit baseline + algorithmic change in the SAME commit so reviewers
   see the cause and effect together.

If you are NOT intentionally changing metrics, do NOT widen tolerances
to make a failing test pass. Investigate first; tolerances are 10% of the
observed gap with a 3x regression threshold and should not require frequent
re-tuning.

## Schema of `quality_baseline_5k_128.json`

```
{
  "fixture": "5k_128_normalized",
  "fixture_path": "<absolute path to the recovar/relion fixture dir>",
  "relion_reference_dir": "<absolute path to RELION run_*.star tree>",
  "relion_reference_dump_dir": "<repo-relative path to dir of iter_NNN.npz>",
  "scenarios": {
    "<scenario_name>": {
      "config": { "init_iter": int, "max_iter": int,
                  "local_engine": "grouped_union|exact_v1|...",
                  "skip_final_iteration": bool,
                  "jax_cache_dir": "<optional override>" },
      "expected_metrics": {
        "<metric>": baseline value,
        "<metric>_tolerance": absolute symmetric tolerance,
        "<metric>_floor": observed must be >= floor,
        "<metric>_band": [lo, hi],
        "<metric>_regression_threshold": absolute REGRESS gap (defaults to _tolerance)
      },
      "wall_time_s_baseline": baseline total iter wall time,
      "wall_time_s_regression_threshold_multiplier": 3.0,
      "optional": false  # set true to skip in default fast suite
    }
  },
  "source_runs": [...notes...]
}
```

## Negative-test recipe

To verify the regression check actually fires:
```bash
# Corrupt a recovar dump to simulate a worst-case regression
pixi run python -c "
import numpy as np
d = dict(np.load('/tmp/dump/iter_008.npz', allow_pickle=False))
d['ave_pmax'] = np.float64(0.0001)        # forces ave_pmax_gap REGRESS
d['half1_best_eulers_total'][:] = 0.0      # forces hard-assign rate to 0
np.savez_compressed('/tmp/corrupt.npz', **d)
"
pixi run python scripts/parity/check_parity.py \
  --baseline tests/baselines/parity/quality_baseline_5k_128.json \
  --scenario iter7_to_8_grouped_union \
  --recovar-dump /tmp/corrupt.npz \
  --relion-dump _agent_scratch/parity/relion/iter_008.npz \
  --exit-code-on-regression
echo "exit $?"   # expect nonzero
```

## Coordinating with the perf baseline

The companion branch `claude/parity-perf-baseline` adds
`tests/baselines/parity/perf_baseline_5k_128_a100.json` for full-trajectory
per-iter wall-time tracking via `scripts/parity/check_perf.py`. The two are
designed to coexist:
- `perf_baseline_*.json` covers iters 1-14 (full trajectory perf only).
- `quality_baseline_*.json` covers single-iter quality + correctness.

When both are present, the baselines coexist under
`tests/baselines/parity/` with non-overlapping schema (different filename
prefix) and different test entrypoints.
