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
- `vol_corr_half1` / `vol_corr_half2`         — real-space volume correlation against RELION half-maps (signed)
- `vol_corr_abs_half1` / `vol_corr_abs_half2` — same, absolute value (sign-invariant; preferred for floors)
- `sigma2_noise_ratio_half{1,2}_med`          — recovar / RELION noise-shell ratio (median over shells)
- `sigma_offset_a`                            — translation prior std (Angstroms)
- `wall_time_s`                               — per-iter elapsed time (recorded by `parity_dump.start_iteration` + `mark_stage`)

Per-stage wall-time fields (`stage_seconds_e_step`, `_recon`, `_fsc`,
`_noise_update`, `_convergence`) are also written by the dump for ad-hoc
investigation; the default checker compares total `wall_time_s`.

### Known metric quirks (pre-parity-completion)

- **`pp_hard_assign_match_lt_5deg_rate` is currently 0.0%**. recovar's
  best Eulers use a different rotation convention from RELION's
  `particle.star` `(rot, tilt, psi)`. Until a convention conversion is
  implemented, this metric is informational; the baseline floor is set
  to 0.0 so it never trips the regression detector. Don't use this
  metric to assert parity until the convention is reconciled.
- **`vol_corr` is negative (~-0.3 at iter5)**. recovar's reconstruction
  is sign-flipped relative to RELION's because `invert_data` is not
  applied. Use `vol_corr_abs` for floors; `vol_corr` is preserved for
  diagnostic purposes.
- **`sigma2_noise_ratio_half{1,2}_med` may be huge negative numbers**.
  recovar's `wsum_sigma2_noise` is the raw `A2 - 2XA` accumulator
  (visible in NOISE-DIAG log lines) which can be negative pre-update;
  RELION's `model.star` `sigma2_noise` is the post-update positive
  variance. Until the comparison is anchored on the same post-update
  quantity, the band is intentionally `[-1e15, 1e15]`.
- **`sigma_offset_a` differs by ~4.5 A from RELION**. recovar's
  C1-update clamps at `sqrt(36.125) = 6.01 A` while RELION computes
  the unclamped posterior std. This is a known parity gap (separate
  from the parity_dump infrastructure).

## Workload choice — why iter4→5 (default), iter7→8 (optional)

Iter 0→1 cold-compile is too slow (>30 min) for a CI-style suite.
Iter 0→14 is too slow (>2 h).

The DEFAULT scenario is **iter4→5 grouped_union**. From init=iter4,
`current_size=80, healpix_order=3, local_search=False`. This exercises
the global-search path (per-image scoring + sparse pass2 + half-map
Wiener reconstruction). Wall time on A100:
- Cold JAX cache: ~15 min (parity_5k_128 cache mostly hits but new
  shapes still trigger compiles)
- Warm cache (after first run): ~6 min, dominated by the per-image
  global pass2 loop

The OPTIONAL scenario is **iter7→8 grouped_union**. From init=iter7,
`current_size=82, healpix_order=5, local_search=True`. This exercises
the local-search engine — `local_em_engine.py` buckets, fine-grid
rotation generation, M-step indexed backprojection. Cold compile takes
20+ min for the 2.36M fine-rotation grid. Warm cache should be under
10 min. Marked `optional: true` in the baseline JSON until a stable
warm baseline is captured.

Default scenarios are chosen to:
- exercise the per-half M-step accumulator and Wiener regularization
- exercise the noise update + sigma_offset update path
- complete in under 8 min on warm cache (so the full pytest run stays
  in the "fast feedback" regime)

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
