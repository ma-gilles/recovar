# Bounded reuse of sparse pruning scores

Private performance candidate on32a636925. The chunked exact-Gaussian path in
[`compute_pass2_stats_sparse_bucketed`](../../recovar/em/sparse_pass2/sparse_pass2_bucketed.py)
already retains raw diff2 values when its existing memory budget admits them.
After finding the common minimum it converts those values to prior-adjusted
scores for normalization. Previously it discarded these scores and projected
and scored the same chunks again to build fine pruning support.

Reuse the admitted list slots for the converted scores until pruning finishes.
No projection slab is retained. The score population and dtype remain unchanged;
the existing raw-score budget (including device/allocator limits and zero-budget
fallback) remains authoritative. One conversion temporarily owns its raw and
converted chunk, as before. The retained list is released before M-step/noise
accumulation. Normalized-CC, non-exact algebraic scoring, score-only and early
log-evidence returns do not acquire this cache. Scoring kernels, reductions,
projection inputs, preprocessing and final accumulation are unchanged.

The modified existing CPU test fails on the parent for the extra pruning sweep
(one failure, three passing policy cases). After the repair, the expanded test
covers admitted/zero budgets, pruning on/off, divisible and short-tail chunks;
all saved stats/noise and main arrays match zero-budget recomputation exactly
at equal chunk boundaries. Nine focused cases pass, including budget admission;
four existing chunked/unchunked winner/pruning cases also pass in the preceding
nine-case selection. These mocked-projection CPU checks do not qualify actual
GPU operands, quality or speed. Final K1/K4 and full runtime gates remain open.

Reproduce with frozen pixi CPU Python, clean Python/conda environment,
`CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONNOUSERSITE=1`:

```bash
python -m pytest tests/unit/test_sparse_pass2_bucketed_perf.py -v \
  -k 'full_support_projection_cache_chunks_scores or exact_raw_diff2_cache_budget'
```

Next: the existing actual10073 saved-input three-call short profile, using sealed
libraries and an unchanged control. Compare all outputs and actual scoring/
projection call counts; do not substitute a whole-refinement speed claim.
