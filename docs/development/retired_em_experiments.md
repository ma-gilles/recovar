# Retired EM experiment tools

The following fixed-workload experiments are archived instead of maintained as
current tools. Their production implementations and independent numerical tests
remain. Retiring a harness does not turn its recorded result into a pass or
waive a scientific gate.

| Experiment | Archived tools | Retained checks |
| --- | --- | --- |
| GF46 raw-cache ABBA/BAAB | Runner, analyzer and tests of their sealed input/performance contract | Raw-cache admission, memory bounds and loader instrumentation tests |
| GF46 coarse GEMM gate, streaming selector, hybrid transition and H100 certificate/hybrid batches | Fixed-geometry runners, analyzers and their exclusive harness tests | Independent coarse GEMM, certificate and hybrid numerical tests |

All 19 files are available at [the last retained source](https://github.com/ma-gilles/recovar/tree/4e3e5396d8c5fee4e207b9df3acb7f9987e17473/scripts),
with [exact paths, hashes, test inventory and caller audit](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_retired_experiments_20260912/result.json).
Their original experiment reports remain in `docs/perf/` and the parity dashboard.
Restore the recorded experiment source and environment for reproduction; do not
assume that these historical hardware/binary/input contracts qualify newer code.

Maintained workflow entry points, generic profilers, shared analyzers, GT/FSC
measurement tools and fixture generators are retained. The active EM/VDAM agents
continue to own their frozen performance experiments and artifacts.
