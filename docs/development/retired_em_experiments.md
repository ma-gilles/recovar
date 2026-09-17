# Retired EM experiment tools

The following fixed-workload experiments are archived instead of maintained as
current tools. Their production implementations and independent numerical tests
remain. Retiring a harness does not turn its recorded result into a pass or
waive a scientific gate.

| Experiment | Archived tools | Retained checks |
| --- | --- | --- |
| GF46 raw-cache ABBA/BAAB | Runner, analyzer and tests of their sealed input/performance contract | Raw-cache admission, memory bounds and loader instrumentation tests |
| GF46 coarse GEMM gate, streaming selector, hybrid transition and H100 certificate/hybrid batches | Fixed-geometry runners, analyzers and their exclusive harness tests | Independent coarse GEMM, certificate and hybrid numerical tests |
| Fixed case-7/case-26 report classification | Four report builders and three exclusive harness-test files | Generic FSC/trajectory/particle-state auditors and independent numerical tests |
| Historical K1 case captures | 88 shell launchers bound to dated external checkouts and fixed experiments | Configurable launchers, reusable capture tools and independent analyzers |

The 19 GF46/raw-cache files are available at [the last retained source](https://github.com/ma-gilles/recovar/tree/4e3e5396d8c5fee4e207b9df3acb7f9987e17473/scripts),
with [exact paths, hashes, test inventory and caller audit](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_retired_experiments_20260912/result.json).
The 88 fixed K1 capture scripts have their own [source/hash inventory and archive](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_fixed_capture_retirement_20260912/result.json).

The seven fixed report-builder files have a separate [source/hash inventory and archive](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_fixed_report_retirement_20260912/result.json).

Their frozen reports are preserved in the private [experiment archive](https://github.com/ma-gilles/recovar-experiments/tree/f34b8e79320116a9965954156f4edfe14257929a), with original paths and hashes in its manifest. The parity dashboard links to immutable archived reports.
Restore the recorded experiment source and environment for reproduction; do not
assume that these historical hardware/binary/input contracts qualify newer code.

Maintained workflow entry points, generic profilers, shared analyzers, GT/FSC
measurement tools and fixture generators are retained. The active EM/VDAM agents
continue to own their frozen performance experiments and artifacts.

The July K4 host-backend baseline and its two frozen-result assertions are
preserved with their [original test context](https://github.com/ma-gilles/recovar-experiments/tree/cd74e768fbab01fa00728d967a29da021319709c/experiments/k4_backend_history_202607).
The reusable trajectory comparator and its five behavior tests remain, as does
the newer snapshot consumed by current scorecards.
