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
| K4 iteration-10 panel12 decomposition | Three-way, cohort, posterior and numerator analyzers with their exclusive tests | Current K4 scorecards, capture validators and production numerical tests |
| K4 iteration-2 fixed raw diagnostics | Raw-diff2 strata, target-pair and operand-repeatability analyzers with their exclusive tests | Current K4 parity routing, operand validators and production numerical tests |

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

The fixed case-10 top-24 Pmax-error capture panel and its data-integrity tests
are preserved in the [experiment archive](https://github.com/ma-gilles/recovar-experiments/tree/dec2c992dc986095e5ed61ba38b48496c8cc2ef7/docs/math/em_k1_case10_it2_capture_panel_v1.json).
This retires a historical selection, not the reusable capture or identity checks.

Completed non-scoring causal, counterfactual, repeatability, and preprocessing
panels are preserved with their dedicated summarizers and tests in the
[causal-scorecard archive](https://github.com/ma-gilles/recovar-experiments/tree/58574b101593004d65cea4b14872e795335ad475/snapshots/em_causal_scorecards_20260921).
The main repository retains the current K1, exactly-K4, VDAM, and real-data
scientific scorecards.

The completed K4 iteration-10 panel12 diagnostic family is preserved
[with its exclusive tests](https://github.com/ma-gilles/recovar-experiments/tree/13f3163/snapshots/k4_iter10_panel12_diagnostics_20260921).
The fixed K4 iteration-2 raw diagnostic family is preserved in a separate
[source-pinned snapshot](https://github.com/ma-gilles/recovar-experiments/tree/578890a/snapshots/k4_iter2_fixed_raw_diagnostics_20260921).

The first-iteration M-step repeatability launcher/analyzer and its recorded result
are [archived](https://github.com/ma-gilles/recovar-experiments/tree/cce8ed8b61bdc754414195f5640608fb9d64e28d/experiments/vdam_mstep_repeatability_20260821).
Use the maintained full-schedule repeat-panel workflow; its tests now retain the
zero-repeat-error case alongside provenance rejection and nonzero repeat ratios.
