# EM / RELION Parity Program Board

This is the living source of truth for the active milestone and next
experiment. Keep permanent rules in `recovar/em/AGENTS.md`, detailed dated
evidence in `docs/math/relion_parity_agent_notes.md`, and accepted completion
runs in `docs/math/em_parity_best_metrics.md`.

## Objective

First achieve near-perfect RELION quality parity for supplied-map K=1
auto-refine and K=4 3D classification. Then optimize to near RELION speed while
holding the accepted quality checkpoint. Treat native InitialModel/VDAM parity
as the next product milestone rather than mixing it into the first closure.

## Mode Contract

- **Strict oracle:** the default during parity closure; pinned RELION GUI
  behavior and full iteration trajectory, including `firstiter_cc` hard-winner
  semantics.
- **Quality:** a later opt-in during parity closure; an intentional RELION
  difference is acceptable only when named and FSC/FSC-AUC against GT is
  neutral or better.
- **Performance:** exact accepted quality behavior with timing instrumentation;
  no algorithmic approximation without separate quality qualification.

## Current State — 2026-07-11

Authoritative clean candidate checkout:
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_em_parity_20260711/recovar`

Immutable broad-candidate checkpoint:
`a6d1d086d81fe7d2be863c50bad33c7ea85e0b7f` on
`codex/em-parity-checkpoint-20260711`. The original dirty checkout remains
unchanged at
`/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_em_min_deferred_abs2_20260709_1745`.

Base HEAD before this board was created:
`4fba8f48a00ca7820a763e7ba41dac4a5a8d8242` on
`codex/em-deferred-bigjit-abs2-min-20260709` with a large dirty candidate stack.
Every new run must record a fresh diff SHA-256 and untracked manifest.

Known evidence:

- K=1 100k/256 map quality is excellent: merged RECOVAR-vs-RELION correlation
  `0.999571`, FSC-AUC `0.994387`, and RECOVAR GT FSC-AUC is `+0.009573` above
  RELION. RECOVAR is about `1.40x` RELION wall time on the recorded run.
- K=1 free-trajectory Pmax diverges beginning at iter 2, but an exact fixed
  RELION it001-to-it002 replay of worst particles gives Pmax correlation `1.0`,
  mean absolute gap `0.000309`, max `0.000917`, and exact pose/translation.
  The earliest proven state-history bug was iter-1 hard-winner Pmax assembly:
  RECOVAR used the correct WTA reconstruction path but recomputed Pmax from
  incompatible coarse/fine score normalizations, yielding mean `2.03e-6`
  instead of RELION's exact `1.0`. A CPU regression now pins the fix; GPU
  trajectory validation is pending.
- K=4 100k/256 map quality is close/better by GT FSC-AUC, but particle-level
  state parity is incomplete: recorded class agreement `0.89025`, pose within
  5 degrees `0.71669`, translation within 1 px `0.77529`. Runtime is `2.181x`
  RELION; sparse K-class pass 2 dominates the completed iteration wall.
- Exact local x-half current/full BPref microbatching now survives the recorded
  3k/128 stress case without OOM. The conservative cap is validated for that
  fixture, not yet a universal optimal cap.
- The repaired 3k/128 strict trajectory matches RELION's ten numbered
  iterations, convergence decision, current-size schedule, and final all-data
  branch. Numbered map correlation rises from the qualified iter-1 tie result
  (`0.995764`) to `0.999775` at iter 2 and `0.999912` at iter 10. The first
  final-map report was semantically invalid because it compared RECOVAR's
  all-data output to RELION's numbered half maps; the true unnumbered RELION
  final comparison is correlation `0.987438`, FSC-AUC `0.980260`. The final
  replay log exposes a concrete stale-state bug: final all-data requested
  replay index 10 but the harness only populated through index 9.

Canonical evidence and paths are in `docs/math/relion_parity_agent_notes.md`
and `docs/math/em_parity_best_metrics.md`.

## Quantitative Gates

These are program gates, not arbitrary test tolerances. Change them only by an
explicit user decision.

### Fixed-state arithmetic

- score/Pmax p95 absolute gap `<=1e-4` where RELION GPU arithmetic permits;
- maximum gap `<1e-3` unless a CPU/double adjudication explains it;
- exact best pose/class/translation agreement when the winning margin is above
  the numerical band; near-tie flips require candidate score/posterior evidence
  that the inputs agree within the numerical contract;
- no systematic drift by half, class, shell, pass, or candidate count.

### K=1 supplied-map quality

- merged RECOVAR-vs-RELION FSC-AUC `>=0.995`;
- RECOVAR GT FSC-AUC no worse than RELION by more than `0.002`;
- shellwise FSC curves and the established FSC score/resolution summaries
  versus both GT and RELION show no unexplained systematic deficit;
- strict-mode per-iteration state differences are arithmetic-level after the
  first-iteration policy is matched;
- convergence iteration and final all-data path agree exactly.

Map correlation is recorded only as a weak diagnostic. It is never a K=1
quality gate and cannot override the FSC/FSC-AUC decision in either direction.

### K=4 supplied-map quality

- every Hungarian-matched RECOVAR-vs-RELION class FSC-AUC `>=0.995`, with
  shellwise FSC curves and established FSC score/resolution summaries reported
  per class;
- per-class GT FSC-AUC no worse than RELION by more than `0.002` without a
  documented quality-mode improvement;
- class agreement `>=99%`, with pose/translation distributions reported per
  class and no collapsed/minority class;
- convergence/finalization semantics agree.

Map correlation is recorded only as a weak diagnostic. It is never a K=4
quality gate, and class averaging must not hide a poor per-class FSC result.

### Performance

- quality freeze first, then intermediate K=4 target `<=1.5x` RELION and K=1
  target `<=1.2x`;
- completion target K=4 `<=1.2x` and K=1 `<=1.1x` on the same pinned hardware;
- report compilation separately and include end-to-end time, per-stage time,
  throughput, and peak memory.

## Milestones And Exit Criteria

1. **Freeze reproducible oracle and candidate.** Create a clean checkpoint or
   reviewable logical commit series from the current stack; pin RELION build,
   fixtures, commands, and hardware. Exit when any result can be reproduced
   from immutable identities.
2. **Close K=1 strict trajectory parity.** Implement or qualify strict
   `firstiter_cc` semantics, compare every state boundary, and match
   convergence/finalization. Exit when K=1 gates pass across small robustness
   cells, at least one real-particle confirmation, and the 100k/256 completion
   case.
3. **Close K=4 quality and state parity.** Find first divergence before final
   maps, repair class/pose/translation trajectory and finalization, and cover
   class imbalance/noise/CTF stress. Exit when K=4 gates pass.
4. **Freeze quality checkpoint.** Tag/commit the accepted behavior and lock a
   reproducible K=1/K=4 benchmark matrix. No performance patch proceeds when
   its quality comparison is missing.
5. **Optimize K=4.** Attack measured sparse pass-2 and M-step/noise bottlenecks
   one at a time with output equivalence tests.
6. **Optimize K=1.** Reduce pass-2/local overhead, compilation, and memory
   traffic while retaining the quality checkpoint.
7. **Expand scope.** Native InitialModel/VDAM, broader distributions, larger
   boxes/counts, and additional GPU architectures after supplied-map closure.

## Active Milestone

Milestone 1 now has an immutable local checkpoint. The active scientific target
is Milestone 2; future changes must be small logical commits on top of the
checkpoint.

### Next experiment

The complete 3k/128 strict firstiter A/B is finished. Strict Pmax is exactly
one particle-by-particle and fixed RELION-it1 arithmetic is in the numerical
band. The iter-1 map correlation is `0.995764`, but correlation is a weak
diagnostic rather than a quality gate. The active hypothesis is that the first
material FSC residual is already present in the iter-1 reconstruction
accumulators.

The iter-1 accumulator boundary is classified. BPref complex averages and
weights agree at arithmetic level for typical coordinates, with small outer
shell/error tails. All five materially different winners have complete
coarse score comparisons, and both same-parent fine flips have coherent fine
comparisons. Every flip is a demonstrated numerical tie; no unexplained
support, frame, score, or accumulator mismatch remains. Retain the iter-1
correlation as diagnostic context, but judge the map only by its shellwise FSC,
FSC-AUC, and FSC score/resolution evidence.

The ten-iteration trajectory is complete and its numbered states are stable.
The stale final-state fix is validated: final all-data now uses RELION
`run_it010` without fallback, improving true-final FSC-AUC from `0.980260` to
`0.991498`, while the fixed-state numbered iter-10 map has diagnostic
correlation `0.999995`. Final particle medians are effectively exact, but
angular p95 is `0.631` degrees and Pmax correlation is only `0.7434` with mean
absolute gap `0.0423`. This localizes the active hypothesis to final
fine-posterior/support or its BPref accumulation, not convergence history,
numbered reconstruction, or output grid correction.

The final BPref and score boundary is classified. RELION's matched final dump
uses 24 fine rotations for a representative same-pose/Pmax outlier, while the
RECOVAR default expands 1,392; candidate counts are 169 versus 4,926 positive
and Pmax is `0.8333` versus `0.2519`. The existing pruned-parent path restores
24 rotations, 156 positive candidates, Pmax `0.8445`, and 6 retained samples
versus RELION's 5. This improves true-final FSC-AUC from `0.991498` to
`0.994527` grid-off and canonical 63-shell FSC-AUC `0.995784` with strict
RELION grid correction. Grid-on RECOVAR-vs-GT is only `0.000887` below RELION,
within the `0.002` gate. This small-cell final map therefore passes the FSC
quality contract; correlation is diagnostic only.

RELION pruned-parent support is now the K=1 local adaptive default, with
full-parent retained as an explicit diagnostic override. The four-way final
cross-replay shows tau2 substitution changes FSC-AUC by less than `1e-5`, while
substituting the RELION BPref accumulator raises strict-oracle FSC-AUC from
`0.997003` to `0.999684`. This exonerates final tau2/Wiener reconstruction and
localizes the remaining measurable high-shell residual to BPref accumulation.

Full clean A100 trajectory validation is running as Slurm job `10990444` from
the canonical-reporting checkpoint; the earlier H100 request `10989654` never
started and was replaced because the pinned RELION oracle ran on A100. Require
the numbered schedule/convergence to remain exact and compare final particle,
BPref, shellwise FSC, FSC-AUC, FSC score/resolution, and diagnostic correlation.
If it passes, advance to additional K=1 robustness seeds/stress cells rather
than treating radial amplitude or correlation as a quality blocker.

In parallel only when authorized: audit K=4 per-iteration dumps to identify the
first class/pose/state divergence; do not optimize sparse pass 2 until that
quality boundary is known.

## Decision Log

Resolved with the user on 2026-07-11:

- RELION GUI semantics are the default until strict parity closes. Major
  behaviors remain typed, switchable options so later grid-correction,
  angle-refinement, and other scientific ablations do not require rewrites.
- Full K=1 trajectory parity closes before K=4; supplied-map EM closes before
  native InitialModel/VDAM.
- Discrete comparisons are tie-aware. A winner flip is acceptable only when
  underlying scores/posteriors prove a numerical near-tie. Convergence and
  finalization are expected to match exactly.
- K=1 closure spans multiple seeds, white/colored noise, CTF/no-CTF,
  uniform/preferred angular distributions, contrast/noise-scale variation,
  translation stress, and junk/outliers, followed by a well-characterized
  real-particle confirmation and a 100k/256 completion run.
- At scale, compare complete aggregate iteration state plus stratified score
  surfaces, automatically dumping every mismatch for full investigation.
- RECOVAR and RELION timing pairs use the same GPU model. Any available cluster
  GPU class is valid; local use is capped at three confirmed-idle GPUs.
- Up to three subagents may perform independent bounded investigations, with
  one writer per source area and the primary agent owning integration.
- Preserve the existing dirty candidate and create a separate clean local
  checkpoint/logical commit history. Do not push without separate approval.
- Long-lived EM development checkouts use
  `/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/`; bulky run outputs and runtime
  caches remain under `/scratch/gpfs/CRYOEM/gilleslab/em_work/`.

Record each resolved decision here with date, rationale, and effect on gates.

## Experiment Record Template

```text
Date / hypothesis:
Mode: strict | quality | performance
Commit / branch / dirty SHA-256 / untracked manifest:
RELION commit/build / command / MPI / GPU:
Fixture / seed / particle count / box / K:
RECOVAR command and environment overrides:
Slurm jobs / node / logs / artifact root / SAFE_TO_DELETE:
First divergence boundary:
Quality metrics and deltas:
Performance metrics and deltas:
Result: supported | falsified | inconclusive
Regression added:
Next cheapest discriminating experiment:
```

## Queue Discipline

Keep at most one active quality hypothesis and one independent read-only audit.
Rank backlog items by expected information gained per GPU-hour. A run without a
predeclared decision it can change should not be submitted. Negative results
must be recorded so future agents do not repeat them.
