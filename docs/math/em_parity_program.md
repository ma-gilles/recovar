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

Current accepted code checkpoint: `f91ba865db652860eea15a2cde66ab11b4f05b72`
on `codex/em-parity-checkpoint-20260711`.

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

The complete 3k/128 strict firstiter A/B is finished. CUDA texture projection
matches all 3,000 RELION iter-1 orientations exactly. Correlation remains a
weak diagnostic rather than a quality gate; iter-1 and all later maps are
judged only by shellwise FSC, FSC-AUC, and FSC-derived score/resolution.

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

Full clean A100 trajectory job `10990444` completed the exact ten-iteration
schedule `[56,56,66,68,80,80,80,80,80,80]`, convergence at iteration 10,
and final all-data branch. The earlier H100 request `10989654` never started
and was replaced because the pinned RELION oracle ran on A100. Numbered iter-10
RECOVAR-vs-RELION FSC-AUC is `0.999324`, and its GT FSC-AUC delta is only
`-0.000037`, but the free-trajectory unnumbered final FSC-AUC falls to
`0.988116`. This fails strict final map parity even though RECOVAR remains
better against GT (`0.669009` versus RELION `0.650835`). Do not launch the
robustness matrix yet.

Final-only job `10992173` enters the final pass directly from the saved free
iter-10 half maps and exactly reproduces `0.988115`, proving no hidden state
history after iter 10. Merged-reference diagnostic `10992266` is worse at
`0.981072`, falsifying early half-reference joining. Exact-RELION-iter1 seeded
job `10992371` reaches final FSC-AUC `0.994488`; iter-1 ties explain most but
not all of the free residual.

The current strict path now also matches RELION's joined final noise semantic:
both particle halves use half-1 `sigma2_noise` only in the post-convergence
K=1 all-data E-step. Exact dumped operands then match a representative RELION
posterior within the fixed-state numerical contract. Fixed-final job
`10994996` still reaches only grid-off FSC-AUC `0.994497`, while RECOVAR GT
FSC-AUC remains better (`0.669846` versus `0.650835`). Matched numbered iter-10
BPref job `10996603` passes its patched-oracle FSC gate and localizes a small
half-2 difference to four missing sub-winner significant samples, not noise,
half joining, mapping, or winner poses. Exact-RELION-iter10 final-only job
`10997070` remains at FSC-AUC `0.994501`, falsifying the tiny numbered-map
difference as the remaining final limiter. Its full final Pmax mean/p95/max
absolute errors are `0.0282/0.0898/0.4592`; original particle 428 / RELION
stack 429 is now the worst case.

The matched stack-429 operand replay has now classified that boundary. RELION
zeros the redundant `kx=0, ky<0` rows in its non-redundant half-plane, while
RECOVAR's full-size local likelihood counted those conjugate rows a second
time. Applying the RELION axis mask offline changes the parent Pmax from
`0.657449` to `0.812126` versus RELION `0.811969`, and restores exactly the 10
parent pairs retained by RELION. In particular it restores the otherwise
missing `(RELION rotation 140, coarse translation 4)` pair. Expanding that
pair and applying the same mask changes the final fine Pmax from `0.173354` to
`0.628355` versus RELION `0.628361`. The shared fine-candidate posterior L1
error falls from `0.3785` to `0.0288` even before the restored parent pair is
added. A centralized scoring-weight fix and focused unit regression are now
present in the dirty candidate.

Exact-RELION-iter10 final-only A100 job `11001328` qualifies the axis-mask
patch. Canonical RECOVAR-vs-RELION FSC-AUC is `0.997302`; the minimum non-DC
shell FSC is `0.995021`. RECOVAR-vs-GT FSC-AUC is `0.670396`, which is
`+0.019561` above RELION, and its FSC=0.5 crossing is one shell better (41
versus 40). RECOVAR is lower than RELION against GT in only three very-low
frequency shells, with worst delta `-0.000266`, inside the arithmetic band;
the other 59 non-identical shells are higher. The fixed-final K=1 small-cell
quality gate therefore passes without grid correction.

Clean free-trajectory A100 job `11002266` reproduces the RELION current-size
schedule `[56,56,66,68,80,80,80,80,80,80]`, convergence at iteration 10, and
the final all-data branch, but the unnumbered final remains below gate at
RECOVAR-vs-RELION FSC-AUC `0.990397`. This improves the pre-mask free result
`0.988116`, while RECOVAR-vs-GT remains better (`0.669518` versus `0.650835`).
Numbered merged FSC-AUC is already `0.997007` at iteration 2 and rises to
`0.999339` at iteration 10. Do not launch robustness.

Exact-RELION-iter1 seed job `11007539` runs numbered iterations 2--10 plus
final and passes at canonical RECOVAR-vs-RELION FSC-AUC `0.997271`.
RECOVAR-vs-GT FSC-AUC is `0.670338` versus RELION `0.650835`. This closes the
later-trajectory hypothesis and localizes the remaining free residual to the
iteration-1 boundary.

The first free run with the Gaussian redundant-axis fix exposed a scoped
regression: 198/3000 iteration-1 orientations and 219/3000 translations no
longer matched. RELION normalized-CC scores every pixel in its rectangular
first-iteration FFTW crop; only Gaussian likelihood scoring removes centered
`kx=0, ky<0` redundant rows. A score-mode-specific correction retains all CC
rows while preserving the qualified Gaussian mask. A100 job `11013677`
restores byte-identical coarse and fine hard assignments to the prior exact
texture run: every orientation matches RELION and only the established
0.5-pixel translation tie remains. Job `11013457` was an infrastructure-only
failure (`CUDA_ERROR_NO_DEVICE`) before science on `della-l07g3`.

Clean score-mode-scoped free-trajectory job `11014763` reproduces RELION's
current-size schedule `[56,56,66,68,80,80,80,80,80,80]`, convergence at
iteration 10, and final all-data path, but final canonical
RECOVAR-vs-RELION FSC-AUC is only `0.990351`. RECOVAR remains better against
GT (`0.669412` versus `0.650835`). Exact poses therefore do not by themselves
close the iteration-1 seed error.

The first divergent iteration-1 state is now identified. RECOVAR calculated
tau2 before applying the `firstiter_cc --ini_high` cutoff and retained nonzero
shells 20--28. RELION applies a squared raised-cosine taper to tau2 and
data-vs-prior after first-iteration reconstruction. The candidate implements
that source-matched taper. It also matches the pinned RELION accelerated GPU
build's single-precision `XFLOAT` BPref accumulator; the earlier float64
one-particle comparison used RELION's CPU/double backprojector and did not
represent the production oracle.

One-iteration A100 job `11021943` gives tau2 shell 18 `106.849670`, shell 19
`0.0235179`, and shells 20 onward zero, versus RELION `106.808`, approximately
`0.0235`, and zero. Merged iteration-1 RECOVAR-vs-RELION FSC-AUC over RELION's
supported shells 1--18 improves from `0.996052` to `0.998430`; shell 18
improves from `0.908735` to `0.948464`.

Clean A100 job `11023037` completes in `579` seconds with the exact schedule,
iteration-10 convergence, and final all-data path. Final canonical
RECOVAR-vs-RELION FSC-AUC improves from `0.990351` to `0.994646`, narrowly
missing the unchanged `0.995` gate; RECOVAR-vs-GT remains better at `0.670285`
versus `0.650835`. Numbered merged-map FSC-AUC is already `0.997721` at
iteration 2 and `0.999746` at iteration 10.

Post-rotation-only cutoff job `11025153` is a null result: supported-shell
iteration-1 FSC-AUC remains `0.9984304911` and shell 18 remains `0.948464267`.
The candidate is reverted. Downsampled BPref shell sums are already effectively
exact, including shell 18, which moves the first residual after reconstruction.

Source inspection identifies an ordering mismatch. RELION reapplies the
`ini_high` Fourier low-pass inside maximization, then calls real-space
`solventFlatten` from the outer iteration loop. RECOVAR did those operations
in reverse order. They do not commute because the final real-space mask
reintroduces a small high-shell tail. The next cheapest experiment corrects
that order and reruns one iteration.

One-iteration A100 job `11025949` confirms the fix. Canonical full-shell
RECOVAR-vs-RELION FSC-AUC is `0.999538`; supported-shell 1--18 FSC-AUC is
`0.999930`, shell 18 is `0.998800`, and the minimum non-DC shell is
`0.996857`.

Clean full-trajectory A100 job `11026304` passes the small-cell strict gate.
It completes in `579` seconds with the exact current-size schedule, convergence
at iteration 10, and final all-data path. Final RECOVAR-vs-RELION FSC-AUC is
`0.997260`; minimum non-DC shell FSC is `0.994984`, fifth percentile is
`0.995371`, and the last-ten-shell minimum is `0.996734`. RECOVAR-vs-GT
FSC-AUC is better (`0.670484` versus `0.650835`); only GT shells 1--3 are
lower, with worst delta `-0.000266`, well inside the `0.002` gate.

The active milestone now advances to K=1 robustness: run source-matched
RECOVAR/RELION pairs across high noise, nonuniform/Kent angles, no CTF,
outliers, contrast/noise-scale variation, and translation stress before the
10k/real/100k confirmations. Any failing cell returns to first-divergence
debugging; K=4 remains gated on this K=1 robustness step.

The eight-cell 3k/128 robustness matrix is now closed on the intended final
product. Jobs `11027056`--`11027063` established six direct passes and exact
convergence/finalization. The apparent failures in heterogeneous
contrast/noise-scale cases 18 and 22 were a parity-harness reporting bug:
`run_multi_iter_parity.py` discarded the joined all-data reconstruction in
`result["mean"]` and instead averaged the two separately Wiener-regularized
half reconstructions. Those operations are not equivalent. Production
`run_full_refinement.py` already saved the joined reconstruction correctly.
Commit `f91ba865` fixes the harness and adds a regression.

Focused A100 validations `11032906` and `11032907` pass after that correction.
Case 18 has RECOVAR-vs-RELION FSC-AUC `0.995571`, minimum non-DC shell FSC
`0.988215`, and GT FSC-AUC `0.765648` versus RELION `0.751884`. Case 22 has
RECOVAR-vs-RELION FSC-AUC `0.996966`, minimum non-DC shell FSC `0.991276`, and
GT FSC-AUC `0.335789` versus RELION `0.326059`. Their worst shellwise GT deltas
are only `-0.000632` and `-0.000188`, respectively. Thus all eight robustness
cells pass the aggregate FSC-AUC, GT-quality, convergence, and finalization
gates. The active K=1 step advances to a 10k intermediate-scale confirmation,
then a characterized real-particle case and the pinned 100k/256 completion
case; K=4 remains gated until those K=1 confirmations pass.

The two-cell 10k/128 intermediate-scale matrix is also closed for map quality,
convergence, finalization, memory, and matched-A100 timing. Immutable setup,
case, and summary jobs `11033444`--`11033447` ran from commit `3cbfd9ea` under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_mid10k_strict_retry_20260711_235500`
with separate marked runtime scratch at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_mid10k_strict_retry_20260711_235500`.
The summary job passed its exact-HEAD and clean-worktree gates.

Uniform/white case 31 matches RELION convergence at iteration 13 and the final
all-data branch. Final RECOVAR-vs-RELION FSC-AUC is `0.998723`; minimum non-DC
shell FSC is `0.997648`, fifth percentile is `0.997682`, and the last-ten-shell
minimum is `0.997653`. RECOVAR-vs-GT FSC-AUC is `0.818646` versus RELION
`0.801765`, with worst shellwise GT delta only `-0.000453`. RECOVAR refinement
wall is `1009.8` seconds versus RELION `1702` seconds on matched A100s, a
RECOVAR/RELION ratio of `0.593`.

Kent/radial-noise-3 case 32 matches RELION convergence at iteration 11 and the
final all-data branch. Final RECOVAR-vs-RELION FSC-AUC is `0.998250`; minimum
non-DC shell FSC is `0.996442`, fifth percentile is `0.996871`, and the
last-ten-shell minimum is `0.997256`. RECOVAR-vs-GT FSC-AUC is `0.272194`
versus RELION `0.268373`. The localized GT-shell swing at shells 3--5 is not a
map-parity deficit: RECOVAR-vs-RELION FSC is `0.999743` or better over shells
1--5, and RECOVAR is better in aggregate and through the later signal-bearing
shells. RECOVAR refinement wall is `950.9` seconds versus RELION `1154`
seconds, a ratio of `0.824`.

The remaining 10k state tail is an explicit diagnostic, not a map-quality
failure. Final pose p95 is arithmetic-level in both cells and every pose is
within 5 degrees, but case 31 Pmax p95/max absolute gaps are
`0.01166/0.254996` and case 32 gaps are `0.002476/0.044970`. Case 32 has 70
adjacent-fine-grid pose flips above 1 degree and three translation differences
above 0.5 pixel. Before the real-particle and pinned 100k gates, adjudicate
representative particles with fixed-state RECOVAR score/posterior dumps and an
uninterrupted instrumented RELION run; continuation dumps are forbidden by the
previously demonstrated finalization-state confound.

The first characterized real-particle gate is open and currently fails strict
parity. A deterministic 10k-particle EMPIAR-10076 subset (seed `20260712`,
exactly 5000 particles per half) lives under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_10k_fixture_20260712`
with a `SAFE_TO_DELETE` marker and manifest. The shared RECOVAR mean volume is
used only as the common initializer; there is no pseudo-GT, so the quality
gate is RECOVAR-vs-RELION shellwise FSC/FSC-AUC plus half-map and state parity.

The first A100 pair exposed a real-subset indexing bug before RECOVAR science.
`half1_idx` and `half2_idx` are row positions in the subset STAR, but replay
overrides treated them as original stack IDs. Synthetic contiguous fixtures
hid the bug. Commit `c3b3a27e` maps each input row through `rlnImageName`,
validates missing and duplicate stack IDs, and adds a shuffled non-contiguous
regression. The full override unit file passes (`38 passed`). Corrected
RECOVAR-only job `11039455` completed from that commit on the same A100 model
as RELION; failed jobs `11039371` and `11039400` were pre-science
`CUDA_ERROR_NO_DEVICE` failures on unhealthy `della-l07g3`, which is excluded
from the corrected runs.

RELION converges at numbered iteration 17 and performs the final all-data
branch; RECOVAR reproduces the same control decision and final branch. The
underlying real-data state is not yet strict, however. Iteration-2 mean Pmax is
`0.0716` in RECOVAR versus `0.112067` in RELION, accompanied by a transient
support tail: median fine support is 32 rotations, but a few particles retain
up to 200704 rotations. Iteration 2 takes `899.3` seconds. The tail collapses
by iteration 3 and mean Pmax nearly recovers by iteration 4 (`0.6624` versus
RELION `0.663427`), local iterations take 31--94 seconds, and final mean Pmax
matches (`0.243046` versus `0.243119`). Per-particle final state still fails:
mean/p95/max absolute Pmax gaps are `0.082570/0.226190/0.570284`; pose p95 is
`0.759678` degrees and translation p95 is `0.641700` pixel.

Final map parity fails and must not be waived by the high correlation. Final
RECOVAR-vs-RELION FSC-AUC is `0.863672` while correlation is `0.988477`.
Truncated FSC-AUC is `0.998450` through shell 16, `0.992968` through shell 32,
and `0.964539` through shell 64, showing a smooth signal-band phase loss rather
than a single corrupt shell. RECOVAR wall time is `2375.7` seconds versus
RELION `1330` seconds; the ratio is `1.786`, dominated by the iteration-2
support transient. Artifacts and summary are under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_10k_strict_20260712_010000`
(`summary_retry.md` and `summary_metrics_retry.json`). The next quality
hypothesis is an iteration-2 firstiter-CC support/normalization mismatch:
select the largest per-particle Pmax/support residuals, freeze the same RELION
state and maps, and compare raw scores, priors, support masks, log-normalizers,
and winners before making performance changes.

Fixed-state jobs refine that boundary further. One-iteration replay job
`11041292` starts from RELION iteration-1 maps and state and produces
iteration-2 FSC-AUC `0.999321`, mean Pmax `0.112001` versus RELION `0.112067`,
mean absolute per-particle Pmax gap `0.000494`, and pose agreement through p99.
Its mean fine support is only `298.5/322.9` rotations per half, versus
`1180.9/1112.4` in the free trajectory. Thus iteration-2 scoring is not the
primary source; it amplifies a preceding map difference. Direct iteration-1
job `11041546` proves the first divergence: Pmax is exactly 1 for every
particle, poses/translations match through p99, but the iteration-1 merged-map
FSC-AUC is only `0.988635`. The active hypothesis is therefore iteration-1
M-step support/weights, BPref accumulation, or reconstruction/filtering.
Compare the saved RECOVAR `Ft_y`, `Ft_ctf`, regularized/unregularized maps,
tau2, and noise under `iter1_map_diag/output/intermediates` against an
uninterrupted, identity-validated RELION iteration-1 BPref dump.

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
