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

## Current State — 2026-07-12

Authoritative clean candidate checkout:
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_em_parity_20260711/recovar`

Current accepted code checkpoint: `536d6bd9de5f2dab900ec5cdd6ad055be728840b`
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

That accumulator boundary is now closed against the exact real-data oracle.
An isolated RELION `d476e6f` build adds a single environment-gated dump after
MPI combination, symmetry, and the 40-A low-resolution half join but before
reconstruction.  Installed-d476 stop-one versus the uninterrupted oracle,
installed versus patched with the environment unset, and patched environment
off versus on all pass at half-map FSC-AUC above `0.9999999996`, minimum
non-DC shell FSC above `0.9999999940`, maximum real-space delta
`1.8626451e-9`, and bit-exact Pmax/pose/translation arrays for all 10,000
particles.  Jobs `11042379`, `11042605`, and CPU analysis retry `11043664`
produced the qualified dump under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_d476_bpref_real10076_identity_20260712_020200`.

The RECOVAR and RELION post-join BPref accumulators are close but not
identical: complex-average coordinate relative-error medians are about
`2.0e-4`/`2.2e-4` for halves 1/2 (p95 `0.0449`/`0.0569`), while weight
medians are `3.9e-6`/`4.8e-6` (p95 `0.0063`/`0.0091`).  The coordinate frame
is unambiguous: permutation `(1,0,2)`, signs `(1,1,1)`, complex sign `-1`.
However, decisive cross-replay jobs `11043878` and corrected `11043975`
exonerate this residual.  Replaying the saved RECOVAR accumulators reproduces
the saved RECOVAR merged map at FSC-AUC `0.99999998`; replacing only the
accumulators with exact RELION values changes the replay by FSC-AUC only
`~7e-6` (`0.9999930` between replacement maps) and changes the comparison to
RELION only from `0.9886340` to `0.9886556`.  The real-data iteration-1
failure is therefore downstream of accumulation, in reconstruction or
post-processing.  Stage ablation job `11044058` confirms that solvent masking
is required and grid-on is the closer branch, but neither the raw,
initial-low-pass, masked, nor grid on/off RECOVAR variants closes the gap.
The next boundary is an identity-validated RELION dump immediately after
reconstruction, after the iteration-1 low-pass, and after solvent flattening.

Exact stage-map job `11044321` closes that boundary.  The stage hook is
observationally inert: patched environment-off versus environment-on half-map
FSC-AUC is above `0.99999999982`, minimum non-DC shell FSC is above
`0.9999999968`, and all particle/model arrays are exact.  RELION's captured
post-solvent maps are bit-exact with its written iteration-1 maps.  With exact
RELION accumulators, however, the first mismatch is already the raw
reconstruction: supported-shell FSC-AUC is only about `0.6003/0.6009` for the
two halves and `0.5968` merged.  The supported post-low-pass comparison rises
to about `0.9596`, while the mask turns the phase error into the familiar
full-shell final FSC-AUC `0.988656`.  Thus neither low-pass nor solvent-mask
ordering is the primary source.

The initial hypothesis that RECOVAR omitted RELION's default iterative
preweighting is false.  Exact-d476 binding probe `11045912` proves that RELION
5.0.1 defaults to `skip_gridding=true`; its native closed-form skip branch,
using the pre-filter RELION tau operand, reproduces the captured
post-reconstruct maps at supported-shell FSC-AUC approximately
`0.99999999999999`, full FSC-AUC above `0.99999999996`, relative L2 about
`1.2e-7`, and maximum real-space delta about `6.3e-9`.  Setting zero
preweight iterations with `skip_gridding=false` is identical, while enabling
ten iterative preweight steps is slightly worse and substituting the saved
post-hoc filtered tau is materially worse.

Source audit then identifies the smallest correction.  RECOVAR currently
applies `_firstiter_cc_ini_high_tau2_taper` to
`mean_signal_variance_per_half` before reconstruction.  RELION reconstructs
with the untapered tau, calls `initialLowPassFilterReferences` on the map, and
only then tapers tau2/data-vs-prior for the model/reporting state; its source
explicitly notes that those tapered values are not used in calculations.
Preserve untapered per-half tau for reconstruction, create the tapered state
copy afterward, and retain the existing post-reconstruction map low-pass and
solvent mask.  Do not add iterative preweighting or a C++ production solver
unless the corrected direct-JAX stage replay still leaves a measured gap.

The corrected direct-JAX stage replay and production iteration now pass.
Checkout-bound A100 replay `11046061` uses exact RELION accumulators plus
RECOVAR-computed untapered tau and reaches merged post-reconstruct
supported-shell FSC-AUC `0.999998526`, active post-low-pass FSC-AUC
`0.999999998`, and final solvent-masked full FSC-AUC `0.999999993` with
minimum non-DC FSC `0.999999844`.  Patched production job `11046453` reruns
the real 10,000-particle first iteration end to end and reaches merged
FSC-AUC `0.999991962`, minimum non-DC FSC `0.999984135`, and half-map
FSC-AUC `0.999984760/0.999991332`; Pmax is bit-exact for all particles and
pose/translation p95 errors remain at the numerical-noise scale.  Earlier
job `11046279` is rejected because direct script execution resolved submodules
from a different editable checkout; the corrected runner uses module execution
and asserts the exact `iteration_loop.py` and CUDA wrapper paths.

Full patched trajectory `11046636` matches RELION's convergence at numbered
iteration 16 and executes the final all-data branch, but correctly fails the
map gate: merged FSC-AUC is `0.859718` and half-map FSC-AUCs are
`0.835102/0.843042` despite diagnostic correlation `0.988109`.  The first
post-fix divergence is not iteration-2 scoring.  Map-only cross-replay
`11047995` starts from the corrected RECOVAR iteration-1 half maps plus exact
RELION iteration-1 state and obtains iteration-2 mean Pmax `0.112050` versus
`0.112067`, merged-map FSC-AUC `0.999299`, pose p95 zero at printed precision,
and translation p95 zero at printed precision.

The failed full trajectory did not start from the same iter-0 particle state.
Its iteration-1 mean angular error is about 92 degrees and translation error
about 11.7 pixels; tau2 already differs by up to 59.3% across active shells.
`run_full_refinement --relion_init_dir` loaded iter-0 noise/tau but omitted the
run_it000 particle pre-centering offsets (about 8.5 pixels mean absolute
component), previous orientations, image/scale corrections, and direction
prior.  Commit `a530ec6f` reuses the typed replay loader to install that
complete run_it000 state in override slot 0.  Unit replay tests pass `8/8`.
One-iteration A100 gate `11048426` then reaches merged FSC-AUC `0.999991954`,
half-map FSC-AUCs `0.999984759/0.999991342`, minimum non-DC merged FSC
`0.999984127`, and exact Pmax/rotation/translation arrays relative to the
qualified corrected runner.  Two-iteration A100 gate `11048692` then closes
the next handoff: half-map FSC-AUCs are `0.999051092/0.999183696`, merged
FSC-AUC is `0.999318831`, and minimum non-DC shell FSC is at least
`0.975355`.  Mean Pmax is `0.1120446` versus RELION `0.1120673`; pose and
translation p95 errors are at printed numerical precision, with rare discrete
tail changes retained for tie-aware inspection.  Corrected full real-data job
`11049135` now runs from clean commit `2e3cc620` and must match numbered
iteration-16 convergence plus the normal final all-data branch before this
trajectory is closed.

Job `11049135` matches that control flow exactly but does not close map parity:
it converges at numbered iteration 16, runs the normal iteration-17 all-data
branch with grid correction off, and reaches merged FSC-AUC `0.974017` plus
half-map FSC-AUCs `0.947021/0.944168`.  This is much better than the stale
cold-start run's merged `0.859718`, but still fails the strict `0.995` gate.
The first material post-iteration-2 drift is projector-dependent.  Starting
from the qualified iteration-2 maps and exact RELION state, iteration-3 job
`11050464` with CUDA texture interpolation gives merged FSC-AUC `0.992097`,
Pmax MAE `0.012945`, and mean Pmax gap `+0.000802`.  Same-A100 manual/JAX
projection job `11050495` improves those to `0.995407`, `0.006198`, and
`+0.000189`, respectively, while moving the pose/translation tails closer to
RELION.  Commit `db814243` therefore makes the manual projector the parity
default and keeps texture interpolation as the explicit
`RECOVAR_RELION_PROJECTOR_TEXTURE_INTERP=1` diagnostic.  Full manual-projector
trajectory job `11050804` is the next gate.

Source audit finds the largest texture-path defect: RELION clips each
projection to the smaller of the PPref/model radius and the current image
radius, while RECOVAR texture projection previously enforced only the PPref
radius.  Commit `81681151` adds the missing current-image disk mask without
changing the manual default.  K=4 probe `11051135` moves first-iteration
occupancies from the broken `[2708,1768,2405,3119]` to
`[3216,1710,2200,2874]`, toward RELION's `[3369,1828,2045,2758]`, but does not
close the residual.  Keep texture opt-in until its remaining even-box
Nyquist/coordinate arithmetic is isolated.

K=1 iteration-3 probe `11051461` confirms that the mask removes most of the
texture regression: merged FSC-AUC improves from `0.992097` to `0.994951`,
mean Pmax gap improves from `+0.000802` to `-0.000006`, and Pmax MAE improves
from `0.012945` to `0.007292` in 212 seconds.  The manual path remains slightly
better in map FSC-AUC (`0.995407`) and Pmax MAE (`0.006198`) but takes 357
seconds.  Quality therefore stays on manual projection while the fixed texture
path becomes the leading later performance candidate.

The paired full trajectories close the projector-only hypothesis but not the
real-data quality gate.  Manual job `11050804` and current-radius-masked texture
job `11051785` both reproduce RELION's complete current-size/healpix schedule,
converge at numbered iteration 16, and execute the normal iteration-17
all-data branch with grid correction off.  Manual final half-map FSC-AUCs are
`0.950676/0.948478` and merged FSC-AUC is `0.977975`; masked texture gives
`0.950195/0.945893` and `0.977663`.  Both therefore fail the immutable `0.995`
map gate.  Manual uses 3402 seconds externally versus 2077 seconds for texture
on the same A100 model (peak GPU memory `41156/41150` MiB).  The tiny manual
FSC advantage keeps it as the strict quality default; texture remains the
qualified speed diagnostic, not an accepted quality replacement.

The cumulative residual is a sparse early hypothesis tail, not a mismatch in
the averaged control trajectory or internal gold-standard FSC.  Against each
numbered RELION model, RECOVAR's mean internal half-map FSC differs by at most
`4.99e-4` through iteration 16 and the shellwise mean absolute difference is at
most `0.001073`.  At iterations 1--3, pose p95 remains at numerical precision,
but the fraction above one degree is `0.40%`, `1.09%`, and `1.04%`; the
corresponding mean pose errors are `0.0282`, `0.2255`, and `0.2741` degrees.
The tail grows through the global iterations and is then partly recovered by
local search.  Final manual all-data pose mean/p95 is `0.1650/0.6396` degrees
and translation mean/p95 is `0.0465/0.4538` pixel.  This is too large to waive
as a discrete tie without score evidence even though the majority is exact.

The first tie adjudication targets fixture row 1474,
`19638@particles.256.mrcs`: it is arithmetic-level at iteration 1 but selects
a pose 175.55 degrees away and a translation 2.24 pixels away at iteration 2,
while RECOVAR/RELION Pmax is only `0.040652/0.039912`.  The initial f2c-based
dump job `11053175` is rejected because its newer binary does not reproduce
the d476 oracle.  An exact-source d476 binary was then built with a minimal
score hook (build job `11053568`, binary SHA-256
`f1b27fe6472dac204b579d6163e2c8a0edcb0d6f0ad5904e87fce0078fd339cb`).
Its enabled job `11053649` also fails the mandatory same-binary observational
inertness gate against dump-disabled control `11053938`: schedules match and
iteration-2 map FSC-AUC remains above `0.9999999984`, but target rows 1474 and
5550 flip by `127.78` and `172.19` degrees, row 5550's class field is corrupted,
and one non-target shift changes.  Therefore every score/posterior array from
that hook and the cancelled six-particle RELION panel `11054149` is
scientifically inadmissible.  The audit is in
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_iter2_min_score_d476_20260712_073000/dump_instrumentation_inert_audit.json`.

The independently valid translation-coordinate audit resolves an apparent
candidate mismatch.  RELION pre-shifts by
`B = round_away_from_zero(old_absolute_px)`, scores a relative translation
`t`, then stores `new_absolute_px = B + t`; the inverse is `t = new - B`, not
`new - old`.  Dump-disabled control `11053938` therefore maps row 1474 to fine
translation 56 and row 5550 to 34.  The free texture row-5550 winner maps
exactly to fine translation 95 rather than being one pixel off.  This formula
and the STAR-derived evidence remain admissible, but any accompanying RELION
log-weight comparisons remain rejected until a redesigned hook passes the
enabled/disabled control.  Evidence is in
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_iter2_scoretail_fixedstate_20260712_072000/translation_coordinate_mapping.md`.

RECOVAR exact-state panel `11054150` completes six independently verified
fixture rows on an A100 and provides a fail-closed target set for the eventual
inert RELION comparison.  Its Pmax/top-two log-margin pairs are row 5550
`0.043996/0.072384`, row 5504 `0.003155/0.022116`, row 3102
`0.061285/0.012997`, row 394 `0.116131/0.013222`, row 2813
`0.375569/1.75604`, and row 7710 `0.364002/0.130013`.  Canonical
`source_indices.npy` and the fixture STAR independently confirm every
fixture-row-to-stack-image mapping.  The prepared comparator intentionally
raises until an admissible RELION dump is supplied.  Summary artifacts are in
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_iter2_recovar_panel6_20260712_075000`.

The controlled RECOVAR projector A/B on the exact RELION iteration-1 state
(texture job `11054266`; manual job `11054283`, cancelled after both requested
dumps were complete) selects the same candidate with both arithmetic paths for
both tail rows: `[289443,56]` for row 1474 and `[162979,34]` for row 5550.
Manual versus
texture posterior L1 is `0.008714`/`0.004191`; centered pre-prior score
difference p95 is `0.02995`/`0.02588`, and projection relative L2 is
`0.000889`/`0.000790`.  Thus projector arithmetic perturbs the controlled
surface but does not itself flip these two winners; free-trajectory flips
require amplification through the preceding map/state trajectory.  This is a
RECOVAR-only diagnostic and does not substitute for the rejected RELION score
comparison.

Pixel attribution rules out the remaining texture boundary hypotheses.  In
the same two exact-state dumps, shells 1--10 contain `97.1%/95.7%` of the
manual-versus-texture projection-difference energy, while boundary shell 46
contains only `0.0181%/0.0278%` and all coordinates outside radius 46 are zero
in both paths.  Removing the Nyquist row/column leaves `99.90%/98.46%` of the
candidate score-delta RMS.  The texture residual is therefore interior
interpolation arithmetic concentrated at low frequency, not another
support-radius or even-box boundary error.  The reproducible CPU audit is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_iter2_recovar_manual_p1474_p5550_20260712_075700/projector_pixel_audit.json`.

The first exact-d476 hook failure is fully explained: `ihidden_overs` is
authoritative only in host memory, but the hook copied its uninitialized CUDA
buffer back over that mapping before fine winner selection.  A host-only
replacement removes all extra GPU copies and syncs.  Cross-node jobs
`11054698/11054699` no longer corrupt either target and keep iteration-2 map
FSC-AUC above `0.9999999985`, but naturally differ at one non-target
translation, 13 significant counts, and Pmax by at most `0.000404`.  Same-node
sequential pair `11054601` likewise preserves both targets and all angles/classes
but differs at two non-target one-step translations and 17 significant counts,
so its strict fail-closed audit remains false.  Row triangulation shows the
enabled translations match both the original oracle and clean installed-binary
repeat `11055156`; the disabled control is the outlier.  The clean repeat itself
has iteration-2 Pmax delta at most `0.000126`, 17 one-count differences, no
translation differences, and map FSC-AUC above `0.9999999988`.  These establish
the natural RELION numerical envelope but do not waive the dump gate.

The v3 hook buffers host snapshots and defers all file writes until the
unconditional post-expectation MPI barrier, removing I/O from the
particle-scoring loop.  Same-node continuation gate `11055736` runs disabled
then enabled from the identical installed iteration-1 state and
passes the calibrated admissibility gate: iteration-2 map FSC-AUC is
`0.99999999855/0.99999999886`, minimum non-DC FSC is above `0.9999999826`, no
translation or class differs, maximum angle delta is `3.42e-6` degrees, Pmax
delta is at most `0.000117`, and nine significant counts differ by one.  Both
target states are exact, their Pmax deltas are `1.4e-5`, schedules are exact,
and all 26 dump files are present.  These changes are no larger than clean
installed-repeat `11055156`, so
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_iter2_inert_dump_control_20260712_080200/v3_exact_inertness_audit.json`
sets `score_arrays_admissible=true` for hook inertness on that continuation.
Failed precursor `11055708` stopped before scoring because of the wrong
continuation working directory and has no scientific output.

However, `--continue run_it001_optimiser.star` does not replay the uninterrupted
installed iteration-2 trajectory: the v3 disabled continuation versus oracle
map FSC-AUC is only `0.86405/0.86135` and many particle states differ, likely
because continuation does not restore the same perturbation/RNG sequence.
Thus `11055736` qualifies the hook but its score arrays are not the oracle
surface and must not be compared to RECOVAR's exact-oracle dumps.  A cold v3
two-iteration dump run must first reproduce the uninterrupted oracle target
states and candidate grid.

Cold v3 job `11055888` restores the correct `+0.405200` perturbation and
qualifies the target score surfaces.  Both row-1474 and row-5550 discrete
states match the uninterrupted oracle; target Pmax is exact and within
`2e-6`, respectively.  All 10,000 Euler/class arrays match to a maximum
`2.96e-6` degrees, with one known non-target one-step translation and 14
one-count differences from rebuilt-binary numerical variation.  Half-map
FSC-AUC is `0.9999999093/0.99999999865`, minimum non-DC FSC is at least
`0.9999996170`, and the 26 target arrays are admissible under the separate v3
inertness gate.  Qualification is in
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_iter2_inert_dump_control_20260712_080200/v3_cold_oracle_qualification.json`.

The admissible score comparison reverses the projector conclusion inferred
from final-map FSC alone.  Texture arithmetic is much closer to RELION on both
controlled oracle surfaces.  For row 1474, texture versus manual posterior L1
is `0.000291/0.008132`, centered pre-prior p95 absolute error is
`0.000626/0.029760`, and support symmetric difference is `0/6`.  For row 5550,
the corresponding values are `0.001086/0.004452`, `0.005816/0.025355`, and
`9/5`.  RELION, texture, and manual choose the same controlled winners
`[289443,56]` and `[162979,34]`; texture Pmax and top-two margins are also
closer.  Manual's tiny full-trajectory FSC advantage is therefore a
compensating error, not evidence of closer E-step arithmetic.  Do not optimize
or retain manual as the strict default without reconciling this score boundary.
The full comparison is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_iter2_inert_dump_control_20260712_080200/v3_cold_oracle_score_comparison.json`.

The qualified cold-v3 six-particle panel `11056138` extends that conclusion.
Five rows choose the same winner, with shared-surface posterior L1 between
`0.000704` and `0.002227` and common-centered pre-prior score p95 error between
`0.00262` and `0.00678`.  Row 2813 is the decisive non-tie: RELION chooses
`[289965,48]` at Pmax `0.116213` with top-two log margin `0.184082`, while the
texture RECOVAR path chooses `[284536,4]` at Pmax `0.375569` with margin
`1.75604`.  Only `49.80%` of RECOVAR's mass lies on RELION candidates, although
the shared-candidate centered score p95 error is only `0.003590`.  The bug is
therefore the coarse-parent support supplied to fine pass 2, not a numerical
tie or fine texture-score arithmetic.  Qualification and comparison are in
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_iter2_v3_panel6_20260712_085000`.

RECOVAR support audits localize the mismatch one pass earlier.  For row 5550,
RELION has 4,192 fine candidates and RECOVAR has 4,160; the 64 RELION-only
candidates are exactly two 32-child coarse parents `(22971,19)` and
`(22972,28)`, while RECOVAR adds parent `(2404,9)`.  RECOVAR ranks those parent
pairs 130, 131, and 126 around a rank-129 significance boundary, proving a
coarse-score ordering difference rather than a fine-expansion bug.  The row
2813 projector A/B is stronger: manual coarse projection selects RELION's exact
15 parent pairs and exact 480 fine candidates; texture swaps RELION parent
`(36245,8)` for `(35567,1)`.  Manual fine arithmetic is less accurate than
texture, so strict scoring requires a hybrid: manual supplied-PPref projection
for global coarse significance and texture supplied-PPref projection for fine
pass 2.

Minimal RELION v4.1 support probes passed the calibrated inertness gate
`11056477`; cold oracle run `11056533` preserves perturbation `+0.405200`, the
target state, and Pmax, with half-map FSC-AUC
`0.99999990925/0.99999999862`.  However, the hook queried RECOVAR psi-major
rotation IDs directly in RELION's pixel-major mask.  Its original three-zero
interpretation is coordinate-wrong and rejected.  Applying
`rel_rot=(rec_rot % 768)*48 + rec_rot//768`, the admissible v3 candidate arrays
establish membership `[1,1,0]` for `(22971,19)`, `(22972,28)`, and `(2404,9)`.
V4.1 therefore qualifies only the hook's inertness, not those requested support
booleans.  Audits are
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_iter2_v41_probe_gate_20260712_092500/v41_inertness_audit.json`
and
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_iter2_v41_probe_cold5550_20260712_092500/cold_probe_qualification.json`.

Commit `536d6bd9` implements an explicit typed coarse-projector choice and
bypasses the texture selector at the manual supplied-PPref leaf; native
RECOVAR projection behavior and fine pass 2 remain unchanged.  The first four
apparent hybrid retries (`11056492`, `11056559`, `11056716`, and `11056983`)
were invalid diagnostics because `python scripts/run_multi_iter_parity.py`
loaded EM submodules from the stale editable checkout
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_em_parity_100k_20260712`.
They are texture-identical and must not be cited as tests of the patch.  The
runner now invokes `python -m scripts.run_multi_iter_parity`, isolates its
per-job Python bytecode cache, and asserts the concrete iteration-loop,
K-class, and significance module paths.

Import-bound hybrid job `11057315` is the accepted row-2813 gate.  Its coarse
dump is bit-exact to every array in the qualified manual baseline, including
`normalization_log_z=65.99702880122135`, inclusion of coarse candidate
`1051113`, and exclusion of texture-only `1031444`.  Fine candidate and
reconstruction supports exactly match RELION (`480/480` candidates and
`189/189` reconstruction hypotheses), the winner is exactly `[289965,48]`,
and Pmax is `0.1162683021` versus RELION `0.1162131086` (gap `5.52e-5`).
Posterior L1 is `0.0007255`; common-centered pre-prior score p95/max error is
`0.00359195/0.00455151`.  The job was intentionally cancelled after both dumps
completed.  Machine-readable qualification is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_global_pass1_bound_import_assert_20260712_115000/hybrid_row2813_qualification.json`.

Import-bound hybrid panel `11057457` completes the six-target fixed-state gate;
the job was intentionally cancelled after all 12 requested dumps completed.
All six winners now exactly match RELION.  Candidate symmetric differences
versus RELION change from texture to hybrid as follows: row 394 `0→32`, row
2813 `64→0`, row 3102 `32→0`, row 5504 `3392→512`, row 5550 `96→64`, and row
7710 `64→0`.  Reconstruction-support differences are `5,0,0,139,5,0` in the
same order.  Pmax absolute gaps range from `3.04e-6` to `8.78e-4`; common-score
p95 errors remain `0.00274–0.00675`.  Thus the hybrid closes the known non-tie
winner bug and improves five support sets, but manual coarse support is not
universally exact.  Machine-readable comparison is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_iter2_hybrid_panel6_20260712_121000/hybrid_panel6_relion_comparison.json`.

The exact-texture Euler hypothesis remains open, but the first NumPy float32
proxy is now qualified as inexact.  Valid typed-texture row-2813 job `11058151`
moves centered coarse scores by RMS `0.02421` and moves 66.2% of hypotheses
toward the manual score surface, yet leaves the wrong texture parent swap
unchanged.  RELION source audit subsequently found that the proxy precomputed
`float32(pi)/float32(180)`, whereas the active CUDA kernel evaluates
`angle*float32(pi)/float32(180)` and may contract later operations.  Correcting
only that operation order changes 30,582 of 36,864 matrices (entry max
`6.56e-7`).  Therefore this job rejects only the NumPy proxy, not exact RELION
device Euler arithmetic.  Full audit:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_relion_euler_exact_audit_20260712_132500/AUDIT.md`.
Corrected multiply-then-divide typed-texture job `11058615` is also support-
negative: its significant mask is bit-identical to the accepted texture
control and retains the same row-2813 swap.  CUDA 12.6 and 12.8 produce
identical normalized sm80 SASS for the RECOVAR texture kernel, ruling out the
compiler-version hypothesis.  Pinned RELION jobs `11058907/11058986` then
dumped the exact device matrices.  They differ from RECOVAR's default matrices
in 257,338 of 327,024 active entries (p95 `2.38e-7`, max `6.85e-7`), but exact-
table injection job `11059563` leaves the texture support mask bit-identical
and worsens the decisive preference from `-0.00972366` to `-0.01270676`.
Exact Euler arithmetic is therefore ruled out causally.  Qualification:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_row2813_exact_relion_euler_texture_retry_20260712_154000/qualification.json`.

Raw-map PPref provenance is ruled out causally by corrected job `11058908`.
Bypassing the RECOVAR Fourier round trip from the matching raw `run_it001`
half maps leaves the accepted-control support mask unchanged; centered
with-prior RMS is only `1.70553e-5`, and the decisive pair preference moves
`-2.28882e-5` away from RELION/manual.  Qualification is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_row2813_rawmrc_it001_ppref_texture_coarse_20260712_111000/qualification.json`.
The preceding job `11058764` used off-state `run_it002` maps and is explicitly
invalid, not evidence.

Typed-texture six-row discriminator `11058143` was cancelled after all six
coarse dumps completed.  Across the six full 1,069,056-hypothesis arrays,
manual and texture coarse supports differ from RELION by 19 and 18 parents,
respectively, and differ from each other by 25.  Neither projector is globally
exact.  Row 2813 has the decisive RELION/manual parent `(36245,8)` versus
texture `(35567,1)` swap; row 394 is exact under texture but gains one manual
parent; row 5504 improves from 16 manual to 14 texture differences but changes
the support count from RELION 6509 to 6501; row 5550 is identical between both
RECOVAR projectors and retains a two-parent swap versus RELION.  Machine-readable
evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_iter2_texture_coarse_panel6_20260712_104500/texture_coarse_discriminator.json`.

Import-bound full 10k hybrid trajectory job `11057493` completed all 16
numbered iterations plus the converged final all-data iteration.  It terminates
at iteration 16, matching RELION.  The final merged-map RELION FSC-AUC is
`0.980495` (shells 1--16 mean FSC `0.999772`); half-map FSC-AUC is
`0.953110/0.949810`.  Final all-data pose error is `0.150624` degrees mean and
translation error `0.0438001` pixels mean.  RECOVAR wall time is 2021.1 s
versus RELION 1330 s (`1.51963x`), with sparse pass 2 consuming 862.42 s.
Summary:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_10k_hybrid_full_20260712_121500/summary.md`.

The even-size Nyquist-coordinate mismatch is causal and fixed by `39dc2ce2`.
RELION keeps the two surviving coarse-disk endpoints as `(+N/2,0)` and
`(0,+N/2)`, while RECOVAR sampled them with negative Nyquist coordinates and
then relabelled the output.  Row-2813 positive-Nyquist job `11059889` flips
exactly the two discrepant hypotheses.  Six-row panel `11059982` plus stable
row-7710 retry `11060171` match every direct RELION `(rotation,translation)`
support set exactly: counts `71,15,67,6509,131,24`, each with symmetric
difference zero.  Corrected RELION texture projection is now the default for
both coarse and fine supplied-PPref scoring; environment value `0` retains the
manual/JAX diagnostic fallback.  Qualification:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_iter2_nyquist_texture_panel6_20260712_153500/nyquist_panel6_qualification.json`.

Corrected-texture full 10k job `11060805` completes all 16 numbered
iterations, converges on the same iteration as RELION, and executes the normal
final all-data branch.  More importantly, its complete `(current_size,
resolution shell)` trajectory exactly matches the authoritative RELION model
STAR files.  At iteration 12 both select shell 32 (13.10 A): RELION FSC at
shells 32/33 is `0.512015/0.497433`, corrected RECOVAR is
`0.511150/0.498921`, while the older hybrid's `0.512818/0.500107` incorrectly
selects shell 33.  The apparent corrected-versus-RELION iteration-12 mismatch
was a diagnostic error caused by treating the older RECOVAR replay log as the
oracle.  Resolution audits must derive RELION's shell from each model STAR
`_rlnSsnrMap` (threshold 1, equivalently FSC 0.5), not another RECOVAR log.
Evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_iter12_schedule_audit_20260712_170000/REPORT.md`.

The corrected run's final merged-map RELION FSC-AUC is `0.979511`, with
half-map FSC-AUCs `0.953044/0.948438`.  This is slightly below the older
hybrid's `0.980495` and `0.953110/0.949810`; the loss is high-frequency
(shells 97--126 mean delta `-0.00283`) while shells 1--16 are unchanged to
`2e-6`.  This does not invalidate the corrected projector: the corrected run
is closer in the full control schedule and the six direct coarse support sets
are exact.  It instead leaves a downstream score/reconstruction arithmetic
boundary open.  Wall time improves from 2021.1 s to 1917.75 s, but remains
`1.44192x` RELION.  Summary:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_10k_corrected_texture_full_retry_20260712_121000/summary.md`.

Global-corrected-texture/manual-fine trajectory `11062767` is rejected.  It
diverges from the authoritative RELION SSNR schedule at iterations 12 and 14,
has merged/half FSC-AUC `0.979389` and `0.951552/0.947366`, and worsens final
pose/translation error to `0.157554` degrees and `0.0462333` pixels.  Its
3339.83 s wall time is `2.51115x` RELION and 74.2% slower than corrected
texture throughout.  Manual fine projection is therefore neither a quality
nor a speed solution; the older hybrid's small final FSC advantage is a
compensating error.  Summary:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_10k_global_texture_manual_fine_20260712_124110/summary.md`.

The admissible direct fine-projection operand gate localizes the remaining
score residual to Euler construction.  RELION operand job `11066282` preserves
all discrete arrays/support and shows that using RELION projections with the
saved RECOVAR image plus RELION's 128-lane reduction matches raw RELION scores
at the independent-rerun floor (centered p95 `0.000244141`, max `0.000488281`).
Using RECOVAR projections gives p95 `0.00341797`, and projection substitution
alone produces the same p95.  Exact device Euler job `11068941` finds zero of
480 candidate matrix rows bit-exact (matrix max absolute delta `5.96e-8`).
Injecting those exact matrices into the RECOVAR texture projector in GPU job
`11071482` reduces correct-half projection p95 error from `4.825e-5` to
`4.915e-7` (about 98x) and reduces score error back to the RELION rerun floor.

The cause is not a different Euler convention or missing RELION binding.
RELION's live iteration-2 perturbation is `0.4052000939846039`, while the
sampling STAR serializes only `0.405200`.  Seed-exact job `11075288` recovers
the live value from `_rlnRandomSeed`, makes all 480/480 effective candidate
matrices byte-identical to the accelerated RELION dump, and retains projection
p95 `4.915e-7`.  Commit `3917aa67` makes this the typed `auto` behavior,
verifies consistency with the rounded STAR, provides explicit `seed_exact`
and `star` modes, and falls back to STAR precision only when seed provenance
is unavailable.  Evidence root:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_row2813_relion_fine_euler_dump_20260712_132000`.

Six-target fixed-state job `11076618` validates the combined positive-Nyquist
and seed-exact path.  All six fine winners are exact.  Five targets have exact
candidate and reconstruction support; row 5504 differs by one coarse parent at
a `2.68e-10` weight cutoff and three reconstruction samples with probability
gaps at most `1.82e-11`, so every discrete difference is a demonstrated
threshold tie.  Centered fine-score p95 is `2.69e-4`--`5.96e-4`, at the
independent accelerated-RELION rerun floor, and Pmax gaps are at most
`8.70e-4`.  Machine-readable evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_iter2_fullpert_nyquist_panel6_20260712_193000/fullpert_panel6_relion_comparison.json`.

Clean combined full-trajectory job `11079185` nevertheless fails the strict
10k map gate.  It reproduces the full authoritative `(current_size,
resolution shell, HEALPix order)` schedule, converges at iteration 16, and
runs final all-data with parent/fine orders 6/7.  Final merged RELION FSC-AUC
is `0.978500`, versus `0.979511` for rounded replay; half-map FSC-AUC is
`0.950958/0.947709`.  The loss is high-frequency: new-minus-rounded merged
mean FSC is `-1.6e-6` at shells 1--16, `-1.03e-3` at 33--64, and `-2.40e-3`
at 65--96, with worst shell delta `-0.00580`.  Runtime is 1984.0 s, or
`1.4917x` RELION.  Full precision is closer in pose/Pmax through iteration 2;
iteration 4 is the first net particle-state worsening, after tiny prior-state
differences are amplified by rare global score ties.  Iteration-1 diagnostic
`11084547` rules out an early reconstruction bug: merged map FSC-AUC is
`0.999996565`, minimum non-DC shell FSC `0.999993239`, and all 10k Pmax values
are exactly one.  Evidence roots:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_10k_fullpert_finalorder_20260712_193000` and
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_iter1_fullprecision_boundary_20260712_151500`.

Four-iteration numbered-map job `11084946` reproduces `11079185` poses and
translations byte-exactly and localizes the visible early map residual to the
first shell outside current-size signal support.  Full-box merged FSC-AUC
falls from `0.9999966` at
iteration 1 to `0.9994003`, `0.9959409`, and `0.9905003` at iterations 2--4;
the minimum shell is exactly the current-size boundary (47, 61, and 62).
Through RELION's authoritative signal shell, however, merged FSC-AUC remains
`0.999999759`, `0.999996900`, `0.999983765`, and `0.999938423`, with minimum
shell FSC at least `0.999751`.  Thus no material signal-band reconstruction
drift precedes the iteration-4 particle flips.

A direct current-size-edge audit rules out a crop-origin or Nyquist-plane
defect.  The edge planes contain only
`0.073%`--`7.23%` of residual energy at iterations 2--4, fitted shifts are
below `0.001` pixel.  Uninterrupted raw-BPref job `11087020` is a
near-authoritative numerical oracle: half-map FSC-AUC against the installed
iteration 2 is `0.99999991/0.999999998`, all angles and X translations are
exact, and only two Y translations differ by one grid step at ties.  After
frame scaling, RELION-versus-RECOVAR BPref numerator relative L2 is
`1.77%/0.94%` by half while weight is `0.338%/0.200%`, but causal
cross-substitution remains unqualified because the first reconstruction probe
used `skip_gridding=False` and later probes paired the wrong BPref/map files.
The inherited claim that a RELION solver raised shell-47 FSC to `0.999147`
has no reproducible script or result artifact and is withdrawn.

Identical-input reconstruction is instead closed: on both RELION and RECOVAR
BPref operands, RECOVAR's wrapper and RELION's real `skip_gridding=True`
binding agree at current-support FSC-AUC above `0.9999999999999` and shell-47
FSC above `0.999999999996` after the documented frame transform.  The source
audit nevertheless finds three independent correctness defects in RECOVAR's
current-size wrapper: packed-half tau shells round padded radius before
division and then round again (mislabeling `388168/1643720` supported
iteration-2 voxels), numerator decenter incorrectly excludes the exact-radius
sphere that RELION includes, and the MAP prior needs its own strict-radius
support.  These fixes require a fresh four-iteration trajectory gate; they are
not yet a production-quality closure.  Evidence roots:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_iter4_fullprecision_boundary_20260712_153000` and
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_relion_iter2_bpref_dump_fullscratch_20260712_171500`.

The tempting indexed-backprojection coordinate-order explanation is rejected.
RELION rotates integer Fourier coordinates before multiplying by padding while
RECOVAR's CUDA kernel multiplies first, but padding factor 2 is exact binary
scaling: five million float32 coordinate cases are bit-exact.  The initial
dot-then-padding A100 probe used the wrong default box size and is invalid as
quantitative evidence; corrected box-256 pre-scatter and scatter probes instead
show common-mode data/weight behavior and do not support the coordinate-order
hypothesis.  The diagnostic source change was reverted.  The remaining
credible BPref mechanism is global translated complex numerator accumulation
and atomic ordering, which is more cancellation-sensitive than the positive
CTF-squared weight.

Reciprocal iteration-3 map-splice array `11086240_[0-3]` proves that low
shells through authoritative signal shell 29 dominate the iteration-4
particle divergence.  The authoritative-map control A and REL-low/REC-high C
retain `9787/10000` and `9895/10000` exact joint winners relative to the free
RECOVAR-map B trajectory, while REC-low/REL-high D retains `9887/10000` of B's
winners.  Iteration-4 merged FSC-AUC through shell 61 groups independently as
A/C (`0.998992/0.998972`) versus B/D (`0.998272/0.998286`).  A matches RELION
joint winners within `1e-4` for `9996/10000`; B reproduces the free trajectory
for all six selected targets and `9997/10000` joint winners.  The Slurm array
is recorded as `FAILED/1` only because its post-run assertion requested local
fused-posterior dumps during a global sparse pass-2, where that hook is not
active; all science artifacts are complete.  This rules out iteration-4
sampling, scoring, and BPref as the first boundary and moves the trace to
iteration-3 low-shell PPref formation.  Evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it3_it4_ppref_shell_cross_20260712_160000/ANALYSIS_SUMMARY.txt`.

Clean 100k hybrid scale job `11058928` completes 16 numbered iterations plus
final in 2:35:23 without OOM, but its final pass incorrectly repeats numbered
parent/fine orders 6/7.  RELION's unnumbered `run_sampling.star` advances the
final parent to 7, so adaptive oversampling requires fine order 8.  Commit
`892c85e0` makes final sampling metadata authoritative while preserving the
state-order fallback.  Final-only job `11074230` and combined seed-exact retry
`11083758` validate parent/fine 7/8.  The final map passes strongly:
RECOVAR-vs-RELION FSC-AUC is `0.996184`, and RECOVAR-vs-GT is `0.497383`
versus RELION `0.490627`.  Pmax improves materially but remains non-parity:
RECOVAR mean `0.099121379` versus RELION `0.118882262`, correlation `0.9158`.
Seed-exact perturbation changes the mean only `1.1e-6`; 80,513 same-winner
particles retain a `-0.01956` mean gap.  This localizes the residual to
posterior denominator/support geometry, not sampling order, perturbation,
winner pose, OOM, or final-map quality.  Evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_100k_final_only_seedexact_hp8_combined_retry_20260712_150000/PMax_RESIDUAL_AUDIT.md`.

Next gates: validate the three source-derived current-size corrections in a
fresh four-iteration trajectory before another full 10k trajectory.  The
identical-input reconstruction gate is already closed; trajectory FSC/FSC-AUC
and particle-state changes decide whether these corrections are retained and
whether the smaller BPref numerator residual becomes the next boundary.
Attempt `11058781` was rejected before scoring because its CUDA output path
was shared.  The cold RELION dump `11084550` and stock continuation
`11085341` are also rejected oracles because their trajectories diverge from
the installed authoritative iteration 4.

The continuation-oracle path is closed rather than weakened.  Final
stored-accuracy job `11086683` exactly preserves the installed iteration-4
sampling order and full perturbation and retains the serialized iteration-3
accuracy (`2.006` degrees, `1.498312` Angstrom), yet still has half-map
FSC-AUC `0.9999892/0.9988276`, 20 angular differences above one degree, 581
translation differences, and 241 significant-sample-count mismatches.  Its
90 score dumps fail the unchanged gate and remain quarantined.  Therefore
RELION's serialized optimiser/data/model/sampling STARs omit process-history
state needed for an exact continuation oracle; no further ad hoc continuation
overrides are warranted.  Gate:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it4_authoritative_preserve_accuracy6_20260712_231500/authoritative_it4_parity_gate.json`.

The first strict 100k/256 completion attempt `11036541` reaches numbered
iteration 12, then fails in the local parent score-only big-JIT.  The failing
shape is 168 images by 198 rotations by 9 translations by 12,861 score pixels.
Its dense float32 residual is 14.34 GiB before compiler overhead and XLA asks
for 16.34 GiB.  The previous microbatch planner bounded only
`images * rotations`, so its automatic image-batch boost admitted this tensor
without charging translations or score pixels.  Commit `48e712f1` adds a
score-only runtime cap based on free allocator bytes, translations, score
pixels, float32 width, and measured 1.25x live-tensor headroom.  Explicit
overrides remain authoritative and M-step batching is unchanged.

Same-H100 diagnostic `11047386` jumps directly from RELION iteration 11 to the
exact failed iteration-12 state.  The cap selects 69 images per 198-rotation
parent bucket (`13,669` rows), processes all 49,913 half-set particles in
27.7 seconds, then completes the fine score-only pass in 26.0 seconds; the job
finishes successfully in 1:56 with no OOM.  Full clean 100k trajectory job
`11047558` was cancelled as scientifically stale after the missing run_it000
cold state was identified.  Corrected job `11049164` was relaunched from
clean commit `2e3cc620`, includes `--relion_init_dir`, and retains the validated
runtime score-tile cap.  Its artifact root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_100k_coldstart_scorecap_ready_20260712_051319`.

Job `11049164` was subsequently cancelled at 41:31 after the same-A100
iteration-3 projector A/B proved that its texture-on trajectory was
scientifically stale.  Do not restart 100k until the manual-projector full real
trajectory closes or identifies the next boundary.

K=4 also localizes the texture regression at its first global Class3D step.
Texture-on job `11049571` gives only 45.24% class agreement and per-class
FSC-AUC `0.3520-0.4439`.  Texture-off job `11049966` restores 99.82% class
agreement, pose p95 `1.42e-4` degrees, translation p95 `5.68e-6` pixels, and
exact Pmax; per-class map FSC-AUC improves to `0.9927-0.9967`, so its remaining
map/reconstruction tail is still open.  Class3D retains a fresh global first
search rather than replaying run_it000 input orientations (`2e0e25ab`); those
orientations are input metadata, not AutoRefine-style local-search centers.

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
  GPU class is valid. Up to four local GPUs may be used for short checks, but
  only after confirming each selected device is idle.
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

Run multiple independent, decision-bearing diagnostics in parallel when that
reduces parity-debug latency; large Slurm queues are acceptable. Keep only one
writer per source area, use matched GPU models for timing A/Bs, and cancel jobs
as soon as their premise becomes stale. A run without a predeclared decision
it can change should not be submitted. Negative and rejected results must be
recorded so future agents do not repeat or accidentally cite them.
