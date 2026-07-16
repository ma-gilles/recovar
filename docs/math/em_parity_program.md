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

## Reproducing the RELION dispatch-v2 oracle

Strict K-class replay needs RELION's authoritative mapping from sorted particle
position to MPI follower and original particle ID.  The diagnostic patch is
versioned at
`docs/patches/relion_dispatch_log_schema_v2_d476e6f.patch` (SHA-256
`6987c5ce397cbdd98835682cf1481a150c38c48cda621e006341d01a77e11c11`).
Apply it only to RELION base
`d476e6f6a4f1f37627c06ace5227fc374c0c2b05`:

```bash
test "$(git -C "$RELION_SRC" rev-parse HEAD)" = \
  d476e6f6a4f1f37627c06ace5227fc374c0c2b05
git -C "$RELION_SRC" apply \
  "$RECOVAR_SRC/docs/patches/relion_dispatch_log_schema_v2_d476e6f.patch"

source /etc/profile.d/modules.sh
module purge
module load relion/5.0.1/gcc-11.5.0-gpu
cmake --fresh -S "$RELION_SRC" -B "$RELION_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$(command -v gcc)" \
  -DCMAKE_CXX_COMPILER="$(command -v g++)" \
  -DMPI_C_COMPILER="$(command -v mpicc)" \
  -DMPI_CXX_COMPILER="$(command -v mpicxx)" \
  -DCUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.6 \
  -DCUDA_ARCH=80 -DGUI=OFF -DBUILD_TESTS=OFF
cmake --build "$RELION_BUILD" --target refine_mpi --parallel 8
strings "$RELION_BUILD/bin/relion_refine_mpi" \
  | grep -Fx RELION_DISPATCH_LOG_SCHEMA_V2
```

The qualified Della build used GCC 11, OpenMPI 4.1.6, CUDA 12.6, and RELION's
existing FFTW installation.  Set `RELION_DISPATCH_LOG` for a one-iteration
K-class smoke with the same fixture, MPI follower count, pool size, and seed as
the intended replay.  The leader writes the marker followed by five integer
columns:

```text
# RELION_DISPATCH_LOG_SCHEMA_V2
2 iteration follower_rank sorted_position original_part_id
```

Require the marker, then use the RECOVAR builder as the smoke validator; it
rejects non-v2 rows and non-bijective sorted positions or original IDs:

```bash
test "$(head -n 1 "$RELION_DISPATCH_LOG")" = \
  '# RELION_DISPATCH_LOG_SCHEMA_V2'
pixi run python -m scripts.build_relion_dispatch_schedule \
  --dispatch-log "$RELION_DISPATCH_LOG" \
  --output "$ORACLE_DIR/dispatch_schedule.npz" \
  --oracle-dir "$ORACLE_DIR" --n-particles "$N_PARTICLES" \
  --n-followers "$N_FOLLOWERS" --pool-size "$POOL_SIZE" \
  --random-seed "$RANDOM_SEED"
```

The hook is inert when `RELION_DISPATCH_LOG` is unset.  Keep the patch and
RELION source identity in run provenance; do not substitute a legacy
four-column range capture.

## Current State — 2026-07-14

Authoritative clean candidate checkout:
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_em_parity_20260711/recovar`

Current accepted code checkpoint: `7302610c23c3839d9acec1a7781d244b9d20e0d6`
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
- Current-head K=1 fixed-fixture boundary-replay job `11144457` matches all
  ten numbered RELION-seeded transitions, the exact current-size schedule, convergence at
  iteration 10, and the valid converged final-all-data path. Independent
  shellwise recomputation puts every numbered half/merged FSC-AUC above
  `0.9999985`; final merged RECOVAR-vs-RELION FSC-AUC is `0.998450626`, and
  RECOVAR final GT FSC-AUC exceeds RELION by `0.019912496`. This closes the
  fixed 3k/128 per-iteration boundary gate. It does **not** close the autonomous
  trajectory gate: `scripts/run_multi_iter_parity.py` injects RELION particle,
  noise, direction-prior, and optimiser-control state at each iteration.
- Current-head autonomous case-20 job `11197313` closes that small-trajectory
  qualification: exact current-size schedule and iteration-11 convergence,
  every numbered half/merged cross FSC-AUC at least `0.999986`, and final
  merged cross FSC-AUC `0.997634`.  RECOVAR final GT FSC-AUC is `+0.001144`
  above RELION.  The science command completed successfully; the recorded
  Slurm exit 2 is only a noncanonical-layout error in the post-run generic
  summarizer.
- K=4 100k/256 map quality is close/better by GT FSC-AUC, but particle-level
  state parity is incomplete: recorded class agreement `0.89025`, pose within
  5 degrees `0.71669`, translation within 1 px `0.77529`. Runtime is `2.181x`
  RELION; sparse K-class pass 2 dominates the completed iteration wall.
- Exact local x-half current/full BPref microbatching now survives the recorded
  3k/128 stress case without OOM. The conservative cap is validated for that
  fixture, not yet a universal optimal cap.
- The earlier 3k/128 final-state replay defect and iteration-1 reconstruction
  boundary are repaired. The current accepted evidence is FSC/FSC-AUC based;
  legacy map correlations are diagnostic only. The next K=1 work is the
  predefined boundary-replay robustness matrix while an autonomous cold-start
  trajectory is qualified separately, followed by 10k, real-particle, and
  100k/256 validation. The existing real EMPIAR-10076 failure is already localized to
  iteration-3 low-shell PPref formation before amplification at the
  iteration-8 global-to-local transition.

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

Current status on 2026-07-14: the seven-case immutable K=1 robustness matrix
at detached commit `f0ef1f0c6c231ff1f9183371d235e0b37a15b825` matches every
RELION current-size schedule and convergence iteration.  The previously
systematic final-map offset is now localized to an explicit output behavior,
not trajectory noise: the quality-oriented RECOVAR final path leaves RELION's
radial sinc-squared gridding correction disabled.  Applying the exact
`padding_factor=2` correction post hoc raises final RECOVAR-vs-RELION FSC-AUC
from `0.997395--0.998870` to `0.999605--0.9999997`.  Corrected RECOVAR-minus-
RELION GT FSC-AUC is between `-9.33e-6` and `+1.03e-5` in all seven cases.

The next K=1 identity target is severe case 26, whose corrected final cross
FSC-AUC remains `0.999605407`; the other six corrected cases are at least
`0.999961340`.  Its first autonomous departure is now localized before
reconstruction to the iteration-1 accelerated BPref accumulator.  Iteration-1
hard-WTA poses, translations, and Pmax are exact, but the matched RECOVAR
post-x0 versus RELION pre-lowres-join BPref numerator/weight relative-L2
residual is `3e-6--6e-6`.  Do not compare against RELION pre-reconstruct here;
that boundary is already after the 40-Angstrom half join.
Three same-H100 RELION captures vary by only `1.0e-8--1.3e-8`, so this residual
is reproducible code arithmetic rather than atomic-order noise.  The device
capture now identifies and fixes the first cause: RECOVAR added the integer
BPref origin before extracting float32 interpolation fractions, whereas
RELION extracts the fractions first.  The captured p8494 boundary becomes
bitwise exact for support, coordinates, all eight indices, and coefficients
after the arithmetic-order fix.  Rerun the trajectory gates; do not add a
score tie-break.

The explicit production grid-on case-25 diagnostic completed
as repaired jobs `11194076--11194077`: it matched RELION's current-size
schedule and iteration-8 convergence, logged radial correction enabled, and
reached final RECOVAR-vs-RELION FSC-AUC `0.999961353`.  Its RECOVAR and RELION
GT FSC-AUCs are `0.317329223` and `0.317318952` (delta `+1.0271e-5`), confirming
the post-hoc boundary through the real finalization path.  Keep
`RECOVAR_FINAL_ALL_DATA_GRID_CORRECT` unset/off outside named strict-parity
diagnostics because the current GUI-quality default remains grid-off.  In
parallel, retain the exact-Gaussian diagnostic conclusion and close the
case-20 accelerated preprocessing
boundary.  Typed `image_fourier_backend="relion_cuda"` at commit `bdda53c4`
now reproduces the source-faithful float32 normalization, zero-fill
translation, 128-by-128 CUDA background reduction, `sqrtf`/`cospif` mask, and
JAX/cuFFT window.  On both A100 and H100, captured particles 365 and 469 reach
bit-exact 65536-pixel masks and 1300/1300 Fourier windows within RELION's own
unordered atomic background envelope.  The existing `host_numpy` default is
unchanged.  Next validate fixed-state score arrays and then the full case-20
trajectory with explicit `relion_cuda`; the captured operand gate alone is not
an end-to-end quality claim.

The replacement exact fine-Gaussian reducer is integrated provisionally as the
default float32 Gaussian path.  A same-A100 paired iteration-2 run changed only
`RECOVAR_DISABLE_RELION_EXACT_FINE_GAUSSIAN=1`: mean Pmax error against RELION
fell from `2.78821e-5` to `1.99846e-5`, rows above `1e-4` fell from 371 to 168,
and wall time fell from `1065.49` to `538.72` seconds.  Iteration-1 state was
bitwise equal; iteration-2 merged cross-engine FSC-AUC was non-regressing
(`0.999999981423` to `0.999999981578`).  This real-data fixture has no GT, so
the result is not a GT-quality claim.  Commit `49e8f416` adds fail-closed
routing, removes unsafe fallback behavior, and restores the relevant tests.
The integration remains experimental until full K=1 and K=4 trajectory gates
pass; revert it if either quality gate regresses.

After those K=1 gates, advance to K=4 quality/state parity.  Performance is
not yet accepted: the broad matrix measures RECOVAR at `1.64--5.89x` RELION
wall time with approximately `97.6%` of an H100's memory occupied.

The current clean autonomous K=1 gate is job `11151255` at commit
`5a5769df37e49674c118697f60e73cbdd706b880`.  All ten numbered iterations
match RELION's current-size schedule `[56,56,66,68,80,80,80,80,80,80]`,
healpix schedule `[3,3,3,4,4,5,5,6,6,6]`, convergence at iteration 10, and
the single final Nyquist branch.  Every numbered FSC-AUC and GT gate passes.
The autonomous final merged map remains below the strict cross gate at
`0.986771`, although its GT FSC-AUC is better than RELION by `+0.020688`.
An exact-RELION-iteration-10 control on the same HEAD (job `11151769`) passes
the final cross gate at `0.998457`, localizing the failure to accumulated
trajectory state rather than final reconstruction mechanics.

The expected-accuracy mismatch is closed. RELION excludes the redundant
packed-FFTW `x=0, y<0` column from `Mresol`; the binding counted it. Adding the
same exclusion changes the exact first-100 result from `1.844` degrees /
`1.6915` Angstrom to RELION's exact `1.858` / `1.6915`. A Nyquist guard alone
does not change the mismatch. Same-process jobs `11152475`, `11152727`, and
`11152933` also reject particle order, CPU/GPU `PPref`, serialization,
anisotropic magnification, and scale-difference transforms. The audit is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/relion_ppref_cpu_ab_20260713_192524/ROOT_CAUSE_REPORT.md`.

Autonomous A100 job `11153043` verifies the fixed iteration-2 accuracy and the
same exact size/order/convergence schedule. Its final merged cross FSC-AUC is
`0.986985443`, still below `0.995`, while RECOVAR GT FSC-AUC is `0.671500068`
versus RELION `0.650834886`. Expected accuracy was therefore a real boundary
bug but is not the final-map cause.

The exact-final accumulator factorial (jobs `11151900`, `11151905`, `11152058`,
and `11152064_0`-`11152064_3`) independently shows that the residual is
posterior/adjoint accumulation, not tau2: replacing only the accumulator raises
FSC-AUC from `0.995640221` to `0.998443887`, while replacing only tau2 changes
it to `0.995652710`. The final-state factorial now localizes the material
autonomous failure. Holding the RELION iteration-10 state except for one
component gives cross FSC-AUC `0.995216198` for RECOVAR poses alone and
`0.996873606` for the RECOVAR map alone; using both gives `0.993407704`.
RECOVAR noise/tau2 and direction-prior substitutions remain near `0.9934`.
Substituting RECOVAR's image/group-scale corrections collapses the result to
`0.986822205`, nearly reproducing the autonomous failure.

RECOVAR's scale array is exactly one because the driver reads group IDs only
from input `particles.star`, where they are absent; RELION's supplied
`run_it000_data.star` contains 3,000 groups derived from 3,000 micrographs.
Restore those groups by image identity, preserve the full group count in both
halves, then enforce RELION's additional `data_vs_prior > 3` shell mask when
collecting scale `XA/AA`.

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

The current strict path now also matches RELION's separate joined-final noise
semantic: both particle halves use half-1 `sigma2_noise` in the
post-convergence K=1 all-data E-step. Exact dumped operands then match a representative RELION
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

Clean boundary-replay A100 job `11026304` passes the small-cell fixed-transition gate.
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
`1180.9/1112.4` in the boundary-replay trajectory. Thus iteration-2 scoring is not the
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

Four-iteration validation job `11089339` retains those corrections.  It runs
from clean commit `fa597a61`, reproduces exact sizes `[48,92,120,122]`, and
completes in 937.6 seconds.  Merged authoritative current-support FSC-AUC at
iterations 2--4 improves from `0.999698/0.998491/0.996590` to
`0.999968/0.999711/0.999096`; the corresponding minimum support-shell FSC
improves from `0.982863/0.910681/0.800538` to
`0.999275/0.993299/0.992760`.  Iteration-4 full-box FSC-AUC rises from
`0.990577` to `0.998519`, and the two half-map edge shells rise from about
`0.789/0.791` to `0.993/0.992`.  Iteration-3 Pmax MAE falls from `0.003763`
to `0.002787`, pose outliers fall from 40 to 29, and translation outliers from
99 to 59.  Iteration-4 Pmax MAE also improves (`0.009267` to `0.008367`),
although sparse angle ties leave mean pose error slightly worse; aggregate
map quality, not correlation, is the retention gate.  Report:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fa597a61_iter4_validation_20260712_171214/VALIDATION_REPORT.md`.

Full corrected 10k job `11090698` also retains commit `fa597a61`.  It
reproduces the exact 16-iteration size/HEALPix schedule, converges at iteration
16, and runs final all-data with parent/fine orders 6/7, seed-exact
perturbation, and grid correction off.  Final merged FSC-AUC versus RELION is
`0.978674`, up from `0.978500`; half-map FSC-AUCs are
`0.952292/0.946336` versus `0.950958/0.947709`.  Final pose mean/p95 improves
from `0.16561/0.64240` to `0.15986/0.63442` degrees and translation mean from
`0.04928` to `0.04656` pixels, while Pmax MAE is essentially flat/slightly
worse (`0.033922` to `0.033939`).  Runtime is 2045.7 seconds, 3.11% slower
than the prior run and `1.538x` RELION.  The fixture contains no GT volume, so
no GT FSC claim is made.  The earliest remaining material boundary is the
iteration-8 global-to-local HEALPix-4 transition: minimum support FSC drops
from about `0.9766` to `0.9144`, pose p95 leaves the numerical floor for about
`2.39` degrees, and Pmax MAE reaches `0.0837`.  Next work must compare local
parent/fine scoring and support at that fixed transition, not revisit the now
closed identical-input reconstruction wrapper.  Report:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fa597a61_full10k_trajectory_20260712_173845/FINAL_REPORT.md`.

The fixed serialized RELION iteration-7 to iteration-8 replay exposes where
the global-to-local transition amplifies a residual, but it is not an
identical-input oracle for RELION's uninterrupted in-memory projector state.
Aggregate replay
job `11092142` starts from RELION iteration-7 maps and particle state, uses
the exact iteration-8 current size 122, parent/fine orders 4/5, and seed-exact
perturbation `-0.360924143344`.  It leaves `8802/10000` particle rotations at
the numerical floor, but the remaining 1198 differ by at least one degree;
pose p95 is `2.2536` degrees, translation p95 is `0.5` pixel, and mean Pmax is
`0.636480` versus RELION `0.628746`.  Its nonzero Slurm exit is an
instrumentation-gate failure after all science completed: the dump filter used
absolute iteration 8 while the one-step replay labels its runtime loop as
iteration 1.  Setup-only jobs `11091925` and `11091997` are also rejected
(CUDA library-path contamination and an unsupported CLI argument,
respectively).  Neither produced a scientific result.

Corrected dump job `11092382` completes with exit 0 and all twelve requested
particle surfaces.  Exact winner-to-grid mapping shows two distinct,
non-numerical failure modes.  For particles 5727 and 932, RELION's winning
coarse rotation parent is absent from RECOVAR pass 2; for particle 9887 the
rotation parent is present but its required coarse translation pair is
masked.  Regenerating the complete order-4 neighborhood and order-5 children
recovers all three RELION rotations, ruling out parent/child enumeration.
Within the serialized-state replay, their first divergence is pass-1 coarse
scoring, normalization, or the 0.999 significance selection.  Conversely,
RELION's uninterrupted winners for particles
3758, 4321, 5375, and 9826 are present but lose on RECOVAR's total score by
`30.306/0.893/0.528/1.367`; saved-operand recomputation agrees with the live
scores within about `0.002`, so reduction rounding cannot explain those
margins.  Particle 3758 is a material raw data-term inversion (`30.720`),
whereas the other three are flipped by RECOVAR's orientation/translation
priors after the RELION winner has the better RECOVAR raw term.  A cold
instrumented RELION iteration-8 run is therefore required to distinguish
coarse support, raw projection/image operands, and prior arithmetic directly;
continuation dumps remain inadmissible.  Evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fa597a61_fixed_rel7_to8_localdump_20260712_183500/retry3/SUPPORT_AUDIT.md`
and
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fa597a61_fixed_rel7_to8_localdump_20260712_183500/retry3/PASS2_SCORE_AUDIT.md`.

Matched RECOVAR coarse-table job `11092553` completes in 5:40 and proves
that the parent-support result is not a dump-mapping artifact.  For all twelve
particles, the coarse `reconstruction_sample_mask` pairs exactly equal the
fine table's finite candidates after collapsing each 8-by-4 child block back
to its parent.  The required RELION winner pairs for particles 5727, 932, and
9887 rank only 17, 13, and 3 in RECOVAR versus retained support sizes 6, 7,
and 1; their score gaps from the RECOVAR winner are `-12.8455`, `-3.98770`,
and `-14.23038`, and their posterior-to-cutoff ratios are
`2.64e-6`, `1.85e-2`, and `6.60e-7`.  These are material coarse score or
probability-table mismatches, not 0.999-threshold ties.  Particle 3758 gives
the complementary boundary: its RELION winner parent is RECOVAR's coarse
rank-1 pose with posterior `0.992`, so its `30.720` raw inversion arises only
after fine-child projection/scoring.  Evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fa597a61_fixed_rel7_to8_localdump_20260712_183500/retry4/coarse_dump/COARSE_DUMP_AUDIT.md`.

Cold operand-enabled RELION job `11092529` is rejected by the mandatory
observational-inertness gate.  It reproduces the exact iteration-1 through
iteration-8 schedule and perturbations, produces the expected 180 files, and
maps every requested particle ID correctly, but its iteration-8 half-map
FSC-AUC against the installed uninterrupted oracle is only
`0.985700/0.993777` with minimum non-DC FSC `0.963725/0.981571`.  It also
changes 684 particle angles above `1e-4` degrees, 818 translations, 1968
significant-sample counts, and Pmax by up to `0.68007`.  The Slurm job's
nonzero exit is the intended fail-closed post-science result (after correcting
an analysis-only STAR parser type assumption).  All 180 score/projection
files are quarantined and must not be used.  A redesigned minimal hook must
pass an enabled-versus-disabled cold control before direct RELION score
comparison resumes.  Gate:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_relion_it8_cold_score_oracle_20260712_184953/cold_it8_inertness_gate.json`.

The attempted score-only V3 pair `11092663/11092664` is also quarantined
before score use because its advertised build path had been mutated in place.
The genuinely qualified V3 binary from job `11055888` had SHA-256
`68982a12...` and produced 13 files per target; the current path instead had
SHA-256 `f77efbf3...`, contained the projection-operand hook, and produced 15
files per target.  The mismatch is scientifically visible: the same-binary
disabled run remains at HEALPix order 3 for iteration 8 while the enabled and
installed runs advance to order 4; enabled-versus-disabled half-map FSC-AUC is
only `0.706429/0.702250`, all 10,000 poses differ, and 8030 significant-count
rows differ.  No accepted marker exists.  Future instrumented builds must use
an immutable new build root and verify both source-diff and binary hashes
before submission.  Quarantine record:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_relion_it8_v3_cold_pair_20260712_191034/paired_cold_it8_qualification.json`.

Rebuilding the true V3 patch in a new immutable source root verifies exact
commit `d476e6f`, patch and full-diff SHA-256 `58579a10...`, the original
toolchain, and absence of all projection-operand code.  Build job `11093013`
correctly notes that byte identity cannot survive the path change: RELION
embeds absolute source/build paths in its executable.  A same-binary cold pair
is the stronger scientific test.  Jobs `11093087/11093088` produce the
expected 156/0 manifests and exact target mapping, but V3 is not inert through
iteration 8: the disabled run remains at HEALPix order 3 while enabled and
installed advance to order 4.  Enabled-versus-disabled half-map FSC-AUC is
`0.708193/0.703562`, with all 10,000 poses, 6292 shifts, and 7983 significant
counts differing.  Enabled-versus-installed FSC-AUC is only
`0.984972/0.992932`.  V3 was qualified for iteration 2 only and must not be
extrapolated to the later adaptive transition; its iteration-8 fine tables
are quarantined.  Gate:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_relion_it8_v3_immutable_pair_20260712_193549/paired_cold_it8_qualification.json`.

Fixed-transition projector A/B jobs `11093381/11093382` initially reveal a
routing defect rather than a scientific hybrid result.  The supplied-PPref
local bucket, packed-noise, and projection-cache helpers discarded their
explicit `projection_relion_texture_interp` argument and fell back to the
process-wide environment.  Consequently A was texture/texture and exactly
reproduced baseline, while B was manual/manual.  Manual/manual does not change
particles 3758, 5727, 932, or 9887 and slightly worsens mean angle,
translation, and Pmax parity; its FSC-AUC gain is only `1.47e-5`.  Their Slurm
`FAILED/1` states are a bad post-science regular-expression gate; both science
outputs are complete.  Report:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fa597a61_rel7to8_projector_ab_20260712_190500/A_B_REPORT.md`.

The production routing fix forwards the explicit selector through all three
supplied-projector paths, pins local parent pass 1 to manual interpolation,
and lets fine pass 2 follow the switchable texture default.  Validation job
`11093570` completes with exit 0 and logs the intended route for both halves.
It is exactly equal to the fixed baseline in every saved pose, translation,
Pmax, and per-half fine significant count; the four target particles remain
unchanged.  FSC-AUC changes by only `-5.22e-8` and maximum shell FSC by
`2.17e-6`.  The fix is retained as a correctness/configuration repair, but
manual-versus-texture selection is not the iteration-8 cause.  Report:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_local_hybrid_routing_validation_20260712_201000/VALIDATION_REPORT.md`.

An unmodified stock `relion_project` audit resolves the apparent particle-3758
fine-score contradiction and changes the causal boundary.  Projecting the two
candidate poses from the serialized iteration-7 half-map matches RECOVAR's
saved projection operands at relative L2 `0.000891/0.000832`.  Combining those
stock projections with RECOVAR's saved image/CTF/noise operands reproduces the
same material raw/total preference for RECOVAR's winner:
`30.7315/30.3182` versus RECOVAR `30.7196/30.3062`.  The opposite half-map
still prefers the RECOVAR winner by `18.98` raw-score units, stock CTF agrees
after the paired sign convention, and stock CTF-subtracted residuals also
favor the RECOVAR winner.  RECOVAR is therefore self-consistent with stock
RELION for the serialized map.  The uninterrupted RELION winner depends on
live reconstruction/projector state not recoverable from its written
map/model/data/sampling files, exactly as the failed continuation gates imply.
Iteration 8 amplifies an earlier map/projector-state difference; it does not
establish a local E-step bug on identical inputs.  Return the trajectory trace
to pre-iteration-8 reconstruction/projector formation.  Evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_it8_p3758_stock_project_20260712_203000/STOCK_PROJECTOR_AUDIT.md`.

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
retain `9787/10000` and `9895/10000` exact joint winners relative to the
boundary-replay RECOVAR-map B trajectory, while REC-low/REL-high D retains
`9887/10000` of B's
winners.  Iteration-4 merged FSC-AUC through shell 61 groups independently as
A/C (`0.998992/0.998972`) versus B/D (`0.998272/0.998286`).  A matches RELION
joint winners within `1e-4` for `9996/10000`; B reproduces the boundary-replay trajectory
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

## 2026-07-12 Iteration-2 BPref Arithmetic Trace

The iteration-2 raw-BPref oracle is now explicitly qualified rather than
treated as exact. Job `11087020` is close enough for percent-level accumulator
diagnostics: iteration-2 half-map FSC-AUC against the installed run is
`0.999999909/0.999999999`, all serialized angles and X translations agree,
and only two Y translations move by one grid step. It is not bitwise inert:
15 significant counts differ by one and maximum Pmax delta is `0.002181`.
All 99 recorded artifact hashes verify. Its arrays may therefore adjudicate
the `0.94%--1.72%` numerator and `0.20%--0.32%` weight residuals, but not
cross-substitution causality, exact ties, or smaller effects. The correct pair
is RELION iteration 2 versus RECOVAR zero-based `it001`; using `it002` pairs
different current sizes. Audit:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_job11087020_readonly_audit_20260712_213000/AUDIT_REPORT.md`.

Source inspection identified an arithmetic difference before BPref scatter:
the active RELION GPU build reduces each orientation/pixel across translations
in `XFLOAT=float`, while RECOVAR normally forms the sufficient statistic with
a higher-precision GEMM before truncating to the float accumulator. Commit
`e6098b58` adds an off-by-default diagnostic switch,
`RECOVAR_RELION_X_HALF_SEQUENTIAL_TRANSLATION_REDUCTION`, that carries the
numerator in complex64 and the positive denominator in float32 in increasing
translation-index order. Focused CPU tests cover exact sequential arithmetic,
dtype, gate-off equivalence, and x-half-only dispatch.

Job `11094394` rejects that difference as the material boundary. It completes
the exact two-iteration `[48,92]` schedule from immutable commit `e6098b58` in
`629.3` science seconds (`00:14:03` including setup). Relative to the default,
the supported-radius raw-BPref numerator residual falls only
`0.0172285 -> 0.0171696` and `0.00944339 -> 0.00943748`; weight falls only
`0.00318974 -> 0.00318260` and `0.00200353 -> 0.00200285`. Iteration-2
half-map FSC-AUC through shell 46 changes only
`0.9999186 -> 0.9999188` and stays `0.9999695` in half 2. Keep the switch off
and do not extend this hypothesis to a full trajectory. RECOVAR repeatability
also rules out nondeterministic inter-particle atomics: two independent default
runs differ by only `2.25e-6--2.65e-6` numerator relative L2 and
`4.4e-7--5.0e-7` weight relative L2 at the same boundary. The remaining
systematic trace moves earlier to RELION's float32 fine-weight exponentiation,
ascending sorted cumulative normalization/significance threshold, and
posterior/support weights. Report:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_e6098b58_f32seq_it2_bpref_20260712_214000/DIAGNOSTIC_REPORT.md`.

RELION's fine-weight normalization is also source-distinct: the accelerated
path keeps raw weights in float32, applies the `+50` max shift and `expf`,
sorts ascending, obtains the denominator and lower-tail significance cutoff
from a float32 cumulative scan, retains cutoff ties, and divides retained raw
weights by the full pre-pruning float32 denominator. Commit `4d9daafc` adds an
off-by-default K=1 x-half diagnostic for exactly that arithmetic. Jobs
`11095329` and `11095487` are rejected setup/diagnostic-routing attempts: the
first allowed the soft diagnostic to override first-iteration CC hard weights;
the second guarded only the chunked dispatcher, leaving the non-chunked call
unprotected. Neither provides an iteration-2 result. Commits `adfd76bd` and
`e7a6e124` add behavioral winner-take-all guards at both dispatcher levels.

Corrected job `11095669` completes the two-iteration `[48,92]` gate from
immutable commit `e7a6e124` in `600.7` science seconds (`00:13:34` including
setup). Its iteration-1 particle state is byte-identical to the accepted
control and its accumulators differ only `3.5e-8--5.4e-8` relative L2. At
iteration 2, however, adding the source-matched float32 posterior changes the
accepted sequential-only accumulators by just `3.6e-6/9.8e-6` numerator and
`6.3e-7/1.3e-6` weight relative L2. Against the near-authoritative RELION raw
BPref, supported-radius residuals remain `0.0171696/0.0094375` numerator and
`0.0031826/0.0020029` weight. The fine exponentiation, sorted denominator,
cutoff/support, and normalization-order hypothesis is therefore rejected as
the percent/sub-percent boundary. Both diagnostic switches remain off by
default. The next trace is per-particle positive operands (CTF squared,
inverse noise, scale) and their rotation scatter; complex image/translation
phase operands follow only after the positive weight path is classified.

## 2026-07-13 Matched BPref Operand And First-Boundary Trace

The per-particle operand boundary is closed rather than inferred. RECOVAR job
`11095951` and RELION diagnostic build/run jobs `11096416/11096516` capture
original particle 3996 at iteration 2/current size 92. RELION's exact
accelerated store-weighted-sums buffers map to RECOVAR with FFTW row
`ky mod 92`; all 116 translations have the same order and differ by at most
`3.18e-7` pixels. The positive `CTF^2*Minvsigma2` reconstruction operand has
relative L2 `1.008e-6` after the exact `256^-4` scale. Every translated
unmasked complex `image*CTF*Minvsigma2` row has fitted relative L2 about
`8.17e-7` after `-256^-2`. This closes particle identity, CTF, inverse noise,
scale, unmasked preprocessing, old-offset application, fine translations, and
phase convention. Report:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_relion_it2_p3996_store_operands_20260712_223000/MATCHED_OPERAND_REPORT.md`.

The same capture closes candidate geometry but exposes inherited posterior
drift. All 24 accelerated Euler matrices agree exactly after the expected
transpose; the 896 indexed candidates and 161 reconstruction samples are
identical with no missing nonzero candidate. Nevertheless the default
posterior differs by L1 `0.0130861` with maximum candidate gap `8.842e-4`,
and live fine projections differ by roughly `2.34e-4--3.17e-4` relative L2.
The affine-aligned raw data-score RMS is `0.0307184`. This is score-surface
drift from the incoming maps, not support or operand arithmetic.

Fixed-reference job `11096872` proves that causal direction. It replays the
exact installed RELION iteration-1 half maps only for RECOVAR scoring
iteration 2. Particle 3996's projection median relative L2 falls to
`2.31e-7`, raw-score RMS to `2.64e-4`, and posterior L1 to `0.0002309`, while
candidate/support/argmax remain exact. Aggregate supported-radius BPref also
improves materially: numerator residual becomes `0.398%/0.439%` and weight
`0.107%/0.111%`, versus approximately `1.72%/0.94%` and `0.318%/0.200%` on
the default trajectory. The remaining fixed-reference residual is
`98%--99.9%` outer-shell power. Thus iteration 2 mostly amplifies the prior
map boundary; its posterior path is not the first cause. Report:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_e7a6e124_relref_it2_control_20260712_234500/FIXED_REFERENCE_AUDIT.md`.

Iteration-1 raw-BPref job `11096911` locates that first cause before any soft
posterior exists. Against RECOVAR zero-based `it000`, hard-assignment
supported-radius numerator residual is `1.741%/1.590%` and weight is
`0.507%/0.601%`; weight DC is exact in both halves. The dump is qualified by
RELION-dump-versus-stock half-map FSC-AUC
`0.999999998872/0.999999998890`. RECOVAR-versus-stock FSC-AUC is already high
(`0.999992264/0.999997461`) but not exact enough to prevent later score
amplification. Shell 24 contributes only 25--32% of raw residual power, so an
outer-boundary-only patch cannot close the full first-iteration discrepancy.
Audit:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_relion_iter1_bpref_dump_20260713_000500/audit/IT1_BPREF_AUDIT.md`.

Finally, matched hard-winner jobs `11097303/11097304` reject a material
one-contribution geometry defect. Particle 3996 has the same unique winner
(orientation 7, translation 88), bitwise phase, and identical translation.
The positive operand has relative L2 `5.88e-7`; there are no radius-support,
base-neighbor, or x-fold decisions that differ. RELION's Euler and RECOVAR's
transpose differ by one float32 ULP in one of nine winner entries, and the
remaining float expression ordering moves padded coordinates by at most
`3.82e-6`, but a full simulated one-particle weight scatter differs by only
`3.21e-6` (`3.91e-6` on shell 24). These effects are real but two to three
orders too small to explain the aggregate gap. Report:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_it1_p3996_matched_winner_audit_20260713_004500/MATCHED_WINNER_AUDIT.md`.

The active discriminator is therefore the outer float32 atomic enumeration.
RELION launches one 128-thread block per orientation and iterates pixels in
serial passes; RECOVAR launches multiple 256-thread pixel blocks per
orientation from a compact centered-row list. Run repeatability only proves
each topology is stable; it does not prove cross-topology equivalence. An
off-by-default launch/order diagnostic must reproduce RELION's orientation and
native FFTW pixel enumeration before any production retention. FSC/FSC-AUC
remain the map acceptance gates; accumulator relative norms only localize this
boundary.

Commit `94dc6224` adds the first narrow launch diagnostic,
`RECOVAR_RELION_X_HALF_BP_BLOCK_TOPOLOGY`. It expands the compact current-size
circle into native FFTW square order and uses one 128-thread block per
orientation with serial pixel passes. The switch is off by default and is
captured at JAX trace time, so the scientific A/B used separate fresh
processes. Same-build A100 jobs `11111001/11111002` completed in `00:05:59`
from the immutable commit and exact iteration-1 state. The off control
reproduces the accepted boundary: supported numerator residual
`1.74143471%/1.59014638%`, weight residual `0.507356356%/0.601081232%`, and
map FSC-AUC `0.9999922641/0.999997461041` against stock RELION. Enabling the
topology changes on-versus-off accumulators by only
`1.184e-7/1.183e-7` numerator and `7.77e-8/7.80e-8` weight relative L2;
on-versus-off map FSC-AUC is `0.9999999940/0.9999999938`. The RELION residual
and map FSC-AUC are unchanged at reported precision. Report:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_94dc6224_it1_bptopology_ab_20260713_092000/audit/AB_REPORT.md`.

This null result rejects only native pixel order, 128-thread lane grouping,
and serial pixel passes as the percent/sub-percent cause. It does not emulate
or reject RELION's per-particle kernel-launch boundaries, in-kernel translation
reduction, or interleaved real/imaginary/weight atomics. RECOVAR currently
flattens active particle/orientation rows, reduces translations before the
scatter, and launches numerator and weight separately. The next diagnostic
must preserve particle ownership and couple these operations before another
trajectory run; the block-topology switch remains diagnostic-only and off.

Commit `813f77f3` adds the next deliberately narrower diagnostic,
`RECOVAR_RELION_X_HALF_BP_PER_PARTICLE_LAUNCH`. In the single-class,
non-rotation-chunked winner-take-all x-half path it asserts strictly increasing
particle ownership, asserts exactly one positive rotation row per particle,
preserves the unpadded `actual_counts`, and gives each particle's eight ordered
fine rotations their own backprojection launch. It remains off by default and
still performs translation reduction before launching separate numerator and
weight kernels.

Same-build A100 jobs `11112039/11112041` completed successfully on
`della-l07g3` in `00:05:53/00:06:06`. The diagnostic fired for all 10,000
particles in 46 batches, always with rotation-count min/median/max `8/8/8`.
The result is another decisive null. Control supported-radius RELION residuals
are `1.74143470%/1.59014641%` for numerator and
`0.507356372%/0.601081213%` for weight; per-particle launches leave them
`1.74143472%/1.59014640%` and `0.507356362%/0.601081202%`. Particle-versus-
control relative L2 is only `9.14e-8/9.11e-8` for numerator and
`5.44e-8/5.48e-8` for weight. Particle-versus-control half-map FSC-AUC is
`0.999999993928/0.999999994128`, while each mode's map-versus-stock-RELION
FSC-AUC remains `0.999992264/0.999997461`.

Report:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_813f77f3_it1_particlelaunch_ab_20260713_095000/audit/AB_REPORT.md`.
This rejects the particle launch boundary under pre-reduced, separate
numerator/weight kernels. It does not reject RELION's fused neighbor atomic
sequence (real, imaginary, then weight) or its in-kernel translation
reduction. The next same-build iteration-1 discriminator is therefore an
off-by-default fused two-accumulator x-half FFI, still using the qualified
pre-reduced operands. Do not run a longer trajectory unless that boundary
materially improves raw BPref residuals and FSC/FSC-AUC.

Commit `2173ab4b` implements that fused discriminator as the strict,
off-by-default `RECOVAR_RELION_X_HALF_BP_FUSED_ATOMICS` path. It accepts the
pre-reduced complex64 numerator and float32 weight rows, preserves the native
current-size square and particle-owned eight-orientation grid, and issues each
neighbor's atomics in RELION order: real, imaginary, then weight. Mixed-type
two-output FFI aliases are validated on both Python and CUDA sides. Unsupported
later-iteration, dense, and joint K-class routes fail closed rather than
silently ignoring the flag. Translation reduction remains outside CUDA.

The immutable A100 GPU smoke job `11114721` passes the fused-versus-separate
topology comparison from nonzero initial accumulators; setup-only job
`11114516` had already built successfully but exited `141` because a
`pipefail` symbol-audit pipeline observed `nm`'s expected SIGPIPE. Same-build
science jobs `11114805/11114806` then complete successfully in
`00:03:56/00:04:05`. Both use the 128-thread native topology and per-particle
launches; only job `11114806` enables fused atomics.

The result is null at the established repeatability floor. Control-versus-
RELION supported-radius residuals are `1.74143472%/1.59014640%` for numerator
and `0.507356363%/0.601081201%` for weight. Fused atomics leave them
`1.74143398%/1.59014674%` and `0.507356467%/0.601081250%`. Fused-versus-
control relative L2 is only `2.49e-6/2.45e-6` numerator and
`7.47e-7/7.45e-7` weight; half-map FSC-AUC is
`0.999999993721/0.999999993560`. Map-versus-stock-RELION FSC-AUC remains
`0.999992264/0.999997461`.

Report:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_2173ab4b_it1_fusedbp_ab_20260713_104500/audit/AB_REPORT.md`.
Do not extend the fused-pre-reduced branch to a trajectory. Pixel enumeration,
thread grouping, per-particle launch boundaries, and fused neighbor atomics are
now all rejected as material causes for qualified pre-reduced operands. The
remaining source-level first boundary must be localized between per-particle
operands and the aggregate scatter. FSC/FSC-AUC remain the acceptance gates.

The completed 12-particle operand panel now rejects the CUDA translation-loop
port as the next positive-weight fix. Jobs `11116157`, `11116203`, `11116239`,
`11117008`, and `11117519` cover both halves, defocus extremes, angular order,
and residual-heavy particles. Every particle has exactly one active hard
winner and RECOVAR agrees on all 12 winners. Fine Euler matrices agree within
`5.96e-8`; the positive `CTF^2*Minvsigma2` operand has relative-L2
min/median/max `1.73e-7/3.58e-7/5.73e-7`. The reconstructed complex winner
rows differ by only `1.94e-5/1.08e-4/3.00e-4` min/median/max, with the remaining
complex comparison qualified by the host phase replay. Because iteration-1
weight has neither a translation sum nor phase, a raw translation-reduction
port cannot explain its `0.507%/0.601%` aggregate residual. Report:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_it1_operand_panel_20260713_110500/OPERAND_PANEL_REPORT.md`.

An independent aggregate audit also rejects comparison mapping, half
ownership, padding, low-resolution joining, dump timing, and a global weight
normalization. Correct mapping is uniquely best at `0.00507/0.00601` relative
L2; swapping halves is approximately `0.230/0.227`; RELION's post-join dump is
byte-identical to its pre-reconstruct input outside the intentional join; and
fitting a global scale barely changes the residual. Total weight and shell
sums are conserved to roughly `1e-6--1e-4` while shell-local voxelwise
redistribution reaches percent scale. Report:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_bpref_weight_residual_audit_20260713_110518/AUDIT_REPORT.md`.

Host replay of all 10,764 active source pixels in the 12-particle panel finds
zero cutoff, x-half fold, base-voxel, or neighbor-index differences. Sorted
float64 scatter relative L2 is `3.38957e-6/2.99566e-6`, about three orders below
the whole-dataset residual; per-particle coefficient relative L2 is
`2.26e-6--3.46e-6`. This closes ordinary-coordinate pre-atomic scatter but not
rare boundary crossings: the nearest non-DC integer margin in this random
panel is only `8.64e-5`, too large to represent the extrema among 10,000
particles. Audit and design:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_scatter_signature_offline_20260713_111555/SCATTER_SIGNATURE_DESIGN.md`.

The active discriminator is therefore a complete pre-atomic signature for a
reproducible, boundary-enriched 128-particle panel: 32 particles per half with
the smallest integer-plane margin, 16 per half with the smallest cutoff
margin, and 16 per half with greatest mass into the top aggregate residual
voxels, deduplicated and stratified by CTF and orientation. Compare radius and
fold decisions, base/neighbor indices, coefficients, `Fweight`, and sorted
deterministic accumulators. If that panel remains at the ordinary `~3e-6`
floor, pre-atomic geometry is rejected and the trace moves to global
particle/candidate ownership or production accumulator correspondence; do not
implement the full translation CUDA port without contrary evidence.

## 2026-07-13 Boundary-Enriched Production Scatter Trace

The 128-particle actual-kernel gate rejects the preceding ordinary-coordinate
null and proves a sparse support-boundary divergence before atomics. Jobs
`11120008`, `11120414`, `11120476`, `11121797`, `11122426`, `11122768`,
`11123306`, and `11123449` capture 64 particles per half directly from the
RELION and RECOVAR production scatter paths. The panel covers all eight fine
children and all four defocus quartiles. RECOVAR signature replay agrees with
its actual zeroed GPU accumulator to `4.157e-6/4.304e-6` relative L2 by half,
so the result is not a host replay or atomic-order inference.

Among 114,816 common positive source records there are 22 radius-cutoff
differences, zero x-half fold differences, and five base/neighbor differences.
All 22 cutoff flips are axial outer-rim sources `(24,0)` or `(0,24)`. They
produce deterministic half-accumulator relative L2 errors of
`0.012369786/0.016898168`; excluding those two source coordinates reduces the
errors to `2.342e-6/2.599e-6`. The 5,376 RELION-only native-square records are
all radius rejects and contribute nothing. Fweight agrees to `3.534e-7` after
the known scale, closing the positive operand at this boundary.

RELION rotates the unpadded integer coordinate and applies `padding_factor`
after the dot product. RECOVAR previously distributed the scale into both dot
products. Isolated A/B jobs `11124285/11124286` prove that matching RELION's
operation order eliminates all five base/neighbor differences and reduces
jointly accepted coefficient relative L2 from `1.132e-2` to `1.072e-6`.
Commit `6f467ea0` applies that correction to the strict single, batched, and
fused x-half backprojectors. CPU policy/geometry tests pass `47/47`; A100 job
`11125183` passes the fused/separate and rectangular/odd-cubic CUDA regressions
`2/2`.

That coordinate-order fix alone does **not** remove any of the 22 axial-rim
cutoff flips or the percent-scale half-accumulator residual. Each flip compares
`radius2=2304.0` with `2304.000244140625`, with 15 RELION-reject/RECOVAR-pass
and seven RELION-pass/RECOVAR-reject decisions. RELION was compiled with CUDA
12.6 and the RECOVAR diagnostic with CUDA 12.8. Same-toolkit A100 job
`11125480` rebuilds the corrected RECOVAR diagnostic with CUDA 12.6.85 and
native `sm_80`; it leaves exactly 22 cutoff flips and unchanged half errors
`0.012369786/0.016898166`, while all fold/base/neighbor differences remain
zero and coefficient relative L2 remains `1.072e-6`. Toolkit version alone is
therefore rejected.

The next two discriminators localize the remaining pre-atomic discrepancy.
RELION evaluates physical `(x,y,z)` radius as `mul(y,y)`, `fma(x,x,y2)`, then
`fma(z,z,xy2)`. Replaying that explicit sequence reduces the 22 cutoff flips
to five and the deterministic half errors to `0.000884305/0.007236143` (job
`11125903`). Each survivor is explained by a one-ULP difference in the Euler
coefficient used by its axial source. Combining the explicit radius sequence
with the exact captured RELION Euler bits removes every cutoff, fold, base, and
neighbor mismatch and reduces deterministic half-accumulator relative L2 to
`2.919113e-7/2.700236e-7` (job `11126507`). This closes the captured panel's
pre-atomic geometry boundary, but does not yet provide a valid production
source for the RELION matrix bits.

Commit `341e778b` pins the physical-axis radius accumulation order in the
strict single, batched, and fused x-half CUDA kernels. CPU policy/geometry
tests pass `48/48`; A100 job `11126678` passes the fused/separate and
rectangular/odd-cubic CUDA regressions `2/2`. A naive matrix-generator fix is
explicitly rejected: A100/CUDA-12.6 job `11126900` recovers the source angle
rows and regenerates current RECOVAR matrices bit-exactly, but literal CUDA
`sincosf` disagrees with the captured RELION matrices in 855 elements across
all 128 winners, and JAX float32 trigonometry disagrees in 1,046 elements.
Current evidence indicates that the captured weighted-average/backprojection
matrices follow RELION's CPU `RFLOAT` perturbed-angle, double-trigonometry, and
matrix-inversion path rather than the GPU projector-plan's unperturbed-angle
plus `doR=true` path. Trace and reproduce that exact call chain before changing
the production Euler generator; do not substitute captured fixture matrices or
weaken the rotated-radius predicate.

That trace identifies the last five matrix-boundary flips as a scaled-RNG
operation-order bug, not a trigonometric-library difference. RELION calls the
two-argument float function `rnd_unif(0.5*pf, pf)` directly. RECOVAR previously
called `rnd_unif(0,1)` and scaled its already rounded result in Python double;
the algebraically equivalent expression is not float-bit equivalent. For
random seed `20260712`, RELION iteration 1 uses
`random_perturbation=-0.04961434006690979`, whereas the old replay used
`-0.049614354968070984`. Commit `e7a1af47` adds a direct scaled-range binding
and float-faithful fallback. The rebuilt binding and 42 focused tests pass.

With the corrected perturbation, the existing RELION C++ oversampled-angle
binding matches every available dumped fine Euler component bit-for-bit
(zero differences over 124 captured winners). Regenerating RELION's CPU
double matrix and applying its explicit `Matrix2D::inv()` 3x3 cofactor order
then matches all 128 captured backprojection matrices bit-for-bit. The same
result is reproducible with NumPy float64 when the inverse operation order is
pinned; float32 CUDA/JAX regeneration remains a decisive null. Evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_scaled_rnd_binding_20260713_132000/captured_gate.log`
and
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_scatter_radius_cuda126_20260713_144500/audit/relion_cpu_euler_gate.json`.

The production follow-up is therefore narrow but architecturally important:
preserve RELION's current scorer matrices, retain perturbed Euler rows in
float64 long enough to generate a separate exact CPU M-step rotation stream,
and route only that stream to x-half backprojection. RELION itself uses these
distinct projector-plan and weighted-sum matrix paths. Do not replace the
scorer matrices with the CPU inverse matrices or infer the latter after Euler
metadata has been cast to float32.

Full evidence and the proposed 22-cutoff/five-base A100 golden regression are
in
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_scatter_signature_gate_20260713_113000/SCATTER_SIGNATURE_REPORT.md`.
After the radius gate closes, rerun the complete iteration-1 raw BPref and map
comparison. Accumulator relative L2 remains a localization metric; acceptance
still requires shellwise FSC, FSC-AUC, and the FSC-derived score/resolution
summaries against RELION and GT.

## 2026-07-13 autonomous K=1 native normalization result

Native RELION group mapping and `data_vs_prior > 3` scale-statistic support
close the single-fixture autonomous final-map gap. A100 job `11154968` matches
the ten-step size schedule and convergence boundary, then passes the final
merged normalized FSC-AUC gate at `0.997935505` with minimum shell FSC
`0.995978466`. RECOVAR GT FSC-AUC is `0.670747694`, versus RELION
`0.650834886`. Correlation was not computed. Treat this as acceptance of the
3k/128 white-noise K=1 fixture only; proceed to heterogeneous robustness,
larger scale, real-particle, and then K=4 gates.

## 2026-07-14 Case-22 TF32 Translation-Phase Root Cause

The first failing robustness fixture, case 22 (3k/128, radial noise 5 with
severe outliers), previously reached final merged RECOVAR-versus-RELION
FSC-AUC `0.8231073` and converged two iterations early. The earliest discrete
boundary was two iteration-1 normalized-CC winner flips. Exact RELION CUDA and
RECOVAR per-pixel captures rule out priors, projection interpolation, CTF,
reference norm, and reduction order. Hybrid replay assigns the material score
gap to the weighted translated image.

The image residual is a linear Fourier phase. Its fitted translation delta
matches A100 TF32 rounding of the requested shifts exactly; weighted phase RMS
falls from `1.24e-4`--`7.73e-4` to `1.24e-7`--`1.47e-7` after removing that
ramp. Commit `c741faee` requests `jax.lax.Precision.HIGHEST` in the generic,
full half-spectrum, and indexed/windowed candidate-translation phase dot
products. Follow-up commit `b658bd8d` covers the separate per-image Fourier
pre-shift phase constructor. A100 job `11177896` restores both RELION winners
and reduces full score-field RMS from `1.33e-5`/`1.58e-5` to
`1.63e-6`/`1.79e-6`; corrected target-score gaps are at most `7.45e-7`.
Forty-two focused tests pass.

Evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_early_score_audit_20260713_235725/REPORT.md`;
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_early_score_audit_20260713_235725/rel_cc_pixels_v7_20260714_014000/REPORT.md`;
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_early_score_audit_20260713_235725/rec_tf32_phase_highest_20260714_014700/phase_precision_audit.md`.

The active experiment is a clean autonomous case-22 full trajectory from
`b658bd8d`, with shellwise FSC/FSC-AUC, exact current-size schedule,
convergence iteration, finalization, and phase-generation timing as the
acceptance gates. Correlation is not an acceptance metric.

### First robustness acceptance after the phase fix

Case 15 (3k/128, 20% outliers, noise scale 1) clean-head job `11178306_1`
passes the complete automated and manual K=1 robustness gate. All 12 numbered
iterations pass per-half and merged FSC-AUC, the current-size schedule matches
RELION exactly (including the formerly divergent iteration-9 size 78), and
convergence/finalization semantics agree. Final merged cross-FSC-AUC is
`0.996927042`; RECOVAR GT FSC-AUC exceeds RELION by `+0.019123653`.

The final cross curve has a shallow high-frequency tail: minimum shell FSC is
`0.993255592` at shell 62, fifth percentile `0.993867482`, and no shell is
below `0.99`. This passes the predefined AUC gate and has no collapse, but the
final-only tail remains tracked as a cross-case diagnostic. Evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_robust_phase_precision_20260714_143051/15_small_outliers_3k_g128_pct20_noise1_bf80/`.

## 2026-07-14 Case-16 retained-posterior norm denominator

Case 16 first diverges at iteration 3 even after the translation-phase fix.
An exact iteration-3 replay from RELION's iteration-2 state matches all 3,000
hard poses, proving that the scorer is correct for identical inputs. Component
factorials instead localize the autonomous divergence to continuous M-step
state, dominated by a nearly common image-normalization multiplier.

Pinned RELION source provides the formula-level cause. It adds each updated
particle `normcorr` to `wsum_model.avg_norm_correction` without posterior
weighting, then divides by `wsum_model.pdf_class.sum()`. The latter is the
retained significant-support posterior mass and is slightly smaller than the
particle count. RECOVAR used the ordinary arithmetic mean and, at the K=1
K-class wrapper boundary, replaced the engine's retained `NoiseStats.sumw`
with the full class responsibility `N`.

The narrow fix preserves the K=1 engine's retained pose mass and divides the
unweighted norm-correction numerator by it. Same-A100 jobs `11183647` and
`11183648` validate the production plumbing: the two half-set denominators are
`1468.013404/1469` and `1529.966030/1531`, versus RELION-inferred
`1468.013624` and `1529.967028`. The internal average norm corrections move
from `5209.3864/5209.2030` to `5212.8874/5212.7233`, explaining about 90% of
the previous common drift. Default and diagnostic float32 fine-posterior paths
agree, so no diagnostic environment override is required. Full autonomous
case-16 trajectory and FSC/FSC-AUC validation remain pending before this fix
is accepted end to end.

Root-cause evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_robust_phase_precision_20260714_143051/16_small_anisotropic_outliers_3k_g128_pct25_noise3_bf80/audit_case16_divergence_20260714/exact_iter3_score_20260714_145500/analysis/NORM_SUM_WEIGHT_ROOT_CAUSE.md`.
Validation root:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/case16_norm_sumw_fix_validation_20260714_171639/`.

### Autonomous case-16 validation

A clean autonomous A100 run at commit `d685ba36` (job `11184169`) proves that
the retained-mass normalization fix restores the numbered control trajectory.
RECOVAR and RELION both execute 11 numbered iterations, use sizes
`[56,56,50,50,50,50,50,50,50,52,52]`, switch from HEALPix order 3 to 4 at
iteration 9, converge after iteration 11, and enter final all-data only after
convergence. Numbered merged cross-FSC-AUC is at least `0.999987621`; numbered
GT FSC-AUC differs by at most `4.01e-5`. Before the fix, pose differences grew
to `60/56` by iteration 6 and triggered false local search and convergence at
iteration 9.

Case 16 also passes end to end. The initially reported final all-data
cross-FSC-AUC of `0.743531728` was an audit-oracle error: it compared
RECOVAR's joined final reconstruction to the average of RELION's two
unfiltered half BackProjectors. RELION's authoritative post-convergence
all-data output is the unnumbered joined `run_class001.mrc`. Against that map,
the autonomous final cross-FSC-AUC is `0.996711421`, the minimum non-DC shell
FSC is `0.991779912`, and RECOVAR's GT FSC-AUC exceeds RELION by
`+0.008337396`. Corrected evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/case16_norm_sumw_fix_validation_20260714_171639/autonomous_final_hp_fix_commit_a78ec7c0/analysis/FINAL_HP_AUDIT.md`.

## 2026-07-14 Case-22 firstiter-CC reduction order

After the TF32 phase fix, case 22 retained one iteration-1 winner difference
on original particle 1552. A same-A100 RELION/RECOVAR candidate capture shows
that both implementations evaluate the same coarse parent and the same 32
fine candidates. The RELION winning margin is one float32 ULP
(`4.76837e-7`), while RECOVAR's complex-GEMM contraction reverses the two
candidates by `5.96046e-8`. Recomputing the cross term as RELION's explicit
float32 real/imaginary products followed by a float32 reduction restores the
RELION ordering; the projection-norm contraction is unchanged.

A production-shaped A100 microbenchmark measures the explicit contraction at
`7.272 ms` versus `4.775 ms` for the complex GEMM. The affected firstiter-CC
pass-2 kernel is a small part of a full refinement, so end-to-end trajectory
timing remains the required performance gate. Same-A100 validation job
`11185051` starts from the canonical RELION iteration-0 state and matches all
3,000 iteration-1 Euler/origin decisions (`0` mismatches at the float32/STAR
threshold), including particle 1552. The full autonomous trajectory remains
the acceptance gate before this numerical compatibility change is accepted.

Evidence roots:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it1_particle1553_capture_20260714_160000/ARITHMETIC_REPORT.md`;
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_explicit_cross_it1_validation_20260714_174500/`.

### Autonomous case-22 result after the firstiter-CC reduction fix

Clean-head A100 job `11185459` confirms that the explicit reduction fixes the
intended boundary but does not by itself close case 22. Numbered merged
cross-FSC-AUC remains at least `0.99887014` through iteration 8. RECOVAR then
chooses size 72 instead of RELION's 70, converges after iteration 9 rather than
11, and ends at final cross-FSC-AUC `0.8245735`. RECOVAR GT FSC-AUC is
`0.32852954` versus RELION `0.32606263`. The next boundary is ordinary
Gaussian pass-2 scoring, which still uses the complex contraction family.
Treat the firstiter fix as accepted at its exact 3,000-pose boundary, not as
acceptance of the complete robustness cell.

Audit:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_explicit_cross_full_20260714_174700/audit_job_11185459.json`.

### Case-22 iteration-2 ordinary-Gaussian operand audit

The native first difference after the firstiter-CC fix is one iteration-2
particle, original index 1203.  The aggregate map is still essentially equal
at that point: merged cross-FSC-AUC is `0.999993339694`.  The difference is
nevertheless trajectory-relevant because the autonomous run later reaches
only `0.990561796343` at iteration 9 and `0.824573534720` in the joined final
map.

A direct A100 RELION operand capture and a reference-fixed, metadata-matched
RECOVAR capture do not reproduce the hard winner difference.  Over
2,349,728 common fine candidates, both choose RECOVAR row 50295, translation
82 (global rotation 98175).  After an additive-constant alignment, the common
log-weight field differs by maximum `7.783e-4`, RMS `1.465e-4`, and 95th
percentile `2.585e-4`.  This experiment deliberately excludes post-cold-start
norm/group-correction replay, so it is not a fully bit-fixed-state claim.

The two supports each contain 2,349,792 candidates but swap 64 keys.  These
decode to two coarse pass-0 significance-boundary pairs per side: ranks
73430--73434, with score/log-weight margins between about `2.29e-5` and
`1.14e-4`.  Each coarse pair expands to eight rotation children by four
translation children.  This is a real cumulative-significance boundary, not
a fine-key mapping error, but none of the swapped keys is the fixed-capture
winner.

At the shared winner, RECOVAR's exact 1,860-position, 256-lane full-grid tree
gives raw `diff2=130.319137573`, versus RELION `130.355621338`.  Substituting
RELION's captured image explains `+0.008590698`; substituting its weight
explains `-0.000152588`; both leave `+0.028045654` unexplained.  The remaining
term cannot be assigned without RELION's fine projected reference.  Thus the
native hard mismatch is classified as upstream operand/state sensitivity,
not an intrinsic fixed-reference Gaussian winner-selection bug.  The next
one-factor experiment must replace the reference, norm/image correction,
noise/weight, and fine projected reference independently.

Evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_fixed_relion_capture_20260714_192000/EARLIEST_DIVERGENCE_AND_OPERAND_AUDIT.md`.
Machine audit:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_it2_fixed_relion_capture_20260714_192000/case22_it2_fixed_relion_field_audit.json`.

#### Closure of the missing projector/high-resolution factors

The follow-up exact-factor experiment uses the correct zero-based RECOVAR
debug state: `it000` is the post-iteration-1 reference scored in iteration 2.
It resolves the previously unexplained absolute score offset. RELION's
captured `highres_Xi2/2` is about `0.02803938`; subtracting it in float32 gives
the factorial trees exactly, while adding it preserves the candidate margin.
Across accepted captures its last bits jitter from `0x3ce5b2d8` through
`0x3ce5b2db`, proving numerical nondeterminism in the atomic reduction, but
all final candidate raw scores are bit-identical at this scale.

The isolated RELION `AccProjectorKernel` capture also separates inactive-grid
values from scoring values. The rectangular 60-by-31 buffer contains 399
pixels with `corr_img=0`; these explain effectively all of the apparent
full-grid projector relative-L2 near `0.5`. On the 1,461 active pixels,
RELION-versus-RECOVAR projector relative-L2 is `6.1965e-9` and `1.0538e-6`
for the two candidates, with corr-weighted values `2.6777e-8` and
`7.2781e-6`. Those residuals change the float64 term sums by only
`-1.70e-7` and `+5.70e-7`; both collapse to zero in RELION's exact 256-lane
float32 tree. With the same RELION reference, image, weight, and translations,
both projector implementations therefore give trees
`130.3275909424`/`130.3275756836`, raw scores
`130.3556365967`/`130.3556213379`, and the same one-ULP RELION winner.

The causal factor is the iteration-1 reference, not an active projector or
high-resolution-constant mismatch. Under one fixed RECOVAR projector, the
native reference gives `130.3276519775`/`130.3276672363` and a one-ULP native
winner; replacing only the reference with RELION's gives the one-ULP RELION
winner above. The two references have FSC-AUC about `0.999999999632`, so this
is a qualified numerical butterfly: an accumulator/BPref-to-reference
perturbation that is globally FSC-inert can still be causal at a knife-edge
candidate. It must not be erased by a new tie-break rule.

Two cross-device A100 repeats and two serial repeats on the same physical A100
all reproduce the RELION raw scores and winner exactly. The same-device
post-iteration-1 map FSC-AUC is `0.9999999999988929`. Across eight RELION maps,
pairwise FSC-AUC lies in
`[0.9999999999982315, 0.9999999999993140]`; maximum non-DC per-shell curve
spread is `1.2258e-11`. Map acceptance remains FSC/FSC-AUC only; correlation
was not computed. This local classification does not accept the autonomous
case-22 trajectory, whose final FSC-AUC deficit remains unresolved.

Exact projector audit:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_particle1203_factorial_20260714_201047/relion_fine_projection_capture/fine_projection_comparison.json`.
Same-device and cross-device FSC audit:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_particle1203_self_jitter_20260714_202700/self_jitter_audit.json`.

## 2026-07-14 Case-16 final SamplingPerturbation order

An exact RELION iteration-11 state replay through RECOVAR's final all-data
branch reaches merged cross-FSC-AUC `0.997899`; a high-precision replay using
autonomous half references and autonomous metadata reaches `0.996712`.
The autonomous and replay final maps agree at cross-FSC-AUC
`0.9999999997`, and all 3,000 final Euler/translation decisions are bitwise
identical. This independently confirms that the earlier `0.743531728` result
was solely the wrong final oracle, not a scorer, BPref, reconstruction, or
serialization failure.

The decisive manifest boundary is the final trial grid. Native autonomous
finalization used the capped exhaustive grid order 3 to scale
SamplingPerturbation, while RELION's final `sampling.star` records the active
local-search parent order 4. Both manifests exactly match their corresponding
canonical construction. Their 36,864 rotations differ by one common right
rotation of `0.682834` degrees. The targeted fix uses `state.healpix_order` for
native local-search final perturbation and preserves the exhaustive order for
global search. Focused merge guards pass `23/23`; the EM fast guard passes
`16/16`. The fix is required for exact RELION workflow semantics, although
its aggregate case-16 map effect is numerically negligible: pre-fix and
post-fix final maps agree at cross-FSC-AUC `0.9999999993`, while their correct
RELION-oracle FSC-AUC values are `0.996711349` and `0.996711421`.

Evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/case16_norm_sumw_fix_validation_20260714_171639/`.

## 2026-07-14 Case-13 final per-half noise bug

The earlier case-13 coarse-score classification used an off-by-one RELION
stack index and is retracted.  A corrected uninterrupted capture for original
particle 1682 uses one-based stack index 1683 and reproduces the authoritative
RELION raw-score checksum exactly: `t5=6559.189453125`,
`t10=6577.24951171875`, and `t10-t5=-18.06005859375`.  The corrected layout
mapping also includes RELION FFTW-y to RECOVAR centered-y roll by 64 rows.

The valid operand comparison identifies a workflow bug rather than a scorer
reduction bug.  Particle 1682 belongs to random subset 2, but RECOVAR's final
all-data branch duplicated half 1's `sigma2_noise` spectrum and used it for
both subsets.  RELION keeps `do_split_random_halves` enabled through the final
expectation, so each follower scores its subset with its own model and noise
spectrum; it combines the weighted sums only afterward and disables the split
only while writing the joined final model.  The measured RECOVAR-to-RELION
weight ratio is shell-constant to relative RMS `8.54e-7`, and substituting the
half-2 noise reduces the float64 `t10-t5` operand discrepancy from about
`0.225` to `3.80e-4`.  The remaining difference to RELION's live float32 raw
score is `1.49e-3` and remains a qualified numerical scorer boundary, not the
cause of this final support failure.

The narrow fix retains the two half-specific noise spectra for the joined
final E-step.  A fixed-state full final replay improves merged
RECOVAR-vs-RELION FSC-AUC from `0.993421942` to `0.997786`, clearing the
`0.995` gate.  RECOVAR merged GT FSC-AUC is `0.312351999`, versus RELION
`0.301136412`; both have FSC<0.5 at shell 19 and FSC<0.143 at shell 27.

Clean immutable A100 job `11190363` at integrated commit `d07915fa` supplies
the end-to-end validation.  It matches convergence at iteration 9 and the
exact current-size schedule `[56,56,48,48,48,48,48,48,48]`.  Every numbered
merged cross-FSC-AUC is at least `0.999999970691`; the final joined-map
cross-FSC-AUC is `0.997779297632`.  Final merged GT FSC-AUC is
`0.312357369405` for RECOVAR and `0.301136422552` for RELION, delta
`+0.011220946854`.  An independent repeat agrees in numbered merged FSC-AUC
within `8.9e-11` and final merged FSC-AUC within `1.70e-8`.  Correlation was
not computed.

This accepts the aggregate case-13 FSC/FSC-AUC trajectory.  Retract the prior
claim that iteration-9 particle 1466 retained a structured
`0.446167/0.180758` raw-score maximum/RMS difference and winner flip.  That
RELION diagnostic was started with `--continue run_it008_optimiser.star`.
Pinned RELION broadcasts `mymodel.sigma2_noise` from rank 1 during MPI
initialization, so the restarted subset-2 follower silently received half-1
noise.  RECOVAR correctly retained the half-2 curve used by the uninterrupted
trajectory.  Shellwise inversion of `corr_img` proved that the contaminated
RELION capture matched half 1 at relative RMS `2.75e-6`, while RECOVAR matched
half 2 at `3.00e-6`; the apparent weight ratio was exactly
`sigma2_half1/sigma2_half2`.

Fixed-state A100 job `11192981` preserved each follower's loaded continuation
noise and then failed closed on the subset-2 half-2 spectrum.  RELION and
RECOVAR effective sigma2 match half 2 at relative RMS `2.75e-6/3.00e-6`.
The captured RELION operands replay all 192 costs at maximum/RMS
`6.10e-5/2.45e-5`.  Replacing every active operand with RECOVAR's gives
maximum/RMS `0.001709/0.000237`, the same winner `(17,43)`, and Pmax
`0.445093562` versus fixed RELION `0.445093651` (delta `-8.8e-8`).  The
iteration-9 structured residual is therefore closed as diagnostic restart
state contamination, not a production RECOVAR behavior difference.

Final particle 188 remains a separate localization target with replay
maximum/RMS `0.187515/0.065117`, although its replay winner agrees and the
replay does not reproduce the uninterrupted hard-pose boundary.  Final
particle 2701 is an exact RELION top tie with only a few-ULP residual, and
particle 2828 has the same robust winner with a few-ULP residual.

Fail-closed capture rule: a mid-trajectory RELION restart is inadmissible for
per-half score attribution unless the capture either runs uninterrupted or
records the particle random subset and proves shellwise that
`CTF^2 * group_scale^2 / corr_img` matches that subset's previous-iteration
model STAR.  A score-array match alone does not validate the state.

Corrected operand evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case13_exact_it9_final_20260714_145938/relion_score_pair_capture_d476e6/case13_matched_rel_rec_operand_analysis.json`.
Fixed-state FSC evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case13_exact_it9_final_20260714_145938/relion_score_pair_capture_d476e6/recovar_parent_pass_dump_20260714/p1682_half_noise_fix_2d7ada89_cuda/benchmark_ledger.json`.
Immutable integrated FSC-only audit:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case13_integrated_d07915fa_immutable_retry_20260714_201100/integrated_fsc_audit.json`.
Matched-grid score audit:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case13_targeted_scores_d079_20260714_201200/JOINT_RELION_RECOVAR_SCORE_AUDIT.md`.
Corrected fixed-state operand audit, job `11192981`:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case13_it9_z1466_fixed_half_noise_20260714_211456/FIXED_STATE_OPERAND_AUDIT.md`.
Machine-readable operand and raw shell audit:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case13_it9_z1466_fixed_half_noise_20260714_211456/operand_substitution_audit.json`.
Corrected capture stdout/stderr:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case13_it9_z1466_fixed_half_noise_20260714_211456/logs/run_11192981.out` and
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case13_it9_z1466_fixed_half_noise_20260714_211456/logs/run_11192981.err`.

## 2026-07-14 Clean-head case-20 scale result

Same-H100 job `11185799` at `68a3f9e6` matches RELION's convergence after
iteration 11 and keeps every numbered merged cross-FSC-AUC near `0.9998` or
better, but chooses sizes 50 instead of 52 at iterations 8 and 10. Final
cross-FSC-AUC is `0.9851148007`; RECOVAR GT FSC-AUC is `0.0858743655` versus
RELION `0.0846081506`. The first discrete differences are only two
iteration-2 Gaussian pass-2 decisions. A same-H100 operand capture is the next
discriminator for the complex Gaussian cross contraction.

Audit:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_cleanhead_68a3f9e6_20260714_180000/case20/audit_job_11185799.json`.

### Case-20 iteration-2 Gaussian operand and iteration-1 BPref boundary

The fine Gaussian arithmetic oracle is RELION's direct ``diff2`` expression,
not RECOVAR's algebraically equivalent complex cross-term contraction.  Source
verification is important here: the CUDA SPA path instantiates
``REF3D=true, DATA3D=false`` with a 256-thread reduction block and translation
chunk size seven.  An initial 128-lane diagnostic was therefore invalid as an
exact RELION emulation and is not a production candidate.  Recomputing the
captured operands with the correct 256-lane tree preserves the scientific
localization.  For original particle 365, the native one-ULP winner gap is
removed by the captured RELION image operand.  For particle 469, RECOVAR's
image and reference favor row 27 by three ULPs; substituting RELION's saved
iteration-1 reference collapses that pair to an exact tie, while RELION's live
kernel favors row 34 by one ULP.  The full native-to-RELION span is therefore
four float32 ULPs.  All candidate masks, priors, rotations, translations, and
non-reference operands are otherwise identical at the captured boundary.

The reduction topology has a second, material detail.  RELION traverses the
entire current-size FFTW packed image, including zero-weight pixels outside
the scoring circle; those zero slots still determine the CUDA lane occupied
by every retained pixel.  RECOVAR stores only 1,275 active case-20 pixels, so
reducing that compact array consecutively is not equivalent to RELION's
56-by-29, 1,624-slot layout.  H100 fixed-state job ``11189614`` restores the
full-grid lane positions without materializing a hypothesis-by-full-image
tensor.  This alone changes particle 365 from compact winner ``(1,29)`` to
RELION-side winner ``(1,32)`` by one float32 ULP.  Particle 469 remains on
RECOVAR's side by three ULPs while the live RELION kernel is on the other side
by one ULP, leaving a qualified four-ULP operand discrepancy.  The kernel uses
under 5.1 MB of compiled temporary storage and takes about 0.09--0.11 ms for
these fixed rows.  This remains a diagnostic candidate, not a production
change: rotation chunks need one external minimum over the complete fine
support, K-class needs that minimum across classes, and centered log-evidence
must carry ``-min_diff2`` rather than the existing image-norm offset.

Validation report:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_it2_gaussian_capture_20260714_182500/case20_full_grid_diff2_validation.json`.

The reference difference is traced one layer earlier.  Same-H100 RELION job
``11188600`` captured the post-join iteration-1 BackProjectors and three map
stages.  With the exact RELION layout and scales ``-256^-2`` for the complex
numerator and ``256^-4`` for the positive weight, the inclusive supported-
radius relative-L2 residual is ``1.985202940e-4``/``1.992240058e-4`` for the
numerator and ``1.504021633e-6``/``1.529009109e-6`` for the weight by half.
Independent RECOVAR repeats differ by only about ``3.3e-8``, so this is a
systematic cross-implementation arithmetic difference rather than RECOVAR
atomic nondeterminism.

Cross-reconstruction job ``11188818`` localizes causality to accumulation.
RECOVAR reconstruction/post-processing fed the RELION accumulators reproduces
RELION's post-reconstruct maps at FSC-AUC
``0.999999999831/0.999999999840`` and the final post-solvent references at
``0.999999999985`` in both halves, with best-scale residual below ``7e-7``.
Feeding RECOVAR accumulators gives raw-stage FSC-AUC
``0.999999981523/0.999999981144`` and final FSC-AUC
``0.999999999449/0.999999999451``.  Tau2 derived from the RELION accumulators
matches the RELION model STAR through the untapered shells to serialized
precision.  Thus neither the Wiener solver, tau2 update, low-pass, nor solvent
flatten is the first cause.  The strong numerator-versus-weight asymmetry
points next to the translated complex hard-winner operand/phase arithmetic
before scatter.  Shell FSC/FSC-AUC remain the map-quality gates; accumulator
norms are localization diagnostics only.

Evidence root:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_it2_gaussian_capture_20260714_182500/`.

## 2026-07-14 Explicit RELION CUDA preprocessing runtime integration

Commit `241db84d` closes a real end-to-end routing bug in the explicit
`relion_cuda` image-preprocessing option.  The source-faithful CUDA operand
kernel was already qualified, but the adaptive coarse significance and sparse
pass-2 paths did not pass per-image float32 normalization factors and int32
integer pre-shifts to it.  The option therefore failed closed before its first
score rather than running the promised workflow.

The fix uses one typed batch-operand selector in both coarse significance
loops and sparse pass 2.  CUDA receives the unshifted real image plus explicit
normalization and shift operands; downstream scoring applies only the
remaining RELION group scale, avoiding a second normalization.  Host behavior
is unchanged.  Focused main-checkout tests pass `67/67`.

H100 job `11196916` at isolated commit `aeb337df` exercised both halves through
coarse normalized-CC scoring and sparse score/reconstruction and completed
one full iteration in 62.5 science seconds (Slurm elapsed 1m49s).  Canonical
non-DC sign-invariant FSC-AUC against RELION iteration 1 is
`0.999999999500` merged and `0.999999999448/0.999999999451` by half.  The
iteration-1 GT FSC-AUC is `0.098134227300` for RECOVAR and
`0.098134193656` for RELION (delta `+3.36e-8`).  The global map sign is the
known arbitrary sign boundary; no correlation metric participates in this
gate.  Evidence root:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_relion_cuda_aeb337df_iter1_20260714_232010/`.

## 2026-07-14 Case-26 native WAVG/atomic discriminator rejected

The source-faithful native CUDA WAVG diagnostic is not a causal repair and
must remain isolated.  An initial two-particle run (`11196641`) exposed a
diagnostic adapter bug: centered RECOVAR source indices were confused with
RELION FFTW scatter-coordinate indices.  Corrected job `11196729` separates
the two typed index vectors and proves exact first-iteration WTA semantics:
one positive float32 posterior equal to one per particle, matching RELION's
`sum_weight=1` and `significant_weight=0.9990000129`.

All-1000 H100 job `11196772` then couples RELION-style translation reduction,
`sincosf`, factor placement, and atomics in one kernel.  It changes the
half-1/half-2 numerator residual only from
`5.782326e-6/5.985726e-6` to `5.781899e-6/5.986944e-6`, and the weight
residual from `2.992471e-6/2.913891e-6` to
`2.993837e-6/2.914697e-6`.  The mixed `0.007--0.046%` changes remain hundreds
of times above RELION's `1.0--1.3e-8` same-H100 repeat floor.  Accumulation
topology is therefore rejected.

The retained earliest boundary is upstream pre-atomic arithmetic.  Aggregate
source values differ by approximately `7.2e-7/1.58e-6` for numerator and
`2.25e-7/2.21e-7` for weight by half, while geometry is exact.  Since
`Fweight=CTF^2*Minvsigma2` in this exact-WTA case, CTF and/or inverse-noise
construction must differ before image or translation phase can matter.  Raw
RELION operand capture job `11197096` and paired RECOVAR job `11197128` are
the next discriminator.  Audit:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_earliest_score_audit_20260714_214916/native_wavg_all1000_report_11196772.md`.

## 2026-07-14 Case-26 paired raw operands close the accumulation branch

Same-H100 jobs `11197096` (RELION) and `11197128` (RECOVAR) capture the two
paired hard winners RECOVAR-original 212/842 and RELION-particle 128/827.
All 1,227 active Fourier pixels map one-to-one between RECOVAR's 128x65 half
image and RELION's 56x29 current-size half image.  Source, scatter-coordinate,
and mapped RELION indices are unique, so support, folding, and layout are
closed exactly.

After applying only known representation conventions, relative L2 residuals
are `1.49e-7` for the unshifted image and `2.18--2.64e-7` for CTF.
Across all 116 translations, coefficient differences are at most one float32
ULP; for the two actual WTA winners, both coefficients and all 1,227
per-pixel float32 phase arguments are bit-exact, so phase contributes no
operand discrepancy.
Inverse noise is the largest raw residual at `9.062e-7`; the resulting
pre-atomic data/weight residuals are `1.00--1.50e-6` and
`9.78--9.84e-7`.  These quantitatively reproduce the earlier aggregate
signature gap without invoking a scatter or atomic-order explanation.

The inverse-noise attribution is independently demonstrated.  RECOVAR is
1,227/1,227 bit-exact to the float32 reciprocal reconstructed from serialized
`run_it000_half1_model.star`.  Fresh RELION's captured Minvsigma2 is exact to
the reciprocal of its retained in-memory bootstrap spectrum for all 29 used
shells.  That spectrum repeats bit-exactly on the same H100 yet differs from
the rounded STAR in all 65 shells.  RELION computes and broadcasts the double
spectrum, writes the model, and continues without re-reading it.  Thus no
science-formula, coordinate, or accumulation bug is supported, but a strict
boundary comparison must either restart RELION from the serialized state or
capture/feed the full-precision in-memory state.  Otherwise near-tie winner
drift at approximately `1e-6` is built into the comparison harness.

Audit artifacts:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_earliest_score_audit_20260714_214916/case26_paired_raw_operand_audit_11197096_11197128.json`
and
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_earliest_score_audit_20260714_214916/case26_paired_raw_operand_audit_11197096_11197128.md`.

## 2026-07-14 Explicit RELION CUDA full-trajectory gate

Case-20 H100 job `11197313` exercises the explicit `relion_cuda` path through
the complete autonomous workflow.  The science command exits zero after 11
numbered iterations and the valid post-convergence all-data pass.  RECOVAR
matches RELION's convergence at iteration 11 and the complete current-size
trajectory `[56,56,52,52,50,50,50,52,50,52,50]`.  Total science time is
638.1 seconds, external refinement wall time is 672 seconds, and observed GPU
memory peaks at 42,339 MiB on H100.

Every numbered merged cross FSC-AUC is at least `0.999988902`; numbered half
cross FSC-AUC is at least `0.999986015`.  The numbered RECOVAR-minus-RELION GT
FSC-AUC delta ranges from `-1.89e-6` to `+1.118e-5`.  The post-convergence
merged map has cross FSC-AUC `0.997634223`; RECOVAR and RELION final GT
FSC-AUC are `0.085752642` and `0.084608151`, respectively.  Mid-trajectory
particle decisions show a small real divergence rather than mere rounding,
but they contract by iteration 11: Pmax mean absolute difference `4.76e-4`,
rotation mean error `0.00681` degrees, and translation mean error `0.00154`
pixels.  Map quality and convergence remain matched.

Slurm records wrapper exit 2 only because the generic completion summarizer
was given a noncanonical directory layout after the successful science run.
The direct FSC-only audit supersedes that reporting failure; science was not
rerun.  Evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_relion_cuda_aeb337df_full_20260714_233032/full_trajectory_fsc_audit.json`
and
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_relion_cuda_aeb337df_full_20260714_233032/full_trajectory_fsc_audit.md`.

## 2026-07-14 Serialized-sigma discriminator proves the case-26 boundary

H100 job `11198286` reruns the same fresh RELION first-iteration workflow with
the serialized iteration-0 noise spectrum supplied through RELION's `--sigma`
boundary.  This preserves firstiter-CC, sampling, the 56x29 runtime layout,
the iteration-0 maps (byte-exact; FSC-AUC 1), and both audited WTA winners.
It avoids RELION continuation, which would disable firstiter-CC.

After this single state alignment, Minvsigma2 matches RECOVAR 1,227/1,227
float32 values for both particles with zero relative L2 and zero maximum
error, versus 298/1,227 in the retained in-memory bootstrap run.  Image and
CTF are bit-exact between the two RELION runs and retain their expected
RECOVAR residuals.  The iteration-1 maps before/after sigma alignment have
non-DC FSC-AUC `0.999999999983/0.999999999984` by half.  The remaining
combined pre-atomic residual is therefore ordinary cross-implementation
float32 FFT/CTF arithmetic: data `1.279e-6/8.260e-7` and weight
`1.010e-6/1.013e-6` for the two particles.  This proves the noise-state
attribution while showing that noise was not the only component of the
composite residual.

The phase-correction addendum supersedes only the phase fields in the first
paired audit: the actual WTA coefficients and arguments are bit-exact.
Evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_serialized_sigma_discriminator_20260714_234311/analysis/serialized_sigma_discriminator.json`,
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_serialized_sigma_discriminator_20260714_234311/analysis/serialized_sigma_operand_metrics.json`,
and
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_earliest_score_audit_20260714_214916/case26_paired_raw_operand_audit_phase_correction_addendum_11197096_11197128.json`.

## 2026-07-15 K=1 real-particle full trajectory and exact-reference boundary

The explicit `relion_cuda` workflow now completes the 10,000-particle real
10076 fixture on H100.  RELION and RECOVAR both converge at numbered iteration
22.  RELION takes 1,154 external seconds and peaks at 79,949 MiB; RECOVAR takes
1,835 seconds and peaks at 42,421 MiB.  These are same-GPU-model measurements,
not yet performance acceptance gates.

The numbered merged cross FSC-AUC first falls below the strict 0.995 gate at
iteration 7 (`0.994590971`).  The first local-search iteration, iteration 12,
is the sharpest early boundary at `0.970153731`, with minimum/p05 non-DC FSC
`0.896640744/0.927132339`.  Both programs have exact replayed incoming
poses, translations, optimiser controls, and half-set assignments at that
scoring boundary.  The post-convergence final all-data outputs are kept
separate because the current GUI-quality RECOVAR path intentionally leaves
final grid correction off.

Exact RELION incoming-reference probes separate accumulated map drift from an
intrinsic iteration-engine mismatch.  Jobs `11200283`, `11200284`, and
`11200243` replace the scoring references at global iterations 3, 5, and 7;
their merged output FSC-AUC values become `0.999997757`, `0.999998854`, and
`0.999998278`, respectively.  Their per-particle Pmax mean absolute errors are
only `1.28e-4--1.48e-4`.  The global engine is therefore effectively correct
given exact incoming references, while small early reconstruction differences
are amplified across autonomous iterations.

The same intervention at the first local iteration is decisive in the other
direction.  Job `11199901` improves iteration-12 merged FSC-AUC only from
`0.970153731` to `0.976932488`; minimum/p05 non-DC FSC remain
`0.920176938/0.943454456`.  Pmax mean absolute error remains `0.0556090`, with
43.24% of particles above 0.01 and 18.68% above 0.1.  Thus accumulated map
drift is a secondary amplifier, but a material intrinsic local-path mismatch
exists at the first local iteration.

Seven of the ten largest iteration-12 Pmax outliers choose a neighboring fine
translation exactly one 0.5-pixel grid step away, with directions varying by
particle.  Two other outliers retain the identical winning pose while Pmax
differs by approximately 0.618.  This rejects a constant translation-origin
offset and proves that posterior score spacing, normalization, or support must
be inspected independently of the discrete winner.  The next discriminator is
three separate exact-reference captures: the untouched fused posterior, the
materialized fine score/operands at current size 122, and the parent
significance score/operands.  Each capture must pass an FSC-AUC/state
instrumentation-inertness gate before its arrays are used.

Primary evidence root:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_10k_relcuda_h100_plan_20260714_234251`.
Exact-reference audits are under
`reference_replay_iter{3,5,7,12}/fsc_pmax_audit.json` in that root.

## 2026-07-15 K=4 iteration-3 cliff and rejected causes

The clean H100 five-iteration K=4 trajectory closes iteration 1 but exposes a
real behavior cliff at iteration 3.  Direct classwise cross FSC-AUC is
`[0.999999979, 0.999893582, 0.999999980, 0.999999974]` at iteration 1 and
`[0.999997892, 0.999883088, 0.999969918, 0.999843680]` at iteration 2, then
falls to `[0.961838483, 0.953936126, 0.955462580, 0.918876565]` at iteration
3.  Class agreement falls from 0.999 at iteration 2 to 0.9386 at iteration 3;
614 classes, 1,339 rotations, and 864 translations differ.  GT FSC-AUC deltas
remain within approximately `9.8e-4`, so the current fixture has not yet lost
ground-truth quality even though strict trajectory parity clearly fails.

The lone iteration-1 pose difference, original particle 3591, is a qualified
numerical tie: both implementations have the same 66,816 coarse candidates,
winner, and 32 fine candidate IDs/order.  RECOVAR's winner margin is
`1.639e-7`, RELION's is `4.470e-7`, and the aligned centered score residual
envelope is `9.537e-7`.  No blanket tie explanation is accepted later:
iteration 2 has 23 unique affected particles, and original particles 2907 and
8083 have the two largest Pmax discrepancies in the entire 10,000-particle
set while also changing rotation/support.

Two controlled hypotheses are rejected.  Same-numbered RELION tau2 replay is
an exact trajectory null; an independent fixed-accumulator substitution
changes supported FSC-AUC by at most `1.73e-6`.  Forcing the manual projector
is worse from iteration 1, so the production texture interpolation remains
the correct path.  Exact RELION iteration-2 class references also fail to
repair iteration 3: classwise FSC-AUC improves by only
`0.000101--0.000451`, class agreement only from 0.9386 to 0.9394, and Pmax
mean absolute error only from `0.047604` to `0.045975`.  The iteration-3 cliff
is therefore intrinsic to K-class scoring/posterior/support or M-step
behavior, not inherited map drift.

Original particle 6388 (RELION internal row 5989) is the primary matched-score
target.  It agrees through iteration 2, including class, pose, shift,
significant count, and Pmax (`0.158637/0.157887`), then changes class with
Pmax `0.975441/0.221761`, 171.577 degrees angular separation, and a one-pixel
shift at iteration 3.  The first H100 capture, job `11200644`, fails the
mandatory instrumentation-inertness gate: its RELION iteration-3 row changes
from the clean class-4, Pmax-`0.221761`, 97-significant-sample result to class
1, Pmax `0.976015`, and 108 significant samples.  None of that job's active
operand/CC-component arrays are admissible causal evidence.  A passive-only
retry, job `11201054`, disables those active paths but also fails the strict
gate.  Its iterations 1--2 remain extremely close (classwise map FSC-AUC above
`0.999999996` at iteration 2), yet its small Pmax/numerical drift is
chaotically amplified at iteration 3 into 644 class changes; target 6388 has
the clean significant count and Pmax within `3.1e-4` but selects class 1
instead of class 4.  Its arrays are therefore excluded too.  A no-dump control,
job `11201123`, reproduces the same cliff, proving that the rebuilt patched
RELION binary itself is non-inert even when all dump environments are disabled.
An attempted one-iteration continuation from the clean iteration-2 optimiser
boundary is also invalid because it does not reproduce the clean continuation
state.  The causal capture therefore moves to the stable iteration-2 boundary,
targeting original particles 2907 and 8083, whose large Pmax/support/rotation
differences precede chaotic iteration-3 amplification.

Trajectory evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_current_head_full5_relcuda_h100_20260715_003000/analysis/trajectory_gate.json`.
Reference replay evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it3_relion_reference_replay_h100_20260715_010941/analysis/reference_replay_fsc_state.json`.
Decision audit:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_it3_decision_audit_20260715_014500/REPORT.md`.

## 2026-07-15 Targeted posterior discriminators

The K=1 iteration-12 fused-posterior implementation is not the local-search
cause.  H100 jobs `11201155` and `11201156` independently capture the fused
and forced-materialized fine paths with exact incoming RELION references.  The
captures pass their instrumentation gates: merged map FSC-AUC is
`0.999999994760/0.999999994665`, p05 non-DC FSC is
`0.999999962714/0.999999961869`, and all Pmax differences from the undumped
exact-reference control are zero.  For their four shared target particles,
candidate rotations, translations, parent/child identities, masks, support,
scores, log normalizers, posteriors, Pmax, and winners are bitwise identical.

The remaining K=1 difference is structural relative to RELION, not a close
tie.  For fixture index 6536 (STAR 85521), RELION's winner is absent from
RECOVAR's finite fine support and the closest same-rotation RECOVAR candidate
is separated by score `7.659`.  For fixture index 4194 (STAR 54772), RELION's
translation is RECOVAR rank 4 with score gap `0.926178`; RECOVAR's close top
pair does not contain the RELION winner.  Fixture indices 8421 and 9640
choose the exact same winner in both programs, but RELION/RECOVAR Pmax are
`0.364345/0.982648` and `0.380272/0.998304`, with RECOVAR top-two gaps
`4.71278` and `6.64923`.

The parent-to-fine expansion is also exact and is no longer a candidate cause.
For all four targets, the finite fine mask is the exact 32-child expansion of
the significant parent cells, with no candidate-ID, rotation, translation,
parent-child, or mask mismatch.  The divergence is already present in the
parent or fine score surface relative to RELION.  For fixture 6536, RELION's
winner belongs to parent rotation 193770 / translation 13, which RECOVAR
scores but prunes at the parent boundary: it is parent rank 2 with posterior
`0.00075794` and score gap `7.18414`, while translation 14 alone is retained.
For fixture 4194 the RELION parent is retained, but its fine translation falls
to RECOVAR rank 4 with posterior `0.118761` and score gap `0.926178`.
Fixtures 8421 and 9640 retain the same final winner but have substantially
over-concentrated RECOVAR posteriors.  These are structural score/posterior
differences, not discrete tie-breaking or fused-kernel behavior.  Instrumented
RELION iteration-12 candidate captures localize them to parent scoring size as
described below.

K=1 capture evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it12_targeted_capture_bf49f93f_20260715_012401/fused_vs_fine_shared_comparison.json`.
Parent-expansion evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it12_targeted_capture_bf49f93f_20260715_012401/parent_to_fine_support_comparison.json`.

The instrumented RELION target captures further localize this to fine
hypothesis/support construction, before priors or posterior normalization.
The capture binary is not globally instrumentation-inert, so these arrays are
used only after target-level qualification against the forced-perturbation
no-dump control.  Targets 85521, 54772, and 126792 pass that target gate; target
110844 differs only in Pmax by `1e-5` with identical winner, pose, shift, and
significant count and remains explicitly marked failed-closed.  Orientation
and offset log-priors agree within `1.43e-6` and `4.77e-7`.  Conditional on the
common finite support, posterior total-variation distance is only
`2.72e-5`, `2.50e-7`, `1.87e-4`, and `1.17e-6`.  The support itself is not the
same: RELION/common/RECOVAR candidate counts are `128/32/32`, `128/64/160`,
`384/160/192`, and `128/32/32`; RELION assigns only `0.2260`, `0.99998`,
`0.36875`, and `0.38080` probability to the common support.  The large Pmax
differences are therefore caused by absent/excluded hypotheses, not by prior
or normalization arithmetic on a shared hypothesis set.

The parent-support cause is the ordering of local angular refinement and
Fourier-size selection.  RELION iteration 12 enters `expectation()` with
sampling order 3, computes its pass-1 parent image size from the old 7.5-degree
sampling (`56` pixels), and only then updates the sampling order to 4 for the
current local parent grid and order-5 fine children.  RECOVAR updated the order
first and recomputed the parent image size from 3.75 degrees, scoring at `110`
pixels.  RELION consequently selects `4/4/12/4` parents for the four targets,
while RECOVAR selects `1/5/6/1`; both expand every selected parent into exactly
8 rotations by 4 translations.  Aligned parent scores across the wrong
56-versus-110 Fourier bands have post-common-shift p95 residuals of
approximately `27.19/26.71/8.85/16.47`, while inferred combined-prior
residuals remain below `6.87e-5`; this is not a tie or prior effect.

RELION/RECOVAR hypothesis-alignment evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it12_relion_target_capture_20260715_022636/analysis/relion_recovar_posterior_alignment.json`.
Parent support-rule audit:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it12_relion_target_capture_20260715_022636/analysis/parent_support_rule_audit.json`.

The K=4 iteration-2 outliers are instead inherited amplification from tiny
reference drift.  With exact standard RELION iteration-1 references, corrected
target-qualified RELION captures and RECOVAR have candidate-support Jaccard
`1.0` for every class of original particles 2907 and 8083: 3,488/3,488 and
3,168/3,168 total candidates, including identical reconstruction support and
all eight classwise top keys.  Combined-prior error is at most `9.54e-7`,
centered score-with-prior p95 is `7.34e-5--1.15e-4` (worst maximum
`0.001005`), and posterior L1 after common renormalization is
`7.51e-6--1.90e-5`.  No score, prior, support, or posterior behavior mismatch
exists at this matched iteration-2 boundary.  The intrinsic K=4 investigation
therefore remains at iteration 3, where exact iteration-2 reference replay did
not close the trajectory cliff.

K=4 score/support evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_orig2907_8083_recovar_exactref_pass2_h100_20260715_020900/analysis/score_support_no_correlation.json`.

The K=4 iteration-3 raw-score cliff is caused by a group-scale state mismatch.
For original particle 6388, RELION scores with runtime scale `0.972485065`
while its rank-1 post-M-step model STAR serializes approximately `1.315036`;
RECOVAR's exact replay used the serialized value.  A clean H100 one-factor A/B
changes only that particle's scoring scale.  The serialized-scale arm retains
the wrong class-3 branch, class masses approximately
`[0, 0.0001786, 0.9998214, 0]`, support Jaccard
`0.351/0.357/0.446`, and centered score-with-prior mean/p95/max error
`16.655/31.314/43.710`.  The runtime-scale arm restores support Jaccard `1.0`
and every classwise top key; RECOVAR class masses become
`[0, 0.0166850, 0.3179502, 0.6653648]` versus RELION
`[0, 0.0166907, 0.3178691, 0.6654402]`, with the exact class-4 winner.
Centered score-with-prior mean/p95/max error falls to
`0.001392/0.003004/0.005733`.  This is causal behavioral evidence, not a
numerical tie or downstream support defect.

A per-rank iteration-2 state dump identifies the underlying RELION behavior.
The piecewise `MlWsumModel::pack` path sizes the group-scale XA/AA payload from
the one optics group instead of the 10,000 particle groups.  Only group 0 is
MPI-combined; among groups 1--9999, 5,027 have rank-1-only statistics and 4,972
have rank-2-only statistics, with no overlap or both-zero group.  For target
group 5989, rank 1 has raw XA/AA scale `1.348988547` and normalizes it to
`1.314142312`, while rank 2 has zero AA, substitutes the default scale 1, and
normalizes it to `0.973957723`.  The writer model matches rank 1 within
`5e-7`; the particle's next E-step can use rank 2's live state instead.  Strict
n=3 parity therefore requires follower-local scale vectors and exact
iteration-to-iteration particle ownership; a single global or rank-1
serialized scale vector cannot reproduce RELION.

K=4 scale A/B evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it3_orig6388_runtime_scale_ab_h100_20260715_024500/runtime_scale/analysis/score_support_no_correlation.json`.
K=4 per-rank scale-state evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_relion_scale_state_rank_audit_h100_20260715_034500/analysis/scale_rank_state_no_correlation.json`.

## 2026-07-15 local-size and continuation-noise fixes

The K=1 iteration-12 parent-score mismatch is closed.  The local-search loop
now snapshots the incoming HEALPix order before RELION's sampling update and
uses that order for pass-1 Fourier-size selection.  The corrected replay uses
size 56 in both programs instead of RECOVAR's previous size 110.  This removes
the structural p95 centered-score residuals of `8.85--27.19`.

A second exact RELION restart semantic explained the remaining subset-2-only
residual.  In `MlOptimiserMpi::initialise`, only follower rank 1 initializes
`sigma2_noise`, then broadcasts its spectrum to every follower.  On an
AutoRefine continuation, rank 1 has loaded the serialized half-1 model, so the
first expectation after process start scores both subsets with half-1 noise.
This broadcast occurs once at process initialization; later uninterrupted
iterations update and retain half-specific spectra.  RECOVAR now emulates the
broadcast only in replay slot 0 and in explicit continuation/final-only probe
initialization.  Numbered replay slots 1 and later remain half-specific.

After both fixes, the four qualified iteration-12 targets have centered raw
score p95 residuals `0.002356/0.001899/0.000731/0.001178`, maximum posterior
TV `7.70e-5`, and exact significance masks.  This is the float32 arithmetic
envelope rather than a remaining behavioral mismatch.  Evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_it12_sizeorder_fix_target_h100_20260715_031500/analysis/size_order_and_process_start_noise_parent_score_gate.json`.

Strict replay currently supports one optics-group noise spectrum per half.
RELION broadcasts every optics group's spectrum, while RECOVAR's scorer is not
yet optics-indexed.  Multi-optics model STAR input therefore fails closed
instead of silently selecting `model_optics_group_1`.

## 2026-07-15 dynamic MPI dispatch correction

The first full K=4 follower-scale implementation used the exact seed-2803
shuffle but incorrectly divided that shuffled order into two static equal
rank ranges.  A three-rank H100 audit rejects that model: static ownership
agrees with the rank-local XA/AA evidence for only 49.0849% of particles
(5,091/9,999 nonzero groups disagree).  By contrast, every captured RELION
leader dispatch bundle has a uniform observed owner.

The source boundary explains the result.  `MlOptimiserMpi::expectation()` is
an on-demand leader/follower queue: the leader sends the next contiguous
shuffled-position bundle to whichever follower requests work next.  In the
actual `--pool 3 --j 4` oracle, the leader's effective `nr_pool` is 12
particles: each numbered iteration contains 834 non-overlapping jobs covering
sorted positions 0--9999 exactly once.
The initialization-time `divide_equally` ranges do not control expectation.
The seed shuffle itself remains exact: a direct all-rank capture from job
`11204924` is byte-identical to RECOVAR's `mt19937(random_seed + 1)` binding,
and internal particle IDs follow the micrograph-sorted `run_it000_data.star`
row order.  Only the runtime chunk-to-rank schedule was wrong.

The static implementation's three-iteration K=4 trajectory is therefore
rejected, despite closing iterations 1--2.  Its classwise cross FSC-AUC at
iteration 3 is `[0.954334, 0.949943, 0.948496, 0.918550]` with class agreement
0.9327.  Exact mode now fails closed unless given a per-iteration dispatch
schedule captured from the same RELION oracle run; an explicit follower count
of zero remains available only as a labeled non-parity diagnostic.  The
schedule records dynamic owners in shuffled-position order, validates complete
non-overlapping pool coverage, follower bounds, iteration coverage, pool size,
particle count, and random seed, then remaps owners through the authoritative
RELION data-STAR identity order for both scoring scales and XA/AA accumulation.

Rank audit evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_relion_scale_state_rank_audit_h100_20260715_034500`.
Shuffle capture evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_sorted_idx_relion_capture_h100_20260715_041000`.
Same-run three-iteration dispatch/oracle evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_dynamic_dispatch_oracle3_h100_20260715_043500`.
Rejected static-ownership trajectory:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_worker_scale_full3_h100_20260715_033000/analysis/trajectory_gate3.json`.

The K=1 one-iteration boundary proof above is not yet promoted to a full
trajectory claim.  An uninterrupted replay previously allowed RECOVAR's
post-M-step state to advance to order 4 before iteration 12, so it still used
coarse size 110.  Strict replay now carries the preceding numbered sampling
STAR order separately from the live post-M-step state: iteration 12 uses the
saved order 3 for size 56, then the current order 4 for parent/fine sampling.
Full same-H100 trajectory job `11205287` confirms that this is a real causal
fix, but does not promote the complete K=1 trajectory.  Iteration-12 merged
cross FSC-AUC improves from `0.97005673` in the previous uninterrupted run to
`0.98852049`; minimum/p05 non-DC FSC improve to
`0.96042513/0.97207520`.  The corrected run nevertheless first fails the
strict FSC-AUC gate at iteration 7 (`0.99483511`), reaches its lowest merged
FSC-AUC at iteration 16 (`0.97643540`), and only recovers to `0.99435727` at
iteration 22.  Its final all-data merged FSC-AUC is `0.98126001`.  This is
trajectory-level behavior, not arithmetic noise.  Detectable drift starts
earlier, and iteration 5 is the first materially amplified global boundary:
merged FSC-AUC is `0.99776762`, per-particle Pmax MAE/p95 are
`0.0139803/0.0442924`, and half-set direction-prior relative-L1 errors are
approximately `1.24%/1.00%`.  Existing exact-incoming-RELION-map probes at
iterations 3, 5, and 7 restore merged FSC-AUC to
`0.99999776/0.99999885/0.99999828`, respectively.  The early failure is
therefore recurrent amplification of a small one-step global E/M residual,
not a schedule mismatch.  The next K=1 discriminator is a current-head exact
iteration-5 boundary capture: compare posterior/support, BPref data/weight
before and after the low-resolution half-join, and cross-reconstruct both
RELION and RECOVAR accumulators through both reconstructors.  This separates
scoring/posterior, accumulation/join, and reconstruction while the search is
still global.

Corrected K=1 full-trajectory evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_full22_saved_order_fix_h100_20260715_042500/analysis/fsc_gate.json`.

The exact dynamic-dispatch K=4 replay is a decisive positive result.  Same-H100
job `11205803` uses the captured owner schedule from the three-iteration
RELION oracle.  Iterations 1 and 2 retain their prior parity.  At iteration 3,
classwise cross FSC-AUC improves from the rejected static-owner values
`[0.954334, 0.949943, 0.948496, 0.918550]` to
`[0.999319, 0.998186, 0.999506, 0.998382]`; class agreement improves from
`0.9327` (673 mismatches) to `0.9989` (11 mismatches).  Per-class GT FSC-AUC
deltas are between `-7.75e-5` and `+1.86e-5`.  The dynamic queue ownership is
therefore the dominant K=4 iteration-3 bug, and the static ownership model is
retired.

The K=4 discrete-decision gate remains fail-closed.  The 11 class changes, 51
rotation changes, and 36 translation changes do not yet have complete
candidate-score tie evidence.  Their map and GT FSC evidence is excellent but
does not by itself prove that every flip is numerical.  The next K=4
discriminator is a fixed iteration-3 replay for exactly those affected
particles, capturing complete raw scores, priors, support, posterior, and
winner margins from the same RELION/RECOVAR boundary.  Also, the schedule
oracle used the previously identified non-inert rebuilt RELION executable;
before a clean standard-RELION trajectory claim, repeat schedule capture with
a production-inert logging method or establish same-run inertness explicitly.

Dynamic-schedule replay evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_dynamic_dispatch_replay3_h100_20260715_044500/analysis/trajectory_gate3.json`.

The iteration-3 RECOVAR score audit rejects a blanket numerical-tie
explanation for the residual 11 class changes.  Job `11206124` captures every
fine pass-2 class surface for the 59-particle union of class, rotation, and
translation disagreements: 236 class-qualified files.  The dump path is
science-inert relative to the undumped dynamic replay: per-class map FSC-AUC
is at least `0.999999955` across all three iterations, class/rotation/
translation decisions are identical, and the largest Pmax change is
`4.43e-5`.  Five of the 11 class changes (original indices 1371, 1680, 2029,
3749, and 9946) prefer the RECOVAR winner over the best candidate in RELION's
reported class by more than `0.1` log-weight units; four exceed `1.0`, and the
largest gap is `20.39`.  Only two class changes have a global top-two margin
below `0.01`.  These are not all reduction-order ties on RECOVAR's propagated
iteration-3 surface.

This does not prove a scorer bug because the audit scores RECOVAR's own
iteration-2 maps.  The matched-boundary discriminator instead closes the
residual completely.  H100 job `11206519` combines exact RELION iteration-2
references with the exact dynamic dispatch schedule.  At iteration 3, class
agreement becomes `1.0`, class/rotation/translation mismatch counts are all
zero, and classwise direct FSC-AUC is
`[0.999999211, 0.999999290, 0.999996695, 0.999997948]`.  GT FSC-AUC deltas
remain within `-5.23e-6--+4.30e-6`.  Therefore the 11 residual class changes
and all observed iteration-3 map loss are amplification of the small
incoming-map drift, not an additional hard-decision or support bug.  Pmax is
not yet fully closed: mean/p95 absolute error falls to
`7.66e-5/1.84e-4`, but original particle 8083 remains at
`0.518075/0.641103` (absolute error `0.123028`) with the same class-4 winner;
eight particles remain above `0.01`.  Their next discriminator is the
incoming follower-scale/norm/noise/prior state, with post-iteration-2 live
follower scale the leading hypothesis.  The owner-transition split makes that
boundary much sharper: all 94 particles above `0.001`, including all eight
above `0.01`, keep the same follower between iterations 2 and 3.  None of the
4,792 owner-switch particles exceeds `3.55e-4`; stable-owner outliers are
balanced 47/47 between the two followers and do not have extreme serialized
rank-1 scales or norm factors.  Job `11206981` materializes RECOVAR's new
pre-score/post-M-step follower-scale trajectories.  Its post-iteration-2
follower-0 vector matches RELION's serialized rank-1 vector with mean/median/
p95 absolute errors `4.54e-5/1.40e-6/2.29e-4`, but has a sparse maximum error
of `0.0111584`.  That maximum is the physical group of the largest owner-0
Pmax outlier.  Among stable owner-0 particles, 21/44 groups with scale error
above `0.001` have Pmax error above `0.001`, versus 26/2524 below that scale
threshold.  This is a real state discrepancy, not an unstructured score-jitter
tail.

A fail-closed diagnostic replay now accepts complete follower-scale matrices
at selected numbered iterations and injects them before owner remapping and
pre-score telemetry.  The first numbered iteration is rejected because its
resident image-normalization/scale factorization does not yet exist.  Every
requested row must be reached and applied exactly once before refinement may
return a successful result.  H100 job `11207267` changes only iteration-3 follower 0
to RELION's serialized post-iteration-2 rank-1 vector, retaining RECOVAR's
captured follower-1 vector.  Stable-owner-0 Pmax errors above `0.001` collapse
from 47 to 1 and errors above `0.01` from 3 to 0; mean/p95/max absolute error
fall from `1.173e-4/4.132e-4/0.0413453` to
`1.294e-5/4.028e-5/0.00489362`.  The three largest owner-0 errors become
`9.05e-6/1.60e-5/1.50e-5`.  Stable owner 1 is unchanged at 47 errors above
`0.001` and five above `0.01`, as required for a causal one-factor test.
Class, Euler, and translation decisions remain exact, class mismatch versus
RELION remains zero, classwise direct FSC-AUC is
`[0.999999949, 0.999999595, 0.999996903, 0.999998020]`, and GT FSC-AUC deltas
remain within `-2.58e-6--+1.87e-6`.  Follower-0 scale mismatch therefore
causes essentially its entire Pmax residual family.  Same-oracle RELION
rank-2 live state remains the sole large Pmax boundary.  The
pass-1 dump count in
`11206124` is zero because its filter incorrectly used fine size 46 while the
iteration-3 coarse pass uses size 14; no rerun is needed because the
matched-boundary discrete state is exact.

K=4 pass-2 score-margin evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_dynamic_dispatch_score59_h100_20260715_052000/analysis/pass2_score_margin_audit.json`.
Instrumentation gate:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_dynamic_dispatch_score59_h100_20260715_052000/analysis/instrumentation_inertness_fsc.json`.
Matched-boundary exact-reference trajectory:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_dynamic_dispatch_exactref3_h100_20260715_052300/analysis/trajectory_gate3.json`.
Pmax owner/state audit:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_dynamic_dispatch_exactref3_h100_20260715_052300/analysis/pmax_owner_scale_audit.md`.
Post-iteration-2 follower-scale audit:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_dynamic_dispatch_scale_trajectory2_h100_20260715_054948/analysis/post_it2_follower_scale_audit.md`.
Rank-1 causal replay:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_exactref3_rank1scale_hybrid_h100_20260715_061000/analysis/causal_result.md`.

The decisive both-follower replay closes the K=4 state boundary. H100 job
`11207973` uses the exact iteration-2 maps, exact dynamic dispatch row, and
both followers' self-consistent post-iteration-2 scale matrices. The
pre-score replay assertion passes bitwise. All 10,000 class assignments and
class occupancies are exact; translations differ by at most `3.814e-7` pixel.
Both stable-follower populations have zero Pmax errors above `0.001`, with
maxima `3.204e-4/5.838e-4`. Overall Pmax mean/median/p95 absolute errors are
`1.297e-5/5.368e-6/3.705e-5`. One particle remains unresolved: particle 6848
has a 150.464-degree rotation change and Pmax `0.962491/0.983205`, while its
complete competing score arrays were not captured. It is therefore not
classified as a numerical tie. Direct per-class FSC-AUC is
`[0.9999999802, 0.9999999797, 0.9999999715, 0.9997842943]`; GT FSC-AUC deltas
are `[-1.34e-6, -3.96e-8, +8.64e-7, +1.50e-5]`. The remaining K=4 work is a
complete-score capture for particle 6848 and then an uninterrupted trajectory
using the now-verified dynamic-dispatch/follower-state boundary.

Both-follower decisive result:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_exactref3_bothscale_selfconsistent_h100_20260715_064500/analysis/decisive_result.md`.
Content-bound reusable schedule:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_dispatch_scale_oracle3_paired_h100_20260715_063000/schedule_v2/dispatch_schedule_schema2.npz`.
Content-bound follower-scale replay:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_dispatch_scale_oracle3_paired_h100_20260715_063000/schedule_v2/follower_scale_replay_it3_schema1.npz`.

A new self-consistent cold capture measures the full competing score arrays
for the same particle identity but does not retroactively close the old
trajectory residual.  Job `11208454` completes all three science iterations
and writes the result NPZ in 821 seconds; Slurm reports `FAILED 1:0` only
because its end-of-wrapper source hash guard correctly detects concurrent
edits to `iteration_loop.py` and `run_full_refinement.py`.  Launch provenance
is preserved and the science arrays pass the corrected strict analysis.
Classes 1/2 have zero fine hypotheses in both implementations; classes 3/4
have exact common support `128/256`, and reconstruction support is exactly
three class-4 hypotheses in each.  Rotation matrices and translation keys are
identical after class-specific permutation, and the winner is the same
class-4 pose.  Across all 384 fine hypotheses, centered pre-prior score
max/p95 errors are `0.002056/0.000747`, posterior L1 is `0.0001520`, and Pmax
is `0.533119/0.533195`.  These are measured small residuals, not a blanket
numerical-noise classification.  The cold RELION Pmax differs from the old
oracle (`0.533` versus `0.962`), so particle 6848 in job `11207973` remains
unresolved until its own complete competing scores are captured.

The capture also exposed and fixed a comparison-tool bug: K-class matrix mode
could select another class's larger fine-Euler table before filtering rows by
class.  Fine-Euler discovery and minimum-row calculation are now
class-specific, with a regression test; strict scratch analysis asserts exact
fine and reconstruction common-support counts.

Particle-6848 cold-capture audit:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_particle6848_scores_selfconsistent_h100_20260715_071500/analysis/particle6848_result.md`.

The old particle-6848 boundary was then tested directly rather than inferred
from the cold capture.  A restart from the saved old `run_it002` checkpoint
with the full-precision rank-1 scale vector first exposed two continuation
changes: sampling perturbation advanced from `-0.38530` to `-0.35825`, and
pool size reset from 3 to 1.  Forcing the old perturbation and pool makes the
RELION continuation select the RECOVAR hypothesis: Pmax `0.983191` versus
RECOVAR `0.983204722`, rotation geodesic error `0.000172` degree, and
translation error `2.16e-5` Angstrom.  The old uninterrupted RELION result
remains Pmax `0.962491` with a 150.464-degree different rotation.  H100 job
`11210025` repeats the probe with the original two-follower MPI/projector
broadcast topology and produces the same RECOVAR-like result, ruling out MPI
topology.  Iteration 3 is a global search, so rounded previous Euler centers
are not the primary explanation.

The strongest remaining boundary distinction is reference precision:
uninterrupted RELION builds PPref from the resident CPU-double Iref, whereas
both continuation and exact-boundary RECOVAR reload the mode-2 float MRC and
rebuild PPref.  The old live Iref/PPref was not captured, so its score arrays
cannot be retroactively reconstructed.  The mismatch remains real and
unresolved, but is now classified specifically as a serialized-boundary
diagnostic artifact, not generic numerical noise and not a demonstrated bug
in an uninterrupted RECOVAR trajectory.

Old-boundary restart audit root:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_old6848_boundary_recapture_h100_20260715_073828/`.
The human-readable and machine-readable audit records are respectively
`analysis/OLD6848_RECAPTURE_AUDIT.md` and
`analysis/old6848_recapture_audit.json`; `provenance/final_audit.sha256`
binds the final evidence bundle.

The first K=1 iteration-5 BPref capture is quarantined.  Job `11206341`
correctly stops before RECOVAR because its dump-enabled RELION arm misses the
strict inertness threshold against the earlier oracle: iteration-5 half-1
FSC-AUC is `0.999992807`, half-1 p05 FSC is `0.999982676`, and merged FSC-AUC
is `0.999997049`.  No relaxed threshold is used and none of its BPref or
particle arrays are admitted as causal evidence.  Same-job/same-H100 paired
job `11206596` also fails closed: dump versus no-dump iteration-5 half-1 and
merged FSC-AUC are `0.999993212/0.999997211`, half-1 p05 FSC is
`0.999983314`, target 5400 differs only in Pmax by `1e-6`, and all-particle
Pmax mean/p95/max absolute differences are
`5.75e-5/1.69e-4/0.03545`.  RECOVAR again does not run under that strict
cross-run gate.

The required two-arm no-dump/no-dump control is decisive.  Same-job,
same-physical-H100 job `11207058` runs the identical cold RELION command twice
with every capture hook disabled.  By iteration 5 the two ordinary runs have
half-1/merged FSC-AUC `0.996578561/0.998526844`, half-1 p05 FSC
`0.993169764`, and Pmax mean/p95/max absolute differences
`0.00710685/0.0234492/0.643981`.  Relative to that intrinsic cold-run
envelope, the dump/no-dump map deficits are 504x smaller for half 1 and 528x
smaller for the merged map; Pmax mean/p95/max are 124x/139x/18x smaller.
Therefore the original strict gate remains red and is not relaxed, but the
hook effect is substantially below ordinary RELION rerun variability.
Captured arrays are admissible only for self-consistent within-run causal
analysis tied to the dump arm's maps and state; they are never evidence of
bitwise cross-run identity and must not be mixed with another cold oracle.

K=1 rejected-capture gate:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it5_exact_boundary_h100_20260715_051100/analysis/inertness_gate.json`.
Paired same-H100 rejected-capture gate:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it5_paired_inertness_h100_20260715_052612/analysis/paired_inertness_gate.json`.
Hook-versus-intrinsic classification:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it5_paired_inertness_h100_20260715_052612/analysis/hook_vs_intrinsic_envelope.md`.
Two-no-dump envelope:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it5_two_nodump_h100_20260715_055424/analysis/two_nodump_envelope.json`.

A separate strict K>1 correctness bug is fixed before convergence testing.
The final joined all-data expectation previously reused the last numbered
iteration's particle owners even though RELION launches a new dynamic work
queue.  The final path now requires the next absolute-iteration dispatch row,
selects post-numbered live follower scales with those owners, preserves the
resident norm/scale ratio, rebuilds expanded XA/AA worker-group IDs, and fails
closed when the final row is absent.  This bug cannot explain numbered
iteration-3 results, does not apply to K=1, and only affects K>1 runs that
actually enter the converged final expectation.

The K=1 iteration-5 within-run boundary capture completed in Slurm job
`11207474`; its wrapper failed only after science because a generic final-map
helper requested `run_class001.mrc` from a deliberately non-converged,
five-numbered-iteration RELION run.  The capture manifest itself passes.  The
raw pre-join accumulator difference is not an RECOVAR M-step error: RELION's
low-resolution half join changes its own numerator by approximately `3.8%`
relative L2 and its weight by `0.324%`.  After that join, RECOVAR versus RELION
relative L2 is `1.028e-4/9.548e-5` for the half-1/half-2 numerators and
`2.148e-5/2.014e-5` for the weights.  RELION's post-join and pre-reconstruct
arrays are exactly identical.  These residuals are small but not declared
numerical noise until the captured cross-reconstruction matrix determines
their FSC/FSC-AUC effect.

The same within-run matched boundary already rules out a hidden score/support
bug for the captured particle.  Its aligned 32-candidate raw-score residual
has p95/max absolute values `0.00148/0.00218`, posterior total variation is
`3.98e-6`, the winner is identical, and both implementations select exactly
two reconstruction candidates with no support disagreement.  With exact
RELION iteration-4 half maps and state injected at the iteration-5 scoring
boundary, the resulting half-1/half-2/merged map FSC-AUC values are
`0.999999981/0.999999982/0.999999984`.  Thus the one-step iteration-5 boundary
is closed to the numerical envelope; the bad uninterrupted trajectory is the
amplification of small recurrent residuals, not a missing discrete branch at
this boundary.

The controlled two-by-two cross-reconstruction matrix localizes the residual
further.  Switching only RELION versus RECOVAR post-join accumulators gives
merged FSC-AUC `0.99999999562` through the RELION reconstructor and
`0.99999999561` through the RECOVAR reconstructor.  Switching only the
reconstructor gives `0.99999999980` on the RELION accumulator and
`0.99999999997` on the RECOVAR accumulator.  RECOVAR's reconstruction
implementation is therefore not the trajectory bug.  Replaying RECOVAR's
native per-half tau closes another `1.07e-8` of FSC-AUC deficit, versus
`4.39e-9` from the accumulator and approximately `2.0e-10` from the
reconstructor.  Valid same-H100 job `11208106` confirms this
ranking: its native-tau replay versus the saved RECOVAR map has merged FSC-AUC
`0.999999994884`; A100/H100 AUC differences are at most `1.68e-11`.  The
earlier job `11208095` is discarded because JAX fell back to CPU.  The
remaining K=1 work is upstream numerical fidelity in the tau/FSC, BPref, and
particle score/posterior operands; it is no longer a hardware,
reconstruction-algorithm, or support question.

The H100 microbatch-order discriminator further excludes reduction partition
as the dominant cause.  Job `11208415` changes only the iteration-5 cap from
`549550` to `35933` (2,500 two-image buckets), retaining the exact injected
RELION iteration-4 maps and all earlier settings.  Post-join BPref numerator
relative L2 changes by only `4.736e-6/4.861e-6` for half 1/2 and weight by
`8.843e-7/9.047e-7`, approximately 4--5% of the persistent RELION gaps.
There are no rotation or translation changes and Pmax changes by at most
`1.407e-4`.  Merged map FSC-AUC versus RELION is
`0.999999984013/0.999999984134` for the baseline/microcap arms, while the two
RECOVAR arms have FSC-AUC `0.999999992781`.  Thus tau formula, shell binning,
reconstruction, hardware, and dominant microbatch reduction order are ruled
out.  The first real mismatch is upstream post-join BPref input arithmetic;
broad matched posterior/contribution capture is still required to separate
score/posterior arithmetic from interpolation/backprojection accumulation,
so the residual is not labeled harmless numerical noise.

Within-run iteration-5 boundary evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it5_withinrun_boundary_h100_20260715_061427/analysis/bpref_boundaries.json`.
Particle score/support evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it5_withinrun_boundary_h100_20260715_061427/analysis/particle396_operands.json`.
Numbered map FSC evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it5_withinrun_boundary_h100_20260715_061427/analysis/numbered_map_fsc.json`.
Cross-reconstruction evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it5_withinrun_boundary_h100_20260715_061427/analysis/cross_reconstruct_it5.json`.
Native-tau replay evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it5_withinrun_boundary_h100_20260715_061427/analysis/replay_recovar_native_tau_it5.json`.
Same-H100 native-tau replay:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it5_withinrun_boundary_h100_20260715_061427/analysis/replay_recovar_native_tau_it5_h100.json`.
Causal boundary conclusion:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it5_withinrun_boundary_h100_20260715_061427/analysis/causal_conclusion.json`.
Microbatch-order discriminator:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_it5_microcap35933_h100_20260715_074500/analysis/microcap_result.json`.

### Optional high-precision canonical replay

When cheaper boundary checks still cannot distinguish an algorithmic mismatch
from floating-point effects, use a small reusable replay diagnostic.  Freeze
the relevant operands and geometry, then compare float32 in original and
canonical order with float64 in each program's order and a shared canonical
order.  If upstream precision is in question, optionally recompute the
operands themselves in float64 rather than merely casting captured float32
values.  This is an escalation tool for ambiguous residuals, not a mandatory
step for every parity run and not a proposal to change production precision.
Keep its artifact schema limited to the boundary being tested; use exact array
metrics internally and FSC/FSC-AUC for map effects.

The reusable implementation is
`scripts/validate_bpref_device_signature.py`.  Its deterministic
`logical_host_order` is explicitly not a reconstruction of CUDA atomic order.
Cross-engine mode requires complete, non-overlapping signature shards and a
native accumulator panel for each engine, verifies the frozen pass/class and
compact-window boundary, and fails nonexact comparisons by default.  Casting
captured float32 operands to float64 is labeled as such and cannot justify a
precision diagnosis.  A `precision_consistent_with_verified_recomputation`
classification requires a versioned, hash-bound artifact that certifies
upstream float64/complex128 operand generation; the diagnostic currently
validates that contract but does not yet produce such recomputed operands.
The hardened CLI and 46 focused tests pass at commit
`df154023c3e0935fcf2b34f97db7371a3326a046`.  A real particle-396 validation
replays 46,728 atomic contributions exactly and passes an exact self-compare;
its audit root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/canonical_bpref_hardened_validation_20260715_175600/`.

Coarse Gaussian score reductions have a separate fail-closed replay in
`recovar/em/gaussian_reduction_replay.py`.  Its v1 schema is deliberately
pinned to the frozen K=1 row-7881 RELION boundary, including source and
executable hashes, GPU UUID, scientific inputs, candidate geometry, and the
exact CUDA lane layout.  It distinguishes production float32, reordered
float32, promoted captured operands, and genuinely recomputed
complex128/float64 contributions; genuine high-precision conclusions are
centered-only unless the initial image-energy term is also recomputed.  This
module is diagnostic only and is not called by production EM scoring.

### Clean uninterrupted K=4 three-iteration gate

H100 job `11210525` is the first fully immutable, autonomous three-iteration
K=4 boundary in this series.  It uses clean detached RECOVAR commit
`8fa143f9e24457fa2cb67781b8ea1b91ff57ea98` and RELION commit
`f2c1a384400aec37dc6805856a5ba645650a44f1`, root-local CUDA and RELION-binding
builds, a same-run content-bound dispatch schedule, both rank-local scale
states, and verified pre/post input manifests.  The science completes and the
job exits 2 at the intended strict trajectory gate.

Quality remains high but a real recurrent residual compounds.  Minimum direct
per-class RECOVAR-versus-RELION FSC-AUC is `0.999893582`, `0.999843695`, and
`0.998209732` in iterations 1--3.  Maximum absolute per-class GT FSC-AUC delta
grows from `2.56e-6` to `2.27e-5` to `8.84e-5`, while class agreement is
`1.0/0.9990/0.9981`.  The union of particles with a class, rotation, or
translation decision mismatch grows from `1` to `23` to `61`.  Divergent
particles have Pmax absolute differences up to `0`, `0.39890`, and `0.56936`;
several later class flips therefore cannot be declared harmless close ties
without their candidate arrays.

The earliest mismatch is iteration-1 particle 3591: its rotation is
16.92 degrees different while first-iteration Pmax is hard at one and all
follower scales are one.  Follower-scale topology is therefore not the root
cause, although exact rank-local replay remains necessary: after iteration 2,
rank 2 differs from the serialized rank-1 writer state by mean/max
`0.13394/0.73682`.  RECOVAR's remaining iteration-3 local-scale error is not
enriched among divergent particles and is not demonstrated as the cause.
The next K=4 diagnostic must capture complete pre-argmax candidate scores,
priors, normalization, support, canonical candidate identities, owner/scale,
and incoming live projector fingerprints beginning at particle 3591; the run
must not be extended beyond iteration 3 until that earliest boundary is
classified.  On the same H100, RELION takes 192 seconds and RECOVAR 576 seconds,
so the current strict implementation is approximately 3x slower.

Immutable K=4 evidence root:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_uninterrupted_cold3_dynamic_scale_commit8fa143f9_h100_20260715_083000_retry3/`.
The principal records are `analysis/trajectory_gate.json`,
`analysis/particle_arrays.npz`, `analysis/live_vs_serialized_scale.json`, and
`provenance/science_artifacts.sha256`.

Strict K>1 launchers now preserve this boundary instead of silently reverting
to static or disabled follower ownership.  The 100k completion launcher
requires an explicit `K4_RELION_DISPATCH_SCHEDULE` captured from its RELION
fixture.  The K-class robustness launcher requires an absolute executable in
`EM_KCLASS_MATRIX_RELION_REFINE_MPI` that honors `RELION_DISPATCH_LOG`, builds
the schedule from that case's just-completed oracle, and passes it to RECOVAR.
The stock RELION executable therefore fails before submission rather than
wasting a long run that cannot produce strict parity evidence.  A strict
replay may remain a numbered-iteration diagnostic without unnumbered final
files, but if it enters RELION's final all-data boundary it now requires both
`run_sampling.star` and `run_optimiser.star`; stale numbered-state or
unperturbed fallbacks are not accepted as parity evidence.

Schedule provenance is content-bound, not path-bound.  RELION text-log schema
v2 records `(iteration, rank, sorted_position, original_particle_id)` for every
particle.  RECOVAR schedule NPZ schema v3 persists both follower ownership and
the exact `original_particle_id_by_sorted_position` permutation for every
iteration; schema-v2 schedules and legacy four-column logs cannot be migrated
exactly and fail closed.  The schedule stores a manifest hash over the exact
RELION state artifacts and captured dispatch log plus a generated
dispatch-topology metadata sidecar, a separate hash
of particle identities and parity-relevant labels in authoritative row order,
and a derived oracle ID.  The runner recomputes those hashes against the active
replay/init directory; relocation without content changes remains valid, while
missing/modified state, reordered particles, legacy schedules, and follower
scale checkpoints from another oracle fail closed.  Follower-scale replay
artifacts additionally declare their numbered pre-score boundary and source
checkpoint paths; those checkpoint bytes must be present in the verified
schedule manifest.  Validation also reconstructs the owner matrix from the raw
dispatch chunks and reconstructs the numbered pre-score matrix from the prior
post-M-step follower TSVs; merely claiming current manifest paths is
insufficient.  The hardened K=4 reusable oracle has ID
`ddb61ddd4d9d8abec93fab9f9ac7e41863c1f7e9da14f2bd65ecc15ad5c2937b`
and 36 manifest artifacts.

## 2026-07-15 Scoped K=1 BPref capture and K=4 particle-3591 boundary

The scoped K=1 BPref capture now has a validated target at iteration 5, half 1,
original particle 396.  Its host replay is exact, while the device shadow is at
the expected float32 reduction scale (data relative L1 `4.06e-8`, weight
relative L1 `3.78e-8`).  The immutable science root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_it5_h1_p396_scoped_capture_commit63ec1bdc_h100_prepared_20260715_133414/`;
the corrected post-hoc audit, using the native SPA CTF width of 9, is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/k1_it5_posthoc_validator_job11225077_20260715_140300/target_capture_audit.json`.

The five-arm unreachable-target experiment did not change any discrete state,
but several small continuous differences exceeded the paired control envelope.
That is unresolved rather than evidence that the inactive diagnostic changed
the algorithm.  The earliest difference is the iteration-1 `tau2_shell_sum`;
its inputs still match and the shell sum uses the JAX/GPU `bincount` reduction,
not a host NumPy fallback.  The next discriminator freezes the iteration-1,
half-1 boundary across four ordinary and four unreachable-environment fresh
processes, records effective mode hashes and zero launch counters, and compares
`H_comb`, `Ft`, and shell sums before invoking canonical float64 replay.  The
five-arm report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_it2_fivearm_single_h100_prepared_20260715_132747/analysis/five_arm_report.json`.

That frozen-boundary discriminator is now complete.  Job `11228588` emitted
all eight independent replay payloads on H100 UUID
`GPU-0d7b80c7-fef8-e346-6332-de36ae1af518`; the Slurm step failed only after
science because the original analyzer looked up one cross pair in reverse
lexical order.  Hash-verified post-hoc analysis shows one exact effective-mode
hash, zero diagnostic launches, and exact `H_comb`, shell sums, and shell
counts across every arm.  The maximum unreachable-to-ordinary relative-L1 is
`7.489e-8` for `Ft_y` and `2.879e-8` for `Ft_ctf`, versus ordinary repeat
diameters `7.461e-8` and `2.827e-8`; both remain inside the predeclared twice-
control envelope.  This closes only the concern that an unreachable diagnostic
target perturbs this iteration-1/half-1 boundary.  It is not a general K=1
trajectory-parity claim.  The independently checked audit is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/k1_it1_h1_frozen_boundary_retry3_posthoc_20260715_152900/AUDIT.md`.

For K=4, the six-arm particle-3591 fixture isolates the first behavioral
divergence to the iteration-1 `firstiter_cc` fine winner: RECOVAR selects fine
candidate 30 and RELION candidate 18, with all class assignments and all other
poses identical.  The two candidates are separated by sub-micro-score margins.
FSC remains the map quality gate: class-2 cross-engine FSC-AUC is
`0.99989358`, versus `0.99999998505` for the RECOVAR repeat.  A frozen operand
factorial exonerates production-precision PPref generation, image/CTF/phase
operands, and reduction order in isolation; a RELION-style 256-lane tree over
RECOVAR operands still selects candidate 30.  The remaining earliest boundary
is device-produced projected-reference interpolation from the common PPref and
matched matrices.  The immutable science root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_p3591_cross_engine_identity6_globalwinner_h100_prepared_20260715_134127/`,
and the preliminary factorial audit is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/k4_p3591_operand_factorial_20260715_143600/AUDIT.md`.

The follow-up RELION device-operand job `11227130` completed on one H100 with
the control and capture arms bound to the same physical GPU.  Target identity,
all 32 candidates, and all 840 packed pixels passed.  Capture inertness is at
repeat scale: per-class FSC-AUC is at least `0.99999999960`, and minimum
non-DC FSC is at least `0.99999999519`.  The capture root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_p3591_relion_fine_operands_sm90_prepared_20260715_151500/`.

An exact layout-and-scale-mapped comparison now localizes the winner change.
Seven projected-reference rotation rows have maximum absolute errors below
`1.96e-8`; rotation row 4 contains one `1.509e-6` outlier at RELION packed
pixel 242 (`y=11`, `x=11`).  With otherwise identical captured RELION
image-side operands, the RELION reference selects candidate 18 and the RECOVAR
reference selects candidate 30.  Replacing only that one candidate-18 pixel is
sufficient to select candidate 30.  This proves the projected-reference operand
boundary explains this discrete near-tie; it does not yet distinguish texture
staging from sub-ULP coordinate/texture-fraction behavior.  The hashed report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/k4_p3591_projection_compare_job11227130_20260715_145100/projected_reference_comparison.json`.

The target-only H100 microharness in job `11228213` resolves that last
distinction.  The RECOVAR compact staging path and the RELION-direct staging
path produce bitwise-identical real and imaginary float32 texture arrays.
Using RELION's exact matrix-x-times-source-x-first expression order then
reproduces every captured RELION projected-reference value bitwise over all
eight rotations and 840 pixels.  RECOVAR's reversed addend order differs from
RELION at exactly the known row-4/pixel-242 boundary at material scale.  At
that pixel the two source orders move the `y` texture coordinate by one
float32 ULP across the hardware interpolation half-bin and change the complex
reference by `1.509e-6`.  Thus the earliest proven K=4 defect is contracted-FMA
association caused by projector source operand order, not texture staging or
reduction noise.  This is an intermediate-array proof only; the production
change still requires a fresh K=4 FSC/FSC-AUC trajectory gate.  The report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_p3591_projector_coordinate_sm90_prepared_20260715_151900/analysis/projector_coordinate_report.json`.

The production correction uses that exact operand order in both CUDA texture
projector output paths.  H100 job `11229870` completed two clean control arms
and one target capture on physical GPU
`GPU-0d7b80c7-fef8-e346-6332-de36ae1af518`; it failed only when an older
instrumented-source validator required metadata fields that the clean current
payload schema does not emit.  A hash-bound post-hoc adapter validates the
current coarse/fine arrays and exact candidate geometry without fabricating
those absent fields.  Candidate 18's target-pixel reference error falls from
`1.509e-6` to `1.58e-10`, its centered-score residual is `1.49e-8`, and both
engines select candidate 18.  All three RECOVAR-to-matched-RELION class-map
comparisons use the identity permutation; the minimum FSC-AUC is
`0.99999997377` and the minimum non-DC FSC is `0.9999995218`.  This closes the
particle-3591 one-iteration projector boundary.  It does not replace the
remaining multi-iteration and heterogeneous robustness gates.  The accepted
reports are under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/k4_p3591_projector_fix3_posthoc_job11229870_20260715_160048/analysis/`.

## 2026-07-15 K=4 projector-fix three-iteration trajectory

The first clean multi-iteration retry exposed a real routing bug after its
iteration-1 science completed.  Job `11231468` reached the fused K-class pass-2
route at iteration 2 and failed because the common sparse dispatcher forwarded
the inactive `bpref_device_signature_active` diagnostic flag to
`compute_k_class_pass2_stats_sparse_fused`, whose signature did not accept it.
Commit `111b8fde65725bb2cebbcfae82dd1f251221dcb9` makes the inactive flag an
explicit optional argument on that fused route.  The false/default path is
inert; an attempted active capture still fails closed because active capture is
currently single-class and non-fused only.  Focused fused-versus-legacy,
signature-scope, API, and Ruff checks pass.

The fresh same-H100 three-iteration trajectory in job `11232258` then completes
all RELION and RECOVAR science.  RELION takes 189 seconds; RECOVAR takes 606
seconds, including 581.2 seconds in refinement.  Postprocessing in the original
launcher fails after science because it opens a trusted local object-bearing
NPZ with `allow_pickle=False` and assumes a root-local fixture path.  A
read-only, checksum-bound post-hoc recovery validates the prepared package,
science inputs, external fixture, native binaries, runtime libraries, source
commit, and results before evaluating the trajectory.

Every declared map-quality gate passes.  Direct per-class RECOVAR-versus-RELION
FSC-AUC minima are `0.999999973956`, `0.999998968049`, and
`0.999636237928` for iterations 1--3.  The minimum RECOVAR-minus-RELION GT
FSC-AUC delta is `-3.51164e-6`, well inside the `-0.002` gate.  Current sizes
`40/44/46`, HEALPix order 1, non-convergence through iteration 3, and the lack
of an invalid final-all-data step also match.

Strict underlying-array parity is not yet closed.  The earliest nontrivial
continuous difference is iteration-2 particle 5993, whose Pmax differs by
`0.0531957` although its discrete decisions match.  At iteration 3, particle
1513 changes class; particles 1513, 2136, 4685, 7661, 7700, and 9357 change
rotation; and all except 7700 also change translation.  Class agreement is
`0.9999`; iteration-3 Pmax absolute differences have mean `0.000213818`, p95
`0.000544221`, and maximum `0.408298`.  These are unresolved parity failures,
not declared numerical ties, because the run did not capture the complete
candidate scores, priors, normalization, support, and posteriors.  The next
diagnostic freezes the exact incoming iteration-2/iteration-3 boundaries for
those particles plus a matched low-error control and captures those arrays
before considering a float64 canonical replay.

The immutable science root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_cold3_projector_fix_111b8fde_h100_retry2_prepared_20260715_164432/`.
The checksum-bound post-hoc root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/k4_cold3_job11232258_posthoc_20260715_170527/`;
its artifact manifest SHA-256 is
`bdfadc9574d9bd78c65a880e89108f60034308fa9a80dcf053084b9733ddbcae`.

## 2026-07-15 K=4 score-capture quarantine and projection-cache discriminator

The first score-discriminator launch, job `11234038`, completed its RELION
arms but rejected the package before RECOVAR because its validator expected
only iteration-2/3 state files while the capture intentionally emitted the
complete iteration-1--3 state set.  Retry job `11234492` corrected that
validator and completed both three-iteration RECOVAR arms on H100 UUID
`GPU-0d7b80c7-fef8-e346-6332-de36ae1af518`.  The original job then failed on
an older capture-schema check after all science had finished.  A hash-bound
post-hoc recovery verifies the checkout, frozen CUDA library, same physical
GPU, complete captures, and FSC/FSC-AUC inertness metrics.

The result is quarantined: the score capture is not inert and therefore cannot
explain the production trajectory.  For iteration-2 particle 5993, the
ordinary RECOVAR arm has Pmax `0.73098427`, while enabling the score dump
changes it to `0.67779261`, close to RELION capture's `0.67780534`.  RELION's
instrumented and control arms keep all iteration-2 discrete decisions equal,
but their Pmax values differ by as much as `3.42e-4`; by iteration 3 this
amplifies to per-class control-versus-capture FSC-AUC values
`0.9605575/0.9456710/0.9594118/0.9224703`.  Captured candidate arrays are thus
observations from a perturbed trajectory, not production parity evidence.

Within that quarantined capture only, particle 5993 has 4,576 RELION and 4,608
RECOVAR candidates, with 4,576 shared and 32 RECOVAR-only candidates.  The
extra candidates carry total posterior mass `6.57e-14`, have zero
reconstruction support, and do not contain the winner.  Seven other targets
have full candidate-geometry bijections.  These facts are retained for
debugging but are explicitly not promoted to production support or score
mismatches.  The accepted quarantine report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/k4_it23_job11234492_posthoc_hardened_20260715_180550/analysis/POSTHOC_RECOVERY.json`;
its prepared-artifact manifest SHA-256 is
`44dae2ea25d55460907d8c3395e80d6f3f37eb102a1a636093d0c1d184a70504`.

The capture unexpectedly disables the sparse pass-2 projection cache.  This
is now a leading causal discriminator rather than an assumed instrumentation
detail: the no-cache captured arm is much closer to RELION at particle 5993
than the ordinary cached arm.  Commit `148e85a7` adds explicit cache
`auto/on/off` and dump-conservative-execution diagnostic controls while
preserving production defaults.  The next fixed-state experiment compares
no-dump cache-on, no-dump cache-off, dump cache-off, and, if needed, dump
cache-on.  If the two cache-off arms agree while cache-on differs, dumping is
inert under a matched execution path and the cache implementation becomes the
production bug candidate.  No float64 classification is warranted before
that structural A/B closes.

## 2026-07-15 Projection-cache exoneration and K=1 full trajectory

The matched no-dump projection-cache A/B closes that structural discriminator.
Job `11236836` ran cache-on and cache-off through two K=4 iterations on the
same H100 with identical inputs and execution controls.  All class, rotation,
and translation decisions are equal.  Iteration-2 Pmax differs by at most
`3.09e-5` (p95 `3.46e-6`), particle 5993 differs by `4.05e-6`, and the minimum
per-class cache-on/cache-off FSC-AUC is `0.999999954461`.  Both arms retain the
production particle-5993 value near `0.73098`, rather than the quarantined
capture value near `0.67779`.  The cache therefore does not explain the
RELION-versus-RECOVAR score gap.

An independent exact projector harness reaches the same conclusion at the
operand boundary.  For all four classes and all 4,608 fine rotations, caching
the complete projection table and gathering the target rows is bitwise equal
to projecting those target rows directly for score, reconstruction, and
absolute-square arrays.  Across fresh full-EM processes, iteration-1 BPref
accumulators vary only at float32 CUDA atomic-reduction scale: data relative L1
is about `5.45e-8` and weight relative L1 about `1.7e-8`, whether the cache
setting is changed or held fixed.  A same-cache repeat is retained as the
control envelope before classifying any later amplified difference as
algorithmic.  The exact harness is under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/k4_projection_cache_audit_20260715_190008/`;
the full A/B is under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_projection_cache_ab_111b8fde_h100_retry2_prepared_20260715_185750/`.

That same-cache control is now complete in job `11237485`, and it exposes an
important nonlinear repeat branch.  Two independent cache-on/no-dump processes
have identical class, rotation, and translation decisions through iteration 2,
but particle 5993's Pmax is `0.7309870` in one and `0.6777922` in the other.
Across all particles the Pmax absolute difference has p95 `3.43e-6` and maximum
`0.0531948`; four significant-support counts differ by one.  The minimum
per-class repeat FSC-AUC is still `0.999998971966`.  The low Pmax branch is the
same value previously attributed to score-dump instrumentation, so that
attribution is withdrawn: it is an ordinary autonomous GPU-reduction
butterfly.  A dump capture may be interpreted only against this repeat envelope
and then frozen at one exact incoming boundary for canonical float32 and
float64 replay.  The repeat report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_cache_on_repeat_111b8fde_h100_prepared_20260715_190604/analysis/cache_on_repeat.json`.

The follow-up candidate audit localizes that branch to one coarse pass-1
support decision.  The low branch contains 4,608 fine candidates for particle
5993, while the high branch is its exact 4,576-candidate subset.  The missing
32 candidates are the eight rotation and four translation children of coarse
parent `(class=0, rotation=14, translation=20)`.  They are also present in
RELION, carry total low-branch posterior mass `0.0727703450`, and include the
second-ranked candidate with posterior `0.0679322077`.  Removing them predicts
Pmax `0.7309867850`, only `1.17e-6` from the observed high-branch value.  Shared
scores differ by at most `3.81e-5`; fixed-support posterior precision and order
span only `1.32e-6`.  The coarse significant count changes `144 -> 143`.

This material parent is distinct from the 32 old-RECOVAR-only candidates that
RELION omits: those come from parent `(class=0, rotation=1, translation=14)`,
carry only `6.57e-14` total mass, and have no reconstruction support.  Given
either branch's fine candidate set, RECOVAR's current threshold and the
source-matched RELION ascending-float32 threshold reproduce the same fine
reconstruction mask (`85` low, `84` high).  The unresolved boundary is
therefore coarse pass-1 significance formation.  RELION filters positive
float32 weights, sorts ascending, and uses a CUB float32 inclusive scan of the
lower tail; RECOVAR currently sorts normalized weights descending and sums the
retained mass with JAX.  These are algebraically equivalent but not
finite-precision/order equivalent.  Raw coarse weights and cutoff-neighbor
cumulative sums must be captured before changing production behavior.  The
versioned three-way report and audit are under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/k4_p5993_dump_posthoc_20260715_203600/`.

Job `11238115` completed both cache-on science arms and all 32 requested dump
payloads.  Its Slurm exit is a post-science harness failure: the environment
regex matched `RECOVAR=` but omitted every `RECOVAR_*` variable, and the exact
log comparison retained leading timestamps.  Envelope-qualified post-hoc
analysis places the dump arm inside the autonomous repeat at trajectory and map
boundaries; its minimum 16-map FSC-AUC is `0.999999954237`.  This validates the
payloads without falsely claiming independent GPU processes are bitwise inert.

The real-data K=1 10,000-particle full trajectory in job `11235095` provides a
separate convergence-scale result.  RELION and RECOVAR both converge at
numbered iteration 16 on the same A100-SXM4-80GB.  Strict array gates first
fail at iteration 2: Pmax absolute error has p95 `3.527e-4` and maximum
`0.02159`.  Direct map FSC-AUC first falls below `0.995` at iteration 7, with
half1, half2, and merged values `0.992872475`, `0.993256130`, and
`0.994349457`.  The lowest half-map value is `0.980633266` at iteration 11;
the numbered iteration-16 maps recover to `0.99481158`, `0.99565323`, and
`0.99597422`.  Matching termination therefore does not make the intervening
trajectory a parity pass.

That job also exposed a real final-boundary off-by-one defect.  After numbered
iteration 16, RECOVAR requested the RELION `run_it016` state but fell back to
`run_it015`; RELION's final all-data expectation consumes the state written by
numbered iteration 16.  The contaminated final output is explicitly invalid
and is not used as quality evidence.  Commit `999278bd` allocates and loads the
extra replay slot so numbered expectations 1--N consume `run_it000` through
`run_it{N-1}`, while the converged final all-data expectation consumes
`run_itN`.  The focused caller regression has 58 passing CPU tests.  A fresh
same-A100 run must confirm the corrected final boundary.  The immutable
before-fix science root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_fulltraj_f10c0386_a10080_retry5_prepared_20260715_181156/`;
the fail-closed post-hoc audit is under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/k1_real10076_job11235095_posthoc_hardened_20260715_191601/`.

## 2026-07-15 Corrected K=1 converged-state replay

Fresh same-A100 job `11238154` confirms the final-boundary correction.  A new
autonomous RELION run and RECOVAR both produce 18 numbered iterations and
converge at iteration 18.  RECOVAR's final all-data branch consumes the exact
`run_it018` state with no fallback, uses sampling iteration 19, and runs with
the strict diagnostic gridding correction enabled.  All expected numbered
STAR files and half maps exist through iteration 18.  The versioned
`em_k1_corrected_final_state_gate_v1` control report passes without exception.
This closes the off-by-one state-loading defect from job `11235095`.

It does not close K=1 trajectory parity.  The fail-closed continuous audit
first rejects iteration 2 Pmax: absolute error has p95
`0.0003533333164453489` and maximum `0.021642681756019605`.  Map FSC-AUC first
falls below `0.995` at iteration 7, with half1, half2, and merged values
`0.993608422474`, `0.993329608226`, and `0.994727977169`.  The lowest half-map
FSC-AUC is `0.977435296241` at iteration 13.  Numbered iteration 18 recovers to
`0.994551451225`, `0.994562872376`, and `0.995379792244`, while the corrected
final all-data merged FSC-AUC is only `0.984411252872`.  Final Pmax absolute
error has p95 `0.0911828649503` and maximum `0.562462863297`.  These values are
FSC/FSC-AUC and exact-array evidence; correlation is not a quality gate.

The Slurm state is `FAILED` with exit code 2 only because the completed science
correctly failed its strict parity classifier.  The immutable run root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_fulltraj_999278bd_a10080_correctedstate_prepared_20260715_192500/`.
Its principal records are `analysis/corrected_final_state_gate.json`,
`analysis/trajectory_gate.json`, and
`analysis/trajectory_classification.json` under that root.  The next K=1
causal target remains the frozen iteration-2 posterior boundary, not the now
validated final-state router.

## 2026-07-15 Same-GPU RELION trajectory repeat envelope

Job `11239471` runs two independent autonomous RELION refinements sequentially
on the same physical A100
`GPU-bd720f2f-c28a-09c0-d51e-d08b1897125a`.  Both arms have the same numbered
schedule, both converge after 16 numbered iterations, and their trajectory
controls are equal.  Runtime is `1459.57` versus `1466.42` seconds.  This is the
required control for separating deterministic cross-engine defects from
nonlinear amplification of RELION's own GPU reductions.

The repeat is not a near-bitwise full-trajectory null.  Iteration-2 Pmax
absolute differences have mean `5.9033e-6`, p95 `2.9e-5`, and maximum
`0.00146`.  By iteration 8, half1 map FSC-AUC is `0.993115371097`; the lowest
numbered half-map FSC-AUC is `0.979059637119` at iteration 11.  Final merged
FSC-AUC is `0.967954843425`, and the minimum across the 37 compared maps is
`0.954744985471` for the final unfiltered half1 map.  Thus late trajectory map
differences, even below the nominal `0.995` point gate, need adjudication
against the same-GPU RELION repeat envelope.  This does not turn them into a
parity pass; it prevents attributing RELION's own nonlinear repeat variation to
RECOVAR without earlier continuous-array evidence.

The two leading K=1 iteration-2 cross-engine targets remain far outside that
envelope.  RELION repeat Pmax differs by only `5.2e-5` for fixture row 6202
(stack image 80654), versus RECOVAR/RELION `0.0216427`, about 416 times larger.
RELION repeat Pmax is identical for fixture row 7881 (stack image 103528),
versus RECOVAR/RELION about `0.02042`.  These targets therefore require the
frozen candidate/operand comparison; they are not classified as reduction
noise.  The immutable repeat root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_relion_repeat_envelope_a10080_retry4_prepared_20260715_195515/`,
with the versioned report at `analysis/repeat_envelope.json`.

## 2026-07-15 K=1 iteration-2 exact-reference boundary

Job `11240978` completed all four RECOVAR science arms and failed only in its
post-science analyzer import.  The recovered versioned report passes its
instrumentation gate.  With exact RELION iteration-1 incoming half maps and
state, six of seven sampled iteration-2 Pmax differences contract to between
approximately `4e-6` and `9.4e-5`.  Fixture row 7881 (stack image 103528) is
the isolated exception: RECOVAR Pmax is `0.2810158432` versus RELION
`0.260604`, a residual `+0.0204118432`.

The exact-reference dump/no-dump control has maximum Pmax change `1.389e-5`,
zero Euler or translation mismatches, one significant-count difference by
one, and half-map FSC-AUC `0.9999999860/0.9999999862`.  Row 7881 has the same
1,536 fine candidates and reconstruction support in the autonomous-map and
exact-map RECOVAR arms.  Canonical replay of the captured RECOVAR operands has
maximum float32-versus-float64 score change `1.406e-5`, float64-versus-
complex128 change about `5.68e-14`, and zero order-only change in this replay.
Those effects are far too small to classify the `0.02041` Pmax residual as
ordinary reduction precision or order variation.

RELION records 49 significant coarse samples for row 7881, while RECOVAR
records 48 and expands them to exactly `48 * 32 = 1536` fine children.  One
additional RELION parent with full-posterior mass `0.0726359` would explain the
entire Pmax residual by support renormalization.  This is a testable leading
hypothesis, not yet a conclusion: passive RELION pass-0/pass-1 arrays and a
full RECOVAR coarse surface must identify the parent and its mass before any
production change.  Row 6202 is the matched control at 15 samples in both
programs.  The report and replay artifact are under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k1_it2_current_exactref_capture_bdc50_20260715_194900/analysis_rows/`.

## 2026-07-15 K=4 particle-5993 frozen-boundary provenance correction

Job `11241611` completed the corrected old-map float32 science arm and then
failed its intentionally strict `144`-parent gate.  An independent exact-set
audit proves that count was overstrict: the current 143-parent support is the
material old-low branch.  It retains `(class 0, global coarse rotation 255,
translation 20)` and differs from the historical 144-parent old-low set only
by excluding `(class 0, rotation 14, translation 14)`.  The excluded parent's
32 fine children had total posterior mass `6.56684e-14` and zero reconstruction
support.

Current versus historical old-low Pmax is `0.6777920723` versus
`0.6777926087`, while class, Euler pose, and translation are exact.  The
material retained parent and negligible excluded parent are ranks 143 and
144, with weights `1.8615471406e-5` and `1.8615400394e-5`; their difference is
only `7.10123e-11`.  Fixture data, particle order, half layout, dispatch,
iteration state, and all eight scoring maps match, including an exact map-frame
round trip.  The archived run did not save the coarse operands or reduction
trace and used a different H100/CUDA binary, so exact reproduction of the
negligible 144th parent is neither possible nor a valid branch-identity gate.
The audit is at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_p5993_archive_provenance_audit_20260715_210100/`.

## 2026-07-15 K=4 frozen cutoff precision classification

Same-H100 job `11242446` completed four frozen particle-5993 arms: old-low and
new-high incoming maps, each with production float32 and genuine upstream
float64 scoring.  The two float32 arms reproduce the nonlinear branches while
keeping exact class, Euler pose, and translation.  Both retain 143 coarse
parents, but ranks 143 and 144 swap.  Old-low retains material parent
`(class 0, rotation 255, translation 20)` and gives Pmax `0.6777920723`;
new-high retains negligible parent `(0, 14, 14)` instead and gives Pmax
`0.7309843898`.

The source-matched CUDA replay applies device float32 max, `expf`, positive
filtering, CUB radix sort, and CUB inclusive scan.  It is bit-exact with each
recorded float32 mask, including the parent swap, so the threshold backend is
not the cause.  Genuine float64 scoring changes all 66,816 scores in each arm.
It leaves old-low unchanged but flips exactly the two cutoff parents in
new-high, restoring the material parent while retaining a 143-parent count.
The float32-versus-float64 maximum score differences are `9.93e-5` and
`1.09e-4` for old-low and new-high.  This classifies the branch as score-
generation precision sensitivity around a near tie, not a stable algorithmic
support mismatch.  The immutable evidence root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_p5993_frozen_pass1_material_continuation_d67ec0a3_h100_prepared_20260715_210300/`.

## 2026-07-15 K=1 current-engine coarse cutoff capture

Job `11242768` passively captures the complete iteration-2 coarse surfaces for
the failing row 7881 and control row 6202 while using exact RELION incoming
maps.  Instrumentation is inert: Euler and translation arrays are exact,
maximum Pmax change is `1.1683e-5`, and half-map FSC-AUC is
`0.9999999864/0.9999999864`.  RECOVAR selects 48 parents for row 7881; its
first excluded parent is `(rotation 30039, translation 2)` with coarse
posterior `0.00011504446`.  The selected 48 parents expand exactly to the
observed 1,536 fine children.  Row 6202 selects 15 parents as expected.

Independent A100 CUDA 12.6 job `11243285` replays RELION's source sequence on
the captured RECOVAR float32 scores.  Its masks are bit-exact with RECOVAR: 48
and 15 parents, respectively.  CUB/JAX cutoff ordering alone therefore cannot
explain RELION's observed 49-parent row 7881.  Passive RELION candidate and
operand arrays are still required before assigning the earliest difference to
score generation, priors, or candidate geometry.  The current-engine capture
is under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k1_it2_coarse_significance_399de551_a100_20260715_211300/`,
and the CUDA replay is under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k1_it2_coarse_cub_replay_a100_cuda126_20260715_212900/`.

## 2026-07-15 K=1 iteration-2 passive cross-engine closure

Same-A100 job `11242107` passively captured RELION's complete coarse and fine
surfaces for rows 6202 and 7881.  Its clean, patched-no-dump, and two dump arms
pass the repeat-calibrated inertness gate; every half-map FSC-AUC is above
`0.9999999983`.  The score transform is independently source- and
array-verified: RELION's `raw_diff2` already includes the half-chi-squared
factor, so the centered pre-prior score is `-raw_diff2` with factor one.  The
captured row-6202 coarse log weights reconstruct bit-exactly for all 1,053,744
finite candidates.

Geometry-based alignment proves exact row-6202 supports: 15 coarse parents,
480 fine candidates, 37 reconstruction candidates, and the same winner.  For
row 7881, RELION has exactly one additional coarse parent, mapped to RECOVAR
key `(class 0, rotation 30039, translation 2)`.  It is rank 49 in both engines
and has posterior `0.00011509424` in RELION versus `0.00011504446` in RECOVAR.
The cumulative mass after rank 48 is `0.99899966069` in RELION and
`0.99900027498` in RECOVAR, so a difference of only `6.14e-7` straddles the
`0.999` cutoff without changing rank order.

The extra parent generates exactly 32 RELION-only fine children with posterior
mass `0.0726669963`.  Removing those children and renormalizing predicts Pmax
`0.2810506426`, versus observed RECOVAR `0.2810158547`; the residual is
`3.48e-5`, and the winning fine hypothesis is exact.  This proves that the
visible `0.0204` Pmax anomaly is cutoff amplification of the coarse posterior
boundary, not a fine-expansion, sort, or winner-selection bug.  It does not yet
classify the underlying score-array residual as numerical or algorithmic.

Genuine float64 score recomputation in job `11243489` leaves row 7881 at 48
parents with the same rank order and boundary identities; its cumulative mass
after rank 48 remains above the cutoff at `0.99900018764`.  Thus neither
RECOVAR's float32 score arithmetic nor its CUB cutoff reduction explains the
cross-engine side of the boundary.  Float64 projection-plus-score job
`11244035` also leaves the row at 48 parents with top-48 cumulative mass
`0.99900018756`.  Relative to float64 scoring alone, genuine float64
projections change centered row-7881 scores by at most `2.39e-6`, posterior L1
by `1.75e-7`, and no support or rank identity.  The job's half-map FSC-AUC is
`0.9999999875/0.9999999877` against the score-only arm.  Ordinary scoring and
projection precision are therefore ruled out for RELION's 49-versus-48
boundary.  The remaining work is exact per-pixel operand and operation-
semantics comparison; no production cutoff change is justified by this
evidence.

Manual supplied-PPref diagnostic job `11244670` does produce 49 row-7881
parents, but only through a compensating error and is rejected as a fix.  Its
selected-union centered-score RMS error against RELION is `0.009148`, versus
`0.002103` for texture, and its coarse Pmax is farther from RELION.  On the
matched row-6202 control, manual RMS error is `0.003159`, versus `0.0001623`
for texture, about 19.5 times worse.  Manual-versus-texture half-map FSC-AUC is
`0.9999689/0.9998694`; exact intermediate arrays, not the accidental support
match, retain texture as the strict coarse projector.

The sealed passive audit is under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k1_it2_cross_engine_alignment_audit_20260715_211308/`.
The float64-scoring audit is under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k1_it2_coarse_float64_399de551_a100_20260715_213200/`.
The float64 projection-plus-scoring audit is under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k1_it2_coarse_float64_scoreproj_399de551_a100_20260715_214600/`.
The rejected manual-projector diagnostic is under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k1_it2_manual_ppref_f32_399de551_a100_20260715_220033/`.
RELION's dumped `ihidden_overs` field is corrupt and non-unique in this hook;
the audit rejects it and uses the unique device tuple plus exact rotation and
translation geometry instead.

## 2026-07-15 K=1 trajectory repeat-envelope classification

The versioned post-hoc trajectory classifier compares the corrected RECOVAR-
versus-RELION run against the same-physical-A100 RELION repeat pair.  Exact
current size, Healpix order, and perturbation schedules align only through
iteration 7; RELION itself changes to a different Healpix schedule at
iteration 8.  Iterations 8 and later are therefore observational nonlinear
amplification, not eligible for a new causal parity classification.

Within the aligned prefix, iteration 1 already exceeds the repeat map envelope:
half1 FSC-AUC loss is `1.49246e-6` cross-engine versus `1.81529e-10` in the
RELION repeat, while Pmax remains exact.  Iteration 2 is the earliest
continuous Pmax exceedance and the strongest early boundary: cross-engine
mean/p95/maximum absolute errors are
`1.15158e-4/3.53333e-4/2.16427e-2`, versus
`5.90331e-6/2.9e-5/1.46e-3` in the RELION repeat.  Rare discrete pose and
translation outputs differ at iteration 1 but are not used alone as causal
evidence.  The classifier fails closed rather than treating the substantial
late RELION repeat variation as either parity or a RECOVAR defect.

The deterministic report, script, audit, and verified manifests are under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k1_fulltraj_repeat_envelope_classifier_v1_20260715_221500/`.

## 2026-07-16 K=1 exact-rotation trajectory and next boundary

Commit `d302a760` restores exact convergence parity on the canonical 10k
fixture: RECOVAR and RELION both converge at numbered iteration 16, and all 48
numbered half1/half2/merged FSC-AUC comparisons improve over the preceding
trajectory.  This is a major correction but not yet a quality pass: grid-off
final merged FSC-AUC is `0.989787314`, and the strict numbered-map gate is
missed during iterations 9--13.

A same-A100 exact-reference counterfactual now localizes the earliest remaining
iteration-2 Pmax/support differences to the iteration-1 map.  For particle 257,
a posterior cumulative-mass movement of only `1.73e-7` across the `0.999`
cutoff changes 15 to 16 significant parents without changing the winner;
particle 8240 also keeps its winner and support count.  The iteration-1 raw
BPref weight arrays are already within about `9.1e-7` relative L2 of RELION.
Data relative L2 is `1.26e-4/1.08e-3` for halves 1/2, and the larger half-2
residual coincides with the sole material iteration-1 pose-output difference:
particle 8494 selects an adjacent translation displaced by 0.5 pixels.

The immediate order is therefore: adjudicate particle 8494 from complete
candidate arrays; then use the existing canonical contribution replay on the
remaining continuous BPref residual.  Escalate to genuine float64/complex128
operand recomputation only if original/canonical float32 and promoted-float64
replays remain ambiguous.  Do not change the production cutoff or accept a
discrete winner merely to match RELION.  Intermediate gates use exact/array
metrics and map gates use FSC/FSC-AUC, never correlation.

Evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k1_exact_rotation_fulltraj_d302a760_20260716_012500/`
and
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k1_it2_residual_dualarm_d302a760_20260716_024500/analysis/`.

## 2026-07-16 K=1 recurrent score and BPref boundary classification

Particle 8494's iteration-1 translation difference is a one-float32-ULP
firstiter-CC near tie.  Replacing only that translation removes `88.4%` of the
anomalous half-2 raw-BPref data residual, reducing relative L2 from
`1.08160e-3` to `1.25343e-4`.  The production device contribution capture is
internally valid; commit `08fb0fa0` fixes the diagnostic routing that originally
prevented this capture on the K=1 firstiter path.

The apparent residual against the C++ single-particle panel is not a production
RECOVAR bug.  RELION GPU/RECOVAR float32 geometry includes one radius-boundary
pixel (`r^2=2303.999759`), while the double-precision C++ diagnostic excludes
it (`r^2=2304.000138`).  Removing that pixel reduces panel data relative L2
from `1.35819e-2` to `6.21507e-6`.  Genuine downstream float64/complex128
recomputation from the captured raw float32 image changes the unmasked BPref
operand by only `4.39994e-7`.  The subsequent RELION CUDA `storeWeightedSums`
capture finds exact support, coordinates, indices, and Hermitian flags, plus
float32-scale data/weight operands (`3.61e-7`/`3.92e-7` relative L2).  The real
difference is interpolation coefficients: RECOVAR relative L2 `4.7538e-6`
versus RELION's `3.5699e-8` canonical envelope.  RECOVAR added the integer
origin before taking the fraction; RELION takes `floorf` and the fraction
first.  Commit `65587ea5` fixes that order, and the captured p8494 replay is
then bitwise exact for all 897 support pixels, coordinates, eight indices, and
coefficients.

The paired particle-1491 capture freezes the earliest recurrent iteration-2
boundary.  All 36,336 RELION coarse rotations and all translations align
exactly, and both programs have the same winner and 30 parent rotations.
RELION's metadata threshold rank is 173, but its `>=` tie expansion evaluates
174 coarse hypotheses; RECOVAR evaluates 173.  The sole RELION-only pair
(`rotation 35017`, translation 5) creates 32 fine descendants carrying
`0.004659639` posterior mass.  This support amplification explains the visible
Pmax error: all-support posterior L1 is `9.31920e-3`, whereas independently
renormalized shared-support L1 is `2.48768e-4`, relative L2 is `1.88068e-4`,
and Pmax error is `5.79469e-5`.

The source-path precision controls do not repair that boundary.  Genuine
float64 score arithmetic on the same physical A100 leaves the 173-hypothesis
support unchanged and reduces centered coarse-score RMS against RELION only
from `1.68293e-4` to `1.64606e-4`.  A preliminary complex128-projector arm also
leaves particle 1491 unchanged and is much farther from RELION; its aggregate
effects require a same-physical-GPU repeat before use.  The exact fine
256-lane replay is likewise not the first cause: on restored RELION support its
posterior L1 is `3.01462e-4` and Pmax error is `9.87415e-5`, slightly worse than
production on independently renormalized shared support.  Candidate
`7ad2526d` therefore remains unmerged; its prior trajectory also showed no
consistent FSC-AUC gain and cost `4.58%` wall time.

The subsequent exact coarse-operand capture classifies this particular tie as
numerical rather than a formulation mismatch.  Candidate geometry is exact,
and cross-program reference, weight, and shifted-image operands are all below
`7.11e-7` relative L2.  The combined production candidate residual is
`-3.62396e-4`; recomputing from the captured operands in float64 reduces it to
`+7.663e-6` (direct) or `+4.456e-6` (decomposed).  The matched-prior float32
reduction envelope `[-2.3079e-4,+1.3542e-4]` spans the support tie.  This closes
particle 1491 without a cutoff change, but does not establish global parity:
the diagnostic hook was not globally inert for every particle and its exact
continuation arm correctly failed closed when sampling geometry changed.

Across 10,000 particles the remaining iteration-2 Pmax error has no signed bias
or isolated subgroup.  Continue with aggregate boundary substitutions and the
RELION CUDA unmasked BPref operand/scatter classification.  Then rerun complete
numbered/final FSC and FSC-AUC trajectories, the K=1
robustness/scale/real-data gates, and K=4.  Do
not change cutoff semantics to force this one tie and do not use correlation as
a map-quality gate.

Evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k1_it2_p1491_paired_a92c35ef_20260716_081502/analysis/p1491_coarse_boundary.json`,
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k1_it2_p1491_paired_a92c35ef_20260716_081502/analysis/aggregate_it2_pmax_support_distribution.json`,
and
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k1_it1_p8494_device_capture_fix_fb4e6b73_20260716_082324/analysis/continuous_residual_localization/report.json`.
The exact coarse replay seal is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/p1491_coarse_operand_replay_20260716_094000/analysis/FINAL_SEAL.json`
(SHA-256 `a4e559c3bc5a2f378d9f2af37ddb2e5348cf630afe529c2a0e736b742d37b274`).

## 2026-07-16 K=1 eight-case robustness trajectory gate

The current eight-case 3k/128 matrix passes the complete numbered FSC audit.
RECOVAR and RELION stop after the same numbered iteration in every case
(`9--15` iterations).  Across all numbered maps, merged cross-engine FSC-AUC
is at least `0.999838371`, and the minimum RECOVAR-minus-RELION merged GT
FSC-AUC is `-0.000279612`.  This covers uniform, anisotropic and Kent angular
distributions, contrast/noise-scale variation, image offsets, no-CTF radial
noise, and two outlier regimes.

With the GUI-quality grid correction intentionally unset/off, final merged
cross-engine FSC-AUC spans `0.997233874--0.998704958`.  RECOVAR final merged GT
FSC-AUC is higher in all eight cases by `+0.007711695--+0.020127071`.  Keep
this final-output policy separate from numbered-map algorithm parity; do not
enable grid correction to improve the cross-engine number.

Quality is closed for this small matrix, but speed is not: RECOVAR takes
`1.41--2.47x` RELION wall time.  The severe-outlier case is `2.18x`; its
iteration 2 alone takes `1285.9` seconds after the posterior expands to about
`100M` hypotheses per half, then later iterations return to tens of seconds.
Treat that support explosion as the first K=1 performance target after the
RELION CUDA BPref coefficient-order fix is validated.

Evidence root and jobs:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_robust_current_65d2c3f1_20260716_091500/`
(`11264619--11264626`, summary `11264627`, FSC audit `11265312`).

## Aggregate particle-state diagnostic

Use `scripts/audit_em_particle_state_distribution.py` after full runs to align
particles by exact `rlnImageName` and compare Pmax, significant-support, pose,
translation, and K-class distributions across every available numbered
iteration.  When a same-physical-GPU RELION repeat is supplied, the report also
measures RECOVAR errors relative to that numerical control envelope.  An
independent repeat pair can be supplied with `--relion-control-reference-star`
and `--relion-control-star`, so the control envelope need not share the
cross-engine reference arm.  K-class agreement is Hungarian-matched once per
iteration and that fixed mapping is used for every subgroup, while retaining
raw label agreement and the full
confusion matrix.  Intermediate gates use these exact/array metrics; map-quality
gates remain FSC/FSC-AUC only.  Escalate from this aggregate report to a
particle capture only when it identifies a systematic cohort or an FSC
trajectory localizes a reproducible boundary.  Use repeated
`--recovar-iteration` arguments when only an explicit boundary subset has all
required state arrays; the default remains fail-closed rather than silently
omitting missing support.  Without that explicit selection, a RELION particle
STAR is required for every numbered RECOVAR iteration; omitted middle or
trailing iterations are errors.

## 2026-07-16 K=4 strict Class3D trajectory closure

The corrected Class3D auditor now follows the actual RELION topology: three
one-based full-map Class3D iterations are compared with the arithmetic means
of the corresponding zero-based RECOVAR half maps. For these capped,
nonconverged runs, the last numbered RELION maps are the final reference and
all RECOVAR `final_classNNN` products must exactly equal the last numbered
half averages. The auditor rejects those semantics if final all-data ran.

Both 10k/128 strict cases pass every per-class shellwise FSC/FSC-AUC and GT
gate. The white/uniform case has minimum numbered direct FSC-AUC
`0.9965686801` and minimum RECOVAR-minus-RELION GT FSC-AUC delta
`-6.1969e-6`. The radial/nonuniform case has minimum numbered direct FSC-AUC
`0.9999188078` and minimum GT delta `-3.5824e-5`. All final RECOVAR class
products exactly match their last numbered half averages. Map correlation
was not computed.

The sole white-case iteration-1 class/pose outlier, particle 9056, is a
classified float32 tie rather than an algorithmic mismatch. All 66,816
candidate identities align, rotations are exact, translations differ by at
most `1.516e-7` pixels, and priors are uniform. RELION's top two coarse
scores are bitwise equal; RECOVAR has the same top-two candidate set split by
`1.788139343e-7`, exactly three float32 ULPs and only `0.103x` the full
centered cross-engine residual envelope. The different fine supports are
entirely downstream of that coarse tie. Same-engine map repeat FSC-AUC is
effectively one, and cross-engine per-class FSC-AUC spans
`0.99991548--0.99999998`; float64 replay is unnecessary because the float32
evidence already resolves the classification.

Canonical evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_strict_dispatchv2_d3b0d78d_20260716_113000/k4_trajectory_matrix_summary.json`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_p9056_it1_capture_d3b0d78d_a100_20260716_132500/analysis/p9056_adjudication_v1.json`

## 2026-07-16 K=1 exact-BPref robustness closure and performance boundary

The current exact-BPref eight-case 3k/128 matrix at source `fc70abc3` passes
the canonical full-trajectory audit in Slurm job `11269304`. Across all
numbered boundaries, minimum merged cross-engine FSC-AUC is `0.9997416604`,
minimum half-map cross-engine FSC-AUC is `0.9996432483`, and minimum merged
RECOVAR-minus-RELION GT FSC-AUC is `-0.0003181674`. Minimum final merged
cross-engine FSC-AUC is `0.9975110649`; every final merged GT delta is
positive, with a minimum of `+0.0082881294`. All eight cases pass, including
the severe 50%-outlier/radial-noise case. Correlation was not used or
computed by the acceptance summary.

Quality and performance are separate at this checkpoint. In the severe case,
RELION takes `959` seconds and RECOVAR takes `4250` seconds. RECOVAR iteration
2 alone takes `3421.4` seconds after support expands to roughly 100 million
hypotheses per half; later iterations return to tens of seconds. Matched
large-support buckets are about `3x` slower per image than the preceding
algebraic-score checkpoint even though the current support is slightly
smaller and GPU utilization is higher. The exact direct CUDA-order Gaussian
path computes a global-min raw-diff2 prepass and then recomputes exact diff2
for downstream conversion and M-step work. A bounded raw-diff2 reuse path is
therefore the next performance target, with mandatory exact intermediate and
FSC/FSC-AUC non-regression gates. Same-GPU exact/algebraic/exact job
`11270918` failed closed before its third arm because a shared CUDA library
changed after the native-library manifest. Its partial timings are
inadmissible and are not required for the accepted quality result.

Canonical evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_exact_bpref_fc70abc3_20260716_111000/trajectory_matrix_fsc_only_summary_v2.json`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_exact_bpref_fc70abc3_20260716_111000/cases/22_small_severe_outliers_3k_g128_radial_noise5_bf80/trajectory_analysis/k1_case22_iter2_performance_diagnosis_v1.json`

## 2026-07-16 real-data repeat-control adjudication

The completed two-arm 10k-particle real-10076 replay shows a systematic early
cross-program difference that approaches the observed RELION-repeat scale
late.  Across the common iteration-1--16 range, the worse of the two
RECOVAR-versus-corresponding-RELION FSC-AUC deficits exceeds the single
RELION-A/B deficit in 44 of 48 half1/half2/merged comparisons.  The four
within-control comparisons are half 1 at iterations 14--16 and merged at
iteration 16.  This comparison is deliberately control-normalized rather
than using a fixed correlation threshold.  Iteration-1 absolute FSC-AUC is
still at least `0.999999954`, so a large deficit ratio at a nearly exact
boundary is not by itself evidence of an algorithmic bug.

The earliest discrete difference, iteration-1 particle 8494, is closed as
`coarse_float32_one_ulp_tie_changes_fine_support`: RELION's top coarse gap is
exactly one float32 ULP (`2.9802322388e-8`), and substituting RELION's support
makes RECOVAR select the same fine winner.  By contrast, the earliest
non-ordinary boundary is iteration-2 particle 8240.  RELION and RECOVAR swap
one hypothesis at the coarse `0.999` support boundary, producing 32 different
fine descendants.  RELION-only descendants carry `0.0777750465` posterior
mass, which explains the Pmax change from `0.174813433` to `0.189518494`.
Same-physical-GPU complex128/float64 source scoring preserves RECOVAR's support,
so ordinary float32 reduction precision is not the cause.  Geometry, priors,
and fine reduction are also ruled out; the unresolved split is upstream
operand generation versus coarse score formulation.

At iteration 13 the worst-arm deficits are `1.264x`, `3.203x`, and `1.786x`
the observed repeat deficit for half 1, half 2, and merged maps.  At iteration
16 they are `0.658x`, `1.404x`, and `0.845x`, respectively.  Iteration 17 is
uncalibrated because one RELION control terminated at iteration 16.  Final
maps are telemetry only because the controls terminated at different
iterations and performed separate final reconstructions: RELION A/B final
FSC-AUC is `0.946113`, while RECOVAR A/B is `0.948730`.  The control remains
one empirical same-model repeat scale, not a confidence interval.

Authoritative evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_dual_replay_2e40e614_20260716_131000/analysis/real10076_completed_dual_repeat_envelope_v1.json`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_dual_replay_2e40e614_20260716_131000/analysis/audit_artifacts.sha256`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it2_p8240_capture_505af690_20260716_124319/analysis/p8240_boundary.json`

## 2026-07-16 authoritative K=4 incoming-reference substitutions

Hardened same-A100 A/B substitutions isolate the visible K=4 failures to the
incoming reconstructed references.  In case 8, replacing only the incoming
iteration-4 references raises iteration-5 minority-class-2 direct FSC-AUC
from `0.978234446` to `0.999999996`; the substituted arm's minimum over all
classes is `0.999999946`, class agreement is `1.0`, and significant-support
differences fall from 16 to zero.  Job `11273615` recorded the same physical
UUID `GPU-a1de512c-f178-a5e1-6c95-c54c6d07c9f3` at all six boundaries.

In case 2, replaying the previous RELION reference closes iteration 2--5:
iteration-5 minimum direct FSC-AUC is `0.999999974`, class agreement is
`1.0`, and the minimum RECOVAR-minus-RELION GT FSC-AUC delta is
`-6.57e-7`.  The production arm's earliest configured failure is iteration 4
at direct FSC-AUC `0.991907530`; substitution raises it to `0.999999909` and
contracts angle p99 from `26.99` degrees to float32 scale.  Job `11273364`
used one validated A100 UUID for both arms.

These controls classify the later E-step/pose/posterior machinery as capable
of parity when supplied RELION's preceding maps.  They demonstrate strong
incoming-reference sensitivity, but do not by themselves prove that the
earlier RECOVAR reconstructed-reference boundary is defective.  A separate
case-8 complex128/float64 M-step perturbation makes the minority cliff much worse
(`0.658960344` at iteration 5 versus `0.978235509` in production), proving
strong numerical sensitivity but not cross-program float64 closure.  Map
acceptance uses shell FSC/FSC-AUC only.

Evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_case8_it5_relref_ab_uuidfix_03c0969b_20260716_131422/analysis/ab_summary.json`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_case2_relref_ab_03c0969b_hardened_20260716_131700/analysis/ab_summary.json`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_case8_mstep64_ab_03c0969b_hardened_20260716_131600/analysis/ab_summary.json`

## 2026-07-16 K=4 stock-RELION repeat calibration

Hardened same-A100 stock RELION/RELION repeats show that the late case-2 and
case-8 Class3D trajectories are intrinsically unstable.  Case 8 is effectively
identical through iteration 2, then its matched-class repeat FSC-AUC minimum
falls to `0.756448814` at iteration 3, `0.377412354` at iteration 4, and
`0.223965129` at iteration 5.  Assignment agreement falls to `0.8880` and the
iteration-5 angle-error p99 is `170.55` degrees.  The original RECOVAR/RELION
minority-class value `0.978234446` is therefore well inside the much larger
native RELION repeat envelope.

Case 2 behaves similarly.  Its repeat minimum is `0.9999999993` at iteration 1
and `0.9999999968` at iteration 2, but falls to `0.900718820` at iteration 3,
`0.893285765` at iteration 4, and `0.863440629` at iteration 5.  Iteration-4
assignment agreement is `0.9019`, compared with RECOVAR/RELION agreement
`0.9914`; RECOVAR/RELION direct FSC-AUC `0.991907530` is much closer than any
of the four matched native-repeat class values (`0.893286--0.904928`).

Both repeats preserve exact dispatch non-rank columns, but runtime follower
ownership changes for `25,468/50,000` case-8 particle-iterations and
`19,712/50,000` case-2 particle-iterations.  That rank/reduction-order seed is
small through iteration 2 and then class dynamics amplify it.  Consequently,
late reference-substitution closure is evidence of trajectory sensitivity,
not an actionable RECOVAR algorithm bug unless a cross-engine difference
exceeds the case-specific native repeat envelope.  Early stable boundaries,
full distribution arrays, GT FSC/FSC-AUC, convergence, and case-specific
repeat controls remain required; bitwise class decisions are not a gate.

Authoritative evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_case8_relion_repeat_full5_uuidfix_20260716_141807/analysis/relion_repeat_full5.json`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_case2_relion_repeat_full5_uuidfix_20260716_144006/analysis/relion_repeat_full5.json`

## 2026-07-16 K=1 local significant-count semantics

The first full K=1 serialization audit exposed a real metadata boundary bug.
RELION writes `_rlnNrOfSignificantSamples` from the first/coarse pass, as
documented and implemented by `my_nr_significant_coarse_samples` in
`acc_ml_optimiser_impl.h`.  RECOVAR instead returned the fine-pass M-step
support count after local search began.  Counts were exact or differed for at
most one particle through numbered iteration 3, then diverged for 2,691/3,000
particles at iteration 4; the mean absolute error grew from `3.263` at
iteration 4 to `75.834` at iteration 10.

The parent-pass correction closes the serialization bug in full-trajectory
job `11275201`: iterations 1, 3, 6, and 7 match all 3,000 counts exactly.  The
other six iterations contain only 16 particle-iteration residuals in total,
each exactly one count.  Iteration 4 improves from 2,691 mismatches to 2;
iteration-10 mean absolute error improves from `75.8343` to `0.002333`.
Across 23 old/new maps on different A100s, minimum FSC-AUC is
`0.999999985554`; this is an envelope, not a bitwise claim.  The remaining
sparse boundaries are upstream coarse-support differences.

The implementation is further hardened to derive serialized counts directly
from the explicit retained coarse-index lists before any pass-2 diagnostic
expansion.  It does
not request or expose the engine's fine reconstruction-support count, and a
non-adaptive local run reports the RELION-compatible count as unavailable
rather than substituting a different semantic quantity.  Fine support still
controls every M-step reconstruction, posterior statistic, noise term, and
accumulator.  Targeted guards pass, including valid local-GPU composite
coverage of 363 tests.  The full trajectory above validates the corrected
coarse-pass semantic boundary; a replay of this exact extraction refinement
is retained as a final runtime guard.

Authoritative artifacts:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_sigcount_fulltraj_52178ed3_20260716_135924/analysis/REPORT_v2.md`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_sigcount_fulltraj_52178ed3_20260716_135924/analysis/validation_v2.json`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_sigcount_fulltraj_52178ed3_20260716_135924/analysis/residual_coarse_count_particles_v2.tsv`

The `SHA256SUMS_v2` manifest SHA-256 is
`190ace970540dd6ad2eb7a35627a2d8e66345fba1f0a3cf536df0c77ba3ee803`.

## 2026-07-16 K=4 heterogeneous robustness expansion

The authoritative four-case, exact-UUID robustness gate passes three cases
and fails one at source `b5dd574a`.  The radial-noise/nonuniform/outlier Ribo
case 10 passes in job `11274945`: its iteration-1--3 minimum direct FSC-AUC is
`0.999634051`, `0.999666644`, and `0.999446341`, minimum class agreement is
`0.9956`, and the worst RECOVAR-minus-RELION GT FSC-AUC delta is `-8.00e-5`.
The 30k-particle Tomotwin case 12 passes in job `11274946` with minimum direct
FSC-AUC `0.999820693`, `0.999157434`, and `0.996495854`, minimum assignment
agreement `0.9986`, and worst GT delta `-6.28e-5`.  Case 14 remains essentially
exact, with minimum direct FSC-AUC `0.999999959` and class agreement `1.0`.

Case 11 (IgG, white noise, uniform classes, 20% outliers) is the genuine
trajectory failure.  Iteration 1 is effectively exact (minimum direct FSC-AUC
`0.999330786`, one class mismatch), but iteration 2 already has 9,999/10,000
Pmax differences, 1,166 support differences, and 24 class mismatches.
Iteration 3 amplifies this into class-2 direct FSC-AUC `0.9735133506`, below
the `0.995` gate.  The GT delta remains small (`-8.363e-5`), but that does not
excuse the cross-engine trajectory failure.

A serial exact-incoming-reference A/B on physical A100 UUID
`GPU-6a3cea75-90ac-d3de-7c1a-a8158412a9f4` proves that the iteration-3 cliff
is inherited from the incoming iteration-2 map state.  Replacing only the
four RECOVAR iteration-2 maps used to score iteration 3 with their exact
RELION counterparts raises minimum direct iteration-3 FSC-AUC from
`0.973512763` to `0.999999632`, raises class agreement from `0.9964` to `1.0`,
and leaves the GT delta at `-1.02e-6`.  The two RECOVAR arms themselves differ
at iteration 3 by minimum direct FSC-AUC `0.973508472`, so the intervention is
material.  Pmax mean/p95/maximum absolute errors fall from
`0.00380994/0.0159218/0.5462` to `2.96e-5/1.33e-4/0.0100`, while particles with
different support sizes fall from 1,114 to 31.  This closes iteration 3 as an
intrinsic scoring/M-step cause; the next same-GPU A/B injects exact RELION
iteration-1 maps only for scoring iteration 2 to locate where the broad state
drift first appears.

The sole iteration-1 label mismatch is particle 7915.  Its RECOVAR class-2
score `0.5038749576` exceeds class 1 `0.5038748384` by exactly
`1.1920928955e-7`, one float32 ULP, at the same rotation and translation.
Treat this isolated discrete label as a numerical tie, not as the cause of the
distribution-wide iteration-2 boundary.

The Slurm allocation for the iteration-3 A/B is recorded as `FAILED 1:0` only
because its final hashing epilogue passed a generated `jobs/__pycache__`
directory to `sha256sum`.  Both science arms, UUID assertions, FSC audits,
particle-state audits, input-manifest stability check, and A/B summary had
already completed successfully; the posthoc seal records that limited
packaging failure.  Map acceptance uses shellwise FSC and FSC-AUC only.

Evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_robust_expand_uuid_recovery2_b5dd574a_20260716_135200/authoritative_combined_k4_robustness_summary.json` (SHA-256 `9f2efdfcb32ae64e77b8076fb71995025fb460c26b85b38128e64901b68b47a8`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_case11_it3_relref_ab_b5dd574a_20260716_141000/analysis/ab_summary.json` (SHA-256 `680ef82e336150058a0d37a3e8bfd77a99ff2232c3621b182c9af4ecb8bcc804`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_case11_it3_relref_ab_b5dd574a_20260716_141000/provenance/POSTHOC_SEAL.json` (SHA-256 `14fd581eabb63cd7e4af5ee664c9d740db35c31529862fe5eb3548f4d17cadc9`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_case11_p7915_rec_capture_b5dd574a_20260716_142900/analysis.json` (SHA-256 `59af9e01552dbb0d35a6277b232ee6594c0a70306b3f654d6acb86c8b2951d15`)
