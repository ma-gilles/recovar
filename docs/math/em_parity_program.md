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

## Current State — 2026-07-17

Authoritative clean candidate checkout:
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_em_parity_20260711/recovar`

Current integrated implementation checkpoint:
`dcd1aa07c54a087631f7bfd706439b64b20cdcfb`
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
- Real-10076 iteration-1 BPref factor replay identified a production local-EM
  weighted-sum precision defect. Full float32 products reduce the frozen
  RELION numerator gap from `2.07567e-4` to `3.39177e-7` (`611.97x`).
  Same-A100 control/fix maps have minimum FSC-AUC `0.9999999863`, and the fix
  slightly improves both half-map and merged RELION FSC-AUC. This closes the
  one-iteration causal boundary only; full real-data trajectory parity and a
  clean warmed performance comparison remain open.

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

Authoritative status on 2026-07-17:

- The same-A100 real-10076 K=1 run matches all 18 forced RELION numbered
  sampling/size boundaries, but its cross-engine map trajectory leaves the
  native repeat envelope at iteration 7. Forced scheduling prevents the
  autonomous iteration-8 collapse, yet forced merged FSC-AUC falls to
  `0.9745783` by iteration 16. This is a real accumulated state mismatch, not
  merely the discrete schedule branch.
- RECOVAR's autonomous trajectory advances HEALPix orders 4, 5, and 6 two
  iterations early and finalizes after numbered iteration 16. The historical
  `7f142d5f` post-cap interpretation was later invalidated by direct source
  audit: RELION has no convergence or sampling boundary after `iter > nr_iter`.
  Commit `607e4344` removes that synthetic boundary. The real-data termination
  difference remains an upstream accumulated map/schedule mismatch; do not
  treat the trajectory as accepted.
- The sealed iteration-2-to-3 operand decomposition is closed without a
  production candidate. It compares all 15 recurrent `>0.1`-degree tail rows
  with 15 deterministic matched controls under exact UID, support, geometry,
  and same-GPU control gates. Native common-prior operand TV is not enriched in
  the tail (median `8.9293e-5` tail versus `1.1321e-4` control; 7/15 paired
  tail values are larger). Canonical float64-from-captured-float32 TV is also
  similar (`4.0843e-6` versus `4.2158e-6`; 10/15 larger). No unit-aligned
  single- or two-field reference/image-weight/score-weight swap passes the
  pre-registered native-and-float64 movement and repeat-envelope gates.
  Classification is `unresolved_combined`; no production change or controlled
  substitution is authorized. Move to a compact full-10,000-particle
  score/posterior distribution diagnostic from a completely sealed
  uninterrupted RELION pre-iteration-3 boundary; do not resume serial
  particle tracing or pixel-operand capture. A restarted iteration-2 boundary
  is inadmissible because it changes the perturbation and can overwrite the
  half-2 follower noise state.
- The RECOVAR side of that diagnostic is implemented through `dcd1aa07`: an
  env-off-inert production score/posterior tap, bounded raw shards with strict
  readback/manifests and per-half identity closure, and an atomic captured
  RELION `Projector::data` replay contract. The full science launch remains
  blocked on a compact RELION live-state capture with corrected device Euler
  copies and complete rank/optics/metadata/control state.
- The K=4 100k/256 compact-score memory run is progressing without OOM on an
  A100; the dependent strict Hungarian FSC/FSC-AUC and state audit remains the
  quality gate.

Priority order: finish and validate the compact uninterrupted RELION live
boundary, then capture sharded full-dataset score/posterior arrays and compare
RECOVAR against that exact boundary. Treat independent RELION runs as whole-run
controls unless every boundary byte matches. Keep capped runs fail-closed
without a synthetic terminal boundary; permit an aggregate
boundary substitution only if the full-dataset evidence selects a systematic
source; then use full FSC/FSC-AUC trajectories to repair the remaining real
K=1 drift and accept or repair the running K=4 trajectory. Avoid further
serial particle tracing unless the aggregate evidence identifies a systematic
subgroup.

Current status on 2026-07-16: the eight-case autonomous K=1 robustness matrix
passes every FSC/FSC-AUC trajectory, schedule, convergence, and finalization
gate.  The remaining K=1 completion cells are the running 100k/256 trajectory
and the real-10076 repeat-qualified production-boundary confirmation.  The
bounded exact raw-diff2 cache has passed its frozen-boundary and same-A100
repeat controls and is integrated as a performance-only optimization.  Cache
and recompute are bitwise identical at the changed boundary; all 3,000 poses,
translations, and hard assignments remain exact, direct map FSC-AUC is at
least `0.999999999845`, and the clean controls improve wall time by
`10.7--15.8%`.

The local weighted-sum precision repair is integrated at `94b8f2b2`. Its
sealed same-A100 diagnostic is accepted as a causal production bug fix, not
as trajectory completion or performance evidence. A corrected 100k/256 full
trajectory is running under Slurm job `11288959`, followed by FSC/FSC-AUC
audit job `11288960`. Compare its complete schedule, convergence,
finalization, cross-engine FSC-AUC, and GT FSC-AUC before expanding the patch
to alternate dense M-step routes.

K=4 case 11 now has a recurrent aggregate iteration-1 membership boundary:
RELION and RECOVAR agree for 9,999/10,000 particles in a same-A100 six-arm
control, and zero-based particle 7915 is the sole recurrent class transfer.
A same-A100 class-routing intervention changes exactly that particle and
restores all four RELION-vs-RECOVAR class-map FSC-AUC values to at least
`0.9999999671`.  The global-winner boundary is therefore causal and a
full-class pre-scatter capture is unnecessary.  The remaining discriminator
is a frozen canonical float64/complex128 replay of that one decision, with
native float32 closure controls, to classify operand generation versus
reduction/order/precision.  Do not return to serial particle tracing or
unstable iteration-3+ cliffs.

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

The same auditor reports numbered current-size, resolution, HEALPix, Pmax,
expected-accuracy, and assignment-change scalars plus convergence and final
all-data topology. RECOVAR `pixel_resolutions` are converted from shell index
to Angstrom and compared with RELION `model_classes.rlnEstimatedResolution`;
`model_general.rlnCurrentResolution` is reported separately as the inherited
scheduling resolution. RELION's converged unnumbered optimiser value
`rlnCurrentIteration=-1` resolves to the highest preceding numbered state.
Final Pmax/pose/translation arrays are compared when present, while unavailable
final support or class arrays remain explicitly not measured. The CLI writes a
compact aligned-array NPZ and SHA-256 manifest beside the JSON. It stays
diagnostic by default; thresholds and exact schedule/convergence gates apply
only when explicitly requested. Correlation is neither computed nor used.

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

That target is now accepted and integrated. Frozen case-22 replay proves the
retained raw diff2, converted score, winner, and bounded Pmax are bitwise
identical to recomputation and to an independent RELION 256-lane float32 tree.
The promoted complex128/float64 replay changes centered diff2 by at most
`2.11653e-5` without changing the winner, classifying the available residual
as ordinary reduction precision rather than cache behavior. Two independent
same-A100 trajectory groups preserve all 3,000 poses, translations, and hard
assignments; their direct map FSC-AUC is
`0.999999999845--0.999999999985`, and wall time improves by
`10.7--15.8%`. The original analyzer's iteration-1 failures were false
positives because the cache is inactive there and the analyzer treated a
two-sample native repeat minimum as a hard quality floor.

Canonical evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_exact_bpref_fc70abc3_20260716_111000/trajectory_matrix_fsc_only_summary_v2.json`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_exact_bpref_fc70abc3_20260716_111000/cases/22_small_severe_outliers_3k_g128_radial_noise5_bf80/trajectory_analysis/k1_case22_iter2_performance_diagnosis_v1.json`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_raw_diff2_float64_audit_20260716_190000/AUDIT.md` (SHA-256 `1cadfd76d816a89359a2b9e68bb72717d466269289021e337021dd229e5caf2d`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_raw_diff2_float64_audit_20260716_190000/provenance/FINAL_MANIFEST.sha256` (SHA-256 `eef8c4ef0c38eed9e9c71b1bf029d85d78628efe4bfe903cda53f9a08e374477`)

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
fine reduction, and coarse score formulation are also ruled out.  A frozen
two-candidate substitution identifies projected-reference generation as the
dominant operand: swapping only the RECOVAR projection into otherwise RELION
operands moves the canonical float64 score margin by `-0.00192113`, within
`1.55e-6` of the full RECOVAR margin.  Image and CTF/noise substitutions move
it by only `+1.59e-6` and `-3.54e-8`.  The raw device rotation matrices are
bitwise exact across all 18 entries, and RELION's production score lies inside
its enumerated 576-order float32 reduction envelope.

The projected-reference delta is `0.01138468` RMS, or `2.96e-5` relative, and
is concentrated in radius 8--16 rather than Nyquist/support-edge pixels.
RELION resident-double versus serialized-MRC PPref precision is not the cause:
the exact same-run corner delta is only `2.20e-11` RMS and `1.86e-9` maximum,
7,889x/1,460x smaller than the raw projection delta.

The production-device discriminator closes the remaining interpolation split.
Across all in-support samples, RECOVAR and RELION texture coordinates,
float32 coefficients, all eight corner indices, and conjugation flags are
bitwise exact.  All 20,064 corner-value accesses differ instead, at raw RMS
`1.80614e-7` and maximum `3.10020e-6`; the resulting hardware projection RMS
is `1.75617e-7`.  The captured projection reproduces ordinary same-run RECOVAR
scoring bitwise after the production output disk and scale.  Supplying RELION
staged values at the common geometry shifts the canonical score margin by
`+0.001919855`, closing the projected-reference factor.

A stronger controlled replay routes the same-run serialized RELION half-2
PPref through the identical RECOVAR device staging path.  Coordinates and
indices remain bitwise exact, while corners and hardware projection differ
from RELION by only `2.20e-11` and `1.19e-11` RMS.  Device staging is therefore
excluded materially.  The earliest proved boundary is upstream PPref grid-
value generation; the specific preceding construction input or operation is
still open.  The device capture is inert, with half1/half2/merged FSC-AUC
`0.99999997850/0.99999997795/0.99999998428` on the same A100 UUID
`GPU-64011c8c-bd98-eb41-2c46-dd201730ef64`.

The serial p8240 diagnostic stops at this upstream grid-value boundary.  Do
not trace the particle's preceding construction input or operation unless an
aggregate audit identifies a systematic cohort.  Continue with
distribution-level PPref-grid comparisons and controlled boundary
substitutions before considering a production change.

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
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it2_p8240_capture_505af690_20260716_124319/distribution_substitution_v1/analysis/recovar_device_corner_coordinate_report.json` (SHA-256 `9e12292a35cdf4d2d6c69a96d682e804bcdf486805f40f4ae66a84f8faf9ea4f`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it2_p8240_capture_505af690_20260716_124319/distribution_substitution_v1/FINAL_SEAL.txt` (manifest SHA-256 `b92c370d690a77ed1d03be0cb7727d69d0339cfc1dffda3edf5de76e2c50a76f`)

### Real-10076 production BPref high-precision replay

The aggregate half-2 production capture now closes bitwise on all 23 shards.
The recomputed production numerator and weight match both captured-active and
captured-signature operands exactly, and two same-device controls are bitwise
repeatable.  The earlier failed closure was a diagnostic-harness bug: the
writer widened live float32 reconstruction probabilities to float64.  The
capture now preserves and binds native dtype, itemsize, and byte count; legacy
bundles require an explicit audited dtype and exact storage round-trip.

With the corrected float32 source, changing only the translation reduction
order gives numerator relative-L1 `1.8471e-4` to `1.8575e-4`, while weights
remain bitwise exact.  Genuine float64/complex128 operand and geometry
recomputation reaches a discrete geometry boundary rather than a verified
same-target replay: 3,728 support decisions change, 128 target entries change,
and no row-conjugation decisions change.  The support differences recur at
exact radial-boundary pixels, and target changes occur in eight-neighbor
groups at integer-lattice boundaries.  The tool therefore emits
`GEOMETRY_PRECISION_BOUNDARY` and deliberately writes no same-target artifact.
This is evidence of a precision-sensitive geometry boundary, not evidence for
a production EM arithmetic defect and not permission to substitute a
cast-only float64 replay for recomputed operands.

An independent same-A100 RELION-only full-trajectory pair converged with last
numbered iterations 17 and 16.  Its fail-closed launcher stopped before
RECOVAR, so it provides only another RELION repeat-envelope observation, not
RECOVAR parity evidence.  Synthetic aggregate convergence remains a strict
gate; real-data convergence claims must account for the observed same-engine
one-iteration variation.

Evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_production_closure_9044b379_20260716_165154/analysis/recomputed_high_precision_901dc198_gpu2_v5_summary.json` (SHA-256 `a9b58ead795f9890026c35ebf232ae3140c1510e478a954e13abb1cc519f6bed`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_production_closure_9044b379_20260716_165154/analysis/HIGH_PRECISION_V5_AUDIT.md` (SHA-256 `228ea927501e7f9f73ea21cad7052e14a8626f593ee5901deb06a487e20c65bc`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_fulltraj_055b7dc4_grid_off_ab_prepared_20260716_115437` (Slurm job `11270504`; RELION-only controls, no RECOVAR arm)

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

The authoritative four-case, exact-UUID robustness gate has three direct
passes and one case with a stable early boundary followed by an unstable late
trajectory at source `b5dd574a`.  The radial-noise/nonuniform/outlier Ribo
case 10 passes in job `11274945`: its iteration-1--3 minimum direct FSC-AUC is
`0.999634051`, `0.999666644`, and `0.999446341`, minimum class agreement is
`0.9956`, and the worst RECOVAR-minus-RELION GT FSC-AUC delta is `-8.00e-5`.
The 30k-particle Tomotwin case 12 passes in job `11274946` with minimum direct
FSC-AUC `0.999820693`, `0.999157434`, and `0.996495854`, minimum assignment
agreement `0.9986`, and worst GT delta `-6.28e-5`.  Case 14 remains essentially
exact, with minimum direct FSC-AUC `0.999999959` and class agreement `1.0`.

Case 11 (IgG, white noise, uniform classes, 20% outliers) has a stable early
cross-engine reconstructed-reference boundary whose numerical-versus-
algorithmic cause remains unresolved.  Its later class cliff is not itself a
stable parity boundary.  Iteration 1 is effectively exact (minimum direct
FSC-AUC
`0.999330786`, one class mismatch), but iteration 2 already has 9,999/10,000
Pmax differences, 1,166 support differences, and 24 class mismatches.
Iteration 3 amplifies this into class-2 direct FSC-AUC `0.9735133506`, below
the uncalibrated `0.995` gate.

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
different support sizes fall from 1,114 to 31.  This rules out an intrinsic
iteration-3 scoring/M-step cause.

The predecessor A/B closes the boundary one iteration earlier.  Job
`11276736` ran both arms serially on physical A100 UUID
`GPU-8321d67b-1e79-11ea-92b6-4347aa290a77` and completed `0:0`.  Arm B changes
only the four iteration-1 class maps used as RECOVAR scoring references for
iteration 2.  Its minimum direct RELION FSC-AUC rises from `0.998533895` to
`0.999999956` at iteration 2 and from `0.973512396` to `0.999924228` at
iteration 3.  Iteration-3 class agreement rises from `0.9964` to `1.0`, while
the worst GT FSC-AUC delta is only `-5.18e-6`.  The two arms differ materially
at iteration 2 (`0.998533543` minimum direct FSC-AUC) and iteration 3
(`0.973511552`).  The broad case-11 state drift therefore begins at the
iteration-1 reconstructed-reference boundary, not at the isolated class tie
or at later scoring.

The hardened same-A100 stock RELION repeat, job `11277907`, adjudicates the
magnitude.  Independent RELION runs on physical UUID
`GPU-6f45f415-9d0b-d562-9ff3-c9fb7bc53aa7` have minimum matched map FSC-AUC
`0.9999999995` at iteration 1 and `0.9999999975` at iteration 2; iteration-2
class agreement is exact, all support sizes agree, and the maximum GT FSC-AUC
change is `1.19e-6`.  RECOVAR/RELION's iteration-2 minimum `0.998533895` is far
outside that native repeat floor, so the iteration-1 reconstructed-reference
state is a real stable parity boundary.  By iteration 3 the stock RELION
repeat itself bifurcates to minimum map FSC-AUC `0.725582397`, class agreement
`0.9719`, 2,297 support differences, and angle p99 `118.12` degrees.  The
RECOVAR/RELION iteration-3 value `0.973512396` is inside that nonlinear repeat
envelope and must not be treated as an independent defect.

The stable boundary is not yet classified as an algorithm bug.  The sole
hard-class mismatch is a one-float32-ULP tie, and small aggregate pose/map
differences may also be reduction or representation effects.  Freeze the
iteration-1 reconstruction operands and compare production order, canonical
order, and recomputed float64/complex128 operands before changing production.
Gate the early arrays and FSC/FSC-AUC; use iteration 3 only as a sensitivity
outcome.

The sole iteration-1 label mismatch is particle 7915.  Its RECOVAR class-2
score `0.5038749576` exceeds class 1 `0.5038748384` by exactly
`1.1920928955e-7`, one float32 ULP, at the same rotation and translation.
This closes that discrete decision as numerical, but does not by itself prove
that all aggregate iteration-1 map differences come from this particle.

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
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_case11_it2_relref_ab_b5dd574a_20260716_143300/analysis/ab_summary.json` (SHA-256 `6715f73fa5aa14787e3a84f4a6dfb22338edfeca58fb17bd964d989c90c1b04e`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_case11_it2_relref_ab_b5dd574a_20260716_143300/provenance/SEAL.json` (SHA-256 `ca4ff7a2100529831c69a6a846523b927599ccfa86d38d137b133fed08fd0b82`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_case11_relion_repeat_full3_hardened_20260716_145000/analysis/relion_repeat_full3.json` (SHA-256 `86e8e5e12dfb75be7289169611a953b0fce9f810d440e3e2bfbb578933ee2e3f`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_case11_relion_repeat_full3_hardened_20260716_145000/provenance/SEAL.json` (SHA-256 `3ceb14d3b11cd46a17648a645e8c02d0ef80e0389c5b202415decba01de03a6f`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_case11_p7915_rec_capture_b5dd574a_20260716_142900/analysis.json` (SHA-256 `59af9e01552dbb0d35a6277b232ee6594c0a70306b3f654d6acb86c8b2951d15`)

## 2026-07-16 K=4 significant-count tie metadata

RELION serializes the pre-tie cutoff rank as
`rlnNrOfSignificantSamples`, while its inclusive threshold mask retains every
sample tied at that cutoff.  RECOVAR previously serialized the expanded mask
cardinality.  The K=4 path now exposes the cutoff rank from the existing
top-k/full-sort computation and uses it only for metadata; the expanded mask,
posterior, and all reconstruction accumulators are unchanged.  Tied, capped,
fallback, tuple-compatibility, first-iteration, and accumulator-invariance
tests pass (74 affected tests in the integrated checkout).

## 2026-07-16 K=4 case-11 firstiter winner boundary

Fresh same-physical-GPU job `11280871` ran two RELION and two RECOVAR arms
serially on GPU UUID `GPU-ed3fe7be-abe7-7c79-06da-bc76e74d6025`.  RELION's
iteration-1 class counts are `[4293,846,797,4064]`; RECOVAR's are
`[4292,847,797,4064]`.  In that selected RECOVAR control arm, exact
image-identity alignment finds one class difference among 10,000 particles:
zero-based particle 7915 (`7916@particles.128.mrcs`) moves from RELION class 1
to RECOVAR class 2.  This is an arm-specific statement, not a claim that every
RECOVAR repeat has exactly one mismatch.

The subsequent six-arm same-H100 seal resolves that repeat qualification.
All three RELION arms are membership-identical with counts
`[4293,846,797,4064]`.  RECOVAR control A and its capture are identical with
counts `[4292,846,798,4064]`, while RECOVAR control B has counts
`[4292,847,797,4064]`.  Particle `6326@particles.128.mrcs` changes between
RECOVAR classes 3 and 2 across native repeats and is therefore inside the
observed RECOVAR repeat envelope.  Particle `7916@particles.128.mrcs` remains
RELION class 1 versus RECOVAR class 2 in every observed arm and is the recurrent
cross-engine decision boundary.  Its 66,816 coarse candidates have an exact
rotation bijection and at most `2.583333094e-7` pixel translation difference;
the centered sign-converted score residual envelope is `1.162290573e-6`, while
the RECOVAR and RELION winner margins are `1.192092896e-7` and
`5.960464478e-8`.  This places the decision inside the measured cross-engine
float32 score envelope but does not replace native-repeat and recomputed
float64/complex128 aggregate controls.

The subsequent sealed same-A100 causal intervention closes that M-step
boundary.  Stock RELION, RECOVAR control, and RECOVAR with only original
zero-based particle 7915 forced from class 1 to class 0 ran serially on one
physical GPU.  The override exactly matches RELION's membership and counts for
all 10,000 particles.  It removes `99.970257%` and `99.998274%` of the
zero-based class-0 and class-1 FSC-AUC defects, raising their RELION/RECOVAR
FSC-AUC from `0.9998860754/0.9993307839` to
`0.9999999661/0.9999999884`.  Classes 2 and 3 remain at the native numerical
floor.  Thus the sole recurrent firstiter winner routing is causally
sufficient for essentially the entire affected first-map gap; no
backprojection-kernel defect is supported at this boundary.

The analyzer initially required bitwise identity for the two unaffected
classes, but the sealed data reject that assumption at the measured GPU atomic
repeat/order floor.  Affected/unaffected accumulator residual ratios are at
least `121,698x` for `Ft_y` and `45,685x` for `Ft_ctf`; the experiment's
descriptive `100x` fail-close is not a general tolerance or confidence
interval.  The next discriminator is the integrated offset-free
all-10,000-particle comparison of winner scores, margins, posteriors, and
support.  Only if that aggregate evidence identifies a systematic
near-boundary subgroup should a bounded canonical float32 and recomputed
float64/complex128 subgroup replay follow.  Do not return to serial particle
tracing or full-class BPref capture without such evidence.  Intermediate gates
remain exact/array metrics and map gates remain shellwise FSC/FSC-AUC;
correlation is not a gate.

The causal intervention and aggregate control are now complete. Slurm job
`11284582` changed only particle 7916's pass-2/M-step routing from RECOVAR
class 2 to RELION class 1. Native class counts `[4292,847,797,4064]` became
`[4293,846,797,4064]`, and RELION-vs-RECOVAR class-map FSC-AUC improved from
`[0.9998860711,0.9993307849,0.9999999891,0.9999999668]` to
`[0.9999999671,0.9999999887,0.9999999892,0.9999999693]`. This proves the
single membership decision fully explains the downstream reconstruction
residual.

Slurm job `11285753` then captured all 10,000 global winners for two RELION
and two RECOVAR arms on one A100 with each arm bound to its own verified
dynamic MPI dispatch schedule. Both engines are internally winner-identical;
the only cross-engine mismatch is particle 7916. RECOVAR selects class 2 at a
captured zero margin, while RELION selects class 1 with margin
`5.960464478e-8`. Native float32 evidence alone cannot determine whether that
last score difference is upstream operand generation or reduction precision,
so the remaining experiment is the bounded canonical float64/complex128 replay
of this decision.

The three-iteration causal replay also passes. Slurm job `11286050` applies
only the same particle-7916 firstiter routing correction, then runs iterations
2 and 3 autonomously. Class agreement is `1.0000`, `0.9999`, and `1.0000`;
minimum per-class cross-engine FSC-AUC is `0.9999999661`, `0.9999999568`, and
`0.9999242330`. The old iteration-3 minimum was `0.9735133506`. The worst
RECOVAR-minus-RELION GT FSC-AUC delta is `-3.52431e-6`. This seals the entire
case-11 trajectory defect as a downstream cascade from the one firstiter
membership decision; only the score-boundary classification remains open.

Canonical evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_case11_it1_mstep_boundary_b5dd_20260716_153035/RESULTS.md` (SHA-256 `393e9b2613662f3d6a0a8aba702d12ca0595443944c9b62621eedb93d2a633fa`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_case11_it1_mstep_boundary_b5dd_20260716_153035/provenance/CLASSIFICATION_SEAL.json` (SHA-256 `02cb2f88bc9dd6c4fdfbddfbb69c706b805b067afa60ab56ab7a6cfc7f9c4f94`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_case11_it1_mstep_boundary_b5dd_20260716_153035/analysis/fresh_f32/summary.json` (SHA-256 `1a42b1eb7e2b615c8154759c7a9cf72f3812330315bc4bc555bbcef1d26872a7`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_case11_p7916_identity6_h100_20260716_165455/analysis/six_arm_global_membership_repeat_join_v1.json` (SHA-256 `c8a0449b611eb77976d3c4f8dce052c757da958052206b88b23cf3b2e550dfeb`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_case11_p7916_forced_membership_20260716_180500/RESULTS.md`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_case11_p7916_forced_membership_20260716_180500/provenance/CLASSIFICATION_SEAL.json` (SHA-256 `f8ed739400fc18cd6abe78e9f7c11442c9128e3e9a518d9a0c82e62322ae3ac3`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_case11_global_winner_aggregate_dynamic_context_de2c_retry2_20260716_185100/analysis/global_winner_analysis.json` (SHA-256 `131b8fe31fb8f5690fa1b572492127e4e764f142a6c7cb6646eea620063a0be2`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_case11_global_winner_aggregate_dynamic_context_de2c_retry2_20260716_185100/provenance/FINAL_MANIFEST.sha256` (SHA-256 `063b38a73432ee2c49bb6d0408a38b76805c279dd173f611c2a932051920dd70`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_case11_forced7916_fulltraj_20260716_192000/analysis/forced_fulltraj_report.json` (SHA-256 `6d9779a242f738ea147628d86b3f9db8ed6124fe8ca2f683e55e957a9afb4c82`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_case11_forced7916_fulltraj_20260716_192000/provenance/FINAL_MANIFEST.sha256` (SHA-256 `029b796d10cb717b990a528d5ba5d0dce588c44b85d70aba405687e0f92c7b79`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k4_particle7915_causal_mstep_a100_20260716_184912/analysis/causal_mstep_report_v1.json` (SHA-256 `564fa793a617303556432fd2f60157d0c208d69473e965d2368b2e4f4062fccd`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k4_particle7915_causal_mstep_a100_20260716_184912/provenance/completion_seal_v1.json` (SHA-256 `cd3970196647ea6ae59a996883ed7b553b1dc12b660a0529789d4f35b396b8f1`)

## 2026-07-16 real-10076 iteration-2 ordinary BPref classification

The sealed particle-8240 boundary now excludes ordinary indexed geometry,
atomic order, and production float32 precision as material explanations for
the recurrent iteration-2 numerator gap. The production flattened capture
closes its known later half-join boundary at data relative L2 `5.16e-8`. A
same-A100 order replay changes the accumulator by at most `2.86e-7`, only
`0.0242%` of the RELION gap, and its minimum map FSC-AUC is
`0.9999999916`. Device-produced ordinary indexed geometry closes the live
production accumulator at relative L2 `1.30e-7`.

Recomputing operands from the raw source in float64/complex128 changes the
accumulator by `1.22e-4` but removes only `0.554%` of the RELION residual; the
controlled map remains at FSC-AUC `0.9999999869` versus production. This is a
real internal precision effect, but not a material explanation for the parity
gap. The reusable ordinary indexed signature path captures exact production
source values, rotation/pixel identities, eight neighbor indices and
coefficients, fold/support flags, and bitwise shadows of every input and the
production accumulator. Its coefficient arithmetic is pinned to the actual
generic production scatter (fractions after adding the integer origin), not
the distinct fused strict-RELION formulation.

The remaining classification is upstream algorithm/formulation difference in
the BPref numerator. Do not continue serial particle-by-particle probes. The
next bounded experiment is an aggregate RELION pre-scatter numerator/support
capture, distribution comparison against frozen RECOVAR rows, and controlled
RELION-numerator substitution through RECOVAR's closed ordinary geometry.
## 2026-07-16 K=1 bounded raw-diff2 reuse closure

At the frozen case-22 A1 iteration-2 boundary, cache OFF repeat, cache ON,
and reversed-input cache OFF are bitwise identical after restoring particle
order for every saved score, log-evidence/log-Z, Pmax, support count, hard
assignment, best pose, and rotation-posterior sum. All six pairwise merged-map
comparisons have normalized FSC-AUC `0.999999999997208--0.999999999997247`
(`1-FSC-AUC = 2.753e-12--2.792e-12`), so cache-ON variation is inside the
same-GPU repeat/order reduction envelope. FSC was evaluated from complex128
inputs and physically clipped before integration; correlation was not used.

Cache ON took `75.07` seconds versus `83.10` seconds for the mean of the two
same-order OFF controls, a `9.7%` speedup on this bounded high-support probe.
This is not a full-iteration or full-run speedup claim. Commit `7e48bcd85f735548f4d39ba1d5cc856581d5d8a2`
subsequently hardened admission to the minimum of 512 MiB, 1% of physical
device memory, 25% of physical free memory, and 25% of free JAX allocator
memory. Unknown/nonpositive memory disables reuse, as does
`RECOVAR_SPARSE_PASS2_EXACT_RAW_DIFF2_CACHE_MAX_BYTES=0`.

Sealed evidence (absolute path followed by SHA-256):

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_raw_diff2_boundary_probe_20260716_184500/artifacts/FINAL_REPORT_v2.md` — `6c74453da31015d5b109c3d6e750063a455a46ba94af6137307409e91daaf537`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_raw_diff2_boundary_probe_20260716_184500/artifacts/analysis_v2.json` — `62abe127d20fb28b11ef1a3dd66757856c6a18f0dd5dbcc57170a0cd6b98fb5a`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_raw_diff2_boundary_probe_20260716_184500/boundary/manifest.json` — `7b062d5d8126f74fb9d8969b39791cf7e9a5ce4dfb97be02297a5c6f50c0d320`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_raw_diff2_boundary_probe_20260716_184500/provenance/artifact_manifest.sha256` — `7737332f72c5ae8981b2fbf73222105cff6a323d514fe199d93d1f1e2504894b`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_raw_diff2_boundary_probe_20260716_184500/SEALED_v2.txt` — `dc164525bf61fd4d7a9a915a4fe2f13631503252dc7b18ff561429354b62c779`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_raw_diff2_allocator_probe_7e48bcd8_20260716_164400/allocator_probe.log` — `19c74ab7657b560e06fccc5373beded700a1e5e54a04ec3e6c308b14eba759f8`

`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_raw_diff2_boundary_probe_20260716_184500/artifacts/analysis_v1.json`
(`65dc3de3e6c3b47fdc3b31d7b8a47aa1ed03c019ccd5bfbb393af9707047b302`)
is explicitly superseded: it integrated float32 FSC values slightly above
one, and its mode arrays were overwritten by the independent v2 repeat and
are not sealed.

## 2026-07-16 K=4 particle-7916 precision classification

The final bounded replay classifies the sole recurrent case-11 firstiter
winner difference as `reduction_order_or_accumulation_precision_sensitive`.
RECOVAR's production float32 scores are tied at their stored precision and
route particle 7916 to zero-based class 1, whereas RELION selects class 0 by
one float32 ULP. Recomputing only RECOVAR's scoring arithmetic in float64 over
the same production complex64/float32 PPref representation selects class 0 by
`1.031123392e-7`, at the same pose. This agrees with RELION and with the
one-particle causal trajectory repair.

This is not a genuine upstream complex128 classification: the projector
operands remain production complex64/float32, and a RELION double-accumulator
or common complex128 operand replay was not completed. The reusable replay
therefore preserves ties and distinguishes promoted captured operands from
genuinely recomputed high-precision operands instead of labeling this result
more strongly.

Evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_case11_p7916_float64_replay_20260716/analysis/FINAL_CLASSIFICATION.json` (SHA-256 `6f8bf46d864b1b10d26c97704120b76edc23766168dfb4b024c779dc40794ebf`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_case11_p7916_float64_replay_20260716/provenance/FINAL_MANIFEST.sha256` (SHA-256 `009d3db710817243f3c561021e773ab5a225af989aae81917e33b8c4367f4ea1`)

## 2026-07-16 real-10076 aggregate pre-scatter classification

The complete passive capture contains 10,000 particles and exactly accounts
for every positive RELION BPref candidate. For all 5,000 half-2 particles,
stack identities, source support, oversampled rotation identities, and
rotation matrices are exact after the known RELION transpose convention.
Capture maps remain inside the same-A100 repeat envelope.

One particle, stack 111721, supplies `98.9083%` of numerator-difference energy
because RELION selects an adjacent 0.5-pixel translation absent from RECOVAR's
pass-2 child mask. Phase substitution reduces that particle's relative L2
from `0.153695` to `0.000207170`; its coarse selection remains separately
unresolved pending a complete float32/float64 score replay.

The other 4,999 particles expose a systematic composite-operand boundary.
Keeping all 4,483,086 source pixels and 35,864,688 device-produced RECOVAR
neighbors fixed, substituting RELION's captured complex source numerator
reduces BPref data relative L2 from `0.00109094374` to `2.79785117e-6`, removing
`99.7435%` of the residual. With common RELION weight and tau, map FSC-AUC
improves from `0.999998815829` to `0.999999999995716`; minimum shell FSC is
`0.999999999993895`. A true float64/complex128 RECOVAR source replay removes
only `0.554455%` of the cross-engine residual, so the recurrent bucket is
composite numerator formulation/generation, not materially production
precision, scatter geometry, or reduction order.

The next bounded discriminator is a stratified 32-particle capture that splits
the composite numerator into preprocessed FFT, translation phase, CTF,
`Minvsigma2`, posterior normalization, per-translation terms, and reduction.
Stack 111721 requires a separate complete coarse-score and generated-child-list
replay. Intermediate gates remain exact/array metrics; map gates use FSC and
FSC-AUC only.

Evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_relion_prescatter_20260716T224000Z/CLASSIFICATION.md` (SHA-256 `a2ea3a990e872089904079ac4f05ca7d7fdc0b2138b1ef8620fb711374150bc4`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_relion_prescatter_20260716T224000Z/aggregate_comparison_v1/SHA256SUMS` (SHA-256 `b3dc0908306a0eded53fb162116096f1d0aee6f77f6726c776929ad867488408`)
## 2026-07-16 real-10076 aggregate pre-scatter substitution

The aggregate substitution closes the iteration-1 half-2 numerator boundary.
On the same frozen ordinary indexed geometry, deterministic complex128/float64
replay gives RECOVAR-versus-RELION source-data relative L2 `0.0010909493`;
replacing the RELION source row for stack `111721` (original particle `8494`)
with RECOVAR reduces it to `0.0001144435`. That particle contributes `0.99434`
of the full data-delta L2 norm and is the already-classified discrete
translation-support outlier.

With one common RELION target weight and tau, the independently reconstructed
RECOVAR-source map has FSC-AUC `0.9999988141`, while the RELION-source map has
FSC-AUC `0.99999999999593`. These reproduce the sealed canonical-float32
results within the same-GPU order-control envelope. Excluding particle 8494
from the RELION-source substitution returns FSC-AUC to `0.9999988356`, showing
that the known discrete support difference dominates this map effect. The
absolute target-map replay is not hash-identical (FSC-AUC `0.9999133341`), so
only the controlled relative FSC/FSC-AUC effect is classified; no bitwise map
claim is made.

This causally classifies the observed aggregate BPref map gap as pre-scatter
operand generation, not ordinary scatter geometry or reduction order. Do not
extend serial particle tracing. Continue with distribution-level score and
posterior comparisons and controlled iteration-boundary substitutions.

## 2026-07-16 real-10076 stack-111721 coarse boundary audit

The complete iteration-1 coarse surfaces align as 36,864 rotations by 29
translations. Rotation identities are bitwise bijective after the RELION
transpose convention, and translations agree to `2.46509e-7` pixels. RELION's
winning translations are separated by exactly one float32 ULP; RECOVAR selects
the adjacent translation by 15 ULP. The cross-engine centered score residual
has RMS `1.55031e-7`, so the discrete winner change remains within the measured
float32 boundary variation and sends the engines to adjacent, disjoint fine
supports.

RECOVAR's frozen fine scorer selects the same winner when evaluated on
RELION's support, excluding a fine-scoring bug. The capture lacks the exact
coarse per-pixel projector/contribution operands, however, and rebuilding from
iteration MRCs does not reproduce the production float32 control. The result
therefore remains fail-closed as
`coarse_float32_near_tie_changes_fine_support_precision_unresolved`, rather
than being labeled pure numerical noise or an upstream algorithm mismatch.

Evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_stack111721_coarse_audit_20260717T000146Z/analysis/report.json` (SHA-256 `0486a185f35d5ed0295583b5fc72aaa339676b2028ab4142a458920c436ef95f`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_stack111721_coarse_audit_20260717T000146Z/analysis/seal.json` (SHA-256 `f0c8f855d54da62dbd71b6c5360b0d41048841115145e6f008222e41fa0f17fd`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_stack111721_coarse_audit_20260717T000146Z/ARTIFACTS.sha256` (SHA-256 `9d7cfd2e03f9e9fdb1faa13adc077b04c3b7b8b231a4f07e858a3c1e32efc0f6`)

## 2026-07-16 real-10076 BPref factor and GEMM-precision closure

A passive RELION factor capture and deterministic 32-particle RECOVAR replay
close the individual processed-image, CTF, noise, translation, shifted-image,
weighted-CTF, numerator-term, and weight-term operands to below `6.43e-7`
per-particle relative L2. RELION's captured terms reduce to its summary rows
within `3.89e-8`. The remaining production numerator residual is caused by
the default A100 GEMM precision: its global relative L2 is `2.0710e-4`, while
explicit `HIGHEST`, sequential float32, and genuine float64 close RELION at
`3.53e-7`, `3.53e-7`, and `2.74e-7`, respectively.

The production repair requests `jax.lax.Precision.HIGHEST` only for the local
complex64 M-step numerator matmul. A same-physical-A100 control/fixed run shows
that this is the only changed selected factor array and that the fixed
production numerator is exactly equal to both the HIGHEST and sequential
controls. Aggregate numerator relative L2 changes by `1.159e-4`/`1.139e-4`
for the two halves; denominator changes remain `3.30e-8`. Control-to-fixed map
FSC-AUC is `0.9999999876`, `0.9999999877`, and `0.9999999919` for half 1,
half 2, and merged maps. Internal half-map FSC-AUC improves by `3.56e-7`.
Cold production wall time was 188 s versus 193 s, with a warm-half pass-2
time of 5.31 s versus 5.25 s; this run does not establish a material speed
regression.

A separate capture-free warmed control/fixed/control run on one A100 now
closes the bounded performance gate. Relative to the bracketed control mean,
the fixed iteration wall is `-1.76%`, E-step `-2.01%`, sparse pass 2 `+0.14%`,
external wall `-0.98%`, and host maximum RSS `+0.26%`. The fixed values lie
inside the control envelope or far below the `10%` regression threshold.
Peak monitored GPU memory for the serial five-arm warmup/A-B-A sequence is
`17,695` MiB. This is one real-data iteration, not full-trajectory timing.

The earlier near-`-1` signed FSC was a diagnostic loader error, not a
production map-sign mismatch. `analyze_bpref_reduction_precision_ab.py`
raw-loaded both MRC files and then applied an ad-hoc RECOVAR multiplier of
`-1`. The canonical comparison instead loads RECOVAR outputs with `load_mrc`
and RELION outputs with `load_relion_volume`, whose explicit frame conversion
is `-transpose(raw_relion, (2,1,0))`; it requires no additional sign
alignment. Under those loaders, the sealed real-10076 iteration-1 FSC-AUC is
positive: control half1/half2/merged are `0.9999999889`, `0.9999999545`, and
`0.9999999823`, while fixed values are `0.9999999898`, `0.9999999556`, and
`0.9999999829` in the independent canonical shell audit. The corrected A/B
analyzer's own FSC implementation gives control `0.9999999860`,
`0.9999999518`, `0.9999999786` and fixed `0.9999999870`, `0.9999999527`,
`0.9999999791`; these are exactly the prior sign-flipped magnitudes to floating
roundoff. The raw-file comparison's sign-flipped FSC magnitudes remain
numerically valid for the A/B effect because FSC is unchanged by the common
axis transpose, but the opposite-production-sign interpretation is withdrawn.
This bounded one-iteration intervention confirms the precision fix; it does
not replace full real-data trajectory parity.

Canonical evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_factor_boundary_20260716T235500Z/artifacts/factor_comparison_v1.json`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_tf32_same_gpu3_ab_20260717T034500Z/analysis_v2.json`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_tf32_same_gpu3_ab_20260717T034500Z/analysis_v3_canonical_frame.json` (SHA-256 `d134051bdcd2bb3e735b02b9b383f0e47ca8dd3f925df758aec829372749d57b`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_sign_boundary_20260717T012312Z/sign_boundary_report_v1.json` (SHA-256 `d7a8231076106e871073d26997ca0b71f18da1d6daeb8e2bad960aa8290e363e`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_highest_ab_20260717T005507Z/real10076_highest_ab_report_v1.json` (SHA-256 `c6a7d81f473fe793bc76363233e76bc26ee66afe4e3fd5169a9a181abf9441c5`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_highest_ab_20260717T005507Z/real10076_highest_ab_seal_v1.json` (SHA-256 `4ea738680379aec8318ce7c569eb272f820369751d45495a31cf535aa6a2ff54`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_highest_ab_20260717T005507Z/real10076_highest_ab_manifest_v1.sha256` (SHA-256 `6f20ff79bc47b83f4f198b28a35a332487a1ea93188e5e8bf769f507df33831c`; all 13 entries verified)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_highest_perf_aba_20260716T212900Z/warmed_aba_performance_report_v1.json` (SHA-256 `0a20edac7a898f1997d13fa1ab75d1f0cbeb2ac46c11a049498af6759a7a79de`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_highest_perf_aba_20260716T212900Z/warmed_aba_manifest_v1.sha256` (SHA-256 `a3552ac5d05e0305544e86560cd2fd92b88d11d410074dd39f737da3387b411b`)

## 2026-07-16 K=1 100k scale and convergence qualification

The canonical 100,000-particle, grid-256 supplied-map trajectory at source
`505af690` passes the complete numbered and final FSC/FSC-AUC audit. RELION and
RECOVAR both converge at numbered iteration 14 with the same current-size
schedule. RECOVAR runs final all-data exactly once after convergence, with grid
correction explicitly unset/off; every expected final half and merged product
is present. Across all numbered boundaries, merged cross-engine FSC-AUC is at
least `0.9999604047`, and the minimum RECOVAR-minus-RELION merged GT FSC-AUC
delta is `-3.079e-5`.

The final merged cross-engine FSC-AUC is `0.9986956770`. RECOVAR final merged
GT FSC-AUC is `0.5444742912`, versus `0.5363817908` for RELION, a delta of
`+0.0080925003`. These are FSC/FSC-AUC results; correlation is not computed or
used.

Both engines ran serially in Slurm science allocation `11268911` on the same
A100-SXM4-80GB. RELION took `30,745` seconds and peaked at `80,053` MiB;
RECOVAR took `8,094` seconds and peaked at `34,597` MiB. RECOVAR is therefore
`3.7985x` faster by external wall time in this gate. Audit job `11291289`
completed `0:0`; it also verifies the clean science allocation, convergence
topology, final products, wall records, GPU samples, and source provenance.

This trajectory predates the local numerator `Precision.HIGHEST` repair, so it
qualifies the existing 100k scale baseline rather than replacing the ongoing
corrected-precision trajectory. The corrected run must retain the same quality,
convergence, and performance conclusions before that repair is promoted as the
full-scale default.

The aggregate particle-state auditor also passes exact schedule and convergence
gates on the explicitly selected iteration-1--5 boundary subset in Slurm job
`11291778`. This older archive saved significant-support counts only for those
five boundaries, so the auditor correctly rejects an implicit full-trajectory
claim. Within the measured subset, support-count differences affect
`0/58/90/155/278` of 100,000 particles. Pmax absolute p95 grows from zero at
iteration 1 to `0.0085577` at iteration 5, with systematic concentration among
lower-Pmax particles. These finite array differences remain diagnostic rather
than being reclassified as a map-quality failure; the independent numbered-map
FSC/FSC-AUC gate above passes every boundary.

The complementary lightweight intermediate auditor covers all 14 numbered
boundaries despite the older support-count retention limit. It finds no missing
or malformed arrays and no current-size, Healpix-order, or iteration-topology
failure. Its largest finite diagnostic is iteration-10 per-particle Pmax
relative L2 `0.0228949` (absolute p95 `0.0147998`, p99 `0.0284984`, maximum
`0.667081`, and mean signed delta `-4.82e-5`). This is intentionally reported,
not threshold-hidden and not called numerical noise. It is also not a map gate:
the iteration-10 cross-engine merged FSC-AUC is `0.9999850043`, and map quality
continues to be decided by FSC/FSC-AUC.

Canonical evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_scale100k_505af690_20260716_112000/cases/1_baseline_100k_g256_white_noise1_bf80/trajectory_analysis/k1_scale_acceptance.json` (SHA-256 `2c0c4de857b509ffcc56fb4caea7ea263775a549710fa3dbe58cadc5974923be`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_scale100k_505af690_20260716_112000/cases/1_baseline_100k_g256_white_noise1_bf80/trajectory_analysis/k1_scale_runtime.json` (SHA-256 `2266cbefbf8bad7f9dca1b95f2b558b9d665cd74558ce0f8555c7bbe9fafcc1a`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_scale100k_505af690_20260716_112000/cases/1_baseline_100k_g256_white_noise1_bf80/trajectory_analysis/k1_fsc_trajectory.json` (SHA-256 `4f876bf1e03f82b2d006a6495461063a2f4a88fadb45a4a30d3bfa8185350960`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_scale100k_505af690_20260716_112000/analysis/aggregate_state_subset_000_004_7de6ae20_20260716T224700Z/aggregate_state_subset_000_004.json` (SHA-256 `ba4265fff0111e12c1f3ce8ef26105820469173525a1a939f352bd71a5f0d5f1`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_scale100k_505af690_20260716_112000/analysis/intermediate_trajectory_b59b90f4_20260716T225300Z/k1_intermediate_trajectory.json` (SHA-256 `f1ee77eb87791b1e68b13ea074861f6423816f837734b9300cbd7544bd3ff2fb`)

## 2026-07-17 sealed native restart at the 100k expected-accuracy boundary

The exact-order native restart closes the remaining implementation ambiguity
at the terminal iteration-16 expected-accuracy boundary. The disposable root
is `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_100k_relion_restart_accuracy_oracle_20260717T054518Z`;
build job `11296008` and native MPI restart job `11296009` completed `0:0`.
The process stops after expected accuracy and before expectation.

A normal restart is inadmissible because RELION reshuffles particle order in a
new process. This control restores the exact original first 100 trial IDs after
the shuffle and proves that they are unique, all in half 1, and row-identical
between iteration-0 and iteration-15 STAR files. The ID file SHA-256 is
`84262018ebd56268dfb8cfa1e674e97b09f8d9a712f967df39b2f3c0a0e6190a`,
the canonical int64 sequence SHA-256 is
`0d4dc2a259d594b2bc656fc763c8a41413c78e5595e2043c36f8947f9388142a`,
and the proof JSON SHA-256 is
`068e5c8955b1ca34a695f92490493bccb3654d972f2c8b94374dcd778a274fbc`.

Native restart gives `0.623` degrees and `0.635375` Angstrom; independent
reduction of its 100 per-trial rows gives `0.6230000000000006` and
`0.6353750000000011`. This exactly matches the serialized standalone
all-RELION replay, while uninterrupted native entry-to-iteration-16 gives
`0.625` degrees and `0.6375` Angstrom. The standalone binding is therefore
exonerated. The unresolved boundary is live in-memory state versus checkpoint
write/reload, or mutation-before-write state that is not serialized. This
approximately `0.002`-degree terminal effect does not explain the much earlier
real-10076 schedule split; that work remains distribution-level and must not
return to serial particle tracing.

Sealed evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_100k_relion_restart_accuracy_oracle_20260717T054518Z/analysis/original_trial_identity_proof.json` (SHA-256 `068e5c8955b1ca34a695f92490493bccb3654d972f2c8b94374dcd778a274fbc`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_100k_relion_restart_accuracy_oracle_20260717T054518Z/analysis/native_restart_accuracy_v1.json` (SHA-256 `4249a06cff0fb63884fa7f68079b56a52fa6a0d1ae90c095bc5ecf1e57e587fd`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_100k_relion_restart_accuracy_oracle_20260717T054518Z/restart/per_trial_errors.csv` (SHA-256 `d29e4b460c5562a0328f4fb5ed6806c138bc4408c6c765e95bb224aa68d91da4`)

## 2026-07-17 real-10076 hidden-change tail and schedule intervention

The reusable hidden-change distribution audit shows why the earlier pose-error
p95 was insufficient at the iteration-6 to iteration-7 schedule boundary.
Against the delayed native RELION target, 518 of 10,000 particles have an
absolute hidden-angle-change difference above 0.1 degrees. All 518 also belong
to the union of cross-engine pose mismatches at iteration 6 or 7. The top 1%
of particles accounts for `92.2177%` of the total absolute difference and
`0.090721` degrees of the `0.091198`-degree signed mean gap. The subgroup is
low-Pmax enriched: RECOVAR iteration-7 mean Pmax is `0.53343`, versus
`0.69695` outside the subgroup. The same approximately 5% tail and top-1%
concentration recur against two independent native RELION repeats and in a
second autonomous RECOVAR control. This is systematic subgroup evidence, not
a reason to resume serial particle tracing.

The exact schedule predicate is nevertheless correct. The delayed RELION
trajectory changes from `4.901576` to `4.743451` degrees at iteration 7;
`1.03 * 4.743451 < 4.901576`, so its hidden-variable counter remains zero and
it stays at HEALPix order 3 through iteration 9. Native repeat A changes from
`4.897129` to `4.759138`, and repeat B from `4.873165` to `4.790615`; both pass
RELION's 3% stability predicate, increment the counter, and enter order 4 at
iteration 8. Both autonomous RECOVAR controls take that same early branch.
The relevant schedules through iteration 10 are therefore:

- delayed RELION: orders `3,3,3,3,3,3,3,3,3,4`, sizes
  `48,92,120,122,122,122,122,122,122,122`;
- native RELION repeats and autonomous RECOVAR: orders
  `3,3,3,3,3,3,3,4,4,4`; the repeat sizes are
  `48,92,120,122,122,122,122,122,126,126`.

The schedule intervention is causal and symmetric under FSC/FSC-AUC. Against
the delayed target, autonomous RECOVAR's iteration-8--10 merged FSC-AUC is
`0.756453`, `0.726126`, and `0.836135`; forcing the delayed HEALPix schedule
raises it to `0.986548`, `0.983385`, and `0.978128`, while additionally forcing
current size changes those values only to `0.986554`, `0.983788`, and
`0.978682`. Against native repeat A, autonomous RECOVAR gives `0.987268`,
`0.980848`, and `0.976246`, whereas the forced-delayed arm gives `0.755084`,
`0.727216`, and `0.837315`. Repeat B gives the same conclusion: autonomous
`0.987249`, `0.980833`, and `0.976848`, versus forced-delayed `0.755261`,
`0.727922`, and `0.836708`.

This closes the categorical schedule mismatch as native RELION near-threshold
repeat variability, not a RECOVAR scheduler defect. Do not patch the scheduler
to force the delayed branch. It does not close the remaining trajectory
quality residual: native RELION repeat A/B merged FSC-AUC is `0.996451`,
`0.993440`, and `0.990081` at iterations 8--10, above RECOVAR's approximately
`0.976` cross-engine value at iteration 10. The latter develops gradually from
iteration 2 and remains an open aggregate posterior/reconstruction diagnostic.

All four RECOVAR intervention arms shared physical A100 UUID
`GPU-f3e94635-d095-bea9-dbe3-26e91dd3ea27`; the delayed target's original
RELION/RECOVAR pair shared UUID
`GPU-a1bb1fb4-d5e3-1c72-3382-63f6032e9fc6`; and native RELION repeats A/B
shared UUID `GPU-bd720f2f-c28a-09c0-d51e-d08b1897125a`. Direct comparisons
between the new intervention arms and the pre-existing targets therefore use
the same A100-80 model but not the same physical UUID. They are causal
diagnostics, not a same-physical-GPU cross-engine acceptance gate. A fresh
live-RELION-derived same-allocation full trajectory is the next acceptance
experiment.

Slurm job `11294177` completed all four science arms but ended `1:0` only
because the scratch analyzer was invoked outside the repository module path
and could not import `scripts`. The corrected read-only analysis ran from the
repository root and produced:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_sampling_oracle_ab_ff2ed9d3_20260717T044116Z/analysis/schedule_oracle_ab_current_target.json` (SHA-256 `1f506256514e5d972531d9721b872582a12b7bbbb96dd590cb26d725be1da5d0`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_sampling_oracle_ab_ff2ed9d3_20260717T044116Z/analysis/schedule_oracle_ab_repeat_arm_a.json` (SHA-256 `9334cf530aedc794ac11d9df68b5f7bb9a0a1facde667be22715ca63898b38c5`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_sampling_oracle_ab_ff2ed9d3_20260717T044116Z/analysis/schedule_oracle_ab_repeat_arm_b.json` (SHA-256 `bf1646cab14f29061fa39b7f6cfbec4a8b83b91b0ee2cd7a64fe23d0bd586aae`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_sampling_oracle_ab_ff2ed9d3_20260717T044116Z/analysis/hidden_change_it006_it007_v1/hidden_change_distribution.json` (SHA-256 `ccb20a0aaffbc73ab9211a546dbc3ea1a9a479b0e85a02d5837004fbc8b16e32`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_sampling_oracle_ab_ff2ed9d3_20260717T044116Z/analysis/hidden_change_it006_it007_v1/hidden_change_distribution_arrays.npz` (SHA-256 `4908eae90123532e99eed169032655074a62c0ae648aefb96f8972c8e2e2b8c2`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_relion_repeat_envelope_a10080_retry4_prepared_20260715_195515/analysis/native_repeat_merged_fsc_v1.json` (SHA-256 `d6eba533c9d11b127855ad10dba4c1f3cb9f201df4ea0a0ff115b482a27e90c7`)

Intermediate classification uses exact/array distribution metrics. Map
quality uses FSC/FSC-AUC only; correlation is neither computed nor used.

## 2026-07-17 complete iteration-1 contribution replay

The real-10076 iteration-1 diagnostic now captures and replays both half-set
accumulators rather than extrapolating from individual particles. Half 1 has
35,864,288 common contribution records and no unmatched records. Its canonical
float32 RECOVAR-versus-raw-RELION data FSC-AUC is
`0.9999999999999207`. Half 2 has 35,864,688 common records and no unmatched
records. Its captured host replay agrees with the production accumulator at
data FSC-AUC `0.9999999999964321`.

The visible half-2 common-geometry residual is concentrated in one particle,
stack index 111721 (fixture row 8494), whose adjacent coarse translation is a
float32 near tie. Excluding that row leaves per-particle data relative L2
`3.4269e-7`; support, rotations, identities, and weights remain exact. Across
the full half-2 accumulator the common-geometry data FSC-AUC is
`0.9999995117964201`, its minimum non-DC shell is `0.9999900761504018`, and
the weight FSC-AUC is `0.999999999999994`.

Genuine upstream float64 geometry recomputation changes 128 target indices and
3,728 support indices among the 35,864,688 captured half-2 records. This is a
precision-boundary classification, not evidence of a systematic scatter or
reduction bug. It also closes the requested particle-1491-era serial probe:
further work proceeds at distribution level unless an aggregate subgroup is
identified.

Canonical evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it1_full_both_capture_replay_863ccafb_20260717/analysis/common_rec_geometry_replay_h1.json` (SHA-256 `6a89bb7364d90897fd499c76a3f822256c8511a73787dbfcda3e99cc2d9bc900`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it1_full_both_capture_replay_863ccafb_20260717/analysis/common_rec_geometry_replay_h2.json` (SHA-256 `80233e98d3d3204a346c2106e7dc3a1b79bf9901bec71889f0865aa68428c655`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it1_full_both_capture_replay_863ccafb_20260717/analysis/replay/h2_captured_replay.json` (SHA-256 `ecb290fc4c4e4d361e83e216eece437e229684d5cc6549d8d56a8fca6049c42c`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it1_full_both_capture_replay_863ccafb_20260717/analysis/replay/h2_genuine_upstream_f64.json` (SHA-256 `d269ae30dccfaad49c623fff17763a160ab03c8f4865f9afc9a8cc8061920e7f`)

## 2026-07-17 same-A100 live real-10076 trajectory

Slurm job `11297956` ran a fresh RELION trajectory, autonomous RECOVAR, and an
exact-schedule RECOVAR intervention serially on physical A100 UUID
`GPU-2f2a8197-bcc8-ec41-fc6f-dfb2b5aaf4fa`. RELION emitted 16 numbered
iterations and then converged/finalized. Both RECOVAR arms emitted 16 numbered
iterations but did not converge, so they correctly skipped final all-data and
their serialized final-half placeholders are not treated as valid final maps.

Autonomous RECOVAR exactly matches RELION's Healpix-order trajectory. Its
current-size trajectory differs only at iterations 13, 14, and 16. Forcing the
exact RELION Healpix and current-size schedules improves the merged
cross-engine FSC-AUC from `0.958310512629` to `0.974104824180` at iteration 13
and from `0.960801450738` to `0.979180351784` at iteration 16. It does not
remove the gradual residual: the forced trajectory's merged FSC-AUC evolves
from `0.999999983783` at iteration 1 to `0.999945877001` at iteration 3,
`0.995813118809` at iteration 6, and `0.971171072751` at iteration 11.

The exact scheduler arithmetic is therefore exonerated, while nonconvergence
and the gradually amplifying score/posterior state remain open. The next probe
is one aggregate 32-row iteration-2-to-3 panel with control/control,
float32/float64-order classification, and bounded state substitutions—not
serial particle tracing.

Evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_live_schedule_ab_955bfe1f_20260717T064000Z/analysis/dynamic_schedule_trajectory.json` (SHA-256 `6623ef84a18f18403cad7697c7727f55e63cd22be8e22f333e9746ad32d1d470`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_live_schedule_ab_955bfe1f_20260717T064000Z/provenance/science_artifacts.sha256` (SHA-256 `9fe6c50e1957d332457a2935e55afa41c09b354506a8c57e9961434729b7a096`)

All map statements in these sections use FSC/FSC-AUC. Intermediate statements
use direct identity, support, and array metrics; correlation is not used.

## 2026-07-17 real-10076 iteration-2-to-3 aggregate score boundary

The frozen 32-row panel localizes the earliest repeat-stable residual to raw
Gaussian scores. Production RELION and RECOVAR candidate UID support is exact
for all 32 rows. Replaying each engine's captured posterior from its native
float32 score order closes within total variation `7.29e-7`. In a deterministic
canonical-float64 factorial, the prior-only posterior effect has median
`7.13e-8` and maximum `8.21e-5`, whereas the raw-score-only effect has median
`1.5357e-4` and maximum `5.4059e-4`. Priors and posterior normalization are
therefore not the leading cause. One row, particle 2449, exchanges one of 144
M-step hypotheses; this is not a systematic subgroup.

A separate 32-row high-precision RECOVAR replay recomputes masked-image
background fill, FFT, CTF/noise weighting, translation phases, Projector
interpolation, and score reduction without intermediate float32 truncation,
while retaining the production float32-origin source samples and geometry.
The valid capture is Slurm job `11305358`. On the three-way common UID support,
median posterior TV is `1.57054e-4` for production RECOVAR versus RELION's
float32-origin scores, `1.28238e-3` for high-precision RECOVAR versus RELION,
and `1.27341e-3` for production versus high-precision RECOVAR. High precision
improves zero of 32 rows. It also substitutes 32 UIDs in each of particles
6007 and 1012; the other 30 supports are exact.

This establishes a large precision sensitivity but not canonical cross-engine
float64 closure: RELION's captured GPU operands remain float32-origin, and the
high-precision run changes pass-1 support. Do not label the production residual
as numerical noise from this experiment alone. The next bounded discriminator
holds production UIDs fixed and cross-swaps the coupled image/CTF/noise/scale
operand against the projected-reference operand, with direct-diff2,
algebraic-score, and reduction-order controls across all 32 rows.

Sealed evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it23_aggregate_panel_586f7fb4_20260717T093000Z/analysis/score_factorial.json` (SHA-256 `4075c4dfb2a65782de52c75f992e85be0cf7f9b22e0c9a24bb73c6880b1df7d0`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it23_aggregate_panel_586f7fb4_20260717T093000Z/provenance/score_factorial.sha256` (SHA-256 `39f8e2d971ab1f3a1806d85b347a292dbf2b4f81f5211b22890abf6b2ca89e3d`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it23_genuine_f64_20260717T145500Z/analysis/genuine_f64_vs_f32_origin.json` (SHA-256 `b54aa3a5221145ce1b6df204f3c5c3197b4230560f39fe3955b624c8b3c6c955`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it23_genuine_f64_20260717T145500Z/provenance/analysis_artifacts.sha256` (SHA-256 `b3137d124d83c5e91210641abf64be04c4444e16fcbdce0ba08455879a1047b0`)

These are direct array and posterior metrics. No correlation is computed; any
downstream map intervention remains gated by shellwise FSC and FSC-AUC.

## 2026-07-17 real-10076 fixed-UID score reduction classification

The fixed-production-UID operand/reduction factorial closes every one of the
32 captured rows. RECOVAR's native float32 reduction tree reproduces the
captured production scores exactly (`max_abs=0`), and the adjusted float64
direct-versus-algebraic closures are at most `4.38e-12`. The fixture contains
17,216 exact five-field candidate UIDs with explicit particle offsets and
priors.

Relative to the deterministic `math.fsum` score, ordinary float32 pairwise
reduction has median posterior TV `7.74e-5` (maximum `2.63e-4`) and RELION's
256-lane float32 tree has median `7.39e-5` (maximum `2.13e-4`). In high
accuracy, swapping only the projected-reference operand has median TV
`1.014e-5` (maximum `9.98e-5`), while swapping the coupled
image/CTF/noise/scale operand has median `6.17e-7` (maximum `2.37e-6`). The
original and canonical pixel-order float32 arms are identical because the
captured window indices are already canonically sorted.

The production raw-score residual is therefore on the measured float32
reduction-sensitivity scale, while the frozen high-accuracy operand difference
is smaller in the median and is projection-dominated. This does not yet prove
that RELION and RECOVAR use different production reduction orders. The final
bounded discriminator captures one immutable common per-pixel contribution
list from each native engine and replays it through both native schedules and
one shared canonical schedule. Do not change production code unless that
replay exposes a systematic operand, geometry, or order defect.

Sealed evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it23_aggregate_panel_586f7fb4_20260717T093000Z/analysis/score_reduction_factorial.json` (SHA-256 `271dbf2fbc84d99659971f5517d8dfe834afe94615a28931f80d1359fc36bb72`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it23_aggregate_panel_586f7fb4_20260717T093000Z/analysis/score_reduction_factorial_arrays.npz` (SHA-256 `ef6e6d87fae1d4e661d74a129ab6235550136b040f1f6692599ed86bc20f9d1b`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it23_aggregate_panel_586f7fb4_20260717T093000Z/provenance/score_reduction_factorial.sha256` (SHA-256 `6cde8dd18e8961c9d48a1a60c78392e1d030f842b4c962ab0059662b6f516a10`)

All intermediate comparisons use exact arrays and posterior distances; no
correlation is computed.

## 2026-07-17 K=4 100k/256 memory-cap acceptance

The production-scale K=4 retry validates the compact-score memory-planner
fixes through two complete numbered iterations. In iteration 1, all 68 compact
pair groups from size 512 through 270,336 completed, including the former OOM
boundaries at 24,576 and 94,208. Peak sampled A100 memory was 33,341 MiB of
81,920 MiB. Iteration 1 completed in 4,710.3 seconds and advanced to iteration
2 with current size 52.

Iteration 2 completed all 34 compact groups and all 16 rectangular groups,
with cumulative sampled peak 33,361 MiB. It completed in 2,727.6 seconds at
resolution 27.20 A and average Pmax 0.5128, remained nonconverged, and advanced
to iteration 3 at the RELION-derived current size 60. Both boundaries wrote
eight finite 256-cubed class/half maps plus finite tau2, noise, and 100,000-row
assignment arrays. No allocator failure, `RESOURCE_EXHAUSTED`, traceback, or
OOM signature occurred. This accepts memory safety and artifact integrity;
class-matched FSC/FSC-AUC quality acceptance remains gated on the dependent
strict audit after the trajectory completes.

Run root:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_100k_memcap_fce9ee48_20260717T132717Z` (science job `11304416`; strict dependent audit `11304830`)

## 2026-07-17 host-matrix incoming-boundary and tail-enrichment diagnostics

The frozen exact-RELION-iteration-2 incoming-iteration-3 A/B confirms that the
source-matched host inverse rotation handoff is causal but does not close the
remaining scorer residual.  Relative to the parent device-matrix path, median,
p95, and mean absolute Pmax error improve from `6.21627e-5`, `2.22109e-4`, and
`8.54588e-5` to `4.95276e-5`, `1.74502e-4`, and `6.70980e-5`; mean and p95
angular error improve from `1.33237e-5` and `2.94768e-5` degrees to
`5.17333e-6` and `1.01020e-5` degrees.  Merged map FSC-AUC versus RELION
changes from `0.999998435067` to `0.999998625826`.  The two host controls have
bitwise-identical Pmax, pose, translation, and significant-count state; their
merged self FSC-AUC is `1.000000027180` and the worst parent/host merged
FSC-AUC is `0.999999970436`.

Exact identity scattering exposes an important exception: the parent has
3/10,000 significant-support-count mismatches against RELION iteration 3,
whereas both host arms have 9/10,000.  The six new cutoff decisions are each
one count away and all move away from RELION (`+1,+1,+1,-1,+1,-1`).  Thus the
host path improves aggregate Pmax, pose, and FSC-AUC while worsening six
discrete support cutoffs.  Keep the source-matched fix; treat these cutoff
changes as evidence for the remaining score/texture arithmetic, not as host
support-parity improvement.

The new exact-identity cross-iteration diagnostic on the sealed autonomous
host trajectory shows a systematic early tail.  At the iteration-2 to
iteration-3 boundary, a significant-count mismatch at `t` raises the rate of
a `>0.1`-degree cross-engine pose error at `t+1` from `0.0848%` to `1.2433%`
(`14.6667x`) and captures 7/15 tail particles.  The exact top 5% of absolute
Pmax deltas raises that rate from `0.1053%` to `1.0%` (`9.5x`) and captures
5/15.  Support-mismatch enrichment remains `7.1413x`, `3.7132x`, `3.2311x`,
and `3.1111x` across the next four boundaries, but capture is incomplete.
At iteration 7 to 8 every particle enters the pose tail because the trajectories
take different sampling branches, so enrichment is exactly 1 and no longer
localizes a subgroup.  After broad divergence, later ratios similarly approach
1.  The iteration-1 top-5% Pmax selection has a zero cutoff and is a deterministic
tie selection, not interpretable enrichment.

This section is descriptive triage only: it computes no correlation and adds
no quality gate.  Continue with aggregate score/posterior distributions and
controlled boundary substitutions; do not return to serial particle tracing.
Map acceptance remains shellwise FSC/FSC-AUC.

Evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_incoming_it3_rotation_ab_7f_vs_8c_retry1_20260717T203134Z/analysis/it3_rotation_ab.json` (job `11318372`, SHA-256 `feb3571d41fdf1277c77f68f9a9c03601781a6f6859101278f023e8d4cea72dd`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_incoming_it3_rotation_ab_7f_vs_8c_retry1_20260717T203134Z/analysis/it3_rotation_ab_fsc_curves.npz` (SHA-256 `619969d0059c07c879ec4d02daf21b4e152657ab7b200f91cba7b510c64f04de`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_tail_enrichment_0fa05894_20260717T204707Z/analysis/particle_state_distribution_tail_enrichment.json` (job `11319327`, SHA-256 `73df4fab3130179c8abee447635fe944439b2827714ec4851ab186342d1a6cef`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_tail_enrichment_0fa05894_20260717T204707Z/analysis/particle_state_distribution_tail_enrichment_arrays.npz` (SHA-256 `6d76535d88f33be7fc11249177c64dc99699c0e529c75cedcbb04e65af468757`)

The A/B preflight and builds were clean at host commit
`8c83202739a31251bc6c10be834f237732e879d3` and parent commit
`7f142d5f00a34a0fd6208bdd6f879ffe31b3e9ea`.  During the serial arms the host
worktree advanced through a documentation-only commit, so the per-arm immutable
HEAD check was not preserved.  The hashed production sources and built CUDA and
binding artifacts did not change.  This remains a causal diagnostic, not an
exact-clean-runtime production acceptance run.

Code references:

- `scripts/audit_em_particle_state_distribution.py:_cross_iteration_tail_enrichment`
- `tests/unit/test_audit_em_particle_state_distribution.py:test_cross_iteration_tail_enrichment_uses_exact_aligned_state_and_is_diagnostic`

## 2026-07-17 native texture-context closure on an uninterrupted boundary

The remaining native texture-coordinate and sampled-register hypothesis is
closed. RELION ran uninterrupted from iteration 0 through iteration 3 on A100
UUID `GPU-c6d48651-75fd-c644-a83f-3879c0a58186`; its sampling perturbations
were exactly `-0.04961`, `+0.405200`, and `-0.30033`. Within each captured
invocation, the untouched production score kernel and a separate diagnostic
context kernel consumed the same inputs in the same stream. Complete score
arrays passed internal `memcmp` for all captured calls. This within-invocation
comparison is the causal control; independent process or binary runs remain
repeat-envelope measurements, not bitwise gates.

The exact iteration-3 UIDs, Euler matrices, and PPref were then replayed through
RECOVAR on the same physical GPU. RELION's row-major Euler matrices were
transposed only at RECOVAR's public projector API boundary. Across all 32 panel
particles, the five-field UID order and RELION/RECOVAR activity masks are
exact. Every active raw coordinate, texture coordinate, Hermitian sign, and
sampled complex64 register is bitwise exact. The deterministic float64 score
replay has zero centered cost delta and zero posterior total variation for all
32 particles. Particle 5676 contains 192 candidates and 5,704 active pixels
per candidate: all seven coordinate/sign planes have `0/1,095,168` mismatches,
sampled registers have `0/2,190,336` scalar mismatches, and posterior TV is
zero.

This exonerates the production native texture projector at this boundary; no
production texture change is warranted. Continue localization in upstream
operand generation and native reduction context. The comparison uses direct
array metrics and posterior TV; no correlation or map-quality claim is made.

Rejected diagnostics are retained but must not be cited as texture evidence:

- Continuing from `run_it002_optimiser.star` reset RELION's sampling RNG and
  produced iteration-3 perturbation `-0.24548`, not the uninterrupted
  `-0.30033`; its hypotheses were physically different.
- The first RECOVAR replay omitted the required Euler transpose at the public
  projection API boundary and therefore projected different orientations.
- Two partial iteration-1 launches tested ownership monitors only. One monitor
  allowed a foreign same-binary process; the next failed to filter
  `nvidia-smi` compute-app rows by GPU UUID. Neither produced iteration-3
  captures. The accepted launch used exact UUID filtering plus PID ancestry.

Sealed evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it23_uninterrupted_dual_corrected_8c832027_20260717T213908Z/analysis/uninterrupted_dual_recovar_replay.json` (SHA-256 `f64dcf488d7451131aaf22b257b060a1a86198898bafdb64bd439829c3bb8b51`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it23_uninterrupted_dual_corrected_8c832027_20260717T213908Z/provenance/uninterrupted_dual_postvalidated.txt` (SHA-256 `4441c39eba244e823e27049658314d19425a6e7a4bfde5e48dd43ba05a7679c3`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it23_uninterrupted_dual_corrected_8c832027_20260717T213908Z/provenance/uninterrupted_dual_artifacts.sha256` (SHA-256 `38dfdca43a66b9160b85e475a33d4bd4d9998b936f2b9ddd0a894e5e6c7a1ff9`)
- RELION dual binary SHA-256 `72c37a0c4efe58397d95255bc98ac2ff3b3784f98aaf9ea2ac49b801a6d37b90`; diagnostic RECOVAR library SHA-256 `fefa277fbcee3f90bde113353e88e76c0fe55aa9807230897cc0194b3022c39c`.

## 2026-07-17 sealed Iref-to-projector replay

Schema-v3 boundary captures now support an independent projector-construction
check. `scripts/parity/rebuild_relion_projector_capture.py` loads the sealed
float64 `Iref` operands and captured accelerator projectors, rebuilds through
RELION's actual `Projector::computeFourierTransformMap`, and compares exact
production complex64 arrays plus the pre-cast complex128 deltas. On the
64-particle iteration-3 boundary, both half-set rebuilds match every captured
complex64 element exactly; the maximum complex128-to-captured-complex64 deltas
are `3.42458e-9` and `3.63481e-9`. This closes projector construction given the
captured float64 operand. It does not make STAR/MRC an exact boundary because
the MRC copy is float32.

The full-10k iteration-3 capture gives the production-size control. Rebuilding
both projectors from their sealed float64 `Iref` arrays changes only 9 and 10
complex elements, respectively. Every changed real or imaginary component is
exactly one float32 ULP from the captured accelerator output (maximum complex64
absolute deltas `5.68e-14` and `2.27e-13`). This is classified as
precision/reduction-order variation rather than a projector-construction bug.

The same boundary's three-arm RECOVAR control/control/injected map comparison
remains a reduction-envelope diagnostic: merged-map FSC-AUC is
`0.999999997654` for control/control and `0.999999997631` for
control/injected. Pointwise injected residuals exceed the single observed
control envelope, while tau2 and Pmax remain inside it, so production-scale
causality is deferred to the sealed 10k three-arm run. Map gates remain
shellwise FSC/FSC-AUC; no correlation or exact discrete-winner gate is used.

Evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_recovar_projector_injection_smoke_audit_0f0e7256_20260718T021700Z/projector_iref_rebuild_v1.json` (SHA-256 `e084ca7be44d6a9a9b9d8ec6f408c8e1283bf423ba3b708ab8eb1ff70db0f79d`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it23_full10k_score_posterior_9b3c737c_20260717T231518Z/analysis/projector_iref_rebuild_11329370_capture_a.json` (SHA-256 `fb894fa8ce229f43545f3a1265bbea1f90a917f3117ccad573720e7d5a9307ae`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_recovar_projector_injection_smoke_audit_0f0e7256_20260718T021700Z/smoke_abc_report_v3.json` (SHA-256 `c64f673eb7a2f962ca04834f1cdcbc83ad603929865b3a7019d360e26cc71ead`)
- RELION replay binding SHA-256 `e04549190318244a62e7b85ea3200e221d37bc4543c4cc9811843f38075e4d22` (the resolved module path and digest are also sealed inside both rebuild reports).

## 2026-07-17 full-10k live-boundary closure

RELION job `11329370` completed a same-allocation capture/untouched/capture
trajectory on one GPU. All eight iteration/half capture-control map edges stay
inside twice the independently observed capture/capture repeat envelope. The
full-trajectory discrete controls also pass. Capture hooks are therefore inert
at this scale under FSC/FSC-AUC and exact control diagnostics; correlation is
not used. The three arms intentionally are not byte-identical, and candidate
counts (`2,100,576`, `2,100,480`, and `2,100,544`) are reported rather than
required to match exactly.

Independent job `11331272` repeated all 24 map reads with the canonical rounded
radial shells and normalized trapezoidal FSC-AUC, excluding DC and incomplete
Nyquist edges. All eight iteration/half control edges remain inside the A/C
repeat envelope. The worst control FSC-AUC loss is `5.44916e-6` against a
`1.11439e-5` bound; the worst shell loss is `2.95458e-5` against a
`5.93251e-5` bound.

The corrected boundary analyzer in job `11331054` reports 95 exact matches, two
lossless conversions, 26 STAR-decimal-serialization-limited comparisons, zero
fatal mismatches, and zero unresolved numeric differences for all 10,000
particles. Exact boundary equivalence remains blocked by nine absent runtime
operand families, not by an observed disagreement. Two analyzer-contract bugs
were found and fixed during fail-closed retries: the identity consumer required
an unused field absent from its producer schema, and it incorrectly required
RELION's coarse pass size to equal current size. The replacement comparison
calls RECOVAR's production scheduling helper and exactly reproduces the
captured `coarse_size=56` from `current_size=120`, HEALPix order 3, 1.6375 A
pixels, 256 box size, and 280 A particle diameter.

The controlled RECOVAR baseline/baseline/RELION-projector experiment was then
sealed against commit `83825ae881500c55d3617f8ad4585096d873625a` and the 38-file
source manifest below. Its first allocation, Slurm job `11331159`, completed
the first baseline's three-iteration refinement cleanly and then stopped in a
wrapper-only gate that searched stdout for Python logging written to stderr.
Offline finalization caught a second diagnostic-integrity defect before a
retry was submitted: half 2 contained 4,999 of the expected 5,000 identities,
with source row 4,573 absent. The missing image is the sole 1,024-rotation tail
bucket, which production evaluated as two rotation chunks while the compact
candidate tap existed only on the unchunked branch. This is a capture-path bug,
not an EM result or FSC failure.

The repair is committed as `4e2f3e7668426f5c2644c75a3de97dedafd2acb1`
with review hardening in `3a221f9305e48d648809370c4963181e6e53b9b0`
(cherry-picked here as `f4a75aa1` and `c673a787`). It retains and joins only
explicitly requested diagnostic chunks under a fail-closed Python-integer
memory bound; the disabled path converts no capture operands. Independent
review also found a genuine production control-state bug: rotation-chunked
winner-take-all probabilities were one-hot, but their returned Pmax remained
the pre-transform soft value. The corrected Pmax is one for a finite winner,
matching the unchunked per-class statistics. Strict first-iteration assembly
already replaces joint Pmax by one for every finite winner, so accepted
ave-Pmax schedules and convergence inputs were shielded; the audited accepted
runs also contain no iteration-1 rotation chunk. The M-step arithmetic was
already one-hot and is unchanged.
Winner-only effective support is now sealed consistently in both capture paths,
and the capture metadata explicitly distinguishes normalized scoring
posteriors from the separate RELION-float32 reconstruction weights.

The focused hardening set passes 15 tests and the broader capture/projector set
passes 96 tests. The wrapper now writes arm stdout and stderr synchronously,
replays them only after the child closes, and then evaluates its strict gates;
this removes both the wrong-stream check and the asynchronous-tee flush race.
The repinned, resealed three-arm job `11332272` completed its first baseline's
three-iteration refinement and repaired raw capture cleanly on one A100.  The
capture contains exactly 10,000 particles, 2,100,480 candidates, and 269
shards, including the 1,024-rotation tail.  The job then failed only in the
external finalizer: a Pandas string column was serialized as a NumPy object
array, while the fail-closed consumer correctly reopens every diagnostic NPZ
with `allow_pickle=False`.  The authoritative raw capture was preserved.

The finalizer now materializes Unicode arrays explicitly and rejects object
dtypes before every atomic NPZ write.  A full offline finalization against an
untouched hard-linked copy of job `11332272` passed: 5,000 identities per half,
all 10,000 source rows, all 2,100,480 candidates, and all 269 shards reopen
without pickle.  The complete three-arm auditor dry path also passed.  After an
independent post-arm package audit approved the stable source and wrapper
hashes, exactly one fresh retry was submitted as Slurm job `11332965`.

Job `11332965` completed all three arms and the fail-closed audit on one A100 in
42:02.  Every arm sealed all 10,000 identities, 269 shards, and 5,000 rows per
half.  Candidate counts were 2,100,480 and 2,100,544 for the two controls and
2,100,736 for the injected arm; support-cardinality equality is therefore not
assumed.  The iteration-3 RELION projector substitution is far outside the
single observed control envelope in both distributions and maps.  Maximum
posterior TV is `0.995891` against control `0.117947`; mean TV is `0.00208`
against `6.45e-5`.  Iteration-3 merged FSC-AUC loss is `1.57796e-4` against
control `5.95794e-8`; the half-1 and half-2 losses are `3.08897e-4` and
`9.03511e-5`.  Correlation is not computed.

The strict causal label nevertheless remains unresolved.  A pre-substitution
veto detected tiny cold-start repeat variation just above the single control
edge: iteration-1 `Ft_y` maximum absolute distance is `3.125e-6` for the
injected/control edge versus `2.589e-6` control/control, while its RMS is not
larger; iteration-2 half-1 map maximum absolute distance is only `1.084x` the
control distance and its FSC-AUC loss does not exceed both controls.  These
small early differences are not evidence against the large iteration-3
effect, but they prevent a same-boundary attribution.  The next diagnostic is
therefore a three-arm iteration-3-only replay from one byte-identical frozen
iteration-2 boundary, not another cold-start repeat or serial particle trace.

Evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it23_full10k_score_posterior_9b3c737c_20260717T231518Z/runs/full10k_abc_11329370/JOB_COMPLETE` (SHA-256 `db532fd5336ce83b6226ffd55c07b00f11c6e1944e28355e609f2904df64bcc2`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it23_full10k_score_posterior_9b3c737c_20260717T231518Z/runs/full10k_abc_11329370/analysis/full10k_abc_audit.json` (SHA-256 `093e57d87ffdb0687a2a82e927bb32f5d9b96d9270f83fffb9e91f7cfa29dfa9`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it23_full10k_score_posterior_9b3c737c_20260717T231518Z/runs/full10k_abc_11329370/analysis/full_trajectory_shellwise_fsc.npz` (SHA-256 `19230b7827187ff688c5503e0ed649881adb282f7816c9a3b3c6d603db50efc2`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it23_full10k_score_posterior_9b3c737c_20260717T231518Z/analysis/canonical_fsc_reaudit_11329370_20260718T031204Z/report.json` (SHA-256 `b136948224614e7337548a2e6241148eebb65d7f21657e21b13751d6f741e526`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it23_full10k_score_posterior_9b3c737c_20260717T231518Z/analysis/canonical_fsc_reaudit_11329370_20260718T031204Z/canonical_shellwise_fsc.npz` (SHA-256 `2fdf19dd47ffa2b2d758510b7ba1e405fa66bcf564ea3fe6178e3374c9622273`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it23_full10k_score_posterior_9b3c737c_20260717T231518Z/analysis/full10k_boundary_equivalence_11329370/report.json` (SHA-256 `dc4403cb84ceddc581d1a480a192d79baec44238e2829277b2a3e06af909178d`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it23_full10k_score_posterior_9b3c737c_20260717T231518Z/analysis/full10k_boundary_equivalence_11329370/ANALYSIS_COMPLETE` (SHA-256 `f9d19db21b133e722798b0164b0aa164c426464d0561981aab83aa326958baa8`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it3_projector_replay_ab_0f0e7256_prepared_20260718T015500Z/provenance/SOURCE_INPUT_SHA256SUMS` (SHA-256 `b3161a715026de9c8f94f5c7383b5788e47ea51b444e4e8cd69870a52b19267d`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it3_projector_replay_ab_0f0e7256_prepared_20260718T015500Z/logs/focused_chunked_capture_hardening_20260718T035703Z.log` (SHA-256 `8230f48ad5ae8f2d7826260110a0b7d238a87526f583c5ef7decce0466a4888f`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it3_projector_replay_ab_0f0e7256_prepared_20260718T015500Z/logs/broad_capture_projector_hardening_20260718T035900Z.log` (SHA-256 `9c9b0e6d40485e98fad10ff204db5536618cf6aeb6c07e4a1f927337df2bf2aa`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it3_projector_replay_ab_0f0e7256_prepared_20260718T015500Z/runs/recovar_projector_ab_11332272/baseline_control_1/diagnostics/candidate_capture/RAW_CAPTURE_COMPLETE.json` (SHA-256 `90546125d583266bc6babcab29a35a563479d7c8ffe3666fbb691906d62951a8`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it3_projector_replay_ab_0f0e7256_prepared_20260718T015500Z/logs/finalizer_offline_job11332272_20260718T043500Z.log` (SHA-256 `d161a38ec91fa2898c7511decfd7178a08e0c61ce5e38aa2c5daac7f03ac8c36`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it3_projector_replay_ab_0f0e7256_prepared_20260718T015500Z/validation/finalizer_job11332272_copy/FINALIZATION_REPORT.json` (SHA-256 `18f1600aaa747fe860fc5679a04c8bd770cbc05474d3dd4978485084870e4a00`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it3_projector_replay_ab_0f0e7256_prepared_20260718T015500Z/validation/finalizer_job11332272_copy/SOURCE_HALF_IDENTITIES.npz` (SHA-256 `311707c01e7e9db918008c01d8ea756c1feecf21b7013aaff15d5007c60d9ad1`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it3_projector_replay_ab_0f0e7256_prepared_20260718T015500Z/runs/recovar_projector_ab_11332965/analysis/projector_abc_audit.json` (SHA-256 `114b98702144cda03491a4d66c873d76090bae0e5b152f0b05e42566ca34500f`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it3_projector_replay_ab_0f0e7256_prepared_20260718T015500Z/runs/recovar_projector_ab_11332965/analysis/projector_abc_shellwise_fsc.npz` (SHA-256 `5e8bc8ba7d935114c988ae990b2c3599d6c2fc82a4d5898b2c62b239dc9d0f98`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it3_projector_replay_ab_0f0e7256_prepared_20260718T015500Z/runs/recovar_projector_ab_11332965/JOB_COMPLETE` (SHA-256 `0ee09db278ed86b26c963aab73c298d06cc3beb73ceb0a58d09a411df58a38c3`)

## 2026-07-18 frozen-boundary and robustness launch blocks

Independent review blocked the first iteration-2 frozen-boundary package before
Slurm submission.  The restart used local iteration zero for physical RELION
iteration three, so the ordinary process-start replay path replaced the sealed
per-half boundary noise with one broadcast noise array.  Relative to the frozen
boundary, the noise consumed by the local smoke differed by radial maximum/RMS
`7.6528/5.8615` for half 1 and `2073.699/249.636` for half 2.  The log first
reported frozen-boundary ownership and then reported the later replay override.
This is a real restart-state ownership defect, not a numerical-noise
classification.

The same review found that external replay STAR files overwrote the bundle's
poses and supplied other scoring state without all consumed artifacts being
sealed.  Frozen-versus-consumed poses differed by as much as `41.525` degrees
and `2.5` pixels.  The bundle also lacked image and scale corrections, and its
validator rewrote a reviewed report instead of requiring the regenerated
result to retain the reviewed digest.  Consequently the prior local A/A repeat
only demonstrated reproducibility of the wrong restart state.  No frozen A/A/B
science job was submitted.  The repair must preserve sealed per-half scoring
noise, establish one unambiguous owner for every scoring primitive, pin every
external STAR actually read, keep the reviewed validation result immutable,
reject dirty source including untracked files, and fail on nonfinite or
shape/dtype-invalid direct arrays.

A separate independent review blocked the K=1 robustness launcher before case
11 submission.  Its semantic acceptance checks and queue-time package
replacement defenses passed, but the dependency verifier did not require
`robustness_acceptance.json` and `ROBUSTNESS_ACCEPTANCE_COMPLETE.json` to be
unique members of the `JOB_COMPLETE`-anchored final-artifact manifest.  The two
files could therefore be coherently replaced after sealing.  The next package
must require both exact manifest members and matching hashes, reject duplicate
paths and paths escaping the sealed run root, and include negative tests for
absent, changed, duplicate, traversal, and symlink-escape cases.  No robustness
science job was submitted.

These reviews change packaging and restart-boundary validity, not the map
quality policy.  Direct maps remain gated only by shellwise FSC and normalized
FSC-AUC; correlation is not computed.

Evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_frozen_it2_projector_abc_prepared_20260718T052910Z/plan/INDEPENDENT_REVIEW_HANDOFF.md`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_frozen_it2_projector_abc_prepared_20260718T052910Z/validation/local_a100_control_scoring_smoke/run_full_refinement.err`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_projector_robust_package_hardening_reaudit2_20260718T120000Z/INDEPENDENT_REAUDIT_HANDOFF.md` (SHA-256 `80429ed77ae0d9ef1cd42fa3dc28966140fd26f07bf1a31c9a7c5f50de205c98`)

## 2026-07-18 frozen-boundary v2 and robustness pre-acceptance findings

The frozen iteration-2 restart was rebuilt with an incompatible v2 schema that
sealed a selected reconstructed-projector iteration-3 diagnostic state.  The
historical text overstated this as the full physical-iteration state; that claim is
withdrawn and superseded by the narrower schema-v3 fixed-arm contract below.
It never established identity to RELION's full in-memory iteration. Poses, per-particle
image and scale corrections, per-half direction priors and noise, translation
sigmas, maps, tau2/FSC/Pmax, sampling, perturbation, and convergence state are
sealed.  Replay slot zero is projector-only, external direction-prior reload
and process-start half-noise broadcast are suppressed, and every scoring array
is checked immediately before scoring by raw SHA-256, dtype, and shape.

The first corrected local A/A exposed a real dtype ownership bug: lossless
float64 radial noise profiles were expanded by JAX into float64 full-image
arrays, while the captured scoring arrays were float32.  Commit `cb7e23b4`
expands frozen noise explicitly in float32.  The replacement 10,000-particle
A/A then matches all 13 sealed pre-score field entries exactly in both arms.
Its supported-shell 1:60 FSC-AUC losses are `1.133149e-9` and `4.476982e-10`
for the two halves and `6.513029e-10` for the merged map.  One half-1 discrete
winner changes between identical controls; this is a diagnostic reduction
repeat, not a bitwise-equality failure.  Correlation is not computed.

The package is nevertheless not yet approved for the RELION-projector arm.
Independent review found an unsealed importable
`jobs/__pycache__/audit_helpers.cpython-311.pyc`; setting
`PYTHONDONTWRITEBYTECODE=1` prevents writes but does not prevent Python from
reading existing bytecode.  The package must remove compiled artifacts, reject
them both before submission and at runtime, add a tamper canary, and be
resealed before Slurm submission.  No frozen projector-B science job has been
submitted.

The K=1 robustness packages separately passed the repaired final-artifact
membership and dependency-seal gates.  Three launch-only defects were then
caught fail-closed without a RECOVAR quality result:

- job `11337711` rejected a malformed embedded SHA-256 before creating a run
  root;
- job `11338227` allocated one Slurm task for `mpirun -n 3` and stopped before
  RELION science;
- job `11338849` used the repaired three-task, four-CPU-per-task contract and
  completed RELION, but stopped before every RECOVAR arm because the capture
  validator's RELION-row particle IDs had been compared with source-fixture
  row IDs.

For job `11338849`, the captured half sets match zero-based rows of
`run_it000_data.star`, `run_it003_data.star`, and the final data STAR exactly.
The incorrect source-name-remapped expectation has total symmetric difference
`2964`; capture half counts are `1482` and `1518`.  The repair must keep both
identity namespaces and seal a bijective five-field UID crosswalk: RELION-row
IDs for capture closure and source-name/source-row identities for RECOVAR
half-set replay.  Identity closure must not be weakened, and downstream
robustness cases remain blocked.

K=4 job `11320255` completed all 15 planned 100,000-particle iterations on one
A100 in `10:02:27`, without OOM.  It remained unconverged and correctly skipped
final all-data.  The first dependent audit, `11337445`, was canceled because a
shared cross-node JAX CPU compilation cache produced host-feature mismatch
warnings.  Its partial output is non-authoritative.  Cache-isolated audit retry
`11338696` verifies a resealed 146-artifact input anchor and is the only audit
eligible to issue K=4 acceptance.  Until its shellwise FSC/FSC-AUC and GT
bundle seals successfully, the clean trajectory is not itself a quality pass.

Current evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_frozen_it2_projector_abc_prepared_20260718T052910Z/validation/local_a100_v2_float32_control_aa/AUDIT.json` (SHA-256 `da671a2db35fc25bf32d18c606b7efcd79b10c24a08f375b1c65c7da5774beac`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_frozen_it2_projector_abc_prepared_20260718T052910Z/validation/local_a100_v2_float32_control_aa/CONTROL_AA_SHELLWISE_FSC.npz` (SHA-256 `92c0913b7fa48558504084679957bcb395bfec4f8a0b82bd51cd2a4c0317ae6e`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_projector_robust_mpi_preflight_reaudit5_20260718T071842Z/INDEPENDENT_MPI_RESOURCE_REAUDIT_HANDOFF.md` (SHA-256 `5cad7f67056715f1a0ddc304f3ab5f5759fbc9e23b73acd002b74a2f2ff7a050`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_projector_robust_case11_83825ae_prepared_20260718T033237Z/runs/robustness_projector_abc_11338849`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_100k_restartfix_030f4a0b_20260717T211041Z/provenance/audit_cache_isolation_retry_11338696.txt`

## 2026-07-18 frozen projector causal result and corrected robustness launch

The final frozen-boundary package passed independent review after adding exact
top-level jobs allowlists, compiled-artifact rejection and tamper canaries,
schema-v1 rejection, and direct-array shape/dtype/nonfinite gates.  Slurm job
`11339646` then completed the serial control-A/control-B/RELION-projector
experiment on one A100 in `00:15:22`.  All three arms consumed byte-identical
sealed physical-iteration-3 scoring state: maps, tau2/FSC/Pmax, poses,
translations, image and scale corrections, per-half direction priors and
noise, translation sigmas, schedule, and perturbation.  The only intervention
was the projector manifest and data.

The RELION-projector effect is decisively above the observed A/A reduction
envelope.  Control versus intervention posterior-TV maxima are
`7.260852e-4` versus `0.9958911`, and mean posterior TV is `3.296e-5` versus
`2.0831e-3`.  Centered-score RMS mean is `1.668e-4` for A/A and `1.2711e-2`
for the intervention.  The maximum accumulator differences grow from
`2.8760e-5` to `4.8182e-2` for `Ft_y` and from `8.7137e-8` to `1.19397e-4`
for `Ft_ctf`.  The 34 changed winners among 10,000 particles are diagnostic
only and are not an acceptance gate.

The one-step maps remain close but are not inside the A/A envelope.  On shells
1--60, intervention FSC-AUC is `0.9997790883` for half 1,
`0.9999384005` for half 2, and `0.9998936311` for the merged map, while the
maximum A/A loss is `1.6587e-10`.  This proves a repeat-stable projector-path
mismatch at the frozen boundary; it does not yet identify operand generation,
geometry/index-coefficient generation, reduction order, or precision as the
source.  The next diagnostic adapts the existing common-contribution replay
to this current-head frozen boundary and compares float32/original order,
float32/canonical order, and float64/complex128 canonical order.  Serial
particle-by-particle debugging is not warranted unless the aggregate
comparison exposes a systematic subgroup.

The robustness repair preserves two distinct identity namespaces.  Compact
RELION captures use live data-STAR row IDs, whereas RECOVAR replay uses the
sealed source-name/source-row mapping; a bijective crosswalk and shuffled-order
canary prevent conflation.  Deep capture now selects RELION rows 0--15 and
fails before RECOVAR unless both live halves are present.  The corrected
validator passes the preserved 3,000-particle capture and all 111,680
five-field candidate UIDs.  After package, input, 128 embedded-hash, and exact
three-rank/four-thread scheduler gates passed independently, case-11 job
`11340226` was submitted.  Cases 18 and 21 remain gated on its sealed pass,
and case 22 remains gated on all three predecessor passes.

Evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_frozen_it2_projector_abc_prepared_20260718T052910Z/runs/frozen_projector_abc_11339646/analysis/frozen_projector_abc_audit.json` (SHA-256 `64c64507b5d0b66b9ab31aa3d2d95dbca879aea99a38480a44e2d34f50c72ee7`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_frozen_it2_projector_abc_prepared_20260718T052910Z/runs/frozen_projector_abc_11339646/analysis/frozen_projector_abc_shellwise_fsc.npz` (SHA-256 `7f2b634ab7f501bd8f83bbb256fafd0f84797cbae849f0ea5707e2b31768d181`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_projector_robust_identity_namespace_repair_20260718T075900Z/ROBUSTNESS_IDENTITY_NAMESPACE_REPAIR_HANDOFF.md` (SHA-256 `15df8ab1a74a10e8ede8c7fad6da3c8fc19c914972db6aca7b77ee87787e6960`)

## 2026-07-18 K=4 mixed reduction-mode oracle invalidation

The cache-isolated strict audit `11338696` completed and intentionally exited
nonzero because the scientific gate failed.  Control topology, restart
coupling, input anchors, schedules, convergence, and finalization all pass.
Direct class-map parity improves through physical iteration 12, where minimum
matched FSC-AUC is `0.99441`, then collapses at iteration 13 to `0.94597` and
ends at `0.93221`.  GT FSC-AUC remains close, so the failed direct trajectory
must not be relabeled as numerical noise or accepted by relaxing thresholds.

The collapse is caused by an invalid spliced RELION oracle.  The primary
iterations 1--11 launcher passed `--dont_combine_weights_via_disc`, selecting
the network segmented-pack path.  Its interrupted-run rescue continued from
iteration 11 without that parser-only option.  RELION does not persist the
choice as optimiser state, so iterations 12--15 reverted to the default
via-disc full-weight combination.  The source signature is exact:
`MlWsumModel::pack` derives `nr_groups` from the one-entry optics noise array,
so the network path combines only the optics prefix and leaves remaining scale
statistics follower-local, whereas the via-disc path combines all scale
groups.  RECOVAR correctly replayed the original follower-local mode.  It
therefore matches both followers' serialized iteration-11 scale vector exactly
before iteration-12 scoring, produces the original-mode follower-local scale
state after the M-step, and necessarily diverges from the mixed-mode RELION
iteration-12 model before maps split at iteration 13.

The mixed oracle and its failed strict audit remain immutable evidence.  A
controlled continuation from the exact iteration-11 boundary is being
regenerated with `--dont_combine_weights_via_disc` restored to verify the
existing emulation.  RELION's Class3D GUI defines `do_combine_thru_disc` with
default `false` and emits `--dont_combine_weights_via_disc` when it is false
(`pipeline_jobs.cpp` Class3D option construction and command generation).
Follower-local/network reduction is therefore the authoritative GUI-default
reduction target, not the via-disc path.  The existing 100k fixture is still a
controlled non-`firstiter_cc`, no-`ini_high` completion fixture; it must not be
mislabelled as the complete GUI-default command shape.  A later clean
GUI-default K=4 oracle must preserve `--dont_combine_weights_via_disc` and also
use the default non-absolute-greyscale path (`--firstiter_cc`) plus the default
60-A initial low-pass.  No oracle may splice reduction modes or infer
continuation behavior from the invalid trajectory.  Map acceptance remains
shellwise FSC/FSC-AUC only.

Evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_100k_restartfix_030f4a0b_20260717T211041Z/strict_k4_acceptance/k4_strict_acceptance_bundle.json`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_100k_restartfix_030f4a0b_20260717T211041Z/strict_k4_acceptance/k4_fsc_trajectory.json` (SHA-256 `e284cba863eb909de4cbc4f8fd66cd6e35bc003dce9d79e07cf1cb2580e727d6`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_100k_restartfix_030f4a0b_20260717T211041Z/strict_k4_acceptance/k4_fsc_trajectory_shellwise.npz` (SHA-256 `48cb5a2f4be19bbc87f66c0f32702aba13f5cb740d8967e033e6ecc148609cf9`)

## 2026-07-18 frozen projector source/construction closure

The production-size A/A'/B factorization closes the physical-iteration-3
projector implementation and moves the earliest remaining boundary upstream to
the incoming reconstructed reference.  A' rebuilds a projector from RECOVAR's
own serialized map and reruns the same 10,000-particle frozen scoring boundary;
B uses the sealed RELION projector.  Relative to A, A' posterior-TV mean,
95th percentile, and maximum are `4.34798e-5`, `1.23037e-4`, and
`7.02824e-3`, while B gives `2.08306e-3`, `4.10242e-3`, and `0.995891`.
The B/A' effect ratios are `47.91` for mean posterior TV, `33.34` at the
95th percentile, and `61.34` for mean centered-score RMS.

On supported shells 1--60, A' FSC-AUC losses are `1.14919e-9`,
`2.08116e-10`, and `5.26143e-10` for half 1, half 2, and the merged map.  The
corresponding B losses are `2.20912e-4`, `6.15995e-5`, and `1.06369e-4`, up
to `192232x` the A' loss.  Rebuilding B from the captured float64 Iref changes
only 9 and 10 of 7,203,978 complex voxels; each changed component is at most
one float32 ULP.  Common complex128 replay and genuinely recomputed float64
source operands retain at least `99.924%` of the production effect.  Projector
construction, indexed geometry, reduction ordering, and production precision
are therefore rejected as material explanations.  The next aggregate probe is
the physical-iteration-2 BPref/numerator-to-reconstructed-reference boundary;
serial particle tracing is not warranted.

The companion five-field UID inventory uses
`[class, coarse_rotation, coarse_translation, fine_rotation_global,
fine_translation_global]`.  A versus A' has 64 left-only candidates, no
right-only candidates, and only seven significant common-UID count
differences across 10,000 particles.  A versus the RELION-source arm has 2,400
left-only and 2,656 right-only candidates, with exclusive posterior-mass
maxima `0.988049` and `0.995891`.  The 34 changed winners are reported only as
a diagnostic; they are not a parity gate.

Evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_frozen_it2_projector_source_constructor_factorial_20260718T083027Z/analysis/aprime_factorial_v3_adjudication.json` (SHA-256 `a3b532aa6ec5db99f759c19a0c04b685355fc5924d2f57e6d3cc83f507ec437d`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_frozen_it2_projector_source_constructor_factorial_20260718T083027Z/analysis/APRIME_FACTORIAL_V3_COMPLETE.json` (SHA-256 `32534753bf621b504038d8e9dc5ee3def0c1870dddb629cc4f86112e6e1e0943`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_frozen_it2_projector_source_constructor_factorial_20260718T083027Z/analysis/aprime_aggregate_v4_seal.json` (SHA-256 `3ad0a0c4d589ce144ddd1f44d1ac155420a3f5e2791779c826c31dcf8833eddf`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_frozen_it2_projector_source_constructor_factorial_20260718T083027Z/analysis/APRIME_AGGREGATE_V4_COMPLETE.json` (SHA-256 `0e2d939a811e1955273c1faabdd885573dcb6a5ffa4faee6c8c8de6a777f9877`)

## 2026-07-18 robustness case-11 acceptance-contract failure

Slurm job `11340226` completed RELION plus all three RECOVAR arms and sealed
the three-arm projector audit, then failed only in the outer robustness
acceptance.  All arms converge on physical iteration 10.  Every numbered
half/merged cross-engine FSC-AUC is above `0.9978`; the three final merged
cross-engine FSC-AUC values are approximately `0.99586`.  The sealed projector
audit remains fail-closed as
`unresolved_within_or_overlapping_observed_control_envelope`, because tiny
pre-injection repeat differences slightly exceed the one observed
control/control edge.  It does not authorize a production change.

The outer failure is a product-matching defect, not permission to lower its
registered `0.995` threshold.  It compares RECOVAR's Wiener-regularized
`final_half1.mrc` and `final_half2.mrc` against RELION's explicitly unfiltered
`run_half1_class001_unfil.mrc` and `run_half2_class001_unfil.mrc`; the resulting
FSC-AUC values are `0.9791265` and `0.9797161` in every arm.  Final half-map
acceptance must compare like products by materializing RECOVAR unfiltered
halves from the sealed final accumulators, or remain non-gating while the
matched product is absent.  The final merged and numbered gates remain
unchanged.  Cases 18, 21, and 22 remain blocked until the corrected acceptance
contract is sealed.

Evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_projector_robust_case11_83825ae_prepared_20260718T033237Z/runs/robustness_projector_abc_11340226/analysis/projector_abc_audit.json` (SHA-256 `4cb75988cff599324b96e628451b4c1fd2ddedb951e9aa1a4ca59e5a9ed8a7c9`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_projector_robust_case11_83825ae_prepared_20260718T033237Z/runs/robustness_projector_abc_11340226/analysis/PROJECTOR_ABC_AUDIT_COMPLETE.json` (SHA-256 `279672665b94da03f3553d8915528082e3ff51c1bd24553bcda43014a706df7a`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_projector_robust_case11_83825ae_prepared_20260718T033237Z/logs/k1_rob11_proj_abc_11340226.err`

### Matched-unfiltered diagnostic exposes pre-join ownership bug

Job `11341828` reconstructed `do_map=false`, no-spherical-mask half maps from
the final RECOVAR BPref dump and compared only shellwise FSC/FSC-AUC against
RELION's unfiltered halves. It completed cleanly, but the half FSC-AUC values
are only `0.903035`/`0.903937` with RECOVAR grid correction off and
`0.868995`/`0.870136` with it on. Thus the earlier `0.979` result was not only
a regularized-versus-unfiltered naming error, and the `0.995` acceptance gate
remains unchanged.

A concrete product-ownership defect occurs across RELION's low-resolution
half join. At convergence RELION calls
`writeTemporaryDataAndWeightArrays()` before
`joinTwoHalvesAtLowResolution()`. It later reconstructs each
`run_half*_class001_unfil.mrc` by rereading those saved pre-join BPref arrays.
RECOVAR's diagnostic/output path instead used the post-join arrays. The final
loop now preserves explicit pre-join numerator/weight arrays for unfiltered
half output while retaining post-join arrays for final FSC, tau2, and joined
reconstruction. This is not yet claimed as the sole cause: the failed curves
plateau near `0.85` (grid off) or `0.80` (grid on) across shells 30--60, well
beyond the low-resolution join support.

Corrected rerun `11342491` validates the new boundary and fails closed at the
unchanged threshold. Pre-join half FSC-AUC is `0.903900`/`0.904756` with grid
correction off and `0.869346`/`0.870440` with it on. Relative to the post-join
diagnostic, preserving the right arrays improves the gated halves by only
`0.000350344` and `0.000305094`. The join changes exactly 91,965 of
17,373,979 entries (`0.529326%`) in each per-half numerator and weight array;
it is a real ownership fix but not the dominant residual. Production-written
and independently materialized grid-on maps agree at FSC-AUC above
`0.99999999998`, ruling out the output loader/materializer. The observed
minimum gated FSC-AUC is `0.869346 < 0.995`, so cases 18, 21, and 22 remain
blocked pending capture job `11342527` and its dependent physical-iteration-2
BPref source-by-reconstruction factorial `11342699`, not another full case-11
rerun.

Evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_final_unfiltered_match_20260718T091500Z/run/job_11341828/materialized/unfiltered_half_match.json`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_final_unfiltered_match_20260718T091500Z/run/job_11341828/materialized/unfiltered_half_shellwise_fsc.npz`
- Slurm job `11341828`, state `COMPLETED`, exit code `0:0`, elapsed `00:10:29`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_final_unfiltered_prejoin_match_20260718T092527Z/run/job_11342491/materialized/unfiltered_half_match.json` (SHA-256 `6d3ac519e9b48437ba8f58640d32c8c709e05320db9ccd0435813cace9254501`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_final_unfiltered_prejoin_match_20260718T092527Z/run/job_11342491/materialized/unfiltered_half_shellwise_fsc.npz` (SHA-256 `62d1de090afd006d44a623620cd0b826dcd53f58b927a37932f81b5c3d693737`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_final_unfiltered_prejoin_match_20260718T092527Z/analysis/job_11342491/FAILURE_ADJUDICATION.json` (SHA-256 `d8ed7aeb69626ce981e2a263b2260179ca763f50ab208bb6e441e6d10ed6030e`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_final_unfiltered_prejoin_match_20260718T092527Z/analysis/job_11342491/FAILURE_ADJUDICATION_COMPLETE.json` (SHA-256 `fcb3598ac1807a4b49bc819d789c74fcbbd004f44872470033e48c1edd687e8c`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_final_unfiltered_prejoin_match_20260718T092527Z/analysis/job_11342491/FAILURE_ADJUDICATION_ARTIFACTS.sha256` (SHA-256 `8744318afc7f296d0d36525d665db661df98f64c450f27fa7358e33dd7e1afe2`)
- Slurm job `11342491`, state `FAILED`, exit code `1:0`, elapsed `00:09:40`; this is the intended fail-closed scientific result. Canceled job `11342385` is invalid packaging evidence only.

## 2026-07-18 physical iteration-2 native-BPref factorial

Capture job `11342527` and dependent factorial job `11342699` completed on the
same A100 and close the reconstruction-engine axis at the physical iteration-2
boundary.  The factorial fixes the exact captured per-half RELION tau2 across
all four source-by-reconstructor cells, with current size 92, `r_max=46`,
padding factor 2, trilinear interpolation, `skip_gridding=true`, and
`minres_map=5`.  Switching only the reconstructor gives half and merged
supported-radius FSC-AUC `1.0` for both the RELION and RECOVAR BPref sources.
The RELION binding also reproduces the immediate native post-reconstruction
Iref with supported-radius FSC-AUC `1.0` and minimum supported-shell FSC at
least `0.9999999999999977`; RECOVAR accumulator frame round-trips are exact.

The remaining map difference follows the native BPref source.  With either
reconstructor fixed, RELION-source versus RECOVAR-source supported-radius
FSC-AUC is `0.9999947157`/`0.9999987768` by half and `0.9999977791` merged.
The uninterrupted live RECOVAR-versus-RELION values are respectively
`0.9999947096`/`0.9999987716` and `0.9999977739`, so the factorial reproduces
the live residual.  After exact frame alignment, RELION-versus-RECOVAR raw
BPref relative L2 is `0.00585173`/`0.00271445` for the numerator and
`0.00119718`/`0.000475082` for the weight.  Construction, current-size support,
tau scaling, map frame conversion, gridding, and mask semantics are therefore
rejected as material explanations at this boundary.

The existing package does not, however, contain the exact incoming physical
iteration-2 RELION state needed for the next substitution.  It has 92 live
state files per rank only for `state_iter3` and no `state_iter2` live state.
The iteration-1 STAR/MRC checkpoint is complete as a serialized product, but
its maps are float32 and its spectra, half-local noise/norm/prior fields,
particle state, and perturbation are decimal-serialized; native iteration-2
dispatch and contribution topology are also absent.  A substitution of
`run_it001_half{1,2}_class001.mrc` into an otherwise frozen RECOVAR state is a
valid **serialized-map-only intervention**, not an exact RELION boundary.

An aggregate common-state accumulation replay claiming an exact incoming
iteration-2 boundary therefore requires a fresh uninterrupted target-2 live
capture, including exact posteriors, contribution identities/geometry, and
native order.  It should then compare float32 original/canonical order with
float64/complex128 canonical controls to separate operand generation from
reduction precision/order.  Do not return to serial particle tracing unless
that aggregate replay identifies a systematic subgroup.

This is a fixed-common-RELION-tau factorial, not a formal live RECOVAR A/A
closure, and `FACTORIAL_COMPLETE.json` records computational completion rather
than a parity threshold.  The four production-map cells use the same outer
solvent mask; the separate unmasked RELION-binding/live-Iref comparison is the
constructor closure.  Intermediate evidence uses exact/direct array metrics;
maps use shellwise FSC/FSC-AUC only.  Correlation is neither computed nor used.

Evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it2_bpref_reconstruction_factorial_20260718T100000Z/runs/capture_pair_11342527/analysis/it2_native_bpref_factorial.json` (SHA-256 `182df85587f8db78b26a12bb14917c781d5717eb405d91666938659575c8c1f3`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it2_bpref_reconstruction_factorial_20260718T100000Z/review/INDEPENDENT_REVIEW.md` (SHA-256 `07ffd7d8b0651405ee0a510f3acdc56607291e0da9be96ceb9de8044fa3d91f1`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_it2_bpref_reconstruction_factorial_20260718T100000Z/review/INCOMING_IT2_BOUNDARY_AUDIT.md` (SHA-256 `e2a1d9f13b76c8dd23b40de5e0d329bc696e3f6eb2594160199dc340b88f259f`)
- Slurm jobs `11342527` and `11342699`, both `COMPLETED` with exit code `0:0`.

## 2026-07-18 K=4 GUI first-iteration resolution-state repair

The clean GUI-default Class3D oracle exposed a real control-state mismatch at
the first autonomous K=4 boundary.  RELION's
`MlOptimiser::updateCurrentResolution()` uses
`ROUND(ori_size * pixel_size / ini_high)` during physical iteration 1 when
`--firstiter_cc` is active.  For the 100k/256 fixture this is shell 9
(`60.444444 A`).  RECOVAR instead retained the live class data-vs-prior shell
10 (`54.40 A`).  Although both programs entered iteration 1 at current size
38, the wrong stored shell made pre-fix job `11343734` schedule and enter
physical iteration 2 at size 40 instead of RELION's size 38.  This is an
algorithm/control mismatch, not numerical noise.

Commits `b670f9bf` and `c390f8bf` make the physical-iteration-1 `ini_high`
shell class-count independent and use it for both the next-size decision and
the stored resolution/convergence state.  Clean detached autonomous job
`11344147` confirms the repair without any per-iteration RELION state replay:
physical iteration 1 stores shell 9 and reports `60.44 A`, its native
iteration-2 decision is raw/quantized size 38, and iteration 2 starts at size
38.  The sealed log prefix contains no replay override and no fatal fault.

The corrected iteration-1 maps also pass the map gate.  Identity class
matching is `[1, 2, 3, 4]`; normalized 127-shell FSC-AUC against the RELION
oracle is `0.9999934704`, `0.9999954881`, `0.9999950882`, and
`0.9999999006` for classes 1--4.  Only shellwise FSC/FSC-AUC is used; no
correlation metric is computed.

The completed iteration-2 causal triangle shows that the repair affects map
quality, not only a displayed resolution or a transient size.  The pre-fix
autonomous arm processed iteration 2 at size 40 and then rejoined the oracle
size-42 schedule at iteration 3, but its per-class FSC-AUC was
`[0.99692609, 0.99826373, 0.99728669, 0.98997695]`; class 4 fails the
unchanged `0.995` gate.  The corrected autonomous arm processes iteration 2
at size 38, matches the shell-11/`49.45 A` state and native size-42 iteration-3
start, and obtains `[0.99999143, 0.99999076, 0.99999052, 0.99999685]`.
Its minimum agrees with the controlled oracle-state component arm
(`0.99998980`) while remaining independently autonomous.  The corrected
class-4 improvement over the pre-fix control is `0.01001990` FSC-AUC.

At iteration 3 the repaired autonomous trajectory remains on the exact
size-42 to size-56 schedule and passes every map gate.  Its per-class FSC-AUC
is `[0.99983851, 0.99971160, 0.99964188, 0.99922248]`; corrected class 4 is
`0.05309456` above the pre-fix autonomous arm and only `0.00012988` below the
controlled oracle-state component.  The pre-fix class-4 map had continued to
worsen to `0.94612792` even though its discrete size schedule had rejoined the
oracle, confirming that the wrong iteration-2 size caused persistent map
damage rather than a harmless transient.

The remaining strict iteration-3 mismatch is average posterior maximum, not
map quality or scheduling.  RECOVAR reports exact `0.265575` (display
`0.2656`) versus the RELION particle-table mean `0.26535582661` (display
`0.2654`); the controlled component also differs at `0.265522`.  Corrected and
controlled hard hypotheses agree for `99,979/100,000` particles.  Because the
controlled arm retains essentially the same Pmax offset, the residual is
localized to within-iteration score/posterior or aggregation arithmetic, but
is not classified as numerical noise without score/posterior and float64/order
controls.

Iteration 4 strengthens the causal closure.  The corrected autonomous arm
matches current size 56, shell 20/`27.20 A`, the nonconverged state, and the
native size-60 iteration-5 start.  Owner dispatch, perturbation, HEALPix order,
class-prior tolerance, provenance, and all map gates pass.  Its identity-class
FSC-AUC is `[0.99888334, 0.99819060, 0.99872446, 0.99868965]`, within
`0.00079405` of the controlled component
`[0.99940351, 0.99885203, 0.99917868, 0.99948369]`.  The pre-fix arm has
broadened damage at `[0.97960769, 0.98920139, 0.97892265, 0.91596351]`.
The corrected-minus-pre-fix gains are therefore
`[0.01927565, 0.00898921, 0.01980181, 0.08272613]` FSC-AUC.  This is persistent
causal evidence for the first-iteration resolution-state repair, not a new
repair target.

Average Pmax remains the sole strict iteration-4 diagnostic mismatch:
corrected is `0.597346`, controlled is `0.597545`, and the RELION
particle-table mean is `0.59754980937`.  The aggregate mean does not classify
the residual as numerical or algorithmic.  Per-particle score, log-evidence,
posterior, and float64/order controls remain required after the active science
load finishes.  Pmax is not substituted and does not replace FSC/FSC-AUC as
the map-quality gate.

This causally closes map and schedule parity through the first four
autonomous boundaries while leaving the small Pmax state residual fail-closed.
Job `11344147` remains running for the full autonomous trajectory,
convergence, final FSC/FSC-AUC, and Pmax-classification gates.

Focused CPU validation after the repair:

- `pytest -q tests/unit/test_refine_relion_mode.py -k firstiter_cc`: 9 passed,
  314 deselected.
- The preceding scheduling patch also passed
  `tests/unit/test_refine_relion_mode.py` and
  `tests/unit/test_resolution_criterion.py` together.

Evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_100k_gui_cc_nodisc_autonomous_030f4a0b_20260718T104146Z/analysis/pre_fix_iteration_001_boundary_adjudication_v1.json`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_100k_gui_cc_nodisc_autonomous_c390f8bf_20260718T110620Z/analysis/corrected_iteration_001_native_boundary_evidence_v1.txt` (SHA-256 `c7120b0d2b036a95bcc5a97f4677a55c4a5444ed9b50dce96e900a63c4fc1988`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_100k_gui_cc_nodisc_autonomous_c390f8bf_20260718T110620Z/analysis/corrected_iteration_001_native_boundary_adjudication_v1.json` (SHA-256 `209244818bfea67d0422bbab44a36a92ccd455fec35a807865e16c35d04f5f04`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_100k_gui_cc_nodisc_autonomous_c390f8bf_20260718T110620Z/analysis/corrected_iteration_001_map_fsc_audit_v1.json` (SHA-256 `bd4b4bf6f572f150101dcf72d2e187fb49e7242a7e29f7b5f97c66dfc141f03d`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_100k_gui_cc_nodisc_autonomous_c390f8bf_20260718T110620Z/analysis/corrected_iteration_001_shellwise_fsc_v1.npz` (SHA-256 `ee2f4037ced98316685efc1e1dd8f83cdaf315afff4fdeaddd706f9669e5e21d`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_100k_gui_cc_nodisc_autonomous_c390f8bf_20260718T110620Z/analysis/corrected_trajectory_iteration_002_audit_v2.json` (SHA-256 `ffcfcd70a4aad7c03a5ffb1602382ec3c3b74f8c89e1d55fa94e2c47649cd978`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_100k_gui_cc_nodisc_autonomous_c390f8bf_20260718T110620Z/analysis/corrected_trajectory_iteration_002_shellwise_fsc_v2.npz` (SHA-256 `0c863c6ddd82d6f3b6a22d22bba7167cfb38ab3c30924214d8b0ed0abe3c2d90`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_100k_gui_cc_nodisc_autonomous_c390f8bf_20260718T110620Z/analysis/iteration_002_causal_triangle_adjudication_v1.json` (SHA-256 `e8ea6ec077e52dc4d92968df4320bc735b30158e8d7528d02e825279270769ae`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_100k_gui_cc_nodisc_autonomous_c390f8bf_20260718T110620Z/analysis/corrected_trajectory_iteration_003_audit_v2.json` (SHA-256 `4be9db4bc84157c9de1e2a0c3f709308e567db6ac065419e85566e2083340856`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_100k_gui_cc_nodisc_autonomous_c390f8bf_20260718T110620Z/analysis/corrected_trajectory_iteration_003_shellwise_fsc_v2.npz` (SHA-256 `5c0ee82203b58aeca7389f393b726624dd2a3688443d9a9f254ce37ded0e295e`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_100k_gui_cc_nodisc_autonomous_c390f8bf_20260718T110620Z/analysis/iteration_003_causal_triangle_adjudication_v1.json` (SHA-256 `93ab9feb5a0c8c7147fdb21a625c1b92ddf8262f4e6075d71e47b98a39c56645`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_100k_gui_cc_nodisc_autonomous_c390f8bf_20260718T110620Z/analysis/iteration_002_hard_hypothesis_causal_diagnostic_v1.json` (SHA-256 `1074fc791931663952d05495ba90d59b98225139339b83ce78b851f696b97fb3`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_100k_gui_cc_nodisc_autonomous_c390f8bf_20260718T110620Z/analysis/corrected_trajectory_iteration_004_audit_v2.json` (SHA-256 `dd4a870ba9503f1ce8736164b5bc77310900f8faea7bf5951d8ff2bc4e523e5d`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_100k_gui_cc_nodisc_autonomous_c390f8bf_20260718T110620Z/analysis/corrected_trajectory_iteration_004_shellwise_fsc_v2.npz` (SHA-256 `80b18129bf3dac5b1bad20501de84cf76cdd029bbdb401577fa6ee32c964921a`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_100k_gui_cc_nodisc_autonomous_c390f8bf_20260718T110620Z/analysis/iteration_004_causal_triangle_adjudication_v1.json` (SHA-256 `1bb8734e21cf328bee70556b4da52829f0b45ba3b2cdb21c0123b579d70c9da1`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_100k_gui_cc_nodisc_autonomous_030f4a0b_20260718T104146Z/analysis/pre_fix_autonomous_iteration_004_map_fsc_audit_v1.json` (SHA-256 `6e9770b3ba1d46663bb9f1d0d23412fa2d83ec56c835a130b2588eefd90f8b60`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_100k_gui_cc_nodisc_autonomous_030f4a0b_20260718T104146Z/analysis/pre_fix_autonomous_iteration_004_map_fsc_audit_v1_shellwise_fsc.npz` (SHA-256 `f0a3fbddb6ef3e18bc0b482df33d628318de370861ce120797e3bcbd14778ae0`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_100k_gui_cc_nodisc_recovar_030f4a0b_20260718T091559Z/analysis/controlled_oracle_state_iteration_004_map_fsc_audit_v1.json` (SHA-256 `9da552c1e843059a1489f26fb283b40b9301a0f4d7de160aa0637aa0f8a6a30f`)
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_100k_gui_cc_nodisc_recovar_030f4a0b_20260718T091559Z/analysis/controlled_oracle_state_iteration_004_map_fsc_audit_v1_shellwise_fsc.npz` (SHA-256 `4d095e5d05424fed3f56f3ad4068dfbc7483689145cc66ceff59973abb18e0c1`)

## 2026-07-18 split-half optimizer-Pmax workflow correction

A source audit identified a real K=1/K-class control-state mismatch. In
split-half MPI refinement, RELION computes `ave_Pmax` independently for each
half, divides the half-1 Pmax numerator by half 1's retained M-step posterior
mass (`sum(wsum_model.pdf_class)`), and broadcasts rank 1's scalar for shared
image-size scheduling. RECOVAR instead averaged raw particle Pmax across both
halves and divided by particle count. The two-half arithmetic mean is not the
optimizer oracle.

Commit `8e7ce8af` implements the half-1 retained-mass scalar, passes it
explicitly into convergence/scheduling, preserves both-half particle arrays as
diagnostics, and records `ave_Pmax_denominator_trajectory`. Asymmetric-half
tests cover both the source-half and denominator rules. Focused tests pass
6/6, and the full convergence unit module passes 82/82.

This correction is required for exact workflow parity and latent threshold
cases, but it does not explain the corrected autonomous K=4 iteration-8 map
failure. Before iteration 8, old and corrected Pmax values take the same active
current-size branches and no convergence topology splits. Independent
score/posterior, backprojection, and float64/order classification therefore
continues; FSC/FSC-AUC remains the map-quality gate.
## Fixed-arm frozen-boundary diagnostic

The reusable fail-closed fixed real-10076 K=1 physical-it2 diagnostic arm,
deterministic finalizer, source ownership, and captured-Iref lineage contract are documented in
[`frozen_boundary_v3.md`](frozen_boundary_v3.md). Schema v2 remains historical
and cannot support the fixed-arm claim. Schema v3 seals an explicitly
enumerated reconstructed-projector diagnostic state; it does not claim identity
to RELION's full in-memory physical iteration.

## 2026-07-19 reusable exact-local BPref accumulator replay

`scripts/replay_bpref_contribution_bundle.py` now turns a complete strict v3
row-capture boundary into repeat, order, and precision controls. It runs the
captured operands through RECOVAR CUDA in complex64/float32 and
complex128/float64, and through RELION's CPU double BackProjector, in both
captured execution order and one common semantic order. Intermediate outputs
use exact array metrics; the shared unregularized reconstruction control uses
shell FSC/FSC-AUC only. The report automatically classifies the resolved scale
as repeat variation, reduction order, scatter precision, geometry/backend
arithmetic, or unresolved.

The real case-26 iteration-7/half-2 capture contains 37,376 rows and genuine
complex128/float64 upstream data/weight operands. Common-canonical RECOVAR and
RELION float64 accumulators close at relative L2 `1.13e-14`. Float32 versus
float64 differs by `0.04270`/`0.04050` in data/weight relative L2, well above
float32 repeat (`1.44e-7`) and order (`9.60e-7`) controls, while its reconstructed
map remains above the parity gate at FSC-AUC `0.998122282`. The boundary is
therefore classified as scatter precision, not an algorithmic geometry mismatch.

Schema v3 does not record production packed-zero rows, so its GPU execution
mode preserves active-row shard partitions but is not labeled bitwise production
launch replay. The clean report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_bpref_accumulator_replay_5bb40f18_20260719T071500Z/output_v3/bpref_accumulator_replay_report_v1.json`
(SHA-256 `b668b011791f0e5951f53346b8779566ff92c66164ce4648b173dd7778330455`).

## 2026-07-19 K=1 iteration-5 resident-state causal localization

Fail-closed accumulated-state and strict single-step controls separate the
case-20 iteration-5 residual from same-GPU repeat variation. The strict
all-RELION arm repeats particle Pmax, poses, translations, and support exactly;
its merged-map repeat FSC-AUC is `0.999999999832`. In the accumulated
trajectory, restoring only the RECOVAR scoring references at iteration 5
reproduces nearly the complete full-state residual: merged FSC-AUC versus the
all-RELION arm is `0.999993939`, Pmax absolute p95 is `0.007577`, and support
differs for 185/3,000 particles. Restoring tau2 and noise alone is negligible:
merged FSC-AUC `0.999999999370`, Pmax p95 `9.96e-5`, and three support
differences.

The strict arm replays RELION references through iterations 1--5. Restoring
the resident RECOVAR reference only at iteration 5 then remains essentially
exact: output-versus-RELION merged FSC-AUC is `0.999999999035` and the GT
FSC-AUC delta is `-3.35e-8`. The dominant residual therefore accumulates
through prior RECOVAR-produced references; it is not a broad single-step
scoring, tau2, or noise mismatch. Together with the native-BPref reconstructor
closure and canonical contribution replay above, the next locus is aggregate
GPU contribution accumulation precision/order, not serial particle decisions.

State-swap controls now expose the half-specific K=1 model-STAR scales as an
explicit live scorer oracle. Generic shared Class3D model STARs fail closed
because they do not identify two exact scorer-owned scales. The particle-state
audit also reports two distinct Pmax scalars: per-particle mean versus
per-particle mean, and RECOVAR M-step `ave_Pmax` versus RELION
`rlnAveragePmax`. Mixing those semantics produced an invalid scalar delta; the
existing per-particle array/distribution comparison was unaffected.

Evidence:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_accumulated_it5_factorial_27e3cfb2_20260719T013500Z/EARLY_RECOVAR_MAPS_CAUSAL_RESULT.md`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_single_step_it5_factorial_27e3cfb2_20260719T013000Z/EARLY_STRICT_RECOVAR_MAPS_RESULT.md`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_single_step_it5_factorial_27e3cfb2_20260719T013000Z/audit/repeat_control_summary.json`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_single_step_it5_factorial_27e3cfb2_20260719T013000Z/audit/incremental_summary.json`

## 2026-07-19 K=1 iteration-4 incoming-map amplification

A same-A100 two-arm intervention localizes the earliest recurrent case-20
physical-iteration-4 gap to the incoming half-reference maps. Both arms replay
the complete RELION state and RELION references through iteration 4; the
intervention restores only the resident RECOVAR-produced references at the
iteration-4 boundary. With exact RELION inputs, iteration-4 merged FSC-AUC is
`0.999999998921`. With the resident maps, it is `0.999997257810`, and the two
arms compare directly at `0.999997260559`.

The resident iteration-3 input itself is extremely close to RELION:
`0.999999999076` merged FSC-AUC. Descriptively, the `1 - FSC-AUC` deficit grows
by about 2,967-fold across the next expectation/M-step. Arm-to-arm Pmax
absolute p95 is `1.226e-4`, and significant-support counts remain exact for
2,996/3,000 particles. This rules out a missing broad iteration-4 state field:
the small prior map residual is sufficient to reproduce the autonomous output
gap. Target-only iteration-4 float64 does not close it, so cumulative early
precision and aggregate contribution-order replay are the next discriminators.

The sealed report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_it4_exactref_maps_ab_local_a100_1e208826_20260719T024600Z/audit/incoming_maps_ab_summary.json`
(SHA-256 `03a76ea6e1c594194e2b08de87bcc9ae9a116d7d4a79d38f2ca3cee1d97b30cd`).
Map quality uses FSC/FSC-AUC only; correlation is not computed.

The matching physical-iteration-4/half-1 schema-v3 BPref capture provides a
stronger precision discriminator. Its 34,456 active rows retain genuine
complex128/float64 upstream operands. Under one common canonical order,
RECOVAR GPU and RELION-style CPU float64 accumulators close at relative L2
`1.38e-14`. In contrast, canonical GPU float32 versus float64 differs by
`0.06407` relative L2, far above float32 repeat (`9.37e-8`) and order-only
(`3.77e-7`) controls. The corresponding unregularized maps compare at FSC-AUC
`0.996829235`; common float64 GPU-versus-CPU map FSC-AUC is `1.0`.

This boundary is classified as `scatter_precision`, ruling out a
geometry/backend-arithmetic mismatch for the captured RECOVAR contribution
list under common canonical float64 replay. It is not yet a native
RECOVAR-versus-RELION contribution comparison: native RELION operands were not
captured, and schema v3 omits packed-zero production rows. The decisive next
control is therefore a live same-GPU trajectory that promotes only the x-half
M-step accumulator while leaving scoring and posterior arithmetic unchanged.
The replay report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_it4_bpref_capture_h1_d88090ec_20260719T031000Z/audit/replay/bpref_accumulator_replay_report_v1.json`
(SHA-256 `88e58c84320fd746b224c5c887d3df6e33c9c1b40efbcd2572a2f146c0bc1633`).

A separate same-A100 control promotes scoring and projections across both
adaptive passes and all four early iterations, while deliberately leaving the
x-half M-step accumulator at its production complex64/float32 default. It
changes the iteration-4 merged map by only `7.74e-10` FSC-AUC deficit
(`0.999999999226` direct f32-versus-f64 FSC-AUC) and improves cross-engine
merged FSC-AUC by only `4.47e-9`, from `0.999997239048` to
`0.999997243517`. Thus neither pass-1 nor pass-2 score/projection precision
explains the recurrent gap. The remaining live precision control must change
the accumulator itself. The sealed report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_it1to4_full_f64_local_a100_4acb1e05_20260719T032200Z/audit/it4_global_f32_f64_summary.json`
(SHA-256 `df9eae38f08c7dba7c64741492bf995bb45f1c06385027b01d8e1ccb0b995ed7`).

The live accumulator-only control is also complete and rejects float64 as the
repair. On the same physical A100, with identical scheduling and all global
score/projection float64 toggles unset, promoting only the x-half M-step
accumulator lowers merged cross-engine FSC-AUC from `0.999997239270` to
`0.999995330192` (a `-1.91e-6` change). The two arms compare directly at
merged FSC-AUC `0.999997952849`. Iteration-1 particle Pmax, support, poses, and
translations are exact between arms; by iteration 4, Pmax p95 differs by
`0.00469` and 98/3,000 particles change significant-support count, while the
size/order schedule remains exactly `56,56,52,52` / order 3.

Accumulation precision is therefore a genuine amplification/sensitivity axis,
but production float64 moves away from native RELION rather than toward it.
Because native RELION's BackProjector accumulation is also float32, the next
causal discriminator is RELION-like float32 reduction order/topology versus
upstream native contribution operands—not making float64 the production
default. The sealed report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_it1to4_mstep_double_pair_local_a100_7390051f_20260719T034000Z/audit/it4_mstep_double_pair_summary.json`
(SHA-256 `844b9086d4e04bd1e62d0e91eb72b7fb609c84c28cd0cdef4de827a236f1c045`).

The qualified live float32 order controls are also null at the recurrent
iteration-4 boundary. On the same physical A100 as the production control,
enabling sequential float32 translation reduction together with RELION's
128-thread/native-FFTW pixel block topology changes the merged map only to
FSC-AUC `0.999999999782` versus production. Cross-engine merged FSC-AUC moves
by `-1.36e-10`, from `0.999997239270` to `0.999997239134`. Significant-support
counts remain exact for all 3,000 particles at every iteration; iteration-4
Pmax absolute p95 is `3.19e-5`, and best poses/translations are exact there.
Thus these two qualified float32 ordering differences do not explain the
recurrent map gap.

An attempted arm adding one launch per particle failed closed before
iteration 2: soft-posterior work is bucketed by support size, so its ownership
order is not native particle order. Relaxing that guard would not constitute a
RELION-order experiment. Per-particle/fused topology is therefore unresolved
for later soft posteriors rather than inferred from the null qualified arm.
The next decisive experiment is a native RELION physical-iteration-4
pre-scatter operand capture with control/control and capture-inertness
envelopes, followed by one common deterministic float64 replay.

The sealed float32 report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_it1to4_relion_order_f32_local_a100_7390051f_20260719T080000Z/audit/it4_sequential_block_f32_summary.json`
(SHA-256 `f95cfb4bbe7e112781a3b1153c313b8ac23ae3165233a0a7edbe8d6e3b5e8857`).
Map quality uses FSC/FSC-AUC only; correlation is not computed.

## 2026-07-19 case-8 low-memory full-trajectory qualification

Pinned diagnostic job `11367838` validates the conservative exact-local
microbatch cap through the recurrent OOM boundary. Numbered iteration 16 and
both final all-data `515^3` split-half M-steps complete without resource
exhaustion. All 17 numbered merged RECOVAR/RELION FSC-AUC values are at least
`0.999436816`. Final merged FSC-AUC is `0.9956749799762161`, passing the
unchanged `0.995` gate; sign-invariant GT FSC-AUC is `0.184732902` for RECOVAR
and `0.183791257` for RELION, a delta of `-0.000941646` that passes the
`>= -0.002` gate.

The last numbered merged FSC-AUC is `0.999964169`, so the drop to `0.995675`
is introduced at final all-data. Final split-half cross-engine FSC-AUC values
are only about `0.84984` even though the merged map passes. This final-boundary
effect remains a separate localization target; it does not invalidate the OOM
repair or the configured merged/GT quality gates.

Fresh same-H100 job `11368287` completed the apple-to-apple confirmation on
physical UUID `GPU-fa2518be-28c1-06bc-a6cc-131e55fe503e`. Final merged
cross-engine FSC-AUC is `0.995676320`; GT FSC-AUC is `0.184729916` for
RECOVAR and `0.183935538` for RELION, a `+0.000794377` RECOVAR delta. RECOVAR
wall time is `8071.71 s` versus RELION `11062 s` (`1.37x` RELION/RECOVAR).
The summary SHA-256 is
`ad6c8a672ca486700c7ce86e45766f099fa02acdc685603b7797a01152771555`.

The sealed trajectory is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_ec25f6dd_autonomous_rerun_launcher_20260718T124500Z/runs/full_ec25f6dd_20260718T184318Z/cases/8_anisotropic_high_noise_100k_g256_white_noise3_bf80_retry3_pinned_lowmem/trajectory_analysis/k1_case8_pinned_lowmem_fulltraj_seal_v1.json`
(SHA-256 `df95eb8275bb89d1708fb4dd10b7662d30e8de861f6897fa188547295da345af`).

## 2026-07-19 capped convergence and autonomous termination classification

RECOVAR had one independent finalization bug: after the numbered loop reached
`max_iter`, it synthesized the angular-sampling and convergence check that
would have occurred at a nonexistent next expectation boundary. RELION checks
convergence only at the top of an iteration satisfying `iter <= nr_iter`; if
the last allowed numbered iteration merely makes the state convergence-ready,
the loop exits without a final all-data pass. The synthetic RECOVAR check is
removed, and a regression now makes a convergence-ready last-numbered state
remain unconverged and unfinalized. This is especially important for K-class,
where final all-data is valid only after actual convergence.

This cap bug does not explain the current autonomous case-22/case-23
termination mismatches because both use `max_iter=999` and converge well before
the cap. Their one-iteration differences are downstream of real adaptive
schedule splits:

- Case 22 matches RELION's resolution through iteration 7. At iteration 8,
  RECOVAR selects shell 20 (`27.2 A`) while RELION selects shell 19
  (`28.631579 A`). RELION therefore advances to HEALPix order 5 at iteration
  9; RECOVAR remains at order 4 and first latches fine-enough sampling before
  iteration 10. The matched-prefix merged cross-engine FSC-AUC is
  `0.999021050685` at iteration 8 and falls to `0.989950113817` at the schedule
  split in iteration 9. RECOVAR converges after 10 numbered iterations and
  RELION after 11.
- Case 23 keeps the same HEALPix orders through the matched prefix, but its
  iteration-11 resolution-stall decision differs: RECOVAR remains at shell 20
  while RELION improves from shell 19 to shell 20 and resets the resolution
  stall counter. RECOVAR can consequently latch fine-enough sampling one
  expectation earlier. Cross-engine merged FSC-AUC remains
  `0.999924748954` through iteration 11 and falls to `0.990005839361` at
  iteration 12; RECOVAR converges after 12 numbered iterations and RELION
  after 13.

Do not repair these autonomous cases by forcing an extra numbered iteration or
loosening convergence thresholds. Their earliest cause is the upstream
half-map FSC/resolution state near a discrete shell boundary. The active
case-20 native RELION pre-scatter capture and common-canonical
float64/complex128 replay are the next discriminators for that recurrent map
state; map acceptance remains shellwise FSC/FSC-AUC only.

Fresh same-A100 case 7 (`11365077`) exposes another discrete downstream
amplifier. Numbered merged cross-engine FSC-AUC stays at least `0.999959757`
through iteration 11. At iteration 12, RECOVAR advances HEALPix order 4 to 5
one boundary before RELION and uses current size 102 instead of 104. Pmax
absolute-difference p95 jumps from `0.0333` to `0.6634`, pose and translation
tails become quantized local-search jumps, and merged cross-engine FSC-AUC
falls to `0.997016`. RELION advances to order 5 at iteration 13, but the
trajectory remains perturbed; RECOVAR converges after 14 numbered iterations
and RELION after 15. The final merged cross-engine FSC-AUC is `0.843573688`.
RECOVAR nevertheless has higher GT FSC-AUC (`0.131848813` versus
`0.125186418`, delta `+0.006662395`), which does not satisfy exact workflow
parity. Early pose/translation writeback is essentially exact.

The iteration-11-to-12 scheduler source audit rejects a scheduler bug.
RECOVAR's internal shell-20 gold-standard FSC is `0.499419838`, while the
RELION model STAR records `0.500996`; this straddles RELION's hard FSC `0.5`
resolution threshold. RECOVAR therefore records another resolution stall and
advances sampling, whereas RELION records a one-shell improvement and resets
the stall counter. RECOVAR float32 forward/reverse and float64 shell-20
controls are `0.499419302`, `0.499419987`, and `0.499419824`; their spread is
below `7e-7`, far smaller than the `0.001576` engine gap. This is source-correct
discrete amplification of upstream reconstructed-reference/FSC drift, not
reduction-order ambiguity. Do not patch `updateAngularSampling` or force an
extra iteration.

The sealed classification is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_upstream_pose_posterior_audit_20260719T101500Z/CASE7_IT11_IT12_SCHEDULER_CLASSIFICATION.md`
(SHA-256 `2c42dc609174389b1e1c992b9f265909c38215385fc0484f3bcbe0f502fa282f`).

## 2026-07-19 recurrent K=1 final-boundary family localization

Cross-case final-only controls localize the shared final-map failure before the
final all-data expectation begins. The inherited last-numbered
poses/translations and the two half-reference maps are jointly causal.
Substituting both raises merged cross-engine FSC-AUC to at least `0.995071180`
in every controlled member: cases 12, 14, 24, 25, 26, and 32 reach
`0.997324903`, `0.996130409`, `0.998028536`, `0.996330031`, `0.995071180`, and
`0.997890293`, respectively. The single-field effects vary by dataset, so
neither poses nor references alone are a universal repair.

Current-head case 20 independently confirms the same boundary. Poses alone
raise final merged FSC-AUC from `0.987915367` to `0.992921005`, references
alone to `0.992453421`, and both to `0.997197191`. Replacing all available
state reaches `0.997434762`, only `+0.000237571` beyond poses plus references;
correction and sampling state are therefore a small residual rather than the
dominant common cause.

The exact-state control closes the final implementation itself. On one
physical H100, RECOVAR initialized from complete RELION iteration-11 state and
matched the original RELION final at `0.996119500` and an independent RELION
repeat at `0.997561693`; the two RELION finals match at `0.997433283`. A
genuine float64 final fine-pass changes merged FSC-AUC by only `-3.19e-7`,
keeps every best pose and translation exact, and changes Pmax by at most
`5.01e-6`. Final reconstruction and final-pass float32 precision are rejected
as the leading shared cause.

The production target is upstream accumulated state: determine where the
autonomous last-numbered pose/reference distributions first diverge enough to
be amplified by the final Nyquist pass. The large case-8 four-arm factorial is
the next scale control. Pending sequential jobs `11371904` and `11372912` were
canceled before execution because their wall limit could not cover four
100k/grid-256 trajectories. Replacement same-UUID H100 array `11374242` runs
the resident, corrections/live-scale, poses-plus-references, and all-state
arms independently; verifier `11374243` performs the fan-in FSC/FSC-AUC and
exact/distribution audit.

The last-numbered mismatch is distributional rather than a collection of
bitwise near ties. Across the controlled family, only `77.9%`--`98.4%` of
rotations and `83.9%`--`99.1%` of translations fall within the tight `0.01`
tolerance; exact significant-support counts range from `30.61%` to `89.7%`,
and the particlewise Pmax absolute-difference p95 ranges from `0.0206` to
`0.139`. No one scalar tail statistic predicts final severity because the
pose/posterior error interacts with the incoming references. Debug the
aggregate score/posterior-to-pose/reference update path, using controlled
boundary substitutions, rather than requiring bitwise particle equality.

The sealed family report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_final_boundary_family_20260719T090000Z/FINAL_BOUNDARY_FAMILY_CLASSIFICATION_V1.json`
(SHA-256 `e447e0946def576314795ae61962039bdb3c5515db2ecd5ee2a89b12b8d04a82`).
Map quality uses FSC/FSC-AUC only; correlation is not computed.

## 2026-07-19 same-GPU K=4 production-versus-float64 trajectory

The bounded 100k/grid-256 K=4 precision A/B is complete on one physical A100
(`GPU-27d0dd53-0c19-7be3-82f4-eaba66bb35aa`). Production and genuine
float64 arms ran sequentially in Slurm job `11361629`; CPU audit job
`11374498` verified exact eight-iteration topology: current sizes
`[38,38,42,56,60,62,68,70]`, HEALPix order 1, and no local-search entry.

Production and float64 both first miss the unchanged per-class direct
RELION FSC-AUC gate at iteration 8, class 3. Production reaches
`0.994738857112`; float64 reaches `0.994700770508`, which is worse by
`3.81e-5`. Their direct map FSC-AUC remains at least `0.997201977521` across
all halves/merged classes and iterations, while same-boundary hard-class
agreement remains at least `0.99729`. Float64 changes the trajectory
materially—maximum Pmax MAE between the arms is `0.00546038`, minimum exact
significant-count fraction is `0.7157`, and maximum class-mass relative L2 is
`0.000341639`—but it does not close RELION parity. RECOVAR-minus-RELION GT
FSC-AUC stays within about `1.21e-4` in either arm.

This rejects production float64 as the repair for the K=4 iteration-8 class-3
gap. The run uses science commit `c390f8bf`, before the separate exact-local
normalization fix, but the bounded prefix never enters that local path. It is
therefore a precision causal diagnostic, not current-head quality acceptance.
Continue upstream at the accumulated reference/posterior boundary and confirm
the full current-head trajectory; do not weaken the `0.995` gate.

The resealed report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_100k_samegpu_prod_f64_it8_c390f8bf_20260719T002650Z/analysis/samegpu_pair_audit_v1/FINAL_AUDIT_SEAL.json`
(SHA-256 `522505c2c16e1642db4b24a0b7b23bd36bf7e42074aa2794eb3142f5b5335673`).
Its rebuilt manifest has SHA-256
`71f6c9464a5341ddf74e4b779d0c64ed20cb6a36f854fda3816891228f1fa973`
and verifies all 25 entries. Map quality uses FSC/FSC-AUC only; correlation is
not computed.

## 2026-07-19 K=1 native RELION pre-scatter operand localization

The same-physical-A100 physical-iteration-4/half-1 case-20 capture closes the
remaining reduction-versus-operand ambiguity. Capture instrumentation is inert:
the captured RELION merged map is within FSC-AUC `0.999999999710` of the
stock oracle repeat. RECOVAR versus captured RELION is `0.999997240709`
merged, consistent with the recurrent autonomous boundary.

The strict common-support audit covers 1,520 half-1 particles, 9,169 common
positive contributor rotations, and 9,716,168 emitted pixel rows. Matched
rotation matrices are exactly equal (`max_abs=0`), every RELION row lies in
RECOVAR's captured window, and every matched row has positive RECOVAR weight.
Contributor membership differs only for three particles: RECOVAR/RELION counts
are 3/4, 9/10, and 3/2; totals are 9,171 versus 9,172. This small support
difference is real but is not needed to expose the common-subset residual.

On common rows, native pre-scatter operand relative L2 is `0.0238642` for
complex data and `0.0221334` for real weight. Casting RECOVAR's captured
complex128/float64 values to complex64/float32 leaves those values unchanged
at the reported scale, so this is not a widening/cast artifact. Under one
common deterministic complex128/float64 replay, RELION operands placed with
RECOVAR versus RELION geometry produce exactly equal accumulators and map
FSC-AUC `1.0`. Replaying RECOVAR versus RELION operands on the same geometry
leaves accumulator relative L2 `0.0221842` for data and `0.00564523` for
weight, with unregularized map FSC-AUC `0.999792368507`.

This localizes the recurrent iteration-4 map difference upstream of neighbor
geometry and reduction order, to native contribution membership and operand
generation. The float64 replay promotes RELION's captured float32 operands; it
does not reconstruct precision already lost upstream. The next aggregate
discriminator is to factor the common-row data/weight operands into posterior,
scorer-scale/correction, CTF, and shifted-image terms, stratified by shell and
particle state. Do not return to serial particle chasing.

The sealed result is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_it4_relion_prescatter_local_same_rec_gpu_cap1tb_retry_20260719T091600Z/analysis/SEALED_RESULTS_V1.json`
(SHA-256 `cefa46b5ab1947859d6c3086d6ae782a1efca6d9997c8765b11a07c2b055695c`).
The full comparison is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_it4_relion_prescatter_local_same_rec_gpu_cap1tb_retry_20260719T091600Z/analysis/multicontributor_prescatter_comparison.json`
(SHA-256 `8e7a9e0a747bae5e3b420682fe1198deb31d5b9bff4a5a7ada1fd4c530058c0b`).
Map quality uses FSC/FSC-AUC only; correlation is not computed.

## 2026-07-19 case-25 incoming-reference null interventions

Case 25 distinguishes a different accumulated-state failure mode from the
case-20 operand boundary. With exact per-iteration RELION metadata, poses,
corrections, noise, priors, sampling state, and an identical oracle schedule,
replacing only the incoming half references at scoring iteration 7 produces a
negligible, non-propagating effect. At the target boundary, control and
RELION-reference cross-engine merged FSC-AUC values are
`0.999999999868` and `0.999999999899`; their direct map FSC-AUC is
`0.999999999894`. Pmax absolute p95 is `1.404e-4`, support counts are exact,
and pose/translation p95 values are zero. At iteration 8, the cross-engine
change reverses to `-1.027e-12` and the direct map remains
`0.999999999923`.

The earlier scoring-iteration-2 reference-only probe is likewise null:
cross-engine merged FSC-AUC improves by only `1.925e-11`, direct arm FSC-AUC
is `0.999999999925`, Pmax p95 is `3.774e-5`, and support/poses/translations
are exact. The effect reverses at iteration 3.

For comparison, the autonomous case-25 iteration-7 boundary has merged
cross-engine FSC-AUC `0.999995965764` and Pmax p95 `0.0120212`. Exact
non-reference state therefore closes that residual by roughly four to eight
orders of magnitude even when RECOVAR references remain resident. The
material autonomous boundary requires accumulated non-reference state and its
interaction with the maps; an incoming reference alone is not sufficient.
The next bounded intervention separates accumulated pose/posterior/correction
state from reference state at the first autonomous boundary.

The sealed classification is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case25_it7_relref_ab_20260719T111500Z/CASE25_REFERENCE_SUBSTITUTION_CLASSIFICATION.md`
(SHA-256 `949e13f9d241709edd651ac3da68709a4fc8cb2b05f9b7c6123de36710cc3ce5`).
All 11 entries in
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case25_it7_relref_ab_20260719T111500Z/CASE25_SCIENCE_ARTIFACTS.sha256`
verify. Map quality uses FSC/FSC-AUC only; correlation is not computed.

## 2026-07-19 case-20 aggregate operand-factor localization

The full common-support population makes the upstream operand residual highly
structured. Across 9,169 common half-1 contributor rotations and 9,716,168
pixel rows, the native complex-data and real-weight operand relative L2 values
are `0.0238642` and `0.0221334`. Fitting one scalar per rotation reduces the
data residual to `0.00145241`; replacing that fit with the pixelwise weight
ratio produces the same residual. The leading mismatch is therefore posterior
mass assigned to contributor rotations, not pixelwise CTF, noise, image, or
phase structure.

The residual is concentrated rather than a broad numerical floor. The top
`0.1%` of rotations carry `0.999996` of both data and weight residual energy.
One common rotation of stack 1969 alone carries `0.993009` of data and
`0.996270` of weight residual energy, and stack 1969 is one of the three
particles with different contributor membership (RECOVAR/RELION `3/2`).
RELION's normalized significance thresholds remain distributionally aligned:
median ratio `0.999958`, p5/p95 `0.998643`/`1.001305`, and relative L2
`2.34e-4`. A global threshold-scale change is not the explanation.

The stack-1969 support identity is now concrete. RECOVAR retains global
rotations 117318, 117319, and 119525 from two fine-parent groups. RELION keeps
two oversampled rotations from one parent; RELION oversample 5 is exactly
RECOVAR rotation 119525, while the other RELION child does not match the two
RECOVAR rotations from the absent parent. RECOVAR's retained masses are
`0.00793385`, `0.70557328`, and `0.28549708`; RELION's common matching child
has mass `0.9971559`. Thus the dominant residual is a pass-1 parent/pass-2
child-support mismatch that renormalizes common rows, not bitwise reduction
noise. The bounded same-GPU score capture is testing the parent/significance
boundary directly.

The aggregate decomposition is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_it4_relion_prescatter_local_same_rec_gpu_cap1tb_retry_20260719T091600Z/analysis/common_operand_upstream_decomposition_v1.json`
(SHA-256 `7929a9e14c3f311200a279af4706c272d27cf76d3ab85a7745fae2ae4b100041`).
Intermediate gates use exact/distribution metrics; map gates use FSC/FSC-AUC
only.

## 2026-07-19 case-25 accumulated non-reference state factorial

Three autonomous-prefix A100 arms establish a tight repeat envelope through
iterations 1--6: maximum pairwise map defect `1-FSC-AUC` is `7.11e-11`, Pmax
absolute p95 is `4.18e-5`, support counts are exact, and rotation/translation
p95 values are zero. At target iteration 7, exact non-reference state with
resident RECOVAR references reaches cross-RELION merged FSC-AUC
`0.999999999507`. Accumulated RECOVAR non-reference state with exact RELION
references reaches `0.999999104446`, and all resident RECOVAR state reaches
`0.999999105247`.

The two accumulated-non-reference arms compare directly at
`0.999999999848`, inside the repeat envelope. Reference-only Pmax p95 is
`2.13e-4`, with exact support, poses, and translations. Accumulated
non-reference state therefore reproduces essentially the entire autonomous
residual at this boundary. Against exact non-reference state, the all-RECOVAR
arm has Pmax absolute median/p95/p99/max
`0.00140`/`0.01186`/`0.02722`/`0.11396`, changes support for 62/1,000
particles, and has pose tails of `2.43559` degrees and `1.50000` Angstrom.

The singleton split is active. Its first qualified results reject accumulated
poses alone (`0.999999999901`, numerically null) and identify image/scale state
as a small contributor (`0.999999952094`) that is still much smaller than the
full accumulated-state defect. Tau/noise, direction-prior, sigma-offset,
scheduler/state, and leave-one-out complements remain required before a
production change is justified.

The sealed parent factorial is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case25_it7_autonomous_prefix_factorial_20260719T095858Z/CASE25_IT7_FACTORIAL_CLASSIFICATION.md`
(SHA-256 `5f6419542f9cfae362a159862146bea29c7edce77e61b7b9774d042e8f6f9295`),
with all 11 manifest entries verified. Map quality uses FSC/FSC-AUC only;
correlation is not computed.

## 2026-07-19 factor-capture postflight contract repair

The 32-particle factor-v2 capture exposed two diagnostic-harness defects. The
directory validator incorrectly treated particle-local fine-orientation
count, significance threshold, and posterior normalizer as run-wide
invariants. The inertness tool was hard-coded to iteration 1 and to one older
reference-report key spelling. The validator now checks those values against
each particle's own arrays while preserving run-wide geometry and policy
checks; it reports heterogeneous counts explicitly. The inertness tool accepts
an explicit positive iteration and both sealed reference schemas.

The real panel validates all 32 captures with orientation counts 8--104 and
accepted-hypothesis counts 11--345. Its same-A100 iteration-4 capture is inert:
half-map FSC-AUC is `0.999999999358` and `0.999999998802`, and all four
pre-join accumulator comparisons remain within the independent repeat
envelope. The patched validation report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_it4_bpref_factor_v2_panel32_same_gpu_20260719T101000Z/analysis/relion_factor_validation_v2.json`
(SHA-256 `62c3d9c965dbab59ae9ab184563b788522fe11ea8e691bd94e30a114127f0a4b`);
the inertness report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_it4_bpref_factor_v2_panel32_same_gpu_20260719T101000Z/analysis/capture_inertness_v2.json`
(SHA-256 `532c3a5a7bbdfb6411f81cc9aeda8dc7c574f5555890e1f97b609b259699dbfb`).

## 2026-07-19 case-20 variable-support factor-panel closure

The sealed 32-particle factor panel rejects a generic BPref factor,
translation-geometry, or reduction-order defect at physical iteration 4.
Every RELION fine rotation has one exact RECOVAR transpose-convention match,
all 116 translation phase increments agree within `7.45e-9`, and 27 of 32
particles have exact hypothesis support. On those 27 particles, global
relative L2 is `2.60e-4` for posterior, `3.73e-5` for the processed image,
`3.93e-5` for CTF, `5.30e-7` for inverse noise, and `2.30e-4`/`2.49e-4`
for orientation-composite data/weight.

The residual is instead concentrated in five preselected particles with
engine-specific posterior support: stack indices 1969, 2855, 1036, 867, and
2327. Stack 1969 has RECOVAR-exclusive retained mass `0.713481489` and
posterior relative L2 `1.149`; its normalized factor operands are already
closed. Stack 2855 combines posterior-support divergence with an image-only
correction-factor relative L2 of `0.2752`. Across the full panel, production
versus sequential-float32 or canonical-float64 reduction controls are only
`4.78e-8`--`2.23e-7` relative L2, about seven orders below the behavioral
cohort effect.

This classifies the boundary as upstream engine-state posterior/support and
image-correction divergence, not factor algebra or numerical reduction noise.
The next causal boundary is a controlled aggregate iteration-2/3 score,
posterior, and state substitution. Do not resume serial particle debugging.
The authoritative seal is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_it4_bpref_factor_v2_panel32_same_gpu_20260719T101000Z/analysis/SEALED_FACTOR_PANEL_V1.json`
(SHA-256 `9f54f2707aa328e6144643a92bc4c61f5871103d27f43fc97cbd8ff18b4f8611`),
and the interpretation is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_it4_bpref_factor_v2_panel32_same_gpu_20260719T101000Z/analysis/variable_support_factor_interpretation_v1.json`
(SHA-256 `1c51c21ba73af7359815e7b554bb98777650c5332ded5892394a30fd95ee14f5`).
Intermediate comparisons use exact/distribution metrics; map quality uses
FSC/FSC-AUC only, and correlation is not computed.

## 2026-07-19 case-20 exact-state iteration-2/3 M-step closure

Read-only same-A100 audits of a fresh uninterrupted RELION trajectory close
the physical-iteration-2 and physical-iteration-3 M-step boundaries when the
incoming state is exact. At iteration 2, the RECOVAR-minus-RELION BPref
FSC-AUC delta is `-8.51282e-7`, and exact-state RECOVAR reconstruction matches
the fresh RELION merged map at FSC-AUC `0.999999999620`. At iteration 3, the
BPref FSC-AUC delta is `+1.84211e-6`, the maximum shellwise FSC delta through
shell 26 is `2.98802e-5`, and the reconstructed merged-map FSC-AUC is
`0.999999999439`.

A companion iteration-3 incoming-map A/B shows the resident iteration-2 map
at merged FSC-AUC `0.999999999726` versus RELION and an arm-to-arm
iteration-3 FSC-AUC of `0.999999999760`. Its earlier descriptive amplification
of `1-FSC-AUC` is therefore an approximately `1e-9` floor effect, superseded
by the direct BPref and reconstruction closure. Physical iterations 2 and 3
do not introduce a material case-20 seed in the low-resolution join or Wiener
reconstruction. The remaining boundary is inherited state and the
score/posterior/contribution inputs that feed this closed path.

The sealed interpretations are
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_it3_relion_mstep_dump_same_gpu_20260719T213000Z/audit/PHYSICAL_IT2_BPREF_BOUNDARY_INTERPRETATION_V1.md`
(SHA-256 `6f4220958da7d4faf403e622db6a50b57a6dd4a09767cbe2d575120d543c9ab6`)
and
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_it3_relion_mstep_dump_same_gpu_20260719T213000Z/audit/PHYSICAL_IT3_BPREF_BOUNDARY_INTERPRETATION_V1.md`
(SHA-256 `862878cba36203dc08a31bb481de9868589a9834c94d36b03d5865f7437e4fce`).
Map quality uses shellwise FSC/FSC-AUC only; correlation is not computed.

## 2026-07-20 case-20 physical-iteration-3 resident-state factorial

A same-physical-A100 six-arm factorial restored selected RECOVAR-produced
resident iteration-2 state only after a complete RELION replay override at
physical iteration 3. All arms used RELION scoring references through
iteration 3, source commit `1e208826`, GPU UUID
`GPU-dc6576aa-e1e4-6055-4a5e-d0fa809f3983`, current sizes `[56, 56, 52]`,
and HEALPix orders `[3, 3, 3]`. GUI final-grid and forced after-max overrides
were unset.

The exact all-RELION control reproduced the physical-iteration-3 merged map
at FSC-AUC `0.999999999074`. The individual RECOVAR-state arms had the
following merged FSC-AUC deltas versus that control: poses `-6.35e-13`, image
norm/group scale `-9.65e-12`, tau2/noise `-1.65e-10`, and image scale plus
poses `-1.42e-11`. Their direct merged FSC-AUC values versus the exact control
were all at least `0.999999999596`. These state components are not material
individual iteration-3 map seeds, and the scale-plus-poses arm shows no
material interaction.

Restoring the complete RECOVAR resident bundle gave merged FSC-AUC
`0.999999894444` versus RELION, a `-1.04631e-7` delta from the exact control,
and direct merged FSC-AUC `0.999999895635` versus the control. This is a
reproducible full-bundle effect, but it remains roughly four orders of
magnitude inside the `0.995` cross-engine gate. Because it is about three
orders larger than the named individual arms, the next bounded split targets
the remaining maps, direction-prior, optimiser-state, and translation-sigma
components before aggregate score/posterior comparison.

The authoritative seal is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_it3_state_factorial_same_gpu_20260720T105500Z/audit/SEAL_V1.json`
(SHA-256 `995fca71c75c5a23bd79043d7fb4f0e2ad2226e4dd8458ffdac4b5ab9ac42564`).
Map quality uses shellwise FSC/FSC-AUC only; correlation is not computed.

## 2026-07-20 case-20 remaining-state split audit correction

The six-arm continuation completed on the same physical A100. Maps,
direction prior, complete optimiser state, and optimiser state excluding the
sampling grid are null at the physical-iteration-3 map boundary. Their merged
cross-RELION FSC-AUC values are respectively `0.999999999018`,
`0.999999999088`, `0.999999999075`, and `0.999999999076`, versus
`0.999999999074` for the exact all-RELION control. Both optimiser-state arms
also reproduce iteration-3 poses, translations, Pmax, and support counts
exactly.

The apparent sigma-offset arm is excluded from production attribution. The
matrix used source commit `1e208826`, whose diagnostic state-swap hook saved
only the scalar mean sigma and rebroadcast it when the treatment restored the
field. It therefore scored with `[4.24445754, 4.24445754]` Angstrom instead
of the RELION pair `[4.255510, 4.233412]`. The normal K=1 production
trajectory had retained the distinct per-half pair, and tracked commit
`25a83be7` already repaired the diagnostic hook. K-class remains intentionally
shared because the active RELION Class3D target has
`_rlnDoSplitRandomHalves=0`; no sigma production change is justified by this
factorial.

The corrected interpretation is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_it3_remaining_state_split_same_gpu_20260720T154031Z/audit/INTERPRETATION.md`
(SHA-256 `c9aac00ed61f236fd4266f593f2f1e59d1c7a5db2d0ad97e53395b8b79a1737d`).
The summary JSON SHA-256 is
`311e1b9c8d0bd243ca68d716ce9e73de05c8f0f57bd9be9288316f6fad3e9523`.
The exact-state 32-particle production-posterior/factor panel completed on GPU
UUID `GPU-dc6576aa-e1e4-6055-4a5e-d0fa809f3983`; its findings are recorded
below. GUI grid and forced after-max overrides were unset.

## 2026-07-20 case-20 physical-iteration-3 exact-state posterior panel

Matched production RECOVAR and passive RELION factor-v2 captures completed on
one physical A100. The RECOVAR capture contains 24 strictly validated shards,
3,000 particles/fragments, and 1,284,288 candidates. Capture and no-capture
RECOVAR runs have exactly equal physical-iteration-2 Pmax and coarse-support
arrays for all particles; their merged map FSC-AUC is `0.999999999840`.
RELION capture/control half-map FSC-AUC is approximately `0.999999999956`, and
all BPref arrays remain inside the two same-GPU repeat envelopes.

The 32-particle exact-state comparison closes rotation geometry exactly and
translation phase increments to maximum error `2.85e-9`. All 32 posterior
winners agree. Fine reconstruction support is exact for 26/32 particles.
Pmax absolute error has median `5.95e-5`, p95 `1.63e-3`, and maximum
`7.31e-3`; posterior-union relative L2 has median `1.63e-4`, p95 `2.68e-2`,
and maximum `5.59e-2`. The three material posterior outliers are stack
particles 1036, 2707, and 1045, and all retain the exact RELION winner.
RECOVAR-versus-RELION merged map FSC-AUC is `0.999999999041`.

The archived `sig_counts_*` trajectory records coarse pass-1 retained
hypotheses, while the production raw capture records exact fine pass-2
reconstruction support. They must not be equated. The analyzer separately
closes both RECOVAR representations and compares the coarse trajectory with
RELION `rlnNrOfSignificantSamples`: 28/32 selected counts are exact, with
three `+1` and one `-1` residuals. The result localizes the remaining
scorer/posterior discrepancy to a small high-Pmax tail but does not yet
justify a production formula or threshold change.

The interpretation is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_it3_exactstate_posterior_panel32_same_gpu_20260720T160500Z/analysis/INTERPRETATION.md`
(SHA-256 `a2b89941a61bcd38da4449b520f8d60cbff46ce931d6cbb1622285f3a00f96f1`).
The 357-entry self-excluding accepted-artifact seal is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_it3_exactstate_posterior_panel32_same_gpu_20260720T160500Z/analysis/accepted_artifacts_v1.sha256`
(SHA-256 `5466f9c194cf27409232ddfc205cbadff0a7119ade077d1efa2f97c4b16470c7`).
Map quality uses shellwise FSC/FSC-AUC only; correlation is not computed.

## 2026-07-20 case-20 exact-state coarse-parent cutoff localization

The exact-state fine candidates partition into 32 children per retained
coarse parent. Candidate support is exact for 26/32 panel particles; all six
mismatches differ by whole parent blocks. After restricting both engines to
their common fine support and renormalizing, posterior relative L2 has median
`1.33e-4` and maximum `3.15e-4`. RELION-posterior-weighted centered-log RMS
has median `2.03e-4` and maximum `4.79e-4`. The common score shape is
therefore closed at the few-`1e-4` scale; the material raw-posterior outliers
come from exclusive parent blocks, not a general fine-score or normalizer
formula mismatch.

A corrected dense global pass-1 capture then replayed the exact iteration-3
state on the same A100 for the six support-mismatch particles. All dump
identities, current sizes, K=1 shapes, posterior normalizations, significance
masks, and archived RECOVAR coarse counts close exactly. Every RECOVAR-only
parent is its final retained rank, and every RELION-only parent is RECOVAR's
first excluded rank. Stack 626, 1266, and 1509 cross the strict `0.999`
criterion only when that final RECOVAR-only parent is added. Stack 1045
crosses `0.999` at rank 19 by only `5.55e-7`, while RELION retains rank 20.

The one-for-one swaps are sub-`0.001` log-score cutoff reversals. For stack 1036,
RECOVAR ranks its retained parent 54 and the RELION-only parent 55; their
final coarse log scores differ by `8.54e-4`. For stack 2707 the analogous
rank-28/rank-29 difference is `5.87e-4`. Those residuals follow much larger
data/prior cancellations: respectively `+0.649132` plus `-0.648279`, and
`-1.917908` plus `+1.918505`. Candidate geometry, parent expansion,
posterior normalization, and significance semantics are no longer viable
primary causes. The remaining discriminator is passive RELION coarse pass-1
data-score/prior capture for these parent pairs; factor-v2 begins at fine pass
2 and cannot separate the two operands. No production threshold expansion or
formula change is justified from the current evidence.

The numerical cutoff report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_it3_exactstate_posterior_panel32_same_gpu_20260720T160500Z/analysis/parent_pass1_cutoff_v1.json`
(SHA-256 `82d1d380021d24b7172153544415b3c1588a70eaf0152536ac326e82b0bd1849`).
The interpretation is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_it3_exactstate_posterior_panel32_same_gpu_20260720T160500Z/analysis/PARENT_PASS1_CUTOFF_INTERPRETATION.md`
(SHA-256 `ab01aea32c36b7f836b831e87b699ec720416938534f101cf223e7be478b97ac`).
The focused analysis seal verifies all 15 named inputs and outputs at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_it3_exactstate_posterior_panel32_same_gpu_20260720T160500Z/analysis/PARENT_PASS1_ANALYSIS_SEAL.sha256`
(SHA-256 `8a27b82fc91e325eec432e20cb085f0a29cbb365218af8d4ebbd74d58f6c2db2`).
GUI grid and forced final-all-data-after-max-iter overrides were unset. Map
quality remains shellwise FSC/FSC-AUC; correlation is not a pass/fail metric.

## 2026-07-20 case-20 native RELION coarse-score operand closure

An env-gated RELION CUDA capture now records physical-iteration-3 global
coarse raw diff2, rotation/translation priors and zero masks, pre-exponent log
weights, and post-exponent weights for the six support-boundary particles.
The accepted diagnostic source is RELION commit
`c2bb4d87176a02977247c738355980bc21f3c19e`; binary SHA-256 is
`6fed307a174f0443e8888b37b76de8ef2e8cfe82b7717e17b70b58c6dc656a39`.
An earlier `9b7d421` capture is quarantined because its default-stream copies
raced the optimizer stream. The corrected hook queues copies on each owning
stream and synchronizes before serialization.

The corrected same-A100 control/capture pair has byte-exact 9,000-row dispatch,
exact poses/translations and significant-count metadata, and half-map FSC-AUC
`0.999999999453` and `0.999999998655`. RELION's
direction-major/psi-inner orientation layout maps exactly to RECOVAR's
psi-major/direction-inner layout; all prior zero masks then close and exactly
the eight previously classified support differences remain. Raw diff2 plus
priors closes to the captured pre-exponent log score within `6.20e-5`.
Allowed-support rotation-prior maximum error is `4.53e-4`, and translation
prior maximum error is `3.81e-6`.

The stack-1036 RECOVAR-minus-RELION pair error is `+0.001304626` in the data
delta and `+0.000162065` in the prior delta, producing `+0.001464844` in
the final delta. For stack 2707 those values are `+0.000793457`,
`+0.000054121`, and `+0.000831604`. The data scorer is the dominant
remaining boundary residual; the prior is smaller but nonzero. The next
diagnostic target is therefore RELION CUDA coarse projection/residual
arithmetic and reduction order for these exact parent pairs. No threshold,
formula, or acceptance-gate change is justified.

The accepted numerical report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_it3_relion_coarse_score_same_gpu_c2bb4d8_20260720T150500Z/analysis/relion_coarse_score_audit_v1.json`
(SHA-256 `85964de2f393f5cd1af11a7fc1be700d9d7ad2c01e30507856d7bca371b93060`).
The interpretation SHA-256 is
`a750ba71e884d4861266c13ddbe25a30c55cf776f3fa52a2e78b06935c5fe4c4`.
The 27-entry accepted-artifact seal is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_it3_relion_coarse_score_same_gpu_c2bb4d8_20260720T150500Z/analysis/ACCEPTED_ARTIFACTS.sha256`
(SHA-256 `49154c1a24ecbe6c7f240a19a717e06379457bf6ac5dadb7433885a3037f0cac`).

A follow-up decomposition over every active captured candidate further narrows
the data-score boundary. After removing the particle-wide additive constant,
the translation main-effect RMS is only `7.73e-6`--`1.29e-5`, with ranges
`2.96e-5`--`5.64e-5`. The rotation main-effect RMS is instead
`4.86e-4`--`8.52e-4`, with ranges `0.0112`--`0.0258`; residual
rotation-by-translation interaction accounts for `22.6%`--`31.1%` of active
centered variance. The disputed stack-1036 parents lie at centered data-score
residuals `+0.0006516` and `-0.0006530`; stack 2707 lies at `+0.0002505`
and `-0.0005429`. This rejects a translation-only convention or prior-offset
explanation and targets orientation-dependent projector/residual operands plus
their native reduction topology. It does not distinguish those two causes, so
the next probe remains an immutable per-pixel contribution capture for the
exact swapped parent pairs.

The decomposition report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_it3_relion_coarse_score_same_gpu_c2bb4d8_20260720T150500Z/analysis/coarse_data_score_residual_structure_v1.json`
(SHA-256 `0e30a6004ad71e5348b38a24f1bb6f4582e63aa12b91bbbe1e80d00e52572b7d`);
its analyzer SHA-256 is
`4903f77b12b8d836c0c1eb6a1a5e66c2a781e53405807e601014d66fc64f4be6`.
Grid correction and forced after-max overrides were unset. Map quality remains
shellwise FSC/FSC-AUC; correlation is not a pass/fail metric.

## 2026-07-20 case-10 x-half acceptance OOM classification

Case-10 science job `11409144` reached physical iteration 12 on A100
`della-l07g7`, then failed status 1 in the exact local x-half M-step. The
iteration-12 half-1 support widened to 584 rotations per image and bucketed
at sizes up to 766. The existing tail cap reduced the row budget from 15,500
to 13,405 hypotheses, but the fused local bucket still requested 10.77 GiB
after 1,000 of 1,431 chunks and exhausted the allocator. Physical GPU memory
telemetry was stable near 33.3 GiB before failure; this is a transient tile
limit, not accumulated map-state growth. Dependent audits `11409145` and
`11409146` remain non-admissible because the science job did not complete.

The exact failure log is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case10_xhalf_tail_accept_9d172278_20260720T071204Z/em_k1_matrix_10_high_res_anisotropic_100k_g384_radial_noise3_bf0.out`.
The next retry must lower the explicit exact-local row/matmul caps while
leaving science and acceptance gates unchanged.

The lower-cap retry is science job `11421265`, with after-ok auditors
`11421266` and `11421267`. It changes only
`RECOVAR_EXACT_LOCAL_TARGET_ROW_PIXELS` from 128M to 64M and
`RECOVAR_EXACT_LOCAL_BIG_JIT_MATMUL_MAX_GB` from 4.0 to 2.0. Science and
acceptance gates are unchanged. The run root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case10_xhalf_tail_lowcap_accept_9d172278_20260720T164557Z`.

## 2026-07-20 K=4 continuation rejection and live-factor replacement

The iteration-10 serialized-continuation retry is rejected as a parity oracle.
Despite an exact recorded follower schedule, retry 7 compares with the
canonical uninterrupted RELION run at per-class merged FSC-AUC
`0.9717330`, `0.9707791`, `0.9642605`, and `0.9643529`, far below
the unchanged `0.999999` continuation-identity gate. It instead repeats the
earlier continuation family at relative L2 `3.49e-7`--`4.58e-7`, including
one assignment/class mismatch. Serialized optimiser files therefore do not
represent the complete live MPI/GPU process state, and their BPref factors are
quarantined. No further continuation-based factor inference is admissible.
The sealed rejection is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_bpref_capture_ac5177d2_20260720T084530Z/provenance/RETRY7_CONTINUATION_REJECTION_20260720T0814-0400.md`
(SHA-256 `1830dde77eed1dcbe36cacf1078f6dcaf0c18dacb9f2bc2987f33cf938f0582e`).

The replacement began with a fresh uninterrupted RELION Class3D run and live
iteration-10 factor capture. A minimal patch at RELION HEAD
`d476e6f6a4f1f37627c06ace5227fc374c0c2b05` emits pre/post-SSNR BPref data,
weight, spectra, metadata, and incoming Iref. The resulting binary SHA-256 is
`af0cf03190f761af694c869945ebb6cce692bf3d7b0ef0524eba2e3044e0c1b1`.

The first launcher encoded the wrong reconstruction contract. This Class3D
run has `_rlnDoSplitRandomHalves=0`: follower accumulators are combined, and
the reconstruction ranks own one aggregate `half0` BPref per class. Rank 1
reconstructs classes 1 and 3; rank 2 reconstructs classes 2 and 4. RELION job
`11413202` completed status 0 in 5,573 seconds and emitted 52 factor files,
then the launcher failed because it expected 96 split-half files. Its RECOVAR
phase never started, dependent audit `11413203` was canceled, and neither is
admissible end-to-end evidence. The corrected capture is independently sealed
at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_100k_live_bpref_pair_9d172278_20260720T083000Z/provenance/RELION_IT10_CLASS3D_CAPTURE_COMPLETE.json`.

The replay oracle copies complete states 0--11 from that live process and
seals all 98 required artifacts. Its schedule SHA-256 is
`bdaf4d87ea3bbd41901f82acf975a74109c3fdcd1544ce4c9c741e5e295c9085`.
RECOVAR commit `9d1722781e1d6c5fc5b2ad0e15ebba3a2becbab0` is replaying iterations
1--10 against the immutable oracle in science job `11414986`; corrected
aggregate-factor audit `11415296` compares data, weight, tau2, sigma2, and
data-vs-prior using exact/relative-L2 and shellwise metrics. Grid correction
and forced final-all-data-after-nonconvergence remain unset. The replay root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_live_factor_recovar_9d172278_20260720T090629Z`.
The live replay science job `11414986` completed status 0 through physical
iteration 10 in 18,709 seconds. It remained non-converged and correctly
skipped final all-data; no forced after-max path ran. Corrected aggregate-factor
audit `11415296` completed status 0. Across classes 1--4, combined BPref data
relative L2 is `0.0299440`, `0.0322863`, `0.0314599`, and
`0.0262171`; weight relative L2 is `0.0069118`, `0.0078684`,
`0.0076745`, and `0.0062358`. Downstream tau2 relative L2 spans
`0.0001014`--`0.0014831`, sigma2 spans
`0.0001098`--`0.0004253`, and data-vs-prior spans
`0.0001528`--`0.0017568`. These are uninterrupted live-process
iteration-10 residuals, not continuation artifacts.

The factor report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_live_factor_recovar_9d172278_20260720T090629Z/analysis/it10_live_bpref_factor_audit.json`
(SHA-256 `4947d59cf4e7cc270b2855a42bb04ac43b2cb3f3fdb0eb24dd1a5dfdfb9f787d`).
Focused map FSC audit `11426828` was rejected before computation because its
absolute-path invocation lacked the checkout on `sys.path`. Retry
`11426923` binds imports to the checkout and retains the unchanged direct
FSC-AUC `0.995` and GT-delta `-0.002` gates. It completed status 0 in 133
seconds. The identity class assignment is optimal for both engines and GT.
Classwise direct RECOVAR-versus-RELION FSC-AUC is `0.9960693`, `0.9953734`,
`0.9954234`, and `0.9966223`; RECOVAR-minus-RELION GT FSC-AUC is only
`-4.29e-5`, `-2.78e-5`, `-4.09e-5`, and `-2.37e-5`. Thus the uninterrupted
iteration-10 BPref factor residual is numerically real but not map-quality
material under the unchanged FSC gates.

The map report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_live_factor_recovar_9d172278_20260720T090629Z/analysis/it10_live_map_fsc_audit.json`
(SHA-256 `f3a842d1840577a988c67a6b18f48dd03643859deaee10f44a32a0791007377d`),
and the shellwise array SHA-256 is
`ef44549cdcfc1157c35b05d6ad26fde4a93783b110a2f481c2e6ed6daa962dc0`.

An independent full-trajectory audit also quantifies why an unconstrained
fresh RELION run is not the deterministic oracle. The two 1.5-million-record
dispatch logs have identical iteration, sorted-position, and particle-index
keys, but dynamic follower assignment differs for 54,436 of 100,000 particles
at iteration 1 and 45,996--55,228 at later iterations. Map FSC-AUC remains
nearly closed through iteration 2 (minimum `0.9999999633`) and then bifurcates
at iteration 3 (minimum `0.9703505529`, mean `0.9843806925`). The complete
iteration-0--15 report SHA-256 is
`6d2ee204c32d59bdff4ae188d219f10862e4837fc520db15e4fc76c68fa9bf67`.
This different-dispatch envelope does not lower the strict same-dispatch
RECOVAR gate.

The unchanged acceptance policy uses shellwise FSC/FSC-AUC for map quality
plus exact/distribution topology and particle-state diagnostics; correlation
is not a pass/fail metric.

## 2026-07-20 adaptive pass-1 CUDA scorer-matrix closure

An exact same-A100 Gaussian-pair capture identified a narrow matrix-construction
mismatch. RELION's adaptive coarse `AccProjectorPlan` creates score matrices
with CUDA `acc_make_eulers_3D`, while its fine-score and weighted-sum paths use
host `generateEulerMatrices(..., inverse=true)`. RECOVAR had routed the host
matrix through both passes. For the four disputed rotation IDs, the existing
RECOVAR CUDA builder reproduces all 36 captured RELION float32 matrix words
exactly; the old host matrices differed by at most `2.09e-7` and crossed a few
texture interpolation fraction bins.

Commit `db1bf3914b6ec6df212a9743fb12ffb86c7b4c23` now uses CUDA-built
matrices only for adaptive pass 1. Fine scoring, M-step backprojection, and
pose metadata retain the host path. CPU guards passed `48/48`, the CUDA Euler
suite passed `9/9`, and `pixi run test-em-fast-guard` passed.

The immutable exact-state iteration-3 replay completed in `361.7 s`. Stack
1036's RECOVAR-minus-RELION production pair residual fell from `0.00151825`
to `0.0000457764`, and its 54-parent significant mask became exact. Stack
2707 fell from `0.000679016` to `0.000305176`; canonical float64 pixel operands
differ by only `3.65e-6`, leaving RELION's captured atomic reduction order as
the boundary discriminator. Its remaining one-for-one 28-parent support swap
is tie-aware: the two posterior values are within `1.87e-8`. Across the two
particles, posterior maximum error and weighted centered-log RMS both improve.

Old-versus-new half-map FSC-AUC is `0.99999999897` and `0.99999999876`.
Sign-corrected new-versus-RELION half-map FSC-AUC is `0.99999999903` and
`0.99999999812`; merged GT FSC-AUC changes by only `4.04e-8` in magnitude.
No threshold or acceptance gate changed. Grid correction and forced final
all-data after non-convergence were unset, and correlation was not used.

The operand report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_it3_adaptive_cuda_score_db1bf391_20260720T203010Z/analysis/coarse_gaussian_pair_operands_v1.json`
(SHA-256 `78dc00c22ea3dbe0b19bec50faa355f13f044b9d382883ede47e077dafd7c52b`).
The interpretation is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_it3_adaptive_cuda_score_db1bf391_20260720T203010Z/analysis/INTERPRETATION.md`
(SHA-256 `5635ec5c2421a18c14f1b9273d95098640261a5027fb5544aecb8f884ae6f056`).
The self-excluding accepted-artifact seal is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_it3_adaptive_cuda_score_db1bf391_20260720T203010Z/analysis/ACCEPTED_ARTIFACTS.sha256`
(SHA-256 `2b11c699ea9ba45edb281a097c8356e0e650e6c6e0d2edfd31eec7e3c88ca818`).

### Six-particle support closure

A broader exact-state replay at commit
`d0e02d5bacd14ee5d5894bbfe7d9c0bce248e941` targeted all six particles
whose old pass-1 parent support differed from RELION. Four now close exactly:
stack indices 626, 1036, 1509, and 1045. Thus exact support on this
mismatch-selected panel improves from 0/6 to 4/6. Median Pmax absolute error
falls from `4.154206e-4` to `8.156474e-6`; stack 1036 specifically falls from
`2.649784e-3` to `9.000207e-6` while its 54-parent support becomes exact.

Stack 2707 retains its already-qualified rank-28/rank-29 reduction-order tie.
The two new parent posterior weights differ by only `1.819e-8`. Stack 1266
retains one extra rank-13 parent with posterior mass `4.6382e-4`, despite a
Pmax absolute error of only `7.3127e-6`; it is the next bounded cutoff-pair
diagnostic target. These residuals do not justify broadening the CUDA-matrix
route or changing the `0.999` significance rule.

Old-versus-new map FSC-AUC is `0.999999998967` and `0.999999998751` for
the two halves and `0.999999999197` after merging. Sign-corrected
new-versus-RELION FSC-AUC is `0.999999999035`, `0.999999998122`, and
`0.999999998879`. Correlation is not computed or used for acceptance.

The accepted report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_it3_adaptive_cuda_score_panel6_d0e02d5b_20260720T165800Z/analysis/panel6_v1.json`
(SHA-256 `88abba193ebcdba0c1e06501b7f8ab2d25101abacf203d1418ba45d48f8f77c3`).
The interpretation SHA-256 is
`8e5b593a95e60f20968fafc82f4fe52ee6d3c65e9ca9d38eb62ffc60625462a6`,
and the self-excluding accepted-artifact seal SHA-256 is
`42eefa38aacb4818e683ae9c5b592e30445a265f4f38be87db7b258a01a60ed0`.
Grid correction and forced final all-data after non-convergence were unset.

### Stack-1266 cutoff-pair closure

The bounded follow-up at RECOVAR commit
`25039b9a8a5398db0bd20303dc96fe4faf65cad1` captures exact projection and
score operands for the historical rank-13/rank-12 cutoff pair. RELION's
targeted diagnostic is passive: control and capture dispatch logs are
byte-exact across 9,001 records, wall time is 379 versus 375 seconds on the
same A100, and control/capture FSC-AUC is at least `0.9999999990393841`
across both halves and iterations 1--3. The only changed stack-1266 metadata
fields are log-likelihood contribution (`11507.539058` versus
`11507.538742`) and Pmax (`0.160829` versus `0.160841`).

RECOVAR's captured pre-prior pair log-score delta is
`0.4567413330078125`; RELION production is `0.456787109375`, a residual of
`-4.57763671875e-5`. RELION production exactly replays its captured
four-lane float32 reduction for both candidates. Reference relative-L2 errors
are approximately `2.1e-7`, with shifted-data and CTF2 errors below
`5.1e-7` and `6.6e-7`. This evidence does not support another projection,
matrix-routing, or reduction-order change.

In the matching current RECOVAR capture, the top 12 posterior weights sum to
`0.9990000036874245`, only `3.6874245e-9` above the configured adaptive
fraction. RECOVAR therefore selects the same 12 parents as RELION; the
historical rank-13 parent is first excluded with posterior weight
`4.6381747e-4`. The earlier 13-versus-12 result is a real floating-point
cutoff toggle between otherwise equivalent same-GPU trajectories, not a
reason to change the `0.999` rule. No additional production code change is
justified by this diagnostic.

The authoritative operand report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_it3_gaussian_pair_stack1266_42f8489_20260720T171800Z/analysis/stack1266_operands_v2.json`
(SHA-256 `34a019bea5f1b2089e340c24e7a94bb7c82f3586af2dcc57fcc7cb4fc6445e2d`).
The capture-passivity report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_it3_gaussian_pair_stack1266_42f8489_20260720T171800Z/analysis/capture_passivity_v1.json`
(SHA-256 `87828fd7e88228a0a82d64b9f61d4b218786db1368d8ddef3a863eb2896a11cd`).
The accepted-artifact seal verifies and has SHA-256
`f7ef3e59677c04a828de3b60d8e6fe2c28a59c0a70e99192f5fb61680cc357b5`.
Correlation was not computed; map passivity uses shellwise FSC/FSC-AUC.

## 2026-07-20 K=1 300k-particle case-3 acceptance

The fresh same-H100 case-3 science job `11384178` completed status 0 on
`della-h21g2` from commit
`ac5177d2b0cd639db7ed6f14225d80fe9cff7d4d`. Final merged
RECOVAR-versus-RELION FSC-AUC is `0.9987867026`, while the two split-half
values are `0.9997955468` and `0.9997952732`. GT FSC-AUC is `0.5874386123`
for RECOVAR and `0.5820647217` for RELION, a delta of `+0.0053738907`.
Final all-data ran only after convergence, and the GUI-quality grid-correction
override remained unset/off. The summary JSON SHA-256 is
`c21bcee4320ff7973aa5cf08c2a1052edf0ad5ebab29dee28e4ce3141ec7406b`.

## 2026-07-20 current-head K=4 three-iteration replay

The adaptive pass-1 CUDA scorer-matrix production change also routes K-class
global searches, so current pushed commit
`dc47d27ddaa753327252ed6997672995a2861911` was replayed for three numbered
iterations against the immutable same-dispatch K=4 RELION oracle from commit
`f2c1a384400aec37dc6805856a5ba645650a44f1`. Science job `11432973`
completed status 0 on H100 `della-h19g1` in 839 seconds. It remained
non-converged and correctly skipped final all-data; grid correction and forced
after-max were unset.

Every map-quality gate passes. Direct per-class RECOVAR-versus-RELION FSC-AUC
is at least `0.9999999746`, `0.9999989689`, and `0.9996362300` in iterations
1--3. The minimum RECOVAR-minus-RELION GT FSC-AUC delta is `-3.96921e-6`,
well inside the unchanged `-0.002` gate. Correlation was not computed or used.

Strict-state audit job `11433052` returned its intentional status 2 because
the frozen fixture does not include complete candidate-score evidence for the
known iteration-3 boundary. The mismatch set is unchanged from the old
RECOVAR oracle at commit `111b8fde65725bb2cebbcfae82dd1f251221dcb9`:
particle 1513 changes class; particles 1513, 2136, 4685, 7661, 7700, and 9357
change rotation; and all except 7700 change translation. Iterations 1 and 2
have exact class, rotation, and translation decisions.

A direct current-versus-old RECOVAR comparison confirms zero class-decision,
rotation-over-`1e-3`-degree, or translation-over-`1e-4`-pixel differences in
all three iterations. Per-class current-versus-old map FSC-AUC is at least
`0.9999999505685849`; maximum Pmax difference is `2.22385e-4`. Current head
records a uniform four-to-one iteration-1 support-count bookkeeping change
and one-count boundary changes for 28 and 14 particles in iterations 2 and 3.
Because the comparison spans multiple intervening commits, these count deltas
are not attributed specifically to the scorer route. They accompany at most
`1.06140e-5` relative L2 in the recorded half-1 M-step numerator, but no
discrete-decision or material map change.
The old-versus-current minimum iteration-3 FSC-AUC against RELION changes by
only `-7.97e-9`. No additional K-class code or threshold change is justified.

The immutable run root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_cold3_currenthead_recovar_replay_dc47d27d_20260720T174500Z`.
The strict trajectory report SHA-256 is
`80fac125179eecb413800e2a9a574b3c0e93495ae19934e2675d0d27ae430b5f`,
the direct current-versus-old report SHA-256 is
`58bad36b0f8708682d5646ba916eec32d61c450fcf5dbb2e76a88dafb17dfc6d`,
the interpretation SHA-256 is
`18eb6bfe13a422d0e76c90bbf12d634cc0f0cf1850bd19ec80912fe54c04d2ba`,
and the self-excluding accepted-artifact seal verifies with SHA-256
`d1e21eff2749cc4561a5e332ef4ac35d847cd888bdcb30360b73f93d0676cc69`.

## 2026-07-20 current-head autonomous case-20 closure

Science job `11435532` completed status 0 on H100 `della-h19g1` from the
exact pushed commit `790ea8a96a2f2b7b063b69c579dd44bb1cf8288c`. It produced 11 numbered
iterations, converged, and only then ran final all-data. Grid correction and
forced final all-data after non-convergence were unset. The immutable run root
is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case20_currenthead_autonomous_790ea8a9_20260720T230537Z`.

All numbered map gates pass against RELION. The minimum split-half and merged
FSC-AUC values are `0.999999998053`, `0.999999998391`, and
`0.999999998577`; the worst RECOVAR-minus-RELION GT FSC-AUC delta is
`-1.63512e-7`. Final split-half FSC-AUC is `0.999991603804` and
`0.999391929757`, final merged FSC-AUC is `0.997760979983`, and the final GT
delta is `+0.001141882262`. Correlation was not computed or used.

The current-size schedule is `[56, 56, 52, 52, 50, 50, 50, 52, 50, 52,
50]`, exactly matching RELION in all 11 iterations. The older autonomous run
at `ac5177d2` incorrectly stayed at size 50 in iterations 8 and 10. Its final
merged FSC-AUC was `0.986023998270`; the current-head run improves that to
`0.997760979983`. Its numbered merged FSC-AUC began diverging at iteration 4
and reached `0.999839148101` at iteration 10, whereas the current-head values
remain at least `0.999999998577`. Final pose mean improves from
`0.6939696` degrees to `0.00270765` degrees, translation mean from
`0.0636256` pixels to `0.000502069` pixels, and Pmax absolute mean from
`0.00965278` to `0.000214967`. This fresh autonomous trajectory closes the
pre-fix case-20 regression without changing the `0.999` significance rule or
the existing map-quality gates.

The trajectory audit is `audit/k1_fsc_trajectory.json` under the run root
(SHA-256 `18329b437c2fd95f6e5bc7e5cdf9b209963988eb35f9cb7498cf473c62e9e530`),
the intermediate-state audit SHA-256 is
`cd5fbd273820b558fdff78ff3d22fa3351da51528b7801ddba9e354b97a819f3`,
and the corrected old-versus-current comparison SHA-256 is
`4b10eec23254ebe3caf88a420447dc00ad9d05d8642d3d068eeb4105aaf1d061`.
The final audit manifest SHA-256 is
`29909291d281a61db29524ca850d2d90948971b1b4e6e1fe8563fd6cd07438ff`.
The old and current runs used the same H100 model and node but different GPU
UUIDs, so this is combined-head acceptance evidence; causal attribution to
the adaptive pass-1 CUDA matrix fix remains grounded in the prior same-A100
exact-state A/B diagnostics. The summarizer correctly resolves the final
RELION source to numbered iteration 11.

## 2026-07-20 current-head autonomous case-26 replay

Science job `11436583` completed status 0 on H100 `della-h19g4` in 1,005
seconds from the exact clean pushed science commit
`790ea8a96a2f2b7b063b69c579dd44bb1cf8288c`. It produced 11 numbered
iterations, converged, and only then ran final all-data. Grid correction and
forced final all-data after non-convergence were unset.

The replay reproduces the prior `ac5177d2` trajectory. Current-size and
HEALPix-order trajectories, convergence iteration, final sampling state, all
1,000 final Euler decisions, and all 1,000 final translations are exact.
Old-versus-current final FSC-AUC is `0.999999999323`, `0.999999999606`, and
`0.999999998322` for half 1, half 2, and merged maps. Maximum `ave_Pmax`
trajectory difference is `2.57592e-6`, and maximum final Pmax difference is
`3.39091e-4`.

Every numbered iteration remains inside the unchanged gates. Minimum numbered
merged cross-engine FSC-AUC is `0.998906419972`; the worst numbered
RECOVAR-minus-RELION merged GT FSC-AUC delta is `-0.000486476689`. The
intermediate topology audit passes. The final-only failure is nevertheless
reproduced: split-half FSC-AUC is `0.962112863489` and `0.944303094765`, and
merged FSC-AUC is `0.954914353690`, below the unchanged `0.995` gate. RECOVAR
final merged GT FSC-AUC is `0.221394864515` versus RELION
`0.211295735975`, a positive delta of `0.010099128540`. Correlation was not
computed or used.

This rejects a broad generalization claim for the adaptive pass-1 CUDA matrix
fix: it closes current-head case 20 but does not change case 26. It also
confirms the existing recurrent final-boundary localization rather than
supporting another final reconstruction, gridding, or threshold patch. The
next causal target remains upstream accumulated pose/reference state and the
pending same-UUID case-8 factorial.

The immutable run root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case26_currenthead_autonomous_790ea8a9_20260720T234054Z`.
The FSC audit SHA-256 is
`e620b1785c6d4b80d7faa4d10cccf8ed78eb0290beb523924d4ea922dd1b2bbc`,
the passing intermediate audit SHA-256 is
`94e75c8cbcfe38086e5897d4e126bba5f402eb4ae52ee465bb843e84982431a9`,
and the current-versus-old interpretation is
`audit/CURRENT_VS_OLD_INTERPRETATION.md` under that root.

## 2026-07-20 inclusive current-size boundary correction

The 400k-particle/grid-128 case 33 exposed a distinct scheduling defect before
its already-known final-map failure.  At numbered iteration 2 both engines had
an available and strongly supported boundary shell 34: RECOVAR FSC
`0.9487659` and FSC-derived data-vs-prior `18.51823997`, versus RELION FSC
`0.948759` and `rlnSsnrMap=18.515718`.  RECOVAR's K=1 post-M-step path
nevertheless zeroed data-vs-prior starting at `current_size // 2`, discarded
that shell, and scheduled size 98 rather than RELION's 100.

Commit `7f5f758474afac40650c5a2760cf124bfe420989` centralizes the truncation
and zeros only beyond the inclusive boundary, starting at
`current_size // 2 + 1`.  This matches pinned RELION source commit
`f2c1a384400aec37dc6805856a5ba645650a44f1`, whose gold-standard FSC path
also zeros from `current_size / 2 + 1`, and keeps the K=1 and K-class state
paths on one contract.  Follow-up commit
`77bcf3bd7f45760ab0671c4883d91a453d58113a` covers the two observed matrix
growth geometries.

A read-only replay of all saved matrix FSC trajectories finds 11 affected
decisions in seven cases.  First-iteration instances in cases 1, 6, 29, and
30 are masked by the explicit `--firstiter_cc` ini-high override and remain
topology-exact.  The three later instances exactly explain observed topology
gaps: cases 2 and 3 change grid-256 size `100 -> 162` to RELION's
`100 -> 164`, and case 33 changes grid-128 size `68 -> 98` to RELION's
`68 -> 100`.  This supports one shared boundary fix and rejects case-specific
schedule overrides.

The bounded clean-checkout case-33 replay, Slurm job `11438037`, completed
`0:0` on H100 `della-h20g2` in 4,265 seconds of science wall time.  Its
fail-closed audit records RECOVAR and RELION current sizes `[56, 68, 100]`
and `matches_relion=true`; iteration 2 reports `res_shell=34`, `raw=100`, and
`quantized=100`.  The audit is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case33_boundary3_currenthead_7f5f7584_20260720T204500Z/audit/schedule_replay.json`
(SHA-256 `a525d47a2d6c7ee02900fb57fbae2ce5aa4f6e7ab69f4a2c9ff908da716a4ea0`).
This accepts the corrected schedule boundary, not full case-33 FSC-AUC.
Grid correction and forced final all-data after non-convergence were unset.

Focused matrix-boundary validation is 5 passed, and the full EM-targeted unit
selection is 345 passed.  The audit is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_case33_boundary_fix_20260720T203400/TEST_AUDIT.md`
(SHA-256 `e105d417ab564832d15fac3b761bd7d0c3cc363928bb910fc7c302a9c6b7b109`),
and the matrix replay is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_case33_boundary_fix_20260720T203400/MATRIX_BOUNDARY_REPLAY.md`
(SHA-256 `1c48d8ed4e316c179158b9900244414a8d5af424be56b1f1ef76c5f2a5ed5e55`).

A read-only full-particle audit of case 7's topology-identical iterations
1--11 supports systematic accumulated particle-state drift before shell 20
straddles FSC 0.5.  Across 100,000 exactly identity-aligned particles,
support-count mismatches grow from 72 at iteration 2 to 536 at iteration 4,
2,166 at iteration 10, and 2,731 at iteration 11.  Pmax absolute p95 grows
from `0.000455` to `0.007820`, `0.028327`, and `0.033425` at the same
boundaries.  The fraction with pose error above 0.1 degree grows from
`0.020%` to `0.140%`, `0.912%`, and `1.105%`; pose p99 first jumps to
`1.844918` degrees at iteration 11.

The iteration-11 pose tail is symmetric across halves and concentrated in
RELION's `Pmax < 0.5` cohort, where its incidence is `5.033%`, versus
`0.044%`--`0.373%` in higher-Pmax cohorts.  The top 5% of absolute Pmax
errors at iteration 10 does not predict the next pose tail (`0.976x`
enrichment), and support mismatches provide only `1.609x` enrichment.  This
rejects a small set of largest-Pmax-error particles as the sole cause and
classifies the downstream shell crossing as an amplifier of a broad
low-confidence posterior/support trajectory.  Keep the exact FSC and
sampling rules.  The next bounded discriminator is a fixed iteration-10
candidate-score/posterior margin capture stratified across low-Pmax pose-tail
and pose-stable particles.  The report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_case7_prefix_particle_audit_20260720T213000/case7_prefix_particle_audit.json`
(SHA-256 `af79c2598ad46b6f2176b57645acc2f16f440ead6a1267774950838a6a424852`).

Before submission, the active case-7 hypothesis is that accumulated incoming
state/reference differences change the physical-iteration-11 local score
surface for the ambiguous cohort; pure same-input GPU near-tie arithmetic is
the alternative.  A deterministic 24-particle panel contains six pose-tail
and six Pmax/support-matched pose-stable controls per half.  Resident and exact
RELION-state/reference arms share the current-size/HEALPix schedule, source
commit, fixture, H100 allocation, and targeted production captures.  Selection:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_it11_stratified_posterior_77bcf3bd_20260720T214500Z/selection/panel24.json`
(SHA-256 `0f441257af9b1152d6bf1eb2126960479826656bfafa3da1b0fb90b514d4dd2b`).

The original Slurm job `11439493` produced only a partial resident arm and
failed; it is not accepted evidence.  Same-H100 retry `11442740` completed
both arms `0:0` on `della-h20g3`.  A full identity audit of its frozen panel
closes the proposed discriminator.  The resident arm reproduces a
greater-than-0.1-degree error for 11/12 tail particles (mean `1.8442902`
degrees), while exact RELION incoming state/reference substitution puts all
12/12 within 0.1 degrees (mean `7.1038e-6` degrees).  All 12 matched stable
controls remain within 0.1 degrees in both arms with exactly unchanged
per-particle errors.  Tail translation agreement within 0.1 Angstrom improves
from 5/12 to 12/12, exact support from 8/12 to 12/12, and mean absolute Pmax
error from `0.0327671` to `0.000302662`.

Thus case 7's shell-20 FSC/scheduler split is a downstream amplifier of
accumulated incoming state/reference drift.  The counterfactual rejects a
change to the FSC `0.5` threshold or `updateAngularSampling`; the next target
remains the earlier broad low-confidence posterior/support drift.  This
diagnostic classification does not change the frozen score.  The sealed JSON
and Markdown SHA-256 values are
`331fa2104424c4fe434a00a81adf763e11bac43d38b490cd40ba64ab64ba078a`
and
`344ab83917c91fb1ebeba789efc6f6d7da3ed7b4c475430f7d32350a72b466d2`.

## 2026-07-20 active full case-33 and g384 acceptance chains

The bounded case-33 schedule replay accepts the inclusive boundary correction
only, so a full current-head run is now active. Science job `11440100` runs
the exact clean pushed commit
`7605c1b017316abc21f5a5f258fa9bbf936df5fe` on H100 `della-h20g2` against
the immutable 400k-particle fixture and RELION oracle. It runs autonomous
RECOVAR to convergence and permits final all-data only after convergence.
`RECOVAR_FINAL_ALL_DATA_GRID_CORRECT`,
`RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER`, and forced current sizes are unset.
Iteration 1 completed in 832.4 seconds and scheduled the expected `56 -> 68`
boundary exactly. Iteration 2 then completed in 1,234.1 seconds, retained the
inclusive shell-34 state (`res_shell=34`), and scheduled the causal
`68 -> 100` boundary exactly (`raw=100`, `quantized=100`). This accepts the
schedule correction in the autonomous 400k-particle run as well as the bounded
replay. All subsequent convergence, final-map, and FSC/FSC-AUC gates remain
pending; this paragraph is not a full case-33 acceptance claim.

A read-only provisional audit of the two complete numbered rows is also inside
the unchanged map gates.  Iteration-1/2 merged cross-engine FSC-AUC is
`0.999999999967`/`0.999999989583`, and the corresponding RECOVAR-minus-RELION
merged GT FSC-AUC delta is `+1.5702e-8`/`+8.6502e-8`.  The minimum reopened
merged shell FSC is `0.999999999685`/`0.999999954737`.  The provisional audit
correctly exits nonzero solely because only two of the 14 RELION numbered rows
exist in RECOVAR while science is active; it is not terminal acceptance.  Its
method note is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/k1_case33_provisional_it2_20260720T225300/PROVISIONAL_AUDIT.md`
(SHA-256 `d070a5c58af2b49d87d85d09adaadbb709d583d4b7f0baf16a790b2d9eed84d1`).
Its exact nine-input MRC manifest has SHA-256
`36974a77ecfe360e756fca54864ce08e72db12182b156b4cb868c50a3b8be6a4`.

Iteration 3 then completed at corrected `current_size=100` and scheduled
iteration 4 at `current_size=128`, matching RELION.  The expanded read-only
snapshot remains inside every map gate: iteration-3 half-1/half-2/merged
cross-engine FSC-AUC is `0.999999861446`/`0.999999810960`/`0.999999916678`,
and its merged GT FSC-AUC delta is `+7.51289e-7`.  The worst reopened merged
shell is `0.999999580507` at shell 52.  The analyzer again exits 2 solely for
the expected incomplete live topology (`RECOVAR=3`, immutable RELION
oracle=14); this remains provisional rather than terminal acceptance.  The
13 exact MRC inputs are sealed in
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/k1_case33_provisional_it3_20260720T232600/input_sha256.txt`
(manifest SHA-256
`2978eb60c6b79540eaabce5fc74b277c75e46fecf89dc5c0a376a396a5ea98a4`).
The method note SHA-256 is
`e87f10f2baa44f379c482be551d827c9054fa84f450960336d8a52e243bd9a04`.

Iteration 4 completed at `current_size=128`, reported 8.77 A, and retained
`current_size=128` for iteration 5, matching RELION.  Its half-1/half-2/merged
cross-engine FSC-AUC is `0.999999564536`/`0.999999659947`/`0.999999801784`,
and its merged GT delta is `+2.19454e-6`.  The worst reopened merged non-DC
shell FSC is `0.999997819649` at shell 62.  The read-only analyzer exits 2
solely for the expected incomplete topology (`RECOVAR=4`, `RELION=14`).  The
17 exact inputs are sealed under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/k1_case33_provisional_it4_20260721T000100`;
the input-manifest SHA-256 is
`d37193a81ca2a9e3d297c7c5a5c92a6698324ae82a6273397272a62298b181fb`
and the method-note SHA-256 is
`04dca2557829ff22c967ff4989fa42a3a0a0da3db8baf8b0fa79a89a8856eb32`.

Iteration 5 also completed at `current_size=128`, reported 8.77 A, and
retained `current_size=128` for iteration 6, matching RELION. Iteration 6 then
entered the expected local-search HEALPix order 4. The five-row read-only
snapshot remains inside every numbered map gate. Iteration-5 half-1/half-2/
merged cross-engine FSC-AUC is `0.999998472037`/`0.999998404940`/
`0.999999192750`, and its merged GT FSC-AUC delta is `+5.18920e-6`. The
worst reopened merged non-DC shell FSC is `0.999993303985` at shell 62. The
analyzer exits 2 solely for incomplete live topology (`RECOVAR=5`,
`RELION=14`); this is not terminal acceptance. The 21 exact numerical inputs
are sealed under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/k1_case33_provisional_it5_20260721T003600`.
The input-manifest SHA-256 is
`9303ea0a9bab79a184fb7aa09dcbf8010d02023dc82972bc679203b0304fd3ca`,
and the method-note SHA-256 is
`d98d106cc66cff8dae6068d35cce59b009ed2ea177c90c4d0bb3163eefdfb479`.

Dependent job `11440102` runs both the shellwise FSC/FSC-AUC trajectory gate
and the intermediate topology/artifact audit. Dependent sealer `11440295`
then fail-closes on exact source provenance, science/audit exit status,
convergence, final-all-data execution, GUI grid-off state, H100 identity, and
unchanged submission hashes. Its synthetic success/failure controls passed.
The run root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case33_full_currenthead_7605c1b0_20260720T222000Z`;
the sealer Python SHA-256 is
`174994cd6719cea2e73fca290ddd318ab9112dcf7d1ec2922f65a78b36f40aad`,
and the self-checking wrapper SHA-256 is
`7b9b89ea87d19905cee26df438918869bb7df52e515e47eacfef1d9712009c2d`.

The active grid-384 cases now also have terminal fail-closed seals. Case 9
science/audits/sealer are `11432807`, `11432810`/`11432811`, and `11440427`;
case 10 uses `11421265`, `11421266`/`11421267`, and `11440428`. The common
sealer runs after both auditors regardless of their exit state, then requires
all three upstream jobs to be `COMPLETED 0:0`, reopens every referenced
shellwise FSC curve, verifies the unchanged `0.995` cross-engine and `-0.002`
GT-delta gates, verifies exact topology and finite artifacts, requires
convergence-only final all-data with grid correction off, and proves exact
RELION/RECOVAR physical-GPU UUID identity.

Before retry 4 reaches the old case-9 OOM boundary, a sealed read-only prefix
comparison tests whether its conservative memory package already perturbs
science.  Against failed default-cap job `11415206`, the first four low-cap
rows have merged cross-run FSC-AUC
`0.999999999891`, `0.999999999606`, `0.999999820591`, and
`0.999997444482`.  Iteration 4 has 96 hard-assignment differences among
100,000 particles and worst merged shell FSC `0.999987323984`.  Summed
iteration time changes from 6,104.5 to 6,138.7 seconds (`+0.5602%`).  The
runs used different physical H100s, and retry 4 changes the allocator, row
target, and matmul cap together, so this accepts the package's prefix
stability but does not isolate the 2 GB cap or justify a production-default
change before the terminal boundary/audits.  The 80 completed input artifacts
are sealed under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/case9_lowcap_prefix_compare_20260720T233100`;
the method note SHA-256 is
`63c40a68414b354a004d9ef1ae095737a96d3c34132aa29d7abfaa24bba2e582`.
Correlation was not computed.

Case 10 has now crossed its terminal scheduling boundary correctly.  RECOVAR
completed numbered iteration 15 at `current_size=68`, HEALPix order 5, then
declared convergence and entered physical iteration 16 as the Nyquist
combined-data iteration (`current_size=384`).  This matches the RELION
trajectory's 15 numbered rows followed by its combined-data terminal row.
The run did not use the after-max escape hatch, and the grid-correction
override remains unset/off.  Science job `11421265` is still executing that
terminal row, so all final-map and shellwise FSC/FSC-AUC gates remain pending;
this is a convergence/topology checkpoint, not case acceptance.

A read-only real-artifact control on completed 400k case 34 passed this sealer:
all 162 referenced shellwise curves reopened, final merged cross-engine
FSC-AUC was `0.9957574121481196`, and the RECOVAR-minus-RELION merged GT
FSC-AUC delta was `+0.0028692403080708972`. The common sealer root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k1_single_case_acceptance_sealer_20260720`;
its Python SHA-256 is
`78a768df2eec4da35bb5a6b30963ae78d9f006317397287cf44fdd4bf9a7563c`.
Map quality remains shellwise FSC/FSC-AUC only; correlation is not computed.

## 2026-07-20 late acceptance checkpoints and case-7 retry

Case-10 science job `11421265` completed the terminal half-1 full-Nyquist
score-only pass over all 49,933 particles in 1,654.9 seconds (30.2
images/second) without an allocator failure.  Its fine M-step then activated
the explicit x-half tail guard at full BPref shape `(771, 771, 771)`, reducing
the hypothesis cap from 863 to 60 before building 49,878 microbatches.  The
fine M-step is still active, so this accepts only the score-pass memory
boundary.  It does not yet accept the M-step, terminal map, or shellwise
FSC/FSC-AUC gates.

Case-7 discriminator job `11439493` completed the resident arm and emitted
all 24 requested iteration-11 fused-posterior captures, but the launcher then
failed closed with status 1 before starting the exact-RELION-state arm.  The
failed assertion expected 24 global significance dumps.  Iteration 11 is in
the exact-local path, which does not invoke that global-pass dumper; the
intended fused-posterior diagnostic was complete.  This is an orchestration
failure and supplies no resident-vs-exact inference.

Retry job `11442740` reruns both arms sequentially on one H100 allocation and
requires the 24 fused-posterior captures per arm without the inapplicable
global-significance assertion.  Grid correction and forced final-all-data
remain unset.  The retry launcher is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_it11_stratified_posterior_77bcf3bd_20260720T214500Z/jobs/run_case7_it11_stratified_retry2.sbatch`
(SHA-256
`770f61354eb4554e1308e0c3b321e065789bc6c62802d694290b8f341ce52dca`).

The sealed case-10 default-cap versus low-cap comparison now covers all 11
numbered rows shared before default-cap job `11409144` failed in iteration
12. Both runs use source commit `9d172278`, `cuda_malloc_async`, and the same
fixture; only the exact-local row target changes from 128 million to 64
million row-pixels and the large-JIT matmul cap changes from 4 GiB to 2 GiB.
Merged cross-run FSC-AUC remains `0.999999999852` at iteration 1 and
`0.999993726467` at iteration 11. The latter row's worst merged non-DC shell
FSC is `0.999956847678`. Hard-assignment mismatches grow to 4,299/100,000
(`4.299%`) by iteration 11, while combined-noise relative L2 remains
`2.14518e-5`; the maps therefore remain exceptionally close despite
accumulating discrete near-tie changes. Correlation was not computed.

The 11-row runtime sum changes from 11,143.5 to 11,481.8 seconds
(`+3.03585%`). The jobs used different physical A100 UUIDs, so neither that
timing delta nor the residual arithmetic drift is an isolated single-cap
effect. This accepts prefix stability and strengthens the memory caps as the
operative boundary control: the low-cap run crossed the default run's
iteration-12 OOM and is executing the terminal Nyquist M-step. It does not
accept the terminal map or justify a production-default change before the
terminal audits and independent case-9 boundary complete. The 220 exact input
artifacts and results are sealed under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/case10_lowcap_prefix_compare_20260721T000700`;
the method-note SHA-256 is
`b4ac0d10c2d538b709201db11d76e3ad508fbb54eb2751c4eb51e0637cc3d517`,
and the input-manifest SHA-256 is
`8668818ea6206ca89e802490bce3622e041ef3dc7ffaa925034ff54131c63cb4`.

A deterministic 2x2 decomposition of the exact source cap formula prevents
over-attributing that bundled low-cap result. At the iteration-12 failure
dimensions, the final caps are 13,405 for 128M/4GiB, 7,750 for 64M/4GiB,
8,682 for 128M/2GiB, and 7,750 for 64M/2GiB. The current unset source defaults
(190M/4GiB for the x-half M-step) also reach the failed 13,405 outer-tail
cap. Either single-knob cross-arm reduces the arithmetic cap, but neither has
run on the GPU, so the completed 64M/2GiB arm cannot yet be attributed to one
control. At terminal half-1 Nyquist geometry, the outer tail guard reduces
all four configurations to 60 hypotheses; neither environment cap controls
that post-guard shape.

The admissible v2 calculation is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/case10_cap_decomposition_20260721T002800/case10_cap_decomposition_v2.json`
(SHA-256
`af5f6d082680f0833c47acd00666372814a5e2f6c64095a837b8f5bd47daacf2`).
The method-note SHA-256 is
`85cb3ead5f23ed198e199671c77571f4a956d20c5273b027252f8d9b18556311`.
The preserved preliminary v1 result is explicitly non-admissible because it
omitted the mixed cross-arms. This arithmetic audit computes no map metric or
correlation and does not replace the pending GPU and FSC/FSC-AUC gates.

## 2026-07-21 old-head full-34 durable negative closure

The autonomous K=1 full-34 matrix at source `ac5177d2` is now durably closed
as a negative acceptance result. Ledger job `11385656` completed `0:0` after
the repaired summary job `11385655` exited `2:0` on scientific failures. The
ledger contains 31 completed and three failed science jobs, 20 trajectory
passes/12 failures/two errors, and 25 intermediate passes/seven topology
mismatches/two errors.

The first terminal sealer never ran because its submission omitted the pinned
script hash. Repair job `11444630` exposed a second orchestration defect: its
launcher still named canceled pre-repair summary job `11384290`. Final repair
job `11444736` bound summary `11385655`, ledger `11385656`, and the exact
sealer hash; it exited `2:0` as intended and wrote the canonical negative seal
with SHA-256
`819c532884408cca35de9ea3ed43c0e516d2be822d23cb7bd14d76c54da9d9e2`.
The ledger SHA-256 is
`b79c18e5feb368782ef3a9fd439413bc3d1f890bcf19d1f97dc23037d09d97f1`.

Cases 2 and 3 first mismatch at iteration 3 (`164` RELION versus `162`
RECOVAR), and case 33 first mismatches there (`100` versus `98`). These are
the exact dropped-boundary-shell decisions corrected by commit `7f5f7584`;
the active current-head case-33 run independently has exact topology and
near-unity FSC-AUC through this boundary. Cases 9 and 10 have missing
old-matrix RECOVAR refinement output and are covered by separate low-memory
acceptance jobs. The remaining negative rows stay unresolved until
current-head or causal evidence closes them; the old-head ledger alone does
not justify another source change.

The complete provenance note is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_guigrid_localhighshell_full34_autonomous_ac5177d2_20260719T174000Z/provenance/FULL34_DURABLE_NEGATIVE_ACCEPTANCE_20260721T0045-0400.md`.
Grid correction and forced after-max finalization were unset. Map quality is
shellwise FSC/FSC-AUC only; correlation was not computed.

## 2026-07-21 case-33 iteration-6 FSC checkpoint

Current-head case-33 science job `11440100` completed numbered iteration 6 at
`current_size=128`, 8.63 A, and scheduled iteration 7 at the same size,
matching RELION. A numbered-map-only provisional audit reports half-1,
half-2, and merged cross-engine FSC-AUC of `0.9999992518331595`,
`0.9999993072698271`, and `0.9999996330509999`. The merged GT FSC-AUC delta
is `-0.00000416399429437`; the worst merged non-DC shell FSC is
`0.9999969085087222` at shell 62.

The analyzer's status 2 is solely the expected incomplete topology
(`RECOVAR=6`, `RELION=14`), not a numerical gate failure. All six available
rows pass the 0.995 cross-engine and -0.002 GT-delta gates. The 25-input
manifest SHA-256 is
`6dd1be6ba086f11ae81a679eee5141c583c13cf24bda637d4bc8530136304e14`;
JSON and shellwise-NPZ SHA-256 values are
`7fa15c93307d8b6ac5ceb407e716cf98863efae44f12a77118b46ee7243c8332`
and `22c67500dca95319797d0d4ac049f10309db769b7c2ff5ea93a00d301f2b3d48`.
The sealed provisional root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/k1_case33_provisional_it6_20260721T004800`.
Terminal acceptance remains pending; correlation was not computed.

## 2026-07-21 case-9 low-cap arm crosses the old OOM boundary

Case-9 low-cap retry job `11432807` completed physical iteration 11 on H100
and entered iteration 12 at `current_size=212`. The failed default-memory job
`11415206` reached iteration-11 half 2 with
`max_hypotheses_per_microbatch=8707`, then failed on an 8.24 GiB allocation.
Retry 4 completed the same half at cap 3879, finished iteration 11 in 419.1
seconds, and continued without an allocator error. This validates the bundled
low-memory package across the exact old failure boundary.

A sealed direct comparison covers all ten complete numbered rows shared
before that boundary. Iteration-10 default-versus-low-cap half-1, half-2, and
merged FSC-AUC are `0.9997898789059761`, `0.9997810550344120`, and
`0.9998737350071399`. The worst merged non-DC shell FSC is
`0.9991935318010615`; hard assignments differ for 2039/100000 particles, and
combined-noise relative L2 is `1.2366320110209453e-5`. Ten-row runtime sums
are 12009.5 versus 12035.7 seconds (`+0.2182%`) across different H100 UUIDs.

Retry 4 bundles `cuda_malloc_async`, a 64-million exact-local row target, and
a 2 GiB large-JIT matmul cap; this cross-node result does not isolate one
control or prove same-device identity. The 200-input manifest SHA-256 is
`35487dd427726a6ba09dc843f4db136f23e2d7ca2a47ee3ec9317e062ffd85ff`;
the method-note SHA-256 is
`1a20297b34d3c640f72e768ca67cf53def2b60f2685f8d30d28c6adf42bcb037`.
Sealed root:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/case9_lowcap_prefix10_compare_20260721T005600`.
Terminal FSC/FSC-AUC audits remain pending, so this does not yet justify a
production-default change. Correlation was not computed.

## 2026-07-21 case-10 terminal half-1 memory checkpoint

Case-10 low-cap job `11421265` completed the physical-iteration-16 final
all-data half-1 path at `current_size=384` on A100
`GPU-4bccbe72-c64a-5f5f-1fa8-ecf0bf6acf37`. The x-half outer-tail guard
reduced the fine M-step cap from 863 to 60 for the `(771, 771, 771)` BPref
accumulator. All 49,878 chunks and 49,933 particles completed in 4,964.2
seconds; Hermitian enforcement and both large-accumulator repacks completed,
the half-1 manifest was written, and half 2 started. Total half-1 wall time was
6,693.4 seconds.

This directly accepts the bundled low-cap package across the extreme half-1
terminal M-step memory boundary, but it neither isolates the three controls
nor accepts terminal map quality. The manifest SHA-256 is
`fead22d62f6e7302e7b931f3e10269f6df87fd65382cf7ccb3a26b43de9502b6`;
the durable checkpoint is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case10_xhalf_tail_lowcap_accept_9d172278_20260720T164557Z/provenance/CASE10_TERMINAL_HALF1_MSTEP_CHECKPOINT_20260721T0113-0400.md`.
Grid correction and forced after-max finalization were unset. Final
FSC/FSC-AUC audits and the sealer remain pending; correlation was not
computed.

## 2026-07-21 case-9 iteration-11 FSC checkpoint

The first numbered maps produced beyond case 9's old OOM boundary pass the
scientific gates. At iteration 11, low-cap job `11432807` has half-1,
half-2, and merged cross-engine FSC-AUC of `0.9997908657983370`,
`0.9998123118852400`, and `0.9998832836013437`. The merged GT FSC-AUC delta
is `-0.000014794080971658463`; the worst merged non-DC shell FSC is
`0.9992117910499290` at shell 103.

All 11 frozen prefix rows pass the 0.995 cross-engine and -0.002 GT-delta
gates. Analyzer status 2 is solely the intentional incomplete topology (11
RECOVAR rows versus 16 RELION rows). An initial live-directory staging
attempt correctly failed closed when iteration 12 arrived; the admissible run
used explicit links to the 22 frozen RECOVAR maps. The exact 45 consumed
GT/map inputs have manifest SHA-256
`abcf195e90278997f890948fbbaf70bb3626fec9ba0e787abb905905f2de96ca`.

The sealed provisional root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/k1_case9_provisional_it11_20260721T010500`;
its audit-note SHA-256 is
`8a807108ba96ada2fab207e8b1d1dc8849f54ad66106b06fba5a83d1891b8a32`
and seal-manifest SHA-256 is
`ac9c9d440cdc540d7ec318ef9fd38e98162c374800652dc7f64a8df0a2426e05`.
This accepts the newly crossed prefix quality, not terminal acceptance or
single-control attribution. Correlation was not computed.

## 2026-07-21 case-9 cap decomposition and case-33 seal repair

Exact 2x2 source-function arithmetic at the case-9 iteration-11 half-2 OOM
geometry gives caps 8,707 for 190M/4GiB, 3,879 for 64M/4GiB, 4,353 for
190M/2GiB, and 3,879 for 64M/2GiB. Thus either single knob lowers the failed
cap, while the 64M row target alone reproduces the successful bundle's cap
exactly. The 2GiB-only arm is 474 rows (12.2%) larger. The outer-tail guard is
inactive because the maximum local width equals the planned rotation block.

No single-knob arm has run on a GPU, so this calculation does not establish
which individual control completes science or isolate the allocator. The
JSON and method-note SHA-256 values are
`521d9bc9a94bd8eb97d2c272b6090bc55bdc2a44aff6a44608a130df2668c73c`
and
`336915db44e4ab1123789bbfbbd80c6feaa7e02b1584da3c86e9d18bfec021b6`.
Root:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/case9_cap_decomposition_20260721T012100`.
No map metric or correlation was computed by this arithmetic audit.

Separately, pending case-33 sealer job `11440295` originally depended on
`afterok:11440102`; that would omit durable rejection if the audit failed.
At 2026-07-21T01:18:49-04:00 its live dependency was repaired in place to
`afterany:11440102`. No launcher, threshold, science artifact, or sealer code
changed. The audit note is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case33_full_currenthead_7605c1b0_20260720T222000Z/provenance/SEALER_DEPENDENCY_REPAIR_20260721T011849-0400.md`
(SHA-256
`083f41579db12f73efe2bc923c75802aed3120c893a1917cad2ba217007bcf25`).

## 2026-07-21 case-33 iteration-11 FSC checkpoint

Current-head job `11440100` remains effectively identical to RELION through
the late numbered prefix. At iteration 11, half-1, half-2, and merged
cross-engine FSC-AUC are `0.9999997772584983`, `0.9999998224078882`, and
`0.9999998956685751`; the merged GT FSC-AUC delta is
`-0.000007236484065309412`. The worst merged non-DC shell FSC is
`0.9999990820988931` at shell 62.

All 11 frozen rows pass the 0.995 cross-engine and -0.002 GT-delta gates.
Analyzer status 2 is solely incomplete live topology (11 RECOVAR versus 14
RELION numbered rows). The audit used explicit numbered-map links and seals
45 consumed inputs with manifest SHA-256
`6b4520d7ff872606ae060da476aafd37d95e269f7206febf2c1b9fb13cee9795`.
The root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/k1_case33_provisional_it11_20260721T012700`;
its seal-manifest SHA-256 is
`2c2a682287b04dd7b1d272961f40ece77a934b5da9ee1c0b4315f4eb31c26b4b`.
Iterations 12--14 and terminal acceptance remain pending; correlation was not
computed.

## 2026-07-21 case-9 terminal half-1 memory checkpoint

Case-9 low-cap job `11432807` converged after numbered iteration 16 and
entered the unforced RELION final all-data pass at full Nyquist size 384. On
H100 `GPU-9f98ccbf-3c62-c54f-7409-7eb58845ad4a`, final half 1 completed all
16,607 score-only chunks/49,820 particles in 451.7 seconds. Its `(771, 771,
771)` x-half BPref M-step then ran with the outer-tail guard reducing cap 863
to 735 and completed all 5,648 chunks/49,820 particles in 310.8 seconds.
Host Hermitian enforcement, both large-accumulator repacks, and the half-1
manifest all completed before half 2 started. Total half-1 wall time was
857.3 seconds.

The manifest SHA-256 is
`7c911e120b4e49ceedd7307657bc84357768709179b3aaa85b3c41dd6415af1d`;
the durable checkpoint is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case9_bucketcap_accept_9d172278_20260720T095000Z/provenance/CASE9_TERMINAL_HALF1_MSTEP_CHECKPOINT_20260721T0151-0400.md`.
Its SHA-256 is
`ca933eab4a07b240c0f287fd0ef1bf5ffa40799ba53561dc53de17b6d00035e8`.
This accepts the bundled low-cap half-1 memory boundary only. It neither
isolates the controls nor accepts terminal map quality before half 2 and the
shellwise FSC/FSC-AUC auditors complete. Grid correction and forced after-max
finalization were unset; correlation was not computed.

## 2026-07-21 case-9 science job completes with favorable terminal FSC-AUC

Case-9 low-cap science job `11432807` completed `0:0` on H100
`GPU-9f98ccbf-3c62-c54f-7409-7eb58845ad4a` after 8:26:49 of Slurm elapsed
time. It converged after 16 numbered iterations and ran the unforced final
all-data pass at full Nyquist size 384 with grid correction off. Final half 2
completed all 16,727 score-only chunks/50,180 particles and all 5,700 x-half
BPref M-step chunks/50,180 particles. The terminal M-step cap was reduced
from 863 to 735, and host Hermitian enforcement plus both large-accumulator
repacks completed. The final half-2 manifest SHA-256 is
`2b1dceb40f86fc63e6638f7b021cfba4c529578a0631d1c39ef957753fbbf0f0`.

The launcher summary status is 0. Its primary correctness gate is favorable:
RECOVAR merged-versus-GT FSC-AUC is `0.31789819923059665`, RELION
merged/final-map-versus-GT is `0.31423365388057617`, and the delta is
`+0.00366454535002048`. Terminal cross-engine FSC-AUC is
`0.9955108928134183` merged, `0.9964421481056874` half 1, and
`0.996708665813443` half 2. The final-all-data half FSC-AUC is
`0.34388473707785183`, with FSC 0.5/0.143 crossings at shells 62/109.

This is provisional terminal science acceptance, not the independent seal.
Eligible audit jobs `11432810` and `11432811` remain scheduler-pending, and
sealer `11440427` correctly depends on both with `afterany`. The durable
checkpoint is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case9_bucketcap_accept_9d172278_20260720T095000Z/provenance/CASE9_SCIENCE_TERMINAL_CHECKPOINT_20260721T0211-0400.md`.
Its SHA-256 is
`a9583cb28f0ca8165117c294496b39a28648bce7e4fe3be4d2924f6d7512c7cb`.
The refinement archive and launcher-summary JSON SHA-256 values are
`95f3852a6710046eff068eb2ebfb02090a6fc036939598b25f08e735322925a7`
and `13e8a4600a6c47e8b03e7977a4b11eeb5b32291bdaff272a34742ecb4b573f1b`.
Case 9 still bundles the allocator and both cap overrides, so it does not
alone establish single-control causality. Correlation was not used as a gate.

## 2026-07-21 case-33 complete numbered trajectory

The frozen read-only audit of all 14 numbered case-33 rows passes. Current
sizes are exactly `[56, 68, 100, 128, 128, 128, 128, 128, 128, 128, 128,
128, 128, 128]`; every row passes the merged cross-engine FSC-AUC threshold
0.995 and RECOVAR-minus-RELION merged GT FSC-AUC threshold -0.002. At
iteration 14, merged cross-engine FSC-AUC is `0.9983139369494863`, RECOVAR
merged-versus-GT is `0.799559821719684`, RELION merged-versus-GT is
`0.7961875736492268`, and the GT delta is `+0.0033722480704572178`.

The last row also exposes a half-specific discrepancy that the passing merged
gate must not hide: half-1 cross-engine FSC-AUC is `0.9504239364636927`,
versus `0.9970095099604138` for half 2. The half-1 minimum non-DC shell FSC
is `0.7437240350067739` at shell 17. RELION half 1 versus GT is
`0.7337998926958064`, below RECOVAR half 1 (`0.7987987477020957`) and RELION
half 2 (`0.7990197613205833`), while RECOVAR's merged GT metric improves on
RELION. This localizes the late-row discrepancy to RELION half-1 state/map
behavior rather than a symmetric RECOVAR quality loss; terminal products and
state provenance still require inspection.

The analyzer consumed exactly 57 hashed numerical inputs from explicit
numbered-map stages. Its JSON and shellwise-NPZ SHA-256 values are
`d45e0c203e8f823bc311a5d2fcc12a0e564b74e31b68ddc4148bc552cd777f2d`
and `b5bfcfb8ca9c9472bec04616a48575c350a24891bc4cb1f905fa7c00914cf452`.
The sealed audit root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/k1_case33_numbered14_20260721T021800`;
its method-note and seal-manifest SHA-256 values are
`618aab5bc2adf02c52e9a8be2d3c3a9efb4066086c384da78ae54f01d1bda029`
and `fccc6abdb15c2b3f7ca85f12155ff3b389502e00ece8d20e49dde60598d3fc02`.
An initial invocation against a full RELION directory correctly failed closed
before numerical evaluation because it exposed final products to a
numbered-only RECOVAR stage; that error is retained in the seal. Science job
`11440100` is still running its valid final all-data pass, followed by audit
`11440102` and sealer `11440295`. Correlation was not computed.

## 2026-07-21 case-33 terminal rejection and state localization

Case-33 science job `11440100` subsequently completed `0:0` after 4:14:05.
It converged after 14 numbered rows and ran the unforced final all-data pass
at size 128 with grid correction off. A local exact mirror of pending audit
job `11440102` rejects exactly one map gate: terminal merged cross-engine
FSC-AUC is `0.9727626280594356`, below 0.995. Half-1 and half-2 cross-engine
values are `0.972007960452858` and `0.995298199193914`. This is not a GT
quality loss: RECOVAR merged-versus-GT FSC-AUC is `0.9769353607293773`,
RELION is `0.9457523102520277`, and the delta is favorable by
`+0.031183050477349594`. Correlation was not computed.

An exact-identity diagnostic aligns all 400,000 particles by unique
`rlnImageName` and rules out a simple row-order error. Iteration 13 remains
closely aligned, but iteration 14 is half-specific: half-1 rotation-geodesic
p95 is `1.0449055358065685e-5` degrees while half 2 is
`0.922373453820857` degrees. RELION's own `run_it014_data.star` Pmax column
is internally asymmetric, with half means `0.16330775623382404` and
`0.8677921732644837`; both were near 0.165 one row earlier. In contrast,
RELION optimiser `rlnAvePmax=0.163466` agrees with RECOVAR's M-step average
`0.16395849217264247` to `+0.0004924921726424669`. The particle Pmax column
therefore is not a coherent whole-dataset oracle at this boundary.

Final sampling parameters are not the gap: expected rotational/translational
accuracy matches `0.533` degrees / `0.6715` Angstrom, perturbation is
`-0.46612`, HEALPix order is 7 with fine order 8, and the translation grid
matches RELION's `1.53` / `1.02` Angstrom range/step. Terminal pose differences
remain measurable (angular-error p95 `1.806696629320861` degrees), but the
combination of accepted numbered merged maps, favorable terminal GT FSC-AUC,
matched final sampling, and a half-asymmetric RELION particle checkpoint does
not justify an algorithm change without an independent coherent oracle.

The particle-state diagnostic JSON and compact-array SHA-256 values are
`aac7dacdd0ffdc33dda5ecffe0c71170a6e9312b9df7571e06115e19fae8a2d5`
and `b76919b053ce59e5679bcfd6f3d8ecc19c18396a78e83ad7245cfdd8c1d31672`.
Root:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/k1_case33_particle_state_terminal_20260721T024500`.
The complete durable note is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case33_full_currenthead_7605c1b0_20260720T222000Z/provenance/CASE33_TERMINAL_REJECTION_AND_STATE_DIAGNOSTIC_20260721T0250-0400.md`
(SHA-256
`8b1208b54ffe1b97f216cf58adcfe7670f1f5936d9b21bb2d744c5ced2f04c34`).
Submitted audit `11440102` and `afterany` sealer `11440295` remain the durable
scheduler graph.

## 2026-07-21 case-10 low-cap completion rejects terminal parity

Case-10 low-cap job `11421265` completed `0:0` on A100
`GPU-4bccbe72-c64a-5f5f-1fa8-ecf0bf6acf37` after 14:20:23. It converged
after 15 numbered rows and ran the unforced final all-data pass at full size
384. Half 1 completed 49,878 x-half BPref M-step chunks in 4,964.2 seconds;
half 2 completed 50,011 chunks in 4,954.7 seconds. Both used the outer-tail
cap `863 -> 60`, completed host Hermitian enforcement and both 771-cube
accumulator repacks, and wrote final manifests. The joined-FSC/tau2 host
fallback took 220.1 seconds and final reconstruction took 7.4 seconds.

The memory/completion result is positive, but the launcher FSC/FSC-AUC
summary rejects terminal cross-engine parity. RECOVAR merged versus RELION
final/merged FSC-AUC is `0.9830065035340728`; half-1 and half-2 values are
`0.9858434465381396` and `0.9853237654697927`, all below the 0.995 gate.
RECOVAR merged-versus-GT is `0.07089150679155766`, RELION final-versus-GT is
`0.07076315983428035`, and the delta remains favorable by
`+0.00012834695727731`. This is therefore a parity rejection, not a GT-quality
regression. Correlation values in the generic launcher output were not used.

The run bundles `cuda_malloc_async`, a 64-million row target, a 2-GiB large-JIT
cap, and the outer-tail guard. It does not establish a safe source default:
the package fixes the OOM boundary but changes terminal cross-engine FSC-AUC
materially. Do not promote the proposed conservative g384 x-half defaults
without causal decomposition or an algorithmically identical lower-memory
path.

Refinement and summary-JSON SHA-256 values are
`ddf15c039f626f79c7f75e8a4a42056ddc90ebe7b34210bd6d660645388554b4`
and `e890da99e3885b86ac5e3b960119875eff75d2d52039f91c14932447e2572068`.
The final half-2 manifest SHA-256 is
`cd2e13aeca426ee244c571d30f1a9cf0cb16c5e07a3268806be5d225d4f360c5`.
The complete science note is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case10_xhalf_tail_lowcap_accept_9d172278_20260720T164557Z/provenance/CASE10_TERMINAL_SCIENCE_REJECTION_20260721T0310-0400.md`
(SHA-256
`450714204192bf9201e4574d3d40d97d24d3c3c852e11d65951352344ed3706f`).
Submitted shellwise auditors `11421266` and `11421267` remain eligible and
scheduler-pending; `afterany` sealer `11440428` retains the independent durable
graph.

## 2026-07-21 local mirrors freeze case-9 acceptance and case-10 final-only rejection

Read-only local mirrors of the pending independent FSC and intermediate-state
auditors have completed with outputs separate from both canonical case roots.
They used exact source commit
`9d1722781e1d6c5fc5b2ad0e15ebba3a2becbab0`, the pinned case environment,
and FSC/FSC-AUC rather than correlation.

Case 9 passes completely: all 16 numbered rows, terminal maps, trajectory
topology, and selected intermediate arrays pass. Iteration-16 merged
cross-engine FSC-AUC is `0.9998449171877147`, with RECOVAR-minus-RELION
merged GT delta `-0.0000031022744303998984`. Terminal merged cross-engine
FSC-AUC is `0.9955108928134183`, and the terminal GT delta is favorable by
`+0.003664545350020476`. The separate audit root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/k1_case9_local_mirror_20260721T025200`.
FSC-JSON and intermediate-JSON SHA-256 values are
`34774cf696aeee2b8f1b7900309af673bd7d324b81cf481e0eaf3416758cf477`
and `3f0ba3cbf90d39db236451abe56229aa436e5752ca240c48a206bf8620e1cfbf`.
The durable note SHA-256 is
`917eb9da9cafc9d749a5468ee280c681f499ae20dd0653c68fbc35921e702977`.

Case 10 has exact 15-versus-15 numbered topology, and every numbered row
passes. Merged cross-engine FSC-AUC ranges from
`0.9999630945465691` to `0.9999999547661423`; iteration 15 is
`0.9999672271217562`, with GT delta
`+0.000011730968016115858`. The intermediate trajectory also passes all 15
rows without a topology or numeric-artifact failure. The only FSC failure is
the final all-data merged map, `0.9830065035340728 < 0.995`; final GT delta
remains favorable by `+0.00012834695727731438`. The rejection therefore
arises specifically at finalization, not in the ordinary numbered EM
trajectory.

The separate case-10 audit root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/k1_case10_local_mirror_20260721T031300`.
FSC-JSON and intermediate-JSON SHA-256 values are
`ec099ad1e4b0079e8d57f4374527e25e1a81cc89777dbb90b33ce61002821f9f`
and `0aa61a53db4a8b993bcf7ba846388fe987a1181139ad728af9b5a58c2634590d`.
The durable note is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case10_xhalf_tail_lowcap_accept_9d172278_20260720T164557Z/provenance/CASE10_LOCAL_TRAJECTORY_AUDIT_20260721T0327-0400.md`
(SHA-256
`d90d6416d53b36691953975c4168d5d61cc5d8d44b80995d18eaf61e779880ad`).
Submitted jobs `11432810`, `11432811`, `11421266`, and `11421267`, plus their
`afterany` sealers, remain the independent durable scheduler graphs.

## 2026-07-21 case-7 panel confirms upstream state/reference locus

Retry job `11442740` completed `0:0` on H100 `della-h20g3` in 3:45:14 and
produced all 48 required iteration-11 fused-posterior captures: resident and
exact-RELION-state/reference arms for a half-balanced panel of 12 low-Pmax
pose-tail particles and 12 matched pose-stable controls. Both arms used source
commit `77bcf3bd7f45760ab0671c4883d91a453d58113a`.

The corrected v2 audit confirms the intended discriminator. Stable controls
retain the same latent and physical winner in 12/12 particles, with candidate
and physical-support median Jaccard 1, posterior total variation
`0.009058262567317964`, reconstruction-support median Jaccard 1, and zero
median winning-rotation displacement. The pose-tail retains the same latent
winner in only 1/12 particles and physical winner in 2/12; its median
candidate posterior total variation is `0.059013740489311566`, physical-pose
TV is `0.058522763`, reconstruction-support Jaccard is `0.8333333333333333`,
and winning-rotation displacement is `1.8481429893285553` degrees.

Candidate topology remains matched at median Jaccard 1 in both cohorts. The
cohort-specific posterior and winner response to substituting exact incoming
state/reference therefore rejects common-input GPU near-tie arithmetic as the
leading cause and keeps the locus in upstream accumulated state/reference
drift.

The preserved v1 output exposed an analyzer-only six-decimal translation-key
bin-edge artifact: stable winners had zero reported physical overlap despite
zero rotation displacement and only `4e-8`--`2.7e-7` pixel translation
differences. V2 uses five decimals, well below the at-least-0.0835-pixel
physical grid separation, and restores all stable physical winners without
changing candidate, posterior, rotation, or reconstruction metrics. V2 JSON
and Markdown SHA-256 values are
`d4ce702b3bae8d7baacb4e9dfe7124714141b825d709f99fe23660f16692b836`
and `6ddf51ae49a8a923137e20b5566f1d65fef8db8e1fed625499ddcbc545aff98c`.
The complete durable note is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_it11_stratified_posterior_77bcf3bd_20260720T214500Z/provenance/CASE7_IT11_PANEL_RETRY2_RESULT_20260721T0345-0400.md`
(SHA-256
`9ee603c6295459d890264d255c25da8fbe9469b4581281bfda2f4b26a19ecc08`).
This bounded posterior diagnostic is non-gating; correlation was not computed.

### Direct RELION-target closure for the selected case-7 cohort

A read-only identity-aligned follow-up verifies that the exact-state/reference
arm moves the selected particles to RELION's physical iteration-11 target,
not merely away from the resident arm.  In the updated `77bcf3bd` resident
run, 11/12 selected tail rotations and 7/12 translations differ from RELION
by more than `0.001`; the exact arm places all 12 rotations and all 12
translations within `0.001`.  Tail median absolute Pmax error contracts
`107.515x`, from `0.02562694075012209` to `0.00023835652923584472`, and the
maximum contracts `150.670x`, from `0.15097260182571415` to
`0.0010020107574463255`.

This closure is cohort-specific.  Across all 100,000 particles, the same
exact arm still has rotation-geodesic p95 `1.8445827782316297` degrees and
Pmax absolute-error p95 `0.2693974259247777`.  It therefore does not justify
a production state replay or relaxed gate; the active component intervention
must be interpreted only as a causal split for this sealed ambiguous cohort.
The complete note is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_it11_stratified_posterior_77bcf3bd_20260720T214500Z/provenance/CASE7_EXACT_ARM_RELION_TARGET_CLOSURE_20260721T0408-0400.md`
(SHA-256
`4d2f28a593e0ae9e9104f54275ed2ab96fdbd4a71ff4ed91e798f707b3710823`).
The identity-aligned JSON SHA-256 values are
`0be054fc856cfb1f5cb3af29fc3f2e63e513801c2856dc2450d6fee62de3559d`
for the exact arm and
`987c76da42c8ba12c22d1a08985369573a9cae1e9e6f0827b7ba234afa1b2464`
for the resident arm.  Significant-support totals are not interpreted because
the archived RECOVAR trajectory and RELION serialized counts have different
ownership semantics.  Correlation was not computed.

### Full-population response to exact incoming state/reference

A second read-only comparison makes the selected-panel result quantitative
across all 100,000 exact identities. At a 0.1-degree threshold, exact incoming
state/reference reduces the physical rotation tail from 5,793 to 5,064:
961 resident tails close, 232 previously matched particles open, and 4,832
remain discordant. At 0.1 pixels, the translation tail falls from 6,230 to
4,963 (1,471 close, 204 open). Absolute Pmax error improves for 71,678
particles, and its median contracts from `0.015977736459732023` to
`0.006330115291595495`.

The effect is half-balanced rather than allocation-half specific. Half 1 has
490 closed versus 106 opened rotation tails; half 2 has 471 closed versus
126 opened. Closure is concentrated in uncertain particles: the RELION
`0.25 <= Pmax < 0.5` cohort changes from 1,807 to 1,199 tails (667 close,
59 open), while the `Pmax >= 0.75` cohort is nearly neutral (79 close,
84 open). Thus the 24-particle panel was directionally representative, but
exact state/reference is not a global replay repair. The active component-arm
experiment remains necessary to split pose from map state.

The reproducible analyzer and JSON live under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/k1_case7_exact_state_relion_target_audit_20260721T040500`;
their SHA-256 values are
`3af8b2b83400dee2d68395fd903098e9c83fcb8e762f77143b6a7faea62721f7`
and
`74161261039659161767bebee41c2898f9eb960a26a8f32fd9570d3a55c4db33`.
The durable interpretation note is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_it11_stratified_posterior_77bcf3bd_20260720T214500Z/provenance/CASE7_FULL_POPULATION_STATE_REFERENCE_SPLIT_20260721T0419-0400.md`
(SHA-256
`1453b5632771f04d7201a3d36ffeb98e013c88fcebc4b76059049946035c681b`).
This is non-gating and uses no correlation; map acceptance remains FSC/FSC-AUC.

The direct stored arm outputs further separate changed from persistent winners.
Only 1,203/100,000 rotation Euler rows change between resident and exact
state/reference. All 961 closed tails and all 232 opened tails change, while
4,822/4,832 persistent tails and all 93,975 stable rows are exactly identical
between arms. Therefore the persistent residual is overwhelmingly the same
resident physical winner that exact state/reference does not dislodge, not a
switch to a second wrong neighbor. Persistent median arm-to-arm absolute Pmax
change is nevertheless `0.01084938645362854`, so confidence can move without
changing the serialized rotation winner.

The v2 analyzer/JSON SHA-256 values are
`db28bf1c9372a4a8e4af5d95f76be0eea6d39fe0dc90aab45199fadad6523faf`
and
`bb0c54c8280931b4858c74a3531be0d09e85add9c5f61d24e1fa84414dd0718e`.
The durable note is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_it11_stratified_posterior_77bcf3bd_20260720T214500Z/provenance/CASE7_FULL_POPULATION_ARM_WINNER_RESPONSE_20260721T0435-0400.md`
(SHA-256
`017eadfa1b1f0ea98f9096908f3f680f78ef4a380419509845030e57e042a8c6`).

A complementary 48-particle residual panel is now running as Slurm job
`11451167`, with fail-closed `afterany` audit `11452890`. Original audit
`11451209` was canceled without running after direct inspection proved its
queued Slurm snapshot still contained the obsolete v1 hash/schema. The panel
contains six
persistent and six opened targets per half, each with a stable matched control,
and captures resident then exact-state/reference posteriors sequentially on one
H100. The selection and analyzer SHA-256 values are
`3335ddfb0821c8daad1981619d9ed8effc6d8b6944ed1b5d9650f151b8d234ad`
and
`5e3479b827df78dab166455ab9a2a72503d6d4db3884961fcb136b1ee181ac56`.
Run root:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_it11_residual_posterior_77bcf3bd_20260721T042500Z`.

### Strict-ledger boundary for current-head case-20/case-33 replays

The completed current-head case-20 and case-33 runs are RECOVAR-only replays
against immutable RELION oracles. They are useful FSC diagnostics, but neither
is a same-physical-GPU pair. Case 20 compares a canonical oracle produced on
`GPU-b9c5d089-cde3-7f8b-717b-6f61c49ef1ae` with a current replay on
`GPU-2ee3da91-970a-6714-84df-530aefe04a08`; case 33 compares
`GPU-2ced982b-7cc9-32c2-a413-a600b1c00a1f` with
`GPU-49c1a223-be61-858b-49d8-d8b0347ac252`.

Their final merged cross-engine FSC-AUC values are respectively
`0.9977609799825519` and `0.9727626280594356`, with RECOVAR-minus-RELION
GT FSC-AUC deltas `+0.00114188226197158` and `+0.0311830504773496`.
Those results remain diagnostic and must not silently replace canonical rows
in the strict full-34 ledger. A future v3 ledger may use the newer case-9 and
case-10 roots, whose RELION/RECOVAR pairs do have matching physical GPU UUIDs,
but cases 20 and 33 require either explicit cross-allocation labelling or new
paired reruns.

The complete provenance boundary is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_full34_superseding_v2_9d172278_20260720T072521Z/provenance/CURRENTHEAD_REPLAY_GPU_PROVENANCE_BOUNDARY_20260721T0445-0400.md`.
Its SHA-256 is
`a5ab9bdb05c2ed600d632405d0b1d7c35e0c291d96c9ffddd62a085eb2e40901`.
Pending v2 jobs `11409642` and `11409643` remain preserved as the historical
fail-closed graph; they are not relabelled as the newer replacement topology.

The staged v3 eligibility audit now proves that the newer case-9 and case-10
roots are admissible strict replacements. Both science jobs completed from a
clean `9d1722781e1d6c5fc5b2ad0e15ebba3a2becbab0` checkout, and each paired
RELION/RECOVAR UUID triple is exact. The 17-image deterministic particle-stack
sample has maximum absolute delta `4.76837158203125e-7`; relative L2 is
`3.0881684775820945e-8` for case 9 and `2.437535301379154e-8` for case 10.
Reference-map maximum absolute deltas are `1.1175870895385742e-8` and
`5.587935447692871e-9`. Exact case config, generation config, CTF, and pose
artifacts plus normalized STAR identities also pass.

The v3 root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_full34_superseding_v3_33ff4287_20260721T044600Z`;
the eligibility JSON SHA-256 is
`81b0d743ca3a6c0217598a31cf9f5105b19ea9f7d67f4ea82126c57616bbc174`.
This is eligibility only: independent case-9/case-10 trajectory and
intermediate audits remain mandatory before aggregate construction.

The fail-closed v3 terminal graph is now submitted. Ledger builder `11452420`
waits `afterany` on `11432810`, `11432811`, `11421266`, and `11421267`;
sealer `11452421` waits `afterany` on the builder. A preterminal self-test
resolved all 34 rows and marked only cases 9/10 structurally incomplete due to
the pending audits. The sealer returned the expected status 2 while preserving
separate structural/parity classifications.

The job registry is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_full34_superseding_v3_33ff4287_20260721T044600Z/JOB_REGISTRY.md`
(SHA-256
`77842f24be89f021d3f95ed6c0f7b89b15a534f1e2369003edbe99fa911f18fd`).

### Residual-panel direct RELION-target discriminator

Before job `11451167` produced any iteration-11 captures, its scratch-only
analyzer was extended to distinguish missing support from lower scoring. For
each arm and identity, analyzer v2 directly checks whether the immutable
RELION iteration-11 physical pose is in the candidate set, its posterior
mass/rank, reconstruction-pruning membership, nearest support displacement,
and winner displacement. Target tolerances are 0.001 degrees and 0.001 pixels;
STAR rows are identity-aligned rather than assumed ordered.

A completed-capture functional smoke shows both discriminator paths. Stable
image 48122 has the RELION target at rank 1. For tail image 11540, the resident
arm contains the target but ranks it second and selects a winner 1.875001
degrees away; exact state/reference promotes the target to rank 1 within
`1.8e-6` degrees. The resident target/winner posterior ratio is
`0.9696178825` (log gap `0.03085322066`). This validates that even near-tied
target-present-but-lower-scored outcomes will not be misclassified as
search-support loss.

Analyzer SHA-256 is
`5e3479b827df78dab166455ab9a2a72503d6d4db3884961fcb136b1ee181ac56`;
the updated fail-closed audit launcher SHA-256 is
`86615bdfdcc042d351d34778e8d27d3cd7e831c44c14778c65ab6282d2e992b5`.
The amendment note is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_it11_residual_posterior_77bcf3bd_20260721T042500Z/provenance/ANALYZER_V2_AMENDMENT_20260721T0455-0400.md`
(SHA-256
`696740a5ec112d77f2beec6c1aa061bfeada05d9904fa4281b088d861552b011`).

Applying this discriminator to the already completed panel24 proves that its
resident tail is a mixed boundary. All 12 exact-state captures contain and
rank the RELION target first. Resident captures split into one already-correct
winner, seven target-present rank-2 near ties (target/winner ratios
`0.8657448936`--`0.9859998237`), and four missing targets. Three missing cases
are displaced only by one `0.0835`-pixel translation-child step; the fourth is
missing the target rotation by `1.844585` degrees. Exact incoming
state/reference therefore repairs both local search support and relative
posterior scoring.

All 12 stable controls contain and rank the RELION target first in both arms,
with candidate-support Jaccard 1 and median posterior TV `0.00905826257`.
The tail median posterior TV is `0.05901374049`, with three support-shift cases
at approximately 1. This rejects a single support-only or scoring-only repair.
Analyzer/JSON SHA-256 values are
`4ab6c895ed626e1949e423835ba67e117727a43777413b9cefc18426d98e42ed`
and
`2c779f535b372221e6739a8550c355d32f3fd8ed1d73de346f978a74e2271743`.
The durable note is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_it11_stratified_posterior_77bcf3bd_20260720T214500Z/provenance/CASE7_COMPLETED_PANEL_RELION_TARGET_RANK_20260721T0512-0400.md`
(SHA-256
`8180615c7d242f192e9b9e7a6997ae46f4ae6c483e10c25e07a979933f2ced8e`).

The already-running state-component science job `11449766` is unchanged.
Before any iteration-11 captures existed, the analyzer was upgraded to test
every arm directly against the immutable RELION target pose. The v3 audit
retains posterior TV, winner displacement, and reconstruction
support Jaccard, and now reports target presence/support membership, rank,
mass, target/winner posterior ratio, nearest-support displacement, and winner
displacement. This makes `restore_recovar_poses` an explicit discriminator for
support/centering loss and `restore_recovar_maps` one for relative score/rank
loss instead of inferring either from arm-to-arm TV alone.

It also freezes independently selected subcohorts from the completed resident
panel: seven target-present rank-2 tails, four target-absent tails, one
already-rank-1 tail, and twelve stable controls. This prevents aggregate tail
statistics from mixing the score and support mechanisms under test.

Direct `scontrol write batch_script` inspection found that original pending
audit `11450599` still contained Slurm's stale v1 submission snapshot; editing
the path after submission had not changed the queued script. It was canceled
without running, and replacement `11452889` was submitted
`afterany:11449766`. Its queued snapshot was verified to contain v3 and the
current hashes. Residual audit `11451209` had the same stale-snapshot issue and
was replaced by verified v2 audit `11452890` afterany `11451167`; neither GPU
science job was changed or interrupted.

Target matching uses `0.001` degree and `0.001` pixel tolerances and aligns the
immutable STAR by particle identity. The completed-capture target-summary
smoke passed. Analyzer, predeclared-subtype JSON, shared target helper, RELION
STAR, audit launcher, and contract SHA-256 values are
`b5ea0a03c451c6d7aaf85d2a1c0961b207f3027add92be7be8dddf6f62a99309`,
`02d94ce3d78b559bcdac3dab255789d5df12ae0ec9252fca3c6a0c83d7204f52`,
`5e3479b827df78dab166455ab9a2a72503d6d4db3884961fcb136b1ee181ac56`,
`022865cdc40d4d4c5813078d81f6f421f2f54949d04e4762498659ce271a9b55`,
`1c3c5395694d34000e276bfaf8a18273287a01a8184790f9c4a7c78862046826`,
and
`566a3e0da1d08b83119fe52411388bbfbf64b542725b163de54b884751439f65`.
The original v2 amendment is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_it11_state_component_77bcf3bd_20260721T034700Z/provenance/ANALYZER_V2_AMENDMENT_20260721T0518-0400.md`
(SHA-256
`0658833c449617880621f9e1e249d8cf4e9048a0a279d6f49c06da0a67e1e412`).
The scheduler-corrected v3 supersession note is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_it11_state_component_77bcf3bd_20260721T034700Z/provenance/AUDIT_SUPERSESSION_V3_20260721T0526-0400.md`
(SHA-256
`8d94944cc288520de7340c5673c2ceb9eb96df48e53a113e6044af827d8f9d74`).
The residual supersession note is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_it11_residual_posterior_77bcf3bd_20260721T042500Z/provenance/AUDIT_SUPERSESSION_V2_20260721T0526-0400.md`
(SHA-256
`5b89f65cc77923b9292cb346fc47360fd4dbb8b184a33f7395b2b0560b3416bb`).
The audit remains diagnostic/non-gating and does not compute correlation.

## 2026-07-21 case-10 final-transition FSC decomposition

The final-only case-10 rejection begins in the unfiltered final half maps, not
only the stored merged-map postprocessing. Last-numbered merged cross-engine
FSC-AUC is `0.9999672271217562`. Final unfiltered half-1 and half-2 values are
`0.9858434465381396` and `0.9853237654697927`; their explicit half-average is
`0.9857213361877246`. Stored final merged products fall another approximately
0.0027 to `0.9830065035340728`.

Both engines make essentially the same large numbered-to-final transition:
RECOVAR's own half-average transition FSC-AUC is `0.16724021722635543`,
RELION's is `0.16722879541673202`, RECOVAR final versus RELION numbered is
`0.1672287703373412`, and RECOVAR numbered versus RELION final is
`0.16720296473855617`. Mixed-engine transitions reproduce the opposite
engine's own transition to roughly `1e-5`--`3e-5`. Final half-average GT
FSC-AUC is likewise tied (`0.07104744838232087` RECOVAR versus
`0.07107504415350836` RELION). This rejects a unique final reconstruction
branch as the leading cause and is consistent with small autonomous
last-numbered state/reference differences being amplified by the shared
Nyquist expectation.

The gap is broadband: the numbered merged FSC never falls below 0.995, while
both final halves first cross below it at shell 24 and the half-average at
shell 25. The half-average minimum is `0.980867547381934` at shell 188; the
stored merged minimum is `0.9662718962179357` at shell 151. Do not change
final reconstruction, enable grid correction, or loosen the acceptance gate
from this evidence. JSON and shellwise-NPZ SHA-256 values are
`a2f485b11fda298960e9c200b9a4855c157861494d562c9b70fc21080cdbc924`
and `0b6b220ed199d2a0dbdc5a2e57e66c11ba23b1552f5e96432021faf94508fe38`.
The complete durable note is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case10_xhalf_tail_lowcap_accept_9d172278_20260720T164557Z/provenance/CASE10_FINAL_TRANSITION_FSC_LOCALIZATION_20260721T0348-0400.md`
(SHA-256
`c6ecb543cbb9626835e7547aa06a900daf8cb02dd7cf8586576abbd7fc2681db`).
Correlation was not computed.

## 2026-07-21 active case-7 state-component discriminator

Slurm job `11449766` is running a three-arm exact-prefix decomposition
sequentially on one H100 allocation. It repeats the all-RELION
state/reference control and then restores only RECOVAR-produced target-boundary
maps or only target-boundary poses. All arms use source commit
`77bcf3bd7f45760ab0671c4883d91a453d58113a`, the sealed 24-particle panel,
the same forced numbered current-size/HEALPix schedule, and the same seed. They
stop after numbered iteration 11 and explicitly skip final all-data; grid
correction and forced after-max finalization are unset.

The all-RELION repeat controls same-allocation arithmetic. Tail/stable
posterior TV, winner identity/displacement, and reconstruction-support Jaccard
will test whether current-boundary maps or poses independently recreate the
tail response, versus the discrepancy requiring earlier accumulated state.
The run and runtime roots are
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_it11_state_component_77bcf3bd_20260721T034700Z`
and
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/k1_case7_it11_state_component_77bcf3bd_20260721T034700Z`;
both contain `SAFE_TO_DELETE`. Launcher and run-contract SHA-256 values are
`c30c08a378ced38208f4236bd5b633550b01b9d7761aaa2f440242339868fe2d`
and `378084186e3dc48804dfabd35bfcc639b8951cc175ef68a2af2f378fe7a628f0`.
The submission note SHA-256 is
`90ac71d738a3710816681b70a50ce1ffffa7c6f67116ff6cc4e125f36f96d0e2`.
This is diagnostic/non-gating; correlation is not computed.

## 2026-07-21 case-7 exact control is reproducible across H100 allocations

The first arm of state-component science job `11449766`,
`all_relion_repeat`, completed successfully on `della-h19g1` and a different
physical H100 from independent exact control job `11442740` on `della-h20g3`.
All 24 immutable RELION targets are present, in reconstruction support, and
rank 1 in the new repeat, including all predeclared tail failure subtypes.

Across the 24 images, the prior and repeated exact controls have candidate,
physical-pose, and reconstruction-support Jaccard `1/1/1`; latent and physical
posterior TV `0/0/0`; zero best-pose displacement; and identical latent and
physical winners for 24/24 images. Thus the exact-state/reference control is
fully reproducible under the audited posterior/support metrics across GPU
allocations. Allocation-specific arithmetic is not a viable explanation for
the resident-versus-exact case-7 response. The map-only, pose-only, and
residual arms remain necessary before changing production code.

The new capture-manifest, refinement-result, and wall-time SHA-256 values are
`31cf0d14eb831eb9c022658ad0af85a137f658f92041bb21bf599b82e66f5a0b`,
`c4978e29cbdef4e69a10fd4fc2c50ba2fa62196da9ee5f4f0fb5607d06eb51ef`,
and `781979ef966a6bff2c42914efcf42646e38a98a39dd895310f6614e67f7cb521`.
The durable partial-result note is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_it11_state_component_77bcf3bd_20260721T034700Z/provenance/ALL_RELION_REPEAT_CROSS_ALLOCATION_RESULT_20260721T0551-0400.md`
(SHA-256
`8f145d75670f08e0136d253bafdc5d43630d84e31f46ed882983067857b3a45d`).
This is diagnostic/non-gating; correlation was not computed.

## 2026-07-21 historical K=1 v2 matrix sealed fail-closed

Historical v2 sealer job `11409643` never started with its oversized
128-GiB/12-hour CPU request. It was canceled and replaced without changing
inputs or sealer logic by job `11453977`, requesting one CPU, 4 GiB, and ten
minutes. `scontrol write batch_script` verified the submitted snapshot.
Replacement job `11453977` ran on `della-i13n21` for two seconds, used 56012
KiB maximum RSS, and exited `2:0` as designed for a failing seal. Stderr is
empty and every output-manifest hash verifies.

The terminal 34-case v2 seal is structural `fail` and parity `fail`. Cases 9
and 10 are the only structural failures because the historical ledger retains
their original incomplete rows. The earliest FSC/FSC-AUC parity failure is
case 2 at iteration 3, merged GT FSC-AUC delta
`-0.003274589 < -0.002`. This closes v2 as a historical fail-closed record; it
does not supersede or weaken the strict v3 builder/audit graph.

Terminal JSON, Markdown, and seal-manifest SHA-256 values are
`804ade09bfb022887cb9c6045d615127b818dff394315297b4f79a43c1dcef52`,
`9f842fd4421dc7fd2990d7a6c01aee64fd23ee13d2799d4fd48b296a3c613de3`,
and `5a35ec9b360fddea4708439677bc6511d177cb86374d83c5f6a445f74d543a5a`.
The scheduler/result note is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_full34_superseding_v2_9d172278_20260720T072521Z/provenance/SEALER_RIGHTSIZE_RESUBMISSION_20260721T0556-0400.md`
(SHA-256
`e4cc94f33a99976ecb51bdd7b8fdbdf1c0d6e1b25be66f8796d8dd637a5524c1`).
Correlation was not computed or gated.

## 2026-07-21 strict K=1 v3 audit graph repaired fail-closed

Right-sizing the four pending independent audits exposed two distinct issues.
The intermediate audits fit comfortably and completed successfully: case 9
job `11432810` used 543884 KiB in 24 seconds and case 10 job `11421267` used
622232 KiB in 20 seconds. Complete trajectory audits exceeded 8 GiB and jobs
`11432811`/`11421266` terminated `OUT_OF_MEMORY 0:125`. More importantly, the
pre-existing v3 registry had case-9 trajectory/intermediate roles reversed,
and the builder accepted any terminal Slurm state, which could permit an OOM
job when stale local artifacts existed.

Pending builder/sealer `11452420`/`11452421` were canceled without running.
Trajectory retries `11454201` and `11454202` use one CPU, 32 GiB, and 30
minutes. The corrected role registry is case 9 trajectory `11454201`,
intermediate `11432810`; case 10 trajectory `11454202`, intermediate
`11421267`. Builder validation now rejects infrastructure outcomes while
allowing only `COMPLETED 0:0` or the audit launchers' intentional fail-closed
`FAILED 2:0`.

Corrected builder `11454286` waits `afterany:11454201:11454202`; sealer
`11454287` waits on the builder. Direct batch-script inspection verifies both
aggregate jobs use one CPU, 8 GiB, 30 minutes, and revalidate their manifests.
Both static/eligibility manifests pass, and a corrected preterminal build
resolves 32/34 rows with only the two running trajectory retries incomplete.

Durable graph-repair and resource-result notes are
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_full34_superseding_v3_33ff4287_20260721T044600Z/provenance/V3_GRAPH_REPAIR_20260721T0610-0400.md`
(SHA-256
`b3dd960f88c70cd9f4caeb07ebec3d730ef465a5259de4cfedd9726cc3ad648a`)
and
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_full34_superseding_v3_33ff4287_20260721T044600Z/provenance/AUDIT_RESOURCE_RIGHTSIZE_20260721T0601-0400.md`
(SHA-256
`7036091ba125a8f98d13d2ee5e9bdb124f6ca5724ec9a4fcf6b3c44a011a7936`).
No acceptance threshold, fixture, science artifact, or FSC/FSC-AUC result was
changed; correlation remains uncomputed and ungated.

## 2026-07-21 strict K=1 v3 matrix is structurally sealed and parity-failing

The preceding active-graph description is superseded. Both 32-GiB trajectory
retries produced complete evidence near their 30-minute limit: case-9 job
`11454201` completed `0:0` in 28m11s at 8669880 KiB maximum RSS; case-10 job
`11454202` intentionally exited `FAILED 2:0` for its FSC gate in 27m58s at
8665864 KiB. Case 9 passes with final merged cross-engine FSC-AUC
`0.9955108928134183`. Case 10's intermediate audit passes, but its final
merged cross-engine FSC-AUC is `0.9830065035340728 < 0.995`; its final
RECOVAR-minus-RELION merged GT FSC-AUC delta is positive
`0.00012834695727731438`.

The first corrected aggregate pair `11454286`/`11454287` was canceled without
running as a safety hold while those near-limit outputs were inspected. Fresh
builder `11454959` then completed `0:0` and resolved all 34 rows structurally.
Sealer `11454960` intentionally exited `2:0`: structural status `pass`, parity
status `fail`. The earliest parity failure remains case 2 iteration 3, merged
GT FSC-AUC delta `-0.003274589 < -0.002`. Independent verification passes for
every ledger and seal manifest entry.

Ledger JSON/Markdown/manifest SHA-256 values are
`5393ee8f1549ccce6dbf7befec7c14f66d58d16b6196ccb52eef8a70e8ddf26f`,
`76036d28f410b55ce2a9bd5a30f524cc921d327339e9fb319af0fc80a0a74d4f`,
and `c0827618c6550e5b15eae94291985173ce9b14c97dc2beb55ea7ba801d5675ee`.
Seal JSON/Markdown/manifest SHA-256 values are
`897a21e317eb5fd77aeaf715736332c8ed0f76dcf2e3199ca8e407e425b73a51`,
`7cc531b9fd92ca7a93590bec3cc097fd5f5652cb722ae208cdb777e4261ff3d5`,
and `3b582c96b65a23888909c219ab2a5fe419da726d8d00acaf34c8bde057e8df10`.
Updated graph-repair, resource, and registry note SHA-256 values are
`de96b2aa7437b1f996cfafe770ff19ef0ebe47c032277841f89ea6c012288984`,
`29f6c623cb9222f3c4075fb289f7ed0d6f2e492ad04d31eb152346865b66df4d`,
and `84f711fec024ce5f77008233a9699e0e96bc63fb46ab36214437f1c04216d813`.
No production behavior or threshold changed; correlation was not computed.

## 2026-07-21 case-7 capture-target observer effect and clean-rerun gate

The preceding case-7 component/residual conclusions are superseded as clean
parity evidence. Exact-local pass 1 is implemented as `score_only=True`, but
its retained parent support is science-critical input to fine pass 2. The
debug capture path applied target-only bucket filtering to every score-only
call by default. Merely changing the requested posterior identities therefore
changed which parent buckets ran and changed refinement output.

The causal comparison held source commit, inputs, seed, schedule, driver, and
H100 class fixed. The 24-target run retained 12 parent buckets in each half;
the disjoint 48-target run retained 24 and 23. Their stripped CUDA libraries
are byte-identical. Nonetheless 534/100000 Euler rows changed, the rotation
tail changed from 5793 to 5620, and the translation tail from 6230 to 6030.
This is capture-target dependence, not H100 arithmetic.

Consequently, the prior full-population resident/exact split and the derived
48-particle `persistent`/`opened` cohort are invalid. The component and
residual science jobs `11449766` and `11451167` were canceled after 03:00:59
and 02:24:49; dependent audits `11452889` and `11452890` were canceled before
running. Their partial artifacts are quarantined and may not be used for
FSC/FSC-AUC or posterior parity claims. Canonical K=1 matrix runs without the
capture environment variables are unaffected.

The fix makes target-only score execution explicit opt-in through
`RECOVAR_LOCAL_SCORE_DUMP_TARGET_ONLY=1`. With the variable unset, the full
science parent pass runs while dump artifacts are still written only for
requested target buckets. The focused regression compares hard assignments,
maximum posterior, rotation posterior sums, and bucket execution with versus
without configured dump targets. The focused debug/capture group passes 7
tests (329 deselected); no map correlation was computed.

The authoritative note is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_it11_stratified_posterior_77bcf3bd_20260720T214500Z/provenance/CAPTURE_TARGET_PARENT_PASS_OBSERVER_EFFECT_20260721T0654-0400.md`
(SHA-256
`312594539d2b6932c86aadb727afe885417282c509e5be63ab2652b689435671`).
Clean component reruns must use unfiltered parent passes. A residual panel may
be selected only from a clean unfiltered full-population run.

## 2026-07-21 clean case-7 component and population results

The clean unfiltered reruns supersede the quarantined case-7 captures above.
Science/audit jobs `11455669`/`11455726` completed the 24-particle component
panel at `c1ee409b`. The independent exact-state/reference repeat remains
identical to its prior cross-allocation control: candidate, physical-pose, and
reconstruction-support Jaccard are all one, posterior TV is zero, all 24
winners match, and the best-pose displacement is zero.

Restoring only RECOVAR target-boundary poses is essentially inert: median
posterior TV is zero, its maximum physical-pose TV is
`4.60258837567923e-6`, and no winner or reconstruction support changes.
Restoring only RECOVAR target-boundary maps produces median physical-pose TV
`0.0023821552614138467` and maximum `0.008934585995142925`, but still changes
no winner or reconstruction support. All immutable RELION targets remain
present, in reconstruction support, and rank one. Thus target-boundary maps,
not poses, contain a measurable local seed, but neither component alone
recreates a discrete selected-particle failure.

Clean full-population science job `11455783` independently ran resident and
exact-RELION-state/reference trajectories with all capture variables unset.
Exact replay improves absolute iteration-11 Pmax agreement for
`98099/100000` particles. Median absolute Pmax error drops from
`0.005740753383636499` to `0.0001427888574600522`. At the 0.1-degree rotation
threshold, the resident tail has 1093 particles and the exact tail has 22:
1083 close, 12 open, 10 persist, and 98895 remain stable. These are clean
diagnostic population counts, not a map-quality gate.

The predeclared six-persistent-targets-per-half residual panel is impossible:
the clean half-specific persistent/opened counts are 4/2 and 6/10. A balanced
fallback must therefore use four persistent and two opened targets per half,
each with a stable control: 24 unique particles. Selector retries `11476363`,
`11476493`, and their never-started residual dependents are rejected
orchestration attempts, not science. Corrected selector `11476990` completed
`0:0` and sealed the 12-target/12-control panel. Residual science/audit
`11477130`/`11477132` use that 24-particle contract and remain source-pinned to
`fa0c93fc`. Correlation was not computed in any of these diagnostics.

Complete component JSON/Markdown are under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_it11_state_component_clean_c1ee409b_20260721T070100Z/provenance/`.
The clean population split and selector retry audit trail are under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_it11_cohort_base_clean_c1ee409b_20260721T071000Z/provenance/`.

## 2026-07-21 K=1 residual-panel live trajectory checkpoint

Science job `11477130` runs the clean 24-particle residual-panel resident arm
and exact-RELION-state/reference arm sequentially on one H100. The resident
capture is complete with 24 unique, hash-verified iteration-11 posterior
files. The exact arm remains active, so these are interim trajectory
invariants rather than the residual-panel conclusion; dependent audit
`11477132` remains gated on science completion.

The two arms have identical printed iteration-1 science state: current size
56, HEALPix order 3, resolution 30.22 A, average Pmax 1.0000, and unchanged
tau/noise/correction summaries. At iteration 2, after the exact arm replaces
the incoming references and state with RELION iteration 1, a small support
split appears independently in both halves. Half-1 support work is
120380 resident versus 120382 exact; half 2 is 120780 versus 120781. In each
half, the exact arm has one additional net image in the size-16 bucket and one
fewer in size 32. Occupancy in the 64/128/256 tails is unchanged.

The split is visible in the completed iteration-2 state without changing its
schedule. Average Pmax is 0.676258 resident versus 0.676240 exact. Printed
normalization/correction extrema differ at approximately `1e-6`--`1e-5`,
while tau2 remains equal at printed precision. Both arms select resolution
27.20 A, next current size 104, and HEALPix order 3. Capture activation remains
false before iteration 11, and target-only parent filtering is unset. This
supports an early incoming-reference-dependent posterior/support divergence,
but does not identify a production repair before the iteration-11
target/control audit lands. Correlation is not computed.

Iteration 3 preserves the same schedule and printed average Pmax `0.8219`, but
the sparse-support split remains measurable. In half 1, support work is 87241
resident versus 87249 exact; one net image moves from the size-16 bucket to
size 32, while the size-64/128 tails remain 9/1. In half 2, support work is
87758 versus 87759; two net images move from size 16 to size 32, while the
size-64 tail remains 11. Both arms report resolution 32.00 A and unchanged
HEALPix order 3. The direction and magnitude therefore vary by half and
iteration rather than forming a monotone support expansion.

## 2026-07-21 K=1 clean residual panel reaches terminal classification

Science job `11477130` completed `0:0` in 03:05:12 and dependent fail-closed
audit `11477132` completed `0:0` in 16 seconds. Both arms produced 24 unique,
hash-verified iteration-11 fused-posterior captures. The audit has schema
`case7-it11-residual-fused-posterior-v2`, status `complete`, 48 inputs, cohort
counts 8/8/4/4 for persistent target/control and opened target/control,
respectively, and explicitly records `diagnostic_non_gating=true` and
`correlation_computed=false`.

The global-search sparse pass-2 work split is small, changes sign, and is
independent in the two halves:

| physical iteration | resident half 1 | exact half 1 | delta | resident half 2 | exact half 2 | delta |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 120380 | 120382 | +2 | 120780 | 120781 | +1 |
| 3 | 87241 | 87249 | +8 | 87758 | 87759 | +1 |
| 4 | 89247 | 89219 | -28 | 89913 | 89921 | +8 |
| 5 | 94623 | 94609 | -14 | 94887 | 94879 | -8 |
| 6 | 95642 | 95586 | -56 | 95976 | 95876 | -100 |

At iteration 6 the difference is confined to the size-16/32 bucket boundary:
six half-1 images and seven half-2 images move from size 32 to size 16 in the
exact arm, while the larger tails are unchanged. Both arms then take the same
HEALPix-4 local-search schedule through iteration 11. Final average Pmax is
`0.7226` resident versus `0.7239` exact.

The persistent controls are stable: median candidate and reconstruction
support Jaccard are both one, median posterior TV is `0.020920583`, and 7/8
latent winners agree. All winner rotation deltas are zero; the one differing
control winner changes translation/candidate identity only. Persistent targets
also have median candidate-support Jaccard one and all eight arm-to-arm physical
winners agree. The immutable RELION target remains present and in reconstruction
support in 8/8; exact state raises its median posterior mass from
`0.36655533` to `0.39146566` and its median target/winner ratio from
`0.93263455` to `0.99932882`, but seven targets remain rank-2 near ties. This
localizes the persistent tail to relative score ordering inside available
support, not missing candidate geometry.

Opened controls remain stable with support Jaccard one, all four winners
agreeing, and median posterior TV `0.007788642`. In contrast, all four opened
targets change physical winner by median `1.8614835` degrees. Two retain exact
candidate support Jaccard one and become rank-2 near ties with target/winner
ratios `0.99932888` and `0.99972535`; two have candidate-support Jaccard `0.5`,
including one absent target and one target demoted to rank 3 with ratio
`0.03210279`. The opened cohort therefore mixes support and relative-score
effects. It does not justify a single support-only or score-only production
change.

The sealed report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case7_it11_residual_clean_fa0c93fc_20260721T082000Z/provenance/residual_panel_audit.json`
(SHA-256
`b3fc8255366b66e17ae9149c456d3fa82ce9c2320fbe08dc120aebfe7cc498f1`);
the Markdown report SHA-256 is
`2fbed88f8fe8abf9d45c638bfd0cdc0ba53bb2d698ce25fe78eb9b8f69d654e1`.
This bounded posterior diagnostic does not replace the FSC/FSC-AUC map-quality
gate and provides no evidence-backed production edit by itself.

## 2026-07-21 current-head case-2 strict-boundary closure

Strict K=1 v3 identified historical case 2 iteration 3 as the earliest ledger
failure, with merged GT FSC-AUC delta `-0.003274589`. That row used older code
whose first structural mismatch was current size 162 versus RELION 164. The
later high-shell route should produce 164, so job `11456044` reran RELION and
current-head RECOVAR sequentially for four numbered iterations on one H100 at
commit `c1ee409b`.

The numbered topology audit passes exactly through all four boundaries.
Audit retry `11476398` uses numbered-only views because final all-data was
explicitly skipped; it completed `0:0` without changing the GPU outputs.
Merged cross-engine FSC-AUC is `0.999999999797`, `0.999999889449`,
`0.999995387764`, and `0.999987472525` at iterations 1--4. The corresponding
RECOVAR-minus-RELION merged GT FSC-AUC deltas are `+1.87897086223e-8`,
`+3.53042218848e-7`, `-4.70789998291e-6`, and `+1.74529697717e-5`.

Current head therefore closes the historical case-2 failure by FSC/FSC-AUC;
the old strict-v3 ledger remains an immutable historical record rather than
an active defect. The audit report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case2_currenthead_it4_samegpu_c1ee409b_20260721T072000Z/analysis_numbered_retry/k1_fsc_trajectory.json`.
Correlation was not computed.

## 2026-07-21 significance-capture boundary gate

Configured significance dumping previously activated diagnostic score-block
collection at every global boundary, even when its requested iteration and
current size could not match. In the joint first-iteration path, merely
setting the dump directory could also run a full significance pass that
science otherwise skipped. This is observationally intended work with a real
performance cost: at clean case-7 iteration 4, the capture-configured arm took
998.7 seconds versus 800.8 seconds without capture. Two pass-1 intervals
account for 187.4 of the 197.9 extra seconds.

Commit `fa0c93fc` moves the iteration/current-size predicate to the boundary
before diagnostic collection and before the optional first-iteration probe.
The affected complete unit-module set passes 417 tests with 109 warnings.
Same-H100 causal science `11476473` compares configured `c1ee409b` and gated
`fa0c93fc` on the 3k/128 case with an impossible future dump target. Total
E-step time drops from 117.719207 to 102.327864 seconds, a 15.391343-second
(13.1%) reduction; external wall time drops from 144 to 127 seconds.

The strict cross-commit exactness audit ultimately ran as `11478462` after
three audit-infrastructure repairs and intentionally exited `2:0`: the outputs
are not bitwise equal. Their merged-map FSC-AUC is nevertheless
`0.9999999999466644`. Because removal of old diagnostic work can itself change
the GPU execution trajectory, same-source science `11478436` compared clean
`fa0c93fc` against the same impossible future target. That exactness audit
`11478437` also exited `2:0`, with merged-map FSC-AUC
`0.9999999999457423`, per-image Pmax relative L2 `6.5196428e-6`, and merged-map
relative L2 `1.4146703e-7`.

Clean-versus-clean calibration `11478823`/`11478841` establishes that this is
the normal repeated-GPU envelope, not a remaining observer effect. It produces
merged-map FSC-AUC `0.9999999999464423`, Pmax relative L2 `6.5306604e-6`, and
merged-map relative L2 `1.4335637e-7`. The same-source clean/configured E-step
totals differ by only 0.008706 seconds, and the configured arm writes no
significance artifacts. The boundary gate is therefore accepted as
observational for a nonmatching target, while the cross-commit timing reduction
is real. The durable classification is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case11_siggate_cleanrepeat_fa0c93fc_20260721T174000Z/provenance/CAUSAL_CLASSIFICATION.md`.
Correlation is not computed.

The frozen-panel graph is now terminally closed. Science `11480333` ended
`FAILED 1:0` after 5h45m38s, only after the clean RELION process returned zero
and sealed all 48 optimiser/data/model STAR files for iterations 0--15.
Verifier `11485567` completed `0:0` and confirms the exact 72/96 gate,
expected exception, zero capture files, and absent science/capture-audit
completion markers. Its JSON SHA-256 is
`0ccb9180f5c37ebd78ac92d6c267c0db0d5a1e3e0ef7987bb189fd9f0fc926e7`.

Oversized pending salvage `11481766` was canceled at zero runtime and
right-sized with the same pinned launcher/comparator as `11487432`; the
replacement failed `1:0` in 3 seconds at the known-absent inertness input,
before any comparator or partial output. Salvage-rejection JSON/helper
SHA-256 values are
`b9db836e284423846918676259a7fa7e89e7f4310f887f07f8220cafe428e2b9`
and
`3f6cc93165f7b55df85547af8816ce2d04cc8a13641d87e7b5e175b9bce3b8a5`.
Success audit `11480664` was canceled at zero runtime after becoming
`DependencyNeverSatisfied`. This closes recovery orchestration, not K=4
parity, and authorizes no production edit.

## 2026-07-21 uninterrupted K=4 class-2 pre-scatter diagnostic

The accepted same-A100 K=4 trajectory at `ac5177d2` first fails the direct
cross-engine map gate at iteration 10, class 2: FSC-AUC is
`0.994676799 < 0.995`. Fixed-stat reconstruction closes when RECOVAR's own
`Ft_y`/`Ft_ctf` are replayed, while swapping RELION tau into the RELION-factor
replay leaves essentially the full gap. The remaining boundary is therefore
upstream in the accumulated source numerator/weight rather than tau or the
reconstructor.

An iteration-9 RELION continuation is not an admissible substitute for the
live iteration-10 boundary. Even after exact particle-order and follower
replay, the best continuation's class-2 map versus the uninterrupted canonical
map has FSC-AUC `0.9707791`. Those continuation artifacts are rejected as
science and preserved only as orchestration history under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_bpref_capture_ac5177d2_20260720T084530Z/`.

The replacement diagnostic runs an uninterrupted clean RELION control, an
uninterrupted passive-capture RELION arm with exact control dispatch replay,
and current-head RECOVAR sequentially on one physical A100. The RELION
diagnostic source is commit `186e0b8`; its binary SHA-256 is
`54d7abea3d3ab472b89c6a5632805b83994e5ab01e8fa5353a5adcf38f086a10`.
RECOVAR is detached and clean at
`77a348c59c28fb0848413af4ec3333192cad5add`. Grid correction and forced final
all-data after non-convergence are unset.

The frozen panel contains 96 unique class-2 particles, 16 each from stable
controls, Pmax tails, rotation tails, translation tails, support mismatches,
and RELION-class-2/RECOVAR-winner disagreements. It is balanced 48/48 across
the canonical follower owners. The payload's `identity_sha256`
`b1c85f635cc342aded1cbe95ffac9d99e0b9ed5afb432dd85f40f9b0e0d085be`
is the full 100,000-row source-identity-order digest, not the selected-panel
digest. The selected contract is independently pinned by `panel96.json` SHA
`03fce7f4943e9bbddb82ecee03574ed9a3d103f0b372afceb546fc9ae09d9cb9`,
one-based selected-index NPY SHA
`2e163ec980dd948f81bf960919a760f941dd359fa26d9a13a891f935c2bc9b60`,
and selection-manifest SHA
`1f8a1341fdf9f087c5e52a2a7231ea8baddd09e4d7a485c984a2dfcf26aa1c50`.
RELION captures native class-2 source rows at iteration 10. RECOVAR captures
the corresponding soft-posterior contribution rows and observational device
geometry at iteration 10, half 1. Target-scoped diagnostic modes are checked
shadows; production score, reduction, and adjoint outputs remain authoritative.

Initial science job `11480222` was cancelled after 1m45s, before iteration 1,
because the launcher omitted the RECOVAR device-geometry shadow required for
the intended comparison. Its partial directories are preserved and excluded.
Superseding uninterrupted science job `11480333` runs from
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_class2_prescatter_panel_ac5177d2_20260721T215100Z/`.
The first dependent audit `11480576` was cancelled while pending at zero wall
time after local preflight caught an incorrect panel-status expectation in the
run-scoped comparator. Corrected audit `11480664` waits on
`afterok:11480333`; the corrected comparator SHA-256 is
`35843376457d496c0dac2568372d53ccbad6b74c438bc2e40a8f6c5bf7633542`.
It passes `py_compile`, Ruff, the actual 96-particle panel contract, and a
synthetic two-rotation soft-class exact-operand test.

The audit predeclares the classification order: rotation support first,
reached-scatter pixel support second, then complex numerator and real weight.
RELION values are converted by `-1/N^2` for data and `1/N^4` for weight with
physical `N=256`; current-size is layout only. Capture inertness is gated with
shellwise FSC/FSC-AUC, intermediate operands use exact/relative-L2 metrics,
and correlation is not computed. No K=4 production change is justified until
science `11480333` and audit `11480664` complete.

CPU diagnostic job `11481280` completed `0:0` and compares the clean control's
iteration-3 maps with the previously accepted uninterrupted canonical maps.
Class 1--4 FSC-AUC values are `0.991321933`, `0.989710691`, `0.987441885`, and
`0.970276020`; relative L2 values are `0.00889732`, `0.00892967`, `0.00853975`,
and `0.0107270`. This is explicitly descriptive and non-gating. The control
uses the capture-instrumented RELION binary, whereas the canonical run used
the older dispatch-instrumented binary, so the result bounds binary-build/GPU
trajectory sensitivity rather than RECOVAR parity. Current size, resolution,
and HEALPix schedule still agree through iteration 4, but average Pmax is
`0.265957` canonical versus `0.265868` control at iteration 3 and `0.598995`
versus `0.597916` at iteration 4. The formal observer gate remains clean
control versus passive capture with the same new binary and exact control
dispatch replay. The diagnostic JSON is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_class2_prescatter_panel_ac5177d2_20260721T215100Z/analysis/CONTROL_IT003_VS_CANONICAL.json`;
correlation is not computed.

The capture-instrumented control is also materially slower: numbered
iterations 1--4 take approximately 20 minutes each, versus 5--7 minutes in
accepted canonical job `11386135`. Instantaneous allocation telemetry during
iteration 5 showed low GPU utilization and both followers in I/O wait, so the
slowdown is not assigned solely to capture arithmetic. Two control-rate
RELION arms plus the prior 32,876-second RECOVAR runtime project beyond the
submitted 16-hour limit. Slurm denied an in-place extension of `11480333` to
24 hours. The job remains uninterrupted; if it cannot seal its declared
completion artifacts, dependent audit `11480664` must remain fail-closed.
Run-scoped recovery policy and telemetry are in
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_class2_prescatter_panel_ac5177d2_20260721T215100Z/provenance/TIME_LIMIT_RISK.md`.

Failure-only salvage audit `11481766` waits on `afternotok:11480333` and
consumes no resources while the science job is healthy. If science ends in a
terminal failure state after sealing the RECOVAR target boundary, it verifies
the same frozen panel, source and comparator hashes, capture/signature
completeness, and RELION-arm inertness, then writes a separately named partial
diagnostic with status
`target_boundary_complete_terminal_trajectory_incomplete`. It cannot create
or satisfy the full `SCIENCE_COMPLETE`/audit contract, and fails closed if the
target boundary itself is incomplete. Full audit `11480664` remains the only
admissible terminal comparison after successful science completion.

### Live iteration-10 panel gate rejects the frozen cohort

The capture-capable clean control reached numbered iteration 10 while science
job `11480333` remained uninterrupted. The predeclared read-only class gate
finds only 72/96 frozen targets in class 2; the complete class counts are
7/72/14/3. Twenty-four targets are off class 2, distributed across the six
16-particle categories as 4 Pmax-tail, 3 rotation-tail, 2 support-mismatch,
3 translation-tail, 11 RELION/RECOVAR-disagreement, and 1 stable-control
particle. The sealed gate JSON is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_class2_prescatter_panel_ac5177d2_20260721T215100Z/analysis/LIVE_CONTROL_IT010_PANEL_CLASS_GATE.json`
with SHA-256
`69f8358e26255d93131d92dce6d51220d8a2a2a2662c05b6c88cc17005296b5a`;
the input STAR SHA-256 is
`a044f3a98457954730a47f2fcabad3a45b87d8336d5fedf10fee501a39ab13d5`.
The sidecar hash and exact schema/count/hash assertions pass.

This invalidates the frozen panel for the current binary/trajectory. The
existing launcher must finish its clean control and then fail closed before
passive capture; the running job is not mutated or canceled. Consequently,
success audit `11480664` cannot become authoritative, and failure audit
`11481766` is expected to reject salvage because no declared RELION/RECOVAR
target captures exist. That is a recovery-path check, not K=4 science.

The live and older canonical iteration-10 maps use the same current size 74
and resolution 19.428571 A, with average Pmax 0.915123 versus 0.915362.
Classwise map FSC-AUC is 0.954572533, 0.949992474, 0.941647625, and
0.944810030; relative L2 is 0.0242146172, 0.0238771880, 0.0257947039, and
0.0214597318. The descriptive JSON is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_class2_prescatter_panel_ac5177d2_20260721T215100Z/analysis/CONTROL_IT010_VS_CANONICAL.json`
with SHA-256
`01bd7b83bf99fe0f4e34751912199421ec9e679142a7fea8b936d257dfa7cc6f`.
This measures RELION binary/trajectory sensitivity, not RECOVAR parity;
correlation is not computed.

The replacement must select its exact comparison panel deterministically from
the completed clean control on the same allocation, after verifying source
identity and follower ownership, and only then run passive capture and
RECOVAR. Its selection rule and candidate reservoir must be sealed before
submission, while the resulting live panel and hashes are sealed before either
comparison arm. A longer time limit is also required. No K=4 production edit
is justified by this panel failure.

The two-stage replacement selector is now sealed. Its static reservoir has
23,394 canonical/robust class-2 agreement candidates and 130
RELION-class-2/RECOVAR-disagreement candidates; selector and reservoir-manifest
SHA-256 values are
`c7f9854abf93314f5ecf0044a5d50888ea9d70567abf281611e4b6d92dfe76a1`
and
`ad96d6e0112f8b0f7aa000e720ab5a38345d06939daa334d5a2530525c6ba648`.
Two resolutions against the current control are deterministic and satisfy all
six 16-particle categories, 48/48 live follower ownership, and 96/96 live
class-2 membership. This is a selector validation, not the new science panel;
the exact submitted panel remains deferred to the new control.

Initial replacement science/audit jobs `11484384`/`11484385` were canceled
after 1m40s/0s. The selector had not run, but reproducing its eventual launch
directory showed that the pixi Python would resolve `recovar` to an unrelated
editable checkout rather than pinned source `77a348c5`. The partial control
root is quarantined. Superseding 24-hour science job `11484481` and afterok
audit `11484482` use fresh run/runtime roots at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_dynamic_class2_prescatter_ac5177d2_20260722T015934Z/`
and
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k4_it10_dynamic_class2_prescatter_ac5177d2_20260722T015934Z/`.
The new launcher asserts `recovar.__file__` under the pinned source and
`jax.__file__` under the pixi environment before control iteration 1, then
returns to that source before live selection. Its SHA-256 is
`4896aac254895a9bb5828fe4badbf42b7060b0d8cbd21397cd1309d172dbbd04`;
the audit launcher SHA-256 is
`7e33e576c477184fc0cc9a2f051c36ad8eece29bce76134483cb68ae1992cd12`.
Science `11484481` started on A100 node `della-l08g5`; the pre-control import
gate resolved the exact pinned RECOVAR and pixi JAX paths before RELION entered
iteration 1. Audit `11484482` remains dependency-gated.

A subordinate scalar-source audit is also sealed before any live panel or
capture. Primary comparator `11484482` remains authoritative for rotation and
pixel-support qualification; scalar audit `11484846` waits on
`afterok:11484482`. It fits one positive real RELION/RECOVAR scale separately
to complex data and real weight for every aligned rotation. Aggregate source
relative-L2 at most `1e-6` is source-close. Otherwise, posterior-mass
compatibility requires data and weight fit residuals, complex phase fraction,
and data/weight scale disagreement each at most `1e-5`, with at least 95%
valid and 95% compatible fits. This label is compatibility evidence, not proof
of posterior normalization. Comparator/launcher SHA-256 values are
`9a6a3b650ead568c07833a00f6c102784be2cd6b687616d84e4cf56a0b9c969f`
and
`48a66034daf483096ed9012bc0ce1172ad61263636f72e81e2659ae510bb22ba`.
Synthetic exact-scalar, pixel-varying, complex-phase, and zero-reference tests
pass. Existing focused float32-posterior/joint-pruning tests pass 7 tests with
152 deselected using `JAX_PLATFORMS=cpu`.

Identity-aligned all-particle comparisons with control `11480333` use a v2
helper that hashes only the immutable 100,000-row dispatch slice for each
iteration. At iteration 2, all 100,000 class labels and Euler tuples agree
although 49,780 particles changed MPI follower ownership. Pmax differences
have median zero, p95 `5e-6`, p99 `9e-6`, and maximum `1.21e-4`; only two
translations differ, each by one 2.125 A pixel. By iteration 3, 5,352 class
labels, 12,383 Euler tuples, and 11,655 translations differ; classwise map
FSC-AUC is 0.991678, 0.991111, 0.987813, and 0.971361. This same-binary
cross-A100 bifurcation reinforces the need for exact same-allocation panel
selection and passive capture. It remains descriptive/non-gating. The
iteration-2/iteration-3 state JSON SHA-256 values are
`9343094d372a994e7c950f6f162d099a761413d6cb476e36a40849f3334bd0e1`
and
`00bd3a367356f5e40ad089468ca73c417d5df8537c019889d40200fa02025216`;
the iteration-3 map JSON and v2 helper SHA-256 values are
`5a139b5e7d42a12084bb35f7bee717b3401f32bba407083328d17f1aec9701d6`
and
`bf0d2708498c3c4287093bc8727cd623bbbe47b728657b49dceecb6f70738f90`.
Correlation is not computed.

### Corrected K=4 dynamic panel is authoritative

Corrected same-allocation science `11484481` completed its clean RELION
control with status zero in 16,597 seconds on
`GPU-3bae32ea-7500-d97f-68d3-b73eaf826482`. All iteration-0--15
optimiser/data/model STAR boundaries and exactly 1,500,000 dispatch rows are
sealed. The submitted selector then produced the authoritative 96-particle
live class-2 panel: 16 targets in each of six predeclared categories, 48/48 by
normalized follower owner, and 96 unique identities. Live class counts are
25,166/23,728/24,845/26,261; the live agreement/disagreement reservoirs contain
22,215 and 74 candidates.

The official panel JSON/manifest SHA-256 values are
`7cf6ed42934460c9540b4f6a66238921e99b3b665117a5a88a66930836ab68f7`
and
`bb86ac1c3f61cb1d14e9314f9bdfb60e6a4abd09becdab55149cfaf656e66262`;
panel identity SHA-256 is
`b1c85f635cc342aded1cbe95ffac9d99e0b9ed5afb432dd85f40f9b0e0d085be`.
Original/stack selected-index arrays retain SHA-256 values
`48058423d876305cf72c23260e514a5ca982508c2791b43234e51f7a8671b489`
and
`d3900f382b275f529ec4365232e105a819138f2add51d1eb2a77b5a457e11105`
and are byte-identical to the immutable iteration-10 preview. The manifest
verified before passive capture; the first capture artifact followed 27
seconds later. Science remains active, audits `11484482`/`11484846` remain
dependency-gated, and no production edit is authorized yet.

### Passive K=4 capture is within the early repeat envelope

Passive capture iteration 1 matches the clean control exactly for all 100,000
class/Euler/translation/Pmax rows and dispatch owners; classwise map FSC-AUC is
at least `0.999999999278`. At iteration 2, dispatch ownership and all class
labels remain exact. Pmax absolute-difference p95 is `4e-6`; one Euler tuple
and two translations choose alternate low-posterior grid winners. One of the
translation identities is the same near tie that moved between the two clean
controls. Classwise map FSC-AUC remains
`0.999999984390/0.999999983143/0.999999981554/0.999999973606`, above the
predeclared `0.999999` formal threshold.

Iteration-1 state/map JSON SHA-256 values are
`e37e3c81dc1b3ec9c08500e87a20da9fc6b5ff0dd52a4e7dc6951a91a2660626`
and
`127b659cc72069c26938041cc7beeff6279422859423a7db2993f0e4096462fe`;
iteration-2 values are
`adf3866ba64744090d28216bbe883d840560fa3f39328a7a2bf815a1cef4ea5d`
and
`a7ef6939bb0bb73e6e00176a7ebace85cce06514f3be9914c2906671fa2e85b0`.
An operator-side exact-particle assertion exited nonzero after sealing the
reports; the corrected descriptive classification is FSC-pass with
repeat-scale particle near ties. Science was unaffected. Formal inertness
remains the iteration-10 dispatch/FSC/capture validator; correlation is not
computed.

### Passive K=4 capture has an iteration-3 inertness warning

At iteration 3, passive-capture dispatch ownership, all class labels, and
schedule scalars still match the clean control exactly; one Euler tuple and
four translations differ. Classwise capture/control FSC-AUC is
`0.999998957108/0.999999983889/0.999999972165/0.999999951124`. Class 1 is
`4.29e-8` below the same `0.999999` numeric threshold used by the formal
validator, while classes 2--4 remain above it. State/map JSON SHA-256 values
are
`b865b1cd370a3b042b3e407c239083d12c61d99b91ec828f9329d16c2f564f81`
and
`3a7c02a08e3fba57f1ece0a11106fb899cdd981bebec3df236c5e4e55baebdfc`.

This does not move or weaken the predeclared gate: only iteration 10 contains
the target capture and is formally authoritative. Science continues
unchanged; iteration-10 inertness must still fail closed before RECOVAR if the
warning amplifies. No production edit is authorized, and correlation is not
computed.

### Passive K=4 capture warning amplifies through iteration 5

At iteration 4, dispatch ownership remains exact, but the passive arm differs
from its same-allocation clean control in 1 class label, 9 Euler tuples, and
10 translations. Classwise capture/control FSC-AUC is
`0.999987631111/0.999999768543/0.999992547236/0.999999858478`; classes 1 and
3 are below the formal `0.999999` threshold. State/map JSON SHA-256 values are
`6453998fc03bb3ded0627e1fa8aab7e301d64c7d1250342b3158cc2cc8879e40`
and
`06a000dc22ba18c8e007ad95bcd7229da0a4ffa5dc9df6d8fe75ebd7b78deb95`.

At iteration 5, the immutable 100,000-row dispatch slice remains exact at
SHA-256
`8a5336e4ab89461ad4b5a9b9261d54c74dfdecaf342a5a5bc38c7ae736b44e96`,
while the state difference grows to 8 class labels, 32 Euler tuples, and 43
translations. Pmax absolute-difference p95 is `0.001055` and maximum is
`0.522492`. Classwise map FSC-AUC is
`0.999933587444/0.999934151441/0.999866285411/0.999992450129`, so all four
classes are below threshold. State/map JSON SHA-256 values are
`45afd7b63fc3ce2d6d5a5b92614dc2fb96fe21ef0cd4745193e02908dc7f3036`
and
`3bd501e9ccd8e373b0e0b172d640c63b15a2a0eaad742a9b36c7b6339912d374`.

These diagnostics make rejection at iteration 10 likely but do not relocate
the declared gate. A fail-only contingency is preflighted, not submitted: on
one A100 allocation, restart an unset control and capture-enabled arm from the
clean `run_it009_optimiser.star`, replay the clean iteration-10 dispatch and
particle order in both arms, and capture the same sealed class-2 panel. The
RECOVAR refinement can continue to consume the original clean full run for
iteration-0 state, optimiser metadata, and the 15-iteration dispatch oracle;
the restart capture bundle is an independent audit operand and is not a
RECOVAR initialization input. The current science job remains untouched, and
no production edit is authorized before an accepted capture and its dependent
geometry/scalar audits.

### Full-start iteration-10 capture is formally rejected; restart gate is active

The authoritative passive full-start iteration-10 boundary sealed at
2026-07-22 06:20:46 EDT. Its capture and control dispatch slices are exact:
100,000 rows, follower ownership 50,140/49,860, row SHA-256
`00059e382a1a4888275fe43d801e463529f0ae07bd046cc748f06400e855fb76`,
source-order SHA-256
`759d64f245c4c8ffcce4c527e990c115391d6607271d4e7f75da5550c9324534`,
and sampling perturbation `0.096421` in both arms. Despite those controls,
classwise capture/control FSC-AUC is
`0.998247194648/0.997600525037/0.997363409604/0.998221443830`, so every
class is decisively below the predeclared `0.999999` threshold. Relative L2
is `0.0046840/0.00508357/0.00520915/0.00375230`. The native artifact check
also reports `ValueError: missing MPI rank identity`. The sealed rejection
JSON SHA-256 is
`3d5134197e3071ce5074d75bc45f5fdfcada794eacbe4bfc227e5552e14ae789`;
correlation is not computed.

Only after this rejection, the no-submit guard was removed from the prepared
restart contingency. Science job `11492718` began on one A100 at 2026-07-22
06:24:07 EDT; primary and scalar audits are dependency-held as `11492719` and
`11492720`. The unset-control and capture-enabled RELION arms both restart
from the immutable clean iteration-9 optimiser and replay the exact clean
iteration-10 dispatch and particle order. RECOVAR is allowed to run on the
same allocation only after capture inertness and closure to the original clean
iteration-10 trajectory both pass. The original full-start job `11484481`
continues untouched toward its independent end-of-run fail-closed validator.
No production edit is authorized before the restart capture and dependent
geometry/scalar audits qualify a mismatch.

### Restart continuation is bounded by absolute `--iter 10`

Rank-corrected science `11492933` sealed a complete unset-control iteration-10
boundary, then entered expectation iteration 11. RELION continuation retains
the optimiser's stored `nr_iter=15`; `--auto_iter_max 10` is parsed separately
but does not override that loop bound. The job was canceled after 21m25s,
before capture or RECOVAR; audits `11492934`/`11492935` never ran.

Source inspection confirms that continuation option `--iter` directly
overrides `nr_iter`. Fresh science `11493435` therefore starts both arms from
the immutable clean iteration-9 optimiser with explicit
`--iter 10 --auto_iter_max 10`, and fails closed on any iteration-11 log line
or optimiser output. It began on A100 node `della-l07g2` at 2026-07-22
07:04:48 EDT after rank probe `0,1,2` and panel/import gates passed. Primary
and scalar audits are `11493436` and `11493437`. No capture or RECOVAR output
from the canceled root is reused, and original full-start science `11484481`
remains untouched.

### Capture completeness cap matches sealed 48/48 ownership

Absolute-bound science `11493435` completed its unset control in 787 seconds
and stopped exactly at iteration 10. Its capture then failed after 23 valid-rank
artifacts because `MAX_PARTICLES_PER_RANK=96` doubled the diagnostic
worst-case file estimate relative to the panel's sealed 48/48 follower
ownership. No temporary artifacts, OOM, RECOVAR run, or dependent audit
occurred; jobs `11493436`/`11493437` never ran.

The fresh launcher tightens the per-follower completeness cap to the exact
sealed value 48 while retaining 96 expected particles, two followers, and the
unchanged 64 GiB maximum. At image size 2,812 and the full 4,608-orientation
fine grid, the resulting all-particle worst case is 49,785,899,520 bytes
(46.37 GiB), below the cap. Fresh science `11494295` began on A100 node
`della-l07g2` at 2026-07-22 07:29:29 EDT after rank, panel, import, and 48/48
owner gates passed; audits are `11494296`/`11494297`. All iteration,
dispatch/order, inertness, closure, and same-A100 RECOVAR controls remain
unchanged.

### Restart capture rank provenance is corrected

Read-only inspection of the formally rejected full-start capture found that
every artifact encoded MPI rank as unsigned `-1`. The diagnostic RELION helper
read only `OMPI_COMM_WORLD_RANK`, while the direct Slurm `srun` launch exposes
`SLURM_PROCID`. Therefore the first restart graph
`11492718`/`11492719`/`11492720` would deterministically repeat the missing-rank
validation failure. It was canceled after 6m39s/0s/0s, before its capture or
RECOVAR phases; original full-start science `11484481` was not touched.

Diagnostic-only RELION commit
`4ab53edf206e9cafd993484a92eccd77e828c497` adds a strict Slurm-rank fallback
and fails closed on missing or invalid identity. It changes no reconstruction,
selection, or arithmetic code. The isolated replacement binary has SHA-256
`dad0ff14a1478b22b1f3ba9acc93934341aaf7b8750205a606256f8c990ce475`.
Replacement science `11492933` started on one A100 at 2026-07-22 06:41:03 EDT
after an independent three-task rank probe passed exactly `0,1,2`; primary and
scalar audits are `11492934` and `11492935`. A three-second preflight-only
attempt `11492919` had first rejected incorrectly pre-created empty child
directories; it produced no science, and dependencies `11492920`/`11492921`
never ran. The active replacement retains the same immutable iteration-9
restart, exact iteration-10 dispatch/order replay, sealed panel, original-clean
closure, and same-A100 RECOVAR gates.

### Full-start K=4 graph closure

Original full-start science `11484481` ended `FAILED 1:0` after 9h51m51s on
`della-l08g5`. Both RELION arms completed, but the wrapper stopped in the
already known validator import path with
`ModuleNotFoundError: No module named 'scripts'`. Its formal iteration-10
inertness gate had already rejected all four classes at FSC-AUC
`0.998247194648/0.997600525037/0.997363409604/0.998221443830` against the
predeclared `0.999999` threshold, so RECOVAR correctly did not run. Dependent
audits `11484482`/`11484846` were canceled at zero runtime. The
restart-qualified continuation `11495747` and audits `11495748`/`11495749`
are now the sole live authoritative comparison graph; its complete submission
history is recorded in `docs/math/relion_parity_agent_notes.md`.

### Fixed-score boundary requalification

The version-1 K=1 suite keeps its denominator and definitions fixed at 34.
Its accepted baseline is 21/34 complete trajectory passes and 27/34 exact
intermediate-topology passes.  Because that snapshot used source heads
`ac5177d2` and `9d172278`, it does not silently claim the later inclusive
boundary fix from `7f5f7584`.

Clean pushed PR HEAD `3dd664c8` launched same-H100 RELION/RECOVAR replacement
runs for fixed-suite cases 2, 3, and 33 only.  Setup/science/summary jobs are
`11497146` and `11497147`/`11497148`/`11497149`.  The initially queued
summary/audits were canceled before execution when the audit-trail commit
advanced the shared worktree HEAD.  Detached-`3dd664c8` replacements are
summary `11497305` and fail-closed FSC/FSC-AUC plus exact-topology audits
`11497302`/`11497303`/`11497304`.
No scorecard checkbox changes until the corresponding audit accepts the row.
The disposable run and runtime roots are
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fixedsuite_boundary_requal_3dd664c8_20260722T111900Z`
and
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_fixedsuite_boundary_requal_3dd664c8_20260722T111900Z`;
both contain `SAFE_TO_DELETE`.

### Case-20 strict-score qualification

Case 20 has current-head functional closure but not yet strict-score closure:
an earlier replay matched all 11 RELION current sizes, kept numbered merged
FSC-AUC at least `0.999999998577`, and reached final merged FSC-AUC
`0.997760979983`, but used a different physical GPU from its RELION oracle.
A clean detached-`3dd664c8` paired replacement now runs both engines on one
H100: setup `11497498`, science `11497499`, summary `11497500`, and
fail-closed FSC/topology audit `11497513`.  Its disposable run/runtime roots
are
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fixedsuite_case20_samegpu_3dd664c8_20260722T113100Z`
and
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_fixedsuite_case20_samegpu_3dd664c8_20260722T113100Z`.

### Small-failure fixed-suite requalification

Five inexpensive failing rows from pre-`db1bf391` source are being measured
again at clean detached `3dd664c8`: cases 22, 23, 24, 26, and 32.  Setup is
`11497554`; paired same-H100 science jobs are `11497555`--`11497559`; summary
is `11497560`; fail-closed FSC/FSC-AUC and topology audits are
`11497575`--`11497579`.  No result is eligible for the fixed score until its
audit passes.  The disposable run/runtime roots are
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fixedsuite_smallfail_requal_3dd664c8_20260722T114000Z`
and
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k1_fixedsuite_smallfail_requal_3dd664c8_20260722T114000Z`.

### Fixed scorecard reaches 23/34; boundary and final-map checks continue

The accepted frozen-suite snapshot is now `strict-k1-v5-20260722`: 23/34
complete strict trajectory passes and 29/34 exact intermediate-topology
passes, up from 20/34 and 27/34 in the first frozen snapshot.  Case 23 earned
the newest checkbox only after paired same-GPU science `11501524` and strict
audit `11501622` passed the exact 13-iteration schedule, shellwise FSC/FSC-AUC,
and finalization contract.  The authoritative checked-in summary is
`docs/math/em_relion_parity_scorecard.md`; its 34-case denominator and fixture
identities remain frozen.

The active K=1 hypothesis is that inclusive current-size boundary commit
`7f5f7584` removes the old iteration-3 `162` versus RELION `164` split in
cases 2 and 3.  Detached source `84143872` contains both `7f5f7584` and
`db1bf391`; paired exact-fixture science jobs `11501888`/`11501889` and
strict auditors `11501907`/`11501908` are running or dependency-held.  No
checkbox changes until the auditors accept the complete contracts.  Case 24
is the separate final-map discriminator: integration commit `6235fb03` has
setup/science/summary/audit jobs
`11504822`/`11504823`/`11504824`/`11504831`; the old exact result missed only
the final merged cross-engine FSC-AUC gate, `0.994805104 < 0.995`.

The K=4 active diagnostic is same-A100 job `11503805` at source `7cd1aa4b`.
It reuses the accepted read-only RELION control/capture pair and exercises the
scoped observational fused-pass-2 capture for the frozen iteration-10,
half-1, class-2, 96-particle panel.  Capture is not a production repair: the
authoritative arrays remain unchanged, and geometry/scalar acceptance remains
fail-closed on the completed artifact bundle.

The integration branch is clean at `6235fb03`.  The repository-mandated
validation graph is `11504666`--`11504675`, with corrected optional-binding
unit rerun `11504718`; the original unit launch `11504665` is infrastructure
invalid because the RELION binding was absent.  Local commits remain unpushed
until the required validation graph is fully green, after which the PR must
report the frozen-score progression and exact evidence jobs.

### Case-24 final-map failure is inherited from numbered-pose divergence

The clean immutable `a2be302c` replacement supersedes the stale shared-
worktree case-24 launch.  Science `11507875` completed normally and strict
auditor `11507904` accepted all 12 numbered FSC trajectories and the exact
RELION topology.  It rejected only the final merged cross-engine FSC-AUC:
`0.991502719959 < 0.995`.  Numbered merged FSC-AUC remains at least
`0.999807820661`, and the final RECOVAR map is better against GT by
`0.008628902885` FSC-AUC.  The frozen score remains 23/34; quality thresholds
are unchanged.

A complete 3,000-particle identity join shows that final all-data scoring is
not the first divergent boundary.  Iteration 12 already has 392 particles
above 0.5 degrees of cross-engine pose error, almost identical to the final
391, with angular-error correlation `0.9987`.  The earliest greater-than-five-
degree split is iteration 2, original index 2767 / RELION stack image 2768 /
half 1.  Both engines report two significant samples and nearly the same Pmax
(`0.385798991` versus `0.385362`), but select poses separated by `9.18615`
degrees.

Commit `9565b8a1` adds no scoring or reconstruction change; it only forwards
the existing pass-1 significance capture controls through the K=1 launcher.
Launcher validation is 39 passed.  Setup `11509108` failed closed on queued-
source HEAD drift before science, so its blocked jobs were canceled.  The
immutable detached-`0da399c4` replacement uses setup/science
`11509611`/`11509612`, and job `11509654` compares pass-1, pass-2, and
reconstruction candidate tables.  The durable capture root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case24_it2_particle2767_capture_0da399c4_20260722T213929Z`.

### Unit validation closes; mandatory outlier gate stays red

Replacement unit job `11507920` completed in `00:50:03` with 5,586 passed,
53 skipped, 0 failed, and 0 errors (5,639 total).  It validates the canonical
pixi-path assertion repair at `a2be302c`.  All other integration groups pass
except the cryo-ET outlier regression, which also reproduces on clean `dev2`.
The repository-mandated gate therefore remains red and pushing remains
disallowed even though the EM/unit regression is closed.  The durable result
root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_parity_longtest_6235fb03_20260722T190500Z`.

### Case-26 strict failure is also an inherited pose tail

Clean detached-`a2be302c` science job `11508258` completed normally in
`00:23:35`.  The strict FSC and topology commands were also run immediately
against the finished immutable output while scheduled CPU auditor `11508284`
waited for priority.  All 11 numbered iterations and the exact finalization
topology pass.  The only FSC failure is the final merged cross-engine FSC-AUC,
`0.963324126445 < 0.995`; RECOVAR-minus-RELION merged GT FSC-AUC is positive,
`+0.009268703399`.  Case 26 therefore remains unchecked and the frozen score
remains 23/34 strict and 29/34 topology.

The input-identity particle audit shows the same boundary shape as case 24.
At numbered iteration 11, 835/1000 particles are within 0.5 degrees across
engines; final all-data has 829/1000 within 0.5 degrees.  Median angular error
stays numerical (`5.63e-6` versus `5.69e-6` degrees), while p95 changes only
from `2.34529` to `2.38760` degrees.  Final all-data therefore exposes an
already-present pose tail at full Nyquist rather than creating a new final-join
tail.  This reinforces the case-24 iteration-2 candidate decision as the
earliest active operand-level target.

The strict FSC, topology, particle JSON, and particle-array SHA-256 values are
`107a6983aa496346e50e875196c45fbb673e8a51187828b2758db6962285227e`,
`3362fb2e785a42922b5d98414fca05c26f5ae6b04029c91e04224394e4851d2b`,
`3de8993c743c3015838286148a7920d1b7116e796a22ebdc2a98d4e94dfd7d60`,
and `2aa103ee09442d4a4fd00cdc38ec0aa89d0e4aaca586582189f37d43cf9258f6`.
The durable case root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fixedsuite_unresolved_a2be302c_20260722T205500Z/cases/26_tiny_severe_1k_g128_radial_noise5_nonuniform_pct30_bf80`.

### Frozen score reaches 25/34; exact-topology final family expands

Snapshot `strict-k1-v6-20260724` advances the fixed suite to 25/34 strict
trajectory passes, 31/34 exact-topology passes, and 34/34 evaluated.  The
denominator, case definitions, thresholds, and manifest remain unchanged.
Exact-fixture case 2 passed with final merged cross-engine FSC-AUC
`0.998574606387` and GT delta `+0.005625197355` (science/audit
`11501888`/`11501907`).  Exact-fixture case 33 passed at
`0.999734254440` and `+0.000244293524` (`11508260`/`11508286`).  Both pairs
used a single physical GPU, converged legitimately, ran final all-data, and
kept grid correction unset/off.  Scorecard v6 ledger SHA-256 is
`32c6512a8507f7b17a59d0be527fa5c9609067e0d8f598a2d108bed9a3fc8a56`.

Cases 4 and 5 now have complete exact-identity particle audits from Slurm
array `11553320` (both `COMPLETED 0:0`).  Their numbered topology and
convergence match RELION exactly.  Their last-numbered versus final fractions
within 0.5 degrees are `93.085% -> 92.671%` for case 4 and
`92.179% -> 91.929%` for case 5; medians remain approximately `4.8e-6`
degrees.  Translation and Pmax distributions likewise change only slightly
across the boundary.  Yet merged cross-engine FSC-AUC drops from
`0.999635726831 -> 0.991973796224` and
`0.999954527650 -> 0.984301765024`.  Cases 4/5 therefore join cases
24/26/32 in the inherited final-boundary family: final all-data exposes and
amplifies an existing pose/translation tail at full grid rather than creating
a new pose-writeback mismatch.

At the last case-4 transition, significant-support mismatch enriches the next
greater-than-0.5-degree pose tail by `1.96x`; the top 5% absolute-Pmax-error
cohort enriches it by `2.81x`.  This keeps the causal target at the earlier
score/posterior/support boundary.  The sealed classification is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_cases04_05_particle_audit_74f89c60_20260724T022000Z/FINAL_BOUNDARY_CLASSIFICATION.md`
(SHA-256
`b28c58a09575d9900f8091725ef2b18ac24c5ff1e76848ba9df4b01de40ff889`).
Map quality remains FSC/FSC-AUC only; correlation is not computed.

The complete per-iteration audit sharpens that boundary.  At RECOVAR
iteration 0 / RELION iteration 1, cases 4 and 5 have exactly equal Pmax and
significant-support arrays for all 100,000 particles; only 9 and 3 particles,
respectively, exceed the 0.01-Angstrom translation-error threshold.  At the
next iteration, all 100,000 Pmax values become non-identical, Pmax absolute
p95 reaches `5.23240e-4` / `1.13895e-4`, significant support differs for
331 / 207 particles, and the greater pose-or-translation tail reaches 60 / 25
particles.  The broad continuous split therefore begins at scoring iteration
2, before the late final-map amplification.

Same-GPU diagnostic `11558427` tests the case-4 map operand at precisely that
boundary.  Both two-iteration arms use the same exact incoming RELION replay
state; one retains the resident RECOVAR iteration-1 half maps and the other
substitutes only the exact RELION iteration-1 half maps for scoring iteration
2.  Identity audits compare both arms to RELION iteration 2 using Pmax,
support, pose, and translation distributions.  Initial job `11558403` failed
in a three-second JAX-path provenance assertion before science and is not
evidence.  The corrected replacement passed checkout, pixi, CUDA-device, and
clean-HEAD gates.  Run root:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_it2_relref_counterfactual_5cb01ec1_20260724T040135Z`.

### K=4 fused capture accepts legitimate single-row soft buckets

K=4 science `11503805` completed nine numbered iterations, then aborted at
the iteration-10 observational capture.  The per-bucket validator required
every target-bearing sparse bucket to contain at least one particle with
multiple positive rotation rows.  The first target compact bucket
legitimately had one positive rotation row per selected particle after
reconstruction pruning, even though the soft algorithm—not WTA—was active.
This was a diagnostic invariant failure, not a science/posterior failure.

Commit `74f89c60` retains the required at-least-one-positive-row invariant in
soft mode and the exactly-one-positive-row invariant in explicit WTA mode,
without demanding a multi-row witness from each independent bucket.  Focused
device-signature, sparse-bucket, and validator coverage passes 83/83; scoped
Ruff and `git diff --check` pass.  Same-A100 replacement `11553264` reruns the
RELION control/capture and RECOVAR arms in one allocation from the immutable
detached commit.  Frozen case-3 longer-budget science/audit
`11553236`/`11553237` separately supersede a 24-hour timeout without changing
the fixture or algorithm.

### Case-24 earliest split localizes to pre-prior likelihood

Exact-physical-GPU RELION replay `11555181` completed both iterations and the
sealed verbose operand capture on the same A100 used by the original RECOVAR
science.  Accepted audits `11556109` and `11556113` join the resulting
StoreWeightedSums and generic fine-pass tables to the immutable RECOVAR
iteration-2 capture for original image 2767.  The accepted generic comparisons
have exact support identity: 64/64 fine candidates and 13/13 reconstruction
candidates are common (Jaccard 1.0).  Rotation, translation, and combined log
priors agree within `4.77e-7`, `4.77e-7`, and `9.54e-7`.

The two competing candidates reveal the causal boundary.  For RECOVAR
candidate 11 minus RELION's candidate 6, RELION's pre-prior likelihood margin
is `-6.068603515625`, while RECOVAR's is `-5.964490234852`; their difference
is `0.104113280773`.  The shared prior swing toward candidate 11 is
`+6.009656250477`.  RELION therefore ends at a `-0.058948218822` total margin
and selects candidate 6, whereas RECOVAR ends at `+0.045166015625` and selects
candidate 11.  This is a likelihood-operand/reduction mismatch, not a support,
prior, scheduler, or tie-break mismatch.

Comparator commit `e1d96a0c` adds the required K=1 compact
StoreWeightedSums-to-global-rotation mapping and treats an inapplicable
coarse-psi grid as optional under automatic matching.  Comparator/parser
coverage passes 32/32; scoped Ruff and `git diff --check` pass.  The durable
generic prior JSON SHA-256 is
`4ea57126e538ad5bc76803bbac790af35b1580a92a40681acff51a8fb27bc02b`;
the full replay history and artifact hashes are in
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case24_it2_particle2767_capture_0da399c4_20260722T213929Z/provenance/PATCHED_RELION_REPLAY_11552815_11552816.md`.
This localization does not change the frozen score: v6 remains 25/34 strict,
31/34 exact topology, and 34/34 evaluated.

An identical-input GPU counterfactual closes the remaining ambiguity.
Job `11557150` feeds the captured RELION PPref and all 16 bit-exact fine
rotations through RECOVAR's immutable `0da399c4` production texture projector.
Every complex projection row is bit-exact to RELION.  The resulting candidate
raw-diff2 margin is `6.068725585938` versus RELION `6.068603515625`, a single
`0.0001220703125` float32 step rather than the live `0.10411` margin gap.
Source-map job `11557190` then rebuilds the PPref through the real RELION
binding: the saved RELION iteration-1 half-1 map matches the captured PPref at
relative L2 `4.65e-8`, while RECOVAR's saved iteration-1 half-1 map is
`0.0804728` away.

Thus the observed likelihood operand is different because the incoming
iteration-1 reconstruction/reference state is already different.  Identical
PPref, matrices, image operands, weights, texture projection, and exact
Gaussian reduction reproduce RELION to its float32 floor.  The next causal
target is iteration-1 half-map/BPref formation, not fine E-step scoring or a
tie-break.  The GPU and source-map report SHA-256 values are
`567db079ad30cbbc5956db19ee2cc9b2d37e3a867a60cca9eac0cb8e836ebe5d`
and
`f3535fe727b15f7e8a8e59bbb48af76ebb4b6da520a7bbdbd09ee335c8c36a4d`.

### Case-24 iteration-1 mismatch is a numerical winner flip

The complete iteration-1 identity audit finds only one pose mismatch among
3,000 particles: original index 1901.  Same-physical-GPU RELION job
`11557396` sealed its full first-iteration score surface, and replacement
RECOVAR job `11557748` completed on the same `della-l09g5` allocation.  Audit
job `11557886` then compared every one of the `1,069,056` coarse candidates
using exact RELION-pixel-major to RECOVAR-psi-major rotation indexing.
Candidate topology is exact: 1,069,056/1,069,056 common, Jaccard `1.0`.

The centered pre-prior score difference has maximum absolute value
`1.7881393432617188e-6`; 99.5518% of candidates are within `5e-7` and all are
within `2e-6`.  RELION's winning candidate `(16551, 14)` exceeds RECOVAR's
winner `(16550, 14)` by only `1.4901161193847656e-7`; RECOVAR reverses the
same two-candidate ordering by `2.086162567138672e-7`.  They are rank 1 and 2
in both engines, use the identical translation, and their coarse rotations
are separated by `7.38055` degrees.

This is ordinary float32-scale winner instability on an effectively tied
first-iteration surface, not evidence for a systematic scorer, prior,
support, scheduler, or tie-threshold defect.  No production threshold is
changed.  It also explains why the isolated iteration-1 label mismatch can
seed the observed iteration-1 map/reference residual and later case-24 pose
tail.  Full-grid decoder commit `b6b2b01e` passes 33 focused comparator/parser
tests plus scoped Ruff.  The accepted audit JSON SHA-256 is
`dbe8d3f4e3749e1491e410376c73993d84cd1929c5d4c3afa14cc3c36df4f53e`.
The durable replay root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case24_it1_particle1901_capture_0da399c4_20260724T030330Z`.
The frozen score remains 25/34 strict, 31/34 exact topology, and 34/34
evaluated.

## 2026-07-25 live K=4 preprocessing discriminator

Same-A100 job `11598766` completed `0:0` in `00:30:22` from clean
integration commit `dd6d4063774e36136bf9551ee828d3e113f46974`. It ran the
production host-NumPy image Fourier path and the existing JAX/cuFFT path
sequentially on physical GPU
`GPU-6f45f415-9d0b-d562-9ff3-c9fb7bc53aa7`, then stopped each arm immediately
after capturing the pinned iteration-10 K=4 fine-score boundary: original
index 42987, class 2, current size 74, global rotation 2956, and translations
56--59. The host and JAX arms took 915 and 886 seconds, respectively; the
bounded JAX arm was 29 seconds (`3.17%`) faster.

All checked topology fields are exact between the live arms: fine
translations, oversampled rotation indices, parent map, candidate mask,
rotation prior, and translation prior. Against the accepted passive RELION
CUDA operand:

| Scope | host residual L2 | JAX residual L2 | JAX residual-energy change |
|---|---:|---:|---:|
| all four candidates | `4.8828125e-4` | `2.44140625e-4` | `-75%` |
| production-exact translations 56/58/59 | `3.9867997e-4` | `1.9933999e-4` | `-75%` |

The live host centered residual is
`[+2.44140625e-4,-2.44140625e-4,-2.44140625e-4,+2.44140625e-4]`; JAX reduces
it to
`[+1.220703125e-4,-1.220703125e-4,-1.220703125e-4,+1.220703125e-4]`.
The host class-2 artifact is byte-identical across three independent live
captures (SHA-256
`ddc8d65de595699107b1e946f0fbe1dcb61d39d43191e67a3a364c6ac863a844`),
so the comparison is not a host-repeat fluctuation.

This live A/B confirms that image Fourier preprocessing is a causal,
candidate-varying part of the K=4 fine-score residual. It does not yet qualify
JAX/cuFFT as the production default: the next gate must carry the alternate
backend through a K=4 trajectory and compare class assignments plus
shellwise FSC/FSC-AUC, with the fully derived RELION-CUDA preprocessing path
kept as a separate discriminator.

The accepted comparison is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_preprocess_live_pair_retry_f2ccc270_20260725T043000ET/analysis/LIVE_PREPROCESS_RELION_COMPARISON.json`
(SHA-256
`348c462c40c62f4b5a3b83de42fbaee81adf984a8318b2c59db1c7e0da685a74`).
The run and runtime roots contain `SAFE_TO_DELETE`; grid correction and
forced final-after-max were unset.

Two earlier attempts are explicitly non-science failures. Job `11597063`
failed before replay because an unrelated inherited BPref fused-atomics flag
lacked its required device-signature scope. Jobs `11597459` and `11598131`
reached the deterministic host target but stopped before the JAX arm because
the diagnostic class selector still emitted four small class files instead of
one. Job `11598766` pinned the loaded sparse-helper source, accepted exactly
those four files, and analyzed only the predeclared class-2 artifact. This
diagnostic fan-out does not affect the scientific comparison.

No frozen K=1 case is promoted. Snapshot `strict-k1-v6-20260724` remains
25/34 strict, 31/34 exact topology, and 34/34 evaluated.

### Fully derived RELION-CUDA preprocessing reproduces the live improvement

Same-A100 job `11599918` completed `0:0` in `00:26:46` from clean
integration commit `ede6df86c2644e07de1fec8c30acc7657821e6db`. It ran the
production host-NumPy path and RECOVAR's fully derived `relion_cuda`
preprocessing path sequentially on physical GPU
`GPU-2f2a8197-bcc8-ec41-fc6f-dfb2b5aaf4fa`. The arms took 878 and 722
seconds, respectively, so `relion_cuda` was 156 seconds (`17.77%`) faster in
this bounded stop-after-capture replay.

The comparison uses the same pinned iteration-10 class-2 boundary and passive
RELION CUDA operand as job `11598766`. Fine translations, oversampled
rotation indices, parent map, candidate mask, rotation prior, and translation
prior are exact. The host and `relion_cuda` residual L2 values are
`4.8828125e-4` and `2.44140625e-4` over all four candidates, and
`3.9867997e-4` and `1.9933999e-4` over production-exact translations
56/58/59. Thus the fully derived backend independently removes exactly 75% of
the accepted centered residual energy, reproducing the selected-candidate
signature from the earlier JAX/cuFFT arm.

The complete JAX/cuFFT and `relion_cuda` score panels are not bitwise
identical: their finite-support masks agree, but 315 of 544 finite
`scores_pre_prior` entries differ, with maximum absolute difference
`2.44140625e-4`. Their selected four-candidate centered signatures nevertheless
agree exactly. This supports the preprocessing family as the causal branch,
but does not establish backend equivalence or authorize a production default
change. The next acceptance gate remains a full K=4 assignment and shellwise
FSC/FSC-AUC trajectory comparison.

The accepted report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_preprocess_relioncuda_pair_ede6df86_20260725T043500ET/analysis/LIVE_PREPROCESS_RELION_CUDA_COMPARISON.json`
(SHA-256
`fdaa3280131683974e4f446fd04ff0cb1cec42345859ddc5db982f07b5fbce37`).
The host and `relion_cuda` class-2 artifact SHA-256 values are
`ddc8d65de595699107b1e946f0fbe1dcb61d39d43191e67a3a364c6ac863a844`
and
`fe802d6aa1bf4d560acbe0ba5aa0a9c5531a810b39a266aa330991a8be9b22df`.
The run and runtime roots contain `SAFE_TO_DELETE`; grid correction and
forced final-after-max were unset.

No frozen K=1 case is promoted. Snapshot `strict-k1-v6-20260724` remains
25/34 strict, 31/34 exact topology, and 34/34 evaluated.

### Full K=4 backend trajectory gate

The score-boundary result is now promoted to the required trajectory-level
experiment without changing production defaults. Same-A100 science job
`11600592` runs 15 autonomous K=4 numbered iterations first with
`host_numpy`, then with `relion_cuda`, from clean detached commit
`4181d340997e548af36c6458cce825e133dba95a`. Dependent CPU audit
`11600593` checks exact dispatch/schedule/convergence/finalization topology
and computes shellwise FSC/FSC-AUC plus class-assignment agreement against the
immutable accepted RELION trajectory.

The fixed K=4 acceptance count is the number of `(iteration, class)` direct
cross-engine FSC-AUC checks at or above `0.995`: 60 checks over 15 iterations
and four classes. The audit also records how many iterations pass all four
class gates, the minimum class agreement, the minimum GT FSC-AUC delta, and
same-physical-GPU wall time. Script
`scripts/compare_k4_backend_trajectories.py` compares those counts
fail-closed and rejects cross-GPU pairs. It also maps both saved RECOVAR
class-assignment trajectories through their independently audited
RECOVAR-to-RELION permutations, then reports direct backend agreement and
mismatch counts at every iteration. It does not use map correlation.

Checked snapshot `k4-host-ac5177d2-20260719` is the fixed production-host
baseline: 40/60 direct class checks pass and 9/15 iterations pass all four
classes, with exact topology. Its full per-iteration count vector and evidence
hashes are stored in
`docs/math/em_k4_backend_trajectory_baseline_v1.json`. The running pair is an
improvement only if it increases those fixed counts without breaking exact
topology; adding later cases cannot change this denominator.

Science and audit run from
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_full15_host_relioncuda_samegpu_4181d340_20260725T051500ET`;
runtime/cache state is isolated under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/em_k4_full15_host_relioncuda_samegpu_4181d340_20260725T051500ET`.
Both roots contain `SAFE_TO_DELETE`. Grid correction and forced final
all-data after non-convergence are unset. This pending K=4 count complements
but does not alter frozen K=1 snapshot `strict-k1-v6-20260724`.

### Current K=4 status: production fine operands close the score boundary

This status summary precedes the detailed seed-exact causal log below.

RELION source commit `96387461fdaa18e4d23d4dbc57477039e3145b77`
adds a bounded passive capture for stack 42988, particle 36655, class 2,
rotation-local 124, and translations 56--59. Build job `11594507` completed
`0:0`. Paired control/capture job `11594695` completed `0:0` in 24:33 on
physical A100 `GPU-46a58f9b-04f9-5785-979f-8d07c76fa054`; control and capture
took 755 and 698 seconds. Dispatch and particle fields are exact, and the four
class-map FSC-AUC values are `0.999999992455`--`0.999999995250`, above the
fixed `0.999999` inertness threshold.

The captured CUDA kernel uses 256 reduction lanes and translation chunks of
seven. Its per-pixel arithmetic is the contracted float32 expression
`fmaf(diff_real, diff_real, roundf(diff_imag * diff_imag))`, followed by the
separate `0.5*corr` multiply. Passive replay is bitwise exact for translations
56, 58, and 59. Translation 57 differs from the production result by one
float32 ULP (`2255.6376953125` versus `2255.637451171875`), so the validator
accepts the structurally complete capture while recording 3/4 exact production
replays.

Backend-faithful analysis from RECOVAR commit
`2544c33a880cf8f7926247fc7f2b0ac81d399048` resolves the remaining operand
boundary:

- RELION's captured reference is bitwise equal to the RECOVAR projection after
  the documented global Fourier-sign alignment.
- Both production paths zero the DC score weight. The largest remaining scaled
  `corr` difference is `8.3673513e-11` (relative L2 `3.4613115e-7`).
- The aligned shifted-image operand differs at relative L2
  `6.3840108e-5`; it is the leading centered counterfactual, removing 72.7% of
  four-candidate residual energy and 75% on the three production-exact
  candidates. Reference-only and `corr`-only substitutions remove none.
- Replaying the captured reference, RECOVAR's captured
  `dataset_native` background-fill preprocessing, its exact
  `image_correction / scale^2` direct-score factor, the production DC rule,
  CUDA FMA, and 256-lane tree reproduces RECOVAR's saved centered candidate
  score on all three production-exact translations to maximum absolute error
  `2.7105054e-20`. The only all-candidate nonclosure is translation 57, the
  already identified one-ULP passive-replay exception.

This closes the causal chain from the previously measured K=4 posterior
residual through fine data score to the shifted-image preprocessing operand.
Projection, priors, normalization/exp, posterior division, significance
support, factor placement, DC handling, and reduction replay are no longer
candidate primary causes. The next bounded production experiment should
compare the native background-fill/host-FFT score image directly with
RECOVAR's existing RELION-CUDA preprocessing path on the same saved candidate
panel before changing default behavior.

Durable evidence:

- accepted science root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_fineoperand_capture_9638746_20260725T055500Z`;
- fine-operand artifact SHA-256:
  `a81cf6c18e9ce47864c119ae3d827e3aeb64121bf8d071e01176e4bc350e1102`;
- final comparison:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_fineoperand_analysis_2544c33a_20260725T031500ET/analysis/FINE_OPERAND_COMPARISON.json`
  (SHA-256
  `5c7d3c625a659c3a23983038a4be52f5b925873805f2b34626f530b959adaa74`);
- final validation SHA-256:
  `676d5b2d98ed1e3990f88ac327b42c6bb69853c532502cd526e8d77561b357d6`.

The fixed progress metric is deliberately unchanged:
`strict-k1-v6-20260724` remains 25/34 strict, 31/34 exact topology, and 34/34
evaluated.

### Seed-exact K4 boundary closes topology and support

The exact-input restart replay uses RELION's live iteration-10 perturbation
`-0.12305957078933716` rather than the rounded sampling-STAR value.  Replay
`11584817` and independent audit `11585023` match all 96 coarse-parent sets,
all 96 fine-candidate sets, and all 96 positive-contributor rotation sets.
Both independently generated matrix panels are exact, and every reached-pixel
set matches with zero one-sided pixels.  The earlier shell-37 mismatch was
therefore entirely due to rounding the replay perturbation.

With topology and scatter support exact, audit `11586748` compares 120
matched class-2 contributors over 257,461 reached pixels.  A positive real
scalar fits 109/120 contributors at relative residual `1e-5`, or `90.8333%`,
below the predeclared 95% causal threshold.  Weight residuals remain small
(maximum `6.96009e-7`) while complex-data residuals reach `4.22837e-5`.
The result is a pixel-varying source difference rather than a support,
geometry, scatter, or per-rotation posterior-normalization difference.
The next causal panel captures RELION's translated-image, CTF, inverse-noise,
posterior, and accepted-term operands for all 11 non-scalar contributors plus
six scalar controls.  This diagnostic does not promote a frozen case:
`strict-k1-v6-20260724` remains 25/34 strict, 31/34 exact topology, and 34/34
evaluated.

### Case-24 initial-projector plus reduction intervention closes iteration 1

The case-24 first-iteration winner residual is a two-factor boundary.  A
full reduction-only run at source `9abd79fb` rescored four margin-qualified
coarse top-two pairs with the RELION 128-lane float32 tree but changed zero
winners.  It therefore rejected reduction order alone as sufficient.  A
separate operand capture localized the other factor to the initial-reference
path: converting the resident complex64 Fourier reference back through real
space before constructing `Projector::data` produced relative L2
`2.75962e-7` against the captured RELION projector.

Commit `a521cfb6` adds an opt-in direct float64 real-reference handoff for the
zero-based iteration-0 RELION projector only.  It does not alter the resident
Fourier state, later projectors, or default behavior.  In the combined
one-iteration run on local A100
`GPU-dc6576aa-e1e4-6055-4a5e-d0fa809f3983`, the saved projector residual
fell by `10.82x` to `2.55164e-8`.  Of 3,000 images, only four top-two pairs
qualified for exact rescore and only original index 1901 changed.  Its
assignment moved from RECOVAR `(16550, 14)` to the exact RELION winner
`(16551, 14)`.

The corrected assignment closes the reconstructed iteration-1 map boundary:

| Map | Baseline RECOVAR vs RELION FSC-AUC | Combined vs RELION FSC-AUC |
| --- | ---: | ---: |
| half 1 | 0.999998394276 | 0.999999999956 |
| half 2 | 0.999999741584 | 0.999999999955 |
| merged | 0.999999380590 | 0.999999999972 |

These are normalized non-DC FSC-AUC values over 62 finite shells; correlation
was not computed.  The accepted diagnostic is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case24_tree_top2_58457106_20260724T033852Z/case24_combined_projector_tree_intervention_audit.json`
(SHA-256
`7343196ea7ca9643bb586ca97159564badc85687102a3f4c179269b6729f1502`).
The full run root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case24_tree_top2_58457106_20260724T033852Z/combined_a521cfb6`.
Slurm attempts `11561699` and `11561811` were canceled while pending with
zero runtime and are not scientific evidence.

This is an opt-in one-iteration causal result, not yet a fixed-suite pass.
The next gate is the complete case-24 trajectory with both interventions and
the unchanged strict FSC/topology auditor.  Snapshot
`strict-k1-v6-20260724` therefore remains 25/34 strict, 31/34 exact topology,
and 34/34 evaluated.

### Case-24 residual is a launch-sensitive one-ULP RELION winner

The complete artifact-pinned case-24 intervention at source `b826bc52` used
the preserved real initial reference and bounded first-iteration top-two
float32 tree rescore.  Setup `11562037`, science `11562038`, and summary
`11562039` completed `0:0`.  Strict audit `11562082` retained exact
intermediate topology but failed final merged cross-engine FSC-AUC at
`0.994801463093 < 0.995`; final half1/half2 FSC-AUC is
`0.999040692683/0.995870180922`, and merged RECOVAR-minus-RELION GT FSC-AUC
delta is `+0.008173125002`.

The first three numbered merged maps are effectively exact at
`0.999999999973`, `0.999999999903`, and `0.999999999901` FSC-AUC.  Exact
particle-state comparison localizes the first discrete split to original
index `2332` at iteration 3: the Euler angle is exact, while x translation
differs by one fine step (`2.125` Angstrom).  Iteration 4 is the first
material map response.

Patched replay `11562574` closed candidate topology and winner-to-pose mapping,
but binary audit found that it used source `f2c1a3` rather than the installed
stock source `d476e6`; it is localization evidence, not an exact stock score
oracle.  No-dump control `11562830` then ran installed stock and the older
patched binary serially on the fixed-case node and A100 UUID
`GPU-6a3cea75-90ac-d3de-7c1a-a8158412a9f4`; both chose translation `59`.
The immediately adjacent installed-stock arm in exact-source job `11563252`
chose translation `57`.  Those two installed-stock runs have exact
iteration-1 state, exact poses/support through iteration 2, and exactly one
iteration-3 translation difference.  Their merged map FSC-AUC remains
`0.999999999999`, `0.999999999993`, and `0.999999999992` through iterations
1--3.

Job `11563252` also ran a source-exact d476e6 rebuild with a hash-pinned
four-file capture patch.  The rebuilt executable is not byte-identical to
installed stock.  Its dormant arm chose translation `57`; its narrow active
capture chose `59` and recorded raw scores `1442.52734375` versus
`1442.5274658203125`.  The `0.0001220703125` gap is exactly one float32 ULP,
and the normalized posterior gap is `3.93430357086353e-5`.  Dormant versus
active maps retain merged FSC-AUC at least `0.999999999992` through iteration
3.  Because installed stock itself selects both sides across adjacent
launches, the active-capture outcome is not interpreted as a causal dump
effect.

This is a qualified launch-sensitive hard-winner boundary.  Forcing RECOVAR
to translation `57`, changing the arithmetic, or adding a tie-break would
overfit one unstable RELION realization.  The finding is recorded as
oracle-stability telemetry and does not rewrite the immutable metric:
`strict-k1-v6-20260724` remains 25/34 strict, 31/34 exact topology, and 34/34
evaluated.  Durable roots are
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case24_it3_relion_winner_probe_20260724T092606Z`
and
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case24_it3_d476_score_probe_20260724T094334Z`.

### Nine iteration-1 winners explain the case-4 accumulator residual

Same-H100 capture job `11561082` completed `0:0` in `00:14:44` on
`GPU-9f98ccbf-3c62-c54f-7409-7eb58845ad4a`.  It captured high-precision
per-particle numerator and weight contributions for all nine known
iteration-1 winner exceptions (original indices `5234`, `6322`, `7738`,
`17353`, `28139`, `43838`, `51977`, `60368`, and `72654`) and replayed their
captured RECOVAR winners through the standard GPU accumulator path.  Both
half-set controls are bit-exact to the accepted standard replay.

A second same-GPU intervention retained every captured RECOVAR image/CTF and
scatter operand but replaced only each target's winning rotation and
translation by the corresponding RELION winner.  The two GPU replays
completed before wrapper job `11561160` rejected an over-strict aggregate
cross-run bit-exact assertion.  An independent CPU recovery audit accepted
the fresh-versus-accepted aggregate differences as a bounded floating
envelope: relative L2 is at most `6.37e-8`, roughly five orders of magnitude
below the RELION residual.

The pose-only intervention removes essentially the entire accepted joined
BPref gap.  Half-1 numerator relative L2 falls from `0.0018737330` to
`1.9293688e-6` and weight from `0.00026939846` to `8.4532979e-7`, removing
`0.99999894` and `0.99999015` of residual energy.  Half-2 numerator falls
from `0.0030036818` to `3.0769983e-5` and weight from `0.00045892132` to
`7.1740414e-6`, removing `0.99989506` and `0.99975563` of residual energy.
The worst remaining residual-norm ratio is `0.0156324`.

This is a causal pose intervention: the nine first-iteration winner choices,
not a general M-step/backprojection discrepancy, account for the case-4
accumulator boundary.  Two targets also expose coarse/global routing directly.
For original index `6322`, the RELION winner is absent from RECOVAR's selected
eight-rotation fine subset and differs by `150.7523` degrees; original index
`60368` differs by `165.1183` degrees.  The next diagnostic target is therefore
the first-iteration coarse/global score grid and winner route for these large
rotation flips.  No reconstruction, tau2, general scatter, or posterior
threshold patch is supported.

The accepted audit JSON SHA-256 is
`6fe333a95b07495185d95103c8b1e70d0c1c9d91cca5dfdd705d14839e3ab553`.
The contribution-manifest SHA-256 is
`8868715c20171df35f19f91cc18cce2b89e7c3fb67e9a8f1e79314cb6c139354`.
The durable run root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_it1_winner9_contributions_608c509d_20260724T064102Z`.
Pre-science job `11561065` failed only its JAX symlink provenance assertion in
seven seconds and produced no scientific result.  The frozen score remains
25/34 strict, 31/34 exact topology, and 34/34 evaluated.

### Case-4 iteration-2 map/state counterfactual is map-leading but mixed

Same-H100 science job `11558427` completed two sequential 100,000-particle
arms on physical GPU
`GPU-49c1a223-be61-858b-49d8-d8b0347ac252` from source `5cb01ec1`.  Both
arms replay exact incoming RELION non-map state through scoring iteration 2.
The resident arm keeps its RECOVAR-produced iteration-1 half maps; the
`relref` arm substitutes the exact RELION iteration-1 half maps.  Neither arm
forces final all-data, and grid correction remains unset/off.

Relative to the autonomous boundary, exact incoming non-map state alone is
nearly null: mean and p95 absolute-Pmax-error ratios are `0.951815` and
`0.989488`, and significant-support mismatches change only from 331 to 323.
Exact maps conditional on that state are the leading effect: the ratios
become `0.256586` and `0.189190`, support mismatches fall from 323 to 92, the
fraction within 0.5 degrees rises from `0.99958` to `0.99992`, and the
fraction within 0.01 Angstrom translation rises from `0.99947` to `0.99983`.
The combined exact-state-plus-map ratios against the original autonomous arm
are `0.244222` and `0.187201`, with support mismatches 331 to 92.

The predeclared dominance gate required both Pmax ratios at most `0.10` plus
fewer support mismatches.  The accepted classification is therefore
**mixed**, not map-dominant: the iteration-1 half maps explain most of the
broad iteration-2 residual, but an identical-input scorer/candidate residual
remains.  This supports a matched iteration-1 BPref/reconstruction capture
and a narrow residual-candidate audit; it does not support a posterior
threshold, state-only, or unconditional map-only production patch.

The GPU wrapper exited `1:0` only after both arm markers and both hash-pinned
identity reports were written because it asserted the obsolete report status
`pass`; the current auditor emits `complete`.  Recovery audit `11559766`
validated both JSON/NPZ manifests and arm provenance and completed `0:0`.
Its accepted classification JSON SHA-256 is
`0496eb4b4247a308f9ab3012ed2fc97389da2ae5a2271f0eb179bde9d0e18a3f`.
The corrected audit launcher SHA-256 is
`b712c20bb4edb6e9bcd13f141e2aa6892c5fb2d953afe9f7413b1feb24dde933`.
Superseded dependency audit `11558553` was canceled with zero runtime.  The
fixed score remains 25/34 strict, 31/34 exact topology, and 34/34 evaluated;
no correlation metric is used.

### Case-4 iteration-1 residual is upstream of reconstruction

Same-H100 capture science `11559949` and independent audit `11559964`
completed `0:0` from source `8a3737af`.  RECOVAR first captured the native
x-half M-step, public pre-join, and public post-low-resolution-join
accumulators.  A minimally patched RELION `d476e6f` then captured the joined
BPref plus post-reconstruct, post-initial-lowpass, and post-solvent-flatten
maps on the same physical H100
`GPU-8fdb5482-ff52-be6a-c41a-cda8af052492`.

The passive RELION capture passes its predeclared inertness gate.  All 100,000
particle poses, translations, Pmax values, and significant-support counts are
exactly equal to the immutable oracle.  Captured-versus-oracle half-map
FSC-AUC is `0.9999999999965` / `0.9999999999966`, with through-shell-28
FSC-AUC `1.0` to displayed precision.  RECOVAR native post-x0 and public
pre-join accumulators are bit-exact, so native-to-public conversion is not the
residual.

The first nonzero cross-engine boundary is the joined BPref.  RECOVAR versus
RELION numerator relative L2 is `0.00187373` / `0.00300368` for halves 1/2;
weight relative L2 is `0.000269398` / `0.000458921`.  Conversely, RELION's
captured BPref reconstructed through RECOVAR matches RELION's
post-reconstruct maps at FSC-AUC `0.999999999699` /
`0.999999999634`.  The first 18 derived tau2 shells match the RELION model at
relative L2 `2.07e-7` / `2.08e-7`, and final flattened-map FSC-AUC is at least
`0.999999999984`.  Reconstruction, tau2 formation, and post-processing are
therefore closed for this boundary.

The iteration-1 map difference remains small after regularization
(RECOVAR-accumulator reconstruction versus RELION post-reconstruct FSC-AUC
`0.999987352` / `0.999960909`) but is sufficient to seed the map-leading
iteration-2 amplification.  The next discriminator is the excess accumulator
residual from the nine known iteration-1 winner exceptions versus
matched-winner M-step/backprojection arithmetic.  No reconstruction, tau2,
posterior-threshold, or unconditional map-substitution patch is supported.

The accepted audit JSON SHA-256 is
`7cf9a60c6fa824a43603d3c095462a40848c697247d1ca32c00910ff671cd13c`.
The science-marker SHA-256 is
`81094fc19ee61b6b329eac7ff87318a170b6d54eecce2f17289b3b721a3fa920`.
The durable run root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_it1_native_relion_bpref_8a3737af_20260724T050935Z`.
The frozen score remains 25/34 strict, 31/34 exact topology, and 34/34
evaluated.

### Case-4 coarse-grid flip is an exact RELION float32 tie

Same-H100 job `11562639` completed `0:0` in `01:24:39` on physical GPU
`GPU-f6cfb4eb-6f8b-0df7-4ec9-8ec065affa8f`.  It captured the complete
first-iteration/current-size-56 coarse score grid for fixed case-4 original
particle 6322 from patched RELION and RECOVAR sequentially.  The comparison
is topology-complete: each engine has 1,069,056 candidates, all 1,069,056
mapped identities are common, Jaccard is 1.0, and there are no duplicate or
engine-only keys.

The score surfaces are otherwise closed.  Their aligned correlation is
`0.9999999999954908`; centered RECOVAR-minus-RELION differences have mean
`-3.108366945831097e-7`, p95 absolute `5.140900611877441e-7`, and maximum
absolute `1.4603137969970703e-6`.  The discrete split is entirely at the top:
RELION selects mapped key `(20057, 8)` while RECOVAR selects `(25798, 0)`.
RELION scores both hypotheses at exactly `0.2807506024837494`; no candidate
scores higher and the exact top-score tie count is two.  RECOVAR scores the
same pair at `0.2807507812976837` and `0.28075096011161804`, respectively,
only `1.7881393432617188e-7` apart.

Thus the previously observed 150.7523-degree fine-subset routing flip is
caused by coarse float32 reduction resolving an exact RELION tie, not missing
candidate topology.  The evidence supports the existing bounded 128-lane
top-two re-reduction and stable lower-pose-ID tie resolution.  It supports no
unbounded scoring, projector, reconstruction, tau2, or posterior-threshold
change.  The official comparison JSON SHA-256 is
`2e3368c5c03db4d0eea9519c746be6c4d4b26f8b8b0f11e98420ee6d878ebcdd`;
the durable result is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case04_it1_p6322_coarsegrid_7a4120c1_20260724T031102Z/provenance/RESULT_11562639.md`.

The frozen-fixture intervention is running from clean detached commit
`c74beea4` with the preserved real initial projector and
`RECOVAR_FIRSTITER_CC_TREE_TOP2_RESCORE_MAX_MARGIN=4e-6`: setup `11563826`,
science `11563827`, matrix summary `11563828`, and unchanged strict
FSC/topology audit `11563842`.  Grid correction and forced final-all-data
after non-convergence remain unset.  Until those audits pass, snapshot
`strict-k1-v6-20260724` remains 25/34 strict, 31/34 exact topology, and
34/34 evaluated.

### Case-5 frozen-fixture generalization arm

The case-4 intervention is also being tested unchanged against frozen case 5.
The accepted baseline particle audit contains only three first-iteration
assignment exceptions, original indices `26055`, `93729`, and `95412`; all
three are translation-only at the reporting tolerance.  No case-5-derived
code, margin, or tie rule was added.

Clean detached commit `c74beea4` and frozen manifest SHA-256
`422a79a0a7703d92f9777266e8c34ccd3a7cf5963b354e57a7d9a18f227babee`
are bound by setup `11564052`, science `11564053`, matrix summary `11564054`,
and unchanged strict FSC/topology audit `11564062`.  Science runs on physical
H100 `GPU-49c1a223-be61-858b-49d8-d8b0347ac252`.  Grid correction and forced
final-all-data after non-convergence remain unset.  This is a post-snapshot
generalization arm and does not change the frozen 25/34 strict,
31/34-topology score while pending.

### K=4 recovery is diagnostic-only after physical-GPU mismatch

Read-only provenance inspection invalidated the formal acceptance graph for
K=4 recovery job `11561204`.  The already accepted RELION control/capture and
superseded RECOVAR run used physical A100
`GPU-803dc869-2e74-273c-1df4-08adbc94e1b3`; the RECOVAR-only recovery uses
`GPU-6ec3d0a5-efc4-2f4c-fa73-7d76b911a412`.  Its audit launchers checked the
RECOVAR runtime UUID against only its own preflight and omitted equality with
the reused RELION UUID.  Pending primary/scalar jobs `11561345` and
`11561350` were canceled at zero runtime before they could emit a false
same-physical-GPU acceptance marker.

The mismatch is scientifically visible before the target capture.  At
numbered iteration 8, the eight superseded-versus-recovery half/class maps
have shellwise FSC-AUC `0.998882600598`--`0.999075807124`.  Fine and coarse
hard assignments differ for `705/100000` and `366/100000` particles,
respectively; noise relative L2 is `4.18455764e-6` and tau2 relative L2 is
`1.71844644e-4`.  Rotations, translations, and iteration metadata are exact.
Science `11561204` remains useful as a cross-A100 diagnostic and continues
toward the capture, but neither it nor its canceled auditors is eligible for
formal K=4 parity acceptance.  A replacement must run RELION control, RELION
passive capture, and RECOVAR sequentially in one allocation and assert all
runtime UUIDs are identical before comparison.

Durable invalidation note:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_fusedcapture_74f89c60_20260724T014000Z/provenance/RECOVERY_UUID_INVALIDATION_20260724.md`.

The corrected all-in-one replacement uses science `11564419`, primary vector
audit `11564442`, and independent scalar audit `11564443`.  It runs all three
science arms sequentially on physical A100
`GPU-6f45f415-9d0b-d562-9ff3-c9fb7bc53aa7`, uses the absolute image-identity
array, and asserts the control/capture/RECOVAR walltime UUID triplet before
capture acceptance.  The primary audit independently repeats that triplet
gate.  Launcher SHA-256 values are
`2650470674f3133ba9848e3a4515b9b054ebe7fa9cab633e8e5f10e4c7a091e6`,
`c369f9ed4992368abc01e5a0761deae90efcace53e2870f01d3906d362d85cfe`,
and `3d209caa65b1286129651b6b30a0eccc24ee399ac277d6f683091f1348cf5ed1`.
Formal K=4 acceptance remains pending all three successful exits.

## 2026-07-25 seed-exact K=4 factor-panel localization

The exact K=4 boundary is now decomposed at production factor level.  RELION
commit `a9ae8d2dd24704d7de52940fbc832fab1029a268` restricts expensive passive
term capture to accepted hypotheses without changing the full candidate
metadata table. Build job `11587833` and paired control/capture job `11587967`
completed both science arms and produced all 17 expected mixed-rank files.
The parent wrapper failed only an over-strict whole-data-STAR byte comparison:
identities, order, poses, origins, and class assignments are exact, while
repeat-scale diagnostics differ within the normal rerun envelope.  The formal
map contract accepts all four class maps at FSC-AUC
`0.999999992492`--`0.999999995085`.

A100 comparison `11590986` completed `0:0` from RECOVAR commit
`0f6356803166b9f9c0e7e17bf1e7af4d39fd3768`.  It covers the fixed 17-particle
panel, all 25 matched class-2 contributor rotations, and all 53 accepted
hypotheses. RELION's column-major matrix records transpose to bitwise-exact
RECOVAR matrices, and every accepted translation maps exactly.  The
dataset-native RECOVAR replay closes against captured `active_summed` at
relative L2 `7.34e-8`--`8.29e-8`, independently validating the factorization.

The scale-aware aggregate residual table is:

| Factor | Relative L2 over RELION | Maximum absolute difference |
|---|---:|---:|
| CTF | `2.8273123e-7` | `1.8626451e-6` |
| inverse noise | `3.1549497e-8` | `7.2759576e-12` |
| translation increment | `3.1649236e-8` | `3.7252903e-9` |
| posterior | `8.2827810e-5` | `8.7499619e-5` |
| shifted image | `0.0069581721` | `7.9328102` |
| weighted CTF | `0.0064766074` | `7.4348645e-7` |
| complex term | `8.4171546e-5` | `4.3499943e-6` |
| real weight term | `8.4192179e-5` | `5.8716978e-9` |
| contributor source sum | `4.2365267e-5` | `3.5632942e-6` |

The approximately `0.0065` standalone image/weighted-CTF residuals are
opposing per-particle normalization/correction placements and mostly cancel
in the complex product.  Geometry, support, translation phase, CTF, inverse
noise, and scatter are therefore closed below the material term residual.
Posterior arithmetic is the leading shared residual because posterior,
complex-term, and real-weight relative L2 all agree near `8.4e-5`.  The next
causal discriminator is a RELION-posterior counterfactual over the same 53
hypotheses; no production correction is justified before it.

Evidence:

- factor validation:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_seedexact_factor_capture_a9ae8d2_20260724T224000ET/analysis/FACTOR_CAPTURE_VALIDATION_POSTHOC.json`
  (SHA-256
  `0833e750bf9109d3cbe7881477143e6a622bd8714c9b20b448dd75612603fd7b`);
- formal inertness:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_seedexact_factor_capture_a9ae8d2_20260724T224000ET/analysis/RELION_CAPTURE_INERTNESS_POSTHOC.json`
  (SHA-256
  `365e85fa249defb07b05f5676462cd4d83811aae59c6b95a585dbfa49ee29fe6`);
- factor comparison:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_seedexact_factor_capture_a9ae8d2_20260724T224000ET/analysis/K4_RELION_RECOVAR_FACTOR_COMPARISON.json`
  (SHA-256
  `e70f404a25c4a43fc768d12a6ee507a61ab9d39e348f527d6d1caffbbe1d590a`).

This K=4 result does not promote a frozen K=1 case.
`strict-k1-v6-20260724` remains 25/34 strict, 31/34 exact topology, and 34/34
evaluated.

### Posterior-only counterfactual closes the K=4 factor residual

The predeclared next discriminator completed as A100 job `11591141` from
commit `f52a8bfc76657b42ef6ec61e219028b57774c018`.  It preserves all RECOVAR
image, CTF, inverse-noise, translation-phase, correction, and scale operands,
but substitutes RELION's captured posterior on the exact same 53 accepted
hypotheses.

Complex-term relative L2 falls from `8.4171546e-5` to `3.6279787e-7`, real
weight-term relative L2 from `8.4192179e-5` to `3.4731528e-7`, and
contributor-source relative L2 from `4.2365267e-5` to `3.6284445e-7`.
Those changes remove `0.99998142`, `0.99998298`, and `0.99992665` of residual
energy, respectively. The remaining `~3.5e-7` floor is consistent with the
already measured CTF/noise/factor arithmetic envelope.

This is causal localization: posterior construction, not factor placement,
geometry, support, or scatter, explains essentially all of the remaining
exact-boundary K=4 term residual. The next discriminator is posterior
reconstruction from captured RELION hypothesis scores/normalizers and
RECOVAR candidate score/log-normalizer operands.

The counterfactual report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_seedexact_factor_capture_a9ae8d2_20260724T224000ET/analysis/K4_RELION_RECOVAR_POSTERIOR_COUNTERFACTUAL.json`
(SHA-256
`e526fdb5b49f4675393b65512864f772be88580a37f1c1a25a8e08b0621d68d4`);
the completion marker SHA-256 is
`138b8490bff01a4233379b2dbe52418fc47c8ad26c3890f18b8035a7f5bdff5d`.
The frozen K=1 score remains unchanged.

### Posterior numerator and normalizer both begin at the score boundary

Job `11591351` decomposes the causal posterior residual in RELION's
exp(50)-shifted frame.  Across the 53 exact accepted hypotheses, raw weight
relative L2 is `1.0194764e-4`.  Across the 17 particles, the all-support
weight-normalizer relative L2 is `7.3824062e-5`.  These combine into the
measured normalized-posterior relative L2 `8.2827810e-5`; the discrepancy is
not produced only by final division.

Taking logs removes the exponential scale. RELION raw-log-weight versus
RECOVAR captured `combined_score - best_score + 50` has median absolute
difference `2.4406874e-4`, p95 `4.8831519e-4`, and maximum
`4.8834586e-4`.  The residual is therefore already present in the shifted
fine-score argument, well above the float32 exp/log round-trip floor.

The next required evidence is a passive RELION capture of pre-exponent fine
`diff2`, orientation prior, and translation prior on this exact panel.
Comparing those fields with RECOVAR's already captured preprior and combined
scores will separate score operands from reduction/rounding.  No
divide-only, exponential, significance-threshold, or factor-placement patch
is supported.

The posterior score-decomposition report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_seedexact_factor_capture_a9ae8d2_20260724T224000ET/analysis/K4_RELION_RECOVAR_POSTERIOR_SCORE_DECOMPOSITION.json`
(SHA-256
`33a6a98d17f3c84ff55c406d4ab49c8d5c337189aa24d668ed14121fccbfea61`);
completion marker SHA-256 is
`f706b25d226e69ccaae2c8f1831f49329eac25742473659452af706d0ba37912`.

### Passive fine-score capture localizes K=4 to candidate-varying data terms

RELION source commit `05398d236147eb71ce7fbbb60c635f2e8c012746`
adds a bounded passive selected-stack capture immediately before
exponentiation. Build `11591782` completed `0:0`; paired control/capture
science `11591945` ran both arms sequentially on physical A100
`GPU-ed3fe7be-abe7-7c79-06da-bc76e74d6025`, taking 1,437 and 1,087 seconds
and sealing all 17 factor plus 17 fine-score sidecars. Its wrapper's `1:0`
exit came only from a superseded assertion that every active hypothesis must
have positive post-exponent weight. RELION production intentionally clamps
shifted float32 scores below `-88` to zero.

The corrected fail-closed validator accepts 46,208/46,208 active candidates,
including exactly 43,842 production-clamp underflows. Pre-exponent and shifted
score algebra close with zero maximum absolute error, while all non-underflow
weights reproduce expf within `2.384185791015625e-07` relative error. Post-hoc
job `11593544` completed `0:0`; control and capture dispatch/particle fields
are exact and the four final class-map FSC-AUC values are
`0.999999992596`--`0.999999995235`. The captured post-exponent fine weight is
bitwise identical to the downstream factor weight on all 108 matched active
hypotheses, eliminating any intervening normalizer/exponent/factor boundary.

The 17-particle, 25-rotation exact panel separates the centered combined-score
residual into data and prior terms:

| Component | Relative L2 | Median abs | p95 abs | Max abs |
|---|---:|---:|---:|---:|
| combined score | `1.2579036e-5` | `2.4414063e-4` | `4.8828125e-4` | `4.8828125e-4` |
| data score | `1.7222145e-5` | `1.9642711e-4` | `5.2545071e-4` | `6.0272217e-4` |
| orientation prior | `7.9367489e-7` | `4.2915344e-6` | `4.2915344e-6` | `4.2915344e-6` |
| translation prior | `5.5783039e-8` | `0` | `4.7683716e-7` | `4.7683716e-7` |

Substituting only RELION's data component removes `0.999735251703` of
combined-score residual energy; orientation and translation priors remove
only `0.005022408445` and `-0.000179557013`. Follow-up job `11593681`
completed `0:0` and rejects a per-particle normalization-only explanation:
subtracting each particle's best scalar offset reduces data-residual L2 from
`0.002692778263` to `0.001808401433`, removing only
`0.548987582535` of its energy. The surviving residual is candidate-varying,
with median/p95/maximum absolute values `0.000152826309`,
`0.000304941088`, and `0.000335693359`.

This moves the supported causal boundary to RELION-versus-RECOVAR fine
projection/residual operands or their per-pixel reduction. Priors, shifted
normalization, expf, posterior division, significance support, factor
placement, and any per-particle score offset are now excluded as primary
causes. The next bounded experiment should capture a matched candidate's
projected reference and per-pixel diff2 terms before reduction.

Evidence root:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it10_finescore_capture_05398d2_20260725T043428Z`.
Validation, inertness, component-decomposition, and score-shape SHA-256 values
are `a4778489664f5d67aff151f3a6f72b3c38764d91febbedea975d67321de91a06`,
`9f0ab46b7ffbd63061c7c41f2fde3fe3daceb15cff80ec1d78201ec7727954cf`,
`591e5ddfa4ed0c725cc18fa7d7ecc17ea9eef79de893a4866b166b2d8304f834`,
and `4502b45fff0b04232d37e67df0bfbe2b7646f09c027674cff8abf2df446bd9bd`.
This diagnostic does not alter the frozen K=1 score.

### Exact-128-add case-4 intervention is rejected

Frozen case-4 science `11579503` completed `0:0` from clean commit
`161cb18f8989d8e83320d539d35a12f597d32ea6` on H100 `della-h20g5`.
Summary `11579504` completed `0:0`; strict audit `11579539` correctly
returned `1:0`. The numbered trajectories remain close through iteration 17
(merged cross-engine FSC-AUC `0.999664883`), but final merged cross-engine
FSC-AUC is `0.992294244`, below the fixed `0.995` threshold. The topology
audit independently reports iteration-15 `current_size` 154 for RELION versus
156 for RECOVAR.

The rejection does not indicate poorer recovered science. Final merged GT
FSC-AUC is `0.352136260` for RECOVAR versus `0.348384999` for RELION, a
`+0.003751261` RECOVAR delta. RECOVAR takes `7,969.70` seconds versus
RELION's `16,053` seconds, or `2.0143x` faster. Compared with the prior
bounded-tree arm's final cross-engine `0.992965912`, exact 128-add/tie
handling changes the result by `-0.000671668`; it is rejected as a production
parity fix. RECOVAR records convergence at iteration 17, so its final all-data
step is convergence-valid; grid correction and forced after-max finalization
remain unset.

The FSC and topology reports are under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fixedsuite_case04_atomic_161cb18f_20260724T231500Z`
with SHA-256 values
`56994ba7e843b0245ca31671d64a60f6fc4ab747d150d6542bfe809ec79f733f`
and `df1aad317d46e15d79a8ece0413cdfb2e533a69e4426e0f14ec5360705667ef1`.
The frozen score remains 25/34 strict, 31/34 exact topology, and 34/34
evaluated.

### Case-5 fine-pass exceptions reduce to one exact tie

Same-physical-H100 capture `11572658` completed `0:0` in 13:53 on
`GPU-0d7b80c7-fef8-e346-6332-de36ae1af518`, the physical GPU required by
the immutable case-5 trajectory.  Both outstanding RELION fine candidates
are present in RECOVAR's support.  Particle `38594` selects the RELION
candidate exactly, with native float32 top-two margin `1.4901161e-8`.
Particle `65070` instead has an exact float32 tie between flat candidates
`332` and `333`; RELION selects `333`, whereas RECOVAR's first-index
`argmax` selects `332`.  Both candidates use rotation index `2` and differ
by one fine translation step (`1.0624999` Angstrom), exactly reproducing the
remaining reported pose exception.

This rejects missing candidate support and a global score-margin threshold
for both case-5 targets.  Since the capture contains RECOVAR scores but not
RELION's two raw fine costs or compact-candidate order, the remaining causal
branch is fine-score arithmetic versus fine-candidate tie ordering.  A passive
RELION capture must separate those possibilities before any production
change.
Fine-summary JSON SHA-256 is
`019d3111c6eda111080bd2e87a81832971d4128535f2a3718bb7352fd452897f`;
the two captured panel SHA-256 values are
`c024a27a8b2f8071a1015e845ed28a938e6d7b3ece309a8789d07b702fddbeb6`
and
`f4e57638c96361f1040374827342a97866b802276810dca61b2ba21f16bee18d`.
The evidence root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case05_finepass_targets_20260724T143000Z`.
Exact-physical-H100 RELION discriminator `11602720` is submitted to dump the
two raw pass-2 costs and compact indices.  Its run root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case05_p65070_exact_relion_fine_20260725T083000ET`;
the fail-closed launcher SHA-256 is
`94db8675962d37e1ab28cb2a20a95605bec7a31682fedaa4c118aa7d43cbc4b8`.
It additionally requires exact identity-aligned stock-versus-dump-enabled
RELION iteration-1 poses, translations, class, Pmax, and support counts for
all 100,000 particles, then maps the eight-rotation fine panels by Euler
matrices.  Superseded pending jobs `11602588` and `11602654` were cancelled
before execution.
This diagnostic does not change the frozen K=1 score.

## 2026-07-26 full K=4 backend trajectory accepts `relion_cuda`

Same-physical-A100 science job `11600592` completed both 15-iteration,
100,000-particle, grid-256 K=4 arms from clean detached commit
`4181d340997e548af36c6458cce825e133dba95a`.  Both arms preserve the exact
RELION dispatch, schedule, convergence, and finalization topology.  Grid
correction and forced final all-data after non-convergence remained unset.

At the fixed direct per-class FSC-AUC gate of `0.995`, `host_numpy` passes
40/60 checks and `relion_cuda` passes 41/60.  Both pass all four classes in
9/15 iterations.  The fixed vectors are
`[4,4,4,4,4,4,4,4,4,3,0,1,0,0,0]` and
`[4,4,4,4,4,4,4,4,4,3,0,2,0,0,0]`, respectively.  The candidate improves
minimum cross-engine FSC-AUC from `0.989158631903` to `0.990091127730`,
minimum GT FSC-AUC delta from `-0.000409355343` to `-0.000352907281`, and
minimum RELION class agreement from `0.99175` to `0.99245`.  Direct
host-versus-candidate class agreement remains at least `0.99413`.
Correlation was not computed or used.

Same-GPU wall time falls from 30,089 to 26,921 seconds, a 3,168-second or
10.5288% reduction.  This accepts `relion_cuda` as the improved checked K=4
backend snapshot; it does not silently change the global K=1/K=4 default,
because the completed experiment is K=4-specific and the shared default also
governs K=1 and non-CUDA audit paths.

Audit job `11600593` finished every scientific report but returned `1:0`
after its checksum manifest recorded its own temporary pathname.  The
scientific reports were not rerun.  A pinned repair generated the temporary
manifest outside the analysis tree, excluded both manifest names, and
verified all 24 entries.  The repaired manifest SHA-256 is
`b5c45ccad205f271a91c0d1fe2a7f068e5674f2833bf26686a55f3dea815099b`;
repair provenance SHA-256 is
`859e36235bff8c8df3b5688a47165242c5f96dfbc7c560e8913edebff0e5a9f3`.

The accepted comparison is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_full15_host_relioncuda_samegpu_4181d340_20260725T051500ET/analysis/HOST_VS_RELION_CUDA_FIXED_SCORE_COMPARISON.json`
(SHA-256
`bca250d659c2ccbf5dc752cb876ecf35efe34447d03bd12850f92be86fc1cedd`).
Checked snapshot `docs/math/em_k4_backend_trajectory_snapshot_v2.json`
preserves the old 40/60 baseline separately and fixes the new 41/60
denominator for future work.

The active K=1 hypothesis is now limited to fixed case 3: completed canonical
science `11587631` should reproduce the already-passing older strict case-3
audit on the newer clean commit `4c8b043a`.  Read-only strict audit
`11632847` is the cheapest disproof and must pass both complete FSC trajectory
and exact intermediate topology before snapshot v7 advances from 25/34.

## 2026-07-26 canonical case 3 advances K=1 to 26/34

Canonical fixed-fixture science `11587631` and read-only strict audit
`11632847` completed `0:0` from clean detached source
`4c8b043a9b80ff12441e36f5a77c6e9f1896197b`.  The unchanged auditors pass
all 17 numbered FSC/FSC-AUC rows and exact intermediate topology.  Worst
numbered merged cross-engine FSC-AUC is `0.9999619013267681` at iteration
10; final merged cross-engine FSC-AUC is `0.9987827326111832`, and the final
RECOVAR-minus-RELION GT FSC-AUC delta is `+0.0054263318347904654`.
The run converged at iteration 17, ran final all-data only after convergence,
and kept grid correction off.

The canonical audit report SHA-256 values are
`0c5b3eccf9324b8c6aece1dcba3f920e49ef0da05eafa074fcc9124bf72fa2de`
for FSC,
`7e47bb0cdb3e488fcbc72cdcba9df7673989ed7cf5bc095238e4e6eddd72dbd7`
for topology, and
`b8358785fd84ff970b4cd4f97483cf98e93f35a6a01d6906abadc0841f59e2bc`
for shellwise evidence.

Fail-closed proposal job `11633116` rejected the launcher stdout name because
it lacked the literal frozen ID `k1-03`; it created no ledger.  Byte-identical
hard-link aliases supplied the validator-required name while preserving the
original stdout/stderr inodes and hashes.  Replacement proposal `11633309`
completed `0:0` after re-hashing the full 470,170,958,467-byte fixture suite.
The accepted v7 ledger is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_scorecard_v7_case03_d27d397c_20260726T114800ET/proposal/em_k1_gui_grid0_local_highshell_full34_superseding_ledger_v7.json`
with SHA-256
`55fb5042a3768c5d44b89aef72412682c6ebad2d832ba3c2a1b02a6a491c7d8e`.

Snapshot `strict-k1-v7-20260726` now records 26/34 strict FSC/FSC-AUC passes,
32/34 exact-topology passes, and 34/34 evaluated.  The denominator, case
definitions, fixture manifest, and acceptance thresholds remain unchanged.
The next causal K=1 discriminator remains exact-H100 job `11602720`; it does
not alter the score until a complete fixed-fixture trajectory passes.

## 2026-07-26 exact case-5 discrepancy classifier queued

Commit `6c483fba9f779533d169a65d67b867b90a443235` adds a fail-closed
classifier for the two possible outcomes of exact-H100 case-5 discriminator
`11602720`.  It accepts only Euler-matrix-matched particle `65070` at current
size `56`, plus an exact eight-field stock-versus-dump-enabled RELION
inertness report for all `100000` particles.  It compares the two engines'
raw pre-prior values with exact equality: two exact ties localize the
different winner to compact-candidate enumeration/tie order; any non-tie
localizes it to fine-score arithmetic.  No tolerance, correlation, FSC
surrogate, or scorecard acceptance is part of this classification.

The classifier and its unit test have SHA-256 values
`9ca93a25c2b795bc10384bd664d4c3ca30a366e66b752c7734c687969971e976`
and
`e1582846e8078b29d292ecc7777ffdd00b51ccfc1ba4b1105b3ddbb7744a5442`.
All 31 classifier/comparator targeted tests pass, and scoped Ruff passes.
Dependent CPU audit `11633508` is queued with
`afterok:11602720`; its launcher SHA-256 is
`22b5f65320d797bf42d461411a0649adce06427ae1c97c3b22f2ab1827b1fd19`.
It re-verifies both science SHA-256 manifests before writing the causal
classification.  This diagnostic does not change frozen K=1 snapshot
`strict-k1-v7-20260726` (`26/34` strict, `32/34` exact topology,
`34/34` evaluated).

## 2026-07-26 current-head fixed case 32 submitted

The remaining strict failures were ranked against the immutable scorecard.
Case 24 is numerically closest but its already-audited iteration-3 RELION
winner flips across adjacent installed-stock launches by one float32 ULP, so
forcing that outcome would overfit an unstable oracle.  Fixed case 32 is the
cheapest independent current-head check that has not been rerun after the
accumulated parity fixes: 10,000 particles, grid 128, exact old topology, and
old final merged cross-engine FSC-AUC `0.97450050098333`.

Clean detached source
`a03c9fd1359c47e69f16904935e6cb755d078b18` submitted exact-fixture setup
`11633606`, same-physical-A100 autonomous science `11633607`, and summary
`11633608`.  Unchanged strict FSC/topology audit `11633676` is queued with
`afterok:11633607`; it uses the same accepted auditor hashes as canonical
case 3 and writes a completion seal only when both pass.  The run root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_fixedsuite_case32_current_a03c9fd1_20260726T124000ET`;
the corresponding runtime root is under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/`, and both contain
`SAFE_TO_DELETE`.  Setup/science/summary launcher SHA-256 values are
`ea333ecbb1f672a528a9b6222b93381cc8ac4a12404c217b027718934bee003c`,
`02ffc2b7b2c58fe51e38f34d073785247b72a2df69ed4b9cf6c73237880ce74c`,
and
`473d05b6e38d5d4f8685269277833876da2762ddefc4703aa9aa022add8407de`.
Strict-audit launcher SHA-256 is
`0f98d966de8eeb8fc66ed062d5da32939b31232007c915501ee516e804738718`.

Scorecard mode pins fixture-manifest SHA-256
`422a79a0a7703d92f9777266e8c34ccd3a7cf5963b354e57a7d9a18f227babee`,
autonomous pairing, per-iteration maps, same physical GPU, grid correction
off, no replayed final numbered state, and no forced finalization or
diagnostic scoring/support overrides.  This launch does not change the
checked 26/34 score; only a complete unchanged FSC/FSC-AUC plus exact-topology
audit can promote it.
