# EM / RELION Parity Contract

This file contains the program's quantitative gates and reproduction contracts.
The active cleanup milestone, current evidence and next check live in
[the EM status page](../development/em_status.md). Permanent development rules
are in [the EM guide](../../recovar/em/AGENTS.md).

The [complete historical program](https://github.com/ma-gilles/recovar-experiments/blob/8e43b06f8a43afc9a406bc9509899c338e7e4102/docs/math/em_parity_program_history_20260916.md)
is preserved verbatim in the private experiment archive, including failed and
rejected experiments. Its dated next actions and source-specific qualifications
are historical, not instructions for the current tip. Current contract sections
below are retained unchanged; archival does not waive any gate.

The serialized-versus-runtime scale section at the end remains here because
[replay overrides](../../recovar/em/diagnostics/relion_replay.py) reference it.
Detailed investigation notes live in [the parity notes](relion_parity_agent_notes.md),
and completion records in [best metrics](em_parity_best_metrics.md).

## Objective

First achieve near-perfect RELION quality parity for supplied-map K=1
auto-refine and K=4 3D classification. Then optimize to near RELION speed while
holding the accepted quality checkpoint. Treat native InitialModel/VDAM parity
as the next product milestone rather than mixing it into the first closure.

## Current VDAM quality priority — September 9

The user provisionally accepts up to **2× RELION runtime** while prioritizing
quality: first establish short-iteration matched-input score/posterior/state
parity, then evaluate final shellwise FSC/FSC-AUC against GT and RELION across
the required scope, including Hungarian-matched K4 and robustness. Once their
causes and numerical bounds are established, late discrete trajectory differences
are diagnostic; unexplained mismatches are not excused as noise. Exact parity
remains preferable. No numerical tolerance or baseline is changed, and the
long-term speed objective remains. This allowance is not a measurement of
current-source representative speed or a completed quality gate. See the
[user-policy handoff](/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/pr179_coordination/handoffs/vdam_quality_priority_20260909.json).

## Mode Contract

- **Production precision (user decision 2026-09-08):** float32 remains the
  intended EM execution path. Double precision is diagnostic only, used to
  distinguish roundoff from implementation bugs with matched inputs and state.
  It is not a substitute for a corrected and qualified float32 implementation.
  Final K1/K4 quality and performance evidence must use the production path;
  preserve deliberate higher-precision host/metadata operations. See the
  [mandatory precision policy](../../recovar/em/AGENTS.md).
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

The piecewise pack is RELION's MPI combine (`combineAllWeightedSums`, used with
`--dont_combine_weights_via_disc` by builds without `USE_MPI_COLLECTIVE`,
which CMake defines only for SYCL/ALTCPU). Without that flag RELION combines
through files with one full `MlWsumModel::pack(Mpack)`, which does reduce every
physical group. `relion_worker_scale.update_relion_follower_scales` therefore
takes the reduction mode that `relion_scale_reduction_mode_from_command` reads
from the oracle's recorded command line. The 2026-09-23 D6 symmetry audit found
the reconciled source hard-coded the file-combine rule; every
`--dont_combine_weights_via_disc` K=4 oracle then diverged from iteration 3,
with rank-1 scales after the iteration-2 M-step up to 0.498 away from RELION's
model STAR (final Q: 3.3e-6).

K=4 scale A/B evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it3_orig6388_runtime_scale_ab_h100_20260715_024500/runtime_scale/analysis/score_support_no_correlation.json`.
K=4 per-rank scale-state evidence:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_relion_scale_state_rank_audit_h100_20260715_034500/analysis/scale_rank_state_no_correlation.json`.

Further dated experiments are preserved in the complete historical program linked above.
