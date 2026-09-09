# Frozen F32 M full200 evidence — September 9, 2026

The explicit F32 M route completes this small fixture and satisfies all four
registered-GT deficit conditions through checkpoints 0–200. **Strict trajectory
parity fails:** all four cross-engine FSC-AUC histories miss the existing 0.999
condition. Whole-child RECOVAR/native time is **1.436880×**, worse than native.
This admits a reproducible diagnostic record, not a quality baseline, precision
default change or current-primary performance qualification.

## Source and experiment

Slurm **13644924** completed 0:0 in 25:17 on H100 `della-h19g1`, physical UUID
`GPU-9f98ccbf-3c62-c54f-7409-7eb58845ad4a`. Four fresh processes ran in order
native 1, candidate 1, candidate 2, native 2. GF43 contains 3,000 particles at
128×128, K=1, seed 29; each arm saved initialization and all 200 updates.

Candidate source is clean **`bae959dabe465dcb7e7d6d3aee586ed2039a9203`**;
producer manifest SHA-256 is
`f0231bbd45443bafc7e73705974522d3d488975ca5fbc6bf1c73a4d28ab7d88d`.
The native executable SHA-256 is
`6c54d2ac962da1e3eb6a6dee171a19a672e5833e9d412a23e0fdd87c600a1664`.
Its recorded source/build identity is the previously reviewed private residual
capture build; no native rebuild was made for this panel or integrator review.
Exact commands, environment, fixture, source and library manifests reside in
`vdam_f32_m_full200_20260909/` under the artifact base below.

The already integrated `bae959dab` route explicitly selects
`--mstep-backend jax --mstep-compute-dtype float32`. Scores/projector operands,
device M computation and six persistent M-owned fields use F32/C64. Bootstrap,
corrected refreshed projector/power, tau/noise/priors/normalization and host BPref
export retain higher-precision numerical work. Native records CPU double on,
accelerated double off. This is not full-algorithm F32 closure. Private product
candidate `907b02ce` is absent; historical `fe847` included it and an F64/C128 M
step. Comparing those panels cannot isolate the effect of M precision.

Primary already incorporates the capability, DC repair and CLI route via
`29d7e38a5` and `cb897cda5`, with additional structural changes. This frozen run
does not qualify that newer composition. Inherited F64 defaults remain unchanged.

## Map-quality measurements

One fresh native-1 final map supplied a proper rigid GT transform, frozen for
all 804 maps; no per-arm refit or reflection. Existing normalized non-DC FSC-AUC
is used. The prespecified conditions in the producer's `analysis_plan.json`
remain −0.002 for candidate-minus-native registered GT and 0.999 for raw
cross-engine FSC-AUC. These are this VDAM panel's conditions, not a replacement
for the separate supplied-map K1/K4 milestone contract.

| Registered GT comparison | Minimum delta | Checkpoint | Final delta | Condition |
| --- | ---: | ---: | ---: | --- |
| Candidate 1 − native 1 | −0.001244779 | 128 | −0.000598522 | Pass all 201 |
| Candidate 1 − native 2 | −0.000727339 | 46 | +0.000820235 | Pass all 201 |
| Candidate 2 − native 1 | −0.001266165 | 127 | −0.000460113 | Pass all 201 |
| Candidate 2 − native 2 | −0.000727622 | 46 | +0.000958644 | Pass all 201 |

Final GT direction is mixed: worse than native 1 and better than native 2.
Held-out shells above 8 are descriptive; they have no separate acceptance gate.

| Raw map-pair FSC-AUC | Minimum | Checkpoint | Final | First below 0.999 | Failed checkpoints |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native 1 / candidate 1 | 0.970018318 | 155 | 0.998870785 | 73 | 128 |
| Native 1 / candidate 2 | 0.970116432 | 155 | 0.998905551 | 73 | 128 |
| Candidate 1 / native 2 | 0.970238930 | 155 | 0.999142727 | 71 | 105 |
| Candidate 2 / native 2 | 0.970292892 | 155 | 0.999161414 | 71 | 105 |

Native-repeat minimum is 0.998337306 at 86; candidate-repeat minimum is
0.999800822 at 174. Repeat variation is context and does not waive any crossed
failure. The peer model audit reports saved size 74 for candidates versus 72
for natives at 155, preceded by shell-27 SSNR values on opposite sides of 1.
That association does not establish a causal explanation of the FSC loss.

## State limits

The separate peer CPU audits report 201 checkpoints, 1,206 particle-pair
comparisons, 402 valid candidate metadata schemas and 181 mismatches among
8,406 available integer checks. Cross-engine rotation accuracy first fails at
20: candidate 6.465° versus native 6.470°. Candidate repeats first disagree on
support count at 14 (particle index 108, 57 versus 58) and pose/translation at
76 (image `1128@particles.128.mrcs`). Native-repeat pose first differs at 23.
These are peer-reported serialized-state findings; this integrator review did
not rerun the particle/model analyzers.

Native selected IDs, hidden support/accumulator state and competing score
surfaces are absent. Same-input replay is required to classify discrete changes;
different-history Pmax gaps are not an arithmetic error bound. All 804 model
STARs publish zero `rlnGoldStandardFsc` values: their equality is not a map FSC
comparison. Strict state, tie, convergence/finalization and broader coverage
remain open, regardless of the registered-GT result.

## Timing and memory

| Arm | Whole child, seconds | Sampled peak GPU memory, MiB |
| --- | ---: | ---: |
| Native 1 | 304.146380 | 79,550 |
| Candidate 1 | 437.168392 | 17,534 |
| Candidate 2 | 437.705276 | 17,534 |
| Native 2 | 304.724010 | 79,550 |

The ratio of candidate/native geometric mean times is 1.4368803454593277.
Pair ratios are 1.437361812 and 1.436399040. Whole-child timing includes startup,
cold compilation/JIT and iteration I/O; output hashing is outside both engine
timers. Candidate import/library checks are included; its separately recorded
CLI times are 431.158071 and 431.646931 seconds. Process/JAX/CUDA caches are
private and fresh; filesystem/page cache was not flushed. Stage and compilation
totals were not separately instrumented, so this run cannot attribute the gap.

One-second memory samples have zero recorded sampling errors and are lower
bounds on actual peaks. Native allocation reservations prevent an intrinsic
memory-efficiency claim. The historical `fe847` 3k/128 ratio was 1.468982×;
the ratio improves descriptively by 2.18527%, with different source, precision,
GPU UUID and timing guards. This is not a causal M-step speedup or current
100k/256 measurement.

## Integrator review and reproduction

At clean primary `0fd89e23b2448a6a9d067a6804111539811e15a4`, em_clean independently
recomputed **1,608 GT curve AUCs, 1,206 raw pair AUCs, 1,608 held-out integrals**,
all extrema/failure lists and 201 iteration records. Every stored value matches
exactly. Four process/CLI timers, both pair ratios, the geometric ratio and all
sampled-memory maxima/counts reproduce the receipts. **87 named file hashes**
match before/after, including the producer manifest's 46 inputs, native binary,
CUDA/binding libraries, registration controls and analysis artifacts.

The producer's 7,578-entry before/after inventories compare equal as receipts;
this review did not rehash every map/source dependency or rerun E/M, FFT or GT
fitting. The peer's separate nine uncached canonical curve checks are retained
as peer evidence, not claimed as integrator recomputation. Registration controls
and a proper orthogonal transform were checked from saved inputs/receipts; their
success does not prove a uniquely correct absolute GT frame.

All five producer/analysis/timing/state roots have `SAFE_TO_DELETE` markers.
Preserve them while they supply active benchmark evidence.

Absolute artifact base:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/`.

- Producer commands/logs: `vdam_f32_m_full200_20260909/`.
- Map curves, transform and report: `vdam_f32_m_full200_analysis_20260909/`;
  arrays and per-iteration records are in `results_v1/`.
- Timers: `vdam_f32_m_full200_speed_20260909/speed_receipt.json`.
- Peer state/model reports: `vdam_f32_m_full200_state_20260909/RESULTS.md` and
  `vdam_f32_m_full200_model_state_20260909/RESULTS.md`.

Integrator audit, exact pins, CPU log and script:
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/hia_source_review_20260906/f32_m_full200_review_20260909/`.
Run `review.py` with the frozen pixi Python, contaminating Python/conda variables
unset, `CUDA_VISIBLE_DEVICES=''`, `JAX_PLATFORMS=cpu`, `PYTHONNOUSERSITE=1` and
BLAS/OMP/MKL threads 1. It reads peer artifacts only and refuses to overwrite its
existing `review.json`; use a fresh copy/output directory for another review.
No new GPU job, native build, source repair, test tolerance or baseline change.
Historical failures remain preserved; full-F32/current-primary, real-particle,
exact-K4 and representative 100k/256 gates are not closed by this fixture.
