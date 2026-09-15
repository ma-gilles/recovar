# Shared fixed-capacity whole-local executor plan (2026-08-31)

## Scope and invariant

This is a performance-only, default-off candidate for the shared exact-local
EM engine. InitialModel/VDAM must enter it through the existing dense adapter;
there will be no VDAM-only executor or duplicate numerical implementation.

The candidate may change floating-point reduction timing/order only if the
difference is repeat-bounded, non-growing, non-directional numerical noise and
the trajectory stays in the same basin. It may not change candidate membership,
within-image candidate order, outer-call chronology, physical particle order,
radix buckets, discrete decisions, convergence, or mathematical formulas.

The candidate is worth a trajectory gate only if a same-H100 fixed-state ABBA
shows a material end-to-end gain (target at least 20% for the local/expectation
boundary or at least 10% end to end). The implementation remains off otherwise.

## Why this is a different experiment

The existing evidence rules out repeating five narrower interventions:

1. **Physical-order smaller chunks.** At the frozen GF46 iteration-181 state,
   `--exact-local-physical-order-chunk-size 220` reduced warm wall time from
   `14.7141 s` to `13.9356 s` (about 5.3%). It still made 10 Python/JAX bucket
   calls and moved preprocessing, posterior, noise, and M-step values across
   each outer boundary. This proposal keeps the authoritative calls but places
   their complete chronology behind one fixed-shape outer executor boundary.
2. **Literal pool-of-three layout.** The sealed GF46 layout audit shows that
   literal pools reduce padded rotation rows by 49.5--82.2%, but create
   67--120 calls instead of 3--7. Even coalescing equal adjacent pool runs leaves
   25--35 calls, 3.86--8.75 times the current call count. This proposal does not
   replace existing calls with literal pools; pool evidence only motivates a
   packed representation that can retain low logical work without paying one
   host boundary per pool.
3. **Tail/batch palettes in one stage.** Quantizing only significance/coarse
   image batches does not stabilize the downstream fine, posterior, noise, and
   x-half shapes. The whole-local plan carries fixed image, candidate-row, and
   call capacities through every stage. `valid_images` is runtime data.
4. **Runtime cutoff in BPref/window primitives.** The stable BPref primitive
   produced one signature and repeat-scale differences, but its measured
   runtime path was 14.8--17.7% slower than the static primitive. This proposal
   does not add another BPref-only runtime kernel. The logical cutoff is one
   runtime operand of the complete executor, so compile reuse can amortize over
   preprocessing, scoring, posterior, noise, and M-step together.
5. **Literal BPref arithmetic variants.** The literal arithmetic probe already
   found the best numerator differences at roughly `3e-8` relative L2 while the
   denominator was exact. No arithmetic expression is changed here. The shared
   qualified preprocessing, fine scorer, posterior, noise, and x-half routines
   remain authoritative.

Evidence roots:

- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_late_it181_profile_baseline_ead78d32e_h21g4_20260831`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_late_it181_profile_chunk_only_f61808a0e_h21g4_20260831`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf46_pool_layout_capture_it80_104dac4ef_20260831/analysis/pool_layout.json`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_stable_bpref_primitive_d366_9986787_20260831/report.json`
- `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_bpref_literal_20260810/LITERAL_BPREF_PROBE.json`

## Data model

The authoritative input remains the ordered `LocalBucketSpec` sequence produced
by `bucket_local_hypothesis_layout`. The first host seam converts each existing
call to `_FixedCapacityLocalCall` without regrouping it:

- `image_indices`: real particles in the exact existing call order;
- `row_counts`: real candidate rows for each particle;
- `radix_bucket`: the existing rectangular rotation capacity.

`_plan_fixed_capacity_whole_local` requires an explicit sealed physical image
order and explicit capacities. It fails closed on a chronology mismatch,
duplicate particle, unsupported radix, row count outside its radix, palette
overflow, or image/row/call/cutoff overflow. It emits fixed-shape arrays:

- packed physical image IDs plus a fixed-capacity CSR `row_offsets` vector;
- chronological call image/row offsets;
- runtime valid image and candidate-row counts per call;
- the original radix bucket per call;
- the selected stable image capacity for each full/tail call;
- a runtime logical Fourier cutoff bounded by a fixed physical cutoff;
- an inert call tail selected by `call_valid_mask`.

The candidate payload packer removes only rectangular invalid radix rows. It
copies each image's real candidate prefix in call/particle/candidate order and
poisons or zeros only the unused fixed-capacity tail. Raw images, CTFs, priors,
corrections, and particle metadata use the same packed physical image axis.

The image-capacity palette is an executor policy, not a new grouping policy.
For example, existing calls with 75 full images and a 25-image tail can all use
physical capacity 75; calls with 75, 100, and 150 images can use a fixed
`(75, 150)` palette. No call is split or merged to make it fit.

## Implementation phases

### Phase 0: host seam (this commit)

- Add the shared call/plan descriptors and generic image/candidate packers in
  `dense_single_volume/batch_planning.py`.
- Leave the seam uncalled by production and default `enabled=False`.
- Pin chronology, radix, full/tail shape stability, runtime cutoff, poison-tail,
  and fail-closed contracts with CPU unit tests.

### Phase 1: shared operand assembly

- Add a conversion from the already-built `LocalBucketSpec` list to the shared
  plan; the InitialModel adapter must only set the typed option.
- Build/fetch the raw image cache once in physical order using `local_caches.py`.
- Pack raw images, CTF parameters, corrections, priors, normalization inputs,
  and candidate arrays once. Do not refetch or repack inside the call program.
- Admit only the exact K=1 RELION x-half topology first. Reject diagnostics,
  external replay, unsupported projector paths, K>1, or noncanonical options.

### Phase 2: one shared whole-local numeric boundary

- Add `run_fixed_capacity_whole_local` beside `run_local_bucket_big_jit`, not in
  `initial_model/`.
- Carry donated `Ft_y`, `Ft_ctf`, noise, scale, and posterior-summary state
  through a chronological call loop.
- Dispatch existing radix-specific scratch tiles from the descriptor program.
  All branch inputs/outputs use fixed physical capacities; `valid_images`, CSR
  row offsets, and logical cutoff are runtime arrays.
- Reuse the current RELION CUDA preprocessing, exact fine-diff2, float32
  posterior, noise, Wavg, and x-half accumulation primitives unchanged.
- Scatter per-image metadata once after the whole boundary. Keep the BPref
  accumulation order identical to the input call/particle/candidate chronology.

### Phase 3: bounded qualification

1. CPU/source contracts and the directly affected EM fast-guard slice.
2. One/few-particle CPU and H100 fixed-state replay with poison tails.
3. Full-versus-tail calls with identical physical shapes; verify compile/HLO
   identity and exact discrete outputs.
4. Same-checkpoint crossed ABBA on the pinned H100. Record wall, expectation,
   pass 1/pass 2/local, compile count/time, kernel sum/union, idle gaps, memory,
   and allocation count.
5. Repeat twice per arm. Numeric acceptance requires cross-arm map/state error
   no larger than the repeat envelope, signed mean/RMS without directional
   drift, exact candidate support and discrete state, and no error growth.
6. Only after the material runtime gate passes, run one representative K=1
   trajectory/repeat gate. No broad RECOVAR suite is part of this candidate.

## Stop conditions

Stop and retain the seam default-off if fixed capacities inflate summed GPU work
enough to erase the overlap/compile gain, if the outer program still creates
per-call host synchronization, if any unsupported topology falls back silently,
or if numerical differences exceed/reliably drift outside the repeat envelope.
