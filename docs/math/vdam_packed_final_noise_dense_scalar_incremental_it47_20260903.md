# VDAM dense-scalar / packed-final incremental report (2026-09-03)

## Decision

The default-off packed final-noise lane now passes the valid same-state direct
oracle gate and has a small but repeatable end-to-end speed benefit over the
already packed-deferred VDAM stack.  Across four prewarmed measurements per
backend, packed final noise reduces median pass 2 by 5.66%, local EM by 5.71%,
noise accumulation by 39.05%, and whole-transition wall time by 1.37%.
Separate ABBA and BAAB panels agree on the sign and size of the effect.

This accepts the implementation as a useful performance building block, not
as a production default.  The evidence is one K=1 same-state transition at
iteration 48; it is deliberately not a trajectory-quality promotion gate.
The lane remains disabled unless
`RECOVAR_INITIAL_MODEL_PACKED_FINAL_NOISE=1` is explicitly set.

## Immutable provenance

| Item | Value |
| --- | --- |
| Validation scope | Performance-only, one same-state K=1 VDAM transition |
| Source commit | `337dfaf7f95e73d430bf3bb572663284f279865b` |
| Branch | `codex/vdam-packed-final-noise-20260903` |
| Source status at launch | Clean; zero tracked or untracked changes |
| Slurm job | `13378581` (`COMPLETED`, `0:0`) |
| Node / GPU | `della-h19g1` / `GPU-0d7b80c7-fef8-e346-6332-de36ae1af518` (NVIDIA H100 80GB HBM3) |
| Checkpoint / transition | direct trajectory through 47; compare iteration 48 from the same in-memory checkpoint |
| Timed panels | packed-deferred / packed-final / packed-final / packed-deferred, then the reverse BAAB panel |
| Run root | `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_packed_final_noise_direct_oracle_incremental_it47_337dfaf7f_20260903T072442Z` |
| Science report SHA-256 | `93097ac4e2dad7373cf69bf08bb30eb16e1111d57ced95dcdcb6083956a53414` |
| Run manifest SHA-256 | `ad8ca1bec06072576fba960a212336280d4ae5e62e171e2bfe548fe6e5437f57` |
| Rebuilt CUDA SHA-256 | `a0ed236bde32c3ad4d74cc02db36b674c6276466a0bd4d8f6f03993b04fc212e` |
| Slurm elapsed | 9 minutes 13 seconds |
| Process maximum RSS | 8,372,420 KiB |
| Observed whole-job peak GPU allocation | 17,597 MiB; not separable by arm |

The run root contains `SAFE_TO_DELETE`, `COMPLETED`, `SHA256SUMS`, and
`SHA256SUMS.sha256`.  The latter verifies the immutable `SHA256SUMS` manifest.

## Valid oracle design

Two earlier measurements were not used for acceptance.  Job `13378155`
placed a nested `jax.jit` boundary around the shared scalar helper, changing
the mature graph rather than merely reusing it.  Job `13378392` removed that
boundary, but still tried to compare independently recomputed iteration-47
checkpoints.  VDAM's GPU atomic accumulation makes those long checkpoint
trajectories numerically non-identical, so cross-job manifest equality is not
a valid exact oracle.

The accepted gate fixes both issues:

- `local_big_jit.py`, including its dense scalar reductions, is byte-for-byte
  unchanged from the established `922abd1699` control;
- the VDAM helper is deliberately not separately JIT-compiled, and a focused
  test proves its JAXPR is the mature inline primitive sequence;
- one direct oracle arm, all packed-deferred arms, and all packed-final arms
  deep-copy the exact same in-memory model, particle, and sampling checkpoint;
- packed-deferred and packed-final are each prewarmed once before timing;
- four measurements per backend are ordered as ABBA followed by BAAB;
- the Slurm wrapper fails unless direct-to-control and direct-to-candidate
  hard, scalar, and stable public state are exact.

The direct oracle is intentionally excluded from timing: it is a cold science
control.  Only the eight prewarmed packed arms estimate the incremental cost of
the final-noise change.

## Implementation boundary

The packed-final path retains the full dense posterior only for inexpensive
scalar and per-image reductions.  Retained mass, translation posterior,
offset numerator, image power, padding mask, and public scalar normalization
therefore use exactly the mature dense shape and reduction order.  Pixel-heavy
projection, weighted-image, CTF, Wavg, normalization-residual, and scale terms
continue to operate on packed final-support rows.  The last padded batch slices
the dense per-image power vector before combining it with packed residual rows.

At the gated source commit, the helper lived in `local_backprojection.py` while
the mature big-JIT source remained unchanged.  The post-gate consolidation now
routes both mature EM and outer VDAM through that same plain/inlined helper,
with `jnp.sum(support_mass)` retained at its original big-JIT call position.
An independent JAXPR test compares the helper against the literal mature
expression and proves the primitive sequence is unchanged.  No EM
implementation was forked and the old dense path remains the default oracle.

## Work reduction

All compared arms selected the same 511 live final-support rows.  The candidate
uses 3,200 physical packed final-noise rows instead of the 55,296-row dense
layout.  It retains the 19,008-row packed score layout shared with the
packed-deferred baseline.

| Layout across four calls | Rows | Relative to packed final |
| --- | ---: | ---: |
| Dense final-noise layout | 55,296 | 17.28x |
| Packed scoring layout | 19,008 | 5.94x |
| Packed final-noise physical layout | 3,200 | 1.00x |
| Live final support | 511 | 0.160x |

Packed radix padding is still 6.26x above live support.  Removing that padding
is a real remaining optimization opportunity, although the now-0.065-second
noise stage limits its whole-iteration ceiling.

## Exact science state

The report contains 16 comparisons: six same-backend repeats, eight
packed-deferred/packed-final comparisons, and two direct-oracle comparisons.
Every one has:

- 13/13 tracked E-step hard/scalar fields bitwise exact, including selected
  IDs, poses, translations, Pmax, class/direction/support sums, retained mass,
  and offset numerator;
- the same support-audit SHA-256,
  `7615d7e4127073719bda630379eac31ec265a8a01f173c307bba11c8f8f30d71`;
- 9/9 particle-state and 22/22 sampling-state fields bitwise exact;
- 9/9 stable public outputs exact: `Mavg`, `ave_Pmax`, `pdf_class`,
  `pdf_direction`, `sigma2_offset`, `tau2_class`, FSC, current resolution, and
  current size.

The direct oracle and all eight timed arms have identical scalar values:

| Quantity | Exact value in all arms |
| --- | ---: |
| retained `noise_sumw` / `sigma2_offset_sumw` | 199.84414672851562 |
| `wsum_sigma2_offset` | 6277.32666015625 |
| halfset Pmax mean | 0.4354361295700073 |
| resulting `ave_Pmax` | 0.43577572182119084 |
| resulting `sigma2_offset` | 15.410241883989046 |

Large BPref and gradient arrays are not bitwise repeatable because the H100
backprojector uses atomic accumulation.  Direct-oracle and cross-backend
distances overlap the measured same-backend scale:

| Field | Same-backend rel-L2 range | Packed A/B range | Direct-oracle range |
| --- | ---: | ---: | ---: |
| half 0 BPref data | 7.935e-8 -- 8.543e-8 | 7.878e-8 -- 9.156e-8 | 8.022e-8 -- 8.281e-8 |
| half 0 BPref weight | 4.675e-8 -- 5.360e-8 | 4.728e-8 -- 5.426e-8 | 5.085e-8 -- 5.193e-8 |
| half 1 BPref data | 8.341e-8 -- 9.201e-8 | 8.625e-8 -- 9.226e-8 | 8.902e-8 -- 9.147e-8 |
| half 1 BPref weight | 4.392e-8 -- 4.871e-8 | 4.394e-8 -- 4.989e-8 | 4.835e-8 -- 4.978e-8 |
| final `Iref` | 3.880e-11 -- 1.119e-10 | 5.162e-11 -- 9.250e-11 | 5.899e-11 -- 9.553e-11 |
| final `Igrad2` | 6.025e-10 -- 6.609e-9 | 2.685e-9 -- 6.415e-9 | 2.706e-9 -- 3.073e-9 |

`sigma2_noise` is also numerically, not bitwise, equal.  Its direct-oracle
relative distances are `1.615e-9 -- 1.622e-9`; the stable downstream public
state listed above remains bitwise exact.  This is explicitly not claimed as
an exact large-array reduction.

## Incremental performance

The table compares medians of four prewarmed packed-deferred arms against four
prewarmed packed-final arms.  Both use flat scoring rows, packed local
projection, and deferred packed VDAM; the final-noise flag is the only backend
difference.

| Stage | Packed-deferred median (s) | Packed-final median (s) | Delta |
| --- | ---: | ---: | ---: |
| Whole transition | 5.563913 | 5.487616 | -1.37% |
| Expectation | 5.327230 | 5.257449 | -1.31% |
| Pass 1 | 3.825235 | 3.808996 | -0.43% |
| Pass 2 | 0.967750 | 0.913011 | -5.66% |
| Local EM | 0.955421 | 0.900852 | -5.71% |
| Accounted local EM | 0.915850 | 0.865395 | -5.51% |
| Big-JIT buckets | 0.684539 | 0.675877 | -1.27% |
| Final noise accumulation | 0.106550 | 0.064942 | -39.05% |
| Outer M-step/update | 0.173811 | 0.172877 | -0.54% |

The unchanged pass-1 component moves by only 16 ms in the aggregate median;
the candidate's 55 ms pass-2 saving survives as a 76 ms whole-transition
saving.  The mirrored panels give the same conclusion:

| Panel | Whole transition | Pass 2 | Final noise |
| --- | ---: | ---: | ---: |
| ABBA | -1.50% | -5.91% | -39.00% |
| BAAB | -1.26% | -5.54% | -38.99% |

The direct oracle took 10.553 seconds because its dense graph compiled in that
arm; neither that value nor the two explicit prewarm arms enters the timing
estimate.

## Focused validation

- Dense scalar value and inline-JAXPR guards: passed.
- Mirrored/prewarm/backend-scope and direct-oracle harness guards: passed.
- Packed VDAM merge guard and gated mature-big-JIT unchanged check: passed.
- Post-gate shared-helper refactor: literal-JAXPR equality and focused merge
  guard passed; no new GPU gate was needed because the compiled primitive
  sequence and retained-mass call position are unchanged.
- Final focused pytest slice: 11 passed, 42 deselected.
- Ruff on all changed Python files: passed.
- Python syntax, Slurm `bash -n`, and `git diff --check`: passed.
- No broad RECOVAR or full trajectory suite was run.

## Next performance discriminator

Keep this lane default-off while integrating it into the broader performance
stack.  A flat segmented final-noise reducer can attack the remaining
3,200-to-511 padding ratio, but the measured ceiling is now small.  The larger
runtime opportunity is pass 1 (about 3.81 of 5.49 seconds here), so subsequent
work should prioritize eliminating controller launches and increasing
matrix-matrix/GPU occupancy there, using the mature EM batching primitives
where their arithmetic contract matches VDAM.
