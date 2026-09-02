# Error certificate for hybrid VDAM coarse GEMM scoring

Status: **design note; not a production qualification**.  The proposed
certificate turns expanded-square GEMM scores into a conservative selector for
exact direct-residual rescoring.  It does not justify accepting a GEMM winner
without direct rescoring.

The recommended center is the promoted-FP64 expanded score while it is still
live, before its one FP32 narrowing for the ordinary posterior path.  The
certificate then needs no cast-error term.  A promoted-FP64 center has also
measured enough standalone headroom to remain useful after selective direct
rescoring, but total hybrid runtime and science parity still require a gate.

The source anchors for this note are
`_relion_coarse_gaussian_gemm_scores_jit` in
`recovar/em/dense_single_volume/helpers/scoring.py`, `_add_priors` and the
coarse block loop in `helpers/significance.py`, and
`relion_fine_diff2_update_f32` plus
`relion_coarse_diff2_rectangular_f32_kernel` in
`recovar/cuda/cuda_backproject.cu`.

## Stored-input objective and sign

Fix one image, rotation, and translation.  Treat the values delivered to both
scorers as exact stored inputs:

- reference pixel `p_k = a_k + i b_k`, stored as complex64;
- corrected shifted-image pixel `y_k = c_k + i d_k`, stored as complex64;
- nonnegative pixel weight `w_k`, stored as float32; and
- nonnegative initial difference `d0`, stored as float32.

Use `n` below as a conservative upper bound on both the number of terms in an
energy dot and the full pixel positions traversed by the direct kernel.
Compact-pixel skips may reduce either count but must not exceed `n`.

The certificate is relative to these stored values, not to the unrounded
projector or preprocessing calculation that produced them.  Define

```text
A = sum_k w_k (a_k^2 + b_k^2)
C = sum_k w_k (c_k^2 + d_k^2)
X = sum_k w_k (a_k c_k + b_k d_k)
Q = d0 + 0.5 (A + C) - X
S = -Q
```

`Q` is the exact direct residual and is nonnegative.  `S` is the raw score used
by `significance.py`: larger is better.  The expanded-square scorer evaluates
`X - 0.5 A - 0.5 C - d0`; the direct CUDA scorer evaluates `Q` and returns its
negation.  This sign convention must be used consistently in all interval
code.

For cancellation bounds, also define

```text
H = sum_k w_k (abs(a_k c_k) + abs(b_k d_k))
```

No extra absolute-product GEMM is needed because weighted Cauchy--Schwarz gives

```text
H <= sqrt(A C) <= 0.5 (A + C).
```

The square-root form is tighter; the arithmetic-mean form is a cheap fallback.

## Floating-point notation

For precision `p`, let `u_p` be its unit roundoff and

```text
gamma_p(k) = k u_p / (1 - k u_p),       provided k u_p < 1.
```

Here `u_32 = 2^-24` and `u_64 = 2^-53`.  Every coefficient used by production
code must be generated with higher-precision arithmetic and rounded upward,
or stored as a reviewed upward-rounded constant.

The operation counts below assume:

1. complex GEMM's real component is a reduction of the `2 n` real products;
2. each real dot contribution is accumulated once with an IEEE FMA in the
   selected precision, in any parenthesization;
3. image weighting is done in that same precision before the cross GEMM;
4. `abs2` is explicitly `real * real + imag * imag` in that precision; and
5. the three final score additions/subtractions are performed in that
   precision without reassociation.

The bound permits any reduction tree satisfying those conditions.  It does
not permit TF32, reduced-precision accumulation, an opaque `abs`/`hypot`
lowering, fast-math reassociation, or a different number of product terms.

## Expanded-square center bound

Suppose `g_p` is the expanded score computed entirely in precision `p`.  Under
the operation contract above,

```text
abs(g_p - S) <= eta_p

eta_p = gamma_p(2 n + 4) H
      + 0.5 gamma_p(n + 5) (A + C)
      + gamma_p(3) d0.
```

The terms account for:

- one image-weight multiply before a `2 n`-term real component of the complex
  cross GEMM, followed by the three score recombination operations;
- two operations for each explicit component-square energy, an `n`-term FMA
  dot, and the three score recombination operations; and
- the recombination error applied to `d0`.

This is a conservative forward-error bound.  It deliberately does not depend
on the small, cancellation-prone value of `S`.

### Computable energy envelopes

Retain the nonnegative energy-dot outputs `Ahat_p` and `Chat_p` already used by
the expanded scorer.  For explicit component squares,

```text
alpha_p = gamma_p(n + 2)
Abar = up(Ahat_p / down(1 - alpha_p))
Cbar = up(Chat_p / down(1 - alpha_p))
Hbar = up(sqrt(up(Abar * Cbar)))
```

Then evaluate `eta_p` using `Abar`, `Cbar`, and `Hbar`, with upward rounding at
every operation.  This needs no third GEMM.  For a cheaper block-uniform FP64
bound, use the maximum `Abar` and `Cbar` over the block before evaluating
`Hbar`.  The resulting looseness is normally negligible because the FP64
coefficients are tiny.  FP32 should use candidate-specific energy envelopes
when practical.

For the GF46 square crop, `n = 100 * 51 = 5100`.  Useful coefficients are:

| coefficient | FP32 | FP64 |
| --- | ---: | ---: |
| `gamma(2 n + 4) = gamma(10204)` | `6.0857593469844254e-4` | `1.1328715743287933e-12` |
| `gamma(n + 5) = gamma(5105)` | `3.0437432711958565e-4` | `5.6676885407146371e-13` |
| `gamma(3)` | `1.7881396630060073e-7` | `3.3306690738754711e-16` |

The FP32 center therefore pays an error proportional to the large cross and
energy terms.  With the FP64 center, the direct FP32 reduction error normally
dominates `eta_64`.

## Direct CUDA residual bound

The current rectangular direct kernel has:

```text
B = 128 threads
P = 4 prefetch fractions
Cchunk = B / P = 32 pixels per chunk
Rblock = 16 rotations
T = translation_count
L = floor(B / T) active lanes per translation
```

This definition requires `1 <= T <= 128`, so that `L >= 1`.

Each active lane visits at most

```text
Nlane = ceil(n / Cchunk) * ceil(Cchunk / L)
```

pixels.  It accumulates nonnegative residual terms in FP32, and each output
receives one atomic add from each of the `L` active lanes.  Zero-valued atomic
adds from inactive lanes are exact and do not enlarge the count.

The direct per-pixel sequence is FP32 round-to-nearest subtraction for both
components, an explicit imaginary square, an FMA for the two squares, an exact
power-of-two half scaling, and a weighted accumulation FMA.  A deliberately
safe path count is

```text
kD = Nlane + L + 5
deltaD = gamma_32(kD).
```

The extra five cover the local subtraction/square path conservatively.  A
sharper proof can reduce this by one by counting the weighted insertion FMA as
the first lane-reduction operation; the one-count saving is not material and
should not be used without a separate kernel-level proof.

Because `d0`, weights, squared residuals, lane sums, and atomics are
nonnegative, any legal atomic lane order satisfies

```text
abs(Dhat - Q) <= deltaD Q,
abs((-Dhat) - S) <= deltaD Q.
```

This positivity is the key to a useful certificate.  A generic
cancellation-based bound on the direct kernel would be much wider.

For GF46 with `T = 29`, `L = 4`, the current mapping assigns exactly `1275`
valid pixels to each lane.  The generic formula conservatively gives
`Nlane <= 160 * 8 = 1280`, and therefore

```text
kD = 1289
deltaD = 7.6836290477420432e-5
```

using an upward-rounded stored value in implementation.

## Candidate interval around the live FP64 center

Let `g = g_64` and let `eta = eta_64` be outward-rounded.  Since `Q = -S`,

```text
Q <= Qbar = up(max(0, up(-g + eta))).
e = up(eta + up(deltaD * Qbar)).
lo = down(g - e)
hi = up(g + e)
```

Then the raw direct FP32 score is guaranteed to lie in `[lo, hi]`.  This is
candidate-specific: a poor candidate has a larger direct-reduction radius,
but it does not inflate every other block's interval.

The ordinary score path may still narrow `g` once to FP32 before adding priors.
That cast is irrelevant to the certificate because `[lo, hi]` is centered on
the live FP64 value and encloses the direct value itself.

If an implementation can only center on `m = RN32(g)`, use the actual cast
defect while `g` is live:

```text
cast = up(abs(float64(m) - g))
r = up(eta + cast)
Qbar = up(max(0, up(-float64(m) + r)))
e = up(r + up(deltaD * Qbar)).
```

This is tighter than a half-ULP worst-case cast allowance.  The live-FP64
variant remains preferable.

The same construction works for an FP32 expanded center by replacing `g` and
`eta_64` with the promoted FP32 value and `eta_32`.  It is formally sound but
usually much wider because `eta_32` contains the expanded-square cancellation
envelope.

## Outward-rounded primitives

Certificate arithmetic should be FP64 and use directed enclosures.  Define
`up64(x)` as the next FP64 value toward positive infinity after an RN64 basic
operation, and `down64(x)` analogously toward negative infinity.  Apply them
after every add, subtract, multiply, divide, and square root in the formulas
above.  Unary negation and exact FP32-to-FP64 promotion need no guard.

For propagation through an actual FP32 addition, define:

```text
floor32(x): greatest finite FP32 value <= x
ceil32(x):  least finite FP32 value >= x
```

These can be implemented without changing the global rounding mode:

1. compute `q = RN32(x)`;
2. for `floor32`, step `q` once toward `-inf` when `float64(q) > x`;
3. for `ceil32`, step `q` once toward `+inf` when `float64(q) < x`.

The FP64 operation producing `x` must first be rounded outward in the relevant
direction.  Nonfinite results are a certificate failure, not an interval.

## Priors

Raw-winner and post-prior support are separate obligations.  Preserve the raw
candidate interval before priors.  For the posterior interval, propagate it
through the same finite FP32 priors in the exact production order: class,
rotation when present, then translation when present.

Starting with `[lo, hi]`, for each stored FP32 prior `p` use

```text
lo = float64(floor32(down64(lo + float64(p))))
hi = float64(ceil32(up64(hi + float64(p))))
```

Round-to-nearest FP32 addition is monotone, so this encloses the direct scorer
after every prior addition.  It is tighter than converting the raw symmetric
radius into a generic multi-add gamma bound.

As a simpler fallback, first form a raw radius `e_m` around the macro FP32
value `m`.  For the live-FP64 interval, a safe choice is
`e_m = up(e + up(abs(float64(m) - g)))`.  For `q` prior additions with
`Pabs = sum abs(p_j)`, a symmetric post-prior radius around the macro FP32 path
is then

```text
e_post = up(e_m + gamma_32(q) * (2 abs(m) + e_m + 2 Pabs)).
```

The directed interval propagation should be the first-line implementation.

## Complete block selector

For every image and each class/source 16-rotation block `b`, reduce all
rotation and translation candidate intervals to bounds on that block's direct
maximum:

```text
Lb = max_j lo_j
Ub = max_j hi_j
```

Do this once for raw intervals and once after prior propagation.  Complete
per-source-block bounds are required across every class; a retained candidate
Top-K is not a certificate.

Let `Lstar_raw = max_b Lb_raw` and `Lstar_post = max_b Lb_post`.

- Raw-winner blocks: select every `b` with `Ub_raw >= Lstar_raw`.
- Posterior-support blocks: select every `b` with
  `Ub_post >= Lstar_post - W`, where the current RELION FP32 support span is
  `W = 138`.
- Directly rescore the union, using inclusive comparisons to preserve ties.

`W = 138` is a separate production contract: RELION shifts the best score to
`50` before exponentiation, and the CUDA exponent helper becomes zero below
`-88`.  If either rule changes, recompute `W`; the floating-point certificate
does not establish that span.

After rescoring, only direct scores may determine the winner and support.

Proof, applied independently to either track: if `Db` is the true direct
maximum in block `b`, the candidate intervals give `Lb <= Db <= Ub`.  Hence
`Lstar <= Dmax`.  A block containing a global raw maximizer has
`Ub >= Dmax >= Lstar`, so the raw selector includes it.  If a block contains a
posterior candidate at least `Dmax - W`, then
`Ub >= Dmax - W >= Lstar - W`, so the support selector includes it.  The
selector may include extra blocks, but it cannot omit a required block while
the intervals and `W` contract hold.

If only a macro block maximum `Mb` and uniform block radius `Eb` are retained,
`[Mb - Eb, Mb + Eb]` is safe but weaker.  A per-image maximum radius is weaker
still because one high-residual candidate widens every block.  Directly
reducing candidate endpoints is the recommended representation.

## Formal contract and fail-closed cases

The mathematics above is formal under its assumptions.  The current source is
not formally qualified until the lowering and runtime contract is audited.
In particular, pre-backend StableHLO for `jnp.abs(z) ** 2` uses an `abs`
operation followed by a multiply; it is not the two-operation component-square
sequence assumed by `gamma(n + 2)`.  The certificate path must instead compute
`real * real + imag * imag` explicitly after promotion, or prove a bound for
the final GPU `abs` implementation.

Use full direct rescoring for the affected image or batch if any of these
conditions fails:

- a reference, image, weight, `d0`, prior, center, energy, coefficient, or
  interval endpoint is NaN or infinite;
- a weight or `d0` is negative;
- `n`, `T`, block size, prefetch fraction, active-lane count, compact-pixel
  mapping, or rotation-tail shape differs from the values used for `kD`;
- `T` is outside `1..128`, so `L = floor(128 / T)` is not positive;
- `k u >= 1`, an interval overflows, or an outward-rounding primitive is not
  available;
- the FP64 path does not promote the complex64/float32 operands before
  weighting, component squares, both GEMMs, and recombination;
- XLA or the GEMM backend uses TF32, reduced-precision accumulation,
  unaccounted split-K terms, reassociation outside the counted tree, or a
  non-IEEE conversion;
- the CUDA direct kernel loses its explicit RN operations, positivity, or
  atomic topology;
- overflow occurs, or gradual underflow/subnormal handling is not covered by
  the audited floating-point model; or
- a partial final rotation block or compact-pixel skip cannot be shown to have
  no more operations than the padded bound.

Zero padding, skipped compact pixels, and a partial final 16-rotation block are
safe only when they remove nonnegative terms or operations.  They must never
increase `n`, `Nlane`, or the number of atomic contributions beyond the bound.
For a production certificate, pin and record the JAX/XLA/CUDA versions,
relevant flags, HLO, and representative SASS.  A compiler or kernel change
invalidates the audit until rechecked.

## Precision choice and expected tightness

Use the live promoted-FP64 expanded score as the first-line certificate center.
It has three advantages:

1. `eta_64` is about nine orders of magnitude smaller than the corresponding
   FP32 cancellation coefficients at GF46;
2. the remaining direct term uses the candidate's own `Qbar`, with no
   absolute-product GEMM; and
3. candidate endpoint reduction prevents a large residual elsewhere from
   widening the candidate or block of interest.

FP32 expanded GEMM is cheaper as a standalone macro kernel, but its formal
`eta_32` scales with `H`, `A`, and `C`, not with the small residual.  It is
therefore likely to select more direct blocks and can lose its kernel-level
advantage in total hybrid runtime.  The decision gate should compare

```text
projection + center GEMMs + interval reduction
+ selected-block reprojection/direct rescore + fallback cost
```

rather than GEMM time alone.

The Cauchy bound can be loose when magnitude vectors are poorly aligned, and
`gamma(kD)` assumes every rounding error has the adverse sign.  Both are
worst-case effects.  FP64 makes the first nearly irrelevant; candidate-specific
`Qbar` and block endpoints localize the second.

## Empirical evidence is not the certificate

The [all-particle GF46 FP32 diagnostic](../perf/vdam_coarse_gemm_streaming_selector_13329608.md)
observed a maximum absolute direct delta of `3.3125` over `1,069,056,000`
finite pairs.  The [all-particle promoted-FP64 diagnostic](../perf/vdam_coarse_gemm_streaming_fp64_13330442.md)
observed `1.5`; an earlier clean timing gate observed `0.6875` on its smaller
gate surface and a `4.512x` coarse comparison speedup over direct.  These
measurements support the precision recommendation, but none is a bound for a
new image or trajectory.

The promoted-FP64 all-particle diagnostic also retained a near-tie winner
mismatch.  This is expected: a more accurate expanded score is still not the
same FP32 reduction topology.  The certificate is useful only as a direct
rescore selector.

## Qualification plan

1. Make component-square energy explicit in the certificate path and capture
   StableHLO, optimized HLO, PTX, and representative SASS for both GEMMs and
   the single ordinary-path FP32 conversion.
2. Unit-test every outward primitive at zeros, powers of two, subnormal and
   normal boundaries, maximum finite values, and halfway cases.
3. On small arrays, compare the expanded center and emulated direct topology
   with an MPFR or exact-rational oracle.  Exercise all legal lane-add orders.
4. Instrument complete GPU paired surfaces and assert, candidate by candidate,
   that raw and post-prior direct scores lie inside their intervals.  Record
   maximum slack and selected block counts, but do not tune away formal slack
   using those observations.
5. Exercise translation counts that change `L`, final pixel chunks, compact
   skips, zero weights, partial rotation blocks, exact ties, the `W = 138`
   boundary, and every fail-closed branch.
6. Compare the hybrid result with full direct scoring for winner, significant
   support, normalization/evidence, and downstream frozen science cases.  The
   selected values must come from the direct scorer.
7. Run a Slurm H100 timing gate for total FP32-center and FP64-center hybrid
   wall time, including reprojection, interval work, exact rescore, and
   fallback frequency.

Until all seven steps pass, keep the hybrid selector and promoted-FP64 scoring
default-off.
