# Source16 coarse-diff2 FP32 range and FTZ audit

Status: **native-cubin audit, not production qualification**. This note covers
the direct selected-rotation-block CUDA scorer only. It does not qualify the
promoted GEMM center, the complete hybrid selector, or an arbitrary installed
CUDA library.

## Provenance and scope

The audited CUDA inputs are the blobs at commit
`925fccd3979aeb3536bf780a069e6d707926c2f4` (tree
`ebff0708b0f86b73d8d785aef4028649b51837ec`):

| file | Git blob | SHA-256 |
| --- | --- | --- |
| `recovar/cuda/cuda_backproject.cu` | `39987cc5eea90ecf60f5fbaa3d11cd001cbbff29` | `3ae0a06712b4497badb96bea832d9cca10f4df90273ad157fec7d8750bb4e66d` |
| `recovar/cuda/Makefile` | `fcd5509ae17ee065ce526577ef6a5fddeda21d09` | `b7965d1472ea43679f76a89df42e0765036004cdc579adb028afa76007c3b4e8` |
| `recovar/cuda/relion_coarse_diff2_projector_body.inc` | `30cdefe39aee838ed5bdec487c68e60c9c2dab9f` | `7154eee2871eb2fbae686b21836dbe66c3abf3641739d392015edcd8de04079e` |

Those three blobs are unchanged at this report branch's parent `6365091bf`
and at integration HEAD `d810048f1` at audit time. The exact source, binary,
SASS, commands, and manifest are under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_source16_ftz_sass_audit_20260902_925fccd/`.
That directory has a `SAFE_TO_DELETE` marker.

The library was rebuilt with CUDA 13.3.73, GCC 11.5.0, Python 3.11.13, and
JAX/JAXLIB 0.9.0.1. Its SHA-256 is
`92c3c098995ed89f32c4d258936363402c0f1c1d0945f9c452d79f9eb1b5dd9f`
and its GNU build ID is `d02a0a419816523977f4491397ca48e5bfb70ff9`.
The Makefile uses `-O3` without `--use_fast_math` or `--ftz=true`; `nvcc
--help` reports `--ftz=false` as the CUDA 13.3 default.

## Source traversal and operation count

The reviewed constants and traversal are

```text
B = 128                         threads per block
C = 128 / 4 = 32               full positions per chunk
T                               translation count, 1 <= T <= 128
L = floor(128 / T)              active lanes per translation
A = ceil(128 / T)               total atomic issuers for one translation
Nlane = ceil(n / 32) ceil(32/L) maximum visited positions in one lane
k = Nlane + L + 5               maximum relative-rounding path length
```

There are exactly `L` active lane sums. When `128 % T != 0`, the first
`128 % T` translations also receive one inactive zero atomic, hence `A`; no
translation receives more than one such extra issuer. Skipped lookup entries
only remove terms. The certificate still requires the exact lookup to visit
each compact scorer pixel once and to use the same `n`, `T`, and lookup passed
to the selected-block FFI.

For every visited pixel, `relion_fine_diff2_update_f32` executes two explicit
RN subtractions, one RN imaginary square, one RN FMA for the square sum, one
RN multiplication by `0.5`, and one RN weighted-accumulation FMA. Output is
initialized by storing the float32 `d0`, then each thread issues an
`atomicAdd`, including inactive zero lanes. All scientific terms and partial
sums are nonnegative when stored weights and `d0` are finite and nonnegative.

## Native SASS result

The selected-block kernel was disassembled for every native image emitted by
the Makefile:

| arch | FADD | FMUL | FFMA | local `.FTZ` | FP32 atomics | atomic `.FTZ.RN` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sm_80 | 32 | 32 | 32 | 0 | 16 | 16 |
| sm_86 | 32 | 32 | 32 | 0 | 16 | 16 |
| sm_89 | 32 | 32 | 32 | 0 | 16 | 16 |
| sm_90 | 32 | 32 | 32 | 0 | 16 | 16 |
| sm_100 | 32 | 32 | 32 | 0 | 16 | 16 |
| sm_120 | 32 | 32 | 32 | 0 | 16 | 16 |

The counts include the fully unrolled 16-rotation body. A representative
`sm_90` pixel sequence is `FADD`, `FADD`, `FMUL`, `FFMA`, `FMUL 0.5`, `FFMA`,
with no `.FTZ`. Its global reductions are
`REDG.E.ADD.F32.FTZ.RN.STRONG.GPU`. Thus these native cubins preserve local
subnormals and flush subnormal operands/results at the global atomics.

This is a binary fact, not a source-language guarantee. Driver JIT from PTX,
an unlisted architecture, another compiler or flag set, a changed CUDA blob,
or a different loaded shared object invalidates it.

## Additive underflow/FTZ supplement

Let

```text
u      = 2^-24          FP32 unit roundoff
lambda = 2^-126         minimum positive normal FP32 value
sigma  = 2^-150         half the minimum positive subnormal FP32 value
D      = up(Pmax + Ymax)
M      = max(1, D)
W      = max(1, wmax)
H      = up(M^2 W).
```

Provided intermediate overflow is excluded, each audited local RN operation
has the mixed model

```text
fl(z) = z (1 + delta) + epsilon,
abs(delta) <= u, abs(epsilon) <= sigma.
```

Following an additive subtraction residue through a square contributes at
most a constant times `M sigma`; square, half-scale, and accumulation
residues contribute at most a constant times `W sigma` or `sigma`. Since
`M,W >= 1`, a deliberately loose operation-by-operation total is
`16 sigma H` per visited pixel before later relative roundings. The condition
`k u < 1` also gives `(1 + u)^k < e < 3`. Therefore

```text
rho_local = up(64 n sigma H)
```

is a conservative local additive envelope.

For a nonnegative global atomic, define `F` as FP32 flush-to-zero. Its audited
operation is modeled as `F(RN(F(x) + F(y)))`. Flushing its two inputs loses
less than `2 lambda`. After that flush, any nonzero input is normal, so the
nonnegative sum is at least `lambda`; result FTZ adds no second subnormal
loss. Counting every active or inactive issuer and later relative
amplification gives

```text
rho_atomic = up(8 A lambda).
```

This includes a subnormal `d0` being flushed as the current-output input and
an inactive zero atomic flushing a subnormal partial. A source-specific bound
for the audited cubins is consequently

```text
rho_source = up(rho_local + rho_atomic).
```

For a deliberately compiler-portable provisional model that allows an
absolute `lambda` perturbation at every local input and operation, including
local DAZ/FTZ, the coarse bound

```text
rho_portable = up(256 (n + A + 1) lambda H)
```

covers at most five loaded inputs and six local FP operations per visited
pixel, their square/weight propagation, and less-than-three downstream
relative amplification. The integration draft's still broader

```text
rho_draft = up(2^20 (n + 128) lambda H)
```

dominates this provisional bound because `A <= 128`. The large factor should
remain labeled provisional unless this derivation and the exact runtime
binary identity become enforced contracts.

## Range guard and first-overflow argument

Use a distinct outward FP64 operation for every addition and multiplication;
one `nextafter` after a compound expression is not sufficient. With `U+` and
`Ux` denoting those single-operation upward enclosures, a conservative local
gate propagates

```text
D0       = U+(Pmax, Ymax)
Dhat     = U+(Ux(1 + u, D0), 3 lambda)
r2       = Ux(Dhat, Dhat)
i2       = U+(Ux(1 + u, r2), lambda)
sum_arg  = U+(r2, i2)
sum_hat  = U+(Ux(1 + u, sum_arg), lambda)
half_hat = U+(Ux(1 + u, Ux(0.5, sum_hat)), lambda)
term_hat = U+(Ux(1 + u, Ux(half_hat, wmax + lambda)), lambda).
```

Reject if any operand, coefficient, or stage is nonfinite, negative where
nonnegativity is required, or greater than `FLT_MAX`. Separately form

```text
Qmax = U+(d0, Ux(n, Ux(Ux(D0, D0), wmax)))
Dmax = U+(Ux(U+(1, gamma_32(k)), Qmax), rho)
```

and require `Dmax <= FLT_MAX`.

The aggregate check is not circular. Assume, for contradiction, the first
overflow after all explicitly gated local stages. Every earlier lane or
atomic operation is finite. Expand that finite prefix as its nonnegative exact
partial objective plus its relative and additive perturbations. The exact
partial is at most `Qmax`, its path is no longer than `k`, and its accumulated
absolute perturbation is at most `rho`. For the would-be first overflowing
operation, omit its final rounding; that has one fewer relative factor and no
larger additive budget. Its exact pre-round argument is therefore at most
`Dmax <= FLT_MAX`, which cannot round to infinity. This contradicts the first
overflow. Positivity is essential: it makes every exact partial no larger
than the complete `Qmax` in every legal atomic order.

Any failed range or identity check must route the affected image/batch to the
full direct scorer. It must never authorize accepting a macro score.

## Counterexample-focused qualification tests

At minimum retain these tests against the exact selected-block FFI:

1. `p=1e20`, `w=1`: local FP32 square overflows and the certificate fails.
2. `p=1e20`, `w=1e-20`, and `p=1e20`, `w=0`: the exact final objective is
   finite (or zero), but the local square still overflows; both must fail.
3. `p=1e-30`, `w=1`: a nonzero promoted square becomes zero in FP32 and zero
   remains inside the additive interval.
4. `d0` equal to the minimum subnormal with zero residuals: the first global
   atomic may flush it; zero remains enclosed.
5. A lane sum of `2^-127` followed by a global atomic: the local subnormal is
   preserved and the atomic flushes it; zero remains enclosed.
6. Values immediately around the component-square, square-sum FMA, weighted
   FMA, and aggregate `FLT_MAX` boundaries; the unsafe side fails closed.
7. Translation counts with and without a remainder in `128 / T`, especially
   `T=29`, to exercise the extra inactive atomic issuer.
8. Runtime library/build-ID and lookup-digest mismatch: certification is
   refused before exact rescoring.

## Code references

- `recovar/cuda/cuda_backproject.cu`:
  `relion_fine_diff2_update_f32`,
  `relion_coarse_diff2_rotation_block_f32`,
  `relion_coarse_diff2_rotation_blocks_initialize_f32_kernel`, and
  `RelionCoarseDiff2RotationBlocksF32Impl`
- `recovar/cuda/Makefile`: `CUDA_ARCH` and `NVCC_FLAGS`
- NVIDIA CUDA Compiler Driver documentation: `--ftz` and `--use_fast_math`
- NVIDIA PTX ISA and CUDA Programming Guide: global FP32 `atom.add` FTZ
  behavior
