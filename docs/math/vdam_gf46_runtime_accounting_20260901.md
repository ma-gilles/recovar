# VDAM GF46 runtime/work accounting (2026-09-01)

## Decision summary

The historical `9.1305x` GF46 wall-time gap is not a scientific-work gap.  Over
the frozen 200-iteration trajectory, RECOVAR executed only `1.2802%` more
physical coarse pixel-candidates than RELION.  The dominant old cost was an
obsolete coarse CUDA implementation that recomputed reference projections
inside translation work.

That defect is fixed on the mature shared EM/VDAM line by
`1d0cacd1990f...` (`perf(em): reuse coarse projections across translations`).
The mature atomic-plus-multistream candidate at `b261d16cfaad...` reduces the
representative iteration-181 internal ratio to approximately `1.94x`, and its
coarse GPU union is already about `9.0%` faster than RELION.  The remaining
large target is host/JAX execution topology and the exact-local sparse fine
pass, not another VDAM-only projector or scorer.

The older lane branch `6f10c2d3f075...` forked before projection reuse and must
not be used as the production base.  Continue from `be26b77dced3...` (or a
reviewed descendant), which contains the mature shared projection-reuse,
native-atomic, and multistream stack.

## Frozen 200-iteration accounting

Both programs processed exactly `112,400` particle-iterations.

| Measurement | Exact total | Relative to RELION |
|---|---:|---:|
| RELION coarse pixel-candidates | 563,153,698,252,800 | control |
| RECOVAR real-row coarse pixel-candidates | 505,872,088,934,400 | -10.1716% |
| Frozen RECOVAR padded physical pixel-candidates | 570,363,092,812,800 | +1.2802% |

Frozen RECOVAR issued 647 coarse scorer calls and allocated 148,872 image
slots: 112,400 real rows and 36,472 padding rows.  The old kernel scored the
padding.  Current worker paths accept the actual batch size and skip it.

For one coarse pass:

```text
P = S * (S/2 + 1)
B = min(500, floor(200,000,000 / (R * T)))
real work = N * R * T * P
old padded work = ceil(N/B) * B * R * T * P
```

`N` is the selected particle count, `R` orientations, `T` translations, and
`P` Fourier pixels.  K=1 uses one joint halfset stream; it does not repeat the
coarse scientific pass for two halfsets.

| Iterations | N | R | T | P | B | Calls |
|---|---:|---:|---:|---:|---:|---:|
| 1--19 | 200 | 576 | 29 | 364 | 500 | 19 |
| 20 | 200 | 4,608 | 29 | 364 | 500 | 1 |
| 21--89 | 200--432 | 4,608 | 29 | 1,300 | 500 | 69 |
| 90 | 440 | 36,864 | 49 | 1,300 | 110 | 4 |
| 91--109 | 448--592 | 36,864 | 49 | 5,100 | 110 | 101 |
| 110--119 | 600--672 | 36,864 | 45 | 5,100 | 120 | 59 |
| 120--129 | 680--752 | 36,864 | 21 | 5,100 | 258 | 30 |
| 130--139 | 760--832 | 36,864 | 37 | 5,100 | 146 | 60 |
| 140--149 | 840--912 | 36,864 | 13 | 5,100 | 417 | 30 |
| 150--159 | 920--992 | 36,864 | 25 | 5,100 | 217 | 50 |
| 160--169 | 1,000 | 36,864 | 45 | 5,100 | 120 | 90 |
| 170--179 | 1,000 | 36,864 | 21 | 5,100 | 258 | 40 |
| 180--189 | 1,000 | 36,864 | 29 | 5,100 | 187 | 60 |
| 190--199 | 1,000 | 36,864 | 13 | 5,100 | 417 | 30 |
| 200 | 1,000 | 36,864 | 21 | 5,100 | 258 | 4 |

At iteration 181 this is six calls with capacity 187, 1,000 real rows, and 122
padded slots.  Each coarse score tensor contains 199,913,472 float32 values
(`~0.745 GiB`).

## Runtime progress

| Boundary | RECOVAR | RELION | Ratio/read |
|---|---:|---:|---:|
| Frozen full GF46 external wall | 4,388.589 s | 480.650 s | 9.1305x |
| Mature iteration-181 coarse GPU union | 3.571 s | ~3.925 s | RECOVAR ~9.0% faster |
| Mature iteration-181 expectation | 7.824 s | 4.076 s | 1.92x |
| Mature iteration-181 internal iteration | 8.236 s | ~4.246 s | ~1.94x |

The mature projection-reuse implementation reduced a representative warm
coarse pass from approximately `32.65 s` to `6.54 s`.  On the current
atomic-plus-multistream iteration-181 profile, the remaining RECOVAR overhead
is approximately:

- `1.618 s` of pass-1 work outside the coarse CUDA union;
- `1.876 s` in exact-local sparse pass 2;
- `0.759 s` of other expectation scaffolding.

This makes the next performance sequence concrete:

1. keep the mature shared EM/VDAM projector, scorer, posterior, and x-half
   primitives;
2. avoid repeated materialization/host handling of the `~0.745 GiB` coarse
   score tensor where the same mathematical result can stay on device;
3. reduce per-bucket device-to-host barriers and helper-JIT shape churn in the
   exact-local fine pass using a shared fixed-capacity/device-carried executor;
4. pipeline preprocessing, posterior bookkeeping, fine scoring, and
   finalization so GPU work overlaps as RELION's does;
5. qualify the mature combined path through a complete joint-trajectory gate,
   then measure a full trajectory rather than extrapolating one iteration.

## Numerical acceptance

Bitwise identity is strong evidence but is not universally required.  A faster
implementation may regroup mathematically equivalent floating-point work only
when the resulting difference is stable, bounded, non-directional, and
non-growing; preserves discrete decisions and the convergence basin; causes no
material final-quality loss; and produces a material reproducible end-to-end
runtime gain.  Unstable cancellation or scale-growing errors fail.

## Evidence

- Frozen root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_full_expansion_v3_984637b7d_87274be_20260826/vdam-gf46/repeat-01/vdam-gf46`
- Mature combined gate, Slurm `13283759`:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_coarse_atomic_multistream_crossed_b261d16cf_h21g4_20260831`
- Combined report JSON SHA-256:
  `c843158cfc6bdbd1a7e5d0c59325c98a0f0f5ec647d1bd458911b9d991a55a79`
- Combined report Markdown SHA-256:
  `6c9862c2837ab1ea8fcc88e0c35a7c0c3adefca1b827c6e860b383abbc413634`
- Combined run-provenance SHA-256:
  `c7fabc266c58463db0f363adffba08d43d0e6bc7592a3b652f45bbb0756c1f0d`
