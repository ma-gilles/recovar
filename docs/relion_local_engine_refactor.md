# RELION Local Engine Refactor

Tracked refactor IDs for the active RELION local-search refactor.

## RELION_LOCAL_ENGINE

| ID | Status | Owning module | Rationale |
|---|---|---|---|
| `RELION_LOCAL_ENGINE/T001` | RESOLVED | `recovar/em/dense_single_volume/iteration_loop.py` | Grouped-union local search and the now-degenerate `local_engine` selector have been removed; local RELION refinement uses the exact per-image engine. |
| `RELION_LOCAL_ENGINE/T002` | RESOLVED | `recovar/em/dense_single_volume/local_layout.py` | Active RELION local search now builds `LocalHypothesisLayout` with per-image rotation neighborhoods; grouped-union dense-grid local search is gone. |
| `RELION_LOCAL_ENGINE/T003` | RESOLVED | `recovar/em/dense_single_volume/local_em_engine.py` | Exact-local scoring and M-step run through `run_local_em_exact` / `run_local_bucket_big_jit`; local tests no longer assert equivalence to dense shared-grid M-step contracts. |
| `RELION_LOCAL_ENGINE/T004` | RESOLVED | `recovar/em/dense_single_volume/iteration_loop.py` | The outer loop now owns only RELION iteration scheduling. Exact-local tuple handling is centralized in `_LocalSearchIterationResult`, and the obsolete exact-v1 dispatch shim has been removed. |

## DENSE_ENGINE_BOUNDARY

| ID | Status | Owning module | Rationale |
|---|---|---|---|
| `DENSE_ENGINE_BOUNDARY/E001` | RESOLVED | `recovar/em/dense_single_volume/em_engine.py` | `em_engine.py` is dense/global-only; local search routing lives in `iteration_loop.py` and `local_em_engine.py`. |
| `DENSE_ENGINE_BOUNDARY/E002` | RESOLVED | `recovar/em/dense_single_volume/helpers/` | The transitional `em_primitives.py` re-export shim is gone; dense, local, and sparse pass-2 code import shared kernels directly from helpers. |
| `DENSE_ENGINE_BOUNDARY/E003` | RESOLVED | `recovar/em/dense_single_volume/helpers/preprocessing.py` | Dense/local image preprocessing now requires native packed-half Fourier preprocessing instead of dense-side full-to-half conversion fallbacks. |
| `DENSE_ENGINE_BOUNDARY/E004` | RESOLVED | `recovar/em/dense_single_volume/helpers/dtype_policy.py` | Dense, local, sparse pass-2, and local big-JIT paths route score dtype choices through `DensePrecisionPolicy`; removed env toggles are not replaced by ad hoc branches. Remaining float64 options are internal diagnostic hooks, not outer-loop user-facing switches. |
| `DENSE_ENGINE_BOUNDARY/E005` | RESOLVED | `recovar/em/dense_single_volume/em_engine.py` | Shared indexed batch fetch, half-volume M-step helpers, host timing accumulators, scoring helpers, and local result packing cover the audited dense/local duplicated support code without changing hot-path kernels. |
| `DENSE_ENGINE_BOUNDARY/E006` | RESOLVED | `recovar/em/dense_single_volume/helpers/scoring.py` | Shared dense scoring/logsumexp/M-step kernels moved out of `em_engine.py`; significance code imports helpers directly instead of reaching back through the engine. |

## Review rule

Tracked IDs should stay in this document after resolution so future agents can
see why the code markers were removed. A tracked code marker may be removed only
if:

1. this file marks it `RESOLVED`
2. the PR explicitly explains why
3. the regression test that pins the ID list is updated intentionally

## Negative Results To Remember

- `NOTE(local-projection-dedupe)`: do not retry per-bucket projection dedupe in the exact local engine unless the measured duplicate factor changes materially on the real `5k` local benchmark. This was tried multiple times and is a bad idea in the current path.
  - Before RELION-style reconstruction gating, the exact-local projection duplicate factor was already too small to justify extra gather complexity.
  - After reconstruction gating, the measured duplicate factor was only about `1.004-1.005`.
  - The dedupe experiment regressed the real `5k` exact-local run from about `76.7s` to about `126.9s`.
  - Keep the simpler direct projection path unless a future benchmark shows a much larger duplicate factor.

- `NOTE(local-half-volume-adjoint)`: the correct exact-local reference for packed half-image rows is the direct `half_volume=True` adjoint or its VJP equivalent, not the old `half_image=True, half_volume=False` path.
  - For out-of-plane rotations, the old `half_image=True, half_volume=False` comparison path produces a non-Hermitian full Fourier volume and is not a valid reference for packed-half accumulation.
  - The direct `half_volume=True` adjoint matches the JAX VJP of `slice_volume(..., half_volume=True, half_image=True)` on the weighted local rows, which is the contract the local engine should follow.
  - Do not reintroduce the “accumulate full then fold to half” workaround as a parity fix; fix tests and downstream consumers against the native packed-half contract instead.

## Resolved Work After The Direct Half-Volume Fix

- The packed half-volume layout is enforced inside the exact-local engine; tests now compare against the native `half_volume=True` contract rather than the old full-volume reference path (`T003`, `T004`).
- Benchmark the direct `half_volume=True` path on the larger local-search fixtures (`20k @ 128`, `20k @ 256`, `50k @ 256`) and update the comparison table against RELION warm/cold timings.
- Keep the current warm `5k @ 128` parity benchmark as the guardrail:
  - direct `half_volume=True` exact local stays at about `20.8s` warm iteration wall
  - map/Pmax parity remains matched to RELION on that fixture
- Run the full repo test suite before merging or pushing beyond the working branch. The targeted half-volume tests pass, but the repo rules still require the full suite.
