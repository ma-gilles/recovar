# RELION Local Engine Refactor

Tracked TODO IDs in the active RELION local-search refactor.

## RELION_LOCAL_ENGINE

| ID | Status | Owning module | Rationale |
|---|---|---|---|
| `RELION_LOCAL_ENGINE/T001` | RESOLVED | `recovar/em/dense_single_volume/iteration_loop.py` | Grouped-union local search and the now-degenerate `local_engine` selector have been removed; local RELION refinement uses the exact per-image engine. |
| `RELION_LOCAL_ENGINE/T002` | OPEN | `recovar/em/dense_single_volume/local_layout.py` | Active RELION local path must use per-image local hypotheses. |
| `RELION_LOCAL_ENGINE/T003` | OPEN | `recovar/em/dense_single_volume/local_em_engine.py` | Local path should not depend on dense shared-grid engine orchestration. |
| `RELION_LOCAL_ENGINE/T004` | OPEN | `recovar/em/dense_single_volume/iteration_loop.py` | RELION-parity hacks should move inward, out of outer-loop control flow. |

## DENSE_ENGINE_BOUNDARY

| ID | Status | Owning module | Rationale |
|---|---|---|---|
| `DENSE_ENGINE_BOUNDARY/E001` | OPEN | `recovar/em/dense_single_volume/em_engine.py` | `em_engine.py` should remain dense/global-only. |
| `DENSE_ENGINE_BOUNDARY/E002` | OPEN | `recovar/em/dense_single_volume/em_primitives.py` | Shared primitives should be extracted so local logic does not grow back into the dense engine. |
| `DENSE_ENGINE_BOUNDARY/E003` | RESOLVED | `recovar/em/dense_single_volume/helpers/preprocessing.py` | Dense/local image preprocessing now requires native packed-half Fourier preprocessing instead of dense-side full-to-half conversion fallbacks. |
| `DENSE_ENGINE_BOUNDARY/E004` | PARTIAL | `recovar/em/dense_single_volume/helpers/dtype_policy.py` | Dense, local, sparse pass-2, and local big-JIT paths now route score dtype choices through the shared precision policy; exact-local projection abs2 is now always on demand and no longer has materialized/fused env toggles. Remaining work is to reduce user-facing float64 toggles once parity no longer needs them. |
| `DENSE_ENGINE_BOUNDARY/E005` | OPEN | `recovar/em/dense_single_volume/em_engine.py` | Audit dense and local engines for duplicated implementation and move reusable pieces into helpers without hurting hot-path performance. |
| `DENSE_ENGINE_BOUNDARY/E006` | RESOLVED | `recovar/em/dense_single_volume/helpers/scoring.py` | Shared dense scoring/logsumexp/M-step kernels moved out of `em_engine.py`; significance code imports helpers directly instead of reaching back through the engine. |

## Review rule

A tracked TODO ID may be removed from code only if:

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

## Remaining Work After The Direct Half-Volume Fix

- Push the packed half-volume layout farther out of the RELION outer loop so the local path no longer depends on dense/global orchestration state (`T001`, `T003`, `T004`).
- Benchmark the direct `half_volume=True` path on the larger local-search fixtures (`20k @ 128`, `20k @ 256`, `50k @ 256`) and update the comparison table against RELION warm/cold timings.
- Keep the current warm `5k @ 128` parity benchmark as the guardrail:
  - direct `half_volume=True` exact local stays at about `20.8s` warm iteration wall
  - map/Pmax parity remains matched to RELION on that fixture
- Run the full repo test suite before merging or pushing beyond the working branch. The targeted half-volume tests pass, but the repo rules still require the full suite.
