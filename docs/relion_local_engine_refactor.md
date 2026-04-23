# RELION Local Engine Refactor

Tracked TODO IDs in the active RELION local-search refactor.

## RELION_LOCAL_ENGINE

| ID | Status | Owning module | Rationale |
|---|---|---|---|
| `RELION_LOCAL_ENGINE/T001` | OPEN | `recovar/em/dense_single_volume/iteration_loop.py` | Grouped-union local search is the wrong active abstraction. |
| `RELION_LOCAL_ENGINE/T002` | OPEN | `recovar/em/dense_single_volume/local_layout.py` | Active RELION local path must use per-image local hypotheses. |
| `RELION_LOCAL_ENGINE/T003` | OPEN | `recovar/em/dense_single_volume/local_em_engine.py` | Local path should not depend on dense shared-grid engine orchestration. |
| `RELION_LOCAL_ENGINE/T004` | OPEN | `recovar/em/dense_single_volume/iteration_loop.py` | RELION-parity hacks should move inward, out of outer-loop control flow. |

## DENSE_ENGINE_BOUNDARY

| ID | Status | Owning module | Rationale |
|---|---|---|---|
| `DENSE_ENGINE_BOUNDARY/E001` | OPEN | `recovar/em/dense_single_volume/em_engine.py` | `em_engine.py` should remain dense/global-only. |
| `DENSE_ENGINE_BOUNDARY/E002` | OPEN | `recovar/em/dense_single_volume/em_primitives.py` | Shared primitives should be extracted so local logic does not grow back into the dense engine. |
| `DENSE_ENGINE_BOUNDARY/E003` | OPEN | `recovar/em/dense_single_volume/em_engine.py` | Half/full spectrum conversions need one explicit boundary. |
| `DENSE_ENGINE_BOUNDARY/E004` | OPEN | `recovar/em/dense_single_volume/em_engine.py` | Dtype policy cleanup is still needed. |

## Review rule

A tracked TODO ID may be removed from code only if:

1. this file marks it `RESOLVED`
2. the PR explicitly explains why
3. the regression test that pins the ID list is updated intentionally
