# Agent brief: refactor and optimize RECOVAR dense single-volume EM

Reviewed against `ma-gilles/recovar`, branch `dev`, on 2026-03-31.

This brief is intentionally scoped to the path that matters most right now:
- **one homogeneous volume**,
- **dense rotation + translation grid**,
- **GPU execution now**,
- **data-parallel multi-GPU next**.

Do **not** broaden scope until this path is clean, stable, and benchmarked.

---

## 1. Mission

Produce a small, professional, maintainable implementation of dense-grid EM for a single volume that:
- preserves current numerical behavior for the supported path,
- isolates the homogeneous dense kernel from heterogeneity and refinement code,
- runs efficiently on one GPU,
- is structurally ready for multi-GPU image-sharded execution,
- has stable documentation and tests.

---

## 2. Hard scope boundary

### In scope
- homogeneous EM mean update,
- dense pose grid,
- `E` step for posterior responsibilities,
- `M` step for mean update,
- current CUDA/JAX slice and adjoint substrate,
- batch planning,
- single-GPU performance,
- multi-GPU preparation via additive reductions.

### Explicitly out of scope for this pass
- low-rank heterogeneity cleanup,
- covariance estimation cleanup,
- refined or hierarchical pose search,
- SGD state cleanup,
- new modeling assumptions,
- redesign of the reconstruction math.

The correct strategy is to make the dense homogeneous path the canonical core and treat the rest as later extensions.

---

## 3. Current diagnosis

The current code already contains the right computational core, but it is obscured by mixed concerns.

### What is already good
- The forward model has been partially centralized in `ForwardModelConfig`.
- The dense E-step is already reduced to projection precompute plus GEMMs.
- The dense M-step already reduces to two additive sufficient statistics.
- Low-level GPU projection/backprojection support already exists.

### What is making the code hard to maintain
- homogeneous EM, heterogeneity, covariance, and refined-grid utilities are mixed in one package,
- legacy and newer Equinox-style kernels coexist in the same modules,
- function names do not always match what they do,
- batching policy is scattered and heuristic,
- accumulator lifecycle is implicit,
- the dense path is harder to identify than it should be.

---

## 4. Immediate cleanup targets

Make these changes first.

### 4.1 Establish one canonical internal implementation

Use the Equinox / `ForwardModelConfig` path as the **only canonical internal kernel path**.

Rules:
- new code must call typed internal kernels,
- legacy function signatures may remain only as thin compatibility wrappers,
- no logic duplication between legacy and canonical paths,
- if a wrapper exists, it must adapt arguments and immediately delegate.

### 4.2 Split the dense homogeneous path into its own package

Create a dedicated package, for example:

```text
recovar/em/dense_single_volume/
    __init__.py
    api.py
    types.py
    plan.py
    projection_cache.py
    posterior.py
    accumulate.py
    solver.py
    engine.py
    distributed.py
```

Do not leave the dense homogeneous path spread across `e_step.py`, `m_step.py`, `states.py`, and `iterations.py` as peer-level concepts.

### 4.3 Fix naming

Rename functions so names reflect semantics.

Recommended renames:
- `E_with_precompute` -> `compute_posteriors_dense_grid`
- `M_with_precompute` -> `accumulate_mean_stats_dense_grid`
- `sum_up_images_fixed_rots_eqx` -> `accumulate_mean_stats_rotation_block`
- `EMState` -> `DenseSingleVolumeState` for this path, or keep `EMState` only as a thin facade over the new dense implementation.

### 4.4 Make accumulator lifecycle explicit

Introduce an explicit stats object such as:

```python
@dataclass
class MeanStats:
    Ft_y: jax.Array
    Ft_ctf: jax.Array
    n_images: int
    log_likelihood_sum: jax.Array | None = None
```

And use explicit creation / finalization:
- `stats = zero_mean_stats(plan)`
- `stats = accumulate_batch(..., stats)`
- `new_mean = solve_mean_from_stats(stats, state, plan)`

Do not rely on mutable class attributes or implicit carry-over.

### 4.5 Remove obvious dead or misleading code

Examples that should be cleaned immediately:
- remove unused allocations,
- remove stale comments that refer to older APIs,
- remove duplicated kernel variants once wrappers are in place,
- replace magic variable names like `mult`, `big_image_batch`, and ambiguous `projections` reuse with precise names.

---

## 5. Target architecture

The dense homogeneous path should expose exactly five internal layers.

### 5.1 `types.py`

Own the typed containers.

Recommended objects:
- `DensePoseGrid(rotations, translations)`
- `DenseEMPlan(forward_config, pose_grid, image_batch_size, rotation_block_size, projection_block_size, probs_dtype, math_dtype)`
- `ProjectionCache(projections, projections_abs2)`
- `ImageBatch(images, ctf_params, dataset_indices)`
- `PosteriorBatch(probabilities, hard_assignment=None, log_norm=None)`
- `MeanStats(Ft_y, Ft_ctf, ...)`
- `DenseSingleVolumeState(mean, mean_variance, noise_variance)`

Keep static metadata separate from dynamic arrays.

### 5.2 `plan.py`

Own every memory and batching decision.

Responsibilities:
- compute rotation precompute batch size,
- compute image batch size,
- compute rotation block size for accumulation,
- choose dtypes,
- decide whether optional caches are resident,
- expose all choices in a printable plan object.

All existing ad hoc multipliers such as `*5`, `*10`, `*20`, and `mult = 5` should be replaced by one planner with named fields and comments that explain the intended memory model.

### 5.3 `projection_cache.py`

Own the iteration-level forward precompute.

Responsibilities:
- precompute $P_r \mu$ for all dense-grid rotations,
- optionally precompute $|P_r \mu|^2$,
- own the device residency of those arrays,
- expose shape-checked cache objects.

This cache is the main iteration-level state that should be shared across all image batches.

### 5.4 `posterior.py`

Own only the dense homogeneous E-step.

Responsibilities:
- preprocess images,
- compute CTF once for the batch,
- generate shifted images,
- compute cross-term GEMM,
- compute norm-term GEMM,
- normalize to posteriors,
- optionally compute hard assignments and diagnostics.

No backprojection logic should live here.

### 5.5 `accumulate.py`

Own only the M-step sufficient statistics.

Responsibilities:
- consume a `PosteriorBatch`,
- reuse batch-local image transforms where possible,
- update `Ft_y`,
- update `Ft_ctf`.

No final solve should live here.

### 5.6 `solver.py`

Own the final mean update from sufficient statistics.

Responsibilities:
- call the RELION-style post-processing routine or a clearer wrapped equivalent,
- apply mean prior regularization,
- return the updated mean.

### 5.7 `engine.py`

Own the orchestration.

Responsibilities:
- create plan,
- build projection cache,
- loop over image batches,
- call posterior and accumulation kernels,
- finalize the mean,
- return diagnostics.

This module should be readable top-to-bottom by someone who only cares about dense homogeneous EM.

---

## 6. Preferred execution model

The best long-term internal pattern is:

```text
prepare iteration
-> precompute projections once
-> for each image batch:
     compute posteriors
     immediately accumulate mean stats
     optionally record hard assignments / diagnostics
-> finalize mean from stats
```

This is the model the public dense homogeneous API should present.

### Strong recommendation

Fuse the per-batch E-step and M-step accumulation at the orchestration level so that batch-local transforms can be reused.

The current code recomputes some batch-local quantities multiple times across E and M. In the cleaned implementation, the preferred flow is:
1. preprocess the batch once,
2. compute CTF once,
3. generate shifted images once if memory allows,
4. compute posteriors,
5. immediately accumulate `Ft_y` and `Ft_ctf`.

Keep "return full posterior tensor" as an optional debugging / analysis path, not as the required core execution model.

### Concrete execution skeletons

#### Dense full-grid path to implement now

For the dense single-volume path, the engine should look like this:

```text
for image_batch in dataset:
    compute batch-local image preprocessing
    for rotation_block in dense_rotation_grid:
        either read precomputed projections for this block
        or compute them once for this iteration and keep them resident
        evaluate E-step scores for all images in the batch against this rotation block
    normalize posterior probabilities per image over the full dense hidden-state grid

    for rotation_block in dense_rotation_grid:
        accumulate probability-weighted image sums and CTF weights
        backproject only these aggregated rotation-block statistics

solve the final mean update from accumulated Ft_y and Ft_ctf
```

Two points matter here. First, the expensive projection work should be amortized over many images, so the default implementation should precompute dense-grid projections once per EM iteration and reuse them across all image batches. Second, the M-step should aggregate over images and translations before backprojection, since that is what gives the large reduction in adjoint work.

For the supported path, prefer the following storage policy:
- if the dense projection cache fits on device, keep it resident on GPU for the full EM iteration,
- otherwise keep a host-side cache and stream rotation blocks to GPU,
- do not start with disk-backed projection caching unless profiling shows that host memory is insufficient.

A disk cache is usually the wrong first optimization here because the dense-grid projections are reused immediately and repeatedly within the same iteration; the likely bottlenecks are device memory and host-device bandwidth, not arithmetic.

#### How this extends to a higher-grid or refined search later

The higher-grid story should be treated as a later extension of the same engine, not as a separate implementation. The clean conceptual split is:

1. **Dense stage:** run exactly the dense full-grid path above on a coarser grid.
2. **Refinement stage:** for each image, refine only a small candidate set of poses.

That refinement stage should look roughly like this:

```text
for image_batch in dataset:
    run dense-grid E-step on coarse grid
    choose per-image candidate poses

    while refinement not converged:
        for each image, evaluate refined local proposals near its current candidates
        update per-image local posterior weights or local best candidates

    accumulate final sufficient statistics from the refined per-image pose sets
```

The important engineering point is that the coarse dense stage still benefits strongly from batched image processing and shared projection blocks, while the refined stage becomes much more image-local. At that point there is much less advantage to forcing the same batchwise rotation aggregation used in the dense stage. One can still sum over translations efficiently, but the clean global matrixized structure is weaker once each image carries its own local refined pose set.

This is also why the present refactor should focus first on the dense single-volume path. The same batch-planning, cache, posterior, and additive-stats interfaces should later support refinement, but the dense path is the right place to get the core abstractions correct.

This same outer structure should also support a future PPCA extension. The main difference will be that the hidden-state work per pose is larger and the practical rotation-block size may have to be smaller, but the overall engine pattern should remain: precompute what can be shared, evaluate posterior quantities in blocks, and accumulate additive sufficient statistics immediately.

---

## 7. Multi-GPU design

The first multi-GPU version should be **image-sharded data parallelism**.

### Required design choice

Replicate the projection cache across devices and shard images across devices.

Why this is the right first version:
- `Ft_y` is additive over images,
- `Ft_ctf` is additive over images,
- communication is then just an all-reduce over a small number of arrays,
- posterior normalization remains purely local because each device sees all rotations and translations for its image shard.

### Do not do this first
- do not shard rotations first,
- do not split the dense hidden-state axis across devices in the first version.

Rotation sharding makes posterior normalization harder because normalization must span all hidden states. It is a valid later optimization, not the first professional implementation.

### Multi-GPU interface shape

Design `MeanStats` so it can be all-reduced directly. The multi-GPU engine should look like:

```text
broadcast / replicate mean and projection cache
-> shard image batches across devices
-> local posterior + local stats accumulation
-> all-reduce MeanStats
-> solve mean once from reduced stats
```

This should be possible without changing the math modules.

---

## 8. Important semantic fixes

These should be treated as correctness and API-consistency issues, not just code style.

### 8.1 Discretization consistency

The dense path must have one of these two behaviors:

1. fully honor `disc_type` in both forward and adjoint operators, or
2. explicitly restrict the dense single-volume path to `linear_interp` and raise a clear error otherwise.

Do not keep an API that appears general while the implementation silently hardcodes part of the pipeline.

### 8.2 Explicit supported-mode contract

At the top of the new package, document the supported mode clearly. For example:
- supported now: `disc_type = linear_interp`, one volume, dense grid, GPU,
- supported later: cubic, half-volume mode, rotation sharding, heterogeneity bridge.

### 8.3 Deterministic batch semantics

Batch outputs should not depend on hidden mutable state.

The dense homogeneous kernels should be pure functions of:
- state,
- plan,
- cache,
- batch,
- current stats.

---

## 9. Testing plan

Before deleting or moving old code, freeze behavior with tests.

### 9.1 Numerical equivalence tests

Add small synthetic tests that compare old and new code for the supported path.

Required comparisons:
- precomputed projections,
- posterior probabilities,
- hard assignments,
- `Ft_y`,
- `Ft_ctf`,
- final mean after one EM iteration.

Use small toy grids so failures are easy to inspect.

### 9.2 Device tests

Required tests:
- CPU smoke test for tiny toy inputs,
- GPU smoke test for the supported dense path,
- if `disc_type != linear_interp` is unsupported, test that the error is explicit and early.

### 9.3 Batch partition invariance

Test that splitting the same dataset into different image-batch partitions yields the same final `Ft_y`, `Ft_ctf`, and mean up to expected floating-point tolerance.

This is essential because batching is part of the API contract for future distributed execution.

### 9.4 Multi-GPU reduction test

Add a test that simulates image sharding and verifies that

```text
reduce(local_stats(shard_1), ..., local_stats(shard_k))
```

matches the single-device stats on the combined dataset.

This test should exist before the full distributed engine is implemented.

### 9.5 Benchmark tests

Add one reproducible benchmark for the dense homogeneous path:
- one iteration,
- fixed image size,
- fixed rotation / translation counts,
- fixed GPU.

Track at least:
- compile time,
- steady-state iteration time,
- GPU memory,
- images per second.

Use this benchmark to justify any optimization that increases complexity.

---

## 10. Recommended refactor sequence

Use this exact order.

### Phase 1. Freeze behavior
- write small numerical equivalence tests,
- identify the exact supported dense homogeneous mode,
- benchmark the current implementation.

### Phase 2. Extract the new package
- create `dense_single_volume/`,
- move typed containers and planning there,
- wrap current kernels without changing math yet.

### Phase 3. Unify the implementation path
- make the Equinox/config path canonical,
- demote legacy functions to wrappers,
- clean names and signatures,
- make stats lifecycle explicit.

### Phase 4. Fuse batch-local work
- avoid recomputing batch-local transforms across E and M when possible,
- make the default engine stream `posterior -> stats accumulation` inside one batch loop.

### Phase 5. Multi-GPU readiness
- add a reducible `MeanStats` object,
- implement image-sharded local accumulation,
- add all-reduce integration points.

### Phase 6. Optional second-stage memory optimization
- if dense grids become too large, implement blockwise posterior normalization so full posterior tensors do not need to be materialized for the full rotation set.

Do not start with Phase 6.

---

## 11. Optional second-stage optimizations

These are good ideas, but they are **after** the cleanup.

### 11.1 Blockwise posterior normalization

For very large dense grids, avoid materializing the full $(image, rotation, translation)$ posterior tensor by using a two-pass blockwise normalization:
- first pass computes per-image normalization statistics,
- second pass recomputes block scores and immediately accumulates `Ft_y` and `Ft_ctf`.

This is the right next step if hidden-state memory becomes the main bottleneck.

### 11.2 Packed half-volume representation

The low-level CUDA projection/backprojection substrate already knows about half-image and half-volume layouts. After the refactor is stable, consider whether the dense homogeneous state and stats should live in packed half-volume form to reduce memory and bandwidth.

### 11.3 Fixed batch shapes

Because JAX specializes compiled programs to input shapes, prefer stable batch shapes in hot paths when practical. Padding the final batch or using a fixed-shape execution helper may reduce compilation churn.

Do not introduce this until the core refactor is benchmarked.

---

## 12. Documentation requirements

This cleanup must land with permanent docs.

Required docs:
1. a mathematical overview of the dense homogeneous path,
2. a code-map doc for the new package,
3. a developer note on batching, projection caching, and future multi-GPU reduction,
4. docstrings on every public function in the new package.

The docs should explain the algorithm by the five conceptual layers:
- model,
- posterior evaluation,
- sufficient-statistics accumulation,
- solve,
- execution policy.

---

## 13. Acceptance checklist

The refactor is done when all of the following are true.

- The dense single-volume path has its own package.
- There is exactly one canonical internal implementation for that path.
- The supported mode is explicit.
- `disc_type` behavior is honest and consistent.
- `MeanStats` is explicit and batch-additive.
- The high-level engine reads clearly from top to bottom.
- Numerical equivalence tests pass.
- Batch partition invariance tests pass.
- A benchmark exists.
- A multi-GPU reduction test exists.
- Permanent docs are checked in.

---

## 14. One-sentence engineering target

Turn the current dense homogeneous EM path from "one important algorithm hidden inside a large mixed module" into "a small, typed, benchmarked package whose public story is: precompute projections, evaluate posteriors, accumulate additive stats, solve the mean, and reduce across image shards when scaling out."
