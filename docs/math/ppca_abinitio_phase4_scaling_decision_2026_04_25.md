# Phase 4 — scaling vs grid decision (2026-04-25)

## Question

Once vol=64/order=2 instrumentation lands, decide whether the next-larger-config follow-up branch should focus on:
- **T2a:** scaling (vol=128, image batching, larger n_img)
- **T2b:** pose-grid coarseness (port RELION HEALPix oversampling + `SamplingPerturbation`)

## What we tried

| Attempt | Job | Config | Slurm `--mem` | Outcome |
|---|---|---|---|---|
| 1 | 7344125 | vol=64, n=1024, order=2 | 128 GB | GPU-OOM 148 GiB |
| 2 | 7345575 | vol=64, n=128, order=2 | 128 GB | host-OOM-killed |
| 3 | 7345585 | vol=64, n=256, order=1 | 192 GB | host-OOM-killed |
| 4 | 7345648 | vol=64, n=256, order=1 | **400 GB** | host-OOM-killed |

Each successive attempt reduced workload (smaller `n_img`, smaller `n_rot`) and increased the host memory budget. Attempt 4 was an aggressive 400 GB host budget on the **smallest realistic vol=64 configuration**, and it still hit the OOM-killer during the first joint M-step iteration.

## Diagnosis

The first OOM (attempt 1) was caused by an `(n_img, n_rot, img_half)` complex128 intermediate inside `_per_rotation_residual_image` (`recovar/em/ppca_abinitio/mean_update.py:131`). I added the corresponding `mean_update_residual_stack` term to the memory model. Subsequent attempts re-OOM'd with this intermediate sized at only ~5 GB — significantly under the host budget — so a *second* large allocation must occur during JIT compilation or activation buffering at vol=64.

The memory model captures the steady-state algorithm tensors but **does not capture XLA/JAX runtime overhead**. At vol=32 the runtime overhead is ~1.5× steady state (well-quantified). At vol=64 the runtime overhead apparently jumps significantly, and we lack the instrumentation to measure it without successfully running the configuration.

## Decision

**Scaling (T2a) is the binding constraint.** The instrumentation goal of Phase 4 is met by the failure: vol=64 with the v0 architecture cannot fit a single H100 host even at 400 GB host RAM. The pose-grid (T2b) question is not the bottleneck — we cannot even run order=1 at vol=64.

**T2a wins.** The follow-up branch must do a real refactor:
1. **Image-level batching** of the entire M-step pipeline (mean + factor + score), not just the E-step. The `_per_rotation_residual_image` and analogous accumulators must consume image batches and accumulate into running tensors rather than materializing the full (n_img × n_rot × img_half) stack.
2. **CUDA backprojection kernel reuse.** The existing `recovar/cuda/` backprojection kernel was designed for exactly this scaling regime; the v0 module currently does its accumulation in pure JAX. Wiring the CUDA kernel into the M-step would amortize the constant per-image overhead.
3. **Streaming dataset loader.** At production n=100k, data tensors alone exceed 26 GB; the loader must yield image batches rather than loading the full dataset.

This is a real engineering project, not a config tweak. Recorded as the **`claude/ppca-abinitio-scale-vol128`** follow-up branch (Phase 6).

## What this does NOT change about v0

- The vol=32 / n=1024 / healpix_order=1 results in the Phase 1 ablation sweep are unaffected.
- The Phase 0 cheat-free contract is unaffected.
- The Phase 2 ProjCov refit (post-EM, vol=32) is unaffected.
- The memory model + instrumentation hooks are useful infrastructure for the follow-up branch.

## What v0 honestly claims

v0 is a **vol=32 algorithm validation**, not a production ab-initio cryo-EM tool. The PR description is updated to reflect that the vol=64 smoke is a **failed-as-designed** outcome that establishes the need for the T2a follow-up branch, not a positive scaling demonstration.

## Phase 4 instrumentation that DID land

Even though vol=64 didn't run, Phase 4 produced two reusable artifacts:
- **`recovar/em/ppca_abinitio/memory_model.py`** — analytic predictions, validated against vol=32 and confirmed lower-bound at vol=64. The follow-up branch can iterate against this.
- **`--instrument` flag** — pose entropy, effective pose count, GPU memory peak. Will be the right Phase 4 input once vol=64 actually runs. The vol=16 CPU smoke from Phase 3.4 verified the pose-entropy / effective-pose-count instrumentation works end-to-end.
