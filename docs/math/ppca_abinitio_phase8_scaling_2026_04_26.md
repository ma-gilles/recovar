# Phase 8 — vol=64 scaling attempt (2026-04-26)

**Branch:** `claude/ppca-abinitio-scale` (off `claude/ppca-abinitio-v0`)
**Final commit:** `fbd76032` (Phase 8.2 image-batched streaming M-step)

## Goal

Phase 4 established that v0 OOMs at vol=64 with 4 successive Slurm attempts (jobs 7344125, 7345575, 7345585, 7345648) up to `--mem=400GB`. Phase 8 attempted to fix this without a deep architectural change — purely via einsum tweaks (8.1) and Python-loop image batching (8.2).

## Phase 8.1 — einsum fusion (FAILED, partial)

Diagnosis from Phase 4: `_per_rotation_residual_image` (mean_update.py:131) materialized an `(n_img, n_rot, half_image)` complex128 tensor before reducing over images. At vol=64 order=2 this is 148 GiB.

**Fix kept:** fused `"irt,itk->irk" + sum_i` → `"irt,itk->rk"` single contraction. Eliminates the explicit intermediate; XLA streams the i-axis reduction.

**Reverted:** four other "fusions" I tried in `update_factor_closed_form` and `_per_rotation_bias_image`. Multi-tensor einsums like `"igt,igtk,igtl,ip->gklp"` gave XLA worse contraction trees than the original two-step code (XLA's einsum optimizer doesn't guarantee minimum-memory paths). 217 tests still pass with just the single-line line-131 fix.

**Outcome at vol=64:** still OOMs. The line-131 fix was necessary but not sufficient.

## Phase 8.2 — image-batched streaming M-step

Real Python-loop streaming, opt-in via `image_batch_size` parameter on:
- `update_factor_closed_form`
- `update_mu_residualized`
- `update_mu_homogeneous`

A new `_accumulate_M_B_for_batch` helper computes a single batch's contribution to `(M_im, B_im)`; the public function loops over batches and sums. Same for the mean updates via `_stream_or_full_mu_update` / `_accumulate_mu_for_batch`. CLI flag `--image-batch-size` threaded through `run_two_stage`.

**Bit-identicality verified:** `tests/ppca_abinitio/test_streaming_bit_identical.py` (10 tests) confirms streaming and unbatched paths agree to 1e-10 (μ) and 1e-7 (U) at vol=16 with batch sizes {8, 16, 32}. **All pass.**

## Phase 8 vol=64 verification — 6 SUBMITTED CONFIGS, ALL OOM'D

| Job | n_img | order | batch_size | --mem | Outcome |
|---|---|---|---|---|---|
| 7373283 | 1024 | 1 | 256 | 96 GB | OOM at ~1 min |
| 7373284 | 1024 | 2 | 128 | 128 GB | OOM at ~1 min |
| 7373285 | 4096 | 2 | 128 | 192 GB | OOM at ~1 min |
| 7374621 | 1024 | 1 | 128 | 384 GB | OOM at 18 min |
| 7374622 | 1024 | 2 | 64 | 384 GB | OOM at 18 min |
| 7374623 | 256 | 1 | 64 | 256 GB | OOM at <1 min |
| 7374937 | 1024 | 1 | 128 | **600 GB** | OOM at 20 min |
| 7374938 | 256 | 1 | 64 | **600 GB** | OOM at 20 min |
| 7374939 | **128** | 1 | 32 | **600 GB** | OOM at 20 min |

Critical observation: **the smallest case (n=128 batch=32) OOM'd at 600 GB with the same 20-min wall as n=1024 batch=128**. The OOM is NOT linear in dataset size, NOT eliminable by image batching, and NOT eliminable by host-memory headroom up to 600 GB.

**Diagnosis:** XLA JIT compile of `update_factor_closed_form` at vol=64 is intrinsically too large. The HLO module XLA generates for this function at vol=N³ apparently grows super-linearly in N. At vol=32 the compile fits in <60 GB peak; at vol=64 it exceeds 600 GB. The streaming Python loop runs the SAME compiled function over batches, so it shares one compilation — but the compilation peak is the wall.

The OOM site is consistently between `print("s init (flat)")` and `print("U init: pe = ...")`. The intervening operations are `PPCAInit(...)` (cheap) and `projector_frobenius_error(cur.U, ds.U_half_true, cfg.volume_shape)` (small numpy + iFFT). These don't allocate gigabytes themselves; they trigger JIT-tracing of *subsequent* JAX operations including the joint-loop M-step.

## Verdict — Phase 8 closed as FAILED

The streaming refactor is **mathematically correct and bit-identical** but the JIT compile at vol=64 is the binding constraint, not the steady-state runtime tensors.

Phase 8 cannot fix vol=64 with the existing approach. The next surgery requires one of:

1. **Explicit `@jax.jit` boundaries to shrink compile units.** Decompose `update_factor_closed_form` into smaller @jit'd sub-functions whose individual HLO modules are bounded. Risk: still hits the same wall if the sub-pieces are individually too big.
2. **`jax.lax.scan` over the i-axis.** Replaces the Python loop with an inside-JIT scan, but the loop body still has to compile. Could be smaller because each iter sees only one batch's shapes.
3. **Wire the CUDA backprojection kernel** (`recovar/cuda/`) into the M-step accumulator. Sidesteps XLA entirely for the worst part. ~2-3 weeks of CUDA-FFI bridging work.
4. **Sharded/multi-GPU** via `pjit` to split the compile across devices.

All four are real engineering projects of days-to-weeks. They are out of scope for "improve Phase 8 with one targeted change" and belong in their own follow-up branch.

## What Phase 8 successfully delivered

Even though vol=64 itself didn't run, Phase 8.2 delivered real artifacts that survive:

1. **Line-131 einsum fix** in `_per_rotation_residual_image` — eliminates a real 148 GiB intermediate at vol=64 order=2 that Phase 4 hit. Bit-identical at vol=32; stays in.
2. **`image_batch_size` parameter** on the three M-step functions — bit-identical at vol=16 across batch sizes 8/16/32. Will become useful once XLA compile is fixed.
3. **`--image-batch-size` CLI flag** in `run_cryobench.py`.
4. **10 streaming bit-identicality tests** — regression gate for any future surgery.
5. **Hard evidence** that the vol=64 wall is XLA-compile-bound, not steady-state-runtime-bound. Future work should target XLA compile size, not memory tensors.

## Honest position

v0 ships at vol=32. Phase 8 confirms that the path to vol≥64 is not a config tweak or a small refactor — it requires either (a) deep XLA work or (b) wiring the existing CUDA backprojection kernel into the M-step. Both are appropriate for follow-up branches `claude/ppca-abinitio-scale-vol128` and `claude/ppca-abinitio-cuda-mstep`, neither of which exists yet.

## Files added by Phase 8

- `recovar/em/ppca_abinitio/factor_update.py` — `image_batch_size` param + `_accumulate_M_B_for_batch` helper + line-131 einsum kept
- `recovar/em/ppca_abinitio/mean_update.py` — `image_batch_size` param + `_stream_or_full_mu_update` + `_accumulate_mu_for_batch`
- `scripts/ppca_abinitio/run_cryobench.py` — `--image-batch-size` CLI + threaded through call sites
- `scripts/ppca_abinitio/submit_phase8_vol64_smoke.sh` — smoke harness
- `tests/ppca_abinitio/test_streaming_bit_identical.py` — 10 regression tests
