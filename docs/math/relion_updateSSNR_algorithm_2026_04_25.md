# RELION `updateSSNRarrays` algorithm + recovar deviation + cold-start fix proposal

**Date**: 2026-04-25
**Author**: algorithm-investigation agent
**Scope**: cold-start `--iter 0` collapse on tip `2163cb0c` of `codex/relion-parity-em-phase01-rebase`

## TL;DR (5-line summary)

RELION computes the **current** iter's FSC from the M-step backprojected
accumulators **before** calling `maximization()`, then `updateSSNRarrays`
uses that **fresh, post-M-step** FSC to compute tau2. Recovar uses the
**previous** iter's FSC (`fsc_history[-1]` or `init_fsc`), which at
iter 1 is the gridding-FSC of the input volume — a bootstrap surrogate
that has artificially-high low-shell values (~0.999) and triggers a
~10⁶ ssnr·sigma2 amplification. **Fix: re-order recovar's iter loop to
match RELION (compute unreg half-maps + FSC FIRST, then compute tau2 from
THIS iter's FSC, then the regularised Wiener solve).** This preserves
late-iter parity (the FSC of iter K is identical regardless of whether
we use it before or after `maximization()` *as long as the prior is
recomputed in the same call*) and removes the 1-step staleness that
catches the cold start.

---

## 1. RELION `updateSSNRarrays` — pseudocode

Source: `/scratch/gpfs/GILLES/mg6942/relion/src/backprojector.cpp:1044-1207`.

```
def updateSSNRarrays(tau2_fudge,           # in/out tau2_io                                    line 1045
                     tau2_io, sigma2_out,
                     data_vs_prior_out,
                     fourier_coverage_out,
                     fsc,                  # CURRENT iter's gold-standard FSC
                     avgctf2,
                     update_tau2_with_fsc, # TRUE for split-half, FALSE for joined run
                     is_whole_instead_of_half,
                     correct_tau2_by_avgctf2):

    max_r2 = round(r_max * pf)**2
    osc    = pf**3                    # oversampling_correction (3D)         line 1061

    # --- 1. Per-shell sigma2 from BACKPROJECTED weight ---                  lines 1067-1093
    sigma2[s] = 0; counter[s] = 0
    for (k, i, j) in weight (the 3-D padded BP weight):                       line 1069
        r2 = k*k + i*i + j*j
        if r2 < max_r2:
            ires = round(sqrt(r2) / pf)                                       line 1074
            sigma2[ires]  += osc * weight[k,i,j]
            counter[ires] += 1
    for s in shells:                                                          line 1082
        if sigma2[s] > 1e-20:
            sigma2[s] = counter[s] / sigma2[s]      # invert: 1 / mean_weight
        elif sigma2[s] == 0:
            sigma2[s] = 0.
        else:  ERROR

    # --- 2. tau2 from FSC (only when update_tau2_with_fsc) ---              lines 1099-1128
    if update_tau2_with_fsc:
        for s in shells:
            myfsc = max(0.001, fsc[s])                                        line 1112  *** clamp ***
            if is_whole_instead_of_half:                                      line 1113
                myfsc = sqrt(2 * myfsc / (myfsc + 1))
            myfsc  = min(0.999, myfsc)                                        line 1119  *** clamp ***
            myssnr = myfsc / (1 - myfsc) * tau2_fudge                         line 1120-22
            tau2[s] = myssnr * sigma2[s]                                      line 1123
            data_vs_prior[s] = myssnr                                         line 1126

    # --- 3. fourier_coverage / data_vs_prior diagnostics ---                lines 1130-1206
    # (not the bug; only used for output reporting)
```

The CALLER then uses `tau2_io` as the prior in the Wiener filter
`reconstruct()` (`backprojector.cpp` Wiener path), or hands it to
`externalReconstruct` for blush.

---

## 2. CRITICAL: order of FSC and tau2 update in RELION's iter loop

Source: `/scratch/gpfs/GILLES/mg6942/relion/src/ml_optimiser_mpi.cpp`.

The auto-refine main loop is in `MlOptimiserMpi::iterateMaximization()`.
Per iter the sequence is:

```
line 3995  if (do_split_random_halves):
line 3999      if low_resol_join_halves > 0: joinTwoHalvesAtLowResolution()
line 4031      compareTwoHalves()    # <-- COMPUTES CURRENT-ITER FSC
                                     #     from BPref's downsampled
                                     #     averages of the half-maps
                                     #     populated by THIS iter's
                                     #     E-step accumulators.
                                     #     Stores into mymodel.fsc_halves_class[ibody]
line 4091  maximization()            # <-- INSIDE: line 5425 calls
                                     #     updateSSNRarrays(...,
                                     #         mymodel.fsc_halves_class[0],
                                     #         ...
                                     #         update_tau2_with_fsc = do_split_random_halves);
                                     #     i.e. uses THE FRESH CURRENT-ITER FSC
                                     #     to set tau2 BEFORE the Wiener solve
                                     #     (reconstruct() at 5578 / 5544).
```

Two more confirmations:
- MPI auto-refine: `ml_optimiser_mpi.cpp:2428` calls
  `updateSSNRarrays(..., mymodel.fsc_halves_class[ibody], avgctf2,
   do_split_random_halves, do_join_random_halves || do_always_join_random_halves, ...)`.
  In standard split-half iters, `update_tau2_with_fsc=true`, `is_whole=false`.
- The FSC stored in `mymodel.fsc_halves_class[ibody]` was JUST written by
  `compareTwoHalves()` from THIS iter's wsum_model.BPref data
  (`backprojector.cpp:936 BackProjector::getDownsampledAverage`,
   `backprojector.cpp:1000 calculateDownSampledFourierShellCorrelation`).

### What about iter 1 specifically?

At iter 0 (recovar `--iter 0`, RELION `it000`):
- `MlModel::initialise` at `ml_model.cpp:60` resizes
  `fsc_halves_class.resize(nr_classes * nr_bodies, aux)` where `aux` is
  empty / zeros.
- `MlOptimiser::initialiseDataVersusPrior` at `ml_model.cpp:1607` does
  `fsc_halves_class[iclass].initZeros(ori_size/2 + 1)` (only when
  `nr_bodies > 1`; otherwise it's left at the zero-init from line 60).
- So **the iter-0 FSC array written to `run_it000_half1_model.star` is
  literally all zeros** (or essentially zeros for a single body).

At iter 1 (the first real EM iteration of auto-refine):
- E-step runs (with the low-pass-filtered initial reference + firstiter_cc).
- `compareTwoHalves()` computes a fresh FSC from the iter-1 BPref accumulators.
- `maximization()` is called next; `updateSSNRarrays` is invoked WITH THIS FRESH FSC.
- So at iter 1, RELION's tau2 is built from iter-1's FSC, which has
  legitimate "early" structure (FSC ≈ 1 only out to ~`ini_high`,
  then drops sharply).

The iter-0 model.star FSC (= zeros) is **never used for tau2** by RELION:
it just gets immediately overwritten by `compareTwoHalves()` at iter 1.

---

## 3. RELION clamps and bootstrap behaviour (specific)

| What | Where | Value |
|---|---|---|
| FSC lower clamp | `backprojector.cpp:1112` | `myfsc = max(0.001, fsc[s])` |
| FSC upper clamp | `backprojector.cpp:1119` | `myfsc = min(0.999, myfsc)` |
| sigma2 floor | `backprojector.cpp:1084` | if `sum_weight_per_shell <= 1e-20`, sigma2 stays 0 |
| oversampling_correction | `backprojector.cpp:1061` | `pf³` for 3D; multiplies the per-voxel weight before shell averaging |
| iter-0 fsc_halves_class | `ml_model.cpp:60, 1607` | initialized to zeros — **NOT** used as prior |
| iter-1 fsc_halves_class | `ml_optimiser_mpi.cpp:4031, 5430` | freshly recomputed from THIS iter's BPref |
| iter-1 tau2 source | `ml_optimiser.cpp:5425-5434` | `updateSSNRarrays(..., fsc_halves_class[0], avgctf2, do_split_random_halves=TRUE, do_join_random_halves=FALSE, ...)` |

Two non-obvious facts that matter for parity:

1. **There is no separate "init tau2 from prior FSC" path in RELION.**
   `initialiseDataVersusPrior` at `ml_model.cpp:1557` does compute an
   initial `tau2_class` from the *power spectrum of the input
   reference*, but the comment at line 1594 explicitly says
   *"This is only for writing out in the it000000_model.star file"*.
   At iter 1, that tau2 is OVERWRITTEN inside `updateSSNRarrays` from
   the fresh iter-1 FSC.

2. **The `do_grad` (SGD) iter-1 special case** (`ml_optimiser_mpi.cpp:4011-4022`)
   sets `old_fscs[ibody] = 1` inside `ini_high` radius, 0 outside, then
   exponentially blends with the fresh iter-1 FSC — but this only fires
   for `do_grad`, NOT auto-refine. We can ignore it for our cold-start
   fix.

---

## 4. Recovar deviation — side-by-side

Source: `/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_codex_em_phase01_sparse_pass2_rebase_20260424/recovar/em/dense_single_volume/iteration_loop.py:2845-2872`.

| Step | RELION (auto-refine, split-half) | recovar (current `iteration_loop.py`) |
|---|---|---|
| **Order** | 1. E-step. 2. **compareTwoHalves → fresh FSC**. 3. maximization → `updateSSNRarrays(fresh_fsc)` → reconstruct(tau2). | 1. E-step. 2. low_resol_join_halves. 3. **`compute_relion_tau2_from_weights(prev_iter_FSC)`**. 4. Reconstruct. 5. Compute *this iter's* FSC LATER (line 2978). |
| **Which FSC drives tau2?** | THIS iter's FSC (post-M-step BPref). | PREVIOUS iter's FSC (`fsc_history[-1]`) or `init_fsc` (= the FSC RELION wrote into `run_itN_half1_model.star` for the iter we are continuing from). |
| **At iter 1 (cold start)** | FSC freshly computed from iter-1 BPref accumulators. Legit shape: 1 out to ~`ini_high`, 0 beyond. | `init_fsc` = the FSC from `run_it000_half1_model.star` = **all zeros** (clamped by `compute_relion_tau2_from_weights` to `[0.001, 0.999]` → ssnr ≈ 0.001 / 0.999 ≈ 0.001). For `--iter K, K>0`, `init_fsc` = legit RELION FSC from `it{K:03d}` star, which is fine. |
| **At iter 2 (cold start)** | Iter-2 fresh FSC (legit). | `fsc_history[-1]` = iter-1's FSC = `regularization.get_fsc_gpu(unreg_means[0], unreg_means[1])` of iter-1 unreg maps. This is the bug: iter-1 unreg half-maps are reconstructed from the iter-1 M-step accumulators with NO well-tuned tau2 (since iter-1 used effectively-zero tau2 from the all-zeros init_fsc), so the unreg maps have artifacts that **artificially correlate** between the two halves at low shells (residual init-volume leakage), producing FSC ≈ 0.999 in shells 0-15. Then `ssnr = 0.999 / 0.001 ≈ 999`, and combined with `sigma2 = 1 / (pf³ * mean_weight_per_shell)` where `mean_weight` is small at low shells → `tau2 = ssnr * sigma2 ≈ 10⁶+`. Wiener with too-large tau2 ⇒ no regularization ⇒ runaway amplitude ⇒ ave_Pmax collapse. |

| Step | RELION clamps | recovar clamps |
|---|---|---|
| FSC lower | `max(0.001, fsc)` (`backprojector.cpp:1112`) | `clip(fsc, FSC_ZERO_THRESHOLD, 1 - FSC_ZERO_THRESHOLD)` (`regularization.py:637`) — same epsilon on both sides; FSC_ZERO_THRESHOLD differs from RELION's 0.001 (need to verify; if it matches, identical). |
| FSC upper | `min(0.999, fsc)` (`backprojector.cpp:1119`) | same clip as above |
| `is_whole_instead_of_half` sqrt-rescale | applied when `do_join_random_halves` (last "joined" iter only) | NOT applied — recovar's `compute_relion_tau2_from_weights` skips this. **For per-half running iters, RELION does NOT apply the sqrt either**, so this is fine; only matters for the final joined iter. |
| oversampling_correction | `pf³ * weight` then shell-averaged (`backprojector.cpp:1061, 1075`) | `1.0 / (pf³ * bottom_avg)` (`regularization.py:643-644`) — equivalent algebraically. |

So the algorithmic clamps on the *prior path itself* are correct. The
deviation is **WHICH FSC IS PASSED IN**.

Cross-check: late-iter parity (1e-5 gap from `--iter 5+ → max_iter 1`)
is intact precisely because:
- For iter-K continuation, recovar uses `init_fsc =
  run_it{K:03d}_model.star`'s FSC, which IS RELION's iter-K
  fsc_halves_class.
- RELION's iter-(K+1) `updateSSNRarrays` would use the iter-(K+1) fresh
  FSC, but RELION's iter-K FSC is so close to iter-(K+1) FSC at
  convergence that the 1-step staleness is invisible — except at the
  cold start, where iter-1's "FSC" doesn't even exist yet.

---

## 5. Proposed minimal fix (NOT applied)

The cleanest fix that matches RELION exactly and preserves late-iter
parity is to **re-order** the iter loop in
`recovar/em/dense_single_volume/iteration_loop.py` so that:

1. After E-step + `low_resol_join_halves`,
2. **First** reconstruct unreg half-maps and compute the current iter's FSC,
3. **Then** call `compute_relion_tau2_from_weights(THIS_ITER_FSC, ...)`,
4. **Then** do the regularized Wiener solve with the freshly-updated tau2.

The unreg-then-FSC reconstruction was already happening 100 lines later
(lines 2944-2977); we just need to move it before the tau2 update and
remove the tau2 update from the `init_fsc` / `fsc_history[-1]` path.

### Diff sketch (NOT applied)

```diff
--- a/recovar/em/dense_single_volume/iteration_loop.py
+++ b/recovar/em/dense_single_volume/iteration_loop.py
@@ -2842,40 +2842,73 @@
             current_resolution_angstrom=prev_res_angstrom,
         )
 
-        # --- Update tau2 (signal prior) BEFORE the Wiener solve ---
-        # RELION's reconstruct() calls updateSSNRarrays(fsc_from_prev_iter)
-        # first, then applies the Wiener filter with the UPDATED tau2.
-        # Use the previous iteration's FSC (or init_fsc at iteration 0).
-        prev_fsc_for_tau2 = None
-        if fsc_history:
-            prev_fsc_for_tau2 = fsc_history[-1]
-        elif init_fsc is not None:
-            prev_fsc_for_tau2 = init_fsc
-        tau2_update_details = None
-        if prev_fsc_for_tau2 is not None:
-            mean_signal_variance, _, tau2_update_details = regularization.compute_relion_tau2_from_weights(
-                Ft_ctf_0,
-                Ft_ctf_1,
-                prev_fsc_for_tau2,
-                volume_shape,
-                tau2_fudge=tau2_fudge,
-                padding_factor=PADDING_FACTOR,
-                return_details=True,
-            )
-            logger.info(
-                "Pre-Wiener tau2 update from %s FSC: old_max=%.4e new_max=%.4e",
-                "init" if not fsc_history else "prev-iter",
-                float(jnp.max(jnp.abs(mean_variance))),
-                float(jnp.max(jnp.abs(mean_signal_variance))),
-            )
-            mean_variance = mean_signal_variance
+        # --- RELION-exact M-step ordering (auto-refine, split-half) ---
+        # RELION (ml_optimiser_mpi.cpp:4031, 4091; backprojector.cpp:1044):
+        #   1. compareTwoHalves() -> CURRENT iter's FSC from BPref
+        #      downsampled averages.
+        #   2. maximization() -> updateSSNRarrays(THIS_ITER_FSC) -> tau2.
+        #   3. reconstruct(tau2) -> regularized half-map.
+        #
+        # We therefore reconstruct UNREGULARIZED half-maps first, compute
+        # the current iter's FSC, THEN compute tau2 from that fresh FSC,
+        # THEN do the regularized Wiener solve below.
+        _t_unreg_first = time.time()
+        _unreg_means_for_fsc = [
+            _reconstruct_volume_eager(
+                Ft_ctf_k_local, Ft_y_k_local, volume_shape,
+                PADDING_FACTOR, tau=None, tau2_fudge=tau2_fudge,
+                projection_padding_factor=PROJECTION_PADDING_FACTOR,
+            )
+            for (Ft_ctf_k_local, Ft_y_k_local) in (
+                (Ft_ctf_0, Ft_y_0), (Ft_ctf_1, Ft_y_1),
+            )
+        ]
+        # Sign-align so FSC is computed in the same convention as below.
+        for k_half in range(2):
+            _unreg_means_for_fsc[k_half], _ = _align_fourier_volume_sign_to_reference(
+                _unreg_means_for_fsc[k_half], previous_means[k_half], volume_shape,
+            )
+        current_iter_fsc = regularization.get_fsc_gpu(
+            _unreg_means_for_fsc[0],
+            _unreg_means_for_fsc[1],
+            volume_shape,
+        )
+        del _unreg_means_for_fsc  # free GPU
+        logger.info("Computed iter-%d FSC for tau2 (RELION order): %.1fs",
+                    iteration + 1, time.time() - _t_unreg_first)
+
+        # Now run updateSSNRarrays-equivalent on THIS iter's fresh FSC.
+        mean_signal_variance, _, tau2_update_details = regularization.compute_relion_tau2_from_weights(
+            Ft_ctf_0,
+            Ft_ctf_1,
+            current_iter_fsc,
+            volume_shape,
+            tau2_fudge=tau2_fudge,
+            padding_factor=PADDING_FACTOR,
+            return_details=True,
+        )
+        logger.info(
+            "tau2 update from THIS-iter FSC: old_max=%.4e new_max=%.4e",
+            float(jnp.max(jnp.abs(mean_variance))),
+            float(jnp.max(jnp.abs(mean_signal_variance))),
+        )
+        mean_variance = mean_signal_variance
 
         # --- Free previous-iteration means to reclaim GPU memory ---
         previous_means = [np.asarray(mean).copy() if mean is not None else None for mean in means]
         for k in range(2):
             means[k] = None
@@ -2940,18 +2973,9 @@
         # --- Combined Fourier weights for data_vs_prior at next iteration ---
         Ft_ctf_combined = Ft_ctf_0 + Ft_ctf_1
 
-        # --- Compute unregularized half-maps for FSC and prior ---
-        _t_unreg = time.time()
-        unreg_means = [
-            _reconstruct_volume_eager(
-                Ft_ctf_0, Ft_y_0, volume_shape, PADDING_FACTOR,
-                tau=None, tau2_fudge=tau2_fudge,
-                projection_padding_factor=PROJECTION_PADDING_FACTOR,
-            ),
-            _reconstruct_volume_eager(
-                Ft_ctf_1, Ft_y_1, volume_shape, PADDING_FACTOR,
-                tau=None, tau2_fudge=tau2_fudge,
-                projection_padding_factor=PROJECTION_PADDING_FACTOR,
-            ),
-        ]
-        for k in range(2):
-            means[k], sign_flipped = _align_fourier_volume_sign_to_reference(means[k], previous_means[k], volume_shape)
-            if sign_flipped:
-                unreg_means[k] = -unreg_means[k]
-                logger.info("Aligned half-%d volume sign to the previous reference", k + 1)
-        logger.info("Unregularized reconstruction (2 halves): %.1fs", time.time() - _t_unreg)
-
-        # --- Compute FSC between half-maps ---
-        fsc = regularization.get_fsc_gpu(
-            unreg_means[0],
-            unreg_means[1],
-            volume_shape,
-        )
+        # FSC was already computed above (same convention) — reuse it.
+        fsc = current_iter_fsc
+        # Sign-align the post-Wiener regularized halves for downstream
+        # consumers (kept identical to the previous code path).
+        for k in range(2):
+            means[k], _ = _align_fourier_volume_sign_to_reference(
+                means[k], previous_means[k], volume_shape,
+            )
         fsc_history.append(fsc)
```

### Why this preserves late-iter parity

For iter K continuation (`--iter K, K>=1, max_iter=1`):
- recovar's iter loop starts at iteration K+1 internally.
- The current iter's FSC is computed from the iter-(K+1) M-step
  accumulators — exactly what RELION does.
- The previous code computed tau2 from `init_fsc` (= RELION iter-K's FSC)
  and got a 1e-5 match because at convergence iter-K and iter-(K+1) FSCs
  are nearly identical. The new code computes tau2 from the iter-(K+1)
  FSC directly (= RELION's exact convention), which can ONLY be more
  accurate, not less.

### Why this fixes the cold start

For `--iter 0`:
- iter 1 of recovar: E-step uses init volume + firstiter_cc; FSC is
  computed from iter-1 BPref accumulators; tau2 derived from iter-1's
  fresh FSC (legit `~ini_high` cutoff shape).
- iter 2: tau2 computed from iter-2's fresh FSC, NOT from iter-1's
  FSC. The iter-2 unreg-FSC has correct shape because iter-1 was
  regularized properly.
- The "fsc_history[-1] poisons iter-2 tau2" failure mode is impossible
  because we never use `fsc_history[-1]` for tau2 — we always use this
  iter's fresh FSC.

### Side effect — `init_fsc` becomes unused for tau2

`init_fsc` is still used at iter 0 to:
- Seed `data_vs_prior_iter` for the iter-1 `current_size` computation
  (`iteration_loop.py:1644-1671`). That logic is independent of the tau2
  bug and matches RELION's `wsum_model.current_size` recomputation
  (which also uses the prev-iter `data_vs_prior` — the iter-K-1 model.star).

That's a different code path and stays as-is.

---

## 6. Validation plan

**Fast (CPU/GPU, ~5 min total)**

1. **Cold-start tiny smoke** (the test that currently fails):
   ```bash
   pixi run python scripts/run_multi_iter_parity.py \
     --relion_dir <tiny/50 RELION dir> --data_star <particles> \
     --iter 0 --max_iter 3 --max_healpix_order 3 \
     --output_dir _agent_scratch/cold_start_fix_smoke
   pixi run python scripts/diff_relion_recovar_per_iter.py \
     --relion_dir <tiny/50 RELION dir> \
     --recovar_dir _agent_scratch/cold_start_fix_smoke
   ```
   **Pass criterion**: ave_Pmax at iter 1, 2, 3 all within 10% of
   RELION's reported value. iter-2 ave_Pmax must be ≥ 0.5
   (NOT collapse to ~1e-4).

2. **Late-iter parity preservation** (the test that currently passes):
   ```bash
   pixi run python scripts/run_multi_iter_parity.py \
     --relion_dir <5k/256 RELION dir> --data_star <particles> \
     --iter 5 --max_iter 1 --max_healpix_order <whatever> \
     --output_dir _agent_scratch/late_iter_after_fix
   pixi run python scripts/diff_relion_recovar_per_iter.py ...
   ```
   **Pass criterion**: max relative gap on iter-6 ave_Pmax,
   tau2 per shell, and final volume CC vs RELION must remain ≤ 1e-5
   (matching the current `2163cb0c` baseline). If this regresses, the
   diff broke late-iter parity and we need to investigate why
   iter-K+1 fresh-FSC differs measurably from iter-K stored-FSC.

---

## 7. Open questions / things I could not fully verify

1. **Is the recovar `regularization.get_fsc_gpu` exactly equivalent
   to RELION's `BackProjector::calculateDownSampledFourierShellCorrelation`
   (`backprojector.cpp:998-1042`)?**
   RELION's version operates on the *downsampled (un-padded) BPref data*
   produced by `getDownsampledAverage` (`backprojector.cpp:936`), which
   weighted-averages padded data to native resolution. Recovar's
   `get_fsc_gpu` operates on the unregularized real-space half-maps
   reconstructed via `_reconstruct_volume_eager(..., tau=None)`. These
   *should* give nearly identical FSCs (both are the cosine-similarity
   per shell of the per-half complex amplitudes), but there may be
   subtle weight-handling differences. If late-iter parity regresses
   after the fix, this is the first thing to check.

2. **`is_whole_instead_of_half` (joined iter)**: RELION's last
   "joined-halves" iter passes `is_whole_instead_of_half=true`, which
   activates the `myfsc = sqrt(2 * myfsc / (myfsc + 1))` rescale at
   `backprojector.cpp:1117`. Recovar's `compute_relion_tau2_from_weights`
   doesn't have this branch. If recovar ever runs the joined iter, this
   will need to be added — but it's irrelevant to the cold-start bug.

3. **Resolution-cutoff zeroing of FSC beyond `current_size/2`**:
   RELION zeros `fsc_halves_class` beyond `current_size/2 + 1`
   (`ml_optimiser_mpi.cpp:3394-3396`). Recovar's
   `iteration_loop.py:1648` does the same when consuming `init_fsc` for
   `data_vs_prior_iter`, but I did NOT verify whether `fsc =
   regularization.get_fsc_gpu(...)` at line 2978 also gets that
   zero-out before being fed to `compute_relion_tau2_from_weights` on
   the next iter. If not, this is a *separate* small deviation that
   should be added inside the new `current_iter_fsc` computation
   (apply `current_iter_fsc[cs//2:] = 0` before passing to the prior).

4. **iter-0 FSC value in `run_it000_half1_model.star`**: I did not
   directly read a real `run_it000_half1_model.star` to confirm it's
   all zeros. The conclusion above is based on `MlModel::initialise`
   at `ml_model.cpp:60` (`fsc_halves_class.resize(..., aux)` where
   `aux` is the empty default-constructed `MultidimArray<RFLOAT>`) and
   `initialiseDataVersusPrior` at `ml_model.cpp:1607` (only zeros for
   `nr_bodies > 1`). For `nr_bodies == 1` (the standard case), the
   fsc_halves_class array stays at its default-constructed state. The
   fix is robust to this question because it never uses the iter-0
   FSC for tau2 anyway.

---

## 8. Citations summary (file: line ranges I read)

- `/scratch/gpfs/GILLES/mg6942/relion/src/backprojector.cpp:1000-1207`
  — `calculateDownSampledFourierShellCorrelation` and `updateSSNRarrays`
- `/scratch/gpfs/GILLES/mg6942/relion/src/backprojector.cpp:936-994`
  — `getDownsampledAverage`
- `/scratch/gpfs/GILLES/mg6942/relion/src/ml_optimiser.cpp:5380-5600`
  — `MlOptimiser::maximization`, the per-class `updateSSNRarrays + reconstruct` block
- `/scratch/gpfs/GILLES/mg6942/relion/src/ml_optimiser.cpp:5860-5919`
  — iter-1 firstiter_cc post-processing of tau2
- `/scratch/gpfs/GILLES/mg6942/relion/src/ml_optimiser_mpi.cpp:2390-2630`
  — MPI maximization with split-half `updateSSNRarrays` calls
- `/scratch/gpfs/GILLES/mg6942/relion/src/ml_optimiser_mpi.cpp:3540-3620`
  — `compareTwoHalves`
- `/scratch/gpfs/GILLES/mg6942/relion/src/ml_optimiser_mpi.cpp:3990-4090`
  — iter loop ordering: low_resol_join → compareTwoHalves → maximization
- `/scratch/gpfs/GILLES/mg6942/relion/src/ml_optimiser.cpp:1930-1950, 2200-2240, 2510-2520, 2810-2900`
  — initialization (iter 0 paths, ini_high default, initialiseDataVersusPrior call)
- `/scratch/gpfs/GILLES/mg6942/relion/src/ml_model.cpp:50-103, 1500-1630`
  — `MlModel::initialise`, `setFourierTransformMaps`, `initialiseDataVersusPrior`
- `/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_codex_em_phase01_sparse_pass2_rebase_20260424/recovar/reconstruction/regularization.py:560-667`
  — `compute_relion_tau2_from_weights`
- `/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_codex_em_phase01_sparse_pass2_rebase_20260424/recovar/em/dense_single_volume/iteration_loop.py:2700-2980`
  — recovar iter loop with the buggy tau2 update site
- `/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_codex_em_phase01_sparse_pass2_rebase_20260424/recovar/em/dense_single_volume/iteration_loop.py:1620-1700`
  — `init_fsc` consumption for `current_size` (a separate, correct path that should NOT be touched)
- `/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar_codex_em_phase01_sparse_pass2_rebase_20260424/scripts/run_multi_iter_parity.py:370-405, 800-820`
  — how `init_fsc` is constructed at script entry
