# RELION-Parity Plan v3 — Step-by-Step

**Date**: 2026-04-08
**Supersedes**: `plan_relion_parity.md` (v1, 2026-03), `plan_relion_parity_v2.md` (v2, 2026-04-02)
**Goal**: Bit-by-bit parity between recovar's `_refine_relion_mode` and
RELION 5.0.1 `relion_refine_mpi --auto_refine` on a single benchmark
dataset.

## Why a v3 plan

We discovered in the 2026-04-08 session that **most of our previous
"recovar beats RELION" / "recovar lags RELION" comparisons were
artifacts of running RELION incorrectly**, not real algorithmic
differences. Specifically:

1. RELION's `--ctf` defaults to OFF on the command line (the GUI
   adds it). Without it, RELION reconstructs the CTF-convolved volume
   (dark halo, ~18-22 Å plateau).
2. RELION's `--firstiter_cc` is needed for non-RELION init volumes;
   without it, the iter-1 Bayesian E-step collapses on a wrong-scale
   reference.
3. Five more GUI defaults are missing from the CLI: `--flatten_solvent`,
   `--zero_mask`, `--low_resol_join_halves 40`, `--norm`, `--scale`.
4. Our simulator wasn't normalizing particles to RELION's intensity
   scale, so RELION needed `--firstiter_cc` to absorb the mismatch
   even with everything else correct.

After fixing all five issues, the actual per-iter trajectories are
much closer than we thought, and most of the remaining gap comes from
a **small number of well-defined algorithmic differences** in
`_refine_relion_mode`. v3 is built on top of those clean baselines and
on a fresh read of RELION's source.

This v3 plan is the ONLY plan to follow going forward. v1 and v2 have
phases that are stale or have been completed; v3 reorganizes the
remaining work in priority order with concrete file:line citations and
verification steps.

## Bedrock: what RELION's auto_refine actually does, in order

(Cite: full spec in `plan_relion_parity_v3_relion_source_spec.md`,
pseudocode in `relion/src/ml_optimiser.cpp:3258` →
`iterate()`.)

For each iteration `iter` ≥ 1:

```
1. checkConvergence()
   - if hidden-variable changes < smallest_changes * 0.97 for 2 iters
     AND nr_iter_wo_resol_gain ≥ 2 AND has_high_fsc_at_limit:
     do_join_random_halves := true
     do_use_all_data := true

2. expectation()           # E-step
   for each particle:
       precompute: Fimg_shifted (masked), Fctf, Minvsigma2 = 1/(σ²_fudge·σ²_noise[shell])
       PASS 0 (coarse):
           for each (idir, ipsi, itrans) at coarse healpix grid:
               Frefctf := CTF · slice(reference, rot)
               diff2   := (1/2) · sum_{ires>cs} |X[ires]|²        # high-freq residual
                        + (1/2) · sum_{n in cs} Minvsigma2[n] · |Frefctf[n] − Fimg_shift[n]|²
               weight  := pdf_orient · pdf_offset · exp(min_diff2 − diff2)
           prune by adaptive_fraction (= 0.999) cumulative weight → significant set
       PASS 1 (fine, on significant):
           for each significant (idir, ipsi, itrans):
               oversample to 2^L · 2^L sub-(rot, trans)
               recompute diff2 → weight at fine grid
       storeWeightedSums:
           for each significant fine (rot, trans):
               wsum_BPref.data[n]    += weight · conj(Fimg_shift[n]) · CTF[n] / σ²_noise[shell]
               wsum_BPref.weight[n]  += weight · CTF[n]² / σ²_noise[shell]
               wsum_sigma2_noise[shell] += weight · |Frefctf[n] − Fimg_shift[n]|² / σ²_noise[shell]²
               wsum_sigma2_offset    += weight · ‖translation − prior‖²
               wsum_pdf_class[c]     += weight     # not used in single-class
               wsum_pdf_direction[c, d] += weight  # not used in global search w/o prior
               wsum_avg_norm_correction += weight · ‖Fimg_nomask‖² / ‖Frefctf‖²
               wsum_signal_product[grp] += weight · sum(conj(Fimg_nomask) · Frefctf / σ²_noise)
               wsum_reference_power[grp] += weight · sum(|Frefctf|² / σ²_noise)
           track exp_max_weight per particle = METADATA_PMAX

3. monitorHiddenVariableChanges()
   - per particle: angular distance from previous iter's best assignment
     (sampling.calculateAngularDistance, NOT raw matrix distance)
   - per particle: L2 offset distance from previous iter's best translation
   - aggregate → mean angular change, RMS offset change, fraction-class-changed

4. maximization()          # M-step
   for each class (1 in our case):
       BPref.reconstruct(...) :
           for each Fourier voxel n:
               Vref[n] := data[n] / (weight[n] + 1/(τ²_fudge · σ²_noise[shell(n)]))
       BPref.updateSSNRarrays() :
           σ²_recon[shell] := num_pixels[shell] / sum_{n in shell}(weight[n])
           myfsc            := clamp(fsc_halves[shell], 0.001, 0.999)
           myssnr           := τ²_fudge · myfsc / (1 − myfsc)
           τ²[shell]        := myssnr · σ²_recon[shell]
           data_vs_prior[shell] := (sum_{n in shell} weight[n]) / (τ²[shell] · num_pixels[shell])

   maximizationOtherParameters():
       # All updates use a momentum factor μ ≈ 0 (i.e., no smoothing in single-particle)
       σ²_noise[group][shell] := μ · old + (1−μ) · wsum_sigma2_noise[shell] / sum_weight
       σ²_offset             := μ · old + (1−μ) · wsum_sigma2_offset / (data_dim · sum_weight)
       scale_correction[grp] := μ · old + (1−μ) · wsum_signal_product[grp] / wsum_reference_power[grp]
                                clamped to [median/5, 5·median] then normalized to mean=1
       pdf_class[c]          := (1−μ) · wsum_pdf_class[c] / sum_weight
       pdf_direction[c, d]   := (1−μ) · wsum_pdf_direction[c, d] / sum_weight
       avg_norm_correction   := (1−μ) · wsum_avg_norm_correction / sum_weight

5. updateCurrentResolution()
   for ires = 1, 2, ...:
       if data_vs_prior[ires] < 1.0:
           maxres := ires − 1
           break
   # high-res recheck for split-half auto_refine:
   for ires2 = ori_size/2−1 down to maxres:
       if data_vs_prior[ires2] > 1.0 and ires2 > maxres + 3:
           maxres := ires2
           break
   current_resolution := getResolution(maxres)
   if current_resolution ≤ best_resolution_so_far + 0.0001:
       nr_iter_wo_resol_gain++
   else:
       nr_iter_wo_resol_gain := 0

6. updateImageSizeAndResolutionPointers()
   maxres := getPixelFromResolution(current_resolution)
   if ave_Pmax > 0.1 and has_high_fsc_at_limit:
       maxres += round(0.25 · ori_size / 2)        # aggressive growth
   else:
       maxres += incr_size                          # default 10 shells, bumped to fsc0143-fsc05+5
   current_size := min(2 · maxres, ori_size)
   image_coarse_size[grp] := from particle_diameter + angular sampling formula

7. compareTwoHalves()      # gold-standard FSC
   fsc_halves[shell] := pearson(Fhalf1[shell], Fhalf2[shell])

8. (between expectation and maximization, after both halves' wsum_BPref are accumulated:)
   joinTwoHalvesAtLowResolution():
       myres := max(low_resol_join_halves_Å, 1/current_resolution)
       lowres_r_max := ceil(ori_size · pixel_size / myres)
       avg_data := (data_h1 + data_h2)/2 in shells [0, lowres_r_max]
       avg_weight := (weight_h1 + weight_h2)/2 in shells [0, lowres_r_max]
       data_h1[shell] := avg_data[shell] for shell ≤ lowres_r_max
       data_h2[shell] := avg_data[shell] for shell ≤ lowres_r_max
       (likewise for weight)
```

That's the canonical loop. Everything in v3 below is in service of
matching this exactly.

## ★ Verified-done items — DO NOT re-audit (2026-04-08 evening)

After re-reading the actual code in
`/scratch/gpfs/GILLES/mg6942/recovar_relion_parity_audit/` against
RELION 5.0.1 source line by line, the following items the original
audit (and the v2 plan) flagged as "MISSING" or "PARTIAL" are
**actually fully implemented** in the current `_refine_relion_mode`.
The earlier audit was based on stale info, conflated the legacy and
relion modes, or both.

| Item | Verdict | Where in worktree | RELION ref | Verification |
|---|---|---|---|---|
| **A2: τ² from real reconstruction weights** | ✅ DONE | `refine.py:2006` calls `compute_relion_prior_from_reconstruction_stats(Ft_ctf_0, Ft_ctf_1, Ft_y_0, Ft_y_1, ...)` | `backprojector.cpp:1096-1125` | The audit was looking at the legacy-mode path; relion-mode already uses real weights |
| **A5: padding factor 2 + zero_pad_fourier_volume** | ✅ DONE | `refine.py:1241 PADDING_FACTOR = 2`; `relion_functions.zero_pad_fourier_volume` | RELION default `--pad 2` | Standalone test: 8³ → 16³ → crop, DC at center, round-trip CC = 1.0. The "known geometry bug" mentioned in earlier audit was fixed at some prior commit |
| **A1: data_vs_prior resolution criterion** | ✅ DONE (equivalent) | `refine.py:2018-2024`: `dvp_iter = fsc_to_relion_ssnr(fsc); resolution_from_data_vs_prior(dvp)` | `ml_optimiser.cpp:5600-5648` | recovar uses `SSNR<1` ⇔ `FSC<0.5`, equivalent to RELION's `data_vs_prior<1` to leading order. The exact RELION formula has an extra `(sum_weight/Npix)²` factor that recovar omits, but it's a small second-order correction |
| **`--low_resol_join_halves` in recovar M-step** | ✅ DONE | `refine.py:1800-1807` calls `regularization.join_halves_at_low_resolution`; helper at `regularization.py:812` | `ml_optimiser_mpi.cpp:3112-3219` | Implemented commit `ebfd2a9`. **Empirically verified NOT to be the dominant cause** of the recovar-vs-RELION FSC gap on the 5k benchmark. With/without join: ΔPmax < 0.0001 per iter, identical `current_size` growth. The joining radius is shell 14 (40 Å) but recovar's FSC drops at shell 21, well past the joining region. The fix is correct and matches RELION; just don't expect it to close the convergence-speed gap on this benchmark. |
| **A4: posterior-weighted σ²_noise (formula)** | ✅ DONE (formula matches) | `engine_v2._compute_noise_block` accumulates the A2−2·XA decomposition, plus per-image P_img; `noise.normalize_wsum_to_sigma2_noise` divides by `2·sumw·Npix_per_shell` | `ml_optimiser.cpp:8634-8638` (`wsum += weight·\|residual\|²`); `5268-5270` (`σ² = wsum/(2·sumw·Npix)`) | Mathematically equivalent: recovar's `A2 − 2·XA + P_img = E_w[\|CTF·proj − img\|²]` is exactly the expected per-pose squared residual, summed over the same posterior weights RELION uses. The maximization formula matches exactly. (Open: are recovar's accumulators using **masked** images per RELION line 8633 "Use FT of masked image for noise estimation!"? See B1 below — the noise accumulator code path uses `summed_masked` from the M-step GEMM, suggesting masking IS used; needs explicit verification.) |

**Implications for the remaining work:**
1. We do **NOT** need to "wire `data_vs_prior`" — already wired.
2. We do **NOT** need to "fix the `zero_pad_fourier_volume` geometry
   bug" — it works. PADDING_FACTOR = 2 is enabled.
3. We do **NOT** need to "use real reconstruction weights for τ²" —
   already done.
4. We do **NOT** need to "implement posterior-weighted noise" — already
   done; formula matches RELION exactly.

**Convergence-speed gap status**: with all the verified-done items
above, recovar's iter-by-iter trajectory STILL lags RELION on the
2026-04-08 5k normalized benchmark. The remaining gap at shells 14–25
must come from one of the items below (in best-guess priority):
- **B1**: masked-alignment / unmasked-reconstruction split
- **A3**: `tau²_fudge` parameter passthrough
- **B2**: `highres_Xi2` high-freq term (non-zero per-particle constant
  added to chi² for shells above current_size)
- A more subtle inference difference (`sigma2_fudge` constant in chi²,
  `pdf_orientation` / `pdf_offset` prior factors, adaptive-oversampling
  quantization, ...)

## Where recovar is, after the 2026-04-08 fixes

Audit verdict (full report in
`plan_relion_parity_v3_recovar_audit.md`):

| Component | Status | Notes |
|---|---|---|
| `--ctf` flag | ✅ added to all 4 RELION slurm scripts | commit `823f79d` |
| `--firstiter_cc` flag | ✅ added | prior commit |
| `--flatten_solvent`, `--zero_mask`, `--low_resol_join_halves 40`, `--norm`, `--scale` | ✅ added | commit `823f79d` |
| Per-particle simulator normalization | ✅ opt-in `relion_normalize=True` | commit `400f148` |
| Vmap signature mismatch on dev branch | ✅ fixed | commit `2cd750d` |
| `low_resol_join_halves` in recovar's M-step | ✅ implemented | commit `ebfd2a9` (this session) |
| Per-iter `noise_radial` / `tau2_radial` dump | ✅ instrumented | commit `823f79d` |
| Per-iter parity diff script | ✅ exists | `scripts/diff_relion_recovar_per_iter.py` |
| `data_vs_prior` resolution criterion | ❌ MISSING (helpers exist, not wired) | Phase A1 below |
| `tau²` from actual reconstruction weights | 🟡 PARTIAL (helper exists, wrong default) | Phase A2 |
| `tau²_fudge` in Wiener inv-tau | ❌ MISSING parameter passthrough | Phase A3 |
| Posterior-weighted `σ²_noise` from M-step accumulator | 🟡 PARTIAL (NoiseStats exists, formula maybe slightly off) | Phase A4 |
| Padding factor 2 for the Wiener solve | ❌ DISABLED (`PADDING_FACTOR=1` due to bug in `zero_pad_fourier_volume`) | Phase A5 |
| Masked-alignment / unmasked-reconstruction split | ❌ MISSING | Phase B1 |
| `exp_highres_Xi2_img` (high-freq residual term) | ❌ MISSING | Phase B2 |
| `monitorHiddenVariableChanges` (per-iter changes_orientations / changes_offsets) | 🟡 PARTIAL | Phase B3 |
| Convergence criterion (sticky smallest_changes + criterion) | 🟡 PARTIAL | Phase B4 |
| Final joined-halves iteration | ❌ MISSING | Phase B5 |
| `σ²_offset` per-iter update from data | ❌ MISSING (hardcoded prior) | Phase C1 |
| `scale_correction[group]` per-iter update | ❌ MISSING (no-op for single-optics) | Phase C2 |
| `pdf_orientation` (orientation prior) | ❌ flat | Phase C3 |
| FFT normalization convention vs RELION (~10⁸ unit gap) | ❌ different | Phase D1 (cosmetic) |

## Phase A — Algorithmic gaps that materially affect convergence

These are the items that, fixed, will close the per-iter trajectory
gap on the 5k benchmark. Do them in order. After each one, re-run the
5k normalized parity benchmark and verify the per-iter diff narrows.

### Note on `low_resol_join_halves` (commit `ebfd2a9`)

We implemented this in this session and **verified empirically that it
is NOT the dominant cause of the convergence-speed gap on our 5k
benchmark**. The function works correctly (verified standalone and via
in-process logs), but on this dataset:

- recovar's iter-1 FSC drops below 0.5 at **shell 21** (= 26 Å)
- RELION's iter-1 FSC drops below 0.5 at **shell 26** (= 21 Å)
- The joining radius is shell 14 (= 40 Å)

Joining shells 0–14 doesn't change `first_shell_below_0.5` because
that drop happens at shell 21 (well past the joining radius). The
trajectory is essentially unchanged: with-join iter-2 cs = 60, iter-3
cs = 60 (same as without-join). Pmax differs by < 0.0001 per iter.

The actual gap is at **shells 14–25**, where recovar's FSC falls off
faster than RELION's. That has to come from one of the OTHER
algorithmic differences below (most likely A2/A3/A4/A5, ranked by
impact best-guess).

The fix is still correct and is needed for full RELION parity (it
prevents pathological half-set divergence at very low resolution on
real data with worse SNR). But the priority for closing the
convergence-speed gap should be the items below, not this one.

### A1. Wire `data_vs_prior` resolution criterion into the loop ★ HIGHEST IMPACT

**What RELION does**: `updateCurrentResolution()` at
`ml_optimiser.cpp:5600-5648` finds the first shell where
`data_vs_prior < 1.0`, subtracts one shell for safety, then high-res
rechecks. `data_vs_prior` itself comes from
`backprojector.cpp::updateSSNRarrays()` at line 1162-1189:

```
data_vs_prior[ires] := (sum_{n in shell ires} weight[n]) / (tau2[ires] · num_pixels[ires])
```

This is **NOT FSC < 0.143** — it's **FSC < 0.5** in the equivalent
SSNR formulation, computed from the actual reconstruction weights
(`Ft_ctf` accumulator), not from a half-map FSC.

**What recovar does**: `_refine_relion_mode` at
`refine.py:~1320` computes
`data_vs_prior_iter = fsc_to_relion_ssnr(fsc_prev)` from the previous
iter's FSC, but the current_size growth heuristic is fed pixel-shell
values from this. The helpers `compute_data_vs_prior` and
`resolution_from_data_vs_prior` exist in `regularization.py:586,624`
but the data_vs_prior is derived from FSC, not from `Ft_ctf / tau²`
as RELION does.

**Fix**:
1. After the M-step (per half), call
   `compute_data_vs_prior(Ft_ctf_combined, tau2, volume_shape, padding_factor)`
   using the actual `Ft_ctf` accumulator (combined across halves) and
   the current `tau2` shell array.
2. Use that as the input to `resolution_from_data_vs_prior`.
3. Verify the resolution shell drives `compute_current_size_relion`
   correctly.

**Files**: `recovar/em/dense_single_volume/refine.py`,
`recovar/reconstruction/regularization.py`

**Verification**: per-iter `current_resolution` (Å) should match
RELION's `_rlnCurrentResolution` from `run_itNNN_half1_model.star`
within 1 shell.

**Effort**: 1 day.

### A2. Use `compute_relion_prior_from_reconstruction_stats` as the default tau² path

**What RELION does** (`backprojector.cpp:1096-1125`): tau² is computed
from the gold-standard half-FSC AND the actual reconstruction noise:

```
σ²_recon[ires] := num_pixels[ires] / sum_{n in shell}(weight[n])     # from Ft_ctf accumulator
myfsc          := clamp(fsc[ires], 0.001, 0.999)
myssnr         := tau2_fudge · myfsc / (1 − myfsc)
tau²[ires]     := myssnr · σ²_recon[ires]
```

The key is that `σ²_recon` is derived from `weight[n]` (the
backprojection weight, which IS recovar's `Ft_ctf` accumulator), NOT
from a separate gridding-count surrogate.

**What recovar does**: there are TWO functions —
`compute_fsc_prior_gpu` (which uses a gridding-count surrogate
denominator, an APPROXIMATION) and
`compute_relion_prior_from_reconstruction_stats` (which uses the actual
`Ft_ctf` accumulator, EXACT). Currently the loop calls
`compute_fsc_prior_gpu` (the surrogate path) by default; the audit
identified this as a Phase C2 gap.

**Fix**: switch the default to
`compute_relion_prior_from_reconstruction_stats`. The function already
exists; just rewire the call site.

**Files**: `recovar/em/dense_single_volume/refine.py:~1965`

**Verification**: per-iter `tau²[shell]` should match RELION's
`_rlnReferenceTau2` from `model_class_1` within ~5%.

**Effort**: 1 day.

### A3. Pass `tau²_fudge` parameter through the Wiener solve

**What RELION does**: at `backprojector.cpp:1379-1800`, the Wiener
filter uses `1/(tau2_fudge · sigma2_noise[ires])` as the regularizer.
`tau2_fudge` is a user-tunable strength (default 1.0).

**What recovar does**:
`relion_functions.adjust_regularization_relion_style` accepts
`tau2_fudge` but the call site at `_refine_relion_mode` doesn't pass
it through.

**Fix**: wire the `tau2_fudge` parameter through the call chain. Trivial.

**Files**: `recovar/em/dense_single_volume/refine.py` and the
`post_process_from_filter_v2` call.

**Verification**: with `tau2_fudge=1.0` (the default), no difference;
with `tau2_fudge=4.0`, the reconstructed volume should be smoother
(stronger prior).

**Effort**: 1 day.

### A4. Posterior-weighted σ²_noise from the M-step accumulator (with momentum)

**What RELION does**: in `storeWeightedSums()`
(`ml_optimiser.cpp:8531-8750`), inside the per-(rot, trans) loop:

```
wsum_sigma2_noise[shell] += weight · |Frefctf[n] − Fimg_shift[n]|² / σ²_noise[shell]²
```

Then after all particles processed, in
`maximizationOtherParameters()`:

```
σ²_noise[group][shell] := μ · old + (1−μ) · wsum_sigma2_noise[shell] / sum_weight
```

The momentum factor μ is essentially 0 in standard refine (no
smoothing). But the formula includes BOTH halves, BOTH passes
(coarse + fine), and ALL particles. And the residual is from the
MASKED Fimg.

**What recovar does**: `engine_v2.py` already accumulates a
`NoiseStats` object with `wsum_sigma2_noise`, but:
- It only uses pass-2 (the fine) accumulator
- The masking choice is unclear
- The post-loop normalization in `noise.normalize_wsum_to_sigma2_noise`
  may differ from RELION's exact formula

**Fix**: cross-check `engine_v2._compute_noise_block` and
`noise.normalize_wsum_to_sigma2_noise` line-by-line against RELION's
code. Specifically:
1. The residual should be `Frefctf − Fimg_shift_masked` (not
   `Frefctf − Fimg_shift_unmasked`).
2. The accumulator should be over BOTH halves (or per-half, then
   combined).
3. The normalization should be `wsum / sum_weight`, where `sum_weight`
   is the count of significant samples × per-particle weight sum (RELION
   typically gets this from `wsum_BPref.weight.sum()`).

**Files**: `recovar/em/dense_single_volume/engine_v2.py`,
`recovar/reconstruction/noise.py`

**Verification**: per-iter `σ²_noise[shell]` should match RELION's
`_rlnSigma2Noise` (from `model_optics_group_1`) within 5%, after
accounting for the FFT normalization convention difference.

**Effort**: 2-3 days. Most of this is reading the C++ carefully and
matching the constants.

### A5. Re-enable `padding_factor=2` for the Wiener solve

**What RELION does**: defaults to `--pad 2`, meaning the
backprojection accumulators are on a `(2N)³` Fourier grid. The Wiener
solve happens on this padded grid; the volume is then cropped back to
`N³` in real space. This reduces interpolation artifacts and is
critical for the final FSC shape at high resolution.

**What recovar does**: `PADDING_FACTOR = 1` is hardcoded in
`_refine_relion_mode` because `relion_functions.zero_pad_fourier_volume`
has a known geometry bug. Effectively recovar runs at half RELION's
internal resolution.

**Fix**: debug `zero_pad_fourier_volume`. The issue is in the
frequency-to-padded-index mapping. Fix the geometry so that when you
zero-pad an N³ centered Fourier volume into a (2N)³ centered
volume, the DC stays at the center and shells map correctly.

**Files**: `recovar/reconstruction/relion_functions.py:24` (the
`zero_pad_fourier_volume` function).

**Verification**: round-trip a known volume. CC between original and
`unpad(pad(vol))` should be > 0.999.

**Effort**: 1-2 days. May be tricky.

## Phase B — Per-iter state and convergence (B depends on A being right)

### B1. Implement masked-alignment / unmasked-reconstruction split

**What RELION does** (`ml_optimiser.cpp:getFourierTransformsAndCtfs`,
~line 5840): for each particle, computes TWO Fourier images:
- `Fimg`: from the particle masked by a soft circular mask of radius
  `particle_diameter/2` (cosine edge of width 2 px). This is what
  enters the E-step diff² and the noise accumulator.
- `Fimg_nomask`: from the unmasked (background-subtracted) particle.
  This is what enters the M-step backprojection (`Pᵀ y`).

**What recovar does**: `_preprocess_batch` in `engine_v2.py:79` uses
`apply_image_mask=False` everywhere. No mask is applied, ever.

**Fix**: thread two image preprocessing paths through the engine —
masked for the score computation and noise accumulator, unmasked for
the M-step backprojection. The dataset already supports an
`apply_image_mask` flag.

**Files**: `recovar/em/dense_single_volume/engine_v2.py`,
`recovar/em/dense_single_volume/refine.py` (call site that decides
mask/no-mask for each pass).

**Verification**: with `particle_diameter == box_size` (our 5k
synthetic case), the effect is small but should be a small Pmax sharpening
(less background noise contributing to the chi²). With smaller
particle diameters (real data), the effect is large.

**Effort**: 2-3 days.

### B2. `exp_highres_Xi2_img` high-freq residual term

**What RELION does**: per-particle, computes the image power above
the current_size cutoff:

```
exp_highres_Xi2_img[i] := sum_{ires > current_size/2} |Fimg[i, ires]|² / sigma²_noise[ires]
```

This constant is added to every (rot, trans) candidate's diff² for
particle `i`. It doesn't change the relative ordering of poses
(since it's the same constant for all candidates of one particle), so
it doesn't affect Pmax or the M-step. But it DOES affect:
- Absolute log-likelihood per particle
- Per-iter `_rlnLogLikelihood` (the optimizer's running total)
- The numerical interpretation of `exp_min_diff2` for the
  log-sum-exp trick

**What recovar does**: this term is missing entirely. Recovar's diff²
only includes the windowed pixels.

**Fix**: precompute `highres_Xi2[i]` per particle from the full-spectrum
masked image power, and add it to the diff² inside `_e_step_block_scores`
(or add it to the per-particle `batch_norm` constant that's already
tracked).

**Files**: `recovar/em/dense_single_volume/engine_v2.py` (preprocessing
and score functions).

**Verification**: per-iter `_rlnLogLikelihood` from RELION's
`model_general` should match recovar's running log-likelihood within
relative error < 1%.

**Effort**: 1-2 days. Mostly bookkeeping.

### B3. Per-iter `changes_orientations` / `changes_offsets` tracking

**What RELION does** (`monitorHiddenVariableChanges`,
`ml_optimiser.cpp:9157-9242`):
- For each particle: angular distance via
  `sampling.calculateAngularDistance(new_rot, new_tilt, new_psi,
  old_rot, old_tilt, old_psi)`. This is the Euler angle metric, NOT
  rotation-matrix Frobenius distance.
- Per-particle offset: L2 distance between new and old translation.
- Aggregate: `mean angular change`, `RMS offset change`,
  `fraction class change` (n/a for single-class).
- Sticky tracker: `smallest_changes_optimal_orientations =
  min(current_changes, smallest_so_far)`, etc.

**What recovar does**: `convergence.py::update_refinement_state` has
some tracking but it's a simplified version, not the exact RELION
formula.

**Fix**: implement the exact formula. Add a per-particle store of the
previous iter's best (rot, trans), then compute angular and L2
distances per particle, aggregate.

**Files**: `recovar/em/dense_single_volume/convergence.py`,
`recovar/em/dense_single_volume/refine.py`.

**Verification**: `changes_orientations` and `changes_offsets` per
iter should match RELION's `rlnSmallestChangesOptimalOrientations`
and `rlnSmallestChangesOptimalOffsets` within 5%.

**Effort**: 2-3 days.

### B4. Convergence criterion (uses B3 and A1)

**What RELION does** (`ml_optimiser.cpp:9244-9288`):

```
if  (new_angle_changes >= 0.97 * smallest_angle_changes_thus_far)
and (ratio_trans_changes < 0.40 or new_offset_changes >= 0.97 * smallest_offset_changes_thus_far)
and (ratio_orient_changes < 0.40 or new_angle_changes >= 0.97 * smallest_angle_changes_thus_far):
    nr_iter_wo_large_hidden_variable_changes++
else:
    nr_iter_wo_large_hidden_variable_changes := 0

# Plus the resolution-gain check at ml_optimiser.cpp:5658-5675:
if current_resolution ≤ best_resolution + 0.0001:
    nr_iter_wo_resol_gain++
else:
    nr_iter_wo_resol_gain := 0

# checkConvergence at ml_optimiser.cpp:10135-10204 (precise quote needed):
has_converged := (nr_iter_wo_resol_gain ≥ 2)
              and (nr_iter_wo_large_hidden_variable_changes ≥ 2)
              and (has_high_fsc_at_limit)
              and (sampling order has stabilized)
```

**Fix**: implement exact criterion in
`convergence.py::check_convergence`.

**Files**: `recovar/em/dense_single_volume/convergence.py`.

**Verification**: termination iteration should match RELION ±1.

**Effort**: 1-2 days after B3 is in.

### B5. Final joined-halves iteration after convergence

**What RELION does**: when `has_converged := True`, the next
iteration runs with `do_join_random_halves := True` and
`do_use_all_data := True`, meaning:
- `current_size := ori_size` (use Nyquist)
- Both halves' particles contribute to a single reconstruction

**What recovar does**: terminates at `max_iter`. No final joined
iteration.

**Fix**: after `check_convergence` returns True, run one more
iteration with `current_size := grid_size` and all particles in a
single E+M sweep (or continue per-half then average).

**Files**: `recovar/em/dense_single_volume/refine.py`.

**Verification**: final volume FSC should match RELION's
`run_class001.mrc` within CC > 0.99.

**Effort**: 2-3 days.

## Phase C — Cosmetic / lower-priority

### C1. `σ²_offset` translation prior update from data

**What RELION does**: `mymodel.sigma2_offset` is updated each iter
from `wsum_sigma2_offset / (data_dim · sum_weight)` (with a momentum
factor). It governs the Gaussian translation prior in the next iter.

**What recovar does**: hardcoded from `init_translation_sigma_angstrom`,
never updated.

**Fix**: accumulate `wsum_sigma2_offset` in the M-step, divide by
`(data_dim · sum_weight)` after both halves, store, use in next iter's
translation prior.

**Effort**: 1-2 days.

### C2. `scale_correction[group]` per-optics-group update

**What RELION does**: per-iter, per-group, updates a scalar that
multiplies the per-image Fimg in the chi² and the M-step. Important
for multi-optics-group (multi-session) data.

**What recovar does**: no-op (single optics group assumption).

**Fix**: needed only for real-data benchmarks. For synthetic, skip.

**Effort**: 1-2 days.

### C3. `pdf_orientation` learned prior

**What RELION does**: tracks `pdf_direction[iclass][idir]` as an
empirical prior over directions, updated each iter. Used in the next
iter's E-step to weight the rotation grid (favors directions with
more particles).

**What recovar does**: flat prior over orientations.

**Fix**: accumulate `wsum_pdf_direction[direction] += sum(weight per
particle whose best direction is this one)`, normalize, use as
log-prior in the next iter.

**Effort**: 2-3 days.

## Phase D — Unit conventions (cosmetic, last)

### D1. FFT normalization convention

**What we observed**: per-shell σ²_noise and τ² differ by ~10⁸ (≈
N² for N=128) between recovar and RELION. This is an FFT normalization
convention difference. Both pipelines are internally self-consistent
(inference is correct on each), so this doesn't affect the FSC,
Pmax, or final volume — but it makes the per-shell numbers in the
diff script look incomparable.

**Fix options**:
- (a) Add a unit-conversion factor to `diff_relion_recovar_per_iter.py`
  so the numbers print on the same scale.
- (b) Standardize on RELION's convention internally in recovar (large
  refactor).

Option (a) is 1 day. Option (b) is 1-2 weeks.

**Recommendation**: option (a). The internal convention isn't worth a
full refactor.

## Validation infrastructure

### V1. Per-iter parity diff script ✅ DONE
`scripts/diff_relion_recovar_per_iter.py`

### V2. Parity benchmark harness — TODO

Single-command end-to-end:
1. Generate small benchmark (1-2k particles, small box) with
   `relion_normalize=True`
2. Run RELION 5-10 iters with all 7 GUI flags
3. Run recovar 5-10 iters with same params + matching half-set split
4. Run diff script
5. Assertion table:
   | Field | Tolerance | Iters checked |
   |---|---|---|
   | `current_size` | exact (±1 shell) | all |
   | `ave_Pmax` | < 10% relative | iter 2+ (skip iter 1 binarization) |
   | `current_resolution` | < 5% relative | all |
   | `_rlnReferenceTau2` per shell | < 5% relative | iter 2+, shells 1..current_size/2 |
   | `_rlnSigma2Noise` per shell | < 10% relative | iter 2+ (after unit normalization) |
   | `data_vs_prior` per shell | < 5% relative | iter 2+ |
   | `_rlnGoldStandardFsc` per shell | < 2% absolute | iter 2+ |
   | `changes_orientations` | < 10% relative | iter 2+ |
   | `changes_offsets` | < 10% relative | iter 2+ |
   | termination iter | ±1 | end |
   | final volume CC vs RELION | > 0.99 | end |

**Files**: `scripts/run_relion_parity_benchmark.py` (to be written).

**Effort**: 2-3 days.

### V3. Regression test

A pytest that runs V2 on a tiny dataset (256 particles, 32-px box) and
asserts a subset of the parity table. Runs in <5 min on a single GPU.
Prevents future regressions.

**Files**: `tests/integration/test_relion_parity_per_iter.py` (to be
written).

**Effort**: 1 day.

## Phase ordering (dependencies)

```
A1 ──► A2 ──► A3 ──► A4 ──► A5    (A is sequential; each feeds into the next)
                            │
                            ▼
                  Per-iter trajectory should match RELION on the 5k benchmark
                            │
                            ▼
                            B1 ──► B2 ──► B3 ──► B4 ──► B5
                            │
                            ▼
                  Convergence + final joined-halves iter
                            │
                            ▼
                            C1 ──► C2 ──► C3
                            │
                            ▼
                  Real-data multi-optics support
                            │
                            ▼
                            D1
                            │
                            ▼
                  Diff script reads same units → easier debugging

V2 (parity benchmark) can be built in parallel with anything in A.
V3 (regression test) follows V2.
```

## Estimated effort (single dev)

| Phase | Items | Effort |
|---|---|---|
| A | 5 items | 1 + 1 + 1 + 2.5 + 1.5 = **7 days** |
| B | 5 items | 2.5 + 1.5 + 2.5 + 1.5 + 2.5 = **10.5 days** |
| C | 3 items | 1.5 + 1.5 + 2.5 = **5.5 days** |
| D | 1 item | **1 day** |
| V | 3 items | 0 + 2.5 + 1 = **3.5 days** |
| **TOTAL** | **17 items** | **27.5 days ≈ 5-6 weeks** |

## Rules for the implementing agent

1. **Run the per-iter diff after every change** — `scripts/diff_relion_recovar_per_iter.py`. Don't ship a phase without confirming it brings recovar closer to RELION on the 5k normalized benchmark.
2. **Don't add hard-CC iter-1 to recovar** to "match" RELION's binarized Pmax = 1.0. That's a RELION quirk (scale-absorption hack), not real inference. See `feedback_relion_iter1_hard_cc_is_not_parity_bug.md`.
3. **Default `relion_normalize=False`** in the simulator. Only opt-in for parity benchmarks; don't force normalization onto existing recovar tests/datasets.
4. **No long-test cycles on this branch** — see `feedback_em_branch_skips_long_test_policy.md`. Push intermediate commits at iteration speed; the per-iter diff is the test signal that matters.
5. **Always run RELION with all 7 GUI flags** (`--ctf`, `--firstiter_cc`, `--flatten_solvent`, `--zero_mask`, `--low_resol_join_halves 40`, `--norm`, `--scale`). See `feedback_relion_required_flags.md`.
6. **Always normalize the simulator output** with `--relion-normalize` for the parity benchmark. This makes RELION's iter-1 Bayesian E-step work without `--firstiter_cc` (so iter-1 Pmax is comparable across the two pipelines).
7. **Every commit's message should cite the file:line in RELION source it's matching**. The audit reports include exact citations.

## What to do RIGHT NOW

1. **Wait for the recovar 5k normalized run with low_resol_join_halves
   to finish** (in flight, ETA ~10 min). Verify per-iter diff narrows.
2. **Start Phase A1 (data_vs_prior wire-up)**. Helpers exist; just
   plumbing.
3. **Don't touch Phase B/C/D until A is fully done and verified**.

## References

- `relion/src/ml_optimiser.cpp` (the canonical algorithm)
- `relion/src/ml_optimiser_mpi.cpp` (MPI driver, half-set joining,
  convergence)
- `relion/src/backprojector.cpp` (Wiener filter, tau² update,
  data_vs_prior)
- `recovar/em/dense_single_volume/refine.py` (`_refine_relion_mode`)
- `recovar/em/dense_single_volume/engine_v2.py` (E+M kernels)
- `recovar/reconstruction/regularization.py` (resolution criterion,
  prior, growth heuristic, low_resol_join_halves)
- `recovar/reconstruction/noise.py` (noise normalization)
- `recovar/em/CLAUDE.md` (developer guide with required-flags table)
- `~/.claude/projects/-home-mg6942/memory/feedback_relion_*` (the
  forensic write-ups of every trap we hit)
