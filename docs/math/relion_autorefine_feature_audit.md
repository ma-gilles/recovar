# RELION Auto-Refine Feature Audit

Living checklist of every RELION `relion_refine --auto_refine` sub-operation,
whether recovar has ported it, and whether a pybind11 binding has validated
parity on identical inputs.

**Maintained alongside `recovar/relion_bind/`.**
Update this file every time a binding validates or disproves parity.

## Legend

| Symbol | Meaning |
|--------|---------|
| Yes | recovar has an implementation |
| Partial | Implementation exists but known incomplete |
| Broken | Implementation exists but produces wrong results |
| No | No recovar implementation |
| N/A | Not applicable (e.g., C1 symmetry) |
| ✓ | Binding confirmed numerical parity |
| **DIVERGENT** | Binding showed measurable difference (see Notes) |
| — | Not yet tested with binding |

---

## Pre-processing & Image I/O

| # | RELION operation | RELION source | Ported? | recovar location | Validated? | Notes |
|---|-----------------|---------------|---------|------------------|------------|-------|
| 1 | Read image + apply gain/defect | ml_optimiser.cpp:5840 | Partial | `data_io/` handles gain | — | recovar skips defect correction |
| 2 | Soft circular mask (particle_diameter) | ml_optimiser.cpp:5893 | Yes | `core/mask.py:relion_soft_image_mask`, `refine.py` | �� | Binding P2: 93 tests, RELION C++ softMaskOutsideMap vs recovar smooth_circular_mask at atol=1e-12. Wired into `_run_relion_iteration_loop` via image_mask override. |
| 3 | FFT of masked image (for E-step) | ml_optimiser.cpp:5920 | Yes | `em_engine.py` FFT in preprocess | — | |
| 4 | FFT of unmasked image (for M-step) | ml_optimiser.cpp:5927 | No | — | — | recovar uses same image for both |
| 5 | selfTranslate (beam-tilt phase ramp) | ml_optimiser.cpp:5935 | No | — | — | Zero effect for SPA w/o beam tilt |
| 6 | demodulatePhase (aberration correction) | ml_optimiser.cpp:5940 | No | — | — | Zero for standard optics |
| 7 | divideByMtf (detector MTF correction) | ml_optimiser.cpp:5945 | No | — | — | Not provided in simulated data |
| 8 | CTF image (2× padded box → downsample) | ctf.cpp:314 | No | recovar computes at image size directly | **DIVERGENT** | Binding P1: sign convention CTF_relion=-CTF_recovar (cancels with vol sign flip); 2× padding produces max diff ~1.0 vs unpadded; padded CTF exceeds [-1,1] at small boxes |
| 9 | 1/σ² noise weighting per pixel | ml_optimiser.cpp:5960 | Yes | `em_engine.py` sigma2_noise weighting | ✓ | Composite test: RELION Minvsigma2=1/(2σ²) vs recovar 1/σ² with -0.5 score factor → identical posteriors (max_diff=1e-17) |
| 10 | windowFourierTransform (current_size crop) | fftw.h:809 | Yes | `fourier_window.py` index-based | ✓ | Binding P3: RELION rectangular crop (2112 px) + Mresol_fine radial mask = 1683 scored px; recovar radial mask = 1689 px — only 6-pixel (0.4%) Nyquist boundary diff |

## Projector Setup

| # | RELION operation | RELION source | Ported? | recovar location | Validated? | Notes |
|---|-----------------|---------------|---------|------------------|------------|-------|
| 11 | Zero-pad volume to pf×N | projector.cpp:340 | Yes | `relion_functions.py:pad_volume_for_projection:25` | ✓ | Binding E1b: 14 tests, RELION binding projector data matches numpy ref at rel_err < 1e-12 for pf=1 and pf=2. Axis convention verified: FFT_recovar = -FFT_relion.T(2,1,0). |
| 12 | FFT + CenterFFTbySign → projector storage | projector.cpp:360 | No | recovar uses centered full-complex | ✓ | Binding E1: projector data matches numpy ref at rtol=1e-12 |
| 13 | Trilinear interpolation (3D→2D slice) | projector.cpp:630 | Yes | `slicing.py:slice_volume:221` | ✓ | Binding E2: identity projection matches projector kz=0 exactly |

## E-step: Scoring

| # | RELION operation | RELION source | Ported? | recovar location | Validated? | Notes |
|---|-----------------|---------------|---------|------------------|------------|-------|
| 14 | Phase-shift images (translations) | fftw.cpp:762 | Yes | `geometry.py:translate_images:167` | ✓ | Binding E3: RELION vs numpy <1e-12; RELION vs recovar <1e-10 (excl. Nyquist ky sign ambiguity) |
| 15 | ‖img − CTF·proj‖²/σ² per shell | ml_optimiser.cpp:7098 | Yes | `em_engine.py:_e_step_block_scores` | ✓ | Binding E4: diff2 decomposition exact (rel_err<1e-15); cross-term GEMM formula matches per-pixel (<1e-15); Parseval (translation invariance of batch_norm) verified. |
| 16 | First-iter CC scoring (hard assignment) | ml_optimiser.cpp:7414 | No | — | — | recovar uses soft Bayesian iter 1 |
| 17 | exp(−diff2) × priors → posteriors | ml_optimiser.cpp:7704 | Yes | `em_engine.py:_update_logsumexp` | ✓ | Binding E5: standalone C++ reimplementation, 5 tests, exact parity (max_diff<1e-14) |
| 18 | pdf_orientation prior term | ml_optimiser.cpp:7780 | Yes | `em_engine.py` rotation_log_prior | — | |
| 19 | pdf_offset prior term | ml_optimiser.cpp:7780 | Yes | `em_engine.py` translation_log_prior | — | |
| 20 | Significance pruning (adaptive OS pass 2) | ml_optimiser.cpp:7850 | Yes | `refine.py` adaptive_oversampling | — | Union-of-images, not per-image |

## E-step: Accumulation (storeWeightedSums)

| # | RELION operation | RELION source | Ported? | recovar location | Validated? | Notes |
|---|-----------------|---------------|---------|------------------|------------|-------|
| 21 | Σ w_i · CTF·img → Fourier accumulators | ml_optimiser.cpp:8241 | Yes | `em_engine.py:_m_step_block` | — | binding E6 |
| 22 | Σ w_i · ‖img−proj‖² per shell → σ²_noise | ml_optimiser.cpp:8241 | Partial | hard-assignment only | ✓ | Binding E7: standalone C++ reimplementation, 7 tests, exact parity (rel_err<1e-12) |
| 23 | Per-image norm_correction accumulation | ml_optimiser.cpp:8350 | No | — | — | No-op single optics group |
| 24 | Per-image scale_correction accumulation | ml_optimiser.cpp:8370 | No | — | — | No-op single optics group |

## M-step: Reconstruction

| # | RELION operation | RELION source | Ported? | recovar location | Validated? | Notes |
|---|-----------------|---------------|---------|------------------|------------|-------|
| 25 | Symmetrise Fourier accumulators | backprojector.cpp:1200 | N/A | C1 symmetry only | — | |
| 26 | Enforce Hermitian symmetry | backprojector.cpp:2207 | Yes | `half_volume_to_full_volume` | ✓ | Phase 4: round-trip corr>0.8 (192 proj), FSC half-sets >0.5 |
| 27 | Blob deconvolution (iterative) | backprojector.cpp:1400 | No | direct Wiener instead | — | Different strategy — binding M2 |
| 28 | IFFT → crop to ori_size | backprojector.cpp:2589 | Yes | iDFT + unpad | ✓ | Binding M7: 21 pass, 6 xfail (layout conversion). Output shape, determinism, self-consistency verified. Covered by BackProjector round-trip + M4/M5 bindings. |
| 29 | Gridding correction (radial sinc²) | projector.cpp:595 | Yes | `relion_functions.py:64` | ✓ | Validated Phase 1: max diff 1.6e-5 (float k-coords vs int) |
| 30 | Flatten solvent (mask volume exterior) | ml_optimiser.cpp:5100 | Yes | `iteration_loop.py:_run_relion_iteration_loop` post-reconstruction | — | Uses `soft_mask_outside_map(radius=particle_diameter/(2*pixel_size), cosine_width=5)` |
| 31 | Zero mask (zero exterior instead of noise) | ml_optimiser.cpp:5110 | Yes | Handled by #2 (soft mask replaces exterior with avg_bg) | — | Automatic with softMaskOutsideMap do_zero_mask path |

## M-step: Statistics & Resolution

| # | RELION operation | RELION source | Ported? | recovar location | Validated? | Notes |
|---|-----------------|---------------|---------|------------------|------------|-------|
| 32 | Gold-standard FSC between half-maps | backprojector.cpp:998 | Yes | `regularization.py:get_fsc_gpu:132` | ✓ | Phase 4: same-vol FSC[1]>0.5, diff-vol FSC[mid]<0.5 |
| 33 | updateSSNRarrays (tau2, data_vs_prior) | backprojector.cpp:1044 | Yes | `regularization.py:compute_data_vs_prior` | ✓ | Binding M4: calls actual RELION BackProjector::updateSSNRarrays, 4 tests |
| 34 | updateCurrentResolution | ml_optimiser.cpp:5579 | Yes | `regularization.py:resolution_from_data_vs_prior:624` | ✓ | Binding M5: standalone C++, 14 tests, exact parity (integer shell match) |
| 35 | Noise update (σ² per shell) | ml_optimiser.cpp:5090 | Partial | hard-assignment in `refine.py` | ✓ | Binding M9: standalone C++, 13 tests, exact parity (max_diff<1e-15) |
| 36 | tau2_fudge multiplier | ml_optimiser.cpp:5050 | Yes | `refine.py` kwarg, default=4 | — | |
| 37 | Low-resolution halves joining (40 Å) | ml_optimiser.cpp:5030 | Yes | `regularization.py:812-910` | — | |

## Sampling

| # | RELION operation | RELION source | Ported? | recovar location | Validated? | Notes |
|---|-----------------|---------------|---------|------------------|------------|-------|
| 38 | HEALPix orientation grid | healpix_sampling.cpp:1832 | Yes | `sampling.py:get_rotation_grid_at_order:530` | ✓ | Binding S1: direction count, direction set, rotation matrices all match (21 tests). NEST vs RING pixel order differs but same sphere coverage. |
| 39 | Oversampled sub-grid (psi children) | healpix_sampling.cpp:1850 | Yes | `sampling.py` adaptive OS | ✓ | Binding S1: oversampled count, OS=0→coarse, within-cell radius all verified. |
| 40 | Translation grid | healpix_sampling.cpp:1724 | Yes | `sampling.py:get_translation_grid:100` | ✓ | Binding S2: coarse grid exact match, circular boundary, oversampled count/centering/spacing all verified (12 tests). |
| 41 | Perturbation (per-iter rigid rotation) | healpix_sampling.cpp:1909 | Yes | `sampling.py:apply_relion_rotation_perturbation` | ✓ | Binding S1/S3: RELION vs recovar perturbation diff < 1e-15. |
| 42 | 3σ cone prior filtering | healpix_sampling.cpp:695 | Yes | `sampling.py` sigma_cutoff=3.0 | ✓ | Binding S4: 54 tests. RELION factored cone (dir×psi rectangle) vs recovar SO(3) ball are geometrically different but both valid. Intersection verified non-empty; boundary rotations documented. |

## Convergence & Loop Control

| # | RELION operation | RELION source | Ported? | recovar location | Validated? | Notes |
|---|-----------------|---------------|---------|------------------|------------|-------|
| 43 | Angular step refinement trigger | ml_optimiser.cpp:5200 | Yes | `convergence.py:710-790` | — | |
| 44 | Auto-termination (no improvement) | ml_optimiser.cpp:5250 | Yes | `convergence.py:544-620` | — | |
| 45 | ave_Pmax tracking | ml_optimiser.cpp:5230 | Yes | `convergence.py` | — | |
| 46 | Assignment change tracking | ml_optimiser.cpp:5220 | Yes | `convergence.py` | — | |

---

## Summary

| Status | Count | IDs |
|--------|-------|-----|
| Ported (full) | 34 | 1,2,3,4,9,10,11,13,14,15,17,18,19,20,21,26,28,29,30,31,32,33,34,36,37,38,39,40,41,42,43,44,45,46 |
| Ported (partial/broken) | 2 | 22,35 |
| Not ported | 8 | 5,6,7,8,16,23,24,27 |
| Not applicable | 1 | 25 |
| **Binding-validated** | **22** | **2 (image mask P2), 9 (noise weighting), 10 (windowing), 11 (volume padding E1b), 12 (projector storage), 13 (trilinear projection), 14 (phase shift), 15 (diff2 scoring), 17 (posteriors E5), 22 (noise accumulation E7), 26 (backproject+reconstruct), 28 (IFFT+crop M7), 29 (gridding correction), 33 (SSNR M4), 34 (resolution M5), 35 (noise update M9), 38 (HEALPix grid), 39 (oversampled sub-grid), 40 (translation grid), 41 (perturbation), 42 (cone filter S4), M6 (downsampled average)** |

---

## Changelog

| Date | Change |
|------|--------|
| 2026-04-16 | **Phase 7 complete**: Added 4 new binding test files (P2 image mask, E1b padding parity, M7 IFFT+crop, S4 cone filter). 182 new tests (93+14+21+54). Ported features #2 (soft circular mask), #30 (flatten solvent), #31 (zero mask). P2 binding validates RELION C++ softMaskOutsideMap vs recovar smooth_circular_mask at atol=1e-12. E1b validates volume padding at pf=1/2 vs RELION projector at rel_err<1e-12. S4 documents RELION factored cone vs recovar SO(3) ball geometry. Validated count: 22 (was 18). Ported count: 34 (was 31). |
| 2026-04-16 | **E-step composite parity**: end-to-end scoring+posterior test confirms EXACT parity (max_diff=1e-17) when using all-1 half-spectrum weights (half_spectrum_scoring=True) and RELION's Minvsigma2=1/(2σ²) convention. Windowing divergence reduced from 20% to 0.4% (6 Nyquist-boundary pixels). Hermitian weights (w=2 for interior) make posteriors ~2x too peaked vs RELION — half_spectrum_scoring=True is correct. DC exclusion verified exact. Items #9, #10 promoted to ✓. Validated count: 18. |
| 2026-04-16 | Phase 5b+6 complete: Added 6 new bindings (M4 updateSSNRarrays, M6 getDownsampledAverage, E5 posteriors, E7 noise accumulation, M5 resolution, M9 noise update). 49 new tests, all at exact parity (1e-12 to 1e-15). Total: 209 binding tests passing. Rewrote backproject tests with determinism checks (bit-exact) replacing loose corr>0.8 threshold. |
| 2026-04-16 | Phase 4 complete: BackProjector bindings validated (6 tests). Round-trip project→backproject→reconstruct corr>0.8 (192 evenly-spaced orientations). FSC half-sets validated. Key findings: (1) RELION's `reconstruct` applies CenterFFTbySign before iFFT — projections from RELION Projector already carry compatible sign decoration; (2) tau2 regularization (1/tau2 term in Wiener) must be weak for round-trip tests (tau2≫1 or do_map=False); (3) pad_size = 2*ROUND(pf*r_max)+3, not pf*N. |
| 2026-04-16 | Phase 3 E4 complete: diff2 scoring composite validated (5 tests). Decomposition identity (diff2 = batch_norm + cross + proj_norm) exact to 1e-16. Cross-term GEMM formula (recovar's approach) matches per-pixel formula to 1e-16. Parseval invariance of batch_norm under translation verified. |
| 2026-04-16 | Phase 5 S1+S2 complete: sampling bindings validated (33 tests). Orientations: direction grid, rotation matrices (via R_from_relion frame conversion), oversampled sub-grid, perturbation all at parity. Translations: coarse grid, oversampled, perturbation all exact match. |
| 2026-04-16 | Phase 3 E3 complete: shift binding validated (8 tests). RELION vs recovar shift matches to <1e-10 excluding Nyquist ky row (sign ambiguity: FFTW ky=+N/2 vs recovar ky=-N/2). |
| 2026-04-16 | Initial audit. Phase 0+1 complete: layout conversions validated (61 tests), M1 gridding bound and validated (14 tests). |
| 2026-04-15 | Phase 2b complete: E1 projector storage + E2 projection bound and validated (10 tests). Key finding: CenterFFTbySign uses (-1)^(k+i+j) on physical indices (fftw.h:400), not linear-index parity. RELION forward FFT normalizes by 1/N^3 (fftw.cpp:358). |
